#!/opt/conda/envs/paste_env/bin/python
"""
PASTE worker for SpatialOmicsLab MCP integration.

- Runs inside /opt/conda/envs/paste_env
- Called by paste_mcp_server.py via subprocess.
- All logs/progress go to stderr (prefixed with [paste-worker]).
- The ONLY thing printed to stdout is a single line of JSON with the final result.

Supported modes:
  - pairwise:   sequential pairwise PASTE alignment + stack_slices_pairwise
  - center:     center_align + stack_slices_center

Inputs:
  - --mode {pairwise,center}
  - --slice-h5ad /path/to/slice.h5ad   (can be passed multiple times)
  - --output-dir /workspace/work/paste_<tag>
  - optional params: --alpha, --use-gpu, --random-seed, --use-spatial-init, --use-raw-counts

What every run does to its slices before PASTE sees them:

  - spots with obs['in_tissue'] == 0 (background glass, which CELLxGENE Visium exports carry) are
    left out as each slice is read (worker_utils.keep_in_tissue) and counted in
    params.in_tissue_filter / params.in_tissue_dropped_per_slice. PASTE gives every spot the same
    transport mass, so glass spots would otherwise take most of the plan and the Procrustes fit.
    The aligned slices hold the in-tissue spots only;
  - the matrix PASTE compares is chosen by worker_utils.choose_counts_matrix: PASTE's expression
    term is a KL divergence between spot profiles and center_align's NMF is a KL factorisation, so
    X is read as counts. Negative or non-finite X is refused; --use-raw-counts reads adata.raw;
  - consecutive slices (pairwise) or all slices (center) must share at least one gene. With none,
    PASTE's expression term is identically zero and the "alignment" rests on geometry alone, which
    upstream accepts without a word; this worker refuses it and reports the shared-gene counts.

``random_seed`` reaches only center mode (center_align's NMF). pst.pairwise_align is deterministic
(a uniform initial plan and exact EMD inside its fused Gromov-Wasserstein), so in pairwise mode the
seed is listed in params.ignored rather than echoed as a setting.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback

import numpy as np
from worker_utils import (
    WorkerOutput,
    choose_counts_matrix,
    id_mismatch_msg,
    keep_in_tissue,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    resolve_compute,
    spatial_coords,
)

try:
    import paste as pst  # type: ignore
    import scanpy as sc  # type: ignore
except Exception as e:  # pragma: no cover
    print("[paste-worker] ERROR: Failed to import scanpy or paste inside paste_env.", file=sys.stderr)
    print(str(e), file=sys.stderr)
    WorkerOutput.emit_error("paste", "Failed to import scanpy/paste in paste_env", task="import")
    sys.exit(1)


def log(msg: str) -> None:
    print(f"[paste-worker] {msg}", file=sys.stderr)


#: Why ``random_seed`` has no effect on a pairwise run (``params.ignored``).
PAIRWISE_SEED_IGNORED = (
    "pst.pairwise_align is deterministic (its fused Gromov-Wasserstein starts from the uniform plan and "
    "solves each step with exact EMD) and stack_slices_pairwise has no random step; the seed reaches only "
    "paste_center_align's NMF"
)


# --- what goes into PASTE ---------------------------------------------------------------------


def _atomic_save_npy(path, array) -> None:
    """``<path>.partial`` then ``os.replace``. Through a handle: np.save appends '.npy' to a name."""
    tmp = f"{path}.partial"
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    os.replace(tmp, path)


def _atomic_write_h5ad(adata, path) -> None:
    tmp = f"{path}.partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def _load_slices(slice_paths, use_raw_counts, log_fn):
    """Read every slice, leave its background out and choose the matrix PASTE compares.

    Returns ``(slices, record)``; ``record`` carries, per slice, the spots supplied, the
    ``obs['in_tissue'] == 0`` spots left out and :func:`worker_utils.choose_counts_matrix`'s account
    of the matrix, for :func:`_record_inputs`.
    """
    slices = []
    record = {"n_supplied": [], "n_dropped": [], "infos": []}
    for i, path in enumerate(slice_paths):
        log_fn(f"Reading slice {i}: {path}")
        adata = sc.read_h5ad(path)
        if "spatial" not in adata.obsm:
            raise ValueError(f"Slice {path} is missing obsm['spatial']; required for PASTE.")
        adata, n_supplied, n_off = keep_in_tissue(adata, f"spots of slice {i}")
        if n_off:
            log_fn(f"  slice {i}: left out {n_off} of {n_supplied} spots with obs['in_tissue'] == 0 (background)")
        try:
            adata, info = choose_counts_matrix(adata, use_raw_counts)
        except ValueError as exc:
            raise ValueError(f"slice {i} ({path}): {exc}") from exc
        record["n_supplied"].append(int(n_supplied))
        record["n_dropped"].append(int(n_off))
        record["infos"].append(info)
        slices.append(adata)
    return slices, record


def _no_shared_genes_msg(slices, i, j):
    """The refusal for two slices that share no gene name, with the CELLxGENE case named."""
    a, b = slices[i], slices[j]
    msg = id_mismatch_msg("genes", f"slice {i}", list(a.var_names), f"slice {j}", list(b.var_names))
    for k, s in ((i, a), (j, b)):
        if "feature_name" in getattr(s.var, "columns", ()):
            msg += (
                f" Slice {k} carries gene symbols in var['feature_name'] (a CELLxGENE export keeps Ensembl IDs "
                "in var_names); set its var_names from that column, or convert the other slice to Ensembl IDs."
            )
            break
    return (
        msg + " PASTE compares spots by their expression over the genes two slices share; with none "
        "shared that term is zero everywhere and the alignment would rest on spot geometry alone, so nothing "
        "was aligned."
    )


def _shared_gene_counts(slices):
    """Genes each consecutive pair shares, as pst.pairwise_align intersects them; none is refused."""
    counts = []
    for i in range(len(slices) - 1):
        in_next = {str(g) for g in slices[i + 1].var_names}
        n = sum(1 for g in slices[i].var_names if str(g) in in_next)
        if n == 0:
            raise ValueError(_no_shared_genes_msg(slices, i, i + 1))
        counts.append(int(n))
    return counts


def _common_gene_count(slices):
    """Genes every slice carries, as filter_for_common_genes / center_align intersect them."""
    common = [str(g) for g in slices[0].var_names]
    for s in slices[1:]:
        keep = {str(g) for g in s.var_names}
        common = [g for g in common if g in keep]
    if common:
        return len(common)
    for j in range(1, len(slices)):
        keep = {str(g) for g in slices[j].var_names}
        if not any(str(g) in keep for g in slices[0].var_names):
            raise ValueError(_no_shared_genes_msg(slices, 0, j))
    raise ValueError(
        f"No gene is carried by all {len(slices)} slices: each shares genes with slice 0, but their "
        "intersection over every slice is empty. PASTE's center alignment compares every slice over one "
        "common gene set; with none its expression term is zero everywhere, so nothing was aligned."
    )


def _record_inputs(out, record):
    """Put what :func:`_load_slices` decided into the payload; return the analysis sentences."""
    n_supplied, n_dropped, infos = record["n_supplied"], record["n_dropped"], record["infos"]
    record_in_tissue(out, sum(n_supplied), sum(n_dropped))
    sentences = []
    if any(n_dropped):
        out.add_params({"in_tissue_dropped_per_slice": [int(v) for v in n_dropped]})
        sentences.append(
            f"{sum(n_dropped)} background spots (obs['in_tissue'] == 0) were left out before alignment "
            f"(per slice: {', '.join(str(v) for v in n_dropped)}); the aligned slices hold the "
            f"{sum(n_supplied) - sum(n_dropped)} in-tissue spots only."
        )
    kinds = [str(info["x_matrix_kind"]) for info in infos]
    record_expression_source(
        out,
        {
            "expression_source": infos[0]["expression_source"],
            "x_matrix_kind": kinds[0] if len(set(kinds)) == 1 else "mixed",
            "warning": None,
        },
    )
    out.add_params({"x_matrix_kind_per_slice": kinds})
    warned = [(i, info["warning"]) for i, info in enumerate(infos) if info.get("warning")]
    if warned and len({w for _, w in warned}) == 1:
        which = ", ".join(str(i) for i, _ in warned)
        out.add_warning(f"slice(s) {which}: {warned[0][1]}")
    else:
        for i, warning in warned:
            out.add_warning(f"slice {i}: {warning}")
    if infos[0]["expression_source"] == "raw.X":
        sentences.append("Expression was read from adata.raw of every slice (use_raw_counts=True).")
    return sentences


def _resolve_paste_backend(device: str, use_gpu: bool) -> tuple[bool, object | None]:
    """Return the ``(use_gpu, backend)`` pair PASTE actually needs, for the device requested.

    ``pst.pairwise_align`` and ``pst.center_align`` default to ``backend=ot.backend.NumpyBackend()``
    and then, in ``paste/PASTE.py``, force ``use_gpu = False`` for any backend that is not a
    ``TorchBackend`` -- printing "please set backend = ot.backend.TorchBackend()" and carrying on
    over the CPU. So ``--use-gpu`` on its own has never moved a PASTE run onto a GPU on *any*
    hardware; it is not a wrong-hardware bug but a flag with no effect anywhere. Supplying the
    backend is what makes the request real.

    ``None`` is returned for the backend on the CPU path rather than an explicit ``NumpyBackend()``,
    so that path keeps calling PASTE with exactly the arguments it always has.
    """
    resolved = resolve_compute(device or use_gpu).device
    if not resolved.startswith("cuda"):
        return False, None

    # Imported here, not at module scope: a CPU-only run must not be able to fail on a missing
    # torch, and POT's torch backend imports torch itself.
    import ot
    import torch

    if ":" in resolved:
        # PASTE's backend has no way to name a card, so pick it at the process level instead --
        # otherwise every concurrent run on a multi-GPU host lands on card zero.
        torch.cuda.set_device(int(resolved.split(":", 1)[1]))
    return True, ot.backend.TorchBackend()


# --- the coordinate contract ------------------------------------------------------------------
#
# pst.stack_slices_pairwise and stack_slices_center return COPIES whose obsm['spatial'] has been
# replaced by the aligned values (paste/visualization.py:65). Writing those out, which is what this
# worker did until 2026-09-21, means the coordinates PASTE was given are not recoverable from what
# it wrote: there is no before frame, so no before/after comparison is possible and the alignment
# cannot be validated at all.
#
# The fix restores the input under obsm['spatial'] and puts the aligned values in a key of their
# own, per spatialomicsgym/spatial3d/contract.py. It does NOT invent a z. PASTE aligns in plane and
# knows nothing about section spacing, so a three-column frame is written only when the caller
# declared one; otherwise the aligned coordinates are two columns under 'spatial_aligned' and the
# payload says what a three-column frame would need.
def _preserve_and_label(original_slices, new_slices, z_spacing, log_fn):
    """Put the input coordinates back and record the aligned ones beside them."""
    import numpy as np

    notes = []
    for i, (before, after) in enumerate(zip(original_slices, new_slices)):
        orig = np.asarray(before.obsm["spatial"], dtype=float)
        aligned = np.asarray(after.obsm["spatial"], dtype=float)[:, :2]
        if len(orig) != len(aligned):
            notes.append(
                f"slice {i}: {len(aligned)} aligned rows against {len(orig)} input rows, so the "
                f"original coordinates were not restored for it"
            )
            continue
        after.obsm["spatial"] = orig
        if z_spacing and z_spacing > 0:
            z = np.full(len(aligned), float(i) * float(z_spacing))
            after.obsm["spatial_3d_aligned"] = np.column_stack([aligned, z])
            after.obs["slice_z"] = z
        else:
            after.obsm["spatial_aligned"] = aligned
    if not (z_spacing and z_spacing > 0):
        notes.append(
            "no z spacing was declared, so the aligned coordinates are two columns under "
            "obsm['spatial_aligned']. Pass --z-spacing to get a three-column "
            "obsm['spatial_3d_aligned']; PASTE aligns in plane and cannot supply a z itself."
        )
    for note in notes:
        log_fn(note)
    return notes


def _spatial_init_mappings(slices, log_fn):
    """``pis_init`` for center_align, built from each slice's COORDINATES.

    Returns ``(pis_init, notes)``; the notes are payload warnings.

    ``pst.match_spots_using_spatial_heuristic(X, Y)`` takes spot coordinates (paste/helper.py:
    "X (array-like): Coordinates for spots X") and centres them by ``norm_and_center_coordinates``.
    This worker handed it ``slices[i].X`` -- the expression matrices -- so on the sparse X every
    library sample carries it died in ``len(X)`` ("sparse matrix length is ambiguous"), and on a dense
    X it matched spots in gene space and fed that to center_align as the "spatial" initial mapping.

    The mapping runs from the first slice (the initial centre, ``A = slices[0].copy()``) to each slice,
    which is the orientation center_align reads (``np.dot(pis[i], slices[i].X)``, rows = centre).
    PASTE's stacking asserts two coordinate columns, so the coordinates are read at that width and a
    wider key is refused rather than truncated.
    """
    import warnings

    coords = [spatial_coords(s, "spatial", want=2, tool="PASTE use_spatial_init")[0] for s in slices]
    pis_init = []
    notes = []
    for i, xy in enumerate(coords):
        log_fn(f"Spatial heuristic mapping: slice 0 ({len(coords[0])} spots) -> slice {i} ({len(xy)} spots)")
        # The heuristic solves an exact transport with ot.emd at POT's default iteration cap; on a
        # slide of a few thousand spots it can stop there and return a feasible, non-optimal plan,
        # saying so only in a Python warning that would otherwise reach stderr alone.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pi = np.asarray(pst.match_spots_using_spatial_heuristic(coords[0], xy), dtype=float)
        for w in caught:
            if "numItermax" in str(w.message):
                notes.append(
                    f"use_spatial_init: the spatial heuristic's transport for slice {i} stopped at its iteration "
                    f"cap ({w.message}); the initial mapping is feasible but not optimal."
                )
                log_fn(notes[-1])
        if pi.shape != (len(coords[0]), len(xy)):
            raise ValueError(
                f"The spatial heuristic returned a {pi.shape} mapping for slice {i}; center_align needs "
                f"({len(coords[0])}, {len(xy)}), centre spots by slice spots."
            )
        if not np.isfinite(pi).all():
            raise ValueError(
                f"The spatial heuristic mapping for slice {i} is not finite. PASTE normalises each slice's "
                "coordinates by its smallest spot-to-spot distance, so two spots at the same position in "
                "obsm['spatial'] divide by zero."
            )
        pis_init.append(pi)
    return pis_init, notes


def run_pairwise_alignment(
    slice_paths: list[str],
    output_dir: str,
    alpha: float,
    use_gpu: bool,
    random_seed: int,
    device: str = "",
    z_spacing: float = 0.0,
    use_raw_counts: bool = False,
) -> dict:
    """
    Sequential pairwise alignment:
      slices: S1, S2, ..., Sk
      compute pi12, pi23, ..., pi_{k-1,k} via pst.pairwise_align
      then stack_slices_pairwise to get new_slices with aligned coordinates.

    Background spots are left out and the counts matrix chosen as each slice is read
    (:func:`_load_slices`); a consecutive pair that shares no gene is refused. ``random_seed`` is
    accepted and ignored: nothing in this mode is random (``params.ignored``).
    """
    os.makedirs(output_dir, exist_ok=True)
    log(f"Running pairwise alignment on {len(slice_paths)} slices.")
    log(f"Output dir: {output_dir}")
    use_gpu, gpu_backend = _resolve_paste_backend(device, use_gpu)
    log(f"alpha={alpha}, use_gpu={use_gpu}, random_seed={random_seed} (ignored: pairwise_align is deterministic)")

    slices, record = _load_slices(slice_paths, use_raw_counts, log)
    n_common_per_pair = _shared_gene_counts(slices)
    log(f"Genes shared by each consecutive pair: {n_common_per_pair}")

    # Compute pairwise alignments
    pis = []
    for i in range(len(slices) - 1):
        log(f"Computing pairwise_align between slice {i} and {i + 1}...")
        pi = pst.pairwise_align(
            slices[i],
            slices[i + 1],
            alpha=alpha,
            use_gpu=use_gpu,
            **({"backend": gpu_backend} if gpu_backend is not None else {}),
        )
        pis.append(pi)
        pi_path = os.path.join(output_dir, f"paste_pairwise_pi_{i}_{i + 1}.npy")
        log(f"Saving pi[{i},{i + 1}] -> {pi_path}")
        _atomic_save_npy(pi_path, pi)

    # Stack slices into common coordinate system
    log("Stacking slices with pst.stack_slices_pairwise...")
    new_slices = pst.stack_slices_pairwise(slices, pis)
    contract_notes = _preserve_and_label(slices, new_slices, z_spacing, log)

    aligned_paths = []
    for i, ad in enumerate(new_slices):
        out_h5ad = os.path.join(output_dir, f"paste_pairwise_aligned_slice_{i}.h5ad")
        log(f"Writing aligned slice {i} -> {out_h5ad}")
        _atomic_write_h5ad(ad, out_h5ad)
        aligned_paths.append(out_h5ad)

    pi_paths = [os.path.join(output_dir, f"paste_pairwise_pi_{i}_{i + 1}.npy") for i in range(len(slices) - 1)]
    out = WorkerOutput("paste", task="pairwise_alignment")
    out.add_warnings(contract_notes)
    out.set_data(
        n_slices=len(slice_paths),
        n_spots_per_slice=[int(s.n_obs) for s in slices],
        n_common_genes_per_pair=n_common_per_pair,
    )
    out.add_output_files(
        {
            **{f"aligned_slice_{i}": p for i, p in enumerate(aligned_paths)},
            **{f"pi_{i}_{i + 1}": p for i, p in enumerate(pi_paths)},
        }
    )
    out.add_params(
        {
            "mode": "pairwise",
            "input_slices": slice_paths,
            "output_dir": output_dir,
            "alpha": alpha,
            "use_gpu": use_gpu,
            "z_spacing": z_spacing,
            "use_raw_counts": bool(use_raw_counts),
        }
    )
    input_sentences = _record_inputs(out, record)
    record_ignored(out, "random_seed", PAIRWISE_SEED_IGNORED)
    out.set_summary(
        n_slices_aligned=len(aligned_paths),
        n_coupling_matrices=len(pis),
        n_common_genes_per_pair=n_common_per_pair,
    )
    out.set_analysis(
        f"PASTE pairwise alignment completed for {len(slice_paths)} slices "
        f"(alpha={alpha}). Produced {len(aligned_paths)} aligned slices. Consecutive slices were compared "
        f"over {', '.join(str(n) for n in n_common_per_pair)} shared genes."
        + "".join(" " + s for s in input_sentences)
        + " random_seed has no effect in pairwise mode: the alignment is deterministic."
    )
    return out.to_dict()


def run_center_alignment(
    slice_paths: list[str],
    output_dir: str,
    alpha: float,
    use_gpu: bool,
    random_seed: int,
    use_spatial_init: bool,
    device: str = "",
    z_spacing: float = 0.0,
    use_raw_counts: bool = False,
) -> dict:
    """
    Center alignment:
      - load all slices as AnnData, ensure obsm['spatial'] exists, leave background spots out and
        choose the counts matrix (:func:`_load_slices`)
      - refuse slices that share no gene
      - optionally build spatial-based initial mappings (match_spots_using_spatial_heuristic)
      - run filter_for_common_genes + center_align
      - stack_slices_center to get aligned slices
    """
    os.makedirs(output_dir, exist_ok=True)
    log(f"Running center alignment on {len(slice_paths)} slices.")
    log(f"Output dir: {output_dir}")
    use_gpu, gpu_backend = _resolve_paste_backend(device, use_gpu)
    log(f"alpha={alpha}, use_gpu={use_gpu}, random_seed={random_seed}, use_spatial_init={use_spatial_init}")

    slices, record = _load_slices(slice_paths, use_raw_counts, log)
    n_common = _common_gene_count(slices)
    log(f"Genes shared by all {len(slices)} slices: {n_common}")

    # initial (center) slice: copy of first slice
    initial_slice = slices[0].copy()
    lmbda = [1.0 / len(slices)] * len(slices)

    # Filter for common genes
    log("Filtering slices for common genes using pst.filter_for_common_genes...")
    pst.filter_for_common_genes(slices)

    # Optional spatial heuristic initialization
    pis_init = None
    init_notes = []
    if use_spatial_init:
        log("Building spatial heuristic initial mappings (pst.match_spots_using_spatial_heuristic)...")
        pis_init, init_notes = _spatial_init_mappings(slices, log)

    np.random.seed(random_seed)

    log("Calling pst.center_align...")
    center_slice, pis = pst.center_align(
        initial_slice,
        slices,
        lmbda,
        alpha=alpha,
        random_seed=random_seed,
        pis_init=pis_init,
        use_gpu=use_gpu,
        **({"backend": gpu_backend} if gpu_backend is not None else {}),
    )

    center_h5ad = os.path.join(output_dir, "paste_center_slice.h5ad")
    log(f"Writing center slice -> {center_h5ad}")
    _atomic_write_h5ad(center_slice, center_h5ad)

    # Stack slices around center
    log("Stacking slices with pst.stack_slices_center...")
    _, new_slices = pst.stack_slices_center(center_slice, slices, pis)
    contract_notes = _preserve_and_label(slices, new_slices, z_spacing, log)

    aligned_paths = []
    for i, ad in enumerate(new_slices):
        out_h5ad = os.path.join(output_dir, f"paste_center_aligned_slice_{i}.h5ad")
        log(f"Writing center-aligned slice {i} -> {out_h5ad}")
        _atomic_write_h5ad(ad, out_h5ad)
        aligned_paths.append(out_h5ad)

    # Save the coupling matrices pis as .npy
    pi_paths = []
    for i, pi in enumerate(pis):
        pi_path = os.path.join(output_dir, f"paste_center_pi_slice_{i}.npy")
        log(f"Saving center pi for slice {i} -> {pi_path}")
        _atomic_save_npy(pi_path, pi)
        pi_paths.append(pi_path)

    out = WorkerOutput("paste", task="center_alignment")
    out.add_warnings(contract_notes)
    out.add_warnings(init_notes)
    out.set_data(
        n_slices=len(slice_paths),
        n_spots_per_slice=[int(s.n_obs) for s in slices],
        n_common_genes=int(n_common),
    )
    out.add_output_files(
        {
            "center_slice": center_h5ad,
            **{f"aligned_slice_{i}": p for i, p in enumerate(aligned_paths)},
            **{f"pi_slice_{i}": p for i, p in enumerate(pi_paths)},
        }
    )
    out.add_params(
        {
            "mode": "center",
            "input_slices": slice_paths,
            "output_dir": output_dir,
            "alpha": alpha,
            "use_gpu": use_gpu,
            "random_seed": random_seed,
            "z_spacing": z_spacing,
            "use_spatial_init": use_spatial_init,
            # What the initial mappings were built from: coordinates, never expression.
            "spatial_init_source": "obsm['spatial']" if use_spatial_init else None,
            "use_raw_counts": bool(use_raw_counts),
        }
    )
    input_sentences = _record_inputs(out, record)
    out.set_summary(
        n_slices_aligned=len(aligned_paths),
        n_coupling_matrices=len(pis),
        n_common_genes=int(n_common),
    )
    out.set_analysis(
        f"PASTE center alignment completed for {len(slice_paths)} slices "
        f"(alpha={alpha}). Produced center slice + {len(aligned_paths)} aligned slices, over the "
        f"{n_common} genes all slices share."
        + "".join(" " + s for s in input_sentences)
        + (
            " center_align started from spatial-heuristic mappings built on obsm['spatial'] "
            "(pst.match_spots_using_spatial_heuristic, slice 0 to each slice)."
            if use_spatial_init
            else ""
        )
    )
    return out.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="PASTE worker for SpatialOmicsLab MCP.")
    parser.add_argument(
        "--mode",
        choices=["pairwise", "center"],
        required=True,
        help="PASTE mode: 'pairwise' for sequential pairwise alignment, 'center' for center_align.",
    )
    parser.add_argument(
        "--slice-h5ad",
        dest="slice_h5ads",
        action="append",
        required=True,
        help="Path to a slice AnnData .h5ad file. Use multiple times for multiple slices, in anatomical order.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where outputs (aligned h5ad + coupling matrices) will be written.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="PASTE alpha parameter (trade-off between expression and spatial distance).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="If set, PASTE will attempt to use GPU backend where supported.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Compute device: 'cpu', 'gpu'/'cuda', or 'cuda:N'. Overrides --use-gpu when set.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        # 0, not 5. Until 2026-09-21 this argparse default and the portal's two functions carried
        # 0, 5 and 5 -- three values for one concept, so the same call seeded differently
        # depending on which door it came through.
        default=0,
        help="Random seed for PASTE / numpy.",
    )
    parser.add_argument(
        "--z-spacing",
        type=float,
        default=0.0,
        # No default spacing, deliberately: 0.0 means "not declared", not "zero apart". PASTE
        # aligns in plane, so a z can only come from the caller, and a stack built on an assumed
        # spacing produces a 3D neighbour graph that is either disconnected or a mush.
        help="Physical distance between consecutive sections, in the coordinates' own units. "
        "When given, aligned coordinates are written as a three-column obsm['spatial_3d_aligned'].",
    )
    parser.add_argument(
        "--use-spatial-init",
        action="store_true",
        help="(center mode only) Use spatial heuristic to initialize mappings.",
    )
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        help="Align on adata.raw.X (raw counts) of every slice instead of X; refused when a slice has no "
        "adata.raw or it does not hold counts.",
    )

    args = parser.parse_args()

    try:
        # PASTE reports progress with print() -- "Filtered all slices for common genes...",
        # "Iteration: 0", its objective values -- which put a dozen lines on stdout ahead of the
        # result, where this worker promises one JSON line and nothing else. The portal scans for the
        # last JSON line and coped; anything that reads stdout as the JSON it is documented to be did
        # not. Everything the run prints goes to stderr with the rest of the log.
        with contextlib.redirect_stdout(sys.stderr):
            if len(args.slice_h5ads) < 2:
                raise ValueError("At least two --slice-h5ad paths are required for PASTE alignment.")

            if args.mode == "pairwise":
                result = run_pairwise_alignment(
                    slice_paths=args.slice_h5ads,
                    output_dir=args.output_dir,
                    alpha=args.alpha,
                    use_gpu=bool(args.use_gpu),
                    random_seed=int(args.random_seed),
                    z_spacing=float(args.z_spacing),
                    device=args.device,
                    use_raw_counts=bool(args.use_raw_counts),
                )
            else:
                result = run_center_alignment(
                    slice_paths=args.slice_h5ads,
                    output_dir=args.output_dir,
                    alpha=args.alpha,
                    use_gpu=bool(args.use_gpu),
                    random_seed=int(args.random_seed),
                    z_spacing=float(args.z_spacing),
                    use_spatial_init=bool(args.use_spatial_init),
                    device=args.device,
                    use_raw_counts=bool(args.use_raw_counts),
                )

        print(json.dumps(result), file=sys.stdout)

    except Exception as e:  # pragma: no cover
        log("ERROR: Exception during PASTE run.")
        log("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        task = args.mode + "_alignment" if hasattr(args, "mode") else "alignment"
        WorkerOutput.emit_error("paste", str(e), task=task)
        sys.exit(1)


if __name__ == "__main__":
    main()
