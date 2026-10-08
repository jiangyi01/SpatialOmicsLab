"""Worker for PASTE2: pairwise alignment of slices that only PARTIALLY overlap.

PASTE assumes the two sections cover the same tissue. PASTE2 does not: it takes an overlap
fraction ``s`` and solves a *partial* optimal-transport problem, which is the branch of the
Phase-2 decision tree that had no tool behind it until now.

Three things this worker does that the upstream call does not.

**It keeps the coordinates it was given.** ``partial_stack_slices_pairwise`` returns copies whose
``obsm['spatial']`` has been replaced by the aligned values, exactly as PASTE's stacking does, so
writing them out loses the before frame and with it any possibility of validating the alignment.
The input is snapshotted and restored, and the aligned values go to a key of their own.

**It does not invent a z.** PASTE2 aligns in plane and knows nothing about section spacing, so a
three-column frame appears only when the caller declares one. ``z_spacing = 0.0`` means *not
declared*, never *zero apart*.

**It reports the overlap fraction it actually used.** ``s`` is the parameter that makes this tool
different from PASTE, and when it is estimated rather than given, the estimate is part of the
answer -- a pair aligned at s=0.9 and a pair aligned at s=0.3 are different claims about the
tissue.

**It leaves the background out.** Spots with ``obs['in_tissue'] == 0`` (the glass CELLxGENE
Visium exports carry) are dropped as each slice is read, counted in ``params.in_tissue_filter`` /
``params.in_tissue_dropped_per_slice``, and the aligned slices hold the in-tissue spots only.

**It says how an estimated overlap was estimated.** PASTE2's estimator hard-codes a GLM-PCA cost
(latent_dim=50) and 20 partial optimal-transport solves per pair, whatever ``dissimilarity`` and
``norm`` the alignment itself uses; ``params.overlap_estimate`` records that.

Everything PASTE2 prints goes to stderr: it writes progress to stdout unconditionally, and one
stray line there is indistinguishable from the payload.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import WorkerOutput, keep_in_tissue, record_in_tissue

TOOL = "paste2"

#: What ``paste2.model_selection.select_overlap_fraction`` actually does (model_selection.py:136-175
#: in paste2 1.0.1), recorded because the caller's ``dissimilarity`` never reaches it: choosing 'kl'
#: for speed did not avoid the GLM-PCA path, and the overlap came from a different cost matrix
#: than the alignment it was applied to (hunt 2026-09-30, u21-3d-19).
OVERLAP_ESTIMATOR = (
    "paste2.model_selection.select_overlap_fraction: a GLM-PCA cost (latent_dim=50) and 20 partial "
    "OT solves per pair (s = 0.99 down to 0.05, norm=True), whatever dissimilarity and norm the "
    "alignment uses; pass s to skip it"
)


def _log(msg: str) -> None:
    print(f"[paste2-worker] {msg}", file=sys.stderr, flush=True)


def _preserve_and_label(originals, new_slices, z_spacing):
    """Put the input coordinates back and record the aligned ones beside them."""
    notes = []
    for i, (orig, after) in enumerate(zip(originals, new_slices)):
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
            "obsm['spatial_aligned']. Pass --z-spacing for a three-column obsm['spatial_3d_aligned']; "
            "PASTE2 aligns in plane and cannot supply a z itself."
        )
    for note in notes:
        _log(note)
    return notes


def _atomic_write_h5ad(adata, path: str) -> None:
    tmp = f"{path}.partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def _load_tissue(slice_paths, task):
    """Read every slice and leave its background out (hunt 2026-09-30, u21-3d-3).

    PASTE2 gave the glass the same transport mass as the tissue, the overlap estimate counted it
    as overlap, and a column of all-zero spots gave GLM-PCA a zero size factor. Returns
    ``(slices, n_supplied, n_dropped)``.
    """
    import anndata as ad

    slices, n_supplied, n_dropped = [], [], []
    for i, path in enumerate(slice_paths):
        adata = ad.read_h5ad(path)
        if "spatial" not in adata.obsm:
            WorkerOutput.emit_error(TOOL, f"{path} has no obsm['spatial']", task=task)
            sys.exit(1)
        adata, n_in, n_off = keep_in_tissue(adata, f"spots of slice {i}")
        if n_off:
            _log(f"slice {i}: left out {n_off} of {n_in} spots with obs['in_tissue'] == 0 (background)")
        slices.append(adata)
        n_supplied.append(n_in)
        n_dropped.append(n_off)
    return slices, n_supplied, n_dropped


def _record_tissue(out, n_supplied, n_dropped) -> None:
    record_in_tissue(out, sum(n_supplied), sum(n_dropped))
    if any(n_dropped):
        out.add_params({"in_tissue_dropped_per_slice": [int(v) for v in n_dropped]})


def run_partial(
    slice_paths: list[str],
    output_dir: str,
    s: float,
    alpha: float,
    dissimilarity: str,
    armijo: bool,
    norm: bool,
    random_seed: int,
    z_spacing: float,
) -> None:
    from paste2 import PASTE2, model_selection, projection

    os.makedirs(output_dir, exist_ok=True)
    _log(f"{len(slice_paths)} slices | s={s} alpha={alpha} dissimilarity={dissimilarity} seed={random_seed}")
    np.random.seed(random_seed)

    slices, n_supplied, n_dropped = _load_tissue(slice_paths, "partial_alignment")
    originals = [np.asarray(sl.obsm["spatial"], dtype=float).copy() for sl in slices]

    pis, used_s = [], []
    for i in range(len(slices) - 1):
        with contextlib.redirect_stdout(sys.stderr):
            if s and s > 0:
                pair_s = float(s)
                how = "given"
            else:
                # 0.0 is the sentinel for "estimate it": zero is not a legal overlap fraction, so
                # the sentinel cannot be confused with a request for no overlap at all.
                pair_s = float(model_selection.select_overlap_fraction(slices[i], slices[i + 1], alpha=alpha))
                how = "estimated"
            _log(f"pair {i}->{i + 1}: s={pair_s:.4f} ({how})")
            pi = PASTE2.partial_pairwise_align(
                slices[i],
                slices[i + 1],
                pair_s,
                alpha=alpha,
                armijo=armijo,
                dissimilarity=dissimilarity,
                norm=norm,
                verbose=False,
            )
        pis.append(pi)
        used_s.append({"pair": f"{i}->{i + 1}", "s": pair_s, "source": how})
        np.save(os.path.join(output_dir, f"paste2_pi_{i}_{i + 1}.npy"), pi)

    with contextlib.redirect_stdout(sys.stderr):
        new_slices = projection.partial_stack_slices_pairwise(slices, pis)
    notes = _preserve_and_label(originals, new_slices, z_spacing)

    aligned_paths = []
    for i, adata in enumerate(new_slices):
        out = os.path.join(output_dir, f"paste2_partial_aligned_slice_{i}.h5ad")
        _atomic_write_h5ad(adata, out)
        aligned_paths.append(out)

    import pandas as pd

    overlap_csv = os.path.join(output_dir, "overlap_fractions.csv")
    pd.DataFrame(used_s).to_csv(overlap_csv, index=False)

    out = WorkerOutput(TOOL, task="partial_alignment")
    out.add_warnings(notes)
    _record_tissue(out, n_supplied, n_dropped)
    if any(r["source"] == "estimated" for r in used_s):
        out.add_params({"overlap_estimate": OVERLAP_ESTIMATOR})
        if dissimilarity != "glmpca" or not norm:
            out.add_warning(
                f"the overlap fractions marked 'estimated' were measured on PASTE2's GLM-PCA cost with "
                f"norm=True -- its estimator hard-codes both and solves 20 partial OT problems per pair -- "
                f"not on the dissimilarity={dissimilarity!r}, norm={norm} this alignment used. Pass s to "
                f"align at a fraction you choose and skip the estimate."
            )
    out.set_data(n_slices=len(slice_paths))
    out.add_output_files(
        {
            **{f"aligned_slice_{i}": p for i, p in enumerate(aligned_paths)},
            **{f"pi_{i}_{i + 1}": os.path.join(output_dir, f"paste2_pi_{i}_{i + 1}.npy") for i in range(len(pis))},
            "overlap_fractions": overlap_csv,
        }
    )
    out.add_params(
        {
            "input_slices": list(slice_paths),
            "output_dir": output_dir,
            "s": s,
            "overlap_fractions_used": used_s,
            "alpha": alpha,
            "dissimilarity": dissimilarity,
            "armijo": armijo,
            "norm": norm,
            "random_seed": random_seed,
            "z_spacing": z_spacing,
        }
    )
    out.set_summary(n_slices_aligned=len(aligned_paths), n_coupling_matrices=len(pis))
    out.set_analysis(
        f"PASTE2 partial alignment over {len(slice_paths)} slices. Overlap fractions used: "
        + ", ".join(f"{r['pair']}={r['s']:.3f} ({r['source']})" for r in used_s)
    )
    out.emit()


def run_estimate(slice_paths: list[str], output_dir: str, alpha: float) -> None:
    """Measure the overlap fraction and write nothing else. A measurement, not an alignment."""
    import pandas as pd
    from paste2 import model_selection

    os.makedirs(output_dir, exist_ok=True)
    slices, n_supplied, n_dropped = _load_tissue(slice_paths, "overlap_estimate")
    rows = []
    for i in range(len(slices) - 1):
        with contextlib.redirect_stdout(sys.stderr):
            s = float(model_selection.select_overlap_fraction(slices[i], slices[i + 1], alpha=alpha))
        _log(f"pair {i}->{i + 1}: estimated s={s:.4f}")
        rows.append({"pair": f"{i}->{i + 1}", "s": s})
    csv = os.path.join(output_dir, "overlap_fractions.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)

    out = WorkerOutput(TOOL, task="overlap_estimate")
    _record_tissue(out, n_supplied, n_dropped)
    out.set_data(n_slices=len(slice_paths))
    out.add_output_files({"overlap_fractions": csv})
    out.add_params(
        {
            "input_slices": list(slice_paths),
            "output_dir": output_dir,
            "alpha": alpha,
            "overlap_estimate": OVERLAP_ESTIMATOR,
        }
    )
    out.set_summary(n_pairs=len(rows), min_overlap=min((r["s"] for r in rows), default=0.0))
    out.set_analysis(
        "Estimated overlap fraction per adjacent pair. A low value is the signal that PASTE's "
        "full-overlap assumption does not hold and PASTE2 is the right tool; a value near 1 means "
        "PASTE would do."
    )
    out.emit()


def main() -> None:
    parser = argparse.ArgumentParser(description="PASTE2 partial pairwise alignment.")
    parser.add_argument("--mode", choices=("partial", "estimate"), default="partial")
    parser.add_argument("--slice-h5ad", action="append", default=[], dest="slices")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--s",
        type=float,
        default=0.0,
        help="Overlap fraction in (0, 1]. 0.0 is the sentinel for 'estimate it per pair'.",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--dissimilarity", default="glmpca", choices=("glmpca", "kl", "euclidean"))
    parser.add_argument("--armijo", action="store_true")
    parser.add_argument("--no-norm", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--z-spacing", type=float, default=0.0)
    args = parser.parse_args()

    if len(args.slices) < 2:
        WorkerOutput.emit_error(TOOL, "paste2 needs at least two --slice-h5ad", task=args.mode)
        sys.exit(1)
    try:
        if args.mode == "estimate":
            run_estimate(args.slices, args.output_dir, args.alpha)
        else:
            run_partial(
                args.slices,
                args.output_dir,
                args.s,
                args.alpha,
                args.dissimilarity,
                args.armijo,
                not args.no_norm,
                args.random_seed,
                args.z_spacing,
            )
    except Exception as exc:
        _log("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        WorkerOutput.emit_error(TOOL, f"{type(exc).__name__}: {exc}", task=args.mode, exc=exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
