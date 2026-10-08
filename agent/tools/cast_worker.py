"""Worker for CAST: non-rigid alignment of sections at single-cell resolution.

CAST embeds each section with a graph neural network (CAST_MARK), then registers the embeddings
onto a reference with an affine fit followed by a free-form deformation (CAST_STACK). It is the
non-rigid branch of the Phase-2 decision tree that STalign otherwise holds alone.

Five deliberate departures from calling the library directly.

**``graph_strategy`` defaults to 'delaunay', not upstream's 'convex'.** Measured on this box: the
delaunay path is pure scipy and networkx and takes 0.01 s for 300 points; the convex path routes
through ``libpysal.cg.voronoi_frames`` and ``weights.Rook.from_dataframe``, takes 0.21 s, emits a
deprecation warning, and drags libpysal, geopandas, Rtree, pyproj and shapely onto the critical
path. Both work today. The divergence is recorded in ``official_defaults.yaml`` with
``adopted: false``.

**It keeps the coordinates it was given.** ``coords_final`` replaces the input frame; the input is
snapshotted and the aligned values go to a key of their own, so a before/after remains possible.

**It does not invent a z.** CAST registers in plane. ``z_spacing = 0.0`` means *not declared*.

**Every slice is embedded over one gene list, in one order.** CAST_MARK's encoder has shared
weights, so the highly variable genes are chosen once over all slices (``batch_key`` = slice) and
every slice is normalised to the same library size; background spots (``obs['in_tissue'] == 0``)
are left out as each slice is read.

**Everything is drawn to a file, never to a screen.** CAST_MARK and the registration call
matplotlib unconditionally, and its intermediate artefacts (``demo_embed_dict.pt`` and friends,
whose names are hardcoded upstream) go to a subdirectory so they cannot be mistaken for the
tool's answer.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import WorkerOutput, keep_in_tissue, record_in_tissue

TOOL = "cast"

#: Every slice is scaled to this library size before log1p, as upstream's own preprocessing does
#: (``CAST.utils.preprocess_fast``, ``target_sum=1e4``). normalize_total's default is each slice's
#: own median, which put the slices on different scales before a shared-weight encoder saw them.
TARGET_SUM = 1e4


def _log(msg: str) -> None:
    print(f"[cast-worker] {msg}", file=sys.stderr, flush=True)


def _atomic_write_h5ad(adata, path: str) -> None:
    tmp = f"{path}.partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def expression_features(slices, names, panel, n_hvg):
    """One gene list for every slice, and each slice's matrix over it in that order.

    CAST_MARK trains ONE encoder with shared weights on every slice (``in_dim`` is read from the
    first), so column j must be the same gene in every matrix. Highly variable genes were picked
    per slice and each slice subset separately: column j held a different gene in each, the
    embeddings compared unrelated genes, and the registration still reported ok
    (hunt 2026-09-30, u21-3d-1). They are now picked once, over all slices together, with the slice
    as scanpy's batch_key -- upstream's own pattern.

    Returns ``(exp_dict, genes)``; ``genes`` is the ordered list every matrix's columns follow.
    """
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp

    normed = []
    for adata in slices:
        sub = adata[:, panel].copy()
        with contextlib.redirect_stdout(sys.stderr):
            sc.pp.normalize_total(sub, target_sum=TARGET_SUM)
            sc.pp.log1p(sub)
        normed.append(sub)

    genes = list(panel)
    if n_hvg and n_hvg < len(panel):
        stacked = (
            sp.vstack([sp.csr_matrix(n.X) for n in normed]).tocsr()
            if any(sp.issparse(n.X) for n in normed)
            else np.vstack([np.asarray(n.X) for n in normed])
        )
        labels = np.concatenate([[name] * n.n_obs for name, n in zip(names, normed)])
        joint = ad.AnnData(
            X=stacked,
            obs=pd.DataFrame({"cast_slice": pd.Categorical(labels)}, index=[str(i) for i in range(len(labels))]),
            var=pd.DataFrame(index=list(panel)),
        )
        with contextlib.redirect_stdout(sys.stderr):
            sc.pp.highly_variable_genes(joint, n_top_genes=n_hvg, batch_key="cast_slice")
        flagged = set(joint.var_names[np.asarray(joint.var["highly_variable"], dtype=bool)])
        genes = [g for g in panel if g in flagged]
        del joint, stacked

    exp_dict = {}
    for name, sub in zip(names, normed):
        picked = sub[:, genes]
        if [str(g) for g in picked.var_names] != genes:
            raise RuntimeError(
                f"{name}: its columns are not the shared gene list, so the joint embedding would mix genes"
            )
        matrix = picked.X.toarray() if hasattr(picked.X, "toarray") else np.asarray(picked.X)
        exp_dict[name] = np.asarray(matrix, dtype=np.float32)
    return exp_dict, genes


def run_align(
    slice_paths: list[str],
    output_dir: str,
    reference_index: int,
    epochs: int,
    affine_iterations: int,
    ffd_iterations: int,
    graph_strategy: str,
    n_hvg: int,
    random_seed: int,
    z_spacing: float,
) -> None:
    import anndata as ad
    import pandas as pd
    import torch
    from CAST import CAST_MARK, CAST_STACK
    from CAST.CAST_Stack import reg_params

    os.makedirs(output_dir, exist_ok=True)
    scratch = os.path.join(output_dir, "cast_intermediate")
    os.makedirs(scratch, exist_ok=True)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    names = [f"slice_{i}" for i in range(len(slice_paths))]
    coords_raw: dict[str, np.ndarray] = {}
    originals: list[np.ndarray] = []
    slices: list[object] = []

    shared: set[str] | None = None
    loaded = []
    n_supplied, n_dropped = [], []
    for i, path in enumerate(slice_paths):
        adata = ad.read_h5ad(path)
        if "spatial" not in adata.obsm:
            WorkerOutput.emit_error(TOOL, f"{path} has no obsm['spatial']", task="alignment")
            sys.exit(1)
        # Background glass (obs['in_tissue'] == 0) is left out as each slice is read, as every other
        # spot-level aligner does; CAST embedded and registered it as tissue (hunt 2026-09-30, u21-3d-3).
        adata, n_in, n_off = keep_in_tissue(adata, f"spots of slice {i}")
        if n_off:
            _log(f"slice {i}: left out {n_off} of {n_in} spots with obs['in_tissue'] == 0 (background)")
        n_supplied.append(n_in)
        n_dropped.append(n_off)
        loaded.append(adata)
        genes = {str(g) for g in adata.var_names}
        shared = genes if shared is None else (shared & genes)
    if not shared:
        WorkerOutput.emit_error(TOOL, "the slices share no genes, so no joint embedding exists", task="alignment")
        sys.exit(1)
    panel = sorted(shared)
    _log(f"{len(slice_paths)} slices | {len(panel)} shared genes | strategy={graph_strategy} | seed={random_seed}")

    exp_dict, feature_genes = expression_features(loaded, names, panel, n_hvg)
    _log(f"{len(feature_genes)} genes in every slice's embedding input, in one order")
    for name, adata in zip(names, loaded):
        coords_raw[name] = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
        originals.append(np.asarray(adata.obsm["spatial"], dtype=float).copy())
        slices.append(adata)

    _log(f"CAST_MARK: {epochs} epochs on CPU" if not torch.cuda.is_available() else "CAST_MARK on GPU")
    with contextlib.redirect_stdout(sys.stderr):
        embed_dict = CAST_MARK(
            coords_raw,
            exp_dict,
            scratch,
            epoch_t=epochs,
            if_plot=False,
            graph_strategy=graph_strategy,
        )

    reference = names[reference_index]
    aligned: dict[str, np.ndarray] = {reference: coords_raw[reference]}
    per_pair = []
    for name in names:
        if name == reference:
            continue
        _log(f"CAST_STACK: {name} -> {reference}")
        # Replicated from CAST_STACK's own `params_dist is None` branch (main.py:71-86) rather
        # than invented: the B-spline lists it fills -- alpha_basis_bs, meshsize, mesh_weight,
        # attention_params_bs, dist_penalty2 -- are indexed unguarded at main.py:177, so a params
        # object missing any of them raises IndexError partway through the registration. Only the
        # two iteration counts are the caller's.
        params = reg_params(
            dataname=name,
            gpu=0 if torch.cuda.is_available() else -1,
            # affine
            iterations=affine_iterations,
            dist_penalty1=0,
            bleeding=500,
            d_list=[3, 2, 1, 1 / 2, 1 / 3],
            attention_params=[None, 3, 1, 0],
            # free-form deformation
            dist_penalty2=[0],
            alpha_basis_bs=[500],
            meshsize=[8],
            iterations_bs=[ffd_iterations],
            attention_params_bs=[[None, 3, 1, 0]],
            mesh_weight=[None],
        )
        with contextlib.redirect_stdout(sys.stderr):
            coords_final = CAST_STACK(
                {k: coords_raw[k] for k in (name, reference)},
                {k: embed_dict[k] for k in (name, reference)},
                scratch,
                [name, reference],
                params_dist=params,
                mid_visual=False,
            )
        moved = np.asarray(coords_final[name], dtype=float)[:, :2]
        aligned[name] = moved
        per_pair.append(
            {
                "query": name,
                "reference": reference,
                "median_displacement": float(np.median(np.linalg.norm(moved - coords_raw[name], axis=1))),
            }
        )

    notes = []
    aligned_paths = []
    for i, (name, adata) in enumerate(zip(names, slices)):
        moved = np.asarray(aligned[name], dtype=float)[:, :2]
        adata.obsm["spatial"] = originals[i]
        if z_spacing and z_spacing > 0:
            z = np.full(len(moved), float(i) * float(z_spacing))
            adata.obsm["spatial_3d_aligned"] = np.column_stack([moved, z])
            adata.obs["slice_z"] = z
        else:
            adata.obsm["spatial_aligned"] = moved
        out = os.path.join(output_dir, f"cast_aligned_slice_{i}.h5ad")
        _atomic_write_h5ad(adata, out)
        aligned_paths.append(out)
    if not (z_spacing and z_spacing > 0):
        notes.append(
            "no z spacing was declared, so the aligned coordinates are two columns under "
            "obsm['spatial_aligned']. Pass --z-spacing for a three-column obsm['spatial_3d_aligned']; "
            "CAST registers in plane and cannot supply a z itself."
        )
    if graph_strategy == "delaunay":
        notes.append(
            "graph_strategy='delaunay', which is this portal's default and not CAST's. Upstream "
            "defaults to 'convex', which routes through libpysal and is ~20x slower here for the "
            "same graph. Pass graph_strategy='convex' for upstream's behaviour."
        )

    displacement_csv = os.path.join(output_dir, "cast_displacement.csv")
    pd.DataFrame(per_pair).to_csv(displacement_csv, index=False)

    out = WorkerOutput(TOOL, task="alignment")
    out.add_warnings(notes)
    record_in_tissue(out, sum(n_supplied), sum(n_dropped))
    if any(n_dropped):
        out.add_params({"in_tissue_dropped_per_slice": [int(v) for v in n_dropped]})
    out.set_data(n_slices=len(slice_paths), n_shared_genes=len(panel), n_embedding_genes=len(feature_genes))
    out.add_output_files(
        {
            **{f"aligned_slice_{i}": p for i, p in enumerate(aligned_paths)},
            "displacement": displacement_csv,
        }
    )
    out.add_params(
        {
            "input_slices": list(slice_paths),
            "output_dir": output_dir,
            "reference": reference,
            "reference_index": reference_index,
            "epochs": epochs,
            "affine_iterations": affine_iterations,
            "ffd_iterations": ffd_iterations,
            "graph_strategy": graph_strategy,
            "n_hvg": n_hvg,
            "hvg_selection": (
                "once across all slices (scanpy highly_variable_genes, batch_key = slice), so every "
                "slice's embedding input has the same genes in the same order"
                if len(feature_genes) < len(panel)
                else "none: every shared gene was used"
            ),
            "normalize_target_sum": TARGET_SUM,
            "random_seed": random_seed,
            "z_spacing": z_spacing,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    )
    out.set_summary(n_slices_aligned=len(aligned_paths), reference=reference)
    out.set_analysis(
        f"CAST aligned {len(slice_paths)} sections onto {reference} with a graph-neural embedding, "
        f"an affine fit and a free-form deformation. ffd_iterations=0 gives an affine-only "
        f"(rigid-ish) result, which is the knob to reach for when the diagnosis said class B."
    )
    out.emit()


def main() -> None:
    parser = argparse.ArgumentParser(description="CAST single-cell non-rigid alignment.")
    parser.add_argument("--slice-h5ad", action="append", default=[], dest="slices")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--affine-iterations", type=int, default=500)
    parser.add_argument("--ffd-iterations", type=int, default=400)
    parser.add_argument("--graph-strategy", default="delaunay", choices=("delaunay", "convex"))
    parser.add_argument("--n-hvg", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--z-spacing", type=float, default=0.0)
    args = parser.parse_args()

    if len(args.slices) < 2:
        WorkerOutput.emit_error(TOOL, "cast needs at least two --slice-h5ad", task="alignment")
        sys.exit(1)
    if not (0 <= args.reference_index < len(args.slices)):
        WorkerOutput.emit_error(
            TOOL,
            f"reference_index {args.reference_index} is outside 0..{len(args.slices) - 1}",
            task="alignment",
        )
        sys.exit(1)
    try:
        run_align(
            args.slices,
            args.output_dir,
            args.reference_index,
            args.epochs,
            args.affine_iterations,
            args.ffd_iterations,
            args.graph_strategy,
            args.n_hvg,
            args.random_seed,
            args.z_spacing,
        )
    except Exception as exc:
        _log("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        WorkerOutput.emit_error(TOOL, f"{type(exc).__name__}: {exc}", task="alignment", exc=exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
