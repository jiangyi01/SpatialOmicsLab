#!/usr/bin/env python3

"""
SOMDE worker (CLI JSON in -> JSON out on stdout)

Robust compatibility fix:
- SOMDE upstream (somde/util.py qvalue, etc.) expects many NumPy functions to be
  accessible as scipy.<name> (legacy behavior). Modern SciPy removed these aliases.
- We patch SciPy by injecting *all missing NumPy symbols* into the scipy module
  (ONLY if scipy doesn't already have them). This prevents repeated failures like:
    AttributeError: Module 'scipy' has no attribute 'zeros_like'

Supported input_mode:
- visium_10x: counts_h5 + tissue_positions(_list).csv. Coordinates are the array grid
  (array_col, array_row). If those are constant the run stops, unless the caller passes
  allow_pixel_coord_fallback=True, in which case the full-resolution pixel columns are used and
  params.used_fallback / params.coord_source say so.
- h5ad: direct .h5ad file. Coordinates from obsm['spatial'] (two columns; a wider key -- a 3D or
  serial-stack frame -- is refused rather than truncated) or, when that key is absent,
  obs['x']/obs['y'].

Spots: background spots flagged ``obs['in_tissue'] == 0`` (CELLxGENE Visium exports carry every array
spot; in visium_10x mode the flag comes from the tissue-positions file) are left out before anything
is computed and reported in params.in_tissue_filter, a warning and the analysis. data.n_spots is the
count supplied, data.n_spots_used the count tested. No other spot is dropped.

Values: SOMDE stabilises the SOM-node table as negative-binomial counts and regresses on
log(total_count), so the expression it is handed must be counts. Every stored value is checked:
NaN/inf or negative values are an error, and so are non-integer values unless round_counts=True
(which rounds them and says so). The message names the layers and says whether adata.raw holds
counts; ``layer`` (h5ad mode) points the run at a raw-count layer instead of X, and
``use_raw_counts=True`` (h5ad mode, no layer) at ``adata.raw.X`` -- CELLxGENE exports keep the counts
there beside a processed X (``worker_utils.choose_counts_matrix``; params.expression_source says
which matrix was tested). A SOM node whose spots are all empty over the
tested genes would make that log -inf and SOMDE's least-squares fit die with "SVD did not converge";
it is refused by name before the fit, and empty spots that share a node with non-empty ones are
counted in data.n_spots_zero_total and a warning.

Gene selection is ours, not SOMDE's (SOMDE takes whatever table it is given): genes whose total
count is below min_counts are dropped, then, above max_genes, the max_genes genes with the largest
variance of log1p(counts normalised to 10,000 per spot) are kept. Both cuts are reported in
params.gene_selection, in a warning, and in the analysis text, so a gene that was never tested does
not read as "not spatially variable". The variance is accumulated sparse-aware in row blocks; the
full matrix is never densified.

Memory: SomNode.mtx takes the expression as a dense genes x spots DataFrame -- that is SOMDE's API,
so the dense table of the *selected* genes is intrinsic -- and SOMDE's Gaussian-process test keeps ten
n_nodes x n_nodes float64 eigenvector matrices (one per lengthscale) at once, plus working copies.
Both are estimated against available memory before SOMDE starts (params.memory_estimate_gib), and the
refusal names the knob for each term: max_genes for the table, som_dim for the kernels (a larger
som_dim means fewer SOM nodes). Spots are never subsampled.

random_seed is accepted for compatibility and listed under params.ignored: SomNode starts from a
homogeneous meshgrid codebook and batch-trains it, and the max_genes ranking is a deterministic sort
(ties keep file order), so no step of this run draws a random number.

Outputs under output_dir (each written to a .partial file and renamed into place):
- somde_result.csv
- somde_top_genes.csv
- figures/*.png

STDOUT MUST be final JSON only; all logs go to STDERR.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from worker_utils import (
    TISSUE_POSITIONS_NAMES,
    WorkerOutput,
    available_memory_bytes,
    build_svg_analysis,
    choose_counts_matrix,
    describe_reduction,
    distinct_significant_genes,
    expression_matrix_kind,
    find_tissue_positions,
    id_mismatch_msg,
    keep_in_tissue,
    make_names_unique_and_report,
    read_tissue_positions,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    spatial_coords,
)


def _eprint(*args):
    print(*args, file=sys.stderr, flush=True)


# -----------------------------------------------------------------------------
# Patch SciPy legacy NumPy-alias expectations used by SOMDE.
# MUST happen before importing/using somde (and before somde.util calls scipy.*).
# -----------------------------------------------------------------------------
def _patch_scipy_numpy_aliases():
    try:
        import scipy as sp  # SOMDE uses `import scipy as sp`

        # Inject ALL missing NumPy symbols into SciPy namespace (only if missing).
        # This is the only practical way to avoid "one error at a time" breakages.
        for name in dir(np):
            if name.startswith("_"):
                continue
            if not hasattr(sp, name):
                try:
                    setattr(sp, name, getattr(np, name))
                except Exception:
                    # Some numpy attributes may be read-only or weird; ignore.
                    pass

        # Also ensure common constants exist
        for name, val in [("inf", np.inf), ("nan", np.nan), ("pi", np.pi), ("e", np.e)]:
            if not hasattr(sp, name):
                try:
                    setattr(sp, name, val)
                except Exception:
                    pass

    except Exception as e:
        _eprint(f"[SOMDE] SciPy patch warning: {e}")


_patch_scipy_numpy_aliases()

# Headless plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Scanpy for reading 10x h5
import scanpy as sc

# SOMDE
import somde

#: What runs. SOMDE is the only implementation here, so there is no fallback method; the one
#: substitution this worker can make is the coordinate frame (see _visium_coordinates).
METHOD_NAME = "SOMDE (SomNode self-organising-map condensation + SpatialDE Gaussian-process test)"

#: normalize_total target used by the max_genes variance ranking and by the figures.
TARGET_SUM = 1e4

#: Entries per row block of the chunked variance; bounds the float64 temporaries of one block.
_VARIANCE_BLOCK_ENTRIES = 1 << 24

#: Bytes per value of the dense genes x spots table SomNode.mtx is handed (float32).
_DENSE_BYTES_PER_VALUE = 4

#: Copies of the genes x SOM-nodes table SOMDE keeps (mtx, stabilize, regress_out), float64.
_NODE_TABLE_COPIES = 3

#: SomNode.run -> Sparun searches ten squared-exponential lengthscales (np.logspace(l_min, l_max, 10))
#: over the SOM-node coordinates, and somde.util.dyn_de keeps every kernel's n_nodes x n_nodes float64
#: eigenvector matrix U in US_mats until all genes are fitted -- the same design as SpatialDE v1.
_N_KERNELS = 10

#: Beside the stored eigenvector matrices at the peak: the last kernel K, eigh's input copy and its
#: workspace, and gower_scaling_factor's temporaries -- four more n_nodes x n_nodes float64 matrices
#: (get_l_limits' n_nodes^2 distance matrices come earlier and are smaller than this peak).
_KERNEL_TRANSIENT_COPIES = 4

VALID_INPUT_MODES = ("h5ad", "visium_10x")

#: Values within this distance of an integer count as integers (float32 noise, not normalisation).
INTEGER_TOL = 1e-6

#: Stored values examined per step by the count audit, so its temporaries stay bounded.
_AUDIT_CHUNK = 1 << 24


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _text(value) -> str:
    """A payload string, with a missing value as ``""`` -- never the literal ``"None"``."""
    return "" if value is None else str(value)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _safe_dense(X):
    """Convert sparse -> dense float32 if needed (one copy, not two)."""
    try:
        import scipy.sparse as sps

        if sps.issparse(X):
            return X.toarray().astype(np.float32, copy=False)
    except Exception:
        pass
    return np.asarray(X, dtype=np.float32)


def _atomic_to_csv(frame: pd.DataFrame, path: str) -> None:
    tmp = path + ".partial"
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _require_inputs(
    input_mode: str,
    output_dir: str,
    h5ad_path: str,
    counts_h5: str,
    spatial_dir: str,
    max_genes: int,
    som_dim: int,
    top_k_genes: int,
    layer: str = "",
    use_raw_counts: bool = False,
) -> None:
    """Refuse, by name, before any file is opened or any directory is created."""
    if not output_dir:
        raise ValueError(
            "output_dir is required: it is the directory SOMDE writes somde_result.csv, "
            "somde_top_genes.csv and figures/ into, and none was given."
        )
    if input_mode not in VALID_INPUT_MODES:
        raise ValueError(f"Unsupported input_mode: {input_mode}. Supported: 'visium_10x', 'h5ad'.")
    if input_mode == "h5ad" and not h5ad_path:
        raise ValueError("input_mode='h5ad' needs h5ad_path (the spatial .h5ad to test); none was given.")
    if input_mode == "visium_10x":
        missing = [name for name, value in (("counts_h5", counts_h5), ("spatial_dir", spatial_dir)) if not value]
        if missing:
            raise ValueError(
                f"input_mode='visium_10x' needs counts_h5 and spatial_dir; missing: {', '.join(missing)}. "
                "For an .h5ad pass input_mode='h5ad' and h5ad_path instead."
            )
    if int(max_genes) < 1:
        raise ValueError(
            f"max_genes must be at least 1 (got {max_genes}); pass a value at or above the gene count to test every gene."
        )
    if int(som_dim) < 1:
        raise ValueError(f"som_dim must be at least 1 (got {som_dim}); it is the average number of spots per SOM node.")
    if int(top_k_genes) < 0:
        raise ValueError(f"top_k_genes must be 0 or more (got {top_k_genes}).")
    if input_mode == "h5ad" and use_raw_counts and layer:
        raise ValueError(
            f"use_raw_counts=True tests adata.raw, and layer='{layer}' names a layer: they name two different "
            "matrices. Pass one of them (layer='' with use_raw_counts=True, or the layer with use_raw_counts=False)."
        )


def _set_xy(adata: sc.AnnData, x, y, source: str) -> None:
    """Attach the coordinates SOMDE will read, refusing non-finite ones by count."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    bad = ~(np.isfinite(x) & np.isfinite(y))
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} of {x.size} spots have a missing or non-finite coordinate in {source}; "
            "SOMDE places every spot on its self-organising map and cannot place those. Fix or drop them first."
        )
    adata.obs["x"] = x
    adata.obs["y"] = y


def _h5ad_coordinates(adata: sc.AnnData) -> str:
    """Set obs x/y from obsm['spatial'] (exactly two columns) or existing obs x/y; return the source.

    ``spatial_coords`` raises on a three-column key instead of dropping z, which would lay every
    section of a serial stack onto one plane and test genes on a tissue that does not exist.
    """
    if "spatial" in adata.obsm:
        coords, _ = spatial_coords(adata, "spatial", 2, "SOMDE")
        _set_xy(adata, coords[:, 0], coords[:, 1], "obsm['spatial']")
        return "obsm['spatial']"
    if "x" in adata.obs.columns and "y" in adata.obs.columns:
        _set_xy(adata, adata.obs["x"].to_numpy(), adata.obs["y"].to_numpy(), "obs['x']/obs['y']")
        return "obs['x']/obs['y']"
    raise KeyError(
        "SOMDE needs spot coordinates: obsm['spatial'] (two columns) or obs['x'] and obs['y']. "
        f"This h5ad has obsm keys {list(adata.obsm.keys())} and neither obs column."
    )


def _visium_coordinates(positions: pd.DataFrame, allow_pixel_coord_fallback: bool = False):
    """``(x, y, source, used_fallback)`` from a tissue-positions frame, the array grid first.

    The array grid is the stable frame. When it is constant it carries no position at all, and the
    old code switched to the pixel columns with only a stderr line to say so. That switch now
    happens only when the caller allows it, and the payload records it.
    """
    x = positions["array_col"].astype(float).to_numpy()
    y = positions["array_row"].astype(float).to_numpy()
    if np.nanstd(x) > 0.0 and np.nanstd(y) > 0.0:
        return x, y, "array_col/array_row", False
    if not allow_pixel_coord_fallback:
        raise ValueError(
            f"array_col/array_row in the tissue-positions file are constant (or missing) across the {len(x)} "
            "spots, so they place every spot on one line and SOMDE cannot run on them. Pass "
            "allow_pixel_coord_fallback=True to run on pxl_col_in_fullres/pxl_row_in_fullres instead (the "
            "payload then records params.used_fallback=True), or fix the positions file."
        )
    _eprint("[SOMDE] array_row/array_col are constant; using pixel coordinates (allow_pixel_coord_fallback=True).")
    px = positions["pxl_col_in_fullres"].astype(float).to_numpy()
    py = positions["pxl_row_in_fullres"].astype(float).to_numpy()
    return px, py, "pxl_col_in_fullres/pxl_row_in_fullres", True


def _load_visium_10x(counts_h5: str, spatial_dir: str) -> sc.AnnData:
    """
    Load 10x Visium filtered_feature_bc_matrix.h5 and attach the tissue-positions columns
    (array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres) to obs. The caller picks the
    coordinate frame with _visium_coordinates.

    spatial_dir can be:
      - /path/to/sample/            (contains spatial/)
      - /path/to/sample/spatial/    (directly)
    """
    _eprint(f"[SOMDE] Reading 10x h5: {counts_h5}")
    adata = sc.read_10x_h5(counts_h5)
    make_names_unique_and_report(adata)

    # Find the spatial subdir
    spatial_path = spatial_dir
    if os.path.isdir(os.path.join(spatial_dir, "spatial")):
        spatial_path = os.path.join(spatial_dir, "spatial")

    tpos = find_tissue_positions(spatial_path)
    if tpos is None:
        raise FileNotFoundError(f"Cannot find {' or '.join(TISSUE_POSITIONS_NAMES)} at: {spatial_path}")

    df = read_tissue_positions(tpos).set_index("barcode")

    common = adata.obs_names.intersection(df.index)
    if len(common) == 0:
        raise ValueError(id_mismatch_msg("barcodes", "10x h5", adata.obs_names, os.path.basename(tpos), df.index))

    adata = adata[common].copy()
    df = df.loc[common].copy()
    # in_tissue travels with the spot, so a raw_feature_bc_matrix (every array spot) has its
    # background left out exactly as an h5ad's is; a filtered matrix is 1 everywhere.
    for col in ("in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"):
        adata.obs[col] = df[col].to_numpy()

    _eprint(f"[SOMDE] Loaded: {adata.n_obs} spots x {adata.n_vars} genes (mode=visium_10x)")
    return adata


def _where_counts_could_be(adata) -> str:
    """The places in this object a count matrix could live, for an error message."""
    layers = list(adata.layers.keys())
    where = f"layers in this file: {layers}" if layers else "this file has no layers"
    raw = getattr(adata, "raw", None)
    if raw is not None:
        try:
            kind = expression_matrix_kind(raw.X)
        except Exception:
            kind = "unreadable"
        if kind == "counts":
            where += (
                f"; adata.raw is present ({raw.n_vars} genes) and holds raw counts -- pass use_raw_counts=True "
                "(with layer='') to test them"
            )
        else:
            where += (
                f"; adata.raw is present ({raw.n_vars} genes) but holds {kind.replace('_', ' ')} values, not counts"
            )
    return where


def _counts_matrix_choice(adata: sc.AnnData, use_raw_counts: bool):
    """``worker_utils.choose_counts_matrix`` for an h5ad tested as X: ``(adata, info)``.

    ``use_raw_counts=True`` swaps in ``adata.raw.X`` (CELLxGENE exports keep the counts there and a
    processed X); a negative or non-finite X is refused naming it. raw.var is a second gene index, so
    it is deduplicated and counted again, keeping the barcode count from the load.
    """
    adata, info = choose_counts_matrix(adata, use_raw_counts)
    if info["expression_source"] == "raw.X":
        before = dict(adata.uns.get("identifier_renames") or {})
        make_names_unique_and_report(
            adata,
            into={"n_genes_renamed": 0, "n_cells_renamed": int(before.get("n_cells_renamed", 0))},
            axes=("var",),
        )
        _eprint(f"[SOMDE] use_raw_counts=True: testing adata.raw.X ({adata.n_vars} genes)")
    return adata, info


def _expression_matrix(adata: sc.AnnData, layer: str, input_mode: str) -> str:
    """Point ``adata.X`` at the matrix SOMDE will be handed and return its name for the payload.

    ``layer`` is h5ad-only; an absent layer is an error listing the ones the file has, never a
    silent use of X.
    """
    if not layer or input_mode != "h5ad":
        return "X"
    if layer not in adata.layers:
        # ValueError, not KeyError: str(KeyError) wraps the whole sentence in quotes on the payload.
        raise ValueError(
            f"layer='{layer}' is not a layer of this {adata.n_obs} x {adata.n_vars} h5ad ({_where_counts_could_be(adata)}). "
            "Pass layer='' to analyse adata.X, or name one of the layers listed."
        )
    adata.X = adata.layers[layer]
    return f"layers['{layer}']"


def audit_values(X, tol: float = INTEGER_TOL, chunk: int = _AUDIT_CHUNK) -> dict:
    """Count non-finite, negative and non-integer values over EVERY stored value, in chunks.

    Sparse: the stored entries (``X.data``); dense: all entries. An integer dtype holds no
    non-finite or non-integer value, so only its sign is checked.
    """
    import scipy.sparse as sps

    values = X.data if sps.issparse(X) else np.asarray(X).ravel()
    report = {"n_checked": int(values.size), "n_non_finite": 0, "n_negative": 0, "n_non_integer": 0, "example": None}
    integral = values.dtype.kind in ("i", "u", "b")
    for start in range(0, int(values.size), int(chunk)):
        block = values[start : start + int(chunk)]
        if integral:
            report["n_negative"] += int((block < 0).sum())
            continue
        block = np.asarray(block, dtype=np.float64)
        finite = np.isfinite(block)
        report["n_non_finite"] += int(block.size - int(finite.sum()))
        vals = block[finite]
        report["n_negative"] += int((vals < 0).sum())
        off = np.abs(vals - np.rint(vals)) > tol
        k = int(off.sum())
        if k and report["example"] is None:
            report["example"] = float(vals[off][0])
        report["n_non_integer"] += k
    return report


def _check_counts(adata: sc.AnnData, source: str, round_counts: bool) -> dict:
    """Refuse values SOMDE cannot model as counts; round on request. Returns the audit report.

    SomNode.norm() is NaiveDE's Anscombe stabilisation of negative-binomial counts followed by a
    regression on log(total_count). Log-normalised X ran to completion as if it were counts, a
    z-scaled X died in the min_counts filter with a message blaming min_counts, and a node of
    zero-total spots died in lstsq with "SVD did not converge".
    """
    import scipy.sparse as sps

    report = audit_values(adata.X)
    report["rounded"] = False
    where = _where_counts_could_be(adata)
    if report["n_non_finite"] or report["n_negative"]:
        raise ValueError(
            f"{source} holds {report['n_non_finite']} NaN/inf and {report['n_negative']} negative values of "
            f"{report['n_checked']} stored. SOMDE models raw counts (NaiveDE's negative-binomial stabilisation, then "
            "a regression on log total counts), so this looks like scaled or centred data, and round_counts cannot "
            f"make it counts. Point layer at a raw-count layer ({where}), or pass input_mode='visium_10x' with Space "
            "Ranger's counts h5."
        )
    if report["n_non_integer"]:
        if not round_counts:
            raise ValueError(
                f"{report['n_non_integer']} of {report['n_checked']} stored values of {source} are not integers "
                f"(e.g. {report['example']!r}): this looks like normalised data, and SOMDE models raw counts. Point "
                f"layer at a raw-count layer ({where}), pass input_mode='visium_10x' with Space Ranger's counts h5, or "
                "pass round_counts=True if these are counts stored as near-integer floats -- the rounding is then "
                "recorded in params and warnings. Nothing was rounded."
            )
        if sps.issparse(adata.X):
            X = adata.X.copy()
            X.data = np.rint(X.data)
            X.eliminate_zeros()
            adata.X = X
        else:
            adata.X = np.rint(np.asarray(adata.X, dtype=np.float32))
        report["rounded"] = True
        _eprint(f"[SOMDE] round_counts=True: rounded {report['n_non_integer']} non-integer values of {source}")
    return report


def _spot_totals(X) -> np.ndarray:
    """Per-spot totals over the columns of ``X`` (float64), sparse-aware."""
    import scipy.sparse as sps

    if sps.issparse(X):
        return np.asarray(X @ np.ones(X.shape[1], dtype=np.float64)).ravel()
    return np.asarray(X).sum(axis=1, dtype=np.float64).ravel()


def _check_som_nodes(ninfo: pd.DataFrame, n_zero_spots: int, n_spots: int, n_genes: int, flag_present: bool) -> None:
    """Refuse, before SOMDE's fit, a SOM node whose total count is zero.

    SomNode.mtx sets each node's value to 0.5 * max + 0.5 * mean over its spots, and norm() regresses
    on log(total_count): a node holding only empty spots is -inf there, and np.linalg.lstsq then dies
    with "SVD did not converge" -- which names neither the node nor the spots.
    """
    totals = pd.to_numeric(ninfo["total_count"], errors="coerce").to_numpy(dtype=np.float64)
    bad = ~(np.isfinite(totals) & (totals > 0))
    if not bad.any():
        return
    background = (
        "Background spots flagged obs['in_tissue'] == 0 were already left out, so these are spots the file "
        "marks as tissue."
        if flag_present
        else "If they are background spots of a Visium array, add obs['in_tissue'] (1 = tissue, 0 = background) "
        "and they are left out automatically."
    )
    raise ValueError(
        f"{int(bad.sum())} of {len(totals)} SOM nodes hold only spots with zero counts over the {n_genes} tested genes "
        f"({n_zero_spots} of {n_spots} spots are empty over them). SOMDE's norm() regresses on log(total_count), "
        "which is -inf for such a node, and its least-squares fit then fails ('SVD did not converge'). Remove the "
        f"empty spots before calling this tool. {background}"
    )


def _gene_totals(X) -> np.ndarray:
    """Per-gene total counts (float64), sparse-aware."""
    import scipy.sparse as sps

    if sps.issparse(X):
        return np.asarray(X.T @ np.ones(X.shape[0], dtype=np.float64)).ravel()
    return np.asarray(X).sum(axis=0, dtype=np.float64).ravel()


def _log_normalized_gene_variance(X, target_sum: float = TARGET_SUM) -> np.ndarray:
    """Per-gene variance of log1p(counts normalised to ``target_sum`` per spot), never densified.

    What ``sc.pp.normalize_total(target_sum) + sc.pp.log1p`` followed by ``np.var(axis=0)`` gives,
    accumulated in float64 over row blocks: a spot's total is taken over the columns of ``X`` (the
    genes that passed min_counts, as before), and a spot whose total is zero stays zero, as
    normalize_total leaves it. The old code densified the whole normalised matrix to get this --
    about 37 GB on a 507,684-bin Visium HD slide -- and ``.var`` then allocated another copy.
    """
    import scipy.sparse as sps

    n_obs, n_vars = int(X.shape[0]), int(X.shape[1])
    s1 = np.zeros(n_vars, dtype=np.float64)
    s2 = np.zeros(n_vars, dtype=np.float64)
    if n_obs == 0:
        return s1
    sparse = sps.issparse(X)
    if sparse:
        X = X.tocsr()
    step = max(1, _VARIANCE_BLOCK_ENTRIES // max(n_vars, 1))
    for start in range(0, n_obs, step):
        stop = min(n_obs, start + step)
        if sparse:
            block = X[start:stop]
            data = np.asarray(block.data, dtype=np.float64)
            rows = np.repeat(np.arange(stop - start), np.diff(block.indptr))
            totals = np.bincount(rows, weights=data, minlength=stop - start)
            scale = np.zeros_like(totals)
            np.divide(target_sum, totals, out=scale, where=totals > 0)
            vals = np.log1p(data * scale[rows])
            s1 += np.bincount(block.indices, weights=vals, minlength=n_vars)
            s2 += np.bincount(block.indices, weights=vals * vals, minlength=n_vars)
        else:
            block = np.asarray(X[start:stop], dtype=np.float64)
            totals = block.sum(axis=1)
            scale = np.zeros_like(totals)
            np.divide(target_sum, totals, out=scale, where=totals > 0)
            vals = np.log1p(block * scale[:, None])
            s1 += vals.sum(axis=0)
            s2 += (vals * vals).sum(axis=0)
    mean = s1 / n_obs
    return np.maximum(s2 / n_obs - mean * mean, 0.0)


def _subset_genes_simple(
    adata: sc.AnnData,
    max_genes: int = 2000,
    min_counts: int = 1,
):
    """
    Robust gene subsetting without seurat_v3/loess dependency:
      1) filter genes by total counts >= min_counts
      2) if > max_genes, rank by variance on log1p(normalized) and keep top max_genes

    Returns ``(adata_small, info)``; ``info`` is what params.gene_selection reports. Sparse-aware
    throughout (see _log_normalized_gene_variance). Genes above the cut keep variance order, as
    before; ties keep file order.
    """
    gene_sum = _gene_totals(adata.X)
    keep = gene_sum >= float(min_counts)
    n_kept = int(keep.sum())
    if n_kept == 0:
        raise ValueError(
            f"After min_counts filtering (min_counts={min_counts}), no genes remain out of {adata.n_vars}."
        )
    keep_idx = np.flatnonzero(keep)
    info: dict[str, Any] = {
        "min_counts": int(min_counts),
        "max_genes": int(max_genes),
        "n_genes_supplied": int(adata.n_vars),
        "n_genes_after_min_counts": n_kept,
        "ranking": None,
    }

    if n_kept <= max_genes:
        _eprint(f"[SOMDE] Gene filter: kept {n_kept} genes (<= max_genes={max_genes}).")
        idx = keep_idx
    else:
        X = adata.X if n_kept == adata.n_vars else adata.X[:, keep_idx]
        var = _log_normalized_gene_variance(X)
        order = np.argsort(-var, kind="stable")[: int(max_genes)]
        idx = keep_idx[order]
        info["ranking"] = (
            f"top {int(max_genes)} of {n_kept} by variance of log1p(counts normalised to "
            f"{int(TARGET_SUM)} per spot); ties keep file order"
        )
        _eprint(f"[SOMDE] Selected {idx.size} genes via variance ranking (max_genes={max_genes}).")
    c = adata[:, idx].copy()
    info["n_genes_used"] = int(c.n_vars)
    return c, info


def _gene_selection_note(info: dict) -> str:
    """describe_reduction's sentence, naming whichever of our two cuts dropped genes."""
    cuts = []
    if info["n_genes_after_min_counts"] < info["n_genes_supplied"]:
        cuts.append(
            f"the min_counts={info['min_counts']} total-count filter "
            f"({info['n_genes_supplied'] - info['n_genes_after_min_counts']} genes)"
        )
    if info["n_genes_used"] < info["n_genes_after_min_counts"]:
        cuts.append(
            f"the max_genes={info['max_genes']} variance ranking "
            f"({info['n_genes_after_min_counts'] - info['n_genes_used']} genes; raise max_genes to test more)"
        )
    return describe_reduction(
        "genes", int(info["n_genes_supplied"]), int(info["n_genes_used"]), reason=" and ".join(cuts)
    )


def _som_grid_side(n_spots: int, som_dim: int) -> int:
    """The side of SOMDE's square SOM grid: ``int(sqrt(n_spots // som_dim))``, as SomNode computes it."""
    side = int(np.sqrt(int(n_spots) // int(som_dim)))
    if side < 2:
        raise ValueError(
            f"som_dim={som_dim} with {n_spots} spots gives SOMDE a {side}x{side} SOM (it builds "
            f"int(sqrt(n_spots // som_dim)) nodes per side), and its Gaussian-process test needs at least a 2x2 "
            f"grid. som_dim is the average number of spots per node: lower it to at most {max(1, int(n_spots) // 4)}."
        )
    return side


def _available_memory_bytes():
    """worker_utils' reader: MemAvailable or the room under the cgroup limit, page cache reclaimable.

    Kept as a module-level seam so the preflight can be exercised with a fixed number. The old local
    reader took the cgroup LIMIT as the room left, which overstated it on a box already using memory.
    """
    return available_memory_bytes()


def _table_bytes(n_spots: int, n_genes: int, n_nodes: int) -> int:
    """The per-gene term: the dense genes x spots table SomNode.mtx takes, and SOMDE's node tables."""
    return int(n_spots) * int(n_genes) * _DENSE_BYTES_PER_VALUE + int(n_genes) * int(n_nodes) * 8 * _NODE_TABLE_COPIES


def _kernel_bytes(n_nodes: int) -> int:
    """The per-node term: dyn_de's stored eigenvector matrices plus the working ones, n_nodes^2 float64 each."""
    return int(n_nodes) * int(n_nodes) * 8 * (_N_KERNELS + _KERNEL_TRANSIENT_COPIES)


def somde_memory_bytes(n_spots: int, n_genes: int, n_nodes: int) -> int:
    """Peak bytes SOMDE needs for ``n_spots`` spots, ``n_genes`` tested genes and ``n_nodes`` SOM nodes."""
    return _table_bytes(n_spots, n_genes, n_nodes) + _kernel_bytes(n_nodes)


def _som_dim_that_fits(n_spots: int, n_genes: int, n_nodes: int, available: float):
    """The smallest som_dim whose SOM grid fits in ``available`` bytes, or None if even a 2x2 grid does not.

    Searches down from the current grid side. The grid side is int(sqrt(n_spots // som_dim)), so
    som_dim = n_spots // (side + 1)^2 + 1 is the smallest value that gives a side of at most ``side``.
    """
    side = int(np.sqrt(max(int(n_nodes), 0)))
    while side >= 2:
        if somde_memory_bytes(n_spots, n_genes, side * side) <= available:
            return int(n_spots) // ((side + 1) * (side + 1)) + 1
        side -= 1
    return None


def _check_dense_table_fits(n_spots: int, n_genes: int, n_nodes: int, available=None) -> int:
    """Refuse, with the numbers and the knob for each term, before SOMDE starts. Never subsamples.

    Two terms. The dense genes x spots table SomNode.mtx takes (its API; max_genes sets its gene count),
    and the Gaussian-process kernels SOMDE's test holds over the SOM nodes: ten n_nodes x n_nodes float64
    eigenvector matrices stored at once, plus working copies (som_dim sets the node count). The old
    estimate had only the first term, so a VisiumHD slide at the default som_dim=20 (159x159 = 25,281
    nodes) passed at ~4.9 GiB while dyn_de needs ~67 GiB, and a smaller host OOM-killed the run with no
    JSON. Returns the estimate in bytes.
    """
    table = _table_bytes(n_spots, n_genes, n_nodes)
    kernels = _kernel_bytes(n_nodes)
    need = table + kernels
    if available is None:
        available = _available_memory_bytes()
    if available is not None and need > available:
        gib = float(1 << 30)
        fits = _som_dim_that_fits(n_spots, n_genes, n_nodes, float(available)) if table < available else None
        som_dim_advice = (
            f" som_dim={fits} or more would give a grid that fits beside this table." if fits is not None else ""
        )
        raise MemoryError(
            f"SOMDE needs about {need / gib:.1f} GiB for {n_spots} spots, {n_genes} genes and {n_nodes} SOM nodes, "
            f"but about {available / gib:.1f} GiB is available here (MemAvailable / room under the cgroup limit). "
            f"Two terms: (1) the dense genes x spots table SomNode.mtx takes (its API), {table / gib:.1f} GiB -- "
            "max_genes sets its gene count (the cut is reported in params.gene_selection and the analysis); "
            f"(2) the Gaussian-process kernels SOMDE's test keeps over the SOM nodes ({_N_KERNELS} stored + "
            f"{_KERNEL_TRANSIENT_COPIES} working {n_nodes}x{n_nodes} float64 matrices), {kernels / gib:.1f} GiB -- "
            "som_dim sets the node count: the grid is int(sqrt(n_spots // som_dim)) nodes per side, so a larger "
            f"som_dim means fewer, coarser nodes.{som_dim_advice} Lower whichever term dominates, or run where more "
            "memory is available. Spots are never subsampled."
        )
    return need


_GENE_COLUMNS = ["g", "gene", "Gene", "gene_name", "name"]


def _gene_column(result: pd.DataFrame) -> str:
    """The column of ``result`` holding gene names, or ``""`` if SOMDE named it something new.

    Mirrors the per-row lookup the summary/figure loops do, so the de-duplication and the names
    that get reported can never disagree about which column is the gene.
    """
    for k in _GENE_COLUMNS:
        if k in result.columns:
            return k
    return ""


def _build_somde_inputs(adata: sc.AnnData) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Build inputs consistent with SOMDE README:
      - X: n_spots x 2 coordinate matrix
      - df: genes x spots expression dataframe (dense: SomNode.mtx's API)
    """
    if ("x" not in adata.obs) or ("y" not in adata.obs):
        raise ValueError("Missing adata.obs['x']/'y'. Coordinates not loaded.")

    dense = _safe_dense(adata.X)  # spots x genes

    df = pd.DataFrame(
        dense.T,  # genes x spots
        index=adata.var_names.astype(str),
        columns=adata.obs_names.astype(str),
    )

    corinfo = pd.DataFrame(
        {
            "x": adata.obs["x"].to_numpy(dtype=np.float32),
            "y": adata.obs["y"].to_numpy(dtype=np.float32),
        },
        index=adata.obs_names.astype(str),
    )
    corinfo["total_count"] = df.sum(axis=0).astype(np.float32)

    X = corinfo[["x", "y"]].to_numpy(dtype=np.float32)
    return df, corinfo, X


def _log_normalized_columns(X, cols, target_sum: float = TARGET_SUM) -> np.ndarray:
    """log1p(normalize_total) values of the columns ``cols`` only, as a dense spots x len(cols) array.

    The spot totals are taken over every column of ``X``, exactly as normalize_total on the whole
    matrix would; only the plotted columns are ever made dense.
    """
    import scipy.sparse as sps

    n_obs = int(X.shape[0])
    if sps.issparse(X):
        totals = np.asarray(X @ np.ones(X.shape[1], dtype=np.float64)).ravel()
        sub = X.tocsc()[:, list(cols)].toarray().astype(np.float64)
    else:
        arr = np.asarray(X)
        totals = arr.sum(axis=1, dtype=np.float64).ravel()
        sub = np.asarray(arr[:, list(cols)], dtype=np.float64)
    scale = np.zeros(n_obs, dtype=np.float64)
    np.divide(target_sum, totals, out=scale, where=totals > 0)
    return np.log1p(sub * scale[:, None]).astype(np.float32)


def _figure_name(gene: str) -> str:
    """``topgene_<gene>_tissue.png``; a path separator inside a gene name cannot escape figures/."""
    safe = gene.replace(os.sep, "_")
    if os.altsep:
        safe = safe.replace(os.altsep, "_")
    return f"topgene_{safe}_tissue.png"


def _plot_spatial_gene(x: np.ndarray, y: np.ndarray, val: np.ndarray, title: str, out_png: str, s: float = 6.0):
    plt.figure(figsize=(6, 5))
    sca = plt.scatter(x, y, c=val, s=s)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(sca, fraction=0.046, pad=0.04)
    plt.tight_layout()
    tmp = out_png + ".partial"
    plt.savefig(tmp, dpi=200, format="png")
    plt.close()
    os.replace(tmp, out_png)


def run_somde(
    input_mode: str,
    counts_h5: str,
    spatial_dir: str,
    output_dir: str,
    max_genes: int = 2000,
    min_counts: int = 1,
    som_dim: int = 20,
    top_k_genes: int = 20,
    random_seed: int = 0,
    h5ad_path: str = "",
    allow_pixel_coord_fallback: bool = False,
    layer: str = "",
    round_counts: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    output_dir = _text(output_dir)
    h5ad_path = _text(h5ad_path)
    counts_h5 = _text(counts_h5)
    spatial_dir = _text(spatial_dir)
    layer = _text(layer).strip()
    use_raw_counts = bool(use_raw_counts)
    _require_inputs(
        input_mode,
        output_dir,
        h5ad_path,
        counts_h5,
        spatial_dir,
        max_genes,
        som_dim,
        top_k_genes,
        layer=layer,
        use_raw_counts=use_raw_counts,
    )
    _ensure_dir(output_dir)

    used_pixel_fallback = False
    if input_mode == "h5ad":
        # scanpy is already imported as `sc` at module scope; a local re-import here made
        # `sc` function-local and broke the visium_10x branch with "referenced before assignment".
        _eprint(f"[SOMDE] Loading h5ad: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path)
        # Symmetric with _load_visium_10x, which dedups on the way in. Without this the h5ad
        # route ranks both copies of a duplicated symbol under one name, and the payload's
        # n_genes_renamed reports 0 -- truthfully, having never looked.
        make_names_unique_and_report(adata)
    elif input_mode == "visium_10x":
        adata = _load_visium_10x(counts_h5=counts_h5, spatial_dir=spatial_dir)
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}. Supported: 'visium_10x', 'h5ad'.")
    flag_present = "in_tissue" in adata.obs.columns
    # Background spots (obs['in_tissue'] == 0) are glass: CELLxGENE Visium exports carry every array
    # spot, most with ambient counts, and on Muscle every one of the 3396 empty spots was one of them.
    # Left out before the coordinates, the gene cut and the SOM are built, and reported.
    adata, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata, "spots")
    if n_spots_off_tissue:
        _eprint(
            f"[SOMDE] Left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0 "
            f"(background); {adata.n_obs} in-tissue spots are tested"
        )
    # Which matrix holds the counts (h5ad tested as X only: a named layer is audited by _check_counts,
    # and Space Ranger's counts h5 has no adata.raw).
    counts_info = None
    if input_mode == "h5ad" and not layer:
        adata, counts_info = _counts_matrix_choice(adata, use_raw_counts)
    if input_mode == "h5ad":
        coord_source = _h5ad_coordinates(adata)
    else:
        x, y, coord_source, used_pixel_fallback = _visium_coordinates(adata.obs, allow_pixel_coord_fallback)
        _set_xy(adata, x, y, coord_source)
    grid_side = _som_grid_side(adata.n_obs, int(som_dim))
    expression_source = _expression_matrix(adata, layer, input_mode)
    if counts_info is not None:
        expression_source = counts_info["expression_source"]
    value_report = _check_counts(adata, expression_source, bool(round_counts))
    adata_small, gene_info = _subset_genes_simple(
        adata,
        max_genes=int(max_genes),
        min_counts=int(min_counts),
    )
    n_genes_supplied = int(adata.n_vars)
    n_genes_renamed = int(adata.uns.get("identifier_renames", {}).get("n_genes_renamed", 0))
    del adata  # the full matrix is not needed past the gene cut; free it before the dense table

    # Empty over the tested genes: harmless inside a node that also holds non-empty spots, fatal to
    # SOMDE's fit when a whole node is empty (checked after mtx, below).
    n_zero_spots = int((_spot_totals(adata_small.X) <= 0).sum())

    need_bytes = _check_dense_table_fits(adata_small.n_obs, adata_small.n_vars, grid_side * grid_side)
    df, corinfo, X = _build_somde_inputs(adata_small)

    # SOMDE prints progress with print(); stdout is this worker's JSON channel.
    with contextlib.redirect_stdout(sys.stderr):
        _eprint(f"[SOMDE] Running SomNode(som_dim={som_dim}) ...")
        som = somde.SomNode(X, int(som_dim))

        # As per SOMDE README: mtx(df) with ONE argument (df is genes x spots)
        _ndf, _ninfo = som.mtx(df)
        _check_som_nodes(_ninfo, n_zero_spots, int(adata_small.n_obs), int(adata_small.n_vars), flag_present)

        _eprint("[SOMDE] Normalizing & running spatial variable gene detection ...")
        _ = som.norm()
        result, svnum = som.run()
    del df

    if not isinstance(result, pd.DataFrame):
        result = pd.DataFrame(result)

    result_csv = os.path.join(output_dir, "somde_result.csv")
    _atomic_to_csv(result, result_csv)

    # SOMDE emits one row per (gene, fitted length-scale) and keeps EVERY row tied for a gene's
    # best max_ll (somde.util.get_mll_results), so a gene whose kernels converge to the same
    # likelihood appears more than once. Collapse to one row per gene -- keeping the first, i.e.
    # the highest LLR, since `result` is sorted by LLR descending -- before taking the top k, or a
    # tie spends a slot on a repeat and overwrites that gene's figure twice. The raw table written
    # above is left exactly as SOMDE produced it.
    gene_col = _gene_column(result)
    result_by_gene = result.drop_duplicates(subset=[gene_col]) if gene_col else result

    top_k = int(top_k_genes)
    top_df = result_by_gene.head(top_k).copy()
    top_csv = os.path.join(output_dir, "somde_top_genes.csv")
    _atomic_to_csv(top_df, top_csv)

    fig_dir = os.path.join(output_dir, "figures")
    _ensure_dir(fig_dir)

    # Extract top gene names for summary (and the figures, in the same order)
    top_gene_names: list[str] = []
    for _, row in top_df.iterrows():
        gene = None
        for k in _GENE_COLUMNS:
            if k in row.index:
                gene = str(row[k])
                break
        if gene is None:
            gene = str(row.iloc[0])
        top_gene_names.append(gene)

    # Plot top genes on tissue: only the plotted columns are made dense.
    gene_to_col = {g: i for i, g in enumerate(adata_small.var_names.astype(str))}
    plotted = [g for g in top_gene_names if g in gene_to_col]
    expr = _log_normalized_columns(adata_small.X, [gene_to_col[g] for g in plotted])
    x_sp = adata_small.obs["x"].to_numpy(dtype=np.float32)
    y_sp = adata_small.obs["y"].to_numpy(dtype=np.float32)

    figure_paths: list[str] = []
    for j, gene in enumerate(plotted):
        out_png1 = os.path.join(fig_dir, _figure_name(gene))
        _plot_spatial_gene(
            x_sp,
            y_sp,
            expr[:, j],
            title=f"SOMDE top gene (tissue): {gene}",
            out_png=out_png1,
            s=6.0,
        )
        figure_paths.append(out_png1)

    # NOT `svnum`: SomNode.run returns result[result.qval < 0.05].shape[0], a ROW count, so the
    # tied duplicates above made it exceed the number of genes tested (151 significant out of 150
    # tested, reported to the user as "100.7%").
    if gene_col and "qval" in result.columns:
        n_significant = len(distinct_significant_genes(result[gene_col], result["qval"] < 0.05))
    elif svnum is not None:
        n_significant = int(svnum)
    else:
        n_significant = len(top_gene_names)

    method = METHOD_NAME
    coord_why = ""
    if used_pixel_fallback:
        method = f"{METHOD_NAME} on full-resolution pixel coordinates (array_row/array_col constant)"
        coord_why = "array_col/array_row were constant; allow_pixel_coord_fallback=True permitted the pixel columns"

    out = WorkerOutput("somde", task="svg_identification")
    out.set_data(
        n_spots=n_spots_supplied,
        n_spots_used=int(adata_small.n_obs),
        n_spots_off_tissue_dropped=int(n_spots_off_tissue),
        n_spots_zero_total=n_zero_spots,
        n_genes=n_genes_supplied,
        n_genes_used=int(adata_small.n_vars),
        n_som_nodes=int(len(_ninfo)),
        som_grid=f"{grid_side}x{grid_side}",
    )
    out.add_output_files(
        {
            "result_csv": result_csv,
            "top_genes_csv": top_csv,
            "fig_dir": fig_dir,
            "figures": figure_paths,
        }
    )
    out.add_params(
        {
            "input_mode": input_mode,
            "som_dim": int(som_dim),
            "max_genes": int(max_genes),
            "min_counts": int(min_counts),
            "top_k_genes": int(top_k_genes),
            "allow_pixel_coord_fallback": bool(allow_pixel_coord_fallback),
            "coord_source": coord_source,
            "gene_selection": gene_info,
            "layer": layer,
            "expression_source": expression_source,
            "round_counts": bool(round_counts),
            "rounded_to_integers": bool(value_report["rounded"]),
            "n_non_integer_values": int(value_report["n_non_integer"]),
            "use_raw_counts": use_raw_counts,
            # The preflight's estimate: the dense table plus the n_nodes^2 kernels of SOMDE's test.
            "memory_estimate_bytes": int(need_bytes),
            "memory_estimate_gib": round(float(need_bytes) / float(1 << 30), 3),
        }
    )
    record_method(out, method, used_fallback=used_pixel_fallback, why=coord_why)
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    if counts_info is not None:
        # Non-integer X only gets this far rounded (round_counts=True): the rounding note says what happened,
        # and "normalised twice" would be wrong for counts stored as floats.
        record_expression_source(out, dict(counts_info, warning=None) if value_report["rounded"] else counts_info)
    if use_raw_counts and input_mode != "h5ad":
        record_ignored(
            out, "use_raw_counts", "input_mode='visium_10x' reads Space Ranger's counts h5, which has no adata.raw"
        )
    if layer and input_mode != "h5ad":
        record_ignored(out, "layer", "layer applies to input_mode='h5ad'; visium_10x reads Space Ranger's counts h5")
    elif round_counts and not value_report["rounded"]:
        record_ignored(
            out, "round_counts", f"every value of {expression_source} is already an integer; nothing was rounded"
        )
    record_ignored(
        out,
        "random_seed",
        f"SOMDE draws no random numbers (random_seed={int(random_seed)} was passed): SomNode starts from a "
        "homogeneous meshgrid codebook and batch-trains it, and the max_genes ranking is a deterministic sort "
        "(ties keep file order), so the seed cannot change the result",
    )
    gene_note = _gene_selection_note(gene_info)
    if gene_note:
        out.add_warning(gene_note.strip())
    spot_note = describe_reduction(
        "spots",
        n_spots_supplied,
        int(adata_small.n_obs),
        reason="leaving out the background spots flagged obs['in_tissue'] == 0",
    )
    rounding_note = (
        f" NOTE: {value_report['n_non_integer']} non-integer values of {expression_source} were rounded to integers "
        "before SOMDE ran (round_counts=True)."
        if value_report["rounded"]
        else ""
    )
    if rounding_note:
        out.add_warning(rounding_note.strip())
    empty_note = (
        f" NOTE: {n_zero_spots} of {int(adata_small.n_obs)} tested spots have zero counts over the "
        f"{int(adata_small.n_vars)} tested genes; SOMDE condensed them into their SOM nodes as zeros."
        if n_zero_spots
        else ""
    )
    if empty_note:
        out.add_warning(empty_note.strip())
    out.set_summary(
        n_significant=n_significant,
        top_genes=top_gene_names,
    )
    coord_note = (
        f" NOTE: array_col/array_row were constant, so SOMDE ran on {coord_source} (allow_pixel_coord_fallback=True)."
        if used_pixel_fallback
        else ""
    )
    out.set_analysis(
        build_svg_analysis(
            int(adata_small.n_vars),
            n_significant,
            top_gene_names,
            method_name="SOMDE",
            n_genes_renamed=n_genes_renamed,
        )
        + spot_note
        + gene_note
        + coord_note
        + rounding_note
        + empty_note
    )
    return out.to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="JSON string payload")
    args = ap.parse_args()

    try:
        cfg = json.loads(args.json)

        payload = run_somde(
            input_mode=str(cfg.get("input_mode", "visium_10x")),
            counts_h5=_text(cfg.get("counts_h5", "")),
            spatial_dir=_text(cfg.get("spatial_dir", "")),
            output_dir=_text(cfg.get("output_dir", "")),
            max_genes=int(cfg.get("max_genes", 2000)),
            min_counts=int(cfg.get("min_counts", 1)),
            som_dim=int(cfg.get("som_dim", 20)),
            top_k_genes=int(cfg.get("top_k_genes", 20)),
            random_seed=int(cfg.get("random_seed", 0)),
            h5ad_path=_text(cfg.get("h5ad_path", "")) or _text(cfg.get("data_path", "")),
            allow_pixel_coord_fallback=_as_bool(cfg.get("allow_pixel_coord_fallback", False)),
            layer=_text(cfg.get("layer", "")),
            round_counts=_as_bool(cfg.get("round_counts", False)),
            use_raw_counts=_as_bool(cfg.get("use_raw_counts", False)),
        )

        print(json.dumps(payload, indent=2, default=str))

    except Exception as e:
        WorkerOutput.emit_error("somde", str(e), task="svg_identification")
        sys.exit(1)


if __name__ == "__main__":
    main()
