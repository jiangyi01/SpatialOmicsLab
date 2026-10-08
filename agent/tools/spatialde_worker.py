#!/usr/bin/env python
"""
SpatialDE worker (runs inside the SpatialDE conda env).

The env ships SpatialDE 1.1.3 -- the Teichlab v1 method of Svensson, Teichmann & Stegle (2018).
``SpatialDE.anndata.spatialde_test`` is NaiveDE.stabilize (Anscombe variance stabilisation of
negative-binomial counts) + NaiveDE.regress_out + ``SpatialDE.base.run``, a Gaussian-process
likelihood-ratio test over a grid of ten squared-exponential lengthscales. It is not the PMBio
TensorFlow rewrite (``SpatialDE.test``), whatever an older comment here said; the payload names the
version that ran under ``params.method`` / ``params.spatialde_version``.

Key design constraints (SpatialOmicsLab MCP pattern):
- ALL logs/progress go to stderr (AEH's own progress prints are redirected there too)
- stdout is JSON-only (final result)
- Supports multiple input modes:
    1) visium_10x: counts_h5 + spatial_dir (10x Visium layout with spatial/)
    2) h5ad: AnnData with coordinates in obsm[spatial_key]
    3) matrix_with_coords: expression matrix + coords table
- Produces analysis figures:
    - svg_volcano.png
    - top gene spatial scatter plots (one per gene)

What the worker does, and does not do, to the input:
- Genes: keeps the top ``hvg_top_n`` by ``hvg_flavor`` -- the worker's own raw-count ``variance``
  ranking (the default), or scanpy's ``seurat_v3`` / ``cell_ranger``. ``seurat_v3`` needs
  scikit-misc, which the SpatialDE env lacks. ``cell_ranger`` bins genes by mean-expression
  percentile and scanpy fails ("Bin edges must be unique") when the lowest percentiles coincide --
  on a whole-transcriptome Visium sample, where 15% or more of the genes are zero in every spot.
  ``seurat`` is scanpy's log-data flavour: it applies expm1 first, and the matrix here is counts, so
  it is refused rather than run. A flavour that cannot run stops the run with the reason and the
  choices that differ from it, unless ``allow_hvg_fallback`` is True, in which case variance ranking
  runs and the payload says so (``params.hvg_method_used``, ``params.used_fallback``, a warning). No
  flavour is ever switched silently. The cut is reported in ``data``, ``params``, ``warnings`` and
  ``analysis``.
- Matrix: in h5ad mode with no ``layer``, ``worker_utils.choose_counts_matrix`` decides what is
  tested. ``use_raw_counts=True`` tests ``adata.raw.X`` (CELLxGENE exports keep the counts there and
  a processed X); a negative or non-finite X is refused and the message says whether ``adata.raw``
  holds counts. The choice is ``params.expression_source`` (``X``, ``raw.X`` or ``layers['<name>']``).
- Spots: never subsampled. ``base.dyn_de`` keeps every kernel's eigendecomposition -- ten dense
  n_spots x n_spots float64 matrices -- alive at once, so a preflight (worker_utils'
  ``available_memory_bytes``, page cache counted reclaimable) estimates that against the memory
  available and refuses, naming both numbers, when it cannot fit.
- Background: spots with ``obs['in_tissue'] == 0`` are left out before anything is computed and
  reported (``params.in_tissue_filter``, a warning, the analysis). CELLxGENE Visium exports carry
  every array spot, and on the library's samples 56-70% of them are glass with ambient counts; tested
  beside the tissue, the tissue/background contrast itself reads as a spatial pattern. The flag comes
  from ``obs`` (h5ad), the tissue-positions file (visium_10x) or an ``in_tissue`` column of the
  coordinates table (matrix_with_coords). ``data.n_spots`` is the count supplied and
  ``data.n_spots_used`` the count tested; the two differ only by those background spots.
- Values: every stored value is checked. NaN/inf or negative values are an error (they are not
  counts, and rounding cannot make them counts). Non-integer values are an error naming the count
  and ``round_counts``; with ``round_counts`` True they are rounded and the payload says so. The
  NB stabilisation models counts, so normalised input is never fitted as if it were counts.
- A spot with zero total counts is an error when ``regress_formula`` takes ``log(total_counts)``:
  NaiveDE's least-squares fit otherwise dies with "SVD did not converge". Beyond the background
  spots above, no spot is dropped.
- Densification is intrinsic to the method (NaiveDE calls ``.var()`` and ``np.log`` on arrays) and
  happens after the gene cut, so the dense matrix is n_spots x n_genes_used.
- Every output file is written to ``<name>.partial`` and moved into place, so a killed run never
  leaves a truncated table under the final name.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from SpatialDE.anndata import automatic_expression_histology, spatialde_test
from worker_utils import (
    TISSUE_POSITIONS_NAMES,
    WorkerOutput,
    available_memory_bytes,
    build_svg_analysis,
    choose_counts_matrix,
    describe_reduction,
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
    require_hvg_flavor,
    spatial_coords,
    unsupported_choice_msg,
)

# ``SpatialDE.base.run`` searches ten SE lengthscales (np.logspace(l_min, l_max, 10)) and
# ``dyn_de`` keeps the n x n float64 eigenvector matrix of every one of them in a list while the
# genes are fitted.
N_KERNELS = 10
# Beside the stored eigenvector matrices, the last kernel's own n x n matrices: the kernel K and
# gower_scaling_factor's three temporaries (P, KP, P*KP) -- or, a moment earlier, eigh's input copy
# and its 2n^2 dsyevd workspace. Either way four more n x n float64 matrices at the peak.
KERNEL_TRANSIENT_COPIES = 4
# Per stored expression value: the dense float32 X, the float64 ``stabilized`` and ``residual``
# layers spatialde_test adds, the float64 records frame it builds from ``residual``, and
# regress_out's fitted-value temporary.
BYTES_PER_EXPRESSION_VALUE = 4 + 8 + 8 + 8 + 8

# Values within this distance of an integer count as integers (float32 noise, not normalisation).
INTEGER_TOL = 1e-6

HVG_FLAVORS = ("seurat_v3", "seurat", "cell_ranger", "variance")
SIGNIFICANCE_ALPHA = 0.05


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _ensure_dir(p: str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _partial(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def _write_csv_atomic(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    tmp = _partial(path)
    df.to_csv(tmp, **kwargs)
    os.replace(str(tmp), str(path))


def _write_text_atomic(text: str, path: Path) -> None:
    tmp = _partial(path)
    tmp.write_text(text)
    os.replace(str(tmp), str(path))


def _savefig_atomic(fig, path: Path, dpi: int = 200) -> None:
    # The format is named explicitly because ``.partial`` is not an extension matplotlib knows.
    tmp = _partial(path)
    fig.savefig(str(tmp), dpi=dpi, format=path.suffix.lstrip(".") or "png")
    os.replace(str(tmp), str(path))


def _safe_sum_rows(X) -> np.ndarray:
    # Works for dense and sparse
    try:
        return np.asarray(X.sum(axis=1)).ravel()
    except Exception:
        return np.array([np.sum(row) for row in X])


def _densify_for_spatialde(adata: ad.AnnData) -> None:
    """
    SpatialDE/NaiveDE expects a dense matrix because it calls .var() on arrays.
    If adata.X is sparse, convert to dense float32 to avoid:
        AttributeError: var not found
    Intrinsic to the method, so it stays; it runs after the gene cut to keep the dense matrix small.
    """
    if sp.issparse(adata.X):
        eprint(f"[SpatialDE] adata.X is sparse ({type(adata.X)}); converting to dense float32 for SpatialDE.")
        adata.X = adata.X.toarray().astype(np.float32, copy=False)
    else:
        # Ensure numeric float for downstream calculations
        try:
            if getattr(adata.X, "dtype", None) is not None and adata.X.dtype.kind in ("i", "u"):
                adata.X = adata.X.astype(np.float32, copy=False)
        except Exception:
            pass


def _make_var_names_unique(adata: ad.AnnData) -> None:
    try:
        # Reports the count on stderr and under adata.uns, so main() can put it on the payload:
        # the ranked gene list IS this tool's answer, and a deduplicated symbol is one we invented.
        make_names_unique_and_report(adata)
    except Exception:
        pass


# ----------------------------------------------------------------------------- memory preflight


def kernel_memory_bytes(n_obs: int, n_vars: int = 0, n_kernels: int = N_KERNELS) -> int:
    """Peak bytes SpatialDE v1 needs for ``n_obs`` spots and ``n_vars`` (post-cut) genes.

    Dominated by the ``(n_kernels + KERNEL_TRANSIENT_COPIES)`` dense n x n float64 matrices that
    ``base.dyn_de`` holds at its peak; the expression term only matters when ``hvg_top_n=0`` on a
    whole-transcriptome input.
    """
    n = int(n_obs)
    kernels = n * n * 8 * (int(n_kernels) + KERNEL_TRANSIENT_COPIES)
    expression = n * int(n_vars) * BYTES_PER_EXPRESSION_VALUE
    return kernels + expression


def _gib(n_bytes) -> float:
    return float(n_bytes) / float(1 << 30)


def check_spot_budget(n_obs: int, n_vars: int = 0, available=None):
    """Refuse, naming both numbers, when the full-size kernels cannot fit. Never subsamples.

    Returns ``(needed_bytes, available_bytes_or_None)``. The old code drew a random 8000-spot
    subset above a fixed cap; that made the ranking depend on the seed and left the rest of the
    slide untested while the table read as if it covered the whole input.
    """
    n = int(n_obs)
    need = kernel_memory_bytes(n, n_vars)
    if available is None:
        available = available_memory_bytes()
    if available is not None and need > available:
        raise MemoryError(
            f"SpatialDE 1.1.3 keeps {N_KERNELS} dense {n}x{n} float64 kernels in memory at once "
            f"(base.dyn_de holds every eigendecomposition): {n} spots x {int(n_vars)} genes need about "
            f"{_gib(need):.1f} GiB, but about {_gib(available):.1f} GiB is available here (MemAvailable / "
            "room under the cgroup limit). This worker tests every spot it is given and never subsamples. The kernel "
            "term grows with the spot count alone and no parameter of this tool changes it; hvg_top_n "
            "sets only the smaller per-gene term. Run it where that much memory is free."
        )
    return need, available


# ----------------------------------------------------------------------------- gene prefilter


def _rank_by_variance(adata: ad.AnnData, n_top: int) -> ad.AnnData:
    """Keep the ``n_top`` genes with the largest raw-count variance. Sparse-aware; no densify."""
    if sp.issparse(adata.X):
        Xc = adata.X.tocsc().astype(np.float32)
        mean = np.asarray(Xc.mean(axis=0)).ravel()
        mean2 = np.asarray(Xc.multiply(Xc).mean(axis=0)).ravel()
        var_per_gene = mean2 - mean**2
    else:
        var_per_gene = np.asarray(adata.X).var(axis=0).ravel()
    top = np.sort(np.argsort(var_per_gene)[-int(n_top) :])
    return adata[:, top].copy()


class SeuratFlavourNeedsLogData(ValueError):
    """``hvg_flavor='seurat'`` asked of a counts matrix: refused before scanpy is called."""


#: Why scanpy's ``seurat`` flavour cannot rank this worker's input. The values reaching the prefilter
#: are the counts SpatialDE's stabilisation needs (non-integer input stops the run at the value audit
#: unless round_counts rounds it), and scanpy 1.9.8's ``seurat`` flavour runs ``expm1`` on the matrix
#: first: on the library's Visium counts a float32 count above ~88 becomes inf and the call dies with
#: "cannot specify integer `bins` when input data contains infinity"; below that it ranks exp(count).
SEURAT_ON_COUNTS = (
    "scanpy's 'seurat' flavour ranks log-normalised data -- it applies expm1 to the matrix before computing "
    "dispersions -- and this worker hands the prefilter raw counts (SpatialDE's variance stabilisation needs "
    "counts), so it would rank exp(count): any count above ~88 is infinite in float32 and scanpy then fails"
)


def _short(text: str, limit: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _zero_gene_count(adata: ad.AnnData) -> int:
    """Genes that are zero in every spot of ``adata`` (sparse-aware)."""
    X = adata.X
    if sp.issparse(X):
        return int((X.getnnz(axis=0) == 0).sum())
    return int((~np.asarray(X != 0).any(axis=0)).sum())


def _hvg_cause(flavor: str, exc: BaseException, adata: ad.AnnData | None = None) -> str:
    """Why a flavour could not run, in words that fit a counts input (no advice to use 'seurat')."""
    if flavor == "seurat_v3" and isinstance(exc, ImportError):
        return "it needs scikit-misc (import name skmisc), which is not installed in the SpatialDE env"
    if isinstance(exc, SeuratFlavourNeedsLogData):
        return SEURAT_ON_COUNTS
    if flavor == "cell_ranger" and "Bin edges must be unique" in str(exc):
        zeros = ""
        if adata is not None:
            try:
                zeros = f"{_zero_gene_count(adata)} of {adata.n_vars} genes are zero in every tested spot, so "
            except Exception:
                zeros = ""
        return (
            "scanpy's 'cell_ranger' flavour bins genes by mean expression at the 10th, 15th, ... 100th percentiles "
            f"and fails when two bin edges coincide; {zeros}the lowest percentiles fall on the same mean "
            f"({type(exc).__name__}: {_short(exc, 80)})"
        )
    return f"{type(exc).__name__}: {_short(exc)}"


def _hvg_unavailable(flavor: str, exc: BaseException, adata: ad.AnnData | None = None) -> Exception:
    """The error for a flavour that could not run. Substitutes nothing; offers only choices that differ.

    The old code caught this and ranked by raw-count variance while ``params.hvg_flavor`` and the
    analysis text kept saying ``seurat_v3``. In the SpatialDE env scikit-misc is absent, so that
    happened on every input with more than ``hvg_top_n`` genes. The message used to advise installing
    a package whatever had failed, and to recommend ``cell_ranger`` even when cell_ranger had failed.
    """
    remedies = []
    if isinstance(exc, ImportError):
        remedies.append("install scikit-misc in the SpatialDE env")
    if flavor != "variance":
        remedies.append("pass hvg_flavor='variance' (the worker's own raw-count variance ranking, the default)")
    remedies.append("pass hvg_top_n=0 to test every gene")
    remedies.append(
        "pass allow_hvg_fallback=True to let variance ranking stand in, reported as params.used_fallback=True"
    )
    msg = (
        f"hvg_flavor='{flavor}' could not run: {_hvg_cause(flavor, exc, adata)}. Nothing was substituted. "
        + "To proceed, "
        + ", or ".join(remedies)
        + "."
    )
    return ImportError(msg) if isinstance(exc, ImportError) else RuntimeError(msg)


def _raw_counts_hint(adata: ad.AnnData) -> str:
    """One sentence on ``adata.raw`` for a refusal: whether use_raw_counts=True would help."""
    raw = getattr(adata, "raw", None)
    if raw is None:
        return ""
    try:
        kind = expression_matrix_kind(raw.X)
    except Exception:
        return ""
    if kind == "counts":
        return " adata.raw holds raw counts: pass use_raw_counts=True (h5ad mode, no layer) to test them."
    return f" adata.raw is present but holds {kind.replace('_', ' ')} values, not counts."


# ----------------------------------------------------------------------------- input values


def audit_values(X, tol: float = INTEGER_TOL, chunk: int = 5000000) -> dict[str, Any]:
    """Count non-finite, negative and non-integer values over EVERY stored value.

    Sparse: the stored entries (``X.data``); dense: all entries, in chunks so the temporary never
    exceeds ``chunk`` floats. The old guard looked at ``flatten()[:1000]`` -- the first row, mostly
    zeros -- so whether a whole matrix was rounded was decided by one spot.
    """
    values = X.data if sp.issparse(X) else np.asarray(X).ravel()
    n_checked = int(values.size)
    n_non_finite = 0
    n_negative = 0
    n_non_integer = 0
    example = None
    for start in range(0, n_checked, int(chunk)):
        block = np.asarray(values[start : start + int(chunk)], dtype=np.float64)
        finite = np.isfinite(block)
        n_non_finite += int((~finite).sum())
        n_negative += int((block[finite] < 0).sum())
        vals = block[finite]
        off = np.abs(vals - np.rint(vals)) > tol
        k = int(off.sum())
        if k and example is None:
            example = float(vals[off][0])
        n_non_integer += k
    return {
        "n_checked": n_checked,
        "n_non_finite": n_non_finite,
        "n_negative": n_negative,
        "n_non_integer": n_non_integer,
        "example": example,
    }


def _check_total_counts_for_formula(adata: ad.AnnData, regress_formula: str) -> None:
    """A spot with zero counts makes ``np.log(total_counts)`` -inf, and NaiveDE.regress_out's
    least-squares fit then dies with ``LinAlgError: SVD did not converge`` -- a message that names
    neither the spot nor the formula."""
    if "total_counts" not in regress_formula or "log" not in regress_formula:
        return
    n_zero = int((np.asarray(adata.obs["total_counts"], dtype=float) <= 0).sum())
    if n_zero:
        raise ValueError(
            f"{n_zero} of {adata.n_obs} spots have zero total counts. regress_formula={regress_formula!r} "
            "takes the log of total_counts, which is -inf for them, and NaiveDE.regress_out's least-squares "
            "fit then fails ('SVD did not converge'). Remove empty spots before calling this tool, or pass a "
            "regress_formula that does not take log(total_counts). This worker leaves out only background "
            "spots flagged obs['in_tissue'] == 0 (already done when the flag is present); it does not drop "
            "unflagged spots on its own."
        )


def _spatialde_version() -> str:
    try:
        import importlib.metadata as md  # 3.8+

        return str(md.version("SpatialDE"))
    except Exception:
        pass
    try:
        import pkg_resources

        return str(pkg_resources.get_distribution("SpatialDE").version)
    except Exception:
        return "unknown"


def _load_visium_10x(counts_h5: str, spatial_dir: str) -> tuple[ad.AnnData, dict[str, Any]]:
    """
    counts_h5: .../filtered_feature_bc_matrix.h5
    spatial_dir: either the outs/ directory containing 'spatial/', or that 'spatial/' directly --
        both are ordinary Space Ranger spellings and find_tissue_positions searches for either.

    Uses tissue_positions_list.csv (or tissue_positions.csv) to attach x,y (pixel coords).
    """
    import scanpy as sc

    adata = sc.read_10x_h5(counts_h5)
    _make_var_names_unique(adata)

    tissue_pos = find_tissue_positions(spatial_dir)
    if tissue_pos is None:
        raise FileNotFoundError(
            f"Cannot find {' or '.join(TISSUE_POSITIONS_NAMES)} under {spatial_dir} or its spatial/ subdirectory"
        )
    # The image sits beside the spot table, so the directory to look in is the one the search
    # landed in -- not one rebuilt from the argument, which is right for only one of the two
    # spellings and put ".../spatial/spatial" in the not-found message for the other.
    spatial_path = Path(tissue_pos).parent

    pos = read_tissue_positions(tissue_pos).set_index("barcode")

    common = adata.obs_names.intersection(pos.index)
    if len(common) == 0:
        raise ValueError(id_mismatch_msg("barcodes", "counts_h5", adata.obs_names, "tissue_positions", pos.index))

    adata = adata[common].copy()
    pos = pos.loc[common]

    # SpatialDE expects numeric coords in obs columns (x,y)
    adata.obs["x"] = pos["pxl_col_in_fullres"].astype(float).values
    adata.obs["y"] = pos["pxl_row_in_fullres"].astype(float).values
    # The flag travels with the spot, so a raw_feature_bc_matrix (every array spot) has its
    # background left out by main() exactly as a CELLxGENE h5ad does; a filtered matrix is all 1.
    adata.obs["in_tissue"] = pos["in_tissue"].values
    adata.obs["total_counts"] = _safe_sum_rows(adata.X)

    meta = {
        "tissue_positions_path": str(tissue_pos),
        "has_hires": (spatial_path / "tissue_hires_image.png").exists(),
        "hires_path": str(spatial_path / "tissue_hires_image.png"),
    }
    return adata, meta


def _load_h5ad(h5ad_path: str, spatial_key: str, layer: str | None) -> tuple[ad.AnnData, dict[str, Any]]:
    import scanpy as sc

    adata = sc.read_h5ad(h5ad_path)
    _make_var_names_unique(adata)

    coords, _ = spatial_coords(adata, spatial_key, want=2, tool="SpatialDE")
    adata.obs["x"] = coords[:, 0]
    adata.obs["y"] = coords[:, 1]

    # Normalize layer argument: treat "" as None
    if layer is not None and str(layer).strip() == "":
        layer = None

    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Requested layer={layer} not found in adata.layers; available: {list(adata.layers.keys())}")
        adata.X = adata.layers[layer]

    adata.obs["total_counts"] = _safe_sum_rows(adata.X)

    meta = {"n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars), "spatial_key": spatial_key, "layer": layer}
    return adata, meta


def _load_matrix_with_coords(
    matrix_path: str,
    coords_path: str,
    orientation: str,
    id_col: str,
    x_col: str,
    y_col: str,
) -> tuple[ad.AnnData, dict[str, Any]]:
    coords = pd.read_csv(coords_path, sep=None, engine="python")
    for c in (id_col, x_col, y_col):
        if c not in coords.columns:
            raise KeyError(f"coords_path missing column '{c}'. Found: {list(coords.columns)}")
    coords = coords.set_index(id_col)

    mat = pd.read_csv(matrix_path, sep=None, engine="python", index_col=0)
    if orientation == "genes_by_cells":
        mat = mat.T
    elif orientation != "cells_by_genes":
        raise ValueError("matrix_orientation must be 'cells_by_genes' or 'genes_by_cells'.")

    common = mat.index.intersection(coords.index)
    if len(common) == 0:
        raise ValueError(id_mismatch_msg("row IDs", "expression matrix", mat.index, "coords", coords.index))

    mat = mat.loc[common]
    coords = coords.loc[common]

    adata = ad.AnnData(mat.values.astype(np.float32, copy=False))
    adata.obs_names = mat.index.astype(str)
    adata.var_names = mat.columns.astype(str)
    _make_var_names_unique(adata)

    adata.obs["x"] = coords[x_col].astype(float).values
    adata.obs["y"] = coords[y_col].astype(float).values
    if "in_tissue" in coords.columns:
        adata.obs["in_tissue"] = coords["in_tissue"].values
    adata.obs["total_counts"] = _safe_sum_rows(adata.X)

    meta = {"n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars)}
    return adata, meta


def _plot_volcano(res: pd.DataFrame, out_png: Path) -> None:
    df = res.copy()

    # Significance axis
    if "qval" in df.columns:
        df["neglog10_sig"] = -np.log10(np.maximum(df["qval"].values.astype(float), 1e-300))
        sig_label = "-log10(qval)"
    elif "pval" in df.columns:
        df["neglog10_sig"] = -np.log10(np.maximum(df["pval"].values.astype(float), 1e-300))
        sig_label = "-log10(pval)"
    else:
        raise KeyError("SpatialDE results missing qval/pval; cannot plot significance.")

    # Effect axis
    xcol = None
    for candidate in ["FSV", "LLR", "MSE", "log_likelihood"]:
        if candidate in df.columns:
            xcol = candidate
            break

    fig, ax = plt.subplots(figsize=(6, 4))
    if xcol is None:
        df = df.sort_values("neglog10_sig", ascending=False).reset_index(drop=True)
        ax.scatter(np.arange(df.shape[0]), df["neglog10_sig"].values, s=4)
        ax.set_xlabel("Gene rank")
    else:
        ax.scatter(df[xcol].values.astype(float), df["neglog10_sig"].values.astype(float), s=4)
        ax.set_xlabel(xcol)

    ax.set_ylabel(sig_label)
    ax.set_title("SpatialDE significance")
    fig.tight_layout()
    _savefig_atomic(fig, out_png)
    plt.close(fig)


def _gene_png_name(gene: str) -> str:
    # A path separator inside a gene symbol would otherwise make savefig fail on a directory that
    # does not exist, after the results table was already written.
    return gene.replace(os.sep, "_") + ".png"


def _plot_gene_spatial(adata: ad.AnnData, gene: str, out_png: Path, point_size: float = 6.0) -> None:
    if gene not in adata.var_names:
        return

    gidx = int(np.where(adata.var_names == gene)[0][0])
    vals = np.asarray(adata.X[:, gidx]).ravel()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    sca = ax.scatter(adata.obs["x"].values, adata.obs["y"].values, s=point_size, c=vals)
    ax.set_title(gene)
    fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _savefig_atomic(fig, out_png)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    payload = json.loads(args.json)
    outdir = _ensure_dir(payload["output_dir"])

    # SpatialDE's test is deterministic; the seed reaches numpy for AEH's random initialisation.
    random_seed = int(payload.get("random_seed", 0))
    np.random.seed(random_seed)

    mode = payload["input_mode"]

    # Cheap knobs are checked before the data are read, so a typo costs nothing.
    hvg_top_n = int(payload.get("hvg_top_n", 2000))
    hvg_flavor = str(payload.get("hvg_flavor", "variance"))
    allow_hvg_fallback = bool(payload.get("allow_hvg_fallback", False))
    round_counts = bool(payload.get("round_counts", False))
    use_raw_counts = bool(payload.get("use_raw_counts", False))
    top_k = int(payload.get("top_k_genes", 20))
    if hvg_flavor not in HVG_FLAVORS:
        raise ValueError(unsupported_choice_msg("hvg_flavor", hvg_flavor, list(HVG_FLAVORS)))
    if top_k < 0:
        # pandas' head(-k) keeps all but the last k rows: top_k_genes=-1 used to publish every tested
        # gene but one as the "top" list and draw a figure for each of them.
        raise ValueError(
            f"top_k_genes must be 0 or more (got {top_k}); it is how many of the most significant genes are "
            "listed in top_genes.tsv and plotted."
        )
    layer_arg = payload.get("layer")
    layer_named = mode == "h5ad" and layer_arg is not None and str(layer_arg).strip() != ""
    if use_raw_counts and layer_named:
        raise ValueError(
            f"use_raw_counts=True tests adata.raw, and layer={layer_arg!r} names a layer: they name two different "
            "matrices. Pass one of them (layer='' with use_raw_counts=True, or the layer with use_raw_counts=False)."
        )

    # ----------------------------
    # Load data according to mode
    # ----------------------------
    if mode == "visium_10x":
        adata, meta = _load_visium_10x(payload["counts_h5"], payload["spatial_dir"])
    elif mode == "h5ad":
        adata, meta = _load_h5ad(
            payload["h5ad_path"],
            payload.get("spatial_key", "spatial"),
            payload.get("layer"),
        )
    elif mode == "matrix_with_coords":
        adata, meta = _load_matrix_with_coords(
            payload["matrix_path"],
            payload["coords_path"],
            payload.get("matrix_orientation", "cells_by_genes"),
            payload.get("id_col", "barcode"),
            payload.get("x_col", "x"),
            payload.get("y_col", "y"),
        )
    else:
        raise ValueError(unsupported_choice_msg("input_mode", mode, ["visium_10x", "h5ad", "matrix_with_coords"]))

    eprint(f"[SpatialDE] Loaded: {adata.n_obs} obs x {adata.n_vars} vars (mode={mode})")

    # What the user handed us. The HVG prefilter below rebinds adata, so after it adata.n_vars is
    # the survivors and the input size is unrecoverable. Spots are never subsampled; n_spots_used is
    # measured on the object the test ran on, so the two spot keys are two measurements.
    n_genes_supplied = int(adata.n_vars)
    # Background spots (obs['in_tissue'] == 0) are glass, not tissue: CELLxGENE Visium exports carry
    # every array spot, most of them with ambient counts, so they pass every count guard and the
    # tissue/background contrast would be tested as a spatial pattern. Left out, and reported.
    n_spots_supplied = int(adata.n_obs)
    adata, _, n_spots_off_tissue = keep_in_tissue(adata, "spots")
    if n_spots_off_tissue:
        eprint(
            f"[SpatialDE] Left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0 "
            f"(background); {adata.n_obs} in-tissue spots are tested"
        )

    # Which matrix holds the counts. CELLxGENE exports keep a processed X beside the counts in
    # adata.raw; use_raw_counts=True tests adata.raw.X, and a negative / non-finite X is refused naming
    # it. Only an h5ad has adata.raw, and a named layer has already been put in X by _load_h5ad (the
    # value audit below covers it), so the other routes record their source directly.
    counts_info = None
    if mode == "h5ad" and not layer_named:
        adata, counts_info = choose_counts_matrix(adata, use_raw_counts)
        if counts_info["expression_source"] == "raw.X":
            # X's gene names were deduplicated at load; raw.var is another index, so it is deduplicated
            # (and counted) again, keeping the barcode count from the first pass.
            before = dict(adata.uns.get("identifier_renames") or {})
            make_names_unique_and_report(
                adata,
                into={"n_genes_renamed": 0, "n_cells_renamed": int(before.get("n_cells_renamed", 0))},
                axes=("var",),
            )
            # _load_h5ad summed X; the totals regress_formula reads must be those of the matrix tested.
            adata.obs["total_counts"] = _safe_sum_rows(adata.X)
            eprint(f"[SpatialDE] use_raw_counts=True: testing adata.raw.X ({adata.n_vars} genes)")
        n_genes_supplied = int(adata.n_vars)
    expression_source = (
        counts_info["expression_source"]
        if counts_info is not None
        else (f"layers['{str(layer_arg).strip()}']" if layer_named else "X")
    )

    # Ensure required columns exist
    if "x" not in adata.obs.columns or "y" not in adata.obs.columns:
        raise ValueError("Missing coordinates: adata.obs must contain numeric columns 'x' and 'y'.")
    if "total_counts" not in adata.obs.columns:
        adata.obs["total_counts"] = _safe_sum_rows(adata.X)

    regress_formula = payload.get("regress_formula", "~np.log(total_counts)")
    _check_total_counts_for_formula(adata, regress_formula)

    # ----------------------------
    # HVG prefilter: top hvg_top_n genes by hvg_flavor, BEFORE densify so the dense matrix is
    # n_spots x n_genes_used. Whole-transcriptome inputs are 17-37k genes; SpatialDE fits every
    # gene against every kernel, so the cut is a runtime choice and is reported as such.
    # ----------------------------
    hvg_method_used = "none"
    used_fallback = False
    fallback_why = ""
    if hvg_top_n > 0 and adata.n_vars > hvg_top_n:
        if hvg_flavor == "variance":
            adata = _rank_by_variance(adata, hvg_top_n)
            hvg_method_used = "variance"
        else:
            import scanpy as sc

            # Narrow guard around the flavour's own dependency and the scanpy call. A failure stops
            # the run naming what is missing, unless the caller allowed variance ranking to stand in.
            # 'seurat' is refused here, not run: it is scanpy's log-data flavour and this is counts.
            try:
                if hvg_flavor == "seurat":
                    raise SeuratFlavourNeedsLogData(SEURAT_ON_COUNTS)
                require_hvg_flavor(hvg_flavor)
                with contextlib.redirect_stdout(sys.stderr):
                    sc.pp.highly_variable_genes(adata, n_top_genes=hvg_top_n, flavor=hvg_flavor, subset=True)
                hvg_method_used = hvg_flavor
            except Exception as exc:
                if not allow_hvg_fallback:
                    raise _hvg_unavailable(hvg_flavor, exc, adata) from exc
                fallback_why = (
                    f"hvg_flavor='{hvg_flavor}' could not run ({_hvg_cause(hvg_flavor, exc, adata)}); "
                    "allow_hvg_fallback=True let raw-count variance ranking stand in"
                )
                eprint(f"[SpatialDE] {fallback_why}")
                adata = _rank_by_variance(adata, hvg_top_n)
                hvg_method_used = "variance"
                used_fallback = True
        eprint(f"[SpatialDE] HVG-filtered to {adata.n_vars} (top {hvg_top_n} by {hvg_method_used})")

    # ----------------------------
    # Spot budget. No cap and no subsample: the run either fits at full size or stops here.
    # ----------------------------
    need_bytes, avail_bytes = check_spot_budget(adata.n_obs, adata.n_vars)
    eprint(
        f"[SpatialDE] {adata.n_obs} spots: ~{_gib(need_bytes):.2f} GiB for {N_KERNELS} dense kernels"
        + (f" ({_gib(avail_bytes):.1f} GiB available)" if avail_bytes else "")
    )

    # ----------------------------
    # Values. NaiveDE.stabilize models NB counts; the audit covers every stored value, and the data
    # are changed only when the caller asked (round_counts=True) -- and then it is recorded.
    # ----------------------------
    audit = audit_values(adata.X)
    if audit["n_non_finite"] or audit["n_negative"]:
        raise ValueError(
            f"of {audit['n_checked']} stored expression values, {audit['n_non_finite']} are NaN/inf and "
            f"{audit['n_negative']} are negative. SpatialDE's NaiveDE stabilisation takes the log of raw "
            "counts, so these are not an input it can model, and round_counts cannot make them counts. Pass "
            "layer=<the raw-count layer> (h5ad mode) or a raw counts matrix." + _raw_counts_hint(adata)
        )
    n_non_integer = int(audit["n_non_integer"])
    rounded = False
    if n_non_integer:
        if not round_counts:
            raise ValueError(
                f"{n_non_integer} of {audit['n_checked']} stored expression values are not integers "
                f"(e.g. {audit['example']!r}). SpatialDE's NaiveDE variance stabilisation models "
                "negative-binomial counts, so it needs raw counts: pass layer=<the raw-count layer> (h5ad "
                "mode) or a counts matrix, or pass round_counts=True to round every value to the nearest "
                "integer -- the rounding is then recorded in params and warnings. Nothing was rounded."
                + _raw_counts_hint(adata)
            )
        if sp.issparse(adata.X):
            X = adata.X.copy()
            X.data = np.round(X.data)
            adata.X = X
        else:
            adata.X = np.round(np.asarray(adata.X)).astype(np.float32, copy=False)
        rounded = True
        eprint(f"[SpatialDE] round_counts=True: rounded {n_non_integer} non-integer values to integers")

    # Critical: densify for SpatialDE/NaiveDE
    _densify_for_spatialde(adata)

    # ----------------------------
    # Run SpatialDE (SVG test)
    # ----------------------------
    # stdout is this worker's JSON channel; anything the test or NaiveDE prints goes to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        sres = spatialde_test(adata, coord_columns=["x", "y"], regress_formula=regress_formula)

    res_csv = outdir / "spatialde_results.csv"
    _write_csv_atomic(sres, res_csv, index=False)

    # Sort and select top genes
    if "qval" in sres.columns:
        sres_sorted = sres.sort_values("qval", ascending=True)
    elif "pval" in sres.columns:
        sres_sorted = sres.sort_values("pval", ascending=True)
    else:
        sres_sorted = sres

    if "g" not in sres_sorted.columns:
        raise KeyError("SpatialDE results missing gene column 'g'.")

    top_genes = sres_sorted["g"].head(top_k).astype(str).tolist()
    top_tsv = outdir / "top_genes.tsv"
    _write_text_atomic("\n".join(top_genes) + "\n", top_tsv)

    # ----------------------------
    # Figures
    # ----------------------------
    volcano_png = outdir / "svg_volcano.png"
    _plot_volcano(sres, volcano_png)

    gene_dir = _ensure_dir(str(outdir / "top_genes_spatial"))
    for g in top_genes:
        _plot_gene_spatial(adata, g, gene_dir / _gene_png_name(g), point_size=6.0)

    # Count significant genes (qval < 0.05 if available)
    n_significant = 0
    sig_col = None
    if "qval" in sres.columns:
        sig_col = "qval"
    elif "pval" in sres.columns:
        sig_col = "pval"
    if sig_col is not None:
        n_significant = int((sres[sig_col] < SIGNIFICANCE_ALPHA).sum())

    out = WorkerOutput("spatialde", task="svg_identification")

    # ----------------------------
    # Optional AEH
    # ----------------------------
    aeh_outputs: dict[str, Any] = {}
    aeh_params: dict[str, Any] = {}
    run_aeh = bool(payload.get("run_aeh", False))
    if run_aeh:
        C = int(payload.get("aeh_C", 8))
        l = float(payload.get("aeh_l", 1000.0))
        # automatic_expression_histology takes "the significant subset of results" (its own
        # docstring; the README passes results.query('qval < 0.05')). The old code handed it every
        # tested gene, so the C patterns were fitted to noise genes as much as to SVGs.
        sig_res = sres[sres[sig_col] < SIGNIFICANCE_ALPHA] if sig_col is not None else sres
        aeh_params = {"aeh_C": C, "aeh_l": l, "aeh_n_genes": int(sig_res.shape[0])}
        if sig_res.shape[0] == 0:
            out.add_warning(
                f"run_aeh=True, but no gene passed {sig_col} < {SIGNIFICANCE_ALPHA}, so there was nothing for "
                "automatic expression histology to decompose; AEH was skipped and wrote no files."
            )
        else:
            # spatial_patterns prints its ELBO trace with print(); stdout is this worker's JSON channel.
            with contextlib.redirect_stdout(sys.stderr):
                histology_results, patterns = automatic_expression_histology(
                    adata,
                    sig_res,
                    C=C,
                    l=l,
                    coord_columns=["x", "y"],
                    layer="residual",
                    verbosity=1,
                )

            aeh_res_csv = outdir / "aeh_histology_results.csv"
            _write_csv_atomic(histology_results, aeh_res_csv, index=False)

            # SpatialDE 1.1.3 returns the patterns as a spots x C DataFrame, which has no .write(), so
            # every run has landed in the pickle branch; that file name is kept. A DataFrame is also
            # written as CSV (aeh_patterns_csv), which reads back without this env's pandas.
            if hasattr(patterns, "write"):
                aeh_patterns_path = outdir / "aeh_patterns.h5ad"
                tmp = _partial(aeh_patterns_path)
                patterns.write(tmp)
                os.replace(str(tmp), str(aeh_patterns_path))
            else:
                import pickle

                aeh_patterns_path = outdir / "aeh_patterns.pkl"
                tmp = _partial(aeh_patterns_path)
                with open(tmp, "wb") as f:
                    pickle.dump(patterns, f)
                os.replace(str(tmp), str(aeh_patterns_path))

            aeh_outputs = {
                "aeh_histology_results_csv": str(aeh_res_csv),
                "aeh_patterns": str(aeh_patterns_path),
            }
            if isinstance(patterns, pd.DataFrame):
                aeh_patterns_csv = outdir / "aeh_patterns.csv"
                _write_csv_atomic(patterns, aeh_patterns_csv, index=True)
                aeh_outputs["aeh_patterns_csv"] = str(aeh_patterns_csv)

    # ----------------------------
    # JSON-only stdout
    # ----------------------------
    version = _spatialde_version()
    method = f"SpatialDE {version} (v1 Gaussian-process likelihood-ratio test: SpatialDE.anndata.spatialde_test)"
    if hvg_method_used != "none":
        method += f"; gene prefilter: top-{hvg_top_n} by {hvg_method_used}"
        if used_fallback:
            method += f" (stand-in for hvg_flavor='{hvg_flavor}')"

    out.set_data(
        n_spots=n_spots_supplied,
        n_spots_used=int(adata.n_obs),
        n_spots_off_tissue_dropped=int(n_spots_off_tissue),
        n_genes=n_genes_supplied,
        n_genes_used=int(adata.n_vars),
    )
    out.add_output_files(
        {
            "results_csv": str(res_csv),
            "top_genes_tsv": str(top_tsv),
            "volcano_png": str(volcano_png),
            "top_genes_spatial_dir": str(gene_dir),
            **aeh_outputs,
        }
    )
    out.add_params(
        {
            "input_mode": mode,
            "regress_formula": regress_formula,
            "top_k_genes": top_k,
            "run_aeh": run_aeh,
            # The knobs that decide which genes survived. Without them in the payload the run
            # cannot be repeated, let alone widened. hvg_flavor is what was asked for;
            # hvg_method_used is the ranking that ran ("none" when nothing was cut).
            "hvg_top_n": hvg_top_n,
            "hvg_flavor": hvg_flavor,
            "hvg_method_used": hvg_method_used,
            "allow_hvg_fallback": allow_hvg_fallback,
            "random_seed": random_seed,
            "round_counts": round_counts,
            "rounded_to_integers": rounded,
            "n_non_integer_values": n_non_integer,
            "use_raw_counts": use_raw_counts,
            "expression_source": expression_source,
            "spatialde_version": version,
            "kernel_memory_estimate_gib": round(_gib(need_bytes), 3),
            **aeh_params,
        }
    )
    record_method(out, method, used_fallback=used_fallback, why=fallback_why)
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    if counts_info is not None:
        # A non-integer X only gets this far rounded (round_counts=True); the rounding warning below says
        # what happened to it, and "normalised twice" would be wrong for counts stored as floats.
        record_expression_source(out, dict(counts_info, warning=None) if rounded else counts_info)
    if use_raw_counts and mode != "h5ad":
        record_ignored(
            out,
            "use_raw_counts",
            f"input_mode='{mode}' reads a count matrix that has no adata.raw; the matrix supplied was tested",
        )
    if hvg_method_used == "none":
        record_ignored(
            out,
            "hvg_flavor",
            f"hvg_top_n={hvg_top_n} disables the gene prefilter"
            if hvg_top_n <= 0
            else f"the input has {n_genes_supplied} genes, not more than hvg_top_n={hvg_top_n}, so no gene was cut",
        )
    if not run_aeh:
        record_ignored(
            out,
            "random_seed",
            "run_aeh is False, and nothing else in this run draws a random number (the SpatialDE test, the NB "
            "stabilisation and every hvg_flavor are deterministic)",
        )
    out.set_summary(
        n_significant=n_significant,
        top_genes=top_genes,
    )
    gene_note = describe_reduction(
        "genes",
        n_genes_supplied,
        int(adata.n_vars),
        reason=f"the top-{hvg_top_n} highly-variable-gene prefilter ({hvg_method_used})",
    )
    fallback_note = (
        f" NOTE: hvg_flavor='{hvg_flavor}' could not run, so raw-count variance ranking chose the genes "
        "(allow_hvg_fallback=True)."
        if used_fallback
        else ""
    )
    rounding_note = (
        f" NOTE: {n_non_integer} non-integer expression values were rounded to integers before the test "
        "(round_counts=True)."
        if rounded
        else ""
    )
    for note in (gene_note, rounding_note):
        if note:
            out.add_warnings(note.strip())
    out.set_analysis(
        build_svg_analysis(
            int(adata.n_vars),
            n_significant,
            top_genes,
            method_name=f"SpatialDE {version}",
            n_genes_renamed=int(adata.uns.get("identifier_renames", {}).get("n_genes_renamed", 0)),
        )
        # The background cut is warned by record_in_tissue; here it reaches the prose.
        + describe_reduction(
            "spots",
            n_spots_supplied,
            int(adata.n_obs),
            reason="leaving out the background spots flagged obs['in_tissue'] == 0",
        )
        + gene_note
        + fallback_note
        + rounding_note
    )
    out.add_extra("meta", meta)
    out.emit()


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        eprint(f"[SpatialDE] ERROR: {ex}")
        WorkerOutput.emit_error("spatialde", str(ex), task="svg_identification")
        sys.exit(1)
