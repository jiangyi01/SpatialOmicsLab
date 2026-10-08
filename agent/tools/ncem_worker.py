#!/usr/bin/env python3
"""
NCEM-style neighbourhood regression worker for cell communication analysis.

Runs inside /opt/conda/envs/ncem conda env.

WHAT RUNS HERE, HONESTLY. The ``ncem`` package (theislab/ncem) is installed in this env but is not
imported, and none of its models -- the linear, interaction, graph-neural-network or variational
estimators -- is run. What runs is a closed-form linear model in the spirit of NCEM's linear
baseline, written here:

1. every spot's ``k`` nearest spatial neighbours are found with a KD-tree (the spot itself is
   excluded, also when another spot sits on the same coordinates);
2. each spot's neighbourhood composition is the fraction of each cell type among those neighbours;
3. counts are library-size normalised (1e4) and log1p-transformed, and the Seurat-flavour highly
   variable genes are kept (``n_hvg``);
4. ``sklearn.linear_model.Ridge(alpha=1.0)`` regresses every kept gene on
   ``[one-hot own cell type | neighbourhood composition]``. The neighbourhood block of the
   coefficients is the gene x cell-type "communication matrix".

Nothing is trained by gradient descent, so ``max_epochs`` and ``learning_rate`` have no effect;
nothing random runs, so ``seed`` has none either. All three stay accepted (removing them would break
callers) and are listed under ``params.ignored`` with a warning. This is the tool's only
implementation, not a fallback, so ``params.used_fallback`` is False and ``params.method`` names it.

Because the neighbour fractions of a spot sum to 1, the neighbourhood effects of a gene are only
identified relative to each other: the ridge solution centres them across cell types, so a
communication-matrix entry is the effect of that neighbour type *relative to the average neighbour*.

INPUT HANDLING. Spots flagged ``obs['in_tissue'] == 0`` (Space Ranger / CELLxGENE exports carry every
array spot) are left out before anything is computed: they are background, not cells, and they would
otherwise be regressed and counted as neighbours (``worker_utils.keep_in_tissue``; ``params.in_tissue_filter``).
A spot with no label in ``obs[annotation_key]`` stops the run with the count unless ``--drop-unlabeled`` is
given, in which case those spots are left out and counted. Every exclusion is reported in ``data``,
``params`` and the analysis text; nothing is subsampled.

The pipeline normalises X as counts, so the matrix follows ``worker_utils.choose_counts_matrix``: X by
default, ``adata.raw.X`` with ``--use-raw-counts``. An X with negative or non-finite values (scaled data,
which normalisation + log1p turns into NaN) is refused before anything is normalised, with the count, X's
smallest value and -- when ``adata.raw`` exists -- its shape and the knob; X is never swapped for
``adata.raw`` unless asked. A non-negative non-integer X runs, with a warning that it was transformed
twice (``params.expression_source``, ``params.x_matrix_kind``).

MEMORY. The regression targets are a dense spots x HVG matrix (sklearn's Ridge takes dense targets
and makes one float64 working copy of them). That is intrinsic to the model as implemented, so it is
kept; before it is allocated the worker estimates its bytes against the memory available
(``worker_utils.available_memory_bytes``: MemAvailable, or the room left under the cgroup memory limit
when that is smaller) and refuses with the numbers. The in-sample R-squared is computed in row blocks,
so no second dense matrix is built.

SECTIONS. The neighbourhood is two-dimensional by design, so ``--dims 3`` is refused, and coordinates are
read through ``worker_utils.spatial_frame``: a file holding two or more sections is refused unless
``--section-key`` names the column. With it the model is fitted per section (``worker_utils.per_section``;
each section its own normalisation, HVGs and fit): the four files per section in ``section_<label>/`` and a
long ncem_communication_strength.csv at the top level with a leading ``section`` column; ``params.mode`` is
``per-section-2d`` (``2d`` for one plane).

Input: spatial h5ad with raw counts in X, coordinates in obsm[spatial_key], labels in obs[annotation_key]
Output: communication matrix, per-type effect strength, neighbourhood composition, top genes per type

All logs go to stderr; stdout is JSON-only (final result).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    choose_counts_matrix,
    describe_reduction,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    per_section,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    spatial_coords,
    spatial_frame,
)
from worker_utils import drop_unlabeled as split_unlabeled

#: The ridge penalty. Fixed: no parameter reaches it, and the payload reports it as ``ridge_alpha``.
RIDGE_ALPHA = 1.0

#: What ``params.method`` says ran. The ncem package is installed but never imported.
METHOD_NAME = (
    "NCEM-style linear model: sklearn Ridge(alpha=1.0) of log-normalised HVG expression on "
    "[one-hot own cell type | cell-type fractions among the k nearest spatial neighbours]; "
    "the ncem package is not run"
)
#: The short form the analysis sentence leads with.
METHOD_SHORT = "NCEM-style linear model (sklearn Ridge on neighbourhood composition; the ncem package was not run)"

#: The refusal for ``dims=3``: the neighbourhood here is two-dimensional by design.
TWO_D_ONLY = "NCEM builds its neighbourhood in two dimensions; run per section with `dims=2, section_key=<column>`."

#: Why the training knobs the portal still accepts have no effect.
_NO_TRAINING = (
    "the model is a closed-form sklearn Ridge solve -- nothing is trained by gradient descent, so there "
    "are no epochs and no learning rate"
)
_NO_RANDOMNESS = (
    "nothing in this method is random (KD-tree neighbours, Seurat HVG ranking and the closed-form "
    "Ridge solve are deterministic), so the seed cannot change the result"
)

#: Bytes per entry of the dense spots x HVG targets at the fit's peak: the float32 matrix itself plus
#: the float64 centred copy sklearn's Ridge makes of it (measured with tracemalloc, sklearn 1.6.1).
_DENSE_BYTES_PER_ENTRY = 4 + 8

#: Rows per block when the in-sample R-squared is accumulated.
_R2_BLOCK_ROWS = 20000


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def _ensure_dir(p: str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _memory_budget_bytes():
    """Memory this run can still allocate, in bytes, or None when nothing can be read.

    ``worker_utils.available_memory_bytes``: the smaller of ``MemAvailable`` and the room left under a
    cgroup memory limit (the page cache counted as reclaimable). This used to read ``MemAvailable``
    alone, which inside a memory-limited container reports the *host*, so the refusal below never fired
    and an oversize fit was OOM-killed with no message. Kept as a seam for tests.
    """
    budget = available_memory_bytes()
    return None if budget is None else int(budget)


def _dense_target_bytes(n_spots: int, n_genes: int) -> int:
    """Peak bytes of the dense regression targets during the Ridge fit."""
    return int(n_spots) * int(n_genes) * _DENSE_BYTES_PER_ENTRY


def _check_dense_targets_fit(n_spots: int, n_genes: int) -> int:
    """Refuse, with the numbers, before densifying targets the machine cannot hold."""
    need = _dense_target_bytes(n_spots, n_genes)
    budget = _memory_budget_bytes()
    if budget is not None and need > budget:
        raise MemoryError(
            f"the Ridge regression targets are a dense {n_spots} spots x {n_genes} genes matrix: "
            f"~{need / 1e9:.1f} GB (float32 targets + sklearn's float64 working copy) but ~{budget / 1e9:.1f} GB "
            "is available (MemAvailable, or the room left under the cgroup memory limit when that is smaller). "
            f"Lower n_hvg (now {n_genes} genes) or run on a machine with more memory; the spots are not "
            "subsampled."
        )
    return need


def _safe_dense(X) -> np.ndarray:
    """Dense float32 view of X, without a second copy when X is already dense float32."""
    try:
        import scipy.sparse as sps

        if sps.issparse(X):
            return np.asarray(X.toarray(), dtype=np.float32)
    except ImportError:
        pass
    return np.asarray(X, dtype=np.float32)


def _sorted_types(values) -> list:
    """Distinct labels in a stable order: natural order when comparable, else by their text."""
    uniq = list(pd.unique(np.asarray(values, dtype=object)))
    try:
        return sorted(uniq)
    except TypeError:
        return sorted(uniq, key=str)


def _neighbour_indices(coords: np.ndarray, n_neighbors: int) -> np.ndarray:
    """``(n_spots, n_neighbors)`` indices of each spot's nearest neighbours, the spot itself excluded.

    The KD-tree is asked for ``n_neighbors + 1``. The spot itself is normally the first hit, but when
    another spot shares its coordinates the tie can put that spot first; dropping column 0 blindly
    would then keep the spot as its own neighbour and lose the duplicate. So the self hit is removed
    wherever it is, and the farthest hit is removed from the (rare) rows where self was not returned.
    """
    from scipy.spatial import cKDTree

    n = int(coords.shape[0])
    tree = cKDTree(coords)
    _distances, indices = tree.query(coords, k=n_neighbors + 1)
    indices = np.asarray(indices).reshape(n, n_neighbors + 1)
    is_self = indices == np.arange(n)[:, None]
    keep = ~is_self
    no_self = ~is_self.any(axis=1)
    keep[no_self, -1] = False
    return indices[keep].reshape(n, n_neighbors)


def _neighbourhood_composition(codes: np.ndarray, neighbour_idx: np.ndarray, n_types: int) -> np.ndarray:
    """Fraction of each cell type among every spot's neighbours, float32, rows summing to 1."""
    n, k = neighbour_idx.shape
    counts = np.zeros((n, n_types), dtype=np.float32)
    np.add.at(counts, (np.repeat(np.arange(n), k), codes[neighbour_idx].ravel()), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


def _pooled_r_squared(model, X: np.ndarray, Y: np.ndarray, block_rows: int = _R2_BLOCK_ROWS) -> float:
    """In-sample R-squared pooled over all genes, accumulated in row blocks (no dense prediction matrix)."""
    n = Y.shape[0]
    col_mean = Y.mean(axis=0, dtype=np.float64)
    ss_res = 0.0
    ss_tot = 0.0
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        y = Y[start:stop].astype(np.float64)
        ss_res += float(np.sum((y - model.predict(X[start:stop])) ** 2))
        ss_tot += float(np.sum((y - col_mean) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _is_sparse(X) -> bool:
    return hasattr(X, "data") and not isinstance(X, np.ndarray)


def _smallest_value(X) -> float:
    """Smallest value of X as supplied (NaN ignored; a sparse matrix's implicit zeros count)."""
    sparse = _is_sparse(X)
    data = np.asarray(X.data if sparse else X)
    smallest = float(np.nanmin(data)) if data.size else float("nan")
    if sparse and data.size < int(X.shape[0]) * int(X.shape[1]):
        smallest = 0.0 if np.isnan(smallest) else min(smallest, 0.0)
    return smallest


def _count_not_counts(X, block: int = 10000000) -> tuple:
    """``(n_bad, n_values, noun)``: negative or non-finite values among the stored values of a sparse X or
    every entry of a dense one. Counted in blocks, so no full-size boolean mask is allocated."""

    def _bad(values) -> int:
        values = np.asarray(values, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            return int((~np.isfinite(values) | (values < 0)).sum())

    if _is_sparse(X):
        values = np.asarray(X.data).reshape(-1)
        return sum(_bad(values[i : i + block]) for i in range(0, values.size, block)), int(values.size), "stored values"
    dense = np.asarray(X)
    rows = max(1, block // max(1, int(dense.shape[1])))
    return sum(_bad(dense[i : i + rows]) for i in range(0, dense.shape[0], rows)), int(dense.size), "values"


def _choose_counts(adata, use_raw_counts: bool):
    """``(adata, info)``: the matrix normalise + log1p runs on, by ``worker_utils.choose_counts_matrix``.

    X by default; ``adata.raw.X`` with ``use_raw_counts`` (refused when there is no ``adata.raw`` or it is
    not counts). A negative or non-finite X is refused before anything is normalised: library-size
    normalisation + log1p turns it into NaN/inf, and scanpy's Seurat HVG step then died with
    ``KeyError: "'[nan] not in index'"`` (Skin FaceTemple DonorB, z-scored X, smallest value -8.5). The
    refusal carries the count, X's smallest value and, when ``adata.raw`` exists, its shape and the knob
    that reads it; X is never swapped for ``adata.raw`` unless ``use_raw_counts`` asks for it. A
    non-negative non-integer X (log-normalised) runs, with a warning that it was transformed twice.
    """
    try:
        adata, info = choose_counts_matrix(adata, use_raw_counts)
    except ValueError as exc:
        if use_raw_counts:
            raise
        n_bad, n_values, noun = _count_not_counts(adata.X)
        msg = (
            f"{exc} {n_bad} of the {n_values} {noun} of X are negative, NaN or infinite (the smallest value in X "
            f"as supplied is {_smallest_value(adata.X):.6g}); library-size normalisation and log1p would turn "
            "them into NaN or infinite values, which the Seurat highly-variable-gene step cannot rank."
        )
        raw = getattr(adata, "raw", None)
        if raw is not None:
            msg += (
                f" adata.raw holds a {raw.shape[0]} x {raw.shape[1]} matrix; this worker does not substitute it "
                "for X on its own -- use_raw_counts=True runs on it."
            )
        raise ValueError(msg) from exc
    if info.get("warning"):
        info = dict(info)
        info["warning"] = "X does not hold non-negative integer counts. " + info["warning"]
    return adata, info


def _atomic_to_csv(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    """Write a CSV through ``<path>.partial`` and ``os.replace`` so a crash never leaves half a file."""
    partial = path.with_name(path.name + ".partial")
    frame.to_csv(partial, **kwargs)
    os.replace(partial, path)


def section_dir_name(label) -> str:
    """``section_<label>`` with every character a path cannot safely carry replaced by ``_``."""
    return "section_" + re.sub(r"[^A-Za-z0-9._-]+", "_", str(label))


def _section_dirs(labels) -> dict:
    """``{label: folder name}``, refusing two labels that would share one folder."""
    names: dict = {}
    for label in labels:
        name = section_dir_name(label)
        clash = [other for other, used in names.items() if used == name]
        if clash:
            raise ValueError(
                f"NCEM: sections '{clash[0]}' and '{label}' would both be written to the folder {name}/; rename "
                "one of them in the section column."
            )
        names[label] = name
    return names


def _write_long_table(frame: pd.DataFrame, path: Path) -> None:
    """A per-section table (leading ``section`` column) at the top level, written atomically."""
    _atomic_to_csv(frame, path, index=False)


def _fit_one(
    adata,
    spatial_key: str,
    annotation_key: str,
    n_neighbors: int,
    n_hvg: int,
    where: str = "",
    random_seed: int = 0,
) -> dict:
    """Fit the neighbourhood Ridge model on one plane; ``run_ncem`` writes its four tables.

    ``adata`` is modified (normalised in place); ``where`` prefixes messages with the section they are
    about (empty for a whole-file run). Returns what the payload reports.
    """
    import scanpy as sc

    n_spots = int(adata.n_obs)
    if n_neighbors >= n_spots:
        raise ValueError(
            f"{where}n_neighbors={n_neighbors} needs at least {n_neighbors + 1} spots and {n_spots} are left to "
            "analyse. Lower n_neighbors."
        )

    coords, _ = spatial_coords(adata, spatial_key, want=2, tool="NCEM")
    labels = np.asarray(adata.obs[annotation_key].to_numpy(), dtype=object)
    unique_types = _sorted_types(labels)
    n_types = len(unique_types)
    eprint(f"[NCEM] {where}Found {n_types} cell types: {unique_types}")
    if n_types < 2:
        raise ValueError(
            f"{where}obs['{annotation_key}'] holds {n_types} cell type ({unique_types}) across the {n_spots} "
            "analysed spots. A neighbourhood composition over one class is constant, so no neighbour effect can "
            "be estimated; pass an annotation_key with at least two cell types."
        )

    # ---- Preprocessing ----
    eprint(f"[NCEM] {where}Preprocessing ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_hvg_actual = min(int(n_hvg), adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg_actual)
    hvg_mask = adata.var["highly_variable"].to_numpy()
    gene_names = adata.var_names[hvg_mask].tolist()
    dense_bytes = _check_dense_targets_fit(n_spots, len(gene_names))
    expression = _safe_dense(adata.X[:, hvg_mask])
    eprint(f"[NCEM] {where}Using {len(gene_names)} HVGs for analysis")

    # ---- Build spatial graph ----
    eprint(f"[NCEM] {where}Building spatial graph with {n_neighbors} neighbors ...")
    neighbor_indices = _neighbour_indices(coords, n_neighbors)

    # ---- Build neighborhood composition matrix ----
    # For each cell, compute fraction of each cell type among neighbors
    eprint(f"[NCEM] {where}Computing neighborhood composition ...")
    type_to_idx = {ct: i for i, ct in enumerate(unique_types)}
    codes = np.fromiter((type_to_idx[v] for v in labels), dtype=np.int64, count=n_spots)
    nhood_composition = _neighbourhood_composition(codes, neighbor_indices, n_types)

    nhood_df = pd.DataFrame(
        nhood_composition,
        index=adata.obs_names,
        columns=unique_types,
    )

    # ---- Fit the neighbourhood Ridge model ----
    # Linear model: expression ~ cell_type + neighborhood_composition
    eprint(f"[NCEM] {where}Fitting linear model (sklearn Ridge, alpha={RIDGE_ALPHA}) ...")
    ct_onehot = np.zeros((n_spots, n_types), dtype=np.float64)
    ct_onehot[np.arange(n_spots), codes] = 1.0

    # Design matrix: [cell_type_onehot | neighborhood_composition]
    X = np.hstack([ct_onehot, nhood_composition])
    Y = expression

    from sklearn.linear_model import Ridge

    model = Ridge(alpha=RIDGE_ALPHA, random_state=random_seed)
    model.fit(X, Y)
    eprint(f"[NCEM] {where}Model fitting complete.")

    # ---- Extract communication effects ----
    # The coefficients for the neighborhood composition block capture
    # how each neighboring cell type shifts each gene's expression
    coefs = np.atleast_2d(model.coef_)  # shape: (n_genes, n_types + n_types)
    nhood_coefs = coefs[:, n_types:]  # neighborhood composition effects

    # Communication effect matrix: cell_type -> gene effects via neighborhood
    comm_matrix = pd.DataFrame(
        nhood_coefs,
        index=gene_names,
        columns=unique_types,
    )

    # Score each neighbour type by the mean magnitude of its effects across genes
    comm_strength = np.abs(nhood_coefs).mean(axis=0)
    comm_strength_df = pd.DataFrame(
        {
            "cell_type": unique_types,
            "mean_effect_strength": comm_strength,
        }
    ).sort_values("mean_effect_strength", ascending=False)

    # In-sample R-squared (own type + neighbourhood), pooled over genes
    r_squared = _pooled_r_squared(model, X, Y)

    # Top genes per cell type (largest neighborhood effects)
    top_genes_per_type: dict[str, list[str]] = {}
    for ct_idx, ct in enumerate(unique_types):
        gene_effects = np.abs(nhood_coefs[:, ct_idx])
        top_idx = np.argsort(-gene_effects)[:10]
        top_genes_per_type[ct] = [gene_names[i] for i in top_idx]

    top_genes_data = []
    for ct, genes in top_genes_per_type.items():
        for rank, g in enumerate(genes, 1):
            top_genes_data.append({"cell_type": ct, "rank": rank, "gene": g})

    return {
        "n_spots": n_spots,
        "n_genes": int(adata.n_vars),
        "n_genes_used": int(len(gene_names)),
        "n_hvg_actual": int(n_hvg_actual),
        "unique_types": unique_types,
        "n_types": n_types,
        "dense_bytes": int(dense_bytes),
        "r_squared": float(r_squared),
        "strength": comm_strength_df,
        "top_genes_per_type": top_genes_per_type,
        "top_communicators": comm_strength_df.head(5)["cell_type"].tolist(),
        # The four tables, written by run_ncem's _write_plane into this plane's folder.
        "matrix": comm_matrix,
        "composition": nhood_df,
        "top_genes": pd.DataFrame(top_genes_data, columns=["cell_type", "rank", "gene"]),
    }


def run_ncem(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    annotation_key: str = "cell_type",
    n_neighbors: int = 10,
    max_epochs: int = 50,
    learning_rate: float = 1e-3,
    n_hvg: int = 2000,
    model_type: str = "linear",
    random_seed: int = 0,
    drop_unlabeled: bool = False,
    use_raw_counts: bool = False,
    dims: int = 2,
    section_key: str | None = None,
) -> dict[str, Any]:
    """Fit the NCEM-style neighbourhood Ridge model (see the module docstring for what runs).

    ``max_epochs``, ``learning_rate`` and ``random_seed`` are accepted and reported as ignored.
    ``drop_unlabeled`` leaves out spots with no label in ``obs[annotation_key]`` instead of refusing.
    ``use_raw_counts`` normalises ``adata.raw.X`` instead of X (see ``_choose_counts``).

    Two-dimensional by design: ``dims=3`` is refused (:data:`TWO_D_ONLY`), and a file holding several
    sections is refused unless ``section_key`` names them. With ``section_key`` the model is fitted per
    section -- one ``section_<label>/`` folder of the four files per section (the gene x type matrix is
    per section only), and a long ``ncem_communication_strength.csv`` at the top level with a leading
    ``section`` column.
    """

    import scanpy as sc

    if int(dims) == 3:
        raise ValueError(TWO_D_ONLY)
    if int(dims) != 2:
        raise ValueError(f"NCEM: dims must be 2 (or 3, which is refused), not {dims}.")
    section_key = section_key or None
    if model_type != "linear":
        raise ValueError(f"model_type={model_type!r} is not available; this worker implements only 'linear'.")
    if int(n_neighbors) < 1:
        raise ValueError(f"n_neighbors={n_neighbors} must be at least 1.")
    if int(n_hvg) < 1:
        raise ValueError(f"n_hvg={n_hvg} must be at least 1.")
    n_neighbors = int(n_neighbors)

    _ensure_dir(output_dir)
    np.random.seed(random_seed)
    outdir = Path(output_dir)

    # ---- Load data ----
    eprint(f"[NCEM] Loading spatial data: {st_h5ad}")
    adata = sc.read_h5ad(st_h5ad)
    n_spots_supplied = int(adata.n_obs)
    eprint(f"[NCEM] Loaded: {adata.n_obs} spots x {adata.n_vars} genes")

    # ---- Validate inputs ----
    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial key '{spatial_key}' not found in adata.obsm. Available: {list(adata.obsm.keys())}")
    if annotation_key not in adata.obs:
        raise ValueError(
            f"annotation_key='{annotation_key}' not found in adata.obs. Available keys: {list(adata.obs.columns)}"
        )
    # Refuse a 3-column key, or a stack read as one plane, before any work.
    _, frame = spatial_frame(adata, spatial_key, 2, section_key, "NCEM", False)

    run_warnings: list = []

    # ---- Off-tissue spots are background, not cells (worker_utils.keep_in_tissue, the shared rule) ----
    adata, _, n_off_tissue = keep_in_tissue(adata, "spots")
    if n_off_tissue:
        eprint(f"[NCEM] Leaving out {n_off_tissue} spots with in_tissue == 0")

    # ---- The matrix normalise + log1p runs on: X, or adata.raw.X with use_raw_counts ----
    adata, counts_info = _choose_counts(adata, use_raw_counts)
    # Gene axis only: the top-genes table below is keyed by gene symbol, so a rename here
    # invents names the user reads back. The cell axis is left as supplied, as it always was.
    renamed = make_names_unique_and_report(adata, axes=("var",))

    # ---- Spots with no label ----
    label_keep, n_unlabeled = split_unlabeled(
        adata.obs[annotation_key].to_numpy(), drop_unlabeled, what=f"spots in obs['{annotation_key}']"
    )
    if n_unlabeled:
        eprint(f"[NCEM] Leaving out {n_unlabeled} spots with no '{annotation_key}' label (drop_unlabeled=True)")
        adata = adata[label_keep].copy()

    def _write_plane(res: dict, folder: Path) -> dict:
        """The four tables of one fitted plane, each written atomically into ``folder``."""
        comm_csv = folder / "ncem_communication_matrix.csv"
        _atomic_to_csv(res["matrix"], comm_csv)
        eprint(f"[NCEM] Saved communication matrix to {comm_csv}")
        strength_csv = folder / "ncem_communication_strength.csv"
        _atomic_to_csv(res["strength"], strength_csv, index=False)
        eprint(f"[NCEM] Saved communication strength to {strength_csv}")
        nhood_csv = folder / "ncem_neighborhood_composition.csv"
        _atomic_to_csv(res["composition"], nhood_csv)
        eprint(f"[NCEM] Saved neighborhood composition to {nhood_csv}")
        top_genes_csv = folder / "ncem_top_genes_per_type.csv"
        _atomic_to_csv(res["top_genes"], top_genes_csv, index=False)
        eprint(f"[NCEM] Saved top genes per type to {top_genes_csv}")
        return {
            "communication_matrix_csv": str(comm_csv),
            "communication_strength_csv": str(strength_csv),
            "neighborhood_composition_csv": str(nhood_csv),
            "top_genes_per_type_csv": str(top_genes_csv),
        }

    per_result: dict = {}
    if section_key:
        labels_in_order = frame.sections or [str(s) for s in pd.unique(adata.obs[section_key].astype(str))]
        dirs = _section_dirs(labels_in_order)

        def run_one(sub, label):
            folder = _ensure_dir(str(outdir / dirs[label]))
            res = _fit_one(sub, spatial_key, annotation_key, n_neighbors, n_hvg, f"section '{label}': ", random_seed)
            _write_plane(res, folder)
            res["folder"] = str(folder)
            per_result[label] = res
            return res["strength"]

        long = per_section(adata, section_key, run_one, "NCEM")
        strength_csv = outdir / "ncem_communication_strength.csv"
        _write_long_table(long, strength_csv)
        eprint(f"[NCEM] Saved per-section communication strength to {strength_csv}")
        section_order = [s for s in labels_in_order if s in per_result]
        results = [per_result[s] for s in section_order]
        n_spots = int(sum(r["n_spots"] for r in results))
        unique_types = sorted({t for r in results for t in r["unique_types"]}, key=str)
        n_genes = int(adata.n_vars)
        n_genes_used = max(r["n_genes_used"] for r in results)
        n_hvg_actual = max(r["n_hvg_actual"] for r in results)
        dense_bytes = max(r["dense_bytes"] for r in results)
        files = {
            "communication_strength_csv": str(strength_csv),
            "section_dirs": [per_result[s]["folder"] for s in section_order],
        }
        mode = "per-section-2d"
    else:
        res = _fit_one(adata, spatial_key, annotation_key, n_neighbors, n_hvg, "", random_seed)
        section_order = None
        n_spots, unique_types = res["n_spots"], res["unique_types"]
        n_genes, n_genes_used, n_hvg_actual = res["n_genes"], res["n_genes_used"], res["n_hvg_actual"]
        dense_bytes = res["dense_bytes"]
        files = _write_plane(res, outdir)
        mode = "2d"
    n_types = len(unique_types)

    # ---- Build output ----
    out = WorkerOutput("ncem", task="cell_communication")
    out.set_data(
        n_spots=n_spots,
        n_spots_supplied=n_spots_supplied,
        n_spots_out_of_tissue_excluded=n_off_tissue,
        n_cells_dropped_unlabeled=int(n_unlabeled),
        n_genes=n_genes,
        n_genes_used=n_genes_used,
        n_cell_types=n_types,
    )
    if section_order is not None:
        out.set_data(
            n_sections=len(section_order),
            n_spots_per_section={s: int(per_result[s]["n_spots"]) for s in section_order},
            n_genes_used_per_section={s: int(per_result[s]["n_genes_used"]) for s in section_order},
        )
    out.add_output_files(files)
    out.add_params(
        {
            "spatial_key": spatial_key,
            "coords_key": spatial_key,
            "dims": 2,
            "section_key": section_key,
            "mode": mode,
            "sections": section_order,
            "frame": frame.to_dict(),
            "annotation_key": annotation_key,
            "n_neighbors": n_neighbors,
            "model_type": model_type,
            "n_hvg": n_hvg_actual,
            "n_hvg_requested": int(n_hvg),
            "random_seed": random_seed,
            "ridge_alpha": RIDGE_ALPHA,
            "ncem_package_used": False,
            "drop_unlabeled": bool(drop_unlabeled),
            "n_cells_dropped_unlabeled": int(n_unlabeled),
            "n_spots_out_of_tissue_excluded": n_off_tissue,
            "dense_target_bytes_estimated": int(dense_bytes),
            "use_raw_counts": bool(use_raw_counts),
        }
    )
    record_in_tissue(out, n_spots_supplied, n_off_tissue)
    record_expression_source(out, counts_info)
    out.add_params(identifier_rename_params(renamed))
    record_method(out, METHOD_NAME, used_fallback=False)
    record_ignored(out, ["max_epochs", "learning_rate"], _NO_TRAINING)
    record_ignored(out, "seed", _NO_RANDOMNESS)
    out.add_warnings(run_warnings)

    reasons = []
    if n_off_tissue:
        reasons.append(f"the in_tissue flag ({n_off_tissue} spots with in_tissue == 0)")
    if n_unlabeled:
        reasons.append(f"drop_unlabeled=True ({n_unlabeled} spots with no '{annotation_key}' label)")
    reduction = describe_reduction("spots", n_spots_supplied, n_spots, " and ".join(reasons))
    if reduction and n_unlabeled:  # the in-tissue cut alone is already warned about by record_in_tissue
        out.add_warning(reduction.strip())

    if section_order is None:
        r_squared = res["r_squared"]
        strength_df = res["strength"]
        top_communicators = res["top_communicators"]
        out.set_summary(
            r_squared=float(r_squared),
            cell_types=unique_types,
            communication_strength={
                row["cell_type"]: float(row["mean_effect_strength"]) for _, row in strength_df.iterrows()
            },
            top_communicating_types=top_communicators,
            top_genes_per_type=res["top_genes_per_type"],
        )
        out.set_analysis(
            f"{METHOD_SHORT} across {n_types} cell types in {n_spots} spatial spots, "
            f"k = {n_neighbors} neighbours, using {n_genes_used} highly variable genes of {n_genes}. "
            f"In-sample R-squared (own cell type + neighbourhood composition, pooled over genes) = {r_squared:.3f}. "
            f"Neighbour types with the largest mean |effect|: {', '.join(str(t) for t in top_communicators[:3])}. "
            "Effects are relative to the average neighbour type (neighbour fractions sum to 1)."
            + reduction
            + identifier_rename_note(renamed)
        )
    else:
        out.set_summary(
            cell_types=unique_types,
            r_squared_per_section={s: float(per_result[s]["r_squared"]) for s in section_order},
            top_communicating_types_per_section={s: per_result[s]["top_communicators"] for s in section_order},
            communication_strength_per_section={
                s: {
                    row["cell_type"]: float(row["mean_effect_strength"])
                    for _, row in per_result[s]["strength"].iterrows()
                }
                for s in section_order
            },
            top_genes_per_type_per_section={s: per_result[s]["top_genes_per_type"] for s in section_order},
        )
        per_text = "; ".join(
            "{}: {} spots, R-squared {:.3f}, strongest {}".format(
                s,
                per_result[s]["n_spots"],
                per_result[s]["r_squared"],
                ", ".join(str(t) for t in per_result[s]["top_communicators"][:3]),
            )
            for s in section_order
        )
        out.set_analysis(
            f"{METHOD_SHORT}, run per section in 2D ({len(section_order)} sections of {section_key}; each "
            f"section has its own neighbours, highly variable genes and fit, and no neighbourhood spans two "
            f"sections): {n_types} cell types in {n_spots} spatial spots, k = {n_neighbors} neighbours. {per_text}. "
            "Effects are relative to the average neighbour type (neighbour fractions sum to 1)."
            + reduction
            + identifier_rename_note(renamed)
        )

    return out.to_dict()


def main():
    ap = argparse.ArgumentParser(description="NCEM-style neighbourhood Ridge cell communication worker")
    ap.add_argument("--st-h5ad", required=True, help="Path to spatial AnnData (.h5ad)")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--spatial-key", default="spatial", help="obsm key for spatial coordinates")
    ap.add_argument(
        "--coords-key", default=None, help="obsm key for spatial coordinates (same as --spatial-key; wins when given)"
    )
    ap.add_argument("--dims", type=int, default=2, choices=[2, 3], help="2 (the only one this tool runs); 3 is refused")
    ap.add_argument(
        "--section-key", default=None, help="obs column naming each spot's section; the run is then per section"
    )
    ap.add_argument("--annotation-key", default="cell_type", help="obs column for cell type labels")
    ap.add_argument("--n-neighbors", type=int, default=10, help="Number of spatial neighbors")
    ap.add_argument(
        "--max-epochs", type=int, default=50, help="Accepted and ignored: the Ridge fit is closed-form (no training)"
    )
    ap.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Accepted and ignored: the Ridge fit is closed-form"
    )
    ap.add_argument("--n-hvg", type=int, default=2000, help="Number of highly variable genes")
    ap.add_argument("--model-type", default="linear", choices=["linear"], help="Model type (only 'linear')")
    ap.add_argument("--seed", type=int, default=0, help="Accepted and ignored: nothing in the method is random")
    ap.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="Leave out spots with no label in obs[annotation-key] instead of refusing (the count is reported)",
    )
    ap.add_argument(
        "--use-raw-counts",
        action="store_true",
        help="Normalise adata.raw.X instead of X (refused when adata.raw is absent or does not hold counts)",
    )
    args = ap.parse_args()

    try:
        # stdout carries only the JSON payload: anything scanpy or sklearn prints goes to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            result = run_ncem(
                st_h5ad=args.st_h5ad,
                output_dir=args.output_dir,
                spatial_key=args.coords_key or args.spatial_key,
                annotation_key=args.annotation_key,
                n_neighbors=args.n_neighbors,
                max_epochs=args.max_epochs,
                learning_rate=args.learning_rate,
                n_hvg=args.n_hvg,
                model_type=args.model_type,
                random_seed=args.seed,
                drop_unlabeled=args.drop_unlabeled,
                use_raw_counts=args.use_raw_counts,
                dims=args.dims,
                section_key=args.section_key,
            )
        print(json.dumps(result, default=str))

    except Exception as e:
        eprint(f"[NCEM] ERROR: {e}")
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("ncem", str(e), task="cell_communication")
        sys.exit(1)


if __name__ == "__main__":
    main()
