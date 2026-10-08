#!/usr/bin/env python
# /workspace/SpatialOmicsGym/tools/cellcharter_worker.py
"""CellCharter spatial-domain worker.

Pipeline: a squidpy spatial graph (``coord_type='grid'`` when ``obs`` has ``array_row``/``array_col``,
otherwise ``'generic'``; always on ``obsm[spatial_key]``, split per library when ``library_key`` is
given) -> ``cellcharter.gr.aggregate_neighbors`` -> ``cellcharter.tl.ClusterAutoK``.

The representation CellCharter aggregates is ``obsm[use_rep]`` when present, else ``obsm['X_scVI']``,
else ``obsm['X_pca']``, else a PCA this worker computes itself: ``normalize_total(1e4)`` + ``log1p``
of ``X``, the ``HVG_N_TOP_GENES`` most highly variable genes by ``hvg_flavor`` (``seurat_v3`` ranks the
raw counts in ``layers['counts']``, which is what that flavour models -- a copy of ``X`` taken before
normalisation when the layer is absent), ``scale(max_value=10)`` and a ``PCA_N_COMPS``-component PCA.
Every step but the seurat_v3 ranking runs on ``X``, so ``X`` is what must hold raw counts. ``scale`` z-scores the HVG matrix, which densifies it
(n_spots x n_HVG); that dense matrix is intrinsic to this preprocessing. The payload names what ran:
``params.use_rep`` is the key CellCharter aggregated, ``params.use_rep_requested`` the key asked for,
``params.use_rep_computed`` whether this worker built it, and a warning says so when they differ.

Background spots (``obs['in_tissue'] == 0``, the glass around a CELLxGENE Visium export) are left out
right after loading, before the spatial graph, the aggregation and ClusterAutoK see them; the cut is
reported as ``params.in_tissue_filter`` with a warning, ``data.n_spots`` is the slide supplied and
``data.n_spots_used`` the spots clustered. When the worker builds its own PCA, the matrix it treats as
counts is chosen by ``worker_utils.choose_counts_matrix``: ``X`` (negative or NaN values are refused,
fractional ones run with a warning) or, with ``use_raw_counts``, ``adata.raw.X``
(``params.expression_source``). The stdout of the squidpy/CellCharter calls (ClusterAutoK prints its
iterations) goes to stderr, so stdout carries only the JSON payload.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback

from worker_utils import (
    WorkerOutput,
    build_cluster_analysis,
    choose_counts_matrix,
    describe_reduction,
    expression_matrix_kind,
    keep_in_tissue,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    require_hvg_flavor,
    unsupported_choice_msg,
)

# --- Fixed settings of the worker's own PCA branch. Reported in ``params.preprocessing`` when that
# --- branch runs; ``hvg_flavor`` is the one caller-facing knob among them.
HVG_FLAVORS = ("seurat_v3", "seurat", "cell_ranger")
DEFAULT_HVG_FLAVOR = "seurat_v3"
HVG_N_TOP_GENES = 2000
NORMALIZE_TARGET_SUM = 1e4
SCALE_MAX_VALUE = 10
PCA_N_COMPS = 30
COUNTS_LAYER = "counts"

# cellcharter.gr.aggregate_neighbors reads obs[<this>] as the sample split when sample_key is None.
CELLCHARTER_IMPLICIT_SAMPLE_KEY = "sample"

METHOD_NAME = "CellCharter (squidpy spatial graph -> aggregate_neighbors -> ClusterAutoK)"


def log(msg):
    print(f"[cellcharter-worker] {msg}", file=sys.stderr, flush=True)


def search_range_error(n_clusters_min, n_clusters_max, max_runs):
    """Why ClusterAutoK cannot run this K search, or "" when it can.

    ClusterAutoK scores each K by the stability between repeated runs at K and its neighbours, so
    it fits every K in ``[n_clusters_min - 1, n_clusters_max + 1]`` ``max_runs`` times and can
    choose only from ``[max(2, n_clusters_min), n_clusters_max]``. Upstream checks neither bound
    before fitting: ``max_runs=1`` fits the whole range and then dies in ``predict`` ("Cannot
    compute stability with max_runs <= 1"), and ``n_clusters_min > n_clusters_max`` dies inside
    ``fit`` with "range() arg 3 must not be zero". Both are rejected here, before any compute.
    """
    problems = []
    if int(max_runs) < 2:
        problems.append(
            f"max_runs={max_runs}: ClusterAutoK selects K by the stability between repeated runs, so it "
            "needs max_runs >= 2 (upstream fits every K and then fails in predict with max_runs <= 1)"
        )
    if int(n_clusters_max) < 2 or int(n_clusters_max) < int(n_clusters_min):
        problems.append(
            f"n_clusters_min={n_clusters_min}, n_clusters_max={n_clusters_max}: ClusterAutoK chooses K from "
            "[max(2, n_clusters_min), n_clusters_max], so n_clusters_max must be >= 2 and >= n_clusters_min"
        )
    return "; ".join(problems)


def _looks_like_counts(X, chunk_values=5000000):
    """True when every stored value is a non-negative integer. Reads every value, a chunk at a time."""
    import numpy as np
    import scipy.sparse as sp

    if sp.issparse(X):
        values = np.asarray(X.data)
        for start in range(0, values.size, chunk_values):
            block = values[start : start + chunk_values]
            if (block < 0).any() or (block != np.floor(block)).any():
                return False
        return True
    arr = np.asarray(X)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1) if arr.ndim > 0 else arr.reshape(1, 1)
    rows = max(1, chunk_values // max(1, arr.shape[1]))
    for start in range(0, arr.shape[0], rows):
        block = arr[start : start + rows]
        if (block < 0).any() or (block != np.floor(block)).any():
            return False
    return True


def _select_hvg(adata, hvg_flavor):
    """Mark ``var['highly_variable']`` with the caller's flavour, or stop -- never carry on with every gene.

    ``seurat_v3`` models raw counts, so it reads ``layers['counts']``; ``seurat`` and ``cell_ranger``
    rank the log-normalised ``X``. The old handler logged a stderr line and went on to scale
    (densify) and PCA the whole gene axis, reporting nothing in the payload.
    """
    import scanpy as sc

    hvg_kwargs = {"layer": COUNTS_LAYER} if hvg_flavor == "seurat_v3" else {}
    try:
        sc.pp.highly_variable_genes(adata, flavor=hvg_flavor, n_top_genes=HVG_N_TOP_GENES, **hvg_kwargs)
    except Exception as exc:
        # Narrow on purpose (test/test_worker_hvg_seurat_v3_fallback.py): a broken scikit-misc raises
        # in here. Nothing is substituted -- no other flavour, and no run on the full gene axis.
        extra = (
            " hvg_flavor='seurat_v3' needs a working scikit-misc (skmisc.loess): if the error names skmisc, "
            "loess or a numpy dtype size, reinstall scikit-misc against this environment's numpy, or pass "
            "hvg_flavor='seurat' explicitly."
            if hvg_flavor == "seurat_v3"
            else ""
        )
        raise RuntimeError(
            f"highly_variable_genes(flavor={hvg_flavor!r}, n_top_genes={HVG_N_TOP_GENES}) failed "
            f"({type(exc).__name__}: {exc}). No other flavour was substituted.{extra}"
        ) from exc
    return f"layers['{COUNTS_LAYER}'] (raw counts)" if hvg_kwargs else "X (log1p-normalised)"


def compute_pca_representation(adata, hvg_flavor):
    """Build ``obsm['X_pca']`` from the expression matrix; return ``(adata_hvg, info)``.

    ``adata_hvg`` is a new AnnData restricted to the selected genes, with the scaled matrix in X
    (that is what ``cellcharter_annotated.h5ad`` then carries). ``info`` records what ran:
    normalize_total, log1p, scale and PCA always run on ``X`` (``normalized_input``, checked as
    ``x_counts_like``); ``layers['counts']`` is read only by the seurat_v3 ranking (``counts_source``
    says where that layer came from, ``counts_like`` whether it holds counts).
    """
    import numpy as np
    import scanpy as sc

    x_counts_like = _looks_like_counts(adata.X)
    if COUNTS_LAYER in adata.layers.keys():
        counts_source = f"layers['{COUNTS_LAYER}']"
        counts_like = _looks_like_counts(adata.layers[COUNTS_LAYER])
    else:
        log(f"Creating adata.layers['{COUNTS_LAYER}'] from adata.X")
        adata.layers[COUNTS_LAYER] = adata.X.copy()
        counts_source = "X"
        counts_like = x_counts_like

    sc.pp.normalize_total(adata, target_sum=NORMALIZE_TARGET_SUM)
    sc.pp.log1p(adata)

    hvg_input = _select_hvg(adata, hvg_flavor)
    if hvg_flavor == "seurat_v3" and not counts_like:
        hvg_input = f"layers['{COUNTS_LAYER}'] (ranked as raw counts, but not all its values are non-negative integers)"
    if "highly_variable" not in adata.var.columns:
        raise RuntimeError(f"highly_variable_genes(flavor={hvg_flavor!r}) left no var['highly_variable'] column.")
    mask = np.asarray(adata.var["highly_variable"], dtype=bool)
    n_hvg = int(mask.sum())
    if n_hvg == 0:
        raise RuntimeError(
            f"highly_variable_genes(flavor={hvg_flavor!r}, n_top_genes={HVG_N_TOP_GENES}) selected no gene out of "
            f"{adata.n_vars}; nothing was substituted. Check that {hvg_input} holds expression, pass another "
            "hvg_flavor, or supply a representation in obsm via use_rep."
        )
    log(f"Selected {n_hvg} highly variable genes (flavor={hvg_flavor!r}, on {hvg_input})")
    adata = adata[:, mask].copy()

    sc.pp.scale(adata, max_value=SCALE_MAX_VALUE)
    sc.tl.pca(adata, n_comps=PCA_N_COMPS)
    log("Computed PCA; stored in adata.obsm['X_pca']")
    info = {
        "counts_source": counts_source,
        "counts_like": bool(counts_like),
        "normalized_input": "X",
        "x_counts_like": bool(x_counts_like),
        "normalize_total_target_sum": NORMALIZE_TARGET_SUM,
        "log1p": True,
        "hvg_flavor": hvg_flavor,
        "hvg_input": hvg_input,
        "hvg_n_top_genes": HVG_N_TOP_GENES,
        "n_hvg_used": n_hvg,
        "scale_max_value": SCALE_MAX_VALUE,
        "pca_n_comps": PCA_N_COMPS,
    }
    return adata, info


def _write_h5ad_atomic(adata, path):
    """Write the AnnData as ``<path>.partial`` and rename over ``path`` once it is complete."""
    tmp = str(path) + ".partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, str(path))


def _write_csv_atomic(df, path):
    """Write ``df`` as ``<path>.partial`` and rename over ``path`` once it is complete."""
    tmp = str(path) + ".partial"
    df.to_csv(tmp, index=False)
    os.replace(tmp, str(path))


def run_cellcharter_clustering(
    st_h5ad,
    output_dir,
    spatial_key="spatial",
    use_rep="X_scVI",
    n_layers=3,
    n_clusters_min=3,
    n_clusters_max=12,
    max_runs=5,
    convergence_tol=0.001,
    cluster_key="cluster_cellcharter",
    library_key=None,
    hvg_flavor=DEFAULT_HVG_FLAVOR,
    use_raw_counts=False,
):
    import anndata as ad
    import cellcharter as cc
    import numpy as np
    import pandas as pd
    import squidpy as sq

    if hvg_flavor not in HVG_FLAVORS:
        raise ValueError(unsupported_choice_msg("hvg_flavor", hvg_flavor, HVG_FLAVORS))

    os.makedirs(output_dir, exist_ok=True)

    log("Task = clustering")
    log(f"ST h5ad      = {st_h5ad}")
    log(f"output_dir   = {output_dir}")
    log(f"spatial_key  = {spatial_key}")
    log(f"use_rep      = {use_rep}")
    log(f"n_layers     = {n_layers}")
    log(f"cluster range= [{n_clusters_min}, {n_clusters_max}]")
    log(f"max_runs     = {max_runs}")
    log(f"convergence_tol = {convergence_tol}")
    log(f"cluster_key  = {cluster_key}")
    log(f"library_key  = {library_key}")
    log(f"hvg_flavor   = {hvg_flavor}")
    log(f"use_raw_counts = {bool(use_raw_counts)}")

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    if not os.path.exists(st_h5ad):
        raise FileNotFoundError(f"st_h5ad not found: {st_h5ad}")

    adata = ad.read_h5ad(st_h5ad)
    log(f"Loaded ST data: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    # Background spots (obs['in_tissue'] == 0) are not tissue: left out before the spatial graph, the
    # neighbour aggregation and ClusterAutoK, and reported. Every other spot-clustering worker has done
    # this since round 2; here the glass got domains of its own and in-tissue edge spots aggregated it.
    adata, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata, "spots")
    if n_spots_off_tissue:
        log(f"Leaving out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0")
    # What the user handed us. The representation branch below may cut the gene axis down to the
    # 2,000 most highly variable, and the payload has to report both numbers when it does.
    n_genes_supplied = int(adata.n_vars)

    # Check spatial key
    if spatial_key not in adata.obsm.keys():
        raise ValueError(
            f"spatial_key='{spatial_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}"
        )
    # A library_key that names no column used to be dropped with a stderr line, and the run went on as
    # one sample while params still echoed the key as if it had split the data.
    if library_key and library_key not in adata.obs.columns:
        raise ValueError(
            f"library_key={library_key!r} is not a column of adata.obs (columns: {list(adata.obs.columns)}). "
            "Name the obs column that holds the sample/library IDs, or leave library_key empty for a single "
            "sample; the run does not go ahead without the sample split it was asked for."
        )
    if library_key:
        # squidpy builds the per-library graph only over a categorical column, and a spot with no
        # library ID would fall out of every library's graph.
        n_no_library = int(adata.obs[library_key].isna().sum())
        if n_no_library:
            raise ValueError(
                f"{n_no_library} of {adata.n_obs} spots have no {library_key!r} value; every spot needs a "
                "library ID to be placed in a per-library spatial graph. Fill the column, or leave library_key "
                "empty for a single sample."
            )
        library_col = adata.obs[library_key]
        if not isinstance(library_col.dtype, pd.CategoricalDtype):
            log(f"Casting obs['{library_key}'] to categorical for squidpy's per-library graph")
            library_col = library_col.astype("category")
        # squidpy loops over the column's *categories* (gr/_build.py), not its values, and fits a kNN
        # on each: a category no spot carries fails with "Found array with 0 sample(s)". No spot
        # changes library here; only the empty categories go.
        categories_before = [str(c) for c in library_col.cat.categories]
        library_col = library_col.cat.remove_unused_categories()
        categories_kept = {str(c) for c in library_col.cat.categories}
        unused = [c for c in categories_before if c not in categories_kept]
        if unused:
            log(f"Dropping {len(unused)} unused categories of obs['{library_key}'] (no spot carries them): {unused}")
        adata.obs[library_key] = library_col

    # -------------------------------------------------------------------------
    # 2. Determine representation (use_rep / X_scVI / X_pca / compute PCA)
    # -------------------------------------------------------------------------
    obsm_keys_supplied = [str(k) for k in adata.obsm.keys()]
    rep_key = None
    rep_computed = False
    preprocessing = None
    counts_choice = None
    raw_holds_counts = False

    if use_rep and use_rep in adata.obsm.keys():
        rep_key = use_rep
        log(f"Using existing representation: adata.obsm['{rep_key}']")
    elif "X_scVI" in adata.obsm.keys():
        rep_key = "X_scVI"
        log("use_rep not found; falling back to adata.obsm['X_scVI']")
    elif "X_pca" in adata.obsm.keys():
        rep_key = "X_pca"
        log("use_rep and 'X_scVI' not found; using existing adata.obsm['X_pca']")
    else:
        log("No low-dimensional representation found; computing PCA with Scanpy")
        # Checked here, not up front: a run that aggregates an obsm representation selects no genes and
        # must not need scikit-misc.
        require_hvg_flavor(hvg_flavor)
        # The preprocessing treats the matrix as counts (normalize_total, log1p, seurat_v3). A scaled X
        # (negative values) or NaN is refused, naming use_raw_counts when adata.raw holds counts; with
        # use_raw_counts the counts in adata.raw.X are what is normalised.
        adata, counts_choice = choose_counts_matrix(adata, use_raw_counts)
        # The panel the counts come with: adata.raw can carry more genes than X.
        n_genes_supplied = int(adata.n_vars)
        if counts_choice["x_matrix_kind"] != "counts" and not use_raw_counts and adata.raw is not None:
            # Only for the warning below: whether the counts it should point to are already in the file.
            raw_holds_counts = expression_matrix_kind(adata.raw.X) == "counts"
        adata, preprocessing = compute_pca_representation(adata, hvg_flavor)
        rep_key = "X_pca"
        rep_computed = True

    # -------------------------------------------------------------------------
    # 3. Build spatial neighbors graph with Squidpy
    # -------------------------------------------------------------------------
    # Grid neighbours for Visium-like data (obs carries the array grid), generic kNN otherwise. Both
    # read obsm[spatial_key]; the grid branch used to fall through to squidpy's default 'spatial'.
    coord_type = "grid" if ("array_row" in adata.obs.columns and "array_col" in adata.obs.columns) else "generic"
    graph_kwargs = {"coord_type": coord_type, "spatial_key": spatial_key}
    if library_key:
        # One graph per library: slices share a coordinate frame, so a pooled kNN links spots of
        # different slices, and aggregate_neighbors then keeps only the within-sample part of it.
        graph_kwargs["library_key"] = library_key
    log(f"Building spatial neighbors: {graph_kwargs}")
    try:
        with contextlib.redirect_stdout(sys.stderr):
            sq.gr.spatial_neighbors(adata, **graph_kwargs)
    except Exception as e:
        detail = ""
        if library_key:
            # squidpy builds a separate nearest-neighbour graph inside each library, so the error is
            # usually one library too small for it; say which, smallest first.
            sizes = adata.obs[library_key].value_counts().sort_values()
            shown = {str(k): int(v) for k, v in sizes.head(10).items()}
            detail = (
                f" -- squidpy builds one spatial graph per library of obs[{library_key!r}]; spots per library, "
                f"smallest first: {shown}{' ...' if sizes.size > 10 else ''}. A library with no more spots than "
                "squidpy's neighbour count (6 by default) cannot get one: drop that library from the input, or "
                "leave library_key empty, which pools every library into one graph."
            )
        raise RuntimeError(f"Failed to compute spatial neighbors with Squidpy ({graph_kwargs}): {e}{detail}") from e

    # -------------------------------------------------------------------------
    # 4. Neighborhood aggregation (CellCharter)
    # -------------------------------------------------------------------------
    agg_key = "X_cellcharter"
    log(f"Running CellCharter neighborhood aggregation: n_layers={n_layers}, use_rep='{rep_key}', out_key='{agg_key}'")

    agg_kwargs = {
        "n_layers": int(n_layers),
        "use_rep": rep_key,
        "out_key": agg_key,
    }
    if library_key:
        agg_kwargs["sample_key"] = library_key
        log(f"Using sample_key='{library_key}' for multi-sample aggregation")
    # What aggregate_neighbors will split by: with sample_key=None it falls back to obs['sample'].
    if library_key:
        sample_key_used = library_key
    elif CELLCHARTER_IMPLICIT_SAMPLE_KEY in adata.obs.columns:
        sample_key_used = CELLCHARTER_IMPLICIT_SAMPLE_KEY
    else:
        sample_key_used = None

    with contextlib.redirect_stdout(sys.stderr):
        cc.gr.aggregate_neighbors(adata, **agg_kwargs)

    if agg_key not in adata.obsm.keys():
        raise RuntimeError(f"Aggregation failed: '{agg_key}' not found in adata.obsm after aggregate_neighbors")
    log(f"Aggregation complete. obsm['{agg_key}'] shape = {adata.obsm[agg_key].shape}")

    # -------------------------------------------------------------------------
    # 5. ClusterAutoK spatial clustering
    # -------------------------------------------------------------------------
    log(
        "Running CellCharter.tl.ClusterAutoK "
        f"n_clusters=({n_clusters_min},{n_clusters_max}), "
        f"max_runs={max_runs}, convergence_tol={convergence_tol}"
    )

    autok = cc.tl.ClusterAutoK(
        n_clusters=(int(n_clusters_min), int(n_clusters_max)),
        max_runs=int(max_runs),
        convergence_tol=float(convergence_tol),
    )

    # ClusterAutoK.fit prints "Iteration i/max_runs" and its convergence line to stdout, which is the
    # payload's channel; they go to stderr with the rest of the log.
    with contextlib.redirect_stdout(sys.stderr):
        autok.fit(adata, use_rep=agg_key)
    log("ClusterAutoK.fit() finished")

    # If k is omitted, CellCharter chooses the best candidate (highest stability)
    with contextlib.redirect_stdout(sys.stderr):
        clusters = autok.predict(adata, use_rep=agg_key)
    clusters = np.asarray(clusters)
    adata.obs[cluster_key] = clusters.astype(str)
    log(f"Assigned clusters in adata.obs['{cluster_key}']")

    unique_clusters = np.unique(clusters)
    n_clusters_found = unique_clusters.size
    log(f"Found {n_clusters_found} clusters: {unique_clusters}")

    # The K ClusterAutoK could choose (the fitted range minus its two stability-only ends) and the one
    # it chose. n_clusters_min=1 fits K=1 but can never select it.
    fitted_k = getattr(autok, "n_clusters", None)
    k_candidates = [int(k) for k in fitted_k[1:-1]] if isinstance(fitted_k, (list, tuple)) else None
    try:
        k_selected = int(autok.best_k)
    except Exception:
        k_selected = None

    # -------------------------------------------------------------------------
    # 6. Save outputs
    # -------------------------------------------------------------------------
    annotated_h5ad = os.path.join(output_dir, "cellcharter_annotated.h5ad")
    clusters_csv = os.path.join(output_dir, "cellcharter_clusters.csv")

    log(f"Writing annotated AnnData to: {annotated_h5ad}")
    _write_h5ad_atomic(adata, annotated_h5ad)

    log(f"Writing cluster assignments to: {clusters_csv}")
    cluster_df = pd.DataFrame(
        {
            "spot_id": adata.obs_names,
            cluster_key: adata.obs[cluster_key].astype(str).values,
        }
    )
    _write_csv_atomic(cluster_df, clusters_csv)

    n_spots = int(adata.n_obs)
    n_genes_used = int(adata.n_vars)
    cluster_sizes = {str(k): int(v) for k, v in adata.obs[cluster_key].value_counts().items()}

    # Self-gating: returns "" when the representation came from obsm and no gene selection ran.
    gene_note = describe_reduction(
        "genes",
        n_genes_supplied,
        n_genes_used,
        f"this worker's own preprocessing, which keeps the {HVG_N_TOP_GENES} most highly variable genes "
        f"(hvg_flavor={hvg_flavor!r}) when it has to build the representation itself",
    )

    if rep_computed:
        # normalize_total, log1p, scale and PCA ran on X; only the HVG ranking may have read the layer.
        rep_sentence = (
            f"Representation: obsm['X_pca'] computed by this worker from {preprocessing['normalized_input']} "
            f"(normalize_total {NORMALIZE_TARGET_SUM:g} + log1p of {preprocessing['normalized_input']}, "
            f"{preprocessing['n_hvg_used']} HVGs by hvg_flavor={hvg_flavor!r} ranked on "
            f"{preprocessing['hvg_input']}, scale, {PCA_N_COMPS}-component PCA), "
            f"because use_rep={use_rep!r} was not in obsm (keys: {obsm_keys_supplied})."
        )
    elif rep_key != use_rep:
        rep_sentence = (
            f"Representation: the existing obsm['{rep_key}'], because use_rep={use_rep!r} was not in obsm "
            f"(keys: {obsm_keys_supplied})."
        )
    else:
        rep_sentence = f"Representation: obsm['{rep_key}'] as requested."

    out = WorkerOutput("cellcharter", task="clustering")
    # n_spots is the slide supplied, n_spots_used the in-tissue spots clustered (the CSV's rows).
    out.set_data(n_spots=n_spots_supplied, n_spots_used=n_spots, n_genes=n_genes_supplied, n_genes_used=n_genes_used)
    out.add_output_files(
        {
            "annotated_h5ad": annotated_h5ad,
            "clusters_csv": clusters_csv,
        }
    )
    out.add_params(
        {
            # The key CellCharter aggregated -- not the one asked for, which is use_rep_requested.
            "use_rep": rep_key,
            "use_rep_requested": use_rep,
            "use_rep_computed": bool(rep_computed),
            "hvg_flavor": hvg_flavor,
            "n_layers": n_layers,
            "n_clusters_min": n_clusters_min,
            "n_clusters_max": n_clusters_max,
            "k_candidates": k_candidates,
            "k_selected": k_selected,
            "max_runs": max_runs,
            "convergence_tol": convergence_tol,
            "cluster_key": cluster_key,
            "library_key": library_key,
            "sample_key_used": sample_key_used,
            "spatial_key": spatial_key,
            "coord_type": coord_type,
            "use_raw_counts": bool(use_raw_counts),
        }
    )
    if preprocessing is not None:
        out.add_param("preprocessing", preprocessing)
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    raw_hint = ""
    if counts_choice is not None:
        # The params keys only: the X warning below is this worker's own, more specific one (it names
        # each step that read X), and a second sentence about the same matrix would repeat it.
        record_expression_source(out, dict(counts_choice, warning=None))
        raw_hint = (
            " adata.raw holds raw counts: pass use_raw_counts=True to run on them."
            if raw_holds_counts
            else " Supply raw counts in X (or in adata.raw, with use_raw_counts=True), or a precomputed "
            "representation via use_rep."
        )
    elif use_raw_counts:
        record_ignored(
            out,
            "use_raw_counts",
            f"the representation came from obsm['{rep_key}'], so this worker normalised no expression matrix",
        )
    record_method(out, METHOD_NAME, used_fallback=False)
    out.set_summary(
        n_clusters=int(n_clusters_found),
        cluster_key=cluster_key,
        cluster_sizes=cluster_sizes,
        clusters=sorted([str(x) for x in unique_clusters]),
    )
    if rep_computed or rep_key != use_rep:
        out.add_warning(rep_sentence)
    if not rep_computed and hvg_flavor != DEFAULT_HVG_FLAVOR:
        record_ignored(
            out,
            "hvg_flavor",
            f"the representation came from obsm['{rep_key}'], so this worker selected no genes",
        )
    # Each matrix is flagged for the step that reads it: X for normalize_total/log1p/scale/PCA, and
    # layers['counts'] for the seurat_v3 ranking -- once, when that layer is the worker's copy of X.
    layer_is_copy_of_x = rep_computed and preprocessing["counts_source"] == "X"
    if rep_computed and not preprocessing["x_counts_like"]:
        # Negative or NaN values never get here (choose_counts_matrix refuses them); what is left is a
        # normalised or log-transformed matrix, which runs as before, with this warning.
        matrix = "adata.raw.X (use_raw_counts=True)" if use_raw_counts else "X"
        out.add_warning(
            f"{matrix} holds non-integer values, but this worker's preprocessing "
            f"treats it as raw counts: it runs normalize_total + log1p on it, then scale and PCA"
            + (
                f", and hvg_flavor='seurat_v3' ranks a copy of it (layers['{COUNTS_LAYER}']) as counts"
                if hvg_flavor == "seurat_v3" and layer_is_copy_of_x
                else ""
            )
            + "."
            + raw_hint
        )
    if rep_computed and hvg_flavor == "seurat_v3" and not layer_is_copy_of_x and not preprocessing["counts_like"]:
        out.add_warning(
            f"layers['{COUNTS_LAYER}'] holds non-integer or negative values, but hvg_flavor='seurat_v3' ranks it "
            f"as raw counts to choose the {preprocessing['n_hvg_used']} HVGs. Supply raw counts in "
            f"layers['{COUNTS_LAYER}'], or a precomputed representation via use_rep."
        )
    if sample_key_used and not library_key:
        n_samples = int(adata.obs[sample_key_used].nunique())
        if n_samples > 1:
            out.add_warning(
                f"library_key was empty, but CellCharter's aggregate_neighbors split the aggregation by "
                f"obs['{sample_key_used}'] ({n_samples} values), its own default sample key; the spatial graph "
                "was built over all spots. Pass library_key to split the graph too."
            )
    # A successful run never carries stderr into the payload (base_mcp attaches stderr_tail only on a
    # non-zero exit), so a cut this large has to travel in the payload itself.
    if gene_note:
        out.add_warning(gene_note.strip())
    tissue_sentence = ""
    if n_spots_off_tissue:
        tissue_sentence = (
            f" {n_spots_off_tissue} of the {n_spots_supplied} spots were left out because obs['in_tissue'] marks "
            "them as background outside the tissue; the spatial graph, the aggregation and the clustering saw "
            f"only the {n_spots} in-tissue spots."
        )
    out.set_analysis(
        build_cluster_analysis(cluster_sizes, cluster_key="cluster", total_spots=n_spots)
        + " "
        + rep_sentence
        + tissue_sentence
        + gene_note
    )

    log("CellCharter clustering finished successfully.")
    return out.to_dict()


def main():
    parser = argparse.ArgumentParser(description="CellCharter worker for spatial domain identification")
    parser.add_argument(
        "--task",
        default="clustering",
        choices=["clustering"],
        help="Task type (currently only 'clustering')",
    )
    parser.add_argument(
        "--st-h5ad",
        required=True,
        help="Path to spatial transcriptomics AnnData (.h5ad)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output files",
    )
    parser.add_argument(
        "--spatial-key",
        default="spatial",
        help="Key in adata.obsm with spatial coordinates, used for the grid and the generic graph (default: 'spatial')",
    )
    parser.add_argument(
        "--use-rep",
        default="X_scVI",
        help="Key in adata.obsm for low-dimensional representation "
        "(default: 'X_scVI'; falls back to 'X_scVI', then 'X_pca', then computes PCA; the payload's "
        "params.use_rep names the one used).",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="Number of neighborhood aggregation layers (default: 3)",
    )
    parser.add_argument(
        "--n-clusters-min",
        type=int,
        default=3,
        help="Minimum number of clusters for ClusterAutoK (default: 3)",
    )
    parser.add_argument(
        "--n-clusters-max",
        type=int,
        default=12,
        help="Maximum number of clusters for ClusterAutoK (default: 12)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=5,
        help="Maximum runs per K for ClusterAutoK, at least 2 (default: 5)",
    )
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=0.001,
        help="Stability convergence tolerance for ClusterAutoK (default: 1e-3)",
    )
    parser.add_argument(
        "--cluster-key",
        default="cluster_cellcharter",
        help="Column name in adata.obs for cluster labels (default: 'cluster_cellcharter')",
    )
    parser.add_argument(
        "--library-key",
        default="",
        help="Optional obs column specifying sample/library ID for multi-sample data (must exist in obs)",
    )
    parser.add_argument(
        "--hvg-flavor",
        default=DEFAULT_HVG_FLAVOR,
        help="scanpy highly_variable_genes flavour for the worker's own PCA branch: seurat_v3 (on raw counts, "
        "needs scikit-misc), seurat or cell_ranger (default: seurat_v3)",
    )
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        help="When the worker builds its own PCA, normalise the counts in adata.raw.X instead of X (for an h5ad "
        "whose X is log-normalised or scaled and whose counts sit in adata.raw). No effect when the "
        "representation comes from obsm.",
    )

    args = parser.parse_args()

    try:
        if args.task != "clustering":
            raise ValueError("Only task 'clustering' is currently supported.")
        problem = search_range_error(args.n_clusters_min, args.n_clusters_max, args.max_runs)
        if problem:
            raise ValueError(problem)

        result = run_cellcharter_clustering(
            st_h5ad=args.st_h5ad,
            output_dir=args.output_dir,
            spatial_key=args.spatial_key,
            use_rep=args.use_rep,
            n_layers=args.n_layers,
            n_clusters_min=args.n_clusters_min,
            n_clusters_max=args.n_clusters_max,
            max_runs=args.max_runs,
            convergence_tol=args.convergence_tol,
            cluster_key=args.cluster_key,
            library_key=args.library_key if args.library_key else None,
            hvg_flavor=args.hvg_flavor,
            use_raw_counts=bool(args.use_raw_counts),
        )
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("cellcharter", str(e), task="clustering")


if __name__ == "__main__":
    main()
