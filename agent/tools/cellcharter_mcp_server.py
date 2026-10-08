#!/usr/bin/env python3
"""CellCharter spatial clustering MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "cellcharter"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "CELLCHARTER",
    "/opt/conda/envs/cellcharter-env/bin/python",
    "/workspace/epic-fermat/agent/tools/cellcharter_worker.py",
)

mcp = create_mcp(TOOL_NAME)


def _search_range_error(n_clusters_min: int, n_clusters_max: int, max_runs: int) -> str:
    """Why ClusterAutoK cannot run this K search, or "" -- mirrors ``cellcharter_worker.search_range_error``.

    Checked here too so a search upstream cannot finish is refused before the worker loads the data:
    ``max_runs=1`` fits every K and then fails in ``predict``; ``n_clusters_min > n_clusters_max``
    fails inside ``fit`` with "range() arg 3 must not be zero".
    """
    problems = []
    if int(max_runs) < 2:
        problems.append(
            f"max_runs={max_runs}: ClusterAutoK selects K by the stability between repeated runs, so it "
            "needs max_runs >= 2"
        )
    if int(n_clusters_max) < 2 or int(n_clusters_max) < int(n_clusters_min):
        problems.append(
            f"n_clusters_min={n_clusters_min}, n_clusters_max={n_clusters_max}: ClusterAutoK chooses K from "
            "[max(2, n_clusters_min), n_clusters_max], so n_clusters_max must be >= 2 and >= n_clusters_min"
        )
    return "; ".join(problems)


@mcp.tool()
def cellcharter_cluster_spatial_domains(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    use_rep: str = "X_scVI",
    n_layers: int = 3,
    n_clusters_min: int = 3,
    n_clusters_max: int = 12,
    max_runs: int = 5,
    convergence_tol: float = 0.001,
    cluster_key: str = "cluster_cellcharter",
    library_key: str = "",
    hvg_flavor: str = "seurat_v3",
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run CellCharter spatial clustering (spatial domain identification) on a
    spatial transcriptomics AnnData (.h5ad).

    Pipeline: a squidpy spatial graph on obsm[spatial_key] (hexagonal grid neighbours when obs has
    array_row/array_col, a generic 6-nearest-neighbour graph otherwise; one graph per library when
    library_key is given), CellCharter's aggregate_neighbors over n_layers hops, then ClusterAutoK,
    which fits every K in the range max_runs times and keeps the most stable one.

    Spots with obs['in_tissue'] == 0 (background outside the tissue, e.g. the glass of a CELLxGENE
    Visium export) are left out before the graph is built and reported under params.in_tissue_filter
    with a warning; data.n_spots is the slide supplied and data.n_spots_used the spots clustered.

    Representation: obsm[use_rep] if present, else obsm['X_scVI'], else obsm['X_pca'], else the
    worker computes a PCA itself (normalize_total 1e4 + log1p of X, the 2,000 most highly variable
    genes by hvg_flavor, scale, 30 components). Every step but the seurat_v3 ranking runs on X, so
    X should hold raw counts: an X with negative or NaN values (scaled data) is refused on that
    branch, and one with fractional values (log-normalised) runs with a warning (params.preprocessing,
    params.x_matrix_kind); use_raw_counts=True normalises the counts in adata.raw.X instead
    (params.expression_source). The 2,000-gene cap is fixed inside the tool; the
    payload then reports the supplied panel as n_genes and the analysed panel as n_genes_used, with
    a warning naming the cut. params.use_rep names the key CellCharter actually aggregated,
    params.use_rep_requested the one asked for, and params.use_rep_computed whether the worker
    built it; a warning says so whenever the requested key was not the one used.

    Parameters
    ----------
    st_h5ad:
        Path to the spatial transcriptomics AnnData (.h5ad).
    output_dir:
        Directory where all results will be written.
    spatial_key:
        Key in adata.obsm containing spatial coordinates; both the grid and the generic graph are
        built from it.
    use_rep:
        Key in adata.obsm for the low-dimensional representation. When absent, the worker uses
        X_scVI, then X_pca, then computes its own PCA (see above); params.use_rep reports which.
    n_layers:
        Number of neighborhood aggregation layers.
    n_clusters_min, n_clusters_max:
        Range of number of clusters for ClusterAutoK. K is chosen from
        [max(2, n_clusters_min), n_clusters_max]; n_clusters_max must be >= 2 and >= n_clusters_min.
        params.k_selected reports the chosen K.
    max_runs:
        Maximum number of runs per K for ClusterAutoK; at least 2, because K is chosen by the
        stability between runs.
    convergence_tol:
        Stability convergence tolerance for ClusterAutoK.
    cluster_key:
        Column name in adata.obs to store cluster labels.
    library_key:
        Optional obs column specifying sample/library ID. It must name an existing obs column (the
        run stops otherwise); the spatial graph is built per library (categories no spot carries
        are dropped first) and CellCharter aggregates per sample. Leave empty for a single sample.
    hvg_flavor:
        scanpy highly_variable_genes flavour used only when the worker builds the PCA itself:
        'seurat_v3' (default; ranks the raw counts in layers['counts'], or a copy of X taken before
        normalisation when that layer is absent, and needs scikit-misc -- a missing package stops
        the run rather than switching flavour), 'seurat' or 'cell_ranger' (log-normalised
        dispersion). Ignored, and listed in params.ignored, when the representation comes from obsm.
    use_raw_counts:
        Used only when the worker builds the PCA itself: False (default) treats X as the counts; True
        normalises adata.raw.X instead, for an h5ad whose X is log-normalised or scaled and whose raw
        counts sit in adata.raw (a file without adata.raw, or whose raw is not counts, is refused).
        Ignored, and listed in params.ignored, when the representation comes from obsm.
    """
    problem = _search_range_error(n_clusters_min, n_clusters_max, max_runs)
    if problem:
        return {"status": "error", "tool": TOOL_NAME, "error": problem}

    args = [
        "--task",
        "clustering",
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--spatial-key",
        spatial_key,
        "--use-rep",
        use_rep,
        "--n-layers",
        str(n_layers),
        "--n-clusters-min",
        str(n_clusters_min),
        "--n-clusters-max",
        str(n_clusters_max),
        "--max-runs",
        str(max_runs),
        "--convergence-tol",
        str(convergence_tol),
        "--cluster-key",
        cluster_key,
        "--hvg-flavor",
        str(hvg_flavor),
    ]
    if library_key:
        args.extend(["--library-key", library_key])
    if use_raw_counts:
        args.append("--use-raw-counts")

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
