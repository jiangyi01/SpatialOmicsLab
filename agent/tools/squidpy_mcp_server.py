#!/usr/bin/env python3
"""Squidpy spatial analysis MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli
from worker_utils import pin_blas_threads

TOOL_NAME = "squidpy"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SQUIDPY",
    "/opt/conda/envs/moscot/bin/python",
    "/workspace/epic-fermat/agent/tools/squidpy_worker.py",
)

mcp = create_mcp(TOOL_NAME)


def _frame_argv(coords_key: str, dims: int, section_key: str | None) -> list[str]:
    """The worker flags for the coordinate frame: the key, 2D or 3D, and the section column."""
    argv = ["--dims", str(int(dims))]
    if coords_key and coords_key != "spatial":
        argv += ["--coords-key", coords_key]
    if section_key:
        argv += ["--section-key", section_key]
    return argv


@mcp.tool()
def squidpy_spatial_neighbors(
    h5ad_path: str,
    output_dir: str,
    cluster_key: str = "cluster",
    coord_type: str = "generic",
    n_neighs: int = 6,
    n_rings: int = 1,
    delaunay: bool = False,
    spatial_key: str = "spatial",
    coords_key: str = "spatial",
    dims: int = 2,
    section_key: str | None = None,
) -> dict[str, Any]:
    """
    Build a spatial neighborhood graph using squidpy.gr.spatial_neighbors.

    This is a prerequisite step for most other squidpy spatial analyses.
    Spots that obs['in_tissue'] marks 0 (background outside the tissue: CELLxGENE Visium exports
    keep every array spot) are left out before the graph is built, so the written h5ad holds the
    in-tissue spots only; params.in_tissue_filter, a warning and a NOTE in the analysis say how
    many, and every count in the payload is after that cut.
    The four graph knobs are echoed in params as passed; the ones squidpy does not read under the
    chosen coord_type / delaunay (n_rings off the grid path, n_neighs with delaunay=True) are listed
    in params.ignored with a warning, and summary.graph names the graph that was built.

    coords_key: the obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
    'spatial_3d_aligned'). dims: 2 or 3; 3 builds the graph in the aligned frame in micrometres and
    needs a frame with recorded units and a measured or registered z. section_key: the obs column
    naming sections; required for a 2D run on a multi-section file (the run is per section) and for
    the cross-section edge count of a 3D run.

    Parameters
    ----------
    h5ad_path:
        Path to AnnData (.h5ad) with spatial coordinates in obsm.
    output_dir:
        Directory to write outputs (annotated h5ad with spatial graph).
    cluster_key:
        Column in adata.obs containing cell type / cluster labels. The graph does not depend on
        it and a missing column is not an error; when present it is stored as categorical in the
        written h5ad, ready for the label-consuming squidpy tools.
    coord_type:
        Coordinate type: 'generic' or 'grid' (for Visium-like data).
    n_neighs:
        Number of neighbors for the spatial graph ('generic': nearest neighbours; 'grid': neighbouring
        tiles per ring, 6 on a Visium hexagonal array). Not read when delaunay=True, and then
        reported in params.ignored.
    n_rings:
        Number of rings of neighbors, read only when coord_type='grid'. Any value other than 1 on a
        'generic' graph is not used and is reported in params.ignored.
    delaunay:
        Whether to use Delaunay triangulation for building the graph; each spot's neighbours are
        then set by the triangulation, not by n_neighs.
    spatial_key:
        Key in adata.obsm containing spatial coordinates.
    coords_key:
        The obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
        'spatial_3d_aligned'). The older spatial_key names the same thing; set one of them.
    dims:
        2 or 3; 3 builds the graph in the aligned frame in micrometres and needs a frame with recorded
        units and a measured or registered z. The result says which in params.mode ("3d",
        "per-section-2d" or "2d") and params.frame (coords_key, dims, units_per_axis_um, z_source,
        section_key, sections).
    section_key:
        The obs column naming sections; required for a 2D run on a multi-section file (the run is per
        section) and for the cross-section edge count of a 3D run (data.cross_section_edge_fraction).
    """
    args = [
        "--task",
        "spatial_neighbors",
        "--h5ad-path",
        h5ad_path,
        "--output-dir",
        output_dir,
        "--cluster-key",
        cluster_key,
        "--coord-type",
        coord_type,
        "--n-neighs",
        str(n_neighs),
        "--n-rings",
        str(n_rings),
        "--spatial-key",
        spatial_key,
    ]
    if delaunay:
        args.append("--delaunay")
    args += _frame_argv(coords_key, dims, section_key)
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def squidpy_nhood_enrichment(
    h5ad_path: str,
    output_dir: str,
    cluster_key: str = "cluster",
    coord_type: str = "generic",
    n_neighs: int = 6,
    n_perms: int = 1000,
    spatial_key: str = "spatial",
    seed: int = 0,
    drop_unlabeled: bool = False,
    coords_key: str = "spatial",
    dims: int = 2,
    section_key: str | None = None,
) -> dict[str, Any]:
    """
    Run neighborhood enrichment analysis using squidpy. Builds spatial neighbors
    internally, then computes neighborhood enrichment z-scores between cell types.
    Writes both the z-score matrix and the matrix of observed neighbor-pair counts
    each z was computed from. z-scores are taken against a global permutation of
    cluster labels, so read them with the counts. The top enriched/depleted pairs are
    ranked over finite off-diagonal z-scores only; a category that labels no cell has a
    NaN row (named in summary.empty_categories).
    Spots that obs['in_tissue'] marks 0 (background outside the tissue: CELLxGENE Visium exports
    keep every array spot) are left out before the analysis; params.in_tissue_filter, a warning and
    a NOTE in the analysis say how many, and every count in the payload is after that cut.
    A label carried only by those background spots is removed with them
    (summary.labels_only_off_tissue).

    coords_key: the obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
    'spatial_3d_aligned'). dims: 2 or 3; 3 builds the graph in the aligned frame in micrometres and
    needs a frame with recorded units and a measured or registered z. section_key: the obs column
    naming sections; required for a 2D run on a multi-section file (the run is per section) and for
    the cross-section edge count of a 3D run.

    Parameters
    ----------
    h5ad_path:
        Path to AnnData (.h5ad) with spatial coordinates.
    output_dir:
        Directory for output files (z-score CSV, count CSV, annotated h5ad).
    cluster_key:
        Column in adata.obs with categorical cell type / cluster labels. The default 'cluster'
        is absent from the library samples, so pass the real column (e.g. 'cell_type').
    coord_type:
        Coordinate type: 'generic' or 'grid'.
    n_neighs:
        Number of neighbors for the spatial graph.
    n_perms:
        Number of permutations for the enrichment test.
    spatial_key:
        Key in adata.obsm containing spatial coordinates.
    seed:
        Random seed for reproducibility.
    drop_unlabeled:
        A cell whose label is missing (NaN / empty / the string 'nan') is not a class. By default
        such a cell stops the run with the count; True leaves those cells out and the payload
        reports n_cells_dropped_unlabeled and says so in the analysis.
    coords_key:
        The obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
        'spatial_3d_aligned'). The older spatial_key names the same thing; set one of them.
    dims:
        2 or 3; 3 builds the graph in the aligned frame in micrometres and needs a frame with recorded
        units and a measured or registered z. The result says which in params.mode ("3d",
        "per-section-2d" or "2d") and params.frame (coords_key, dims, units_per_axis_um, z_source,
        section_key, sections).
    section_key:
        The obs column naming sections; required for a 2D run on a multi-section file (the run is per
        section) and for the cross-section edge count of a 3D run (data.cross_section_edge_fraction).
    """
    args = [
        "--task",
        "nhood_enrichment",
        "--h5ad-path",
        h5ad_path,
        "--output-dir",
        output_dir,
        "--cluster-key",
        cluster_key,
        "--coord-type",
        coord_type,
        "--n-neighs",
        str(n_neighs),
        "--n-perms",
        str(n_perms),
        "--spatial-key",
        spatial_key,
        "--seed",
        str(seed),
    ]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    args += _frame_argv(coords_key, dims, section_key)
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def squidpy_co_occurrence(
    h5ad_path: str,
    output_dir: str,
    cluster_key: str = "cluster",
    spatial_key: str = "spatial",
    n_steps: int = 50,
    drop_unlabeled: bool = False,
    coords_key: str = "spatial",
    dims: int = 2,
    section_key: str | None = None,
    max_estimated_s: float | None = None,
) -> dict[str, Any]:
    """
    Compute cell type co-occurrence probability at varying spatial distances
    using squidpy.gr.co_occurrence.
    Spots that obs['in_tissue'] marks 0 (background outside the tissue: CELLxGENE Visium exports
    keep every array spot) are left out before the analysis; params.in_tissue_filter, a warning and
    a NOTE in the analysis say how many, and every count in the payload is after that cut.
    A label carried only by those background spots is removed with them
    (summary.labels_only_off_tissue).

    coords_key: the obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
    'spatial_3d_aligned'). dims: 2 or 3; 3 builds the graph in the aligned frame in micrometres and
    needs a frame with recorded units and a measured or registered z. section_key: the obs column
    naming sections; required for a 2D run on a multi-section file (the run is per section) and for
    the cross-section edge count of a 3D run.

    Parameters
    ----------
    h5ad_path:
        Path to AnnData (.h5ad) with spatial coordinates.
    output_dir:
        Directory for output files (co-occurrence CSV, annotated h5ad).
    cluster_key:
        Column in adata.obs with categorical cell type / cluster labels. The default 'cluster'
        is absent from the library samples, so pass the real column (e.g. 'cell_type').
    spatial_key:
        Key in adata.obsm containing spatial coordinates.
    n_steps:
        Number of distance thresholds squidpy evaluates. Co-occurrence is computed per bin between
        consecutive thresholds, so the mean CSV averages over n_steps - 1 bins; the payload reports
        both counts (n_distance_thresholds, n_distance_bins).
    drop_unlabeled:
        A cell whose label is missing (NaN / empty / the string 'nan') is not a class. By default
        such a cell stops the run with the count; True leaves those cells out and the payload
        reports n_cells_dropped_unlabeled and says so in the analysis.
    coords_key:
        The obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
        'spatial_3d_aligned'). The older spatial_key names the same thing; set one of them.
    dims:
        2 or 3; 3 builds the graph in the aligned frame in micrometres and needs a frame with recorded
        units and a measured or registered z. The result says which in params.mode ("3d",
        "per-section-2d" or "2d") and params.frame (coords_key, dims, units_per_axis_um, z_source,
        section_key, sections).
    section_key:
        The obs column naming sections; required for a 2D run on a multi-section file (the run is per
        section) and for the cross-section edge count of a 3D run (data.cross_section_edge_fraction).
    max_estimated_s:
        The wall-time budget, in seconds, the run's estimate must fit. squidpy scans every pair of cells at each
        distance bin, so a run costs about cells^2 x (n_steps - 1) per plane (3D on 309,599 cells took 7,001 s);
        the cost is estimated before squidpy starts and reported as params.cost_estimate, and a run estimated
        over the budget is refused with the estimate and the ways to bring it down (fewer cells: a section or
        bounding-box subset, or per section; fewer intervals: n_steps). Default: the host's
        SOG_SQUIDPY_COOCCURRENCE_MAX_SECONDS, else 1800 s. 0 or less means no budget.
    """
    args = [
        "--task",
        "co_occurrence",
        "--h5ad-path",
        h5ad_path,
        "--output-dir",
        output_dir,
        "--cluster-key",
        cluster_key,
        "--spatial-key",
        spatial_key,
        "--n-steps",
        str(n_steps),
    ]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    if max_estimated_s is not None:
        args += ["--max-estimated-s", str(max_estimated_s)]
    args += _frame_argv(coords_key, dims, section_key)
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def squidpy_spatial_autocorr(
    h5ad_path: str,
    output_dir: str,
    cluster_key: str = "cluster",
    coord_type: str = "generic",
    n_neighs: int = 6,
    mode: str = "moran",
    n_perms: int | None = 100,
    n_jobs: int = 1,
    spatial_key: str = "spatial",
    top_n: int = 200,
    pvalue_threshold: float = 0.05,
    seed: int = 0,
    use_fdr: bool = False,
    use_highly_variable: bool = False,
    coords_key: str = "spatial",
    dims: int = 2,
    section_key: str | None = None,
) -> dict[str, Any]:
    """
    Compute spatial autocorrelation (Moran's I or Geary's C) for gene expression
    using squidpy.gr.spatial_autocorr, and emit a curated SVG list.
    Spots that obs['in_tissue'] marks 0 (background outside the tissue: CELLxGENE Visium exports
    keep every array spot) are left out before the analysis; params.in_tissue_filter, a warning and
    a NOTE in the analysis say how many, and every count in the payload is after that cut.
    (Background spots carry ambient counts, so testing them scores the tissue-versus-glass contrast
    as spatial autocorrelation.)

    coords_key: the obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
    'spatial_3d_aligned'). dims: 2 or 3; 3 builds the graph in the aligned frame in micrometres and
    needs a frame with recorded units and a measured or registered z. section_key: the obs column
    naming sections; required for a 2D run on a multi-section file (the run is per section) and for
    the cross-section edge count of a 3D run.

    Parameters
    ----------
    h5ad_path:
        Path to AnnData (.h5ad) with spatial coordinates.
    output_dir:
        Directory for output files (autocorrelation CSV, annotated h5ad,
        predicted_genes.json with the top-N spatially variable genes).
    cluster_key:
        Accepted for signature compatibility only. This task tests genes, not labels:
        squidpy.gr.spatial_autocorr takes no label column and the worker never reads it.
        A value other than the default 'cluster' is reported under params.ignored with a warning.
    coord_type:
        Coordinate type: 'generic' or 'grid'.
    n_neighs:
        Number of neighbors for the spatial graph.
    mode:
        Autocorrelation metric: 'moran' (Moran's I) or 'geary' (Geary's C).
    n_perms:
        Number of permutations for the permutation test (default 100, seeded by `seed`). When set,
        the permutation p-values (CSV column pval_sim) drive the significance count and the SVG
        filter; the smallest attainable permutation p is 1/(n_perms + 1). Pass None to skip the
        permutation test: the analytic p-values (pval_norm) drive them instead. The payload names
        the column read in params.pvalue_column.
    n_jobs:
        Number of parallel jobs.
    spatial_key:
        Key in adata.obsm containing spatial coordinates.
    top_n:
        Maximum number of top spatially variable genes to write to
        predicted_genes.json (used by the SVG benchmarking pipeline).
        Genes are first filtered by pvalue_threshold, then ranked by Moran's I
        (descending) or Geary's C (ascending).
    pvalue_threshold:
        P-value cutoff applied before the top-N ranking; only genes with positive spatial
        autocorrelation (Moran's I above its expectation, Geary's C below 1) pass, because squidpy's
        one-tailed p is folded and a strongly negatively autocorrelated gene also gets a tiny p.
        Set to a non-positive value to disable filtering (every gene is then ranked).
    seed:
        Random seed for the permutation test (only used when n_perms is set).
    use_fdr:
        Apply Benjamini-Hochberg to the chosen p-value column before the significance count and
        the SVG filter (params.pvalue_correction says which applied). Adjusted over the finite
        p-values only: squidpy's own *_fdr_bh CSV columns turn entirely NaN when any gene has a NaN
        statistic (a constant gene), and the CSV is left as squidpy wrote it.
    use_highly_variable:
        Test only the genes flagged in adata.var['highly_variable']. Default False tests every gene
        (squidpy's own default would silently use the flag whenever the column exists). A subset
        is reported in params.genes_tested_source and as a reduction note in the analysis.
    coords_key:
        The obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
        'spatial_3d_aligned'). The older spatial_key names the same thing; set one of them.
    dims:
        2 or 3; 3 builds the graph in the aligned frame in micrometres and needs a frame with recorded
        units and a measured or registered z. The result says which in params.mode ("3d",
        "per-section-2d" or "2d") and params.frame (coords_key, dims, units_per_axis_um, z_source,
        section_key, sections).
    section_key:
        The obs column naming sections; required for a 2D run on a multi-section file (the run is per
        section) and for the cross-section edge count of a 3D run (data.cross_section_edge_fraction).
    """
    # squidpy.gr.spatial_autocorr farms the permutation test out to `n_jobs` joblib workers; if EACH
    # also lets its BLAS use every core that is n_jobs x n_cores threads, and the cores thrash
    # instead of computing (commit d1689ee measured load average ~328 and a stalled RCTD on a
    # 96-core box). The default of 1 makes that harmless today, but n_jobs is the agent's to set.
    # Pin BLAS to one thread per worker so squidpy's own parallelism is the only parallelism. Must
    # precede the worker launch: OpenBLAS reads the count once, when it is loaded, and the
    # subprocess inherits os.environ.
    pin_blas_threads()

    args = [
        "--task",
        "spatial_autocorr",
        "--h5ad-path",
        h5ad_path,
        "--output-dir",
        output_dir,
        "--cluster-key",
        cluster_key,
        "--coord-type",
        coord_type,
        "--n-neighs",
        str(n_neighs),
        "--mode",
        mode,
        "--n-jobs",
        str(n_jobs),
        "--spatial-key",
        spatial_key,
        "--top-n",
        str(top_n),
        "--pvalue-threshold",
        str(pvalue_threshold),
        "--seed",
        str(seed),
    ]
    if n_perms is not None:
        args.extend(["--n-perms", str(n_perms)])
    if use_fdr:
        args.append("--use-fdr")
    if use_highly_variable:
        args.append("--use-highly-variable")
    args += _frame_argv(coords_key, dims, section_key)
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def squidpy_ripley(
    h5ad_path: str,
    output_dir: str,
    cluster_key: str = "cluster",
    mode: str = "L",
    spatial_key: str = "spatial",
    n_steps: int = 50,
    n_simulations: int = 100,
    drop_unlabeled: bool = False,
    coords_key: str = "spatial",
    dims: int = 2,
    section_key: str | None = None,
) -> dict[str, Any]:
    """
    Compute Ripley's statistics (F, G, or L function) for spatial point patterns
    using squidpy.gr.ripley. The annotated h5ad is the input with the result added
    under uns['<cluster_key>_ripley_<mode>']; its obs columns and obsm are left as they were.
    Spots that obs['in_tissue'] marks 0 (background outside the tissue: CELLxGENE Visium exports
    keep every array spot) are left out before the analysis; params.in_tissue_filter, a warning and
    a NOTE in the analysis say how many, and every count in the payload is after that cut.
    A label carried only by those background spots is removed with them
    (summary.labels_only_off_tissue).

    coords_key: the obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
    'spatial_3d_aligned'). dims: 2 or 3; 3 builds the graph in the aligned frame in micrometres and
    needs a frame with recorded units and a measured or registered z. section_key: the obs column
    naming sections; required for a 2D run on a multi-section file (the run is per section) and for
    the cross-section edge count of a 3D run.

    Parameters
    ----------
    h5ad_path:
        Path to AnnData (.h5ad) with spatial coordinates.
    output_dir:
        Directory for output files (Ripley's statistics CSV, annotated h5ad).
    cluster_key:
        Column in adata.obs with categorical cell type / cluster labels. The default 'cluster'
        is absent from the library samples, so pass the real column (e.g. 'cell_type').
    mode:
        Ripley's function type: 'F', 'G', or 'L'.
    spatial_key:
        Key in adata.obsm containing spatial coordinates.
    n_steps:
        Number of distance steps for Ripley's function evaluation.
    n_simulations:
        Number of simulations for the confidence envelope.
    drop_unlabeled:
        A cell whose label is missing (NaN / empty / the string 'nan') is not a class. By default
        such a cell stops the run with the count; True leaves those cells out and the payload
        reports n_cells_dropped_unlabeled and says so in the analysis.
    coords_key:
        The obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
        'spatial_3d_aligned'). The older spatial_key names the same thing; set one of them.
    dims:
        2 or 3; 3 builds the graph in the aligned frame in micrometres and needs a frame with recorded
        units and a measured or registered z. The result says which in params.mode ("3d",
        "per-section-2d" or "2d") and params.frame (coords_key, dims, units_per_axis_um, z_source,
        section_key, sections).
    section_key:
        The obs column naming sections; required for a 2D run on a multi-section file (the run is per
        section) and for the cross-section edge count of a 3D run (data.cross_section_edge_fraction).
    """
    args = [
        "--task",
        "ripley",
        "--h5ad-path",
        h5ad_path,
        "--output-dir",
        output_dir,
        "--cluster-key",
        cluster_key,
        "--mode",
        mode,
        "--spatial-key",
        spatial_key,
        "--n-steps",
        str(n_steps),
        "--n-simulations",
        str(n_simulations),
    ]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    args += _frame_argv(coords_key, dims, section_key)
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def squidpy_centrality_scores(
    h5ad_path: str,
    output_dir: str,
    cluster_key: str = "cluster",
    coord_type: str = "generic",
    n_neighs: int = 6,
    spatial_key: str = "spatial",
    drop_unlabeled: bool = False,
    coords_key: str = "spatial",
    dims: int = 2,
    section_key: str | None = None,
) -> dict[str, Any]:
    """
    Compute graph centrality scores (closeness, degree, clustering coefficient)
    for cell types using squidpy.gr.centrality_scores. A category of cluster_key that
    labels no cell has nothing to score and is left out (named in
    summary.empty_categories_removed).
    Spots that obs['in_tissue'] marks 0 (background outside the tissue: CELLxGENE Visium exports
    keep every array spot) are left out before the analysis; params.in_tissue_filter, a warning and
    a NOTE in the analysis say how many, and every count in the payload is after that cut.
    A label carried only by those background spots is removed with them
    (summary.labels_only_off_tissue).

    coords_key: the obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
    'spatial_3d_aligned'). dims: 2 or 3; 3 builds the graph in the aligned frame in micrometres and
    needs a frame with recorded units and a measured or registered z. section_key: the obs column
    naming sections; required for a 2D run on a multi-section file (the run is per section) and for
    the cross-section edge count of a 3D run.

    Parameters
    ----------
    h5ad_path:
        Path to AnnData (.h5ad) with spatial coordinates.
    output_dir:
        Directory for output files (centrality CSV, annotated h5ad).
    cluster_key:
        Column in adata.obs with categorical cell type / cluster labels. The default 'cluster'
        is absent from the library samples, so pass the real column (e.g. 'cell_type').
    coord_type:
        Coordinate type: 'generic' or 'grid'.
    n_neighs:
        Number of neighbors for the spatial graph.
    spatial_key:
        Key in adata.obsm containing spatial coordinates.
    drop_unlabeled:
        A cell whose label is missing (NaN / empty / the string 'nan') is not a class. By default
        such a cell stops the run with the count; True leaves those cells out and the payload
        reports n_cells_dropped_unlabeled and says so in the analysis.
    coords_key:
        The obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
        'spatial_3d_aligned'). The older spatial_key names the same thing; set one of them.
    dims:
        2 or 3; 3 builds the graph in the aligned frame in micrometres and needs a frame with recorded
        units and a measured or registered z. The result says which in params.mode ("3d",
        "per-section-2d" or "2d") and params.frame (coords_key, dims, units_per_axis_um, z_source,
        section_key, sections).
    section_key:
        The obs column naming sections; required for a 2D run on a multi-section file (the run is per
        section) and for the cross-section edge count of a 3D run (data.cross_section_edge_fraction).
    """
    args = [
        "--task",
        "centrality_scores",
        "--h5ad-path",
        h5ad_path,
        "--output-dir",
        output_dir,
        "--cluster-key",
        cluster_key,
        "--coord-type",
        coord_type,
        "--n-neighs",
        str(n_neighs),
        "--spatial-key",
        spatial_key,
    ]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    args += _frame_argv(coords_key, dims, section_key)
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
