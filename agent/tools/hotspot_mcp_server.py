#!/usr/bin/env python3
"""Hotspot spatial modules MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli
from worker_utils import pin_blas_threads

TOOL_NAME = "hotspot"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "HOTSPOT",
    "/opt/conda/envs/hotspot/bin/python",
    "/workspace/epic-fermat/agent/tools/hotspot_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def hotspot_spatial_modules(
    st_h5ad: str,
    output_dir: str,
    layer_key: str = "",
    model: str = "danb",
    latent_obsm_key: str = "spatial",
    umi_counts_obs_key: str = "total_counts",
    n_neighbors: int = 30,
    autocorr_fdr: float = 0.05,
    min_gene_threshold: int = 30,
    module_fdr_threshold: float = 0.05,
    n_jobs: int = 4,
    n_top_modules: int = 3,
    seed: int = 0,
    round_counts: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Identify spatially variable genes and gene modules using Hotspot on a single
    spatial transcriptomics AnnData (.h5ad).

    Parameters
    ----------
    st_h5ad:
        Path to the spatial AnnData (.h5ad) file to analyze.
    output_dir:
        Directory to write Hotspot outputs (annotated h5ad, CSVs, PNGs).
    layer_key:
        AnnData layer holding the matrix Hotspot models. Empty (the default) analyses
        adata.X. A named layer that the file does not have is an error listing the layers
        it does have; the worker never substitutes X for a missing layer. The payload
        records what was analysed as params.expression_source ("X", "raw.X" with
        use_raw_counts, or "layers['<key>']").
    model:
        Background model for Hotspot ('danb', 'bernoulli', 'normal', 'none'). 'danb' and
        'bernoulli' are count models, and the worker checks the analysed matrix before
        running (hotspotsc itself does not): finite and non-negative for both, and
        integer-valued for 'danb' -- and for 'bernoulli' unless obs holds the per-cell UMI
        totals, since Bernoulli reads only detection (value > 0) from the matrix but uses the
        totals as trial counts. Log-normalised or scaled values are refused with a message
        naming the remedies: a raw-count layer_key, use_raw_counts when adata.raw holds counts,
        model='normal' / 'none', or round_counts. 'normal' and 'none' take normalised values by
        design and are never refused for them.
    latent_obsm_key:
        Name of the obsm key defining the similarity space (e.g. 'spatial' coordinates or
        'X_pca'). A key absent from obsm is an error; there is no fallback to 'spatial'.
        The key used is echoed as params.latent_obsm_key.
    umi_counts_obs_key:
        obs key containing UMI counts per cell (the size factor). If the column is absent,
        totals are the row sums of the analysed matrix -- Hotspot's own default -- and the
        payload says so in params.umi_counts_source.
    n_neighbors:
        Number of neighbors per cell in the Hotspot KNN graph.
    autocorr_fdr:
        FDR threshold to select informative genes from autocorrelation results. A value
        outside (0, 1) disables the cut: every gene goes on to local correlations and module
        building, and params.gene_selection says so.
    min_gene_threshold:
        min_gene_threshold for hs.create_modules.
    module_fdr_threshold:
        fdr_threshold for hs.create_modules.
    n_jobs:
        Number of parallel jobs.
    n_top_modules:
        Number of top modules (ranked by mean absolute score) drawn as spatial maps,
        hotspot_spatial_module_<m>.png; 0 draws none and a negative value is refused. On a
        CELLxGENE slide, whose uns['spatial'] holds its one library beside a scalar
        'is_single' key, that library is named to scanpy (and its own spot diameter used), so
        the map is drawn. A map that still cannot be drawn (no obsm['spatial'], or scanpy
        refuses) is named in a warning and the analysis with the reason;
        output_files.spatial_module_plots is the list of map paths written (top module
        first), summary.plotted_modules their module ids and summary.n_module_plots their
        count.
    seed:
        Random seed for reproducibility.
    round_counts:
        Round the analysed matrix to the nearest integer before a count model
        (danb / bernoulli). Off by default, so non-integer values are an error rather than
        silently modelled. Use it for counts stored as near-integer floats; the payload
        reports how many values were rounded (data.n_values_rounded) and warns.
        Ignored, and reported as ignored, under model='normal' / 'none'.
    use_raw_counts:
        Model adata.raw.X instead of adata.X. CELLxGENE Visium exports keep the integer
        counts in adata.raw beside a log-normalised or scaled X, which the danb / bernoulli
        count models refuse. Refused when the file has no adata.raw or adata.raw does not hold
        counts, and together with a non-empty layer_key. Off by default.

    Notes
    -----
    Background spots flagged obs['in_tissue'] == 0 (CELLxGENE Visium exports carry every
    array spot) are left out before anything is computed and reported in
    params.in_tissue_filter, data.n_spots_off_tissue_dropped, a warning and the analysis;
    data.n_spots counts the analysed spots and data.n_spots_input the spots supplied.
    Genes constant across all spots are removed before Hotspot runs (it refuses them);
    the filter is sparse-aware and runs on the matrix that is analysed. The payload
    reports data.n_genes_input beside data.n_genes (analysed) and
    data.n_genes_zero_variance_removed, and the analysis text notes any reduction.
    """
    # Hotspot hands `n_jobs` straight to multiprocessing.Pool(processes=jobs) -- once in
    # compute_autocorrelations, twice more in compute_local_correlations. At the shipped default
    # that is four child processes, and each imports numpy for its own solve, so an unpinned pool
    # asks for n_jobs x n_cores threads (4 x 96 = 384 on a 96-core box). That is the thrash that
    # made RCTD stall in init and time out (commit d1689ee, load average ~328). Pin BLAS to one
    # thread per worker so Hotspot's own parallelism is the only parallelism. The pin has to happen
    # here, not in the worker: Pool forks on Linux, so a child inherits whatever thread count the
    # parent's OpenBLAS already committed to at import. The subprocess inherits os.environ.
    pin_blas_threads()

    args = [
        "--task",
        "spatial_modules",
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--layer-key",
        layer_key or "",
        "--model",
        model,
        "--latent-obsm-key",
        latent_obsm_key,
        "--umi-counts-obs-key",
        umi_counts_obs_key,
        "--n-neighbors",
        str(n_neighbors),
        "--autocorr-fdr",
        str(autocorr_fdr),
        "--min-gene-threshold",
        str(min_gene_threshold),
        "--module-fdr-threshold",
        str(module_fdr_threshold),
        "--n-jobs",
        str(n_jobs),
        "--n-top-modules",
        str(n_top_modules),
        "--seed",
        str(seed),
    ]
    if round_counts:
        args.append("--round-counts")
    if use_raw_counts:
        args.append("--use-raw-counts")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
