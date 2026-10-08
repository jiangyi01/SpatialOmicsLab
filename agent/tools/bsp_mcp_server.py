#!/usr/bin/env python3
"""BSP (scbsp) spatially variable gene detection MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "bsp"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "BSP",
    "/opt/conda/envs/bsp/bin/python",
    "/workspace/epic-fermat/agent/tools/bsp_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def bsp_identify_svg(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    layer: str = "",
    top_k_genes: int = 200,
    pvalue_cutoff: float = 1.0,
    seed: int = 0,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Identify spatially variable genes using BSP (scbsp) on spatial transcriptomics data.

    scBSP (single-cell big-small patch) scores each gene by how the variance of its local means
    changes between a small and a large neighbourhood patch around every spot, fits one log-normal
    null to those scores across all tested genes, and returns a p-value per gene. It draws no random
    numbers. The expression matrix is kept sparse end to end (scbsp works on CSR throughout).

    Spots flagged ``obs['in_tissue'] == 0`` (Space Ranger / CELLxGENE exports can carry every array
    spot) are background and are left out before the test; the count is reported in
    ``data.n_spots_out_of_tissue_excluded``, ``params`` and the analysis text. Nothing is subsampled.

    Genes detected (value > 0) in fewer than 10 spots are not tested: their degenerate score would
    turn every p-value of the run into NaN. How many were left out is reported in
    ``params.n_genes_dropped_sparse`` and in the analysis text. If no gene reaches 10 spots, or if
    scbsp still returns NaN for every gene (too few genes for its null fit), the run stops with an
    error; no p-value is invented.

    Parameters
    ----------
    st_h5ad:
        Path to the spatial AnnData (.h5ad) file. Must contain spatial
        coordinates in obsm[spatial_key] and expression data in .X or a layer.
    output_dir:
        Directory to write BSP outputs (results CSV, top genes CSV, predicted_genes.json).
    spatial_key:
        Name of the obsm key containing spatial coordinates (default: 'spatial').
    layer:
        AnnData layer to use for expression. Empty string uses adata.X. scbsp.granp documents its
        input as the raw expression matrix: a matrix with negative or non-finite values (scaled /
        z-scored data) is refused, naming use_raw_counts when adata.raw holds counts, and a
        non-integer (normalised) matrix is tested as supplied with a warning.
        ``params.expression_source`` and ``params.x_matrix_kind`` say which matrix ran.
    top_k_genes:
        Maximum number of genes to write to predicted_genes.json and bsp_top_genes.csv (default 200;
        must be >= 0). When pvalue_cutoff<1.0 is used, the pool is filtered by significance first,
        then capped at top_k_genes. To match the manual BSP runner's behavior
        ("return ALL p<0.05 genes"), pass top_k_genes=10000 (or any value larger
        than the expected significant-gene count) together with pvalue_cutoff=0.05.
    pvalue_cutoff:
        When <1.0, restrict the prediction pool to genes with p<cutoff BEFORE
        taking top_k_genes; if no gene passes, the reported list is empty and a warning says so
        (the full ranking stays in bsp_results.csv). ``summary.n_significant`` counts genes with
        p < pvalue_cutoff. Default 1.0 disables the filter (top-K-only ranking) and
        ``summary.n_significant`` then counts p < 0.05. Must be > 0. Set pvalue_cutoff=0.05 to
        match the manual runner. The cutoff, whether it was applied and the threshold counted are
        echoed in ``params``.
    seed:
        Accepted and IGNORED: scbsp draws no random numbers (ball-tree neighbourhoods, sparse local
        means and a fitted log-normal null are deterministic). Listed under ``params.ignored``.
    use_raw_counts:
        Test adata.raw.X instead of X (default False). Use it when X is log-normalised or scaled and
        the integer counts sit in adata.raw (CELLxGENE exports); refused when the file has no
        adata.raw or it does not hold counts. A layer named alongside it is not read and is listed
        in ``params.ignored``.
    """
    args = [
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--spatial-key",
        spatial_key,
        "--layer",
        layer,
        "--top-k-genes",
        str(top_k_genes),
        "--pvalue-cutoff",
        str(pvalue_cutoff),
        "--seed",
        str(seed),
    ]
    if use_raw_counts:
        args.append("--use-raw-counts")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
