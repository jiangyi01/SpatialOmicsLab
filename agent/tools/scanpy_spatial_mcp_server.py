#!/usr/bin/env python3
"""Scanpy spatial domain MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "scanpy-spatial"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SCANPY_SPATIAL",
    "/opt/conda/envs/SpaGCN/bin/python",
    "/workspace/epic-fermat/agent/tools/scanpy_spatial_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_scanpy_spatial_domain(
    data_path: str,
    output_dir: str,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    min_counts: int = 0,
    target_n_clusters: int = 0,
    n_top_genes: int = 2000,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Scanpy Leiden clustering of a slide (normalize → log1p → HVG → scale → PCA → kNN
    graph → Leiden → UMAP), published as spatial domains.

    Spots with obs['in_tissue'] == 0 (background outside the tissue) are left out first;
    data.n_spots is the count supplied, data.n_spots_used the count clustered, and
    params.in_tissue_filter with a warning says how many were left out.

    The kNN graph is built on the first `n_pcs` components of the expression PCA only
    (also for panels of 50 genes or fewer, where scanpy would otherwise use X directly):
    spatial coordinates are NOT used in clustering, they are read solely to draw the
    spatial figure. The payload states this in params.method (params.used_fallback is
    false: this is the tool's only implementation). sc.pp.scale densifies the
    HVG-subset matrix (n_spots x n_top_genes), which is intrinsic to scaling; the worker
    estimates that working set first and stops with the numbers, naming n_top_genes,
    when it exceeds the memory available (the smaller of the machine's free memory and the
    room left under a container's memory limit). n_pcs must be below the smaller of the spot
    and kept-gene counts, or the run stops saying so.

    X is expected to hold raw counts: normalize_total + log1p are always applied. A
    matrix with negative or NaN values (scaled / z-scored data) is refused before
    anything runs, and the error says whether adata.raw holds counts. A non-integer
    non-negative matrix (already normalised) is normalised again; the run proceeds with a
    warning that says so. use_raw_counts=True clusters adata.raw.X instead (for a
    CELLxGENE-style h5ad whose X is processed and whose counts sit in adata.raw; refused
    when there is no adata.raw or it does not hold counts). params.expression_source
    ("X" or "raw.X") and params.x_matrix_kind say which matrix ran and what X held. A run
    that fails still returns the warnings it had raised before the failure.

    Only the top `n_top_genes` highly variable genes are clustered on; every other
    gene is discarded, after the in-tissue filter. Both reductions are reported in the
    payload (n_genes/n_genes_used, n_spots/n_spots_used).

    For benchmark datasets you MUST pass `target_n_clusters` (the integer
    GT-cluster count given in the prompt). The worker then binary-searches the Leiden
    resolution (up to 25 Leiden runs) for the cluster count closest to the target; an
    exact match is not guaranteed, and the run says what happened: params.resolution_used
    is the resolution that ran, params.resolution_search records the target, the count
    achieved, whether it matched and every trial, summary.n_domains_requested sits beside
    summary.n_domains, and a warning is raised when the target was not reached. With
    `target_n_clusters` > 0 the `resolution` parameter has no effect and is listed in
    params.ignored. Leaving `target_n_clusters` at 0 clusters at `resolution`, which on
    Visium typically produces hundreds of micro-clusters and drives ARI toward zero
    (real measurement: target_n_clusters=7 → ARI≈0.16, resolution=1.0 → ARI≈0.01 on the
    DLPFC 7-layer GT).

    Output: scanpy_spatial_domains.h5ad with obs['spatial_domain'] (the search's trial
    columns are not kept), spatial_domains_summary.csv (spatial_domain, n_spots), and
    the UMAP / spatial PNGs.
    """
    args = [
        "--data-path",
        data_path,
        "--output-dir",
        output_dir,
        "--resolution",
        str(resolution),
        "--n-neighbors",
        str(n_neighbors),
        "--n-pcs",
        str(n_pcs),
        "--min-counts",
        str(min_counts),
        "--target-n-clusters",
        str(int(target_n_clusters)),
        "--n-top-genes",
        str(int(n_top_genes)),
    ]
    if use_raw_counts:
        args.append("--use-raw-counts")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
