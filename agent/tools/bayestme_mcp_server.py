#!/usr/bin/env python3
"""BayesTME deconvolution MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_json

TOOL_NAME = "bayestme"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "BAYESTME",
    "/opt/conda/envs/bayestme/bin/python",
    "/workspace/epic-fermat/agent/tools/bayestme_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def bayestme_deconvolution(
    output_dir: str,
    input_mode: str = "visium_h5_spatial",
    spaceranger_dir: str = "",
    visium_h5_path: str = "",
    visium_spatial_dir: str = "",
    coord_type: str = "array",
    h5ad_path: str = "",
    layout: str = "",
    counts_h5ad_path: str = "",
    coords_csv: str = "",
    knn_k: int = 6,
    gene_filtering: dict[str, Any] | None = None,
    deconvolution: dict[str, Any] | None = None,
    marker_genes: dict[str, Any] | None = None,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """BayesTME deconvolution with flexible ST input loaders.

    Requires deconvolution={"n_components": <number of cell types>}; BayesTME cannot infer it and
    the worker refuses a missing or null value. Other deconvolution keys: rho (spatial smoothing
    strength; unset means BayesTME_VI's own 0.5, reported as params.rho), n_svi_steps (10000),
    n_samples (100), use_spatial_guide (True), seed (0).

    Modes (input_mode, default "visium_h5_spatial"):
    - visium_h5_spatial: visium_h5_path + visium_spatial_dir
    - spaceranger_outs: spaceranger_dir
    - h5ad: h5ad_path (needs obsm['spatial']; obs['in_tissue'] is optional, absent = all in tissue)
    - generic_counts_coords: counts_h5ad_path + coords_csv

    layout (input_mode="h5ad" only) is the spot lattice BayesTME builds its spatial prior on:
    "HEX" (Visium), "SQUARE" (a regular grid) or "IRREGULAR" (Slide-seq, MERFISH, Xenium and any
    other free-position platform). Left empty (the default), the h5ad's own uns['layout'] is used,
    and HEX when the file has none. The other modes derive it from the input.

    knn_k does not reach the model: BayesTME builds its own neighbour graph from the in-tissue
    positions and the layout, and knn_k only shapes obsp['connectivities'] in the saved h5ad. It
    is accepted and listed in params.ignored.

    gene_filtering.spot_threshold keeps genes detected in at most that fraction of in-tissue spots
    (it removes near-ubiquitous genes: 0.95 drops genes seen in more than 95% of spots).
    gene_filtering.filter_ribosomal_genes drops genes whose name matches BayesTME's ribosomal
    pattern ([Rr][Pp][SsLl]). The pattern is matched against var_names, or, when var_names are
    Ensembl IDs (a CELLxGENE h5ad), against a gene-symbol column of var (SYMBOL, gene_symbols,
    gene_name, GeneName, GeneName-2 or feature_name); params.gene_filtering names the column
    matched (matched_on), and a warning says so. Ensembl var_names with no symbol column match
    nothing, and a warning says that too.

    marker_genes.n_marker_genes (a whole number >= 1; 5.0 reads as 5; null means 10) and
    marker_genes.alpha (a number in (0, 1]; null means 0.05) are checked with marker_genes.method
    before the SVI run, not by the marker CLI after it; the values used are reported as
    params.marker_n_marker_genes and params.marker_alpha. marker_genes.method is BEST_AVAILABLE
    (default; ranks every gene and does not read alpha, so a given alpha is listed in
    params.ignored), FALSE_DISCOVERY_RATE or TIGHT. BayesTME's select_marker_genes CLI writes its
    gene list for every gene and stops as soon as one fails the cutoff, so TIGHT, and
    FALSE_DISCOVERY_RATE with alpha below 1, are refused before the SVI run.

    Only in-tissue spots are deconvolved (data.n_spots_used); out-of-tissue rows of the obsm
    results are zeros, and the spots left out are reported in params.in_tissue_filter.
    deconvolution_result.h5 omits the derived reads_trace array.

    BayesTME fits a Poisson likelihood, so the matrix it fits must hold raw counts (non-negative
    whole numbers). A log-normalised or scaled X is refused before the SVI run with what was
    found and whether adata.raw holds counts. use_raw_counts=True (input_mode "h5ad" or
    "generic_counts_coords" only; default False) fits adata.raw instead of X, e.g. for a
    CELLxGENE Visium h5ad, whose X is processed and whose counts are in adata.raw. The matrix
    fitted is reported as params.counts_source ("X" or "raw").
    """
    payload: dict[str, Any] = {
        "__tool__": "bayestme_deconvolution",
        "output_dir": output_dir,
        "input_mode": input_mode,
        "spaceranger_dir": spaceranger_dir,
        "visium_h5_path": visium_h5_path,
        "visium_spatial_dir": visium_spatial_dir,
        "coord_type": coord_type,
        "h5ad_path": h5ad_path,
        "layout": layout,
        "counts_h5ad_path": counts_h5ad_path,
        "coords_csv": coords_csv,
        "knn_k": knn_k,
        "gene_filtering": gene_filtering or {},
        "deconvolution": deconvolution or {},
        "marker_genes": marker_genes or {},
        "use_raw_counts": bool(use_raw_counts),
    }
    return run_worker_json(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, payload)


if __name__ == "__main__":
    mcp.run()
