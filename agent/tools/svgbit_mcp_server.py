#!/usr/bin/env python3
"""SVGBit MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_json

TOOL_NAME = "svgbit"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SVGBIT",
    "/opt/conda/envs/svgbit/bin/python",
    "/workspace/epic-fermat/agent/tools/svgbit_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def svgbit_run(
    input_mode: str = "visium_10x",
    counts_h5: str | None = None,
    spatial_dir: str | None = None,
    adata_path: str | None = None,
    output_dir: str = default_output_dir("svgbit_out"),
    k: int = 6,
    max_genes: int = 2000,
    min_counts: int = 1,
    low_variance_var: float = 0.0,
    quantile: float = 0.99,
    normalize: bool = True,
    n_svgs: int = 1000,
    n_svg_clusters: int = 5,
    cores: int = 1,
    top_k_genes: int = 20,
    random_seed: int = 0,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run the SVGBit pipeline via svgbit_worker.py.

    n_svgs (default 1000, svgbit's own default) is how many top-AI genes svgbit clusters into
    svg_cluster.csv. It does not change the AI ranking in AI.csv / svg_ranked.csv: svgbit computes
    AI before its clustering step. Seeded runs of one slide at n_svgs=1000 and n_svgs=200 wrote
    AI.csv tables that agree to 1e-16 with the same gene order, so an F1 difference seen between two
    n_svgs settings on an AI.csv-scored benchmark (an earlier note here recorded 0.26 at 1000 and
    0.18 at 1980 on visium_svg) did not come from n_svgs.

    SVGBit calls spatially variable genes from esda's conditional-randomization
    Moran's I, which draws 999 permutations and takes no seed of its own, so
    random_seed seeds numpy's global RNG before the library runs. Two unseeded
    runs of one slide differed by up to 0.215 in AI (a score bounded in [0, 1])
    and returned different gene lists; under a fixed seed they were identical.
    The seed reproduces a run only at the same `cores`: the worker pool chunks
    genes by worker count, so seed=0 at cores=3 diverged from cores=2 by 0.166.

    Modes (there is no CSV input mode):
    - input_mode=visium_10x: require counts_h5 + spatial_dir. Coordinates are the tissue-positions
      table's pxl_col_in_fullres / pxl_row_in_fullres.
    - input_mode=h5ad: require adata_path. Coordinates are read from obs['x'] / obs['y'], else
      obs['center_x'] / obs['center_y'], else a two-column obsm['spatial'] (a three-column one is
      refused, not flattened). params.coord_source says which was used.

    Genes are cut in this order before SVGBit ranks anything, and every cut is reported in
    params.n_genes_after_*, in data.n_genes_used (the genes actually ranked) and in a warning:
    1. min_counts: genes non-zero in fewer than min_counts spots are dropped, whatever max_genes is
       (0 keeps every gene).
    2. max_genes > 0: the max_genes highest-variance survivors are kept (0 keeps them all).
    3. low_variance_var > 0: svgbit.filters.low_variance_filter keeps genes whose variance is above it.
    4. quantile < 1: svgbit.filters.quantile_filter keeps genes whose mean is strictly below that
       quantile of all gene means, so the default 0.99 drops the ~1% most highly expressed genes.
       quantile=1.0 skips the filter (upstream's strict '<' would still drop the top gene at 1.0);
       quantile must lie in (0, 1].
    A requested filter or normalize=True that the installed svgbit cannot perform stops the run
    rather than being skipped. AI.csv and svg_ranked.csv rank every analysed gene.

    Spots: background spots -- obs['in_tissue'] == 0 in an h5ad, in_tissue == 0 in the tissue-positions
    table in visium_10x mode -- are left out before anything is ranked (CELLxGENE Visium exports carry
    every array spot, 56-70% of them background on the library's samples). data.n_spots is the count
    ranked; params.in_tissue_filter (n_spots_supplied, n_spots_off_tissue_dropped), a warning and the
    analysis say how many were left out. Without an in_tissue flag every spot is ranked.

    Memory: svgbit's density step builds a dense n_spots x n_spots neighbour matrix (8 * n_spots^2
    bytes per copy: two in the parent, one per `cores` worker). The worker estimates this before
    the run (available memory counts the page cache as reclaimable) and stops with the numbers when it
    cannot fit; it never drops a tissue spot to make it fit.

    Matrix: normalize=True (the default) runs svgbit's log-CPM normalizer, which treats the matrix as
    counts. A matrix with negative or non-finite values (scaled / z-scored data) is then refused, naming
    use_raw_counts when adata.raw holds counts, and a non-integer (already normalised) matrix runs with a
    warning that it was normalised twice. normalize=False ranks the matrix as supplied, so normalised
    or scaled data is accepted there (non-finite values are still refused). params.expression_source
    and params.x_matrix_kind say which matrix ran and what X held.

    use_raw_counts (default False): rank adata.raw.X instead of X (h5ad mode). Use it when X is
    log-normalised or scaled and the integer counts sit in adata.raw (CELLxGENE exports); refused when
    the file has no adata.raw or it does not hold counts. In visium_10x mode the count matrix is counts
    already, so it is listed in params.ignored.

    top_k_genes must be 0 or more.
    """
    payload: dict[str, Any] = {
        "__tool__": "svgbit_run",
        "input_mode": input_mode,
        "output_dir": output_dir,
        "k": int(k),
        "max_genes": int(max_genes),
        "min_counts": int(min_counts),
        "low_variance_var": float(low_variance_var),
        "quantile": float(quantile),
        "normalize": bool(normalize),
        "n_svgs": int(n_svgs),
        "n_svg_clusters": int(n_svg_clusters),
        "cores": int(cores),
        "top_k_genes": int(top_k_genes),
        "random_seed": int(random_seed),
        "use_raw_counts": bool(use_raw_counts),
    }

    if input_mode == "visium_10x":
        if not counts_h5 or not spatial_dir:
            raise ValueError("For input_mode=visium_10x, counts_h5 and spatial_dir are required.")
        payload["counts_h5"] = counts_h5
        payload["spatial_dir"] = spatial_dir
    elif input_mode == "h5ad":
        if not adata_path:
            raise ValueError("For input_mode=h5ad, adata_path is required.")
        payload["adata_path"] = adata_path
    else:
        raise ValueError(f"Unsupported input_mode={input_mode!r}. Valid input_mode values: 'visium_10x', 'h5ad'.")

    return run_worker_json(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, payload)


if __name__ == "__main__":
    mcp.run()
