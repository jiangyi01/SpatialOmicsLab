#!/usr/bin/env python3
"""SOMDE (spatially variable genes) MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_json

TOOL_NAME = "somde"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SOMDE",
    "/opt/conda/envs/somde/bin/python",
    "/workspace/epic-fermat/agent/tools/somde_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def somde_identify_svg(
    input_mode: str,
    output_dir: str,
    counts_h5: str | None = None,
    spatial_dir: str | None = None,
    h5ad_path: str | None = None,
    max_genes: int = 2000,
    min_counts: int = 1,
    som_dim: int = 20,
    top_k_genes: int = 20,
    random_seed: int = 0,
    allow_pixel_coord_fallback: bool = False,
    layer: str | None = None,
    round_counts: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Run SOMDE to identify spatially variable genes (SVGs) and save figures.

    Modes (input_mode, required):
    - visium_10x: counts_h5 + spatial_dir
    - h5ad: h5ad_path (spatial coordinates read from obsm["spatial"], which must have two columns,
      or from obs["x"]/obs["y"])

    max_genes caps how many genes are tested (default 2000); it drives the result more than any
    other setting here, so raise it when recall matters. Genes below min_counts total counts are
    dropped first. Both cuts are ours, not SOMDE's, and the payload reports them
    (params.gene_selection, a warning and the analysis text).

    som_dim is SOMDE's k: the average number of spots per SOM node. The SOM grid is
    int(sqrt(n_spots // som_dim)) nodes per side, so a larger value condenses more coarsely.

    Memory: SOMDE holds a dense genes x spots table (max_genes sets its gene count) and, in its
    Gaussian-process test, ten n_nodes x n_nodes float64 kernel eigenvector matrices at once plus
    working copies (som_dim sets the node count; at the default 20 a 500,000-spot VisiumHD slide
    gets 25,281 nodes and ~67 GiB of kernels). Both are estimated before SOMDE starts
    (params.memory_estimate_gib); a run that cannot fit is refused with both numbers and the knob
    for each.

    random_seed has no effect -- SOMDE draws no random numbers -- and is listed under
    params.ignored. allow_pixel_coord_fallback (visium_10x only): when array_row/array_col are
    constant the run stops unless this is True, in which case the full-resolution pixel columns
    are used and params.used_fallback says so.

    Spots flagged obs["in_tissue"] == 0 (background glass; CELLxGENE Visium exports carry every
    array spot, and in visium_10x mode the flag comes from the tissue-positions file) are left out
    before anything is computed and reported (params.in_tissue_filter, a warning, the analysis);
    data.n_spots is the count supplied and data.n_spots_used the count tested.

    SOMDE models raw counts, so every stored value is checked: NaN/inf or negative values are an
    error, and so are non-integer values (normalised data) unless round_counts=True, which rounds
    them and records it (params.rounded_to_integers, a warning). layer (h5ad mode) names a
    raw-count layer to analyse instead of X; an absent layer is an error. use_raw_counts=True
    (h5ad mode, with layer empty) analyses adata.raw.X instead -- CELLxGENE exports keep the counts
    there beside a processed X; it is refused when adata.raw is absent or not counts, is an error
    together with layer, and is listed in params.ignored with input_mode='visium_10x'. The matrix
    tested is params.expression_source ("X", "raw.X" or "layers['<name>']"), and a refusal of X
    says whether adata.raw holds counts. A SOM node whose spots are all empty over the tested genes
    is refused by name before SOMDE's fit (it would die with "SVD did not converge").
    """
    payload = {
        "input_mode": input_mode,
        "output_dir": output_dir,
        "counts_h5": counts_h5,
        "spatial_dir": spatial_dir,
        "h5ad_path": h5ad_path,
        "max_genes": max_genes,
        "min_counts": min_counts,
        "som_dim": som_dim,
        "top_k_genes": top_k_genes,
        "random_seed": random_seed,
        "allow_pixel_coord_fallback": allow_pixel_coord_fallback,
        "layer": layer,
        "round_counts": round_counts,
        "use_raw_counts": use_raw_counts,
    }
    return run_worker_json(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, payload)


@mcp.tool()
def somde_run(
    input_mode: str = "h5ad",
    h5ad_path: str = "",
    counts_h5: str = "",
    spatial_dir: str = "",
    output_dir: str = "",
    max_genes: int = 2000,
    min_counts: int = 1,
    som_dim: int = 20,
    top_k_genes: int = 20,
    random_seed: int = 0,
    allow_pixel_coord_fallback: bool = False,
    layer: str = "",
    round_counts: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Run SOMDE to identify spatially variable genes (simplified interface).

    For h5ad input (the usual case): pass `input_mode='h5ad'` and `h5ad_path=<file>`.
    For raw 10x Visium: pass `input_mode='visium_10x'`, `counts_h5`, and `spatial_dir`.
    `output_dir` is required; the worker refuses a run without it before reading anything.

    max_genes / min_counts are our gene pre-filter (reported in params.gene_selection and the
    analysis). som_dim is the average number of spots per SOM node. random_seed has no effect
    (SOMDE is deterministic) and is listed under params.ignored. allow_pixel_coord_fallback lets a
    visium_10x run whose array_row/array_col are constant use the pixel columns instead of stopping.

    Background spots (obs["in_tissue"] == 0) are left out and reported (params.in_tissue_filter).
    SOMDE models raw counts: NaN/inf or negative values are refused, and non-integer values too
    unless `round_counts=True` (rounded and recorded). `layer` (h5ad mode) names a raw-count layer
    to analyse instead of X; `use_raw_counts=True` (h5ad mode, layer empty) analyses adata.raw.X
    (params.expression_source says which). The memory preflight counts the dense table (max_genes)
    and the SOM-node kernels (som_dim) and names both knobs when a run cannot fit.
    """
    payload = {
        "input_mode": input_mode,
        "h5ad_path": h5ad_path,
        "counts_h5": counts_h5,
        "spatial_dir": spatial_dir,
        "output_dir": output_dir,
        "max_genes": max_genes,
        "min_counts": min_counts,
        "som_dim": som_dim,
        "top_k_genes": top_k_genes,
        "random_seed": random_seed,
        "allow_pixel_coord_fallback": allow_pixel_coord_fallback,
        "layer": layer,
        "round_counts": round_counts,
        "use_raw_counts": use_raw_counts,
    }
    return run_worker_json(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, payload)


if __name__ == "__main__":
    mcp.run()
