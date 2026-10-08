#!/usr/bin/env python3
"""PROST MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "prost"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "PROST",
    "/opt/conda/envs/PROST_ENV/bin/python",
    "/workspace/epic-fermat/agent/tools/prost_worker.py",
)

mcp = create_mcp(TOOL_NAME)


def _fallback_flag(allow_spectral_fallback: bool) -> str:
    """Always say which way the substitute gate is set, so the worker never falls back to a default.

    ``--no-spectral-fallback`` is kept for the refusing side because existing command lines and
    older worker copies read it; the worker refuses by default either way.
    """
    return "--allow-spectral-fallback" if allow_spectral_fallback else "--no-spectral-fallback"


@mcp.tool()
def prost_index_svg(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    layer_key: str = "",
    n_neighbors: int = 20,
    n_eigs: int = 50,
    low_freq_fraction: float = 0.1,
    min_detected_frac: float = 0.05,
    n_top_genes: int = 200,
    seed: int = 0,
    platform: str = "visium",
    allow_spectral_fallback: bool = False,
) -> dict[str, Any]:
    """Identify spatially variable genes by PROST Index (PI).

    PROST is given the expression named by layer_key (adata.X when empty; a layer the file does not
    have is an error) and the coordinates in obsm[spatial_key]; its 'visium' preset rasterises
    obs['array_row']/obs['array_col'] instead when the file has them, and params['coordinate_source']
    says which was read. Spots with obs['in_tissue'] == 0 (background outside the tissue) are left
    out before PROST runs; data.n_spots is the count supplied, data.n_spots_used the count analysed,
    and params['in_tissue_filter'] with a warning says how many were left out.

    platform selects PROST's gene-image path: 'visium'/'ST' rasterise the spots onto an integer
    lattice (obs['array_row']/obs['array_col'] when present, else obsm[spatial_key]); any other name
    interpolates the expression onto a regular grid. 'visium' -- including an explicit 'visium' -- is
    replaced by 'irregular' (the interpolating path) when the coordinates cannot form that lattice: a
    spot at a negative image index, or a median nearest-spot spacing above 4 pixels (e.g. micrometre or
    pixel coordinates without array_row/array_col). params['platform'] is the preset PROST ran with, and
    the analysis text says when and why it was replaced.

    If PROST cannot run, the call fails with PROST's own error. allow_spectral_fallback=True instead
    accepts a substitute -- a SpaGFT-style spectral score on a kNN graph of the spot coordinates, which
    is not PROST -- and the result then says so in params['method'], params['used_fallback'] and the
    analysis text. n_neighbors, n_eigs, low_freq_fraction, min_detected_frac and seed configure that
    substitute only (PROST's Index is deterministic); on a PROST run they are listed in
    params['ignored'].
    """
    args = [
        "--task",
        "index",
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--spatial-key",
        spatial_key,
        "--n-neighbors",
        str(n_neighbors),
        "--n-eigs",
        str(n_eigs),
        "--low-freq-fraction",
        str(low_freq_fraction),
        "--min-detected-frac",
        str(min_detected_frac),
        "--n-top-genes",
        str(n_top_genes),
        "--seed",
        str(seed),
        "--platform",
        platform,
    ]
    if layer_key:
        args += ["--layer-key", layer_key]
    args.append(_fallback_flag(allow_spectral_fallback))
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def prost_pnn_domains(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    layer_key: str = "",
    n_neighbors: int = 20,
    n_eigs: int = 30,
    n_domains: int = 6,
    seed: int = 0,
    platform: str = "visium",
    pnn_init: str = "kmeans",
    pnn_k_neighbors: int = 7,
    pnn_n_top_genes: int = 3000,
    allow_spectral_fallback: bool = False,
    pnn_preprocessing: str = "normalize_log1p",
    pnn_sparse: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Detect spatial domains with PROST PNN, honouring an exact n_domains.

    Runs PROST's published pipeline: prepare_for_PI -> cal_PI -> normalize_total + log1p ->
    feature_selection(by='prost') -> run_PNN. pnn_preprocessing='normalize_log1p' (the default, as
    in PROST's tutorial) needs raw counts and refuses a non-integer matrix (the error names
    use_raw_counts when adata.raw holds counts); 'none' gives PNN the matrix as stored (for input
    that is already normalised and log-transformed). use_raw_counts=True gives PROST adata.raw.X
    instead of X -- for a CELLxGENE-style h5ad whose X is processed and whose counts sit in
    adata.raw, where layer_key cannot reach them; it is refused without an adata.raw that holds
    counts, and together with layer_key. params['expression_source'] says which matrix ran
    ('raw.X', 'X' or the layer) and data.n_genes counts that matrix's genes.

    pnn_n_top_genes is an upper bound: feature_selection keeps at most the genes with a positive
    PROST Index among those prepare_for_PI kept (detected in at least 10% of spots), so a targeted
    panel can give fewer. params['pnn_n_genes_used'] (and data.n_genes_used) is the count PNN
    clustered on. Spots with obs['in_tissue'] == 0 (background outside the tissue) are left out
    before PROST runs and get no domain; data.n_spots is the count supplied, data.n_spots_used the
    count labelled, and params['in_tissue_filter'] with a warning says how many were left out.

    pnn_init must be 'kmeans' or 'mclust'; PROST's own default ('leiden') ignores n_clusters and
    picks the count from a resolution instead. pnn_k_neighbors sizes PROST's cell graph (PROST's
    default is 7) and is separate from n_neighbors, which sizes the spectral substitute's graph.

    platform selects PROST's gene-image path: 'visium'/'ST' rasterise the spots onto an integer
    lattice (obs['array_row']/obs['array_col'] when present, else obsm[spatial_key]); any other name
    interpolates the expression onto a regular grid. 'visium' -- including an explicit 'visium' -- is
    replaced by 'irregular' (the interpolating path) when the coordinates cannot form that lattice: a
    spot at a negative image index, or a median nearest-spot spacing above 4 pixels (e.g. micrometre or
    pixel coordinates without array_row/array_col). params['platform'] is the preset PROST ran with, and
    the analysis text says when and why it was replaced.

    Memory: PROST's run_PNN densifies the n_spots x n_spots spot graph and trains a dense
    graph-attention layer on it, about 56 bytes per spot pair at the peak (~8 GiB at 12,000 spots,
    ~13 TiB at 500,000). pnn_sparse=True runs PROST's own run_PNN_sparse instead (same init,
    n_domains and pnn_k_neighbors; sparse graph and attention layer, a different attention formula;
    its backward pass still builds one n x n float32 gradient, about 5 bytes per pair). The chosen
    variant's peak is estimated before prepare_for_PI starts; a run that cannot fit stops there with
    the numbers and names pnn_sparse (or allow_spectral_fallback=True returns the labelled substitute).
    params['pnn_function'] names the PROST function that clustered and params['pnn_memory'] the
    estimate, and params['method'] says which variant ran.

    If PROST cannot run, the call fails with PROST's own error. allow_spectral_fallback=True instead
    accepts a substitute -- KMeans on the Laplacian eigenvectors of a kNN graph of the spot
    coordinates, which uses no gene expression and is not PROST -- and the result then says so in
    params['method'], params['used_fallback'] and the analysis text. n_neighbors and n_eigs
    configure that substitute only.
    """
    args = [
        "--task",
        "domains",
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--spatial-key",
        spatial_key,
        "--n-neighbors",
        str(n_neighbors),
        "--n-eigs",
        str(n_eigs),
        "--n-domains",
        str(n_domains),
        "--seed",
        str(seed),
        "--platform",
        platform,
        "--pnn-init",
        pnn_init,
        "--pnn-k-neighbors",
        str(pnn_k_neighbors),
        "--pnn-n-top-genes",
        str(pnn_n_top_genes),
        "--pnn-preprocessing",
        pnn_preprocessing,
    ]
    if pnn_sparse:
        args.append("--pnn-sparse")
    if use_raw_counts:
        args.append("--use-raw-counts")
    if layer_key:
        args += ["--layer-key", layer_key]
    args.append(_fallback_flag(allow_spectral_fallback))
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
