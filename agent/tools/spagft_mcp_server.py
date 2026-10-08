#!/usr/bin/env python3
"""SpaGFT spatially-varying gene MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "spagft"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPAGFT",
    "/opt/conda/envs/spagft_env/bin/python",
    "/workspace/epic-fermat/agent/tools/spagft_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def spagft_identify_svg(
    st_h5ad: str,
    output_dir: str,
    layer_key: str = "counts",
    spatial_key: str = "spatial",
    n_neighbors: int = 20,
    n_eigs: int = 50,
    low_freq_fraction: float = 0.1,
    min_detected_frac: float = 0.05,
    n_top_genes: int = 200,
    seed: int = 0,
    allow_gft_fallback: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Identify spatially-varying genes (SVGs) with the official SpaGFT package (``SpaGFT.detect_svg``).

    SpaGFT builds a KNN graph on the spot coordinates, projects each gene's normalize_total + log1p
    expression onto the graph's low- and high-frequency Fourier modes, ranks genes by GFT score and
    calls a gene significant when it passes SpaGFT's kneedle cutoff on that score
    (``params.significance_basis``). Spots with ``obs['in_tissue'] == 0`` are background and are left
    out (``data.n_spots_out_of_tissue_excluded``); genes detected in no analysed spot are dropped by
    SpaGFT itself (``data.n_genes`` supplied vs ``data.n_genes_used``). Nothing is subsampled.
    SpaGFT densifies the spots x genes matrix internally; a slide too large for this machine's
    memory is refused with the numbers before anything runs.

    Parameters
    ----------
    st_h5ad:
        Path to a spatial transcriptomics AnnData (.h5ad).
    output_dir:
        Directory where SpaGFT outputs will be written (spagft_svg_scores.csv,
        spagft_top_svg_genes.csv, predicted_genes.json). The from-scratch substitute
        (allow_gft_fallback) writes spagft_graph_spectrum.npz instead of predicted_genes.json.
    layer_key:
        AnnData layer holding raw counts. adata.X is analysed when the object has no such layer;
        ``params.expression_source`` says which matrix was read, and a warning says so whenever a
        layer other than the default 'counts' was named and is absent. The matrix is
        normalize_total + log1p-transformed as counts: one with negative or non-finite values
        (scaled / z-scored data) is refused, naming use_raw_counts when adata.raw holds counts, and
        a non-integer (already normalised) one runs with a warning that it is not raw counts.
    spatial_key:
        Key in adata.obsm holding the 2D coordinates the KNN graph is built on. Two obs column
        names joined by a comma ('array_row,array_col', SpaGFT's tutorial convention for Visium)
        select those columns instead; the obs grid is never read in place of an obsm key.
        ``params.spatial_info`` names the coordinates used. An analysed spot with no finite
        coordinate is refused with the count.
    n_neighbors:
        KNN size of the from-scratch substitute only. SpaGFT sets its own, ceil(sqrt(n_spots)/2)
        (4 at <= 500 spots), reported as ``params.num_neighbors``; this value is listed in
        ``params.ignored`` when SpaGFT runs.
    n_eigs:
        Eigenpairs of the from-scratch substitute only. SpaGFT uses ceil(sqrt(n_spots)) low- and
        high-frequency modes each (``params.n_low_frequency_modes``); listed in ``params.ignored``.
    low_freq_fraction:
        Low-frequency share of the from-scratch substitute only; listed in ``params.ignored`` when
        SpaGFT runs.
    min_detected_frac:
        Gene detection filter of the from-scratch substitute only. SpaGFT keeps every gene detected
        in at least one spot; listed in ``params.ignored`` when SpaGFT runs.
    n_top_genes:
        Number of top-ranked significant SVGs written to spagft_top_svg_genes.csv and
        predicted_genes.json (at least 1; a value above the significant count keeps them all).
    seed:
        Seed of the from-scratch substitute only. SpaGFT draws no random number; listed in
        ``params.ignored`` when SpaGFT runs.
    allow_gft_fallback:
        Whether to accept a substitute when SpaGFT cannot be imported. Default False: the call then
        fails and names what is missing, so nothing that is not SpaGFT is reported under its name.
        True accepts a from-scratch low-frequency energy ratio on a normalized-Laplacian KNN graph
        (not SpaGFT, no significance test), computed on the same normalize_total + log1p matrix
        with each gene centred; the result then says so in ``params.method``,
        ``params.used_fallback`` and the analysis text.
    use_raw_counts:
        Analyse adata.raw.X instead of X or the layer_key layer (default False). Use it when X is
        log-normalised or scaled and the integer counts sit in adata.raw (CELLxGENE exports);
        refused when the object has no adata.raw or it does not hold counts. A layer_key layer the
        object has is then not read and is listed in ``params.ignored``.
    """
    args = [
        "--task",
        "svg",
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--layer-key",
        layer_key,
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
    ]
    if allow_gft_fallback:
        args.append("--allow-gft-fallback")
    if use_raw_counts:
        args.append("--use-raw-counts")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
