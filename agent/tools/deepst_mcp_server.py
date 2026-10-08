#!/usr/bin/env python3
"""DeepST spatial domain identification MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "deepst"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "DEEPST",
    "/opt/conda/envs/deepst-env/bin/python",
    "/workspace/epic-fermat/agent/tools/deepst_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def deepst_identify_domains(
    st_h5ad: str,
    output_dir: str,
    n_domains: int = 7,
    pre_epochs: int = 500,
    epochs: int = 500,
    pca_n_comps: int = 200,
    spatial_type: str = "BallTree",
    dist_type: str = "KDTree",
    use_morphological: bool = False,
    use_gpu: str = "auto",
    seed: int = 0,
    allow_resolution_fallback: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Identify spatial domains using DeepST (deepstkit) on a single Visium-like
    spatial transcriptomics AnnData (.h5ad).

    DeepST learns an embedding with a graph autoencoder, clusters it with Leiden,
    and spatially refines the labels into obs["DeepST_refine_domain"]. The Leiden
    resolution is found by a sweep (2.49 down to 0.10, step 0.01) for a partition
    with exactly n_domains clusters. When no resolution yields that count the run
    stops by default; see allow_resolution_fallback. The payload reports what ran:
    params.method, params.used_fallback, params.resolution_used,
    params.resolution_search_matched, summary.n_domains_leiden (before spatial
    refinement) and summary.n_clusters (after it).

    Raw counts: deepstkit normalises the (augmented) expression matrix as counts
    (normalize_total, then log1p), so X is checked before it runs. A negative or non-finite X
    (scaled or z-scored data, e.g. a CELLxGENE export whose counts sit in adata.raw) is refused,
    and the error says whether adata.raw holds counts; a non-negative non-integer X (already
    normalised or log-transformed) runs as before but is normalised a second time, with a
    warning; use_raw_counts=True runs on adata.raw.X instead. params.expression_source ("X" or
    "raw.X") and params.x_matrix_kind say which matrix ran.

    Parameters
    ----------
    st_h5ad:
        Path to the spatial AnnData (.h5ad) file to analyze. Needs obsm["spatial"] and raw
        counts in X (or in adata.raw, with use_raw_counts=True).
        Spots with obs["in_tissue"] == 0 (background outside the tissue, which
        CELLxGENE Visium exports ship beside it) are left out and counted under
        params.in_tissue_filter with a warning; data.n_spots is the supplied count and
        data.n_spots_used the analysed one. The in-tissue spots are analysed whole:
        DeepST builds dense spot-by-spot matrices, and a slide whose matrices cannot fit
        in the memory available to the run (cgroup limit included) is refused up front
        with the numbers rather than cut down.
    output_dir:
        Directory to write DeepST outputs (annotated h5ad, domains CSV, PNG). When the
        spatial plot cannot be drawn, output_files.spatial_plot_png is None and a
        warning says why; the domain outputs are still complete.
    n_domains:
        Target number of spatial domains. Leiden cannot be told a count directly,
        so deepstkit sweeps its resolution for one that yields exactly this many
        clusters; spatial refinement may then merge some, so summary.n_clusters
        can be lower than n_domains even when the sweep matched. Must be between
        1 and the number of in-tissue spots (checked before training).
    pre_epochs:
        Pretraining epochs for DeepST GNN autoencoder.
    epochs:
        Main training epochs for DeepST.
    pca_n_comps:
        Number of PCs used as input to DeepST. Capped to min(n_spots, n_genes) - 1
        when the request exceeds it; params.pca_n_comps reports the value that
        ran and params.pca_n_comps_requested the request, with a warning.
    spatial_type:
        Spatial neighbor search in DeepST._get_augment: "BallTree" (default),
        "KDTree", "NearestNeighbors", or "LinearRegress" (which reads obs columns
        imagerow, imagecol, array_row and array_col instead of obsm["spatial"]).
        Any other value is refused before training.
    dist_type:
        Distance type in DeepST._get_graph.
    use_morphological:
        Whether to weight neighbours by H&E similarity. deepstkit reads the image
        features from obsm["image_feat_pca"] (its _get_image_crop output); this tool
        does not extract them, so True needs an input that already carries that key
        and is refused before training otherwise.
    use_gpu:
        Compute device: "auto" (follow the hardware), "cpu", "gpu"/"cuda", or "cuda:N".
        A GPU request on a machine without CUDA runs on the CPU and says so in the log.
    seed:
        Random seed for reproducibility.
    allow_resolution_fallback:
        When no Leiden resolution in the sweep yields exactly n_domains, deepstkit
        would run Leiden at its default resolution 1.0 and return however many
        domains that gives. Off (the default) refuses that substitute with an error
        naming this switch and the cluster counts Leiden reached at 0.10 and 2.49,
        so a reachable n_domains can be chosen; True accepts it, and the payload then carries
        params.used_fallback=true and params.resolution_used=1.0 so the result is
        never mistaken for the requested partition.
    use_raw_counts:
        False (default): run on X. True: run on the counts in adata.raw.X (obs and obsm are
        kept); an h5ad without adata.raw, or whose adata.raw does not hold counts, is refused.
    """
    args = [
        "--task",
        "identify_domains",
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--n-domains",
        str(n_domains),
        "--pre-epochs",
        str(pre_epochs),
        "--epochs",
        str(epochs),
        "--pca-n-comps",
        str(pca_n_comps),
        "--spatial-type",
        spatial_type,
        "--dist-type",
        dist_type,
        "--use-morphological",
        "True" if use_morphological else "False",
        "--use-gpu",
        use_gpu,
        "--seed",
        str(seed),
    ]
    if allow_resolution_fallback:
        args.append("--allow-resolution-fallback")
    if use_raw_counts:
        args.append("--use-raw-counts")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
