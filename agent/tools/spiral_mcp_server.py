#!/usr/bin/env python3
"""SPIRAL spatial transcriptomics integration & alignment MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "spiral"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPIRAL",
    "/opt/conda/envs/spiral/bin/python",
    "/workspace/epic-fermat/agent/tools/spiral_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def spiral_integrate(
    h5ad_paths: list[str],
    output_dir: str,
    n_epochs: int = 200,
    hidden_dim: int = 32,
    latent_dim: int = 32,
    knn: int = 6,
    batch_size: int = 1024,
    n_clusters: int = 7,
    cluster_method: str = "leiden",
    device: str = "auto",
    resolution: float = 0.8,
) -> dict[str, Any]:
    """
    Run SPIRAL batch correction / integration of multiple spatial transcriptomics slices.

    SPIRAL (upstream ``spiral.main.SPIRAL_integration``) jointly embeds multiple spatial slices
    into a shared latent space using a graph-based adversarial domain adaptation model, then the
    embedding is clustered. The expression matrix ``X`` of each slice is used as given (no
    normalisation or gene selection here; SPIRAL min-max scales each spot), restricted to the genes
    every slice shares. Cell-type labels are not used. SPIRAL holds that table dense; a table this
    machine cannot hold is refused with the numbers rather than cut down. A slice whose
    obs['in_tissue'] marks spots as 0 (background) has them left out as it is read, counted in
    params.in_tissue_filter with a warning.

    Outputs: ``spiral_embeddings.csv``, ``spiral_corrected_expression.csv`` (in SPIRAL's per-spot
    min-max [0, 1] space, not counts), ``spiral_clusters.csv`` and ``spiral_integrated.h5ad``,
    whose ``obsm['spatial']`` is each slice's own coordinates stacked -- not a common frame.
    ``params.method`` names what ran; ``params.batch_size_effective`` and ``params.hidden_width``
    give the values training actually used.

    Parameters
    ----------
    h5ad_paths:
        List of paths to .h5ad files, one per spatial slice. Each must have obsm['spatial'] with
        (x, y) coordinates; a third column is accepted only when it is constant within the slice
        (one plane) and is refused when it varies. At least 2 slices required.
    output_dir:
        Directory where SPIRAL outputs will be written.
    n_epochs:
        Number of training epochs for the integration model.
    hidden_dim:
        Hidden size unit: the autoencoder and GraphSAGE hidden layers are hidden_dim * 16 wide
        (512 at the default 32, SPIRAL's published width).
    latent_dim:
        Latent embedding dimension.
    knn:
        Number of nearest neighbors for spatial graph construction.
    batch_size:
        Training batch size. A value not below the total spot count is halved (a batch must be
        smaller than the graph); the value used is params.batch_size_effective.
    n_clusters:
        Number of spatial domains, used by cluster_method "mclust" only. leiden and louvain are
        resolution-driven and ignore it (listed in params.ignored).
    cluster_method:
        Clustering method: "leiden", "louvain", or "mclust". "louvain" is scanpy's sc.tl.louvain,
        which needs the ``louvain`` package; where it cannot be imported the run is refused before
        training (use "leiden") rather than failing after it. No other method is substituted.
    device:
        Compute device: "auto" (follow the hardware), "cpu", "gpu"/"cuda", or "cuda:N".
    resolution:
        Clustering resolution for "leiden"/"louvain" (ignored by "mclust"). Higher gives more
        domains.
    """
    if not h5ad_paths or len(h5ad_paths) < 2:
        return {
            "status": "error",
            "error": "spiral_integrate requires at least two h5ad_paths.",
        }

    args: list = [
        "--task",
        "integrate",
        "--output-dir",
        output_dir,
        "--n-epochs",
        str(n_epochs),
        "--hidden-dim",
        str(hidden_dim),
        "--latent-dim",
        str(latent_dim),
        "--knn",
        str(knn),
        "--batch-size",
        str(batch_size),
        "--n-clusters",
        str(n_clusters),
        "--cluster-method",
        cluster_method,
        "--resolution",
        str(resolution),
        "--device",
        device,
    ]
    for p in h5ad_paths:
        args.extend(["--h5ad", p])

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def spiral_align(
    h5ad_path_1: str,
    h5ad_path_2: str,
    output_dir: str,
    n_epochs: int = 200,
    hidden_dim: int = 32,
    latent_dim: int = 32,
    knn: int = 6,
    batch_size: int = 1024,
    n_clusters: int = 7,
    cluster_method: str = "leiden",
    alpha: float = 0.5,
    device: str = "auto",
    resolution: float = 0.8,
) -> dict[str, Any]:
    """
    Map the second spatial slice into the first slice's coordinate frame.

    Runs the same SPIRAL integration as spiral_integrate on the two slices, clusters the joint
    embedding, and then -- for each cluster present in BOTH slices -- solves a fused
    Gromov-Wasserstein transport (POT) between the two slices' spots and moves each slice-2 spot
    to the transport-weighted mean of its slice-1 partners. This mapping is this wrapper's own
    implementation; SPIRAL's ``CoordAlignment`` (which also places slice-specific clusters with a
    Procrustes fit) is not run, and ``params.method`` says so. Slice-2 spots outside every shared
    cluster (or in a shared cluster with fewer than 2 spots in a slice) are not placed: their rows
    in ``spiral_aligned_coordinates.csv`` and ``obsm['spatial_aligned']`` are NaN, ``obs
    ['spiral_aligned']`` is False, and ``data.n_spots_unaligned_slice_1`` counts them. A run that
    places no spot at all fails. Background spots (obs['in_tissue'] == 0) are left out of both
    slices as they are read, counted in params.in_tissue_filter.

    Parameters
    ----------
    h5ad_path_1:
        Path to the first (reference) spatial slice .h5ad; its coordinates are kept as they are.
    h5ad_path_2:
        Path to the second spatial slice .h5ad, the one that is moved.
    output_dir:
        Directory where alignment outputs will be written.
    n_epochs:
        Number of training epochs for the integration model.
    hidden_dim:
        Hidden size unit: the hidden layers are hidden_dim * 16 wide (512 at the default 32).
    latent_dim:
        Latent embedding dimension.
    knn:
        Number of nearest neighbors for spatial graph construction.
    batch_size:
        Training batch size (see spiral_integrate; the value used is params.batch_size_effective).
    n_clusters:
        Number of clusters to align through, used by cluster_method "mclust" only; leiden and
        louvain ignore it.
    cluster_method:
        Clustering method: "leiden", "louvain", or "mclust". "louvain" needs the ``louvain``
        package; where it cannot be imported the run is refused before training.
    alpha:
        Tradeoff between expression and spatial distance in the fused Gromov-Wasserstein
        transport (0 = expression only, 1 = spatial structure only).
    device:
        Compute device: "auto" (follow the hardware), "cpu", "gpu"/"cuda", or "cuda:N".
    resolution:
        Clustering resolution for "leiden"/"louvain" (ignored by "mclust").
    """
    args: list = [
        "--task",
        "align",
        "--output-dir",
        output_dir,
        "--n-epochs",
        str(n_epochs),
        "--hidden-dim",
        str(hidden_dim),
        "--latent-dim",
        str(latent_dim),
        "--knn",
        str(knn),
        "--batch-size",
        str(batch_size),
        "--n-clusters",
        str(n_clusters),
        "--cluster-method",
        cluster_method,
        "--resolution",
        str(resolution),
        "--alpha",
        str(alpha),
        "--device",
        device,
        "--h5ad",
        h5ad_path_1,
        "--h5ad",
        h5ad_path_2,
    ]

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
