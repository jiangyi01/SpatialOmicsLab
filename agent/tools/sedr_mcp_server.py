#!/usr/bin/env python3
"""SEDR spatial embedding and clustering MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "sedr"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SEDR",
    "/opt/conda/envs/sedr_env/bin/python",
    "/workspace/epic-fermat/agent/tools/sedr_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_sedr(
    spatial_h5ad_path: str,
    output_dir: str,
    n_clusters: int = 7,
    using_dec: bool = True,
    device: str = "cuda",
    hvg_flavor: str = "seurat",
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run SEDR spatial embedding, then KMeans on that embedding for spatial domains.

    SEDR learns a low-dimensional embedding of spatial transcriptomics data by jointly
    modeling gene expression and a spatial kNN graph (k=12) with a variational graph
    autoencoder. The domain labels are then KMeans(k=n_clusters) on that embedding -- in
    both DEC modes. SEDR itself writes no labels: with using_dec=True its Deep Embedding
    Clustering step refines the embedding toward 10 KMeans centroids (upstream's default,
    independent of n_clusters) before the final KMeans runs; with using_dec=False the
    autoencoder is trained without that step. The payload names what ran under
    params.method and params.clustering_method, and reports params.dec_cluster_n.

    Preprocessing is this wrapper's own, NOT the upstream SEDR tutorial's (which filters at
    min_cells=50/min_counts=10, CPM-normalises without log, keeps 2,000 seurat_v3 HVGs and
    feeds a 200-component PCA to SEDR). Here: genes detected in fewer than 3 spots are
    dropped, counts are normalised to 1e4 and log1p-transformed, the 3,000 most highly
    variable genes by hvg_flavor are kept (a cap fixed inside the tool), the matrix is
    scaled (no centering, clipped at 10) and fed to SEDR densely (SEDR wraps its input in a
    torch tensor) with no PCA, so SEDR's input dimensionality is the HVG count. The steps
    are listed under params.preprocessing. The payload reports the supplied panel as
    n_genes and the analysed panel as n_genes_used, and carries a warning naming the cut
    when the two differ. The run is seeded (params.random_seed, 2023), which makes it
    bit-reproducible only on one CPU thread (OMP_NUM_THREADS=1); with several threads,
    parallel floating-point reductions can move spots between domains from run to run
    (two 3-thread runs on one slide agreed at ARI 0.86). params.torch_num_threads records
    how many threads the run used.

    Spots with obs["in_tissue"] == 0 (background outside the tissue, which CELLxGENE Visium
    exports ship beside it) are left out before anything runs and counted under
    params.in_tissue_filter with a warning; data.n_spots is the supplied count and
    data.n_spots_used the analysed one.

    Raw counts: the preprocessing normalises the matrix as counts, so X is checked before it
    runs. A negative or non-finite X (scaled or z-scored data, e.g. a CELLxGENE export whose
    counts sit in adata.raw) is refused, and the error says whether adata.raw holds counts; a
    non-negative non-integer X (already normalised or log-transformed) runs as before but is
    normalised a second time, with a warning; use_raw_counts=True runs on adata.raw.X instead.
    params.expression_source ("X" or "raw.X") and params.x_matrix_kind say which matrix ran.

    Memory: upstream SEDR builds its spatial graph from dense spot-by-spot matrices (about
    24 bytes x n_spots^2 at peak, measured, e.g. ~56 GiB for 50,000 spots) beside the dense
    expression matrix. A slide that cannot fit in the memory available to the run (a cgroup
    limit included, with its reclaimable page cache counted as free) is refused before any
    work with the estimate and the memory available; it is never subsampled. Duplicate gene
    symbols are made unique and counted under params.n_genes_renamed.

    Parameters
    ----------
    spatial_h5ad_path:
        Path to spatial transcriptomics AnnData (.h5ad) with raw counts in .X (or in
        adata.raw, with use_raw_counts=True) and spatial coordinates in obsm['spatial'].
    output_dir:
        Directory to write SEDR outputs (embedding, clusters CSV,
        annotated h5ad, plots). When the spatial plot cannot be drawn,
        output_files.spatial_plot_png is None and a warning says why.
    n_clusters:
        Number of spatial domains: the k of the KMeans run on the SEDR embedding.
        Must be between 1 and the number of in-tissue spots (checked before training).
    using_dec:
        Whether SEDR's Deep Embedding Clustering step refines the embedding before
        KMeans. It does not change which method assigns the domains (always KMeans) and
        its centroid count is upstream's default of 10, not n_clusters.
    device:
        PyTorch device string: "cuda" (preferred) or "cpu".
    hvg_flavor:
        Scanpy highly_variable_genes flavour for the top-3,000 gene selection: "seurat"
        (default; dispersion on the log-normalised data, needs no extra package),
        "cell_ranger", or "seurat_v3" (variance-stabilised on the raw counts, as upstream
        SEDR does; needs scikit-misc). When the flavour's dependency is missing the run
        stops with an error naming the package -- it does not switch flavour silently.
    use_raw_counts:
        False (default): run on X. True: run on the counts in adata.raw.X (obs and obsm are
        kept); an h5ad without adata.raw, or whose adata.raw does not hold counts, is refused.
    """
    args = [
        "--spatial-h5ad",
        str(Path(spatial_h5ad_path).expanduser()),
        "--output-dir",
        str(Path(output_dir).expanduser()),
        "--n-clusters",
        str(n_clusters),
        "--using-dec",
        "True" if using_dec else "False",
        "--device",
        device,
        "--hvg-flavor",
        str(hvg_flavor),
    ]
    if use_raw_counts:
        args.append("--use-raw-counts")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
