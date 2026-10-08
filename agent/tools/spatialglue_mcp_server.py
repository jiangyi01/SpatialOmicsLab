#!/usr/bin/env python3
"""SpatialGlue multi-omics integration MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "spatialglue"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPATIALGLUE",
    "/opt/conda/envs/spatialglue/bin/python",
    "/workspace/epic-fermat/agent/tools/spatialglue_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def spatialglue_integrate(
    rna_h5ad: str,
    protein_h5ad: str,
    output_dir: str,
    datatype: str = "10X",
    epochs: int = 600,
    dim_output: int = 64,
    n_clusters: int = 7,
    n_neighbors: int = 3,
    cluster_method: str = "leiden",
    resolution: float = 1.0,
    device: str = "auto",
    seed: int = 2022,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run SpatialGlue multi-omics spatial integration.

    SpatialGlue integrates two spatial omics modalities (RNA + protein, or RNA + ATAC/epigenome)
    into a unified latent embedding using a graph attention network with cross-modality
    correspondence learning. The upstream ``Train_SpatialGlue`` trainer runs; the preprocessing
    follows the upstream tutorials.

    Parameters
    ----------
    rna_h5ad:
        Path to the first omics AnnData (.h5ad), spatial transcriptomics (RNA). Must have
        obsm['spatial'] and raw counts in .X (non-integer or negative values are refused; when the
        counts sit in adata.raw the refusal says so, and use_raw_counts=True reads them). RNA
        features: filter_genes(min_cells=10), 3000 seurat_v3 highly variable genes on the counts,
        normalize_total(1e4), log1p, scale, then PCA to (number of proteins - 1) components, or 50
        for 'Spatial-epigenome-transcriptome'. If .var has 'feature_types' (10x), only the
        'Gene Expression' features are used.
    protein_h5ad:
        Path to the second omics AnnData (.h5ad): protein (CITE-seq/SPOTS/10x Antibody Capture) or
        ATAC/epigenome. Must have obsm['spatial']. Protein: raw ADT counts in .X, then CLR, scale and
        PCA(n_proteins - 1); ATAC/epigenome: LSI (a precomputed obsm['X_lsi'] is used as is). Pass
        the same path as rna_h5ad for one combined 10x file (e.g. CytAssist Gene Expression +
        Antibody Capture in one X): it is split by var['feature_types'], and a file with no
        feature_types to split by is refused. Spots are paired between the modalities by barcode
        (obs_names), never by row position; spots in only one modality are left out and counted.
    output_dir:
        Directory where SpatialGlue outputs will be written.
    datatype:
        Upstream SpatialGlue preset, matched case-insensitively: "10x" (also "10X"), "SPOTS",
        "Stereo-CITE-seq", "Spatial-epigenome-transcriptome"; any other value is refused. It sets
        the loss weight factors (reported as params.weight_factors) and the second modality's
        kind: the first three are protein, "Spatial-epigenome-transcriptome" is ATAC/epigenome.
    epochs:
        Number of training epochs, always honoured. Each upstream preset carries its own epoch
        count (10x 200, SPOTS 600, Stereo-CITE-seq 1500, Spatial-epigenome-transcriptome 1600);
        that value is reported as params.epochs_preset and does not override this one.
    dim_output:
        Dimension of the output latent embedding.
    n_clusters:
        Number of spatial domains, used by cluster_method="mclust" only. Leiden/Louvain run at
        ``resolution`` and do not read it (it is then listed in params.ignored).
    n_neighbors:
        Nearest neighbours per spot for the spatial graph. Upstream fixes it at 6 for
        "Stereo-CITE-seq" and "Spatial-epigenome-transcriptome" (the value used is reported as
        params.n_neighbors_spatial_effective, and n_neighbors goes to params.ignored). The feature
        graph always uses k=20, fixed upstream.
    cluster_method:
        Clustering method: "leiden" (needs leidenalg), "louvain" (scanpy's default 'vtraag' flavour,
        needs the louvain package) or "mclust" (needs rpy2 and an R with the mclust package). The
        chosen method is tried on a 40-point toy embedding before any data is read; one whose
        packages are missing is refused then, naming them, instead of after the training run.
        No other method is substituted.
    resolution:
        Resolution parameter for leiden/louvain clustering; mclust does not read it.
    device:
        Compute device: "auto" (follow the hardware), "cpu", "gpu"/"cuda", or "cuda:N".
    seed:
        Random seed for numpy, torch and Python's random (PCA, model initialisation, training),
        set through SpatialGlue's fix_seed. Leiden/Louvain keep scanpy's fixed random_state=0 and
        mclust its fixed seed.
    use_raw_counts:
        Default False. True reads the counts of each input that carries adata.raw from it (refused
        when adata.raw does not hold counts); an input without adata.raw keeps its X, with a
        warning. For a CELLxGENE-style h5ad whose X is normalised or scaled.
        params.expression_source (RNA) and params.expression_source_omics2 say which matrix ran.
        The epigenome modality (LSI) reads X as supplied unless this asks for adata.raw.

    Notes
    -----
    Upstream SpatialGlue builds dense n_spots x n_spots adjacency matrices; a slide whose graphs
    cannot fit in the memory available is refused up front with the numbers. The estimate also
    counts the dense feature matrices: the RNA highly variable genes and, for the second modality,
    every protein, or only the LSI components when the epigenome matrix is sparse (upstream LSI
    keeps it sparse) or obsm['X_lsi'] is precomputed. Spots either file marks obs['in_tissue'] == 0
    (background outside the tissue) are left out of both modalities and counted in
    data.n_spots_off_tissue_dropped, params.in_tissue_filter and a warning.
    """
    args = [
        "--rna-h5ad",
        rna_h5ad,
        "--protein-h5ad",
        protein_h5ad,
        "--output-dir",
        output_dir,
        "--datatype",
        datatype,
        "--epochs",
        str(epochs),
        "--dim-output",
        str(dim_output),
        "--n-clusters",
        str(n_clusters),
        "--n-neighbors",
        str(n_neighbors),
        "--cluster-method",
        cluster_method,
        "--resolution",
        str(resolution),
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    if use_raw_counts:
        args.append("--use-raw-counts")

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
