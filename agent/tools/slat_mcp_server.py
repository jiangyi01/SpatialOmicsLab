#!/usr/bin/env python3
"""scSLAT spatial alignment MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "slat"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SLAT",
    "/opt/conda/envs/slat/bin/python",
    "/workspace/epic-fermat/agent/tools/slat_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def slat_align_slices(
    h5ad_path_1: str,
    h5ad_path_2: str,
    output_dir: str,
    k_cutoff: int = 10,
    feature_type: str = "DPCA",
    epochs: int = 80,
    allow_feature_fallback: bool = False,
) -> dict[str, Any]:
    """
    Align two spatial transcriptomics slices using scSLAT (graph neural
    network-based spatial alignment).

    scSLAT learns spatial-aware embeddings via a graph convolutional network
    and matches cells/spots across two slices using those embeddings.

    X must hold raw counts: scSLAT's feature step normalises, log-transforms
    and selects genes (scanpy seurat_v3) itself, from X, on the genes the two
    slices share. Nothing is preprocessed before it.

    Every spot of each file is aligned, including spots with
    obs['in_tissue'] == 0 (background glass, as CELLxGENE Visium exports
    carry): slat_matching.csv indexes the rows of the input files. Their
    count per slice is reported in params.n_spots_off_tissue_slice1 /
    params.n_spots_off_tissue_slice2 and a warning; remove them from the h5ad
    beforehand to align the tissue alone.

    Parameters
    ----------
    h5ad_path_1:
        Path to the first spatial AnnData (.h5ad) slice.
    h5ad_path_2:
        Path to the second spatial AnnData (.h5ad) slice.
    output_dir:
        Directory where outputs will be written:
          - slat_matching.csv       (cell-to-cell matching; columns slice1_idx,
                                     slice2_idx are 0-based row positions in
                                     slice 1 and slice 2. One row per spot of
                                     the smaller slice -- slice 2 when the two
                                     are equal -- paired with its most similar
                                     spot in the other slice. The payload names
                                     the enumerated side in
                                     params.matching_query_slice.)
          - slat_embeddings.npz     (learned embeddings)
          - slat_alignment_plot.png (visualization)
    k_cutoff:
        Number of neighbors for spatial graph construction (default 10).
    feature_type:
        Feature type for load_anndatas. One of "DPCA", "PCA", "HVG"
        (default "DPCA"). DPCA is a dual PCA (50 dims) over the 12,000 most
        variable shared genes (seurat_v3, fixed inside scSLAT); PCA is a
        joint PCA (50 dims) over the 2,500 most variable. The payload gives
        the count in params.n_genes_used beside params.n_genes_shared.
        "HVG" fails in scSLAT 0.3.0 on every input (an upstream bug in
        load_anndatas), so it stops the run unless allow_feature_fallback
        is true.
    epochs:
        Number of training epochs for the GAN model (default 80). scSLAT's
        own run_SLAT default is 6, the value its tutorials use, and its
        docstring advises not exceeding 10.
    allow_feature_fallback:
        Default false: when the requested feature type cannot be built the
        run stops with scSLAT's error. True: the run continues on the next
        type (DPCA -> PCA -> raw, PCA -> raw, HVG -> PCA -> raw; 'raw' is the
        dense expression matrix of every shared gene), and the payload names
        the one that ran in params.feature_type and params.method and sets
        params.used_fallback to true.
    """
    args = [
        "--h5ad-path-1",
        h5ad_path_1,
        "--h5ad-path-2",
        h5ad_path_2,
        "--output-dir",
        output_dir,
        "--k-cutoff",
        str(k_cutoff),
        "--feature-type",
        feature_type,
        "--epochs",
        str(epochs),
    ]
    if allow_feature_fallback:
        args.append("--allow-feature-fallback")

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
