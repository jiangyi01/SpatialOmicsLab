#!/usr/bin/env python3
"""GPSA spatial alignment MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "gpsa"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "GPSA",
    "/opt/conda/envs/gpsa/bin/python",
    "/workspace/epic-fermat/agent/tools/gpsa_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def gpsa_align_slices(
    slice1_h5ad: str,
    slice2_h5ad: str,
    output_dir: str,
    n_spatial_dims: int = 2,
    n_latent_gps: int = 3,
    num_epochs: int = 500,
    learning_rate: float = 0.001,
    batch_size: int = 128,
    n_top_genes: int = 2000,
    layer_key: str = "",
    spatial_key: str = "spatial",
    random_seed: int = 0,
    hvg_flavor: str = "seurat_v3",
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Align two spatial transcriptomics slices with GPSA (Gaussian Process Spatial
    Alignment, upstream ``gpsa.VariationalGPSA``). Slice 1 is held fixed as the
    template; GPSA learns a warp of slice 2's coordinates into slice 1's frame
    from the expression of a joint set of highly variable genes.

    Coordinates are rescaled jointly into [0, 1] for training and mapped back
    afterwards: the aligned coordinates in ``gpsa_aligned_coordinates.csv`` and in
    ``obsm['spatial_aligned']`` of both output h5ads are in the units of
    ``obsm[spatial_key]``, so they sit beside the input coordinates in one frame
    (the [0, 1] values are kept in ``obsm['spatial_aligned_normalized']``, and the
    min/range used in ``uns['gpsa_alignment']``). Each output h5ad holds the genes
    both slices share, with X = log1p(normalize_total(1e4)) of the expression source.
    GPSA models the expression as one dense spots x HVGs matrix; the run estimates
    it first and stops, naming ``n_top_genes``, when it cannot fit in memory
    (against the smaller of MemAvailable and the room under a cgroup limit).

    Spots with obs['in_tissue'] == 0 (background glass, carried by CELLxGENE
    Visium exports) are left out of both slices before anything else and counted
    in params.in_tissue_filter and params.in_tissue_dropped_per_slice; the outputs
    and data.n_spots_slice1/2 cover the in-tissue spots only. The expression is
    normalised as counts, so a matrix with negative or non-finite values is
    refused and non-integer values run with a warning.

    Parameters
    ----------
    slice1_h5ad:
        Path to the first AnnData (.h5ad) spatial slice (the fixed template).
    slice2_h5ad:
        Path to the second AnnData (.h5ad) spatial slice (warped onto slice 1).
    output_dir:
        Directory for output files (aligned coordinates CSV, annotated h5ad).
    n_spatial_dims:
        Number of spatial dimensions (typically 2 for 2D tissue sections).
    n_latent_gps:
        Number of latent Gaussian processes for the alignment model.
    num_epochs:
        Number of training epochs for the variational GPSA model.
    learning_rate:
        Learning rate for the Adam optimizer.
    batch_size:
        Accepted for compatibility and ignored: GPSA trains full-batch (every spot
        of both slices at every step). Reported under ``params.ignored``.
    n_top_genes:
        Number of highly variable genes, chosen jointly over both slices, that
        drive the alignment (capped at the number of shared genes).
    layer_key:
        AnnData layer holding the expression, present in both slices (raw counts
        for the default ``hvg_flavor='seurat_v3'``). A layer missing from either
        slice is an error. Empty string uses adata.X.
    spatial_key:
        Key in adata.obsm containing spatial coordinates.
    random_seed:
        Random seed for reproducibility.
    hvg_flavor:
        Scanpy flavour for the joint HVG selection: 'seurat_v3' (ranks raw counts;
        needs scikit-misc), 'seurat' or 'cell_ranger' (rank log-normalised data).
        A flavour that cannot run stops the run; no other flavour is substituted.
    use_raw_counts:
        Read both slices' expression from adata.raw.X (raw counts) instead of X.
        Refused together with layer_key, or when a slice has no adata.raw or it
        does not hold counts. Default False reads X (or the layer_key layer).
    """
    args = [
        "--slice1-h5ad",
        slice1_h5ad,
        "--slice2-h5ad",
        slice2_h5ad,
        "--output-dir",
        output_dir,
        "--n-spatial-dims",
        str(n_spatial_dims),
        "--n-latent-gps",
        str(n_latent_gps),
        "--num-epochs",
        str(num_epochs),
        "--learning-rate",
        str(learning_rate),
        "--batch-size",
        str(batch_size),
        "--n-top-genes",
        str(n_top_genes),
        "--spatial-key",
        spatial_key,
        "--random-seed",
        str(random_seed),
        "--hvg-flavor",
        str(hvg_flavor),
    ]
    if layer_key:
        args.extend(["--layer-key", layer_key])
    if use_raw_counts:
        args.append("--use-raw-counts")

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
