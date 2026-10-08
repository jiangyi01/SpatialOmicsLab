#!/usr/bin/env python3
"""stPlus spatial gene imputation MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "stplus"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "STPLUS",
    "/opt/conda/envs/stplus_env/bin/python",
    "/workspace/epic-fermat/agent/tools/stplus_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_stplus(
    spatial_df_path: str,
    scrna_df_path: str,
    genes_to_impute_path: str,
    output_dir: str = default_output_dir(),
    n_neighbors: int = 50,
    n_epochs: int = 200,
    batch_size: int = 512,
    device: str = "auto",
    normalize: str = "auto",
    top_k: int = 2000,
) -> dict[str, Any]:
    """
    Run stPlus spatial gene imputation pipeline.

    stPlus trains an autoencoder on the spatial spots and the scRNA-seq reference together, using
    the genes they share as anchors, then predicts each spot's unmeasured genes as a weighted
    average over its nearest reference cells (cosine distance in the learned embedding). Inputs can
    be h5ad files (auto-converted to DataFrames internally), a 10x Cell Ranger / Space Ranger .h5
    matrix (e.g. filtered_feature_bc_matrix.h5, read with scanpy.read_10x_h5: gene symbols as
    columns, barcodes as rows), or delimited text; the delimiter is read from the file's first line
    rather than assumed from its name, and .gz/.bz2/.xz are decompressed on the way in.

    Spots whose obs['in_tissue'] is 0 (background glass: a CELLxGENE Visium h5ad carries every
    array spot) are left out of training and imputation, and the payload reports them in
    params.in_tissue_filter and data.n_spots_supplied; data.n_spots counts the spots imputed.

    stPlus is written for normalized, log-transformed input. With ``normalize="auto"`` a matrix of
    raw integer counts is log1p(normalize_total(target_sum=1e4))'d before training and any other
    matrix is used as given; the payload's ``params.normalization_spatial`` /
    ``params.normalization_scrna`` say what was done to each. Imputed values are a KNN-weighted
    average of the reference's values, so they are on the reference's (normalized) scale;
    stplus_combined.csv holds the measured genes on the scale stPlus saw them, beside the imputed
    genes, keyed by the spot barcodes. stplus_spatial.h5ad keeps the spatial file's own X (for the
    spots imputed) and adds obsm['stplus_imputed'], whose column names are in
    uns['stplus_imputed_genes']. Checkpoints are
    written to a fresh directory inside output_dir for each run and removed when it finishes.

    Parameters
    ----------
    spatial_df_path:
        Path to spatial transcriptomics data (.h5ad, a 10x .h5 matrix, or delimited text such as
        .csv/.tsv/.txt, optionally compressed). Rows are spots, columns are
        genes. For h5ad, the expression matrix (.X) is extracted.
    scrna_df_path:
        Path to scRNA-seq reference data (.h5ad, a 10x .h5 matrix, or delimited text such as
        .csv/.tsv/.txt, optionally compressed). Rows are cells, columns are
        genes. For h5ad, the expression matrix (.X) is extracted.
    genes_to_impute_path:
        Path to a gene list: one gene per line, or a delimited table whose
        first column holds the gene names (optionally compressed). These genes
        should be present in scrna_df but absent or poorly measured in spatial
        data. A listed gene that is also measured is set aside from the anchors
        and imputed; a gene listed twice is imputed once.
    output_dir:
        Directory where stPlus outputs will be saved.
    n_neighbors:
        Number of nearest reference cells each spot's imputed values are
        averaged over (default: 50). Must be at least 2 and at most the number
        of reference cells.
    n_epochs:
        Maximum number of training epochs for the autoencoder (default: 200).
        stPlus stops earlier once the training loss changes by less than 0.4%
        between epochs; ``params.epochs_run`` reports how many it trained.
    batch_size:
        Batch size for training (default: 512).
    device:
        Compute device: "auto" (follow the hardware), "cpu", "gpu"/"cuda", or "cuda:N".
    normalize:
        Input scale: "auto" (default) log-normalizes a matrix of raw counts and
        uses anything else as given; "always" log-normalizes both matrices;
        "never" uses both as given.
    top_k:
        Number of highly variable reference genes -- neither anchors nor genes
        to impute -- that stPlus appends to the model (default: 2000, stPlus's
        own default; fewer when the reference has fewer such genes). 0 appends
        none.
    """
    spatial_path = str(Path(spatial_df_path).expanduser())
    scrna_path = str(Path(scrna_df_path).expanduser())
    genes_path = str(Path(genes_to_impute_path).expanduser())
    out_dir = str(Path(output_dir).expanduser())

    args = [
        "--spatial-path",
        spatial_path,
        "--scrna-path",
        scrna_path,
        "--genes-path",
        genes_path,
        "--output-dir",
        out_dir,
        "--n-neighbors",
        str(n_neighbors),
        "--n-epochs",
        str(n_epochs),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--normalize",
        str(normalize),
        "--top-k",
        str(int(top_k)),
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
