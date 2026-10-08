#!/usr/bin/env python3
"""STdeconvolve reference-free LDA-based deconvolution MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "stdeconvolve"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "STDECONVOLVE",
    "/opt/conda/envs/stdeconvolve_env/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/stdeconvolve_worker.R",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_stdeconvolve(
    spatial_counts_csv: str,
    output_dir: str,
    n_topics: int = 10,
    n_top_genes: int = 1000,
    remove_below: float | None = None,
    seed: int = 42,
    counts_orientation: str = "auto",
) -> dict[str, Any]:
    """
    Run STdeconvolve reference-free LDA-based deconvolution on spatial transcriptomics data.

    STdeconvolve uses Latent Dirichlet Allocation (LDA) to identify latent cell
    type topics and their proportions across spatial spots WITHOUT requiring a
    single-cell RNA-seq reference.

    Spots with zero counts, and spots with no counts in the over-dispersed genes kept for
    fitting, are empty LDA documents that STdeconvolve cannot place; they are left out, counted in
    ``data.n_spots_empty`` / ``data.n_spots_no_corpus_counts``, named in ``warnings``, and have no
    row in ``stdeconvolve_theta.csv`` (``data.n_spots_used`` is its row count).

    Parameters
    ----------
    spatial_counts_csv:
        Path to spatial gene expression counts CSV, raw integer counts, genes x spots or
        spots x genes, with identifiers in the first column and the header row. Which axis holds
        the genes is read off the identifiers (Ensembl IDs and gene symbols vs spot barcodes);
        when they do not settle it the run stops and asks for ``counts_orientation``.
    output_dir:
        Directory for STdeconvolve output files (theta and beta CSVs).
    n_topics:
        Number of LDA topics (cell types) to fit.
    n_top_genes:
        Number of top over-dispersed genes to select for LDA fitting.
    remove_below:
        Minimum fraction of spots a gene must be detected in to be kept. Leave unset to let
        the worker choose from the data (0.01 when the median per-spot UMI of the spots with
        counts is below 500, as on Slide-seq and MERFISH, otherwise 0.05); the payload reports
        which one ran.
    seed:
        Random seed for reproducibility.
    counts_orientation:
        'auto' (default) detects which axis holds the genes from the row and column names;
        'genes_x_spots' or 'spots_x_genes' says so outright. The payload reports the orientation
        used and how it was decided (``params.counts_orientation``,
        ``params.counts_orientation_source``).
    """
    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--spatial-counts-csv",
        spatial_counts_csv,
        "--output-dir",
        output_dir,
        "--n-topics",
        str(n_topics),
        "--n-top-genes",
        str(n_top_genes),
        "--seed",
        str(seed),
    ]

    # Forward nothing when unset: the worker picks the filter from the data, and a
    # portal-side default would silently overrule that choice on every call.
    if remove_below is not None:
        args += ["--remove-below", str(remove_below)]
    # Same convention: an unchanged call builds the argv it always did.
    if counts_orientation != "auto":
        args += ["--counts-orientation", counts_orientation]

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
