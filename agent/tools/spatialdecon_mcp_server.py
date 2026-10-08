#!/usr/bin/env python3
"""SpatialDecon Nanostring spatial deconvolution MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "spatialdecon"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "SPATIALDECON",
    "/opt/conda/envs/spatialdecon_env/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/spatialdecon_worker.R",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_spatialdecon(
    spatial_counts_csv: str,
    ref_counts_csv: str,
    ref_celltypes_csv: str,
    output_dir: str,
    normalize: bool = True,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Run SpatialDecon deconvolution on spatial transcriptomics data.

    SpatialDecon (Nanostring) estimates cell type abundances in spatial spots
    using constrained log-normal regression against a profile matrix. The worker
    builds that profile from the scRNA-seq reference as the per-cell-type mean of
    the reference cells' counts, and uses a constant background of 0.1 (there
    are no negative probes outside GeoMx data); both are reported in params.
    A spot with no counts on the shared genes is still fitted by SpatialDecon
    and gets a composition no data informed; the payload counts those spots in
    data.n_spots_without_counts and names one in a warning.

    Memory: both CSVs are read block-wise into sparse matrices. The reference
    is reduced to its per-cell-type mean profile as one sparse product and is
    never made dense, so a whole-atlas reference costs its non-zero entries.
    The spots are dense by necessity: spatialdecon() takes them as a base
    matrix and rbind()s it with the background and the weights while it fits,
    so the shared-genes x spots matrix is 8 bytes per entry with at least six
    copies alive at once. A run whose lower bound exceeds the available memory
    stops before SpatialDecon starts, with the numbers. Missing or negative
    values in either CSV are refused: the profile and the log-normal model take
    non-negative expression (raw or normalised counts).

    Parameters
    ----------
    spatial_counts_csv:
        Path to spatial gene expression counts CSV (genes x spots; spots x genes
        is detected from the gene names and transposed).
    ref_counts_csv:
        Path to the scRNA-seq reference counts CSV at cell level: genes x cells
        (cells x genes is detected and transposed), one column per cell barcode
        listed in ref_celltypes_csv. A ready-made genes x cell-types profile matrix
        is not accepted: its columns are cell-type names, not barcodes, and the run
        stops saying so.
    ref_celltypes_csv:
        Path to reference cell type annotation CSV (cell barcode in the first
        column, cell type label in the second). Used to average the reference
        cells into the profile matrix.
    output_dir:
        Directory for SpatialDecon output files (proportions CSV).
    normalize:
        Whether to normalize spatial counts before deconvolution (default True).
    drop_unlabeled:
        Reference cells whose label is missing (NA, empty, 'nan') are refused by
        default, because a missing label is not a cell type. True leaves them out
        of the profile and reports how many.
    """
    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--spatial-counts-csv",
        spatial_counts_csv,
        "--ref-counts-csv",
        ref_counts_csv,
        "--ref-celltypes-csv",
        ref_celltypes_csv,
        "--output-dir",
        output_dir,
        "--normalize",
        str(normalize).lower(),
    ]
    if drop_unlabeled:
        args += ["--drop-unlabeled", "true"]

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
