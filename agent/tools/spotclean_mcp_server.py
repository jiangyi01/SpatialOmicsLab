#!/usr/bin/env python3
"""SpotClean contamination cleanup MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "spotclean"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "SPOTCLEAN",
    "/opt/conda/envs/spotclean_env/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/spotclean_worker.R",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_spotclean(
    spatial_counts_csv: str,
    spatial_coords_csv: str,
    output_dir: str,
    gene_cutoff: int = 10,
    spot_cutoff: int = 100,
    verbose: bool = True,
    allow_array_index_fallback: bool = False,
) -> dict[str, Any]:
    """
    Run SpotClean contamination cleanup on spatial transcriptomics data.

    SpotClean models and removes ambient RNA contamination (bleeding) in
    spatial transcriptomics data by leveraging the spatial pattern of
    background signal.

    Parameters
    ----------
    spatial_counts_csv:
        Path to spatial gene expression counts CSV (genes x spots).
    spatial_coords_csv:
        Path to spatial coordinates CSV (spots as rows, spot names in the first column). Axis columns are
        matched by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres or x/y -- so a Space
        Ranger tissue-positions file can be passed as it is, headed (v2) or headerless (v1). Give pixel
        coordinates rather than array indices: SpotClean's kernel bandwidth is a distance. A file with
        no pixel columns is read by its array indices (array_row/array_col, else row/col) only where
        they are distances: a square lattice's are used as given, while Visium's hexagonal indices --
        what the converter's metadata.csv for a CELLxGENE Visium object carries -- are refused unless
        allow_array_index_fallback is set (params.coordinate_kind says which). The
        tissue/background flag is read from a 'tissue' or 'in_tissue' column holding 0/1 (or
        TRUE/FALSE); a column of either name that holds anything else -- CELLxGENE's text 'tissue'
        label, say -- is set aside and named in warnings, and the column used is reported in
        params.tissue_flag_column. Without background (0) spots nothing is decontaminated, and the
        warning says why: no flag column; a flag that marks every spot as tissue; background the
        counts do not carry (a filtered matrix -- SpotClean needs Space Ranger's raw_feature_bc_matrix);
        or background spot_cutoff removed. data.n_background_spots_in_coordinates,
        data.n_background_spots_not_in_counts and data.n_background_spots_removed_by_cutoff count them.
    output_dir:
        Directory for SpotClean output files (cleaned counts CSV, QC metrics).
    gene_cutoff:
        Minimum total count per gene; genes below this threshold are filtered.
    spot_cutoff:
        Minimum total count per spot; spots below this threshold are filtered -- background spots
        too, which are what SpotClean estimates the ambient RNA from, and which usually have few
        counts. How many it removed is data.n_background_spots_removed_by_cutoff; 0 keeps every spot.
    verbose:
        Whether to print verbose progress messages during cleanup.
    allow_array_index_fallback:
        With no pixel columns in spatial_coords_csv, read Visium's hexagonal array indices as the
        lattice's physical layout (imagerow = array_row x sqrt(3), imagecol = array_col, which puts
        all six neighbours of a spot at the same distance) instead of refusing (default: false).
        Reported as params.used_fallback and params.coordinate_kind, with a warning.
    """
    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--spatial-counts-csv",
        spatial_counts_csv,
        "--spatial-coords-csv",
        spatial_coords_csv,
        "--output-dir",
        output_dir,
        "--gene-cutoff",
        str(gene_cutoff),
        "--spot-cutoff",
        str(spot_cutoff),
        "--verbose",
        str(verbose).upper(),
        "--allow-array-index-fallback",
        str(bool(allow_array_index_fallback)).upper(),
    ]

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
