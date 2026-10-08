#!/usr/bin/env python3
"""SpotSweeper spatially-aware QC MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "spotsweeper"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "SPOTSWEEPER",
    "/opt/conda/envs/spotsweeper_env2/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/spotsweeper_worker.R",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_spotsweeper(
    spatial_counts_csv: str,
    spatial_coords_csv: str,
    output_dir: str,
    n_neighbors: int = 36,
    threshold: float = 3.0,
    allow_array_index_fallback: bool = False,
) -> dict[str, Any]:
    """
    Run SpotSweeper spatially-aware quality control on spatial transcriptomics data.

    SpotSweeper detects local outliers and regional artifacts in spot-based
    spatial transcriptomics data (e.g. 10x Visium) by comparing each spot's
    QC metrics (library size, unique genes, mitochondrial ratio) against its
    spatial neighbors using a modified z-score approach.

    The tool runs two main analyses:
      1. localOutliers() -- flags spots with unusually low library size,
         low unique gene counts, or high mitochondrial ratio relative to
         their spatial neighborhood.
      2. findArtifacts() -- identifies regional artifacts based on local
         variance patterns in mitochondrial expression (requires gene names
         starting MT- or mt-; Ensembl IDs are not recognised). It always splits
         the slide into two k-means clusters and labels one of them artifact,
         so its count is never zero on a slide it runs on.

    Spots are flagged in the QC table, not removed. When findArtifacts is
    skipped or fails, ``artifact_summary.status`` says which ("skipped" /
    "failed"), ``artifact_summary.findArtifacts_error`` carries the error, and
    the local outlier results are still returned. ``params`` records the
    coordinate columns read and what they are (``coordinate_kind``), whether
    the file had a header row, the method that ran and the neighbourhood size
    actually used. The counts are read sparse.

    Parameters
    ----------
    spatial_counts_csv:
        Path to spatial gene expression counts CSV (genes x spots), raw counts.
        Spots with no row in the coordinates file are left out of the analysis
        and counted in ``data.n_spots_without_coords``. Missing or negative
        values are refused; non-integer values are named in a warning.
    spatial_coords_csv:
        Path to spatial coordinates CSV (spots as rows, spot names in the first column). Axis columns are
        matched by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col,
        row/col or x/y -- so Space Ranger's tissue_positions.csv can be passed as it is. A file with no
        header row is read as Space Ranger's headerless tissue_positions_list.csv (six columns: barcode,
        in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres) or, with three columns,
        as barcode, x, y; any other headerless layout is refused. When the file has an in_tissue column
        (Space Ranger's positions files, the converter's metadata.csv), spots it marks 0 (background
        outside the tissue) are left out, counted in ``data.n_spots_off_tissue`` and
        ``params.in_tissue_filter``, and named in a warning. SpotSweeper's neighbourhoods are Euclidean,
        so the axes must be positions: when the file names only array indices (array_row/array_col or
        row/col) and also carries x/y, x/y are read instead; a square lattice's indices are read as
        they are; Visium's hexagonal indices -- the only positions in the converter's metadata.csv
        for a CELLxGENE Visium object -- are refused unless allow_array_index_fallback is set
        (``params.coordinate_kind`` says which).
    output_dir:
        Directory for SpotSweeper output files (QC results CSV, SPE RDS).
    n_neighbors:
        Number of nearest neighbors for local outlier detection (default 36,
        about three hexagonal rings of Visium spots: 6 + 12 + 18). On a slide
        with fewer spots it is capped at n_spots - 1, reported as
        ``n_neighbors_effective``. Must be a positive whole number.
    threshold:
        Modified z-score cutoff for calling outliers (default 3.0). Must be
        finite.
    allow_array_index_fallback:
        With only Visium's hexagonal array indices for positions, read them as the lattice's
        physical layout (imagerow = array_row x sqrt(3), imagecol = array_col, which puts all six
        neighbours of a spot one pitch apart) instead of refusing (default: false). Reported as
        ``params.used_fallback`` and ``params.coordinate_kind``, with a warning.
    """
    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--spatial-counts-csv",
        spatial_counts_csv,
        "--spatial-coords-csv",
        spatial_coords_csv,
        "--output-dir",
        output_dir,
        "--n-neighbors",
        str(n_neighbors),
        "--threshold",
        str(threshold),
        "--allow-array-index-fallback",
        str(bool(allow_array_index_fallback)).lower(),
    ]

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
