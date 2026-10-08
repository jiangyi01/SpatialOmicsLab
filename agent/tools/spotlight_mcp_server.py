#!/usr/bin/env python3
"""SPOTlight NMF deconvolution MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

import os
from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "spotlight"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "SPOTLIGHT",
    "/opt/conda/envs/spotlight_env/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/spotlight_worker.R",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_spotlight(
    spatial_counts_csv: str,
    spatial_coords_csv: str | None = None,
    ref_counts_csv: str | None = None,
    ref_celltypes_csv: str | None = None,
    output_dir: str = default_output_dir("spotlight_output"),
    n_top: int | None = 100,
    min_cont: float | None = 0.09,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Run SPOTlight NMF-based deconvolution on spatial transcriptomics data.

    SPOTlight uses seeded non-negative matrix factorization (NMF) to decompose
    each spatial spot into cell type proportions by leveraging a single-cell
    RNA-seq reference.

    Marker genes follow the SPOTlight vignette: genes whose mean.AUC (the mean,
    over every other cell type, of the pairwise AUC on log-normalised reference
    counts; what scran::scoreMarkers reports) exceeds 0.8. The worker calls
    scran when it is installed and otherwise computes the same statistic in
    base R; ``params.marker_method`` says which one ran. A cell type with no
    gene above 0.8 is seeded with its 25 best genes by mean.AUC instead, and the
    payload lists it under ``summary.cell_types_seeded_by_rank`` and in
    ``warnings``. SPOTlight seeds each topic with the type's top ``n_top``
    markers minus every gene another type's top list also holds; a type left
    with none starts unseeded and is listed in
    ``summary.cell_types_without_usable_markers`` (seed counts per type in
    ``summary.n_seed_genes_per_cell_type``).

    SPOTlight pairs topics with cell types by position, which only lines up
    when the reference's cells come grouped in sorted cell-type order, so the
    worker hands them over in that order (``params.reference_cells_regrouped_by_type``
    says whether it had to); no value changes.

    Parameters
    ----------
    spatial_counts_csv:
        Path to spatial gene expression counts CSV (genes x spots). REQUIRED.
    spatial_coords_csv:
        Path to spatial coordinates CSV (spots as rows, spot names in the first column). Axis columns are
        matched by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col,
        row/col or x/y -- so Space Ranger's tissue_positions.csv can be passed as it is, and so can the
        headerless tissue_positions_list.csv of Space Ranger 1 (``params.coordinates_header`` says how
        the first line was read). When the file carries an ``in_tissue`` column, spots of the counts
        marked 0 (background) are left out and counted in ``params.in_tissue_filter`` and ``warnings``.
        Required at runtime — if omitted, the wrapper returns a clean error
        rather than letting Pydantic raise a validation error in the MCP framework.
    ref_counts_csv:
        Path to scRNA-seq reference counts CSV (genes x cells). Required at
        runtime; same null-check pattern as above.
    ref_celltypes_csv:
        Path to reference cell type annotation CSV (cell barcode as index,
        first column is cell type label; with more columns, the first one named
        celltype, cell_type, cell.type, annotation, annot, cluster or label is
        used). Required at runtime; same null-check pattern as above.
    output_dir:
        Directory for SPOTlight output files (proportions CSV, RDS object).
    n_top:
        Number of top marker genes per cell type (ranked by mean.AUC) that seed
        each cell type's NMF topic; passed to SPOTlight as ``n_top``. A cell type
        with fewer markers uses all of its own. ``None`` sends no flag and the
        worker uses 100.
    min_cont:
        Per-spot proportion threshold, passed to SPOTlight as ``min_prop``: in
        each spot a cell type whose share is below it is set to zero and the
        remaining shares are rescaled to sum to 1. Must lie in [0, 1]. A spot in
        which every cell type falls below it has no proportions left and is
        written as NA; the payload counts those spots in
        ``summary.n_spots_without_proportions``. ``None`` sends no flag and the
        worker uses 0.09.
    drop_unlabeled:
        Reference cells whose label is missing (NA, empty, "nan", "none") are
        not a cell type. False (default): the run stops and says how many there
        are. True: they are left out and counted in
        ``data.n_ref_cells_unlabeled_dropped``.
    """
    missing = []
    if spatial_coords_csv is None:
        missing.append("spatial_coords_csv")
    if ref_counts_csv is None:
        missing.append("ref_counts_csv")
    if ref_celltypes_csv is None:
        missing.append("ref_celltypes_csv")
    if missing:
        return {
            "status": "error",
            "tool": TOOL_NAME,
            "error": (
                f"Missing required input(s): {', '.join(missing)}. "
                "SPOTlight needs all four inputs (spatial_counts_csv, spatial_coords_csv, "
                "ref_counts_csv, ref_celltypes_csv) plus an output_dir."
            ),
        }

    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--spatial-counts-csv",
        spatial_counts_csv,
        "--spatial-coords-csv",
        spatial_coords_csv,
        "--ref-counts-csv",
        ref_counts_csv,
        "--ref-celltypes-csv",
        ref_celltypes_csv,
        "--output-dir",
        output_dir,
    ]
    if n_top is not None:
        args += ["--n-top", str(n_top)]
    if min_cont is not None:
        args += ["--min-cont", str(min_cont)]
    if drop_unlabeled:
        args += ["--drop-unlabeled", "true"]

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
