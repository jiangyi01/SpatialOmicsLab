#!/usr/bin/env python3
"""CELINA cell type-specific spatially variable gene detection MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli
from worker_utils import pin_blas_threads

TOOL_NAME = "celina"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "CELINA",
    "/opt/conda/envs/celina_env/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/celina_worker.R",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_celina(
    spatial_counts_csv: str,
    spatial_coords_csv: str,
    cell_proportions_csv: str,
    sc_counts_csv: str,
    sc_celltype_labels_csv: str,
    output_dir: str,
    num_cores: int = 1,
    sc_celltype_column: str = "",
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Run CELINA cell type-specific spatially variable gene detection.

    CELINA identifies spatially variable genes (SVGs) whose spatial patterns
    are associated with specific cell types. It requires pre-computed cell type
    proportions (e.g., from SPOTlight, RCTD, or cell2location) AND a single-cell
    reference, in addition to spatial expression and coordinates: CELINA derives
    each cell type's marker gene list from the reference, so it cannot run without one.

    Workflow: Create_Celina_Object -> preprocess_input -> Calculate_Kernel ->
    Testing_interaction_all. Results are written as one long table with a cell_type
    column, a gene column, and CELINA's per-test columns (Gaussian1..5, Matern1..5,
    Spline, CombinedPvals); CombinedPvals is the significance CELINA reports.

    CELINA tests only the spots of spatial_counts_csv that have a coordinate row, are on the tissue and
    have a proportion row. Every counts spot is accounted for: data.n_spots_in_counts is what was supplied,
    data.n_spots what was analysed, and data.n_spots_without_coordinates, data.n_spots_off_tissue_dropped
    and data.n_spots_without_proportions say why the rest were left out (a deconvolution such as RCTD or
    CARD leaves low-count spots out of its proportions). Any cut adds a warning and a NOTE to the analysis.

    Parameters
    ----------
    spatial_counts_csv:
        Path to spatial gene expression counts CSV (genes x spots).
    spatial_coords_csv:
        Path to spatial coordinates CSV (spots as rows, spot names in the first column). Axis columns are
        matched by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col,
        row/col or x/y -- so a Space Ranger tissue-positions file can be passed as it is, headed
        (tissue_positions.csv, Space Ranger 2+) or headerless (tissue_positions_list.csv, Space Ranger 1).
        A first line whose every field after the barcode is a number is read as a spot, not as a header.
        Otherwise the two chosen axis columns are checked rather than read by position: one named more than
        once in the header, or named by a number (a spot's line in a partly-text header), is refused, and so
        is a constant or non-finite axis, before CELINA scales it to NaN.
        When the file carries Space Ranger's tissue flag (in_tissue, or the second field of the headerless
        file) as 0/1, spots flagged 0 are background and are left out; params.tissue_flag_column names the
        column read and params.in_tissue_filter counts the cut. A column of that name holding anything else
        (CELLxGENE's text 'tissue' label) is not used as a flag.
    cell_proportions_csv:
        Path to pre-computed cell type proportions CSV (spots x cell types). Each column besides the spot
        identifier is a cell type and must be numeric. The identifier column is found by name wherever it
        sits -- spot, barcode, spot_id, cell, cell_id or an unnamed first column -- so CARD's
        card_proportions.csv (cell types first, a trailing "spot" column) and a converter-style file (spot
        names first) both read as they are; with no such name the first column is taken. A missing or
        non-finite proportion in a tested cell type's column is refused; a column with no reference cells is
        not tested, CELINA does not read it, and its missing values are only counted
        (data.n_untested_proportion_values_not_finite).
    sc_counts_csv:
        Path to the single-cell reference counts CSV (genes x cells; cells as rows is also
        accepted and transposed). Only genes shared with the slide are used, and only cells with a row in
        sc_celltype_labels_csv (data.n_reference_cells_without_label_row counts the others).
    sc_celltype_labels_csv:
        Path to the single-cell cell type labels CSV: cell barcodes in the first column, then the label
        column(s). With exactly one column besides the barcodes (the converter's celltypes.csv) it is used;
        with more (the converter's metadata.csv) the run refuses unless sc_celltype_column names one -- the
        first column is never taken by position. The labels must use the same names as the columns of
        cell_proportions_csv -- only cell types present in both are tested.
    output_dir:
        Directory for CELINA output files (interaction results CSV, RDS object).
    num_cores:
        Number of cores for the interaction tests. CELINA has no default for this.
    sc_celltype_column:
        Column of sc_celltype_labels_csv holding the cell type labels. Leave empty when the file has one
        label column; a file with several is refused until one is named here. Echoed as
        params.sc_celltype_column (the column that was used, named or sole).
    drop_unlabeled:
        A reference cell with a missing label (NA, empty, "nan", "None") is not a cell type. CELINA averages
        each tested cell type over the cells with `labels == cell_type`, so an NA label puts NAs into every
        cell type's mean. False (default): an NA label, or a missing-label spelling that the proportions also
        carry as a column, is refused with the count; a blank label that no tested cell type selects is
        inert, is kept as before, and is counted in data.n_reference_cells_unlabeled. True: every cell with a
        missing label is left out and data.n_reference_cells_dropped says how many.

    Returns
    -------
    The worker payload. Besides the counts, data.coord_columns names the two coordinate columns CELINA was
    given and data.proportions_id_column the proportions column read as spot identifiers. summary.n_significant
    counts (cell type, gene) rows with CombinedPvals < 0.05, uncorrected for multiple testing.
    """
    # CELINA runs `num_cores` interaction tests in parallel; if EACH also lets its BLAS use every
    # core that is num_cores x n_cores threads, and the cores thrash instead of computing (commit
    # d1689ee measured load average ~328 and a stalled RCTD on a 96-core box). Pin BLAS to one
    # thread per worker so CELINA's own parallelism is the only parallelism. Must precede the
    # Rscript launch: OpenBLAS reads the count at R startup, and the subprocess inherits os.environ.
    pin_blas_threads()

    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--spatial-counts-csv",
        spatial_counts_csv,
        "--spatial-coords-csv",
        spatial_coords_csv,
        "--cell-proportions-csv",
        cell_proportions_csv,
        "--sc-counts-csv",
        sc_counts_csv,
        "--sc-celltype-labels-csv",
        sc_celltype_labels_csv,
        "--output-dir",
        output_dir,
        "--num-cores",
        str(num_cores),
    ]
    if sc_celltype_column:
        args += ["--sc-celltype-column", sc_celltype_column]
    if drop_unlabeled:
        args += ["--drop-unlabeled", "true"]

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
