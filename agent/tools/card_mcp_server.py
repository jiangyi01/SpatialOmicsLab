#!/usr/bin/env python3
"""CARD spatially-informed deconvolution MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "card"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "CARD",
    "/opt/conda/envs/card_env/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/card_worker.R",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_card(
    spatial_counts_csv: str,
    spatial_coords_csv: str,
    ref_counts_csv: str,
    ref_celltypes_csv: str,
    output_dir: str,
    min_count_gene: int = 100,
    min_count_spot: int = 5,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Run CARD spatially-informed deconvolution on spatial transcriptomics data.

    CARD (Conditional Autoregressive-based Deconvolution) estimates cell type
    proportions for each spatial location by combining a scRNA-seq reference
    with spatial correlation modeling via a conditional autoregressive framework.

    Parameters
    ----------
    spatial_counts_csv:
        Path to spatial gene expression counts CSV (genes x spots). CARD models raw counts: a negative,
        missing or non-numeric value on a spot CARD analyses stops the run by name; non-integer
        (normalised) values run as before with a warning. params.spatial_counts_kind says which.
    spatial_coords_csv:
        Path to spatial coordinates CSV (spots as rows, spot names in the first column). Axis columns are
        matched by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col,
        row/col or x/y -- so Space Ranger's tissue_positions.csv can be passed as it is. Space Ranger 1's
        headerless tissue_positions_list.csv is recognised too (its first line is a spot, not a header),
        and its pixel row/col are read; data.coord_columns names the two columns read. A chosen column whose
        name appears twice, a non-numeric coordinate on a matched spot, or every matched spot at one position
        stops the run. A counts spot with no coordinate row is left out and counted (data.n_spots_in_counts
        and the warnings). Background spots are left out too: a column named in_tissue (Space Ranger's
        tissue_positions.csv, the converter's metadata.csv), or the tissue flag of the headerless
        tissue_positions_list.csv, marks them 0; they get no row in card_proportions.csv and are counted
        in data.n_spots_off_tissue_dropped, params.in_tissue_filter and the warnings. data.n_spots counts
        the in-tissue spots that enter CARD.
    ref_counts_csv:
        Path to scRNA-seq reference counts CSV (genes x cells). A cell with no row in ref_celltypes_csv is
        left out of the reference and counted (data.n_ref_cells_in_counts and the warnings). Its values
        are checked as spatial_counts_csv's are (params.ref_counts_kind).
    ref_celltypes_csv:
        Path to reference cell type annotation CSV (cell barcode as index). With one column, that column
        is the label. With several (e.g. the converter's metadata.csv), the label column is chosen by
        name, in tiers -- celltype/cell_type/cell.type, then annotation/annot, then label, then cluster
        (any case; the first in file order within a tier) -- and params.ref_celltype_column says which
        was read. Pass a one-column file to choose another.
    output_dir:
        Directory for CARD output files: card_proportions.csv (one row per spot CARD kept, cell types
        as columns, spot names in a trailing ``spot`` column) and card_result.rds.
    min_count_gene:
        CARD's ``minCountGene``, which despite its name is a SPOT filter: a spot whose total count, over
        the genes kept by ``min_count_spot``, is below this value is dropped (as is any spot above 1e6).
        Dropped spots have no row in card_proportions.csv; data.n_spots_used and the warnings say how many
        and why. 0 turns the lower bound off.
    min_count_spot:
        CARD's ``minCountSpot``, which despite its name is a GENE filter: a spatial gene that is non-zero
        in this many spots or fewer is dropped before deconvolution.
    drop_unlabeled:
        Reference cells whose label is missing (NA, empty, "nan", "none") are not a cell type. False
        (default) stops the run and says how many there are; True leaves them out of the reference and
        reports the count in data.n_ref_cells_unlabeled_dropped.

    CARD's spatial kernel is a dense spots x spots matrix (intrinsic to the method); a slide whose kernel
    cannot fit in the memory available (MemAvailable, bounded by the room left under a cgroup memory
    limit) is refused before the fit with the sizes involved.
    """
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
        "--min-count-gene",
        str(min_count_gene),
        "--min-count-spot",
        str(min_count_spot),
    ]
    if drop_unlabeled:
        args += ["--drop-unlabeled", "true"]

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
