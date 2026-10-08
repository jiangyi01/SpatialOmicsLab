#!/usr/bin/env python3
"""CytoSPACE cell-to-spot assignment MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "cytospace"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "CYTOSPACE",
    "/opt/conda/envs/cytospace_env/bin/python3.9",
    "/workspace/epic-fermat/agent/tools/cytospace_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_cytospace(
    sc_h5ad_path: str,
    spatial_h5ad_path: str,
    output_dir: str,
    cell_type_key: str = "cell_type",
    n_cells: int = 0,
    n_top_genes: int = 5000,
    seed: int = 0,
    single_cell: bool = False,
    mean_cell_numbers: int = 5,
    cell_type_fractions_path: str = "",
    drop_unlabeled: bool = False,
    timeout_s: int = 1100,
) -> dict[str, Any]:
    """
    Run CytoSPACE to assign single cells to spatial spots.

    CytoSPACE solves a linear assignment (lapjv) of reference cells to spatial locations using
    scRNA-seq and spatial transcriptomics data. The worker converts h5ad inputs to the
    tab-delimited text files CytoSPACE reads.

    How many cells of each type are placed is fixed in advance by a cell-type fractions table.
    Upstream estimates it from the spatial data with an R/Seurat script that this environment
    cannot run, so by default the fractions are the scRNA-seq reference's own label frequencies:
    the assigned composition then mirrors the reference and is not a finding about the tissue.
    Pass ``cell_type_fractions_path`` to set it from the slide. The payload names the source
    (``params.cell_type_fraction_source``) and the method that ran (``params.method``).

    Spatial locations with ``obs['in_tissue'] == 0`` (background) receive no cells: they are left
    out and counted in ``params.in_tissue_filter`` and the warnings. CytoSPACE holds its inputs as
    dense frames; a run whose estimated peak does not fit in memory is refused with the numbers
    before anything is staged, and never subsampled.

    Parameters
    ----------
    sc_h5ad_path:
        Path to the scRNA-seq AnnData (.h5ad) with cell type annotations.
    spatial_h5ad_path:
        Path to the spatial AnnData (.h5ad) file.
    output_dir:
        Directory to write CytoSPACE outputs.
    cell_type_key:
        Column in sc_h5ad.obs containing cell type labels.
    n_cells:
        Optional cap on reference cells. 0 (default) uses every cell of the reference. A
        positive value keeps that many cells, drawn at random with ``seed``; the cut is reported
        in the payload's warnings and analysis.
    n_top_genes:
        Match on the top N genes shared by both inputs, ranked by raw (unnormalised) variance
        in the spatial data -- not a highly-variable-gene selection. Reduces the problem size
        for large panels. Set to 0 to match on every shared gene.
    seed:
        Random seed for reproducibility (the optional reference draw and the solver).
    single_cell:
        Place exactly one reference cell per location (CytoSPACE ``--single-cell``). Use for
        segmented-cell data (Xenium, CosMx, MERFISH) and Visium HD bins of 8 um or less.
        Default False: Visium spot mode, several cells per location.
    mean_cell_numbers:
        Spot mode only: mean number of cells per location (CytoSPACE ``--mean-cell-numbers``;
        5 suits Visium, about 20 legacy ST). Each location's count is estimated from its RNA
        content around this mean. Ignored when single_cell is True.
    cell_type_fractions_path:
        Optional table of cell-type fractions estimated from the spatial data: one row with cell
        types as columns, a spots x cell types proportion table (e.g. a deconvolution result;
        summed per type), or one column indexed by cell type. Types must be reference labels.
        The spot-identifier column is found wherever it sits (first and unnamed, or named spot,
        barcode, spot_id, cell_id ...), so run_spotlight and run_card output (trailing "spot"
        column) is read as it is. A row with no value at all (a spot the deconvolution left
        without a composition) is left out and counted in
        ``params.cell_type_fraction_rows_without_values``; a partly empty row is refused. A name
        that is the '/'->'_' rewrite deconvolution tools apply to exactly one reference label
        (SPOTlight's 'Treg_Tfr' for 'Treg/Tfr') is matched to it and listed in
        ``params.cell_type_fraction_names_matched``.
        Empty (default): the reference's own label frequencies are used.
    drop_unlabeled:
        Leave out reference cells whose label is missing (NaN/empty). Default False: such cells
        stop the run with their count, because a missing label is not a cell type.
    timeout_s:
        Seconds the CytoSPACE solver subprocess may run before it is stopped (0 = no limit).
    """
    args = [
        "--sc-h5ad",
        sc_h5ad_path,
        "--spatial-h5ad",
        spatial_h5ad_path,
        "--output-dir",
        output_dir,
        "--cell-type-key",
        cell_type_key,
        "--n-cells",
        str(n_cells),
        "--n-top-genes",
        str(n_top_genes),
        "--seed",
        str(seed),
        "--mean-cell-numbers",
        str(mean_cell_numbers),
        "--timeout-s",
        str(timeout_s),
    ]
    if single_cell:
        args.append("--single-cell")
    if cell_type_fractions_path:
        args += ["--cell-type-fractions-path", cell_type_fractions_path]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
