#!/usr/bin/env python3
"""COMMOT spatial communication MCP wrapper for SpatialOmicsLab."""

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "commot"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "COMMOT",
    "/opt/conda/envs/COMMOT/bin/python",
    "/workspace/epic-fermat/agent/tools/commot_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def commot_spatial_communication(
    st_h5ad: str,
    output_dir: str,
    lr_database: str = "CellChat",
    species: str = "human",
    database_name: str = "CellChat",
    dis_thr: float = 0.0,
    heteromeric: bool = True,
    pathway_sum: bool = True,
    normalize: bool = True,
    dis_thr_unit: str = "coordinates",
    coords_key: str = "spatial",
    dims: int = 2,
    section_key: str | None = None,
    dis_thr_um: float = 0.0,
    block_sections: list[str] | str = "all",
    block_bbox_um: list[float] | None = None,
    block_max_cells: int = 40000,
    refuse_over_cap: bool = False,
) -> dict[str, Any]:
    """
    Run COMMOT spatial_communication on a spatial transcriptomics AnnData.

    Parameters
    ----------
    st_h5ad:
        Path to a spatial AnnData (.h5ad) with expression matrix and
        obsm['spatial'] coordinates. Gene names must be symbols of `species`; when var_names
        are Ensembl IDs and var has a symbol column (feature_name, gene_symbols, gene_symbol,
        gene_name, ...), genes are named by that column and the payload says so. Spots with
        obs['in_tissue'] == 0 (the background a CELLxGENE Visium export carries) are left out
        before normalisation and the ligand-receptor filter, so the outputs hold the in-tissue
        spots only; the count is in data.n_spots_off_tissue_dropped and params.in_tissue_filter.
    output_dir:
        Directory where COMMOT outputs will be written.
    lr_database:
        Ligand-receptor database: "CellChat" (human, mouse, zebrafish) or "CellPhoneDB_v4.0"
        (human, mouse) -- the only two the installed commot.pp.ligand_receptor_database
        implements; any other name is refused with the valid values. Only the "Secreted
        Signaling" pairs are loaded (upstream's default), then filtered to pairs whose ligand
        and receptor subunits are detected in at least 5% of spots.
    species:
        Species of the database: "human", "mouse", or "zebrafish" (CellChat only).
    database_name:
        Label used inside AnnData keys and output file names
        (commot_<database_name>_results.h5ad). A plain name; leave it equal to lr_database
        so the files say which database they came from.
    dis_thr:
        Maximum signalling distance. COMMOT measures it in the units of obsm['spatial'] as
        stored -- full-resolution image pixels for 10x Visium, where one 100 um spot pitch is
        roughly 30-370 pixels depending on the scan; usually microns for imaging platforms.
        0 (default) or less = automatic: 200 um, converted with the slide's
        uns['spatial'] scalefactors (spot_diameter_fullres spans 65 um, or microns_per_pixel),
        or 200 coordinate units when the file carries no scale. A threshold shorter than every
        spot's nearest-neighbour distance is refused (COMMOT would score only within-spot
        signalling). The payload reports the effective threshold in both units, the median
        spot spacing, and how many spot pairs fall within it. When the input carries
        obsp['spatial_distance'], COMMOT measures dis_thr on that matrix instead of
        obsm['spatial'], so an explicit value is in the matrix's units; the automatic value and
        dis_thr_unit="um" are converted only when that matrix is the Euclidean distance of
        obsm['spatial'], and are refused otherwise. A sparse obsp['spatial_distance'] is refused.
    heteromeric:
        Whether to handle heteromeric receptors.
    pathway_sum:
        Whether to compute pathway-level summarized communication scores.
    normalize:
        If True, run normalize_total + log1p inside the worker (X must be counts; a matrix
        marked log-transformed in uns['log1p'] is refused -- pass False to use it as is).
    dis_thr_unit:
        Unit of dis_thr: "coordinates" (default; obsm['spatial'] units, as COMMOT reads them)
        or "um" (microns, converted with the slide's scalefactors; refused when the file has
        none). Ignored when dis_thr is 0 (automatic 200 um). Kept for bare 2D data without units;
        a frame with units takes dis_thr_um.
    coords_key, dims, section_key:
        coords_key: the obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such
        as 'spatial_3d_aligned'). dims: 2 or 3; 3 builds the graph in the aligned frame in micrometres
        and needs a frame with recorded units and a measured or registered z. section_key: the obs
        column naming sections; required for a 2D run on a multi-section file (the run is per
        section) and for the cross-section edge count of a 3D run.
        For COMMOT: dims=2 with section_key writes one section_<label>/ folder per section
        (params.mode "per-section-2d"); dims=3 scores one bounded block (params.mode "3d"), whose
        result h5ad holds the block's three micrometre columns in obsm['spatial'], declared in
        uns['spatial_3d']['frames']['spatial'] (role aligned) -- the block result's own convention,
        an exception to the two-column spatial rule, so the explorer reads it as 3D. A 3D result counts
        the links that join two sections (data.n_cross_section_links of data.n_links_scored, and the
        candidate pairs within the threshold that do, data.n_cross_section_pairs_within); with none it
        says so and gives the closest pair of cells in two sections (data.closest_cross_section_um).
        Its params.frame.sections and commot_block.json name the sections scored. A 2D run on a
        file holding several sections without section_key is refused, naming both ways out.
    dis_thr_um:
        Maximum signalling distance in micrometres, for a frame with recorded units (0 = automatic:
        200 um in the frame's units). Replaces dis_thr/dis_thr_unit there; passing both is refused.
        A two-column obsm['spatial'] has units only when declared (uns['spatial_3d']['frames']['spatial']
        with xy_units, e.g. via spatial3d.contract.write_frame); on undeclared coordinates dis_thr_um is
        refused -- declare them, or pass dis_thr in coordinate units.
    block_sections, block_bbox_um, block_max_cells:
        block_*: the cells COMMOT scores in a 3D run -- a few adjacent sections and a bounding box,
        at most block_max_cells; a whole stack over the cap is refused with the count and the block
        offered. block_sections: section labels of section_key, or "all" (default).
        block_bbox_um: [a0, b0, a1, b1] in micrometres -- lower corner, then upper corner -- over
        the frame's two IN-PLANE axes in column order: the axes its axis_map does not name as the
        section stacking axis (without such an entry, x and y). On the Zhuang/Allen CCF frame, cut
        coronally and stacked along CCF x, a is CCF y (dorsal-ventral) and b is CCF z
        (medial-lateral): block_bbox_um=[4900, 1700, 5600, 2400] keeps CCF y 4900-5600 um and
        CCF z 1700-2400 um. commot_block.json records the axes. Null = the whole sections. block_max_cells (default 40000): COMMOT builds a dense n x n distance
        matrix (about 9 bytes x n squared); the refusal states it, the estimated total peak and
        the memory available.
    refuse_over_cap:
        Before any transport the run counts the cell pairs within the threshold (n_links) and the
        entries the result's obsp matrices may store, 2 x n_links + n_cells (stored_nnz_estimate,
        an upper bound). Over the explorer's cap (50,000,000 stored entries; SOG_CCC_MAX_NNZ) the
        direction, matrix and dot-plot views will be refused there, so the run warns -- or, with
        refuse_over_cap=True, refuses.
        The cell cap never implies the link cap.
    """
    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--task",
        "spatial_communication",
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--lr-database",
        lr_database,
        "--species",
        species,
        "--database-name",
        database_name,
        "--dis-thr",
        str(dis_thr),
        "--dis-thr-unit",
        dis_thr_unit,
    ]
    # Inside a portal turn the step has a time budget (SOG_TOOL_BUDGET_S, set only there): the worker fits the number
    # of ligand-receptor pairs it scores to it and says so (2026-10-05, PERF-1). A CLI run or a scored trial carries
    # no budget and scores every pair, exactly as before.
    budget = (os.environ.get("SOG_TOOL_BUDGET_S") or "").strip()
    if budget:
        args += ["--time-budget-s", budget]
    # The frame and block flags are passed only when they differ from the worker's defaults, so a plain 2D call
    # sends exactly the command line it always did.
    if coords_key != "spatial":
        args += ["--coords-key", coords_key]
    if int(dims) != 2:
        args += ["--dims", str(int(dims))]
    if section_key:
        args += ["--section-key", section_key]
    if dis_thr_um:
        args += ["--dis-thr-um", str(dis_thr_um)]
    sections = [block_sections] if isinstance(block_sections, str) else [str(s) for s in block_sections or []]
    if sections and sections != ["all"]:
        args += ["--block-sections", *sections]
    if block_bbox_um is not None:
        args += ["--block-bbox-um", *(str(float(v)) for v in block_bbox_um)]
    if int(block_max_cells) != 40000:
        args += ["--block-max-cells", str(int(block_max_cells))]
    if refuse_over_cap:
        args.append("--refuse-over-cap")
    if not heteromeric:
        args.append("--no-heteromeric")
    if not pathway_sum:
        args.append("--no-pathway-sum")
    if not normalize:
        args.append("--no-normalize")

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
