#!/usr/bin/env python3
"""IRIS spatial domain identification MCP wrapper for SpatialOmicsLab (one slice per call)."""

from __future__ import annotations

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "iris_spatial"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "IRIS",
    "/opt/conda/envs/iris_spatial_env/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/iris_spatial_worker.R",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_iris(
    spatial_counts_csv: str,
    spatial_coords_csv: str,
    output_dir: str,
    ref_counts_csv: str = "",
    ref_celltypes_csv: str = "",
    n_clusters: int = 7,
    seed: int = 42,
    ct_varname: str = "",
    sample_varname: str = "",
    drop_unlabeled: bool = False,
    min_spot_counts: int = 100,
    min_gene_spots: int = 5,
) -> dict[str, Any]:
    """
    Run IRIS spatial domain identification on one slice.

    IRIS identifies spatial domains by integrating gene expression with spatial coordinates. The
    method can integrate several slices; this wrapper takes one counts/coordinates pair and runs it
    as a single slice. With ref_counts_csv and ref_celltypes_csv it runs reference-based IRIS; with
    neither it runs IRISfree on marker groups this wrapper derives from the spatial counts (the top
    100 genes by variance cut into contiguous blocks), which are gene blocks rather than cell types.
    params.mode and params.method name the one that ran.

    IRIS's own QC leaves spots out: createIRISObject keeps a gene only when it is detected in more
    than min_gene_spots spots and a spot only when its total over those genes reaches
    min_spot_counts, and IRIS_spatial then drops spots with no counts on the genes it models. Those
    spots get no domain and no row in iris_domains.csv. data.n_spots is the number of spots that got
    a domain; data.n_spots_input, n_spots_dropped and a warning account for the rest, and
    data.n_genes is the number of informative genes the model ran on.

    Parameters
    ----------
    spatial_counts_csv:
        Path to spatial gene expression counts CSV (genes x spots).
    spatial_coords_csv:
        Path to spatial coordinates CSV (spots as rows, spot names in the first column). Axis columns are
        matched by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col,
        row/col or x/y -- so Space Ranger's tissue_positions.csv can be passed as it is, and so can its
        headerless tissue_positions_list.csv (Space Ranger before 2.0), which is read under Space Ranger's
        six column names (params.coordinates_header says which, params.coordinate_columns which two were
        read). Spots the file marks in_tissue = 0 (background outside the tissue) are left out and counted
        in data.n_spots_off_tissue_dropped, params.in_tissue_filter and a warning.
    output_dir:
        Directory for IRIS output files (domain labels CSV).
    ref_counts_csv:
        Optional path to a scRNA-seq reference counts CSV, supplied together with ref_celltypes_csv.
        Giving both switches the tool from IRISfree to reference-based IRIS. Orientation is read off
        ref_celltypes_csv's row names, so either genes x cells or cells x genes is accepted.
    ref_celltypes_csv:
        Optional path to the cell type annotation CSV for ref_counts_csv, with the cell barcodes as
        row names. Rows are matched to the counts by cell ID, so their order does not matter. The
        cell type column is the one ct_varname names. Left empty, the first of these rules with a
        single hit decides: a column named celltype or cell_type (any case), the only column whose
        name contains celltype or cell_type, a column named type, one named cluster, the only
        column whose name contains cluster, a lone column. Anything else is refused with the list of
        columns rather than guessed. IRIS needs at least 3 cell types with two or more cells each.
    n_clusters:
        Number of spatial domains (clusters) to identify.
    seed:
        Accepted and recorded, but it has no effect: IRIS reseeds every random step itself
        (set.seed(islice) before its Dirichlet/LIGER start, set.seed(12345678) in its k-means
        helpers, rliger optimizeALS rand.seed = 1), so the run is deterministic. Listed in
        params.ignored.
    ct_varname:
        Column of ref_celltypes_csv that holds the cell types. Empty matches it by name as described
        under ref_celltypes_csv. Reference mode only; recorded in params.ignored without a reference.
    sample_varname:
        Column of ref_celltypes_csv that holds the sample (donor/batch) of each cell; IRIS averages
        each cell type's reference profile over these samples. Empty uses the first column whose
        name contains sample, batch or donor (a warning names the others when several do), else
        treats all cells as one sample. Reference mode only.
    drop_unlabeled:
        Reference mode only. False (default) stops the run when a reference cell has no label (NA,
        empty or no row in ref_celltypes_csv); True leaves those cells out and reports how many.
    min_spot_counts:
        createIRISObject's minCountGene: a spot is kept only when its total count over the genes
        that pass min_gene_spots is at least this. Default 100, IRIS's own default.
    min_gene_spots:
        createIRISObject's minCountSpot: a gene is kept only when it is detected in more than this
        many spots. Default 5, IRIS's own default.
    """
    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--spatial-counts-csv",
        spatial_counts_csv,
        "--spatial-coords-csv",
        spatial_coords_csv,
        "--output-dir",
        output_dir,
        "--n-clusters",
        str(n_clusters),
        "--seed",
        str(seed),
        "--min-spot-counts",
        str(min_spot_counts),
        "--min-gene-spots",
        str(min_gene_spots),
    ]

    if ref_counts_csv:
        args.extend(["--ref-counts-csv", ref_counts_csv])
    if ref_celltypes_csv:
        args.extend(["--ref-celltypes-csv", ref_celltypes_csv])
    if ct_varname:
        args.extend(["--ct-varname", ct_varname])
    if sample_varname:
        args.extend(["--sample-varname", sample_varname])
    if drop_unlabeled:
        args.extend(["--drop-unlabeled", "true"])

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
