#!/usr/bin/env python3
"""PRECAST spatially-aware clustering and multi-sample integration MCP wrapper for SpatialOmicsLab."""

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli
from worker_utils import pin_blas_threads

TOOL_NAME = "precast"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "PRECAST",
    "/opt/conda/envs/precast/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/precast_worker.R",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def precast_spatial_clustering(
    counts_csvs: list[str],
    coords_csvs: list[str],
    output_dir: str,
    K: int = 7,
    platform: str = "auto",
    gene_number: int = 2000,
    core_num: int = 1,
    max_iter: int = 50,
    sigma_equal: bool = False,
    seed: int = 0,
    allow_fixed_number_fallback: bool = False,
) -> dict[str, Any]:
    """
    Run PRECAST spatially-aware clustering and multi-sample integration on
    spatial transcriptomics data.

    PRECAST (PRobabilistic Embedding, Clustering, and Alignment for Spatial
    Transcriptomics) performs spatially-aware clustering while aligning
    embeddings across multiple tissue slices. It identifies shared spatial
    domains across samples even with batch effects.

    Parameters
    ----------
    counts_csvs:
        List of paths to gene expression count CSV files, one per sample
        (genes x spots, with row/column names). Each file represents one
        spatial tissue slice.
    coords_csvs:
        List of paths to coordinate CSV files, one per sample (spots as rows, spot names in the first
        column, matching that sample's counts columns). Must have same length as counts_csvs. Axis
        columns are matched by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres,
        array_row/array_col, row/col or x/y -- so Space Ranger's tissue_positions.csv can be passed
        as it is, and so can the headerless tissue_positions_list.csv (a first line that is a
        barcode followed only by numbers is read as a spot: six columns get Space Ranger's names,
        three are read as barcode, x, y, and any other width is refused). The ST and Visium
        neighbour definitions read array_row/array_col when the file has them. When the file has an
        in_tissue column, spots with in_tissue == 0 (background outside the tissue) are left out and
        reported (params.in_tissue_filter and a warning); the converter's metadata.csv carries
        in_tissue and array_row/array_col, its coordinates.csv only x/y. Spots in the counts with no
        coordinates row are left out and counted (data.n_spots_without_coordinates).
    output_dir:
        Directory for PRECAST output files (cluster CSV, RDS object).
    K:
        Number of spatial domains (clusters) to identify.
    platform:
        How each spot's neighbours are found. 'auto' (default) reads the coordinates: when
        array_row/array_col (Space Ranger's positions files, or the converter's metadata.csv) -- or
        an integer x/y pair -- form a square grid it uses 'ST', a hexagonal grid 'Visium', and
        otherwise (e.g. the converter's pixel x/y coordinates.csv) 'Other_SRT'. 'ST' and 'Visium'
        match neighbours at exact array-index offsets (ST: +-1 on one axis; Visium: col +-2, or col
        +-1 and row +-1), so on pixel positions or the wrong grid they find none -- the run then
        stops with an error rather than clustering without space. 'Other_SRT' is PRECAST's radius
        search on the coordinates (below roughly 1,000 spots it often cannot reach its target of 4
        neighbours; see allow_fixed_number_fallback). params.platform is the definition that ran,
        params.coord_columns the two columns read per sample, and data.median_neighbors /
        data.n_spots_with_neighbors describe the graph.
    gene_number:
        Number of spatially variable genes SPARK-X selects per sample (CreatePRECASTObject keeps
        the ones shared across samples, then filters genes seen in fewer than 15 spots). PRECAST
        lowers it, with a warning in the payload, when a sample has fewer; params.gene_number is
        the request, params.gene_number_used / data.n_genes_used the genes the fit used and
        data.n_genes_read the genes in the counts.
    core_num:
        Number of CPU cores for parallel computation.
    max_iter:
        Maximum number of EM iterations.
    sigma_equal:
        If True, assume equal covariance matrices across clusters.
    seed:
        Random seed for reproducibility; handed to PRECAST's own initialisation (AddParSetting
        seed=), which re-seeds itself before fitting.
    allow_fixed_number_fallback:
        If True and the 'Other_SRT' radius search fails ("radius.upper is too smaller"), build the
        graph from each spot's 6 nearest neighbours instead (PRECAST's type='fixed_number'); the
        payload then says params.adjacency_type='fixed_number', params.used_fallback=true and
        carries a warning. False (default): that failure is an error naming this switch.
    """
    # PRECAST spawns `core_num` parallel R workers; if EACH also lets its BLAS use every core that is
    # core_num x n_cores threads, and the cores thrash instead of computing (commit d1689ee measured
    # load average ~328 and a stalled RCTD on a 96-core box). Pin BLAS to one thread per worker so
    # PRECAST's own parallelism is the only parallelism. Must precede the Rscript launch: OpenBLAS
    # reads the count at R startup, and the subprocess inherits os.environ.
    pin_blas_threads()

    os.makedirs(output_dir, exist_ok=True)

    if len(counts_csvs) != len(coords_csvs):
        return {
            "status": "error",
            "error": "counts_csvs and coords_csvs must have the same length",
        }

    if len(counts_csvs) < 1:
        return {
            "status": "error",
            "error": "At least one sample is required",
        }

    args = [
        "--output-dir",
        output_dir,
        "--K",
        str(K),
        "--platform",
        platform,
        "--gene-number",
        str(gene_number),
        "--core-num",
        str(core_num),
        "--max-iter",
        str(max_iter),
        "--seed",
        str(seed),
    ]

    for csv in counts_csvs:
        args.extend(["--counts-csv", csv])
    for csv in coords_csvs:
        args.extend(["--coords-csv", csv])

    if sigma_equal:
        args.append("--sigma-equal")
    if allow_fixed_number_fallback:
        args.append("--allow-fixed-number-fallback")

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
