#!/usr/bin/env python3
"""SPARK spatially variable gene detection MCP wrapper for SpatialOmicsLab."""

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "spark"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "SPARK",
    "/opt/conda/envs/spark_spatial/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/spark_worker.R",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def spark_svg_detection(
    counts_csv: str,
    coords_csv: str,
    output_dir: str,
    n_top: int = 100,
    pval_cutoff: float = 0.05,
    percentage: float = 0.1,
    min_total_counts: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Detect spatially variable genes (SVGs) using SPARK.

    SPARK uses generalized linear spatial models with a set of spatial kernels
    to identify genes whose expression varies across spatial locations. It is the only
    method this tool runs: params.method names it and params.used_fallback is always false.

    Parameters
    ----------
    counts_csv:
        Path to gene expression counts CSV (genes x spots, with row/col headers).
        Rows are genes, columns are spot/cell barcodes. A spots x genes table is recognised
        when none of its column names matches a coordinate row; it is then transposed, and
        params.counts_orientation plus a warning say so. The values must be raw integer
        counts: SPARK fits a Poisson count model, so a missing, negative or non-integer value
        (normalised or scaled data) stops the run with the count and an example; nothing is
        rounded. convert_h5ad_to_csv writes adata.X, so export from an h5ad whose X holds the
        raw counts.
    coords_csv:
        Path to spatial coordinates CSV (spots as rows, spot names in the first column). Axis columns are
        matched by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col,
        row/col or x/y -- so Space Ranger's tissue_positions.csv can be passed as it is. The headerless
        tissue_positions_list.csv that Space Ranger 1 writes is recognised from its first line (every
        field after the barcode is a number, and not pandas' default column labels 0, 1, ...) and read
        with SpotClean's column names, so no spot is spent on the header and the pixel coordinates are
        the ones used. When the file carries Space Ranger's in_tissue flag, matched spots flagged 0
        (background glass, e.g. beside a raw matrix of every array spot) are left out and reported in
        params.in_tissue_filter, data.n_spots_off_tissue_dropped, a warning and the analysis.
    output_dir:
        Directory for SPARK output files (results CSV, significant SVGs).
    n_top:
        Number of top SVGs to report in the summary file.
    pval_cutoff:
        Cutoff applied to SPARK's adjusted_pvalue -- the Benjamini-Yekutieli correction of the
        Cauchy-combined kernel p-value -- for identifying significant SVGs. params.pvalue_column names
        the column the cutoff was applied to.
    percentage:
        Minimum fraction of spots (0-1; the default 0.1 is 10%) in which a gene must be non-zero to be
        tested. SPARK drops genes below it before the fit; data.n_genes_used and the analysis report
        how many were tested.
    min_total_counts:
        Minimum total count per SPOT, not per gene: after the percentage gene filter, SPARK drops every
        spot whose total count over the retained genes is at or below this value. Dropped spots are
        reported in data.n_spots_used, in a warning and in the analysis.
    seed:
        Accepted and set for interface compatibility, but SPARK's variance-component fit and kernel
        tests use no random-number generator, so it has no effect on the result; it is listed under
        params.ignored.
    """
    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--counts-csv",
        counts_csv,
        "--coords-csv",
        coords_csv,
        "--output-dir",
        output_dir,
        "--n-top",
        str(n_top),
        "--pval-cutoff",
        str(pval_cutoff),
        "--percentage",
        str(percentage),
        "--min-total-counts",
        str(min_total_counts),
        "--seed",
        str(seed),
    ]

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
