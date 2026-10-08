#!/usr/bin/env python3
"""spVC spatially varying coefficients MCP wrapper for SpatialOmicsLab."""

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "spvc"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "SPVC",
    "/opt/conda/envs/spvc/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/spvc_worker.R",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def spvc_svg_detection(
    counts_csv: str,
    coords_csv: str,
    output_dir: str,
    n_top: int = 100,
    pval_cutoff: float = 0.05,
    max_genes: int = 1000,
    max_spots: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Detect spatially variable genes using spVC (spatially varying coefficients).

    spVC fits each gene with a quasi-Poisson generalized additive model (mgcv) whose
    coefficients vary over space, and tests whether expression varies across spatial
    locations. This wrapper runs ``test.spVC``'s reduced model (a spatially varying
    intercept, no covariates) on a bounding-rectangle mesh of two triangles, and analyses
    every spot (every in-tissue spot when coords_csv carries an ``in_tissue`` flag) unless
    ``max_spots`` is set.

    test.spVC applies two filters of its own, and the payload counts both: spots whose
    counts over the submitted genes sum to fewer than 5 are dropped (``data.n_spots_fitted``
    of ``data.n_spots``), and genes non-zero in 5 or fewer of the remaining spots are skipped
    (``data.n_genes_fitted`` of ``data.n_genes_submitted``); skipped genes are absent from the
    output tables. spVC clamps p-values at 2e-17, so strong genes tie there;
    ``data.n_genes_at_pvalue_floor`` says how many, and ties are ranked by count variance,
    highest first. Warnings mgcv raises while fitting are counted in ``data.n_fit_warnings``
    and summarised in ``warnings``. In ``spvc_results.csv`` the ``statistic`` column is the
    fitted model's deviance, not a test statistic.

    Parameters
    ----------
    counts_csv:
        Path to gene expression counts CSV (genes x spots, with row/col headers).
        Rows are genes, columns are spot/cell barcodes.
    coords_csv:
        Path to spatial coordinates CSV (spots as rows, spot names in the first column). Axis columns are
        matched by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col,
        row/col or x/y -- so Space Ranger's tissue_positions.csv can be passed as it is, and so can the
        headerless tissue_positions_list.csv of Space Ranger before 2.0 (read under Space Ranger's own
        column names; ``params.coordinates_header`` says so). A spot in counts_csv with no row here is left
        out and counted in ``data.n_spots_without_coords``. When the file has an ``in_tissue`` column,
        spots marked 0 are background outside the tissue and are left out before anything else
        (``data.n_spots_off_tissue_dropped`` of ``data.n_spots_supplied``, ``params.in_tissue_filter``, a
        warning and the analysis).
    output_dir:
        Directory for spVC output files (incl. predicted_genes.json).
    n_top:
        Number of top spatially variable genes (sorted ascending by p-value, ties
        by count variance) to emit to predicted_genes.json. The manual SpVC runner
        uses top 10% of the HVG-1000 pool, i.e. n_top=100. Must be a whole number
        of 0 or more; anything else is refused.
    pval_cutoff:
        Benjamini-Hochberg adjusted p-value cutoff used only for
        spvc_significant_svgs.csv. Not applied to predicted_genes.json (which uses
        top-N ranking, matching the manual benchmark runner). A value that is not
        a finite number is refused.
    max_genes:
        HVG pre-filter cap before running spVC. Manual runner uses 1000
        (variance-based top-1000) — see /workspace/hands_by_myself/runners/
        spvc_svg_detection.R lines 92-99. Lower values (e.g. 200) speed up
        BPST mesh construction but yield far fewer candidate SVGs. Genes cut by
        this cap are absent from the output tables, not ranked last in them.
        Must be a whole number of at least 1; anything else is refused.
    max_spots:
        Optional cap on the number of spots analysed. Omit it (the default) and
        every spot is analysed. When set, a slide with more spots than this has a
        random subset of this size drawn (seeded by ``seed``), the rest of the
        slide is not tested, and the payload's ``warnings`` say how much was left
        out. Must be a whole number of at least 1. spVC fits a bivariate spline
        over the spot set, so runtime grows with the number of spots.
    seed:
        Random seed for reproducibility. It seeds the random spot draw when ``max_spots`` is set;
        spVC's own fit (a quasi-Poisson GAM per gene) draws no random numbers.
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
        "--max-genes",
        str(max_genes),
        "--seed",
        str(seed),
    ]
    # Forward nothing when unset: the worker stays the single source of the default (no cap,
    # every spot analysed), and an unchanged call builds the same argv it always did.
    if max_spots is not None:
        args += ["--max-spots", str(max_spots)]

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
