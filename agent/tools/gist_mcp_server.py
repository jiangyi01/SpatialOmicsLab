#!/usr/bin/env python3
"""GIST Bayesian spatial deconvolution MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli
from worker_utils import pin_blas_threads

TOOL_NAME = "gist"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "GIST",
    "/opt/conda/envs/gist_env/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/gist_worker.R",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_gist(
    spatial_counts_csv: str,
    spatial_coords_csv: str,
    ref_counts_csv: str,
    ref_celltypes_csv: str,
    output_dir: str,
    impute_st: bool = True,
    impute_k: int = 5,
    impute_d: int = 10,
    normalize: str = "sct",
    prior_lambda: float = 50.0,
    num_cores: int = 1,
    seed: int = 42,
    n_iter: int = 2000,
    n_chains: int = 4,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Run GIST Bayesian spatial deconvolution on spatial transcriptomics data.

    GIST fits, for every spot separately, a Stan model of the spot's expression as a Student-t regression
    on a scRNA-seq signature matrix (mean normalised expression per reference cell type), with the
    cell-type proportions as a simplex, and reports their posterior mean from rstan NUTS sampling. GIST's
    image-guided (enhanced) model adds a beta prior on one cell type from a per-spot image-derived value;
    this tool takes no such prior, so GIST's base model always runs (``params.method``) and the image is
    not used. Spatial coordinates are only aligned to the spots and stored in ``gist_result.rds``; the
    model does not use them.

    Runtime is one Stan fit per spot: ``n_chains`` x ``n_iter`` iterations each, so it grows with the
    spot count, the genes used and the number of reference cell types. ``num_cores`` spreads the spots
    over parallel R workers.

    Parameters
    ----------
    spatial_counts_csv:
        Path to spatial gene expression counts CSV (genes x spots).
    spatial_coords_csv:
        Path to spatial coordinates CSV (spots as rows, spot names in the first column). Axis columns are
        matched by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col,
        row/col or x/y -- so Space Ranger's tissue_positions.csv can be passed as it is, and so can the
        headerless tissue_positions_list.csv of Space Ranger 1 (a first line that is a spot is read as
        one). When the file has an in_tissue column, counts spots it marks 0 (background) are left out
        and counted in ``data.n_spots_off_tissue_dropped`` and ``params.in_tissue_filter``.
    ref_counts_csv:
        Path to scRNA-seq reference counts CSV (genes x cells).
    ref_celltypes_csv:
        Path to reference cell type annotation CSV (cell barcode as index,
        first column is cell type label). A cell with a missing label (NA, empty, "nan", "<NA>") is an
        error unless ``drop_unlabeled`` is True.
    output_dir:
        Directory for GIST output files (proportions CSV, RDS object).
    impute_st:
        Whether to apply KNN smoothing imputation to spatial counts.
    impute_k:
        Number of nearest neighbors for KNN smoothing imputation. Used only when ``impute_st`` is True;
        otherwise listed under ``params.ignored`` with a warning.
    impute_d:
        Number of principal components for KNN smoothing. Used only when ``impute_st`` is True;
        otherwise listed under ``params.ignored`` with a warning.
    normalize:
        Normalization method: 'sct' (SCTransform), 'scale', 'quantile',
        or 'scale-quantile'. 'sct' keeps only SCTransform's variable genes, chosen separately in the
        reference and in the spatial data, and GIST uses the genes both keep; the other methods keep
        every gene. The counts are in ``data.n_genes_ref_after_normalize``,
        ``data.n_genes_spatial_after_normalize`` and ``data.n_genes_used``.
    prior_lambda:
        Accepted but not used. GIST reads it only in its image-guided model, which needs a per-spot
        prior this tool does not take; the base model runs, and the payload lists prior_lambda under
        ``params.ignored`` with a warning.
    num_cores:
        Number of cores for parallel processing of spots.
    seed:
        Random seed: seeds R, the KNN-smoothing PCA and every per-spot Stan run, so a run is
        reproducible for any ``num_cores``.
    n_iter:
        Stan iterations per chain for each spot, half of them warmup (rstan's default, 2000). Lowering
        it shortens the run and makes each posterior mean noisier.
    n_chains:
        Stan chains per spot (rstan's default, 4).
    drop_unlabeled:
        Reference cells whose label is missing (NA / empty / "nan" / "<NA>") are refused by default; True
        leaves them out and reports the count in ``data.n_ref_cells_dropped_unlabeled``.
    """
    # GIST spawns `num_cores` parallel R workers; if EACH also lets its BLAS use every core that is
    # num_cores x n_cores threads, and the cores thrash instead of computing (commit d1689ee measured
    # load average ~328 and a stalled RCTD on a 96-core box). Pin BLAS to one thread per worker so
    # GIST's own parallelism is the only parallelism. Must precede the Rscript launch: OpenBLAS
    # reads the count at R startup, and the subprocess inherits os.environ.
    pin_blas_threads()

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
        "--impute-st",
        str(impute_st).upper(),
        "--impute-k",
        str(impute_k),
        "--impute-d",
        str(impute_d),
        "--normalize",
        normalize,
        "--prior-lambda",
        str(prior_lambda),
        "--num-cores",
        str(num_cores),
        "--seed",
        str(seed),
        "--n-iter",
        str(n_iter),
        "--n-chains",
        str(n_chains),
        "--drop-unlabeled",
        # str(), not str(bool()): bool("false") is True. The worker reads TRUE/T/YES/1 as true, as for impute_st.
        str(drop_unlabeled).upper(),
    ]

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
