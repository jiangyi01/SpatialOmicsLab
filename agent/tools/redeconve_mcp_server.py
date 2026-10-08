#!/usr/bin/env python3
"""Redeconve spatial deconvolution MCP wrapper for SpatialOmicsLab."""

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "redeconve"
DEFAULT_RSCRIPT = "/opt/conda/envs/redeconve/bin/Rscript"
DEFAULT_WORKER = "/workspace/epic-fermat/agent/tools/redeconve_worker.R"

RSCRIPT, WORKER_R = get_worker_paths("REDECONVE", DEFAULT_RSCRIPT, DEFAULT_WORKER)

# Redeconve requires single-threaded BLAS to avoid deadlocks. Applied per run in the tool function
# below, not here at import: an import-time write pins every other tool sharing this process, and a
# thread variable set behind another worker's back is not harmless -- prost_worker.py pins its own
# threads with os.environ.setdefault, which silently becomes a no-op if someone else got there first,
# and PROST's single thread is what makes its domain labels reproducible, not what makes them fast.
_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def redeconve_deconvolution(
    spatial_counts_csv: str,
    ref_counts_csv: str,
    ref_celltypes_csv: str,
    output_dir: str,
    n_top_genes: int = 0,
    seed: int = 0,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Run Redeconve spatial deconvolution using quadratic programming.

    Redeconve estimates, for each spot, a non-negative abundance of every reference cell by
    quadratic programming against the scRNA-seq reference (each cell is a regressor), and the
    worker sums those abundances per cell type into proportions. The per-spot program has one
    variable per reference cell and a dense cells x cells Hessian, so memory grows with the
    square of the number of reference cells and each spot's solve faster still; every cell and
    every spot supplied is used. A spot whose abundances all come back zero keeps an all-zero row
    in the proportions CSV and is counted in data.n_spots_without_estimate.

    Parameters
    ----------
    spatial_counts_csv:
        Path to spatial gene expression counts CSV (genes x spots).
    ref_counts_csv:
        Path to scRNA-seq reference counts CSV (genes x cells).
    ref_celltypes_csv:
        Path to reference cell type annotation CSV (cell barcode as index,
        first column is cell type label).
    output_dir:
        Directory for Redeconve output files (proportions CSV).
    n_top_genes:
        0 (default) uses every gene the reference and the spatial data share, as Redeconve's
        own genemode="default" does. A positive value below the shared count keeps only that
        many shared genes, those whose log1p(CPM) varies most across the reference cells, and
        passes them to Redeconve as a customized gene list. The payload reports the genes used
        (data.n_genes_used) and the rule (params.gene_selection).
    seed:
        Set before the run, but Redeconve's quadratic programming draws no random numbers, so
        it has no effect on the result; the payload lists it under params.ignored.
    drop_unlabeled:
        Reference cells whose label is missing (NA, empty, 'nan') are refused by default,
        because a missing label is not a cell type. True leaves them out and reports how many.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Single-threaded BLAS for the R subprocess (see _THREAD_VARS above). The whole family, because
    # R links against OpenBLAS *or* MKL *or* a reference BLAS driven by OpenMP and each reads its own
    # variable -- pinning only OPENBLAS_NUM_THREADS left the deadlock reachable on the other two.
    # setdefault, because an operator who set a thread count deliberately should keep it. The R
    # subprocess inherits os.environ and reads these at startup, so this has to happen before launch.
    for _thr_var in _THREAD_VARS:
        os.environ.setdefault(_thr_var, "1")

    args = [
        "--spatial-counts-csv",
        spatial_counts_csv,
        "--ref-counts-csv",
        ref_counts_csv,
        "--ref-celltypes-csv",
        ref_celltypes_csv,
        "--output-dir",
        output_dir,
        "--n-top-genes",
        str(n_top_genes),
        "--seed",
        str(seed),
    ]
    if drop_unlabeled:
        args += ["--drop-unlabeled", "true"]

    return run_worker_cli(TOOL_NAME, RSCRIPT, WORKER_R, args)


if __name__ == "__main__":
    mcp.run()
