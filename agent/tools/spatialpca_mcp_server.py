#!/usr/bin/env python3
"""SpatialPCA spatially-aware dimensionality reduction MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "spatialpca"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "SPATIALPCA",
    "/opt/conda/envs/spatialpca_env/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/spatialpca_worker.R",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_spatialpca(
    spatial_counts_csv: str,
    spatial_coords_csv: str,
    output_dir: str,
    n_components: int = 20,
    seed: int = 42,
    allow_hvg_fallback: bool = False,
) -> dict[str, Any]:
    """
    Run SpatialPCA spatially-aware dimensionality reduction.

    SpatialPCA builds a kernel-based latent variable model that incorporates
    spatial location information into PCA, producing spatially informed
    low-dimensional embeddings suitable for downstream clustering and
    visualization. As published, its genes are the spatially variable genes
    SPARK selects (at most 3000). If SPARK's selection is unusable -- SPARK
    fails, or selects fewer genes than n_components -- the run stops and says
    what SPARK found, unless allow_hvg_fallback=True. params.method,
    params.used_fallback and data.gene_selection say which selection ran.

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
        Directory for SpatialPCA output files (spatial PCs and loadings CSVs).
    n_components:
        Number of spatial principal components to compute. At most (selected
        genes - 1) and (analysed spots - 1) can be; a larger request is lowered
        with a warning. params.n_components is the number computed and
        params.n_components_requested the number asked for.
    seed:
        Accepted and set, but it has no effect on the result: SpatialPCA reseeds
        its random steps itself (SCTransform with Seurat's fixed seed.use, and
        SpatialPCA_EstimateLoading with set.seed(1234)), and SPARK and the kernel
        steps draw no random numbers. Listed under params.ignored with a warning.
    allow_hvg_fallback:
        When SPARK's spatial gene selection is unusable, run SpatialPCA over
        highly variable genes (SCTransform, at most 3000) instead of stopping.
        That is a different analysis from SpatialPCA as published; a run that
        used it reports params.used_fallback=True, data.gene_selection='hvg',
        the reason, and a warning. Default False: the run stops with SPARK's
        finding.
    """
    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--spatial-counts-csv",
        spatial_counts_csv,
        "--spatial-coords-csv",
        spatial_coords_csv,
        "--output-dir",
        output_dir,
        "--n-components",
        str(n_components),
        "--seed",
        str(seed),
    ]
    if allow_hvg_fallback:
        args += ["--allow-hvg-fallback", "true"]

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
