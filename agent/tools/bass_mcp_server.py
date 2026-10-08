#!/usr/bin/env python3
"""BASS Bayesian multi-scale spatial domain analysis MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "bass"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "BASS",
    "/opt/conda/envs/bass_env/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/bass_worker.R",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_bass(
    spatial_counts_csv: str,
    spatial_coords_csv: str,
    output_dir: str,
    n_clusters: int = 7,
    n_cell_types: int = 5,
    burn_in: int = 2000,
    n_samples: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Run BASS Bayesian multi-scale spatial domain analysis on spatial transcriptomics data.

    BASS (Bayesian Analytics for Spatial Segmentation) performs multi-scale
    analysis that jointly models cell type composition and spatial domain
    structure via a Bayesian hierarchical framework with Markov random fields.
    The worker makes the four upstream calls (``createBASSObject`` ->
    ``BASS.preprocess`` -> ``BASS.run`` -> ``BASS.postprocess``); there is no
    substitute method, so ``params.method`` names them and ``params.used_fallback``
    is always False.

    What reaches the model:

    - Spots: only spots present in both files are modelled, and a spot the coordinates file
      marks in_tissue = 0 is left out as background (``data.n_spots_off_tissue_dropped``,
      ``params.in_tissue_filter``). A spot with zero total
      counts (typically an off-tissue spot in a CELLxGENE export) is left out, because
      BASS normalises each spot by its total; ``data.n_spots_empty``, a warning and the
      analysis text say how many. ``data.n_spots`` is the number of spots modelled and
      written to ``bass_domains.csv``; ``data.n_spots_input`` is the number in the counts.
    - Genes: BASS log-normalises the counts, keeps the 3000 most significant SPARK-X
      spatially expressed genes when the file has more than 3000 genes (all genes
      otherwise), drops genes with no count, and reduces them to 20 principal
      components. ``data.n_genes_used``, ``params.gene_selection`` and ``params.n_pcs``
      report it. The input needs at least 20 genes with counts and 20 spots.
    - Values: raw counts are expected. Missing or negative values are refused;
      non-integer (already normalised) values run but are counted in
      ``data.n_values_non_integer`` and warned about, since BASS normalises again.
    - Memory: the CSV is read into a dense genes x spots matrix and ``BASS.preprocess``
      normalises and scales it densely; that is intrinsic to BASS.

    Outputs: ``bass_domains.csv`` (columns ``spot_id, domain, cell_type, x, y``) and
    ``bass_result.rds`` (the fitted BASS object). ``data.n_clusters`` and
    ``data.n_cell_types`` are the numbers of occupied domain and cell-type labels,
    which can be fewer than requested; the requested values are in ``params``.

    Parameters
    ----------
    spatial_counts_csv:
        Path to spatial gene expression counts CSV (genes x spots; spots x genes is
        also accepted and transposed). Raw counts, gene names in the first column.
    spatial_coords_csv:
        Path to spatial coordinates CSV (spots as rows, spot names in the first column). Axis columns are
        matched by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col,
        row/col or x/y -- so Space Ranger's tissue_positions.csv can be passed as it is, and so can its
        headerless tissue_positions_list.csv (Space Ranger before 2.0), which is read under Space Ranger's
        six column names (params.coordinates_header says which, params.coordinate_columns which two were
        read). Spots the file marks in_tissue = 0 (background outside the tissue) are left out and counted
        in data.n_spots_off_tissue_dropped, params.in_tissue_filter and a warning.
    output_dir:
        Directory for BASS output files (``bass_domains.csv``, ``bass_result.rds``).
    n_clusters:
        Number of spatial domains to identify: BASS's ``R``. The ``domain`` column has at
        most this many labels.
    n_cell_types:
        Number of cell-type clusters in the composition model: BASS's ``C``. The
        ``cell_type`` column has at most this many labels.
    burn_in:
        Number of burn-in iterations for MCMC sampling (BASS's ``burnin``; 2000 is BASS's own default).
    n_samples:
        Number of posterior samples after burn-in (BASS's ``nsample``; 5000 is BASS's own default).
    seed:
        Random seed for reproducibility.
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
        "--n-cell-types",
        str(n_cell_types),
        "--burn-in",
        str(burn_in),
        "--n-samples",
        str(n_samples),
        "--seed",
        str(seed),
    ]

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
