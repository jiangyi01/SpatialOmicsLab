#!/usr/bin/env python3
"""Celloscope probabilistic deconvolution MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "celloscope"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "CELLOSCOPE",
    "/opt/conda/envs/celloscope_env/bin/python3.8",
    "/workspace/epic-fermat/agent/tools/celloscope_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_celloscope(
    spatial_counts_csv: str,
    sc_counts_csv: str,
    cell_type_labels_csv: str,
    output_dir: str,
    n_iterations: int = 1000,
    n_markers_per_type: int = 20,
    n_cells_per_spot: int = 5,
    n_cells_csv: str = "",
) -> dict[str, Any]:
    """
    Run Celloscope deconvolution of spatial transcriptomics spots into cell-type proportions.

    Celloscope is a Bayesian model of marker-gene counts, fitted by MCMC (Metropolis-within-Gibbs).
    The sampler is Celloscope's own (upstream ``code/impl.py``, one chain). Its inputs are prepared
    here rather than by Celloscope's manual procedure:

    * the binary marker matrix B is derived from the single-cell reference: for each cell type, the
      genes whose library-size-normalised mean is highest in that type, ranked by fold change over the
      next-highest type, at most ``n_markers_per_type`` per type, among genes counted at least once on
      the slide. Only those genes reach the model. B ends with Celloscope's all-zero "dummy type";
    * each spot's cell number is taken from ``n_cells_csv`` or, without it, set to the constant
      ``n_cells_per_spot``; either way it is the centre of a Normal(n, 2) prior (Celloscope's
      ``ASPRIORS`` mode), not a fixed value.

    Output: ``celloscope_proportions.csv`` (spots x reference cell types, rows sum to 1) from
    Celloscope's point estimate ``celloscope_results/chain01/thetas_est.csv``, with the dummy type's
    share removed and each spot renormalised; ``celloscope_h_with_dummy_type.csv`` keeps the shares
    including the dummy type. ``chain01/result_h.csv`` is the MCMC trace, not a result table.

    Memory: neither counts table is held whole. Each is read a block of rows at a time (about 64 MB
    of values per block), twice: the spatial table for per-gene totals and then only the marker
    columns, the reference for cell IDs and library sizes and then per-type sums over the candidate
    genes. What stays in memory is the marker genes x spots matrix Celloscope models (8 bytes a value,
    e.g. ~1.6 GB for 400 markers on a 507,684-bin VisiumHD slide) plus the IDs and sums.

    Parameters
    ----------
    spatial_counts_csv:
        Path to spatial counts CSV (spots x genes): spot IDs in the first column, raw integer counts.
        Celloscope's negative-binomial likelihood refuses fractional, negative or missing values.
    sc_counts_csv:
        Path to single-cell counts CSV (cells x genes), cell IDs in the first column.
    cell_type_labels_csv:
        Path to CSV with cell type labels for single-cell data: cell IDs in the first column (matched to
        sc_counts_csv by ID), the label in the first data column. Cells with no label are left out of
        marker selection and counted in the payload. At least two labelled cell types are required.
    output_dir:
        Directory to write Celloscope outputs.
    n_iterations:
        Number of MCMC iterations (at least 6). The first max(5, n_iterations // 2) are burn-in; the
        estimate is the mean over every max(1, n_iterations // 10)-th iteration after that. Celloscope's
        own example runs 15000. With too few the estimate stays near the sampler's random start: the
        payload reports that as ``data.mcmc_start_correlation`` and warns above 0.5 (library Visium
        measured 0.81-0.91 at 200-700 iterations; a run that recovered known proportions, 0.18).
    n_markers_per_type:
        At most this many marker genes per cell type in B.
    n_cells_per_spot:
        Cell-number estimate used for every spot when ``n_cells_csv`` is not given (1 suits
        single-cell-resolution data). Not used when ``n_cells_csv`` is given; a value other than 5 is then
        listed in ``params.ignored`` with a warning.
    n_cells_csv:
        Optional per-spot cell-number estimates (e.g. from nuclei segmentation): spot IDs in the first
        column, then one count column (or a ``cellCount`` column; Celloscope's own ``n_cells.csv``
        layout with a ``spotId`` column is also read). Must cover every spot with a whole number.
    """
    args = [
        "--spatial-counts",
        spatial_counts_csv,
        "--sc-counts",
        sc_counts_csv,
        "--cell-type-labels",
        cell_type_labels_csv,
        "--output-dir",
        output_dir,
        "--n-iterations",
        str(n_iterations),
        "--n-markers-per-type",
        str(n_markers_per_type),
        "--n-cells-per-spot",
        str(n_cells_per_spot),
    ]
    if n_cells_csv:
        args += ["--n-cells-csv", n_cells_csv]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
