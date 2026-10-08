#!/usr/bin/env python3
"""Cell2location MCP wrapper for SpatialOmicsLab.

Thin portal: builds the command line for ``tools/cell2location_worker.py`` and runs it in the
cell2location conda env (``CELL2LOCATION_PYTHON`` / ``CELL2LOCATION_WORKER`` override the paths).
Every parameter the portal accepts is forwarded; nothing is silently defaulted on the way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "cell2location"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "CELL2LOCATION",
    "/opt/conda/envs/cell2loc_env/bin/python",
    "/workspace/epic-fermat/agent/tools/cell2location_worker.py",
)

# The CLI cannot carry None; this spelling is what the worker maps back to "no batch covariate".
NO_BATCH = "none"
# labels_key=None used to drop the flag and let the worker's own default apply; the same default is
# now sent explicitly, so what ran is on the command line and in params.labels_key.
DEFAULT_LABELS_KEY = "CellType"

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_cell2location(
    sc_h5ad_path: str,
    spatial_h5ad_path: str | None = None,
    results_dir: str = default_output_dir("cell2location_output"),
    labels_key: str | None = "CellType",
    batch_key: str | None = "Sample",
    n_cells_per_location: int | None = 30,
    detection_alpha: float | None = 20.0,
    max_epochs_ref: int | None = 250,
    max_epochs_map: int | None = 30000,
    cell_count_cutoff: int | None = 5,
    cell_percentage_cutoff2: float | None = 0.03,
    nonz_mean_cutoff: float | None = 1.12,
    n_neighbors: int | None = 15,
    leiden_resolution: float | None = 1.1,
    round_counts: bool = False,
    drop_unlabeled: bool = False,
    batch_size_map: int | None = None,
) -> dict[str, Any]:
    """
    MCP tool: run the cell2location pipeline via subprocess in cell2loc_env.

    What runs: cell2location's ``RegressionModel`` (negative-binomial regression) on the scRNA
    reference to estimate per-cell-type signatures, then ``Cell2location`` to map them onto the
    spatial AnnData (per-location cell abundance, ``obsm['q05_cell_abundance_w_sf']``). A
    downstream scanpy KNN + Leiden on that abundance gives coarse spatial domains
    (``obs['region_cluster']``); that step is an add-on, not part of cell2location.

    Outputs under ``results_dir``: ``reference_signatures/inf_aver.csv`` (gene x cell-type
    signatures), ``cell2location_map/sp.h5ad`` (spatial AnnData with the abundance in obsm, one
    obs column per cell type and ``region_cluster``) and QC/diagnostic PNGs. Every one of them --
    the CSV, the h5ad and each PNG -- is written beside its final path and renamed into place, so
    a crashed run never leaves a truncated file under a final name.

    Memory: with ``batch_size_map`` unset the spatial model trains full-batch, and upstream then
    holds the whole spots x shared-genes matrix as one dense tensor, with every training step's
    intermediates the same size (measured ~70 bytes per spot-gene pair); the posterior export then
    keeps 1000 samples of every spot's local variables. Before the reference model is trained the
    worker estimates that peak against the memory it can still allocate and, if it does not fit,
    refuses with the numbers and the ``batch_size_map`` that would fit (when one does) -- instead
    of being OOM-killed hours later (a Visium HD or Xenium slide does not fit full-batch). The data
    is never subsampled.
    ``params.memory_estimate`` carries the estimate.

    Background spots: when the spatial file's ``obs['in_tissue']`` marks spots as background (0)
    -- CELLxGENE Visium exports carry every array spot -- only in-tissue spots are mapped;
    ``data.n_spots`` counts them, ``data.n_spots_input`` counts the file, and
    ``params.in_tissue_filter`` plus a warning say how many were left out.

    Input contract (checked by the worker before anything is trained):

    * ``X`` of both files must hold raw integer counts (cell2location fits Gamma-Poisson
      likelihoods). Every stored value is checked. A non-integer matrix -- normalised or
      log-transformed data -- is refused with the count of offending values unless
      ``round_counts=True``, which rounds every value to the nearest integer (``np.rint``) and
      records the transform in ``params.rounded_inputs`` and a warning. Nothing is rounded silently.
    * ``labels_key`` must name an obs column of the reference; the error lists the columns
      otherwise (``None`` means the default ``'CellType'``). Cells whose label is missing (NaN/empty) are refused unless
      ``drop_unlabeled=True``, in which case they are left out and
      ``params.n_reference_cells_dropped_unlabeled`` says how many. Fewer than two cell types is
      refused. Unused categories of a categorical label column are not fitted as cell types.
    * ``batch_key`` names the reference obs column with the sample/batch id used as the
      RegressionModel's batch covariate. Pass ``None`` (or ``'none'``) to fit WITHOUT a batch
      covariate -- the worker receives ``'none'``; before, ``None`` dropped the flag and the
      worker fell back to ``'Sample'``, so there was no way to ask for no batch. A named column
      must exist and be fully populated.
    * ``X`` must not contain negative or NaN/inf values (``round_counts`` cannot repair those).

    Parameters
    ----------
    sc_h5ad_path:
        Path to single-cell reference AnnData (.h5ad). REQUIRED.
    spatial_h5ad_path:
        Path to spatial transcriptomics AnnData (.h5ad). Required at runtime
        -- if omitted, the wrapper returns a clean error rather than letting
        Pydantic raise a validation error in the MCP framework.
    results_dir:
        Base output directory (``reference_signatures/`` and ``cell2location_map/`` are created).
    labels_key:
        Reference obs column with cell-type labels (default ``'CellType'``; ``None`` means the
        default). cell2location learns one signature per label, so there is no "no labels" mode.
    batch_key:
        Reference obs column with the batch/sample id (default ``'Sample'``); ``None`` = no batch.
    n_cells_per_location, detection_alpha:
        cell2location's ``N_cells_per_location`` and ``detection_alpha`` hyper-priors.
    max_epochs_ref, max_epochs_map:
        Training epochs for the RegressionModel and the spatial model. The defaults (250 /
        30000) are the tutorial's paper-grade values and the worker's own defaults, so passing
        ``None`` for either means the same thing as omitting it. 30000 mapping epochs on CPU is
        slow (hours on a full Visium slide); lower them for a quick look and note it -- the data
        itself is never subsampled.
    cell_count_cutoff, cell_percentage_cutoff2, nonz_mean_cutoff:
        cell2location ``filter_genes`` thresholds applied to the reference; the payload reports
        how many genes passed and how many are shared with the spatial data.
    n_neighbors, leiden_resolution:
        The downstream KNN graph and Leiden resolution for ``region_cluster``.
    round_counts:
        Default False. True rounds a non-integer ``X`` to the nearest integer instead of refusing it.
    drop_unlabeled:
        Default False. True leaves out reference cells with a missing label instead of refusing.
    batch_size_map:
        Spots per minibatch for the spatial model's training and its posterior export. Default
        ``None`` = full batch (what upstream and this tool always did). A number trains in
        minibatches of that many spots -- one epoch is then n_spots / batch_size_map steps, so
        ``max_epochs_map`` means more optimisation steps than full batch -- and the export
        summarises each batch as it is sampled, so memory follows the batch, not the slide.
        Recorded in ``params.batch_size_map`` and ``params.posterior_sample_kwargs``.
    """
    if spatial_h5ad_path is None:
        return {
            "status": "error",
            "tool": TOOL_NAME,
            "error": "spatial_h5ad_path is required: provide the path to the spatial transcriptomics AnnData (.h5ad).",
        }
    if labels_key is None:
        labels_key = DEFAULT_LABELS_KEY

    sc_path = str(Path(sc_h5ad_path).expanduser())
    spatial_path = str(Path(spatial_h5ad_path).expanduser())
    out_dir = str(Path(results_dir).expanduser())

    args = [
        "--sc-h5ad",
        sc_path,
        "--spatial-h5ad",
        spatial_path,
        "--results-dir",
        out_dir,
        "--labels-key",
        str(labels_key),
        # Always sent: None means "no batch covariate", and the worker must hear that rather than
        # fall back to a default column the reference may not have.
        "--batch-key",
        NO_BATCH if batch_key is None else str(batch_key),
    ]
    if max_epochs_ref is not None:
        args += ["--max-epochs-ref", str(max_epochs_ref)]
    if max_epochs_map is not None:
        args += ["--max-epochs-map", str(max_epochs_map)]
    if n_cells_per_location is not None:
        args += ["--n-cells-per-location", str(n_cells_per_location)]
    if detection_alpha is not None:
        args += ["--detection-alpha", str(detection_alpha)]
    if cell_count_cutoff is not None:
        args += ["--cell-count-cutoff", str(cell_count_cutoff)]
    if cell_percentage_cutoff2 is not None:
        args += ["--cell-percentage-cutoff2", str(cell_percentage_cutoff2)]
    if nonz_mean_cutoff is not None:
        args += ["--nonz-mean-cutoff", str(nonz_mean_cutoff)]
    if n_neighbors is not None:
        args += ["--n-neighbors", str(n_neighbors)]
    if leiden_resolution is not None:
        args += ["--leiden-resolution", str(leiden_resolution)]
    if round_counts:
        args.append("--round-counts")
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    if batch_size_map is not None:
        args += ["--batch-size-map", str(batch_size_map)]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
