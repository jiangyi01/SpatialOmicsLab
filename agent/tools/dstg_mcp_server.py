#!/usr/bin/env python3
"""DSTG-style spatial deconvolution MCP wrapper for SpatialOmicsLab.

The worker runs its own two-layer GCN over a kNN graph of spots + reference cells (label
propagation from the reference cell types). It does not run the upstream DSTG pseudo-spot pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "dstg"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "DSTG",
    "/opt/conda/envs/dstg_env/bin/python",
    "/workspace/epic-fermat/agent/tools/dstg_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_dstg(
    spatial_data_path: str,
    sc_data_path: str,
    output_dir: str,
    n_clusters: int = 7,
    learning_rate: float = 0.01,
    epochs: int = 200,
    cell_type_key: str = "",
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    DSTG-style deconvolution with a TensorFlow 1.x graph convolutional network.

    What runs: a two-layer GCN over a kNN graph joining the spatial spots and
    the reference cells, trained on the reference cell-type labels and read out
    on the spots as per-type probabilities (SpatialOmicsLab's reimplementation
    in the spirit of DSTG; the upstream DSTG pseudo-spot pipeline is NOT run).
    ``params.method`` names it and ``params.used_fallback`` is False: it is the only
    implementation, not a substitute.

    Spots whose ``obs['in_tissue']`` is 0 (background glass; CELLxGENE Visium exports carry every
    array spot) are left out of the graph and of both CSVs; ``data.n_spots`` counts the in-tissue
    spots, ``data.n_spots_supplied`` the file, and ``params.in_tissue_filter`` plus a warning say
    how many were left out. Output CSVs are written atomically.

    Parameters
    ----------
    spatial_data_path:
        Path to spatial transcriptomics data. Accepts an AnnData .h5ad file
        or a directory containing DSTG-formatted input files (mix_count.csv,
        mix_coord.csv).
    sc_data_path:
        Path to single-cell reference data. Accepts an AnnData .h5ad file
        or a directory containing DSTG-formatted files (sc_count.csv,
        sc_labels.csv).
    output_dir:
        Directory to write deconvolution outputs (dstg_proportions.csv,
        dstg_dominant_celltype.csv). No annotated h5ad is written.
    n_clusters:
        k of the kNN graph over spots + reference cells that the GCN
        propagates on (the name is historical; reported as params.knn_k).
    learning_rate:
        Learning rate for the GCN training.
    epochs:
        Number of training epochs.
    cell_type_key:
        obs column of the single-cell reference holding the cell-type labels.
        Leave empty to try the conventional names (cell_type, CellType,
        celltype, cell_type_key, annotation); the run fails if none exists.
        A missing label (NaN/empty, or a categorical code of -1) is not a cell
        type: the run refuses such a reference unless ``drop_unlabeled`` is True.
    drop_unlabeled:
        Default False. True leaves out reference cells whose label is missing
        instead of refusing the reference; ``params.n_reference_cells_dropped_unlabeled``
        and a warning say how many.
    """
    args = [
        "--spatial-data",
        str(Path(spatial_data_path).expanduser()),
        "--sc-data",
        str(Path(sc_data_path).expanduser()),
        "--output-dir",
        str(Path(output_dir).expanduser()),
        "--n-clusters",
        str(n_clusters),
        "--learning-rate",
        str(learning_rate),
        "--epochs",
        str(epochs),
    ]
    if cell_type_key:
        args += ["--cell-type-key", cell_type_key]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
