#!/usr/bin/env python3
"""STdGCN spatial deconvolution MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "stdgcn"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "STDGCN",
    "/opt/conda/envs/stdgcn_env/bin/python",
    "/workspace/epic-fermat/agent/tools/stdgcn_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_stdgcn(
    spatial_h5ad_path: str,
    sc_h5ad_path: str,
    output_dir: str,
    cell_type_key: str = "cell_type",
    n_epochs: int = 200,
    device: str = "CPU",
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Run STdGCN spatial deconvolution using a graph convolutional network.

    STdGCN estimates cell-type proportions for each spatial spot by learning
    from a single-cell RNA-seq reference dataset. It constructs a spatial
    graph and uses a GCN to propagate cell-type information.

    The spatial graph is built in spot pitches: the coordinates are divided by the
    median nearest-neighbour distance before STdGCN reads them, and spots closer than
    1.9 pitches are linked (the tutorial's neighbourhood: every ring below 2 pitches),
    whatever the platform's unit.
    The payload reports the pitch (params.spot_pitch) and the number of spatial edges
    (summary.n_spatial_edges). STdGCN's graphs are dense matrices over every spot and
    pseudo-spot (10 per spot, at most 30000); a run that cannot fit them in memory is refused
    with the numbers as soon as the slide is read, before anything is staged, and never
    subsampled. Spots with obs['in_tissue'] == 0 (background) are left out and counted in
    params.in_tissue_filter and the warnings.

    Parameters
    ----------
    spatial_h5ad_path:
        Path to spatial transcriptomics AnnData (.h5ad) with counts in .X
        and spatial coordinates in obsm['spatial'].
    sc_h5ad_path:
        Path to annotated single-cell reference AnnData (.h5ad) with
        cell-type labels in obs[cell_type_key].
    output_dir:
        Directory to write deconvolution outputs (proportions CSV,
        annotated h5ad). The proportion CSVs keep the reference's labels; in the h5ad's obs a
        label HDF5 cannot hold as a key (one containing '/') is spelled with '_' instead, listed
        in params.annotated_h5ad_obs_columns_renamed.
    cell_type_key:
        Column name in sc_h5ad.obs containing cell-type labels.
    n_epochs:
        Number of training epochs for the STdGCN model.
    device:
        Compute device: "CPU" or "GPU".
    drop_unlabeled:
        Reference cells with a missing (NaN/empty) label are refused by default; true
        leaves them out and reports the count in params.n_reference_cells_dropped_unlabeled.
    """
    args = [
        "--spatial-h5ad",
        str(Path(spatial_h5ad_path).expanduser()),
        "--sc-h5ad",
        str(Path(sc_h5ad_path).expanduser()),
        "--output-dir",
        str(Path(output_dir).expanduser()),
        "--cell-type-key",
        cell_type_key,
        "--n-epochs",
        str(n_epochs),
        "--device",
        device,
    ]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
