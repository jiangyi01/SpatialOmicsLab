#!/usr/bin/env python3
"""run_bulk2space MCP wrapper for SpatialOmicsLab: NNLS deconvolution (upstream Bulk2Space is not run)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "bulk2space"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "BULK2SPACE",
    "/opt/conda/envs/bulk2space_env/bin/python",
    "/workspace/epic-fermat/agent/tools/bulk2space_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_bulk2space(
    sc_h5ad_path: str,
    spatial_h5ad_path: str,
    output_dir: str = default_output_dir(),
    cell_type_key: str = "cell_type",
    max_genes: int = 1000,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Estimate per-spot cell-type proportions by non-negative least squares (NNLS) against the
    scRNA-seq reference's per-cell-type mean expression.

    What runs is NNLS, not upstream Bulk2Space: no VAE is trained, no single cells are generated,
    no bulk profile is used and nothing is mapped to single-cell resolution. The payload says so:
    ``params.method`` names NNLS, ``params.used_fallback`` is False (NNLS is this tool's only
    implementation, not a substitute) and the legacy ``params.used_nnls_fallback`` is True.

    Steps: leave out background spots (``obs['in_tissue'] == 0``, every array spot of a CELLxGENE
    Visium export; counted in ``params.in_tissue_filter`` and ``data.n_spots_supplied``, with no row in
    the outputs); harmonise gene identifiers and intersect the two gene panels; keep the ``max_genes``
    shared genes with the highest variance across reference cells (reported in ``params`` and the
    analysis text); average each cell type's reference profile; solve NNLS per spot and normalise the
    coefficients to sum to 1. A spot with no fit keeps an all-zero row and is counted in
    ``n_spots_unassigned``. ``X`` of both files is used as supplied, so give them on the same scale
    (NNLS does not normalise it as counts, so normalised data is accepted). Spatial coordinates are
    not read. No cells or spots are subsampled. The dense signature and proportion table are sized
    against the memory available (MemAvailable, bounded by a cgroup limit) before they are allocated.

    Outputs: ``bulk2space_proportions.csv`` (spots x cell types) and ``bulk2space_spatial.h5ad``
    (the spatial AnnData with the proportions in ``obsm['bulk2space_proportions']`` and one obs
    column per cell type). No CSV inputs are staged; ``output_files.csv_input_dir`` is None.

    Parameters
    ----------
    sc_h5ad_path:
        Path to single-cell reference AnnData (.h5ad) with cell-type labels.
    spatial_h5ad_path:
        Path to spatial transcriptomics AnnData (.h5ad), e.g. 10x Visium.
    output_dir:
        Directory where the proportions CSV and annotated h5ad are saved.
    cell_type_key:
        obs column in scRNA AnnData containing cell-type labels.
    max_genes:
        Number of shared genes, ranked by variance across reference cells, that the NNLS is solved
        on (default 1000). 0 keeps every shared gene.
    drop_unlabeled:
        Reference cells with a missing label (NaN/empty) stop the run by default; True leaves them
        out and reports how many were dropped.
    """
    sc_path = str(Path(sc_h5ad_path).expanduser())
    spatial_path = str(Path(spatial_h5ad_path).expanduser())
    out_dir = str(Path(output_dir).expanduser())

    args = [
        "--sc-h5ad",
        sc_path,
        "--spatial-h5ad",
        spatial_path,
        "--output-dir",
        out_dir,
        "--cell-type-key",
        cell_type_key,
        "--max-genes",
        str(int(max_genes)),
    ]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
