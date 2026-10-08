#!/usr/bin/env python3
"""SpiceMix spatial factorization MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "spicemix"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPICEMIX",
    "/opt/conda/envs/spicemix_env/bin/python",
    "/workspace/epic-fermat/agent/tools/spicemix_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_spicemix(
    spatial_h5ad_path: str,
    output_dir: str = default_output_dir(),
    K: int = 10,
    n_epochs: int = 100,
    device: str = "cpu",
    min_spots_per_gene: int = 1,
) -> dict[str, Any]:
    """
    Run SpiceMix spatial factorization on one spatial transcriptomics slice.

    SpiceMix decomposes spatial gene expression into K latent factors
    (metagenes) while accounting for spatial neighborhood structure (an
    undirected 6-nearest-neighbour graph over the spot coordinates). It
    jointly learns a metagene dictionary and per-spot factor loadings.
    Each spot's dominant factor is written to obs['spicemix_factor'].

    Parameters
    ----------
    spatial_h5ad_path:
        Path to a spatial AnnData (.h5ad) file with expression in .X and spot
        coordinates in .obsm['spatial'] (or .obsm['X_spatial']). Non-negative
        integer counts in .X are normalised as SpiceMix specifies,
        log(1 + 1e4 * E / total counts of the spot); any other non-negative .X is
        used as supplied (taken to be normalised already, with a warning).
        Negative or non-finite values are refused. Spots with obs['in_tissue'] == 0
        (background outside the tissue) are left out and reported
        (params.in_tissue_filter and a warning); data.n_spots counts the spots
        factorised.
    output_dir:
        Directory to save factorization results (loadings, factors, plots). Every
        file is written to <name>.partial and moved into place when complete.
    K:
        Number of latent factors (metagenes) to learn.
    n_epochs:
        Total optimisation iterations: min(10, n_epochs // 2) NMF warm-up
        iterations (2 when n_epochs <= 5), then the rest (at least 1) with the
        spatial prior. Both counts are reported in params.
    device:
        Compute device: 'cpu' or 'cuda'. A GPU request on a machine without
        CUDA runs on CPU, and the payload says so.
    min_spots_per_gene:
        Leave out genes detected (non-zero) in fewer than this many spots. The
        default 1 drops only genes that are zero in every spot; 0 keeps every
        gene. SpiceMix holds the spots x genes matrix densely in float64, so
        raising this is the knob that shrinks it; data.n_genes reports the genes
        used and data.n_genes_input the genes read. Every in-tissue spot is kept.
    """
    h5ad = str(Path(spatial_h5ad_path).expanduser())
    out = str(Path(output_dir).expanduser())

    args = [
        "--spatial-h5ad",
        h5ad,
        "--output-dir",
        out,
        "--K",
        str(K),
        "--n-epochs",
        str(n_epochs),
        "--device",
        device,
        "--min-spots-per-gene",
        str(min_spots_per_gene),
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
