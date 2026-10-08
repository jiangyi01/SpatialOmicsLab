#!/usr/bin/env python3
"""``run_scresolve`` MCP wrapper for SpatialOmicsLab.

The tool keeps its historical name, but upstream scResolve is not run: what runs is reference
marker-gene scoring at the input spot resolution (see ``tools/scresolve_worker.py``).
"""

from __future__ import annotations

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "scresolve"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SCRESOLVE",
    "/opt/conda/envs/scresolve/bin/python3.8",
    "/workspace/epic-fermat/agent/tools/scresolve_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_scresolve(
    spatial_h5ad_path: str,
    sc_h5ad_path: str,
    output_dir: str,
    cell_type_key: str = "cell_type",
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Score every spatial spot for each reference cell type's marker signature (upstream scResolve is NOT run).

    What runs: Wilcoxon markers per reference cell type (scanpy ``rank_genes_groups``, top 20, on
    the genes shared with the spatial data after normalize_total + log1p), then one
    ``score_<cell type>`` obs column per type on the spots (scanpy ``score_genes``). The scores are
    computed at the input spot resolution: the output keeps one row per input spot and no
    resolution enhancement happens (ratio 1.0). Spots with ``obs['in_tissue'] == 0`` (background,
    as in CELLxGENE Visium exports) are left out first and counted in ``params.in_tissue_filter``.
    Upstream scResolve super-resolves a slide from its paired histology image, which this tool does
    not take, so it cannot run here. Nothing is downloaded at run time.

    Output: ``scresolve_enhanced.h5ad`` (historical name) -- the spatial input plus the score
    columns, with the markers used in ``uns['scresolve_marker_scoring']``. The payload names the
    method in ``params.method``.

    Parameters
    ----------
    spatial_h5ad_path:
        Path to the spatial AnnData (.h5ad) file (expression in X, raw counts expected). When its
        var_names are Ensembl IDs and the reference's are symbols, they are mapped through its first
        gene-symbol var column (``SYMBOL``, CELLxGENE's ``feature_name``, ``gene_symbols``,
        ``gene_name``, ``GeneName``), named in ``params.gene_symbol_column``.
    sc_h5ad_path:
        Path to the scRNA-seq AnnData (.h5ad) reference; its labelled cell types define the marker
        signatures.
    output_dir:
        Directory to write the output h5ad.
    cell_type_key:
        Column in sc_h5ad.obs containing cell type labels (at least 2 types, 2 cells each).
    drop_unlabeled:
        Drop reference cells whose label is missing (NaN/empty) and report how many. Default False:
        a missing label stops the run, because it is not a cell type.
    """
    args = [
        "--spatial-h5ad",
        spatial_h5ad_path,
        "--sc-h5ad",
        sc_h5ad_path,
        "--output-dir",
        output_dir,
        "--cell-type-key",
        cell_type_key,
    ]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
