#!/usr/bin/env python3
"""
Spatial Dataset Library MCP server.

Provides search_spatial_datasets tool for finding relevant datasets
from the built-in spatial transcriptomics dataset collection.
"""

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

MCP_NAME = "spatial_library"
TOOL_NAME = "spatial_library"

_default_python = "/opt/conda/bin/python"
_default_worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spatial_library_worker.py")
PYTHON_ENV, WORKER_PY = get_worker_paths("SPATIAL_LIBRARY", _default_python, _default_worker)

mcp = create_mcp(MCP_NAME)


# The docstring described a Visium-only library; it also holds Visium HD, Xenium and a single-cell atlas
# without coordinates (hunt 2026-09-30, u30-uncovered-mcp-9).
@mcp.tool()
def search_spatial_datasets(
    organism: str = "",
    organ: str = "",
    disease: str = "",
    technology: str = "",
    keyword: str = "",
    is_healthy: bool | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """
    Search the built-in spatial transcriptomics dataset library.

    Use this when a user asks about spatial analysis but has NOT provided
    any data file path. Datasets marked ``available`` come back with a
    resolved ``h5ad_path`` that spatial analysis MCP tools (clustering, SVG
    detection, deconvolution, etc.) can read.

    The library holds human datasets across many tissues (Brain, Breast,
    Lung, etc.), both healthy and diseased. Most are Visium or Visium
    CytAssist; it also holds Visium HD (bins, about 100x the spots of a
    Visium slide), Xenium, and one single-cell lung atlas that has no spatial
    coordinates. Each result's ``technology`` says which.

    Parameters
    ----------
    organism:
        Filter by organism. Examples: "human", "mouse", "Homo sapiens".
    organ:
        Filter by organ/tissue. Examples: "Brain", "Lung", "Breast".
    disease:
        Filter by disease keyword. Examples: "cancer", "glioblastoma", "healthy".
    technology:
        Filter by platform. Examples: "Visium", "MERFISH", "CytAssist".
    keyword:
        Free-text search across all metadata fields.
    is_healthy:
        If True, return only healthy samples. If False, only diseased.
    max_results:
        Maximum number of results to return (default 5).

    Returns
    -------
    Dictionary with ``n_results``, ``n_available`` and a ``results`` list. Each
    result carries ``available`` — whether the dataset's files are readable on
    this machine. The registry is a metadata catalog, so a match is not a
    guarantee of local data: when ``n_available`` is 0, the accompanying
    ``message`` explains how to point at a downloaded copy
    (``SOG_SPATIAL_LIBRARY_ROOT``) rather than the paths being broken.
    """
    args = []
    if organism:
        args.extend(["--organism", organism])
    if organ:
        args.extend(["--organ", organ])
    if disease:
        args.extend(["--disease", disease])
    if technology:
        args.extend(["--technology", technology])
    if keyword:
        args.extend(["--keyword", keyword])
    if is_healthy is not None:
        args.extend(["--is-healthy", str(is_healthy).lower()])
    args.extend(["--max-results", str(max_results)])

    return run_worker_cli(TOOL_NAME, PYTHON_ENV, WORKER_PY, args)


if __name__ == "__main__":
    mcp.run()
