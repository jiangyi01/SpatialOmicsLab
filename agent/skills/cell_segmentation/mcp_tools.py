"""MCP tool knowledge base for cell segmentation."""

from __future__ import annotations

from typing import Any

CELL_SEGMENTATION_MCP_TOOLS: dict[str, dict[str, Any]] = {
    "cellpose": {
        "task": "cell_segmentation",
        "mcp_function": "run_cellpose_segmentation",
        "full_name": "Cellpose Segmentation",
        "description": "Segment cells in microscopy images using Cellpose's generalist deep learning segmentation model.",
        "input_requirements": {"images": True},
        "gpu": False,
        "priority": 1,
    },
    "deepcell": {
        "task": "cell_segmentation",
        "mcp_function": "run_deepcell_segmentation",
        "full_name": "DeepCell Segmentation",
        "description": "Perform cell and nuclear segmentation in tissue images using DeepCell's deep learning framework.",
        "input_requirements": {"images": True},
        "gpu": False,
        "priority": 1,
    },
    "bidcell": {
        "task": "cell_segmentation",
        "mcp_function": "run_bidcell",
        "full_name": "BIDCell",
        "description": "Segment cells from subcellular spatial transcriptomics data using BIDCell's biologically-informed approach.",
        "input_requirements": {"sc_reference": True, "images": True},
        "gpu": False,
        "priority": 2,
    },
    "clustermap": {
        "task": "cell_segmentation",
        "mcp_function": "run_clustermap",
        "full_name": "ClusterMap",
        "description": "Segment cells from single-molecule FISH data using ClusterMap's density-based clustering.",
        "gpu": False,
        "priority": 2,
    },
}
