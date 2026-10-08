"""MCP tool knowledge base for spatially variable gene detection."""

from __future__ import annotations

from typing import Any

SVG_DETECTION_MCP_TOOLS: dict[str, dict[str, Any]] = {
    "hotspot": {
        "task": "svg_detection",
        "mcp_function": "hotspot_spatial_modules",
        "full_name": "Hotspot Spatial Modules",
        "description": "Identify spatial gene expression modules and spatially variable genes using Hotspot.",
        "gpu": False,
        "priority": 1,
    },
    "somde": {
        "task": "svg_detection",
        "mcp_function": "somde_run",
        "full_name": "SOMDE",
        "description": "Detect spatially variable genes using SOMDE's self-organizing map differential expression approach.",
        "gpu": False,
        "priority": 1,
    },
    "spatialde": {
        "task": "svg_detection",
        "mcp_function": "spatialde_run_svg",
        "full_name": "SpatialDE",
        "description": "Identify spatially variable genes using SpatialDE's Gaussian process statistical framework.",
        "gpu": False,
        "priority": 1,
    },
    "svgbit": {
        "task": "svg_detection",
        "mcp_function": "svgbit_run",
        "full_name": "SVGbit",
        "description": "Detect spatially variable genes using SVGbit's binary imaging and spatial statistics approach.",
        "gpu": False,
        "priority": 2,
    },
    "bsp": {
        "task": "svg_detection",
        "mcp_function": "bsp_identify_svg",
        "full_name": "BSP",
        "description": "Identify spatially variable genes with scBSP (single-cell big-small patch): a granularity test on local-mean variances with a fitted log-normal null, giving a p-value per gene.",
        "gpu": False,
        "priority": 2,
    },
    "spark": {
        "task": "svg_detection",
        "mcp_function": "spark_svg_detection",
        "full_name": "SPARK",
        "description": "Detect spatially variable genes with SPARK's generalized spatial linear model framework.",
        "gpu": False,
        "priority": 2,
    },
    "spvc": {
        "task": "svg_detection",
        "mcp_function": "spvc_svg_detection",
        "full_name": "SPVC",
        "description": "Identify spatially variable genes using spVC's quasi-Poisson GAM with spatially varying coefficients.",
        "gpu": False,
        "priority": 2,
    },
    "spagft": {
        "task": "svg_detection",
        "mcp_function": "spagft_identify_svg",
        "full_name": "SpaGFT",
        "description": "Detect spatially variable genes using SpaGFT's graph Fourier transform on spatial data.",
        "gpu": False,
        "priority": 2,
    },
    "prost_svg": {
        "task": "svg_detection",
        "mcp_function": "prost_index_svg",
        "full_name": "PROST SVG Index",
        "description": "Identify spatially variable genes using the PROST index for spatial expression pattern ranking.",
        "gpu": False,
        "priority": 2,
    },
    "celina": {
        "task": "svg_detection",
        "mcp_function": "run_celina",
        "full_name": "CELINA",
        "description": "Detect cell-type-specific spatially variable genes using CELINA's conditional expression analysis.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 3,
    },
}
