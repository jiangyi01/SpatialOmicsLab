"""Spatial analysis skill implementation."""

from __future__ import annotations

from typing import Any

from skills.base import BaseSkill
from skills.spatial.mcp_tools import SPATIAL_ANALYSIS_MCP_TOOLS


class SpatialAnalysisSkill(BaseSkill):
    """Skill for spatial transcriptomics analysis, QC, imputation, and integration."""

    @property
    def name(self) -> str:
        return "spatial_analysis"

    @property
    def description(self) -> str:
        return "Spatial transcriptomics analysis skills including QC, neighbor analysis, imputation, factorization, and multiomics integration."

    @property
    def task_types(self) -> list[str]:
        return sorted({t["task"] for t in SPATIAL_ANALYSIS_MCP_TOOLS.values()})

    def get_mcp_tool_mapping(self) -> dict[str, dict[str, Any]]:
        return SPATIAL_ANALYSIS_MCP_TOOLS
