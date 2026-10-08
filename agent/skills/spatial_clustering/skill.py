"""Spatial clustering skill implementation."""

from __future__ import annotations

from typing import Any

from skills.base import BaseSkill
from skills.spatial_clustering.mcp_tools import SPATIAL_CLUSTERING_MCP_TOOLS


class SpatialClusteringSkill(BaseSkill):
    """Skill for spatial domain identification and clustering in spatial transcriptomics."""

    @property
    def name(self) -> str:
        return "spatial_clustering"

    @property
    def description(self) -> str:
        return "Spatial domain identification and clustering skills for spatial transcriptomics data."

    @property
    def task_types(self) -> list[str]:
        return sorted({t["task"] for t in SPATIAL_CLUSTERING_MCP_TOOLS.values()})

    def get_mcp_tool_mapping(self) -> dict[str, dict[str, Any]]:
        return SPATIAL_CLUSTERING_MCP_TOOLS
