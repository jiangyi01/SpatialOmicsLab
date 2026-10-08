"""Spatial alignment skill implementation."""

from __future__ import annotations

from typing import Any

from skills.base import BaseSkill
from skills.spatial_alignment.mcp_tools import SPATIAL_ALIGNMENT_MCP_TOOLS


class SpatialAlignmentSkill(BaseSkill):
    """Skill for aligning and integrating multiple spatial transcriptomics sections."""

    @property
    def name(self) -> str:
        return "spatial_alignment"

    @property
    def description(self) -> str:
        return (
            "Spatial alignment and 3D reconstruction skills for registering multiple spatial transcriptomics sections."
        )

    @property
    def task_types(self) -> list[str]:
        return sorted({t["task"] for t in SPATIAL_ALIGNMENT_MCP_TOOLS.values()})

    def get_mcp_tool_mapping(self) -> dict[str, dict[str, Any]]:
        return SPATIAL_ALIGNMENT_MCP_TOOLS
