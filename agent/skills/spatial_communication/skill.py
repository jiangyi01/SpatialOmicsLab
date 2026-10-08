"""Spatial communication skill implementation."""

from __future__ import annotations

from typing import Any

from skills.base import BaseSkill
from skills.spatial_communication.mcp_tools import SPATIAL_COMMUNICATION_MCP_TOOLS


class SpatialCommunicationSkill(BaseSkill):
    """Skill for inferring spatial cell-cell communication in spatial transcriptomics."""

    @property
    def name(self) -> str:
        return "spatial_communication"

    @property
    def description(self) -> str:
        return (
            "Spatial cell-cell communication skills for inferring intercellular signaling from spatial transcriptomics."
        )

    @property
    def task_types(self) -> list[str]:
        return sorted({t["task"] for t in SPATIAL_COMMUNICATION_MCP_TOOLS.values()})

    def get_mcp_tool_mapping(self) -> dict[str, dict[str, Any]]:
        return SPATIAL_COMMUNICATION_MCP_TOOLS
