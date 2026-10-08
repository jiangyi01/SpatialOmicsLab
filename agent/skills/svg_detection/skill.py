"""SVG detection skill implementation."""

from __future__ import annotations

from typing import Any

from skills.base import BaseSkill
from skills.svg_detection.mcp_tools import SVG_DETECTION_MCP_TOOLS


class SvgDetectionSkill(BaseSkill):
    """Skill for identifying spatially variable genes in spatial transcriptomics data."""

    @property
    def name(self) -> str:
        return "svg_detection"

    @property
    def description(self) -> str:
        return "Spatially variable gene detection skills for identifying genes with spatial expression patterns."

    @property
    def task_types(self) -> list[str]:
        return sorted({t["task"] for t in SVG_DETECTION_MCP_TOOLS.values()})

    def get_mcp_tool_mapping(self) -> dict[str, dict[str, Any]]:
        return SVG_DETECTION_MCP_TOOLS
