"""Cell segmentation skill implementation."""

from __future__ import annotations

from typing import Any

from skills.base import BaseSkill
from skills.cell_segmentation.mcp_tools import CELL_SEGMENTATION_MCP_TOOLS


class CellSegmentationSkill(BaseSkill):
    """Skill for cell segmentation in spatial transcriptomics and microscopy images."""

    @property
    def name(self) -> str:
        return "cell_segmentation"

    @property
    def description(self) -> str:
        return (
            "Cell segmentation skills for identifying individual cells in spatial transcriptomics and microscopy data."
        )

    @property
    def task_types(self) -> list[str]:
        return sorted({t["task"] for t in CELL_SEGMENTATION_MCP_TOOLS.values()})

    def get_mcp_tool_mapping(self) -> dict[str, dict[str, Any]]:
        return CELL_SEGMENTATION_MCP_TOOLS
