"""Data conversion skill implementation."""

from __future__ import annotations

from typing import Any

from skills.base import BaseSkill
from skills.data_conversion.mcp_tools import DATA_CONVERSION_MCP_TOOLS


class DataConversionSkill(BaseSkill):
    """Skill for converting spatial transcriptomics data between file formats."""

    @property
    def name(self) -> str:
        return "data_conversion"

    @property
    def description(self) -> str:
        return "Data format conversion skills for transforming spatial transcriptomics files between H5AD, CSV, and Seurat formats."

    @property
    def task_types(self) -> list[str]:
        return sorted({t["task"] for t in DATA_CONVERSION_MCP_TOOLS.values()})

    def get_mcp_tool_mapping(self) -> dict[str, dict[str, Any]]:
        return DATA_CONVERSION_MCP_TOOLS
