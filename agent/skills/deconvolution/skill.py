"""Deconvolution skill implementation."""

from __future__ import annotations

from typing import Any

from skills.base import BaseSkill
from skills.deconvolution.mcp_tools import DECONVOLUTION_MCP_TOOLS


class DeconvolutionSkill(BaseSkill):
    """Skill for cell type deconvolution and single-cell to spatial mapping."""

    @property
    def name(self) -> str:
        return "deconvolution"

    @property
    def description(self) -> str:
        return "Cell type deconvolution and spatial mapping skills for spatial transcriptomics data."

    @property
    def task_types(self) -> list[str]:
        return sorted({t["task"] for t in DECONVOLUTION_MCP_TOOLS.values()})

    def get_mcp_tool_mapping(self) -> dict[str, dict[str, Any]]:
        return DECONVOLUTION_MCP_TOOLS
