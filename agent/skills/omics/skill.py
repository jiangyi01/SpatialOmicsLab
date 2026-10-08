"""Omics skill implementation."""

from __future__ import annotations

from typing import Any

from skills.base import BaseSkill
from skills.omics.mcp_tools import OMICS_MCP_TOOLS


class OmicsSkill(BaseSkill):
    """Skill for multi-omics analysis including literature search and data access."""

    @property
    def name(self) -> str:
        return "omics"

    @property
    def description(self) -> str:
        return "Multi-omics analysis skills including literature search, DOI validation, and impact assessment."

    @property
    def task_types(self) -> list[str]:
        return sorted({t["task"] for t in OMICS_MCP_TOOLS.values()})

    def get_mcp_tool_mapping(self) -> dict[str, dict[str, Any]]:
        return OMICS_MCP_TOOLS
