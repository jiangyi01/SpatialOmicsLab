"""Dynamic skill class for user-installed MCP tools.

Reads tool metadata from tools_user/install_log.json and exposes
active user tools to the SkillRegistry for ToolRetriever discovery.

This class is registered at runtime only when tool_creation_enabled=True.
It reads install_log.json each time get_mcp_tool_mapping() is called,
so newly installed tools are immediately discoverable.
"""

from __future__ import annotations

import json
from pathlib import Path

from spatialomicsgym.mcp_user_config import install_log_path


class UserToolSkill:
    """Skill class that dynamically reads user tool metadata from install_log.json."""

    @property
    def name(self) -> str:
        return "user_tools"

    @property
    def description(self) -> str:
        return "User-installed MCP tools from GitHub/paper links"

    @property
    def task_types(self) -> list[str]:
        return list({t.get("task_type", "other") for t in self._active_tools()})

    def _active_tools(self) -> list[dict]:
        """Read active tools from install_log.json, wherever this install keeps it.

        Resolved, not CWD-relative. ``stcoscientist._register_user_skill`` was taught to resolve
        the same file before deciding whether to register this skill, and this reader was not --
        so from any working directory but the checkout root the skill registered and then
        enumerated nothing. Registered-and-empty is the same thing to the user as never
        registered: every created tool is missing from the retriever, with nothing said. One
        ladder for the whole user-tool layer, which is what ``resolve_user_path`` is for.
        """
        log_path = Path(install_log_path())
        if not log_path.exists():
            return []
        try:
            entries = json.loads(log_path.read_text())
            if not isinstance(entries, list):
                return []
            return [e for e in entries if isinstance(e, dict) and e.get("status") == "active"]
        except (json.JSONDecodeError, OSError):
            return []

    def get_mcp_tool_mapping(self) -> dict:
        """Return tool metadata for all active user tools."""
        mapping = {}
        for entry in self._active_tools():
            tool_id = entry.get("tool_id", "")
            mapping[f"user_{tool_id}"] = {
                "task": entry.get("task_type", "other"),
                "mcp_function": entry.get("function_name", f"run_{tool_id}"),
                "full_name": entry.get("tool_name", tool_id),
                "description": entry.get("description", "User-installed tool"),
                "gpu": entry.get("gpu", False),
                "priority": 10,  # Lower priority than built-in tools (1-3)
                "source": "user",
                "strengths": entry.get("strengths", []),
                "limitations": ["User-installed, may have compatibility issues"],
                "best_for": entry.get("best_for", []),
            }
        return mapping
