"""Base class for all skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSkill(ABC):
    """Abstract base class for domain-specific skills.

    A skill encapsulates knowledge about which MCP tools to use for specific
    tasks within a domain, including tool capabilities, requirements, and
    recommended configurations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique skill name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this skill covers."""

    @property
    @abstractmethod
    def task_types(self) -> list[str]:
        """List of task types this skill can handle."""

    @abstractmethod
    def get_mcp_tool_mapping(self) -> dict[str, dict[str, Any]]:
        """Return the full MCP tool knowledge base for this skill.

        Returns:
            Dict mapping tool_key -> tool metadata including:
                - task: task type
                - mcp_function: MCP function name
                - full_name: human-readable name
                - description: what the tool does
                - strengths: list of strengths
                - limitations: list of limitations
                - best_for: list of ideal use cases
                - input_requirements: dict of required inputs -- ``sc_reference`` / ``images``
                  set True on a tool that cannot run without one; absent means not needed
                - key_params: dict of key parameters with defaults
                - gpu: whether GPU is needed
                - priority: selection priority (lower = preferred)
        """

    def get_available_tools(self) -> list[str]:
        """List all available tool keys in this skill."""
        return list(self.get_mcp_tool_mapping().keys())

    def get_tools_for_task(self, task_type: str) -> list[dict[str, Any]]:
        """Get tools suitable for a specific task type, sorted by priority.

        Args:
            task_type: The type of analysis task (e.g., 'spatial_clustering').

        Returns:
            List of tool metadata dicts, sorted by priority (lower first).
        """
        tools = [
            {**info, "key": key} for key, info in self.get_mcp_tool_mapping().items() if info.get("task") == task_type
        ]
        return sorted(tools, key=lambda t: t.get("priority", 99))

    def get_tool_by_key(self, key: str) -> dict[str, Any] | None:
        """Get metadata for a specific tool by its key."""
        return self.get_mcp_tool_mapping().get(key)

    def recommend_tools(
        self,
        task_type: str,
        has_gpu: bool = False,
        has_images: bool = False,
        has_sc_reference: bool = False,
        max_results: int = 3,
    ) -> list[dict[str, Any]]:
        """Recommend tools based on task type and available resources.

        Args:
            task_type: The analysis task type.
            has_gpu: Whether GPU is available.
            has_images: Whether a tissue or microscopy image (H&E, DAPI, fluorescence) is available.
            has_sc_reference: Whether single-cell reference is available.
            max_results: Maximum number of recommendations.

        Returns:
            List of recommended tool metadata dicts; empty when every tool for the task needs a
            resource the caller does not have.
        """
        candidates = self.get_tools_for_task(task_type)
        filtered = []
        for tool in candidates:
            reqs = tool.get("input_requirements", {})
            if reqs.get("images") and not has_images:
                continue
            if reqs.get("sc_reference") and not has_sc_reference:
                continue
            if tool.get("gpu") and not has_gpu:
                continue
            filtered.append(tool)

        # No "if nothing passes, return everything": that handed back exactly the tools just ruled
        # out, and until the rows declared ``input_requirements`` (taken from each tool's config
        # [DATA REQUIREMENTS] and required parameters) the filter above never fired at all, so
        # deconvolution without a reference recommended cell2location, Tangram and RCTD (hunt
        # 2026-09-30, u29b-skills-config-6). An empty list says no tool fits these resources.
        return filtered[:max_results]
