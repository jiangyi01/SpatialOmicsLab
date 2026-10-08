"""Skill registry - central management for all domain skills."""

from __future__ import annotations

from typing import Any

from skills.base import BaseSkill


class SkillRegistry:
    """Central registry for domain-specific skills.

    Provides discovery, lookup, and aggregation across all registered skills.
    """

    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        """Register a skill instance."""
        self._skills[skill.name] = skill

    def get_skill(self, name: str) -> BaseSkill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        """List all registered skill names."""
        return list(self._skills.keys())

    def get_all_tools(self) -> dict[str, dict[str, Any]]:
        """Aggregate all MCP tool mappings from all registered skills.

        Returns:
            Combined dict of tool_key -> tool metadata across all skills.
        """
        all_tools: dict[str, dict[str, Any]] = {}
        for skill in self._skills.values():
            all_tools.update(skill.get_mcp_tool_mapping())
        return all_tools

    def get_all_task_types(self) -> list[str]:
        """List unique task types across all skills."""
        task_types: set[str] = set()
        for skill in self._skills.values():
            task_types.update(skill.task_types)
        return sorted(task_types)

    def find_tools_for_task(self, task_type: str) -> list[dict[str, Any]]:
        """Find tools across all skills for a given task type."""
        results: list[dict[str, Any]] = []
        for skill in self._skills.values():
            results.extend(skill.get_tools_for_task(task_type))
        return sorted(results, key=lambda t: t.get("priority", 99))

    def recommend(
        self,
        task_type: str,
        has_gpu: bool = False,
        has_images: bool = False,
        has_sc_reference: bool = False,
        max_results: int = 3,
    ) -> list[dict[str, Any]]:
        """Recommend tools across all skills for a task."""
        candidates: list[dict[str, Any]] = []
        for skill in self._skills.values():
            candidates.extend(skill.recommend_tools(task_type, has_gpu, has_images, has_sc_reference, max_results=99))
        candidates.sort(key=lambda t: t.get("priority", 99))
        return candidates[:max_results]

    @classmethod
    def create_default(cls) -> SkillRegistry:
        """Create a registry with all built-in skills pre-registered."""
        registry = cls()

        from skills.cell_segmentation.skill import CellSegmentationSkill
        from skills.data_conversion.skill import DataConversionSkill
        from skills.deconvolution.skill import DeconvolutionSkill
        from skills.omics.skill import OmicsSkill
        from skills.spatial.skill import SpatialAnalysisSkill
        from skills.spatial_alignment.skill import SpatialAlignmentSkill
        from skills.spatial_clustering.skill import SpatialClusteringSkill
        from skills.spatial_communication.skill import SpatialCommunicationSkill
        from skills.svg_detection.skill import SvgDetectionSkill

        registry.register(SpatialClusteringSkill())
        registry.register(DeconvolutionSkill())
        registry.register(SvgDetectionSkill())
        registry.register(CellSegmentationSkill())
        registry.register(SpatialAlignmentSkill())
        registry.register(SpatialCommunicationSkill())
        registry.register(SpatialAnalysisSkill())
        registry.register(DataConversionSkill())
        registry.register(OmicsSkill())

        return registry
