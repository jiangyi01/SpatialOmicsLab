"""Skills module - standalone skill management for SpatialOmicsLab.

Provides a registry-based framework for organizing domain-specific knowledge
about which MCP tools to use for which tasks.
"""

from skills.base import BaseSkill
from skills.registry import SkillRegistry

__all__ = ["BaseSkill", "SkillRegistry"]
