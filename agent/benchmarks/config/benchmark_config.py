"""Benchmark configuration dataclass and loader."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def repository_root(agent_root: str | Path) -> Path:
    """The repository root that goes with an agent tree: where ``data/`` and ``test/test_data`` live.

    In a checkout the agent-side trees (``benchmarks/``, ``MCP_server/``, ``tools/``) sit under
    ``<repo>/agent`` while ``data/`` and ``test/`` stay at the repository root, so the answer is the
    agent tree's parent. Any other root -- a flat tree, a test's scratch directory -- holds both kinds
    itself and is returned unchanged.
    """
    root = Path(agent_root)
    if root.name == "agent" and (root.parent / "pyproject.toml").is_file():
        return root.parent
    return root


@dataclass
class ToolFilter:
    """Filter for selecting which tools to benchmark."""

    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=lambda: ["python", "r"])


@dataclass
class BenchmarkConfig:
    """Central configuration for benchmark runs.

    Attributes:
        project_root: Path to the repository root (the launch directory by default).
        data_dir: Path to benchmark data directory.
        output_dir: Path for benchmark outputs and reports.
        mcp_config_path: Path to MCP configuration YAML, relative to project_root (see resolve_path).
        tool_filter: Filter for selecting tools to benchmark.
        timeout_seconds: Per-tool execution timeout.
        max_workers: Maximum parallel workers for benchmark execution.
        save_intermediate: Whether to save intermediate results.
        metrics: List of metric names to compute.
    """

    project_root: str = field(default_factory=lambda: (os.environ.get("SOG_PROJECT_ROOT") or os.environ.get("BIOMNI_PROJECT_ROOT") or "."))
    data_dir: str = "benchmark_data"
    output_dir: str = "benchmark_results"
    mcp_config_path: str = "MCP_server/mcp_config.yaml"
    tool_filter: ToolFilter = field(default_factory=ToolFilter)
    timeout_seconds: int = 600
    max_workers: int = 4
    save_intermediate: bool = True
    metrics: list[str] = field(default_factory=lambda: ["ARI", "NMI", "Jaccard", "RMSE", "Pearson"])

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        tool_filter_data = raw.pop("tool_filter", {})
        tool_filter = ToolFilter(**tool_filter_data) if tool_filter_data else ToolFilter()
        return cls(tool_filter=tool_filter, **raw)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dict for reproducibility snapshots."""
        return {
            "project_root": self.project_root,
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            "mcp_config_path": self.mcp_config_path,
            "tool_filter": {
                "include": self.tool_filter.include,
                "exclude": self.tool_filter.exclude,
                "task_types": self.tool_filter.task_types,
                "languages": self.tool_filter.languages,
            },
            "timeout_seconds": self.timeout_seconds,
            "max_workers": self.max_workers,
            "save_intermediate": self.save_intermediate,
            "metrics": self.metrics,
        }

    def resolve_path(self, relative: str) -> Path:
        """Resolve a relative path against project_root.

        In a checkout the agent-side trees (``MCP_server/``, ``benchmarks/``) sit under
        ``<project_root>/agent``, so a path into one of them that is absent at ``project_root``
        is read from there; a flat tree that holds them at its top level resolves as written.
        """
        base = Path(self.project_root)
        path = base / relative
        if not path.exists() and not Path(relative).is_absolute():
            moved = base / "agent" / relative
            if moved.exists():
                return moved
        return path
