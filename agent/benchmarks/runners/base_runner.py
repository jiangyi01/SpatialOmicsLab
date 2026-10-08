"""Abstract base class for benchmark runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from benchmarks.config.benchmark_config import BenchmarkConfig
from benchmarks.data.data_registry import DatasetEntry
from benchmarks.strategies.base_strategy import BenchmarkResult


class BenchmarkRunner(ABC):
    """Abstract base for executing benchmark tool runs.

    Subclasses implement the actual tool invocation logic (MCP subprocess,
    direct Python call, pipeline orchestration, etc.).
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    @abstractmethod
    def run_tool(
        self,
        tool_name: str,
        dataset: DatasetEntry,
        config: BenchmarkConfig,
    ) -> BenchmarkResult:
        """Run a single tool on a dataset and return the result.

        Args:
            tool_name: Name of the MCP tool to invoke.
            dataset: Dataset entry with input paths and metadata.
            config: Benchmark configuration.

        Returns:
            BenchmarkResult with status, metrics, and timing.
        """

    @abstractmethod
    def run_pipeline(
        self,
        steps: list[dict[str, Any]],
        dataset: DatasetEntry,
        config: BenchmarkConfig,
    ) -> list[BenchmarkResult]:
        """Run a multi-step pipeline on a dataset.

        Args:
            steps: List of pipeline step definitions, each with 'tool' and 'params'.
            dataset: Initial dataset entry.
            config: Benchmark configuration.

        Returns:
            List of BenchmarkResult, one per pipeline step.
        """
