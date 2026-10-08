"""Abstract base class for benchmark strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarks.config.benchmark_config import BenchmarkConfig
    from benchmarks.data.data_registry import DataRegistry


def is_numeric_metric(value: object) -> bool:
    """Is this reported metric a number that can be averaged or subtracted?

    ``BenchmarkResult.metrics`` holds the worker's block verbatim (see the note on that field), so
    every consumer that does arithmetic on it has to ask first. They ask through here rather than
    each spelling out its own ``isinstance``, because three copies of one rule is how the rule ends
    up meaning three different things.

    ``bool`` counts: ``isinstance(True, int)`` is true in Python, and a mean over a pass/fail flag is
    a rate worth having. The text report deliberately answers differently -- see
    ``report_generator._render_metric`` -- because rendering a single run's flag as ``1.0000`` would
    misreport what the tool actually said.
    """
    return isinstance(value, (int, float))


@dataclass
class BenchmarkResult:
    """Result from a single benchmark execution.

    Attributes:
        tool_name: Name of the benchmarked tool.
        dataset_name: Name of the dataset used.
        task_type: Task type (spatial_clustering, svg_detection, etc.).
        status: Execution status (success, failure, timeout, error).
        metrics: The metric block the tool reported, verbatim. Mostly numbers, but NOT only
            numbers, and the annotation says so on purpose. ``MCPToolRunner.run`` fills this from
            the worker's stdout JSON, and the shared ``WorkerOutput`` envelope is documented to put
            non-scalars in it: ``set_summary(n_clusters=5, cluster_sizes={"0": 200, ...})`` carries
            a dict, and an evaluation that raised is reported as ``{"error": str, "task_type":
            str}``. It used to be annotated ``dict[str, float]``, which read as a guarantee and was
            never one -- both readers below took it at its word and raised ``TypeError`` on the
            first tool that reported either shape. Consumers must ask what they actually need:
            averages over the numeric values, text over all of them.
        elapsed_seconds: Wall-clock execution time.
        output_files: Paths to output files produced.
        error_message: Error message if status is not success.
        config_snapshot: Configuration used for this run.
    """

    tool_name: str
    dataset_name: str
    task_type: str
    status: str = "pending"
    metrics: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    output_files: list[str] = field(default_factory=list)
    error_message: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "status": self.status,
            "metrics": self.metrics,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "output_files": self.output_files,
            "error_message": self.error_message,
        }


class BenchmarkStrategy(ABC):
    """Abstract base class for benchmark execution strategies.

    A strategy defines how tools are selected, paired with datasets,
    and evaluated. Subclasses implement specific benchmarking patterns.
    """

    def __init__(self, config: BenchmarkConfig, data_registry: DataRegistry) -> None:
        self.config = config
        self.data_registry = data_registry

    @abstractmethod
    def select_tool_dataset_pairs(self) -> list[tuple[str, str]]:
        """Select (tool_name, dataset_name) pairs for benchmarking.

        Returns:
            List of (tool_name, dataset_name) tuples to benchmark.
        """

    @abstractmethod
    def execute(self, runner: Any) -> list[BenchmarkResult]:
        """Execute the benchmark strategy using the provided runner.

        Args:
            runner: A BenchmarkRunner instance for executing individual tool runs.

        Returns:
            List of BenchmarkResult objects.
        """

    @abstractmethod
    def summarize(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Generate a summary of benchmark results.

        Args:
            results: List of completed BenchmarkResult objects.

        Returns:
            Summary dictionary with aggregated metrics and statistics.
        """
