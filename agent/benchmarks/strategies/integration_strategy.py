"""Strategy for benchmarking multi-tool integration pipelines."""

from __future__ import annotations

from typing import Any

from benchmarks.config.benchmark_config import BenchmarkConfig
from benchmarks.data.data_registry import DataRegistry
from benchmarks.strategies.base_strategy import BenchmarkResult, BenchmarkStrategy


class IntegrationStrategy(BenchmarkStrategy):
    """Benchmark tool chains where output of one tool feeds into the next.

    Tests multi-tool pipelines (e.g., preprocessing -> clustering -> deconvolution)
    to verify tools work together correctly.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        data_registry: DataRegistry,
        pipelines: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(config, data_registry)
        self.pipelines = pipelines or []

    def select_tool_dataset_pairs(self) -> list[tuple[str, str]]:
        """Select datasets compatible with all tools in each pipeline."""
        pairs: list[tuple[str, str]] = []
        for pipeline in self.pipelines:
            dataset_name = pipeline.get("dataset", "")
            for step in pipeline.get("steps", []):
                tool_name = step.get("tool", "")
                pairs.append((tool_name, dataset_name))
        return pairs

    def execute(self, runner: Any) -> list[BenchmarkResult]:
        """Execute integration pipelines sequentially, passing outputs forward."""
        all_results: list[BenchmarkResult] = []
        for pipeline in self.pipelines:
            dataset_name = pipeline.get("dataset", "")
            dataset = self.data_registry.get(dataset_name)
            if dataset is None:
                continue
            pipeline_results = runner.run_pipeline(pipeline.get("steps", []), dataset, self.config)
            all_results.extend(pipeline_results)
        return all_results

    def summarize(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Summarize integration pipeline results."""
        return {
            "strategy": "integration",
            "total_pipelines": len(self.pipelines),
            "total_steps": len(results),
            "success_count": sum(1 for r in results if r.is_success()),
            "failure_count": sum(1 for r in results if not r.is_success()),
            "results": [r.to_dict() for r in results],
        }
