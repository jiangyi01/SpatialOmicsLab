"""Strategy for benchmarking individual MCP tools against compatible datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmarks.strategies.base_strategy import BenchmarkResult, BenchmarkStrategy, is_numeric_metric

if TYPE_CHECKING:
    from benchmarks.config.benchmark_config import BenchmarkConfig
    from benchmarks.data.data_registry import DataRegistry


class SingleToolStrategy(BenchmarkStrategy):
    """Benchmark each tool independently against its compatible datasets.

    For each tool in the filter, finds all compatible datasets and runs
    the tool on each one. Computes metrics per (tool, dataset) pair.
    """

    def __init__(self, config: BenchmarkConfig, data_registry: DataRegistry) -> None:
        super().__init__(config, data_registry)

    def select_tool_dataset_pairs(self) -> list[tuple[str, str]]:
        """Pair each filtered tool with its compatible datasets."""
        pairs: list[tuple[str, str]] = []
        for task_type in self.config.tool_filter.task_types:
            datasets = self.data_registry.filter_by_task(task_type)
            for dataset in datasets:
                for tool_name in dataset.compatible_tools:
                    if self._tool_matches_filter(tool_name):
                        pairs.append((tool_name, dataset.name))
        return pairs

    def execute(self, runner: Any) -> list[BenchmarkResult]:
        """Execute single-tool benchmarks for all selected pairs."""
        pairs = self.select_tool_dataset_pairs()
        results: list[BenchmarkResult] = []
        for tool_name, dataset_name in pairs:
            dataset = self.data_registry.get(dataset_name)
            if dataset is None:
                continue
            result = runner.run_tool(tool_name, dataset, self.config)
            results.append(result)
        return results

    def summarize(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Summarize per-tool benchmark results."""
        by_tool: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            by_tool.setdefault(r.tool_name, []).append(r)

        summary: dict[str, Any] = {
            "strategy": "single_tool",
            "total_runs": len(results),
            "success_count": sum(1 for r in results if r.is_success()),
            "failure_count": sum(1 for r in results if not r.is_success()),
            "per_tool": {},
        }

        for tool_name, tool_results in by_tool.items():
            successes = [r for r in tool_results if r.is_success()]
            avg_metrics: dict[str, float] = {}
            if successes:
                all_keys = set()
                for r in successes:
                    all_keys.update(r.metrics.keys())
                for key in all_keys:
                    # An average is only defined over numbers, and a worker's metric block is not
                    # only numbers -- its evaluation-failure shape is two strings, and its own
                    # documented summary example carries a per-cluster dict. Summing those raised
                    # TypeError and took the whole run's reporting with it. The denominator is
                    # already "the runs that reported this key" rather than "all runs", so
                    # narrowing it to the runs that reported a NUMBER for it is the same rule, not
                    # a new one. Keys no run put a number under simply do not get an average.
                    values = [r.metrics[key] for r in successes if is_numeric_metric(r.metrics.get(key))]
                    if values:
                        avg_metrics[key] = round(sum(values) / len(values), 4)

            summary["per_tool"][tool_name] = {
                "runs": len(tool_results),
                "successes": len(successes),
                "failures": len(tool_results) - len(successes),
                "avg_metrics": avg_metrics,
            }

        return summary

    def _tool_matches_filter(self, tool_name: str) -> bool:
        """Check if a tool passes the include/exclude filter."""
        f = self.config.tool_filter
        if f.include and tool_name not in f.include:
            return False
        if f.exclude and tool_name in f.exclude:
            return False
        return True
