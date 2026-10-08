"""Comprehensive benchmark workflow - runs all strategies with full evaluation."""

from __future__ import annotations

from typing import Any

from benchmarks.config.benchmark_config import BenchmarkConfig
from benchmarks.data.data_registry import DataRegistry
from benchmarks.reporting.report_generator import ReportGenerator
from benchmarks.reporting.result_schema import ResultSchema
from benchmarks.runners.base_runner import BenchmarkRunner
from benchmarks.strategies.single_tool_strategy import SingleToolStrategy


class FullBenchmarkWorkflow:
    """Comprehensive benchmark workflow.

    Runs single-tool benchmarks on all compatible tool-dataset pairs,
    computes full evaluation metrics, and generates detailed reports.
    Optionally runs regression checks against a baseline.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        data_registry: DataRegistry,
        runner: BenchmarkRunner,
        baseline_path: str | None = None,
    ) -> None:
        self.config = config
        self.data_registry = data_registry
        self.runner = runner
        self.baseline_path = baseline_path

    def run(self) -> ResultSchema:
        """Execute the full benchmark workflow.

        Steps:
        1. Run single-tool benchmarks for all filtered tools.
        2. Optionally run regression checks against baseline.
        3. Aggregate results and generate summary.
        """
        import uuid

        strategy = SingleToolStrategy(config=self.config, data_registry=self.data_registry)
        results = strategy.execute(self.runner)
        summary = strategy.summarize(results)

        if self.baseline_path:
            from benchmarks.strategies.regression_strategy import RegressionStrategy

            regression = RegressionStrategy(
                config=self.config,
                data_registry=self.data_registry,
                baseline_path=self.baseline_path,
            )
            regression_summary = regression.summarize(results)
            summary["regression"] = regression_summary

        return ResultSchema(
            run_id=f"full-{uuid.uuid4().hex[:8]}",
            strategy="full_benchmark",
            config_snapshot=self.config.to_dict(),
            results=results,
            summary=summary,
        )

    def run_and_report(self, output_dir: str | None = None) -> dict[str, Any]:
        """Run full benchmarks and generate reports."""
        schema = self.run()
        report_dir = output_dir or self.config.output_dir
        generator = ReportGenerator(report_dir)
        paths = generator.generate_all(schema)
        return {"schema": schema, "report_paths": paths}
