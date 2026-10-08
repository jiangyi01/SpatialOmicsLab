"""Quick validation workflow - runs each tool with minimal data to verify functionality."""

from __future__ import annotations

from typing import Any

from benchmarks.config.benchmark_config import BenchmarkConfig
from benchmarks.data.data_registry import DataRegistry
from benchmarks.reporting.report_generator import ReportGenerator
from benchmarks.reporting.result_schema import ResultSchema
from benchmarks.runners.base_runner import BenchmarkRunner
from benchmarks.strategies.single_tool_strategy import SingleToolStrategy


class SmokeTestWorkflow:
    """Quick validation workflow for MCP tools.

    Runs each tool with the smallest available dataset to verify basic
    functionality (tool starts, processes data, returns valid JSON).
    Does not evaluate metric quality.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        data_registry: DataRegistry,
        runner: BenchmarkRunner,
    ) -> None:
        self.config = config
        self.data_registry = data_registry
        self.runner = runner

    def configure(self) -> SingleToolStrategy:
        """Create a single-tool strategy configured for smoke testing."""
        smoke_config = BenchmarkConfig(
            project_root=self.config.project_root,
            data_dir=self.config.data_dir,
            output_dir=self.config.output_dir + "/smoke_test",
            mcp_config_path=self.config.mcp_config_path,
            tool_filter=self.config.tool_filter,
            timeout_seconds=min(self.config.timeout_seconds, 120),
            max_workers=self.config.max_workers,
            save_intermediate=False,
            metrics=["ARI", "NMI"],
        )
        return SingleToolStrategy(config=smoke_config, data_registry=self.data_registry)

    def run(self) -> ResultSchema:
        """Execute the smoke test workflow.

        Returns:
            ResultSchema with all smoke test results.
        """
        import uuid

        strategy = self.configure()
        results = strategy.execute(self.runner)
        summary = strategy.summarize(results)

        return ResultSchema(
            run_id=f"smoke-{uuid.uuid4().hex[:8]}",
            strategy="smoke_test",
            config_snapshot=self.config.to_dict(),
            results=results,
            summary=summary,
        )

    def run_and_report(self, output_dir: str | None = None) -> dict[str, Any]:
        """Run smoke tests and generate reports."""
        schema = self.run()
        report_dir = output_dir or f"{self.config.output_dir}/smoke_test"
        generator = ReportGenerator(report_dir)
        paths = generator.generate_all(schema)
        return {"schema": schema, "report_paths": paths}
