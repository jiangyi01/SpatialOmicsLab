"""Runner for multi-tool pipeline benchmarks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.runners.base_runner import BenchmarkRunner
from benchmarks.strategies.base_strategy import BenchmarkResult

if TYPE_CHECKING:
    from benchmarks.config.benchmark_config import BenchmarkConfig
    from benchmarks.data.data_registry import DatasetEntry


def next_step_dataset(current: DatasetEntry, result: BenchmarkResult) -> DatasetEntry:
    """The dataset the next pipeline step reads: this step's h5ad output, when it wrote one.

    Both runners promised that each step's output_files feed the next step and assigned
    ``current_dataset`` once, so every step read the raw input and a pipeline whose steps could not
    work end to end reported every step "success" (hunt 2026-09-30, u33a-bench-runner-21). A step
    that wrote no h5ad hands on the dataset it was given.
    """
    produced = next(
        (Path(f) for f in result.output_files or [] if str(f).endswith(".h5ad") and Path(f).is_file()), None
    )
    if produced is None:
        return current
    # Absolute, so resolve_input's data_dir / input_path is the file itself.
    return replace(current, input_path=str(produced.resolve()))


class PipelineRunner(BenchmarkRunner):
    """Execute multi-tool pipelines with inter-step data passing.

    Wraps an MCPToolRunner to execute individual steps, managing
    output-to-input connections between pipeline stages.
    """

    def __init__(self, config: BenchmarkConfig, tool_runner: BenchmarkRunner | None = None) -> None:
        super().__init__(config)
        self._tool_runner = tool_runner

    def run_tool(
        self,
        tool_name: str,
        dataset: DatasetEntry,
        config: BenchmarkConfig,
    ) -> BenchmarkResult:
        """Delegate single tool runs to the underlying tool runner."""
        if self._tool_runner is None:
            return BenchmarkResult(
                tool_name=tool_name,
                dataset_name=dataset.name,
                task_type=dataset.task_type,
                status="error",
                error_message="No tool runner configured",
            )
        return self._tool_runner.run_tool(tool_name, dataset, config)

    def run_pipeline(
        self,
        steps: list[dict[str, Any]],
        dataset: DatasetEntry,
        config: BenchmarkConfig,
    ) -> list[BenchmarkResult]:
        """Execute pipeline steps sequentially, chaining outputs to inputs.

        Each step reads the h5ad the previous step wrote (:func:`next_step_dataset`).
        Pipeline halts on first failure.
        """
        results: list[BenchmarkResult] = []
        current_dataset = dataset

        for i, step in enumerate(steps):
            tool_name = step.get("tool", "")
            result = self.run_tool(tool_name, current_dataset, config)
            results.append(result)

            if not result.is_success():
                for remaining_step in steps[i + 1 :]:
                    results.append(
                        BenchmarkResult(
                            tool_name=remaining_step.get("tool", ""),
                            dataset_name=dataset.name,
                            task_type=dataset.task_type,
                            status="skipped",
                            error_message=f"Skipped due to failure in step {i}: {tool_name}",
                        )
                    )
                break
            current_dataset = next_step_dataset(current_dataset, result)

        return results
