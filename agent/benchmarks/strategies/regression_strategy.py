"""Strategy for regression testing - detecting performance degradation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.strategies.base_strategy import BenchmarkResult, BenchmarkStrategy, is_numeric_metric

if TYPE_CHECKING:
    from benchmarks.config.benchmark_config import BenchmarkConfig
    from benchmarks.data.data_registry import DataRegistry

#: Metrics where a smaller value is the better result, beyond the tuning module's own table: the
#: deconvolution errors and divergences the evaluator reports. Matched case-insensitively, with a
#: leading ``mean_``/``median_`` ignored.
_LOWER_IS_BETTER = frozenset({"rmse", "mae", "mse", "jsd", "js_divergence", "kl_divergence"})


def lower_is_better(metric_name: str) -> bool:
    """Whether a rise in *metric_name* is a regression rather than an improvement.

    Every delta was read as higher-is-better, so an RMSE that rose from 0.10 to 0.30 was listed
    under improvements and a genuine RMSE drop was flagged as a regression (hunt 2026-09-30,
    u33a-bench-runner-22).
    """
    from spatialomicsgym.tuning.core import METRIC_LOWER_IS_BETTER

    if METRIC_LOWER_IS_BETTER.get(metric_name):
        return True
    name = metric_name.lower()
    for prefix in ("mean_", "median_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name in _LOWER_IS_BETTER


class RegressionStrategy(BenchmarkStrategy):
    """Compare current results against a saved baseline to detect regressions.

    Loads previous benchmark results as baseline, runs the same tool-dataset
    pairs, and flags any metric degradations beyond a configurable threshold.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        data_registry: DataRegistry,
        baseline_path: str | Path | None = None,
        regression_threshold: float = 0.05,
    ) -> None:
        super().__init__(config, data_registry)
        self.baseline_path = Path(baseline_path) if baseline_path else None
        self.regression_threshold = regression_threshold
        # Not dict[str, dict[str, float]]: _load_baseline copies each entry's metric block out of a
        # previous run's save_json verbatim, so the baseline side carries exactly the same shapes the
        # current side does. Annotating it float was how the subtraction below came to be written.
        self._baseline: dict[str, dict[str, Any]] = {}
        # A baseline the caller named and that is not there is not "no regressions": it is no
        # comparison. Skipped silently, it summarised as regressions_found 0, indistinguishable from
        # a clean run (hunt 2026-09-30, u33a-bench-runner-22).
        self.baseline_missing = bool(self.baseline_path) and not self.baseline_path.exists()
        if self.baseline_path and self.baseline_path.exists():
            self._load_baseline()

    def _load_baseline(self) -> None:
        """Load baseline results from JSON file."""
        with open(self.baseline_path) as f:  # type: ignore[arg-type]
            data = json.load(f)
        for entry in data.get("results", []):
            key = f"{entry['tool_name']}::{entry['dataset_name']}"
            self._baseline[key] = entry.get("metrics", {})

    def select_tool_dataset_pairs(self) -> list[tuple[str, str]]:
        """Select the same pairs that exist in the baseline."""
        pairs: list[tuple[str, str]] = []
        for key in self._baseline:
            tool_name, dataset_name = key.split("::", 1)
            pairs.append((tool_name, dataset_name))
        return pairs

    def execute(self, runner: Any) -> list[BenchmarkResult]:
        """Run tools and compare against baseline."""
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
        """Compare results against baseline and flag regressions."""
        if self.baseline_missing:
            return {
                "strategy": "regression",
                "threshold": self.regression_threshold,
                "baseline_status": "missing",
                "error": f"baseline file {self.baseline_path} does not exist; nothing was compared",
                "total_comparisons": 0,
                "regressions_found": None,
                "improvements_found": None,
                "regressions": [],
                "improvements": [],
            }

        regressions: list[dict[str, Any]] = []
        improvements: list[dict[str, Any]] = []

        for r in results:
            key = f"{r.tool_name}::{r.dataset_name}"
            baseline_metrics = self._baseline.get(key, {})
            if not baseline_metrics or not r.is_success():
                continue

            for metric_name, current_value in r.metrics.items():
                baseline_value = baseline_metrics.get(metric_name)
                if baseline_value is None:
                    continue
                # A degradation is a difference between two numbers, and a worker's metric block is
                # not only numbers on either side: the shared WorkerOutput envelope reports a failed
                # evaluation as {"error": str, "task_type": str} and its own documented summary
                # example carries a per-cluster dict -- and the baseline here is a previous run's
                # save_json, so both shapes arrive from the past as well as the present. Subtracting
                # them raised TypeError and took the whole regression report with it.
                # The run is not lost by skipping: save_json still records the metric block verbatim
                # and _format_summary still prints it. It just has no numeric delta to report.
                if not is_numeric_metric(current_value) or not is_numeric_metric(baseline_value):
                    continue
                delta = current_value - baseline_value
                # Oriented so a positive gain is always better, whichever way the metric points.
                gain = -delta if lower_is_better(metric_name) else delta
                if gain < -self.regression_threshold:
                    regressions.append(
                        {
                            "tool": r.tool_name,
                            "dataset": r.dataset_name,
                            "metric": metric_name,
                            "baseline": baseline_value,
                            "current": current_value,
                            "delta": round(delta, 4),
                        }
                    )
                elif gain > self.regression_threshold:
                    improvements.append(
                        {
                            "tool": r.tool_name,
                            "dataset": r.dataset_name,
                            "metric": metric_name,
                            "baseline": baseline_value,
                            "current": current_value,
                            "delta": round(delta, 4),
                        }
                    )

        return {
            "strategy": "regression",
            "threshold": self.regression_threshold,
            "baseline_status": "loaded" if self._baseline else "empty",
            "total_comparisons": len(results),
            "regressions_found": len(regressions),
            "improvements_found": len(improvements),
            "regressions": regressions,
            "improvements": improvements,
        }
