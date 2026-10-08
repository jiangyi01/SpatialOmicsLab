"""Generate human-readable and machine-readable benchmark reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.strategies.base_strategy import is_numeric_metric
from spatialomicsgym.utils.file_io import json_safe

if TYPE_CHECKING:
    from benchmarks.reporting.result_schema import ResultSchema


def _render_metric(value: object) -> str:
    """Print a reported metric, whatever the tool actually reported.

    ``BenchmarkResult.metrics`` is the worker's metric block verbatim, and the shared
    ``WorkerOutput`` envelope is documented to put non-numbers in it -- a per-cluster
    ``cluster_sizes`` dict in its own ``set_summary`` example, and ``{"error": ..., "task_type":
    ...}`` when evaluation raised. This line used to be ``f"{v:.4f}"``, so the first such tool
    raised ``TypeError`` and no summary was written at all.

    Numbers keep their four decimals, because that is what makes a column of scores comparable at a
    glance. Everything else is written out as the tool said it: this is the only surface a reader
    would look at to find out that the metric they wanted is missing *and why*, so dropping the
    values that are not numbers would hide exactly the case worth seeing.

    ``bool`` is deliberately printed as ``True``/``False`` rather than ``1.0000``, even though
    ``is_numeric_metric`` counts it and ``SingleToolStrategy`` does average it -- the two are asking
    different questions. A mean over a pass/fail flag is a rate and worth having; a single run's flag
    rendered as a float would just misreport what the tool said.
    """
    if is_numeric_metric(value) and not isinstance(value, bool):
        return f"{value:.4f}"
    return str(value)


class ReportGenerator:
    """Generate benchmark reports in JSON and text formats.

    Produces both machine-readable JSON for downstream processing
    and human-readable summaries for quick review.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, schema: ResultSchema, filename: str = "benchmark_results.json") -> Path:
        """Save full results as JSON.

        ``json_safe`` is what makes the "as JSON" true. A metric that could not be computed arrives
        here as a non-finite float, and ``json.dump`` writes those as the bare tokens ``NaN`` and
        ``Infinity``, which RFC 8259 has no literals for. Python's ``json.loads`` accepts them, so
        nothing on this side of the project notices; ``JSON.parse``, ``jq``, R's ``jsonlite`` and
        Arrow reject the *whole document*, so one unmeasurable metric loses the entire report.
        """
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(json_safe(schema.to_dict()), f, indent=2)
        return path

    def save_summary(self, schema: ResultSchema, filename: str = "benchmark_summary.txt") -> Path:
        """Save human-readable summary."""
        path = self.output_dir / filename
        lines = self._format_summary(schema)
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    def _format_summary(self, schema: ResultSchema) -> list[str]:
        """Format a human-readable summary of benchmark results."""
        lines: list[str] = []
        lines.append("=" * 80)
        lines.append(f"Benchmark Report: {schema.run_id}")
        lines.append(f"Strategy: {schema.strategy}")
        lines.append(f"Timestamp: {schema.timestamp}")
        lines.append("=" * 80)
        lines.append("")

        summary = schema.summary
        lines.append(f"Total runs: {summary.get('total_runs', len(schema.results))}")
        lines.append(f"Successes: {summary.get('success_count', 0)}")
        lines.append(f"Failures: {summary.get('failure_count', 0)}")
        lines.append("")

        # FullBenchmarkWorkflow nests the RegressionStrategy summary under ``summary['regression']``;
        # only the top level was read, so its regressions never reached this file (hunt 2026-09-30,
        # u33b-bench-scoring-14). A RegressionStrategy run writes them at the top level.
        reg_summary = summary.get("regression") if isinstance(summary.get("regression"), dict) else summary
        if reg_summary.get("baseline_status") == "missing":
            # No baseline means no comparison ran; "Regressions detected: None" read as a clean
            # comparison (hunt 2026-09-30, hc-bench-runner request).
            lines.append(f"Regression check not run: {reg_summary.get('error') or 'no baseline to compare against'}")
        elif "regressions" in reg_summary:
            lines.append(f"Regressions detected: {reg_summary.get('regressions_found', 0)}")
            for reg in reg_summary.get("regressions", []):
                lines.append(
                    f"  - {reg['tool']}/{reg['dataset']}: {reg['metric']} "
                    f"{reg['baseline']:.4f} -> {reg['current']:.4f} ({reg['delta']:+.4f})"
                )
            lines.append("")

        lines.append("-" * 80)
        lines.append("Per-Result Details:")
        lines.append("-" * 80)
        for r in schema.results:
            status_icon = "PASS" if r.is_success() else "FAIL"
            lines.append(f"  [{status_icon}] {r.tool_name} on {r.dataset_name} ({r.elapsed_seconds:.1f}s)")
            if r.metrics:
                metric_str = ", ".join(f"{k}={_render_metric(v)}" for k, v in r.metrics.items())
                lines.append(f"         Metrics: {metric_str}")
            if r.error_message:
                lines.append(f"         Error: {r.error_message[:200]}")

        lines.append("")
        lines.append("=" * 80)
        return lines

    def generate_all(self, schema: ResultSchema) -> dict[str, Path]:
        """Generate all report formats and return paths."""
        return {
            "json": self.save_json(schema),
            "summary": self.save_summary(schema),
        }
