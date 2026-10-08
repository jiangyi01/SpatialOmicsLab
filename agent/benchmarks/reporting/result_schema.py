"""Schema definitions for benchmark results.

Provides validation and serialization for benchmark output data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from benchmarks.strategies.base_strategy import BenchmarkResult


@dataclass
class ResultSchema:
    """Top-level schema for a complete benchmark run.

    Attributes:
        run_id: Unique identifier for this benchmark run.
        timestamp: ISO format timestamp of when the run started.
        strategy: Name of the strategy used.
        config_snapshot: Full configuration used.
        results: List of individual benchmark results.
        summary: Aggregated summary from the strategy.
    """

    run_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    strategy: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    results: list[BenchmarkResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "strategy": self.strategy,
            "config": self.config_snapshot,
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResultSchema:
        """Deserialize from dictionary."""
        results = []
        for r in data.get("results", []):
            results.append(BenchmarkResult(
                tool_name=r.get("tool_name", ""),
                dataset_name=r.get("dataset_name", ""),
                task_type=r.get("task_type", ""),
                status=r.get("status", ""),
                metrics=r.get("metrics", {}),
                elapsed_seconds=r.get("elapsed_seconds", 0.0),
                output_files=r.get("output_files", []),
                error_message=r.get("error_message", ""),
            ))
        return cls(
            run_id=data.get("run_id", ""),
            timestamp=data.get("timestamp", ""),
            strategy=data.get("strategy", ""),
            config_snapshot=data.get("config", {}),
            results=results,
            summary=data.get("summary", {}),
        )
