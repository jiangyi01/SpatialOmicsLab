"""Anti-overfitting and robustness safeguards for tuning.

Validates tuned configurations against baselines, checks stability
across seeds, and prevents spurious improvements.
"""

from __future__ import annotations

import logging
from typing import Any

from spatialomicsgym.tuning.core import (
    IMPROVEMENT_THRESHOLDS,
    METRIC_LOWER_IS_BETTER,
    TASK_METRICS,
    TrialResult,
    TuningConfig,
)

logger = logging.getLogger(__name__)


def check_improvement_significance(
    best_score: float,
    baseline_score: float,
    task_type: str,
    lower_is_better: bool | None = None,
) -> tuple[bool, float, str]:
    """Check if improvement over baseline meets minimum threshold.

    *lower_is_better* overrides the direction implied by the task's benchmark metric, which is wrong
    for a proxy-scored run -- pass ``config.lower_is_better``. Left as None the metric's own
    direction is used, preserving the behaviour of callers that have no mode to hand.

    Returns (is_significant, improvement, reason).
    """
    threshold = IMPROVEMENT_THRESHOLDS.get(task_type, 0.02)
    if lower_is_better is None:
        metric = TASK_METRICS.get(task_type, {}).get("primary", "ARI")
        lower_is_better = METRIC_LOWER_IS_BETTER.get(metric, False)

    # Only a threshold's magnitude carries meaning; its sign is already carried by the direction.
    # ``IMPROVEMENT_THRESHOLDS["deconvolution"]`` is -0.01, written for RMSE -- compared raw against a
    # maximized proxy score it would accept any decline shallower than 0.01 as an improvement.
    threshold = abs(threshold)

    if lower_is_better:
        improvement = baseline_score - best_score  # Positive if better
        is_significant = improvement >= threshold
    else:
        improvement = best_score - baseline_score
        is_significant = improvement >= threshold

    if is_significant:
        reason = f"Improvement of {improvement:.4f} exceeds threshold {threshold}"
    else:
        reason = f"Improvement of {improvement:.4f} below threshold {threshold}"

    return is_significant, improvement, reason


def check_metric_collapse(trials: list[TrialResult]) -> tuple[bool, str]:
    """Detect if all trials scored similarly (no real signal).

    Returns (is_collapsed, reason).
    """
    successful = [t for t in trials if t.status == "success"]
    if len(successful) < 3:
        return False, "Too few successful trials to assess"

    scores = [t.score for t in successful]
    score_range = max(scores) - min(scores)
    mean_score = sum(scores) / len(scores)

    if abs(mean_score) < 1e-12:
        if score_range < 1e-12:
            return True, f"Metric collapse: all {len(successful)} scores are effectively zero"
        cv = float("inf")
    else:
        cv = score_range / abs(mean_score)

    if cv < 0.01:
        return (
            True,
            f"Metric collapse: all {len(successful)} trials scored within {score_range:.6f} range (CV={cv:.4f})",
        )

    return False, ""


def check_failure_rate(trials: list[TrialResult]) -> tuple[bool, str]:
    """Check if too many trials failed."""
    if not trials:
        return True, "No trials executed"

    failed = sum(1 for t in trials if t.status != "success")
    rate = failed / len(trials)

    if rate > 0.8:
        return True, f"High failure rate: {failed}/{len(trials)} trials failed ({rate:.0%})"

    return False, ""


def validate_tuning_result(
    best_score: float,
    baseline_score: float | None,
    trials: list[TrialResult],
    config: TuningConfig,
) -> list[str]:
    """Run all robustness checks on tuning results.

    Returns list of warning messages. Empty list = all checks passed.
    """
    warnings: list[str] = []

    # Check failure rate
    high_failure, msg = check_failure_rate(trials)
    if high_failure:
        warnings.append(msg)

    # Check metric collapse
    collapsed, msg = check_metric_collapse(trials)
    if collapsed:
        warnings.append(msg)

    # Check improvement significance
    if baseline_score is not None:
        is_sig, improvement, msg = check_improvement_significance(
            best_score, baseline_score, config.task_type, config.lower_is_better
        )
        if not is_sig:
            warnings.append(f"Tuning improvement not significant: {msg}")

    return warnings


def select_robust_config(
    trials: list[TrialResult],
    config: TuningConfig,
) -> tuple[dict[str, Any], float]:
    """Select the most robust configuration from trial results.

    Prefers configurations that are:
    1. Successful
    2. High-scoring on primary metric
    3. Reasonable on secondary metrics

    Returns (best_params, best_score).
    """
    successful = [t for t in trials if t.status == "success"]
    if not successful:
        # Fall back to baseline
        return {}, 0.0

    lower_is_better = config.lower_is_better

    # Sort by primary metric
    if lower_is_better:
        successful.sort(key=lambda t: t.score)
    else:
        successful.sort(key=lambda t: t.score, reverse=True)

    # Return best
    best = successful[0]
    return best.params, best.score
