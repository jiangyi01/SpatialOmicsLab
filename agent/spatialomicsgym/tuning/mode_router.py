"""Mode routing — decides which tuning mode to use automatically.

This is the brain of the auto-tuning system. Given a tool, dataset, and
context (benchmark vs non-benchmark), it selects the correct mode:

  benchmark_tuning.light  — known benchmark dataset, fast search
  benchmark_tuning.full   — known benchmark dataset, thorough search
  adaptive_tuning         — non-benchmark data, heuristic adaptation
  default_fallback        — tuning not useful or not feasible

The routing logic:
  1. Check tuning policy — is tuning useful for this tool?
  2. Detect context     — is this a benchmark dataset?
  3. Check dataset      — is data available and big enough?
  4. Check cache        — do we already have tuned results?
  5. Select mode        — benchmark or adaptive or fallback
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from spatialomicsgym.tuning.core import TuningMode

logger = logging.getLogger(__name__)

_POLICIES_PATH = Path(__file__).parent / "configs" / "tuning_policies.yaml"
_POLICIES_CACHE: dict | None = None

# Known benchmark dataset paths (substrings that indicate benchmark data)
_BENCHMARK_INDICATORS = [
    "benchmark_data",
    "benchmarks/",
    "Eval1",
    "mock/",
    "_for_spatial_domain",
    "_for_spatial_variable",
    "_for_spatial_deconvolution",
    "Standard_h5ad",
]


def _load_policies() -> dict:
    global _POLICIES_CACHE
    if _POLICIES_CACHE is None:
        if _POLICIES_PATH.exists():
            with open(_POLICIES_PATH) as f:
                _POLICIES_CACHE = yaml.safe_load(f) or {}
        else:
            _POLICIES_CACHE = {}
    return _POLICIES_CACHE


def is_benchmark_context(dataset_path: str | None) -> bool:
    """Detect if the dataset is from the benchmark suite."""
    if not dataset_path:
        return False
    path_str = str(dataset_path).replace("\\", "/")
    return any(ind in path_str for ind in _BENCHMARK_INDICATORS)


def get_tool_policy(tool_name: str) -> dict:
    """Get the tuning policy for a specific tool."""
    policies = _load_policies()
    return policies.get("tool_policies", {}).get(tool_name, {})


def get_task_policy(task_type: str) -> dict:
    """Get the tuning policy for a task type."""
    policies = _load_policies()
    return policies.get("task_policies", {}).get(task_type, {})


def select_mode(
    tool_name: str,
    task_type: str,
    dataset_path: str | None = None,
    force_mode: TuningMode | None = None,
    consult_cache: bool = True,
) -> tuple[TuningMode, str]:
    """Automatically select the best tuning mode.

    Returns (mode, reason) explaining the decision.

    *consult_cache* is for a caller that will apply a cached config itself -- the agent, which
    reads ``load_best_config`` when the reason says "cached". ``tune()`` passes False: it is asked
    to search, and the cache branch hands it a fallback that it can only answer with the baseline
    (hunt 2026-09-30, u32-tuning-6).

    Decision tree:
      1. If force_mode set → use it
      2. If tool policy says tuning_useful=false → default_fallback
      3. If no dataset → default_fallback
      4. If dataset too small → default_fallback
      5. If cached tuned results exist → default_fallback (use cache)
      6. If benchmark context → benchmark_tuning.light
      7. If non-benchmark context → adaptive_tuning
    """
    # 1. Forced mode
    if force_mode is not None:
        return force_mode, f"Forced mode: {force_mode.value}"

    # 2. Check tool policy
    tool_pol = get_tool_policy(tool_name)
    if tool_pol.get("tuning_useful") is False:
        skip = tool_pol.get("skip_conditions", ["tuning not useful for this tool"])
        return TuningMode.DEFAULT_FALLBACK, f"Policy skip: {skip[0] if skip else 'tuning not useful'}"

    # 3. Check task policy
    task_pol = get_task_policy(task_type)
    if not task_pol:
        # Unknown task type — fallback
        if task_type not in ("spatial_clustering", "svg_detection", "deconvolution"):
            return TuningMode.DEFAULT_FALLBACK, f"Unknown task type: {task_type}"

    # 4. Check dataset availability
    if not dataset_path:
        return TuningMode.DEFAULT_FALLBACK, "No dataset path — using defaults"

    path = Path(dataset_path)
    if not path.exists():
        return TuningMode.DEFAULT_FALLBACK, f"Dataset not found: {dataset_path}"

    try:
        file_size = path.stat().st_size
    except OSError:
        return TuningMode.DEFAULT_FALLBACK, f"Cannot stat dataset: {dataset_path}"
    if file_size < 1024:
        return TuningMode.DEFAULT_FALLBACK, f"Dataset too small ({file_size} bytes)"

    # 5. Check dataset size via profiling
    try:
        from spatialomicsgym.tuning.adaptive import profile_dataset, should_downgrade_to_fallback

        profile = profile_dataset(dataset_path)
        should_down, reason = should_downgrade_to_fallback(profile, task_type)
        if should_down:
            return TuningMode.DEFAULT_FALLBACK, reason
    except Exception:
        pass  # If profiling fails, continue with mode selection

    # 6. Check for cached results
    from spatialomicsgym.tuning.persistence import load_best_config

    cached = load_best_config(tool_name) if consult_cache else None
    if cached and cached.get("params"):
        return TuningMode.DEFAULT_FALLBACK, f"Using cached tuned config (score={cached.get('score', '?')})"

    # 7. Benchmark vs non-benchmark
    if is_benchmark_context(dataset_path):
        return TuningMode.BENCHMARK_LIGHT, "Benchmark dataset detected — light tuning"
    else:
        return TuningMode.ADAPTIVE, "Non-benchmark dataset — adaptive tuning"
