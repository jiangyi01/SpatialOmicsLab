"""SpatialOmicsLab Hyperparameter Tuning System.

Provides four tuning modes for optimizing tool hyperparameters:
- benchmark_tuning.light: Small search around the baseline over the 2-3 most impactful parameters
- benchmark_tuning.full: Staged search over every tunable parameter
- adaptive_tuning: Heuristic tuning for non-benchmark tasks
- default_fallback: Use baseline defaults when tuning is inappropriate
"""

from spatialomicsgym.tuning.core import (
    ParameterValue,
    TrialResult,
    TuningConfig,
    TuningMode,
    TuningResult,
)
from spatialomicsgym.tuning.executor import StagedTuningExecutor

try:
    from spatialomicsgym.tuning.optuna_strategy import OPTUNA_AVAILABLE
except ImportError:
    OPTUNA_AVAILABLE = False

__all__ = [
    "OPTUNA_AVAILABLE",
    "ParameterValue",
    "StagedTuningExecutor",
    "TrialResult",
    "TuningConfig",
    "TuningMode",
    "TuningResult",
    "tune",
]


def tune(
    tool_name: str,
    task_type: str,
    dataset_path: str | None = None,
    mode: TuningMode | str | None = None,
    budget: int | None = None,
    seed: int = 42,
    timeout_per_run: int = 600,
    sc_reference_path: str | None = None,
    svg_ground_truth_path: str | None = None,
    output_dir: str | None = None,
    strategy: str | None = None,
    ground_truth_key: str | None = None,
    fixed_params: dict | None = None,
) -> TuningResult:
    """Run hyperparameter tuning for a tool.

    Args:
        tool_name: MCP tool name (e.g., 'run_scanpy_spatial_domain').
        task_type: One of 'spatial_clustering', 'svg_detection', 'deconvolution'.
        dataset_path: Path to input dataset (.h5ad).
        mode: Tuning mode. Auto-selected if None.
        budget: Max configurations to evaluate. Auto-set by mode if None.
        seed: Random seed for reproducibility.
        timeout_per_run: Seconds per individual tool run.
        sc_reference_path: Path to single-cell reference (deconvolution tasks).
        svg_ground_truth_path: Path to the curated spatially-variable gene list (svg_detection
            tasks). Clustering and deconvolution take their ground truth from ``dataset_path``;
            SVG does not, so without this an svg_detection search cannot score any trial.
        output_dir: Directory for tuning outputs.
        strategy: Search strategy override. One of "grid", "random", "staged",
            "bayesian". If None, ``SOG_TUNING_STRATEGY`` (``default_config.tuning_strategy``)
            when set, otherwise auto-selected by mode. "bayesian" requires
            optuna (install with ``pip install spatialomicsgym[tuning]``).
        ground_truth_key: The dataset's ground-truth obs column (benchmark modes). A benchmark
            dataset's registry entry names it; ``None`` names none, so the evaluator auto-detects
            the column and records which one it used. A run whose named column is absent from the
            dataset falls back instead of scoring another one.
        fixed_params: Values every trial runs with and none varies -- the dataset's cluster count,
            the reference's cell-type column (``integration.registry_fixed_params`` reads them from
            a registry entry). Kept out of the search and out of the reported parameters.

    Returns:
        TuningResult with best parameters, scores, and provenance.
    """
    if isinstance(mode, str):
        try:
            mode = TuningMode(mode)
        except ValueError:
            import logging

            logging.getLogger(__name__).warning("Invalid tuning mode '%s', falling back to auto-select", mode)
            mode = None

    # Auto-select mode if not specified. The router's reason travels with the run: a fallback used to
    # be reported as "Explicitly set to fallback" whatever the router had said -- a missing dataset,
    # a policy skip. Nor does an explicit call to tune stop at the agent's tuned-config cache: that
    # branch returned the baseline, not the cached values it named, and ran no search
    # (hunt 2026-09-30, u32-tuning-6).
    reason = None
    if mode is None:
        from spatialomicsgym.tuning.mode_router import select_mode

        mode, reason = select_mode(tool_name, task_type, dataset_path, consult_cache=False)

    # ``SOG_TUNING_STRATEGY`` is documented as "Force strategy" and nothing read it: an operator who
    # set it got the mode's default strategy and no word of it (hunt 2026-09-30, u32-tuning-25). An
    # explicit *strategy* still wins.
    from spatialomicsgym.config import default_config

    config = TuningConfig(
        tool_name=tool_name,
        task_type=task_type,
        mode=mode,
        dataset_path=dataset_path,
        budget=budget,
        seed=seed,
        timeout_per_run=timeout_per_run,
        sc_reference_path=sc_reference_path,
        svg_ground_truth_path=svg_ground_truth_path,
        output_dir=output_dir,
        strategy_override=strategy if strategy is not None else default_config.tuning_strategy,
        skip_reason=reason if mode == TuningMode.DEFAULT_FALLBACK else None,
        ground_truth_key=ground_truth_key,
        fixed_params=dict(fixed_params) if fixed_params else None,
    )

    executor = StagedTuningExecutor(config)
    return executor.run()
