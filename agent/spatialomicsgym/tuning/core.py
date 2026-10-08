"""Core data models for the hyperparameter tuning system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TuningMode(Enum):
    """Four supported tuning modes."""

    BENCHMARK_LIGHT = "benchmark_tuning.light"
    BENCHMARK_FULL = "benchmark_tuning.full"
    ADAPTIVE = "adaptive_tuning"
    DEFAULT_FALLBACK = "default_fallback"


class ParameterType(Enum):
    """Types of tunable parameters."""

    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    CATEGORICAL = "categorical"


class DefaultSource(Enum):
    """Provenance classification for parameter defaults."""

    OFFICIAL_REPO = "official_repo"
    DOCUMENTATION = "documentation"
    PAPER = "paper"
    SOG_CONFIG = "spatialomicsgym_config"
    INFERRED = "inferred"


@dataclass
class ParameterValue:
    """A parameter value with provenance tracking."""

    name: str
    value: Any
    source: DefaultSource
    source_file: str  # GitHub URL, config path, or paper reference
    confidence: str  # "high", "medium", "low"
    version: str | None = None
    notes: str | None = None
    # False marks a row that records what upstream uses *without* adopting it -- written during a
    # divergence audit, where we deliberately run something else. Such a row is provenance, not a
    # default, so ``get_official_defaults`` withholds it rather than overlaying it onto the
    # mcp_config baseline and reinstating the value the row's own notes say we rejected.
    adopted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "source": self.source.value,
            "source_file": self.source_file,
            "confidence": self.confidence,
            "version": self.version,
            "notes": self.notes,
            "adopted": self.adopted,
        }


@dataclass
class ParameterSpec:
    """Specification for a tunable parameter."""

    name: str
    param_type: ParameterType
    default: Any
    description: str = ""
    min_value: float | None = None
    max_value: float | None = None
    choices: list[Any] | None = None
    step: float | None = None
    tunable: bool = True
    risk_level: str = "safe"  # "safe", "moderate", "dangerous"
    dependencies: list[str] | None = None
    official_default: ParameterValue | None = None
    # True for a knob that caps how many results the tool reports (``n_top_genes`` on an SVG tool).
    # A proxy objective that counts results would only maximise it, so proxy searches leave it out.
    output_cap: bool = False
    # Fixed parameters that, set to anything but 0/empty, leave this one without effect: scanpy's
    # ``resolution`` once ``target_n_clusters`` is fixed, since the worker then searches the
    # resolution itself. A search varying it re-ran one computation, and the "best" value it reported
    # was never measured (hunt 2026-09-30, u32-tuning-5 follow-up).
    inert_when_fixed: list[str] | None = None

    def validate(self, value: Any) -> bool:
        """Check if a value is valid for this parameter."""
        if self.choices is not None:
            return value in self.choices
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.param_type.value,
            "default": self.default,
            "tunable": self.tunable,
            "risk_level": self.risk_level,
        }
        if self.min_value is not None:
            d["min"] = self.min_value
        if self.max_value is not None:
            d["max"] = self.max_value
        if self.choices is not None:
            d["choices"] = self.choices
        if self.official_default:
            d["official_default"] = self.official_default.to_dict()
        return d


@dataclass
class TrialResult:
    """Result of a single hyperparameter configuration trial."""

    params: dict[str, Any]
    score: float  # Primary metric score
    secondary_scores: dict[str, float] = field(default_factory=dict)
    status: str = "success"  # "success", "failed", "timeout", "crashed"
    error_message: str = ""
    elapsed_seconds: float = 0.0
    seed: int = 42
    output_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "score": self.score,
            "secondary_scores": self.secondary_scores,
            "status": self.status,
            "error_message": self.error_message,
            "elapsed_seconds": self.elapsed_seconds,
            "seed": self.seed,
        }


# Budget defaults per mode
MODE_BUDGETS: dict[TuningMode, int] = {
    TuningMode.BENCHMARK_LIGHT: 8,
    TuningMode.BENCHMARK_FULL: 30,
    TuningMode.ADAPTIVE: 5,
    TuningMode.DEFAULT_FALLBACK: 1,
}

# Minimum improvement thresholds to accept tuned config over baseline
IMPROVEMENT_THRESHOLDS: dict[str, float] = {
    "spatial_clustering": 0.02,  # ARI improvement
    "svg_detection": 0.02,  # Jaccard improvement
    "deconvolution": -0.01,  # RMSE improvement (negative = better)
}

# Primary and secondary metrics per task type. These are read out of the dict the evaluator returns,
# so they have to be spelled the way it spells them -- ``eval_metrics.eval_deconvolution`` returns
# ``RMSE`` and ``mean_pearson_r``, not ``rmse`` and ``pearson_mean``. Under the old spellings every
# deconvolution trial missed its key and took the ``metrics.get(primary, 0.0)`` default, which for an
# error term is the *best* score there is, so the trials all tied at perfect. Same case drift #250
# fixed in the eval tool's prose.
TASK_METRICS: dict[str, dict[str, str]] = {
    "spatial_clustering": {"primary": "ARI", "secondary": "NMI"},
    "svg_detection": {"primary": "jaccard", "secondary": "f1"},
    "deconvolution": {"primary": "RMSE", "secondary": "mean_pearson_r"},
}

# Whether lower is better for each primary metric. Keyed by the same strings as TASK_METRICS above,
# and ``.get(metric, False)`` means an unknown name reads as higher-is-better -- so renaming a metric
# in one table without the other points the search at the worst configuration it can find.
METRIC_LOWER_IS_BETTER: dict[str, bool] = {
    "ARI": False,
    "NMI": False,
    "jaccard": False,
    "f1": False,
    "RMSE": True,
    "mean_pearson_r": False,
}

# The modes that score a trial against ground truth. Every other mode is scored by
# ``objectives.evaluate_proxy_trial`` instead -- see ``executor._evaluate_single``.
BENCHMARK_MODES: frozenset[TuningMode] = frozenset(
    {
        TuningMode.BENCHMARK_LIGHT,
        TuningMode.BENCHMARK_FULL,
    }
)


def score_is_lower_is_better(task_type: str, mode: TuningMode, primary_metric: str | None = None) -> bool:
    """Whether a trial score produced under *mode* is better when lower.

    Direction is a property of the evaluator, not of the metric name. Only the benchmark modes
    compute the task's real metric; everything else is scored by ``objectives._proxy_*``, and each of
    those returns a 0-1 quality composite where higher is better -- including for deconvolution,
    whose benchmark metric (RMSE) points the other way. Reading direction off ``primary_metric``
    alone therefore inverted every proxy-scored deconvolution run, making
    ``select_robust_config`` return the worst configuration tried.
    """
    if mode not in BENCHMARK_MODES:
        return False
    metric = primary_metric or TASK_METRICS.get(task_type, {}).get("primary", "")
    return METRIC_LOWER_IS_BETTER.get(metric, False)


@dataclass
class TuningConfig:
    """Configuration for a tuning run."""

    tool_name: str
    task_type: str
    mode: TuningMode
    dataset_path: str | None = None
    budget: int | None = None
    seed: int = 42
    timeout_per_run: int = 600
    primary_metric: str | None = None
    skip_reason: str | None = None
    sc_reference_path: str | None = None
    # The curated spatially-variable gene list an svg_detection search is scored against. Unlike
    # clustering and deconvolution, whose ground truth is a column inside ``dataset_path``, SVG has
    # no ground truth in the dataset -- without this the search runs and no trial can be scored.
    svg_ground_truth_path: str | None = None
    output_dir: str | None = None
    strategy_override: str | None = None  # "grid", "random", "staged", "bayesian"
    # The dataset's ground-truth obs column (the registry's ``ground_truth_key``). Trials used to be
    # scored against a hardcoded ``cell_type``, and where that column is absent the evaluator falls
    # back through its candidate list -- on MERFISH to ``Cell_class`` while the benchmark scores ``z``,
    # so the search optimised a different target than the one reported (hunt 2026-09-30,
    # u32-tuning-4). ``None`` names no column: the evaluator auto-detects one and records which
    # (u33b-bench-scoring-4); a named column the dataset lacks is refused, never substituted.
    ground_truth_key: str | None = None
    # Values every trial runs with and no trial varies: the dataset's cluster count, the reference's
    # cell-type column. A property of the dataset, not of the configuration, so they are passed to
    # each command but kept out of the search and out of the reported parameters (u32-tuning-5).
    fixed_params: dict[str, Any] | None = None

    def __post_init__(self):
        if self.budget is None:
            self.budget = MODE_BUDGETS.get(self.mode, 8)
        if self.primary_metric is None and self.task_type in TASK_METRICS:
            self.primary_metric = TASK_METRICS[self.task_type]["primary"]

    @property
    def lower_is_better(self) -> bool:
        return score_is_lower_is_better(self.task_type, self.mode, self.primary_metric)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "task_type": self.task_type,
            "mode": self.mode.value,
            "dataset_path": self.dataset_path,
            "budget": self.budget,
            "seed": self.seed,
            "timeout_per_run": self.timeout_per_run,
            "primary_metric": self.primary_metric,
            "skip_reason": self.skip_reason,
            "sc_reference_path": self.sc_reference_path,
            "svg_ground_truth_path": self.svg_ground_truth_path,
            "output_dir": self.output_dir,
            "strategy_override": self.strategy_override,
            "ground_truth_key": self.ground_truth_key,
            "fixed_params": self.fixed_params,
        }


@dataclass
class TuningResult:
    """Complete result of a tuning run."""

    config: TuningConfig
    best_params: dict[str, Any]
    best_score: float
    all_trials: list[TrialResult]
    baseline_params: dict[str, Any]
    baseline_score: float | None = None
    improvement: float | None = None
    mode_used: TuningMode = TuningMode.DEFAULT_FALLBACK
    mode_reason: str = ""
    #: Name of the strategy that actually ran, e.g. "grid_search". Distinct from
    #: ``config.strategy_override``, which is only what the caller asked for -- provenance recorded
    #: the request, so a run that ignored or downgraded it was written down as though it had not.
    strategy_used: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    elapsed_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def is_improvement(self) -> bool:
        """Whether tuning found a meaningful improvement over the baseline.

        ``improvement`` arrives already normalized: ``executor`` stores ``baseline - best`` for a
        minimized run and ``best - baseline`` for a maximized one, so it is positive whenever the
        tuned configuration is the better one, whichever way the metric points. This used to apply
        the direction a *second* time -- ``improvement <= threshold`` when ``lower_is_better`` --
        which is an exact inversion: an RMSE cut from 0.40 to 0.30 stores +0.10 and failed
        ``0.10 <= -0.01``, while the same run regressed to 0.50 stores -0.10 and passed. The other
        branch compared against the raw signed threshold, so deconvolution's -0.01 also accepted
        any decline shallower than 0.01. ``check_improvement_significance`` gets both of these
        right, so this defers to it rather than keeping a second, drifted copy of the rule.
        """
        if self.baseline_score is None or self.improvement is None:
            return False
        # Local: robustness imports this module, so a module-level import would be circular.
        from spatialomicsgym.tuning.robustness import check_improvement_significance

        is_significant, _, _ = check_improvement_significance(
            best_score=self.best_score,
            baseline_score=self.baseline_score,
            task_type=self.config.task_type,
            lower_is_better=self.config.lower_is_better,
        )
        return is_significant

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "best_params": self.best_params,
            "best_score": self.best_score,
            "all_trials": [t.to_dict() for t in self.all_trials],
            "baseline_params": self.baseline_params,
            "baseline_score": self.baseline_score,
            "improvement": self.improvement,
            "mode_used": self.mode_used.value,
            "mode_reason": self.mode_reason,
            "strategy_used": self.strategy_used,
            "provenance": self.provenance,
            "timestamp": self.timestamp,
            "elapsed_seconds": self.elapsed_seconds,
            "warnings": self.warnings,
        }
