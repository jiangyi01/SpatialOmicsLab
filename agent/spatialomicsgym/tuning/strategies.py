"""Search strategies for hyperparameter tuning.

Implements GridSearch, RandomSearch, StagedNarrowing, and PresetSelection
strategies. Each strategy generates candidate configurations and selects
the best based on evaluation results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from spatialomicsgym.tuning.core import TuningMode

if TYPE_CHECKING:
    from spatialomicsgym.tuning.search_space import SearchSpace


class SearchStrategy(ABC):
    """Base class for search strategies."""

    @abstractmethod
    def generate_candidates(self, space: SearchSpace, budget: int, seed: int = 42) -> list[dict[str, Any]]:
        """Generate candidate configurations to evaluate."""

    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""

    def optimize(
        self,
        space: SearchSpace,
        budget: int,
        evaluate_fn: Any,  # Callable[[dict[str, Any]], TrialResult]
        seed: int = 42,
    ) -> list[Any]:
        """Run optimization loop. Default: generate all candidates then evaluate each.

        Sequential strategies (e.g., Bayesian) override this to get
        per-trial feedback before suggesting the next candidate.
        """
        candidates = self.generate_candidates(space, budget, seed)
        results = []
        for params in candidates[:budget]:
            result = evaluate_fn(params)
            result.params = params
            results.append(result)
        return results


class GridSearch(SearchStrategy):
    """Small grid search over parameter space."""

    def generate_candidates(self, space: SearchSpace, budget: int, seed: int = 42) -> list[dict[str, Any]]:
        return space.generate_grid(max_configs=budget)

    def name(self) -> str:
        return "grid_search"


class RandomSearch(SearchStrategy):
    """Random sampling from parameter space."""

    def generate_candidates(self, space: SearchSpace, budget: int, seed: int = 42) -> list[dict[str, Any]]:
        return space.generate_random(n_configs=budget, seed=seed)

    def name(self) -> str:
        return "random_search"


class StagedNarrowing(SearchStrategy):
    """Two-stage search: broad exploration then focused refinement.

    Stage 1: Random exploration using 60% of budget
    Stage 2: Grid refinement around best result using 40% of budget
    """

    def __init__(self):
        self._stage2_configs: list[dict[str, Any]] = []

    def generate_candidates(self, space: SearchSpace, budget: int, seed: int = 42) -> list[dict[str, Any]]:
        """Returns stage 1 candidates. Call refine() after evaluation."""
        if budget < 2:
            baseline = space.baseline.copy() if space.baseline else {}
            return [baseline] if budget == 1 else []
        stage1_budget = min(max(3, int(budget * 0.6)), budget)
        return space.generate_random(n_configs=stage1_budget, seed=seed)

    def refine(
        self,
        space: SearchSpace,
        best_params: dict[str, Any],
        remaining_budget: int,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Generate stage 2 candidates narrowed around best result."""
        self._stage2_configs = space.narrow_around(best_params, n_configs=remaining_budget, seed=seed + 1)
        return self._stage2_configs

    def name(self) -> str:
        return "staged_narrowing"


@dataclass
class Preset:
    """A named preset configuration."""

    name: str
    description: str
    params: dict[str, Any]
    suitable_for: list[str]  # Dataset characteristics this preset suits


class PresetSelection(SearchStrategy):
    """Select from predefined parameter presets based on dataset characteristics.

    Used for adaptive tuning where ground truth is unavailable.
    """

    def __init__(self, presets: list[Preset] | None = None):
        self.presets = presets or []

    def generate_candidates(self, space: SearchSpace, budget: int, seed: int = 42) -> list[dict[str, Any]]:
        """Generate candidates from presets, augmented with baseline."""
        configs: list[dict[str, Any]] = []

        # Always include baseline
        if space.baseline:
            configs.append(space.baseline.copy())

        # Add preset configs
        for preset in self.presets[: budget - 1]:
            config = space.baseline.copy()
            config.update(preset.params)
            configs.append(config)

        # Fill remaining budget with slight variations of best presets
        if len(configs) < budget and self.presets:
            import random

            rng = random.Random(seed)
            while len(configs) < budget:
                base_preset = rng.choice(self.presets)
                config = space.baseline.copy()
                config.update(base_preset.params)
                # Small random perturbation
                for spec in space.parameters:
                    if spec.name in config and rng.random() < 0.3:
                        from spatialomicsgym.tuning.search_space import _perturb_value

                        config[spec.name] = _perturb_value(spec, config[spec.name], rng, factor=0.15)
                configs.append(config)

        return configs[:budget]

    def name(self) -> str:
        return "preset_selection"


# Default presets for common tool categories
CLUSTERING_PRESETS = [
    Preset(
        name="high_resolution",
        description="More clusters, finer spatial domains",
        params={"resolution": 1.0, "n_neighbors": 20},
        suitable_for=["large_dataset", "many_celltypes"],
    ),
    Preset(
        name="low_resolution",
        description="Fewer clusters, broad domains",
        params={"resolution": 0.3, "n_neighbors": 10},
        suitable_for=["small_dataset", "few_celltypes"],
    ),
    Preset(
        name="balanced",
        description="Balanced resolution for typical datasets",
        params={"resolution": 0.6, "n_neighbors": 15},
        suitable_for=["medium_dataset"],
    ),
]

DECONVOLUTION_PRESETS = [
    Preset(
        name="high_density",
        description="High cell density per location (e.g., Visium)",
        params={"n_cells_per_location": 30, "detection_alpha": 20.0},
        suitable_for=["visium", "high_density"],
    ),
    Preset(
        name="low_density",
        description="Low cell density (e.g., MERFISH, Slide-seqV2)",
        params={"n_cells_per_location": 5, "detection_alpha": 200.0},
        suitable_for=["merfish", "slide_seq", "low_density"],
    ),
    Preset(
        name="long_training",
        description="Extended training for complex datasets",
        params={"max_epochs_map": 5000, "max_epochs_ref": 100},
        suitable_for=["complex_dataset", "many_celltypes"],
    ),
]

SVG_PRESETS = [
    Preset(
        name="stringent",
        description="Strict SVG detection with fewer false positives",
        params={"n_neighbors": 20, "n_pcs": 50},
        suitable_for=["noisy_data", "many_genes"],
    ),
    Preset(
        name="sensitive",
        description="More lenient SVG detection",
        params={"n_neighbors": 10, "n_pcs": 20},
        suitable_for=["clean_data", "few_genes"],
    ),
]

TASK_PRESETS: dict[str, list[Preset]] = {
    "spatial_clustering": CLUSTERING_PRESETS,
    "deconvolution": DECONVOLUTION_PRESETS,
    "svg_detection": SVG_PRESETS,
}


#: The names ``tune(strategy=...)`` documents, mapped to what runs. "bayesian" is not here because
#: it needs constructor arguments and an optional dependency; it is handled separately below.
STRATEGY_BY_NAME: dict[str, type[SearchStrategy]] = {
    "grid": GridSearch,
    "random": RandomSearch,
    "staged": StagedNarrowing,
}


def select_strategy(
    mode: TuningMode,
    task_type: str = "",
    strategy_override: str | None = None,
) -> SearchStrategy:
    """Select appropriate search strategy for a tuning mode.

    An explicit ``strategy_override`` wins over the mode default -- that is what makes it an
    override. Only "bayesian" used to be honoured, so seven of the fifteen mode x strategy
    combinations quietly ran something else, and ``RandomSearch`` was unreachable from any caller.
    An unrecognised name still falls back to the mode default rather than raising, but says so,
    matching how an invalid ``mode`` is handled in ``tuning/__init__.py``.
    """
    if strategy_override is not None and strategy_override in STRATEGY_BY_NAME:
        return STRATEGY_BY_NAME[strategy_override]()

    if strategy_override is not None and strategy_override != "bayesian":
        import logging

        logging.getLogger(__name__).warning(
            "Unknown tuning strategy %r; using the %s default. Known strategies: %s",
            strategy_override,
            mode.value,
            ", ".join([*sorted(STRATEGY_BY_NAME), "bayesian"]),
        )

    if strategy_override == "bayesian":
        from spatialomicsgym.tuning.core import score_is_lower_is_better
        from spatialomicsgym.tuning.optuna_strategy import create_bayesian_strategy

        # Mode-aware: a proxy-scored run is maximized whatever its benchmark metric would have been,
        # or the TPE sampler spends the whole budget walking toward worse configurations.
        lower = score_is_lower_is_better(task_type, mode)
        return create_bayesian_strategy(lower_is_better=lower)

    if mode == TuningMode.BENCHMARK_LIGHT:
        return GridSearch()
    elif mode == TuningMode.BENCHMARK_FULL:
        return StagedNarrowing()
    elif mode == TuningMode.ADAPTIVE:
        presets = TASK_PRESETS.get(task_type, [])
        return PresetSelection(presets=presets)
    else:
        return GridSearch()  # Fallback just returns baseline
