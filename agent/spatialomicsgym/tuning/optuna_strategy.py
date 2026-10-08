"""Optuna TPE Bayesian optimization strategy.

Optional dependency: install with ``pip install spatialomicsgym[tuning]``
Falls back to StagedNarrowing if optuna is not installed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from spatialomicsgym.tuning.core import ParameterType, TrialResult
from spatialomicsgym.tuning.search_space import SearchSpace

logger = logging.getLogger(__name__)

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    OPTUNA_AVAILABLE = False


def _define_param(trial: Any, spec: Any) -> Any:
    """Map a ParameterSpec to an Optuna suggest call."""
    if spec.param_type == ParameterType.BOOL:
        return trial.suggest_categorical(spec.name, [True, False])
    if spec.choices is not None:
        return trial.suggest_categorical(spec.name, spec.choices)
    if spec.param_type == ParameterType.CATEGORICAL:
        return spec.default

    # Derive bounds for numeric types
    low = spec.min_value
    high = spec.max_value
    if low is None or high is None:
        try:
            d = float(spec.default) if spec.default is not None and spec.default != 0 else 1.0
        except (TypeError, ValueError):
            d = 1.0
        low = low if low is not None else d * 0.3
        high = high if high is not None else d * 3.0

    if spec.param_type == ParameterType.INT:
        low_i, high_i = int(low), int(high)
        if low_i >= high_i:
            return spec.default
        kwargs: dict[str, Any] = {"name": spec.name, "low": low_i, "high": high_i}
        if spec.step is not None:
            kwargs["step"] = int(spec.step)
        return trial.suggest_int(**kwargs)
    if spec.param_type == ParameterType.FLOAT:
        low_f, high_f = float(low), float(high)
        if low_f >= high_f:
            return spec.default
        kwargs = {"name": spec.name, "low": low_f, "high": high_f}
        if spec.step is not None:
            kwargs["step"] = float(spec.step)
        return trial.suggest_float(**kwargs)
    # STR without choices -- use default
    return spec.default


class OptunaBayesianSearch:
    """Bayesian optimization using Optuna's TPE sampler.

    Unlike batch strategies, this is sequential: each trial result
    informs the next suggestion via Tree-structured Parzen Estimator.
    """

    def __init__(self, n_startup_trials: int = 3, lower_is_better: bool = False):
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for Bayesian search. Install with: pip install spatialomicsgym[tuning]")
        self.n_startup_trials = n_startup_trials
        self.lower_is_better = lower_is_better

    def generate_candidates(
        self,
        space: SearchSpace,
        budget: int,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Fallback: generate random candidates (satisfies strategy contract)."""
        return space.generate_random(n_configs=budget, seed=seed)

    def optimize(
        self,
        space: SearchSpace,
        budget: int,
        evaluate_fn: Callable[[dict[str, Any]], TrialResult],
        seed: int = 42,
    ) -> list[TrialResult]:
        """Run sequential Bayesian optimization.

        Creates an Optuna study with TPE sampler. Evaluates each trial
        via evaluate_fn before suggesting the next.
        """
        assert optuna is not None  # guarded by __init__
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        direction = "minimize" if self.lower_is_better else "maximize"
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=self.n_startup_trials,
        )
        study = optuna.create_study(direction=direction, sampler=sampler)

        # Enqueue baseline as trial 0 so it's always evaluated first
        if space.baseline:
            baseline_params = {}
            for spec in space.parameters:
                if spec.name in space.baseline:
                    baseline_params[spec.name] = space.baseline[spec.name]
            if baseline_params:
                study.enqueue_trial(baseline_params)

        trial_results: list[TrialResult] = []
        penalty = float("inf") if self.lower_is_better else float("-inf")

        def objective(trial: Any) -> float:
            try:
                params: dict[str, Any] = {}
                for spec in space.parameters:
                    params[spec.name] = _define_param(trial, spec)

                # Merge with baseline (non-tunable params keep their values)
                full_params = dict(space.baseline) if space.baseline else {}
                full_params.update(params)

                result = evaluate_fn(full_params)
                if result is None:
                    result = TrialResult(
                        params=full_params, score=penalty,
                        status="crashed", error_message="evaluate_fn returned None",
                    )
                result.params = full_params
                trial_results.append(result)

                if result.status != "success":
                    return penalty
                return result.score
            except Exception as e:
                logger.warning("Optuna trial %d failed: %s", trial.number, e)
                trial_results.append(
                    TrialResult(
                        params={},
                        score=penalty,
                        status="crashed",
                        error_message=str(e),
                    )
                )
                return penalty

        try:
            study.optimize(objective, n_trials=budget, catch=(Exception,))
        except Exception as e:
            logger.error("Optuna study failed: %s", e)
        return trial_results

    def name(self) -> str:
        return "optuna_bayesian"


def create_bayesian_strategy(
    lower_is_better: bool = False,
    n_startup_trials: int = 3,
) -> Any:
    """Factory: create OptunaBayesianSearch, or fallback to StagedNarrowing if optuna missing."""
    if not OPTUNA_AVAILABLE:
        from spatialomicsgym.tuning.strategies import StagedNarrowing

        logger.warning("optuna not installed, falling back to StagedNarrowing")
        return StagedNarrowing()
    return OptunaBayesianSearch(
        n_startup_trials=n_startup_trials,
        lower_is_better=lower_is_better,
    )
