"""Search space definitions for hyperparameter tuning.

Defines how search spaces are constructed for each tuning mode,
anchored around official/SpatialOmicsLab baseline defaults.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from spatialomicsgym.tuning.core import BENCHMARK_MODES, ParameterSpec, ParameterType, TuningMode


@dataclass
class SearchSpace:
    """A search space for a specific tool and tuning mode."""

    tool_name: str
    mode: TuningMode
    parameters: list[ParameterSpec]
    baseline: dict[str, Any] = field(default_factory=dict)

    def generate_grid(self, max_configs: int = 10) -> list[dict[str, Any]]:
        """Generate grid search configurations.

        Produces configurations centered around the baseline, respecting the max_configs budget.

        Each parameter's axis is its baseline value and its neighbours -- half and double the
        baseline, clipped into the declared range, or the other choices. The full product is used
        when it fits the budget, and otherwise each parameter is varied alone with the rest at the
        baseline. This used to be an evenly spaced grid over [min, max] at
        ``max(2, int(budget ** (1/n)))`` points, which at the light budget of 8 is exactly the two
        range corners: no value near the default was ever tried, and the corners included 1500/1500
        DeepST epochs and 2000 STAGATE epochs, past the per-run timeout (hunt 2026-09-30,
        u32-tuning-12).
        """
        if not self.parameters:
            return [self.baseline.copy()] if self.baseline else [{}]

        axes = {spec.name: _grid_axis(spec, self.baseline.get(spec.name, spec.default)) for spec in self.parameters}

        size = 1
        for values in axes.values():
            size *= len(values)

        configs = [self.baseline.copy()] if self.baseline else []
        if size <= max_configs:
            import itertools

            names = list(axes)
            for combo in itertools.product(*(axes[n] for n in names)):
                merged = self.baseline.copy()
                merged.update(zip(names, combo, strict=True))
                if merged not in configs:
                    configs.append(merged)
        else:
            # One at a time, in the file's impact order, so a truncated budget drops the least
            # impactful parameter's variants first.
            for name, values in axes.items():
                for value in values[1:]:
                    merged = self.baseline.copy()
                    merged[name] = value
                    if merged not in configs:
                        configs.append(merged)

        return configs[:max_configs]

    def generate_random(self, n_configs: int = 10, seed: int = 42) -> list[dict[str, Any]]:
        """Generate random search configurations centered around baseline."""
        rng = random.Random(seed)
        configs: list[dict[str, Any]] = []

        # Always include baseline first
        if self.baseline:
            configs.append(self.baseline.copy())

        for _ in range(n_configs - 1):
            config = self.baseline.copy()
            for spec in self.parameters:
                config[spec.name] = _random_value(spec, rng)
            configs.append(config)

        return configs

    def generate_staged(self, budget: int = 20, seed: int = 42) -> list[list[dict[str, Any]]]:
        """Generate staged narrowing configurations.

        Stage 1: Broad random exploration (60% of budget)
        Stage 2: Focused grid around best from stage 1 (40% of budget)

        Returns list of stages, each containing configs.
        """
        stage1_budget = max(3, int(budget * 0.6))

        stage1 = self.generate_random(n_configs=stage1_budget, seed=seed)
        # Stage 2 configs will be generated after stage 1 evaluation
        # using narrow_around()
        return [stage1, []]  # Stage 2 placeholder

    def narrow_around(self, best_params: dict[str, Any], n_configs: int = 5, seed: int = 42) -> list[dict[str, Any]]:
        """Generate configs narrowed around best-performing parameters.

        Used for stage 2 of staged narrowing search. *best_params* itself is not among them: it has
        been run already, and listing it first spent one full tool run of the stage-2 budget on a
        repeat and gave the robust selector a second draw of one configuration
        (hunt 2026-09-30, u32-tuning-22).
        """
        rng = random.Random(seed)
        configs: list[dict[str, Any]] = []

        # Bounded: a space of few discrete values can run out of distinct neighbours.
        for _ in range(max(0, n_configs) * 10):
            if len(configs) >= n_configs:
                break
            config = best_params.copy()
            for spec in self.parameters:
                if spec.name not in best_params:
                    continue
                base_val = best_params[spec.name]
                config[spec.name] = _perturb_value(spec, base_val, rng, factor=0.2)
            if config != best_params and config not in configs:
                configs.append(config)

        return configs


def _grid_axis(spec: ParameterSpec, base: Any) -> list[Any]:
    """The values a grid tries for *spec*: the baseline first, then its neighbours.

    Numeric neighbours are half and double the baseline, clipped into the declared range. A
    baseline outside the range -- or zero, which has no multiples -- is a mode rather than a point
    in it (STAGATE's ``rad_cutoff=0`` derives the radius from the spot spacing), so it is kept and
    the neighbours are two interior points of the range instead.
    """
    if spec.choices is not None:
        return [base, *(c for c in spec.choices if c != base)]
    if spec.param_type == ParameterType.BOOL:
        return [base, not base] if isinstance(base, bool) else [base, True, False]
    if spec.param_type not in (ParameterType.INT, ParameterType.FLOAT) or not isinstance(base, (int, float)):
        return [base]

    lo, hi = spec.min_value, spec.max_value
    inside = (lo is None or base >= lo) and (hi is None or base <= hi)
    if inside and base != 0:
        neighbours = [base * 0.5, base * 2.0]
    elif lo is not None and hi is not None:
        neighbours = [lo + (hi - lo) / 3.0, lo + 2.0 * (hi - lo) / 3.0]
    else:
        return [base]

    axis: list[Any] = [base]
    for value in neighbours:
        if lo is not None:
            value = max(lo, value)
        if hi is not None:
            value = min(hi, value)
        value = int(round(value)) if spec.param_type == ParameterType.INT else round(float(value), 6)
        if value not in axis:
            axis.append(value)
    return axis


def _random_value(spec: ParameterSpec, rng: random.Random) -> Any:
    """Generate a random value within a parameter's search space."""
    if spec.choices is not None:
        return rng.choice(spec.choices) if spec.choices else spec.default

    if spec.param_type == ParameterType.BOOL:
        return rng.choice([True, False])

    if spec.param_type == ParameterType.INT:
        if spec.min_value is not None:
            lo = int(spec.min_value)
        elif spec.default is not None and spec.default != 0:
            lo = max(1, int(spec.default * 0.3))
        else:
            lo = 1
        if spec.max_value is not None:
            hi = int(spec.max_value)
        elif spec.default is not None and spec.default != 0:
            hi = int(spec.default * 3.0)
        else:
            hi = 10
        if lo >= hi:
            return spec.default
        return rng.randint(lo, hi)

    if spec.param_type == ParameterType.FLOAT:
        if spec.min_value is not None:
            lo = spec.min_value
        elif spec.default is not None and spec.default != 0:
            lo = spec.default * 0.3
        else:
            lo = 0.1
        if spec.max_value is not None:
            hi = spec.max_value
        elif spec.default is not None and spec.default != 0:
            hi = spec.default * 3.0
        else:
            hi = 10.0
        if lo >= hi:
            return spec.default
        return round(rng.uniform(lo, hi), 6)

    return spec.default


def _perturb_value(spec: ParameterSpec, base: Any, rng: random.Random, factor: float = 0.2) -> Any:
    """Perturb a value by a small factor around its current value, staying inside the declared range.

    A base outside the declared range is returned unperturbed. Two such bases are real: STAGATE's
    ``rad_cutoff=0`` ("derive it from the spot spacing", against a range of fixed radii from 50),
    where every perturbation came out below the declared minimum; and a portal default the range
    had not caught up with (cell2location's 30000 mapping epochs against a maximum of 10000), where
    the window ``[24000, 10000]`` was empty and ``randint`` raised out of the whole tuning run
    (hunt 2026-09-30, u32-tuning-15/16). An empty window after clipping also keeps the base.
    """
    import math

    if spec.choices is not None:
        return rng.choice(spec.choices) if spec.choices else spec.default

    if spec.param_type == ParameterType.BOOL:
        return rng.choice([True, False])

    if spec.param_type not in (ParameterType.INT, ParameterType.FLOAT) or not isinstance(base, (int, float)):
        return base
    if (spec.min_value is not None and base < spec.min_value) or (spec.max_value is not None and base > spec.max_value):
        return base

    if spec.param_type == ParameterType.INT:
        delta = max(1, int(abs(base) * factor))
        lo = max(spec.min_value if spec.min_value is not None else 1, base - delta)
        hi = min(spec.max_value if spec.max_value is not None else base * 5, base + delta)
        lo, hi = math.ceil(lo), math.floor(hi)
        return rng.randint(lo, hi) if lo <= hi else base

    delta = abs(base) * factor if base != 0 else 0.1
    lo = max(spec.min_value if spec.min_value is not None else 0.0, base - delta)
    hi = min(spec.max_value if spec.max_value is not None else base * 5, base + delta)
    return round(rng.uniform(lo, hi), 6) if lo <= hi else base


def build_search_space(
    tool_name: str,
    mode: TuningMode,
    baseline: dict[str, Any] | None = None,
) -> SearchSpace:
    """Build a search space for a tool and mode.

    Uses parameter_registry to get tunable params, and defaults module
    for baseline values.
    """
    from spatialomicsgym.tuning.defaults import (
        get_official_defaults,
        load_spatialomicsgym_defaults_from_mcp_config,
    )
    from spatialomicsgym.tuning.parameter_registry import get_light_params, get_tunable_params

    # Get baseline from official defaults, falling back to SpatialOmicsLab config
    if baseline is None:
        official = get_official_defaults(tool_name)
        spatialomicsgym = load_spatialomicsgym_defaults_from_mcp_config().get(tool_name, {})
        baseline = {}
        for param_name, pv in official.items():
            baseline[param_name] = pv.value
        # Fill gaps from SpatialOmicsLab config
        for param_name, value in spatialomicsgym.items():
            if param_name not in baseline:
                baseline[param_name] = value

    # Get parameters based on mode
    if mode in (TuningMode.BENCHMARK_LIGHT, TuningMode.ADAPTIVE):
        params = get_light_params(tool_name)
    elif mode == TuningMode.BENCHMARK_FULL:
        params = get_tunable_params(tool_name)
    else:
        params = []

    # A proxy score cannot rank a knob that caps the output it counts: the SVG proxy is, for a
    # table with no p-values, the table's length -- so a search over n_top_genes / top_k_genes /
    # n_svgs / n_top just maximised the cap (hunt 2026-09-30, u32-tuning-19). Against ground truth
    # the list length does matter, so the benchmark modes keep them.
    if mode not in BENCHMARK_MODES:
        params = [p for p in params if not p.output_cap]

    return SearchSpace(
        tool_name=tool_name,
        mode=mode,
        parameters=params,
        baseline=baseline,
    )
