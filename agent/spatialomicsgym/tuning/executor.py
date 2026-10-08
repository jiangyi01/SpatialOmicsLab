"""Staged tuning executor - orchestrates the 11-step tuning pipeline.

Steps:
1. Load official defaults
2. Parameter discovery
3. Mode selection
4. Dataset readiness check
5. Dataset preparation (subset/mock if needed)
6. Search execution (light/full/adaptive)
7. Candidate filtering
8. Robustness check
9. Best config selection
10. Final validation (optional, full mode)
11. Result persistence
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from spatialomicsgym.tuning.core import (
    BENCHMARK_MODES,
    MODE_BUDGETS,
    TrialResult,
    TuningConfig,
    TuningMode,
    TuningResult,
)

logger = logging.getLogger(__name__)


class StagedTuningExecutor:
    """Orchestrates the full tuning pipeline."""

    def __init__(self, config: TuningConfig):
        self.config = config
        self._start_time = 0.0
        self._baseline_params: dict[str, Any] = {}
        self._baseline_score: float | None = None
        self._all_trials: list[TrialResult] = []
        self._warnings: list[str] = []
        # Name of the strategy that actually searched, filled in by step 6. Empty means no search
        # ran at all, which is a different fact from "the caller asked for one".
        self._strategy_used: str = ""
        # Why the last ``_run_tool`` call did not succeed, for the trial that records it.
        self._last_run_error: str = ""

    def run(self) -> TuningResult:
        """Execute the full staged tuning pipeline."""
        self._start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(
            "Starting tuning: tool=%s, task=%s, mode=%s",
            self.config.tool_name,
            self.config.task_type,
            self.config.mode.value,
        )

        # Step 1: Load official defaults
        baseline = self._step1_load_defaults()

        # Step 2: Parameter discovery
        tunable_params = self._step2_discover_params()

        # Step 3: Mode selection (may override configured mode)
        mode, mode_reason = self._step3_select_mode(tunable_params)

        # Step 4: Dataset readiness check
        data_ready, data_msg = self._step4_check_dataset()
        if not data_ready and mode != TuningMode.DEFAULT_FALLBACK:
            mode = TuningMode.DEFAULT_FALLBACK
            mode_reason = f"Downgraded to fallback: {data_msg}"
            self._warnings.append(mode_reason)

        # Short-circuit for default_fallback
        if mode == TuningMode.DEFAULT_FALLBACK:
            return self._build_fallback_result(baseline, mode_reason, timestamp)

        # Step 5: Prepare dataset (subset for light mode)
        working_dataset = self._step5_prepare_dataset()

        # Step 6: Execute search
        trials = self._step6_execute_search(baseline, tunable_params, mode, working_dataset)
        self._all_trials.extend(trials)

        # Step 7: Filter candidates
        successful_trials = self._step7_filter_candidates(trials)

        if not successful_trials:
            # The trials and the reason they failed are the result of this run. Both used to be
            # dropped here -- ``all_trials=[]``, one generic warning -- so a search that never got a
            # single command past argparse read the same as one that never started
            # (hunt 2026-09-30, u32-tuning-10).
            reasons = sorted({t.error_message for t in trials if t.error_message})
            reason = "All trial configurations failed"
            if len(reasons) == 1:
                reason += f": {reasons[0]}"
            self._warnings.append("All trials failed, falling back to baseline")
            return self._build_fallback_result(baseline, reason, timestamp, searched=True)

        # Step 9: Select best config (moved before robustness check so we compare actual best)
        from spatialomicsgym.tuning.robustness import select_robust_config

        best_params, best_score = select_robust_config(trials, self.config)

        # Step 8: Robustness check (uses actual best_score, not first trial)
        from spatialomicsgym.tuning.robustness import validate_tuning_result

        robustness_warnings = validate_tuning_result(
            best_score,
            self._baseline_score,
            trials,
            self.config,
        )
        self._warnings.extend(robustness_warnings)

        # Step 10: Optional full validation (benchmark_tuning.full only)
        if mode == TuningMode.BENCHMARK_FULL and self.config.dataset_path:
            # Re-run best config on full dataset if we used a subset
            pass  # Full dataset was already used in full mode

        # Step 11: Persist results
        improvement = None
        if self._baseline_score is not None:
            if self.config.lower_is_better:
                improvement = self._baseline_score - best_score
            else:
                improvement = best_score - self._baseline_score

        result = TuningResult(
            config=self.config,
            best_params=best_params,
            best_score=best_score,
            all_trials=self._all_trials,
            baseline_params=self._baseline_params,
            baseline_score=self._baseline_score,
            improvement=improvement,
            mode_used=mode,
            mode_reason=mode_reason,
            strategy_used=self._strategy_used,
            provenance=self._build_provenance(baseline),
            timestamp=timestamp,
            elapsed_seconds=time.time() - self._start_time,
            warnings=self._warnings,
        )

        from spatialomicsgym.tuning.persistence import save_tuning_result

        save_tuning_result(result, base_dir=self.config.output_dir)

        logger.info(
            "Tuning complete: best_score=%.4f, baseline=%.4f, improvement=%.4f, elapsed=%.1fs",
            best_score,
            self._baseline_score or 0.0,
            improvement or 0.0,
            result.elapsed_seconds,
        )

        return result

    def _step1_load_defaults(self) -> dict[str, Any]:
        """Step 1: Load official and SpatialOmicsLab defaults."""
        from spatialomicsgym.tuning.defaults import (
            get_official_defaults,
            load_spatialomicsgym_defaults_from_mcp_config,
        )

        baseline: dict[str, Any] = {}

        # Start with SpatialOmicsLab defaults from mcp_config
        spatialomicsgym_defaults = load_spatialomicsgym_defaults_from_mcp_config()
        if self.config.tool_name in spatialomicsgym_defaults:
            baseline.update(spatialomicsgym_defaults[self.config.tool_name])

        # Override with official defaults where available
        official = get_official_defaults(self.config.tool_name)
        for param_name, pv in official.items():
            baseline[param_name] = pv.value

        # A fixed value belongs to the dataset, not to any configuration: it is passed to every
        # command in ``_run_tool`` and kept out of what the search varies and reports (u32-tuning-5).
        # So is a parameter a fixed value leaves without effect: reported, it reads as a measured
        # choice that no trial made (hunt 2026-09-30, u32-tuning-5 follow-up).
        for name in [*(self.config.fixed_params or {}), *self._inert_params()]:
            baseline.pop(name, None)

        self._baseline_params = baseline
        logger.info("Loaded %d baseline parameters for %s", len(baseline), self.config.tool_name)
        return baseline

    def _inert_params(self) -> set[str]:
        """The parameters a fixed value leaves without effect (``inert_when_fixed`` in the search space).

        scanpy's ``resolution`` once ``target_n_clusters`` is fixed: the worker then searches the
        resolution itself, so two of the seven light trials repeated the baseline computation and
        the reported "best" resolution was never measured. A fixed 0 or empty value switches nothing
        off -- ``target_n_clusters=0`` is how the worker is told to use ``resolution``.
        """
        from spatialomicsgym.tuning.parameter_registry import get_all_params

        fixed = self.config.fixed_params or {}
        return {
            spec.name
            for spec in get_all_params(self.config.tool_name)
            if any(fixed.get(name) not in (None, 0, "") for name in spec.inert_when_fixed or ())
        }

    def _step2_discover_params(self) -> list:
        """Step 2: Discover tunable parameters."""
        from spatialomicsgym.tuning.parameter_registry import get_all_params, get_tunable_params

        all_params = get_all_params(self.config.tool_name)
        tunable = get_tunable_params(self.config.tool_name)
        logger.info(
            "Discovered %d params (%d tunable) for %s",
            len(all_params),
            len(tunable),
            self.config.tool_name,
        )
        return tunable

    def _step3_select_mode(self, tunable_params: list) -> tuple[TuningMode, str]:
        """Step 3: Validate and potentially adjust tuning mode."""
        mode = self.config.mode

        if not tunable_params:
            return TuningMode.DEFAULT_FALLBACK, "No tunable parameters found"

        if mode == TuningMode.DEFAULT_FALLBACK:
            return mode, self.config.skip_reason or "Explicitly set to fallback"

        # Check if dataset path is needed but missing
        if mode in (TuningMode.BENCHMARK_LIGHT, TuningMode.BENCHMARK_FULL):
            if not self.config.dataset_path:
                return TuningMode.DEFAULT_FALLBACK, "No dataset path for benchmark tuning"

        if mode == TuningMode.ADAPTIVE:
            if not self.config.dataset_path:
                return TuningMode.DEFAULT_FALLBACK, "No dataset path for adaptive tuning"
            # Check if we should downgrade
            from spatialomicsgym.tuning.adaptive import profile_dataset, should_downgrade_to_fallback

            profile = profile_dataset(self.config.dataset_path)
            should_downgrade, reason = should_downgrade_to_fallback(profile, self.config.task_type)
            if should_downgrade:
                return TuningMode.DEFAULT_FALLBACK, reason

        return mode, f"Using requested mode: {mode.value}"

    def _step4_check_dataset(self) -> tuple[bool, str]:
        """Step 4: Check dataset readiness."""
        if self.config.mode == TuningMode.DEFAULT_FALLBACK:
            return True, "Fallback mode, no dataset needed"

        if not self.config.dataset_path:
            return False, "No dataset path provided"

        path = Path(self.config.dataset_path)
        if not path.exists():
            return False, f"Dataset not found: {path}"

        if path.stat().st_size < 1024:
            return False, f"Dataset too small: {path.stat().st_size} bytes"

        # A requested ground-truth column the dataset does not have is not scored as some other
        # column: the evaluator would fall back through its candidates and the search would optimise
        # a target nobody asked for (u32-tuning-4). Unreadable obs is no evidence either way.
        key = self.config.ground_truth_key
        if key and self.config.mode in BENCHMARK_MODES and self.config.task_type != "svg_detection":
            from spatialomicsgym.tuning.objectives import _input_obs_columns

            columns = _input_obs_columns(str(path))
            if columns and key not in columns:
                return False, f"Ground-truth column {key!r} is not in the dataset's obs"

        return True, "Dataset ready"

    def _step5_prepare_dataset(self) -> str | None:
        """Step 5: Prepare dataset (subset for light mode)."""
        if not self.config.dataset_path:
            return None

        if self.config.mode == TuningMode.BENCHMARK_LIGHT:
            # Light mode runs on the full dataset -- subsetting spatial data breaks its spatial
            # structure -- and saves its time by searching fewer parameters with a smaller budget.
            # Nothing here reduces a trial's training budget; this used to say it did (u32-tuning-12).
            logger.info("Light mode: full dataset, the light parameters only")
            return self.config.dataset_path

        return self.config.dataset_path

    def _step6_execute_search(
        self,
        baseline: dict[str, Any],
        tunable_params: list,
        mode: TuningMode,
        dataset_path: str | None,
    ) -> list[TrialResult]:
        """Step 6: Execute the search strategy."""
        from spatialomicsgym.tuning.search_space import build_search_space
        from spatialomicsgym.tuning.strategies import SearchStrategy, StagedNarrowing, select_strategy

        budget = self.config.budget or MODE_BUDGETS[mode]
        space = build_search_space(self.config.tool_name, mode, baseline)
        left_out = {*(self.config.fixed_params or {}), *self._inert_params()}
        space.parameters = [spec for spec in space.parameters if spec.name not in left_out]
        strategy = select_strategy(mode, self.config.task_type, self.config.strategy_override)
        self._strategy_used = strategy.name()

        logger.info(
            "Executing %s with budget=%d on %s",
            strategy.name(),
            budget,
            self.config.tool_name,
        )

        def evaluate_fn(params: dict[str, Any]) -> TrialResult:
            return self._evaluate_single(params, dataset_path)

        # Sequential strategies (Bayesian) override optimize() for per-trial feedback.
        # Batch strategies inherit the default optimize() from SearchStrategy.
        has_custom_optimize = type(strategy).optimize is not SearchStrategy.optimize

        if has_custom_optimize:
            trials = strategy.optimize(space, budget, evaluate_fn, self.config.seed)
            if trials:
                self._baseline_score = trials[0].score if trials[0].status == "success" else None
            return trials

        # --- Batch strategy: existing flow (unchanged) ---

        # Generate candidates
        candidates = strategy.generate_candidates(space, budget, self.config.seed)

        # If adaptive mode, add heuristic candidates
        if mode == TuningMode.ADAPTIVE and dataset_path:
            from spatialomicsgym.tuning.adaptive import apply_adaptive_rules, profile_dataset

            profile = profile_dataset(dataset_path)
            adaptive_candidates = apply_adaptive_rules(self.config.task_type, baseline, profile)
            # Merge: keep first (baseline), add adaptive, then search candidates
            seen = [str(c) for c in candidates[:1]]
            for ac in adaptive_candidates:
                if str(ac) not in seen:
                    candidates.insert(1, ac)
                    seen.append(str(ac))

        # Evaluate each candidate
        trials: list[TrialResult] = []
        for i, params in enumerate(candidates[:budget]):
            logger.info("Trial %d/%d: %s", i + 1, budget, params)
            trial = self._evaluate_single(params, dataset_path)
            trial.params = params
            trial.seed = self.config.seed
            trials.append(trial)

            # Track baseline score (first trial is always baseline)
            if i == 0:
                self._baseline_score = trial.score if trial.status == "success" else None

        # Stage 2 for staged narrowing
        if isinstance(strategy, StagedNarrowing) and trials:
            successful = [t for t in trials if t.status == "success"]
            if successful:
                best_so_far = (
                    min(successful, key=lambda t: t.score)
                    if self.config.lower_is_better
                    else max(successful, key=lambda t: t.score)
                )
                remaining = budget - len(trials)
                if remaining > 0:
                    # A configuration stage 1 already ran is not run again: it costs a full tool run
                    # and hands select_robust_config a second draw of the same point (u32-tuning-22).
                    tried = [t.params for t in trials]
                    stage2_configs = [
                        c
                        for c in strategy.refine(space, best_so_far.params, remaining, self.config.seed)
                        if c not in tried
                    ]
                    for j, params in enumerate(stage2_configs):
                        logger.info("Stage 2 trial %d/%d: %s", j + 1, len(stage2_configs), params)
                        trial = self._evaluate_single(params, dataset_path)
                        trial.params = params
                        trial.seed = self.config.seed
                        trials.append(trial)

        return trials

    def _evaluate_single(self, params: dict[str, Any], dataset_path: str | None) -> TrialResult:
        """Evaluate a single parameter configuration.

        Runs the tool with given params and evaluates the output.
        """
        start = time.time()

        # Create temp output dir for this trial
        trial_dir = tempfile.mkdtemp(prefix=f"tuning_{self.config.tool_name}_")

        try:
            # Run the MCP tool with these parameters
            self._last_run_error = ""
            success = self._run_tool(params, dataset_path, trial_dir)

            if not success:
                return TrialResult(
                    params=params,
                    score=float("inf") if self.config.lower_is_better else 0.0,
                    status="failed",
                    error_message=self._last_run_error or "Tool execution failed",
                    elapsed_seconds=time.time() - start,
                )

            # Evaluate output
            has_gt = self.config.mode in (
                TuningMode.BENCHMARK_LIGHT,
                TuningMode.BENCHMARK_FULL,
            )

            if has_gt and dataset_path:
                from spatialomicsgym.tuning.objectives import evaluate_benchmark_trial

                trial = evaluate_benchmark_trial(
                    task_type=self.config.task_type,
                    output_dir=trial_dir,
                    ground_truth_path=dataset_path,
                    # No key named is "" -- auto-detect, and the evaluator records the column it
                    # used. The hardcoded ``cell_type`` this used to send is a *named* key, which
                    # the evaluator now refuses wherever the dataset lacks it, so every trial of
                    # such a dataset failed (hunt 2026-09-30, u33b-bench-scoring-4).
                    ground_truth_key=self.config.ground_truth_key or "",
                    svg_ground_truth_path=self.config.svg_ground_truth_path,
                    tool_name=self.config.tool_name,
                )
            else:
                from spatialomicsgym.tuning.objectives import evaluate_proxy_trial

                trial = evaluate_proxy_trial(
                    task_type=self.config.task_type,
                    output_dir=trial_dir,
                    dataset_path=dataset_path,
                    seed=self.config.seed,
                    tool_name=self.config.tool_name,
                )

            trial.params = params
            trial.elapsed_seconds = time.time() - start
            return trial

        except Exception as e:
            logger.warning("Trial failed: %s", e)
            return TrialResult(
                params=params,
                score=float("inf") if self.config.lower_is_better else 0.0,
                status="crashed",
                error_message=str(e),
                elapsed_seconds=time.time() - start,
            )
        finally:
            # Clean up trial dir
            try:
                shutil.rmtree(trial_dir, ignore_errors=True)
            except Exception:
                pass

    def _run_tool(
        self,
        params: dict[str, Any],
        dataset_path: str | None,
        output_dir: str,
    ) -> bool:
        """Run the MCP tool with given parameters.

        Uses subprocess to invoke the tool's worker script directly,
        bypassing the full agent pipeline for efficiency.
        """
        from spatialomicsgym.tuning.integration import explain_tool_command

        try:
            cmd, why = explain_tool_command(
                tool_name=self.config.tool_name,
                params={**params, **(self.config.fixed_params or {})},
                dataset_path=dataset_path,
                output_dir=output_dir,
                sc_reference_path=self.config.sc_reference_path,
            )

            if not cmd:
                logger.warning("Could not build command for %s", self.config.tool_name)
                self._last_run_error = f"No command was built: {why}" if why else "No command was built"
                return False

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_per_run,
            )

            if result.returncode != 0:
                # The tail, not the head: argparse prints its usage banner first and the line that
                # says what was wrong last, so the first 500 characters cut exactly that line off
                # (u32-tuning-10).
                tail = (result.stderr or "")[-1500:]
                logger.warning("Tool %s failed (rc=%d): %s", self.config.tool_name, result.returncode, tail)
                self._last_run_error = f"Worker exited {result.returncode}: {tail.strip()[-500:]}"
                return False

            # Exit 0 is not success for a worker that reports its failure in its JSON line and returns
            # normally (scanpy_spatial does). The portal reads that line; a trial must too, or the
            # worker's own message is lost behind "No output h5ad found" (u32-tuning-20).
            reported = _worker_error(result.stdout)
            if reported:
                logger.warning("Tool %s reported an error: %s", self.config.tool_name, reported)
                self._last_run_error = f"Worker reported an error: {reported}"
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.warning("Tool %s timed out after %ds", self.config.tool_name, self.config.timeout_per_run)
            self._last_run_error = (
                f"Timed out after {self.config.timeout_per_run}s (raise timeout_per_run to give each trial longer)"
            )
            return False
        except Exception as e:
            logger.warning("Tool execution error: %s", e)
            self._last_run_error = f"Tool execution error: {e}"
            return False

    def _step7_filter_candidates(self, trials: list[TrialResult]) -> list[TrialResult]:
        """Step 7: Filter out failed/crashed candidates."""
        successful = [t for t in trials if t.status == "success"]
        failed_count = len(trials) - len(successful)
        if failed_count > 0:
            logger.info("Filtered %d failed trials, %d remaining", failed_count, len(successful))
        return successful

    def _build_fallback_result(
        self, baseline: dict[str, Any], reason: str, timestamp: str, searched: bool = False
    ) -> TuningResult:
        """Build a TuningResult for default_fallback mode.

        *searched* is True when the fallback follows a search whose trials all failed: those trials
        and the warnings gathered on the way are kept, since they are the only record of why.
        """
        return TuningResult(
            config=self.config,
            best_params=baseline,
            best_score=0.0,
            all_trials=list(self._all_trials) if searched else [],
            baseline_params=baseline,
            baseline_score=None,
            improvement=None,
            mode_used=TuningMode.DEFAULT_FALLBACK,
            mode_reason=reason,
            # "none" where the fallback fired before step 6: the baseline is handed back unsearched,
            # so naming the requested strategy here would assert a search that never started. Where
            # the fallback fired *after* a search whose trials all failed, the real name is kept.
            strategy_used=self._strategy_used or "none",
            provenance=self._build_provenance(baseline),
            timestamp=timestamp,
            elapsed_seconds=time.time() - self._start_time,
            warnings=[*self._warnings, f"Fallback: {reason}"] if searched else [f"Fallback: {reason}"],
        )

    def _build_provenance(self, baseline: dict[str, Any]) -> dict[str, Any]:
        """Build provenance metadata for the tuning result."""
        from spatialomicsgym.tuning.defaults import get_official_defaults

        official = get_official_defaults(self.config.tool_name)
        provenance: dict[str, Any] = {
            "tool_name": self.config.tool_name,
            "official_defaults_available": len(official) > 0,
            "official_params": {name: pv.to_dict() for name, pv in official.items()},
            "spatialomicsgym_baseline": baseline,
        }
        return provenance


def _worker_error(stdout: str | None) -> str:
    """The error a worker reported in its JSON result line, or ``""`` when it reported none.

    Read the way the portal reads it (``base_mcp._parse_result``): the last line of stdout that
    parses as a JSON object. Any ``status`` other than ``"ok"`` is a failure -- ``"error"``, and
    ``"dep_missing"`` for an upstream dependency that is absent.
    """
    import json

    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and "status" in payload:
            if payload.get("status") == "ok":
                return ""
            return str(payload.get("error") or f"status {payload.get('status')!r}")
    return ""
