"""Config persistence - save, load, and version tuning results.

Manages the lifecycle of tuning configurations:
- Save trial results with full provenance
- Load best configs for reuse
- Version configs with timestamps
- Support per-tool, per-task, and per-dataset configs
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from spatialomicsgym.tuning.core import TuningResult

logger = logging.getLogger(__name__)

#: Deployment override for where tuned configs are cached, the way ``SOG_MEMORY_PATH`` overrides the
#: memory store. Replaces the default only -- an explicit ``base_dir=`` still wins.
TUNING_DIR_ENV = "SOG_TUNING_DIR"


def default_tuning_dir(package_file: Path | None = None) -> Path:
    """Where tuned configs live when the caller names no directory.

    Three levels up from this module is the ``agent/`` part of a source checkout (one more is the repo
    root), and this project's recorded tuning results are under ``agent/benchmarks/results/tuning``
    -- so a checkout keeps reading and writing beside the benchmarks tree. Three levels up from an *installed* copy is
    ``site-packages``, and the same expression quietly made the default
    ``site-packages/benchmarks/results/tuning``: ``save_tuning_result`` creates that directory and
    writes into the package tree, which is a PermissionError on a root-owned install, is deleted by
    the next ``pip install --upgrade``, and is a path no fresh install ever ships, so the read side
    could never find a cached config either. The agent tells users to run ``tune()``
    ("pre-compute with tune() for better results"), so this is on a path we recommend.

    A ``pyproject.toml`` one level above the derived ``agent/`` directory is the marker for "a source
    tree"; an editable install resolves through it too. Anywhere else the cache belongs in the user's
    home -- absolute, so it does not follow the working directory the way the memory store used to (R80/AK).
    """
    agent = (package_file or Path(__file__)).parent.parent.parent
    if (agent.parent / "pyproject.toml").is_file():
        return agent / "benchmarks" / "results" / "tuning"
    return Path.home() / ".spatialomicsgym" / "tuning"


DEFAULT_TUNING_DIR = default_tuning_dir()


def _write_yaml_atomically(path: Path, data: Any) -> None:
    """Serialise to a sibling temp file, then rename it over the target.

    Both callers overwrite a file that already holds something worth keeping: the rolling
    ``best_config.yaml``, and ``tuning_log.yaml``, which is read whole, appended to and rewritten. A
    plain ``open(path, "w")`` truncates first, so a failure inside ``yaml.dump`` -- a full disk, a
    value the dumper cannot represent -- left the previous contents gone and nothing in their place.
    Renaming a finished file over the old one cannot lose it.

    The temp name carries the pid, because a fixed one is not atomicity when two runs tune two tools
    into the same directory at once (R79/N).
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def _exclusive(path: Path):
    """Hold an exclusive lock on *path*'s directory for a read-modify-write of *path*.

    ``_write_yaml_atomically`` keeps a file whole, not current: two runs that both read N log entries
    each wrote N+1, and two runs that both passed the best-config check each wrote theirs, the later
    -- possibly worse -- one winning (hunt 2026-09-30, u32-tuning-23). The directory is what is
    locked, so no lock file is left among the results. Advisory, via ``fcntl``; where that does not
    exist (Windows) the sequence runs unlocked, as it always did.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - not POSIX
        yield
        return
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def _scoring_family(mode: str | None) -> str:
    """``"benchmark"`` for a score measured against ground truth, ``"proxy"`` for every other mode."""
    from spatialomicsgym.tuning.core import BENCHMARK_MODES

    return "benchmark" if mode in {m.value for m in BENCHMARK_MODES} else "proxy"


def _recorded_metric(result: TuningResult) -> str | None:
    """The metric name a result's score is actually in.

    ``TuningConfig.primary_metric`` names the task's benchmark metric whatever the mode, but only
    the benchmark modes compute it; every other mode is scored by a 0-1 proxy composite. Recording
    that as ``ARI`` or ``RMSE`` mislabelled the score, and let a proxy number be compared against a
    benchmark one (u32-tuning-9).
    """
    if _scoring_family(result.config.mode.value) == "benchmark":
        return result.config.primary_metric
    return f"proxy_{result.config.task_type}"


def _tuning_dir(base_dir: str | None = None) -> Path:
    """Where tuning results live, without touching the filesystem.

    Every reader in this module used to resolve through :func:`get_tuning_dir`, which creates. So
    asking for a config that was never saved created the directory it was looking in -- and
    ``DEFAULT_TUNING_DIR`` is ``benchmarks/results/tuning``, inside the user's recorded-experiment
    tree, which a fresh clone does not have at all. A read answers "nothing there" whether or not the
    directory exists; it has no business making one.
    """
    if base_dir:
        return Path(base_dir)
    override = os.environ.get(TUNING_DIR_ENV, "").strip()
    return Path(override) if override else DEFAULT_TUNING_DIR


def get_tuning_dir(base_dir: str | None = None) -> Path:
    """Get or create the tuning results directory. For writers -- readers use :func:`_tuning_dir`."""
    d = _tuning_dir(base_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_tuning_result(
    result: TuningResult,
    base_dir: str | None = None,
) -> Path:
    """Save a complete tuning result to YAML.

    Saves to: {base_dir}/{tool_name}/tuning_result_{timestamp}.yaml
    Also updates best_config.yaml if this is an improvement.

    Returns path to saved file.
    """
    tuning_dir = get_tuning_dir(base_dir)
    tool_dir = tuning_dir / result.config.tool_name
    tool_dir.mkdir(parents=True, exist_ok=True)

    # Save full result
    timestamp = result.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = tool_dir / f"tuning_result_{timestamp}.yaml"

    result_dict = result.to_dict()
    with open(result_path, "w") as f:
        yaml.dump(result_dict, f, default_flow_style=False, sort_keys=False)

    logger.info("Saved tuning result: %s", result_path)

    # Update best_config.yaml if this is an improvement
    best_path = tool_dir / "best_config.yaml"
    with _exclusive(best_path):
        _update_best_config(best_path, result, timestamp)

    # Append to tuning log
    _append_to_log(tuning_dir, result, timestamp)

    return result_path


def _update_best_config(best_path: Path, result: TuningResult, timestamp: str) -> None:
    """Replace ``best_config.yaml`` with *result* unless the entry there is a comparable, better score."""
    should_update = True
    metric = _recorded_metric(result)

    if best_path.exists():
        try:
            with open(best_path) as f:
                existing = yaml.safe_load(f)
            # Only an existing *score* can veto this result. A file that holds no usable one --
            # empty, comment-only, a list, a score that is not a number -- records nothing to beat,
            # and the default 0.0 it used to stand in for was a real score in the comparison: for a
            # lower_is_better task no RMSE is below 0.0, so one empty best_config.yaml silently
            # discarded every successful run for that tool from then on. The ``except`` below is no
            # help -- an empty file parses to None without raising.
            existing_score = existing.get("score") if isinstance(existing, dict) else None
            # Nor can a score on another scale. A proxy composite and a benchmark metric -- or two
            # benchmark metrics -- are not comparable: an adaptive 0.62 vetoed every later
            # ground-truth ARI below it, and an RMSE of 0.15 "beat" a proxy 0.62 by being smaller
            # (hunt 2026-09-30, u32-tuning-9). The family is read off the stored ``mode``, which
            # entries written before the metric was labelled honestly also carry.
            comparable = isinstance(existing, dict) and (
                _scoring_family(existing.get("mode")) == _scoring_family(result.config.mode.value)
                and (_scoring_family(existing.get("mode")) == "proxy" or existing.get("metric") == metric)
            )
            if isinstance(existing_score, bool) or not isinstance(existing_score, (int, float)) or not comparable:
                should_update = True
            elif result.config.lower_is_better:
                should_update = result.best_score < existing_score
            else:
                should_update = result.best_score > existing_score
        except Exception:
            should_update = True

    if should_update and result.best_params:
        best_config = {
            "tool_name": result.config.tool_name,
            "task_type": result.config.task_type,
            "params": result.best_params,
            "score": result.best_score,
            "metric": metric,
            # Which dataset the score was measured on. The per-run file written beside this one has
            # carried it all along through ``config.to_dict()``; this file -- the only one anything
            # ever reads back -- dropped it, and ``load_best_config`` is keyed on the tool name
            # alone. So a score from one dataset was returned for any other, with nothing on disk
            # able to say so. See ``describe_cached_score``.
            "dataset": result.config.dataset_path,
            "mode": result.mode_used.value,
            "timestamp": timestamp,
            "baseline_score": result.baseline_score,
            "improvement": result.improvement,
            # The strategy that ran, not the one requested. Recording the request meant an override
            # that was ignored -- or a "bayesian" run that fell back for want of optuna -- was
            # written into the provenance as though it had been honoured.
            "strategy": result.strategy_used or result.config.strategy_override or "auto",
        }
        _write_yaml_atomically(best_path, best_config)
        logger.info("Updated best config: %s", best_path)


def load_best_config(
    tool_name: str,
    base_dir: str | None = None,
) -> dict[str, Any] | None:
    """Load the best tuned configuration for a tool.

    Returns dict with params, score, metadata, or None if not found -- and "not found" covers a file
    that parses to something other than a mapping, the way :func:`load_tuning_log` has always
    required a list. Callers subscript what they get (``best["params"]``), so handing back a list or
    a bare string turned a lookup whose contract is to return ``None`` into an ``AttributeError``
    raised out of ``export_config_for_mcp``.

    The same applies one level down. Every reader of this cache treats ``params`` as a mapping --
    two iterate it (``dict.update``), three subscript it -- so a ``params`` key that is present and
    holds something else makes the entry unusable, which is what ``None`` already means here. The
    guard the three careful readers write, ``if cached and cached.get("params")``, is a truthiness
    test: it stops ``params: {}`` and ``params:`` left empty, and a list, a string or a number goes
    straight past it. ``params:`` left empty is also the one case the ``.get("params", {})``
    spelling looks like it handles and does not -- that default applies when the key is *absent*.

    An absent ``params`` key is a different case and is deliberately left alone: the ``{}`` default
    does fire for it, and the readers already agree on what it means.
    """
    tuning_dir = _tuning_dir(base_dir)
    best_path = tuning_dir / tool_name / "best_config.yaml"

    if not best_path.exists():
        return None

    try:
        with open(best_path) as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as e:
        logger.warning("Failed to load best config for %s: %s", tool_name, e)
        return None

    if not isinstance(data, dict):
        if data is not None:
            logger.warning("Ignoring best config for %s: expected a mapping, got %s", tool_name, type(data).__name__)
        return None

    if "params" in data and not isinstance(data["params"], dict):
        logger.warning(
            "Ignoring best config for %s: its params key holds %s, not a mapping of parameters",
            tool_name,
            type(data["params"]).__name__,
        )
        return None
    return data


def describe_cached_score(cached: dict[str, Any], dataset_path: str | None) -> str:
    """How to state a cached tuning score, given the dataset it is about to be applied to.

    :func:`load_best_config` is keyed on the tool name, so the config it hands back may have been
    tuned on entirely different data. The score is still the best evidence there is for those
    parameters -- but it is evidence about *that* run. Stated bare, as the reason a model should use
    these values on the data in front of it, it asserts a measurement nobody made.

    Only the file name of the recorded dataset is used. A cached config can carry an absolute path
    from another machine or another user's tree, and a prompt is not the place for it.
    """
    score = cached.get("score", "?")
    recorded = cached.get("dataset")
    if not recorded:
        return f"score={score} from an earlier run, dataset not recorded"

    name = Path(recorded).name or str(recorded)
    if dataset_path is None:
        return f"score={score} measured on {name}"
    try:
        same = Path(recorded).resolve() == Path(dataset_path).resolve()
    except OSError:  # an unresolvable path is not evidence that the two match
        same = False
    if same:
        return f"score={score} on this dataset"
    return f"score={score} measured on {name}, not this dataset"


def load_tuning_history(
    tool_name: str,
    base_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Load all tuning results for a tool, sorted by timestamp."""
    tuning_dir = _tuning_dir(base_dir)
    tool_dir = tuning_dir / tool_name

    if not tool_dir.exists():
        return []

    results = []
    for path in sorted(tool_dir.glob("tuning_result_*.yaml")):
        try:
            with open(path) as f:
                result = yaml.safe_load(f)
            if result:
                results.append(result)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)

    return results


def load_tuning_log(base_dir: str | None = None) -> list[dict[str, Any]]:
    """Load the global tuning log."""
    tuning_dir = _tuning_dir(base_dir)
    log_path = tuning_dir / "tuning_log.yaml"

    if not log_path.exists():
        return []

    try:
        with open(log_path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, list) else []
    except (yaml.YAMLError, OSError) as e:
        logger.warning("Failed to load tuning log: %s", e)
        return []


def _append_to_log(tuning_dir: Path, result: TuningResult, timestamp: str) -> None:
    """Append a summary entry to the global tuning log."""
    log_path = tuning_dir / "tuning_log.yaml"
    with _exclusive(log_path):
        _append_locked(log_path, result, timestamp)


def _append_locked(log_path: Path, result: TuningResult, timestamp: str) -> None:
    """The read-append-rewrite of the log, run under its lock."""
    entries = []
    if log_path.exists():
        try:
            with open(log_path) as f:
                existing = yaml.safe_load(f)
            if isinstance(existing, list):
                entries = existing
        except (yaml.YAMLError, OSError):
            entries = []

    entry = {
        "timestamp": timestamp,
        "tool_name": result.config.tool_name,
        "task_type": result.config.task_type,
        "mode": result.mode_used.value,
        "mode_reason": result.mode_reason,
        "best_score": result.best_score,
        "baseline_score": result.baseline_score,
        "improvement": result.improvement,
        "n_trials": len(result.all_trials),
        "n_successful": sum(1 for t in result.all_trials if t.status == "success"),
        # The strategy that ran, not the one requested. Recording the request meant an override that
        # was ignored -- or a "bayesian" run that fell back for want of optuna -- was written into
        # the provenance as though it had been honoured.
        "strategy": result.strategy_used or result.config.strategy_override or "auto",
        "elapsed_seconds": result.elapsed_seconds,
        "warnings": result.warnings,
    }
    entries.append(entry)

    _write_yaml_atomically(log_path, entries)


def export_config_for_mcp(
    tool_name: str,
    base_dir: str | None = None,
) -> dict[str, Any]:
    """Export tuned parameters in a format compatible with MCP tool invocation.

    Returns dict of {param_name: value} ready to pass to an MCP tool.
    """
    best = load_best_config(tool_name, base_dir)
    if best is None:
        return {}
    return best.get("params", {})


def clear_tool_results(tool_name: str, base_dir: str | None = None) -> int:
    """Remove all tuning results for a tool. Returns count of entries removed.

    ``unlink`` on a subdirectory raises ``IsADirectoryError``, and nothing here caught it: one
    subdirectory -- a tool that wrote its trial artefacts into a folder, a stray ``.ipynb_checkpoints``
    -- aborted the clear part-way, so the caller got an exception instead of a count, the files after
    it in iteration order survived, and ``tool_dir.rmdir()`` was never reached.
    """
    tuning_dir = _tuning_dir(base_dir)
    tool_dir = tuning_dir / tool_name

    if not tool_dir.exists():
        return 0

    count = 0
    for path in tool_dir.iterdir():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        count += 1

    try:
        tool_dir.rmdir()
    except OSError:
        pass

    return count
