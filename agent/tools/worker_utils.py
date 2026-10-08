#!/usr/bin/env python3
"""
worker_utils.py - Standard output helpers for SpatialOmicsLab MCP workers.

Provides a consistent JSON output format for all workers:

    {
        "status": "ok" | "error",
        "tool": "<tool_name>",
        "task": "<task_name>",           # optional, for multi-task workers
        "data": {                         # input data dimensions
            "n_spots": ...,
            "n_genes": ...,
            ...
        },
        "output_files": {                 # all output file paths
            "annotated_h5ad": "...",
            "clusters_csv": "...",
            ...
        },
        "params": {                       # key parameters used
            ...
        },
        "summary": {                      # analysis results & metrics
            "n_clusters": ...,
            "cluster_sizes": {...},
            "top_genes": [...],
            ...
        },
        "analysis": "...",                # human-readable interpretation
        "error": "...",                   # only on error
        "traceback": "...",               # only on error
    }

Usage in a worker:

    from worker_utils import WorkerOutput

    out = WorkerOutput("my_tool", task="clustering")
    out.set_data(n_spots=1000, n_genes=2000)
    out.add_output_file("annotated_h5ad", "/path/to/out.h5ad")
    out.add_param("resolution", 0.8)
    out.set_summary(n_clusters=5, cluster_sizes={"0": 200, "1": 300, ...})
    out.set_analysis("Found 5 spatial domains. Domain 1 is the largest with 300 spots.")
    out.emit()  # prints JSON to stdout
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable

# Name of the run-provenance sidecar dropped next to a worker's outputs.
#
# Deliberately not a name any benchmark profile declares. The scoring path is narrow and was checked
# before choosing this: output_inspector collects only {.h5ad, .csv, .tsv} as prediction candidates,
# and output_standardizer parses a .json only when the registry names it explicitly (the sole such
# name is "predicted_genes.json"). No profile globs *.json. test/test_worker_provenance_sidecar.py
# pins all of that, so if someone adds a wildcard pattern later the test says so before scores move.
PROVENANCE_FILENAME = "sog_run_provenance.json"

# Cap on recorded package versions. A worker env can have a few hundred importable top-level
# modules; the point is reproducibility, not an inventory, and a sidecar should stay readable.
_MAX_RECORDED_PACKAGES = 200


def _provenance_enabled() -> bool:
    """Whether to write the sidecar. On unless explicitly switched off."""
    return (os.environ.get("SOG_WRITE_PROVENANCE") or "").strip().lower() not in {"0", "false", "no", "off"}


def _is_third_party(module: Any, stdlib_dir: str) -> bool:
    """Whether a loaded module is an installed dependency rather than part of the interpreter.

    ``sys.stdlib_module_names`` is the direct answer but only exists on 3.10+, and most per-tool
    envs are older (the env running scanpy_spatial is Python 3.8.8), so its first real sidecar
    listed ``json``, ``re``, ``csv`` and ``zlib`` among the dependencies. Location works on every
    version: built-ins have no ``__file__``, stdlib lives under the interpreter's stdlib directory,
    and site-packages sits *inside* that directory in a conda env -- hence the explicit re-admission
    of site-/dist-packages. Anything else (an editable install, a source checkout -- how several of
    the bio tools are installed) is kept, because dropping those would lose the versions that matter
    most.
    """
    path = getattr(module, "__file__", None)
    if not isinstance(path, str) or not path:
        return False
    if "site-packages" in path or "dist-packages" in path:
        return True
    return not (stdlib_dir and path.startswith(stdlib_dir))


def _loaded_package_versions() -> dict[str, str]:
    """Versions of the third-party packages this worker actually imported.

    Read from the live ``sys.modules`` of the process that produced the files, which is the only
    place the answer is truthful: a later inspection runs in a different environment. (A live probe
    asked the agent to write a methods section and it had to caveat exactly that — "these reflect my
    inspection environment, not necessarily the worker's runtime".)
    """
    stdlib_names = getattr(sys, "stdlib_module_names", frozenset())
    try:
        import sysconfig

        stdlib_dir = sysconfig.get_paths().get("stdlib") or ""
    except Exception:
        stdlib_dir = ""
    versions: dict[str, str] = {}
    # sorted() materialises the whole view before the loop body runs, which matters here: reading
    # __version__ can trigger a lazy import in some packages, and that would mutate sys.modules
    # underneath a live iterator.
    for name, module in sorted(sys.modules.items()):
        if "." in name or name.startswith("_") or name in stdlib_names:
            continue
        if not _is_third_party(module, stdlib_dir):
            continue
        try:
            # Some packages (click) deprecated __version__ and warn on access. Recording a version
            # must not add lines to a worker's stderr.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                version = getattr(module, "__version__", None)
        except Exception:
            continue
        if not isinstance(version, str):
            continue
        # Strip: PROST 1.1.2 really does ship ``__version__ = " 1.1.2 "``, and a provenance record
        # exists to be compared between runs, where " 1.1.2 " != "1.1.2". Drop whitespace-only
        # strings entirely rather than record an empty version.
        version = version.strip()
        if version:
            versions[name] = version
        if len(versions) >= _MAX_RECORDED_PACKAGES:
            break
    return versions


class WorkerOutput:
    """Builder for standardized worker JSON output."""

    def __init__(self, tool: str, task: str | None = None):
        self._tool = tool
        self._task = task
        self._data: dict[str, Any] = {}
        self._output_files: dict[str, Any] = {}
        self._params: dict[str, Any] = {}
        self._summary: dict[str, Any] = {}
        self._analysis: str | None = None
        self._extra: dict[str, Any] = {}

    def set_data(self, **kwargs: Any) -> WorkerOutput:
        """Set input data dimensions (n_spots, n_genes, etc.)."""
        self._data.update(kwargs)
        return self

    def add_output_file(self, key: str, path: Any) -> WorkerOutput:
        """Register an output file path (or list of paths)."""
        if path is None:
            self._output_files[key] = None
        elif isinstance(path, (list, tuple)):
            self._output_files[key] = [str(p) for p in path]
        else:
            self._output_files[key] = str(path)
        return self

    def add_output_files(self, files: dict[str, Any]) -> WorkerOutput:
        """Register multiple output file paths at once."""
        for k, v in files.items():
            self.add_output_file(k, v)
        return self

    def add_param(self, key: str, value: Any) -> WorkerOutput:
        """Record a parameter used in the analysis."""
        self._params[key] = value
        return self

    def add_params(self, params: dict[str, Any]) -> WorkerOutput:
        """Record multiple parameters at once."""
        self._params.update(params)
        return self

    def set_summary(self, **kwargs: Any) -> WorkerOutput:
        """Set analysis summary metrics (n_clusters, top_genes, etc.)."""
        self._summary.update(kwargs)
        return self

    def set_analysis(self, text: str) -> WorkerOutput:
        """Set a human-readable analysis/interpretation string."""
        self._analysis = text
        return self

    def add_extra(self, key: str, value: Any) -> WorkerOutput:
        """Add extra tool-specific data that doesn't fit other categories."""
        self._extra[key] = value
        return self

    def add_warning(self, message: str) -> WorkerOutput:
        """Append a non-fatal warning message to the output.

        Warnings accumulate in _extra['warnings'] as a list. Use for
        soft-fail conditions: missing optional deps, fallback code paths,
        unexpected-but-recoverable input shapes.
        """
        warnings = self._extra.setdefault("warnings", [])
        if isinstance(warnings, list):
            warnings.append(str(message))
        else:
            self._extra["warnings"] = [str(message)]
        return self

    def add_warnings(self, messages) -> WorkerOutput:
        """Bulk add; messages can be a single string or an iterable of strings."""
        if isinstance(messages, str):
            return self.add_warning(messages)
        for m in messages or ():
            self.add_warning(m)
        return self

    def add_info(self, message: str) -> WorkerOutput:
        """Append an informational message (accumulates in _extra['info']).

        Exists permanently for the same reason as set_meta/add_warning: STCoscientist-generated
        workers reach for an `add_info` API, and its absence surfaced as
        `'WorkerOutput' object has no attribute 'add_info'` at runtime. Kept in the canonical
        worker_utils.py so the gap never recurs (never auto-patched per-tool -- see
        tools_user/self_review._r_WORKER_OUTPUT_API).
        """
        info = self._extra.setdefault("info", [])
        if isinstance(info, list):
            info.append(str(message))
        else:
            self._extra["info"] = [str(message)]
        return self

    def add_note(self, message: str) -> WorkerOutput:
        """Append a note (accumulates in _extra['notes']). Present for the same reason as add_info."""
        notes = self._extra.setdefault("notes", [])
        if isinstance(notes, list):
            notes.append(str(message))
        else:
            self._extra["notes"] = [str(message)]
        return self

    def set_meta(self, meta_or_key=None, value: Any = None, **kwargs: Any) -> WorkerOutput:
        """Set tool-specific metadata (e.g. tool version, subcommand used).

        Three calling conventions — all supported, pick whichever is natural:
          out.set_meta({"version": "1.2.3", "device": "cpu"})   # dict form
          out.set_meta("version", "1.2.3")                      # key/value form
          out.set_meta(version="1.2.3", device="cpu")           # kwargs form

        Stored under `_extra["meta"]` so it round-trips through to_dict() and emit()
        without colliding with first-class fields (status, data, summary, etc.).

        This method exists specifically because STCoscientist-generated workers frequently
        reach for a `set_meta` API. Historically the absence of this method
        surfaced as `'WorkerOutput' object has no attribute 'set_meta'` at
        runtime and blocked real-data execution of otherwise-working tools.
        """
        if isinstance(self._extra.get("meta"), dict):
            meta = self._extra["meta"]
        else:
            meta = {}
        if isinstance(meta_or_key, dict):
            meta.update(meta_or_key)
        elif isinstance(meta_or_key, str):
            meta[meta_or_key] = value
        meta.update(kwargs)
        self._extra["meta"] = meta
        return self

    def run_evaluation(
        self,
        task_type: str,
        gold_standard: dict[str, Any] | None = None,
    ) -> WorkerOutput:
        """
        Run evaluation metrics and attach to output.

        Args:
            task_type: One of "clustering", "svg", "deconvolution", "other"
            gold_standard: Optional gold-standard data dict (see eval_metrics.py)
        """
        try:
            from eval_metrics import evaluate_tool_output

            eval_result = evaluate_tool_output(
                tool_name=self._tool,
                task_type=task_type,
                output_dir=self._output_files.get("output_dir", ""),
                output_files=self._output_files,
                summary=self._summary,
                gold_standard=gold_standard,
            )
            self._extra["evaluation"] = eval_result
        except Exception as e:
            self._extra["evaluation"] = {
                "error": str(e),
                "task_type": task_type,
            }
        return self

    def to_dict(self) -> dict[str, Any]:
        """Build the final output dict, and record on disk how the run produced it.

        This is a worker's terminal act, which is why the provenance write lives here rather than in
        :meth:`emit`: only 15 of the 65 workers call ``emit()``, while 52 call ``to_dict()`` and
        ``print(json.dumps(...))`` themselves. Hooking ``emit`` looked like the single choke point and
        silently skipped 80% of the tools -- a real scanpy_spatial run on a real Visium slide finished
        with no sidecar, which is how that was caught.

        The write is idempotent (same path, overwritten) and cannot raise, so calling this more than
        once, or in a read-only directory, is harmless.
        """
        self.write_provenance()
        result: dict[str, Any] = {"status": "ok", "tool": self._tool}
        if self._task:
            result["task"] = self._task
        if self._data:
            result["data"] = self._data
        if self._output_files:
            result["output_files"] = self._output_files
        if self._params:
            result["params"] = self._params
        if self._summary:
            result["summary"] = self._summary
        if self._analysis:
            result["analysis"] = self._analysis
        if self._extra:
            result.update(self._extra)
        return result

    def _provenance_dir(self) -> Path | None:
        """The directory the sidecar belongs in, or None if there is nothing to sit beside.

        Prefers the worker's declared ``output_dir``; otherwise the folder its first written file
        went into. Never creates a directory: provenance describes a run that produced artifacts, so
        if no artifact directory exists there is nothing to describe.
        """
        candidates: list[Any] = []
        declared = self._output_files.get("output_dir")
        if isinstance(declared, str) and declared:
            candidates.append(declared)
        for key, value in self._output_files.items():
            if key != "output_dir" and isinstance(value, str) and value:
                candidates.append(str(Path(value).parent))
        for candidate in candidates:
            try:
                path = Path(candidate)
                if path.is_dir():
                    return path
            except OSError:
                continue
        return None

    def write_provenance(self) -> Path | None:
        """Persist how this run was produced, next to what it produced.

        The parameters already exist -- every worker calls ``add_params`` -- but they only ever
        travelled in the stdout JSON, which lives inside one agent's observation and is gone by the
        next session. The .h5ad and .csv files outlive it and carried nothing: no tool name, no
        parameters, no versions. So a user who came back to a results folder a day later could not
        say what had made it, and neither could the agent.

        Returns the path written, or None if nothing was written. Never raises: a worker's job is the
        analysis, and losing a courtesy file must not cost a completed run.
        """
        if not _provenance_enabled():
            return None
        try:
            out_dir = self._provenance_dir()
            if out_dir is None:
                return None
            record = {
                "schema": "sog.run_provenance/1",
                "tool": self._tool,
                "task": self._task,
                # time.gmtime over datetime: the per-tool envs run Pythons as old as 3.8, where
                # datetime.UTC does not exist -- and ruff's UP017 would happily rewrite
                # timezone.utc into it under --unsafe-fixes and break every one of them.
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "params": self._params,
                "input_data": self._data,
                "output_files": self._output_files,
                "python": ".".join(str(v) for v in sys.version_info[:3]),
                "executable": sys.executable,
                # argv is deliberately reduced to the script path: the semantic arguments are already
                # in `params`, and echoing a raw command line would create a new place for a secret
                # passed on the command line to end up on disk.
                "worker_script": sys.argv[0] if sys.argv else None,
                "package_versions": _loaded_package_versions(),
            }
            target = out_dir / PROVENANCE_FILENAME
            target.write_text(
                json.dumps(record, indent=2, ensure_ascii=False, default=_json_default) + "\n",
                encoding="utf-8",
            )
            return target
        except Exception:
            # Read-only output dir, a full disk, a path that vanished between the check and the
            # write -- all of it is strictly less important than the result being reported.
            return None

    def emit(self) -> None:
        """Print the JSON output to stdout (worker convention).

        Flushes stdout before printing to ensure the JSON line is cleanly
        separated from any accidental prior stdout output. If stdout has
        been written to before this call (detectable only partially), a
        warning is emitted to stderr.
        """
        # Provenance is written by to_dict() below, which every worker reaches. The sidecar is
        # intentionally NOT added to output_files: the stdout contract downstream code parses stays
        # byte-for-byte what it was.

        # Flush any buffered stdout content so our JSON starts on a fresh line
        sys.stdout.flush()

        # Write a newline fence to guarantee the JSON starts on its own line,
        # even if prior code wrote to stdout without a trailing newline.
        # The _parse_result scanner handles blank lines gracefully.
        sys.stdout.write("\n")
        sys.stdout.write(json.dumps(self.to_dict(), ensure_ascii=False, default=_json_default))
        sys.stdout.write("\n")
        sys.stdout.flush()

    @staticmethod
    def error(
        tool: str,
        error_msg: str,
        task: str | None = None,
        status: str = "error",
        exc: BaseException | None = None,
    ) -> dict[str, Any]:
        """Create a standard error output dict with diagnostic context.

        Includes the tool name, task, working directory, Python version,
        and the most informative line of the traceback for faster debugging.

        ``status`` defaults to ``"error"``. Use ``"dep_missing"`` to signal
        that a required upstream dependency was unavailable so the orchestrator
        can distinguish it from a genuine runtime error.

        Pass ``exc`` when the caller is no longer inside the ``except`` block. Every worker
        redirects stdout while the tool runs and restores it in a ``finally``, so the error JSON
        is necessarily written *after* the handler has exited -- and by then
        ``traceback.format_exc()`` has nothing left to report. An exception object keeps its own
        ``__traceback__`` regardless, so a worker that held on to it can still name every frame.
        """
        if exc is not None:
            tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        else:
            tb_text = traceback.format_exc()

        # Outside a handler with nothing handed over, format_exc returns the literal
        # "NoneType: None". That is worse than an empty field: it reads like a crash on a None
        # value, which is a misdiagnosis rather than a missing diagnosis. Say where the worker
        # gave up instead -- for a self-detected failure ("input file not found") that is the
        # whole truth, because no exception ever existed.
        reported_at = ""
        if tb_text.strip() == "NoneType: None":
            tb_text = ""
            here = os.path.abspath(__file__)
            for frame in reversed(traceback.extract_stack()[:-1]):
                if os.path.abspath(frame.filename) != here:
                    reported_at = f"{frame.filename}:{frame.lineno} in {frame.name}"
                    break

        # Extract the first meaningful traceback line (the root cause)
        tb_first_line = ""
        for line in tb_text.strip().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("Traceback") and not stripped.startswith("During"):
                tb_first_line = stripped
                break

        # A bare ``assert`` (common in research code) raises AssertionError with no message, so
        # every worker's ``except Exception as e: ... str(e)`` handler reports the empty string and
        # the orchestrator sees {"error": ""} -- no signal at all. Reconstruct something usable from
        # the exception. Non-empty messages are forwarded verbatim.
        if not str(error_msg).strip():
            exc_type = type(exc) if exc is not None else sys.exc_info()[0]
            type_name = getattr(exc_type, "__name__", "") or "Error"
            error_msg = f"{type_name} (no message)"
            if tb_first_line:
                error_msg = f"{type_name} (no message): {tb_first_line}"

        context: dict[str, Any] = {
            "working_directory": os.getcwd(),
            "python_version": sys.version.split()[0],
            "traceback_hint": tb_first_line,
        }
        if reported_at:
            context["reported_at"] = reported_at

        result: dict[str, Any] = {
            "status": status,
            "tool": tool,
            "error": error_msg,
            "traceback": tb_text,
            "context": context,
        }
        if task:
            result["task"] = task
        return result

    @staticmethod
    def emit_error(
        tool: str,
        error_msg: str,
        task: str | None = None,
        status: str = "error",
        exc: BaseException | None = None,
    ) -> None:
        """Print a standard error JSON to stdout with newline fencing."""
        sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.write(
            json.dumps(
                WorkerOutput.error(tool, error_msg, task, status=status, exc=exc),
                ensure_ascii=False,
                default=_json_default,
            )
        )
        sys.stdout.write("\n")
        sys.stdout.flush()


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` so the result is real JSON.

    ``json.dumps`` writes ``float('nan')`` as the bare token ``NaN`` and the infinities as
    ``Infinity``/``-Infinity``. RFC 8259 has no literals for any of the three, so the document is
    not JSON: Python's own ``json.loads`` accepts them, but ``JSON.parse``, ``jq``, R's ``jsonlite``
    and Arrow reject the *whole file*, so one metric that could not be computed costs a reader the
    entire run.

    ``default=`` does not help, and every writer this guards passes it. ``default`` is consulted
    only for objects json cannot serialise at all, and a NaN *is* a float -- it serialises fine,
    just to an invalid token. Worse, a ``default`` actively re-opens the hole for the numpy floats
    this sweep must therefore catch itself: ``numpy.float32``/``float16`` do NOT subclass ``float``
    (only ``float64`` does), so a non-finite one used to sail past the ``isinstance(value, float)``
    check and reach ``default`` -- where :func:`_json_default` turns it into ``float('nan')`` and
    the bare ``NaN`` token is emitted after all (and ``default=str`` writes the string ``"nan"``,
    which readers take for a measurement). The numpy branch below closes that; it looks numpy up in
    ``sys.modules`` because a numpy scalar can only be *in* the payload if numpy is already
    imported, so no import is ever triggered and numpy-free envs pay nothing.

    ``None`` rather than ``0.0`` is deliberate: a zero reads back as a measurement that was taken.

    This is the ``tools/`` copy of :func:`spatialomicsgym.utils.file_io.json_safe`, kept here
    because a worker or a portal can run in a per-tool env with no package on the path. The two
    are pinned to identical behaviour by ``test_the_tools_side_copy_agrees_with_the_package_helper``.
    """
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.floating):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def _json_default(obj: Any) -> Any:
    """Default JSON serializer for numpy/pandas types."""
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    return str(obj)


# A Unix domain socket path is capped at 108 bytes by ``sun_path``, and ``multiprocessing`` builds
# one under the temp dir: ``<tmp>/pymp-XXXXXXXX/listener-XXXXXXXX`` adds 32 characters. Anything
# past ~75 characters of temp dir makes ``Manager()`` -- and so squidpy's ``parallelize``, joblib
# and every other multiprocessing user -- die with ``OSError: AF_UNIX path too long``.
_SUN_PATH_BUDGET = 75


def _is_own_link_to(link: str, real: str) -> bool:
    """Is ``link`` a symlink THIS account made, resolving to ``real``?

    The name is predictable and the system temp root is shared. Comparing only where the link points
    let another local account create it first, aimed at our directory so the check passed, and later
    re-point it -- after which this worker's temp files and sockets landed in a directory that account
    controls (hunt 2026-09-30, u29a-mcp-transport-16). A link we did not create is never used.
    """
    try:
        st = os.lstat(link)
    except OSError:
        return False
    if not os.path.islink(link):
        return False
    getuid = getattr(os, "getuid", None)
    if getuid is not None and st.st_uid != getuid():
        return False
    return os.path.realpath(link) == os.path.realpath(real)


def _prune_dangling_temp_links(root: str) -> int:
    """Remove this account's ``sog-tmp-*`` links whose directory is gone; how many were removed.

    Every distinct long TMPDIR -- one per benchmark trial -- left its link behind for good (588 on
    the build box, 393 of them pointing at nothing). Only links this account owns, and only dead
    ones: a live link may belong to a run still in progress.
    """
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return 0
    uid = getuid()
    try:
        names = os.listdir(root)
    except OSError:
        return 0
    removed = 0
    for name in names:
        if not name.startswith("sog-tmp-"):
            continue
        path = os.path.join(root, name)
        try:
            if os.lstat(path).st_uid != uid or not os.path.islink(path) or os.path.exists(path):
                continue
            os.unlink(path)
            removed += 1
        except OSError:
            continue
    return removed


def _shorten_temp_dir_for_unix_sockets() -> str | None:
    """Give this process a short temp path when the inherited one cannot hold a socket.

    Benchmark harnesses and batch runners point ``TMPDIR`` inside a per-trial directory so that a
    run's scratch files stay with the run. Those paths get deep, and then a tool that parallelizes
    fails on a limit that has nothing to do with the analysis -- the tool looks broken when only the
    path was.

    Rather than move the scratch files out (which would cost that auditability), point a short
    symlink in the system temp root at the real directory and use the symlink. ``bind()`` measures
    the path string it is handed, not the resolved one, so the socket fits while every file still
    lands in the caller's directory.

    Returns the short path now in use, or None if no change was needed or possible.
    """
    import tempfile

    real = tempfile.gettempdir()
    if len(real) <= _SUN_PATH_BUDGET:
        return None

    system_tmp = os.environ.get("SOG_SOCKET_TMP_ROOT", "/tmp")  # short by construction
    digest = hashlib.sha256(os.path.realpath(real).encode()).hexdigest()[:12]
    link = os.path.join(system_tmp, f"sog-tmp-{digest}")
    if len(link) > _SUN_PATH_BUDGET:
        return None  # even the shortcut would not fit; leave the caller's setting alone

    try:
        os.makedirs(real, exist_ok=True)
        _prune_dangling_temp_links(system_tmp)
        if os.path.islink(link):
            if not _is_own_link_to(link, real):
                return None  # someone else's link on the same name; do not steal it
        else:
            os.symlink(real, link)
    except FileExistsError:
        if not _is_own_link_to(link, real):
            return None
    except OSError:
        return None  # read-only /tmp, no symlink permission: nothing safe to do

    tempfile.tempdir = link
    os.environ["TMPDIR"] = link  # so child processes and re-execed workers inherit the short path

    # multiprocessing memoises its temp dir on the current process the first time anything asks for
    # it. If something already did, the long path is cached and the symlink would not be consulted.
    try:
        import multiprocessing.process as _mp_process

        _mp_process.current_process()._config.pop("tempdir", None)
    except Exception:  # best effort; the common case is nothing cached yet
        pass
    return link


SHORT_TEMP_DIR = _shorten_temp_dir_for_unix_sockets()


def _register_null_h5ad_reader() -> bool:
    """Teach this env's anndata to read a ``null``-encoded element, e.g. ``uns/log1p/base``.

    ``scanpy.pp.log1p`` records ``uns["log1p"]["base"] = None``, which anndata writes as an empty
    dataset tagged ``encoding-type: null``. A reader that predates that spec raises
    ``IORegistryError: No read method registered for IOSpec(encoding_type='null', ...)`` and the
    whole h5ad becomes unreadable over one throwaway scalar.

    That bites us because the agent env and the per-tool worker envs pin anndata independently: the
    agent writes an intermediate h5ad with its own newer scanpy, then hands the path to a worker
    whose env cannot read it back. Registering the reader here -- every worker that touches an h5ad
    imports this module -- fixes the whole fleet at one site instead of 53 call sites.

    Purely additive: it only supplies a reader for a spec that currently has none, so no read that
    works today can change behaviour. Returns whether the reader is in place.
    """
    try:
        from anndata._io.specs.registry import _REGISTRY, IOSpec
    except Exception:  # no anndata, or it moved; a worker without it needs no shim
        return False

    def _read_null(*_args: Any, **_kwargs: Any) -> None:
        return None

    spec = IOSpec("null", "0.1.0")
    registered = False
    for src_module, src_name in (
        ("anndata._io.specs.methods", "H5Array"),
        ("anndata._io.specs.registry", "H5Array"),
    ):
        try:
            module = __import__(src_module, fromlist=[src_name])
            src_type = getattr(module, src_name)
        except Exception:
            continue
        try:
            _REGISTRY.register_read(src_type, spec)(_read_null)
            registered = True
        except Exception:  # already registered, or a signature we do not know
            continue
    return registered


NULL_H5AD_READER_REGISTERED = _register_null_h5ad_reader()


def _read_nullable_string(elem: Any, _reader: Any = None, **_kwargs: Any) -> Any:
    """Read a ``nullable-string-array`` group as something the SAME old anndata can write back.

    A worker reads, computes, then writes, and it writes with the anndata that read. So this never
    returns a pandas ``StringArray``: 0.8, 0.9 and 0.10 have no writer for one. The first version of
    this reader did, and that only moved FM-02 from the read to ``adata.write`` -- scanpy_spatial in
    the SpaGCN env ran its whole pipeline on the real visium_bone input, raised ``IORegistryError:
    No method registered for writing StringArray ... while writing key '_index'`` after ~26 s, and
    left a 4992 x 2000 h5ad with an empty obs behind. What comes back instead:

    * **no missing entry** -- an object ndarray of ``str``, which is exactly what these versions read
      a plain ``string-array`` as. The staged visium_bone file is this case in all three of its
      nullable elements (``obs/_index``, ``var/_index``, ``var/gene_ids``: every mask all False).
    * **any missing entry** -- a ``pd.Categorical`` with those entries NaN. Measured in GraphST
      (0.8.0), SpaGCN (0.9.2), spacel_env (0.10.5) and novosparc (0.10.9), it is the one form of a
      missing string that all of them write back as an obs column, as an index and inside a ``uns``
      DataFrame. An object array holding ``None`` is refused in a ``uns`` DataFrame (``Can't
      implicitly convert non-string objects to strings``) and, as an index, is refused by 0.8 and
      written by 0.9/0.10 as the literal string ``'None'``. A categorical is also what their own
      ``write_h5ad`` makes of an obs string column with a gap, so such a column lands on disk as it
      would have anyway. A reader cannot tell an index from a column, and an index with a gap does
      occur (anndata 0.12.6 writes one once ``allow_write_nullable_strings`` is on); it gets the same
      categorical. Building the AnnData casts an index to ``str`` unless pandas calls it a string
      dtype, and pandas 1.x does not call a categorical one: the gap stays NaN under pandas 2.x
      (SpaGCN, spacel_env, novosparc) and becomes the string ``'nan'`` under 1.x (every 0.8 env, and
      0.9.2 in bulk2space_env).

    anndata 0.8 calls a reader with the element alone (``get_reader(...)(elem)``); 0.9 and 0.10 add
    ``_reader=`` as a keyword. Both work here, as they do for the ``null`` reader above: with no
    ``_reader`` the ``values``/``mask`` children go through the module-level ``read_elem``. The first
    version required ``_reader``, registered fine on 0.8 -- so the flag said True -- and then failed
    every read there with ``TypeError: missing 1 required positional argument: '_reader'``.
    """
    import numpy as np
    import pandas as pd

    if _reader is not None:
        read = _reader.read_elem
    else:
        from anndata._io.specs.registry import read_elem as read

    values = np.asarray(read(elem["values"]), dtype=object)
    mask = np.asarray(read(elem["mask"]), dtype=bool) if "mask" in elem else None
    if mask is None or not mask.any():
        return values
    values = values.copy()
    values[mask] = None
    return pd.Categorical(values)


def _register_nullable_string_reader() -> bool:
    """Teach an older anndata to read ``nullable-string-array`` (a pandas ``StringArray`` column).

    anndata >= 0.11 writes a string column with missing values as a group of ``values`` + ``mask``
    tagged ``encoding-type: nullable-string-array``. anndata < 0.11 has no reader for it, so the whole
    file is unreadable: ``IORegistryError: No read method registered for IOSpec(encoding_type=
    'nullable-string-array', ...)``. Measured over the 456 archived SpatialBench trials, that one
    encoding is ALL of FM-02 -- 30 trials, every one on the visium_bone eval, raised in the SpaGCN
    worker (anndata 0.9.2) and novosparc, reading the benchmark's own staged input; 6 of the 30 were
    lost. The agent never wrote the file, so a write-side downgrade could never have helped. What the
    reader hands back, and why that is not a ``StringArray``, is on ``_read_nullable_string``.

    Purely additive, like the ``null`` reader above: it registers only where NO reader exists for
    that spec -- a newer anndata keeps its own native reader untouched. Returns whether the reader
    is (now) available.
    """
    try:
        from anndata._io.specs.registry import _REGISTRY, IOSpec
    except Exception:
        return False
    spec = IOSpec("nullable-string-array", "0.1.0")
    try:
        if any(spec in key for key in list(_REGISTRY.read)):
            return True  # this anndata already reads it natively
    except Exception:
        return False

    registered = False
    for src_module, src_name in (("anndata._io.specs.methods", "H5Group"), ("anndata._io.specs.registry", "H5Group")):
        try:
            module = __import__(src_module, fromlist=[src_name])
            src_type = getattr(module, src_name)
        except Exception:
            continue
        try:
            _REGISTRY.register_read(src_type, spec)(_read_nullable_string)
            registered = True
        except Exception:
            continue
    return registered


NULLABLE_STRING_READER_REGISTERED = _register_nullable_string_reader()


def read_obsm_matrix(h5file: Any, key: str) -> Any:
    """Read ``obsm[key]`` from an open h5ad as an (n_obs, n_cols) float array, either encoding.

    AnnData writes an ``obsm`` entry two different ways, and both are ordinary AnnData:

    * an ndarray becomes an HDF5 **dataset**, which ``[:]`` reads;
    * a ``pandas.DataFrame`` becomes an HDF5 **group** -- one dataset per column, an ``_index``,
      and a ``column-order`` attribute naming the order the author meant.

    ``adata.obsm['spatial']`` returns a usable object for either. Only a worker that skips AnnData
    and walks the HDF5 tree itself has to tell them apart, and doing ``f["obsm"]["spatial"][:]`` on
    the group form raises ``TypeError: Accessing a group is done with bytes or str, not slice`` --
    which is how SVCA came to fail outright on the canonical 78329-spot MERFISH SVG slide, whose
    coordinates are a DataFrame of ``center_x``/``center_y``.

    Column order comes from ``column-order``, never from sorting the HDF5 keys: HDF5 hands back
    group members alphabetically, so a frame written as ``(y, x)`` would silently come back
    transposed and every downstream distance would be wrong in a way nothing reports.

    Raises ``KeyError`` if ``obsm`` or ``key`` is missing -- the callers already distinguish that
    from a malformed entry, and turning it into a return value would hide a typo'd key.
    """
    import numpy as np

    node = h5file["obsm"][key]

    # A Dataset has no .keys(); a Group does. Duck-typed rather than isinstance so this helper
    # stays importable in the several worker envs that have no h5py at all.
    if not hasattr(node, "keys"):
        return np.asarray(node[:])

    def _text(value: Any) -> str:
        return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)

    index_name = _text(node.attrs.get("_index", "_index"))
    order = node.attrs.get("column-order")
    if order is None:
        # Pre-0.8 frames, and anything hand-written. Alphabetical is a guess, so say so rather
        # than let a transposed slide pass as correct.
        columns = sorted(k for k in node.keys() if k != index_name)
        warnings.warn(
            f"obsm[{key!r}] is a DataFrame with no 'column-order' attribute; falling back to "
            f"alphabetical column order {columns}. Verify the coordinates are not transposed.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        columns = [c for c in (_text(v) for v in order) if c != index_name]

    if not columns:
        raise ValueError(f"obsm[{key!r}] is a DataFrame with no data columns")

    vectors = []
    for name in columns:
        column = node[name]
        if hasattr(column, "keys"):
            # A categorical column (codes + categories). Real coordinates are never categorical,
            # and averaging integer codes as if they were microns is worse than refusing.
            raise TypeError(f"obsm[{key!r}] column {name!r} is categorical, not numeric")
        vectors.append(np.asarray(column[:], dtype=np.float64))
    return np.column_stack(vectors)


#: Compression this module can see through, mapped to the stdlib module that opens it as text.
#: Mirrors ``spatialomicsgym.utils.file_io._TEXT_OPENERS`` and, like it, lists only stdlib formats:
#: the point is to read one header line, not to take on a dependency inside 88 worker envs.
_TEXT_OPENERS = {".gz": "gzip", ".bz2": "bz2", ".xz": "lzma"}

#: What reading one header line through those openers can raise when the file is unreadable.
#: OSError covers the filesystem and both gzip's and bz2's *header* checks (``BadGzipFile`` is an
#: OSError), EOFError covers truncation -- but a corrupt *body* surfaces the decompressor's own
#: exception, and neither ``zlib.error`` (gzip) nor ``lzma.LZMAError`` (.xz) subclasses OSError.
#: zlib is compiled into every CPython the worker envs use; lzma is optional at interpreter build
#: time, so its absence must cost nothing when no ``.xz`` is ever read.
try:
    import lzma as _lzma

    _LZMA_ERRORS = (_lzma.LZMAError,)
except ImportError:  # pragma: no cover - a CPython built without liblzma
    _LZMA_ERRORS = ()
import zlib as _zlib

HEADER_READ_ERRORS = (OSError, EOFError, ValueError, _zlib.error, *_LZMA_ERRORS)

#: What ``pd.read_csv`` wants for "split on runs of whitespace".
WHITESPACE_SEP = r"\s+"

#: How much of the first line the sniff is allowed to hold. Only two characters are counted, so the
#: length of the line is incidental to the answer, but ``readline()`` with no argument holds all of
#: it -- and a counts matrix written cells-as-columns has one field per cell, so a 400,000-cell
#: reference is a multi-megabyte header in a process that is about to load the matrix itself.
_HEADER_MAX_CHARS = 1 << 20


def _looks_whitespace_delimited(header: str) -> bool:
    """Whether ``header`` is the first line of a bare numeric matrix written with spaces."""
    if "\t" in header or "," in header:
        return False
    fields = header.split()
    if len(fields) <= 1:
        return False
    for field in fields:
        try:
            float(field)
        except ValueError:
            return False
    return True


def uncompressed_suffix(path) -> str:
    """The suffix ``path`` would have with any recognised compression stripped, lowercased.

    ``a.csv.gz`` -> ``.csv``; ``a.csv`` -> ``.csv``. ``Path.suffix`` answers ``.gz`` for the first,
    which cannot tell a gzipped table from a gzipped image volume.
    """
    path = Path(path)
    if path.suffix.lower() in _TEXT_OPENERS:
        return Path(path.stem).suffix.lower()
    return path.suffix.lower()


def sniff_tabular_sep(path, trust_extension: dict | None = None, default: str = ","):
    """Field separator for a delimited text file, decided by its first line.

    The worker-side copy of ``spatialomicsgym.utils.file_io.sniff_tabular_sep``. It has to be a
    copy: ``tools/`` runs inside per-tool worker envs where ``spatialomicsgym`` is not installed, so
    a worker cannot import the agent-side one. ``test/`` holds the two to the same answers on the
    same bytes, because two implementations that are allowed to drift are the defect this exists to
    fix -- a reader that picks the separator from the filename, which is what every worker did
    before and what left a tab-delimited ``.txt`` counts matrix parsing into zero columns.

    Tab and comma are decided first and the header's larger count wins. Space is not a peer of
    those two: a tool that writes cell types like ``B cells`` into a tab-separated file has more
    spaces in its header than tabs, and a whitespace-preferring sniff would split six columns into
    twelve. Runs of whitespace are offered last and only to a header holding neither of the other
    two, and only when every field on it parses as a number -- that is the ``numpy.savetxt``
    default, and a matrix written with it otherwise reads as one column named after its own first
    row of data.

    ``trust_extension`` maps a suffix to the separator to use without looking, for a caller that
    must not change how a file already on disk is read. ``default`` is the answer when the header
    holds nothing to split on, or cannot be read at all; there is nothing to sniff in those, so the
    caller's convention stands.
    """
    path = Path(path)
    suffix = uncompressed_suffix(path)
    if trust_extension and suffix in trust_extension:
        return trust_extension[suffix]
    opener = _TEXT_OPENERS.get(path.suffix.lower())
    try:
        if opener is None:
            handle = open(path, encoding="utf-8", errors="replace")
        else:
            import importlib

            handle = importlib.import_module(opener).open(path, "rt", encoding="utf-8", errors="replace")
        with handle as fh:
            header = fh.readline(_HEADER_MAX_CHARS)
    except HEADER_READ_ERRORS:
        return default
    if header.count("\t") > header.count(","):
        return "\t"
    if header.count(",") > header.count("\t"):
        return ","
    if _looks_whitespace_delimited(header):
        return WHITESPACE_SEP
    return default


def read_indexed_table(path, what: str = "table"):
    """A delimited table whose first column is row IDs, indexed by it, with the separator sniffed.

    Five workers read a user-supplied counts matrix, single-cell reference, or label file this way,
    and every one of them hardcoded pandas' default comma. On a tab-delimited file each line parses
    into one field, ``index_col=0`` consumes it, and the frame comes back with no data columns at
    all and the whole header line as the name of its index. What the caller does next -- intersect
    the index with a barcode list, take ``.iloc[:, 0]``, call ``.unique()`` -- then fails in a way
    that names identifiers, or names nothing at all.

    Raises ``ValueError`` naming the path, what it was being read as, and the line that became the
    index name. ``what`` is the caller's own word for the file, so a worker with several tabular
    inputs says which one would not parse.
    """
    import pandas as pd

    frame = pd.read_csv(path, sep=sniff_tabular_sep(path), index_col=0)
    if frame.shape[1] == 0:
        raise ValueError(
            f"{path}: read as {what}, but it parsed into 0 data columns -- the whole first line "
            f"became the index name ({frame.index.name!r}). Check the file's field separator; a "
            f"{what} file needs an ID column and at least one data column."
        )
    return frame


#: The two names Space Ranger has given the spot-coordinate file, the current one first. Version
#: 2.0 (2022) renamed ``tissue_positions_list.csv`` to ``tissue_positions.csv`` *and* started
#: writing a header row. Those two changes travel together, which is why finding the file and
#: reading it are both here: a caller that learns only the new name reads its header as a spot.
TISSUE_POSITIONS_NAMES = ("tissue_positions.csv", "tissue_positions_list.csv")

#: The six columns 10x documents, in order. Read positionally, because position is the only thing
#: the two layouts agree on -- the older file has no names to read.
TISSUE_POSITIONS_COLUMNS = (
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
)


def find_tissue_positions(spatial_dir):
    """The Visium spot-coordinate file under ``spatial_dir``, or ``None`` if there is not one.

    Looks for both Space Ranger spellings, current first, in ``spatial_dir`` itself and in a
    ``spatial/`` beneath it -- callers pass either the ``outs/`` directory or the ``spatial/``
    directory inside it, and both are ordinary.

    Returns ``None`` rather than raising so each worker keeps the not-found message it already
    words for its own arguments.
    """
    root = Path(spatial_dir)
    for directory in (root, root / "spatial"):
        for name in TISSUE_POSITIONS_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    return None


def _tissue_positions_first_row_is_a_header(row) -> bool:
    """Whether row 0 holds column names rather than a spot.

    Decided from the row, not from the filename: a file copied, renamed, or re-exported on its way
    here still reads correctly. ``in_tissue``/``array_row``/``array_col`` are integers in every
    data row and words in every header, so they answer this without touching the barcode -- which
    is a string either way.
    """
    for value in list(row)[1:4]:
        try:
            float(str(value).strip())
        except (TypeError, ValueError):
            return True
    return False


def read_tissue_positions(path):
    """A Visium spot-coordinate file as a frame of the six columns 10x documents.

    Handles both Space Ranger layouts: the headerless pre-2.0 file and the 2.0 file with a header.
    ``barcode`` is returned as a string column, not an index, so a caller can ``set_index`` or
    ``intersect1d`` as it prefers.

    The separator is sniffed rather than assumed. Space Ranger writes commas, so the sniff answers
    "," for every file this repo has ever seen; a tab-delimited re-export used to parse into one
    column and be reported as a file with the wrong shape.

    Raises ``ValueError`` naming the path and the count when the file has fewer than six columns.
    """
    import pandas as pd

    sep = sniff_tabular_sep(path)
    frame = pd.read_csv(path, sep=sep, header=None)
    if len(frame) and _tissue_positions_first_row_is_a_header(frame.iloc[0]):
        frame = pd.read_csv(path, sep=sep, header=0)
    if frame.shape[1] < len(TISSUE_POSITIONS_COLUMNS):
        raise ValueError(
            f"{path} has {frame.shape[1]} column(s); a Visium tissue-positions file has at least "
            f"{len(TISSUE_POSITIONS_COLUMNS)}: {', '.join(TISSUE_POSITIONS_COLUMNS)}"
        )
    frame = frame.iloc[:, : len(TISSUE_POSITIONS_COLUMNS)].copy()
    frame.columns = list(TISSUE_POSITIONS_COLUMNS)
    frame["barcode"] = frame["barcode"].astype(str)
    return frame


#: The coordinate file a worker accepts beside a counts matrix when it is not given a Space Ranger
#: tree. Four workers read this file, and all four called ``pd.read_csv`` with pandas' default
#: comma, so a tab-delimited export parsed as one column named after the whole header line.
COORDS_CSV_COLUMNS = ("barcode", "x", "y")


def read_coords_csv(path, barcode_aliases=(), barcode_from_first_column: bool = False):
    """A ``barcode,x,y[,in_tissue]`` coordinate file as a frame carrying those four columns.

    The separator is sniffed from the header rather than assumed, and ``in_tissue`` is filled with
    1 where the file does not carry it. ``barcode`` is returned as a string column, not an index,
    so a caller can ``set_index`` or ``intersect1d`` as it prefers.

    The leniencies are opt-in, so no worker starts accepting a file it refuses today:
    ``barcode_aliases`` names other spellings of the spot-ID column to try, in order, and
    ``barcode_from_first_column`` allows the first column to be taken as the spot ID when it is not
    one of the coordinate names.

    Raises ``ValueError`` naming the path, the columns that are missing, and the columns that were
    actually parsed. That last one is the difference between diagnosing a delimiter in one turn and
    sending the user off to audit barcodes that were never the problem.
    """
    import pandas as pd

    frame = pd.read_csv(path, sep=sniff_tabular_sep(path))
    if "barcode" not in frame.columns:
        for alias in barcode_aliases:
            if alias in frame.columns:
                frame = frame.rename(columns={alias: "barcode"})
                break
        else:
            first = frame.columns[0] if frame.shape[1] else None
            if barcode_from_first_column and first is not None and first not in ("x", "y", "in_tissue"):
                frame = frame.rename(columns={first: "barcode"})
    missing = [name for name in COORDS_CSV_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {', '.join(missing)}. "
            f"Parsed {frame.shape[1]} column(s): {list(frame.columns)}. "
            f"A coordinate file has {','.join(COORDS_CSV_COLUMNS)}[,in_tissue]."
        )
    if "in_tissue" not in frame.columns:
        frame["in_tissue"] = 1
    frame["barcode"] = frame["barcode"].astype(str)
    return frame


#: The coordinate-column pairs a worker recognises, in the order it tries them. Pixel coordinates
#: outrank the array lattice because only the former is a distance; ``x``/``y`` is last because it
#: is the vaguest, so a file that also names an image axis is read on that. Mirrored verbatim by
#: the ``resolve_coord_cols`` helper inlined in fourteen ``tools/*.R`` workers -- R has no shared
#: library on the path, Python does -- and the two orders are pinned equal by
#: ``test/test_a_python_worker_reads_the_coordinate_columns_the_file_names.py``.
COORD_COLUMN_PAIRS = (
    ("imagerow", "imagecol"),
    ("pxl_row_in_fullres", "pxl_col_in_fullres"),
    ("array_row", "array_col"),
    ("row", "col"),
    ("x", "y"),
)

#: Columns that ride along in a coordinate file without being coordinates. Set aside before the
#: fallback picks the two survivors, so a Space Ranger export does not offer ``in_tissue`` as an axis.
COORD_FLAG_COLUMNS = (
    "barcode",
    "barcodes",
    "spot",
    "spot_id",
    "spotid",
    "cell",
    "cell_id",
    "cellid",
    "sample",
    "sample_id",
    "index",
    "tissue",
    "in_tissue",
)


def resolve_coord_columns(columns, source_path):
    """Which two of ``columns`` hold the coordinates, decided by name rather than by order.

    For a file the *user* already had, order is not a contract: Space Ranger's own
    ``tissue_positions.csv`` leads with ``in_tissue, array_row``, so taking the first two columns
    gives an axis with one distinct value and an integer lattice index -- a spatial graph carrying
    no spatial information, at exit code 0.

    Returns the pair in the file's own spelling, so the caller can index the frame with it. Matching
    ignores case and surrounding whitespace. Where no known pair is present the two columns that are
    not flags are taken, which is the shape a hand-written file usually has.

    Raises ``ValueError`` naming the file, listing the columns it actually has, and naming the
    spellings that would work -- the difference between one turn and an audit.

    This is the third coordinate policy in this module and the only one for a file whose layout
    nobody promised: :func:`read_tissue_positions` reads positionally because 10x documents that
    file's column order, and :func:`read_coords_csv` demands a literal ``barcode,x,y``.
    """
    names = list(columns)
    lowered = [str(name).strip().lower() for name in names]
    for pair in COORD_COLUMN_PAIRS:
        if all(wanted in lowered for wanted in pair):
            # Indexed rather than zipped: workers run on Python 3.8 in some tool envs, where
            # zip(strict=) does not exist, and a silent truncation here would swap an axis.
            return (names[lowered.index(pair[0])], names[lowered.index(pair[1])])

    rest = [names[i] for i, low in enumerate(lowered) if low not in COORD_FLAG_COLUMNS]
    if len(rest) < 2:
        raise ValueError(
            f"Need two coordinate columns in {source_path} and found {len(rest)} once the barcode "
            f"and tissue-flag columns were set aside; the file has: {', '.join(str(n) for n in names)}. "
            "Name the two coordinates imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, "
            "array_row/array_col, row/col or x/y, or give them as the only two columns besides the flags."
        )
    return (rest[0], rest[1])


#: obs columns that name which section a cell came from, checked in this order when a projection
#: has to say how many sections it is about to collapse. Same list as ``viz/layers.py``, which is
#: the other place in the tree that has to find the section axis without being told.
SECTION_OBS_COLUMNS = (
    "slice_id",
    "slice",
    "section",
    "section_id",
    "batch",
    "library_id",
    "sample",
    "sample_id",
    "brain_section_label",
    "Bregma",
    "z_slice",
)


def _section_count(adata, section_key=None):
    """How many sections an object holds, or 0 when it does not say. Never raises."""
    try:
        columns = list(getattr(adata, "obs", {}).columns)
    except Exception:
        return 0, ""
    names = [section_key] if section_key else list(SECTION_OBS_COLUMNS)
    for name in names:
        if name and name in columns:
            try:
                return int(adata.obs[name].astype(str).nunique()), name
            except Exception:
                return 0, name
    return 0, ""


def _two_column_advice(adata, spatial_key, want):
    """What a caller should actually do about a key that is too wide, given what the object holds.

    Two different situations wear the same error. Usually a 3D key was named while the untouched
    original sits beside it, and the answer is one argument. But when ``obsm['spatial']`` is ITSELF
    three columns -- the shape STARmap's converter and ST-GEARS both write -- there is no 2D
    original to fall back to, and telling the caller to name one would be a dead end.
    """
    try:
        obsm = getattr(adata, "obsm", None) or {}
        narrow = sorted(
            k for k in obsm if k != spatial_key and getattr(obsm[k], "ndim", 0) == 2 and obsm[k].shape[1] == want
        )
    except Exception:
        narrow = []
    if narrow:
        return (
            f"Pass spatial_key={narrow[0]!r} instead -- under the 3D coordinate contract "
            "obsm['spatial'] is the untouched 2D original and is never written by an aligner."
        )
    return (
        f"This object has no {want}-column coordinate key at all, which breaks the 3D coordinate "
        "contract: obsm['spatial'] is required to stay the 2D original precisely so a 2D tool has "
        "something correct to read. Run the spatial3d_inspector portal's inspect_3d_coordinates to "
        "see which frames it holds, and write the 2D original back before running this tool."
    )


def spatial_coords(adata, spatial_key="spatial", want=2, tool="", project=False, section_key=None):
    """The coordinates a worker asked for, at the width it asked for. ``(coords, note)``.

    **Raises rather than truncates.** Thirteen workers reached into ``obsm[key]`` and wrote
    ``[:, :2]``, which on a three-column key -- the shape the 3D coordinate contract produces, and
    the shape ST-GEARS and STARmap already write -- drops the z and lays every section of a serial
    stack on top of each other at the same plane. The run then finishes, exits 0, writes a result
    file, and reports spatially variable genes computed on a tissue that is several sections thick
    pretending to be one. Nothing in the output says so.

    Collapsing a stack is a decision about the data, so it is the caller's to make and to state:
    ``project=True`` performs it and returns a note saying how many sections were flattened, which
    the caller is expected to put in its payload. The default is the refusal, because the
    repository's rule is refuse-don't-invent and a projected z is an invented coordinate.

    ``adata`` may be an AnnData, in which case ``spatial_key`` is looked up in ``obsm`` and a
    missing key is reported with the keys the object does have; or a plain array already read out
    of an HDF5 file, in which case ``spatial_key`` is used only to name it in the messages.

    ``want`` and ``project`` are ordinary positional-or-keyword arguments rather than keyword-only:
    this module is imported by 66 workers on interpreters as old as 3.7, and a signature those
    cannot parse takes every one of them down together.
    """
    import numpy as np

    obsm = getattr(adata, "obsm", None)
    if obsm is not None:
        if spatial_key not in obsm:
            raise KeyError(f"Spatial key '{spatial_key}' not found in adata.obsm. Available keys: {list(obsm.keys())}")
        raw = obsm[spatial_key]
    else:
        raw = adata

    arr = raw.toarray() if hasattr(raw, "toarray") else raw
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"{tool or 'this tool'}: obsm['{spatial_key}'] has {arr.ndim} dimensions, not 2; "
            "it does not hold coordinates."
        )

    width = int(arr.shape[1])
    want = int(want)
    if width == want:
        return arr, ""
    if width < want:
        raise ValueError(
            f"{tool or 'this tool'}: obsm['{spatial_key}'] has {width} column(s) and this tool needs {want}."
        )

    n_sections, section_from = _section_count(adata, section_key)
    where = f" across {n_sections} sections in obs['{section_from}']" if n_sections > 1 else ""
    if not project:
        raise ValueError(
            f"{tool or 'this tool'}: obsm['{spatial_key}'] has {width} columns and this tool reads "
            f"{want}. Taking the first {want} would drop the remaining axis and lay every section "
            f"on one plane{where} -- the run would succeed and the result would be about a tissue "
            f"that does not exist. {_two_column_advice(adata, spatial_key, want)}"
        )
    note = (
        f"obsm['{spatial_key}'] had {width} columns; the first {want} were used and the remaining "
        f"axis was dropped, collapsing{where or ' the stack'} onto one plane. Every distance this "
        "tool computed is an in-plane distance."
    )
    return arr[:, :want], note


#: Micrometres per unit, by the unit names ``spatial3d/contract.py`` records (plus their spelled-out
#: forms). ``None`` means "not a physical length": a pixel, an array index or an undeclared unit has
#: no micrometre equivalent this module can know, and inventing one is what coupled every cell on
#: Zhuang's millimetres to every other under a 200 "um" threshold.
UNIT_UM = {
    "um": 1.0,
    "micrometre": 1.0,
    "micrometer": 1.0,
    "micron": 1.0,
    "mm": 1000.0,
    "millimetre": 1000.0,
    "px": None,
    "pixel": None,
    "pixels": None,
    "unknown": None,
}

#: obs columns that name a section, in the order they are tried. A copy of
#: ``spatialomicsgym.viz.layers.SECTION_COLUMNS`` rather than an import: workers run in per-tool envs
#: that do not have the portal package. A test keeps the two equal.
SECTION_COLUMN_NAMES = (
    "library_id",
    "slice_id",
    "section",
    "section_id",
    "brain_section_label",
    "Bregma",
    "z_index",
    "batch",
)

#: The names the flattened-stack refusal reads: every section name except ``batch``, which as often
#: labels merged replicates of one plane as it labels sections, and refusing those would be wrong.
STACK_COLUMN_NAMES = tuple(name for name in SECTION_COLUMN_NAMES if name != "batch")

#: Same cap as ``viz.layers.MAX_SECTION_LEVELS``: a column with more levels than this is an
#: identifier or a continuous value stored as a category, not a stack of sections.
MAX_SECTION_LEVELS = 200


class Frame:
    """What a worker learned about the coordinates it was given: the key, the dimensions it read, micrometres
    per unit on each axis (None when unknown), the z provenance, and the sections if any.

    ``units_per_axis`` has one entry per column returned. ``z_source`` is what the 3D contract recorded
    for the key (``None`` when the key has no declaration, which is the normal case for a 2D ``spatial``).
    ``sections`` is the list of section labels in stacking order -- the contract's ``slice_order`` when it
    names exactly these labels, otherwise the order of first appearance -- or ``None`` when no section
    column was used.
    """

    def __init__(self, key, dims, units_per_axis, z_source, section_key, sections):
        self.key = key
        self.dims = int(dims)
        self.units_per_axis = tuple(units_per_axis)
        self.z_source = z_source
        self.section_key = section_key
        self.sections = list(sections) if sections is not None else None

    def to_dict(self):
        """JSON-ready, for a worker's provenance file."""
        return {
            "coords_key": self.key,
            "dims": self.dims,
            "units_per_axis_um": list(self.units_per_axis),
            "z_source": self.z_source,
            "section_key": self.section_key,
            "sections": self.sections,
        }

    def __repr__(self):
        return (
            f"Frame(key={self.key!r}, dims={self.dims}, units_per_axis={self.units_per_axis!r}, "
            f"z_source={self.z_source!r}, section_key={self.section_key!r}, sections={self.sections!r})"
        )


def _declared_frame(adata, key):
    """The contract's declaration for ``obsm[key]`` as a plain dict, or ``None``. Never raises."""
    try:
        block = adata.uns.get("spatial_3d")
        frames = block.get("frames") if hasattr(block, "get") else None
        declared = frames.get(key) if hasattr(frames, "get") else None
    except Exception:
        return None
    return dict(declared) if hasattr(declared, "keys") else None


def _section_labels(adata, column):
    """Section labels of ``obs[column]`` in stacking order: the contract's ``slice_order`` when it names
    exactly these labels (a reader never sorts it), otherwise the order of first appearance."""
    import pandas as pd

    present = [str(v) for v in pd.unique(adata.obs[column].astype(str))]
    try:
        block = adata.uns.get("spatial_3d")
        order = block.get("slice_order") if hasattr(block, "get") else None
        if order is not None and not isinstance(order, (str, bytes)):
            order = [str(s) for s in order]
            if sorted(order) == sorted(present):
                return order
    except Exception:
        pass
    return present


def _stack_column(adata):
    """The first section-like column (:data:`STACK_COLUMN_NAMES`) with 2..:data:`MAX_SECTION_LEVELS`
    levels, as ``(n, column)``, or ``(0, "")``."""
    for name in STACK_COLUMN_NAMES:
        n, column = _section_count(adata, name)
        if 2 <= n <= MAX_SECTION_LEVELS:
            return n, column
    return 0, ""


def spatial_frame(adata, coords_key="spatial", dims=2, section_key=None, tool="", three_d_ok=True, per_section_ok=True):
    """-> (coords_um ndarray[n, dims] float64, Frame). Raises ValueError with the sentences below.

    The one way a communication worker reads coordinates. Checked in this order, each a refusal rather
    than a warning, because every one of them otherwise produces a run that exits 0 with a wrong answer:

    1. ``coords_key`` absent from ``obsm`` -> ``KeyError`` listing the keys that exist.
    2. ``obsm['spatial']`` with three or more columns -> refused: the 3D contract keeps ``spatial``
       two-dimensional, and the aligned frame belongs in its own key; or run per section in 2D.
    3. ``dims=3`` on a key with no entry in ``uns['spatial_3d']['frames']``, or with unknown units ->
       refused, naming ``spatialomicsgym.spatial3d.contract.write_frame`` as the way to declare it.
    4. ``dims=3`` on a z whose source is ``rank_index`` -> refused: a section rank is not a distance.
    5. ``dims=2`` with no ``section_key`` on a file holding two or more sections -> refused: a 2D run
       would overlay them. The message names the column and both ways out. Only section-like columns
       count (:data:`STACK_COLUMN_NAMES`, so not ``batch``), and only up to :data:`MAX_SECTION_LEVELS`
       levels, so an identifier-like column never triggers it.
    6. A key narrower (or, off ``spatial``, wider) than ``dims`` -> :func:`spatial_coords`'s refusal.
    7. Each axis is converted to micrometres by :data:`UNIT_UM`. In 3D a unit with no micrometre
       equivalent (pixels, array indices) is refused. **In 2D the coordinates are left as the file
       holds them** and ``units_per_axis`` is ``(None, None)`` unless the key declares its units: 2D
       tools have run on raw units since forever, and ``spatial`` carries no unit declaration under the
       contract. A caller converting a micrometre threshold must therefore check ``units_per_axis``.

    ``section_key`` names the obs column of section labels. In 2D it is the caller's statement that the
    run is per section (the caller is expected to use :func:`per_section`); in 3D it is only recorded,
    and defaults to the contract's ``slice_key``. Cross-section is always defined by these labels, never
    by assuming the third column is depth -- Zhuang is cut coronally and stacks along CCF x, column 0.

    ``three_d_ok=False`` is for a tool that is two-dimensional by design (DeepLinc, NCEM, MISTy): the
    refusals then name only the per-section way out and say that a 3D run is not offered, instead of
    pointing the caller at a ``dims=3`` run the tool would refuse. The default keeps every message as it was.

    ``per_section_ok=False`` is for a tool that does not run per section (SpaOTsc builds one spot-to-spot
    distance matrix over its whole input): the refusals then name the 3D way out and a file subset to a single
    section, never ``dims=2, section_key=<column>``, and a 2D ``section_key`` over two or more sections is
    itself refused. The two flags cannot both be False: such a tool would have no way out to offer.

    ``dims``, ``section_key``, ``tool``, ``three_d_ok`` and ``per_section_ok`` are positional-or-keyword: this
    module is imported by workers on interpreters as old as 3.7.
    """
    import numpy as np

    who = tool or "this tool"
    dims = int(dims)
    if dims not in (2, 3):
        raise ValueError(f"{who}: dims must be 2 or 3, not {dims}.")
    if not three_d_ok and not per_section_ok:
        raise ValueError(f"{who}: three_d_ok and per_section_ok are both False, so no way out could be offered.")
    # The 2D way out a refusal names: per section, or (for a tool that does not run per section) a subset.
    if per_section_ok:
        way_2d = "run per section with `dims=2, section_key=<column>`"
    else:
        way_2d = f"subset the file to a single section and run with `dims=2`; {who} does not run per section"

    # (1)
    obsm = getattr(adata, "obsm", None)
    if obsm is None or coords_key not in obsm:
        keys = list(obsm.keys()) if obsm is not None else []
        raise KeyError(f"Spatial key '{coords_key}' not found in adata.obsm. Available keys: {keys}")
    raw = obsm[coords_key]
    width = int(np.shape(raw)[1]) if len(np.shape(raw)) == 2 else 0

    # (2)
    if coords_key == "spatial" and width >= 3:
        if not three_d_ok:
            raise ValueError(
                f"{who}: `obsm['spatial']` holds {width} columns; the 3D contract keeps `spatial` two-dimensional. "
                "Run per section with `dims=2, section_key=<column>` on two-column coordinates; "
                f"{who} is two-dimensional, so a 3D run is not offered."
            )
        if not per_section_ok:
            raise ValueError(
                f"{who}: `obsm['spatial']` holds {width} columns; the 3D contract keeps `spatial` two-dimensional. "
                "Write the aligned frame to `obsm['spatial_3d_aligned']` (or name your frame with `coords_key`) "
                f"and pass `dims=3`, or {way_2d}."
            )
        raise ValueError(
            f"{who}: `obsm['spatial']` holds {width} columns; the 3D contract keeps `spatial` two-dimensional. "
            "Write the aligned frame to `obsm['spatial_3d_aligned']` (or name your frame with `coords_key`) "
            "and pass `dims=3`, or run per section with `dims=2, section_key=<column>` on two-column coordinates."
        )

    if section_key is not None:
        columns = [str(c) for c in adata.obs.columns]
        if section_key not in columns:
            raise ValueError(
                f"{who}: section_key '{section_key}' is not a column of obs. Columns: {columns}. "
                "Name the column that holds each cell's section label."
            )

    declared = _declared_frame(adata, coords_key)
    xy_units = str(declared.get("xy_units") or "unknown") if declared else "unknown"
    z_units = str(declared.get("z_units") or "unknown") if declared else "unknown"
    z_source = (str(declared.get("z_source")) if declared.get("z_source") else None) if declared else None

    if dims == 3:
        # (3)
        if declared is None:
            raise ValueError(
                f"{who}: `obsm['{coords_key}']` has no entry in uns['spatial_3d']['frames'], so its units and "
                "how its z was obtained are unknown, and no distance on it can be trusted. Declare it with "
                f"spatialomicsgym.spatial3d.contract.write_frame (xy_units, z_units, z_source), or {way_2d}."
            )
        if xy_units == "unknown" or z_units == "unknown":
            raise ValueError(
                f"{who}: `obsm['{coords_key}']` declares xy_units='{xy_units}', z_units='{z_units}'; a 3D "
                "distance needs both. Re-declare the frame with spatialomicsgym.spatial3d.contract.write_frame "
                f"giving its units (um or mm), or {way_2d}."
            )
        # (4)
        if z_source == "rank_index" and not three_d_ok:
            raise ValueError(
                f"{who}: z of `obsm['{coords_key}']` is a section rank, not a distance. Run per section with "
                f"`dims=2, section_key=<column>`; {who} is two-dimensional, so a 3D run is not offered."
            )
        if z_source == "rank_index":
            raise ValueError(
                f"{who}: z of `obsm['{coords_key}']` is a section rank, not a distance; a 3D graph cannot be "
                "built on it — give a measured or registered z (z_source measured / section_metadata / "
                f"atlas_registration), or {way_2d}."
            )
    elif section_key is None:
        # (5)
        n, column = _stack_column(adata)
        if n >= 2 and not three_d_ok:
            raise ValueError(
                f"{who}: this file holds {n} sections in obs['{column}']; a 2D run would overlay them. Choose "
                f"per-section 2D with `dims=2, section_key='{column}'`; {who} is two-dimensional, so a 3D run is "
                "not offered."
            )
        if n >= 2 and not per_section_ok:
            raise ValueError(
                f"{who}: this file holds {n} sections in obs['{column}']; a 2D run would overlay them. Choose 3D "
                f"with `coords_key=<aligned frame>, dims=3`, or run with `dims=2` on a file subset to one "
                f"obs['{column}'] label; {who} does not run per section."
            )
        if n >= 2:
            raise ValueError(
                f"{who}: this file holds {n} sections in obs['{column}']; a 2D run would overlay them. Choose "
                f"per-section 2D with `dims=2, section_key='{column}'`, or 3D with `coords_key=<aligned frame>, "
                "dims=3`."
            )
    elif not per_section_ok:
        n = len(_section_labels(adata, section_key))
        if n >= 2:
            raise ValueError(
                f"{who}: obs['{section_key}'] holds {n} sections, and {who} does not run per section: a 2D run would "
                f"lay them on one plane. Choose 3D with `coords_key=<aligned frame>, dims=3`, or run with `dims=2` "
                f"on a file subset to one obs['{section_key}'] label."
            )

    # (6)
    coords, _ = spatial_coords(adata, coords_key, dims, tool, False, section_key)
    coords = np.array(coords, dtype=np.float64)

    # (7)
    unit_names = [xy_units, xy_units] + ([z_units] if dims == 3 else [])
    factors = [UNIT_UM.get(u.lower()) for u in unit_names]
    if dims == 3:
        unknown = [u for u, f in zip(unit_names, factors) if f is None]
        if unknown:
            raise ValueError(
                f"{who}: `obsm['{coords_key}']` is declared in {sorted(set(unknown))}, which is not a length "
                f"this tool can convert to micrometres (known: um, mm). Re-declare the frame with "
                f"spatialomicsgym.spatial3d.contract.write_frame in um or mm, or {way_2d}."
            )
    if all(f is not None for f in factors):
        coords = coords * np.asarray(factors, dtype=np.float64)
        units_per_axis = tuple(float(f) for f in factors)
    else:
        units_per_axis = (None,) * dims

    # Sections: the caller's column, else (3D only) the contract's slice_key, else none.
    column = section_key
    if column is None and dims == 3:
        try:
            slice_key = adata.uns["spatial_3d"].get("slice_key")
        except Exception:
            slice_key = None
        if slice_key and str(slice_key) in [str(c) for c in adata.obs.columns]:
            column = str(slice_key)
        else:
            column = _stack_column(adata)[1] or None
    sections = _section_labels(adata, column) if column else None

    return coords, Frame(coords_key, dims, units_per_axis, z_source, column, sections)


def per_section(adata, section_key, run_one, tool=""):
    """run_one(adata_section, label) -> pandas.DataFrame; concatenation with a leading 'section' column, in the
    file's order of first appearance; ValueError naming the column when it has fewer than 2 levels.

    Each ``adata_section`` is an in-memory copy holding only that section's cells, so ``run_one`` may
    modify it. A ``section`` column ``run_one`` returns is replaced by the label, which is the one fact
    this function knows for certain about those rows.
    """
    import pandas as pd

    who = tool or "this tool"
    columns = [str(c) for c in adata.obs.columns]
    if section_key not in columns:
        raise ValueError(f"{who}: section_key '{section_key}' is not a column of obs. Columns: {columns}.")
    labels_per_cell = adata.obs[section_key].astype(str).to_numpy()
    labels = [str(v) for v in pd.unique(labels_per_cell)]
    if len(labels) < 2:
        raise ValueError(
            f"{who}: obs['{section_key}'] has {len(labels)} level(s), so there is nothing to run per section. "
            "Run once on the whole file with `dims=2` and no section_key."
        )

    parts = []
    for label in labels:
        sub = adata[labels_per_cell == label].copy()
        result = run_one(sub, label)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"{who}: run_one returned {type(result).__name__} for section '{label}'; it must return a "
                "pandas.DataFrame."
            )
        result = result.copy()
        if "section" in result.columns:
            result = result.drop(columns="section")
        result.insert(0, "section", label)
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def default_output_dir(subdir: str = "") -> str:
    """A worker's fallback scratch directory when the caller supplies no ``output_dir``.

    Delegates to :func:`base_mcp.default_output_dir` -- the *one* place the resolution order
    (``SOG_WORK_DIR`` -> a writable ``/workspace/work`` -> the relative ``./work`` the shipped
    config advertises) is written down. Re-exported here because that is the module every worker
    already imports, while ``base_mcp`` is the portals' module; a worker that restated the three
    lines would be free to drift from the portal that calls it, which is precisely how the workers
    came to keep the absolute literal after the portals stopped using it.

    Imported inside the function, not at module scope: ``worker_utils`` is imported by 66 workers on
    interpreters as old as 3.7, and an unconditional new top-level import would make every one of
    them fail together if it ever went wrong. Both files sit in the same directory, which is
    ``sys.path[0]`` for a worker launched as ``python .../tools/<x>_worker.py``.
    """
    from base_mcp import default_output_dir as _resolve

    return _resolve(subdir)


def safe_symlink_or_copy(src: str, dst: str) -> None:
    """Stage ``src`` at ``dst``, preferring a symlink and falling back to a copy.

    Re-run safe. ``Path.exists()`` follows symlinks, so a link left dangling by an earlier run
    (its target moved or was deleted) reads as *absent*: ``os.symlink`` then raises
    ``FileExistsError`` and ``shutil.copy2`` writes *through* the dead link, failing with a
    baffling ``FileNotFoundError`` that names a path which is plainly there. A link pointing at
    the wrong source is worse still -- the tool silently analyses the wrong file. Both are
    re-staged; an already-correct link and a real file already in place are left untouched.
    """
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)

    if dst_p.is_symlink():
        try:
            if dst_p.exists() and dst_p.resolve() == src_p.resolve():
                return
        except OSError:
            pass  # unresolvable (dangling / loop) -> re-stage below
        dst_p.unlink()
    elif dst_p.exists():
        return

    try:
        os.symlink(src_p.resolve(), dst_p)
    except Exception:
        shutil.copy2(src_p, dst_p)


def graph_batch_size(requested: int, total_nodes: int) -> int:
    """Size a graph mini-batch so neighbour extension can still grow it.

    GNN trainers that sample a batch and then extend it with the batch's graph neighbours
    (SPIRAL, GraphSAGE-style) require the extended set to be *strictly* larger than the batch.
    A batch holding every node cannot satisfy that -- SPIRAL asserts it outright
    (``assert set(target_nodes) < set(unique_nodes_batch)``) and dies with a bare
    ``AssertionError`` on any dataset no larger than the requested batch size.

    So: a batch that already fits inside the dataset is returned unchanged; one that would
    cover the whole dataset is halved to leave real headroom for extension. The result is
    always >= 1, because ``DataLoader(drop_last=True)`` yields no batches at all for 0.
    """
    requested = max(1, int(requested))
    total_nodes = int(total_nodes)
    if total_nodes <= 2:
        return 1
    if requested < total_nodes:
        return requested
    return max(1, total_nodes // 2)


# --------------------------------------------------------------------------- #
# Compute selection -- one vocabulary, one CUDA probe, one CPU budget.
#
# The portals expose device/thread selection to the LLM in four incompatible vocabularies:
# ``device`` ('cuda'/'cpu'/'CPU'/'GPU'/'auto'), ``use_gpu`` (bool or 'auto'), ``gpu`` (int index,
# -1 meaning CPU -- see spaceflow_worker.py:240) and ``accelerator`` ('gpu'/'cpu'/'auto'). The agent
# fills them in from a natural-language prompt, so any spelling can arrive at any worker. Workers
# that resolved this privately each had a different hole: an unrecognised token passed straight to
# ``torch.device()`` (RuntimeError), or a CUDA request honoured on a box with no CUDA, or 'GPU'
# quietly demoted to CPU on a box that has one. Resolve here instead.
# --------------------------------------------------------------------------- #
class Compute(NamedTuple):
    """A resolved hardware choice: a device string ``torch.device()`` accepts + a CPU thread budget."""

    device: str
    n_threads: int


_CPU_WORDS = frozenset({"cpu", "false", "no", "off", "none"})
_GPU_WORDS = frozenset({"gpu", "cuda", "true", "yes", "on"})
_AUTO_WORDS = frozenset({"", "auto", "default", "any"})
# 'cuda:1', 'gpu:1', 'cuda1' -- an explicit accelerator index.
_INDEXED_GPU = re.compile(r"^(?:gpu|cuda)[:_ ]?(\d+)$")
# The int vocabulary after a trip through argparse/JSON, where it arrives as text.
_INTEGER = re.compile(r"^-?\d+$")
# What torch.device() will actually parse. Anything else raises RuntimeError there.
_TORCH_DEVICE = re.compile(r"^(cpu|cuda|mps|xpu)(:\d+)?$")


def _cuda_available() -> bool:
    """Is a CUDA device really usable here? False -- never an exception -- if torch is absent."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


#: (limit, usage, stat, hierarchical prefix in stat) for cgroup v2, then v1.
_CGROUP_MEMORY_FILES = (
    ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.stat", ""),
    (
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/memory/memory.stat",
        "total_",
    ),
)
#: cgroup v1 reports "no limit" as a page-rounded LONG_MAX; anything this large is no limit.
_NO_CGROUP_LIMIT = float(1 << 60)


def _read_small_text(path):
    """The text of a small kernel file, or None when it cannot be read."""
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, ValueError):
        return None


def _meminfo_available_bytes(meminfo_path="/proc/meminfo"):
    """MemAvailable from /proc/meminfo in bytes, or None when it cannot be read."""
    for line in (_read_small_text(meminfo_path) or "").splitlines():
        if line.startswith("MemAvailable:"):
            try:
                return float(line.split()[1]) * 1024.0
            except (IndexError, ValueError):
                return None
    return None


def _own_cgroup_dirs(v1_controller, proc_cgroup="/proc/self/cgroup", mount="/sys/fs/cgroup"):
    """``[(directory, version)]`` of this process's own cgroup and each ancestor below the mount root.

    Innermost first, v2 (the unified ``0::`` line) and the v1 hierarchy carrying ``v1_controller``.
    A cgroup namespace shows ``0::/`` and gives nothing here -- the root files are then this process's
    own, and the callers read those as well. Without a namespace (a Slurm step, ``systemd-run -p
    MemoryMax``) the limit sits on a nested directory such as
    ``system.slice/slurmstepd.scope/job_N``, and the root files say "no limit".
    """
    found = []
    for line in (_read_small_text(proc_cgroup) or "").splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, rel = parts
        if hierarchy == "0" and not controllers:
            base, version = mount, 2
        elif v1_controller in controllers.split(","):
            base, version = os.path.join(mount, v1_controller), 1
        else:
            continue
        steps = [p for p in rel.split("/") if p]
        for depth in range(len(steps), 0, -1):
            found.append((os.path.join(base, *steps[:depth]), version))
    return found


def _own_cgroup_memory_files(proc_cgroup="/proc/self/cgroup", mount="/sys/fs/cgroup"):
    """:data:`_CGROUP_MEMORY_FILES` rows for this process's own cgroup and its ancestors, then the root's.

    The root rows are built under ``mount`` like the rest, so a caller that points ``mount`` at a fixture
    reads no file of the real host (the root rows were the host's own before; review of u29a-mcp-transport-17).
    """
    rows = []
    for d, version in _own_cgroup_dirs("memory", proc_cgroup, mount):
        if version == 2:
            rows.append((f"{d}/memory.max", f"{d}/memory.current", f"{d}/memory.stat", ""))
        else:
            rows.append((f"{d}/memory.limit_in_bytes", f"{d}/memory.usage_in_bytes", f"{d}/memory.stat", "total_"))
    v1 = os.path.join(mount, "memory")
    rows.append((f"{mount}/memory.max", f"{mount}/memory.current", f"{mount}/memory.stat", ""))
    rows.append((f"{v1}/memory.limit_in_bytes", f"{v1}/memory.usage_in_bytes", f"{v1}/memory.stat", "total_"))
    return tuple(rows)


def _cgroup_memory_room_bytes(files=None):
    """Room left under this cgroup's memory limit, or None when no limit is set or none can be read.

    ``memory.current`` (v2) and ``memory.usage_in_bytes`` (v1) count the page cache, and a
    memory-limited container sits near its limit on cache alone after any file I/O. The kernel reclaims
    both file LRU lists (``active_file``, ``inactive_file`` in ``memory.stat``) before it OOM-kills
    anything in the cgroup, so the working set is usage minus those and the room is the limit minus the
    working set. When the file LRU counters cannot be read, the cache cannot be told apart, so usage is
    not subtracted at all and the room is the limit.

    ``files`` defaults to this process's own cgroup and every ancestor, then the root files, and the
    room is the tightest over every level that sets a limit. Reading the root files alone missed a
    Slurm or systemd job limit on a host without a cgroup namespace -- the root says "no limit", the
    host's MemAvailable stood in, and a run the job could not hold was OOM-killed instead of refused
    (hunt 2026-09-30, u29a-mcp-transport-17).
    """
    if files is None:
        files = _own_cgroup_memory_files()
    rooms = []
    for limit_file, usage_file, stat_file, stat_prefix in files:
        raw = (_read_small_text(limit_file) or "").strip()
        if not raw.isdigit():  # absent, or "max": no limit at this level
            continue
        limit = float(raw)
        if not 0 < limit < _NO_CGROUP_LIMIT:
            continue
        stat = {}
        for line in (_read_small_text(stat_file) or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                stat[parts[0]] = float(parts[1])
        file_lru = [stat.get(stat_prefix + k, stat.get(k)) for k in ("active_file", "inactive_file")]
        used = (_read_small_text(usage_file) or "").strip()
        if not used.isdigit() or all(v is None for v in file_lru):
            rooms.append(limit)
            continue
        working_set = max(float(used) - sum(v for v in file_lru if v is not None), 0.0)
        rooms.append(max(limit - working_set, 0.0))
    return min(rooms) if rooms else None


def available_memory_bytes():
    """Memory this process can still allocate, in bytes, or None if nothing can be read.

    The smaller of ``MemAvailable`` and the room under the cgroup memory limit, with the cgroup's page
    cache counted as reclaimable. Workers that refuse a run up front because an intrinsic dense
    intermediate would not fit use this one reader, so a container at its limit on page cache alone is
    not mistaken for a full one (2026-09 verification found four workers that were).
    """
    found = [v for v in (_meminfo_available_bytes(), _cgroup_memory_room_bytes()) if v is not None]
    return min(found) if found else None


def _cgroup_cpu_quota(proc_cgroup="/proc/self/cgroup", mount="/sys/fs/cgroup") -> float | None:
    """CPU quota this cgroup is allowed, in whole CPUs, or None if unlimited/unreadable.

    The tightest quota over this process's own cgroup, its ancestors and the root files -- the same
    walk, and for the same reason, as :func:`_cgroup_memory_room_bytes` (hunt 2026-09-30,
    u29a-mcp-transport-17): a Slurm job's CPU allocation sits on its own directory, not the root.
    """
    levels = _own_cgroup_dirs("cpu", proc_cgroup, mount) + [(mount, 2), (os.path.join(mount, "cpu"), 1)]
    quotas = []
    for d, version in levels:
        try:
            if version == 2:
                text = (_read_small_text(os.path.join(d, "cpu.max")) or "").split()
                if len(text) == 2 and text[0] != "max" and float(text[1]) > 0:
                    quotas.append(float(text[0]) / float(text[1]))
            else:
                quota = float((_read_small_text(os.path.join(d, "cpu.cfs_quota_us")) or "-1").strip())
                period = float((_read_small_text(os.path.join(d, "cpu.cfs_period_us")) or "0").strip())
                if quota > 0 and period > 0:
                    quotas.append(quota / period)
        except ValueError:
            continue
    return min(quotas) if quotas else None


def cpu_budget(cap: int | None = None, reserve: int = 0, minimum: int = 1) -> int:
    """How many CPUs this process may actually use.

    ``os.cpu_count()`` reports the *machine*. Under a CPU affinity mask or a cgroup quota
    (container, Slurm, k8s) the process gets far fewer, and sizing a worker pool from the machine
    count re-creates the oversubscription that commit d1689ee had to fix for RCTD -- N workers each
    believing they own the whole box. Take the smallest of every limit that is actually in force.

    ``reserve`` leaves headroom for the parent process; ``cap`` bounds pools whose per-worker memory
    is the real constraint; ``minimum`` guarantees a runnable answer (a pool of 0 does nothing).
    """
    allowance = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            allowance = min(allowance, len(os.sched_getaffinity(0)) or allowance)
        except Exception:
            pass
    quota = _cgroup_cpu_quota()
    if quota is not None and quota > 0:
        allowance = min(allowance, int(quota))  # floored: 2.5 CPUs of quota runs 2 workers, not 3
    allowance -= max(0, int(reserve))
    if cap is not None:
        allowance = min(allowance, int(cap))
    return max(int(minimum), allowance)


# The BLAS/OpenMP thread-count family. All three, not just OPENBLAS_NUM_THREADS: which one is
# consulted depends on whether the child's numpy/R links OpenBLAS, MKL or a plain libgomp, and
# pinning only one leaves the oversubscription reachable through the other two.
_BLAS_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def pin_blas_threads(n_threads: int = 1) -> None:
    """Cap each BLAS thread pool in this process *and its children* at ``n_threads``.

    For a tool that spawns its own pool of parallel workers, the two layers multiply: commit
    d1689ee measured RCTD's 4 R workers x 96 cores of OpenBLAS as 384 threads and a load average of
    ~328, at which point the tool stalled in its init phase and the agent timed it out. Pinning the
    inner layer to one thread makes the tool's own parallelism the only parallelism.

    Must be called *before* the child process starts: OpenBLAS and MKL read these at load time, so
    a ``Sys.setenv`` inside R or an assignment after ``import numpy`` is too late. ``setdefault``
    throughout -- an operator who set a thread count deliberately keeps it.
    """
    for var in _BLAS_THREAD_VARS:
        os.environ.setdefault(var, str(max(1, int(n_threads))))


def resolve_compute(
    request: Any = None,
    *,
    cap: int | None = None,
    reserve: int = 0,
    quiet: bool = False,
) -> Compute:
    """Turn any of the fleet's device vocabularies into a ``(device, n_threads)`` pair.

    Accepts ``'cpu'``/``'CPU'``/``'cuda'``/``'CUDA'``/``'gpu'``/``'GPU'``/``'cuda:1'``/``'auto'``,
    ``True``/``False`` (``use_gpu``), and an int index (``0`` = first GPU, ``-1`` = CPU, the
    convention documented at ``spaceflow_worker.py:240``). ``None``/``'auto'``/``''`` follow the
    hardware.

    The returned device is always one ``torch.device()`` parses, and is never a CUDA device unless
    CUDA is really present -- a GPU request on a CPU-only box degrades to ``'cpu'`` and says so,
    rather than dying inside the model. ``n_threads`` is :func:`cpu_budget`.
    """
    index: str = ""
    if request is None:
        want = "auto"
    elif isinstance(request, bool):  # before int: bool IS an int, and False means CPU, not GPU 0
        want = "gpu" if request else "cpu"
    elif isinstance(request, int):
        want, index = ("gpu", f":{request}") if request >= 0 else ("cpu", "")
    else:
        text = str(request).strip().lower()
        matched = _INDEXED_GPU.match(text)
        if _INTEGER.match(text):
            # Every knob crosses a CLI/JSON boundary as text, so '-1'/'0' must mean what the int
            # form means (spaceflow_worker.py:240: -1 => cpu, >=0 => that GPU index).
            number = int(text)
            want, index = ("gpu", f":{number}") if number >= 0 else ("cpu", "")
        elif matched:
            want, index = "gpu", f":{matched.group(1)}"
        elif text in _CPU_WORDS:
            want = "cpu"
        elif text in _GPU_WORDS:
            want = "gpu"
        elif text in _AUTO_WORDS:
            want = "auto"
        elif _TORCH_DEVICE.match(text):
            want = text  # a device torch understands that is neither CPU nor CUDA (e.g. 'mps')
        else:
            want = "auto"
            if not quiet:
                print(
                    f"[worker_utils] unrecognised device request {request!r}; choosing automatically", file=sys.stderr
                )

    if want in ("gpu", "auto"):
        if _cuda_available():
            device = f"cuda{index}"
        else:
            device = "cpu"
            if want == "gpu" and not quiet:
                print(
                    f"[worker_utils] {request!r} requested but no CUDA device is available; using CPU", file=sys.stderr
                )
    else:
        device = "cpu" if want == "cpu" else want

    return Compute(device=device, n_threads=cpu_budget(cap=cap, reserve=reserve))


def env_bin(name: str) -> str:
    """Resolve a console script *name* to the copy in the env this worker is running in.

    Workers that shell out to a CLI -- STRIDE's deconvolution, BayesTME's marker-gene step -- named
    theirs by the absolute path of the box they were written on
    (``/opt/conda/envs/stride/bin/STRIDE``). Anywhere the fleet is deployed under a different conda
    root that path does not exist, and ``subprocess.run`` raises ``FileNotFoundError``: an uncaught
    traceback rather than the worker's structured error JSON, and for STRIDE that is the entire
    analysis rather than an optional extra.

    The interpreter has already been resolved by the time a worker runs. The portal calls
    ``base_mcp.get_worker_paths``, which honours the ``<TOOL>_PYTHON`` the setup wizard writes and
    re-roots an absent ``.../envs/<name>/bin/python`` onto the live conda root -- so
    ``sys.executable`` names the env we are actually in, on whatever box that is. A sibling of it
    inherits every one of those repairs for free, which is why this resolves against the running
    interpreter rather than re-deriving a root of its own. ``cytospace_worker`` already did it this
    way; this is that pattern, shared.

    Order: the sibling of ``sys.executable`` if it exists, then ``PATH`` (a site install of a tool
    whose conda recipe did not ship the script), then the sibling path regardless -- so a genuinely
    missing binary produces an error naming the env the user has, not a directory from a machine
    they have never seen.
    """
    sibling = os.path.join(os.path.dirname(sys.executable), name)
    if os.path.exists(sibling):
        return sibling
    return shutil.which(name) or sibling


def ensure_r_home(explicit: str | None = None, env_var: str | None = None) -> str | None:
    """Point ``R_HOME`` at the R this box actually has, and return it (``None`` if there is none).

    Workers that call into R through rpy2 -- ``mclust`` clustering, mostly -- have to set this
    before the first ``import rpy2``. Two of them spelled the answer as a literal from the machine
    they were written on (``/opt/conda/envs/<tool>/lib/R``), which is wrong anywhere the fleet is
    deployed under a different conda root, and one of those literals does not exist even here.

    A wrong value is worse than no value. rpy2 consults ``R_HOME`` first and falls back to its own
    detection only when it is unset, so naming a directory that is not an R installation
    *suppresses* the fallback -- a box with a perfectly good R then fails to initialise, and the
    user reads it as "mclust failed". That is why nothing is set when nothing resolves, and why
    this never raises: refusing to run would turn a box where rpy2 would have succeeded into one
    that cannot.

    Priority, following ``graphst_worker._set_r_home_if_needed``, which had it right:

    1. an ``R_HOME`` already in the environment -- ``conda run -n <env>`` sets one, so does a site
       install, and the caller knows this box better than we do;
    2. ``explicit``, for a worker with a ``--r-home`` flag;
    3. ``os.environ[env_var]``, for a worker with its own override variable;
    4. ``<sys.prefix>/lib/R`` -- the worker's own env, which is where the R its recipe installed
       lives, and the only one of these that is right by construction rather than by luck;
    5. ``R RHOME``, for a site R outside any conda prefix (the last resort the tool-creation
       playbook documents).

    Candidates that are not directories are skipped rather than trusted, so a stale override cannot
    shadow a working auto-detection.
    """
    existing = os.environ.get("R_HOME")
    if existing:
        return existing

    candidates = [explicit, os.environ.get(env_var) if env_var else None, os.path.join(sys.prefix, "lib", "R")]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            os.environ["R_HOME"] = candidate
            return candidate

    probed = _probe_r_home()
    if probed:
        os.environ["R_HOME"] = probed
        return probed
    return None


def _probe_r_home() -> str | None:
    """Ask the ``R`` on ``PATH`` where it lives, or ``None`` if there isn't one that answers."""
    import subprocess

    if not shutil.which("R"):
        return None
    try:
        out = subprocess.run(["R", "RHOME"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    home = (out.stdout or "").strip().splitlines()
    candidate = home[-1].strip() if home else ""
    return candidate if candidate and os.path.isdir(candidate) else None


# ---------------------------------------------------------------------------------------------
# Honest-payload helpers: one vocabulary for "what ran", "what was ignored", "what was dropped".
# Every worker that keeps a substitute method, accepts a knob it cannot use, or has to remove
# unlabelled observations says so through these, so the payload keys mean the same thing in
# every tool (``params.method``, ``params.used_fallback``, ``params.ignored``).
# ---------------------------------------------------------------------------------------------

#: The package ``scanpy.pp.highly_variable_genes(flavor="seurat_v3")`` needs and several tool envs
#: lack. Naming it beats switching flavour behind the caller's back.
_HVG_SEURAT_V3_PACKAGE = "scikit-misc (import name skmisc)"


def record_method(out: WorkerOutput, method: str, used_fallback: bool = False, why: str = "") -> WorkerOutput:
    """Record which method produced the result.

    ``method`` is the honest name of what ran (e.g. ``"PROST PNN (kmeans init)"`` or
    ``"spectral clustering on coordinates (PROST unavailable)"``). ``used_fallback`` is True only
    when a substitute ran in place of the advertised method *because the caller allowed it*; a
    substitute that is the tool's only implementation is not a fallback, it is the method.
    """
    out.add_params({"method": str(method), "used_fallback": bool(used_fallback)})
    if used_fallback:
        out.add_warning(f"fallback ran: {method}" + (f" -- {why}" if why else ""))
    return out


def keep_in_tissue(adata, what="spots"):
    """``(adata restricted to obs['in_tissue'] == 1, n_supplied, n_dropped)``.

    Space Ranger folders hold in-tissue spots only, but CELLxGENE Visium exports carry every array
    spot with ``obs['in_tissue']`` 0/1: on the library's four such samples 56-70% of the spots are
    background glass. Workers that cluster, score or map spots call this right after loading so the
    background is not analysed as tissue (``scanpy_spatial`` has always done so). No column, or a
    column that is 1 everywhere, returns the object unchanged. True/"1"/1 count as in tissue; a column
    with no in-tissue spot at all is refused rather than analysed as empty.
    """
    n = int(adata.n_obs)
    if "in_tissue" not in adata.obs.columns:
        return adata, n, 0
    import numpy as np
    import pandas as pd

    raw = adata.obs["in_tissue"]
    flag = pd.to_numeric(raw.astype(str).str.strip().str.lower().replace({"true": "1", "false": "0"}), errors="coerce")
    keep = np.asarray(flag == 1)
    n_keep = int(keep.sum())
    if n_keep == n:
        return adata, n, 0
    if n_keep == 0:
        seen = sorted({str(v) for v in raw.unique()})[:8]
        raise ValueError(
            f"obs['in_tissue'] marks none of the {n} {what} as in tissue (values seen: {seen}); "
            "fix the column so in-tissue spots are 1, or remove it if every spot is tissue."
        )
    return adata[keep].copy(), n, n - n_keep


def record_in_tissue(out, n_supplied, n_dropped, what="spots"):
    """Report what :func:`keep_in_tissue` left out: ``params.in_tissue_filter`` and a warning."""
    if not n_dropped:
        return
    out.add_params(
        {
            "in_tissue_filter": {
                f"n_{what}_supplied": int(n_supplied),
                f"n_{what}_off_tissue_dropped": int(n_dropped),
                f"n_{what}_used": int(n_supplied - n_dropped),
            }
        }
    )
    out.add_warning(
        f"{n_dropped} of {n_supplied} {what} have obs['in_tissue'] == 0 (background outside the tissue) and "
        f"were left out; {n_supplied - n_dropped} in-tissue {what} were analysed."
    )


def expression_matrix_kind(X, chunk_rows=4096):
    """What a matrix holds, read from every stored value in row blocks (sparse-aware, never densified whole).

    ``'counts'`` (finite, non-negative, integer-valued), ``'nonnegative_noninteger'`` (normalised or
    log-transformed data), ``'negative'`` (scaled / z-scored data), ``'nonfinite'`` or ``'empty'``.
    """
    import numpy as np

    n = int(X.shape[0])
    kind = "empty"
    for start in range(0, n, chunk_rows):
        block = X[start : min(start + chunk_rows, n)]
        values = block.data if hasattr(block, "data") and hasattr(block, "indptr") else np.asarray(block)
        values = np.asarray(values).ravel()
        if values.size == 0:
            continue
        if not np.all(np.isfinite(values)):
            return "nonfinite"
        if np.any(values < 0):
            return "negative"
        if kind != "nonnegative_noninteger" and not np.allclose(values, np.round(values), rtol=0.0, atol=1e-3):
            kind = "nonnegative_noninteger"
        elif kind == "empty":
            kind = "counts"
    return kind


def choose_counts_matrix(adata, use_raw_counts=False):
    """``(adata, info)`` -- the matrix a counts-based pipeline should run on, and what was decided.

    CELLxGENE exports can hold a processed X with the counts in ``adata.raw.X``: on the library,
    Muscle's X is log-normalised and Skin's is z-scored. A pipeline that normalises X as counts then
    normalises Muscle twice without a word and turns Skin into NaN. The rule, shared by every worker
    that calls this:

    * ``use_raw_counts=True`` runs on ``adata.raw.X`` (obs/obsm/uns kept) and refuses when there is
      no ``adata.raw`` or it does not hold counts;
    * otherwise X is used; negative or non-finite values are refused (they are not counts and the
      pipeline would fail opaquely), naming ``use_raw_counts`` when ``adata.raw`` holds counts;
      non-negative non-integer values run as before with a warning in ``info``.
    """
    import anndata as ad

    raw = getattr(adata, "raw", None)
    raw_kind = expression_matrix_kind(raw.X) if raw is not None else None
    hint = (
        " adata.raw holds raw counts: pass use_raw_counts=True to run on them."
        if raw_kind == "counts"
        else " Supply an h5ad whose X (or adata.raw with use_raw_counts=True) holds raw counts."
    )
    if use_raw_counts:
        if raw is None:
            raise ValueError("use_raw_counts=True runs on adata.raw, and this h5ad has no adata.raw.")
        if raw_kind != "counts":
            raise ValueError(
                f"use_raw_counts=True, but adata.raw.X holds {raw_kind.replace('_', ' ')} values, not counts."
            )
        chosen = ad.AnnData(
            X=raw.X, obs=adata.obs.copy(), var=raw.var.copy(), obsm=dict(adata.obsm), uns=dict(adata.uns)
        )
        return chosen, {"expression_source": "raw.X", "x_matrix_kind": expression_matrix_kind(adata.X), "warning": None}
    x_kind = expression_matrix_kind(adata.X)
    if x_kind in ("negative", "nonfinite"):
        what = "negative values (scaled or z-scored data)" if x_kind == "negative" else "NaN or infinite values"
        raise ValueError(f"X holds {what}, not counts, and this tool normalises X as counts." + hint)
    warning = None
    if x_kind == "nonnegative_noninteger":
        warning = (
            "X holds non-integer values (normalised or log-transformed data?), and this tool normalises X as "
            "counts, so the result was computed on a matrix normalised twice." + hint
        )
    return adata, {"expression_source": "X", "x_matrix_kind": x_kind, "warning": warning}


def record_expression_source(out, info):
    """``params.expression_source`` / ``params.x_matrix_kind`` and the warning from :func:`choose_counts_matrix`."""
    out.add_params({"expression_source": info["expression_source"], "x_matrix_kind": info["x_matrix_kind"]})
    if info.get("warning"):
        out.add_warning(info["warning"])


def record_ignored(out: WorkerOutput, names, why: str) -> WorkerOutput:
    """Record parameters that were accepted but had no effect, and say why.

    Kept-but-ignored is the honest shape for a knob the method cannot use: removing it would break
    every caller, and silently echoing it in ``params`` says it was applied when it was not.
    """
    names = [names] if isinstance(names, str) else [str(n) for n in names]
    if not names:
        return out
    ignored = out._params.setdefault("ignored", [])
    for name in names:
        if name not in ignored:
            ignored.append(name)
    out.add_warning(f"ignored parameter(s) {', '.join(names)}: {why}")
    return out


def drop_unlabeled(labels, allow_drop: bool, what: str = "observations"):
    """Split a label vector into (keep_mask, n_dropped) around missing labels.

    A NaN / None / "" / "nan" label is not a class. The squidpy niche tools crashed on them with a
    bare ``KeyError: nan``; DeepLinc cast them to the string ``"nan"`` and scored it as a cell
    type. With ``allow_drop`` False (every tool's default) a missing label is an error that names
    the count and the knob; with it True the rows are dropped and the count is returned so the
    caller can report it.

    pandas' own missing markers count too: ``pd.NA`` and ``pd.NaT`` (a nullable string or
    categorical column hands them over as objects), and ``"<NA>"``, which is what ``str(pd.NA)``
    turns into once a column has been cast to str. pandas is never imported here (Python 3.7
    workers import this module without it): an ``NA`` object can only exist if pandas is loaded.
    """
    import numpy as np

    pd = sys.modules.get("pandas")
    na_objects = tuple(x for x in (getattr(pd, "NA", None), getattr(pd, "NaT", None)) if x is not None)

    def _is_missing(v):
        if v is None or any(v is na for na in na_objects):
            return True
        if isinstance(v, float) and v != v:
            return True
        return str(v).strip().lower() in ("", "nan", "none", "na", "<na>")

    values = np.asarray(labels, dtype=object)
    missing = np.array([_is_missing(v) for v in values], dtype=bool)
    n_missing = int(missing.sum())
    if n_missing and not allow_drop:
        raise ValueError(
            f"{n_missing} of {len(values)} {what} have no label (NaN/empty). Pass drop_unlabeled=True to leave them "
            "out, or label them first; a missing label is not a class."
        )
    return ~missing, n_missing


def require_hvg_flavor(flavor: str, alternative: str | None = "seurat") -> str:
    """Check that an HVG flavour's dependency is importable *before* scanpy is asked for it.

    ``seurat_v3`` needs scikit-misc; when it is absent scanpy raises inside the call and several
    workers used to catch that and run ``flavor="seurat"`` on log data instead, saying nothing.
    The caller chose a flavour; if it cannot run, the run stops and the message names the package.

    ``alternative`` is the flavour the message offers through the tool's ``hvg_flavor`` parameter.
    ``None`` is for a tool that fixes its flavour and has no such parameter: the message then offers
    only the install, because telling the agent to pass an argument the tool rejects costs it a turn
    (hunt 2026-09-30, u29a-mcp-transport-15).
    """
    if flavor == "seurat_v3":
        if alternative is None:
            subject = f"This tool selects highly variable genes with flavor='{flavor}', which needs"
            knob = "; this tool has no parameter that selects another flavour."
        else:
            subject = f"hvg_flavor='{flavor}' needs"
            knob = f", or pass hvg_flavor='{alternative}' (log-data dispersion) explicitly."
        try:
            import skmisc  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"{subject} {_HVG_SEURAT_V3_PACKAGE}, which is not installed in this environment. Install it" + knob
            ) from exc
        except Exception as exc:
            # Installed but unimportable: a ~/.local scikit-misc built against numpy 1.x raises
            # ValueError ("numpy.dtype size changed") in a numpy-2 env, not ImportError.
            raise ImportError(
                f"{subject} {_HVG_SEURAT_V3_PACKAGE}, which is installed but cannot be imported in this "
                f"environment ({type(exc).__name__}: {exc}). Repair it" + knob
            ) from exc
    return flavor


def build_cluster_analysis(
    cluster_sizes: dict[Any, int],
    cluster_key: str = "cluster",
    total_spots: int | None = None,
    n_requested: int | None = None,
) -> str:
    """
    Generate human-readable analysis text for clustering results.

    Pass ``n_requested`` when the caller asked for a specific number of clusters. Resolution-based
    methods (Leiden/Louvain) cannot hit an arbitrary k -- deepstkit, for instance, searches for a
    resolution yielding k and silently falls back to 1.0 when none exists -- so the count returned
    may not be the count requested. Saying so is the difference between "the data supports 6
    domains" and "we asked for 5 and got 6".

    A **met** request is disclosed too, which is the less obvious half. This text is what the agent
    quotes to the user, and "Found 7 domains" reads as a discovery even when 7 was the caller's own
    input -- a live run asked "how many distinct regions does the data support", passed
    ``--n-domains 7``, watched the resolution sweep hit 7, and reported back "the data supports 7
    distinct regions". Grounding rules cannot catch that: the agent was faithfully quoting an
    observation, and the observation was the thing making the false claim. So when the count came
    from the caller, this text says so and declines the word "Found".

    Only the framing moves. Every measured sentence -- sizes, percentages, balance ratio -- is
    byte-identical whatever ``n_requested`` is, and no scoring path reads this string.

    Example output:
        "Found 5 spatial domains. Largest: domain 2 (312 spots, 31.2%).
         Smallest: domain 4 (45 spots, 4.5%). Distribution is moderately balanced."
    """
    if not cluster_sizes:
        return "No clusters found."

    n = len(cluster_sizes)
    total = total_spots or sum(cluster_sizes.values())
    sorted_sizes = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    largest_id, largest_n = sorted_sizes[0]
    smallest_id, smallest_n = sorted_sizes[-1]

    largest_pct = largest_n / total * 100 if total > 0 else 0
    smallest_pct = smallest_n / total * 100 if total > 0 else 0
    ratio = largest_n / smallest_n if smallest_n > 0 else float("inf")

    if ratio < 2:
        balance = "well-balanced"
    elif ratio < 5:
        balance = "moderately balanced"
    else:
        balance = "imbalanced"

    request_met = n_requested is not None and n_requested == n
    headline = (
        f"Partitioned {total} spots into the {n} {cluster_key}s that were requested."
        if request_met
        else f"Found {n} {cluster_key}s across {total} spots."
    )
    lines = [
        headline,
        f"Largest: {cluster_key} {largest_id} ({largest_n} spots, {largest_pct:.1f}%).",
        f"Smallest: {cluster_key} {smallest_id} ({smallest_n} spots, {smallest_pct:.1f}%).",
        f"Size distribution is {balance} (max/min ratio: {ratio:.1f}x).",
    ]
    if request_met:
        lines.append(
            f"NOTE: {n} was supplied by the caller, so it is an input and not a finding; this run "
            f"does not show that the data supports {n} {cluster_key}s. Answering "
            f"'how many {cluster_key}s are there' from this number would be circular -- it would "
            f"need a sweep over candidate counts scored by some criterion."
        )
    elif n_requested is not None:
        lines.append(
            f"NOTE: {n_requested} {cluster_key}s were requested but {n} were produced; "
            f"this method cannot guarantee an exact count, so treat {n} as the method's own "
            f"answer rather than the requested partition."
        )
    return " ".join(lines)


def distinct_significant_genes(
    gene_names: Iterable[Any],
    keep: Iterable[Any] | None = None,
) -> list[str]:
    """Distinct gene names in first-occurrence order, optionally masked by ``keep``.

    An SVG result table has one row per (gene, fitted kernel), so a gene whose kernels tie on
    likelihood occupies several rows. Counting rows then overstates how many genes were found --
    SOMDE reported 151 spatially variable genes out of 150 tested -- and slicing the head for
    "top k" spends a slot on a repeat. Callers pass the result's gene column and a significance
    mask and get genes back.

    First-occurrence order is preserved because these tables arrive sorted by effect size, so the
    order *is* the ranking.
    """
    names = [str(g) for g in gene_names]
    if keep is not None:
        # Indexed rather than zipped: a length mismatch means the caller paired the wrong columns,
        # and zip would silently truncate to the shorter one -- under-reporting significant genes
        # with no sign anything went wrong. (`zip(strict=)` is 3.10+; workers run on 3.9 too.)
        flags = [bool(k) for k in keep]
        if len(flags) != len(names):
            raise ValueError(f"gene_names has {len(names)} entries but keep has {len(flags)}")
        names = [g for i, g in enumerate(names) if flags[i]]

    seen = set()
    distinct = []
    for name in names:
        if name not in seen:
            seen.add(name)
            distinct.append(name)
    return distinct


def make_names_unique_and_report(
    adata: Any, into: dict | None = None, axes: tuple[str, ...] = ("obs", "var")
) -> dict[str, int]:
    """Deduplicate ``obs_names``/``var_names`` and report which identifiers had to be renamed.

    ``*_names_make_unique()`` invents identifiers in silence: a gene symbol present twice becomes
    ``TBCE`` and ``TBCE-1``, a name that exists in no nomenclature and joins against no annotation.
    A duplicated *barcode* is the same defect one axis over: ``AAACGT-1`` becomes ``AAACGT-1-1``,
    the row counts still match, and a join back to the user's own object then drops exactly the
    renamed rows. Ordinary input carries both -- 10x references duplicate symbols, concatenated
    sections duplicate barcodes -- so a name we generated has to be declared rather than published
    as if the user had supplied it.

    Returns ``{"n_genes_renamed": N, "n_cells_renamed": M}`` for the caller to put on its payload,
    logs up to three worked examples per axis on stderr, and records the same counts under
    ``adata.uns["identifier_renames"]`` so a payload built several call hops from the load can still
    reach them. When nothing was renamed it says nothing, so a clean input reads exactly as it did
    before -- and it clears any ``identifier_renames`` the input file arrived with, because a count
    an upstream tool wrote about its own run must not be republished as this one's finding.

    Pass ``into`` to accumulate across more than one deduplication of the same object: several
    workers dedup at load and again after remapping ``var_names`` from a symbol column, which is
    where a collision is most likely. Both the return value and the ``uns`` record then carry the
    total, not the last pass alone.

    Pass ``axes=("var",)`` for the workers that deduplicate gene symbols and deliberately leave the
    cell axis as the user supplied it (ncem, novosparc, stplus). Deduplicating an axis a worker
    does not deduplicate today would change what it publishes for any object with duplicate
    barcodes, so the default stays both axes and the narrowing is opt-in. The unasked-for axis is
    reported as 0, which is a true statement about the run: it renamed nothing there.
    """
    renamed = {"n_genes_renamed": 0, "n_cells_renamed": 0} if into is None else into
    for _key in ("n_genes_renamed", "n_cells_renamed"):
        renamed.setdefault(_key, 0)
    for axis, key, noun in (("obs", "n_cells_renamed", "cell"), ("var", "n_genes_renamed", "gene")):
        if axis not in axes:
            continue
        before = [str(n) for n in getattr(adata, axis + "_names")]
        getattr(adata, axis + "_names_make_unique")()
        after = [str(n) for n in getattr(adata, axis + "_names")]
        # Indexed rather than zipped: make_unique renames in place and keeps the first occurrence,
        # so position i before is position i after. (`zip(strict=)` is 3.10+; workers run on 3.8 too.)
        changed = [(before[i], after[i]) for i in range(len(before)) if before[i] != after[i]]
        renamed[key] += len(changed)
        if changed:
            shown = "; ".join(f"{b!r} -> {a!r}" for b, a in changed[:3])
            more = f" (+{len(changed) - 3} more)" if len(changed) > 3 else ""
            print(
                f"[worker_utils] {len(changed)} duplicate {noun} name(s) were renamed to make them "
                f"unique: {shown}{more}. The renamed identifiers appear in this run's output tables "
                "and exist in no external annotation.",
                file=sys.stderr,
            )
    try:
        if renamed["n_genes_renamed"] or renamed["n_cells_renamed"]:
            adata.uns["identifier_renames"] = dict(renamed)
        else:
            # These workers write annotated h5ad files that are then fed to other tools; a record
            # left by whoever wrote this file describes their run, not ours.
            adata.uns.pop("identifier_renames", None)
    except Exception:
        # uns is unwritable on some backed/view objects; the return value is the primary channel.
        pass
    return renamed


def identifier_rename_params(renamed: dict | None, suffix: str = "") -> dict:
    """The deduplication counts, as payload parameters, under names that say which axis.

    A stable pair of keys whatever happened, so a reader never has to tell "absent" from "nothing
    was renamed", and plain ``int`` values so the dict survives ``json.dumps`` unchanged.

    ``suffix`` keeps two objects apart for the workers that deduplicate a spatial object *and* a
    single-cell reference (or two slices): ``suffix="sc"`` yields ``n_cells_renamed_sc`` /
    ``n_genes_renamed_sc``. Single-object workers pass nothing and publish the same bare names the
    spatially-variable-gene workers already do.
    """
    renamed = renamed or {}
    tail = "_" + suffix if suffix else ""
    return {
        "n_cells_renamed" + tail: int(renamed.get("n_cells_renamed") or 0),
        "n_genes_renamed" + tail: int(renamed.get("n_genes_renamed") or 0),
    }


def identifier_rename_note(renamed: dict | None, subject: str = "") -> str:
    """The deduplication, as one sentence for the analysis prose.

    The stderr line ``make_names_unique_and_report`` writes reaches the server log, not the caller:
    ``base_mcp._parse_result`` attaches ``stderr_tail`` only when the run FAILED, so on a successful
    run -- the run whose results are then used -- nothing tells the reader that some of the row keys
    of the tables below are identifiers we generated. That matters most for the cell axis, because
    those keys are what a user joins on to get back to their own object.

    Empty string when nothing was renamed, so a run that needed no deduplication reads exactly as it
    did before. Deliberately free of the substring ``Spatial:``, which some analysis lines use as a
    field marker.

    The worked example follows the axis that was actually renamed. Illustrating a gene-only rename
    with a Visium barcode reads as though the spots had been renamed, and for the workers that
    deduplicate only ``var_names`` gene-only is the only case there is.
    """
    if not renamed:
        return ""
    n_cells = int(renamed.get("n_cells_renamed") or 0)
    n_genes = int(renamed.get("n_genes_renamed") or 0)
    if not n_cells and not n_genes:
        return ""
    parts = []
    if n_cells:
        parts.append(f"{n_cells} duplicate cell/spot identifier(s)")
    if n_genes:
        parts.append(f"{n_genes} duplicate gene identifier(s)")
    # The same example the SVG prose uses for genes, so the two surfaces read alike.
    example = "a second TBCE becomes TBCE-1" if not n_cells else "a second AAACGT-1 becomes AAACGT-1-1"
    where = f" in the {subject}" if subject else ""
    return (
        f" NOTE: {' and '.join(parts)}{where} were renamed to make them unique ({example}); those "
        "renamed identifiers key this run's outputs and exist in no external annotation, so joining "
        "the results back on the identifiers supplied will not match those rows."
    )


def describe_reduction(noun: str, n_supplied: int, n_used: int, reason: str = "") -> str:
    """One sentence naming both counts, for a cap that dropped part of the user's input.

    A worker that pre-filters genes or subsamples spots announces the cut with a stderr line.
    ``base_mcp._parse_result`` attaches ``stderr_tail`` only when the run FAILED, so on a
    successful run -- the run whose results are then used -- nothing tells the reader that the
    table below covers a fraction of what they supplied. A gene missing from an SVG ranking then
    reads as "tested and not spatially variable" when it was never tested at all.

    Empty string when nothing was dropped, so a run that used the whole input reads exactly as it
    did before: a warning on a complete run trains the reader to skip the one that matters.

    ``reason`` names the cut in the caller's own vocabulary ("the top-2000 highly-variable-gene
    prefilter"), because the remedy -- raise the cap, or accept the subset -- depends on which cut
    it was. Deliberately free of the substring ``Spatial:``, which some analysis lines use as a
    field marker.
    """
    if n_supplied <= 0 or n_used >= n_supplied:
        return ""
    pct = 100.0 * n_used / n_supplied
    because = f" by {reason}" if reason else ""
    return (
        f" NOTE: of the {n_supplied} {noun} supplied, {n_used} ({pct:.1f}%) were analysed; "
        f"{n_supplied - n_used} were dropped before the method ran{because}. The results below "
        f"describe the {n_used} analysed {noun}, not the full input."
    )


def build_svg_analysis(
    n_genes_tested: int,
    n_significant: int | None = None,
    top_genes: list[str] | None = None,
    method_name: str = "SVG analysis",
    n_genes_renamed: int = 0,
) -> str:
    """
    Generate human-readable analysis text for spatially variable gene results.

    ``n_significant`` is how many genes passed the method's significance threshold. Pass
    ``None`` for methods that only rank genes by a continuous score and never threshold
    (PROST's index, SpaGFT's fallback, SVGbit's AI/Di). For those, the size of the reported
    top-k is the caller's display preference -- rendering it as a discovery rate turns
    "show me 50 genes" into "50 genes are spatially variable", which is not a finding.

    ``n_genes_renamed`` is what :func:`make_names_unique_and_report` returned for this run. A
    ranked gene list is the whole deliverable here, so if any of those names were generated by
    the deduplicator the prose says so; passing 0 leaves the text byte-identical.

    Example output:
        "SpaGFT identified 156 spatially variable genes out of 2000 tested (7.8%).
         Top SVGs: GFAP, AQP4, MBP, PLP1, SNAP25."
        "PROST ranked 362 genes by spatial score; the 20 highest-scoring are reported.
         PROST applies no significance threshold, so this is a ranking, not a discovery
         rate. Top SVGs: IFI6, MX1, OAS1."
    """
    if n_significant is None:
        n_reported = len(top_genes) if top_genes else 0
        lines = [
            f"{method_name} ranked {n_genes_tested} genes by spatial score; the "
            f"{n_reported} highest-scoring are reported. {method_name} applies no "
            f"significance threshold, so this is a ranking, not a discovery rate.",
        ]
    else:
        pct = n_significant / n_genes_tested * 100 if n_genes_tested > 0 else 0
        lines = [
            f"{method_name} identified {n_significant} spatially variable genes "
            f"out of {n_genes_tested} tested ({pct:.1f}%).",
        ]
    if n_genes_renamed:
        # Said before the gene list, because the list is where the invented names appear.
        lines.append(
            f"Note: {n_genes_renamed} gene symbol(s) in this dataset were duplicated and were renamed "
            "to make them unique (e.g. a second TBCE becomes TBCE-1), so any such name below is one "
            "this run generated and will not match an external annotation."
        )
    if top_genes:
        top_str = ", ".join(top_genes[:10])
        lines.append(f"Top SVGs: {top_str}.")
    return " ".join(lines)


def build_deconv_analysis(
    n_celltypes: int,
    dominant_counts: dict[str, int] | None = None,
    total_spots: int | None = None,
    method_name: str = "Deconvolution",
) -> str:
    """
    Generate human-readable analysis text for deconvolution results.

    Example output:
        "Cell2location mapped 12 cell types across 3000 spots.
         Most prevalent: Fibroblast (890 spots dominant), Macrophage (456 spots dominant)."
    """
    lines = [f"{method_name} mapped {n_celltypes} cell types"]
    if total_spots:
        lines[0] += f" across {total_spots} spots."
    else:
        lines[0] += "."

    if dominant_counts:
        sorted_ct = sorted(dominant_counts.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_ct[:3]
        top_str = ", ".join(f"{ct} ({cnt} spots dominant)" for ct, cnt in top_3)
        lines.append(f"Most prevalent: {top_str}.")

    return " ".join(lines)


_IDS_SHOWN = 3
_ROW_NUMBER_RE = re.compile(r"^\d+$")
_ENSEMBL_RE = re.compile(r"^ENS[A-Z]{0,4}[GTP]\d{6,}(\.\d+)?$")
_SUFFIX_RE = re.compile(r"-\d+$")


def _as_str_list(ids: Any) -> list:
    """Coerce anything index-like -- list, ``pd.Index``, ``np.ndarray`` -- to a list of ``str``."""
    try:
        return [str(x) for x in ids]
    except TypeError:  # not iterable; a single value was passed
        return [str(ids)]


def _format_ids(ids: list) -> str:
    if not ids:
        return "<none>"
    shown = ", ".join(f'"{i}"' for i in ids[:_IDS_SHOWN])
    return f"[{shown}{', ...' if len(ids) > _IDS_SHOWN else ''}]"


def _mostly(ids: list, pred: Any, sample: int = 50) -> bool:
    """True when ``pred`` holds for at least 80% of the first ``sample`` IDs.

    Sampled rather than exhaustive so one stray ID (a spike-in among Ensembl genes, say) does not
    suppress an otherwise-correct hint, and so the check stays cheap on a 30k-gene index.
    """
    head = ids[:sample]
    if not head:
        return False
    return sum(1 for i in head if pred(i)) >= max(1, int(0.8 * len(head)))


def _looks_like_positional_index(ids: list) -> bool:
    """Exactly ``0..n-1`` -- the signature of a DataFrame index written out as an ID column."""
    if not ids or not all(_ROW_NUMBER_RE.match(i) for i in ids):
        return False
    nums = sorted(int(i) for i in ids)
    return nums[0] == 0 and nums == list(range(len(nums)))


def _id_mismatch_hint(what: str, a_label: str, a_ids: list, b_label: str, b_ids: list) -> str:
    """Name the specific mistake when the two ID sets betray one, else return ""."""
    sides = ((a_label, a_ids, b_ids), (b_label, b_ids, a_ids))

    for label, ids, other in sides:
        if _looks_like_positional_index(ids) and not _looks_like_positional_index(other):
            return (
                f"Every ID on the {label} side is a row number (0, 1, 2, ...), so an index was "
                f"written where the {what} belong."
            )
    for label, ids, other in sides:
        if _mostly(ids, _ROW_NUMBER_RE.match) and not _mostly(other, _ROW_NUMBER_RE.match):
            return (
                f"The {label} IDs are integers and the other side's are not, so a positional row "
                f"number was probably written where the {what} belong."
            )

    if {i.lower() for i in a_ids} & {i.lower() for i in b_ids}:
        return "The IDs differ only in case."
    if {_SUFFIX_RE.sub("", i) for i in a_ids} & {_SUFFIX_RE.sub("", i) for i in b_ids}:
        return "The IDs match once a trailing '-1'-style suffix is dropped, so one side kept it and the other did not."
    for label, ids, other in sides:
        if _mostly(ids, _ENSEMBL_RE.match) and not _mostly(other, _ENSEMBL_RE.match):
            return (
                f"The {label} IDs are Ensembl accessions and the other side's are symbols; map both to one convention."
            )
    return ""


def id_mismatch_msg(
    what: str,
    a_label: str,
    a_ids: Any,
    b_label: str,
    b_ids: Any,
    n_common: int = 0,
) -> str:
    """Build an actionable abort message for two identifier sets that do not line up.

    A message naming only the two *inputs* is not actionable: the reader cannot tell whether it
    passed row indices, mismatched gene-ID conventions, or barcodes with a different suffix. That
    is not hypothetical -- a live agent hit exactly this against RCTD, guessed, and recorded a plan
    to abandon the tool if the second guess also failed. See
    ``test/test_workers_report_id_mismatches.py``.

    ``what`` names the identifiers ("barcodes", "genes", "spot IDs"); ``n_common`` is the size of a
    non-empty-but-insufficient overlap, so a "too few in common" abort does not claim there were
    none.

    The R workers carry an inlined ``id_mismatch_msg`` of their own -- seventeen copies, because
    each is a standalone Rscript with no shared library to source. It reports the same two ID sets
    in the same order, but its hint is a fixed sentence rather than one derived from the IDs; the
    two are deliberately not identical and no test ties them together.
    """
    a = _as_str_list(a_ids)
    b = _as_str_list(b_ids)
    if n_common:
        lead = f"Only {n_common} matching {what} between {a_label} and {b_label}."
    else:
        lead = f"No matching {what} between {a_label} and {b_label}."
    hint = _id_mismatch_hint(what, a_label, a, b_label, b)
    return (
        f"{lead} {a_label}: {len(a)} IDs {_format_ids(a)}; {b_label}: {len(b)} IDs {_format_ids(b)}. "
        f"{hint + ' ' if hint else ''}The two must use the same identifiers."
    )


def unsupported_choice_msg(param: str, value: Any, valid: Any, extra: str = "") -> str:
    """Reject a caller-supplied value while naming the ones that would have worked.

    A rejection that only echoes the bad value is a dead end. The accepted set lives in the
    ``if param == ...`` chain above the raise, which the caller never sees, and often nowhere else
    either -- five portals with an ``input_mode`` document none of their modes. Naming the
    alternatives turns a wrong guess into a one-turn correction.

    The mode names are not consistent across tools (``visium_10x`` in somde, svgbit and
    spatialprompt; ``visium_h5_spatial`` in bayestme, spaceflow, starfysh and ucdeconvolve), so an
    agent that used one tool correctly will hand its neighbour a value it has never heard of. That
    is the common case this exists for. See ``test/test_workers_name_valid_choices.py``.
    """
    options = _as_str_list(valid)
    lead = f"Unsupported {param}={value!r}."
    if not options:
        return f"{lead}{' ' + extra if extra else ''}".strip()
    hint = ""
    close = difflib.get_close_matches(str(value), options, n=1, cutoff=0.6)
    if close and close[0] != str(value):
        hint = f" Did you mean {close[0]!r}?"
    listed = ", ".join(repr(o) for o in options)
    return f"{lead}{hint} Valid {param} values: {listed}.{' ' + extra if extra else ''}"


def preflight_check(
    inputs: dict[str, str],
    output_dir: str,
    packages: list[str] | None = None,
) -> None:
    """Verify inputs exist, output_dir is writable, and packages are importable.

    Call this at the start of a worker to fail fast with a clear message
    instead of crashing deep inside analysis code.

    Args:
        inputs: Mapping of label -> file_path for required input files.
        output_dir: Directory where outputs will be written.
        packages: Optional list of package names to verify are importable.

    Raises:
        FileNotFoundError: If any input file does not exist.
        PermissionError: If the output directory is not writable.
        ImportError: If any required package cannot be imported.
    """
    import importlib

    # Check input files exist
    for label, path in inputs.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preflight: required input '{label}' not found at: {path}")

    # Ensure output directory exists and is writable
    os.makedirs(output_dir, exist_ok=True)
    try:
        probe = os.path.join(output_dir, f".preflight_probe_{os.getpid()}")
        with open(probe, "w") as f:
            f.write("")
        os.remove(probe)
    except OSError as e:
        raise PermissionError(f"Preflight: output directory is not writable: {output_dir} ({e})") from e

    # Verify required packages are importable
    if packages:
        missing = []
        for pkg in packages:
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            raise ImportError(f"Preflight: required packages not importable: {', '.join(missing)}")


def _try_remap_var_names(adata, col: str) -> bool:
    """
    Try to remap adata.var_names using a symbol column in adata.var.

    Uses numpy arrays for indexing to stay compatible with old pandas (1.1.x).
    Modifies adata in-place. Returns True if any names were changed.
    """
    import numpy as np

    if col not in adata.var.columns:
        return False

    # object dtype, not str: a fixed-width '<U15' copy of Ensembl IDs silently truncated every longer
    # symbol ('ANKRD62P1-PARP4P3' -> 'ANKRD62P1-PARP4') once CELLxGENE's feature_name became a candidate.
    symbols = np.asarray(adata.var[col].astype(str), dtype=object)
    original = np.asarray(adata.var_names, dtype=object)
    new_names = original.copy()

    # Only rename where symbol is non-null and non-empty
    valid = np.array([s is not None and str(s).strip() != "" and str(s) != "nan" and str(s) != "None" for s in symbols])
    if valid.sum() == 0:
        return False

    new_names[valid] = symbols[valid]

    # Handle duplicates: keep Ensembl ID for duplicated entries
    seen = set()
    for i in range(len(new_names)):
        if new_names[i] in seen:
            new_names[i] = original[i]  # revert to original
        seen.add(new_names[i])

    adata.var_names = list(new_names)
    adata.var_names_make_unique()
    return True


def harmonize_gene_ids(adata_sc, adata_st, report: dict | None = None) -> int:
    """
    Harmonize gene identifiers between scRNA and spatial AnnData objects.

    Handles the common case where one dataset uses gene symbols and the other
    uses Ensembl IDs by checking for symbol mapping columns in .var.

    Uses numpy-based indexing for compatibility with old pandas (>=1.1).
    Modifies AnnData objects in-place (renaming var_names).

    The mapping adopted is whichever candidate shares the MOST genes, counting "remap
    neither side" as a candidate. Stopping at the first mapping that shares anything is
    not the same question: a Cell Ranger reference keeps the Ensembl ID wherever the
    symbol column was empty, so a few hundred such rows in a 20,000-gene reference match
    an Ensembl-named slide by accident. That non-empty intersection used to end the search,
    and the caller then deconvolved on those few hundred genes at status ok, because every
    caller guards only ``== 0``.

    Every candidate is measured from the original indices and rolled back unless it wins,
    so a column that maps nothing cannot poison the column tried after it, and a pair that
    already agrees is left byte-identical.

    Pass a dict as ``report`` to learn what was done. It is filled with
    ``n_shared_genes`` / ``n_shared_before_harmonization`` / ``spatial_var_column`` /
    ``sc_var_column`` (the two column keys are None when that side was left alone), and
    ``gene_id_harmonization_note`` turns it into a sentence for the payload's analysis prose.
    The renamed identifiers ARE the deliverable -- proportions rows, the written h5ad's
    var_names, every downstream join -- and a caller that publishes only the count leaves the
    reader unable to tell that their gene names were replaced, or from where. The stderr line
    below is not a substitute: ``base_mcp._parse_result`` attaches ``stderr_tail`` only when
    the run FAILED, so on a successful run it never reaches the payload.

    Returns the number of shared genes after harmonization.
    """
    import numpy as np

    def _fill(n_after: int, n_before: int, st_col, sc_col) -> int:
        if report is not None:
            report["n_shared_genes"] = n_after
            report["n_shared_before_harmonization"] = n_before
            report["spatial_var_column"] = st_col
            report["sc_var_column"] = sc_col
        return n_after

    sc_original = adata_sc.var_names
    st_original = adata_st.var_names

    def restore():
        adata_sc.var_names = sc_original
        adata_st.var_names = st_original

    def n_shared():
        return int(
            len(
                np.intersect1d(
                    np.asarray(adata_sc.var_names, dtype=str),
                    np.asarray(adata_st.var_names, dtype=str),
                )
            )
        )

    baseline = n_shared()

    # Try common symbol columns on both datasets. feature_name: CELLxGENE exports keep Ensembl var_names and
    # the gene symbols in var['feature_name']. It is tried last, so on a tie the earlier columns keep winning.
    symbol_cols = ["SYMBOL", "gene_symbols", "gene_name", "GeneName", "GeneName-2", "feature_name"]
    st_cols = [None] + [c for c in symbol_cols if c in adata_st.var.columns]
    sc_cols = [None] + [c for c in symbol_cols if c in adata_sc.var.columns]

    best = baseline
    best_st_col = None
    best_sc_col = None
    mutated = False

    for st_col in st_cols:
        for sc_col in sc_cols:
            if st_col is None and sc_col is None:
                continue  # the do-nothing candidate; already scored as `baseline`
            if mutated:
                restore()
                mutated = False
            applied = False
            if st_col is not None and _try_remap_var_names(adata_st, st_col):
                applied = mutated = True
            if sc_col is not None and _try_remap_var_names(adata_sc, sc_col):
                applied = mutated = True
            if not applied:
                continue
            n = n_shared()
            # Strictly greater, so a tie leaves the user's own identifiers in place.
            if n > best:
                best, best_st_col, best_sc_col = n, st_col, sc_col

    if mutated:
        restore()

    if best_st_col is None and best_sc_col is None:
        return _fill(baseline, baseline, None, None)

    if best_st_col is not None:
        _try_remap_var_names(adata_st, best_st_col)
    if best_sc_col is not None:
        _try_remap_var_names(adata_sc, best_sc_col)

    where = " and ".join(
        f"{side} var[{col!r}]" for side, col in (("spatial", best_st_col), ("scRNA", best_sc_col)) if col is not None
    )
    print(
        f"[worker_utils] Gene identifiers harmonized from {where}: {baseline} -> {best} shared genes. "
        f"Those renamed identifiers are what every downstream result is keyed by.",
        file=sys.stderr,
    )
    return _fill(best, baseline, best_st_col, best_sc_col)


def gene_id_harmonization_params(report: dict | None) -> dict:
    """The harmonisation facts, keyed so no caller's existing parameter can be overwritten.

    Deliberately NOT ``report`` splatted straight onto the payload: three of the four callers
    already publish ``n_shared_genes``, and destvi's is a different quantity -- the intersection
    left after ITS hvg selection, computed long after harmonisation ran. Splatting would silently
    redefine a key readers already have. The ``gene_ids_`` prefix keeps both numbers, under names
    that say which is which.

    Every value is a plain int, str or None, so the dict survives ``json.dumps`` unchanged.
    """
    report = report or {}
    return {
        "gene_ids_shared_after_harmonization": report.get("n_shared_genes"),
        "gene_ids_shared_before_harmonization": report.get("n_shared_before_harmonization"),
        "gene_ids_spatial_var_column": report.get("spatial_var_column"),
        "gene_ids_sc_var_column": report.get("sc_var_column"),
    }


def gene_id_harmonization_note(report: dict | None) -> str:
    """The disclosure ``harmonize_gene_ids`` writes to stderr, as one sentence for the payload.

    The stderr line reaches the server log, not the caller: ``base_mcp._parse_result`` attaches
    ``stderr_tail`` only when the run FAILED, so on the successful runs -- the ones whose results
    are then used -- nothing tells the reader that every gene name in their object was replaced,
    or from which column. This is the same two-channel disclosure the identifier deduplication
    already keeps (payload counts plus a sentence of prose).

    Empty string when nothing was renamed -- no report, an empty report, or a pair whose
    identifiers already agreed -- so a run that needed no harmonisation reads exactly as it did
    before. Deliberately free of the substring ``Spatial:``, which some analysis lines use as a
    field marker.
    """
    if not report:
        return ""
    st_col = report.get("spatial_var_column")
    sc_col = report.get("sc_var_column")
    if st_col is None and sc_col is None:
        return ""
    where = " and ".join(
        f"the {side} object's var[{col!r}]" for side, col in (("spatial", st_col), ("scRNA", sc_col)) if col is not None
    )
    before = report.get("n_shared_before_harmonization")
    after = report.get("n_shared_genes")
    return (
        f" NOTE: gene identifiers were harmonized from {where}, raising the shared-gene count from "
        f"{before} to {after}; the results below are keyed by those renamed identifiers, not the "
        "ones supplied."
    )


def sanitize_cell_type_names(names, replace_space: bool = True) -> tuple[list[str], dict[str, str]]:
    """The rewrite every deconvolution worker applies to the reference's cell-type names.

    ``/`` is both a path separator and an HDF5 group separator, and a space is awkward as an ``obs``
    column name, so the cell-type labels that came out of the user's reference are rewritten before
    anything is written. That rewrite is correct and has to stay. What was missing is a record of
    what it changed, which is the second return value.

    ``replace_space=False`` is tacco's and tangram's variant, which folds only ``/``. The two are
    kept apart deliberately: unifying them would rename columns in proportion tables those tools
    have already published.

    The mapping holds ONLY the names that actually changed, so ``len(mapping)`` counts renames and
    not cell types. A caller that already has a mapping of its own may hand it to
    ``cell_type_rename_params``/``cell_type_rename_note`` unfiltered -- both drop unchanged entries.
    """
    safe: list[str] = []
    mapping: dict[str, str] = {}
    for name in names:
        original = str(name)
        new = original.replace("/", "_")
        if replace_space:
            new = new.replace(" ", "_")
        safe.append(new)
        if new != original:
            mapping[original] = new
    return safe, mapping


def _changed_cell_type_names(mapping: dict | None) -> dict:
    """Only the entries that are a real rename, whatever shape the caller's mapping is in."""
    return {str(k): str(v) for k, v in (mapping or {}).items() if str(k) != str(v)}


def cell_type_rename_params(mapping: dict | None) -> dict:
    """The cell-type rename count, as a payload parameter.

    A stable key whatever happened, so a reader never has to tell "absent" from "nothing was
    renamed", and a plain ``int`` so the dict survives ``json.dumps`` unchanged. Entries whose name
    did not change are not counted -- some callers build a mapping over every cell type.
    """
    return {"n_cell_types_renamed": len(_changed_cell_type_names(mapping))}


def cell_type_rename_note(mapping: dict | None) -> str:
    """The cell-type rewrite, as one sentence for the analysis prose.

    The rewritten names are the deliverable: they are the column headers of the proportion table and
    they are ``cell_type_names`` on the payload, which is what the caller reads and repeats. Three of
    the workers that do this rewrite record it nowhere; the rest record it only on stderr or in
    ``uns``, and ``base_mcp._parse_result`` attaches ``stderr_tail`` only when the run FAILED -- so on
    a successful run, the run that produced the table, nothing says ``Macrophage_Mono`` is a spelling
    we invented for a reference label that reads ``Macrophage/Mono``.

    Empty string when nothing was rewritten, so a reference whose labels were already safe reads
    exactly as it did before. Deliberately free of the substring ``Spatial:``, which some analysis
    lines use as a field marker.
    """
    changed = _changed_cell_type_names(mapping)
    if not changed:
        return ""
    original, rewritten = next(iter(changed.items()))
    return (
        f" NOTE: {len(changed)} cell-type name(s) from the reference were rewritten to be safe as "
        f"column and HDF5 group names ({original!r} became {rewritten!r}); the proportion table and "
        "the cell-type names "
        "reported here use the rewritten spelling, not the one the reference supplied."
    )
