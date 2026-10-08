"""Self-review library for STCoscientist user-MCP-tool creation.

Before Phase 5 rolls back on a test failure, self-review inspects the
error, classifies it into one of 11 canonical classes, and attempts a
targeted remediation. If the remediation succeeds and the failing
test now passes, creation continues. Otherwise, up to 5 remediations
are tried (total budget 30 min); then the original rollback fires.

OFF by default for now — toggle via `default_config.self_review_enabled = True`
or `SOG_SELF_REVIEW_ENABLED=true`.

Design plan: debug_log/patterns/07_plan_self_review_strategy.md
65 risks addressed there.

Bug mitigations:
- S1: TOTAL cap 5 per phase
- S2: per-env .remediation.lock
- S3: proc.terminate() + kill() cleanup
- S4: network re-try with backoff
- S5: disk-space pre-check
- S7: 30-min total budget
- S8: UNKNOWN-rate telemetry
- S11: rerun from N onwards
- S13: classifier input truncation
- S15: re-run safety audit after code edits
- S16: regex timeout
- S17: self_review_history.json persistence
- S18: blocked-classes opt-out
- S19: remediated_deps accounting
- S20: memory cache invalidation
- S21: classifier priority order
- S23: LC_ALL=C env, quiet conda flags
- S25: explicit priority
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import get_close_matches  # used by WRONG_API remediation
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from filelock import FileLock
    from filelock import Timeout as FileLockTimeout

    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False

LOGGER = logging.getLogger("spatialomicsgym.self_review")

# --- Config-driven caps (S1, S7, S10) ---
DEFAULT_TOTAL_CAP = 5
DEFAULT_TOTAL_BUDGET_SEC = 30 * 60
DEFAULT_CLASS_TIMEOUT_SEC = 600
DEFAULT_NETWORK_RETRIES = 3
DEFAULT_DISK_MIN_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB (S5)
DEFAULT_CLASSIFIER_MAX_INPUT = 10 * 1024  # 10 KB (S13)

# --- Failure classes (priority order; S21) ---
#
# The ORDER is the priority — first match wins.
CLASSES = [
    "MISSING_PY_MODULE",  # F1 — ModuleNotFoundError
    "MISSING_R_PACKAGE",  # F2
    "MISSING_CLI_BINARY",  # F3 — binary not on PATH
    "MISSING_SO_LIB",  # F4 — shared object
    "PIN_INCOMPATIBLE",  # F5
    "WRONG_API",  # F6 — AttributeError on module
    "SCHEMA_VIOLATION",  # F7 — pydantic
    "WORKER_OUTPUT_API",  # F8
    "PATH_NOT_FOUND",  # F9 — FileNotFoundError
    "UNKNOWN_MCP_TOOL",  # F11 — FastMCP router miss
    "DISK_EXHAUSTED",  # F12 — no space
    "NETWORK_TRANSIENT",  # F13 — flaky network
    "UNKNOWN",  # fallback
]

BLOCKED_CLASSES_ENV = "SOG_SELF_REVIEW_BLOCKED_CLASSES"


@dataclass
class RemediationContext:
    tool_id: str
    env_name: str
    source_url: str
    worker_path: Path
    server_path: Path
    language: str
    knowledge_dir: Path
    # Runtime tracking (S1, S7)
    total_remediations: int = 0
    total_budget_remaining_sec: float = DEFAULT_TOTAL_BUDGET_SEC
    class_history: list[str] = field(default_factory=list)


def _enabled() -> bool:
    """Check if self-review is enabled.

    The explicit spellings mirror ``config._env_bool`` (strip, lower; true/1/yes/on and
    false/0/no/off) because the setup gates that decide whether to call into this module compare
    against that same set -- a narrower set here would make those gates open on a value this
    function then reads as off, reporting an attempted repair that returned instantly.

    Only a blank or unrecognised value falls through to the config attribute, and that fallthrough
    cannot stand in for a missing spelling: ``default_config`` is built at import, so a flag set
    after ``spatialomicsgym.config`` was first imported (the CLI, the web UI and the agent all
    import it early) reads back as whatever the process started with.
    """
    raw = os.environ.get("SOG_SELF_REVIEW_ENABLED") or os.environ.get("BIOMNI_SELF_REVIEW_ENABLED") or ""
    env_v = raw.strip().lower()
    if env_v in ("true", "1", "yes", "on"):
        return True
    if env_v in ("false", "0", "no", "off"):
        return False
    try:
        from spatialomicsgym.config import default_config

        return bool(getattr(default_config, "self_review_enabled", False))
    except Exception:
        return False


def _blocked_classes() -> set[str]:
    """Classes the user has opted out of (S18)."""
    env_v = os.environ.get(BLOCKED_CLASSES_ENV, "")
    from_env = {c.strip() for c in env_v.split(",") if c.strip()}
    try:
        from spatialomicsgym.config import default_config

        from_cfg = set(getattr(default_config, "self_review_blocked_classes", []) or [])
    except Exception:
        from_cfg = set()
    return from_env | from_cfg


def _disk_free_bytes(path: Path) -> int:
    try:
        stat = os.statvfs(str(path))
        return stat.f_bavail * stat.f_frsize
    except Exception:
        return 1 << 40  # assume plenty


def _safe_re_search(pattern: str, text: str, flags: int = 0) -> re.Match | None:
    """Regex search with input-length cap and fallback (S13, S16)."""
    if not text:
        return None
    if len(text) > DEFAULT_CLASSIFIER_MAX_INPUT:
        text = text[-DEFAULT_CLASSIFIER_MAX_INPUT:]
    try:
        return re.search(pattern, text, flags)
    except Exception:
        return None


def classify_failure(error_msg: str, stdout: str = "", stderr: str = "") -> str:
    """Classify a failure into one of 13 canonical classes (priority order).

    Returns one of CLASSES. Never raises.
    """
    try:
        combined = (error_msg or "") + "\n" + (stdout or "") + "\n" + (stderr or "")
        if _safe_re_search(r"(?i)no space left on device|ENOSPC|disk quota exceeded", combined):
            return "DISK_EXHAUSTED"
        if _safe_re_search(
            r"(?i)temporary failure in name resolution|connection reset|"
            r"read timed out|connectionerror|could not resolve host|"
            r"http.*(503|504|502|429)",
            combined,
        ):
            return "NETWORK_TRANSIENT"
        if _safe_re_search(r"No module named ['\"]([^'\"]+)['\"]", combined):
            return "MISSING_PY_MODULE"
        if _safe_re_search(r"ModuleNotFoundError", combined):
            return "MISSING_PY_MODULE"
        if _safe_re_search(r"there is no package called ['\"]([^'\"]+)['\"]", combined):
            return "MISSING_R_PACKAGE"
        if _safe_re_search(r"No such file or directory: ['\"]([^'\"]+)['\"]", combined):
            # MISSING_CLI_BINARY if the missing path is a bare command name
            m = _safe_re_search(r"No such file or directory: ['\"]([^'\"/]+)['\"]", combined)
            if m:
                return "MISSING_CLI_BINARY"
            return "PATH_NOT_FOUND"
        if _safe_re_search(r"unable to load shared object|cannot find -l|libsomething\.so", combined):
            return "MISSING_SO_LIB"
        if _safe_re_search(r"No matching distribution found for .+==", combined):
            return "PIN_INCOMPATIBLE"
        if _safe_re_search(r"AttributeError: module .* has no attribute", combined):
            return "WRONG_API"
        if _safe_re_search(r"(?i)pydantic.*validation error|validation errors? for", combined):
            return "SCHEMA_VIOLATION"
        if _safe_re_search(r"'WorkerOutput' object has no attribute", combined):
            return "WORKER_OUTPUT_API"
        if _safe_re_search(r"Unknown tool: ['\"]([^'\"]+)['\"]", combined):
            return "UNKNOWN_MCP_TOOL"
        if _safe_re_search(r"FileNotFoundError", combined):
            return "PATH_NOT_FOUND"
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _run_subprocess(cmd: list[str], timeout: int = 600, extra_env: dict | None = None) -> tuple[int, str, str]:
    """Run a subprocess with stability controls (S3, S23, S25)."""
    env = dict(os.environ)
    env.setdefault("LC_ALL", "C")  # S25
    env.setdefault("PYTHONUNBUFFERED", "1")  # S23
    env.setdefault("CONDA_ALWAYS_YES", "true")  # S23
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        # S3 — terminate then kill
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        return -1, "", f"timeout after {timeout}s"


def _with_network_retries(fn: Callable, *args, retries: int = DEFAULT_NETWORK_RETRIES, **kwargs) -> Any:
    """Retry a function on network-transient errors (S4)."""
    backoff = 1.0
    last_result = None
    for attempt in range(retries):
        last_result = fn(*args, **kwargs)
        # Convention: fn returns (returncode, out, err)
        if isinstance(last_result, tuple) and len(last_result) == 3:
            rc, out, err = last_result
            if rc == 0:
                return last_result
            cls = classify_failure(err, out, err)
            if cls == "NETWORK_TRANSIENT" and attempt + 1 < retries:
                time.sleep(backoff)
                backoff *= 2
                continue
        return last_result
    return last_result


# --- Remediation library ---


def _r_MISSING_PY_MODULE(ctx: RemediationContext, err: str) -> bool:
    m = _safe_re_search(r"No module named ['\"]([^'\"]+)['\"]", err)
    if not m:
        return False
    pkg = m.group(1).split(".")[0]
    rc, out, err_out = _with_network_retries(
        _run_subprocess,
        ["conda", "run", "-n", ctx.env_name, "pip", "install", "--prefer-binary", pkg],
        timeout=DEFAULT_CLASS_TIMEOUT_SEC,
    )
    if rc == 0:
        _append_remediated_dep(ctx, "pip", pkg)
    return rc == 0


def _r_MISSING_R_PACKAGE(ctx: RemediationContext, err: str) -> bool:
    m = _safe_re_search(r"there is no package called ['\"]([^'\"]+)['\"]", err)
    if not m:
        return False
    pkg = m.group(1)
    # Try CRAN → Bioc → GitHub (B4)
    rscript = (
        f"ok <- FALSE; "
        f'try({{install.packages("{pkg}", repos="https://cloud.r-project.org"); '
        f'ok <- requireNamespace("{pkg}", quietly=TRUE)}}, silent=TRUE); '
        f'if (!ok) try({{BiocManager::install("{pkg}", ask=FALSE, update=FALSE); '
        f'ok <- requireNamespace("{pkg}", quietly=TRUE)}}, silent=TRUE); '
        f'if (!ok) cat("REMEDIATION_FAILED") else cat("REMEDIATION_OK")'
    )
    rc, out, _ = _with_network_retries(
        _run_subprocess,
        ["conda", "run", "-n", ctx.env_name, "Rscript", "-e", rscript],
        timeout=DEFAULT_CLASS_TIMEOUT_SEC * 2,
    )
    if "REMEDIATION_OK" in out:
        _append_remediated_dep(ctx, "R", pkg)
        return True
    return False


def _r_MISSING_CLI_BINARY(ctx: RemediationContext, err: str) -> bool:
    m = _safe_re_search(r"No such file or directory: ['\"]([^'\"]+)['\"]", err)
    if not m:
        return False
    cli = m.group(1).split("/")[-1]
    rc, out, err_out = _with_network_retries(
        _run_subprocess,
        ["conda", "install", "-n", ctx.env_name, "-c", "conda-forge", "-c", "bioconda", cli, "-y", "--quiet"],
        timeout=DEFAULT_CLASS_TIMEOUT_SEC,
    )
    if rc == 0:
        _append_remediated_dep(ctx, "conda", cli)
    return rc == 0


def _r_MISSING_SO_LIB(ctx: RemediationContext, err: str) -> bool:
    m = _safe_re_search(r"lib([A-Za-z0-9_+-]+)\.so", err)
    if not m:
        return False
    SO_TO_CONDA = {
        "xml2": "libxml2",
        "ssl": "openssl",
        "crypto": "openssl",
        "curl": "curl",
        "hdf5": "hdf5",
        "geos": "geos",
        "gdal": "gdal",
        "proj": "proj",
        "udunits2": "udunits2",
        "tiff": "libtiff",
        "png": "libpng",
        "jpeg": "libjpeg-turbo",
        "freetype": "freetype",
    }
    so = m.group(1)
    pkg = SO_TO_CONDA.get(so) or so
    rc, _, _ = _with_network_retries(
        _run_subprocess,
        ["conda", "install", "-n", ctx.env_name, "-c", "conda-forge", pkg, "-y", "--quiet"],
        timeout=DEFAULT_CLASS_TIMEOUT_SEC,
    )
    if rc == 0:
        _append_remediated_dep(ctx, "conda", pkg)
    return rc == 0


def _r_PIN_INCOMPATIBLE(ctx: RemediationContext, err: str) -> bool:
    # Strip pin, try --no-deps, chain into pip check for transitive gaps (B5)
    m = _safe_re_search(r"No matching distribution found for ([A-Za-z0-9_.-]+)==", err)
    if not m:
        return False
    pkg = m.group(1)
    rc, _, _ = _with_network_retries(
        _run_subprocess,
        ["conda", "run", "-n", ctx.env_name, "pip", "install", "--no-deps", "--prefer-binary", pkg],
        timeout=DEFAULT_CLASS_TIMEOUT_SEC,
    )
    if rc != 0:
        return False
    _append_remediated_dep(ctx, "pip-nodeps", pkg)
    # Chain: run `pip check`; for each missing transitive, install it
    rc2, out2, _ = _run_subprocess(
        ["conda", "run", "-n", ctx.env_name, "pip", "check"],
        timeout=60,
    )
    for line in out2.splitlines():
        m2 = _safe_re_search(r"requires ([A-Za-z0-9_.-]+)", line)
        if m2:
            dep = m2.group(1).split(";")[0].strip()
            _with_network_retries(
                _run_subprocess,
                ["conda", "run", "-n", ctx.env_name, "pip", "install", "--prefer-binary", dep],
                timeout=300,
            )
            _append_remediated_dep(ctx, "pip-transitive", dep)
    return True


def _r_WORKER_OUTPUT_API(ctx: RemediationContext, err: str) -> bool:
    """A missing WorkerOutput method is a gap in the SHARED, canonical ``worker_utils.py`` -- never a
    per-tool issue -- so this remediation refuses to auto-patch it.

    ``tools_user/worker_utils.py`` is a symlink to the git-tracked ``tools/worker_utils.py`` used by
    EVERY worker; a per-tool self-review has no backup/test-gate for a global change, the modify
    playbook explicitly forbids editing it, and mutating shared infra mid-tool-creation can corrupt or
    dirty the repo for all 88 tools. The methods generated workers reach for
    (set_meta/add_warning/add_info/add_note) already exist permanently in the canonical file; a
    genuinely new gap is a one-time developer fix there, not an auto-patch. Refuse and escalate.
    """
    return False


def _r_DISK_EXHAUSTED(ctx: RemediationContext, err: str) -> bool:
    # S5 — we can't solve disk-full automatically; refuse and escalate
    return False


def _r_NETWORK_TRANSIENT(ctx: RemediationContext, err: str) -> bool:
    # Already retried in _with_network_retries; if still failing, escalate
    return False


def _r_PATH_NOT_FOUND(ctx: RemediationContext, err: str) -> bool:
    # Input file genuinely missing — remediation can't help
    return False


#: Refused by name, not by directory. ``envdoctor._worker_dir`` iterates ``("tools", "tools_user")``
#: and returns the first hit, so for these two it hands back the REAL ``tools/`` file and the
#: ``tools_user/`` symlink is never consulted -- a symlink-only guard would miss the one path that
#: actually reaches them. No shipped spec can land here: all 88 name ``<tool>_worker.{py,R}`` and
#: ``<tool>_mcp_server.py``.
_SHARED_MODULES = frozenset({"base_mcp.py", "worker_utils.py"})


def _write_source(path: Path, text: str) -> None:
    """Replace a source file's contents without leaving a partial file or following a symlink.

    Both hazards are real here, and the remediations below are the wrong place to meet either --
    self-review runs *because* a tool is already failing, so its writes land on a box that is
    already in trouble.

    ``Path.write_text`` truncates the destination on open. An interruption between the truncate and
    the write -- a ^C inside the 30-minute budget, an ENOSPC on the box that just filled its disk
    running the tool -- leaves the worker at zero bytes, so the repair path destroys the tool it
    was called to fix. Measured: the original comes back ``''``. ``_append_remediated_dep`` already
    writes tmp+rename for ``install_log.json``; a worker and an MCP server are more load-bearing
    than a log.

    ``Path.write_text`` also follows symlinks. ``tools_user/base_mcp.py`` and
    ``tools_user/worker_utils.py`` are tracked links into ``../tools/``, and
    ``envdoctor._worker_dir`` picks a directory with ``.exists()``, which follows them -- so a
    write aimed at one overwrites the shared module all 88 tools import and leaves the link intact.
    That is how ``tools/worker_utils.py`` has been destroyed three times in this repo.
    ``_r_WORKER_OUTPUT_API`` refuses to patch shared infra in prose; this refuses it at the write,
    which is where the file name is actually known.

    Unlike a plain overwrite this needs a writable *directory*. ``tools_user/`` is the wizard's own
    generated tree, so that holds wherever this is reached.
    """
    path = Path(path)
    if path.name in _SHARED_MODULES:
        raise OSError(
            f"refusing to patch {path}: {path.name} is shared infrastructure that all 88 tools "
            f"import, and a per-tool self-review has no backup or test gate for a global change. "
            f"A genuine gap there is a one-time developer fix -- see _r_WORKER_OUTPUT_API."
        )
    if path.is_symlink():
        raise OSError(
            f"refusing to patch {path}: it is a symlink -> {os.readlink(path)}. Writing would "
            f"overwrite the shared module every tool imports, not the link."
        )
    tmp = path.with_name(path.name + ".sog-tmp")
    try:
        tmp.write_text(text)
        tmp.replace(path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _r_WRONG_API(ctx: RemediationContext, err: str) -> bool:
    """Targeted API rename based on tutorial harvest output (B6)."""
    m = _safe_re_search(r"module ['\"]([^'\"]+)['\"] has no attribute ['\"]([^'\"]+)['\"]", err)
    if not m:
        return False
    module, wrong_attr = m.group(1), m.group(2)
    # Try to find the correct attribute name from tutorials.json or from
    # introspection of the installed module
    tutorials = ctx.knowledge_dir / "tutorials.json"
    candidates: set[str] = set()
    if tutorials.exists():
        try:
            t = json.loads(tutorials.read_text())
            for snip in t.get("top_snippets", []):
                for call_m in re.finditer(
                    rf"\b{re.escape(module.split('.')[-1])}\.([A-Za-z_][A-Za-z_0-9]*)\s*\(", snip.get("code", "")
                ):
                    candidates.add(call_m.group(1))
        except Exception:
            pass
    if not candidates:
        # Fall back to runtime introspection
        rc, out, _ = _run_subprocess(
            ["conda", "run", "-n", ctx.env_name, "python", "-c", f"import {module}; print(' '.join(dir({module})))"],
            timeout=30,
        )
        if rc == 0:
            candidates = {n for n in out.strip().split() if not n.startswith("_")}
    # Heuristic: find closest-name candidate
    best = get_close_matches(wrong_attr, list(candidates), n=1, cutoff=0.75)
    if not best:
        return False
    correct = best[0]
    # Apply targeted replace in worker
    src = ctx.worker_path.read_text()
    pattern = rf"\b{re.escape(wrong_attr)}\b"
    if not re.search(pattern, src):
        return False
    patched = re.sub(pattern, correct, src)
    if patched == src:
        return False
    try:
        _write_source(ctx.worker_path, patched)
    except Exception as ex:
        # Refuse and escalate, the same way _r_WORKER_OUTPUT_API does: an unremediated class is a
        # rollback, whereas an exception out of a remediation would abort the whole review loop.
        LOGGER.warning("[self-review] did not patch %s: %s", ctx.worker_path, ex)
        return False
    return True


def _r_SCHEMA_VIOLATION(ctx: RemediationContext, err: str) -> bool:
    """Convert unnecessarily-required MCP params to Optional with defaults (P20)."""
    import ast

    server_src = ctx.server_path.read_text()
    try:
        tree = ast.parse(server_src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        is_tool = any(
            (isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "tool")
            or (isinstance(d, ast.Attribute) and d.attr == "tool")
            for d in node.decorator_list
        )
        if not is_tool:
            continue
        n_required = len(node.args.args) - len(node.args.defaults)
        if n_required <= 2:
            return False
        # Heuristic patch: for every required arg beyond #2, add default None
        # via a simple text substitution — brittle but bounded
        src_lines = server_src.split("\n")
        # Find function-signature lines; add " = None" defaults for extras
        fn_sig_start = node.lineno - 1
        fn_sig_end = fn_sig_start
        for i in range(fn_sig_start, min(fn_sig_start + 50, len(src_lines))):
            if ")" in src_lines[i]:
                fn_sig_end = i
                break
        sig = "\n".join(src_lines[fn_sig_start : fn_sig_end + 1])
        # For every arg name not already having =, add = None
        new_sig = re.sub(
            r"(\b[a-z_][a-z_0-9]*):\s*(Optional\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_.\[\], ]+)(?!\s*=)",
            lambda m: f"{m.group(1)}: {m.group(2)} = None",
            sig,
        )
        new_lines = src_lines[:fn_sig_start] + new_sig.split("\n") + src_lines[fn_sig_end + 1 :]
        try:
            _write_source(ctx.server_path, "\n".join(new_lines))
        except Exception as ex:
            LOGGER.warning("[self-review] did not patch %s: %s", ctx.server_path, ex)
            return False
        return True
    return False


def _r_UNKNOWN_MCP_TOOL(ctx: RemediationContext, err: str) -> bool:
    """Re-merge mcp_config_user.yaml with the main config."""
    try:
        from spatialomicsgym.agent.mcp_config_merger import build_merged_mcp_config
        from spatialomicsgym.mcp_config_path import find_mcp_config
        from spatialomicsgym.mcp_user_config import resolve_user_config_path

        # Resolved, not CWD-relative: the configs live under the agent tree, not the launch directory.
        shipped = find_mcp_config()
        if shipped is None:
            return False
        build_merged_mcp_config(str(shipped), user_path=resolve_user_config_path(), merge_user=True)
        return True
    except Exception:
        return False


def _r_UNKNOWN(ctx: RemediationContext, err: str) -> bool:
    # No remediation available for unrecognized errors
    return False


REMEDIATIONS: dict[str, Callable[[RemediationContext, str], bool]] = {
    "MISSING_PY_MODULE": _r_MISSING_PY_MODULE,
    "MISSING_R_PACKAGE": _r_MISSING_R_PACKAGE,
    "MISSING_CLI_BINARY": _r_MISSING_CLI_BINARY,
    "MISSING_SO_LIB": _r_MISSING_SO_LIB,
    "PIN_INCOMPATIBLE": _r_PIN_INCOMPATIBLE,
    "WRONG_API": _r_WRONG_API,
    "SCHEMA_VIOLATION": _r_SCHEMA_VIOLATION,
    "WORKER_OUTPUT_API": _r_WORKER_OUTPUT_API,
    "PATH_NOT_FOUND": _r_PATH_NOT_FOUND,
    "UNKNOWN_MCP_TOOL": _r_UNKNOWN_MCP_TOOL,
    "DISK_EXHAUSTED": _r_DISK_EXHAUSTED,
    "NETWORK_TRANSIENT": _r_NETWORK_TRANSIENT,
    "UNKNOWN": _r_UNKNOWN,
}


# --- Orchestration ---


def self_review_loop(
    ctx: RemediationContext,
    error_msg: str,
    *,
    stdout: str = "",
    stderr: str = "",
    rerun: Callable[[], tuple[bool, str, str]] | None = None,
) -> tuple[bool, str, list[dict]]:
    """Main loop. Returns (success, final_error, history).

    rerun: callable that returns (success_bool, stdout, stderr) on each retry.
    """
    if not _enabled():
        return False, error_msg, []

    blocked = _blocked_classes()
    history: list[dict] = []
    current_err = error_msg
    current_stdout = stdout
    current_stderr = stderr
    start = time.time()
    lock_path = Path(f"/tmp/spatialomicsgym_self_review_{ctx.env_name}.lock")

    # S2 — per-env filelock
    if HAS_FILELOCK:
        lock = FileLock(str(lock_path), timeout=60)
    else:
        lock = None

    if lock:
        try:
            lock.acquire(timeout=60)
        except FileLockTimeout:
            # Another process is already self-reviewing this env. That is the lock doing its job,
            # not a failure: this loop's own budget is DEFAULT_TOTAL_BUDGET_SEC -- thirty times the
            # wait above -- so a busy lock is the expected result of two concurrent provisions of
            # one env, not a rare one.
            #
            # Reported the way every other early-out here reports, because this used to be the only
            # one that raised instead. Both callers wrap the call in `except Exception` and record
            # {"attempted": True, "error": ...}, so a review that never started -- never classified
            # the failure, never touched the env -- read back as one that ran and crashed, with a
            # reasonless `rounds: []` persisted beside it by the `finally` below.
            #
            # Acquired outside that try/finally so its release stays paired with a lock we hold,
            # which is also why this branch persists its own history.
            history.append({"round": 0, "outcome": "lock_unavailable", "ts": _ts()})
            _persist_history(ctx, history)
            return False, current_err, history
        except OSError as e:
            # The other way an acquire fails: the lock file cannot be opened at all. `lock_path` is a
            # fixed, un-namespaced name under a world-writable /tmp, so it is not this process's to
            # assume it can have -- a stale directory left by a killed run, a path another UID owns
            # with restrictive bits, a read-only or full /tmp all land here. Without this, the
            # OSError escapes the acquire (which sits outside the try/finally below, so no history is
            # persisted either) and both callers record it as {"attempted": True, "error": ...}: a
            # review that never classified the failure and never touched the env, reported as one
            # that ran and crashed.
            #
            # MUST stay below the FileLockTimeout handler: filelock.Timeout subclasses TimeoutError,
            # which subclasses OSError, so this clause placed first would swallow every busy lock and
            # relabel it. The two situations have different advice -- wait, versus go look at the
            # path -- so they get different outcomes, and `lock_path` is carried because "a lock
            # problem" is not actionable while the name and the errno are.
            history.append(
                {
                    "round": 0,
                    "outcome": "lock_unusable",
                    "lock_path": str(lock_path),
                    "error": f"{type(e).__name__}: {e}",
                    "ts": _ts(),
                }
            )
            _persist_history(ctx, history)
            return False, current_err, history

    try:
        for round_idx in range(DEFAULT_TOTAL_CAP):
            # S7 — total budget
            elapsed = time.time() - start
            if elapsed > DEFAULT_TOTAL_BUDGET_SEC:
                history.append({"round": round_idx, "outcome": "budget_exceeded"})
                break
            # S5 — disk-space pre-check
            if _disk_free_bytes(Path("/tmp")) < DEFAULT_DISK_MIN_BYTES:
                history.append({"round": round_idx, "outcome": "disk_low"})
                break
            # Classify (S13, S16, S20)
            cls = classify_failure(current_err, current_stdout, current_stderr)
            history.append({"round": round_idx, "class": cls, "ts": _ts()})
            if cls in blocked:
                history[-1]["outcome"] = "blocked_by_config"
                break
            if cls in ("UNKNOWN", "DISK_EXHAUSTED", "NETWORK_TRANSIENT", "PATH_NOT_FOUND"):
                history[-1]["outcome"] = "no_remediation_available"
                break
            rem_fn = REMEDIATIONS.get(cls)
            if not rem_fn:
                history[-1]["outcome"] = "no_fn"
                break
            # Apply remediation
            try:
                remediated = rem_fn(ctx, current_err)
            except Exception as e:
                history[-1]["outcome"] = f"remediation_raised: {e}"
                break
            history[-1]["remediated"] = remediated
            if not remediated:
                history[-1]["outcome"] = "remediation_returned_false"
                break
            ctx.total_remediations += 1
            ctx.class_history.append(cls)
            # Rerun
            if rerun is None:
                # A remediation was applied and nothing re-ran it, so nothing here knows whether it
                # worked. This used to answer `True, ""` -- claiming the repair succeeded and, worse,
                # replacing the error text that would have said otherwise with an empty string.
                #
                # Neither caller passes `rerun`, so this is the branch every self-review in this repo
                # actually takes, and both persist what comes back: `provision._maybe_self_review`
                # and `testing._maybe_self_review_tier1` each store {"success": ..., "final_error":
                # ...}, which the wizard writes into state.json. The blanked error is the worse half
                # -- it is the diagnostic a user reads to find out why their tool would not build,
                # discarded exactly when this loop is least entitled to say the problem is gone.
                #
                # `False, current_err` is what every other early-out here returns. It costs no real
                # recovery: both callers confirm independently rather than believe this triple --
                # provision re-derives with its own classify(), Tier-1 re-probes `import <pkg>` and
                # then re-runs the worker cases.
                history[-1]["outcome"] = "no_rerun_callable"
                return False, current_err, history
            try:
                success, new_stdout, new_stderr = rerun()
            except Exception as e:
                history[-1]["outcome"] = f"rerun_raised: {e}"
                break
            if success:
                history[-1]["outcome"] = "success"
                _persist_history(ctx, history)
                return True, "", history
            # Check for progress: did class change?
            new_class = classify_failure(new_stderr or new_stdout, new_stdout, new_stderr)
            history[-1]["outcome"] = f"still_failing, new_class={new_class}"
            current_err = new_stderr or new_stdout
            current_stdout = new_stdout
            current_stderr = new_stderr
    finally:
        if lock:
            try:
                lock.release()
            except Exception:
                pass
        _persist_history(ctx, history)

    return False, current_err, history


# --- Helpers ---


def _ts() -> str:
    return datetime.now(UTC).isoformat()


def _append_remediated_dep(ctx: RemediationContext, method: str, pkg: str) -> None:
    """S19 — track what remediation added."""
    try:
        from spatialomicsgym.mcp_user_config import install_log_path

        log_path = Path(install_log_path())
        if not log_path.exists():
            return
        entries = json.loads(log_path.read_text())
        for e in entries:
            if e.get("tool_id") == ctx.tool_id:
                remediated = e.setdefault("remediated_deps", [])
                remediated.append({"method": method, "pkg": pkg, "ts": _ts()})
                break
        # Atomic write: install_log.json is a shared, load-bearing file (trash/knowledge managers read
        # it); a crash mid-``write_text`` would truncate/corrupt it. tmp + atomic rename never leaves a
        # partial file.
        tmp = log_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, indent=2))
        tmp.replace(log_path)
    except Exception as ex:
        LOGGER.warning("[self-review] failed to log remediated dep: %s", ex)


def _persist_history(ctx: RemediationContext, history: list[dict]) -> None:
    """S17 — history for reproducibility and debug."""
    try:
        ctx.knowledge_dir.mkdir(parents=True, exist_ok=True)
        path = ctx.knowledge_dir / "self_review_history.json"
        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = []
        existing.append({"ts": _ts(), "rounds": history, "total_remediations": ctx.total_remediations})
        path.write_text(json.dumps(existing, indent=2))
    except Exception as ex:
        LOGGER.warning("[self-review] failed to persist history: %s", ex)
