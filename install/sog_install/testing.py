"""
Tiered testing — prove each provisioned env actually works, on tiny local data.

Two tiers, both writing per-server JSON under ``test/installation/``:

* **Tier-1** (no LLM, always on): run the tool's *worker* subprocess against the
  mini Visium dataset through the SAME interpreter provisioning just built
  (``<basic>_<server>``), by reusing ``smoke.smoketest_all_mcp.run_one`` (the development
  tree's ``test/smoke/``, loaded by path) with ``env=<target>``. A server with no smoke-test
  case falls back to a light ``--help`` import probe; when ``test/`` is absent every server
  does, and the run says so up front. This is the load-bearing check — it dispatches the real
  worker the exact way the agent will.
* **Tier-2** (one ``STCoscientist(...).go(prompt)`` per selected category; gated on a
  working LLM key): exercises the full retrieval→dispatch→answer pipeline through
  the wizard's OWN generated MCP config (:func:`constants.generated_mcp_config`), so
  it can never perturb the agent's real config. Optional and best-effort — a missing
  key, an unbuilt tool, or a low eval metric is a ``SKIP``, never a failure.

Isolation: nothing here mutates a conda env or the agent's config; it only *reads*
envs and *writes* result JSON under the artifact dir. Heavy imports (``run_one``,
``STCoscientist``) are lazy so a bare launcher env can import this module.

Stdlib + pyyaml only.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from . import constants, llm_setup, progress, provision, testdata
from .category_prompts import prompt_for
from .demo_tool_names import display_name, with_tool_named
from .session_log import _sanitize, redact  # C6: every persisted sink passes redact()/_sanitize (stdlib leaf)

if TYPE_CHECKING:
    from pathlib import Path

    from .categories import Category
    from .decisions import TestDecision
    from .envtools import Conda
    from .prompts import PromptIO
    from .session_log import SessionLog
    from .specs import ToolSpec

# -- status vocabulary (mirrors the smoke harness) ---------------------------- #
PASS, FAIL, SKIP, TIMEOUT, ERROR, WARN = "PASS", "FAIL", "SKIP", "TIMEOUT", "ERROR", "WARN"
_BAD = frozenset({FAIL, TIMEOUT, ERROR})

# Truthy spellings for the opt-in LLM-remediation gate (mirrors provision._TRUTHY).
_TRUTHY = {"1", "true", "yes", "y", "on"}

# LLM provider keys that, if present, mean Tier-2 has a chance of working. Used only
# as a cheap pre-check; the real gate is a clean agent construction + go().
_LLM_KEY_HINTS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_ACCESS_KEY_ID",
    "SOG_CUSTOM_BASE_URL",
)

# A value set for one of these providers is a REAL key/URL only if it starts with the signature
# prefix; a value lacking it is a placeholder. Providers with no stable public prefix
# (AWS_BEARER_TOKEN_BEDROCK) are left off — only the sentinel/length checks apply to them.
_KEY_REAL_PREFIXES: dict[str, tuple[str, ...]] = {
    "ANTHROPIC_API_KEY": ("sk-ant-",),
    "OPENAI_API_KEY": ("sk-",),
    "GEMINI_API_KEY": ("AIza",),
    "GROQ_API_KEY": ("gsk_",),
    "AWS_ACCESS_KEY_ID": ("AKIA", "ASIA"),
    "SOG_CUSTOM_BASE_URL": ("http://", "https://"),
}
# Substrings that betray a stand-in even when it carries a real-looking prefix (e.g. "sk-ant-xxxx").
# Chosen to never occur in genuine key material, so a real key is never rejected on a sentinel.
_KEY_PLACEHOLDER_SENTINELS = (
    "xxx",
    "your",
    "placeholder",
    "changeme",
    "change-me",
    "change_me",
    "example",
    "dummy",
    "replace",
    "redacted",
    "yourkey",
    "my-key",
    "api-key",
    "api_key",
    "apikey",
    "put-your",
    "insert",
    "<",
    ">",
    "...",
)


# --------------------------------------------------------------------------- #
# Result records
# --------------------------------------------------------------------------- #
@dataclass
class WorkerTest:
    name: str
    status: str
    detail: str = ""
    covered: bool = True  # False => fell back to a --help import probe

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail, "covered": self.covered}


@dataclass
class ServerTest:
    server_key: str
    target_env: str
    worker_kind: str = "python"
    status: str = SKIP  # rollup over worker_tests
    worker_tests: list[WorkerTest] = field(default_factory=list)
    repairs: list[dict] = field(default_factory=list)  # envdoctor RepairResult dicts, in order
    self_review: dict | None = None
    # A specific, actionable Lane-3 message when the failure is real but NOT an env-build problem
    # (external token / special input / data shape / …). Persisted so ``_finish`` and the Part-D
    # demo summary can tell the user exactly what to fix rather than a bare FAIL.
    needs_attention: str = ""

    @property
    def verified_on_mini_data(self) -> bool:
        """Whether a real mini-data case ran and passed. A ``PASS`` from the ``--help`` import probe
        alone (no smoke case for this worker, or none importable on a wheel install) is not one: it
        exercised no data, and was counted as "passed on mini data" anyway (hunt 2026-09-30,
        uL4-honesty-1)."""
        return any(w.covered and w.status == PASS for w in self.worker_tests)

    def to_dict(self) -> dict:
        return {
            "server": self.server_key,
            "env": self.target_env,
            "worker_kind": self.worker_kind,
            "status": self.status,
            "verified_on_mini_data": self.verified_on_mini_data,
            "worker_tests": [w.to_dict() for w in self.worker_tests],
            "repairs": self.repairs,
            "self_review": self.self_review,
            "needs_attention": self.needs_attention,
        }


@dataclass
class CategoryTest:
    category: str
    server_key: str = ""
    status: str = SKIP
    detail: str = ""
    eval: dict | None = None
    repairs: list[dict] = field(default_factory=list)  # envdoctor RepairResult dicts from un-masking
    masked_env_issue: str | None = None  # EnvIssueKind if a tool env failure hid behind the answer

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "server": self.server_key,
            "status": self.status,
            "detail": self.detail,
            "eval": self.eval,
            "repairs": self.repairs,
            "masked_env_issue": self.masked_env_issue,
        }


@dataclass
class TestReport:
    basic_env: str
    servers: list[ServerTest] = field(default_factory=list)
    categories: list[CategoryTest] = field(default_factory=list)
    started: str = ""
    finished: str = ""

    def summary(self) -> dict:
        t1: dict[str, int] = {}
        for s in self.servers:
            t1[s.status] = t1.get(s.status, 0) + 1
        t2: dict[str, int] = {}
        for c in self.categories:
            t2[c.status] = t2.get(c.status, 0) + 1
        return {
            "basic_env": self.basic_env,
            "started": self.started,
            "finished": self.finished,
            "tier1": {
                "counts": t1,
                "servers": {s.server_key: s.status for s in self.servers},
                # The PASSes split by what they proved: a mini-data case, or only the --help import
                # probe (uL4-honesty-1). Every key in either list also counts as PASS above.
                "verified_on_mini_data": [s.server_key for s in self.servers if s.verified_on_mini_data],
                "import_probe_only": [
                    s.server_key for s in self.servers if s.status == PASS and not s.verified_on_mini_data
                ],
            },
            "tier2": {"counts": t2, "categories": {c.category: c.status for c in self.categories}},
        }


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


#: The Tier-1 smoke harness inside the development tree (``test/smoke/smoketest_all_mcp.py``).
_SMOKE_HARNESS = "smoke.smoketest_all_mcp"


def _smoke_harness():
    """The smoke harness module, imported from ``test/`` by path (:func:`constants.load_test_module`).

    Looked up on every call, never cached here, so a test that patches the module's attributes
    (``run_one``, ``TESTS``) is the module this reads.
    """
    return constants.load_test_module(_SMOKE_HARNESS)


def tier1_harness_note() -> str | None:
    """Why Tier-1 cannot run mini-data cases on this install, or ``None`` when it can.

    Without the harness every server falls back to the ``--help`` import probe; that is a weaker
    check, and the run must say so rather than let ``_tests_index``'s empty table pass silently.
    """
    try:
        _smoke_harness()
    except Exception as exc:
        if constants.dev_test_dir() is None:
            return constants.TEST_TREE_MISSING
        return f"the smoke harness in test/smoke could not be loaded ({type(exc).__name__}: {str(exc)[:160]})"
    return None


def _tests_index() -> dict[str, list[tuple]]:
    """``{worker_basename: [smoke TESTS tuple, ...]}`` — the Tier-1 join table.

    Keyed by the worker filename because that is the one field a smoke case and a
    :class:`ToolSpec` share verbatim (a server_key like ``graphst`` maps to *two*
    cases, ``graphst_cluster`` + ``graphst_deconv``, both on ``graphst_worker.py``).
    ``{}`` when the harness cannot be loaded; :func:`tier1_harness_note` says why, and
    :func:`run_tests` reports it.
    """
    try:
        TESTS = _smoke_harness().TESTS  # the harness's own name for the table
    except Exception:
        return {}
    idx: dict[str, list[tuple]] = {}
    for entry in TESTS:
        worker = entry[2]
        idx.setdefault(os.path.basename(worker), []).append(entry)
    return idx


def _resolve_tools_dir(worker_file: str) -> str:
    """Which dir holds this worker basename — ``agent/tools/`` (base) or ``agent/tools_user/``."""
    for d in (constants.tools_dir(), constants.tools_user_dir()):
        if (d / worker_file).exists():
            return str(d)
    return str(constants.tools_dir())


def _default_envs_root() -> str:
    """The live conda ``envs/`` dir on *this* machine — never the hardcoded ``/opt/conda``.

    This process runs inside a per-clone conda env, so its own interpreter prefix locates the
    shared ``envs/`` dir the per-tool envs also live under: a named env sits at
    ``<root>/envs/<name>`` (parent is ``envs/``); the base env sits at ``<root>`` (envs at
    ``<root>/envs``). Correct whether conda is at ``/opt/conda``, ``~/miniconda3``, or anywhere
    else — so the Tier-1 test finds the tool interpreters instead of silently SKIPping them all
    on a machine whose conda isn't at ``/opt/conda``."""
    prefix = (os.environ.get("CONDA_PREFIX") or sys.prefix).rstrip("/")
    parent = os.path.dirname(prefix)
    if os.path.basename(parent) == "envs":
        return parent  # running inside <root>/envs/<name>
    return os.path.join(prefix, "envs")  # running in the base env <root>


def _conda_envs_root(conda, target_env: str) -> str | None:
    """The ``envs/`` dir conda itself reports ``target_env`` in, or ``None`` when it cannot say.

    ``_default_envs_root`` guesses from ``CONDA_PREFIX``/``sys.prefix``, which is wrong whenever the tool
    envs live somewhere else: conda's root is read-only on a shared box (``conda create -n`` lands in
    ``~/.conda/envs``), or ``sog-setup`` runs from a venv with no base activated. Tier-1 then SKIPped
    every tool as "env missing" while ``doctor`` and the wiring -- which ask conda -- called them healthy
    (hunt 2026-09-30, u37-setup-checks-11). Never raises; a conda without an answer keeps the guess."""
    lookup = getattr(conda, "env_prefix", None)
    if not callable(lookup):
        return None
    try:
        prefix = lookup(target_env)
    except Exception:
        return None
    if not isinstance(prefix, str) or not prefix.strip():
        return None
    prefix = prefix.rstrip("/\\")
    # ``_interp`` joins ``<root>/<target_env>``, so the answer is only usable when conda's prefix ends in
    # the env's own directory (always, for a named env; not for ``base``).
    return os.path.dirname(prefix) if os.path.basename(prefix) == target_env else None


def _cases_in(spec: ToolSpec, target: str, envs_root: str | None) -> list[WorkerTest]:
    """``_run_worker_cases`` with conda's envs root when there is one. The two-argument call is kept
    when there is not, so a caller (or a test double) written against that signature is unchanged."""
    return _run_worker_cases(spec, target, envs_root) if envs_root else _run_worker_cases(spec, target)


def _interp(target_env: str, worker_kind: str, envs_root: str | None = None) -> str:
    # Per-OS interpreter layout lives in constants.interp_path — the single source of truth every
    # path builder shares (a1c#2), so testing / wiring / specs / mcp_resolver can't drift apart.
    exe = "Rscript" if worker_kind == "rscript" else "python"
    root = (envs_root or _default_envs_root()).rstrip("/")
    return constants.interp_path(f"{root}/{target_env}", exe, posix=_POSIX)


def _rollup(worker_tests: list[WorkerTest]) -> str:
    """One status for the server: worst covered result wins; else the probe result."""
    covered = [w for w in worker_tests if w.covered]
    pool = covered or worker_tests
    if not pool:
        return SKIP
    for bad in (ERROR, TIMEOUT, FAIL):
        if any(w.status == bad for w in pool):
            return FAIL
    if any(w.status == PASS for w in pool):
        return PASS
    if any(w.status == WARN for w in pool):
        return WARN
    return SKIP


def _tail(text: str, n: int = 20) -> str:
    """Last ``n`` non-empty lines of ``text`` — the evidence tail fed to ``classify_failure``."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def _last_stderr_reason(err: str, *, limit: int = 160) -> str:
    """The single most-diagnostic (REDACTED) stderr line, for surfacing WHY a probe emitted no
    fenced contract — e.g. conda's ``Could not find conda environment: <name>`` when the base env is
    absent, or a base-env ``ModuleNotFoundError``. Prefers an explicit error/not-found line over the
    literal last line (which is often a trailing hint), falling back to the last line. Redacted and
    length-clamped; ``""`` for blank stderr; never raises."""
    try:
        lines = [ln.strip() for ln in (err or "").splitlines() if ln.strip()]
        if not lines:
            return ""
        signals = ("error", "not found", "could not", "no such", "not a conda", "no module named", "traceback")
        for ln in reversed(lines):
            if any(s in ln.lower() for s in signals):
                return redact(ln)[:limit]
        return redact(lines[-1])[:limit]
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Portable process-group kill: POSIX puts the child in its own session so a timeout reaps the
# whole tree (killpg); Windows lacks setsid/killpg/SIGKILL — degrade to a single-child terminate.
# --------------------------------------------------------------------------- #
_POSIX = os.name == "posix"
_SIGKILL = getattr(signal, "SIGKILL", getattr(signal, "SIGTERM", 15))


def _session_kwargs() -> dict:
    """``start_new_session=True`` on POSIX (own process group for group-kill); ``{}`` on Windows,
    where the kwarg has no effect and ``setsid`` does not exist."""
    return {"start_new_session": True} if _POSIX else {}


def _hard_kill(proc: subprocess.Popen) -> None:
    """Kill the whole process tree on POSIX (``killpg``), else just the child. Never raises —
    ``os.killpg``/``os.getpgid``/``SIGKILL`` are absent on Windows, so we guard on them."""
    if _POSIX and hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            os.killpg(os.getpgid(proc.pid), _SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass  # fall through to a direct child kill
    with contextlib.suppress(Exception):
        proc.kill()


def _run_subprocess_cancellable(cmd: list[str], *, timeout: int, cwd: str) -> tuple[str, str, int, bool]:
    """Run ``cmd`` in its OWN session and hard-kill the whole process group on timeout.

    Returns ``(stdout, stderr, returncode, timed_out)``. ``start_new_session=True`` puts the
    child and any grandchildren it spawns (e.g. MCP servers) in a fresh process group, so on a
    timeout we ``killpg`` the ENTIRE tree instead of leaving orphans — the exact hang the old
    daemon-thread ``join`` could never stop. Never raises on a timeout or a launch failure.
    """
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",  # non-UTF-8 tool output must not raise UnicodeDecodeError in communicate()
            cwd=cwd,
            **_session_kwargs(),
        )
    except OSError as exc:
        return "", f"launch failed: {exc}", 127, False
    try:
        out, err = proc.communicate(timeout=timeout)
        return out or "", err or "", proc.returncode, False
    except subprocess.TimeoutExpired:
        _hard_kill(proc)  # group-kill on POSIX, child-kill on Windows — never raises
        try:
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return out or "", err or "", -_SIGKILL, True
    except BaseException:
        # Ctrl-C / the wizard SIGINT handler's SystemExit(130): the probe runs DETACHED in its own
        # session (paid LLM calls + GPU + MCP-tool grandchildren) and never saw the terminal's SIGINT.
        # Reap the whole group before the interrupt propagates — else it keeps running unbounded (the
        # timeout that would have bounded it died with this parent). Re-raise so the wizard exits 130.
        _hard_kill(proc)
        with contextlib.suppress(Exception):
            proc.communicate(timeout=30)
        raise


def _feed_box(line: str, box) -> None:
    """Forward ONE probe stdout line to the live thinking box iff it is a ``SOG_PROBE_STEP`` marker.

    Everything else (the agent's banner noise, the JSON fence) is ignored here — the caller still
    accumulates the full stdout so ``_extract_probe_json`` finds the contract. Best-effort: a missing
    box, a non-live box, a non-marker line, or malformed JSON is a silent no-op — the box only ever
    *reflects* the run, never perturbs it."""
    if box is None or not getattr(box, "live", False):
        return
    s = line.strip()
    from ._agent_probe import PROBE_STEP  # lazy (mirrors _extract_probe_json) — keeps the import graph thin

    if not s.startswith(PROBE_STEP):
        return
    try:
        step = json.loads(s[len(PROBE_STEP) :].strip())
    except (json.JSONDecodeError, TypeError, ValueError):
        return
    if not isinstance(step, dict):
        return
    with contextlib.suppress(Exception):  # a rendering hiccup must never break the reader loop
        box.update(
            mode=step.get("mode"),
            status=step.get("status"),
            label=step.get("label"),
            thought=step.get("thought"),
            action=step.get("action"),
        )


def _run_subprocess_streaming(cmd: list[str], *, timeout: int, cwd: str, box) -> tuple[str, str, int, bool]:
    """Like :func:`_run_subprocess_cancellable`, but tee each stdout line live to ``box`` as it
    arrives so the agent's ``SOG_PROBE_STEP`` markers drive the thinking box during the run.

    Identical kill contract (own session + ``killpg`` group-kill, ``(-SIGKILL, True)`` on timeout,
    ``("", "launch failed…", 127, False)`` on a launch OSError) and identical
    ``(stdout, stderr, returncode, timed_out)`` return, so the downstream ``_extract_probe_json`` parse
    is byte-for-byte what the captured path produces. Two daemon readers keep the stdout/stderr split
    intact and drain the tail after exit; ``_feed_box`` only *reads* the stdout it is already buffering.
    """
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",  # non-UTF-8 tool output must not raise in the reader
            cwd=cwd,
            **_session_kwargs(),  # own process group on POSIX → killpg reaps agent + MCP grandchildren
            bufsize=1,  # line-buffered: readline() yields each SOG_PROBE_STEP marker as it arrives
        )
    except OSError as exc:
        return "", f"launch failed: {exc}", 127, False

    out_buf: list[str] = []
    err_buf: list[str] = []

    def _read(pipe, buf: list[str], *, tee: bool) -> None:
        try:
            for line in iter(pipe.readline, ""):
                buf.append(line)
                if tee:
                    _feed_box(line, box)
        except (OSError, ValueError):  # pipe torn down under us (kill) — stop quietly
            pass
        finally:
            with contextlib.suppress(Exception):
                pipe.close()

    t_out = threading.Thread(target=_read, args=(proc.stdout, out_buf), kwargs={"tee": True}, daemon=True)
    t_err = threading.Thread(target=_read, args=(proc.stderr, err_buf), kwargs={"tee": False}, daemon=True)
    t_out.start()
    t_err.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _hard_kill(proc)  # group-kill on POSIX, child-kill on Windows — never raises
        with contextlib.suppress(Exception):
            proc.wait(timeout=30)
        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)
        return "".join(out_buf), "".join(err_buf), -_SIGKILL, True
    except BaseException:
        # Ctrl-C / the wizard SIGINT handler's SystemExit(130): the probe runs DETACHED in its own
        # session (paid LLM calls + GPU + MCP-tool grandchildren) and never saw the terminal's SIGINT.
        # Reap the whole group before the interrupt propagates, then re-raise so the wizard exits 130.
        _hard_kill(proc)
        with contextlib.suppress(Exception):
            proc.wait(timeout=30)
        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)
        raise
    # Child exited → both pipes hit EOF → readers drain the final buffered lines and stop.
    t_out.join(timeout=5.0)
    t_err.join(timeout=5.0)
    return "".join(out_buf), "".join(err_buf), proc.returncode, False


def _probe_command(conda: Conda, basic_env: str, request_json: str) -> list[str]:
    """The ``conda run … python -m …_agent_probe <json>`` argv, executed in the base env
    (where the agent + torch live), not the wizard's launcher env."""
    from .envtools import no_capture_output_args  # micromamba's `run` rejects --no-capture-output

    return [
        conda.exe,
        "run",
        *no_capture_output_args(conda.exe),
        "-n",
        basic_env,
        "python",
        "-m",
        "sog_install._agent_probe",
        request_json,
    ]


def _extract_probe_json(stdout: str) -> dict | None:
    """Pull the fenced contract dict out of the probe's stdout (``None`` if absent/garbled).

    The probe fences its one line of JSON with sentinels so the agent's own banner/print
    noise on stdout can never corrupt parsing.
    """
    from ._agent_probe import PROBE_BEGIN, PROBE_END

    # The contract is printed LAST (begin / json / end), so the authoritative close is the final END
    # sentinel (rfind, not find — a payload that echoes the end-sentinel inside its JSON, or a
    # grandchild writing a stray fence to fd-1, can't truncate the parse). Its matching BEGIN is
    # normally the nearest one to its left — BUT the payload can legitimately echo the BEGIN sentinel
    # too (the agent's own answer text, or a stray grandchild fence). Anchoring blindly on that inner
    # BEGIN would slice a truncated, unparseable fragment and DISCARD a perfectly good contract. So
    # walk BEGIN fences from the rightmost leftward and return the first slice that parses to a dict —
    # the true opening fence — instead of giving up on the first one that doesn't.
    e = stdout.rfind(PROBE_END)
    if e == -1:
        return None
    search_end = e
    while True:
        b = stdout.rfind(PROBE_BEGIN, 0, search_end)
        if b == -1:
            return None
        blob = stdout[b + len(PROBE_BEGIN) : e].strip()
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            obj = None
        if isinstance(obj, dict):
            return obj
        search_end = b  # this fence didn't parse — step to an earlier BEGIN and retry


@dataclass
class ProbeResult:
    """Parsed outcome of one out-of-process agent probe."""

    answer: str = ""
    log: list[str] = field(default_factory=list)
    error: str = ""
    timed_out: bool = False
    returncode: int = 0
    stderr_tail: str = ""
    traceback: str = ""  # C6: the probe's own crash traceback (from its fenced contract), for the post-mortem
    # Why the agent's turn stopped early (a give-up, the step budget, a provider error the stream turned
    # into a note), or "" when it finished. See ``_agent_probe._run`` (hunt 2026-09-30, uL4-honesty-6).
    degraded: str = ""

    @property
    def log_text(self) -> str:
        return "\n".join(self.log)


def _persist_log_artifact(target: str, suffix: str, text: str) -> Path | None:
    """Redact-and-write ``text`` to ``logs_dir()/<target>.<suffix>`` (best-effort; never raises).

    Gives each self-heal surface its OWN durable sink instead of sharing the one provision-owned
    ``<target>.build.log`` (C6): the Tier-1 / Tier-2 seed evidence (``.tier1.log`` / ``.tier2.log``,
    threaded into ``remediate(log_path=…)`` so ``log_monitor.gather`` reads THIS run's real error, not a
    stale provision log) and the expensive real-run probe's post-mortem (``.agent.log``). Every string
    passes ``redact()`` and the body is size-clipped, so a credential echoed in stderr never lands on
    disk raw and a pathological log can't fill the state dir."""
    try:
        logs = constants.logs_dir()
        logs.mkdir(parents=True, exist_ok=True)
        path = logs / f"{target}.{suffix}"
        body = redact(text or "")
        cap = constants.BUILD_LOG_CLIP_BYTES
        if len(body) > cap:
            body = body[-cap:]
        path.write_text(body, encoding="utf-8")
        return path
    except OSError:
        return None


def _run_probe(
    conda: Conda,
    basic_env: str,
    root: str,
    cfg_path: str,
    prompt: str,
    *,
    timeout: int,
    add_data: dict[str, str] | None = None,
    box=None,
    log_target: str | None = None,
) -> ProbeResult:
    """Run ONE cancellable agent probe in the base env and parse its JSON contract.

    ``add_data`` (optional ``{abs_path: description}``) is forwarded to ``agent.add_data`` inside
    the probe — used by the Part-D demo / real-run to hand the agent the user's own file. Omitted
    for Tier-2, whose dataset is staged into the data lake under ``root``; when absent the request
    is byte-for-byte the pre-existing one.

    ``box`` (optional live thinking-box handle) turns on the line-streaming subprocess path so the
    agent's per-step ``SOG_PROBE_STEP`` markers drive the box live. When it is ``None`` or inert
    (a disabled / non-TTY box, ``.live`` False) we take the original captured ``_run_subprocess_
    cancellable`` path — byte-for-byte the pre-box behavior.

    ``log_target`` (optional) names the sink for a durable, REDACTED ``<log_target>.agent.log``
    post-mortem of this run (C6); omitted ⇒ no artifact is written (back-compat)."""
    req = {"root": root, "config_path": cfg_path, "prompt": prompt}
    if add_data:
        req["add_data"] = {str(k): str(v) for k, v in add_data.items()}
    request = json.dumps(req)
    cmd = _probe_command(conda, basic_env, request)
    cwd = str(constants.repo_root())
    if box is not None and getattr(box, "live", False):
        out, err, rc, timed_out = _run_subprocess_streaming(cmd, timeout=timeout, cwd=cwd, box=box)
    else:
        out, err, rc, timed_out = _run_subprocess_cancellable(cmd, timeout=timeout, cwd=cwd)
    if timed_out:
        result = ProbeResult(timed_out=True, returncode=rc, stderr_tail=_tail(err))
    else:
        contract = _extract_probe_json(out)
        if contract is None:
            # No fenced contract → the probe never got far enough to emit one (e.g. a conda/env launch
            # failure — an absent base env, a broken interpreter, an ImportError before main()). Surface
            # the actual reason (REDACTED) IN the error so the user sees "…: Could not find conda
            # environment: domains" instead of a bare, baffling "no probe contract returned". The
            # "no probe contract" PREFIX is preserved so _score_probe still classifies it as
            # agent-unavailable (SKIP, substring match), and stderr_tail still carries the full tail for
            # classify_failure.
            reason = _last_stderr_reason(err)
            detail = "no probe contract returned" + (f" — {reason}" if reason else "")
            result = ProbeResult(error=detail, returncode=rc, stderr_tail=_tail(err))
        else:
            result = ProbeResult(
                answer=str(contract.get("answer", "")),
                log=[str(x) for x in (contract.get("log") or [])],
                error=str(contract.get("error", "")),
                returncode=rc,
                stderr_tail=_tail(err),
                traceback=str(contract.get("traceback", "")),  # C6: the probe's own crash trace, if any
                degraded=str(contract.get("degraded") or ""),
            )
    # C6: leave a durable, REDACTED post-mortem of this (expensive, opaque) agent run — its full log,
    # the probe's own crash traceback, and the FULL stderr (not just the ~20-line tail) — so a failed
    # real-run / Tier-2 probe is debuggable after the fact instead of vanishing into _extract_probe_json.
    # Best-effort + size-clipped; skipped when the caller names no target (keeps back-compat byte-for-byte).
    if log_target:
        parts = [result.log_text]
        if result.traceback:
            parts.append("--- probe traceback ---\n" + result.traceback)
        if result.error:
            parts.append("--- probe error ---\n" + result.error)
        if err:
            parts.append("--- stderr ---\n" + err)
        _persist_log_artifact(log_target, "agent.log", "\n\n".join(p for p in parts if p))
    return result


def _key_value_is_usable(name: str, value: str, *, azure: bool = False) -> bool:
    """Shape check: does this env value read as a REAL provider key/URL, not a placeholder?

    Conservative by design — rejects only unmistakable stand-ins (a placeholder sentinel, or a
    well-known provider value lacking its signature prefix, or a too-short token) so a real key is
    never gated out. A plausible-but-dead key that slips through is caught later by the auth-failure
    backstop, which SKIPs-with-warning rather than hard-ERRORing. Only prefix/sentinel *shape* is
    inspected — the secret's contents are never logged or returned.

    ``azure``: ``OPENAI_API_KEY`` holds an Azure OpenAI key here (:func:`_openai_key_is_azure`). An
    Azure key is 32 hex or 84 base62 characters and never starts ``sk-``, so the OpenAI prefix rule
    called every key of the shipped default provider a placeholder and skipped Tier-2 on a working
    install (hunt 2026-09-30, u37-setup-checks-5, uL4-honesty-2). It gets the sentinel and length
    checks a prefix-less provider gets."""
    v = value.strip().strip('"').strip("'").strip()
    if not v:
        return False
    if any(s in v.lower() for s in _KEY_PLACEHOLDER_SENTINELS):
        return False
    prefixes = _KEY_REAL_PREFIXES.get(name)
    if prefixes and not (azure and name == "OPENAI_API_KEY"):
        return v.startswith(prefixes)
    # No stable public prefix (e.g. AWS_BEARER_TOKEN_BEDROCK): accept a non-sentinel token of
    # plausible length; too short to be a real credential → treat as a stand-in.
    return len(v) >= 16


def _collect_llm_key_values() -> dict[str, str]:
    """``{hint_name: raw_value}`` for every provider var set in ``.env`` then env (env wins),
    non-empty only. Used only to test each value's *shape* — nothing here is logged or surfaced."""
    vals: dict[str, str] = {}
    # Reuse the canonical .env reader (handles `export FOO=bar`, quotes, comments) instead of a
    # hand-rolled split that missed `export` — a user whose .env uses `export ANTHROPIC_API_KEY=…`
    # was wrongly told "no key" and the live demo silently downgraded (B4).
    for key, val in llm_setup.read_dotenv_values().items():
        if key in _LLM_KEY_HINTS and val:
            vals[key] = val
    for k in _LLM_KEY_HINTS:  # a real process-env value overrides whatever .env carried
        v = (os.environ.get(k) or "").strip()
        if v:
            vals[k] = v
    return vals


def _openai_key_is_azure() -> bool:
    """Does ``OPENAI_API_KEY`` belong to Azure OpenAI on this install, rather than to OpenAI?

    Both providers keep their key in that one variable (``credentials._AZURE`` / ``_OPENAI``), so the
    variable's name cannot say which shape to expect. Asked in the agent's own order
    (``llm.resolve_source``), ``.env`` then the process env (env wins): the provider the ``SOG_LLM``
    model name proves (``azure-gpt-6-astra``, the shipped default) unless ``SOG_SOURCE``/``LLM_SOURCE``
    names a provider such a name cannot override, else that env source, else ``OPENAI_ENDPOINT`` --
    which only Azure ever sets (``credentials._AZURE`` is its sole writer). Ranking ``SOG_SOURCE``
    first called the real Azure key a placeholder on a box with a stale ``SOG_SOURCE=OpenAI`` beside
    ``SOG_LLM=azure-…``, which the agent routes to Azure (hunt 2026-09-30, u37-setup-checks-5 repair)."""
    from spatialomicsgym.provider_names import OVERRIDABLE_ENV_SOURCES, canonical_source, source_from_model_prefix

    dotenv = llm_setup.read_dotenv_values()

    def value(name: str) -> str:
        return (os.environ.get(name) or "").strip() or (dotenv.get(name) or "").strip()

    env_source = canonical_source(value("SOG_SOURCE") or value("LLM_SOURCE"))
    prefix_source = source_from_model_prefix(value("SOG_LLM"))
    if prefix_source is not None and (env_source is None or env_source in OVERRIDABLE_ENV_SOURCES):
        source = prefix_source
    else:
        source = env_source
    if source:
        return source == "AzureOpenAI"
    return bool(value("OPENAI_ENDPOINT"))


def _usable_llm_keys() -> list[str]:
    """Provider vars whose value passes the placeholder/shape check (i.e. Tier-2 has a real chance)."""
    values = _collect_llm_key_values()
    azure = "OPENAI_API_KEY" in values and _openai_key_is_azure()
    return [k for k, v in values.items() if _key_value_is_usable(k, v, azure=azure)]


def _llm_key_present() -> bool:
    """Gate for the real-agent tier: is a *usable* (non-placeholder) LLM key visible?

    Presence-only would let this box's placeholder ``.env`` keys open Tier-2 and then hard-ERROR on
    auth for every category; validity-awareness turns that into a clean SKIP-with-warning instead."""
    return bool(_usable_llm_keys())


def _llm_key_skip_reason() -> str:
    """The user-facing reason Tier-2 is being skipped — distinguishes a placeholder from an absent key."""
    if _collect_llm_key_values():  # something is set, but nothing passed the shape check → placeholder
        return "LLM key looks like a placeholder — real-agent tier skipped; add a real key to enable it"
    return "no LLM key detected (provisioning + Tier-1 are unaffected)"


# --------------------------------------------------------------------------- #
# Public re-exports — the Part-D demo phase reuses the *exact* Tier-2 machinery
# (out-of-process probe + key-presence gate) instead of forking its own.
# --------------------------------------------------------------------------- #
def run_agent_probe(
    conda: Conda,
    basic_env: str,
    root: str,
    cfg_path: str,
    prompt: str,
    *,
    timeout: int = constants.AGENT_TEST_TIMEOUT_SEC,
    add_data: dict[str, str] | None = None,
    box=None,
    log_target: str | None = None,
) -> ProbeResult:
    """Run ONE real ``STCoscientist(...).go_stream(prompt)`` out-of-process in the base env and return
    its parsed contract. The public face of :func:`_run_probe`: same cancellable subprocess, same
    fenced-JSON parse, same env-only boundary (it only TRIGGERS the agent — never repairs). The
    Part-D demo / real-run call this so a jargon-free biologist prompt is routed by the agent's own
    recommender, exactly like Tier-2, with the user's file supplied via ``add_data``.

    ``box`` (optional live thinking-box handle) streams the agent's per-step reasoning into the box as
    it runs; omitted / inert ⇒ the original captured path, byte-for-byte. ``log_target`` (optional)
    names the ``<log_target>.agent.log`` post-mortem sink so the expensive real-run leaves a durable,
    redacted artifact (C6)."""
    return _run_probe(
        conda, basic_env, root, cfg_path, prompt, timeout=timeout, add_data=add_data, box=box, log_target=log_target
    )


def llm_key_present() -> bool:
    """Public gate: is a *usable* (non-placeholder) LLM key visible? The demo/real-run key-gate —
    run the agent now when True; otherwise save a runnable script and print the command."""
    return _llm_key_present()


def llm_key_skip_reason() -> str:
    """Public: the user-facing reason the agent can't be launched now (placeholder vs absent key)."""
    return _llm_key_skip_reason()


# --------------------------------------------------------------------------- #
# Tier-1 — worker on mini data
# --------------------------------------------------------------------------- #
def _worker_help_probe(target_env: str, worker_kind: str, worker_abs: str, envs_root: str | None = None) -> WorkerTest:
    """Fallback for a server with no smoke case: does the worker import + show usage?"""
    interp = _interp(target_env, worker_kind, envs_root)
    name = f"{os.path.basename(worker_abs)}:help"
    if not os.path.exists(interp):
        return WorkerTest(name, SKIP, f"interpreter missing: {interp}", covered=False)
    if not os.path.exists(worker_abs):
        return WorkerTest(name, SKIP, "worker missing", covered=False)
    # Route through the session-group runner (not a bare ``subprocess.run``): a worker whose ``--help``
    # path forks grandchildren (an R worker sourcing scripts, a py worker whose import spawns a helper)
    # must be reaped as a WHOLE tree on the 90s timeout — ``subprocess.run``'s own kill only reaps the
    # direct child, leaking the rest to hold env/pkg locks. ``_run_subprocess_cancellable`` never raises:
    # a timeout comes back as ``timed_out=True``; a launch failure (OSError) as ``rc=127`` + a
    # ``launch failed:`` stderr — the two arms the old ``except`` clauses handled.
    out, err, rc, timed_out = _run_subprocess_cancellable(
        [interp, worker_abs, "--help"], timeout=90, cwd=os.path.dirname(worker_abs)
    )
    if timed_out:
        return WorkerTest(name, WARN, "--help timed out", covered=False)
    if err.startswith("launch failed:"):
        # redact-before-clip (Class-4): err is RAW worker stderr (a separate subprocess that holds none
        # of the parent's registered secrets), so redact the FULL text before clipping — a secret
        # straddling char 200 would otherwise survive the persisted WorkerTest detail unmatchable.
        return WorkerTest(name, ERROR, redact(err)[:200], covered=False)
    blob = (out + err).lower()
    # argparse: 0 on --help, 2 on missing-args-with-usage. R workers may not support
    # --help at all — as long as they didn't die on an *import*, that's acceptable.
    if rc == 0 or (rc == 2 and "usage" in blob):
        return WorkerTest(name, PASS, f"rc={rc}", covered=False)
    # ``there is no package called`` is R's import error: an R worker whose library is missing printed
    # it, matched none of the Python markers, and rolled up WARN -- which the verdict counts as neither
    # pass nor fail, so the broken tool still read "ready" (hunt 2026-09-30, uL4-honesty-1).
    if "traceback" in blob or "modulenotfound" in blob or "importerror" in blob or "there is no package called" in blob:
        tail = (err or out).strip().splitlines()[-1:] or [""]
        # redact-before-clip (Class-4): tail is RAW worker stderr — redact FULL before clipping.
        return WorkerTest(name, FAIL, f"import error: {redact(tail[0])[:200]}", covered=False)
    return WorkerTest(name, WARN, f"rc={rc} (no usage text)", covered=False)


def run_tier1_server(
    conda: Conda,
    spec: ToolSpec,
    basic_env: str,
    *,
    io: PromptIO,
    log: SessionLog | None = None,
    remediation=None,
) -> ServerTest:
    """Run every mini-data smoke case for ``spec``'s worker through ``<basic>_<server>``.

    On a FAILure, the install-specialized :class:`InstallerScientist` ReAct loop tries to
    heal a broken *environment* case by case and re-runs the worker to confirm (deterministic,
    default on). ``remediation`` is the wizard's stdlib chat handle: with it plus the opt-in
    ``SOG_TEST_LLM_REMEDIATION`` gate, the loop may also consult the fail-closed LLM planner.
    The legacy ``_maybe_self_review_tier1`` hook is layered *after* it, only if still failing
    and separately gated.
    """
    managed = spec.target_env(basic_env)
    # Env-reuse (R1): probe the env the config is actually wired to. When the managed
    # <basic>_<server> env was never built because an existing healthy env already satisfies this
    # tool, `effective_tool_env` returns that reused env — so the worker runs there (real mini-data
    # coverage) instead of SKIPping with a misleading "interpreter missing". A built tool keeps
    # target == managed, so the normal path stays byte-identical.
    target = provision.effective_tool_env(conda, spec, basic_env)
    reused = target != managed
    st = ServerTest(server_key=spec.server_key, target_env=target, worker_kind=spec.worker_kind)
    envs_root = _conda_envs_root(conda, target)  # where conda says the env is, not a guess (u37-setup-checks-11)

    st.worker_tests = _cases_in(spec, target, envs_root)
    st.status = _rollup(st.worker_tests)

    # Deterministic, bounded, env-only auto-repair (the default self-heal path). Both this and the
    # LLM self-review below operate on the MANAGED env (InstallerScientist / self_review_loop build
    # and mutate spec.target_env(basic_env)); when we're validating a *reused* env they would rebuild
    # the very clone reuse avoided and can't touch the user's env anyway (assert_deletable_env refuses
    # a foreign env) — so a FAIL on a reused env is reported honestly, without them.
    if st.status == FAIL and not reused:
        _attempt_env_repairs(conda, spec, basic_env, target, st, io=io, log=log, remediation=remediation)

    # Optional deep tier: LLM self-review, gated + only if the env is still failing.
    if st.status == FAIL and not reused:
        st.self_review = _maybe_self_review_tier1(conda, spec, basic_env, target, _bad_detail(st), io, log)
        if st.self_review and st.self_review.get("repaired"):
            # A claimed repair is trusted ONLY after the SAME authoritative worker re-run the
            # deterministic path uses (:476-477 / :571-572) — never the self-review's own
            # `import <import_check>` probe, and never its bare `success` when import_check is
            # empty. The confirmed re-pass is the one and only thing that flips FAIL→PASS.
            #
            # PASS, not `!= FAIL`: three of the four statuses `_rollup` can return satisfy the
            # weaker test and only one of them means a case ran and passed. SKIP is what a re-run
            # yields when the case's input isn't on this clone, when the tool wants a GPU there
            # isn't one of, and when a FAILure is softened as a demo-data shape mismatch — a
            # verdict that moved FAIL→SKIP verified nothing. WARN is no better: it is built at
            # exactly two places, both from the `--help` probe and both `covered=False`, and since
            # `_rollup` uses `pool = covered or worker_tests`, a WARN roll-up means no authoritative
            # case ran at all.
            st.worker_tests = _cases_in(spec, target, envs_root)
            st.status = _rollup(st.worker_tests)
            st.self_review["confirmed"] = st.status == PASS
            if st.status == PASS:
                io.ok(f"{target}: Tier-1 recovered after LLM self-review")
            else:
                # Not "the worker still fails" — that is untrue of the SKIP and WARN this now keeps.
                io.warn(f"{target}: self-review claimed a fix but no worker case passed — kept {st.status}")

    icon = {PASS: io.ok, FAIL: io.err, WARN: io.warn}.get(st.status, io.note)
    probe_only = " — import probe only, no mini-data case" if st.status == PASS and not st.verified_on_mini_data else ""
    icon(f"{target}: Tier-1 {st.status} ({len(st.worker_tests)} case(s){probe_only})")
    if log is not None:
        log.event(
            "tier1",
            server=spec.server_key,
            env=target,
            status=st.status,
            cases=len(st.worker_tests),
            repairs=len(st.repairs),
        )
    return st


def _bad_detail(st: ServerTest) -> str:
    """The concatenated evidence lines from a server's failing worker cases — the
    text fed to ``classify_failure``."""
    return "\n".join(w.detail for w in st.worker_tests if w.status in _BAD)


# Extensions that mark a smoke-case argument as an *input data file/dir* (so an absent one on
# this clone means SKIP-with-reason, not FAIL). A path is treated as a required input if it is
# absolute AND (ends with one of these OR lives under a ``test_data`` tree).
_DATA_SUFFIXES = (".h5ad", ".h5", ".gem", ".csv", ".loom", ".mtx", ".tsv")

# Unambiguous GPU/CUDA failure signatures. A Tier-1 FAIL matching one of these on a host with no
# GPU is reclassified SKIP ("requires a GPU"), never a hard fail — the smoke cases all force CPU
# mode, so a genuine CUDA error here means the tool cannot honor that on this box.
_GPU_FAIL_SIGNS = ("cuda", "cudnn", "cublas", "nvidia", "no gpu", "requires gpu", "requires a gpu", "device=cuda")


def _has_gpu() -> bool:
    """True iff an NVIDIA GPU is visible (same cheap probe preflight uses)."""
    import shutil

    return shutil.which("nvidia-smi") is not None


def _looks_like_gpu_failure(text: str) -> bool:
    low = (text or "").lower()
    return any(s in low for s in _GPU_FAIL_SIGNS)


# A tool whose worker REQUIRES the richer spatial structure a full Visium slide carries — a
# histology library under ``adata.uns['spatial']``, image tiles, a ``library_id`` — dies with a
# data-SHAPE error (not an env fault) when handed the small git-tracked demo fallback, which is a
# bare expression matrix (no ``uns['spatial']``). stLearn's ``convert_scanpy`` is the canonical
# case: ``adata.uns["spatial"]`` → ``KeyError: 'spatial'``. ``classify_failure`` correctly returns
# ``None`` for these (nothing for the self-heal loop to fix), so on the DEMO fallback we reclassify
# the FAIL to SKIP-with-reason — the env is healthy, it just can't be fully exercised on demo data.
_DEMO_DATA_STRUCTURE_SIGNS = (
    "keyerror: 'spatial'",
    'keyerror: "spatial"',
    "uns['spatial']",
    'uns["spatial"]',
    "no histology",
    "requires histology",
    "histology image",
    "library_id",
)


def _looks_like_demo_data_incompatible(text: str) -> bool:
    """A FAIL whose signature is "the input slide lacks spatial/histology structure this tool
    needs" — safe to SKIP *only* on the demo fallback (see :func:`_is_fallback_spatial`)."""
    low = (text or "").lower()
    return any(s in low for s in _DEMO_DATA_STRUCTURE_SIGNS)


def _is_fallback_spatial(spatial: str | None) -> bool:
    """True iff the resolved Tier-1 spatial file is the small git-tracked demo fallback
    (:data:`constants.FALLBACK_SPATIAL_REL`, a fresh clone) rather than the validated mini/full
    slide. Gating the data-incompatibility SKIP on this guarantees a genuine tool break on REAL
    data still FAILs — only the reduced demo fixture earns the softer verdict."""
    if not spatial:
        return False
    try:
        return os.path.abspath(spatial) == os.path.abspath(str(constants.fallback_spatial_dataset()))
    except (OSError, ValueError):
        return False


def _is_required_input_path(token: str) -> bool:
    """Does this argument value name an input data file/dir the worker must read?

    Only absolute paths qualify (so ``{out}``, ``cpu``, ``0.6``, obs-key names never do). A
    genuine input is an absolute path that either ends in a data suffix or sits under a
    ``test_data`` tree (covers dir inputs like ``visium_mock/spatial`` and the ``istar_data/``
    prefix). Uses ``os.path.isabs`` (OS-aware) rather than a POSIX-only ``startswith('/')`` so a
    Windows ``C:\\...`` input path is recognized too; on POSIX the two are identical."""
    t = token.strip()
    if not os.path.isabs(t):
        return False
    low = t.lower()
    return low.endswith(_DATA_SUFFIXES) or "test_data" in low


def _resolve_tier1_data() -> tuple[str | None, str | None]:
    """Clone-local ``(spatial, sc_ref)`` for the Tier-1 cases: ``test/test_data``'s ``mini_*.h5ad``
    when present, else its ``creation_demo`` demo files.

    Returns absolute paths, or ``None`` for either if neither the mini file nor the tracked demo
    exists — the caller then leaves the case's original (absent) path in place, which the
    missing-input check turns into a SKIP-with-reason rather than a FAIL."""
    sp = testdata.resolve_source()  # mini_spatial.h5ad else creation_demo/demo_spatial.h5ad
    sc = testdata.resolve_sc_ref()  # mini_sc_ref.h5ad  else creation_demo/demo_sc_ref.h5ad
    return (str(sp) if sp else None, str(sc) if sc else None)


def _rewrite_case_data(args, smoke_spatial: str, smoke_sc_ref: str, spatial: str | None, sc_ref: str | None):
    """Repoint a smoke case's hardcoded ``SPATIAL``/``SC_REF`` paths at the clone-local data and
    report any required input that is absent here.

    The stock harness bakes THIS repo's absolute ``/workspace/.../test/test_data/mini_*.h5ad``
    into every case; those files are gitignored, so on the user's clone they don't exist and the
    worker died → the whole run read "WITH PROBLEMS". We substitute the resolved clone-local data
    (mini when present, else the tracked demo) by value-equality, then flag any remaining required
    input path (a 3-D variant, the visium/istar/gem/coords mocks) that isn't on this clone so the
    runner SKIPs it with a reason instead of FAILing. Walks lists and JSON-payload dicts alike;
    returns ``(new_args, missing_basenames)``."""
    missing: list[str] = []

    def fix(s: str) -> str:
        if spatial and s == smoke_spatial:
            return spatial
        if sc_ref and s == smoke_sc_ref:
            return sc_ref
        return s

    def walk(obj):
        if isinstance(obj, str):
            new = fix(obj)
            if _is_required_input_path(new) and not os.path.exists(new):
                missing.append(os.path.basename(new.rstrip("/")) or new)
            return new
        if isinstance(obj, list):
            return [walk(x) for x in obj]
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        return obj

    return walk(args), missing


def _run_worker_cases(spec: ToolSpec, target: str, envs_root: str | None = None) -> list[WorkerTest]:
    """Run every mini-data smoke case for ``spec``'s worker in ``target`` (or a --help
    import probe when the server has no smoke case).

    Each case's hardcoded ``SPATIAL``/``SC_REF`` paths are repointed at the clone-local data
    (mini when present, else the git-tracked demo files) so the check works on any clone; a case
    whose required input isn't available here, or that only fails for lack of a GPU on a GPU-less
    box, is SKIPped with a reason rather than FAILed.

    Factored out of :func:`run_tier1_server` so the auto-repair loop can RE-RUN the exact
    same authoritative check after a repair — the confirmed re-pass is the only thing that
    flips FAIL→PASS.
    """
    worker_file = spec.worker_file or ""
    idx = _tests_index()
    cases = idx.get(worker_file, []) if worker_file else []
    out: list[WorkerTest] = []
    if cases:
        smoke = _smoke_harness()  # lazy: pulls the data-path constants
        SC_REF, SPATIAL = smoke.SC_REF, smoke.SPATIAL
        rebase_case_args, run_one, scratch_out_root = smoke.rebase_case_args, smoke.run_one, smoke.scratch_out_root

        from .envdoctor import classify_failure  # lazy: the same env/data verdict the self-heal loop trusts

        spatial, sc_ref = _resolve_tier1_data()
        has_gpu = _has_gpu()
        on_demo_fallback = _is_fallback_spatial(spatial)
        tools_dir = _resolve_tools_dir(worker_file)
        root = envs_root or _default_envs_root()  # the live conda envs dir — never /opt/conda
        out_root = scratch_out_root()  # never the tracked smoke_outputs (hunt 2026-09-30, u39-test-infra-3)
        for entry in cases:
            name, _env, worker, mode, args = entry[:5]
            timeout_override = entry[5] if len(entry) > 5 else constants.WORKER_TEST_TIMEOUT_SEC
            # Rebased first, so the missing-input check looks for an earlier case's output (the seurat_*
            # entries' SEURAT_RDS) in this run's root, where run_one writes it, not under OUTBASE.
            new_args, missing = _rewrite_case_data(rebase_case_args(args, out_root), SPATIAL, SC_REF, spatial, sc_ref)
            if missing:
                out.append(
                    WorkerTest(name, SKIP, f"no mini-data on this clone for: {', '.join(missing)}", covered=False)
                )
                continue
            status, detail = run_one(
                name,
                target,
                worker,
                mode,
                new_args,
                timeout_override,
                tools_dir=tools_dir,
                envs_root=root,
                out_root=out_root,
            )
            if status in _BAD and spec.gpu and not has_gpu and _looks_like_gpu_failure(detail):
                out.append(WorkerTest(name, SKIP, f"requires a GPU (none detected): {detail[:160]}", covered=False))
                continue
            # Soften FAIL→SKIP ONLY when the failure is a demo-data shape mismatch AND the env itself is
            # sound. ``classify_failure(...) is None`` is the hard gate: if envdoctor can name an
            # env-repairable fault (ModuleNotFoundError, broken ABI, missing R/pip dist, off-index wheel,
            # …) the env is BROKEN — it stays FAIL and flows into the self-heal repair loop, even if the
            # traceback also happens to mention ``uns['spatial']``/``library_id``. We never hide a broken
            # env behind "provide a real slide"; we only excuse a tool that a bare demo matrix can't feed.
            if (
                status in _BAD
                and on_demo_fallback
                and _looks_like_demo_data_incompatible(detail)
                and classify_failure(detail, spec=spec) is None
            ):
                out.append(
                    WorkerTest(
                        name,
                        SKIP,
                        "needs a full Visium slide with histology (uns['spatial']); the fresh-clone demo "
                        f"data is a bare matrix — env looks healthy, provide a real slide to fully test: {detail[:120]}",
                        covered=False,
                    )
                )
                continue
            out.append(WorkerTest(name, status, detail, covered=True))
    else:
        worker_abs = os.path.join(_resolve_tools_dir(worker_file), worker_file) if worker_file else ""
        out.append(
            _worker_help_probe(target, spec.worker_kind, worker_abs, envs_root=envs_root or _default_envs_root())
        )
    return out


def _attempt_env_repairs(conda, spec, basic_env, target, st: ServerTest, *, io, log=None, remediation=None) -> None:
    """Detect an ENVIRONMENT problem behind a Tier-1 FAILure and self-heal it, case by case.

    This is the Tier-1 failure surface of the install-specialized :class:`InstallerScientist`
    ReAct loop. The agent is seeded with the failing worker cases' evidence (``_bad_detail``);
    each turn it classifies the error, applies a guarded env-only repair, then this closure
    RE-RUNS the authoritative worker cases (the same check :func:`run_tier1_server` trusts) and
    reports the *fresh* detail — so the worker logs genuinely participate in the loop. A non-env
    failure (LLM/task/timeout/special-input) is surfaced honestly and never "repaired"; the
    verdict flips to PASS only on a confirmed re-pass. ``st`` is mutated in place.

    Deterministic repair is bounded (``MAX_ENV_REPAIRS``) and default-on (``SOG_ENV_AUTOREPAIR``);
    the LLM hand-off lane is opt-in (``SOG_TEST_LLM_REMEDIATION`` + a wired ``remediation`` chat
    handle). Env-only; never touches agent/tool source.
    """
    # Documented SOG_* falsy set (``config._env_bool``: strip, lower, false/0/no/off), inline for
    # the reason recorded at ``provision._attempt_provision_repairs``. ``off`` and the strip were
    # both missing here, so the documented opt-out did not opt out.
    if os.environ.get("SOG_ENV_AUTOREPAIR", "1").strip().lower() in ("false", "0", "no", "off"):
        return
    from .installer_scientist import InstallerScientist

    seed = {"first": True}
    envs_root = _conda_envs_root(conda, target)  # the re-run checks the env where conda keeps it

    def observe() -> str:
        # First turn: the evidence already gathered. Subsequent turns: RE-RUN the authoritative
        # worker cases and report the fresh failing detail — ``""`` once no case is BAD.
        if seed["first"]:
            seed["first"] = False
            return _bad_detail(st)
        st.worker_tests = _cases_in(spec, target, envs_root)
        st.status = _rollup(st.worker_tests)
        return _bad_detail(st)

    def recheck() -> bool:
        # PASS, not `!= FAIL` — same reason as the self-review gate above. `_bad_detail` finds
        # nothing BAD in a SKIP or a WARN, so `observe()` returns "" for both and the ReAct loop
        # takes its `if not text:` branch; this predicate is then the only thing standing between a
        # re-run that tested nothing and `installer_scientist` finishing with "env healthy".
        return st.status == PASS

    gate = os.environ.get("SOG_TEST_LLM_REMEDIATION", "").strip().lower() in _TRUTHY
    chat = remediation if (gate and remediation is not None) else None
    agent = InstallerScientist(conda, spec, basic_env, io=io, chat=chat, llm_enabled=bool(chat), log=log)
    # C6: persist THIS Tier-1 failure's seed evidence to its own sink and point the self-heal loop's log
    # monitor at it, so gather() reads the real Tier-1 worker error — never a stale provision-era build.log.
    seed_log = _persist_log_artifact(agent.target, "tier1.log", _bad_detail(st))
    try:
        with progress.think(
            getattr(conda, "progress", None), f"Self-heal · {agent.target}", fallback_stream=io.out
        ) as box:
            outcome = agent.remediate(observe=observe, recheck=recheck, box=box, log_path=seed_log)
    except Exception as exc:  # a repair must never sink the test phase
        io.err(f"{target}: env repair crashed: {exc}")
        if log is not None:
            # redact-before-clip (Class-4): generic except → redact FULL text before clipping.
            log.event("env_repair_crash", server=spec.server_key, env=target, error=redact(str(exc))[:200])
        return
    st.repairs.extend(outcome.repairs)
    if outcome.repaired:
        io.ok(f"{target}: Tier-1 recovered after env repair ({outcome.last_action})")
    elif outcome.needs_attention and outcome.reason.startswith("surface:"):
        # A genuine non-env blocker (external token / special input / data shape / …): tell the
        # user exactly what to do rather than a bare FAIL. The worker verdict stays FAIL; record
        # the reason on the ServerTest so it is persisted and surfaced in the demo summary.
        st.needs_attention = outcome.needs_attention
        io.warn(f"{target}: {outcome.needs_attention}")


def _maybe_self_review_tier1(conda, spec, basic_env, target_env, stderr, io, log):
    """Gated, best-effort LLM self-review after the deterministic envdoctor path could
    not fix a Tier-1 failure. Off unless ``SOG_SELF_REVIEW_ENABLED`` is truthy in the documented
    ``config._env_bool`` sense (stripped, lowered, one of true/1/yes/on); never
    raises. Builds a fully-populated ``RemediationContext`` (the previously-inert hook
    passed only ``env_name`` and mis-named ``error_msg``, so every call raised
    ``TypeError`` and repaired nothing). On a claimed success it re-probes ``import
    <pkg>`` in the env to confirm before flipping the verdict.

    Env-boundary guard (SETUP-2): the agent-side ``self_review_loop`` MUTATES ``target_env`` via
    ``conda install -n <env>``. Before invoking it we assert ``target_env`` is a managed ``<basic>_*``
    tool env — never the base, a foreign env, or a :data:`constants.PROTECTED_ENVS` entry — reusing the
    same ``assert_deletable_env`` boundary the destructive paths use (delete-safe ⇒ install-safe). A
    refused env skips the review entirely (env untouched), rather than silently mutating something we
    do not own."""
    # Same set, and the same reasoning, as ``provision._maybe_self_review`` -- see the note there:
    # the documented ``config._env_bool`` spellings, not this module's ``_TRUTHY`` (which carries an
    # extra ``y`` that ``self_review_loop._enabled()`` reads as off).
    if os.environ.get("SOG_SELF_REVIEW_ENABLED", "").strip().lower() not in ("true", "1", "yes", "on"):
        return None
    try:
        constants.assert_deletable_env(basic_env, target_env)
    except PermissionError as exc:
        if log is not None:
            log.event("self_review_refused", server=spec.server_key, env=target_env, reason=str(exc))
        io.warn(f"{target_env}: skipping self-review — {exc}")
        return {"attempted": False, "refused": str(exc)[:200]}
    try:
        from tools_user.self_review import self_review_loop

        from .envdoctor import build_remediation_context
    except Exception as exc:
        if log is not None:
            log.event("self_review_unavailable", server=spec.server_key, error=str(exc))
        return None
    out: dict = {"attempted": True}
    try:
        ctx = build_remediation_context(spec, target_env)
        success, final_error, history = self_review_loop(ctx, stderr, stderr=stderr)
        out["success"] = bool(success)
        out["final_error"] = redact(str(final_error))[:200]  # redact-before-clip: self_review is persisted (C6)
        out["rounds"] = len(history)
        # Confirm with a cheap in-env import probe (reuses the same signal as provision). Kept INSIDE
        # this try: ``conda.run(check=False)`` still raises ``CondaError`` on a timeout/OSError, and
        # this hook is documented as best-effort / never-raising — its unguarded caller (:772) would
        # otherwise let a slow-box probe error unwind through the whole Tier-1 test (SETUP-3).
        pkg = spec.import_check
        if pkg:
            probe = ["python", "-c", f"import {pkg}"]
            if spec.worker_kind == "rscript":
                probe = ["Rscript", "-e", f"library({pkg})"]
            res = conda.run(target_env, probe, timeout=constants.CONDA_RUN_TIMEOUT_SEC, check=False)
            out["repaired"] = bool(res.ok)
            if res.ok:
                io.ok(f"{target_env}: repaired via self-review")
        else:
            out["repaired"] = bool(out.get("success"))
    except Exception as exc:
        out["error"] = redact(str(exc))[:200]  # redact-before-clip: self_review is persisted (C6)
    return out


# --------------------------------------------------------------------------- #
# Tier-2 — one agent.go per category (gated)
# --------------------------------------------------------------------------- #
def _best_passed_server(cat: Category, passed: set[str]) -> str | None:
    """Highest-priority server in the category that cleared Tier-1 (else None)."""
    cands = [s for s in cat.servers if s in passed]
    if not cands:
        return None
    return min(cands, key=lambda s: cat.server_meta.get(s, {}).get("priority", 99))


def _tier2_prompt(category: str, filename: str, server_key: str = "") -> str:
    """The category's natural-language task plus a minimal harness note naming the staged file.

    ``category_prompts.py`` stays path-free (a real biologist's voice); this pointer is a
    *test-harness* affordance so the real-data run deterministically loads THE staged dataset —
    and, in the break-env capstone, actually invokes the tool whose env we broke (without which
    there would be no masked error to detect).

    ``server_key`` is the tool this run is reported against ("asking the agent … via <server>"). The
    prompt used to name neither it nor an absolute path, so the recommender's pin never fired and a
    Tier-2 PASS "via graphst" said nothing about graphst (hunt 2026-09-30, u37-setup-checks-2). When
    the tool has a verified natural name it is named, in the demo's words, and ``filename`` should be
    the staged file's ABSOLUTE path: the pin only fires on an existing absolute ``.h5ad`` path."""
    task = prompt_for(category)
    name = display_name(server_key) if server_key else None
    if name:
        task = with_tool_named(task, name)
    return f"{task}\n\n(For this run, use the dataset already staged in your data lake as the file '{filename}'.)"


# LLM auth/quota failure signatures. These are NOT env issues (classify_failure→None): a real but
# rejected/exhausted key must degrade to SKIP-with-warning, never a hard ERROR (documented
# "missing key ⇒ SKIP" contract). Kept specific enough not to swallow a genuine tool RuntimeError.
_LLM_AUTH_SIGNS = (
    "authenticationerror",
    "authentication_error",
    "invalid api key",
    "invalid_api_key",
    "incorrect api key",
    "invalid x-api-key",
    "no api key",
    "missing api key",
    "api key not",
    "unauthorized",
    "permission_error",
    "rate limit",
    "rate_limit",
    "ratelimiterror",
    "insufficient_quota",
    "insufficient quota",
    "credit balance",
    "billing",
    "quota exceeded",
)


def _looks_like_llm_auth_error(text: str) -> bool:
    """Heuristic: does this error text read as an LLM auth/quota rejection (not a tool failure)?"""
    low = (text or "").lower()
    return any(s in low for s in _LLM_AUTH_SIGNS)


def _score_probe(ct: CategoryTest, probe: ProbeResult, dt: float) -> None:
    """Turn a (final) probe outcome into the category's status/detail.

    A *masked* env issue that survived repair is a FAIL even when the agent returned a truthy
    answer — a hollow answer hiding a broken tool must never read as PASS. Agent-unavailable
    (import/contract) and an LLM auth/quota rejection are both SKIP-with-warning: provisioning +
    Tier-1 already proved the tool envs, and a dead key must never hard-fail the install."""
    if probe.timed_out:
        ct.status, ct.detail = TIMEOUT, f">{constants.AGENT_TEST_TIMEOUT_SEC}s (killed)"
        return
    # A detected-but-unresolved masked env issue is a FAIL — checked BEFORE the generic error→SKIP
    # branch so a re-run that came back "agent unavailable" can't launder a known-broken tool env
    # into a SKIP. (A clean re-run clears the flag upstream, so reaching here means still-broken.)
    if ct.masked_env_issue:
        ct.status, ct.detail = FAIL, f"masked env issue unresolved: {ct.masked_env_issue}"
        return
    if probe.error:
        low = probe.error.lower()  # raw, for keyword routing only (never persisted)
        # redact-before-clip (Class-4): probe.error is RAW output from the _agent_probe subprocess, which
        # holds none of the parent's registered secrets and cannot self-redact. Clipping first would let a
        # secret straddling the cut survive downstream log.event()/_sanitize (redact is exact-substring).
        err = redact(probe.error)
        if "no probe contract" in low or "modulenotfound" in low or "importerror" in low or "no module named" in low:
            ct.status, ct.detail = SKIP, f"agent unavailable: {err[:160]}"
        elif _looks_like_llm_auth_error(low):
            # Backstop: a plausible-but-dead key that slipped the pre-gate → SKIP, not ERROR.
            ct.status, ct.detail = SKIP, f"LLM key rejected (auth/quota) — add a working key: {err[:120]}"
        else:
            ct.status, ct.detail = ERROR, err[:200]
        return
    # A turn that stopped early is not a working pipeline, whatever text it left behind. The stream
    # swallows a provider error into a note, so ``probe.error`` stays empty for a rejected key, and the
    # answer used to be the user's own prompt read back off the initial state -- scored PASS (hunt
    # 2026-09-30, u37-setup-checks-1, uL4-honesty-6). The auth backstop reads the note as well.
    if probe.degraded:
        note = redact(probe.degraded)  # redact-before-clip (Class-4): raw text from the probe subprocess
        if _looks_like_llm_auth_error(probe.degraded):
            ct.status, ct.detail = SKIP, f"LLM key rejected (auth/quota) — add a working key: {note[:120]}"
        else:
            ct.status, ct.detail = ERROR, f"agent stopped early ({dt:.0f}s): {note[:160]}"
        return
    if not probe.answer.strip():
        ct.status, ct.detail = FAIL, f"empty answer ({dt:.0f}s)"
        return
    ct.status, ct.detail = PASS, f"{dt:.0f}s, {len(probe.answer)} chars"


# The code the agent ran itself, in its own REPL (the base env): ``<execute>`` bodies and fenced blocks.
_CODE_REGION_RE = re.compile(r"<execute>(.*?)(?:</execute>|$)|```[\w-]*\n(.*?)(?:```|$)", re.S | re.I)
# The module named in a ``No module named 'X'`` an ENV_BROKEN issue carries for an R spec.
_NO_MODULE_DETAIL_RE = re.compile(r"No module named\s*['\"]?([A-Za-z_][\w.]*)")
# One dependency line of an env recipe (conda or its ``pip:`` sub-list): the name before any pin.
_RECIPE_DEP_RE = re.compile(r"^\s*-\s*([A-Za-z0-9_.\-]+)", re.M)


def _dist_key(name: str) -> str:
    """A package/module name folded the way pip folds distribution names (case, ``-``/``_``/``.``)."""
    return re.sub(r"[-_.]+", "_", name or "").lower()


def _tool_packages(spec: ToolSpec) -> set[str]:
    """What the tool's own env is known to carry: its ``import_check`` and its recipe's dependencies.

    Folded with :func:`_dist_key`. Best-effort -- an unreadable recipe leaves just the import check."""
    names = {_dist_key((spec.import_check or "").split(".")[0])} - {""}
    recipe = getattr(spec, "recipe", None)
    if recipe:
        path = constants.resolve_recipe_path(recipe)
        try:
            names |= {_dist_key(m) for m in _RECIPE_DEP_RE.findall(path.read_text(encoding="utf-8"))}
        except (OSError, UnicodeDecodeError):
            pass
    return names


def _missing_module_of(issue) -> str | None:
    """The top-level module an import-miss issue is about, or ``None`` when it is not an import miss."""
    from .envdoctor import EnvIssueKind

    if issue.kind == EnvIssueKind.MISSING_PY_MODULE and issue.package:
        return issue.package.split(".")[0]
    if issue.kind == EnvIssueKind.ENV_BROKEN:  # an R spec's reading of a Python "No module named"
        m = _NO_MODULE_DETAIL_RE.search(issue.detail or "")
        if m:
            return m.group(1).split(".")[0]
    return None


def _agent_code(step: str) -> str:
    """The code the agent ran itself in one log ``step``: its ``<execute>`` bodies and fenced blocks.

    A Human message holds none -- the prompt (with the recommender's call skeleton) and the loop's
    nudges are text the agent was given, not code it ran (the reading ``demo._route_hint`` uses)."""
    s = str(step or "")
    head, _, body = s.lstrip().partition("\n") if s.lstrip().startswith("=") else ("", "", s)
    if "human message" in head.lower():
        return ""
    return "\n".join(a or b for a, b in _CODE_REGION_RE.findall(body))


def _code_behind(log: list[str], i: int) -> str:
    """The step whose code produced step ``i``: ``i`` itself when it holds the agent's own code, else
    the nearest earlier step that does (an ``<observation>`` is a message of its own, after the
    ``<execute>`` it reports on). ``""`` when no earlier step ran any code."""
    for j in range(i, -1, -1):
        if _agent_code(log[j]).strip():
            return log[j]
    return ""


def _calls_the_tool(spec: ToolSpec, code: str) -> bool:
    """Whether ``code`` calls one of ``spec``'s own MCP functions (``graphst_spatial_clustering(...)``)."""
    names = [f.name for f in (getattr(spec, "functions", None) or []) if getattr(f, "name", "")]
    return bool(names) and re.search(rf"\b(?:{'|'.join(map(re.escape, names))})\s*\(", code) is not None


def _not_the_tools(issue, spec: ToolSpec | None, log_text: str) -> str:
    """Why a masked import miss is NOT ``spec``'s env failure, or ``""`` when it may be.

    ``classify_failure`` reads the whole agent transcript and has no idea which env produced an error.
    The Tier-2 prompt does not force the tool, so the agent often improvises ``import squidpy`` /
    ``import cellcharter`` in its own REPL -- the base env, which deliberately lacks the per-tool
    packages -- and that miss was blamed on whichever server Tier-2 had picked: a pip install into, or
    a RECREATE of, a healthy tool env, then a FAIL charged to a tool that never ran (hunt 2026-09-30,
    u37-setup-checks-2, uL4-honesty-3). Only an import miss the tool's own worker could have raised
    is its env's problem: not from an R worker, and not an import the agent wrote itself. A miss in
    the observation of a call to one of the tool's own functions IS its env's, whatever the package:
    the worker reports what it lacks. Otherwise the package must be one the tool's env is supposed to
    carry, compared by distribution name -- ``sklearn`` is ``scikit-learn`` in a recipe, ``PIL`` is
    ``pillow`` -- since comparing the bare import name dismissed those as "not a package the env
    carries" and repaired nothing (hunt 2026-09-30, u37-setup-checks-2 repair).

    ``log_text`` is the agent step(s) whose code produced the miss (see :func:`_code_behind`). Any
    other issue kind keeps the existing routing; with no ``spec`` there is nothing to attribute
    against, and detection stays unconditional."""
    from .envdoctor import _dist_for

    module = _missing_module_of(issue)
    if module is None or spec is None:
        return ""
    if getattr(spec, "worker_kind", "") == "rscript":
        return f"a Python import ('{module}') cannot come from {spec.server_key}'s R worker"
    code = "\n".join(a or b for a, b in _CODE_REGION_RE.findall(log_text or ""))
    if re.search(rf"^\s*(?:import|from)\s+{re.escape(module)}\b", code, re.M):
        return f"the agent imported '{module}' in its own code, which runs in the base env, not {spec.server_key}'s"
    if _calls_the_tool(spec, code):
        return ""
    if not {_dist_key(module), _dist_key(_dist_for(module))} & _tool_packages(spec):
        return f"'{module}' is not a package {spec.server_key}'s env carries"
    return ""


def _tool_env_issue(log: list[str], spec: ToolSpec | None):
    """``(issue, why_not, evidence)`` for an agent log: the env issue that belongs to ``spec``'s tool
    env and the text that shows it, or ``(None, reason, "")`` for one that does not (``(None, "", "")``
    when the log shows none at all). ``evidence`` is what the repair loop is seeded with, so it acts on
    the tool's error and not on whatever the whole transcript classifies as first.

    The whole log is classified first, exactly as before; an issue that is not an import miss, or a
    log with no ``spec`` to attribute against, is returned as it was. An import miss is then judged
    step by step, each against the code that produced THAT step. Judging every step against the whole
    transcript's code let the agent's improvised ``import GraphST`` early in a run hide the same miss
    raised later by graphst's own worker (hunt 2026-09-30, u37-setup-checks-2 repair). The first step
    that is the tool's wins; when no earlier step was set aside, the whole log stays the evidence."""
    from .envdoctor import classify_failure

    text = "\n".join(log)
    issue = classify_failure(text, spec=spec)
    if issue is None:
        return None, "", ""
    if spec is None or _missing_module_of(issue) is None:
        return issue, "", text
    why_not = ""
    for i, step in enumerate(log):
        found = classify_failure(step, spec=spec)
        if found is None:
            continue
        why = _not_the_tools(found, spec, _code_behind(log, i))
        if not why:
            return (found, "", step) if why_not else (issue, "", text)
        why_not = why_not or why
    if not why_not:  # the miss reads only across steps, never within one: judged against all the agent's code
        why_not = _not_the_tools(issue, spec, "\n".join(s for s in log if _agent_code(s).strip()))
        if not why_not:
            return issue, "", text
    return None, why_not, ""


def _maybe_unmask_and_repair(
    conda, basic_env, spec, staged, cfg, prompt, probe: ProbeResult, ct: CategoryTest, *, io, log=None, remediation=None
) -> ProbeResult:
    """Detect a MASKED tool env failure behind the agent's answer and self-heal it case by case,
    re-running the agent to confirm. Returns the (possibly re-run) probe.

    The agent's REPL swallows a tool's env error into an ``"Error: …"`` observation and the run
    returns cleanly, so the failure hides in ``probe.log``. We classify that surface; a non-env
    failure (LLM/task/timeout) classifies as ``None`` → nothing is touched. **Detection is
    unconditional** — an env issue always sets ``ct.masked_env_issue`` so scoring can never pass a
    hollow answer, even when auto-repair is off. Only the *repair* is gated by
    ``SOG_ENV_AUTOREPAIR`` (and needs the tool ``spec``): when enabled, the tool env is fixed with
    the guarded primitives (Phase 2) and the run is retried once; the flag is cleared only when the
    re-run is clean. Env-only — agent/tool source is never touched."""
    from .envdoctor import classify_failure

    # Two DIFFERENT surfaces, two DIFFERENT owners:
    #  • the agent's observation log holds a MASKED *tool* env error (a tool's ModuleNotFoundError
    #    swallowed into an "Error: …" observation) → a <basic>_<server> tool-env problem the guarded
    #    primitives can repair.
    #  • the probe's structured `error` / stderr tail holds a *construction/import* failure in the
    #    BASE agent env, BEFORE any tool ran → routing THAT to a tool-env pip_install is misdirected.
    #    It is reported (so the user sees it), never tool-repaired.
    issue, not_the_tools, evidence = _tool_env_issue(probe.log, spec)
    if not_the_tools:
        # Reported, never repaired: the miss is real but it is not this tool's env (see _not_the_tools).
        io.warn(
            f"Tier-2 {ct.category}: an import failed during the run, but not in {ct.server_key}'s env — {not_the_tools}"
        )
        if log is not None:
            log.event("tier2_env_issue_not_the_tools", category=ct.category, server=ct.server_key, why=not_the_tools)
        return probe
    if issue is None:
        base_issue = classify_failure("\n".join([probe.error, probe.stderr_tail]), spec=spec)
        if base_issue is not None:  # a base-env problem — surfaced, but NEVER a tool-env repair
            io.warn(
                f"Tier-2 {ct.category}: base-env issue at agent construction "
                f"({base_issue.kind}: {base_issue.package or '?'}) — reported, not routed to a tool env"
            )
            if log is not None:
                log.event(
                    "tier2_base_env_issue", category=ct.category, kind=str(base_issue.kind), package=base_issue.package
                )
        return probe  # nothing tool-env-shaped hiding behind the answer
    ct.masked_env_issue = str(issue.kind)  # detection is unconditional → a hollow answer never passes
    io.warn(f"Tier-2 {ct.category}: masked env issue behind the answer ({issue.kind}: {issue.package or '?'})")
    if log is not None:
        log.event(
            "tier2_unmasked", category=ct.category, server=ct.server_key, kind=str(issue.kind), package=issue.package
        )
    # Repair is the opt-out (default on) and needs the tool spec to target the right env.
    if os.environ.get("SOG_ENV_AUTOREPAIR", "1").strip().lower() in ("false", "0", "no", "off") or spec is None:
        return probe  # detected + reported, but not auto-repaired → keep the flag → FAIL
    # Env-reuse (R1): the InstallerScientist loop below builds/mutates the MANAGED <basic>_<server>
    # env, but a reused tool's config points at an existing foreign env instead — so a rebuild would
    # recreate the very clone reuse avoided AND recheck() (which re-runs the agent against the LIVE
    # config → the reused env) could never observe the "fix". The masked issue is already DETECTED and
    # reported above (the flag stays set → honest FAIL); we simply don't auto-repair a user-owned env
    # we must not mutate. Mirrors run_tier1_server's reuse gate.
    if provision.effective_tool_env(conda, spec, basic_env) != spec.target_env(basic_env):
        io.warn(f"Tier-2 {ct.category}: masked env issue is in a reused env — reported, not auto-repaired")
        return probe
    from .installer_scientist import InstallerScientist

    # Drive the install-specialized agent at the Tier-2 surface. ``observe`` re-runs the WHOLE
    # agent (the authoritative check — the masked error only exists in the agent's log) and hands
    # back a fresh masked-tool-env error to act on; ``recheck`` is the single authority on healthy,
    # matching the single-shot path's clear condition exactly (a COMPLETED re-run whose log no
    # longer carries the masked failure). ``holder`` keeps the latest probe to return + re-check.
    holder = {"probe": probe, "first": True}
    t2_target = spec.target_env(basic_env)  # <basic>_<server>; spec is non-None past the guard above

    def observe() -> str:
        if holder["first"]:
            holder["first"] = False
            return evidence  # the tool's own error, not the transcript's first miss (u37-setup-checks-2)
        p2 = _run_probe(
            conda,
            basic_env,
            str(staged.root),
            str(cfg),
            prompt,
            timeout=constants.AGENT_TEST_TIMEOUT_SEC,
            log_target=t2_target,  # C6: each re-run leaves a redacted <target>.agent.log post-mortem
        )
        holder["probe"] = p2
        if p2.timed_out or p2.error:
            return ""  # an incomplete re-run proves nothing → recheck() keeps the flag set → FAIL
        return _tool_env_issue(p2.log, spec)[2]

    def recheck() -> bool:
        p = holder["probe"]
        return not p.timed_out and not p.error and _tool_env_issue(p.log, spec)[0] is None

    gate = os.environ.get("SOG_TEST_LLM_REMEDIATION", "").strip().lower() in _TRUTHY
    chat = remediation if (gate and remediation is not None) else None
    agent = InstallerScientist(conda, spec, basic_env, io=io, chat=chat, llm_enabled=bool(chat), log=log)
    # C6: persist the masked-issue seed to its own Tier-2 sink and steer the loop's log monitor at it, so
    # gather() reasons on THIS agent run's real error instead of a stale provision-era build.log.
    seed_log = _persist_log_artifact(agent.target, "tier2.log", evidence)
    try:
        with progress.think(
            getattr(conda, "progress", None), f"Self-heal · {agent.target}", fallback_stream=io.out
        ) as box:
            outcome = agent.remediate(observe=observe, recheck=recheck, box=box, log_path=seed_log)
    except Exception as exc:  # a repair must never sink the test phase
        io.err(f"Tier-2 {ct.category}: env repair crashed: {exc}")
        return holder["probe"]
    ct.repairs.extend(outcome.repairs)
    if outcome.repaired:
        ct.masked_env_issue = None  # cleared: a confirmed clean re-run has no masked env failure
        io.ok(f"Tier-2 {ct.category}: masked env issue resolved after repair")
    return holder["probe"]


def run_tier2_category(
    category: str,
    server_key: str,
    *,
    conda: Conda,
    basic_env: str,
    spec: ToolSpec | None,
    io: PromptIO,
    do_eval: bool = False,
    log: SessionLog | None = None,
    remediation=None,
) -> CategoryTest:
    """Run ONE full ``agent.go`` for ``category`` on staged REAL data — out-of-process,
    cancellable, and un-masking.

    The real dataset is staged into an isolated root under the artifact dir (never the user's
    ``./data``); the agent runs in a *subprocess the wizard can kill* (fixing the old
    un-killable daemon-thread hang) in the base env, through the wizard's OWN MCP config. The
    returned log is scanned for a masked tool env failure and, if found, the tool env is
    repaired and the run is retried once. Env-only — agent/tool source is never touched. A
    missing key/config/dataset/tool is a SKIP, never a hard failure."""
    ct = CategoryTest(category=category, server_key=server_key)
    cfg = constants.generated_mcp_config()
    if not cfg.exists():
        ct.status, ct.detail = SKIP, "no generated MCP config (wiring did not run)"
        return ct
    try:
        staged = testdata.stage_mini_dataset()
    except OSError as exc:
        # A missing dataset (FileNotFoundError) OR a staging-infra failure — the mkdir/copy2 inside
        # stage_mini_dataset can raise PermissionError/ENOSPC (both OSError, NOT FileNotFoundError).
        # Either way we could not put real data in front of the agent, which is a Tier-2 SKIP (the
        # documented "never a hard failure"), not a tool ERROR the outer guard would blame on the
        # tool. The sibling demo.py stagers catch OSError for the same reason.
        # redact-before-clip (Class-4): OSError str can embed a path/URL — redact FULL before clipping.
        ct.status, ct.detail = SKIP, f"could not stage mini dataset: {redact(str(exc))[:160]}"
        io.note(f"Tier-2 {category}: skipped ({ct.detail})")
        return ct

    prompt = _tier2_prompt(category, str(staged.h5ad), server_key)
    io.say(f"  Tier-2 {category}: asking the agent on real data (via {server_key})…")
    t0 = time.time()
    probe = _run_probe(
        conda,
        basic_env,
        str(staged.root),
        str(cfg),
        prompt,
        timeout=constants.AGENT_TEST_TIMEOUT_SEC,
        log_target=constants.tool_env_name(basic_env, server_key),  # C6: durable redacted post-mortem
    )
    probe = _maybe_unmask_and_repair(
        conda, basic_env, spec, staged, cfg, prompt, probe, ct, io=io, log=log, remediation=remediation
    )
    dt = time.time() - t0

    _score_probe(ct, probe, dt)
    if do_eval:
        # ARI/NMI scoring is dataset/category-specific and intentionally never gates the
        # install; left as a best-effort hook that records nothing by default.
        ct.eval = None

    (io.ok if ct.status == PASS else io.warn if ct.status not in (ERROR, TIMEOUT) else io.err)(
        f"Tier-2 {category}: {ct.status} ({ct.detail})"
    )
    if log is not None:
        log.event(
            "tier2", category=category, server=server_key, status=ct.status, detail=ct.detail, repairs=len(ct.repairs)
        )
    return ct


# --------------------------------------------------------------------------- #
# Orchestration + results
# --------------------------------------------------------------------------- #
def run_tests(
    conda: Conda,
    specs: dict[str, ToolSpec],
    server_keys: list[str],
    basic_env: str,
    decision: TestDecision,
    categories: list[Category],
    *,
    io: PromptIO,
    log: SessionLog | None = None,
    remediation=None,
) -> TestReport:
    """Run Tier-1 (each provisioned server) then optional Tier-2 (each category),
    write per-server + summary JSON under the artifact dir, and return the report.

    ``server_keys`` are the servers that provisioning built successfully. ``remediation`` is the
    wizard's stdlib chat handle, threaded to both tiers' :class:`InstallerScientist` loops so the
    opt-in LLM hand-off lane (``SOG_TEST_LLM_REMEDIATION``) can consult the planner.
    """
    report = TestReport(basic_env=basic_env, started=_now_iso())
    artifacts_ok = _ensure_artifacts(io)

    if decision.run_tier1:
        io.section("Testing tools on a tiny dataset")
        harness_note = tier1_harness_note() if server_keys else None
        if harness_note:
            # Every server would otherwise fall back to the --help probe with nothing said: a
            # weaker check reported in the same words as the real one.
            io.warn(f"Tier-1 is degraded: {harness_note}. Each tool gets an import probe only, not a mini-data run.")
            if log is not None:
                log.event("tier1_degraded", reason=harness_note)
        for key in server_keys:
            spec = specs.get(key)
            if spec is None:
                report.servers.append(
                    ServerTest(
                        server_key=key,
                        target_env=constants.tool_env_name(basic_env, key),
                        status=SKIP,
                        worker_tests=[WorkerTest(key, SKIP, "no spec", covered=False)],
                    )
                )
                continue
            try:
                st = run_tier1_server(conda, spec, basic_env, io=io, log=log, remediation=remediation)
            except Exception as exc:  # one server's crash must not abort Tier-1/Tier-2
                io.err(f"{key} crashed during Tier-1: {exc}")
                # redact-before-clip (Class-4): a GENERIC except — `exc` need not be a CondaError (whose str
                # is message-only), so redact the FULL text before clipping, else a straddling secret's head
                # survives the persisted worker detail / log event unmatchable.
                st = ServerTest(
                    server_key=key,
                    target_env=constants.tool_env_name(basic_env, key),
                    status=ERROR,
                    worker_tests=[WorkerTest(key, ERROR, redact(f"crash: {exc}")[:300], covered=False)],
                )
                if log is not None:
                    log.event("tier1_crash", server=key, target=st.target_env, error=redact(str(exc))[:300])
            report.servers.append(st)
            if artifacts_ok:
                _write_server_json(st)

    if decision.run_tier2:
        # Tier-2 must run ONLY against tools Tier-1 actually passed. When Tier-1 ran
        # (report.servers non-empty) trust its verdict verbatim — even if that set is
        # empty (all failed), so a broken build can't leak into the LLM pipeline. The
        # `set(server_keys)` fallback is used ONLY when Tier-1 didn't run at all
        # (run_tier1=False): there's no per-server signal, so trust what was built.
        if report.servers:
            passed = {s.server_key for s in report.servers if s.status == PASS}
        else:
            passed = set(server_keys)
        cat_by_name = {c.name: c for c in categories}
        to_test = decision.categories_to_test or list(cat_by_name)
        if not _llm_key_present():
            why = _llm_key_skip_reason()
            io.warn(f"Tier-2 skipped: {why}")
            for name in to_test:
                report.categories.append(CategoryTest(category=name, status=SKIP, detail=why))
        else:
            io.section("Testing one full question per category")
            for name in to_test:
                cat = cat_by_name.get(name)
                srv = _best_passed_server(cat, passed) if cat else None
                if srv is None:
                    report.categories.append(
                        CategoryTest(category=name, status=SKIP, detail="no Tier-1-passing tool in category")
                    )
                    continue
                try:
                    ct = run_tier2_category(
                        name,
                        srv,
                        conda=conda,
                        basic_env=basic_env,
                        spec=specs.get(srv),
                        io=io,
                        do_eval=decision.tier2_eval,
                        log=log,
                        remediation=remediation,
                    )
                except Exception as exc:  # one category's crash must not lose the whole report
                    io.err(f"Tier-2 {name} crashed: {exc}")
                    # redact-before-clip (Class-4): generic except → redact FULL text before clipping.
                    if log is not None:
                        log.event("tier2_crash", category=name, server=srv, error=redact(str(exc))[:300])
                    ct = CategoryTest(category=name, server_key=srv, status=ERROR, detail=redact(f"crash: {exc}")[:200])
                report.categories.append(ct)

    report.finished = _now_iso()
    if artifacts_ok:
        _write_summary_json(report)
    _print_rollup(report, io)
    return report


def _ensure_artifacts(io: PromptIO) -> bool:
    """Create the test-artifact dir; degrade a read-only/full artifact FS to a one-time warning.

    The artifact tree lives under the repo (``constants.artifact_dir()`` = ``test/installation``). On an
    immutable-image / read-only-mount deploy (state redirected via ``SOG_SETUP_STATE_DIR``), or a disk
    that fills during the multi-GB env builds, the mkdir + JSON writes raise ``OSError``. Every OTHER
    repo-write phase already degrades gracefully — provision's setup-config write (``except Exception``),
    finalize's canonical/backup writes (``except Exception``), the build logs (``except OSError``). The
    test phase was the one unguarded repo-write cluster, so on a condition every other phase survives the
    whole run aborted with a generic "unexpected error" AFTER all envs built + passed, before finalize
    wired the config. Warn once and let testing proceed (results just aren't persisted to disk)."""
    try:
        constants.ensure_artifact_dir()
        return True
    except OSError as exc:
        io.warn(
            f"couldn't create the test-results dir ({exc.__class__.__name__}: {exc}); "
            "tools still run — results just won't be written to disk"
        )
        return False


def _write_server_json(st: ServerTest) -> None:
    try:
        path = constants.artifact_dir() / f"{st.server_key}.json"
        payload = st.to_dict()
        payload["written"] = _now_iso()
        # C6 root pass: this durable artifact bypasses SessionLog.event, so deep-redact the WHOLE
        # payload (self_review / repairs / worker_tests can echo a registered secret) before it lands on
        # disk. The per-site redact-before-clip above only guards a straddle; _sanitize covers the rest.
        path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    except OSError:
        pass  # best-effort: a disk that fills mid-run must not abort Tier-1 (dir-create already guarded)


def _write_summary_json(report: TestReport) -> None:
    try:
        path = constants.artifact_dir() / "summary.json"
        doc = report.summary()
        doc["servers"] = [s.to_dict() for s in report.servers]
        doc["categories"] = [c.to_dict() for c in report.categories]
        # C6 root pass (see _write_server_json): deep-redact the aggregate before it hits disk.
        path.write_text(json.dumps(_sanitize(doc), indent=2), encoding="utf-8")
    except OSError:
        pass  # best-effort: see _write_server_json


def _print_rollup(report: TestReport, io: PromptIO) -> None:
    s = report.summary()
    t1 = s["tier1"]["counts"]
    n_tot = sum(t1.values())
    # "passed on mini data" counts only servers a mini-data case actually verified; a --help import
    # probe is reported as what it is (hunt 2026-09-30, uL4-honesty-1).
    n_verified = len(s["tier1"]["verified_on_mini_data"])
    n_probe_only = len(s["tier1"]["import_probe_only"])
    line = f"  Tier-1: {n_verified}/{n_tot} tool env(s) passed on mini data"
    if n_probe_only:
        line += f"; {n_probe_only} more passed an import probe only (no mini-data case exists for them)"
    io.say(line)
    if report.categories:
        t2 = s["tier2"]["counts"]
        io.say(f"  Tier-2: {t2.get(PASS, 0)}/{sum(t2.values())} category pipeline(s) passed")
    io.note(f"results → {constants.artifact_dir()}")
