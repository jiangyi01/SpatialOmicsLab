"""MCP (Model Context Protocol) integration for the STCoscientist agent."""

import builtins
import inspect
import keyword
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from spatialomicsgym import platform_root
from spatialomicsgym.agent.tool_call_memo import (
    duplicate_call_notice,
    lookup_successful_call,
    remember_successful_call,
    tool_call_memo_key,
)
from spatialomicsgym.mcp_config_path import CANONICAL_CONFIG_DEFAULT
from spatialomicsgym.redaction import looks_like_credential
from spatialomicsgym.utils.execution import abandoned_by_its_caller

_MAX_ERROR_CHARS = 4000


#: Names the model's cells use as module aliases by convention. A tool bound over one breaks every
#: cell that relies on the alias, just as a tool bound over a builtin does.
_REPL_ALIASES = frozenset({"pd", "np", "sc", "ad", "sq", "plt", "sns", "os", "sys", "re", "json", "Path"})


def is_bindable_tool_name(name: Any) -> bool:
    """Whether ``name`` can be bound into the REPL namespace as a tool without breaking cells.

    Tool names were never checked, and the REPL binds each one as a global: a user tool named
    ``print`` made every ``print('...')`` in every account's cells raise TypeError, one named ``pd``
    broke pandas, and ``liana-run`` was registered and listed yet could never be called (hunt
    2026-09-30, u14-mcp-wiring-8). A name must be an identifier that is not a keyword, a builtin or a
    common REPL alias.
    """
    return (
        isinstance(name, str)
        and name.isidentifier()
        and not keyword.iskeyword(name)
        and not hasattr(builtins, name)
        and name not in _REPL_ALIASES
    )


def _exception_leaves(exc: BaseException, _depth: int = 0) -> list[BaseException]:
    """Flatten a (possibly nested) ``ExceptionGroup`` down to its non-group leaves."""
    subs = getattr(exc, "exceptions", None)
    if subs and isinstance(subs, (list, tuple)) and _depth < 16:
        leaves: list[BaseException] = []
        for sub in subs:
            if isinstance(sub, BaseException):
                leaves.extend(_exception_leaves(sub, _depth + 1))
        if leaves:
            return leaves
    return [exc]


def describe_exception(exc: BaseException) -> str:
    """Describe ``exc`` in terms of what actually went wrong.

    ``str()`` on an ``ExceptionGroup`` yields only ``"unhandled errors in a TaskGroup
    (N sub-exceptions)"`` -- it never includes its children. anyio, which the MCP stdio client
    runs on, wraps every failure in (often nested) task groups, so formatting the caught
    exception directly hands the ReAct loop no signal at all: a simple wrong-keyword tool call
    became unactionable noise and the loop abandoned the tool instead of retrying correctly.
    Flatten to the leaves and report those, bounded so the observation stays prompt-sized.
    """
    seen: set[str] = set()
    parts: list[str] = []
    for leaf in _exception_leaves(exc):
        text = str(leaf).strip()
        described = f"{type(leaf).__name__}: {text}" if text else type(leaf).__name__
        if described not in seen:
            seen.add(described)
            parts.append(described)

    out = "; ".join(parts) if parts else f"{type(exc).__name__}: {exc}"
    if len(out) > _MAX_ERROR_CHARS:
        suffix = " ...[truncated]"
        out = out[: _MAX_ERROR_CHARS - len(suffix)] + suffix
    return out


def _handshake_timeout_seconds() -> float:
    """The ``session.initialize``/``list_tools`` budget -- an operator knob, not a literal.

    Reads ``default_config.mcp_handshake_timeout_seconds`` (env override
    ``SOG_MCP_HANDSHAKE_TIMEOUT``); a missing/zero/garbage value degrades to the historical 30s.
    Lazy import: this module must stay importable without dragging config load-order forward.
    """
    try:
        from spatialomicsgym.config import default_config

        v = float(getattr(default_config, "mcp_handshake_timeout_seconds", 30.0))
    except Exception:
        return 30.0
    return v if v > 0 else 30.0


#: How much of a crashed portal's stderr to put in front of the model.
#:
#: THE BUG THIS EXISTS FOR. ``_interp_runnable`` and ``_missing_script_arg`` correctly refuse to
#: register a server whose interpreter or script is absent. A portal whose interpreter and script
#: both exist but which **crashes on import** passes both gates, is registered, and fails every
#: call with ``McpError: Connection closed`` -- one phrase naming no cause and no remedy. Its
#: traceback went to ``errlog=sys.stderr``, which ``_restore_real_stdio`` had just pointed at the
#: process's real stderr, while the ReAct observation is built from the REPL's captured *stdout*.
#: So the one thing that explains the failure was written to the one stream the model cannot read.
#:
#: That is the normal failure shape for an agent-written tool in ``tools_user/`` -- a bad import, a
#: typo at module scope -- which is exactly the loop that needs the traceback to self-repair. The
#: two neighbouring failure paths (handshake timeout, unknown tool) are both well-instrumented;
#: this one was not.
_PORTAL_STDERR_TAIL_CHARS = 2000


#: The first line of a rich log record, ``[09/30/26 14:14:32] INFO     Starting MCP server ...``. Rich
#: leaves the time column blank when a record shares the previous one's second, so the stamp is
#: optional; a wrapped record continues on lines that start with whitespace.
_RICH_RECORD_RE = re.compile(r"^(?:\[[^\]\n]*\])?\s*(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s{2,}\S")

#: Said instead of the tail when all the portal wrote was its startup chatter. ``env_fallback``
#: reads it: a portal that got as far as its banner did not die on import.
PORTAL_STARTED_NOTE = "had started -- its stderr holds only the startup banner -- when the connection closed"


def _drop_startup_chatter(text: str) -> str:
    """``text`` without FastMCP's banner or its INFO/DEBUG log records.

    Every stdio portal prints a box-drawn logo, an "Update available" advert and an INFO line as it
    starts. None of it is a reason for anything, and handed to the model beside a failure it read as
    one. WARNING and above are kept, with their continuation lines, as is anything else.
    """
    kept: list[str] = []
    in_quiet_record = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and "\u2500" <= stripped[0] <= "\u259f":
            in_quiet_record = False
            continue
        record = _RICH_RECORD_RE.match(line)
        if record:
            in_quiet_record = record.group(1) in ("DEBUG", "INFO")
            if in_quiet_record:
                continue
        elif in_quiet_record and line[:1].isspace():
            continue
        else:
            in_quiet_record = False
        kept.append(line)
    return "\n".join(kept).strip()


def _portal_stderr_note(text: str) -> str:
    """The stderr tail as one appended sentence, or ``""`` when there is nothing to say.

    For a portal that went down (see ``_portal_went_down``) and only then: after a tool's own
    error or a spent budget the portal was healthy and its stderr explains nothing.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    text = _drop_startup_chatter(raw)
    if not text:
        return f"\nThe tool's server process {PORTAL_STARTED_NOTE}."
    if len(text) > _PORTAL_STDERR_TAIL_CHARS:
        text = "...\n" + text[-_PORTAL_STDERR_TAIL_CHARS:]
    return (
        f"\nThe tool's server process wrote this to stderr before the connection closed -- it is "
        f"the reason, and re-running unchanged will reproduce it:\n{text}"
    )


def _portal_went_down(exc: BaseException) -> bool:
    """Whether ``exc`` is the portal's process going away rather than an answer from it.

    Only then is its stderr the explanation. The dispatch below marks the exceptions it raises
    itself: ``sog_portal_down`` on a handshake that never came back, ``sog_portal_answered`` on an
    ``isError`` result or a spent budget -- the portal was up and said so. An answer wins over a
    closed stream seen in the same teardown.
    """
    try:
        import anyio
        from mcp.shared.exceptions import McpError
        from mcp.types import CONNECTION_CLOSED
    except Exception:  # pragma: no cover - the mcp client is a hard dependency of dispatch
        return False
    closed = (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream, BrokenPipeError, EOFError)
    down = False
    for leaf in _exception_leaves(exc):
        if getattr(leaf, "sog_portal_answered", False):
            return False
        if getattr(leaf, "sog_portal_down", False) or isinstance(leaf, closed):
            down = True
        elif isinstance(leaf, McpError) and getattr(getattr(leaf, "error", None), "code", None) == CONNECTION_CLOSED:
            down = True
    return down


def _marked(exc: BaseException, attr: str) -> BaseException:
    setattr(exc, attr, True)
    return exc


def _handshake_timeout_message(tool_name: str, seconds: float) -> str:
    """The ``session.initialize`` bound expired: the portal never came up.

    Kept distinct from the tool-budget message below because the remedies are opposite: a dead
    handshake is an environment problem (missing interpreter, broken server command) that retrying
    the same call cannot fix, while a tool-budget expiry is a healthy tool that needed more time.
    Both used to surface as the bare word ``TimeoutError``, indistinguishable from each other.
    Names the knob (house style set by ``_tool_budget_timeout_message``) so the model can relay a
    real remedy for the rare healthy-but-slow-to-start server instead of inventing one.
    """
    return (
        f"the MCP server for '{tool_name}' did not answer its handshake within {int(seconds)}s. The server "
        "process failed to start or is wedged (a missing interpreter or a broken server command, "
        "not bad tool arguments) - re-running the same call will fail the same way until the "
        "server's environment is repaired. If the server is healthy but genuinely slow to start, "
        "the handshake budget is raised via the SOG_MCP_HANDSHAKE_TIMEOUT environment variable."
    )


def _tool_budget_timeout_message(tool_name: str, seconds: float) -> str:
    """The per-call tool budget expired while the worker was (as far as we know) still working.

    A live full-scale run hit this: a 527MB reference export was killed at 52% after 600s, and the
    model-facing observation was the bare word ``TimeoutError`` -- no budget named, no knob named,
    no warning that the killed worker's half-written files were still on disk. The guidance below
    is ordered by what the ReAct loop can actually do: never reuse the partial files, never shrink
    the dataset to fit the budget (full-dataset analysis is the product), and name the budget knobs
    verbatim so the model can relay them to the operator instead of inventing a remedy.
    """
    return (
        f"'{tool_name}' was still running when the {int(seconds)}s per-call tool budget expired, "
        "and its worker subprocess has been terminated. Any output files it was mid-writing are "
        "incomplete - re-run the tool rather than reusing them. Do NOT subsample or shrink the "
        "input to fit the budget. If this full-dataset step legitimately needs longer, the budget "
        "is raised via STCoscientist(timeout_seconds=...), the --timeout CLI flag, or the "
        "SOG_TIMEOUT_SECONDS environment variable."
    )


_DEFAULT_CALL_BUDGET_SECONDS = 600.0

#: Set to ``"1"`` in the portal's agent worker and nowhere else (``sog_portal/boundary.agent_env``).
_BUDGET_AWARE_ENV = "SOG_TOOL_BUDGET_AWARE"

# Effective input throughput of the one live expiry this warning exists to pre-empt: a 527MB
# reference export had reached 52% of its work when the 600s budget killed it -- roughly 0.46MB of
# input consumed per second. One point, one tool, one box, so it is used as a *scale* and never as
# a forecast: it sets the size above which the budget is worth mentioning before launching, and the
# message says so in those words. Calibrating on an invented fleet-wide MB/s constant instead would
# make the warning unfalsifiable, which is worse than not warning.
_OBSERVED_INPUT_BYTES_PER_SECOND = 527 * 1024 * 1024 * 0.52 / 600.0

# The scan below runs on the dispatch path of every tool call, so it is bounded rather than
# exhaustive: at most this many stat/scandir calls across all of one call's kwargs. A 10x/Visium
# directory is a few hundred files and a reference export is one, so the cap is generous for both
# while keeping a pathological kwargs dict (a deep tree, a thousand strings) from costing more than
# the subprocess launch it precedes.
_MAX_INPUT_STATS = 512


def _call_budget_seconds(agent: Any) -> float:
    """The per-call tool budget this dispatch will actually be bounded at.

    Factored out of the dispatch site so the pre-launch warning and the ``asyncio.wait_for`` that
    enforces it read one number. A second copy of this resolution order is exactly how a warning
    ends up naming a budget the call is not run under -- the same class of lie
    ``_tool_budget_timeout_message`` was written to end, one step earlier.

    The order is the agent's own ``timeout_seconds`` (a ``--timeout`` or web-settings override)
    first, the global config second, and the historical 600s for anything missing, zero, negative
    or unparseable.
    """
    try:
        from spatialomicsgym.config import default_config

        fallback = getattr(default_config, "timeout_seconds", _DEFAULT_CALL_BUDGET_SECONDS)
    except Exception:
        fallback = _DEFAULT_CALL_BUDGET_SECONDS
    try:
        seconds = float(getattr(agent, "timeout_seconds", None) or fallback)
    except (TypeError, ValueError):
        return _DEFAULT_CALL_BUDGET_SECONDS
    return seconds if seconds > 0 else _DEFAULT_CALL_BUDGET_SECONDS


def _budget_is_raisable(agent: Any) -> bool:
    """Can anyone who reads this run's observations raise its budget before the call runs again?

    Only on an interactive front door, and ``conversation_memory`` is the one signal of one this
    code has: the constructor documents that only ``chat_cli`` and ``sog-web`` turn it on, and that a
    benchmark instance must run with it off. Everywhere else -- a script's ``go()``, sog-setup's
    probe, the SpatialBench trial that probe runs -- the budget was fixed by whoever started the
    run, and the answer is final when the model writes it, so naming the knobs is advice nobody
    there can take. A notebook caller who could is told them by the kill message if it comes to
    that. Under process isolation ``agent`` is the worker's proxy, which the server mirrors this
    onto (``execution._inject_into_process_repl``).
    """
    try:
        return bool(getattr(agent, "conversation_memory", False))
    except Exception:
        return False


def _input_bytes(kwargs: dict[str, Any]) -> int:
    """How many bytes of on-disk input one call's kwargs name. Bounded, and never raises.

    Only strings (and the strings inside a one-level list/tuple) are considered, and only as
    *candidate* paths: whether a value is an input is decided by the filesystem, not by the keyword
    it arrived under, because the 88 portals spell that keyword dozens of ways (``adata_path``,
    ``sc_ref``, ``input_dir``, ``h5ad_file``...). A value that is neither a file nor a directory
    contributes nothing -- which is also exactly what a cluster count or a method name contributes.

    Directories are walked, because a 10x/Visium input *is* a directory and its size is the whole
    signal for those tools. The walk shares the one ``_MAX_INPUT_STATS`` budget with the top-level
    scan so a wide tree cannot make the cheap case expensive.

    Every file is counted once, keyed on its resolved path. Both ways a byte gets named twice are
    real and both over-count in the direction that makes the warning *louder*: two kwargs naming one
    reference export, and a file that also sits inside a directory kwarg being walked. A heuristic
    whose failure mode is scaring the model off a tool that would have finished is worse than one
    that stays quiet, so the duplicate is dropped.

    Returns what it managed to add up. A partial sum is the right degrade for a caller that only
    warns: under-counting makes the warning quieter, never wrong.
    """
    budget = _MAX_INPUT_STATS
    total = 0
    seen: set[str] = set()

    def _add(path: str, size: int) -> None:
        nonlocal total
        try:
            key = os.path.realpath(path)
        except (OSError, ValueError):
            key = path
        if key in seen:
            return
        seen.add(key)
        total += size

    def _candidates(value: Any):
        if isinstance(value, str):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    yield item

    for key, value in kwargs.items():
        # Where a tool WRITES is not its input. A reused output folder full of earlier h5ad/model
        # files made every later call warn that it was handed gigabytes (u14-mcp-wiring-20).
        lowered = str(key).lower()
        if lowered.startswith(("output", "out_")) or lowered in ("out", "outdir"):
            continue
        for name in _candidates(value):
            if budget <= 0:
                return total
            # Rejected before the syscall, not by catching its error: an embedded NUL raises
            # ValueError rather than OSError, and a multi-line value is prose (a prompt, a gene
            # list) that no filesystem call should be spent on.
            if not name or len(name) > 4096 or "\n" in name or "\0" in name:
                continue
            budget -= 1
            try:
                st = os.stat(name)
            except (OSError, ValueError):
                continue
            if stat.S_ISREG(st.st_mode):
                _add(name, st.st_size)
            elif stat.S_ISDIR(st.st_mode):
                for parent, _subdirs, files in os.walk(name):
                    for fname in files:
                        if budget <= 0:
                            return total
                        budget -= 1
                        member = os.path.join(parent, fname)
                        try:
                            _add(member, os.stat(member).st_size)
                        except OSError:
                            continue
    return total


def _budget_warning_message(tool_name: str, input_bytes: int, seconds: float, raisable: bool = True) -> str:
    """Said *before* the clock starts, when the inputs are large against the budget bounding them.

    The mirror image of ``_tool_budget_timeout_message`` and deliberately the same shape: name the
    budget, name the three knobs verbatim, forbid subsampling. Two things differ. The tense -- this
    one still has a remedy that is free, because raising the budget before dispatch costs nothing
    and raising it after a kill costs the whole budget again. And the hedging: this is a rule of
    thumb from a single measurement, it says so, and it tells the model to run the tool anyway.
    A warning the model reads as a prediction would make it abandon tools that would have finished,
    which is a worse failure than the timeout it is trying to pre-empt.

    The remedy is only offered where someone can take it (``raisable``, from
    ``_budget_is_raisable``). Elsewhere the knobs are left out and the notice says instead that the
    budget was fixed before the run started -- the one thing about it such a run can know, and the
    reason no remedy follows. With ``raisable`` the text is byte-identical to what it was before
    the distinction existed; the default is that text, the longest this notice gets.
    """
    gb = input_bytes / (1024.0 * 1024.0 * 1024.0)
    facts = (
        f"[budget notice] '{tool_name}' was handed about {gb:.1f}GB of input and this call is "
        f"bounded at {int(seconds)}s. A comparable full-dataset run consumed input at roughly "
        "0.5MB/s, so this one may not fit -- that is a rule of thumb from one measurement and not "
        "a prediction, so go ahead and run it. Do NOT subsample or shrink the input to make it "
        "fit. If it is killed the half-written outputs are unusable"
    )
    if not raisable:
        return facts + ". This run's budget was fixed before it started."
    return facts + (
        " and the budget has to be spent again, so if a longer run is wanted it is cheapest to "
        "raise the budget now: STCoscientist(timeout_seconds=...), the --timeout CLI flag, or the "
        "SOG_TIMEOUT_SECONDS environment variable."
    )


#: How a tool says, in the ``[DATA REQUIREMENTS]`` block every shipped description ends with, that
#: it produces nothing. The inspector portals (``viz_inspector``, ``spatial3d_inspector``) make it
#: their contract: no output directory, no file written, no directory created.
_WRITES_NOTHING_RE = re.compile(r"\[DATA REQUIREMENTS\].*?\bWrites nothing\.", re.DOTALL)


def _declares_it_writes_nothing(doc: str) -> bool:
    """Does this tool's description declare it a pure read? The config's own classification.

    The notice's one rate is an analysis consuming its input: a reference export being fitted at
    about 0.5MB/s. A read is not a comparable run. The viz inspector opens its file backed and
    reads ``sample_spots`` rows of it; ``inspect_3d_coordinates`` loads the file once and reports
    key shapes. Both go at the speed of the disk. On E-03 the notice fired once in 48 trials, on
    ``inspect_dataset(sample_spots=1000)`` over a 1.0GB file, telling it that it "may not fit".

    Keyed on the declaration the model reads, not on a list of names kept here: a new inspector is
    exempt by saying what it is, and a tool that stops saying so is warned about again. Only the
    ``[DATA REQUIREMENTS]`` block counts, because that is where the contract is written; prose
    before it may say anything.
    """
    return bool(_WRITES_NOTHING_RE.search(doc or ""))


def _budget_warning(
    tool_name: str, kwargs: dict[str, Any], seconds: float, *, doc: str = "", raisable: bool = False
) -> str | None:
    """The notice to print before dispatching, or ``None`` to stay quiet. Never raises.

    Silent under ``benchmarking_enabled`` -- the same "benchmarking wins" gate
    ``next_step.post_analysis_active()`` applies, for the same reason and with more force here:
    this text lands in the model-facing observation, this module is on the eval path, and every
    step of a benchmark is by construction a full-dataset step, so an ungated notice would fire on
    most of them. A run that sets the flag reads byte-identically whatever an operator has
    configured on the box.

    That covers the runs that set it and no others. The SpatialBench probe, as read on 2026-09-25,
    deliberately does not (its own comment: the flag "changes what the agent is"), and nothing else
    it sets reads here as "scored", so its trials DO print this notice -- once in E-03's 48.
    Silencing them would take a signal the probe sets, and the probe is not in this repo. So the
    text is made true where it prints instead: the budget knobs are named only when ``raisable``
    (an interactive front door, ``_budget_is_raisable``); a run that cannot raise its budget is
    told the facts and that the budget was fixed before it started. ``raisable`` defaults to the
    run that cannot.

    Silent, too, for a tool whose description (``doc``) declares that it writes nothing: a read is
    not sized against an analysis's throughput -- see ``_declares_it_writes_nothing``.

    Quiet is also the answer to every failure: a config that will not import, a filesystem that
    will not answer. Nothing here is worth failing a dispatch over.
    """
    try:
        from spatialomicsgym.config import default_config

        if getattr(default_config, "benchmarking_enabled", False):
            return None
    except Exception:
        return None
    try:
        if _declares_it_writes_nothing(doc):
            return None
        total = _input_bytes(kwargs)
        if total <= seconds * _OBSERVED_INPUT_BYTES_PER_SECOND:
            return None
        return _budget_warning_message(tool_name, total, seconds, raisable=raisable)
    except Exception:
        return None


def _env_fallback_notice(agent: Any, tool_name: str, result: Any) -> str | None:
    """The env-fallback notice for this call, or ``None``. Never raises; silent under benchmarking.

    Thin on purpose: the judgement (what counts as an environment failure, once per tool per turn,
    the wording) lives in ``agent/env_fallback.py``; this wrapper only owns the channel.
    """
    try:
        from spatialomicsgym.agent.env_fallback import notice_for

        return notice_for(agent, tool_name, result)
    except Exception:
        return None


def _local_tools_dir() -> Path:
    """The ``tools/`` directory serving this process. Used to rebase a worker-script path
    that a config pinned to a *different* clone root (the shipped canonical, or a config
    copied from the machine that generated it) onto this installation.

    ``platform_root.tools_dir()`` answers ``<repo>/agent/tools`` on a checkout -- byte-identical to
    the ``__file__``-derived sibling this used to return -- and on a pip-only install it is
    the seeded SOG_HOME's ``tools/`` or the wheel's read-only ``_platform/tools`` copy, both
    of which mirror the checkout layout (so the caller's ``tools_dir.parent / "tools_user"``
    sibling walk stays coherent). The old derivation remains as the unconditional last resort
    so behavior with every platform rung dark is unchanged.
    """
    resolved = platform_root.tools_dir()
    if resolved is not None:
        return resolved
    return Path(__file__).resolve().parent.parent.parent / "tools"


def _interp_runnable(cmd: str) -> bool:
    """Whether a server's interpreter (``command[0]``) can actually be launched here.

    An **absolute** path is the **base** (agent-core) env python ``sog-setup`` wrote when it
    wired this config — ``wiring.rewire_server_meta`` sets ``command = [base_python, script]``
    for *every* server, so it is one value per config and never the per-tool worker
    interpreter (that one lives in ``env[{PREFIX}_PYTHON]``, varies tool by tool, and is
    repaired separately by ``base_mcp._resolve_worker_python``). It must exist on *this* box;
    a **bare** name (a portable ``python``/``Rscript``) must resolve on ``PATH`` instead. A
    config copied from another device, or one whose base env has since been deleted, points
    ``command[0]`` at an interpreter that isn't installed — registering its tools anyway
    yields calls that only blow up at call time, so the caller skips the server instead
    (mirrors ``chat_cli._interpreter_present`` / the ``conncheck`` L4 check).

    Dropping such a server is the *designed* outcome, not a gap to be "repaired" by falling
    back to ``sys.executable``: a base env this box does not have is exactly the state
    ``sog-setup`` exists to rewire, and the skip message says so. Pinned by
    ``test_round11_hardening.py::test_resolver_disables_server_when_base_env_absent_even_if_tool_env_healthy``.
    """
    if not cmd or not isinstance(cmd, str):
        return False
    if os.path.isabs(cmd):
        return os.path.exists(cmd)
    return shutil.which(cmd) is not None


def _rebase_script_args(args: list, tools_dir: Path) -> list:
    """Rebase a **missing absolute** worker-script arg onto this clone's worker directories.

    The shipped canonical (and any config generated on another box) pins ``command[1]`` to an
    absolute ``<other-clone>/tools/<name>_mcp_server.py``. The *basename* is stable across
    clones, so when the pinned path is absent but a local copy of that basename exists, rewrite
    it to the local copy — making a fresh clone at ANY filesystem path wire correctly without
    re-running setup. Everything else is left byte-identical: a present absolute path, a
    bare/relative arg, a flag, or an absolute path with no local counterpart — so a
    correctly-pinned config (the common case, incl. this same box) is untouched.

    Two directories are searched, because there are two kinds of worker script and both get an
    absolute path pinned into the config: the built-in portals in ``tools_dir``, and the ones the
    agent wrote itself in the sibling ``tools_user/``. Searching only the former meant a moved
    checkout dropped every user-created tool — and the skip message's advice to re-run
    ``sog-setup`` could not restore them, since setup's capture only scans ``tools/`` either.
    Built-ins win a basename collision so a user tool cannot shadow a shipped portal.
    """
    if not isinstance(args, list):
        return args
    search_dirs = (tools_dir, tools_dir.parent / "tools_user")
    out = []
    for a in args:
        if isinstance(a, str) and a.endswith((".py", ".R")) and os.path.isabs(a) and not os.path.exists(a):
            base = os.path.basename(a)
            local = next((d / base for d in search_dirs if (d / base).exists()), None)
            if local is not None:
                out.append(str(local))
                continue
        out.append(a)
    return out


def _missing_script_arg(args: list) -> str | None:
    """First **absolute** ``.py``/``.R`` worker-script arg that does NOT exist here, else None.

    ``_interp_runnable`` gates the interpreter (``command[0]``), but a foreign/copied config can also
    pin ``command[1]`` to a worker script whose basename isn't in this clone's ``tools/`` — so
    ``_rebase_script_args`` couldn't rebase it and left the dead absolute path. Registering that
    server yields tools that fail EVERY call with ``python: can't open file``; the caller skips it up
    front instead, mirroring the interpreter gate. A no-op for the canonical config (all scripts
    present) and for bare/relative args.
    """
    if not isinstance(args, list):
        return None
    for a in args:
        if isinstance(a, str) and a.endswith((".py", ".R")) and os.path.isabs(a) and not os.path.exists(a):
            return a
    return None


def _resolve_portal_interp(cmd: str) -> str:
    """Pin a **bare** portal interpreter (``python``/``python3``) to this agent's own
    interpreter (``sys.executable``).

    Every MCP ``command[0]`` launches a thin ``*_mcp_server.py`` FastMCP portal, which needs
    only the agent-core env (``fastmcp`` + ``base_mcp`` + ``pyyaml``) — i.e. the very env the
    agent is already running in. A bare ``python`` otherwise resolves against the *inherited*
    ``PATH`` (``mcp.client.stdio.get_default_environment`` always forwards ``PATH``), which is
    whatever python is first on ``PATH`` at launch: base/system python if the agent was
    started via its absolute interpreter without ``conda activate``, so the portal can't
    import ``fastmcp`` and discovery silently yields nothing. Pinning bare python to
    ``sys.executable`` makes portal launch activation-independent and portable across devices
    (no hard-coded env path in the shipped config). An **absolute** interpreter — the base
    env python ``sog-setup`` pinned for this whole config, never a per-tool worker env — is
    honoured as-is (then existence-checked by ``_interp_runnable``); a non-python bare name
    (none are emitted today: all 88 portals run a ``.py``) is likewise left for
    ``_interp_runnable``.
    """
    if isinstance(cmd, str) and not os.path.isabs(cmd) and os.path.basename(cmd) in ("python", "python3"):
        return sys.executable
    return cmd


def _extract_tool_result(result) -> str:
    """Extract a tool's textual output from an MCP ``CallToolResult``.

    The happy path — a non-empty ``content`` list whose first block is a ``TextContent`` —
    returns that block's ``.text`` unchanged. Degrades without crashing when the tool
    returned no content (``result.content == []`` → IndexError) or a non-text block such
    as an image / embedded resource (no ``.text`` → AttributeError).
    """
    content = getattr(result, "content", None)
    if not content:
        structured = getattr(result, "structuredContent", None)
        return str(structured) if structured is not None else ""
    block = content[0]
    text = getattr(block, "text", None)
    if text is not None:
        return text
    data = getattr(block, "data", None)
    return str(data) if data is not None else str(block)


_SCHEMA_TYPE_MAP = {
    # JSON-Schema spellings, which is what an auto-discovered tool's `inputSchema` carries.
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    # The manual-tool spellings used by this repo's own mcp_config entries.
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "dict": dict,
    "list": list,
    "List[str]": list,
}


def attach_kwarg_signature(fn, required_params, optional_params) -> None:
    """Declare on ``fn`` the keyword names its MCP tool actually accepts.

    ``make_mcp_wrapper`` returns a ``**kwargs``-only closure and sets just ``__name__``/``__doc__``,
    so ``inspect.signature`` reports ``(**kwargs)``. That reads as "any keyword is fine" when in
    truth the kwargs go straight to ``session.call_tool`` and a wrong or missing name comes back
    ``isError=True``, re-raised as a RuntimeError. The system prompt directs the model to "use the
    wrapper's actual signature", so ``(**kwargs)`` is a dead end: observed live, an agent inspected a
    tool, learned nothing, and bailed with a plan ("the next step is to call it with the documented
    inputs") instead of ever running it. The structured parameter lists needed to answer the question
    are already built at the registration call site; this puts them on the function object.

    Parameters are declared KEYWORD_ONLY because that is what the wrapper accepts. Required ones get
    no default; optional ones default to ``None`` and are annotated ``T | None``, matching
    ``generate_mcp_wrapper_from_spatialomicsgym_schema``, which has always done this.

    Changes reporting only, never behaviour: ``__signature__`` is advisory -- CPython does not consult
    it when binding a call -- so the wrapper still accepts ``**kwargs`` verbatim, and nothing else in
    the package reads ``inspect.signature`` on these wrappers.

    Degrades rather than raising. An unusable entry (not a mapping, no usable name, a duplicate, a
    non-identifier, a Python keyword) is skipped, an unrecognised type falls back to ``str``, and a
    tool that declares no parameters keeps the honest ``(**kwargs)`` -- an empty ``()`` would be a
    different lie, claiming the tool takes no arguments. A malformed schema in one tool entry must
    never abort registration for the rest.
    """
    params: list[inspect.Parameter] = []
    seen: set[str] = set()
    for group, is_required in ((required_params, True), (optional_params, False)):
        for spec in group or []:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
                continue
            if name in seen:
                continue
            seen.add(name)
            declared = spec.get("type")
            annotation = _SCHEMA_TYPE_MAP.get(declared, str) if isinstance(declared, str) else str
            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=inspect.Parameter.empty if is_required else None,
                    annotation=annotation if is_required else annotation | None,
                )
            )
    if not params:
        return
    try:
        # `str`: the wrapper returns the tool's JSON text (_extract_tool_result), not a dict, so
        # code written against `-> dict` failed on r["status"] (u14-mcp-wiring-15).
        fn.__signature__ = inspect.Signature(params, return_annotation=str)
    except (TypeError, ValueError) as e:
        # Leave the honest (**kwargs) in place rather than half-declaring a broken signature.
        print(f"Warning: could not declare a signature for '{getattr(fn, '__name__', '?')}': {e}")


def _drop_credential_defaults(parameters: dict) -> tuple[dict, list[str]]:
    """Strip any default that holds a credential. Returns (parameters, names dropped).

    A tool's ``parameters`` block is data read off disk, and it becomes the model-facing schema
    verbatim: ``utils.formatting.textify_api_dict`` renders every optional parameter as
    ``[Default: <value>]`` with no branch on what the value is. So a credential parked in a default
    is in the system prompt on every turn, and goes to the LLM provider.

    One was. The generated config on the author's box carried a live UCDeconvolve API token as
    ``ucdeconvolve_base.token``'s default; the tracked config it derives from declares ``None`` for
    the same parameter, because it was scrubbed there and the generated copy -- which the resolver
    treats as a cache, rewriting only ``command``/``env``/``enabled`` and passing ``tools:`` through
    untouched -- was never rebuilt. Nothing between the YAML loader and the prompt had ever looked at
    what a default *is*.

    The parameter itself survives; only its value goes. The model still learns the knob exists, and
    every wrapper that takes one reads the real credential from the environment. Measured over the
    tracked config this touches nothing: one parameter matches, and it declares ``None`` there.

    The caller's dict is not mutated -- ``add_mcp`` is handed the loaded config, and a later reader
    must still see what the file says.
    """
    dropped: list[str] = []
    cleaned: dict = {}
    for name, spec in parameters.items():
        if isinstance(spec, dict) and looks_like_credential(name, spec.get("default")):
            spec = {k: v for k, v in spec.items() if k != "default"}
            dropped.append(name)
        cleaned[name] = spec
    return cleaned, dropped


# Not LD_LIBRARY_PATH, R_HOME or R_LIBS*: each tool runs in its own conda env, and the agent env's
# library or R paths pointed into it would load another env's native libraries or R packages.
_FORWARDED_TO_TOOLS = (
    "CUDA_VISIBLE_DEVICES",
    "TMPDIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    # The rest of the portal worker's thread budget (``sog_portal/boundary.THREAD_VARS``). Inert unless set,
    # and nothing but the portal's worker sets them -- a scored run's environment carries none.
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    # The operator's locale and its documented remedy, for base_mcp._worker_env: with none of these
    # forwarded it forced LC_ALL=C.UTF-8 on every MCP worker, which a host without that locale (macOS,
    # old glibc) cannot honour, and SOG_WORKER_LOCALE could not arrive to name another (hunt
    # 2026-09-30, u29a-mcp-transport-10).
    "SOG_WORKER_LOCALE",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
)


def resolve_server_env(env_spec: Any) -> dict[str, str]:
    """Expand a config ``env:`` block into the concrete mapping a worker subprocess receives.

    ``${NAME}`` is read from this process's environment; anything else is passed through as a
    string, because ``StdioServerParameters`` rejects non-str values.

    **Why this is a function and not four inline lines.** The MCP SDK spawns a worker with
    ``env={**get_default_environment(), **server.env}``, and ``get_default_environment()`` inherits
    an allowlist of exactly ``HOME, LOGNAME, PATH, SHELL, TERM, USER``. Nothing else crosses the
    process boundary. So a variable the agent process sets -- ``SOG_WORK_DIR``,
    ``SOG_SPATIAL_LIBRARY_REGISTRY`` -- is **inert for every worker** unless some server's ``env:``
    block names it. That is not a hypothetical: both were documented deployment overrides that
    silently did nothing for the MCP path, which is the path the agent actually calls.

    The expansion happens **per call** rather than once at wiring time. Wiring means handshaking
    every configured server; a caller that wants a per-run scratch directory cannot pay that on
    every turn. Reading the variable when the worker is spawned makes the forward live, so
    ``env: {SOG_WORK_DIR: "${SOG_WORK_DIR}"}`` means "whatever the agent process has set right
    now", which is what a reader of that line already assumes it means.

    An unset ``${NAME}`` becomes ``""``, deliberately: every consumer in this repo strips and
    treats empty as unset (``base_mcp.default_output_dir``, ``spatial_library_worker``), and
    dropping the key instead would change what an existing config means.
    """
    # The operator's machine settings cross too: GPU pinning, the temp dir, proxies and thread caps. The SDK forwards only HOME/LOGNAME/PATH/SHELL/TERM/USER, so a GPU the operator
    # hid was used and a proxy was bypassed by every tool (u14-mcp-wiring-16). A server's own env
    # block still wins. No secrets: none of these names holds a key.
    out: dict[str, str] = {name: os.environ[name] for name in _FORWARDED_TO_TOOLS if os.environ.get(name) is not None}
    if not isinstance(env_spec, dict) or not env_spec:
        return out
    for key, value in env_spec.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            out[str(key)] = os.getenv(value[2:-1], "")
        else:
            out[str(key)] = str(value)
    return out


# ── Fix for "fileno" error under redirect_stdout/redirect_stderr ──
# When benchmark_runner (or any caller) uses contextlib.redirect_stdout
# to capture output, sys.stdout/stderr become StringIO objects which lack
# fileno(). MCP's stdio_client calls asyncio.create_subprocess_exec which
# needs real OS file descriptors for stdin/stdout/stderr pipes.
#
# Solution: save real OS-level file descriptors at module load time and
# temporarily restore them when spawning MCP subprocesses.
def _restore_real_stdio():
    """Temporarily restore real C-level stdio for subprocess creation.

    sys.__stdin__/__stdout__/__stderr__ are always the original streams
    set at interpreter startup — they survive contextlib.redirect_*
    and any number of nesting levels. This is what asyncio's subprocess
    needs to get valid file descriptors.

    Returns the saved (stdin, stdout, stderr) tuple to put back later.
    """
    saved = (sys.stdin, sys.stdout, sys.stderr)

    # __stdin__/__stdout__/__stderr__ can be None under a daemonized or fully-captured harness
    # (pythonw, nohup, some pytest capture modes) -- or CLOSED (pytest's capfd teardown closes
    # them; a double-forking daemon closes inherited stdio). Swapping in a closed stream breaks
    # the stdio_client subprocess launch (errlog=sys.stderr -> stderr.fileno() raises
    # "I/O operation on closed file") and every print() -- and it breaks EVERY LATER MCP call
    # in the process, not just this one. Fall back to the current stream in both cases.
    def _usable(original, current):
        try:
            if original is None or original.closed:
                return current
        except Exception:  # a detached TextIOWrapper raises even on .closed
            return current
        return original

    sys.stdin = _usable(sys.__stdin__, sys.stdin)
    sys.stdout = _usable(sys.__stdout__, sys.stdout)
    sys.stderr = _usable(sys.__stderr__, sys.stderr)
    return saved


def _put_back_stdio(saved, installed=None):
    """Put ``saved`` back -- but only over a stream that is still the one this call ``installed``.

    An unconditional restore reintroduced the hazard PythonREPL.run already fixed: a tool call
    orphaned by a budget timeout restored ITS saved capture buffer while the next cell was running,
    so that cell's prints went to the real stdout and its observation came back empty; and two
    overlapping calls ended with the cell's sys.stdout left on the real stream (hunt 2026-09-30,
    u14-mcp-wiring-5). A stream someone else has replaced since is theirs to restore.

    And never stdout on a thread ``run_with_timeout`` has given up on: it handed stdout back to its
    caller at the deadline, and what this saved is the timed-out cell's capture. Putting it back when
    the tool call finally returns sent everything printed after that into a buffer nobody reads -- or,
    if the next cell had started, replaced THAT cell's capture mid-run (program9; merged 2026-10-02).
    """
    current = (sys.stdin, sys.stdout, sys.stderr)
    if installed is None:
        installed = current
    abandoned = abandoned_by_its_caller()
    restored = []
    for index, (old, now, ours) in enumerate(zip(saved, current, installed, strict=True)):
        mine = now is ours and not (index == 1 and abandoned)
        restored.append(old if mine else now)
    sys.stdin, sys.stdout, sys.stderr = restored


def make_mcp_wrapper(agent: Any, cmd: str, args: list[str], tool_name: str, doc: str, env_spec: dict | None = None):
    """Create a synchronous wrapper for an async MCP tool call.

    ``env_spec`` is the config's DECLARED ``env:`` mapping, not a resolved one: the expansion
    happens inside the wrapper, on every call. See :func:`resolve_server_env`.

    Module-level, not a closure inside :func:`add_mcp`, so the SAME factory can rebuild a wrapper
    in another process from the spec ``add_mcp`` records on the function (``_sog_spec``): under
    ``repl_isolation = "process"`` the REPL worker (``tool/repl_host.py``) builds its own wrappers
    and spawns the tool portals itself, as the unprivileged user. ``agent`` is whatever object
    carries ``timeout_seconds`` and ``_env_failures`` -- the real agent here, a proxy there.
    """
    import asyncio

    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    def sync_tool_wrapper(**kwargs):
        """Synchronous wrapper for MCP tool execution."""
        # Duplicate-dispatch guard (armed only while an <execute> script runs -- see
        # tool_call_memo.py). Live case s4a: one model message carried the same pipeline block
        # twice (a hallucinated-transcript turn), the action layer deliberately concatenated
        # them, and cell2location ran 4h15m then 3h39m again for a byte-identical overwrite.
        # Same tool + byte-identical kwargs after a SUCCESS in the same script -> return the
        # earlier result with a loud printed notice instead of re-dispatching. Failures are
        # never remembered, and a later block or turn always dispatches for real.
        memo_key = tool_call_memo_key(tool_name, kwargs)
        memo_hit, remembered_result = lookup_successful_call(memo_key)
        if memo_hit:
            print(duplicate_call_notice(tool_name))
            return remembered_result
        # Say it BEFORE the clock starts. ``_tool_budget_timeout_message`` is the post-hoc half
        # of this and only reaches the model once the whole budget has been spent on a run that
        # got killed mid-write -- at which point the only remedy left is to spend it again.
        # Printed into the observation, the channel ``duplicate_call_notice`` above already
        # establishes. Silent under ``benchmarking_enabled``, which a SpatialBench trial does not
        # set; there it prints without the knobs -- see ``_budget_warning``.
        launch_notice = _budget_warning(
            tool_name, kwargs, _call_budget_seconds(agent), doc=doc, raisable=_budget_is_raisable(agent)
        )
        if launch_notice:
            print(launch_notice)
        # Defined before the ``try``: the ``except`` below reads it, and a failure in
        # ``resolve_server_env`` used to reach that handler with the name unbound.
        portal_stderr = ""
        try:
            # Expanded HERE, per call -- not closed over from wiring time. A caller that sets
            # SOG_WORK_DIR for the duration of one run reaches the worker through this, and the
            # agent is wired once rather than re-handshaking 88 servers per turn.
            server_env = resolve_server_env(env_spec)
            if os.environ.get(_BUDGET_AWARE_ENV) == "1":
                # The portal's worker only (``sog_portal/boundary.agent_env`` sets the flag): the tool learns
                # this call's budget, so ``base_mcp`` stops a long worker a minute before it and the
                # cell -- and the session's variables -- survive (2026-10-04, R3).
                server_env = {**(server_env or {}), "SOG_TOOL_BUDGET_S": str(int(_call_budget_seconds(agent)))}
            server_params = StdioServerParameters(command=cmd, args=args, env=server_env)

            # Filled by ``async_tool_call``'s teardown, read by the ``except`` below. A
            # portal that dies on import explains itself on stderr and nowhere else, and that
            # stream does not reach the model -- see ``_portal_stderr_note``.

            async def async_tool_call():
                nonlocal portal_stderr
                # Restore real stdio FDs so asyncio subprocess can get
                # file descriptors for pipes. This fixes the "fileno"
                # error when agent.go() is called inside redirect_stdout.
                # Captured rather than passed through, so the traceback can be put in front of
                # the model. It is still echoed to the real stderr below, so operator logs and
                # anything tailing them are unchanged. Created BEFORE the stdio swap: a failure
                # here between the swap and the try left the cell's stdout on the real stream,
                # so nothing after it reached the observation (u14-mcp-wiring-21).
                errlog = tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace")
                saved = _restore_real_stdio()
                installed = (sys.stdin, sys.stdout, sys.stderr)
                try:
                    # errlog explicitly -- same closed-import-time-default hazard as in
                    # _discover_async above.
                    async with stdio_client(server_params, errlog=errlog) as (reader, writer):
                        async with ClientSession(reader, writer) as session:
                            # Bound the handshake tightly (SOG_MCP_HANDSHAKE_TIMEOUT, default 30s) --
                            # portals are thin (they only import base_mcp + dispatch to the worker
                            # env). The tool CALL is bounded below.
                            _hs_timeout = _handshake_timeout_seconds()
                            try:
                                await asyncio.wait_for(session.initialize(), timeout=_hs_timeout)
                            except TimeoutError as exc:
                                raise _marked(
                                    TimeoutError(_handshake_timeout_message(tool_name, _hs_timeout)), "sog_portal_down"
                                ) from exc
                            # Bound the tool call at the AGENT's OWN timeout_seconds (honor a
                            # --timeout / web-settings override -- not the global default_config,
                            # which the constructor arg never mutates). A wedged tool then times out
                            # here and the async context tears its worker SUBPROCESS down (the outer
                            # run_with_timeout thread can't kill native code). Set it EQUAL to the
                            # outer budget -- NOT below it -- so a legitimately-long full-dataset tool
                            # the outer timeout would allow is never preempted early; teardown is at
                            # most ~init-time late, an acceptable trade for not cutting real tools.
                            _call_timeout = _call_budget_seconds(agent)
                            try:
                                result = await asyncio.wait_for(
                                    session.call_tool(tool_name, kwargs), timeout=_call_timeout
                                )
                            except TimeoutError as exc:
                                raise _marked(
                                    TimeoutError(_tool_budget_timeout_message(tool_name, _call_timeout)),
                                    "sog_portal_answered",
                                ) from exc
                            # Always prefer .text which contains the actual tool output.
                            # content.json() returns the Pydantic model envelope
                            # ({"type":"text","text":"..."}) not the tool result.
                            extracted = _extract_tool_result(result)
                            # An ``isError=True`` CallToolResult is a tool-level FAILURE (a
                            # validation error from bad/missing/extra kwargs, or a worker error).
                            # Surface it as an exception -- the same channel as a transport error
                            # below -- so the ReAct loop gets a clear error observation and
                            # self-corrects. Otherwise the error TEXT is handed back as a truthy
                            # "successful" return that generated code (`out = run_tool(...); if out:`)
                            # treats as a real result while the tool never actually ran.
                            if getattr(result, "isError", False):
                                raise _marked(
                                    RuntimeError(extracted or "tool reported an error (isError)"), "sog_portal_answered"
                                )
                            return extracted
                finally:
                    # Read before restoring stdio: this must not depend on where sys.stderr
                    # currently points. Wrapped whole because a capture that fails to read
                    # back must not turn a tool result into an exception.
                    try:
                        errlog.seek(0)
                        portal_stderr = errlog.read()
                        if portal_stderr:
                            print(portal_stderr, end="", file=sys.__stderr__, flush=True)
                    except Exception:
                        portal_stderr = ""
                    finally:
                        try:
                            errlog.close()
                        except Exception:
                            pass
                    _put_back_stdio(saved, installed)

            # Detect a running loop WITHOUT wrapping the tool call in the same try. A
            # RuntimeError raised by the tool coroutine itself (e.g. anyio/stdio-client
            # teardown under nest_asyncio, which fires AFTER the subprocess already ran and
            # wrote its outputs) must NOT be caught and retried — that would execute the tool a
            # SECOND time (double file writes / wasted GPU compute). Only get_running_loop() is
            # guarded here.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            # nest_asyncio lets us re-enter a running loop and get the ACTUAL result; create_task
            # would hand the caller an un-awaited Task instead of the tool's output. Applied here,
            # when a loop is running: add_mcp applies it, but a wrapper the REPL worker rebuilt
            # never went through add_mcp and failed with "This event loop is already running"
            # (u14-mcp-wiring-19). apply() is idempotent.
            if loop is not None:
                try:
                    import nest_asyncio

                    nest_asyncio.apply(loop)
                except Exception:
                    pass
                tool_output = loop.run_until_complete(async_tool_call())
            else:
                tool_output = asyncio.run(async_tool_call())
            # Remembered only when it returned normally AND the payload does not say it
            # failed. The second half is not redundant: this comment used to claim the
            # exception channel was enough, and it never was -- `tools/base_mcp.py` states
            # "return error dict, never raise" and has zero `raise` statements, so for all 88
            # shipped portals a worker failure arrives here as a perfectly normal return. The
            # check now lives in `remember_successful_call` itself, where the contract is
            # written, so every caller gets it rather than this one site.
            remember_successful_call(memo_key, tool_output)
            # A tool whose ENVIRONMENT failed returns a normal error dict (base_mcp never
            # raises). Say so once, in the observation, with the way on -- the same channel as
            # the two notices above and silent under benchmarking. See agent/env_fallback.py.
            env_notice = _env_fallback_notice(agent, tool_name, tool_output)
            if env_notice:
                print(env_notice)
            return tool_output

        except Exception as e:
            # Unwrap: anyio nests the real cause inside task-group ExceptionGroups whose
            # str() drops their children, so formatting `e` directly tells the loop nothing.
            # Then, when the portal went down, append what it said on its way: for a portal that
            # crashes on import, "Connection closed" is all the exception carries and the
            # traceback is all that is useful. Not after a tool's own error or a spent budget --
            # the portal was up, and its stderr is only its startup banner.
            failure = f"MCP tool execution failed for '{tool_name}': {describe_exception(e)}"
            if _portal_went_down(e):
                failure += _portal_stderr_note(portal_stderr)
            # The exception path is where a dead portal lands (handshake timeout, "Connection
            # closed"). Printed BEFORE the raise so it reaches the captured stdout the
            # observation is built from -- the traceback follows it.
            env_notice = _env_fallback_notice(agent, tool_name, failure)
            if env_notice:
                print(env_notice)
            raise RuntimeError(failure) from e

    sync_tool_wrapper.__name__ = tool_name
    # __qualname__ too, not just __name__: CPython formats a TypeError from the QUALname, so a
    # wrapper renamed only via __name__ still reports itself as
    # `add_mcp.<locals>.make_mcp_wrapper.<locals>.sync_tool_wrapper` -- a closure path that names
    # neither the tool nor anything else in the model's namespace. These wrappers are keyword-only
    # (kwargs are forwarded verbatim to session.call_tool), so a positional call is exactly the
    # error a model hits, and with both set the message it gets back is byte-identical to the one
    # a genuinely keyword-only function raises.
    sync_tool_wrapper.__qualname__ = tool_name
    sync_tool_wrapper.__doc__ = doc
    return sync_tool_wrapper


def _process_isolation() -> bool:
    """``support_tools.process_isolation()``, imported late so this module stays importable alone."""
    try:
        from spatialomicsgym.tool.support_tools import process_isolation
    except Exception:
        return False
    try:
        return bool(process_isolation())
    except Exception:
        return False


def _benchmarking_now() -> bool:
    """Is this a scored run? The canonical five-line gate.

    Unreadable configuration answers *no*, because the failure direction that keeps behaviour
    identical to what it was before this check existed is the one that adds nothing and skips
    nothing. (The know-how loader answers *yes* to the same question, for the opposite reason:
    there the additive thing is the corpus, and the safe direction is the smaller prompt.)
    """
    try:
        from spatialomicsgym.config import default_config
    except Exception:
        return False
    return bool(getattr(default_config, "benchmarking_enabled", False))


def add_mcp(agent, config_path: str | Path = CANONICAL_CONFIG_DEFAULT) -> None:
    """
    Add MCP (Model Context Protocol) tools from configuration file.

    This method dynamically registers MCP server tools as callable functions within
    the spatialomicsgym agent system. Each MCP server is loaded as an independent module
    with its tools exposed as synchronous wrapper functions.

    Supports both manual tool definitions and automatic tool discovery from MCP servers.

    Args:
        agent: The STCoscientist agent instance
        config_path: Path to the MCP configuration YAML file containing server
                    definitions and tool specifications.

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the config file is malformed
        RuntimeError: If MCP server initialization fails
    """
    import asyncio
    import os
    import sys
    import types
    from pathlib import Path

    import nest_asyncio
    import yaml
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    nest_asyncio.apply()

    def discover_mcp_tools_sync(server_params: StdioServerParameters) -> list[dict]:
        """Discover available tools from MCP server synchronously."""
        try:

            async def _discover_async():
                saved = _restore_real_stdio()
                installed = (sys.stdin, sys.stdout, sys.stderr)
                try:
                    # errlog EXPLICITLY: stdio_client's default binds sys.stderr at ITS import
                    # time, and a capture harness (pytest capfd) may have closed that very object
                    # by now -- the launch then dies on errlog.fileno(). Call-time sys.stderr is
                    # the stream _restore_real_stdio just vetted.
                    async with stdio_client(server_params, errlog=sys.stderr) as (reader, writer):
                        async with ClientSession(reader, writer) as session:
                            # Bound the handshake: a worker that launches but stalls during init /
                            # list_tools (a heavy-import deadlock, a contended GPU, a wedged portal)
                            # must not hang add_mcp -> agent construction forever. The TimeoutError is
                            # caught below and degrades to "skip discovery for this server".
                            _hs_timeout = _handshake_timeout_seconds()
                            await asyncio.wait_for(session.initialize(), timeout=_hs_timeout)

                            # Get available tools
                            tools_result = await asyncio.wait_for(session.list_tools(), timeout=_hs_timeout)
                            tools = tools_result.tools if hasattr(tools_result, "tools") else tools_result

                            discovered_tools = []
                            for tool in tools:
                                if hasattr(tool, "name"):
                                    discovered_tools.append(
                                        {
                                            "name": tool.name,
                                            "description": tool.description,
                                            "inputSchema": tool.inputSchema,
                                        }
                                    )
                                else:
                                    print(f"Warning: Skipping tool with no name attribute: {tool}")

                            return discovered_tools
                finally:
                    _put_back_stdio(saved, installed)

            return asyncio.run(_discover_async())
        except Exception as e:
            print(f"Failed to discover tools: {e}")
            return []

    # Initialize registries if they don't exist
    agent._custom_functions = getattr(agent, "_custom_functions", {})
    agent._custom_tools = getattr(agent, "_custom_tools", {})

    # Load and validate configuration
    try:
        config_content = Path(config_path).read_text(encoding="utf-8")
        cfg: dict[str, Any] = yaml.safe_load(config_content) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"MCP config file not found: {config_path}") from None
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in MCP config: {e}") from e
    except (UnicodeDecodeError, OSError) as e:
        # A directory path, permission error, or non-UTF-8 file must not escape as a raw
        # OSError/UnicodeDecodeError — callers only guard for the documented exceptions.
        raise RuntimeError(f"Could not read MCP config '{config_path}': {e}") from e

    mcp_servers: dict[str, Any] = cfg.get("mcp_servers", {})
    if not mcp_servers:
        print("Warning: No MCP servers found in configuration")
        return

    # Manual-tool name validation (below) spawns a full portal subprocess + MCP handshake for EVERY
    # server just to compare spatialomicsgym_name against the live tool list and emit a warning. With
    # the shipped config (all 88 servers use manual `tools:`) that is ~88 serial handshakes -- tens of
    # seconds added to every agent construction / REPL `/mcp` / web re-wire, for a diagnostic the real
    # tool call would surface anyway. Default it OFF so startup is fast; opt in when debugging a
    # hand-written mcp_config_user.yaml. Computed once, not per-server.
    _validate_tool_names = os.getenv("SOG_MCP_VALIDATE_TOOL_NAMES", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    # Process each MCP server configuration
    for server_name, server_meta in mcp_servers.items():
        if not isinstance(server_meta, dict):
            print(f"Warning: Skipping server '{server_name}' — expected a mapping, got {type(server_meta).__name__}")
            continue
        if not server_meta.get("enabled", True):
            continue
        if server_meta.get("benchmark_visible") is False and _benchmarking_now():
            # A server may declare itself out of scope for a scored run. The tool registry is not
            # benchmarking-conditional anywhere else, so every tool added here widens the pool the
            # retriever ranks over -- and a visualization tool is never what a benchmark task is
            # scored on. Skipping it keeps the candidate pool and the prompt byte-identical to what
            # they were before the toolkit existed, which is a stronger position than adding to
            # them and re-measuring.
            continue

        # Validate command configuration
        cmd_list = server_meta.get("command", [])
        if not cmd_list or not isinstance(cmd_list, list):
            print(f"Warning: Invalid command configuration for server '{server_name}'")
            continue

        cmd, *args = cmd_list

        # Cross-device robustness (fresh clone / copied config / shipped canonical). The
        # canonical pins command[1] to an absolute <other-clone>/tools/<name>_mcp_server.py;
        # rebase a missing one onto THIS clone's tools/ (basename is stable) so a checkout at
        # any path wires without re-running setup. Then pin a bare `python` portal interpreter
        # to this agent's own interpreter (sys.executable) so the thin FastMCP portal launches
        # in the agent-core env regardless of PATH/activation on this box. Finally, if the
        # interpreter still can't run here (an absent per-tool env python from another box, or
        # a non-python bare name not on PATH), skip the server with a clear note instead of
        # registering tools that only fail at call time. All three are no-ops on a
        # correctly-pinned box.
        args = _rebase_script_args(args, _local_tools_dir())
        cmd = _resolve_portal_interp(cmd)
        if not _interp_runnable(cmd):
            print(
                f"Skipping MCP server '{server_name}': interpreter '{cmd}' is not available on "
                f"this machine — re-run `sog-setup` to (re)wire the config for this device."
            )
            continue
        _missing_script = _missing_script_arg(args)
        if _missing_script is not None:
            print(
                f"Skipping MCP server '{server_name}': worker script '{_missing_script}' is not present "
                f"on this machine — re-run `sog-setup` to (re)wire the config for this device."
            )
            continue

        # The DECLARED environment for this server, ``${VAR}`` placeholders left intact. They are
        # expanded per call by ``resolve_server_env`` rather than here -- see that function.
        env_spec = server_meta.get("env", {})
        if env_spec and not isinstance(env_spec, dict):
            print(f"Warning: Ignoring non-mapping 'env' for server '{server_name}'")
            env_spec = {}
        # A concrete snapshot for the handshakes below, which happen now.
        env_vars = resolve_server_env(env_spec)

        # Create module namespace for this MCP server
        mcp_module_name = f"mcp_servers.{server_name}"
        if mcp_module_name not in sys.modules:
            sys.modules[mcp_module_name] = types.ModuleType(mcp_module_name)
        server_module = sys.modules[mcp_module_name]

        tools_config = server_meta.get("tools", [])
        if tools_config and not isinstance(tools_config, list):
            # A scalar or mapping `tools:` raised TypeError at the loop below, outside every
            # per-tool guard, and took every other server's tools down with it (u14-mcp-wiring-9).
            print(f"Warning: Skipping MCP server '{server_name}': its `tools:` is not a list.")
            continue

        if not tools_config:
            # Discovery RUNS the server -- in this process, as this user. Under process isolation
            # that is the one thing model-reachable configuration must never make the server do
            # (an agent-writable mcp_config_user.yaml whose entry omits `tools:` would otherwise be
            # spawned as root at the next reload). Every shipped server declares `tools:`; a
            # hand-written one must too, and the reason is printed rather than swallowed.
            if _process_isolation():
                print(
                    f"Skipping MCP server '{server_name}': it declares no `tools:` and the portal runs "
                    f"model code under process isolation, so the server is not run here to discover "
                    f"them -- list its tools in the config."
                )
                continue
            try:
                server_params = StdioServerParameters(command=cmd, args=args, env=env_vars)
                tools_config = discover_mcp_tools_sync(server_params)

                if tools_config:
                    print(f"Discovered {len(tools_config)} tools from {server_name} MCP server")
                else:
                    print(f"Warning: No tools discovered from {server_name} MCP server")
                    continue

            except Exception as e:
                print(f"Failed to discover tools for {server_name}: {e}")
                continue

        # For manual tool definitions, validate spatialomicsgym_name against actual MCP server tools
        # (opt-in only -- see _validate_tool_names above; each check is a full portal handshake).
        # Never under process isolation: the handshake RUNS the server's command in this process,
        # and a user-config entry is model-reachable -- the same reason discovery is refused above.
        if _validate_tool_names and server_meta.get("tools") and not _process_isolation():
            try:
                server_params_check = StdioServerParameters(command=cmd, args=args, env=env_vars)
                actual_tools = discover_mcp_tools_sync(server_params_check)
                actual_names = {t.get("name") for t in actual_tools if t.get("name")}
                for tool_meta_check in tools_config:
                    if isinstance(tool_meta_check, dict):
                        bn = tool_meta_check.get("spatialomicsgym_name", "")
                        if bn and actual_names and bn not in actual_names:
                            print(
                                f"WARNING: spatialomicsgym_name '{bn}' in config for {server_name} "
                                f"does not match server tools {actual_names}. "
                                f"Tool calls will fail. Fix mcp_config_user.yaml."
                            )
            except Exception:
                pass  # Non-blocking: don't break tool loading

        # Register each tool
        for tool_meta in tools_config:
            # A single malformed tool entry (a null `parameters`, a null/str `inputSchema`, a
            # non-mapping property, a null `required`) must skip only THAT tool -- never crash the
            # whole server loop, which would drop every remaining server's tools AND skip
            # agent.configure()/the REPL injection below. Coerce each shape defensively.
            if not isinstance(tool_meta, dict):
                print(f"Warning: Skipping non-dict tool entry in {server_name}")
                continue
            if "spatialomicsgym_name" in tool_meta:
                # Manual tool definition
                tool_name = tool_meta.get("spatialomicsgym_name")
                description = tool_meta.get("description", f"MCP tool: {tool_name}")
                parameters = tool_meta.get("parameters") or {}
                if not isinstance(parameters, dict):
                    parameters = {}
                # For manual tools, check if each parameter has a "required" field
                required_param_names = []
                for param_name, param_spec in parameters.items():
                    if isinstance(param_spec, dict) and param_spec.get("required", False):
                        required_param_names.append(param_name)
            else:
                # Auto-discovered tool
                tool_name = tool_meta.get("name")
                description = tool_meta.get("description", f"MCP tool: {tool_name}")
                input_schema = tool_meta.get("inputSchema") or {}
                if not isinstance(input_schema, dict):
                    input_schema = {}
                parameters = input_schema.get("properties") or {}
                if not isinstance(parameters, dict):
                    parameters = {}
                # For auto-discovered tools, get required list from inputSchema top level
                required_param_names = input_schema.get("required") or []
                if not isinstance(required_param_names, list):
                    required_param_names = []

            if not tool_name:
                print(f"Warning: Skipping tool with no name in {server_name}")
                continue
            if not is_bindable_tool_name(tool_name):
                print(
                    f"Warning: Skipping {server_name}.{tool_name!r}: a tool name must be a Python identifier "
                    "that is not a keyword, a builtin or a common REPL alias, or it would shadow that name "
                    "in every account's cells."
                )
                continue

            # Everything below publishes `parameters` to the model -- the parameter lists, the tool
            # schema, and the prompt renderer that turns each one into "[Default: <value>]". A
            # credential parked in a default would be mailed to the LLM provider on every turn, so
            # it is dropped once, here, and every consumer downstream inherits the clean version.
            parameters, credential_defaults = _drop_credential_defaults(parameters)
            for param_name in credential_defaults:
                # Names only: the point is to say which parameter to fix, not to print the secret.
                print(
                    f"Warning: {server_name}.{tool_name} declares a credential as the default for "
                    f"'{param_name}'; it was withheld from the tool catalog. Remove it from the "
                    f"config and let the wrapper read the credential from the environment."
                )

            # Create wrapper function
            wrapper_function = make_mcp_wrapper(agent, cmd, args, tool_name, description, env_spec)

            # Add to module namespace
            setattr(server_module, tool_name, wrapper_function)

            # Build parameter lists
            required_params, optional_params = [], []
            for param_name, param_spec in parameters.items():
                if not isinstance(param_spec, dict):
                    param_spec = {}
                param_info = {
                    "name": param_name,
                    "type": str(param_spec.get("type", "string")),
                    "description": param_spec.get("description", ""),
                    "default": param_spec.get("default", None),
                }

                # Check if parameter is required based on the required_param_names list
                if param_name in required_param_names:
                    required_params.append(param_info)
                else:
                    optional_params.append(param_info)

            # Tell the function object what it accepts. Without this the wrapper reports
            # `(**kwargs)`, and a model told by the prompt to "use the wrapper's actual signature"
            # learns nothing and has to guess kwarg names it cannot guess.
            attach_kwarg_signature(wrapper_function, required_params, optional_params)
            # Everything another process needs to rebuild THIS wrapper with the same factory: the
            # spawn triple was closed over and recorded nowhere, so the REPL worker could not have
            # re-created it. Additive; nothing in-process reads it.
            wrapper_function._sog_spec = {
                "name": tool_name,
                "cmd": cmd,
                "args": list(args),
                "doc": description,
                "env_spec": dict(env_spec or {}),
                "required": required_params,
                "optional": optional_params,
                # The import path the retrieval prompt names ("from mcp_servers.<server> import
                # <tool>"), so the worker can make it resolve there as it does here.
                "module": mcp_module_name,
            }

            # Create tool schema
            tool_schema = {
                "name": tool_name,
                "description": description,
                "parameters": parameters,
                "required_parameters": required_params,
                "optional_parameters": optional_params,
                "module": mcp_module_name,
                "fn": wrapper_function,
            }

            # Register in tool registry (only if retriever is enabled)
            if hasattr(agent, "tool_registry"):
                agent.tool_registry.register_tool(tool_schema)

            # Add to module2api mapping. Dedup by tool name (mirror add_tool's
            # module2api path and ToolRegistry.register_tool): a second add_mcp for a
            # module that already exists -- REPL ``/mcp`` run twice, a programmatic
            # re-wire, or reloading an edited user config -- must REPLACE the existing
            # entry, not append a duplicate. module2api is the sole tool catalog the
            # system prompt is built from when ``use_tool_retriever=False`` (execution.py),
            # so a blind append lists each re-wired tool N times and grows the prompt
            # unboundedly.
            if mcp_module_name not in agent.module2api:
                agent.module2api[mcp_module_name] = []
            existing_entry = None
            for existing in agent.module2api[mcp_module_name]:
                if existing.get("name") == tool_schema["name"]:
                    existing_entry = existing
                    break
            if existing_entry is not None:
                existing_entry.update(tool_schema)
            else:
                agent.module2api[mcp_module_name].append(tool_schema)

            # Add to instance registries
            agent._custom_functions[tool_name] = wrapper_function
            agent._custom_tools[tool_name] = {
                "name": tool_name,
                "description": description,
                "module": mcp_module_name,
            }

    # Update agent configuration
    agent.configure()


def create_mcp_server(agent, tool_modules=None):
    """
    Create an MCP server object that exposes internal SpatialOmicsLab tools.
    This gives you control over when and how to run the server.

    Args:
        agent: The STCoscientist agent instance
        tool_modules: List of module names to expose (default: all in agent.module2api)

    Returns:
        FastMCP server object that you can run manually
    """
    import importlib

    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("SpatialOmicsGymTools")
    modules = tool_modules or list(agent.module2api.keys())

    registered_tools = 0

    for module_name in modules:
        try:
            # Import the actual module
            module = importlib.import_module(module_name)
            # Get tools for this module
            module_tools = agent.module2api.get(module_name, [])

            for tool_schema in module_tools:
                tool_name = tool_schema.get("name")
                if not tool_name:
                    continue

                try:
                    # Get the actual function
                    fn = getattr(module, tool_name, None)
                    if fn is None:
                        fn = getattr(agent, "_custom_functions", {}).get(tool_name)

                    if fn is None:
                        print(f"Warning: Could not find function '{tool_name}' in module '{module_name}'")
                        continue

                    # Extract parameters from your specific schema format
                    required_params = tool_schema.get("required_parameters", [])
                    optional_params = tool_schema.get("optional_parameters", [])

                    # Generate the wrapper function
                    wrapper_func = generate_mcp_wrapper_from_spatialomicsgym_schema(
                        fn, tool_name, required_params, optional_params
                    )

                    # Register with MCP
                    mcp.tool()(wrapper_func)
                    registered_tools += 1

                except Exception as e:
                    print(f"Warning: Failed to register tool '{tool_name}': {e}")
                    continue

        except ImportError as e:
            print(f"Warning: Could not import module '{module_name}': {e}")
            continue

    print(f"Created MCP server with {registered_tools} tools")
    return mcp


def generate_mcp_wrapper_from_spatialomicsgym_schema(original_func, func_name, required_params, optional_params):
    """Generate wrapper function based on SpatialOmicsLab schema format."""

    # Combine all parameters
    all_params = required_params + optional_params

    if not all_params:
        # No parameters
        def wrapper() -> dict:
            try:
                result = original_func()
                if isinstance(result, dict):
                    return result
                return {"result": result}
            except Exception as e:
                return {"error": str(e)}

        wrapper.__name__ = func_name
        wrapper.__qualname__ = func_name  # see make_mcp_wrapper: __name__ alone leaves tracebacks anonymous
        wrapper.__doc__ = original_func.__doc__
        return wrapper

    else:
        # Has parameters
        def wrapper(**kwargs) -> dict:
            try:
                # Build arguments dict
                filtered_kwargs = {}

                # Add required parameters
                for param_info in required_params:
                    param_name = param_info["name"]
                    if param_name in kwargs and kwargs[param_name] is not None:
                        filtered_kwargs[param_name] = kwargs[param_name]

                # Add optional parameters only if provided and not None
                for param_info in optional_params:
                    param_name = param_info["name"]
                    if param_name in kwargs and kwargs[param_name] is not None:
                        filtered_kwargs[param_name] = kwargs[param_name]

                result = original_func(**filtered_kwargs)
                if isinstance(result, dict):
                    return result
                return {"result": result}
            except Exception as e:
                return {"error": str(e)}

        # Set function metadata
        wrapper.__name__ = func_name
        wrapper.__qualname__ = func_name  # see make_mcp_wrapper: __name__ alone leaves tracebacks anonymous
        wrapper.__doc__ = original_func.__doc__

        # Create proper signature
        new_params = []

        # Map the schema's type names to Python types. FastMCP builds both the advertised JSON schema
        # and the pydantic validation model from these annotations, so an unrecognised name must NOT
        # fall back to ``str``: that annotation is advertised as {"type": "string"} and rejects the
        # parameter's real value ("Input should be a valid string ... input_type=list") before the
        # tool function is entered. Fall back to permissive ``Any`` instead -- the same choice, for
        # the same reason, as the sibling converter in ``utils/tool_conversion.py``: an unknown type
        # name is unconstrained here and the tool function does the real checking. Both spellings of
        # each name are listed because the shipped schemas use both (``List[str]`` and ``list[str]``).
        type_map = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "number": float,
            "bool": bool,
            "boolean": bool,
            "dict": dict,
            "Dict": dict,
            "list": list,
            "List": list,
            "List[str]": list[str],
            "list[str]": list[str],
            "List[int]": list[int],
            "list[int]": list[int],
            "List[float]": list[float],
            "list[float]": list[float],
            "Any": Any,
        }

        def resolve_type(type_name):
            # ``type_name`` comes from a file- or LLM-provided schema, so it is not guaranteed to be
            # a string: an unhashable value (a nested schema dict) would raise from ``.get``.
            return type_map.get(type_name, Any) if isinstance(type_name, str) else Any

        # Add required parameters
        for param_info in required_params:
            param_name = param_info["name"]
            param_type = resolve_type(param_info["type"])

            new_params.append(inspect.Parameter(param_name, inspect.Parameter.KEYWORD_ONLY, annotation=param_type))

        # Add optional parameters
        for param_info in optional_params:
            param_name = param_info["name"]
            param_type = resolve_type(param_info["type"])

            # Make it optional
            optional_type = param_type | None

            new_params.append(
                inspect.Parameter(param_name, inspect.Parameter.KEYWORD_ONLY, default=None, annotation=optional_type)
            )

        # Set the signature
        wrapper.__signature__ = inspect.Signature(new_params, return_annotation=dict)

        return wrapper
