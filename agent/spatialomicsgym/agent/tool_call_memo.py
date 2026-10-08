"""A per-script memo of successful MCP tool calls, so one code block never runs the same call twice.

Why this exists (live round-4, case s4a): some models write out a whole imagined session in ONE
message -- an ``<execute>`` block, then a fabricated observation, then the "next" ``<execute>``
block re-typed almost verbatim.  ``spatialomicsgym.action`` deliberately concatenates multiple
plain-python blocks from one message into a single script (the model's server-side stop-sequence
stripping makes it emit whole plans, and running only the first block strands the rest), so both
copies of the re-typed block really run.  In s4a that meant cell2location dispatched twice on the
full lymph-node pair: 4h15m of compute, then 3h39m more for a byte-identical result that
overwrote the first.  The two blocks differed in six cosmetic lines (a dropped import, a reworded
print), so text-level dedup of the blocks cannot catch this -- but the expensive part, the tool
DISPATCH, was identical.  This memo dedups at the dispatch.

The contract, enforced with the functions below:

- **Scope is one script run.**  ``execution.py``'s execute node arms the memo just before each
  python ``<execute>`` script and disarms it (clearing it) right after.  A deliberate re-run in a
  LATER block or turn -- "run it again to check stability" -- always dispatches for real.
- **Disarmed means invisible.**  Library users who call the wrappers directly (notebooks,
  ``add_mcp`` outside the ReAct loop) never arm the memo, so their calls are untouched.
- **Success only.**  Failures are never remembered: a retry after a transient error must really
  retry.  Only a call that returned normally is eligible.
- **Byte-identical arguments only.**  The key is the tool name plus the ``repr`` of the kwargs
  sorted by name -- keyword ORDER does not defeat it, but any changed value is a different call
  and runs for real.  Near-identical is different on purpose: a changed seed, path, or epoch
  count must dispatch.
- **Loud, never silent.**  A hit prints a notice into the captured stdout (the model's
  observation) naming the tool and saying how to force a fresh run.  A tool whose output varies
  on identical arguments (an unseeded sampler called twice in one block) loses that variation for
  the second call -- the notice discloses exactly that trade.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

#: Hard caps so a pathological script (hundreds of distinct calls, or megabytes of inline data in
#: a kwarg) cannot turn the memo into a memory sink.  Past the cap we simply stop remembering --
#: every call still runs.
_MAX_ENTRIES = 128
_MAX_KEY_CHARS = 1_000_000

_memo: dict[tuple[str, str], Any] = {}
_armed = False
#: The thread the current script scope was armed on. Writes and reads from any other thread are
#: ignored, which is what stops a TIMED-OUT cell's orphan poisoning the next cell's memo.
#:
#: In-process timeouts are not kills: ``utils.execution`` raises ``SystemExit`` into the thread with
#: ``PyThreadState_SetAsyncExc``, which its own comment calls "not 100% reliable" and which cannot
#: interrupt a thread inside a C call. So the cell's ``finally`` disarms while the orphan is still
#: running, the NEXT cell arms a fresh memo, and the orphan's in-flight MCP wrapper calls
#: ``remember_successful_call`` into it. The new cell then makes the identical call and is handed
#: the ORPHAN's result, with a notice saying it "was already called with these exact arguments
#: earlier in this code block and succeeded". Low probability, high blast radius: this memo exists
#: for four-hour cell2location calls.
#:
#: Thread identity is the right discriminator because the orphan is by construction the PREVIOUS
#: cell's thread. Under process isolation the memo is per-process and this never applies.
_owner_thread: int | None = None


_scope_lock = threading.Lock()


def arm_tool_call_memo() -> None:
    """Start a fresh script scope: forget everything remembered and begin deduplicating.

    On the thread that will RUN the script. The in-process execute node armed on the LangGraph
    thread and then ran the cell in ``run_with_timeout``'s own thread, whose wrapper calls the
    ownership check then ignored -- so the memo never deduplicated anything on the CLI or in a
    benchmark, the runs it was written for (hunt 2026-09-30, u15-validation-1).
    """
    global _armed, _owner_thread
    with _scope_lock:
        _memo.clear()
        _armed = True
        _owner_thread = threading.get_ident()


def disarm_tool_call_memo() -> None:
    """End the script scope: forget everything and stop deduplicating (the default state)."""
    global _armed, _owner_thread
    with _scope_lock:
        _memo.clear()
        _armed = False
        _owner_thread = None


def release_tool_call_memo() -> None:
    """End the scope THIS thread armed, and nothing else.

    For a cell's own thread. A cell that timed out keeps running (an in-process timeout is not a
    kill) and reaches its ``finally`` after the next cell has armed; an unconditional disarm there
    would end the next cell's scope.
    """
    global _armed, _owner_thread
    with _scope_lock:
        if _owner_thread != threading.get_ident():
            return
        _memo.clear()
        _armed = False
        _owner_thread = None


def _this_thread_owns_the_scope() -> bool:
    return _owner_thread is not None and threading.get_ident() == _owner_thread


def _inputs_fingerprint(kwargs: dict[str, Any]) -> str:
    """The working directory and the (mtime, size) of each INPUT path the arguments name.

    Byte-identical arguments are not an identical call when the file they name was rewritten in
    between: a per-sample loop writing tmp.h5ad and calling the tool on it got sample 1's result for
    every later sample, and a relative path means another file after os.chdir (u15-validation-11).
    Output arguments are left out on purpose -- the first call writes there, and counting that would
    defeat the dedup this memo exists for (two identical dispatches in one block).
    """
    parts: list[Any] = []
    try:
        parts.append(os.getcwd())
    except OSError:
        parts.append(None)
    for key, value in sorted(kwargs.items()):
        lowered = str(key).lower()
        if lowered.startswith(("output", "out_")) or lowered in ("out", "outdir"):
            continue
        if not isinstance(value, str) or not value or len(value) > 4096 or "\n" in value or "\0" in value:
            continue
        try:
            st = os.stat(value)
        except (OSError, ValueError):
            continue
        parts.append((key, st.st_mtime_ns, st.st_size))
    return repr(parts)


def tool_call_memo_key(tool_name: str, kwargs: dict[str, Any]) -> tuple[str, str] | None:
    """The identity of one call: tool name + kwargs repr, sorted by keyword name.

    Returns ``None`` -- never raises -- when the kwargs cannot be canonicalized (an object whose
    ``repr`` raises) or the repr is unreasonably large; such calls are simply never memoized.
    """
    try:
        canonical = repr(sorted(kwargs.items())) + "|" + _inputs_fingerprint(kwargs)
    except Exception:
        return None
    if len(canonical) > _MAX_KEY_CHARS:
        return None
    return (str(tool_name), canonical)


def lookup_successful_call(key: tuple[str, str] | None) -> tuple[bool, Any]:
    """``(True, earlier_result)`` if this exact call already succeeded in this script scope."""
    if not _armed or key is None or key not in _memo or not _this_thread_owns_the_scope():
        return (False, None)
    return (True, _memo[key])


#: Top-level ``status`` values a portal uses to say the call did not work.
_FAILURE_STATUSES = frozenset({"error", "dep_missing"})


def _reports_failure(result: Any) -> bool:
    """True when the tool's own payload says it failed.

    THE BUG THIS EXISTS FOR. The contract above says "Success only. Failures are never
    remembered: a retry after a transient error must really retry." That was implemented at the
    call site as *did the wrapper raise?* -- and **no shipped portal raises**. `tools/base_mcp.py`
    states its own contract as "return error dict, never raise" and contains zero `raise`
    statements, so for all 88 portals a worker failure is a normal return. Every failure was
    therefore remembered as a success, and the next identical call in the same block got the
    cached error back with a notice telling the model the first call *succeeded*.

    What that cost is the commonest self-repair a model does: call a tool, read "no such file",
    create the directory or write the missing input, retry the same arguments. The retry was
    refused and the model then reasoned from an error it had already fixed.

    Results arrive as TEXT here (`mcp_integration._extract_tool_result` returns a string), so
    this parses rather than duck-types.

    ANYTHING UNPARSEABLE IS STILL MEMOIZED, deliberately. A tool whose output is not JSON cannot
    be read as a failure, and refusing to memoize everything unreadable would reopen the case
    this memo was written for -- cell2location dispatched twice on the full lymph-node pair,
    4h15m then 3h39m more for a byte-identical result. Narrow the memo only where the payload
    actually says "error".
    """
    if isinstance(result, dict):
        payload: Any = result
    elif isinstance(result, str):
        text = result.strip()
        if not text.startswith("{"):
            return False
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return False
    else:
        return False
    if not isinstance(payload, dict):
        return False
    status = payload.get("status")
    return isinstance(status, str) and status.strip().lower() in _FAILURE_STATUSES


def remember_successful_call(key: tuple[str, str] | None, result: Any) -> None:
    """Record a call that returned normally AND did not report failure.

    No-op when disarmed, keyless, at capacity, or when the payload says the call failed. The
    last of those is what makes the "success only" line in the contract above true; see
    :func:`_reports_failure` for why checking the exception channel alone never could.
    """
    if not _armed or key is None or not _this_thread_owns_the_scope():
        return
    if _reports_failure(result):
        return
    if len(_memo) >= _MAX_ENTRIES and key not in _memo:
        return
    _memo[key] = result


def duplicate_call_notice(tool_name: str) -> str:
    """The line printed into the observation when a duplicate dispatch is skipped."""
    return (
        f"[duplicate-call guard] '{tool_name}' was already called with these exact arguments "
        "earlier in this code block and succeeded -- returning that earlier result instead of "
        "running the tool again. Change an argument or call it from a new <execute> block to "
        "force a fresh run."
    )
