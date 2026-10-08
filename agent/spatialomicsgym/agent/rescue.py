"""One rescue round for a turn that ended unsolved -- bounded, disclosed, and off under benchmarking.

THE GAP THIS FILLS. Three guards in ``execution.generate`` end a turn without an answer: the same
execution error repeated, reasoning with no action, no parseable action. Each writes a
``degraded`` note and stops. The only default-on post-turn hook, ``_run_post_analysis_review``,
fires only when a ``manifest.json`` was written -- and a turn that gave up on an error wrote none
-- so the layer that could have kept going was inert precisely when it was needed.

WHAT A RESCUE IS. One more ReAct round on the same graph and thread, whose prompt was written by
this module rather than by the user: the task, why the first attempt stopped, the last real error,
what already succeeded and must not be re-run, what is still bound in the REPL, and which tools'
environments failed this turn. ``RESCUE_ROUNDS`` is a constant, not a knob: the user asked for one.
It shares the post-analysis follow-on budget -- a turn that already ran a follow-on round gets no
rescue -- so no turn ever gets two extra rounds.

WHAT IT IS NOT. Not a retry: the prompt says "take a different approach". Not for every early
stop: a turn cut off by the output-token limit, the step budget or a provider error is the T4
recovery audit's territory and gets no rescue here (``rescue_reason`` returns ``None``). And it
never judges its own result -- whatever the round produces is the turn's answer, with
``last_turn_degraded`` describing the round's own outcome and ``last_turn_rescued`` recording why
the first attempt stopped, so every front door can say both facts.

GATE. :func:`rescue_active` is false under ``benchmarking_enabled`` whatever the knob says; a
scored turn invokes the graph exactly once.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

#: How many rescue rounds a turn may get. A constant on purpose.
RESCUE_ROUNDS = 1

RESCUE_TAG = "[ST-Coscientist rescue round]"
NO_SOLUTION_REASON = "the turn ended without a <solution>"

_SOLUTION_TAG_RE = re.compile(r"<solution>", re.IGNORECASE)
_PROVIDER_NOTE_PREFIX = "[ST-Coscientist:"


def rescue_active() -> bool:
    """False whenever this layer must be invisible: benchmarking wins, then the user's knob."""
    try:
        from spatialomicsgym.config import default_config
    except Exception:
        return False
    if getattr(default_config, "benchmarking_enabled", False):
        return False
    return bool(getattr(default_config, "unsolved_rescue_enabled", True))


def rescue_reason(degrade_note: str | None, answer_text: str | None) -> str | None:
    """Why this turn qualifies for a rescue, or ``None`` when it does not.

    Eligible: the three give-ups ``execution.generate`` writes (imported from there, so the strings
    cannot drift apart), and a turn that ended with no ``<solution>`` at all. Not eligible: the
    output-token truncation give-up, the step-budget note and the provider-error note -- those
    start with ``[ST-Coscientist:`` and belong to the recovery audit, not to a second round.
    """
    from spatialomicsgym.agent.execution import RESCUABLE_GIVEUPS

    note = (degrade_note or "").strip()
    if note:
        return note if note in RESCUABLE_GIVEUPS else None
    text = answer_text if isinstance(answer_text, str) else ""
    if _SOLUTION_TAG_RE.search(text):
        return None
    return NO_SOLUTION_REASON


def succeeded_tools(agent: Any) -> list[str]:
    """MCP tools that ran successfully this turn, read off ``_execution_results``.

    A cell that ran clean and named a registered tool is taken to have used it; the memo layer
    remembers exact calls but not names, and a name is what the prompt needs.
    """
    # A tool whose environment failed this turn returned an error dict the cell printed without
    # raising, so the cell is ``ok`` and the name is in its code -- and the prompt then said the
    # tool "SUCCEEDED and must NOT be re-run" one line above "its environment failed -- do NOT
    # call it again". The ledger the notice writes is the authority on which tools did not run.
    failed = set(getattr(agent, "_env_failures", {}) or {})
    names = sorted(name for name in (getattr(agent, "_custom_functions", {}) or {}) if name not in failed)
    entries = getattr(agent, "_execution_results", None) or []
    used: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("ok"):
            continue
        called = _called_names(str(entry.get("code") or ""))
        for name in names:
            if name not in used and name in called:
                used.append(name)
    return used


def _called_names(code: str) -> set[str]:
    """Names CALLED in ``code`` -- a name token followed by ``(`` -- never ones in comments or strings.

    A regex over the raw text counted ``# next step: spagcn_cluster(adata, n_clusters=7)`` in a clean
    cell as a successful call, and the rescue prompt then said spagcn_cluster "SUCCEEDED and must NOT
    be re-run" one line below the error that very tool had just hit four times (u12-react-9). Code
    that does not tokenize (a cell the loop cleaned oddly) falls back to the text with its ``#``
    comments dropped.
    """
    import io
    import tokenize

    called: set[str] = set()
    try:
        skip = (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT)
        tokens = [t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type not in skip]
        for prev, cur in zip(tokens, tokens[1:], strict=False):
            if prev.type == tokenize.NAME and cur.type == tokenize.OP and cur.string == "(":
                called.add(prev.string)
        return called
    except (tokenize.TokenError, IndentationError, SyntaxError):
        text = "\n".join(line.split("#", 1)[0] for line in code.splitlines())
        return set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text))


def build_rescue_prompt(
    *,
    task: str,
    reason: str,
    last_error: str = "",
    succeeded: list[str] | None = None,
    repl_names: str = "",
    env_failed: list[str] | None = None,
) -> str:
    """The rescue round's prompt. The template is ASCII only -- a non-ASCII bullet in an
    ``<execute>`` block has crashed the executor before, and this text is the model's template for
    the round.

    The task and the last error are NOT folded. The round is run on this message alone, so it is
    the model's only statement of the task, and folding it turned a Chinese question into "?????"
    and a path with one accented letter into a path that does not exist -- the round then answered
    something else (hunt 2026-09-30, u12-react-4). They lose only control characters.
    """
    ran = ", ".join(succeeded or []) or "none"
    bound = repl_names.strip() or "none"
    failed = ", ".join(env_failed or []) or "none"
    error = (last_error or "").strip() or "(no execution error was recorded)"

    def ascii_only(line: str) -> str:
        return line.encode("ascii", "replace").decode("ascii")

    lines = [
        ascii_only(f"{RESCUE_TAG} The previous attempt at this task stopped without solving it."),
        f"Task: {_verbatim(task)}",
        ascii_only(f"Why it stopped: {reason}."),
        f"The last error it hit: {_verbatim(error)}",
        ascii_only(
            f"Steps that already SUCCEEDED this turn and must NOT be re-run (their results are in the REPL): {ran}"
        ),
        ascii_only(f"Names still bound in the REPL from the previous attempt: {bound}"),
        ascii_only(f"Tools whose environment failed this turn -- do NOT call them again: {failed}"),
        "Take a DIFFERENT approach from the one that stopped: a different tool, a different method, or a "
        "simpler step that reaches part of the goal. Keep every <execute> block ASCII-only. End with a "
        "<solution> that says what was obtained and what is still missing, and names any substitute "
        "method used in place of a tool that could not run.",
    ]
    return "\n".join(lines)


def _verbatim(text: Any) -> str:
    """``text`` as written, minus control characters (newline and tab kept) and lone surrogates."""
    return "".join(ch for ch in str(text or "") if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cs"))
