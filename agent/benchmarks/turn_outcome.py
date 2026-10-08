"""What a finished turn actually reports, as opposed to what it returned without raising.

Two facts that every harness in this directory needed, and that every harness got wrong in the
same two ways.

**A turn that stopped early returns normally.** ``STCoscientist.go`` does not raise on a give-up,
on a ``GraphRecursionError`` once the step budget is spent, or on a provider 429:
``_iter_react_stream`` catches all three, records ``result.degrade_note`` and yields the partial
transcript (``agent/stcoscientist.py:441-447``), and ``go`` copies the note to
``last_turn_degraded`` (``:2207``). The note is the only place the fact lives. ``chat_cli``
(``_EXIT_DEGRADED = 4``) and the portal read it; the benchmarks did not, so a run that burned
~1000 paid calls and then gave up was scored as a run that finished, and its empty output
directory was read as a tool that produced nothing rather than a turn that never got there.

**``tool_name in " ".join(log)`` is a tautology.** The log's first entry is the pretty-printed
prompt, and the retriever writes the recommended tool's name into that prompt up to six times --
so the test was true before the model had emitted a token, and ``invoked: True`` meant only that
the harness had asked for the tool. This module answers it from the two places an invocation
actually leaves a mark: ``_execution_results``, which ``execution.py:889`` appends the *cleaned
code of every cell that ran* to, and the ``Tool:``/``Input:`` lines ``pretty_print`` writes for a
provider tool-call block. Neither can be satisfied by the prompt.

Every accessor here is a ``getattr`` with a default. These harnesses drive stub agents in their
own tests, and a missing attribute must read as "nothing to report" rather than raise inside a
scorer -- a scorer that crashes loses the run it was scoring.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

__all__ = ["degrade_note", "executed_code", "set_aside_previous_outputs", "tool_was_invoked"]

#: Transcript entries that carry what the agent was TOLD rather than what it did. ``pretty_print``
#: titles each entry with ``get_msg_title_repr(message.type.title() + " Message")``, so the human
#: prompt and any system block are identifiable by their own header and nothing else has to be.
#: ``.upper()`` covers the list-content branch, which titles with ``.title().upper()``.
_NOT_THE_AGENTS_DOING = ("HUMAN MESSAGE", "SYSTEM MESSAGE")


def degrade_note(agent) -> str:
    """Why this turn stopped early, or ``""`` when it ran to a real answer."""
    return str(getattr(agent, "last_turn_degraded", "") or "").strip()


def _full_cell(record: dict) -> str:
    """The whole of the code a recorded cell ran, read back from the message that carried it.

    The record's ``code`` is capped at 4000 characters by the writer (``execution.py``). A long
    fix-then-run cell -- the runner's own prompt invites one -- that called the tool past that
    point read as "not invoked" and was recorded fail_prompt although the tool ran and wrote valid
    outputs (hunt 2026-09-30, u33a-bench-runner-30). ``triggering_message`` is the AI message the
    cell came from, recorded beside it, and ``action.extract_runnable_code`` is the extractor
    ``execute()`` itself ran on it -- so this is the cell that ran, uncapped, and nothing else in
    the message (prose naming the tool is not read).
    """
    capped = str(record.get("code") or "")
    message = record.get("triggering_message")
    if not isinstance(message, str) or not message:
        return capped
    try:
        from spatialomicsgym.action import extract_runnable_code

        full = extract_runnable_code(message)
    except Exception:
        return capped
    return full if full and len(full) > len(capped) else capped


def executed_code(agent) -> str:
    """Every cell this turn actually ran, concatenated; ``""`` when the turn ran nothing.

    ``go`` clears ``_execution_results`` at the top of each turn (``stcoscientist.py:2187``), so
    this is this turn's cells and not the session's. Each cell is read whole -- see
    :func:`_full_cell` for why the recorded ``code`` alone is not enough.
    """
    parts = []
    for record in getattr(agent, "_execution_results", None) or []:
        if isinstance(record, dict):
            parts.append(_full_cell(record))
    return "\n".join(parts)


def _tool_call_lines(log) -> str:
    """The provider tool-call blocks in the transcript, as ``pretty_print`` wrote them.

    ``logging_utils.pretty_print`` renders a ``tool_use`` content block as ``Tool: <name>`` and
    ``Input: <args>``. Those two lines are the entire record of a call the model made through the
    provider's tool API instead of through an ``<execute>`` cell, so they are kept and everything
    else in the entry -- reasoning, observation, final answer -- is not. Prompt and system entries
    are dropped whole, which is what keeps this from re-introducing the tautology if a prompt ever
    starts a line with ``Tool:``.
    """
    kept = []
    for entry in log or []:
        text = str(entry)
        head = text[:400].upper()
        if any(marker in head for marker in _NOT_THE_AGENTS_DOING):
            continue
        # The block's exact shape: a ``Tool:`` line with its ``Input:`` line right under it. A bare
        # ``Tool: <name>`` in the model's own prose -- a plan, a summary -- was counted as a call
        # (hunt 2026-09-30, u33a-bench-runner-30).
        lines = [line.strip() for line in text.splitlines()]
        for here, after in zip(lines, lines[1:], strict=False):
            if here.startswith("Tool:") and after.startswith("Input:"):
                kept.extend((here, after))
    return "\n".join(kept)


def tool_was_invoked(agent, log, tool_name: str) -> bool:
    """Whether ``tool_name`` appears in something the agent DID, not in what it was asked."""
    if not tool_name:
        return False
    return tool_name in executed_code(agent) or tool_name in _tool_call_lines(log)


def set_aside_previous_outputs(output_dir) -> str | None:
    """Give this run an empty output directory, moving a previous run's contents aside first.

    The harnesses write each (dataset, tool[, config]) run into a fixed directory and score
    whatever they find in it -- the inspector's rglob, the newest ``*.h5ad``, the first gene table.
    Nothing cleared it between runs, so a re-run whose turn invoked the tool and then wrote nothing
    was scored on the previous run's file and published as this run's number; in the auto-tuning
    harness a crashed candidate could inherit an earlier one's score (hunt 2026-09-30,
    u33a-bench-runner-15). Renamed, never deleted -- ``<name>.superseded-<timestamp>`` beside it --
    because that earlier output is someone's result. Returns where it went, or ``None`` when there
    was nothing there.
    """
    out = Path(output_dir)
    if not out.is_dir() or not any(out.iterdir()):
        out.mkdir(parents=True, exist_ok=True)
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = out.with_name(f"{out.name}.superseded-{stamp}")
    n = 1
    while dest.exists():
        dest = out.with_name(f"{out.name}.superseded-{stamp}-{n}")
        n += 1
    os.replace(out, dest)
    out.mkdir(parents=True, exist_ok=True)
    return str(dest)
