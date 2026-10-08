"""Carrying one turn of a session into the next.

``go()`` and ``go_stream()`` are stateless: each builds ``inputs = {"messages": [HumanMessage(...)]}``
from scratch, and ``AgentState.messages`` is a plain ``list[BaseMessage]``, so LangGraph's default
channel overwrites rather than appends. The ``MemorySaver`` and the hardcoded ``thread_id: 42`` look
like conversation memory but persist a state the next call discards.

For a benchmark that is exactly right -- instances must not contaminate each other. For a person
holding a conversation it is not: a live two-turn probe had the agent spend 116 s searching ``/tmp``
for output it had just written, then ask for the input path it had used two minutes earlier.

So continuity is opt-in, and what gets carried is a *recap*, not a transcript: the earlier questions
and final answers, bounded and ASCII-flattened. Replaying whole ReAct transcripts would blow the
context window on the second follow-up and re-feed stale observations as if they were current.

The recap also carries a one-line *trajectory digest* per turn -- which tools ran, which files came
out. Questions and answers alone turned out not to be enough: a final answer is written for a person
and routinely says "the deconvolution is complete, results are in the output directory" without
naming the tool or the directory, so the next turn could not tell whether ``cell2location`` or
``card`` had produced the numbers it was being asked to compare. The digest is mined from the step
log rather than from the prose, so it says what happened even when the answer does not. It is
deliberately a *digest* and not the transcript: names and paths, no observations, no generated code.
"""

from __future__ import annotations

import re

# How many earlier turns a recap may mention. Three covers the follow-up patterns people actually
# use ("redo it with 12", "now compare those two", "what about the other slide") without letting the
# preamble crowd out the question being asked.
MAX_RECAP_TURNS = 3

# Per-field clips. A question is short by nature; an answer can be thousands of characters of
# checklists and tables, of which the tail (the conclusion, the paths) is the useful part.
_MAX_QUESTION_CHARS = 600
_MAX_ANSWER_CHARS = 1200
# The digest is names and paths, so it is short by construction; the clip is a backstop against a
# pathological run, not the normal case. Three turns of it must stay small next to the answers.
_MAX_TRAJECTORY_CHARS = 400

# How many of each to name. Past a handful the list stops being a reminder and becomes noise, and
# the ones that matter to a follow-up are the last few -- what the turn ended up doing.
_MAX_DIGEST_TOOLS = 6
_MAX_DIGEST_ARTIFACTS = 6

# ``Tool: <name>`` is what utils.logging_utils.pretty_print writes for a provider tool_use block.
_TOOL_CALL_RE = re.compile(r"^\s*Tool:\s*([A-Za-z_][\w.\-]{1,80})", re.MULTILINE)
# The portal functions the model calls from inside generated code, where there is no tool_use block
# to read -- ``run_cell2location(...)``. This is the same shape ``sog_install/demo.py`` mines for routing.
_RUN_TOOL_RE = re.compile(r"\brun_[a-z0-9_]{2,60}")
# Files a turn produced. Anchored on a separator so a bare "results.csv" in prose does not qualify:
# the point of the digest is to hand the next turn a path it can reuse verbatim.
_ARTIFACT_RE = re.compile(
    r"(?:[A-Za-z]:\\|/|\./|\.\./)[\w./\\+\-]{2,200}?"
    r"\.(?:h5ad|h5|loom|mtx|csv|tsv|txt|parquet|xlsx|json|png|pdf|svg|html)\b"
)


def _ascii(text: str) -> str:
    """Flatten to ASCII, keeping the shape of the text.

    A recap is the model's own prior output fed back into a prompt, and non-ASCII characters in a
    prompt are a known failure here: bullets and check marks reach generated code and raise
    SyntaxError, after which the agent loops rewriting the same broken line. The real turn-0 answer
    this was built from contained U+2713.
    """
    replacements = {
        "✓": "[done]",
        "✔": "[done]",
        "✗": "[failed]",
        "✘": "[failed]",
        "•": "-",
        "‣": "-",
        "●": "-",
        "→": "->",
        "←": "<-",
        "—": "--",
        "–": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "…": "...",
        " ": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Anything still outside ASCII (a gene name with a Greek letter, an emoji) is dropped rather
    # than guessed at: it is never the load-bearing part of a recap.
    return text.encode("ascii", "ignore").decode("ascii")


def _clip(text: str, limit: int, *, keep: str = "head") -> str:
    text = " ".join(text.split()) if len(text) < 200 else text
    if len(text) <= limit:
        return text
    if keep == "tail":
        return "[...truncated...] " + text[-limit:]
    return text[:limit] + " [...truncated...]"


def summarize_trajectory(log) -> str:
    """One ASCII line naming the tools a turn ran and the files it produced, or ``""``.

    ``log`` is ``agent.log``: the list of pretty-printed ReAct steps, which are plain strings. Mining
    them is deliberate. The alternative -- asking the model to state what it did -- is the thing that
    already fails: final answers say "results have been saved to the output directory" and the next
    turn then has to guess which directory and which tool, which is exactly the 116-second ``/tmp``
    search that motivated session memory in the first place.

    Order is preserved and duplicates collapse, so a tool called four times in a retry loop appears
    once, at the position it was first reached. Only the *last* few artifacts are kept: a pipeline
    writes intermediates first and the result last, and it is the result a follow-up asks about.

    Never raises and never returns non-ASCII: this feeds a prompt, where a stray bullet character
    has historically reached generated code and cost a whole retry loop to SyntaxError.
    """
    text = _log_text(log)
    if not text:
        return ""

    tools = _tools_in(text)
    artifacts = _ordered_unique(_ARTIFACT_RE.findall(_log_text(log, observations_only=True)))

    parts = []
    if tools:
        parts.append("ran " + ", ".join(tools[:_MAX_DIGEST_TOOLS]))
    if artifacts:
        parts.append("wrote " + ", ".join(artifacts[-_MAX_DIGEST_ARTIFACTS:]))
    if not parts:
        return ""
    return _clip(_ascii("; ".join(parts)), _MAX_TRAJECTORY_CHARS, keep="tail")


def summarize_tools(log) -> str:
    """Only the tool half of :func:`summarize_trajectory` -- ``"ran a, b"``, or ``""``.

    The full digest is clipped from the *tail*, because the turn after this one cares most about the
    newest artifacts. That is the right trade for a prompt and the wrong one for a label: six
    absolute output paths are longer than the whole budget, so the tool names at the head are the
    first thing to fall off, and what survives begins mid-word.

    A produced dataset wants the other half. Every harvested file already carries its own path, its
    own name and its own size, so repeating six paths on each of twenty-five sibling records says
    nothing that is not already on the record -- while "ran deepst_identify_domains" is the one fact
    about it that is nowhere else.
    """
    text = _log_text(log)
    if not text:
        return ""
    tools = _tools_in(text)
    if not tools:
        return ""
    return _clip(_ascii("ran " + ", ".join(tools[:_MAX_DIGEST_TOOLS])), _MAX_TRAJECTORY_CHARS)


def _log_text(log, *, observations_only: bool = False) -> str:
    """``agent.log`` flattened to one searchable string, without the prompt. Never raises.

    The prompt is a Human message that is not an observation, and on a portal turn it is the
    ENRICHED prompt: the tool-recommendation block, the post-analysis block, the input paths and the
    previous turn's recap. Mined with the rest, a turn that answered in words was recorded as having
    run the tools it had only been offered and written the files it had only been given (hunt
    2026-09-30, u15-validation-2). ``observations_only`` keeps just what the tools said back -- the
    only place a file the turn WROTE is reported; the model's own text names its inputs too.
    """
    try:
        kept = []
        for entry in log or []:
            if not isinstance(entry, str):
                continue
            stripped = entry.lstrip()
            head, _, rest = stripped.partition("\n") if stripped.startswith("=") else ("", "", stripped)
            body = rest.lstrip()
            # The loop appends an observation as an AI message whose content IS the tag; pretty_print
            # puts a "Human Message" or "Ai Message" banner above whichever it is.
            observation = body.startswith("<observation>") or "Tool Message" in head
            if "Human Message" in head and not observation:
                continue
            if observations_only and head and not observation:
                continue
            kept.append(entry)
        return "\n".join(kept)
    except Exception:
        return ""


#: Where a ``run_*`` name means a call: inside code the loop runs. In prose it is a plan or an offer.
_CODE_REGION_RE = re.compile(r"<execute>(.*?)(?:</execute>|$)|```[\w-]*\n(.*?)(?:```|$)", re.S)


def _tools_in(text: str) -> list[str]:
    """The tools a step log shows being called, in the order they were first reached."""
    code = "\n".join(a or b for a, b in _CODE_REGION_RE.findall(text))
    tools = _ordered_unique(_TOOL_CALL_RE.findall(text) + _RUN_TOOL_RE.findall(code))
    # Our own step-log furniture, not something the agent chose to call.
    return [t for t in tools if t.lower() not in _NOT_A_TOOL]


_NOT_A_TOOL = frozenset({"none", "null", "message", "human", "ai", "system"})


def _ordered_unique(items) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip().rstrip(".,;:)]}\"'")
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def record_turn(turns: list, question: str, answer, trajectory: str = "") -> None:
    """Append a completed turn, in place, and keep the stored history bounded.

    A turn with no answer is not recorded. An interrupted or crashed run has nothing to carry, and
    listing it would invite the next turn to reason about an analysis that never finished.

    Stored as ``(question, answer, trajectory)``. Readers index defensively rather than unpack,
    because 2-tuples remain valid input: an agent restored from a pickle written before the digest
    existed carries them, and so does any caller that builds a history by hand.
    """
    if not isinstance(answer, str) or not answer.strip():
        return
    if not isinstance(question, str) or not question.strip():
        return
    turns.append((question, answer, trajectory if isinstance(trajectory, str) else ""))
    # Twice the recap width: enough that a couple of dropped/blank turns do not empty the recap,
    # small enough that a day-long session cannot grow unboundedly in memory.
    excess = len(turns) - MAX_RECAP_TURNS * 2
    if excess > 0:
        del turns[:excess]


def build_conversation_recap(turns) -> str:
    """Render the last few turns as a prompt preamble, or ``""`` when there is nothing to say.

    Returning the empty string for an empty history is what keeps the disabled and first-turn paths
    byte-identical to the behaviour before this existed.
    """
    if not turns:
        return ""
    recent = list(turns)[-MAX_RECAP_TURNS:]
    lines = [
        "EARLIER IN THIS SESSION (for context only -- do not repeat this work unless the new request asks for it):",
    ]
    for i, turn in enumerate(recent, start=1):
        # Indexing, not unpacking: a 2-tuple is still valid history (see record_turn), and a turn
        # that arrived from an older pickle must not raise ValueError in the middle of a live turn.
        question = turn[0] if len(turn) > 0 else ""
        answer = turn[1] if len(turn) > 1 else ""
        trajectory = turn[2] if len(turn) > 2 else ""
        lines.append(f"[turn {i}] I asked: {_clip(_ascii(str(question)), _MAX_QUESTION_CHARS)}")
        if trajectory:
            # Before the answer, because it is what the answer is *about*: the model reads "you ran
            # run_card and wrote /work/card_output/proportions.csv" and then reads a summary that
            # says "the deconvolution is complete", instead of the summary alone.
            lines.append(f"[turn {i}] You then: {_clip(_ascii(str(trajectory)), _MAX_TRAJECTORY_CHARS, keep='tail')}")
        # Keep the tail of the answer: the conclusion, the output paths and the keys live at the end,
        # while the opening is usually reasoning scaffolding.
        lines.append(f"[turn {i}] You answered: {_clip(_ascii(str(answer)), _MAX_ANSWER_CHARS, keep='tail')}")
    lines.append(
        "Reuse the files, paths, tools and parameters named above instead of searching for them or "
        "asking me to repeat them. NOW ANSWER THIS:"
    )
    return "\n".join(lines) + "\n\n"
