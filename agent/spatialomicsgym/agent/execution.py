"""Execution and workflow configuration for the STCoscientist agent."""

import glob
import inspect
import json
import re
import time
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

# The <execute>/fence reader moved to spatialomicsgym.action so readers OUTSIDE the loop (the
# HuggingFace corpus builder) can ask it the same questions without paying ~0.8s of langchain/langgraph
# and loading the user's .env. Re-exported here under the original names: this module is where the
# parsing tests and every in-repo caller already import them from, so F401 is expected on the ones this
# module does not itself call -- they are deliberate re-exports, not dead imports.
from spatialomicsgym.action import (  # noqa: F401
    _CODE_FENCE_RE,
    _EXECUTE_TAG_RE,
    _LANGUAGE_MARKERS,
    _NESTED_EXECUTE_OPEN_RE,
    _is_language_marked,
    _is_prose_mention,
    _reanchor_prose_mention,
    answer_tag_match,
    classify_and_clean_code,
    dropped_execute_blocks,
    extract_final_answer,
    extract_runnable_code,
    final_answer_span,
    strip_extracted_blocks,
    strip_premature_solution,
)
from spatialomicsgym.agent.prompt_builder import generate_system_prompt
from spatialomicsgym.agent.tool_call_memo import arm_tool_call_memo, disarm_tool_call_memo, release_tool_call_memo
from spatialomicsgym.agent.usage import TurnUsage
from spatialomicsgym.answer import in_inline_code
from spatialomicsgym.know_how.enrolment import (
    baseline_documents,
    enrolment_mode,
    index_documents,
    pack_budget,
    packs_enabled,
)
from spatialomicsgym.llm import message_to_text
from spatialomicsgym.provider_backoff import invoke_with_backoff
from spatialomicsgym.tool.support_tools import OUTPUT_CAP_MARKER, run_python_repl
from spatialomicsgym.utils import (
    inject_custom_functions_to_repl,
    parse_tool_calls_from_code,
    parse_tool_calls_with_modules,
    run_bash_script,
    run_r_code,
    run_with_timeout,
)
from spatialomicsgym.utils.execution import OBSERVATION_CHARS, TIMEOUT_KEPT_NOTE

# --------------------------------------------------------------------------- #
# Parsing helpers — module-level + pure so the ReAct router (generate) and the code extractor (execute)
# share ONE source of truth (they used to disagree and loop the graph to recursion_limit=1000), and so
# the parsing is unit-testable (the graph nodes are closures with no seam).
#
# The <execute>/fence half of that source of truth now lives in spatialomicsgym.action and is
# re-exported above under its original names: readers outside the ReAct loop (the HuggingFace corpus
# builder) need the same answers, and importing THIS module to get them costs ~0.8s of
# langchain/langgraph and loads the user's .env. Everything below is loop-specific.
# --------------------------------------------------------------------------- #
# Whether a turn is answering is ``answer_tag_match`` (spatialomicsgym.action), which reads
# spatialomicsgym.answer's SOLUTION_TAG_RE -- the one definition every reader of a solution block
# shares (the dataset builder used to carry a case-sensitive copy of it).
_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
# <observation> is emitted by the SYSTEM (execute()) wrapping REAL run output. A model must never produce
# one; when it does it is fabricating a result. Used to scrub such hallucinations out of a model turn.
#: Both spellings, because the two halves of the system disagree on the tag's name: the runtime
#: (execute()) wraps real output in <observation>, while the inherited ReAct scaffold text tells the
#: model results arrive "within <observe></observe>" — so a transcript-hallucinating model fabricates
#: exactly the spelling this regex used to miss (live round 4: a conversion that failed was reported
#: as verified-successful from two fabricated <observe> blocks the scrub left standing).
#:
#: A scrubbed span may not cross an action tag. Non-greedy alone is not enough under DOTALL: a
#: prose mention of `<observe>` paired with a later closing tag and deleted the `<execute>` opener
#: between them (2 archived trials; one then ended with no answer at all).
_OBSERVATION_TAG_RE = re.compile(
    r"<(observation|observe)>(?:(?!</?(?:execute|solution)\b).)*?</\1>", re.DOTALL | re.IGNORECASE
)

# The exact reprompt sent when a reply carries no actionable tag. Counting THESE (a HumanMessage) is how
# the 2-strike give-up fires — kept as one constant so the counter and the emitted message can't drift.
_NO_TAGS_REPROMPT = (
    "Each response must include thinking process followed by either <execute> or <solution> tag. "
    "But there are no tags in the current response. Please follow the instruction, fix and regenerate "
    "the response again."
)

# Sent INSTEAD of _NO_TAGS_REPROMPT when the provider reported that the reply hit the output-token cap.
# Telling a model whose answer was severed that it "forgot the tags" is both false and unhelpful — it
# rewrites the same over-long turn and is cut off again. Counted by the same 2-strike guard.
_TRUNCATED_REPROMPT = (
    "Your last response was cut off by the output token limit before it finished — it is incomplete, "
    "not incorrectly formatted. Send that step again in a shorter form: drop the commentary, and if the "
    "code was long, do only the next step this turn so the response ends with a complete "
    "<execute>...</execute> block."
)

_THINK_ONLY_REPROMPT = (
    "Your last response contained a <think> block but no <execute> or <solution> tag. Thinking alone "
    "does not advance the task. You must now act: emit an <execute> block to run code, or a <solution> "
    "block if the task is complete."
)

# Sent when a reply is a plan-only "solution": an unchecked to-do list plus a <solution> that only
# PROMISES future work, with no <execute> and nothing run yet. Kept as one constant so the counter that
# bounds the nudge (count_plan_only_solution_reprompts) and the emitted text can never drift.
_PLAN_ONLY_SOLUTION_REPROMPT = (
    "Your last response laid out a numbered plan and then a <solution> that only DESCRIBES what you will "
    "do next — you did not actually run anything. Announcing intent is not a result. Emit an <execute> "
    "block now that calls the required tool(s) on the real data, wait for the observation, and only then "
    "write a <solution> reporting the ACTUAL numbers. Do not just restate the plan."
)

# A line that is an UNCHECKED to-do item: "1. [ ] ...". A genuine answer — conceptual prose or a completed
# tool result — does not enumerate an empty-checkbox plan, so requiring >=2 of these (plus no prior
# observation) is an airtight signature of "planned but never executed" that cannot match a real solution.
_UNCHECKED_PLAN_ITEM_RE = re.compile(r"(?m)^\s*\d+\.\s*\[\s*\]")

# What each provider calls "I stopped because I ran out of output tokens". Compared lowercased, so
# Gemini's ``MAX_TOKENS`` and OpenAI's ``length`` both land here.
_TRUNCATION_STOP_REASONS = frozenset({"length", "max_tokens", "max_output_tokens", "model_length"})

# Where each provider hangs that reason off ``response_metadata``: OpenAI / Azure / Gemini use
# ``finish_reason``, Anthropic ``stop_reason``, Bedrock's converse API camel-cases it, Ollama uses
# ``done_reason``. All are checked because llm.py builds clients for every one of them.
_STOP_REASON_KEYS = ("finish_reason", "stop_reason", "stopReason", "done_reason")


def completion_was_truncated(response) -> bool:
    """True only when the provider *says* generation stopped at the output-token cap.

    Anything else — no metadata, a wrapper that reports nothing, a reason we do not recognise — is
    False, so a client that says nothing keeps exactly today's behaviour. The asymmetry is
    deliberate: a false positive would stop restoring the closing tag that the configured stop
    sequence legitimately consumed, and would do it on every healthy turn, whereas a false negative
    only leaves today's bug in place for one provider.
    """
    meta = getattr(response, "response_metadata", None)
    if not isinstance(meta, dict):
        return False
    if any(
        isinstance(meta.get(key), str) and meta[key].strip().lower() in _TRUNCATION_STOP_REASONS
        for key in _STOP_REASON_KEYS
    ):
        return True
    # The OpenAI/Azure RESPONSES API -- the shipped default, azure-gpt-5.5 -- sets no finish_reason at
    # all: langchain_openai copies `status` and `incomplete_details` instead, and a cut-off answer is
    # `status="incomplete"` with `incomplete_details.reason="max_output_tokens"`. Unrecognised here,
    # a severed <execute> had its closing tag restored and ran (u12-react-6). Only the output-token
    # reason counts: an incomplete answer stopped by a content filter was not truncated by length.
    details = meta.get("incomplete_details")
    return (
        str(meta.get("status") or "").strip().lower() == "incomplete"
        and isinstance(details, dict)
        and str(details.get("reason") or "").strip().lower() in _TRUNCATION_STOP_REASONS
    )


def _usage_of(agent: Any) -> TurnUsage:
    """The turn's usage ledger, created on first use and hung off the agent.

    Attached lazily rather than in ``__init__`` because every constructor of every agent-shaped
    object in the tests would otherwise have to grow a field, and a telemetry change that makes
    fixtures fail is a telemetry change nobody keeps. ``getattr``/``setattr`` also means an object
    that forbids new attributes degrades to a throwaway ledger instead of raising on the live path.
    """
    existing = getattr(agent, "_turn_usage", None)
    if isinstance(existing, TurnUsage):
        return existing
    fresh = TurnUsage()
    try:
        agent._turn_usage = fresh
    except Exception:
        pass  # a frozen or slotted object still gets a working (if unread) ledger
    return fresh


#: The loop's stop sequences -- the list ``stcoscientist.py`` hands ``get_llm``.
STOP_SEQUENCES: tuple[str, ...] = ("</execute>", "</solution>")


def cut_at_stop_sequence(msg: str, stops: tuple[str, ...] = STOP_SEQUENCES) -> str:
    """The reply as a provider that honours ``stop`` would have returned it: ended at its first stop.

    The ReAct loop is built on the model stopping at its first ``</execute>`` or ``</solution>``:
    one action per turn, and the next turn written against the real observation. The OpenAI
    Responses family (gpt-5.x) and the o-series reject ``stop``, so ``llm.py`` drops it from their
    requests -- and nothing put it back. Measured over the recorded SpatialBench trials, 346 of
    3,156 model turns carried text past their first stop: an invented observation, reasoning about
    it, further cells written in advance (223 turns) and premature answers (68). ``_runnable_code``
    then ran every block of the turn as ONE cell, so code written against an invented result --
    including writes of the scored answer file -- ran before any real output existed.

    Enforced here for every provider: where the provider already stopped, the reply contains no
    stop string and this changes nothing. The stop string itself is kept, so the block stays closed.

    A stop string inside an inline code span is prose, not a tag: an odd number of backticks
    between the start of its line and it. Two recorded turns needed that -- one wrote
    `` `</execute>` ``, and one quoted the loop's own fallback, "`... a single <execute>...</execute>
    block ...`", which a plain first-occurrence cut would have ended the turn on.
    """
    first = first_stop_end(msg, stops=stops)
    return msg if first is None else msg[:first]


def first_stop_end(msg: str, after: int = 0, stops: tuple[str, ...] = STOP_SEQUENCES) -> int | None:
    """Where the reply's first stop sequence ends, or None: the rule :func:`cut_at_stop_sequence` cuts by.

    ``after`` is for a reader that grows the reply a piece at a time and has already found no stop in
    its first ``after`` characters: only a stop that ends past it is looked for, so each piece costs
    its own length, not the reply's. The answer is the one the whole string gives, because the rule
    reads only what comes BEFORE a stop -- whether it sits in an inline code span is decided by the
    backticks between its line's start and it -- so a prefix that holds a stop already holds the
    stop the whole reply would be cut at.
    """
    first: int | None = None
    for stop in stops:
        start = max(0, after - len(stop) + 1)
        while (i := msg.find(stop, start)) >= 0:
            end = i + len(stop)
            if in_inline_code(msg, i):
                start = end
                continue
            if first is None or end < first:
                first = end
            break
    return first


class _FirstStopReader:
    """``llm`` for ``invoke_with_backoff``, read as a stream to its first stop sequence.

    ``invoke`` is the one method the backoff calls, so a transient failure is retried as the same
    call exactly as before. ``stopped`` says how the last read ended: closed at a stop, or read to
    its end.
    """

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.stopped = False

    def invoke(self, messages: Any) -> Any:
        from spatialomicsgym.responses_stream import read_to_first_stop

        self.stopped = False
        reply, self.stopped = read_to_first_stop(self.llm, messages, first_stop_end)
        return reply


def close_dangling_tags(msg: str, *, truncated: bool) -> str:
    """Give back the closing tag the stop sequence ate — unless the turn was severed instead.

    The loop runs with ``stop_sequences=["</execute>", "</solution>"]`` (``stcoscientist.py``), so a
    *complete* turn arrives with its closing tag stripped and must be handed one back. A turn that
    hit the token cap is textually identical and semantically the opposite: nothing closed it
    because nothing finished it. Only the provider's stop reason separates the two.

    Fabricating ``</execute>`` on the severed one is the damaging case. The truncated prefix is
    typically still valid python — read the h5ad, filter it, and stop short of the write — so it
    runs, half-completes, and the agent reads the missing output back as a finished result with
    nothing in the transcript saying the code was incomplete. Left open it parses as no action at
    all, and the model is told it was cut off.

    ``<solution>`` is deliberately still closed even when truncated: a visibly half-written answer
    is better than reprompting a model that is already truncating, which can truncate again, spend
    the strike guard, and end the run with no answer at all.
    """
    # Case-insensitive, because every reader of these tags is. ``_EXECUTE_TAG_RE``,
    # ``SOLUTION_TAG_RE`` and ``_THINK_TAG_RE`` are all re.IGNORECASE, and so was the execute node's
    # own auto-close (removed: execute() now reads the message as this function leaves it) -- only
    # this function tested with ``in``, which is not.
    #
    # THE BUG THIS EXISTS FOR. The loop runs with ``stop_sequences=["</execute>", "</solution>"]``,
    # so a *complete* turn always arrives missing its closer and always comes through here. A model
    # that wrote ``<Execute>`` therefore had its closer withheld, the router found no code, and a
    # turn carrying real code -- or the final answer -- was classified as "no tags". The model was
    # then sent ``_NO_TAGS_REPROMPT``, which is also untrue about what it just did, and two such
    # turns end the run through the give-up guard. Measured: ``<Execute>\nimport scanpy as sc\n``
    # was left unclosed and the router returned None, while the downstream auto-close would have
    # matched it perfectly well.
    #
    # And only a tag written AS a tag, by the rule cut_at_stop_sequence already applies: a tag inside
    # an inline code span is prose. A model that paraphrased the prompt's gate -- "a final
    # `<solution>` must be preceded by an observation" -- and then wrote a real <execute> had
    # </solution> appended here, so the turn carried execute+solution and the premature-solution
    # strip erased everything from the mention to the <execute>: 31 recorded turns in 29 trials carry
    # the scar, "a final `<execute>" with the code straight after the backtick (measured 2026-09-25
    # over the archive and E-01..E-03). A quoted `<execute>` was closed the same way: one archived
    # turn (v6 cn7 r3) ran "` or `<solution>` tags." as python before the answer it already carried.
    # And a quoted `</solution>` inside a real answer the stop sequence cut short kept the answer's
    # own closer out.
    if _written_as_tag(msg, "<execute>") and not _written_as_tag(msg, "</execute>") and not truncated:
        msg += "</execute>"
    if _written_as_tag(msg, "<solution>") and not _written_as_tag(msg, "</solution>"):
        msg += "</solution>"
    if _written_as_tag(msg, "<think>") and not _written_as_tag(msg, "</think>"):
        msg += "</think>"
    return msg


def _written_as_tag(msg: str, tag: str) -> bool:
    """Whether ``tag`` occurs in ``msg`` outside an inline code span, in any case."""
    return any(not in_inline_code(msg, m.start()) for m in re.finditer(re.escape(tag), msg, re.IGNORECASE))


def clip_observation(result: str, limit: int = OBSERVATION_CHARS, *, kind: str = "") -> str:
    """Bound an execute observation for context, keeping its beginning AND its end.

    The head-only clip this replaces cost a live run four hours. A cell2location call on the full
    lymph-node pair succeeded after 4h15m, but everything that proved it -- the
    ``Exported CSV exists: True`` line, the post-analysis summary -- sat past the 10K mark, so the
    model re-dispatched the byte-identical 4-hour analysis just to look for a confirmation the clip
    was never going to show it, and finished with "I cannot truthfully claim those later steps
    completed". The export had worked all along; the file was on disk the whole time.

    Long tool output puts its decision-relevant lines at the END -- final summaries, export
    confirmations, tracebacks -- so the tail is the part a model deciding "did this succeed" needs,
    and it gets the larger share. The head keeps the launch banner and early errors. The marker
    names what was elided and says outright that the code already ran and that re-running is not
    the way to see more; the files the run wrote are.

    **Unless there is no end to show.** ``_BoundedStringIO`` caps the REPL's stdout at 2 MB at the
    source, and past that the end of the output was never captured -- so "its beginning and end
    are shown below" and "this is only a display limit" both become false, and "read the files the
    run wrote" points at something that does not contain the missing text either. When
    ``support_tools`` says it dropped characters, this says the other thing: the tail is missing,
    and printing less and running again is the way to see it. The two headers must not contradict
    each other in the same observation, which is what they did before.

    **And files are not the only place the middle is.** ``kind`` is the cell's language, as
    ``classify_and_clean_code`` names it. A python cell's variables outlive it -- in this process and
    in the worker alike -- so the value it printed from is usually one ``print`` of a slice or a key
    away, and that is no re-run. The header said only "read the files or logs the run wrote", which
    for a cell that wrote nothing named a place the middle is not, and left the one place it is
    unsaid. An R or shell cell keeps nothing between cells, so for any other kind the header is
    unchanged.
    """
    if len(result) <= limit:
        return result
    head, tail = (limit * 2) // 5, limit - (limit * 2) // 5
    omitted = len(result) - head - tail
    if OUTPUT_CAP_MARKER in result:
        header = (
            "The output is too long to be added to context, AND it was cut off at the source "
            f"before that: {omitted} characters are elided from the middle here, and the note at "
            "the end says how many more were never captured at all. What you see below is the "
            "beginning and the middle -- not the end. Re-running unchanged would lose the end "
            "again; print less (summarise, slice, or write the result to a file and print its "
            "path) if you need it.\n"
        )
    else:
        if kind == "python":
            where = (
                "print a slice or a key of a variable the cell bound -- this Python session still "
                "holds them, so that is not a re-run -- or read a file or log the run wrote.\n"
            )
        else:
            where = "read the files or logs the run wrote.\n"
        header = (
            "The output is too long to be added to context: its beginning and end are shown "
            f"below, with {omitted} characters elided from the middle. The code already ran to "
            "completion; this is only a display limit. Do not re-run the analysis to see more -- "
            "it would run again in full. To inspect the elided part, " + where
        )
    return header + result[:head] + f"\n... [{omitted} characters elided] ...\n" + result[-tail:]


def count_no_tags_reprompts(messages) -> int:
    """How many no-actionable-tag reprompts we've ALREADY sent (a ``HumanMessage`` carrying
    ``_NO_TAGS_REPROMPT`` **or** ``_TRUNCATED_REPROMPT``). The old code counted ``AIMessage``s containing
    capitalized ``"There are no tags"`` — a string that never appears — so the count was always 0 and the
    2-strike give-up was dead code, letting a tag-less model loop to the recursion limit (~1000 paid LLM
    calls) then crash.

    Both reprompts are counted together because they are the same strike: the branch picks one *or* the
    other by cause, so counting only the first would let a persistently-truncating model loop forever."""
    return sum(
        1
        for m in messages
        if isinstance(m, HumanMessage)
        and (_NO_TAGS_REPROMPT in (m.content or "") or _TRUNCATED_REPROMPT in (m.content or ""))
    )


def count_think_only_reprompts(messages) -> int:
    """How many think-only reprompts we've ALREADY sent (a ``HumanMessage`` carrying
    ``_THINK_ONLY_REPROMPT``). A response that parses as a ``<think>`` block with no ``<execute>`` /
    ``<solution>`` used to route straight back to ``generate`` with no counter — so a model that keeps
    reasoning without ever acting looped ``generate``→``generate`` to the recursion limit (~1000 paid
    LLM calls) then crashed. This mirrors the no-tags 2-strike give-up for that path."""
    return sum(1 for m in messages if isinstance(m, HumanMessage) and _THINK_ONLY_REPROMPT in (m.content or ""))


# ``_EXEC_ERROR_RE`` and ``_execute_error_signature`` live in ``spatialomicsgym.action`` (langchain-free,
# so the corpus miner reads observations with the loop's own judgement) and are re-exported here.
from spatialomicsgym.action import (  # noqa: F401 -- the public pair is re-exported, as action.py promises
    _EXEC_ERROR_RE,
    EXEC_ERROR_RE,
    WARNINGS_HEADER,
    _execute_error_signature,
    execute_error_signature,
)

#: The notes the three give-up guards in ``generate`` write onto ``AgentState.degraded``. Module
#: constants so ``agent/rescue.py`` (which rescues exactly these three) and the drift test import
#: them rather than restating the sentences.
#: The tags a ReAct transcript is parsed with. Text a TOOL wrote must not be able to spell any of
#: them, because that text is interpolated into an ``<observation>`` and appended as an
#: ``AIMessage`` -- the assistant role, the highest-trust position in the transcript.
#:
#: Measured on this tree, all three reachable from one string of worker stdout:
#:
#: * ``</observation>`` splits one observation into three, and the model reads the LAST as the
#:   result of its own cell -- so a tool that printed a closing tag followed by
#:   "The deconvolution finished successfully" had that read as its own output.
#: * ``<solution>…</solution>`` is extracted by ``extract_final_answer``. ``_is_agent_answer``
#:   accepts any non-``HumanMessage``, so when a turn ends on an observation -- a spent step
#:   budget, a give-up -- ``go()`` returns that observation as the answer, and the benchmark
#:   harnesses parse ``<solution>(.*?)</solution>`` straight out of it.
#:   Text a tool wrote became the scored answer.
#: * ``<execute>…</execute>`` is extracted by ``extract_runnable_code``.
#:
#: ``observe`` is here for the reason ``_OBSERVATION_TAG_RE`` carries it: the scaffold text tells
#: the model results arrive within ``<observe></observe>``, so both spellings are live.
#:
#: Rewritten to a bracket form rather than HTML-escaped: the reader is a language model, and
#: ``[/observation]`` reads as what it is where ``&lt;/observation&gt;`` reads as noise. This is
#: the technique ``tools_user/memory_manager.py:139`` already uses on its own ``<prior_attempt>``
#: tags, for this exact reason.
_TAG_IN_TOOL_OUTPUT_RE = re.compile(r"<\s*(/?)\s*(observation|observe|execute|solution|think)\s*>", re.IGNORECASE)


def as_tool_data(text: Any) -> str:
    """Text a tool produced, made safe to place inside the transcript.

    R4.4 in ``SECURITY_RULES.md``: third-party text is data, and the code must say so where it
    ENTERS. This is that place for worker stdout. Nothing is removed -- every tag keeps its
    spelling in a bracket form, so a tool whose help output legitimately mentions ``<execute>``
    still reads correctly and simply cannot be parsed as one.
    """
    return _TAG_IN_TOOL_OUTPUT_RE.sub(lambda m: f"[{m.group(1)}{m.group(2).lower()}]", str(text or ""))


GIVEUP_REPEATED_ERROR = "the same execution error repeated after several attempts; the loop stopped early"
GIVEUP_THINK_ONLY = "reasoning was produced but no action followed; the loop stopped early"
GIVEUP_NO_ACTION = "no complete action could be parsed from the responses; the loop stopped early"
GIVEUP_TRUNCATED = "the responses kept being cut off by the output token limit; the loop stopped early"
#: The rotating-error cap below fired: every recent execution failed, and no one error ran long
#: enough in a row to trip the identical-error guard -- which is not the same as each one differing.
GIVEUP_EVERY_ATTEMPT_FAILED = "every recent execution failed, with changing errors; the loop stopped early"
#: A fourth, and the only one that fires on a turn that LOOKS finished. ``close_dangling_tags``
#: closes a ``<solution>`` the output-token limit severed -- deliberately, because a visibly
#: half-written answer beats none -- and ``truncated`` was then discarded on the answer path. So the
#: provider reported MAX_TOKENS, the answer was cut mid-sentence, the tag was closed for it, and the
#: turn ended reporting a clean finish: ``giveup_note()`` answered None, ``last_turn_degraded``
#: stayed None, the CLI exited 0 and a benchmark scored the half answer as a whole one. The fact
#: needed to say otherwise was in hand one branch earlier. Only ``degraded`` moves; the answer text
#: is untouched.
GIVEUP_TRUNCATED_SOLUTION = "the final answer was cut off by the output token limit; it is incomplete"
#: The give-ups a rescue round may follow. Truncation is not one: a model that cannot fit an action
#: in its output budget will not fit one on a second try either, and that is the audit's territory.
RESCUABLE_GIVEUPS = frozenset({GIVEUP_REPEATED_ERROR, GIVEUP_EVERY_ATTEMPT_FAILED, GIVEUP_THINK_ONLY, GIVEUP_NO_ACTION})


def _last_execute_error_text(messages, limit: int = 600) -> str:
    """The most recent execute-error observation, trimmed, for a termination message to quote.

    Exists because the give-up text used to name a cause it could not know. Returns "" when no
    error observation can be found, and the caller then says nothing rather than guessing.
    """
    for msg in reversed(list(messages or [])):
        content = getattr(msg, "content", None)
        if not isinstance(content, str):
            continue
        match = _EXEC_ERROR_RE.search(content)
        if not match:
            continue
        text = content[match.start() : match.start() + limit].strip()
        return text if len(text) < limit else text + " ..."
    return ""


def _is_execution_observation(message) -> bool:
    """Whether ``message`` is an observation ``execute()`` appended -- a run, not text quoting one.

    The counters below end a turn on a run of failed executions, so what they count has to be
    executions. They counted any message containing ``<observation>``, and the rescue round opens on
    a message that is not one and carries the tag: its prompt (``agent/rescue.py``) is a
    ``HumanMessage`` quoting the last error verbatim, and ``_last_execute_error_text`` keeps the tag
    whenever the cell raised before printing anything (``Error: ...`` straight after it, which is
    what the REPL returns for an exception with no prior stdout). Replayed through the real graph,
    the prompt read as one more failure: a rescue round's rotating cap fired after 7 executions and
    said 8, and on an identical error longer than the 200 characters a signature keeps, the
    same-error guard fired after 3 of its 4.

    ``execute()`` appends every observation, the no-code fallback included, as the assistant's
    message opening with the tag. A user's message is never one, whatever it pastes; and a model
    turn that merely mentions the tag -- an unclosed ``<observation>`` in its code survives the
    scrub, which removes pairs -- is not one either, where it used to read as a clean run and reset
    the streak.
    """
    if isinstance(message, HumanMessage):
        return False
    content = getattr(message, "content", "")
    return isinstance(content, str) and content.lstrip().startswith("<observation>")


def count_trailing_execute_errors(messages) -> int:
    """How many of the MOST RECENT consecutive execution observations report the SAME execution error.

    A model that keeps re-emitting code that fails the SAME way -- classically a stray non-ASCII
    character (an em-dash / curly quote) in a code comment, which raises ``SyntaxError`` and which the
    model then re-pastes verbatim -- would otherwise loop generate<->execute to ``recursion_limit=1000``
    (~1000 paid LLM calls) before crashing. Count only the unbroken tail of IDENTICAL error observations:
    a CLEAN (successful) observation OR a DIFFERENT error (the model is making progress, not stuck)
    resets the streak. So a successful observation that merely contains an error-word is never miscounted,
    and a genuine converging debug session (different error each try) is never cut short."""
    sig = None
    n = 0
    for m in reversed(messages):
        if not _is_execution_observation(m):
            continue  # a code turn / reprompt between observations -- not an execute result
        content = m.content
        if not _EXEC_ERROR_RE.search(content):
            break  # a clean (successful) observation ends the streak
        this_sig = _execute_error_signature(content)
        if sig is None or this_sig == sig:
            sig = this_sig
            n += 1
        else:
            break  # a DIFFERENT error -> the model is making progress, not stuck
    return n


#: Consecutive failed executions, of ANY error signature, after which the loop stops.
#:
#: ``count_trailing_execute_errors`` counts only IDENTICAL errors, so a converging debug session is
#: never cut -- and so a model cycling through different errors was bounded only by
#: ``recursion_limit=1000``. Measured over the 456 archived SpatialBench trials: the longest
#: all-error run with at least two distinct errors is 5, and a run of 4 still went on to pass. 8 is
#: a safety net that has never fired on a recorded trial; the replay sweep pins that it still does not.
ROTATING_ERROR_LIMIT = 8


def _trailing_failed_observations(messages) -> list[str]:
    """The unbroken run of failed execution observations at the end of ``messages``, newest first."""
    failed: list[str] = []
    for m in reversed(messages):
        if not _is_execution_observation(m):
            continue
        if not _EXEC_ERROR_RE.search(m.content):
            break
        failed.append(m.content)
    return failed


def count_trailing_failed_executions(messages) -> int:
    """How many of the most recent consecutive execution observations report an execution error,
    whatever the error. A clean observation ends the run."""
    return len(_trailing_failed_observations(messages))


def _rotating_error_giveup_text(messages) -> str:
    """What the rotating cap leaves as the turn's last message, saying only what the run shows.

    The cap needs no error to differ from the one before it. It fires on any run of
    ``ROTATING_ERROR_LIMIT`` failures the identical-error guard let through, which means only that
    no one error ran 4 times in a row -- yet it said every failure had "a different error".
    Replayed through the real graph: a model alternating between ``KeyError: 'leiden'`` and
    ``KeyError: 'spatial'`` was stopped at 8 under that sentence, and ``go()`` returns this message
    as the answer when no rescue replaces it. "The same error" is judged by
    ``_execute_error_signature``, the guard's own test, so the count cannot disagree with it.
    """
    failed = _trailing_failed_observations(messages)
    distinct = len({_execute_error_signature(content) for content in failed})
    if distinct == len(failed):
        spread = "each with a different error"
    else:
        spread = f"with {distinct} different error{'s' if distinct != 1 else ''} among them"
    last_error = _last_execute_error_text(messages)
    detail = f" The most recent error was:\n{last_error}" if last_error else ""
    return (
        f"Execution terminated: the last {len(failed)} executions all failed, {spread}, so the loop "
        "stopped rather than keep rotating." + detail
    )


def count_plan_only_solution_reprompts(messages) -> int:
    """How many plan-only-solution reprompts we've ALREADY sent (a ``HumanMessage`` carrying
    ``_PLAN_ONLY_SOLUTION_REPROMPT``). A model that emits an unchecked to-do plan then a promise-only
    ``<solution>`` with no ``<execute>`` would otherwise terminate the ReAct loop having run no tool at
    all (observed live: a weaker model narrated "Next I will run X" and stopped). This bounds the rescue
    nudge to fire at most once, so a determined narrate-only model is still accepted rather than looped."""
    return sum(1 for m in messages if isinstance(m, HumanMessage) and _PLAN_ONLY_SOLUTION_REPROMPT in (m.content or ""))


def is_plan_only_solution_bail(msg: str | None, messages) -> bool:
    """True when a ``<solution>``-bearing reply is really a *plan-only bail* that should be nudged instead
    of accepted as the final answer. All three must hold:

      * nothing has executed yet in this conversation (no message contains ``<observation>``),
      * the message enumerates a real (>=2-item) UNCHECKED to-do plan (``1. [ ]`` ...), and
      * the one-shot rescue nudge has not already been sent.

    A genuine conceptual answer has no empty-checkbox plan, and a completed tool task has an observation —
    so this can only match a turn that planned work and then emitted a promise-only ``<solution>`` without
    running anything (observed live with a weaker model that narrated "Next I will run X" and stopped).
    Returning False keeps the normal ``<solution>`` -> end behaviour untouched for every real answer."""
    already_executed = any("<observation>" in (getattr(m, "content", "") or "") for m in messages)
    unchecked_items = len(_UNCHECKED_PLAN_ITEM_RE.findall(msg or ""))
    return not already_executed and unchecked_items >= 2 and count_plan_only_solution_reprompts(messages) < 1


# Sent when a <solution> carries an empty answer object. One constant, so the counter that bounds the
# nudge and the emitted text cannot drift. It never suggests shrinking the data: a timed-out method is
# replaced by a cheaper METHOD on the same data, never by a smaller dataset.
_EMPTY_ANSWER_REPROMPT = (
    "Your <solution> contains an empty answer. An empty answer is never correct, and you still have time "
    "budget left. Re-read what your observations already established. If a computation timed out, do not "
    "repeat it unchanged: choose a method that finishes within the per-step time limit on the same data "
    "(for example a registered tool, a sparse or backed computation, or fewer parameter sweeps), run it with "
    "an <execute> block, and then answer with the values your observations support. Never report a value "
    "you did not observe."
)

#: One outer tag around an answer (``<EVAL_ANSWER>{}</EVAL_ANSWER>``); peeled whatever its name, so the
#: loop stays agnostic of any one benchmark's answer tag.
_ONE_WRAPPER_RE = re.compile(r"\A<([A-Za-z_][\w-]*)>(.*)</\1>\Z", re.DOTALL)


def is_empty_answer(msg: str | None) -> bool:
    """True when the final answer is structurally empty: a blank ``<solution>``, or ``{}``, ``[]``,
    ``""`` or ``null`` inside at most two outer tags. A zero, a ``false`` or ``{"count": 0}`` is an answer, not an empty
    one -- that distinction is the whole design."""
    body = extract_final_answer(msg or "")
    if body is None:
        # extract_final_answer skips a blank block by design, but generate() detects a solution with
        # answer_tag_match, which matches one -- so a blank <solution> used to END the turn, cleanly,
        # with no answer. It is the emptiest answer of all.
        return answer_tag_match(msg) is not None and extract_runnable_code(msg or "") is None
    text = body.strip()
    for _ in range(2):
        m = _ONE_WRAPPER_RE.match(text)
        if m is None:
            break
        text = m.group(2).strip()
    if not text:
        return True
    try:
        value = json.loads(text)
    except ValueError:
        return False
    return value is None or (isinstance(value, (dict, list, str)) and len(value) == 0)


def count_empty_answer_reprompts(messages) -> int:
    return sum(1 for m in messages if isinstance(m, HumanMessage) and _EMPTY_ANSWER_REPROMPT in (m.content or ""))


def is_empty_answer_bail(msg: str | None, messages) -> bool:
    """An empty answer is nudged ONCE. Measured over the 456 archived SpatialBench trials: 12 ended on
    an empty answer object and all 12 failed -- 9 on one eval, each right after a cell timed out, at
    turn 4-9 of a budget the median trial uses 0.65% of. A second empty answer is accepted, so a model
    that genuinely cannot answer is never looped."""
    return is_empty_answer(msg) and count_empty_answer_reprompts(messages) < 1


#: C6. The one sentence the pre-answer reprompt opens with -- a constant, so the counter that bounds
#: the check and the emitted text cannot drift (the plan-only nudge's idiom).
_PRE_ANSWER_CHECK_OPENER = "Before this answer is accepted: the automatic check of the output this turn produced"
_VERDICT_RANK = {"ok": 0, "suspicious": 1, "unusable": 2}


def count_pre_answer_checks(messages) -> int:
    return sum(
        1 for m in messages if isinstance(m, HumanMessage) and str(m.content or "").startswith(_PRE_ANSWER_CHECK_OPENER)
    )


def pre_answer_check(agent, messages) -> str | None:
    """C6 (``t4_candidates_verification.md``): the reprompt to send instead of accepting a
    ``<solution>``, or ``None`` to accept it.

    The post-analysis verdict already ran inside scored trials -- AFTER the answer, where it could
    only append a second one (FM-12). This runs the same L1 analysis and L2 review before the answer
    is accepted, and when the worst verdict over this turn's outputs is ``suspicious`` or
    ``unusable`` it hands the model the reasons, once. A second answer is always accepted.

    OFF unless ``pre_answer_verification`` (``SOG_PRE_ANSWER_VERIFICATION``) is set: it changes what
    the model sees before it answers, so it is a Class B change that waits for its own arm. Also off
    wherever the post-analysis layer is (``post_analysis_active``). Never raises -- a check that
    breaks must cost the check, not the turn.
    """
    try:
        from spatialomicsgym.config import default_config

        if not getattr(default_config, "pre_answer_verification", False):
            return None
        if count_pre_answer_checks(messages) >= 1:
            return None
        from spatialomicsgym.postanalysis.next_step import post_analysis_active

        if not post_analysis_active():
            return None
        since = getattr(agent, "_turn_started_at", None)
        roots_of = getattr(agent, "_post_analysis_search_roots", None)
        if since is None or not callable(roots_of):
            return None
        from spatialomicsgym.postanalysis.autorun import analyse_new_outputs
        from spatialomicsgym.postanalysis.review import discover_results_dirs, review_results_dir

        roots = roots_of()
        exclude = tuple(getattr(agent, "_foreign_dirs", None) or ())
        try:
            analyse_new_outputs(roots, since, exclude=exclude)
            worst, reasons = "ok", []
            for directory in discover_results_dirs(roots, since, exclude=exclude):
                review = review_results_dir(directory) or {}
                verdict = str(review.get("verdict") or "ok")
                if _VERDICT_RANK.get(verdict, 0) > _VERDICT_RANK[worst]:
                    worst = verdict
                if verdict != "ok":
                    reasons.extend(str(r) for r in (review.get("reasons") or [])[:3])
        finally:
            # The manifests just written describe this turn's output AS IT IS NOW, and this check
            # exists to make the model change it. Stamped after the writes, so the post-answer
            # review (``STCoscientist._ensure_post_analysis_ran``) can tell them from a manifest
            # the model's own call wrote later, and re-analyse what changed since.
            agent._pre_answer_analysed_at = time.time()
        if worst == "ok":
            return None
        listed = "\n".join(f"- {r}" for r in reasons[:5]) or "- (no reason was given)"
        return (
            f"{_PRE_ANSWER_CHECK_OPENER} says **{worst}**:\n{listed}\n\n"
            "Re-examine the result your answer depends on. If the concern holds, fix it in an <execute> "
            "block and then answer; if it does not apply here, answer again and say in one sentence why."
        )
    except Exception:
        return None


def maybe_resync_user_tools(agent) -> None:
    """After an execute step, re-sync the agent's live tool catalogs if a user-tool create/delete/
    modify changed ``mcp_config_user.yaml`` during the session.

    User-tool CRUD writes files/config but never touches in-memory state, so a trashed tool would
    stay callable, a modified tool would keep its OLD wrapper, and a just-created tool wouldn't be
    callable until restart. Detecting the config mtime change here (the turn right after the agent
    ran the CRUD code) and calling ``agent.reload_user_tools()`` fixes all three without the REPL
    needing an agent handle. GATED behind ``tool_creation_enabled`` -> a strict no-op on the eval
    path, where the user config never changes. Never raises (a resync failure must not break a turn).
    """
    try:
        from spatialomicsgym.config import default_config

        if not default_config.tool_creation_enabled:
            return
        import os

        from spatialomicsgym.agent.mcp_config_merger import resolve_user_config_path

        # Must match the baseline STCoscientist recorded at wire-up (same helper), otherwise the
        # two read different files and this either never fires or fires on every single turn.
        up = resolve_user_config_path()
        mtime = os.path.getmtime(up) if os.path.exists(up) else 0.0
        last = getattr(agent, "_user_config_mtime", None)
        if last is not None and mtime != last:
            agent._user_config_mtime = mtime  # update BEFORE reload so it can't re-trigger itself
            if hasattr(agent, "reload_user_tools"):
                agent.reload_user_tools()
    except Exception as e:
        print(f"Warning: post-execute user-tool resync failed: {e}")


def strip_hallucinated_observations(msg: str | None) -> str:
    """Remove any ``<observation>...</observation>`` block from a MODEL turn. Observations are appended by
    the system (execute()) wrapping REAL run output — a model never legitimately emits one. A weaker model
    sometimes hallucinates an observation inside its own turn (fabricating a tool's return value or a
    function signature) and then reasons from that fiction; scrubbing it keeps the invented result out of
    the conversation so only the genuine observation (from the actual run) informs the next step. Real
    observations live in prior messages, not in the fresh model output this is called on, so this never
    touches a legitimate one. Returns the cleaned, stripped text.

    ``<observe>...</observe>`` is scrubbed too: the inherited ReAct scaffold names THAT tag for run
    output, so it is the spelling a fabricating model reaches for first. Live round 4 (case s1d):
    the model wrote two fake ``<observe>`` success blocks, the real run then FAILED, and because the
    fakes survived in its own message history it answered "completed successfully" over an
    observation that said ``File exists: False``."""
    return _OBSERVATION_TAG_RE.sub("", msg or "").strip()


_MISSING_MODULE_RE = re.compile(r"No module named ['\"]([A-Za-z0-9_.]+)['\"]")


#: The project's former import name. Twelve corpus trajectories import ``biomni.tool.*`` and fail;
#: the hint below names the rename instead of leaving the model to guess at a package that no
#: longer exists.
_LEGACY_PACKAGE = "biomni"


def mcp_tool_hint_for_failed_import(result: str | None, custom_function_names, kind: str = "python") -> str | None:
    """When a python ``<execute>`` fails with ``No module named 'X'`` AND a registered MCP tool maps to
    ``X`` (its name equals ``X`` or starts with ``X_``), return a hint telling the model to CALL that
    injected tool function instead of importing the library — otherwise ``None``.

    ``kind`` is the cell's language. A **bash/cli** cell that fails the same way (a ``python -c`` or a
    heredoc importing ``mcp_servers`` -- ~19 corpus trajectories) is told the tools exist only as
    callables inside a python cell, because no hint that names the function can help a subprocess
    that cannot see the REPL. A cell of any kind that imports the pre-rebrand ``biomni`` package is
    told the package's current name.

    This is the fix for a live-observed MCP-dispatch failure: given registered ``squidpy_*`` MCP tools, a
    model wrote ``import squidpy`` in the REPL and got ``No module named 'squidpy'`` — because heavy bio
    libraries live in per-tool WORKER envs reached over MCP, not in the thin agent-core env the REPL runs
    in. The registered ``squidpy_spatial_neighbors(...)`` function dispatches to that worker env; the model
    just has to call it. The hint is purely additive to an already-failed observation and only ever fires
    when a matching MCP tool is registered, so it can never perturb a passing run or a non-MCP session."""
    if not result:
        return None
    m = _MISSING_MODULE_RE.search(result)
    if not m:
        return None
    mod = m.group(1).split(".")[0]  # top-level package of "squidpy.gr" etc.
    names = list(custom_function_names or [])
    if mod == _LEGACY_PACKAGE:
        # The rename, said once. Whatever the model was reaching for under the old name lives under
        # the new one; when a registered tool matches the submodule it wanted, name that too.
        wanted = m.group(1).split(".")
        sub = wanted[2] if len(wanted) > 2 and wanted[1] == "tool" else ""
        related = sorted(n for n in names if sub and (n == sub or n.startswith(sub + "_")))
        tools = (
            " If you were reaching for an analysis tool, the registered MCP function(s) are: "
            + ", ".join(f"{n}(...)" for n in related[:8])
            + " -- call them directly inside a python <execute> cell."
            if related
            else ""
        )
        return (
            f"\n\n[hint] '{_LEGACY_PACKAGE}' is this project's FORMER name and is no longer importable; the "
            "package is 'spatialomicsgym' (e.g. `from spatialomicsgym.tool.<module> import <function>`)." + tools
        )
    if not names:
        return None
    if kind in ("bash", "cli"):
        related = sorted(n for n in names if n == mod or n.startswith(mod + "_"))
        if mod not in ("mcp_servers", "mcp_server") and not related:
            return None
        listed = ", ".join(f"{n}(...)" for n in (related or sorted(names))[:8])
        return (
            "\n\n[hint] The MCP tool functions exist ONLY as top-level callables inside a python <execute> "
            "cell -- a bash heredoc, `python -c` or any subprocess cannot import them ('mcp_servers' is not "
            f"a package). Write the call in a python <execute> cell instead: {listed}."
        )
    # The internal injection namespace: a model sometimes tries `import mcp_servers` / `from mcp_servers
    # import X` to introspect a tool, but the tools are injected as TOP-LEVEL callables — that namespace is
    # not an importable package, so it always fails. Point the model at the registered functions directly.
    if mod in ("mcp_servers", "mcp_server"):
        listed = ", ".join(f"{n}(...)" for n in sorted(names)[:12])
        return (
            f"\n\n[hint] Do NOT import 'mcp_servers' — it is not an importable package. The MCP tools are "
            f"already injected as top-level callable functions in this environment: {listed}. Call the one "
            f"you need directly inside <execute> with its keyword arguments (do not import anything to reach "
            f"it)."
        )
    related = sorted(n for n in names if n == mod or n.startswith(mod + "_"))
    if not related:
        return None
    listed = ", ".join(f"{n}(...)" for n in related[:8])
    return (
        f"\n\n[hint] '{mod}' is not importable in this agent environment and must NOT be imported "
        f"directly — it runs in a separate per-tool worker environment. Call the already-registered MCP "
        f"tool function(s) instead, directly inside <execute>: {listed}. They dispatch to the correct "
        f"worker env and return the analysis result."
    )


def configure_agent(agent, self_critic=False, test_time_scale_round=0):
    """Configure the agent with the initial system prompt and workflow.

    Args:
        agent: The STCoscientist agent instance
        self_critic: Whether to enable self-critic mode
        test_time_scale_round: Number of rounds for test time scaling

    """
    from spatialomicsgym.agent.stcoscientist import AgentState

    # Store self_critic for later use
    agent.self_critic = self_critic
    # Recorded on the agent, not only captured in the routing closure below, so that
    # ``STCoscientist.configure()`` can default to "keep what is set". Without this the scaling
    # round was unreadable after the fact and every bare ``configure()`` -- six call sites in
    # ``tool_management`` plus one per turn from ``maybe_resync_user_tools`` -- silently reset it.
    agent.test_time_scale_round = test_time_scale_round

    # Get data lake content
    data_lake_path = agent.path + "/data_lake"
    # SORTED: these names go into the system prompt and into what the retriever ranks, and `glob`
    # returns directory order -- so the same data lake on a different box produced a different
    # prompt. See the note in `know_how/loader.py::_load_documents`.
    data_lake_content = sorted(glob.glob(data_lake_path + "/*"))
    data_lake_items = [x.split("/")[-1] for x in data_lake_content]

    # data_lake_dict and library_content_dict are already set in __init__

    # Prepare tool descriptions
    tool_desc = {i: [x for x in j if x["name"] != "run_python_repl"] for i, j in agent.module2api.items()}

    # Prepare data lake items with descriptions
    data_lake_with_desc = []
    for item in data_lake_items:
        description = agent.data_lake_dict.get(item, f"Data lake item: {item}")
        data_lake_with_desc.append({"name": item, "description": description})

    # Add custom data items if they exist
    if hasattr(agent, "_custom_data") and agent._custom_data:
        for name, info in agent._custom_data.items():
            data_lake_with_desc.append({"name": name, "description": info["description"]})

    # Prepare library content list including custom software
    library_content_list = list(agent.library_content_dict.keys())
    if hasattr(agent, "_custom_software") and agent._custom_software:
        for name in agent._custom_software:
            if name not in library_content_list:  # Avoid duplicates
                library_content_list.append(name)

    # Generate the system prompt for initial configuration (is_retrieval=False)
    # Prepare custom resources for highlighting
    custom_tools = []
    if hasattr(agent, "_custom_tools") and agent._custom_tools:
        for name, info in agent._custom_tools.items():
            custom_tools.append(
                {
                    "name": name,
                    "description": info["description"],
                    "module": info["module"],
                }
            )

    custom_data = []
    if hasattr(agent, "_custom_data") and agent._custom_data:
        for name, info in agent._custom_data.items():
            # The path rides along: the prompt named only the basename (hunt 2026-09-30, uL4-honesty-4).
            custom_data.append({"name": name, "description": info["description"], "path": info.get("path")})

    custom_software = []
    if hasattr(agent, "_custom_software") and agent._custom_software:
        for name, info in agent._custom_software.items():
            custom_software.append({"name": name, "description": info["description"]})

    # Which know-how goes into the INITIAL prompt. `all` -- the default, and what every scored
    # run gets -- is the historical behaviour: every document, in full. The other modes trade prose
    # for an index, which matters only on the turns where this prompt is still the prompt (turn
    # zero, retriever off, retrieval raised, fail-safe fired); on an ordinary turn
    # update_system_prompt_with_selected_resources replaces all of it anyway.
    know_how_docs: list[dict] = []
    know_how_index: list[dict] = []
    if hasattr(agent, "know_how_loader") and agent.know_how_loader.documents:
        mode = enrolment_mode()
        know_how_docs = baseline_documents(agent.know_how_loader, mode)
        know_how_index = index_documents(agent.know_how_loader, mode)
        loaded_chars = sum(len(d.get("content") or "") for d in know_how_docs)
        print(
            f"📚 Know-how enrolment '{mode}': {len(know_how_docs)} document(s) in full "
            f"({loaded_chars:,} chars), {len(know_how_index)} listed by title"
        )

    agent.system_prompt = generate_system_prompt(
        agent,
        tool_desc=tool_desc,
        data_lake_content=data_lake_with_desc,
        library_content_list=library_content_list,
        self_critic=self_critic,
        is_retrieval=False,
        custom_tools=custom_tools if custom_tools else None,
        custom_data=custom_data if custom_data else None,
        custom_software=custom_software if custom_software else None,
        know_how_docs=know_how_docs if know_how_docs else None,
        know_how_index=know_how_index if know_how_index else None,
    )

    # Define the nodes
    def generate(state: AgentState) -> AgentState:
        # A turn the portal gave up on ends here, without another paid model call (see `execute`).
        if getattr(agent, "_turn_revoked", False):
            state["next_step"] = "end"
            return state
        # Add OpenAI-specific formatting reminders if using OpenAI models
        system_prompt = agent.system_prompt
        if hasattr(agent.llm, "model_name") and (
            "gpt" in str(agent.llm.model_name).lower() or "openai" in str(type(agent.llm)).lower()
        ):
            system_prompt += "\n\nIMPORTANT FOR GPT MODELS: You MUST use XML tags <execute> or <solution> in EVERY response. Do not use markdown code blocks (```) - use <execute> tags instead."

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        # A client whose provider will not take `stop` is read as a stream and closed at the first
        # stop sequence, instead of left to write on past it (see responses_stream). What is stored
        # is what the whole reply would give: the cut below finds the same stop in what was read.
        reader = _FirstStopReader(agent.llm) if getattr(agent.llm, "reads_to_first_stop", False) is True else None
        response = invoke_with_backoff(reader or agent.llm, messages)
        # Record what it cost. Reads what the provider already returned and appends a row -- no
        # prompt, message list, tool choice or retry is touched, which is what makes this safe to
        # land while RL-3 is in force. `BASELINE.md` lists cost as unreportable without it, so the
        # whole family of candidates that trades model calls for accuracy cannot be judged.
        _usage_of(agent).record("generate", response)
        if reader is not None and reader.stopped:
            print(
                "[execution.generate] Stopped reading the reply at its first stop sequence and closed the "
                "stream (the provider does not take `stop`). The provider reports token usage only at the "
                "end of a reply, so this call's is recorded as not measured."
            )

        # Normalize Responses API content blocks (list of dicts) into a plain string. The shared
        # helper is used by every consumer of `.content` so the implementations cannot drift.
        msg = message_to_text(response)

        # Stop where the loop's stop sequences say, on every provider (see cut_at_stop_sequence).
        cut = cut_at_stop_sequence(msg)
        if len(cut) < len(msg):
            print(
                f"[execution.generate] Cut the reply at its first stop sequence: {len(msg) - len(cut)} "
                "chars written past it (the provider does not take `stop`) were dropped."
            )
            msg = cut

        # A missing closing tag means one of two opposite things — the configured stop sequence ate it, or
        # the reply ran out of tokens mid-write — and only the provider's stop reason tells them apart.
        truncated = completion_was_truncated(response)
        if truncated:
            print("[execution.generate] Provider reports the response hit the output-token limit (truncated).")
        msg = close_dangling_tags(msg, truncated=truncated)

        # Scrub any <observation> the model fabricated in its own turn (a hallucinated tool result /
        # signature) BEFORE routing, so a real <execute> beside it still runs and the invented result never
        # enters the conversation. Real observations are appended by execute() as separate messages.
        if _OBSERVATION_TAG_RE.search(msg):
            print("[execution.generate] Stripping hallucinated <observation> from model output.")
            msg = strip_hallucinated_observations(msg)

        # One source of truth for actionable code (canonical <execute>, case-insensitive, then a fenced
        # ```code``` fallback) shared with execute() so the router and extractor can never disagree.
        think_match = _THINK_TAG_RE.search(msg)
        # A <solution> quoted in backticks is the model naming the tag, not answering (see
        # answer_tag_match); the strip below reads the same blocks.
        answer_match = answer_tag_match(msg)
        runnable_code = extract_runnable_code(msg)

        # STCoscientist (esp. GPT-5 family) sometimes hallucinates a full ReAct transcript in one turn:
        # <execute>...</execute>...<solution>premature bail</solution>. Honor the execute and strip the
        # bogus solution so the next generate() does not see STCoscientist as already done.
        if runnable_code is not None and answer_match:
            print(
                "[execution.generate] Detected execute+solution in same message; "
                "honoring execute, stripping premature solution."
            )
            # Span-aware: subtract the answer MINUS the span the code came from. A blanket
            # substitution deletes the honored <execute> whenever the model nests it inside the
            # <solution> (29 recorded turns), leaving execute() nothing to run.
            msg = strip_premature_solution(msg).rstrip()
            answer_match = None

        # Add the message to the state before checking for errors
        state["messages"].append(AIMessage(content=msg.strip()))

        if answer_match:
            # A <solution> normally ends the run. But a model (esp. a weaker one) sometimes emits an
            # UNCHECKED to-do plan and then a <solution> that only PROMISES future work ("Next I will run
            # X ...") without ever emitting an <execute> — the stop-sequence solution then terminates the
            # loop having run zero tools and produced no results. That is the think-only failure
            # (reason-without-action) wearing <solution> tags, so extend the same strike-guarded nudge to
            # it. Fire ONLY when nothing has executed yet in this conversation AND the message carries a
            # real (>=2-item) unchecked checklist — a signature no genuine conceptual answer or completed
            # tool result ever has — so this can never perturb a real solution. One nudge max: a second
            # plan-only bail is accepted, so a determined narrate-only model is never looped.
            if is_plan_only_solution_bail(msg, state["messages"]):
                print(
                    "[execution.generate] Detected an unchecked plan + a promise-only <solution> with no "
                    "execution; nudging the model to actually run the tool before solving."
                )
                state["messages"].append(HumanMessage(content=_PLAN_ONLY_SOLUTION_REPROMPT))
                state["next_step"] = "generate"
            elif is_empty_answer_bail(msg, state["messages"]):
                print("[execution.generate] The <solution> is an empty answer; nudging once for a real one.")
                state["messages"].append(HumanMessage(content=_EMPTY_ANSWER_REPROMPT))
                state["next_step"] = "generate"
            elif (check := pre_answer_check(agent, state["messages"])) is not None:
                print("[execution.generate] The pre-answer check found a problem; one more turn before accepting.")
                state["messages"].append(HumanMessage(content=check))
                state["next_step"] = "generate"
            else:
                state["next_step"] = "end"
                if truncated:
                    # The answer stands; what changes is that the turn stops claiming to be clean.
                    state["degraded"] = GIVEUP_TRUNCATED_SOLUTION
        elif runnable_code is not None:
            # Strike guard for a repeated-execution-failure loop (a non-ASCII SyntaxError the model keeps
            # re-pasting, etc.): if the last several execute observations were all errors, give up with a
            # clear, actionable message instead of burning toward ~1000 paid LLM calls. A clean
            # observation resets the streak, so a normal debug-and-fix flow is never affected.
            if count_trailing_execute_errors(state["messages"]) >= 4:
                print("[execution.generate] Repeated execution errors; ending to avoid a paid retry loop.")
                state["next_step"] = "end"
                # Recorded so the turn is not reported as a clean finish -- see AgentState.degraded.
                state["degraded"] = GIVEUP_REPEATED_ERROR
                # Quote the error that actually repeated rather than guessing at a cause. The old
                # text asserted "often a stray non-ASCII character -- an em-dash or curly quote",
                # which is one specific SyntaxError; it was printed for every repeated failure
                # whatever it was, so a reader debugging a missing file or an unimportable module
                # was sent looking for a curly quote that was never there.
                last_error = _last_execute_error_text(state["messages"])
                detail = f" The error that repeated was:\n{last_error}" if last_error else ""
                state["messages"].append(
                    AIMessage(
                        content=(
                            "Execution terminated: the same error repeated after several attempts, so "
                            "the loop stopped rather than pay for more identical retries." + detail
                        )
                    )
                )
            elif count_trailing_failed_executions(state["messages"]) >= ROTATING_ERROR_LIMIT:
                print("[execution.generate] Every recent execution failed; ending instead of rotating further.")
                state["next_step"] = "end"
                state["degraded"] = GIVEUP_EVERY_ATTEMPT_FAILED
                state["messages"].append(AIMessage(content=_rotating_error_giveup_text(state["messages"])))
            else:
                state["next_step"] = "execute"
        elif think_match:
            # A <think> block with no <execute>/<solution>: the model reasoned but did not act. One such
            # turn is legitimate planning, but a model that keeps emitting think-only turns loops
            # generate->generate to recursion_limit=1000 — the no-tags 2-strike guard below does NOT cover
            # this because think_match short-circuits it. Mirror that guard: nudge the model to act, then
            # give up after a second unheeded nudge instead of burning ~1000 paid LLM calls then crashing.
            if count_think_only_reprompts(state["messages"]) >= 2:
                print("Detected repeated think-only turns without an action, ending conversation")
                state["next_step"] = "end"
                state["degraded"] = GIVEUP_THINK_ONLY
                state["messages"].append(
                    AIMessage(
                        content="Execution terminated: reasoning was produced but no <execute> or <solution> "
                        "action followed after repeated attempts."
                    )
                )
            else:
                state["messages"].append(HumanMessage(content=_THINK_ONLY_REPROMPT))
                state["next_step"] = "generate"
        else:
            print("response was truncated..." if truncated else "parsing error...")
            # Count the reprompts already sent (the HumanMessage below). The old counter looked for a
            # capitalized string in AIMessages that never appears, so this 2-strike give-up was dead and a
            # tag-less model looped to recursion_limit=1000 (~1000 paid LLM calls) then crashed.
            if count_no_tags_reprompts(state["messages"]) >= 2:
                print("Detected repeated parsing errors, ending conversation")
                state["next_step"] = "end"
                state["degraded"] = GIVEUP_TRUNCATED if truncated else GIVEUP_NO_ACTION
                state["messages"].append(
                    AIMessage(
                        content=(
                            "Execution terminated: the responses kept being cut off by the output token "
                            "limit before a complete action was produced. Raise the model's output token "
                            "limit, or ask for the task in smaller steps."
                        )
                        if truncated
                        else "Execution terminated due to repeated parsing errors. Please check your input and try again."
                    )
                )
            else:
                # Name the real cause. A severed reply has not "forgotten the tags", and telling it so
                # makes it rewrite the same over-long turn and get cut off in the same place.
                state["messages"].append(HumanMessage(content=_TRUNCATED_REPROMPT if truncated else _NO_TAGS_REPROMPT))
                state["next_step"] = "generate"
        return state

    def execute(state: AgentState) -> AgentState:
        from datetime import datetime

        # The message exactly as the router read it: generate() is the only way in, and it has already
        # closed what the stop sequence ate (close_dangling_tags) and left open what the token cap
        # severed. A second auto-close here, by a plainer rule, used to undo both decisions whenever a
        # fence routed the turn: a quoted `<execute>` was closed and the English after it ran instead of
        # the fence, and a severed cell was handed the closer close_dangling_tags had withheld.
        last_message = state["messages"][-1].content

        # Extract via the SAME helper the router used, so a routed turn always yields code. A case
        # ("<Execute>") or markdown-fence mismatch used to route here then extract nothing -> no
        # observation -> the execute->generate edge looped the graph to the recursion limit.
        code = extract_runnable_code(last_message)
        if code is not None and getattr(agent, "_turn_revoked", False):
            # The portal gave this turn up (a stall, an abandoned stream) and handed the agent's
            # place to a new one. A producer still unwinding here must not run another cell: the
            # shared worker is respawned for whichever ACCOUNT is bound now, which may be the next
            # turn's (u11-stcoscientist-extra-14).
            state["messages"].append(
                AIMessage(content="<observation>Not run: this turn was stopped by the portal.</observation>")
            )
            state["next_step"] = "end"
            return state
        if code is not None:
            timeout = agent.timeout_seconds
            # Classify by leading marker + strip it on the STRIPPED code (a block opening with a newline,
            # e.g. "<execute>\n#!CLI ...", now de-markers correctly instead of leaving #!CLI to be folded
            # into a bash comment and silently never run).
            kind, cleaned = classify_and_clean_code(code)
            # What is running, WHILE it runs. Nothing else knows: the stream yields only on node
            # completion, so between the cell starting and its observation arriving the portal
            # has nothing to say but "the agent reasons in steps". Cleared in the `finally`
            # below, so a crash cannot leave the page claiming a cell is still going.
            agent._running_cell = {"language": kind, "started": time.time()}
            # Cleared for EVERY language, not just python. ``get_captured_plots()`` is read
            # unconditionally further down and stamped onto this step's entry, so clearing only in
            # the python branch meant an R or bash cell inherited whatever the previous python
            # cell had drawn. Measured: a python cell that saved one figure, then a `#!BASH` cell
            # -- both entries carried the same image object, so the portal's per-step view
            # credited the figure to the step that did not make it, and the base64 payload was
            # duplicated in memory for the life of the process.
            clear_execution_plots(agent)
            try:
                if kind in ("r", "bash", "cli"):
                    # Through the worker under process isolation, exactly as the python branch
                    # below does. Until 2026-09-23 these two went straight to `utils.execution`,
                    # which starts the script with shell=True and env=os.environ.copy() IN THIS
                    # PROCESS -- and the portal is root, so a `#!BASH` cell ran as root with every
                    # provider key while the python half of the same turn was confined to uid 999.
                    # `prompt_builder` advertises `#!BASH` to the model by name, so this was a
                    # documented route around the boundary, not an obscure one.
                    #
                    # The in-process runners stay for the CLI, where `boundary.describe()` already
                    # reports "none" and the terminal is the trust boundary.
                    if _process_isolation():
                        result = _run_shell_in_worker("r" if kind == "r" else "bash", cleaned, timeout)
                    elif kind == "r":
                        result = run_with_timeout(run_r_code, [cleaned], {"timeout": timeout}, timeout=timeout)
                    else:
                        result = run_with_timeout(run_bash_script, [cleaned], {"timeout": timeout}, timeout=timeout)
                else:  # python
                    inject_custom_functions(agent)
                    # Duplicate-dispatch memo, armed for exactly this script's lifetime: a message that
                    # carries the same tool call twice (hallucinated-transcript turns concatenate into
                    # one script by design -- see action.py) must not pay for the dispatch twice. Live
                    # cost that motivated this: s4a re-ran a 4-hour cell2location fit for a
                    # byte-identical overwrite. Disarm in finally, so wrappers called OUTSIDE a script
                    # (library users, notebooks) are never affected, even after a timeout here.
                    arm_tool_call_memo()
                    try:
                        if _process_isolation():
                            # The worker client enforces the budget itself -- a real SIGKILL of the
                            # process group at the deadline -- which is stronger than a thread that
                            # cannot be killed. Running it under run_with_timeout as well let the
                            # thread be abandoned mid-upcall and wedged the worker (the audit's S1).
                            result = _run_cell_in_worker(cleaned)
                        else:
                            # A timeout here abandons the cell's thread and keeps its namespace, and
                            # the observation says so -- the worker's says the opposite, truly.
                            result = run_with_timeout(
                                _python_cell_with_its_own_memo, [cleaned], timeout=timeout, after=TIMEOUT_KEPT_NOTE
                            )
                    finally:
                        disarm_tool_call_memo()
            finally:
                # Always, including on a timeout or a raise: a stale `_running_cell`
                # would have the page reporting a cell that finished minutes ago.
                agent._running_cell = None
            result = clip_observation(result, kind=kind)

            # If this turn's code created/deleted/modified a user tool, re-sync the live catalogs so
            # the change takes effect in-session (no-op unless tool_creation_enabled and the config
            # actually changed).
            maybe_resync_user_tools(agent)

            # Store the execution result with the triggering message
            if not hasattr(agent, "_execution_results"):
                agent._execution_results = []

            # Get any plots that were generated during this execution
            execution_plots = []
            try:
                from spatialomicsgym.tool.support_tools import get_captured_plots

                current_plots = get_captured_plots()
                execution_plots = current_plots.copy()
            except Exception as e:
                print(f"Warning: Could not capture plots from execution: {e}")
                execution_plots = []

            # Store the execution result with metadata
            #
            # `language`, `ok` and `code` are recorded because THIS node knows them and nothing
            # downstream does. `STREAM_PROTOCOL.md` 4: the portal's `_classify_step` reconstructs
            # the kind of a step from langchain's banner text, which is a classifier standing in
            # for a fact the code already had. A reader watching a turn wants "it ran R and it
            # failed", and that sentence exists here and nowhere else.
            #
            # Recording only. Nothing about the prompt, the message list or tool selection moves,
            # which is what makes this safe to add while agent performance is frozen -- there is
            # no path from these keys back into anything the model sees.
            execution_entry = {
                "triggering_message": last_message,  # The AI message that contained <execute>
                "images": execution_plots,  # Base64 encoded images from this execution
                "timestamp": datetime.now().isoformat(),
                # "python" | "r" | "bash" | "cli", from `classify_and_clean_code` -- the same
                # value that chose which runner to call, not a second guess at it.
                "language": kind,
                # The observation as the model will see it, judged by the SAME regex the ReAct
                # loop uses to decide whether a step failed. A second opinion here would be a
                # second answer to "did that work", and the loop's is the one that acts.
                "ok": not bool(re.search(_EXEC_ERROR_RE, result or "")),
                # Bounded: a cell can be thousands of lines and this list lives for the whole
                # session. Enough to show what ran, not a second copy of the transcript.
                "code": (cleaned or "")[:4000],
            }
            agent._execution_results.append(execution_entry)
            # A counter that NEVER resets, alongside a list that does. ``sse_events`` decides
            # which observation belongs to which cell by asking whether the record list has
            # GROWN since the turn began -- a length comparison, which a per-turn clear would
            # silently break by making the list shorter than the cursor. The sequence number
            # answers the same question and is immune to the clear.
            agent._execution_seq = getattr(agent, "_execution_seq", 0) + 1

            # If a python cell failed by importing a bio library that is actually exposed as a registered
            # MCP tool (heavy libs live in per-tool worker envs, not the thin agent-core env the REPL runs
            # in), append a hint pointing the model at the injected tool function so it self-corrects on the
            # next turn instead of retrying the same doomed import. Additive; fires only for python cells
            # with a matching registered MCP tool, so it never touches a passing run or a non-MCP session.
            # The tool's own text, neutralised BEFORE anything of ours is appended to it: the
            # hint and the [NOT RUN] note below are written by this file and legitimately name
            # `<execute>`, so escaping the whole body afterwards would mangle our own prose while
            # the thing that needed it had already been interpolated. See `as_tool_data`.
            body = as_tool_data(result)
            if kind in ("python", "bash", "cli"):
                hint = mcp_tool_hint_for_failed_import(
                    result, list(getattr(agent, "_custom_functions", {}) or {}), kind=kind
                )
                if hint:
                    body = body + hint
            # Name the blocks this turn wrote and did not run. Only the first runs when any block
            # names a language, which is right -- folding #!R or #!BASH into a python cell would be
            # worse -- but saying nothing let the model reason as though both had run. Measured at
            # 7 of 40 recorded multi-execute turns.
            dropped = dropped_execute_blocks(last_message)
            if dropped:
                body = (
                    f"{body}\n\n[NOT RUN] This turn contained {len(dropped) + 1} <execute> blocks and only "
                    f"the first ran, because a later one names its own language and cannot be folded into "
                    f"the same cell: {'; '.join(dropped)}. Send the remaining block(s) as their own turn."
                )
            observation = f"\n<observation>{body}</observation>"
            state["messages"].append(AIMessage(content=observation.strip()))
        else:
            # The router sent us here but nothing could be extracted (should be unreachable now that both
            # sides share extract_runnable_code, but guard it): append a corrective observation rather than
            # nothing, since appending nothing loops execute->generate with no new info -> recursion crash.
            state["messages"].append(
                AIMessage(
                    content="<observation>No runnable code block was found to execute. Put the code inside "
                    "a single <execute>...</execute> block and try again.</observation>"
                )
            )

        return state

    def routing_function(
        state,
    ) -> Literal["execute", "generate", "end"]:
        next_step = state.get("next_step")
        if next_step == "execute":
            return "execute"
        elif next_step == "generate":
            return "generate"
        elif next_step == "end":
            return "end"
        else:
            raise ValueError(f"Unexpected next_step: {next_step}")

    def routing_function_self_critic(
        state,
    ) -> Literal["generate", "end"]:
        next_step = state.get("next_step")
        if next_step == "generate":
            return "generate"
        elif next_step == "end":
            return "end"
        else:
            raise ValueError(f"Unexpected next_step: {next_step}")

    def execute_self_critic(state) -> dict:
        if agent.critic_count < test_time_scale_round:
            # Generate feedback based on message history
            messages = state["messages"]
            feedback_prompt = f"""
                Here is a reminder of what is the user requested: {agent.user_task}
                Examine the previous executions, reaosning, and solutions.
                Critic harshly on what could be improved?
                Be specific and constructive.
                Think hard what are missing to solve the task.
                No question asked, just feedbacks.
                """
            feedback = invoke_with_backoff(agent.llm, messages + [HumanMessage(content=feedback_prompt)])
            _usage_of(agent).record("self_critic", feedback)

            # Add feedback as a new message
            state["messages"].append(
                HumanMessage(
                    content=f"Wait... this is not enough to solve the task. Here are some feedbacks for improvement:\n{message_to_text(feedback)}"
                )
            )
            agent.critic_count += 1
            # The critic sends EVERY end back to generate, the three give-ups included, and each of
            # those had set `degraded`. Left set, a real <solution> from the next generate was still
            # reported as a give-up and bought a rescue round (u12-react-15). The next generate sets
            # it again if it gives up again.
            state["degraded"] = None
            state["next_step"] = "generate"
        else:
            state["next_step"] = "end"

        return state

    # Create the workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("generate", generate)
    workflow.add_node("execute", execute)

    if self_critic:
        workflow.add_node("self_critic", execute_self_critic)
        # Add conditional edges
        workflow.add_conditional_edges(
            "generate",
            routing_function,
            path_map={
                "execute": "execute",
                "generate": "generate",
                "end": "self_critic",
            },
        )
        workflow.add_conditional_edges(
            "self_critic",
            routing_function_self_critic,
            path_map={"generate": "generate", "end": END},
        )
    else:
        # Add conditional edges
        workflow.add_conditional_edges(
            "generate",
            routing_function,
            path_map={"execute": "execute", "generate": "generate", "end": END},
        )
    workflow.add_edge("execute", "generate")
    workflow.add_edge(START, "generate")

    # Compile the workflow.
    #
    # No checkpointer, deliberately. One used to be attached right after this line, and it was
    # pure cost: ``AgentState.messages`` is a plain ``list[BaseMessage]`` with no reducer, so its
    # channel is ``LastValue`` and each turn's input *replaces* the checkpointed list. Driven --
    # two turns on one thread_id, and turn 2 comes back with 2 messages, its own. Nothing in the
    # package calls ``get_state`` or ``update_state``, so the retained history was never read by
    # anything, while ``sog-web`` is a long-lived process whose RSS grew linearly with it:
    # measured at ~150 KB per turn, ~1.5 MB after ten, for a small turn.
    #
    # ``sog_portal/server.py``'s reset docstring already told the truth about the design -- "the graph
    # is compiled without a checkpointer and each turn is invoked with a fresh message list" --
    # and was wrong only about this line. Now it is right.
    #
    # Resuming a turn from a checkpoint would be a real feature; it would need a reducer on
    # ``messages`` and a bounded saver, and it is not what these two lines were doing.
    agent.app = workflow.compile()
    # display(Image(agent.app.get_graph().draw_mermaid_png()))


def _pack_candidates(agent) -> list[dict]:
    """The tier-2 candidates for THIS turn -- empty unless the packs are enabled right now.

    ``packs_enabled()`` is asked per turn, not once at wire-up, so a loader that was filled before
    ``benchmarking_enabled`` was switched on offers nothing to the scored turn that follows. The
    loader is asked through ``getattr``: the stub loaders in the test suite, and any loader built
    before tier 2 existed, have no ``get_pack_summaries`` and must read as "no packs".
    """
    try:
        if not packs_enabled():
            return []
    except Exception:
        return []
    summaries = getattr(getattr(agent, "know_how_loader", None), "get_pack_summaries", None)
    if not callable(summaries):
        return []
    try:
        offered = summaries()
    except Exception:
        return []
    return [item for item in (offered or []) if isinstance(item, dict) and item.get("id")]


def _call_with_supported_kwargs(fn, *args, **optional):
    """Call ``fn`` with only the optional keyword arguments its signature actually accepts.

    ``agent.retriever`` is a REPLACEABLE collaborator. Six tests in this suite install a duck-typed
    stub with the old three-parameter signature, and nothing stops a user's own retriever from
    doing the same -- so adding ``usage=`` to the call site raised
    ``TypeError: got an unexpected keyword argument 'usage'`` and took the whole turn down. A
    telemetry field and a determinism pin are not worth that.

    Signature inspection rather than ``except TypeError``: a ``TypeError`` raised INSIDE the
    retriever looks identical from out here, and swallowing that one would hide a real failure
    behind a silent fallback. A retriever declaring ``**kwargs`` is given everything.
    """
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args)  # a builtin or a C callable: give it only what it certainly takes
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return fn(*args, **optional)
    return fn(*args, **{name: value for name, value in optional.items() if name in parameters})


def _llm_for_retrieval(agent):
    """The LLM the tool retriever should use, pinned to temperature 0 for a run being SCORED.

    The retriever is not an aside: it decides WHICH TOOLS the assembled system prompt names, so it
    picks the contents of the thing the benchmark measures. It runs on ``agent.llm``, whose
    temperature is ``SpatialOmicsGymConfig.temperature`` -- 0.7 by default -- with no seed. Two runs
    of the same task on the same commit could therefore be given different tools and produce
    different answers, and nothing in the transcript would say why. `use_tool_retriever` ships ON.

    Pinned only under ``benchmarking_enabled``, so ordinary use keeps the configured sampling; and
    only where the provider allows an explicit temperature at all -- the newest Claude generation,
    the OpenAI Responses family (the shipped default is ``azure-gpt-5.5``) and the o-series all
    answer HTTP 400 to any non-default value. Where it is refused this returns the LLM unchanged
    and the determinism is simply not available; it is not worth a 400 on every scored turn to
    pretend otherwise.
    """
    from spatialomicsgym.config import default_config

    if not getattr(default_config, "benchmarking_enabled", False):
        return agent.llm
    model = getattr(default_config, "llm", "") or getattr(agent, "llm_model", "") or ""
    from spatialomicsgym.llm import accepts_an_explicit_temperature

    if not accepts_an_explicit_temperature(model):
        return agent.llm
    try:
        return agent.llm.bind(temperature=0)
    except Exception:
        # A chat model that will not take the bind is not a reason to lose the retrieval.
        return agent.llm


def prepare_resources_for_retrieval(agent, prompt):
    """Prepare resources for retrieval and return selected resource names.

    Args:
        agent: The STCoscientist agent instance
        prompt: The user's query

    Returns:
        dict: Dictionary containing selected resource names for tools, data_lake, and libraries
    """
    if not agent.use_tool_retriever:
        return None

    # Gather all available resources
    # 1. Tools from the registry
    all_tools = agent.tool_registry.tools if hasattr(agent, "tool_registry") else []

    # 2. Data lake items with descriptions
    data_lake_path = agent.path + "/data_lake"
    # SORTED: these names go into the system prompt and into what the retriever ranks, and `glob`
    # returns directory order -- so the same data lake on a different box produced a different
    # prompt. See the note in `know_how/loader.py::_load_documents`.
    data_lake_content = sorted(glob.glob(data_lake_path + "/*"))
    data_lake_items = [x.split("/")[-1] for x in data_lake_content]

    # Create data lake descriptions for retrieval
    data_lake_descriptions = []
    for item in data_lake_items:
        description = agent.data_lake_dict.get(item, f"Data lake item: {item}")
        data_lake_descriptions.append({"name": item, "description": description})

    # Add custom data items to retrieval if they exist
    if hasattr(agent, "_custom_data") and agent._custom_data:
        for name, info in agent._custom_data.items():
            data_lake_descriptions.append({"name": name, "description": info["description"]})

    # 3. Libraries with descriptions - use library_content_dict directly
    library_descriptions = []
    for lib_name, lib_desc in agent.library_content_dict.items():
        library_descriptions.append({"name": lib_name, "description": lib_desc})

    # Add custom software items to retrieval if they exist
    if hasattr(agent, "_custom_software") and agent._custom_software:
        for name, info in agent._custom_software.items():
            # Check if it's not already in the library descriptions to avoid duplicates
            if not any(lib["name"] == name for lib in library_descriptions):
                library_descriptions.append({"name": name, "description": info["description"]})

    # 4. Know-how documents
    know_how_summaries = agent.know_how_loader.get_document_summaries()

    # 5. Inject skills metadata as additional context for the retriever.
    #
    # This candidate is synthetic: no know-how document backs it, so the resolution below cannot
    # find it through know_how_loader.get_document_by_id, and the `if doc:` guard there used to drop
    # it without a word. It is the only id in the candidate list that can ever fail to resolve --
    # every other summary came from documents.values() -- which is why a catalog the model was
    # explicitly invited to ask for could never be delivered to it. Keep the agent-shaped copy here
    # and resolve from it. The description text is unchanged, so the retrieval prompt is identical.
    synthetic_documents: dict[str, dict] = {}
    if hasattr(agent, "_skills_context") and agent._skills_context:
        catalog_text = (
            "Structured catalog of all available MCP tools organized by task type "
            "(spatial_clustering, deconvolution, svg_detection, cell_segmentation, "
            "spatial_alignment, cell_communication, spatial_analysis, data_conversion). "
            "Each tool has a priority (1=recommended, 2=alternative, 3=specialized). "
            "Use this to select the best tool for a given analysis task.\n" + agent._skills_context
        )
        know_how_summaries.append(
            {"id": "skills_catalog", "name": "MCP Tool Skills Catalog", "description": catalog_text}
        )
        synthetic_documents["skills_catalog"] = {
            "id": "skills_catalog",
            "name": "MCP Tool Skills Catalog",
            "description": catalog_text,
            "content": catalog_text,
            "metadata": {},
        }

    # Use retrieval to get relevant resources
    resources = {
        "tools": all_tools,
        "data_lake": data_lake_descriptions,
        "libraries": library_descriptions,
        "know_how": know_how_summaries,
    }

    # Use prompt-based retrieval with the agent's LLM
    selected_resources = _call_with_supported_kwargs(
        agent.retriever.prompt_based_retrieval,
        prompt,
        resources,
        llm=_llm_for_retrieval(agent),
        usage=_usage_of(agent),
    )
    print("\n" + "=" * 60)
    print("🔍 RESOURCE RETRIEVAL")
    print("=" * 60)
    print("Using prompt-based retrieval with the agent's LLM")

    # Extract the names from the selected resources for the system prompt
    selected_resources_names = {
        "tools": selected_resources["tools"],
        "data_lake": [],
        "libraries": [lib["name"] if isinstance(lib, dict) else lib for lib in selected_resources["libraries"]],
        "know_how": [],
    }

    # Process data lake items to extract just the names
    for item in selected_resources["data_lake"]:
        if isinstance(item, dict):
            selected_resources_names["data_lake"].append(item["name"])
        elif isinstance(item, str) and ": " in item:
            # If the item already has a description, extract just the name
            name = item.split(": ")[0]
            selected_resources_names["data_lake"].append(name)
        else:
            selected_resources_names["data_lake"].append(item)

    # Process know-how documents - get the full content for selected documents
    if "know_how" in selected_resources and selected_resources["know_how"]:
        print("\n📚 Know-How Documents Retrieved:")
        for item in selected_resources["know_how"]:
            # Every drop below used to be silent, so a run could print this header, list not one
            # document, and summarise "Know-How: 0 selected" -- which is what two of the recorded
            # retrievals show. Say which selection was lost instead.
            doc_id = item.get("id") if isinstance(item, dict) else None
            doc = agent.know_how_loader.get_document_by_id(doc_id) if doc_id else None
            if doc:
                # Create a copy with content_without_metadata for agent context
                doc_for_agent = {
                    "id": doc["id"],
                    "name": doc["name"],
                    "description": doc["description"],
                    "content": doc["content_without_metadata"],  # Use stripped version for agent
                    "metadata": doc["metadata"],
                }
            elif doc_id in synthetic_documents:
                doc_for_agent = synthetic_documents[doc_id]
            else:
                print(f"  ⚠️  Selected know-how {doc_id or item!r} matches no document -- skipped")
                continue
            print(f"  ✓ {doc_for_agent['name']}")
            selected_resources_names["know_how"].append(doc_for_agent)
    else:
        print("\n📚 Know-How: None retrieved for this query")

    # Tier 2 -- the merged external packs -- is a SECOND pass, after pass 1 has returned, over the
    # packs alone, with its own budget and its own key. Pass 1 above is not given a pack name and
    # is not edited; that is what keeps its prompt and its selection byte-identical with the packs
    # on or off. The key is always present so every consumer can read it without a guard.
    selected_resources_names["know_how_packs"] = []
    pack_candidates = _pack_candidates(agent)
    if pack_candidates:
        budget = pack_budget()
        picks = []
        if budget > 0:
            try:
                picks = _call_with_supported_kwargs(
                    agent.retriever.retrieve_packs,
                    prompt,
                    pack_candidates,
                    llm=_llm_for_retrieval(agent),
                    budget=budget,
                    usage=_usage_of(agent),
                )
            except Exception as exc:
                print(f"  ⚠️  Tier-2 pack retrieval raised ({exc}); no pack is added this turn")
                picks = []
        # Resolve every pick BEFORE printing the header. The tier-1 block above fixed exactly this:
        # a header printed over a list that then resolved to nothing. And the loader is asked the
        # way _pack_candidates asks it -- through getattr, inside a guard -- because a loader that
        # offered summaries but has no get_pack_document_by_id (a stub; a loader from before tier 2)
        # must read as "no packs", not take the turn down.
        resolve = getattr(getattr(agent, "know_how_loader", None), "get_pack_document_by_id", None)
        resolved: list[dict] = []
        unresolved: list[str] = []
        for item in list(picks or [])[:budget]:
            doc_id = item.get("id") if isinstance(item, dict) else None
            doc = None
            if doc_id and callable(resolve):
                try:
                    doc = resolve(doc_id)
                except Exception:
                    doc = None
            if not doc:
                unresolved.append(f"{doc_id or item!r}")
                continue
            resolved.append(
                {
                    "id": doc["id"],
                    "name": doc["name"],
                    "description": doc["description"],
                    "content": doc["content_without_metadata"],
                    "metadata": doc["metadata"],
                    "pack": doc.get("pack"),
                }
            )
        if resolved:
            print("\n📦 Know-How Packs Retrieved (tier 2):")
            for doc in resolved:
                print(f"  ✓ {doc['name']}")
        for what in unresolved:
            print(f"  ⚠️  Selected pack {what} matches no document -- skipped")
        selected_resources_names["know_how_packs"].extend(resolved)

    # Print summary of what was retrieved
    print("\n" + "-" * 60)
    print("📊 RETRIEVAL SUMMARY:")
    print(f"  🔧 Tools: {len(selected_resources_names['tools'])} selected")
    print(f"  📊 Data Lake: {len(selected_resources_names['data_lake'])} selected")
    print(f"  ⚙️  Libraries: {len(selected_resources_names['libraries'])} selected")
    print(f"  📚 Know-How: {len(selected_resources_names['know_how'])} selected")
    print(f"  📦 Know-How packs (tier 2): {len(selected_resources_names['know_how_packs'])} selected")
    print("=" * 60 + "\n")

    return selected_resources_names


#: Grouping label for a selected tool whose module neither the tool itself nor ``module2api``
#: records. It is deliberately NOT a dotted path: ``textify_api_dict`` renders whatever key it is
#: given as ``Import file: <key>``, and the previous fallback named ``spatialomicsgym.tool.scRNA_tools``
#: -- a module that has not existed since the biomni rename -- so every ``from ... import`` the model
#: copied from that line raised ModuleNotFoundError. Saying we do not know beats naming somewhere
#: that is not there, and the tool still reaches the prompt, so nothing is silently dropped.
UNKNOWN_TOOL_MODULE = "unknown -- module not recorded for these tools; do not guess an import path"


def update_system_prompt_with_selected_resources(agent, selected_resources):
    """Update the system prompt with the selected resources.

    Args:
        agent: The STCoscientist agent instance
        selected_resources: Dictionary of selected resources
    """
    # Extract tool descriptions for the selected tools
    tool_desc = {}
    for tool in selected_resources["tools"]:
        # Get the module name from the tool
        if isinstance(tool, dict):
            module_name = tool.get("module", None)

            # If module is not specified, try to find it in the module2api
            if not module_name and hasattr(agent, "module2api"):
                for mod, apis in agent.module2api.items():
                    for api in apis:
                        if api.get("name") == tool.get("name"):
                            module_name = mod
                            break
                    if module_name:
                        break

            # If still not found, say so rather than inventing a module
            if not module_name:
                module_name = UNKNOWN_TOOL_MODULE
        else:
            module_name = getattr(tool, "module_name", None)

            # If module is not specified, try to find it in the module2api
            if not module_name and hasattr(agent, "module2api"):
                tool_name = getattr(tool, "name", str(tool))
                for mod, apis in agent.module2api.items():
                    for api in apis:
                        if api.get("name") == tool_name:
                            module_name = mod
                            break
                    if module_name:
                        break

            # If still not found, say so rather than inventing a module
            if not module_name:
                module_name = UNKNOWN_TOOL_MODULE

        if module_name not in tool_desc:
            tool_desc[module_name] = []

        # Add the tool to the appropriate module.
        if isinstance(tool, dict):
            # A COPY, carrying the module this call resolved. The retriever hands back the very
            # dicts that live in ``agent.tool_registry.tools``, so writing ``module`` into them
            # pinned the answer for the rest of the session -- including the UNKNOWN sentinel.
            # Once written, ``if not module_name`` never ran again, so a tool whose module became
            # known later (``add_mcp``, ``reload_user_tools``) stayed filed under "unknown --
            # module not recorded for these tools" in every subsequent prompt. The same mechanism
            # cached a *correct* module too, which then went stale when a tool was re-wired to a
            # different server. Measured: one retrieval, and the registry entry carried the
            # sentinel permanently.
            # ``module_name`` already IS ``tool["module"]`` whenever the tool carried one -- it is
            # seeded from it at the top of this loop and the resolution below only runs when that
            # is falsy. So there is nothing to prefer here, and writing ``tool.get("module") or
            # module_name`` would suggest a precedence that does not exist.
            tool_desc[module_name].append({**tool, "module": module_name})
        else:
            # Convert tool object to dictionary
            tool_dict = {
                "name": getattr(tool, "name", str(tool)),
                "description": getattr(tool, "description", ""),
                "parameters": getattr(tool, "parameters", {}),
                "module": module_name,  # Explicitly include the module
            }
            tool_desc[module_name].append(tool_dict)

    # Prepare data lake items with descriptions
    data_lake_with_desc = []
    for item in selected_resources["data_lake"]:
        description = agent.data_lake_dict.get(item, f"Data lake item: {item}")
        data_lake_with_desc.append({"name": item, "description": description})

    # Prepare custom resources for highlighting
    custom_tools = []
    if hasattr(agent, "_custom_tools") and agent._custom_tools:
        for name, info in agent._custom_tools.items():
            custom_tools.append(
                {
                    "name": name,
                    "description": info["description"],
                    "module": info["module"],
                }
            )

    custom_data = []
    if hasattr(agent, "_custom_data") and agent._custom_data:
        for name, info in agent._custom_data.items():
            # The path rides along: the prompt named only the basename (hunt 2026-09-30, uL4-honesty-4).
            custom_data.append({"name": name, "description": info["description"], "path": info.get("path")})

    custom_software = []
    if hasattr(agent, "_custom_software") and agent._custom_software:
        for name, info in agent._custom_software.items():
            custom_software.append({"name": name, "description": info["description"]})

    # Extract know-how documents if present
    know_how_docs = selected_resources.get("know_how", [])
    # Tier 2, when the second pass added any; None otherwise, and None renders nothing at all.
    know_how_packs = selected_resources.get("know_how_packs", [])

    agent.system_prompt = generate_system_prompt(
        agent,
        tool_desc=tool_desc,
        data_lake_content=data_lake_with_desc,
        library_content_list=selected_resources["libraries"],
        self_critic=getattr(agent, "self_critic", False),
        is_retrieval=True,
        custom_tools=custom_tools if custom_tools else None,
        custom_data=custom_data if custom_data else None,
        custom_software=custom_software if custom_software else None,
        know_how_docs=know_how_docs if know_how_docs else None,
        know_how_packs=know_how_packs if know_how_packs else None,
    )
    try:  # CTX-1: what this turn was given; leaves a scored trial beside the usage, in the probe contract
        tools_given = selected_resources.get("tools", []) or []
        _usage_of(agent).note_context(
            source="retrieval",
            tools=sorted({str(t.get("name") if isinstance(t, dict) else getattr(t, "name", t)) for t in tools_given}),
            know_how=[str(d.get("id") or d.get("name")) for d in know_how_docs if isinstance(d, dict)],
            know_how_packs=[str(d.get("id") or d.get("name")) for d in know_how_packs if isinstance(d, dict)],
            libraries=len(selected_resources.get("libraries", []) or []),
            data_lake=len(selected_resources.get("data_lake", []) or []),
            system_prompt_chars=len(agent.system_prompt or ""),
        )
    except Exception:
        pass  # telemetry must never cost the turn


def parse_tool_calls_from_code_wrapper(agent, code: str) -> list[str]:
    """Parse code to detect imported tools by looking for import statements.

    Args:
        agent: The STCoscientist agent instance
        code: The Python code to parse

    Returns:
        List of detected tool names
    """
    module2api = getattr(agent, "module2api", {})
    custom_functions = getattr(agent, "_custom_functions", {})
    return parse_tool_calls_from_code(code, module2api, custom_functions)


def parse_tool_calls_with_modules_wrapper(agent, code: str) -> list[tuple[str, str]]:
    """Parse code to detect imported tools and their modules.

    Args:
        agent: The STCoscientist agent instance
        code: The Python code to parse

    Returns:
        List of tuples (tool_name, module_name)
    """
    module2api = getattr(agent, "module2api", {})
    custom_functions = getattr(agent, "_custom_functions", {})
    return parse_tool_calls_with_modules(code, module2api, custom_functions)


def inject_custom_functions(agent):
    """Inject custom functions into the Python REPL execution environment.
    This makes custom tools available during code execution.

    Args:
        agent: The STCoscientist agent instance
    """
    custom_functions = getattr(agent, "_custom_functions", {})
    # Remember whose turn this is, for `make_prompt_tool`. Set here rather than captured in a
    # closure: the REPL's namespace persists across turns, so a captured agent would be whichever
    # one happened to run first.
    _CURRENT_AGENT[:] = [agent]
    # The declarative tools this owner has made, injected as callables alongside the one that
    # makes them. Without this the tier was a registration path with NO DISPATCHER: a tool was
    # created, announced as "It is yours and is listed in your tools panel", shown in the panel
    # as `ready`, and could never be invoked -- on that turn or any later one. `declarative.render`
    # had been written for this and had no caller anywhere in the repo.
    #
    # That is the exact failure the tier was built to end. `declarative.py` opens by recording that
    # the env-backed tier was "100% broken" because a failed creation scored as a good action; a
    # creation that succeeds and yields something uncallable scores as a good action too.
    inject_custom_functions_to_repl(
        {
            **custom_functions,
            **_declarative_callables(agent),
            "make_prompt_tool": make_prompt_tool,
        }
    )
    _seed_recovery_helper()
    # Everything above still runs in process mode: the builtins mirror it writes is what the
    # USER prompt's tool-recommendation stage reads, and that prompt must not depend on where the
    # cells run. The worker is configured in addition, from the same set.
    if _process_isolation():
        _inject_into_process_repl(agent, custom_functions)


def _process_isolation() -> bool:
    try:
        from spatialomicsgym.tool.support_tools import process_isolation

        return bool(process_isolation())
    except Exception:
        return False


def _run_shell_in_worker(kind: str, cleaned: str, timeout: float) -> str:
    """One bash or R cell through the worker; every failure is a string the loop can read.

    The backend is whatever ``support_tools`` installed -- the real worker, or ``BrokenBackend``,
    which refuses in words. It is deliberately NOT a fallback to the in-process runner: a portal
    whose boundary could not be built must not answer a shell cell by running it as root.
    """
    try:
        import os as _os

        from spatialomicsgym.tool.repl_client import FORWARDED_ENV
        from spatialomicsgym.tool.support_tools import _backend

        env = {k: _os.environ.get(k) for k in FORWARDED_ENV}
        return str((_backend().exec_shell(kind, cleaned, timeout, env=env) or {}).get("output") or "")
    except Exception as exc:
        return f"Error in execution: {type(exc).__name__}: {exc}"


def _run_cell_in_worker(cleaned: str) -> str:
    """One cell through the worker; every failure is a string the loop reads as a failed step."""
    try:
        return run_python_repl(cleaned)
    except Exception as exc:
        return f"Error in execution: {type(exc).__name__}: {exc}"


def _inject_into_process_repl(agent: Any, custom_functions: dict[str, Any]) -> None:
    """Describe this turn's tools to the REPL worker and bind the cell that is about to run.

    The worker (``tool/repl_host.py``) builds every MCP wrapper itself from the spec ``add_mcp``
    recorded on the function; callables without one are named by module and qualname for the
    worker to import; declarative records travel as data (``render`` is pure); anything that
    needs the server -- the LLM-backed ``make_prompt_tool``, and the brokered creation verbs --
    is offered as an upcall the worker answers by asking us. Never raises: a worker that cannot
    be described must cost the tools, not the turn, and the cell then fails in words.
    """
    try:
        import os as _os

        from spatialomicsgym.agent.env_fallback import recovery_active
        from spatialomicsgym.config import default_config
        from spatialomicsgym.tool import support_tools
        from spatialomicsgym.tool.repl_client import FORWARDED_ENV

        backend = support_tools._backend()
        if backend is None:
            return
        tools: list[dict[str, Any]] = []
        import_tools: list[dict[str, str]] = []
        for name, fn in custom_functions.items():
            spec = getattr(fn, "_sog_spec", None)
            if isinstance(spec, dict):
                tools.append(spec)
                continue
            module = str(getattr(fn, "__module__", "") or "")
            qualname = str(getattr(fn, "__qualname__", "") or "")
            if module and qualname and "<locals>" not in qualname and module != "__main__":
                import_tools.append({"name": str(name), "module": module, "qualname": qualname})
        owner = str(getattr(agent, "_turn_owner", "") or "")
        declarative: list[dict[str, Any]] = []
        try:
            from tools_user import declarative as _declarative

            declarative = [dict(r) for r in _declarative.list_tools(owner)]
        except Exception:
            declarative = []
        brokered = _brokered_upcalls(agent)
        upcalls = {"make_prompt_tool": make_prompt_tool, **brokered}
        backend.set_upcalls(upcalls)
        protected: list[str] = []
        if brokered:
            try:
                from spatialomicsgym.agent import broker as _broker

                protected = _broker.protected_roots()
            except Exception:
                protected = []
        knobs = {
            "timeout_seconds": getattr(agent, "timeout_seconds", None),
            "env_fallback_enabled": getattr(default_config, "env_fallback_enabled", True),
            "unsolved_rescue_enabled": getattr(default_config, "unsolved_rescue_enabled", True),
            "general_env_python": getattr(default_config, "general_env_python", None),
            "general_env_max_calls": getattr(default_config, "general_env_max_calls", 12),
            "benchmarking_enabled": False,  # process mode is off under benchmarking by construction
            # Whether the budget notice may name the knobs (mcp_integration._budget_is_raisable).
            "conversation_memory": bool(getattr(agent, "conversation_memory", False)),
        }
        env = {k: _os.environ.get(k) for k in FORWARDED_ENV}
        # Bound BEFORE the configure, so a configure that raises leaves this turn's binding in
        # place (or none at all, below) rather than the previous turn's (the audit's S8).
        support_tools.bind_process_cell(agent, timeout=getattr(agent, "timeout_seconds", None), env=env)
        reply = backend.configure(
            {
                "knobs": knobs,
                "tools": tools,
                "import_tools": import_tools,
                "declarative": declarative,
                "upcalls": [
                    {"name": n, "doc": str(getattr(f, "__doc__", "") or ""), "params": _param_names(f)}
                    for n, f in upcalls.items()
                ],
                "seed_helper": bool(recovery_active()),
                "owner": owner,
                "env": env,
                # Where an Errno 13 means "ask the server": the worker prints the broker notice
                # the first time a cell is denied under one of these. Empty when nothing is brokered.
                "protected_roots": protected,
            }
        )
        for line in (reply or {}).get("unresolved") or []:
            print(f"Warning: the REPL worker could not bind a tool -- {line}")
    except Exception as exc:
        print(f"Warning: the REPL worker could not be configured for this cell: {exc}")
        try:
            from spatialomicsgym.tool import support_tools as _st

            _st.unbind_process_cell()
        except Exception:
            pass


def _param_names(fn: Any) -> list[str]:
    """The positional-or-keyword parameter names of ``fn``, for the worker's stub to bind by."""
    try:
        import inspect

        return [
            p.name
            for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except Exception:
        return []


def _brokered_upcalls(agent: Any) -> dict[str, Any]:
    """The server-side verbs a worker may ask for, beyond ``make_prompt_tool``. Filled by the
    brokered tool-creation policy (``agent/broker.py``); empty when it is absent."""
    try:
        from spatialomicsgym.agent import broker

        return broker.upcalls_for(agent)
    except Exception:
        return {}


def _seed_recovery_helper() -> None:
    """Bind ``run_in_general_env`` in the REPL namespace while the env-fallback layer is active;
    unbind it otherwise. The namespace is process-global, so a benchmark turn that follows a
    portal turn in one process must not inherit a helper the scored prompt never named.

    Written to the namespace DIRECTLY and never through ``inject_custom_functions_to_repl``: that
    seeder mirrors every name into ``builtins._spatialomicsgym_custom_functions``, which
    ``_registered_tool_names`` reads to build the tool-recommendation stage of the USER prompt --
    so a portal turn's helper reached the enriched prompt of the benchmark turn after it (driven
    2026-09-20: the two prompts differed by one line naming the helper). The mirror is scrubbed of
    the name on every call, for a process that ran the older seeder.
    """
    try:
        import builtins

        from spatialomicsgym.agent.env_fallback import recovery_active
        from spatialomicsgym.tool import general_env
        from spatialomicsgym.tool.support_tools import _default_repl, forget_repl_name

        mirror = getattr(builtins, "_spatialomicsgym_custom_functions", None)
        if isinstance(mirror, dict):
            mirror.pop(general_env.HELPER_NAME, None)
        if recovery_active():
            _default_repl._namespace[general_env.HELPER_NAME] = general_env.run_in_general_env
        else:
            forget_repl_name(general_env.HELPER_NAME)
    except Exception:
        pass


def _python_cell_with_its_own_memo(code: str) -> str:
    """``run_python_repl`` with the duplicate-dispatch memo armed on the thread that runs the cell.

    ``run_with_timeout`` runs the cell in a thread of its own, and the memo only listens to the
    thread that armed it -- which is what keeps a timed-out cell's orphan out of the next cell's
    scope. Armed on the execute node's thread instead, it listened to nobody (u15-validation-1).
    """
    arm_tool_call_memo()
    try:
        return run_python_repl(code)
    finally:
        release_tool_call_memo()


def clear_execution_plots(agent):
    """Clear execution plots before new execution.

    Args:
        agent: The STCoscientist agent instance
    """
    try:
        from spatialomicsgym.tool.support_tools import clear_captured_plots

        clear_captured_plots()
    except Exception as e:
        print(f"Warning: Could not clear execution plots: {e}")


def _declarative_callables(agent: Any) -> dict[str, Any]:
    """This turn's owner's declarative tools, as functions the REPL can call.

    Keyed by tool name, which ``declarative.create`` already forces to be a Python identifier and
    already refuses to let shadow a builtin or an injected name -- ``reserved_names()`` exists for
    exactly this moment, and its docstring says so: a shadowing record "would be a tool the user
    made, sees listed, and can never call".

    ``make_prompt_tool`` is merged AFTER these deliberately. A user cannot create a tool by that
    name (it is reserved), but if the reservation is ever weakened, the injected builtin must still
    win over a record -- losing the ability to make tools is worse than losing one tool.

    Never raises. This runs before every execution step, and a store that cannot be read must cost
    the tools, not the turn.
    """
    owner = ""
    try:
        owner = str(getattr(agent, "_turn_owner", "") or "")
        from tools_user import declarative as _declarative

        records = _declarative.list_tools(owner)
    except Exception:
        return {}

    out: dict[str, Any] = {}
    for record in records:
        name = str(record.get("name") or "")
        if not name.isidentifier():
            continue
        out[name] = _prompt_tool_callable(_declarative, record)
    return out


def _prompt_tool_callable(declarative_mod: Any, record: dict[str, Any]) -> Any:
    """One declarative record, as a callable.

    Returns the filled template as a STRING rather than calling the model with it. The record is a
    prompt the user wrote; handing its text back into the transcript is what makes it usable by the
    agent that asked for it, and it keeps this dispatcher free of a second inference path that
    nothing measures.
    """

    import inspect

    name = str(record.get("name") or "prompt_tool")
    spec = record.get("spec") or {}
    inputs = [str(i) for i in (spec.get("inputs") or []) if str(i).isidentifier()]

    # Positional arguments too, in the order the record lists its inputs. It took `**arguments`
    # only, so `tool("text")` failed with "_prompt_tool_callable.<locals>._call() takes 0 positional
    # arguments", naming neither the tool nor what it takes (u12-react-16).
    def _call(*args: Any, **arguments: Any) -> Any:
        if len(args) > len(inputs):
            return declarative_mod.as_error(
                f"{name} takes {len(inputs)} input(s) ({', '.join(inputs) or 'none'}); {len(args)} were given."
            )
        for key, value in zip(inputs, args, strict=False):
            if key in arguments:
                return declarative_mod.as_error(f"{name} got two values for {key}.")
            arguments[key] = value
        try:
            return declarative_mod.render(record, arguments)
        except Exception as exc:
            # The project's own error shape, so `_EXEC_ERROR_RE` sees a failed tool call as failed
            # instead of the REPL printing a traceback the loop reads as ordinary output.
            return declarative_mod.as_error(str(exc))

    _call.__name__ = name
    _call.__qualname__ = name
    try:
        _call.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter(i, inspect.Parameter.POSITIONAL_OR_KEYWORD) for i in inputs]
        )
    except (TypeError, ValueError):
        pass  # help() falls back to (*args, **arguments); the call itself still works
    _call.__doc__ = (
        f"{record.get('description') or 'A prompt tool you made.'} Takes: {', '.join(inputs) or 'no inputs'}."
    )
    return _call


def make_prompt_tool(name: str, description: str, template: str) -> dict:
    """Create a prompt tool owned by whoever asked for this turn. Returns a result dict.

    Called from an ``<execute>`` block. There is no environment to build and nothing to restart:
    the tool is a validated record, available to this account from the moment this returns.

    The owner is NOT an argument, and cannot be. It is stamped onto the agent by the front door
    before the turn starts -- the same rule every route follows by overwriting ``body["_user"]``
    -- so a model that wanted to file a tool under another account has nothing to write it into.
    Without a stamp the verb refuses: a library user has no accounts to own anything, and
    guessing one would invent an owner out of nothing.

    Failure comes back as ``{"status": "error", "message": ...}``, which is the shape
    ``_EXEC_ERROR_RE`` recognises -- so a refusal reads to the ReAct loop as a failed action and
    it tries something else, rather than scoring a refusal as progress.
    """
    try:
        from tools_user import declarative as _declarative
    except Exception as exc:  # pragma: no cover - only when tools_user is absent entirely
        return {"status": "error", "message": f"tool creation is not available here: {exc}"}

    owner = str(getattr(_CURRENT_AGENT[0], "_turn_owner", "") or "") if _CURRENT_AGENT else ""
    if not owner:
        return {
            "status": "error",
            "message": (
                "There is no account on this turn, so a tool made now would belong to nobody. "
                "Tool creation works from the portal, where the turn carries a signed-in account."
            ),
        }
    try:
        record = _declarative.create(
            owner, name=name, description=description, kind="prompt", spec={"template": template}
        )
    except _declarative.DeclarativeError as exc:
        return {"status": "error", "message": str(exc)}
    return {
        "status": "ok",
        "tool": {"id": record["id"], "name": record["name"], "inputs": record["spec"]["inputs"]},
        "message": (
            f"Created {record['name']}. It takes "
            + (", ".join(record["spec"]["inputs"]) or "no inputs")
            + ". It is yours and is listed in your tools panel."
        ),
    }


#: The agent whose turn is running, for the verb above. A one-slot list rather than a module
#: global rebind, so `inject_custom_functions` can set it without the verb capturing a stale one.
#: Safe because ONE global turn lock serialises turns -- `server.py:_TURN_LOCK`.
_CURRENT_AGENT: list = []
