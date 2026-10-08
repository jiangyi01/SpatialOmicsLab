import os
import re
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# Load .env BEFORE importing spatialomicsgym.config (below): default_config is instantiated at
# import time and snapshots os.environ (SOG_LLM, SOG_SOURCE, ...). If a key lives only in .env,
# it must be in the environment first, or default_config silently falls back to its coded default.
# override=False keeps any real exported shell vars authoritative over the file.
#
# WHICH .env is the install's own (chat_cli.install_env_file: the repo's on a checkout, the instance
# root's off one), never ./.env. Read from the working directory, a dataset folder carrying a .env
# could set SOG_MCP_CONFIG -- which picks the commands the agent spawns -- or SOG_CUSTOM_BASE_URL
# before default_config snapshotted them, for anyone who imported the agent from there (hunt
# 2026-09-30, u16-llm-config-21). chat_cli is imported for the resolver alone: it loads neither
# config nor the agent at module scope, so the order above still holds.
#
# The REPL worker sets SOG_SKIP_DOTENV: a same-user worker (the CLI's) must not re-read the
# provider keys its spawner deliberately left out of its environment; the sog-agent worker could
# not read the file anyway.
#
# And only when the importing process stands where that file is -- which is exactly when the old
# ``./.env`` load fired for it. Loading the install's .env from ANY directory would hand scored
# runs (whose probe imports the agent from a trial directory with no .env, and so loaded nothing)
# the repo .env's keys and SOG_* knobs, moving every benchmark arm run from this tree. So a stray
# .env is never read, and a scored import still reads nothing (2026-09-30, main-loop review).
if not os.environ.get("SOG_SKIP_DOTENV"):
    from spatialomicsgym.chat_cli import install_env_file

    _install_env = install_env_file()
    try:
        _here_is_install = _install_env is not None and Path(".env").resolve() == Path(_install_env).resolve()
    except OSError:
        _here_is_install = False
    if _here_is_install:
        load_dotenv(str(_install_env), override=False)
        print(f"Loaded environment variables from {_install_env}")
        from spatialomicsgym import mirror_legacy_env

        mirror_legacy_env()

from spatialomicsgym import redaction
from spatialomicsgym.agent.conversation import build_conversation_recap, record_turn, summarize_trajectory
from spatialomicsgym.agent.execution import (
    STOP_SEQUENCES,
    clear_execution_plots,
    configure_agent,
    inject_custom_functions,
    parse_tool_calls_from_code_wrapper,
    parse_tool_calls_with_modules_wrapper,
    prepare_resources_for_retrieval,
    update_system_prompt_with_selected_resources,
)
from spatialomicsgym.agent.mcp_integration import add_mcp as _add_mcp
from spatialomicsgym.agent.mcp_integration import create_mcp_server as _create_mcp_server
from spatialomicsgym.agent.mcp_integration import generate_mcp_wrapper_from_spatialomicsgym_schema
from spatialomicsgym.agent.premise_check import check_tissue_premise
from spatialomicsgym.agent.prompt_builder import (
    _before_portal_binding,
    _is_portal_turn,
    _portal_readable,
    enrich_prompt_with_spatial_diagnosis,
    enrich_prompt_with_tool_recommendation,
    enrich_prompt_with_workflow_context,
)
from spatialomicsgym.agent.tool_management import (
    add_data as _add_data,
)
from spatialomicsgym.agent.tool_management import (
    add_software as _add_software,
)
from spatialomicsgym.agent.tool_management import (
    add_tool as _add_tool,
)
from spatialomicsgym.agent.tool_management import (
    filter_know_how_for_commercial_mode,
    get_custom_data,
    get_custom_software,
    get_custom_tool,
    list_custom_data,
    list_custom_software,
    list_custom_tools,
    remove_custom_data,
    remove_custom_software,
    remove_custom_tool,
)
from spatialomicsgym.answer import answer_to_text
from spatialomicsgym.config import default_config
from spatialomicsgym.know_how import KnowHowLoader
from spatialomicsgym.know_how.enrolment import packs_enabled
from spatialomicsgym.llm import SourceType, effective_source, get_llm
from spatialomicsgym.model.retriever import ToolRetriever
from spatialomicsgym.paths import tool_output_root
from spatialomicsgym.tool.tool_registry import ToolRegistry
from spatialomicsgym.utils import (
    check_and_download_s3_files,
    clean_message_content,
    configured_benchmark_mirror,
    convert_markdown_to_pdf,
    create_parsing_error_html,
    find_matching_execution,
    format_execute_tags_in_content,
    format_lists_in_text,
    format_observation_as_terminal,
    has_execution_results,
    pretty_print,
    read_module2api,
    should_skip_message,
)


class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_step: str | None
    #: Why the loop stopped early, or absent when it finished on its own terms.
    #:
    #: The three give-up guards in ``execution.generate`` -- repeated execute errors, think-only
    #: turns, no actionable tag -- end the run by setting ``next_step = "end"`` and appending a
    #: plain ``AIMessage``. They do not raise, so ``_iter_react_stream`` never set
    #: ``degrade_note``, so ``last_turn_degraded`` stayed ``None`` and every caller read the turn
    #: as clean. ``stcoscientist --json`` emitted ``{"answer": "Execution terminated due to
    #: repeated parsing errors..."}`` and **exited 0**, so a batch script, a CI job or a results
    #: pipeline recorded that sentence as the scientific answer. ``_EXIT_DEGRADED`` exists to
    #: prevent exactly that and covered only the exception path.
    #:
    #: This is the fact, carried out of the graph. The answer text is untouched.
    degraded: str | None


# --------------------------------------------------------------------------- #
# ReAct-stream driver + graceful degradation
#
# ``go()`` and ``go_stream()`` both consume ``self.app.stream(...)`` — a LangGraph
# execution that can (a) hit ``recursion_limit`` and raise ``GraphRecursionError``
# after ~1000 paid LLM calls, or (b) surface a transient provider error (429/500/
# network) mid-stream. Either raw exception used to unwind straight to the caller
# (chat_cli / sog-web / a notebook) as a traceback, throwing away the partial
# transcript. These helpers give both methods ONE loop and ONE degradation policy:
# keep the partial transcript and append a plain-language note instead of crashing.
# --------------------------------------------------------------------------- #
def _react_error_note(exc: BaseException) -> str:
    """A short, single-line transcript note for a ReAct loop that stopped early.

    ``GraphRecursionError`` (matched by type *name* so no langgraph import is required) means the
    step budget was exhausted before a final ``<solution>``. Any other exception is treated as a
    transient provider/runtime error: its message is scrubbed (redact-before-clip) and truncated so
    the note stays one readable, ASCII line.

    Scrubbed by :func:`spatialomicsgym.redaction.redact`, the same list the CLI, the web UI and
    the published corpus use. This note used to carry a sixth private pattern that
    knew three of the ten shapes -- and this is the line most likely to contain a credential, since
    the exception it renders is usually the provider's own 401 body. It reaches every front door at
    once, ``--json`` on stdout included.
    """
    if type(exc).__name__ == "GraphRecursionError":
        return (
            "\n[ST-Coscientist: the reasoning loop reached its step budget before producing a "
            "final answer. Returning the partial transcript above.]"
        )
    detail = redaction.redact(str(exc)).replace("\n", " ").strip()
    if len(detail) > 240:
        detail = detail[:240] + "..."
    suffix = f" ({detail})" if detail else ""
    return (
        f"\n[ST-Coscientist: the reasoning loop stopped early after a {type(exc).__name__}{suffix}. "
        "Returning the partial transcript above.]"
    )


def _config_banner_lines(
    *,
    effective: dict[str, Any],
    agent_llm: str,
    agent_source: Any,
    base_url: str | None,
    api_key: str | None,
) -> list[str]:
    """The startup configuration banner, as printable lines.

    Built from ``default_config`` so the database LLM is still on show, but every agent-wide
    setting the constructor can override is replaced by the value actually in force. Printing
    the module default there told a user who passed ``--path``/``--timeout``/``--commercial``
    that their flag had been ignored -- and since ``./data`` exists in a normal checkout, the
    wrong answer looks entirely plausible.

    ``effective`` carries the resolved ``path``, ``timeout_seconds``, ``use_tool_retriever``
    and ``commercial_mode``. The LLM fields stay at their defaults here on purpose: they
    describe the *database* LLM, and an agent-level override is reported by the second block.
    """
    lines = [
        "",
        "=" * 50,
        "🔧 SPATIALOMICSLAB CONFIGURATION",
        "=" * 50,
        "📋 ACTIVE CONFIG (LLM rows describe the default/database LLM):",
    ]

    config_dict = dict(default_config.to_dict())
    config_dict.update(effective)

    for key, value in config_dict.items():
        if value is None:
            continue
        label = key.replace("_", " ").title()
        if key == "commercial_mode":
            mode_text = "Commercial (licensed datasets only)" if value else "Academic (all datasets)"
            lines.append(f"  {label}: {mode_text}")
        elif key == "api_key":
            # Never echo a secret in full — mirror the agent-LLM mask below.
            masked = "*" * 8 + value[-4:] if isinstance(value, str) and len(value) > 8 else "***"
            lines.append(f"  {label}: {masked}")
        else:
            lines.append(f"  {label}: {value}")

    # Show agent-specific LLM if different from default
    if agent_llm != default_config.llm or agent_source != default_config.source:
        lines.extend(["", "🤖 AGENT LLM (Constructor Override):", f"  LLM Model: {agent_llm}"])
        if agent_source is not None:
            lines.append(f"  Source: {agent_source}")
        if base_url is not None:
            lines.append(f"  Base URL: {base_url}")
        if api_key is not None and api_key != "EMPTY":
            lines.append(f"  API Key: {'*' * 8 + api_key[-4:] if len(api_key) > 8 else '***'}")

    lines.extend(["=" * 50, ""])
    return lines


class _StreamResult:
    """Mutable carrier for the terminal message/state a ReAct stream produced, so
    ``_iter_react_stream`` can stay a generator (yielding each step in real time for
    ``go_stream``) while still handing back the final state ``go`` needs for its memory hook."""

    __slots__ = ("message", "final_state", "degrade_note")

    def __init__(self) -> None:
        self.message = None
        self.final_state = None
        self.degrade_note = None


#: What ``STCoscientist._stream_while_running`` tells the review thread it starts, and no other
#: thread: ``stop``, set once that generator's consumer has stopped asking for steps. Thread-local
#: rather than an attribute of the agent because it must outlive a consumer that gave up waiting -- a
#: second Ctrl-C breaks the join -- while never reaching a round that ``go()`` runs on its own thread.
_REVIEW_THREAD = threading.local()


def _is_agent_answer(message) -> bool:
    """Whether ``message`` may be shown to the user as the agent's own output.

    Only agent output may. ``stream_mode="values"`` yields the INITIAL state before the model is
    ever called, whose last message is the user's prompt; accepting it meant a failure on the first
    LLM call (expired key, 400, rate limit) handed the prompt straight back as the answer, with the
    degrade note reachable only in the step log. The same goes for the ``HumanMessage`` nudges
    ``execution.py`` appends mid-loop ("Your last response contained a <think> block but no
    <execute> or <solution> tag...") — prose written to the model, never to the reader.

    Shared by every front door on purpose: the legacy Gradio door (removed 2026-09-30) hand-rolled
    this as ``message.content == text_input``, which recognised the opening prompt and no nudge.
    """
    return not isinstance(message, HumanMessage)


def _final_agent_message(state):
    """The last message in a LangGraph ``state`` that ``_is_agent_answer`` accepts, else ``None``.

    ``state`` may be ``None`` when a stream raised before its first yield, and a turn may legitimately
    contain no agent message at all. Returns the message rather than its text, because callers want
    ``.content`` untouched — on a Responses-API turn that is a list of content blocks.
    """
    for message in reversed((state or {}).get("messages") or []):
        if _is_agent_answer(message):
            return message
    return None


def giveup_note(final_state) -> str | None:
    """Why the loop stopped early, read off the graph's last state -- ``None`` if it did not.

    ONE policy, read by ``_iter_react_stream``, which serves ``go()`` and ``go_stream()`` and so
    every front door: none of them can report an early stop as a clean turn.

    Give-ups do not raise, so the exception channel cannot see them; see ``AgentState.degraded``.
    """
    if not isinstance(final_state, dict):
        return None
    stopped = final_state.get("degraded")
    return str(stopped) if stopped else None


def _merge_round_states(first: Any, second: Any) -> Any:
    """The rescue round's state with the first round's messages in front of its own.

    Both are LangGraph state dicts; everything else in the second wins (it is the newer), and
    the message list is the concatenation -- the task, the first attempt, the rescue prompt, the
    rescue attempt -- which is the transcript that actually happened. Either side that is not a
    state with a message list leaves the other as it is.
    """
    first_msgs = first.get("messages") if isinstance(first, dict) else None
    second_msgs = second.get("messages") if isinstance(second, dict) else None
    if not isinstance(first_msgs, list) or not isinstance(second_msgs, list):
        return second if second is not None else first
    merged = dict(second)
    merged["messages"] = [*first_msgs, *second_msgs]
    return merged


def _iter_react_stream(app, inputs, config, result: "_StreamResult"):
    """Yield each pretty-printed step of ``app.stream(...)`` and fill ``result`` with the terminal
    message/state. On a ``GraphRecursionError`` (step budget) or any transient provider/runtime
    error, record + yield one plain-language note and stop — never propagate. ``KeyboardInterrupt``
    and ``SystemExit`` still propagate."""
    # How much of the transcript has already been emitted. ``stream_mode="values"`` yields once
    # per SUPERSTEP, not once per message, and four ``generate`` branches append **two** messages
    # in one node call: the model's own turn plus a nudge (think-only, no-tags, plan-only) or plus
    # a termination notice (all three give-ups). Reading only ``[-1]`` dropped the first of each
    # pair, so a paid model turn vanished from ``self.log`` -- which is what the CLI, the web UI,
    # ``save_conversation_history()`` and the PDF all render. The reader saw "Your last response
    # contained a <think> block" with no <think> block anywhere before it, and on the give-up
    # paths the thing that disappeared was the model's last code block: the single most useful
    # artifact for working out why the run stopped. The published trajectory corpus is built from
    # these transcripts, so this was a correctness defect in the data, not only in the display.
    emitted = 0
    try:
        for s in app.stream(inputs, stream_mode="values", config=config):
            messages = s["messages"]
            for message in messages[emitted:]:
                out = pretty_print(message)
                if _is_agent_answer(message):
                    result.message = message
                result.final_state = s
                yield out
            emitted = len(messages)
        # A give-up is not an exception, so nothing above notices it. The guards in
        # ``execution.generate`` record their reason on the state instead; fold it in here so a
        # turn that stopped early is never reported as one that finished. Unreachable with a
        # note already set -- an exception leaves this block by the ``except`` below -- but the
        # guard states the precedence rather than leaving it to the control flow.
        if not result.degrade_note:
            result.degrade_note = giveup_note(result.final_state)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # user-facing agent: degrade to the partial answer, don't crash
        note = _react_error_note(exc)
        result.degrade_note = note
        yield note


# A leading conversational qualifier that precedes a conceptual question ("In one sentence, what is
# X?", "Please explain Y", "Briefly, describe Z"). Stripped BEFORE the Layer-1 ``startswith`` check in
# _enrich_prompt_with_question_guard so a prefixed definition question is still caught as conceptual
# instead of falling to the LLM classifier (which — notably gpt-5 on the Azure Responses API — may
# mis-route it to ANALYSIS and demand a dataset). Only the ``startswith`` half uses the stripped text;
# ``has_action`` still scans the whole prompt, so a request that ALSO names an action keyword is never
# turned conceptual — eval ANALYSIS prompts (data paths / run-verbs) are unaffected.
_LEADING_QUALIFIER_RE = re.compile(
    r"\A(?:please|kindly|quickly|briefly|simply|just|hey|hi|"
    r"in (?:one|a|a single|a couple(?: of)?|a few|two|three|1|2|3) (?:sentence|sentences|word|words|line|lines)|"
    r"in short|in brief|to be brief|can you|could you|would you|will you)[\s,:;.\-]+",
    re.IGNORECASE,
)

_H5AD_PATH_RE = re.compile(r"[\w/\-.]+\.h5ad")


def _premise_check_data_path(prompt: str, prior_turns=None) -> str | None:
    """The ``.h5ad`` a turn is about: named in the prompt, else carried over from earlier turns.

    A module-level function rather than a method so it works for any caller -- including an agent
    restored from a pickle and the tests' stand-ins -- and so the session lookup can be exercised
    without building an agent.

    ``prior_turns`` only ever fills when ``conversation_memory`` is on, so a benchmark instance --
    which runs with it off so that instances sharing a process stay independent -- gets the same
    ``None`` the caller saw before this fallback existed.
    """
    match = _H5AD_PATH_RE.search(prompt)
    if match:
        return match.group(0)
    for turn in reversed(list(prior_turns or [])):
        # Question, then answer, then the trajectory digest: the user names the input slide, an
        # answer tends to name derived copies written alongside it, and the digest names everything
        # the run touched. All describe the same tissue, but the input is the file the user is
        # actually asking about, so the most-specific source wins.
        #
        # Indexed rather than unpacked because a turn is a 2-tuple or a 3-tuple depending on when it
        # was recorded (``conversation.record_turn``); a fixed-arity unpack here raised ValueError on
        # the first follow-up of any session that had carried history across the change.
        if not isinstance(turn, (tuple, list)):
            continue
        for text in tuple(turn)[:3]:
            found = _H5AD_PATH_RE.findall(text if isinstance(text, str) else "")
            if found:
                # The FIRST, as for the current prompt (``re.search`` above): "deconvolve
                # /data/st.h5ad with /data/atlas_ref.h5ad" names the slide first and the reference
                # after, and the last one checked the reference -- a different file from turn one's
                # (hunt 2026-09-30, u15-validation-5).
                return found[0]
    return None


# The phrase _enrich_prompt_with_question_guard writes into the prompt when Layer 1 hard-blocks a turn
# as conceptual. Later enrichers that would append action mandates read it back to stay out of the way,
# so the two must be one string rather than two that can drift.
_QUESTION_GUARD_KNOWLEDGE_MARKER = "conceptual/knowledge question"

#: The conversation every caller lands in when it does not ask for another one. This value was
#: hardcoded in three places before it had a name, which is why one process could serve two people
#: and hand the second one the first one's transcript. It stays 42 so that a benchmark and the CLI
#: resume exactly the threads they resumed before.
DEFAULT_THREAD_ID = 42


def _tuning_target(user_text: str, prompt: str, tunable: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The tunable tool to tune: the one named EARLIEST in the user's own words, else in ``prompt``.

    The enriched prompt names P1/P2/P3 by this stage, so scanning it in config order could pick a
    P2/P3 alternative rather than the tool the user asked for (u11-stcoscientist-5).
    """

    def mentioned(text: str) -> dict[str, Any] | None:
        text = text.lower()
        hits = []
        for t in tunable:
            spots = [i for i in (text.find(t["tool_name"]), text.find(t["tool_name"].replace("_", " "))) if i >= 0]
            if spots:
                hits.append((min(spots), t))
        return min(hits, key=lambda h: h[0])[1] if hits else None

    return mentioned(user_text) or mentioned(prompt)


def _checked_timeout(value: Any) -> Any:
    """``value`` when it is a usable per-step budget in seconds, else a ``ValueError`` naming the range."""
    if value is None:
        return value
    from spatialomicsgym.config import NUMERIC_RANGES

    low, high = NUMERIC_RANGES["timeout"]
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"timeout_seconds must be a number of seconds, not {value!r}.") from None
    if not (low <= seconds <= high):
        raise ValueError(f"timeout_seconds must be between {low:g} and {high:g} seconds; got {value!r}.")
    return value


class STCoscientist:
    #: Bound per turn by ``_bind_conversation_thread``. Declared here as well so that an instance
    #: restored from a pickle written before threads had names, or a subclass that reaches the graph
    #: without going through ``go``/``go_stream``, still has one rather than an AttributeError.
    _thread_id = DEFAULT_THREAD_ID

    def __init__(
        self,
        path: str | None = None,
        llm: str | None = None,
        source: SourceType | None = None,
        use_tool_retriever: bool | None = None,
        timeout_seconds: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        commercial_mode: bool | None = None,
        expected_data_lake_files: list | None = None,
        conversation_memory: bool = False,
    ):
        """Initialize the spatialomicsgym agent.

        Args:
            path: Path to the data
            llm: LLM to use for the agent
            source (str): Source provider: "OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", or "Custom"
            use_tool_retriever: If True, use a tool retriever
            timeout_seconds: Timeout for code execution in seconds
            base_url: Base URL for custom model serving (e.g., "http://localhost:8000/v1")
            api_key: API key for the custom LLM
            commercial_mode: If True, excludes datasets that require commercial licenses or are non-commercial only
            conversation_memory: If True, each ``go()``/``go_stream()`` call is told what the earlier
                turns of this session asked and answered, so a follow-up ("redo it with 12 domains")
                does not have to restate the input path. Off by default, and it must stay that way:
                benchmark instances share one process, and a run that inherited a previous instance's
                context would no longer be measuring the instance it claims to. Only the interactive
                front doors (``chat_cli``, ``sog-web``) turn it on.

        """
        # Every secret-named value in the environment enters the redaction registry, so the notes
        # this agent renders (``_react_error_note``) mask a prefix-free key such as an AWS secret.
        # Only the CLI/web loader did this, so the Python API and the benchmark runners showed one
        # raw (hunt 2026-09-30, uL6-parity-9). Name-gated and never raising; see the helper.
        from spatialomicsgym.chat_cli import _register_env_secrets

        _register_env_secrets()
        # Use default_config values for unspecified parameters
        if path is None:
            path = default_config.path
        if llm is None:
            llm = default_config.llm
        # `source` is deliberately NOT pre-filled from default_config. It is forwarded to
        # `get_llm` below as that call's *explicit* `source=` argument, and `resolve_source`
        # returns an explicit source verbatim without ever reading the model name. Pre-filling it
        # therefore handed an environment default to the one lane that cannot weigh it against
        # what the caller asked for: on a stock `.env` (SOG_SOURCE=Anthropic),
        # `STCoscientist(llm="gpt-4o")` built a ChatAnthropic client, and
        # `STCoscientist(llm="azure-gpt-5.4-mini")` built one too — the example `resolve_source`'s
        # own docstring cites as the failure it exists to stop. `get_llm` already guards this
        # (`_config_source_override`); leaving `source` None is what lets that guard run. The CLI
        # and the web UI were fixed in their own layer; this is the Python API's half.
        if use_tool_retriever is None:
            use_tool_retriever = default_config.use_tool_retriever
        if timeout_seconds is None:
            timeout_seconds = default_config.timeout_seconds
        if base_url is None:
            base_url = default_config.base_url
        if api_key is None:
            api_key = default_config.api_key if default_config.api_key else "EMPTY"
        if commercial_mode is None:
            commercial_mode = default_config.commercial_mode

        # Import appropriate env_desc based on commercial_mode
        if commercial_mode:
            from spatialomicsgym.env_desc_cm import data_lake_dict, library_content_dict

            print("🏢 Commercial mode: Using commercial-licensed datasets only")
        else:
            from spatialomicsgym.env_desc import data_lake_dict, library_content_dict

            print("🎓 Academic mode: Using all datasets (including non-commercial)")

        # Store as instance attributes for later use -- COPIES, not the module dicts themselves.
        #
        # ``add_data`` and ``add_software`` write into whatever these names point at, and by
        # reference that is the module-level dict inside ``env_desc`` / ``env_desc_cm``. Measured:
        # after one agent called ``add_data("secret_patient.h5ad", "agent A's private upload")``,
        # a fresh ``from spatialomicsgym.env_desc import data_lake_dict`` saw the entry -- so every
        # agent built later in the same process started with the previous agent's datasets and
        # software in its prompt, including the *description* of an uploaded file, which is the
        # one place a private filename appears. It also crossed the academic/commercial boundary
        # in the sense that whichever module an agent loaded was the one it polluted for every
        # later agent of that mode.
        self.data_lake_dict = dict(data_lake_dict)
        self.library_content_dict = dict(library_content_dict)
        self.commercial_mode = commercial_mode
        # Step log — initialized here so save_conversation_history() works even if it is
        # called before the first go()/go_stream() (which also reset it).
        self.log = []
        self._conversation_state = None  # always defined, so a save before the first run can't AttributeError
        # The last turn's degradation note, or None if it finished. `_iter_react_stream` keeps the
        # partial transcript instead of raising, which is right for a human reading a terminal and
        # invisible to everything else: a caller sees only the returned string, and on the shape
        # where the model had already produced output the note is not even in it. Front doors read
        # this to set an exit status / mark a payload. Defined here so a read before the first turn
        # -- a front door whose `go()` raised, say -- cannot AttributeError.
        self.last_turn_degraded: str | None = None
        # Why the last turn's FIRST attempt stopped, when a rescue round then ran (agent/rescue.py);
        # None when none did. `last_turn_degraded` then describes the rescue's own outcome, so a
        # front door can say both "the first attempt gave up" and "the second finished" -- or not.
        self.last_turn_rescued: str | None = None
        # Tools whose environment failed THIS turn, and why (agent/env_fallback.py). Reset per turn.
        self._env_failures: dict[str, str] = {}
        # Tools told THIS turn to re-run on the CPU after their GPU path failed -- kept out of the
        # "do not call again" ledger above (env_fallback._first_gpu_failure). Reset per turn.
        self._gpu_retry_noticed: set[str] = set()
        # Session recap for follow-up questions. Empty and unused unless conversation_memory is on.
        self.conversation_memory = bool(conversation_memory)
        # (question, answer, trajectory digest). 2-tuples from an older pickle are still read.
        self._prior_turns: list[tuple[str, str, str]] = []

        # The provider this run will actually post to, asked of the same resolver `get_llm` is
        # about to consult — not re-derived here, where it could drift. A name no rule can place
        # falls back to the configured default: `get_llm` raises on it a moment later with a far
        # better message than a half-printed banner would give.
        try:
            agent_source = effective_source(llm, source=source, base_url=base_url, config=default_config)
        except ValueError:
            agent_source = source if source is not None else default_config.source

        # Display configuration in a nice, readable format. The agent-wide settings are reported
        # at the values this constructor resolved, not at the module defaults — see
        # ``_config_banner_lines``.
        for line in _config_banner_lines(
            effective={
                "path": path,
                "timeout_seconds": timeout_seconds,
                "use_tool_retriever": use_tool_retriever,
                "commercial_mode": commercial_mode,
            },
            agent_llm=llm if llm is not None else default_config.llm,
            agent_source=agent_source,
            base_url=base_url,
            api_key=api_key,
        ):
            print(line)

        # Resolved once, here. Kept relative ("./data"), every later use re-resolved it against the
        # CURRENT directory -- and model-written code in the in-process REPL shares that directory,
        # so one `os.chdir` in a cell silently repointed retrieval (the data lake vanished from the
        # prompt) and the post-analysis search (u11-stcoscientist-12). The scored runners already
        # pass an absolute path, so this changes nothing for them.
        if isinstance(path, str) and path:
            path = os.path.abspath(path)
        self.path = path

        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

        # --- Begin custom folder/file checks ---
        benchmark_dir = os.path.join(path, "spatialomicsgym_data", "benchmark")
        data_lake_dir = os.path.join(path, "spatialomicsgym_data", "data_lake")

        # Create the spatialomicsgym_data directory structure
        os.makedirs(benchmark_dir, exist_ok=True)
        os.makedirs(data_lake_dir, exist_ok=True)

        if expected_data_lake_files is None:
            # The legacy Biomni biomedical data-lake download was removed: SpatialOmicsLab never used
            # those files and the S3 bucket that hosted them is gone (every file 404'd on each build).
            # The data_lake/ directory above stays — it's the staging area for the user's real spatial
            # data, surfaced to the agent by a runtime glob (see execution.py), not by a download.
            # Only the benchmark completeness check/download remains below.
            benchmark_ok = False
            if os.path.isdir(benchmark_dir):
                patient_gene_detection_dir = os.path.join(benchmark_dir, "hle")
                if os.path.isdir(patient_gene_detection_dir):
                    benchmark_ok = True

            # Only fetch when someone has actually named a mirror that might answer. The original
            # release bucket is gone, and the "already have it?" marker above is benchmark/hle — a
            # corpus no SpatialOmicsLab user has — so this branch ran on EVERY construction and
            # printed a 404 as the first thing a fresh clone ever saw (offline: after a 10 s stall).
            benchmark_mirror = configured_benchmark_mirror() if not benchmark_ok else None
            if benchmark_mirror:
                print("Checking and downloading benchmark files...")
                check_and_download_s3_files(
                    s3_bucket_url=benchmark_mirror,
                    local_data_lake_path=benchmark_dir,
                    expected_files=[],  # Empty list - will download entire folder
                    folder="benchmark",
                )
        else:
            print("Skipping benchmark download (offline mode).")

        # Post-analysis review keeps searching the root the caller named: users point their
        # prompts (and tools their outputs) at this directory, not at the data subtree below.
        self._user_root = path
        self.path = os.path.join(path, "spatialomicsgym_data")
        module2api = read_module2api()

        self.llm = get_llm(
            llm,
            stop_sequences=list(STOP_SEQUENCES),
            source=source,
            base_url=base_url,
            api_key=api_key,
            config=default_config,
        )
        self.module2api = module2api
        self.use_tool_retriever = use_tool_retriever

        if self.use_tool_retriever:
            self.tool_registry = ToolRegistry(module2api)
            self.retriever = ToolRetriever()

        # Initialize skill registry for MCP tool metadata
        try:
            import sys as _sys
            from pathlib import Path as _Path

            _project_root = str(_Path(__file__).resolve().parents[2])
            if _project_root not in _sys.path:
                _sys.path.insert(0, _project_root)

            from skills import SkillRegistry

            self.skill_registry = SkillRegistry.create_default()
            _all_skill_tools = self._rebuild_skills_context()
            print(
                f"🧬 Loaded {len(_all_skill_tools)} skill tool mappings across {len(self.skill_registry.list_skills())} domains"
            )
        except Exception as _skill_exc:
            # Degrade gracefully on ANY failure (module absent, or a skill mapping missing a key), not
            # just ImportError -- the agent runs fine without the optional skills context; a KeyError
            # from one malformed skill must not crash __init__.
            self.skill_registry = None
            self._skills_context = ""
            print(f"⚠️  Skills registry unavailable ({type(_skill_exc).__name__}), continuing without it")

        # Initialize know-how loader
        self.know_how_loader = KnowHowLoader()

        # Tier 2 -- the merged external packs -- is loaded here and only here, behind a gate that is
        # False by default and False under benchmarking_enabled whatever the configuration says, so
        # the corpus a scored run is measured against is exactly the tier-1 corpus above.
        if packs_enabled():
            try:
                n_packs = self.know_how_loader.load_packs()
            except Exception as _pack_exc:
                n_packs = 0
                print(f"⚠️  Know-how packs unavailable ({type(_pack_exc).__name__}), continuing without them")
            print(f"📦 Loaded {n_packs} tier-2 know-how pack documents (off on every scored run)")

        # Filter know-how documents based on commercial mode
        if commercial_mode:
            self._filter_know_how_for_commercial_mode()

        print(f"📚 Loaded {len(self.know_how_loader.documents)} know-how documents")

        # Add timeout parameter. Held to the same range the env var and the settings panel enforce
        # (config.NUMERIC_RANGES): `--timeout 0` or a negative value made every cell "time out after
        # 0 seconds" while it kept running in a daemon thread, and a huge one overflowed
        # `thread.join` (u11-stcoscientist-6). Refused in words rather than clamped: a budget someone
        # typed is not silently replaced by another.
        self.timeout_seconds = _checked_timeout(timeout_seconds)
        self.configure()

    # --- Tool management (delegated to tool_management module) ---

    def add_tool(self, api):
        """Add a new tool to the agent's tool registry and make it available for retrieval.

        Args:
            api: A callable function to be added as a tool

        """
        return _add_tool(self, api)

    def get_custom_tool(self, name):
        """Get a custom tool by name."""
        return get_custom_tool(self, name)

    def list_custom_tools(self):
        """List all custom tools that have been added."""
        return list_custom_tools(self)

    def remove_custom_tool(self, name):
        """Remove a custom tool."""
        return remove_custom_tool(self, name)

    def add_data(self, data):
        """Add new data to the data lake.

        Args:
            data: Dictionary with file path as key and description as value

        """
        return _add_data(self, data)

    def get_custom_data(self, name):
        """Get a custom data item by name."""
        return get_custom_data(self, name)

    def list_custom_data(self):
        """List all custom data items that have been added."""
        return list_custom_data(self)

    def remove_custom_data(self, name):
        """Remove a custom data item."""
        return remove_custom_data(self, name)

    def add_software(self, software):
        """Add new software to the software library.

        Args:
            software: Dictionary with software name as key and description as value

        """
        return _add_software(self, software)

    def get_custom_software(self, name):
        """Get a custom software item by name."""
        return get_custom_software(self, name)

    def list_custom_software(self):
        """List all custom software items that have been added."""
        return list_custom_software(self)

    def remove_custom_software(self, name):
        """Remove a custom software item."""
        return remove_custom_software(self, name)

    def _filter_know_how_for_commercial_mode(self):
        """Filter out know-how documents that don't allow commercial use."""
        filter_know_how_for_commercial_mode(self)

    # --- MCP integration (delegated to mcp_integration module) ---

    def add_mcp(self, config_path: str | Path | None = None, *, merge_user: bool | None = None) -> None:
        """Add MCP (Model Context Protocol) tools from configuration file.

        If ``merge_user`` (default: ``tool_creation_enabled``), merges the user tools from the
        user MCP config with the original config. On ANY merge failure, falls back to loading
        the original config only, and a failure of that too is recorded, never raised.

        Args:
            config_path: Path to the MCP configuration YAML file. Defaults to the config the
                ``stcoscientist --mcp`` and ``sog-web --mcp`` front doors serve
                (``chat_cli._resolve_mcp_config``): the ``SOG_MCP_CONFIG`` pointer sog-setup
                records, then the setup-generated ``install/recipes/mcp_config.setup.yaml``, then the
                canonical ``agent/MCP_server/mcp_config.yaml`` -- wherever the process runs from. (The
                previous default pointed at ``tutorials/examples/mcp_config.yaml``, which has
                never existed in this repo, so a bare ``add_mcp()`` warned and wired ZERO tools
                while the prompt still told the model its MCP functions were in scope.)
            merge_user: whether to overlay the user-created tools. ``None`` follows
                ``default_config.tool_creation_enabled``; a caller that decides for itself -- a
                benchmark run that must not wire user tools into the scored namespace because the
                operator's shell exported ``SOG_TOOL_CREATION_ENABLED`` -- passes it explicitly.
        """
        # No path given: the config the front doors serve. It was ``find_mcp_config()``, which
        # answers "what does the package ship" -- the canonical list, never verified against this
        # box's envs -- so on a sog-setup'd checkout a notebook's add_mcp() ignored the setup config
        # and wired servers setup had disabled (hunt 2026-09-30, uL6-parity-6). The literal default
        # "MCP_server/mcp_config.yaml" before that was CWD-relative (u11-stcoscientist-11).
        if config_path is None:
            from spatialomicsgym.chat_cli import _resolve_mcp_config
            from spatialomicsgym.mcp_config_path import CANONICAL_CONFIG_DEFAULT, find_mcp_config

            config_path = _resolve_mcp_config("__default__") or find_mcp_config() or CANONICAL_CONFIG_DEFAULT
        # Remember the base config path so reload_user_tools() can re-wire with the same one after a
        # user-tool create/delete/modify (recorded up front so it survives a wiring failure below).
        self._mcp_config_path = str(config_path)
        # Why this wire failed, or None. The failure is caught below so a bad config never takes the
        # agent down, and a caller that took the normal return for success told the user "tools:
        # on" with nothing registered (u17-cli-report-2); the front doors read this instead.
        self._mcp_wiring_error = None
        merged_path = None
        # The caller's own choice, kept for reload_user_tools: the resync re-wires through here, and
        # called bare it fell back to tool_creation_enabled -- which is on exactly when the post-execute
        # resync runs, so a caller's merge_user=False lasted until the first user-tool change (hunt
        # 2026-09-30, uL6-parity-16). ``None`` stays ``None``: a default caller follows the flag live.
        self._mcp_merge_user = merge_user
        if merge_user is None:
            merge_user = default_config.tool_creation_enabled
        merge_user = bool(merge_user)
        try:
            from spatialomicsgym.agent.mcp_config_merger import build_merged_mcp_config, resolve_user_config_path

            merged_path = build_merged_mcp_config(
                original_path=str(config_path),
                # Resolved, not CWD-relative: the front doors hand us an absolute ``config_path``,
                # so without this the base tools loaded and the user's own tools silently did not
                # whenever the agent ran from anywhere but the repo root.
                user_path=resolve_user_config_path(),
                merge_user=merge_user,
            )
            _add_mcp(self, merged_path)
        except Exception as e:
            print(f"WARNING: wiring MCP tools failed ({e}).")
            self._mcp_wiring_error = f"{type(e).__name__}: {e}"
            # Only a genuine USER-tool merge makes the merged config differ from the original, so only
            # then is retrying the original worth it. With merging off (the eval default) the merged
            # config is content-identical to the original -- a retry would re-run the SAME failing/
            # hanging wiring and double the subprocess launches, so skip it.
            if merge_user and merged_path is not None:
                print("Retrying with the original config (user-tool overlay dropped)...")
                # Guarded like the first attempt: a base config that is itself the problem raised out
                # of add_mcp here, only when merging was on, skipping everything below (hunt
                # 2026-09-30, uL6-parity-23). The error stays recorded for the front doors.
                try:
                    _add_mcp(self, config_path)
                    self._mcp_wiring_error = None
                except Exception as retry_exc:
                    print(f"WARNING: wiring the original config failed too ({retry_exc}).")
                    self._mcp_wiring_error = f"{type(retry_exc).__name__}: {retry_exc}"

        # Inject all tool functions into builtins for REPL access
        import builtins

        if not hasattr(builtins, "_spatialomicsgym_custom_functions"):
            builtins._spatialomicsgym_custom_functions = {}
        for name, fn in getattr(self, "_custom_functions", {}).items():
            builtins._spatialomicsgym_custom_functions[name] = fn

        # Register UserToolSkill if user tools exist -- and only when they were wired.
        if merge_user:
            self._register_user_skill()
        # The catalog string was built in __init__; which servers this run hides is decided by the
        # benchmarking flag as it stands at wiring, so rebuild it here (hunt 2026-09-30, u29b-skills-config-1).
        # Called on the class: front doors and tests drive add_mcp on agent stand-ins that carry no
        # skill registry, and the rebuild answers those with {} (it reads the registry by getattr).
        STCoscientist._rebuild_skills_context(self)

        # Snapshot the user-config mtime so the ReAct loop's post-execute hook can detect a
        # create/delete/modify that happened during the session and re-sync (see reload_user_tools).
        import os as _os

        from spatialomicsgym.agent.mcp_config_merger import resolve_user_config_path

        # Same resolution the post-execute watcher uses, or the baseline and the watcher would be
        # reading two different files and the resync would never fire (or fire every turn).
        _up = resolve_user_config_path()
        self._user_config_mtime = _os.path.getmtime(_up) if _os.path.exists(_up) else 0.0

    def reload_user_tools(self, config_path=None):
        """Re-sync live tool catalogs with ``mcp_config_user.yaml`` after a user-tool create/delete/
        modify (which write files/config but never touch in-memory state).

        Drops user tools whose server was removed/disabled and (re)loads current ones — so a trashed
        tool stops being callable, a modified tool picks up its NEW wrapper, and a just-created tool
        becomes callable in the SAME session. Delegates to ``tool_management.resync_user_tools``; uses
        the config path this agent was wired with. Never raises.
        """
        from spatialomicsgym.agent.tool_management import resync_user_tools
        from spatialomicsgym.mcp_config_path import CANONICAL_CONFIG_DEFAULT

        cfg = config_path or getattr(self, "_mcp_config_path", None) or CANONICAL_CONFIG_DEFAULT
        # resync_user_tools re-wires with the merge_user add_mcp recorded (hunt 2026-09-30, uL6-parity-16).
        pruned = resync_user_tools(self, cfg)
        # A tool created during this turn is now callable. Make it *findable* as well: the skill
        # may not have been registered at startup (no install log then), and `_skills_context` is
        # a snapshot either way. Both are idempotent. Gated as add_mcp gates it: by the caller's
        # merge_user when it chose one, else tool_creation_enabled -- a caller that kept user tools
        # out of the namespace must not be offered them in the catalog either.
        try:
            chosen = getattr(self, "_mcp_merge_user", None)
            if default_config.tool_creation_enabled if chosen is None else chosen:
                self._register_user_skill()
        except Exception as exc:
            print(f"WARNING: could not refresh the user-tool skill after resync: {exc}")
        return pruned

    def _rebuild_skills_context(self) -> dict:
        """Rebuild ``_skills_context`` from whatever is registered *now*, and return the tools.

        WHY THIS IS A METHOD AND NOT A LINE IN ``__init__``. ``_skills_context`` is the only
        prompt-facing consumer of the skill registry: ``execution.py:898-904`` folds it in as the
        ``skills_catalog`` retrieval candidate, and nothing else reads the registry on the way to
        the model. It used to be built inline from a ``get_all_tools()`` snapshot taken at
        ``__init__``, roughly 150 lines *before* ``_register_user_skill()`` ran -- so the registry
        gained the user's tool and the string the model sees never did.

        Measured on this box with one active tool in the install log: the registry goes 121 -> 122
        and ``run_mytool`` is absent from ``_skills_context``. The tool was callable and
        undiscoverable, which is the same shape as the declarative tier's registration with no
        dispatcher, one layer up. ``reload_user_tools``'s docstring promises a just-created tool
        "becomes callable in the SAME session"; it was telling the truth and still leaving the
        retriever unable to find it.

        Cheap to call repeatedly: ``SkillRegistry.register`` is keyed on ``skill.name`` and
        ``get_all_tools()`` re-reads each skill's mapping live, so re-running this is idempotent
        and picks up an install log that changed mid-session.

        Never raises, and never blanks a context it failed to rebuild -- a registry that throws
        should cost the *new* entries, not the ones already being offered.
        """
        registry = getattr(self, "skill_registry", None)
        if registry is None:
            return {}
        try:
            tools = registry.get_all_tools()
            # A ``benchmark_visible: false`` server is never bound under benchmarking -- add_mcp
            # skips it, and the transcriptomics listing drops its names -- but this string kept all
            # 26, so a scored run was offered viz and 3D tools it had no callable for (hunt
            # 2026-09-30, u29b-skills-config-1). Read per rebuild, as the listing reads it per call.
            from spatialomicsgym.tool.transcriptomics_skills import _benchmark_hidden_mcp_tools

            hidden = _benchmark_hidden_mcp_tools()
        except Exception as exc:  # a single malformed skill must not empty the catalog
            print(f"WARNING: could not rebuild skills context: {type(exc).__name__}: {exc}")
            return {}
        self._skills_context = "\n".join(
            f"- {v.get('mcp_function')}: {v.get('full_name')} ({v.get('task')}, priority={v.get('priority')})"
            for v in sorted(tools.values(), key=lambda x: (x.get("task", ""), x.get("priority", 0)))
            if v.get("mcp_function") not in hidden
        )
        return tools

    def _register_user_skill(self):
        """Register UserToolSkill with skill_registry if user tools exist.

        Safe to call even if no user tools — silently does nothing.
        """
        from pathlib import Path

        from spatialomicsgym.mcp_user_config import install_log_path

        # Resolved, not CWD-relative. This read `Path("tools_user/install_log.json")`, so an agent
        # started from a data directory -- the normal case -- answered "no user tools", never
        # registered the skill, and every created tool disappeared from the retriever with nothing
        # said. Same class as the documented `.env` CWD bug, and the same ladder the user-config
        # resolver already climbs.
        if not Path(install_log_path()).exists():
            return
        try:
            from tools_user.user_skill import UserToolSkill

            if hasattr(self, "skill_registry") and self.skill_registry is not None:
                self.skill_registry.register(UserToolSkill())
                # Registering is not enough: `_skills_context` was built before this ran.
                self._rebuild_skills_context()
        except Exception as e:
            print(f"WARNING: Failed to register UserToolSkill: {e}")

    def create_mcp_server(self, tool_modules=None):
        """Create an MCP server object that exposes internal SpatialOmicsLab tools.

        Args:
            tool_modules: List of module names to expose (default: all in self.module2api)

        Returns:
            FastMCP server object that you can run manually
        """
        return _create_mcp_server(self, tool_modules)

    def _generate_mcp_wrapper_from_spatialomicsgym_schema(
        self, original_func, func_name, required_params, optional_params
    ):
        """Generate wrapper function based on SpatialOmicsLab schema format."""
        return generate_mcp_wrapper_from_spatialomicsgym_schema(
            original_func, func_name, required_params, optional_params
        )

    # --- Prompt building (delegated to prompt_builder module) ---

    def _generate_system_prompt(
        self,
        tool_desc,
        data_lake_content,
        library_content_list,
        self_critic=False,
        is_retrieval=False,
        custom_tools=None,
        custom_data=None,
        custom_software=None,
        know_how_docs=None,
    ):
        """Generate the system prompt based on the provided resources."""
        from spatialomicsgym.agent.prompt_builder import generate_system_prompt

        return generate_system_prompt(
            self,
            tool_desc=tool_desc,
            data_lake_content=data_lake_content,
            library_content_list=library_content_list,
            self_critic=self_critic,
            is_retrieval=is_retrieval,
            custom_tools=custom_tools,
            custom_data=custom_data,
            custom_software=custom_software,
            know_how_docs=know_how_docs,
        )

    def _enrich_prompt_with_spatial_diagnosis(self, prompt: str) -> str:
        """Detect spatial data paths in the user prompt and inject diagnostic context."""
        return enrich_prompt_with_spatial_diagnosis(prompt)

    def _enrich_prompt_with_premise_check(self, prompt: str) -> str:
        """Contradict a named tissue that the data's most abundant transcripts do not support.

        Handed a human DLPFC slide described as a breast tumour, the agent clustered it, ran DE, and
        reported an ER+ call from an ESR1 enrichment -- ESR1 ranks 5939th of 33538 by counts on that
        file. Every number was real; the premise was not, and nothing in the answer marked it.

        Silent unless the prompt names exactly one tissue AND the file's top transcripts point
        somewhere else by a clear margin, so benchmark prompts (which name no tissue) never see it.

        When the prompt names no file, the session's earlier turns are searched for one. Without that
        fallback the check was live only on turn 1 and dead from turn 2 on: a follow-up never repeats
        the path -- the recap explicitly tells the model to reuse the paths named above rather than
        restate them -- so the regex found nothing and the check returned before looking at anything.
        That is the wrong way round. Turn 1 is where the user has just typed the path and usually the
        tissue with it; the later turns are where they stop pasting paths and start making biological
        claims, which is when a false premise is both likeliest and least visible. (The recap does
        carry the path, but it is prepended *after* this stage runs, so it is not there to be found.)
        """
        # getattr guards an agent restored from a pickle, which predates _prior_turns.
        path = _premise_check_data_path(prompt, getattr(self, "_prior_turns", None))
        # MED-10's confinement, which the two other pre-model readers got: on a portal turn only this
        # account's own data is opened. This reader read any .h5ad the server could, and its note
        # told the caller another account's tissue and top genes (hunt 2026-09-30, uL2-concurrency-4).
        readable = _portal_readable(prompt)
        if path and readable is not None and not readable(path):
            return prompt
        mismatch = check_tissue_premise(prompt, path)
        return f"{prompt}\n\n{mismatch.note()}" if mismatch else prompt

    def _enrich_prompt_with_tool_recommendation(self, prompt: str) -> str:
        """Pre-call recommend_analysis_tools for detected spatial h5ad + inferred goal.

        Injects ranked P1/P2/P3 tools and the canonical MCP invocation pattern into
        the prompt as a deterministic signal — verified 2026-05-11 that
        know-how-only mandates are ignored by gpt-5.4-mini in 3/3 task types.

        Passes down the set of MCP functions that are ACTUALLY callable so the injected
        invocation block cannot promise a function this session never registered.

        Also passes the raw task, because by this point four stages have appended to the prompt
        and one of them names a tool: parameter validation's "Call convert_h5ad_to_csv MCP tool
        first" was being read back as the user's own choice, replacing the analysis they asked
        for. Only the user's own words decide the pin.
        """
        return enrich_prompt_with_tool_recommendation(
            prompt,
            available_tools=self._registered_tool_names(),
            user_text=getattr(self, "user_task", None),
        )

    def _registered_tool_names(self) -> set:
        """Names callable inside ``<execute>`` on THIS agent's next step.

        What ``execution``'s per-step seeder will inject: this agent's wrappers (``add_mcp`` and
        ``add_tool`` both land in ``_custom_functions``), the declarative tools of this turn's owner,
        and ``make_prompt_tool`` once anything is wired. An agent that never wired anything yields
        an empty set -- which is the honest answer, not "unknown".

        It used to add ``builtins._spatialomicsgym_custom_functions``, a process-wide mirror that is
        only ever added to. A tool the operator had disabled stayed "ALREADY in scope" in the prompt
        of the rebuilt agent that no longer had it, and the previous turn's account's private
        declarative tool names were listed in the next account's prompt (hunt 2026-09-30,
        u11-stcoscientist-4).
        """
        names = set(getattr(self, "_custom_functions", {}) or {})
        if not names:
            return names
        try:
            from spatialomicsgym.agent.execution import _declarative_callables

            names |= set(_declarative_callables(self))
        except Exception:
            pass
        names.add("make_prompt_tool")
        return names

    def _enrich_prompt_with_auto_tuning(self, prompt: str) -> str:
        """Hot-pluggable tuning stage — OFF by default, opt-in via config.

        When tuning is DISABLED (default):
          - No tuning imports, no search, no profiling
          - Returns prompt unchanged
          - Logs: "tuning: disabled"

        When tuning is ENABLED (config.tuning_enabled=True):
          - Detects MCP tool + dataset in prompt
          - Routes through mode_router to select the correct mode
          - Injects optimal parameters into the prompt
          - Logs which mode was used and why

        Tuning modes (only active when enabled):
          benchmark_tuning.light  — cached results or official defaults
          benchmark_tuning.full   — cached results or official defaults
          adaptive_tuning         — dataset profiling + heuristic rules
          default_fallback        — official defaults, no search
        """
        # ── Global off switch: tuning is disabled by default ──
        if not default_config.tuning_enabled:
            return prompt

        # ── Per-tool / per-task gating ──
        try:
            import re

            from spatialomicsgym.tuning.core import TuningMode
            from spatialomicsgym.tuning.integration import get_tunable_tools
            from spatialomicsgym.tuning.mode_router import select_mode
            from spatialomicsgym.tuning.parameter_registry import get_light_params

            # Detect which MCP tool is mentioned -- in the USER'S OWN WORDS first. By this stage the
            # recommendation block has named P1/P2/P3 (and a pinned tool's defaults), so scanning the
            # enriched text took the first tunable tool in config order, which could be a P2/P3
            # alternative rather than the tool the user named (u11-stcoscientist-5). The same rule the
            # recommendation stage states: only the user's own words decide the pin. The enriched text
            # is the fallback for a request that named no tool.
            tunable = get_tunable_tools()

            detected_tool = _tuning_target(str(getattr(self, "user_task", "") or ""), prompt, tunable)
            if not detected_tool:
                return prompt

            tool_name = detected_tool["tool_name"]
            task_type = detected_tool["task_type"]

            # Per-tool gating
            if default_config.tuning_tools and tool_name not in default_config.tuning_tools:
                print(f"🔧 Tuning: skipped for {tool_name} (not in tuning_tools list)")
                return prompt

            # Per-task gating
            if default_config.tuning_tasks and task_type not in default_config.tuning_tasks:
                print(f"🔧 Tuning: skipped for task {task_type} (not in tuning_tasks list)")
                return prompt

            # Detect data path (.h5ad)
            h5ad_match = re.search(r"[\w/\-\.]+\.h5ad", prompt)
            data_path = h5ad_match.group(0) if h5ad_match else None

            # Force mode from config if set
            force_mode = None
            if default_config.tuning_mode:
                try:
                    force_mode = TuningMode(default_config.tuning_mode)
                except ValueError:
                    pass

            # ── Mode routing ──
            mode, reason = select_mode(tool_name, task_type, data_path, force_mode=force_mode)

            # ── Get parameters based on selected mode ──
            params: dict = {}
            source = reason

            if mode == TuningMode.DEFAULT_FALLBACK:
                if "cached" in reason.lower():
                    from spatialomicsgym.tuning.persistence import describe_cached_score, load_best_config

                    cached = load_best_config(tool_name)
                    if cached and cached.get("params"):
                        params = cached["params"]
                        # The cache is keyed on the tool name, so this score may have been measured
                        # on other data entirely. It went into the prompt bare, as the reason the
                        # model should use these parameters here.
                        score_note = describe_cached_score(cached, data_path)
                        source = f"cached tuning ({score_note}, mode={cached.get('mode', '?')})"
                else:
                    from spatialomicsgym.tuning.defaults import (
                        get_official_defaults,
                        load_spatialomicsgym_defaults_from_mcp_config,
                    )

                    spatialomicsgym_defaults = load_spatialomicsgym_defaults_from_mcp_config().get(tool_name, {})
                    official = get_official_defaults(tool_name)
                    params = dict(spatialomicsgym_defaults)
                    for k, pv in official.items():
                        params[k] = pv.value
                    source = f"default_fallback: {reason}"

            elif mode in (TuningMode.BENCHMARK_LIGHT, TuningMode.BENCHMARK_FULL):
                from spatialomicsgym.tuning.defaults import (
                    get_official_defaults,
                    load_spatialomicsgym_defaults_from_mcp_config,
                )

                spatialomicsgym_defaults = load_spatialomicsgym_defaults_from_mcp_config().get(tool_name, {})
                official = get_official_defaults(tool_name)
                params = dict(spatialomicsgym_defaults)
                for k, pv in official.items():
                    params[k] = pv.value
                source = f"{mode.value}: official defaults (pre-compute with tune() for better results)"

            elif mode == TuningMode.ADAPTIVE:
                from spatialomicsgym.tuning.adaptive import (
                    apply_adaptive_rules,
                    profile_conditions,
                    profile_dataset,
                )
                from spatialomicsgym.tuning.defaults import (
                    get_official_defaults,
                    load_spatialomicsgym_defaults_from_mcp_config,
                )
                from spatialomicsgym.tuning.strategies import TASK_PRESETS

                spatialomicsgym_defaults = load_spatialomicsgym_defaults_from_mcp_config().get(tool_name, {})
                official = get_official_defaults(tool_name)
                baseline = dict(spatialomicsgym_defaults)
                for k, pv in official.items():
                    baseline[k] = pv.value

                profile = profile_dataset(data_path)
                candidates = apply_adaptive_rules(task_type, baseline, profile)

                # A preset declares the dataset characteristics it was written for. Nothing here
                # scores the candidates -- one is picked by position and injected -- so that
                # declaration is the only thing standing between a Visium run and the preset built
                # for MERFISH. Presets the profile does not fit are not candidates.
                conditions = profile_conditions(profile)
                for preset in TASK_PRESETS.get(task_type, []):
                    if not conditions.intersection(preset.suitable_for):
                        continue
                    cand = dict(baseline)
                    cand.update(preset.params)
                    if cand not in candidates:
                        candidates.append(cand)

                if len(candidates) > 1:
                    params = candidates[1]
                    source = f"adaptive_tuning: {profile.n_spots} spots, {profile.size_category()}"
                else:
                    params = baseline
                    source = "adaptive_tuning: no rules matched, using defaults"

            # ── Inject tunable params only ──
            tunable_names = {p.name for p in get_light_params(tool_name)}
            param_lines = [f"  {k} = {v}" for k, v in params.items() if k in tunable_names]

            if param_lines:
                injection = (
                    f"\n\n[Auto-Tuning | mode={mode.value}] Parameters for {tool_name}:\n"
                    f"  Source: {source}\n"
                    + "\n".join(param_lines)
                    + "\nPlease use these parameter values when calling the tool.\n"
                )
                print(f"🔧 Tuning [ON|{mode.value}]: {len(param_lines)} params for {tool_name} ({source})")
                return prompt + injection

        except Exception as e:
            # Never break the workflow — log and pass through
            print(f"Tuning enrichment skipped: {e}")

        return prompt

    # --- Execution and workflow (delegated to execution module) ---

    def configure(self, self_critic=None, test_time_scale_round=None):
        """Configure the agent with the initial system prompt and workflow.

        Args:
            self_critic: Whether to enable self-critic mode. ``None`` keeps what is already set.
            test_time_scale_round: Number of rounds for test time scaling. ``None`` keeps it.

        Both default to "keep" rather than to off. ``_add_mcp`` and six call sites in
        ``tool_management`` call a bare ``agent.configure()`` purely to re-wire tools, and
        ``maybe_resync_user_tools`` runs one *inside the execute node*, once per turn, whenever
        tool creation is on and the user config changed. With ``self_critic=False`` as the
        default, every one of those silently turned self-critic off and reset the scaling round
        for the rest of the session -- neither of which the caller asked about. Measured:
        ``self_critic`` True before a bare ``configure()``, False after.
        """
        configure_agent(
            self,
            self_critic=getattr(self, "self_critic", False) if self_critic is None else self_critic,
            test_time_scale_round=(
                getattr(self, "test_time_scale_round", 0) if test_time_scale_round is None else test_time_scale_round
            ),
        )

    def _retrieval_query(self, prompt: str) -> str:
        """What the tool retriever ranks against: the follow-up, with the questions it follows.

        The recap that names the session's tool, dataset and task is prepended only as the LAST
        enrichment stage, and retrieval runs first -- so "plot those proportions" was ranked with no
        idea the session was a deconvolution, and the selection then REPLACED the system prompt's
        tools and know-how for that task family (u11-stcoscientist-13). A first turn, and every
        scored run, has no earlier questions and is ranked on its own text exactly as before.
        """
        turns = getattr(self, "_prior_turns", None) or []
        if not getattr(self, "conversation_memory", False) or not turns:
            return prompt
        asked = [str(t[0]) for t in turns[-3:] if t and t[0]]
        if not asked:
            return prompt
        earlier = "\n".join(f"- {q[:400]}" for q in asked)
        return f"Earlier questions in this conversation:\n{earlier}\n\nThe question now: {prompt}"

    def _prepare_resources_for_retrieval(self, prompt):
        """Prepare resources for retrieval and return selected resource names."""
        return prepare_resources_for_retrieval(self, prompt)

    def update_system_prompt_with_selected_resources(self, selected_resources):
        """Update the system prompt with the selected resources."""
        update_system_prompt_with_selected_resources(self, selected_resources)

    def _parse_tool_calls_from_code(self, code: str) -> list[str]:
        """Parse code to detect imported tools by looking for import statements."""
        return parse_tool_calls_from_code_wrapper(self, code)

    def _parse_tool_calls_with_modules(self, code: str) -> list[tuple[str, str]]:
        """Parse code to detect imported tools and their modules."""
        return parse_tool_calls_with_modules_wrapper(self, code)

    def _inject_custom_functions_to_repl(self):
        """Inject custom functions into the Python REPL execution environment."""
        inject_custom_functions(self)

    def _clear_execution_plots(self):
        """Clear execution plots before new execution."""
        clear_execution_plots(self)

    def _enrich_prompt_with_benchmarking(self, prompt: str) -> str:
        """Hot-pluggable benchmarking skill — OFF by default.

        When benchmarking_enabled=True:
          - Injects mandatory input checking instructions
          - Injects mandatory output inspection instructions
          - Injects mandatory evaluation gating instructions
          - Tells the agent to verify at every stage

        When benchmarking_enabled=False (default):
          - Returns prompt unchanged, zero overhead
        """
        if not default_config.benchmarking_enabled:
            return prompt

        eval_note = ""
        if default_config.evaluation_enabled:
            eval_note = (
                "\n== STAGE 3: EVALUATION ==\n"
                "After output inspection passes, evaluate the results:\n"
                "- For spatial_clustering: compute ARI and NMI against ground truth labels\n"
                "- For svg_detection: compute Jaccard/F1 overlap with curated SVG list, "
                "and Moran's I spatial autocorrelation (REQUIRED when spatial coordinates exist)\n"
                "- For deconvolution: compute RMSE, Pearson correlation, and JSD against "
                "ground truth cell type proportions\n"
                "IMPORTANT: Do NOT compute metrics if output inspection found issues. "
                "Only evaluate validated, complete outputs.\n"
            )
        else:
            eval_note = (
                "\n== STAGE 3: NO EVALUATION ==\n"
                "Do NOT run any evaluation or metric computation. "
                "Only run the tool and report the output files produced.\n"
            )

        sep = "=" * 60
        injection = (
            "\n\n" + sep + "\n"
            "[BENCHMARKING MODE: ON — MANDATORY DATA CHECKING]\n" + sep + "\n\n"
            "You MUST follow these three mandatory stages. Do NOT skip any stage.\n\n"
            "== STAGE 1: INPUT CHECKING (before tool execution) ==\n"
            "Before calling the MCP tool, you MUST:\n"
            "1. Load the input h5ad and inspect: shape, obs columns, obsm keys, layers\n"
            "2. Check that obsm['spatial'] exists (required for all spatial tools)\n"
            "3. Check that the tool's required obs columns exist (e.g., cell_type for "
            "deconvolution, communication)\n"
            "4. Check that raw counts are available if the tool requires them "
            "(SVG detection, deconvolution)\n"
            "5. If a single-cell reference is needed (deconvolution), verify it exists "
            "and has cell type annotations\n"
            "6. If obsm['spatial'] is missing but obs has coordinate columns (x, y), "
            "construct it\n"
            "7. If required columns have different names (e.g., 'CellType' instead of "
            "'cell_type'), rename them\n"
            "8. Report the data readiness status BEFORE proceeding to tool execution\n"
            "If input checking fails with unfixable issues, STOP and report the failure. "
            "Do NOT call the tool on invalid data.\n\n"
            "== STAGE 2: OUTPUT INSPECTION (after tool execution) ==\n"
            "After the tool runs, you MUST:\n"
            "1. List ALL files in the output directory (use os.listdir or glob)\n"
            "2. Identify the authoritative prediction output file:\n"
            "   - For clustering: h5ad file with cluster labels in obs (look for columns "
            "like 'spatial_domain', 'domain', 'cluster', 'leiden', 'louvain')\n"
            "   - For SVG: CSV/TSV with gene names and significance values (FDR < 0.05)\n"
            "   - For deconvolution: CSV with cell type proportions (rows=spots, "
            "cols=cell_types)\n"
            "3. Verify the prediction is COMPLETE:\n"
            "   - Not empty (n_predictions > 0)\n"
            "   - Not all NaN or all identical values\n"
            "   - For clustering: >1 unique cluster label\n"
            "   - For SVG: >0 significant genes after filtering blank probes\n"
            "   - For deconvolution: >=2 cell type columns, values sum approximately to 1\n"
            "4. Report: prediction file path, prediction key/column, number of predictions, "
            "number of unique values\n"
            "5. If no valid prediction output is found, report the failure clearly with "
            "the list of files that were checked\n"
            "If output inspection fails, STOP and report the failure. "
            "Do NOT proceed to evaluation.\n"
            f"{eval_note}" + sep + "\n"
        )
        print("Benchmarking [ON]: mandatory 3-stage checking enabled")
        return prompt + injection

    def _enrich_prompt_with_question_guard(self, prompt: str) -> str:
        """Hybrid guard: rule-based fast path + agent-based classification for ambiguous cases.

        Layer 1 (rule-based): catches obvious conceptual questions and obvious action requests.
        Layer 2 (agent-based): for ambiguous cases, injects a classification instruction so
        the LLM decides whether to run tools or just answer.
        """
        # Classified on what the person typed: a portal turn with a dataset bound ends in the binding
        # sentence and its '.h5ad' path, so has_action was always True and "What is Moran's I?" never got
        # the KNOWLEDGE note (hunt 2026-09-30, u13-prompt-extra-19). A scored prompt carries no binding
        # sentence, so it is classified exactly as before.
        prompt_lower = _before_portal_binding(prompt).lower().strip()

        # --- Layer 1: Rule-based fast path ---
        conceptual_starts = (
            "what is",
            "what are",
            "what does",
            "how does",
            "how do",
            "explain",
            "describe",
            "tell me about",
            "define",
            "compare",
            "difference between",
            "why does",
            "why is",
            "can you explain",
            "could you explain",
            "what's the",
        )

        action_keywords = (
            "run ",
            "execute",
            "analyze my",
            "analyze this",
            "perform",
            "apply to",
            "cluster this",
            "deconvolve",
            "detect svg",
            ".h5ad",
            ".csv",
            "/workspace/",
            "/data/",
            "/tmp/",
        )

        # Peel a few stacked leading qualifiers ("please, in one sentence, what is X") so the
        # startswith check below still catches a prefixed conceptual question. has_action stays on the
        # full prompt so a qualifier never masks an action keyword.
        classify_lower = prompt_lower
        for _ in range(4):
            trimmed = _LEADING_QUALIFIER_RE.sub("", classify_lower, count=1).lstrip()
            if trimmed == classify_lower:
                break
            classify_lower = trimmed

        is_conceptual = any(classify_lower.startswith(s) for s in conceptual_starts)
        has_action = any(k in prompt_lower for k in action_keywords)

        if is_conceptual and not has_action:
            # Layer 1 hard block: definitely conceptual
            prompt += (
                f"\n\nNOTE: This is a {_QUESTION_GUARD_KNOWLEDGE_MARKER}. "
                "Answer with explanation and knowledge ONLY. "
                "Do NOT run any MCP tools, do NOT search for datasets, "
                "do NOT execute any code. Just provide a clear, informative answer."
            )
            return prompt

        if has_action or _is_portal_turn(prompt):
            # Layer 1 skip: definitely wants action -- or a portal turn with data bound, which always
            # skipped here and still does unless the person asked a conceptual question
            return prompt

        # --- Layer 2: Ambiguous — let the agent classify ---
        prompt += (
            "\n\nBEFORE taking any action, classify this request into one of:\n"
            "- KNOWLEDGE: User wants explanation or information "
            "-> answer with text only, do NOT run tools\n"
            "- ANALYSIS: User wants computation or results on data "
            "-> proceed with tools\n"
            "- EXPLORATION: User is exploring options "
            "-> describe available tools/datasets, do NOT run tools yet\n"
            "State your classification first, then act accordingly. "
            "If unsure, default to KNOWLEDGE (do not run tools).\n"
        )
        return prompt

    def _enrich_prompt_with_parameter_validation(self, prompt: str) -> str:
        """Auto-detect tool parameter requirements and validate against input data.

        Scans the prompt for data file paths and tool names, then checks each
        tool's parameter schema for mismatches:
        - Format: tool needs CSV but user provided h5ad (or vice versa)
        - Reference: tool needs sc_reference but none provided
        - Image: tool can use histology but none provided

        Injects specific warnings/instructions into the prompt so the LLM
        knows exactly what conversions or data to provide.
        """
        try:
            return self._parameter_validation_inner(prompt)
        except Exception:
            return prompt

    @staticmethod
    def _reference_signal_present(paths, prompt_lower: str) -> bool:
        """True if a single-cell reference is signalled by a data path or the prompt text.

        Path matching is token-aware: a bare ``"ref"`` substring used to match unrelated
        anatomy such as ``prefrontal`` (the DLPFC benchmark) and wrongly suppress the
        missing-reference advisory. Only reference-shaped tokens count.
        """
        for p in paths:
            pl = p.lower()
            if "reference" in pl or "sc_" in pl or "ref_" in pl or "_ref" in pl:
                return True
        return any(
            kw in prompt_lower
            for kw in (
                "reference",
                "sc_ref",
                "sc_h5ad",
                "ref_count",
                "ref_celltype",
                "single-cell",
                "single_cell",
                "scrna",
            )
        )

    def _parameter_validation_inner(self, prompt: str) -> str:
        import re

        # Step 1: Extract file paths from prompt
        path_pattern = r"(/[\w/._-]+\.(?:h5ad|csv|tsv|h5|rds))"
        paths = re.findall(path_pattern, prompt)
        h5ad_paths = [p for p in paths if p.endswith(".h5ad")]
        csv_paths = [p for p in paths if p.endswith((".csv", ".tsv"))]
        has_h5ad = len(h5ad_paths) > 0
        has_csv = len(csv_paths) > 0

        if not paths:
            return prompt  # No data paths found, nothing to validate

        # Step 2: Find tool names mentioned in prompt
        detected_tools = []
        if hasattr(self, "tool_registry") and self.tool_registry:
            for tool in self.tool_registry.tools:
                name = tool.get("name", "") if isinstance(tool, dict) else ""
                if name and name in prompt:
                    detected_tools.append(tool)

        if not detected_tools:
            return prompt  # No specific tool mentioned

        # Step 3: Validate each tool's parameters
        warnings = []
        for tool in detected_tools:
            name = tool.get("name", "")
            params = tool.get("parameters", {})
            if not isinstance(params, dict):
                continue

            param_names = list(params.keys())
            required = [p for p in param_names if params[p].get("required", False)]

            # Check 1: Format mismatch — tool needs CSV but user has h5ad
            csv_params = [p for p in required if "csv" in p.lower() and p != "output_dir"]

            if csv_params and has_h5ad and not has_csv:
                csv_names = ", ".join(csv_params)
                warnings.append(
                    f"FORMAT: {name} requires CSV input ({csv_names}) but you provided h5ad. "
                    f"Call convert_h5ad_to_csv MCP tool first to convert the h5ad file(s) to CSV."
                )

            # Check 2: Missing sc_reference
            ref_params = [
                p
                for p in required
                if any(k in p.lower() for k in ("ref_", "sc_h5ad", "sc_data", "scrna", "ref_counts"))
            ]
            if ref_params:
                # Check if any reference path is in the prompt
                prompt_lower = prompt.lower()
                has_ref = self._reference_signal_present(paths, prompt_lower)
                if not has_ref:
                    ref_names = ", ".join(ref_params)
                    warnings.append(
                        f"REFERENCE: {name} requires single-cell reference data ({ref_names}). "
                        f"Provide the sc_reference h5ad or CSV path."
                    )

            # Check 3: Both spatial + reference CSV needed (deconvolution R tools)
            spatial_csv = [p for p in csv_params if "spatial" in p.lower()]
            ref_csv = [p for p in csv_params if "ref" in p.lower()]
            if spatial_csv and ref_csv and has_h5ad:
                warnings.append(
                    f"CONVERSION: {name} requires BOTH spatial AND reference data as CSV. "
                    f"Call convert_h5ad_to_csv on BOTH the spatial h5ad AND the sc_reference h5ad."
                )

        if not warnings:
            return prompt

        # Inject warnings
        warning_block = "\n\n=== PARAMETER VALIDATION WARNINGS ===\n"
        for w in warnings:
            warning_block += f"- {w}\n"
        warning_block += "=====================================\n"

        return prompt + warning_block

    def _enrich_prompt_with_post_analysis(self, prompt: str) -> str:
        """Inject post-task scan/analysis/visualization workflow.

        Activates when:
          - benchmarking_enabled is False (not in benchmark mode)
          - post_analysis_enabled is True (user hasn't turned it off)
          - the question guard has not already hard-blocked the turn as conceptual

        When active, points STCoscientist at :func:`spatialomicsgym.postanalysis.run_post_analysis`
        and tells it to report the resulting manifest.

        This block used to describe the work instead of naming it: three phases of prose plus a
        scanpy int-category workaround, ~1900 characters re-derived by the LLM on every run, with
        no tested code behind any of it. The scanning, the task-specific analysis, the plotting and
        the data guards now live in ``spatialomicsgym/postanalysis/`` under test, so the prompt only
        has to name the entry point and insist the manifest's warnings reach the user.
        """
        if default_config.benchmarking_enabled:
            return prompt  # Gate 1: SKIP during benchmarking
        if not default_config.post_analysis_enabled:
            return prompt  # Gate 2: SKIP if user turned off
        if _QUESTION_GUARD_KNOWLEDGE_MARKER in prompt:
            # Gate 3: SKIP once the guard has ruled the turn conceptual.
            #
            # `go()` runs question_guard second and this enricher seventh, so without this gate a
            # plain "what is X?" carried both "Do NOT run any MCP tools, do NOT search for datasets,
            # do NOT execute any code" and, fifteen lines below it, "you MUST perform the following
            # 3 phases. Do NOT skip any phase." The mandate was nine times longer than the verdict
            # and came last. Live, that imbalance made the routing a coin flip: the same follow-up
            # question was answered as text with conversation memory off, and re-ran a 187-second
            # pipeline with it on -- the session recap only had to add a little more weight to the
            # action side.
            #
            # The verdict is read back out of the prompt rather than tracked on `self`: a second copy
            # would need clearing every turn and would go stale whenever `_safe_enrich` swallowed a
            # guard exception. No marker means no verdict, not a KNOWLEDGE verdict, so a caller that
            # never ran the guard is unaffected.
            return prompt

        # ASCII only. A Unicode bullet or em-dash in an injected block has repeatedly ended up
        # copied into generated code, where it raises SyntaxError and burns a whole retry loop.
        prompt += (
            "\n\n"
            "=== POST-TASK ANALYSIS WORKFLOW (applies only if you run an analysis tool) ===\n"
            "After an analysis tool writes output files, do not hand-write scanning or\n"
            "plotting code. Call the tested post-analysis engine, in one execute block:\n"
            "\n"
            "    from spatialomicsgym.postanalysis import run_post_analysis\n"
            '    results = run_post_analysis(OUTPUT_DIR, tool_name="<the tool you called>")\n'
            "    print(results)\n"
            "\n"
            "It detects the task type, runs the task-specific analysis, and writes figures/,\n"
            "tables/ and a manifest.json into the directory it returns. It does not raise: a\n"
            'problem is reported in the manifest\'s "warnings" and "status".\n'
            "\n"
            "Then read manifest.json and report what it says: the findings, the figures by\n"
            "title, and every warning verbatim. A warning such as a signal-free, transposed\n"
            "or topic-labelled result means the tool's output is not usable and the user has\n"
            "to hear that.\n"
            "Those figures and that manifest are the PRIMARY DELIVERABLE of an analysis run;\n"
            "raw output files alone are not useful.\n"
            "\n"
            "Write a plot by hand only for something the engine did not produce.\n"
            "\n"
            "End your final answer with one line that begins exactly 'Suggested next step:'\n"
            "followed by ONE follow-up question in plain biology that builds on what this\n"
            "run established. No tool names, no file paths. If the manifest verdict is\n"
            "unusable, do not suggest building on the result: state what is wrong instead.\n"
            "================================================\n"
        )
        return prompt

    def _enrich_prompt_with_workflow_context(self, prompt: str) -> str:
        """Orient the model on the spatial-study arc (qc -> ... -> cell_communication).

        The text itself lives in :func:`prompt_builder.enrich_prompt_with_workflow_context`,
        which owns the text-level gates (spatial context present, not already injected). This
        wrapper owns the config-level gates, the same pair as post-analysis: benchmark prompts
        must stay byte-identical, and a turn the question guard ruled conceptual gets no
        analysis-workflow framing.
        """
        if default_config.benchmarking_enabled:
            return prompt
        if _QUESTION_GUARD_KNOWLEDGE_MARKER in prompt:
            return prompt
        return enrich_prompt_with_workflow_context(prompt)

    # --- Main public API ---

    def _enrich_prompt_with_memory_hints(self, prompt: str) -> str:
        """When memory_enabled=True AND the prompt contains a `GitHub source: <url>`
        line (typical for user-MCP-tool lifecycle prompts), auto-consult the
        memory store and prepend hints to the prompt. This is the
        hard-wired path — does NOT require STCoscientist to voluntarily import MemoryManager.

        Two reads, because the two things live in different files: the per-tool attempt log
        (short-term) for what worked last time, and `known_hang_for` (long-term) for a source a
        previous run had to abandon. The second is what stops an empty attempt log from being
        announced as "a fresh creation" for a tool that is known to hang.

        Stores `self._active_memory_source_url` so the post-run hook can record
        the outcome via `_record_memory_attempt(...)`.
        """
        self._active_memory_source_url = None
        try:
            from spatialomicsgym.config import default_config as _cfg

            if not getattr(_cfg, "memory_enabled", False):
                return prompt
            # Extract `GitHub source: <url>` (case-insensitive). If absent,
            # this isn't a tool-creation prompt; skip silently.
            m = re.search(r"GitHub\s+source\s*:\s*(\S+)", prompt, re.I)
            if not m:
                return prompt
            source_url = m.group(1).rstrip("/.,")
            self._active_memory_source_url = source_url
            from tools_user.memory_manager import (
                OUTCOME_FULL_PASS,
                OUTCOME_HUNG,
                OUTCOME_MOD_ROLLED_BACK,
                OUTCOME_ROLLED_BACK,
                OUTCOME_VENDOR_FALLBACK,
                MemoryManager,
            )

            mm = MemoryManager.get()
            # A hang is recorded in long-term memory, not in the per-tool attempt log, so it needs
            # its own read — and it matters most in exactly the case the attempt log is empty,
            # where "fresh creation" would send the model back into the install the last run had to
            # be killed to escape. add_new_mcp_tool.md tells the agent to record these; this is
            # where the record is spent.
            hang = mm.known_hang_for(source_url) or {}
            hang_block = []
            if hang:
                seen = hang.get("observed_times") or 1
                hang_block = [
                    f"[memory hint] KNOWN HANG for {source_url} (observed {seen}x, last seen "
                    f"{hang.get('last_seen')}): {hang.get('reason') or 'no reason recorded'}",
                    "[memory hint] Do NOT loop on this install. One bounded attempt at most, then "
                    "roll back rather than waiting it out.",
                ]
            hints = mm.read_short_term(source_url) or {}
            attempts = hints.get("attempts") or []
            if not attempts:
                if hang_block:
                    return prompt + "\n\n" + "\n".join(hang_block)
                return prompt + (
                    f"\n\n[memory hint] No prior attempts recorded for {source_url}. "
                    f"This is a fresh creation — record outcome at the end."
                )
            # One definition of "the best attempt" lives on MemoryManager. Resolving the id here as
            # well is how this reader and format_hint_for_prompt came to disagree.
            best = mm.best_attempt(source_url) or {}
            best_outcome = best.get("outcome") or "unknown"
            best_strategy = best.get("strategy") or {}
            best_api = best.get("api_shape") or {}
            gotchas = best.get("gotchas") or []
            # best_attempt RANKS, it does not filter: with no success on file it hands back the most
            # recent FAILURE. So the outcome picks the wording. Three bands and not two, because
            # either binary puts a false label on the middle one -- soft_degraded and
            # modification_passed did pass, just not cleanly, and neither is an unqualified win.
            won = best_outcome in (OUTCOME_FULL_PASS, OUTCOME_VENDOR_FALLBACK)
            failed = best_outcome in (OUTCOME_ROLLED_BACK, OUTCOME_HUNG, OUTCOME_MOD_ROLLED_BACK)
            # A tool can both have hung once and have succeeded since; the hang leads because it is
            # the one that changes what the model should try, not just how.
            hint_block = list(hang_block)
            hint_block.append(f"[memory hint] {len(attempts)} prior attempt(s) recorded for {source_url}.")
            # The freshness half of the reuse gate, applied before the outcome labelling so a stale
            # win is not read as a current one. Empty string when fresh, undateable, or gate off.
            stale_note = mm.staleness_note(best, hints)
            if stale_note:
                hint_block.append(f"[memory hint] {stale_note}")
            if hints.get("unavailable_reason") == "all_attempts_failed":
                # The store derives this at append time; spending it here is the difference between
                # the model reusing a recipe and the model knowing there is no recipe to reuse.
                hint_block.append(
                    "[memory hint] EVERY recorded attempt failed (rolled back or hung). Nothing "
                    "below is a proven recipe -- it is what did NOT work."
                )
            if best_strategy:
                if won:
                    hint_block.append(f"[memory hint] Last winning strategy: {best_strategy}")
                elif failed:
                    hint_block.append(
                        f"[memory hint] Strategy of the last attempt, which did not pass "
                        f"(outcome: {best_outcome}): {best_strategy}"
                    )
                else:
                    hint_block.append(
                        f"[memory hint] Strategy of the last attempt (outcome: {best_outcome}): {best_strategy}"
                    )
            if best_api:
                if won:
                    hint_block.append(f"[memory hint] Verified API shape: {best_api}")
                else:
                    hint_block.append(
                        f"[memory hint] API shape recorded on a {best_outcome} attempt, unverified: {best_api}"
                    )
            if gotchas:
                hint_block.append(f"[memory hint] Past gotchas: {gotchas[:5]}")
            hint_block.append(
                "[memory hint] Use these as advisory inputs; verify against current "
                "Phase 1 discovery before committing to a strategy."
            )
            return prompt + "\n\n" + "\n".join(hint_block)
        except Exception as e:
            # Non-blocking: memory failure must NEVER halt the run.
            print(f"[memory-read warn — non-blocking]: {e}")
            return prompt

    def _record_memory_attempt(self, outcome: str, response: str = "") -> None:
        """Record this `agent.go()` outcome into the memory store. Called from go()
        after the streaming finishes. `outcome` is one of: 'full_pass',
        'rolled_back', 'partial', 'unknown'.

        Not a second record of a turn the know-how already recorded. The lifecycle playbooks
        append the real attempt INSIDE the turn -- outcome, strategy, api_shape, gotchas -- and a
        bare ``{outcome, finished, response_tail}`` appended after it became ``best_attempt``
        (most recent success wins), so the next creation of the same source was offered no
        winning strategy, no verified API shape and no gotchas; every attempt also counted twice
        (hunt 2026-09-30, u11-stcoscientist-3). ``_memory_turn_started`` is when this turn began;
        go() and go_stream() stamp it.
        """
        try:
            from spatialomicsgym.config import default_config as _cfg

            if not getattr(_cfg, "memory_enabled", False):
                return
            source_url = getattr(self, "_active_memory_source_url", None)
            if not source_url:
                return
            # Map the inferred outcome vocab (full_pass/rolled_back/partial/unknown)
            # onto the memory store's VALID_OUTCOMES. 'partial' -> soft_degraded;
            # 'unknown' is not recorded (don't pollute memory with an invalid outcome).
            _outcome_map = {
                "full_pass": "full_pass",
                "vendor_fallback_pass": "vendor_fallback_pass",
                "soft_degraded": "soft_degraded",
                "partial": "soft_degraded",
                "rolled_back": "rolled_back",
                "hung": "hung",
            }
            mapped = _outcome_map.get(outcome)
            if mapped is None:
                return
            outcome = mapped
            from datetime import datetime as _dt

            from tools_user.memory_manager import MemoryManager

            mm = MemoryManager.get()
            recorded_in_turn = False
            turn_started = getattr(self, "_memory_turn_started", None)
            if isinstance(turn_started, (int, float)):
                attempts = (mm.read_short_term(source_url, use_cache=False) or {}).get("attempts") or []
                recorded_in_turn = any(
                    isinstance(a, dict) and mm._parse_ts(str(a.get("finished") or "")) >= turn_started for a in attempts
                )
            if not recorded_in_turn:
                mm.append_attempt(
                    source_url,
                    {
                        "outcome": outcome,
                        "finished": _dt.now().isoformat(),
                        "response_tail": (response or "")[-1000:],
                    },
                )
            # Also drop the .last_op marker so the dev_test driver's
            # verify_mode_gated_markers passes. The marker lives inside the store, so it is
            # derived from the manager's root: the hardcoded relative path it replaced landed
            # in whichever directory the run started from, and never followed a relocated store.
            import json as _json

            _marker = mm.root / ".last_op"
            _marker.parent.mkdir(parents=True, exist_ok=True)
            _marker.write_text(
                _json.dumps(
                    {
                        "ts": _dt.now().isoformat(),
                        "source_url": source_url,
                        "op": f"a1.go.post_run.{outcome}",
                    }
                )
            )
            if recorded_in_turn:
                print(f"[memory] {outcome}: this turn's attempt for {source_url} was already recorded")
            else:
                print(f"[memory] recorded {outcome} attempt for {source_url}")
        except Exception as e:
            print(f"[memory-write warn — non-blocking]: {e}")

    def _infer_outcome_from_response(self, response: str) -> str:
        """Best-effort outcome classification from the response text. Used as the
        argument to `_record_memory_attempt`. Errs on the side of 'unknown' to
        avoid skewing memory data.

        Applies to ALL tools — the classifier is tool-agnostic and parses
        only the terminator markers (`LIFECYCLE_COMPLETE`,
        `LIFECYCLE_HARD_ERROR`, `LIFECYCLE_UNBUILDABLE`) that STCoscientist is
        contracted to emit at the end of any lifecycle response.

        Why pick the LAST occurrence (not the first): STCoscientist frequently embeds
        the terminator strings inside Python `print(...)` statements within
        conditional cleanup code, e.g.
            if not residue and not real_data_fail:
                print("LIFECYCLE_COMPLETE")
            else:
                print("LIFECYCLE_UNBUILDABLE: <reason>")
        Both tokens then appear in the response text. A naïve first-match
        check would falsely classify the run as full_pass even when the
        actually-emitted terminator was UNBUILDABLE. The lifecycle prompt
        contract requires the terminator be the LAST line of the response,
        so we use `rfind` and pick the latest position.
        """
        if not response:
            return "unknown"
        # Look at last 4KB (was 2KB). Some responses have long residue check
        # output before the terminator.
        tail = response[-4096:]
        # Pick the LAST occurrence of any terminator token. The actually-
        # emitted terminator is at the END of the response; code-literal
        # occurrences in print() statements appear earlier.
        positions = [
            (tail.rfind("LIFECYCLE_COMPLETE"), "full_pass"),
            (tail.rfind("LIFECYCLE_HARD_ERROR"), "rolled_back"),
            (tail.rfind("LIFECYCLE_UNBUILDABLE"), "rolled_back"),
        ]
        positions = [(p, o) for p, o in positions if p >= 0]
        if positions:
            positions.sort(reverse=True)  # latest position first
            return positions[0][1]
        # Fall back to keyword heuristics
        if re.search(r"\brolled\s+back\b|rollback\(", tail, re.I):
            return "rolled_back"
        if re.search(r"all\s+(?:steps?|phases?)\s+(?:passed|completed|PASS)", tail, re.I):
            return "full_pass"
        return "unknown"

    def _bind_conversation_thread(self, thread_id) -> None:
        """Point ``_prior_turns`` at the turns belonging to ``thread_id``. Never raises.

        The portal builds ONE agent and serializes every account's turns through it, so without
        this the recap that ``_enrich_prompt_with_conversation_recap`` prepends is whatever the
        *previous* person asked -- their question, their answer, and the digest of the tools their
        turn ran. ``server.py`` said so in its own words long before there was a fix: *"two signed-in
        people share one context."*

        ``_prior_turns`` keeps its type. It is still the list every existing reader expects, and is
        simply rebound to this thread's bucket; the buckets live beside it. That matters because an
        agent restored from a pickle predates both attributes, a benchmark instance never calls this
        at all, and ``go()`` keeps writing wherever the last binding pointed -- so nothing outside a
        multi-user caller changes shape or behaviour.

        The default thread *adopts* whatever history the instance already had, rather than starting
        empty: an agent that has been answering through ``go()`` must not lose its recap the first
        time some caller names a thread explicitly.
        """
        self._thread_id = thread_id
        if not getattr(self, "conversation_memory", False):
            return
        try:
            buckets = getattr(self, "_prior_turns_by_thread", None)
            if not isinstance(buckets, dict):
                buckets = {}
                self._prior_turns_by_thread = buckets
            key = str(thread_id)
            if key not in buckets:
                current = getattr(self, "_prior_turns", None)
                adopt = isinstance(current, list) and key == str(DEFAULT_THREAD_ID)
                buckets[key] = current if adopt else []
            self._prior_turns = buckets[key]
        except Exception as exc:  # a recap is a convenience; never fail a run over its bookkeeping
            print(f"[stcoscientist] conversation thread not bound ({type(exc).__name__})")

    def forget_conversation(self, thread_id=None) -> None:
        """Start a fresh conversation: drop the recap earlier turns left behind. Never raises.

        ``thread_id=None`` forgets *every* conversation this agent holds, which is what a
        single-user front door -- the CLI's ``/reset``, a portal with nobody signed in -- means by
        "new chat". Naming a thread forgets only that one, so one researcher pressing "new chat"
        does not wipe the session of whoever else is signed in.

        The lists are emptied **in place**, not replaced. Replacing them clears only the attribute:
        the per-thread bucket goes on holding the old list, and the very next turn rebinds the
        attribute straight back to it -- a "new chat" button that reports success and forgets
        nothing.
        """
        try:
            buckets = getattr(self, "_prior_turns_by_thread", None)
            if thread_id is None:
                for bucket in list(buckets.values()) if isinstance(buckets, dict) else []:
                    if isinstance(bucket, list):
                        bucket.clear()
                if isinstance(getattr(self, "_prior_turns", None), list):
                    self._prior_turns.clear()
                return
            key = str(thread_id)
            target = buckets.get(key) if isinstance(buckets, dict) else None
            if isinstance(target, list):
                target.clear()
            elif key == str(getattr(self, "_thread_id", DEFAULT_THREAD_ID)) and isinstance(
                getattr(self, "_prior_turns", None), list
            ):
                # The named thread is the one currently bound but has no bucket yet -- the agent has
                # been answering through the plain ``go()`` path, which never split its history.
                self._prior_turns.clear()
        except Exception as exc:  # clearing a convenience must not fail the request that asked
            print(f"[stcoscientist] conversation not cleared ({type(exc).__name__})")

    def restore_conversation(self, thread_id, turns) -> int:
        """Refill one thread's recap from turns that were stored elsewhere. Never raises.

        The recap lives in memory, so a restart -- or ``AgentHandle._stalled()`` rebuilding the
        agent under a live session -- silently empties a conversation the user can still see on
        screen, and the next turn answers as if the earlier ones never happened. The portal now
        keeps its conversations on disk (:mod:`sog_portal.services.conversations`) and calls this
        when it opens one whose in-memory bucket is empty but whose file is not.

        ``turns`` is any iterable of ``(question, answer, trajectory)`` -- the shape
        ``conversation.record_turn`` produces and ``conversations.recap_turns`` hands back. Each is
        replayed through ``record_turn`` rather than assigned, so its rules (no answerless turn, the
        bound on stored history) apply identically to a restored conversation and a live one.

        The bucket is refilled **in place**, for the same reason ``forget_conversation`` empties in
        place: ``_prior_turns`` is an alias for the bucket, and replacing either one leaves the
        other pointing at the list nobody is writing to. Returns how many turns were restored.

        A conversation that already has turns in memory is left alone: the RAM copy is the live one,
        and replaying over it would double every turn the recap mentions.
        """
        try:
            self._bind_conversation_thread(thread_id)
            if not getattr(self, "conversation_memory", False):
                return 0
            bucket = getattr(self, "_prior_turns", None)
            if not isinstance(bucket, list) or bucket:
                return 0
            restored = 0
            for turn in turns or []:
                try:
                    question = turn[0] if len(turn) > 0 else ""
                    answer = turn[1] if len(turn) > 1 else ""
                    trajectory = turn[2] if len(turn) > 2 else ""
                except (TypeError, IndexError):
                    continue
                before = len(bucket)
                record_turn(bucket, question, answer, trajectory)
                restored += len(bucket) > before
            return restored
        except Exception as exc:  # a recap is a convenience; never fail a run over its bookkeeping
            print(f"[stcoscientist] conversation not restored ({type(exc).__name__})")
            return 0

    def _enrich_prompt_with_conversation_recap(self, prompt: str) -> str:
        """Prepend what earlier turns of this session asked and answered.

        Returns the prompt unchanged unless ``conversation_memory`` was requested AND this session
        already has a completed turn — so a benchmark run, and the first turn of any session, are
        byte-identical to the behaviour before session recaps existed.

        ``getattr`` guards rather than direct attribute reads: an agent restored from a pickle, or a
        subclass that does not chain ``super().__init__``, predates these attributes and must keep
        running.
        """
        if not getattr(self, "conversation_memory", False):
            return prompt
        recap = build_conversation_recap(getattr(self, "_prior_turns", None) or [])
        return recap + prompt if recap else prompt

    def _record_conversation_turn(self, question: str, answer) -> None:
        """Remember a completed turn so the next one can refer back to it.

        Records the user's *original* question, not the enriched prompt: replaying enrichment (tool
        recommendations, spatial diagnosis, a previous recap) would compound every turn until the
        preamble buried the actual question.

        Alongside the question and answer goes a digest of what the turn actually *did*, mined from
        ``self.log``. The answer alone is written for a person and often names neither the tool nor
        the output directory, which left a follow-up ("plot those proportions") with nothing to act
        on but a search.
        """
        if not getattr(self, "conversation_memory", False):
            return
        try:
            if not isinstance(getattr(self, "_prior_turns", None), list):
                self._prior_turns = []
            record_turn(self._prior_turns, question, answer, summarize_trajectory(getattr(self, "log", None)))
        except Exception as exc:  # a recap is a convenience; never fail a completed run over it
            print(f"[stcoscientist] session recap not updated ({type(exc).__name__})")

    @staticmethod
    def _safe_enrich(stage, enrich_fn, prompt):
        """Apply one prompt-enrichment stage, degrading to the un-enriched prompt on failure.

        The enrichers (question guard, parameter validation, spatial diagnosis, tool recommendation,
        auto-tuning, ...) are best-effort augmentations layered onto the user's prompt. A bug in any
        one of them — an unreadable path token, a degenerate obs column, a retriever hiccup — must
        never take down a live ``go()``/``go_stream()`` run, so a raising stage is skipped and the
        prompt passes through unchanged. The happy path is byte-identical to calling
        ``enrich_fn(prompt)`` directly; only a *crashing* stage changes behaviour (skip vs. raise).
        ``KeyboardInterrupt``/``SystemExit`` are deliberately not caught.
        """
        try:
            return enrich_fn(prompt)
        except Exception as exc:  # best-effort enrichment stage must not crash the run
            try:
                print(f"[stcoscientist] prompt-enrichment stage '{stage}' skipped ({type(exc).__name__})")
            except Exception:
                pass
            return prompt

    def go(self, prompt, thread_id=DEFAULT_THREAD_ID):
        """Execute the agent with the given prompt.

        Args:
            prompt: The user's query
            thread_id: Which conversation this turn belongs to; see :meth:`go_stream`. The default
                is the same one this method has always used, so an existing caller is unaffected.

        """
        self.critic_count = 0
        self.user_task = prompt
        self._bind_conversation_thread(thread_id)

        if self.use_tool_retriever:
            selected_resources_names = self._prepare_resources_for_retrieval(self._retrieval_query(prompt))
            self.update_system_prompt_with_selected_resources(selected_resources_names)

        # Memory hints: when memory_enabled, auto-consult MemoryManager and
        # prepend prior-attempt hints. Hard-wired — no know-how compliance
        # required.
        prompt = self._safe_enrich("memory_hints", self._enrich_prompt_with_memory_hints, prompt)

        # Question guard: block conceptual questions from triggering tools
        enriched_prompt = self._safe_enrich("question_guard", self._enrich_prompt_with_question_guard, prompt)

        # Parameter validation: auto-detect format mismatches, missing references
        enriched_prompt = self._safe_enrich(
            "parameter_validation", self._enrich_prompt_with_parameter_validation, enriched_prompt
        )

        # Auto-diagnose spatial data paths found in the user prompt
        enriched_prompt = self._safe_enrich(
            "spatial_diagnosis", self._enrich_prompt_with_spatial_diagnosis, enriched_prompt
        )

        # Pre-call recommend_analysis_tools and inject ranked P1/P2/P3 + canonical
        # MCP invocation pattern. Doc-level mandates were verified insufficient
        # (3/3 fail) on 2026-05-11; this is the deterministic signal.
        enriched_prompt = self._safe_enrich(
            "tool_recommendation", self._enrich_prompt_with_tool_recommendation, enriched_prompt
        )

        # Workflow context: place the request on the qc -> ... -> cell_communication arc.
        # After tool_recommendation (orientation must not outrank the concrete tool signal),
        # before benchmarking/post_analysis; skipped entirely in benchmark mode.
        enriched_prompt = self._safe_enrich(
            "workflow_context", self._enrich_prompt_with_workflow_context, enriched_prompt
        )

        # Auto-tune: inject optimal parameters for detected MCP tools
        enriched_prompt = self._safe_enrich("auto_tuning", self._enrich_prompt_with_auto_tuning, enriched_prompt)

        # Benchmarking mode: inject mandatory output inspection instructions
        enriched_prompt = self._safe_enrich("benchmarking", self._enrich_prompt_with_benchmarking, enriched_prompt)

        # Post-task analysis: scan/analyze/visualize (skipped during benchmarking)
        enriched_prompt = self._safe_enrich("post_analysis", self._enrich_prompt_with_post_analysis, enriched_prompt)

        # Premise check: contradict a named tissue the file's own top transcripts refute. Deliberately
        # last of the appending stages so the warning is the final thing the model reads -- a data
        # contradiction buried above the tool block is exactly what got skimmed before.
        enriched_prompt = self._safe_enrich("premise_check", self._enrich_prompt_with_premise_check, enriched_prompt)

        # Session recap LAST, so it sits at the very top of the prompt and the user's own question
        # stays the final line. No-op unless conversation_memory is on and a turn already completed.
        enriched_prompt = self._safe_enrich(
            "conversation_recap", self._enrich_prompt_with_conversation_recap, enriched_prompt
        )

        inputs = {"messages": [HumanMessage(content=enriched_prompt)], "next_step": None}
        config = {"recursion_limit": 1000, "configurable": {"thread_id": self._thread_id}}
        self.log = []
        # Cleared with the log, and for the same reason: both are THIS turn's record. It was
        # created once and appended to forever, and each entry carries base64 PNGs of every
        # figure its cell drew plus ~4 KB of source -- so a long-lived `sog-web` process grew
        # monotonically with conversation volume. It also made `find_matching_execution` match
        # across turns: it returns the FIRST substring match over the whole list, so the first
        # turn that ran `<execute>print(adata)</execute>` owned that text for the life of the
        # process and a later identical block was rendered with the older turn's figures.
        # Scoped to one turn, both problems are gone. The portal's growth cursor reads
        # `_execution_seq`, which is monotonic and deliberately NOT cleared here.
        self._execution_results = []
        self.last_turn_degraded = None  # cleared here too, so a raise below cannot report the PRIOR turn
        self._reset_recovery_state()  # the env notices' once-per-tool ledger and the helper's call budget
        _post_analysis_started = time.time()  # L2 only reviews a manifest written during THIS turn
        self._memory_turn_started = _post_analysis_started  # the memory hook's "this turn" too
        self._turn_started_at = _post_analysis_started  # read by execution.pre_answer_check (C6)
        self._pre_answer_analysed_at = None  # set by that check; one left by an earlier turn is not this turn's
        self._conversation_state = (
            None  # reset at run start so an interrupted run can't later serialize a PRIOR transcript
        )

        # Consume the ReAct stream through the shared driver: a step-budget exhaustion
        # (GraphRecursionError) or a transient provider error degrades to the partial
        # transcript + a note instead of a raw traceback (see _iter_react_stream).
        result = _StreamResult()
        for out in _iter_react_stream(self.app, inputs, config, result):
            self.log.append(out)

        message = result.message
        # Carry the fact, not the prose. On the shape where the model had already answered before
        # the provider failed, `message` is that partial answer and the note is returned nowhere --
        # it stays in `self.log`, which every pipe contract suppresses. A caller that wants to know
        # whether this turn finished has nothing to read but the note's own wording otherwise.
        self.last_turn_degraded = result.degrade_note
        # Store the conversation state for markdown generation
        self._conversation_state = result.final_state

        # L2 self-review + bounded follow-on analysis. No-op under benchmarking (see
        # postanalysis.next_step.post_analysis_active) and whenever this turn wrote no manifest.
        _log_before_review = len(self.log)
        self._run_post_analysis_review(_post_analysis_started, follow_on=not self._rescue_pending(result))
        # One rescue round for a turn that ended unsolved (agent/rescue.py), sharing the follow-on
        # budget above: a turn that already ran a second round gets no third. No-op under
        # benchmarking. When it runs, `result` carries the round's answer and outcome from here on.
        for _ in self._run_unsolved_rescue(result, followup_ran=len(self.log) > _log_before_review):
            pass
        message = result.message

        # Memory write: hard-wired post-run hook. Records the outcome to
        # MemoryManager + writes the .last_op marker. No-op when memory_enabled=False. After the
        # rescue on purpose: a turn the rescue solved is remembered as solved, not as its give-up.
        try:
            if message is not None:
                _final_text = message.content if hasattr(message, "content") else str(message)
            else:
                _final_text = (result.degrade_note or "").strip()
            _outcome = self._infer_outcome_from_response(_final_text)
            self._record_memory_attempt(_outcome, _final_text)
        except Exception as _e:
            print(f"[memory post-run warn — non-blocking]: {_e}")

        if message is not None:
            # Recorded for the recap only when the turn FINISHED. ``message`` is the last agent
            # message whatever it says, so a give-up ("Execution terminated: the same error
            # repeated") or a rescue round's own give-up is a message too -- and was being handed
            # to the next turn as the answer to build on (driven 2026-09-20).
            if not result.degrade_note:
                self._record_conversation_turn(self.user_task, message.content)
            return self.log, message.content
        # A degraded run records nothing: record_turn drops empty answers, and a bare degradation
        # note ("the reasoning loop stopped early...") is not something a follow-up should build on.
        return self.log, (result.degrade_note or "").strip()

    def go_stream(self, prompt, thread_id=DEFAULT_THREAD_ID) -> Generator[dict, None, None]:
        """Execute the agent with the given prompt and return a generator that yields each step.

        Args:
            prompt: The user's query
            thread_id: Which conversation this turn belongs to. The default is unchanged, so every
                existing caller -- benchmarks, the CLI, the eval harnesses -- is byte-identical. A
                caller serving more than one person passes a value per person, which separates
                *both* halves of the conversation: LangGraph's checkpointed message history, and
                this agent's own session recap. Separating only the first is
                worse than separating neither, because the recap would then quote one user's work
                into another user's prompt while the history looked clean.

        Yields:
            dict: Each step of the agent's execution containing the current message and state
        """
        self.critic_count = 0
        self.user_task = prompt
        self._bind_conversation_thread(thread_id)

        if self.use_tool_retriever:
            selected_resources_names = self._prepare_resources_for_retrieval(self._retrieval_query(prompt))
            self.update_system_prompt_with_selected_resources(selected_resources_names)

        # Memory hints: when memory_enabled, auto-consult MemoryManager and prepend
        # prior-attempt hints. Hard-wired — no know-how compliance required. Parity with
        # go(): go_stream (the streaming / web-SSE entry) previously skipped this,
        # so every streamed run silently lost memory-assisted planning.
        prompt = self._safe_enrich("memory_hints", self._enrich_prompt_with_memory_hints, prompt)

        # Question guard: block conceptual questions from triggering tools
        enriched_prompt = self._safe_enrich("question_guard", self._enrich_prompt_with_question_guard, prompt)

        # Parameter validation: auto-detect format mismatches, missing references
        enriched_prompt = self._safe_enrich(
            "parameter_validation", self._enrich_prompt_with_parameter_validation, enriched_prompt
        )

        # Auto-diagnose spatial data paths found in the user prompt
        enriched_prompt = self._safe_enrich(
            "spatial_diagnosis", self._enrich_prompt_with_spatial_diagnosis, enriched_prompt
        )

        # Pre-call recommend_analysis_tools and inject ranked P1/P2/P3 + canonical
        # MCP invocation pattern. Doc-level mandates were verified insufficient
        # (3/3 fail) on 2026-05-11; this is the deterministic signal.
        enriched_prompt = self._safe_enrich(
            "tool_recommendation", self._enrich_prompt_with_tool_recommendation, enriched_prompt
        )

        # Workflow context: place the request on the qc -> ... -> cell_communication arc.
        # After tool_recommendation (orientation must not outrank the concrete tool signal),
        # before benchmarking/post_analysis; skipped entirely in benchmark mode.
        enriched_prompt = self._safe_enrich(
            "workflow_context", self._enrich_prompt_with_workflow_context, enriched_prompt
        )

        # Auto-tune: inject optimal parameters for detected MCP tools
        enriched_prompt = self._safe_enrich("auto_tuning", self._enrich_prompt_with_auto_tuning, enriched_prompt)

        # Benchmarking mode: inject mandatory output inspection instructions
        enriched_prompt = self._safe_enrich("benchmarking", self._enrich_prompt_with_benchmarking, enriched_prompt)

        # Post-task analysis: scan/analyze/visualize (skipped during benchmarking)
        enriched_prompt = self._safe_enrich("post_analysis", self._enrich_prompt_with_post_analysis, enriched_prompt)

        # Premise check: contradict a named tissue the file's own top transcripts refute. Deliberately
        # last of the appending stages so the warning is the final thing the model reads -- a data
        # contradiction buried above the tool block is exactly what got skimmed before.
        enriched_prompt = self._safe_enrich("premise_check", self._enrich_prompt_with_premise_check, enriched_prompt)

        # Session recap LAST — same as go(). The web UI relies on this path for its multi-turn chat.
        enriched_prompt = self._safe_enrich(
            "conversation_recap", self._enrich_prompt_with_conversation_recap, enriched_prompt
        )

        inputs = {"messages": [HumanMessage(content=enriched_prompt)], "next_step": None}
        config = {"recursion_limit": 1000, "configurable": {"thread_id": self._thread_id}}
        self.log = []
        # Cleared with the log, and for the same reason: both are THIS turn's record. It was
        # created once and appended to forever, and each entry carries base64 PNGs of every
        # figure its cell drew plus ~4 KB of source -- so a long-lived `sog-web` process grew
        # monotonically with conversation volume. It also made `find_matching_execution` match
        # across turns: it returns the FIRST substring match over the whole list, so the first
        # turn that ran `<execute>print(adata)</execute>` owned that text for the life of the
        # process and a later identical block was rendered with the older turn's figures.
        # Scoped to one turn, both problems are gone. The portal's growth cursor reads
        # `_execution_seq`, which is monotonic and deliberately NOT cleared here.
        self._execution_results = []
        self.last_turn_degraded = None  # cleared here too, so a raise below cannot report the PRIOR turn
        self._reset_recovery_state()  # the env notices' once-per-tool ledger and the helper's call budget
        _post_analysis_started = time.time()  # L2 only reviews a manifest written during THIS turn
        self._memory_turn_started = _post_analysis_started  # the memory hook's "this turn" too
        self._turn_started_at = _post_analysis_started  # read by execution.pre_answer_check (C6)
        self._pre_answer_analysed_at = None  # set by that check; one left by an earlier turn is not this turn's
        self._conversation_state = None  # reset at run start; an early client disconnect must not leave a stale state

        # Same single loop + degradation policy as go(): stream each step to the caller,
        # and on a step-budget / transient provider error yield one final note instead of
        # letting a raw traceback break the SSE stream mid-response.
        result = _StreamResult()
        for out in _iter_react_stream(self.app, inputs, config, result):
            self.log.append(out)
            yield {"output": out}

        # Store the conversation state for markdown generation
        self._conversation_state = result.final_state
        self.last_turn_degraded = result.degrade_note  # parity with go(): see the note there

        # L2 self-review + bounded follow-on analysis. Parity with go(). The review runs on a thread
        # (``_stream_while_running``) and each step its follow-on round appends to self.log is
        # handed to the caller as the round runs. Until 2026-09-25 they arrived only once the round
        # had finished: measured on 2026-09-15, an answer streamed at minute 11 of a turn and the
        # round then ran for a further 15+ minutes with nothing on the wire. The round starts each
        # step only when the caller asks for the next chunk, as the first round above does, so a
        # caller that stops asking -- close(), Ctrl-C, the portal's pump once Stop is pressed --
        # stops it. That method's docstring says which step can still run after that, and what
        # else the thread finishes before it is joined.
        #
        # ``after_answer`` says these arrive once the answer is already fixed. Only the producer
        # knows that, and a consumer picking the answer with "last chunk wins" cannot guess it: the
        # web chat used to show the follow-on round's own report as the user's answer while the
        # recap recorded the real one, so the next turn's memory contradicted what the user read.
        _log_before = len(self.log)
        yield from self._stream_post_analysis_review(
            _post_analysis_started, _log_before, follow_on=not self._rescue_pending(result)
        )

        # The ONE rescue round for a turn that ended unsolved (agent/rescue.py). It streams live
        # too, on this thread rather than a worker, and its chunks are tagged ``rescue``, NOT
        # ``after_answer``: the first attempt produced no answer, so the round's own solution is
        # the answer and a consumer picking "last chunk wins" must be allowed to adopt it. An
        # afterword tag here would render the turn as an empty answer over a complete trace.
        for _out in self._run_unsolved_rescue(result, followup_ran=len(self.log) > _log_before):
            yield {"output": _out, "rescue": True}

        # Memory write: same hard-wired post-run hook as go() — records the outcome to
        # MemoryManager + writes the .last_op marker. No-op when memory_enabled=False.
        # Parity with go(): without this a streamed run enriched with memory hints (above)
        # never recorded its own outcome, so the memory only ever grew from non-stream calls.
        # After the rescue, as in go(): a rescued turn is remembered by how it ended.
        try:
            message = result.message
            if message is not None:
                _final_text = message.content if hasattr(message, "content") else str(message)
            else:
                _final_text = (result.degrade_note or "").strip()
            _outcome = self._infer_outcome_from_response(_final_text)
            self._record_memory_attempt(_outcome, _final_text)
        except Exception as _e:
            print(f"[memory post-run warn — non-blocking]: {_e}")

        # Session recap: same as go(). Only a real answer from a turn that FINISHED is recorded --
        # a give-up is an agent message too -- so a follow-up never builds on a run that did not.
        if result.message is not None and not result.degrade_note:
            _answer = result.message.content if hasattr(result.message, "content") else str(result.message)
            self._record_conversation_turn(self.user_task, _answer)

    # --- L2: results self-review and the bounded next-step loop ---

    #: An absolute ``.../post_analysis`` directory as it appears in a transcript entry -- the
    #: engine returns that path to whoever called it, so the model's observation carries it
    #: verbatim (often wrapped in quotes or backticks, which the class therefore excludes).
    _POST_ANALYSIS_DIR_RE = re.compile(r"""/[^\s'"`<>|;,()\[\]{}\\]+/post_analysis(?![\w.-])""")

    def _post_analysis_search_roots(self) -> list[str]:
        """Where a manifest written by this turn could plausibly be.

        Three configured places, most-specific first:

        - The root the *caller* named (``self._user_root``). The constructor rebinds ``self.path``
          to ``<root>/spatialomicsgym_data`` before any turn runs, and a user who hands the agent
          a working directory points their prompts at that directory, not at the data subtree.
          The first live campaign wrote ten manifests that way and every one kept ``review: null``,
          because the only agent-side root searched was the data dir *beside* the results.
        - The agent's data root (``self.path``), added whole rather than as ``<path>/outputs``:
          generated code is *told* to write under ``outputs``, but a portal handed an explicit
          ``output_dir`` elsewhere under the data root still wrote a real result, and narrowing
          the search would lose it.
        - The portal default from :func:`spatialomicsgym.paths.tool_output_root`, not from
          ``from tools.base_mcp import default_output_dir``. That import only resolves when the
          repo root is on ``sys.path``; on a wheel install it raised ``ModuleNotFoundError`` into
          a bare ``except`` and this method quietly returned half its roots.

        Plus every ``.../post_analysis`` directory this turn's transcript already names
        (:meth:`_manifest_dirs_from_log`): the user may direct output *anywhere* -- the web front
        door demonstrated a sibling of the agent root -- and the model's own observation of the
        engine's return value is the one record of where the manifest really is.

        The repo root is deliberately NOT searched: it holds ``benchmarks/results/``, and walking
        it on every turn is expensive.
        """
        roots: list[str] = []
        # The folder this turn was TOLD to write to, first. On the portal ``sog_portal.binding`` points
        # the prompt at ``outputs/<account>/<chat>/`` and, when a dataset is bound, points
        # ``SOG_WORK_DIR`` at that dataset's run directory instead -- so the chat folder was in no
        # root at all, and a manifest at ``outputs/<account>/<chat>/<tool>/post_analysis`` sits
        # five levels under ``self.path``, one past the review's depth cap. Autorun analysed the
        # output; the review never found what it wrote; the turn kept ``review: null`` and no
        # ``report.html`` (driven 2026-09-20). Unset for the CLI, a notebook and the benchmark,
        # whose roots are exactly the three below.
        for candidate in (
            getattr(self, "_output_root", None),
            getattr(self, "_user_root", None),
            getattr(self, "path", None),
        ):
            if candidate:
                roots.append(str(candidate))
        roots.append(tool_output_root())
        roots.extend(self._manifest_dirs_from_log())
        seen: set[str] = set()
        return [r for r in roots if r and not (r in seen or seen.add(r))]

    def _manifest_dirs_from_log(self, cap: int = 8) -> list[str]:
        """Every existing ``.../post_analysis`` directory this session's transcript mentions.

        When the model calls ``run_post_analysis`` itself, the observation carries the engine's
        return value: the absolute directory holding ``manifest.json``. Reading it back out of
        ``self.log`` is what lets the review find a result the user directed outside every
        configured root. Existence-checked so a hallucinated or relative spelling never becomes a
        root, and capped so a transcript cannot conscript the review into an unbounded glob.
        """
        found: list[str] = []
        # On a portal turn -- one bound to a chat's own folder -- only directories under that folder.
        # The transcript is text the model and its tools wrote, so any existing absolute
        # ``.../post_analysis`` it printed became a root the review read and wrote beside, in the
        # portal process (u11-stcoscientist-extra-15). The CLI binds no folder and keeps finding a
        # result the user directed anywhere.
        bound = getattr(self, "_output_root", None)
        inside = os.path.realpath(str(bound)) if isinstance(bound, str) and bound else None
        for entry in getattr(self, "log", None) or ():
            text = entry if isinstance(entry, str) else str(entry)
            if "post_analysis" not in text:
                continue
            for match in self._POST_ANALYSIS_DIR_RE.findall(text):
                if match in found or not os.path.isdir(match):
                    continue
                if inside is not None:
                    real = os.path.realpath(match)
                    if real != inside and not real.startswith(inside + os.sep):
                        continue
                found.append(match)
                if len(found) >= cap:
                    return found
        return found

    def _run_followup_stream(self, prompt: str, result: "_StreamResult"):
        """Run one follow-on ReAct round and yield each step as it lands, appending it to ``self.log``.

        Deliberately the *same* graph, thread and degradation policy as ``go()``: the follow-on is a
        normal turn whose prompt happens to have been written by this class rather than by the user,
        so it can call the same MCP tools and its steps land in the same ``self.log``. ``result`` is
        filled with the round's terminal message/state/degrade note, exactly as for the first round.
        """
        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 1000, "configurable": {"thread_id": self._thread_id}}
        for out in _iter_react_stream(self.app, inputs, config, result):
            self.log.append(out)
            yield out

    def _stream_while_running(self, work, *, heartbeat: float | None = None):
        """Run ``work`` on a thread and yield each follow-on step it produces AS it lands.

        The post-analysis review runs its follow-on round through a synchronous callback deep inside
        ``review_and_act``, so ``go_stream`` could only hand the round's steps over once it had
        finished -- measured 2026-09-15 at 15+ minutes of nothing on the wire after an answer.
        ``_run_followup_turn`` now reports each step to ``_followup_step_sink``; this points that sink
        at a queue and yields from it on the caller's thread.

        Three properties are tested. An exception from ``work`` is re-raised here, after the join,
        exactly as a direct call would have raised it. The thread is joined before this returns,
        including when the consumer stops early, so two turns do not mutate this agent at once --
        unless a second Ctrl-C abandons the join (last paragraph). And the round starts each step only
        once the consumer has asked for it, so a consumer that stops asking stops the round.

        The third is what moving the round onto a thread first cost. On the caller's thread the round
        ran only when pulled, like the first round, and Ctrl-C raised inside it. On the thread it ran
        ahead into an unbounded queue: the first Ctrl-C was held in the join until the round finished
        -- measured, 12 s of a two-round 12 s stub -- and a second broke out of the join, leaving the
        round mutating this agent and ``_followup_step_sink`` bound to a queue nobody would read. A
        ``stop`` checked after each step (``a9305ec``) was not enough for the portal, whose pump sees
        its cancel only when a chunk arrives, by which time a round running ahead had started its
        next step: measured with 0.5 s stub steps, Stop at 0.75 s, the next step started at 1.0 s
        and ran to its end. In a real round that step can be a ten-minute tool run. So each step is
        handed over and the round waits for the next request before starting another -- the throttle
        the portal's one-slot queue exists to give, which the unbounded queue here undid. However the
        consumer leaves, ``stop`` is set and the round ends at its next hand-over; no round starts
        after it. What still runs depends on where the consumer was:

        * holding a step (``close()``, ``throw()``): the round is parked in the hand-over, so nothing;
        * waiting for the next one (where Ctrl-C in a script usually lands): the step in flight -- an
          LLM call, a tool run -- finishes, because a thread cannot be interrupted from outside;
        * the portal's Stop: ``sog_portal.server.sse_events._pump``, as written on 2026-09-25, checks
          ``cancel`` when a chunk arrives and then stops asking. Pressed during a step, that step is
          the last. Pressed while the review is between steps -- the L1 analysis before the round,
          the re-check between rounds -- the next step still runs, because the pump is waiting in
          here and cannot see its flag until that step arrives.

        ``heartbeat`` (seconds): with nothing handed over for that long, ``None`` is yielded -- a
        heartbeat, not a step, so the round is not asked for another. The portal counts it as progress,
        which keeps its no-progress watchdog fed through a silent phase (hunt 2026-09-30,
        u11-stcoscientist-2), and never renders it.

        After that the thread still finishes the review, which runs no agent step: ``review_and_act``
        registers the round's artifacts, re-reads the manifest and writes the review, and
        ``_run_post_analysis_review`` renders ``report.html`` and reviews the turn's other runs. The
        join waits for all of it. A second Ctrl-C abandons the join: the sink is restored anyway, but
        the thread finishes the step in flight (appending it to ``self.log``) and that bookkeeping on
        its own, and a turn started before it is done shares this agent with it.
        """
        import queue

        steps: queue.Queue = queue.Queue()
        finished = object()
        failure: list[BaseException] = []
        stop = threading.Event()
        asked = threading.Semaphore(0)  # one permit per step the consumer came back for, one on leaving

        def hand_over(step) -> None:
            # Only the review thread's steps are this generator's. A sink left on the agent by a
            # generator nobody closed (``g = agent.go_stream(...)`` held in a notebook) must neither
            # collect the steps of a round ``go()`` later runs on another thread nor park that round.
            # And once the consumer has gone nothing waits here, so no path can leave the join below
            # waiting on a hand-over nobody will answer.
            if threading.current_thread() is not worker or stop.is_set():
                return
            steps.put(step)
            asked.acquire()

        def run() -> None:
            _REVIEW_THREAD.stop = stop
            try:
                work()
            except BaseException as exc:  # re-raised on the caller's thread below
                failure.append(exc)
            finally:
                steps.put(finished)

        worker = threading.Thread(target=run, name="stcoscientist-post-analysis", daemon=True)
        previous = getattr(self, "_followup_step_sink", None)
        self._followup_step_sink = hand_over
        worker.start()
        try:
            while True:
                try:
                    item = steps.get(timeout=heartbeat) if heartbeat else steps.get()
                except queue.Empty:
                    yield None  # a heartbeat: nothing landed, and the round is not asked for another step
                    continue
                if item is finished:
                    break
                yield item
                asked.release()  # the consumer came back for another: the round may start its next step
        finally:
            try:
                stop.set()  # nothing to stop once the work finished; otherwise the round ends at its next hand-over
                asked.release()  # ... which may already be waiting for a consumer that has now gone
                worker.join()
            finally:
                self._followup_step_sink = previous
        if failure:
            raise failure[0]

    def _run_followup_turn(self, prompt: str) -> str:
        """Run one follow-on ReAct turn to completion and return its answer text.

        This is what makes a next step an action, not a suggestion -- the ``runner`` the post-analysis
        review is handed. A thin wrapper over :meth:`_run_followup_stream` so the review's contract
        (a string in, a string out) is unchanged.

        On the thread :meth:`_stream_while_running` starts, the sink hands each step over and returns
        once the consumer asks for the next; if the consumer has gone instead, the round ends there
        and a round asked for after that does not start. On any other thread nothing is set and the
        round runs to its end, as it always has.
        """
        stop = getattr(_REVIEW_THREAD, "stop", None)
        if stop is not None and stop.is_set():
            return ""
        result = _StreamResult()
        sink = getattr(self, "_followup_step_sink", None)
        stream = self._run_followup_stream(prompt, result)
        try:
            for out in stream:
                if sink is not None:
                    sink(out)  # go_stream's live channel; None everywhere else
                if stop is not None and stop.is_set():
                    break
        finally:
            stream.close()
        message = result.message
        if message is None:
            return (result.degrade_note or "").strip()
        return message.content if hasattr(message, "content") else str(message)

    def _reset_recovery_state(self) -> None:
        """Per-turn state of the self-repair layer: the once-per-tool env ledger, the rescue mark,
        and the general-env helper's call budget. Called wherever a turn starts."""
        self._env_failures = {}
        self._gpu_retry_noticed = set()
        self.last_turn_rescued = None
        try:
            from spatialomicsgym.tool import general_env

            general_env.reset_turn_budget(getattr(self, "timeout_seconds", None))
        except Exception:
            pass

    def _rescue_pending(self, result: "_StreamResult") -> bool:
        """Whether this turn qualifies for the rescue round, before anything else spends it."""
        try:
            from spatialomicsgym.agent import rescue as _rescue

            if not _rescue.rescue_active():
                return False
            content = getattr(result.message, "content", None) if result.message is not None else None
            answer_text = content if isinstance(content, str) else (answer_to_text(content) if content else "")
            return _rescue.rescue_reason(result.degrade_note, answer_text) is not None
        except Exception:
            return False

    def _rescue_plan(self, result: "_StreamResult", followup_ran: bool) -> str | None:
        """The rescue round's prompt when this turn qualifies for its one extra round, else ``None``.

        Qualifies: the layer is active (never under benchmarking), no follow-on round has been run
        this turn (the budget is shared), and the first attempt ended on one of the three give-ups
        or with no ``<solution>`` -- see ``rescue.rescue_reason`` for what is deliberately excluded.
        """
        try:
            from spatialomicsgym.agent import rescue as _rescue
            from spatialomicsgym.agent.execution import _last_execute_error_text
            from spatialomicsgym.tool.support_tools import repl_namespace_summary

            if not _rescue.rescue_active() or followup_ran:
                return None
            content = getattr(result.message, "content", None) if result.message is not None else None
            answer_text = content if isinstance(content, str) else (answer_to_text(content) if content else "")
            reason = _rescue.rescue_reason(result.degrade_note, answer_text)
            if reason is None:
                return None
            messages = (result.final_state or {}).get("messages") or [] if isinstance(result.final_state, dict) else []
            return _rescue.build_rescue_prompt(
                task=str(self.user_task or ""),
                reason=reason,
                last_error=_last_execute_error_text(messages),
                succeeded=_rescue.succeeded_tools(self),
                repl_names=repl_namespace_summary(),
                env_failed=sorted(getattr(self, "_env_failures", {}) or {}),
            )
        except Exception as exc:  # the rescue is best effort; a turn that stopped still stopped
            try:
                print(f"[stcoscientist] rescue skipped ({type(exc).__name__}: {exc})")
            except Exception:
                pass
            return None

    def _run_unsolved_rescue(self, result: "_StreamResult", *, followup_ran: bool):
        """Yield the rescue round's steps; empty when the turn does not qualify.

        When a round runs, ``result`` is updated in place -- its message becomes the round's answer
        when the round produced one, and its degrade note becomes the round's own -- and
        ``last_turn_rescued`` records why the first attempt stopped, ``last_turn_degraded`` how the
        round itself ended (``None`` when it finished cleanly). The round's outcome is never judged
        here: whatever it produced is what the turn reports.
        """
        prompt = self._rescue_plan(result, followup_ran)
        if prompt is None:
            return
        original = (result.degrade_note or "").strip() or "the turn ended without a <solution>"
        # Set BEFORE the round: the SSE bridge announces the rescue on its first chunk and reads
        # this to say why.
        self.last_turn_rescued = original
        try:
            print(f"[stcoscientist] rescue round: {original}")
        except Exception:
            pass
        rescued = _StreamResult()
        yield from self._run_followup_stream(prompt, rescued)
        if rescued.message is not None:
            result.message = rescued.message
        if rescued.final_state is not None:
            # BOTH rounds, in order. The PDF and the markdown transcript are rendered from
            # ``_conversation_state``; keeping only the round's state dropped the user's question,
            # the first attempt and the error it stopped on, so the saved transcript opened on the
            # rescue prompt as if it were the task.
            merged = _merge_round_states(result.final_state, rescued.final_state)
            result.final_state = merged
            self._conversation_state = merged
        note = (rescued.degrade_note or "").strip() or None
        if note:
            # The round's own ending, and the one it was answering. ``last_turn_rescued`` still
            # says why the first attempt stopped; without this the answer and the note described
            # two different events and a reader could not tell the rescue had been tried.
            note = f"{note} [after a rescue round; the first attempt stopped because: {original}]"
            note = note.encode("ascii", "replace").decode("ascii")
        result.degrade_note = note
        self.last_turn_degraded = note

    @staticmethod
    def _ensure_post_analysis_ran(roots, started_at: float, *, exclude=(), analysed_at: float | None = None) -> None:
        """If the turn produced output but no manifest, analyse the output here rather than hope.

        Everything downstream of L1 -- the review, ``report.html``, the results portal, the figures
        the chat panel shows inline -- is keyed on a ``manifest.json``. Nothing wrote one. The engine
        was reached only through ``_enrich_prompt_with_post_analysis``, which *asks* the model to
        call it, and asking is not a call: there was not a single manifest anywhere in the checkout
        outside the test fixtures, so the entire post-analysis stack was inert in production while
        passing its own tests.

        This runs only when the turn produced no manifest of its own, which keeps the prompt path
        primary. A model that made the call gets its own results dir reviewed exactly as before and
        never reaches here; a model that skipped it gets the same artifact anyway. Discovery itself
        (:mod:`spatialomicsgym.postanalysis.autorun`) owns the question of what is safe to analyse.

        Silent by design when there is nothing to do -- most turns answer a question and write no
        files -- and never raises: this sits on the success path of a finished analysis, so a failure
        to *describe* a result must not be reported as a failure to produce one.

        ``analysed_at`` is when the C6 pre-answer check (``execution.pre_answer_check``, off by
        default) last ran this same analysis during the turn. Its manifests are not "the model
        called the engine": they describe the output as it was before the check told the model to
        re-examine it. Measured with C6 on: a one-label clustering CSV was checked (``n_domains=1``,
        ``signal_free``), the model rewrote it with 8 domains, and this returned on the check's
        manifest -- the manifest, the tables, ``report.html`` and the follow-on prompt all said one
        domain under an answer that said eight. So from the check on, both questions are asked of
        what came after it: a newer manifest is still the model's own call, and output written
        since is analysed again. Output the model left alone keeps the check's analysis, unrepeated.
        """
        try:
            from spatialomicsgym.postanalysis.next_step import discover_results_dir, post_analysis_active

            if not post_analysis_active():
                return
            # ``max``: a stamp older than this turn's start is an earlier turn's. go() and go_stream()
            # clear it when a turn starts, and this keeps it from widening the window for any caller
            # that does not -- where it found the earlier turn's manifest and analysed nothing new.
            since = started_at if analysed_at is None else max(started_at, analysed_at)
            if discover_results_dir(roots or (), since, exclude=exclude) is not None:
                return  # the model called the engine itself; L2 will pick that manifest up

            from spatialomicsgym.postanalysis.autorun import analyse_new_outputs

            for directory in analyse_new_outputs(roots or (), since, exclude=exclude):
                print(f"[stcoscientist] post-analysis: wrote {directory}")
        except Exception as exc:
            try:
                print(f"[stcoscientist] post-analysis autorun skipped ({type(exc).__name__}: {exc})")
            except Exception:
                pass

    #: Seconds between heartbeats while the post-answer review round is silent. Well under the
    #: portal's no-progress window (step timeout + 120 s), so a live round is never taken for a stall.
    _REVIEW_HEARTBEAT_SECONDS = 30.0

    def _stream_post_analysis_review(self, started_at: float, log_before: int, *, follow_on: bool = True):
        """Run :meth:`_run_post_analysis_review` beside this generator; yield its steps live, and heartbeats.

        Yields ``{"output": step, "after_answer": True}`` for every step the follow-on round hands over
        (:meth:`_stream_while_running`: one at a time, as the consumer asks, so Stop and Ctrl-C end the
        round), then for anything else the review appended to ``self.log``; and
        ``{"output": None, "after_answer": True}`` after each :attr:`_REVIEW_HEARTBEAT_SECONDS` with
        nothing to hand on. The two lines' versions of this, merged on 2026-10-02: program9's hand-over
        and its de-duplication, and this line's heartbeat and ``follow_on``.
        """
        streamed: set[int] = set()
        for extra in self._stream_while_running(
            lambda: self._run_post_analysis_review(started_at, follow_on=follow_on),
            heartbeat=self._REVIEW_HEARTBEAT_SECONDS,
        ):
            if extra is None:
                yield {"output": None, "after_answer": True}
                continue
            streamed.add(id(extra))
            yield {"output": extra, "after_answer": True}
        for extra in self.log[log_before:]:  # anything the review logged outside the follow-on round
            if id(extra) not in streamed:
                yield {"output": extra, "after_answer": True}

    def _run_post_analysis_review(self, started_at: float, *, follow_on: bool = True):
        """Check this turn's results, write the verdict, render the page, and act on the plan.

        Returns the L2 outcome dict, or ``None`` when the layer was inactive (benchmarking mode,
        ``post_analysis_enabled=False``) or this turn produced no manifest. Never raises: a bad
        review must not turn a completed analysis into a failed run.
        """
        try:
            from spatialomicsgym.postanalysis.next_step import review_and_act

            roots = self._post_analysis_search_roots()
            # Folders this turn must not read or write: on the portal, every OTHER account's
            # folder under the outputs tree, stamped by ``sog_portal.binding.Binding`` for the turn.
            # Empty for the CLI, a notebook and the benchmark, which own the whole tree.
            foreign = tuple(getattr(self, "_foreign_dirs", None) or ())
            self._ensure_post_analysis_ran(
                roots, started_at, exclude=foreign, analysed_at=getattr(self, "_pre_answer_analysed_at", None)
            )
            outcome = review_and_act(
                roots=roots,
                since=started_at,
                # No follow-on round for a turn that GAVE UP: it shares one extra round with the
                # rescue, and a follow-on that built on a result the turn never reported spent it,
                # so the rescue that exists to answer the unsolved question never ran
                # (u11-stcoscientist-10). The review, the verdict and the page still happen.
                runner=self._run_followup_turn if follow_on else None,
                exclude=foreign,
            )
        except Exception as exc:  # best-effort layer; a completed run is still a completed run
            try:
                print(f"[stcoscientist] post-analysis review skipped ({type(exc).__name__}: {exc})")
            except Exception:
                pass
            return None
        if outcome is not None:
            self._write_post_analysis_report(outcome.get("results_dir"))
            self._review_the_rest_of_this_turn(roots, started_at, outcome.get("results_dir"), exclude=foreign)
        return outcome

    @staticmethod
    def _review_the_rest_of_this_turn(roots, started_at: float, primary, *, exclude=()) -> None:
        """Give every *other* run this turn produced its verdict and its page.

        ``autorun`` writes one results directory per output directory it finds, so a turn that ran
        three tools produces three manifests -- but ``review_and_act`` reviews the newest one, and
        the report is rendered from its outcome. The other two kept ``review: null`` and no
        ``report.html``, which in the portal is a run listed beside a reviewed one with no verdict
        against it: indistinguishable from a run that was checked and had nothing to say.

        Only the verdict and the page. The follow-on *action* loop stays on the primary directory,
        because acting on three results in one turn is unbounded work the user did not ask for.

        Never raises, for the same reason its caller does not: this runs after the answer is already
        final, and failing to describe a result must not be reported as failing to produce one.
        """
        try:
            from spatialomicsgym.postanalysis.next_step import propose_next_steps
            from spatialomicsgym.postanalysis.review import (
                discover_results_dirs,
                read_manifest,
                review_manifest,
                write_review,
            )

            primary_key = str(Path(primary).resolve()) if primary else None
            for directory in discover_results_dirs(roots or (), started_at, exclude=exclude):
                try:
                    if primary_key and str(directory.resolve()) == primary_key:
                        continue
                    manifest = read_manifest(directory)
                    if manifest is None:
                        continue
                    review = review_manifest(manifest, directory)
                    write_review(directory, review, propose_next_steps(manifest, review))
                except Exception:
                    continue  # one unreadable run must not cost the others their verdict
                STCoscientist._write_post_analysis_report(directory)
        except Exception as exc:
            try:
                print(f"[stcoscientist] secondary reviews skipped ({type(exc).__name__}: {exc})")
            except Exception:
                pass

    @staticmethod
    def _write_post_analysis_report(results_dir) -> None:
        """Render ``report.html`` beside the manifest -- the artifact a person actually opens.

        L3 is composed here rather than inside the engine on purpose. The contract forbids L1 from
        touching report HTML, and ``write_report``'s own docstring makes a failed report the
        caller's problem "because L1 must be free to treat a failed report as a warning rather than
        a failed run". Running it *after* L2 is what puts the verdict on the page; rendering from
        the engine would produce a page whose review block is always null.

        Inactive layer, missing manifest and unwritable directory all end here as a printed note,
        never an exception: the analysis is already finished and complete by the time we render.
        """
        if not results_dir:
            return
        try:
            from spatialomicsgym.postanalysis.as_owner import foreign_writer, run_module_as

            writer = foreign_writer(results_dir)
            if writer is not None:
                # Root over a directory the agent can write: render as the agent (see as_owner).
                done = run_module_as(*writer, "spatialomicsgym.report", [str(results_dir)])
                if done.returncode != 0:
                    print(f"[stcoscientist] post-analysis report skipped: {(done.stderr or '').strip()[-300:]}")
                return
            import spatialomicsgym.report as report

            report.write_report(results_dir)
        except Exception as exc:
            try:
                print(f"[stcoscientist] post-analysis report skipped ({type(exc).__name__}: {exc})")
            except Exception:
                pass

    def result_formatting(self, output_class, task_intention):
        self.format_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are evaluateGPT, tasked with extract and parse the task output based on the history of an agent. "
                        "Review the entire history of messages provided. "
                        "Here is the task output requirement: \n"
                        f"'{task_intention.replace('{', '{{').replace('}', '}}')}'.\n"
                    ),
                ),
                ("placeholder", "{messages}"),
            ]
        )

        checker_llm = self.format_check_prompt | self.llm.with_structured_output(output_class)
        _r = checker_llm.invoke({"messages": [("user", str(self.log))]})
        # with_structured_output may return a plain dict (TypedDict/dict schema), which has neither
        # model_dump nor .dict — pass it through instead of AttributeError'ing.
        result = _r if isinstance(_r, dict) else (_r.model_dump() if hasattr(_r, "model_dump") else _r.dict())
        return result

    # --- Conversation history / PDF generation ---

    def save_conversation_history(self, filepath: str, include_images: bool = True, save_pdf: bool = True) -> None:
        """Save the complete conversation history as PDF only.

        Args:
            filepath: Path where to save the PDF file (without extension).
            include_images: Whether to include captured plots and images in the output.
            save_pdf: Whether to save as PDF. Defaults to True.
        """
        import tempfile

        if not save_pdf:
            print("PDF saving is disabled. No file will be saved.")
            return

        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:  # Only create directory if it's not empty
            os.makedirs(directory, exist_ok=True)

        # Create PDF file path - use the user's filename and add .pdf extension
        if filepath.endswith(".pdf"):
            pdf_path = filepath
        else:
            # Remove any existing .md extension if present, then add .pdf
            base_name = filepath
            if base_name.endswith(".md"):
                base_name = base_name[:-3]  # Remove .md extension
            pdf_path = f"{base_name}.pdf"

        # Create markdown content + temp file. Best-effort: this convenience export must only WARN,
        # never crash the caller (chat_cli / notebook) -- a malformed log entry (_generate_markdown_
        # content) or an unwritable/full $TMPDIR (NamedTemporaryFile) would otherwise raise here,
        # outside the PDF-conversion try below, and lose the run's results to a traceback.
        temp_markdown_path = None
        try:
            markdown_content = self._generate_markdown_content(include_images)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as temp_file:
                temp_file.write(markdown_content)
                temp_markdown_path = temp_file.name
        except Exception as e:
            print(f"Warning: Could not build the conversation transcript: {e}")
            return

        try:
            # Add timeout for PDF generation to prevent hanging.
            # SIGALRM only works in the main thread and only on POSIX; off-thread or on
            # Windows we skip the alarm rather than crash, and we always restore the
            # previously-installed handler so we don't leak our timeout handler.
            import signal
            import threading

            def timeout_handler(signum, frame):
                raise TimeoutError("PDF generation timed out")

            use_alarm = threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM")
            old_handler = None
            if use_alarm:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)  # Set timeout to 60 seconds

            try:
                self._convert_markdown_to_pdf(temp_markdown_path, pdf_path)
                print(f"Conversation history saved as PDF: {pdf_path}")
                print(f"Total steps recorded: {len(self.log)}")
            finally:
                if use_alarm:
                    signal.alarm(0)  # Cancel the alarm
                    signal.signal(signal.SIGALRM, old_handler)  # Restore prior handler

        except TimeoutError:
            print("Warning: PDF generation timed out after 60 seconds")
        except Exception as e:
            print(f"Warning: Could not convert to PDF: {e}")
        finally:
            # Clean up the temporary markdown file
            try:
                os.unlink(temp_markdown_path)
            except OSError:
                pass  # File might already be deleted

    def _generate_markdown_content(self, include_images: bool = True) -> str:
        """Generate markdown content from conversation history."""
        # Initialize content and tracking variables
        content = """# SpatialOmicsLab Agent Conversation History

"""
        added_plots = set()
        step_number = 0
        first_human_shown = False

        # Get data source (conversation state or log)
        messages = self._get_messages_for_processing()

        # Process all messages using unified logic
        for message_data in messages:
            content, step_number, first_human_shown = self._process_message(
                message_data, content, step_number, first_human_shown, added_plots, include_images
            )

        return content

    def _get_messages_for_processing(self):
        """Get messages from conversation state or fallback to log."""
        conversation_state = getattr(self, "_conversation_state", None)

        if conversation_state and hasattr(conversation_state, "get") and "messages" in conversation_state:
            return self._normalize_conversation_state_messages(conversation_state["messages"])
        else:
            return self._normalize_log_messages(self.log)

    def _normalize_conversation_state_messages(self, messages):
        """Convert conversation state messages to unified format."""
        normalized = []
        for message in messages:
            if hasattr(message, "content"):
                content = str(message.content)
            else:
                content = str(message)

            # Determine message type
            if isinstance(message, HumanMessage):
                msg_type = "human"
            elif isinstance(message, AIMessage):
                msg_type = "ai"
            else:
                msg_type = "other"

            normalized.append({"content": content, "type": msg_type, "original": message})

        return normalized

    def _normalize_log_messages(self, log_entries):
        """Convert log entries to unified format."""
        normalized = []
        for log_entry in log_entries:
            content = str(log_entry)

            # Determine message type from log format
            if "Human Message" in content:
                msg_type = "human"
            elif "Ai Message" in content:
                msg_type = "ai"
            else:
                msg_type = "other"

            normalized.append({"content": content, "type": msg_type, "original": log_entry})

        return normalized

    def _process_message(self, message_data, content, step_number, first_human_shown, added_plots, include_images):
        """Process a single message and return updated state."""
        clean_output = clean_message_content(message_data["content"])
        msg_type = message_data["type"]

        if msg_type == "human":
            return self._process_human_message(clean_output, content, step_number, first_human_shown)
        elif msg_type == "ai":
            return self._process_ai_message(clean_output, content, step_number, added_plots, include_images)
        else:
            return self._process_other_message(
                clean_output, content, step_number, first_human_shown, added_plots, include_images
            )

    def _process_human_message(self, clean_output, content, step_number, first_human_shown):
        """Process human messages."""
        if "each response must include thinking process" in clean_output.lower():
            parsing_error_content = create_parsing_error_html()
            content += f"{parsing_error_content}\n\n"
        elif not first_human_shown:
            content += "#### Human Prompt\n\n"
            content += f"*{clean_output}*\n\n"
            first_human_shown = True

        return content, step_number, first_human_shown  # step_number unchanged

    def _process_ai_message(self, clean_output, content, step_number, added_plots, include_images):
        """Process AI messages."""
        # Check if this message contains observation tags and process accordingly
        observation_pattern = r"<observation>(.*?)</observation>"
        observation_matches = re.findall(observation_pattern, clean_output, re.DOTALL | re.IGNORECASE)

        if observation_matches:
            # Extract content before, between, and after observation tags
            parts = re.split(observation_pattern, clean_output, flags=re.DOTALL | re.IGNORECASE)

            # Process each part
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Even indices are non-observation content
                    if part.strip():
                        # This is regular content - process it normally
                        if not should_skip_message(part):
                            if part.strip():
                                step_number += 1
                                content += f"#### Step {step_number}\n\n"

                                # Handle execution results if present
                                execution_results = getattr(self, "_execution_results", None)
                                if has_execution_results(part, execution_results):
                                    content, added_plots = self._process_execution_with_results(
                                        part, content, added_plots, include_images, execution_results
                                    )
                                else:
                                    content = self._process_regular_ai_message(part, content)
                else:  # Odd indices are observation content
                    if part.strip():
                        # This is observation content - format as terminal
                        formatted_observation = format_observation_as_terminal(f"<observation>{part}</observation>")
                        if formatted_observation is not None:
                            content += f"{formatted_observation}\n\n"

            return content, step_number, True

        # Skip empty or error messages
        if should_skip_message(clean_output):
            return content, step_number, True

        if clean_output.strip():
            step_number += 1
            content += f"#### Step {step_number}\n\n"

            # Handle execution results if present
            execution_results = getattr(self, "_execution_results", None)
            if has_execution_results(clean_output, execution_results):
                content, added_plots = self._process_execution_with_results(
                    clean_output, content, added_plots, include_images, execution_results
                )
            else:
                content = self._process_regular_ai_message(clean_output, content)

        return content, step_number, True

    def _process_other_message(
        self, clean_output, content, step_number, first_human_shown, added_plots, include_images
    ):
        """Process other message types."""
        # Check if this is actually an observation (has <observation> tags)
        if not re.search(r"<observation>", clean_output, re.IGNORECASE):
            content += f"{clean_output}\n\n"
        return content, step_number, first_human_shown

    def _process_execution_with_results(self, clean_output, content, added_plots, include_images, execution_results):
        """Process AI message with execution results."""
        matching_execution = find_matching_execution(clean_output, execution_results)

        if matching_execution:
            content = self._format_and_add_content(clean_output, content)
            content, added_plots = self._add_execution_plots(matching_execution, content, added_plots, include_images)
        else:
            content = self._format_and_add_content(clean_output, content)

        return content, added_plots

    def _format_and_add_content(self, clean_output, content):
        """Format and add content to markdown."""
        # Process lists first, then execute tags
        formatted_content = format_lists_in_text(clean_output)

        # Create a wrapper function for the tool parsing
        def parse_tool_calls_wrapper(code):
            return self._parse_tool_calls_with_modules(code)

        formatted_content = format_execute_tags_in_content(formatted_content, parse_tool_calls_wrapper)
        return content + f"{formatted_content}\n\n"

    def _add_execution_plots(self, matching_execution, content, added_plots, include_images):
        """Add plots from execution results."""
        if include_images and matching_execution.get("images"):
            for plot_data in matching_execution["images"]:
                if plot_data not in added_plots:
                    content += f"![Plot]({plot_data})\n\n"
                    added_plots.add(plot_data)
        return content, added_plots

    def _process_regular_ai_message(self, clean_output, content):
        """Process regular AI message without execution results."""
        return self._format_and_add_content(clean_output, content)

    def _convert_markdown_to_pdf(self, markdown_path: str, pdf_path: str) -> None:
        """Convert markdown file to PDF."""
        convert_markdown_to_pdf(markdown_path, pdf_path)
