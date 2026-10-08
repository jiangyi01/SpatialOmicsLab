"""
Out-of-process, cancellable Tier-2 agent probe.

Run as a module IN THE BASE ENV (where the agent + torch live)::

    conda run --no-capture-output -n <basic> \
        python -m sog_install._agent_probe '<json-request>'

It reads a JSON request (argv[1], else stdin), triggers ONE real
``STCoscientist(...).go_stream(prompt)`` on the staged real dataset, and prints a JSON
*contract* on stdout, fenced by sentinels so the agent's own banner/print noise on
stdout can never corrupt it::

    SOG_PROBE_STEP {"mode": "acting", "thought": "...", "action": "...", "status": "..."}
    SOG_PROBE_STEP {...}                      # one live "thinking screenshot" per ReAct step
    ...
    SOG_PROBE_JSON_BEGIN
    {"answer": str, "log": [str, ...], "error": str, "degraded": str, "rescued": str}
    SOG_PROBE_JSON_END

The ``SOG_PROBE_STEP`` lines stream *during* the run so the parent (``testing.py``) can drive a live
thinking box; the fenced ``{answer, log, error}`` contract is printed last and is unchanged. A parent
that doesn't consume stdout live just sees the step lines as harmless noise before the fence
(``_extract_probe_json`` ``rfind``s the fence, so interleaved step markers never corrupt it). We prefer
the streaming ``go_stream`` API and fall back to the blocking ``go()`` for older agents / stubs.

Request keys::

    {"root": <STCoscientist path=>, "config_path": <generated MCP config>, "prompt": <task>,
     "add_data": {<abs h5ad path>: <plain description>}}   # optional

``add_data`` is optional. Tier-2 leaves it out and stages its dataset into the data lake under
``root`` (the agent discovers it there); the Part-D demo / real-run pass an explicit
``{path: description}`` so the agent registers the user's own file exactly as the tutorial driver
does (``agent.add_data({...})``). Absent/empty ⇒ the pre-existing behavior, byte-for-byte.

Why a subprocess and not a thread: the old Tier-2 ran ``agent.go()`` on a *daemon
thread* and ``join(timeout)``d it — on timeout the orphaned thread kept executing the
full run inside the wizard's own interpreter, an un-killable hang. As its own process
(in its own session, see ``testing._run_subprocess_cancellable``) the whole tree — agent
plus any MCP-server grandchildren — is killable with a real wall-clock timeout, and the
heavy import lives in the base env, not the launcher env.

This module only TRIGGERS the agent; it never repairs anything. Detection + repair happen
in the parent (``testing.py``) by scanning the returned ``log`` with ``envdoctor``. The
env-only boundary is preserved — nothing here mutates a conda env or agent/tool source.

Stdlib + the agent package only.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

# The shared solution-wrapper pattern (see spatialomicsgym/answer.py). Free at module scope: this
# module IS in the package, so ``spatialomicsgym/__init__`` — which re-exports from ``answer`` —
# has already run by the time this line does. ``answer`` is stdlib-only, so it cannot raise here.
from spatialomicsgym.answer import SOLUTION_TAG_RE as _RE_SOLUTION

# Fences delimiting the one line of contract JSON on stdout. Imported by the parent
# (``testing._extract_probe_json``) so the two sides can never drift.
PROBE_BEGIN = "SOG_PROBE_JSON_BEGIN"
PROBE_END = "SOG_PROBE_JSON_END"

# Prefix of a live "thinking screenshot" line, one per streamed ReAct step, emitted to the REAL
# stdout *before* the JSON fence. The parent (``testing._feed_box``) forwards these to the live
# thinking box; ``_extract_probe_json`` ignores them (it ``rfind``s the fence), so they can never
# corrupt the contract. Payload is a compact JSON ``{mode, thought, action, status}`` from
# ``_summarize_step``. When nothing consumes stdout live these lines are simply harmless noise.
PROBE_STEP = "SOG_PROBE_STEP"

_MAX_THOUGHT = 400  # clip the rolling thought body put on the wire (the box re-wraps + tail-clamps)
_MAX_ACTION = 200  # clip the one-line action
_MAX_STATUS = 48  # clip the short right-aligned status label (e.g. a tool name)

# ReAct step vocabulary the STCoscientist agent emits (see agent/execution.py): <think>/<execute>/
# <solution>, plus <observation> for a tool result and a ``Tool: <name>`` line in Anthropic tool_use
# renders. Parsing is best-effort + display-only — an unrecognized shape degrades to a plain thought.
_RE_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_RE_EXECUTE = re.compile(r"<execute>(.*?)</execute>", re.DOTALL | re.IGNORECASE)
# _RE_SOLUTION is the shared pattern, imported above.
# Just the opening tag (for a truncated turn with no close) and either tag as a bare token (to scrub a
# stray/nested marker out of extracted text). Matching on the ORIGINAL string — never a ``.lower()`` copy,
# whose length can shift under Unicode case-folding (``İ`` → ``i̇``) and slice the answer at the wrong offset.
_RE_SOLUTION_OPEN = re.compile(r"<solution>", re.IGNORECASE)
_RE_SOLUTION_TOKEN = re.compile(r"</?solution>", re.IGNORECASE)
_RE_OBSERVE = re.compile(r"<observation>(.*?)</observation>", re.DOTALL | re.IGNORECASE)
_RE_TOOL = re.compile(r"^Tool:\s*(.+)$", re.MULTILINE)
_RE_MSG_TYPE = re.compile(r"=+\s*(\w+)\s+Message\s*=+", re.IGNORECASE)

# Scratchpad section headers the ReAct agent emits before <solution>. When a turn has NO solution
# tag we drop these lines so the "answer" is the deliverable, not the private reasoning. Display-only.
_SCRATCHPAD_HEADERS = re.compile(
    r"^\s*(most recent observation|known vs assumed|assumed vs unknown|alternatives?|"
    r"pre-?mortem|plan|reasoning|scratchpad|thought)\b.*$",
    re.IGNORECASE,
)


def _clean_deliverable(text: str) -> str:
    """Drop leading routing/thinking scaffolding from already-extracted solution text, delegating to
    the CLI's canonical cleaner so the two front doors cannot drift into different ideas of "the
    deliverable".

    ``_SCRATCHPAD_HEADERS`` above only runs on the *no-tag* branch, so scaffolding the model nested
    INSIDE ``<solution>`` used to come through verbatim (observed live: 3809 of 3832 characters were
    a ``Classification:`` label plus a full Deliberative Thinking Protocol). Safe to apply here
    because every ``<solution>``/``</solution>`` token has already been scrubbed from ``text``, so
    the cleaner's own solution-extraction is a no-op and only its scaffolding strips apply.

    Imported lazily and defensively: this module's contract is "never raises", and it also runs
    under partial installs where ``chat_cli`` may be absent — there we return the text unchanged,
    which is the pre-existing behaviour."""
    try:
        from spatialomicsgym.chat_cli import _clean_final_answer
    except Exception:
        return text
    try:
        return _clean_final_answer(text) or text
    except Exception:
        return text


def extract_solution(text: str) -> str:
    """The user-facing deliverable from an agent's raw final message. Never raises; empty-safe.

    1. A well-formed ``<solution>…</solution>`` → its inner text.
    2. An *opening* ``<solution>`` with no close (a truncated / malformed turn — the observed real
       case) → everything from the tag to the end.
    3. No tag at all → the message with its *leading* scratchpad section headers
       (``Known vs assumed…`` / ``Pre-mortem`` / ``Plan:`` …) removed.

    In cases 1 & 2 any residual ``<solution>`` / ``</solution>`` marker (a nested/duplicated tag) is
    scrubbed so a control token never surfaces in the deliverable, and the result is then passed
    through ``_clean_deliverable`` — models routinely nest the routing label and the thinking
    protocol *inside* the tag, where the case-3 header strip below can never see them. In case 3
    only the *contiguous* leading header block is dropped — a ``Plan:`` line in the MIDDLE of a real
    answer survives — and a message that is nothing but headers yields ``""`` (an honest empty),
    never the raw scratchpad.
    """
    text = (text or "").strip()
    if not text:
        return ""
    m = _RE_SOLUTION.search(text)
    if m:
        return _clean_deliverable(_RE_SOLUTION_TOKEN.sub("", m.group(1)).strip())
    opener = _RE_SOLUTION_OPEN.search(text)  # search (not rfind-on-lowered): first tag, offset-exact
    if opener:  # opening tag, no close — take tag-to-end, minus any stray marker
        return _clean_deliverable(_RE_SOLUTION_TOKEN.sub("", text[opener.end() :]).strip())
    # no tag: skip the contiguous run of leading scratchpad headers (blank lines don't end the run),
    # then keep everything from the first real content line onward — verbatim.
    lines = text.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if _SCRATCHPAD_HEADERS.match(ln):
            start = i + 1
        elif ln.strip():  # first non-blank, non-header line ends the leading scratchpad block
            break
    return "\n".join(lines[start:]).strip()


def _clip(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` chars with a trailing ellipsis; empty-safe, never raises."""
    text = (text or "").strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


# Display-only noise filters for step previews: the injected post-task-analysis prompt scaffolding
# and the ReAct format-correction chatter. Dropping these never changes agent behavior or the full
# on-disk log — only what the live box / scrollback shows.
# Anchor the banner alternative to a WHOLE line of ``===…===`` (a pretty-print rule), so a real
# reasoning line that merely mentions ``=== A ===`` mid-sentence is not dropped. ``^\s*mandatory\b`` is
# gone: the genuine scaffolding line ("MANDATORY POST-TASK ANALYSIS") is already caught by
# ``post-?task analysis``, and the bare word false-positived on real steps like "Mandatory QC: …".
_NOISE_LINE = re.compile(
    r"(^\s*=+.*=+\s*$|post-?task analysis|"
    r"each response must include|you must respond with|"
    r"wrap .*<(?:execute|solution)>|<(?:execute|solution)> tag)",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    """Strip the ``pretty_print`` banner rules (``==== Ai Message ====``) + ``Name:`` lines, drop the
    injected-prompt / format-error scaffolding, and collapse the remainder to one clean line."""
    out: list[str] = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s or set(s) == {"="}:  # blank or a pure '=====' rule
            continue
        if s.startswith("Name:") or _RE_MSG_TYPE.match(ln):  # '==== Ai Message ====' banner (word mid-line)
            continue
        if _NOISE_LINE.search(s):  # injected-prompt scaffolding / format-error chatter — display noise
            continue
        out.append(s)
    return " ".join(out).strip()


def _step_label(mode: str, status: str, action: str) -> str:
    """A friendly, present-tense step name for the live timeline. ``acting`` with a ``run_*`` / tool
    ``status`` reads as "Calling <tool>"; a bare code turn reads as "Running the analysis"."""
    mode = (mode or "thinking").lower()
    status = (status or "").strip()
    if mode == "acting":
        if status and status not in ("running code", ""):
            return f"Calling {status}"
        return "Running the analysis"
    return {
        "thinking": "Understanding the task",
        "observing": "Reading the results",
        "answering": "Writing the answer",
    }.get(mode, mode.capitalize())


def _summarize_step(text: str) -> dict:
    """One ``pretty_print(message)`` step string → the thinking box's ``{mode, status, thought, action,
    label}``. Thin wrapper over :func:`_summarize_step_raw` that adds the human ``label`` (via
    :func:`_step_label`); the raw shape is unchanged for any caller reading the four base fields."""
    step = _summarize_step_raw(text)
    step["label"] = _step_label(step.get("mode", ""), step.get("status", ""), step.get("action", ""))
    return step


def _summarize_step_raw(text: str) -> dict:
    """One ``pretty_print(message)`` step string → the thinking box's ``{mode, thought, action,
    status}``. Pure + best-effort: recognizes the ReAct tag vocabulary, else falls back to a plain
    'thinking' body. NEVER raises — unexpected agent output must not break the live box or the probe.

    ``<execute>`` is checked before ``<solution>`` so a turn that (wrongly) emits both — the GPT
    hallucination ``execution.py`` guards against by preferring execute — displays as 'acting', matching
    what the agent will actually do."""
    text = text or ""
    try:
        exe = _RE_EXECUTE.search(text)
        if exe:
            think = _RE_THINK.search(text)
            thought = _clean(think.group(1)) if think else _clean(text[: exe.start()])
            code = " ".join(exe.group(1).split())
            return {
                "mode": "acting",
                "status": "running code",
                "thought": _clip(thought, _MAX_THOUGHT),
                "action": _clip(code, _MAX_ACTION),
            }
        sol = _RE_SOLUTION.search(text)
        if sol:
            return {
                "mode": "answering",
                "status": "final answer",
                "thought": _clip(_clean(sol.group(1)), _MAX_THOUGHT),
                "action": "",
            }
        obs = _RE_OBSERVE.search(text)
        if obs:
            return {
                "mode": "observing",
                "status": "reading result",
                "thought": _clip(_clean(obs.group(1)), _MAX_THOUGHT),
                "action": "",
            }
        tool = _RE_TOOL.search(text)
        if tool:
            name = tool.group(1).strip()
            return {
                "mode": "acting",
                "status": _clip(name, _MAX_STATUS),
                "thought": _clip(_clean(text[: tool.start()]), _MAX_THOUGHT),
                "action": _clip(name, _MAX_ACTION),
            }
        think = _RE_THINK.search(text)
        if think:
            return {
                "mode": "thinking",
                "status": "",
                "thought": _clip(_clean(think.group(1)), _MAX_THOUGHT),
                "action": "",
            }
        mtype = _RE_MSG_TYPE.search(text)
        if mtype and mtype.group(1).lower() == "tool":  # a tool result rendered as a Tool Message
            return {
                "mode": "observing",
                "status": "reading result",
                "thought": _clip(_clean(text), _MAX_THOUGHT),
                "action": "",
            }
        return {"mode": "thinking", "status": "", "thought": _clip(_clean(text), _MAX_THOUGHT), "action": ""}
    except Exception:
        return {"mode": "thinking", "status": "", "thought": "", "action": ""}


def _is_human_message(message) -> bool:
    """Whether ``message`` is the user's side of the conversation (the prompt or a loop nudge).

    The same test as ``stcoscientist._is_agent_answer``, read off the message's ``type`` so the probe
    keeps working against the hermetic stubs that never import langchain: a ``HumanMessage`` has
    ``type == "human"``, and a plain-dict message says so under ``type`` or ``role``."""
    kind = message.get("type") or message.get("role") if isinstance(message, dict) else getattr(message, "type", None)
    return str(kind or "").lower() in ("human", "user")


def _final_answer(agent, log: list[str]) -> str:
    """Mirror ``go()``'s return value from the streamed state: the content of the last message the
    AGENT wrote (``go_stream`` sets ``agent._conversation_state`` at the end). Falls back to the last
    rendered non-prompt log line when the state isn't exposed. Never raises.

    Only an agent message may be the answer. ``stream_mode="values"`` yields the INITIAL state first,
    so when the very first model call failed (a rejected key, a 400, a rate limit) the state's last
    message was the user's own prompt, and this handed it back as the answer: Tier-2 scored the run
    PASS and the demo printed the user's question in its Result box (hunt 2026-09-30,
    u37-setup-checks-1). A state with no agent message at all answers ``""`` -- an honest empty --
    and the reason travels in the contract's ``degraded`` field.

    ``log`` must be the trace up to the answer, not the whole run: ``go_stream`` also streams what
    the L2 follow-on round appended afterwards, and the last line of *that* is not this run's
    answer. :func:`_drive_agent` does the filtering, on the producer's ``after_answer`` marker."""
    state = getattr(agent, "_conversation_state", None)
    try:
        msgs = state.get("messages") if isinstance(state, dict) else None
        if msgs:
            for last in reversed(msgs):
                if _is_human_message(last):
                    continue
                content = getattr(last, "content", None)
                if content is None and isinstance(last, dict):
                    content = last.get("content")
                if content is not None:
                    return content if isinstance(content, str) else str(content)
            return ""  # the state exists and holds no agent message: nothing was answered
    except Exception:
        pass
    # The fallback log line is a rendered ``pretty_print`` output that still carries the
    # ``==== Ai Message ====`` banner + ``Name:`` scaffolding; run it through ``_clean`` so the
    # fallback answer reads like the real one, not raw banner text. A ``Human Message`` line is the
    # prompt, never an answer (same reason as above).
    for line in reversed(log or []):
        banner = _RE_MSG_TYPE.search(line or "")
        if banner and banner.group(1).lower() == "human":
            continue
        return _clean(line)
    return ""


def _drive_agent(agent, prompt: str, emit) -> tuple[list[str], object]:
    """Run one agent task and return ``(log, answer)``, emitting a live step per turn.

    Prefers the streaming API (``go_stream``) so every ReAct step surfaces live in the thinking box;
    falls back to the blocking ``go()`` (older agents / hermetic stubs) and summarizes each log line
    post-hoc. The accumulated ``log`` and the final ``answer`` match ``go()``'s contract either way."""
    go_stream = getattr(agent, "go_stream", None)
    if callable(go_stream):
        log: list[str] = []
        # The full log is what we return (the follow-on round is real work and belongs in it); the
        # answer may only be read off the part written before the answer was fixed. See _final_answer.
        answerable: list[str] = []
        for step in go_stream(prompt):
            raw = step.get("output", "") if isinstance(step, dict) else step
            if raw is None:
                continue  # a heartbeat from the post-answer review round: progress, not a step
            out = str(raw)
            log.append(out)
            if not (isinstance(step, dict) and step.get("after_answer")):
                answerable.append(out)
            try:
                emit(_summarize_step(out))
            except Exception:  # a feeder must never break the run
                pass
        return log, _final_answer(agent, answerable)
    # Fallback: the blocking API. Emit a summarized step per already-rendered log line (post-hoc).
    log, answer = agent.go(prompt)
    log = [str(x) for x in (log or [])]
    for out in log:
        try:
            emit(_summarize_step(out))
        except Exception:
            pass
    return log, answer


def _run(req: dict, emit=None) -> dict:
    """Build the agent, run one task, and return the raw contract dict.

    ``emit`` (optional) is a ``callable(step_dict)`` invoked once per streamed ReAct step so the parent
    can render a live thinking box; when omitted the run is silent (byte-for-byte the old behavior).

    The heavy ``from spatialomicsgym.agent import STCoscientist`` is deliberately *inside*
    this function so a hermetic test can inject a stub ``spatialomicsgym.agent`` module
    into ``sys.modules`` and exercise :func:`main` without importing torch.
    """
    from spatialomicsgym.agent import STCoscientist

    emit = emit or (lambda _step: None)
    root = req["root"]
    config_path = req.get("config_path") or ""
    prompt = req["prompt"]
    add_data = req.get("add_data") or None  # {abs_path: description}, optional

    agent = STCoscientist(path=root, expected_data_lake_files=[])
    if config_path:
        agent.add_mcp(config_path=config_path)
    if add_data and isinstance(add_data, dict):
        # Register the caller's own dataset(s) exactly as the tutorial driver does — the agent
        # reads the file at its absolute path; the description carries NO tool hint (the
        # recommender must still route). Only used by the Part-D demo / real-run.
        agent.add_data({str(k): str(v) for k, v in add_data.items()})
    log, answer = _drive_agent(agent, prompt, emit)
    return {
        "answer": "" if answer is None else str(answer),
        "log": [str(x) for x in (log or [])],
        "error": "",
        # Why the turn stopped early -- a give-up guard, the step budget, or a provider error the
        # stream swallowed into a note -- or "" when it finished on its own terms. ``go_stream``
        # sets ``last_turn_degraded`` so a front door never reads a give-up as an answer; this door
        # dropped it, and Tier-2 scored a run that never reached the model PASS (hunt 2026-09-30,
        # u37-setup-checks-1, uL4-honesty-6). ``rescued`` is why a first attempt needed the rescue
        # round, carried for the record only: a rescue that then finished cleanly is an answer.
        "degraded": str(getattr(agent, "last_turn_degraded", None) or ""),
        "rescued": str(getattr(agent, "last_turn_rescued", None) or ""),
        # The token ledger, or `None`. Without this line it is built, filled and then discarded
        # with the subprocess on every trial, so the benchmark harness -- the one consumer that
        # exists for it -- never sees a number. This dict is what gets JSON-fenced back to the
        # parent, and it is the only way out of the child.
        #
        # The absent-versus-empty distinction this preserves is real -- "the agent made no
        # measured call" is a different fact from "there is no ledger" -- but see
        # `_usage_payload` for what actually enforces it, which is NOT what it first looks like.
        "usage": _usage_payload(agent),
    }


def _usage_payload(agent: Any) -> dict | None:
    """What the agent's ledger holds, as plain JSON, or ``None`` if it has none.

    WHAT THIS NUMBER ACTUALLY IS, because the class name says something narrower. `TurnUsage`
    accumulates for the life of the agent OBJECT, not for one turn: `execution._usage_of` attaches
    a ledger on first use and returns that same instance forever after, and nothing resets it. The
    probe builds one `STCoscientist` per trial, so what leaves here is a WHOLE-TRIAL total across
    every `generate` and `self_critic` call -- which is what a benchmark wants, and is why this is
    emitted at the end of `_run` rather than per turn.

    The hazard is that the name invites a future change to make it genuinely per-turn, and that
    change would silently turn this from a trial total into a last-turn total with nothing
    failing. `test/test_a_turn_records_what_it_cost.py` pins the accumulating behaviour so the
    change goes red instead.
    """
    # No ledger, no import: the hermetic probe tests stub ``spatialomicsgym.agent`` as a plain module,
    # and importing ``.usage`` under that stub raised, so every contract came back as an error
    # whenever no earlier test had already imported the real module (order-dependent reds).
    if getattr(agent, "_turn_usage", None) is None:
        return None
    from spatialomicsgym.agent.usage import TurnUsage

    # `isinstance` here is an EXPLICIT type check with no behavioural consequence, and it is
    # worth saying so rather than inventing a reason for it. Two plausible ones were written down
    # and both turned out false when they were actually tested:
    #
    #   "a truthiness test would collapse absent into empty" -- no: an empty `TurnUsage` is
    #   TRUTHY, a dataclass with neither `__bool__` nor `__len__`, so `if not ledger` admits it
    #   exactly as this does.
    #
    #   "it refuses a wrong-typed value that would otherwise raise" -- no: the `except` below
    #   already turns that into the same `None`. A dict, a string and a bare object all come back
    #   `None` either way.
    #
    # So it is kept for legibility -- the reader sees what shape is expected without tracing the
    # except -- and a mutation swapping it for a truthiness test is GREEN because the two really
    # are equivalent, not because the test is blind. That distinction is the point of chasing a
    # green mutation to the ground instead of loosening something until it goes red.
    ledger = getattr(agent, "_turn_usage", None)
    if not isinstance(ledger, TurnUsage):
        return None
    try:
        return ledger.as_dict()
    except Exception:
        return None


def _emit_step(stream, step: dict) -> None:
    """Write one ``SOG_PROBE_STEP <json>`` live-step line to the real stdout. Best-effort — a broken
    pipe / closed stream must never perturb the run, so every failure is swallowed."""
    try:
        stream.write(f"{PROBE_STEP} {json.dumps(step, ensure_ascii=True)}\n")
        stream.flush()
    except Exception:
        pass


def main(argv: list[str]) -> int:
    # Route the agent's banner/print noise to stderr so stdout carries only the fenced contract (plus
    # the SOG_PROBE_STEP live-step lines we write explicitly to real_stdout). Keep a handle to the real
    # stdout to emit those steps during the run and the contract at the end.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    out: dict = {"answer": "", "log": [], "error": ""}
    try:
        raw = argv[1] if len(argv) > 1 else sys.stdin.read()
        req = json.loads(raw)
        out = _run(req, emit=lambda step: _emit_step(real_stdout, step))
    except BaseException as exc:  # the probe must ALWAYS emit a contract, whatever failed
        import traceback

        out = {
            "answer": "",
            "log": [],
            "error": f"{type(exc).__name__}: {exc}"[:600],
            "traceback": traceback.format_exc()[:4000],
        }
    finally:
        sys.stdout = real_stdout
    print(PROBE_BEGIN)
    print(json.dumps(out))
    print(PROBE_END)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
