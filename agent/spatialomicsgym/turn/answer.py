"""Canonical cleanup of the agent's *final answer* text.

``STCoscientist.go`` returns the raw final AI message. The ReAct machinery frames that message for
its own parser -- ``<solution>``/``<think>`` wrapper tags, a routing ``Classification:`` label, and
the ``Deliberative Thinking Protocol`` loop scaffold from ``prompt_builder.py`` -- and models
frequently emit all of it *inside* the answer. Every front door needs the same cleanup, so it lives
here rather than in any one of them:

* ``chat_cli`` (terminal + ``--json`` + history export) and the ``webui`` SSE stream import it;
* library users reach it as ``from spatialomicsgym import clean_answer``.

**Display layer only.** ``go()`` deliberately does NOT apply this: benchmark harnesses and
``huggingface_data/build_dataset.py`` parse ``<solution>(.*?)</solution>`` out of its raw return, so
cleaning there would silently empty every extracted answer. Pinned by
``test/test_public_answer_api.py``.

Stdlib-only by design (``re``). ``chat_cli`` keeps a stdlib-only module-level import list so the CLI
starts instantly and works before any heavy env exists, and ``spatialomicsgym/__init__`` re-exports
from here -- importing ``spatialomicsgym.utils`` instead would drag ~0.7s of scientific stack into
both paths.
"""

from __future__ import annotations

import re
from typing import Any

# THE solution-wrapper pattern. Every reader of a ``<solution>`` block imports this one:
# agent/execution.py (which ends the ReAct run on it), agent/stcoscientist.py (streamed
# transcript), utils/formatting.py (HTML), sog_install/_agent_probe.py and
# huggingface_data/build_dataset.py (published trajectory records). They used to spell it out
# separately and disagreed on IGNORECASE, so a model writing ``<SOLUTION>`` ended its run
# normally and then produced a dataset record with a null ``final_answer``. Models do emit the
# shouted spelling — that is why the ReAct side carries IGNORECASE. Lives here because this
# module is stdlib-only and imports nothing from the package, so any of them can reach it.
SOLUTION_TAG_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)

# Agent wrapper tags (``<solution>…</solution>`` / ``<think>…</think>``) frame the final
# answer for the ReAct parser (agent/execution.py) and the stop sequences; they are not
# meant for a human reader. Strip the tags for plain-text display, keeping the inner
# content. Safe on a stop-sequence-truncated answer: cutting at ``</solution>`` leaves only
# the opening ``<solution>`` tag, and each tag is removed independently either way. This
# mirrors the repo's existing ``<solution>``-neutralisation for the HTML/PDF path
# (utils/formatting.py), extended to the plain-text CLI/SSE front doors that lacked it.
_AGENT_TAG_RE = re.compile(r"</?(?:solution|think)>", re.IGNORECASE)


def in_inline_code(text: str, index: int) -> bool:
    """Whether ``text[index]`` sits inside an inline code span: an odd number of backticks between
    the start of its line and it.

    The one rule the loop's tag readers share for telling a tag the model USED from one it wrote
    ABOUT. `` `<solution>` `` is the model naming the token -- usually paraphrasing the prompt's
    own instructions back -- and never the wrapper it emits. ``cut_at_stop_sequence`` and
    ``close_dangling_tags`` (``agent/execution.py``), :func:`~spatialomicsgym.action.answer_tag_match`
    and the display cleaner below all ask it here, so the stop-sequence cut, the dangling-tag
    repair, the router and the display cannot come apart on which tags are real. A tag on a fence's
    own line, after its three backticks, reads as quoted; no recorded turn has one (0 of 3,364).
    """
    line_start = text.rfind("\n", 0, index) + 1
    return text.count("`", line_start, index) % 2 == 1


def _quoted_in_code_span(text: str, match: re.Match) -> bool:
    """Whether a wrapper tag sits inside an inline code span, i.e. the model QUOTED it.

    A tag between backticks is the model naming the token -- explaining the protocol it was asked
    to follow -- not emitting the wrapper, so removing it mangles the sentence. Two recorded turns
    answer "the fix is to ensure the entire answer is enclosed in ``<solution>...</solution>``" and
    were displayed as "enclosed in ``...``"; two more say the response "is constrained to a single
    final ``<solution>`` block" and were displayed with an empty code span.

    Odd backtick count from the start of the tag's line is the whole test, and the line is the right
    scope both ways: an inline span does not cross a newline, and the wrapper the agent emits is
    never inside one. Position on the line cannot answer this -- an emitted close routinely ends a
    line of prose that itself ends in a code span (``…`qval < 0.05`</solution>``), and an emitted
    close can be followed by more answer (``<think>…</think>done``). A tag inside a triple-backtick
    fenced block is not recognised and still strips, unchanged from before.
    """
    return in_inline_code(text, match.start())


def _strip_agent_tags(text: str) -> str:
    """Strip the agent's ``<solution>``/``<think>`` wrapper tags, keeping inner content.

    Every tag except one the model quoted (:func:`_quoted_in_code_span`). Measured over 3,683
    recorded assistant turns, 585 of which carry a wrapper tag: 4 displayed answers change, all 4
    a quoted mention, and no turn keeps a tag the agent emitted.
    """
    out: list[str] = []
    last = 0
    for match in _AGENT_TAG_RE.finditer(text):
        if _quoted_in_code_span(text, match):
            continue
        out.append(text[last : match.start()])
        last = match.end()
    out.append(text[last:])
    return "".join(out).strip()


_SOLUTION_CLOSE_RE = re.compile(r"</solution>", re.IGNORECASE)
_SOLUTION_OPEN_RE = re.compile(r"<solution>", re.IGNORECASE)
_SOLUTION_TOKEN_RE = re.compile(r"</?solution>", re.IGNORECASE)

# Greedy, unlike SOLUTION_TAG_RE, and only ever applied to bounds solution_block_spans already
# decided -- so it reports the block's inner text without asking the pairing question a second time.
SOLUTION_BLOCK_RE = re.compile(r"<solution>(.*)</solution>", re.DOTALL | re.IGNORECASE)


def _is_structural(text: str, match: re.Match) -> bool:
    """Whether an opening tag sits where a tag the model USED sits, rather than one it mentioned.

    Two accepted positions, both measured against the recorded corpus: the tag starts its line, or
    what precedes it on that line ends in ``>`` -- the ``</execute><solution>`` one-liner, which is
    how 331 of 1,084 recorded turns open the real block. A mention is prose, so it sits after a word.
    """
    before = text[text.rfind("\n", 0, match.start()) + 1 : match.start()].rstrip()
    return not before or before.endswith(">")


def solution_block_spans(text: str) -> list[tuple[int, int]]:
    """Every ``<solution>`` block in ``text`` as ``(start, end)`` offsets, paired from the END.

    Which ``<solution>`` a given ``</solution>`` belongs to is a real question, because a model
    routinely writes the tag it is being asked to use *about* rather than *in* -- both
    ``"…wrap it in <solution> then stop…"`` before the answer, and ``"…enclose the answer in
    `<solution>...</solution>`…"`` inside it. Counting from either end alone gets one of those
    right and the other wrong, and this package used to do exactly that in two places, in opposite
    directions: a non-greedy ``SOLUTION_TAG_RE`` closed the block on the model's own mention and
    published a truncated answer, while a ``opens[-1]``-before-the-last-close scan opened it there
    and dropped the 2,020 characters of answer that came first. Same two messages, two front doors,
    two different halves lost.

    Depth settles it whenever the tags are balanced: walk back from a close, ``+1`` for each close
    and ``-1`` for each open, and the token that brings the count to zero opened the block -- the
    same argument :func:`~spatialomicsgym.action._reanchor_prose_mention` makes for ``<execute>``.
    Blocks are then taken right-to-left, so a stray closer with no opener before it is skipped rather
    than swallowing the real block behind it.

    A mention of *one half* of the wrapper is what depth cannot survive, and models write them --
    ``"this interface only allows me to send either `<execute>` or `<solution>`"``, mid-answer. The
    count is then off by one in whichever direction the lone tag points: a bare close ends the block
    early (leaving a tail, so :func:`_extract_solution` declines it and the reasoning is displayed
    instead of the answer), a bare open is chosen as the opener (so everything answered before that
    sentence is dropped -- the measured worst case published a single ``.``).

    So prefer a *structural* open: one that starts its line, or follows another tag on it. That is
    where a tag the model USED sits; a mention sits mid-sentence. Both halves are load-bearing --
    331 of 1,084 recorded turns open the real block right after ``</execute>`` on one line, so a
    plain "starts its line" rule would drop every one of them. Where no structural open exists the
    model simply wrote the block inline, and the depth walk runs unchanged, so those turns provably
    cannot move.

    Measured over all 6,728 recorded assistant turns: the pairing moves on 2, both of them a
    mid-answer mention of the wrapper, and on both it moves to the real opener at offset 0. Every
    consumer -- :func:`clean_answer`, :func:`~spatialomicsgym.action.extract_final_answer`,
    :func:`~spatialomicsgym.action.final_answer_span`,
    :func:`~spatialomicsgym.action.strip_extracted_blocks` -- is byte-identical on both.
    """
    spans: list[tuple[int, int]] = []
    limit = len(text)
    while True:
        matched: tuple[int, int] | None = None
        for close in reversed(list(_SOLUTION_CLOSE_RE.finditer(text, 0, limit))):
            structural = [m for m in _SOLUTION_OPEN_RE.finditer(text, 0, close.start()) if _is_structural(text, m)]
            if structural:
                matched = (structural[-1].start(), close.end())
                break
            depth = 1
            for token in reversed(list(_SOLUTION_TOKEN_RE.finditer(text, 0, close.start()))):
                depth += 1 if token.group(0).startswith("</") else -1
                if depth == 0:
                    matched = (token.start(), close.end())
                    break
            if matched is not None:
                break
        if matched is None:
            break
        spans.append(matched)
        limit = matched[0]
    spans.reverse()
    return spans


def _extract_solution(text: str) -> str | None:
    """If the agent demarcated its final answer with a COMPLETE, TERMINAL ``<solution>…</solution>``
    block, return ONLY that inner content (nested/stray wrapper tags stripped). Some models — notably
    gpt-5 on the Azure Responses API — emit their whole ReAct monologue (``Loop 1 …``, ``Plan: …``,
    ``Classification: …``) inline *before* the block, all in one message; showing only the block
    content drops that reasoning from the **display** (the raw text the ReAct loop + eval see is
    untouched).

    The answer is the LAST block :func:`solution_block_spans` pairs — not the first — so a model
    restating its answer means the restatement, and a bare/illustrative ``<solution>`` it may type
    mid-reasoning (``"…wrap it in <solution> then stop…"``) isn't swallowed into the answer and
    leaked to the display.

    Returns ``None`` — so the caller shows the whole (tag-stripped) message instead — when there is
    (a) no ``</solution>`` close at all, (b) a close but with non-whitespace text *after* it (the
    agent's real answer ends at ``</solution>``, its stop sequence, so a trailing tail means the block
    is prose — an example or a quote — not the answer wrapper), or (c) a block that is empty once its
    inner ``<think>``/``<solution>`` tags are stripped. This keeps a real answer from being blanked or
    truncated by a stray/illustrative ``<solution>`` pair."""
    spans = solution_block_spans(text)
    if not spans:
        return None
    start, end = spans[-1]
    if text[end:].strip():  # real text follows the close → not the terminal answer wrapper
        return None
    inner = SOLUTION_BLOCK_RE.match(text, start, end).group(1)
    return _strip_agent_tags(inner) or None  # empty after tag-strip → None (caller shows full)


# The Layer-2 routing prompt (agent/stcoscientist.py) asks an ambiguous request to "State
# your classification first" — KNOWLEDGE / ANALYSIS / EXPLORATION. That label is internal
# routing scaffolding; when the model emits it into the final answer it becomes user-facing
# noise (``Classification: KNOWLEDGE\n\n<answer>``, as a live Azure run surfaced). Drop a
# *leading* label line only. Narrow by design: matches only the three literal routing labels
# as the answer's first line (optionally markdown-bolded), so a genuine answer that merely
# discusses classification is left intact. Display-layer only — the raw text the ReAct loop
# and eval see is unchanged, so routing behaviour and benchmark scoring are untouched.
#
# Only the label and its trailing punctuation go -- NOT the rest of its line. The model often
# continues on the same line ("Classification: ANALYSIS. Clustering found 7 domains..."), and
# consuming ``[^\n]*`` dropped that finding from what the user read (hunt 2026-09-30, u13-prompt-9).
# The newline goes too only when nothing but whitespace follows the label on its line.
_CLASSIFICATION_PREFIX_RE = re.compile(
    r"\A\s*[*_]*[ \t]*Classification[*_]*[ \t]*[:\-][ \t*_]*"
    r"(?:KNOWLEDGE|ANALYSIS|EXPLORATION)\b[*_]*[ \t]*[.:;\-\u2014\u2013]*[ \t]*(?:\n+|\Z)?",
    re.IGNORECASE,
)

# …but nothing in the prompt asks for the word "Classification". Its options are bullets of the
# form ``- KNOWLEDGE: User wants explanation``, and a live Azure run answered
# ``<solution>KNOWLEDGE: OK</solution>`` — the bare label, inline, no keyword and no newline, so
# the pattern above cannot match it. Handle that second shape separately, and deliberately
# case-SENSITIVELY: the routing vocabulary is always upper-case, so requiring upper-case keeps
# ordinary prose ("Knowledge: the field has grown…") intact. The label must be followed by ``:``
# or a line break — not ``-`` — since a dash after an upper-case word is far more likely to be a
# genuine heading than a routing leak.
_BARE_CLASSIFICATION_PREFIX_RE = re.compile(
    r"\A\s*[*_]*(?:KNOWLEDGE|ANALYSIS|EXPLORATION)[*_]*[ \t]*(?::[ \t]*|\n+)",
)


def leading_classification(text: str | None) -> str | None:
    """The routing label a final message opens with -- ``KNOWLEDGE``, ``ANALYSIS`` or ``EXPLORATION`` -- or ``None``.

    Read with the two patterns :func:`_strip_leading_classification` removes, from inside ``<solution>`` when
    there is one, so the portal can tell a question answered from knowledge from an analysis step (2026-10-04,
    U6: "what can you do" was recorded as a finished analysis step). Pure; changes nothing it reads.
    """
    body = str(text or "")
    inside = re.search(r"<solution>([\s\S]*?)(?:</solution>|\Z)", body)
    if inside:
        body = inside.group(1)
    for pattern in (_CLASSIFICATION_PREFIX_RE, _BARE_CLASSIFICATION_PREFIX_RE):
        hit = pattern.match(body)
        if hit:
            label = re.search(r"KNOWLEDGE|ANALYSIS|EXPLORATION", hit.group(0), re.IGNORECASE)
            if label:
                return label.group(0).upper()
    return None


def _strip_leading_classification(text: str) -> str:
    """Drop a leading routing-classification label from a final answer (see regex notes).

    Tries the ``Classification: LABEL`` form first, then the bare ``LABEL:`` form; never both, so a
    genuine answer opening with a label can lose at most its own first token. Returns ``text``
    unchanged if stripping would leave nothing — a reply that is *only* the label is better shown
    verbatim than as an empty bubble."""
    for pattern in (_CLASSIFICATION_PREFIX_RE, _BARE_CLASSIFICATION_PREFIX_RE):
        stripped = pattern.sub("", text, count=1).lstrip()
        if stripped != text.lstrip():
            return stripped or text
    return text.lstrip()


# ``prompt_builder.py`` mandates a "Deliberative Thinking Protocol" — numbered ``Loop N`` sections
# the model must work through *before* it writes an <execute> or <solution> tag. Emitted there, it
# is legitimate trace content that solution-extraction already drops. But models also nest it
# INSIDE the tag, and then it is indistinguishable from the answer: a live Azure gpt-5 run replied
# to a procurement question with 1.4 kB of loops ahead of a perfectly good response, all of it
# printed to the user as the agent's reply.
#
# These strings come from our own prompt, so matching them is narrow by construction. Two further
# bounds keep real prose safe: the protocol must be a *preamble* (loops form an uninterrupted run
# from the top), and a single ``Loop 1`` heading is never enough on its own — it needs either the
# protocol title above it or a second loop below it. "Loop 1 of the amplification cycle…" survives.
_PROTOCOL_TITLE_RE = re.compile(r"\A\s*(?:[#>*_\s]*)Deliberative\s+Thinking\s+Protocol[^\n]*\n", re.IGNORECASE)
# Header line, e.g. ``**Loop 3 — Generate 2-3 alternatives:**``. Any dash/colon separator, because
# models re-emit the prompt's em-dash as an en-dash, hyphen or colon interchangeably.
# ``[*_]*`` after the number: ``**Loop 1** — ...`` puts the bold around the number alone.
_LOOP_HEADER_RE = re.compile(r"^[ \t]*[*_#>]*[ \t]*Loop[ \t]+\d+[ \t]*[*_]*[ \t]*[—–:-]", re.IGNORECASE | re.MULTILINE)
# A header line that is only a heading (it ends in the prompt's colon), with the body below it.
_HEADING_ONLY_RE = re.compile(r":[ \t]*[*_]*[ \t]*$")


def _strip_thinking_protocol(text: str) -> str:
    """Drop a leading Deliberative-Thinking-Protocol preamble from a final answer (see notes above).

    Returns ``text`` unchanged when the shape is absent or when stripping would leave nothing — a
    reply that is *only* protocol is better shown verbatim than as an empty bubble."""
    title = _PROTOCOL_TITLE_RE.match(text)
    body_start = title.end() if title else 0
    headers = list(_LOOP_HEADER_RE.finditer(text, body_start))
    if not headers:
        return text
    # Preamble only: nothing but whitespace may precede the first loop (past an optional title).
    if text[body_start : headers[0].start()].strip():
        return text
    if len(headers) < 2 and not title:
        return text  # one bare "Loop 1" heading — assume content, not scaffolding
    # The last loop's own text runs to the next blank line; the answer proper starts after it. When
    # the header line is only a heading, the blank lines right below it belong to the loop, not to
    # the answer: ending there left the reflection list on screen as answer text (u13-prompt-17).
    last = headers[-1]
    line_end = text.find("\n", last.end())
    start = last.end()
    if line_end != -1 and _HEADING_ONLY_RE.search(text[last.start() : line_end]):
        start = line_end
        while start < len(text) and text[start] in " \t\r\n":
            start += 1
    end = text.find("\n\n", start)
    if end == -1:
        return text  # no body after the protocol — show the whole thing rather than nothing
    return text[end:].strip() or text


# The fourth shape (FM-08). The model restates its own working state at the top of the answer: a
# numbered plan with check marks (``1. [✓] Load the subset (completed)``) and status paragraphs
# (``Most recent observation: ...``, ``- KNOWN: ...``, ``Updated plan: ...``). Measured: 21 of 146
# portal answers leaked it -- 15 of 41 answers to ONE identical prompt -- and 16 archived
# SpatialBench solutions open with it. Same bounds as the protocol stripper: a PREAMBLE only
# (paragraphs from the top, stopping at the first that is not scaffold), a checklist needs two
# items so a one-line "1. [x] Done" answer survives, and never an empty bubble.
_CHECKLIST_ITEM_RE = re.compile(r"^[ \t]*\d+\.[ \t]*\[[^\]\n]{0,3}\]")
_STATUS_OPENER_RE = re.compile(
    r"^[ \t]*[*_#>]*[ \t]*(?:-[ \t]*)?(?:(?:the[ \t]+)?most[ \t]+recent[ \t]+observation|recent[ \t]+observation"
    r"|known|updated[ \t]+plan)[*_]*[ \t]*(?:[:/(,]|was\b|is\b|showed\b|says\b)",
    re.IGNORECASE,
)


def _is_scaffold_paragraph(paragraph: str) -> bool:
    lines = [ln for ln in paragraph.split("\n") if ln.strip()]
    if not lines:
        return True
    if _STATUS_OPENER_RE.match(lines[0]):
        return True
    items = sum(1 for ln in lines if _CHECKLIST_ITEM_RE.match(ln))
    continuation = all(_CHECKLIST_ITEM_RE.match(ln) or ln[:1] in (" ", "\t") for ln in lines)
    return items >= 2 and continuation and bool(_CHECKLIST_ITEM_RE.match(lines[0]))


def _strip_status_preamble(text: str) -> str:
    """Drop a leading plan/status preamble from a final answer (see notes above). Returns ``text``
    unchanged when there is none, or when stripping it would leave nothing."""
    paragraphs = re.split(r"\n[ \t]*\n", text.strip())
    i = 0
    while i < len(paragraphs) and _is_scaffold_paragraph(paragraphs[i]):
        i += 1
    if i == 0:
        return text
    rest = "\n\n".join(paragraphs[i:]).strip()
    return rest or text


def answer_to_text(answer: Any) -> str:
    """Flatten whatever ``go()`` returned into readable text.

    ``go()`` returns the raw ``message.content``. For a plain turn that is a ``str``, but whenever
    the final turn carries tool use -- the normal case for an agent that just ran MCP tools --
    langchain hands back a LIST of content blocks::

        [{"type": "text", "text": "..."}, {"type": "tool_use", "id": "toolu_01", ...}]

    Printing that directly dumps a Python list repr at the reader, and running a regex over it
    raises ``TypeError``. Only the text blocks are the answer; the rest is machinery, so it is
    dropped rather than rendered. Some langchain versions yield block *objects* carrying a ``.text``
    attribute instead of dicts, so both are handled.

    Anything else degrades to ``str(...)`` rather than raising: this sits on the far end of an LLM
    call, where an unexpected shape must not take the caller's program down. ``None`` maps to ``""``.
    """
    if isinstance(answer, str):
        return answer
    if answer is None:
        return ""
    if isinstance(answer, list):
        parts: list[str] = []
        for block in answer:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                txt = block.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
            else:  # some langchain versions yield block objects with a `.text` attribute
                txt = getattr(block, "text", None)
                if isinstance(txt, str):
                    parts.append(txt)
        if parts:
            return "\n".join(parts)
        # A list with no text-bearing blocks has nothing to join. Fall through to ``str()`` rather
        # than returning "" -- showing the reader nothing at all hides that the agent ended on a
        # tool call with no prose, which is exactly the case they need to see.
    return str(answer)


def clean_answer(answer: Any) -> str:
    """Canonical display cleanup for a *final* agent answer: if a ``<solution>`` block is present show
    ONLY its content (dropping any reasoning/plan the model emitted before it), else strip stray
    ``<solution>``/``<think>`` wrapper tags; then drop a leading routing-classification line, a
    leading thinking-protocol preamble, and a leading plan/status preamble. Display-layer only — the raw text the ReAct loop and eval
    see is unchanged, so routing behaviour and benchmark scoring are untouched. Intermediate
    reasoning steps keep both (that is legitimate trace content); only this *final*-answer path
    removes them.

    Accepts the value ``go()`` really returns -- ``str`` or a list of content blocks -- via
    :func:`answer_to_text`, so ``clean_answer(agent.go(task)[1])`` is safe on a tool-use turn.
    """
    text = answer_to_text(answer)
    sol = _extract_solution(text)
    body = sol if sol else _strip_agent_tags(text)
    return _strip_status_preamble(_strip_thinking_protocol(_strip_leading_classification(body)))
