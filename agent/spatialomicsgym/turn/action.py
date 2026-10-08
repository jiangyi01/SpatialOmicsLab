"""How to read the *action* out of one assistant turn.

The ReAct loop asks two questions of every model turn: "is there code to run?" and "is this the
final answer?". :mod:`spatialomicsgym.answer` owns the second one; this module owns the first, plus
the precedence rule between them.

It lives here rather than in ``agent/execution.py`` for the same reason ``SOLUTION_TAG_RE`` lives in
``answer.py``: readers outside the ReAct loop need the same answers, and importing ``execution``
costs ~0.8s of langchain/langgraph **and loads the user's ``.env``** -- unacceptable for
``huggingface_data/build_dataset.py``, a stdlib-speed script whose job is to publish a public
corpus. That builder used to restate these rules with its own regexes and got three of them
backwards, publishing 314k characters fewer executed code than the runs contained, 175 attempts
answered by a ``<solution>`` the loop had *stripped*, and the string ``or`` as executed python.

``execution.py`` re-exports every public name below, so the ReAct loop and the corpus read one
definition and cannot drift again; the helpers the split introduced stay private to this module.
That correspondence is a test, not a promise -- ``final_answer_span`` was added here after the
re-export list was written, and this sentence claimed it was reachable from ``execution`` for as
long as it was not.

Stdlib-only by design (``re``, ``ast``), importing nothing from this package but ``answer``.
"""

from __future__ import annotations

import ast
import re

from spatialomicsgym.answer import SOLUTION_BLOCK_RE, SOLUTION_TAG_RE, in_inline_code, solution_block_spans

_EXECUTE_TAG_RE = re.compile(r"<execute>(.*?)</execute>", re.DOTALL | re.IGNORECASE)

_EXECUTE_OPEN_RE = re.compile(r"<execute>", re.IGNORECASE)
_EXECUTE_CLOSE_RE = re.compile(r"</execute>", re.IGNORECASE)


class _Block:
    """One ``<execute>`` block, after its closing tag has been decided.

    Stands in for a ``re.Match`` because the close can be re-chosen (see :func:`_execute_blocks`),
    and exposes only the four things the readers below ask a match for.
    """

    __slots__ = ("_body", "_body_start", "_end")

    def __init__(self, body: str, body_start: int, end: int) -> None:
        self._body, self._body_start, self._end = body, body_start, end

    def group(self, _n: int = 0) -> str:
        return self._body

    def start(self, n: int = 0) -> int:
        return self._body_start if n == 1 else self._body_start - len("<execute>")

    def end(self) -> int:
        return self._end

    def span(self) -> tuple[int, int]:
        return (self.start(), self._end)


def _execute_blocks(message: str) -> list[_Block]:
    """Every ``<execute>`` block, pairing each open with the close that really ends it.

    ``_EXECUTE_TAG_RE`` is non-greedy, so the FIRST ``</execute>`` closes the block -- including one
    the model wrote *inside* the cell, in a comment, a docstring, a string literal or a prompt
    template. That models write the literal tag is measured in this file already
    (:func:`_is_prose_mention` records a turn where the model echoed the reprompt's "either
    ``<execute>`` or ``</execute>``"), and the system prompt spells it out verbatim in four places.

    The failure is silent and it is the worst shape there is. Driven::

        <execute>
        adata = ad.read_h5ad('/data/slide.h5ad')
        print(adata.shape)
        # reminder: always close the block with </execute>
        adata.write('/out/result.h5ad')
        print('WROTE /out/result.h5ad')
        </execute>

    The extracted program stops at the comment. It is still VALID PYTHON, so it runs, and the
    observation is a clean shape with no error in it. The write never happened; the model reads the
    observation as success and answers with a path that does not exist. Nothing in the transcript
    says the cell was cut -- which is the same class of failure ``close_dangling_tags`` exists to
    prevent, from a different cause and with no defence.

    The rule: among the closes that could end this open, take the LAST whose body parses, and stop
    at any ``<execute>`` that opens in between. That guard is what keeps two real blocks two blocks:
    the text between them contains the tags themselves, which are not Python, so a merged body does
    not parse -- and where they are genuinely adjacent the intervening open stops it before the
    parse is even tried. Where nothing parses, the first close is kept unchanged, so a model that
    wrote broken code still sees its own SyntaxError.
    """
    blocks: list[_Block] = []
    pos = 0
    while True:
        opening = _EXECUTE_OPEN_RE.search(message, pos)
        if opening is None:
            return blocks
        body_start = opening.end()
        closes = list(_EXECUTE_CLOSE_RE.finditer(message, body_start))
        if not closes:
            return blocks
        # Candidates stop at the next open: past it we would be merging two separate blocks.
        nxt = _EXECUTE_OPEN_RE.search(message, body_start)
        limit = nxt.start() if nxt else len(message)
        within = [c for c in closes if c.start() < limit]

        def parses(close: re.Match[str], start: int = body_start) -> bool:
            try:
                ast.parse(message[start : close.start()].strip())
            except SyntaxError:
                return False
            return True

        chosen = next((c for c in reversed(within) if parses(c)), None) if len(within) > 1 else None
        if chosen is None and nxt is not None and not (within and parses(within[0])):
            # That next open may sit INSIDE this cell -- a string literal, a template, a comment --
            # when the provider applied no stop sequence (the gpt-5 family). The limit above then left
            # only the close inside the literal, and the fragment before it ran instead of the cell
            # (u12-react-13). A later close whose body PARSES proves the open was the cell's own text:
            # two real blocks never merge into valid Python, because the tags between them are not.
            chosen = next((c for c in reversed(closes) if c.start() >= limit and parses(c)), None)
        if chosen is None:
            chosen = within[0] if within else closes[0]
        blocks.append(_Block(message[body_start : chosen.start()], body_start, chosen.end()))
        pos = chosen.end()


# A leading language token (```python / ```py / ```r / ```bash …) is stripped ONLY when it is
# immediately followed by a newline — the standard markdown fence shape. Without that newline guard
# the old ``(?:python|bash|r)?`` alternative ate the leading ``r`` of an inline ```read_h5ad()``` /
# ```return x``` block (→ ``ead_h5ad()`` / ``eturn x``) and leaked unknown tags (``py`` / ``python3``)
# as a bogus first code line. An inline single-line fence with no newline is now kept verbatim.
_CODE_FENCE_RE = re.compile(r"```[ \t]*(?:[A-Za-z0-9_+-]+[ \t]*(?=\r?\n))?\r?\n?(.*?)```", re.DOTALL | re.IGNORECASE)

# A block whose FIRST non-blank content is one of these is not python and must never be judged by
# compile() or folded in with python blocks -- exactly how classify_and_clean_code decides.
# A POSIX shebang is the most explicit statement of language a block can carry, and it is what a
# model writes unprompted -- the product writes one itself in utils/execution.py. Without these,
# a "#!/bin/bash" block fell through to the python branch and compile() reported whatever the first
# shell line looked like as an expression ("head -n 20 x.csv" -> "invalid decimal literal"), naming
# neither the language nor the marker.
_SHEBANG_BASH = (
    "#!/bin/bash",
    "#!/usr/bin/env bash",
    "#!/bin/sh",
    "#!/usr/bin/env sh",
    "#!/bin/zsh",
    "#!/usr/bin/env zsh",
)
_SHEBANG_R = ("#!/usr/bin/env Rscript", "#!/usr/bin/Rscript", "#!/usr/local/bin/Rscript")
_LANGUAGE_MARKERS = (
    (
        "#!R",
        "# R code",
        "# R script",
        "#!CLI",
        "#!BASH",
        "# Bash script",
    )
    + _SHEBANG_BASH
    + _SHEBANG_R
)


def _is_language_marked(block: str) -> bool:
    # Scanning deeper lines would false-fire on a legit python comment ("# R code equivalent: ...") or
    # an embedded shell-script string; a marker that isn't first is a malformed block that runs as
    # python either way.
    return block.strip().startswith(_LANGUAGE_MARKERS)


#: Punctuation English does not use the way code does. One hit is enough to call a one-liner an
#: action rather than a sentence fragment: brackets of any kind, an assignment or comparison, a
#: statement separator, a shell operator, a glob, a flag, or a path segment.
#:
#: The prose this has to keep rejecting is the body BETWEEN two inline tag mentions -- `" or "`,
#: `" and "`, `" your code here "` -- and none of it carries any of these.
_CODE_SHAPED = re.compile(r"[(){}\[\]=;*|]|&&|--|/\w")


def _looks_like_code(body: str) -> bool:
    """Is this one-liner code that happens to be broken, or a fragment of a sentence?

    Asked only when :func:`_is_prose_mention` has already established the narrow gate -- one line,
    tag opened mid-sentence -- and ``ast.parse`` has raised. Before this, "does not parse" was read
    as "is prose", which is the opposite of what the gate's own docstring promises: *"Broken code
    still runs; its SyntaxError is how the model learns to fix it."*

    Found live (SpatialBench-Long arm v8, lineage_metastasis r1): the model pasted a SHELL command
    into the python REPL on one line, glued to the end of a sentence. The parse raised, the block
    was dropped, ``extract_runnable_code`` returned None, the execute+solution guard in
    ``generate()`` could not fire, and the premature ``<solution>`` -- "Need the tool output from
    the last execute block to proceed" -- became the final answer. 75 seconds of an 18,000-second
    budget, and the model never saw the SyntaxError it needed.

    Ambiguity resolves toward running it. A prose fragment that happens to carry a bracket becomes
    one SyntaxError observation the model reads and moves past; a dropped action becomes a run that
    ends without an answer.
    """
    return bool(_CODE_SHAPED.search(body))


def _is_prose_mention(message: str, match: re.Match) -> bool:
    """True when the tags were written *about* rather than *used*.

    Found live: the no-tags reprompt spells the tags out, so the model answered "...exactly one XML
    tag: either <execute> or </execute>". Those two inline mentions pair up, and the text between
    them -- ``" or "`` -- went to the REPL as python. The resulting ``invalid syntax`` observation
    counts as a real execution, so the no-tags strike guard never fires; the run spent 14 steps
    apologising about formatting and never attempted the user's deconvolution.

    The gate is deliberately narrow -- the body stays on one line, the tag opens mid-sentence rather
    than on its own line, and the body carries no ``#!R``/``#!CLI`` marker -- because real code is
    multi-line or starts its own line. Inside that gate the body is prose if it cannot run at all, or
    if it is a lone literal (``<execute> ... </execute>``, the placeholder-elision idiom): neither is
    an action. Broken *code* still runs; its SyntaxError is how the model learns to fix it --
    which is what :func:`_looks_like_code` decides, and what this used to get backwards for the one
    case that reaches the parse check.
    """
    body = match.group(1)
    if "\n" in body:
        return False
    line_start = message.rfind("\n", 0, match.start()) + 1
    if not message[line_start : match.start()].strip():
        return False
    if _is_language_marked(body):
        return False
    try:
        # Strip first: leading whitespace on a one-liner is an IndentationError in "exec" mode, and
        # ``<execute> print(x) </execute>`` is a real block. This is also what actually gets run.
        tree = ast.parse(body.strip())
    except SyntaxError:
        # Broken code is still an action. Only a body that is ALSO shapeless is prose.
        return not _looks_like_code(body)
    return len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)


_NESTED_EXECUTE_OPEN_RE = re.compile(r"<execute>", re.IGNORECASE)


def _reanchor_prose_mention(body: str) -> str:
    """Rescue a block whose opening tag was stolen by a prose mention of ``<execute>``.

    ``_EXECUTE_TAG_RE`` is non-greedy, so it anchors on the EARLIEST ``<execute>`` — and a model
    that reasons about its own formatting ("Retry with a clean Python ``<execute>`` block") puts one
    in its prose. The match then runs from that mention to the real ``</execute>``, swallowing the
    prose *and* the genuine opening tag, and the executor compiles English. Observed live on a real
    Visium run: ``Error: unterminated string literal (detected at line 1)`` on a block of perfectly
    valid scanpy code. It is also self-reinforcing — being told "invalid syntax" makes the model
    write more prose about ``<execute>`` blocks, breaking the retry the same way.

    Tags do not nest, so an ``<execute>`` inside the body means the outer opener was prose: the real
    code starts after the LAST one. Applied only when it can strictly help — the outer body must
    fail to parse and the re-anchored body must parse — so code that legitimately contains the
    literal string ``<execute>`` (a prompt template) compiles fine and is left alone, and a block
    that is broken either way still reaches the model as its own SyntaxError.
    """
    hits = list(_NESTED_EXECUTE_OPEN_RE.finditer(body))
    if not hits:
        return body
    inner = body[hits[-1].end() :]
    if not inner.strip():
        return body
    outer_kind, outer_clean = classify_and_clean_code(body)
    inner_kind, inner_clean = classify_and_clean_code(inner)
    if inner_kind != "python":
        # The inner block names its own language (#!R / #!BASH / #!CLI) — unambiguous, and the outer
        # reading would have run English through that interpreter.
        return inner
    if outer_kind != "python":
        return body  # outer is language-marked: not parseable here, leave the existing behaviour
    try:
        ast.parse(outer_clean)
    except SyntaxError:
        pass
    else:
        return body  # the outer reading is valid code — nothing to rescue
    try:
        ast.parse(inner_clean)
    except SyntaxError:
        return body  # both broken: keep the original so the model sees its own error
    return inner


def dropped_execute_blocks(message: str) -> list[str]:
    """The ``<execute>`` blocks this turn wrote and did NOT run, described one per entry.

    When a turn carries several blocks and any of them names a language, only the first runs:
    folding ``#!R`` or ``#!BASH`` into a python cell would be worse than dropping it. That decision
    is right. Making it SILENTLY was not -- ``strip_extracted_blocks``' own docstring measures it at
    7 of 40 recorded multi-execute turns, where "blocks 2..N are then in no field at all". The model
    wrote two cells, saw one observation, and reasoned as though both had run: python writes a CSV,
    bash was supposed to move it into place, and the answer names a file that is not there.

    Naming them in the observation is the minimum honest thing. Running them properly -- dispatching
    per language, in order -- is a larger change and is not what this does.
    """
    blocks = [m.group(1) for m in _execute_blocks(message) if m.group(1).strip() and not _is_prose_mention(message, m)]
    if len(blocks) < 2 or not any(_is_language_marked(b) for b in blocks):
        return []
    said: list[str] = []
    for i, body in enumerate(blocks[1:], start=2):
        marker = body.strip().split("\n", 1)[0].strip()
        # A comment-style marker (``# R script ...``) names R or bash as surely as ``#!R`` does.
        kind = classify_and_clean_code(body)[0]
        language = marker if marker.startswith("#!") else {"r": "R", "bash": "bash"}.get(kind, "python")
        first = next((line for line in body.strip().splitlines() if line.strip() and not line.startswith("#!")), "")
        said.append(f"block {i} ({language}): {first.strip()[:80]}")
    return said


_SOLUTION_OPENER_RE = re.compile(r"<solution>", re.IGNORECASE)


def _answer_blocks(message: str) -> list[re.Match]:
    """The ``<solution>`` blocks the loop reads as answers: ``SOLUTION_TAG_RE``'s matches, opened
    outside an inline code span.

    A `` `<solution>` `` between backticks is the model naming the tag, and models name it most
    often while paraphrasing the prompt's own gate -- "a final `<solution>` must be preceded by an
    observation" -- right before the ``<execute>`` they then write. Read as an answer, that mention
    did two things. ``close_dangling_tags`` found no closer and appended ``</solution>`` at the end
    of the turn; ``generate()`` then saw execute+solution, logged a premature answer the model never
    wrote, and :func:`strip_premature_solution` deleted everything from the mention to the real
    ``<execute>``. Measured 2026-09-25 over the 3,364 recorded model turns of the archive and
    E-01..E-03: 31 turns in 29 trials carry what that cut leaves -- the executed block opening
    straight after a backtick, "a final `<execute>". On 25 the model's KNOWN / ASSUMED /
    Alternatives and plan were erased and its cell ran as written (E-03: cn7 r2 and r3, follicle r2,
    oocyte r2); on 6 the cut also took the real opener or fused the cell, and English ran as python.

    Scanned opener by opener so a quoted mention cannot hide the real block behind it: a plain
    ``finditer`` starts its match at the mention and runs to the real ``</solution>``. Where no
    opener is quoted this is exactly ``SOLUTION_TAG_RE.finditer``.
    """
    blocks: list[re.Match] = []
    end = 0
    for opener in _SOLUTION_OPENER_RE.finditer(message):
        if opener.start() < end or in_inline_code(message, opener.start()):
            continue
        block = SOLUTION_TAG_RE.match(message, opener.start())
        if block is not None:
            blocks.append(block)
            end = block.end()
    return blocks


def answer_tag_match(message: str | None) -> re.Match | None:
    """The first ``<solution>`` block in ``message`` that is a block and not a quoted mention, or None.

    What ``generate()`` asks to decide "is this turn answering?" -- and so what decides whether the
    execute+solution guard strips anything -- and what :func:`strip_premature_solution` then removes.
    One reader for both, so the router cannot see an answer the strip does not, or the reverse.
    """
    if not message:
        return None
    blocks = _answer_blocks(message)
    return blocks[0] if blocks else None


def _runnable_code(message: str) -> tuple[str | None, list[tuple[int, int]]]:
    """The code and the regions of ``message`` it was taken from, decided once for both readers.

    Returning the spans alongside the code is what lets :func:`strip_extracted_blocks` remove exactly
    what ran without restating any of the rules below — the tag can match text that is not an action
    (a prose mention, a block a language-marked sibling stopped us running) and an action can come
    from text the tag does not match at all (the fence fallback).
    """
    # A blank / whitespace-only <execute></execute> body is NOT an action: filter it so we fall through
    # to the solution/fence check. Otherwise an empty execute emitted beside a real <solution> looked like
    # runnable code, tripping the execute+solution guard below and stripping the answer.
    matches = [m for m in _execute_blocks(message) if m.group(1).strip() and not _is_prose_mention(message, m)]
    if matches:
        blocks: list[str] = []
        spans: list[tuple[int, int]] = []
        for m in matches:
            body = m.group(1)
            block = _reanchor_prose_mention(body)
            blocks.append(block)
            # Re-anchoring means the opening tag was prose and the code is a SUFFIX of the body, so
            # the prose the tag swallowed is the model's reasoning and only the suffix ran.
            spans.append(m.span() if len(block) == len(body) else (m.start(1) + len(body) - len(block), m.end()))
        if len(blocks) == 1:
            # A body that is exactly one fenced block runs as the language its fence names (hunt
            # 2026-09-30, u23-transcriptomics-skills-15): a ```bash fence inside <execute> reached the
            # REPL whole, as python, where the bare-fence fallback below already read the language.
            fence = _CODE_FENCE_RE.fullmatch(blocks[0].strip())
            if fence and "```" not in fence.group(1):
                return _with_fence_language(fence), spans
            return blocks[0], spans
        # MULTIPLE <execute> blocks in ONE turn: run them all (not just the first). A stop-honoring model
        # emits one block per turn, so this branch is never taken for it; but gpt-5's stop sequence is
        # stripped server-side, so it sometimes emits its whole multi-step plan as several <execute> blocks
        # in a single message — dropping blocks 2..N made it "solve" from incomplete results. The blocks
        # share the persistent REPL namespace, so concatenating them is equivalent to running them in
        # order. Only safe for plain-python blocks: if any carries a language marker (#!R/#!CLI/#!BASH),
        # fall back to the first so R/bash is never folded into python.
        if any(_is_language_marked(b) for b in blocks):
            return blocks[0], spans[:1]
        return "\n".join(b.strip() for b in blocks), spans
    if not _answer_blocks(message):
        fence = _CODE_FENCE_RE.search(message)
        if fence:
            return _with_fence_language(fence), [fence.span()]
    return None, []


#: A fence's language token, read off the match (the body is group 1; the token is not captured).
_FENCE_LANG_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]+)[ \t]*\r?\n")
_FENCE_BASH = frozenset({"bash", "sh", "shell", "zsh", "console"})
_FENCE_R = frozenset({"r"})


def _with_fence_language(fence: re.Match[str]) -> str:
    """The fence's body, marked with the language the fence named when that is not python.

    The fallback stripped the language token and returned the body alone, so ```` ```bash ```` and
    ```` ```r ```` code reached ``classify_and_clean_code`` unmarked and ran as PYTHON -- ``ls -la``
    answered with a SyntaxError, ``library(SPARK)`` with a NameError, both about code that was never
    python (u12-react-12). A body that already carries a marker or a shebang keeps its own.
    """
    body = fence.group(1)
    head = _FENCE_LANG_RE.match(fence.group(0))
    lang = head.group(1).lower() if head else ""
    if body.lstrip().startswith("#!"):
        return body
    if lang in _FENCE_BASH:
        return "#!BASH\n" + body
    if lang in _FENCE_R:
        return "#!R\n" + body
    return body


def extract_runnable_code(message: str | None) -> str | None:
    """Return the runnable code in ``message`` (or ``None``): the canonical ``<execute>...</execute>``
    body (case-insensitive), else a fenced ```` ```code``` ```` block IF there is no ``<solution>`` (a
    fence alongside a solution is illustrative, not an action). BOTH ``generate`` (the router) and
    ``execute`` (the extractor) call this, so a turn the router sends to ``execute`` always yields code —
    the old router/extractor mismatch (markdown fallback, or a ``<Execute>`` case difference) routed to
    ``execute``, extracted nothing, appended no observation, and looped the graph to the recursion limit."""
    if not message:
        return None
    return _runnable_code(message)[0]


def _without_spans(message: str, spans: list[tuple[int, int]]) -> str:
    """``message`` with each ``(start, end)`` region removed; spans sorted and non-overlapping."""
    if not spans:
        return message
    out: list[str] = []
    cursor = 0
    for start, end in spans:
        out.append(message[cursor:start])
        cursor = end
    out.append(message[cursor:])
    return "".join(out)


def _minus(span: tuple[int, int], keep: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The parts of ``span`` not covered by ``keep`` (sorted, non-overlapping), in order."""
    start, end = span
    pieces: list[tuple[int, int]] = []
    for keep_start, keep_end in keep:
        if keep_end <= start or keep_start >= end:
            continue
        if keep_start > start:
            pieces.append((start, keep_start))
        start = keep_end
        if start >= end:
            return pieces
    if start < end:
        pieces.append((start, end))
    return pieces


_EXECUTE_OPENER_BEFORE_RE = re.compile(r"<execute\b[^>]*>\s*\Z", re.IGNORECASE)
_EXECUTE_CLOSER_AFTER_RE = re.compile(r"\A\s*</execute\s*>", re.IGNORECASE)


def _with_block_tags(message: str, span: tuple[int, int]) -> tuple[int, int]:
    """``span`` (a code body) widened to its own ``<execute>`` opener and ``</execute>`` closer.

    Keeping only the body let a cut remove the opener: a turn that mentioned both tags in prose --
    "either an `<execute>` or `<solution>` tag" -- had the prose ``<solution>`` closed by
    ``close_dangling_tags`` at the very end, the resulting block wrapped the real ``<execute>``, and
    the cut deleted that opener. Extraction then anchored on the backticked prose mention and ran
    "` or `" as line 1 of the cell (control arm 2026-09-25, cumulus r3: SyntaxError, one step lost).
    """
    start, end = span
    head = message[max(0, start - 200) : start]
    m = _EXECUTE_OPENER_BEFORE_RE.search(head)
    if m is not None:
        start -= len(head) - m.start()
    tail = _EXECUTE_CLOSER_AFTER_RE.match(message, end)
    if tail is not None:
        end = tail.end()
    return start, end


def strip_premature_solution(message: str | None) -> str:
    """``message`` with a premature ``<solution>`` removed and the honoured action left in place.

    What ``generate()`` needs when a turn hallucinates a whole ReAct transcript at once: the answer
    must go, so the next turn does not read the run as finished, and the code must stay, because the
    router has already announced it will be honoured and ``execute()`` re-reads it from the message
    this returns.

    A blanket ``SOLUTION_TAG_RE.sub("", msg)`` does the first and breaks the second whenever the two
    blocks are not disjoint. Of the 175 turns in the recorded corpus that emit both tags, 31 are
    nested: 29 wrap the code *inside* the answer (``<solution>… <execute>c</execute> …</solution>``),
    where the sub deletes the code along with its wrapper -- ``execute()`` then finds nothing and
    tells the model to "put the code inside a single ``<execute>...</execute>`` block", which is what
    it did -- and 2 overlap partially, where the sub cuts across ``</execute>`` and the dangling-tag
    repair re-closes the wound around a truncated program.

    So the answer spans are subtracted MINUS the span the code came from. Text that is part of the
    action is never touched, including a literal ``<solution>`` built inside a cell (a prompt
    template), which the blanket sub silently edited.
    """
    if not message:
        return ""
    code = [_with_block_tags(message, span) for span in _runnable_code(message)[1]]
    if not code:
        return message
    cuts: list[tuple[int, int]] = []
    # Blank blocks included: generate() detects the answer with ``answer_tag_match``, which reads
    # these same blocks, so an empty <solution></solution> triggers this path too and has always
    # been removed with the rest. A quoted `<solution>` is not one of them (:func:`_answer_blocks`).
    for m in _answer_blocks(message):
        cuts.extend(_minus(m.span(), code))
    return _without_spans(message, cuts)


def _final_answer_match(message: str) -> re.Match | None:
    """The ``<solution>`` match :func:`extract_final_answer` reads, decided once for both readers.

    Same reason :func:`_runnable_code` returns spans: :func:`strip_extracted_blocks` has to remove
    exactly the block that got published, and a second regex asking the question again is how the
    two came apart in the first place.

    Which close belongs to which open is :func:`~spatialomicsgym.answer.solution_block_spans`'
    question, not this function's. Reading it off a non-greedy ``SOLUTION_TAG_RE`` closes the block
    on a literal ``</solution>`` the model wrote inside its own answer -- 2 recorded turns, each
    published 2,840 characters short, cut mid-sentence at the model's own note about the wrapper it
    had been told to use.

    And none on a turn the router reads no answer in (:func:`answer_tag_match`): a wrapper the model
    only quoted -- "I will answer in `<solution>...</solution>` form" -- does not end the run, so
    there is nothing to publish and its prose keeps the pair.
    """
    if _runnable_code(message)[0] is not None or not _answer_blocks(message):
        return None
    blocks = [SOLUTION_BLOCK_RE.match(message, *span) for span in solution_block_spans(message)]
    non_blank = [m for m in blocks if m is not None and m.group(1).strip()]
    return non_blank[-1] if non_blank else None


def strip_extracted_blocks(message: str | None) -> str:
    """``message`` with whatever the corpus publishes elsewhere removed, and nothing else.

    The inverse of the pair :func:`extract_runnable_code` / :func:`extract_final_answer`, for readers
    that want the turn's prose: what the model said, once what it *did* and what it *concluded* are
    accounted for in fields of their own. The precedence rule means at most one of the two blocks
    exists on any turn, so this is one span set, never a merge.

    Deleting everything tag-shaped is not the same thing, and on both tags the difference is large.
    Measured over the recorded corpus:

    ``<execute>`` -- 40 turns, 40,460 characters:

    * 31 turns where the model wrote the literal ``<execute>`` in its own prose ("retry with a clean
      ``<execute>`` block"). The non-greedy tag anchors on that mention, so the match runs from
      mid-sentence to the real closing tag; :func:`_reanchor_prose_mention` rescues the code, and a
      regex strip deletes the swallowed paragraph with it. 33,358 characters.
    * 7 turns whose second ``<execute>`` names another language, so only block 1 ran (see
      :func:`_runnable_code`). Blocks 2..N are then in no field at all: not in the code, because they
      did not run, and not in the prose, because they are tag-shaped. 7,056 characters.
    * 2 turns quoting the no-tags reprompt -- "exactly one XML tag: either ``<execute>`` or
      ``</execute>``" -- which :func:`_is_prose_mention` correctly declines to run and a regex strip
      silently empties, leaving "exactly one XML tag: either".

    ``<solution>`` -- 178 turns, 318,257 characters. Only ONE block per turn is ever published, and
    none at all on a turn that also ran code:

    * 142 turns carry a premature answer the loop discarded, disjoint from the code. 230,611
      characters -- the model answering before it had run anything, which is the single most
      interesting thing a trajectory corpus can record about a turn.
    * 29 turns wrap the code *inside* the answer, so a blanket sub takes the action with it.
      70,633 characters.
    * 6 turns restate their answer; only the last block wins, and the drafts before it are the model
      correcting itself. 16,992 characters.
    * 1 empty ``<solution></solution>``.

    It also runs the other way on both tags: code taken from a bare markdown fence is an action the
    ``<execute>`` regex cannot see, so a tag-only strip publishes it twice.

    The two halves cannot be chained in either order, because each unguards the other -- removing the
    answer makes an illustrative fence look like an action (the fence fallback fires only when there
    is no ``<solution>``), and removing the code makes a premature answer look like the answer. Both
    span sets are therefore taken against the original ``message``.
    """
    if not message:
        return ""
    code = _runnable_code(message)[1]
    if code:
        return _without_spans(message, code)
    answer = _final_answer_match(message)
    return _without_spans(message, [answer.span()]) if answer else message


def extract_final_answer(message: str | None) -> str | None:
    """The ``<solution>`` this turn ends the run on, or None if the loop would not stop here.

    Three rules, all of which a naive ``SOLUTION_TAG_RE.search()`` gets wrong:

    * **An ``<execute>`` in the same turn wins.** The GPT-5 family sometimes hallucinates a whole
      ReAct transcript in one message -- ``<execute>…</execute> … <solution>done</solution>`` -- so
      the loop honours the execute and *strips* the premature solution (``generate()`` in
      ``agent/execution.py``) rather than ending on an answer produced before any tool ran.
    * **The LAST block, not the first.** A model that restates its answer means the restatement;
      ``answer._extract_solution``, which every front door reads through, takes the last one for
      exactly this reason.
    * **A wrapper quoted in backticks is not one.** ``generate()`` routes on
      :func:`answer_tag_match`, which skips an opener inside an inline code span, so a turn that
      only names `` `<solution>...</solution>` `` keeps the loop going and has no answer here.
    """
    if not message:
        return None
    match = _final_answer_match(message)
    return match.group(1) if match else None


def final_answer_span(message: str | None) -> tuple[int, int] | None:
    """Where in ``message`` the block :func:`extract_final_answer` published starts and ends.

    Same reason :func:`_runnable_code` returns spans alongside the code: a caller that has to reason
    about the text *around* the published block must ask the reader that chose it, not a second
    regex asking the question again. The one caller is ``huggingface_data/build_dataset.py``, which
    needs the end of the answer as a floor -- a benchmark wave writes many runs to one captured
    stdout, so what follows the run's last block is the next run's driver output rather than
    anything the model said.
    """
    if not message:
        return None
    match = _final_answer_match(message)
    return match.span() if match else None


# --------------------------------------------------------------------------------------------- #
# Did the step fail?  The ONE regex the ReAct loop acts on, kept here so the corpus miner
# (``huggingface_data/mine_failures.py``) can read observations with the loop's own judgement and
# without langchain. ``execution.py`` imports it from here.
# --------------------------------------------------------------------------------------------- #

# What an execute-observation looks like when the step failed: the REPL's own prefix (`Error: ` at line
# start), a Python traceback, the R/bash wrappers, the run_with_timeout message, an MCP-tool JSON error
# status, or the MCP wrapper RuntimeError. Deliberately NOT the bare word "error" -- that false-matches
# SUCCESSFUL tool output like "standard error = 0.03", a JSON ``"errors": None`` field,
# "reconstruction_error", MACS2's "Standard Error:" log, or "0 errors found".
_EXEC_ERROR_RE = re.compile(
    r"(?:^|\n|<observation>)\s*Error: "
    r"""|["']status["']\s*:\s*["']error["']"""  # a tool JSON/dict reporting failure (either quote style)
    r"|Traceback \(most recent call last\)"
    r"|Error running (?:R code|[Bb]ash script)"  # run_bash_script emits capital "Bash script"
    r"|Error in execution: "  # run_with_timeout thread-raised path
    r"|ERROR: Code execution timed out"
    r"|MCP tool execution failed"
)
EXEC_ERROR_RE = _EXEC_ERROR_RE


#: How every line the REPL puts above a cell's warnings begins (``support_tools._WarningLog``): the
#: block's header, and the one line the block shrinks to when the output fills the observation --
#: which is the whole block. Defined here, in the langchain-free module, so the error signature
#: below can take out the block it heads; a prefix of the header every earlier REPL wrote, so an
#: observation recorded by one still has its block taken out when a trial is replayed.
WARNINGS_HEADER = "[warnings] While this cell ran, Python emitted"
#: One line of that block, in the exact form ``_WarningLog.block`` writes it: ``  Name: text (places)``
#: with an optional ``xN`` -- the name a warning's category, or ``LEVEL:logger`` for a log message --
#: or the ``... and N more`` line, in its current wording or the one E-06 observations carry. Exact,
#: so an error message that quotes the header keeps its own indented lines (``  at cell line N: ...``)
#: and stays its own error. The REPL keeps ``(`` and ``)`` out of the places, anything but ``[\w.]``
#: out of a category or level and anything but ``[\w.-]`` out of a logger's name, so every line it
#: writes has this form.
_WARNING_LINE = (
    r"  (?:[A-Za-z_][\w.]*(?::[\w.-]+)?: [^\n]* \([^()\n]*\)(?: x\d+)?"
    r"|\.\.\. and \d+ more warning\(s\)(?: or log message\(s\))? of other kinds, not listed\.)(?=\n|\Z)"
)
#: The whole block: the header, where it starts a line, and the warning lines under it.
_WARNINGS_BLOCK_RE = re.compile(r"(?m)^" + re.escape(WARNINGS_HEADER) + r"[^\n]*(?:\n" + _WARNING_LINE + r")*")


def _execute_error_signature(content: str) -> str:
    """A stable signature for an execute-error observation: the text AROUND the error marker (not the
    raw tail -- R/bash errors put the message at the FRONT and trailing stdout at the end), with volatile
    volatile bits (traceback line numbers, object addresses) normalized out, so the SAME failing code
    re-pasted yields the SAME signature while a DIFFERENT error (a converging debug session) yields a
    different one -- which is the whole point of the guard and was NOT true until 2026-09-19."""
    # The REPL lists a cell's warnings after its error (support_tools._WarningLog.block). They are not
    # the error, and the default filter shows each location once, so the first failing run carries
    # them and an identical re-run does not: inside the window they gave one error two signatures and
    # started the identical-error streak a run late. The block is REMOVED, not cut at -- a timeout's
    # own message, or execute()'s import hint, follows it -- and the closing tag goes too, since the
    # block's newline otherwise sits where a bare error ends straight on the tag. Whitespace runs are
    # then one space: removing the block leaves a newline more or less before whatever followed it,
    # and no two errors differ by whitespace alone.
    content = _WARNINGS_BLOCK_RE.sub("", content).rstrip().removesuffix("</observation>")
    m = _EXEC_ERROR_RE.search(content)
    window = content[m.start() : m.start() + 400] if m else content[-400:]
    window = " ".join(window.split()).lower()
    # Volatile bits ONLY. This used to strip every digit and every quoted run, and that destroyed
    # exactly the content that tells two errors apart: `KeyError: 'leiden'`, `KeyError: 'spatial'`,
    # `KeyError: 'X_pca'` and `KeyError: 'cell_type'` all normalised to the single string
    # "error: traceback ... keyerror:" -- so a model CONVERGING on a fix (four different missing
    # keys, each one found and fixed in turn) read as one error repeated four times and the run
    # was ended on its fifth, already-written, corrected block.
    #
    # Measured: 4 distinct KeyErrors -> 1 signature; 4 distinct missing files -> 1 signature.
    # The docstring above promised the opposite and had promised it since it was written.
    window = re.sub(r"0x[0-9a-f]+", "", window)  # object addresses differ run to run
    window = re.sub(r"\bline \d+", "line", window)  # traceback line numbers move as code is edited
    return window.strip()[:200]


execute_error_signature = _execute_error_signature


def classify_and_clean_code(code: str | None) -> tuple[str, str]:
    """Classify an extracted block by its leading shebang-style marker and strip that marker.

    Returns ``(kind, cleaned)`` with kind in ``{"r", "bash", "cli", "python"}``. The marker is matched on
    the STRIPPED code, so a block that opens with a newline (the common ``<execute>\\n#!CLI ...``) still
    de-markers correctly — the old ``^``-anchored ``re.sub`` ran on un-stripped code, so a leading newline
    left ``#!CLI`` in place, it got folded into a ``#`` bash comment, and the command silently never ran."""
    stripped = (code or "").strip()
    # The comment-style markers are left in place, whole. ``# R script to run SPARK-X`` is a comment
    # in R and in bash, which is what the model meant by it; stripping only the marker turned the
    # rest of the line into the first statement -- bash ran ``to`` (exit 127) and R raised a parse
    # error on a line the model never wrote as code (hunt 2026-09-30, u12-react-5).
    if stripped.startswith(("# R code", "# R script")):
        return "r", stripped
    if stripped.startswith("# Bash script"):
        return "bash", stripped
    if stripped.startswith("#!R"):
        return "r", re.sub(r"^#!R", "", stripped, count=1).strip()
    if stripped.startswith("#!CLI"):
        # Lines kept as lines: the block runs as a bash script (`run_bash_script`), where a newline
        # ends a command. Joining them made `samtools index a.bam` + `samtools idxstats a.bam` one
        # command whose first tool got the second as arguments, and the second never ran
        # (u12-react-18). A command wrapped with `\` still continues, as in bash.
        return "cli", re.sub(r"^#!CLI", "", stripped, count=1).strip()
    if stripped.startswith("#!BASH"):
        return "bash", re.sub(r"^#!BASH", "", stripped, count=1).strip()
    # A shebang is kept, not stripped: unlike the invented "#!BASH" marker it is valid in the
    # language it names, and the script writer emits its own, where a second one is just a comment.
    if stripped.startswith(_SHEBANG_R):
        return "r", stripped
    if stripped.startswith(_SHEBANG_BASH):
        return "bash", stripped
    return "python", stripped
