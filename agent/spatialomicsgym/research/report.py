"""The document a research run hands back, and the sentence that says why it ended.

A run that analysed someone's data for an hour owes them more than a chat bubble. This module
renders ``research_report.html`` from the journal the loop persisted -- so it renders from a file,
which means it renders a run that died at round 3, a run copied off this box, and a run whose
portal has since been restarted.

A run that was *killed* persisted no journal, and that is the run whose report is worth the most.
``research.json`` is rewritten whole after each round, so it cannot describe the round it is
inside of: a real run here (``research_20260917_114717``) died in round 1 and left a ``round_1/``
directory and nothing else. So the journal is the preferred source and not the only one --
:func:`recovered_journal` rebuilds one from the append-only ledger and the per-round files the run
did write, and the document then says, in its own opening sentence, that the run did not finish.

It **composes** the post-analysis renderer rather than growing a second one. Every figure, table,
check and verdict inside a round is :func:`~spatialomicsgym.report.render.render_report_body` over
that round's own manifest -- byte for byte what ``/results`` serves, because it is handed the same
text filter the portal hands it (see :func:`_embedded_run`), and the same sections ``write_report``
leaves in the run directory. What this layer adds is only what post-analysis has no way to know:
the question that was asked, the narrative across rounds, the references that checked out, the
claims that did not, and why the loop stopped.

Two properties are deliberate:

* **It never raises.** It is called when a run has just finished, and a report that throws would
  turn a finished investigation into a failed one. A missing manifest, an unreadable round, a
  journal from a future version -- each degrades to a sentence in the document saying so.
* **A round that produced nothing still gets a section.** "Round 3 wrote no files" is a finding
  about the investigation. Rendering only the rounds that worked would leave a reader counting
  headings to notice.
* **Figures are files, not payload.** Given somewhere to put them (``figure_dir``), every figure is
  copied into ``figures/`` beside the document and linked from it, through the ``figure_src``
  callback ``render_report_body`` already takes for the portal. Measured on the last real run on
  this box, the embedded alternative was 4,192,024 characters of base64 in a 4,255,265-character
  file: 98.5% of a report that a mail client would refuse and a browser would think about. The
  cost is that the document is then not self-contained, which is why it says so itself.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import html
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spatialomicsgym import paths as _paths
from spatialomicsgym import report as _report
from spatialomicsgym.report.render import IMAGE_MIME as _IMAGE_MIME
from spatialomicsgym.research import ledger as _rounds
from spatialomicsgym.research import loop

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping

__all__ = [
    "FIGURES_DIR",
    "REPORT_NAME",
    "STOP_INTERRUPTED",
    "STOP_SENTENCES",
    "recovered_journal",
    "render_research_report",
    "metric_sentence",
    "stop_sentence",
    "write_research_report",
]

REPORT_NAME = "research_report.html"

#: Where a written report keeps its pictures: a folder beside the document, which is the layout
#: the post-analysis renderer already writes (``report.html`` beside ``figures/``).
FIGURES_DIR = "figures"

#: A run that was still going when the process ended. Deliberately **not** added to ``loop``'s stop
#: reasons: the loop can never report this one, because reaching any of its own reasons means it
#: persisted the journal on the way out. This is the *reader's* finding about a directory -- a
#: ledger whose last round never closed, or no journal at all -- and it needs a sentence for
#: exactly the reason the loop's reasons need one.
STOP_INTERRUPTED = "interrupted"

#: What to tell the user when a run stopped **without** writing an answer. Keyed by the loop's stop
#: reasons; ``{rounds}`` and ``{plural}`` are the only substitutions. Shared with the portal's SSE
#: stream (:mod:`sog_portal.research`) so the sentence a user reads while the run is
#: finishing is the same sentence the report opens with -- one vocabulary, not two spellings of it.
STOP_SENTENCES: dict[str, str] = {
    loop.STOP_BENCHMARKING: (
        "Research runs are switched off while this installation is in benchmarking mode, so nothing was run."
    ),
    loop.STOP_CANCELLED: (
        "You stopped this investigation after {rounds} round{plural}. Everything it had already "
        "finished is saved with the run."
    ),
    loop.STOP_TIME_CAP: (
        "The investigation ran out of its time budget after {rounds} round{plural}, before it wrote an answer."
    ),
    loop.STOP_ROUND_ERROR: "The investigation stopped after {rounds} round{plural} because a round could not finish.",
    loop.STOP_AGENT_BUSY: (
        "The investigation stopped after {rounds} round{plural} because another turn took the agent "
        "and did not give it back in time. Nothing is wrong with the run -- ask again when the other "
        "turn has finished, and the rounds below are still what it found."
    ),
    loop.STOP_UNUSABLE: (
        "The investigation stopped after {rounds} round{plural}: the data it was given could not be used."
    ),
    loop.STOP_STEPS_EXHAUSTED: (
        "The investigation stopped after {rounds} round{plural}: there was nothing further to follow up."
    ),
    loop.STOP_NO_NEW_ARTIFACT: (
        "The investigation stopped after {rounds} round{plural}: the last round produced no result to build on."
    ),
    loop.STOP_CONCLUSION_FAILED: (
        "The investigation ran {rounds} round{plural} and then could not write them up: the closing "
        "turn produced no answer. What each round found is below -- this is a failure of the last "
        "step, not of the work before it."
    ),
    loop.STOP_ROUND_EMPTY: (
        "The investigation stopped after {rounds} round{plural}: the round returned no answer, wrote no "
        "files and produced nothing to review, so there was nothing to build a next round on. This usually "
        "means the turn failed to start rather than that the question was answered."
    ),
    loop.STOP_ROUND_CAP: "The investigation used all {rounds} of its rounds without writing an answer.",
    loop.STOP_CONCLUDED: "The investigation finished but did not write an answer.",
    STOP_INTERRUPTED: (
        "This run did not finish. It was still going when the process ended, so it never wrote an "
        "answer or a wrap-up. What is below is what its {rounds} round{plural} left on disk before "
        "it stopped."
    ),
}

#: The same nine reasons for a run that **did** write one. A stop reason means something different
#: once there is an answer to read -- "stopped early" and "stopped early, and here is what it could
#: still say" send a reader to different next actions -- so the two are not one sentence with a
#: clause bolted on.
ANSWERED_SENTENCES: dict[str, str] = {
    loop.STOP_AGENT_BUSY: (
        "The investigation answered after {rounds} round{plural}, then stopped because another turn "
        "took the agent. There may be more to find if you ask again."
    ),
    loop.STOP_CANCELLED: (
        "You stopped this investigation after {rounds} round{plural}; the answer below is as far as it got."
    ),
    loop.STOP_TIME_CAP: (
        "The investigation reached its time budget after {rounds} round{plural} and concluded with what it had."
    ),
    loop.STOP_ROUND_ERROR: (
        "A round could not finish, so the investigation concluded early, after {rounds} round{plural}."
    ),
    loop.STOP_UNUSABLE: (
        "The investigation stopped after {rounds} round{plural} because the data could not be used. "
        "The answer below says what it could tell about it."
    ),
    loop.STOP_STEPS_EXHAUSTED: (
        "The investigation ran {rounds} round{plural} and concluded: there was nothing further to follow up."
    ),
    loop.STOP_NO_NEW_ARTIFACT: (
        "The last of {rounds} round{plural} produced no result to build on, so the investigation concluded there."
    ),
    loop.STOP_ROUND_CAP: "The investigation used all {rounds} of its rounds and concluded.",
    loop.STOP_CONCLUDED: "The investigation ran {rounds} round{plural} and concluded.",
}

_FALLBACK_SENTENCE = "The investigation stopped before it could write an answer."

#: How the stop reason is coloured. Absent means neutral: a run that used its whole budget did
#: nothing wrong, and painting it red would teach a reader to ignore the red ones.
_STOP_CLASS: dict[str, str] = {
    loop.STOP_CONCLUDED: "ok",
    loop.STOP_STEPS_EXHAUSTED: "ok",
    loop.STOP_ROUND_CAP: "partial",
    loop.STOP_TIME_CAP: "partial",
    loop.STOP_NO_NEW_ARTIFACT: "partial",
    loop.STOP_CANCELLED: "partial",
    loop.STOP_UNUSABLE: "unusable",
    loop.STOP_ROUND_ERROR: "failed",
    loop.STOP_BENCHMARKING: "failed",
    # Not "failed": a killed run is a run that was interrupted, and painting it the colour of a
    # round that crashed would tell a reader the analysis went wrong when the process did.
    STOP_INTERRUPTED: "partial",
}

#: Added to :data:`~spatialomicsgym.report.render.REPORT_CSS`, never instead of it. Four rules, and
#: each one obeys the sheet it joins: no bare ``width`` on a box holding generated words, no
#: ``nowrap`` without somewhere for the ink to stop, and no fixed grid track (see the report
#: stylesheet's own ``minmax(min(...,100%),1fr)``).
RESEARCH_CSS = """
.round{border-left:3px solid var(--line);padding-left:1rem;margin:1.1rem 0 0}
.round.err{border-left-color:var(--err)}
.said{margin:.15rem 0 .8rem;overflow-wrap:anywhere;white-space:pre-wrap}
.embed{margin:.7rem 0 0;border:1px solid var(--line);border-radius:10px;background:var(--card)}
.embed>summary{cursor:pointer;padding:.6rem .95rem;font-size:.9rem;color:var(--dim)}
.embed .wrap{padding:0 1.15rem 1.4rem;max-width:none}
.embed h1{font-size:1.15rem}
.cite .id{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.82em;overflow-wrap:anywhere}
"""

_ROUND_ANSWER_CHARS = 1200
_PROMPT_CHARS = 700
_CLAIM_CHARS = 260
_STAMP = "%Y-%m-%d %H:%M %Z"


# --------------------------------------------------------------------------- #
# small, total helpers -- none of these may raise on a journal from anywhere
# --------------------------------------------------------------------------- #
def _text(value: Any) -> str:
    """Model-authored prose, with credentials masked and absolute paths renamed, as a plain string.

    Path scrubbing belongs here, beside the credential masking, because they are the same concern
    arriving by the same route: prose the model wrote, rendered into a document a person opens.
    A real report shipped with seven absolute paths in it -- the round prompts carry
    ``--output /workspace/.../round1_scanpy_spatial_domain/scanpy_spatial_domains.h5ad`` because
    that is what the agent was told to write, and ``results_dir`` is stored raw on purpose so the
    journal stays usable for debugging. The journal keeps the truth; the *rendering* is where the
    machine's layout stops being anybody's business.

    ``paths.scrub`` and not a redaction: it rewrites a path under a known anchor to its anchored
    relative form, so ``--output outputs/round1_.../scanpy_spatial_domains.h5ad`` still says which
    file, while a path under no anchor of ours is left alone rather than mangled. Every text in
    this report reaches HTML through here, via ``_esc`` or ``_clip``, so one call covers the round
    prompts, the answers, the review reasons, the next steps and the failure lines together.
    """
    try:
        return _paths.scrub(_report.redact_text(value))
    except Exception:  # both promise never to raise; this is belt and braces
        return str(value or "")


def _esc(value: Any) -> str:
    return html.escape(_text(value), quote=True)


def _clip(value: Any, limit: int) -> str:
    body = " ".join(_text(value).split()) if limit <= _CLAIM_CHARS else _text(value).strip()
    return body if len(body) <= limit else body[: limit - 1].rstrip() + "…"


def _rows(journal: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    value = journal.get(key)
    return [dict(r) for r in value if isinstance(r, dict)] if isinstance(value, list) else []


def _plural(n: int) -> str:
    return "" if n == 1 else "s"


def _duration(seconds: Any) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return ""
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total // 3600}h {(total % 3600) // 60:02d}m"


def stop_sentence(reason: str, rounds_run: int, *, answered: bool = False) -> str:
    """Why the run ended, in a sentence a reader who did not watch it can act on."""
    table = ANSWERED_SENTENCES if answered else STOP_SENTENCES
    template = table.get(str(reason)) or STOP_SENTENCES.get(str(reason)) or _FALLBACK_SENTENCE
    count = max(0, int(rounds_run or 0))
    return template.format(rounds=count, plural=_plural(count))


# --------------------------------------------------------------------------- #
# figures: copied beside the document, not poured into it
# --------------------------------------------------------------------------- #
#: How long a staged file's name may be before the digest that makes it unique.
_STAGED_STEM = 48


def _staged_name(source: Path) -> str:
    """The file name one figure gets in :data:`FIGURES_DIR`: readable, unique, and URL-safe.

    The digest is of the *source path*, not of the bytes: two rounds that shared one results
    directory -- which happens, and has a callout of its own two functions down -- must resolve to
    one copy, while two different directories that both wrote ``spatial_domains.png`` must not
    collide. Restricting the readable part to ASCII letters, digits, dot, dash and underscore is
    what lets the link go into the document verbatim, with no percent-encoding to get wrong.
    """
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source.stem).strip("-.") or "figure"
    digest = hashlib.sha1(str(source).encode("utf-8", "replace"), usedforsecurity=False).hexdigest()[:8]
    return f"{stem[:_STAGED_STEM]}-{digest}{source.suffix.lower()}"


class _FigureStore:
    """Every figure the report shows, copied into one folder beside it, and the links to them.

    This is not a new mechanism. :func:`~spatialomicsgym.report.render.render_report_body` already
    takes a ``figure_src`` callback -- it is how ``/results`` shows a run without copying megabytes
    into the page -- and this is that same callback pointed at a folder instead of a route. The
    renderer keeps both of its existing contracts: a callback that answers ``None`` gets a sentence
    saying the figure cannot be shown here, and a figure missing from disk gets its own.

    ``staged`` is why the object exists rather than a bare closure: the document has to *say* that
    its pictures live next to it, and only a report that actually staged one may say so.
    """

    def __init__(self, directory: str | Path) -> None:
        self._root = Path(directory) / FIGURES_DIR
        #: Absolute source path -> the link already written for it. Both halves of "copy once".
        self._links: dict[str, str] = {}
        self.staged = 0

    def source_for(self, results_dir: Path) -> Callable[[str], str | None]:
        """The ``figure_src`` callback for one round, whose manifest paths are relative to it."""

        def link(rel: str) -> str | None:
            return self._stage(results_dir, rel)

        return link

    def _stage(self, results_dir: Path, rel: str) -> str | None:
        """Copy one figure next to the report and return its link, or ``None`` to decline it.

        ``safe_subpath`` again, even though the renderer has already resolved this path to decide
        the file exists: this call is the one that *reads* the bytes, and a containment check that
        the reader of a file does not itself make is a check some later refactor can walk past.

        Declining is the honest answer for a type the renderer would not link anyway (a manifest
        may declare ``figures/panel.pdf``) and for a file this process cannot read. Both reach the
        reader as the renderer's own sentence rather than as an ``<img>`` that can only break.
        """
        target = _report.safe_subpath(results_dir, rel)
        if target is None or target.suffix.lower() not in _IMAGE_MIME:
            return None
        key = str(target)
        if key in self._links:
            return self._links[key]
        name = _staged_name(target)
        try:
            payload = target.read_bytes()
            self._root.mkdir(parents=True, exist_ok=True)
            # Written the way every other file in a run directory is written -- ``.partial`` then
            # ``os.replace`` -- so a reader who opens the report while it is being written sees a
            # whole image or no image, never half of one.
            tmp = self._root / f".{name}.partial"
            tmp.write_bytes(payload)
            os.replace(tmp, self._root / name)
        except OSError:
            return None
        self._links[key] = f"{FIGURES_DIR}/{name}"
        self.staged += 1
        return self._links[key]


def _figures_note(store: _FigureStore | None) -> str:
    """Say that the pictures are files beside this document -- but only once one of them is.

    The whole of the standalone claim this report used to make is given up here, so it is given up
    out loud and in the place a reader looking at an empty frame will be. A run whose rounds
    produced no figure stages nothing, says nothing, and renders byte for byte what it rendered
    before this folder existed.
    """
    if store is None or not store.staged:
        return ""
    return (
        "<p class='quiet'>The pictures below are files in the <code>figures</code> folder beside "
        "this document, not pictures inside it -- a report that carried them would be several "
        "megabytes. Keep the two together: opened on its own, without that folder next to it, "
        "this report shows empty frames where the figures are.</p>"
    )


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def _answer_section(journal: Mapping[str, Any]) -> str:
    answer = _text(journal.get("answer"))
    if not answer.strip():
        # "The rounds it did finish" is the wrong half-sentence for a run that was killed inside
        # its only round: no round finished, and the reader is about to read one.
        said = (
            "The rounds it had started are below, with whatever each one reached."
            if journal.get("rebuilt_from_ledger")
            else "The rounds it did finish are below, with whatever each one produced."
        )
        return f"<h2>What it found</h2><div class='callout warn'><p>This run ended without writing an answer. {said}</p></div>"
    return f"<h2>What it found</h2><div class='card'><div class='said'>{html.escape(answer, quote=True)}</div></div>"


def _embedded_run(results_dir: str, number: int, figures: _FigureStore | None = None) -> str:
    """Round ``number``'s own post-analysis report, rendered by the renderer that owns it.

    Not re-implemented and not summarised: the same function ``/results`` serves and ``write_report``
    writes, over the same manifest, so a figure that appears on one appears identically on the
    other. A round with no readable manifest says so in one line -- which is itself a finding, since
    a round that wrote no manifest wrote nothing post-analysis could check.

    ``figures`` is the only thing this layer adds to that render: with a place to put them, the
    round's figures are copied there and linked; without one, the renderer embeds them as it always
    has. Passing ``figure_src`` is what ``/results`` does for the same reason, and passing nothing
    is what ``write_report`` does -- so this chooses between two behaviours the renderer already
    has rather than adding a third.
    """
    if not results_dir:
        return "<p class='quiet'>This round registered no results directory, so there is nothing to show for it.</p>"
    directory = Path(results_dir)
    try:
        manifest = _report.load(directory)
    except _report.ManifestError:
        return (
            "<p class='quiet'>This round wrote no post-analysis manifest, so its output could not be "
            "checked or rendered.</p>"
        )
    except Exception:
        return "<p class='quiet'>This round's results could not be read back.</p>"
    try:
        body = _report.render_report_body(
            manifest,
            directory,
            figure_src=figures.source_for(directory) if figures is not None else None,
            heading=f"Round {number}",
            # The embedding boundary's own filter, not the renderer's default: everything this
            # document prints reaches HTML through ``_text``, and a body composed into it must not
            # be the one exception. (Kept alongside ``figure_src`` -- both are arguments to this
            # one call and neither replaces the other.)
            text_filter=_text,
        )
    except Exception:
        return "<p class='quiet'>This round's results could not be rendered.</p>"
    return f"<details class='embed'><summary>Round {number}: figures, tables and checks</summary>{body}</details>"


def _artifact_span(record: Mapping[str, Any]) -> str:
    """The round's file count as the meta row shows it, or nothing when it never reported one.

    "0 new files" is a measurement, and a round that was still running when the run was killed made
    no measurement: its record is the one the loop wrote *before* the turn, which has no count in
    it. Printing a zero there tells the reader the round produced nothing, when what is true is
    that nobody ever asked it.
    """
    if "new_artifacts" not in record:
        return ""
    try:
        made = int(record.get("new_artifacts") or 0)
    except (TypeError, ValueError):
        made = 0
    return f"<span>{made} new file{_plural(made)}</span>"


def _expectation(record: Mapping[str, Any]) -> str:
    """What the round said it was looking for, written before it looked.

    ``loop.py`` preregisters every round -- the claim goes to ``round.json`` *before* the turn runs
    and is carried into the journal byte-for-byte -- and until this function nothing read it. A
    preregistration nobody reads is a diary: the whole value is that a reader can see what the
    round set out to find beside what it came back with, and judge the second against the first.

    Printed above "It was asked", because the order is the point. A reader who has already read the
    result cannot un-know it while forming a view of the expectation.

    Returns ``""`` for a journal that has no such block -- every run written before the field
    existed, and the conclusion turn, which preregisters nothing because it tests nothing.
    """
    block = record.get("hypothesis")
    if not isinstance(block, dict):  # `Mapping` is a typing-only import in this module
        return ""
    expects = _clip(block.get("expects"), _CLAIM_CHARS)
    if not expects:
        return ""
    because = _clip(block.get("because"), _CLAIM_CHARS)
    revised = block.get("revised")
    changed = isinstance(revised, dict) and bool(revised.get("plan_changed"))
    out = [
        "<div class='quiet'>Before it ran, it said it was looking for</div>",
        f"<div class='said'>{html.escape(expects, quote=True)}</div>",
    ]
    if because:
        out.append(f"<div class='quiet'>{html.escape(because, quote=True)}</div>")
    if changed:
        # A reviewer that proposes a different action from the one the previous round expected is
        # a run that changed its mind, which is legitimate and is also the thing a reader scanning
        # four rounds for a moved goalpost is looking for.
        out.append("<div class='quiet'>This is a change of plan from the previous round.</div>")
    return "".join(out)


def _round_section(record: Mapping[str, Any], figures: _FigureStore | None = None) -> str:
    number = record.get("round")
    number = int(number) if isinstance(number, int) else 0
    verdict = _text(record.get("verdict")) or "unknown"
    failure = _text(record.get("error"))
    # The round's own account of itself, which is not the reviewer's. A round that crashed has no
    # verdict -- the review ran over a directory the turn never finished writing -- so the badge
    # printed "unknown", which reads as "the reviewer could not decide" about a round that never
    # reached a reviewer. ``loop.ROUND_CRASHED`` is the journal's word for it, and reading it here
    # is what makes that field load-bearing: a journal that goes back to writing ``done`` for a
    # crashed round puts this badge back to lying.
    crashed = _text(record.get("status")).strip().lower() == loop.ROUND_CRASHED
    label = "did not finish" if crashed else verdict
    tone = "failed" if crashed else _report.badge_class(verdict, _report.VERDICTS)
    reasons = [r for r in (record.get("reasons") or []) if str(r).strip()]
    steps = [s for s in (record.get("next_steps") or []) if isinstance(s, dict)]

    parts = [f"<div class='round{' err' if failure else ''}'>"]
    parts.append(
        "<div class='meta'>"
        f"<span class='badge {tone}'>{_esc(label)}</span>"
        f"<span><b>Round {number}</b></span>"
        f"{_artifact_span(record)}"
        f"<span class='quiet'>{_esc(_duration(record.get('seconds')))}</span>"
        "</div>"
    )
    parts.append(_expectation(record))
    parts.append("<div class='quiet'>It was asked</div>")
    parts.append(f"<div class='said'>{html.escape(_clip(record.get('prompt'), _PROMPT_CHARS), quote=True)}</div>")
    answer = _clip(record.get("answer"), _ROUND_ANSWER_CHARS)
    if answer:
        parts.append("<div class='quiet'>It reported</div>")
        parts.append(f"<div class='said'>{html.escape(answer, quote=True)}</div>")
    if failure:
        parts.append(f"<div class='callout err'><p>This round did not finish: {_esc(failure)}</p></div>")
    shared = record.get("shared_results_with")
    if isinstance(shared, int) and shared:
        # Said before the figures rather than after them. A reader who has already looked at a
        # plot and formed a view is not going to un-form it from a footnote.
        parts.append(
            "<div class='callout'><p>This round wrote into the same results directory as round "
            f"{shared}, so the files below may include that round&rsquo;s. The count of new files "
            "is this round&rsquo;s own; the figures are whatever is in the directory.</p></div>"
        )
    if reasons:
        parts.append("<div class='quiet'>The review said</div><ul class='plain'>")
        parts.extend(f"<li>{_esc(r)}</li>" for r in reasons[:6])
        parts.append("</ul>")
    if steps:
        parts.append("<div class='quiet'>It planned next</div><ol class='steps'>")
        parts.extend(f"<li>{_esc(s.get('action') or s.get('why') or 'an unnamed step')}</li>" for s in steps[:5])
        parts.append("</ol>")
    parts.append(_embedded_run(str(record.get("results_dir") or ""), number, figures))
    parts.append("</div>")
    return "".join(parts)


def _href(url: Any) -> str:
    """A citation's link, or ``""`` when it must not become one.

    ``esc_url`` escapes but does not judge the scheme, and it is right not to: everywhere else in
    the report a URL is an identifier *this package built*, and its docstring says so. Here it is
    not. A citation's ``url`` is whatever the registry put in ``abs_url`` -- a third party's string,
    reached over the network, landing in an ``href`` inside a document the user opens locally from
    their own disk. ``javascript:`` is a scheme. So the two http schemes are allowed through and
    everything else is printed as text, which still shows the reader the identifier.
    """
    text = _text(url).strip()
    return _report.esc_url(text) if text[:8].lower().startswith(("http://", "https://")) else ""


def _citation_item(entry: Mapping[str, Any]) -> str:
    key = _text(entry.get("key"))
    kind = _text(entry.get("kind"))
    ident = f"arXiv:{key}" if kind == "arxiv" else f"https://doi.org/{key}"
    formatted = _text(entry.get("formatted"))
    bits = [b for b in (_text(entry.get("title")), _text(entry.get("venue")), _text(entry.get("year"))) if b]
    label = formatted or (", ".join(bits) if bits else ident)
    href = _href(entry.get("url")) or _href(ident)
    shown = html.escape(ident, quote=True)
    link = f" <a class='id' href='{href}'>{shown}</a>" if href else f" <span class='id'>{shown}</span>"
    out = [f"<li class='cite'>{html.escape(label, quote=True)}{link}"]
    note = _text(entry.get("note"))
    if note:
        out.append(f"<div class='quiet'>{html.escape(note, quote=True)}</div>")
    claim = _clip(entry.get("claim"), _CLAIM_CHARS)
    if claim:
        out.append(f"<div class='quiet'>Cited for: {html.escape(claim, quote=True)}</div>")
    out.append("</li>")
    return "".join(out)


def _citation_sections(journal: Mapping[str, Any]) -> str:
    entries = _rows(journal, "citations")
    verified = [c for c in entries if c.get("validated") is True]
    doubtful = [c for c in entries if c.get("validated") is not True]
    parts: list[str] = []
    if verified:
        parts.append("<h2>References</h2>")
        parts.append(
            "<p class='quiet'>Each of these was looked up in the registry that minted it and found. "
            "Titles and authors below are the registry's, not the model's.</p>"
        )
        parts.append("<ul class='plain'>" + "".join(_citation_item(c) for c in verified) + "</ul>")
    if doubtful:
        parts.append("<h2>Unverified claims</h2>")
        parts.append(
            "<div class='callout warn'><p>These identifiers were produced by the model and either "
            "could not be found or could not be checked. They are printed rather than dropped so "
            "that the claim they were offered as support for is visible too. Do not cite them "
            "without looking them up.</p></div>"
        )
        parts.append("<ul class='plain'>" + "".join(_citation_item(c) for c in doubtful) + "</ul>")
    if not entries:
        parts.append("<h2>References</h2>")
        # "Cited nothing" is a claim about the run, and a rebuilt journal cannot make it. The
        # citation ledger lives in memory until ``persist`` writes it, so a run that was killed
        # may well have checked a dozen papers and left no trace of any of them. Absence of the
        # record is not absence of the work, and the two must not print the same sentence.
        parts.append(
            "<p class='quiet'>What this run cited cannot be recovered: the citation record is "
            "written with the journal, and this run never wrote one.</p>"
            if journal.get("rebuilt_from_ledger")
            else "<p class='quiet'>This run cited no literature, so there is nothing to verify.</p>"
        )
    return "".join(parts)


def _grounding_section(journal: Mapping[str, Any]) -> str:
    missing = journal.get("missing_grounding")
    rows = [dict(m) for m in missing if isinstance(m, dict)] if isinstance(missing, list) else []
    if not rows:
        return ""
    items = []
    for row in rows:
        name = _esc(row.get("name") or row.get("tool") or "a literature tool")
        need = _text(row.get("requirement") or row.get("package") or "")
        fix = f" Install it with <code>pip install {html.escape(need, quote=True)}</code>." if need else ""
        items.append(f"<li>{name} was not available on this installation.{fix}</li>")
    return (
        "<h2>What was missing</h2>"
        "<div class='callout warn'><p>Some literature lookups could not run here, so this report is "
        "grounded in fewer sources than it would be on a fully installed box.</p></div>"
        "<ul class='plain'>" + "".join(items) + "</ul>"
    )


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
#: Each verdict in the reader's words, not the loop's. The same four words the live row uses --
#: `lib/researchPhase.ts`'s `VERDICT_WORD` -- because a reader who watched the run and then opens
#: the report must not meet two vocabularies for the same fact.
_VERDICT_WORD: dict[str, str] = {
    "progressing": "still improving",
    "plateau": "levelled off",
    "blocked": "nothing measurable",
    "undecided": "too early to call",
}


def metric_sentence(metric: Mapping[str, Any] | None) -> str:
    """One sentence about what the numbers did, or ``""`` when the run measured nothing.

    A separate clause from :func:`stop_sentence` on purpose. That one says why the loop stopped;
    this says whether it was still getting better, and a run that used one round of four was
    reported only by the first.

    Silence, not a zero, when nothing was measured. A metric is something a run has to DECLARE and
    no run before this feature declared one, so "0 to 0" would be the report inventing a
    measurement for every historical run it renders.
    """
    if not isinstance(metric, dict):
        return ""
    first = metric.get("metric_first")
    last = metric.get("metric_last")
    try:
        uncomputable = max(0, int(metric.get("uncomputable") or 0))
    except (TypeError, ValueError):
        uncomputable = 0
    word = _VERDICT_WORD.get(str(metric.get("verdict") or ""), "")
    if not isinstance(first, (int, float)) or not isinstance(last, (int, float)):
        if metric.get("verdict") == "blocked" and uncomputable:
            return (
                f"No round produced a number to compare -- {uncomputable} "
                f"round{_plural(uncomputable)} tried and could not."
            )
        return ""
    direction = str(metric.get("direction") or "")
    which = (
        " (higher is better)"
        if direction == "higher_is_better"
        else (" (lower is better)" if direction == "lower_is_better" else "")
    )
    if not which and metric.get("verdict") != "blocked":
        # With no declared direction there is no "better", so no trend word may claim one. The
        # ledger now says ``undecided`` here, but a journal written before that fix carries the
        # higher-is-better guess -- "still improving" for a loss that rose -- so the word is not
        # taken from it (hunt 2026-09-30, u19-pa-tasks-research-2).
        word = "no trend can be called: no direction was declared, so nothing here claims an improvement"
    gaps = (
        f" {uncomputable} round{_plural(uncomputable)} produced no number and "
        f"{'is' if uncomputable == 1 else 'are'} left out of that reading."
        if uncomputable
        else ""
    )
    moved = f"The metric went from {_number(first)} to {_number(last)}{which}"
    return f"{moved} -- {word}.{gaps}" if word else f"{moved}.{gaps}"


def _number(value: Any) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "?"
    if not n == n or n in (float("inf"), float("-inf")):  # NaN or infinity
        return "?"
    if abs(n) >= 1000 or (0 < abs(n) < 0.001):
        return f"{n:.3g}"
    return f"{round(n, 3):g}"


def _unfinished_sentence(journal: Mapping[str, Any]) -> str:
    """Which rounds began and never closed, from the ledger summary the journal carries.

    Read out of ``metric`` rather than from a key of its own because that dict *is* the ledger's
    summary -- ``loop.persist`` writes it after every round and :func:`recovered_journal` builds
    it the same way -- and a second copy of the same fact is a second thing that can disagree.

    A run that finished cleanly has nothing here and says nothing.
    """
    metric = journal.get("metric")
    rounds = [str(r).strip() for r in (metric or {}).get("unfinished", ()) if str(r).strip()] if metric else []
    if not rounds:
        return ""
    if len(rounds) == 1:
        return f"Round {rounds[0]} began and never finished -- the run ended inside it."
    listed = ", ".join(rounds[:-1]) + f" and {rounds[-1]}"
    return f"Rounds {listed} began and never finished."


def _rebuilt_sentence(journal: Mapping[str, Any]) -> str:
    """Where this document came from, when it did not come from the journal."""
    if not journal.get("rebuilt_from_ledger"):
        return ""
    return (
        "There is no research.json for this run. This report was rebuilt from the append-only "
        "ledger of rounds the run wrote as it went, plus each round's own file -- so it holds what "
        "was on disk when the run stopped, and nothing the loop was still keeping in memory."
    )


def render_research_report(
    journal: Mapping[str, Any],
    *,
    generated_at: str | None = None,
    figure_dir: str | Path | None = None,
) -> str:
    """A complete HTML document for one research run. Never raises.

    ``figure_dir`` is where the run's figures may be staged: given one, each figure is copied into
    ``<figure_dir>/figures/`` and linked, and the document says that it needs that folder. Given
    nothing -- which is what a caller rendering into a response or a string has -- every figure is
    embedded as base64 exactly as before, because there is nowhere to put a file.
    """
    data = dict(journal) if isinstance(journal, dict) else {}
    rounds = _rows(data, "rounds")
    # Rendered before the page is assembled, because staging the figures is what decides whether
    # the page may claim to have any beside it.
    store = _FigureStore(figure_dir) if figure_dir is not None else None
    sections = [_round_section(r, store) for r in rounds]
    try:
        rounds_run = int(data.get("rounds_run") or len(rounds))
    except (TypeError, ValueError):
        rounds_run = len(rounds)
    reason = str(data.get("stopped_because") or "")
    answered = bool(_text(data.get("answer")).strip())
    question = _text(data.get("question")).strip() or "An unnamed investigation"
    stamp = generated_at or _dt.datetime.now().astimezone().strftime(_STAMP)
    summary = data.get("citation_summary") if isinstance(data.get("citation_summary"), dict) else {}

    meta = [
        f"<span class='badge {_STOP_CLASS.get(reason, '')}'>{_esc(reason.replace('_', ' ') or 'unknown')}</span>",
        f"<span>{rounds_run} round{_plural(rounds_run)}</span>",
    ]
    elapsed = _duration(data.get("elapsed"))
    if elapsed:
        meta.append(f"<span>{_esc(elapsed)}</span>")
    dataset = _text(data.get("dataset")).strip()
    if dataset:
        meta.append(f"<span>on {_esc(dataset)}</span>")
    try:
        verified = int(summary.get("verified") or 0)
        total = int(summary.get("total") or 0)
    except (TypeError, ValueError):
        verified, total = 0, 0
    if total:
        meta.append(f"<span>{verified} of {total} citation{_plural(total)} verified</span>")
    meta.append(f"<span class='quiet'>generated {_esc(stamp)}</span>")

    body = [
        "<div class='eyebrow'>SpatialOmicsLab &middot; research report</div>",
        f"<h1>{html.escape(question, quote=True)}</h1>",
        "<div class='meta'>" + "".join(meta) + "</div>",
        f"<div class='callout'><p>{_esc(stop_sentence(reason, rounds_run, answered=answered))}</p>"
        + (f"<p>{_esc(metric_sentence(data.get('metric')))}</p>" if metric_sentence(data.get("metric")) else "")
        + (f"<p>{_esc(_unfinished_sentence(data))}</p>" if _unfinished_sentence(data) else "")
        + (f"<p>{_esc(_rebuilt_sentence(data))}</p>" if _rebuilt_sentence(data) else "")
        + "</div>",
        _answer_section(data),
    ]
    if rounds:
        body.append("<h2>How it got there</h2>")
        body.append(_figures_note(store))
        body.extend(sections)
    body.append(_citation_sections(data))
    body.append(_grounding_section(data))
    source = "the round ledger this run left on disk" if data.get("rebuilt_from_ledger") else None
    body.append(
        f"<div class='foot'>Generated by SpatialOmicsLab (ST-Coscientist) from {source}. "
        "Each round&rsquo;s figures and checks are the same post-analysis report the portal "
        "serves for that run.</div>"
        if source
        else "<div class='foot'>Generated by SpatialOmicsLab (ST-Coscientist) from this run&rsquo;s "
        "<code>research.json</code>. Each round&rsquo;s figures and checks are the same "
        "post-analysis report the portal serves for that run.</div>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(_clip(question, 90), quote=True)} &mdash; SpatialOmicsLab research report</title>"
        f"<style>{_report.REPORT_CSS}{RESEARCH_CSS}</style></head><body><div class='wrap'>"
        + "".join(body)
        + "</div></body></html>"
    )


# --------------------------------------------------------------------------- #
# reading a run directory: the journal if there is one, the ledger if there is not
# --------------------------------------------------------------------------- #
def _round_files(where: Path) -> dict[int, dict[str, Any]]:
    """Every ``round_<n>/round.json`` in the directory, by round number. Never raises.

    These files are the reason a rebuilt report is worth reading rather than merely present. Each
    one is written twice -- once before the turn, carrying the prompt and the preregistered claim,
    and once after it, carrying the answer, the review and the results directory -- and the keys it
    uses are the keys :func:`_round_section` already reads. A killed run leaves the first version
    of the round it died in and the second version of every round before it.
    """
    out: dict[int, dict[str, Any]] = {}
    try:
        found = sorted(where.glob("round_*/round.json"))
    except OSError:
        return out
    for path in found:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        try:
            number = int(record.get("round") or path.parent.name.split("_")[-1])
        except (TypeError, ValueError):
            continue
        out[number] = record
    return out


def _ledger_numbers(rows: list[dict[str, str]], event: str) -> set[int]:
    numbers: set[int] = set()
    for row in rows:
        if str(row.get("event", "")).strip() != event:
            continue
        try:
            numbers.add(int(str(row.get("round", "")).strip()))
        except (TypeError, ValueError):
            continue
    return numbers


def recovered_journal(directory: str | Path) -> dict[str, Any] | None:
    """A journal in the shape of ``research.json``, rebuilt from what a killed run left. Never raises.

    ``None`` when the directory holds no evidence a round ever started -- an empty directory is not
    an investigation, and a report saying "0 rounds" about one would be a file with nothing in it.

    What is here is only what is on disk: the rounds, their prompts and answers, and which of them
    never closed. What is deliberately **not** here is anything the loop held in memory -- the
    answer, the elapsed time, the dataset name, the citations. Those are absent rather than
    guessed, and the sections that would have printed them say so rather than printing a blank.
    """
    where = Path(directory)
    rows = _rounds.read_rounds(where)
    records = _round_files(where)
    began = _ledger_numbers(rows, "begin")
    ended = _ledger_numbers(rows, "end")
    numbers = sorted(began | set(records))
    if not numbers:
        return None

    rounds: list[dict[str, Any]] = []
    for number in numbers:
        record = dict(records.get(number) or {})
        record["round"] = number
        # A round is finished if the ledger closed it, or if its own file stopped saying it was
        # running. Either witness is enough; requiring both would call every round of a run with
        # no ledger unfinished, and requiring neither would miss the round the run died in.
        closed = number in ended or str(record.get("status") or "running") not in ("", "running")
        if not closed and not str(record.get("error") or "").strip():
            record["error"] = "the run ended while this round was still going, so it never reported"
        rounds.append(record)

    first = records.get(numbers[0]) or {}
    hypothesis = first.get("hypothesis") if isinstance(first.get("hypothesis"), dict) else {}
    return {
        "question": str(hypothesis.get("question") or ""),
        "rounds": rounds,
        "rounds_run": len(numbers),
        "stopped_because": STOP_INTERRUPTED,
        "answer": "",
        "citations": [],
        # The same summary a persisted journal carries, read from the same ledger by the same
        # function -- so ``unfinished`` and the metric sentence mean one thing in both documents.
        "metric": _rounds.summary(where),
        "rebuilt_from_ledger": True,
    }


def _journal_on_disk(where: Path) -> dict[str, Any] | None:
    """What to render for this directory: the journal, the journal corrected, or the ledger.

    Three cases, and the middle one is the one that is easy to miss. A run killed *after* round 2
    did persist a journal -- with ``stopped_because`` still empty, because only the loop's exit
    sets it -- so it renders, and renders as though it simply stopped for no stated reason. The
    ledger knows better: it holds a round that began and never closed. Reading the two together is
    what lets one sentence say the run was killed, whether or not the journal survived.
    """
    data: Any = None
    try:
        data = json.loads((where / "research.json").read_text(encoding="utf-8"))
    except Exception:
        data = None
    if not isinstance(data, dict):
        return recovered_journal(where)
    if str(data.get("stopped_because") or "").strip():
        return data
    summary = _rounds.summary(where)
    if not summary.get("unfinished"):
        return data
    return {**data, "stopped_because": STOP_INTERRUPTED, "metric": data.get("metric") or summary}


def write_research_report(directory: str | Path, *, journal: Mapping[str, Any] | None = None) -> Path | None:
    """Render ``research_report.html`` in the run's directory. Never raises.

    Returns the path written, or ``None`` when there was nothing to render or nowhere to write it:
    this is called as a run finishes, and a report that failed must cost the user a file, never the
    investigation. Written atomically, like every other file this system leaves in a run directory,
    so a reader following the portal's link never opens half a document.

    The source is ``research.json`` when there is one and :func:`recovered_journal` when there is
    not, which is what makes the killed run -- the one that motivated the ledger -- produce a
    document at all. Before this, a run that died inside round 1 left a ``round_1/`` directory, a
    ledger, and no report.

    The figures go in ``figures/`` beside the document rather than inside it: see
    :class:`_FigureStore`, and :func:`_figures_note` for what the document then owes the reader.
    """
    where = Path(directory)
    try:
        data = journal if journal is not None else _journal_on_disk(where)
        if not isinstance(data, dict):
            return None
        document = render_research_report(data, figure_dir=where)
        out = where / REPORT_NAME
        tmp = where / f".{REPORT_NAME}.partial"
        tmp.write_text(document, encoding="utf-8")
        os.replace(tmp, out)
    except Exception:
        return None
    return out
