"""The bounded outer loop of a research run.

A research run is several ordinary agent turns in a row, each one reviewed before the next is
composed, ending in a written conclusion whose citations have been checked. Almost none of that is
new machinery: the turn is whatever ``runner`` is, the review is
:func:`spatialomicsgym.postanalysis.next_step.review_and_act` called with no runner (which is
already a pure "check, write the verdict, propose the plan, act on nothing" call), the follow-on
instruction is that module's own ``build_followup_prompt``, and the citations are
:class:`spatialomicsgym.research.citations.CitationLedger`. What this module contributes is the arc
between them and, mostly, the reasons to stop.

**Nothing in :mod:`spatialomicsgym.postanalysis` is modified by this**, and in particular
``MAX_FOLLOWUP_ROUNDS_CEILING`` is untouched. The one interaction runs the other way: an agent turn
ends by running its own bounded follow-on rounds, so a research round would silently spawn a second
loop inside itself. :func:`quiet_followups` turns that acting off for each turn, from its ``start``
frame to its last (:func:`_quiet_turn`) -- the knob's own docstring says ``0`` "keeps the
self-review and turns the acting off" -- and restores it in a ``finally``. Per turn, not per run:
the knob is process-wide, and between rounds it belongs to whoever else is using the portal.

Headless on purpose. ``runner`` is any callable that takes a prompt and yields frames, so the whole
loop is testable with a fake runner and no server, no browser and no network; the web layer adapts
:func:`sog_portal.server.sse_events` to that shape and adds nothing to the control flow.

Frames yielded
--------------
``{"event": "research", "data": {...}}``
    This module's own. ``data["phase"]`` is ``planned`` (once, before any turn), ``round`` (before
    each turn), ``reviewed`` (after each turn's verdict is in) or ``done`` (once, last).
``{"event": <the runner's>, "data": {...}, "round": n, "last": bool}``
    A frame from the turn, passed through untouched and annotated with the round it belongs to and
    whether that round is the final one. The annotation is what lets a presentation layer decide
    that round 2's answer is an intermediate step while round 4's is *the* answer, without this
    module having to know what a presentation layer is.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from contextlib import closing as _closing
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spatialomicsgym.research import ledger as _rounds
from spatialomicsgym.research.citations import CitationLedger
from spatialomicsgym.research.prompts import (
    closing_prompt,
    continuing_prompt,
    grounding_tools,
    opening_prompt,
    unavailable_grounding,
)
from spatialomicsgym.stream_events import (
    BUSY_KINDS,
    ERROR_KIND_BUSY,
    EVENT_END,
    EVENT_ERROR,
    EVENT_FINAL,
    EVENT_RESEARCH,
    EVENT_START,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------
# bounds
# ----------------------------------------------------------------------------------------------

#: Analysis rounds a run gets before it must conclude. Four is enough for "look, follow up, check,
#: conclude" and small enough that an unattended run cannot become an afternoon.
MAX_ROUNDS_DEFAULT = 4

#: No configuration can exceed this. A research run is unattended work against someone's data, and
#: the ceiling is what makes "bounded" a property of the code rather than of the deployment.
MAX_ROUNDS_CEILING = 8

#: Wall-clock budget. Rounds are the primary bound; this one catches the round that does not end.
MAX_SECONDS_DEFAULT = 3600
MAX_SECONDS_CEILING = 14400

ROUNDS_ENV = "SOG_RESEARCH_MAX_ROUNDS"
SECONDS_ENV = "SOG_RESEARCH_MAX_SECONDS"

#: Why a run stopped. Every exit from :func:`run_research` sets exactly one of these, including the
#: ones that never ran a round -- "it did nothing" is an answer the user is owed a reason for.
STOP_CONCLUDED = "concluded"
STOP_ROUND_CAP = "round_cap"
STOP_TIME_CAP = "time_cap"
STOP_UNUSABLE = "verdict_unusable"
STOP_STEPS_EXHAUSTED = "steps_exhausted"
STOP_NO_NEW_ARTIFACT = "no_new_artifact"
STOP_ROUND_EMPTY = "round_empty"
STOP_CONCLUSION_FAILED = "conclusion_failed"
STOP_CANCELLED = "cancelled"
STOP_BENCHMARKING = "benchmarking"
STOP_ROUND_ERROR = "round_error"
STOP_AGENT_BUSY = "agent_busy"

#: What a round record says about itself, in ``round_N/round.json`` and in the journal's
#: ``rounds``. The vocabulary is the LEDGER's on purpose. ``rounds.tsv`` has written ``crash`` for
#: a failed round since it was added, while the journal record hardcoded ``done`` for the same
#: round two lines later -- so the two files this loop writes said different things about one
#: round, and a reader could believe either. Named here so that cannot be re-introduced by editing
#: one of the two call sites.
#:
#: ``running`` is written before the turn and replaced after it. It survives only where nothing
#: replaced it: a run killed inside a round leaves ``round.json`` saying ``running``, which is the
#: truth about that round and the reason the pre-write exists.
ROUND_RUNNING = "running"
ROUND_DONE = "done"
ROUND_CRASHED = "crash"

#: ``error.kind`` for "another turn holds the agent". Transient by construction: the server
#: emits it from the lock acquisition, BEFORE the turn starts, so a round refused this way did
#: no work and can simply be asked again.
#:
#: Imported rather than spelled again. This comparison is what decides whether a round WAITS or
#: is recorded as having failed, and the string it compares against is produced three files away.
#: A rename there would have made this loop stop recognising a busy answer and report the round
#: as an error instead -- silently, and in a different sentence in the report.
_BUSY = ERROR_KIND_BUSY
_BUSY_KINDS = BUSY_KINDS  # this conversation's own turn, or the process-wide agent lock

#: How many times a round waits out somebody else's turn before giving up, and for how long.
#:
#: A research run releases the global turn lock between rounds -- it must, or a 20-minute
#: investigation would freeze every other account -- so one chat message sent from another tab
#: can land in that gap. It used to end the whole investigation, and the portal said "Your
#: message was not sent" about a run that had already completed rounds and written a report.
#:
#: Bounded on purpose. The turn in the way might be a 25-minute deconvolution, and waiting it
#: out would strand the reader with no output at all; a short ladder covers the ordinary case
#: (a chat turn in another tab) and then stops with a reason that says what actually happened.
_BUSY_WAITS = (2.0, 5.0, 10.0)

#: Runner frames this loop reads rather than only forwarding. They are the agent-turn vocabulary,
#: not a transport detail: a turn ends by answering (``final``), by failing (``error``) or by
#: running out of stream (``end``).
#:
#: Imported rather than spelled, because these three were a *second definition* of a vocabulary
#: that also lives in the server, in ``sog_portal/research`` and in the client -- and a second
#: definition is how ``phase`` came to exist in some of those places and not others. The one
#: definition is :mod:`spatialomicsgym.stream_events`, which is stdlib-only and sits above
#: ``webui`` precisely so a headless loop can import it without pulling a web server in.
_FINAL = EVENT_FINAL
_ERROR = EVENT_ERROR
_END = EVENT_END


def _bounded(env: str, configured: str, default: int, ceiling: int) -> int:
    """``env``, else the config field, else ``default`` -- clamped into ``[1, ceiling]``.

    Written as a copy of ``next_step.max_followup_rounds`` because it must inherit that function's
    one non-obvious rule: **an unparseable value falls back to the default, not to zero.** A typo in
    a deployment's environment should not silently turn a feature off, and the shape that does that
    (``int(raw or 0)``) is the one a reviewer reads as obviously correct. The floor differs -- zero
    follow-on rounds is a meaningful setting, zero research rounds is a run that cannot answer, so
    this clamps to 1.
    """
    raw = os.environ.get(env, "").strip()
    if not raw:
        try:
            from spatialomicsgym.config import default_config

            raw = str(getattr(default_config, configured, default))
        except Exception:
            raw = str(default)
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return max(1, min(ceiling, value))


def max_rounds() -> int:
    """Analysis rounds this installation allows, clamped into ``[1, MAX_ROUNDS_CEILING]``."""
    return _bounded(ROUNDS_ENV, "research_max_rounds", MAX_ROUNDS_DEFAULT, MAX_ROUNDS_CEILING)


def max_seconds() -> int:
    """Wall-clock budget this installation allows, clamped into ``[1, MAX_SECONDS_CEILING]``."""
    return _bounded(SECONDS_ENV, "research_max_seconds", MAX_SECONDS_DEFAULT, MAX_SECONDS_CEILING)


def research_allowed() -> tuple[bool, str]:
    """``(allowed, reason)``. A research run refuses to start under benchmarking.

    Mirrors ``next_step.post_analysis_active``'s gate 1 and for the same reason: a scored run must
    not have extra unrequested agent turns appear inside it. Unlike that gate this one does not also
    consult ``post_analysis_enabled`` -- turning the automatic follow-ons off is a statement about
    what should happen *without* being asked, and a research run was asked for explicitly.
    """
    try:
        from spatialomicsgym.config import default_config
    except Exception:
        return True, ""
    if getattr(default_config, "benchmarking_enabled", False):
        return False, STOP_BENCHMARKING
    return True, ""


_QUIET_LOCK = threading.Lock()
_QUIET_DEPTH = 0
_QUIET_SAVED: list[Any] = []


@contextmanager
def quiet_followups() -> Iterator[None]:
    """Set ``post_analysis_max_followup_rounds`` to 0 for the block, restoring it in a ``finally``.

    Without this each research round ends by launching the ordinary post-analysis follow-on rounds,
    so a 4-round run is up to 4x3 unrequested extra turns nested inside it -- and the review this
    loop then reads would be of a directory several turns further on than the round it is judging.
    Zero is the knob's own documented "keep the checking, turn the acting off", which is exactly the
    half this loop wants: it does its own acting.

    The config object is process-wide, so the caller holds the turn lock across this block. Restored
    in a ``finally`` on every path including ``GeneratorExit``, because the alternative is a portal
    whose post-analysis stays switched off until it is restarted.

    Nesting is reference-counted: the first block in saves the real value and the last one out
    restores it. Each block used to save "whatever it found", so two overlapping research runs
    exited out of order -- A restored 3, then B restored the 0 it had found -- and post-analysis
    follow-on rounds stayed switched off for the rest of the process (hunt 2026-09-30,
    u02-server-b-3 / u06-history-5).
    """
    global _QUIET_DEPTH
    try:
        from spatialomicsgym.config import default_config
    except Exception:  # nothing to quieten and nothing to restore
        yield
        return
    with _QUIET_LOCK:
        if _QUIET_DEPTH == 0:
            _QUIET_SAVED[:] = [getattr(default_config, "post_analysis_max_followup_rounds", None)]
        _QUIET_DEPTH += 1
        default_config.post_analysis_max_followup_rounds = 0
    try:
        yield
    finally:
        with _QUIET_LOCK:
            _QUIET_DEPTH -= 1
            if _QUIET_DEPTH == 0:
                previous = _QUIET_SAVED[0] if _QUIET_SAVED else None
                if previous is None:
                    try:
                        del default_config.post_analysis_max_followup_rounds
                    except Exception:
                        pass
                else:
                    default_config.post_analysis_max_followup_rounds = previous


def _quiet_turn(frames: Iterable[Any]) -> Iterator[Any]:
    """``frames`` unchanged, with the follow-on acting off from the turn's ``start`` to its last frame.

    The window is per turn and opens on ``start`` for the reason ``server.one_round`` gives: the
    knob is process-wide, and ``start`` is emitted inside the turn lock, so it is the first instant
    at which no other account's turn can be in flight. ``run_research`` used to hold it at 0 for
    the WHOLE run instead -- including the gaps between rounds, when the lock is free and another
    account's chat turn runs, reads 0 and silently loses its own follow-on rounds for up to an hour
    (hunt 2026-09-30, u19-pa-tasks-research-5). A turn that never starts (the agent was busy) never
    opens it. Reference-counted, so the portal's own window nested inside this one is harmless.
    """
    with ExitStack() as quiet:
        opened = False
        for raw in frames:
            if not opened and isinstance(raw, Mapping) and str(raw.get("event", "")) == EVENT_START:
                quiet.enter_context(quiet_followups())
                opened = True
            yield raw


# ----------------------------------------------------------------------------------------------
# the run directory
# ----------------------------------------------------------------------------------------------


def research_directory(root: str | Path, *, run_id: str = "", now: float | None = None) -> Path:
    """``<root>/research_<ts>_<id>/``, created.

    Named to sort beside the agent's other output directories and to be recognisable in a listing;
    ``sog_portal.api.routers.results`` resolves this root already, so the run appears on ``/results`` with no new code.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now if now is not None else time.time()))
    token = "".join(ch for ch in str(run_id) if ch.isalnum() or ch in "-_")[:24] or "run"
    directory = Path(root) / f"research_{stamp}_{token}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` where a reader either sees the old file or the new one, never half of one.

    A research run is read *while it is running* -- the report page, the portal card, the next
    round's own resume check -- so a plain truncating write is a window in which the file is
    syntactically broken. Never raises: losing the journal must not lose the run.
    """
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    except Exception as exc:
        logger.debug("could not persist %s: %s: %s", path, type(exc).__name__, exc)
        try:
            temporary.unlink()
        except Exception:
            pass


# ----------------------------------------------------------------------------------------------
# reading a round
# ----------------------------------------------------------------------------------------------


def _artifact_count(manifest: Mapping[str, Any] | None) -> int:
    if not isinstance(manifest, Mapping):
        return 0
    return sum(len(manifest.get(kind) or ()) for kind in ("figures", "tables"))


def _review_round(roots: Sequence[Any], since: float, exclude: Sequence[Any] = ()) -> dict[str, Any]:
    """Make sure the round has a manifest, fold in what it wrote, and read the verdict.

    Every call here already exists and is made the same way the agent makes it. The ordering is the
    one ``review_and_act`` uses internally and for its stated reason: register first, so the
    re-check judges a directory it fully sees.

    Never raises. A round that produced a result nobody could describe is still a round that
    produced a result, and the loop's own stop conditions have to keep working when the description
    layer does not.
    """
    out: dict[str, Any] = {"results_dir": "", "verdict": "", "reasons": [], "next_steps": [], "new_artifacts": 0}
    try:
        from spatialomicsgym.postanalysis.autorun import analyse_new_outputs
        from spatialomicsgym.postanalysis.next_step import executable_steps, review_and_act
        from spatialomicsgym.postanalysis.review import (
            discover_results_dir,
            read_manifest,
            register_new_artifacts,
        )
    except Exception as exc:
        logger.debug("post-analysis unavailable: %s: %s", type(exc).__name__, exc)
        return out

    directory = None
    try:
        # ``exclude`` is every OTHER account's folder under the shared outputs tree. Without it a round
        # that waited behind another account's turn adopted that account's newest manifest as its own
        # results -- and wrote its verdict into it (hunt 2026-09-30, u02-server-b-1).
        directory = discover_results_dir(roots or (), since, exclude=exclude)
        if directory is None:
            written = analyse_new_outputs(roots or (), since, exclude=exclude)
            directory = written[-1] if written else None
    except Exception as exc:
        logger.debug("could not locate this round's results: %s: %s", type(exc).__name__, exc)
    if directory is None:
        return out

    out["results_dir"] = str(directory)
    try:
        out["new_artifacts"] = len(register_new_artifacts(directory))
    except Exception as exc:
        logger.debug("could not register round artifacts: %s: %s", type(exc).__name__, exc)
    try:
        # No `runner`: this is the pure "check, write the verdict, propose the plan, act on nothing"
        # call. The acting is this loop's job and doing it here would be the nested loop again.
        outcome = review_and_act(results_dir=directory) or {}
    except Exception as exc:
        logger.debug("could not review round: %s: %s", type(exc).__name__, exc)
        outcome = {}
    review = outcome.get("review") or {}
    out["verdict"] = str(review.get("verdict", "") or "")
    out["reasons"] = [str(r) for r in (review.get("reasons") or [])][:4]
    try:
        out["next_steps"] = executable_steps(outcome.get("next_steps") or [])
    except Exception:
        out["next_steps"] = []
    try:
        out["manifest_artifacts"] = _artifact_count(read_manifest(directory))
    except Exception:
        out["manifest_artifacts"] = 0
    return out


#: What a round is allowed to claim about itself, and when. Every value here is knowable before the
#: turn runs, because it is derived from what the *previous* round already returned.
_OPENING_REASON = "the opening round: the standing question is the only thing proposed so far"


def _preregistration(
    number: int,
    *,
    question: str,
    step: Mapping[str, Any] | None,
    carried: Mapping[str, Any] | None,
    prior: Mapping[str, Any] | None,
    now: float,
) -> dict[str, Any]:
    """The claim a round is about to test, recorded before the turn that tests it.

    Preregistration is nearly free here, because the loop already knows the answer: every round
    after the first exists because the previous round's review proposed one specific next step, and
    that proposal is in hand before the prompt is even composed. Writing it down is therefore not a
    new inference -- it is refusing to discard one the loop was about to throw away.

    The timing is the whole point, and it is the only part that cannot be added later. A round that
    records what it was looking for *after* seeing what it found is not recording what it was
    looking for; it is recording a reading of the result, and the two are indistinguishable once
    both are in the same file. So this is written to ``round.json`` before ``runner`` is called, and
    the post-round write carries it through unchanged rather than recomputing it.

    ``revised`` is the same discipline pointed backwards: what the previous round said it expected,
    what it actually returned, and whether the plan moved as a result. See :func:`_revision`.
    """
    action = str((step or {}).get("action", "") or "").strip()
    expects = action or (str(question).strip() if number == 1 else "")
    if number == 1:
        because = _OPENING_REASON
    else:
        verdict = str((carried or {}).get("verdict", "") or "unknown")
        reasons = [str(r) for r in (carried or {}).get("reasons", ()) if str(r).strip()]
        because = f"round {number - 1} was reviewed {verdict}" + (f": {reasons[0]}" if reasons else "")
    out: dict[str, Any] = {
        "round": number,
        "question": str(question),
        "expects": expects,
        "because": because,
        # Not a claim about this record's own content -- a claim about when it reached the disk. The
        # test that earns it reads the file from inside the turn, while the answer does not exist.
        "written_before_the_turn": True,
        "written_at": round(float(now), 3),
    }
    revised = _revision(number, carried=carried, prior=prior, expects=expects)
    if revised is not None:
        out["revised"] = revised
    return out


def _revision(
    number: int,
    *,
    carried: Mapping[str, Any] | None,
    prior: Mapping[str, Any] | None,
    expects: str,
) -> dict[str, Any] | None:
    """How the previous round's expectation fared, or ``None`` when there is no previous round.

    Every field here varies. That is a real constraint, not a style note: the loop's own stop
    conditions mean a round only exists at all if the one before it was not judged ``unusable`` and
    left either an artifact or a results directory behind. So a ``held`` or ``produced_something``
    flag computed from those facts would be ``True`` in every file this loop will ever write -- a
    field that looks like a check, reads like a check and can never fail one. Left out deliberately.

    What is left does move. ``verdict`` separates ``ok`` from ``suspicious`` (both of which continue
    the run), ``reasons`` says what the reviewer objected to, and ``plan_changed`` compares the two
    rounds' stated expectations -- so a reviewer proposing the same action twice reads as a loop
    that is not advancing, which is something a reader can act on and something a transcript of
    prompts alone does not show.

    And nothing here says the expectation was *correct*. No reviewer in this loop reports that, and
    a field implying it would be the loop marking its own work.
    """
    if number <= 1 or not prior:
        return None
    seen = dict(carried or {})
    predicted = str(prior.get("expects", "") or "")
    return {
        "after_round": number - 1,
        "expected": predicted,
        "verdict": str(seen.get("verdict", "") or "") or "unknown",
        "reasons": [str(r) for r in seen.get("reasons", ())][:3],
        "plan_changed": bool(expects) and expects != predicted,
    }


def _round_outcome(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """``raw`` with every key this loop reads present and of the type it reads it as.

    :func:`_review_round` already promises that shape; this is about the injected reviewer, which is
    someone else's function. The loop's stop conditions are the whole safety story, and a missing
    key turning into a ``KeyError`` mid-round would take them out.
    """
    out = dict(raw or {})
    out["results_dir"] = str(out.get("results_dir") or "")
    out["verdict"] = str(out.get("verdict") or "")
    out["reasons"] = [str(r) for r in (out.get("reasons") or ())][:4]
    out["next_steps"] = [s for s in (out.get("next_steps") or ()) if isinstance(s, Mapping)]
    try:
        out["new_artifacts"] = max(0, int(out.get("new_artifacts") or 0))
    except (TypeError, ValueError):
        out["new_artifacts"] = 0
    return out


#: What the prompt asks a multi-round run to declare, and the only form this loop reads.
#:
#: Strict on purpose. A number scraped out of prose is a number nobody agreed on, and the ledger
#: already has an honest answer for "there is no number this round": ``unknown``. So the model is
#: asked for exactly one line and exactly this shape, and anything else is not a metric.
_METRIC_RE = re.compile(
    r"^[ \t]*METRIC[ \t]*:[ \t]*(?P<name>[^=\n]{1,60}?)[ \t]*=[ \t]*"
    r"(?P<value>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)[ \t]*$",
    re.MULTILINE,
)
_DIRECTION_RE = re.compile(r"^[ \t]*DIRECTION[ \t]*:[ \t]*(higher_is_better|lower_is_better)[ \t]*$", re.MULTILINE)


def _declared_metric(text: str) -> tuple[str, float] | None:
    """``(name, value)`` from the round's own ``METRIC: <name> = <number>`` line, or ``None``.

    The LAST match wins. A round that restates its metric after correcting itself means the
    correction, and taking the first would score the run on the value the model withdrew.
    """
    matches = list(_METRIC_RE.finditer(str(text or "")))
    if not matches:
        return None
    last = matches[-1]
    try:
        return last.group("name").strip()[:60], float(last.group("value"))
    except (TypeError, ValueError):
        return None


def _declared_direction(text: str) -> str:
    """``higher_is_better`` / ``lower_is_better`` from the round's own line, else ``""``.

    First match wins here, unlike the metric: the direction is a property of the RUN, declared
    once in round 1, and a later line that disagrees is the flip the ledger refuses anyway.
    """
    found = _DIRECTION_RE.search(str(text or ""))
    return found.group(1).lower() if found else ""


def _progress_so_far(records: Sequence[Mapping[str, Any]], *, limit: int = 6) -> str:
    """One line per finished round: what it found, and whether it found anything.

    A round that is not told what the previous rounds did will re-try the setting that was already
    discarded -- and this loop has been chaining each round off the last one's follow-up text with
    no memory of the run as a whole. The follow-up prompt describes ONE result; this describes the
    run, which is what makes round 4 an improvement on round 1 rather than a fourth analysis.

    ``verdict`` is the reviewer's word for the round; ``new_artifacts`` of zero is a round that
    produced nothing, and saying so is what stops the next round proposing the same thing again.
    Bounded, because this goes into a prompt and a long run must not push the actual question out
    of it; the most recent rounds are the ones that matter, so the tail is what survives.

    Never raises and never guesses: a record missing a field contributes what it has.
    """
    lines: list[str] = []
    # Filter first, THEN take the tail. Slicing first counts unusable entries against the budget,
    # so a run whose last few records are malformed hands the next round an empty ledger and it
    # starts over blind -- the exact failure this function exists to prevent.
    usable = [r for r in records if isinstance(r, Mapping)]
    for record in usable[-max(1, int(limit)) :]:
        number = record.get("round", "?")
        verdict = str(record.get("verdict") or "").strip() or "no verdict"
        made = record.get("new_artifacts")
        try:
            made = max(0, int(made))
        except (TypeError, ValueError):
            made = 0
        produced = f"{made} new file{'' if made == 1 else 's'}" if made else "nothing new"
        reasons = [str(r) for r in (record.get("reasons") or ())][:1]
        why = f" -- {reasons[0]}" if reasons else ""
        error = str(record.get("error") or "").strip()
        if error:
            lines.append(f"  round {number}: did not complete ({error[:120]})")
            continue
        lines.append(f"  round {number}: {verdict}, {produced}{why}")
    return "\n".join(lines)


def _followup_text(results_dir: str, review: Mapping[str, Any], step: Mapping[str, Any]) -> str:
    """``build_followup_prompt``'s text for ``step``, or ``""`` if it cannot be built.

    The manifest is re-read here rather than carried: it is what the function wants, it changed
    during the round, and a stale copy is how a prompt comes to describe the previous round.
    """
    try:
        from spatialomicsgym.postanalysis.next_step import build_followup_prompt
        from spatialomicsgym.postanalysis.review import read_manifest

        manifest = read_manifest(Path(results_dir))
        if manifest is None:
            return ""
        return build_followup_prompt(manifest, dict(review), dict(step), results_dir)
    except Exception as exc:
        logger.debug("could not build the follow-on prompt: %s: %s", type(exc).__name__, exc)
        return ""


# ----------------------------------------------------------------------------------------------
# the loop
# ----------------------------------------------------------------------------------------------


def run_research(
    question: str,
    *,
    runner: Callable[[str], Iterable[Mapping[str, Any]]],
    output_root: str | Path,
    roots: Sequence[Any] = (),
    exclude: Callable[[], Sequence[Any]] | Sequence[Any] = (),
    dataset: str = "",
    run_id: str = "",
    cancel: Callable[[], bool] | None = None,
    rounds: int | None = None,
    seconds: float | None = None,
    ledger: CitationLedger | None = None,
    clock: Callable[[], float] = time.time,
    review: Callable[[Sequence[Any], float], Mapping[str, Any]] | None = None,
    followup_for: Callable[[str, Mapping[str, Any], Mapping[str, Any]], str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Run ``question`` as a bounded investigation, yielding frames as it goes.

    ``runner`` takes one prompt and yields that turn's frames. ``cancel`` is polled at every round
    boundary, so Stop takes effect within one round rather than within one run.

    Persists ``research.json`` after every round, so a run that dies at round 3 still renders 1-2.

    ``review`` and ``followup_for`` default to the post-analysis layer and exist as arguments for
    the same reason the citation lookups do: the interesting behaviour here is *which round runs
    next and why the loop stopped*, and a test that has to produce a real manifest to ask that
    question ends up testing the manifest writer. Production passes neither.
    """

    def _current_exclude() -> Sequence[Any]:
        # A callable is re-read per round: an account created mid-run has a folder the first round
        # did not know about.
        try:
            return tuple(exclude() if callable(exclude) else exclude)
        except Exception:
            return ()

    if review is None:

        def read_round(roots: Sequence[Any], since: float) -> dict[str, Any]:
            return _review_round(roots, since, exclude=_current_exclude())

    else:
        read_round = review
    next_prompt = _followup_text if followup_for is None else followup_for
    allowed, why = research_allowed()
    budget_rounds = max(1, min(MAX_ROUNDS_CEILING, int(rounds))) if rounds is not None else max_rounds()
    budget_seconds = max(1, min(MAX_SECONDS_CEILING, int(seconds))) if seconds is not None else max_seconds()
    started = clock()

    directory = research_directory(output_root, run_id=run_id, now=started)
    journal_path = directory / "research.json"
    missing = unavailable_grounding()
    journal: dict[str, Any] = {
        "question": str(question),
        "dataset": str(dataset),
        "run_id": str(run_id),
        "started": started,
        "max_rounds": budget_rounds,
        "max_seconds": budget_seconds,
        "grounding": [name for name, _ in grounding_tools()],
        "missing_grounding": missing,
        "rounds": [],
        "citations": [],
        "citation_summary": {"total": 0, "verified": 0, "refuted": 0, "unchecked": 0},
        "stopped_because": "",
        "answer": "",
    }

    def persist(stopped: str = "") -> None:
        journal["stopped_because"] = stopped or journal["stopped_because"]
        journal["elapsed"] = round(clock() - started, 3)
        journal["citations"] = [asdict(c) for c in book.entries.values()]
        journal["citation_summary"] = book.summary()
        # What the NUMBERS did, beside the reason the loop gives for stopping. The two answer
        # different questions -- "why did it stop" and "was it still getting better" -- and a run
        # that used one round of four was reported only by the first. Read from the ledger rather
        # than accumulated here: the ledger is the append-only record and a second running total
        # would be a second thing that can disagree with it.
        journal["metric"] = _rounds.summary(directory)
        _write_json(journal_path, journal)

    book = ledger if ledger is not None else CitationLedger()

    if not allowed:
        persist(why)
        yield _frame("done", directory=directory, rounds_run=0, stopped_because=why, citations=book.summary())
        return

    # Opened before the first round, not after it. `research.json` is rewritten whole after each
    # round, which cannot say anything about the round it is inside of -- and a real run here
    # (`research_20260917_114717`) died inside round 1 and left a `round_1/` directory and no
    # journal at all, against `run_research`'s own stated guarantee.
    _rounds.open_ledger(directory)

    yield _frame(
        "planned",
        directory=directory,
        question=str(question),
        dataset=str(dataset),
        max_rounds=budget_rounds,
        max_seconds=budget_seconds,
        grounding=journal["grounding"],
        missing=missing,
    )

    stopped = STOP_ROUND_CAP
    rounds_run = 0
    answer = ""
    followup = ""
    review_seen: dict[str, Any] = {}
    # Three things the next round needs that only this one can supply: the step that will justify
    # it, the outcome that step came out of, and this round's own stated expectation to be measured
    # against. Carried rather than re-read, because re-reading them after the next turn is exactly
    # the retrofit the preregistration exists to prevent.
    pending_step: Mapping[str, Any] | None = None
    carried: dict[str, Any] = {}
    prior: dict[str, Any] = {}
    # Which round first wrote into each results directory.
    #
    # In a real run here (`research_20260917_144841`) rounds 1 and 2 both recorded
    # `round1_scanpy_spatial_domain/post_analysis`: the agent reused the path, so round 2's
    # artifacts landed in round 1's directory and the report drew round 1's figures under round
    # 2's heading. The loop cannot stop the agent choosing a directory, but it can stop the
    # report from silently mis-attributing what it finds there.
    first_writer: dict[str, int] = {}

    for number in range(1, budget_rounds + 1):
        if cancel is not None and cancel():
            stopped = STOP_CANCELLED
            break
        if clock() - started >= budget_seconds:
            stopped = STOP_TIME_CAP
            break

        if number == 1:
            prompt = opening_prompt(question, dataset=dataset, rounds=budget_rounds)
        elif followup:
            prompt = continuing_prompt(
                followup,
                round_number=number,
                rounds=budget_rounds,
                progress=_progress_so_far(journal["rounds"]),
            )
        else:
            # No follow-on text means the previous round proposed nothing executable. That is a
            # finished investigation, not a round to improvise, and improvising is how an
            # unattended loop wanders off the question it was given.
            stopped = STOP_STEPS_EXHAUSTED
            break

        round_started = clock()
        round_dir = directory / f"round_{number}"
        round_dir.mkdir(parents=True, exist_ok=True)
        prereg = _preregistration(
            number,
            question=str(question),
            step=pending_step,
            carried=carried,
            prior=prior,
            now=round_started,
        )
        # On disk before `runner` is called. Everything else about this record is provisional
        # -- the answer is empty, the verdict unknown -- and saying so in `status` is cheaper
        # than a reader having to infer it from the absent keys. The post-round write replaces
        # the file and carries `hypothesis` through byte-for-byte.
        _write_json(
            round_dir / "round.json",
            {"round": number, "prompt": prompt, "status": ROUND_RUNNING, "hypothesis": prereg},
        )
        # The append that survives a kill. `round.json` above is a whole-file write inside the
        # round's own directory; this one line is in the run's ledger and says a round began,
        # so a run stopped mid-round is legible instead of absent.
        _rounds.begin_round(directory, number, change=str(prereg.get("claim") or "")[:200], now=round_started)
        yield _frame(
            "round",
            round=number,
            of=budget_rounds,
            directory=round_dir,
            expects=prereg["expects"],
            because=prereg["because"],
        )

        text, failure, busy = "", "", False
        for wait in (*_BUSY_WAITS, None):
            text, failure, busy = "", "", False
            try:
                with _closing(_quiet_turn(runner(prompt))) as frames:
                    for raw in frames:
                        event = str(raw.get("event", "")) if isinstance(raw, Mapping) else ""
                        data = raw.get("data") if isinstance(raw, Mapping) else None
                        payload = data if isinstance(data, Mapping) else {}
                        if event == _FINAL:
                            text = str(payload.get("text", "") or "")
                        elif event == _ERROR:
                            if str(payload.get("kind") or "") in _BUSY_KINDS:
                                # Not this round's failure -- somebody else's turn. Swallowed
                                # rather than forwarded, because a reader shown an error frame
                                # for a round that is about to be retried has been told the run
                                # broke when it did not.
                                busy = True
                                continue
                            failure = str(payload.get("message", "") or "the round failed")
                        if event != _END:  # one `end` per run, emitted by whoever presents this
                            yield {"event": event, "data": dict(payload), "round": number, "last": False}
            except Exception as exc:  # a failing round ends the run; it does not crash the caller
                logger.debug("research round %s raised: %s: %s", number, type(exc).__name__, exc)
                failure = f"{type(exc).__name__}: {exc}"
            if not busy or text or failure or wait is None:
                break
            if cancel is not None and cancel():
                break
            logger.debug("research round %s waited %.0fs for the agent", number, wait)
            time.sleep(wait)

        rounds_run = number
        if text:
            answer = text
            book.add(text)
            book.verify_all()

        outcome = _round_outcome(read_round(roots, round_started))
        review_seen = {"verdict": outcome["verdict"], "reasons": outcome["reasons"]}
        record = {
            "round": number,
            "prompt": prompt,
            # Two words, not one. This record used to say ``done`` unconditionally, including for
            # the round whose ``error`` two keys below says it did not finish -- and the ledger
            # write further down has always said ``crash`` for exactly that round. One of the
            # two was lying to every reader of either file.
            #
            # Read by :func:`spatialomicsgym.research.report._round_section`, which badges a
            # crashed round "did not finish" instead of printing the reviewer's ``unknown``
            # over a round that never reached a reviewer. That reader is what makes this field
            # load-bearing rather than decorative: put ``done`` back and the report lies too.
            "status": ROUND_CRASHED if failure else ROUND_DONE,
            # The claim this round preregistered, carried through from the pre-turn write
            # byte-for-byte. Read by the report, which prints what the round said it was
            # looking for above what it found -- a preregistration nobody reads is a diary.
            "hypothesis": prereg,
            "answer": text,
            "error": failure,
            "results_dir": outcome["results_dir"],
            "verdict": outcome["verdict"],
            "reasons": outcome["reasons"],
            "new_artifacts": outcome["new_artifacts"],
            "next_steps": outcome["next_steps"],
            "seconds": round(clock() - round_started, 3),
        }
        if outcome["results_dir"]:
            shared_with = first_writer.setdefault(outcome["results_dir"], number)
            if shared_with != number:
                # Recorded, not corrected. The files are where they are; what this stops is
                # the report presenting another round's figures as this one's work.
                record["shared_results_with"] = shared_with
        journal["rounds"].append(record)
        # No metric yet -- that is the next item in the plan's Stage R order -- so the round
        # closes as `unknown` rather than 0. A zero is a measurement and this is the absence
        # of one, and `ledger.verdict` is only correct because it keeps them apart.
        measured = _declared_metric(text)
        if number == 1:
            # Declared once, by the round the prompt asks for it from. The ledger refuses a
            # second declaration, so a later round cannot flip which way is better and have
            # every round after it read as an improvement.
            _rounds.declare_direction(directory, _declared_direction(text))
        _rounds.close_round(
            directory,
            number,
            metric=measured[1] if measured else None,
            guard=("ok" if outcome["new_artifacts"] else "no new output"),
            # The same two words the record above uses, plus the reviewer's verdict, which
            # the ledger has no other column for. A row that says ``crash`` and a record that
            # says ``done`` about one round is the disagreement these names exist to prevent.
            status=(ROUND_CRASHED if failure else (outcome["verdict"] or ROUND_DONE)),
            # The basename, never the path. `_public` basenames `dir` for this reason and a
            # second file with the same content must not undo it.
            results_dir=Path(outcome["results_dir"]).name if outcome["results_dir"] else "",
            note=(
                failure
                or (f"{measured[0]} = {measured[1]:g}" if measured else "")
                or (outcome["reasons"][0] if outcome["reasons"] else "")
            ),
        )
        _write_json(round_dir / "round.json", record)
        persist()

        yield _frame(
            "reviewed",
            round=number,
            verdict=outcome["verdict"] or "unknown",
            reasons=outcome["reasons"],
            new_artifacts=outcome["new_artifacts"],
            next_steps=[str(s.get("action", "")) for s in outcome["next_steps"]][:3],
            citations=book.summary(),
        )

        if busy and not text and not failure:
            # Waited out the ladder and the agent is still somebody else's. Its own reason, so
            # the sentence can say that rather than blaming the round or the reader's message.
            stopped = STOP_AGENT_BUSY
            break
        if failure:
            stopped = STOP_ROUND_ERROR
            break
        if outcome["verdict"] == "unusable":
            # The verdict's own branch of `propose_next_steps` says "report this and stop; do
            # not build further analysis on it". Building three more rounds on it is precisely
            # what an unattended loop would otherwise do.
            stopped = STOP_UNUSABLE
            break
        if not text and not outcome["results_dir"] and not outcome["verdict"]:
            # The round produced nothing whatsoever -- no answer, no directory, no review.
            # Checked before the two below because neither of their sentences is true here:
            # "there was nothing further to follow up" says the plan completed, and a real run
            # on this box reported exactly that after 1 of 4 rounds and 5.9 seconds. A round
            # that did not run is not a plan that finished.
            stopped = STOP_ROUND_EMPTY
            break
        if not outcome["next_steps"]:
            # Checked before the artifact test on purpose: when both are true, "there is nothing
            # left to do" is the more informative of the two reasons, and this is also the
            # answer an injected or degraded reviewer returning nothing should get. That holds
            # for a round that PRODUCED something; the barren case is caught above.
            stopped = STOP_STEPS_EXHAUSTED
            break
        if not outcome["new_artifacts"] and not outcome["results_dir"]:
            # Nothing was produced and nothing was even described. Another round would be told
            # to follow up on a result that does not exist.
            stopped = STOP_NO_NEW_ARTIFACT
            break

        pending_step = outcome["next_steps"][0]
        carried = {
            "verdict": outcome["verdict"],
            "reasons": outcome["reasons"],
            "new_artifacts": outcome["new_artifacts"],
            "results_dir": outcome["results_dir"],
        }
        prior = prereg
        followup = next_prompt(outcome["results_dir"], review_seen, outcome["next_steps"][0])

    # The conclusion is its own turn, and it runs on every path that produced anything at all --
    # including the ones that stopped early. A run that hit the round cap and a run whose data
    # turned out to be unusable both owe the user a written answer; "the loop ended" is not one.
    #
    # Two consequences worth stating rather than discovering. It runs *past* the wall-clock
    # budget: that budget bounds analysis rounds, and the alternative is a run that spent an
    # hour on someone's data and never said what it found. It does not run after Stop, because
    # Stop means stop -- the rounds that did finish are in ``research.json`` either way.
    # No conclusion after a cancel, and none after the agent was taken: the closing turn
    # goes through the same lock that just refused four times, so it would spend the
    # whole retry ladder again and still write nothing. The rounds that did run are in
    # the journal either way.
    if rounds_run and stopped not in (STOP_CANCELLED, STOP_AGENT_BUSY) and not (cancel is not None and cancel()):
        closing = closing_prompt(question, rounds_run=rounds_run)
        yield _frame("round", round=rounds_run + 1, of=rounds_run + 1, kind="conclusion")
        # What the CLOSING turn produced, kept apart from `answer`. They used to be the same
        # variable, via `answer = text or answer`, so a conclusion that raised left the last
        # analysis round's text standing and the run reported `concluded` over it -- handing
        # the reader, as the final answer, the one thing `sog_portal/research.py` refuses to show:
        # "An intermediate round's `final` is dropped, not shown."
        concluded = ""
        broke = ""
        busy = False
        # The same busy ladder the analysis rounds use. The closing turn forwarded another turn's
        # `busy` refusal as this run's last frame -- which the page renders as "This chat already
        # has a turn running. Your message was not sent" over a run whose rounds had all run --
        # and recorded `conclusion_failed` for a conclusion that never started (u06-history-8).
        for wait in (*_BUSY_WAITS, None):
            concluded, broke, busy = "", "", False
            try:
                with _closing(_quiet_turn(runner(closing))) as frames:
                    for raw in frames:
                        event = str(raw.get("event", "")) if isinstance(raw, Mapping) else ""
                        data = raw.get("data") if isinstance(raw, Mapping) else None
                        payload = data if isinstance(data, Mapping) else {}
                        if event == _FINAL:
                            concluded = str(payload.get("text", "") or "") or concluded
                        elif event == _ERROR:
                            if str(payload.get("kind") or "") in _BUSY_KINDS:
                                busy = True
                                continue
                            broke = str(payload.get("message", "") or "the closing turn failed")
                        if event != _END:
                            yield {"event": event, "data": dict(payload), "round": rounds_run + 1, "last": True}
            except Exception as exc:
                logger.debug("research conclusion raised: %s: %s", type(exc).__name__, exc)
                broke = f"{type(exc).__name__}: {exc}"
            if not busy or concluded or broke or wait is None:
                break
            if cancel is not None and cancel():
                break
            logger.debug("research conclusion waited %.0fs for the agent", wait)
            time.sleep(wait)

        if busy and not concluded and not broke:
            # Never ran: the agent stayed taken. Said as that, not as a conclusion that failed.
            journal["conclusion_error"] = "the agent was busy with another turn, so the conclusion never ran"
            answer = ""
            stopped = STOP_AGENT_BUSY
        elif concluded:
            answer = concluded
            book.add(answer)
            book.verify_all()
            if stopped == STOP_ROUND_CAP:
                stopped = STOP_CONCLUDED
        else:
            # No answer was written. Say so, and do not leave an analysis round's text sitting
            # in `answer` where the report renders it as the run's conclusion -- each round
            # still carries its own text under its own heading, so nothing is lost by being
            # honest here.
            journal["conclusion_error"] = broke or "the closing turn produced no answer"
            answer = ""
            if stopped in (STOP_ROUND_CAP, STOP_CONCLUDED):
                stopped = STOP_CONCLUSION_FAILED

    journal["answer"] = answer
    persist(stopped)
    yield _frame(
        "done",
        directory=directory,
        rounds_run=rounds_run,
        stopped_because=stopped,
        # The same summary the journal carries, so the live panel and the written report describe
        # one run rather than two.
        metric=journal.get("metric") or {},
        citations=book.summary(),
        unverified=[c.display_id for c in book.unverified()][:20],
        answer=answer,
    )


def _frame(phase: str, *, directory: Path | None = None, **fields: Any) -> dict[str, Any]:
    """One ``research`` frame. ``directory`` is carried as a string because a frame is JSON."""
    data: dict[str, Any] = {"phase": phase}
    if directory is not None:
        data["dir"] = str(directory)
    data.update(fields)
    return {"event": EVENT_RESEARCH, "data": data}
