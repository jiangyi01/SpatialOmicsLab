"""``run_post_analysis`` -- the one public entry point of the post-analysis engine.

    run_post_analysis(source, *, tool_name=None, task_type=None, results_dir=None) -> Path

Everything the prompt used to describe in 279 lines of prose happens behind this call. What the
function guarantees, in the order the guarantees matter:

1. **It always returns a directory containing a complete ``manifest.json``.** A missing source, an
   empty directory, an unreadable file and a crashing analysis all end in a manifest that says so.
   There is no exception path for a caller to handle, because the caller is a language model.
2. **It never writes under ``benchmarks/results/``** -- those are recorded experiment outputs.
   Pointed at one, it redirects and says where it went.
3. **It refuses to run at all while benchmarking is on** (contract non-negotiable 1b). The three
   gates in ``_enrich_prompt_with_post_analysis`` govern prompt text; they cannot govern a call that
   never built a prompt, so the gate is re-checked here.
4. **One failed analysis degrades the run to ``partial``**, never aborts it.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from spatialomicsgym.task_types import canonicalize

from .context import AnalysisContext
from .detect import (
    AMBIGUOUS_OUTPUT_NOTE,
    READABLE_BUT_UNRECOGNIZED,
    detect_task_type,
    tool_name_from_filenames,
)
from .manifest import DEFAULT_RESULTS_DIRNAME, TASK_TYPES, Manifest, no_handler_warning, step, write_manifest
from .sources import collect_files, leads_under_benchmark_results, scan_rows, staged_scan_root
from .tasks import runner_for

logger = logging.getLogger(__name__)

__all__ = ["run_post_analysis", "DEFAULT_RESULTS_DIRNAME"]


def run_post_analysis(
    source,
    *,
    tool_name: str | None = None,
    task_type: str | None = None,
    results_dir=None,
) -> Path:
    """Analyse a tool's output and write a results directory. Returns that directory.

    ``source`` is the file or directory the tool wrote. ``tool_name`` is the portal function name
    (``run_spagcn``); giving it lets the tool output registry supply the task type and the
    authoritative output filenames instead of guessing from the data. ``task_type`` overrides
    detection outright. ``results_dir`` defaults to ``<source>/post_analysis``, derived from the
    directory the caller already named -- never from a guessed global output root.
    """
    source = Path(source)
    # For the duration of this run, staged-input judgement is relative to the tree the caller
    # handed over: the markers describe worker scratch directories *inside* a run, and the run's
    # own name must not disqualify its contents (a live agent writing its post-conversion attempt
    # into ``scanpy_on_converted/`` had every result file dropped as "staged" without this).
    with staged_scan_root(source if source.is_dir() else source.parent):
        return _run_post_analysis(source, tool_name=tool_name, task_type=task_type, results_dir=results_dir)


def _run_post_analysis(
    source: Path,
    *,
    tool_name: str | None = None,
    task_type: str | None = None,
    results_dir=None,
) -> Path:
    # Contract non-negotiable 1b: a scored run must not be able to reach post-analysis at all, by
    # any path. Nothing is written -- not even an empty results dir. Checked BEFORE resolution and
    # not after, because resolution is not side-effect-free: a source under ``benchmarks/results``
    # has no derivable results directory, so ``_resolve_results_dir`` calls ``tempfile.mkdtemp`` --
    # and left a stray empty ``sog_post_analysis_*`` in the temp tree on every scored run.
    if _benchmarking_is_on():
        return _resolve_results_dir(source, results_dir, write=False)[0]

    results_dir, redirect_note = _resolve_results_dir(source, results_dir)

    try:
        results_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # Contract non-negotiable 4, and the first place it was breached (later writes go through
        # the same guard in :func:`_publish`). This function's docstring
        # and the prompt the agent is given both say it does not raise, and every caller is written
        # to that promise -- but a read-only output tree (PermissionError) or a ``post_analysis``
        # that is already a *file* (NotADirectoryError, FileExistsError) made it raise before a
        # manifest existed to record anything. Nothing can be written now, so returning the path is
        # all that is left; the caller finds no manifest there and reports the run unanalysed.
        logger.warning("post-analysis could not create %s: %s: %s", results_dir, type(exc).__name__, exc)
        return results_dir
    manifest = Manifest(tool_name=tool_name)
    _stamp_incomplete(results_dir, tool_name)
    if redirect_note:
        manifest.warn(redirect_note)

    # The deliberate exception to the rule :func:`_nothing_to_analyse` follows -- this one message
    # keeps the whole path, and a later reader should not "make it consistent". Its entire content is
    # "the path you gave me is wrong", and a basename makes a typo in a directory component
    # undiagnosable: ``Source 'celina' does not exist`` when ``celina`` is plainly right there. Nor
    # does it leak anything: a path that does not exist is not part of the sender's real layout.
    if not source.exists():
        manifest.fail(f"Source {str(source)!r} does not exist; there is nothing to analyse")
        _publish(results_dir, manifest)
        return results_dir

    # ``ours`` collects what was dropped for belonging to a previous post-analysis report. Without
    # it the one directory that skip can empty out -- a previous report, pointed at directly -- gets
    # described as holding no readable files while holding ten, which is exactly the kind of
    # confident-and-wrong sentence this engine exists to stop emitting.
    # ``not_analysed`` collects the rest of the directory -- images, PDFs, logs, empty files. It is
    # never analysis input, and it is not debris either: on the recorded stlearn run it is 200 of
    # the 202 files, and those 200 ``.jpeg`` tiles are the deliverable. Only the inventory reads it.
    ours: list[Path] = []
    not_analysed: list[Path] = []
    files = collect_files(source, exclude=results_dir, skipped=ours, not_analysed=not_analysed)
    if not files:
        manifest.fail(_nothing_to_analyse(source, ours, not_analysed))
        _publish(results_dir, manifest)
        return results_dir

    # ``notes`` is filled only when detection had to fall back to the data *and* the data supported
    # more than one task type. A run that quietly analysed one of two results and reported ``ok`` is
    # the most expensive kind of wrong this engine can be, because nothing in the report shows the
    # reader that a choice was made at all.
    # Recover the tool name from the registry when the caller named none, which in production is
    # always: ``autorun`` analyses a directory it discovered and has nothing to name it with. The
    # detection below already computes this and discards it, and the cost of discarding it is that
    # ``match_authoritative`` and L2's ``published_output_slot`` are dead code on every real run.
    if not tool_name:
        inferred = tool_name_from_filenames(files)
        if inferred:
            tool_name = inferred
            manifest.tool_name = inferred

    notes: list[str] = []
    if task_type:
        # The caller's word for the task, translated into the contract's where a mapping exists:
        # the recommender says "spatial_communication" for the task this engine calls
        # "cell_communication", and untranslated it sailed past the handler into the no-handler
        # partial branch. ``canonicalize`` never guesses -- unmapped spellings pass through
        # unchanged, so the pinned degrade paths for genuinely unknown types keep firing.
        resolved = canonicalize(task_type)
        why = "given by the caller" if resolved == task_type else f"given by the caller as {task_type!r}"
    else:
        resolved, why = detect_task_type(files, tool_name, notes=notes)
    manifest.task_type = resolved
    for note in notes:
        manifest.warn(note)

    if any(str(note).startswith(AMBIGUOUS_OUTPUT_NOTE) for note in notes):
        # The note says, in its own words, that the file analysed "comes first in filename order,
        # which is a tie-break and not a judgement". ``warn`` alone leaves the status at ``ok``, and
        # everything downstream reads the status and not the sentence: L2 finds no failing check and
        # returns the verdict ``ok``, L3 badges it green, and ``propose_next_steps`` offers follow-on
        # work built on the number. Same treatment as the no-handler branch below, for the same
        # reason -- the findings stand, the confidence in them does not.
        manifest.degrade()

    if resolved is None and READABLE_BUT_UNRECOGNIZED not in str(why or ""):
        # No count here. ``why`` already carries one, from the list detection actually reasoned
        # over -- everything here minus the tool's staged input -- and ``len(files)`` is the list
        # before that subtraction. Both clauses said "these files"/"the file(s) here" about one
        # directory, so the recorded xfuse run read "produced these 26 files: none of the 18 file(s)
        # here is a table or an h5ad". Neither number was wrong where it was computed: one of the
        # eight staged files is ``tissue_positions_list.csv``, so 18 is the only number that makes
        # the second clause true, and 26 the only one that makes the first. The reader is a model
        # choosing a next move with nothing else to go on -- this branch returns before
        # ``_write_scan``, so there is no scan table and no findings -- and two totals invite it to
        # go looking at the other eight, which are the staged input this run already published as
        # its result once. ``_write_scan`` performs exactly this subtraction for ``n_output_files``,
        # for the same reason; the layer that did the looking states the number, and only it.
        manifest.fail(f"Could not tell what kind of analysis produced these files: {why}")
        _publish(results_dir, manifest)
        return results_dir

    ctx = AnalysisContext(
        manifest=manifest,
        files=files,
        results_dir=results_dir,
        tool_name=tool_name,
        not_analysed=not_analysed,
    )
    # Above the no-handler branch, not below it. ``_write_scan``'s docstring states the invariant --
    # "every run publishes the same inventory, so L2/L3 can always answer 'what was there'" -- and
    # returning first was the one path past ``if not files`` that broke it, on the branch the
    # contract singles out as *not* a failure ("a run that produced real output must not be reported
    # as a failure merely because post-analysis has no plots for it"). The recorded ``scresolve``
    # run holds one 51 MB ``scresolve_enhanced.h5ad``; the engine listed it, declined to analyse it,
    # and published 0 tables, 0 figures and 0 findings. L2 read that as "nothing was verified" and
    # returned ``suspicious``, and the follow-on prompt closed with "No tool output was recorded for
    # this run; find it instead of re-running the tool" -- sending the model to search a directory
    # this function had already enumerated down to its single file. One registry profile of 63 and
    # one recorded run of 92 land here, and for that tool it is every run.
    #
    # The scan stats files; it does not open them, and it does not call ``ctx.analysed``, so
    # ``source_outputs`` stays empty and ``review._resolve_source`` still has no prediction to
    # adopt. ``shallow.py`` records what happened the last time a runner filled that list from a
    # directory listing.
    _write_scan(ctx, files, source)

    if resolved is None:
        # Real output the engine opened, parsed, and could not classify -- the live round-4 case is
        # a converted Visium ``.h5ad``: the conversion succeeded, ``fail`` here made review call a
        # good file ``unusable``, and the portal told the user to surface a broken result. The
        # contract reserves ``failed`` for results that are wrong (its ``unusable`` note says so in
        # those words); unclassified-but-readable is the no-handler branch below in different
        # words, so it gets the same treatment -- the sentence, ``partial``, and the scan
        # inventory above. Review then lands on ``suspicious`` ("nothing was verified about the
        # prediction's values"), which is the honest verdict for a file nothing judged. Broken
        # artifacts -- unparseable, zero rows, all-NaN, unsplit delimiter -- still take the
        # ``fail`` branch above; ``READABLE_BUT_UNRECOGNIZED`` appears only in the one reason
        # detect.py writes about files it actually read.
        manifest.warn(f"Could not tell what kind of analysis produced these files: {why}")
        manifest.degrade()
        _publish(results_dir, manifest)
        return results_dir

    if resolved not in TASK_TYPES:
        # A real registry task type (``resolution``) with no handler. The tool's output is fine; it
        # is post-analysis that has nothing to say about it, so this is partial and not a failure.
        manifest.warn(no_handler_warning(resolved, why))
        manifest.degrade()
        _publish(results_dir, manifest)
        return results_dir

    runner = runner_for(resolved)
    if runner is None:  # pragma: no cover - TASK_TYPES and the dispatch table are pinned together
        manifest.fail(f"No runner registered for contract task type {resolved!r}")
    else:
        with step(manifest, f"{resolved} analysis"):
            runner(ctx)

    _publish(results_dir, manifest)
    return results_dir


# ---------------------------------------------------------------------------------------


def _publish(results_dir: Path, manifest: Manifest) -> None:
    """Write a manifest that ends the run; a disk that stops cooperating degrades, never raises.

    Contract non-negotiable 4: ``run_post_analysis`` does not raise, and every caller is written to
    that promise -- ``autorun``, the CLI door, and the bare call the agent prompt tells generated
    code to make. The ``mkdir`` guard at the top covers a tree that was unwritable from the start,
    but a disk can fill and a tree can turn read-only *during* the run, and then this write is the
    first to hear about it. Nothing can be written now; the stamp from :func:`_stamp_incomplete` --
    or no manifest at all -- is already the honest story on disk, and the caller finds it and
    reports the run unanalysed.
    """
    try:
        write_manifest(results_dir, manifest)
    except OSError as exc:
        logger.warning("post-analysis could not publish %s: %s: %s", results_dir, type(exc).__name__, exc)


def _stamp_incomplete(results_dir: Path, tool_name: str | None) -> None:
    """Claim the results directory as ours the instant it exists, before anything is written into it.

    ``manifest.json`` is written last so that finding one means finding a complete report. The cost
    of that ordering is a window -- from ``mkdir`` here to :func:`write_manifest` at the end, which
    spans the output scan and every figure the runner draws -- in which the directory is full of our
    files and carries nothing saying so. A signal, an OOM kill or a notebook interrupt lands in it,
    and two things follow:

    * nothing on disk records that the run failed. The only signal is a *missing* file, which reads
      identically to a run that was never launched;
    * :func:`sources.collect_files` recognises "a previous report of ours" by exactly one thing --
      a readable ``manifest.json`` -- so the leftovers are invisible to it. Re-run the same source
      with ``results_dir`` pointing anywhere else and the first run's ``output_scan.csv`` is scanned
      as if the *tool* had produced it. That is the recorded stPlus failure in ``collect_files``'
      docstring ("8 files instead of 3 ... ``top_cell_type: "n_spots"``") reached by a second route.

    So a failed placeholder goes down first and every later ``write_manifest`` overwrites it. An
    interrupted run then leaves something true instead of nothing, and it leaves it *identifiably
    ours*. This run's own scan is unaffected: ``collect_files`` is called with ``exclude=results_dir``.

    A separate :class:`Manifest` is used rather than failing the live one, because
    :meth:`Manifest.fail` is sticky -- degrading the real manifest here would pin every completed
    run to ``failed``.
    """
    placeholder = Manifest(tool_name=tool_name)
    placeholder.fail(
        "Post-analysis was interrupted before it finished, so this report is incomplete. Anything "
        "in this directory was written by post-analysis, not by the tool that produced the output."
    )
    try:
        write_manifest(results_dir, placeholder)
    except OSError as exc:
        # Same reasoning as the ``mkdir`` guard above: contract non-negotiable 4 says this function
        # does not raise. A directory we cannot write is a directory the final manifest cannot reach
        # either, and that is reported there; losing the placeholder is not worth an exception.
        logger.warning("post-analysis could not stamp %s: %s: %s", results_dir, type(exc).__name__, exc)


def _nothing_to_analyse(source: Path, ours: list[Path], not_analysed: list[Path] | None = None) -> str:
    """Why there is nothing to analyse -- an empty directory, or one holding only our own reports.

    The source is named, not spelled out. These strings are stored in the manifest and then printed
    verbatim by the HTML report and the web portal -- and the report is the artefact built to be
    forwarded, so a full path here publishes the sender's home directory and project layout to
    whoever opens the page (13 of the 92 recorded runs did). It is also the same run named twice:
    the page identifies the run as ``celina/post_analysis`` and the warning called it
    ``/home/someone/work/runs/celina``. The absolute path is not lost -- ``source_outputs`` in the
    same manifest is the machine record and stays absolute.
    """
    named = source.name or str(source)
    if not ours:
        message = f"Source {named!r} contains no readable output files"
        if not_analysed:
            # Same distinction the ``ours`` clause below draws, for the same reason: "no readable
            # output files" is true of a directory of 200 ``.jpeg`` tiles and reads as "the tool
            # wrote nothing", which sends the reader off to look for output that is right there.
            suffixes = sorted({p.suffix.lower().lstrip(".") or "no suffix" for p in not_analysed})
            message += (
                f" -- it holds {len(not_analysed)} file(s) ({', '.join(suffixes)}) that this "
                "analysis cannot read a table out of, so the tool did write something"
            )
        return message
    return (
        f"Source {named!r} holds {len(ours)} files, but all of them belong to a previous "
        "post-analysis report rather than to a tool -- reading them back in would report this "
        "package's own findings as if a tool had produced them. Point post-analysis at the "
        "directory the tool wrote."
    )


def _benchmarking_is_on() -> bool:
    """Read the switch at call time; a stale import-time copy would defeat the gate."""
    try:
        from spatialomicsgym.config import default_config

        return bool(getattr(default_config, "benchmarking_enabled", False))
    except Exception:
        return False


def _lands_under_benchmark_results(path: Path) -> bool:
    """Contract non-negotiable 2, the one guarantee here that cannot depend on how a path was typed.

    The implementation is :func:`sources.leads_under_benchmark_results` and is deliberately shared
    rather than restated: L2 restated it, dropped the ``resolve()``, and rewrote a recorded
    experiment's manifest through ``benchmarks/manuscript/per_cell_results/multi_llm``.
    """
    return leads_under_benchmark_results(path)


def _resolve_results_dir(source: Path, requested, *, write: bool = True) -> tuple[Path, str]:
    """``(results_dir, note)``. ``note`` is non-empty only when the request had to be redirected.

    ``write=False`` answers the same question without touching the disk, for the caller that only
    needs a path to return. The one branch that would otherwise act is the ``mkdtemp`` fallback.
    """
    if requested is not None:
        requested = Path(requested)
        if not _lands_under_benchmark_results(requested):
            return requested, ""
        fallback = _fallback_dir(source)
        return (
            fallback,
            f"Refusing to write under benchmarks/results/ ({requested}): those are recorded "
            f"experiment outputs and are read-only. The analysis was written to {fallback} instead.",
        )

    default = (source if source.is_dir() else source.parent) / DEFAULT_RESULTS_DIRNAME
    if not _lands_under_benchmark_results(default):
        return default, ""
    if not write:
        return default, ""
    fallback = Path(tempfile.mkdtemp(prefix="sog_post_analysis_"))
    return (
        fallback,
        f"The source is under benchmarks/results/, which is read-only, so no results directory "
        f"could be derived from it. The analysis was written to {fallback} instead; pass "
        "results_dir= to choose somewhere durable.",
    )


def _fallback_dir(source: Path) -> Path:
    derived = (source if source.is_dir() else source.parent) / DEFAULT_RESULTS_DIRNAME
    if not _lands_under_benchmark_results(derived):
        return derived
    return Path(tempfile.mkdtemp(prefix="sog_post_analysis_"))


def _write_scan(ctx: AnalysisContext, files: list[Path], source: Path) -> None:
    """Every run publishes the same inventory, so L2/L3 can always answer "what was there"."""
    with step(ctx.manifest, "output scan"):
        import pandas as pd

        root = source if source.is_dir() else source.parent
        # The inventory is of the directory, so it is scanned over both lists. ``files`` alone is
        # the *analysis-candidate* list -- ``collect_files`` drops eight image/log suffixes and every
        # empty file before returning it -- and describing that as "Files found in the tool output"
        # understated 22 of the 92 recorded runs, stlearn by 2 against 202. ``ctx.not_analysed``
        # holds exactly what was dropped for being unreadable here, and nothing that was dropped for
        # belonging to somebody else, so re-uniting them cannot re-admit a previous report.
        frame = pd.DataFrame(scan_rows(sorted(set(files) | set(ctx.not_analysed)), root))
        ctx.table(frame, "output_scan.csv", title="Files found in the tool output", index=False)
        # In a benchmark or converted-input layout the directory also holds the dataset the tool was
        # *handed*. Under the label "Files written by the tool" that is a claim about provenance, and
        # on three recorded runs it is wrong by half: celloscope reported 8 files of which 4 are its
        # staged input, starfysh 9 of which 3. The scan table one line above already answers this per
        # row -- ``scan_rows`` writes a ``staged_input`` column -- so the number is taken from the
        # column rather than from ``len``, and what was excluded is published rather than dropped.
        # Three of the corpus's 394 unreadable files are staged too (two under cell2location's input
        # directory, one under xfuse's), which is why the column is read after the union and not
        # before it.
        staged = int(frame["staged_input"].sum()) if len(frame) and "staged_input" in frame else 0
        ctx.find("n_output_files", int(len(frame)) - staged, "Files written by the tool")
        if staged:
            ctx.find("n_staged_inputs", staged, "Staged input files in the same directory")
