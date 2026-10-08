"""L2 -- what the co-scientist does next, and when it stops.

Reads the manifest plus the ``review`` block from :mod:`spatialomicsgym.postanalysis.review` and
produces the contract's ``next_steps`` list::

    [{"action": "run_marker_enrichment", "why": "8 domains found, no functional annotation yet", "priority": 1}]

Exactly those three keys. Executability is *not* a fourth key -- it is derived from the action name
via :data:`NON_EXECUTABLE_ACTIONS`, so L3 can render a step without knowing this module exists.

Two guards are load-bearing:

**A wrong result never schedules more work on itself.** ``verdict == "unusable"`` proposes exactly
one step, ``surface_unusable_result``, and that action is non-executable. The failure being fenced
off is "do more analysis on garbage": a signal-free prediction or a transposed proportion table is
not a finding to build on, it is a defect to report.

**The self-analysis loop is bounded.** ``SOG_POST_ANALYSIS_MAX_FOLLOWUP_ROUNDS`` caps how many
follow-on ReAct turns one user turn may spawn. The default is
:data:`MAX_FOLLOWUP_ROUNDS_DEFAULT` = 1 and the value is clamped to
:data:`MAX_FOLLOWUP_ROUNDS_CEILING`; ``0`` keeps the checking and turns the acting off. One round
per step, highest priority first -- rounds are never re-proposed from their own output, so the
agent cannot talk itself into a longer run.

**Eval neutrality.** ``benchmarking_enabled`` short-circuits :func:`review_and_act` before it reads
or writes anything, mirroring the gate in ``_enrich_prompt_with_post_analysis``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spatialomicsgym.postanalysis.actions import NON_EXECUTABLE_ACTIONS, hint_for_action
from spatialomicsgym.postanalysis.manifest import BOOKKEEPING_FINDINGS
from spatialomicsgym.postanalysis.review import (
    RUN_STATUS_FAILED_DETAIL,
    discover_results_dir,
    failed_steps,
    has_no_prediction_slot,
    read_manifest,
    register_new_artifacts,
    review_manifest,
    write_review,
)
from spatialomicsgym.task_types import canonicalize

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

#: One follow-on analysis per user turn. Enough to close the loop -- "you found 8 domains, now
#: annotate them" -- without letting a single question become an unattended research programme.
MAX_FOLLOWUP_ROUNDS_DEFAULT = 1

#: Hard ceiling regardless of what the environment asks for.
MAX_FOLLOWUP_ROUNDS_CEILING = 3

#: Matches the ``SOG_POST_ANALYSIS_*`` naming of its sibling switch in ``config.py``; the config
#: field ``post_analysis_max_followup_rounds`` already reads this same variable, so there is exactly
#: one name a user can set.
MAX_ROUNDS_ENV = "SOG_POST_ANALYSIS_MAX_FOLLOWUP_ROUNDS"

# NON_EXECUTABLE_ACTIONS (actions that report rather than compute -- proposed, shown, never
# auto-run) is imported above from :mod:`spatialomicsgym.postanalysis.actions`, where it lives so
# the webui can filter next-step buttons without importing this engine-adjacent module.

_STOP_ROUND_CAP = "round_cap"
_STOP_STEPS_EXHAUSTED = "steps_exhausted"
_STOP_UNUSABLE = "verdict_unusable"
_STOP_NO_RUNNER = "no_runner"
_STOP_ERROR = "followup_error"
#: The round ran, and then the manifest could not be read back. Distinct from the two "the loop
#: finished" reasons on purpose -- see the re-check in :func:`review_and_act`.
_STOP_MANIFEST_LOST = "manifest_unreadable"

# action -> (why, keywords that mean "the manifest already has this")
_FOLLOWUPS: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {
    "deconvolution": [
        (
            "map_dominant_celltype_per_spot",
            "proportions are estimated but no per-spot dominant cell type has been mapped",
            ("dominant", "argmax"),
        ),
        (
            "run_celltype_colocalization",
            "with more than one cell type, which pairs share spots is the next question",
            ("coloc", "correlation", "neighborhood"),
        ),
        (
            "compare_against_a_second_deconvolution_tool",
            "a single method's proportions have no error bar; a second method gives one",
            ("compare", "concordance"),
        ),
    ],
    "spatial_clustering": [
        (
            "run_marker_enrichment",
            "domains were identified but carry no functional annotation yet",
            ("marker", "rank_genes", "deg"),
        ),
        (
            "annotate_domains_from_markers",
            "marker genes are only useful once they are turned into domain labels",
            ("annotat", "celltype_label"),
        ),
        (
            "test_domain_spatial_coherence",
            "a domain that is scattered across the slide is usually a clustering artefact",
            ("coheren", "spatial_stat", "moran"),
        ),
    ],
    "svg_detection": [
        (
            "run_functional_enrichment_on_svgs",
            "a gene list is a result only once it is tied to a pathway or process",
            ("enrich", "go_", "pathway"),
        ),
        (
            "cluster_svgs_into_spatial_modules",
            "co-varying SVGs form spatial programmes that individual genes do not show",
            ("module", "cluster"),
        ),
    ],
    "cell_communication": [
        (
            "rank_ligand_receptor_pairs_by_spatial_support",
            "interaction scores need a spatial sanity check before interpretation",
            ("rank", "ligand", "lr_"),
        ),
    ],
    # The four task types above the line and the three below it are the same seven
    # ``manifest.TASK_TYPES`` names, and every one of them now has a plan. The three here had none:
    # they fell through to ``_GENERIC_FOLLOWUP``, so an alignment run and a trajectory run were both
    # told "turn this output into an interpretation" and nothing else. They are also exactly the
    # task types ``tasks/shallow.py`` handles, and it already states per task type what a deep
    # runner would add -- ``shallow._WHAT_IS_MISSING``. These steps are that sentence made
    # actionable, so the two places do not describe different gaps.
    "alignment": [
        (
            "measure_registration_error",
            "an alignment with no residual does not say how far apart the matched points ended up",
            ("residual", "registration_error", "rmse"),
        ),
        (
            "compare_the_aligned_sections_region_by_region",
            "sections are aligned so that the same region can be compared between them",
            ("cross_section", "section_compar", "shared_domain"),
        ),
    ],
    "imputation": [
        (
            "validate_imputed_genes_against_held_out_measurements",
            "imputed values carry no accuracy estimate until they are checked against measured ones",
            ("held_out", "holdout", "cross_valid"),
        ),
        (
            "test_whether_the_imputed_genes_are_spatially_structured",
            "an imputed gene that is spatially flat carries no more information than its own mean",
            ("moran", "spatial_autocorr", "spatially_variable"),
        ),
    ],
    "trajectory": [
        (
            "plot_gene_trends_along_the_trajectory",
            "the genes that change along the path are what a trajectory is computed for",
            ("gene_trend", "trend", "along_pseudotime"),
        ),
        (
            "map_pseudotime_onto_the_tissue",
            "pseudotime scattered at random across the slide is usually an artefact, not a lineage",
            ("pseudotime_map", "spatial_pseudotime"),
        ),
    ],
}

#: The fallback for a ``task_type`` outside :data:`manifest.TASK_TYPES` -- ``resolution``, or
#: anything else a registry profile names that this contract has no handler for.
#:
#: The keyword tuple is empty on purpose. It used to be ``("summary",)``, which matched
#: ``column_summary.csv`` -- the column-means table ``tasks/shallow.py`` writes as its fallback
#: figure, from the runner whose own warning says post-analysis did no deep analysis of this run.
#: A scan is not an interpretation, and nothing a manifest records means "the findings have already
#: been summarised for the user", so there is no keyword that can honestly stand for it.
_GENERIC_FOLLOWUP = (
    "summarize_findings_for_the_user",
    "the run produced output that has not been turned into an interpretation yet",
    (),
)


# ----------------------------------------------------------------------------------------------
# knobs
# ----------------------------------------------------------------------------------------------


def max_followup_rounds() -> int:
    """Hard cap on follow-on ReAct rounds per user turn.

    ``SOG_POST_ANALYSIS_MAX_FOLLOWUP_ROUNDS`` overrides the default; anything unparseable falls back
    to the default rather than to zero, so a typo does not silently disable the feature, and the
    value is clamped into ``[0, MAX_FOLLOWUP_ROUNDS_CEILING]``.
    """
    raw = os.environ.get(MAX_ROUNDS_ENV, "").strip()
    if not raw:
        try:
            from spatialomicsgym.config import default_config

            raw = str(getattr(default_config, "post_analysis_max_followup_rounds", MAX_FOLLOWUP_ROUNDS_DEFAULT))
        except Exception:
            raw = str(MAX_FOLLOWUP_ROUNDS_DEFAULT)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return MAX_FOLLOWUP_ROUNDS_DEFAULT
    return max(0, min(MAX_FOLLOWUP_ROUNDS_CEILING, value))


def post_analysis_active() -> bool:
    """False whenever this whole layer must be invisible: benchmarking, or the user turned it off."""
    try:
        from spatialomicsgym.config import default_config
    except Exception:
        return False
    if getattr(default_config, "benchmarking_enabled", False):
        return False  # Gate 1, mirroring _enrich_prompt_with_post_analysis: benchmarking wins.
    return bool(getattr(default_config, "post_analysis_enabled", True))


# ----------------------------------------------------------------------------------------------
# proposals
# ----------------------------------------------------------------------------------------------


def _unusable_why(manifest: dict[str, Any], reasons: list[str]) -> str:
    """What a fatal verdict is allowed to claim, given how much of the result was actually read.

    ``Manifest.fail`` has ten call sites meaning two things. Post-analysis either opened a file *as
    the result* and judged it -- "Columns [...] are spot barcodes, not cell types", "every one of its
    spots is empty" -- or it recognised nothing among files it never opened: "Could not tell what kind
    of analysis produced these files". Both set ``status="failed"``, both become one fatal check,
    both arrived here as ``unusable`` and got the same sentence: "the result is wrong, not merely
    uninteresting ... Report this and stop; do not build further analysis on it." For the second kind
    that is a verdict on a file nothing looked at, and the instruction to stop discards a live run
    whose remedy L1 spelled out in the same breath -- pass ``tool_name=`` or ``task_type=``. Of the 28
    failed runs in the recorded corpus, 20 are of the second kind.

    ``source_outputs`` separates them exactly, with no threshold and no new contract: it is the list
    of files post-analysis opened as the result, non-empty for all 8 of the first kind and empty for
    all 20 of the second. ``engine.py`` draws the same line in a comment for the sibling case -- "The
    tool's output is fine; it is post-analysis that has nothing to say about it". Absent the key at
    all, the run gets the benefit of the doubt: no evidence that a result was read is not evidence
    that one was.

    The *reason* is the second half. A status-driven failure carries exactly one, and it restates the
    status it came from, so all 28 were handed ``run_status: L1 reported status='failed'`` -- a
    tautology, even for the eight the sentence is right about. What happened is in the manifest's own
    warnings, which this layer had all along. :data:`RUN_STATUS_FAILED_DETAIL` is imported rather than
    matched again so the two layers cannot disagree about what it says.
    """

    # Check details are sentence fragments and warnings are sentences, and both land mid-sentence
    # here: unjoined, the recorded spotclean warning reads "...in another file.. Report this".
    def _join(parts: list[str]) -> str:
        return "; ".join(p[:-1] if p.endswith(".") else p for p in (s.strip() for s in parts) if p)

    detail = _join([r for r in reasons if RUN_STATUS_FAILED_DETAIL not in r][:2])
    if not detail:
        detail = _join([str(w) for w in ((manifest or {}).get("warnings") or [])][:2])
    detail = detail or "a fatal check failed"

    if (manifest or {}).get("source_outputs"):
        return (
            f"the result is wrong, not merely uninteresting: {detail}. "
            "Report this and stop; do not build further analysis on it."
        )
    return (
        "post-analysis could not identify a result to read, so nothing here says the tool's output is "
        f"wrong: {detail}. Read the tool's own output, or name the task type, before concluding "
        "anything about this run."
    )


def propose_next_steps(manifest: dict[str, Any], review: dict[str, Any]) -> list[dict[str, Any]]:
    """Ordered ``{"action", "why", "priority"}`` list. Priorities start at 1 and ascend."""
    verdict = str((review or {}).get("verdict", "ok"))
    reasons = [str(r) for r in (review or {}).get("reasons", [])]

    if verdict == "unusable":
        return [_step("surface_unusable_result", _unusable_why(manifest, reasons), 1)]

    steps: list[dict[str, Any]] = []
    if verdict == "suspicious":
        steps.append(
            _step(
                "verify_flagged_result",
                "the self-check flagged: " + ("; ".join(reasons[:2]) or "an advisory check failed"),
                1,
            )
        )

    if str(manifest.get("status", "")).lower() == "partial":
        # ``partial`` alone does not mean a step failed. L1 degrades to ``partial`` for notes
        # routinely -- "no coordinates, so the tissue maps were skipped", "these columns are latent
        # topics, not cell types", "this task type has no deep runner yet" -- and on this repo's 92
        # recorded runs *all sixteen* partial manifests are of that kind: not one carries a warning
        # naming a step that raised. Every one of them proposed this retry, and since the action is
        # executable and lands at priority 1 under an ``ok`` verdict, the single follow-on round the
        # layer allows per user turn was spent re-running a tool that had not failed, with a
        # sentence about a decision L1 made on purpose handed over as the reason.
        #
        # ``review.py`` draws the distinction already -- ``failed_steps`` matches the one format
        # ``manifest.step()`` writes for an exception, and ``_run_checks`` reports "N analysis
        # step(s) did not complete, M warning(s) recorded" off it. The question is asked there, not
        # restated here, so the two layers cannot disagree about what failed.
        incomplete = failed_steps(manifest)
        if incomplete:
            steps.append(
                _step(
                    "retry_failed_analyses",
                    f"{len(incomplete)} analysis step(s) did not complete: " + "; ".join(incomplete[:2]),
                    len(steps) + 1,
                )
            )

    covered = _already_covered(manifest)
    # Canonicalized defensively: manifests written by this engine already carry contract spellings,
    # but a manifest handed over by another registry's vocabulary ("spatial_communication") used to
    # sail past every plan into the generic fallback.
    task_type = canonicalize(str(manifest.get("task_type", "")))
    candidates = _FOLLOWUPS.get(task_type) or [_GENERIC_FOLLOWUP]
    for action, why, keywords in candidates:
        if any(k in covered for k in keywords):
            continue
        steps.append(_step(action, why, len(steps) + 1))

    if not steps:
        steps.append(_step(*_GENERIC_FOLLOWUP[:2], 1))
    return steps


def executable_steps(steps) -> list[dict[str, Any]]:
    """The subset a runner may actually execute."""
    return [s for s in (steps or []) if str(s.get("action", "")) not in NON_EXECUTABLE_ACTIONS]


def build_followup_prompt(
    manifest: dict[str, Any], review: dict[str, Any], step: dict[str, Any], results_dir: str | Path
) -> str:
    """ASCII-only follow-on instruction.

    ASCII is not cosmetic: a Unicode bullet injected into a prompt has already produced a
    SyntaxError retry loop in the execute stage. Short and action-first for the same reason the
    other prompt guidance is -- a long preamble makes models skim the parameters.
    """
    # ``.get(k, default)`` returns the default only when the key is *absent*. L1 writes
    # ``"tool_name": null`` whenever the caller named no tool, so the key is there holding a null
    # and the fallback never fired: the model was told "The deconvolution run with None finished".
    task_type = _ascii(manifest.get("task_type") or "analysis")
    # No tool name means the clause goes away, rather than becoming "run with the tool".
    named_tool = _ascii(manifest.get("tool_name") or "")
    ran_with = f" with {named_tool}" if named_tool else ""
    verdict = _ascii(str((review or {}).get("verdict", "ok")))
    action = _ascii(str(step.get("action", "")))
    why = _ascii(str(step.get("why", "")))
    findings = _named_findings(manifest)
    caveats = "; ".join(_ascii(r) for r in (review or {}).get("reasons", [])[:2]) if verdict != "ok" else ""
    # ``propose_next_steps`` builds a suspicious run's first step as "the self-check flagged: " plus
    # this same ``reasons[:2]``, so printing both put one sentence in the prompt twice, four lines
    # apart -- for somde a 300-character sentence about a saturated q-value column, in a prompt whose
    # whole design is short and action-first. The step line is the instruction, so the caveat line is
    # what goes, and only when the step is about to say the same thing: the loop runs several steps
    # per verdict, and from round two the ``why`` is an ordinary follow-up that carries no reason.
    # The reason is *not* stripped from ``why`` instead, because the portal's run card and the HTML
    # report's next-step item render ``why`` with no reasons beside it -- there it is the whole
    # explanation, and "the self-check flagged" alone is a fragment.
    if caveats and caveats in why:
        caveats = ""

    lines = [
        f"FOLLOW-ON STEP: {action}",
        f"The {task_type} run{ran_with} finished and its results were self-checked (verdict: {verdict}).",
    ]
    # The tool's own output, named, because the step has to read the prediction and the prediction
    # is not in the results directory -- that holds this package's report. Pointed only there, a
    # step like compare_against_a_second_deconvolution_tool had nothing to compare: the only
    # per-cell-type table beside the manifest is five averages. L1 names the same hazard in
    # ``engine._nothing_to_analyse``: reading our own report back reports our findings as a tool's.
    sources = _source_files(manifest)
    if sources:
        lines.append(f"Tool output to analyse: {'; '.join(sources)}")
    lines.append(f"Post-analysis report (manifest, figures, tables): {_ascii(str(results_dir))}")
    if findings:
        lines.append(f"What it found: {findings}")
    if caveats:
        lines.append(f"Carry these caveats into the interpretation: {caveats}")
    lines.append(f"Do this next, and say why it matters: {why}")
    # HOW, not just WHAT: the action tokens are named for biologists, and a model handed only
    # "run_celltype_colocalization" invented its own method as often as not. One terse
    # ACTION-FIRST line from the shared table; absent for actions the table does not know.
    hint = hint_for_action(action)
    if hint:
        lines.append(f"How: {_ascii(hint)}")
    lines += [
        # Only an ok figure-only run is told its figures are the whole output; a flagged one (a figure
        # missing on disk) must not be told the opposite of its own reason (review of S3, 2026-10-01).
        _what_to_read(sources, figure_only=verdict == "ok" and has_no_prediction_slot(manifest)),
        # Not "into the post-analysis directory", which is where a model that followed the
        # instruction exactly put them: at its root, where ``manifest.check_artifact_path`` cannot
        # hold a path and therefore no reader can be pointed at one. The two subdirectories are the
        # manifest's whole vocabulary of places, and ``review.register_new_artifacts`` declares what
        # lands in them.
        f"Write any new figure into {_ascii(str(results_dir))}/figures/ and any new table into "
        f"{_ascii(str(results_dir))}/tables/.",
        "If the step turns out not to be possible on this data, say so in one sentence and stop.",
    ]
    return "\n".join(lines)


def _what_to_read(sources: list[str], *, figure_only: bool = False) -> str:
    """Point at the tool's output, or admit there is none to point at.

    The empty branch used to say "Read the existing outputs instead of re-running the tool", which
    presumes exactly what the lines above it deny. A run reaches that branch only when the manifest
    declared no source output at all, and the reason is already in the same prompt, in the caveat
    L2 wrote: "no readable source output was declared, so neither the slot nor the values could be
    checked". A model given both sentences can do one of two things, and both waste the turn --
    invent a reading, or spend the step discovering the contradiction. R79/AJ fixed the *diagnosis*
    for this run; this is the *instruction* that outlived it.

    The branch is not a corpus accident. ``docs/design/post_analysis_contract.md`` reserves ``status:
    "partial"`` for a registry task type with no post-analysis handler, ``partial`` runs still get a
    follow-on step, and a declined run has an empty ``source_outputs`` by construction -- so every
    no-handler run that gets a step lands here.

    Nothing here can name a location instead: the manifest records the outputs, not the directory
    they came from, and when there are none it records neither. So this says what is true and leaves
    the search to the step, which already has "say so in one sentence and stop" for when it fails.
    Both branches close on the same clause, because the guard they exist to carry is the same one:
    the step is worth taking only if it costs less than running the tool again.
    """
    if sources:
        return "Read the tool output named above instead of re-running the tool."
    if figure_only:
        # A run that predicts nothing has nothing else to find: "find it" sent the step hunting for
        # a prediction the plot tool never wrote (live run 2026-10-01, S3).
        return (
            "The figures in the manifest are this run's whole output; read them there instead of re-running the tool."
        )
    return "No tool output was recorded for this run; find it instead of re-running the tool."


#: How many findings the prompt names before it stops listing. Same reason as ``_MAX_NAMED_SOURCES``.
_MAX_NAMED_FINDINGS = 4

#: Post-analysis counting its own scan, not something the tool found. ``engine._write_scan`` appends
#: these before any task runner has appended anything, so one held slot 1 of four in all 71 recorded
#: runs that have findings at all -- and the task's headline number, computed and appended last, is
#: what fell off the end. An SVG run's prompt named the genes *tested* and never the genes found
#: significant. They are also redundant: the line above the findings names those files.
#:
#: The list now lives in :mod:`spatialomicsgym.postanalysis.manifest`, because ``review._verdict``
#: needs the same answer for the same reason and cannot import this module -- this one imports
#: ``review``. The name is kept bound to the same object so this module reads as it did, and so the
#: anti-drift test that runs ``_write_scan`` against it keeps testing the one list both surfaces use.
_BOOKKEEPING_FINDINGS = BOOKKEEPING_FINDINGS


def _named_findings(manifest: dict[str, Any]) -> str:
    """The findings the model is shown, plus an honest count of the ones it is not.

    Cutting the list in silence lets four numbers read as the whole result: 62 of the 92 recorded
    runs dropped at least one finding, among them ``n_significant``, ``top_gene``, ``top_cell_type``
    and ``n_dominant_types``. ``_source_files`` below has always said "and N more"; this says the
    same, and points at the manifest the prompt has just named. The remainder counts *every* unnamed
    finding, bookkeeping included, so it agrees with what the manifest holds.
    """
    all_findings = [f for f in (manifest.get("findings") or []) if isinstance(f, dict)]
    worth_naming = [f for f in all_findings if f.get("key") not in _BOOKKEEPING_FINDINGS]
    shown = [_ascii(f"{f.get('label', f.get('key'))}={f.get('value')}") for f in worth_naming[:_MAX_NAMED_FINDINGS]]
    if not shown:
        return ""
    rest = len(all_findings) - len(shown)
    return "; ".join([*shown, f"and {rest} more in the manifest"] if rest else shown)


#: How many source files the prompt names before it stops listing. Long prompts make models skim.
_MAX_NAMED_SOURCES = 3


def _source_files(manifest: dict[str, Any]) -> list[str]:
    """What the tool actually wrote, as the manifest recorded it -- not re-scanned from disk."""
    raw = [_ascii(str(s)) for s in (manifest.get("source_outputs") or ()) if str(s).strip()]
    if len(raw) <= _MAX_NAMED_SOURCES:
        return raw
    # This said "in the same directory", which ``collect_files`` does not promise -- it walks the
    # source with ``rglob``. The recorded run_scanpy_spatial_domain/merfish run wrote 19 outputs
    # across ten directories, eight of them one clustering resolution each; the prompt named three
    # from ``res_0_15`` and told the model the other sixteen were beside them. The step is
    # instructed to read the named output instead of re-running the tool, so those eight
    # resolutions were unreachable. The common root is already in the paths.
    return [*raw[:_MAX_NAMED_SOURCES], f"and {len(raw) - _MAX_NAMED_SOURCES} more {_where(raw)}"]


def _where(paths: list[str]) -> str:
    """The directory the unnamed outputs are under, or no claim about where they are."""
    try:
        root = os.path.commonpath(paths)
    except ValueError:  # absolute and relative paths mixed together have no common root
        return "listed in the manifest"
    return f"under {root}" if root else "listed in the manifest"


# ----------------------------------------------------------------------------------------------
# the bounded loop
# ----------------------------------------------------------------------------------------------


def review_and_act(
    results_dir: str | Path | None = None,
    *,
    roots=None,
    since: float = 0.0,
    runner: Callable[[str], Any] | None = None,
    max_rounds: int | None = None,
    exclude=(),
) -> dict[str, Any] | None:
    """Check the results, write the verdict into the manifest, then act on the plan.

    Returns ``None`` -- having touched nothing -- whenever the layer is inactive. Otherwise returns
    ``{"results_dir", "review", "next_steps", "rounds_run", "stopped_because", "prompts"}``.
    """
    if not post_analysis_active():
        return None

    directory = (
        Path(results_dir) if results_dir is not None else discover_results_dir(roots or (), since, exclude=exclude)
    )
    if directory is None:
        return None
    manifest = read_manifest(directory)
    if manifest is None:
        return None

    review = review_manifest(manifest, directory)
    steps = propose_next_steps(manifest, review)
    write_review(directory, review, steps)

    outcome: dict[str, Any] = {
        "results_dir": str(directory),
        "review": review,
        "next_steps": steps,
        "rounds_run": 0,
        "prompts": [],
        "stopped_because": _STOP_STEPS_EXHAUSTED,
    }

    if review["verdict"] == "unusable":
        outcome["stopped_because"] = _STOP_UNUSABLE
        return outcome
    if runner is None:
        outcome["stopped_because"] = _STOP_NO_RUNNER
        return outcome

    cap = max_followup_rounds() if max_rounds is None else max(0, min(MAX_FOLLOWUP_ROUNDS_CEILING, int(max_rounds)))
    todo = executable_steps(steps)
    if cap <= 0 or not todo:
        outcome["stopped_because"] = _STOP_ROUND_CAP if cap <= 0 else _STOP_STEPS_EXHAUSTED
        return outcome

    for step in todo[:cap]:
        prompt = build_followup_prompt(manifest, review, step, directory)
        try:
            runner(prompt)
        except Exception as exc:  # a follow-on is a bonus; a failing one must not fail the turn
            logger.debug("follow-on round failed: %s", exc, exc_info=True)
            outcome["stopped_because"] = _STOP_ERROR
            return outcome
        outcome["prompts"].append(prompt)
        outcome["rounds_run"] += 1

        # Declare what the round wrote, before anything reads the manifest again. The round is the
        # only writer in this system that produces artifacts and cannot register them -- it is an
        # agent turn, not a call into ``Manifest.add_figure`` -- so this is where its output enters
        # the one list the re-check below, the portal card and ``report.html`` all read. Registering
        # after the re-check instead would leave the re-check judging a directory it half sees.
        register_new_artifacts(directory)

        # Re-check after the round: a follow-on that produced a wrong result must not be followed
        # by another round. Re-reading the manifest is what makes this a loop rather than a list.
        #
        # A round is a full agent turn against this same directory, so it can leave the manifest
        # unreadable -- moved, replaced by a truncated write, deleted by a tool that failed after
        # starting. Continuing past that used to run the remaining rounds against the review
        # written *before* the round: the one state this re-check exists to distrust, and now with
        # no evidence of what the round did. Each further round is another real agent turn, so the
        # cost was rounds of work, not a stale field. Stopping under its own reason is what keeps
        # ``steps_exhausted`` meaning "the plan was carried out and checked".
        refreshed = read_manifest(directory)
        if refreshed is None:
            logger.debug("manifest unreadable after follow-on round; stopping: %s", directory)
            outcome["stopped_because"] = _STOP_MANIFEST_LOST
            return outcome
        manifest = refreshed
        review = review_manifest(manifest, directory)
        # The plan is re-derived with the review, not carried over it. Recomputing only half of the
        # pair wrote "verdict: unusable" beside the three analyses proposed before the round that
        # made it unusable -- and the ``unusable`` branch of ``propose_next_steps`` exists to say
        # the opposite, "report this and stop; do not build further analysis on it". L3's report and
        # the web portal read ``next_steps`` out of the manifest, so that is what a reader was
        # shown. ``todo`` deliberately keeps its pre-round contents: those steps were planned
        # against a result still judged usable, and re-planning the worklist mid-flight is a
        # different loop from the bounded one this is.
        steps = propose_next_steps(manifest, review)
        outcome["review"] = review
        outcome["next_steps"] = steps
        write_review(directory, review, steps)
        if review["verdict"] == "unusable":
            outcome["stopped_because"] = _STOP_UNUSABLE
            return outcome

    outcome["stopped_because"] = _STOP_ROUND_CAP if len(todo) > cap else _STOP_STEPS_EXHAUSTED
    return outcome


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------


def _step(action: str, why: str, priority: int) -> dict[str, Any]:
    return {"action": action, "why": why, "priority": int(priority)}


def _already_covered(manifest: dict[str, Any]) -> str:
    """Lower-cased blob of what L1 already produced or deliberately refused, so a step is not
    proposed twice -- and so work L1 declined is not proposed at all."""
    bits: list[str] = []
    for kind in ("figures", "tables"):
        for entry in manifest.get(kind) or ():
            if isinstance(entry, dict):
                # ``path`` and ``kind`` are post-analysis's own vocabulary; ``title`` is not.
                # ``tasks/shallow.py`` titles its fallback figure ``f"Column means: {path.name}"``,
                # which puts *the tool's* filenames into a blob that is read as a record of what
                # post-analysis did. A cell-communication run writing ``lr_ranking.csv`` therefore
                # got the title "Column means of lr_ranking.csv", matching ``rank`` and ``lr_`` --
                # and lost ``rank_ligand_receptor_pairs_by_spatial_support``, its only follow-up, to
                # a figure whose caption says this task type has no deep support yet.
                #
                # Nothing is lost by dropping it: every keyword the deep runners genuinely cover is
                # carried by an artefact path (``markers.csv``, ``dominant_celltype.csv``,
                # ``colocalization.csv``) or a finding key, and on the four recorded post-analysis
                # manifests the proposed step list is unchanged under both verdicts.
                bits.extend(str(entry.get(k, "")) for k in ("path", "kind"))
    for finding in manifest.get("findings") or ():
        if isinstance(finding, dict):
            bits.append(str(finding.get("key", "")))
    if _l1_found_no_signal(manifest):
        # Everything above infers "already done" from what the manifest *contains*, which reads an
        # analysis L1 skipped as one still to do. On a signal-free result L1 skips exactly the
        # value-dependent analyses -- `_dominant`, `_colocalization`, the per-domain markers -- so
        # their keywords are missing precisely because L1 refused to run them, and the planner
        # re-proposed the work L1 had just declined. A prediction that does not vary cannot support
        # any of them: the dominant cell type of an identical composition and the co-localization of
        # a constant matrix are artefacts of the arithmetic, not measurements.
        for _action, _why, keywords in _FOLLOWUPS.get(str(manifest.get("task_type", "")), ()):
            bits.extend(keywords)
    return " ".join(bits).lower()


def _l1_found_no_signal(manifest: dict[str, Any]) -> bool:
    """True when L1's own ``signal_free`` finding says the prediction carries no information."""
    for finding in manifest.get("findings") or ():
        if isinstance(finding, dict) and finding.get("key") == "signal_free":
            return bool(finding.get("value"))
    return False


def _ascii(text: str) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")
