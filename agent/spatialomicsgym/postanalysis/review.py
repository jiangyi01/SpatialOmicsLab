"""L2 -- the co-scientist's self-check over its own results.

Consumes the ``manifest.json`` written by L1 (see ``docs/design/post_analysis_contract.md``) and
produces the contract's ``review`` block::

    {
        "verdict": "ok" | "suspicious" | "unusable",
        "reasons": [...],
        "checks": [{"name": ..., "passed": ..., "detail": ...}],
    }

``unusable`` is reserved for results that are **wrong**, not merely uninteresting. Every fatal
check below is anchored to a failure this repository actually recorded and debugged, and each one
reuses the existing benchmarking guard rather than re-deriving the heuristic:

  ``published_output_slot``      the file analysed is not the slot the tool's own registry profile
                                 declares. This is how Celloscope's ``celloscope_data/matB.csv``
                                 -- a genes x cell-types *marker flag* table whose column names are
                                 genuine cell types -- came to be read as a proportions matrix, and
                                 how STRIDE's LDA topic matrices were scored as cell types.
                                 Reuses ``tool_output_registry`` + ``_is_staged_input_path``.
  ``topic_matrix_masquerade``    the "cell type" columns are integer / ``topic_3`` / ``factor_1``
                                 indices. Reuses the inspector's archetype-column rejection.
  ``marker_table_masquerade``    every value is 0 or 1: a marker/indicator table, not proportions.
  ``group_table_masquerade``     a cluster or domain column names every row once: the rows are groups,
                                 not spots. A model's cluster x donor cross-tabulation was reviewed
                                 ``ok`` as a deconvolution with a donor as its top cell type.
                                 Reuses ``tables.group_key``, the question L1 now asks. A
                                 warning, not a fatal, when nothing names the rows and there are too
                                 few to be spots (``tables.spot_evidence``): the same table with its
                                 key written as the unnamed index.
  ``proportion_orientation``     oriented by **row sums**, never by label text. ``spot_0`` is short,
                                 has no dash and is not all digits, so it reads as a cell-type name;
                                 a correct 6142 x 7 matrix was transposed to 7 x 6142 on exactly
                                 that reasoning. Proportion vectors sum to 1 -- ask the numbers.
                                 Ask shape too: a proportion vector runs over cell types and there
                                 are never more cell types than spots, so "the columns sum to 1" on
                                 a 4035 x 44 Tangram file (it normalises each cell type across the
                                 spots) is not a transposed composition and is no longer fatal.
  ``signal_free``                a constant prediction. SVCA once reported every gene at exactly
                                 1/3 with ``converged=True``; standardization turned that into
                                 "every gene is a significant SVG" and recall against any truth set
                                 became 1.0 by construction.
  ``run_status``                 L1 declared the run failed.

Warnings (-> ``suspicious``) rather than fatals:

  ``score_column_retained``      the tool reported a per-gene ranking statistic under a name the
                                 standardizer's ``_SVG_SCORE_COLUMNS`` whitelist does not know, so
                                 the standardized ``score`` silently becomes 0.0/NaN and the
                                 ranking is lost. SpatialDE, SOMDE, SpaGFT, squidpy and SpotGF all
                                 lost their scores this way before their names were whitelisted;
                                 this check is what catches the next tool to do it.
  ``pvalue_is_measured``         the p-value column carries one or two distinct values -- a
                                 placeholder flag rather than a computed test.
  ``stale_prediction``           the analysed file is much older than its siblings. Reuses
                                 ``output_inspector._stale_file_warning``.
  ``label_from_input``           the column published as the domains has the name of one the
                                 input arrived with: the file descends from a conversion utility's
                                 output, whose own record lists that name among the columns it
                                 converted. Names, not values -- a tool that wrote over such a
                                 column fails it too, and the detail says "unless". Emitted only
                                 when such a record exists.
  ``manifest_schema`` / ``declared_artifacts_exist``  contract violations in the manifest itself.

This module never mutates a prediction and never writes outside ``results_dir``. It is a no-op
under ``benchmarking_enabled`` -- that gate lives in :mod:`spatialomicsgym.postanalysis.next_step`,
which is the only caller the agent uses.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import stat
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# The contract's "One schema module" rule: `postanalysis.manifest` owns the task-type vocabulary and
# L2 imports it. A second literal copy here would have to be pinned by an equality test, and a test
# that pins a duplication is worse than no duplication -- an import cannot drift. `manifest` is
# stdlib-only, so this costs nothing at import time (non-negotiable 3).
from spatialomicsgym.postanalysis.manifest import (
    BOOKKEEPING_FINDINGS,
    DEFAULT_RESULTS_DIRNAME,
    TASK_TYPES,
    declined_for_no_handler,
    discard_staging,
    is_our_manifest_payload,
    open_staging,
    staging_name,
)

# The recorded-output guard, imported and never restated. L2 restated it once, dropped the
# ``resolve()`` that makes it work, and rewrote a recorded experiment's manifest through a symlink.
# `sources` is stdlib-only at import time, same as `manifest`, so non-negotiable 3 still holds.
from spatialomicsgym.postanalysis.sources import (
    is_under_benchmark_results,
    leads_under_benchmark_results,
    staged_scan_root,
)

# The one question "does this axis carry cell types?" asked in one place. L1's reader
# (`tables.orient_proportions`) and this reviewer both branch on it, and when they were two separate
# spellings of the same rule they disagreed the moment either was tuned: a run L1 had oriented
# correctly came back from L2 as `suspicious`. `tables` is stdlib-only at import time, same as the
# two above, so non-negotiable 3 still holds.
from spatialomicsgym.postanalysis.tables import why_not_cell_types

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Every name this module can emit, in the order it emits them. Frozen: L3 renders these and the
# next-step planner branches on them, so a rename is a contract change.
CHECK_NAMES = (
    "manifest_schema",
    "run_status",
    "declared_artifacts_exist",
    "published_output_slot",
    "stale_prediction",
    "label_from_input",
    "topic_matrix_masquerade",
    "marker_table_masquerade",
    "group_table_masquerade",
    "proportion_orientation",
    "signal_free",
    "score_column_retained",
    "pvalue_is_measured",
    # Emitted only when a checker raised. It used to reuse ``manifest_schema``, which put the same
    # name in the list twice with opposite answers and told the reader a well-formed manifest was
    # the problem. Last, because it is about the review rather than about the run.
    "review_aborted",
)

VERDICTS = ("ok", "suspicious", "unusable")

#: The whole of what the ``run_status`` check has to say when L1 gave up: it restates the status it
#: was derived from. Every status-driven failure carries this one reason and nothing else, so a
#: consumer that renders ``reasons`` verbatim publishes a tautology. Exported so ``next_step`` can
#: recognise and drop it rather than matching the sentence again -- the same reason ``_FAILED_STEP_RE``
#: is imported there instead of restated, so the two layers cannot disagree about what it says. The
#: cause L1 actually recorded is in ``manifest["warnings"]``.
RUN_STATUS_FAILED_DETAIL = "L1 reported status='failed'"

#: How many rows a degeneracy rule needs before its answer means anything. Below it the rules do not
#: run -- and the checks used to publish a *passing* claim anyway ("5 distinct values across 3 rows",
#: "'pvalue' takes 1 distinct values over 10 genes"), which ``_verdict`` then counted as the examined
#: value check that pays for "ok". The recorded 3-row squidpy output reached "ok" that way. The floor
#: itself is unchanged; what changes is that a check below it reports ``examined=False``.
_MIN_ROWS_TO_JUDGE = 10

#: The same idea for the 0/1 rule, which needs more rows: a handful of proportions that all land on
#: 0 or 1 is not yet evidence of an indicator table.
_MIN_ROWS_FOR_BINARY = 20

#: The subset of :data:`CHECK_NAMES` that judges the prediction's *numbers*. A review holding none
#: of these has verified nothing about the result -- the file was in the right slot and no older
#: than its siblings, which says nothing about what is inside it -- and :func:`_verdict` refuses to
#: call that "ok". This is the general form of the recorded defect: L2 parses only delimited text,
#: so the ``.h5ad`` a spatial clustering tool normally publishes reached a verdict on the structural
#: checks alone, and a run where every spot landed in one domain came back "ok" while the identical
#: run written as CSV came back "unusable". The one exception is a run whose task type predicts
#: nothing (:func:`has_no_prediction_slot`): there the written figure is what pays for "ok".
#:
#: Membership is decided by what a check reads, not by how serious it is. ``score_column_retained``
#: and ``topic_matrix_masquerade`` were in here and read only column *names* -- "'intrinsic' is a
#: name the standardizer reads", "columns are cell-type names" -- and a name check passing was
#: enough to buy "ok". On the recorded SVCA run that was the entire verdict: ten genes with
#: ``pvalue`` equal to 0.01 in every row, below every row floor L2 has, so nothing examined a number
#: and the fallback to L1's own ``signal_free`` finding never fired because the name check counted as
#: one. L1 had recorded "Result carries no information" and warned that recall 1.0 was by
#: construction; L2 published "ok" over it. Both checks still run and are still reported -- one of
#: them is still fatal -- they just no longer stand in for having looked at the values.
VALUE_CHECK_NAMES = frozenset(
    {
        "marker_table_masquerade",
        "proportion_orientation",
        "signal_free",
        "pvalue_is_measured",
    }
)

# "Are these column labels cell types" is L1's question, asked in ``tables.topic_like_columns`` and
# imported here rather than restated. The private pair that used to sit at this line -- a bare-index
# pattern and a ``factor|arch|archetype|component|topic|program`` word list -- was narrower in the
# one way that matters: a word list can only recognise stems it has already met. L1 carries a second
# test for the *shape* a generated axis has, one shared stem plus an ungapped counter, precisely
# because three recorded tools name their axis something the word list never heard of. So Starfysh's
# ``arch_0`` failed here and SpatialGlue's ``alpha_omics1`` passed, on runs L1 had described with
# the same sentence, and the two attention weights L1 named as positional were quoted back by this
# check -- ``fatal=True``, so a pass is strong evidence -- as its example of names that are fine.

# Names that read as "a per-gene ranking statistic" when the standardizer's whitelist has never
# heard of them. Deliberately broad: a false positive costs one advisory line.
_SCORE_NAME_HINTS = ("score", "index", "stat", "moran", "geary", "fsv", "llr", "autocorr", "variance", "rank")

# The significance-column names live in ``output_standardizer._SVG_PVALUE_COLUMNS`` and are read
# from there, not restated here. The private copy that used to sit at this line held the same names
# as an unordered set, which threw away the half of that tuple that is load-bearing: it is ordered
# adjusted-before-raw because selection must be thresholded on the adjusted value, and scanning a
# file's own column order picks the raw one on every recorded tool that publishes both.

# How -- and only how -- L1 records a step that raised: ``manifest.step()`` writes
# ``f"{label} failed: {type(exc).__name__}: {exc}"``. Everything else in ``warnings`` is a note, and
# L1 degrades a run to ``partial`` for notes routinely ("Marker ranking needs the expression
# matrix"; "no coordinates, so the domain map was skipped"; a signal-free result). Counting warnings
# reported those as steps that did not complete, which sends the planner after a step that never ran.
_FAILED_STEP_RE = re.compile(r"\bfailed:\s+[A-Za-z_][A-Za-z0-9_]*:")


@dataclass
class _Check:
    name: str
    passed: bool
    detail: str
    fatal: bool = False  # a failing fatal check means "unusable"; otherwise "suspicious"
    #: False when the check reached its conclusion without inspecting anything. "no numeric
    #: proportions to check" and "no label column to check" are *passing* checks that are not
    #: evidence of anything, and they were being quoted back as the reasons the result is sound.
    #: Not a key of its own -- the contract's check entry is exactly ``{name, passed, detail}``.
    examined: bool = True

    def as_dict(self) -> dict[str, Any]:
        """``passed`` is tri-state on the wire and two-state in memory, deliberately.

        In memory it must stay ``True`` for a check that examined nothing, because ``_verdict``
        counts ``not c.passed`` and "there was nothing to check" is not a failure. On the wire it
        must not be ``True``, because every reader downstream treats ``True`` as evidence and paints
        it green: ``PASS proportion_orientation -- every value in the matrix is 0, so ... the
        orientation could not be decided`` is an honest sentence under a badge that contradicts it.

        ``None`` is not a new state to teach anyone. ``report/manifest._tristate`` already declines
        to guess for it and ``report/render`` already has the grey ``--`` glyph -- that branch was
        simply unreachable, because nothing ever produced the value it tests for.
        """
        return {
            "name": self.name,
            "passed": self.passed if self.examined else None,
            "detail": self.detail,
        }


# ----------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------


def failed_steps(manifest: dict[str, Any]) -> list[str]:
    """The warnings naming a step that raised -- not every warning a ``partial`` run carries.

    ``partial`` is L1's word for two different things. A step did not run: ``manifest.step()``
    caught an exception, a figure wrote no bytes, there is no handler for the task type. Or every
    step ran and the confidence in the answer is lower: the columns are positional labels rather
    than cell types, almost every gene passes the threshold, the task type was picked by a
    tie-break. Only the first kind is something to go back and finish, and :data:`_FAILED_STEP_RE`
    is what tells them apart.

    Every layer that needs the distinction asks here -- L2's ``run_status`` detail, the follow-on
    planner's retry step, L3's banner -- so none of them can give the reader a different answer.
    """
    return [str(w) for w in (manifest.get("warnings") or []) if _FAILED_STEP_RE.search(str(w))]


def review_manifest(manifest: dict[str, Any], results_dir: str | Path) -> dict[str, Any]:
    """Judge one analysis run. Returns the contract's ``review`` block; never raises.

    As root over a directory the agent can write, the judging runs in a child as that account. It
    reads every table ``source_outputs`` names, and the agent can write the manifest that names
    them: in the root portal process it read another account's prediction table and quoted that
    table's labels into this account's review (hunt 2026-09-30, rp-u18). In the child it reads only
    what the account could read itself. A child that fails gives a review that says so; root does
    not judge the run here instead.
    """
    d = Path(results_dir)
    delegated = _as_the_owner(d, "review", {"manifest": manifest})
    if delegated is None:
        return _review_here(manifest, d)
    review = delegated.get("review")
    if isinstance(review, dict) and review.get("verdict") in VERDICTS:
        return review
    return _verdict(
        [
            _Check(
                "review_aborted",
                False,
                f"the review could not run as the account that can write {d.name}/, so nothing about "
                "this run was checked",
            )
        ],
        manifest,
    )


def _review_here(manifest: dict[str, Any], results_dir: str | Path) -> dict[str, Any]:
    """:func:`review_manifest` in this process -- the whole of it, minus the choice of process."""
    checks: list[_Check] = []
    try:
        checks.extend(_manifest_checks(manifest, Path(results_dir)))
        # Judged relative to the directory L1 was handed, as L1 judged it. The engine scopes its
        # staged-input reading to that directory and L2 runs after the engine has returned, so the
        # slot check read the whole absolute path and called a correct run in a caller-named
        # ``bass_on_converted/`` unusable -- the follow-on loop then stopped on a result L1 had
        # analysed as ok (hunt 2026-09-30, u18-postanalysis-2).
        run_root = _run_root(Path(results_dir))
        with staged_scan_root(run_root) if run_root is not None else nullcontext():
            checks.extend(_prediction_checks(manifest))
    except Exception as exc:  # a checker that crashes must not take down the run it is checking
        logger.debug("review checks aborted: %s", exc, exc_info=True)
        # Under its own name, not the schema check's. Reusing ``manifest_schema`` put that name in
        # the list twice with opposite answers -- which the contract-shape test forbids, and which it
        # only ever exercised on the healthy path -- and handed the reader "manifest_schema: review
        # aborted: IndexError" about a manifest that conformed. The value checks are the ones that
        # were lost, so the sentence has to say the review stopped, not that the manifest is wrong.
        checks.append(
            _Check(
                "review_aborted",
                False,
                f"the value checks did not finish: {type(exc).__name__}: {exc}",
            )
        )

    order = {n: i for i, n in enumerate(CHECK_NAMES)}
    checks.sort(key=lambda c: order.get(c.name, len(order)))
    return _verdict(checks, manifest)


def _run_root(results_dir: Path) -> Path | None:
    """The directory L1 analysed, when the layout says which: ``<source>/post_analysis``.

    That is where the engine puts a run it was not given a ``results_dir`` for, and the directory it
    scopes staged-input judgement to (``engine.run_post_analysis``). Any other results directory
    says nothing about its source, so ``None`` keeps the full-path reading, the stricter one.
    """
    return results_dir.parent if results_dir.name == DEFAULT_RESULTS_DIRNAME else None


def review_results_dir(results_dir: str | Path) -> dict[str, Any]:
    """Load ``<results_dir>/manifest.json`` and review it. Never raises."""
    d = Path(results_dir)
    manifest = read_manifest(d)
    if manifest is None:
        return {
            "verdict": "suspicious",
            "reasons": [f"no readable manifest.json under {d}"],
            "checks": [_Check("manifest_schema", False, "manifest.json is missing or not valid JSON").as_dict()],
        }
    return review_manifest(manifest, d)


_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


def read_manifest(results_dir: str | Path) -> dict[str, Any] | None:
    """Read a manifest, or ``None`` if it is absent / unreadable / not a JSON object.

    ``None`` too when ``results_dir`` or its ``manifest.json`` is a symlink, or the manifest is not a
    regular file. Every L2 writer reads through here before it rewrites, and the portal ran L2 as
    root over directories the agent's code can change: a ``manifest.json`` swapped for a link to the
    provider key store was read, merged with the verdict and published as an ordinary file in the
    account's run (hunt 2026-09-30, u18-postanalysis-1). The directory is opened without following
    a link and the file is opened relative to it the same way, so neither can be one at the moment
    of the read; ``O_NONBLOCK`` keeps a FIFO planted under the name from parking the review forever.
    """
    dir_fd = _open_results_dir(results_dir)
    if dir_fd is None:
        return None
    try:
        return _read_manifest_at(dir_fd)
    finally:
        os.close(dir_fd)


def _open_results_dir(results_dir: str | Path) -> int | None:
    """A descriptor on ``results_dir`` itself, never on a link standing at its name; ``None`` if absent."""
    try:
        return os.open(results_dir, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW)
    except (OSError, TypeError, ValueError):
        return None


def _read_manifest_at(dir_fd: int) -> dict[str, Any] | None:
    """:func:`read_manifest` for a results directory already open as ``dir_fd``."""
    try:
        fd = os.open("manifest.json", os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK, dir_fd=dir_fd)
    except (OSError, NotImplementedError):
        return None
    try:
        with os.fdopen(fd, "rb") as fh:
            if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                return None
            data = json.loads(fh.read().decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


#: The operations :func:`_as_the_owner` can ask of its child: this module's two writers, and the
#: judging that reads what the manifest names (hunt 2026-09-30, rp-u18).
_OWNER_OPS = ("write-review", "register-artifacts", "review")


def _as_the_owner(d: Path, op: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Run one L2 step in a child as the account that can write ``d``, or ``None`` to run it here.

    Root over a directory the agent can write does not write it itself (see
    :mod:`~spatialomicsgym.postanalysis.as_owner`): L1 and L3 already ran that way and the two L2
    writers did not, so root read the agent's ``manifest.json`` -- through any link -- and published
    what it found (hunt 2026-09-30, u18-postanalysis-1). The child's answer is the result; a child
    that fails is a write that did not happen, never a reason to write here after all.
    """
    from spatialomicsgym.postanalysis.as_owner import foreign_writer, run_module_as

    writer = foreign_writer(d)
    if writer is None:
        return None
    stderr = ""
    try:
        done = run_module_as(
            # Absolute, not resolved: the child starts in a scratch directory, and any link in the
            # path is for the child to meet, as the account it runs as.
            *writer,
            "spatialomicsgym.postanalysis.review",
            [op, str(d.absolute())],
            stdin_text=json.dumps(payload),
        )
        stderr = done.stderr or ""
        lines = [line for line in (done.stdout or "").splitlines() if line.strip()]
        answer = json.loads(lines[-1]) if done.returncode == 0 and lines else None
    except Exception as exc:
        stderr, answer = f"{type(exc).__name__}: {exc}", None
    if not isinstance(answer, dict):
        logger.debug("L2 %s (as uid %s) wrote nothing for %s: %s", op, writer[0], d, stderr.strip()[-400:])
        return {}
    return answer


def write_review(results_dir: str | Path, review: dict[str, Any], next_steps: list[dict[str, Any]]) -> bool:
    """Write the ``review`` and ``next_steps`` blocks back into the manifest, atomically.

    Every other key is preserved byte-for-byte in value, including keys this version of L2 has
    never heard of -- the contract requires v2 fields to survive a v1 consumer.

    Refuses, returning ``False``, for anything that leads under ``benchmarks/results/``. This is the
    one place every L2 write converges -- discovery, an explicit ``results_dir``, and the re-check
    after a follow-on round all end here -- so it is where the guarantee belongs. It was previously
    only on discovery, and only against the *spelling*: a path through
    ``benchmarks/manuscript/per_cell_results/multi_llm``, a symlink into the recorded tree, reached
    this function and rewrote a recorded experiment's manifest.

    The manifest's timestamp is carried across the rewrite. It is this system's only answer to "when
    was this run produced" -- ``find_runs`` sorts the portal by it, ``runs_since`` selects a turn's
    output by it, :func:`discover_results_dirs` decides membership of a turn from it -- and a review
    is metadata *about* a run, not a new one. Letting the rewrite restate it did not merely blur the
    order, it reversed it: the agent reviews a turn's directories newest-first, so the oldest was
    rewritten last and came out on top. ``chat_attachment`` asks for the single newest run since the
    turn began, and after a two-tool turn that is the run the turn was least about -- the figures
    under the answer belonged to the first tool while the follow-on step acted on the second.
    """
    delegated = _as_the_owner(Path(results_dir), "write-review", {"review": review, "next_steps": list(next_steps)})
    if delegated is not None:
        return delegated.get("written") is True

    def _apply(manifest: dict[str, Any]) -> bool:
        manifest["review"] = review
        manifest["next_steps"] = list(next_steps)
        return True

    return _rewrite_manifest(Path(results_dir), _apply, what="review")


def _rewrite_manifest(d: Path, apply: Callable[[dict[str, Any]], bool], *, what: str) -> bool:
    """Read the manifest, let ``apply`` change it in place, publish it by ``os.replace``.

    The write half of :func:`write_review`, extracted when a second L2 writer appeared
    (:func:`register_new_artifacts`). Its three guarantees are the reason it is shared rather than
    restated -- L2 restated the recorded-output guard once already and rewrote a recorded
    experiment's manifest through a symlink:

    * nothing under ``benchmarks/results/`` is written, following symlinks to decide;
    * the file is published atomically, from a staging name no other writer computes;
    * the run's timestamp is carried across the rewrite.

    ``apply`` returns whether the manifest actually changed. ``False`` means no write at all, so a
    caller with nothing to say does not restate the file -- and cannot report a write that did not
    happen. Every other key is preserved by construction: ``apply`` mutates the dict this function
    read, so keys this version of L2 has never heard of go back out untouched.
    """
    if leads_under_benchmark_results(d):
        logger.debug("refusing to write a %s into recorded experiment output: %s", what, d)
        return False
    # Read, judged and written through one descriptor. By path, the manifest was read from one
    # directory and written into whatever stood at ``d`` a moment later: as root over a ``d`` whose
    # parent the agent can rename, a link swapped in between the two took root's write -- the run's
    # manifest with the review merged in -- into the directory it named (hunt 2026-09-30, rp-u18).
    dir_fd = _open_results_dir(d)
    if dir_fd is None:
        return False
    try:
        from spatialomicsgym.postanalysis.as_owner import foreign_writer_of_fd

        if foreign_writer_of_fd(dir_fd) is not None:
            # The caller found ``d`` root's own and so writes it here; the directory now open is one
            # someone else can write, so it is not the one that was judged.
            logger.debug("refusing to write a %s as root into a directory another account can write: %s", what, d)
            return False
        return _rewrite_manifest_at(dir_fd, d, apply, what=what)
    finally:
        os.close(dir_fd)


def _rewrite_manifest_at(dir_fd: int, d: Path, apply: Callable[[dict[str, Any]], bool], *, what: str) -> bool:
    """:func:`_rewrite_manifest` for a results directory already open as ``dir_fd``."""
    manifest = _read_manifest_at(dir_fd)
    if manifest is None:
        return False
    if not apply(manifest):
        return False
    target = "manifest.json"
    tmp = staging_name("manifest.json", "l2")
    try:
        # Not followed: a link swapped in after the read is not the run whose time this carries
        # (hunt 2026-09-30, u18-postanalysis-1).
        was = os.stat(target, dir_fd=dir_fd, follow_symlinks=False)
    except OSError:
        was = None  # unreadable, so there is no time to preserve; the write below still decides
    try:
        with open_staging(tmp, dir_fd=dir_fd) as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        if was is not None:
            # Its own ``try``, and deliberately after the ``replace``: the review is on disk by this
            # point, so a filesystem that will not take a ``utime`` costs the ordering, not the
            # verdict. Reporting ``False`` here would send the caller round again.
            try:
                os.utime(target, ns=(was.st_atime_ns, was.st_mtime_ns), dir_fd=dir_fd, follow_symlinks=False)
            except (OSError, NotImplementedError) as exc:
                logger.debug("could not preserve the manifest timestamp on %s: %s", d / target, exc)
        return True
    except Exception as exc:
        logger.debug("could not write %s into %s: %s", what, d / target, exc)
        discard_staging(tmp, dir_fd=dir_fd)
        return False
    except BaseException:
        # After the ``Exception`` clause, and re-raising rather than returning: ``KeyboardInterrupt``
        # and ``SystemExit`` do not derive from ``Exception``, so they fall through to here, and a
        # Ctrl-C is not a failed review to report back -- it is the run ending. All this clause owes
        # is the staging file, which under a per-writer name nothing else will ever reclaim.
        discard_staging(tmp, dir_fd=dir_fd)
        raise


def register_new_artifacts(results_dir: str | Path) -> list[str]:
    """Declare any file in ``figures/`` or ``tables/`` that the manifest does not list yet.

    Returns the paths added, relative to the results dir, in the order they were declared.

    A follow-on round is a full agent turn pointed at this directory and told to write its new
    figures and tables into it (``next_step.build_followup_prompt``). It does, and until this
    function existed nothing registered them: ``write_manifest`` is L1's and runs before the round,
    and ``write_review`` writes only the ``review`` and ``next_steps`` blocks. The manifest's two
    lists are what the loop's own post-round re-check validates, what the portal card counts
    (``report.discover._summarize``), and what ``report.html`` draws -- so the round's work existed
    for a file browser and for nothing else.

    What is claimed about a discovered file is only what is known:

    * ``kind`` is ``""``. This package did not draw it and cannot say what it plots. An empty kind
      is the case L3 already renders (it appends "kind: x" to the caption only when there is one).
    * a table's ``rows`` is ``None``. Counting them means reading a file of unknown size, and L3
      already prints a row count only where one was declared.
    * the caption says where it came from, because "post-analysis drew this" would not be true.

    Placement is the whole rule for *what* gets declared: ``manifest.check_artifact_path`` accepts
    ``figures/``- and ``tables/``-rooted paths and nothing else, so those two directories are the
    only place a manifest entry can point at, and a file in one of them is an artifact by being
    there. Subdirectories are included -- a round may organise its own output. Dotfiles are not:
    that is the shape :func:`~spatialomicsgym.postanalysis.manifest.staging_name` gives an
    interrupted write, and a stranded staging file is not a result.
    """
    delegated = _as_the_owner(Path(results_dir), "register-artifacts", {})
    if delegated is not None:
        found = delegated.get("added")
        return [str(a) for a in found] if isinstance(found, list) else []

    added: list[str] = []

    def _apply(manifest: dict[str, Any]) -> bool:
        for kind, prefix in (("figures", "figures/"), ("tables", "tables/")):
            entries = manifest.get(kind)
            if not isinstance(entries, list):
                # Malformed enough that ``_manifest_checks`` will report it. Appending to it here
                # would replace that report with a different, quieter one.
                continue
            known = {str(e.get("path", "")) for e in entries if isinstance(e, dict)}
            for found in _undeclared_artifacts(Path(results_dir) / kind, prefix, known):
                entry: dict[str, Any] = {
                    "path": found,
                    "title": found.rsplit("/", 1)[-1],
                    "caption": FOLLOW_ON_CAPTION,
                    "kind": "",
                }
                if kind == "tables":
                    entry = {"path": found, "title": entry["title"], "rows": None}
                entries.append(entry)
                added.append(found)
        return bool(added)

    # Only what was really written. The return was ignored, so a refused or failed rewrite still
    # reported its figures as registered, and research/loop.py counts this list (hunt 2026-09-30).
    if not _rewrite_manifest(Path(results_dir), _apply, what="artifact registration"):
        return []
    return added


#: Said of every artifact :func:`register_new_artifacts` declares, and of nothing else. A reader
#: that finds a figure with no caption in a post-analysis report is looking at one L1 drew and
#: chose not to caption; this sentence is how the two are told apart on the page.
FOLLOW_ON_CAPTION = "Written by a follow-on analysis round; not drawn by the deterministic pass."


def _undeclared_artifacts(directory: Path, prefix: str, known: set[str]) -> list[str]:
    """Relative paths of the regular files under ``directory`` that ``known`` does not contain.

    Sorted, so a directory listed in whatever order the filesystem returns produces the same
    manifest twice running. Unreadable directories yield nothing: a rescan that cannot see is not
    evidence that there is nothing to see, and the caller's other artifacts still register.
    """
    try:
        candidates = sorted(p for p in directory.rglob("*") if p.is_file())
    except OSError as exc:
        logger.debug("could not scan %s for new artifacts: %s", directory, exc)
        return []

    found = []
    for path in candidates:
        try:
            rel = path.relative_to(directory).as_posix()
        except ValueError:  # pragma: no cover -- rglob cannot leave its own root
            continue
        if any(part.startswith(".") for part in rel.split("/")):
            continue
        if prefix + rel not in known:
            found.append(prefix + rel)
    return found


#: How many of a turn's results directories L2 will review. A turn that analyses more than this
#: many is not a normal turn; the cap keeps the review bounded without capping the analysis.
MAX_REVIEWED_PER_TURN = 12


def discover_results_dir(roots, since: float, max_depth: int = 4, *, exclude=()) -> Path | None:
    """Newest results dir holding a manifest written during this turn, or ``None``.

    L1 resolves ``results_dir`` from the caller, so the agent does not necessarily know where the
    manifest landed. ``since`` is the turn's start time: an older manifest belongs to a previous
    run and reviewing it would attribute someone else's result to this turn.

    ``benchmarks/results/`` is excluded structurally, not by convention.

    This is the *primary* directory -- the one the follow-on loop may act on, of which there is
    deliberately one. It is not the only one a turn can produce: see :func:`discover_results_dirs`.
    """
    found = discover_results_dirs(roots, since, max_depth, limit=1, exclude=exclude)
    return found[0] if found else None


def discover_results_dirs(
    roots, since: float, max_depth: int = 4, *, limit: int | None = None, exclude=()
) -> list[Path]:
    """Every results dir holding a manifest written during this turn, newest first.

    ``autorun.analyse_new_outputs`` writes one results directory *per output directory it finds*,
    and a turn that ran three tools produces three. L2 asked only for the newest, so the other two
    kept ``review: null`` and ``next_steps: []`` for good -- and because the agent renders
    ``report.html`` from the L2 outcome, they got no page either. In the portal those runs list
    with no verdict beside the one that has one, which reads as "this run was not checked" when
    what happened is that nothing ever offered to check it.

    The follow-on *action* loop still runs on one directory only; this is about the verdict and the
    page, which every analysed run should have.

    ``exclude`` -- directories never entered and never returned. On the portal these are every
    OTHER account's folder under the agent's outputs tree; a manifest another account's turn wrote
    a moment ago is not this turn's result, and reviewing it would write this turn's verdict into
    that account's folder.
    """
    cap = MAX_REVIEWED_PER_TURN if limit is None else max(0, int(limit))
    if cap <= 0:
        return []
    banned: set[Path] = set()
    for raw in exclude or ():
        try:
            banned.add(Path(str(raw)).resolve())
        except (OSError, ValueError, TypeError):
            continue
    found: list[tuple[float, Path]] = []
    for root in roots or ():
        try:
            base = Path(root).resolve()
        except Exception:
            continue
        if not base.is_dir():
            continue
        if banned and (base in banned or any(b in base.parents for b in banned)):
            continue
        for manifest in _bounded_glob(base, "manifest.json", max_depth):
            if banned and _manifest_inside(manifest, banned):
                continue
            # ``leads_``, not the spelling: the glob descends through
            # ``benchmarks/manuscript/per_cell_results/multi_llm``, whose *name* says "manuscript"
            # and whose destination is the recorded tree. The relative form is still checked
            # because a root given as an absolute path inside the recorded tree makes the absolute
            # spelling the only one that carries the two components.
            try:
                relative = manifest.relative_to(base)
            except ValueError:
                relative = manifest
            if leads_under_benchmark_results(manifest) or is_under_benchmark_results(relative):
                continue
            try:
                mtime = manifest.stat().st_mtime
            except OSError:
                continue
            if mtime < since:
                continue
            data = read_manifest(manifest.parent)
            # The shared predicate, not a local spelling of it. ``"tool_name" in data`` admitted any
            # ``manifest.json`` that merely named a tool, and a hit here is read by
            # ``_ensure_post_analysis_ran`` as "the model already called the engine" -- so a foreign
            # manifest in a tool's output directory returned the turn before autorun and cost it its
            # analysis, while autorun, asked directly, called the same directory not ours.
            if not is_our_manifest_payload(data):
                continue
            found.append((mtime, manifest.parent))
    # Two roots can reach one directory (the agent root and the portal root overlap on some
    # layouts), and reviewing it twice would write the verdict twice. Resolved paths, not spellings.
    ordered: list[Path] = []
    seen: set[str] = set()
    for _, directory in sorted(found, key=lambda item: -item[0]):
        try:
            key = str(directory.resolve())
        except OSError:
            key = str(directory)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(directory)
        if len(ordered) >= cap:
            break
    return ordered


# ----------------------------------------------------------------------------------------------
# discovery helpers
# ----------------------------------------------------------------------------------------------


def _manifest_inside(manifest: Path, banned: set[Path]) -> bool:
    try:
        resolved = manifest.resolve()
    except (OSError, ValueError):
        return False
    return any(b in resolved.parents for b in banned)


def _bounded_glob(base: Path, name: str, max_depth: int):
    """``rglob`` with a depth cap -- an agent data root can hold a very large tree."""
    for depth in range(max_depth + 1):
        pattern = "/".join(["*"] * depth + [name]) if depth else name
        try:
            yield from base.glob(pattern)
        except OSError:
            continue


# ----------------------------------------------------------------------------------------------
# manifest-level checks
# ----------------------------------------------------------------------------------------------


def _manifest_checks(manifest: dict[str, Any], results_dir: Path) -> list[_Check]:
    problems: list[str] = []
    for key in ("schema_version", "tool_name", "task_type", "status"):
        if key not in manifest:
            problems.append(f"missing required key {key!r}")

    # ``or ""`` and not ``.get(k, "")``: L1 writes an explicit ``"task_type": null`` whenever
    # detection could not settle it, so the key is present holding a null and the default never
    # applies -- ``str(None)`` then made this report `task_type 'None' is not one of the contract's
    # task types`, a contract violation invented out of a correctly recorded "unknown". Same bug,
    # same line, as the ``tool_name`` one already fixed in ``_prediction_checks`` below.
    #
    # Third instance of the same class, and this one had an authority to contradict. The design
    # contract: "``resolution`` is a valid registry task type with no post-analysis handler. Treat
    # it, and any other unlisted string, as ``status: "partial"`` with one warning -- not
    # ``"failed"``." L1 does exactly that and says so in a warning; L2 read the result back as a
    # *schema* violation, i.e. a malformed manifest. On the recorded scresolve run -- 0 findings, 0
    # tables, 0 figures -- that warn-level failure short-circuits ``_verdict`` before the branch
    # written for precisely this case, so the model was never told "nothing was verified about the
    # prediction's values" and was instead handed ``verify_flagged_result`` over our own enum.
    # A string no layer ever recorded still lands here; ``declined_for_no_handler`` is per task type
    # and anchored to L1's spelling of the sentence.
    task_type = str(manifest.get("task_type") or "")
    if task_type and task_type not in TASK_TYPES and not declined_for_no_handler(manifest, task_type):
        problems.append(f"task_type {task_type!r} is not one of the contract's task types")

    declared: list[tuple[str, str]] = []
    for kind in ("figures", "tables"):
        entries = manifest.get(kind)
        if entries is None:
            problems.append(f"{kind!r} must always be present (possibly empty)")
            continue
        if not isinstance(entries, list):
            problems.append(f"{kind!r} must be a list")
            continue
        for entry in entries:
            if not isinstance(entry, dict) or "path" not in entry:
                problems.append(f"{kind} entry without a 'path'")
                continue
            path = str(entry["path"])
            if os.path.isabs(path):
                problems.append(f"{kind} path {path!r} is absolute; the contract requires it relative to results_dir")
            declared.append((kind, path))

    missing = [p for _, p in declared if not (results_dir / p).exists()]

    checks = [
        _Check(
            "manifest_schema",
            not problems,
            "manifest conforms to contract v1" if not problems else "; ".join(problems[:6]),
        ),
        _Check(
            "declared_artifacts_exist",
            not missing,
            (
                f"all {len(declared)} declared artifact(s) present"
                if not missing
                else f"{len(missing)} declared artifact(s) missing on disk: {missing[:4]}"
            ),
        ),
    ]

    status = str(manifest.get("status", "")).lower()
    if status == "failed":
        checks.append(_Check("run_status", False, RUN_STATUS_FAILED_DETAIL, fatal=True))
    elif status == "partial":
        warns = [str(w) for w in (manifest.get("warnings") or [])]
        incomplete = failed_steps(manifest)
        checks.append(
            _Check(
                "run_status",
                True,
                f"status='partial'; {len(incomplete)} analysis step(s) did not complete, "
                f"{len(warns)} warning(s) recorded",
            )
        )
    else:
        checks.append(_Check("run_status", True, f"status={status or 'ok'!r}"))
    return checks


# ----------------------------------------------------------------------------------------------
# prediction-level checks
# ----------------------------------------------------------------------------------------------


def _prediction_checks(manifest: dict[str, Any]) -> list[_Check]:
    task_type = str(manifest.get("task_type") or "")
    # ``.get(k, default)`` returns the default only when the key is *absent*. L1 writes
    # ``"tool_name": null`` whenever the caller named no tool -- 70 of the 92 recorded manifests --
    # so the key is present holding a null and this became the string ``"None"``: the report read
    # "no registry profile for 'None'; slot not verifiable", blaming a tool by that name.
    # ``next_step.py:237-243`` already fixed this class for the follow-on prompt.
    tool_name = str(manifest.get("tool_name") or "")
    source = _resolve_source(manifest)

    checks: list[_Check] = []
    if source is None:
        # A figure-only run has no slot to miss. "no readable source output was declared" reads as
        # something lost, and on a visualization run it was quoted back to the model as the concern
        # to verify (live run 2026-10-01, S3).
        detail = (
            f"a {task_type} run publishes a figure, not a prediction, so there is no slot or value to check"
            if has_no_prediction_slot(manifest)
            else "no readable source output was declared, so neither the slot nor the values could be checked"
        )
        checks.append(_Check("published_output_slot", True, detail, examined=False))
    else:
        checks.extend((_slot_check(tool_name, source), _stale_check(source)))
        if task_type == "spatial_clustering":
            checks.extend(_label_from_input_check(manifest, source))
        frame = _read_table(source)
        if frame is not None:
            if task_type == "deconvolution":
                checks.extend(_deconvolution_checks(frame))
            elif task_type == "svg_detection":
                checks.extend(_svg_checks(frame))
            elif task_type == "spatial_clustering":
                checks.extend(_clustering_checks(frame, tool_name))
            else:
                checks.append(_generic_signal_free(frame))
    return _with_recorded_measurement(checks, manifest, source)


def has_no_prediction_slot(manifest: dict[str, Any]) -> bool:
    """True when the run's task type predicts nothing -- its result is the figure it drew.

    The registry is the authority: a profile with no ``authoritative_output``, ``prediction_key`` or
    ``prediction_type`` is one whose notes say "this tool predicts nothing -- it draws what another
    tool predicted" (``tool_output_registry.VIZ_NOTE``). A task type qualifies only when L1 declined
    it as outside the contract's vocabulary and every registered profile of it is of that kind, so
    ``visualization`` does, ``resolution`` and ``functional_enrichment`` (which publish a table or an
    ``.h5ad``) do not, and an alignment run keeps the runner and every check it already had. A type
    the registry has never heard of has no evidence either way and keeps every check.
    """
    task_type = str(manifest.get("task_type") or "")
    if not task_type or task_type in TASK_TYPES or not declined_for_no_handler(manifest, task_type):
        return False
    # A run that declared a prediction of its own keeps the value check, whatever it was typed as: a
    # clustering manifest merged and retyped "visualization" came back ok with nobody having read its
    # prediction (correctness review of S3, 2026-10-01).
    if manifest.get("source_outputs"):
        return False
    try:
        from spatialomicsgym.benchmarking.tool_output_registry import get_all_profiles

        profiles = [p for p in get_all_profiles().values() if p.task_type == task_type]
    except Exception:
        return False
    return bool(profiles) and not any(p.authoritative_output or p.prediction_key or p.prediction_type for p in profiles)


def _with_recorded_measurement(checks: list[_Check], manifest: dict[str, Any], source: Path | None) -> list[_Check]:
    """Fall back to L1's own measurement whenever L2 could not make one itself.

    L2 reads delimited text and nothing else, so an ``.h5ad`` prediction -- the standard
    ``spatial_clustering`` output -- used to reach the verdict with no value check at all. L1 *did*
    open that file: its ``signal_free`` finding is a measurement of the values, taken by the layer
    that could parse them. Reading it is a value check performed one level down, not a second
    opinion, and it is strictly better than the alternative of judging a prediction no one looked at.

    Nothing happens when L2 made its own measurement -- a readable table produces byte-identical
    checks, so a correct result is unaffected.
    """
    if any(c.examined and c.name in VALUE_CHECK_NAMES for c in checks):
        return checks
    recorded = _signal_free_from_findings(manifest, source)
    if recorded is None:
        return checks
    # Replaces the "nothing to check" placeholder rather than sitting beside it: two checks under
    # one name would leave L3 rendering a contradiction.
    return [c for c in checks if c.name != "signal_free"] + [recorded]


def _signal_free_from_findings(manifest: dict[str, Any], source: Path | None) -> _Check | None:
    """L1's ``signal_free`` finding as a check, or ``None`` when L1 recorded no such measurement."""
    findings = {
        str(f["key"]): f.get("value") for f in manifest.get("findings") or () if isinstance(f, dict) and "key" in f
    }
    flagged = _as_bool(findings.get("signal_free"))
    if flagged is None:
        return None

    where = Path(source).name if source is not None else "the prediction"
    scale = ", ".join(f"{k}={findings[k]!r}" for k in ("n_spots", "n_domains", "n_cell_types") if k in findings)
    measured = f"L1's own measurement of {where}" + (f" ({scale})" if scale else "")
    if not flagged:
        return _Check("signal_free", True, f"{measured} says the prediction varies; L2 could not re-read it itself")

    recorded = next((str(w) for w in manifest.get("warnings") or () if "signal-free" in str(w).lower()), "")
    return _Check(
        "signal_free",
        False,
        (
            f"{measured} says it carries no information, and L2 could not re-read the file to "
            f"check independently. {recorded or 'A prediction with no signal cannot be built on.'}"
        ),
        fatal=True,
    )


def _as_bool(value) -> bool | None:
    """``True``/``False`` for a value that clearly means one of them, else ``None``.

    ``None`` is the safe answer: it means "L1 recorded nothing usable", which leaves the review
    without a value check and therefore unable to return "ok" -- the direction that errs towards
    saying so rather than towards confidence.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "yes", "1"):
            return True
        if text in ("false", "no", "0"):
            return False
    return None


def _resolve_source(manifest: dict[str, Any]) -> Path | None:
    """The file the review reads values from.

    The manifest says *what was analysed*; L2 does not re-scan the disk to decide what exists. When
    ``source_outputs`` names a directory, the tool's own registry profile picks the file inside it,
    so the decoys stay decoys.
    """
    for raw in manifest.get("source_outputs") or ():
        p = Path(str(raw))
        if p.is_file():
            return p
        if p.is_dir():
            picked = _pick_from_dir(p, str(manifest.get("tool_name") or ""), str(manifest.get("task_type") or ""))
            if picked is not None:
                return picked
    return None


def _pick_from_dir(directory: Path, tool_name: str, task_type: str) -> Path | None:
    try:
        from spatialomicsgym.benchmarking.output_inspector import inspect_tool_output

        res = inspect_tool_output(tool_name=tool_name, task_type=task_type, output_dir=str(directory))
        if res.prediction_file:
            return Path(res.prediction_file)
    except Exception as exc:
        logger.debug("inspector could not pick a prediction in %s: %s", directory, exc)
    return None


def _slot_check(tool_name: str, source: Path) -> _Check:
    # Through sources.is_staged_input, not the inspector's raw predicate: inside an engine run the
    # judgement must be relative to the run's own directory, or the caller-chosen name of that
    # directory (``scanpy_on_converted/``, live) fails this check for every file in the run.
    from spatialomicsgym.postanalysis.detect import resolve_profile
    from spatialomicsgym.postanalysis.sources import conversion_evidence, is_staged_input

    converted = conversion_evidence(source)
    if converted is not None:
        # Before the directory sentence, which would name a folder -- ``outputs/`` -- that is not
        # a staging directory at all. What is wrong is the file: the input, re-encoded.
        return _Check(
            "published_output_slot",
            False,
            (
                f"{source.name} is the input re-encoded by a conversion utility, as {converted} "
                "says -- an input to the tool, never its prediction"
            ),
            fatal=True,
        )
    if is_staged_input(source):
        return _Check(
            "published_output_slot",
            False,
            (
                f"{source.name} lies in a staging/derived directory "
                f"({'/'.join(source.parts[-3:-1])}) -- that is an input to the tool or an artifact "
                "of a previous inspection, never its prediction"
            ),
            fatal=True,
        )

    if not tool_name:
        # Distinct from "the registry has never heard of this tool": no tool was named, so there was
        # nothing to look up. Saying it the other way sent the reader to the registry to add an
        # entry for a tool that does not exist.
        return _Check(
            "published_output_slot",
            True,
            f"the run names no tool, so there is no published slot to check {source.name} against",
            examined=False,
        )

    profile = resolve_profile(tool_name)
    if profile is None:
        return _Check(
            "published_output_slot",
            True,
            f"no registry profile for {tool_name!r}; slot not verifiable",
            examined=False,
        )

    patterns = profile.authoritative_output or [o.filename_pattern for o in profile.outputs if o.role == "prediction"]
    patterns = _as_list(patterns)
    if not patterns:
        return _Check(
            "published_output_slot",
            True,
            f"{tool_name!r} declares no authoritative output",
            examined=False,
        )

    flat: list[str] = []
    for p in patterns:
        flat.extend(_as_list(p))
    if any(fnmatch.fnmatch(source.name, pat) for pat in flat):
        return _Check("published_output_slot", True, f"{source.name} matches the slot {tool_name!r} publishes")
    return _Check(
        "published_output_slot",
        False,
        (
            f"{source.name} is not an output {tool_name!r} publishes as its prediction "
            f"(expected one of {flat[:4]}). {profile.notes.strip()[:200]}".strip()
        ),
        fatal=True,
    )


def _label_from_input_check(manifest: dict[str, Any], source: Path) -> list[_Check]:
    """Fail when the column L1 published as the domains has a name the converted input already had.

    Asked only where it can be answered: ``source`` descends from a conversion utility's output, so
    it carries the names of the obs columns that conversion wrote -- every one of them the input's
    own or a QC count computed from it (:mod:`spatialomicsgym.tool.conversion_record`). Nothing is
    emitted otherwise, so a review of any other run is exactly what it was.

    Warn-level, not fatal: the labels are real data and the tool may have written its own column
    beside them. What they are not is the tool's clustering, and a count of distinct values in a
    column that came with the input is no evidence one was produced -- which is what the E-03
    qc_01 reviews quoted, under ``ok``, for the input's ``cell_type_annot``. The record keeps
    names, not values, so a tool that wrote its clustering over a column of the same name fails
    this as well; the detail says "unless a tool wrote over it" rather than claim to know which.
    """
    from spatialomicsgym.postanalysis.sources import conversion_record

    findings = {
        str(f["key"]): f.get("value") for f in manifest.get("findings") or () if isinstance(f, dict) and "key" in f
    }
    key = findings.get("cluster_key")
    if key is None or not str(key):
        return []
    record = conversion_record(source)
    if record is None or not record.obs_columns:
        return []
    made_by = record.producer or "a conversion utility"
    if str(key) in record.obs_columns:
        return [
            _Check(
                "label_from_input",
                False,
                (
                    f"{key!r} is the name of a column the input arrived with: {made_by} wrote an obs "
                    f"column of that name into the file {source.name} descends from, before any tool "
                    "ran. Unless a tool wrote over it, its labels are the input's own annotation and "
                    "not a clustering this run produced"
                ),
            )
        ]
    return [
        _Check(
            "label_from_input",
            True,
            f"{key!r} is not among the {len(record.obs_columns)} obs column(s) {made_by} wrote from the input",
        )
    ]


def _stale_check(source: Path) -> _Check:
    from spatialomicsgym.benchmarking.output_inspector import _stale_file_warning

    try:
        siblings = [f for f in source.parent.iterdir() if f.is_file()]
    except OSError:
        return _Check("stale_prediction", True, "sibling mtimes unavailable", examined=False)
    warning = _stale_file_warning(source, siblings)
    if warning:
        return _Check("stale_prediction", False, warning)
    return _Check("stale_prediction", True, f"{source.name} is no older than its siblings")


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


# ----------------------------------------------------------------------------------------------
# table reading
# ----------------------------------------------------------------------------------------------


def _read_table(path: Path):
    """Read a delimited prediction table exactly as L1 read it, or ``None`` when it is not one.

    Delegates to ``tables.read_table`` -- the reader the task runners use -- rather than reading the
    file a second way. ``index_col=0`` was that second way, and it made L2 review a *different
    table* than L1 analysed: it consumed whichever column happened to be written first, whether or
    not that column was an identifier. On ``spatialde_results.csv`` (written ``index=False``, ``FSV``
    first) it ate ``FSV`` -- the effect size the standardizer ranks on -- leaving float statistics
    standing in for gene names and ``LLR`` validated in its place; on a proportions CSV with no
    identifier column it dropped a whole cell type, which pushed the row sums below 1 and turned a
    normalised matrix into "abundances rather than proportions".

    ``tables.read_table`` sniffs the separator through the inspector's own ``_tabular_sep`` (STRIDE
    publishes a tab-separated ``.txt``; reading it with commas collapses it into one column) and
    puts an identifier column on the index only when there is one.
    """
    from spatialomicsgym.postanalysis import tables

    if not tables.is_tabular(path):
        return None
    try:
        return tables.read_table(path)
    except Exception as exc:
        logger.debug("could not read %s as a table: %s", path, exc)
        return None


def _numeric(frame):
    return frame.select_dtypes(include="number")


def _finite_values(numeric):
    import numpy as np

    arr = numeric.to_numpy(dtype="float64", na_value=np.nan)
    return arr[np.isfinite(arr)]


# ----------------------------------------------------------------------------------------------
# deconvolution
# ----------------------------------------------------------------------------------------------


def _deconvolution_checks(frame) -> list[_Check]:
    import numpy as np

    from spatialomicsgym.postanalysis import tables

    # ``_numeric`` alone is every numeric column, and a deconvolution tool writing the spot
    # coordinates it was given into the same CSV as the composition it produced is ordinary. Reading
    # them as cell types is not: on the recorded ``Slice, spotName, x, y, CD14+ Monocytes, ...``
    # fixture it named ``x`` a cell type, then summed pixel coordinates into the row totals and
    # reported "neither axis sums to 1 (median row sum 10821.500); abundances rather than
    # proportions" about a table whose rows sum to 1.0 to fourteen decimal places. L1 solved this --
    # ``celltype_columns``'s docstring names the same median -- so the filter is imported, not
    # restated. It is scoped to this branch: it also strips cluster/QC names, which is right for a
    # composition and would be wrong for the other task types that share ``_numeric``.
    numeric = _numeric(tables.celltype_columns(frame))
    # Asked of the table as read, before any column is filtered: the key is usually ``leiden`` or
    # ``cluster``, which ``celltype_columns`` drops as a label -- correctly, and it is still the column
    # that says what the rows are.
    groups = _group_table_check(frame)
    if numeric.shape[1] < 1 or numeric.shape[0] < 1:
        return [_Check("signal_free", True, "no numeric proportions to check", examined=False), groups]

    cols = [str(c) for c in numeric.columns]
    positional = tables.topic_like_columns(cols)
    # Which names the predicate flags on their own. Asked one at a time it is exactly its word-list
    # half, because the shape half needs at least two names to see a counter -- so this separates
    # "the label is recognisably a latent index" from "the set has the shape of a generated axis".
    by_name = [c for c in cols if tables.topic_like_columns([c])]
    # Fatal only for the first. The shape half is deliberately wide -- ``topic_like_columns``'s own
    # docstring accepts ``Layer1..Layer6``, a real cortical panel, as its price -- and L1 can afford
    # that because it only warns and degrades the run to ``partial``. Here a failing fatal check
    # means ``unusable``, so the same net cannot cost the same: shape alone fails non-fatally, which
    # ``_verdict`` reads as ``suspicious``, which is the "warning asking for confirmation, not a
    # lost result" L1's docstring promises. Declaring a correct panel unusable is R80/K.
    recognised = bool(cols) and len(by_name) / len(cols) > 0.5
    topic = _Check(
        "topic_matrix_masquerade",
        not positional,
        (
            # What was tested, not what a reader would like it to mean. Passing means L1's predicate
            # found no latent axis: no bare integer, no ``topic_3``, no stem-plus-counter. It has
            # never had any way to tell a cell type from a barcode, and on this repo's own
            # ``spotclean_cleaned_counts.csv`` -- whose columns are Visium barcodes -- it affirms
            # them, which is exactly what the inspector's own barcode guard exists to reject. That
            # is a different question and is left to the guard that asks it.
            f"no column name is a latent index, a topic id or a stem and a counter (e.g. {cols[:3]})"
            if not positional
            # The reason has to be true of what is failing. "This is a topic / factor / archetype
            # matrix" was false of an attention weight and of a match index, and condemning a result
            # with a false sentence is the R80/K defect. Each branch says only what it established.
            else (
                f"columns are positional labels, not cell types (e.g. {by_name[:3]}) -- a latent "
                "index, an archetype or a topic id; map them to cell types before reading these as "
                "proportions"
            )
            if recognised
            else (
                f"columns look positional rather than annotated (e.g. {cols[:3]}) -- one shared "
                "stem and a counter, the shape a generated axis has rather than a named panel; "
                "confirm these are cell types, or map them to cell types, before reading these as "
                "proportions"
            )
        ),
        # Only ever read for a *failing* check, so a passing one keeps the True it always carried.
        fatal=recognised or not positional,
    )

    values = _finite_values(numeric)
    n_rows = int(numeric.shape[0])
    # Whether the rule could run, then what it found -- not one expression standing in for both. The
    # single ``else`` used to answer "values are continuous, as proportions must be" in three cases
    # where nothing had been measured: a table below the row floor (the recorded istar, stride and
    # celloscope deconvolutions are 3, 5 and 9 rows), a table whose values are all NaN, and a table
    # of all zeros -- which is 0/1-valued, just not informatively so. All three are in
    # ``VALUE_CHECK_NAMES``, so the untested affirmative is also what paid for a verdict of "ok".
    if not values.size:
        marker = _Check(
            "marker_table_masquerade",
            True,
            "no finite value in the matrix, so nothing could be checked",
            examined=False,
        )
    elif n_rows <= _MIN_ROWS_FOR_BINARY:
        marker = _Check(
            "marker_table_masquerade",
            True,
            f"{n_rows} row(s) is too few to tell an indicator table from a coarse composition, so this was not checked",
            examined=False,
        )
    elif int(np.unique(values).size) <= 1:
        # The third case ``:690-695`` names and the one the code did not implement. An all-zero
        # matrix fails ``values.max() > 0`` below and fell through to the ``else``, which printed
        # "values take more than the two states 0 and 1" about a matrix with exactly one state --
        # directly above ``signal_free`` reporting that every value is 0.0. One state is not a
        # measurement of how many states there are.
        marker = _Check(
            "marker_table_masquerade",
            True,
            f"every finite value in the matrix is {float(values[0]):g}, so there is nothing to tell an "
            "indicator table from a composition",
            examined=False,
        )
    elif bool(np.isin(values, (0.0, 1.0)).all()) and values.max() > 0:
        marker = _Check(
            "marker_table_masquerade",
            False,
            (
                f"every one of {values.size} values is 0 or 1 -- this is a marker / indicator table, "
                "not a proportions matrix (its columns being real cell-type names proves nothing)"
            ),
            fatal=True,
        )
    else:
        marker = _Check("marker_table_masquerade", True, "values take more than the two states 0 and 1")

    row_sums = numeric.sum(axis=1).to_numpy(dtype="float64")
    col_sums = numeric.sum(axis=0).to_numpy(dtype="float64")
    if not values.size:
        # ``sum`` skips NaN, so an all-NaN matrix sums to 0.0 along both axes and this check used to
        # answer "neither axis sums to 1 (median row sum 0.000); abundances rather than proportions"
        # -- a number derived from no number, and the last examined value check standing between an
        # unreadable prediction and a verdict of "ok".
        return [
            topic,
            marker,
            _Check(
                "proportion_orientation",
                True,
                "no finite value in the matrix, so neither axis could be summed",
                examined=False,
            ),
            _degenerate_matrix_check(numeric, rows_are_compositions=True),
            groups,
        ]
    n_rows, n_cols = int(numeric.shape[0]), int(numeric.shape[1])
    # A vector that sums to 1 is a proportion vector, and a proportion vector runs over cell types.
    # So "the rows sum to 1" nominates the COLUMNS as the cell types and "the columns sum to 1"
    # nominates the ROWS -- and a nomination naming thousands of cell types is not a reading of the
    # matrix, it is the tool having normalised over the other thing. Tangram writes 4035 spots x 44
    # cell types with each column summing to 1 (it normalises each cell type's mapping across the
    # spots); without this gate that read as "the COLUMNS sum to 1, not the rows" and was reported
    # *fatal* on a correctly oriented file. The predicate is shared with
    # ``tables.orient_proportions`` and ``tasks/deconvolution``, so the three cannot drift -- and it
    # asks whether the COUNT is plausible, not merely whether the axis is the longer one: a 9-region
    # x 20-cell-type mean composition is an ordinary table, and rejecting it here flipped its
    # verdict to ``suspicious`` on a correct result.
    rows_sum_to_one, cols_sum_to_one = _median_is_one(row_sums), _median_is_one(col_sums)
    why_not_rows = why_not_cell_types(n_cols, n_rows, numeric.columns)
    why_not_cols = why_not_cell_types(n_rows, n_cols, numeric.index)
    rows_are_props = rows_sum_to_one and why_not_rows is None
    cols_are_props = cols_sum_to_one and why_not_cols is None
    sums_over_a_non_celltype_axis = (rows_sum_to_one and not rows_are_props) or (cols_sum_to_one and not cols_are_props)
    # The sums decide this. The aspect ratio used to be ANDed in beside them, and it made the same
    # column-normalised matrix fatal at 5 x 11 and a pass at 5 x 10, where it read "neither axis
    # sums to 1 ... abundances" about a table whose columns sum to 1.000 -- a false sentence, and a
    # transposed matrix sent on to L3 as a correct deconvolution. Rows-as-cell-types is the recorded
    # stdgcn shape and a deconvolution over a small region is ordinary, so narrow is not innocent.
    # One numeric column is a per-cell-type summary, not a matrix with an orientation to get wrong:
    # ``redeconve_summary.csv`` is five cell types and one ``mean_proportion`` column that sums to 1.
    # The old ratio excluded it by arithmetic (1 > 2*rows is never true); say so on purpose instead.
    transposed = cols_are_props and not rows_are_props and numeric.shape[1] > 1
    if rows_are_props and cols_are_props:
        # Both axes normalise, which needs a square matrix: the numbers cannot say which axis is
        # which, so the shape does. This is the case the ratio was always for.
        transposed = numeric.shape[1] > numeric.shape[0] * 2
    if transposed:
        orientation = _Check(
            "proportion_orientation",
            False,
            (
                # What was measured, then the usual cause -- not the cause asserted as the
                # measurement. The recorded scdot coupling normalises per column too, and its rows
                # are Visium barcodes: saying outright "rows are cell types" would be a confident
                # sentence its own row labels contradict. Either way it cannot be read per spot.
                f"matrix is {numeric.shape[0]} x {numeric.shape[1]} and it is the COLUMNS that sum to 1, "
                f"not the rows (median row sum {np.median(row_sums):.3f}) -- read as spot composition "
                "the rows do not normalise; the usual cause is a transposed table, cell types down "
                "the rows and spots across the columns"
            ),
            fatal=True,
        )
    elif rows_are_props:
        orientation = _Check(
            "proportion_orientation",
            True,
            f"rows sum to 1 (median {np.median(row_sums):.3f}), so rows are spots -- correctly oriented",
        )
    elif sums_over_a_non_celltype_axis:
        # Reported, not passed over in silence: something IS normalised, and the ``else`` below
        # would have said "neither axis sums to 1" about a matrix where one of them sums to 1.000.
        # Not fatal and not an examined pass -- the numbers genuinely do not settle the orientation
        # here, they only rule out the reading that would have condemned the file.
        if cols_sum_to_one and not cols_are_props:
            which, n_summed, n_other, reason = "columns", n_cols, n_rows, why_not_cols
        else:
            which, n_summed, n_other, reason = "rows", n_rows, n_cols, why_not_rows
        orientation = _Check(
            "proportion_orientation",
            True,
            (
                f"matrix is {n_rows} x {n_cols} and the {which} sum to 1, but that would need "
                f"{n_other} cell types across {n_summed} spots and {reason} -- so the sum runs over "
                "something other than the cell types (a per-cell-type normalisation across spots, as "
                "Tangram writes, or an axis that is not cell types at all), it is not a per-spot "
                "composition, and the orientation is not decidable from the numbers"
            ),
            examined=False,
        )
    elif not np.any(values):
        # Same class as the all-NaN guard at the top of this block, and it was missed because an
        # all-zero matrix has ``values.size > 0``: ``sum`` gives 0.0 along both axes, ``_median_is_one``
        # is False for both, and the matrix landed in the ``else`` below described as "abundances
        # rather than proportions" -- an *examined* pass. Above the row floor ``signal_free`` catches
        # it, but the recorded istar, stride and celloscope deconvolutions are 3, 5 and 9 rows, where
        # the degeneracy rules do not run at all: there this sentence was the entire evidence for a
        # verdict of "ok" on a prediction of literally all zeros. Zero is not an abundance scale.
        orientation = _Check(
            "proportion_orientation",
            True,
            "every value in the matrix is 0, so neither axis sums to anything and the orientation could not be decided",
            examined=False,
        )
    else:
        orientation = _Check(
            "proportion_orientation",
            True,
            (
                f"neither axis sums to 1 (median row sum {np.median(row_sums):.3f}); "
                "abundances rather than proportions, orientation not decidable from the numbers"
            ),
            # The fourth way of declining, and the one that used to claim it had decided something.
            # In this ladder ``examined`` does not track whether bytes were read -- the branch above
            # sums both axes and still declares itself unexamined -- it tracks whether the check
            # reached a conclusion about the thing it is *named* for. This one ends in the same
            # words as the other three and was the only one counted as evidence, so ``_verdict``
            # (:1454) put it at the head of the reasons list on the recorded spacet and tangram
            # runs: "orientation not decidable from the numbers", offered as why the result can be
            # trusted. On spacet it is also the sentence that anchors the raw reading -- its row
            # sums are over the 35 rows as written, beside an L1 finding of "Spots deconvolved:
            # 200" for the same 35 x 200 file.
            #
            # Only the flag moves. "Abundances rather than proportions" is a real observation and
            # keeps its wording; it is a statement about the values, which ``marker_table_masquerade``
            # and ``signal_free`` are the checks for, and both stay examined -- so the value check
            # that pays for a verdict of ``ok`` (:1439) is still there and no verdict moves.
            examined=False,
        )

    # ``rows_are_compositions``: this is the deconvolution path, so the columns are cell types and a
    # row is one spot's composition over them. That is what licenses the flat-profile rule, and it
    # is knowledge no other caller of the check has.
    return [topic, marker, orientation, _degenerate_matrix_check(numeric, rows_are_compositions=True), groups]


def _group_table_check(frame) -> _Check:
    """Are the rows spots at all? A table whose rows are clusters is a summary over groups.

    Every other check here takes the rows to be spots and asks whether the columns are cell types --
    and on a cluster x donor cross-tabulation each of them passes: the donors are not positional, the
    counts are not 0/1, the rows sum to cluster sizes or to 1. The recorded manifests of the
    ``stcoscientist-v8p9`` and ``-L1`` trials reviewed exactly that table ``ok``, with a donor as the
    most abundant cell type. L1 no longer types such a table ``deconvolution``; this asks the same two
    questions of whatever manifest reaches L2 saying it is one.

    Fatal when a cluster or domain column names every row once, alone or with one other label column
    (``tables.group_key``), like the marker-table check: the numbers may be a correct summary of
    something, but as a deconvolution they are wrong -- each "spot" is a cluster and each "cell type"
    is whatever was tallied per cluster. A warning when nothing names the rows and there are too few
    of them to be spots (``tables.spot_evidence``): that is what the same table looks like with its
    key written as the unnamed index, and it is also what a tiny genuine result looks like, so it is
    not proof.

    The detail says what was looked at, and only that. The one it replaced said "nothing says they
    are groups rather than spots" of every table the key rule missed -- including a recorded
    ``cluster_mbkm_harmony`` x timepoint table whose key said exactly that.
    """
    from spatialomicsgym.postanalysis import tables

    key = tables.group_key(frame)
    if key is not None:
        named = (
            f"value of {key[0]!r}" if len(key) == 1 else "pair of " + " and ".join(f"{c!r}" for c in key) + " values"
        )
        return _Check(
            "group_table_masquerade",
            False,
            (
                f"every one of the {len(frame)} rows is a different {named} -- each row is a group, "
                "not a spot, so this table summarises groups (the recorded case is a cluster x donor "
                "composition) and cannot be a per-spot deconvolution"
            ),
            fatal=True,
        )
    evidence = tables.spot_evidence(frame)
    if evidence is None:
        return _Check(
            "group_table_masquerade",
            False,
            (
                f"the {len(frame)} rows are numbered rather than named and fewer than "
                f"{tables.MIN_UNNAMED_SPOTS}, so nothing in the table says they are spots -- a table over "
                "clusters written with its key as the unnamed index looks exactly like this"
            ),
        )
    return _Check(
        "group_table_masquerade",
        True,
        f"no cluster, domain or group column names each row once, alone or with one other label column, and {evidence}",
    )


def _median_is_one(sums) -> bool:
    import numpy as np

    finite = sums[np.isfinite(sums)]
    return bool(finite.size) and bool(np.isclose(np.median(finite), 1.0, atol=0.02))


def _degenerate_matrix_check(numeric, *, rows_are_compositions: bool = False) -> _Check:
    """Is there a prediction in this matrix at all?

    ``rows_are_compositions`` says the caller knows each row to be one composition over
    commensurable columns -- which is what the deconvolution routing establishes and nothing else
    does. Two of the three rules below hold for any numeric table; the third does not, and it is the
    only one that can be *contradicted* by a correct result rather than merely be uninformative.
    See the flat-profile rule for what that cost.

    The default is the safe one, so a caller that does not know what its columns are gets the two
    rules that cannot be wrong about them.
    """
    import numpy as np

    #: What a row is, in the sentences below. "spots" is a claim about the data, and it is false of
    #: an alignment table (matched pairs), an SVG table (genes) and a communication table (pathways)
    #: -- the three kinds that reach here through ``_generic_signal_free``.
    unit = "spots" if rows_are_compositions else "rows"

    values = _finite_values(numeric)
    n_distinct = int(np.unique(values).size) if values.size else 0
    if not values.size:
        # "0 distinct values across 40 rows" reads as a measurement of a flat prediction. There was
        # no prediction to measure: every cell is NaN.
        return _Check(
            "signal_free", True, "no finite value in the matrix, so degeneracy could not be checked", examined=False
        )
    if numeric.shape[0] > _MIN_ROWS_TO_JUDGE and n_distinct <= 1:
        # ``float(...)``, not ``{!r}``: numpy renders its own scalars as ``np.float64(0.0)`` under
        # ``repr``, and this string is read by a scientist, not by an interpreter.
        only = float(values[0]) if values.size else float("nan")
        return _Check(
            "signal_free",
            False,
            (
                f"every one of {values.size} values across {numeric.shape[0]} {unit} is {only:g} -- "
                "the prediction carries no signal; any metric computed against it is an artefact of "
                "the comparison, not a measurement"
            ),
            fatal=True,
        )
    if numeric.shape[0] > _MIN_ROWS_TO_JUDGE:
        # Constancy *across spots*. The two rules around this one both miss it: a matrix where every
        # spot carries the identical composition -- 200 rows of [0.40, 0.30, 0.15, 0.10, 0.05] --
        # has five distinct values globally and a perfectly non-flat profile within each spot, so it
        # passed as "5 distinct values across 200 rows". Variation between spots is the entire
        # content of a deconvolution; without it there is nothing to map, rank or correlate.
        spread = numeric.std(axis=0).to_numpy(dtype="float64")
        spread = spread[np.isfinite(spread)]
        if spread.size and float(spread.max()) < 1e-9:
            # "no spatial content" is the deconvolution reading of it. Generically the claim is the
            # weaker and still-sufficient one: no column of this table distinguishes one row from
            # another, so there is nothing in it to rank, map or correlate whatever the rows are.
            tail = (
                "so the prediction has no spatial content and any metric against it is an artefact"
                if rows_are_compositions
                else "so the prediction distinguishes nothing and any metric against it is an artefact"
            )
            return _Check(
                "signal_free",
                False,
                (
                    f"every one of {numeric.shape[0]} {unit} carries the identical profile across all "
                    f"{numeric.shape[1]} column(s) -- nothing varies from {unit[:-1]} to {unit[:-1]}, {tail}"
                ),
                fatal=True,
            )
    if rows_are_compositions and numeric.shape[0] > _MIN_ROWS_TO_JUDGE and numeric.shape[1] > 1:
        # Same rule as workflow_gates.post_execution_gate: >90% of spots with a flat profile.
        #
        # Gated on the caller, because unlike the two rules above this one is not a statement about
        # information content -- it is a statement about *compositions*, and it needs the columns to
        # be commensurable parts of one whole to mean anything. Ungated it reached every other task
        # type through ``_generic_signal_free``, and slat's ``slat_matching.csv`` -- 200 rows of
        # ``slice1_idx,slice2_idx`` recording a perfect 1:1 pairing between two sections -- was
        # declared ``unusable`` because a perfect matching has equal indices on every row, so 100%
        # of rows are "flat". The best result an alignment can produce read as no result at all,
        # under a sentence naming cell types and a deconvolution the run never attempted.
        #
        # Not gated on "do the rows sum to 1": spaotsc's transport plan is the one recorded run this
        # rule legitimately catches, and its rows sum to 0.0026. Nor can the two rules above cover
        # it -- 59,700 distinct values, and a column std of 1.7e-6 against their 1e-9 threshold. So
        # the gate has to be the one piece of knowledge that actually licenses the question.
        flat = float((numeric.std(axis=1).to_numpy(dtype="float64") < 1e-6).mean())
        if flat > 0.9:
            return _Check(
                "signal_free",
                False,
                f"{flat:.0%} of spots have identical proportions across all cell types -- no deconvolution happened",
                fatal=True,
            )
    if numeric.shape[0] <= _MIN_ROWS_TO_JUDGE:
        # None of the three rules above ran. "5 distinct values across 3 rows" is a true sentence and
        # not a finding: at three rows a collapsed prediction and a real one are indistinguishable.
        return _Check(
            "signal_free",
            True,
            f"{numeric.shape[0]} row(s) is too few for the degeneracy rules to run, so this was not checked",
            examined=False,
        )
    # "rows", not ``unit``, and deliberately inconsistent with every branch above. This is the one
    # sentence published by a *passing* check, so it is the one a reader is most likely to take at
    # face value -- and ``rows_are_compositions`` is what the deconvolution routing asserts, not
    # what the table is. R79/BI recorded two alignment outputs typed as deconvolutions; on those,
    # ``unit`` is "spots" over rows that are matched pairs. Every branch above earns the stronger
    # noun by having found something wrong, and states it beside the evidence. This one has found
    # nothing, so it claims nothing: "rows" is true whatever the rows turn out to be.
    return _Check("signal_free", True, f"{n_distinct} distinct values across {numeric.shape[0]} rows")


# ----------------------------------------------------------------------------------------------
# SVG detection
# ----------------------------------------------------------------------------------------------


def _svg_checks(frame) -> list[_Check]:
    import pandas as pd

    from spatialomicsgym.benchmarking.output_standardizer import (
        _SVG_PVALUE_COLUMNS,
        _SVG_SCORE_COLUMNS,
        _pick_significance_column,
    )

    pvalue_names = frozenset(_SVG_PVALUE_COLUMNS)
    numeric = _numeric(frame)
    by_lower = {str(c).lower(): c for c in reversed(list(frame.columns))}
    score_col = next((by_lower[n] for n in _SVG_SCORE_COLUMNS if n in by_lower), None)

    checks: list[_Check] = []
    n_rows = int(frame.shape[0])

    if score_col is not None:
        # By name and then coerced, not gated on dtype. One non-numeric cell -- a `-` placeholder, a
        # censored `<0.001`, a decimal comma -- makes the whole column read back as object, and the
        # dtype gate that used to stand here sent the file to the `else` branch, whose `candidates`
        # list is *also* built from `numeric.columns`. So a recognised column went invisible twice
        # and the check answered "tool reports no per-gene ranking statistic" about a file whose
        # column is named `fsv`. The standardizer picks the score column the same way this does and
        # applies no dtype filter (output_standardizer.py:479-481), so it will find that column;
        # denying it exists was wrong about the one thing this check predicts. Same fix as the
        # p-value branch: ask the values, not the dtype.
        scores = pd.to_numeric(frame[score_col], errors="coerce").dropna()
        distinct = int(scores.nunique())
        if scores.empty:
            # A column that is present and NaN in every row is not "every gene ties" -- nothing was
            # ranked at all. Saying it the old way meant reading `scores.iloc[0]` off an empty
            # Series, and the IndexError took the whole review down with it.
            checks.append(
                _Check(
                    "signal_free",
                    True,
                    f"{score_col!r} is present but holds no numeric value in any row, so the ranking "
                    "could not be checked",
                    examined=False,
                )
            )
        elif n_rows > 10 and distinct <= 1:
            # ``distinct`` is counted after ``dropna()`` and ``n_rows`` is the whole frame, so
            # "all 500 genes share one value" was printed for a table with one scored gene and 499
            # blanks. Nothing tied -- the tool scored almost nothing -- and that is a different
            # repair. Count the scored genes, and name the unscored ones when there are any.
            unscored = n_rows - int(scores.size)
            missing = f"; {unscored} of the {n_rows} rows carry no value at all" if unscored else ""
            checks.append(
                _Check(
                    "signal_free",
                    False,
                    (
                        f"all {int(scores.size)} scored genes share one {score_col!r} value "
                        f"({float(scores.iloc[0]):g}) -- the tool produced no ranking, so "
                        f"every gene ties and recall against any truth set is 1.0 by construction{missing}"
                    ),
                    fatal=True,
                )
            )
        elif n_rows <= _MIN_ROWS_TO_JUDGE:
            checks.append(
                _Check(
                    "signal_free",
                    True,
                    f"{score_col!r} takes {distinct} distinct value(s) over {n_rows} gene(s) -- too few "
                    "genes for the degeneracy rule to run, so this is not evidence of a ranking",
                    examined=False,
                )
            )
        else:
            checks.append(_Check("signal_free", True, f"{score_col!r} takes {distinct} distinct values over {n_rows}"))
        checks.append(_Check("score_column_retained", True, f"{score_col!r} is a name the standardizer reads"))
    else:
        checks.append(_generic_signal_free(frame))
        candidates = [
            str(c)
            for c in numeric.columns
            if str(c).lower() not in pvalue_names
            and not str(c).lower().startswith("unnamed")
            and any(h in str(c).lower() for h in _SCORE_NAME_HINTS)
        ]
        if candidates:
            checks.append(
                _Check(
                    "score_column_retained",
                    False,
                    (
                        f"the tool reports a ranking statistic in {candidates!r}, but no column name here is in "
                        "output_standardizer._SVG_SCORE_COLUMNS, so standardization writes score=0.0/NaN for "
                        "every gene and the ranking is lost silently"
                    ),
                )
            )
        else:
            checks.append(_Check("score_column_retained", True, "tool reports no per-gene ranking statistic"))

    # Which column, chosen the way selection chooses it. The old scan walked the file's own column
    # order over an unordered name set, and every recorded SVG tool that publishes both a raw and an
    # adjusted p-value writes the raw one first, so it validated the raw column on 6 of the 12
    # recorded runs. On somde that meant reporting "'pval' takes 497 distinct values over 500 genes"
    # and a verdict of "ok", while 'qval' -- the column `significant` is derived from -- held the
    # single value 0.898 in all 500 rows. `_pick_significance_column` is the picker the standardizer
    # and the inspector already share; importing it is what keeps the three from drifting again.
    pval_col = _pick_significance_column(frame)
    if pval_col is None:
        # It skips a column that is present but unpopulated, which is a third state: squidpy writes
        # `pval_norm_fdr_bh` as NaN in every row whether or not it corrected anything. That used to
        # be picked (all-NaN is still numeric dtype), measured at 0 distinct values, and failed as
        # "a yes/no flag written into a p-value column". Nothing was measured, so say that.
        empty = [str(c) for c in frame.columns if str(c).lower() in pvalue_names]
        if empty:
            checks.append(
                _Check(
                    "pvalue_is_measured",
                    True,
                    f"{empty!r} is present but holds no numeric value in any row, so nothing could be checked",
                    examined=False,
                )
            )
        else:
            # ``examined=False`` for the same reason as the sibling one branch up, which has always
            # had it: no p-value was looked at. Without the flag this is the only vacuous branch in
            # the module that counts as a value check, and since ``pvalue_is_measured`` is in
            # ``VALUE_CHECK_NAMES`` it alone paid for "ok" -- on a six-gene table whose scores are
            # constant and whose manifest already carries L1's signal-free warning. It also
            # short-circuited ``_with_recorded_measurement``, so L1's measurement was never
            # consulted. The verdict turned on whether the file happened to carry an empty column.
            checks.append(_Check("pvalue_is_measured", True, "no p-value column reported", examined=False))
    else:
        # Coerced, not filtered on dtype. `c in numeric.columns` required a numeric column, and an
        # R-written table carrying one `NA` reads as object dtype -- reported as "no p-value column
        # reported" about a file that has one.
        values = pd.to_numeric(frame[pval_col], errors="coerce")
        distinct = int(values.nunique(dropna=True))
        if n_rows <= _MIN_ROWS_TO_JUDGE:
            # The recorded svca run is exactly ten genes -- one row below the floor -- and its
            # p-value column holds 1.0 in all ten. "'pvalue' takes 1 distinct values over 10 genes"
            # was reported as a passing check: the precise failure this check exists to catch,
            # offered as the evidence the run is sound.
            checks.append(
                _Check(
                    "pvalue_is_measured",
                    True,
                    f"{n_rows} gene(s) is too few to tell a constant column from a real test, so this was not checked",
                    examined=False,
                )
            )
        elif distinct > 2:
            checks.append(
                _Check("pvalue_is_measured", True, f"{pval_col!r} takes {distinct} distinct values over {n_rows} genes")
            )
        else:
            uniq = sorted(float(v) for v in set(values.dropna()))
            # A flag needs both of its states present. One repeated value is a constant, and a
            # correction that saturates produces one: somde's q-values are computed, they just all
            # came out at 0.898. Calling that "a yes/no flag" would trade one false sentence for
            # another. Report what varies instead, and name the usual cause only when it fits.
            if distinct == 2 and set(uniq) <= {0.0, 1.0}:
                cause = "those are the two states of a yes/no flag, not a computed significance test"
            else:
                cause = (
                    "so it cannot separate one gene from another -- a correction that saturates does "
                    "this, and so does a constant written in place of a test"
                )
            varying = next(
                (
                    str(c)
                    for c in frame.columns
                    if str(c) != str(pval_col)
                    and str(c).lower() in pvalue_names
                    and int(pd.to_numeric(frame[c], errors="coerce").nunique(dropna=True)) > distinct
                ),
                None,
            )
            also = f"; the tool's own {varying!r} does vary, but selection is not thresholded on it" if varying else ""
            checks.append(
                _Check(
                    "pvalue_is_measured",
                    False,
                    (
                        f"{pval_col!r} -- the column selection is thresholded on -- takes only {distinct} "
                        f"distinct value(s) ({uniq!r}) over {n_rows} genes, {cause}{also}"
                    ),
                )
            )
    return checks


# ----------------------------------------------------------------------------------------------
# clustering / fallback
# ----------------------------------------------------------------------------------------------


def _clustering_checks(frame, tool_name: str = "") -> list[_Check]:
    # ``_is_label_column`` asks the values; the name is only a preference among columns that already
    # qualify. The previous fallback was ``frame.columns[0]``, and on the ``barcode, x, y`` layout a
    # tool writes when it has no assignment to report that is the barcode: this repo's recorded
    # ``mist_region_assignments.csv`` came back "193 distinct labels in 'spot_id' over 193 spots",
    # examined and passing, which is what lets ``_verdict`` call such a review "ok". L1 solved this
    # already -- its comment names the same failure -- so the heuristic is imported, not restated.
    #
    # ``_choose_key``, not a local re-derivation of it. The four-keyword list that stood here was a
    # narrower copy of L1's preference order that also skipped the registry lookup entirely, so L2
    # never consulted the ``prediction_key`` the tool declares even though it is holding
    # ``manifest["tool_name"]``. That is how ``barcode, array_row, array_col`` -- the layout a tool
    # writes when it has no assignment to report -- got past: the barcode is rejected for being
    # unique, no keyword matched, and the *array row* was published as "20 distinct labels in
    # 'array_row' over 300 spots", examined and passing.
    from spatialomicsgym.postanalysis.tasks.clustering import _choose_key, _key_evidence, _registry_keys

    columns = [str(c) for c in frame.columns]
    col = _choose_key(columns, tool_name, frame)
    if col is None:
        return [_Check("signal_free", True, "no label column to check", examined=False)]

    # Whether the pick rests on a name or only on the values' shape. ``_choose_key`` falls back to
    # the first qualifying column, and at that point "which column holds the assignment" is a guess:
    # an integer coordinate has exactly the dtype and cardinality of a domain label. The guess is
    # still worth reporting -- and a collapsed clustering in it is still worth failing on, since no
    # column of that table varies -- but a *passing* count from a column nobody identified is not a
    # measurement of the tool's clustering, and it must not be what pays for "ok".
    #
    # A name match is only evidence while the tool has declared nothing. When it *has* declared a
    # ``prediction_key`` and the output does not carry it, the name match is evidence the other way:
    # the recorded ``spacel`` run declares ``splane_cluster``, neither of its two output files has
    # it, and the column that stood in -- ``cluster``, 3 distinct labels -- is element-wise identical
    # to the ``mini_spatial.h5ad`` the tool was handed. That count was examined and passing.
    declared = _registry_keys(tool_name)
    evidence = _key_evidence(col, declared)
    identified = evidence == "registry" or (evidence == "name" and not declared)
    labels = frame[col].dropna()
    distinct = int(labels.nunique())
    if len(labels) <= _MIN_ROWS_TO_JUDGE:
        # The collapse rule below needs spots before "one label" means the clustering failed rather
        # than that the region is small. Saying "2 distinct labels in 'domain' over 4 spots" is the
        # only value check this task type has, so it was also the sentence that bought "ok".
        return [
            _Check(
                "signal_free",
                True,
                f"{len(labels)} labelled spot(s) is too few to tell a collapsed clustering from a "
                "small region, so this was not checked",
                examined=False,
            )
        ]
    if distinct <= 1:
        return [
            _Check(
                "signal_free",
                False,
                f"all {len(labels)} spots carry the single label {labels.iloc[0]!r} in {col!r} -- "
                "clustering collapsed; this is a failed run, not a one-domain tissue",
                fatal=True,
            )
        ]
    if not identified:
        why = (
            f"this tool declares its assignment in {', '.join(repr(d) for d in declared)} and no column here is it"
            if declared
            else f"no column of {columns[:6]} is named as a domain assignment and none is the key this tool declares"
        )
        return [
            _Check(
                "signal_free",
                True,
                f"{distinct} distinct labels in {col!r} over {len(labels)} spots -- but {why}, so "
                "which column holds the clustering is a guess and this is not evidence that one was "
                "produced",
                examined=False,
            )
        ]
    return [_Check("signal_free", True, f"{distinct} distinct labels in {col!r} over {len(labels)} spots")]


def _generic_signal_free(frame) -> _Check:
    numeric = _numeric(frame)
    if numeric.empty:
        return _Check("signal_free", True, "no numeric prediction column to check", examined=False)
    return _degenerate_matrix_check(numeric)


# ----------------------------------------------------------------------------------------------
# verdict
# ----------------------------------------------------------------------------------------------


def _verdict(checks: list[_Check], manifest: dict[str, Any]) -> dict[str, Any]:
    failed_fatal = [c for c in checks if not c.passed and c.fatal]
    failed_warn = [c for c in checks if not c.passed and not c.fatal]
    # A run whose task type predicts nothing pays for "ok" with its figure instead of a value check.
    # Without this a visualization run that drew exactly what it was asked came back "suspicious" --
    # "nothing was verified about the prediction's values" -- and the one follow-on round of the turn
    # went to verify_flagged_result, hunting for a prediction no plot tool writes (live run
    # 2026-10-01, S3). Every task type with a prediction keeps the value-check rule unchanged.
    figure_only = has_no_prediction_slot(manifest)
    figures = manifest.get("figures")
    declared_figures = [
        f for f in (figures if isinstance(figures, (list, tuple)) else ()) if isinstance(f, dict) and f.get("path")
    ]
    figures_written = bool(declared_figures) and any(
        c.name == "declared_artifacts_exist" and c.examined and c.passed for c in checks
    )

    if failed_fatal:
        verdict = "unusable"
        reasons = [f"{c.name}: {c.detail}" for c in failed_fatal]
    elif failed_warn:
        verdict = "suspicious"
        reasons = [f"{c.name}: {c.detail}" for c in failed_warn]
    elif figure_only and not figures_written:
        verdict = "suspicious"
        reasons = [f"a {manifest.get('task_type')} run's result is its figure, and no figure was declared"]
    elif not figure_only and not any(c.examined and c.name in VALUE_CHECK_NAMES for c in checks):
        # "ok" has to be paid for with a value check. The structural checks say the manifest is well
        # formed and the file sits in the slot the tool publishes; they say nothing about whether
        # the numbers inside it are a result. A prediction nobody could read earns uncertainty --
        # that is the honest verdict, and it is the one that keeps the planner from scheduling
        # further analysis on an unverified run.
        verdict = "suspicious"
        skipped = [c.detail for c in checks if not c.examined]
        reasons = [
            "nothing was verified about the prediction's values: "
            + ("; ".join(skipped[:2]) or "no value check could be run on this output")
        ]
    else:
        verdict = "ok"
        quoted = [c for c in checks if c.examined and c.name in ("signal_free", "proportion_orientation") and c.passed]
        reasons = [c.detail for c in quoted]
        if figure_only:
            reasons.insert(
                0,
                f"{len(declared_figures)} figure(s) written and on disk ({declared_figures[0]['path']}); "
                f"a {manifest.get('task_type')} run predicts nothing, so the figure is its result",
            )
        # A finding whose key is the name of a check just quoted is that check's own sentence,
        # said again -- the SVG runner names its finding ``signal_free`` after the check, so
        # ``'Z' takes 500 distinct values over 500`` is followed three slots later by ``Result
        # carries no information: False``. Four reasons, three facts, and the one asking the reader
        # to invert a negative is the one that adds nothing. On 9 of the 50 recorded ``ok`` runs,
        # all of them svg_detection; ``signal_free`` is the only finding key in the corpus that is
        # also a check name. Keyed on what was *quoted*, not on every check: a check that examined
        # nothing contributes no reason, so its finding is the only place that fact appears and has
        # to keep its slot. Nothing is deleted either way -- ``render._section_findings`` writes
        # every finding to the page regardless; what is freed is a slot in the case *for* the page.
        said = {c.name for c in quoted}
        # Filtered *before* the slice, not after. These reasons are rendered under a green ``ok``
        # badge, in the report's review section and on the portal card, as why the result can be
        # trusted -- and ``engine._write_scan`` appends its bookkeeping findings before any task
        # runner appends anything, so one of them was reason #1 on every run that got this far. All
        # 49 recorded ``ok`` runs cited a file count; five cited two; on the ncem run
        # ``Files written by the tool: 4`` was half of the reviewer's entire case. It is a true
        # sentence about the directory and no evidence at all about the numbers. Slicing first would
        # merely have hidden it while still spending the slot; filtering first hands the slot to
        # what the tool actually found. Same list ``next_step`` filters its prompt with, for the
        # same reason -- see :data:`manifest.BOOKKEEPING_FINDINGS`.
        findings = [
            f
            for f in (manifest.get("findings") or [])
            if isinstance(f, dict) and f.get("key") not in BOOKKEEPING_FINDINGS and f.get("key") not in said
        ]
        for f in findings[:3]:
            if "label" in f and "value" in f:
                reasons.append(f"{f['label']}: {f['value']}")
        if not reasons:
            reasons = [f"all {len(checks)} checks passed"]

    return {"verdict": verdict, "reasons": reasons, "checks": [c.as_dict() for c in checks]}


def _main(argv: list[str] | None = None) -> int:
    """``python -m spatialomicsgym.postanalysis.review {write-review|register-artifacts|review} DIR``.

    The door :func:`_as_the_owner` opens as the account that can write ``DIR``, not one for people:
    the verdict (or, for ``review``, the manifest to judge) arrives as JSON on stdin and the answer
    leaves as one JSON line on stdout.
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in _OWNER_OPS:
        print(f"usage: python -m spatialomicsgym.postanalysis.review {{{'|'.join(_OWNER_OPS)}}} DIR", file=sys.stderr)
        return 2
    op, directory = args
    if op == "register-artifacts":
        print(json.dumps({"added": register_new_artifacts(directory)}))
        return 0
    payload = json.loads(sys.stdin.read() or "{}")
    if op == "review":
        manifest = payload.get("manifest")
        if not isinstance(manifest, dict):
            print("review needs {'manifest': {...}} on stdin", file=sys.stderr)
            return 2
        # ``_review_here``, not ``review_manifest``: this process is the account's, and asking again
        # which process should judge could only send the question round once more.
        print(json.dumps({"review": _review_here(manifest, directory)}))
        return 0
    review, steps = payload.get("review"), payload.get("next_steps")
    if not isinstance(review, dict) or not isinstance(steps, list):
        print("write-review needs {'review': {...}, 'next_steps': [...]} on stdin", file=sys.stderr)
        return 2
    print(json.dumps({"written": write_review(directory, review, steps)}))
    return 0


if __name__ == "__main__":  # pragma: no cover - the as_owner child's entry point
    raise SystemExit(_main())
