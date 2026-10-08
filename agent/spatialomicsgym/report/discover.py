"""
Find post-analysis runs on disk, and resolve a client-supplied run identifier back to a directory.

A "run" is any directory holding a readable ``manifest.json`` (the contract writes it LAST and
atomically, so its presence means the run is complete enough to show). Discovery is deliberately
bounded -- a results root can be a live scratch tree with hundreds of thousands of files, and this
runs inside a web request:

* depth-capped, so a deep unrelated tree is not walked to the bottom;
* directory-count-capped, so a pathological tree ends the scan instead of the request;
* run directories are not descended into (their ``figures/``/``tables/`` hold no nested runs);
* symlinked directories are never followed, so the walk cannot be steered out of the root.

:func:`resolve_run` is the *only* way a client-supplied string becomes a path. It refuses anything
:func:`~spatialomicsgym.report.manifest.safe_subpath` refuses (absolute, ``..``, drive letters, NUL,
symlink escapes) and then requires the result to hold a manifest.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

from .manifest import (
    DEFAULT_RESULTS_DIRNAME,
    MANIFEST_NAME,
    MAX_MANIFEST_BYTES,
    ManifestError,
    is_run_dir,
    load,
    safe_subpath,
)

logger = logging.getLogger(__name__)

#: How deep below a root a run directory may sit. A run is ``<tool output directory>/post_analysis``,
#: so this must clear the deepest output directory any layout produces -- and at 4 it did not clear
#: the one this repo writes itself: ``benchmark_multi_llm/<llm>/<tool>/<dataset>`` puts a run at
#: depth 5, so a portal pointed at ``benchmarks/results`` listed the single-model runs and omitted
#: every multi-LLM one, with nothing on the page to say so. Set one past that, which leaves room for
#: a wrapping project directory. Not a cost bound -- :data:`MAX_DIRS` is, and a deeper walk spends
#: that budget faster rather than without limit.
MAX_DEPTH = 6
#: Hard ceiling on directories visited per scan, across all roots. Unlike the depth bound this one
#: is about cost, so reaching it is reported (``on_limit``) rather than treated as a scope decision.
MAX_DIRS = 8000
#: Hard ceiling on runs returned.
MAX_RUNS = 300
#: Hard ceiling on declared artefacts stat-ed per run when summarising it. See :func:`_n_present`.
MAX_DECLARED_TO_STAT = 2000

#: Never walked into: version control, caches, virtualenvs, and the per-run asset subdirectories.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".ipynb_checkpoints",
        "__pycache__",
        "node_modules",
        "site-packages",
        "figures",
        "tables",
        "envs",
        ".conda",
        ".cache",
    }
)


#: Characters of the root digest carried alongside a run's relative path. Deliberately far shorter
#: than the 32-hex run ``redact_text`` masks, so it survives being printed into a link.
ROOT_KEY_CHARS = 8


def root_key(root: Path | str) -> str:
    """A short, stable handle for one results root -- the missing half of a run's identifier.

    ``dir`` names a run by its path *relative to* its root, which is what keeps an absolute server
    path off the wire. But nothing says two roots cannot hold the same relative subtree, and when
    they do (``~/work/results`` and ``~/archive/results``, each with ``run_spagcn_visium/post_analysis``)
    both rows carry the identical ``dir`` and :func:`resolve_run` hands both to whichever root sorts
    first. The second run is listed on the portal and unreachable from it.

    A digest rather than the path, because the absolute path is exactly what ``dir`` exists not to
    leak; and rather than an index into the roots list, because that renumbers whenever
    ``SOG_RESULTS_ROOTS`` changes and would silently re-point an old bookmark at a different run.
    """
    return hashlib.sha256(str(root).encode("utf-8", "surrogateescape")).hexdigest()[:ROOT_KEY_CHARS]


def normalized_roots(candidates) -> list[Path]:
    """Existing directories, resolved, de-duplicated, and with nested roots collapsed into their
    ancestor -- otherwise the same run is discovered twice under two different relative names."""
    seen: list[Path] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            path = Path(str(candidate)).expanduser().resolve()
            if not path.is_dir():
                continue
        except (OSError, ValueError, RuntimeError):
            continue
        if any(path == kept or kept in path.parents for kept in seen):
            continue
        seen = [kept for kept in seen if path not in kept.parents]
        seen.append(path)
    return seen


def find_runs(
    roots,
    *,
    max_depth: int = MAX_DEPTH,
    max_dirs: int | None = None,
    limit: int = MAX_RUNS,
    on_error: Callable[[OSError], None] | None = None,
    on_limit: Callable[[int], None] | None = None,
    on_depth: Callable[[Path], None] | None = None,
    on_total: Callable[[int], None] | None = None,
    on_root: Callable[[Path, bool], None] | None = None,
    on_manifest: Callable[[Path, Path, str, dict[str, Any]], None] | None = None,
    skip: Any = (),
) -> list[dict[str, Any]]:
    """Summaries of every run under ``roots``, newest first. Best-effort -- never raises.

    ``skip`` is a set of directories the walk never enters and never offers: on the portal, every
    OTHER account's folder under the agent's outputs tree. The tree is still a root -- the runs at
    its top predate per-account filing and belong to nobody -- but a signed-in reader's listing
    must not descend into a folder that is somebody else's. A root that itself lies inside a
    skipped directory is not walked at all.

    Each summary is JSON-safe: ``{dir, root, root_key, name, mtime, tool_name, task_type, status,
    n_figures, n_figures_present, n_tables, n_tables_present, n_findings, n_warnings, verdict}``.
    The two ``_present`` counts are what is actually on disk, and the reason both pairs are carried:
    a card that promises figures it cannot show is what R79/B fixed. ``dir`` is the run's path
    RELATIVE to its root, which is the identifier every endpoint takes -- an absolute path is never
    handed to the browser and never accepted back from it. ``root_key`` (see :func:`root_key`)
    disambiguates the roots two same-named runs would otherwise share.

    ``limit`` caps the *result*, not the search, and that distinction is the whole of this function.
    Stopping the walk at ``limit`` runs and sorting the survivors afterwards -- which is what this
    did -- makes "newest first" true only of the subset it happened to reach: the walk descends in
    sorted name order, so above the cap the list was whichever runs sorted first alphabetically, and
    the run the user finished a minute ago could be absent. The portal shows this list as "your
    runs", so that reads as a lost run.

    Selection is therefore over everything the walk sees, decided by a ``stat`` of the manifest and
    held in a heap of at most ``limit`` entries, and only the survivors are parsed. That keeps the
    expensive half -- reading and validating JSON -- capped at ``limit`` rather than paid per run,
    which a plain collect-sort-truncate would not. The cost is that a tree with more than ``limit``
    runs is now walked to the end instead of abandoned early; ``max_dirs`` still bounds that, and
    directory iteration is the cheap half.

    ``on_error``, if given, receives each :class:`OSError` the walk hits -- a subtree the process
    cannot open. Walking on past it is right, since one locked directory must not empty the whole
    listing, but the runs beneath it are then missing from a list the caller goes on to present as
    complete. The sink is how a caller can say so instead.

    ``on_limit`` is the same idea for the other way this list goes short: it is called once, with
    the count, if ``max_dirs`` is reached. The budget is spent in root order, so an early root that
    is a large scratch tree leaves nothing for the later ones -- each is entered and abandoned on
    its first yield -- and without this the caller cannot tell that from an empty disk. The portal
    printed "No analysis runs yet" over exactly that. Neither of its existing signals covers it:
    its ``truncated`` flag means "more than ``limit`` runs matched", not "the walk stopped".

    ``on_depth`` is the third, and the last of the walk's ceilings to reach a reader: it is called
    once, with the first directory whose children were pruned for depth. ``max_depth`` is applied
    silently otherwise, so a root pointed at a project rather than at an output directory --
    ``2026/visium/sampleA/batch3/rep1/<tool>/post_analysis`` is seven levels, one past the default --
    listed nothing and said the disk was empty. It is deliberately not folded into ``on_limit``: the
    two have the same consequence and opposite remedies, and "this root holds too many directories"
    sends someone with a deep tree to prune a tree that is not too big.

    It fires only where there was something left to descend into, judged *after* ``_SKIP_DIRS``. A
    tree that merely reaches the ceiling has lost nothing, and a ``.git`` at the bottom of one is not
    a run that went missing -- hedging on either would put the warning on every deep checkout.

    ``on_total`` is not a warning like the other three -- it is called once, always, with the number
    of run directories the walk *found*, which is not ``len`` of what is returned. Two things come
    between the two numbers: ``limit``, and ``_summarize`` declining a row. A caller cannot recover
    it afterwards, and the caller that needs it is the portal, which reports "there are more on
    disk". It used to infer that by asking for one row past its own cap and seeing whether the row
    arrived -- so a single run directory whose name is not valid UTF-8, dropped by ``_summarize``
    after it had already spent a slot in the heap, silently turned "showing 300 runs (more were
    found)" into "300 run(s) found", a claim about the disk that nothing had established.

    The count, not a "was it capped" flag, because a flag cannot be phrased to answer it: the heap
    is full at exactly ``limit`` candidates with nothing displaced, and the cap the caller compares
    against is its own, not the ``limit`` it passed.

    ``on_root`` is ``on_limit`` asked per root: it is called once for each root actually walked, with
    that root and whether its walk ran to the end. ``on_limit`` carries a count, which tells a caller
    *that* the budget ran out and not *where* -- and where is the only actionable half, because the
    budget is spent in ``normalized_roots`` order and the roots after the one that spent it are
    entered for a single ``stat`` and abandoned on their first yield. That is a fact only this loop
    holds; nothing at the call site can recover it afterwards. The portal was printing "searched"
    beside a root it had never opened, having no other source for the word.

    The CLI door needs none of this because it calls this function once per root, so each gets its
    own budget and its ``on_limit`` closure can bind the root it was made for. A page load has to be
    bounded across all roots at once, which is why the portal shares one and needs the answer
    threaded out.

    ``on_manifest`` is not a warning at all -- it is the parsed manifest itself, handed to the
    caller as each row is built: ``on_manifest(root, run, rel, manifest)``. This function already
    reads and validates every surviving run's manifest to fill the row, and then drops it; a
    caller that needs more than a row carries -- the findings, the review's checks -- would
    otherwise walk the tree a second time and parse all of it again. Offering what is already in
    hand costs one call. It fires only for a manifest that *parsed*: an unreadable one has
    nothing to offer, and the row it produces already says so.

    ``max_dirs`` defaults to :data:`MAX_DIRS` read at call time rather than bound at definition, so
    a deployment or a test can move the ceiling without editing a default.
    """
    if limit <= 0:
        return []
    if max_dirs is None:
        max_dirs = MAX_DIRS
    banned: set[Path] = set()
    for raw in skip or ():
        try:
            banned.add(Path(str(raw)).resolve())
        except (OSError, ValueError, TypeError):
            continue

    def off_limits(path: Path) -> bool:
        if not banned:
            return False
        try:
            resolved = path.resolve()
        except (OSError, ValueError):
            return False
        return resolved in banned or any(b in resolved.parents for b in banned)

    newest: list[tuple[float, str, str, Path, Path]] = []
    visited = 0
    offered = 0
    limit_reported = False
    depth_reported = False

    def note(error: OSError) -> None:
        if on_error is None:
            return
        try:
            on_error(error)
        except Exception:  # discovery is best-effort; a caller's reporting bug is not fatal here
            pass

    def note_limit(count: int) -> None:
        nonlocal limit_reported
        if on_limit is None or limit_reported:
            return
        limit_reported = True
        try:
            on_limit(count)
        except Exception:  # as above: telling the caller must not be able to empty the listing
            pass

    def note_depth(where: Path) -> None:
        nonlocal depth_reported
        if on_depth is None or depth_reported:
            return
        depth_reported = True
        try:
            on_depth(where)
        except Exception:  # as above: telling the caller must not be able to empty the listing
            pass

    def note_total(count: int) -> None:
        if on_total is None:
            return
        try:
            on_total(count)
        except Exception:  # as above: telling the caller must not be able to empty the listing
            pass

    def note_root(root: Path, complete: bool) -> None:
        if on_root is None:
            return
        try:
            on_root(root, complete)
        except Exception:  # as above: telling the caller must not be able to empty the listing
            pass

    def offer(root: Path, run: Path, rel: str) -> None:
        """Weigh one candidate against the ``limit`` best so far, on the strength of a ``stat``."""
        nonlocal offered
        offered += 1
        try:
            mtime = (run / MANIFEST_NAME).stat().st_mtime
        except OSError:
            mtime = 0.0
        # ``rel`` and the root break ties, so two runs written in the same second keep a stable
        # order between listings; the portal re-lists on every page load. The trailing paths are
        # payload and never decide a comparison -- (mtime, rel, root) is already unique, since
        # ``normalized_roots`` collapses nested roots.
        entry = (mtime, rel, str(root), root, run)
        if len(newest) < limit:
            heapq.heappush(newest, entry)
        else:
            heapq.heappushpop(newest, entry)

    for root in normalized_roots(roots):
        if off_limits(root):
            continue
        root_kind = run_dir_kind(root)
        if root_kind:  # the root itself is one run
            offer(root, root, ".")
            if root_kind == RUN_DIR_OURS:
                # Not an early exit from the search: a run holds no nested runs, so there is nothing
                # below this root left to look at and it is as finished as a walked one.
                note_root(root, True)
                continue
            # An opaque manifest at the root is still offered above -- it may be ours, truncated --
            # but it says nothing about the subtree, so the walk goes on rather than reporting the
            # root's own broken card as the whole of what is there.
        root_complete = True
        for dirpath, dirnames, _files in os.walk(root, followlinks=False, onerror=note):
            visited += 1
            if visited > max_dirs:
                # Only this root's walk ends here. The loop still enters the remaining roots, so a
                # root that is itself a single run is still offered -- that costs one `stat` and is
                # the behaviour every caller already has.
                note_limit(visited - 1)
                root_complete = False
                dirnames[:] = []
                break
            here = Path(dirpath)
            try:
                depth = len(here.relative_to(root).parts)
            except ValueError:  # pragma: no cover - os.walk always yields paths under root
                dirnames[:] = []
                continue
            # Filtered before the ceiling is judged, not after. The prune used to come first, which
            # made "was anything left below?" unanswerable -- by the time the depth was known the
            # list was already empty -- and would have counted a ``.git`` at the bottom of a deep
            # checkout as a run the walk failed to reach.
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
            if banned:
                dirnames[:] = [d for d in dirnames if not off_limits(here / d)]
            if depth >= max_depth and dirnames:
                note_depth(here)
                dirnames[:] = []
            if here != root:
                kind = run_dir_kind(here)
                if kind:
                    offer(root, here, here.relative_to(root).as_posix())
                if kind == RUN_DIR_OURS:
                    dirnames[:] = []  # a run holds no nested runs
        # Only the directory budget is reported here. A subtree this process could not open, and a
        # subtree pruned for depth, are the other two ways a root's walk comes up short, and each
        # already has its own sink and its own note -- one fact per channel, so a page can name the
        # right remedy rather than a generic "something was missed".
        note_root(root, root_complete)

    # Before the parse, so nothing the parse declines can subtract from it.
    note_total(offered)

    out: list[dict[str, Any]] = []
    for mtime, rel, _root_key, root, run in sorted(newest, reverse=True):
        record = _summarize(root, run, rel, mtime, on_manifest)
        if record:
            out.append(record)
    # Returned exactly as :func:`run_label` named them. A pass over this list used to rename the rows
    # that still collided, which is the one thing a *list* must not do: see ``run_label``'s closing
    # note, and ``test/test_a_run_is_never_renamed_by_who_it_collides_with.py``.
    return out


def runs_since(roots, since: float, *, limit: int = 2, skip: Any = ()) -> list[dict[str, Any]]:
    """Runs whose manifest was written after ``since`` -- what the turn that just ended produced.

    Lives here, next to :func:`find_runs`, because all three front doors need it and none of them
    owns it: the web chat attaches the newest run to its answer bubble, the CLI prints it after a
    turn, and the portal lists it. A second discovery implementation in any one of them is how they
    would come to disagree about which runs exist, so this one filters what ``find_runs`` already
    found rather than walking again.

    Newest first, because ``find_runs`` returns newest first and the caller wants "the run this turn
    produced", not "a run". Never raises: a front door that cannot describe the result must still
    deliver the result.

    ``limit`` is handed to ``find_runs`` rather than applied afterwards. ``mtime >= since`` selects a
    *prefix* of a newest-first list, so the two are equivalent in what they return, but summarising
    ``MAX_RUNS`` manifests to keep two of them cost 0.298 s against 0.040 s over 300 runs -- paid
    inside the turn lock, on every answer, by all three front doors.

    Known limitation -- it cannot tell whose turn wrote a run. The selector is a clock reading against
    a shared directory, so a manifest written into the same results root by *another process* while
    this turn was running -- a second CLI session, a benchmark sweep, a worker invoked by hand -- is
    newer than ``since`` and is attached to this turn's answer. Within one process the boundary is
    exact; across processes it is a heuristic. Closing it needs the engine to stamp each run with the
    turn that asked for it, so that identity rather than recency decides. Recorded rather than
    narrowed: mtime attribution is right for the single-session use all three front doors are built
    for, and guessing at provenance to exclude a stranger's run would also drop real ones.
    """
    try:
        runs = find_runs(roots, limit=max(1, int(limit)), skip=skip)
    except Exception:
        return []
    fresh: list[dict[str, Any]] = []
    for run in runs:
        mtime = run.get("mtime")
        # ``bool`` is an ``int``; a manifest whose mtime came back as ``True`` would otherwise
        # compare as 1.0 and make every run before 1970 look fresh.
        if not isinstance(mtime, (int, float)) or isinstance(mtime, bool):
            continue
        if mtime >= since:
            fresh.append(run)
    return fresh[:limit]


def run_label(run: Path, rel: str = "") -> str:
    """A name for this run that a reader can tell apart from the next one's.

    ``run.name`` alone is the results directory, and L1's default results directory is called
    ``post_analysis`` for every tool it has ever analysed: all 92 recorded runs carry that name, so
    the portal drew 92 identical cards and the download route named 92 identical files. When the
    directory name carries no information the directory it sits in does -- that is the tool's own
    output directory, which is what the user named -- so the label becomes ``<tool dir>/post_analysis``.

    A results directory the caller named itself is returned unchanged -- among its siblings that name
    is already distinguishing.

    ``rel``, the run's path relative to the root it was found under, is used whenever it has two or
    more components, because it is unique within that root by construction: two components are enough
    among *siblings* and not across a tree, where ``sample1/bass/post_analysis`` and
    ``sample2/bass/post_analysis`` -- one tool over two samples -- both reduce to ``bass/post_analysis``.

    A one-component ``rel`` falls through to the suffix rule instead of being returned, and that is
    the whole reason this is not simply ``rel or ...``: for a run sitting directly *under* its root,
    ``rel`` is ``post_analysis`` -- shorter than today's answer, which borrows the root's own
    directory name. Taking ``rel`` there would rename every such run to the one string all 92
    recorded runs would have shared.

    THIS IS A LOCAL RULE, AND THAT IS THE POINT
    ...........................................
    It needs only ``(root, run)``, so the five places that name a run -- the portal card, the report
    page's heading, its browser title, the downloaded filename, the chat bubble -- each compute it
    alone and agree by construction. The first repair for the collision above was in the run *list*,
    renaming only the rows that collided; a collision is cross-row knowledge, and three of those five
    surfaces see exactly one run, so the card said ``sample1/bass/post_analysis`` and the page it
    opened said ``bass/post_analysis``. That is R79/M, reopened at depth 2 by the fix for depth 2.

    Called with no ``rel`` -- the renderer standalone, which has no root to be relative to and is
    forbidden by R80/S from printing the absolute path it does have -- the answer is exactly what it
    always was. At the flat ``<tool>/post_analysis`` depth the two rules produce the same characters,
    which is why all 92 recorded runs come through this unchanged.

    TWO COLLISIONS SURVIVE IT, AND NEITHER IS ANSWERED BY RENAMING
    ..............................................................
    Both are a root's own name reappearing inside it: a run directly under its root, named
    ``<root dirname>/post_analysis`` because its ``rel`` is a single component, sitting beside a
    genuine ``<root dirname>/post_analysis`` subtree; and the same ``rel`` under two different roots,
    unique within each and colliding across them.

    A ``_distinguish`` pass over the run list used to rewrite the first one's ``name``, and its
    ``tool_name`` with it. That reopened R79/M in exactly the layout it existed to repair -- one card
    read ``post_analysis`` while the page it opened read ``lab/post_analysis``, because the four
    single-run surfaces cannot see a collision and so cannot reproduce a name derived from one. A
    collision is knowledge only the listing has, and what the listing does with it belongs in what it
    *shows*: a card whose title it shares with another prints where its run is (R81/C, R81/F). The
    name stays this function's answer, everywhere.
    """
    # ``rel`` is always ``as_posix()`` of a ``relative_to`` result, so counting separators counts
    # components. A root that *is* the run yields ``"."``, which has none and falls through.
    if rel.count("/") >= 1:
        return rel
    if run.name and run.name != DEFAULT_RESULTS_DIRNAME:
        return run.name
    parent = run.parent.name
    if parent:
        return f"{parent}/{run.name}"
    return run.name or rel


#: Image extensions offered from a tool's own output directory. A display allowlist, matching the
#: portal's inline-servable set: this decides what a route reads out of a directory no manifest ever
#: described, so it stays exactly as wide as showing a picture needs.
TOOL_FIGURE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")

#: How many of a tool's own figures a page offers. A layout bound, not a judgement -- one recorded
#: output directory holds 31 of them. The total is reported separately, so a capped list says so.
MAX_TOOL_FIGURES = 24

#: Directory entries examined while looking for those images, and image hits kept. Both are cost
#: bounds: this runs inside a web request against a live scratch tree.
_MAX_SOURCE_ENTRIES = 20_000
_MAX_SOURCE_HITS = 500


def tool_figure_dir(run: Path, root: Path | str) -> Path | None:
    """The directory whose images belong to ``run``, or ``None`` when that cannot be known.

    ``run.parent`` is the tool's own output directory *by construction* -- but only for L1's default
    layout, where the results dir is ``<source>/post_analysis``. A caller who passed ``results_dir=``
    put the run somewhere of their own choosing, and that parent is then an unrelated directory
    whose images are not this run's, so the answer there is "unknown" rather than a guess.

    ``root`` is required, and the parent must still lie inside it. The case that needs the check is
    narrow and real: a results root that is *itself* a ``post_analysis`` directory makes ``run`` the
    root, and its parent one level ABOVE everything this server is allowed to read. One caller lists
    from this and another serves from it, so the decision has to be made once, here.
    """
    if run.name != DEFAULT_RESULTS_DIRNAME:
        return None
    try:
        base = Path(root).expanduser().resolve()
        parent = run.parent.resolve()
        if not parent.is_dir():
            return None
    except (OSError, ValueError, RuntimeError):
        return None
    if parent != base and base not in parent.parents:
        return None
    return parent


def tool_figures(
    run: Path, root: Path | str, *, limit: int = MAX_TOOL_FIGURES
) -> tuple[list[dict[str, Any]], int, bool]:
    """``(shown, total, capped)`` -- images the *tool* wrote beside this run. Never raises.

    A manifest describes only what post-analysis itself drew. The tool has usually already produced
    its own figures -- one recorded work tree holds 327 PNGs against 15 manifest figures -- and they
    were reachable only by knowing the server's filesystem layout. The two cases that make surfacing
    them worth the code are the ones where the manifest has least to say: a run that ``failed``
    (tangram published a transposed matrix, so the engine drew nothing while the tool's five plots
    sat beside it), and a task type the engine has no handler for.

    One directory, no recursion, and only names ending in :data:`TOOL_FIGURE_SUFFIXES`. Each is put
    through :func:`~spatialomicsgym.report.manifest.safe_subpath` here, at listing time, on exactly
    the terms the serving route applies -- including that the resolved file is a *direct child* of
    this directory, so a symlinked image is dropped from the list rather than listed and then
    refused as a broken picture.

    Newest first. Alphabetical put whatever sorts first in front of whatever the turn just drew,
    which on a directory holding more images than ``limit`` meant the four shown in a chat bubble
    were reliably not the four the question was about.

    ``total`` is exact unless the scan hit one of its two ceilings, and ``capped`` is how the caller
    knows which -- it renders "500+" rather than asserting a number it did not finish counting.

    ``capped`` is reported from the scan and not re-derived downstream, because it cannot be
    re-derived: the fact lived only here. It used to be recomputed as ``total >= _MAX_SOURCE_HITS``,
    which compares a *post*-filter count against a *pre*-filter ceiling and is wrong in both
    directions. Two dozen plots of which several are the zero-byte files matplotlib leaves behind on
    failure stop the scan at the ceiling and then count back under it, so the page printed
    "Showing 4 of 7" -- an exact total, no "+", about a directory it gave up on. And the entry
    ceiling has no expression in ``total`` at all: a directory of per-gene ``.txt`` logs exhausts it
    before most of the images are reached, and the undercount that comes back is indistinguishable
    from a complete reading.
    """
    source = tool_figure_dir(run, root)
    if source is None:
        return [], 0, False
    names: list[str] = []
    stopped = False
    try:
        with os.scandir(source) as entries:
            for seen, entry in enumerate(entries):
                if seen >= _MAX_SOURCE_ENTRIES or len(names) >= _MAX_SOURCE_HITS:
                    stopped = True
                    break
                # ``_encodable`` here and not at the two f-strings that consume this list, because
                # a name that cannot leave the process is not a figure this page can offer: the
                # ``quote()`` that builds its ``src`` raises, and so does escaping it into the body.
                # One ``plot_\xff.png`` beside a good run 500'd ``/results/view``, and -- via the
                # chat attachment's blanket except -- silently deleted the whole run card, the
                # figures that were fine along with it.
                if entry.name.lower().endswith(TOOL_FIGURE_SUFFIXES) and _encodable(entry.name):
                    names.append(entry.name)
    except OSError:
        # "I could not read it" and "there is nothing in it" are different answers, and this returned
        # the second one's value for both. ``tool_figure_dir`` has already established the directory
        # exists and is a directory, so what is left here is EACCES (a 0700 tool output directory and
        # a portal running as someone else), ESTALE on clustered NFS, EIO -- every one of them a
        # failure to look. The caller's whole reason for taking ``capped`` is to keep those apart, and
        # on ``([], 0, False)`` it deleted the "Figures the tool produced" section outright: the page
        # a reader opens *because the run failed* told them the tool drew nothing. Same treatment as
        # the two ceilings above -- keep whatever was reached, and say the count is a floor.
        stopped = True
    found: list[tuple[float, str, dict[str, Any]]] = []
    for name in sorted(names):
        target = safe_subpath(source, name)
        # ``parent != source`` is the containment the serving route enforces: this lists one
        # directory, so a name that resolves into a subdirectory of it was never on offer.
        if target is None or target.parent != source:
            continue
        try:
            if not target.is_file():
                continue
            stat = target.stat()
        except OSError:
            continue
        # A zero-byte plot is a matplotlib failure, not a figure; embedding it draws a broken icon.
        if stat.st_size > 0:
            found.append((stat.st_mtime, name, {"name": name, "size": int(stat.st_size)}))
    found.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in found[: max(0, int(limit))]], len(found), stopped


def _n_present(run: Path, entries) -> int | None:
    """How many of ``entries``' declared paths the detail page will actually be able to show.

    The same two steps that page takes -- contain, then ``is_file`` -- because the card is a
    promise about the click, and counting a declaration the renderer will refuse to embed makes it
    a false one. Costs one ``stat`` per declared artefact, paid only for the runs that survive the
    ``limit`` heap and whose manifests are already being parsed.

    ``None`` when there are more than :data:`MAX_DECLARED_TO_STAT` of them, meaning "not counted".
    The cap exists because ``MAX_MANIFEST_BYTES`` bounds the manifest's *bytes* and not its entry
    count: a 4.4 MB manifest declaring 100,000 figures is well inside the byte cap and turned
    ``GET /results`` into 7.3 s of ``stat`` calls, and the chat attachment -- which walks the
    entries a second time -- into 14.6 s inside the turn lock.

    Returning the *floor* past that cap, as this did, is the one answer the caller cannot use: it
    compares this number with the declared total and warns when they differ, so a run declaring
    2005 figures with every one of them on disk rendered as an amber ``2000 of 2005 fig`` -- telling
    the scientist five figures are missing when none were. "We did not finish counting" has to be
    distinguishable from "we counted, and some are gone".
    """
    try:
        if len(entries) > MAX_DECLARED_TO_STAT:
            return None
    except TypeError:  # pragma: no cover - manifest entries are a list by the time they reach here
        pass
    n = 0
    for seen, entry in enumerate(entries):
        if seen >= MAX_DECLARED_TO_STAT:
            break
        target = safe_subpath(run, entry.get("path"))
        if target is None:
            continue
        try:
            if target.is_file():
                n += 1
        except OSError:  # as everywhere in this module: an unreadable path is "not shown", not fatal
            continue
    return n


def _encodable(text: str) -> bool:
    """True when ``text`` survives a round trip to UTF-8 -- i.e. it can leave this process.

    A filename is bytes, not text. ``os.walk`` hands back the ones that are not valid UTF-8 with
    their undecodable bytes as lone surrogates (PEP 383), and *everything downstream that leaves
    the process* raises on those: ``urllib.parse.quote`` building the card's link, Starlette's
    ``JSONResponse`` encoding the run list, and the SSE encoder that carries a finished answer to
    the browser. One directory named ``bad_\\xff_run`` under a results root turned ``GET /results``
    and ``GET /api/results`` into 500s, and -- because the chat attachment is built *after* the
    agent has answered and outside the stream's own try/except -- truncated the SSE stream and lost
    the user a completed answer.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def encodable_text(text: str) -> str:
    """``text`` made safe to leave this process, by spelling out what cannot -- never raises.

    The counterpart to :func:`_encodable`, for the fields that are *display only*. Dropping is the
    right answer for an identifier: a repaired spelling names a directory that is not there, so the
    card would link to a 404. It is the wrong answer for a label, because the thing dropped is not
    the label, it is the whole row -- and a run vanishing is a worse lie than a run whose path is
    printed with ``\\udcff`` in it.

    Which matters most for the results *root*: one root spelled in non-UTF-8 is one string shared by
    every run under it, so dropping on it empties the portal and ``_roots_note`` then states
    "searched &middot; 0 run(s)" about a directory holding all of them.

    The overwhelmingly common case returns the argument itself, unexamined beyond one encode.
    """
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        pass
    try:
        return text.encode("utf-8", "backslashreplace").decode("utf-8", "replace")
    except Exception:  # pragma: no cover - defensive; the codecs above do not raise
        return repr(text)


def _summarize(
    root: Path,
    run: Path,
    rel: str,
    mtime: float,
    on_manifest: Callable[[Path, Path, str, dict[str, Any]], None] | None = None,
) -> dict[str, Any] | None:
    """One list row, or ``None``. A run whose manifest is corrupt is still listed, flagged, never fatal.

    ``on_manifest`` receives the parsed manifest this row was derived from, for a caller that
    needs more of it than a row carries. It is handed the object this function is about to stop
    using, so it costs nothing -- see :func:`find_runs` for why that matters.
    """
    label = run_label(run, rel)
    if not _encodable(rel) or not _encodable(label):
        # Dropped rather than sanitised: ``dir`` is this run's *identifier*, and a repaired spelling
        # names a directory that is not there, so the card would link to a 404. Nothing about the
        # run is lost that was ever reachable -- it had no addressable URL to begin with.
        logger.debug("results listing skipped a run whose path is not valid UTF-8: %r", rel)
        return None
    row: dict[str, Any] = {
        "dir": rel,
        # Display only -- ``root_key`` is the half of the identifier that has to survive the round
        # trip, and it digests the bytes rather than the text. So a root the filesystem spells in
        # something that is not UTF-8 costs this row its printed path and nothing else.
        "root": encodable_text(str(root)),
        "root_key": root_key(root),
        "name": label,
        "mtime": mtime,
    }
    # No ``has_report`` here. It was a ``stat`` per listed run for a question no door asks: the
    # portal's report route re-renders from the manifest on every request, so whether a previous
    # ``write_report`` left a file behind changes nothing anyone can see, and no client ever read
    # the key. A row is a promise to a reader; a field with no reader is a promise to nobody.
    try:
        m = load(run)
    except ManifestError as exc:
        row.update(
            tool_name=run_label(run, rel),
            task_type="unknown",
            status="unreadable",
            n_figures=0,
            n_tables=0,
            n_figures_present=0,
            n_tables_present=0,
            n_findings=0,
            n_warnings=0,
            verdict=None,
            error=str(exc.args[0] if exc.args else exc)[:200],
        )
        return row
    if on_manifest is not None:
        try:
            on_manifest(root, run, rel, m)
        except Exception:  # as with every sink here: a caller's bug must not empty the listing
            pass
    review = m["review"]
    row.update(
        # ``tool_name`` is optional and usually absent; the card's subtitle then falls back to the
        # same label the card's title uses, which at least names the directory the tool wrote.
        tool_name=m["tool_name"] or run_label(run, rel),
        task_type=m["task_type"],
        status=m["status"],
        n_figures=len(m["figures"]),
        n_tables=len(m["tables"]),
        # Declared and shown are separate facts and the card needs both: that the run produced three
        # figures is what makes "none of them are here" worth reading, and reporting only the
        # survivors would hide it. ``n_figures``/``n_tables`` keep the meaning ``/api/results``
        # already publishes.
        n_figures_present=_n_present(run, m["figures"]),
        n_tables_present=_n_present(run, m["tables"]),
        n_findings=len(m["findings"]),
        n_warnings=len(m["warnings"]),
        verdict=review["verdict"] if review else None,
    )
    return row


#: Return values of :func:`run_dir_kind`.
RUN_DIR_ORDINARY = ""
RUN_DIR_OURS = "ours"
RUN_DIR_OPAQUE = "opaque"


def run_dir_kind(candidate: Path | str) -> str:
    """How a directory holding a ``manifest.json`` should be treated. Never raises.

    :func:`~spatialomicsgym.report.manifest.is_run_dir` answers "is there a file with that name",
    which is the right question for a gate that only has to decide whether an *already issued*
    identifier still points at something. It is the wrong question for a walk, because deciding a
    directory is a run also decides to stop descending -- and ``manifest.json`` is not this
    package's name to reserve. A viewer config, an unpacked archive's inventory or another tool's
    own manifest at the top of a shared scratch directory would otherwise take every run beneath it
    out of every listing, with no truncation flag and no unreadable note to say so.

    Three answers, because two of them are not the same kind of "not ours":

    ``RUN_DIR_OURS``
        Parsed, and carries the two fields this package's writer always emits. Offer it and prune:
        our runs never nest runs, and that prune is what keeps the walk budget for later roots.
    ``RUN_DIR_OPAQUE``
        There is a ``manifest.json`` we could not read -- truncated, oversized, unreadable. It may
        well be ours, so it is still offered and still renders as the "unreadable" card that says
        what is wrong with it. But a manifest we cannot parse is not evidence of an empty subtree,
        so the walk continues past it.
    ``RUN_DIR_ORDINARY``
        No manifest, or one that parses and is plainly somebody else's. An ordinary directory: not
        offered, walked into like any other.

    The read costs one small file per directory that has a ``manifest.json`` at all, bounded by
    ``MAX_DIRS`` and by :data:`~spatialomicsgym.report.manifest.MAX_MANIFEST_BYTES`. The parse
    mirrors :func:`~spatialomicsgym.report.manifest.load` exactly -- same size cap, same
    ``errors="replace"``, same exception set -- so the two can never disagree about what is
    readable.
    """
    if not is_run_dir(candidate):
        return RUN_DIR_ORDINARY
    path = safe_subpath(candidate, MANIFEST_NAME)
    if path is None:  # a manifest.json linked in from outside the run is not this run's (see load)
        return RUN_DIR_OPAQUE
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            return RUN_DIR_OPAQUE
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError, RecursionError):
        return RUN_DIR_OPAQUE
    if isinstance(payload, dict) and "schema_version" in payload and "task_type" in payload:
        return RUN_DIR_OURS
    return RUN_DIR_ORDINARY


def contain(rel: Any, roots) -> tuple[Path, Path] | None:
    """``(root, path)`` for the first root ``rel`` legally sits under, or ``None`` if none does.

    Says nothing about the path *existing* -- that is the point. It lets a caller tell "you asked
    for something outside what I serve" (a refusal) apart from "that run is not here" (a 404),
    without ever leaking which of the two a probing client hit for an out-of-bounds path.
    """
    for root in normalized_roots(roots):
        target = safe_subpath(root, rel)
        if target is not None:
            return root, target
    return None


def resolve_run(rel: Any, roots, key: Any = None) -> tuple[Path, Path] | None:
    """Turn a client-supplied ``dir`` into ``(root, run_dir)``, or ``None``.

    The ONLY accepted form is a path relative to one of ``roots``. Absolute paths are refused even
    when they happen to point inside a root: accepting them would make every future bug in root
    resolution an arbitrary-read, and the run list only ever hands out relative identifiers anyway.

    ``key``, when given, is the :func:`root_key` the run list handed out beside ``rel``; only roots
    with that digest are considered, so a ``dir`` that exists under two roots resolves to the one
    the card was drawn for. It narrows and never widens.

    An **absent** key falls back to first-match-wins over all roots: that is what every identifier
    issued before the digest existed carries, so an old bookmark keeps resolving exactly as it did.
    A key that is present but matches no root is the opposite fact -- the link asserts a root this
    server is not serving, usually because the operator narrowed ``SOG_RESULTS_ROOTS`` since it was
    issued. Falling back there would hand the reader the same-named run under whichever root is
    walked first, under the heading of the run they asked for; refusing lets the caller say the
    honest "no run by that name" instead. Digests are taken over resolved roots, so a re-spelt
    root (trailing slash, relative, symlink) still matches and only a genuinely dropped one does not.
    """
    candidates = normalized_roots(roots)
    wanted = str(key).strip() if key is not None else ""
    if wanted:
        candidates = [root for root in candidates if root_key(root) == wanted]
    for root in candidates:
        target = safe_subpath(root, rel)
        if target is not None and is_run_dir(target):
            return root, target
    return None
