"""Running the engine when the model forgot to.

``run_post_analysis`` is a tested engine with, until now, no caller. The only thing that invoked it
was a paragraph of *prompt* (``stcoscientist._enrich_prompt_with_post_analysis``) suggesting the
model write the call itself. That is a suggestion, and models decline suggestions: across the whole
checkout there was not one ``manifest.json`` outside the test fixtures. Everything downstream --
L2's review, the ``report.html`` page, the results portal, the figures the web UI wants to show
inline -- is keyed on a manifest, so all of it was dead code waiting on an LLM's goodwill.

This module closes that loop from the other end. After a turn finishes, if no manifest was written,
the agent looks for output the turn *did* write and analyses it itself. The prompt block stays: when
the model does call the engine, its results dir is found by L2 and this never fires. The two paths
converge on the same artifact, so nothing downstream has to know which one produced it.

The whole risk here is picking the wrong directory, because the engine writes a ``post_analysis/``
subdirectory into whatever it is pointed at. So discovery is a *narrow allowlist* -- recent, non-
empty, in a format the engine reads -- crossed with the same exclusions the rest of the layer
already enforces: never ``benchmarks/results/`` (non-negotiable 1a), never a staged input, never our
own previous output, never the staged data corpus, never a copy of the input a conversion utility
wrote. When nothing passes, nothing runs; a turn that answered a question in prose is the common
case, not an error.

Stdlib only at import. Everything heavy (the engine, and therefore matplotlib) is imported inside
:func:`analyse_new_outputs` after discovery has found something worth the cost -- contract
non-negotiable 3 forbids the 1.6 GB agent env from paying for a plotting stack it will not use.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from pathlib import Path

from .tables import TABULAR_SUFFIXES_WITH_COMPRESSION

logger = logging.getLogger(__name__)

__all__ = ["discover_unanalysed_outputs", "analyse_new_outputs"]

#: The one format the pickers open that is not a table. ``sources``, ``detect``, ``clustering``,
#: ``deconvolution`` and ``svg`` each match it by name; there is no shared constant to import, so it
#: is spelled here and the tests below assert the two spellings still agree.
_ANNDATA_SUFFIX = ".h5ad"

#: What the engine can actually read. An *allowlist*, unlike :func:`sources.collect_files`, which
#: denies a handful of suffixes and accepts the rest. The difference is which mistake each side can
#: afford: ``collect_files`` has already been told this directory is a result, so a stray file is
#: noise in a scan; here a stray file is the entire reason a directory gets analysed at all, and a
#: ``.log`` next to a ``config.yaml`` would be enough to publish a report about a scratch folder.
#:
#: Derived, not restated. Hand-listing it drifted: it grew ``.parquet``, ``.mtx``, ``.xlsx``,
#: ``.loom`` and ``.h5``, none of which any picker in this layer opens. Nominating one of those is
#: worse than missing it: the engine writes ``post_analysis/`` into the directory it is pointed at,
#: so a turn whose tool wrote a single unreadable file ended with a *failed* manifest in the user's
#: own output folder and a red card in the results portal for a run that succeeded. ``tables`` is
#: stdlib-only at import, so reading the vocabulary from it costs this module nothing.
#:
#: The compressed spellings are in that vocabulary because the readers below decompress: a
#: ``.csv.gz`` is a table this layer can open, and leaving it out is what made xfuse -- whose only
#: real output is a gzipped 500 x 4 metagene table -- get no post-analysis at all. The list stays a
#: list of *compound* endings rather than gaining a bare ``.gz`` because the match below is
#: ``str.endswith``: ``stacker_warped.nii.gz`` is a NIfTI volume, and a bare ``.gz`` would nominate
#: the directory holding it and hand the volume to ``pd.read_csv``.
_ANALYSABLE_SUFFIXES: tuple[str, ...] = TABULAR_SUFFIXES_WITH_COMPRESSION + (_ANNDATA_SUFFIX,)

#: Path components that are never a tool result, checked case-insensitively per component.
#:
#: ``data_lake`` is the one that matters in practice: staging a dataset writes tens of fresh,
#: perfectly readable ``.h5ad`` files under the agent's own data root, which is also a search root.
#: Without this, "the user asked me to download a dataset" would end with a deconvolution report
#: about the dataset's raw counts.
#:
#: Only the plural ``benchmarks``. It earns its place -- 295 tracked paths hang off it, and it keeps
#: this repo's benchmark machinery (``manuscript/``, ``workflows/``, ``evaluation/``,
#: ``benchmark_data/``) out of discovery, which is wider than the ``benchmarks/results/`` write guard
#: below. The singular sat here too and matched nothing anywhere: no exact ``benchmark`` component in
#: ``git ls-files``, none under the full ``benchmarks/`` tree, none in the work tree. The only thing
#: it could ever exclude was a *user's* folder called ``benchmark`` -- an ordinary name for a folder
#: of outputs one is comparing -- whose results then silently got no report at all.
#:
#: ``rank_cluster_markers`` is where :func:`spatialomicsgym.tool.cluster_comparison.rank_cluster_markers`
#: writes its full per-gene table when it is given no ``output_dir`` -- under the work root, which is
#: a search root. No reader in this layer opens a marker table, and the content route types it
#: anyway, by what is in it, so the same columns (``cluster, rank, gene, score, log2fc, pval_adj``)
#: come back as different results: at 5 genes a cluster, "a non-negative spot x cell-type matrix",
#: a deconvolution result, reviewed ``ok`` (driven 2026-09-27); from the same cells at 25 genes, a
#: table that "ranks gene symbols by 'pval_adj'", an SVG result (driven 2026-09-28). Any of these is
#: a report about an analysis the run never did, so the directory is excluded by name, not by type.
#: The name is spelled here because this module is stdlib-only at import and that one imports
#: pandas; the marker tests assert the two spellings agree.
_NEVER_A_RESULT = frozenset(
    {
        "data_lake",
        "post_analysis",
        "benchmarks",
        ".git",
        ".cache",
        "__pycache__",
        ".ipynb_checkpoints",
        "node_modules",
        "site-packages",
        "rank_cluster_markers",
    }
)

#: Give up rather than walk a data root forever. Four levels is what L2's discovery already uses.
_MAX_DEPTH = 4

#: Directories analysed per turn. A turn can legitimately run two tools; it does not run ten, and an
#: unbounded loop here would turn one bad discovery into a filesystem full of reports.
_MAX_DIRS = 3


def discover_unanalysed_outputs(
    roots, since: float, *, max_depth: int = _MAX_DEPTH, limit: int = _MAX_DIRS, exclude=()
):
    """Directories holding analysable files written after ``since``, newest first.

    Newest first among the folders that qualify outright; a folder admitted only because the
    staging-like name is the turn's own top-level folder (``bass_on_converted/``) comes after them.

    ``since`` is the turn's start time, which is what keeps this from re-analysing the whole history
    of a long-lived data root on every question. A directory qualifies on its *own* files only --
    freshness is not inherited from a child, or staging one file inside a large tree would nominate
    the tree.

    Returns at most ``limit`` directories, and never raises: discovery runs on the success path of a
    finished analysis, so a permission error on some unrelated subtree must cost nothing.

    ``exclude`` is a set of directories the walk must not enter and must never nominate: on the
    portal, every OTHER account's folder under the agent's outputs tree. The agent's data root is
    searched whole (see ``_post_analysis_search_roots``), and that root holds every account's
    per-chat folders -- so without this the engine wrote ``post_analysis/`` reports into folders
    the turn did not own, timed by whoever ran last (driven 2026-09-20).
    """
    from .sources import is_conversion_output, is_staged_input, leads_under_benchmark_results, staged_scan_root

    banned = _resolved_set(exclude)
    newest: dict[Path, float] = {}
    exempted: set[Path] = set()
    for root in roots or ():
        # A blank root is not "no root": ``Path("")`` is ``Path(".")``, which is a real directory,
        # so an empty string in the list quietly nominates the *current working directory* -- the
        # repo root, under a developer checkout -- and the engine starts writing reports into it.
        if root is None or not str(root).strip():
            continue
        try:
            base = Path(str(root).strip()).expanduser()
        except (OSError, ValueError, TypeError):
            continue
        try:
            if not base.is_dir():
                continue
        except OSError:
            continue
        if banned and _inside_any(base, banned):
            continue
        for path in _bounded_walk(base, max_depth, banned):
            name = path.name.lower()
            if not name.endswith(_ANALYSABLE_SUFFIXES):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size == 0 or stat.st_mtime < since:
                continue
            # A converted or repaired copy of the input is fresh, readable and not a result, and on
            # its own it is enough to nominate the folder it was written to. All three E-03 qc_01
            # trials ended that way: ``run_spatial_pipeline`` wrote its copy into ``outputs/``
            # (``repaired_merfish.h5ad``; ``converted_merfish.h5ad`` in r3), nothing else in
            # ``outputs/`` was new, and the engine published the input's own
            # ``cell_type_annot`` as 13 spatial domains, reviewed ``ok``. The MCP ``data_converter``'s
            # output is the same file by another route -- its ``converted.h5ad`` carried the input's
            # cell types into 13 domains the same way (driven 2026-09-25). Asked per file and not
            # per directory, so a tool result written beside the copy still nominates it.
            if is_conversion_output(path):
                continue
            parent = path.parent
            if parent in newest:
                newest[parent] = max(newest[parent], stat.st_mtime)
                continue
            # Staging is judged below the folder the turn made at the top of this root, not along the
            # whole absolute path. Discovery set no scan root, so a result the agent wrote into a
            # ``scanpy_on_converted/`` it named itself -- the recorded live case the engine already
            # tolerates -- was dropped as staged and got no manifest, review or report. A worker's
            # scratch directory inside a tool's output (``starfysh_out/_starfysh_input``) is still
            # below that folder, so it stays excluded (hunt 2026-09-30, u18-postanalysis-3).
            top = _top_level_under(base, parent)
            with staged_scan_root(top) if top is not None else nullcontext():
                if _excluded(parent, is_staged_input, leads_under_benchmark_results):
                    continue
            # Admitted only because its own top-level folder is exempt -- ``converted_input/`` as much
            # as ``bass_on_converted/``. That rule is the decided policy and stands, but such a folder
            # ranks after every folder admitted outright: the converted input the agent wrote last took
            # the place of the run it was converted for (hunt 2026-09-30, rp-u18). Judged below
            # ``base``, so a staging-like name above the searched root costs no folder its rank.
            if top is not None and top != base:
                with staged_scan_root(base):
                    if is_staged_input(parent):
                        exempted.add(parent)
            newest[parent] = stat.st_mtime

    ranked = sorted(newest.items(), key=lambda item: (item[0] in exempted, -item[1], str(item[0])))
    return [directory for directory, _ in ranked[:limit]]


def analyse_new_outputs(roots, since: float, *, limit: int = _MAX_DIRS, exclude=()) -> list[Path]:
    """Run the engine over every directory :func:`discover_unanalysed_outputs` nominates.

    Returns the results directories written, which is empty far more often than not. Each directory
    is analysed independently: the engine already treats one broken figure as a degraded run rather
    than a failed one, and one unreadable *directory* gets the same treatment here.

    Callers are expected to have checked the benchmarking and ``post_analysis_enabled`` gates first,
    but the engine re-checks benchmarking itself (non-negotiable 1b) and would write nothing anyway.
    """
    directories = discover_unanalysed_outputs(roots, since, limit=limit, exclude=exclude)
    if not directories:
        return []

    from . import run_post_analysis
    from .as_owner import foreign_writer, run_module_as

    written: list[Path] = []
    for directory in directories:
        try:
            writer = foreign_writer(directory)
            if writer is None:
                written.append(Path(run_post_analysis(directory)))
                continue
            # Root over a directory the agent can write: analyse it as the agent (see as_owner).
            done = run_module_as(*writer, "spatialomicsgym.postanalysis", [str(directory), "--no-report"])
            lines = [line for line in (done.stdout or "").splitlines() if line.strip()]
            if done.returncode == 0 and lines:
                written.append(Path(lines[-1].strip()))
            else:
                logger.debug(
                    "post-analysis autorun (as uid %s) wrote nothing for %s: %s",
                    writer[0],
                    directory,
                    (done.stderr or "").strip()[-400:],
                )
        except Exception as exc:  # one unanalysable directory must not lose the others
            logger.debug("post-analysis autorun skipped %s: %s: %s", directory, type(exc).__name__, exc)
    return written


# ----------------------------------------------------------------------------------------------
# discovery helpers
# ----------------------------------------------------------------------------------------------


def _resolved_set(paths) -> set[Path]:
    out: set[Path] = set()
    for raw in paths or ():
        try:
            out.add(Path(str(raw)).resolve())
        except (OSError, ValueError, TypeError):
            continue
    return out


def _top_level_under(base: Path, directory: Path) -> Path | None:
    """The directory directly under the search root ``base`` that ``directory`` is in.

    ``base`` itself for a file at the top of the root, and ``None`` -- no scan root, the full-path
    reading -- for a directory the walk did not reach from ``base``.
    """
    try:
        parts = directory.relative_to(base).parts
    except ValueError:
        return None
    return base / parts[0] if parts else base


def _inside_any(path: Path, banned: set[Path]) -> bool:
    try:
        resolved = path.resolve()
    except (OSError, ValueError):
        return False
    return resolved in banned or any(b in resolved.parents for b in banned)


def _bounded_walk(base: Path, max_depth: int, banned: set[Path] = frozenset()):
    """Every file at most ``max_depth`` levels under ``base``, pruning at directories we skip.

    Pruning during the walk rather than filtering afterwards is what keeps this affordable: the
    excluded names (``data_lake``, ``.git``, ``site-packages``) are exactly the enormous subtrees, so
    descending into them to reject their files one at a time is the expensive way to learn nothing.

    A directory holding one of our manifests is pruned the same way, and for the same reason
    :func:`sources.collect_files` drops every file *within* such a directory rather than only the
    directory itself: a report is a tree. The manifest sits at its root, the CSVs sit in ``tables/``,
    and ``tables/`` holds no manifest of its own -- so under any ``results_dir`` the name check does
    not already cover, the previous report's own gene ranking reads as a brand new tool result and
    gets re-published beside the real run. Asking during the walk is also cheaper than asking per
    candidate directory, and ``filenames`` is already in hand, so a directory with no manifest costs
    a list scan and no syscall.
    """
    base_depth = len(base.parts)
    try:
        walker = os.walk(base, followlinks=False, onerror=None)
    except OSError:
        return
    for dirpath, dirnames, filenames in walker:
        current = Path(dirpath)
        depth = len(current.parts) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        else:
            dirnames[:] = [d for d in dirnames if d.lower() not in _NEVER_A_RESULT]
            if banned:
                dirnames[:] = [d for d in dirnames if not _inside_any(current / d, banned)]
        if "manifest.json" in filenames and _holds_a_manifest_of_ours(current):
            dirnames[:] = []
            continue
        for filename in filenames:
            yield current / filename


def _excluded(directory: Path, is_staged_input, leads_under_benchmark_results) -> bool:
    """Whether ``directory`` is something other than a fresh tool result.

    The manifest check is by *content*: a previous run's results directory is recognised by the
    manifest it holds, not by being called ``post_analysis``. That covers the case the name check
    cannot -- a caller who passed an explicit ``results_dir`` somewhere with an ordinary name, which
    would otherwise be analysed back into a report about the last report.

    Only the directory itself is answered for here. Its subtree -- the ``tables/`` the report's CSVs
    are actually in -- is pruned by :func:`_bounded_walk`, which is the half that matters and the
    half :func:`sources.collect_files` has always done. This stays as the answer for a directory
    handed here directly.
    """
    if any(part.lower() in _NEVER_A_RESULT for part in directory.parts):
        return True
    try:
        if leads_under_benchmark_results(directory):
            return True
    except Exception:
        return True  # cannot prove it is safe to write here, so do not
    try:
        if is_staged_input(directory):
            return True
    except Exception:
        pass
    return _holds_a_manifest_of_ours(directory)


def _holds_a_manifest_of_ours(directory: Path) -> bool:
    from .manifest import is_our_manifest

    manifest = directory / "manifest.json"
    try:
        if not manifest.is_file():
            return False
    except OSError:
        return False
    try:
        return bool(is_our_manifest(manifest))
    except Exception:
        return False
