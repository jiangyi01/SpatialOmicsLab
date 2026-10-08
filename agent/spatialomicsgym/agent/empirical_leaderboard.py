"""Empirical leaderboard for STCoscientist tool routing.

Aggregates benchmark results (agent + manual + multi-LLM) into a (task, platform, tool)
score table. Used by recommend_analysis_tools to pick the empirically-best-compatible
tool for the user's data, rather than relying on hand-curated heuristic priorities.

Data sources (all already on disk):
  - /workspace/hands_by_myself/results/_unified_domain_results.csv  (ARI for clustering)
  - /workspace/hands_by_myself/results/_unified_deconv_results.csv  (mean_pearson_r for deconv)
  - <pkg>/data/unified_svg_hvg_filtered.csv (F1_hvg for SVG, canonical; tracked, and shipped in the
    wheel through MANIFEST.in's data/*.csv)
  - /workspace/hands_by_myself/results/_unified_svg_results.csv     (F1 fallback if not in canonical)

Output: spatialomicsgym/data/empirical_leaderboard.json

Only cells ``recommend_analysis_tools`` can act on are kept: a tool its ``_MCP_TOOLS`` catalogue
offers for that task. The sources also score wired tools the recommender never offers (``run_bass``
topped Visium clustering) and whole competitor agents (``biomni_no_mcp``, ``dsgym``, side
``other_agent``); those are listed under ``_meta.dropped_cells`` with the reason, and
``lookup_score`` refuses them in a table built before this rule (hunt 2026-09-30,
u15-validation-12).

CLI:
    python -m spatialomicsgym.agent.empirical_leaderboard --refresh   # rebuild cache
    python -m spatialomicsgym.agent.empirical_leaderboard --show      # top-3 per cell + drift check

``--show`` follows the table with a read-only comparison against the source CSVs
(``compare_cache_to_sources``), which reports per task whether the shipped table still matches its
sources — or, where those sources are not on this machine, that the question could not be answered.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Score key per task — fixed; switching is a one-line change here.
_SCORE_KEYS: dict[str, str] = {
    "spatial_clustering": "ARI",
    "svg_detection": "F1_hvg",
    "deconvolution": "mean_pearson_r",
}

# Minimum cells required for a tool to enter the leaderboard.
_MIN_CELLS = 2

# Source rows that score a whole competitor agent on the same instances rather than a tool.
_NON_TOOL_SIDES = frozenset({"other_agent"})

# Dataset name → canonical platform. Copied from the benchmark dataset registry
# (benchmarks/benchmark_data/registry.yaml), which is no longer part of the repository.
_DATASET_PLATFORM: dict[str, str] = {
    "merfish_spatial_domain": "MERFISH",
    "visium_dlpfc_domain": "Visium",
    "merfish_svg": "MERFISH",
    "visium_svg": "Visium",
    "merfish_deconvolution": "MERFISH",
    "visium_deconvolution": "Visium",
    "slide_seqv2_aorta_domain": "Slide-seqV2",
    "slide_seqv2_rca_domain": "Slide-seqV2",
    "slide_seqv2_aorta_svg": "Slide-seqV2",
    "slideseqv2_aorta_deconvolution": "Slide-seqV2",
    "slideseqv2_rca_deconvolution": "Slide-seqV2",
    "mini_merfish_clustering": "MERFISH",
    "mini_merfish_svg": "MERFISH",
    "mini_merfish_deconv": "MERFISH",
    "mini_visium_clustering": "Visium",
    "mini_visium_svg": "Visium",
    "mini_visium_deconv": "Visium",
}

# Default file locations — overridable for testing. ``_REPO_ROOT`` is the agent part (``<repo>/agent``), which
# holds ``benchmarks/``; the name predates the re-layout, when the package sat at the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "empirical_leaderboard.json"
# The two unified trees are external results directories: not part of the repo, so not derivable.
# The canonical SVG table is repo-internal and therefore is derived. It is the tracked copy inside
# the package (MANIFEST.in includes data/*.csv), so a checkout and a wheel install read the same
# bytes; the manuscript tree that once held the published original is no longer part of the
# repository. It must stay a tracked path: an earlier revision pointed at an untracked local copy,
# so on any checkout but the one that wrote it, the SVG quarter of the routing table was built from
# nothing.
_DOMAIN_CSV = Path("/workspace/hands_by_myself/results/_unified_domain_results.csv")
_DECONV_CSV = Path("/workspace/hands_by_myself/results/_unified_deconv_results.csv")
_SVG_CANONICAL_CSV = Path(__file__).resolve().parents[1] / "data" / "unified_svg_hvg_filtered.csv"
# The packaged copy, the same file as the canonical above. It keeps its own name so
# ``_svg_canonical_source`` can still be pointed at two different files; never edited by hand.
_SVG_PACKAGE_CSV = Path(__file__).resolve().parents[1] / "data" / "unified_svg_hvg_filtered.csv"
_SVG_FALLBACK_CSV = Path("/workspace/hands_by_myself/results/_unified_svg_results.csv")


def _svg_canonical_source() -> Path:
    """The path SVG scores are read from: the canonical table, or the packaged copy of it.

    The canonical wins whenever it exists; the packaged copy answers only where it is absent. As
    shipped the two name the same tracked file, so a checkout and a wheel install both read it.
    When neither exists the canonical path is returned so the caller's warning names the primary
    location. The externals under /workspace/hands_by_myself/ are a different axis (cells the
    canonical *misses*) and stay with ``_SVG_FALLBACK_CSV``.
    """
    if _SVG_CANONICAL_CSV.exists():
        return _SVG_CANONICAL_CSV
    if _SVG_PACKAGE_CSV.exists():
        return _SVG_PACKAGE_CSV
    return _SVG_CANONICAL_CSV


def _normalize_tool_name(name: str) -> str:
    """Strip wrapper prefixes and unify naming."""
    n = name.strip()
    return n


def _dataset_to_platform(dataset: str) -> str | None:
    """Map a dataset name to a platform; returns None if unknown."""
    return _DATASET_PLATFORM.get(dataset)


def _parse_float(s: Any) -> float | None:
    if s is None or s == "" or (isinstance(s, str) and s.lower() in {"none", "null", "nan"}):
        return None
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _load_clustering_rows() -> list[dict]:
    """Load clustering benchmark rows from the unified domain CSV.

    Returns rows of the form {task, platform, tool, side, score}.
    """
    rows = []
    if not _DOMAIN_CSV.exists():
        logger.warning("Clustering unified CSV not found at %s", _DOMAIN_CSV)
        return rows
    with _DOMAIN_CSV.open() as f:
        for r in csv.DictReader(f):
            if r.get("status") != "evaluated":
                continue
            platform = _dataset_to_platform(r["dataset"])
            if platform is None:
                continue
            score = _parse_float(r.get("ARI"))
            if score is None:
                continue
            rows.append(
                {
                    "task": "spatial_clustering",
                    "platform": platform,
                    "tool": _normalize_tool_name(r["tool"]),
                    "side": r.get("side", "unknown"),
                    "dataset": r["dataset"],
                    "score": score,
                }
            )
    return rows


def _load_deconv_rows() -> list[dict]:
    """Load deconvolution benchmark rows from the unified deconv CSV."""
    rows = []
    if not _DECONV_CSV.exists():
        logger.warning("Deconv unified CSV not found at %s", _DECONV_CSV)
        return rows
    with _DECONV_CSV.open() as f:
        for r in csv.DictReader(f):
            if r.get("status") != "evaluated":
                continue
            platform = _dataset_to_platform(r["dataset"])
            if platform is None:
                continue
            score = _parse_float(r.get("mean_pearson_r"))
            if score is None:
                continue
            rows.append(
                {
                    "task": "deconvolution",
                    "platform": platform,
                    "tool": _normalize_tool_name(r["tool"]),
                    "side": r.get("side", "unknown"),
                    "dataset": r["dataset"],
                    "score": score,
                }
            )
    return rows


def _load_svg_rows() -> list[dict]:
    """Load SVG benchmark rows.

    Prefer the canonical HVG-filtered CSV (per project memory:
    feedback_svg_canonical_uses_hvg_filter), read through ``_svg_canonical_source``, which falls
    back to the packaged copy if the canonical path is absent. Fall back to the legacy unified SVG
    CSV for any (dataset, tool) cells not present in the canonical.
    """
    rows: list[dict] = []
    seen_cells: set[tuple[str, str, str]] = set()  # (dataset, tool, source) keys we've ingested

    # Pass 1 — canonical HVG-filtered CSV (preferred), or its packaged copy off-checkout
    svg_source = _svg_canonical_source()
    if svg_source.exists():
        with svg_source.open() as f:
            for r in csv.DictReader(f):
                if r.get("status") not in {"OK", "evaluated", "ok"}:
                    continue
                platform = _dataset_to_platform(r["dataset"])
                if platform is None:
                    continue
                score = _parse_float(r.get("F1_hvg"))
                if score is None:
                    continue
                source = r.get("source", "unknown")
                tool = _normalize_tool_name(r["tool"])
                key = (r["dataset"], tool, source)
                seen_cells.add(key)
                rows.append(
                    {
                        "task": "svg_detection",
                        "platform": platform,
                        "tool": tool,
                        "side": source,
                        "dataset": r["dataset"],
                        "score": score,
                    }
                )
    else:
        logger.warning("SVG canonical CSV not found at %s (no packaged copy either)", svg_source)

    # Pass 2 — fallback unified CSV for any (dataset, tool) cells the canonical missed
    if _SVG_FALLBACK_CSV.exists():
        # Only add a fallback row if no canonical row for the same (dataset, tool) pair exists.
        canonical_dataset_tool = {(r["dataset"], r["tool"]) for r in rows}
        with _SVG_FALLBACK_CSV.open() as f:
            for r in csv.DictReader(f):
                if r.get("status") != "evaluated":
                    continue
                platform = _dataset_to_platform(r["dataset"])
                if platform is None:
                    continue
                score = _parse_float(r.get("F1"))
                if score is None:
                    continue
                tool = _normalize_tool_name(r["tool"])
                if (r["dataset"], tool) in canonical_dataset_tool:
                    continue
                rows.append(
                    {
                        "task": "svg_detection",
                        "platform": platform,
                        "tool": tool,
                        "side": r.get("side", "unknown"),
                        "dataset": r["dataset"],
                        "score": score,
                    }
                )

    return rows


def _aggregate(rows: list[dict]) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Aggregate raw rows into nested dict[task][platform][tool] → stats.

    Drops tool cells with fewer than _MIN_CELLS observations.
    """
    bucket: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        bucket.setdefault((r["task"], r["platform"], r["tool"]), []).append(r)

    out: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    dropped = 0
    for (task, platform, tool), recs in bucket.items():
        scores = [r["score"] for r in recs]
        if len(scores) < _MIN_CELLS:
            dropped += 1
            continue
        sides = sorted({r["side"] for r in recs})
        datasets = sorted({r["dataset"] for r in recs})
        mean = statistics.fmean(scores)
        stddev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        out.setdefault(task, {}).setdefault(platform, {})[tool] = {
            "score": round(mean, 4),
            "n_cells": len(scores),
            "stddev": round(stddev, 4),
            "sides": sides,
            "datasets": datasets,
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
        }
    if dropped:
        logger.info("Dropped %d (task, platform, tool) cells with n < %d", dropped, _MIN_CELLS)
    return out


def _routable_tools() -> dict[str, frozenset[str]]:
    """task -> the functions ``recommend_analysis_tools`` can offer for it.

    ``_recommend_for_goal`` looks up only ``_MCP_TOOLS`` entries, so a cell for any other name is
    never routed to, however well it scores.
    """
    from spatialomicsgym.tool.transcriptomics_skills import _MCP_TOOLS

    routable: dict[str, set[str]] = {}
    for entry in _MCP_TOOLS.values():
        routable.setdefault(entry["task"], set()).add(entry["mcp_function"])
    return {task: frozenset(names) for task, names in routable.items()}


def _drop_reason(task: str, tool: str, cell: dict[str, Any], routable: dict[str, frozenset[str]]) -> str | None:
    """Why a cell cannot route, or ``None`` when it can."""
    sides = set(cell.get("sides") or ()) if isinstance(cell, dict) else set()
    if sides and sides <= _NON_TOOL_SIDES:
        return "a whole-agent baseline, not a tool"
    if tool not in routable.get(task, ()):
        return "not offered by recommend_analysis_tools for this task"
    return None


def _split_routable(table: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``(kept, dropped)``: the task tables with only routable cells, and each dropped cell with why.

    Keys starting with ``_`` (``_meta``) pass through untouched.
    """
    routable = _routable_tools()
    kept: dict[str, Any] = {}
    dropped: list[dict[str, Any]] = []
    for task, platforms in table.items():
        if task.startswith("_"):
            kept[task] = platforms
            continue
        for platform, tools in platforms.items():
            for tool, cell in tools.items():
                reason = _drop_reason(task, tool, cell, routable)
                if reason is None:
                    kept.setdefault(task, {}).setdefault(platform, {})[tool] = cell
                else:
                    score = cell.get("score") if isinstance(cell, dict) else None
                    dropped.append({"task": task, "platform": platform, "tool": tool, "score": score, "reason": reason})
    return kept, dropped


def build_leaderboard() -> dict[str, Any]:
    """Build the leaderboard dict from all sources. Does not write to disk."""
    rows: list[dict] = []
    rows.extend(_load_clustering_rows())
    rows.extend(_load_deconv_rows())
    rows.extend(_load_svg_rows())

    aggregated, dropped = _split_routable(_aggregate(rows))
    if dropped:
        logger.info("Dropped %d cells recommend_analysis_tools cannot route to (_meta.dropped_cells)", len(dropped))

    return {
        "_meta": {
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_source_rows": len(rows),
            "score_keys_per_task": _SCORE_KEYS,
            "min_cells_threshold": _MIN_CELLS,
            "sources": {
                "clustering": str(_DOMAIN_CSV),
                "deconvolution": str(_DECONV_CSV),
                # The path actually read: the canonical table, or the packaged copy off-checkout.
                "svg_canonical": str(_svg_canonical_source()),
                "svg_fallback": str(_SVG_FALLBACK_CSV),
            },
            "dropped_cells": dropped,
        },
        **aggregated,
    }


def _existing_cache(cache_path: Path) -> dict[str, Any] | None:
    """Return the cache already on disk, or None if there is nothing readable there."""
    try:
        data = json.loads(cache_path.read_text())
    except (ValueError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _task_names(table: dict[str, Any]) -> set[str]:
    """The tasks a leaderboard table answers. Everything else at top level is metadata (``_meta``)."""
    return {k for k in table if not k.startswith("_")}


def write_cache(cache_path: Path = DEFAULT_CACHE_PATH) -> dict[str, Any]:
    """Build the leaderboard and persist it to ``cache_path``. Returns the table now in effect.

    A build is not persisted when it would *lose* ground the existing file already holds. The three
    unified CSVs are absolute paths outside the package, under ``/workspace/hands_by_myself/``, and
    exist only on the box that ran the benchmarks; the canonical SVG table is repo-internal and
    tracked, so it reads everywhere. A clone therefore builds ``svg_detection`` and nothing else,
    while the file it would replace (``spatialomicsgym/data/empirical_leaderboard.json``) is
    *tracked* and holds the empirical rankings ``rank_tools`` routes on for all three tasks. Both
    ``--refresh`` and the rebuild-on-corrupt branch of ``load_cache`` land here, so without this
    guard one stray refresh — or one truncated read — silently deletes two thirds of the shipped
    table, permanently, and routing for those tasks falls back to hand-curated priorities with no
    diagnostic.

    The test is per task rather than on the total row count. Counting rows answered this question
    correctly only while *every* source was external, so that a clone read none of them; once one
    source moved into the repo a clone's build was no longer empty, just incomplete.

    Two builds are written even though they change the file. One that *adds* a task loses nothing,
    so a table already degraded heals as soon as the CSVs are readable again. One whose predecessor
    cannot be parsed also loses nothing knowable — that is how the corrupt-read path recovers. The
    exception is a build with no tasks at all, which is never written over an existing file however
    unreadable that file is: overwriting a corrupt cache with an empty table would make the next
    start read it as valid and stop rebuilding.
    """
    leaderboard = build_leaderboard()
    built = _task_names(leaderboard)
    if cache_path.exists():
        existing = _existing_cache(cache_path)
        lost = sorted(_task_names(existing) - built) if existing is not None else []
        if not built or lost:
            reason = (
                f"is missing {', '.join(lost)}, which it already answers"
                if lost
                else "read no source rows (no benchmark CSV present)"
            )
            logger.warning(
                "Leaderboard rebuild %s; keeping %s unchanged. The paths it needs are recorded under _meta.sources.",
                reason,
                cache_path,
            )
            return existing if existing is not None else leaderboard
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(leaderboard, indent=2, sort_keys=True))
    return leaderboard


_CACHE: dict[str, Any] | None = None


def load_cache(cache_path: Path = DEFAULT_CACHE_PATH, refresh: bool = False) -> dict[str, Any]:
    """Load the leaderboard JSON, building it if missing.

    Args:
        cache_path: Where the JSON cache lives.
        refresh: If True, rebuild the cache even if the file exists.
    """
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE
    if refresh or not cache_path.exists():
        _CACHE = write_cache(cache_path)
    else:
        try:
            _CACHE = json.loads(cache_path.read_text())
        except (ValueError, OSError) as e:
            # A truncated/corrupt cache (or an unreadable file) must not permanently
            # disable the leaderboard — rebuild it from the on-disk source CSVs instead.
            logging.warning("Leaderboard cache %s unreadable (%s); rebuilding.", cache_path, e)
            _CACHE = write_cache(cache_path)
    return _CACHE


def lookup_score(
    task_type: str,
    platform: str,
    tool: str,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> dict[str, Any] | None:
    """Look up a single (task, platform, tool) cell. Returns None if no data.

    Tool matching is exact on the mcp_function name (the same field used in the
    unified CSVs and in `_MCP_TOOLS[*]["mcp_function"]`). A cell the build would drop
    (``_split_routable``) is no data, also in a table built before that rule.
    """
    cache = load_cache(cache_path)
    cell = cache.get(task_type, {}).get(platform, {}).get(tool)
    if cell is None or _drop_reason(task_type, tool, cell, _routable_tools()) is not None:
        return None
    return cell


def rank_tools(
    task_type: str,
    platform: str,
    candidate_tools: list[str],
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> list[dict[str, Any]]:
    """Rank a list of candidate tools by empirical mean score (desc).

    Tools with no benchmark data appear at the end of the list with score=None.
    Caller is responsible for the compatibility filter; this function does NOT
    drop tools based on requirements — it only orders by empirical performance.

    Returns a list of dicts:
        [
          {"tool": "bsp_identify_svg",       "score": 0.385, "n_cells": 5, ...},
          {"tool": "hotspot_spatial_modules", "score": 0.300, "n_cells": 4, ...},
          {"tool": "svgbit_run",              "score": None,  "n_cells": 0},
          ...
        ]
    """
    ranked: list[dict[str, Any]] = []
    for tool in candidate_tools:
        info = lookup_score(task_type, platform, tool, cache_path)
        if info is None:
            ranked.append({"tool": tool, "score": None, "n_cells": 0})
        else:
            ranked.append({"tool": tool, **info})

    def sort_key(d):
        # Tools with score sort by score desc; tools without come after.
        s = d.get("score")
        return (0 if s is not None else 1, -(s or 0.0))

    ranked.sort(key=sort_key)
    return ranked


# Which source files each task is built from. A task can only be compared against its sources
# when *all* of them are readable: SVG takes its scores from the canonical table and then fills
# the cells that table misses from the legacy one, so a rebuild with the fallback absent is a
# different table, not a stale one.
_TASK_SOURCES: dict[str, tuple[Path, ...]] = {
    "spatial_clustering": (_DOMAIN_CSV,),
    "deconvolution": (_DECONV_CSV,),
    "svg_detection": (_SVG_CANONICAL_CSV, _SVG_FALLBACK_CSV),
}


def _top_tool(tools: dict[str, Any]) -> dict[str, Any] | None:
    """The winning tool of one (task, platform) cell, plus whether the win was a tie.

    Ordered by score descending then by name, so the answer does not depend on dict insertion
    order the way ``_show_top`` does. ``tied`` matters for drift: two tools level at the top swap
    places on any rebuild without anything having changed, and reporting that as a flip would be
    a false alarm.
    """
    scored = [(round(v["score"], 4), t) for t, v in tools.items() if isinstance(v, dict) and v.get("score") is not None]
    if not scored:
        return None
    scored.sort(key=lambda st: (-st[0], st[1]))
    best_score = scored[0][0]
    return {
        "tool": scored[0][1],
        "score": best_score,
        "tied": sum(1 for s, _ in scored if s == best_score) > 1,
    }


def compare_cache_to_sources(cache_path: Path = DEFAULT_CACHE_PATH) -> dict[str, Any]:
    """Read-only: report where the shipped routing table disagrees with a fresh build.

    Writes nothing and does not touch the process-wide ``_CACHE`` — ``build_leaderboard`` only
    reads, and the cache is read through ``_existing_cache`` rather than ``load_cache`` precisely
    because ``load_cache`` rebuilds and persists when the file is missing or corrupt.

    The result separates *no drift* from *could not look*, per task. Three of the four source CSVs
    live outside the repo, under ``/workspace/hands_by_myself/``, so on any machine but the one that
    ran the benchmarks most tasks have nothing to compare against. Answering "no drift" there would
    be the more dangerous of the two mistakes: it would report a stale table as current. So a task
    whose sources are not all present gets ``drift: None`` and its missing paths, never ``False``,
    and the top-level ``drift_detected`` is ``None`` when no task could be checked at all.

    Returns a dict with ``cache_readable``, ``cache_built_at``, ``drift_detected``, and a ``tasks``
    map giving, per task: ``sources_available``, ``missing_sources``, ``drift``, and — when the
    comparison ran — ``top_tool_changes``, ``cells_added``, ``cells_removed`` and ``score_changes``.
    """
    existing = _existing_cache(cache_path)
    if existing is not None:
        # Compared as routing sees it: a cell a rebuild drops by rule is not drift in the sources.
        existing = _split_routable(existing)[0]
    rebuilt = build_leaderboard()

    report: dict[str, Any] = {
        "cache_path": str(cache_path),
        "cache_readable": existing is not None,
        "cache_built_at": (existing or {}).get("_meta", {}).get("built_at"),
        "rebuilt_at": rebuilt["_meta"]["built_at"],
        "tasks": {},
    }

    for task in sorted(_SCORE_KEYS):
        sources = _TASK_SOURCES.get(task, ())
        missing = [str(p) for p in sources if not p.exists()]
        entry: dict[str, Any] = {
            "sources": [str(p) for p in sources],
            "missing_sources": missing,
            "sources_available": not missing,
            "drift": None,
        }
        if missing or existing is None:
            if existing is None:
                entry["note"] = "the cache on disk could not be read, so there is nothing to compare"
            else:
                entry["note"] = "not all source CSVs are present on this machine; drift is unknown"
            report["tasks"][task] = entry
            continue

        cached_task = existing.get(task, {})
        rebuilt_task = rebuilt.get(task, {})
        cells_added, cells_removed, score_changes, top_changes = [], [], [], []

        for platform in sorted(set(cached_task) | set(rebuilt_task)):
            cached_tools = cached_task.get(platform, {})
            rebuilt_tools = rebuilt_task.get(platform, {})
            for tool in sorted(set(cached_tools) | set(rebuilt_tools)):
                was, now = cached_tools.get(tool), rebuilt_tools.get(tool)
                if was is None:
                    cells_added.append(f"{platform}/{tool}")
                elif now is None:
                    cells_removed.append(f"{platform}/{tool}")
                elif round(was.get("score", 0.0), 4) != round(now.get("score", 0.0), 4):
                    score_changes.append(
                        {
                            "platform": platform,
                            "tool": tool,
                            "cached": round(was["score"], 4),
                            "rebuilt": round(now["score"], 4),
                            "n_cells_cached": was.get("n_cells"),
                            "n_cells_rebuilt": now.get("n_cells"),
                        }
                    )
            was_top, now_top = _top_tool(cached_tools), _top_tool(rebuilt_tools)
            if was_top and now_top and was_top["tool"] != now_top["tool"]:
                top_changes.append(
                    {
                        "platform": platform,
                        "cached_top": was_top["tool"],
                        "cached_score": was_top["score"],
                        "rebuilt_top": now_top["tool"],
                        "rebuilt_score": now_top["score"],
                        # A flip out of a tie is bookkeeping, not a change in what the data says.
                        "tie": was_top["tied"] or now_top["tied"],
                    }
                )

        entry.update(
            {
                "drift": bool(cells_added or cells_removed or score_changes or top_changes),
                "top_tool_changes": top_changes,
                "cells_added": cells_added,
                "cells_removed": cells_removed,
                "score_changes": score_changes,
            }
        )
        report["tasks"][task] = entry

    checked = [t for t, e in report["tasks"].items() if e["drift"] is not None]
    report["checked_tasks"] = sorted(checked)
    report["unchecked_tasks"] = sorted(t for t in report["tasks"] if t not in checked)
    # None, not False: with nothing checkable there is no evidence either way.
    report["drift_detected"] = any(report["tasks"][t]["drift"] for t in checked) if checked else None
    return report


def _show_drift(cache_path: Path = DEFAULT_CACHE_PATH) -> None:
    """Print the drift report, saying plainly which tasks could not be checked and why."""
    report = compare_cache_to_sources(cache_path)
    print("\n=== Drift: shipped table vs a fresh build ===")
    if not report["cache_readable"]:
        print("  the cache on disk is unreadable — nothing to compare against")
        return
    for task in sorted(report["tasks"]):
        e = report["tasks"][task]
        if e["drift"] is None:
            print(f"  {task}: NOT CHECKED — {e['note']}")
            for path in e["missing_sources"]:
                print(f"      missing: {path}")
            continue
        if not e["drift"]:
            print(f"  {task}: up to date")
            continue
        print(f"  {task}: DRIFTED")
        for c in e["top_tool_changes"]:
            tie = "  (tie — either ordering is consistent with the data)" if c["tie"] else ""
            print(
                f"      top tool on {c['platform']}: {c['cached_top']} ({c['cached_score']:.4f})"
                f" -> {c['rebuilt_top']} ({c['rebuilt_score']:.4f}){tie}"
            )
        for c in e["score_changes"][:5]:
            print(f"      {c['platform']}/{c['tool']}: {c['cached']:.4f} -> {c['rebuilt']:.4f}")
        if len(e["score_changes"]) > 5:
            print(f"      ... and {len(e['score_changes']) - 5} more score changes")
        if e["cells_added"]:
            print(f"      {len(e['cells_added'])} cells only in the rebuild: {', '.join(e['cells_added'][:4])}")
        if e["cells_removed"]:
            print(f"      {len(e['cells_removed'])} cells only in the cache: {', '.join(e['cells_removed'][:4])}")
    if report["drift_detected"] is None:
        print("  verdict: unknown — no task had all of its source CSVs on this machine")
    elif report["drift_detected"]:
        print("  verdict: the shipped table no longer matches its sources; --refresh to rebuild")
    else:
        print(f"  verdict: no drift across {len(report['checked_tasks'])} checked task(s)")


def _show_top(top_n: int = 3) -> None:
    """Print the top-N tools per (task, platform) cell."""
    cache, dropped = _split_routable(load_cache())
    print(f"=== Empirical Leaderboard (built_at={cache['_meta']['built_at']}) ===")
    print(f"    n_source_rows={cache['_meta']['n_source_rows']}, min_cells={cache['_meta']['min_cells_threshold']}")
    if dropped:
        print(f"    {len(dropped)} cells left out: tools recommend_analysis_tools cannot route to, or agent baselines")
    for task in sorted(k for k in cache if not k.startswith("_")):
        print(f"\n# {task}")
        for platform in sorted(cache[task]):
            tools = cache[task][platform]
            ranked = sorted(tools.items(), key=lambda kv: -kv[1]["score"])
            print(f"  {platform}:")
            for i, (tool, info) in enumerate(ranked[:top_n]):
                marker = "P1" if i == 0 else f"P{i + 1}"
                print(
                    f"    {marker}: {tool:42s} score={info['score']:.4f}  n={info['n_cells']}  "
                    f"σ={info['stddev']:.3f}  sides={info['sides']}"
                )


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Build / inspect the empirical leaderboard.")
    ap.add_argument("--refresh", action="store_true", help="Rebuild the cache from source CSVs.")
    ap.add_argument("--show", action="store_true", help="Print top-3 tools per (task, platform).")
    ap.add_argument("--top", type=int, default=3, help="N for --show.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.refresh:
        cache = write_cache()
        # Reports the table now in effect rather than claiming a write: write_cache declines to
        # persist a build that found no sources, and says so on the logger above.
        meta = cache.get("_meta", {})
        print(f"Leaderboard cache in effect at {DEFAULT_CACHE_PATH}")
        print(f"  n_source_rows: {meta.get('n_source_rows', 'unknown')}")
        for task in sorted(k for k in cache if not k.startswith("_")):
            n_tools = sum(len(v) for v in cache[task].values())
            print(f"  {task}: {len(cache[task])} platforms, {n_tools} tool cells")

    if args.show or not args.refresh:
        _show_top(args.top)
        _show_drift()


if __name__ == "__main__":
    _cli()
