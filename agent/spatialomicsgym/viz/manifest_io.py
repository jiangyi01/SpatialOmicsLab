"""The seam that decides whether a figure is ever seen.

This is not a convenience wrapper. A figure-only output directory **never reaches the chat at
all**: the post-analysis engine nominates a directory for review only when it holds a table or an
AnnData object, so a directory containing nothing but PNGs is never nominated, no manifest is
written, and the chat attachment resolves to nothing. The tool returns success, its payload lists
the image it wrote, and the answer bubble is empty. Nothing raises, nothing is logged, and the
debugging goes to the wrong layer.

So every producing pipeline in this toolkit writes its own manifest, here, and two properties of
that are fixed rather than configurable.

**The results directory name is not a parameter.** The tool-figure fallback -- the path that
rescues a run whose manifest ended up with zero figures -- is keyed on that exact directory name.
Accepting a caller's name would look like a kindness and would silently disable the fallback on
the one day it is needed, which is the day a render step failed.

**The write is skipped under benchmarking**, by the same switch the engine itself reads. A scored
run produces no review artefacts, and this must not be the one thing that does.

The manifest declares figures and the values behind them. It does not declare the figure's spec
sidecar or its alternate exports: those are named inside the spec, because a declared artefact
costs a slot in a four-figure chat card and an entry in a per-figure existence check.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

#: What a figure entry needs from a caller. Kept as a plain dict rather than a dataclass so a
#: worker can build one without importing this module's types.
#: ``{"path": "figures/x.png", "title": ..., "caption": ..., "kind": ..., "rows": int|None}``
FigureEntry = dict[str, Any]


def benchmarking_is_on() -> bool:
    """The canonical gate, in the shape every other opt-in behaviour in this repository uses.

    Unreadable configuration answers *yes*, because the failure direction that keeps a scored run
    byte-identical to what it was before this toolkit existed is the one that writes nothing.
    """
    if str(os.environ.get("SOG_BENCHMARKING_ENABLED", "")).strip().lower() in ("1", "true", "yes", "on"):
        return True
    try:
        from spatialomicsgym.config import default_config
    except Exception:
        return True
    return bool(getattr(default_config, "benchmarking_enabled", False))


def results_dir_for(output_dir: str | Path) -> Path:
    """Where the manifest goes. Takes no name: see this module's docstring."""
    from spatialomicsgym.postanalysis.manifest import DEFAULT_RESULTS_DIRNAME

    return Path(output_dir) / DEFAULT_RESULTS_DIRNAME


def figures_dir_for(output_dir: str | Path) -> Path:
    """Where a figure file must be written for the manifest to be able to declare it."""
    return results_dir_for(output_dir) / "figures"


def tables_dir_for(output_dir: str | Path) -> Path:
    return results_dir_for(output_dir) / "tables"


def _existing(output_dir: str | Path) -> dict[str, Any]:
    """What a manifest in this directory already declares, or an empty shape."""
    from spatialomicsgym.report import manifest as _reader

    try:
        loaded = _reader.load(results_dir_for(output_dir))
    except Exception:
        return {}
    return loaded or {}


def publish(
    output_dir: str | Path,
    *,
    tool_name: str,
    figures: list[FigureEntry],
    tables: list[FigureEntry] | None = None,
    findings: list[tuple[str, Any, str]] | None = None,
    warnings: list[str] | None = None,
    source_outputs: list[str] | None = None,
    task_type: str = "visualization",
    status_partial: bool = False,
) -> dict[str, Any]:
    """Declare what was drawn, atomically, so the chat can show it.

    Returns a small report rather than raising: ``{"written": bool, "path": str, "why": str}``.
    A failure here must never lose the figures that were already drawn, and it must never be the
    reason a tool call reports failure -- but it must be *visible*, so the caller puts ``why``
    into its payload warnings.

    ``task_type`` is deliberately outside the post-analysis contract's own vocabulary. That is the
    same position the enrichment tool occupies and it is the right one: it means no engine runner
    will re-derive or second-guess a figure the tool already drew, and the run is recorded as
    partial with one warning rather than mis-typed by content.
    """
    report: dict[str, Any] = {"written": False, "path": "", "why": ""}
    if benchmarking_is_on():
        report["why"] = "benchmarking is on; no review artefacts are written on a scored run"
        return report
    if not figures:
        report["why"] = "no figure was drawn, so there is nothing to declare"
        return report
    try:
        from spatialomicsgym.postanalysis.manifest import Manifest, write_manifest
    except Exception as exc:
        report["why"] = (
            f"the manifest writer is not importable here ({type(exc).__name__}); the figures are on disk but the chat will not show them"
        )
        return report

    # Merge, never replace. Several figures in one turn land in one output directory, and a turn
    # attaches exactly one run -- so a manifest that is rewritten from scratch each time leaves
    # every earlier figure on disk, undeclared, and invisible in the chat. The manifest's own
    # writer already replaces an entry that names the same path, so re-declaring is idempotent.
    existing = _existing(output_dir)
    # The run's own identity is merged too. A figure drawn into an ANALYSIS tool's output directory
    # rebuilt the manifest as this figure's, so the run lost its findings (n_domains, ari), was
    # retyped "visualization" and went from ok to partial (hunt 2026-09-30, u20b-viz-rest-12). A
    # manifest some other task wrote keeps its tool, type and status; this figure is added to it.
    foreign = bool(existing) and str(existing.get("task_type") or "") not in ("", "unknown", task_type)
    if foreign:
        manifest = Manifest(tool_name=existing.get("tool_name") or tool_name, task_type=existing.get("task_type"))
        if str(existing.get("status") or "") in ("ok", "partial", "failed"):
            manifest.status = str(existing["status"])
    else:
        manifest = Manifest(tool_name=tool_name, task_type=task_type)
        # Say, in L1's own words, that this task type has no handler here -- otherwise the report
        # reads an unrecognised task type as a CONTRACT VIOLATION and prints "(not a contract task
        # type)" under a run that did exactly what it was asked. The sentence is imported rather than
        # respelled, because a second spelling stops being recognised the first time either is
        # reworded and the drift is silent.
        try:
            from spatialomicsgym.postanalysis.manifest import TASK_TYPES, no_handler_warning

            if task_type not in TASK_TYPES:
                manifest.warn(
                    no_handler_warning(
                        task_type,
                        "the figure and its record ARE the result; there is nothing here for a task runner to re-derive",
                    )
                )
                manifest.degrade()
        except Exception:
            pass
    for entry in existing.get("findings", []):
        try:
            manifest.add_finding(str(entry["key"]), entry.get("value"), str(entry.get("label") or ""))
        except Exception:
            continue
    for entry in existing.get("figures", []):
        try:
            manifest.add_figure(
                entry["path"], entry.get("title", ""), entry.get("caption", ""), entry.get("kind", "scatter")
            )
        except Exception:
            continue
    for entry in existing.get("tables", []):
        try:
            manifest.add_table(entry["path"], entry.get("title", ""), int(entry.get("rows") or 0))
        except Exception:
            continue
    for message in existing.get("warnings", []):
        manifest.warn(str(message))
    for item in dict.fromkeys([*(existing.get("source_outputs") or []), *(source_outputs or [])]):
        manifest.source_outputs.append(str(item))
    for entry in figures:
        try:
            manifest.add_figure(
                str(entry["path"]),
                str(entry.get("title") or ""),
                str(entry.get("caption") or ""),
                str(entry.get("kind") or "scatter"),
            )
        except Exception as exc:
            manifest.warn(f"a figure could not be declared ({type(exc).__name__}: {exc})")
    for entry in tables or []:
        try:
            manifest.add_table(str(entry["path"]), str(entry.get("title") or ""), int(entry.get("rows") or 0))
        except Exception:
            pass
    for key, value, label in findings or []:
        try:
            manifest.add_finding(str(key), value, str(label))
        except Exception:
            pass
    for message in warnings or []:
        manifest.warn(str(message))
    if status_partial:
        manifest.degrade()

    try:
        path = write_manifest(results_dir_for(output_dir), manifest)
    except Exception as exc:
        report["why"] = f"the manifest could not be written ({type(exc).__name__}: {exc})"
        return report
    report.update({"written": True, "path": str(path)})
    return report


def declared_figure_count(output_dir: str | Path) -> int:
    """How many figures the manifest at *output_dir* declares, or -1 if there is no manifest.

    Used by the smoke driver: "the tool exited zero" is not the check that matters here, because
    the failure this module exists to prevent leaves a zero exit and an empty answer.
    """
    from spatialomicsgym.report import manifest as _reader

    try:
        loaded = _reader.load(results_dir_for(output_dir))
    except Exception:
        return -1
    if not loaded:
        return -1
    return len(loaded.get("figures") or [])
