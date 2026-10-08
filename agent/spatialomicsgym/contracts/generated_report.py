"""Recognising a report this system generated *about* a tool's output, rather than the output itself.

Two layers ask this question and neither may own it:

* the post-analysis package writes the report, and skips a directory holding one so that a second
  run does not read the first run's findings table back as a tool result;
* the benchmarking package must not score one as a tool's prediction, and must not copy one into a
  tool's output directory as recovered output.

The rule lives here, in ``contracts/``, outside both packages and importing nothing but the standard library,
because the dependency between those two runs strictly one way and is fenced in both directions by
``test/test_postanalysis_engine.py::TestEvalNeutrality``: the benchmarking package may not reach
the post-analysis package at all, and the post-analysis package may not reach the scoring path. A
marker that belongs to neither satisfies both fences without weakening either -- recognising a file
marker gives the scoring side no way to *run* an analysis -- and keeps the rule in one place instead
of re-derived on each side, which is the drift this repo keeps paying for.

That fence was written against code coupling. The leak it did not cover was through the filesystem:
the report is written *inside* the directory the scorer is later pointed at, so post-analysis output
was being picked up as the tool's prediction with no import involved anywhere. Measured over this
repo's own recorded outputs, 36 of the 92 run directories holding a report had the prediction picked
from under it.

Decided by content, never by name, in both directions. ``manifest.json`` is not this package's name
to reserve -- a tool is free to write one, and judging by name would drop that tool's real output --
while the results directory is a caller-supplied argument, so a report written under any directory
name is still ours.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

#: A manifest we wrote is a few KB. Past this, do not read the file to ask whether it is one --
#: something else is using the name and the answer is no either way.
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024


def is_generated_report_payload(payload: object) -> bool:
    """Whether an already-parsed ``manifest.json`` payload is one this system wrote.

    Split out so a caller holding the parsed payload can ask without reading the file a second time.
    That was the reason the reviewer had grown a *local* spelling of the question instead of
    importing this one, and the two spellings did not agree. ``Manifest.to_dict`` emits
    ``schema_version``, ``tool_name`` and ``task_type`` unconditionally, so ``"tool_name" in data``
    matched on every manifest this system writes -- all 92 recorded runs -- and parted company only
    on somebody else's ``manifest.json``. There it read a foreign file that merely named a tool as a
    report of ours, and since ``_ensure_post_analysis_ran`` treats a hit as "the model already called
    the engine", the turn returned before autorun and produced no analysis at all.
    """
    return isinstance(payload, dict) and "schema_version" in payload and "task_type" in payload


def is_generated_report(path: Path) -> bool:
    """Whether ``path`` is a manifest *this system* wrote -- read, not inferred from its name.

    ``schema_version`` and ``task_type`` together are what the manifest writer always emits.
    :func:`is_generated_report_payload` asks the same question of a payload already in hand.
    """
    try:
        if Path(path).stat().st_size > _MAX_MANIFEST_BYTES:
            return False
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return False
    return is_generated_report_payload(payload)


def declared_artifact_paths(path: Path) -> list[str]:
    """The results-dir-relative paths a manifest records having written, plus its own filename.

    Needed for the one layout that cannot be dropped wholesale: the results directory is a
    caller-supplied argument, so it may be the tool's output directory itself. Our report is then
    interleaved with the tool's files in one directory, and dropping everything beneath the manifest
    would drop the tool's prediction too.

    A manifest already knows what it wrote -- the writer validated every one of these on the way in,
    so they are relative, ``..``-free and inside the results dir. Reading them back is exact where a
    name heuristic would be a guess.

    Returns ``[]`` for anything unreadable; the caller then keeps the file.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    declared = [Path(path).name]
    for key in ("figures", "tables"):
        for entry in payload.get(key) or []:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                declared.append(entry["path"])
    return declared


# What ``report.render.write_report`` names the page by default. Spelled here, not imported: this module
# sits on the benchmark gate's path, which does not import the renderer.
REPORT_FILENAME = "report.html"

# ``postanalysis.manifest.staging_name``'s shape -- ``.<name>.[<tag>.]<pid>.<n>.<16 hex>.tmp`` -- and
# the fixed names it replaced (``manifest.json.tmp``, ``manifest.json.l2tmp``, ``.report.html.tmp``).
_STAGING_NAME = re.compile(
    r"^(?:\..+\.(?:[A-Za-z0-9_-]+\.)?\d+\.\d+\.[0-9a-f]{16}\.tmp"
    r"|manifest\.json\.(?:tmp|l2tmp)|\.report\.html\.tmp)$"
)


def split_report_files(root: Path, files: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    """Split ``files`` into ``(the tool's, ours)``.

    ``root`` is the directory being scanned, and matters only for the degenerate layout above: a
    manifest found *at* ``root`` describes a report sharing the directory with the tool's output, so
    only what it declares is ours. A manifest found below ``root`` owns its whole directory, which
    is what the default ``<tool output dir>/post_analysis/`` layout produces.
    """
    root = Path(root)
    files = list(files)
    manifests = [f for f in files if Path(f).name == "manifest.json" and is_generated_report(f)]
    if not manifests:
        return files, []

    subtrees = [f.parent for f in manifests if f.parent != root]
    rooted_here = {(root / rel).resolve() for f in manifests if f.parent == root for rel in declared_artifact_paths(f)}
    manifest_at_root = any(f.parent == root for f in manifests)
    if manifest_at_root:
        # The page render writes beside its manifest is not in the manifest, and neither is a
        # staging file a killed writer left; both were counted as the tool's own output in this
        # layout, so the gate's n_files and file_types included our report (u17-cli-report-25).
        rooted_here.add((root / REPORT_FILENAME).resolve())

    theirs: list[Path] = []
    ours: list[Path] = []
    for f in files:
        staged_here = manifest_at_root and f.parent == root and _STAGING_NAME.match(f.name) is not None
        if staged_here or f.resolve() in rooted_here or any(d == f.parent or d in f.parents for d in subtrees):
            ours.append(f)
        else:
            theirs.append(f)
    return theirs, ours
