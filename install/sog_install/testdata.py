"""
Stage a mini REAL dataset into an isolated data root for the Tier-2 agent probe.

The real-data test run must exercise a genuine analysis, but it must never touch the
user's ``./data`` or the agent's real data lake. This module copies one small, local,
already-validated mini dataset into a throwaway root UNDER the per-run artifact dir,
laid out exactly the way :class:`~spatialomicsgym.agent.STCoscientist` expects
(``stcoscientist.py:178-216``)::

    <root>/spatialomicsgym_data/data_lake/<name>.h5ad

The agent is then constructed with ``STCoscientist(path=<root>, expected_data_lake_files=[])``
so the constructor SKIPS its S3 download and finds the staged file locally. ``<root>``
lives under :func:`constants.artifact_dir` (``SOG_SETUP_ARTIFACT_DIR``-redirectable in
tests), so cleanup removes it and the :func:`constants.assert_deletable_artifact` guard
already covers it — a stage can never write outside the artifact dir.

Stdlib only.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import constants

if TYPE_CHECKING:
    from pathlib import Path

# Default staged dataset: the local mini Visium slide validated end-to-end through the
# real SpatialDE worker by the smoke harness (evidence A in the plan). Small, local, no
# network. Override with any other ``test/test_data/mini_*.h5ad`` via ``name=``.
DEFAULT_DATASET = constants.MINI_SPATIAL  # "mini_spatial.h5ad"
DEFAULT_SC_REF = constants.MINI_SC_REF  # "mini_sc_ref.h5ad"
DEFAULT_IMAGE = constants.MINI_VISIUM_HE  # "mini_visium_he.h5ad"
STAGE_SUBDIR = "agent_probe_data"  # sits directly under the per-run artifact dir


@dataclass(frozen=True)
class StagedData:
    """Where a staged mini dataset landed.

    ``root`` is the value to pass as ``STCoscientist(path=...)``; ``data_lake`` is the
    ``<root>/spatialomicsgym_data/data_lake`` dir; ``h5ad`` is the staged file; ``filename``
    is its basename (the token the probe task names so the agent uses the real input).

    ``sc_ref`` is the matching single-cell reference staged in the SAME data lake — set only
    when a deconvolution/mapping run asks for it (``with_sc_ref=True`` and a reference is on
    disk); ``None`` otherwise. Deconvolution tools (cell2location, tangram, RCTD, …) need this
    second input, so the demo/probe names it alongside the slide.

    ``has_image`` is ``True`` when the staged ``h5ad`` is the Visium + H&E slide (``image=True`` and
    an image fixture resolved) rather than the plain expression slide. Image tools (cell/nucleus
    segmentation) need a histology image, so their demo names the H&E slide; ``False`` otherwise.
    """

    root: Path
    data_lake: Path
    h5ad: Path
    filename: str
    sc_ref: Path | None = None
    has_image: bool = False

    def to_dict(self) -> dict:
        d = {"root": str(self.root), "h5ad": str(self.h5ad), "filename": self.filename}
        if self.sc_ref is not None:
            d["sc_ref"] = str(self.sc_ref)
        if self.has_image:
            d["has_image"] = True
        return d


def source_dataset(name: str = DEFAULT_DATASET) -> Path:
    """Absolute path to the requested local mini source dataset (``test/test_data/<name>``).

    This is the *requested* path, which may not exist (the mini files are gitignored, and ``test/``
    is absent on a fresh clone) — use :func:`resolve_source` to get the best *available* dataset
    (with the demo fallback).
    """
    return constants.mini_data_dir() / name


def resolve_source(name: str = DEFAULT_DATASET) -> Path | None:
    """The best available local source for the real-data probe, or ``None`` if none exists.

    Prefer the requested ``test/test_data/<name>`` (the ~50 MB validated slide, present on
    a full development tree). When it is absent — ``mini_*.h5ad`` are gitignored — fall back to
    the small spatial AnnData in ``test/test_data/creation_demo``
    (:data:`constants.FALLBACK_SPATIAL_REL`) so the run still exercises a genuine dataset.
    """
    primary = source_dataset(name)
    if primary.is_file():
        return primary
    fallback = constants.fallback_spatial_dataset()
    if fallback.is_file():
        return fallback
    return None


def resolve_sc_ref(name: str = DEFAULT_SC_REF) -> Path | None:
    """The best available local single-cell reference for the Tier-1 deconv/mapping cases,
    or ``None`` if none exists.

    Mirrors :func:`resolve_source`: prefer the requested ``test/test_data/<name>`` (the
    gitignored ``mini_sc_ref.h5ad``, present on a full development tree); when it is absent,
    fall back to ``creation_demo/demo_sc_ref.h5ad``
    (:data:`constants.FALLBACK_SC_REF_REL`), which carries the ``louvain``/``Sample`` obs keys
    those cases reference. Kept separate from :func:`resolve_source` because the spatial and
    single-cell files fall back independently.
    """
    primary = constants.mini_data_dir() / name
    if primary.is_file():
        return primary
    fallback = constants.fallback_sc_ref_dataset()
    if fallback.is_file():
        return fallback
    return None


def resolve_image_dataset(name: str = DEFAULT_IMAGE) -> Path | None:
    """The best available local Visium + H&E slide for the image/segmentation cases, or ``None``.

    Mirrors :func:`resolve_source` / :func:`resolve_sc_ref`: prefer the requested
    ``test/test_data/<name>`` (the gitignored ``mini_visium_he.h5ad``, present on a full development
    tree); when it is absent, fall back to ``creation_demo/demo_visium_he.h5ad``
    (:data:`constants.FALLBACK_IMAGE_REL`), which carries ``uns['spatial']`` (hires + lowres RGB) +
    the Visium scalefactors segmentation tools read. Falls back INDEPENDENTLY of the spatial/sc-ref
    slides, so a checkout can have a real mini expression slide but still use the demo image."""
    primary = constants.mini_data_dir() / name
    if primary.is_file():
        return primary
    fallback = constants.fallback_image_dataset()
    if fallback.is_file():
        return fallback
    return None


def stage_root() -> Path:
    """The isolated data root under the per-run artifact dir (never the user's ``./data``)."""
    return constants.artifact_dir() / STAGE_SUBDIR


def _copy_if_needed(src: Path, dst: Path, *, force: bool) -> None:
    """Idempotent copy: skip when a same-size file is already staged (unless ``force``)."""
    if force or not dst.is_file() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)


def stage_mini_dataset(
    name: str = DEFAULT_DATASET,
    *,
    force: bool = False,
    dest_root: Path | None = None,
    with_sc_ref: bool = False,
    image: bool = False,
) -> StagedData:
    """Copy the best available mini dataset into an isolated data-lake root; return where it landed.

    Uses :func:`resolve_source`, so a tree without the gitignored ``mini_*.h5ad`` transparently
    stages the creation_demo dataset instead. The staged file keeps its *real* basename — the
    probe task names that basename, so the agent references the file that is actually present.
    Idempotent: an already-staged file of the same size is reused unless ``force`` is set. Raises
    ``FileNotFoundError`` when neither the requested dataset nor the demo fallback exists (its
    message says so when the whole ``test/`` tree is absent); the
    ``mkdir``/``copy2`` staging step can also raise other ``OSError`` (``PermissionError``, ENOSPC).
    Both are staging failures the caller turns into a Tier-2 ``SKIP``, never a hard failure — so
    every caller guards this with ``except OSError`` (which subsumes ``FileNotFoundError``).

    ``dest_root`` overrides where the data lake is created. The default (``None``) uses
    :func:`stage_root` under the per-run artifact dir — the throwaway root the Tier-2 probe wants
    (cleanup removes it). The Part-D demo/real-run passes a **durable** root under ``state_dir()``
    instead, so a key-absent script deferred to "run later" still finds its dataset after cleanup.

    ``with_sc_ref`` also stages a matching single-cell reference (:func:`resolve_sc_ref`) into the
    SAME data lake and records it on :attr:`StagedData.sc_ref`. Deconvolution/mapping tools
    (cell2location, tangram, RCTD, …) need that second input, so their demo names both files.
    Best-effort: if no reference resolves, the slide still stages and ``sc_ref`` stays ``None`` —
    the agent then honestly reports it can't find a reference rather than the run hard-failing.

    ``image`` stages the Visium + H&E slide (:func:`resolve_image_dataset`) as the run's dataset
    instead of the plain expression slide, and sets :attr:`StagedData.has_image`. Image tools
    (cell/nucleus segmentation) need a histology image. Best-effort: if no image fixture resolves,
    it falls back to the plain slide with ``has_image=False`` — the run still proceeds and the agent
    reports honestly. ``with_sc_ref`` and ``image`` are independent, but the wizard only ever asks
    for one (the two categories are disjoint).
    """
    img_src = resolve_image_dataset() if image else None  # image run: prefer the H&E slide
    src = img_src if img_src is not None else resolve_source(name)  # else fall back to the plain slide
    if src is None:
        why = constants.mini_data_note()
        raise FileNotFoundError(
            f"no mini dataset available: neither {source_dataset(name)} nor the demo fallback "
            f"{constants.fallback_spatial_dataset()} exists" + (f" ({why})" if why else "")
        )
    staged_image = img_src is not None  # True only when the H&E slide is what we actually staged
    dst_name = src.name  # honor the file we actually found (mini_spatial / demo_spatial / demo_visium_he)
    root = dest_root if dest_root is not None else stage_root()
    data_lake = root / "spatialomicsgym_data" / "data_lake"
    data_lake.mkdir(parents=True, exist_ok=True)
    dst = data_lake / dst_name
    _copy_if_needed(src, dst, force=force)
    sc_dst: Path | None = None
    if with_sc_ref:
        sc_src = resolve_sc_ref()
        if sc_src is not None:
            sc_dst = data_lake / sc_src.name
            _copy_if_needed(sc_src, sc_dst, force=force)
    return StagedData(
        root=root, data_lake=data_lake, h5ad=dst, filename=dst_name, sc_ref=sc_dst, has_image=staged_image
    )
