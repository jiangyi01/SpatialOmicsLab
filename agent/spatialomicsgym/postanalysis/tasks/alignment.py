"""Alignment: whether it helped, measured on the same pairs and the same metrics as the diagnosis.

Until 2026-09-21 ``alignment`` routed to the shallow runner, whose own table said what a deep one
would add: "landmark residuals, before/after overlays and a per-section registration error". This
is that runner, and it gets those from
:mod:`spatialomicsgym.spatial3d` rather than growing a second implementation -- a before/after
comparison whose two halves were written separately quietly stops being one.

**The finding that matters is per pair, not on average.** A recorded run in this repository
reported a mean pairwise improvement of 3922 and called itself an alignment, over a transform
table with a *negative* improvement on nine of its eleven pairs. So ``n_pairs_worsened`` is a
first-class finding, and a run with any regressed pair is degraded however good its medians look.

**With no before frame there is no registration error.** A result carrying only aligned
coordinates can be described but not judged: "how far did it move things" has no answer without
the coordinates it started from. That is reported as a degrade naming the missing frame, never
silently skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from spatialomicsgym.postanalysis.manifest import step

if TYPE_CHECKING:
    from spatialomicsgym.postanalysis.context import AnalysisContext

#: The frames a before/after needs, in the order they are looked for. ``spatial`` is last and is a
#: fallback rather than a frame: on a result whose aligner overwrote it, it holds the *aligned*
#: values, so it is used only when the provenance block says it is safe to.
_BEFORE_KEYS = ("spatial_3d_raw", "spatial_raw", "spatial")
_AFTER_KEYS = ("spatial_3d_aligned", "spatial_3d", "spatial_aligned", "moscot_spatial_warp")

#: What ``diagnose_3d_stack`` writes, and the tool that writes it. The registry types a diagnosis as
#: ``alignment`` so it reaches this runner, but it measures a stack and aligns nothing: judged as an
#: aligner it was warned for "writing no .h5ad" and degraded on every run (hunt 2026-09-30,
#: u21-3d-25).
_DIAGNOSIS_REPORT = "alignment_diagnosis.json"
_DIAGNOSIS_TOOL = "diagnose_3d_stack"


def _find_h5ad(ctx: AnalysisContext) -> str:
    """The aligned object this run produced, or an empty string.

    Staged inputs are not candidates, as in every other runner: ``slat_input/section1.h5ad`` beside
    ``slat_matching.csv`` is the section SLAT was handed, and taking it as "the aligned object"
    published the tool's own input as its ``source_outputs`` (hunt 2026-09-30, u19-pa-tasks-research
    skeptic note). The directory-walk fallback that used to follow read ``ctx.output_dir``, which
    :class:`AnalysisContext` has never had, so it walked nothing and is gone.

    Of several candidates, the registry's authoritative output wins, then the first in filename
    order -- the order detection reads them in -- that carries an aligned frame. It used to be the
    newest file, so a QC ``.h5ad`` written after the aligned object was judged in its place and the
    run degraded for "carrying none of" the frames sitting in the file beside it (hunt 2026-09-30,
    u19 review). A pick among several aligned objects is a tie-break and is said to be one.
    """
    import os

    from spatialomicsgym.postanalysis.detect import match_authoritative
    from spatialomicsgym.postanalysis.sources import is_staged_input, open_anndata

    candidates = [
        Path(p)
        for p in (getattr(ctx, "files", None) or [])
        if str(p).endswith(".h5ad") and os.path.exists(p) and not is_staged_input(p)
    ]
    if not candidates:
        return ""
    preferred = match_authoritative(candidates, ctx.tool_name)
    if preferred:
        return str(preferred[0])
    if len(candidates) == 1:
        return str(candidates[0])

    aligned: list[Path] = []
    for path in candidates:
        with open_anndata(path, backed=True) as adata:
            if adata is not None and any(str(k) in _AFTER_KEYS for k in adata.obsm):
                aligned.append(path)
    if not aligned:
        ctx.warn(
            f"none of the {len(candidates)} .h5ad files here carries one of {list(_AFTER_KEYS)}; "
            f"{candidates[0].name} is described because it comes first in filename order."
        )
        return str(candidates[0])
    if len(aligned) > 1:
        ctx.warn(
            f"{len(aligned)} .h5ad files here carry aligned coordinates; {aligned[0].name} was judged because "
            f"it comes first in filename order, which is a tie-break and not a judgement. Point "
            f"run_post_analysis at the one file that is the result."
        )
        ctx.manifest.degrade()
    return str(aligned[0])


def _frames(adata: Any) -> tuple[str, str, list[str]]:
    """(before key, after key, notes). Either may be empty."""
    import numpy as np

    present = {str(k) for k in adata.obsm}
    notes: list[str] = []
    after = next((k for k in _AFTER_KEYS if k in present), "")
    before = ""
    for key in _BEFORE_KEYS:
        if key not in present or key == after:
            continue
        if key == "spatial":
            block = adata.uns.get("spatial_3d") if hasattr(adata, "uns") else None
            original = (block or {}).get("original_key") if isinstance(block, dict) else None
            if after and original != "spatial":
                notes.append(
                    "obsm['spatial'] was used as the before frame without a provenance block "
                    "declaring it untouched. Some aligners overwrite it, in which case this "
                    "comparison is the aligned coordinates against themselves."
                )
        before = key
        break
    if before and after:
        try:
            if np.asarray(adata.obsm[before]).shape[0] != np.asarray(adata.obsm[after]).shape[0]:
                notes.append("the two frames have different row counts and cannot be compared")
                before = ""
        except Exception:
            before = ""
    return before, after, notes


def _files_figure(ctx: AnalysisContext, why: str = "No AnnData was produced") -> None:
    """The figure a run with no AnnData still owes the reader: what it actually wrote.

    The contract is that a task type leaves something to look at. A transform table or a matching
    CSV is a real alignment output -- SLAT and STACKer write nothing else -- so the run is
    described rather than treated as a failure.

    Drawn from ``ctx.files`` and ``ctx.not_analysed``, the two lists the engine already enumerated.
    It used to walk ``ctx.output_dir``, which :class:`AnalysisContext` does not have: ``os.walk("")``
    yields nothing, so every such run got one "(nothing found)" bar under the caption "These are the
    files the run did write" -- for a SLAT run that wrote ``slat_matching.csv``,
    ``slat_embeddings.npz`` and a plot (hunt 2026-09-30, u19-pa-tasks-research-1). The suffix filter
    went with it; it would have dropped the ``.npz`` anyway.
    """
    from spatialomicsgym.postanalysis import plots
    from spatialomicsgym.postanalysis.sources import is_staged_input
    from spatialomicsgym.postanalysis.tasks.shallow import _display_names, _size

    with step(ctx.manifest, "output inventory"):
        # The same two lists and the same staged-input rule as ``shallow._inventory_figure``, so
        # the two inventories cannot disagree about one directory.
        files = sorted(set(ctx.files) | set(ctx.not_analysed))
        outputs = [f for f in files if not is_staged_input(f)]
        drawn = min(len(outputs), plots.INVENTORY_BAR_LIMIT)
        if not outputs:
            listed = "The run wrote no file of its own here"
        elif drawn == len(outputs):
            listed = f"The run wrote {len(outputs)} file{'' if len(outputs) == 1 else 's'}, all drawn here"
        else:
            listed = f"The run wrote {len(outputs)} files; the {drawn} largest are drawn here"
        if len(outputs) != len(files):
            listed += f"; {len(files) - len(outputs)} staged input file(s) excluded"
        fig = plots.inventory_chart(
            _display_names(outputs) or ["(nothing found)"],
            [_size(f) for f in outputs] or [0],
            title="What this alignment run wrote",
        )
        ctx.figure(
            fig,
            "output_inventory.png",
            title="What this alignment run wrote",
            caption=f"{why}, so no before/after registration error could be computed. {listed}.",
            kind="inventory",
        )


def _inventory_figure(ctx: AnalysisContext, adata: Any, why: str) -> None:
    """One figure on every exit, including the refusals.

    The post-analysis contract is that a task type owes the reader something to look at. A run
    that refused still has a fact worth drawing -- how many cells each section holds -- and a
    caption saying why it stopped there.
    """
    from spatialomicsgym.postanalysis import plots

    with step(ctx.manifest, "section inventory"):
        key = next(
            (
                c
                for c in ("slice_id", "section", "brain_section_label", "library_id", "batch", "Bregma")
                if c in adata.obs.columns
            ),
            "",
        )
        if key:
            counts = adata.obs[key].astype(str).value_counts().sort_index()
            labels, values = list(counts.index), [int(v) for v in counts.to_numpy()]
        else:
            labels, values = ["whole object"], [int(adata.n_obs)]
        fig = plots.bar_chart(labels, values, title="Cells per section", xlabel="section", ylabel="cells")
        ctx.figure(
            fig,
            "section_inventory.png",
            title="Cells per section",
            caption=f"What this object holds. No before/after was computed: {why}.",
            kind="bar",
        )


def _diagnosis_report(ctx: AnalysisContext) -> Path | None:
    """The diagnosis report this run wrote, when the run is a diagnosis and not an alignment.

    Only for a run named ``diagnose_3d_stack`` or named nothing: an aligner whose output directory
    also holds an earlier diagnosis is still judged as the aligner it is.
    """
    from spatialomicsgym.postanalysis.detect import resolve_profile
    from spatialomicsgym.postanalysis.sources import is_staged_input

    tool = str(ctx.tool_name or "").strip()
    if tool:
        profile = resolve_profile(tool)
        if getattr(profile, "tool_name", tool) != _DIAGNOSIS_TOOL:
            return None
    for path in ctx.files:
        if Path(path).name == _DIAGNOSIS_REPORT and not is_staged_input(path):
            return Path(path)
    return None


def _summarise_diagnosis(ctx: AnalysisContext, path: Path) -> None:
    """Report what the diagnosis found, in its own numbers, and judge no ``.h5ad``."""
    import json

    from spatialomicsgym.postanalysis.sources import is_staged_input

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        report, why = None, f"{type(exc).__name__}: {exc}"
    else:
        why = "it has no 'class'" if isinstance(report, dict) else "it is not a JSON object"
    if not isinstance(report, dict) or "class" not in report:
        ctx.warn(
            f"{path.name} could not be read as a diagnose_3d_stack report ({why[:160]}), so nothing is summarised."
        )
        _files_figure(ctx, why="The diagnosis report could not be read")
        ctx.manifest.degrade()
        return

    ctx.analysed(path)
    ctx.manifest.add_finding(
        "diagnosis_class", str(report["class"]), "the stack's class from diagnose_3d_stack: A, B, C or unknown"
    )
    for key, label in (
        ("n_sections", "sections in the stack"),
        ("n_adjacent_pairs", "adjacent section pairs the diagnosis measured"),
        ("summary", "the diagnosis in one sentence"),
    ):
        if report.get(key) is not None:
            ctx.manifest.add_finding(key, report[key], label)
    for question in report.get("questions") or []:
        if str(question).strip():
            ctx.warn(str(question))
    # Not judged, and said so: in a shared directory an aligner's result can sit beside the report,
    # and with no tool named nothing settles which of the two this run was.
    beside = [p for p in ctx.files if str(p).endswith(".h5ad") and not is_staged_input(p)]
    if beside:
        ctx.warn(
            f"{len(beside)} .h5ad file(s) here were not judged: this run was read as a diagnosis. To "
            f"judge an alignment written here, pass tool_name= naming the aligner."
        )
        if not str(ctx.tool_name or "").strip():
            ctx.manifest.degrade()

    counts = report.get("counts")
    if isinstance(counts, dict) and counts:
        with step(ctx.manifest, "adjacent pairs by class"):
            from spatialomicsgym.postanalysis import plots

            labels = [str(k) for k in counts]
            fig = plots.bar_chart(
                labels,
                [int(counts[k] or 0) for k in counts],
                title="Adjacent pairs by class",
                xlabel="class",
                ylabel="pairs",
            )
            ctx.figure(
                fig,
                "diagnosis_pair_classes.png",
                title="Adjacent pairs by class",
                caption=(
                    f"How diagnose_3d_stack classed each adjacent pair; the stack as a whole is class "
                    f"{report['class']}. A diagnosis aligns nothing, so there is no before/after here."
                ),
                kind="bar",
            )


def run(ctx: AnalysisContext) -> None:
    # A diagnosis is summarised, never judged as an aligner (hunt 2026-09-30, u21-3d-25).
    diagnosis = _diagnosis_report(ctx)
    if diagnosis is not None:
        _summarise_diagnosis(ctx, diagnosis)
        return

    path = _find_h5ad(ctx)
    if not path:
        # A legitimate output shape, not a broken run: SLAT writes a matching table and STACKer a
        # warped image. It is described and marked partial rather than treated as a failure.
        ctx.warn(
            "this run wrote no .h5ad, so there are no coordinates to compare. An alignment that "
            "produces only a transform or matching table can be described but not judged -- "
            "spatialomicsgym.spatial3d.adapters says which tools those are and what they leave."
        )
        _files_figure(ctx)
        ctx.manifest.degrade()
        return

    from spatialomicsgym.postanalysis.sources import open_anndata

    # Backed: everything below reads obsm, obs and uns, never X, and a 3D atlas's matrix is tens of
    # GB pulled through RAM for a two-column frame (hunt 2026-09-30, u19-pa-tasks-research-7).
    with open_anndata(path, backed=True) as adata:
        if adata is None:
            # ``open_anndata`` yields None for a file it cannot read. Handing that to ``_describe``
            # recorded the unreadable file as this run's ``source_outputs`` and then died on
            # ``None.obsm``, leaving no figure (hunt 2026-09-30, u19-pa-tasks-research skeptic note).
            ctx.warn(f"{path} could not be read as an AnnData, so its coordinates could not be compared.")
            _files_figure(ctx, why="The aligned .h5ad could not be read")
            ctx.manifest.degrade()
            return
        _describe(ctx, path, adata)


def _describe(ctx: AnalysisContext, path: str, adata: Any) -> None:
    """Everything that needs the object open, so the handle is released on every exit.

    Split out rather than wrapped in a try: the runner returns early on four different refusals,
    and a close that sits on a path an ``except`` jumps over holds an HDF5 lock on the input until
    the object is collected -- so whatever rewrites that path next fails with errno 11.
    """
    import numpy as np

    from spatialomicsgym.spatial3d import validate as val

    try:
        ctx.analysed(path)
    except Exception:
        pass

    before_key, after_key, notes = _frames(adata)
    for note in notes:
        ctx.warn(note)

    if not after_key:
        ctx.warn(
            f"{path} carries none of {list(_AFTER_KEYS)}, so no aligned coordinates could be "
            f"located. spatialomicsgym.spatial3d.adapters lists where each shipped aligner leaves "
            f"its answer."
        )
        _inventory_figure(ctx, adata, "no aligned coordinates could be located")
        ctx.manifest.degrade()
        return
    ctx.manifest.add_finding("aligned_frame", after_key, "obsm key holding the aligned coordinates")

    section_key = ""
    for candidate in ("slice_id", "section", "brain_section_label", "library_id", "batch", "Bregma"):
        if candidate in adata.obs.columns and adata.obs[candidate].astype(str).nunique() >= 2:
            section_key = candidate
            break
    if not section_key:
        ctx.warn(
            "no obs column names each cell's section, so the object cannot be read as a stack and "
            "no adjacent pair exists to measure."
        )
        _inventory_figure(ctx, adata, "no section column, so the object is not a stack")
        ctx.manifest.degrade()
        return

    sections = adata.obs[section_key].astype(str).to_numpy()
    ctx.manifest.add_finding("n_sections", len(set(sections)), "sections in the stack")

    if not before_key:
        ctx.warn(
            f"the aligned coordinates are in obsm[{after_key!r}] but the frame they started from "
            f"is not in this object, so there is no registration error to report -- 'how far did "
            f"it move things' has no answer without the coordinates it began with. Write the "
            f"pre-alignment stack to obsm['spatial_3d_raw'] to make this measurable."
        )
        _inventory_figure(ctx, adata, "no before frame, so there is no registration error")
        ctx.manifest.degrade()
        return
    ctx.manifest.add_finding("before_frame", before_key, "obsm key holding the pre-alignment coordinates")

    after_xy = np.asarray(adata.obsm[after_key], dtype=float)[:, :2]
    before_xy = np.asarray(adata.obsm[before_key], dtype=float)[:, :2]

    # A string sort paired s1 with s10 and Bregma '-0.04' with '0.01' (hunt 2026-09-30, u21-3d-13).
    pairs, order_from = val.stack_pairs(adata, sections, before_xy)
    if not pairs:
        ctx.warn(f"no adjacent pair could be formed: {order_from}.")
        _inventory_figure(ctx, adata, "the section order is unknown, so no pair is adjacent")
        ctx.manifest.degrade()
        return
    ctx.manifest.add_finding("pair_order_from", order_from, "where the adjacent-pair order came from")

    verdict = None
    with step(ctx.manifest, "before/after adjacent-pair metrics"):
        frames_before = [(a, before_xy[sections == a], b, before_xy[sections == b]) for a, b in pairs]
        frames_after = [(a, after_xy[sections == a], b, after_xy[sections == b]) for a, b in pairs]
        # Named so a failed verdict's next step says which tool to move on from (hunt 2026-09-30,
        # u21-3d-23).
        verdict = val.before_after(
            val.regeometry(frames_before), val.regeometry(frames_after), aligner=ctx.tool_name or ""
        )

        import pandas as pd

        ctx.table(
            pd.DataFrame([d.as_row() for d in verdict.pairs]),
            "adjacent_pair_metrics_before_after.csv",
            title="Adjacent-pair geometry, before and after alignment",
            index=False,
        )

    if verdict is None:
        ctx.manifest.degrade()
        return

    ctx.manifest.add_finding("n_pairs", len(verdict.pairs), "adjacent section pairs measured")
    ctx.manifest.add_finding("n_pairs_improved", verdict.n_improved, "pairs that improved")
    ctx.manifest.add_finding(
        "n_pairs_worsened",
        verdict.n_worsened,
        "pairs that got WORSE -- a global mean hides these, and a run with any of them does not pass",
    )
    for metric in ("centroid_offset_frac", "containment", "local_shift_dispersion"):
        if metric in verdict.medians_before and metric in verdict.medians_after:
            ctx.manifest.add_finding(
                f"{metric}_before", round(verdict.medians_before[metric], 5), f"median {metric} before"
            )
            ctx.manifest.add_finding(
                f"{metric}_after", round(verdict.medians_after[metric], 5), f"median {metric} after"
            )
    ctx.manifest.add_finding("alignment_passed", bool(verdict.passed), "whether the alignment may be believed")

    with step(ctx.manifest, "registration error by pair"):
        from spatialomicsgym.postanalysis import plots

        labels = [f"{d.a}->{d.b}" for d in verdict.pairs]
        values = [d.after.get("centroid_offset_frac", float("nan")) for d in verdict.pairs]
        fig = plots.bar_chart(
            labels,
            values,
            title="Centroid offset after alignment, per adjacent pair",
            xlabel="adjacent pair",
            ylabel="offset (fraction of tissue diagonal)",
        )
        ctx.figure(
            fig,
            "registration_error_by_pair.png",
            title="Registration error by pair",
            caption=(
                "Centroid offset after alignment for each adjacent pair, as a fraction of the "
                "tissue diagonal. Read the worst bar, not the average: an alignment that improves "
                "the median while collapsing one pair is the recorded failure this runner exists "
                "to catch."
            ),
            kind="bar",
        )

    if not verdict.passed:
        ctx.warn(verdict.summary())
        ctx.manifest.degrade()
