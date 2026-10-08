"""The four task types with no deep runner yet: cell communication, alignment, imputation, trajectory.

Shallow is a statement about depth, never about honesty. A run here still scans every output file,
still draws at least one figure, and states in ``warnings`` -- naming the task type in full -- that
post-analysis has no purpose-built support for it. Silence would be worse than a thin result: an
empty ``figures`` list reads to L2 and L3 as "the tool produced nothing".

The figure comes from a ladder, first rung that works:

1. the spots in space, if any file carries coordinates -- true for most of these outputs;
2. the column means of the largest numeric table, which for a communication result is the
   per-pathway signal and for an imputation result the per-gene mean;
3. the file inventory, which is at minimum an honest answer to "what came out of the run".

Status is always at least ``partial``: the analysis that a deep runner would do did not happen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialomicsgym.postanalysis import plots
from spatialomicsgym.postanalysis.manifest import step
from spatialomicsgym.postanalysis.sources import find_coordinates, is_staged_input
from spatialomicsgym.postanalysis.tables import is_tabular, numeric_frame, read_table

if TYPE_CHECKING:
    from spatialomicsgym.postanalysis.context import AnalysisContext

#: What a deep runner for each of these would add, said plainly rather than left to the reader.
_WHAT_IS_MISSING = {
    "cell_communication": ("ligand-receptor pair ranking, sender/receiver summaries and pathway-level maps"),
    "imputation": "held-out gene correlation, imputed-vs-measured scatter and per-gene error",
    "trajectory": "pseudotime ordering, branch assignment and gene trends along the trajectory",
}

#: Columns summarised in the fallback bar chart.
_MAX_SUMMARY_COLUMNS = 25


def run(ctx: AnalysisContext) -> None:
    task = str(ctx.manifest.task_type)
    ctx.warn(
        f"{task}: post-analysis has no deep support for this task type yet, so the run produced a "
        f"scan and a summary figure only. A deep runner would add {_WHAT_IS_MISSING.get(task, 'more')}."
    )
    ctx.manifest.degrade()

    # Nothing is recorded here. `ctx.analysed` means "this analysis read this file" -- that is the
    # documented contract on the method and the meaning three consumers rely on -- and at this point
    # the runner has read nothing at all. What used to stand here was a bulk loop over the first
    # twenty non-staged files, which turned sort order into a provenance claim.
    #
    # It was not a cosmetic claim. `review._resolve_source` takes `source_outputs[0]` as *the
    # prediction* and computes every value check from it, without re-scanning the disk; `render`
    # prints the list to a human under "Analysed"; `next_step` names the first three to the agent.
    # So in a directory holding a QC sidecar and a result, the sidecar decided the verdict on
    # alphabetical order alone -- a constant one publishing `unusable` and `surface_unusable_result`
    # (non-executable: the agent stops) over a healthy result, and a varying one publishing `ok`
    # over a dead one, with the signal_free check having examined the sidecar in both cases.
    #
    # The two figure steps below already call `ctx.analysed` on exactly what they opened -- the file
    # `find_coordinates` accepted, and the table `_summary_figure` summarised. That is the whole
    # truthful answer. When neither runs, the inventory figure stats files without reading them, and
    # an empty list is then the honest report rather than a gap to be filled.
    if _coordinate_figure(ctx):
        _summary_figure(ctx)
        return
    if _summary_figure(ctx):
        return
    _inventory_figure(ctx)


# ---------------------------------------------------------------------------------------


def _coordinate_figure(ctx: AnalysisContext) -> bool:
    coordinates, source = find_coordinates(ctx.files)
    if coordinates is None:
        return False
    drawn = False
    with step(ctx.manifest, "spot layout"):
        ctx.analysed(source)
        ctx.figure(
            plots.spatial_scatter(
                coordinates["x"],
                coordinates["y"],
                [1.0] * len(coordinates),
                title="Spots in space",
                categorical=False,
            ),
            "spot_layout.png",
            title="Spots in space",
            caption=f"{len(coordinates)} spots from {source.name if source else 'the output'}",
            kind="spatial_map",
        )
        drawn = True
    return drawn


def _summary_figure(ctx: AnalysisContext) -> bool:
    picked = _largest_numeric_table(ctx)
    if picked is None:
        return False
    frame, path = picked
    drawn = False
    with step(ctx.manifest, "column summary"):
        mean = frame.mean(axis=0).sort_values(ascending=False).head(_MAX_SUMMARY_COLUMNS)
        ctx.analysed(path)
        ctx.table(mean.rename("mean").to_frame(), "column_summary.csv", title=f"Column means of {path.name}")
        ctx.figure(
            plots.bar_chart(
                mean.index,
                mean.to_numpy(),
                title=f"Column means: {path.name}",
                ylabel="Mean",
                horizontal=True,
            ),
            "column_summary.png",
            title=f"Column means: {path.name}",
            caption=f"{min(len(mean), _MAX_SUMMARY_COLUMNS)} of {frame.shape[1]} numeric columns",
            kind="bar",
        )
        drawn = True
    return drawn


def _inventory_figure(ctx: AnalysisContext) -> None:
    with step(ctx.manifest, "output inventory"):
        # Both lists, for the reason ``engine._write_scan`` scans both: this is a picture of the
        # directory, not of what could be parsed out of it. ``ctx.files`` alone is the analysis
        # candidates, and under the title "Files written by the tool" that drew 2 bars for the
        # recorded 202-file stlearn run -- with the 200 ``.jpeg`` tiles it wrote nowhere on the
        # chart. Sorted so the picture does not depend on which list a file arrived on.
        files = sorted(set(ctx.files) | set(ctx.not_analysed))
        if not files:
            return
        # The chart is sorted by size and the staged input is the largest file in all three recorded
        # runs that have one, so under the title "Files written by the tool" the dataset the tool was
        # *handed* was bar #1 -- carrying most of the plotted mass in a picture of what the tool
        # produced. ``is_staged_input`` is the shared rule the rest of this module already routes
        # through; asking it here keeps the two answers from drifting.
        outputs = [f for f in files if not is_staged_input(f)]
        # With no output at all there is nothing to draw and the reader is still owed a figure --
        # ``test_postanalysis_a_staged_input_is_not_a_summary`` pins that, and it is the right call:
        # "here is what is in the directory, all of it staged input" is information, and an empty
        # panel is not. The title carries the difference rather than the caption, because the title
        # is what the report tile and the portal card show.
        staged_only = not outputs
        shown = files if staged_only else outputs
        title = "Files in the output directory" if staged_only else "Files written by the tool"
        # Every file goes to the chart; ``inventory_chart`` owns which ones survive the cap, and
        # ``plots.INVENTORY_BAR_LIMIT`` is read back here so the caption counts the bars that were
        # actually drawn. Slicing here and captioning ``len(ctx.files)`` was the defect: a 40-file
        # directory read "40 files" under 25 bars, and the 25 were the alphabetically-first paths,
        # so the largest file in the directory was missing from a chart sorted by size.
        drawn = min(len(shown), plots.INVENTORY_BAR_LIMIT)
        caption = f"{len(shown)} files" if drawn == len(shown) else f"the {drawn} largest of {len(shown)} files"
        if staged_only:
            caption += " -- all of them staged input, so the tool wrote nothing here"
        elif len(shown) != len(files):
            caption += f"; {len(files) - len(shown)} staged input file(s) excluded"
        ctx.figure(
            plots.inventory_chart(
                _display_names(shown),
                # ``ctx.not_analysed`` can carry a file whose ``stat`` raised -- that is one of the
                # two reasons a file lands on it -- and ``ctx.files`` never could, so this call was
                # unguarded until the chart started describing the whole directory. A bar of length
                # zero for a file nobody can measure is the same answer ``scan_rows`` gives.
                [_size(f) for f in shown],
                title=title,
            ),
            "output_inventory.png",
            title=title,
            caption=caption,
            kind="inventory",
        )


def _size(path) -> int:
    """Bytes, or 0 for a file that cannot be stat'ed. ``scan_rows`` records -1; a bar cannot."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _display_names(files: list) -> list[str]:
    """One label per file that a reader can tell apart from the others.

    ``Path.name`` alone renders ``runA/report.txt`` and ``runB/report.txt`` as two bars both reading
    ``report.txt``, on the one figure whose whole job is "what came out of the run". Only the names
    that actually collide are qualified -- with the parent directory, and with more of the path if
    that still collides -- so the common case stays as short as it was.
    """
    from collections import Counter

    counts = Counter(f.name for f in files)
    labels: list[str] = []
    for path in files:
        if counts[path.name] == 1:
            labels.append(path.name)
            continue
        parts = path.parts
        # Widen from the right until this file's tail is unique among the colliding set, capped by
        # the path itself so a pathological tree cannot loop.
        depth = 2
        while depth < len(parts) and sum(1 for f in files if f.parts[-depth:] == parts[-depth:]) > 1:
            depth += 1
        labels.append("/".join(parts[-depth:]))
    return labels


def _largest_numeric_table(ctx: AnalysisContext):
    best = None
    for path in [f for f in ctx.files if is_tabular(f) and not is_staged_input(f)]:
        frame = read_table(path)
        if frame is None:
            continue
        numeric = numeric_frame(frame)
        if numeric.shape[1] == 0 or numeric.shape[0] == 0:
            continue
        size = numeric.shape[0] * numeric.shape[1]
        if best is None or size > best[0]:
            best = (size, numeric, path)
    return None if best is None else (best[1], best[2])
