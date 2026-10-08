"""Every figure the engine can draw, and the only place matplotlib is touched.

Contract non-negotiable #3: ``import spatialomicsgym.postanalysis`` must succeed in the 1.6 GB
agent env, so matplotlib is imported *inside* :func:`_pyplot` with the ``Agg`` backend selected
before ``pyplot`` -- selecting it afterwards is a no-op and a headless box then dies on a Tk import.

Each function returns a Figure and never writes a file; :func:`save_figure` owns the filesystem, so
a figure that fails to render leaves nothing half-written for L3 to link to. Callers reach these
through the module (``plots.bar_chart(...)``) rather than by importing the names, which keeps the
engine's step-level guards patchable in tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: The contract's default. Overridable per run, not per figure.
DEFAULT_DPI = 200

#: Legend entries per column. A single column of forty is a wall of text -- which is why the legend
#: used to be *dropped* past twenty -- but the answer is a second column, not no key at all: a
#: categorical map whose colours cannot be named is decoration.
_LEGEND_COLUMN_SIZE = 20

#: The qualitative palettes, in order, that :func:`_category_colours` draws from. ``tab10`` is the
#: first ten of ``tab20`` (verified: ``tab10[i] == tab20[2i]``), so keeping the historical split --
#: ``tab10`` at ten categories or fewer, ``tab20`` above -- costs nothing and moves no recorded
#: figure. The three tables together hold 60 mutually distinct colours.
_QUALITATIVE_CMAPS = ("tab20", "tab20b", "tab20c")

#: Distinct colours available for one categorical map: 20 + 20 + 20.
_MAX_DISTINCT_CATEGORIES = 60

#: Everything past :data:`_MAX_DISTINCT_CATEGORIES` shares this, and the legend says how many. Not
#: in any of the three tables, so it cannot be mistaken for a category that has its own colour.
_POOLED_COLOUR = "#b0b0b0"

#: Panels in a multi-panel grid. More than this and each panel is too small to read.
_MAX_GRID_PANELS = 12

#: Bars on the inventory chart. Public, and read by the caller that writes the caption, because a
#: caption counting one thing while the chart draws another is exactly the defect that made this a
#: constant: ``shallow`` captioned ``len(ctx.files)`` over a chart holding ``ctx.files[:25]``.
INVENTORY_BAR_LIMIT = 25

#: Ceiling on either side of a canvas, in inches.
#:
#: The per-item sizes below grow with the number of cell types or genes and had no upper bound.
#: A co-localization matrix is ``n_cell_types`` square, and a reference with a hundred-odd
#: fine-grained types is ordinary; at ``DEFAULT_DPI`` one such figure measured 75 x 62.5 in / 8.3 GB
#: peak RSS, and 240 types measured 147 x 122.5 in / 31.8 GB. Past 546 columns the canvas is wider
#: than Agg's 65,536 px limit and ``savefig`` raises, which ``step()`` records as a warning -- the
#: figure is missing but the run survives. Below that there is no error to catch: the process asks
#: for tens of GB and a smaller box OOM-kills it, and since ``run_post_analysis`` writes
#: ``manifest.json`` as its last statement, the tables and findings already computed die with it.
#: 30 in is 6,000 px at ``DEFAULT_DPI``; every figure this tree actually draws asks for 5.5 to 21.
_MAX_FIGURE_INCHES = 30.0


def _bounded_inches(value: float) -> float:
    """One side of a canvas, never past :data:`_MAX_FIGURE_INCHES`."""
    return min(float(value), _MAX_FIGURE_INCHES)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig, results_dir: Path, filename: str, dpi: int = DEFAULT_DPI) -> str:
    """Write ``fig`` to ``<results_dir>/figures/<filename>`` and return the *relative* path.

    Deterministic names, no timestamps: the directory carries the run identity, so L3 can link a
    figure by name and a second run of the same input overwrites rather than accumulates.
    """
    plt = _pyplot()
    figures = Path(results_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    target = figures / filename
    try:
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return f"figures/{filename}"


def _finish(fig, ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def spatial_scatter(
    x,
    y,
    values,
    *,
    title: str,
    categorical: bool,
    value_label: str = "",
    ax=None,
    point_size: float | None = None,
):
    """One tissue map. Categorical values get a legend, continuous values a colourbar."""
    import numpy as np

    plt = _pyplot()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    size = point_size if point_size is not None else max(2.0, min(40.0, 6000.0 / max(len(x), 1)))

    if categorical:
        import math

        labels = [str(v) for v in values]
        uniq = sorted(set(labels), key=_natural_key)
        colours, pooled = _category_colours(labels, uniq)
        for label in uniq:
            if label in pooled:
                continue
            mask = np.array([lab == label for lab in labels])
            ax.scatter(x[mask], y[mask], s=size, color=colours[label], label=label, linewidths=0)
        if pooled:
            mask = np.array([lab in pooled for lab in labels])
            ax.scatter(
                x[mask],
                y[mask],
                s=size,
                color=_POOLED_COLOUR,
                label=f"other ({len(pooled)} categories)",
                linewidths=0,
            )
        # Always a legend. Every colour on the map is now unique, so the legend is what makes the
        # map readable at all; past one column's worth it grows sideways instead of disappearing.
        entries = len(uniq) - len(pooled) + (1 if pooled else 0)
        ax.legend(
            markerscale=2,
            fontsize="small",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=max(1, math.ceil(entries / _LEGEND_COLUMN_SIZE)),
        )
    else:
        vals = np.asarray(values, dtype="float64")
        sc = ax.scatter(x, y, c=vals, s=size, cmap="viridis", linewidths=0)
        # A colourbar is a claim that the colours encode a quantity the reader can read off it.
        # Over an array with no variation there is no quantity: ``shallow`` draws "Spots in space"
        # -- a map of where the spots are and nothing else -- by passing ``[1.0] * n_spots`` to
        # satisfy this signature, and the recorded commot run published a viridis scale running
        # 0.9 to 1.1 beside 200 identically-coloured dots. Same for an all-NaN column, which the
        # per-cell-type grid can produce for a type no spot carries.
        finite = vals[np.isfinite(vals)]
        if finite.size and float(finite.min()) != float(finite.max()):
            cbar = ax.get_figure().colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            if value_label:
                cbar.set_label(value_label)
        elif value_label:
            # Still no colourbar -- there is still no gradient to read -- but dropping it dropped the
            # number with it. ``Normalize`` collapses a degenerate range onto one point of the
            # colormap, so a cell type the tool placed nowhere and one it put at 0.9 in every spot
            # come out the same PNG byte for byte; five of stdgcn's nine recorded panels are the
            # first of those, and "absent from the whole section" is the most useful thing that
            # figure could say. Say it in words. ``value_label`` is the test for whether the caller
            # means its colours to encode anything: ``shallow`` passes ``[1.0] * n_spots`` to satisfy
            # this signature while drawing "Spots in space", and must stay silent.
            ax.set_xlabel(
                f"{value_label}: {float(finite.min()):.3g} at every spot"
                if finite.size
                else f"{value_label}: no value at any spot"
            )

    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()  # image coordinates: row 0 is the top of the section
    ax.set_xticks([])
    ax.set_yticks([])
    return _finish(fig, ax, title)


def spatial_grid(x, y, frame, *, title: str, value_label: str = ""):
    """One small tissue map per column of ``frame`` (cell types, marker genes).

    ``value_label`` names the quantity every panel's colourbar shows. It was accepted and dropped,
    so a grid of cell-type proportions and a grid of gene expression -- the two things this draws --
    came out with nine unlabelled colourbars apiece and nothing on the figure saying which was which.
    """
    import math

    plt = _pyplot()
    columns = list(frame.columns)[:_MAX_GRID_PANELS]
    ncols = min(3, len(columns)) or 1
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)
    for i, column in enumerate(columns):
        ax = axes[i // ncols][i % ncols]
        spatial_scatter(
            x,
            y,
            frame[column].to_numpy(),
            title=str(column),
            categorical=False,
            value_label=value_label,
            ax=ax,
        )
    for j in range(len(columns), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


def bar_chart(labels, values, *, title: str, xlabel: str = "", ylabel: str = "", horizontal: bool = False):
    """``xlabel`` names the categories, ``ylabel`` names the measured value -- in both orientations.

    ``horizontal=True`` swaps which matplotlib axis carries which, and the labels used not to swap
    with them: the caller's description of the value landed on the axis showing the category names
    and the axis the bars actually run along was left blank. Every horizontal caller was affected;
    the headline SVG figure came out with gene names running up an axis labelled ``-log10(qval)``.
    """
    plt = _pyplot()
    labels = [str(lab) for lab in labels]
    height = max(3.0, min(12.0, 0.32 * len(labels) + 1.5))
    if horizontal:
        fig, ax = plt.subplots(figsize=(8, height))
        ax.barh(labels[::-1], list(values)[::-1], color="#4878a8")
    else:
        fig, ax = plt.subplots(figsize=(max(6.0, min(14.0, 0.55 * len(labels) + 2)), 4.5))
        ax.bar(labels, list(values), color="#4878a8")
        if max((len(lab) for lab in labels), default=0) > 4 or len(labels) > 8:
            ax.tick_params(axis="x", labelrotation=45)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")
    ax.grid(axis="x" if horizontal else "y", alpha=0.3)
    ax.set_axisbelow(True)
    if horizontal:
        xlabel, ylabel = ylabel, xlabel
    return _finish(fig, ax, title, xlabel, ylabel)


def heatmap(frame, *, title: str, cmap: str = "RdBu_r", center_zero: bool = True, annotate: bool | None = None):
    """A matrix of values as a colour grid, diverging around zero so the sign is readable at a glance.

    ``center_zero`` sets symmetric ``vmin``/``vmax`` from the largest absolute value, so a diverging
    colormap puts zero at the midpoint instead of wherever the data happens to straddle. Cells carry
    their value when the grid is small enough to read it -- 64 cells or fewer -- and are left bare
    above that, where the numbers would overlap into noise.
    """
    import numpy as np

    plt = _pyplot()
    data = np.asarray(frame.to_numpy(), dtype="float64")
    n_rows, n_cols = data.shape
    fig, ax = plt.subplots(
        figsize=(
            _bounded_inches(max(5.0, 0.6 * n_cols + 3)),
            _bounded_inches(max(4.0, 0.5 * n_rows + 2.5)),
        )
    )
    kw: dict[str, Any] = {"cmap": cmap}
    if center_zero:
        limit = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 1.0
        kw["vmin"], kw["vmax"] = -limit, limit
    image = ax.imshow(data, aspect="auto", **kw)
    ax.set_xticks(range(n_cols), [str(c) for c in frame.columns], rotation=45, ha="right")
    ax.set_yticks(range(n_rows), [str(r) for r in frame.index])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if annotate is None:
        annotate = n_rows * n_cols <= 64
    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                if np.isfinite(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)
    return _finish(fig, ax, title)


def histogram(values, *, title: str, xlabel: str, bins: int = 50, vline: float | None = None, log_y: bool = False):
    """The distribution of one value over many observations, with an optional reference line.

    Non-finite values are dropped before binning rather than raising: the arrays that reach this
    drawer are tool output, and a method that could not score a spot writes NaN there. ``vline``
    marks a threshold and labels itself with the value, so a cutoff drawn on the figure stays
    legible when the figure is read without its caption.
    """
    import numpy as np

    plt = _pyplot()
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(arr, bins=bins, color="#4878a8", edgecolor="white", linewidth=0.4)
    if vline is not None:
        ax.axvline(vline, color="#c44e52", linestyle="--", label=f"{xlabel} = {vline:g}")
        ax.legend(fontsize="small")
    if log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    return _finish(fig, ax, title, xlabel, "Count")


def box_plot(frame, *, title: str, ylabel: str):
    """One box per column, drawn to the true extremes rather than to 1.5x the interquartile range.

    The whiskers span the full range and outliers are not re-drawn as markers. The comment in the
    body records why: a cell type confined to one region is near zero in most spots and near one in
    the few it occupies, so its signal is entirely tail, and the default rule drew four such runs as
    a flat line at zero.
    """
    plt = _pyplot()
    columns = list(frame.columns)
    fig, ax = plt.subplots(figsize=(_bounded_inches(max(6.0, 0.7 * len(columns) + 2)), 4.5))
    # ``whis=(0, 100)`` -- whiskers to the true extremes -- because this figure's caption promises a
    # distribution "over all spots" and the default 1.5x-IQR rule, with ``showfliers=False`` below it,
    # was quietly dropping the part of that distribution the reader came for. A cell type confined to
    # one region is near zero in most spots and near one in the few it occupies, so its IQR is tiny
    # and its real signal is entirely tail: four recorded runs drew such a cell type as a flat line at
    # zero while it was the sole occupant of up to a quarter of the tissue, and celloscope left 13.6%
    # of its values off the figure. ``showfliers=True`` is the other repair and the wrong one --
    # novosparc would add 2411 markers across a 200-column axis and bury the boxes.
    ax.boxplot(
        [frame[c].dropna().to_numpy() for c in columns],
        tick_labels=[str(c) for c in columns],
        showfliers=False,
        whis=(0, 100),
    )
    ax.tick_params(axis="x", labelrotation=45)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    return _finish(fig, ax, title, "", ylabel)


def inventory_chart(names, sizes, *, title: str, limit: int = INVENTORY_BAR_LIMIT):
    """The last-resort figure: what the tool actually wrote, largest first.

    Every task type owes the user at least one figure. When a task type has no deep support and no
    plottable numbers were found, this is still an honest answer to "what came out of the run".

    "Largest first" was the docstring's promise and never the code's: the caller handed over the
    first 25 paths in *path* order and nothing here sorted them, so a 40-file directory got 25 bars
    chosen alphabetically and the largest file in it was not drawn at all.

    Ordering and truncation are done *here*, not by the caller, and the cap is
    :data:`INVENTORY_BAR_LIMIT` rather than a literal, so the caption -- which is written by the
    caller from the same constant -- cannot describe a different chart than the one drawn.

    The chart is capped rather than made to draw every file, which is the other half of the same
    choice. A tool output directory has no bounded file count (``xfuse`` writes 26, a checkpointing
    run writes thousands), a horizontal ``bar_chart`` clamps its canvas at 12 in, and past a few
    dozen rows the labels overprint into a smear -- the same unbounded per-item growth
    :data:`_MAX_FIGURE_INCHES` exists to stop. So the figure keeps a fixed, readable size and shows
    the files that carry the most bytes, and the caller says out loud that it is a selection.
    """
    pairs = [(str(n), float(s)) for n, s in zip(names, sizes, strict=False)]
    # Descending by size, name as the tie-break, so equal-sized files draw in a stable order.
    pairs.sort(key=lambda pair: (-pair[1], pair[0]))
    kept = pairs[: max(int(limit), 0)]
    return bar_chart(
        [name for name, _ in kept],
        [size / 1024.0 for _, size in kept],
        title=title,
        xlabel="",
        ylabel="Size (KB)",
        horizontal=True,
    )


def _category_colours(labels, uniq):
    """``(colour_per_label, pooled_labels)`` -- and no two returned colours are ever the same.

    ``tab20`` was indexed as ``cmap(i % cmap.N)``, which does not fail, it *wraps*: the twenty-first
    category came out in the first category's blue. The recorded ``novosparc`` deconvolution has 21
    dominant cell types, so its map was drawn with two cell types the reader cannot separate -- and
    with no legend, because the legend was dropped at the same threshold the colours started
    repeating. Silently reusing a colour is the worst of the three options available here.

    Below :data:`_MAX_DISTINCT_CATEGORIES` every category gets its own colour from
    :data:`_QUALITATIVE_CMAPS`. Above it the palette genuinely runs out, so the categories with the
    fewest points are pooled into one grey and the caller labels that entry with how many were
    pooled: the figure states what it cannot show rather than pretending. Rarest-first because the
    pooled entry should cost the reader as few points on the map as possible; ties break on the
    label so the same data always draws the same figure.
    """
    plt = _pyplot()
    palette = [plt.get_cmap(name)(i) for name in _QUALITATIVE_CMAPS for i in range(plt.get_cmap(name).N)]
    if len(uniq) <= 10:
        palette = [plt.get_cmap("tab10")(i) for i in range(10)]

    if len(uniq) <= _MAX_DISTINCT_CATEGORIES:
        return {label: palette[i] for i, label in enumerate(uniq)}, frozenset()

    from collections import Counter

    counts = Counter(labels)
    keep = sorted(
        sorted(uniq, key=lambda label: (counts[label], _natural_key(label)), reverse=True)[
            : _MAX_DISTINCT_CATEGORIES - 1
        ],
        key=_natural_key,
    )
    return (
        {label: palette[i] for i, label in enumerate(keep)},
        frozenset(uniq) - frozenset(keep),
    )


def _natural_key(text: str):
    """Sort ``domain_2`` before ``domain_10`` so legends read the way a human numbers domains."""
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]
