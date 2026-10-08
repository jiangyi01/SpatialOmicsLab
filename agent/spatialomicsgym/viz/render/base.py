"""Canvases, saving, exporting, and the one place a figure becomes a file.

Every drawing primitive in this package takes an ``ax``. That is not a style preference: it is
what lets a set of panels be composed onto one canvas at draw time rather than stitched together
from saved images afterwards. There is no vector-stitching library in this environment, so a
composition made from PNGs would silently stop being a vector figure, and the repository's own
spatial scatter already accepts an ``ax`` for the same reason.

Three inherited constraints, all of which have a measured failure behind them:

* matplotlib's Agg backend is selected **before** pyplot is imported, via the repository's own
  helper, so importing this package in a headless worker does not try to open a display;
* a canvas is capped at thirty inches in either direction, because a hundred-type matrix at two
  hundred dots per inch asked for a hundred and forty-seven inches and thirty-two gigabytes and
  took the whole run down with it;
* a scatter above fifty thousand points is rasterised inside a vector export. Measured on this
  machine: a hundred and twenty thousand points is a 16.9 MB SVG unrasterised and 3.66 MB
  rasterised, and the report renderer refuses a figure above six megabytes. The axes, the text
  and the legend stay vector, which is what a publication actually needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Formats a figure may be written in. PNG and SVG display inline in the chat; PDF is served as a
#: download, because the portal has no embedded viewer for it.
FORMATS = ("png", "svg", "pdf")

#: Above this many marks, the point layer of a vector export is rasterised.
RASTERIZE_ABOVE = 50_000


def pyplot() -> Any:
    """The repository's own accessor: Agg is forced before pyplot is imported."""
    from spatialomicsgym.postanalysis.plots import _pyplot

    return _pyplot()


def bounded_inches(value: float) -> float:
    from spatialomicsgym.postanalysis.plots import _bounded_inches

    return float(_bounded_inches(value))


def default_dpi() -> int:
    from spatialomicsgym.postanalysis.plots import DEFAULT_DPI

    return int(DEFAULT_DPI)


def max_panels() -> int:
    from spatialomicsgym.postanalysis.plots import _MAX_GRID_PANELS

    return int(_MAX_GRID_PANELS)


def panel_grid(n_items: int, ncols: int = 3) -> tuple[int, int, int]:
    """``(rows, cols, shown)`` for *n_items* panels, capped.

    The cap is the repository's, and the number of panels actually shown is returned rather than
    silently applied, so the caller can say in the caption how many were left out.
    """
    cap = max_panels()
    shown = max(1, min(int(n_items), cap))
    cols = max(1, min(int(ncols) or 3, shown))
    rows = (shown + cols - 1) // cols
    return rows, cols, shown


def canvas(
    rows: int,
    cols: int,
    panel_w: float = 4.2,
    panel_h: float = 3.8,
    *,
    dpi: int | None = None,
    subplot_kw: dict | None = None,
):
    """A figure and a flat list of axes, sized from the panel grid and capped.

    ``subplot_kw`` is passed through so a caller can ask for ``{"projection": "3d"}``. The size cap
    and the dpi default still apply, which is the reason 3D panels come through here rather than
    calling pyplot directly.
    """
    plt = pyplot()
    width = bounded_inches(max(1.0, cols * panel_w))
    height = bounded_inches(max(1.0, rows * panel_h))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(width, height),
        dpi=dpi or default_dpi(),
        squeeze=False,
        subplot_kw=subplot_kw or {},
    )
    return fig, [ax for row in axes for ax in row]


def finish(fig: Any, ax: Any, title: str, xlabel: str = "", ylabel: str = "") -> None:
    from spatialomicsgym.postanalysis.plots import _finish

    _finish(fig, ax, title, xlabel, ylabel)


def blank(ax: Any) -> None:
    """Hide an unused panel rather than leaving an empty framed box in the grid."""
    ax.set_axis_off()


def pooled_mask(labels: Any, pooled: Any) -> Any:
    """Which observations carry a level the palette pooled. ``None`` when nothing was pooled."""
    import numpy as np

    if not pooled:
        return None
    names = {str(p) for p in pooled}
    return np.fromiter((str(v) in names for v in np.asarray(labels).tolist()), dtype=bool, count=len(labels))


def shared_legend(fig: Any, levels: Any, colours: dict[Any, Any], n_pooled: int = 0, title: str = "") -> bool:
    """One legend for a grid whose panels share a categorical palette. Returns whether it was drawn.

    A faceted figure draws its panels with no legend each, which is right -- twelve copies of one
    legend is noise -- and wrong when it means no legend at all: nothing then tells a reader which
    colour is which level (hunt 2026-09-30, u20b-viz-rest-10).
    """
    from matplotlib.lines import Line2D

    from spatialomicsgym.viz.palette import pooled_colour

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=6, color=colours[level], label=str(level))
        for level in levels
        if level in colours
    ]
    if n_pooled:
        handles.append(
            Line2D(
                [], [], marker="o", linestyle="", markersize=6, color=pooled_colour(), label=f"{n_pooled} pooled levels"
            )
        )
    if not handles or len(handles) > 61:
        return False
    ncol = 1 if len(handles) <= 20 else (2 if len(handles) <= 40 else 3)
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=8,
        ncol=ncol,
        title=title or None,
    )
    return True


def rasterize_if_large(artist: Any, n_points: int, fmt: str) -> bool:
    """Rasterise a point layer inside a vector export. Returns whether it did."""
    if fmt not in ("svg", "pdf") or int(n_points) <= RASTERIZE_ABOVE:
        return False
    try:
        artist.set_rasterized(True)
    except Exception:
        return False
    return True


def save(fig: Any, figures_dir: str | Path, stem: str, fmt: str = "png", dpi: int | None = None) -> str:
    """Write the figure and return the path **relative to the results directory**.

    That relative shape is what the manifest requires, and returning it rather than an absolute
    path is also what keeps this machine's directory layout out of anything a reader sees.
    """
    plt = pyplot()
    fmt = str(fmt).lower()
    if fmt not in FORMATS:
        raise ValueError(f"unknown figure format {fmt!r}; this toolkit writes {', '.join(FORMATS)}")
    target = Path(figures_dir)
    target.mkdir(parents=True, exist_ok=True)
    name = f"{stem}.{fmt}"
    path = target / name
    # `bbox_inches="tight"` is deliberate and is what the repository's own writer uses: a legend
    # placed outside the axes is otherwise cropped off the saved file even though it was visible
    # in the figure.
    fig.savefig(path, dpi=dpi or default_dpi(), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{name}"


def export_dpi_for(fig: Any, requested: int) -> int:
    """A raster resolution the canvas can actually produce.

    Thirty inches at six hundred dots per inch is eighteen thousand pixels, which Agg will draw
    and which can exceed the hundred-and-twenty-eight-megabyte ceiling the file route serves
    under -- so the request is capped against the canvas rather than honoured blindly, and the
    figure's record says what was achieved rather than what was asked for.
    """
    try:
        longest = float(max(fig.get_size_inches()))
    except Exception:
        return int(requested)
    if longest <= 0:
        return int(requested)
    return max(72, min(int(requested), int(16000 / longest)))


def point_size_for(n_obs: int, span: float = 0.0) -> float:
    """A marker size that suits the density, rather than a constant that suits one dataset.

    The rule is the repository's own: a four-thousand-spot Visium section and a four-hundred-
    thousand-cell Xenium section both have to come out readable, and a single default makes one
    of them a solid block of colour.
    """
    base = max(2.0, min(40.0, 6000.0 / max(1, int(n_obs))))
    return round(float(base), 2)


def subsample(n_obs: int, limit: int, seed: int = 0):
    """A deterministic stride, and the record of it. ``(index or None, Sampling)``.

    A stride rather than a random choice: it is reproducible without carrying a generator state
    into the figure's record, and it does not preferentially drop dense regions the way a naive
    spatial subsample does.
    """
    import numpy as np

    from spatialomicsgym.viz.palette import Sampling

    if int(n_obs) <= int(limit):
        return None, Sampling(n_obs, n_obs, "no sampling", seed)
    step = int(np.ceil(n_obs / float(limit)))
    index = np.arange(0, n_obs, step)
    return index, Sampling(n_obs, int(index.size), f"every {step}th observation", seed)
