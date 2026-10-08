"""Tissue maps: values painted where they were measured, with the image under them.

The one thing a spatial figure must get right is that a spot appears where the tissue actually
is, and it is the easiest thing to get wrong in a way that still looks like a picture of tissue.
Four conventions are enforced here rather than left to each caller:

* the image and its scalefactor are chosen **as a pair**, never independently -- the high- and
  low-resolution factors on a real Visium slide differ by 3.3x, and the wrong pairing leaves
  every spot inside the frame, just huddled in one corner;
* the aspect ratio is one, so a section is not stretched to the shape of the axes box and no
  statement about its morphology is distorted;
* the y axis is inverted when an image is drawn, because image rows increase downward and data
  coordinates increase upward, and forgetting it mirrors the tissue;
* coordinates that look like array row and column indices are refused for an overlay upstream, in
  the profile, rather than being scaled by a factor that does not apply to them; a plain map of
  them is drawn with its axes labelled as indices.

The drawing functions take an ``ax`` so a set of panels can be composed onto one canvas.
"""

from __future__ import annotations

from typing import Any

from spatialomicsgym.viz import palette as _palette
from spatialomicsgym.viz.render import base as _base


def histology_underlay(ax: Any, image: Any, scalef: float, alpha: float = 1.0) -> None:
    """Draw the tissue image in the coordinate frame the spots will be plotted in.

    The spots are scaled onto the image rather than the image onto the spots: the image is the
    fixed thing with known pixel dimensions, and scaling it would resample it.
    """
    import numpy as np

    array = np.asarray(image)
    ax.imshow(array, alpha=float(alpha), interpolation="nearest")


def tissue_scatter(
    ax: Any,
    coords: Any,
    values: Any,
    *,
    categorical: bool,
    title: str = "",
    value_label: str = "",
    image: Any = None,
    image_scalef: float = 1.0,
    image_alpha: float = 1.0,
    point_alpha: float = 1.0,
    point_size: float | None = None,
    cmap: str = "viridis",
    vlim: tuple[float | None, float | None] = (None, None),
    crop: tuple[float, float, float, float] | None = None,
    fmt: str = "png",
    show_colorbar: bool = True,
    palette_labels: Any = None,
    show_legend: bool = True,
) -> dict[str, Any]:
    """One tissue map. Returns what the caption needs to know about what was drawn.

    ``categorical`` decides legend-versus-colourbar rather than being guessed from the dtype: a
    cluster id stored as an integer is categorical and a continuous colour bar over it is a
    picture that implies an ordering the labels do not have.

    ``palette_labels`` is the whole column when this map is one panel of several, so the palette is
    built once over every level and a level is the same colour in each panel.
    """
    import numpy as np

    xy = np.asarray(coords, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("coordinates must be an n-by-2 array of positions")
    xy = xy[:, :2]

    note: dict[str, Any] = {"n_drawn": int(xy.shape[0]), "pooled": [], "rasterized": False}

    if image is not None:
        histology_underlay(ax, image, image_scalef, image_alpha)
        xy = xy * float(image_scalef)
        note["image_scalef"] = float(image_scalef)

    size = point_size if point_size else _base.point_size_for(xy.shape[0])

    if categorical:
        labels = np.asarray(values).astype(object)
        reference = labels if palette_labels is None else np.asarray(palette_labels).astype(object)
        colours, pooled = _palette.categorical(reference)
        note["pooled"] = sorted(str(p) for p in pooled)
        artist = None
        for level, colour in colours.items():
            mask = labels == level
            if not mask.any():
                continue
            artist = ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=size,
                c=[colour],
                label=str(level),
                alpha=float(point_alpha),
                linewidths=0,
            )
            _base.rasterize_if_large(artist, int(mask.sum()), fmt) and note.update({"rasterized": True})
        # Above sixty levels the rarest are pooled, and the loop above draws only levels that kept a
        # colour -- so every observation of a pooled level vanished from the map while the caption
        # said they "share one grey" (hunt 2026-09-30, u20b-viz-rest-9). They are drawn, in it.
        rest = _base.pooled_mask(labels, pooled)
        if rest is not None and rest.any():
            artist = ax.scatter(
                xy[rest, 0],
                xy[rest, 1],
                s=size,
                c=[_palette.pooled_colour()],
                label=f"{len(pooled)} pooled levels",
                alpha=float(point_alpha),
                linewidths=0,
            )
            _base.rasterize_if_large(artist, int(rest.sum()), fmt) and note.update({"rasterized": True})
            note["n_pooled_drawn"] = int(rest.sum())
        n_levels = len([lvl for lvl in colours if (labels == lvl).any()])
        if show_legend and 0 < n_levels <= 60:
            ncol = 1 if n_levels <= 20 else (2 if n_levels <= 40 else 3)
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
                fontsize=8,
                markerscale=2.0,
                ncol=ncol,
                title=value_label or None,
            )
        note["n_levels"] = n_levels
    else:
        numbers = np.asarray(values, dtype=float)
        artist = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=numbers,
            s=size,
            cmap=cmap,
            vmin=vlim[0],
            vmax=vlim[1],
            alpha=float(point_alpha),
            linewidths=0,
        )
        if _base.rasterize_if_large(artist, numbers.size, fmt):
            note["rasterized"] = True
        if show_colorbar:
            # A constant array gets no colour bar: a bar whose two ends are the same number tells
            # a reader there is a gradient where there is none.
            finite = numbers[np.isfinite(numbers)]
            if finite.size and float(finite.min()) != float(finite.max()):
                bar = ax.figure.colorbar(artist, ax=ax, shrink=0.72, pad=0.02)
                if value_label:
                    bar.set_label(value_label, fontsize=9)
            else:
                note["constant_value"] = True

    if crop:
        x0, y0, x1, y1 = crop
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0) if image is not None else ax.set_ylim(y0, y1)

    ax.set_aspect("equal")
    if image is not None:
        # imshow has already put row zero at the top; the scatter must agree with it.
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
    else:
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if image is None:
        from spatialomicsgym.viz.profile import coords_look_like_array_indices

        if coords_look_like_array_indices(xy):
            # Array-index coordinates are drawn now rather than refused (a policy decision, hunt
            # 2026-09-30, u20b-viz-rest-26), so the figure itself says what its axes are: an index
            # lattice, not positions -- on a Visium array one row step is ~1.73 column steps.
            ax.set_xlabel("array column index", fontsize=8)
            ax.set_ylabel("array row index", fontsize=8)
            note["array_indices"] = True
    if title:
        ax.set_title(title, fontsize=10)
    return note


def tissue_panels(
    coords: Any,
    panels: dict[str, Any],
    *,
    categorical: bool,
    scales: Any,
    ncols: int = 3,
    title: str = "",
    value_label: str = "",
    image: Any = None,
    image_scalef: float = 1.0,
    image_alpha: float = 1.0,
    point_alpha: float = 1.0,
    point_size: float | None = None,
    cmap: str = "viridis",
    fmt: str = "png",
) -> tuple[Any, dict[str, Any]]:
    """A grid of tissue maps over the same coordinates, one per named value.

    The colour scales come from a ``PanelScales`` the caller built, which is what forces the
    shared-versus-independent decision to be made explicitly and written into the caption.
    """
    names = list(panels)
    rows, cols, shown = _base.panel_grid(len(names), ncols)
    fig, axes = _base.canvas(rows, cols)
    notes: dict[str, Any] = {"panels": [], "dropped": max(0, len(names) - shown)}
    for i in range(shown):
        name = names[i]
        vlim = scales.for_panel(i) if scales is not None else (None, None)
        note = tissue_scatter(
            axes[i],
            coords,
            panels[name],
            categorical=categorical,
            title=name,
            value_label=value_label if i == 0 else "",
            image=image,
            image_scalef=image_scalef,
            image_alpha=image_alpha,
            point_alpha=point_alpha,
            point_size=point_size,
            cmap=cmap,
            vlim=vlim,
            fmt=fmt,
            show_colorbar=not categorical,
        )
        note["name"] = name
        notes["panels"].append(note)
    for j in range(shown, len(axes)):
        _base.blank(axes[j])
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig, notes


def neighbor_graph(ax: Any, coords: Any, edges: Any, *, title: str = "", linewidth: float = 0.3) -> dict[str, Any]:
    """The edges a spatial statistic was computed over, drawn on the tissue.

    Worth drawing because the graph is an assumption, not an observation: a statistic computed
    over a six-neighbour grid and one computed over a fifty-micron radius answer different
    questions, and the picture is the only place that choice is visible.
    """
    import numpy as np
    from matplotlib.collections import LineCollection

    xy = np.asarray(coords, dtype=float)[:, :2]
    segments = [[(xy[i, 0], xy[i, 1]), (xy[j, 0], xy[j, 1])] for i, j in edges]
    ax.add_collection(LineCollection(segments, linewidths=linewidth, alpha=0.5, colors="#666666"))
    ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#222222", linewidths=0)
    ax.set_aspect("equal")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    return {"n_edges": len(segments), "n_nodes": int(xy.shape[0])}
