"""Drawing primitives for a stack of sections. One axes in, a note out, as elsewhere in render/.

Three hazards are handled here rather than left to each caller, because each of them produces a
figure that looks fine and is wrong.

**Aspect.** An ``Axes3D`` left on its default aspect stretches every axis to fill the box, so a
brain 13.5 mm deep and 11 mm wide comes out square. The box aspect is set from the real data
spans and the ratio is recorded in the note.

**Point count.** ``Axes3D`` scatter has no z-buffer: it draws in painter's order, so beyond a few
tens of thousands of points the picture is both illegible and wrong about what is in front. Above
:data:`MAX_3D_POINTS` the caller's data is strided down through the shared ``base.subsample``,
which returns the ``Sampling`` record that reaches the caption and the manifest -- a subsampled 3D
view that does not say so is a lie the reader can rotate.

**No y-inversion.** ``render/spatial.tissue_scatter`` inverts y because image rows run downward
and a tissue map is drawn over histology. There is no image in a 3D view, and inverting there
would mirror the specimen. It is not done, and this paragraph is why.
"""

from __future__ import annotations

from typing import Any

from spatialomicsgym.viz.render import base

#: Above this many points an Axes3D scatter is illegible and its depth ordering is painter's-order
#: wrong. The same number ``base.RASTERIZE_ABOVE`` uses for the 2D vector-export cutoff, because it
#: was measured on the same question: how many marks can a reader still resolve.
MAX_3D_POINTS = 50_000

#: The three viewing angles a stack is drawn from. Azimuth varies, elevation does not: the reader
#: is turning around the specimen, not tumbling it, and a fixed elevation keeps the three panels
#: comparable.
VIEW_ANGLES = ((20, -60), (20, -15), (20, 30))


def _box_aspect(ax: Any, xyz: Any) -> tuple[float, float, float]:
    """Set the box aspect from the data's own spans and return it."""
    import numpy as np

    arr = np.asarray(xyz, dtype=float)
    # nan-aware: a frame with uncovered cells carries NaN rows, and np.ptp then returned NaN spans
    # that set_box_aspect rejected silently, leaving the box stretched (hunt 2026-09-30,
    # u20b-viz-rest-27).
    finite = arr[np.isfinite(arr).all(axis=1)] if arr.ndim == 2 else arr
    spans = (finite.max(axis=0) - finite.min(axis=0)).astype(float) if len(finite) else np.ones(3)
    spans[~np.isfinite(spans) | (spans <= 0)] = 1.0
    spans = spans / spans.max()
    try:
        ax.set_box_aspect(tuple(spans))
    except Exception:
        pass
    return tuple(float(v) for v in spans)


def cloud_3d(
    ax: Any,
    xyz: Any,
    values: Any = None,
    *,
    categorical: bool = False,
    view: tuple[int, int] = VIEW_ANGLES[0],
    point_size: float = 0.0,
    cmap: str = "viridis",
    labels: tuple[str, str, str] = ("x", "y", "z"),
) -> dict[str, Any]:
    """One 3D scatter on a caller-supplied ``Axes3D``."""
    import numpy as np

    arr = np.asarray(xyz, dtype=float)
    note: dict[str, Any] = {"n_points": int(len(arr))}
    size = point_size or max(0.5, min(6.0, 4000.0 / max(len(arr), 1)))

    if values is None:
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=size, c="#4C6EF5", depthshade=False, linewidths=0)
    elif categorical:
        from spatialomicsgym.viz import palette

        # ``palette.categorical`` takes the LABELS and returns one colour per observation, keyed on
        # the level set so the same level is the same colour in every panel. It was called here with
        # ``len(levels)`` -- a count -- which raised ``TypeError: 'int' object is not iterable`` the
        # first time anything drew a 3D cloud coloured by a category. Nothing had: the scatter had
        # only ever been driven with a gene.
        codes = np.asarray(values).astype(str)
        colours, pooled = palette.categorical(codes)
        # One call per level rather than one colour per point, the same way `render/spatial.py`
        # draws a categorical map: it is what gives each level a legend entry, and it is what makes
        # the pooled-rarest entry visible instead of silently sharing a colour.
        for level, colour in colours.items():
            mask = codes == level
            if not mask.any():
                continue
            ax.scatter(
                arr[mask, 0],
                arr[mask, 1],
                arr[mask, 2],
                s=size,
                c=[colour],
                label=str(level),
                depthshade=False,
                linewidths=0,
            )
        # Pooled levels have no colour of their own, so the loop above skipped them and their cells
        # were missing from the volume (hunt 2026-09-30, u20b-viz-rest-9). Drawn in the shared grey.
        rest = base.pooled_mask(codes, pooled)
        if rest is not None and rest.any():
            ax.scatter(
                arr[rest, 0],
                arr[rest, 1],
                arr[rest, 2],
                s=size,
                c=[palette.pooled_colour()],
                label=f"{len(pooled)} pooled levels",
                depthshade=False,
                linewidths=0,
            )
            note["n_pooled_drawn"] = int(rest.sum())
        note["n_levels"] = len(colours)
        note["pooled"] = sorted(str(p) for p in pooled)
    else:
        vals = np.asarray(values, dtype=float)
        ax.scatter(
            arr[:, 0],
            arr[:, 1],
            arr[:, 2],
            s=size,
            c=vals,
            cmap=cmap,
            depthshade=False,
            linewidths=0,
        )

    ax.view_init(elev=view[0], azim=view[1])
    note["view"] = {"elev": view[0], "azim": view[1]}
    note["box_aspect"] = _box_aspect(ax, arr)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.grid(False)
    return note


def cloud_3d_views(
    xyz: Any,
    values: Any = None,
    *,
    categorical: bool = False,
    title: str = "",
    angles: tuple[tuple[int, int], ...] = VIEW_ANGLES,
    labels: tuple[str, str, str] = ("x", "y", "z"),
    seed: int = 0,
):
    """The same stack from three azimuths. ``(figure, note)``.

    Three, because one view of a point cloud hides whatever is behind it and a reader cannot
    rotate a PNG. The angles are fixed rather than chosen from the data so that two runs of the
    same analysis produce comparable pictures.
    """
    import numpy as np

    arr = np.asarray(xyz, dtype=float)
    index, sampling = base.subsample(len(arr), MAX_3D_POINTS, seed)
    if index is not None:
        arr = arr[index]
        if values is not None:
            values = np.asarray(values)[index]

    fig, axes = base.canvas(1, len(angles), panel_w=4.6, panel_h=4.4, subplot_kw={"projection": "3d"})
    notes = []
    for ax, view in zip(axes, angles, strict=False):
        notes.append(cloud_3d(ax, arr, values, categorical=categorical, view=view, labels=labels))
        ax.set_title(f"azim {view[1]}", fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, {
        "views": [n["view"] for n in notes],
        "box_aspect": notes[0]["box_aspect"] if notes else None,
        "n_points": int(len(arr)),
        "sampling": _sampling_record(sampling),
    }


def _sampling_record(sampling: Any) -> dict[str, Any]:
    """The sampling a 3D view was drawn under, in both spellings its readers use.

    This returned ``Sampling.__dict__`` -- ``n_drawn``/``method`` -- while ``plot_spatial_3d``
    reads ``n_kept``/``how``, which were never there, so a strided view was never disclosed: the
    "lie the reader can rotate" this module's docstring forbids (hunt 2026-09-30,
    u20b-viz-rest-14). The record now carries both names and the caption sentence itself.
    """
    n_total = int(getattr(sampling, "n_total", 0))
    n_drawn = int(getattr(sampling, "n_drawn", n_total))
    method = str(getattr(sampling, "method", ""))
    caption = sampling.caption() if hasattr(sampling, "caption") else ""
    return {
        "n_total": n_total,
        "n_drawn": n_drawn,
        "method": method,
        "seed": int(getattr(sampling, "seed", 0)),
        "sampled": n_drawn < n_total,
        "caption": caption,
        # The names plot_spatial_3d reads.
        "n_kept": n_drawn,
        "how": method,
    }


def section_panels(
    xy: Any,
    sections: Any,
    values: Any = None,
    *,
    categorical: bool = False,
    title: str = "",
    order: list[str] | None = None,
    ncols: int = 4,
):
    """One panel per section, on one shared scale. ``(figure, note)``.

    This is what ``spatial.sections`` promised in the catalogue and no function implemented: until
    now a multi-section object drew every section overlaid in one frame. Panels are capped by the
    shared grid limit and the number dropped is reported rather than silently truncated.
    """
    import numpy as np

    from spatialomicsgym.viz.render import spatial as spatial_render

    arr = np.asarray(xy, dtype=float)[:, :2]
    labels = np.asarray(sections).astype(str)
    names = order or sorted(dict.fromkeys(labels.tolist()))
    rows, cols, cap = base.panel_grid(len(names), ncols=ncols)
    shown, dropped = names[:cap], names[cap:]

    fig, axes = base.canvas(rows, cols, panel_w=3.4, panel_h=3.2)
    vmin = vmax = None
    if values is not None and not categorical:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            vmin, vmax = float(finite.min()), float(finite.max())
    all_values = None if values is None else np.asarray(values)

    for ax, name in zip(axes, shown, strict=False):
        mask = labels == name
        if all_values is None:
            # Nothing to paint: the positions alone, with the tissue-map conventions.
            ax.scatter(arr[mask, 0], arr[mask, 1], s=base.point_size_for(int(mask.sum())), c="#4C6EF5", linewidths=0)
            ax.set_aspect("equal")
            if not ax.yaxis_inverted():
                ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # `vmin=`/`vmax=` are not tissue_scatter keywords -- it takes `vlim` -- so every call
            # raised TypeError, a bare `except TypeError` drew uncoloured dots, and the caption
            # still promised "one shared colour scale" (hunt 2026-09-30, u20b-viz-rest-2). The
            # fallback is gone: a drawing error is an error, not a blank panel. A categorical
            # value takes its palette from the whole column, so a level is one colour throughout.
            spatial_render.tissue_scatter(
                ax,
                arr[mask],
                all_values[mask],
                categorical=categorical,
                vlim=(vmin, vmax),
                palette_labels=all_values if categorical else None,
                show_legend=False,
            )
        ax.set_title(f"{name} (n={int(mask.sum())})", fontsize=9)
    for ax in axes[len(shown) :]:
        base.blank(ax)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    legend = False
    if all_values is not None and categorical:
        from spatialomicsgym.viz import palette

        colours, pooled = palette.categorical(all_values.astype(object))
        legend = base.shared_legend(fig, list(colours), colours, len(pooled))
    return fig, {
        "n_sections": len(names),
        "n_drawn": len(shown),
        "n_not_drawn": len(dropped),
        "not_drawn": dropped[:20],
        "shared_scale": None if vmin is None else [vmin, vmax],
        "shared_legend": legend,
    }


#: Above this many distinct z values the z is continuous rather than a stack of planes, and one bar
#: per value is neither readable nor cheap to draw. The profile's ``z_is_layered`` uses the same cut.
MAX_DEPTH_PLANES = 500

#: How many equal-width bins a continuous z is summarised in, when it is.
DEPTH_BINS = 100


def depth_profile(z: Any, values: Any = None, *, title: str = "", z_units: str = ""):
    """Cells per z plane, and the mean value per plane. ``(figure, note)``.

    The figure that makes a broken z axis visible. Coincident planes show as one bar where there
    should be several; a z built from a slice ordinal shows as perfectly even spacing where the
    real sections are not evenly cut.

    Cells with no z are left out and counted (``n_nonfinite``): ``set()`` kept each NaN as its own
    plane, so 20 uncovered cells became 20 extra planes, a NaN z range and a false "spacing is not
    uniform" (hunt 2026-09-30, u20b-viz-rest-27). A continuous z -- more than
    :data:`MAX_DEPTH_PLANES` distinct values -- is binned, and the note says so.
    """
    import numpy as np

    zz_all = np.asarray(z, dtype=float).ravel()
    finite = np.isfinite(zz_all)
    zz = zz_all[finite]
    vals = None if values is None else np.asarray(values, dtype=float).ravel()[finite]

    planes, inverse, counts = np.unique(zz, return_inverse=True, return_counts=True)
    n_distinct = int(len(planes))
    binned = n_distinct > MAX_DEPTH_PLANES
    if binned:
        edges = np.linspace(float(zz.min()), float(zz.max()), DEPTH_BINS + 1)
        inverse = np.clip(np.digitize(zz, edges[1:-1]), 0, DEPTH_BINS - 1)
        planes = (edges[:-1] + edges[1:]) / 2.0
        counts = np.bincount(inverse, minlength=DEPTH_BINS)

    n_panels = 1 if vals is None else 2
    fig, axes = base.canvas(1, n_panels, panel_w=5.0, panel_h=3.4)
    width = (np.min(np.diff(planes)) * 0.7) if len(planes) > 1 else 1.0
    axes[0].bar(planes, counts, width=width)
    base.finish(fig, axes[0], "Cells per plane", f"z ({z_units})" if z_units else "z", "cells")

    if vals is not None:
        # One pass over the cells rather than one per plane: the per-plane mask was O(n x planes).
        ok = np.isfinite(vals)
        sums = np.bincount(inverse[ok], weights=vals[ok], minlength=len(planes))
        hits = np.bincount(inverse[ok], minlength=len(planes))
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(hits > 0, sums / np.maximum(hits, 1), np.nan)
        axes[1].plot(planes, means, marker="o" if not binned else None)
        base.finish(fig, axes[1], "Mean value per plane", f"z ({z_units})" if z_units else "z", "mean")
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    gaps = np.diff(planes) if (len(planes) > 1 and not binned) else np.array([])
    return fig, {
        "n_planes": n_distinct,
        "z_range": [float(zz.min()), float(zz.max())] if len(zz) else None,
        "gaps": [float(g) for g in gaps[:32]],
        "uniform": bool(len(gaps) and np.allclose(gaps, gaps[0], rtol=0.25)),
        "n_nonfinite": int((~finite).sum()),
        "binned": binned,
        "n_bins": DEPTH_BINS if binned else n_distinct,
    }
