"""Scatter plots in an abstract space: embeddings, volcanoes, anything with two axes and marks.

Separate from the tissue module because the axes mean something different. A tissue map has a
physical coordinate system with a fixed aspect ratio and an inverted y axis; an embedding has
neither, and forcing a square aspect on a UMAP is as wrong as not forcing one on a section.
"""

from __future__ import annotations

from typing import Any

from spatialomicsgym.viz import palette as _palette
from spatialomicsgym.viz.render import base as _base


def embedding_scatter(
    ax: Any,
    xy: Any,
    values: Any,
    *,
    categorical: bool,
    title: str = "",
    value_label: str = "",
    basis_label: str = "",
    point_size: float | None = None,
    cmap: str = "viridis",
    vlim: tuple[float | None, float | None] = (None, None),
    legend: str = "right margin",
    background: Any = None,
    fmt: str = "png",
    show_colorbar: bool = True,
    palette_labels: Any = None,
) -> dict[str, Any]:
    """One embedding panel.

    ``background`` draws the cells that are not in this facet in grey underneath, which is what
    makes a split panel readable: without it each panel is an island and a reader cannot see
    where its cells sit in the whole.

    ``palette_labels`` is the whole column when this panel is one facet of several, so the palette
    is built once over every level and a level is the same colour in each facet.
    """
    import numpy as np

    coords = np.asarray(xy, dtype=float)[:, :2]
    note: dict[str, Any] = {"n_drawn": int(coords.shape[0]), "pooled": [], "rasterized": False}
    size = point_size if point_size else _base.point_size_for(coords.shape[0])

    if background is not None:
        bg = np.asarray(background, dtype=float)[:, :2]
        ax.scatter(bg[:, 0], bg[:, 1], s=max(1.0, size * 0.6), c="#dfe3e8", linewidths=0, zorder=1)

    if categorical:
        labels = np.asarray(values).astype(object)
        reference = labels if palette_labels is None else np.asarray(palette_labels).astype(object)
        colours, pooled = _palette.categorical(reference)
        note["pooled"] = sorted(str(p) for p in pooled)
        shown = 0
        for level, colour in colours.items():
            mask = labels == level
            if not mask.any():
                continue
            shown += 1
            artist = ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=size,
                c=[colour],
                label=str(level),
                linewidths=0,
                zorder=2,
            )
            if _base.rasterize_if_large(artist, int(mask.sum()), fmt):
                note["rasterized"] = True
        # Pooled levels have no entry in `colours`, so the loop above never drew them: the rarest
        # types of a fine-grained atlas disappeared from the embedding (hunt 2026-09-30,
        # u20b-viz-rest-9). They are drawn in the shared grey, with one legend entry.
        rest = _base.pooled_mask(labels, pooled)
        if rest is not None and rest.any():
            artist = ax.scatter(
                coords[rest, 0],
                coords[rest, 1],
                s=size,
                c=[_palette.pooled_colour()],
                label=f"{len(pooled)} pooled levels",
                linewidths=0,
                zorder=2,
            )
            if _base.rasterize_if_large(artist, int(rest.sum()), fmt):
                note["rasterized"] = True
            note["n_pooled_drawn"] = int(rest.sum())
        note["n_levels"] = shown
        if legend == "on data":
            for level in colours:
                mask = labels == level
                if mask.any():
                    ax.annotate(
                        str(level),
                        (float(coords[mask, 0].mean()), float(coords[mask, 1].mean())),
                        ha="center",
                        va="center",
                        fontsize=8,
                        zorder=3,
                    )
        elif legend != "none" and 0 < shown <= 60:
            ncol = 1 if shown <= 20 else (2 if shown <= 40 else 3)
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
                fontsize=8,
                markerscale=2.0,
                ncol=ncol,
                title=value_label or None,
            )
    else:
        numbers = np.asarray(values, dtype=float)
        artist = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=numbers,
            s=size,
            cmap=cmap,
            vmin=vlim[0],
            vmax=vlim[1],
            linewidths=0,
            zorder=2,
        )
        if _base.rasterize_if_large(artist, numbers.size, fmt):
            note["rasterized"] = True
        finite = numbers[np.isfinite(numbers)]
        if show_colorbar and finite.size and float(finite.min()) != float(finite.max()):
            bar = ax.figure.colorbar(artist, ax=ax, shrink=0.72, pad=0.02)
            if value_label:
                bar.set_label(value_label, fontsize=9)
        elif finite.size and float(finite.min()) == float(finite.max()):
            note["constant_value"] = True

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if basis_label:
        ax.set_xlabel(f"{basis_label} 1", fontsize=8)
        ax.set_ylabel(f"{basis_label} 2", fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return note


def embedding_panels(
    xy: Any,
    panels: dict[str, Any],
    *,
    categorical: bool,
    scales: Any = None,
    ncols: int = 3,
    title: str = "",
    basis_label: str = "",
    point_size: float | None = None,
    cmap: str = "viridis",
    legend: str = "right margin",
    fmt: str = "png",
) -> tuple[Any, dict[str, Any]]:
    """A grid of embedding panels over the same coordinates."""
    names = list(panels)
    rows, cols, shown = _base.panel_grid(len(names), ncols)
    fig, axes = _base.canvas(rows, cols, panel_w=4.4, panel_h=4.0)
    notes: dict[str, Any] = {"panels": [], "dropped": max(0, len(names) - shown)}
    for i in range(shown):
        name = names[i]
        vlim = scales.for_panel(i) if scales is not None else (None, None)
        note = embedding_scatter(
            axes[i],
            xy,
            panels[name],
            categorical=categorical,
            title=name,
            value_label=name,
            basis_label=basis_label if i == 0 else "",
            point_size=point_size,
            cmap=cmap,
            vlim=vlim,
            legend=legend,
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


def facet_by(
    xy: Any,
    labels: Any,
    *,
    colour_by: Any,
    categorical: bool,
    ncols: int = 3,
    title: str = "",
    basis_label: str = "",
    point_size: float | None = None,
    cmap: str = "viridis",
    vlim: tuple[float | None, float | None] = (None, None),
    fmt: str = "png",
) -> tuple[Any, dict[str, Any]]:
    """One panel per level of *labels*, on shared axes, with the other cells greyed behind."""
    import numpy as np

    coords = np.asarray(xy, dtype=float)[:, :2]
    groups = np.asarray(labels).astype(object)
    levels = list(dict.fromkeys(groups.tolist()))
    rows, cols, shown = _base.panel_grid(len(levels), ncols)
    fig, axes = _base.canvas(rows, cols, panel_w=4.0, panel_h=3.8)
    notes: dict[str, Any] = {"panels": [], "dropped": max(0, len(levels) - shown)}
    colour_all = np.asarray(colour_by)
    for i in range(shown):
        level = levels[i]
        mask = groups == level
        note = embedding_scatter(
            axes[i],
            coords[mask],
            colour_all[mask],
            categorical=categorical,
            title=f"{level} (n={int(mask.sum()):,})",
            basis_label=basis_label if i == 0 else "",
            point_size=point_size,
            cmap=cmap,
            vlim=vlim,
            legend="none",
            background=coords,
            fmt=fmt,
            show_colorbar=False,
            # The palette was built per facet, from the levels present in it, so one cluster was
            # blue in one panel and green in the next -- under no legend at all (hunt 2026-09-30,
            # u20b-viz-rest-10). It is built once, over the whole column.
            palette_labels=colour_all if categorical else None,
        )
        note["name"] = str(level)
        note["n"] = int(mask.sum())
        notes["panels"].append(note)
        # Shared axes, so a panel with few cells is not silently magnified.
        axes[i].set_xlim(coords[:, 0].min(), coords[:, 0].max())
        axes[i].set_ylim(coords[:, 1].min(), coords[:, 1].max())
    for j in range(shown, len(axes)):
        _base.blank(axes[j])
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    notes["shared_axes"] = True
    if categorical:
        reference = colour_all.astype(object)
        colours, pooled = _palette.categorical(reference)
        notes["shared_legend"] = _base.shared_legend(fig, list(colours), colours, len(pooled))
    return fig, notes


def volcano(
    ax: Any,
    log_fold_change: Any,
    p_adjusted: Any,
    labels: Any,
    *,
    title: str = "",
    fc_threshold: float = 1.0,
    p_threshold: float = 0.05,
    label_top: int = 10,
) -> dict[str, Any]:
    """Effect against significance, with the thresholds drawn rather than implied.

    Both axes come from stored values; neither is derived from the other. A ranking that holds a
    test statistic but no fold change and no p-value cannot be drawn here, and the caller refuses
    upstream rather than substituting the statistic for an effect size.
    """
    import numpy as np

    fc = np.asarray(log_fold_change, dtype=float)
    padj = np.asarray(p_adjusted, dtype=float)
    names = np.asarray(labels).astype(object)
    finite = np.isfinite(fc) & np.isfinite(padj)
    fc, padj, names = fc[finite], padj[finite], names[finite]

    # A zero adjusted p-value is a floor, not a fact: it is what the correction returns when the
    # smallest representable value was reached. Plotting it at infinity would put a spike at the
    # top of the axis that no data supports, so it is clamped and the clamp is reported.
    floor = float(np.nextafter(0, 1))
    clamped = int((padj <= 0).sum())
    y = -np.log10(np.clip(padj, floor, None))

    significant = (padj < p_threshold) & (np.abs(fc) >= fc_threshold)
    ax.scatter(fc[~significant], y[~significant], s=8, c="#c3c9d0", linewidths=0, label="not selected")
    ax.scatter(fc[significant], y[significant], s=12, c="#0f7d76", linewidths=0, label="selected")
    ax.axhline(-np.log10(p_threshold), color="#98a2ac", linewidth=0.8, linestyle="--")
    for line in (-fc_threshold, fc_threshold):
        ax.axvline(line, color="#98a2ac", linewidth=0.8, linestyle="--")

    if label_top and significant.any():
        order = np.argsort(-(y * (np.abs(fc) + 1e-9)))
        shown = 0
        for idx in order:
            if not significant[idx]:
                continue
            ax.annotate(str(names[idx]), (fc[idx], y[idx]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            shown += 1
            if shown >= int(label_top):
                break

    ax.set_xlabel("log2 fold change", fontsize=9)
    ax.set_ylabel("-log10 adjusted p", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    return {
        "n_points": int(fc.size),
        "n_selected": int(significant.sum()),
        "p_floor_clamped": clamped,
        "thresholds": {"log2fc": float(fc_threshold), "p_adjusted": float(p_threshold)},
    }
