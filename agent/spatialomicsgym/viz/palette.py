"""Colour, and the rules that stop it lying.

Colour is where a figure misleads most easily, and three of the four ways are defaults rather
than mistakes: each panel of a grid autoscaling to its own range, a sequential map stretched over
signed values, and a diverging map whose white is not at zero. So the colour limits for every
plot in this toolkit come from here, and they come with a sentence for the caption saying what
was done.

The categorical palette is not re-invented. The repository already has a sixty-colour qualitative
palette with a pooled grey beyond it and a legend that wraps in columns, and the wraparound bug in
it has already been found and fixed once. This module re-exports that rather than forking it.

The one exception is :func:`categorical_hex`, the same rule answered as hex strings for the
explorer, whose process must not import matplotlib. It holds the four tables' colours as text and
reads every other part of the rule -- the table order, the limit, the pooled grey, the natural sort
-- from ``postanalysis.plots``; a parity test pins it to ``_category_colours`` colour for colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Which colour map suits which kind of quantity. Named by what is being shown rather than by
#: what it looks like, so a caller asks for "a proportion" and gets a map suited to one.
SEQUENTIAL: dict[str, str] = {
    "expression": "viridis",
    "density": "magma",
    "proportion": "cividis",
    "count": "viridis",
    "score": "viridis",
    "pvalue": "rocket_r",
    "distance": "mako",
    # A pseudotime or other ordering. Named here so every word a caller passes is a word this module
    # decided about, rather than one that fell through to the default (hunt 2026-09-30, u20b-viz-rest-5).
    "ordering": "viridis",
}

#: Quantities that have a meaningful zero and must be centred on it.
DIVERGING: dict[str, str] = {
    "logfc": "RdBu_r",
    "zscore": "RdBu_r",
    "correlation": "RdBu_r",
    "enrichment": "RdBu_r",
    "difference": "RdBu_r",
    # The word the standardised marker heatmap, the pathway activity map and the neighbourhood
    # z-score all pass. It was in neither table, so all three got viridis autoscaled to the data
    # with no centring -- the sequential-over-signed failure this module exists to prevent
    # (hunt 2026-09-30, u20b-viz-rest-5).
    "signed": "RdBu_r",
}


def is_diverging(semantic: str) -> bool:
    return str(semantic).lower() in DIVERGING


def continuous(semantic: str, override: str = "") -> str:
    """The colour map for a quantity, or the caller's override."""
    if override:
        return str(override)
    key = str(semantic).lower()
    return DIVERGING.get(key) or SEQUENTIAL.get(key) or "viridis"


def categorical(labels: Sequence[Any], uniq: Sequence[Any] | None = None) -> tuple[dict[str, Any], frozenset[str]]:
    """The shared sixty-colour categorical mapping, plus the levels that were pooled.

    Delegates to the repository's own implementation so a category is the same colour in a figure,
    in the report and in the mark. Returns the mapping and the set of pooled level names, because
    a legend that silently merges forty rare types into one grey is a figure that draws no
    distinction while implying one.
    """
    import numpy as np

    from spatialomicsgym.postanalysis.plots import _category_colours

    if uniq is None:
        # The helper takes the level set as a separate argument on purpose -- a caller drawing a
        # facet wants every panel to use the union of levels, not the subset present in that
        # panel, or one cluster is blue in one panel and green in the next. When no union is
        # given the levels are the ones actually present, in first-seen order so the same data
        # always draws the same figure.
        uniq = list(dict.fromkeys(np.asarray(labels).astype(object).tolist()))
    return _category_colours(labels, list(uniq))


def pooled_colour() -> str:
    """The grey every level past the palette's sixty shares. The repository's own, not a second one."""
    from spatialomicsgym.postanalysis.plots import _POOLED_COLOUR

    return str(_POOLED_COLOUR)


def max_categories() -> int:
    from spatialomicsgym.postanalysis.plots import _MAX_DISTINCT_CATEGORIES

    return int(_MAX_DISTINCT_CATEGORIES)


#: matplotlib's qualitative tables as ``to_hex`` prints them (matplotlib 3.10 and 3.11 agree), so the
#: explorer can colour a category exactly as a figure does without importing matplotlib. Which tables
#: are used, and in what order, is ``postanalysis.plots._QUALITATIVE_CMAPS``'s decision, not this
#: table's. test/test_categorical_hex_matches_the_figures.py compares every entry to matplotlib.
_TABLE_HEX: dict[str, tuple[str, ...]] = {
    "tab10": (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
        "#17becf",
    ),
    "tab20": (
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd",
        "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
        "#17becf", "#9edae5",
    ),
    "tab20b": (
        "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939", "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31",
        "#bd9e39", "#e7ba52", "#e7cb94", "#843c39", "#ad494a", "#d6616b", "#e7969c", "#7b4173", "#a55194",
        "#ce6dbd", "#de9ed6",
    ),
    "tab20c": (
        "#3182bd", "#6baed6", "#9ecae1", "#c6dbef", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2", "#31a354",
        "#74c476", "#a1d99b", "#c7e9c0", "#756bb1", "#9e9ac8", "#bcbddc", "#dadaeb", "#636363", "#969696",
        "#bdbdbd", "#d9d9d9",
    ),
}  # fmt: skip


@dataclass(frozen=True)
class PooledLevels:
    """The levels that share the pooled grey, because the palette ran out. Never silent.

    ``labels`` are the pooled levels in natural order, ``n_total`` how many observations they hold
    together (``None`` when no counts were given), ``colour`` the grey.
    """

    labels: tuple[Any, ...]
    n_levels: int
    n_total: int | None
    colour: str

    @property
    def label(self) -> str:
        """The legend entry, worded as the figures word it."""
        return f"other ({self.n_levels} categories)"

    def as_dict(self) -> dict[str, Any]:
        """``{label, n_levels, n_total, colour}``: the explorer contract's ``pooled`` object."""
        return {"label": self.label, "n_levels": self.n_levels, "n_total": self.n_total, "colour": self.colour}


def categorical_hex(
    levels: Sequence[Any],
    counts: Sequence[int] | Mapping[Any, int] | None = None,
    *,
    max_levels: int = 60,
) -> tuple[dict[Any, str], PooledLevels | None]:
    """``(hex colour per level, what was pooled)`` -- ``postanalysis.plots._category_colours`` in hex.

    The same rule, step for step, so a category is the same colour in the explorer as in every
    static figure: the levels are put in natural order first, as a figure puts them (a stable sort,
    so the order they arrive in cannot change a colour); ten levels or fewer take ``tab10``; up to
    ``max_levels`` take the 60-colour table, each level the colour at its natural position; above it
    the ``max_levels - 1`` levels with the most observations keep a colour -- ties broken on the
    label, larger natural key first, exactly as the figures break them -- and are coloured in
    natural order, and every other level is pooled into one grey. The pooled levels are absent from
    the mapping, as they are from the figures', and named in the returned :class:`PooledLevels`.

    ``counts`` are the observations per level over the FULL data -- a sequence aligned with
    ``levels``, or anything keyed by level: a mapping, or a pandas Series such as ``value_counts()``,
    which is read by its index -- because which levels are pooled must not depend on what a sample
    happened to draw. A level the counts do not mention counts as zero; no counts at all pools by
    label alone. ``levels`` must be distinct. Imports nothing heavy.
    """
    from collections import abc

    from spatialomicsgym.postanalysis.plots import (
        _MAX_DISTINCT_CATEGORIES,
        _POOLED_COLOUR,
        _QUALITATIVE_CMAPS,
        _natural_key,
    )

    levels = list(levels)
    if len(set(levels)) != len(levels):
        raise ValueError("categorical_hex: levels must be distinct")
    if not 1 <= int(max_levels) <= _MAX_DISTINCT_CATEGORIES:
        raise ValueError(f"categorical_hex: max_levels must be between 1 and {_MAX_DISTINCT_CATEGORIES}")
    max_levels = int(max_levels)

    if counts is None:
        per_level: dict[Any, int] = {}
    elif isinstance(counts, abc.Mapping) or callable(getattr(counts, "items", None)):
        # Keyed by level. A pandas Series is not an ``abc.Mapping``, and read by position a
        # ``value_counts()`` -- ordered by count -- pooled the most common levels, not the rarest.
        keyed = dict(counts.items())
        per_level = {level: int(keyed[level]) for level in levels if level in keyed}
    else:
        values = list(counts)
        if len(values) != len(levels):
            raise ValueError("categorical_hex: counts must have one entry per level")
        per_level = {level: int(n) for level, n in zip(levels, values, strict=True)}
    # The figures colour by position among the natural-sorted levels; colouring by position in the
    # caller's list agreed with them only when the caller happened to pass that order.
    levels = sorted(levels, key=_natural_key)

    palette = [colour for name in _QUALITATIVE_CMAPS for colour in _TABLE_HEX[name]]
    if len(levels) <= 10:
        palette = list(_TABLE_HEX["tab10"])

    if len(levels) <= max_levels:
        return {level: palette[i] for i, level in enumerate(levels)}, None

    keep = sorted(
        sorted(levels, key=lambda level: (per_level.get(level, 0), _natural_key(level)), reverse=True)[
            : max_levels - 1
        ],
        key=_natural_key,
    )
    kept = set(keep)
    pooled = tuple(level for level in levels if level not in kept)
    pooled_total = sum(per_level.get(level, 0) for level in pooled) if counts is not None else None
    return (
        {level: palette[i] for i, level in enumerate(keep)},
        PooledLevels(labels=pooled, n_levels=len(pooled), n_total=pooled_total, colour=str(_POOLED_COLOUR)),
    )


def _percentile(values: Any, spec: str) -> float | None:
    import numpy as np

    text = str(spec).strip().lower()
    if not text:
        return None
    if text.startswith("p"):
        try:
            return float(np.nanpercentile(values, float(text[1:])))
        except Exception:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def limits(
    values: Any,
    vmin: str = "",
    vmax: str = "",
    *,
    semantic: str = "expression",
) -> tuple[float | None, float | None, str]:
    """Colour limits, and the sentence that discloses them.

    A diverging quantity is symmetrised around zero unless both ends were given explicitly,
    because a fold change of plus two reading as unchanged is a picture that says the opposite of
    the data. A percentile clip is always disclosed, because the colour bar's top label would
    otherwise be read as the maximum.
    """
    import numpy as np

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return (None, None, "")

    low = _percentile(finite, vmin)
    high = _percentile(finite, vmax)
    notes: list[str] = []

    if is_diverging(semantic) and (low is None or high is None):
        bound = float(np.nanmax(np.abs(finite)))
        low = -bound if low is None else low
        high = bound if high is None else high
        notes.append("the colour scale is centred on zero")

    data_lo, data_hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if low is not None and low > data_lo:
        notes.append(f"values below {low:.3g} are clipped")
    if high is not None and high < data_hi:
        notes.append(f"values above {high:.3g} are clipped (the data reaches {data_hi:.3g})")

    return (low, high, "; ".join(notes))


class PanelScales:
    """Whether the panels of a grid share a colour scale, and the sentence that says so.

    Not optional and not a boolean buried in a call: a grid whose panels each autoscale is the
    single most common way a multi-gene figure misleads, because a gene with a maximum of 0.1 and
    a gene with a maximum of 14.7 come out equally bright. Construct one of these and the caption
    is written for you.
    """

    def __init__(self, shared: bool, per_panel: list[tuple[float, float]], names: list[str]) -> None:
        self.shared = bool(shared)
        self.per_panel = list(per_panel)
        self.names = list(names)

    @classmethod
    def build(
        cls, panels: dict[str, Any], *, share: bool, vmin: str = "", vmax: str = "", semantic: str = "expression"
    ) -> PanelScales:
        import numpy as np

        def effective(values: Any) -> tuple[float | None, float | None]:
            """The range the colour bar will SHOW: the limit when one was set, the data otherwise.

            `limits` answers None for "let the axes autoscale", which is the right instruction to
            matplotlib and the wrong thing to print in a caption -- a reader told the panels have
            different scales needs the numbers, and an empty pair reads as a bug.
            """
            low, high, _note = limits(values, vmin, vmax, semantic=semantic)
            finite = np.asarray(values, dtype=float).ravel()
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                return (low, high)
            return (
                float(low) if low is not None else float(np.nanmin(finite)),
                float(high) if high is not None else float(np.nanmax(finite)),
            )

        names = list(panels)
        if share:
            stacked = (
                np.concatenate([np.asarray(v, dtype=float).ravel() for v in panels.values()]) if names else np.array([])
            )
            pair = effective(stacked)
            return cls(True, [pair] * len(names), names)
        return cls(False, [effective(panels[name]) for name in names], names)

    def for_panel(self, index: int) -> tuple[float | None, float | None]:
        if not self.per_panel:
            return (None, None)
        return self.per_panel[min(index, len(self.per_panel) - 1)]

    def caption(self) -> str:
        """What a reader must be told about comparing these panels."""
        if len(self.names) <= 1:
            return ""
        if self.shared:
            low, high = self.for_panel(0)
            if low is None or high is None:
                return "All panels share one colour scale, so brightness is comparable between them."
            return (
                f"All panels share one colour scale ({low:.3g} to {high:.3g}), so brightness is "
                "comparable between them."
            )
        spans = ", ".join(
            f"{name} {lo:.3g}-{hi:.3g}"
            for name, (lo, hi) in zip(self.names[:4], self.per_panel[:4], strict=False)
            if lo is not None and hi is not None
        )
        more = "" if len(self.names) <= 4 else f", and {len(self.names) - 4} more"
        return f"Each panel has its own colour scale ({spans}{more}), so brightness is NOT comparable between panels."


class Sampling:
    """How many marks were drawn out of how many exist, and by what rule.

    Required whenever fewer points are drawn than the dataset holds. A picture of a sample that
    does not say it is a sample is the quietest of the misleading figures, because nothing about
    it looks wrong.
    """

    def __init__(self, n_total: int, n_drawn: int, method: str, seed: int = 0) -> None:
        self.n_total = int(n_total)
        self.n_drawn = int(n_drawn)
        self.method = str(method)
        self.seed = int(seed)

    @property
    def sampled(self) -> bool:
        return self.n_drawn < self.n_total

    def caption(self) -> str:
        if not self.sampled:
            return ""
        return (
            f"Drawn from {self.n_drawn:,} of {self.n_total:,} observations "
            f"({self.method}, seed {self.seed}); the picture is a sample."
        )
