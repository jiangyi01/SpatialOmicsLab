"""What every communication adapter is handed, what it answers, and the answer shapes they share.

An adapter is one backend's truth: which of the six facets its real output can give, read from the files it really
writes, and what it cannot give said in ``warnings``. Its answers are plain JSON-able dicts; NaN is turned into
``None`` and the size checked by the caller (:func:`spatialomicsgym.viz.ccc.child.serve`). No payload key is ever
``note``, ``reason``, ``detail`` or ``caption``: the child's parent rewrites the strings under those keys as prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from spatialomicsgym.viz.ccc.errors import CCCRefusal
from spatialomicsgym.viz.ccc.facets import MAX_GROUPS, MAX_RANK_ROWS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialomicsgym.viz.h5lite import H5AD


@dataclass
class Source:
    """One file of a bundle: its role (``detect``'s), its open descriptor, and its separator (``None`` for an h5ad,
    whose opened :class:`~spatialomicsgym.viz.h5lite.H5AD` the child puts in ``h5``). ``fields`` is what ``detect``
    read from the file's name (``mode`` for a squidpy Ripley table): the reader never sees the name itself."""

    role: str
    fd: int
    sep: str | None
    h5: H5AD | None = None
    fields: dict[str, str] = field(default_factory=dict)


class Adapter(Protocol):
    backend: str

    def describe(self, sources: list[Source], limits: Mapping[str, int]) -> dict:
        """``{label, n_obs | None, unit, facets[], groups[], warnings[]}``: what this result offers."""
        ...

    def data(
        self,
        sources: list[Source],
        facet: str,
        params: Mapping[str, Any],
        group: dict | None,
        rows: np.ndarray | None,
        limits: Mapping[str, int],
    ) -> dict:
        """``{facet, payload, warnings}``: one facet's answer."""
        ...


# ------------------------------------------------------------------------------------------------ helpers
def by_role(sources: list[Source]) -> dict[str, Source]:
    """The bundle's sources by role (``detect.bundle_of`` gives one per role)."""
    return {s.role: s for s in sources}


def h5ad_of(sources: list[Source]) -> H5AD | None:
    source = by_role(sources).get("h5ad")
    return source.h5 if source is not None else None


def choice(params: Mapping[str, Any], name: str, allowed: list[str] | tuple[str, ...], default: str | None) -> str:
    """``params[name]`` when it is one of ``allowed``, ``default`` when absent; else ``bad_request``."""
    value = params.get(name, default)
    if value is None or value not in allowed:
        shown = ", ".join(str(a) for a in list(allowed)[:12])
        raise CCCRefusal("bad_request", f"The request's {name} is not one this result has ({shown}).")
    return str(value)


def not_offered(facet: str, why: str) -> CCCRefusal:
    return CCCRefusal("unsupported", f"This result offers no {facet} view: {why}")


def symmetric(values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=np.float64)
    return (
        values.ndim == 2 and values.shape[0] == values.shape[1] and bool(np.allclose(values, values.T, equal_nan=True))
    )


def as_list(values: np.ndarray | None) -> list | None:
    return None if values is None else np.asarray(values, dtype=np.float64).tolist()


def matrix_payload(
    *,
    key: str,
    key_label: str,
    stat: str,
    group: dict,
    labels: list[str],
    values: np.ndarray,
    n: list[int] | None = None,
    pooled: dict | None = None,
    mean: np.ndarray | None = None,
    z: np.ndarray | None = None,
    p: np.ndarray | None = None,
    permutations: int = 0,
    seed: int = 0,
    n_links: int | None = None,
    rows_used: int | None = None,
    rows_total: int | None = None,
) -> dict:
    """The ``matrix`` payload. ``sum`` holds the matrix ``stat`` names (``"sum"`` for COMMOT's derived sums; the
    tool's own statistic -- a count, a z-score, a correlation -- for a group-level result); ``mean`` only where a
    per-pair mean can be derived."""
    return {
        "facet": "matrix",
        "key": key,
        "key_label": key_label,
        "stat": stat,
        "group": group,
        "labels": list(labels),
        "n": n,
        "pooled": pooled,
        "sum": as_list(values),
        "mean": as_list(mean),
        "z": as_list(z),
        "p": as_list(p),
        "orientation": "sender rows, receiver columns",
        "symmetric": symmetric(values),
        "permutations": int(permutations),
        "seed": int(seed),
        "n_links": n_links,
        "rows_used": rows_used,
        "rows_total": rows_total,
    }


def cap_groups(payload: dict) -> list[str]:
    """A tool's own groups-by-groups matrix cut to the :data:`facets.MAX_GROUPS` groups with the most signal (the sum
    of the absolute values of their row and column), in their order; the warning that says so, or ``[]``.

    A tool's statistic cannot be pooled into an "other" group the way COMMOT's sums are -- a z-score or a correlation
    of a pooled group is not the pool of the groups' z-scores -- so the weakest groups are left out instead.
    """
    labels = payload["labels"]
    if len(labels) <= MAX_GROUPS:
        return []
    values = np.abs(np.nan_to_num(np.asarray(payload["sum"], dtype=np.float64)))
    keep = np.sort(np.argsort(-(values.sum(axis=0) + values.sum(axis=1)), kind="stable")[:MAX_GROUPS])
    for name in ("sum", "mean", "z", "p"):
        if payload.get(name) is not None:
            payload[name] = np.asarray(payload[name], dtype=np.float64)[np.ix_(keep, keep)].tolist()
    payload["labels"] = [labels[i] for i in keep.tolist()]
    if payload.get("n") is not None:
        payload["n"] = [payload["n"][i] for i in keep.tolist()]
    return [f"The {MAX_GROUPS} of {len(labels)} groups with the most signal are shown."]


def ranking_payload(
    *,
    table: str,
    label: str,
    columns: list[tuple[str, str]],
    rows: list[list],
    n_total: int,
    sort_by: str | None,
    desc: bool,
    heatmap: tuple[list[str], list[str], np.ndarray] | None = None,
) -> dict:
    """The ``ranking`` payload: at most :data:`facets.MAX_RANK_ROWS` rows, already sorted by the caller.

    ``heatmap`` (row labels, column labels, values) makes it a heatmap and adds ``rows_labels``/``cols``/``values``.
    """
    kept = rows[:MAX_RANK_ROWS]
    out = {
        "facet": "ranking",
        "table": table,
        "label": label,
        "kind": "heatmap" if heatmap is not None else "bars",
        "columns": [{"name": name, "kind": kind} for name, kind in columns],
        "rows": kept,
        "n_total": int(n_total),
        "truncated": int(n_total) > len(kept),
        "sort": {"by": sort_by, "desc": bool(desc)},
    }
    if heatmap is not None:
        row_labels, col_labels, values = heatmap
        out["rows_labels"] = list(row_labels)[:MAX_RANK_ROWS]
        out["cols"] = list(col_labels)
        out["values"] = np.asarray(values, dtype=np.float64)[:MAX_RANK_ROWS].tolist()
    return out


def frame_rows(frame: Any, sort_by: str | None, desc: bool) -> tuple[list[tuple[str, str]], list[list]]:
    """A table's columns (``number`` when the column parses as numbers, else ``text``) and its rows, sorted by
    ``sort_by`` (missing values last) and cut to :data:`facets.MAX_RANK_ROWS`."""
    import pandas as pd

    from spatialomicsgym.viz.ccc.tables import labels, numbers

    if sort_by is not None and len(frame):
        frame = frame.sort_values(sort_by, ascending=not desc, na_position="last", kind="stable")
    frame = frame.head(MAX_RANK_ROWS)
    columns: list[tuple[str, str]] = []
    data: list[list] = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            columns.append((str(name), "number"))
            data.append(numbers(series).tolist())
        else:
            columns.append((str(name), "text"))
            data.append(labels(series))
    rows = [list(r) for r in zip(*data, strict=True)] if data else []
    return columns, rows


#: The leading column a per-section run's long tables carry (``tools/worker_utils.per_section``).
SECTION_COLUMN = "section"


def has_section(header: list[str]) -> bool:
    """Whether a table is a per-section run's long table: its FIRST column is ``section``, as the workers write it."""
    return bool(header) and str(header[0]).strip() == SECTION_COLUMN


def table_sections(source: Source, limits: Mapping[str, int]) -> list[str]:
    """A long per-section table's sections, in the order the file lists them (that column read alone)."""
    from spatialomicsgym.viz.ccc import tables

    seen: dict[str, None] = {}
    for block in tables.blocks(source.fd, source.sep or ",", limits, usecols=[SECTION_COLUMN], dtype=str):
        for value in tables.labels(block[SECTION_COLUMN]):
            seen.setdefault(value, None)
    return list(seen)


def sections_of_tables(sources: list[Source], roles: tuple[str, ...], limits: Mapping[str, int]) -> list[str]:
    """Every section the bundle's long tables of ``roles`` name, in first-seen order; ``[]`` for a single result."""
    from spatialomicsgym.viz.ccc import tables

    seen: dict[str, None] = {}
    for source in sources:
        if source.role in roles and has_section(tables.header(source.fd, source.sep or ",", limits)):
            for label in table_sections(source, limits):
                seen.setdefault(label, None)
    return list(seen)


def per_section_fields(sections: list[str]) -> dict[str, Any]:
    """A describe answer's mode fields: per-section 2D when any table names sections, else what the tables cannot
    tell (``mode: None``)."""
    if not sections:
        return {"mode": None, "sections": [], "frame": None, "legacy_rank_z": False}
    return {
        "mode": "per-section-2d",
        "sections": [{"label": label, "z_um": None} for label in sections],
        "frame": None,
        "legacy_rank_z": False,
    }


def one_section(frame: Any, params: Mapping[str, Any], *, required: bool) -> tuple[Any, str | None, list[str]]:
    """``(frame, section, warnings)`` of a long per-section table: its rows of the ``section`` asked for, that column
    dropped. Without one asked: every row (the column kept) -- or, when ``required`` (a table whose rows are one
    section's own: one value per type, a curve per group), the first section's, and the warning saying so. A table
    with no ``section`` column refuses a section asked of it."""
    asked = params.get("section")
    if SECTION_COLUMN not in frame.columns or (len(frame.columns) and str(frame.columns[0]) != SECTION_COLUMN):
        if asked is not None:
            raise CCCRefusal("bad_request", "This result names no sections, so no section can be asked for.")
        return frame, None, []
    from spatialomicsgym.viz.ccc import tables

    labels = list(dict.fromkeys(tables.labels(frame[SECTION_COLUMN])))
    if asked is None and not required:
        return frame, None, []
    if asked is not None and asked not in labels:
        raise CCCRefusal(
            "bad_request", f"The request's section is not one of this result's ({', '.join(labels[:12])})."
        )
    label = asked if asked is not None else (labels[0] if labels else None)
    warnings = []
    if asked is None and len(labels) > 1:
        warnings.append(
            f"This per-section 2D result has {len(labels)} sections; section {label} is shown. Choose another with "
            "section."
        )
    kept = frame[[v == label for v in tables.labels(frame[SECTION_COLUMN])]].drop(columns=[SECTION_COLUMN])
    return kept.reset_index(drop=True), label, warnings


def halve_rows(payload: dict) -> bool:
    """kept the first half of the rows"""
    rows = payload.get("rows")
    if not rows or len(rows) < 2:
        return False
    keep = len(rows) // 2
    payload["rows"] = rows[:keep]
    for name in ("rows_labels", "values"):
        if isinstance(payload.get(name), list):
            payload[name] = payload[name][:keep]
    payload["truncated"] = True
    return True


__all__ = [
    "SECTION_COLUMN",
    "Adapter",
    "Source",
    "as_list",
    "by_role",
    "cap_groups",
    "choice",
    "frame_rows",
    "h5ad_of",
    "halve_rows",
    "has_section",
    "matrix_payload",
    "not_offered",
    "one_section",
    "per_section_fields",
    "ranking_payload",
    "sections_of_tables",
    "symmetric",
    "table_sections",
]
