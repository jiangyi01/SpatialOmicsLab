"""squidpy's spatial statistics: neighbourhood enrichment, co-occurrence, Ripley's statistics, centrality and
spatial autocorrelation -- as the CSVs its worker writes and the h5ad's ``uns`` they come from.

What each gives, and what it cannot:

* **neighbourhood enrichment** -- a groups-by-groups z-score (and the count of neighbouring pairs), from squidpy's own
  permutations. squidpy reports no p-value, so none is shown.
* **co-occurrence** -- the CSV is one mean ratio per pair of groups (a matrix); the curves over distance are only in
  the h5ad (``uns['<key>_co_occurrence']['occ']``, groups x groups x bins).
* **Ripley** -- one curve per group (CSV, long); their p-values against random placement only in the h5ad.
* **centrality, Moran's I, Geary's C** -- tables. Autocorrelation is per GENE: nothing in it is per spot or per group.

The h5ad is recognised by its ``uns`` keys (``<key>_nhood_enrichment``, ``<key>_co_occurrence``,
``<key>_ripley_<F|G|L>``), the CSVs by their header; ``<key>`` is the obs column squidpy grouped by.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from spatialomicsgym.viz.ccc import geometry as geo
from spatialomicsgym.viz.ccc import h5 as h5x
from spatialomicsgym.viz.ccc import tables
from spatialomicsgym.viz.ccc.errors import CCCRefusal
from spatialomicsgym.viz.ccc.facets import MAX_CURVE_BINS, MAX_GROUPS, MAX_NNZ, fit

from .base import (
    Source,
    by_role,
    choice,
    frame_rows,
    h5ad_of,
    halve_rows,
    matrix_payload,
    not_offered,
    one_section,
    per_section_fields,
    ranking_payload,
    sections_of_tables,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialomicsgym.viz.h5lite import H5AD

_NHOOD_RE = re.compile(r"^(?P<ck>.+)_nhood_enrichment$")
_COOC_RE = re.compile(r"^(?P<ck>.+)_co_occurrence$")
_RIPLEY_RE = re.compile(r"^(?P<ck>.+)_ripley_(?P<mode>[FGL])$")
_CENTRALITY_RE = re.compile(r"^(?P<ck>.+)_centrality_scores$")


def _pooled(n_sections: int) -> str:
    """Why a per-section run's neighbourhood enrichment has no section to cut out (R2-1, the 2026-10-06 real case)."""
    return (
        f"This neighbourhood enrichment is one matrix for all {n_sections} sections: a per-section run builds each "
        "section's neighbour graph on its own (no edge crosses a section) and squidpy scores them in one test, so no "
        "section can be cut from it."
    )


#: An uns array a facet reads at most: 64 groups x 64 groups x 1,000 distance bins.
_UNS_MAX_VALUES = MAX_GROUPS * MAX_GROUPS * 1000
_NO_P = (
    "Neighbourhood enrichment has no p-values: squidpy reports a z-score per pair of groups from its own "
    "permutations, and nothing more."
)
_PER_GENE = (
    "Spatial autocorrelation (Moran's I, Geary's C) is per gene, not per spot or per group: it is a ranking of genes, "
    "with nothing to colour spots by."
)
_TABLE_LABELS = {
    "centrality": "Centrality scores per group",
    "moran": "Moran's I per gene (spatial autocorrelation)",
    "geary": "Geary's C per gene (spatial autocorrelation)",
}


def reduce_bins(x: np.ndarray, y: np.ndarray, max_bins: int = MAX_CURVE_BINS) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` with at most ``max_bins`` points: consecutive bins averaged in groups of ``ceil(len(x) / max_bins)``
    (the last group may be shorter). ``y``'s last axis is the bins. Within the cap nothing changes."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size <= int(max_bins):
        return x, y
    step = math.ceil(x.size / int(max_bins))
    starts = np.arange(0, x.size, step)
    counts = np.diff(np.append(starts, x.size))
    small_x = np.add.reduceat(x, starts) / counts
    small_y = np.add.reduceat(y, starts, axis=-1) / counts
    return small_x, small_y


def _uns_names(h5: H5AD) -> list[str]:
    group = h5.file.get("uns")
    return sorted(n for n in group if isinstance(n, str)) if group is not None else []


def _first(h5: H5AD | None, pattern: re.Pattern) -> re.Match | None:
    if h5 is None:
        return None
    return next((m for name in _uns_names(h5) if (m := pattern.match(name))), None)


def _obs_labels(h5: H5AD | None, ck: str | None) -> tuple[list[str] | None, list[int] | None, str | None]:
    """The obs column squidpy grouped by: ``(labels, sizes, id)``, or Nones when the file has no such column."""
    if h5 is None or ck is None:
        return None, None, None
    column = h5.obs_column(ck)
    if column is None:
        return None, None, None
    codes, labels, _ = h5x.group_codes(h5, column.index)
    return labels, np.bincount(codes[codes >= 0], minlength=len(labels)).tolist(), f"obs.{column.index}"


def _square_csv(
    source: Source, limits: Mapping[str, int], params: Mapping[str, Any] | None = None
) -> tuple[list[str], np.ndarray, str | None, list[str]]:
    """A groups-by-groups CSV: ``(groups, values, section, warnings)``. A per-section run's long table (a leading
    ``section`` column, one block of rows per section) is read for one section (:func:`base.one_section`)."""
    frame, section, warnings = one_section(
        tables.read(source.fd, source.sep or ",", limits), params or {}, required=True
    )
    rows, cols, values = tables.square(frame)
    if rows != cols:
        raise CCCRefusal(
            "unsupported", "A squidpy groups-by-groups table does not have its groups as both rows and columns."
        )
    return rows, values, section, warnings


class SquidpyAdapter:
    backend = "squidpy"

    def describe(self, sources: list[Source], limits: Mapping[str, int]) -> dict:
        roles = by_role(sources)
        h5 = h5ad_of(sources)
        nhood, cooc, ripley = _first(h5, _NHOOD_RE), _first(h5, _COOC_RE), _first(h5, _RIPLEY_RE)
        for role, source in roles.items():
            if role != "h5ad" and not tables.header(source.fd, source.sep or ",", limits):
                raise CCCRefusal("unsupported", f"squidpy's {role} table is empty.")
        centrality = _first(h5, _CENTRALITY_RE)
        if h5 is not None and not (nhood or cooc or ripley or centrality or {"moranI", "gearyC"} & set(_uns_names(h5))):
            raise CCCRefusal("unsupported", "This h5ad holds no squidpy result in uns.")
        facets: list[dict] = []
        warnings: list[str] = []
        keys = []
        if nhood or {"zscore", "count"} & set(roles):
            keys.append({"key": "nhood_enrichment", "label": "Neighbourhood enrichment (z-score)"})
            warnings.append(_NO_P)
        if "cooc_mean" in roles:
            keys.append({"key": "co_occurrence_mean", "label": "Co-occurrence ratio (mean over distances)"})
        if keys:
            facets.append({"id": "matrix", "keys": keys, "p": False})
        kinds = [k for k, there in (("co_occurrence", cooc), ("ripley", ripley or "ripley" in roles)) if there]
        if kinds:
            facets.append({"id": "curves", "kinds": kinds})
        elif "cooc_mean" in roles:
            facets.append({"id": "curves", "kinds": ["co_occurrence"], "needs": ["h5ad"]})
            warnings.append("The co-occurrence curves over distance are in the h5ad only (uns['<key>_co_occurrence']).")
        ranked = [r for r in ("centrality", "moran", "geary") if r in roles]
        if ranked:
            facets.append({"id": "ranking", "tables": [{"id": r, "label": _TABLE_LABELS[r]} for r in ranked]})
        if {"moran", "geary"} & set(roles) or (h5 is not None and {"moranI", "gearyC"} & set(_uns_names(h5))):
            warnings.append(_PER_GENE)
        mode = per_section_fields(sections_of_tables(sources, ("ripley", "cooc_mean"), limits))
        graph = geo.graph_mode(h5, MAX_NNZ) if h5 is not None else None
        if graph is None and h5 is not None and not mode["sections"]:
            # No graph (co-occurrence, Ripley): the worker's own record of its mode and frame (R2-3).
            graph = geo.recorded_mode(h5)
        if graph is None and ripley and not mode["sections"]:
            # A per-section Ripley run keeps its statistic concatenated with a ``section`` column, and no graph.
            frame = h5.uns_frame(ripley.group(0), f"{ripley.group('mode')}_stat", max_rows=_UNS_MAX_VALUES) or {}
            if "section" in frame:
                mode = per_section_fields(list(dict.fromkeys(tables.labels(frame["section"]))))
        if graph is not None:
            mode = {
                "mode": graph["mode"],
                "sections": graph["sections"],
                "frame": graph["frame"],
                "legacy_rank_z": False,
            }
            for name in ("cross_section_edge_fraction", "n_cross_section_edges"):
                if name in graph:
                    mode[name] = graph[name]
            if graph["mode"] == "per-section-2d" and (nhood or {"zscore", "count"} & set(roles)):
                warnings.append(_pooled(len(graph["sections"])))
                # The panel's section scrub would ask every stop of this one matrix and be refused (R2-1).
                mode["section_filterable"] = False
            if graph["mode"] == "3d":
                fraction = graph.get("cross_section_edge_fraction")
                said = "" if fraction is None else f" {fraction:.4%} of its edges join two sections."
                warnings.append(
                    "This graph was built in the aligned 3D frame; an edge between sections is inferred "
                    "cross-section communication, not a measured one." + said
                )
        return {
            "label": "squidpy spatial statistics",
            "n_obs": h5.n_obs if h5 is not None else None,
            "unit": "spot" if h5 is not None else "group",
            "facets": facets,
            "groups": h5x.obs_groups(h5, int(limits.get("max_levels", 4096))) if h5 is not None else [],
            "warnings": warnings,
            **mode,
        }

    def data(
        self,
        sources: list[Source],
        facet: str,
        params: Mapping[str, Any],
        group: dict | None,
        rows: np.ndarray | None,
        limits: Mapping[str, int],
    ) -> dict:
        warnings = (
            ["squidpy's statistics are over every spot: a brushed subset does not change them."]
            if rows is not None
            else []
        )
        if facet == "matrix":
            answer = self._matrix(sources, params, limits)
        elif facet == "curves":
            answer = self._curves(sources, params, limits)
        elif facet == "ranking":
            answer = self._ranking(sources, params, limits)
        else:
            raise not_offered(facet, "squidpy gives matrix, curves and ranking.")
        answer["warnings"] = warnings + answer["warnings"]
        return answer

    def _matrix(self, sources: list[Source], params: Mapping[str, Any], limits: Mapping[str, int]) -> dict:
        roles = by_role(sources)
        h5 = h5ad_of(sources)
        nhood = _first(h5, _NHOOD_RE)
        offered = (["nhood_enrichment"] if nhood or {"zscore", "count"} & set(roles) else []) + (
            ["co_occurrence_mean"] if "cooc_mean" in roles else []
        )
        if not offered:
            raise not_offered("matrix", "this squidpy result has no groups-by-groups table.")
        key = choice(params, "key", offered, offered[0])
        ck = nhood.group("ck") if nhood else None
        labels = n = group_id = None
        z = None
        section, said = None, []
        if key == "nhood_enrichment":
            if nhood:
                z = h5.uns_array(nhood.group(0), "zscore", max_values=_UNS_MAX_VALUES)
                count = h5.uns_array(nhood.group(0), "count", max_values=_UNS_MAX_VALUES)
                labels, n, group_id = _obs_labels(h5, ck)
            else:
                count = None
            # A per-section run's long tables answer the section asked for (the first, said, when none is), as the
            # co-occurrence table does; the request's section was ignored here before.
            if z is None and "zscore" in roles:
                labels, z, section, said = _square_csv(roles["zscore"], limits, params)
            if count is None and "count" in roles:
                labels, count, section, counted = _square_csv(roles["count"], limits, params)
                said = said or counted
            values, stat = (count, "count") if count is not None else (z, "zscore")
            label = "Neighbourhood enrichment"
        else:
            labels, values, section, said = _square_csv(roles["cooc_mean"], limits, params)
            stat, label = "co_occurrence_ratio", "Co-occurrence ratio, mean over distances"
        if values is None or labels is None or np.asarray(values).shape != (len(labels), len(labels)):
            raise CCCRefusal("unreadable", "squidpy's matrix and its groups do not have the same number of groups.")
        described = (
            {"id": group_id, "label": ck, "source": "h5ad"}
            if group_id
            else {"id": None, "label": "the groups squidpy compared", "source": "table"}
        )
        payload = matrix_payload(
            key=key,
            key_label=label,
            stat=stat,
            group=described,
            labels=labels,
            values=np.asarray(values, float),
            n=n,
            z=None if z is None else np.asarray(z, float),
        )
        if section is not None:
            payload.update({"section": section, "mode": "per-section-2d"})
        elif params.get("section") is not None and key == "nhood_enrichment":
            h5 = h5ad_of(sources)
            graph = geo.graph_mode(h5, MAX_NNZ) if h5 is not None else None
            if graph is not None and graph["mode"] == "per-section-2d":
                raise CCCRefusal("bad_request", _pooled(len(graph["sections"])))
        return {
            "facet": "matrix",
            "payload": payload,
            "warnings": said + ([_NO_P] if key == "nhood_enrichment" else []),
        }

    def _curves(self, sources: list[Source], params: Mapping[str, Any], limits: Mapping[str, int]) -> dict:
        roles = by_role(sources)
        h5 = h5ad_of(sources)
        cooc, ripley = _first(h5, _COOC_RE), _first(h5, _RIPLEY_RE)
        offered = [k for k, there in (("co_occurrence", cooc), ("ripley", ripley or "ripley" in roles)) if there]
        if not offered:
            why = (
                "the co-occurrence curves are in the h5ad only." if "cooc_mean" in roles else "no curves were written."
            )
            raise not_offered("curves", why)
        kind = choice(params, "kind", offered, offered[0])
        if kind == "co_occurrence":
            return self._co_occurrence(h5, cooc, params)
        return self._ripley(roles, h5, params, limits)

    def _co_occurrence(self, h5: H5AD, cooc: re.Match, params: Mapping[str, Any]) -> dict:
        occ = h5.uns_array(cooc.group(0), "occ", max_values=_UNS_MAX_VALUES)
        interval = h5.uns_array(cooc.group(0), "interval", max_values=_UNS_MAX_VALUES)
        if occ is None or interval is None or occ.ndim != 3 or occ.shape[0] != occ.shape[1]:
            raise CCCRefusal("unreadable", "squidpy's co-occurrence in this file is not groups x groups x distances.")
        k, bins = occ.shape[0], occ.shape[2]
        x = interval[1:] if interval.size == bins + 1 else interval[:bins]
        labels, _, _ = _obs_labels(h5, cooc.group("ck"))
        if labels is None or len(labels) != k:
            labels = [str(i) for i in range(k)]
        small_x, small_occ = reduce_bins(x, occ)
        firsts = range(k)
        if "a" in params:
            if int(params["a"]) >= k:
                raise CCCRefusal("bad_request", f"The request's group a is past the {k} groups of this result.")
            firsts = [int(params["a"])]
        series = [{"a": labels[a], "b": labels[b], "y": small_occ[a, b].tolist()} for a in firsts for b in range(k)]
        said = f" The {bins} distance bins were averaged down to {len(small_x)}." if len(small_x) < bins else ""
        payload = {
            "facet": "curves",
            "kind": "co_occurrence",
            "labels": labels,
            "x": small_x.tolist(),
            "x_label": "distance",
            "series": series,
            "pvalues": None,
            "mode": None,
            "note_text": "Each curve is p(b near a) / p(b): above 1, group b is found within that distance of group a "
            "more often than its share of the tissue." + said,
        }
        warnings = [said.strip()] if said else []
        return {"facet": "curves", "payload": payload, "warnings": warnings + fit(payload, [_halve_series])}

    def _ripley(
        self, roles: dict[str, Source], h5: H5AD | None, params: Mapping[str, Any], limits: Mapping[str, int]
    ) -> dict:
        ripley = _first(h5, _RIPLEY_RE)
        section, sections_said = None, []
        if "ripley" in roles:
            frame, section, sections_said = one_section(
                tables.read(roles["ripley"].fd, roles["ripley"].sep or ",", limits), params, required=True
            )
            names = [str(c) for c in frame.columns]
            if "bins" not in names or "stats" not in names or len(names) != 3:
                raise CCCRefusal("unsupported", "squidpy's Ripley table is not (bins, <group>, stats).")
            ck = next(c for c in names if c not in ("bins", "stats"))
            bins, groups, stats = (
                tables.numbers(frame["bins"]),
                tables.labels(frame[ck]),
                tables.numbers(frame["stats"]),
            )
        elif ripley:
            frame = h5.uns_frame(ripley.group(0), f"{ripley.group('mode')}_stat", max_rows=_UNS_MAX_VALUES)
            if not frame or "bins" not in frame or "stats" not in frame:
                raise CCCRefusal("unreadable", "squidpy's Ripley statistic in this file has no bins and stats.")
            if "section" in frame:  # a per-section run's statistic, concatenated with its section
                import pandas as pd

                table, section, sections_said = one_section(
                    pd.DataFrame({"section": frame["section"], **{k: v for k, v in frame.items() if k != "section"}}),
                    params,
                    required=True,
                )
                frame = {str(c): table[c].to_numpy() for c in table.columns}
            ck = next(c for c in frame if c not in ("bins", "stats"))
            bins, groups, stats = frame["bins"], tables.labels(frame[ck]), frame["stats"]
        else:  # pragma: no cover - _curves offers ripley only when one is there
            raise not_offered("curves", "no Ripley statistic was written.")
        labels = list(dict.fromkeys(groups))
        by_label = {g: np.flatnonzero(np.asarray(groups, dtype=object) == g) for g in labels}
        lengths = {len(i) for i in by_label.values()}
        if len(lengths) != 1:
            raise CCCRefusal("unreadable", "squidpy's Ripley curves do not all have the same distances.")
        x = np.asarray(bins, float)[by_label[labels[0]]]
        y = np.vstack([np.asarray(stats, float)[by_label[g]] for g in labels])
        mode = None
        pvalues = None
        found = _first(h5, re.compile(rf"^{re.escape(ck)}_ripley_(?P<mode>[FGL])$"))
        if found is None and "ripley" in roles:
            mode = roles["ripley"].fields.get("mode")  # squidpy_ripley_<mode>.csv: detect read it from the name
        if found is not None:
            mode = found.group("mode")
            raw = h5.uns_array(found.group(0), "pvalues", max_values=_UNS_MAX_VALUES)
            obs_labels, _, _ = _obs_labels(h5, ck)
            if raw is not None and raw.ndim == 2 and raw.shape[1] == x.size and raw.shape[0] == len(labels):
                if obs_labels is not None and sorted(obs_labels) == sorted(labels):
                    raw = raw[[obs_labels.index(g) for g in labels]]
                pvalues = raw
        small_x, small_y = reduce_bins(x, y)
        if pvalues is not None:
            _, pvalues = reduce_bins(x, pvalues)
        said = f" The {x.size} distance bins were averaged down to {len(small_x)}." if len(small_x) < x.size else ""
        payload = {
            "facet": "curves",
            "kind": "ripley",
            "labels": labels,
            "x": small_x.tolist(),
            "x_label": "distance",
            "series": [{"a": g, "b": None, "y": small_y[i].tolist()} for i, g in enumerate(labels)],
            "pvalues": None if pvalues is None else np.asarray(pvalues).tolist(),
            "mode": mode,
            "note_text": "Ripley's statistic per group over distance; "
            + (
                "p-values are against random placement."
                if pvalues is not None
                else "its p-values are in the h5ad only."
            )
            + said,
        }
        if section is not None:
            # ``mode`` is Ripley's statistic (F, G or L) in a curves answer; the run's mode rides as ``per_section``.
            payload.update({"section": section, "per_section": True})
        return {"facet": "curves", "payload": payload, "warnings": sections_said + ([said.strip()] if said else [])}

    def _ranking(self, sources: list[Source], params: Mapping[str, Any], limits: Mapping[str, int]) -> dict:
        roles = by_role(sources)
        offered = [r for r in ("centrality", "moran", "geary") if r in roles]
        if not offered:
            raise not_offered("ranking", "no centrality or autocorrelation table was written.")
        table = choice(params, "table", offered, offered[0])
        frame = tables.read(roles[table].fd, roles[table].sep or ",", limits)
        first = str(frame.columns[0])
        if first.startswith("Unnamed: 0") or first == "":
            frame = frame.rename(columns={frame.columns[0]: "group" if table == "centrality" else "gene"})
        numeric = [c for c in frame.columns[1:] if np.isfinite(tables.numbers(frame[c])).any()]
        sort_by = str(numeric[0]) if numeric else None
        desc = table != "geary"  # a low Geary's C is the strong autocorrelation
        columns, rows = frame_rows(frame, sort_by, desc)
        payload = ranking_payload(
            table=table,
            label=_TABLE_LABELS[table],
            columns=columns,
            rows=rows,
            n_total=len(frame),
            sort_by=sort_by,
            desc=desc,
        )
        warnings = [_PER_GENE] if table in ("moran", "geary") else []
        return {"facet": "ranking", "payload": payload, "warnings": warnings + fit(payload, [halve_rows])}


def _halve_series(payload: dict) -> bool:
    """kept the curves of the first half of the groups"""
    firsts = list(dict.fromkeys(s["a"] for s in payload["series"]))
    if len(firsts) < 2:
        return False
    kept = set(firsts[: len(firsts) // 2])
    payload["series"] = [s for s in payload["series"] if s["a"] in kept]
    return True


ADAPTER = SquidpyAdapter()
