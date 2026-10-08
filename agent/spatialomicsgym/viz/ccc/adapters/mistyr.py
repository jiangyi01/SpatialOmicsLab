"""MISTy (mistyR): how well each marker's expression is predicted from markers in each spatial view, and which
predictors matter -- predictor by target, never cell type by cell type, so rankings only.

Its tables carry a ``sample`` column holding the run's results FOLDER -- an absolute path -- so that column is never
parsed into an answer (the reader is never told a path, and must never send one).

``mistyr_importances.csv`` is long (view, Predictor, Target, Importance) and big: 500,000 rows, 60 MB, on the smoke
run. It is read in blocks under the table caps, each block folded into a running top :data:`facets.MAX_RANK_ROWS` per
view, so no more than a block and the running tops are ever held.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from spatialomicsgym.viz.ccc import detect, tables
from spatialomicsgym.viz.ccc.errors import CCCRefusal
from spatialomicsgym.viz.ccc.facets import MAX_RANK_ROWS, fit

from .base import (
    SECTION_COLUMN,
    Source,
    by_role,
    choice,
    frame_rows,
    halve_rows,
    not_offered,
    one_section,
    per_section_fields,
    ranking_payload,
    sections_of_tables,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_NO_MATRIX = "MISTy relates predictor markers to target markers, not cell types to cell types: there is no matrix."
_PATH_COLUMN = "sample"
_TABLES = {
    "top": ("top_interactions", "Strongest predictor-target importances"),
    "importances": ("importances_by_view", "Importances in one view"),
    "performance": ("performance", "Prediction performance per target"),
}


def _columns(source: Source, limits: Mapping[str, int]) -> list[str]:
    """The table's columns, the path-holding ``sample`` left out."""
    return [c for c in tables.header(source.fd, source.sep or ",", limits) if c != _PATH_COLUMN]


def _long(columns: list[str]) -> dict:
    found = detect.table_columns(columns)
    if found is None or found["shape"] != "long" or "view" not in columns:
        raise CCCRefusal("unsupported", "A MISTy importance table has no view, Predictor, Target and Importance.")
    return found


class MistyrAdapter:
    backend = "mistyr"

    def _offered(self, sources: list[Source], limits: Mapping[str, int]) -> list[str]:
        roles = by_role(sources)
        offered = []
        for role in ("top", "importances", "performance"):
            if role not in roles:
                continue
            columns = _columns(roles[role], limits)
            if role == "performance" and not {"target", "measure", "value"} <= set(columns):
                raise CCCRefusal("unsupported", "MISTy's performance table has no target, measure and value.")
            if role != "performance":
                _long(columns)
            offered.append(_TABLES[role][0])
        return offered

    def describe(self, sources: list[Source], limits: Mapping[str, int]) -> dict:
        offered = self._offered(sources, limits)
        labels = dict(_TABLES.values())
        return {
            "label": "MISTy marker interactions",
            "n_obs": None,
            "unit": "marker",
            "facets": [{"id": "ranking", "tables": [{"id": t, "label": labels[t]} for t in offered]}]
            if offered
            else [],
            "groups": [],
            "warnings": [_NO_MATRIX],
            **per_section_fields(sections_of_tables(sources, ("top", "importances", "performance"), limits)),
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
        offered = self._offered(sources, limits)
        if facet != "ranking" or not offered:
            raise not_offered(facet, _NO_MATRIX)
        table = choice(params, "table", offered, offered[0])
        roles = by_role(sources)
        if table == "top_interactions":
            source = roles["top"]
            columns = _columns(source, limits)
            score = _long(columns)["score"]
            frame = tables.read(source.fd, source.sep or ",", limits, usecols=columns)
            frame, section, warnings = one_section(frame, params, required=False)
            names, out_rows = frame_rows(frame, score, True)
            payload = ranking_payload(
                table=table,
                label=_TABLES["top"][1],
                columns=names,
                rows=out_rows,
                n_total=len(frame),
                sort_by=score,
                desc=True,
            )
        elif table == "importances_by_view":
            payload, warnings, section = self._by_view(roles["importances"], params, limits)
        else:
            payload, warnings, section = self._performance(roles["performance"], params, limits)
        if section is not None:
            payload.update({"section": section, "mode": "per-section-2d"})
        return {"facet": "ranking", "payload": payload, "warnings": warnings + fit(payload, [halve_rows])}

    def _by_view(
        self, source: Source, params: Mapping[str, Any], limits: Mapping[str, int]
    ) -> tuple[dict, list, str | None]:
        import pandas as pd

        columns = _columns(source, limits)
        found = _long(columns)
        sectioned = bool(columns) and columns[0] == SECTION_COLUMN
        asked = params.get("section")
        if asked is not None and not sectioned:
            raise CCCRefusal("bad_request", "This result names no sections, so no section can be asked for.")
        keep = ([SECTION_COLUMN] if sectioned and asked is None else []) + [
            "view",
            found["source"],
            found["target"],
            found["score"],
        ]
        read = keep if not sectioned or asked is None else [SECTION_COLUMN, *keep]
        score = found["score"]
        tops: dict[str, pd.DataFrame] = {}
        counts: dict[str, int] = {}
        seen_sections: set[str] = set()
        for block in tables.blocks(source.fd, source.sep or ",", limits, usecols=read):
            if sectioned and asked is not None:
                here = tables.labels(block[SECTION_COLUMN])
                seen_sections.update(here)
                block = block[[v == asked for v in here]]
            block = block[keep]
            block[score] = tables.numbers(block[score])
            for view, part in block.groupby("view", sort=False):
                view = str(view)
                counts[view] = counts.get(view, 0) + len(part)
                joined = part if view not in tops else pd.concat([tops[view], part], ignore_index=True)
                tops[view] = joined.nlargest(MAX_RANK_ROWS, score, keep="first")
        views = list(counts)
        if sectioned and asked is not None and asked not in seen_sections:
            known = ", ".join(sorted(seen_sections)[:12])
            raise CCCRefusal("bad_request", f"The request's section is not one of this result's ({known}).")
        if not views:
            raise CCCRefusal("unsupported", "MISTy's importance table has no rows.")
        view = choice(params, "view", views, views[0])
        names, out_rows = frame_rows(tops[view], score, True)
        payload = ranking_payload(
            table="importances_by_view",
            label=f"Importances in the {view} view",
            columns=names,
            rows=out_rows,
            n_total=counts[view],
            sort_by=score,
            desc=True,
        )
        payload["views"] = views
        said = [f"The {MAX_RANK_ROWS} strongest of the {counts[view]:,} importances in the {view} view."]
        return payload, said, asked

    def _performance(
        self, source: Source, params: Mapping[str, Any], limits: Mapping[str, int]
    ) -> tuple[dict, list, str | None]:
        columns = _columns(source, limits)
        wanted = ([SECTION_COLUMN] if columns and columns[0] == SECTION_COLUMN else []) + ["target", "measure", "value"]
        frame = tables.read(source.fd, source.sep or ",", limits, usecols=wanted)
        frame = frame[wanted]
        frame, section, warnings = one_section(frame, params, required=True)
        targets = list(dict.fromkeys(tables.labels(frame["target"])))
        measures = list(dict.fromkeys(tables.labels(frame["measure"])))
        where_t = {t: i for i, t in enumerate(targets)}
        where_m = {m: j for j, m in enumerate(measures)}
        values = np.full((len(targets), len(measures)), np.nan)
        for t, m, v in zip(
            tables.labels(frame["target"]),
            tables.labels(frame["measure"]),
            tables.numbers(frame["value"]).tolist(),
            strict=True,
        ):
            values[where_t[t], where_m[m]] = v
        kept = values[:MAX_RANK_ROWS]
        payload = ranking_payload(
            table="performance",
            label=_TABLES["performance"][1],
            columns=[("target", "text"), *((m, "number") for m in measures)],
            rows=[[t, *row] for t, row in zip(targets[:MAX_RANK_ROWS], kept.tolist(), strict=True)],
            n_total=len(targets),
            sort_by=None,
            desc=False,
            heatmap=(targets[:MAX_RANK_ROWS], measures, kept),
        )
        return payload, warnings, section


ADAPTER = MistyrAdapter()
