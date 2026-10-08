"""NCEM: how a gene's expression in one cell type depends on its neighbours' types -- Ridge coefficients, gene by
neighbour type -- and a mean effect strength per type.

``ncem_communication_matrix.csv`` is genes by neighbour types (an unnamed index of genes), so it is a heatmap of the
genes whose coefficient is largest in size, not a type-by-type matrix: NCEM gives no communication matrix and no
direction. ``ncem_communication_strength.csv`` (``cell_type``, ``mean_effect_strength``) is a bar ranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from spatialomicsgym.viz.ccc import detect, tables
from spatialomicsgym.viz.ccc.errors import CCCRefusal
from spatialomicsgym.viz.ccc.facets import MAX_RANK_ROWS, fit

from .base import (
    Source,
    by_role,
    choice,
    halve_rows,
    has_section,
    not_offered,
    one_section,
    per_section_fields,
    ranking_payload,
    sections_of_tables,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_NOT_TYPES = (
    "NCEM's coefficients are gene x neighbour type, not type x type: there is no communication matrix or direction."
)
_TABLES = {"gene_by_type": "Genes by neighbour type (Ridge coefficients)", "strength": "Mean effect strength per type"}


class NcemAdapter:
    backend = "ncem"

    def _offered(self, sources: list[Source], limits: Mapping[str, int]) -> list[str]:
        roles = by_role(sources)
        offered = []
        if "matrix" in roles:
            header = tables.header(roles["matrix"].fd, roles["matrix"].sep or ",", limits)
            if (detect.table_columns(header) or {}).get("shape") != "square":
                raise CCCRefusal("unsupported", "NCEM's communication matrix is not genes by neighbour types.")
            offered.append("gene_by_type")
        if "strength" in roles:
            header = tables.header(roles["strength"].fd, roles["strength"].sep or ",", limits)
            if has_section(header):
                header = header[1:]  # a per-section run's long table: one value per type within each section
            if (detect.table_columns(header) or {}).get("shape") != "vector":
                raise CCCRefusal("unsupported", "NCEM's strength table is not one value per cell type.")
            offered.append("strength")
        return offered

    def describe(self, sources: list[Source], limits: Mapping[str, int]) -> dict:
        offered = self._offered(sources, limits)
        facets = [{"id": "ranking", "tables": [{"id": t, "label": _TABLES[t]} for t in offered]}] if offered else []
        return {
            "label": "NCEM niche effects",
            "n_obs": None,
            "unit": "gene",
            "facets": facets,
            "groups": [],
            "warnings": [_NOT_TYPES],
            **per_section_fields(sections_of_tables(sources, ("strength",), limits)),
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
            raise not_offered(facet, _NOT_TYPES)
        table = choice(params, "table", offered, offered[0])
        source = by_role(sources)["matrix" if table == "gene_by_type" else "strength"]
        frame, section, warnings = one_section(tables.read(source.fd, source.sep or ",", limits), params, required=True)
        if table == "gene_by_type":
            genes, types, values = tables.square(frame)
            peak = np.nanmax(np.abs(np.nan_to_num(values, nan=0.0)), axis=1) if values.size else np.zeros(0)
            order = np.argsort(-peak, kind="stable")[:MAX_RANK_ROWS]
            kept = values[order]
            names = [genes[i] for i in order.tolist()]
            payload = ranking_payload(
                table=table,
                label=_TABLES[table],
                columns=[("gene", "text"), *((t, "number") for t in types)],
                rows=[[g, *row] for g, row in zip(names, kept.tolist(), strict=True)],
                n_total=len(genes),
                sort_by="max |coefficient|",
                desc=True,
                heatmap=(names, types, kept),
            )
        else:
            label, score = str(frame.columns[0]), str(frame.columns[1])
            values = tables.numbers(frame[score])
            order = np.argsort(-np.nan_to_num(values, nan=-np.inf), kind="stable")
            names = tables.labels(frame[label])
            payload = ranking_payload(
                table=table,
                label=_TABLES[table],
                columns=[(label, "text"), (score, "number")],
                rows=[[names[i], float(values[i])] for i in order.tolist()],
                n_total=len(frame),
                sort_by=score,
                desc=True,
            )
        if section is not None:
            payload.update({"section": section, "mode": "per-section-2d"})
        return {"facet": "ranking", "payload": payload, "warnings": warnings + fit(payload, [halve_rows])}


ADAPTER = NcemAdapter()
