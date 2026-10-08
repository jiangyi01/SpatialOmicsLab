"""Neighbor-seq: which cell types sit next to each other, inferred from expression, as enrichment per pair.

``neighborseq_interactions.csv`` is R's ``write.csv`` of the result: an unnamed row-name column, then ``sample``,
``Cell_1``, ``Cell_2``, ``Counts``, ``EnrichmentScore``, ``pval``, ``padj`` -- one row per unordered pair of types per
sample, so each pair is mirrored into both halves of the matrix. ``neighborseq_top_interactions.csv`` is the same
columns cut to the strongest rows. Neighbor-seq has no spot-level output at all: no field, no direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from spatialomicsgym.viz.ccc import detect, tables
from spatialomicsgym.viz.ccc.errors import CCCRefusal
from spatialomicsgym.viz.ccc.facets import fit

from .base import Source, by_role, choice, frame_rows, halve_rows, matrix_payload, not_offered, ranking_payload

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

_NO_SPOTS = "Neighbor-seq infers which cell types neighbour each other from expression: it has no spot-level data."
_P_COLUMNS = ("pval", "padj")


def _interactions(source: Source, limits: Mapping[str, int]) -> tuple[pd.DataFrame, dict]:
    frame = tables.read(source.fd, source.sep or ",", limits)
    columns = detect.table_columns([str(c) for c in frame.columns])
    if columns is None or columns["shape"] != "long":
        raise CCCRefusal("unsupported", "Neighbor-seq's interaction table has no Cell_1, Cell_2 and score columns.")
    return frame, columns


def _samples(frame: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(tables.labels(frame["sample"]))) if "sample" in frame.columns else []


class NeighborseqAdapter:
    backend = "neighborseq"

    def describe(self, sources: list[Source], limits: Mapping[str, int]) -> dict:
        roles = by_role(sources)
        facets: list[dict] = []
        if "interactions" in roles:
            frame, columns = _interactions(roles["interactions"], limits)
            facets.append(
                {
                    "id": "matrix",
                    "keys": [{"key": columns["score"], "label": "Enrichment score"}],
                    "samples": _samples(frame),
                    "p": [c for c in _P_COLUMNS if c in frame.columns],
                }
            )
        if "top" in roles:
            facets.append({"id": "ranking", "tables": [{"id": "top", "label": "Strongest neighbouring pairs"}]})
        if not facets:
            raise CCCRefusal("unsupported", "This Neighbor-seq result has no interaction table.")
        return {
            "label": "Neighbor-seq cell-type neighbourhoods",
            "n_obs": None,
            "unit": "cell type",
            "facets": facets,
            "groups": [],
            "warnings": [_NO_SPOTS],
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
        roles = by_role(sources)
        # Neighbor-seq's tables are over every spot: a brush changes nothing, and says so as DeepLinc and squidpy do.
        brushed = (
            ["Neighbor-seq's tables are over every spot: a brushed subset does not change them."]
            if rows is not None
            else []
        )
        if facet == "ranking" and "top" in roles:
            frame = tables.read(roles["top"].fd, roles["top"].sep or ",", limits)
            choice(params, "table", ["top"], "top")
            columns, out_rows = frame_rows(frame, None, False)
            payload = ranking_payload(
                table="top",
                label="Strongest neighbouring pairs",
                columns=columns,
                rows=out_rows,
                n_total=len(frame),
                sort_by=None,
                desc=False,
            )
            return {"facet": "ranking", "payload": payload, "warnings": brushed + fit(payload, [halve_rows])}
        if facet != "matrix" or "interactions" not in roles:
            raise not_offered(facet, _NO_SPOTS)
        frame, columns = _interactions(roles["interactions"], limits)
        samples = _samples(frame)
        order = _appearance(frame, columns)
        warnings: list[str] = list(brushed)
        if samples:
            sample = choice(params, "sample", samples, samples[0])
            frame = frame[np.asarray(tables.labels(frame["sample"]), dtype=object) == sample]
            if len(samples) > 1:
                warnings.append(f"Sample {sample} of {len(samples)} is shown.")
        offered_p = [c for c in _P_COLUMNS if c in frame.columns]
        p = None
        if offered_p:
            p_column = choice(params, "p", offered_p, offered_p[0])
            _, p = tables.long_matrix(frame, columns["source"], columns["target"], p_column, mirror=True, order=order)
        labels, values = tables.long_matrix(
            frame, columns["source"], columns["target"], columns["score"], mirror=True, order=order
        )
        payload = matrix_payload(
            key=columns["score"],
            key_label="Enrichment score",
            stat=columns["score"],
            group={"id": None, "label": "cell types", "source": "table"},
            labels=labels,
            values=values,
            p=p,
        )
        return {"facet": "matrix", "payload": payload, "warnings": warnings}


def _appearance(frame: pd.DataFrame, columns: dict) -> list[str]:
    """Every cell type of the whole table, in order of first appearance -- so every sample shares one set of axes."""
    seen: dict[str, None] = {}
    for a, b in zip(tables.labels(frame[columns["source"]]), tables.labels(frame[columns["target"]]), strict=True):
        seen.setdefault(a, None)
        seen.setdefault(b, None)
    return list(seen)


ADAPTER = NeighborseqAdapter()
