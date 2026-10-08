"""DeepLinc: an interaction score per pair of cell types (observed over expected), and its p-values.

``deeplinc_interaction_scores.csv`` and ``deeplinc_pvalues.csv`` are types-by-types matrices whose row labels are
their column labels; that is what confirms them (a square frame whose index equals its header). The significant
table is often empty (one byte on the smoke run) and adds nothing the two matrices do not hold; the adjacency is the
spot graph DeepLinc reconstructed, not a cell-type result. There is no per-spot field or direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from spatialomicsgym.viz.ccc import tables
from spatialomicsgym.viz.ccc.errors import CCCRefusal

from .base import Source, by_role, matrix_payload, not_offered

if TYPE_CHECKING:
    from collections.abc import Mapping

_NO_FIELD = "DeepLinc scores pairs of cell types: there is no per-spot field or direction to draw."


def _matrix(source: Source, limits: Mapping[str, int], what: str) -> tuple[list[str], np.ndarray]:
    rows, cols, values = tables.square(tables.read(source.fd, source.sep or ",", limits))
    if rows != cols:
        raise CCCRefusal(
            "unsupported", f"DeepLinc's {what} table is not a cell-type matrix: its row labels are not its columns."
        )
    return rows, values


class DeeplincAdapter:
    backend = "deeplinc"

    def describe(self, sources: list[Source], limits: Mapping[str, int]) -> dict:
        roles = by_role(sources)
        if "scores" not in roles:
            raise CCCRefusal("unsupported", "This DeepLinc result has no interaction-score table.")
        labels, _ = _matrix(roles["scores"], limits, "interaction-score")
        return {
            "label": "DeepLinc cell-type interactions",
            "n_obs": None,
            "unit": "cell type",
            "facets": [
                {
                    "id": "matrix",
                    "keys": [{"key": "interaction_score", "label": "Interaction score (observed / expected)"}],
                    "p": "pvalues" in roles,
                    "n_groups": len(labels),
                }
            ],
            "groups": [],
            "warnings": [_NO_FIELD],
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
        if facet != "matrix":
            raise not_offered(facet, _NO_FIELD)
        roles = by_role(sources)
        labels, scores = _matrix(roles["scores"], limits, "interaction-score")
        p = None
        warnings = []
        if "pvalues" in roles:
            p_labels, pvalues = _matrix(roles["pvalues"], limits, "p-value")
            if sorted(p_labels) == sorted(labels):
                order = [p_labels.index(name) for name in labels]
                p = pvalues[np.ix_(order, order)]
            else:
                warnings.append("DeepLinc's p-value table names other cell types than its scores, so it is left out.")
        if rows is not None:
            warnings.append("DeepLinc's scores are over every cell: a brushed subset does not change them.")
        payload = matrix_payload(
            key="interaction_score",
            key_label="Interaction score (observed / expected)",
            stat="interaction_score",
            group={"id": None, "label": "cell types", "source": "table"},
            labels=labels,
            values=scores,
            p=p,
        )
        return {"facet": "matrix", "payload": payload, "warnings": warnings}


ADAPTER = DeeplincAdapter()
