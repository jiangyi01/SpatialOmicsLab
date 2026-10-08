"""SpaCET: cell-type colocalisation -- a correlation per pair of cell types, against two references -- and the
deconvolved proportions behind it.

``spacet_cci_colocalization.csv`` (R ``write.csv``, no row names): ``cell_type_1``, ``cell_type_2``,
``fraction_product``, ``fraction_rho``, ``fraction_pv``, ``reference_rho``, ``reference_pv``; one row per unordered
pair, mirrored into both halves of the matrix. ``spacet_proportions.csv`` is cell types by SPOTS (a column per spot),
which no colour table of the explorer reads, so it gives no field in this version.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spatialomicsgym.viz.ccc import detect, tables
from spatialomicsgym.viz.ccc.errors import CCCRefusal

from .base import Source, by_role, choice, matrix_payload, not_offered

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

_STATS = ("fraction_rho", "reference_rho")
_PROPORTIONS = (
    "spacet_proportions.csv holds cell types by spots (a column per spot), so it gives no field in this version."
)
_NO_CCI = "SpaCET's colocalisation table (spacet_cci_colocalization.csv) was not written, so there is no matrix."


class SpacetAdapter:
    backend = "spacet"

    def _cci(self, source: Source, limits: Mapping[str, int]) -> tuple[Any, dict, list[str]]:
        frame = tables.read(source.fd, source.sep or ",", limits)
        columns = detect.table_columns([str(c) for c in frame.columns])
        stats = [s for s in _STATS if s in frame.columns]
        if columns is None or columns["shape"] != "long" or not stats:
            raise CCCRefusal("unsupported", "SpaCET's colocalisation table has no cell_type_1, cell_type_2 and rho.")
        return frame, columns, stats

    def describe(self, sources: list[Source], limits: Mapping[str, int]) -> dict:
        roles = by_role(sources)
        facets: list[dict] = []
        warnings: list[str] = []
        if "cci" in roles:
            _, _, stats = self._cci(roles["cci"], limits)
            facets.append(
                {"id": "matrix", "keys": [{"key": s, "label": s.replace("_", " ")} for s in stats], "p": True}
            )
        else:
            warnings.append(_NO_CCI)
        if "proportions" in roles:
            warnings.append(_PROPORTIONS)
        return {
            "label": "SpaCET cell-type colocalisation",
            "n_obs": None,
            "unit": "cell type",
            "facets": facets,
            "groups": [],
            "warnings": warnings,
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
        if facet != "matrix" or "cci" not in roles:
            raise not_offered(facet, _NO_CCI if "cci" not in roles else "SpaCET gives a cell-type matrix only.")
        frame, columns, stats = self._cci(roles["cci"], limits)
        stat = choice(params, "stat", stats, stats[0])
        source, target = columns["source"], columns["target"]
        labels, values = tables.long_matrix(frame, source, target, stat, mirror=True)
        p_column = stat.replace("_rho", "_pv")
        p = (
            tables.long_matrix(frame, source, target, p_column, mirror=True, order=labels)[1]
            if p_column in frame
            else None
        )
        payload = matrix_payload(
            key=stat,
            key_label=f"Colocalisation ({stat.replace('_', ' ')})",
            stat=stat,
            group={"id": None, "label": "cell types", "source": "table"},
            labels=labels,
            values=values,
            p=p,
        )
        warnings = [_PROPORTIONS] if "proportions" in roles else []
        if rows is not None:
            # As DeepLinc and squidpy say it: a table over every spot is not changed by a brush, and silence read as if
            # the matrix were the selection's.
            warnings.append("SpaCET's colocalisation is over every spot: a brushed subset does not change it.")
        return {"facet": "matrix", "payload": payload, "warnings": warnings}


ADAPTER = SpacetAdapter()
