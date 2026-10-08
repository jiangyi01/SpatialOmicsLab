"""SpaOTsc: a cell-by-cell signalling score from optimal transport, grouped by the clusters SpaOTsc wrote.

``signaling_scores.csv`` is a bare n x n matrix (header ``0,1,...,n-1``, no row labels; sender rows, receiver
columns); ``labels.csv`` (``cell``, ``cluster``, ``expression_cluster``) names each row's groups, row for row -- an
empty ``cluster`` is a cell SpaOTsc kept in no spatial subcluster, and takes no group. There are no cell ids to join
to a dataset, so there is no field or direction; without ``labels.csv`` there is nothing to group by at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from spatialomicsgym.viz.ccc import detect, tables
from spatialomicsgym.viz.ccc import groups as grp
from spatialomicsgym.viz.ccc import h5 as h5x
from spatialomicsgym.viz.ccc.errors import CCCRefusal
from spatialomicsgym.viz.ccc.facets import DEFAULT_PERMUTATIONS, MAX_PERMUTATIONS, PERMUTE_BUDGET

from .base import Source, by_role, matrix_payload, not_offered

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

_NO_LABELS = "SpaOTsc's labels.csv is not here, so the cells cannot be grouped: nothing is drawn without it."
_NO_FIELD = "SpaOTsc's scores carry no cell ids to place on the tissue: there is no field or direction."


def _labels(source: Source, limits: Mapping[str, int]) -> pd.DataFrame:
    frame = tables.read(source.fd, source.sep or ",", limits, dtype=str)
    if frame.shape[1] < 2:
        raise CCCRefusal("unsupported", "SpaOTsc's labels.csv has no group column.")
    return frame


def _groups(frame: pd.DataFrame) -> list[dict]:
    """Every column that groups the cells -- not one naming each cell once (an identifier)."""
    out = []
    for j, name in enumerate(frame.columns):
        values = frame[name].dropna()
        n_levels = int(values.nunique())
        if 2 <= n_levels < len(frame):
            out.append({"id": f"labels.{j}", "label": str(name), "n_levels": n_levels})
    return out


class SpaotscAdapter:
    backend = "spaotsc"

    def describe(self, sources: list[Source], limits: Mapping[str, int]) -> dict:
        roles = by_role(sources)
        if "signaling" not in roles:
            raise CCCRefusal("unsupported", "This SpaOTsc result has no signaling_scores.csv.")
        header = tables.header(roles["signaling"].fd, roles["signaling"].sep or ",", limits)
        if (detect.table_columns(header, "signaling_scores.csv") or {}).get("shape") != "square":
            raise CCCRefusal("unsupported", "SpaOTsc's signaling_scores.csv is not a numbered n x n matrix.")
        warnings = [_NO_FIELD]
        facets: list[dict] = []
        groups: list[dict] = []
        if "labels" in roles:
            groups = _groups(_labels(roles["labels"], limits))
            facets.append(
                {
                    "id": "matrix",
                    "keys": [{"key": "signaling", "label": "Signalling score"}],
                    "stat": "sum",
                    "permutations": {"default": DEFAULT_PERMUTATIONS, "max": MAX_PERMUTATIONS},
                }
            )
        else:
            warnings.append(_NO_LABELS)
        return {
            "label": "SpaOTsc signalling",
            "n_obs": len(header),
            "unit": "cell",
            "facets": facets,
            "groups": groups,
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
        if facet != "matrix":
            raise not_offered(facet, _NO_FIELD)
        if "labels" not in roles:
            raise not_offered(facet, _NO_LABELS)
        frame = tables.read(roles["signaling"].fd, roles["signaling"].sep or ",", limits)
        scores = frame.to_numpy(dtype=np.float64, na_value=np.nan)
        n = scores.shape[0]
        if scores.shape != (n, n):
            raise CCCRefusal("unsupported", "SpaOTsc's signaling_scores.csv is not square.")
        labelled = _labels(roles["labels"], limits)
        if len(labelled) != n:
            raise CCCRefusal(
                "no_join", f"labels.csv names {len(labelled):,} cells, the scores {n:,}: they do not match."
            )
        offered = _groups(labelled)
        if group is None:
            if not offered:
                raise CCCRefusal("bad_request", "labels.csv has no column that groups the cells.")
            column = int(offered[0]["id"].split(".")[1])
        elif group.get("source") != "labels" or int(group["column"]) not in [
            int(g["id"].split(".")[1]) for g in offered
        ]:
            raise CCCRefusal("bad_request", "The group asked for is not a grouping column of labels.csv.")
        else:
            column = int(group["column"])
        import pandas as pd

        codes, uniques = pd.factorize(labelled.iloc[:, column], use_na_sentinel=True)
        codes = codes.astype(np.int64)
        mask = h5x.rows_mask(rows, n)
        if mask is not None:
            codes = np.where(mask, codes, -1)
        codes, labels, pooled = grp.pooled_codes(codes, [str(u) for u in uniques])
        senders, receivers = np.nonzero(np.nan_to_num(scores) != 0)
        values = scores[senders, receivers]
        k = len(labels)
        sums, n_links = grp.aggregate(senders, receivers, values, codes, k)
        sizes = np.bincount(codes[codes >= 0], minlength=k)
        wanted = int(params.get("permutations", DEFAULT_PERMUTATIONS))
        seed = int(params.get("seed", 0))
        ran = grp.permutation_budget(values.size, wanted)
        warnings: list[str] = []
        z = p = None
        if ran > 0:
            z, p = grp.permutation(senders, receivers, values, codes, k, n=ran, seed=seed)
        if ran < wanted:
            warnings.append(
                f"{ran} of the {wanted} permutations asked for were run: links x permutations is held under "
                f"{PERMUTE_BUDGET:,}."
            )
        payload = matrix_payload(
            key="signaling",
            key_label="Signalling score",
            stat="sum",
            group={"id": f"labels.{column}", "label": str(labelled.columns[column]), "source": "labels"},
            labels=labels,
            values=sums,
            n=sizes.tolist(),
            pooled=pooled,
            mean=grp.mean_per_pair(sums, sizes),
            z=z,
            p=p,
            permutations=ran,
            seed=seed,
            n_links=n_links,
            rows_used=int(n if mask is None else mask.sum()),
            rows_total=n,
        )
        return {"facet": "matrix", "payload": payload, "warnings": warnings}


ADAPTER = SpaotscAdapter()
