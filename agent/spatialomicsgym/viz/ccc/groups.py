"""Who signals to whom, group by group: COMMOT's spot-by-spot links summed into a groups-by-groups matrix.

COMMOT's own ``cluster_communication`` would write ``uns['commot_cluster-...']``, but the platform's worker never calls
it, so the matrix is derived here from ``obsp['commot-<db>-<key>']`` and a group column: ``sum[a, b]`` is the signal
every spot of group ``a`` sends to every spot of group ``b`` -- ``Sᵀ W R`` with ``S``/``R`` the one-hot memberships,
computed as one ``bincount`` over the links (O(nnz), no dense n x n anywhere). The same goes for spaotsc's dense
signalling scores grouped by its ``labels.csv``.

Whether a cell of that matrix is more than the group sizes would give by chance is a permutation test: the group
labels are shuffled among the spots ``n`` times (a seeded PCG64, so the same request answers the same), each shuffle one
more ``bincount``. Its cost is links x permutations, held under :data:`facets.PERMUTE_BUDGET` by
:func:`permutation_budget`.
"""

from __future__ import annotations

import numpy as np

from .facets import MAX_GROUPS, PERMUTE_BUDGET


def pooled_codes(
    codes: np.ndarray, labels: list[str], max_groups: int = MAX_GROUPS
) -> tuple[np.ndarray, list[str], dict[str, int] | None]:
    """``(codes, labels, pooled)`` with at most ``max_groups`` groups.

    Within the limit nothing changes and ``pooled`` is ``None``. Past it the ``max_groups - 1`` largest groups keep
    their label (in their original order) and the rest become one ``"other (N groups)"``, the last; ``pooled`` is
    ``{"n_levels": N, "n_total": rows in them}``. A code of ``-1`` (no group) stays ``-1``.
    """
    codes = np.asarray(codes, dtype=np.int64)
    if len(labels) <= int(max_groups):
        return codes, list(labels), None
    sizes = np.bincount(codes[codes >= 0], minlength=len(labels))
    largest = np.sort(np.argsort(-sizes, kind="stable")[: int(max_groups) - 1])
    remap = np.full(len(labels), int(max_groups) - 1, dtype=np.int64)
    remap[largest] = np.arange(largest.size)
    out = np.where(codes >= 0, remap[np.clip(codes, 0, None)], -1)
    n_pooled = len(labels) - largest.size
    rest = np.ones(len(labels), dtype=bool)
    rest[largest] = False
    new_labels = [labels[i] for i in largest.tolist()] + [f"other ({n_pooled} groups)"]
    return out, new_labels, {"n_levels": int(n_pooled), "n_total": int(sizes[rest].sum())}


def aggregate(
    rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, codes: np.ndarray, k: int
) -> tuple[np.ndarray, int]:
    """``(sum K x K, links used)``: each link's value added at (its sender's group, its receiver's group).

    A link with an end in no group (code ``-1``) is dropped.
    """
    codes = np.asarray(codes, dtype=np.int64)
    a, b = codes[rows], codes[cols]
    kept = (a >= 0) & (b >= 0)
    flat = a[kept] * int(k) + b[kept]
    sums = np.bincount(flat, weights=np.asarray(vals, dtype=np.float64)[kept], minlength=int(k) * int(k))
    return sums.reshape(int(k), int(k)), int(np.count_nonzero(kept))


def mean_per_pair(sum_kk: np.ndarray, n_by_group: np.ndarray) -> np.ndarray:
    """``sum[a, b] / (n_a * n_b)``: the signal per pair of spots, NaN where a group is empty."""
    n = np.asarray(n_by_group, dtype=np.float64)
    pairs = np.outer(n, n)
    return np.divide(sum_kk, pairs, out=np.full(pairs.shape, np.nan), where=pairs > 0)


def permutation(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    codes: np.ndarray,
    k: int,
    *,
    n: int,
    seed: int,
    strata: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(z, p)`` for every cell of :func:`aggregate`'s matrix against ``n`` shuffles of the group labels.

    The labels are shuffled among the spots that have one (a spot with none keeps none), by
    ``np.random.Generator(np.random.PCG64(seed))``. ``p`` is two-sided, ``(1 + #{|perm - mean| >= |obs - mean|}) /
    (n + 1)`` (a tie within rounding counts as extreme: a shuffle that gives the observed matrix back must not
    read as more extreme than it); ``z`` is ``(obs - mean) / sd``, NaN where the shuffles never moved the cell (sd 0).

    ``strata`` (one code per spot) shuffles the labels within each stratum only -- a per-section result's sections,
    whose links never cross one, so a label moved across sections would test a placement no run could make.
    """
    codes = np.asarray(codes, dtype=np.int64)
    observed, _ = aggregate(rows, cols, vals, codes, k)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    labelled = np.flatnonzero(codes >= 0)
    if strata is None:
        blocks = [labelled]
    else:
        strata = np.asarray(strata, dtype=np.int64)[labelled]
        blocks = [labelled[strata == s] for s in np.unique(strata)]
    shuffled = codes.copy()
    draws = np.empty((int(n), int(k), int(k)))
    for i in range(int(n)):
        for block in blocks:
            shuffled[block] = rng.permutation(codes[block])
        draws[i], _ = aggregate(rows, cols, vals, shuffled, k)
    mean = draws.mean(axis=0)
    sd = draws.std(axis=0)
    extreme = np.abs(draws - mean) >= np.abs(observed - mean) - 1e-12 * np.maximum(1.0, np.abs(observed))
    p = (1.0 + extreme.sum(axis=0)) / (int(n) + 1.0)
    z = np.divide(observed - mean, sd, out=np.full(sd.shape, np.nan), where=sd > 0)
    return z, p


def permutation_budget(nnz: int, n: int) -> int:
    """How many of ``n`` permutations fit :data:`facets.PERMUTE_BUDGET` links x permutations; 0 runs none."""
    return int(min(int(n), PERMUTE_BUDGET // max(int(nnz), 1)))


__all__ = ["aggregate", "mean_per_pair", "permutation", "permutation_budget", "pooled_codes"]
