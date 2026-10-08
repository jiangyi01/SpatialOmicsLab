"""The display sampler (``sog.vizsample/1``): which points the explorer draws when it cannot draw them all.

A browser can draw a few hundred thousand points; a dataset can hold twenty million. So the explorer
draws a sample, and this module decides which. The sample is for display only: it never changes a
dataset or a result, and the caption it writes says so, together with how many points are drawn out
of how many exist and by what rule.

Three properties are the point of it:

* **Reproducible.** The same rows, seed and budget give the same bytes on any machine. Each row's
  priority is a hash of its ORIGINAL row index, not of its position in this view, so the same rows
  win in every view of a dataset and a point selected in one view is the same point in the next.
* **Prefix-stable.** One seeded order is computed and the sample is its first ``budget`` entries, so
  ``order(budget=B1).order == order(budget=B2).order[:B1]`` for any ``B1 < B2``. Raising the budget
  only appends, and a chunk of the sample is a plain slice of it.
* **Level-of-detail order.** The first points cover the picture -- every occupied region and every
  group -- and later points fill it in at the data's own density, so a partly loaded view is already
  the right shape.

The public API (numpy only; this module imports nothing heavier, so a child process or a notebook can
use it without pandas, h5py or anndata):

``priority(rows, seed=0) -> uint64 array``
    ``splitmix64(seed_key(seed) XOR row)`` for each original row index.
``seed_key(seed) -> int``
    The first 8 bytes, little-endian, of ``sha256("sog.vizsample/1|<seed>")``.
``splitmix64(x) -> uint64 array``
    ``z = x + 0x9E3779B97F4A7C15; z = (z ^ z >> 30) * 0xBF58476D1CE4E5B9;
    z = (z ^ z >> 27) * 0x94D049BB133111EB; z ^ z >> 31`` (mod 2**64). A bijection, so distinct rows
    never share a priority.
``order(coords, *, budget, seed=0, groups=None, valid=None, method="balanced", rows=None, grid=None,
phi=0.2, unit="points") -> SampleOrder``
    ``.order`` holds uint32 POSITIONS into ``coords`` (not original rows; map them through ``rows``),
    ``min(budget, n_valid)`` of them, in level-of-detail order. ``.record`` is the explorer's
    ``sampling`` record (algorithm, method, seed, seed_key, stratify -- left ``None`` for the caller
    to fill --, grid, floor_share, caption) plus ``counts`` and, when groups are given, ``groups``:
    one ``{code, n_total, n_drawn}`` per distinct code, so a column of a million codes gives a list
    a million long (the explorer stratifies only by columns within ``SOG_VIZ_MAX_LEVELS``, at most 8,192).

How ``balanced`` (the default) orders the valid points:

1. Cells. The bounds are each axis's 0.5th and 99.5th percentiles over the valid points, and points
   beyond them clamp to the edge cells, so one stray coordinate cannot squash the grid. 2D uses 64 x 64
   cells; 3D (three coordinate columns) uses 16 x 16 in x and y and one z layer per distinct z value
   when there are at most 64 of them (a stack of sections), else 16 z bins. The distinct values are
   found exactly, by probing 65,536 of them per pass; a stack sorted or interleaved by section takes
   one pass, one with a section too small for the probe two, and a layout built to dodge the probe
   one pass per plane (at most 65).
2. Key. ``K = min(r / n_cell, q * (k / (phi * n_valid)))``, evaluated in float64 in that order, where
   ``r`` is the point's rank by priority within its cell, ``n_cell`` the valid points in that cell,
   ``q`` its rank by priority within its group and ``k`` the number of groups among the valid points
   (missing, code -1, is a group of its own here). The first point of every cell and of every group
   has key 0, so the coverage map comes first; the cell term then admits each cell in proportion to
   its density, and the group term holds a floor for every group. In a prefix of ``b`` points, with
   ``c`` occupied cells, every group keeps at least ``min(n_group, phi * (b - c - k) / ((1 + phi) *
   k))`` -- nearer ``phi * b / k`` in practice, but no floor at all below ``c + k`` points, where the
   coverage-first points have not all been drawn; the caption then says how many regions and groups
   the sample reaches. Ties go to the lower priority, then the lower position (rows repeated in
   ``rows`` share a priority). Above 4,096 groups (not counting missing) the group floor is switched
   off and the record says so.
3. Order. The sample is the ``budget`` smallest keys, in key order.

The groups the caption and ``record["coverage"]`` count are the column's levels: rows with no level
get a floor of their own but are not a group there, as they are not one in the 4,096 limit.

``uniform`` orders by priority alone: a seeded uniform random sample, still prefix-stable. When every
valid point fits in the budget, all of them are returned, still in level-of-detail order, and the
record says nothing is sampled.

Non-finite coordinates are never drawn and are counted in ``n_nonfinite``; finite rows that ``valid``
excludes (a filter, such as hidden groups) are counted in ``n_filtered_out``; so
``n_total == n_valid + n_nonfinite + n_filtered_out`` always. ``fraction`` is ``n_drawn / n_total``
rounded DOWN to four significant figures, so a sampled view never reads 1.0.

The caption counts in ``unit``, a plural ("spots"); a count of one takes the singular, which is the
plural less its final s, or ``nucleus`` for ``nuclei``.
"""

from __future__ import annotations

import hashlib
import math
import operator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: The algorithm id every record carries. A change to anything that moves which points are drawn is a
#: new id, so a recorded sample names the rule that reproduces it.
ALGORITHM = "sog.vizsample/1"
METHODS = ("balanced", "uniform")
#: Cells per axis in 2D, and in x and y in 3D.
GRID_2D = (64, 64)
GRID_3D_XY = (16, 16)
#: A 3D view with at most this many distinct z values is a stack of planes, one layer each; above it, z
#: is binned into :data:`Z_BINS` layers like any other axis.
MAX_Z_PLANES = 64
Z_BINS = 16
#: The share of any prefix the group floor sets aside, split evenly over the groups.
FLOOR_SHARE = 0.2
#: Above this many groups a floor would be a point or less each; it is switched off and the record says so.
MAX_FLOOR_GROUPS = 4096
#: The per-axis percentiles that bound the grid. Points outside clamp to the edge cells.
BOUND_PERCENTILES = (0.5, 99.5)
MISSING = -1

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31 = np.uint64(30), np.uint64(27), np.uint64(31)
# The radix sort numpy uses for a stable argsort of 16-bit keys is what keeps the two within-cell and
# within-group rankings linear; the grid and the group count are capped so both fit in 16 bits.
_MAX_CELLS = 1 << 16
# Distinct z values are found by probing this many values per pass, spread over the rows by a golden-ratio
# (Weyl) sequence rather than a stride, so a stack whose rows are sorted by section, or interleaved
# section by section, is seen whole in one pass rather than one section per pass.
_Z_PROBE = 65536
_GOLDEN_FRACTION = (math.sqrt(5.0) - 1.0) / 2.0
# Above this group code a bincount would be a large table; the codes are counted by sorting instead.
_BINCOUNT_MAX_CODE = 1 << 22
# Unit words whose singular is not the plural less a final s ("1 nucleus", not "1 nuclei").
_SINGULAR = {"nuclei": "nucleus"}


@dataclass(frozen=True, eq=False)
class SampleOrder:
    """The order the explorer draws a view in, and the record that reproduces and describes it.

    ``order`` holds uint32 positions into the ``coords`` that were passed (not original rows), the
    first ``min(budget, n_valid)`` of one seeded order. ``record`` is the ``sampling`` record plus
    ``counts`` and, when groups were given, ``groups``.
    """

    order: np.ndarray
    record: dict[str, Any] = field(default_factory=dict)


def seed_key(seed: int) -> int:
    """The 64-bit key a seed stands for: the first 8 bytes, little-endian, of ``sha256("sog.vizsample/1|<seed>")``."""
    digest = hashlib.sha256(f"{ALGORITHM}|{_seed(seed)}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little")


def splitmix64(x: Any) -> np.ndarray:
    """SplitMix64's output function over a uint64 array (a new array; ``x`` is not modified)."""
    return _mix(np.array(x, dtype=np.uint64, copy=True))


def priority(rows: Any, seed: int = 0) -> np.ndarray:
    """Each original row's priority for ``seed``: ``splitmix64(seed_key(seed) XOR row)``, uint64.

    Lower draws first. It depends on the row and the seed only -- not on the view, the budget or how
    many rows there are -- so the same rows win in every view of one dataset.
    """
    return _priority(_row_indices(rows, "rows"), _seed(seed))


def order(
    coords: Any,
    *,
    budget: int,
    seed: int = 0,
    groups: Any = None,
    valid: Any = None,
    method: str = "balanced",
    rows: Any = None,
    grid: Any = None,
    phi: float = FLOOR_SHARE,
    unit: str = "points",
) -> SampleOrder:
    """The first ``budget`` points of one seeded level-of-detail order over ``coords``.

    ``coords`` is ``(n, 2)`` or ``(n, 3)``. ``groups`` are int codes per row (``-1`` = missing) that
    get a floor in ``balanced``; ``valid`` is a boolean mask (a filter) applied on top of finiteness;
    ``rows`` are the original row indices the priorities come from, one per row of ``coords`` whether
    it is shown or not (default ``arange(n)``); neither they nor any other input is written. ``grid``
    overrides the cells per axis: ``(gx, gy)`` in 2D, ``(gx, gy, gz)`` in 3D, where ``gz`` is the z
    bin count used only when z has more than 64 distinct values. ``unit`` is the word the caption
    counts in ("spots", "cells").
    """
    seed = _seed(seed)
    xyz = np.asarray(coords)
    if xyz.ndim != 2 or xyz.shape[1] not in (2, 3):
        raise ValueError(f"coords must be an (n, 2) or (n, 3) array; got shape {tuple(xyz.shape)}")
    if xyz.dtype.kind not in "fiu":
        raise ValueError(f"coords must be numeric; got dtype {xyz.dtype}")
    n, dims = int(xyz.shape[0]), int(xyz.shape[1])
    if n > np.iinfo(np.uint32).max:
        raise ValueError(f"{n:,} rows do not fit the uint32 positions a sample is returned as")
    budget = _positive_int(budget, "budget")
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}; got {method!r}")
    phi = float(phi)
    if not (math.isfinite(phi) and 0.0 < phi <= 1.0):
        raise ValueError(f"phi (the floor share) must be in (0, 1]; got {phi!r}")
    unit = str(unit or "").strip() or "points"
    shape = _grid_shape(grid, dims)

    keep = np.isfinite(xyz).all(axis=1) if xyz.dtype.kind == "f" else np.ones(n, dtype=bool)
    n_nonfinite = n - int(np.count_nonzero(keep))
    n_filtered_out = 0
    if valid is not None:
        mask = np.asarray(valid)
        if mask.dtype != np.bool_ or mask.shape != (n,):
            raise ValueError(f"valid must be a boolean mask of shape ({n},); got {mask.dtype} {tuple(mask.shape)}")
        n_filtered_out = int(np.count_nonzero(keep & ~mask))
        keep &= mask
    # ``pos`` maps a valid point back to its position in ``coords``; None when every row is valid, so
    # the common case does not pay for an identity array.
    pos = None if n_nonfinite == 0 and n_filtered_out == 0 else np.flatnonzero(keep)
    del keep
    m = n if pos is None else int(pos.size)

    if rows is None:
        # The positions are the rows, so they are distinct and so are their priorities.
        p = _priority(np.arange(n, dtype=np.uint64) if pos is None else pos.astype(np.uint64), seed)
        distinct_rows = True
    else:
        row_index = _row_indices(rows, "rows")
        if row_index.shape != (n,):
            raise ValueError(f"rows must have one entry per coordinate row ({n}); got {row_index.shape[0]}")
        p = _priority(row_index if pos is None else row_index[pos], seed)
        distinct_rows = False
        del row_index

    codes = _group_codes(groups, n) if groups is not None else None
    take = min(budget, m)
    record: dict[str, Any] = {
        "algorithm": ALGORITHM,
        "method": method,
        "seed": seed,
        "seed_key": f"{seed_key(seed):016x}",
        "stratify": None,
        "grid": None,
        "floor_share": None,
    }
    notes: list[str] = []
    coverage: dict[str, int] | None = None
    floor_groups = 0
    # Dense id 0 is the missing group when there is one (codes are renumbered in ascending order), and
    # the caption counts levels only, so the named groups are the ids from here on.
    first_named = 0
    floor_off = 0

    if m == 0:
        chosen = np.empty(0, dtype=np.int64)
    elif method == "uniform":
        chosen = _priority_order(p, distinct_rows) if take >= m else _smallest(p, take)
    else:
        cell, grid_used, z_layers = _cells(xyz, pos, shape)
        record["grid"] = list(grid_used)
        if z_layers:
            record["z_layers"] = z_layers
        by_priority = _priority_order(p, distinct_rows)
        cell_sizes = np.bincount(cell, minlength=int(np.prod(grid_used)))
        key = _within_rank_term(cell, by_priority, cell_sizes, per_label=True)
        dense = None
        if codes is not None:
            dense, k, k_named = _dense_codes(codes if pos is None else codes[pos])
            if k_named > MAX_FLOOR_GROUPS:
                floor_off = k_named
                dense = None
                notes.append(
                    f"the group floor is off: {k_named:,} groups is more than the {MAX_FLOOR_GROUPS:,} it can serve"
                )
            else:
                floor_groups = k
                first_named = k - k_named
                group_sizes = np.bincount(dense, minlength=k)
                scale = k / (phi * m)
                term = _within_rank_term(dense, by_priority, group_sizes, per_label=False, scale=scale)
                np.minimum(key, term, out=key)
                del term
                record["floor_share"] = phi
        if take >= m:
            # The points are already in priority order (ties by position), so one stable sort of their
            # keys gives the full order: key, then priority, then position.
            chosen = by_priority[np.argsort(key[by_priority], kind="stable")]
        else:
            chosen = _smallest(key, take, tiebreak=p)
        del key, by_priority
        if take < m:
            drawn_cells = np.bincount(cell[chosen], minlength=cell_sizes.size)
            coverage = {
                "cells_drawn": int(np.count_nonzero(drawn_cells)),
                "cells_occupied": int(np.count_nonzero(cell_sizes)),
            }
            if dense is not None:
                drawn_groups = np.bincount(dense[chosen], minlength=floor_groups)[first_named:]
                coverage["groups_drawn"] = int(np.count_nonzero(drawn_groups))
                coverage["groups_present"] = floor_groups - first_named
            record["coverage"] = coverage
        del cell, dense
    del p

    out = chosen if pos is None else pos[chosen]
    del pos, chosen
    n_drawn = int(out.size)
    counts = {
        "n_total": n,
        "n_valid": m,
        "n_nonfinite": n_nonfinite,
        "n_filtered_out": n_filtered_out,
        "n_drawn": n_drawn,
        "sampled": n_drawn < m,
        "fraction": _fraction(n_drawn, n),
    }
    record["caption"] = _caption(counts, method=method, seed=seed, unit=unit, coverage=coverage, floor_off=floor_off)
    record["counts"] = counts
    if notes:
        record["notes"] = notes
    if codes is not None:
        record["groups"] = _group_counts(codes, codes[out])
    return SampleOrder(order=out.astype(np.uint32), record=record)


# --------------------------------------------------------------------------------------------------
# Priorities


def _mix(z: np.ndarray) -> np.ndarray:
    """SplitMix64's output function, in place on a uint64 array (wraps mod 2**64 by design)."""
    z += _GOLDEN
    z ^= z >> _S30
    z *= _MIX1
    z ^= z >> _S27
    z *= _MIX2
    z ^= z >> _S31
    return z


def _priority(rows: np.ndarray, seed: int) -> np.ndarray:
    """The priorities of ``rows`` (a fresh uint64 array the caller hands over; it is overwritten)."""
    rows ^= np.uint64(seed_key(seed))
    return _mix(rows)


def _priority_order(p: np.ndarray, distinct: bool) -> np.ndarray:
    """Positions sorted by priority, ties by position.

    Distinct rows have distinct priorities (the hash is a bijection), so the fast unstable sort is
    already exact. Repeated rows share one priority and fall to the stable sort, which breaks the tie by
    position; that is the only case it costs anything.
    """
    by_priority = np.argsort(p)
    if not distinct and by_priority.size > 1:
        ranked = p[by_priority]
        if bool(np.any(ranked[1:] == ranked[:-1])):
            by_priority = np.argsort(p, kind="stable")
    return by_priority


# --------------------------------------------------------------------------------------------------
# Cells


def _cells(xyz: np.ndarray, pos: np.ndarray | None, shape: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, ...], str]:
    """``(cell per valid point as uint16, the grid used, "planes" | "bins" in 3D else "")``.

    ``pos`` selects the valid points (None: all of them). Each axis is copied to float64 before it is
    binned in place, so the caller's array is never written.
    """
    m = xyz.shape[0] if pos is None else pos.size
    cell = np.zeros(m, dtype=np.int32)
    used: list[int] = []
    z_layers = ""
    for axis in range(xyz.shape[1]):
        values = xyz[:, axis].astype(np.float64) if pos is None else xyz[pos, axis].astype(np.float64, copy=False)
        if axis == 2:
            planes = _z_planes(values)
            if planes is not None:
                index, n_planes = planes
                z_layers = "planes"
                cell *= n_planes
                cell += index
                used.append(n_planes)
                continue
            z_layers = "bins"
        bins = shape[axis]
        cell *= bins
        cell += _bin(values, bins)
        used.append(bins)
    return cell.astype(np.uint16), tuple(used), z_layers


def _bin(values: np.ndarray, bins: int) -> np.ndarray:
    """Each value's cell along one axis, over the percentile bounds; outliers clamp to the edge cells."""
    lo, hi = np.percentile(values, BOUND_PERCENTILES)
    span = float(hi) - float(lo)
    if not (math.isfinite(span) and span > 0.0):
        return np.zeros(values.size, dtype=np.int32)
    values -= lo
    values *= bins / span
    np.floor(values, out=values)
    np.clip(values, 0, bins - 1, out=values)
    return values.astype(np.int32)


def _z_planes(z: np.ndarray) -> tuple[np.ndarray, int] | None:
    """``(plane index per point, number of planes)`` when z has at most :data:`MAX_Z_PLANES` distinct values.

    Exact, and linear in practice: each pass adds the distinct values of a probe of the points not yet
    accounted for and drops every point those values account for, so a stack sorted or interleaved by
    section takes one pass, and one with a section too small for the probe to land on takes two,
    rather than a sort of every z. Every probe includes the first remaining point, so each pass finds
    at least one new value and the loop ends within :data:`MAX_Z_PLANES` + 1 passes.
    """
    planes = np.empty(0, dtype=np.float64)
    rest = z
    first_index = None
    while rest.size:
        planes = np.union1d(planes, rest[_z_probe(rest.size)])
        if planes.size > MAX_Z_PLANES:
            return None
        at = np.searchsorted(planes, rest)
        np.minimum(at, planes.size - 1, out=at)
        found = planes[at] == rest
        if rest is z and bool(found.all()):
            first_index = at
        rest = rest[~found]
    index = first_index if first_index is not None else np.searchsorted(planes, z)
    return index.astype(np.int32), int(planes.size)


def _z_probe(size: int) -> np.ndarray | slice:
    """The positions one pass of :func:`_z_planes` reads out of ``size`` remaining points.

    All of them up to :data:`_Z_PROBE`; above it, :data:`_Z_PROBE` positions ``floor(frac(j * g) *
    size)`` for the golden ratio's fraction ``g``. That sequence fills the range evenly at every scale
    and never lines up with a period in the rows, which a stride would; ``j = 0`` is position 0.
    """
    if size <= _Z_PROBE:
        return slice(None)
    spread = np.arange(_Z_PROBE, dtype=np.float64) * _GOLDEN_FRACTION
    spread -= np.floor(spread)
    spread *= size
    return np.minimum(spread.astype(np.int64), size - 1)


# --------------------------------------------------------------------------------------------------
# Keys and order


def _within_rank_term(
    labels: np.ndarray, by_priority: np.ndarray, sizes: np.ndarray, *, per_label: bool, scale: float = 1.0
) -> np.ndarray:
    """Each point's rank by priority among the points sharing its label, as a key term (float64).

    ``per_label``: ``rank / size_of_its_label`` (the cell term). Otherwise ``rank * scale`` (the group
    term). ``labels`` are uint16, so the stable argsort below is numpy's linear radix sort.
    """
    m = labels.size
    within = np.argsort(labels[by_priority], kind="stable")
    # Sorted by (label, priority), the labels run in blocks of ``sizes``, so a point's rank in its block
    # is its place in the run minus where the block starts -- built from the sizes, with no gather.
    starts = (np.cumsum(sizes) - sizes).astype(np.float64)
    term = np.arange(m, dtype=np.float64)
    term -= np.repeat(starts, sizes)
    if per_label:
        term /= np.repeat(sizes.astype(np.float64), sizes)
    else:
        term *= scale
    out = np.empty(m, dtype=np.float64)
    out[by_priority[within]] = term
    return out


def _smallest(key: np.ndarray, take: int, tiebreak: np.ndarray | None = None) -> np.ndarray:
    """Positions of the ``take`` smallest keys (``take < key.size``), in key order; ties by ``tiebreak``, then position.

    The cut is found by partition, so only the chosen points (and the few that tie at the cut) are
    sorted -- which is also what makes the result the exact prefix of the full order. The full order
    is the callers' own: for ``uniform`` it is the priority order, and for ``balanced`` one stable sort
    of the keys over the points already in priority order.
    """

    def ordered(index: np.ndarray) -> np.ndarray:
        if tiebreak is None:
            return index[np.lexsort((index, key[index]))]
        return index[np.lexsort((index, tiebreak[index], key[index]))]

    cut = np.partition(key, take - 1)[take - 1]
    below = np.flatnonzero(key < cut)
    tied = np.flatnonzero(key == cut)
    need = take - below.size
    if need < tied.size:
        tied = ordered(tied)[:need]
    return ordered(np.concatenate([below, tied]))


# --------------------------------------------------------------------------------------------------
# Groups


def _group_codes(groups: Any, n: int) -> np.ndarray:
    codes = np.asarray(groups)
    if codes.shape != (n,):
        raise ValueError(f"groups must have one code per coordinate row ({n}); got shape {tuple(codes.shape)}")
    if codes.dtype.kind not in "iu":
        raise ValueError(f"groups must be integer codes with {MISSING} for missing; got dtype {codes.dtype}")
    codes = codes.astype(np.int64, copy=False)
    if n and int(codes.min()) < MISSING:
        raise ValueError(f"group codes must be >= {MISSING} ({MISSING} = missing); got {int(codes.min())}")
    return codes


def _code_counts(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(distinct codes ascending, how many of each)``."""
    if codes.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    top = int(codes.max())
    if top < _BINCOUNT_MAX_CODE:
        counts = np.bincount(codes + 1, minlength=top + 2)
        present = np.flatnonzero(counts)
        return present - 1, counts[present]
    return np.unique(codes, return_counts=True)


def _dense_codes(codes: np.ndarray) -> tuple[np.ndarray | None, int, int]:
    """``(codes renumbered 0..k-1 as uint16, k, the named groups among them)``; missing is a group of its own."""
    present, _ = _code_counts(codes)
    k = int(present.size)
    k_named = k - int(bool(k) and int(present[0]) == MISSING)
    if k_named > MAX_FLOOR_GROUPS:
        return None, k, k_named
    if k and int(present[-1]) < _BINCOUNT_MAX_CODE:
        table = np.zeros(int(present[-1]) + 2, dtype=np.uint16)
        table[present + 1] = np.arange(k, dtype=np.uint16)
        return table[codes + 1], k, k_named
    return np.searchsorted(present, codes).astype(np.uint16), k, k_named


def _group_counts(codes: np.ndarray, drawn: np.ndarray) -> list[dict[str, int]]:
    """``[{code, n_total, n_drawn}]`` over every code in the column, missing (-1) first when present."""
    present, totals = _code_counts(codes)
    drawn_codes, drawn_counts = _code_counts(drawn)
    n_drawn = np.zeros(present.size, dtype=np.int64)
    n_drawn[np.searchsorted(present, drawn_codes)] = drawn_counts
    return [
        {"code": int(c), "n_total": int(t), "n_drawn": int(d)}
        for c, t, d in zip(present.tolist(), totals.tolist(), n_drawn.tolist(), strict=True)
    ]


# --------------------------------------------------------------------------------------------------
# Caption and arguments


def _caption(
    counts: dict[str, Any], *, method: str, seed: int, unit: str, coverage: dict[str, int] | None, floor_off: int
) -> str:
    n_total, n_drawn = counts["n_total"], counts["n_drawn"]
    if n_total == 0:
        return f"There are no {unit} to show."
    if not counts["sampled"]:
        if n_drawn == n_total:
            lead = f"Showing all {n_total:,} {unit}." if n_total > 1 else f"Showing {_many(1, unit)}."
        else:
            lead = f"Showing {n_drawn:,} of {_many(n_total, unit)}."
        sentences = [lead, "Nothing is sampled."]
    elif method == "uniform":
        sentences = [
            f"Showing {n_drawn:,} of {n_total:,} {unit}: a uniform random display sample (seed {seed}).",
            "The data are unchanged.",
        ]
    else:
        cov = coverage or {}
        regions_whole = cov.get("cells_drawn") == cov.get("cells_occupied")
        groups_whole = cov.get("groups_drawn") == cov.get("groups_present")
        with_groups = bool(cov.get("groups_present"))
        if regions_whole and groups_whole:
            what = "every group and region" if with_groups else "every region"
            how = f"that keeps {what} represented"
        else:
            reach = f"{cov.get('cells_drawn', 0):,} of {_many(cov.get('cells_occupied', 0), 'regions')}"
            if with_groups:
                reach += f" and {cov['groups_drawn']:,} of {_many(cov['groups_present'], 'groups')}"
            how = f"that reaches {reach} at this budget"
        sentences = [
            f"Showing {n_drawn:,} of {n_total:,} {unit}: a balanced display sample (seed {seed}) {how}.",
            "The data are unchanged.",
        ]
        if floor_off:
            sentences.append(
                f"With {floor_off:,} groups, more than the {MAX_FLOOR_GROUPS:,} a group floor serves, "
                "small groups are not guaranteed a place."
            )
    if counts["n_nonfinite"]:
        one = counts["n_nonfinite"] == 1
        sentences.append(
            f"{_many(counts['n_nonfinite'], unit)} {'has' if one else 'have'} no finite coordinates "
            f"and {'is' if one else 'are'} not drawn."
        )
    if counts["n_filtered_out"]:
        one = counts["n_filtered_out"] == 1
        sentences.append(f"{_many(counts['n_filtered_out'], unit)} {'is' if one else 'are'} left out by the filter.")
    return " ".join(sentences)


def _many(count: int, plural: str) -> str:
    """``"1,234 spots"``; for one, ``"1 spot"`` (the plural less its s) or ``"1 nucleus"`` (:data:`_SINGULAR`)."""
    if count != 1:
        return f"{count:,} {plural}"
    if plural in _SINGULAR:
        return f"1 {_SINGULAR[plural]}"
    return f"1 {plural[:-1]}" if len(plural) > 1 and plural.endswith("s") else f"1 {plural}"


def _fraction(n_drawn: int, n_total: int) -> float:
    """``n_drawn / n_total`` rounded DOWN to four significant figures, so only a whole view reads 1.0.

    Integer arithmetic throughout: ``places`` puts the first significant digit of the fraction in the
    thousands, and floor division drops the rest. Rounding to nearest would report 99,999 of 100,000
    as 1.0 beside ``sampled: true``.
    """
    if n_total <= 0 or n_drawn <= 0:
        return 0.0
    places = 3
    while n_drawn * 10 ** (places - 3) < n_total:
        places += 1
    return (n_drawn * 10**places // n_total) / 10**places


def _seed(seed: Any) -> int:
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer, not a boolean")
    try:
        return operator.index(seed)
    except TypeError:
        raise ValueError(f"seed must be an integer; got {seed!r}") from None


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, not a boolean")
    try:
        number = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be a positive integer; got {value!r}") from None
    if number < 1:
        raise ValueError(f"{name} must be at least 1; got {number}")
    return number


def _row_indices(rows: Any, name: str) -> np.ndarray:
    index = np.asarray(rows)
    if index.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array of row indices; got shape {tuple(index.shape)}")
    if index.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integer row indices; got dtype {index.dtype}")
    if index.dtype.kind == "i" and index.size and int(index.min()) < 0:
        raise ValueError(f"{name} must be non-negative row indices; got {int(index.min())}")
    return index.astype(np.uint64)


def _grid_shape(grid: Any, dims: int) -> tuple[int, ...]:
    if grid is None:
        return GRID_2D if dims == 2 else (*GRID_3D_XY, Z_BINS)
    try:
        shape = tuple(operator.index(v) for v in grid)
    except TypeError:
        raise ValueError(f"grid must be a tuple of integers, one per axis; got {grid!r}") from None
    if len(shape) != dims:
        raise ValueError(f"grid needs one entry per coordinate axis ({dims}); got {len(shape)}")
    if any(v < 1 for v in shape):
        raise ValueError(f"every grid entry must be at least 1; got {shape}")
    cells = shape[0] * shape[1] * (max(shape[2], MAX_Z_PLANES) if dims == 3 else 1)
    if cells > _MAX_CELLS:
        raise ValueError(f"a grid of {cells:,} cells is more than the {_MAX_CELLS:,} the sampler ranks within")
    return shape
