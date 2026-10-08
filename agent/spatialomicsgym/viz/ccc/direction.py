"""Which way a signal travels at each spot, derived from COMMOT's spot-by-spot matrix, and binned into arrows.

COMMOT's own ``communication_direction`` writes ``obsm['commot_sender_vf-<db>-<key>']``, but the platform's COMMOT
worker never calls it (``tools/commot_worker.py`` runs ``spatial_communication`` only), so no real output on this box
has a vector field. What every output does have is ``obsp['commot-<db>-<key>']``: sender rows, receiver columns, one
stored value per link. The field is derived from that, the way COMMOT derives it -- per spot, the weight-sum of UNIT
vectors to its ``k`` strongest receivers (sender) or from its ``k`` strongest senders (receiver) -- with no transport
problem solved again. Both roles point in the direction of travel, sender to receiver, so one arrow style reads both.

A field over a million spots is no picture, so it is binned onto a grid of at most :data:`facets.MAX_ARROWS` cells:
one arrow per occupied cell, at the mean position of its spots, the cell's summed vector as its direction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .facets import MAX_ARROWS


def csr_rows(indptr: np.ndarray) -> np.ndarray:
    """The row of every stored value of a CSR matrix (int64)."""
    indptr = np.asarray(indptr, dtype=np.int64)
    return np.repeat(np.arange(indptr.size - 1, dtype=np.int64), np.diff(indptr))


def transpose_csr(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The CSR of the transpose of an ``n``-column CSR matrix: its columns become rows (a stable sort by column)."""
    rows = csr_rows(indptr)
    order = np.argsort(indices, kind="stable")
    counts = np.bincount(indices, minlength=n) if indices.size else np.zeros(n, dtype=np.int64)
    pointers = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return pointers, rows[order], data[order]


def top_k_per_row(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rows, cols, vals)`` of the ``k`` largest stored values of every row; ``k == 0`` keeps all of them.

    One lexicographic sort (row, then value descending) and a rank within the row, rather than an ``argpartition`` per
    row: the same selection with no Python loop over a million rows. Of equal values the first stored wins.
    """
    rows = csr_rows(indptr)
    indices = np.asarray(indices, dtype=np.int64)
    data = np.asarray(data, dtype=np.float64)
    if int(k) <= 0 or data.size == 0:
        return rows, indices, data
    order = np.lexsort((-data, rows))
    rank = np.arange(order.size, dtype=np.int64) - np.asarray(indptr, dtype=np.int64)[rows[order]]
    keep = np.sort(order[rank < int(k)])
    return rows[keep], indices[keep], data[keep]


def vector_field(xy: np.ndarray, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, role: str) -> np.ndarray:
    """``(n, d)``: per spot the weight-sum of the unit vectors of its links, sender ``rows`` to receiver ``cols``.

    ``xy`` has two or three columns (``d``); a 3D result's field is three-dimensional. ``sender`` sums each link at
    its sender, ``receiver`` at its receiver; both point from sender to receiver. A link between two spots at one
    position has no direction and adds nothing.
    """
    if role not in ("sender", "receiver"):
        raise ValueError("role is 'sender' or 'receiver'")
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] not in (2, 3):
        raise ValueError("positions have two or three columns")
    step = xy[cols] - xy[rows]
    norm = np.sqrt(np.einsum("ij,ij->i", step, step))
    unit = np.divide(step, norm[:, None], out=np.zeros_like(step), where=norm[:, None] > 0)
    weighted = unit * np.asarray(vals, dtype=np.float64)[:, None]
    at = rows if role == "sender" else cols
    n = xy.shape[0]
    return np.column_stack([np.bincount(at, weights=weighted[:, j], minlength=n) for j in range(xy.shape[1])])


def cross_section_mask(rows: np.ndarray, cols: np.ndarray, section_codes: np.ndarray) -> np.ndarray:
    """Which links join two different sections: sender ``rows`` and receiver ``cols`` whose ``section_codes``
    differ, both ends in a section (a code of ``-1`` is no section, and a link touching one is never cross-section).

    Sections are told by their labels alone, never by a coordinate: Zhuang is cut coronally, so its stacking axis is
    CCF x, and no column is assumed to be depth."""
    codes = np.asarray(section_codes, dtype=np.int64)
    a, b = codes[np.asarray(rows, dtype=np.int64)], codes[np.asarray(cols, dtype=np.int64)]
    return (a >= 0) & (b >= 0) & (a != b)


def bin_arrows(
    xy: np.ndarray, vf: np.ndarray, *, max_arrows: int = MAX_ARROWS, bounds: tuple[Any, Any] | None = None
) -> dict[str, Any]:
    """The field ``vf`` at positions ``xy`` binned onto a square grid of ``floor(sqrt(max_arrows))`` cells a side.

    One arrow per cell holding at least one spot and a non-zero summed vector: ``[x, y, dx, dy, n, mag]`` -- the mean
    position of its spots, their summed vector, how many, and the mean length of their own vectors. ``max_mag`` is the
    longest summed vector, for scaling. ``bounds`` (``(min_xy, max_xy)``) defaults to the finite positions' extent;
    a spot outside them, or at a non-finite position, is left out.
    """
    xy = np.asarray(xy, dtype=np.float64)
    vf = np.asarray(vf, dtype=np.float64)
    side = max(1, math.isqrt(max(1, int(max_arrows))))
    finite = np.isfinite(xy).all(axis=1) & np.isfinite(vf).all(axis=1)
    if bounds is None:
        lo = xy[finite].min(axis=0) if finite.any() else np.zeros(2)
        hi = xy[finite].max(axis=0) if finite.any() else np.zeros(2)
    else:
        lo, hi = np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64)
    span = np.where(hi > lo, hi - lo, 1.0)
    cell = span / side
    inside = finite & (xy >= lo).all(axis=1) & (xy <= hi).all(axis=1)
    pts, vec = xy[inside], vf[inside]
    ij = np.clip(np.floor((pts - lo) / cell).astype(np.int64), 0, side - 1)
    flat = ij[:, 1] * side + ij[:, 0]
    cells = side * side
    n = np.bincount(flat, minlength=cells)
    sums = [np.bincount(flat, weights=w, minlength=cells) for w in (pts[:, 0], pts[:, 1], vec[:, 0], vec[:, 1])]
    length = np.bincount(flat, weights=np.hypot(vec[:, 0], vec[:, 1]), minlength=cells)
    summed = np.hypot(sums[2], sums[3])
    drawn = np.flatnonzero((n > 0) & (summed > 0))
    arrows = [
        [
            float(sums[0][c] / n[c]),
            float(sums[1][c] / n[c]),
            float(sums[2][c]),
            float(sums[3][c]),
            int(n[c]),
            float(length[c] / n[c]),
        ]
        for c in drawn.tolist()
    ]
    return {
        "grid": [side, side],
        "cell_size": [float(cell[0]), float(cell[1])],
        "bounds": {"min": [float(lo[0]), float(lo[1])], "max": [float(hi[0]), float(hi[1])]},
        "arrows": arrows,
        "max_mag": float(summed[drawn].max()) if drawn.size else 0.0,
    }


def bin_arrows_3d(
    xyz: np.ndarray, vf: np.ndarray, *, max_arrows: int = MAX_ARROWS, bounds: tuple[Any, Any] | None = None
) -> dict[str, Any]:
    """The 3D field ``vf`` at positions ``xyz`` binned onto a cube of ``floor(cbrt(max_arrows))`` cells a side (16
    at the cap of 4,096).

    One arrow per cell holding at least one spot and a non-zero summed vector: ``[x, y, z, dx, dy, dz, n, mag]`` --
    the mean position of its spots, their summed vector, how many, and the mean length of their own vectors -- so the
    arrows' vectors add up to the field's own sum over the spots binned. ``bounds`` (``(min_xyz, max_xyz)``) defaults
    to the finite positions' extent; a spot outside them, or at a non-finite position, is left out.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    vf = np.asarray(vf, dtype=np.float64).reshape(-1, 3)
    side = max(1, round(int(max(1, int(max_arrows))) ** (1.0 / 3.0)))
    while side**3 > max(1, int(max_arrows)):
        side -= 1
    side = max(1, side)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(vf).all(axis=1)
    if bounds is None:
        lo = xyz[finite].min(axis=0) if finite.any() else np.zeros(3)
        hi = xyz[finite].max(axis=0) if finite.any() else np.zeros(3)
    else:
        lo, hi = np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64)
    span = np.where(hi > lo, hi - lo, 1.0)
    cell = span / side
    inside = finite & (xyz >= lo).all(axis=1) & (xyz <= hi).all(axis=1)
    pts, vec = xyz[inside], vf[inside]
    ijk = np.clip(np.floor((pts - lo) / cell).astype(np.int64), 0, side - 1)
    flat = (ijk[:, 2] * side + ijk[:, 1]) * side + ijk[:, 0]
    cells = side**3
    n = np.bincount(flat, minlength=cells)
    sums = [np.bincount(flat, weights=pts[:, j], minlength=cells) for j in range(3)]
    sums += [np.bincount(flat, weights=vec[:, j], minlength=cells) for j in range(3)]
    length = np.bincount(flat, weights=np.sqrt(np.einsum("ij,ij->i", vec, vec)), minlength=cells)
    summed = np.sqrt(sums[3] ** 2 + sums[4] ** 2 + sums[5] ** 2)
    drawn = np.flatnonzero((n > 0) & (summed > 0))
    arrows = [
        [
            float(sums[0][c] / n[c]),
            float(sums[1][c] / n[c]),
            float(sums[2][c] / n[c]),
            float(sums[3][c]),
            float(sums[4][c]),
            float(sums[5][c]),
            int(n[c]),
            float(length[c] / n[c]),
        ]
        for c in drawn.tolist()
    ]
    return {
        "grid": [side, side, side],
        "cell_size": [float(c) for c in cell],
        "bounds": {"min": [float(v) for v in lo], "max": [float(v) for v in hi]},
        "arrows": arrows,
        "max_mag": float(summed[drawn].max()) if drawn.size else 0.0,
    }


__all__ = [
    "bin_arrows",
    "bin_arrows_3d",
    "cross_section_mask",
    "csr_rows",
    "top_k_per_row",
    "transpose_csr",
    "vector_field",
]
