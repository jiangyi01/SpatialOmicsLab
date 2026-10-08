"""The reads a communication panel makes of an ``.h5ad``, over :class:`spatialomicsgym.viz.h5lite.H5AD`.

h5lite does the reading (through the descriptor, sizes checked first); this module decides what may be read and
says it in the communication panels' terms. The one costly read is a spot-by-spot obsp matrix, so it is preflighted
against the reader's memory cap before a value is touched:

* reading it costs ``nnz x 12`` bytes (int32 indices + float64 values as stored), held as int64 + float64 + the rows
  each link belongs to and the working copies the kernels make -- :data:`BYTES_PER_LINK` = 96 bytes a link all told,
  with :data:`BASELINE_BYTES` for the interpreter, numpy, pandas and h5py already in the child;
* aggregating is O(nnz); a permutation test is O(nnz x permutations), held under ``facets.PERMUTE_BUDGET``.

describe reads no obsp at all: the keys and each matrix's stored count come from ``indptr[-1]`` alone.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from .errors import CCCRefusal
from .facets import MAX_NNZ

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialomicsgym.viz.h5lite import H5AD

#: What one stored link of an obsp matrix costs once read and worked on (see the module docstring).
BYTES_PER_LINK = 96
#: What the reader child holds before it reads a matrix.
BASELINE_BYTES = 512 << 20
#: Labelled obs column kinds a group can be read from.
_LABELLED = ("categorical", "bool", "nullable-bool", "string")
_COMMOT_SUM_RE = re.compile(r"^commot-(?P<db>.+)-sum-(?P<role>sender|receiver)$")


def obs_groups(h5: H5AD, max_levels: int) -> list[dict]:
    """The obs columns a matrix can be grouped by: categoricals with 2 to ``max_levels`` categories.

    Counted from the categories alone -- no column's codes are read for a listing. ``id`` is ``obs.<i>``, ``i`` the
    column's position in ``column-order`` (the explorer's own obs ids).
    """
    out = []
    for column in h5.obs_columns():
        if column.kind != "categorical":
            continue
        categories = h5.file["obs"][column.name].get("categories")
        n = int(categories.shape[0]) if categories is not None and categories.ndim == 1 else 0
        if 2 <= n <= int(max_levels):
            out.append({"id": f"obs.{column.index}", "label": column.name, "n_levels": n})
    return out


def group_codes(h5: H5AD, column: int) -> tuple[np.ndarray, list[str], str]:
    """``(codes int64, labels, column name)`` of the obs column at position ``column``; ``-1`` is no group."""
    found = next((c for c in h5.obs_columns() if c.index == int(column)), None)
    if found is None or found.kind not in _LABELLED:
        raise CCCRefusal("bad_request", "The group asked for is not a labelled obs column of this file.")
    read = h5.obs_labels(found)
    return read.codes.astype(np.int64, copy=False), list(read.labels), found.name


#: Micrometres per unit, by the unit names ``spatial3d/contract.py`` records (``tools/worker_utils.UNIT_UM``).
#: A pixel, an array index or an undeclared unit has no micrometre equivalent.
UM_PER_UNIT = {"um": 1.0, "micrometre": 1.0, "micrometer": 1.0, "micron": 1.0, "mm": 1000.0, "millimetre": 1000.0}
#: The obs columns a section label is read from, after ``uns['spatial_3d']['slice_key']`` (``viz.layers``'
#: ``SECTION_COLUMNS`` without ``batch``: merged replicates are not sections, as the workers' flattened-stack rule says).
SECTION_COLUMNS = ("library_id", "slice_id", "section", "section_id", "brain_section_label", "Bregma", "z_index")
_AXIS = {"x": 0, "y": 1, "z": 2}


def declared_frames(h5: H5AD) -> dict[str, dict]:
    """``uns['spatial_3d']['frames']`` as plain dicts (the 3D contract's declarations), ``{}`` when there are none."""
    block = h5.uns("spatial_3d")
    frames = block.get("frames") if isinstance(block, dict) else None
    return {str(k): v for k, v in frames.items() if isinstance(v, dict)} if isinstance(frames, dict) else {}


def frame_coords(h5: H5AD) -> tuple[np.ndarray, dict[str, str | None], str | None]:
    """``(coords, units, z_source)``: ``obsm['spatial']`` at its real width (2 or 3 columns, float64 -- never cut to
    two), the units its frame declares in ``uns['spatial_3d']['frames']['spatial']`` (``{"xy": ..., "z": ...}``,
    ``None`` where undeclared; ``z`` is ``None`` for two columns) and the frame's ``z_source`` (``None`` when
    undeclared). A wider or narrower ``spatial`` is refused: the explorer draws positions in two or three dimensions.
    """
    info = h5.obsm_key("spatial")
    if info is None or info.kind not in ("array", "dataframe") or info.n_cols < 2:
        raise CCCRefusal("unsupported", "This file has no obsm['spatial'] positions to draw directions on.")
    if info.n_cols > 3:
        raise CCCRefusal(
            "unsupported",
            f"obsm['spatial'] holds {info.n_cols} columns; the explorer draws positions in two or three dimensions.",
        )
    coords = np.asarray(h5.obsm("spatial"), dtype=np.float64)
    frames = declared_frames(h5)
    declared = frames.get("spatial", {})
    if not declared and coords.shape[1] == 3:
        # A three-column ``spatial`` that declares nothing of its own -- a 3D run made before blocks declared their
        # frame -- carries the declaration of the declared frame it is a copy of (its rank z, its units), if any.
        source = copy_of(h5, coords)
        if source is not None:
            declared = frames[source]
    xy = declared.get("xy_units")
    z = declared.get("z_units") if coords.shape[1] == 3 else None
    z_source = declared.get("z_source")
    units = {"xy": str(xy) if isinstance(xy, str) else None, "z": str(z) if isinstance(z, str) else None}
    return coords, units, str(z_source) if isinstance(z_source, str) else None


def units_text(units: dict[str, str | None]) -> str | None:
    """The frame's units in a word: ``"um"`` when every axis says so, ``"um/mm"`` (xy/z) when they differ, ``None``
    when undeclared."""
    xy, z = units.get("xy"), units.get("z")
    if xy is None:
        return None
    return xy if z in (None, xy) else f"{xy}/{z}"


def copy_of(h5: H5AD, coords: np.ndarray | None = None) -> str | None:
    """The declared three-column frame a three-column ``spatial`` is a copy of, or ``None``.

    A copy may be in other units than its source (a COMMOT block writes micrometres from a frame declared in mm), so
    each column is compared after both sides are put in micrometres where both declare their units; as stored where
    either does not. The first frame (in declaration order) that matches is the one."""
    if coords is None:
        info = h5.obsm_key("spatial")
        if info is None or info.kind not in ("array", "dataframe") or info.n_cols != 3:
            return None
        coords = np.asarray(h5.obsm("spatial"), dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return None
    frames = declared_frames(h5)
    mine = frames.get("spatial", {})
    for key, other in frames.items():
        info = h5.obsm_key(key)
        if key == "spatial" or info is None or info.kind != "array" or info.n_cols != 3 or info.shape[0] != len(coords):
            continue
        theirs = np.asarray(h5.obsm(key), dtype=np.float64)
        scale = np.ones(3)
        for j, unit_key in enumerate(("xy_units", "xy_units", "z_units")):
            a, b = um_per_unit(other.get(unit_key)), um_per_unit(mine.get(unit_key))
            if a is not None and b is not None:
                scale[j] = a / b
        if np.allclose(theirs * scale, coords, equal_nan=True):
            return key
    return None


def stacking_column(h5: H5AD, n_cols: int) -> tuple[int, str]:
    """``(column, why)``: the axis sections are stacked along in a three-column ``spatial``.

    The column a declared frame's ``axis_map`` calls the stacking axis (Zhuang is cut coronally: CCF x, column 0) --
    ``spatial``'s own frame first, then the frame ``spatial`` is a copy of (a COMMOT block keeps its source frame's
    declaration beside its own, which names no axes), then any other declared frame; without one, column 2, the
    contract's default (``tools/commot_worker.in_plane_axes``' rule).
    """
    if n_cols != 3:
        return -1, "two-dimensional"
    frames = declared_frames(h5)
    source = copy_of(h5)
    ordered = [frames["spatial"]] if "spatial" in frames else []
    ordered += [frames[source]] if source is not None else []
    ordered += [f for k, f in frames.items() if k not in ("spatial", source)]
    for frame in ordered:
        axis_map = frame.get("axis_map")
        if not isinstance(axis_map, dict):
            continue
        for axis, text in axis_map.items():
            if str(axis).strip().lower() in _AXIS and "stack" in str(text).lower():
                return _AXIS[str(axis).strip().lower()], "the frame's axis_map names it the section stacking axis"
    return 2, "no axis_map names a stacking axis, so the third column is taken"


def section_column(h5: H5AD) -> tuple[np.ndarray, list[str], str] | None:
    """``(codes, labels, column name)`` of the obs column that names each spot's section, or ``None``.

    ``uns['spatial_3d']['slice_key']`` first, then :data:`SECTION_COLUMNS`; a labelled column of at least one level.
    The labels are its own (a category of no spot is kept: a section can be outside a block)."""
    block = h5.uns("spatial_3d")
    candidates = list(SECTION_COLUMNS)
    if isinstance(block, dict) and isinstance(block.get("slice_key"), str):
        candidates.insert(0, block["slice_key"])
    for name in candidates:
        column = h5.obs_column(name)
        if column is None or column.kind not in _LABELLED:
            continue
        read = h5.obs_labels(column)
        return read.codes.astype(np.int64, copy=False), [str(v) for v in read.labels], column.name
    return None


def um_per_unit(unit: str | None) -> float | None:
    """Micrometres per ``unit``, or ``None`` when the unit is no physical length."""
    return UM_PER_UNIT.get(str(unit).strip().lower()) if unit else None


def preflight(nnz: int, limits: Mapping[str, int]) -> None:
    """Refuse a matrix whose read would not fit: above :data:`facets.MAX_NNZ` links whatever the cap, or above what
    ``limits['memory_bytes']`` leaves after :data:`BASELINE_BYTES` at :data:`BYTES_PER_LINK` a link."""
    memory = int(limits["memory_bytes"])
    need = int(nnz) * BYTES_PER_LINK + BASELINE_BYTES
    if int(nnz) > MAX_NNZ:
        raise CCCRefusal(
            "too_large",
            f"This signal's spot-by-spot matrix stores {int(nnz):,} links; the explorer reads at most {MAX_NNZ:,} "
            "from one matrix, whatever the memory allowed.",
            "SOG_VIZ_MEMORY_BYTES",
        )
    if need > memory:
        raise CCCRefusal(
            "too_large",
            f"Reading this signal's spot-by-spot matrix ({int(nnz):,} links) needs about {need / 2**30:,.1f} GiB, "
            f"more than the reader's {memory / 2**30:,.1f} GiB (SOG_VIZ_MEMORY_BYTES). Raising SOG_VIZ_MEMORY_BYTES "
            "allows it.",
            "SOG_VIZ_MEMORY_BYTES",
        )


def read_obsp(h5: H5AD, key: str, limits: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """``obsp[key]`` as CSR ``(indptr, indices, data, transposed)`` (h5lite's :meth:`~H5AD.obsp_csr`), preflighted."""
    info = next((k for k in h5.obsp_keys() if k.key == key), None)
    if info is None:
        raise CCCRefusal("bad_request", "That signal has no spot-by-spot matrix in this file.")
    preflight(info.nnz, limits)
    return h5.obsp_csr(key, max_nnz=MAX_NNZ)


def commot_dbs(h5: H5AD) -> list[str]:
    """The databases COMMOT wrote sums for (``obsm['commot-<db>-sum-sender|receiver']``), sorted."""
    return sorted({m.group("db") for k in h5.obsm_keys() if (m := _COMMOT_SUM_RE.match(k.key))})


def rows_mask(rows: np.ndarray | None, n: int) -> np.ndarray | None:
    """A boolean mask of the brushed rows over ``n`` spots, or ``None`` for all of them. Rows past ``n`` are refused."""
    if rows is None:
        return None
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= int(n)):
        raise CCCRefusal("bad_request", f"A brushed row is not one of this file's {int(n):,} spots.")
    mask = np.zeros(int(n), dtype=bool)
    mask[rows] = True
    return mask


__all__ = [
    "BASELINE_BYTES",
    "BYTES_PER_LINK",
    "commot_dbs",
    "copy_of",
    "declared_frames",
    "frame_coords",
    "group_codes",
    "obs_groups",
    "preflight",
    "read_obsp",
    "rows_mask",
    "section_column",
    "stacking_column",
    "um_per_unit",
    "units_text",
]
