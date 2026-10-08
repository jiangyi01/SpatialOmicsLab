"""Where a communication result's spots are: in 2D, in the aligned 3D frame, or one section of a per-section set.

The explorer learns a result's mode from the result's own files, never from ``sog_run_provenance.json`` (which is not a
registered record) nor from ``commot_block.json`` (a ``.json`` report the run harvest does not promote):

* **3d** -- ``obsm['spatial']`` holds three columns (a COMMOT block writes the block's micrometre coordinates there and
  declares them in ``uns['spatial_3d']['frames']['spatial']``);
* **per-section-2d** -- the bundle is a per-section set: one run per ``section_<label>/`` folder
  (``detect.bundle_of``), or a long table with a leading ``section`` column;
* **2d** -- one plane.

"Cross-section" is told by the section LABELS (``uns['spatial_3d']['slice_key']``, else an obs column named like a
section), never by assuming a column is depth: Zhuang-ABCA-1 is cut coronally, so its stacking axis is CCF x. A
section's depth (``z_um``) is the median of the stacking axis the frame's ``axis_map`` names, in micrometres when the
frame's units say how many. A frame whose ``z_source`` is ``rank_index`` is a **legacy** result: its z is an order,
not a distance, and no cross-section link of it is offered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from . import h5 as h5x
from .errors import CCCRefusal
from .facets import LINKS, MODES

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialomicsgym.viz.h5lite import H5AD

LEGACY_SENTENCE = "legacy result: z is a rank index, cross-section links are not distances"
#: A three-column ``spatial`` that declares no frame (nor copies a declared one): read like a legacy result.
UNDECLARED_SENTENCE = (
    "legacy result: this file's three-column `spatial` declares no frame; cross-section links are not distances"
)
PER_SECTION_NONE = "a per-section 2D result has no cross-section links by construction"
ZERO_CROSS = (
    "There is no inferred cross-section communication at this threshold: no link of this signal joins two sections."
)
INFERRED = (
    "Links between sections are inferred cross-section communication: computed in the aligned 3D frame (z from "
    "{z_source}), not measured."
)
NO_SECTIONS_3D = (
    "This 3D result names no section column (uns['spatial_3d']['slice_key'] or an obs column such as 'section'), "
    "so which links cross a section cannot be told."
)


@dataclass
class Geometry:
    """One h5ad's positions and what they are. ``codes``/``labels`` are its section column (``None`` without one);
    ``sections`` the sections with at least one spot, ``[{label, z_um}]`` in depth order (label order when no depth
    is known)."""

    coords: np.ndarray
    units: dict[str, str | None]
    z_source: str | None
    dims: int
    mode: str
    legacy: bool
    stack: int
    stack_why: str
    codes: np.ndarray | None
    labels: list[str] | None
    section_key: str | None
    sections: list[dict[str, Any]] = field(default_factory=list)

    copy_of: str | None = None
    #: A three-column ``spatial`` with no ``uns['spatial_3d'].frames.spatial`` and no declared frame it copies.
    undeclared: bool = False

    @property
    def not_distances(self) -> str | None:
        """Why this 3D result's cross-section links are not distances (the sentence), or ``None`` when they are."""
        if self.dims != 3:
            return None
        if self.legacy:
            return LEGACY_SENTENCE
        if self.undeclared:
            return UNDECLARED_SENTENCE
        return None

    @property
    def frame(self) -> dict[str, Any]:
        out = {"key": "spatial", "units": h5x.units_text(self.units), "dims": self.dims, "z_source": self.z_source}
        if self.copy_of is not None:
            # The declared frame ``spatial`` copies (a COMMOT block's CCF): what the frame is, where "spatial" says
            # only where it is stored.
            out["copy_of"] = self.copy_of
        return out

    def code_of(self, label: str) -> int:
        if self.labels is None or label not in self.labels:
            known = ", ".join(s["label"] for s in self.sections[:12]) or "none"
            raise CCCRefusal("bad_request", f"The request's section is not one of this result's ({known}).")
        return self.labels.index(label)


def _z_factor(units: Mapping[str, str | None], column: int) -> float | None:
    """Micrometres per unit of ``column`` (0 and 1 are xy, 2 is z)."""
    return h5x.um_per_unit(units.get("z") if column == 2 else units.get("xy"))


def _medians(values: np.ndarray, codes: np.ndarray, n_levels: int) -> list[float | None]:
    out: list[float | None] = []
    for c in range(n_levels):
        here = values[codes == c]
        here = here[np.isfinite(here)]
        out.append(float(np.median(here)) if here.size else None)
    return out


def of(h5: H5AD) -> Geometry:
    """The geometry of one h5ad (:func:`h5.frame_coords`, its section column, its sections at depth)."""
    coords, units, z_source = h5x.frame_coords(h5)
    dims = int(coords.shape[1])
    stack, why = h5x.stacking_column(h5, dims)
    found = h5x.section_column(h5)
    codes, labels, key = found if found is not None else (None, None, None)
    geometry = Geometry(
        coords=coords,
        units=units,
        z_source=z_source,
        dims=dims,
        mode="3d" if dims == 3 else "2d",
        legacy=dims == 3 and z_source == "rank_index",
        stack=stack,
        stack_why=why,
        codes=codes,
        labels=labels,
        section_key=key,
        copy_of=h5x.copy_of(h5, coords) if dims == 3 else None,
    )
    geometry.undeclared = dims == 3 and "spatial" not in h5x.declared_frames(h5) and geometry.copy_of is None
    if codes is not None and labels is not None:
        present = np.bincount(codes[codes >= 0], minlength=len(labels)) if codes.size else np.zeros(len(labels))
        depth: list[float | None] = [None] * len(labels)
        factor = _z_factor(units, stack) if dims == 3 and not geometry.legacy else None
        if factor is not None:
            depth = [None if m is None else m * factor for m in _medians(coords[:, stack], codes, len(labels))]
        sections = [{"label": labels[c], "z_um": depth[c]} for c in range(len(labels)) if present[c] > 0]
        if all(s["z_um"] is not None for s in sections):
            sections.sort(key=lambda s: s["z_um"])
        geometry.sections = sections
    return geometry


def depth_of_section(h5: H5AD) -> float | None:
    """For one section's 2D file of a per-section set: the depth of its spots in micrometres, from a declared 3D frame
    the file still carries (the median of that frame's stacking axis), or ``None`` when it carries none with units."""
    frames = h5x.declared_frames(h5)
    for key, declared in frames.items():
        if key == "spatial":
            continue
        info = h5.obsm_key(key)
        if info is None or info.kind != "array" or info.n_cols != 3 or declared.get("z_source") == "rank_index":
            continue
        stack = 2
        axis_map = declared.get("axis_map")
        if isinstance(axis_map, dict):
            for axis, text in axis_map.items():
                if str(axis).strip().lower() in ("x", "y", "z") and "stack" in str(text).lower():
                    stack = "xyz".index(str(axis).strip().lower())
                    break
        unit = declared.get("z_units") if stack == 2 else declared.get("xy_units")
        factor = h5x.um_per_unit(unit if isinstance(unit, str) else None)
        if factor is None:
            continue
        values = np.asarray(h5.obsm_column(key, stack), dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            return float(np.median(values)) * factor
    return None


def _stack_of(declared: Mapping[str, Any]) -> int:
    axis_map = declared.get("axis_map")
    if isinstance(axis_map, dict):
        for axis, text in axis_map.items():
            if str(axis).strip().lower() in ("x", "y", "z") and "stack" in str(text).lower():
                return "xyz".index(str(axis).strip().lower())
    return 2


#: Edges a graph's mode is checked on: their stored distances against each frame's own.
_GRAPH_SAMPLE = 2000


def graph_mode(h5: H5AD, max_nnz: int) -> dict[str, Any] | None:
    """What a squidpy result's spatial graph (``obsp['spatial_connectivities']``/``['spatial_distances']``) was built
    on, from the graph itself -- the h5ad keeps no record of it: ``{mode, frame, sections, n_cross_section_edges,
    cross_section_edge_fraction}``, or ``None`` when the file holds no graph.

    An edge joining two sections can only be a 3D graph's (a per-section graph is built section by section); with
    none, the stored distances of a sample of edges say which frame they were measured in -- a declared 3D frame in
    micrometres (``3d``) or the two-column ``spatial`` (``per-section-2d`` across sections, ``2d`` on one).
    """
    keys = {k.key: k for k in h5.obsp_keys()}
    if "spatial_connectivities" not in keys:
        return None
    found = h5x.section_column(h5)
    indptr, indices, _data, _t = h5.obsp_csr("spatial_connectivities", max_nnz=max_nnz)
    rows = np.repeat(np.arange(indptr.size - 1, dtype=np.int64), np.diff(np.asarray(indptr, dtype=np.int64)))
    cols = np.asarray(indices, dtype=np.int64)
    n_edges = int(cols.size)
    codes = found[0] if found is not None else None
    n_levels = int(np.unique(codes[codes >= 0]).size) if codes is not None else 0
    n_cross = int(np.count_nonzero(cross_section_edges(rows, cols, codes))) if codes is not None else None
    frame_key = _measured_in(h5, keys, max_nnz)
    if n_cross:
        mode = "3d"
    elif frame_key is not None and frame_key != "spatial":
        mode = "3d"
    else:
        mode = "per-section-2d" if n_levels >= 2 else "2d"
    out: dict[str, Any] = {"mode": mode, "frame": None, "sections": []}
    if codes is not None and n_levels >= 2:
        out["n_cross_section_edges"] = n_cross
        out["cross_section_edge_fraction"] = (n_cross / n_edges) if n_edges else 0.0
    if mode == "3d" and frame_key not in (None, "spatial"):
        out["frame"], out["sections"] = _frame_and_sections(h5, frame_key, found)
    elif found is not None and n_levels >= 2:
        out["sections"] = _flat_sections(found)
    return out


def _flat_sections(found: tuple[np.ndarray, list[str], str]) -> list[dict[str, Any]]:
    present = np.bincount(found[0][found[0] >= 0], minlength=len(found[1]))
    return [{"label": found[1][c], "z_um": None} for c in range(len(found[1])) if present[c] > 0]


def _frame_and_sections(
    h5: H5AD, frame_key: str, found: tuple[np.ndarray, list[str], str] | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A declared 3D frame as describe names it (``{key, units, dims, z_source}``) and its sections ordered along the
    stacking axis, each at its median depth in micrometres (``[]`` without a section column or known units)."""
    declared = h5x.declared_frames(h5).get(frame_key, {})
    stack = _stack_of(declared)
    unit = declared.get("z_units") if stack == 2 else declared.get("xy_units")
    factor = h5x.um_per_unit(unit if isinstance(unit, str) else None)
    xy = declared.get("xy_units")
    z = declared.get("z_units")
    units = {"xy": xy if isinstance(xy, str) else None, "z": z if isinstance(z, str) else None}
    frame = {"key": frame_key, "units": h5x.units_text(units), "dims": 3, "z_source": declared.get("z_source")}
    sections: list[dict[str, Any]] = []
    if found is not None and factor is not None:
        values = np.asarray(h5.obsm_column(frame_key, stack), dtype=np.float64)
        medians = _medians(values, found[0], len(found[1]))
        present = np.bincount(found[0][found[0] >= 0], minlength=len(found[1]))
        sections = [
            {"label": found[1][c], "z_um": None if medians[c] is None else medians[c] * factor}
            for c in range(len(found[1]))
            if present[c] > 0
        ]
        sections.sort(key=lambda s: (s["z_um"] is None, s["z_um"] or 0.0))
    return frame, sections


#: ``uns`` key where squidpy's worker records the run's ``{mode, frame}`` (``frame`` = its Frame.to_dict(), JSON text).
RECORD_UNS = "sog_ccc"


def recorded_mode(h5: H5AD) -> dict[str, Any] | None:
    """The mode and frame a worker recorded in the result's ``uns['sog_ccc']`` -- for a result with no graph to read
    them from (squidpy co-occurrence, Ripley): ``{mode, frame, sections}`` as :func:`graph_mode` answers, or ``None``
    when nothing usable is recorded. A 3D record names its frame only when that frame is declared in the file."""
    import json

    record = h5.uns(RECORD_UNS)
    if not isinstance(record, dict) or record.get("mode") not in MODES:
        return None
    mode = str(record["mode"])
    try:
        frame = json.loads(record["frame"]) if isinstance(record.get("frame"), str) else {}
    except ValueError:
        frame = {}
    key = frame.get("coords_key") if isinstance(frame, dict) else None
    found = h5x.section_column(h5)
    out: dict[str, Any] = {"mode": mode, "frame": None, "sections": []}
    if mode == "3d" and isinstance(key, str) and key != "spatial" and key in h5x.declared_frames(h5):
        out["frame"], out["sections"] = _frame_and_sections(h5, key, found)
    elif mode == "per-section-2d" and found is not None:
        out["sections"] = _flat_sections(found)
    return out


def cross_section_edges(rows: np.ndarray, cols: np.ndarray, codes: np.ndarray) -> np.ndarray:
    a, b = codes[rows], codes[cols]
    return (a >= 0) & (b >= 0) & (a != b)


def _measured_in(h5: H5AD, keys: Mapping[str, Any], max_nnz: int) -> str | None:
    """Which obsm frame a graph's stored distances were measured in: ``spatial``, a declared three-column frame
    (converted to micrometres, as the workers hand squidpy its coordinates), or ``None`` when none matches."""
    if "spatial_distances" not in keys:
        return None
    indptr, indices, data, _t = h5.obsp_csr("spatial_distances", max_nnz=max_nnz)
    rows = np.repeat(np.arange(indptr.size - 1, dtype=np.int64), np.diff(np.asarray(indptr, dtype=np.int64)))
    cols = np.asarray(indices, dtype=np.int64)
    data = np.asarray(data, dtype=np.float64)
    live = np.flatnonzero(data > 0)[:_GRAPH_SAMPLE]
    if not live.size:
        return None
    rows, cols, stored = rows[live], cols[live], data[live]
    frames = h5x.declared_frames(h5)
    candidates: list[tuple[str, float]] = [("spatial", 1.0)]
    for key, declared in frames.items():
        info = h5.obsm_key(key)
        if key == "spatial" or info is None or info.kind != "array" or info.n_cols != 3:
            continue
        factor = h5x.um_per_unit(declared.get("xy_units") if isinstance(declared.get("xy_units"), str) else None)
        if factor is not None:
            candidates.append((key, factor))
    for key, factor in candidates:
        info = h5.obsm_key(key)
        if info is None or info.kind not in ("array", "dataframe"):
            continue
        coords = np.asarray(h5.obsm(key), dtype=np.float64)
        if key == "spatial":
            xy = frames.get("spatial", {}).get("xy_units")
            factor = h5x.um_per_unit(xy if isinstance(xy, str) else None) or 1.0
        step = (coords[cols] - coords[rows]) * factor
        measured = np.sqrt(np.einsum("ij,ij->i", step, step))
        if np.allclose(measured, stored, rtol=1e-4, atol=1e-6 * max(1.0, float(np.abs(stored).max()))):
            return key
    return None


def links_param(params: Mapping[str, Any]) -> str:
    links = params.get("links", "all")
    if links not in LINKS:
        raise CCCRefusal("bad_request", "The request's links must be 'all' or 'cross-section'.")
    return str(links)


def refuse_cross_section(geometry: Geometry | None, per_section: bool) -> None:
    """Refuse ``links: "cross-section"`` where there are none to tell: a per-section set, a legacy rank-z result,
    a 2D result, a 3D result with no section column."""
    if per_section:
        raise CCCRefusal("unsupported", f"No cross-section links: {PER_SECTION_NONE}.")
    if geometry is None or geometry.dims != 3:
        raise CCCRefusal(
            "unsupported", "No cross-section links: this result is two-dimensional, so no link joins two sections."
        )
    if geometry.not_distances:
        raise CCCRefusal("unsupported", f"No cross-section links: {geometry.not_distances}.")
    if geometry.codes is None:
        raise CCCRefusal("unsupported", NO_SECTIONS_3D)


def describe_fields(geometry: Geometry) -> tuple[dict[str, Any], list[str]]:
    """``({mode, sections, frame, legacy_rank_z, undeclared_3d}, warnings)`` for a single h5ad's describe answer."""
    warnings: list[str] = []
    if geometry.dims == 3:
        if geometry.not_distances:
            warnings.append(f"This is a {geometry.not_distances}.")
        else:
            warnings.append(INFERRED.format(z_source=geometry.z_source or "an undeclared source"))
        if geometry.units.get("xy") is None and not geometry.undeclared:
            warnings.append(
                "This 3D frame declares no units (uns['spatial_3d']['frames']['spatial']), so no depth is given in "
                "micrometres."
            )
        if geometry.codes is None:
            warnings.append(NO_SECTIONS_3D)
    fields = {
        "mode": geometry.mode,
        "sections": geometry.sections if geometry.dims == 3 else [],
        "frame": geometry.frame,
        "legacy_rank_z": geometry.legacy,
        "undeclared_3d": geometry.undeclared,
    }
    return fields, warnings


__all__ = [
    "INFERRED",
    "LEGACY_SENTENCE",
    "LINKS",
    "MODES",
    "NO_SECTIONS_3D",
    "PER_SECTION_NONE",
    "UNDECLARED_SENTENCE",
    "ZERO_CROSS",
    "Geometry",
    "depth_of_section",
    "cross_section_edges",
    "describe_fields",
    "graph_mode",
    "links_param",
    "of",
    "refuse_cross_section",
]
