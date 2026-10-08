"""Where a 3D coordinate lives, what is known about it, and who put it there.

Twelve alignment portals ship in this repository and no two agree about where they leave their
answer. PASTE overwrites ``obsm['spatial']`` in place, so the coordinates it was given are not
recoverable from what it wrote. GPSA, SPACEL and SPIRAL write ``obsm['spatial_aligned']``; moscot
writes ``moscot_spatial_warp`` (onto the named reference batch, or else the first batch it meets). ST-GEARS writes
``st_gears_xyz``, whose third column is the slice *rank*. SLAT writes a matching table and STACKer
a warped image, neither of which is a coordinate at all. A Phase-3 tool that wants "the aligned
coordinates" therefore has no key to open, and that -- not the absence of aligners -- is what
stops a multi-slice study reaching its downstream analysis.

This module is the one place that is decided.

**``obsm['spatial']`` stays two columns and nothing here ever writes it.** That is forced, not
stylistic: PASTE asserts ``X.shape[1] == 2`` (``paste/visualization.py:172``), and a recorded run
died on exactly that when a three-column ``spatial`` reached it. Putting z in ``spatial`` breaks
at least one shipped aligner on contact, so z lives in its own keys and the original is preserved
beside them, byte for byte.

**The two-column rule is canonical, and one shipped aligner needs a third column.** ST-GEARS indexes
``obsm['spatial'][:, 2]`` unconditionally (``st_gears/recons.py:150``, ``granularity_adjusting.py:102``),
so the package itself *requires* a three-column ``spatial`` -- the exact shape PASTE asserts against.
This contract sides with PASTE, because two columns is what every other tool and every converter in
the repository writes; the ST-GEARS worker therefore hands the package a working copy whose third
column is the slice ordinal and writes the two-column original back (``params.z_column_added``), so
both aligners take the same file. ``st_gears_xyz`` appears nowhere in the upstream package: it is the
worker's own copy of ``spatial_3d_aligned``. The adapter reads ``spatial_elas_reuse`` /
``spatial_elas`` / ``spatial_rigid`` instead.

**A frame has a role, and the roles are kept apart.** ``spatial_3d_raw`` is the stack before
alignment, ``spatial_3d_aligned`` after, and ``spatial_3d_<source>`` is somebody else's
registration -- Zhuang-ABCA-1 arrives already placed in the Allen CCF. That last one is deliberately
*not* folded into ``aligned``: if it were, Phase 2 would take credit for the Allen Institute's work
and then compare a frame against itself and report that nothing moved.

**Every frame carries how its z was obtained, and "unknown" is an answer.** A z that is a slice
ordinal and a z that is a measured section thickness produce different 3D neighbour graphs from
the same array -- measured cross-section edge fractions of 0.56 and 0.00 for the same slide stack
-- so a consumer must be able to ask which one it has and refuse when the answer is neither. There
is no default z spacing anywhere in this package, and no function invents one.

Nothing heavy is imported at module scope: the capability registry that reads this is read by a
portal, and the portal must stay importable where anndata is absent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

SCHEMA = "sog.spatial3d/1"

#: The untouched in-plane coordinates. Read by everything, written by nothing in this package.
ORIGINAL_KEY = "spatial"

#: The stack as it was before any transform was fitted -- ``spatial[:, :2]`` plus a declared z.
RAW_FRAME = "spatial_3d_raw"

#: The stack after an aligner ran.
ALIGNED_FRAME = "spatial_3d_aligned"

#: An external registration is ``spatial_3d_<source>``; ``spatial_3d_ccf`` is the worked example.
EXTERNAL_PREFIX = "spatial_3d_"

#: Accepted as role ``aligned`` when it is the only three-column key present. The STARmap and
#: seqFISH converters have written this name since before this contract existed
#: (``spatial_data_converter.py:1007``), and a reader that rejected it would refuse files the
#: repository itself produces.
LEGACY_FRAME = "spatial_3d"

ROLES = ("raw", "aligned", "external")

#: What :func:`find_frames` calls a three-column key that neither declares a role nor carries one of
#: the contract's names. Reported, never written: :func:`write_frame` still accepts only :data:`ROLES`.
UNDECLARED_ROLE = "undeclared"

#: ``unknown`` is a verdict. A metric whose threshold is expressed in physical units must refuse
#: rather than silently compare a millimetre to a pixel.
UNITS = ("um", "mm", "px", "array_index", "unknown")

#: How the third column came to exist. ``rank_index`` is the honest label for ST-GEARS's output:
#: an ordinal wearing a coordinate's clothes. ``rank_index_replaced_by_declared_z`` records that
#: an adapter substituted a physical spacing the caller supplied. ``atlas_registration`` is a z
#: somebody else placed by registering to an atlas -- Zhuang-ABCA-1's Allen CCF coordinates -- a
#: physical distance, but neither measured by us nor read from section metadata.
Z_SOURCES = (
    "measured",
    "section_metadata",
    "asked_user",
    "rank_index",
    "rank_index_replaced_by_declared_z",
    "unknown",
    "atlas_registration",
)

#: Sixteen hex characters, not a sha256. The portal's redactor masks any hex run of 32 or more, so
#: a full digest reaches the reader as ``[redacted]`` and stops being checkable. Same construction
#: and same reason as ``viz.profile.fingerprint``.
DIGEST_CHARS = 16


class ContractError(Exception):
    """The 3D contract was asked for something the object cannot honestly provide."""


@dataclass(frozen=True)
class Refusal:
    """A named no.

    Returned rather than raised wherever the caller is enumerating -- an adapter table walking ten
    aligners wants nine frames and one stated reason, not an exception that ends the walk.
    """

    what: str
    reason: str
    remedy: str = ""

    def __str__(self) -> str:
        tail = f" {self.remedy}" if self.remedy else ""
        return f"{self.what}: {self.reason}.{tail}".strip()


@dataclass(frozen=True)
class Frame:
    """One three-column coordinate array and everything known about it."""

    key: str
    role: str
    xy_units: str = "unknown"
    z_units: str = "unknown"
    z_source: str = "unknown"
    z_spacing: float | None = None
    z_spacing_uniform: bool | None = None
    axis_map: dict[str, str] = field(default_factory=dict)
    xy_from: str = ORIGINAL_KEY
    created_by: str = ""
    created_at: str = ""
    # role == "aligned"
    source_frame: str = ""
    aligner: str = ""
    aligner_function: str = ""
    aligner_version: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    #: False for six of the ten shipped portals, which expose no seed at all. Recorded rather than
    #: omitted, so a reproducibility claim is not made on their behalf by silence.
    seed_exposed: bool = False
    reference_slice: str = ""
    transform_table: str = ""
    # role == "external"
    registered_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k == "key":
                continue
            if v in ("", {}, None) and k not in ("z_spacing", "z_spacing_uniform", "seed"):
                continue
            out[k] = v
        return out


def coordinate_digest(arr: Any) -> str:
    """A short, stable fingerprint of a coordinate array.

    Over the raw little-endian float64 bytes and the shape, so a transposed or reshaped array of
    the same numbers does not collide. Truncated to :data:`DIGEST_CHARS` for the redactor reason
    given above.
    """
    import numpy as np

    a = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:DIGEST_CHARS]


def _block(adata: Any) -> dict[str, Any]:
    raw = adata.uns.get("spatial_3d") if hasattr(adata, "uns") else None
    block = dict(raw) if isinstance(raw, dict) else {}
    # anndata reads a list of strings back from disk as a numpy array, and the truth value of an
    # array is an error: validate() raised on every file written per the contract and read back,
    # the very object the inspector exists to check (hunt 2026-09-30, u21-3d-2).
    order = block.get("slice_order")
    if order is not None and not isinstance(order, (str, bytes)) and hasattr(order, "__iter__"):
        block["slice_order"] = [str(s) for s in order]
    return block


def _frames_meta(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    frames = block.get("frames")
    return dict(frames) if isinstance(frames, dict) else {}


def three_column_keys(adata: Any) -> list[str]:
    """Every obsm key holding exactly three columns, in a stable order."""
    import numpy as np

    out = []
    for k in list(getattr(adata, "obsm", {}) or {}):
        try:
            v = np.asarray(adata.obsm[k])
        except Exception:
            continue
        if v.ndim == 2 and v.shape[1] == 3:
            out.append(str(k))
    return sorted(out)


def find_frames(adata: Any) -> dict[str, Frame]:
    """Map obsm key -> :class:`Frame` for every three-column key this object carries.

    The declared metadata in ``uns['spatial_3d']['frames']`` wins where it exists. A three-column
    key with no declaration still yields a Frame, with its role inferred from the name and every
    unit left at ``unknown`` -- an undeclared frame is a real thing that needs reporting, not an
    error, because the two converters and several aligners predate this contract.
    """
    meta = _frames_meta(_block(adata))
    keys = three_column_keys(adata)
    out: dict[str, Frame] = {}
    for key in keys:
        declared = meta.get(key)
        if isinstance(declared, dict):
            fields = {k: v for k, v in declared.items() if k in Frame.__dataclass_fields__}
            out[key] = Frame(key=key, **{**{"role": _infer_role(key, keys)}, **fields})
        else:
            out[key] = Frame(key=key, role=_infer_role(key, keys))
    return out


def _infer_role(key: str, all_keys: list[str]) -> str:
    if key == RAW_FRAME:
        return "raw"
    if key == ALIGNED_FRAME:
        return "aligned"
    if key == LEGACY_FRAME:
        # Only a stand-in for "aligned" when nothing else claims that role; with a real
        # spatial_3d_aligned beside it, this is somebody's external registration.
        return "aligned" if ALIGNED_FRAME not in all_keys else "external"
    if key.startswith(EXTERNAL_PREFIX):
        return "external"
    # Any other three-column key -- ST-GEARS's spatial_elas_reuse, GPSA's 3-D spatial_aligned, a
    # 3-component X_umap -- is not somebody else's registration just because it has three columns,
    # and read_frame(role='external') handed an embedding back as one (hunt 2026-09-30, u21-3d-15).
    return UNDECLARED_ROLE


def read_frame(adata: Any, role: str = "aligned", *, key: str = "") -> tuple[Any, Frame]:
    """The coordinates a consumer should use, and everything known about them.

    This is the only supported way for a downstream tool to obtain 3D coordinates. Hardcoding an
    obsm key is what the ten disagreeing aligners already did.

    Raises :class:`ContractError` naming the keys that *are* present, because "no 3D coordinates"
    and "3D coordinates under a name you did not expect" need different fixes from the caller.
    """
    import numpy as np

    frames = find_frames(adata)
    if key:
        if key not in frames:
            raise ContractError(
                f"obsm[{key!r}] is not a three-column coordinate key. Present: {sorted(frames) or 'none'}"
            )
        chosen = frames[key]
    else:
        if role not in ROLES:
            raise ContractError(f"role must be one of {ROLES}, not {role!r}")
        candidates = [f for f in frames.values() if f.role == role]
        if not candidates:
            raise ContractError(
                f"no frame with role {role!r}. Three-column keys present: {sorted(frames) or 'none'}. "
                # adopt_aligned_coordinates and assemble_3d_stack exist nowhere in the repository;
                # name the two calls that do (hunt 2026-09-30, u21-3d-9).
                f"Read an aligner's output with spatialomicsgym.spatial3d.adapters.read_aligned and "
                f"write it with contract.write_frame (role 'aligned'), or write a raw stack with "
                f"contract.write_frame(adata, coords, Frame(key={RAW_FRAME!r}, role='raw', ...))."
            )
        # Deterministic when an object carries two externals: the canonical names sort first.
        chosen = sorted(candidates, key=lambda f: (f.key != ALIGNED_FRAME, f.key != RAW_FRAME, f.key))[0]

    coords = np.asarray(adata.obsm[chosen.key], dtype="float64")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ContractError(f"obsm[{chosen.key!r}] has shape {coords.shape}, expected (n_obs, 3)")
    return coords, chosen


def original_coordinates(adata: Any) -> Any:
    """``obsm['spatial']``, as float64, or a :class:`ContractError` saying it is absent."""
    import numpy as np

    if ORIGINAL_KEY not in (getattr(adata, "obsm", {}) or {}):
        raise ContractError(
            f"obsm[{ORIGINAL_KEY!r}] is absent, so there is no original to preserve and no in-plane "
            f"coordinate to build a stack from."
        )
    return np.asarray(adata.obsm[ORIGINAL_KEY], dtype="float64")


def assert_original_intact(adata: Any, expected_digest: str) -> None:
    """Refuse to continue if ``obsm['spatial']`` has moved.

    Called by every writer in this package before and after it touches the object. The invariant
    it defends is the one Phase 2 is specified around, and it is cheap enough to check twice.
    """
    actual = coordinate_digest(original_coordinates(adata))
    if actual != expected_digest:
        raise ContractError(
            f"obsm[{ORIGINAL_KEY!r}] changed during this operation "
            f"(digest {expected_digest} -> {actual}). The original coordinates must survive "
            f"alignment; refusing to write a frame that would record otherwise."
        )


def write_frame(
    adata: Any,
    coords: Any,
    frame: Frame,
    *,
    slice_key: str = "",
    slice_order: list[str] | None = None,
    z_key: str = "",
    coordinate_coverage: float | None = None,
) -> None:
    """Attach a 3D frame and its provenance, having first proved the original is untouched.

    ``frame.z_source`` must be one of :data:`Z_SOURCES` and ``frame.role`` one of :data:`ROLES`;
    both are rejected rather than coerced, because a frame whose provenance is a typo is worse
    than one with none -- it reads as declared.
    """
    import numpy as np

    if frame.role not in ROLES:
        raise ContractError(f"role must be one of {ROLES}, not {frame.role!r}")
    if frame.z_source not in Z_SOURCES:
        raise ContractError(f"z_source must be one of {Z_SOURCES}, not {frame.z_source!r}")
    if frame.xy_units not in UNITS or frame.z_units not in UNITS:
        raise ContractError(f"units must be drawn from {UNITS}")

    if frame.key == ORIGINAL_KEY:
        _declare_original(adata, coords, frame)
        # The stack's keys belong to the object, not to a frame: a units declaration is handed them too and keeps
        # them (dropped silently before, so a reader guessed the section column the caller had named).
        _record_stack_keys(adata.uns["spatial_3d"], slice_key, slice_order, z_key, coordinate_coverage)
        return

    arr = np.asarray(coords, dtype="float64")
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ContractError(f"a frame is (n_obs, 3); got {arr.shape}")
    if arr.shape[0] != adata.n_obs:
        raise ContractError(f"frame has {arr.shape[0]} rows, object has {adata.n_obs}")

    # A frame keyed 'spatial' never reaches obsm: it is a units declaration (_declare_original, above) or
    # refused there (hunt 2026-09-30, u21-3d-14).
    digest = coordinate_digest(original_coordinates(adata))
    had_previous = frame.key in adata.obsm
    previous = adata.obsm[frame.key] if had_previous else None
    adata.obsm[frame.key] = arr
    try:
        assert_original_intact(adata, digest)
    except ContractError:
        if had_previous:
            adata.obsm[frame.key] = previous
        else:
            del adata.obsm[frame.key]
        raise

    block = _block(adata)
    block.setdefault("schema", SCHEMA)
    block["original_key"] = ORIGINAL_KEY
    block["original_digest"] = digest
    _record_stack_keys(block, slice_key, slice_order, z_key, coordinate_coverage)
    frames = _frames_meta(block)
    frames[frame.key] = frame.to_dict()
    block["frames"] = frames
    adata.uns["spatial_3d"] = block


def _record_stack_keys(
    block: dict, slice_key: str, slice_order: list[str] | None, z_key: str, coordinate_coverage: float | None
) -> None:
    """The section column, its order, the z column and the coverage a write was handed, into ``block``."""
    if slice_key:
        block["slice_key"] = slice_key
    if slice_order is not None:
        # Explicit and ordered. A reader never sorts this: for Zhuang-ABCA-1 four sections share
        # z == 0.0 exactly, so any sort silently invents an adjacency that is not in the tissue.
        block["slice_order"] = [str(s) for s in slice_order]
    if z_key:
        block["z_key"] = z_key
    if coordinate_coverage is not None:
        block["coordinate_coverage"] = float(coordinate_coverage)


def _declare_original(adata: Any, coords: Any, frame: Frame) -> None:
    """Record the units of the two-column ``obsm['spatial']`` -- a declaration, never a write.

    A 2D tool that takes a threshold in micrometres must know what one unit of ``spatial`` is, and only a
    declaration can say. ``coords`` must be the original itself (same digest), so the call cannot move it;
    the frame is ``role 'raw'`` with ``z_source`` and ``z_units`` 'unknown', because two columns have no z.
    """
    import numpy as np

    refusal = (
        f"a frame cannot be written to obsm[{ORIGINAL_KEY!r}]: that key holds the original two-column "
        f"coordinates and nothing in this package writes it. Use {RAW_FRAME!r} or {ALIGNED_FRAME!r}; "
        f"only a declaration of its units (the original's own two columns, role 'raw') is accepted."
    )
    arr = np.asarray(coords, dtype="float64")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ContractError(refusal)
    digest = coordinate_digest(original_coordinates(adata))
    if coordinate_digest(arr) != digest:
        raise ContractError(refusal)
    if frame.role != "raw" or frame.z_source != "unknown" or frame.z_units != "unknown":
        raise ContractError(
            f"obsm[{ORIGINAL_KEY!r}] has two columns and no z: declare it with role 'raw', z_source 'unknown' and "
            f"z_units 'unknown' (got role {frame.role!r}, z_source {frame.z_source!r}, z_units {frame.z_units!r})."
        )
    block = _block(adata)
    block.setdefault("schema", SCHEMA)
    frames = _frames_meta(block)
    declared = {"role": "raw", "xy_units": frame.xy_units, "z_units": "unknown", "z_source": "unknown"}
    declared.update({k: v for k, v in (("created_by", frame.created_by), ("created_at", frame.created_at)) if v})
    frames[ORIGINAL_KEY] = declared
    block["frames"] = frames
    adata.uns["spatial_3d"] = block


def _declares_two_column_original(adata: Any, key: str) -> bool:
    """True for ``frames['spatial']`` over a two-column ``obsm['spatial']``: a units declaration, not a frame."""
    import numpy as np

    obsm = getattr(adata, "obsm", {}) or {}
    if key != ORIGINAL_KEY or ORIGINAL_KEY not in obsm:
        return False
    arr = np.asarray(obsm[ORIGINAL_KEY])
    return arr.ndim == 2 and arr.shape[1] == 2


def validate(adata: Any) -> list[str]:
    """Everything wrong or unstated about this object's 3D coordinates, as sentences.

    An empty list means the contract holds. Returned rather than raised: a diagnosis reports on
    what it found, and half a contract is still worth describing.
    """
    import numpy as np

    problems: list[str] = []
    obsm = getattr(adata, "obsm", {}) or {}

    if ORIGINAL_KEY in obsm:
        orig = np.asarray(obsm[ORIGINAL_KEY])
        if orig.ndim == 2 and orig.shape[1] != 2:
            problems.append(
                f"obsm[{ORIGINAL_KEY!r}] has {orig.shape[1]} columns. It must stay two: PASTE "
                f"asserts it, and a three-column spatial has already killed a real run. Move the "
                f"third column into {RAW_FRAME!r}."
            )
    else:
        problems.append(f"obsm[{ORIGINAL_KEY!r}] is absent; there is no original to preserve.")

    frames = find_frames(adata)
    if not frames:
        problems.append("no three-column coordinate key is present, so there is no 3D frame.")

    block = _block(adata)
    declared = set(_frames_meta(block))
    for key in frames:
        if key not in declared:
            problems.append(
                f"obsm[{key!r}] is a three-column key with no entry in uns['spatial_3d']['frames'], "
                f"so its units, its z source and the tool that wrote it are all unknown."
            )
    for key in sorted(declared - set(frames)):
        if _declares_two_column_original(adata, key):
            continue
        problems.append(f"uns['spatial_3d']['frames'] declares {key!r}, which is not in obsm.")

    for key, f in frames.items():
        if f.z_source == "rank_index":
            problems.append(
                f"obsm[{key!r}] has z_source 'rank_index': its third column is a slice ordinal, "
                f"not a distance. A 3D neighbour graph built on it is not a 3D graph."
            )
        if f.role == "aligned" and not f.source_frame:
            problems.append(
                f"obsm[{key!r}] is an aligned frame with no source_frame, so there is nothing to "
                f"compare it against and no before/after is possible."
            )
        if f.z_units == "unknown" or f.xy_units == "unknown":
            problems.append(f"obsm[{key!r}] has undeclared units, so no metric in physical units can be checked.")
        if f.role not in ("external", UNDECLARED_ROLE) and not f.axis_map:
            problems.append(
                f"obsm[{key!r}] declares no axis_map. Zhuang-ABCA-1's two coordinate tables disagree "
                f"about which letter is the stacking axis, so this cannot be inferred from names."
            )

    order = block.get("slice_order")
    if frames and (order is None or len(order) == 0):
        problems.append("uns['spatial_3d']['slice_order'] is absent, so adjacency between sections is undefined.")

    return problems
