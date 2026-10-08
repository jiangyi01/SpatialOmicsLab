"""What a stack of serial sections is, per section, before anything is measured or moved.

Phase 1 opens with an inventory, and the inventory is where most of the ways a 3D study goes
wrong are still cheap to catch. Every fact here is read, never inferred, and where a fact cannot
be established this module says so rather than choosing a plausible value.

**Slice order is never inferred silently.** It comes from an explicit order the caller supplies,
or from a z column. With neither, ``slice_order`` is unknown and every adjacent-pair metric
downstream is refused with that reason -- because "adjacent" is the one input the whole diagnosis
rests on, and a guessed order produces a confident measurement of a relationship that does not
exist.

**Coincident z is a first-class fact, not a tie to break.** Zhuang-ABCA-1 has four sections at
exactly z = 0.0: the provider clamps negative anterior estimates to zero, so four physically
distinct planes share one coordinate. Sorted into an order they do not have, those four produce
adjacent-pair centroid offsets of 0.20 and local shifts of 0.08 -- they look badly misaligned
purely because they were sorted. They are reported as a coincident group and excluded from
pairwise geometry.

**Units are tri-state, and ``unknown`` forces a clause rather than a default.** Every threshold in
this package is a fraction of the tissue diagonal and so is unit-free, which is what lets a
diagnosis run at all on a slide whose units nobody recorded. But a z spacing, a neighbourhood
radius or a physical claim is not expressible without units, and saying so is the honest output.

**An external coordinate table is a first-class input.** The Zhuang h5ad carries expression and
nothing else -- ``obsm`` is empty and ``obs`` has one column. Its coordinates live in side-car
CSVs, and 31.7% of the cells in the matrix have no row in them. A profile that inner-joined and
then reported on the remaining 68% as though it were the dataset would be wrong in the direction
nobody checks, so ``coordinate_coverage`` is a headline fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

UNITS = ("um", "mm", "px", "array_index", "unknown")

#: Spans below this, in the coordinates' own units, are too small to be micrometres of tissue and
#: are read as millimetres. A mouse brain is ~13 mm and ~13000 um across, so the two are three
#: orders of magnitude apart and the inference is safe in a way most unit guesses are not.
MM_SPAN_CEILING = 100.0

#: Above this a span is micrometres or pixels; the two are not distinguishable from coordinates
#: alone, so the verdict stays 'unknown' rather than picking one.
UM_SPAN_FLOOR = 1000.0

#: Two sections whose z differ by less than this fraction of the median non-zero gap are treated
#: as occupying one plane. Not zero: a float z read from a CSV can differ in the last bit.
COINCIDENT_Z_TOLERANCE = 0.01


def coords_look_like_array_indices(coords: Any) -> bool:
    """Whole, small and non-negative: a Visium row/column lattice, not a position.

    This mirrors the rule in ``viz/profile.py`` (in ``_spatial_facts``, under the comment about a
    tissue map a hundred and seventy times too small) **verbatim and deliberately**, and
    ``test/test_spatial3d_the_unit_rule_is_the_one_the_viz_profiler_uses.py`` asserts the two
    still agree. Two different answers to "is this an array index" in one repository is how a
    slide gets drawn at the wrong scale in one place and the right scale in another.
    """
    import numpy as np

    a = np.asarray(coords, dtype=float)
    finite = a[np.isfinite(a).all(axis=1)]
    if not len(finite):
        return False
    try:
        whole = bool(np.allclose(finite, np.round(finite)))
        small = bool(finite.max() < 1000 and finite.min() >= 0)
        return bool(whole and small)
    except Exception:
        return False


@dataclass
class SliceProfile:
    """One section: how many cells, where they are, and how far apart."""

    name: str
    n_cells: int = 0
    z: float | None = None
    bbox: list[float] = field(default_factory=list)
    span: list[float] = field(default_factory=list)
    pitch: float = 0.0
    n_duplicate_coords: int = 0
    n_non_finite: int = 0
    n_at_origin: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class StackProfile:
    """The whole stack, and everything about it that could not be established."""

    slice_key: str = ""
    z_key: str = ""
    slice_order: list[str] = field(default_factory=list)
    slice_order_known: bool = False
    slice_order_from: str = "unknown"
    slices: list[SliceProfile] = field(default_factory=list)
    coincident_z_groups: list[list[str]] = field(default_factory=list)
    z_spacings: list[float] = field(default_factory=list)
    z_spacing_uniform: bool | None = None
    xy_units: str = "unknown"
    units_from: str = ""
    coordinate_coverage: float = 1.0
    obsm_keys: dict[str, list[int]] = field(default_factory=dict)
    three_column_keys: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def adjacent_pairs(self) -> list[tuple[str, str]]:
        """Pairs that are genuinely adjacent: consecutive in a known order, distinct in z.

        A coincident-z group contributes no internal pair, because there is no fact of the matter
        about which of its sections follows which.
        """
        if not self.slice_order_known:
            return []
        # Only a pair INSIDE one group is skipped. Two sections in different groups -- z = 0 and
        # z = 100 -- are distinct in z; skipping any pair whose ends were both in some group left a
        # stack of same-plane pairs (z = 0, 0, 100, 100, 200) with one pair of four
        # (hunt 2026-09-30, u21-3d-7 side note).
        group_of = {n: i for i, g in enumerate(self.coincident_z_groups) for n in g}
        out = []
        for a, b in zip(self.slice_order, self.slice_order[1:], strict=False):
            if a in group_of and group_of.get(a) == group_of.get(b):
                continue
            out.append((a, b))
        return out

    def summary(self) -> str:
        bits = [f"{len(self.slices)} sections"]
        if self.slice_order_known:
            bits.append(f"order from {self.slice_order_from}")
        else:
            bits.append("order UNKNOWN")
        bits.append(f"xy units {self.xy_units}")
        if self.z_spacings:
            import statistics

            bits.append(f"median z gap {statistics.median(self.z_spacings):g}")
        if self.coincident_z_groups:
            n = sum(len(g) for g in self.coincident_z_groups)
            bits.append(f"{n} sections share a z")
        if self.coordinate_coverage < 1.0:
            bits.append(f"coordinate coverage {self.coordinate_coverage:.1%}")
        return "; ".join(bits)


def infer_units(span: float, is_array_index: bool) -> tuple[str, str]:
    """A units verdict and the evidence for it. ``unknown`` is a real answer."""
    if is_array_index:
        return "array_index", "coordinates are whole, small and non-negative: an array lattice"
    if span <= 0:
        return "unknown", "the coordinates have no extent"
    if span < MM_SPAN_CEILING:
        return "mm", f"a span of {span:g} is far too small to be micrometres of tissue"
    if span >= UM_SPAN_FLOOR:
        return (
            "unknown",
            f"a span of {span:g} is micrometres or pixels; the two are not distinguishable from "
            f"coordinates alone, and guessing one would put a physical scale on a figure that "
            f"does not have one",
        )
    return "unknown", f"a span of {span:g} matches no unit this rule can name"


def profile_stack(
    coords: Any,
    sections: Any,
    *,
    z: Any = None,
    slice_order: list[str] | None = None,
    slice_key: str = "",
    z_key: str = "",
    obsm_keys: dict[str, list[int]] | None = None,
    coordinate_coverage: float = 1.0,
) -> StackProfile:
    """Per-section coordinate metadata for a stack, with every unestablished fact named."""
    import numpy as np

    from . import geometry as geom

    xy = np.asarray(coords, dtype=float)[:, :2]
    sec = np.asarray(sections).astype(str)
    p = StackProfile(slice_key=slice_key, z_key=z_key, obsm_keys=dict(obsm_keys or {}))
    p.three_column_keys = sorted(k for k, shape in p.obsm_keys.items() if len(shape) == 2 and shape[1] == 3)
    p.coordinate_coverage = float(coordinate_coverage)
    if p.coordinate_coverage < 1.0:
        p.notes.append(
            f"{1.0 - p.coordinate_coverage:.1%} of the observations have no coordinate row. Every "
            f"number below describes the {p.coordinate_coverage:.1%} that do, and nothing here "
            f"should be read as describing the whole object."
        )

    zs: dict[str, float] = {}
    if z is not None:
        zarr = np.asarray(z, dtype=float)
        for s in np.unique(sec):
            v = zarr[sec == s]
            v = v[np.isfinite(v)]
            if len(v):
                zs[str(s)] = float(np.median(v))

    # --- order: explicit, else from z, else unknown. Never from the order rows happen to appear.
    if slice_order:
        p.slice_order = [str(s) for s in slice_order]
        p.slice_order_known = True
        p.slice_order_from = "the caller"
    elif zs and len(zs) == len(set(sec)):
        p.slice_order = [str(s) for s, _ in sorted(zs.items(), key=lambda kv: kv[1])]
        p.slice_order_known = True
        p.slice_order_from = f"obs[{z_key!r}]" if z_key else "a z column"
    else:
        p.slice_order = [str(s) for s in sorted(set(sec))]
        p.slice_order_known = False
        p.slice_order_from = "unknown"
        p.questions.append(
            "What is the physical order of these sections? Without it, which sections are "
            "adjacent is undefined, and no geometric or biological consistency metric can be "
            "computed. Give the order, or name the obs column holding each section's z."
        )

    for s in p.slice_order:
        m = sec == s
        a = xy[m]
        sp = SliceProfile(name=s, n_cells=int(m.sum()), z=zs.get(s))
        finite = a[np.isfinite(a).all(axis=1)]
        sp.n_non_finite = int(len(a) - len(finite))
        if len(finite):
            lo, hi = finite.min(0), finite.max(0)
            sp.bbox = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
            sp.span = [float(hi[0] - lo[0]), float(hi[1] - lo[1])]
            sp.pitch = geom.median_pitch(finite)
            sp.n_duplicate_coords = int(len(finite) - len(np.unique(finite, axis=0)))
            sp.n_at_origin = int((np.abs(finite).sum(1) == 0).sum())
        if sp.n_non_finite:
            sp.notes.append(f"{sp.n_non_finite} cells have a non-finite coordinate and were excluded")
        if sp.n_at_origin > 1:
            sp.notes.append(
                f"{sp.n_at_origin} cells sit at exactly (0, 0), which is usually a missing "
                f"coordinate written as a zero rather than a position"
            )
        p.slices.append(sp)

    # --- units, from the whole stack rather than one section
    finite_all = xy[np.isfinite(xy).all(axis=1)]
    if len(finite_all):
        span = float(np.hypot(*(finite_all.max(0) - finite_all.min(0))))
        p.xy_units, p.units_from = infer_units(span, coords_look_like_array_indices(finite_all))
    if p.xy_units == "unknown":
        p.questions.append(
            f"What are the coordinate units? {p.units_from}. The class A/B/C diagnosis does not "
            f"need them -- every threshold it uses is a fraction of the tissue diagonal -- but a z "
            f"spacing, a 3D neighbourhood radius and any physical figure axis all do."
        )

    # --- z spacing and coincident planes
    if zs and p.slice_order_known:
        ordered = [zs[s] for s in p.slice_order if s in zs]
        if len(ordered) >= 2:
            # Coincidence and spacing are read from the size of each gap, not its sign. Signed, an
            # explicit order running posterior to anterior made every gap negative, so no coincident
            # group was found and the z = 0 sections were paired; one inversion made a negative gap
            # pass `g <= tol` and invented a group (hunt 2026-09-30, u21-3d-7).
            gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]
            p.z_spacings = [float(abs(g)) for g in gaps]
            nonzero = [abs(g) for g in gaps if g != 0]
            if nonzero:
                med = float(np.median(nonzero))
                tol = COINCIDENT_Z_TOLERANCE * med
                p.z_spacing_uniform = bool(np.allclose([g for g in nonzero if g > tol], med, rtol=0.25))
                rising, falling = any(g > tol for g in gaps), any(g < -tol for g in gaps)
                if rising and falling:
                    zlabel = f"obs[{z_key!r}]" if z_key else "the z column"
                    shown = ", ".join(f"{s}={zs[s]:g}" for s in p.slice_order if s in zs)
                    p.notes.append(
                        f"the section order from {p.slice_order_from} is not monotone in {zlabel} "
                        f"({shown}): it runs up and down the z axis, so either the order or the z column "
                        f"is wrong, and the pairs below are adjacent only in the order given."
                    )
                    p.questions.append(
                        f"The section order given contradicts {zlabel}. Which is right -- the order, or the z values?"
                    )
                group: list[str] = []
                names = [s for s in p.slice_order if s in zs]
                for i, g in enumerate(gaps):
                    if abs(g) <= tol:
                        if not group:
                            group = [str(names[i])]
                        group.append(str(names[i + 1]))
                    elif group:
                        p.coincident_z_groups.append(group)
                        group = []
                if group:
                    p.coincident_z_groups.append(group)
        if p.coincident_z_groups:
            n = sum(len(g) for g in p.coincident_z_groups)
            p.notes.append(
                f"{n} sections share a z with another section. Adjacency among them is undefined "
                f"-- any ordering would be invented -- so no pair is formed among them. On "
                f"Zhuang-ABCA-1 this is the four anterior sections the provider clamps to z = 0."
            )
        if p.z_spacing_uniform is False:
            p.notes.append(
                f"the z spacing is not uniform: gaps range {min(p.z_spacings):g} to "
                f"{max(p.z_spacings):g}. A 3D neighbourhood built on a single spacing would be "
                f"wrong wherever the real gap differs."
            )
    elif not zs:
        p.questions.append(
            "What is the spacing between sections, and in which units? There is no z column here, "
            "and this package has no default: a stack assembled on an assumed spacing produces a "
            "3D neighbour graph that is either four disconnected 2D graphs or a single mush, and "
            "neither announces itself."
        )
    return p
