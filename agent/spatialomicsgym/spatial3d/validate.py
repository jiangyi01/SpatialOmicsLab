"""Did the alignment help? The same measurements as Phase 1, run twice.

Phase 2's deliverable is not an aligned file; it is the evidence that aligning changed something
for the better. This module produces that evidence, and it does so by calling **the same functions**
:mod:`~spatialomicsgym.spatial3d.classify` calls -- not a second implementation of "the same
metric, after". ``test/test_spatial3d_before_and_after_are_the_same_measurement.py`` asserts that
by AST, because a before/after comparison whose two halves were written separately quietly stops
being one.

**The pass criteria are four, and the fourth is the one that matters.** A recorded run in this
repository reported a "mean pairwise improvement in nearest-neighbor overlap score" of 3922 and
called itself an alignment. Its own transform table
(``demo_outputs/moffitt_hypothalamus_aligned_multimethod_3d_20260527_113551/alignment/adjacent_slice_transforms_animal1.csv``)
records a *negative* improvement on nine of its eleven pairs. A global mean hides a collapsed pair,
so the fourth criterion is that **no pair got worse by more than a stated tolerance**, and it is
checked per pair rather than on an average.

**A failure names the next tool rather than just failing.** The Phase-2 protocol says that if
validation fails, try an alternative -- so a failed verdict carries a recommendation
(:func:`recommend_alternative`), with the reason it is the right next thing to try.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import geometry as geom
from . import thresholds as T

#: A pair may get this much worse on a metric before it counts as a regression. Not zero: the
#: metrics are estimates with their own noise, and a fit that improves the stack overall can move
#: one pair fractionally the wrong way without anything having gone wrong.
REGRESSION_TOLERANCE = 0.10

#: ...and a change smaller than this, in the metric's own units, is noise however large it is
#: relative to the value before. A tenth of the class-A threshold. Purely relative, a centroid
#: offset already at 0.0036 -- a thirtieth of T_A_centroid -- "regressed" by moving to 0.0043, and a
#: correct rotation fix failed validation on it (hunt 2026-09-30, u21-3d-6).
REGRESSION_FLOOR = {
    "centroid_offset_frac": 0.1 * float(T.T_A_CENTROID),
    "local_shift_dispersion": 0.1 * float(T.T_A_LOCAL_SHIFT),
    "containment": 0.1 * float(T.T_A_CONTAINMENT),
}

#: Which way is better, per metric. Explicit rather than inferred from the name, because
#: ``centroid_offset_frac`` and ``iou`` disagree and a sign error here inverts the verdict.
DIRECTION = {
    "centroid_offset_frac": "lower",
    "local_shift_dispersion": "lower",
    "resid_rigid": "lower",
    "iou": "higher",
    "iou_recentred": "higher",
    "containment": "higher",
    "aspect_change_log2": "lower",
}

#: The metrics the verdict is decided on, in the order a report should read them.
GATED = ("centroid_offset_frac", "containment", "local_shift_dispersion")


@dataclass
class PairDelta:
    """One adjacent pair, before and after."""

    a: str
    b: str
    before: dict[str, float] = field(default_factory=dict)
    after: dict[str, float] = field(default_factory=dict)
    delta: dict[str, float] = field(default_factory=dict)
    regressed: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"a": self.a, "b": self.b}
        for m in self.before:
            row[f"{m}_before"] = self.before.get(m)
            row[f"{m}_after"] = self.after.get(m)
            row[f"{m}_delta"] = self.delta.get(m)
        row["regressed"] = "; ".join(self.regressed)
        return row


@dataclass
class ValidationVerdict:
    """Whether the alignment may be believed, and what to do if not."""

    passed: bool = False
    pairs: list[PairDelta] = field(default_factory=list)
    n_improved: int = 0
    n_worsened: int = 0
    medians_before: dict[str, float] = field(default_factory=dict)
    medians_after: dict[str, float] = field(default_factory=dict)
    criteria: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    recommended_alternative: str = ""

    def summary(self) -> str:
        head = "alignment PASSED validation" if self.passed else "alignment FAILED validation"
        body = f"{len(self.pairs)} pairs, {self.n_improved} improved, {self.n_worsened} worsened."
        tail = " ".join(self.failures) if self.failures else " ".join(self.criteria)
        out = f"{head}. {body} {tail}".strip()
        if self.recommended_alternative:
            out += f" {self.recommended_alternative}"
        return out


def _improved(metric: str, before: float, after: float) -> bool:
    return after < before if DIRECTION.get(metric, "lower") == "lower" else after > before


def _regressed(metric: str, before: float, after: float) -> bool:
    import math

    if not (math.isfinite(before) and math.isfinite(after)):
        return False
    scale = abs(before) if abs(before) > 1e-12 else 1.0
    slack = max(REGRESSION_TOLERANCE * scale, REGRESSION_FLOOR.get(metric, 0.0))
    return (after > before + slack) if DIRECTION.get(metric, "lower") == "lower" else (after < before - slack)


def before_after(
    pairs_before: list[Any],
    pairs_after: list[Any],
    *,
    metrics: tuple[str, ...] = tuple(DIRECTION),
    aligner: str = "",
) -> ValidationVerdict:
    """Compare two runs of the Phase-1 geometry over the same adjacent pairs.

    Both lists must describe the *same* pairs in the same order; a validation that silently
    compared different pairs would be the most convincing wrong answer this module could give,
    so a mismatch raises rather than aligning by name. ``aligner`` names the tool whose output
    the "after" is, for the recommendation a failed verdict carries.
    """
    import math
    import statistics

    if len(pairs_before) != len(pairs_after):
        raise ValueError(f"before/after must cover the same pairs: {len(pairs_before)} vs {len(pairs_after)}")
    v = ValidationVerdict()
    for b, a in zip(pairs_before, pairs_after, strict=True):
        if (b.a, b.b) != (a.a, a.b):
            raise ValueError(f"pair mismatch: {b.a}->{b.b} before, {a.a}->{a.b} after")
        d = PairDelta(a=b.a, b=b.b)
        for m in metrics:
            x, y = getattr(b, m, float("nan")), getattr(a, m, float("nan"))
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                continue
            d.before[m], d.after[m] = float(x), float(y)
            d.delta[m] = float(y) - float(x)
            if m in GATED and _regressed(m, float(x), float(y)):
                d.regressed.append(m)
        v.pairs.append(d)

    gated_improved = 0
    for d in v.pairs:
        if d.regressed:
            v.n_worsened += 1
        elif any(_improved(m, d.before[m], d.after[m]) for m in GATED if m in d.before):
            gated_improved += 1
    v.n_improved = gated_improved

    for m in metrics:
        xs = [d.before[m] for d in v.pairs if m in d.before and math.isfinite(d.before[m])]
        ys = [d.after[m] for d in v.pairs if m in d.after and math.isfinite(d.after[m])]
        if xs:
            v.medians_before[m] = float(statistics.median(xs))
        if ys:
            v.medians_after[m] = float(statistics.median(ys))

    # --- the four criteria ---------------------------------------------------------------------
    checks: list[tuple[bool, str]] = []
    cb, ca = v.medians_before.get("centroid_offset_frac"), v.medians_after.get("centroid_offset_frac")
    # A median already on the right side of its class-A threshold passes unless it got worse by
    # more than the regression tolerance; a strict decrease was demanded before, so a median at the
    # noise floor failed on a wobble (hunt 2026-09-30, u21-3d-6).
    if cb is not None and ca is not None:
        ok = ca < float(T.T_A_CENTROID) and not _regressed("centroid_offset_frac", cb, ca)
        checks.append((ok, f"median centroid offset {cb:.4g} -> {ca:.4g} ({T.T_A_CENTROID.cite(ca)})"))
    ib, ia = v.medians_before.get("containment"), v.medians_after.get("containment")
    if ib is not None and ia is not None:
        ok = ia > float(T.T_A_CONTAINMENT) and not _regressed("containment", ib, ia)
        checks.append((ok, f"median containment {ib:.4g} -> {ia:.4g} ({T.T_A_CONTAINMENT.cite(ia)})"))
    sb, sa = v.medians_before.get("local_shift_dispersion"), v.medians_after.get("local_shift_dispersion")
    if sb is not None and sa is not None:
        ok = not _regressed("local_shift_dispersion", sb, sa)
        checks.append((ok, f"median local-shift dispersion {sb:.4g} -> {sa:.4g}"))
    # The fourth, and the reason this module exists in this shape.
    no_regression = v.n_worsened == 0
    checks.append(
        (
            no_regression,
            f"{v.n_worsened} of {len(v.pairs)} pairs got worse by more than "
            f"{REGRESSION_TOLERANCE:.0%} on a gated metric",
        )
    )

    v.criteria = [text for ok, text in checks if ok]
    v.failures = [text for ok, text in checks if not ok]
    v.passed = bool(checks) and not v.failures
    # The module promised that a failure names the next tool, and no live caller ever asked for
    # it, so the post-analysis warning carried no next step (hunt 2026-09-30, u21-3d-23).
    if not v.passed:
        v.recommended_alternative = recommend_alternative(aligner or "this alignment", v)
    return v


def recommend_alternative(current: str, verdict: ValidationVerdict) -> str:
    """The next aligner to try, and why, when validation failed."""
    if verdict.passed:
        return ""
    shift_after = verdict.medians_after.get("local_shift_dispersion", float("nan"))
    nonrigid = shift_after == shift_after and shift_after >= float(T.T_C_LOCAL_SHIFT)
    if nonrigid:
        return (
            f"The residual after {current} is still not explained by one global transform "
            f"(local-shift dispersion {shift_after:.4g}). That is a non-rigid deformation, and a "
            # Named only the point-CSV tool; CAST is the non-rigid aligner that reads AnnData
            # (hunt 2026-09-30, u21-3d-9).
            f"similarity transform cannot undo it: try cast_align_slices, which fits a free-form "
            f"deformation to AnnData sections, or stalign_align_points, which fits a diffeomorphism "
            f"to point CSVs, rather than another rigid aligner."
        )
    return (
        f"{current} left a residual that one global transform should have been able to remove. "
        f"Before reaching for a different method, check the section order and the pairs listed as "
        f"regressed: a single mis-ordered or fractured section drags a whole-stack fit. "
        # moscot_run takes problem_type, not problem: the literal call failed on an unexpected
        # keyword (hunt 2026-09-30, u21-3d-9).
        f"moscot_run(problem_type='alignment', batch_key=<the section column>, reference_batch=<one "
        f"section>) is the next tool to try."
    )


def stack_pairs(adata: Any, sections: Any, xy: Any) -> tuple[list[tuple[str, str]], str]:
    """The adjacent pairs a before/after must measure, and where their order came from.

    Never a sort of the labels: ``sorted(set(sections))`` puts s10 between s1 and s2 and Bregma
    '-0.04' beside '0.01', so a before/after over it reports on pairs that are not adjacent
    (hunt 2026-09-30, u21-3d-13). The order is ``uns['spatial_3d']['slice_order']`` when it names
    exactly the sections present, else the z column through the same profile the diagnosis uses
    (coincident planes contribute no pair). With neither, the pairs are empty and the string says
    why, for the caller to report rather than guess.
    """
    import numpy as np

    from . import contract, profile

    sec = np.asarray(sections).astype(str)
    present = sorted(set(sec))
    block = contract._block(adata)
    obs = getattr(adata, "obs", None)
    columns = list(getattr(obs, "columns", []))
    zk = str(block.get("z_key") or "") or next((c for c in ("slice_z", "z", "Bregma") if c in columns), "")
    z = None
    if zk and zk in columns:
        try:
            z = obs[zk].to_numpy(dtype=float)
        except Exception:
            z, zk = None, ""
    declared = [str(s) for s in (block.get("slice_order") or [])]
    if declared:
        if sorted(declared) != present or len(set(declared)) != len(declared):
            return [], (
                f"uns['spatial_3d']['slice_order'] lists {declared[:12]}, which is not exactly the sections "
                f"present ({present[:12]}), so which sections are adjacent is not established"
            )
        prof = profile.profile_stack(xy, sec, z=z, slice_order=declared, z_key=zk if z is not None else "")
        return prof.adjacent_pairs, "uns['spatial_3d']['slice_order']"
    if z is not None:
        prof = profile.profile_stack(xy, sec, z=z, z_key=zk)
        if prof.slice_order_known:
            return prof.adjacent_pairs, f"obs[{zk!r}]"
    return [], (
        "neither uns['spatial_3d']['slice_order'] nor a z column (slice_z, z, Bregma) gives the section "
        "order, so which sections are adjacent is unknown; record the order with "
        "contract.write_frame(..., slice_order=[...]) to make this measurable"
    )


def regeometry(
    pairs: list[tuple[str, Any, str, Any]],
) -> list[Any]:
    """Run the Phase-1 geometry over a list of (name_a, xy_a, name_b, xy_b) tuples.

    The single entry point both halves of a before/after use, so neither half can drift into
    measuring something the other does not.
    """
    return [geom.pair_geometry(na, xa, nb, xb) for na, xa, nb, xb in pairs]
