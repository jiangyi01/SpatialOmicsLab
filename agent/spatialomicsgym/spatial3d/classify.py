"""A (already aligned) / B (rigid) / C (non-rigid) / unknown -- with the evidence attached.

The verdict is the deliverable of Phase 1, and the number beside it is what makes it one. Every
statement this module produces names the metric, its measured value, the threshold it was
compared against, and where that threshold was measured. A bare class letter is not an output of
this module.

**``unknown`` is a verdict, never a fallback to A.** Class A means "skip Phase 2", so a stack
wrongly called A is a stack whose misalignment is carried silently into every downstream result.
When the evidence is missing, refused or contradictory, the honest answer is that the class was
not determined, together with the specific question whose answer would determine it.

**No gate depends on the principal-axis angle.** It is refused on 82 of the 142 adjacent pairs of
the real class-A atlas -- brain sections are frequently too round for a leading eigenvector to
mean anything -- so a rule that required it would be undecidable more often than not. The angle
is reported as evidence where it exists.

**No gate depends on IoU either.** The measured median on a known class-A stack is 0.37. See
:mod:`~spatialomicsgym.spatial3d.thresholds` for what that rules out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import thresholds as T

CLASSES = ("A", "B", "C", "unknown")

DESCRIPTIONS = {
    "A": "already in a shared coordinate system",
    "B": "rigid misalignment -- translation, rotation or uniform scale only",
    "C": "non-rigid deformation -- tearing, stretching or partial overlap",
    "unknown": "not determined from the available evidence",
}


@dataclass
class PairVerdict:
    """One adjacent pair's class, and why."""

    a: str
    b: str
    label: str = "unknown"
    evidence: list[str] = field(default_factory=list)
    unmet: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def sentence(self) -> str:
        head = f"{self.a} -> {self.b}: {self.label} ({DESCRIPTIONS[self.label]})"
        body = "; ".join(self.triggers or self.unmet or self.evidence)
        return f"{head}. {body}" if body else head


@dataclass
class StackVerdict:
    """The stack's class, the pair table behind it, and the question to ask if it is unknown."""

    label: str = "unknown"
    n_pairs: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    pairs: list[PairVerdict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    question: str = ""

    def summary(self) -> str:
        head = f"class {self.label} -- {DESCRIPTIONS[self.label]}"
        tally = ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()) if v)
        out = f"{head}. {self.n_pairs} adjacent pairs ({tally})."
        if self.reasons:
            out += " " + " ".join(self.reasons)
        if self.question:
            out += f" {self.question}"
        return out


def _finite(x: Any) -> bool:
    import math

    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def classify_pair(geom: Any, bio: Any = None) -> PairVerdict:
    """Classify one adjacent pair from its geometry, and its biology where available."""
    v = PairVerdict(a=geom.a, b=geom.b)

    needed = {
        "centroid_offset_frac": geom.centroid_offset_frac,
        "local_shift_dispersion": geom.local_shift_dispersion,
        "containment": geom.containment,
        "iou_recentred": geom.iou_recentred,
        "aspect_change_log2": geom.aspect_change_log2,
    }
    v.missing = [k for k, x in needed.items() if not _finite(x)]
    if v.missing:
        v.label = "unknown"
        v.unmet.append("could not measure " + ", ".join(v.missing))
        return v

    cent, shift = needed["centroid_offset_frac"], needed["local_shift_dispersion"]
    contain, iou_rec = needed["containment"], needed["iou_recentred"]
    aspect = needed["aspect_change_log2"]
    partial = contain - iou_rec

    # --- positively non-rigid? Any one criterion is a trigger; the stack rule needs several. ----
    if shift >= float(T.T_C_LOCAL_SHIFT):
        v.triggers.append(f"local shift {T.T_C_LOCAL_SHIFT.cite(shift)}")
    if aspect >= float(T.T_C_ASPECT):
        v.triggers.append(f"aspect change {T.T_C_ASPECT.cite(aspect)}")
    if partial >= float(T.T_C_PARTIAL):
        v.triggers.append(f"partial overlap, containment minus recentred IoU {T.T_C_PARTIAL.cite(partial)}")
    if v.triggers:
        v.label = "C"
        return v

    # --- already aligned? every criterion must hold -------------------------------------------
    checks = [
        (cent < float(T.T_A_CENTROID), f"centroid offset {T.T_A_CENTROID.cite(cent)}"),
        (shift < float(T.T_A_LOCAL_SHIFT), f"local shift {T.T_A_LOCAL_SHIFT.cite(shift)}"),
        (contain > float(T.T_A_CONTAINMENT), f"containment {T.T_A_CONTAINMENT.cite(contain)}"),
    ]
    if bio is not None and _finite(getattr(bio, "bio_gap", None)):
        gap = bio.bio_gap
        checks.append((gap > float(T.T_BIO_GAP), f"expression gap against its null {T.T_BIO_GAP.cite(gap)}"))

    v.evidence = [text for ok, text in checks if ok]
    v.unmet = [text for ok, text in checks if not ok]
    v.label = "A" if not v.unmet else "B"
    return v


def classify_stack(
    pairs: list[Any],
    bios: list[Any] | None = None,
    *,
    slice_order_known: bool = True,
    units_known: bool = True,
    coincident_z_groups: list[list[str]] | None = None,
) -> StackVerdict:
    """Roll adjacent-pair verdicts up into one class for the stack."""
    bios = bios or [None] * len(pairs)
    verdicts = [classify_pair(g, b) for g, b in zip(pairs, bios, strict=True)]
    out = StackVerdict(n_pairs=len(verdicts), pairs=verdicts)
    out.counts = {c: sum(1 for v in verdicts if v.label == c) for c in CLASSES}

    if not verdicts:
        out.label = "unknown"
        out.reasons.append("There were no adjacent pairs to measure.")
        out.question = "Which sections are adjacent, and in what order?"
        return out

    n = len(verdicts)
    frac = {c: out.counts[c] / n for c in CLASSES}

    # Conditions under which no pair-level tally can be trusted, checked before the tally.
    if not slice_order_known:
        out.label = "unknown"
        out.reasons.append(
            "The section order is not known, so which sections are adjacent is undefined and "
            "every pairwise measurement above was made against a guess."
        )
        out.question = "What is the physical order of the sections, or which obs column holds their z?"
        return out
    if coincident_z_groups:
        names = ", ".join(", ".join(g) for g in coincident_z_groups[:2])
        out.reasons.append(
            f"{sum(len(g) for g in coincident_z_groups)} sections share a z with another section "
            f"({names}); adjacency among them is undefined and those pairs were excluded."
        )
    if frac["unknown"] > float(T.STACK_UNKNOWN_FRACTION):
        out.label = "unknown"
        out.reasons.append(
            f"{out.counts['unknown']} of {n} pairs could not be measured, above the "
            f"{float(T.STACK_UNKNOWN_FRACTION):.0%} the verdict tolerates."
        )
        out.question = "Are the coordinates and the section labels complete for every section?"
        return out
    if not units_known:
        out.reasons.append(
            "The coordinate units could not be established. Every threshold used here is a "
            "fraction of the tissue diagonal and so is unit-free, but any physical statement "
            "downstream -- a z spacing, a neighbourhood radius -- is not yet expressible."
        )

    if frac["C"] > float(T.STACK_C_FRACTION):
        out.label = "C"
        out.reasons.append(
            f"{out.counts['C']} of {n} pairs show deformation no single transform can undo, above "
            f"the {float(T.STACK_C_FRACTION):.0%} tolerance."
        )
    elif frac["A"] >= float(T.STACK_A_FRACTION):
        out.label = "A"
        worst = max(verdicts, key=lambda v: len(v.unmet))
        out.reasons.append(
            f"{out.counts['A']} of {n} pairs meet every class-A criterion."
            + (f" The weakest pair is {worst.a} -> {worst.b}: {'; '.join(worst.unmet)}." if worst.unmet else "")
            + _minority_c(verdicts)
        )
    else:
        out.label = "B"
        # "No pair shows a deformation" was said unconditionally, including beside a class-C pair
        # the tolerance let through; and a C pair never fills `unmet`, so the A branch's "weakest
        # pair" could not name it either (hunt 2026-09-30, u21-3d-11).
        out.reasons.append(
            f"{out.counts['A']} of {n} pairs are aligned, below the "
            f"{float(T.STACK_A_FRACTION):.0%} required for class A"
            + (
                ", and no pair shows a deformation a similarity transform could not undo."
                if not out.counts["C"]
                else "."
            )
            + _minority_c(verdicts)
        )
    return out


def _minority_c(verdicts: list[PairVerdict]) -> str:
    """The class-C pairs a stack verdict of A or B tolerated, by name and trigger -- or nothing."""
    deformed = [v for v in verdicts if v.label == "C"]
    if not deformed:
        return ""
    shown = "; ".join(f"{v.a} -> {v.b} ({', '.join(v.triggers)})" for v in deformed[:3])
    more = f"; and {len(deformed) - 3} more" if len(deformed) > 3 else ""
    return (
        f" {len(deformed)} of {len(verdicts)} pairs nonetheless show a deformation no single transform "
        f"can undo, within the {float(T.STACK_C_FRACTION):.0%} the stack verdict tolerates: {shown}{more}."
    )
