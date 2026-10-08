"""Do the sections differ in how they were sequenced, as opposed to where they are?

This is the other half of the Phase-1 question, and keeping it apart from the first half is the
point of the module. A stack can be perfectly registered and still have one section sequenced to
twice the depth of its neighbour; a stack can be badly misregistered and chemically identical.
Those two findings have different remedies -- one is an alignment, the other an integration -- and
reporting either as the other sends the study down the wrong branch.

The separation is structural, not a convention: **this module imports nothing from**
:mod:`~spatialomicsgym.spatial3d.geometry`, **and never reads a coordinate.** Every statistic here
is computed from the expression matrix and the section label alone, so it is arithmetically
impossible for a coordinate problem to show up in this report.

**What is deliberately not computed, and why.** There is no kBET score and no expression-space
mixing statistic. Both need a PCA and a cross-section neighbour graph over every cell, which on a
4.2-million-cell atlas is not affordable inside a diagnosis that is supposed to be cheap and
non-destructive. The report says that it was not computed and what it would have told you, rather
than substituting a cheaper statistic under the same name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Sections whose median library size differs from the stack median by more than this factor are
#: called out. Two-fold is the point at which a depth difference starts to dominate a per-gene
#: comparison that has not been depth-normalised.
DEPTH_FOLD_FLAG = 2.0

#: The Kruskal-Wallis p below which per-section library sizes are reported as differing. This is a
#: flag on a description, not a hypothesis test anyone will act on directly -- with millions of
#: cells almost any difference is "significant", which is exactly why the fold change above is
#: reported beside it and is the number the text leads with.
KW_ALPHA = 0.001


@dataclass
class BatchReport:
    """Per-section expression differences, and what was not measured."""

    sections: list[str] = field(default_factory=list)
    median_total_counts: dict[str, float] = field(default_factory=dict)
    median_genes_detected: dict[str, float] = field(default_factory=dict)
    depth_fold_range: float = 1.0
    depth_kw_p: float = float("nan")
    adjacent_profile_pearson: dict[str, float] = field(default_factory=dict)
    min_adjacent_profile_pearson: float = float("nan")
    genes_rejected_as_batchy: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    not_computed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.sections:
            return "no sections to compare"
        bits = [
            f"{len(self.sections)} sections",
            f"library-size fold range {self.depth_fold_range:.2f}x",
        ]
        if self.adjacent_profile_pearson:
            bits.append(f"min adjacent expression-profile r {self.min_adjacent_profile_pearson:.3f}")
        if self.genes_rejected_as_batchy:
            bits.append(f"{len(self.genes_rejected_as_batchy)} genes carry a section-level shift")
        return "; ".join(bits)


def batch_report(
    sections: Any,
    totals: Any,
    genes_detected: Any,
    section_order: list[str],
    *,
    profiles: dict[str, Any] | None = None,
    rejected_batchy: list[str] | None = None,
) -> BatchReport:
    """Describe how the sections differ chemically. No coordinate is read.

    ``profiles`` maps section label -> mean expression vector over a shared gene set, which the
    caller has already computed section-wise; passing it in rather than recomputing keeps the one
    gene-wise pass in :mod:`biology` the only pass over the matrix.
    """
    import numpy as np

    rep = BatchReport(sections=list(section_order))
    sec = np.asarray(sections)
    tot = np.asarray(totals, dtype=float)
    det = np.asarray(genes_detected, dtype=float)

    per_section_totals = []
    for s in section_order:
        m = sec == s
        if not m.any():
            continue
        rep.median_total_counts[s] = float(np.median(tot[m]))
        rep.median_genes_detected[s] = float(np.median(det[m]))
        per_section_totals.append(tot[m])

    meds = np.array(list(rep.median_total_counts.values()), dtype=float)
    meds = meds[meds > 0]
    if len(meds) >= 2:
        rep.depth_fold_range = float(meds.max() / meds.min())
        if rep.depth_fold_range > DEPTH_FOLD_FLAG:
            rep.notes.append(
                f"library size differs {rep.depth_fold_range:.1f}-fold between the deepest and "
                f"shallowest section; a per-gene comparison across sections must be depth-normalised "
                f"first, and a cross-slice integration is likely to be needed before clustering"
            )
    if len(per_section_totals) >= 2:
        try:
            from scipy.stats import kruskal

            rep.depth_kw_p = float(kruskal(*per_section_totals).pvalue)
        except Exception:
            rep.depth_kw_p = float("nan")

    if profiles:
        rs = []
        for a, b in zip(section_order, section_order[1:], strict=False):
            if a in profiles and b in profiles:
                x, y = np.asarray(profiles[a], float), np.asarray(profiles[b], float)
                if x.std() > 0 and y.std() > 0:
                    r = float(np.corrcoef(x, y)[0, 1])
                    rep.adjacent_profile_pearson[f"{a}->{b}"] = r
                    rs.append(r)
        if rs:
            rep.min_adjacent_profile_pearson = float(min(rs))

    rep.genes_rejected_as_batchy = list(rejected_batchy or [])
    if rep.genes_rejected_as_batchy:
        rep.notes.append(
            f"{len(rep.genes_rejected_as_batchy)} genes were excluded from the alignment "
            f"consistency metric because their per-section means move more than their spatial "
            f"pattern does; they describe the batch, not the anatomy"
        )

    rep.not_computed.append(
        "an expression-space mixing statistic (kBET or equivalent) was not computed: it needs a "
        "PCA and a cross-section neighbour graph over every cell, which a non-destructive "
        "diagnosis cannot afford on an atlas-scale object. What it would add is whether cells from "
        "different sections intermingle in expression space after integration."
    )
    return rep
