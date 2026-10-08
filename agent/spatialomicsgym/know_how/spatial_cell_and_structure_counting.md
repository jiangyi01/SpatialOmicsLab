# Counting Cells and Anatomical Structures in Spatial Data

## Metadata

**Version**: 1.0
**Scope**: How to count cells, beads, or spots of a named cell type, and how to count anatomical structures (follicles, glomeruli, islets, niches), from marker panels in spatial transcriptomics -- strict seeds, spatial denoising, units versus structures, and biological sanity gates.

---

## Overview

"How many X cells" and "how many X structures" look similar but are different deliverables
with different failure modes. Loose marker thresholds overcount cells severalfold;
aggregate-tuned detectors undercount structures badly. This playbook gives the discipline
for both.

## Build a real marker panel

Use the full literature-standard marker panel for the named type -- a broad panel of genes,
never one or two markers. Single-marker positivity in spatial data is mostly ambient
counts and segmentation noise, not the cell type.

## Count units strictly

- A unit (cell / bead / spot) counts only if it co-expresses a LARGE fraction of the panel
  (strict seed: for example, more than half the panel detected).
- Units positive for only one or two markers are background. Expect the strict count to be
  severalfold smaller than the raw marker-positive count; if the two are close, the
  threshold is too loose.

## Use spatial structure as a denoiser, not as the deliverable

Group candidate units into connected components by spatial proximity (for example
scipy.spatial.cKDTree pairs within a radius, then connected components). Keep only
components anchored by at least one strict seed. Then report what was actually asked:

- Asked for CELLS -- report the number of units inside kept components (`keep.sum()`),
  never the number of components.
- Asked for STRUCTURES -- report the number of distinct kept components.

## Calibrate to physical scale, detect the smallest class

- Choose the linking radius from the structure's physical size expressed in the file's own
  coordinate units (inspect coordinate ranges first; units differ across platforms).
- Verify the detector still finds the SMALLEST valid class of structure -- including
  structures containing a single cell (a primordial follicle is one oocyte). A detector
  that only fires on large aggregates undercounts by an order of magnitude.

## Apply the biological definition before counting

If the named entity cannot exist in the queried context -- a developmental stage before
the entity forms, an anatomical compartment that lacks it, a condition that excludes it --
the correct count is 0. Report 0 with the reasoning. Do not launder marker noise into a
nonzero count; equally, never nudge a well-reasoned 0 upward just because 0 looks
suspicious.

## Sanity-check the magnitude

Before finalizing, check the count against tissue anatomy and dataset scale (organ-level
expectations, fraction of total units). A count that implies most of the tissue is one
rare type, or that a whole organ contains a handful of its defining structures, signals a
thresholding or radius error -- find which by the rule under "A plateau, operationally"
below, and rework before reporting.

## Write the existence check before the counting code

Before any counting code, state in one or two sentences the biological conditions under
which the entity exists, and verify a positive signature in this dataset: its signature
markers present in var_names and expressed, and (for spatially organized entities) a
near/far spatial contrast around candidate anchors. A flat contrast or an absent
signature means the count is 0 -- report it with that reasoning.

## Marker arithmetic runs on the original matrix

Compute marker gates only on the original input expression matrix, and print the
panel-vs-var_names overlap before gating. Derived or HVG-subset outputs from upstream
tools silently drop canonical markers; a panel that intersects to 0-2 genes produces
fabricated zeros or one-gene calls. If fewer than 3 canonical markers survive the
intersection, re-derive the panel from cluster differential expression instead.

## Ship counts only from a plateau

Sweep the structural parameter (radius, threshold) and report the count from a range
where it plateaus. A count read off a steep flank of the sweep is a parameter artifact,
not a measurement; if no plateau exists, follow the no-plateau rule under
"A plateau, operationally" below.

## Two distinct populations must be distinct

When counting or assigning two or more named populations, assign by differential
evidence (score_A minus score_B per cell or cluster), never by two independent absolute
maxima -- one broad cluster can top both panels. Identical masks or values for two
"distinct" populations is an error signal: re-cluster finer or gate per-cell, then redo.

## Observe in one step, choose in the next

Code that produces evidence (a sweep table, a detection verdict, a marker overlap) and
the decision that consumes it belong in DIFFERENT execution steps. Print the table,
end the step, and make the selection in the next step, quoting the printed rows that
justify it. A selection rule written into the same step that computes the evidence
(a hardcoded fallback radius, a pre-chosen threshold) decides before the evidence
exists -- that is fabrication by another name, and the printed sweep becomes ritual.

## A plateau, operationally

A plateau is a window of 3 or more ADJACENT parameter settings whose counts differ by
at most 10-15% relative. Never require exact integer equality in code -- it never
fires and silently hands control to a fallback. Comparing two reparameterizations of
the same gate is not a plateau. The chosen parameter must sit inside a window you can
point to in the printed table; on a monotone sweep there is no plateau, no center,
and no valid mid-ramp pick. Anchor the choice on an evidence axis independent of the
sweep (the structure's physical scale, a marker prior, a biological size range) and
name it.

When the sweep has no plateau, or a count or trajectory sanity check fails, the
failure alone does not say which way the gates are wrong -- an over-strict gate fails
on the same monotone slope an over-loose one does, and a structure count comes out
low when neighboring structures merge as well as when structures are missed.
Diagnose the failing end from printed evidence the sweep did not produce, and change
only what that evidence names. Read the kept components' sizes first. Components
larger than the structure's physical size mean neighboring structures were bridged:
shrink the linking radius if it exceeds what that size calls for, otherwise tighten
the candidate gate that lets the bridging units in -- even when the count is low. A
strict count close to the raw marker-positive count means the seed gate is too
loose: tighten it. An exclusion term that deletes units carrying the panel is too
strict: relax it. A count far below what the anatomy implies, while kept components
are within the structure's physical size, means structures were missed and a gate is
too strict: relax the most fragile one first, an exclusion term before the seed gate.
Evidence that names one gate in both directions does not name that gate, and neither
does a re-sweep that would move a gate back the way the last round moved it from: leave
it, and recheck the panel and the positivity definition. Re-sweep and print the new
table. A gate moved only because the table then looks flatter has been fitted to the
table, not to the tissue. If no diagnosis is left and there is still no plateau, stop
changing gates: take the parameter from the independent axis above, name it, and
state that the count comes from a sweep with no plateau.

## Positivity comes from the assay, not the population

Never define marker positivity by a population quantile (e.g. cells above the 99th
percentile of a score): the cutoff then fixes the positive fraction by construction
and predetermines the count before any biology is measured. Set positivity from the
assay side -- nonzero or moderate absolute expression, a fitted background, or a
published threshold. On bead or spot assays where each unit mixes several cells,
membership is DOMINANCE (which lineage's signal is strongest in the unit), never
mutual exclusion -- a bead inside a structure legitimately carries both the
structure's signal and its neighbors', and excluding on the neighbor's marker
deletes exactly the units being counted.

## The existence check is a terminal branch

For stage- or condition-dependent structures, run the existence check as its own step
with NO counting code in it, ending in a printed verdict: PRESENT (signature markers
expressed AND a positive near/far spatial contrast) or ABSENT. If ABSENT, the count is
0 and the analysis is finished -- do not reinterpret the task as proximity counting,
and answer options are never evidence. Write counting code only after a PRESENT
verdict has been observed in a previous step's output.

## One primary gate per count

Never AND cluster membership with per-cell marker gates with percentile cutoffs --
stacked heterogeneous gates multiply false-negative rates and crush recall. Pick one
primary gate, sweep it (2D where feasible), and use the other evidence types only as
printed sanity checks. "No robustness sweep is necessary" is never a valid claim.
