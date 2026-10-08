# 3D Reconstruction from Serial Sections: Diagnose, Align, Then Analyse

## Metadata

**Version**: 1.0
**Short Description**: Multi-slice serial sections into one 3D object: diagnose whether the stack is already aligned, rigidly misaligned or deformed, stop and ask, then align and re-measure before/after.
**Scope**: The phased protocol for a stack of serial sections -- coordinate inventory, the A/B/C diagnosis, the two user checkpoints, alignment into a new coordinate key, and downstream analysis in 3D.
**Applies when**: more than one section of one block is in play -- several h5ad files, an obs column of slice labels, or a request mentioning serial sections, z, stacking, a 3D reconstruction, registering or aligning slides.
**Does not apply to**: a single section; several unrelated samples, which are a batch question and not a geometric one (see `spatial_batch_integration.md`); and mapping dissociated cells onto one section, which is deconvolution.

---

## Overview

A stack of serial sections is three different datasets depending on one answer that cannot be
read off the file: whether the sections already sit in a shared frame, are displaced from it, or
are bent out of it. That answer chooses the transform, the transform decides which biology
survives into every later figure, and part of the evidence for it lives with whoever cut and
imaged the sections. So this protocol runs in phases, and it stops twice.

The failures are symmetric: align a stack that was already registered and you discard somebody
else's registration; rigidly align a torn one and the deformation is absorbed into the
translations, leaving smooth domains and plausible 3D genes that are artefacts of the fit.

**`obsm['spatial']` stays two columns and nothing here writes it.** Forced, not stylistic: PASTE
asserts `X.shape[1] == 2` (`paste/visualization.py:172`) and a recorded run died on a
three-column `spatial`. Z lives in its own key, beside the untouched original.

**Serial sections are far less self-similar than intuition suggests.** On Zhuang-ABCA-1, a real
147-section stack the Allen Institute had already placed in a shared frame -- a known class A --
the median adjacent-pair outline IoU is **0.37**, the maximum over every pair 0.465. A class-A
gate of "IoU above 0.70" calls that whole atlas misaligned, so overlap is reported, never gated.

## Step 0: pick the tool with `recommend_analysis_tools()`

Call `recommend_analysis_tools(h5ad_path="<file>.h5ad", analysis_goals="<your goal>")` first and
`json.loads()` what it returns; use the P1 unless you have a documented, data-driven reason not
to. Resolve a fuzzy name with `resolve_tool_name(query)`, and peek at an unfamiliar tool's schema
as `spatial_tool_schema_precheck.md` describes. From that document, repeated because this
protocol touches many unfamiliar portals: after `add_mcp()` a tool's `spatialomicsgym_name` is
already in scope inside `<execute>` -- call it directly. `from mcp_servers.paste import ...` and
every other `mcp_servers` import form is forbidden (P13): in a python cell it gains nothing over the
name already in scope, and from a `#!BASH` or `#!CLI` cell it fails with `ModuleNotFoundError`.

## The phases

| Phase | Pre-condition | Post-condition | On-failure |
|-------|---------------|----------------|------------|
| **0 Inventory** | Sections readable; one file each, or a slice label per cell | Per section: coordinate keys, units, ranges, count, pitch, z and where the z came from; coincident-z groups listed | Unknown units or z spacing: ask (the stop below). Nothing assumed, no stack written |
| **1 Diagnose** | `obsm['spatial']` present, two columns; **nothing modified here** | Per-pair geometry and expression gap against its null, a per-section batch report, and a verdict A/B/C/unknown, each number beside its threshold | An unmeasurable metric is refused, never substituted. Too many refusals, or an unknown section order, gives `unknown` plus the question that would settle it |
| **CHECKPOINT 1** | Verdict and evidence printed | The user has answered the one question below | No answer means stop. Stopping is this phase succeeding |
| **2 Align** (B or C) | Verdict B or C, user confirmed, a raw frame to compare against | `obsm['spatial_3d_aligned']` with explicit z and provenance; `obsm['spatial']` unchanged; tool, parameters, seed recorded | Not improved: say so, try one alternative, then report the diagnosis as the deliverable |
| **CHECKPOINT 2** | Before/after table, per pair, from the functions Phase 1 used | The user has answered the one question below | As checkpoint 1. A decision not to continue is a result |
| **3 Analyse** | A frame read via `contract.read_frame(adata, role=...)` | Integration where Phase 1 found batch effects, 3D neighbourhoods, domains and SVGs, figures, new keys documented | A tool with no coordinate-key parameter is 2D: run per section and label it so |

## Why this one asks, when the tool playbooks say not to

`add_new_mcp_tool.md` is emphatic in the opposite direction:

> **When these conditions are met, proceed IMMEDIATELY without asking for confirmation.
> The user's explicit request + tool_creation_enabled=True IS the confirmation.
> Do NOT ask "are you sure?" or "please confirm". Just execute all phases.**

Both are right, because they govern different things. That rule governs **executing an
instruction already given**: "install this package from this link" has one interpretation, and
asking again converts a decision the user has made into a wait.

This one governs **choosing between interpretations of the user's tissue**. Class B and class C
lead to different transforms and different biology, and the wrong choice is invisible in the
output, because a rigidly aligned deformed stack produces clean-looking domains. What separates
them lies partly outside the file -- how the sections were cut, mounted and imaged, whether any
were folded, torn or re-cut -- and only the user has that. The checkpoints are not requests for
permission to act; they are the two points where the analysis needs a fact it cannot measure.

## The coordinate rules (they hold in every phase)

Read `spatialomicsgym/spatial3d/contract.py` once before writing anything: it is the only module
that decides where a 3D coordinate lives, and it exists because the ten shipped aligners each
leave their answer somewhere different.

- **`obsm['spatial']` is never overwritten, reordered or extended.** Writers digest it before and
  after and refuse on a change (`contract.assert_original_intact`).
- **Frames are role-suffixed**: `spatial_3d_raw` before any transform, `spatial_3d_aligned` after
  one, `spatial_3d_<source>` for somebody else's registration -- deliberately not `aligned`,
  since folding it in would credit Phase 2 with the provider's work and then compare a frame
  against itself. Plain `spatial_3d` counts as `aligned` when it is the only three-column key.
- **`uns['spatial_3d']` is the provenance registry**, and `contract.read_frame(adata, role=...)`
  the only supported way to obtain coordinates; hardcoding an obsm key is what the ten
  disagreeing aligners already did.
- **Z spacing is asked about, never assumed.** No default exists here and no function invents one.
  A z that is a slice ordinal and a z that is a measured thickness give different 3D neighbour
  graphs from the same array, so `z_source` records which it is, and `unknown` is an answer.
- **Coincident-z sections are reported, not sorted.** Four Zhuang sections sit at exactly z = 0
  because the provider clamps negative anterior estimates; adjacency among them is undefined and
  sorting invents it. `slice_order` is explicit and no reader re-sorts it.

Writing a frame, run end to end on a toy object -- afterwards `contract.validate` returns `[]`:

```python
from spatialomicsgym.spatial3d import contract
frame = contract.Frame(key=contract.ALIGNED_FRAME, role="aligned", xy_units="um", z_units="um",
    z_source="asked_user", z_spacing=10.0, axis_map={"x": "col0", "y": "col1", "z": "col2"},
    source_frame=contract.RAW_FRAME, aligner="paste", seed=0, seed_exposed=True)
contract.write_frame(adata, xyz, frame, slice_key="slice", slice_order=["s1", "s2", "s3"])
print(contract.validate(adata))   # [] means the contract holds; sentences mean it does not
```

## Phase 0 -- inventory

Read, do not write. Per section: which coordinate keys exist and how many columns each has, the
ranges and units, the cell count, the median nearest-neighbour spacing, the z. Then list the
sections sharing a z with another. If the units or the z spacing are unknown, stop:

```python
print("obsm['spatial'] is in unnamed units, ranges 0-19,505 and 0-18,742, median spacing 138 --")
print("consistent with Visium full-resolution pixels. No z: no obs column, none in uns.")
print("Question: what is the spacing between adjacent sections, and in what units?")
print("Default if you cannot say: I build the stack with z_source='unknown', which keeps every")
print("in-plane metric and refuses any statement in physical z units. No spacing is invented.")
# WAIT for user confirmation before proceeding
```

**After the user answers:** record the spacing and its origin (`measured`, `section_metadata` or
`asked_user`) on the frame and go to Phase 1. If they could not say, carry `z_source='unknown'`
and let later steps refuse what needs it, rather than filling the gap with an ordinal.

## Phase 1 -- diagnose (do not modify the data)

Four measurements, deliberately kept apart: the geometry module never opens an expression matrix
and the batch module never reads a coordinate, because "the sections are in different places" and
"the sections were sequenced differently" are different findings with different remedies.

**1. Coordinate metadata** -- the Phase 0 inventory, per section and per adjacent pair.

**2. Geometry, per adjacent pair** (`spatial3d/geometry.py`):

```python
from spatialomicsgym.spatial3d import geometry, classify
g = geometry.pair_geometry(name_a, xy_a, name_b, xy_b)   # in-plane coordinates only
```

Centroid offset, axis angle **or the reason it was refused**, hull-area ratio, aspect change,
occupancy IoU, recentred IoU, containment, a fitted similarity transform with its residual, and
local-shift dispersion -- as a fraction of the pooled robust bbox diagonal as well as in the
coordinates' own units, since the unit verdict is often unknown. Two refusals are load-bearing:
the axis angle is refused on 82 of 142 real adjacent pairs of Zhuang-ABCA-1, sections being too
round for a leading eigenvector to mean anything; and a fit that does not beat translation-only
is `translation_only`, sign flips having been reported here as 136.9-degree rotations between
adjacent sections.

**3. Biology, per adjacent pair** (`biology.py`): five to ten robust genes, all cells, never a
cell subset -- `select_genes`, `smooth_within_section`, then `pair_biology`. It reports a **gap
against a null**, the same computation with one section rotated ninety degrees, because a rho of
0.3 between two sections of one organ is uninterpretable alone and the gap is not.

**4. Expression batch effects** (`batch.py`), kept apart: per-section library size and genes
detected, the fold range between deepest and shallowest, and the genes rejected from the
consistency metric for carrying a section-level shift.

### The A/B/C criteria

```python
verdict = classify.classify_stack(geoms, bios, slice_order_known=..., coincident_z_groups=...)
print(verdict.summary())
for pair in verdict.pairs:
    print(pair.sentence())
```

- **A -- already aligned.** Every gated criterion holds: centroid offset below `T_A_centroid`,
  local shift below `T_A_local_shift`, containment above `T_A_containment`, and, where biology
  could be measured, an expression gap above `T_bio_gap`.
- **C -- non-rigid.** Any one trigger: local shift at or above `T_C_local_shift`, aspect change
  at or above `T_C_aspect`, or containment minus recentred IoU at or above `T_C_partial`, the
  partial-overlap signature. The stack rule needs a fraction of pairs to agree.
- **B -- rigid.** Neither: displaced, turned or rescaled, but explicable by one transform.
- **unknown** -- a verdict, never a quiet fall-back to A. Class A means "skip Phase 2", so a
  stack wrongly called A carries its misalignment into every later result.

Each threshold carries its value, units, calibration set and rationale in
`spatial3d/thresholds.py`, measured in `docs/design/spatial_3d_thresholds.csv` -- **cite that
file rather than retyping numbers**, and print `Threshold.cite(observed)` for the sentence a
report needs. The exception is the number that contradicts intuition: real adjacent class-A
sections measure a **median IoU of 0.37**.

## CHECKPOINT 1 -- the classification, before any transform

Print the class, the evidence and the refusals, then ask exactly one question. The numbers below
are the example's; print the ones you measured.

```python
print("PHASE 1 COMPLETE -- classification: B (rigid misalignment).")
print("  11 of 12 adjacent pairs exceed the class-A centroid offset; the largest is s7->s8.")
print("  No pair trips a non-rigid criterion (local shift, aspect change, partial overlap).")
print("  Outline IoU reported, not gated: median 0.29. Axis angle refused on 7 of 12 pairs.")
print("  Batch, separately: library size differs 2.4-fold between deepest and shallowest.")
print("Question: shall I treat this as class B and align it with paste_pairwise_align?")
print("Default if you simply say go: yes -- class B, paste_pairwise_align, writing")
print("obsm['spatial_3d_aligned'] and leaving obsm['spatial'] untouched.")
# WAIT for user confirmation before proceeding
```

**After the user answers:** record it, and whether it agreed with the measured class, then run
Phase 2 with the tool they settled on. A class-A stack ends here -- report the diagnosis, skip
Phase 2, read the existing frame in Phase 3.

**Stopping here is the success condition of Phase 1, not an incomplete answer.** A reply ending
with the classification, its evidence and this question is a complete Phase 1. Do not route
around the stop by picking the likelier class and carrying on, and do not soften it into "let me
know if you'd like me to continue" while continuing.

## Phase 2 -- align (class B or C only)

Justify the choice in two or three sentences against the diagnosis, then run it.

| The diagnosis says | Call | Notes |
|--------------------|------|-------|
| Class B, full overlap | `paste_pairwise_align` | Pairwise optimal transport, rigid; `paste_center_align` for a consensus centre slice. Both take `random_seed`; pairwise accepts and ignores it, because it is deterministic |
| Partial overlap | `paste2_partial_align` | PASTE2's partial optimal transport. Its overlap fraction `s` defaults to 0.0, which means estimate it per adjacent pair; `paste2_estimate_overlap` reports those fractions without aligning anything. `random_seed` seeds the glmpca initialisation |
| Non-rigid, single-cell resolution | `cast_align_slices`, `stalign_align_points`, or `stalign_align_to_image` | CAST: graph-neural embedding, then affine and free-form registration onto the slice `reference_index` names; `random_seed` seeds numpy and torch. STalign: LDDMM diffeomorphic registration |
| Large data, complex mapping | `moscot_run` with `problem_type='alignment'` and `batch_key` naming each spot's section | Writes `obsm['moscot_spatial_warp']` for every section. `reference_batch` is the section held fixed; left out, the first level of `batch_key` is used and reported as `params.reference_batch_used` |

Also registered: `spiral_align`, `gpsa_align_slices`, `slat_align_slices` (a matching table, not
a coordinate) and `stacker_register` (a warped image). ST-GEARS's third column is the slice
**rank**, not a distance.

Then adopt the result into the contract. PASTE, PASTE2 and CAST keep `obsm['spatial']`
byte-identical and write the aligned coordinates to `obsm['spatial_aligned']` (or
`obsm['spatial_3d_aligned']` when a z spacing was given). Read the output with
`adapters.read_aligned`, attach the z from Phase 0, and write
`obsm['spatial_3d_aligned']` with `contract.write_frame`, recording aligner, function,
parameters, seed and `source_frame`. Six of the twelve shipped portals expose no seed at all; record
`seed_exposed=False` rather than implying reproducibility by silence.

**Validate with the functions Phase 1 used.** Re-run `pair_geometry` and `pair_biology` on the
aligned frame and tabulate before and after per pair -- a second implementation of "the same
metric, after" is how a before/after comparison quietly stops being one. Report per pair: a pair
that got worse is the finding. If the stack did not improve, say so, try one alternative from the
table, and if that fails too the diagnosis is the deliverable.

## CHECKPOINT 2 -- before and after

```python
print("PHASE 2 COMPLETE -- paste_pairwise_align, alpha=0.1 (deterministic; random_seed is ignored).")
print("  centroid offset (fraction of tissue diagonal): median 0.31 -> 0.04 over 12 pairs")
print("  expression gap against the rotated null:       median 0.06 -> 0.14")
print("  one pair got worse: s7->s8, local shift 0.031 -> 0.048 (below the class-C floor)")
print("  obsm['spatial'] unchanged; aligned coordinates in obsm['spatial_3d_aligned'].")
print("Question: shall I run Phase 3 on the aligned frame, s7->s8 included?")
print("Default if you simply say go: yes -- Phase 3 on obsm['spatial_3d_aligned'], with s7->s8")
print("flagged in the report rather than dropped.")
# WAIT for user confirmation before proceeding
```

**After the user answers:** run Phase 3 on the frame they chose, or repeat Phase 2 with the
alternative tool. **Stopping here is the success condition of Phase 2**, as at checkpoint 1: the
before/after table plus this question is a complete phase, and a run that carried straight on
into clustering skipped the only point at which a bad alignment is cheap to catch.

## Phase 3 -- downstream analysis in 3D

**1. Integration, if and only if Phase 1 found batch effects.** Follow
`spatial_batch_integration.md`, and state the method and why. Integration corrects expression,
not coordinates; run instead of Phase 2 it leaves the geometry exactly where it was.

**2. 3D neighbourhoods.** Pass the aligned key to a tool that exposes a coordinate-key parameter
rather than copying z into `spatial`: `squidpy_spatial_neighbors` takes `spatial_key` and
`coord_type`, and `generic` is the one to use, since `grid` assumes a 2D lattice.
`stagate_spatial_domains`, `cellcharter_cluster_spatial_domains`, `spatialde_run_svg` and
`squidpy_spatial_autocorr` expose it too. That the parameter exists is checkable in the schema;
that the implementation accepts three columns is not, so make the first 3D call small and read
its output. A tool without the parameter reads `obsm['spatial']` and, handed a merged object,
treats every section as one squashed plane.

**3. 3D domains.** Cluster on a graph built from the 3D frame; neighbours across sections are the
point. `precast_spatial_clustering` finds shared domains across sections while modelling batch,
but it takes one two-column coordinate table per sample (and CSVs, not h5ad), so its
neighbourhoods cannot cross sections: that is a stack of comparable 2D maps, not a 3D domain
call, and the report must say which was run.

**4. 3D spatially variable genes.** Same rule: the coordinate matrix the test sees must be the
three-column frame, or the result is a 2D SVG list. Where no registered tool takes three columns,
cross-section reproducibility of per-slice lists is a fair alternative, labelled as
reproducibility rather than as a 3D test.

**5. Optional extras** -- deconvolution, cell-cell communication, or an expression trend along an
anatomical axis. **A spatial gradient along an anatomical axis is not a trajectory**: the
repository refuses that claim outright (`viz/capabilities.py`, `trajectory.spatial_axis` --
"spatial proximity is not temporal order"), and the honest form is a distance-to-boundary or
position-along-axis trend.

**6. Visualisation and post-analysis.** Read `visual_analysis.md` before describing any figure.
Four calls on the `spatial_viz` portal cover this phase, and none of them is the ordinary tissue
map -- `plot_spatial_expression` panels per *gene*, so on a merged serial-section object it draws
every section overlaid in one frame, which renders perfectly and shows nothing:

| call | what it answers |
|---|---|
| `plot_section_grid(data_path, obs_key=…)` | what each section looks like, on one shared colour scale |
| `plot_spatial_3d(data_path, view='depth')` | where the sections actually sit -- **run this before any 3D neighbourhood**: coincident planes show as one bar, and a z built from a slice ordinal shows as perfectly even spacing where the real cuts are not |
| `plot_spatial_3d(data_path, view='scatter', genes=…)` | the volume itself, from three fixed angles |
| `plot_spatial_3d(data_path, view='interactive', genes=…)` | the same volume, rotatable in the chat; the PNG remains the figure of record |
| `plot_alignment_qc(data_path)` | one adjacent pair, before frame beside after frame, with that pair's centroid offset in both |

`plot_alignment_qc` refuses unless two distinct coordinate frames are present. That refusal is
information, not an obstacle: an aligner that overwrote the coordinates it was given leaves nothing
to compare, which is the same condition under which its alignment cannot be validated at all.

`plot_spatial_3d(data_path, view='interactive', ...)` also writes a data-only spec that the portal
renders as a rotatable volume in the chat (`interactive.volume`). The PNG remains the declared figure,
and every other interactive figure stays refused (`interactive.any`: a tool hands the browser no
script).

Then `run_post_analysis(OUTPUT_DIR, tool_name=..., task_type='alignment')`, the `task_type` passed
explicitly because an aligned h5ad looks like any other to the detector. Alignment has a **deep**
runner as of 2026-09-22, so the manifest carries the frames it found and a figure of them; the
before/after *comparison* is still yours to state, because only you know which frame was the input.
Quote the manifest's warnings verbatim (`post_task_analysis.md`).

## Quality gates

- [ ] Phase 1 ran before any coordinate was written, and modified nothing.
- [ ] Every section's coordinate keys, units, ranges and z source are recorded, `unknown` included.
- [ ] Z spacing was supplied by the user or recorded as unknown -- never defaulted or inferred.
- [ ] Sections sharing a z are listed as coincident, and no reader re-sorted `slice_order`.
- [ ] The class is A, B, C or unknown, every number printed with its threshold and its calibration
      source (`docs/design/spatial_3d_thresholds.csv`).
- [ ] Outline IoU and the principal-axis angle are reported as evidence, not used as gates.
- [ ] Geometric misalignment and expression batch effects are reported as separate findings.
- [ ] Both checkpoints were presented and answered before the phase after them ran.
- [ ] `obsm['spatial']` holds the same values in the same two columns as at the start, and the
      aligned coordinates are in `obsm['spatial_3d_aligned']` with an explicit z and full
      `uns['spatial_3d']` provenance: aligner, function, parameters, seed, `source_frame`.
- [ ] Before/after metrics come from the functions Phase 1 used, are per pair, and name every
      pair that got worse.
- [ ] Every Phase 3 tool took the 3D frame by an explicit coordinate key, or is labelled a
      per-section 2D result.
- [ ] The final object documents every new key, and the answer lists the assumptions made and the
      points where the result is uncertain.
