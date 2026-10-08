# Spatial Analysis Workflow Guide

## Metadata

**Version**: 1.0
**Scope**: How a spatial transcriptomics study proceeds end to end -- which question comes next, which tool answers it, and what must be true before trusting the answer.

---

## Overview

The order of a spatial study, the first tool to reach for per task, and the quality gates between
stages. Follow the arc; do not redo completed stages, and run missing prerequisites first.

## The spatial analysis arc

qc -> clustering -> annotation -> deconvolution -> svg_detection -> cell_communication

Why each stage precedes the next:

- **qc before clustering** -- domains found on low-quality spots reproduce sequencing depth, not
  tissue architecture. Cluster only spots that survived QC.
- **clustering before annotation** -- a region can only be named after it has been found; the
  marker genes that name it come from the domain labels.
- **annotation before deconvolution** -- per-spot cell-type mixtures are interpretable once the
  tissue context is known, and deconvolution needs an annotated scRNA-seq reference.
- **deconvolution before svg_detection** -- a "spatially variable" gene may simply track where one
  cell type sits; composition first makes the SVG list readable.
- **svg_detection before cell_communication** -- a ligand-receptor call is credible when both genes
  vary spatially and the interacting cell types are co-located.

Not every study runs every stage. A question that names its own task (for example "which genes are
spatially variable?") enters the arc directly after QC; the stages before it are prerequisites only
when their outputs are actually consumed.

## Quick Reference

| Task | Recommended First Tool | Alternative | Key Metric |
|------|------------------------|-------------|------------|
| Spatial clustering | deepst_identify_domains | stlearn_spatial_clustering, run_bass | ARI, NMI |
| SVG detection | somde_run | hotspot_spatial_modules, spatialde_run_svg | Jaccard, F1 |
| Deconvolution (w/ ref) | spacexr_rctd_deconvolution | run_spatialscope, tacco_annotate | RMSE, Pearson r |
| Deconvolution (no ref) | starfysh_deconvolution | ucdeconvolve_base | visual inspection |
| Cell communication | commot_spatial_communication | spaotsc_run | pathway significance |
| Slice alignment | paste_pairwise_align | moscot_run | alignment score |
| Super-resolution | istar_full_pipeline | xfuse_run | spatial correlation |

Tool names are the canonical registered MCP function names; a name outside that registry cannot be
called no matter how it is phrased. When several fit, prefer the recommendation the tool
recommender returns for the data at hand -- it carries the empirical benchmark ranking.

## Decision points

- **Is there an H&E or fluorescence image?** Histology-aware clustering (the first-tool column)
  uses it; without an image, pick an expression-only alternative from the same row.
- **Is there a matched scRNA-seq reference with cell-type labels?** Reference-based deconvolution
  needs one; without it use the reference-free row, and say so in the report -- its output is a
  hypothesis, not a measurement.
- **One section or several?** Multiple sections from the same tissue must be aligned before any
  cross-section comparison; treat alignment as its own arc stage inserted before the comparison.
- **How large is the slide?** Kernel- and GP-based methods grow quadratically with spot count;
  if a tool documents a spot cap, report what fraction of the slide was analysed rather than
  silently letting a subsample read as the whole tissue.
- **Platform granularity** -- spot-based platforms (Visium) mix cells, so deconvolution is how
  composition is read; single-cell-resolution platforms (MERFISH, Xenium, CosMx) skip
  deconvolution and annotate cells directly.

## Quality gates

- **After QC**: spots retained are in-tissue, above minimum counts/genes, below the mitochondrial
  fraction threshold; report how many spots each filter removed.
- **After clustering**: the number of domains is in a plausible range for the tissue, and domains
  are spatially coherent regions rather than salt-and-pepper noise.
- **After deconvolution**: per-spot proportions sum to 1, no cell type is constant across all
  spots, and the proportions are not uniform 1/k (both patterns mean the fit carried no signal).
- **After SVG detection**: the score column actually varies across genes, and significance comes
  from an FDR-adjusted column when the tool provides one.
- **Everywhere**: correct shape and plausible names prove nothing -- check that values vary and
  come from the tool's published output slot before reporting a number.

## After a task finishes

The post-analysis engine proposes these automatically; they are the biologically natural next
questions per task:

- **spatial_clustering** -- find each domain's marker genes; annotate domains biologically; test
  the domains' spatial coherence.
- **svg_detection** -- run functional enrichment on the spatial genes; group them into
  co-expressed spatial modules.
- **deconvolution** -- map the dominant cell type per spot; test which cell types co-locate;
  cross-check with a second deconvolution method.
- **cell_communication** -- rank ligand-receptor pairs by their spatial support.
- **alignment** -- measure the registration error; compare the aligned sections region by region.
- **imputation** -- validate imputed genes against held-out measurements; test whether imputed
  genes are spatially structured.
- **trajectory** -- plot gene trends along the trajectory; map pseudotime onto the tissue.

Automatic follow-up rounds are capped; the operator knob is
`SOG_POST_ANALYSIS_MAX_FOLLOWUP_ROUNDS`.
