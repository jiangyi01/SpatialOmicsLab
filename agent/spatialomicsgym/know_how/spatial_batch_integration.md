# Batch Integration and Composition Analysis

## Metadata

**Version**: 1.0
**Scope**: When and how to correct batch effects in spatial and single-cell data -- detecting batch structure, reusing precomputed integrated embeddings, running harmony correctly, and choosing clustering granularity for composition questions.

---

## Overview

Any dataset combining multiple samples, sections, runs, or timepoints carries batch
structure. Clustering uncorrected data answers "which batch is this from", not "which cell
state is this". This playbook covers detecting the batch axis, integrating properly, and
reporting composition metrics.

## First: inspect what the file already contains

Before computing anything, print `adata.obs.columns`, `adata.obsm.keys()`, and
`adata.uns.keys()`.

- If obsm already carries an integrated embedding (`X_pca_harmony`, `X_harmony`,
  `X_scvi`, `X_scANVI`), USE it -- recomputing integration that the data provider already
  ran wastes the run and often lands on worse parameters.
- Identify the batch column: look for `sample`, `batch`, `run`, `donor`, `section`,
  `timepoint`, or similar obs columns whose values partition the cells.

## Canonical integration recipe (when no integrated embedding ships)

1. If X holds raw counts: `sc.pp.normalize_total(adata, target_sum=1e4)` then
   `sc.pp.log1p(adata)`.
2. `sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")`, subset.
3. `sc.pp.scale(adata, max_value=10)`; `sc.tl.pca(adata, svd_solver="arpack")`.
4. `sce.pp.harmony_integrate(adata, "<batch_column>")` (scanpy.external). This step must
   ACTUALLY EXECUTE -- confirm `X_pca_harmony` appears in obsm afterwards. If harmony is
   unavailable, install or use another true integration method; do not silently fall back
   to uncorrected PCA and report its batch purity.
5. `sc.pp.neighbors(adata, n_neighbors=15-30, use_rep="X_pca_harmony", n_pcs=~30)`.

## Clustering granularity for composition questions

For metrics about cluster composition or mixing (fraction of a sample or timepoint per
cluster, number of batch-dominated clusters), cluster at moderate granularity:
`sc.tl.leiden(resolution=0.3-0.5)`. Over-fragmented clusters (resolution >= 1.0) make
every cluster look pure and inflate domination counts. The coarse band holds ONLY
while every population the task asks about owns a DE-confirmed cluster of its own: a
minority population (under ~10% of cells) often co-embeds inside a dominant lineage
at coarse resolution, and an argmax over coarse clusters then lands on the host
lineage. Raise resolution or subcluster until each sought population separates, then
compute the metric. Report the metric from
`pd.crosstab(adata.obs["leiden"], adata.obs["<group_column>"])`.

## Verify the integration did something

After integration, the batch mixing metric (for example max single-batch fraction per
cluster) should move materially versus the uncorrected embedding. A bit-identical metric
before and after integration means the corrected embedding was never used -- fix the
pipeline, do not report the number.

## QC before composition metrics

Filter obvious noise (off-tissue beads, near-empty spots or cells) before computing any
clustering-based composition metric, especially when the dataset description itself warns
of such noise. Unfiltered noise shatters clustering into hundreds of micro-clusters.
After clustering, check cluster health: a composition analysis expects roughly 5-30
clusters; hundreds means lower the resolution or use a fixed-K method (KMeans on the
integrated embedding) instead.

## Saturated metrics are artifacts

A mixing or domination metric that lands exactly at a degenerate bound (a fraction of
exactly 1.0 or 0.0) almost always reflects micro-clusters or missing QC, not biology.
Treat it as a pipeline error: redo with QC and coarser clusters. Never report a
saturated value while time remains to fix it.

## A timeout means switch method

If an integration or clustering tool times out or fails, do not retry the identical
call. Switch to a different tool or a plain scanpy implementation written inline
(harmony via scanpy.external, or PCA + neighbors + leiden on the corrected embedding).
A plain-library fallback nearly always exists; an empty answer never beats a fallback.

## One job per execution step on large matrices

Never fuse loading a large matrix, a heavy tool call, counting, and answer-writing
into a single step -- one timeout then destroys all of it and invites a placeholder.
Run the heavy tool on the smallest task-relevant subset (subsetting to the named
sample or timepoint is not subsampling). A reparameterized call of the same tool on
the same input is still an identical retry: after a timeout, switch method CLASS,
e.g. direct per-cell marker gating on the original matrix instead of a pipeline tool.

## Audit gates before metrics

Before computing any downstream metric from gated cells, print the gated union as a
percent of all cells and the per-group (per-condition, per-timepoint) fractions. Stop
and loosen the gates when the seed-to-gate collapse exceeds about 5x, the union falls
below a few percent, or an injury or immune population peaks in the control condition
-- each is a gate artifact, not biology. Confounder exclusion must be RELATIVE (drop
cells HIGH in off-lineage markers: upper quantile, or off-score above own-score),
never "any nonzero count" -- imaging and bead assays carry ambient counts everywhere,
and any-nonzero exclusion deletes cells preferentially in dense injured tissue.
