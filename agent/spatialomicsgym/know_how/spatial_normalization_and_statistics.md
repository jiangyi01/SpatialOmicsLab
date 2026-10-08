# Normalization Choices and Statistical Criteria in Spatial Analysis

## Metadata

**Version**: 1.0
**Scope**: Choosing the right normalization for imaging-based panels (cell geometry versus library size), honoring the exact statistical criterion a task specifies, and default rigor (FDR, pseudobulk) when none is specified.

---

## Overview

Two quiet killers of spatial analyses: normalizing a small targeted panel by library size
(a compositional artifact that can flip the sign of fold changes and correlations), and
applying a different statistical rule than the question asked for. Both produce
confident, wrong numbers.

## Imaging panels: normalize by cell geometry for expression comparisons

For expression-level comparisons (fold change between groups, gene-gene coexpression or
correlation, normalized mean expression) on imaging-based platforms (MERFISH, Xenium,
CosMx -- targeted panels of a few hundred genes):

- Library-size normalization distorts the panel: with few genes, one abundant program
  (for example myelin genes in oligodendrocytes) dominates total counts, so dividing by
  totals anti-correlates everything with that program.
- The published standard is geometry normalization: divide counts by cell VOLUME or AREA.
  Check obs for geometry columns (`volume`, `area`) or compute area from per-cell
  min/max coordinate bounds when those ship in obs.
- Fall back to library-size normalization only when no geometry is available.
- QC-style metrics (doublet or ambient assessment) keep their standard default workflow;
  this rule targets expression comparisons.

## The stated criterion is the criterion

When the question pins an exact statistical rule -- test name, p threshold, sidedness,
correction -- apply exactly that rule. Do not add corrections it did not request: no FDR
on top of a stated raw p cutoff, no switching a two-sided test to one-sided. Extra rigor
that changes the counted quantity is a wrong answer, not a safer one.

## A ranking by a statistic is not a ranking by importance

Most questions do not ask for the largest effect. They ask for the most important ones, the most
informative ones, the ones that *explain* some named phenomenon, the ones *attributable* to some
factor. That qualifier is part of the criterion, not decoration on it, and the place to honour it
is in how the ranking is built -- not in an argument afterwards about which rows to skip. Sorting
by raw magnitude and then reaching past five rows to the entry you wanted is the same judgement
call the question asked you to compute, made later and with no record of the rule.

So decide what makes an entry eligible, apply that rule to **every** row before sorting, state the
rule in one line, and print the rows it removed. Then fill the answer from the top of that ranking
and stop. There are two failure directions here and a defensible answer clears both:

- Ranking by a proxy the question did not ask for, then hand-picking from the result.
- Applying the qualifier only to the rows that happened to occur to you, which is a preference with
  a justification attached rather than a criterion.

When the answer has a fixed number of slots, those slots are not a committee with seats to share
out. If the ranking puts several entries of one kind at the top, several entries of one kind are
the answer. A list naming one of this, one of that and one familiar entry reads as balanced and
considered, but that is a fact about how it reads, not about what was measured.

## What the null shuffles decides what "enriched" means

"Enriched beyond chance" means nothing until you say what chance was. Neighborhood-enrichment and
co-occurrence routines default to permuting cell-type labels across the whole tissue, which
destroys every scale of spatial structure at once. Under that null the top of the ranking is
dominated by pairs that share an anatomical compartment or are physically obligate neighbours --
types that wrap or line one another, or that are simply confined to the same region. Those pairs
are genuinely enriched under the null that was used, and they are also a restatement of the
tissue's architecture rather than a finding about this sample.

- Name the null in the write-up, not just the statistic. "z = 14 under a global label permutation"
  is a claim; "z = 14" on its own is not.
- Before accepting the top pairs, ask of each whether it would score just as high in any healthy
  tissue of this type. If it would, the statistic is measuring architecture, and the question was
  probably about something else.
- Where the routine supports it, prefer a null that preserves local composition -- permuting within
  region, compartment or neighbourhood -- so the statistic tests affinity rather than
  co-residence. Report which null you chose and why.
- Read the contact counts alongside the z-scores. A large z backed by a handful of contacts is a
  small-sample artifact, and only the count matrix separates the two.
- Self-self pairs sit on the diagonal of these matrices and are usually the largest values in them.
  When the question asks for pairs of *different* types, drop the diagonal before ranking rather
  than noticing it afterwards.

## Default rigor when no criterion is stated

- Genome-wide differential expression: report FDR-corrected results
  (Benjamini-Hochberg), log2 fold changes from the library's published slot
  (`rank_genes_groups` logfoldchanges), and state the threshold used.
- Cross-condition comparisons with replicate donors or samples: aggregate to donor
  pseudobulk (mean per donor) before testing, so n reflects donors, not cells.
- Report effect sizes with their signs anchored to an explicitly named reference group.

## Match normalization to the assay and its processing state

Before normalizing, detect what the matrix already is. Integer-like X means raw counts:
normalize and log-transform. Non-integer X, or ATAC-style QC columns in obs (TSS
enrichment, nucleosome signal, fragment counts), means the values are already processed
(gene activities, normalized expression): do not re-run normalize_total/log1p on them,
and prefer the assay-appropriate representation the file ships (a precomputed obsm
embedding, LSI for ATAC). Re-normalizing processed values distorts every downstream
statistic.

## Resolution choice for max-based metrics

Metrics built from a mean of per-cluster maxima inflate mechanically as clustering gets
finer. Prefer the coarse end of the standard band (resolution near 0.3), run a small
robustness sweep (0.3/0.4/0.5), and report the stable central value rather than the
finest-grained one.

## Small-n comparisons use exact tests

With 5 or fewer biological replicates per group, prefer exact nonparametric tests
(Mann-Whitney) over t-tests, and always state both the test and the replication unit
actually used. Barcodes or cells are never independent replicates for cross-condition
claims -- aggregate to donors or samples first, whatever n that leaves.

## Plausibility tripwires (hard stops)

Stop and redo -- never report -- when: more than half of all tested features come out
"significant"; a fraction lands exactly at 1.0 or 0.0; or two supposedly distinct
populations produce identical outputs. Each of these is a pipeline artifact signature,
not a plausible biological result.

## Detect processing state in its own step, verdict first

Detection of raw-vs-processed is its own execution step ending in a one-line printed
verdict -- "RAW -> will normalize" or "PROCESSED -> skipping normalize_total/log1p" --
BEFORE any pipeline code is written. A task's "normalize" verb means ENSURE the data
is normalized: on already-processed input, skipping re-normalization IS the correct
execution of that instruction, not disobedience. Detection code that runs inside the
same step as an unconditional normalization chain changes nothing and is ritual.

## Sweeps are decision rules, not decoration

The "stable central value" rule applies ONLY to clustering-resolution sweeps that
show a plateau -- never the first value that falls inside an accepted range. A
gate-strictness or threshold sweep is usually MONOTONE and has no center: its
midpoint is an arbitrary number, not an answer. Choose such a threshold from a
second evidence axis independent of the sweep (a QC convention, a physical scale, a
marker prior) and name it; a small strict-gate fraction is a legitimate result, not
a degenerate one. If all sweep values cluster tightly outside expectation, the
answer is pipeline-limited, not resolution-limited: revisit the upstream choices
instead of submitting.

## Artifact rates are not membership calls

Quantifying an artifact (doublets, ambient contamination, segmentation errors) is not
the same task as calling members of a population. The raw marker co-expression
fraction is the SYMPTOM under adjudication: report it only as an upper bound, never
as the answer. Conjoin at least two evidence axes INDEPENDENT of that co-expression
(a dedicated detection method; physical inflation in area and counts against
marker-negative singletons of the same type; spatial adjacency), so the estimate
lands strictly below the raw fraction and is reproduced by two structurally different
criteria. Never drop a discriminating conjunct merely because it makes the estimate
small -- a small nonzero rate is a legitimate answer.
