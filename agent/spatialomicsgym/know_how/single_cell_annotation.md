# Single Cell RNA-seq Cell Type Annotation

---

## Metadata

**Short Description**: Best practices for annotating cell types in single-cell RNA-seq data using marker-based, automated, and reference-based approaches.

**Authors**: Distilled from "Single-cell best practices" by Luecken, M.D. et al.

**Affiliations**: Helmholtz Munich, Wellcome Sanger Institute, Harvard Medical School, and contributors

**Version**: 1.0

**Last Updated**: January 2025

**License**: CC BY 4.0

**Commercial Use**: ✅ Allowed

**Source**: https://www.sc-best-practices.org/cellular_structure/annotation.html

**Citation**: Luecken, M.D., Theis, F.J. et al. (2023). Current best practices in single-cell RNA-seq analysis: a tutorial. Molecular Systems Biology.

---

## Overview

Cell type annotation is the process of assigning cell type labels to clusters or individual cells in single-cell RNA-seq data. This guide covers three main approaches and their practical implementation.

## Three Annotation Approaches

### 1. Manual Marker-Based Annotation
Identify cell types by examining expression of known marker genes in each cluster.

**Tools**: Scanpy, Seurat
**Best for**: Small datasets, novel cell types, high confidence needs

### 2. Automated Annotation
Use pre-trained classifiers to automatically assign cell type labels.

**Tools**: CellTypist, scAnnotate
**Best for**: Standard tissues, quick preliminary annotation, large datasets

### 3. Reference-Based Label Transfer
Transfer labels from annotated reference datasets to your query data.

**Tools**: scArches, scANVI, Azimuth, SingleR
**Best for**: Well-characterized tissues, integration with public data

## Recommended Workflow

### Step 1: Quality Control First
- **Remove low-quality cells before annotation**
- Filter doublets (expected doublet rate: 0.8% per 1000 cells)
- Check for ambient RNA contamination
- Verify cluster quality and resolution

### Step 2: Initial Marker-Based Assessment

```python
# Scanpy example
import scanpy as sc

# Calculate marker genes for clusters
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

# Visualize top markers
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

# Plot known markers
markers = {
    'T cells': ['CD3D', 'CD3E', 'CD4', 'CD8A'],
    'B cells': ['CD19', 'MS4A1', 'CD79A'],
    'Monocytes': ['CD14', 'FCGR3A', 'LYZ'],
    'NK cells': ['NCAM1', 'NKG7', 'GNLY']
}

sc.pl.dotplot(adata, markers, groupby='leiden')
```

### Step 3: Use Automated Tools for Validation

```python
# CellTypist example (fast, accurate for immune cells)
import celltypist
from celltypist import models

# Download immune cell model
model = models.Model.load(model='Immune_All_Low.pkl')

# Predict cell types
predictions = celltypist.annotate(adata, model=model, majority_voting=True)
adata = predictions.to_adata()
```

### Step 4: Reference-Based Refinement

```python
# scArches example for label transfer
import scarches as sca

# Load pre-trained reference model
model = sca.models.SCANVI.load_query_data(
    adata=adata,  # Your query data
    reference_model="path/to/reference_model"
)

# Transfer labels
model.train(max_epochs=100)
adata.obs['transferred_labels'] = model.predict()
```

## Best Practices

### Do's:
1. **Always combine multiple approaches** - Use marker-based validation even with automated tools
2. **Check cluster purity** - Ensure clusters represent single cell types
3. **Validate with multiple marker sets** - Don't rely on single markers
4. **Consider biological context** - Tissue type, disease state, developmental stage
5. **Document confidence levels** - Note uncertain annotations
6. **Use hierarchical annotation** - Broad categories first, then subtypes

### Don'ts:
1. **Don't over-cluster** - Too fine resolution creates artificial distinctions
2. **Don't ignore batch effects** - Correct before annotation
3. **Don't trust automation blindly** - Always validate predictions
4. **Don't mix cell states with cell types** - Activated vs. resting cells are states, not types
5. **Don't annotate low-quality cells** - Remove them first

## Common Pitfalls

### 1. Doublet Clusters
**Problem**: Clusters with markers from multiple cell types
**Solution**: Use doublet detection tools (Scrublet, DoubletFinder) before annotation

### 2. Ambient RNA
**Problem**: Background markers in all cells
**Solution**: Use SoupX or CellBender for decontamination

### 3. Over-interpretation of Small Clusters
**Problem**: Rare clusters may be technical artifacts
**Solution**: Require minimum cell count (e.g., >25 cells), validate with independent data

### 4. Reference Mismatch
**Problem**: Reference from different tissue/species/condition
**Solution**: Use tissue-matched references, check marker overlap before transfer

## Scoring Clusters Against a Fixed Label Set

Much annotation work is not open-ended naming. You are handed a closed list of allowed labels -- a
controlled vocabulary, an ontology subset, a collaborator's schema -- and every cluster has to be
assigned from that list. The constraint changes the method, and the changes are easy to miss
because the open-ended habits still feel productive while they quietly decide the answer.

### The tables, without writing the loop

The rules below are implemented in `spatialomicsgym.tool.cluster_comparison`, so they hold by
construction rather than by being remembered at the moment you write the loop:

```python
from spatialomicsgym.tool.cluster_comparison import (
    compare_cluster_abundance,  # which clusters moved between conditions, and in what order
    rank_cluster_markers,       # what each cluster is, from its own top genes
    score_label_vocabulary,     # every label in the list x every cluster, with denominators
)

print(rank_cluster_markers(h5ad, "leiden", n_genes=25))  # one line per cluster; every gene on disk
print(score_label_vocabulary(h5ad, "leiden", vocabulary))  # a dict, or the vocabulary file itself
print(compare_cluster_abundance(h5ad, "leiden", "condition"))
```

`rank_cluster_markers` returns one line per cluster under `markers` -- its top genes, best first,
each as `gene log2fc/score` -- and writes the whole ranked table, with `pval_adj`, to the CSV named
in `csv_path`. Each line shows `genes_shown_per_cluster` genes, fewer than `n_genes` when that many
would not fit in half of one observation; the rest are in that file, so read it rather than ranking
again.

`score_label_vocabulary` reads the vocabulary document directly -- a term list beside a glossary is
understood, including markers written in the `SYMBOL+` style glossaries use -- so the label set is
never retyped and a label cannot go missing between the file and the table. Labels it cannot score
come back in `unevidenceable_terms` with the reason, which is the honest answer to "is this label
absent?" when the assay cannot settle it.

Read the tables together before naming anything. `compare_cluster_abundance` says which clusters
moved; `score_label_vocabulary` says how confidently each is named, through `best_score` and the
`margin` to the runner-up. A cluster that moved a great deal but is named weakly -- a low or
negative `best_score`, a margin near zero, one usable marker -- is a cluster you have not
identified, and reporting its label as a finding claims more than the evidence supports. Rank on
both, and state which of the two decided each call.

One label shape falls outside marker scoring entirely: a label defined by *which populations
co-occur* rather than by genes of its own. It has no marker set to score and will arrive
unevidenceable, correctly. Derive that one from the composition table instead, and say that is what
you did.

The glossary's wording is part of the label's definition, not packaging around its markers, and
`score_label_vocabulary` returns it verbatim as `definition` on every row. Read it. A glossary
usually says more than which genes a label carries: it says which population the label names, and
often how its author classed it -- resident or intruding, specific or catch-all, expected in this
tissue or not. That half decides whether a label answers the question you were actually asked, and
no score can decide it for you. When the question is which populations *changed*, a label whose own
definition describes the normal or expected state is answering a different question however well it
scores, and a label the glossary flags as a catch-all is answering a vaguer one.

Evidence and relevance are different axes, so do not let one stand in for the other. The
best-evidenced label in the list can be the one the glossary marks as the resident baseline, and the
label the question is really about can be one with no markers at all. Rank on the evidence, then
check each candidate's definition against the question before you spend a slot on it.

### Score every label in the list, not the ones that came to mind

Build a score table with one row per label **in the supplied list**, including the labels you are
confident are absent. A shortlist assembled from memory is a prior, not a measurement: it settles
the answer before any data is consulted, and a label it silently omits can never win no matter what
the expression says. Scoring the whole list costs one more loop and turns an invisible omission
into a visible zero.

Keep one table per cluster. Pooling every cluster's scores into a single ranking and reading the
best rows off the top answers a different question -- it tells you which labels scored well
*somewhere*, not which label belongs to which cluster, and it will cheerfully spend two slots on
one label while leaving another cluster unnamed.

### Normalise overlap by the size of the marker set

Raw overlap counts are not comparable across labels, and the bias runs one way: a label you gave
six markers out-scores a label you gave two, on the same cluster, for no reason but the length of
the list. Hand-written marker dictionaries vary in depth by a factor of three or more, so a ranking
built on raw overlap is substantially a ranking of your own dictionary rather than of the tissue.

Use a size-aware score, and print the denominator in its own column so a reader can see it:

```python
import pandas as pd

def score_labels(cluster_markers, marker_sets, detectable=None):
    """One row per label, for a single cluster. Never pool these tables across clusters."""
    present = set(cluster_markers)
    rows = []
    for label, genes in marker_sets.items():
        # A marker the assay cannot measure is not evidence of absence -- drop it from both sides.
        usable = [g for g in genes if detectable is None or g in detectable]
        hits = [g for g in usable if g in present]
        rows.append(
            {
                "label": label,
                "n_markers_supplied": len(genes),
                "n_markers_usable": len(usable),   # the denominator, always shown
                "n_hit": len(hits),
                "fraction_hit": len(hits) / len(usable) if usable else float("nan"),
                "hits": ",".join(hits),
            }
        )
    return pd.DataFrame(rows).sort_values("fraction_hit", ascending=False)
```

`fraction_hit` removes the length bias. It does not remove every bias -- a label whose markers are
all ubiquitously expressed still scores well -- so read the `hits` column before accepting a
winner, and record in one line why the runner-up lost. Where the data allows it, score against a
reference expression profile instead of a hand-written list: the feature set is then defined by the
data and is identical for every label, which removes the problem at its source rather than
correcting for it.

### Genes the assay cannot see

A targeted panel contains only the genes someone chose to put on it, and panels are designed around
the tissue's expected biology. A label whose markers are off-panel scores zero everywhere, which
reads as absence and is really invisibility. Intersect every marker set with the genes actually
present before scoring, and report which labels lost markers to that intersection -- a label left
with one usable marker out of five is a label you have not really tested.

### Labels from separate runs are not the same labels

Cluster 3 in one sample and cluster 3 in another are two unrelated integers. Comparing their
abundances across conditions compares nothing, however neatly the two tables line up on the page.
Either cluster the conditions together so the labels are common by construction, or match the
per-condition clusters to each other by their marker profiles and print the correspondence you
used. Only then does a difference in abundance between conditions mean anything.

### Count the labels in cells before you use them

An assignment table is read one cluster at a time, which hides the only number that matters to
everything downstream: what share of the sample each label ended up holding. A label on a quarter
of the cells is a claim about the tissue, and every count, ranking, co-localisation or enrichment
computed afterwards inherits it -- a label that large will rank near the top of any pairwise
statistic by size alone, whatever the biology.

`score_label_vocabulary` reports this as `assignment_audit.cells_per_term`. Two fields beside the
share say whether it was earned. `thinnest_margin` is the narrowest win behind it: a genuine
population wins its clusters decisively, while a merged or catch-all label wins many clusters by
very little, because it is competing on markers it shares with the finer labels it swallowed.
`weakest_score_won_on` is the other tell, and a different one: a negative score means the label won
a cluster it is *below its own average* in -- that cluster is not positive for it, it simply had no
competitor, and a comfortable margin can hide that completely.

When a label is both large and thinly won, split it before going on. `marker_overlap` names the
candidates directly: where one term's usable markers are a subset of another's
(`share_of_smaller` 1.0), the broader term wins any cluster where both are expressed, so the finer
one can never surface no matter how the clustering is done.

### A label with zero cells is a claim about your clustering first

Reporting that a population is absent is a strong result and is usually wrong. Before making it,
check `assignment_audit.terms_never_assigned`: every term the assay can actually measure that won
no cluster at all, with the cluster it came closest in, its rank there, and how far behind it
finished. Terms with no usable marker are deliberately not in this list -- they are unanswerable
from the panel, not absent, and are reported separately as `unevidenceable_terms`.

Read the rank and the gap, not the zero. A term sitting second by a hair is present and merged into
its neighbour; a term ranked last by a wide margin across every cluster is genuinely not there. The
first case is a finding about resolution and the remedy is to re-cluster: a vocabulary of sixteen
terms cannot be expressed by seven clusters, and at that resolution the smaller populations are
guaranteed zero cells before any marker is scored. Compare the number of terms you were given with
the number of clusters you made, and if there are fewer clusters than plausible populations,
increase the resolution and score again, or sub-cluster only the labels the audit flags as large
and thinly won.

State the remedy you applied. "Absent from this sample" and "not separable at this resolution" are
different answers, and only one of them is about the tissue.

### An ordering is a claim about a tolerance, so report the tolerance with it

`compare_cluster_abundance` renders each cluster's conditions as a string like `treated > control ~ vehicle`.
Every `>` and every `~` in that string is a claim that two numbers are, or are not, far enough
apart to separate — and "far enough" is the `tie_tolerance_pct` argument, not a property of the
data. Two facts follow, and both are easy to get wrong.

The first is that the tolerance must be **yours**, not the default's. If the task, a schema, or a
notation legend states when two values count as tied, pass that number as `tie_tolerance_pct`.
Leaving it unset does not mean "no rule applies"; it means the rule is
`DEFAULT_TIE_TOLERANCE_PCT`, chosen without reference to your task. The output reports
`tie_tolerance_source` as `caller` or `default` so you can tell which happened — if it says
`default` and your task stated a rule, the orderings are answering a question nobody asked.

The second is that the string flattens away how firm each relation is. Under a 5-point rule a gap
of 0.2 points and a gap of 4.9 points both render `~`, and 5.1 points renders `>` — the reader
cannot tell a settled call from one that would reverse under a slightly different rule. So read
`ordering_margins` next to the string. Each row gives the two fractions, the `gap_pct_points`
between them, the `relation` that gap produced, and `margin_to_flip_pct_points` — how far the gap
sits from the tolerance, which is the entire margin of safety in one number. The per-cluster
`narrowest_margin_to_flip_pct_points` is the weakest link, and
`orderings_resting_on_the_narrowest_calls` lists the shakiest orderings first.

Compute the relation; do not eyeball it. The failure this guards against is not forgetting that
ties exist — it is knowing the rule, computing the two fractions, and then writing the ordering
out by hand with a strict `>` because one number was visibly larger than the other. Visibly
larger and larger-by-more-than-the-tolerance are different claims, and only the second is what
a tie rule asks for. When you state an ordering, state the gaps that produced it.

### When a vocabulary is larger than the answer, most of it is there to be declined

A supplied vocabulary is rarely a list of equally plausible labels. It usually mixes the terms
that describe the sample with near-synonyms of each other, generic catch-alls, and terms drawn
from tissues or taxonomies the sample cannot contain. A glossary entry that says a term is a
"candidate synonym for" another, or a "generic term", or belongs to a system your sample is not
from, is telling you the cost of choosing it.

So when a task caps how many items you may report, treat each slot as scarce and ask of every
candidate what it would assert that the others do not. A term that is merely true of your data —
a broad lineage that is obviously present, a catch-all that nothing contradicts — spends a slot
without making a claim. Prefer the term whose glossary entry names a specific, checkable
condition of the sample, and say in your diagnostics which candidates you declined and why.
Declining a term on the record is evidence; omitting it silently is indistinguishable from never
having considered it.

### When the answer space is enumerated, answer from the enumeration

Some tasks hand you a controlled vocabulary: a fixed list of the strings a field is allowed to
take. That list is not a style guide and it is not a set of examples. It is the set of admissible
answers, and a value outside it is not a differently-worded answer — it is a non-answer, scored
the same as a blank.

So make membership a computed step, not an intention. Before you emit a field whose vocabulary
was supplied, check the exact string you are about to write against the list, and print the check:

```python
legal = vocab["<the field's vocabulary key>"]          # the list you were handed, verbatim
print(repr(value), "in vocabulary:", value in legal)
if value not in legal:
    raise SystemExit(f"{value!r} is not one of the {len(legal)} allowed values: {legal}")
```

Three ways a value that looks right fails that check, all of them silent if you skip it.

The first is assembly. A string you built by joining, concatenating or appending is not a member
merely because a member is inside it. `"A > B"` is an answer; `"A > B < A > B"` contains that
answer twice and is not one. This happens when a value is written once while reasoning and again
while composing, and the two copies end up in the same field. Compare the whole field to the
list — never ask whether the legal value appears somewhere in what you wrote, because in exactly
this failure it does.

The second is narration. `"A > B, driven by the higher fraction in A"` is a sentence about the
answer, not the answer. If you want to report the reasoning, the diagnostics are where it goes;
the field takes the bare member.

The third is near-miss spelling: different spacing, a different separator, a synonym, a plural, a
different case. The list is literal. Copy the member out of the list rather than retyping it, and
the whole class disappears.

Operand order is part of the spelling, and it is the one that catches people who are otherwise
careful. Where a member expresses a relation between named things — `"A ~ B > C"` — the list has
chosen an order for the operands, including for the ones the relation says are equivalent. `"B ~ A
> C"` makes the identical claim and is still not a member, because equivalence is a property of
the two values and not of the two positions in the string. So do not derive the member from your
result and write it out; derive the member, then find the row in the list that means it and emit
that row.

If your computed value is genuinely not in the list, that is information, not an obstacle to
route around. Either the list encodes a distinction your analysis did not make — in which case go
back and make it — or your value is a restatement of a member, in which case emit the member.
Padding a non-member into the field so that something is there converts a recoverable near-miss
into a zero.

### 5. Scoring Against a Marker List You Wrote Yourself
**Problem**: Overlap counts against hand-written marker sets rank the dictionary, not the tissue --
longer sets win, and off-panel markers make real cell types look absent.
**Solution**: Normalise by usable set size, intersect with detectable genes first, print both
counts, and prefer a reference profile where one exists.

### 6. Answering From a Truncated Table
**Problem**: A long table is cut off before it is read, and the unseen rows are assumed to resemble
the visible ones.
**Solution**: The rows you were shown are a prefix, not a sample. Write the table to disk and read
back the rows you need, or aggregate first and print the aggregate. Never state a conclusion about
a group whose rows you have not actually seen.

## Tool Selection Guide

| Scenario | Recommended Tool | Why |
|----------|------------------|-----|
| Immune cells (human) | CellTypist | Pre-trained on large immune atlases |
| Mouse tissues | scArches + Mouse Cell Atlas | Comprehensive mouse reference |
| Novel cell types | Manual + Scanpy/Seurat | Need domain expertise |
| Large datasets (>100k cells) | CellTypist | Fast, scalable |
| Cross-species | Manual markers | Limited reference transfer |
| Developmental data | scArches | Handles continuous states |

## Key Marker Genes by Cell Type

> **Illustrative, not balanced.** These lists are written at different depths -- some cell types get
> four markers, others two -- because they are a reading aid, not a scoring dictionary. Used
> directly as one they import that imbalance straight into the ranking, and the deeper entries win
> on length alone. Normalise by set size (see *Scoring Clusters Against a Fixed Label Set*) or
> replace them with a reference profile.

### Blood/Immune:
- **T cells**: CD3D, CD3E (all T cells); CD4, CD8A (subtypes)
- **B cells**: CD19, MS4A1 (CD20), CD79A
- **Monocytes/Macrophages**: CD14, CD68, LYZ
- **NK cells**: NCAM1 (CD56), NKG7, KLRD1
- **Dendritic cells**: FCER1A, CD1C

### Epithelial:
- **General epithelial**: EPCAM, KRT18, KRT19
- **Lung AT1**: AGER, PDPN
- **Lung AT2**: SFTPC, SFTPA1
- **Intestinal**: VIL1, MUC2

### Stromal:
- **Fibroblasts**: COL1A1, DCN, LUM
- **Endothelial**: PECAM1 (CD31), VWF, CDH5
- **Smooth muscle**: ACTA2, MYH11, TAGLN

## Validation Checklist

- [ ] Cluster purity: >80% cells with same label per cluster
- [ ] Marker consistency: Top DE genes match expected markers
- [ ] Biological plausibility: Expected proportions for tissue type
- [ ] Cross-method agreement: Manual and automated annotations align
- [ ] Reference quality: >70% cells successfully transferred
- [ ] Doublet check: No clusters with multi-lineage markers
- [ ] Documentation: Record confidence levels and uncertain calls

## Resources

### Tools:
- **Scanpy**: https://scanpy.readthedocs.io/
- **CellTypist**: https://www.celltypist.org/
- **scArches**: https://scarches.readthedocs.io/
- **Seurat**: https://satijalab.org/seurat/

### References:
- **PanglaoDB**: Database of marker genes
- **Human Cell Atlas**: Reference datasets

### Pre-trained Models:
- **CellTypist models**: 30+ tissue-specific models
- **Azimuth references**: PBMC, lung, kidney, etc.
- **scArches models**: Multiple tissue references

## Troubleshooting

**Issue**: All clusters look similar
→ Increase clustering resolution, check if data is normalized

**Issue**: Too many small clusters
→ Decrease resolution, merge similar clusters based on markers

**Issue**: Automated tool gives inconsistent results
→ Check input normalization, try multiple tools, fall back to manual

**Issue**: Can't find clear markers for cluster
→ May be transitional state, doublet, or low-quality cells

**Issue**: Reference transfer fails
→ Check batch correction, ensure overlapping gene sets, verify tissue match
