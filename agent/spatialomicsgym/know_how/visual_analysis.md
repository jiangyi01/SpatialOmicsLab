# Visual Analysis: Choosing a Figure, and Saying What It Shows

## Metadata

**Version**: 1.0
**Short Description**: Choose a figure from what the dataset can support rather than from what was asked for, read the caption as the figure's own account of itself, and never let the answer claim more than the picture does.
**Scope**: How to decide which figure answers a question, how to react when the data cannot support the one that was asked for, and what a figure may and may not be said to show.
**Applies when**: any request to plot, show, visualise, map, compare or export -- and any point where you are about to describe a figure you produced.
**Does not apply to**: choosing an analysis method. This is about reading and presenting a result, not computing one.

---

## Overview

A figure is the part of an answer a reader believes without checking. That is exactly why the
discipline here is about restraint rather than technique: the tools already draw well, and the way
this goes wrong is not an ugly plot, it is a correct-looking plot that means something other than
what the answer says it means.

Four rules, in the order they bite.

## 1. Inspect before you draw

Call `inspect_dataset` first, or `recommend_visualizations` when the question is open. Both are
pure reads: they write nothing, create no directory, and are safe to call while you are still
deciding.

They answer a question you cannot answer by looking at a filename: which slot holds counts, whether
the coordinates are positions or array indices, whether a stored differential-expression result is
complete enough to plot, whether there is an image and a scalefactor to go with it.

Do not skip this because the request named a plot. "Draw me a volcano" is a request for a volcano
*if the data can support one*; if it cannot, the useful answer is which field is missing and which
call would produce it, and that is what the tool returns.

## 2. A refusal is an answer, not an error

When a plotting call comes back refused, it names four things: what is missing, why it matters,
the call that would produce it, and what can be drawn instead. Use all four.

    prerequisite_not_met
      why:      the stored ranking carries names and scores but no fold changes and no p-values
      produced_by: sc.tl.rank_genes_groups(adata, groupby=<column>, method='wilcoxon')
      instead:  de.ranked, markers.dotplot, markers.heatmap

Two responses are wrong here and both are tempting. Do not plot the test statistic on an axis
labelled as an effect size -- a statistic is not an effect size, and no p-value can be recovered
from it without the null it was computed against. And do not report the refusal as a tool failure:
the tool worked, the data does not support the figure, and those are different sentences.

Either run the call it named, or draw what it offered and say which you did.

## 3. The caption is the figure's account of itself

Every figure this platform draws carries a composed caption, and the clauses are ordered by how
badly a reader is misled without them. Read it before you write about the figure.

The clauses that change what the picture means:

- **Which matrix was drawn.** `Values: layers['counts'], log1p of the stored counts` is a different
  figure from `Values: raw.X`. If the caption says the matrix type could not be verified, do not
  describe the values as counts.
- **Whether panels share a scale.** "Each panel has its own colour scale" means brightness is not
  comparable between panels, and an answer comparing two panels' intensity is wrong even though
  the figure is right.
- **What was sampled.** If fewer points were drawn than exist, the caption says so and says how
  they were chosen.
- **Inference.** A communication figure opens "Inferred, not measured". That is not boilerplate:
  the score is co-expression of an annotated pair filtered through a curated database, and it is
  not evidence that the interaction occurs in this tissue.
- **Ordering, not time.** Pseudotime has no units and its direction comes from whichever cell was
  chosen as the root. A pseudotime map on tissue is the same ordering drawn at each spot's
  position; distance across a section is not elapsed time.

**The answer may not claim more than the caption claims.** This is the sentence that matters most
in this document. The renderer's honesty is worth nothing if the prose above the picture overrides
it.

## 4. What the unit of observation is

Cells and spots are not biological replicates. A figure grouping thousands of cells from four
donors into two arms is descriptive; the p-value you would get from treating those cells as
independent is largely a function of sequencing depth.

So: describe differences between groups within a sample, and say the unit. A condition-level claim
needs replicates, and if the dataset does not have at least two per arm, say that rather than
softening the claim into something unfalsifiable.

Words like *enriched*, *significant*, *upregulated* and *depleted* belong to a test that was run.
Under a composition bar chart or a proportion map, use *higher*, *lower*, *more of*, and name the
comparison.

## 5. Few figures, chosen

`run_visualization_pipeline` draws against a budget -- three for an overview, six for a detailed
report -- and then lists what it considered and did not draw, and what needs you to name something.
Read that list: it is usually where the figure the reader actually wanted is hiding, one parameter
away.

Do not ask for everything the catalogue can draw. A dozen near-identical panels buries the two that
mattered, and the chat shows at most four.

## 6. Changing a figure

To change how a figure looks, pass its record to `update_visualization`. Appearance and layout
come back as version two, drawn beside version one -- the earlier figure is never overwritten,
because an earlier turn's picture must not change underneath a reader scrolling back.

A change that selects different values is not a new view of the same figure. It comes back as
`needs_redraw` with the call to make, and that is correct: different values are a different figure,
and both should be able to exist.

## What not to do

- Do not describe a spatial gradient as a trajectory. There is no spatial-trajectory family, and
  its absence is a refusal, not a gap.
- Do not present a figure drawn from a stored result as evidence that you ran the analysis.
- Do not compare panels the caption says are on different scales.
- Do not report a contact sheet's panels as comparable; they were drawn separately, each on its own
  scale.
- Do not call an uncorrected p-value an FDR. When the table carries only a raw p-value, the caption
  says so, and so should you.
- Do not invent a limitation that is not in the caption, either. Over-hedging a sound figure is its
  own kind of inaccuracy.
