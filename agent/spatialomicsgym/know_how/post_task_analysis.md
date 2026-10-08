# Post-Task Visualization and Follow-Up Analysis

## Overview

After a spatial analysis tool writes output files, the scan, the follow-up analysis and the
figures come from one tested library call, not from hand-written code. This document says how to
call it, what it produces, and what to do with what it reports.

This replaces the hand-written 3-phase workflow that used to live here. Writing the plotting code
fresh on every run re-derived the same thirty snippets, and got them subtly wrong in ways that were
invisible in the output: a transposed proportion table plotted with spot IDs as cell-type names, a
constant SVG score plotted as a ranking. Those checks are now in the library, under test.

## The call

One `execute` block, immediately after the tool finishes:

```python
from spatialomicsgym.postanalysis import run_post_analysis

results = run_post_analysis(OUTPUT_DIR, tool_name="the_tool_you_called")
print(results)
```

- `OUTPUT_DIR` is the directory the tool wrote to. A single file works too.
- `tool_name` is optional but worth passing: it lets the engine read the tool's registered output
  profile instead of inferring the task from the data.
- `task_type=` overrides detection. Only pass it when detection is wrong, or for a task the data
  cannot reveal (`alignment` and `imputation` look like any other h5ad).
- It returns the results directory, which by default is `<output_dir>/post_analysis`.
- It does not raise. A missing input, an unreadable file or a failed figure becomes a warning and
  a non-`ok` status, never a traceback.

## What it writes

```
post_analysis/
  manifest.json      what was analysed, findings, figures, tables, warnings, status
  figures/*.png
  tables/*.csv
```

`manifest.json` is the deliverable. Read it and report it:

- `status` -- `ok`, `partial` (something was skipped or failed) or `failed` (no result found).
- `findings` -- the numbers: domain counts, cell-type counts, significant-gene counts, and the
  guard flags `signal_free`, `orientation_transposed`, `celltype_names_resolved`.
- `figures` -- each with a title and a caption. Name them by title when presenting.
- `warnings` -- **quote these to the user verbatim.** They are the whole point of the layer.

## Reading the warnings

A warning is not a formatting nit; it usually means the tool's result cannot be used:

- **Signal-free** -- every spot got the same value, or one domain, or a constant score. The
  numbers carry no information; any metric computed on them is meaningless. Say so plainly and
  suggest re-running with different parameters.
- **Transposed** -- the proportion table was written cell-types-by-spots. The engine fixes the
  orientation, but a downstream file the tool wrote may still be wrong.
- **Latent topics, not cell types** -- columns are `1 2 3` or `topic_4`. LDA-style tools need a
  topic-to-cell-type mapping before the composition means anything.
- **Degenerate threshold** -- more than 90% of genes "significant". The threshold, not the tissue,
  produced that.
- **No coordinates found** -- the spatial maps were skipped. Keeping the spatial h5ad in the
  output directory gets them.

## Per-task coverage

Deep support (task-specific analysis plus figures):

- **spatial_clustering** -- domain sizes, spatial domain map, Wilcoxon markers per domain,
  marker-expression heatmap.
- **deconvolution** -- mean composition, per-spot spread, dominant cell type, co-localization
  correlation, per-cell-type tissue maps.
- **svg_detection** -- significant-gene count, top-gene table, score/p-value distribution, ranked
  bar chart, spatial maps of the top genes.

Shallow support (file scan, an overview figure, and a warning saying what is missing):
`cell_communication`, `alignment`, `imputation`, `trajectory`. For these, adding your own analysis
on top of the manifest is worthwhile.

## Writing your own plots

Only for something the engine did not produce, or that the user asked for by name. Then:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

Save at `dpi=300` with `bbox_inches="tight"`, label the axes, print the path, and interpret the
figure biologically. Before any group-wise scanpy call (`rank_genes_groups`, `dendrogram`,
`matrixplot`), cast an integer label column to string -- scanpy joins category names with `,` and
raises `expected str instance, int found` otherwise:

```python
col = adata.obs[GROUP_KEY]
if col.dtype.name != "category" or col.cat.categories.dtype != "object":
    adata.obs[GROUP_KEY] = col.astype(str).astype("category")
```
