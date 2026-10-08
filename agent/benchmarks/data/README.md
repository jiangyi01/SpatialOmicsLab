# Benchmark Data Organization

The datasets live in `agent/benchmarks/benchmark_data/`, and `agent/benchmarks/benchmark_data/registry.yaml`
is the only index the runners and the evaluator read. Every path in the registry is relative to
that directory.

## Directory Structure

```
benchmark_data/
├── registry.yaml                                  # The dataset index (a top-level `datasets:` list)
├── Visium_for_spatial_domain/                     # One directory per dataset, named <platform>_for_<task>
│   ├── Spatial_data/
│   │   ├── Standard_h5ad/spatial_transcriptomics.h5ad   # Input AND clustering ground truth (in obs)
│   │   └── metadata.csv                         # obs exported for reading; the evaluator does not use it
│   └── ref_info.txt                               # Source and citation
├── Visium_for_spatial_variable_genes/
│   ├── Spatial_data/Standard_h5ad/spatial_transcriptomics.h5ad
│   └── curated_benchmarking_SVGs.csv              # SVG ground truth: a gene list, first column
├── Visium_for_spatial_deconvolution/
│   ├── Spatial_data/Standard_h5ad/spatial_transcriptomics.h5ad      # Input AND per-spot cell-type labels (in obs)
│   └── SingleCell_data/Standard_h5ad/single_cell_transcriptomics.h5ad # The sc reference
├── MERFISH_for_* / Slide-seqV2_standardized/...   # The same shape for the other platforms
└── mini/                                          # Downsampled copies for fast pipeline tests
```

## What the evaluator reads as ground truth

`agent/benchmarks/evaluation/evaluator.py` reads ground truth in exactly these forms. A layout outside
them is not scored -- a proportions table or a per-spot label CSV is not a supported ground truth,
and such a dataset is blocked at the pre-evaluation gate, which reads the truth as an h5ad.

| Task | `ground_truth_path` | Where the truth is |
|---|---|---|
| `spatial_clustering` | an h5ad (usually the input itself) | `obs[metadata.ground_truth_key]`, one label per spot |
| `deconvolution` | an h5ad (usually the input itself) | `obs[metadata.ground_truth_key]`, one cell-type label per spot; the evaluator one-hot encodes it |
| `svg_detection` | a `.csv`/`.tsv`/`.txt` gene list | the first column; `null` means no curated list, scored on Moran's I only |

- `metadata.ground_truth_key` is required for clustering and deconvolution. For clustering, a
  named key that is absent from the h5ad is an error, not a cue to use another column.
- Deconvolution entries also name the single-cell reference: `metadata.sc_reference_path` and
  `metadata.sc_reference_celltype_key`. Its label set decides which cell types a prediction is
  scored on: a reference class the prediction leaves out counts as a predicted 0, and a
  ground-truth class absent from the reference is excluded for every tool.
- When the ground truth is coarser than the reference (Visium lymph node: 4 zones vs 44 reference
  labels), `metadata.celltype_aggregation` names the table in
  `agent/benchmarks/evaluation/celltype_aggregation/` that maps fine labels onto the coarse classes.

## Adding New Datasets

1. Put the spatial h5ad under `benchmark_data/<Dataset>/Spatial_data/Standard_h5ad/`, with the
   ground-truth labels as an `obs` column (clustering, deconvolution), or write the curated gene
   list as a CSV next to it (SVG).
2. Add an entry to `registry.yaml`: `name`, `task_type`, `input_path`, `ground_truth_path`,
   `compatible_tools`, and `metadata.ground_truth_key` (plus the sc reference keys, and
   `celltype_aggregation` if needed, for deconvolution).
3. Run one tool through `agent/benchmarks/workflows/benchmark_runner.py` on it and read the
   `metrics.json`: `_spot_coverage` says how much of the slide was scored, and for clustering
   `_ground_truth_column_matched` says which column it was scored against.
