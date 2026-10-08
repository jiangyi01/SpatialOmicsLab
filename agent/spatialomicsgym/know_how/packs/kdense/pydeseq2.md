# PyDESeq2

## Metadata

**Short Description**: Differential gene expression analysis for bulk RNA-seq with PyDESeq2, including formulaic designs, Wald tests, FDR correction, LFC shrinkage, and result visualization.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/pydeseq2/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: not stated upstream for the tool; the upstream frontmatter `license` field reads "MIT license", and upstream uses that field for the skill text in some files and for the tool in others, so it is not taken as the tool's licence
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 1 install fence(s) replaced by a provisioning pointer; 12 upstream script/reference path(s) marked as not vendored; 1 fence(s) left with nothing runnable replaced by a pointer; 1 manifest replacement(s) and 0 excision(s) applied.

---

## Overview

PyDESeq2 is a Python implementation of DESeq2 for differential expression analysis with bulk RNA-seq data. Design and execute complete workflows from data loading through result interpretation, including formulaic single-factor and multi-factor designs, Wald tests with multiple testing correction, optional apeGLM shrinkage, and integration with pandas and AnnData.

## When to Use This Skill

This skill should be used when:
- Analyzing bulk RNA-seq count data for differential expression
- Comparing gene expression between experimental conditions (e.g., treated vs control)
- Performing multi-factor designs accounting for batch effects or covariates
- Converting R-based DESeq2 workflows to Python
- Integrating differential expression analysis into Python-based pipelines
- Users mention "DESeq2", "differential expression", "RNA-seq analysis", or "PyDESeq2"

## Quick Start Workflow

For users who want to perform a standard differential expression analysis:

```python
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

# 1. Load data
counts_df = pd.read_csv("counts.csv", index_col=0).T  # Transpose to samples × genes
metadata = pd.read_csv("metadata.csv", index_col=0)

# 2. Filter low-count genes
genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
counts_df = counts_df[genes_to_keep]

# 3. Make the reference level explicit and fit DESeq2
metadata["condition"] = pd.Categorical(
    metadata["condition"], categories=["control", "treated"]
)
inference = DefaultInference(n_cpus=4)
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~condition",
    refit_cooks=True,
    inference=inference,
)
dds.deseq2()

# 4. Perform statistical testing
ds = DeseqStats(
    dds,
    contrast=["condition", "treated", "control"],
    inference=inference,
)
ds.summary()

# 5. Access results
results = ds.results_df
significant = results[results.padj < 0.05]
print(f"Found {len(significant)} significant genes")
```

## Core Workflow Steps

The six steps, with code, are in
(upstream reference file, not included) (upstream reference file, not included):

1. **Data preparation** — raw integer counts with genes as columns and samples as rows,
   and matching metadata. Never feed normalized or transformed values to DESeq2.
2. **Design specification** — the design factors and the reference level for each.
3. **DESeq2 fitting** — size factors, dispersions, and the GLM fit.
4. **Statistical testing** — Wald tests for a named contrast.
5. **Optional LFC shrinkage** — for ranking and visualization.
6. **Result export** — the results table with adjusted p-values.

Multi-factor designs, contrasts, and interaction terms are in
(upstream reference file, not included) (upstream reference file, not included).

## Using the Analysis Script

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

**Script features:**
- Automatic data loading and validation
- Gene and sample filtering
- Complete DESeq2 pipeline execution
- Statistical testing with customizable parameters
- Result export (CSV and portable AnnData/H5AD)
- Explicit LFC shrinkage coefficient support for PyDESeq2 0.5.x
- Optional visualization (volcano and MA plots)

Refer users to (upstream helper script, not vendored) when they need a standalone analysis tool or want to batch process multiple datasets.

## Result Interpretation

### Identifying Significant Genes

```python
# Filter by adjusted p-value
significant = ds.results_df[ds.results_df.padj < 0.05]

# Filter by both significance and effect size
sig_and_large = ds.results_df[
    (ds.results_df.padj < 0.05) &
    (abs(ds.results_df.log2FoldChange) > 1)
]

# Separate up- and down-regulated
upregulated = significant[significant.log2FoldChange > 0]
downregulated = significant[significant.log2FoldChange < 0]

print(f"Upregulated: {len(upregulated)}")
print(f"Downregulated: {len(downregulated)}")
```

### Ranking and Sorting

```python
# Sort by adjusted p-value
top_by_padj = ds.results_df.sort_values("padj").head(20)

# Sort by absolute fold change (use shrunk values)
ds.lfc_shrink(coeff="condition[T.treated]")
ds.results_df["abs_lfc"] = abs(ds.results_df.log2FoldChange)
top_by_lfc = ds.results_df.sort_values("abs_lfc", ascending=False).head(20)

# Sort by a combined metric
ds.results_df["score"] = -np.log10(ds.results_df.padj) * abs(ds.results_df.log2FoldChange)
top_combined = ds.results_df.sort_values("score", ascending=False).head(20)
```

### Quality Metrics

```python
# Check normalization (size factors should be close to 1)
print("Size factors:", dds.obs["size_factors"])

# Examine dispersion estimates
import matplotlib.pyplot as plt
plt.hist(dds.var["dispersions"], bins=50)
plt.xlabel("Dispersion")
plt.ylabel("Frequency")
plt.title("Dispersion Distribution")
plt.show()

# Check p-value distribution (should be mostly flat with peak near 0)
plt.hist(ds.results_df.pvalue.dropna(), bins=50)
plt.xlabel("P-value")
plt.ylabel("Frequency")
plt.title("P-value Distribution")
plt.show()
```

## Visualization Guidelines

### Volcano Plot

Visualize significance vs effect size:

```python
import matplotlib.pyplot as plt
import numpy as np

results = ds.results_df.copy()
results["-log10(padj)"] = -np.log10(results.padj)

plt.figure(figsize=(10, 6))
significant = results.padj < 0.05

plt.scatter(
    results.loc[~significant, "log2FoldChange"],
    results.loc[~significant, "-log10(padj)"],
    alpha=0.3, s=10, c='gray', label='Not significant'
)
plt.scatter(
    results.loc[significant, "log2FoldChange"],
    results.loc[significant, "-log10(padj)"],
    alpha=0.6, s=10, c='red', label='padj < 0.05'
)

plt.axhline(-np.log10(0.05), color='blue', linestyle='--', alpha=0.5)
plt.xlabel("Log2 Fold Change")
plt.ylabel("-Log10(Adjusted P-value)")
plt.title("Volcano Plot")
plt.legend()
plt.savefig("volcano_plot.png", dpi=300)
```

### MA Plot

Show fold change vs mean expression:

```python
plt.figure(figsize=(10, 6))

plt.scatter(
    np.log10(results.loc[~significant, "baseMean"] + 1),
    results.loc[~significant, "log2FoldChange"],
    alpha=0.3, s=10, c='gray'
)
plt.scatter(
    np.log10(results.loc[significant, "baseMean"] + 1),
    results.loc[significant, "log2FoldChange"],
    alpha=0.6, s=10, c='red'
)

plt.axhline(0, color='blue', linestyle='--', alpha=0.5)
plt.xlabel("Log10(Base Mean + 1)")
plt.ylabel("Log2 Fold Change")
plt.title("MA Plot")
plt.savefig("ma_plot.png", dpi=300)
```

## Troubleshooting Common Issues

### Data Format Problems

**Issue:** "Index mismatch between counts and metadata"

**Solution:** Ensure sample names match exactly
```python
print("Counts samples:", counts_df.index.tolist())
print("Metadata samples:", metadata.index.tolist())

# Take intersection if needed
common = counts_df.index.intersection(metadata.index)
counts_df = counts_df.loc[common]
metadata = metadata.loc[common]
```

**Issue:** "All genes have zero counts"

**Solution:** Check if data needs transposition
```python
print(f"Counts shape: {counts_df.shape}")
# If genes > samples, transpose is needed
if counts_df.shape[1] < counts_df.shape[0]:
    counts_df = counts_df.T
```

### Design Matrix Issues

**Issue:** "Design matrix is not full rank"

**Cause:** Confounded variables (e.g., all treated samples in one batch)

**Solution:** Remove confounded variable or add interaction term
```python
# Check confounding
print(pd.crosstab(metadata.condition, metadata.batch))

# Either simplify design or add interaction
design = "~condition"  # Remove batch
# OR
design = "~condition + batch + condition:batch"  # Model interaction
```

### No Significant Genes

**Diagnostics:**
```python
# Check dispersion distribution
plt.hist(dds.var["dispersions"], bins=50)
plt.show()

# Check size factors
print(dds.obs["size_factors"])

# Look at top genes by raw p-value
print(ds.results_df.nsmallest(20, "pvalue"))
```

**Possible causes:**
- Small effect sizes
- High biological variability
- Insufficient sample size
- Technical issues (batch effects, outliers)

## Reference Documentation

For comprehensive details beyond this workflow-oriented guide:

- **API Reference** ((upstream reference file, not included)): Complete documentation of PyDESeq2 classes, methods, and data structures. Use when needing detailed parameter information or understanding object attributes.

- **Workflow Guide** ((upstream reference file, not included)): In-depth guide covering complete analysis workflows, data loading patterns, multi-factor designs, troubleshooting, and best practices. Use when handling complex experimental designs or encountering issues.

Load these references into context when users need:
- Detailed API documentation: (upstream reference file, not included)
- Comprehensive workflow examples: (upstream reference file, not included)
- Troubleshooting guidance: (upstream reference file, not included) (see Troubleshooting section)

## Key Reminders

1. **Data orientation matters:** Count matrices typically load as genes × samples but need to be samples × genes. Always transpose with `.T` if needed.

2. **Sample filtering:** Remove samples with missing metadata before analysis to avoid errors.

3. **Gene filtering:** Filter low-count genes (e.g., < 10 total reads) to improve power and reduce computational time.

4. **Design formula order:** Put adjustment variables before the variable of interest (e.g., `"~batch + condition"` not `"~condition + batch"`).

5. **LFC shrinkage timing:** Apply shrinkage after statistical testing and only for visualization/ranking purposes. P-values remain based on unshrunken estimates.

6. **Result interpretation:** Use `padj < 0.05` for significance, not raw p-values. The Benjamini-Hochberg procedure controls false discovery rate.

7. **Contrast specification:** The format is `[variable, test_level, reference_level]` where test_level is compared against reference_level.

8. **Save intermediate objects:** Prefer `dds.to_picklable_anndata().write_h5ad("dds_result.h5ad")` for portable outputs. Only load pickle files that you created yourself and trust.

## Installation and Requirements

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

**System requirements:**
- Python 3.11+
- PyDESeq2 0.5.4 -- PyDESeq2 API as checked against the v0.5.4 source: since 0.5.2 size factors are `dds.obs["size_factors"]` and dispersions `dds.var["dispersions"]` (0.5.1 and earlier kept them in `obsm`/`varm`); since 0.5.0 `DeseqStats` requires `contrast` and `lfc_shrink(coeff=...)` requires the design-column name (e.g. `"condition[T.treated]"`); `DeseqDataSet(n_cpus=...)` and `inference=DefaultInference(n_cpus=...)` are both accepted.
- pandas 2.2.0+
- numpy 2.0.0+
- scipy 1.12.0+
- scikit-learn 1.4.0+
- anndata 0.11.0+
- formulaic 1.0.2+ and formulaic-contrasts 0.2.0+

**Optional for visualization:**
- matplotlib
- seaborn

## Additional Resources

- **Official Documentation:** https://pydeseq2.readthedocs.io
- **GitHub Repository:** https://github.com/scverse/PyDESeq2
- **Publication:** Muzellec et al. (2023) Bioinformatics, DOI: 10.1093/bioinformatics/btad547
- **Original DESeq2 (R):** Love et al. (2014) Genome Biology, DOI: 10.1186/s13059-014-0550-8
