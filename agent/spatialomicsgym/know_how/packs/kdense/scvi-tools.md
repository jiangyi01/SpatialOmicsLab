# scvi-tools

## Metadata

**Short Description**: Direct use of scvi-tools models (scVI, scANVI, DestVI, totalVI, MultiVI, spatial). Not needed for the platform's own deconvolution, integration or annotation tools, which wrap these; pick only when the task names scvi-tools or a model.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/scvi-tools/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: not stated upstream for the tool; the upstream frontmatter `license` field reads "BSD-3-Clause license", and upstream uses that field for the skill text in some files and for the tool in others, so it is not taken as the tool's licence
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 1 install fence(s) replaced by a provisioning pointer; 1 inline install span(s) rewritten; 1 upstream reference file(s) folded in (Spatial Transcriptomics Models); 8 upstream script/reference path(s) marked as not vendored; 1 manifest replacement(s) and 0 excision(s) applied; the Short Description is the manifest's, not the upstream frontmatter's.

---

## Overview

scvi-tools is a comprehensive Python framework for probabilistic models in single-cell genomics. Built on PyTorch and PyTorch Lightning, it provides deep generative models using variational inference for analyzing diverse single-cell data modalities. Current stable release: **scvi-tools 1.4.3** (May 2026).

**Model namespaces matter:** core models (scVI, scANVI, totalVI, MultiVI, PeakVI, AUTOZI, CondSCVI, DestVI, LinearSCVI, AmortizedLDA, JaxSCVI) live under `scvi.model`. Most other models (VeloVI, contrastiveVI, CellAssign, PoissonVI, scBasset, MrVI, MethylVI/MethylANVI, CytoVI, SysVI, Decipher, gimVI, scVIVA, ResolVI, Stereoscope, Solo, totalANVI, DIAGVI) live under `scvi.external`. The reference files specify the correct namespace per model.

## When to Use This Skill

Use this skill when:
- Analyzing single-cell RNA-seq data (dimensionality reduction, batch correction, integration)
- Working with single-cell ATAC-seq or chromatin accessibility data
- Integrating multimodal data (CITE-seq, multiome, paired/unpaired datasets)
- Analyzing spatial transcriptomics data (deconvolution, spatial mapping)
- Performing differential expression analysis on single-cell data
- Conducting cell type annotation or transfer learning tasks
- Working with specialized single-cell modalities (methylation, cytometry, RNA velocity)
- Building custom probabilistic models for single-cell analysis

## Core Capabilities

scvi-tools provides models organized by data modality:

### 1. Single-Cell RNA-seq Analysis
Core models for expression analysis, batch correction, and integration. See (upstream reference file, not included) for:
- **scVI**: Unsupervised dimensionality reduction and batch correction
- **scANVI**: Semi-supervised cell type annotation and integration
- **AUTOZI**: Zero-inflation detection and modeling
- **VeloVI**: RNA velocity analysis
- **contrastiveVI**: Perturbation effect isolation

### 2. Chromatin Accessibility (ATAC-seq)
Models for analyzing single-cell chromatin data. See (upstream reference file, not included) for:
- **PeakVI**: Peak-based ATAC-seq analysis and integration
- **PoissonVI**: Quantitative fragment count modeling
- **scBasset**: Deep learning approach with motif analysis

### 3. Multimodal & Multi-omics Integration
Joint analysis of multiple data types. See (upstream reference file, not included) for:
- **totalVI**: CITE-seq protein and RNA joint modeling
- **totalANVI**: Semi-supervised CITE-seq (totalVI with cell-type labels)
- **MultiVI**: Paired and unpaired multi-omic integration (MuData-based)
- **MrVI**: Multi-resolution cross-sample analysis
- **DIAGVI**: Diagonal integration of unpaired single-cell datasets (added in 1.4.3)

### 4. Spatial Transcriptomics
Spatially-resolved transcriptomics analysis. See (see the section "Reference: Spatial Transcriptomics Models" below) for:
- **DestVI**: Multi-resolution spatial deconvolution
- **Stereoscope**: Cell type deconvolution
- **Tangram**: Spatial mapping and integration
- **scVIVA**: Cell-environment relationship analysis

### 5. Specialized Modalities
Additional specialized analysis tools. See (upstream reference file, not included) for:
- **MethylVI/MethylANVI**: Single-cell methylation analysis
- **CytoVI**: Flow/mass cytometry batch correction
- **Solo**: Doublet detection
- **CellAssign**: Marker-based cell type annotation

## Typical Workflow

All scvi-tools models follow a consistent API pattern:

```python
# 1. Load and preprocess data (AnnData format)
import scvi
import scanpy as sc

adata = scvi.data.heart_cell_atlas_subsampled()
sc.pp.filter_genes(adata, min_counts=3)
sc.pp.highly_variable_genes(adata, n_top_genes=1200)

# 2. Register data with model (specify layers, covariates)
scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",  # Use raw counts, not log-normalized
    batch_key="batch",
    categorical_covariate_keys=["donor"],
    continuous_covariate_keys=["percent_mito"]
)

# 3. Create and train model
model = scvi.model.SCVI(adata)
model.train()

# 4. Extract latent representations and normalized values
latent = model.get_latent_representation()
normalized = model.get_normalized_expression(library_size=1e4)

# 5. Store in AnnData for downstream analysis
adata.obsm["X_scVI"] = latent
adata.layers["scvi_normalized"] = normalized

# 6. Downstream analysis with scanpy
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.umap(adata)
sc.tl.leiden(adata)
```

**Key Design Principles:**
- **Raw counts required**: Models expect unnormalized count data for optimal performance
- **Unified API**: Consistent interface across all models (setup → train → extract)
- **AnnData-centric**: Seamless integration with the scanpy ecosystem
- **GPU acceleration**: Automatic utilization of available GPUs
- **Batch correction**: Handle technical variation through covariate registration

## Common Analysis Tasks

### Differential Expression
Probabilistic DE analysis using the learned generative models:

```python
de_results = model.differential_expression(
    groupby="cell_type",
    group1="TypeA",
    group2="TypeB",
    mode="change",  # Use composite hypothesis testing
    delta=0.25      # Minimum effect size threshold
)
```

See (upstream reference file, not included) for detailed methodology and interpretation.

### Model Persistence
Save and load trained models:

```python
# Save model
model.save("./model_directory", overwrite=True)

# Load model
model = scvi.model.SCVI.load("./model_directory", adata=adata)
```

### Batch Correction and Integration
Integrate datasets across batches or studies:

```python
# Register batch information
scvi.model.SCVI.setup_anndata(adata, batch_key="study")

# Model automatically learns batch-corrected representations
model = scvi.model.SCVI(adata)
model.train()
latent = model.get_latent_representation()  # Batch-corrected
```

## Theoretical Foundations

scvi-tools is built on:
- **Variational inference**: Approximate posterior distributions for scalable Bayesian inference
- **Deep generative models**: VAE architectures that learn complex data distributions
- **Amortized inference**: Shared neural networks for efficient learning across cells
- **Probabilistic modeling**: Principled uncertainty quantification and statistical testing

See (upstream reference file, not included) for detailed background on the mathematical framework.

## Additional Resources

- **Workflows**: (upstream reference file, not included) contains common workflows, best practices, hyperparameter tuning, and GPU optimization
- **Model References**: Detailed documentation for each model category in the (upstream reference file, not included) directory
- **Official Documentation**: https://docs.scvi-tools.org/en/stable/
- **Tutorials**: https://docs.scvi-tools.org/en/stable/tutorials/index.html
- **API Reference**: https://docs.scvi-tools.org/en/stable/api/index.html

## Installation

Requires Python **3.12+** (scvi-tools 1.4 dropped older versions).

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

For reproducible environments, pin a version: (not installed from this document: if the import fails, the library is not available here -- say so, do not install it).

**Compute backends:** training defaults to PyTorch (CPU/GPU/TPU). A JAX backend
(`scvi.model.JaxSCVI`) and an experimental MLX backend for Apple silicon
(`scvi.model.mlxSCVI`) are available for select models.

## Best Practices

1. **Use raw counts**: Always provide unnormalized count data to models
2. **Filter genes**: Remove low-count genes before analysis (e.g., `min_counts=3`)
3. **Register covariates**: Include known technical factors (batch, donor, etc.) in `setup_anndata`
4. **Feature selection**: Use highly variable genes for improved performance
5. **Model saving**: Always save trained models to avoid retraining
6. **GPU usage**: Enable GPU acceleration for large datasets (`accelerator="gpu"`)
7. **Scanpy integration**: Store outputs in AnnData objects for downstream analysis

## Reference: Spatial Transcriptomics Models

This document covers models for analyzing spatially-resolved transcriptomics data in scvi-tools.

### DestVI (Deconvolution of Spatial Transcriptomics using Variational Inference)

**Purpose**: Multi-resolution deconvolution of spatial transcriptomics using single-cell reference data.

**Key Features**:
- Estimates cell type proportions at each spatial location
- Uses single-cell RNA-seq reference for deconvolution
- Multi-resolution approach (global and local patterns)
- Accounts for spatial correlation
- Provides uncertainty quantification

**When to Use**:
- Deconvolving Visium or similar spatial transcriptomics
- Have scRNA-seq reference data with cell type labels
- Want to map cell types to spatial locations
- Interested in spatial organization of cell types
- Need probabilistic estimates of cell type abundance

**Data Requirements**:
- **Spatial data**: Visium or similar spot-based measurements (target data)
- **Single-cell reference**: scRNA-seq with cell type annotations
- Both datasets should share genes

**Basic Usage** (DestVI requires a `CondSCVI` reference model, not plain `SCVI`):
```python
from scvi.model import CondSCVI, DestVI

# Step 1: Train the single-cell LVM (CondSCVI) on the labeled reference
CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key="cell_type")
sc_model = CondSCVI(sc_adata, weight_obs=False)
sc_model.train(max_epochs=300)

# Step 2: Set up the spatial data and build DestVI from the trained reference
DestVI.setup_anndata(spatial_adata, layer="counts")
st_model = DestVI.from_rna_model(spatial_adata, sc_model)
st_model.train(max_epochs=2500)

# Step 3: Get cell type proportions per spot
proportions = st_model.get_proportions()
spatial_adata.obsm["proportions"] = proportions

# Step 4: Get cell type-specific expression
# Expression of genes specific to each cell type at each spot
ct_expression = st_model.get_scale_for_ct("T cells")
```

**Key Parameters**:
- `weight_obs` (CondSCVI): downweight common cell types when training the reference
- `amortization`: Amortization strategy ("both", "latent", "proportion")
- `n_latent`: Latent dimensionality (inherited from the CondSCVI reference)

**Outputs**:
- `get_proportions()`: Cell type proportions at each spot
- `get_scale_for_ct(cell_type)`: Cell type-specific expression patterns
- `get_gamma()`: Proportion-specific gene expression scaling

**Visualization**:
```python
import scanpy as sc
import matplotlib.pyplot as plt

# Visualize specific cell type proportions spatially
sc.pl.spatial(
    spatial_adata,
    color="T cells",  # If proportions added to .obs
    spot_size=150
)

# Or use obsm directly
for ct in cell_types:
    plt.figure()
    sc.pl.spatial(
        spatial_adata,
        color=spatial_adata.obsm["proportions"][ct],
        title=f"{ct} proportions"
    )
```

### Stereoscope

**Purpose**: Cell type deconvolution for spatial transcriptomics using probabilistic modeling.

**Key Features**:
- Reference-based deconvolution
- Probabilistic framework for cell type proportions
- Works with various spatial technologies
- Handles gene selection and normalization

**When to Use**:
- Similar to DestVI but simpler approach
- Deconvolving spatial data with reference
- Faster alternative for basic deconvolution

**Basic Usage** (Stereoscope is split into two `scvi.external` models):
```python
from scvi.external import RNAStereoscope, SpatialStereoscope

# Train the single-cell reference model
RNAStereoscope.setup_anndata(sc_adata, labels_key="cell_type", layer="counts")
sc_model = RNAStereoscope(sc_adata)
sc_model.train(max_epochs=100)

# Build the spatial model from the trained reference
SpatialStereoscope.setup_anndata(spatial_adata, layer="counts")
spatial_model = SpatialStereoscope.from_rna_model(spatial_adata, sc_model)
spatial_model.train(max_epochs=2000)

# Get cell-type proportions per spot
proportions = spatial_model.get_proportions()
```

### Tangram

**Purpose**: Spatial mapping and integration of single-cell data to spatial locations.

**Key Features**:
- Maps single cells to spatial coordinates
- Learns optimal transport between single-cell and spatial data
- Gene imputation at spatial locations
- Cell type mapping

**When to Use**:
- Mapping cells from scRNA-seq to spatial locations
- Imputing unmeasured genes in spatial data
- Understanding spatial organization at single-cell resolution
- Integrating scRNA-seq and spatial transcriptomics

**Data Requirements**:
- Single-cell RNA-seq data with annotations
- Spatial transcriptomics data
- Shared genes between modalities

> scvi-tools wraps the Tangram method as `scvi.external.Tangram`. The example
> below uses the standalone `tangram-sc` package, whose helper functions
> (annotation/gene projection) are convenient for downstream analysis.

**Basic Usage**:
```python
import tangram as tg

# Map cells to spatial locations
ad_map = tg.map_cells_to_space(
    adata_sc=sc_adata,
    adata_sp=spatial_adata,
    mode="cells",  # or "clusters" for cell type mapping
    density_prior="rna_count_based"
)

# Get mapping matrix (cells × spots)
mapping = ad_map.X

# Project cell annotations to space
tg.project_cell_annotations(
    ad_map,
    spatial_adata,
    annotation="cell_type"
)

# Impute genes in spatial data
genes_to_impute = ["CD3D", "CD8A", "CD4"]
tg.project_genes(ad_map, spatial_adata, genes=genes_to_impute)
```

**Visualization**:
```python
# Visualize cell type mapping
sc.pl.spatial(
    spatial_adata,
    color="cell_type_projected",
    spot_size=100
)
```

### gimVI (Gaussian Identity Multivi for Imputation)

**Purpose**: Cross-modality imputation between spatial and single-cell data.

**Key Features**:
- Joint model of spatial and single-cell data
- Imputes missing genes in spatial data
- Enables cross-dataset queries
- Learns shared representations

**When to Use**:
- Imputing genes not measured in spatial data
- Joint analysis of spatial and single-cell datasets
- Mapping between modalities

**Basic Usage**:
```python
# gimVI (scvi.external) jointly models a single-cell and a spatial dataset
scvi.external.GIMVI.setup_anndata(sc_adata, layer="counts")
scvi.external.GIMVI.setup_anndata(spatial_adata, layer="counts")

model = scvi.external.GIMVI(sc_adata, spatial_adata)
model.train()

# Per-dataset latent representations and imputed expression
sc_latent, spatial_latent = model.get_latent_representation()
_, imputed_spatial = model.get_imputed_values(normalized=True)
```

### scVIVA (Variation in Variational Autoencoders for Spatial)

**Purpose**: Analyzing cell-environment relationships in spatial data.

**Key Features**:
- Models cellular neighborhoods and environments
- Identifies environment-associated gene expression
- Accounts for spatial correlation structure
- Cell-cell interaction analysis

**When to Use**:
- Understanding how spatial context affects cells
- Identifying niche-specific gene programs
- Cell-cell interaction studies
- Microenvironment analysis

**Data Requirements**:
- Spatial transcriptomics with coordinates
- Cell type annotations (optional)

**Basic Usage** (scVIVA lives in `scvi.external`):
```python
# scVIVA jointly models each cell and its niche. Precompute neighborhood
# composition / embeddings first, then register the resulting keys.
scvi.external.SCVIVA.preprocess_anndata(
    spatial_adata,
    sample_key="sample",
    labels_key="cell_type",
    cell_coordinates_key="spatial",  # coordinates in .obsm
)

scvi.external.SCVIVA.setup_anndata(
    spatial_adata,
    labels_key="cell_type",
    sample_key="sample",
)

model = scvi.external.SCVIVA(spatial_adata)
model.train()

# Latent representation capturing cell state in its spatial context
spatial_adata.obsm["X_scVIVA"] = model.get_latent_representation()
```

See the [scVIVA user guide](https://docs.scvi-tools.org/en/stable/user_guide/models/scviva.html)
for the exact preprocessing keys and niche/environment outputs.

### ResolVI

**Purpose**: Addressing spatial transcriptomics noise through resolution-aware modeling.

**Key Features**:
- Accounts for spatial resolution effects
- Denoises spatial data
- Multi-scale analysis
- Improves downstream analysis quality

**When to Use**:
- Noisy spatial data
- Multiple spatial resolutions
- Need denoising before analysis
- Improving data quality

**Basic Usage** (ResolVI lives in `scvi.external`):
```python
scvi.external.RESOLVI.setup_anndata(
    spatial_adata,
    layer="counts",
)

model = scvi.external.RESOLVI(spatial_adata)
model.train()

# Denoised / corrected expression
denoised = model.get_normalized_expression()
```

### Model Selection for Spatial Transcriptomics

#### DestVI
**Choose when**:
- Need detailed deconvolution with reference
- Have high-quality scRNA-seq reference
- Want multi-resolution analysis
- Need uncertainty quantification

**Best for**: Visium, spot-based technologies

#### Stereoscope
**Choose when**:
- Need simpler, faster deconvolution
- Basic cell type proportion estimates
- Limited computational resources

**Best for**: Quick deconvolution tasks

#### Tangram
**Choose when**:
- Want single-cell resolution mapping
- Need to impute many genes
- Interested in cell positioning
- Optimal transport approach preferred

**Best for**: Detailed spatial mapping

#### gimVI
**Choose when**:
- Need bidirectional imputation
- Joint modeling of spatial and single-cell
- Cross-dataset queries

**Best for**: Integration and imputation

#### scVIVA
**Choose when**:
- Interested in cellular environments
- Cell-cell interaction analysis
- Neighborhood effects

**Best for**: Microenvironment studies

#### ResolVI
**Choose when**:
- Data quality is a concern
- Need denoising
- Multi-scale analysis

**Best for**: Noisy data preprocessing

### Complete Workflow: Spatial Deconvolution with DestVI

```python
import scvi
import scanpy as sc
import squidpy as sq

# ===== Part 1: Load both datasets; both models read raw counts =====
sc_adata = sc.read_h5ad("reference_scrna.h5ad")
sc_adata.layers["counts"] = sc_adata.X.copy()  # X must hold raw counts here
spatial_adata = sc.read_visium("path/to/visium")
spatial_adata.var_names_make_unique()
spatial_adata.layers["counts"] = spatial_adata.X.copy()  # read_visium creates no layers

# QC and filtering (seurat_v3 selects HVGs on raw counts; it needs scikit-misc)
sc.pp.filter_genes(sc_adata, min_cells=10)
sc.pp.filter_genes(spatial_adata, min_cells=10)
sc.pp.highly_variable_genes(sc_adata, n_top_genes=4000, flavor="seurat_v3", layer="counts", subset=True)

# DestVI.from_rna_model reuses the reference decoder, so both objects need the same genes in
# the same order: keep the reference HVGs that the spatial data also measured
genes = sc_adata.var_names.intersection(spatial_adata.var_names)
sc_adata = sc_adata[:, genes].copy()
spatial_adata = spatial_adata[:, genes].copy()

# ===== Part 2: Train the CondSCVI reference model (labels_key carries cell types) =====
from scvi.model import CondSCVI, DestVI

CondSCVI.setup_anndata(
    sc_adata,
    layer="counts",
    labels_key="cell_type",
)

sc_model = CondSCVI(sc_adata, weight_obs=False)
sc_model.train(max_epochs=300)

# ===== Part 3: Run DestVI =====
DestVI.setup_anndata(
    spatial_adata,
    layer="counts"
)

destvi_model = DestVI.from_rna_model(
    spatial_adata,
    sc_model,
)

destvi_model.train(max_epochs=2500)

# ===== Part 4: Extract results =====
# get_proportions() returns a DataFrame: one column per cell type, indexed like spatial_adata.obs
proportions = destvi_model.get_proportions()
spatial_adata.obsm["proportions"] = proportions

# Add proportions to .obs for easy plotting
for ct in proportions.columns:
    spatial_adata.obs[f"prop_{ct}"] = proportions[ct].values

# ===== Part 5: Visualization =====
# Plot specific cell types
cell_types = ["T cells", "B cells", "Macrophages"]

for ct in cell_types:
    sc.pl.spatial(
        spatial_adata,
        color=f"prop_{ct}",
        title=f"{ct} proportions",
        spot_size=150,
        cmap="viridis"
    )

# ===== Part 6: Spatial analysis =====
# Compute spatial neighbors
sq.gr.spatial_neighbors(spatial_adata)

# Spatial autocorrelation of cell types
for ct in cell_types:
    sq.gr.spatial_autocorr(
        spatial_adata,
        attr="obs",
        mode="moran",
        genes=[f"prop_{ct}"]
    )

# ===== Part 7: Save results =====
destvi_model.save("destvi_model")
spatial_adata.write("spatial_deconvolved.h5ad")
```

### Best Practices for Spatial Analysis

1. **Reference quality**: Use high-quality, well-annotated scRNA-seq reference
2. **Gene overlap**: Ensure sufficient shared genes between reference and spatial
3. **Spatial coordinates**: Properly register spatial coordinates in `.obsm["spatial"]`
4. **Validation**: Use known marker genes to validate deconvolution
5. **Visualization**: Always visualize results spatially to check biological plausibility
6. **Cell type granularity**: Consider appropriate cell type resolution
7. **Computational resources**: Spatial models can be memory-intensive
8. **Quality control**: Filter low-quality spots before analysis
