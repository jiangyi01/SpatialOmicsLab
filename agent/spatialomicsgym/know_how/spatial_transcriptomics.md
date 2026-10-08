# Spatial Transcriptomics Analysis

Best practices and decision framework for spatial transcriptomics analysis using SpatialOmicsLab MCP tools.

## Overview
Which spatial transcriptomics tool to use for each task, how to call it, what input it needs, and platform notes for Visium, Xenium, MERFISH, CosMx, Slide-seq and Stereo-seq.

## Instructions

1. **MANDATORY before any tool choice — call the recommender.** Run `recommend_analysis_tools(h5ad_path="<file>.h5ad", analysis_goals="<your goal>")`. It returns a JSON **string** — `json.loads()` it before reading; slicing it gives you characters, not tools. Use the returned P1 (`priority==1`) tool unless you have a documented, data-driven reason to deviate. Do NOT guess from prior knowledge — the recommender encodes empirical multi-LLM benchmark winners and is the source of truth. Skipping this step is a known STCoscientist failure mode (gpt-5.4-mini ignored Quick-Ref P1s in 3/3 task types on 2026-05-11).
2. **Always diagnose first** (in parallel with step 1): Call `diagnose_transcriptomics_data(input_path)` to detect platform, format, and completeness before any analysis.
3. **Prepare data**: Run `run_spatial_pipeline()` to convert raw platform output to MCP-compatible h5ad.
4. **QC before analysis**: Run `run_transcriptomics_qc()` to filter low-quality spots and generate QC metrics.
5. **Compare multiple tools**: For clustering, SVG, and deconvolution, run at least 2 tools and compare their outputs yourself on the task's key metric (Quick Reference below); no registered tool makes that comparison for you.

## How to invoke MCP tools

MCP tools are registered in the live Python REPL by `add_mcp()` and become callable by their `spatialomicsgym_name` directly. **Do NOT write any of these patterns.** In a python `<execute>` cell they gain nothing: the `from mcp_servers...` and `importlib.import_module` forms hand back, at best, the function already in scope, and `import mcp_servers.<name>` fails with `ModuleNotFoundError: No module named 'mcp_servers'`. From a `#!BASH` or `#!CLI` cell (`python -c`, a heredoc, a script) all three fail that way:

```python
# WRONG — these are all forbidden P13 patterns:
from mcp_servers.somde import somde_run
import mcp_servers.cell2location
importlib.import_module("mcp_servers.prost")
```

**Correct usage — call the spatialomicsgym_name directly:**

```python
# Right — the function is already in scope inside <execute> blocks:
result = somde_run(
    h5ad_path="/path/to/input.h5ad",
    output_dir="/path/to/out",
)
```

If unsure of the canonical name, call `resolve_tool_name(query)` (it accepts fuzzy/alias inputs and returns the registered name).

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

**Note (2026-05-11):** Tool names above are the canonical MCP function names — the `spatialomicsgym_name` values registered in `MCP_server/mcp_config.yaml`, which is exactly the set `add_mcp()` makes callable. Most are also what `recommend_analysis_tools()` returns; a few (e.g. `run_bass`) are callable but reach the agent through the skills catalog rather than `_MCP_TOOLS`. P1 selections reflect multi-LLM benchmark winners (4 GPT models × ~30 tools × 3 task types).

## Input Requirements

Prerequisites:
- Spatial transcriptomics data in any supported format (Visium, Xenium, MERFISH, CosMx, Slide-seq, Stereo-seq, or h5ad).
- For deconvolution: a matched or similar-tissue scRNA-seq reference h5ad with cell type annotations.
- For image-dependent tools (MISO, DeepST, stLearn, iStar, XFuse): H&E histology or fluorescence images.

Standard MCP-compatible h5ad requirements:
- `adata.X`: raw count matrix (sparse CSR preferred).
- `adata.obsm['spatial']`: numpy array of shape (n_spots, 2) with spatial coordinates.
- `adata.obs['total_counts']`: total UMI counts per spot.
- `adata.var_names`: unique gene names.
- `adata.uns['spatial']` (optional): embedded images with scale factors.

## Platform-Specific Guidance

### 10x Visium
- Typical: 1,000-10,000 spots, 55μm diameter, H&E image included.
- All MCP tools work well with Visium data.
- Use Space Ranger output directly — `convert_visium_spaceranger()` handles conversion.
- H&E images enable MISO, DeepST, stLearn, iStar, XFuse for enhanced results.
- For deconvolution, Cell2Location is the gold standard for Visium.

### 10x Xenium
- Subcellular resolution, 100-500 gene panel, millions of transcripts.
- Convert transcripts to cell-level h5ad with `convert_xenium()`.
- High cell count may limit some tools — prefer Hotspot for SVGs, scanpy_spatial for clustering.
- Images are OME-TIFF fluorescence — use `process_xenium_images()` then `convert_fluorescence_to_pseudo_he()` if needed for H&E-only tools.

### MERFISH / Vizgen
- Subcellular, imaging-based, 100-500 gene panel.
- Requires both `cell_by_gene.csv` and `cell_metadata.csv`.
- DAPI/PolyT mosaic images available — process with `process_merfish_images()`.
- Limited gene panel may affect some SVG and clustering tools. Prefer methods robust to low gene counts.

### Nanostring CosMx
- Single-cell resolution, FOV-based, 960+ gene panel.
- Needs `exprMat_file.csv` + `metadata_file.csv`. FOV composite images optional.
- Process FOV images with `process_cosmx_fov_images()`.

### Slide-seq / Slide-seqV2
- Near-single-cell resolution (10μm beads), no histology images.
- Sparse data — use methods robust to sparsity (GraphST, STAGATE).
- No images: skip image-dependent tools (MISO, DeepST, iStar, XFuse).

### Stereo-seq
- Nanometer resolution, requires spatial binning (default bin_size=50).
- GEM/GEF format — convert a text GEM with `convert_stereoseq(gem_path, output_path, bin_size=50)`. A binary `.gef` is not readable by it: run `geftools gef2gem` on it first (SAW writes both, so the GEM may already be beside it).
- Very large datasets — prefer scalable tools (Hotspot, SOMDE, scanpy_spatial).

## Tool Selection Decision Tree

### Spatial Domain Identification
1. **Start with scanpy_spatial** as fast baseline (Leiden clustering on an expression-PCA kNN graph; spatial coordinates are not used, so it is an expression-only baseline).
2. **Add a GNN-based method** for comparison:
   - Has H&E images? → MISO or DeepST (multi-modal RNA + image)
   - No images, moderate data? → GraphST or STAGATE (spatial GNN)
   - Hierarchical tissue? → CellCharter (multi-scale)
   - Continuous gradients? → SpaceFlow (flow field)
3. **Compare with ARI/NMI** if ground truth annotations available.

### Spatially Variable Gene Detection
1. **Start with Hotspot** — fast, well-calibrated, identifies gene modules.
2. **Add SpatialDE** for gold-standard GP-based detection (slower, use for <10K spots).
3. **For large data** (>50K spots): SOMDE (SOM-accelerated) or SVGbit (binary, fast).
4. **For pattern classification**: PROST categorizes patterns (hotspot, streak, gradient).
5. **Compare overlap** between methods using Jaccard index. Genes found by 2+ methods are high-confidence SVGs.

### Cell Type Deconvolution
1. **With scRNA reference**:
   - Prepare reference: `prepare_reference_data(sc_h5ad, labels_key="CellType")`
   - Run Cell2Location (gold standard for Visium) + Tangram (flexible, DNN-based)
   - Compare with RMSE and Pearson correlation per cell type.
2. **Without scRNA reference**:
   - UCDeconvolve (pre-trained universal model, no reference needed)
   - Starfysh (gene signature-based, no paired reference needed)
3. **Key consideration**: Ensure the reference covers all cell types expected in the spatial tissue. Missing types cause deconvolution artifacts.

### Spatial Cell-Cell Communication
1. Use COMMOT with CellChat or CellPhoneDB ligand-receptor database.
2. Validate with SPAOTSC for orthogonal optimal-transport-based analysis.
3. Focus on biologically relevant pathways rather than raw communication scores.

### Multi-Slice Alignment
1. PASTE for pairwise alignment (standard, optimal transport).
2. MOSCOT for multi-modal integration with batch correction.
3. ST-GEARS for geometric deformation handling (serial sections).

## Quality Gates

- [ ] Data diagnosed with `diagnose_transcriptomics_data()` and format confirmed.
- [ ] QC completed: total counts, genes detected, mitochondrial % all within expected ranges.
- [ ] Less than 30% of spots/cells removed during QC filtering (investigate if higher).
- [ ] h5ad validated as MCP-compatible: has .X, obsm['spatial'], QC metrics, unique var_names.
- [ ] For deconvolution: scRNA reference has ≥10 cells per cell type, raw counts available.
- [ ] For image tools: images embedded in h5ad or paths verified.
- [ ] Analysis results saved with proper output file paths.
- [ ] When comparing tools: at least 2 tools run per task for robustness.

## Troubleshooting

**Issue**: MCP tool fails with "spatial not found in obsm"
**Solution**: Run `repair_spatial_h5ad()` to move coordinates from obs columns to obsm['spatial'].

**Issue**: Deconvolution results show negative or nonsensical proportions
**Solution**: Ensure spatial h5ad has RAW counts in .X (not normalized). Check gene overlap between spatial and reference (>100 shared genes needed).

**Issue**: Clustering gives too many or too few domains
**Solution**: Adjust resolution parameter (Leiden) or n_clusters. Start with the number of known tissue regions plus 1-2 for unknown domains.

**Issue**: SVG detection finds very few significant genes
**Solution**: Check if data is too sparse. Try relaxing FDR threshold to 0.1. Ensure coordinates are correct (spatial autocorrelation depends on accurate positions).

**Issue**: Very large dataset causes memory errors
**Solution**: Use scalable tools (Hotspot, SOMDE for SVGs; scanpy_spatial for clustering). Increase bin_size for Stereo-seq.

**Issue**: H5ad from R (Seurat object) missing spatial info
**Solution**: Use `convert_r_object()` which extracts spatial coordinates, reductions, images, and metadata from Seurat v3/v4/v5 objects.
