# Histolab WSI Processing

## Metadata

**Short Description**: WSI processing for digital pathology. Tissue detection, tile extraction (random, grid, score-based), filter pipelines for H&E/IHC. For dataset prep, tile-based DL, slide QC. Use pathml for multiplexed imaging.
**Source**: https://github.com/jaechang-hits/SciAgent-Skills/blob/fe505cae14d20b6c33be2e49666425be98f005bb/skills/medical-imaging/histolab-wsi-processing/SKILL.md
**License**: CC BY 4.0, Copyright (c) 2024 jaechang-hits (THIRD_PARTY_LICENSES/SciAgent-Skills-CC-BY-4.0.txt). Upstream inherits K-Dense-AI/claude-scientific-skills (MIT) and snap-stanford/Biomni (Apache-2.0); those notices travel with this text. Changes were made -- see Modifications.
**Wrapped Tool License**: not stated upstream for the tool; the upstream frontmatter `license` field reads "Apache-2.0", and upstream uses that field for the skill text in some files and for the tool in others, so it is not taken as the tool's licence
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Related Skills); 1 install fence(s) replaced by a provisioning pointer; 3 upstream reference file(s) folded in (Filters and Preprocessing Reference, Tile Extraction Reference, Visualization, Slides, and Tissue Masks Reference); 1 manifest replacement(s) and 0 excision(s) applied.

---

## Overview

Histolab is a Python library for processing whole slide images (WSI) in digital pathology. It automates tissue detection, extracts informative tiles from gigapixel images using multiple strategies, and provides composable filter pipelines for preprocessing. The library handles SVS, TIFF, NDPI, and other WSI formats via OpenSlide.

## When to Use

- Extracting tiles from whole slide images for deep learning model training
- Detecting tissue regions and filtering background/artifacts in histopathology slides
- Building preprocessing pipelines for H&E or IHC stained tissue sections
- Creating quality-driven tile datasets ranked by nuclei density or cellularity
- Performing batch tile extraction across slide collections with consistent parameters
- Assessing slide quality and tissue coverage before computational pathology workflows
- For raw slide access without tile extraction, use `openslide-python` directly
- For complex multiplexed imaging or spatial proteomics pipelines, use `pathml` instead

## Prerequisites

- **Python packages**: `histolab` (includes OpenSlide Python bindings)
- **System dependency**: OpenSlide C library must be installed separately
- **Supported formats**: SVS, TIFF, NDPI, VMS, SCN, MRXS (via OpenSlide)

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

## Quick Start

```python
from histolab.slide import Slide
from histolab.tiler import RandomTiler

# Load slide
slide = Slide("slide.svs", processed_path="output/")
print(f"Dimensions: {slide.dimensions}, Levels: {slide.levels}")

# Configure tiler
tiler = RandomTiler(
    tile_size=(512, 512), n_tiles=100, level=0, seed=42,
    check_tissue=True, tissue_percent=80.0
)

# Preview and extract
tiler.locate_tiles(slide, n_tiles=20)
tiler.extract(slide)
```

## Core API

### Module 1: Slide Management

The `Slide` class is the primary interface for loading and inspecting WSI files.

```python
from histolab.slide import Slide
from histolab.data import prostate_tissue

# Load from built-in sample data (prostate, ovarian, breast, heart, kidney)
prostate_svs, prostate_path = prostate_tissue()
slide = Slide(prostate_path, processed_path="output/")

# Inspect properties
print(f"Dimensions: {slide.dimensions}")       # (width, height) at level 0
print(f"Levels: {slide.levels}")               # Number of pyramid levels
print(f"Level dims: {slide.level_dimensions}") # Dimensions per level
print(f"Magnification: {slide.properties.get('openslide.objective-power', 'N/A')}")
print(f"MPP-X: {slide.properties.get('openslide.mpp-x', 'N/A')}")

# Thumbnail and scaled image
slide.save_thumbnail()  # Saves to processed_path
scaled = slide.scaled_image(scale_factor=32)

# Extract region at specific coordinates
region = slide.extract_region(location=(1000, 2000), size=(512, 512), level=0)
```

### Module 2: Tissue Detection

Mask classes identify tissue regions and filter background for tile extraction.

```python
from histolab.masks import TissueMask, BiggestTissueBoxMask, BinaryMask
import numpy as np

# TissueMask: segments ALL tissue regions (multiple sections)
tissue_mask = TissueMask()
mask_array = tissue_mask(slide)  # Binary NumPy array: True=tissue, False=background
print(f"Tissue coverage: {mask_array.sum() / mask_array.size * 100:.1f}%")

# BiggestTissueBoxMask: bounding box of largest tissue region (default)
biggest_mask = BiggestTissueBoxMask()

# Visualize mask on slide thumbnail
slide.locate_mask(tissue_mask)

# Custom mask via BinaryMask subclass
class RectangularROI(BinaryMask):
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def _mask(self, slide):
        thumb = slide.thumbnail
        mask = np.zeros(thumb.shape[:2], dtype=bool)
        mask[self.y:self.y+self.h, self.x:self.x+self.w] = True
        return mask
```

### Module 3: Tile Extraction

Three strategies for extracting tiles: random sampling, grid coverage, and score-based selection.

```python
from histolab.tiler import RandomTiler, GridTiler, ScoreTiler
from histolab.scorer import NucleiScorer
from histolab.masks import TissueMask

# RandomTiler: fixed number of randomly positioned tiles
random_tiler = RandomTiler(
    tile_size=(512, 512), n_tiles=100, level=0,
    seed=42, check_tissue=True, tissue_percent=80.0
)
random_tiler.locate_tiles(slide, n_tiles=20)  # Preview first
random_tiler.extract(slide)

# GridTiler: systematic grid coverage
grid_tiler = GridTiler(
    tile_size=(512, 512), level=0,
    pixel_overlap=0, check_tissue=True, tissue_percent=70.0
)
grid_tiler.extract(slide, extraction_mask=TissueMask())

# ScoreTiler: top-ranked tiles by scoring function
score_tiler = ScoreTiler(
    tile_size=(512, 512), n_tiles=50, level=0,
    scorer=NucleiScorer(), check_tissue=True
)
score_tiler.extract(slide, report_path="tiles_report.csv")
# Report CSV: tile_name, x_coord, y_coord, level, score, tissue_percent
```

### Module 4: Filters and Preprocessing

Composable image and morphological filters for tissue detection and preprocessing.

```python
from histolab.filters.image_filters import (
    RgbToGrayscale, RgbToHsv, RgbToHed,
    OtsuThreshold, AdaptiveThreshold,
    StretchContrast, HistogramEqualization, Invert
)
from histolab.filters.morphological_filters import (
    BinaryDilation, BinaryErosion, BinaryOpening, BinaryClosing,
    RemoveSmallObjects, RemoveSmallHoles
)
from histolab.filters.compositions import Compose

# Standard tissue detection pipeline
tissue_pipeline = Compose([
    RgbToGrayscale(),
    OtsuThreshold(),
    BinaryDilation(disk_size=5),
    RemoveSmallHoles(area_threshold=1000),
    RemoveSmallObjects(area_threshold=500)
])

# Use custom pipeline with TissueMask
from histolab.masks import TissueMask
custom_mask = TissueMask(filters=tissue_pipeline)

# Stain deconvolution (H&E)
hed_filter = RgbToHed()  # Hematoxylin-Eosin-DAB separation
```

```python
# Apply filters to individual tiles
from histolab.tile import Tile

filter_chain = Compose([RgbToGrayscale(), StretchContrast()])
filtered_tile = tile.apply_filters(filter_chain)

# Lambda for custom inline filters
from histolab.filters.image_filters import Lambda
import numpy as np

brightness = Lambda(lambda img: np.clip(img * 1.2, 0, 255).astype(np.uint8))
red_channel = Lambda(lambda img: img[:, :, 0])
```

### Module 5: Scoring

Scorers rank tiles by tissue content quality for use with ScoreTiler.

```python
from histolab.scorer import NucleiScorer, CellularityScorer, Scorer
import numpy as np

# Built-in scorers
nuclei = NucleiScorer()       # Scores by nuclei density (grayscale threshold + count)
cellularity = CellularityScorer()  # Scores by overall cellular content

# Custom scorer
class ColorVarianceScorer(Scorer):
    def __call__(self, tile):
        """Score tiles by color variance (higher = more informative)."""
        tile_array = np.array(tile.image)
        return np.var(tile_array, axis=(0, 1)).sum()

score_tiler = ScoreTiler(
    tile_size=(512, 512), n_tiles=30,
    scorer=ColorVarianceScorer()
)
```

### Module 6: Visualization

Built-in methods and matplotlib patterns for inspecting slides, masks, and tiles.

```python
import matplotlib.pyplot as plt
from histolab.masks import TissueMask

# Built-in: mask overlay on slide thumbnail
slide.locate_mask(TissueMask())

# Built-in: tile location preview
tiler.locate_tiles(slide, n_tiles=20)

# Manual side-by-side: slide vs mask
mask = TissueMask()
mask_array = mask(slide)
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes[0].imshow(slide.thumbnail); axes[0].set_title("Slide"); axes[0].axis('off')
axes[1].imshow(mask_array, cmap='gray'); axes[1].set_title("Mask"); axes[1].axis('off')
plt.tight_layout()
plt.show()

# Display extracted tiles in grid
from pathlib import Path
from PIL import Image
tile_paths = list(Path("output/tiles/").glob("*.png"))[:16]
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for idx, tp in enumerate(axes.ravel()):
    if idx < len(tile_paths):
        tp.imshow(Image.open(tile_paths[idx]))
        tp.set_title(tile_paths[idx].stem, fontsize=8)
    tp.axis('off')
plt.tight_layout()
plt.show()
```

## Key Concepts

### WSI Pyramid Levels

Whole slide images use a pyramidal structure with multiple resolution levels. Level 0 is the highest resolution (native scan). Higher levels provide progressively lower resolutions for faster access.

```python
for level in range(slide.levels):
    dims = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    print(f"Level {level}: {dims}, downsample: {downsample:.0f}x")
# Level 0: (98304, 221184), downsample: 1x
# Level 1: (24576, 55296), downsample: 4x
```

### Filter Composition Pattern

Filters are designed to be chained via `Compose`. The output of one filter becomes the input of the next. Image filters operate on RGB/grayscale arrays; morphological filters operate on binary arrays. Order matters: always convert to the expected input type before applying downstream filters.

### Mask-Tiler Integration

All tilers accept an `extraction_mask` parameter. The default is `BiggestTissueBoxMask()`. Override with `TissueMask()` for multi-section slides or a custom `BinaryMask` subclass for ROI-specific extraction.

## Common Workflows

### Workflow 1: Exploratory Slide Analysis

**Goal**: Quickly inspect a slide, detect tissue, and sample diverse regions for review.

```python
from histolab.slide import Slide
from histolab.tiler import RandomTiler
from histolab.masks import TissueMask
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

slide = Slide("slide.svs", processed_path="output/exploratory/")
print(f"Dimensions: {slide.dimensions}, Levels: {slide.levels}")
slide.save_thumbnail()

# Visualize tissue detection
tissue_mask = TissueMask()
slide.locate_mask(tissue_mask)
mask_arr = tissue_mask(slide)
print(f"Tissue coverage: {mask_arr.sum() / mask_arr.size * 100:.1f}%")

# Sample tiles
tiler = RandomTiler(
    tile_size=(512, 512), n_tiles=50, level=0,
    seed=42, check_tissue=True, tissue_percent=80.0
)
tiler.locate_tiles(slide, n_tiles=20)
tiler.extract(slide)
```

### Workflow 2: Deep Learning Dataset Preparation

**Goal**: Build a quality-controlled tile dataset from multiple slides for model training.

```python
from pathlib import Path
from histolab.slide import Slide
from histolab.tiler import ScoreTiler
from histolab.scorer import NucleiScorer
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

slide_dir = Path("slides/")
output_base = Path("output/dataset/")
all_reports = []

tiler = ScoreTiler(
    tile_size=(512, 512), n_tiles=100, level=0,
    scorer=NucleiScorer(), check_tissue=True, tissue_percent=80.0
)

for slide_path in sorted(slide_dir.glob("*.svs")):
    out_dir = output_base / slide_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    slide = Slide(str(slide_path), processed_path=str(out_dir))
    slide.save_thumbnail()
    report_path = str(out_dir / "report.csv")
    tiler.extract(slide, report_path=report_path)
    df = pd.read_csv(report_path)
    df["slide"] = slide_path.stem
    all_reports.append(df)
    print(f"{slide_path.stem}: {len(df)} tiles, mean score {df['score'].mean():.3f}")

combined = pd.concat(all_reports, ignore_index=True)
combined.to_csv(output_base / "dataset_manifest.csv", index=False)
print(f"Total: {len(combined)} tiles from {len(all_reports)} slides")
```

### Workflow 3: Custom Tissue Detection with Artifact Removal

**Goal**: Handle slides with pen annotations or unusual staining using custom filter pipelines.

```python
from histolab.slide import Slide
from histolab.masks import TissueMask
from histolab.tiler import GridTiler
from histolab.filters.compositions import Compose
from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
from histolab.filters.morphological_filters import (
    BinaryDilation, RemoveSmallObjects, RemoveSmallHoles
)

# Aggressive artifact removal pipeline
aggressive_filters = Compose([
    RgbToGrayscale(),
    OtsuThreshold(),
    BinaryDilation(disk_size=10),
    RemoveSmallHoles(area_threshold=5000),
    RemoveSmallObjects(area_threshold=3000)
])

custom_mask = TissueMask(filters=aggressive_filters)
slide = Slide("artifact_slide.svs", processed_path="output/clean/")

# Compare default vs custom mask
slide.locate_mask(TissueMask())     # Default
slide.locate_mask(custom_mask)      # Custom (tighter)

# Extract with custom mask
grid_tiler = GridTiler(
    tile_size=(512, 512), level=1,
    check_tissue=True, tissue_percent=70.0
)
grid_tiler.extract(slide, extraction_mask=custom_mask)
```

## Key Parameters

| Parameter | Module | Default | Range / Options | Effect |
|-----------|--------|---------|-----------------|--------|
| `tile_size` | All Tilers | `(512, 512)` | Any `(w, h)` tuple | Tile dimensions in pixels |
| `level` | All Tilers | `0` | `0` to `slide.levels-1` | Pyramid level (0=highest resolution) |
| `check_tissue` | All Tilers | `True` | `True`/`False` | Filter tiles by tissue content |
| `tissue_percent` | All Tilers | `80.0` | `0.0`-`100.0` | Minimum tissue coverage threshold |
| `n_tiles` | Random/ScoreTiler | varies | Any positive int | Number of tiles to extract |
| `seed` | RandomTiler | `None` | Any int | Random seed for reproducibility |
| `pixel_overlap` | GridTiler | `0` | `0`+ | Overlap between adjacent tiles in pixels |
| `scorer` | ScoreTiler | required | `NucleiScorer()`, `CellularityScorer()`, custom | Scoring function for tile ranking |
| `extraction_mask` | All Tilers | `BiggestTissueBoxMask()` | Any `BinaryMask` | Mask defining valid extraction region |
| `disk_size` | Morphological filters | `5` | `1`-`20` | Structuring element size |
| `area_threshold` | RemoveSmall* | `500`/`1000` | `0`+ | Minimum area for objects/holes in pixels |

## Best Practices

1. **Always preview before extracting**: Call `locate_tiles()` and `locate_mask()` to validate settings before committing to full extraction. This saves hours on large slide collections.

2. **Match level to analysis resolution**: Level 0 provides maximum detail but is slow; level 1-2 is typically sufficient for initial analysis. Use level 0 only for tasks requiring cellular detail.

3. **Choose the right tiler strategy**:
   - `RandomTiler` for exploratory analysis and diverse sampling
   - `GridTiler` for complete coverage and spatial analysis
   - `ScoreTiler` for quality-driven dataset curation

4. **Use seeds for reproducibility**: Always set `seed` in `RandomTiler` to ensure consistent tile selection across runs.

5. **Customize masks for specific stains**: H&E and IHC stains have different color profiles. Adjust filter parameters or build custom `Compose` pipelines for non-standard stains.

6. **Anti-pattern -- Don't use TissueMask when BiggestTissueBoxMask suffices**: `TissueMask` is more computationally expensive. Use it only when the slide has multiple tissue sections.

7. **Enable logging for batch processing**: `logging.basicConfig(level=logging.INFO)` provides progress tracking during extraction.

8. **Generate CSV reports with ScoreTiler**: Use `report_path` in `extract()` to create metadata manifests for downstream ML pipelines.

9. **Anti-pattern -- Don't extract at level 0 for initial exploration**: Use level 1 or 2 for fast iteration, then switch to level 0 for final dataset generation.

## Common Recipes

### Recipe: Nuclei Enhancement Pipeline

When to use: Isolate and enhance nuclei signal from H&E stained sections for analysis.

```python
from histolab.filters.image_filters import RgbToHed, HistogramEqualization, Lambda
from histolab.filters.compositions import Compose

nuclei_pipeline = Compose([
    RgbToHed(),
    Lambda(lambda hed: hed[:, :, 0]),  # Extract hematoxylin channel
    HistogramEqualization()
])

# Apply to slide thumbnail for visualization
enhanced = nuclei_pipeline(slide.thumbnail)
```

### Recipe: Score Distribution Analysis

When to use: Assess tile quality distribution and identify optimal score thresholds.

```python
import pandas as pd
import matplotlib.pyplot as plt

report = pd.read_csv("tiles_report.csv")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(report['score'], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Tile Score'); axes[0].set_ylabel('Frequency')
axes[0].set_title('Score Distribution')

axes[1].scatter(report['tissue_percent'], report['score'], alpha=0.5)
axes[1].set_xlabel('Tissue %'); axes[1].set_ylabel('Score')
axes[1].set_title('Score vs Tissue Coverage')

plt.tight_layout()
plt.savefig("quality_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
```

### Recipe: Multi-Level Hierarchical Extraction

When to use: Extract tiles at multiple magnification levels from the same locations for multi-scale analysis.

```python
from histolab.tiler import RandomTiler

for level in [0, 1, 2]:
    tiler = RandomTiler(
        tile_size=(512, 512), n_tiles=50,
        level=level, seed=42,  # Same seed = same locations
        prefix=f"level{level}_"
    )
    tiler.extract(slide)
    print(f"Extracted level {level} tiles")
```

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `OpenSlideError: Unsupported format` | OpenSlide C library not installed or slide format unsupported | OpenSlide is not installed from this document: if `import openslide` fails, say so; verify the format with `openslide-show-properties` |
| No tiles extracted | `tissue_percent` too high or mask misses tissue | Lower `tissue_percent` (try 60-70%); preview mask with `locate_mask()` |
| Many background tiles | `check_tissue=False` or poor mask | Enable `check_tissue=True`; increase `tissue_percent`; use `TissueMask()` |
| Extraction very slow | Level 0 on large slide or `TissueMask` on many sections | Extract at level 1-2; use `BiggestTissueBoxMask`; reduce `n_tiles` |
| Tiles have pen artifacts | Default mask includes pen marks | Build custom filter with HSV-based pen detection; increase `RemoveSmallObjects` threshold |
| `MemoryError` during extraction | Level 0 tile access on very large WSI | Extract at lower level; process fewer tiles per batch |
| Inconsistent results across runs | Missing `seed` in `RandomTiler` | Always set `seed` parameter |
| Tiles too small/large for model | `tile_size` mismatch with model input | Adjust `tile_size` to match model requirements (commonly 224, 256, 512) |

## Bundled Resources

### (see the section "Reference: Filters and Preprocessing Reference" below)

Comprehensive filter reference covering all image filters (RgbToGrayscale, RgbToHsv, RgbToHed, OtsuThreshold, AdaptiveThreshold, Invert, StretchContrast, HistogramEqualization, Lambda) and morphological filters (BinaryDilation, BinaryErosion, BinaryOpening, BinaryClosing, RemoveSmallObjects, RemoveSmallHoles) with individual code examples, use-case descriptions, and common preprocessing pipelines (tissue detection, pen removal, nuclei enhancement, stain normalization).

- **Covers**: All filter types with individual code blocks, `Compose` chaining, quality control filters, custom mask integration
- **Relocated inline**: Standard tissue detection pipeline and Lambda filter basics moved to Core API Module 4
- **Omitted**: Filter effect visualization step-by-step (covered in (see the section "Reference: Visualization, Slides, and Tissue Masks Reference" below)); best practices list partially consolidated into main Best Practices section

### (see the section "Reference: Tile Extraction Reference" below)

Detailed tile extraction reference covering all three tiler strategies (RandomTiler, GridTiler, ScoreTiler) with full parameter documentation, all built-in scorers (NucleiScorer, CellularityScorer), custom scorer creation, extraction workflows with logging and CSV reporting, and advanced patterns (multi-level, hierarchical, post-extraction blur filtering).

- **Covers**: Per-tiler parameters and use cases, scorer API, tile preview, extraction with reports, advanced patterns, performance optimization
- **Relocated inline**: Basic tiler usage and scorer creation moved to Core API Modules 3 and 5; common parameters table moved to Key Parameters
- **Omitted**: ASCII grid pattern diagrams (trivial visual aid)

### (see the section "Reference: Visualization, Slides, and Tissue Masks Reference" below)

Consolidated visualization, slide management, and tissue mask reference. Covers slide inspection workflows, mask comparison visualization, tile grid display, quality assessment (score distributions, top/bottom tile comparison), multi-slide collection thumbnails, tissue coverage bar charts, filter effect visualization pipeline, PDF report generation, and interactive Jupyter widgets.

- **Source**: Consolidates visualization.md (547 lines) + slide_management.md (172 lines) + tissue_masks.md (251 lines)
- **Covers**: All visualization patterns from original, slide inspection workflow, sample datasets, pyramid level enumeration, mask classes and customization, annotation exclusion pattern
- **Relocated inline**: Core slide properties and mask basics moved to Core API Modules 1-2; thumbnail display and basic locate_mask/locate_tiles moved to Module 6
- **Omitted**: Slide name/path trivial properties (2 lines); custom tile location visualization with manual coordinate calculation (conceptual only, not practical)

### Original reference file disposition (5 files):

1. **slide_management.md** (172 lines) -- (b) Consolidated: core content into Core API Module 1 (Slide class, properties, sample data, thumbnails, regions, pyramid levels); advanced slide inspection and multi-slide workflow into (see the section "Reference: Visualization, Slides, and Tissue Masks Reference" below)
2. **tissue_masks.md** (251 lines) -- (b) Consolidated: core mask classes and usage into Core API Module 2; custom masks (RectangularMask, AnnotationExclusionMask) and mask comparison into (see the section "Reference: Visualization, Slides, and Tissue Masks Reference" below)
3. **tile_extraction.md** (421 lines) -- (a) Migrated to (see the section "Reference: Tile Extraction Reference" below) with condensation
4. **filters_preprocessing.md** (514 lines) -- (a) Migrated to (see the section "Reference: Filters and Preprocessing Reference" below) with condensation
5. **visualization.md** (547 lines) -- (a) Migrated to (see the section "Reference: Visualization, Slides, and Tissue Masks Reference" below) (consolidated with slide_management.md and tissue_masks.md content)

## References

- [Histolab documentation](https://histolab.readthedocs.io/) -- official API reference and tutorials
- [Histolab GitHub repository](https://github.com/histolab/histolab) -- source code and examples
- [OpenSlide](https://openslide.org/) -- underlying C library for WSI format support
- [TCGA sample data](https://portal.gdc.cancer.gov/) -- source of built-in sample datasets

## Reference: Filters and Preprocessing Reference

Comprehensive reference for histolab's filter system. Filters are composable building blocks for tissue detection, quality control, artifact removal, and image preprocessing.

### Image Filters

#### RgbToGrayscale

Convert RGB images to single-channel grayscale. Required before thresholding operations.

```python
from histolab.filters.image_filters import RgbToGrayscale

gray_filter = RgbToGrayscale()
gray_image = gray_filter(rgb_image)
```

#### RgbToHsv

Convert RGB to Hue-Saturation-Value color space. Useful for color-based segmentation (e.g., detecting pen marks by hue range).

```python
from histolab.filters.image_filters import RgbToHsv

hsv_filter = RgbToHsv()
hsv_image = hsv_filter(rgb_image)
# Channel 0: Hue (0-1), Channel 1: Saturation, Channel 2: Value
```

#### RgbToHed

Convert RGB to Hematoxylin-Eosin-DAB color space for stain deconvolution. Separates H&E stain components for quantitative analysis.

```python
from histolab.filters.image_filters import RgbToHed

hed_filter = RgbToHed()
hed_image = hed_filter(rgb_image)
# Channel 0: Hematoxylin (nuclei), Channel 1: Eosin (cytoplasm), Channel 2: DAB
```

**Use cases**: Separating nuclear (hematoxylin) vs cytoplasmic (eosin) staining, quantifying stain intensity, nuclei enhancement pipelines.

#### OtsuThreshold

Automatic thresholding using Otsu's method. Determines optimal threshold to separate foreground from background by minimizing intra-class variance.

```python
from histolab.filters.image_filters import OtsuThreshold

otsu_filter = OtsuThreshold()
binary_image = otsu_filter(grayscale_image)
```

**Use cases**: Tissue detection, nuclei segmentation, binary mask creation.

#### AdaptiveThreshold

Local thresholding for images with non-uniform illumination or variable staining intensity.

```python
from histolab.filters.image_filters import AdaptiveThreshold

adaptive_filter = AdaptiveThreshold(
    block_size=11,  # Size of local neighborhood (must be odd)
    offset=2        # Constant subtracted from mean
)
binary_image = adaptive_filter(grayscale_image)
```

#### Invert, StretchContrast, HistogramEqualization

Intensity manipulation filters for preprocessing and normalization.

```python
from histolab.filters.image_filters import Invert, StretchContrast, HistogramEqualization

Invert()(image)                      # Invert intensity values
StretchContrast()(image)             # Stretch intensity range for low-contrast features
HistogramEqualization()(grayscale)   # Equalize histogram to standardize contrast
```

#### Lambda

Create custom inline filters without subclassing.

```python
from histolab.filters.image_filters import Lambda
import numpy as np

brightness = Lambda(lambda img: np.clip(img * 1.2, 0, 255).astype(np.uint8))
red_channel = Lambda(lambda img: img[:, :, 0])
```

### Morphological Filters

All morphological filters operate on binary images. Apply after thresholding.

#### BinaryDilation

Expand white (True) regions. Connects nearby tissue fragments and fills small gaps.

```python
from histolab.filters.morphological_filters import BinaryDilation

dilation = BinaryDilation(disk_size=5)  # Structuring element size (default: 5)
dilated = dilation(binary_image)
```

#### BinaryErosion

Shrink white regions. Removes small protrusions and separates connected objects.

```python
from histolab.filters.morphological_filters import BinaryErosion

erosion = BinaryErosion(disk_size=5)
eroded = erosion(binary_image)
```

#### BinaryOpening

Erosion followed by dilation. Removes small objects while preserving larger shapes.

```python
from histolab.filters.morphological_filters import BinaryOpening

opening = BinaryOpening(disk_size=3)
opened = opening(binary_image)
```

#### BinaryClosing

Dilation followed by erosion. Fills small holes while preserving overall shape.

```python
from histolab.filters.morphological_filters import BinaryClosing

closing = BinaryClosing(disk_size=5)
closed = closing(binary_image)
```

#### RemoveSmallObjects

Remove connected components smaller than a pixel area threshold.

```python
from histolab.filters.morphological_filters import RemoveSmallObjects

remove_small = RemoveSmallObjects(area_threshold=500)  # Minimum area in pixels
cleaned = remove_small(binary_image)
```

#### RemoveSmallHoles

Fill holes in tissue regions smaller than a threshold area.

```python
from histolab.filters.morphological_filters import RemoveSmallHoles

fill_holes = RemoveSmallHoles(area_threshold=1000)  # Maximum hole size to fill
filled = fill_holes(binary_image)
```

### Filter Composition

#### Compose

Chain multiple filters into a reusable pipeline. Output of each filter feeds into the next.

```python
from histolab.filters.compositions import Compose
from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
from histolab.filters.morphological_filters import (
    BinaryDilation, RemoveSmallHoles, RemoveSmallObjects
)

# Standard tissue detection pipeline
tissue_detection = Compose([
    RgbToGrayscale(),
    OtsuThreshold(),
    BinaryDilation(disk_size=5),
    RemoveSmallHoles(area_threshold=1000),
    RemoveSmallObjects(area_threshold=500)
])

result = tissue_detection(rgb_image)
```

### Common Preprocessing Pipelines

#### Pen Mark Removal

Remove blue/green pen markings common on pathology slides.

```python
from histolab.filters.image_filters import RgbToHsv, Lambda
from histolab.filters.compositions import Compose
import numpy as np

def remove_pen_marks(hsv_image):
    """Remove blue/green pen markings from slide."""
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    pen_mask = ((h > 0.45) & (h < 0.7) & (s > 0.3))
    hsv_image[pen_mask] = [0, 0, 1]  # Set pen regions to white
    return hsv_image

pen_removal = Compose([RgbToHsv(), Lambda(remove_pen_marks)])
```

#### Nuclei Enhancement

Isolate and enhance nuclear staining from H&E images.

```python
from histolab.filters.image_filters import RgbToHed, HistogramEqualization, Lambda
from histolab.filters.compositions import Compose

nuclei_enhancement = Compose([
    RgbToHed(),
    Lambda(lambda hed: hed[:, :, 0]),  # Extract hematoxylin channel
    HistogramEqualization()
])
```

#### Contrast Normalization

Normalize contrast across slides with variable staining quality.

```python
from histolab.filters.image_filters import RgbToGrayscale, StretchContrast, HistogramEqualization
from histolab.filters.compositions import Compose

contrast_norm = Compose([
    RgbToGrayscale(),
    StretchContrast(),
    HistogramEqualization()
])
```

#### Basic Stain Normalization

Simple H&E stain normalization using channel-wise z-score standardization.

```python
from histolab.filters.image_filters import RgbToHed, Lambda
from histolab.filters.compositions import Compose
import numpy as np

def normalize_hed(hed_image, target_means=[0.65, 0.70], target_stds=[0.15, 0.13]):
    """Z-score normalize H&E channels to target distribution."""
    h = hed_image[:, :, 0]
    e = hed_image[:, :, 1]
    hed_image[:, :, 0] = (h - h.mean()) / (h.std() + 1e-8) * target_stds[0] + target_means[0]
    hed_image[:, :, 1] = (e - e.mean()) / (e.std() + 1e-8) * target_stds[1] + target_means[1]
    return hed_image

stain_norm = Compose([RgbToHed(), Lambda(normalize_hed)])
```

### Applying Filters to Tiles

Filters can be applied to individual `Tile` objects for post-extraction preprocessing.

```python
from histolab.tile import Tile
from histolab.filters.compositions import Compose
from histolab.filters.image_filters import RgbToGrayscale, StretchContrast

filter_chain = Compose([RgbToGrayscale(), StretchContrast()])
processed_tile = tile.apply_filters(filter_chain)
```

### Custom Mask Integration

Use custom filter pipelines with `TissueMask` for specialized tissue detection.

```python
from histolab.masks import TissueMask
from histolab.filters.compositions import Compose
from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
from histolab.filters.morphological_filters import BinaryDilation, RemoveSmallObjects

aggressive_filters = Compose([
    RgbToGrayscale(),
    OtsuThreshold(),
    BinaryDilation(disk_size=10),
    RemoveSmallObjects(area_threshold=5000)
])

custom_mask = TissueMask(filters=aggressive_filters)
```

### Quality Control Filters

#### Blur Detection

Score tile sharpness using Laplacian variance. Low values indicate blurry tiles.

```python
from histolab.filters.image_filters import RgbToGrayscale, Lambda
import cv2
import numpy as np

def laplacian_blur_score(gray_image):
    """Laplacian variance blur metric. Higher = sharper."""
    return cv2.Laplacian(np.array(gray_image), cv2.CV_64F).var()

blur_detector = Lambda(
    lambda img: laplacian_blur_score(RgbToGrayscale()(img))
)
```

#### Tissue Coverage Calculation

Calculate percentage of tissue in an image region.

```python
from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold, Lambda
from histolab.filters.compositions import Compose

def tissue_coverage(image):
    """Return tissue percentage (0-100)."""
    mask = Compose([RgbToGrayscale(), OtsuThreshold()])(image)
    return mask.sum() / mask.size * 100

coverage = Lambda(tissue_coverage)
```


Condensed from original: 514 lines. Retained: all 10 image filter types, all 6 morphological filter types, Compose chaining, 4 preprocessing pipelines (tissue detection, pen removal, nuclei enhancement, stain normalization), tile filter application, custom mask integration, QC filters (blur detection, tissue coverage). Omitted: duplicate tissue detection pipeline (shown identically in main SKILL.md Core API Module 4); per-filter "Use cases" lists shortened to inline descriptions for filters with obvious purpose (Invert, StretchContrast, HistogramEqualization); troubleshooting section (4 items consolidated into main Troubleshooting table).

## Reference: Tile Extraction Reference

Detailed reference for histolab's tile extraction strategies, scoring functions, extraction workflows, and advanced patterns.

### Common Parameters

All tiler classes share these parameters:

```python
tile_size: tuple = (512, 512)       # Tile dimensions (width, height) in pixels
level: int = 0                      # Pyramid level (0=highest resolution)
check_tissue: bool = True           # Filter tiles by tissue content
tissue_percent: float = 80.0        # Minimum tissue coverage (0-100)
prefix: str = ""                    # Prefix for saved tile filenames
suffix: str = ".png"                # File extension for saved tiles
extraction_mask: BinaryMask = BiggestTissueBoxMask()  # Region constraint
```

### RandomTiler

Extract a fixed number of randomly positioned tiles from tissue regions.

```python
from histolab.tiler import RandomTiler

random_tiler = RandomTiler(
    tile_size=(512, 512),
    n_tiles=100,            # Number of tiles to extract
    level=0,
    seed=42,                # Random seed for reproducibility
    check_tissue=True,
    tissue_percent=80.0,
    max_iter=1000           # Maximum attempts to find valid tiles
)

random_tiler.extract(slide, extraction_mask=TissueMask())
```

**When to use**:
- Exploratory analysis and quick sampling
- Creating diverse training datasets
- Balanced sampling from multiple slides
- Fast initial assessment of slide content

**Advantages**: Computationally efficient, good morphological diversity, reproducible with seed.

**Limitations**: May miss rare patterns, no coverage guarantee, random distribution may not capture structured features.

### GridTiler

Systematically extract tiles across tissue in a regular grid pattern.

```python
from histolab.tiler import GridTiler

grid_tiler = GridTiler(
    tile_size=(512, 512),
    level=0,
    check_tissue=True,
    tissue_percent=80.0,
    pixel_overlap=0         # Overlap between adjacent tiles
)

grid_tiler.extract(slide)
```

**`pixel_overlap` settings**:
- `0`: Non-overlapping tiles (default)
- `64-256`: Sliding window with partial overlap
- Use overlap for segmentation tasks requiring boundary coverage

**When to use**:
- Complete tissue coverage for whole-slide analysis
- Spatial analysis requiring positional information
- Image reconstruction from tiles
- Semantic segmentation tasks

**Advantages**: Complete coverage, preserves spatial relationships, predictable positions.

**Limitations**: Computationally intensive on large slides, may generate many low-tissue tiles (mitigated by `check_tissue`), larger output datasets.

### ScoreTiler

Extract top-ranked tiles based on a scoring function.

```python
from histolab.tiler import ScoreTiler
from histolab.scorer import NucleiScorer

score_tiler = ScoreTiler(
    tile_size=(512, 512),
    n_tiles=50,             # Number of top-scoring tiles to keep
    level=0,
    scorer=NucleiScorer(),
    check_tissue=True
)

score_tiler.extract(slide, report_path="tiles_report.csv")
```

**When to use**:
- Extracting the most informative regions
- Prioritizing tiles with specific features (nuclei, cellularity)
- Quality-based dataset curation
- Focusing on diagnostically relevant areas

**Advantages**: Focuses on most informative content, reduces dataset size while maintaining quality, customizable scoring.

**Limitations**: Slower than RandomTiler (scores all candidates), requires appropriate scorer, may miss low-scoring but relevant regions.

### Scorers

#### NucleiScorer

Scores tiles based on nuclei detection and density. Converts tile to grayscale, applies thresholding, counts nuclei-like structures, and assigns density score.

```python
from histolab.scorer import NucleiScorer

nuclei_scorer = NucleiScorer()
# Best for: cell-rich regions, tumor detection, mitosis analysis
```

#### CellularityScorer

Scores tiles based on overall cellular content (tissue vs stroma ratio).

```python
from histolab.scorer import CellularityScorer

cellularity_scorer = CellularityScorer()
# Best for: tumor cellularity, dense vs sparse tissue separation
```

#### Custom Scorers

Subclass `Scorer` to implement domain-specific scoring.

```python
from histolab.scorer import Scorer
import numpy as np

class ColorVarianceScorer(Scorer):
    def __call__(self, tile):
        """Score by color variance — higher values indicate more varied tissue."""
        tile_array = np.array(tile.image)
        return np.var(tile_array, axis=(0, 1)).sum()

class StainIntensityScorer(Scorer):
    def __call__(self, tile):
        """Score by hematoxylin intensity (nuclear staining)."""
        from histolab.filters.image_filters import RgbToHed
        hed = RgbToHed()(np.array(tile.image))
        return np.mean(hed[:, :, 0])  # Mean hematoxylin channel

# Use with ScoreTiler
score_tiler = ScoreTiler(
    tile_size=(512, 512), n_tiles=30,
    scorer=StainIntensityScorer()
)
```

### Tile Preview

#### locate_tiles()

Preview tile locations before extraction to validate configuration.

```python
# RandomTiler: show n_tiles sample locations
random_tiler.locate_tiles(slide, n_tiles=20)

# GridTiler: show all grid positions
grid_tiler.locate_tiles(slide)

# ScoreTiler: show top-n tile locations
score_tiler.locate_tiles(slide, n_tiles=15)
```

Displays colored rectangles on the slide thumbnail indicating where tiles will be extracted.

### Extraction Workflows

#### Basic Extraction

```python
from histolab.slide import Slide
from histolab.tiler import RandomTiler

slide = Slide("slide.svs", processed_path="output/tiles/")
tiler = RandomTiler(tile_size=(512, 512), n_tiles=100, level=0, seed=42)
tiler.extract(slide)
# Tiles saved to: output/tiles/*.png
```

#### Extraction with Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
tiler.extract(slide)
# INFO: Tile 1/100 saved...
# INFO: Tile 2/100 saved...
```

#### Extraction with CSV Report

ScoreTiler generates a CSV report with tile metadata.

```python
score_tiler = ScoreTiler(
    tile_size=(512, 512), n_tiles=50,
    scorer=NucleiScorer()
)
score_tiler.extract(slide, report_path="tiles_report.csv")
```

Report columns:
```
tile_name,x_coord,y_coord,level,score,tissue_percent
tile_001.png,10240,5120,0,0.89,95.2
tile_002.png,15360,7680,0,0.85,91.7
```

### Advanced Patterns

#### Multi-Level Extraction

Extract tiles at different magnification levels for multi-scale analysis.

```python
for level in [0, 1, 2]:
    tiler = RandomTiler(
        tile_size=(512, 512), n_tiles=50,
        level=level, prefix=f"level{level}_"
    )
    tiler.extract(slide)
```

#### Hierarchical Extraction (Same Locations, Different Scales)

Use the same seed to extract tiles at the same positions across levels.

```python
for level in [0, 1]:
    tiler = RandomTiler(
        tile_size=(512, 512), n_tiles=30,
        level=level, seed=42,  # Same seed = same locations
        prefix=f"level{level}_"
    )
    tiler.extract(slide)
```

#### Post-Extraction Blur Filtering

Remove blurry tiles after extraction using Laplacian variance.

```python
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

def filter_blurry_tiles(tile_dir, threshold=100):
    """Remove tiles below blur quality threshold."""
    for tile_path in Path(tile_dir).glob("*.png"):
        img = Image.open(tile_path)
        gray = np.array(img.convert('L'))
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < threshold:
            tile_path.unlink()
            print(f"Removed blurry: {tile_path.name} (score={laplacian_var:.1f})")

tiler.extract(slide)
filter_blurry_tiles("output/tiles/", threshold=100)
```

### Performance Optimization

1. **Extract at appropriate level**: Level 1-2 is 4-16x faster than level 0
2. **Adjust tissue_percent**: Higher thresholds reduce invalid tile attempts
3. **Use BiggestTissueBoxMask**: Faster than TissueMask for single sections
4. **Limit n_tiles**: For RandomTiler and ScoreTiler initial exploration
5. **Use pixel_overlap=0**: Minimize tile count in GridTiler
6. **Enable logging**: Monitor progress to estimate completion time

### Tiler Selection Guide

| Criterion | RandomTiler | GridTiler | ScoreTiler |
|-----------|-------------|-----------|------------|
| Speed | Fast | Slow (large slides) | Medium |
| Coverage | Sampled | Complete | Top-N |
| Reproducibility | With seed | Deterministic | Deterministic |
| Output size | Fixed (n_tiles) | Variable | Fixed (n_tiles) |
| Best for | Exploration, training | Spatial analysis | Quality curation |
| Preserves spatial info | No | Yes | Partial |


Condensed from original: 421 lines. Retained: all three tiler strategies with full parameters, all scorer types (NucleiScorer, CellularityScorer, custom), locate_tiles preview, extraction workflows (basic, logging, CSV report), advanced patterns (multi-level, hierarchical, blur filtering), performance optimization, tiler selection guide. Omitted: ASCII grid overlap diagrams (trivial visual aid); per-tiler advantages/limitations prose shortened to bullet summaries.

## Reference: Visualization, Slides, and Tissue Masks Reference

Consolidated reference covering slide management, tissue mask classes, and visualization patterns for histolab.

### Slide Inspection

#### Loading and Properties

```python
from histolab.slide import Slide

slide = Slide("slide.svs", processed_path="output/")

# Core properties
print(f"Name: {slide.name}")                    # Filename without extension
print(f"Dimensions: {slide.dimensions}")         # (width, height) at level 0
print(f"Levels: {slide.levels}")                 # Number of pyramid levels
print(f"Level dimensions: {slide.level_dimensions}")

# OpenSlide metadata
props = slide.properties
print(f"Objective power: {props.get('openslide.objective-power', 'N/A')}")
print(f"MPP-X: {props.get('openslide.mpp-x', 'N/A')}")
print(f"MPP-Y: {props.get('openslide.mpp-y', 'N/A')}")
print(f"Vendor: {props.get('openslide.vendor', 'N/A')}")
```

#### Sample Datasets

Built-in TCGA samples for testing and demonstration:

```python
from histolab.data import prostate_tissue, ovarian_tissue, breast_tissue, heart_tissue, kidney_tissue

# Each returns (svs_data, path_to_file)
prostate_svs, prostate_path = prostate_tissue()
slide = Slide(prostate_path, processed_path="output/")
```

#### Pyramid Level Enumeration

```python
for level in range(slide.levels):
    dims = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    print(f"Level {level}: {dims}, downsample: {downsample:.0f}x")
```

#### Thumbnails and Scaled Images

```python
# Access thumbnail (PIL Image)
thumbnail = slide.thumbnail

# Save thumbnail to processed_path
slide.save_thumbnail()

# Get scaled image at specific factor
scaled = slide.scaled_image(scale_factor=32)
```

#### Region Extraction

```python
# Extract region at specific coordinates
region = slide.extract_region(
    location=(x, y),       # Top-left at level 0
    size=(width, height),  # Region size
    level=0
)
```

### Tissue Mask Classes

#### TissueMask

Segments all tissue regions using automated filters: grayscale conversion, Otsu threshold, binary dilation, small hole removal, small object removal.

```python
from histolab.masks import TissueMask

mask = TissueMask()
mask_array = mask(slide)  # Binary NumPy array: True=tissue
print(f"Tissue: {mask_array.sum() / mask_array.size * 100:.1f}%")
```

**Best for**: Multiple tissue sections, comprehensive analysis, when all regions matter.

#### BiggestTissueBoxMask

Returns bounding box of largest connected tissue region. Default mask for all tilers.

```python
from histolab.masks import BiggestTissueBoxMask

biggest = BiggestTissueBoxMask()
mask_array = biggest(slide)
```

**Best for**: Single main tissue section, excluding small fragments and artifacts.

#### Custom Masks with Filter Chains

```python
from histolab.masks import TissueMask
from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
from histolab.filters.morphological_filters import BinaryDilation, RemoveSmallHoles

custom = TissueMask(filters=[
    RgbToGrayscale(),
    OtsuThreshold(),
    BinaryDilation(disk_size=5),
    RemoveSmallHoles(area_threshold=500)
])
```

#### Custom BinaryMask Subclass

For region-of-interest or annotation-exclusion masks:

```python
from histolab.masks import BinaryMask
import numpy as np

class RectangularMask(BinaryMask):
    def __init__(self, x_start, y_start, width, height):
        self.x_start, self.y_start = x_start, y_start
        self.width, self.height = width, height

    def _mask(self, slide):
        thumb = slide.thumbnail
        mask = np.zeros(thumb.shape[:2], dtype=bool)
        mask[self.y_start:self.y_start+self.height,
             self.x_start:self.x_start+self.width] = True
        return mask

roi = RectangularMask(x_start=1000, y_start=500, width=2000, height=1500)
```

#### Annotation Exclusion Mask

Exclude pen markings (blue/green) from tissue detection:

```python
from histolab.masks import BinaryMask, TissueMask
import numpy as np
import cv2

class AnnotationExclusionMask(BinaryMask):
    def _mask(self, slide):
        thumb = slide.thumbnail
        hsv = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        pen_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        tissue_mask = TissueMask()(slide)
        return tissue_mask & ~pen_mask.astype(bool)
```

#### Mask-Tiler Integration

```python
from histolab.tiler import RandomTiler
from histolab.masks import TissueMask

tiler = RandomTiler(
    tile_size=(512, 512), n_tiles=100, level=0,
    extraction_mask=TissueMask()  # Override default BiggestTissueBoxMask
)
```

### Slide Visualization

#### Thumbnail Display

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.imshow(slide.thumbnail)
plt.title(f"Slide: {slide.name}")
plt.axis('off')
plt.show()
```

#### Mask Visualization with locate_mask()

```python
from histolab.masks import TissueMask, BiggestTissueBoxMask

# Built-in overlay display
slide.locate_mask(TissueMask())
slide.locate_mask(BiggestTissueBoxMask())
```

#### Manual Mask Comparison

```python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from histolab.masks import TissueMask

mask = TissueMask()
mask_array = mask(slide)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

axes[0].imshow(slide.thumbnail)
axes[0].set_title("Original"); axes[0].axis('off')

axes[1].imshow(mask_array, cmap='gray')
axes[1].set_title("Tissue Mask"); axes[1].axis('off')

overlay = slide.thumbnail.copy()
axes[2].imshow(overlay)
axes[2].imshow(mask_array, cmap=ListedColormap(['none', 'red']), alpha=0.3)
axes[2].set_title("Mask Overlay"); axes[2].axis('off')

plt.tight_layout()
plt.show()
```

#### Comparing Multiple Masks

```python
from histolab.masks import TissueMask, BiggestTissueBoxMask

masks = {'TissueMask': TissueMask(), 'BiggestTissueBoxMask': BiggestTissueBoxMask()}

fig, axes = plt.subplots(1, len(masks) + 1, figsize=(20, 6))
axes[0].imshow(slide.thumbnail); axes[0].set_title("Original"); axes[0].axis('off')

for idx, (name, mask) in enumerate(masks.items(), 1):
    axes[idx].imshow(mask(slide), cmap='gray')
    axes[idx].set_title(name); axes[idx].axis('off')

plt.tight_layout()
plt.show()
```

### Tile Visualization

#### Tile Location Preview

```python
from histolab.tiler import RandomTiler, GridTiler, ScoreTiler
from histolab.scorer import NucleiScorer

# Each tiler has locate_tiles() for preview
RandomTiler(tile_size=(512, 512), n_tiles=50, seed=42).locate_tiles(slide, n_tiles=20)
GridTiler(tile_size=(512, 512), level=0).locate_tiles(slide)
ScoreTiler(tile_size=(512, 512), n_tiles=30, scorer=NucleiScorer()).locate_tiles(slide, n_tiles=15)
```

#### Display Extracted Tiles

```python
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

tile_paths = list(Path("output/tiles/").glob("*.png"))[:16]
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for idx, ax in enumerate(axes.ravel()):
    if idx < len(tile_paths):
        ax.imshow(Image.open(tile_paths[idx]))
        ax.set_title(tile_paths[idx].stem, fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

#### Tile Mosaic

```python
def create_tile_mosaic(tile_dir, grid_size=(4, 4)):
    """Display tiles in a grid layout."""
    paths = list(Path(tile_dir).glob("*.png"))[:grid_size[0] * grid_size[1]]
    fig, axes = plt.subplots(*grid_size, figsize=(16, 16))
    for idx, path in enumerate(paths):
        row, col = idx // grid_size[1], idx % grid_size[1]
        axes[row, col].imshow(Image.open(path))
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig("tile_mosaic.png", dpi=150, bbox_inches='tight')
    plt.show()

create_tile_mosaic("output/tiles/", grid_size=(5, 5))
```

### Quality Assessment

#### Score Distribution

```python
import pandas as pd
import matplotlib.pyplot as plt

report = pd.read_csv("tiles_report.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].hist(report['score'], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Tile Score'); axes[0].set_ylabel('Frequency')
axes[0].set_title('Score Distribution')

axes[1].scatter(report['tissue_percent'], report['score'], alpha=0.5)
axes[1].set_xlabel('Tissue %'); axes[1].set_ylabel('Score')
axes[1].set_title('Score vs Tissue Coverage')
plt.tight_layout()
plt.show()
```

#### Top vs Bottom Tile Comparison

```python
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

report = pd.read_csv("tiles_report.csv").sort_values('score', ascending=False)
top_tiles = report.head(8)
bottom_tiles = report.tail(8)

fig, axes = plt.subplots(2, 8, figsize=(20, 6))
for idx, (_, row) in enumerate(top_tiles.iterrows()):
    axes[0, idx].imshow(Image.open(f"output/tiles/{row['tile_name']}"))
    axes[0, idx].set_title(f"{row['score']:.3f}", fontsize=8)
    axes[0, idx].axis('off')
for idx, (_, row) in enumerate(bottom_tiles.iterrows()):
    axes[1, idx].imshow(Image.open(f"output/tiles/{row['tile_name']}"))
    axes[1, idx].set_title(f"{row['score']:.3f}", fontsize=8)
    axes[1, idx].axis('off')
axes[0, 0].set_ylabel('Top'); axes[1, 0].set_ylabel('Bottom')
plt.tight_layout()
plt.savefig("score_comparison.png", dpi=150)
plt.show()
```

### Multi-Slide Visualization

#### Slide Collection Thumbnails

```python
from pathlib import Path
from histolab.slide import Slide
import matplotlib.pyplot as plt

slide_paths = list(Path("slides/").glob("*.svs"))[:9]
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for idx, (ax, sp) in enumerate(zip(axes.ravel(), slide_paths)):
    slide = Slide(str(sp), processed_path="output/")
    ax.imshow(slide.thumbnail)
    ax.set_title(slide.name, fontsize=10); ax.axis('off')
plt.tight_layout()
plt.savefig("slide_collection.png", dpi=150)
plt.show()
```

#### Tissue Coverage Comparison Across Slides

```python
from pathlib import Path
from histolab.slide import Slide
from histolab.masks import TissueMask
import matplotlib.pyplot as plt

slide_paths = list(Path("slides/").glob("*.svs"))
names, coverages = [], []
for sp in slide_paths:
    slide = Slide(str(sp), processed_path="output/")
    mask = TissueMask()(slide)
    coverages.append(mask.sum() / mask.size * 100)
    names.append(slide.name)

plt.figure(figsize=(12, 6))
plt.bar(range(len(names)), coverages)
plt.xticks(range(len(names)), names, rotation=45, ha='right')
plt.ylabel('Tissue Coverage (%)')
plt.title('Tissue Coverage Across Slides')
plt.tight_layout()
plt.show()
```

### Filter Effect Visualization

#### Multi-Step Pipeline Visualization

```python
from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
from histolab.filters.morphological_filters import BinaryDilation, RemoveSmallObjects
from histolab.filters.compositions import Compose
import matplotlib.pyplot as plt

steps = [
    ("Original", None),
    ("Grayscale", RgbToGrayscale()),
    ("Otsu", Compose([RgbToGrayscale(), OtsuThreshold()])),
    ("Dilated", Compose([RgbToGrayscale(), OtsuThreshold(), BinaryDilation(disk_size=5)])),
    ("Cleaned", Compose([RgbToGrayscale(), OtsuThreshold(), BinaryDilation(disk_size=5),
                         RemoveSmallObjects(area_threshold=500)]))
]

fig, axes = plt.subplots(1, len(steps), figsize=(20, 4))
for idx, (title, f) in enumerate(steps):
    if f is None:
        axes[idx].imshow(slide.thumbnail)
    else:
        axes[idx].imshow(f(slide.thumbnail), cmap='gray')
    axes[idx].set_title(title, fontsize=10); axes[idx].axis('off')
plt.tight_layout()
plt.show()
```

### Export and Reports

#### High-Resolution Figure Export

```python
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(slide.thumbnail); ax.axis('off')
plt.savefig("slide_hires.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
```

#### Multi-Page PDF Report

```python
from matplotlib.backends.backend_pdf import PdfPages
from histolab.masks import TissueMask
from histolab.tiler import RandomTiler

with PdfPages('slide_report.pdf') as pdf:
    # Page 1: Thumbnail
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.imshow(slide.thumbnail); ax1.set_title(f"Slide: {slide.name}"); ax1.axis('off')
    pdf.savefig(fig1); plt.close()

    # Page 2: Tissue mask
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.imshow(TissueMask()(slide), cmap='gray'); ax2.set_title("Tissue Mask"); ax2.axis('off')
    pdf.savefig(fig2); plt.close()
```


Condensed from 3 originals: visualization.md (547 lines) + slide_management.md (172 lines) + tissue_masks.md (251 lines) = 970 lines total. ~80 lines of overlapping content (thumbnail display, mask visualization, locate_mask patterns) deducted from denominator.

Retained: slide properties/metadata, sample datasets, pyramid level enumeration, region extraction, all 3 mask classes with code, custom mask patterns (rectangular, annotation exclusion), mask-tiler integration, all visualization categories (thumbnails, masks, tiles, quality assessment, multi-slide, filter effects, export, interactive Jupyter), PDF report generation.

Combined coverage: ~220 retained lines + ~65 lines relocated to SKILL.md Core API Modules 1-2 and Module 6 = ~285 lines / 890 effective lines = ~32% standalone, 80%+ capability coverage.

Omitted from slide_management.md: slide.name/scaled_image trivial property access (2 lines, self-evident from API). Omitted from tissue_masks.md: common issues section (4 items consolidated into main Troubleshooting table); best practices list (consolidated into main Best Practices). Omitted from visualization.md: interactive Jupyter ipywidgets exploration (niche use case, standard ipywidgets pattern); custom tile location visualization with manual coordinate calculation (conceptual example with empty tile_coords list, not practically useful); tile-with-tissue-mask-overlay using Tile.calculate_tissue_mask() (undocumented method, uncertain API stability).
