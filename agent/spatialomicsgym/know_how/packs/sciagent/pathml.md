# pathml

## Metadata

**Short Description**: Computational pathology toolkit for whole-slide images (WSIs): load slides, extract tiles, stain normalization, nuclear segmentation, feature extraction, and ML training. Supports H&E and multiplex. For end-to-end pipelines from raw WSIs to quantitative outputs.
**Source**: https://github.com/jaechang-hits/SciAgent-Skills/blob/fe505cae14d20b6c33be2e49666425be98f005bb/skills/medical-imaging/pathml/SKILL.md
**License**: CC BY 4.0, Copyright (c) 2024 jaechang-hits (THIRD_PARTY_LICENSES/SciAgent-Skills-CC-BY-4.0.txt). Upstream inherits K-Dense-AI/claude-scientific-skills (MIT) and snap-stanford/Biomni (Apache-2.0); those notices travel with this text. Changes were made -- see Modifications.
**Wrapped Tool License**: GPL-2.0 (PathML, per its source tag and the kdense pathml document) (recorded in packs/MANIFEST.yaml)
**Commercial Use**: This text is CC BY 4.0 and may be used commercially. The wrapped library, PathML, is GPL-2.0: that licence governs redistribution of the library itself, not of this description, and nothing from the library is vendored here.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 install fence(s) replaced by a provisioning pointer; 2 inline install span(s) rewritten; 22 manifest replacement(s) and 0 excision(s) applied.

---

## Overview

PathML is a Python toolkit designed for computational pathology workflows on whole-slide images (WSIs). It provides a unified pipeline from raw slide files (SVS, NDPI, MRXS, TIFF) through tile extraction, preprocessing (stain normalization, nuclear segmentation, tissue detection), feature extraction, and machine learning. PathML integrates with popular Python ML and image processing libraries while abstracting the complexity of WSI handling through its `SlideData` and `Pipeline` abstractions.

## When to Use

- **Processing whole-slide H&E images**: Tiling a large WSI, normalizing staining variability across slides from different scanners or batches.
- **Nuclear segmentation on pathology slides**: Detecting and segmenting nuclei in H&E or DAPI-stained WSIs using built-in segmentation pipelines.
- **Building ML training datasets from WSIs**: Extracting tiles with associated labels for training tissue classifiers, tumor detectors, or survival prediction models.
- **Multiplex immunofluorescence (mIF) image analysis**: Processing multi-channel IF slides with channel-specific preprocessing and feature extraction.
- **Stain normalization across cohorts**: Applying Macenko or Vahadane stain normalization to harmonize H&E slides from multiple institutions.
- **Feature extraction for downstream ML**: Extracting handcrafted or deep learning features from tiles for patient-level prediction tasks.
- For standard 2D microscopy images (non-WSI), use `scikit-image` or `cellpose` directly without PathML overhead.

## Prerequisites

- **Python packages**: `pathml`, `torch`, `torchvision`, `numpy`, `scikit-image`, `openslide-python`
- **System**: OpenSlide C library (required for WSI reading)
- **Data requirements**: WSI files in SVS, NDPI, MRXS, or TIFF format; GPU recommended for segmentation
- **Environment**: Python 3.10-3.12 (PathML 3.0.5); the code below is checked against the PathML v3.0.5 source: `HESlide`/`SlideData.run()`, `StainNormalizationHE`, `NucleusDetectionHE`, and `.h5path` files read back with `SlideData(path)`

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

## Quick Start

```python
from pathml.core import HESlide
from pathml.preprocessing import BoxBlur, Pipeline, TissueDetectionHE

# Load → build pipeline → tile and preprocess in one call (SlideData.run() tiles the slide)
slide = HESlide("tumor.svs", name="demo", backend="openslide")
pipeline = Pipeline([BoxBlur(kernel_size=3), TissueDetectionHE(mask_name="tissue")])
slide.run(pipeline, distributed=False, tile_size=256, tile_stride=256, level=0)

# Inspect tiles
tiles = [t for t in slide.tiles if t.masks["tissue"].any()]
print(f"Tissue tiles: {len(tiles)} of {len(slide.tiles)}")
```

## Workflow

### Step 1: Load a Whole-Slide Image

```python
from pathml.core import HESlide

# Load an H&E whole-slide image (HESlide is SlideData with the H&E slide type)
slide = HESlide("path/to/slide.svs", name="tumor_slide_001", backend="openslide")
print(f"Slide name: {slide.name}")
print(f"Slide shape: {slide.shape}")
print(f"Pyramid levels: {slide.slide.level_count}")
```

### Step 2: Define a Preprocessing Pipeline

```python
from pathml.preprocessing import BoxBlur, Pipeline, StainNormalizationHE, TissueDetectionHE

# Build a preprocessing pipeline for H&E slides
pipeline = Pipeline([
    BoxBlur(kernel_size=5),                         # smooth image
    TissueDetectionHE(mask_name="tissue"),          # detect tissue regions
    StainNormalizationHE(target="normalize", stain_estimation_method="macenko"),  # normalize H&E staining
])
print(f"Pipeline steps: {len(pipeline)}")
```

### Step 3: Preview Tiles

```python
from itertools import islice

# SlideData.run() (Step 4) tiles the slide itself. To look at a few tiles first, iterate the
# generate_tiles() generator -- it does not add anything to slide.tiles.
for tile in islice(slide.generate_tiles(shape=256, stride=256, level=0), 4):
    print(tile.coords, tile.image.shape)
```

### Step 4: Run the Preprocessing Pipeline

```python
# Tile the slide and apply the pipeline to every tile (distributed=False: no Dask cluster needed)
slide.run(pipeline, distributed=False, tile_size=256, tile_stride=256, level=0, tile_pad=False)
print(f"Pipeline complete: {len(slide.tiles)} tiles")

# Inspect a single tile
tile = slide.tiles[0]
print(f"Tile shape: {tile.image.shape}")      # (256, 256, 3)
print(f"Tile masks: {list(tile.masks.keys())}")
```

### Step 5: Nuclear Segmentation

```python
from pathml.preprocessing import NucleusDetectionHE
from skimage.measure import label

# PathML 3.0.5 has no NuclearSegmentation transform. NucleusDetectionHE writes a binary nucleus
# mask (hematoxylin channel + superpixels + Otsu), so nuclei are counted as connected components.
seg_pipeline = Pipeline([
    TissueDetectionHE(mask_name="tissue"),
    NucleusDetectionHE(mask_name="nuclei"),
])

slide.run(seg_pipeline, distributed=False, tile_size=256, tile_stride=256, overwrite_existing_tiles=True)

# Count nuclei in the first tiles
for i in range(min(5, len(slide.tiles))):
    tile = slide.tiles[i]
    n_nuclei = label(tile.masks["nuclei"] > 0).max()
    print(f"Tile {tile.coords}: {n_nuclei} nuclei detected")
```

### Step 6: Feature Extraction

```python
import numpy as np
from skimage.measure import label

features = []
for tile in slide.tiles:
    if "tissue" in tile.masks and tile.masks["tissue"].any():
        img = tile.image
        feat = {
            "mean_r":    img[:, :, 0].mean(),
            "mean_g":    img[:, :, 1].mean(),
            "mean_b":    img[:, :, 2].mean(),
            "std_r":     img[:, :, 0].std(),
            "n_nuclei":  int(label(tile.masks["nuclei"] > 0).max()) if "nuclei" in tile.masks else 0,
            "tile_x":    tile.coords[0],
            "tile_y":    tile.coords[1],
        }
        features.append(feat)

import pandas as pd
df = pd.DataFrame(features)
df.to_csv("slide_features.csv", index=False)
print(f"Extracted features from {len(df)} tissue tiles -> slide_features.csv")
```

### Step 7: Save and Export Processed Slide

```python
# Save slide data (tiles + masks) in PathML's h5path format
slide.write("processed_slide.h5path")
print("Slide saved to processed_slide.h5path")

# Reload: SlideData reads an .h5path file directly (there is no SlideData.read)
from pathml.core import SlideData
slide_loaded = SlideData("processed_slide.h5path")
print(f"Reloaded: {len(slide_loaded.tiles)} tiles")
```

## Key Parameters

| Parameter | Default | Range / Options | Effect |
|-----------|---------|-----------------|--------|
| `tile_size` | `256` | `64` – `1024` | Tile edge in pixels (`SlideData.run`) |
| `tile_stride` | equals `tile_size` | any value ≤ `tile_size` | Step between tiles; smaller than `tile_size` gives overlapping tiles |
| `level` | `0` | `0` – max pyramid level | Pyramid resolution level (0 = full resolution) |
| `kernel_size` | `5` | odd integers `3`–`21` | Smoothing kernel size in `BoxBlur` |
| `mask_name` | required | any string | Name of output mask stored in `tile.masks` |
| `distributed` | `True` | `True`, `False` | Dask distributed processing (needs a Dask client); pass `False` to run locally |
| `tile_pad` | `False` | `True`, `False` | Pad edge tiles to the full tile size |

## Common Recipes

### Recipe: Tissue-Only Tile Filtering

When to use: Exclude background tiles to reduce memory and computation in downstream steps.

```python
# Filter tiles to only tissue regions after running tissue detection pipeline
tissue_tiles = [t for t in slide.tiles if "tissue" in t.masks and t.masks["tissue"].mean() > 0.5]
print(f"Tissue tiles: {len(tissue_tiles)} / {len(slide.tiles)} total")
```

### Recipe: Export Tiles as PNG Files

When to use: Create a labeled tile dataset for training a custom classifier in PyTorch.

```python
from PIL import Image
import numpy as np
from pathlib import Path

output_dir = Path("tiles_png")
output_dir.mkdir(exist_ok=True)

for i, tile in enumerate(slide.tiles):
    if "tissue" in tile.masks and tile.masks["tissue"].mean() > 0.5:
        img = Image.fromarray(tile.image.astype(np.uint8))
        img.save(output_dir / f"tile_{i:05d}_x{tile.coords[0]}_y{tile.coords[1]}.png")

print(f"Saved {i+1} tiles to {output_dir}/")
```

### Recipe: Batch Process Multiple Slides

When to use: Running the same preprocessing pipeline on a directory of WSI files.

```python
from pathlib import Path
from pathml.core import HESlide
from pathml.preprocessing import Pipeline, StainNormalizationHE, TissueDetectionHE

pipeline = Pipeline([
    TissueDetectionHE(mask_name="tissue"),
    StainNormalizationHE(target="normalize", stain_estimation_method="macenko"),
])

wsi_dir = Path("slides/")
for wsi_path in sorted(wsi_dir.glob("*.svs")):
    slide = HESlide(str(wsi_path), name=wsi_path.stem, backend="openslide")
    slide.run(pipeline, distributed=False, tile_size=256, tile_stride=256, level=0)
    slide.write(f"processed/{wsi_path.stem}.h5path")
    print(f"Processed {wsi_path.name}: {len(slide.tiles)} tiles")
```

## Expected Outputs

- `slide.tiles` — iterable of `Tile` objects, each with `.image` (numpy array) and `.masks` (dict of numpy arrays)
- `slide_features.csv` — tabular per-tile features (color statistics, nucleus counts, coordinates)
- `processed_slide.h5path` — PathML h5path file with tiles, masks, and metadata; reload with `SlideData("processed_slide.h5path")`
- PNG tile files (optional) — ready for PyTorch `ImageFolder` dataset loading

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `openslide.lowlevel.OpenSlideUnsupportedFormatError` | OpenSlide C library not installed or WSI format unsupported | (not installed from this document: if the import fails, the library is not available here -- say so, do not install it); check format compatibility |
| `CUDA out of memory` during segmentation | Tile size too large for GPU | Reduce `tile_size` to 128, or run with `distributed=False` on CPU |
| `slide.tiles` is empty | `generate_tiles()` is a generator and never fills `slide.tiles`, or the level is out of range | Tile with `slide.run(...)`; check `slide.slide.level_count` |
| Stain normalization produces black tiles | Source slide too low contrast or failed tissue detection | Apply `TissueDetectionHE` before normalization; inspect tissue mask coverage |
| `KeyError: 'nuclei'` in tile.masks | Nucleus detection not yet run | Run a pipeline with `NucleusDetectionHE(mask_name="nuclei")` via `slide.run()` before accessing masks |
| Very slow tile generation | High-resolution level 0 on large SVS | Use a lower pyramid level (`level=1` or `level=2`) for faster prototyping |
| `AttributeError: SlideData has no attribute 'write'` | A PathML far older than 3.0.5 | This document targets PathML 3.0.5; with an older version, say the h5path API is not available here |

## References

- [PathML Documentation](https://pathml.readthedocs.io/) — official docs with tutorials
- [PathML GitHub (Dana-Farber/PathML)](https://github.com/Dana-Farber-AIOS/pathml) — source code and examples
- [Rosenthal et al. (2022), Cell Systems — PathML paper](https://doi.org/10.1016/j.cels.2022.01.004) — original publication
- [OpenSlide Documentation](https://openslide.org/) — WSI reading library underlying PathML
