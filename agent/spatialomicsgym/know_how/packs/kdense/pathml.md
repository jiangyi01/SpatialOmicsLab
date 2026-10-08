# PathML

## Metadata

**Short Description**: Use PathML for local, research-only computational pathology workflows: load and tile slides, build preprocessing and QC pipelines, manage h5path data, quantify multiplex images, construct spatial graphs, and plan bounded model inference.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/pathml/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: GPL-2.0 (PathML, per this document's body, with upstream commercial licensing options; this skill's own text is MIT) (recorded in packs/MANIFEST.yaml)
**Commercial Use**: This text is MIT and may be used commercially. The wrapped library, PathML, is GPL-2.0 (with upstream commercial licensing options): that licence governs redistribution of the library itself, not of this description, and nothing from the library is vendored here.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 2 install fence(s) replaced by a provisioning pointer; 14 upstream script/reference path(s) marked as not vendored; 1 fence(s) left with nothing runnable replaced by a pointer; 1 manifest replacement(s) and 0 excision(s) applied.

---

## Scope and safety boundary

Use PathML for **local computational pathology research**. It is beta research
software, not a validated medical device, diagnostic system, clinical decision
support tool, or substitute for a pathologist. Do not use outputs to diagnose,
grade, stage, or treat a patient.

Pathology files may contain faces, labels, accession numbers, patient identifiers,
DICOM tags, filenames, or linked clinical data. Before processing:

1. Confirm authorization, consent/waiver, data-use terms, and institutional policy.
2. De-identify pixels and metadata; keep the re-identification key outside the
   analysis workspace.
3. Use pseudonymous `patient_id`, `slide_id`, and `specimen_id` values. Do not put
   direct identifiers in filenames, logs, `.h5path` labels, model cards, or reports.
4. Keep inputs, intermediates, and outputs on approved local encrypted storage.
5. Split by patient (then slide) before tiling or fitting any preprocessing step.

## Version baseline, verified 2026-07-23

- **Installable stable release:** PyPI `pathml==3.0.5`, published 2026-03-24.
- The v3.0.5 release notes state Python **3.10-3.12** and sunset 3.9.
  PyPI does not declare `Requires-Python` and still has a stale 3.8 classifier, so
  use the release statement and test the exact environment.
- GitHub releases v3.0.6 (2026-04-14) and v3.0.7 (2026-07-09) exist, but PyPI has
  no artifacts for them as of this review. v3.0.7 updates Torch/TorchVision/
  torch-geometric and ONNX export code. Do not mix those source dependencies with
  the 3.0.5 wheel.
- ReadTheDocs `/latest` identifies itself as 3.0.5. Examples here were checked
  against the v3.0.5 tag and PyPI wheel metadata, not unversioned snippets.
- This skill is MIT-licensed. PathML itself is GPL-2.0 with upstream commercial
  licensing options; review upstream terms before redistribution.

## Reproducible installation

Use Python 3.11 unless the project has tested another supported interpreter:

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

PathML 3.0.5 declares no package extras: do **not** use `pathml[all]`. Its base
distribution pins a large scientific/ML stack, including Torch 2.8.0, ONNX 1.17.0,
ONNX Runtime 1.17.x, OpenSlide Python 1.3.1, python-bioformats 4.1.0, and
python-javabridge 4.0.4.

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

Java/Bio-Formats is needed for the broad multidimensional format backend.
OpenSlide handles common brightfield WSI formats more efficiently. CUDA is
optional and must match the pinned PyTorch build; follow PyTorch's platform
selector rather than guessing a CUDA wheel. See (upstream reference file, not included).

## Stable minimal workflow

PathML 3.0.5 uses slide convenience classes and `SlideData.run()`. It does not
provide `SlideData.from_slide()`, and `Pipeline` does not have `run()`:

```python
from pathml.core import HESlide
from pathml.preprocessing import BoxBlur, Pipeline, TissueDetectionHE

slide = HESlide("data/pseudonymous_slide.svs", backend="openslide")
pipeline = Pipeline(
    [
        BoxBlur(kernel_size=5),
        TissueDetectionHE(mask_name="tissue", min_region_size=5000),
    ]
)
slide.run(
    pipeline,
    distributed=False,
    tile_size=512,
    tile_stride=512,
    level=0,
    tile_pad=False,
)
slide.write("derived/pseudonymous_slide.h5path")
```

Start with a bounded manual sample before a full run:

```python
from itertools import islice

for tile in islice(slide.generate_tiles(shape=512, stride=512, level=0), 8):
    pipeline.apply(tile)
    assert tile.masks["tissue"].shape[:2] == tile.image.shape[:2]
```

Tiles use `(i, j)` = `(row, column)` coordinates at the selected pyramid level.
For OpenSlide, PathML maps them to level-0 coordinates internally. Record the
level and downsample; convert to `(x, y)` or micrometres explicitly downstream.

## Research workflow

1. **Inventory locally.** Validate the manifest, reject URLs/symlinks, inspect only
   allowlisted technical metadata, and remove identifiers.
2. **Freeze splits.** Assign every patient and all their slides to one split before
   generating overlapping tiles, graphs, normalization references, or features.
3. **Plan bounds.** Estimate tile count, RAM, output size, and pipeline stages.
4. **Pilot preprocessing.** Inspect tissue masks, whitespace/artifact labels,
   stain behavior, edge padding, and empty-mask cases on representative training
   slides. Do not tune from test slides.
5. **Run and preserve coordinates.** Keep tile level, `(i, j)`, downsample, MPP,
   mask names, QC decisions, and failed/skipped tiles.
6. **Build spatial data deliberately.** Validate channel order, physical units,
   instance labels, node-feature alignment, graph edges, and cell-to-tissue
   assignments.
7. **Infer in bounded batches.** Verify model provenance and checksum without
   loading unknown pickle checkpoints. Keep predictions linked to slide/tile
   coordinates and stitch overlaps with a documented rule.
8. **Report provenance and limits.** Include package lock, source hashes, scanner,
   stain, parameters, seeds, split manifest, model card, exclusions, and QC.

## No-network default and explicit consent gate

Do not instantiate download-capable classes or set dataset `download=True` unless
the user explicitly opts in after receiving the endpoint and disclosure:

- `SegmentMIFRemote` downloads an ONNX file from
  `https://huggingface.co/pathml/test/resolve/main/mesmer.onnx` at construction,
  then runs inference locally. Stable source does **not** upload image pixels.
  The request still discloses network metadata such as IP address and headers and
  creates `temp.onnx`; there is no built-in checksum or offline flag.
- Deprecated `SegmentMIF` imports local DeepCell Mesmer, but DeepCell model
  initialization may need separately provisioned weights. It is not a PathML
  extra and is not the preferred stable API.
- `RemoteTestHoverNet` downloads a model from Hugging Face.
- `PanNukeDataModule(download=True)` contacts Warwick; `DeepFocusDataModule`
  contacts Zenodo. Both default to `download=False`.

Before any future hosted prediction call, state the exact destination, pixel
channels/regions, metadata, identifiers, retention, legal basis, and safeguards;
obtain explicit consent; and never send PHI by default. Prefer reviewed,
checksummed local model artifacts and local inference.

## Model-code security

- PyTorch `model.eval()` means **evaluation mode** for modules; it is not Python's
  dangerous built-in evaluator. Never use Python dynamic evaluation or execution.
- Do not name local files `pathml.py`, `torch.py`, `onnx.py`, or after standard
  libraries; shadow modules can silently change imports.
- PathML's `EntityDataset` loads `.pt` objects with `weights_only=False`. Never
  open an untrusted graph/checkpoint. Treat pickle-based pipelines and `.pt` files
  as executable code.
- ONNX is safer than pickle but not inherently trusted. Verify source, SHA-256,
  expected input/output schema, file size, and runtime limits; use isolation for
  third-party models.

## Upstream local CLIs (not vendored)

All helpers reject URLs and symlinks, cap inputs/work, use strict JSON, avoid
network access, and require no PathML import for `--help`:

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

The inference planner reads numbers or a bounded JSON model card only; it never
imports a model framework or opens a checkpoint.

## Detailed references

- (upstream reference file, not included) — slide classes, backends, formats, levels,
  coordinates, technical metadata, and privacy.
- (upstream reference file, not included) — stable transforms, masks/QC, stain processing,
  pipeline execution, and leakage prevention.
- (upstream reference file, not included) — `.h5path`, manifests, datasets, provenance,
  splits, and safe downloads.
- (upstream reference file, not included) — multidimensional layout, CODEX/Vectra,
  quantification, AnnData, DeepCell/Mesmer, and network disclosure.
- (upstream reference file, not included) — instance maps, feature alignment, KNN/RAG/HACT graphs,
  spatial units, schemas, and validation.
- (upstream reference file, not included) — HoVer-Net/HACTNet, local ONNX inference,
  batching, checkpoint trust, evaluation, and model provenance.

## Primary sources

All checked 2026-07-23:

- PyPI metadata: https://pypi.org/project/pathml/3.0.5/
- Stable source tag: https://github.com/Dana-Farber-AIOS/pathml/tree/v3.0.5
- Releases: https://github.com/Dana-Farber-AIOS/pathml/releases
- Stable documentation: https://pathml.readthedocs.io/en/stable/
- Rosenthal et al. (2022), PathML toolkit:
  https://doi.org/10.1158/1541-7786.MCR-21-0665
- Omar et al. (2025), multiplex workflows:
  https://doi.org/10.1016/j.labinv.2025.104220
