"""Spatial transcriptomics image processing pipeline.

Processes histology and fluorescence images and embeds them into AnnData .h5ad files
in the format expected by SpatialOmicsLab MCP spatial analysis tools.

MCP tools consume images in two ways:
  1. Embedded in h5ad: adata.uns['spatial'][library_id]['images']['hires'/'lowres']
     + adata.uns['spatial'][library_id]['scalefactors']
     Used by: stLearn, Cell2Location, scanpy_spatial, Starfysh, SpatialPrompt
  2. Separate file path: histology_image_path parameter
     Used by: MISO (.tif), iStar

This module handles:
  - H&E stained histology (10x Visium, general histology)
  - Multi-channel fluorescence (Xenium, MERFISH, CosMx, STARmap)
  - ssDNA / DAPI staining (Stereo-seq, Xenium)
  - Image registration to spatial coordinates via scale factors
  - OME-TIFF and standard TIFF multi-channel images
  - Large image tiling and downsampling
  - Tissue detection and masking
"""

from __future__ import annotations

import json
import numbers
import os
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import scanpy as sc

# Pixel size of a full-resolution Xenium morphology OME-TIFF, per 10x's Xenium output specification.
# Xenium obsm['spatial'] is in microns, so this is the constant that ties the two together.
XENIUM_MORPHOLOGY_UM_PER_PX = 0.2125

# ---------------------------------------------------------------------------
# Core: Embed image into h5ad (MCP-compatible)
# ---------------------------------------------------------------------------


def embed_image_in_h5ad(
    h5ad_path: str,
    image_path: str,
    output_path: str | None = None,
    library_id: str = "spatial_sample",
    hires_target_px: int = 2000,
    lowres_target_px: int = 600,
    spot_diameter_fullres: float | None = None,
    microns_per_pixel: float | None = None,
    _preloaded_image: np.ndarray | None = None,
    _fullres_shape: tuple[int, int] | None = None,
    _metadata: dict | None = None,
) -> str:
    """Embed a histology/fluorescence image into an h5ad file for MCP tool compatibility.

    Loads the image, creates hires and lowres versions, computes scale factors
    that link pixel coordinates to obsm['spatial'], and stores everything in
    adata.uns['spatial'] in the standard 10x Visium format.

    This makes the h5ad compatible with stLearn (CNN feature extraction),
    Cell2Location (visualization), Starfysh, SpatialPrompt, and other MCP tools
    that expect embedded images.

    A scale factor answers "multiply obsm['spatial'] by what to get hires-image pixels?", so it has
    two parts: the image downsample ratio, and the units of obsm['spatial'] itself. Visium
    coordinates are already full-resolution image pixels, so the second part is 1 and
    ``microns_per_pixel`` is left None. Xenium and MERSCOPE coordinates are in MICRONS, and there
    the image's um/px has to be divided out or every overlay is off by that factor.

    Args:
        h5ad_path: Path to the input h5ad file (must have obsm['spatial']).
        image_path: Path to the image file (PNG, TIFF, JPG, or OME-TIFF).
        output_path: Path for the output h5ad. Defaults to a new ``<input>_image.h5ad`` under the
            session's output directory; the input is overwritten only when its own path is passed.
        library_id: Library ID key for adata.uns['spatial'].
        hires_target_px: Target max dimension for hires image in pixels.
        lowres_target_px: Target max dimension for lowres image in pixels.
        spot_diameter_fullres: Spot diameter in full-resolution pixels. If None, auto-estimated.
        microns_per_pixel: Physical size of one full-resolution image pixel, when obsm['spatial'] is
            in microns. None (default) means the coordinates are already in full-resolution pixels.
        _preloaded_image: Internal: pre-loaded image array (skips file loading).
        _fullres_shape: Internal: (height, width) of the full-resolution image when
            ``_preloaded_image`` was already reduced by the caller.
        _metadata: Internal: stored as uns['spatial'][library_id]['metadata'] (e.g. the source
            platform, which tells generate_tissue_mask that tissue is bright on a dark background).

    Returns:
        str: JSON report with embedding details and output path.

    """
    adata = sc.read_h5ad(h5ad_path)
    notes: list[str] = []

    try:
        coord_scale = _coordinate_scale(microns_per_pixel)
    except ValueError as exc:
        return json.dumps({"status": "error", "message": str(exc)}, indent=2)

    if "spatial" not in adata.obsm:
        return json.dumps({"status": "error", "message": "h5ad has no obsm['spatial'] coordinates"})

    if _preloaded_image is not None:
        raw_img = _preloaded_image
    else:
        raw_img = _load_image(image_path, warnings_out=notes, native=True)

    coords = np.asarray(adata.obsm["spatial"])
    fullres_h, fullres_w = (int(n) for n in (_fullres_shape or raw_img.shape[:2]))

    # Two distinct factors that used to be one: how far the image is shrunk, and how many
    # full-resolution pixels one coordinate unit is worth.
    hires_img_scale = min(hires_target_px / max(fullres_h, fullres_w), 1.0)
    lowres_img_scale = min(lowres_target_px / max(fullres_h, fullres_w), 1.0)
    hires_hw = (int(fullres_h * hires_img_scale), int(fullres_w * hires_img_scale))
    lowres_hw = (int(fullres_h * lowres_img_scale), int(fullres_w * lowres_img_scale))

    # Shrunk in the image's own dtype before anything makes it float RGB (hunt 2026-09-30,
    # uT8-imaging-6): see _reduce_native.
    img = _ensure_rgb(_reduce_native(raw_img, hires_hw))
    del raw_img

    hires_img = _resize_image(img, hires_img_scale, size=hires_hw)
    lowres_img = _resize_image(img, lowres_img_scale, size=lowres_hw)

    if spot_diameter_fullres is None:
        # The estimate comes back in coordinate units; this field is consumed as full-res pixels.
        spot_diameter_fullres = _estimate_spot_diameter(coords) * coord_scale

    scalefactors = {
        "tissue_hires_scalef": float(hires_img_scale * coord_scale),
        "tissue_lowres_scalef": float(lowres_img_scale * coord_scale),
        "spot_diameter_fullres": float(spot_diameter_fullres),
        "fiducial_diameter_fullres": float(spot_diameter_fullres * 1.2),
    }

    fit_warning = _coordinate_fit_warning(
        coords, scalefactors["tissue_hires_scalef"], hires_img.shape[:2], microns_per_pixel
    )
    if fit_warning:
        notes.append(fit_warning)

    entry = {"images": {"hires": hires_img, "lowres": lowres_img}, "scalefactors": scalefactors}
    if _metadata:
        entry["metadata"] = dict(_metadata)
    _store_library(adata, library_id, entry, notes)

    out_path = _write_h5ad(adata, output_path or _default_output_path(h5ad_path, "image"), notes)

    return json.dumps(
        {
            "status": "success",
            "output_path": out_path,
            "library_id": library_id,
            "fullres_shape": [fullres_h, fullres_w],
            "hires_shape": list(hires_img.shape[:2]),
            "lowres_shape": list(lowres_img.shape[:2]),
            "scalefactors": scalefactors,
            "n_channels": img.shape[2] if img.ndim == 3 else 1,
            "microns_per_pixel": microns_per_pixel,
            "warnings": notes or None,
        },
        indent=2,
    )


def export_image_from_h5ad(
    h5ad_path: str,
    output_dir: str,
    quality: str = "hires",
    format: str = "png",
) -> str:
    """Export embedded images from h5ad to separate files for MCP tools needing file paths.

    Extracts images from adata.uns['spatial'] and saves as PNG/TIFF files.
    Useful for tools like MISO that need a histology_image_path parameter.

    Args:
        h5ad_path: Path to h5ad with embedded images.
        output_dir: Directory to save exported images.
        quality: Image quality to export ('hires', 'lowres', or 'all').
        format: Output format ('png' or 'tiff').

    Returns:
        str: JSON report with exported file paths.

    """
    from PIL import Image

    adata = sc.read_h5ad(h5ad_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    libraries = _spatial_libraries(adata.uns.get("spatial"))
    if not libraries:
        return json.dumps({"status": "error", "message": "No spatial images embedded in h5ad"})

    exported = {}
    scalefactors = {}

    for lib_id, lib_data in libraries.items():
        images = lib_data.get("images", {})
        sf = lib_data.get("scalefactors", {})
        scalefactors[lib_id] = {k: (v.item() if hasattr(v, "item") else v) for k, v in dict(sf).items()}

        qualities = list(images.keys()) if quality == "all" else [quality]
        for q in qualities:
            if q not in images:
                continue
            img_array = images[q]
            # Convert float [0,1] to uint8 if needed
            if img_array.dtype in (np.float32, np.float64):
                img_array = (img_array * 255).clip(0, 255).astype(np.uint8)

            suffix = "tif" if format == "tiff" else "png"
            filename = f"{lib_id}_{q}.{suffix}"
            filepath = out_dir / filename

            Image.fromarray(img_array).save(str(filepath))
            exported[f"{lib_id}_{q}"] = str(filepath)

    # Also export scale factors, in Space Ranger's flat format: that is what a reader of a
    # ``scalefactors_json.json`` beside the image expects (MISO's loader among them), and the
    # ``{library: {...}}`` nesting written before gave it no ``tissue_*_scalef`` key at all. With
    # several libraries each gets its own flat file rather than one file no reader can use.
    if len(scalefactors) == 1:
        sf_path = out_dir / "scalefactors_json.json"
        with open(sf_path, "w") as f:
            json.dump(next(iter(scalefactors.values())), f, indent=2)
        exported["scalefactors_json"] = str(sf_path)
    else:
        for lib_id, sf in scalefactors.items():
            sf_path = out_dir / f"{lib_id}_scalefactors_json.json"
            with open(sf_path, "w") as f:
                json.dump(sf, f, indent=2)
            exported[f"{lib_id}_scalefactors_json"] = str(sf_path)

    return json.dumps({"status": "success", "exported_files": exported}, indent=2)


# ---------------------------------------------------------------------------
# Platform-specific image processors
# ---------------------------------------------------------------------------


def process_visium_images(
    spatial_dir: str,
    h5ad_path: str,
    output_path: str | None = None,
    library_id: str = "spatial_sample",
) -> str:
    """Load and embed 10x Visium H&E images from a spatial/ directory into an h5ad.

    Reads tissue_hires_image.png, tissue_lowres_image.png, and scalefactors_json.json
    from the standard Space Ranger spatial/ output directory.

    Args:
        spatial_dir: Path to the spatial/ directory from Space Ranger output.
        h5ad_path: Path to the h5ad file to add images to.
        output_path: Output path for the updated h5ad. Defaults to a new ``<input>_image.h5ad`` under
            the session's output directory; the input is overwritten only when its own path is passed.
        library_id: Library ID for adata.uns['spatial'].

    Returns:
        str: JSON report with embedding details.

    """
    from PIL import Image

    sp_dir = Path(spatial_dir)
    adata = sc.read_h5ad(h5ad_path)

    images = {}
    for name, key in [("tissue_hires_image.png", "hires"), ("tissue_lowres_image.png", "lowres")]:
        img_file = sp_dir / name
        if img_file.exists():
            img = np.array(Image.open(str(img_file)))
            # Normalize to float32 [0,1] as scanpy convention
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            images[key] = img

    if not images:
        return json.dumps({"status": "error", "message": f"No tissue images found in {spatial_dir}"})

    # Load scale factors
    sf_path = sp_dir / "scalefactors_json.json"
    scalefactors = {}
    if sf_path.exists():
        with open(sf_path) as f:
            scalefactors = json.load(f)

    # Without the scale factors nothing maps obsm['spatial'] (full-resolution pixels) onto these
    # images: scanpy/squidpy plots, generate_tissue_mask and MISO's frame detection then index
    # full-resolution coordinates into a hires image. That was reported as a plain success
    # (hunt 2026-09-30, uT8-imaging-30).
    notes: list[str] = []
    missing = [f"tissue_{key}_scalef" for key in images if f"tissue_{key}_scalef" not in scalefactors]
    if missing:
        where = "it has no " + ", ".join(missing) if sf_path.exists() else f"{sf_path} does not exist"
        notes.append(
            f"The images were embedded without the scale factors that place spots on them ({where}). "
            "Every tool that overlays obsm['spatial'] on these images will put the spots in the wrong "
            "place; supply Space Ranger's scalefactors_json.json in the spatial/ directory and re-run."
        )

    _store_library(adata, library_id, {"images": images, "scalefactors": scalefactors}, notes)

    out_path = _write_h5ad(adata, output_path or _default_output_path(h5ad_path, "image"), notes)

    return json.dumps(
        {
            "status": "success",
            "output_path": out_path,
            "images_loaded": list(images.keys()),
            "image_shapes": {k: list(v.shape) for k, v in images.items()},
            "scalefactors": scalefactors,
            "warnings": notes or None,
        },
        indent=2,
    )


def process_xenium_images(
    morphology_path: str,
    h5ad_path: str,
    output_path: str | None = None,
    channel: int | str = 0,
    library_id: str = "spatial_sample",
    hires_target_px: int = 2000,
    lowres_target_px: int = 600,
    microns_per_pixel: float | None = XENIUM_MORPHOLOGY_UM_PER_PX,
) -> str:
    """Process 10x Xenium morphology images (DAPI/fluorescence) and embed in h5ad.

    Xenium produces OME-TIFF morphology images (morphology_focus.ome.tif or
    morphology_mip.ome.tif) that are multi-channel. This extracts the specified
    channel (or creates a composite), converts to a pseudo-grayscale or RGB image,
    and embeds it in the h5ad.

    Xenium Onboarding Analysis >= 2.0 ships ``morphology_focus/`` as a *directory* of one OME-TIFF
    per stain (0000 = DAPI, then the boundary stains); passing that directory works and its files
    are combined as channels.

    Args:
        morphology_path: Path to Xenium morphology image (.ome.tif, .tif, or .png), or to an
            XOA >= 2.0 ``morphology_focus/`` directory.
        h5ad_path: Path to the h5ad file (must have obsm['spatial']).
        output_path: Output path for updated h5ad (default: a new file under the session's output
            directory, never the input).
        channel: Channel index (int, or a string of digits such as "1") or 'composite' for
            multi-channel merge. Any other string is refused.
        library_id: Library ID for adata.uns['spatial'].
        hires_target_px: Target max dimension for hires image.
        lowres_target_px: Target max dimension for lowres image.
        microns_per_pixel: Physical size of one morphology-image pixel. Xenium obsm['spatial'] is in
            microns, so this is what converts it to image pixels; the default is the documented
            full-resolution Xenium value. Pass the real value for a downsampled image, or None to
            state that the coordinates are already in this image's pixels.

    Returns:
        str: JSON report with processing details.

    """
    notes: list[str] = []
    # The description advertised ``channel`` as a string ("0") while every branch below keys on
    # ``isinstance(channel, int)``, so channel="1" fell through to an RGB composite of the first
    # three planes and the report never said the request was ignored (hunt 2026-09-30,
    # uT8-imaging-2). A string of digits is the index it spells; any other string but
    # 'composite' is refused.
    if isinstance(channel, numbers.Integral) and not isinstance(channel, bool):
        channel = int(channel)
    elif isinstance(channel, str) and channel.strip().lower() == "composite":
        channel = "composite"
    elif isinstance(channel, str) and channel.strip().lstrip("-").isdigit():
        channel = int(channel.strip())
    else:
        return json.dumps(
            {
                "status": "error",
                "message": f"channel={channel!r} is not a channel index (an integer such as 0) or 'composite'.",
            },
            indent=2,
        )
    if microns_per_pixel is None:
        notes.append(
            "Xenium obsm['spatial'] is in MICRONS, but microns_per_pixel=None asks for it to be "
            "treated as full-resolution image pixels. The scale factors will be off by the image's "
            "um/px (0.2125 for a full-resolution morphology OME-TIFF), so overlays will not line up."
        )
    try:
        sources = _resolve_morphology_sources(morphology_path)
    except FileNotFoundError as exc:
        return json.dumps({"status": "error", "message": str(exc)}, indent=2)

    img_raw, sources = _load_morphology_stack(sources, channel, notes)
    n_channels = img_raw.shape[2] if img_raw.ndim == 3 else 1
    fullres_hw = (int(img_raw.shape[0]), int(img_raw.shape[1]))
    hires_scale = min(hires_target_px / max(fullres_hw), 1.0)
    hires_hw = (int(fullres_hw[0] * hires_scale), int(fullres_hw[1] * hires_scale))

    # The channel is chosen and the stack shrunk in its own dtype before any float conversion
    # (hunt 2026-09-30, uT8-imaging-6); embed is told the full-resolution shape separately.
    if channel == "composite" and img_raw.ndim == 3:
        # 3+ channels: RGB composite from the first 3; fewer: pseudo-colour.
        img = _multichannel_to_rgb(_reduce_native(img_raw, hires_hw))
    elif isinstance(channel, int) and img_raw.ndim == 3:
        ch = min(channel, n_channels - 1)
        img = _grayscale_to_rgb(_reduce_native(img_raw[:, :, ch], hires_hw))
    else:
        img = _ensure_rgb(_reduce_native(img_raw, hires_hw))
    del img_raw

    report = embed_image_in_h5ad(
        h5ad_path=h5ad_path,
        image_path=_TEMP_MARKER,  # bypass, we already have the array
        output_path=output_path,
        library_id=library_id,
        hires_target_px=hires_target_px,
        lowres_target_px=lowres_target_px,
        microns_per_pixel=microns_per_pixel,
        _preloaded_image=img,
        _fullres_shape=fullres_hw,
        # A morphology image is fluorescence: bright tissue on a dark background.
        _metadata={"source": "xenium_morphology", "channel": str(channel)},
    )
    # The array was loaded here, so any loader caveat has to be carried into embed's report by hand.
    report = _merge_report_warnings(report, notes)
    payload = json.loads(report)
    payload["source_images"] = [str(s) for s in sources]
    return json.dumps(payload, indent=2)


def process_merfish_images(
    mosaic_path: str,
    h5ad_path: str,
    output_path: str | None = None,
    stain: str = "DAPI",
    library_id: str = "spatial_sample",
    hires_target_px: int = 2000,
    lowres_target_px: int = 600,
    microns_per_pixel: float | None = None,
) -> str:
    """Process MERFISH/Vizgen mosaic images and embed in h5ad.

    MERFISH platforms produce large mosaic images per stain channel (DAPI, PolyT, etc.).
    This loads the specified stain image, converts to RGB, and embeds in the h5ad.

    MERSCOPE obsm['spatial'] is in MICRONS while the mosaic is in pixels, so the two are linked by
    the ``micron_to_mosaic_pixel_transform.csv`` that ships beside the mosaic; it is read
    automatically. Mosaic pixel size varies by instrument and run, so when that file is missing this
    says so rather than assuming a constant.

    Args:
        mosaic_path: Path to the mosaic TIFF image.
        h5ad_path: Path to the h5ad file (must have obsm['spatial']).
        output_path: Output path for updated h5ad (default: a new file under the session's output
            directory, never the input).
        stain: Stain channel name (for metadata only).
        library_id: Library ID for adata.uns['spatial'].
        hires_target_px: Target max dimension for hires image.
        lowres_target_px: Target max dimension for lowres image.
        microns_per_pixel: Physical size of one mosaic pixel. None (default) reads it from
            ``micron_to_mosaic_pixel_transform.csv``.

    Returns:
        str: JSON report with processing details.

    """
    notes: list[str] = []
    if microns_per_pixel is None:
        microns_per_pixel, transform_notes = _vizgen_microns_per_pixel(mosaic_path)
        notes.extend(transform_notes)

    try:
        coord_scale = _coordinate_scale(microns_per_pixel)
    except ValueError as exc:
        return json.dumps({"status": "error", "message": str(exc)}, indent=2)

    adata = sc.read_h5ad(h5ad_path)
    if "spatial" not in adata.obsm:
        return json.dumps({"status": "error", "message": "h5ad has no obsm['spatial'] coordinates"})

    raw = _load_image(mosaic_path, warnings_out=notes, native=True)
    coords = np.asarray(adata.obsm["spatial"])
    fullres_h, fullres_w = raw.shape[:2]

    hires_img_scale = min(hires_target_px / max(fullres_h, fullres_w), 1.0)
    lowres_img_scale = min(lowres_target_px / max(fullres_h, fullres_w), 1.0)
    hires_hw = (int(fullres_h * hires_img_scale), int(fullres_w * hires_img_scale))
    lowres_hw = (int(fullres_h * lowres_img_scale), int(fullres_w * lowres_img_scale))

    # A mosaic is shrunk in its own dtype before it becomes float RGB (hunt 2026-09-30, uT8-imaging-6).
    img = _ensure_rgb(_reduce_native(raw, hires_hw))
    del raw

    hires_img = _resize_image(img, hires_img_scale, size=hires_hw)
    lowres_img = _resize_image(img, lowres_img_scale, size=lowres_hw)

    # The estimate is in coordinate units (microns here); this field is consumed as full-res pixels.
    spot_diameter = _estimate_spot_diameter(coords) * coord_scale

    scalefactors = {
        "tissue_hires_scalef": float(hires_img_scale * coord_scale),
        "tissue_lowres_scalef": float(lowres_img_scale * coord_scale),
        "spot_diameter_fullres": float(spot_diameter),
        "fiducial_diameter_fullres": float(spot_diameter * 1.2),
    }

    fit_warning = _coordinate_fit_warning(
        coords, scalefactors["tissue_hires_scalef"], hires_img.shape[:2], microns_per_pixel
    )
    if fit_warning:
        notes.append(fit_warning)

    _store_library(
        adata,
        library_id,
        {
            "images": {"hires": hires_img, "lowres": lowres_img},
            "scalefactors": scalefactors,
            "metadata": {"stain": stain, "source": "merfish_mosaic"},
        },
        notes,
    )

    out_path = _write_h5ad(adata, output_path or _default_output_path(h5ad_path, "image"), notes)

    return json.dumps(
        {
            "status": "success",
            "output_path": out_path,
            "stain": stain,
            "fullres_shape": [fullres_h, fullres_w],
            "hires_shape": list(hires_img.shape[:2]),
            "lowres_shape": list(lowres_img.shape[:2]),
            "scalefactors": scalefactors,
            "microns_per_pixel": microns_per_pixel,
            "warnings": notes or None,
        },
        indent=2,
    )


def process_cosmx_fov_images(
    composite_dir: str,
    h5ad_path: str,
    output_path: str | None = None,
    library_id: str = "spatial_sample",
    hires_target_px: int = 2000,
    lowres_target_px: int = 600,
) -> str:
    """Stitch CosMx FOV composite images into a single mosaic and embed in h5ad.

    CosMx produces per-FOV composite images (CellComposite_Fnnn.tif). This
    stitches them based on spatial coordinates into a single mosaic image.

    Args:
        composite_dir: Directory containing CellComposite_F*.tif FOV images.
        h5ad_path: Path to the h5ad file (must have obsm['spatial']).
        output_path: Output path for updated h5ad (default: a new file under the session's output
            directory, never the input).
        library_id: Library ID for adata.uns['spatial'].
        hires_target_px: Target max dimension for hires mosaic.
        lowres_target_px: Target max dimension for lowres mosaic.

    Returns:
        str: JSON report with stitching and embedding details.

    """
    from PIL import Image

    comp_dir = Path(composite_dir)
    fov_files = sorted(comp_dir.glob("CellComposite_F*.tif"))
    if not fov_files:
        fov_files = sorted(comp_dir.glob("CellComposite_F*.jpg"))
    if not fov_files:
        fov_files = sorted(comp_dir.glob("CellComposite_F*.png"))

    if not fov_files:
        return json.dumps({"status": "error", "message": f"No CellComposite images found in {composite_dir}"})

    adata = sc.read_h5ad(h5ad_path)
    if "spatial" not in adata.obsm:
        return json.dumps({"status": "error", "message": "h5ad has no obsm['spatial'] coordinates"})

    coords = np.asarray(adata.obsm["spatial"])

    # Load first image to get FOV dimensions
    first_fov = np.array(Image.open(str(fov_files[0])))

    # Since we don't have exact FOV positions, use the hires image as a single composite
    # For production, FOV positions should come from the manifest
    # Fallback: use a single representative FOV image or the first composite
    img = _ensure_rgb(first_fov)
    if img.dtype == np.float32:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    # Embed the best available image
    img_float = img.astype(np.float32) / 255.0
    hires_scale = min(hires_target_px / max(img_float.shape[:2]), 1.0)
    lowres_scale = min(lowres_target_px / max(img_float.shape[:2]), 1.0)

    hires_img = _resize_image(img_float, hires_scale)
    lowres_img = _resize_image(img_float, lowres_scale)

    spot_diameter = _estimate_spot_diameter(coords)
    scalefactors = {
        "tissue_hires_scalef": float(hires_scale),
        "tissue_lowres_scalef": float(lowres_scale),
        "spot_diameter_fullres": float(spot_diameter),
        "fiducial_diameter_fullres": float(spot_diameter * 1.2),
    }

    # Honest single-FOV caveat: with multiple FOVs and no position manifest we embed only the first FOV
    # while obsm['spatial'] spans the whole slide, so these scalefactors do NOT map global coords onto
    # this image. Surface it instead of returning a bare "success".
    stitch_warning = None
    if len(fov_files) > 1:
        stitch_warning = (
            f"{len(fov_files)} FOV images present but only the first was embedded (true FOV stitching "
            "needs the FOV-position manifest, which is unavailable). The image + scalefactors cover ONE "
            "FOV while obsm['spatial'] spans the whole slide - coordinate/image overlay (tissue masks, "
            "image-based tools) will be unreliable; use the image for QC only."
        )

    # With exactly one FOV there was no stitch warning and no fit check at all, so a single FOV
    # (its own ~4,000 px frame) embedded against convert_cosmx's slide-global pixel coordinates
    # (CenterX_global_px, tens of thousands) put every cell off the image under a plain success.
    # The check embed and merfish already run settles it from the geometry (hunt 2026-09-30,
    # uT8-imaging-28).
    notes: list[str] = [stitch_warning] if stitch_warning else []
    fit_warning = _coordinate_fit_warning(coords, scalefactors["tissue_hires_scalef"], hires_img.shape[:2], None)
    if fit_warning:
        notes.append(
            fit_warning + " For CosMx this usually means obsm['spatial'] holds slide-global pixels "
            "(CenterX_global_px) while the image is one FOV in its own local frame."
        )

    _store_library(
        adata,
        library_id,
        {
            "images": {"hires": hires_img, "lowres": lowres_img},
            "scalefactors": scalefactors,
            "metadata": {"source": "cosmx_fov", "n_fovs": len(fov_files), "stitched": False},
        },
        notes,
    )

    out_path = _write_h5ad(adata, output_path or _default_output_path(h5ad_path, "image"), notes)

    return json.dumps(
        {
            "status": "success",
            "output_path": out_path,
            "n_fov_images": len(fov_files),
            "hires_shape": list(hires_img.shape[:2]),
            "lowres_shape": list(lowres_img.shape[:2]),
            "scalefactors": scalefactors,
            "warning": stitch_warning,
            "warnings": notes or None,
        },
        indent=2,
    )


def generate_tissue_mask(
    h5ad_path: str,
    output_path: str | None = None,
    method: str = "coords",
    threshold: float = 0.8,
) -> str:
    """Generate a tissue mask from spatial coordinates or embedded histology image.

    Useful for pre-filtering spots and for tools that need tissue boundaries.

    Args:
        h5ad_path: Path to h5ad file.
        output_path: Output path for h5ad with tissue mask added. Defaults to a new
            ``<input>_tissue_mask.h5ad`` under the session's output directory, never the input.
        method: 'coords' keeps an existing obs['in_tissue'] (Space Ranger's call) or, without one,
                marks every spot that has coordinates as in tissue -- coordinates alone cannot tell
                tissue from background, so it detects nothing. 'image' thresholds the embedded image
                with Otsu: tissue darker than the cut for H&E, brighter than it for a fluorescence
                image (Xenium/MERFISH/CosMx, from uns['spatial'][...]['metadata']['source']).
        threshold: For 'image' method, a MULTIPLIER on the Otsu cut, not an intensity. The cut is
            ``otsu * threshold``: below 1.0 keeps only the darker (more strongly stained) part of
            what Otsu calls tissue, above 1.0 admits paler tissue. Passing a pixel value such as 128
            makes every pixel tissue, so anything outside (0, 2] is refused as a units mistake. On a
            fluorescence image the cut is ``otsu / threshold``, so below 1.0 is still the stricter.

    Returns:
        str: JSON report with mask statistics.

    """
    import pandas as pd

    adata = sc.read_h5ad(h5ad_path)
    otsu_backend = None
    polarity = None
    warnings_out: list[str] = []

    if method == "coords":
        if "spatial" not in adata.obsm:
            return json.dumps({"status": "error", "message": "No spatial coordinates for coord-based masking"})
        # Advertised as a "coordinate hull", this set in_tissue = 1 for every spot -- over a real
        # Space Ranger call when the file had one, in the input file itself by default -- and
        # reported n_in_tissue == n_spots as success, so nothing was filtered while the agent
        # believed off-tissue spots had been (hunt 2026-09-30, uT8-imaging-4). A hull of the spots
        # contains every spot; coordinates alone cannot find tissue. An existing call is kept, and
        # the all-spots answer says what it is.
        if "in_tissue" in adata.obs:
            existing = pd.to_numeric(adata.obs["in_tissue"], errors="coerce").fillna(0)
            n_in_tissue = int((existing != 0).sum())
            warnings_out.append(
                f"obs['in_tissue'] already existed ({n_in_tissue} of {adata.n_obs} spots in tissue) and was "
                "kept: method='coords' cannot detect tissue, so it never overwrites a real call. Use "
                "method='image' to recompute it from the embedded image."
            )
        else:
            coords = np.asarray(adata.obsm["spatial"], dtype=float)
            has_coords = np.isfinite(coords[:, :2]).all(axis=1)
            adata.obs["in_tissue"] = has_coords.astype(int)
            n_in_tissue = int(has_coords.sum())
            warnings_out.append(
                f"method='coords' marked every spot with coordinates as in tissue ({n_in_tissue} of "
                f"{adata.n_obs}); it does not detect tissue, so no off-tissue spot was removed. Use "
                "method='image' with an embedded image to find the tissue."
            )

    elif method == "image":
        # ``threshold`` scales the Otsu value; it is not an intensity, though the name invites that
        # reading. ``gray < otsu * 128`` selects the whole slide and reports success, and a mask that
        # selects everything is indistinguishable from no mask -- while being used to filter spots.
        # Guessing that 128 meant 128/255 would be inventing intent, so the units are stated instead.
        try:
            multiplier = float(threshold)
        except (TypeError, ValueError):
            multiplier = float("nan")
        if not np.isfinite(multiplier) or not 0.0 < multiplier <= 2.0:
            return json.dumps(
                {
                    "status": "error",
                    "message": (
                        f"threshold={threshold!r} is out of range. It is a multiplier on the Otsu cut "
                        "(the cut is otsu * threshold), not a pixel intensity, so it must fall in "
                        "(0, 2]; 1.0 is plain Otsu and the default 0.8 is slightly stricter. A value "
                        "like 128 selects every pixel; a value <= 0 selects none."
                    ),
                },
                indent=2,
            )
        # Only mapping-valued entries are libraries: CELLxGENE's scalar ``is_single`` beside the
        # library made ``.get`` raise AttributeError whenever it sorted first (hunt 2026-09-30,
        # uT8-imaging-3). With several libraries (concatenated sections) every spot would be
        # mapped onto the first section's image, so none is guessed.
        libraries = _spatial_libraries(adata.uns.get("spatial"))
        if not libraries:
            return json.dumps({"status": "error", "message": "No embedded image for image-based masking"})
        if len(libraries) > 1:
            return json.dumps(
                {
                    "status": "error",
                    "message": (
                        f"uns['spatial'] holds {len(libraries)} image libraries ({', '.join(map(str, libraries))}). "
                        "Image masking maps every spot onto one image, which is only right for a single "
                        "section; split the object by library and mask each one."
                    ),
                },
                indent=2,
            )
        lib_id, library = next(iter(libraries.items()))
        images = library.get("images", {})
        img_key = "hires" if "hires" in images else next(iter(images), None)
        if img_key is None:
            return json.dumps({"status": "error", "message": "No image found in uns['spatial']"})

        img = images[img_key]
        if img.ndim == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img.copy()

        # Otsu-style tissue detection (tissue is darker than background in H&E).
        # scikit-image is not part of the agent env, and an unguarded import let ModuleNotFoundError
        # escape this function's JSON-report contract; fall back to the equivalent numpy Otsu.
        try:
            from skimage.filters import threshold_otsu

            otsu_backend = "skimage"
        except ImportError:
            threshold_otsu = _otsu_threshold
            otsu_backend = "numpy"

        thresh = threshold_otsu(gray)
        # ``gray < cut`` is the H&E rule -- dark tissue on a white slide. On a fluorescence image
        # (DAPI, a MERSCOPE mosaic, a CosMx composite) tissue is bright on black, and the same rule
        # labelled every on-tissue spot 0 and every background spot 1 under a plain success
        # (hunt 2026-09-30, uT8-imaging-1). The embedders record the platform; an image with no
        # such record is treated as H&E and a dark border is reported rather than guessed from.
        metadata = library.get("metadata")
        source = str(metadata.get("source", "")) if isinstance(metadata, Mapping) else ""
        if source in _FLUORESCENCE_SOURCES:
            polarity = "bright tissue on a dark background (fluorescence)"
            tissue_mask_image = gray > thresh / multiplier
        else:
            polarity = "dark tissue on a bright background (H&E)"
            tissue_mask_image = gray < thresh * multiplier
            border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
            if border.size and float(np.median(border)) < thresh:
                warnings_out.append(
                    "The image's border is darker than the Otsu cut, which is what a fluorescence image "
                    "(bright tissue on a dark background) looks like, but nothing in uns['spatial'] "
                    "says it is one, so the H&E rule (tissue darker than the cut) was applied. If this "
                    "is a fluorescence image the mask is inverted; embed it with process_xenium_images "
                    "/ process_merfish_images / process_cosmx_fov_images so the platform is recorded."
                )

        # Map image-space mask to spot coordinates
        sf = library.get("scalefactors", {})
        scale = sf.get(f"tissue_{img_key}_scalef", 1.0)
        coords = np.asarray(adata.obsm["spatial"])
        scaled_coords = (coords * scale).astype(int)
        height, width = tissue_mask_image.shape[0], tissue_mask_image.shape[1]
        # Clamping is a fair last resort for a spot or two at the margin, but it turns "these
        # coordinates do not belong to this image" into a confident per-spot label: every stray spot
        # lands on one edge pixel and inherits its class. A missing or mismatched scale factor sends
        # the whole grid to a corner, and the report then reads `n_in_tissue: 0, status: success` --
        # indistinguishable from a slide with no tissue on it. Count them before the clamp erases
        # the evidence. `_coordinate_fit_warning` flags the same geometry one function away, and
        # says in as many words that tissue masks are what it goes wrong in.
        off_image = int(
            np.count_nonzero(
                (scaled_coords[:, 0] < 0)
                | (scaled_coords[:, 0] >= width)
                | (scaled_coords[:, 1] < 0)
                | (scaled_coords[:, 1] >= height)
            )
        )
        if off_image:
            warnings_out.append(
                f"{off_image} of {adata.n_obs} spots fall outside the {width} x {height} px "
                f"'{img_key}' image (scale factor {scale:g}) and were clamped to its border, so their "
                "in_tissue value is the border pixel's, not their own. The coordinate units and the "
                "image disagree -- check uns['spatial'] scalefactors, or re-embed the image with the "
                "right microns_per_pixel."
            )
        scaled_coords[:, 0] = np.clip(scaled_coords[:, 0], 0, width - 1)
        scaled_coords[:, 1] = np.clip(scaled_coords[:, 1], 0, height - 1)

        in_tissue = tissue_mask_image[scaled_coords[:, 1], scaled_coords[:, 0]]
        adata.obs["in_tissue"] = in_tissue.astype(int)
        n_in_tissue = int(in_tissue.sum())
    else:
        return json.dumps({"status": "error", "message": f"Unknown method: {method}"})

    out_path = _write_h5ad(adata, output_path or _default_output_path(h5ad_path, "tissue_mask"), warnings_out)

    return json.dumps(
        {
            "status": "success",
            "output_path": out_path,
            "method": method,
            "n_spots_total": adata.n_obs,
            "n_in_tissue": n_in_tissue,
            "n_spots_off_image": off_image if method == "image" else 0,
            "otsu_backend": otsu_backend,
            "polarity": polarity,
            "warnings": warnings_out or None,
        },
        indent=2,
    )


def prepare_miso_image(
    h5ad_path: str,
    image_path: str,
    output_dir: str,
    pixel_size_raw: float = 0.254,
    pixel_size: float = 0.5,
    spot_diameter_microns: float = 55.0,
) -> str:
    """Write an H&E image as a TIFF for the MISO tool, plus a locs.csv for MISO's upstream scripts.

    The wired ``run_miso`` takes the h5ad and ``histology_image_path`` (this TIFF, or Space Ranger's
    tissue_hires_image.png directly) and reads spot coordinates from obsm['spatial'] itself; it
    takes no locs.csv. ``run_miso_args`` in the report are the arguments to pass it. The TIFF keeps
    the source file's name, which is how run_miso's ``image_resolution='auto'`` recognises a hires
    or lowres image.

    Args:
        h5ad_path: Path to the h5ad file with spatial coordinates.
        image_path: Path to the H&E histology image (.tif, .png, .jpg).
        output_dir: Directory to write the prepared MISO inputs.
        pixel_size_raw: Raw image microns per pixel (default 0.254 for Visium). Used only for
            ``miso_params.computed_radius_px``; do not forward it to run_miso, which derives the
            image's scale from the scalefactors when its own pixel_size_raw is left 0.
        pixel_size: Target microns per pixel for feature extraction.
        spot_diameter_microns: Physical diameter of spots in microns.

    Returns:
        str: JSON with prepared file paths and MISO parameters.

    """
    from PIL import Image

    adata = sc.read_h5ad(h5ad_path)
    # Out of contract with the wired run_miso in three ways (hunt 2026-09-30, uT8-imaging-27): a
    # missing obsm['spatial'] became all-zero locs under a success; the rename to histology.tif
    # removed the file-name cue run_miso's image_resolution='auto' reads; and the advertised
    # locs.csv is an input run_miso never takes. The report now names run_miso's real arguments.
    if "spatial" not in adata.obsm:
        return json.dumps(
            {
                "status": "error",
                "message": "h5ad has no obsm['spatial'] coordinates; MISO samples the image at each spot's "
                "coordinates, so there is nothing to prepare without them.",
            },
            indent=2,
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and save image as .tif (MISO expects TIFF). Go through the shared loader rather than a
    # bare Image.open: a full-resolution H&E exceeds PIL's decompression-bomb limit, and opening it
    # directly raised DecompressionBombError from here on files _load_image reads without complaint.
    notes: list[str] = []
    arr = _load_image(image_path, warnings_out=notes)
    img_h, img_w = arr.shape[:2]
    # ``<source name>_miso.tif``: keeps "hires"/"lowres" for run_miso's frame detection, and can
    # never be the input file itself even when output_dir is the image's own folder.
    tif_path = out_dir / f"{Path(image_path).name.split('.')[0]}_miso.tif"
    Image.fromarray(_to_uint8(arr)).save(str(tif_path), format="TIFF")

    # Build locs DataFrame matching MISO's expected format:
    # col1=in_tissue, col2=array_row, col3=array_col, col4=pixel_y, col5=pixel_x
    import pandas as pd

    in_tissue = adata.obs.get("in_tissue", pd.Series(np.ones(adata.n_obs, dtype=int), index=adata.obs_names))
    array_row = adata.obs.get("array_row", pd.Series(np.zeros(adata.n_obs, dtype=int), index=adata.obs_names))
    array_col = adata.obs.get("array_col", pd.Series(np.zeros(adata.n_obs, dtype=int), index=adata.obs_names))

    coords = np.asarray(adata.obsm["spatial"])
    pixel_x = coords[:, 0]
    pixel_y = coords[:, 1]

    locs = pd.DataFrame(
        {
            "in_tissue": in_tissue.values.astype(int),
            "array_row": array_row.values.astype(int),
            "array_col": array_col.values.astype(int),
            "pixel_y": pixel_y,
            "pixel_x": pixel_x,
        },
        index=adata.obs_names,
    )
    locs_path = out_dir / "locs.csv"
    locs.to_csv(str(locs_path))

    rad = spot_diameter_microns / (2.0 * pixel_size_raw)
    notes.append(
        "run_miso takes run_miso_args (the h5ad and this TIFF) and reads coordinates from obsm['spatial']; "
        "locs.csv is for MISO's upstream scripts only. Do not forward miso_params.pixel_size_raw to "
        "run_miso: its own pixel_size_raw=0 derives the image scale from the scalefactors, and a "
        "Visium 0.254 um/px forced onto a hires image puts every feature window about 10x off."
    )

    return json.dumps(
        {
            "status": "success",
            "histology_tif": str(tif_path),
            "locs_csv": str(locs_path),
            "image_size": [int(img_w), int(img_h)],  # (width, height), matching PIL's Image.size
            "run_miso_args": {"h5ad_path": str(h5ad_path), "histology_image_path": str(tif_path)},
            "warnings": notes or None,
            "miso_params": {
                "pixel_size_raw": pixel_size_raw,
                "pixel_size": pixel_size,
                "spot_diameter_microns": spot_diameter_microns,
                "computed_radius_px": float(rad),
            },
            "n_spots": adata.n_obs,
        },
        indent=2,
    )


def convert_fluorescence_to_pseudo_he(
    image_path: str,
    output_path: str,
    dapi_channel: int = 0,
    membrane_channel: int | None = 1,
    cyto_channel: int | None = None,
) -> str:
    """Convert multi-channel fluorescence image to pseudo-H&E for tools expecting RGB histology.

    Creates a synthetic H&E-like image from fluorescence channels. Maps:
    - DAPI (nuclear stain) → Hematoxylin (blue-purple)
    - Membrane/protein channel → Eosin (pink)
    - Optional cytoplasm channel blended into eosin

    This allows fluorescence-based platforms (Xenium, MERFISH, CosMx) to use
    tools designed for H&E input (stLearn, MISO).

    Args:
        image_path: Path to multi-channel fluorescence image.
        output_path: Path to save the pseudo-H&E RGB image.
        dapi_channel: Channel index for DAPI/nuclear stain.
        membrane_channel: Channel index for membrane/protein marker. None to skip.
        cyto_channel: Channel index for cytoplasm marker. None to skip.

    Returns:
        str: JSON report with conversion details.

    """
    from PIL import Image

    notes: list[str] = []
    img = _load_image(image_path, warnings_out=notes)
    cyto = None
    if img.ndim == 2:
        # Single channel, use as DAPI
        dapi = img.astype(np.float32)
        memb = np.zeros_like(dapi)
        if cyto_channel is not None:
            notes.append(f"cyto_channel={cyto_channel} was ignored: {Path(image_path).name} is single-channel.")
    elif img.ndim == 3:
        n_ch = img.shape[2]
        dapi = img[:, :, min(dapi_channel, n_ch - 1)].astype(np.float32)
        if membrane_channel is not None and membrane_channel < n_ch:
            memb = img[:, :, membrane_channel].astype(np.float32)
        else:
            memb = np.zeros_like(dapi)
        if cyto_channel is not None and cyto_channel < n_ch:
            cyto = img[:, :, cyto_channel].astype(np.float32)
        elif cyto_channel is not None:
            notes.append(f"cyto_channel={cyto_channel} was ignored: the image has only {n_ch} channels.")
    else:
        return json.dumps({"status": "error", "message": "Unexpected image dimensions"})

    # Normalize to [0, 1]
    if dapi.max() > 1:
        dapi = dapi / dapi.max()
    if memb.max() > 1:
        memb = memb / memb.max()

    # Create pseudo-H&E:
    # Background is white (1,1,1), Hematoxylin is blue-purple, Eosin is pink
    # H channel (absorbs from white): darker = more stain
    h_stain = dapi  # nuclear: hematoxylin
    if cyto is None:
        e_stain = memb  # membrane: eosin
    else:
        if cyto.max() > 1:
            cyto = cyto / cyto.max()
        # Eosin stains cytoplasm as well as membrane, so the two channels add into the one stain,
        # saturating at full absorption rather than exceeding it.
        e_stain = np.clip(memb + cyto, 0, 1)

    # Map to RGB via Beer-Lambert approximation
    # Hematoxylin color vector: [0.65, 0.70, 0.29] (blue-purple)
    # Eosin color vector: [0.07, 0.99, 0.11] (pink)
    rgb = np.ones((*dapi.shape, 3), dtype=np.float32)
    rgb[:, :, 0] -= h_stain * 0.65 + e_stain * 0.07
    rgb[:, :, 1] -= h_stain * 0.70 + e_stain * 0.99
    rgb[:, :, 2] -= h_stain * 0.29 + e_stain * 0.11

    rgb = np.clip(rgb, 0, 1)
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    Image.fromarray(rgb_uint8).save(output_path)

    return json.dumps(
        {
            "status": "success",
            "output_path": output_path,
            "image_shape": list(rgb_uint8.shape),
            "dapi_channel": dapi_channel,
            "membrane_channel": membrane_channel,
            "cyto_channel": cyto_channel if cyto is not None else None,
            "warnings": notes or None,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

_TEMP_MARKER = "__preloaded__"

#: ``uns['spatial'][lib]['metadata']['source']`` values whose images are fluorescence: bright tissue
#: on a dark background, the opposite of H&E. Written by the embedders below.
_FLUORESCENCE_SOURCES = frozenset({"xenium_morphology", "merfish_mosaic", "cosmx_fov"})


def _spatial_libraries(uns_spatial) -> dict:
    """The image libraries in ``uns['spatial']``: the entries whose value is a mapping.

    Scalar markers beside them are not libraries -- every CELLxGENE Visium h5ad stores
    ``{'<library>': {...}, 'is_single': True}`` -- and calling ``.get`` on the bool raised
    AttributeError in place of the JSON report (hunt 2026-09-30, uT8-imaging-3). The same rule as
    ``tools/miso_worker.py::_single_uns_library``.
    """
    if not isinstance(uns_spatial, Mapping):
        return {}
    return {name: entry for name, entry in uns_spatial.items() if isinstance(entry, Mapping)}


def _store_library(adata, library_id: str, entry: dict, notes: list[str]) -> None:
    """Make ``entry`` the one library in ``uns['spatial']``, saying which libraries it replaced.

    The embedders replace ``uns['spatial']`` wholesale (one image per object is what the readers
    downstream expect). That silently dropped a Space Ranger library's own images and scale factors
    from the output; the input file keeps them, and the report now says so.
    """
    replaced = [name for name in _spatial_libraries(adata.uns.get("spatial")) if name != library_id]
    if replaced:
        notes.append(
            f"uns['spatial'] held {', '.join(map(str, replaced))}; the output carries only '{library_id}' "
            "(the input file is unchanged)."
        )
    adata.uns["spatial"] = {library_id: entry}


def _default_output_path(h5ad_path: str, tag: str) -> str:
    """Where a tool writes when its caller names no output_path: the session's output root.

    Every tool here defaulted to ``output_path or h5ad_path`` -- rewriting the dataset the user
    pointed at, through a staged symlink into the data library when that is what it was, against
    the prompt's own "never write next to the input files" (hunt 2026-09-30, uT8-imaging-5).
    """
    from spatialomicsgym.paths import tool_output_root

    out_dir = os.path.abspath(tool_output_root("spatial_image_processor"))
    os.makedirs(out_dir, exist_ok=True)
    name = Path(h5ad_path).name
    stem = name[: -len(".h5ad")] if name.lower().endswith(".h5ad") else name
    return os.path.join(out_dir, f"{stem}_{tag}.h5ad")


def _write_h5ad(adata, out_path: str, notes: list[str]) -> str:
    """Write ``adata`` to ``out_path`` atomically and return the absolute path written.

    anndata opens the target with h5py mode 'w', which truncates first: a failure mid-write (disk
    full, the REPL's budget, a MemoryError) left the file destroyed, and the file was the input by
    default. It is written beside the target and moved into place. ``os.replace`` onto a symlink
    replaces the link and leaves what it pointed to alone, so a staged link into a data library is
    never written through (hunt 2026-09-30, uT8-imaging-5).
    """
    target = os.path.abspath(out_path)
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    if os.path.islink(target):
        notes.append(
            f"{target} was a symbolic link to {os.path.realpath(target)}; the link was replaced by the new "
            "file and the file it pointed to was left unchanged."
        )
    # Created by h5py itself, so the file gets the umask's mode (mkstemp's 0600 would make the
    # result unreadable to anyone but its writer).
    partial = os.path.join(parent, f".{os.path.basename(target)}.{os.getpid()}.partial")
    try:
        adata.write_h5ad(partial)
        os.replace(partial, target)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    return target


def _coordinate_scale(microns_per_pixel: float | None) -> float:
    """How many full-resolution image pixels one obsm['spatial'] unit is worth.

    ``None`` means the coordinates are already in full-resolution pixels (Visium), giving exactly
    1.0 so those scale factors are unchanged bit for bit.
    """
    if microns_per_pixel is None:
        return 1.0
    try:
        mpp = float(microns_per_pixel)
    except (TypeError, ValueError):
        raise ValueError(f"microns_per_pixel must be a positive number, got {microns_per_pixel!r}") from None
    if not np.isfinite(mpp) or mpp <= 0:
        raise ValueError(f"microns_per_pixel must be a positive finite number, got {microns_per_pixel!r}")
    return 1.0 / mpp


def _vizgen_microns_per_pixel(mosaic_path: str) -> tuple[float | None, list[str]]:
    """Read the mosaic pixel size out of MERSCOPE's ``micron_to_mosaic_pixel_transform.csv``.

    The file is a 3x3 affine written beside the mosaic images: ``pixel = micron * scale + offset``,
    so its diagonal is pixels per micron. Mosaic pixel size varies by instrument and run, which is
    why there is no constant to fall back on -- a missing transform is reported, not guessed around.

    Returns:
        (microns_per_pixel or None, notes) -- None means "unknown", not "one".

    """
    p = Path(mosaic_path)
    name = "micron_to_mosaic_pixel_transform.csv"
    candidates = [p.parent / name, p.parent.parent / name, p.parent.parent / "images" / name]
    notes: list[str] = []

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            numbers = [float(token) for token in candidate.read_text().replace(",", " ").split()]
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            notes.append(f"Could not read the mosaic transform {candidate}: {exc}")
            continue
        if len(numbers) < 9:
            notes.append(f"The mosaic transform {candidate} does not hold a 3x3 matrix; ignoring it.")
            continue

        matrix = np.asarray(numbers[:9], dtype=float).reshape(3, 3)
        scale_x, scale_y = float(matrix[0, 0]), float(matrix[1, 1])
        if not np.isfinite([scale_x, scale_y]).all() or scale_x <= 0 or scale_y <= 0:
            notes.append(f"The mosaic transform {candidate} has a non-positive scale; ignoring it.")
            continue

        if abs(scale_x - scale_y) > 0.01 * max(scale_x, scale_y):
            notes.append(
                f"The mosaic transform is anisotropic ({scale_x:.4f} vs {scale_y:.4f} px/um); a "
                "single scale factor cannot express that, so their mean is used."
            )
        offset_x, offset_y = float(matrix[0, 2]), float(matrix[1, 2])
        if max(abs(offset_x), abs(offset_y)) > 1.0:
            notes.append(
                f"The mosaic transform also translates by ({offset_x:.1f}, {offset_y:.1f}) px, which a "
                "multiplicative scale factor cannot represent. Spots will be offset by that much; "
                "apply the full affine to obsm['spatial'] beforehand if the overlay must be exact."
            )
        return 2.0 / (scale_x + scale_y), notes

    notes.append(
        f"MERSCOPE obsm['spatial'] is in MICRONS, but no {name} was found beside {p.name}. The "
        "coordinates are being treated as if they were already mosaic pixels, so the scale factors "
        "will not overlay spots on this image. Pass microns_per_pixel explicitly to fix it."
    )
    return None, notes


def _coordinate_fit_warning(
    coords: np.ndarray, hires_scalef: float, hires_shape: tuple[int, ...], microns_per_pixel: float | None
) -> str | None:
    """Flag spots that the scale factor sends outside the image they are supposed to index.

    Landing off the canvas is unambiguous evidence that the coordinate units and the image's pixel
    size disagree -- most often a downsampled morphology image paired with the full-resolution
    ``microns_per_pixel``. Partial coverage is legitimate (tissue rarely fills a slide) and is not
    flagged here.
    """
    if coords.size == 0 or coords.shape[1] < 2:
        return None
    height, width = int(hires_shape[0]), int(hires_shape[1])
    plotted = np.nanmax(np.asarray(coords, dtype=float)[:, :2] * hires_scalef, axis=0)
    if plotted[0] <= width * 1.02 and plotted[1] <= height * 1.02:
        return None
    return (
        f"Spots land at up to ({plotted[0]:.0f}, {plotted[1]:.0f}) px on a {width} x {height} px hires "
        f"image, i.e. outside it. The coordinate units and the image pixel size disagree "
        f"(microns_per_pixel={microns_per_pixel!r}); the image is probably not at the resolution that "
        "value describes. Overlays, tissue masks and image-based tools will be wrong until it matches."
    )


def _resolve_morphology_sources(morphology_path: str) -> list[Path]:
    """Resolve a Xenium morphology argument to the image file(s) that actually hold the pixels.

    Xenium Onboarding Analysis >= 2.0 replaced the single ``morphology_focus.ome.tif`` with a
    ``morphology_focus/`` directory holding one OME-TIFF per stain (``morphology_focus_0000.ome.tif``
    is DAPI). Opening that path as a file raises ``IsADirectoryError``, which the conversion pipeline
    catches and reports as "no image" -- so a dataset that ships a morphology image was analysed
    without one. Sorted order is the stain order 10x writes.
    """
    p = Path(morphology_path)
    if not p.exists():
        raise FileNotFoundError(f"Morphology image not found: {morphology_path}")
    if not p.is_dir():
        return [p]

    for pattern in ("*.ome.tif", "*.ome.tiff", "*.tif", "*.tiff", "*.png"):
        found = sorted(p.glob(pattern))
        if found:
            return found
    raise FileNotFoundError(f"No OME-TIFF/TIFF/PNG images inside the morphology directory {morphology_path}")


def _load_morphology_stack(sources: list[Path], channel: int | str, notes: list[str]) -> tuple[np.ndarray, list[Path]]:
    """Load one morphology image, or stack a per-stain XOA 2.0 directory into a channel axis.

    Each XOA 2.0 file holds a single stain, so ``channel='composite'`` has to combine *files*; with
    an integer ``channel`` the corresponding stain file is used on its own. Returns the array
    alongside the files it actually came from, so the report names what was read.
    """
    if len(sources) == 1:
        return _load_image(str(sources[0]), warnings_out=notes, native=True), sources

    if isinstance(channel, int):
        picked = sources[min(max(channel, 0), len(sources) - 1)]
        return _load_image(str(picked), warnings_out=notes, native=True), [picked]

    used = sources[:3]
    planes = [_load_image(str(s), warnings_out=notes, native=True) for s in used]
    planes = [p[:, :, 0] if p.ndim == 3 else p for p in planes]
    if len({p.shape for p in planes}) > 1:
        _record_warning(
            notes,
            f"The {len(used)} morphology stain images do not share a shape "
            f"({[p.shape for p in planes]}); only {used[0].name} was used.",
        )
        return planes[0], [used[0]]
    return np.stack(planes, axis=-1), used


def _merge_report_warnings(report: str, extra: list[str]) -> str:
    """Fold a caller's caveats into a JSON report built by another function in this module."""
    if not extra:
        return report
    try:
        payload = json.loads(report)
    except json.JSONDecodeError:
        return report
    existing = list(payload.get("warnings") or [])
    payload["warnings"] = existing + [m for m in extra if m not in existing]
    return json.dumps(payload, indent=2)


def _record_warning(warnings_out: list[str] | None, message: str) -> None:
    """Surface a loader-level caveat to the caller's JSON report and to the Python warning stream."""
    if warnings_out is not None:
        warnings_out.append(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _load_image(path: str, warnings_out: list[str] | None = None, native: bool = False) -> np.ndarray:
    """Load an image from various formats into a numpy array.

    Args:
        path: Path to the image file.
        warnings_out: Optional list; caveats about *how* the image had to be read are appended here
            so the calling tool can put them in its JSON report instead of losing them.
        native: Return the pixels in the file's own dtype. The default converts to float32 -- a
            full-size copy at 2x (uint16) to 4x (uint8) the source, which the embedders cannot
            afford on a full-resolution slide; they shrink the native array first.

    """
    p = Path(path)
    tifffile_missing = False

    # OME-TIFF / multi-page TIFF
    if p.suffix.lower() in (".tif", ".tiff") or str(p).endswith(".ome.tif"):
        try:
            import tifffile

            img = tifffile.imread(str(p))
            # Handle multi-page TIFF: take first page or max projection
            if img.ndim == 4:
                # (pages, channels, H, W) or (pages, H, W, channels)
                img = img[0]  # Take first page
            if img.ndim == 3 and img.shape[0] < img.shape[2]:
                # (channels, H, W) -> (H, W, channels)
                img = np.moveaxis(img, 0, -1)
            if native:
                return img
            return img.astype(np.float32) if img.max() > 1 else img
        except ImportError:
            # Not silently swallowable: the PIL fallback below returns only the FIRST page of a
            # multi-page TIFF, so one z-plane or one stain channel would stand in for the whole
            # stack. Defer the message until PIL tells us how many pages were really there.
            tifffile_missing = True

    # Standard image formats via PIL
    from PIL import Image

    # Full-resolution H&E routinely exceeds PIL's ~178 MP DecompressionBomb limit; these are trusted
    # local histology files, so lift the cap deliberately instead of raising DecompressionBombError.
    Image.MAX_IMAGE_PIXELS = None
    pil_img = Image.open(str(p))
    n_pages = int(getattr(pil_img, "n_frames", 1) or 1)
    arr = np.array(pil_img)

    if tifffile_missing and n_pages > 1:
        _record_warning(
            warnings_out,
            f"tifffile is not installed, so {p.name} was read with PIL: only page 1 of its "
            f"{n_pages} pages was loaded. For an OME-TIFF that means a single z-plane or a single "
            "stain channel is standing in for the whole stack. Install tifffile in this environment "
            "to read the full image.",
        )

    if native:
        return arr.astype(np.uint8) * 255 if arr.dtype == bool else arr
    return arr.astype(np.float32) / 255.0 if arr.max() > 1 else arr.astype(np.float32)


def _otsu_threshold(image: np.ndarray, nbins: int = 256) -> float:
    """Otsu's threshold in pure numpy: the histogram split that maximises between-class variance.

    A drop-in stand-in for ``skimage.filters.threshold_otsu``, which is not installed in the agent
    env. Same 256-bin histogram over [min, max] and same bin-centre return value, so masks computed
    here line up with masks computed where scikit-image *is* available.
    """
    values = np.asarray(image, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return lo  # a flat tile has no between-class variance to maximise

    counts, edges = np.histogram(values, bins=nbins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0

    weight_below = np.cumsum(counts)
    weight_above = np.cumsum(counts[::-1])[::-1]
    # Class means, guarding the empty-class ends of the sweep (their variance term is zero anyway).
    mean_below = np.cumsum(counts * centers) / np.maximum(weight_below, 1)
    mean_above = (np.cumsum((counts * centers)[::-1]) / np.maximum(weight_above[::-1], 1))[::-1]

    between_class_variance = weight_below[:-1] * weight_above[1:] * (mean_below[:-1] - mean_above[1:]) ** 2
    return float(centers[int(np.argmax(between_class_variance))])


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a loaded image array back to uint8 for writing, without rescaling what already fits.

    ``_load_image`` hands back float32 in [0,1] for 8-bit sources; ``x / 255 * 255`` is exact for
    every uint8 value, so a round trip through here does not re-quantise the pixels a tool will read.
    """
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    peak = float(arr.max()) if arr.size else 0.0
    if peak <= 1.0:
        arr = arr * 255.0
    elif peak > 255.0:  # e.g. a 16-bit source: compress to the 8-bit range rather than clipping it
        arr = arr / peak * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _reduce_native(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Box-average ``img`` down by a whole factor, in its own dtype, keeping it at least ``target_hw``.

    The embedders made the full-resolution array float32 RGB before shrinking it -- an astype, a
    normalised copy, a 3-channel stack and two full-size ``(img * 255).clip`` temporaries, about
    21x a uint16 source -- to produce a 2000 px image, so a 1-3 Gpx Xenium morphology image needed
    40-120 GB (hunt 2026-09-30, uT8-imaging-6). Averaging k x k blocks one band of k rows at a time
    holds one band beyond the source; the exact LANCZOS resize then runs on an image at most twice
    the target. Edge blocks are averaged over the pixels they have. Integer images come back in
    their dtype (rounded), so the conversions downstream see the value range they always saw.
    """
    h, w = int(img.shape[0]), int(img.shape[1])
    k = min(h // max(int(target_hw[0]), 1), w // max(int(target_hw[1]), 1))
    if k <= 1:
        return img
    starts = np.arange(0, w, k)
    widths = (np.minimum(starts + k, w) - starts).reshape(-1, *([1] * (img.ndim - 2)))
    out = np.empty((-(-h // k), len(starts), *img.shape[2:]), dtype=np.float32)
    for i, r0 in enumerate(range(0, h, k)):
        band = img[r0 : r0 + k]
        block_sums = np.add.reduceat(band.sum(axis=0, dtype=np.float64), starts, axis=0)
        out[i] = block_sums / (band.shape[0] * widths)
    if np.issubdtype(img.dtype, np.integer):
        return np.rint(out, out=out).astype(img.dtype)
    return out.astype(img.dtype, copy=False)


def _resize_image(img: np.ndarray, scale: float, size: tuple[int, int] | None = None) -> np.ndarray:
    """Resize image by a scale factor, or to an exact ``(height, width)`` when ``size`` is given."""
    h, w = img.shape[:2]
    new_h, new_w = (int(size[0]), int(size[1])) if size is not None else (int(h * scale), int(w * scale))
    if (new_h, new_w) == (h, w):
        return img.copy()
    from PIL import Image

    if img.dtype in (np.float32, np.float64):
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        img_uint8 = img

    pil_img = Image.fromarray(img_uint8)
    pil_resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    result = np.array(pil_resized).astype(np.float32) / 255.0
    return result


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Convert image to RGB float32 [0,1]."""
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.max() > 1.0:
        img = img.astype(np.float32) / img.max()

    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 2:
        # Two fluorescence channels (DAPI + a membrane stain) came back as (h, w, 2), which no
        # spatial plotter renders, under a success (hunt 2026-09-30, uT8-imaging-31). The same
        # pseudo-colour process_xenium_images uses: DAPI blue, the second channel green.
        return _multichannel_to_rgb(img)
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3]  # Drop alpha
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[:, :, :3]
    return img


def _grayscale_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert single-channel to RGB."""
    if img.max() > 1.0:
        img = img.astype(np.float32) / img.max()
    return np.stack([img, img, img], axis=-1).astype(np.float32)


def _multichannel_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert multi-channel fluorescence to RGB composite."""
    if img.ndim == 2:
        return _grayscale_to_rgb(img)

    n_ch = img.shape[2]
    img_f = img.astype(np.float32)
    for c in range(n_ch):
        ch_max = img_f[:, :, c].max()
        if ch_max > 0:
            img_f[:, :, c] /= ch_max

    if n_ch >= 3:
        # Map first 3 channels to RGB directly
        rgb = img_f[:, :, :3]
    elif n_ch == 2:
        # Ch0=blue (DAPI), Ch1=green
        rgb = np.zeros((*img_f.shape[:2], 3), dtype=np.float32)
        rgb[:, :, 0] = img_f[:, :, 1] * 0.3  # red from ch1
        rgb[:, :, 1] = img_f[:, :, 1]  # green from ch1
        rgb[:, :, 2] = img_f[:, :, 0]  # blue from DAPI
    else:
        rgb = _grayscale_to_rgb(img_f[:, :, 0])

    return np.clip(rgb, 0, 1)


def _estimate_spot_diameter(coords: np.ndarray) -> float:
    """Estimate spot/cell diameter from nearest-neighbor spacing.

    Returns ~0.55x the median center-to-center spacing (the Visium spot/pitch ratio: 55um spot over a
    100um grid) so ``spot_diameter_fullres`` sizes plotted spots correctly. Returning the raw spacing
    (as before) rendered spot circles ~1.8x too large / overlapping.

    The sample is drawn from a dedicated, fixed-seed generator. Drawing it from ``np.random`` (as
    before) advanced the *global* seed, so simply embedding an image shifted every stochastic step
    that followed and made the value itself drift between runs on any dataset above the sample size.
    """
    from scipy.spatial import KDTree

    if coords.shape[0] < 2:
        return 50.0  # fallback

    tree = KDTree(coords)
    # Get distance to nearest neighbor for a sample
    sample_size = min(500, coords.shape[0])
    sample_idx = np.random.default_rng(0).choice(coords.shape[0], sample_size, replace=False)
    dists, _ = tree.query(coords[sample_idx], k=2)
    median_nn_dist = float(np.median(dists[:, 1]))
    # Convert center-to-center spacing -> spot/cell diameter (Visium 55um spot / 100um pitch).
    return median_nn_dist * 0.55
