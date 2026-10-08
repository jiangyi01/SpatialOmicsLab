#!/usr/bin/env python

"""
miso_worker.py

Run MISO (Multi-modal Spatial Omics, Coleman et al.) on a spatial transcriptomics .h5ad file.

- Input:  h5ad (AnnData) with:
    * adata.X   = gene expression (spots x genes)
    * adata.var = gene names
    * adata.obsm['spatial'] = spot centres in full-resolution image pixels (Space Ranger's frame)

- Optional: an H&E image, turned into a second modality by miso.hist_features.get_features exactly as
  the MISO tutorial does it.

  MISO itself reads no coordinates. Its affinity graph is built from each modality's features, so
  spatial information reaches the model ONLY through image features sampled at each spot. With the
  RNA modality alone the clusters are expression clusters, and the payload says so.

  The image may be the full-resolution TIF or one of Space Ranger's downscaled PNGs
  (tissue_hires_image.png / tissue_lowres_image.png). obsm['spatial'] is in full-resolution pixels,
  so for a downscaled image the coordinates are multiplied by tissue_hires_scalef /
  tissue_lowres_scalef, and the image's microns-per-pixel comes from the scalefactors
  (microns_per_pixel, or spot_diameter_fullres read as the physical spot it measures: bin_size_um on
  Visium HD, otherwise the 55 um Visium spot) instead of the tutorial's constant. The caller's
  spot_diameter_microns sets only the feature window, never the image scale. Every spot's window must
  lie inside the image or the run stops: an empty window is a NaN feature row, not a feature.

- Counts: miso.utils.preprocess log1p's X as counts, so the matrix is chosen by
  worker_utils.choose_counts_matrix right after the in_tissue cut: X (negative or NaN values are
  refused, fractional ones run with a warning), or adata.raw.X with --use_raw_counts.

- Affinity: sparse=False (the MISO tutorial's setting, and the default) builds MISO's dense Gaussian
  affinity -- an N x N matrix per modality plus a dense N x N distance matrix on every training
  epoch. sparse=True builds MISO's own k-nearest-neighbour affinity (neighbors, upstream default
  100). The peak memory of the configuration asked for is estimated before anything is allocated,
  and a run that cannot fit stops with the numbers. Nothing is ever subsampled.

- Output:
    * <output_dir>/miso_clusters.csv          (obs + miso_cluster)
    * <output_dir>/adata_with_miso_clusters.h5ad
    * JSON printed to stdout with basic summary (paths, counts, etc.)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from miso import Miso
from miso.hist_features import get_features
from miso.utils import preprocess
from PIL import Image
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    build_cluster_analysis,
    choose_counts_matrix,
    describe_reduction,
    keep_in_tissue,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    resolve_compute,
    spatial_coords,
    unsupported_choice_msg,
)

# MISO's own preprocessing drops genes detected in fewer than this many spots. The threshold is
# hardcoded by the library -- see miso/utils.py:preprocess, `sc.pp.filter_genes(adata,min_cells=10)`
# -- and MISO exposes no way to change it, so we mirror it here only to disclose it. The measured
# supplied-vs-analysed gene counts below stay correct even if a future MISO changes this number.
MISO_MIN_CELLS_PER_GENE = 10

# The MISO tutorial's raw-image pixel size in microns per pixel. It describes the tutorial's own
# full-resolution TIF and no other image; it is used only when nothing about the supplied image says
# otherwise, and the payload then says the scale was not derived from the data.
MISO_TUTORIAL_PIXEL_SIZE_RAW = 65.0 / 255.54640512302527
# The resolution get_features extracts features at (the tutorial's `pixel_size`).
MISO_FEATURE_PIXEL_SIZE = 0.5
# Visium spot diameter (the tutorial's `rad = 55 / (2 * pixel_size_raw)`).
VISIUM_SPOT_DIAMETER_UM = 55.0
# miso/model.py: `if neighbors is None and self.sparse: neighbors=100`.
MISO_SPARSE_NEIGHBORS_DEFAULT = 100
# miso/model.py: every modality is embedded to 32 dimensions (`MLP(..., output_shape=32)`).
MISO_EMBEDDING_DIM = 32
# miso/hist_features.py: 192 cls + 384 sub channels, on a grid of 16 x 16-pixel cells.
HIPT_CHANNELS = 192 + 384
HIPT_CELL_PX = 16

IMAGE_RESOLUTIONS = ("auto", "fullres", "hires", "lowres")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5ad",
        required=True,
        help="Path to spatial transcriptomics .h5ad file (e.g. /workspace/spatial_demo_data/lung_scc_visium_raw.h5ad)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write MISO outputs (CSV + annotated h5ad).",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        required=True,
        help="Number of clusters (spatial domains) for Miso.cluster().",
    )
    parser.add_argument(
        "--histology_tif",
        default=None,
        help="Optional path to the matching H&E image (full-resolution TIF, or Space Ranger's "
        "tissue_hires_image.png / tissue_lowres_image.png). If provided, histology features are added "
        "as a second modality.",
    )
    parser.add_argument(
        "--image_resolution",
        default="auto",
        help="Which pixel frame the histology image is in: 'fullres' (obsm['spatial'] used as-is), "
        "'hires' / 'lowres' (coordinates multiplied by tissue_hires_scalef / tissue_lowres_scalef), or "
        "'auto' (hires/lowres when the file name or the size of uns['spatial'] images says so, else fullres).",
    )
    parser.add_argument(
        "--scalefactors_json",
        default=None,
        help="Space Ranger scalefactors_json.json. Default: adata.uns['spatial'][<library>]['scalefactors'], "
        "then a scalefactors_json.json beside the image.",
    )
    parser.add_argument(
        "--pixel_size_raw",
        type=float,
        default=None,
        help="Microns per pixel of the supplied image. Default: derived from the scalefactors "
        "(microns_per_pixel, or spot_diameter_fullres against the spot diameter); the MISO tutorial's "
        "value only when no scalefactors exist, and the payload then says so.",
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=MISO_FEATURE_PIXEL_SIZE,
        help="Microns per pixel that get_features rescales the image to before extracting features "
        "(the MISO tutorial's 0.5).",
    )
    parser.add_argument(
        "--spot_diameter_microns",
        type=float,
        default=None,
        help="Diameter in microns of the image window each spot's features are averaged over (the window "
        "radius only; the image scale comes from the scalefactors or --pixel_size_raw). Default: bin_size_um "
        "from the scalefactors (Visium HD), else the 55 um Visium spot.",
    )
    parser.add_argument(
        "--use_raw_counts",
        action="store_true",
        help="Run on the counts in adata.raw.X instead of X (for an h5ad whose X is log-normalised or scaled "
        "and whose counts sit in adata.raw).",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Use MISO's k-nearest-neighbour affinity (Miso(sparse=True)) instead of the dense N x N one.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=MISO_SPARSE_NEIGHBORS_DEFAULT,
        help="k of the sparse kNN affinity (used only with --sparse; MISO's own default is 100).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device: 'auto', 'cpu', 'gpu'/'cuda', or 'cuda:N' for a specific GPU.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    try:
        _main_inner(args)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("miso", str(e), task="spatial_domain_clustering")


# ----------------------------------------------------------------------------- memory


# The memory this process can still allocate is worker_utils.available_memory_bytes: the smaller of
# MemAvailable and the room left under the cgroup limit, with the cgroup's page cache counted as
# reclaimable. A reader of its own here took the cgroup LIMIT as the room, so in a container already
# holding most of its limit the refusal below passed and the run was OOM-killed with no payload. It is
# imported into this module's namespace, and check_miso_memory looks it up there at call time.


def miso_peak_bytes(n_spots, n_genes, itemsize, sparse, neighbors, image_hw=None, scaled_hw=None):
    """``(peak_bytes, phase, {phase: bytes})`` for what MISO holds in memory. Coarse and upper-leaning.

    * ``miso.utils.preprocess`` returns the spots x genes matrix dense (``adata.X.A``), and
      ``Miso.__init__`` keeps a float32 torch copy of every modality, standardises a copy for the
      affinity and gives ``PCA(128)`` another. That N x G block is intrinsic to MISO.
    * ``sparse=False``: ``calculate_affinity`` builds ``pairwise_distances`` and its Gaussian as float64
      N x N arrays; the model keeps that array and a float32 torch copy per modality, and every
      epoch's loss is ``torch.triu(torch.cdist(Y, Y)) * torch.triu(A)`` -- more N x N float32 tensors
      plus their gradients.
    * ``sparse=True``: an N x k kNN graph per modality, and per-edge embedding differences in training.
    * With an image: ``get_features`` rescales it to ``pixel_size`` as float, then holds 576 HIPT
      channels on a 16-pixel grid of the rescaled image.

    It exists to refuse a run that cannot fit, not to size one.
    """
    n = float(n_spots)
    cells = n * float(n_genes)
    n_modalities = 2 if scaled_hw else 1
    rna = cells * itemsize
    features = rna + cells * 4.0 + 2.0 * cells * itemsize
    if sparse:
        edges = n * float(neighbors)
        build = 64.0 * edges
        adjacency = n_modalities * 40.0 * edges
        train = 4.0 * edges * MISO_EMBEDDING_DIM * 4.0
    else:
        pairs = n * n
        build = 24.0 * pairs
        adjacency = n_modalities * 12.0 * pairs
        train = 24.0 * pairs
    phases = {
        "building the model": features + adjacency + build,
        "training": rna + cells * 4.0 + adjacency + train,
    }
    if scaled_hw:
        h, w = image_hw
        hs, ws = scaled_hw
        grid = (math.ceil(hs / 256.0) * 256.0 / HIPT_CELL_PX) * (math.ceil(ws / 256.0) * 256.0 / HIPT_CELL_PX)
        image = float(h) * w * 3 * (1 + 4 + 8) + float(hs) * ws * 3 * (8 + 1 + 1) + 3.0 * HIPT_CHANNELS * grid * 4
        phases["extracting histology features"] = rna + image
    phase = max(phases, key=phases.get)
    return phases[phase], phase, phases


def check_miso_memory(
    n_spots, n_genes, itemsize, sparse, neighbors, image_hw=None, scaled_hw=None, pixel_size=None, available=None
):
    """Refuse, naming the numbers, before MISO allocates what cannot fit. Never subsamples.

    Returns ``(need_bytes, available_bytes)``; ``available`` is None when this platform cannot say.
    """
    need, phase, _ = miso_peak_bytes(n_spots, n_genes, itemsize, sparse, neighbors, image_hw, scaled_hw)
    if available is None:
        available = available_memory_bytes()
    if available is None or need <= available:
        return need, available
    here = f"about {available / 1e9:.1f} GB is available here (MemAvailable, or the room left under the cgroup limit)"
    if not sparse:
        sparse_need, _, _ = miso_peak_bytes(n_spots, n_genes, itemsize, True, neighbors, image_hw, scaled_hw)
        if sparse_need <= available:
            raise MemoryError(
                f"MISO's dense affinity (sparse=False, the MISO tutorial's setting) holds one {n_spots} x {n_spots} "
                f"matrix per modality and a dense {n_spots} x {n_spots} distance matrix on every training epoch: "
                f"about {need / 1e9:.1f} GB at the peak ({phase}), and {here}. Pass sparse=True to build MISO's "
                f"own k-nearest-neighbour affinity instead (neighbors={neighbors}; about {sparse_need / 1e9:.1f} GB "
                "here). It is a different affinity from the dense one, and the payload records which one ran. "
                "This worker never subsamples spots."
            )
    if phase == "extracting histology features":
        why = (
            f"get_features rescales the image to {pixel_size} um/px ({scaled_hw[0]} x {scaled_hw[1]} px) as float "
            "and holds 576 HIPT feature channels over it"
        )
    else:
        why = (
            f"miso.utils.preprocess returns the {n_spots} x {n_genes} spots x genes matrix dense, and Miso() keeps a "
            "float32 copy of it plus a standardised and a centred copy"
        )
        if not sparse:
            why += f", beside its dense {n_spots} x {n_spots} affinity (sparse=True would not fit either)"
    raise MemoryError(
        f"MISO on {n_spots} spots x {n_genes} genes needs about {need / 1e9:.1f} GB at its peak ({phase}), and "
        f"{here}: {why}. That is how MISO works, and this worker never subsamples spots or genes. Run where "
        "that much memory is available."
    )


# ----------------------------------------------------------------------------- histology geometry


def _positive(value):
    """``float(value)`` when it is a finite positive number, else None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _single_uns_library(adata):
    """``(name, entry)`` of the one library in ``uns['spatial']``, or ``(None, None)``.

    A library is an entry whose value is a mapping (``images`` / ``scalefactors``). Scalar markers
    beside it are not libraries: every CELLxGENE Visium sample in the library stores
    ``{'<library>': {...}, 'is_single': True}``, and counting ``is_single`` as a second library hid
    that sample's scalefactors and image sizes -- a hires image was then refused for a missing
    ``tissue_hires_scalef``, and a full-resolution one sampled at the MISO tutorial's 0.2544 um/px
    instead of the 2.85 um/px its own ``spot_diameter_fullres`` gives (windows about 11x too large).
    With two or more library entries none is guessed.
    """
    spatial = adata.uns.get("spatial") if hasattr(adata, "uns") else None
    if not isinstance(spatial, dict):
        return None, None
    libraries = [(name, entry) for name, entry in spatial.items() if isinstance(entry, dict)]
    if len(libraries) != 1:
        return None, None
    return libraries[0]


def load_scalefactors(adata, scalefactors_json, image_path):
    """``(scalefactors, source)`` for the section: ``({}, "")`` when none can be found.

    Order: an explicit ``scalefactors_json`` (an unreadable one is an error, not a skip); the
    ``scalefactors`` of the single library in ``adata.uns['spatial']`` (scalar markers such as
    ``is_single`` are not libraries); a ``scalefactors_json.json`` beside the image -- the Space
    Ranger ``spatial/`` folder layout. With several libraries in ``uns['spatial']`` none is guessed.
    """
    if scalefactors_json:
        path = os.path.abspath(scalefactors_json)
        try:
            with open(path) as fh:
                found = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ValueError(f"scalefactors_json={scalefactors_json!r} could not be read as JSON: {exc}") from exc
        if not isinstance(found, dict):
            raise ValueError(f"scalefactors_json={scalefactors_json!r} holds a {type(found).__name__}, not an object.")
        return dict(found), path
    name, entry = _single_uns_library(adata)
    if entry is not None and isinstance(entry.get("scalefactors"), dict) and entry["scalefactors"]:
        return dict(entry["scalefactors"]), f"adata.uns['spatial']['{name}']['scalefactors']"
    beside = os.path.join(os.path.dirname(os.path.abspath(image_path)), "scalefactors_json.json")
    if os.path.isfile(beside):
        try:
            with open(beside) as fh:
                found = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{beside} could not be read as JSON: {exc}") from exc
        if isinstance(found, dict):
            return dict(found), beside
    return {}, ""


def _uns_image_shapes(adata):
    """``{'hires': (h, w), ...}`` for the images stored with the single ``uns['spatial']`` library."""
    _, entry = _single_uns_library(adata)
    images = entry.get("images") if entry is not None else None
    shapes = {}
    if isinstance(images, dict):
        for key, value in images.items():
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) >= 2:
                shapes[str(key)] = (int(shape[0]), int(shape[1]))
    return shapes


def resolve_image_frame(requested, image_path, image_hw, scalefactors, uns_image_shapes):
    """``(resolution, coord_scale, how)``: which pixel frame the supplied image is in.

    ``obsm['spatial']`` is in full-resolution pixels. A Space Ranger hires/lowres PNG is that image
    scaled by ``tissue_hires_scalef`` / ``tissue_lowres_scalef``, so its coordinates are the
    full-resolution ones times that factor. ``auto`` reads the frame from the file name
    (``tissue_hires_image.png``) or from an exact size match with an image stored in
    ``uns['spatial']``; otherwise it takes the image as full resolution. The window check that follows
    catches a full-resolution guess made about a downscaled image.
    """
    requested = (requested or "auto").strip().lower()
    if requested not in IMAGE_RESOLUTIONS:
        raise ValueError(unsupported_choice_msg("image_resolution", requested, IMAGE_RESOLUTIONS))
    resolution, how = requested, "requested"
    if requested == "auto":
        name = os.path.basename(str(image_path)).lower()
        matches = [r for r in ("hires", "lowres") if uns_image_shapes.get(r) == tuple(image_hw)]
        if "hires" in name:
            resolution, how = "hires", "auto: file name"
        elif "lowres" in name:
            resolution, how = "lowres", "auto: file name"
        elif len(matches) == 1:
            resolution, how = matches[0], f"auto: same size as uns['spatial'] images['{matches[0]}']"
        else:
            resolution, how = "fullres", "auto: no hires/lowres marker in the file name or the size"
    if resolution == "fullres":
        return "fullres", 1.0, how
    key = f"tissue_{resolution}_scalef"
    scale = _positive(scalefactors.get(key))
    if scale is None:
        raise ValueError(
            f"The histology image is read as the {resolution} image ({how}), and mapping obsm['spatial'] "
            f"(full-resolution pixels) into it needs '{key}', which no scalefactors supplied. Pass "
            "scalefactors_json (Space Ranger's spatial/scalefactors_json.json), or image_resolution='fullres' "
            "if this is the full-resolution image."
        )
    return resolution, scale, how


def physical_spot_diameter(scalefactors):
    """``(microns, from)``: the physical spot the section's ``spot_diameter_fullres`` measures.

    Space Ranger writes ``spot_diameter_fullres`` as the full-resolution pixel size of the capture area's
    own spot: ``bin_size_um`` on Visium HD, the 55 um spot on standard Visium. It is a property of the
    slide, not of the window a caller wants features averaged over.
    """
    bin_um = _positive(scalefactors.get("bin_size_um"))
    if bin_um is not None:
        return bin_um, "scalefactors bin_size_um"
    return VISIUM_SPOT_DIAMETER_UM, "Visium spot diameter (55 um)"


def resolve_pixel_geometry(scalefactors, coord_scale, pixel_size_raw=None, spot_diameter_microns=None):
    """``(pixel_size_raw, pixel_from, spot_um, spot_from)`` for the supplied image.

    Two different diameters, kept apart:

    * The image SCALE (microns per pixel of the image the caller supplied -- not the tutorial's TIF):
      an explicit ``pixel_size_raw``; else the scalefactors' ``microns_per_pixel`` (Visium HD); else the
      physical spot that ``spot_diameter_fullres`` measures (:func:`physical_spot_diameter`) over that
      many pixels -- each divided by the image's scale factor; else, with no scalefactors at all, the
      MISO tutorial's constant, which the caller is told was not derived from the data.
    * The feature WINDOW (``spot_um``): the caller's ``spot_diameter_microns``, else that same physical
      spot. It sets the window radius and nothing else.

    The scale used to be derived from the caller's window diameter: the radius
    ``spot_um / (2 * pixel_size_raw)`` then collapsed to ``spot_diameter_fullres * scale / 2`` whatever
    was asked, and a 110 um request magnified the image 2x instead of widening the window.
    """
    physical_um, physical_from = physical_spot_diameter(scalefactors)
    if spot_diameter_microns is not None:
        spot_um, spot_from = float(spot_diameter_microns), "caller"
    else:
        spot_um, spot_from = physical_um, physical_from
    if pixel_size_raw is not None:
        return float(pixel_size_raw), "caller", spot_um, spot_from
    per = f", divided by the image scale factor {coord_scale:g}" if coord_scale != 1.0 else ""
    mpp = _positive(scalefactors.get("microns_per_pixel"))
    if mpp is not None:
        return mpp / coord_scale, f"scalefactors microns_per_pixel {mpp:g}{per}", spot_um, spot_from
    diameter = _positive(scalefactors.get("spot_diameter_fullres"))
    if diameter is not None:
        pixel = (physical_um / diameter) / coord_scale
        return (
            pixel,
            f"{physical_um:g} um spot ({physical_from}) over spot_diameter_fullres {diameter:g} px{per}",
            spot_um,
            spot_from,
        )
    return MISO_TUTORIAL_PIXEL_SIZE_RAW, "MISO tutorial default (no scalefactors found)", spot_um, spot_from


def plan_histology(adata, image_path, requested_resolution, scalefactors_json, pixel_size_raw, pixel_size, spot_um):
    """Everything get_features needs, checked against the image before any feature is extracted.

    Returns a dict with the opened image, the ``locs`` frame get_features reads (col '4' = row pixel,
    col '5' = column pixel, in the supplied image's frame), the radius, and the provenance of every
    number. Raises when ``obsm['spatial']`` is absent or when any spot's window falls outside the image.
    """
    he_path = os.path.abspath(image_path)
    image = Image.open(he_path)
    width, height = image.size
    image_hw = (int(height), int(width))

    # Required, not defaulted: the old code put every spot at pixel (0, 0) when obsm['spatial'] was
    # missing, so all spots shared one image patch.
    coords, _ = spatial_coords(adata, "spatial", 2, "miso")

    scalefactors, sf_from = load_scalefactors(adata, scalefactors_json, he_path)
    resolution, coord_scale, how = resolve_image_frame(
        requested_resolution, he_path, image_hw, scalefactors, _uns_image_shapes(adata)
    )
    px_raw, px_from, spot_diameter, spot_from = resolve_pixel_geometry(
        scalefactors, coord_scale, pixel_size_raw, spot_um
    )
    rad = spot_diameter / (2.0 * px_raw)

    x = coords[:, 0] * coord_scale
    y = coords[:, 1] * coord_scale
    outside = ~(np.isfinite(x) & np.isfinite(y))
    outside |= (x - rad < 0) | (y - rad < 0) | (x + rad > width) | (y + rad > height)
    n_out = int(outside.sum())
    if n_out:
        raise ValueError(
            f"{n_out} of {len(x)} spots fall outside the histology image once obsm['spatial'] is mapped into it. "
            f"The image is {width} x {height} px, read as the {resolution} image ({how}; coordinates x "
            f"{coord_scale:g}), and the spot windows (radius {rad:.1f} px) span x {np.nanmin(x - rad):.0f}.."
            f"{np.nanmax(x + rad):.0f}, y {np.nanmin(y - rad):.0f}..{np.nanmax(y + rad):.0f}. get_features "
            "would average image features over empty windows and MISO would cluster NaN. If the image is "
            "Space Ranger's tissue_hires_image.png or tissue_lowres_image.png, pass image_resolution='hires' or "
            "'lowres' with its scalefactors (scalefactors_json, or uns['spatial'] in the h5ad); otherwise supply "
            "the full-resolution image of this section."
        )

    locs = pd.DataFrame(
        {
            "1": np.ones(len(x), dtype=int),
            "2": _obs_int(adata, "array_row"),
            "3": _obs_int(adata, "array_col"),
            "4": y,  # row pixel coordinate in the supplied image
            "5": x,  # column pixel coordinate in the supplied image
        }
    )
    scale = px_raw / float(pixel_size)
    scaled_hw = (int(round(height * scale)), int(round(width * scale)))
    notes = []
    if px_from.startswith("MISO tutorial default"):
        notes.append(
            f"The histology image's scale was not derived from the data: no scalefactors were found (uns['spatial'], "
            f"scalefactors_json, or a scalefactors_json.json beside the image), so the MISO tutorial's "
            f"{MISO_TUTORIAL_PIXEL_SIZE_RAW:.4f} um/px -- a property of the tutorial's own image -- was used. Pass "
            "pixel_size_raw or scalefactors_json for this image's real scale."
        )
    if scale > 1.01:
        notes.append(
            f"The histology image is {px_raw:.3f} um/px, coarser than the {pixel_size:g} um/px MISO extracts features "
            f"at, so get_features upsampled it {scale:.2f}x ({height} x {width} -> {scaled_hw[0]} x {scaled_hw[1]} px): "
            "the image features were computed on interpolated pixels, not on detail the image contains."
        )
    return {
        "image": image,
        "path": he_path,
        "image_hw": image_hw,
        "scaled_hw": scaled_hw,
        "locs": locs,
        "rad": float(rad),
        "pixel_size_raw": float(px_raw),
        "notes": notes,
        "record": {
            "image": he_path,
            "image_width_px": int(width),
            "image_height_px": int(height),
            "image_resolution": resolution,
            "image_resolution_from": how,
            "coordinate_scale": float(coord_scale),
            "scalefactors_from": sf_from or None,
            "pixel_size_raw_um": float(px_raw),
            "pixel_size_raw_from": px_from,
            "spot_diameter_um": float(spot_diameter),
            "spot_diameter_from": spot_from,
            "spot_radius_px": float(rad),
            "feature_pixel_size_um": float(pixel_size),
            "rescale_factor": float(scale),
        },
    }


def _obs_int(adata, key):
    if key in adata.obs.columns:
        return pd.to_numeric(adata.obs[key], errors="coerce").fillna(0).astype(int).to_numpy()
    return np.zeros(adata.n_obs, dtype=int)


def _atomic_write(final_path, write):
    """``write(<final>.partial)``, then ``os.replace`` onto the final name; no partial file survives."""
    final_path = str(final_path)
    partial = final_path + ".partial"
    try:
        write(partial)
        os.replace(partial, final_path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def _validate_args(args):
    if args.neighbors < 1:
        raise ValueError(f"neighbors={args.neighbors}: the kNN affinity needs at least one neighbour.")
    for name in ("pixel_size_raw", "spot_diameter_microns"):
        value = getattr(args, name)
        if value is not None and _positive(value) is None:
            raise ValueError(f"{name}={value!r} must be a positive number of microns (omit it to derive it).")
    if _positive(args.pixel_size) is None:
        raise ValueError(f"pixel_size={args.pixel_size!r} must be a positive number of microns per pixel.")
    if (args.image_resolution or "auto").strip().lower() not in IMAGE_RESOLUTIONS:
        raise ValueError(unsupported_choice_msg("image_resolution", args.image_resolution, IMAGE_RESOLUTIONS))


def _main_inner(args):
    _validate_args(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- 0. seed & device (as the tutorial does) -----------------
    seed = 100
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # The caller's request, not the hardware's answer. Probing `torch.cuda.is_available()` here
    # answered "is there a GPU", which is a different question from "was one asked for": on a box
    # with a GPU there was no way -- CLI or portal -- to keep MISO off it, or to send it to any card
    # other than the default one.
    device = resolve_compute(args.device).device
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[miso-worker] Using device: {device}", file=sys.stderr)

    # ----------------- 1. read the h5ad -----------------
    h5ad_path = os.path.abspath(args.h5ad)
    adata = sc.read_h5ad(h5ad_path)

    # Both cuts below shrink `adata` itself: the in_tissue filter rebinds it to a subset, and MISO's
    # own preprocess() mutates it in place. After this point adata.n_obs/n_vars mean the panel the
    # method analysed, not the one the caller handed us -- so record the supplied counts here, where
    # they are still unambiguous.
    n_spots_supplied = int(adata.n_obs)

    # Keep only in-tissue spots (when the column exists), by the fleet's one rule: 1/True/"1"/"true"
    # count as tissue, and a column that marks no spot as tissue is refused. `.astype(int) == 1`
    # raised on a "True"/"False" string column and on a missing flag, and a column of zeros left an
    # empty object for MISO to fail on far from the cause.
    adata, _, n_off_tissue = keep_in_tissue(adata)

    # preprocess() log1p's X as counts. A scaled X (negative values) or NaN is refused, naming
    # use_raw_counts when adata.raw holds counts; use_raw_counts runs on adata.raw.X; a fractional
    # (already log-normalised) X runs as before, with a warning.
    use_raw_counts = bool(getattr(args, "use_raw_counts", False))
    adata, counts_choice = choose_counts_matrix(adata, use_raw_counts)
    # The panel of the matrix MISO is handed: adata.raw can carry more genes than X.
    n_genes_supplied = int(adata.n_vars)

    sparse = bool(args.sparse)
    neighbors = int(args.neighbors)
    if sparse and neighbors >= adata.n_obs:
        raise ValueError(
            f"neighbors={neighbors} but only {adata.n_obs} spots are analysed; the kNN affinity needs fewer "
            "neighbours than spots."
        )

    # ----------------- 2. the histology plan, checked before anything heavy runs -----------------
    plan = None
    if args.histology_tif is not None:
        plan = plan_histology(
            adata,
            args.histology_tif,
            args.image_resolution,
            args.scalefactors_json,
            args.pixel_size_raw,
            args.pixel_size,
            args.spot_diameter_microns,
        )

    # ----------------- 3. memory, before MISO densifies anything -----------------
    # preprocess() drops genes seen in < MISO_MIN_CELLS_PER_GENE spots and then densifies; count the
    # survivors without mutating adata so the estimate is about the matrix MISO will really hold.
    genes_kept = sc.pp.filter_genes(adata, min_cells=MISO_MIN_CELLS_PER_GENE, inplace=False)[0]
    n_genes_estimate = int(np.asarray(genes_kept).sum())
    itemsize = 4 if np.dtype(adata.X.dtype) == np.float32 else 8
    need, available = check_miso_memory(
        int(adata.n_obs),
        n_genes_estimate,
        itemsize,
        sparse,
        neighbors,
        plan["image_hw"] if plan else None,
        plan["scaled_hw"] if plan else None,
        args.pixel_size,
    )

    # ----------------- 4. RNA modality via preprocess, as the tutorial does -----------------
    # preprocess() mutates the object we pass IN PLACE (filter_genes then log1p) and returns the
    # matrix, so this call is also where the gene cut lands on `adata`.
    with contextlib.redirect_stdout(sys.stderr):
        rna = preprocess(adata, modality="rna")

    modalities = [rna]
    used_histology = False

    # ----------------- 5. optional H&E image -> image modality -----------------
    if plan is not None:
        # get_features prints its progress ("Scaling image", "Smoothing embeddings", ...) to stdout, which
        # is the payload's channel; it goes to stderr with the rest of the log.
        with contextlib.redirect_stdout(sys.stderr):
            image_emb = get_features(
                plan["image"],
                plan["locs"],
                plan["rad"],
                plan["pixel_size_raw"],
                args.pixel_size,
                pretrained=True,
                device=device,
            )
        image_emb = np.asarray(image_emb)
        if image_emb.shape[0] != adata.n_obs:
            raise RuntimeError(f"get_features returned {image_emb.shape[0]} rows for {adata.n_obs} spots.")
        bad = ~np.isfinite(image_emb.reshape(image_emb.shape[0], -1)).all(axis=1)
        if bad.any():
            raise ValueError(
                f"get_features returned non-finite image features for {int(bad.sum())} of {adata.n_obs} spots "
                "(empty windows); MISO would cluster NaN. Check that the image is this section's and that "
                "image_resolution / scalefactors_json / pixel_size_raw describe it."
            )
        modalities.append(image_emb)
        used_histology = True

    # ----------------- 6. train MISO and cluster -----------------
    # tutorial: model = Miso([rna, protein, image_emb], ind_views='all', combs='all', sparse=False, device=device)
    # When only one modality is available, there are no pairwise combinations,
    # so we must pass combs=None to avoid an empty-concatenate error in Miso.train().
    combs = "all" if len(modalities) >= 2 else None
    with contextlib.redirect_stdout(sys.stderr):
        model = Miso(
            modalities,
            ind_views="all",
            combs=combs,
            sparse=sparse,
            neighbors=neighbors if sparse else None,
            device=device,
        )
        model.train()

        clusters = model.cluster(n_clusters=args.n_clusters)
    clusters = np.asarray(clusters).reshape(-1).astype(int)

    if clusters.shape[0] != adata.n_obs:
        raise RuntimeError(f"MISO returned {clusters.shape[0]} clusters, but AnnData has {adata.n_obs} spots.")

    # ----------------- 7. write the h5ad + CSV -----------------
    adata.obs["miso_cluster"] = clusters.astype(str)

    # CSV: obs + (when present) spatial coordinates
    out_df = adata.obs.copy()
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        out_df["spatial_x"] = coords[:, 0]
        out_df["spatial_y"] = coords[:, 1]

    clusters_csv = out_dir / "miso_clusters.csv"
    _atomic_write(clusters_csv, out_df.to_csv)

    annotated_h5ad = out_dir / "adata_with_miso_clusters.h5ad"
    _atomic_write(annotated_h5ad, adata.write)

    unique, counts = np.unique(clusters, return_counts=True)
    cluster_counts = {int(k): int(v) for k, v in zip(unique, counts)}

    # ----------------- 8. JSON payload on stdout -----------------
    # House polarity: the bare key is what the caller supplied, the `_used` key is what MISO actually
    # saw. Both counts below are post-cut, which is why they are published under the `_used` names --
    # reading them off `adata` here reported the survivors as if they were the input.
    n_spots_used = int(adata.n_obs)
    n_genes_used = int(adata.n_vars)

    spot_note = describe_reduction(
        "spots", n_spots_supplied, n_spots_used, "the in_tissue flag, which marks spots off the tissue"
    )
    gene_note = describe_reduction(
        "genes",
        n_genes_supplied,
        n_genes_used,
        f"MISO's own preprocessing, which drops genes detected in fewer than {MISO_MIN_CELLS_PER_GENE} spots",
    )

    if sparse:
        affinity = f"sparse k-nearest-neighbour Gaussian affinity (k={neighbors})"
    else:
        affinity = "dense Gaussian affinity over every pair of spots"

    out = WorkerOutput("miso", task="spatial_domain_clustering")
    out.set_data(
        n_spots=n_spots_supplied,
        n_genes=n_genes_supplied,
        n_spots_used=n_spots_used,
        n_genes_used=n_genes_used,
    )
    out.add_output_files(
        {
            "annotated_h5ad": str(annotated_h5ad),
            "clusters_csv": str(clusters_csv),
        }
    )
    record_method(out, "MISO (RNA + histology image features)" if used_histology else "MISO (RNA modality only)")
    out.add_params(
        {
            "used_histology": bool(used_histology),
            "input_h5ad": h5ad_path,
            # Fixed by MISO, not by us -- reported so a caller can see which threshold produced the
            # gene count above without reading the library source.
            "min_cells_per_gene": MISO_MIN_CELLS_PER_GENE,
            # Reported, so a caller can tell after the fact which device actually ran -- and see a
            # GPU request that degraded to CPU rather than assuming it was honoured.
            "device": device,
            # Which of MISO's two affinities ran; the dense one is N x N per modality.
            "sparse": sparse,
            "neighbors": neighbors if sparse else None,
            "affinity": affinity,
            # MISO reads no coordinates: without an image nothing spatial enters the model.
            "uses_spatial_coordinates": bool(used_histology),
            "histology": plan["record"] if plan else None,
            "memory_estimate_gb": round(need / 1e9, 3),
            "memory_available_gb": round(available / 1e9, 3) if available else None,
            "use_raw_counts": use_raw_counts,
        }
    )
    record_expression_source(out, counts_choice)
    if not sparse and neighbors != MISO_SPARSE_NEIGHBORS_DEFAULT:
        record_ignored(
            out,
            "neighbors",
            "only MISO's sparse kNN affinity (sparse=True) has neighbours; the dense affinity weighs every pair",
        )
    if plan is None:
        image_knobs = []
        if (args.image_resolution or "auto").strip().lower() != "auto":
            image_knobs.append("image_resolution")
        for name in ("scalefactors_json", "pixel_size_raw", "spot_diameter_microns"):
            if getattr(args, name):
                image_knobs.append(name)
        if args.pixel_size != MISO_FEATURE_PIXEL_SIZE:
            image_knobs.append("pixel_size")
        if image_knobs:
            record_ignored(out, image_knobs, "no histology image was supplied; these describe the image")
    # The number of clusters the labels hold, beside the number asked for: KMeans returns fewer than
    # asked when the embedding has fewer distinct points than k.
    n_clusters_found = int(len(cluster_counts))
    out.set_summary(
        n_clusters=n_clusters_found,
        n_clusters_requested=int(args.n_clusters),
        cluster_key="miso_cluster",
        cluster_sizes=cluster_counts,
    )
    if n_clusters_found != int(args.n_clusters):
        out.add_warning(
            f"n_clusters={int(args.n_clusters)} was requested, but MISO's KMeans returned {n_clusters_found} "
            "distinct clusters (the embedding has fewer distinct points than the clusters asked for)."
        )
    # A successful run never carries stderr into the payload (base_mcp attaches stderr_tail only on a
    # non-zero exit), so a cut this large has to travel in the payload itself. The spot cut is only
    # ever the in_tissue flag, and record_in_tissue carries it (params.in_tissue_filter + a warning).
    record_in_tissue(out, n_spots_supplied, n_off_tissue)
    if gene_note:
        out.add_warning(gene_note.strip())
    if plan is not None and plan["notes"]:
        out.add_warnings(plan["notes"])

    if used_histology:
        record = plan["record"]
        method_note = (
            f" MISO combined the RNA modality with histology features from {os.path.basename(record['image'])}, "
            f"read as the {record['image_resolution']} image (coordinates x {record['coordinate_scale']:g}, "
            f"{record['pixel_size_raw_um']:.3f} um/px from {record['pixel_size_raw_from']}); affinity: {affinity}."
        )
    else:
        method_note = (
            f" MISO was given the RNA modality only and reads no spot coordinates -- its {affinity} is built from "
            "expression -- so these are expression clusters; spatial information enters MISO only through a "
            "histology image (histology_image_path)."
        )
    out.set_analysis(
        build_cluster_analysis(
            cluster_counts,
            cluster_key="domain" if used_histology else "cluster",
            total_spots=n_spots_used,
            n_requested=int(args.n_clusters),
        )
        + spot_note
        + gene_note
        + method_note
    )
    out.emit()


if __name__ == "__main__":
    main()
