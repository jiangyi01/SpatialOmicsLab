#!/usr/bin/env python3
"""
DeepCell worker: cell segmentation via Mesmer (or NuclearSegmentation) for microscopy images.

- Runs inside /opt/conda/envs/deepcell_env
- All logs/progress go to stderr.
- stdout is reserved for a single final JSON line.

Uses the DeepCell 0.12.10 applications:

* ``model_type='mesmer'`` -- Mesmer (MultiplexSegmentation-9): whole-cell and/or nuclear
  segmentation of a two-marker (nuclear + membrane) fluorescence image.
* ``model_type='nuclear'`` -- NuclearSegmentation version 1.0 (NuclearSegmentation-75): nuclear
  segmentation of one nuclear-marker channel.

NOTE: deepcell >= 0.12.7 fetches weights through users.deepcell.org, which needs a
DEEPCELL_ACCESS_TOKEN. Both archives above are still served from the original public S3 bucket
(https://deepcell-data.s3-us-west-1.amazonaws.com/saved-models/<archive>), and their md5 is the one
deepcell itself pins for them. This worker caches them under ~/.deepcell/models and loads the
extracted SavedModel directly, so no token is ever needed -- including when the tarball has been
deleted and only the extracted weights remain.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    default_output_dir,
    preflight_check,
    record_ignored,
    record_method,
    unsupported_choice_msg,
)

#: The image_mpp the portal and the CLI default to. It is Mesmer's own ``model_mpp`` -- the one
#: resolution at which Mesmer leaves the image unscaled -- not a measurement of any image. A run
#: at this value is reported as having *assumed* it (``params.image_mpp_source``) unless the caller
#: says it is the image's measured pixel size (``image_mpp_is_measured``): the value alone cannot
#: tell an explicit 0.5 from the default.
DEFAULT_IMAGE_MPP = 0.5

#: The diameter, in um, that Space Ranger's ``spot_diameter_fullres`` spans: "the number of pixels
#: that span the diameter of a theoretical 65 um spot in the original, full-resolution image".
#: Checked on the library's Visium_FFPE_Human_Prostate_IF: neighbouring spots (100 um centre to
#: centre) are 290 full-resolution pixels apart, 0.3448 um/px, and 65 / spot_diameter_fullres gives
#: 0.3447 (55 / it gave 0.2917, 15% too fine). It does NOT hold on CytAssist slides: measured
#: against the spot pitch, 65 / spot_diameter_fullres is ~7.7% too coarse on the library's 11 mm
#: CytAssist samples (Lung Cancer 0.2965 vs 0.2753 um/px) and ~7% too fine on the 6.5 mm ones. So it
#: is only the last resort, used when no ``tissue_positions`` file sits beside the scale factors to
#: measure the pitch from (``VISIUM_SPOT_PITCH_UM``). commot_worker uses the same constant.
VISIUM_SPOT_DIAMETER_FULLRES_UM = 65.0

#: Centre-to-centre distance, in um, between neighbouring Visium spots -- the array's physical
#: pitch on every Visium slide (standard, CytAssist 6.5 mm and 11 mm alike). The median
#: nearest-neighbour distance of ``pxl_row/col_in_fullres`` in ``tissue_positions(.csv|_list.csv)``
#: is this many um, which measures the full-resolution pixel size directly.
VISIUM_SPOT_PITCH_UM = 100.0

#: Written inside an extracted SavedModel directory once this worker has extracted it completely
#: (into a temporary sibling, renamed into place) or checked it file by file against the archive.
EXTRACTED_MARKER = ".extraction_complete.json"

#: A ``.<SavedModel>.extracting-*`` staging directory whose newest entry is older than this was
#: left by a run killed mid-extraction (a SIGKILL or timeout kill skips the ``finally`` that removes
#: it) and is deleted; a younger one may be another run's extraction in progress and is left alone.
#: A real extraction takes about a minute.
STALE_STAGING_SECONDS = 24 * 3600

_S3 = "https://deepcell-data.s3-us-west-1.amazonaws.com/saved-models/"

#: What each model_type loads. ``mem_*`` is the run's measured peak resident memory -- TensorFlow
#: and the weights, plus bytes per pixel of the model-scale input -- used only to refuse, before it
#: starts, a run that cannot fit in the memory available. ``md5`` is the hash deepcell pins for the
#: same archive (``mesmer.MODEL_HASH``; ``nuclear_segmentation.CONFIGS['1.0'].model_hash``). ``extract_into`` is
#: relative to ~/.deepcell/models: Mesmer's goes where ``Mesmer()`` itself extracts it; the nuclear
#: model gets its own directory because deepcell's 1.1 archive extracts to the same
#: ``NuclearSegmentation/`` name, and a token user's 1.1 weights must not be read as 1.0.
MODEL_SPECS = {
    "mesmer": {
        "archive": "MultiplexSegmentation-9.tar.gz",
        "md5": "a1dfbce2594f927b9112f23a0a1739e0",
        "extract_into": "",
        "saved_model": "MultiplexSegmentation",
        "method": "DeepCell Mesmer (MultiplexSegmentation-9)",
        "model_mpp": 0.5,
        # Peak RSS measured on CPU in deepcell_env (2026-09-29): 1.71 / 3.15 GiB at 3.77 / 15.1 Mpx
        # model-scale inputs (whole-cell; compartment='both' peaked at 1.66 GiB on the 3.77 Mpx one).
        "mem_base_bytes": int(1.3 * 2**30),
        "mem_bytes_per_model_px": 140,
    },
    "nuclear": {
        "archive": "NuclearSegmentation-75.tar.gz",
        "md5": "efc4881db5bac23219b62486a4d877b3",
        "extract_into": "NuclearSegmentation-75",
        "saved_model": "NuclearSegmentation",
        "method": "DeepCell NuclearSegmentation 1.0 (NuclearSegmentation-75)",
        "model_mpp": 0.65,
        # Peak RSS measured the same way: 3.66 / 4.07 GiB at 2.23 / 8.93 Mpx model-scale inputs.
        "mem_base_bytes": int(3.6 * 2**30),
        "mem_bytes_per_model_px": 70,
    },
}

#: NuclearSegmentation version 1.0's post-processing, as deepcell 0.12.10 declares it in
#: ``nuclear_segmentation.CONFIGS['1.0']``. Read from there when it exists; these are the values.
_NUCLEAR_V1_0_KWARGS = {
    "model_mpp": 0.65,
    "radius": 10,
    "maxima_threshold": 0.1,
    "interior_threshold": 0.01,
    "exclude_border": False,
    "small_objects_threshold": 0,
    "min_distance": 10,
}

MODEL_TYPES = ("mesmer", "nuclear")
COMPARTMENTS = ("whole-cell", "nuclear", "both")


def log(msg: str) -> None:
    """Print log messages to stderr with a prefix."""
    print(f"[deepcell-worker] {msg}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """Redirect stdout to stderr so library prints do not pollute JSON output."""
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


def _model_cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".deepcell", "models")


def _md5(path: str) -> str:
    import hashlib

    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _savedmodel_is_marked(model_dir: str, expected_hash: str) -> bool:
    """Whether ``model_dir`` holds this worker's record of a complete extraction of that archive."""
    if not os.path.isfile(os.path.join(model_dir, "saved_model.pb")):
        return False
    try:
        with open(os.path.join(model_dir, EXTRACTED_MARKER), encoding="utf-8") as fh:
            return json.load(fh).get("md5") == expected_hash
    except (OSError, ValueError, AttributeError):
        return False


def _write_marker(model_dir: str, spec: dict) -> None:
    """Record, atomically, that ``model_dir`` is a complete extraction of ``spec['archive']``."""
    path = os.path.join(model_dir, EXTRACTED_MARKER)
    partial = path + ".partial"
    try:
        with open(partial, "w", encoding="utf-8") as fh:
            json.dump({"archive": spec["archive"], "md5": spec["md5"]}, fh)
        os.replace(partial, path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.remove(partial)
        log(f"Could not record the complete extraction in {path} ({exc}); it is checked again on the next run.")


def _archive_file_sizes(tar_path: str, top: str) -> dict:
    """``{member name: size}`` of every regular file under ``top/`` in the archive."""
    import tarfile

    with tarfile.open(tar_path, "r:gz") as archive:
        return {m.name: int(m.size) for m in archive.getmembers() if m.isfile() and m.name.startswith(top + "/")}


def _first_mismatch(extract_root: str, sizes: dict):
    """The first archive file missing on disk or of another size there, described; None when all match."""
    for name in sorted(sizes):
        path = os.path.join(extract_root, name)
        if not os.path.isfile(path):
            return f"{name} is missing"
        size = os.path.getsize(path)
        if size != sizes[name]:
            return f"{name} is {size:,} bytes, the archive's copy {sizes[name]:,}"
    return None


def _newest_mtime(path: str) -> float:
    """The newest modification time of ``path`` and everything beneath it (symlinks not followed)."""
    newest = os.lstat(path).st_mtime
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            with contextlib.suppress(OSError):
                newest = max(newest, os.lstat(os.path.join(root, name)).st_mtime)
    return newest


def _remove_stale_staging(extract_root: str, spec: dict) -> None:
    """Delete this model's staging directories that a killed extraction left behind (``STALE_STAGING_SECONDS``)."""
    import shutil
    import time

    prefix = f".{spec['saved_model']}.extracting-"
    try:
        names = [n for n in os.listdir(extract_root) if n.startswith(prefix)]
    except OSError:
        return
    now = time.time()
    for name in names:
        path = os.path.join(extract_root, name)
        try:
            if os.path.islink(path) or not os.path.isdir(path):
                continue
            age = now - _newest_mtime(path)
        except OSError:
            continue
        if age > STALE_STAGING_SECONDS:
            shutil.rmtree(path, ignore_errors=True)
            log(f"Removed {path}, left {age / 3600:.0f} h ago by an extraction that was killed before it finished.")


def _occupied_without_savedmodel(model_dir: str) -> bool:
    """Whether something is at ``model_dir`` that has no ``saved_model.pb`` and would block a rename into place.

    An empty real directory does not: ``os.rename`` replaces it. A non-empty one is what an in-place
    extraction leaves when cut short before ``saved_model.pb`` (both archives store
    ``keras_metadata.pb`` first); a file or a symlink there blocks the rename as well.
    """
    if not os.path.lexists(model_dir) or os.path.isfile(os.path.join(model_dir, "saved_model.pb")):
        return False
    if os.path.islink(model_dir) or not os.path.isdir(model_dir):
        return True
    try:
        return bool(os.listdir(model_dir))
    except OSError:
        return True


def _move_aside(model_dir: str, why: str, tar_path: str) -> None:
    """Rename a broken ``model_dir`` to a free ``<dir>.incomplete-<pid>[-<n>]`` name, keeping it for inspection."""
    base = f"{model_dir}.incomplete-{os.getpid()}"
    aside, n = base, 0
    while os.path.lexists(aside):
        n += 1
        aside = f"{base}-{n}"
    try:
        os.rename(model_dir, aside)
    except FileNotFoundError:
        log(f"{model_dir} is an interrupted extraction ({why}); another run moved it aside first.")
        return
    log(f"{model_dir} is an interrupted extraction ({why}); moved it to {aside} to extract {tar_path} again.")


def _extract_atomically(tar_path: str, extract_root: str, model_dir: str, spec: dict) -> None:
    """Extract into a temporary sibling, mark it complete, then rename it into place.

    An extraction cut short (a timeout kill, a full disk) leaves only the temporary directory, which
    is never read as the cache; and a second run started at the same time finds either nothing or a
    complete, marked directory. Both archives store ``saved_model.pb`` before their largest
    ``variables`` file, so extracting in place left a directory that looked cached and was not.
    """
    import shutil
    import tarfile
    import tempfile

    os.makedirs(extract_root, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=f".{spec['saved_model']}.extracting-", dir=extract_root)
    try:
        with tarfile.open(tar_path, "r:gz") as archive:
            archive.extractall(staging)
        fresh = os.path.join(staging, spec["saved_model"])
        if not os.path.isfile(os.path.join(fresh, "saved_model.pb")):
            raise RuntimeError(f"{tar_path} did not contain {spec['saved_model']}/saved_model.pb")
        _write_marker(fresh, spec)
        try:
            os.rename(fresh, model_dir)
        except OSError:
            if not _savedmodel_is_marked(model_dir, spec["md5"]):
                raise
            log(f"{model_dir} was completed by another run meanwhile; using that copy.")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _ensure_model_cached(model_type: str = "mesmer") -> str:
    """Make ``model_type``'s SavedModel available on disk and return its directory.

    Order: a SavedModel this worker extracted completely (``EXTRACTED_MARKER``) is used as-is. One
    extracted some other way -- by an earlier version of this worker, which extracted in place, or by
    deepcell itself -- is checked file by file against the cached archive: used when it matches,
    moved aside to ``<dir>.incomplete-<pid>`` and extracted again when it does not (an interrupted
    extraction), and used as it is when no intact archive is left to check it against. A non-empty
    directory with no ``saved_model.pb`` at all (an in-place extraction cut short before it) is moved
    aside the same way. Otherwise a cached tarball with the pinned md5 is extracted; otherwise the
    tarball is downloaded from the public S3 bucket (atomically: ``.partial`` then ``os.replace``),
    hash-checked and extracted. Extraction is atomic too (``_extract_atomically``); staging
    directories a killed extraction left behind are removed once stale (``STALE_STAGING_SECONDS``).
    No step needs a DEEPCELL_ACCESS_TOKEN.
    """
    import urllib.request

    spec = MODEL_SPECS[model_type]
    cache_dir = _model_cache_dir()
    extract_root = os.path.join(cache_dir, spec["extract_into"]) if spec["extract_into"] else cache_dir
    model_dir = os.path.join(extract_root, spec["saved_model"])
    tar_path = os.path.join(cache_dir, spec["archive"])
    expected_hash = spec["md5"]

    _remove_stale_staging(extract_root, spec)
    if _savedmodel_is_marked(model_dir, expected_hash):
        log(f"{spec['archive']}: extracted weights already cached at {model_dir}.")
        return model_dir

    cached_ok = False
    if os.path.exists(tar_path):
        md5 = _md5(tar_path)
        cached_ok = md5 == expected_hash
        if not cached_ok:
            log(f"Cached {tar_path} has md5 {md5}, not {expected_hash}.")

    if os.path.isfile(os.path.join(model_dir, "saved_model.pb")):
        if not cached_ok:
            log(
                f"{spec['archive']}: extracted weights at {model_dir}, with no intact archive left to check them "
                f"against; using them as they are."
            )
            return model_dir
        mismatch = _first_mismatch(extract_root, _archive_file_sizes(tar_path, spec["saved_model"]))
        if mismatch is None:
            _write_marker(model_dir, spec)
            log(f"{spec['archive']}: extracted weights at {model_dir} match the archive file by file.")
            return model_dir
        _move_aside(model_dir, mismatch, tar_path)
    elif _occupied_without_savedmodel(model_dir):
        # Without saved_model.pb it is no SavedModel whatever else it holds, and left in place it would
        # block the atomic rename below on every run.
        _move_aside(model_dir, f"{spec['saved_model']}/saved_model.pb is missing", tar_path)

    if not cached_ok:
        # A missing archive, or a cache file failing its own checksum (an interrupted fetch), is
        # fetched again, atomically.
        os.makedirs(cache_dir, exist_ok=True)
        url = _S3 + spec["archive"]
        partial = tar_path + ".partial"
        log(f"Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, partial)
        except Exception as exc:
            with contextlib.suppress(OSError):
                os.remove(partial)
            raise RuntimeError(
                f"model_type={model_type!r} needs DeepCell's {spec['archive']} (md5 {expected_hash}), which is not "
                f"cached in {cache_dir} and could not be downloaded from {url} ({exc}). Place that file in {cache_dir} "
                f"and rerun."
                + (
                    " model_type='mesmer' with compartment='nuclear' segments nuclei with the Mesmer weights instead."
                    if model_type == "nuclear"
                    else ""
                )
            ) from exc
        md5 = _md5(partial)
        if md5 != expected_hash:
            with contextlib.suppress(OSError):
                os.remove(partial)
            raise RuntimeError(f"Model download hash mismatch for {url}: got {md5}, expected {expected_hash}")
        os.replace(partial, tar_path)
        log(f"Downloaded {os.path.getsize(tar_path) / 1e6:.1f} MB")

    log(f"Extracting {tar_path} ...")
    _extract_atomically(tar_path, extract_root, model_dir, spec)
    log("Model weights cached successfully.")
    return model_dir


def _load_application(model_type: str, model_dir: str):
    """Build the deepcell application from the SavedModel in ``model_dir``.

    Passing ``model=`` is what keeps this token-free: ``Mesmer()`` with no model calls
    ``fetch_data``, which needs the tarball (or a token) even when the extracted weights are on
    disk, and ``NuclearSegmentation()`` with no model is a TypeError in deepcell 0.12.10 (its
    ``model`` argument is required; the loader is ``from_version``, which needs a token).
    """
    import tensorflow as tf

    log(f"Loading SavedModel from {model_dir} ...")
    try:
        model = tf.keras.models.load_model(model_dir)
    except Exception as exc:
        if os.path.isfile(os.path.join(model_dir, EXTRACTED_MARKER)):
            raise
        raise RuntimeError(
            f"TensorFlow could not load the SavedModel at {model_dir} ({exc}). This worker did not extract it "
            f"(it has no {EXTRACTED_MARKER}) and no intact archive was left beside it to check it against, so it may "
            f"be an interrupted extraction. Move {model_dir} away and rerun: the worker then fetches and extracts "
            f"the weights again."
        ) from exc
    if model_type == "mesmer":
        from deepcell.applications import Mesmer

        return Mesmer(model=model)

    from deepcell.applications import NuclearSegmentation

    kwargs = dict(_NUCLEAR_V1_0_KWARGS)
    try:
        from deepcell.applications import nuclear_segmentation as _ns

        cfg = getattr(_ns, "CONFIGS", {}).get("1.0")
    except ImportError:
        cfg = None
    if cfg is not None:
        kwargs = {k: getattr(cfg, k, v) for k, v in kwargs.items()}
    return NuclearSegmentation(model, **kwargs)


_DUPLICATED_CHANNEL_NOTE = (
    "The image has one channel, so it was used as both the nuclear and the membrane marker. "
    "Whole-cell boundaries derived from a duplicated nuclear channel are nuclear boundaries "
    "wearing another name; pass a 2-channel image, or compartment='nuclear' (or "
    "model_type='nuclear'), for a claim the input supports."
)

_SAME_CHANNEL_NOTE = (
    "nuclear_channel and membrane_channel name the same channel ({0}), so it was used as both the "
    "nuclear and the membrane marker. Whole-cell boundaries derived from a duplicated nuclear channel "
    "are nuclear boundaries wearing another name; compartment='nuclear' (or model_type='nuclear') is "
    "the claim that input supports."
)


def _channel_means(img) -> list:
    """Mean intensity of each channel of a channel-last image, for choosing the markers."""
    import numpy as np

    flat = img.reshape(-1, img.shape[-1])
    return [round(float(v), 3) for v in np.mean(flat, axis=0, dtype=np.float64)]


def _check_single_channel_mapping(shape, nuclear_channel: int, membrane_channel: int, model_type: str) -> None:
    """A one-channel image has only channel 0; refuse a mapping that names another one.

    ``membrane_channel`` at its default (1) or 0 means "the one channel", as it always has; the
    nuclear model never reads the membrane slot, so only ``nuclear_channel`` is checked for it.
    """
    bad_membrane = model_type == "mesmer" and membrane_channel not in (0, 1)
    if nuclear_channel != 0 or bad_membrane:
        raise ValueError(
            f"The image {shape} has one channel, so the only channel index is 0; "
            f"nuclear_channel={nuclear_channel}, membrane_channel={membrane_channel} cannot be honoured."
        )


def _prepare_channels(img, nuclear_channel: int = 0, membrane_channel: int = 1, model_type: str = "mesmer"):
    """Reduce an image to Mesmer's ``[H, W, 2]`` nuclear-plus-membrane layout.

    Returns ``(img_input, notes, info)``. ``info`` records the channel indices actually used
    (after any channel-first transpose) and the channel count. ``notes`` is empty only when nothing
    was adapted: an image that already had two channels, read with the mapping the caller asked for.
    """
    import numpy as np

    original_shape = img.shape
    notes: list = []
    for name, value in (("nuclear_channel", nuclear_channel), ("membrane_channel", membrane_channel)):
        if int(value) != value or value < 0:
            raise ValueError(f"{name}={value!r}: a channel index is a non-negative integer (0 is the first channel).")
    nuclear_channel, membrane_channel = int(nuclear_channel), int(membrane_channel)
    default_mapping = (nuclear_channel, membrane_channel) == (0, 1)

    if img.ndim == 2:
        # Grayscale: the one channel has to stand in for both markers.
        _check_single_channel_mapping(tuple(original_shape), nuclear_channel, membrane_channel, model_type)
        info = {"n_channels": 1, "nuclear_channel": 0, "membrane_channel": 0}
        return np.stack([img, img], axis=-1), ([_DUPLICATED_CHANNEL_NOTE] if model_type == "mesmer" else []), info

    if img.ndim != 3:
        raise ValueError(
            f"Unsupported image shape: {original_shape}. Mesmer reads a 2D image or a 3D image "
            f"with a channel axis; this array has {img.ndim} dimensions."
        )

    # Which end axis holds the channels? tifffile returns a multiplexed OME-TIFF channel-first as
    # (C, H, W); skimage.io.imread returns an RGB image channel-last as (H, W, C). Both are 3D, so
    # the layout is read off the sizes: the channel axis is the shorter of the two end axes, and a
    # tie keeps the channel-last reading. Testing shape[-1] first instead would claim every
    # channel-first image wider than two pixels, and slice it along its width.
    if img.shape[0] < img.shape[-1]:
        notes.append(
            f"Read the image as channel-first {tuple(original_shape)} (axis 0 is shorter than "
            f"axis 2) and moved the channel axis last. If it was really a "
            f"{img.shape[0]}-row image with {img.shape[-1]} channels, transpose it before rerunning."
        )
        img = np.moveaxis(img, 0, -1)

    n_channels = img.shape[-1]
    if n_channels == 1:
        _check_single_channel_mapping(tuple(original_shape), nuclear_channel, membrane_channel, model_type)
        info = {"n_channels": 1, "nuclear_channel": 0, "membrane_channel": 0}
        extra = [_DUPLICATED_CHANNEL_NOTE] if model_type == "mesmer" else []
        return np.stack([img[..., 0], img[..., 0]], axis=-1), [*notes, *extra], info

    wanted = [nuclear_channel] if model_type == "nuclear" else [nuclear_channel, membrane_channel]
    out_of_range = [c for c in wanted if c >= n_channels]
    if out_of_range:
        raise ValueError(
            f"The image has {n_channels} channels (indices 0..{n_channels - 1}); nuclear_channel={nuclear_channel}, "
            f"membrane_channel={membrane_channel} names channel {out_of_range[0]}, which does not exist. "
            f"Per-channel means: {_channel_means(img)}."
        )
    if model_type == "nuclear":
        # NuclearSegmentation reads one channel; the membrane slot is never looked at.
        membrane_channel = nuclear_channel
    info = {"n_channels": int(n_channels), "nuclear_channel": nuclear_channel, "membrane_channel": membrane_channel}

    if n_channels == 2 and (nuclear_channel, membrane_channel) == (0, 1):
        # Already nuclear + membrane in the layout Mesmer wants: nothing was adapted.
        return img, notes, info

    used = {nuclear_channel, membrane_channel}
    n_unused = n_channels - len(used)
    if model_type == "nuclear":
        if n_unused:
            notes.append(
                f"The image has {n_channels} channels; channel {nuclear_channel} was used as the nuclear marker and "
                f"the other {n_unused} were not used (per-channel means: {_channel_means(img)}). "
                f"Pass nuclear_channel if that is not the nuclear marker you meant."
            )
    elif n_unused and default_mapping:
        notes.append(
            f"The image has {n_channels} channels; channel 0 was used as the nuclear marker and "
            f"channel 1 as the membrane marker, and the other {n_channels - 2} were discarded "
            f"(per-channel means: {_channel_means(img)}). Pass nuclear_channel and membrane_channel "
            f"to choose the markers; the default reads the first two."
        )
    elif n_unused:
        notes.append(
            f"The image has {n_channels} channels; channel {nuclear_channel} was used as the nuclear marker and "
            f"channel {membrane_channel} as the membrane marker, as asked, and the other {n_unused} were not used."
        )
    if model_type == "mesmer" and nuclear_channel == membrane_channel:
        notes.append(_SAME_CHANNEL_NOTE.format(nuclear_channel))
    return np.stack([img[..., nuclear_channel], img[..., membrane_channel]], axis=-1), notes, info


def _prepare_two_channel_image(img, nuclear_channel: int = 0, membrane_channel: int = 1):
    """Reduce an image to Mesmer's ``[H, W, 2]`` nuclear-plus-membrane layout.

    Returns ``(img_input, notes)``. ``notes`` is empty only when the input already had two
    channels in the layout Mesmer wants; every other input needs an adaptation, and each
    adaptation changes what the model is looking at, so it is reported back to the caller.
    ``nuclear_channel`` / ``membrane_channel`` pick the markers (defaults: the first two).
    """
    img_input, notes, _ = _prepare_channels(img, nuclear_channel, membrane_channel)
    return img_input, notes


def _read_image(img_path: Path):
    """Read the image at full resolution; returns ``(array, notes)``.

    TIFFs go through tifffile. Everything else goes through ``skimage.io.imread`` (imageio ->
    Pillow), with Pillow's decompression-bomb limit lifted for this one read: a whole-slide JPEG
    (the Lung atlas' 33264 x 18688 images, 621.6 Mpx) is over Pillow's 179 Mpx hard limit and used
    to fail with ``DecompressionBombError``. The file is one the caller named; it is read whole,
    never downsampled, and the lifted limit is reported.
    """
    notes: list = []
    if img_path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        return tifffile.imread(str(img_path)), notes

    from PIL import Image as PILImage
    from skimage import io as skio

    limit = PILImage.MAX_IMAGE_PIXELS
    PILImage.MAX_IMAGE_PIXELS = None
    try:
        img = skio.imread(str(img_path))
    finally:
        PILImage.MAX_IMAGE_PIXELS = limit
    n_px = int(img.shape[0]) * int(img.shape[1]) if img.ndim >= 2 else int(img.size)
    if limit and n_px > limit:
        notes.append(
            f"The image is {img.shape[1]} x {img.shape[0]} = {n_px:,} pixels, over Pillow's "
            f"decompression-bomb limit of {int(limit):,}; the limit was lifted for this read and the "
            f"image was read at full resolution."
        )
    return img, notes


def _fullres_mpp_from_spot_pitch(spatial_dir: Path):
    """``(um per full-resolution px, basis)`` measured from the spot pitch, or None.

    Reads ``tissue_positions.csv`` / ``tissue_positions_list.csv`` beside the scale factors (both
    Space Ranger layouts, via ``worker_utils``), takes every spot's nearest neighbour in
    ``pxl_row/col_in_fullres`` and divides Visium's 100 um centre-to-centre pitch by the median of
    those distances. Every spot on the file is used (background spots sit on the same lattice).
    """
    try:
        import numpy as np
        from scipy.spatial import cKDTree
        from worker_utils import find_tissue_positions, read_tissue_positions

        path = find_tissue_positions(spatial_dir)
        if path is None:
            return None
        frame = read_tissue_positions(path)
        xy = np.column_stack(
            [
                np.asarray(frame["pxl_row_in_fullres"], dtype=float),
                np.asarray(frame["pxl_col_in_fullres"], dtype=float),
            ]
        )
    except Exception as exc:  # an unreadable positions file leaves the hint to the scale factors
        log(f"Could not read spot positions beside the scale factors: {exc}")
        return None
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if xy.shape[0] < 3:
        return None
    dist, _ = cKDTree(xy).query(xy, k=2)
    nearest = dist[:, 1]
    nearest = nearest[nearest > 0]
    if nearest.size == 0:
        return None
    pitch_px = float(np.median(nearest))
    basis = (
        f"{Path(path).name}: the median nearest-neighbour spot distance, {pitch_px:.1f} full-resolution px over "
        f"{xy.shape[0]} spots, is Visium's {VISIUM_SPOT_PITCH_UM:g} um centre-to-centre pitch"
    )
    return VISIUM_SPOT_PITCH_UM / pitch_px, basis


def _mpp_implied_by_scalefactors(img_path: Path):
    """The microns-per-pixel a Space Ranger ``scalefactors_json.json`` implies for this image, or None.

    Only for ``tissue_hires_image.*`` / ``tissue_lowres_image.*`` with the JSON beside them (the
    ``spatial/`` layout). Full-resolution mpp comes, in order, from ``microns_per_pixel`` when the
    file has it (Visium HD); from the spot pitch in the ``tissue_positions`` file beside it
    (``VISIUM_SPOT_PITCH_UM`` over the median nearest-neighbour distance); and only without that
    file from ``spot_diameter_fullres``, which Space Ranger defines as the pixels spanning a
    theoretical 65 um spot -- approximate, because on CytAssist slides it spans about 60 um.
    ``basis`` says which was used.
    """
    name = img_path.name.lower()
    if name.startswith("tissue_hires_image"):
        scale_key = "tissue_hires_scalef"
    elif name.startswith("tissue_lowres_image"):
        scale_key = "tissue_lowres_scalef"
    else:
        return None
    sf_path = img_path.parent / "scalefactors_json.json"
    if not sf_path.is_file():
        return None
    try:
        with open(sf_path, encoding="utf-8") as fh:
            sf = json.load(fh)
        scalef = float(sf[scale_key])
        pitch = None if "microns_per_pixel" in sf else _fullres_mpp_from_spot_pitch(img_path.parent)
        if "microns_per_pixel" in sf:
            fullres_mpp = float(sf["microns_per_pixel"])
            basis = "microns_per_pixel"
        elif pitch is not None:
            fullres_mpp, basis = pitch
        else:
            fullres_mpp = VISIUM_SPOT_DIAMETER_FULLRES_UM / float(sf["spot_diameter_fullres"])
            basis = (
                f"spot_diameter_fullres, which spans a theoretical {VISIUM_SPOT_DIAMETER_FULLRES_UM:g} um spot "
                f"(Space Ranger's definition; approximate -- on CytAssist slides it spans about 60 um, and no "
                f"tissue_positions file sits beside the scale factors to measure the spot pitch from)"
            )
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
        return None
    if not (scalef > 0 and fullres_mpp > 0 and math.isfinite(scalef) and math.isfinite(fullres_mpp)):
        return None
    return {"mpp": fullres_mpp / scalef, "basis": f"{sf_path.name} {scale_key} and {basis}", "path": str(sf_path)}


def _mem_available_bytes():
    """Memory this run can still allocate, from ``worker_utils.available_memory_bytes``, or None.

    That is the smaller of ``MemAvailable`` and the room under the cgroup memory limit (page cache
    counted as reclaimable). ``MemAvailable`` alone is the host's free memory, so in a
    memory-limited container it let a whole-slide run through to an OOM kill with no JSON.
    """
    return available_memory_bytes()


def _check_memory(image_hw, image_mpp: float, spec: dict) -> None:
    """Refuse a run whose model-scale input cannot fit in the memory available now.

    DeepCell rescales the image by ``image_mpp / model_mpp`` and holds the whole rescaled image, its
    tiles, the model outputs and the post-processing arrays at once: that is intrinsic to the method.
    The estimate is the model's measured footprint in ``MODEL_SPECS``; the memory available is the
    smaller of MemAvailable and the cgroup limit's room (``_mem_available_bytes``), and the check is
    skipped only where neither can be read.
    """
    model_mpp = float(spec["model_mpp"])
    base, per_px = spec["mem_base_bytes"], spec["mem_bytes_per_model_px"]
    scale = float(image_mpp) / model_mpp
    h, w = int(image_hw[0] * scale), int(image_hw[1] * scale)
    need = base + per_px * h * w
    avail = _mem_available_bytes()
    if avail is not None and need > avail:
        raise MemoryError(
            f"DeepCell would segment a {h} x {w} model-scale input ({h * w:,} pixels: the "
            f"{image_hw[0]} x {image_hw[1]} image rescaled by image_mpp / model_mpp = {image_mpp:g} / "
            f"{model_mpp:g}), which needs about {need / 2**30:.1f} GiB by this worker's measured estimate "
            f"(~{per_px} bytes per model-scale pixel plus {base / 2**30:.1f} GiB); "
            f"{avail / 2**30:.1f} GiB is available. image_mpp sets that scale and must be the image's real "
            f"microns per pixel; run where that much memory is free."
        )


def _object_areas(mask):
    """``(areas, n_objects, max_label)`` of a label mask, in one pass over the pixels.

    ``areas`` holds the pixel count of every label present (background 0 excluded), in label
    order. The old per-label ``np.sum(mask == cid)`` was O(n_labels * H * W).
    """
    import numpy as np

    flat = np.asarray(mask).ravel()
    if flat.size == 0:
        return np.zeros(0, dtype=np.int64), 0, 0
    if flat.dtype.kind not in "iu" or (flat.dtype.kind == "u" and flat.dtype.itemsize >= 8):
        flat = flat.astype(np.int64)  # bincount cannot take floats or uint64
    counts = np.bincount(flat)
    areas = counts[1:]
    areas = areas[areas > 0]
    return areas, int(areas.size), int(counts.size - 1)


def _atomic_write(path: Path, write) -> None:
    """Write ``path`` via ``path.partial`` + ``os.replace`` so a crash never leaves half a file."""
    tmp = path.with_name(path.name + ".partial")
    try:
        write(tmp)
        os.replace(str(tmp), str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(str(tmp))
        raise


def _save_npy(path: Path, arr) -> None:
    import numpy as np

    def _write(tmp):
        with open(tmp, "wb") as fh:
            np.save(fh, arr)

    _atomic_write(path, _write)


def _save_tiff(path: Path, arr) -> None:
    import tifffile

    _atomic_write(path, lambda tmp: tifffile.imwrite(str(tmp), arr))


def _run_deepcell(
    image_path: str,
    output_dir: str,
    model_type: str,
    compartment: str,
    image_mpp: float,
    nuclear_channel: int = 0,
    membrane_channel: int = 1,
    image_mpp_is_measured: bool = False,
) -> WorkerOutput:
    """Run DeepCell segmentation and return WorkerOutput (caller emits).

    ``image_mpp_is_measured`` says ``image_mpp`` is the image's measured pixel size. The value alone
    cannot say so when it equals the default (0.5): without the flag a run at 0.5 is reported as
    having assumed it, and its areas stay in pixels.
    """
    import numpy as np

    if model_type not in MODEL_TYPES:
        raise ValueError(unsupported_choice_msg("model_type", model_type, MODEL_TYPES))
    if compartment not in COMPARTMENTS:
        raise ValueError(unsupported_choice_msg("compartment", compartment, COMPARTMENTS))
    try:
        image_mpp = float(image_mpp)
    except (TypeError, ValueError):
        raise ValueError(f"image_mpp={image_mpp!r} is not a number of microns per pixel.") from None
    if not (math.isfinite(image_mpp) and image_mpp > 0):
        raise ValueError(
            f"image_mpp={image_mpp!r}: DeepCell rescales the image by image_mpp / model_mpp, so it must be a "
            f"positive number of microns per pixel."
        )
    image_mpp_is_measured = bool(image_mpp_is_measured)
    image_mpp_assumed = image_mpp == DEFAULT_IMAGE_MPP and not image_mpp_is_measured
    requested_compartment = compartment

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = MODEL_SPECS[model_type]

    # -- Ensure model weights are available without token (before a large image is read) --
    model_dir = _ensure_model_cached(model_type)

    # -- Load image -------------------------------------------------------
    img_path = Path(image_path)
    log(f"Loading image: {img_path}")
    img, read_notes = _read_image(img_path)
    log(f"Image shape: {img.shape}, dtype: {img.dtype}")

    # -- Prepare image (Mesmer: [batch, H, W, 2] nuclear + membrane; nuclear model: [batch, H, W, 1]) --
    original_shape = img.shape
    img_input, channel_notes, channel_info = _prepare_channels(
        img, nuclear_channel, membrane_channel, model_type=model_type
    )
    channel_notes = [*read_notes, *channel_notes]
    for note in channel_notes:
        log(f"Warning: {note}")

    img_batch = np.expand_dims(img_input, axis=0)
    _check_memory(img_input.shape[:2], image_mpp, spec)

    # -- Initialize application -------------------------------------------
    log(f"Loading {spec['method']} ...")
    app = _load_application(model_type, model_dir)
    if model_type == "nuclear":
        # NuclearSegmentation expects single-channel input [batch, H, W, 1]: the nuclear marker.
        img_batch = img_batch[..., :1]
        compartment = "nuclear"
    log(f"Model input shape: {img_batch.shape}")

    # -- Run prediction ---------------------------------------------------
    log(f"Running {model_type} prediction (compartment={compartment}, image_mpp={image_mpp})...")

    if model_type == "mesmer":
        masks = app.predict(
            img_batch,
            image_mpp=image_mpp,
            compartment=compartment,
        )
    else:
        masks = app.predict(img_batch, image_mpp=image_mpp)

    # masks shape: [batch, H, W, n_compartments]
    log(f"Prediction complete. Output shape: {masks.shape}")

    # -- Process masks ----------------------------------------------------
    output_files = {}
    write_warnings: list = []
    label_notes: list = []

    def _count(mask, what):
        areas, n_objects, max_label = _object_areas(mask)
        if max_label != n_objects:
            label_notes.append(
                f"The {what} mask's label ids run to {max_label} but it holds {n_objects} distinct objects "
                f"(some ids are absent, e.g. objects too small to survive the resize back to the image's "
                f"scale); n_cells reports the {n_objects} objects in the mask."
            )
        return areas, n_objects

    if model_type == "mesmer" and compartment == "both":
        # Two masks: whole-cell and nuclear
        whole_cell_mask = masks[0, ..., 0]
        nuclear_mask = masks[0, ..., 1]

        wc_path = out_dir / "deepcell_wholecell_mask.npy"
        nuc_path = out_dir / "deepcell_nuclear_mask.npy"
        _save_npy(wc_path, whole_cell_mask)
        _save_npy(nuc_path, nuclear_mask)
        output_files["wholecell_mask_npy"] = str(wc_path)
        output_files["nuclear_mask_npy"] = str(nuc_path)

        cell_sizes, n_cells_wc = _count(whole_cell_mask, "whole-cell")
        _, n_cells_nuc = _count(nuclear_mask, "nuclear")
        log(f"Whole-cell: {n_cells_wc} cells, Nuclear: {n_cells_nuc} nuclei")

        # Save as TIFF
        try:
            wc_tiff = out_dir / "deepcell_wholecell_mask.tif"
            nuc_tiff = out_dir / "deepcell_nuclear_mask.tif"
            _save_tiff(wc_tiff, whole_cell_mask.astype(np.uint32))
            _save_tiff(nuc_tiff, nuclear_mask.astype(np.uint32))
            output_files["wholecell_mask_tiff"] = str(wc_tiff)
            output_files["nuclear_mask_tiff"] = str(nuc_tiff)
        except Exception as e:
            log(f"Warning: could not save TIFF masks: {e}")
            write_warnings.append(f"The TIFF masks were not written ({e}); the .npy masks were.")

        primary_mask = whole_cell_mask
        n_cells = n_cells_wc

    else:
        # Single mask output
        mask = masks[0, ..., 0]
        cell_sizes, n_cells = _count(mask, compartment)
        log(f"Detected {n_cells} objects")

        mask_name = "deepcell_mask"
        mask_npy = out_dir / f"{mask_name}.npy"
        _save_npy(mask_npy, mask)
        output_files["mask_npy"] = str(mask_npy)

        try:
            mask_tiff = out_dir / f"{mask_name}.tif"
            _save_tiff(mask_tiff, mask.astype(np.uint32))
            output_files["mask_tiff"] = str(mask_tiff)
        except Exception as e:
            log(f"Warning: could not save TIFF mask: {e}")
            write_warnings.append(f"The TIFF mask was not written ({e}); the .npy mask was.")

        primary_mask = mask

    output_files["output_dir"] = str(out_dir)

    # -- Overlay plot -----------------------------------------------------
    overlay_path = out_dir / "deepcell_overlay.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Original image
        if img_input.shape[-1] == 2 and model_type == "mesmer":
            # Show as composite: nuclear=blue, membrane=green
            display_img = np.zeros((*img_input.shape[:2], 3), dtype=np.float32)
            for ch in range(2):
                ch_data = img_input[..., ch].astype(np.float32)
                vmin, vmax = np.percentile(ch_data, [1, 99])
                if vmax > vmin:
                    ch_data = np.clip((ch_data - vmin) / (vmax - vmin), 0, 1)
                else:
                    ch_data = np.zeros_like(ch_data)
                if ch == 0:
                    display_img[..., 2] = ch_data  # Blue for nuclear
                else:
                    display_img[..., 1] = ch_data  # Green for membrane
            axes[0].imshow(display_img)
        else:
            axes[0].imshow(img_input[..., 0], cmap="gray")
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # Mask overlay
        axes[1].imshow(img_input[..., 0], cmap="gray", alpha=0.5)
        masked = np.ma.masked_where(primary_mask == 0, primary_mask)
        axes[1].imshow(masked, cmap="nipy_spectral", alpha=0.5, interpolation="nearest")
        axes[1].set_title(f"Segmentation ({n_cells} objects)")
        axes[1].axis("off")

        plt.tight_layout()
        _atomic_write(overlay_path, lambda tmp: plt.savefig(str(tmp), dpi=150, bbox_inches="tight", format="png"))
        plt.close(fig)
        output_files["overlay_png"] = str(overlay_path)
        log(f"Saved overlay to {overlay_path}")
    except Exception as e:
        log(f"Warning: could not generate overlay: {e}")
        write_warnings.append(f"The overlay PNG was not written ({e}).")

    # -- Cell size statistics ---------------------------------------------
    size_stats = {}
    if len(cell_sizes) > 0:
        size_stats = {
            "mean_area_px": float(np.mean(cell_sizes)),
            "median_area_px": float(np.median(cell_sizes)),
            "min_area_px": int(np.min(cell_sizes)),
            "max_area_px": int(np.max(cell_sizes)),
            "std_area_px": float(np.std(cell_sizes)),
        }
        # um2 areas are a measurement only when the pixel size is: at the assumed default they would
        # state Mesmer's training resolution as this image's (about 85x off on a ~4.6 um/px Visium hires image).
        if not image_mpp_assumed:
            um2_per_px = image_mpp**2
            size_stats["mean_area_um2"] = float(np.mean(cell_sizes) * um2_per_px)
            size_stats["median_area_um2"] = float(np.median(cell_sizes) * um2_per_px)

    # -- What the image's own metadata says about its pixel size ----------
    implied = _mpp_implied_by_scalefactors(img_path)
    mpp_notes: list = []
    if image_mpp_assumed:
        mpp_notes.append(
            f"image_mpp is {DEFAULT_IMAGE_MPP} um/px, its default, and image_mpp_is_measured was not set: "
            f"{DEFAULT_IMAGE_MPP} is Mesmer's training resolution, not a measurement of this image. The model "
            f"treated the image as {DEFAULT_IMAGE_MPP} um/px, and areas are reported in pixels only. Pass image_mpp "
            f"(the image's microns per pixel) for a run at the image's real scale and areas in um2; if "
            f"{DEFAULT_IMAGE_MPP} um/px is this image's measured pixel size, pass image_mpp_is_measured=True with it."
        )
    if implied is not None and abs(implied["mpp"] - image_mpp) > 0.1 * image_mpp:
        mpp_notes.append(
            f"{implied['path']} implies about {implied['mpp']:.3g} um/px for this image ({implied['basis']}), "
            f"but it was segmented as though it were {image_mpp} um/px (image_mpp)."
        )

    # -- Build output -----------------------------------------------------
    out = WorkerOutput("deepcell_seg", task="segmentation")
    out.set_data(
        image_shape=list(original_shape),
        image_dtype=str(img.dtype),
        model_input_shape=list(img_batch.shape),
    )
    out.add_output_files(output_files)
    params = {
        "model_type": model_type,
        "compartment": compartment,
        "image_mpp": image_mpp,
        "image_mpp_source": "assumed default" if image_mpp_assumed else "caller",
        "image_mpp_is_measured": image_mpp_is_measured,
        "model_mpp": spec["model_mpp"],
        "nuclear_channel": channel_info["nuclear_channel"],
        "membrane_channel": channel_info["membrane_channel"],
        "n_image_channels": channel_info["n_channels"],
    }
    if implied is not None:
        params["image_mpp_implied_by_scalefactors"] = round(float(implied["mpp"]), 4)
        params["image_mpp_implied_basis"] = implied["basis"]
    out.add_params(params)
    record_method(out, spec["method"])
    if model_type == "nuclear" and requested_compartment != "nuclear":
        record_ignored(
            out,
            ["compartment"],
            f"compartment={requested_compartment!r} applies to Mesmer; NuclearSegmentation segments nuclei only.",
        )
    if model_type == "nuclear" and membrane_channel != 1:
        record_ignored(out, ["membrane_channel"], "NuclearSegmentation reads only the nuclear channel.")
    for note in [*channel_notes, *label_notes, *mpp_notes, *write_warnings]:
        out.add_warning(note)

    summary = {"n_cells": n_cells, "cell_size_stats": size_stats}
    if model_type == "mesmer" and compartment == "both":
        summary["n_cells_wholecell"] = n_cells_wc
        summary["n_cells_nuclear"] = n_cells_nuc
    out.set_summary(**summary)

    # Analysis text
    mpp_text = (
        f"at the assumed default {image_mpp} um/px (not read from the image)"
        if image_mpp_assumed
        else f"at {image_mpp} um/px (caller-supplied{', measured' if image_mpp_is_measured else ''})"
    )
    analysis_lines = [
        f"{spec['method']}, compartment={compartment}, segmented {n_cells} objects "
        f"from image of shape {list(original_shape)} {mpp_text}.",
    ]
    if size_stats:
        um2_text = f" ({size_stats['mean_area_um2']:.1f}um2)" if "mean_area_um2" in size_stats else ""
        analysis_lines.append(
            f"Cell areas: mean={size_stats['mean_area_px']:.0f}px{um2_text}, "
            f"median={size_stats['median_area_px']:.0f}px, "
            f"range=[{size_stats['min_area_px']}, {size_stats['max_area_px']}]px."
        )
    if implied is not None and mpp_notes:
        analysis_lines.append(f"The image's scalefactors imply about {implied['mpp']:.3g} um/px.")
    out.set_analysis(" ".join(analysis_lines))

    return out


def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="DeepCell cell segmentation worker")
    parser.add_argument("--image-path", required=True, help="Path to input image (TIF/PNG/JPG)")
    parser.add_argument("--output-dir", default=default_output_dir(), help="Output directory")
    parser.add_argument(
        "--model-type",
        default="mesmer",
        choices=list(MODEL_TYPES),
        help="Model type: 'mesmer' (whole-cell/nuclear) or 'nuclear' (NuclearSegmentation 1.0, nuclei only)",
    )
    parser.add_argument(
        "--compartment",
        default="whole-cell",
        choices=list(COMPARTMENTS),
        help="Segmentation compartment (mesmer only): 'whole-cell', 'nuclear', or 'both'",
    )
    parser.add_argument(
        "--image-mpp",
        type=float,
        default=DEFAULT_IMAGE_MPP,
        help=(
            "Microns per pixel of the image. The default 0.5 is Mesmer's training resolution, not a "
            "measurement: a run at it is reported as having assumed it, and areas stay in pixels, unless "
            "--image-mpp-is-measured says 0.5 was measured."
        ),
    )
    parser.add_argument(
        "--image-mpp-is-measured",
        action="store_true",
        help=(
            "image_mpp is the image's measured pixel size. Needed only when that measurement is the default "
            "0.5: without it a run at 0.5 is reported as having assumed it, and areas stay in pixels."
        ),
    )
    parser.add_argument(
        "--nuclear-channel",
        type=int,
        default=0,
        help="Index of the nuclear-marker channel (e.g. DAPI) in a multi-channel image. Default 0.",
    )
    parser.add_argument(
        "--membrane-channel",
        type=int,
        default=1,
        help="Index of the membrane/whole-cell-marker channel (Mesmer only). Default 1.",
    )

    args = parser.parse_args()

    # Preflight checks
    try:
        preflight_check(
            inputs={"image_path": args.image_path},
            output_dir=args.output_dir,
            packages=["deepcell"],
        )
    except (FileNotFoundError, PermissionError, ImportError) as e:
        WorkerOutput.emit_error("deepcell_seg", str(e), task="segmentation")
        sys.exit(1)

    run_error = None
    with _redirect_stdout_to_stderr():
        try:
            worker_out = _run_deepcell(
                image_path=args.image_path,
                output_dir=args.output_dir,
                model_type=args.model_type,
                compartment=args.compartment,
                image_mpp=args.image_mpp,
                nuclear_channel=args.nuclear_channel,
                membrane_channel=args.membrane_channel,
                image_mpp_is_measured=args.image_mpp_is_measured,
            )
        except Exception as e:
            log(f"ERROR: {e}")
            import traceback

            traceback.print_exc(file=sys.stderr)
            run_error = e

    # Emit JSON to real stdout (after redirect context is closed)
    if run_error is not None:
        WorkerOutput.emit_error("deepcell_seg", str(run_error), task="segmentation", exc=run_error)
        sys.exit(1)
    worker_out.emit()


if __name__ == "__main__":
    _cli_main()
