#!/usr/bin/env python3
"""
Cellpose worker: cell segmentation for microscopy images.

- Runs inside /opt/conda/envs/cellpose_env (cellpose 4.1.1).
- All logs/progress go to stderr.
- stdout is reserved for a single final JSON line.

Uses the Cellpose v4+ API (``CellposeModel``, not the removed ``Cellpose`` class). Cellpose 4.x
ships exactly one model, ``cpsam`` (Cellpose-SAM): ``models.MODEL_NAMES == ["cpsam"]``,
``CellposeModel`` logs "model_type argument is not used in v4.0.1+" and ``eval`` logs "channels
deprecated in v4.0.1+". So ``model_type`` and ``channels`` select nothing here. Both are still
accepted (removing them would break every caller), neither is passed on, and the payload lists them
under ``params.ignored`` and names the model that actually ran (``params.model``, from the loaded
weights file).

The model runs on CPU with float32 weights. Upstream's default is bfloat16, which on CPU is broken
rather than approximate: on a synthetic image of 40 disks cpsam found 2 cells in bfloat16 and 38 in
float32, and the mask step crashed on the bfloat16 flows of every library Visium image tried.
``params.weights_dtype`` records the precision.

Cellpose 4.x has no size model either: ``diameter`` only sets the network's rescale factor
``30 / diameter``, and ``diameter=0`` means no rescaling -- the same as ``diameter=30``. Nothing is
estimated; ``params.image_scaling`` records the factor that was applied.

Every per-cell step (areas, outlines, overlay) is one pass over the mask plus work proportional to
each cell's bounding box. The upstream helpers the worker used to call (``utils.outlines_list``,
``plot.mask_overlay``) and its own area loop each compared the whole mask against every label --
``O(n_cells * H * W)``, which never finishes on a whole-slide image.
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
from worker_utils import WorkerOutput, default_output_dir, preflight_check, record_ignored, record_method

#: cpsam's training diameter. Upstream ``CellposeModel.eval`` rescales the image by
#: ``30 / diameter`` and uses 1.0 when diameter is None or <= 0 (models.py, "image_scaling").
CPSAM_NATIVE_DIAMETER = 30.0

#: The one model cellpose 4.x ships. It is also the portal's model_type default, so a default call
#: names the model that runs.
DEFAULT_MODEL_TYPE = "cpsam"

#: Pixels per row chunk when counting label areas: bounds the intp copy ``np.bincount`` makes.
_AREA_CHUNK_PIXELS = 1 << 24

#: Rows per chunk when colouring the overlay: bounds the float32 HSV intermediates.
_OVERLAY_CHUNK_PIXELS = 1 << 22

#: A lower bound on the resident memory of a segmentation, measured with cellpose 4.1.1 on CPU. Fixed:
#: ~3 GB for torch, the float32 cpsam weights and one batch of ViT activations (3.47 GB peak on a
#: 0.34 Mpx image). Then ~70 B per pixel of the image the network sees -- ``h * w * image_scaling ** 2``
#: -- for the normalised and resized float32 image, its tiles, the network outputs over them and the
#: flows resampled back (peak RSS of this worker with the ViT forward replaced by outputs of the same
#: shape, 4-16 Mpx images). The mask step adds ~380 B per foreground pixel on top; how many pixels are
#: foreground is not known before the network has run, so the estimate leaves it out and only
#: refuses a run that cannot fit.
_SEG_FIXED_BYTES = 3.0e9
_SEG_BYTES_PER_SCALED_PIXEL = 70.0
_MASK_STEP_BYTES_PER_FOREGROUND_PIXEL = 380.0

#: The message torch raises when cellpose 4.1.1's ``dynamics.get_masks_torch`` slices an 11 x 11 window
#: around a seed that sits within 5 bins of the histogram's top/left edge (a negative slice start).
_UPSTREAM_EDGE_SEED_ERROR = "expanded size of the tensor (11)"

#: Where the memory this process may still allocate is read from.
_MEMINFO_PATH = "/proc/meminfo"
#: (limit, usage, stat, hierarchical prefix in stat) for cgroup v2, then v1.
_CGROUP_MEMORY_FILES = (
    ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.stat", ""),
    (
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/memory/memory.stat",
        "total_",
    ),
)
#: Any cgroup "limit" at or above this is the no-limit sentinel, not a limit.
_NO_CGROUP_LIMIT = float(1 << 60)


def log(msg: str) -> None:
    """Print log messages to stderr with a prefix."""
    print(f"[cellpose-worker] {msg}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """Redirect stdout to stderr so library prints do not pollute JSON output."""
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


# ----------------------------------------------------------------------------- small helpers


def _gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.1f} GB"


def _write_atomic(path, write) -> None:
    """Call ``write(tmp_path)``, then move the finished file over ``path``; no partial file survives."""
    tmp = str(path) + ".partial"
    try:
        write(tmp)
        os.replace(tmp, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def _cellpose_version() -> str:
    try:
        from importlib.metadata import version as _dist_version  # Python 3.8+

        return str(_dist_version("cellpose"))
    except Exception:
        pass
    try:
        import cellpose

        return str(getattr(cellpose, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _accepts_kwarg(func, name: str) -> bool:
    """Whether ``func`` (a callable or a class) takes keyword ``name``; False if it cannot be inspected."""
    import inspect

    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _read_text(path: str):
    """The text of a small kernel file, or None when it cannot be read."""
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, ValueError):
        return None


def _meminfo_available_bytes():
    """MemAvailable from /proc/meminfo, or None when it cannot be read."""
    text = _read_text(_MEMINFO_PATH)
    for line in (text or "").splitlines():
        if line.startswith("MemAvailable:"):
            try:
                return float(line.split()[1]) * 1024.0
            except (IndexError, ValueError):
                return None
    return None


def _cgroup_room_bytes():
    """Room left under this process's cgroup memory limits, or None when no limit is set or none can be read.

    Delegates to ``worker_utils._cgroup_memory_room_bytes`` over this process's own cgroup and its
    ancestors, then the root rows in :data:`_CGROUP_MEMORY_FILES`; the tightest level wins. The root
    rows alone missed a Slurm or systemd job limit on a host without a cgroup namespace, so a run the
    job could not hold was OOM-killed instead of refused (hunt 2026-09-30, u29a-mcp-transport-17).
    Page cache counts as reclaimable there, as it did here.
    """
    import worker_utils

    own = worker_utils._own_cgroup_memory_files()[:-2]  # the last two rows are the root's
    return worker_utils._cgroup_memory_room_bytes(files=tuple(own) + tuple(_CGROUP_MEMORY_FILES))


def _available_bytes():
    """Memory this process can still allocate: the smaller of MemAvailable and the room under the
    cgroup limit (page cache counted as reclaimable). None if neither can be read."""
    found = [v for v in (_meminfo_available_bytes(), _cgroup_room_bytes()) if v is not None]
    return min(found) if found else None


def _image_hw(shape) -> tuple:
    """(height, width) of an image array the way cellpose reads it (a leading axis shorter than the
    trailing one is the channel axis)."""
    shape = tuple(int(s) for s in shape)
    if len(shape) < 2:
        raise ValueError(f"an image has at least two dimensions; this one has shape {shape}")
    if len(shape) == 2:
        return shape
    if len(shape) == 3:
        return (shape[1], shape[2]) if shape[0] < shape[2] else (shape[0], shape[1])
    return (shape[-3], shape[-2])  # a stack of channel-last planes: the size of one plane


def _image_scaling(diameter: float) -> float:
    """The rescale factor cellpose 4.x applies: 30/diameter, or 1.0 for diameter 0."""
    return CPSAM_NATIVE_DIAMETER / diameter if diameter > 0 else 1.0


def _validate_diameter(diameter) -> float:
    try:
        value = float(diameter)
    except (TypeError, ValueError):
        raise ValueError(f"diameter={diameter!r} is not a number; pass the cell diameter in pixels") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"diameter={diameter!r} is not a cell diameter. Pass the typical cell diameter in pixels (the image "
            f"is rescaled by {CPSAM_NATIVE_DIAMETER:g}/diameter), or 0 for no rescaling. Cellpose 4.x has no "
            "diameter estimation."
        )
    return value


def _segmentation_bytes(h: int, w: int, image_scaling: float) -> float:
    """A lower bound on the peak resident memory of segmenting an ``h x w`` image at ``image_scaling``."""
    return _SEG_FIXED_BYTES + float(h) * float(w) * image_scaling**2 * _SEG_BYTES_PER_SCALED_PIXEL


def _check_memory(h: int, w: int, image_scaling: float) -> None:
    """Refuse, with the numbers, a segmentation that cannot fit -- before anything large is allocated."""
    need = _segmentation_bytes(h, w, image_scaling)
    have = _available_bytes()
    if have is None or need <= have:
        return
    raise MemoryError(
        f"Segmenting this {h} x {w} image ({h * w:,} px) at image_scaling={image_scaling:g} needs at least "
        f"{_gb(need)} (cpsam weights plus ~{_SEG_BYTES_PER_SCALED_PIXEL:g} B per pixel of the rescaled image, "
        f"measured with cellpose 4.1.1 on CPU; the mask step adds ~{_MASK_STEP_BYTES_PER_FOREGROUND_PIXEL:g} B per "
        f"foreground pixel on top); {_gb(have)} is available. image_scaling is {CPSAM_NATIVE_DIAMETER:g}/diameter, "
        f"so a diameter below {CPSAM_NATIVE_DIAMETER:g} px enlarges the image the network sees -- set diameter to "
        "the true cell diameter in pixels, not to save memory. The image is always segmented at full resolution: "
        "free memory or run on a machine with more."
    )


def _peek_hw(image_path: str):
    """(height, width) from the file header without decoding the pixels, or None if it cannot be read."""
    path = Path(image_path)
    try:
        if path.suffix.lower() in (".tif", ".tiff"):
            import tifffile

            with tifffile.TiffFile(str(path)) as tif:
                return _image_hw(tif.series[0].shape)
        from PIL import Image

        limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None  # only the header is read here
        try:
            with Image.open(str(path)) as im:
                w, h = im.size
        finally:
            Image.MAX_IMAGE_PIXELS = limit
        return (int(h), int(w))
    except Exception:
        return None


def _read_image(image_path: str):
    """Read the whole image at full resolution. Returns ``(array, note)``.

    TIFFs go through tifffile. Everything else goes through imageio's Pillow plugin, whose
    decompression-bomb guard refuses anything above ``2 * Image.MAX_IMAGE_PIXELS`` (178,956,970 px
    on Pillow 12) -- every full-resolution Visium CytAssist JPG is larger. The file is the caller's
    own choice and the memory check has already sized it, so the guard is lifted for this one read
    and restored after; ``note`` says so when it mattered. The image is never downscaled.
    """
    path = Path(image_path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        return tifffile.imread(str(path)), None

    try:
        import imageio.v2 as imageio  # the pre-v3 imread behaviour, without its deprecation warning
    except ImportError:  # imageio < 2.16 has no v2 namespace; its imread is the v2 one
        import imageio
    from PIL import Image

    limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        img = imageio.imread(str(path))
    finally:
        Image.MAX_IMAGE_PIXELS = limit
    h, w = _image_hw(getattr(img, "shape", (0, 0)))
    note = None
    if limit and h * w > limit:
        note = (
            f"Pillow's decompression-bomb guard ({int(limit):,} px) was lifted for this user-supplied image "
            f"({h} x {w} = {h * w:,} px); it was read and segmented at full resolution."
        )
    return img, note


# ----------------------------------------------------------------------------- per-cell work


def _label_areas(masks):
    """Pixel count of every label 1..max in a single pass (index ``i`` is label ``i + 1``)."""
    import numpy as np

    m = np.asarray(masks)
    if m.size == 0:
        return np.zeros(0, dtype=np.int64)
    top = int(m.max())
    if top <= 0:
        return np.zeros(0, dtype=np.int64)
    if int(m.min()) < 0:
        raise ValueError("a label mask cannot hold negative labels")
    rows = m.reshape(m.shape[0], -1) if m.ndim > 1 else m.reshape(1, -1)
    step = max(1, _AREA_CHUNK_PIXELS // max(1, rows.shape[1]))
    counts = np.zeros(top + 1, dtype=np.int64)
    for start in range(0, rows.shape[0], step):
        chunk = rows[start : start + step].ravel().astype(np.intp, copy=False)
        counts += np.bincount(chunk, minlength=top + 1)
    return counts[1:]


def _mask_area_stats(masks):
    """``(n_cells, stats)`` over the labels present in ``masks``; the same statistics as before, one pass."""
    import numpy as np

    areas = _label_areas(masks)
    present = areas[areas > 0]
    if present.size == 0:
        return 0, {}
    return int(present.size), {
        "mean_area_px": float(np.mean(present)),
        "median_area_px": float(np.median(present)),
        "min_area_px": int(np.min(present)),
        "max_area_px": int(np.max(present)),
        "std_area_px": float(np.std(present)),
    }


def _mask_outlines(masks):
    """``cellpose.utils.outlines_list`` computed per bounding box instead of per whole image.

    Same output: one ``(k, 2)`` array of (x, y) pixel coordinates per label present, in label
    order -- the longest external contour of that label, or an empty ``(0, 2)`` array when it has
    four points or fewer. Each label is traced on a zero-padded crop of its own bounding box and
    shifted back, so the cost is one ``find_objects`` pass plus the sum of the boxes.
    """
    import cv2
    import numpy as np
    from scipy.ndimage import find_objects

    m = np.asarray(masks)
    if m.ndim != 2:
        raise ValueError(f"outlines are traced on a 2-D label mask; this mask is {m.ndim}-D with shape {m.shape}")
    outlines = []
    if m.size == 0 or int(m.max()) <= 0:
        return outlines
    for index, box in enumerate(find_objects(m)):
        if box is None:
            continue
        rows, cols = box
        crop = np.pad((m[rows, cols] == index + 1).astype(np.uint8), 1)
        contours = cv2.findContours(crop, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[-2]
        longest = int(np.argmax([c.shape[0] for c in contours]))
        pix = contours[longest].astype(int).squeeze()
        if len(pix) > 4:
            outlines.append(pix + np.array([cols.start - 1, rows.start - 1]))
        else:
            outlines.append(np.zeros((0, 2)))
    return outlines


def _gray(img):
    """float32 intensity (H, W): the mean of the first three channels, channel axis found as cellpose does."""
    import numpy as np

    a = np.asarray(img)
    if a.ndim == 2:
        return a.astype(np.float32)
    if a.ndim == 3 and a.shape[0] < a.shape[2]:
        a = np.moveaxis(a, 0, -1)
    if a.ndim != 3:
        raise ValueError(f"cannot draw an overlay for an image of shape {a.shape}")
    return a[..., :3].astype(np.float32).mean(axis=-1)


def _hsv_to_rgb(h, s, v):
    """Vectorised ``colorsys.hsv_to_rgb`` (the per-pixel Python call cellpose's plot helper used)."""
    import numpy as np

    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i.astype(np.int64) % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def _mask_overlay(img, masks, seed: int = 0):
    """``cellpose.plot.mask_overlay`` in one vectorised pass: each cell gets its own hue over the
    grayscale image. Row-chunked so the float intermediates stay small; the result is uint8 RGB."""
    import numpy as np

    m = np.asarray(masks)
    gray = _gray(img)
    if gray.shape != m.shape:
        raise ValueError(f"image {gray.shape} and mask {m.shape} differ in shape")
    n = max(0, int(m.max())) if m.size else 0
    hues = np.linspace(0, 1, n + 1)[np.random.default_rng(seed).permutation(n)].astype(np.float32)
    scale = 255.0 if float(gray.max()) > 1 else 1.0
    out = np.zeros(m.shape + (3,), dtype=np.uint8)
    step = max(1, _OVERLAY_CHUNK_PIXELS // max(1, m.shape[1]))
    for start in range(0, m.shape[0], step):
        lab = m[start : start + step].astype(np.int64, copy=False)
        v = np.clip(gray[start : start + step] / scale * 1.5, 0, 1)
        cell = lab > 0
        h = np.zeros(lab.shape, dtype=np.float32)
        if n:
            h[cell] = hues[lab[cell] - 1]
        s = cell.astype(np.float32)
        out[start : start + step] = (_hsv_to_rgb(h, s, v) * 255).astype(np.uint8)
    return out


def _display_image(img):
    """What the 'Original Image' panel can show: RGB(A) as is, anything else as grayscale."""
    import numpy as np

    a = np.asarray(img)
    if a.ndim == 3 and a.shape[-1] in (3, 4) and not a.shape[0] < a.shape[2]:
        return a, None
    return _gray(a), "gray"


# ----------------------------------------------------------------------------- the run


def _run_cellpose(
    image_path: str,
    output_dir: str,
    model_type: str,
    diameter: float,
    channels,
    flow_threshold: float,
    cellprob_threshold: float,
) -> WorkerOutput:
    """Run Cellpose segmentation and return WorkerOutput (caller emits)."""
    import numpy as np

    diameter = _validate_diameter(diameter)
    image_scaling = _image_scaling(diameter)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = WorkerOutput("cellpose_seg", task="segmentation")

    # ── Load image (full resolution; sized before it is decoded) ─────────────────────────
    img_path = Path(image_path)
    log(f"Loading image: {img_path}")
    peek = _peek_hw(str(img_path))
    if peek is not None:
        _check_memory(peek[0], peek[1], image_scaling)
    img, read_note = _read_image(str(img_path))
    if read_note:
        out.add_warning(read_note)
    h, w = _image_hw(img.shape)
    if peek is None:
        _check_memory(h, w, image_scaling)
    log(f"Image shape: {img.shape}, dtype: {img.dtype}")

    # ── Initialize model ────────────────────────────────────────────────
    # cellpose 4.x: model_type selects nothing (one model, cpsam), so it is not passed on.
    from cellpose import models

    cp_version = _cellpose_version()
    model_kwargs = {"gpu": False}
    if _accepts_kwarg(models.CellposeModel, "use_bfloat16"):
        # Upstream defaults to bfloat16 weights. On CPU that wrecks the output: on a synthetic image of
        # 40 disks, cpsam found 2 cells in bfloat16 and 38 in float32, and every library Visium image
        # tried crashed in the mask step on the bfloat16 flows. This worker always runs on CPU.
        model_kwargs["use_bfloat16"] = False
    model = models.CellposeModel(**model_kwargs)
    weights_dtype = "float32" if model_kwargs.get("use_bfloat16") is False else "cellpose default"
    model_name = os.path.basename(str(getattr(model, "pretrained_model", "") or "")) or DEFAULT_MODEL_TYPE
    device = str(getattr(model, "device", "cpu"))
    log(f"CellposeModel loaded: model={model_name}, cellpose {cp_version}, device={device}, weights={weights_dtype}")

    # ── Run segmentation ────────────────────────────────────────────────
    # diameter=None is upstream's "no rescaling" (image_scaling 1.0); it is not an estimate.
    log(f"Running model.eval() (image_scaling={image_scaling:g})...")
    try:
        masks, flows, styles = model.eval(
            img,
            diameter=diameter if diameter > 0 else None,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
    except RuntimeError as exc:
        if _UPSTREAM_EDGE_SEED_ERROR not in str(exc):
            raise
        raise RuntimeError(
            f"cellpose {cp_version} failed in its mask step (dynamics.get_masks_torch): the predicted flows carried "
            "more than 10 pixels over 15 px past the top or left edge of the image, and upstream cannot cut its "
            "11 x 11 seed window there. This is a cellpose defect triggered by the flows the network predicted for "
            "this image, not a problem with the file. diameter (the true cell diameter in pixels) and "
            f"cellprob_threshold change which pixels are followed. Upstream message: {exc}"
        ) from exc
    masks = np.asarray(masks)
    log(f"Segmentation complete. Masks shape: {masks.shape}")

    n_cells, size_stats = _mask_area_stats(masks)
    max_label = int(masks.max()) if masks.size else 0
    log(f"Detected {n_cells} cells")

    # ── Save masks ──────────────────────────────────────────────────────
    masks_path = out_dir / "cellpose_masks.npy"

    def _save_masks(tmp):
        with open(tmp, "wb") as fh:
            np.save(fh, masks)

    _write_atomic(masks_path, _save_masks)
    log(f"Saved masks to {masks_path}")

    # Save masks as labeled TIFF for visualization
    masks_tiff_path = out_dir / "cellpose_masks.tif"
    try:
        import tifffile

        _write_atomic(masks_tiff_path, lambda tmp: tifffile.imwrite(tmp, masks.astype(np.uint32)))
        log(f"Saved masks TIFF to {masks_tiff_path}")
    except Exception as e:
        log(f"Warning: could not save masks TIFF: {e}")
        out.add_warning(f"cellpose_masks.tif was not written: {e}")
        masks_tiff_path = None

    # ── Save cell outlines ──────────────────────────────────────────────
    outlines = _mask_outlines(masks)
    outlines_arr = np.empty(len(outlines), dtype=object)  # always 1-D, even when every outline has one shape
    for i, pix in enumerate(outlines):
        outlines_arr[i] = pix
    outlines_path = out_dir / "cellpose_outlines.npy"

    def _save_outlines(tmp):
        with open(tmp, "wb") as fh:
            np.save(fh, outlines_arr, allow_pickle=True)

    _write_atomic(outlines_path, _save_outlines)
    log(f"Saved {len(outlines)} outlines to {outlines_path}")

    # ── Save overlay plot ───────────────────────────────────────────────
    overlay_path = out_dir / "cellpose_overlay.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        shown, cmap = _display_image(img)
        ax[0].imshow(shown, cmap=cmap)
        ax[0].set_title("Original Image")
        ax[0].axis("off")
        ax[1].imshow(_mask_overlay(img, masks))
        ax[1].set_title(f"Cellpose Segmentation ({n_cells} cells)")
        ax[1].axis("off")
        plt.tight_layout()
        _write_atomic(overlay_path, lambda tmp: plt.savefig(tmp, format="png", dpi=150, bbox_inches="tight"))
        plt.close(fig)
        log(f"Saved overlay to {overlay_path}")
    except Exception as e:
        log(f"Warning: could not generate overlay plot: {e}")
        out.add_warning(f"cellpose_overlay.png was not drawn: {e}")
        overlay_path = None

    # ── Emit output ─────────────────────────────────────────────────────
    out.set_data(
        image_shape=list(img.shape),
        image_dtype=str(img.dtype),
        image_pixels=int(h) * int(w),
    )
    out.add_output_file("masks_npy", str(masks_path))
    out.add_output_file("outlines_npy", str(outlines_path))
    if masks_tiff_path:
        out.add_output_file("masks_tiff", str(masks_tiff_path))
    if overlay_path:
        out.add_output_file("overlay_png", str(overlay_path))
    out.add_output_file("output_dir", str(out_dir))

    out.add_params(
        {
            "model_type": model_type,
            "model": model_name,
            "cellpose_version": cp_version,
            "device": device,
            "weights_dtype": weights_dtype,
            "diameter": diameter,
            "image_scaling": image_scaling,
            "channels": channels,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
        }
    )
    record_method(out, f"Cellpose {cp_version} CellposeModel, pretrained model {model_name}")
    ignored_why = []
    ignored = []
    if str(model_type or "").strip().lower() not in ("", model_name.lower()):
        ignored.append("model_type")
        ignored_why.append(
            f"model_type={model_type!r} selects nothing in cellpose {cp_version}, which ships one model; "
            f"{model_name} ran"
        )
    ignored.append("channels")
    ignored_why.append(
        f"channels={channels!r} is not used by cellpose {cp_version}; {model_name} reads the image's first "
        "3 channels as given"
    )
    record_ignored(out, ignored, "; ".join(ignored_why))

    out.set_summary(
        n_cells=n_cells,
        cell_size_stats=size_stats,
    )
    if max_label != n_cells:
        out.add_warning(f"the mask's labels are not consecutive: {n_cells} cells carry labels up to {max_label}")

    if diameter > 0:
        scale_text = f"diameter={diameter:g} px (image rescaled by {image_scaling:g})"
    else:
        scale_text = "diameter=0: no rescaling (cellpose 4.x has no size model, so nothing was estimated)"
    analysis_lines = [
        f"Cellpose {cp_version} ({model_name}, {weights_dtype} weights on {device}) segmented {n_cells} cells from "
        f"image of shape {list(img.shape)}; {scale_text}.",
    ]
    if size_stats:
        analysis_lines.append(
            f"Cell areas: mean={size_stats['mean_area_px']:.0f}px, "
            f"median={size_stats['median_area_px']:.0f}px, "
            f"range=[{size_stats['min_area_px']}, {size_stats['max_area_px']}]px."
        )
    out.set_analysis(" ".join(analysis_lines))
    return out


def _parse_channels(text):
    """The channels value as given: a parsed JSON list, or the raw text. Nothing is substituted --
    cellpose 4.x does not use it, and the payload lists it as ignored either way."""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="Cellpose cell segmentation worker")
    parser.add_argument("--image-path", required=True, help="Path to input image (TIF/PNG/JPG)")
    parser.add_argument("--output-dir", default=default_output_dir(), help="Output directory")
    parser.add_argument(
        "--model-type",
        default=DEFAULT_MODEL_TYPE,
        help="Kept for compatibility; cellpose 4.x ships one model (cpsam) and selects nothing by this",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=30.0,
        help="Cell diameter in pixels; the image is rescaled by 30/diameter. 0 = no rescaling (not an estimate)",
    )
    parser.add_argument(
        "--channels", default="[0,0]", help="Kept for compatibility; cellpose 4.x takes no channel selection"
    )
    parser.add_argument("--flow-threshold", type=float, default=0.4, help="Flow error threshold")
    parser.add_argument("--cellprob-threshold", type=float, default=0.0, help="Cell probability threshold")

    args = parser.parse_args()
    channels = _parse_channels(args.channels)

    # Preflight checks
    try:
        preflight_check(
            inputs={"image_path": args.image_path},
            output_dir=args.output_dir,
            packages=["cellpose"],
        )
    except (FileNotFoundError, PermissionError, ImportError) as e:
        WorkerOutput.emit_error("cellpose_seg", str(e), task="segmentation")
        sys.exit(1)

    run_error = None
    with _redirect_stdout_to_stderr():
        try:
            worker_out = _run_cellpose(
                image_path=args.image_path,
                output_dir=args.output_dir,
                model_type=args.model_type,
                diameter=args.diameter,
                channels=channels,
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
            )
        except Exception as e:
            log(f"ERROR: {e}")
            import traceback

            traceback.print_exc(file=sys.stderr)
            run_error = e

    # Emit JSON to real stdout (after redirect context is closed)
    if run_error is not None:
        WorkerOutput.emit_error("cellpose_seg", str(run_error), task="segmentation", exc=run_error)
        sys.exit(1)
    worker_out.emit()


if __name__ == "__main__":
    _cli_main()
