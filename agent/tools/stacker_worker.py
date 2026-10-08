#!/usr/bin/env python
"""
Stacker worker: register and align spatial tissue section images.

No upstream "Stacker" package is imported. The three modes run, directly:

- ``affine``: ANTsPy ``ants.registration(type_of_transform="Affine")``.
- ``deformable``: ANTsPy ``ants.registration(type_of_transform="SyN")``.
- ``synthmorph``: a VoxelMorph ``VxmDense`` network whose trained weights the caller supplies as
  ``--model-path`` (for example one trained with SynthMorph). No weights ship with the tool, so a
  run without a model stops, unless ``--allow-untrained-model-fallback`` accepts a randomly
  initialised network -- a near-identity warp that is flagged ``used_fallback`` and never passes
  for a learned registration.

The payload's ``params.method`` names what ran.

- Runs inside the stacker conda environment.
- All logs/progress go to stderr.
- stdout is reserved for a single final JSON line.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import traceback

# Suppress TensorFlow warnings before any TF import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Make worker_utils importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path

import numpy as np
from worker_utils import WorkerOutput, record_ignored, record_method, unsupported_choice_msg

MODES = ["affine", "deformable", "synthmorph"]

METHOD_AFFINE = "ANTsPy ants.registration(type_of_transform='Affine')"
METHOD_DEFORMABLE = "ANTsPy ants.registration(type_of_transform='SyN')"
METHOD_VXM_TRAINED = "VoxelMorph VxmDense with the weights loaded from model_path"
METHOD_VXM_UNTRAINED = (
    "VoxelMorph VxmDense with random, untrained weights (a near-identity warp, not a learned registration)"
)

# The architecture the worker builds when it is allowed to run an untrained network. Four encoder
# levels, so each image side must be a multiple of 2**4 = 16 (see _vxm_shape_multiple).
UNTRAINED_VXM_CONFIG = {
    "nb_unet_features": [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
    "nb_unet_conv_per_level": 1,
    "int_resolution": 2,
    "svf_resolution": 1,
}

# voxelmorph.tf.networks.default_unet_features() -- used when a saved config has
# nb_unet_features=None.
_VXM_DEFAULT_ENCODER_LEVELS = 4

# PIL modes that carry one channel at more than 8 bits. ``convert("L")`` clips these at 255
# rather than rescaling them, which turned a 16-bit TIFF into a near-uniformly white image.
_FULL_RANGE_MODES = ("I;16", "I;16B", "I;16L", "I;16N", "I", "F")


def log(msg):
    print(f"[stacker-worker] {msg}", file=sys.stderr)


# ----------------------------------------------------------------------------- reading images


def _is_nifti(path):
    return str(path).endswith((".nii", ".nii.gz"))


def _read_image_array(image_path):
    """Read a PNG/JPEG/TIFF as one grayscale float32 array, and say how it was read.

    Colour images become PIL luma ("L"). Single-channel images deeper than 8 bits are read at
    their own range: converting those to "L" saturates everything above 255.
    """
    from PIL import Image as PILImage

    pil_img = PILImage.open(str(image_path))
    mode = pil_img.mode
    if mode in _FULL_RANGE_MODES:
        arr = np.array(pil_img, dtype=np.float32)
        how = f"{mode}, single channel read at its full range"
    else:
        arr = np.array(pil_img.convert("L"), dtype=np.float32)
        how = mode if mode == "L" else f"{mode} converted to 8-bit grayscale (PIL 'L')"
    return arr, how


def _load_image_as_ants(image_path):
    """Load an image file as an ANTsPy image (grayscale)."""
    img, _how = _load_image_as_ants_with_mode(image_path)
    return img


def _load_image_as_ants_with_mode(image_path):
    import ants

    path = str(image_path)
    # NIfTI files can be read directly
    if _is_nifti(path):
        return ants.image_read(path), "NIfTI"
    arr, how = _read_image_array(path)
    return ants.from_numpy(arr), how


# ----------------------------------------------------------------------------- atomic writes


def _partial_path(dest):
    """A sibling temp name that keeps the extension a writer infers the format from."""
    dest = str(dest)
    for ext in (".nii.gz", ".nii", ".npy", ".mat", ".txt", ".h5"):
        if dest.endswith(ext):
            return dest[: -len(ext)] + ".partial" + ext
    return dest + ".partial"


def _write_ants_image_atomic(img, dest):
    import ants

    tmp = _partial_path(dest)
    ants.image_write(img, tmp)
    os.replace(tmp, str(dest))


def _save_npy_atomic(arr, dest):
    tmp = _partial_path(dest)
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, str(dest))


def _copy_atomic(src, dest):
    tmp = _partial_path(dest)
    shutil.copy2(str(src), tmp)
    os.replace(tmp, str(dest))


# ----------------------------------------------------------------------------- ANTsPy modes


def _run_ants_registration(moving_path, fixed_path, output_dir, type_of_transform, method):
    import ants

    log("Loading moving image...")
    moving, moving_how = _load_image_as_ants_with_mode(moving_path)
    log(f"Moving image shape: {moving.shape}")

    log("Loading fixed image...")
    fixed, fixed_how = _load_image_as_ants_with_mode(fixed_path)
    log(f"Fixed image shape: {fixed.shape}")

    log(f"Running {method}...")
    result = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=type_of_transform,
    )

    warped = result["warpedmovout"]
    fwd_transforms = result["fwdtransforms"]

    # Save warped image
    warped_path = str(output_dir / "stacker_warped.nii.gz")
    _write_ants_image_atomic(warped, warped_path)
    log(f"Saved warped image to {warped_path}")

    # Copy the forward transforms (Affine: one .mat; SyN: a warp .nii.gz and an affine .mat)
    transform_paths = []
    for i, tpath in enumerate(fwd_transforms):
        suffix = Path(tpath).suffix
        if suffix == ".gz":
            suffix = ".nii.gz"
        dest = str(output_dir / f"stacker_transform_{i}{suffix}")
        if str(tpath) != dest:
            _copy_atomic(tpath, dest)
        transform_paths.append(dest)
        log(f"Saved transform to {dest}")

    info = {
        "method": method,
        "used_fallback": False,
        "why": "",
        "params": {"type_of_transform": type_of_transform},
        "data": {"moving_image_read_as": moving_how, "fixed_image_read_as": fixed_how},
        "warnings": [],
        "analysis": (
            f"The {len(transform_paths)} forward transform file(s) are ANTs transforms in the fixed "
            "image's physical space, in the order ants.apply_transforms expects."
        ),
    }
    return warped, transform_paths, moving.shape, fixed.shape, info


def run_affine_registration(moving_path, fixed_path, output_dir):
    """ANTsPy affine registration."""
    return _run_ants_registration(moving_path, fixed_path, output_dir, "Affine", METHOD_AFFINE)


def run_deformable_registration(moving_path, fixed_path, output_dir):
    """ANTsPy SyN deformable registration."""
    return _run_ants_registration(moving_path, fixed_path, output_dir, "SyN", METHOD_DEFORMABLE)


# ----------------------------------------------------------------------------- VoxelMorph mode


def _resolve_synthmorph_model(model_path, allow_untrained_model_fallback=False):
    """Decide, before anything heavy is imported, whether a trained network is available.

    Returns ``"trained"`` when ``model_path`` names an existing file and ``"untrained"`` when no
    model was given and the caller allowed the untrained substitute. Raises otherwise: a path that
    does not exist is always an error (a typo must not quietly become a random network), and no
    path at all is an error unless ``allow_untrained_model_fallback`` is set.
    """
    if model_path:
        if not os.path.isfile(str(model_path)):
            raise FileNotFoundError(
                f"model_path={model_path!r} does not exist. mode='synthmorph' registers with the trained "
                "VoxelMorph VxmDense network saved in that .h5 file; give the path of an existing one."
            )
        return "trained"
    if allow_untrained_model_fallback:
        return "untrained"
    raise ValueError(
        "mode='synthmorph' needs model_path: the path of a trained VoxelMorph VxmDense network saved "
        "as .h5 (for example one trained with SynthMorph). No weights ship with this tool, and a "
        "randomly initialised network produces a near-identity warp, not a registration. Pass "
        "model_path, use mode='affine' or mode='deformable' (ANTsPy, no model needed), or set "
        "allow_untrained_model_fallback=True to accept the untrained network knowingly (the payload "
        "then says so)."
    )


def _as_int_factor(value):
    """An integer resolution factor > 1, or None when the value adds no divisibility constraint."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f > 1 and float(int(f)) == f:
        return int(f)
    return None


def _vxm_shape_multiple(config):
    """The number every image side must be a multiple of for a VxmDense built from ``config``.

    The U-Net max-pools by 2 once per encoder level and concatenates each level's skip connection
    on the way back up, so a side that does not halve evenly that many times breaks the
    Concatenate layer (1725 -> ... -> 107 -> 53 vs 54). The flow is also integrated at
    ``int_resolution`` and predicted at ``svf_resolution``, which must divide the side too.
    """
    config = dict(config or {})
    feats = config.get("nb_unet_features")
    conv_per_level = int(config.get("nb_unet_conv_per_level") or 1)
    if feats is None:
        n_down = _VXM_DEFAULT_ENCODER_LEVELS
    elif isinstance(feats, (int, np.integer)):
        levels = config.get("nb_unet_levels")
        if levels is None:
            raise ValueError("the model config sets nb_unet_features to an integer without nb_unet_levels")
        n_down = int(levels) - 1
    else:
        n_down = len(list(feats[0])) // max(conv_per_level, 1)
    multiple = 2 ** max(int(n_down), 0)
    resolution_keys = ("int_resolution", "svf_resolution", "int_downsize")
    for factor in (_as_int_factor(config.get(k)) for k in resolution_keys):
        if factor:
            multiple = multiple * factor // math.gcd(multiple, factor)
    return multiple


def _padded_shape(shape, multiple):
    return tuple(int(math.ceil(int(s) / float(multiple)) * multiple) for s in shape)


def _pad_to_shape(array, shape):
    """Zero-pad ``array`` to ``shape``, centred, and return the slices that crop it back.

    The same convention as ``voxelmorph.py.utils.pad``. A displacement field is relative, so the
    field predicted on the padded canvas and cropped with these slices is the field of the
    original pixels.
    """
    shape = tuple(int(s) for s in shape)
    if tuple(array.shape) == shape:
        return array, tuple(slice(0, s) for s in shape)
    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple(slice(o, o + n) for o, n in zip(offsets, array.shape))
    padded[slices] = array
    return padded, slices


def _read_for_network(image_path):
    if _is_nifti(image_path):
        img, how = _load_image_as_ants_with_mode(image_path)
        return img.numpy().astype(np.float32), how
    return _read_image_array(image_path)


def _resize_2d(arr, target_shape):
    """Bilinear resize of a 2-D float image, kept in float (no 8-bit round trip)."""
    from PIL import Image as PILImage

    pil = PILImage.fromarray(np.ascontiguousarray(arr, dtype=np.float32), mode="F")
    pil = pil.resize((int(target_shape[1]), int(target_shape[0])), PILImage.BILINEAR)
    return np.array(pil, dtype=np.float32)


def _check_model_config(config, model_path, ndims):
    inshape = config.get("inshape")
    if inshape is not None and len(inshape) != ndims:
        raise ValueError(
            f"model_path={model_path!r} holds a {len(inshape)}-D VxmDense (inshape {list(inshape)}), but "
            f"these images are {ndims}-D."
        )
    for key in ("src_feats", "trg_feats"):
        feats = config.get(key)
        if feats is not None and int(feats) != 1:
            raise ValueError(
                f"model_path={model_path!r} expects {key}={feats} channels; this tool feeds one grayscale channel."
            )


def run_synthmorph_registration(
    moving_path, fixed_path, output_dir, model_path=None, allow_untrained_model_fallback=False
):
    """VoxelMorph VxmDense registration with a caller-supplied trained network."""
    which = _resolve_synthmorph_model(model_path, allow_untrained_model_fallback)

    import tensorflow as tf
    import voxelmorph as vxm

    log("Loading moving image...")
    moving_arr, moving_how = _read_for_network(moving_path)
    moving_shape = moving_arr.shape
    log(f"Moving image shape: {moving_shape}")

    log("Loading fixed image...")
    fixed_arr, fixed_how = _read_for_network(fixed_path)
    fixed_shape = fixed_arr.shape
    log(f"Fixed image shape: {fixed_shape}")

    # Normalize to [0, 1]
    if moving_arr.max() > 0:
        moving_arr = moving_arr / moving_arr.max()
    if fixed_arr.max() > 0:
        fixed_arr = fixed_arr / fixed_arr.max()

    params = {}
    warnings_out = []
    notes = []

    # VxmDense takes both images on one grid: the moving image is resampled onto the fixed shape.
    target_shape = tuple(fixed_arr.shape)
    if tuple(moving_arr.shape) != target_shape:
        if moving_arr.ndim != 2 or len(target_shape) != 2:
            raise ValueError(
                f"mode='synthmorph' needs the two images on one grid; the moving image is {list(moving_arr.shape)} "
                f"and the fixed image {list(target_shape)}, and only 2-D images are resampled onto the fixed shape."
            )
        log(f"Resizing moving image from {moving_arr.shape} to {target_shape}")
        params["moving_resized_from"] = [int(s) for s in moving_arr.shape]
        params["moving_resized_to"] = [int(s) for s in target_shape]
        moving_arr = _resize_2d(moving_arr, target_shape)
        notes.append(
            f"The moving image was resampled (bilinear) from {params['moving_resized_from']} to the fixed image's "
            f"{params['moving_resized_to']} before registration, so the warp field is in that resampled grid."
        )
        ratio_moving = params["moving_resized_from"][0] / float(params["moving_resized_from"][1])
        ratio_fixed = target_shape[0] / float(target_shape[1])
        if abs(ratio_moving - ratio_fixed) > 0.01 * ratio_fixed:
            warnings_out.append(
                "the moving image's aspect ratio differs from the fixed image's; resampling it onto the fixed "
                f"shape stretched it ({params['moving_resized_from']} -> {params['moving_resized_to']})"
            )
    else:
        params["moving_resized_from"] = None
        params["moving_resized_to"] = None

    if which == "trained":
        try:
            config = vxm.networks.VxmDense.load_config(model_path)
        except Exception as e:
            raise ValueError(
                f"model_path={model_path!r} is not a VoxelMorph VxmDense .h5 saved with its model config: {e}"
            ) from e
        _check_model_config(config, model_path, fixed_arr.ndim)
    else:
        config = dict(UNTRAINED_VXM_CONFIG)

    multiple = _vxm_shape_multiple(config)
    padded_shape = _padded_shape(target_shape, multiple)
    fixed_p, crop = _pad_to_shape(fixed_arr, padded_shape)
    moving_p, _ = _pad_to_shape(moving_arr, padded_shape)
    params["shape_multiple"] = int(multiple)
    params["padded_shape"] = [int(s) for s in padded_shape]
    if tuple(padded_shape) != target_shape:
        log(f"Zero-padding {list(target_shape)} to {list(padded_shape)} (sides must be multiples of {multiple})")
        notes.append(
            f"The network ran on a zero-padded {list(padded_shape)} canvas (VxmDense needs every side to be a "
            f"multiple of {multiple}); the warped image and warp field were cropped back to {list(target_shape)}."
        )

    if which == "trained":
        log(f"Loading VoxelMorph VxmDense from {model_path} at inshape {list(padded_shape)}")
        # Upstream's own register script rebuilds the saved architecture at the image's shape the
        # same way: LoadableModel.load(path, inshape=..., input_model=None).
        model = vxm.networks.VxmDense.load(model_path, inshape=tuple(padded_shape), input_model=None)
    else:
        log("Building an untrained VoxelMorph VxmDense (allow_untrained_model_fallback=True)")
        model = vxm.networks.VxmDense(
            inshape=tuple(padded_shape),
            nb_unet_features=UNTRAINED_VXM_CONFIG["nb_unet_features"],
        )

    # VxmDense's second output is the half-resolution, pre-integration velocity field (its
    # regularisation target), not the displacement that warped the image. Ask for that one.
    reg_model = tf.keras.Model(model.inputs, [model.references.y_source, model.references.pos_flow])

    moving_input = moving_p[np.newaxis, ..., np.newaxis]
    fixed_input = fixed_p[np.newaxis, ..., np.newaxis]
    log("Running VoxelMorph prediction...")
    warped_arr, warp_field = reg_model.predict([moving_input, fixed_input], verbose=0)

    warped_2d = np.asarray(warped_arr[0, ..., 0])[crop]
    field = np.asarray(warp_field[0])[crop]

    import ants

    warped_ants = ants.from_numpy(np.ascontiguousarray(warped_2d, dtype=np.float32))
    warped_path = str(output_dir / "stacker_warped.nii.gz")
    _write_ants_image_atomic(warped_ants, warped_path)
    log(f"Saved warped image to {warped_path}")

    warp_path = str(output_dir / "stacker_warp_field.npy")
    _save_npy_atomic(field.astype(np.float32), warp_path)
    log(f"Saved warp field to {warp_path}")

    params["warp_field"] = (
        f"full-resolution displacement in pixels, shape {list(field.shape)} (fixed-image grid x {field.shape[-1]} "
        "components, 'ij' indexing): the field that produced stacker_warped.nii.gz"
    )
    params["model_trained"] = which == "trained"
    params["model_path_used"] = str(model_path) if which == "trained" else None

    if which == "trained":
        method, used_fallback, why = METHOD_VXM_TRAINED, False, ""
    else:
        method = METHOD_VXM_UNTRAINED
        used_fallback = True
        why = "no model_path was given and allow_untrained_model_fallback=True accepted a randomly initialised network"
        max_disp = float(np.abs(field).max()) if field.size else 0.0
        notes.append(
            f"No trained model was supplied: the network had random weights, so the warp is near-identity "
            f"(largest displacement {max_disp:.3g} px) and the warped image is the moving image on the fixed grid, "
            "not a registration."
        )

    info = {
        "method": method,
        "used_fallback": used_fallback,
        "why": why,
        "params": params,
        "data": {"moving_image_read_as": moving_how, "fixed_image_read_as": fixed_how},
        "warnings": warnings_out,
        "analysis": " ".join(notes),
    }
    return warped_2d, [warp_path], moving_shape, fixed_shape, info


# ----------------------------------------------------------------------------- CLI


def main():
    parser = argparse.ArgumentParser(description="Stacker worker: register and align spatial tissue sections.")
    parser.add_argument("--moving-image", required=True, help="Path to moving image")
    parser.add_argument("--fixed-image", required=True, help="Path to fixed (reference) image")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    parser.add_argument(
        "--mode",
        default="affine",
        choices=MODES,
        help="Registration mode: affine, deformable, or synthmorph",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a trained VoxelMorph VxmDense model (.h5). Required for mode=synthmorph.",
    )
    parser.add_argument(
        "--allow-untrained-model-fallback",
        action="store_true",
        default=False,
        help="mode=synthmorph without --model-path: run a randomly initialised network (flagged as a fallback).",
    )

    args = parser.parse_args()

    try:
        out_dir = Path(args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == "affine":
            warped, transform_paths, mov_shape, fix_shape, info = run_affine_registration(
                args.moving_image, args.fixed_image, out_dir
            )
        elif args.mode == "deformable":
            warped, transform_paths, mov_shape, fix_shape, info = run_deformable_registration(
                args.moving_image, args.fixed_image, out_dir
            )
        elif args.mode == "synthmorph":
            warped, transform_paths, mov_shape, fix_shape, info = run_synthmorph_registration(
                args.moving_image,
                args.fixed_image,
                out_dir,
                model_path=args.model_path,
                allow_untrained_model_fallback=args.allow_untrained_model_fallback,
            )
        else:
            raise ValueError(unsupported_choice_msg("mode", args.mode, MODES))

        # Build output
        out = WorkerOutput("stacker", task="spatial_registration")
        out.set_data(
            moving_image=args.moving_image,
            fixed_image=args.fixed_image,
            moving_shape=list(mov_shape) if hasattr(mov_shape, "__iter__") else mov_shape,
            fixed_shape=list(fix_shape) if hasattr(fix_shape, "__iter__") else fix_shape,
            **info["data"],
        )
        out.add_output_files(
            {
                "warped_image": str(out_dir / "stacker_warped.nii.gz"),
                "transforms": transform_paths,
            }
        )
        out.add_params(
            {
                "mode": args.mode,
                "model_path": args.model_path,
                "allow_untrained_model_fallback": bool(args.allow_untrained_model_fallback),
            }
        )
        out.add_params(info["params"])
        record_method(out, info["method"], used_fallback=info["used_fallback"], why=info["why"])
        if args.mode != "synthmorph":
            unused = []
            if args.model_path:
                unused.append("model_path")
            if args.allow_untrained_model_fallback:
                unused.append("allow_untrained_model_fallback")
            record_ignored(out, unused, f"only mode='synthmorph' uses a network; mode='{args.mode}' is ANTsPy")
        out.add_warnings(info["warnings"])
        out.set_summary(
            registration_mode=args.mode,
            method=info["method"],
            n_transforms=len(transform_paths),
        )
        out.set_analysis(
            f"Stacker ran {info['method']} to align the moving image (shape {list(mov_shape)}) to the fixed "
            f"image (shape {list(fix_shape)}). Output warped image and {len(transform_paths)} transform "
            f"file(s) saved." + (f" {info['analysis']}" if info["analysis"] else "")
        )

        # Single JSON line to stdout
        print(json.dumps(out.to_dict(), default=str))
        sys.stdout.flush()

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("stacker", str(e), task="spatial_registration")
        sys.exit(1)


if __name__ == "__main__":
    main()
