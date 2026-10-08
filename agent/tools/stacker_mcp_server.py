#!/usr/bin/env python3
"""Stacker spatial tissue section registration MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "stacker"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "STACKER",
    "/opt/conda/envs/stacker/bin/python",
    "/workspace/epic-fermat/agent/tools/stacker_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def stacker_register(
    moving_image_path: str,
    fixed_image_path: str,
    output_dir: str,
    mode: str = "affine",
    model_path: str | None = None,
    allow_untrained_model_fallback: bool = False,
) -> dict[str, Any]:
    """
    Register one tissue section image to another (image-to-image registration).

    No upstream "Stacker" package runs: 'affine' and 'deformable' call ANTsPy's
    ants.registration directly, and 'synthmorph' runs a VoxelMorph VxmDense network
    whose trained weights you supply. Produces the moving image warped into the fixed
    image's grid and the estimated transform(s). The payload names what ran in
    params.method (and params.used_fallback).

    Parameters
    ----------
    moving_image_path:
        Path to the moving image file (PNG, JPEG, TIFF, or NIfTI). This image will
        be warped to align with the fixed image. Colour images are converted to
        8-bit grayscale; single-channel 16-bit/32-bit images are read at their full
        range. An h5ad is not an image and is not accepted.
    fixed_image_path:
        Path to the fixed (reference) image file (same formats). The moving image
        will be aligned to match this image's coordinate frame.
    output_dir:
        Directory where outputs will be written:
          - stacker_warped.nii.gz  (aligned moving image, every mode)
          - stacker_transform_<i>.<ext>  ('affine' and 'deformable': the ANTs forward
            transforms in order -- 'affine' writes stacker_transform_0.mat; 'deformable'
            writes the SyN warp stacker_transform_0.nii.gz and the affine
            stacker_transform_1.mat)
          - stacker_warp_field.npy  ('synthmorph': the full-resolution displacement
            field in pixels, fixed-image shape x 2, that produced the warped image)
    mode:
        Registration mode. One of:
          - 'affine': ANTsPy affine registration (fast, rigid + scaling)
          - 'deformable': ANTsPy SyN deformable registration (non-rigid)
          - 'synthmorph': VoxelMorph VxmDense deep-learning registration with the
            trained network in model_path. The moving image is resampled onto the
            fixed image's shape, and both are zero-padded to the multiple the
            network's U-Net needs (16 for the default architecture); outputs are
            cropped back to the fixed shape.
    model_path:
        Path to a trained 2-D VoxelMorph VxmDense model (.h5), for example one
        trained with SynthMorph. Only used when mode='synthmorph', and required
        there: no weights ship with this tool. A path that does not exist is an
        error. Ignored (and listed in params.ignored) in the ANTsPy modes.
    allow_untrained_model_fallback:
        mode='synthmorph' without model_path stops with an error by default.
        True instead runs a randomly initialised VxmDense -- a near-identity warp,
        not a registration -- and the payload says so (params.used_fallback=true,
        params.model_trained=false). Ignored in the ANTsPy modes.
    """
    args = [
        "--moving-image",
        moving_image_path,
        "--fixed-image",
        fixed_image_path,
        "--output-dir",
        output_dir,
        "--mode",
        mode,
    ]
    if model_path is not None and model_path != "":
        args.extend(["--model-path", model_path])
    if allow_untrained_model_fallback:
        args.append("--allow-untrained-model-fallback")

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
