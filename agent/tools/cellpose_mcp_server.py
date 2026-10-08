#!/usr/bin/env python3
"""Cellpose cell segmentation MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "cellpose_seg"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "CELLPOSE",
    "/opt/conda/envs/cellpose_env/bin/python",
    "/workspace/epic-fermat/agent/tools/cellpose_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_cellpose_segmentation(
    image_path: str,
    output_dir: str = default_output_dir(),
    model_type: str = "cpsam",
    diameter: float = 30.0,
    channels: str = "[0,0]",
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
) -> dict[str, Any]:
    """
    Run Cellpose cell segmentation on a microscopy image.

    Runs the cellpose 4.x ``CellposeModel`` with its one shipped model, ``cpsam`` (Cellpose-SAM), on
    CPU with float32 weights (upstream's bfloat16 default is broken on CPU), to segment cells in
    fluorescence or brightfield images at full resolution. Outputs
    segmentation masks, cell outlines, an overlay figure and cell count/area statistics. The payload
    names the model that ran (``params.model``), the cellpose version and the rescale factor applied.

    Parameters
    ----------
    image_path:
        Path to the input image file (TIFF via tifffile; PNG, JPG and other formats via imageio).
        The image is read and segmented at full resolution, never downscaled; Pillow's
        decompression-bomb limit is lifted for this file and a warning says so. A run whose
        estimated memory exceeds what is available stops before loading, with the numbers.
    output_dir:
        Directory to save segmentation results (masks, outlines, overlay).
    model_type:
        Kept for compatibility. Cellpose 4.x ships exactly one model, 'cpsam', and selects nothing
        by this value: 'cyto3', 'nuclei', 'cyto2' or a model path do not change the result. Any value
        other than 'cpsam' is listed under params.ignored with a warning.
    diameter:
        Typical cell diameter in pixels. The image is rescaled by 30/diameter before the network
        (cpsam is trained at 30 px). 0 means no rescaling -- identical to 30; cellpose 4.x has no
        size model, so nothing is estimated. Negative values are refused.
    channels:
        Kept for compatibility. Cellpose 4.x takes no channel selection: cpsam reads the image's
        first 3 channels as given. The value is echoed and listed under params.ignored.
    flow_threshold:
        Flow error threshold for mask filtering (higher = more permissive).
    cellprob_threshold:
        Cell probability threshold (higher = fewer cells, more confident).
    """
    img = str(Path(image_path).expanduser())
    out = str(Path(output_dir).expanduser())

    args = [
        "--image-path",
        img,
        "--output-dir",
        out,
        "--model-type",
        model_type,
        "--diameter",
        str(diameter),
        "--channels",
        channels,
        "--flow-threshold",
        str(flow_threshold),
        "--cellprob-threshold",
        str(cellprob_threshold),
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
