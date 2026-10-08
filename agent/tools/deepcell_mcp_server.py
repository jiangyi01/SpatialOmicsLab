#!/usr/bin/env python3
"""DeepCell cell segmentation MCP wrapper for SpatialOmicsLab.

Wraps the DeepCell Mesmer application (whole-cell and nuclear segmentation of
multiplexed fluorescence images) and NuclearSegmentation 1.0 (nuclei from one
nuclear-marker channel). Model weights are cached from DeepCell's public S3
bucket and loaded from disk, so the DEEPCELL_ACCESS_TOKEN that deepcell >= 0.12.7
asks for is never needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "deepcell_seg"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "DEEPCELL",
    "/opt/conda/envs/deepcell_env/bin/python",
    "/workspace/epic-fermat/agent/tools/deepcell_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_deepcell_segmentation(
    image_path: str,
    output_dir: str = default_output_dir(),
    model_type: str = "mesmer",
    compartment: str = "whole-cell",
    image_mpp: float = 0.5,
    nuclear_channel: int = 0,
    membrane_channel: int = 1,
    image_mpp_is_measured: bool = False,
) -> dict[str, Any]:
    """
    Run DeepCell cell segmentation on a microscopy image.

    ``model_type='mesmer'`` runs DeepCell Mesmer (MultiplexSegmentation-9), a deep learning model
    trained on two-marker fluorescence images of diverse tissues, for whole-cell and/or nuclear
    segmentation. ``model_type='nuclear'`` runs DeepCell NuclearSegmentation version 1.0
    (NuclearSegmentation-75) on the nuclear channel alone. Both models' weights are cached from
    DeepCell's public S3 bucket; no DEEPCELL_ACCESS_TOKEN is needed. ``params.method`` names the
    model that ran.

    Mesmer reads two channels: a nuclear marker (e.g. DAPI) and a membrane/whole-cell marker
    (e.g. Na+K+ATPase, E-cadherin). ``nuclear_channel`` and ``membrane_channel`` pick them out of a
    multi-channel image (channel-first stacks are transposed first; the transpose is reported).
    Left at their defaults they take channels 0 and 1 -- on an RGB composite that is red and green,
    which is usually not the DAPI channel -- and the warning lists every channel's mean intensity.
    A single-channel image is used as both markers, which makes whole-cell boundaries nuclear
    boundaries; ask for compartment='nuclear' on such input. Mesmer was trained on fluorescence, not
    brightfield: an H&E image's channels are not nuclear/membrane markers.

    Parameters
    ----------
    image_path:
        Path to the input image file (TIFF via tifffile; PNG/JPEG and other formats via
        skimage). Read at full resolution: images over Pillow's decompression-bomb limit
        (whole-slide JPEGs) are read whole, and the lifted limit is reported.
    output_dir:
        Directory to save segmentation results (masks as .npy and .tif, overlay PNG).
    model_type:
        'mesmer' (whole-cell + nuclear, recommended) or 'nuclear' (NuclearSegmentation 1.0 on
        ``nuclear_channel``; ``compartment`` and ``membrane_channel`` do not apply and are listed in
        ``params.ignored`` when set).
    compartment:
        Segmentation compartment (mesmer only): 'whole-cell', 'nuclear', or 'both'.
    image_mpp:
        Microns per pixel of the image. DeepCell rescales the image by image_mpp / model_mpp
        (Mesmer 0.5, NuclearSegmentation 0.65) before segmenting. The default 0.5 is Mesmer's
        training resolution, not a measurement of your image: a run at 0.5 is reported as
        ``params.image_mpp_source='assumed default'``, with a warning, and cell areas are given in
        pixels only, unless ``image_mpp_is_measured`` is True. Any other value is taken as yours and
        also gives areas in um2. For a Visium ``tissue_hires_image``/``tissue_lowres_image`` beside
        its ``scalefactors_json.json`` the implied value is reported in
        ``params.image_mpp_implied_by_scalefactors`` (it is not applied for you), with its basis in
        ``params.image_mpp_implied_basis``: ``microns_per_pixel`` when present, else the 100 um
        spot pitch measured from the ``tissue_positions`` file beside it, else (approximately; about
        8% off on CytAssist slides) ``spot_diameter_fullres`` as a 65 um spot.
    nuclear_channel:
        Index of the nuclear-marker channel in a multi-channel image. Default 0.
    membrane_channel:
        Index of the membrane/whole-cell-marker channel (mesmer only). Default 1. Equal to
        ``nuclear_channel`` uses that one channel as both markers (reported).
    image_mpp_is_measured:
        True when ``image_mpp`` is the image's measured pixel size. Needed only when that measurement
        is 0.5, the default, which the value alone cannot tell from the default: the run is then
        reported as ``params.image_mpp_source='caller'`` with areas in um2. Default False.
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
        "--compartment",
        compartment,
        "--image-mpp",
        str(image_mpp),
        "--nuclear-channel",
        str(nuclear_channel),
        "--membrane-channel",
        str(membrane_channel),
    ]
    if image_mpp_is_measured:
        args.append("--image-mpp-is-measured")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
