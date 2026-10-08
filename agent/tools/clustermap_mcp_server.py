#!/usr/bin/env python3
"""ClusterMap cell segmentation MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "clustermap"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "CLUSTERMAP",
    "/opt/conda/envs/clustermap_env/bin/python",
    "/workspace/epic-fermat/agent/tools/clustermap_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_clustermap(
    spots_csv: str,
    output_dir: str = default_output_dir(),
    xy_radius: float = 1.0,
    z_radius: float = 1.0,
    num_dims: int = 2,
    min_spots: int = 5,
    dapi_image_path: str = "",
) -> dict[str, Any]:
    """
    Run ClusterMap cell segmentation on spatial transcriptomics spot data.

    ClusterMap segments cells from single-molecule FISH or spatial
    transcriptomics data using density peak clustering on gene-weighted
    spatial coordinates. Optionally integrates a DAPI nuclear stain image
    for improved accuracy.

    Parameters
    ----------
    spots_csv:
        Path to a transcript-level table, one row per molecule, with columns
        'spot_location_1', 'spot_location_2' and 'gene' (gene names or integer
        IDs; they are re-encoded to contiguous codes internally, and the outputs
        keep the genes as supplied). 'spot_location_3' is required for 3D data.
    output_dir:
        Directory to save segmentation results.
    xy_radius:
        Radius for XY neighborhood in the clustering (in coordinate units).
    z_radius:
        Radius for Z neighborhood (for 3D data). ClusterMap's density-peak step
        searches within max(xy_radius, z_radius) even in 2D, so a z_radius
        larger than xy_radius widens a 2D search (reported as a warning).
    num_dims:
        Number of spatial dimensions: 2 or 3.
    min_spots:
        Smallest cell kept: cells with fewer than min_spots assigned spots are
        erased after segmentation. (ClusterMap's own erase step drops cells of
        size <= its threshold, so it is handed min_spots - 1.)
    dapi_image_path:
        Optional path to a single-channel DAPI image (TIFF): (y, x) for 2D, or
        (y, x, z) with z last for 3D. ClusterMap reads it at pixel
        (spot_location_2 - 1, spot_location_1 - 1), so with an image the
        coordinates must be whole 1-based pixel indices inside it; anything else
        is refused, never rounded. Without an image, nucleus seeds come from a
        synthetic all-ones placeholder sampled as a uniform lattice, and the
        coordinates are rounded (and translated if any axis reaches below 1)
        onto ClusterMap's 1-based pixel grid; the payload reports both.

    Returns
    -------
    dict
        Worker payload. ``clustermap_cell_assignments.csv`` has one row per input
        molecule in the caller's terms: spot_location_1, spot_location_2, gene (as
        supplied), clustermap (cell id, -1 = unassigned), spot_location_3 for 3D,
        molecule_index (input row) and gene_code (ClusterMap's integer code).
        ``clustermap_segmentation.csv`` is ClusterMap's own frame (grid coordinates,
        gene codes, is_noise, clustermap) plus molecule_index and gene_name.
        ``params.method`` says whether a DAPI image or the synthetic lattice seeded
        the nuclei.
    """
    csv_path = str(Path(spots_csv).expanduser())
    out = str(Path(output_dir).expanduser())

    args = [
        "--spots-csv",
        csv_path,
        "--output-dir",
        out,
        "--xy-radius",
        str(xy_radius),
        "--z-radius",
        str(z_radius),
        "--num-dims",
        str(num_dims),
        "--min-spots",
        str(min_spots),
    ]
    if dapi_image_path:
        args.extend(["--dapi-image-path", str(Path(dapi_image_path).expanduser())])

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
