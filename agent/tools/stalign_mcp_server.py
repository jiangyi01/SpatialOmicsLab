#!/usr/bin/env python3
"""STalign spatial alignment MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "stalign"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "STALIGN",
    "/opt/conda/envs/stalign/bin/python",
    "/workspace/epic-fermat/agent/tools/stalign_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def stalign_align_points(
    source_csv: str,
    target_csv: str,
    output_dir: str,
    niter: int = 200,
    a: float | None = None,
) -> dict[str, Any]:
    """
    Align two point clouds using STalign LDDMM diffeomorphic registration.

    Rasterizes each point cloud into a density image, then jointly estimates an
    affine transform and a diffeomorphism by matching the two images (LDDMM),
    giving a non-rigid alignment. No landmarks or point correspondences are used:
    the two files may hold different points, in any order. Input CSVs should
    contain (y, x) coordinates (two columns, no header or with columns named y,x).

    Alignment quality is reported without pairing rows of the two files:
    summary.density_overlap_before / _after is the overlap of the two clouds'
    point densities, each smoothed at the target's median point spacing
    (1 = identical, 0 = disjoint). summary.mean_nearest_target_distance_before /
    _after is the mean distance from each source point to its nearest target
    point; on a regular spot lattice (e.g. two Visium sections) it stays within
    about one spot spacing (summary.target_median_point_spacing) however the
    tissue lines up, so read it with the overlap. summary.mean_point_displacement
    says how far the transform moved the points; params.method names what ran.

    LDDMM starts from the identity with STalign's default step sizes and noise
    settings (only niter and a are exposed) and no landmark initialisation, so
    the affine part moves little: offsets of a few percent of the clouds' extent
    are recovered, a large translation or rotation only partly. Bring clouds
    that are far apart into rough register first (e.g. centre each on its
    centroid).

    Parameters
    ----------
    source_csv:
        Path to CSV with source point cloud coordinates (y, x columns).
    target_csv:
        Path to CSV with target point cloud coordinates (y, x columns).
    output_dir:
        Directory where outputs will be written:
          - stalign_aligned_points.csv
          - stalign_transform.npz
          - stalign_overlay.png
    niter:
        Number of LDDMM iterations (default 200; at least 1).
    a:
        LDDMM kernel width: the smoothness scale of the velocity field, in the
        coordinates' units (must be positive). If None, one tenth of the larger
        extent of the two clouds (at least 1).
    """
    args = [
        "--task",
        "align_points",
        "--source-csv",
        source_csv,
        "--target-csv",
        target_csv,
        "--output-dir",
        output_dir,
        "--niter",
        str(niter),
    ]
    if a is not None:
        args.extend(["--a", str(a)])

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def stalign_align_to_image(
    points_csv: str,
    image_path: str,
    output_dir: str,
    niter: int = 200,
) -> dict[str, Any]:
    """
    Align a point cloud to a tissue image using STalign LDDMM registration.

    Rasterizes the point cloud into a density image, then registers it against the
    provided tissue image (converted to grayscale) using diffeomorphic LDDMM. No
    landmarks are used. The image is placed on its own pixel grid (row = y,
    column = x, origin at the top-left pixel), so the point coordinates should be
    in that image's pixel units: scale full-resolution coordinates by the image's
    scale factor first.

    No alignment-quality measure is computed (there is no second point set to
    score against); summary.mean_point_displacement says how far the points
    moved. LDDMM runs with STalign's default step sizes and noise settings
    (a = one tenth of the image's larger side) and no landmark initialisation;
    its affine part barely moves. On a real Visium H&E (2026-09-29, SpinalCord
    lowres image, 200 iterations) points already at the Space Ranger
    registration were moved a mean of 52 px, about 8 spot spacings, and a 35 px
    offset was not recovered. Check stalign_overlay.png before using the
    coordinates.

    Parameters
    ----------
    points_csv:
        Path to CSV with point cloud coordinates (y, x columns).
    image_path:
        Path to a tissue image (PNG, JPEG, or TIFF).
    output_dir:
        Directory where outputs will be written:
          - stalign_aligned_points.csv
          - stalign_transform.npz
          - stalign_overlay.png
    niter:
        Number of LDDMM iterations (default 200; at least 1).
    """
    args = [
        "--task",
        "align_to_image",
        "--points-csv",
        points_csv,
        "--image-path",
        image_path,
        "--output-dir",
        output_dir,
        "--niter",
        str(niter),
    ]

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
