#!/usr/bin/env python
"""
STalign worker: spatial alignment via LDDMM diffeomorphic registration.

- Runs inside /opt/conda/envs/stalign
- All logs/progress go to stderr.
- stdout is reserved for a single final JSON line.

Tasks:
  - align_points:    align two point clouds (source -> target)
  - align_to_image:  align a point cloud to a tissue image
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

# Ensure worker_utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worker_utils import WorkerOutput, record_method, sniff_tabular_sep

# Force non-interactive matplotlib backend before any plotting imports
os.environ["MPLBACKEND"] = "Agg"

import numpy as np

try:
    from STalign import STalign as st
except Exception as e:
    print("[stalign-worker] ERROR: Failed to import STalign.", file=sys.stderr)
    print(str(e), file=sys.stderr)
    WorkerOutput.emit_error("stalign", "Failed to import STalign: " + str(e), task="import")
    sys.exit(1)


# What every image handed to LDDMM went through; published in params so a run says how it was scaled.
INTENSITY_SCALING = "min-max to [0, 1] per image before LDDMM (as STalign.normalize in the upstream tutorials)"


def _log(msg):
    print(f"[stalign-worker] {msg}", file=sys.stderr)


def _first_row_names_the_columns(row):
    """True when row 0 holds names rather than coordinates, so it is a header and not a landmark."""
    for value in list(row)[:2]:
        try:
            float(value)
        except (TypeError, ValueError):
            return True
    return False


def _load_points_csv(path):
    """Load a (y, x) coordinate CSV. Accepts header or headerless 2-column files."""
    import pandas as pd

    # Both defaults of a bare read_csv are wrong for the file this advertises. header="infer"
    # promoted the first landmark to column names, so the headerless input offered by the docstring
    # above and by stalign_mcp_server.py:33 lost its first point in silence -- LDDMM fits the
    # remaining ones happily and the tool then reports the smaller number as the count it aligned.
    # And the comma turned any tab-delimited export into one column, raising "found 1" about a file
    # with two. Read the first row, then decide from the row itself what it is.
    sep = sniff_tabular_sep(path)
    df = pd.read_csv(path, sep=sep, header=None)
    if _first_row_names_the_columns(df.iloc[0]):
        df = pd.read_csv(path, sep=sep, header=0)

    if df.shape[1] < 2:
        raise ValueError(
            f"CSV {path} must have at least 2 columns (y, x); found {df.shape[1]} reading it with separator {sep!r}"
        )

    # Try to detect named columns. A headerless file now keeps pandas' integer labels, which have
    # no .lower(); str() leaves them as "0"/"1", so the search below simply finds nothing and the
    # positional branch takes over, which is what a headerless file wants.
    cols_lower = [str(c).lower().strip() for c in df.columns]
    if "y" in cols_lower and "x" in cols_lower:
        yi = cols_lower.index("y")
        xi = cols_lower.index("x")
        y = df.iloc[:, yi].values.astype(float)
        x = df.iloc[:, xi].values.astype(float)
    else:
        # Assume first two columns are y, x
        y = df.iloc[:, 0].values.astype(float)
        x = df.iloc[:, 1].values.astype(float)
    return y, x


def _auto_a(y_src, x_src, y_tgt, x_tgt):
    """Estimate LDDMM 'a' parameter from the data extent."""
    all_y = np.concatenate([y_src, y_tgt])
    all_x = np.concatenate([x_src, x_tgt])
    extent = max(all_y.max() - all_y.min(), all_x.max() - all_x.min())
    # Heuristic: a ~ extent / 10, clamped to reasonable range
    a = max(extent / 10.0, 1.0)
    _log(f"Auto-computed a={a:.2f} from data extent={extent:.2f}")
    return a


def _require_usable_points(y, x, path):
    """Refuse a point cloud STalign cannot rasterize, naming the file and the rows.

    A blank cell reads as NaN, and NaN reaches ``np.arange(nan, nan, dx)`` inside
    ``STalign.rasterize`` as "arange: cannot compute length" -- a message about neither the file nor
    the row. A header-only file or a cloud with no spread along one axis gives a raster with no
    pixels along that axis and fails deeper still, in LDDMM.
    """
    n = len(y)
    if n == 0:
        raise ValueError(f"{path} holds no points; STalign needs a point cloud to rasterize")
    bad = ~(np.isfinite(y) & np.isfinite(x))
    n_bad = int(bad.sum())
    if n_bad:
        rows = [int(i) for i in np.flatnonzero(bad)[:10]]
        more = " ..." if n_bad > 10 else ""
        raise ValueError(
            f"{path}: {n_bad} of {n} points have a missing or non-finite coordinate (0-based data rows "
            f"{rows}{more}). A point with no position cannot be aligned; fill in those coordinates."
        )
    for axis, values in (("y", y), ("x", x)):
        if float(np.max(values) - np.min(values)) <= 0:
            raise ValueError(
                f"{path}: all {n} points share the same {axis} ({float(values[0])}), so the cloud has no "
                f"extent along {axis} and cannot be rasterized into an image for LDDMM"
            )


def _rasterize_rows_cols(x, y, dx, what):
    """Rasterize a point cloud; return ``([row_axis, col_axis], image)`` in the order LDDMM reads.

    ``STalign.rasterize(x, y)`` returns ``(X, Y, W[, fig])`` -- the x (column) axis FIRST, then the y
    (row) axis -- with ``W`` shaped ``(channels, len(Y), len(X))``. ``STalign.LDDMM`` takes each
    image's pixel grid in row-column order, ``[Y, X]``; the upstream tutorials pass ``xI = [YI, XI]``.
    This worker used to read the first two returns as ``(y, x)`` and so handed LDDMM a transposed
    grid: on an L-shaped cloud translated by (+80, +150), 200 iterations left a mean error of 130 of
    the initial 170, against 4.6 with the grid the right way round. The shape check below makes the
    pairing something the run verifies rather than assumes.
    """
    ret = st.rasterize(x, y, dx=dx)
    x_axis, y_axis, image = ret[0], ret[1], np.asarray(ret[2])
    if len(ret) > 3 and ret[3] is not None:
        import matplotlib.pyplot as plt

        plt.close(ret[3])
    if image.ndim != 3 or image.shape[1] != len(y_axis) or image.shape[2] != len(x_axis):
        raise RuntimeError(
            f"STalign.rasterize returned an image of shape {tuple(image.shape)} with a first axis of "
            f"length {len(x_axis)} and a second of length {len(y_axis)}; this worker expects (X, Y, image) "
            "with the image shaped (channels, len(Y), len(X)). The installed STalign does not match, so "
            "the grid handed to LDDMM would be mislabelled."
        )
    if len(y_axis) < 2 or len(x_axis) < 2:
        raise ValueError(
            f"{what}: at pixel size dx={dx:.4g} the rasterized cloud is {len(y_axis)} x {len(x_axis)} pixels "
            "(rows x columns); LDDMM needs at least 2 along each axis. The cloud is too thin along one axis "
            "relative to the other to be registered as an image."
        )
    return [y_axis, x_axis], image


def _to_unit_range(image, what):
    """Min-max scale an image to [0, 1], as STalign's own tutorials do before LDDMM.

    The upstream Xenium-to-H&E notebook calls ``STalign.normalize`` on both the image and the point
    raster before ``LDDMM``. LDDMM's matching term and its fixed step sizes are in the images'
    intensity units, so the scale decides whether a run converges at all, and this worker passed
    both through raw. ``matplotlib.image.imread`` returns a PNG as floats in [0, 1] but a JPEG or an
    8-bit TIFF as integers in [0, 255]; with a 0-255 target the affine ran off by tens of thousands of
    pixels. A point raster's scale is its local point density, so a sparser cloud barely moved: on
    a 1,500-point L-shape translated by (+80, +150), 200 iterations left a mean error of 76 raw and
    4.7 scaled (4.6 and 4.3 at 3,000 points). Unlike ``STalign.normalize``, a uniform image is refused
    by name instead of becoming NaN.
    """
    image = np.asarray(image, dtype=float)
    lo = float(np.min(image))
    hi = float(np.max(image))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"{what} contains non-finite pixel values; LDDMM cannot register it")
    if hi <= lo:
        raise ValueError(f"{what} is uniform (every pixel is {lo:g}); there is no structure to register")
    return (image - lo) / (hi - lo)


def _mean_nearest_distance(y_from, x_from, y_to, x_to):
    """Mean distance from each ``from`` point to its nearest ``to`` point.

    The two inputs are two sections' point clouds. Nothing says row i of one is row i of the other,
    and an equal row count does not make it so -- any two samples of one platform can have the same
    number of spots. The residual this worker used to report paired row i with row i whenever the
    counts matched, which measured the files' row order, not the alignment. A nearest-point distance
    needs no correspondence; compared before and after the transform, it says how much closer the
    clouds were brought.
    """
    from scipy.spatial import cKDTree

    to_pts = np.column_stack([np.asarray(y_to, dtype=float), np.asarray(x_to, dtype=float)])
    from_pts = np.column_stack([np.asarray(y_from, dtype=float), np.asarray(x_from, dtype=float)])
    dist, _ = cKDTree(to_pts).query(from_pts, k=1)
    return float(np.mean(dist))


def _median_point_spacing(y, x):
    """Median distance from each point to its nearest other point: the cloud's own sampling scale."""
    from scipy.spatial import cKDTree

    pts = np.column_stack([np.asarray(y, dtype=float), np.asarray(x, dtype=float)])
    if len(pts) < 2:
        return 0.0
    dist, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(dist[:, 1]))


def _density_overlap(y_from, x_from, y_to, x_to, frame, dx, sigma_px):
    """Overlap of two clouds' smoothed point densities on one fixed grid, from 0 (disjoint) to 1 (identical).

    Each cloud is binned on the same ``dx`` grid over ``frame`` (fixed by the INPUT clouds, so a
    transform that throws points far away cannot grow the grid), smoothed by a Gaussian of
    ``sigma_px`` pixels, and divided by its own point count; the score is the sum of the pixel-wise
    minimum (histogram intersection). A point outside the frame still counts in its cloud's total,
    so points pushed out of the frame lower the score instead of vanishing from it.

    The nearest-target distance alone cannot say whether an alignment helped on a spot lattice: two
    Visium sections sit on the same hexagonal grid, so every source spot is already within about one
    spot spacing of some target spot, and a non-rigid move that lines the tissue up better takes spots
    off that grid. On V1_Human_Brain_Section_1 -> _2 it rose from 112 to 131 while this overlap rose
    from 0.956 to 0.968; with the pre-fix transposed grid, which did not move the cloud, both stayed put.
    Smoothing at the target's own point spacing makes a lattice read as the tissue it samples.
    """
    from scipy.ndimage import gaussian_filter

    y0, y1, x0, x1 = frame
    shape = (int(np.ceil((y1 - y0) / dx)) + 1, int(np.ceil((x1 - x0) / dx)) + 1)

    def _density(y, x):
        rows = (np.asarray(y, dtype=float) - y0) / dx
        cols = (np.asarray(x, dtype=float) - x0) / dx
        inside = np.isfinite(rows) & np.isfinite(cols)
        inside &= (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
        grid = np.zeros(shape)
        np.add.at(grid, (rows[inside].astype(int), cols[inside].astype(int)), 1.0)
        return gaussian_filter(grid, sigma_px, mode="constant") / float(len(rows))

    return float(np.minimum(_density(y_from, x_from), _density(y_to, x_to)).sum())


def _displacement(y_before, x_before, y_after, x_after):
    """(mean, max) distance each point moved. Row i before and after is the same point, so this pairing is valid."""
    step = np.hypot(np.asarray(y_after, dtype=float) - y_before, np.asarray(x_after, dtype=float) - x_before)
    return float(np.mean(step)), float(np.max(step))


def _require_finite_transform(y_aligned, x_aligned, what):
    """Refuse a transform that sent points to NaN/inf instead of publishing them as an alignment."""
    bad = ~(np.isfinite(y_aligned) & np.isfinite(x_aligned))
    n_bad = int(bad.sum())
    if n_bad:
        raise RuntimeError(
            f"STalign LDDMM diverged: {n_bad} of {len(y_aligned)} {what} have a non-finite position after the "
            "transform, so there is no alignment to write. The settings that shape the optimisation here are "
            "niter and, for align_points, a (the velocity-field smoothness scale)."
        )


def _require_run_settings(niter, a_param):
    """Reject settings LDDMM cannot use before any file is read."""
    if int(niter) < 1:
        raise ValueError(
            f"niter must be at least 1 (got {niter}); with no iterations STalign.LDDMM never builds a "
            "transform and fails returning it"
        )
    if a_param is not None and not float(a_param) > 0:
        raise ValueError(
            f"a must be a positive length in the coordinates' units (got {a_param}); it sets the "
            "velocity-field grid spacing (a/2), so zero or a negative value leaves no grid to build"
        )


def run_align_points(source_csv, target_csv, output_dir, niter, a_param):
    """Align source point cloud to target point cloud via LDDMM."""
    _log("Task: align_points")
    _log(f"source_csv = {source_csv}")
    _log(f"target_csv = {target_csv}")
    _log(f"output_dir = {output_dir}")
    _log(f"niter = {niter}, a = {a_param}")

    _require_run_settings(niter, a_param)
    out_dir = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load point clouds
    ys, xs = _load_points_csv(source_csv)
    yt, xt = _load_points_csv(target_csv)
    _require_usable_points(ys, xs, source_csv)
    _require_usable_points(yt, xt, target_csv)
    _log(f"Source points: {len(ys)}, Target points: {len(yt)}")

    # Determine 'a' parameter
    a_val = a_param if a_param is not None else _auto_a(ys, xs, yt, xt)

    # Rasterize both point clouds to images for LDDMM
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _log("Rasterizing source point cloud...")
    extent_y = max(ys.max(), yt.max()) - min(ys.min(), yt.min())
    extent_x = max(xs.max(), xt.max()) - min(xs.min(), xt.min())
    dx = max(extent_y, extent_x) / 200.0  # resolution: ~200 pixels across
    if dx <= 0:
        dx = 1.0

    # rasterize(x, y, ...) returns (X_axis, Y_axis, image, fig); LDDMM wants the grid as [Y, X].
    grid_s, I_s = _rasterize_rows_cols(xs, ys, dx, "source_csv")
    grid_t, I_t = _rasterize_rows_cols(xt, yt, dx, "target_csv")
    I_s = _to_unit_range(I_s, f"the rasterized source cloud ({source_csv})")
    I_t = _to_unit_range(I_t, f"the rasterized target cloud ({target_csv})")

    _log("Running LDDMM registration...")
    result = st.LDDMM(
        grid_s,
        I_s,
        grid_t,
        I_t,
        niter=niter,
        a=a_val,
    )

    A = result["A"]
    v = result["v"]
    xv = result["xv"]

    # Transform source points
    _log("Transforming source points to target space...")
    points_src = np.stack([ys, xs], axis=1)
    points_transformed = st.transform_points_source_to_target(xv, v, A, points_src)
    # Convert from torch tensor to numpy if needed
    import torch as _torch_pt

    if isinstance(points_transformed, _torch_pt.Tensor):
        points_transformed = points_transformed.detach().cpu().numpy()
    y_aligned = np.array(points_transformed[:, 0], dtype=float)
    x_aligned = np.array(points_transformed[:, 1], dtype=float)
    _require_finite_transform(y_aligned, x_aligned, "source points")

    # Save aligned points
    import pandas as pd

    aligned_csv = os.path.join(out_dir, "stalign_aligned_points.csv")
    pd.DataFrame({"y": y_aligned, "x": x_aligned}).to_csv(aligned_csv, index=False)
    _log(f"Saved aligned points to {aligned_csv}")

    # Save transform (convert torch tensors to numpy for saving)
    import torch as _torch

    def _to_np(t):
        if isinstance(t, _torch.Tensor):
            return t.detach().cpu().numpy()
        return np.array(t)

    transform_path = os.path.join(out_dir, "stalign_transform.npz")
    np.savez(
        transform_path,
        A=_to_np(A),
        v=_to_np(v),
        xv0=_to_np(xv[0]),
        xv1=_to_np(xv[1]),
    )
    _log(f"Saved transform to {transform_path}")

    # Save overlay plot
    overlay_path = os.path.join(out_dir, "stalign_overlay.png")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # Before alignment
        axes[0].scatter(xs, ys, s=1, c="blue", alpha=0.5, label="source")
        axes[0].scatter(xt, yt, s=1, c="red", alpha=0.5, label="target")
        axes[0].set_title("Before alignment")
        axes[0].legend(markerscale=5)
        axes[0].invert_yaxis()
        # After alignment
        axes[1].scatter(x_aligned, y_aligned, s=1, c="blue", alpha=0.5, label="aligned source")
        axes[1].scatter(xt, yt, s=1, c="red", alpha=0.5, label="target")
        axes[1].set_title("After alignment")
        axes[1].legend(markerscale=5)
        axes[1].invert_yaxis()
        # Displacement field
        axes[2].quiver(
            xs[::5],
            ys[::5],
            (x_aligned - xs)[::5],
            (y_aligned - ys)[::5],
            scale_units="xy",
            scale=1,
            alpha=0.5,
        )
        axes[2].set_title("Displacement vectors (subsampled)")
        axes[2].invert_yaxis()

        plt.tight_layout()
        fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _log(f"Saved overlay plot to {overlay_path}")
    except Exception as plot_err:
        _log(f"Warning: overlay plot failed: {plot_err}")
        overlay_path = None

    # Alignment quality without assuming row i of one file is row i of the other.
    nn_before = _mean_nearest_distance(ys, xs, yt, xt)
    nn_after = _mean_nearest_distance(y_aligned, x_aligned, yt, xt)
    target_spacing = _median_point_spacing(yt, xt)
    lo_y, hi_y = min(ys.min(), yt.min()), max(ys.max(), yt.max())
    lo_x, hi_x = min(xs.min(), xt.min()), max(xs.max(), xt.max())
    margin = 0.1 * max(hi_y - lo_y, hi_x - lo_x)
    frame = (lo_y - margin, hi_y + margin, lo_x - margin, hi_x + margin)
    sigma_px = max(1.0, target_spacing / dx)
    overlap_before = _density_overlap(ys, xs, yt, xt, frame, dx, sigma_px)
    overlap_after = _density_overlap(y_aligned, x_aligned, yt, xt, frame, dx, sigma_px)
    moved_mean, moved_max = _displacement(ys, xs, y_aligned, x_aligned)

    out = WorkerOutput("stalign", task="align_points")
    out.set_data(
        n_source_points=int(len(ys)),
        n_target_points=int(len(yt)),
    )
    out.add_output_files(
        {
            "aligned_points_csv": aligned_csv,
            "transform_npz": transform_path,
            "overlay_png": str(overlay_path) if overlay_path else None,
        }
    )
    out.add_params(
        {
            "source_csv": source_csv,
            "target_csv": target_csv,
            "niter": niter,
            "a": a_val,
            "dx": dx,
            "intensity_scaling": INTENSITY_SCALING,
            "density_overlap_smoothing_sigma": float(sigma_px * dx),
        }
    )
    record_method(
        out,
        "STalign LDDMM: affine + diffeomorphism fitted by matching the two clouds' rasterized density images (no landmarks or point correspondences)",
    )
    if overlap_after < overlap_before:
        out.add_warning(
            f"the transform lowered the overlap of the source's point density with the target's from "
            f"{overlap_before:.3f} to {overlap_after:.3f}; LDDMM did not bring these clouds closer"
        )
    summary_dict = {
        "source_extent_y": float(ys.max() - ys.min()),
        "source_extent_x": float(xs.max() - xs.min()),
        "target_extent_y": float(yt.max() - yt.min()),
        "target_extent_x": float(xt.max() - xt.min()),
        "density_overlap_before": overlap_before,
        "density_overlap_after": overlap_after,
        "mean_nearest_target_distance_before": nn_before,
        "mean_nearest_target_distance_after": nn_after,
        "target_median_point_spacing": target_spacing,
        "mean_point_displacement": moved_mean,
        "max_point_displacement": moved_max,
    }
    out.set_summary(**summary_dict)
    out.set_analysis(
        f"STalign LDDMM aligned {len(ys)} source points to {len(yt)} target points "
        f"over {niter} iterations (a={a_val:.2f}), fitting an affine transform and a diffeomorphism "
        "by matching the two clouds' rasterized density images; no landmarks were used. "
        f"Overlap of the two clouds' smoothed point densities (1 = identical, 0 = disjoint): "
        f"{overlap_before:.3f} before, {overlap_after:.3f} after. "
        f"Mean distance from each source point to its nearest target point: {nn_before:.2f} before, "
        f"{nn_after:.2f} after, against a median spacing of {target_spacing:.2f} between the target's own "
        "points; on a regular spot lattice this distance stays within about one spacing however the tissue "
        "lines up. No row-to-row correspondence between the two files is assumed. "
        f"The transform moved each source point by a mean of {moved_mean:.2f} (max {moved_max:.2f})."
    )
    return out.to_dict()


def run_align_to_image(points_csv, image_path, output_dir, niter):
    """Align a point cloud to a tissue image via LDDMM."""
    _log("Task: align_to_image")
    _log(f"points_csv = {points_csv}")
    _log(f"image_path = {image_path}")
    _log(f"output_dir = {output_dir}")
    _log(f"niter = {niter}")

    _require_run_settings(niter, None)
    out_dir = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load point cloud
    yp, xp = _load_points_csv(points_csv)
    _require_usable_points(yp, xp, points_csv)
    _log(f"Loaded {len(yp)} points from {points_csv}")

    # Load tissue image
    from matplotlib.image import imread

    img = imread(image_path)
    _log(f"Loaded image {image_path}: shape={img.shape}, dtype={img.dtype}")

    # Convert to grayscale if RGB, then to [0, 1] whatever the file format's integer range was.
    if img.ndim == 3:
        img_gray = np.mean(np.asarray(img[:, :, :3], dtype=float), axis=2)
    else:
        img_gray = img.astype(float)
    img_gray = _to_unit_range(img_gray, image_path)

    # Build coordinate arrays for the image (pixel grid)
    ny, nx = img_gray.shape
    yI = np.linspace(0, ny - 1, ny)
    xI = np.linspace(0, nx - 1, nx)

    # Rasterize point cloud
    extent_y = max(yp.max(), ny) - min(yp.min(), 0)
    extent_x = max(xp.max(), nx) - min(xp.min(), 0)
    dx = max(extent_y, extent_x) / 200.0
    if dx <= 0:
        dx = 1.0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _log("Rasterizing point cloud...")
    # rasterize(x, y, ...) returns (X_axis, Y_axis, image, fig); LDDMM wants the grid as [Y, X].
    grid_points, J = _rasterize_rows_cols(xp, yp, dx, "points_csv")
    J = _to_unit_range(J, f"the rasterized point cloud ({points_csv})")

    # Prepare target image with channel dimension for LDDMM
    img_target = img_gray[np.newaxis, :, :]  # (1, ny, nx)

    # Estimate a from image size
    a_val = max(ny, nx) / 10.0
    _log(f"Using a={a_val:.2f} (based on image size {ny}x{nx})")

    _log("Running LDDMM registration (points -> image)...")
    result = st.LDDMM(
        grid_points,
        J,
        [yI, xI],
        img_target,
        niter=niter,
        a=a_val,
    )

    A = result["A"]
    v = result["v"]
    xv = result["xv"]

    # Transform points
    _log("Transforming points to image coordinate space...")
    points_src = np.stack([yp, xp], axis=1)
    points_transformed = st.transform_points_source_to_target(xv, v, A, points_src)
    # Convert from torch tensor to numpy if needed
    import torch as _torch_img

    if isinstance(points_transformed, _torch_img.Tensor):
        points_transformed = points_transformed.detach().cpu().numpy()
    y_aligned = np.array(points_transformed[:, 0], dtype=float)
    x_aligned = np.array(points_transformed[:, 1], dtype=float)
    _require_finite_transform(y_aligned, x_aligned, "points")

    # Save aligned points
    import pandas as pd

    aligned_csv = os.path.join(out_dir, "stalign_aligned_points.csv")
    pd.DataFrame({"y": y_aligned, "x": x_aligned}).to_csv(aligned_csv, index=False)
    _log(f"Saved aligned points to {aligned_csv}")

    # Save transform (convert torch tensors to numpy for saving)
    import torch as _torch2

    def _to_np2(t):
        if isinstance(t, _torch2.Tensor):
            return t.detach().cpu().numpy()
        return np.array(t)

    transform_path = os.path.join(out_dir, "stalign_transform.npz")
    np.savez(
        transform_path,
        A=_to_np2(A),
        v=_to_np2(v),
        xv0=_to_np2(xv[0]),
        xv1=_to_np2(xv[1]),
    )
    _log(f"Saved transform to {transform_path}")

    # Save overlay plot
    overlay_path = os.path.join(out_dir, "stalign_overlay.png")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        # Before
        axes[0].imshow(img_gray, cmap="gray", origin="upper")
        axes[0].scatter(xp, yp, s=1, c="red", alpha=0.4, label="points (before)")
        axes[0].set_title("Before alignment")
        axes[0].legend(markerscale=5)
        # After
        axes[1].imshow(img_gray, cmap="gray", origin="upper")
        axes[1].scatter(x_aligned, y_aligned, s=1, c="lime", alpha=0.4, label="points (after)")
        axes[1].set_title("After alignment")
        axes[1].legend(markerscale=5)

        plt.tight_layout()
        fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _log(f"Saved overlay plot to {overlay_path}")
    except Exception as plot_err:
        _log(f"Warning: overlay plot failed: {plot_err}")
        overlay_path = None

    moved_mean, moved_max = _displacement(yp, xp, y_aligned, x_aligned)

    out = WorkerOutput("stalign", task="align_to_image")
    out.set_data(
        n_points=int(len(yp)),
        image_shape=list(img.shape),
    )
    out.add_output_files(
        {
            "aligned_points_csv": aligned_csv,
            "transform_npz": transform_path,
            "overlay_png": str(overlay_path) if overlay_path else None,
        }
    )
    out.add_params(
        {
            "points_csv": points_csv,
            "image_path": image_path,
            "niter": niter,
            "a": a_val,
            "dx": dx,
            "intensity_scaling": INTENSITY_SCALING,
        }
    )
    record_method(
        out,
        "STalign LDDMM: affine + diffeomorphism fitted by matching the rasterized point density to the "
        "grayscale image (image pixel grid; no landmarks)",
    )
    out.set_summary(
        points_extent_y=float(yp.max() - yp.min()),
        points_extent_x=float(xp.max() - xp.min()),
        image_height=int(ny),
        image_width=int(nx),
        mean_point_displacement=moved_mean,
        max_point_displacement=moved_max,
    )
    out.set_analysis(
        f"STalign aligned {len(yp)} points to tissue image ({ny}x{nx}) "
        f"using LDDMM with {niter} iterations (a={a_val:.2f}), matching the rasterized point density to the "
        "grayscale image with no landmarks. The transform moved the points by a mean of "
        f"{moved_mean:.2f} image pixels (max {moved_max:.2f}). No alignment-quality measure is computed: "
        "there is no second point set to score against, so check stalign_overlay.png before using the "
        "coordinates."
    )
    return out.to_dict()


def main():
    parser = argparse.ArgumentParser(
        description="STalign worker: spatial alignment via LDDMM diffeomorphic registration."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["align_points", "align_to_image"],
        help="Task to run: align_points or align_to_image",
    )
    parser.add_argument("--source-csv", help="Path to source point cloud CSV (y, x)")
    parser.add_argument("--target-csv", help="Path to target point cloud CSV (y, x)")
    parser.add_argument("--points-csv", help="Path to point cloud CSV for image alignment (y, x)")
    parser.add_argument("--image-path", help="Path to tissue image (PNG, JPEG, TIFF)")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    parser.add_argument("--niter", type=int, default=200, help="Number of LDDMM iterations")
    parser.add_argument("--a", type=float, default=None, help="LDDMM kernel width parameter")

    args = parser.parse_args()

    # Redirect stdout to stderr during processing to keep stdout clean for JSON
    # (LDDMM prints iteration progress to stdout internally)
    orig_stdout = sys.stdout
    sys.stdout = sys.stderr

    result = None
    error_msg = None
    error_exc = None
    try:
        if args.task == "align_points":
            if not args.source_csv or not args.target_csv:
                raise ValueError("align_points requires --source-csv and --target-csv")
            result = run_align_points(
                source_csv=args.source_csv,
                target_csv=args.target_csv,
                output_dir=args.output_dir,
                niter=args.niter,
                a_param=args.a,
            )
        elif args.task == "align_to_image":
            if not args.points_csv or not args.image_path:
                raise ValueError("align_to_image requires --points-csv and --image-path")
            result = run_align_to_image(
                points_csv=args.points_csv,
                image_path=args.image_path,
                output_dir=args.output_dir,
                niter=args.niter,
            )
        else:
            raise ValueError(f"Unknown task: {args.task}")

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        error_msg = str(e)
        error_exc = e
    finally:
        sys.stdout = orig_stdout

    # Print a single JSON line to stdout
    if result is None:
        task = args.task if hasattr(args, "task") else "alignment"
        WorkerOutput.emit_error("stalign", error_msg or "Unknown error", task=task, exc=error_exc)
        sys.exit(1)
    else:
        print(json.dumps(result, default=str))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
