#!/usr/bin/env python3
"""
ClusterMap worker: cell segmentation for spatial transcriptomics.

- Runs inside /opt/conda/envs/clustermap_env
- All logs/progress go to stderr.
- stdout is reserved for a single final JSON line.

ClusterMap uses density peak clustering on gene-weighted coordinates
to assign RNA spots to cells, optionally guided by DAPI images.

Upstream reads its binarised DAPI image at every molecule's pixel,
``dapi[spot_location_2 - 1, spot_location_1 - 1(, spot_location_3 - 1)]``, with no bounds check
(preprocessing.py, postprocessing.py), and describes each molecule's neighbourhood in an
``n_spots x len(gene_list)`` matrix whose column for gene ``g`` is ``g - min(gene_list)``
(utils.NGC). So before ClusterMap sees the table:

- genes are re-encoded to contiguous codes ``0..n_genes-1``; the names go to the outputs;
- with a DAPI image, coordinates must already be whole 1-based pixel indices inside that image
  (checked, never rounded);
- without one, coordinates are rounded onto a 1-based grid -- an axis reaching below 1 is
  translated, not clipped -- and an all-ones placeholder is synthesised over it. Both are reported.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# <TOOL>_SRC seam (as spatialscope_worker.py does): this checkout exists only where it was
# installed, and a sys.path entry that does not exist fails silently -- the run dies later in an
# ImportError naming an upstream module, with no way to redirect it. The literal stays the default.
sys.path.insert(
    0,
    os.environ.get("CLUSTERMAP_SRC")
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "ClusterMap"),
)
from worker_utils import WorkerOutput, default_output_dir, preflight_check, record_method, sniff_tabular_sep


def log(msg: str) -> None:
    """Print log messages to stderr with a prefix."""
    print(f"[clustermap-worker] {msg}", file=sys.stderr, flush=True)


# Ceiling on how many points a *synthetic* DAPI placeholder may contribute to segmentation.
# Measured on the 2000-spot molecules.csv fixture: 3.8M points never finished (killed at
# 1200s), 102k was still running after 5 minutes, 21.7k finished in 61s.
_MAX_SYNTHETIC_DAPI_POINTS = 25_000

# Which DAPI axis each coordinate column indexes: upstream reads dapi[loc2 - 1, loc1 - 1, loc3 - 1].
_DAPI_AXIS = {"spot_location_2": 0, "spot_location_1": 1, "spot_location_3": 2}

_METHOD_WITH_DAPI = "ClusterMap density-peak clustering; nucleus seeds sampled from the supplied DAPI image"
_METHOD_NO_DAPI = (
    "ClusterMap density-peak clustering; no DAPI supplied, so the nucleus seeds are a synthetic uniform lattice"
)


def _lattice_size(shape, interval: int) -> int:
    """Points ClusterMap's ``add_dapi_points`` samples from an image that is non-zero everywhere.

    It keeps pixels with ``index % interval == 0`` on every axis, i.e. ``prod(ceil(dim / interval))``.
    """
    total = 1
    for dim in shape:
        total *= max(1, -(-int(dim) // int(interval)))  # ceil(dim / interval)
    return total


def _synthetic_dapi_grid_interval(
    dapi_shape, base_interval: int = 5, max_points: int = _MAX_SYNTHETIC_DAPI_POINTS
) -> int:
    """Pick the finest sampling interval whose lattice stays under ``max_points``.

    ClusterMap samples the DAPI image every ``interval`` pixels along each axis and treats each
    non-zero sample as a nucleus point. Our placeholder is all-ones and spans from pixel 0 to the
    largest coordinate (+10) on each axis, so it contributes ``prod(ceil(dim / interval))`` points
    -- a number set by the caller's coordinate range rather than by the data. On a 10232 x 9375
    slide that is 3.8 million fabricated points against 2000 real spots, and DPC never finishes.

    Returns ``base_interval`` unchanged whenever the lattice is already modest, so datasets that
    segment correctly today are unaffected.
    """
    interval = int(base_interval)
    while _lattice_size(dapi_shape, interval) > max_points:
        interval += 1
    return interval


def _coordinate_columns(spots_df, num_dims: int) -> list:
    """The coordinate columns ClusterMap will read, checked and made numeric in place.

    ``num_dims=3`` without a z column used to run until a KeyError deep inside upstream NGC, and a
    blank or non-numeric coordinate reached ``astype(int)`` or an index expression.
    """
    import numpy as np
    import pandas as pd

    if num_dims not in (2, 3):
        raise ValueError(f"num_dims must be 2 or 3, got {num_dims!r}.")
    columns = ["spot_location_1", "spot_location_2"]
    if num_dims == 3:
        if "spot_location_3" not in spots_df.columns:
            raise ValueError(
                "num_dims=3 needs a z coordinate in a 'spot_location_3' column, and spots_csv has only "
                f"{list(spots_df.columns)}. Pass num_dims=2 for a 2-D table."
            )
        columns.append("spot_location_3")
    for col in columns:
        numeric = pd.to_numeric(spots_df[col], errors="coerce")  # an integer column stays integer
        bad = ~np.isfinite(numeric.to_numpy(dtype=float))
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} of {len(numeric)} molecules have a missing or non-numeric {col}; "
                "ClusterMap places every molecule, so every coordinate must be a finite number."
            )
        spots_df[col] = numeric
    return columns


def _encode_genes(genes):
    """Contiguous integer codes ``0..n-1`` for ClusterMap, and the gene each code stands for.

    Upstream NGC writes gene ``g`` into column ``g - min(gene_list)`` of a ``len(gene_list)``-wide
    matrix, so the codes must run ``min..min+n-1`` with no gap. Integer gene IDs used to be passed
    as given, and one ID absent from the table (1-based IDs, a gene missing from a cropped tile) put
    a column past the end: IndexError. String names were encoded, but the names were then thrown
    away, so no output could say which gene a molecule was. Sorted, as ``astype("category")`` sorts,
    so a table of names gets the same codes it always did.
    """
    import numpy as np
    import pandas as pd

    series = pd.Series(genes).reset_index(drop=True)
    missing = series.isna()
    if bool(missing.any()):
        raise ValueError(
            f"{int(missing.sum())} of {len(series)} molecules have no gene (empty or NaN in the 'gene' column). "
            "ClusterMap describes every molecule by its gene; remove those rows from spots_csv first."
        )
    codes, uniques = pd.factorize(series, sort=True)
    return np.asarray(codes, dtype=np.int64), list(uniques)


def _snap_to_pixel_grid(coords):
    """Round ``(n, d)`` coordinates onto ClusterMap's 1-based integer pixel grid.

    Rounding is intrinsic: with no image, ClusterMap still indexes the placeholder by coordinate.
    An axis whose rounded minimum is below 1 is *translated* so that minimum becomes 1. The old
    ``clip(lower=1)`` piled every zero or negative coordinate onto 1 -- in a centred or
    global-micron frame, most of the slide on one line. Returns ``(grid, offsets, report)``.
    """
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    grid = np.rint(coords).astype(np.int64)  # half-to-even, as the old pandas .round() did
    shift = np.abs(coords - grid)
    moved = (shift > 0).any(axis=1)
    offsets = []
    for j in range(grid.shape[1]):
        low = int(grid[:, j].min())
        offset = 1 - low if low < 1 else 0
        grid[:, j] += offset
        offsets.append(int(offset))
    report = {"n_molecules_moved": int(moved.sum()), "max_axis_shift": float(shift.max()) if shift.size else 0.0}
    if report["n_molecules_moved"]:
        report["distinct_positions_before"] = int(len(np.unique(coords, axis=0)))
        report["distinct_positions_after"] = int(len(np.unique(grid, axis=0)))
    return grid, offsets, report


def _dapi_pixel_coordinates(spots_df, columns, dapi_shape) -> dict:
    """Check the coordinates index this DAPI image the way ClusterMap will; return them as ints.

    Upstream reads ``dapi[spot_location_2 - 1, spot_location_1 - 1]`` unchecked: a fractional
    coordinate is an IndexError, one past the edge is an IndexError, and a 0-based coordinate
    reads ``-1`` -- the far edge of the image, silently. A DAPI run is never rounded here; a table
    that is not in this image's 1-based pixels is refused with the numbers.
    """
    import numpy as np

    n = len(spots_df)
    as_ints = {}
    for col in columns:
        values = spots_df[col].to_numpy(dtype=float)
        fractional = values != np.rint(values)
        if fractional.any():
            raise ValueError(
                "With a DAPI image, ClusterMap reads the image at each molecule's pixel "
                "(dapi[spot_location_2 - 1, spot_location_1 - 1]), so the coordinates must be whole 1-based "
                f"pixel indices of that image; {int(fractional.sum())} of {n} molecules have a fractional "
                f"{col} (e.g. {values[fractional][0]!r}). Convert them to the image's pixel units first, or omit "
                "dapi_image_path to segment without an image (the coordinates are then rounded, and the "
                "rounding is reported)."
            )
        axis = _DAPI_AXIS[col]
        size = int(dapi_shape[axis])
        outside = (values < 1) | (values > size)
        if outside.any():
            raise ValueError(
                f"{col} spans [{values.min():g}, {values.max():g}], but it indexes axis {axis} of the DAPI image "
                f"(shape {tuple(int(s) for s in dapi_shape)}), and ClusterMap reads pixel {col} - 1, so values "
                f"must lie in 1..{size}. {int(outside.sum())} of {n} molecules fall outside. A 0-based table "
                "needs 1 added; a table in microns needs converting to this image's pixels."
            )
        as_ints[col] = values.astype(np.int64)
    return as_ints


def _write_csv_atomic(df, path: Path) -> None:
    """Write ``df`` next to ``path`` and rename it into place, so a crash never leaves half a table."""
    tmp = path.with_name(path.name + ".partial")
    df.to_csv(str(tmp), index=False)
    os.replace(str(tmp), str(path))


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """Redirect stdout to stderr so library prints do not pollute JSON output."""
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


def _run_clustermap(
    spots_csv: str,
    output_dir: str,
    xy_radius: float,
    z_radius: float,
    num_dims: int,
    min_spots: int,
    dapi_image_path: str,
) -> WorkerOutput:
    """Run ClusterMap segmentation and return WorkerOutput (caller emits)."""
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings = []

    # ── Load spots CSV ──────────────────────────────────────────────────
    log(f"Loading spots from {spots_csv}")
    # The user chooses this file, so its delimiter is read from its header rather than assumed.
    # A tab-delimited spots table parsed as commas becomes a single column named after the whole
    # header line, and the check below then reports the required columns as missing -- which reads
    # as "your file has the wrong columns" when the columns are all there.
    sep = sniff_tabular_sep(spots_csv)
    spots_df = pd.read_csv(spots_csv, sep=sep)

    # Validate required columns
    required_cols = ["spot_location_1", "spot_location_2", "gene"]
    for col in required_cols:
        if col not in spots_df.columns:
            raise ValueError(
                f"Required column '{col}' not found in spots CSV (read with sep={sep!r}). "
                f"Available columns: {list(spots_df.columns)}"
            )

    # Row i of every output is row i of the input: ClusterMap keeps the frame's index and order.
    spots_df = spots_df.reset_index(drop=True)
    n_spots_input = len(spots_df)
    if n_spots_input == 0:
        raise ValueError(f"spots_csv has a header but no molecules: {spots_csv}")
    if min_spots < 0:
        raise ValueError(f"min_spots must be 0 or more, got {min_spots}.")

    coord_cols = _coordinate_columns(spots_df, num_dims)
    # What the caller supplied, kept for the outputs: ClusterMap is handed rounded or cast
    # coordinates and integer gene codes, and neither can be joined back to the input by itself.
    supplied = spots_df[coord_cols + ["gene"]].copy()

    gene_codes, gene_names = _encode_genes(spots_df["gene"])
    spots_df["gene"] = gene_codes
    gene_list = np.arange(len(gene_names))
    n_genes = len(gene_names)
    log(f"Loaded {n_spots_input} spots, {n_genes} genes, num_dims={num_dims}")

    # ── Load or synthesize DAPI image ──────────────────────────────────
    coordinate_offset = [0] * len(coord_cols)
    rounding = {}
    if dapi_image_path:
        log(f"Loading DAPI image from {dapi_image_path}")
        import tifffile

        dapi = tifffile.imread(dapi_image_path)
        log(f"DAPI shape: {dapi.shape}, dtype: {dapi.dtype}")
        if dapi.ndim != num_dims:
            raise ValueError(
                f"The DAPI image has shape {tuple(dapi.shape)} ({dapi.ndim}-D) but num_dims={num_dims}; ClusterMap "
                "needs a single-channel stain of the same dimensionality -- (y, x) for 2-D, (y, x, z) with z last "
                "for 3-D. Reduce a multi-channel image to its DAPI channel first."
            )
        for col, values in _dapi_pixel_coordinates(spots_df, coord_cols, dapi.shape).items():
            spots_df[col] = values
    else:
        # ClusterMap requires a DAPI image for preprocessing (dapi_binary). When none is provided,
        # synthesize an all-ones image that spans the coordinate range so that no spots are
        # erroneously discarded. ClusterMap indexes that image with the coordinates, so they are
        # put on its 1-based integer grid first -- rounded, and translated where an axis dips below 1.
        log("No DAPI image provided; synthesizing a placeholder DAPI from spot coordinates")
        grid, coordinate_offset, rounding = _snap_to_pixel_grid(spots_df[coord_cols].to_numpy(dtype=float))
        for j, col in enumerate(coord_cols):
            spots_df[col] = grid[:, j]

        # Determine image dimensions that cover all spots with a margin.
        # ClusterMap indexes DAPI as dapi[spot_location_2, spot_location_1],
        # so axis-0 must cover spot_location_2 and axis-1 must cover spot_location_1.
        dim_0 = int(spots_df["spot_location_2"].max()) + 10  # axis 0 = spot_location_2
        dim_1 = int(spots_df["spot_location_1"].max()) + 10  # axis 1 = spot_location_1
        if num_dims == 3:
            dim_2 = int(spots_df["spot_location_3"].max()) + 10
            dapi = np.ones((dim_0, dim_1, dim_2), dtype=np.float32)
        else:
            dapi = np.ones((dim_0, dim_1), dtype=np.float32)
        log(f"Synthesized DAPI shape: {dapi.shape}")

    # ClusterMap samples this image every `dapi_grid_interval` pixels and feeds each non-zero
    # sample to DPC as a nucleus. A real DAPI is only non-zero on nuclei, so its lattice is
    # bounded by the tissue; the all-ones placeholder is non-zero *everywhere*, so its lattice
    # is set by the coordinate range instead. Coarsen the grid for the placeholder only -- a
    # real image keeps the upstream default and behaves exactly as before.
    dapi_grid_interval = 5
    n_synthetic_dapi_points = None
    if not dapi_image_path:
        dapi_grid_interval = _synthetic_dapi_grid_interval(dapi.shape)
        n_synthetic_dapi_points = _lattice_size(dapi.shape, dapi_grid_interval)
        if dapi_grid_interval != 5:
            log(
                f"Placeholder DAPI spans {dapi.shape}; sampling it every 5 pixels would fabricate "
                f"far more nuclei than there are spots, so coarsening dapi_grid_interval to "
                f"{dapi_grid_interval} (<= {_MAX_SYNTHETIC_DAPI_POINTS} synthetic points)"
            )
        warnings.append(
            f"No DAPI image was supplied, so no nuclear stain guided this segmentation: ClusterMap's nucleus "
            f"seeds are a synthetic uniform lattice of {n_synthetic_dapi_points} points, one every "
            f"{dapi_grid_interval} px over an all-ones placeholder of shape {tuple(dapi.shape)} spanning the "
            "coordinates. Pass dapi_image_path for DAPI-guided segmentation."
        )
        if rounding.get("n_molecules_moved"):
            warnings.append(
                f"Coordinates were rounded to whole units for ClusterMap's pixel grid: "
                f"{rounding['n_molecules_moved']} of {n_spots_input} molecules moved (by up to "
                f"{rounding['max_axis_shift']:.3g} per axis), and {rounding['distinct_positions_before']} distinct "
                f"positions became {rounding['distinct_positions_after']}. clustermap_cell_assignments.csv keeps "
                "your coordinates; clustermap_segmentation.csv holds the rounded ones."
            )
        shifted = {col: off for col, off in zip(coord_cols, coordinate_offset) if off}
        if shifted:
            warnings.append(
                f"Coordinates below 1 were translated onto ClusterMap's 1-based grid ({shifted} added per axis); "
                "relative positions are unchanged, clustermap_cell_assignments.csv keeps your coordinates, and "
                "clustermap_segmentation.csv holds the translated ones."
            )

    if num_dims == 2 and z_radius > xy_radius:
        warnings.append(
            f"num_dims=2, but ClusterMap's density-peak step searches within max(xy_radius, z_radius) = {z_radius}: "
            f"z_radius widened the 2-D neighbourhood beyond xy_radius={xy_radius}."
        )

    # ── Create ClusterMap object ────────────────────────────────────────
    from ClusterMap.clustermap import ClusterMap

    log(f"Creating ClusterMap (xy_radius={xy_radius}, z_radius={z_radius})")
    cm = ClusterMap(
        spots=spots_df.copy(),
        gene_list=gene_list,
        dapi=dapi,
        num_dims=num_dims,
        xy_radius=xy_radius,
        z_radius=z_radius,
    )

    # Upstream erase_small_clusters (postprocessing.py) erases a cell whose spot count is
    # <= min_spot_per_cell. min_spots is the smallest cell *kept*, so ClusterMap gets one less:
    # handing it min_spots itself silently raised the bar to min_spots + 1.
    upstream_min_spot_per_cell = int(min_spots) - 1
    cm.min_spot_per_cell = upstream_min_spot_per_cell

    # ── Preprocessing ───────────────────────────────────────────────────
    log("Running preprocessing...")
    cm.preprocess(dapi_grid_interval=dapi_grid_interval, LOF=False, pct_filter=0.1)

    # ── Segmentation ────────────────────────────────────────────────────
    log("Running segmentation...")
    add_dapi = dapi is not None
    cm.segmentation(
        cell_num_threshold=0.01,
        dapi_grid_interval=dapi_grid_interval,
        add_dapi=add_dapi,
    )

    # ── Extract results ─────────────────────────────────────────────────
    cell_ids = cm.spots["clustermap"].to_numpy()
    valid_mask = cell_ids >= 0
    n_assigned = int(valid_mask.sum())
    # One pass over the labels; a per-cell `cell_ids == cid` scan was O(n_cells * n_spots).
    unique_cells, spots_per_cell = np.unique(cell_ids[valid_mask], return_counts=True)
    n_cells = len(unique_cells)
    log(f"Segmentation complete: {n_cells} cells, {n_assigned}/{n_spots_input} spots assigned")

    # ── Save segmentation results ───────────────────────────────────────
    # ClusterMap's own frame (its grid coordinates, integer gene codes, is_noise, clustermap), plus
    # the input row and the gene name so it can be joined back to spots_csv.
    rows = cm.spots.index
    cm.spots["molecule_index"] = np.asarray(rows)
    cm.spots["gene_name"] = supplied["gene"].reindex(rows).to_numpy()
    seg_csv_path = out_dir / "clustermap_segmentation.csv"
    seg_tmp = seg_csv_path.with_name(seg_csv_path.name + ".partial")
    cm.save_segmentation(str(seg_tmp))
    os.replace(str(seg_tmp), str(seg_csv_path))
    log(f"Saved segmentation CSV to {seg_csv_path}")

    # Save cell assignments as a separate clean CSV, in the caller's terms: the coordinates and gene
    # as supplied, the cell, and the input row. gene_code is the column ClusterMap used for the gene.
    assignments_path = out_dir / "clustermap_cell_assignments.csv"
    assignments_df = pd.DataFrame(
        {
            "spot_location_1": supplied["spot_location_1"].reindex(rows).to_numpy(),
            "spot_location_2": supplied["spot_location_2"].reindex(rows).to_numpy(),
            "gene": supplied["gene"].reindex(rows).to_numpy(),
            "clustermap": cell_ids,
        }
    )
    if num_dims == 3:
        assignments_df["spot_location_3"] = supplied["spot_location_3"].reindex(rows).to_numpy()
    assignments_df["molecule_index"] = np.asarray(rows)
    assignments_df["gene_code"] = cm.spots["gene"].to_numpy()
    _write_csv_atomic(assignments_df, assignments_path)
    log(f"Saved cell assignments to {assignments_path}")

    # ── Compute cell statistics ─────────────────────────────────────────
    size_stats = {}
    if n_cells > 0:
        size_stats = {
            "mean_spots_per_cell": float(np.mean(spots_per_cell)),
            "median_spots_per_cell": float(np.median(spots_per_cell)),
            "min_spots_per_cell": int(np.min(spots_per_cell)),
            "max_spots_per_cell": int(np.max(spots_per_cell)),
        }

    # ── Save overlay plot ───────────────────────────────────────────────
    # x = spot_location_1, y = spot_location_2: ClusterMap's own convention (loc2 is the image row).
    plot_path = out_dir / "clustermap_segmentation.png"
    plot_tmp = plot_path.with_name(plot_path.name + ".partial")
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(
            assignments_df["spot_location_1"].to_numpy()[valid_mask],
            assignments_df["spot_location_2"].to_numpy()[valid_mask],
            # tab20 over raw ids spreads n_cells ids across 20 colours, so consecutive ids shared one
            # and neighbouring cells read as merged; cycle the ids through the palette instead.
            c=np.mod(cell_ids[valid_mask], 20),
            cmap="tab20",
            vmin=0,
            vmax=19,
            s=0.5,
            alpha=0.7,
        )
        ax.set_title(f"ClusterMap Segmentation ({n_cells} cells)")
        ax.set_xlabel("spot_location_1")
        ax.set_ylabel("spot_location_2")
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(str(plot_tmp), format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        os.replace(str(plot_tmp), str(plot_path))
        log(f"Saved plot to {plot_path}")
    except Exception as e:
        log(f"Warning: could not generate plot: {e}")
        warnings.append(f"The segmentation plot could not be drawn: {e}")
        with contextlib.suppress(OSError):
            os.remove(str(plot_tmp))
        plot_path = None

    # ── Emit output ─────────────────────────────────────────────────────
    out = WorkerOutput("clustermap", task="segmentation")
    out.set_data(
        n_spots_input=n_spots_input,
        n_genes=n_genes,
        num_dims=num_dims,
        dapi_shape=[int(s) for s in dapi.shape],
    )
    out.add_output_file("segmentation_csv", str(seg_csv_path))
    out.add_output_file("cell_assignments_csv", str(assignments_path))
    if plot_path:
        out.add_output_file("segmentation_plot", str(plot_path))
    out.add_output_file("output_dir", str(out_dir))

    out.add_params(
        {
            "xy_radius": xy_radius,
            "z_radius": z_radius,
            "num_dims": num_dims,
            "min_spots": min_spots,
            # A placeholder is always synthesized when none is supplied, so `dapi is not None` was
            # true on every run and told the reader nothing. Report whether a real image was used.
            "dapi_used": bool(dapi_image_path),
            "dapi_grid_interval": dapi_grid_interval,
            # The value ClusterMap's erase step was given: it erases cells with <= this many spots.
            "upstream_min_spot_per_cell": upstream_min_spot_per_cell,
            # DPC searches within max(xy_radius, z_radius) even in 2-D.
            "dpc_search_radius": max(xy_radius, z_radius),
            "coordinates_rounded": bool(rounding.get("n_molecules_moved")),
            "coordinate_offset": dict(zip(coord_cols, coordinate_offset)),
        }
    )
    if rounding.get("n_molecules_moved"):
        out.add_param("coordinate_rounding", rounding)
    record_method(out, _METHOD_WITH_DAPI if dapi_image_path else _METHOD_NO_DAPI, used_fallback=False)

    summary = {
        "n_cells": n_cells,
        "n_spots_assigned": n_assigned,
        "n_spots_unassigned": n_spots_input - n_assigned,
        "assignment_rate": round(n_assigned / n_spots_input * 100, 1) if n_spots_input > 0 else 0.0,
        "cell_size_stats": size_stats,
    }
    if n_synthetic_dapi_points is not None:
        summary["n_synthetic_dapi_points"] = n_synthetic_dapi_points
    out.set_summary(**summary)

    assignment_pct = round(n_assigned / n_spots_input * 100, 1) if n_spots_input > 0 else 0.0
    analysis_lines = [
        f"ClusterMap segmented {n_cells} cells from {n_spots_input} spots ({n_genes} genes, {num_dims}D).",
        f"Assignment rate: {assignment_pct}% ({n_assigned} spots assigned; cells with fewer than {min_spots} "
        "spots were erased).",
    ]
    if size_stats:
        analysis_lines.append(
            f"Spots per cell: mean={size_stats['mean_spots_per_cell']:.1f}, "
            f"median={size_stats['median_spots_per_cell']:.0f}, "
            f"range=[{size_stats['min_spots_per_cell']}, {size_stats['max_spots_per_cell']}]."
        )
    if not dapi_image_path:
        analysis_lines.append(
            f"No DAPI image was supplied, so no nuclear stain guided this: the nucleus seeds were a synthetic "
            f"uniform lattice of {n_synthetic_dapi_points} points, one every {dapi_grid_interval} px."
        )
        if rounding.get("n_molecules_moved"):
            analysis_lines.append(
                f"Coordinates were rounded to whole units for ClusterMap's pixel grid "
                f"({rounding['n_molecules_moved']} of {n_spots_input} molecules moved; "
                f"{rounding['distinct_positions_before']} distinct positions became "
                f"{rounding['distinct_positions_after']})."
            )
        if any(coordinate_offset):
            analysis_lines.append(
                "Axes reaching below 1 were translated onto ClusterMap's 1-based grid "
                f"({dict(zip(coord_cols, coordinate_offset))} added)."
            )
    out.set_analysis(" ".join(analysis_lines))
    out.add_warnings(warnings)
    return out


def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="ClusterMap cell segmentation worker")
    parser.add_argument("--spots-csv", required=True, help="Path to spots CSV")
    parser.add_argument("--output-dir", default=default_output_dir(), help="Output directory")
    parser.add_argument("--xy-radius", type=float, default=1.0, help="XY neighborhood radius")
    parser.add_argument("--z-radius", type=float, default=1.0, help="Z neighborhood radius")
    parser.add_argument("--num-dims", type=int, default=2, help="Number of spatial dimensions (2 or 3)")
    parser.add_argument(
        "--min-spots", type=int, default=5, help="Smallest cell kept (cells with fewer spots are erased)"
    )
    parser.add_argument("--dapi-image-path", default="", help="Optional DAPI image path")

    args = parser.parse_args()

    # Preflight checks
    inputs = {"spots_csv": args.spots_csv}
    if args.dapi_image_path:
        inputs["dapi_image"] = args.dapi_image_path

    try:
        preflight_check(
            inputs=inputs,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, PermissionError, ImportError) as e:
        WorkerOutput.emit_error("clustermap", str(e), task="segmentation")
        sys.exit(1)

    run_error = None
    with _redirect_stdout_to_stderr():
        try:
            worker_out = _run_clustermap(
                spots_csv=args.spots_csv,
                output_dir=args.output_dir,
                xy_radius=args.xy_radius,
                z_radius=args.z_radius,
                num_dims=args.num_dims,
                min_spots=args.min_spots,
                dapi_image_path=args.dapi_image_path,
            )
        except Exception as e:
            log(f"ERROR: {e}")
            import traceback

            traceback.print_exc(file=sys.stderr)
            run_error = e

    # Emit JSON to real stdout (after redirect context is closed)
    if run_error is not None:
        WorkerOutput.emit_error("clustermap", str(run_error), task="segmentation", exc=run_error)
        sys.exit(1)
    worker_out.emit()


if __name__ == "__main__":
    _cli_main()
