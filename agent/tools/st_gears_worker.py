#!/usr/bin/env python
"""
ST-GEARS worker script for SpatialOmicsLab MCP integration.

This script runs INSIDE the st_gears environment (/opt/conda/envs/st_gears)
and does the real work:

- Load a multi-section spatial AnnData .h5ad file. Spots with obs['in_tissue'] == 0 (background glass,
  which CELLxGENE Visium exports carry) are left out and counted in params.in_tissue_filter.
- Group spots into serial sections by an obs column (slice_key). Sections are ordered numerically
  when every id is a number, and in natural order otherwise ("S2" before "S10").
- Run the full ST-GEARS pipeline on the sections from start_idx to end_idx:

    - optional binning (granularity adjusting)
    - anchor computation via serial_align
    - rigid alignment via stack_slices_pairwise_rigid
    - elastic registration via stack_slices_pairwise_elas_field
    - interpolation back to the original resolution

- Concatenate the aligned sections into a single AnnData.
- Save the aligned AnnData and a small metadata JSON.

What the upstream package needs and the library's files do not carry:

- ST-GEARS reads and writes ``obsm['spatial'][:, 2]`` (recons.py:150 and :519,
  granularity_adjusting.py:102), so a two-column ``obsm['spatial']`` -- the shape every converter
  in this repository writes -- is an IndexError. The worker hands ST-GEARS a working copy with the
  slice ordinal as the third column and puts the two-column original back before writing.
- ``st_gears.binning`` calls ``X.todense()``, which a dense ``X`` does not have; the binning input
  gets a CSR copy of X.
- Linear interpolation (``st_gears.interpolate``) leaves NaN for spots outside the binned grid's
  convex hull. They are counted and reported; ``--allow-nearest-fallback`` fills them from the
  nearest bin's displacement instead, and the payload says so.

IMPORTANT:
- All progress / logging goes to stderr, prefixed with [st-gears-worker].
- The ONLY thing printed to stdout is a single line of JSON summarizing
  the run, so the FastMCP wrapper can safely parse it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import traceback

import numpy as np
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    drop_unlabeled,
    keep_in_tissue,
    record_ignored,
    record_in_tissue,
    record_method,
    resolve_compute,
)

try:
    import anndata as ad  # type: ignore
except Exception as e:  # pragma: no cover
    print("[st-gears-worker] ERROR: anndata is required in the st_gears env.", file=sys.stderr)
    print(str(e), file=sys.stderr)
    WorkerOutput.emit_error("st_gears", "anndata import failed: " + str(e), task="3d_reconstruction")
    sys.exit(1)

try:
    import st_gears  # type: ignore
except Exception as e:  # pragma: no cover
    print("[st-gears-worker] ERROR: st_gears is not importable in this env.", file=sys.stderr)
    print(str(e), file=sys.stderr)
    WorkerOutput.emit_error("st_gears", "st_gears import failed: " + str(e), task="3d_reconstruction")
    sys.exit(1)


#: Peak bytes per (spot x gene) element while ``st_gears.binning`` runs. It densifies X
#: (``X.todense()``) and concatenates it with the label column into an object-typed array that a
#: pandas groupby then sums. Measured 47-84 B/element in the st_gears env (pandas 1.4.3) on
#: 2,000-4,000 spot sections; rounded up.
BINNING_BYTES_PER_ELEMENT = 96

#: Peak bytes per point of the elastic-field grid. ``stack_slices_pairwise_elas_field`` applies the
#: field with ``scipy.interpolate.griddata(method='linear')`` over every grid point, which builds a
#: Qhull Delaunay triangulation of the whole grid. Measured 1,867 B/point (scipy 1.10.1, 0.25M and
#: 1M points); rounded up.
FIELD_BYTES_PER_POINT = 2000

#: What ``serial_align`` holds per section pair: dense spots x genes copies of both sections (X,
#: its shifted copy, the row-normalised copy and its log), and float64 n_A x n_B / n_A^2 / n_B^2
#: matrices (feature cost, two structure matrices, the initial and the solved plan, solver
#: workspace). Measured ~126 B per n_A*n_B entry on two 2,000-spot sections (POT 0.9.1), which
#: 8 * (n_A^2 + n_B^2 + 14 n_A n_B) covers. The pair matrices are float64; the spots x genes copies
#: keep X's own dtype (float32 counts stay float32 through the KL cost), so they are costed at X's
#: itemsize. Two full Visium sections (4,910 and 4,972 spots x 36,601 float32 genes, binning off)
#: peaked at 7.2 GB against an estimate of 10.3 GB.
ALIGN_DENSE_COPIES = 5
ALIGN_PAIR_MATRICES = 14

_GIB = float(1 << 30)


def _log(msg: str) -> None:
    print(f"[st-gears-worker] {msg}", file=sys.stderr)


def str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ST-GEARS worker for SpatialOmicsLab MCP")
    p.add_argument("--st-h5ad", required=True, help="Path to multi-section spatial AnnData (.h5ad)")
    p.add_argument("--output-dir", required=True, help="Directory to write outputs")

    p.add_argument(
        "--slice-key", default="slice_id", help="obs column defining serial section ID (e.g. 'slice_id', 'section')"
    )
    p.add_argument(
        "--group-key", default="annotation", help="obs column with cluster/annotation used as grouping (anchors)"
    )

    p.add_argument(
        "--binning-on", type=str, default="true", help="Whether to run granularity adjusting / binning (true/false)"
    )
    p.add_argument("--bin-step", type=int, default=2, help="Step size for binning; see ST-GEARS README")

    p.add_argument(
        "--uniform-weight", type=str, default="false", help="Pass to serial_align (False uses Distributive Constraints)"
    )
    p.add_argument(
        "--filter-by-label", type=str, default="true", help="Filter label pairs that never co-occur in two sections"
    )

    p.add_argument(
        "--tune-alpha-li",
        type=str,
        default="0.8,0.2,0.05,0.013",
        help="Comma-separated list of alphas for serial_align, e.g. '0.8,0.2,0.05,0.013'",
    )
    p.add_argument("--num-itermax", type=int, default=200, help="numItermax for serial_align")

    p.add_argument("--fil-pc", type=int, default=20, help="fil_pc for stack_slices_pairwise_rigid / elas_field")

    p.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help=(
            "Pixel size for the elastic field, in coordinate units. If omitted, the median nearest-neighbour "
            "distance between the spots (bins when binning is on) handed to ST-GEARS -- upstream's own guidance, "
            "'a rough average of spots distance'."
        ),
    )
    p.add_argument(
        "--sigma", type=float, default=1.0, help="Gaussian kernel sigma for elastic field (not recommended to change)"
    )

    p.add_argument(
        "--start-idx", type=int, default=None, help="Start index of sections from slicesl to be aligned (default 0)"
    )
    p.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index of sections from slicesl to be aligned (default last index)",
    )

    p.add_argument("--use-gpu", type=str, default="false", help="Whether to let serial_align use GPU (true/false)")

    p.add_argument("--seed", type=int, default=0, help="Random seed for numpy; ST-GEARS itself is mostly deterministic")

    p.add_argument(
        "--drop-unlabeled",
        type=str,
        default="false",
        help=(
            "Drop spots whose group_key label or slice_key id is missing (NaN/empty) instead of refusing the run "
            "(true/false). The count is reported."
        ),
    )
    p.add_argument(
        "--allow-nearest-fallback",
        type=str,
        default="false",
        help=(
            "When binning is on, give spots outside the binned grid's convex hull (NaN under ST-GEARS's linear "
            "interpolation) the displacement of the nearest bin instead (true/false). Reported per spot."
        ),
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------------------------------
# Input shaping
# ---------------------------------------------------------------------------------------------


def parse_alphas(text) -> list:
    """``'0.8,0.2'`` -> ``[0.8, 0.2]``. A Python-repr spelling (``'[0.8, 0.2]'``) is read the same way.

    Anything else is refused. Until 2026-09-29 an unparseable list was replaced with the default
    list, and ``params.tune_alpha_li`` then reported the default as though it had been requested.
    """
    raw = str(text).strip()
    parts = [x.strip() for x in raw.strip("[]() ").split(",")]
    parts = [x for x in parts if x]
    if not parts:
        raise ValueError(f"tune_alpha_li={text!r} holds no values; pass a comma-separated list such as '0.8,0.2'.")
    values = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            raise ValueError(
                f"tune_alpha_li={text!r} is not a list of numbers ({part!r} is not one); pass a "
                "comma-separated list such as '0.8,0.2,0.05,0.013'."
            ) from None
    return values


def section_id_set(values) -> set:
    """The distinct, non-missing section ids in ``values``, as text."""
    import pandas as pd

    return set(pd.Series(np.asarray(values, dtype=object), dtype=object).dropna().astype(str))


def _natural_key(value) -> list:
    """``'S2'`` before ``'S10'``: digit runs compare as integers, everything else as text."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", str(value))]


def order_slice_ids(values):
    """The serial order of the section ids in ``values``, and how it was decided.

    ``sorted()`` on ids stored as text -- which is how an h5ad stores most obs columns -- put
    ``'10'`` before ``'2'``, so a stack of eleven or more sections was aligned out of anatomical
    order and every pair after the ninth registered two sections that are not neighbours. Ids that
    are all numbers are ordered by value; any other ids in natural order.
    """
    import pandas as pd

    seen = []
    known = set()
    for v in pd.unique(pd.Series(np.asarray(values, dtype=object), dtype=object)):
        k = str(v)
        if k not in known:
            known.add(k)
            seen.append(v)
    try:
        return sorted(seen, key=lambda v: float(str(v))), "numeric"
    except ValueError:
        return sorted(seen, key=_natural_key), "natural"


def add_ordinal_z(slices, ordinals):
    """Give each two-column ``obsm['spatial']`` a third column: the slice ordinal.

    ST-GEARS indexes ``obsm['spatial'][:, 2]`` unconditionally -- ``interpolate`` reads it from the
    original sections and both stacking functions copy it into their outputs -- so a two-column
    input raised ``IndexError: index 2 is out of bounds`` on every path. Returns the originals,
    ``None`` where the input already had three columns, for :func:`restore_xy`.
    """
    originals = []
    for sl, z in zip(slices, ordinals):
        if "spatial" not in sl.obsm:
            raise KeyError(f"obsm['spatial'] not found. Available obsm keys: {list(sl.obsm.keys())}")
        arr = np.asarray(sl.obsm["spatial"])
        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            raise ValueError(
                f"obsm['spatial'] has shape {arr.shape}; ST-GEARS takes two columns (x, y) or three (x, y, z)."
            )
        if arr.shape[1] == 2:
            originals.append(sl.obsm["spatial"])
            sl.obsm["spatial"] = np.column_stack([arr.astype(float), np.full(arr.shape[0], float(z))])
        else:
            originals.append(None)
    return originals


def restore_xy(slices, originals) -> None:
    """Put the two-column input back: under the 3D coordinate contract ``obsm['spatial']`` is the
    untouched original and an aligner never writes it."""
    for sl, orig in zip(slices, originals):
        if orig is not None:
            sl.obsm["spatial"] = orig


def binning_input(sl, group_key):
    """What ``st_gears.binning`` can read. It calls ``adata.X.todense()``, which a dense ndarray lacks."""
    from scipy import sparse

    if sparse.issparse(sl.X):
        return sl
    shim = ad.AnnData(X=sparse.csr_matrix(np.asarray(sl.X)), obs=sl.obs[[group_key]].copy())
    shim.obsm["spatial"] = sl.obsm["spatial"]
    return shim


def median_spot_spacing(slices) -> float:
    """Median nearest-neighbour distance between the spots of ``slices`` (xy only)."""
    from scipy.spatial import cKDTree

    found = []
    for sl in slices:
        xy = np.asarray(sl.obsm["spatial"], dtype=float)[:, :2]
        if xy.shape[0] < 2:
            continue
        d, _ = cKDTree(xy).query(xy, k=2)
        d = d[:, 1]
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            found.append(d)
    if not found:
        raise ValueError(
            "cannot estimate the spot spacing for the elastic field (every section has fewer than two distinct "
            "positions); pass pixel_size explicitly."
        )
    return float(np.median(np.concatenate(found)))


# ---------------------------------------------------------------------------------------------
# Memory: the dense intermediates are ST-GEARS's own, so they are costed, not avoided
#
# ``available_memory_bytes`` is the shared worker_utils reader (MemAvailable, capped by the room under
# the cgroup limit with the page cache counted as reclaimable). This worker used to carry its own copy
# that subtracted ``memory.current`` / ``memory.usage_in_bytes`` -- page cache included -- from the
# limit, so a memory-limited container at its limit on cache alone refused runs that fit.
# ---------------------------------------------------------------------------------------------


def estimate_binning_bytes(n_obs, n_vars) -> int:
    return int(BINNING_BYTES_PER_ELEMENT * int(n_obs) * int(n_vars))


def estimate_align_bytes(n_a, n_b, n_genes, itemsize=8) -> int:
    n_a, n_b, g = int(n_a), int(n_b), int(n_genes)
    dense = ALIGN_DENSE_COPIES * max(4, int(itemsize)) * (n_a + n_b) * g
    pair = 8 * (n_a * n_a + n_b * n_b + ALIGN_PAIR_MATRICES * n_a * n_b)
    return int(dense + pair)


def _itemsize(X) -> int:
    """Bytes per element of ``X`` once densified (8 when the dtype cannot be read)."""
    try:
        return int(np.dtype(X.dtype).itemsize)
    except (AttributeError, TypeError):
        return 8


def field_grid_points(xy, pixel_size) -> tuple:
    """Shape and point count of the elastic field ST-GEARS builds over ``xy`` at ``pixel_size``.

    ``generate_fields_by_offset`` sizes it ``floor(extent / pixel_size) + 1`` per axis and
    ``stack_slices_pairwise_elas_field`` pads one more row and column.
    """
    xy = np.asarray(xy, dtype=float)[:, :2]
    extent = np.nanmax(xy, axis=0) - np.nanmin(xy, axis=0)
    shape = tuple(int(v) for v in (np.floor(extent / float(pixel_size)) + 2))
    return shape, int(shape[0]) * int(shape[1])


def check_memory(stage, need, remedy, available=None):
    """Refuse, with both numbers and the knob, when ``need`` bytes cannot fit. Never subsamples."""
    if available is None:
        available = available_memory_bytes()
    if available is not None and need > available:
        raise MemoryError(
            f"{stage} needs an estimated {need / _GIB:.1f} GiB and {available / _GIB:.1f} GiB is available. {remedy}"
        )
    return need, available


# ---------------------------------------------------------------------------------------------
# After interpolation
# ---------------------------------------------------------------------------------------------


def nearest_displacement(com, xy):
    """Displacement of the nearest bin of ``com`` (spatial -> spatial_elas) at each point of ``xy``."""
    from scipy.interpolate import griddata

    src = np.asarray(com.obsm["spatial"], dtype=float)[:, :2]
    moved = np.asarray(com.obsm["spatial_elas"], dtype=float)[:, :2]
    xy = np.asarray(xy, dtype=float)[:, :2]
    dx = griddata(src, moved[:, 0] - src[:, 0], xy, method="nearest")
    dy = griddata(src, moved[:, 1] - src[:, 1], xy, method="nearest")
    return np.column_stack([dx, dy])


def build_frame(sl, z, com=None, allow_nearest=False):
    """The contract frame for one aligned section: ``(xyz, xy_source, aligned_key)``.

    ``xy_source`` says per spot where its aligned xy came from: ``'elastic'`` (ST-GEARS moved the
    spot itself), ``'interpolated'`` (linear interpolation from the bins), ``'nearest'`` (the
    opt-in nearest-bin fill) or ``'none'`` (no aligned coordinate: the row is NaN).
    """
    aligned_key = ""
    for candidate in ("spatial_elas_reuse", "spatial_elas", "spatial_rigid"):
        if candidate in sl.obsm:
            aligned_key = candidate
            break
    if not aligned_key:
        raise RuntimeError(
            "ST-GEARS wrote none of spatial_elas_reuse, spatial_elas or spatial_rigid, so there are no aligned "
            "coordinates to record."
        )
    xy = np.array(np.asarray(sl.obsm[aligned_key], dtype=float)[:, :2], dtype=float)
    source = np.full(xy.shape[0], "interpolated" if aligned_key == "spatial_elas_reuse" else "elastic", dtype=object)
    missing = ~np.isfinite(xy).all(axis=1)
    if missing.any() and allow_nearest and com is not None and aligned_key == "spatial_elas_reuse":
        start = np.asarray(sl.obsm["spatial"], dtype=float)[missing, :2]
        xy[missing] = start + nearest_displacement(com, start)
        source[missing] = "nearest"
        missing = ~np.isfinite(xy).all(axis=1)
    source[missing] = "none"
    xyz = np.column_stack([xy, np.full(xy.shape[0], float(z))])
    return xyz, source, aligned_key


def _upstream(fn, *args, **kwargs):
    """Call an ST-GEARS function with its prints sent to stderr.

    ``serial_align`` prints its progress and POT prints its iteration table, both to stdout, which
    this worker reserves for the one JSON result line.
    """
    with contextlib.redirect_stdout(sys.stderr):
        return fn(*args, **kwargs)


def _write_atomic_h5ad(adata, path) -> None:
    partial = path + ".partial"
    adata.write_h5ad(partial)
    os.replace(partial, path)


def _write_atomic_json(obj, path) -> None:
    partial = path + ".partial"
    with open(partial, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(partial, path)


def _fail(msg) -> None:
    """Report a refusal and stop. Called inside the handler when there is an exception, so the payload's
    traceback is the live one."""
    _log(msg)
    WorkerOutput.emit_error("st_gears", msg, task="3d_reconstruction")
    sys.exit(1)


def main(argv=None) -> None:
    args = parse_args(argv)

    # Convert string flags to bools
    binning_on = str2bool(args.binning_on)
    uniform_weight = str2bool(args.uniform_weight)
    filter_by_label = str2bool(args.filter_by_label)
    allow_drop = str2bool(args.drop_unlabeled)
    allow_nearest = str2bool(args.allow_nearest_fallback)
    # st_gears/helper.py answers an unavailable CUDA with a bare `quit()` -- a SystemExit, which
    # derives from BaseException and so escapes the `except Exception` around serial_align below.
    # The worker then ends rc=0 with an empty stdout and the portal reports "produced no output"
    # for what is really "this box has no GPU". Resolve the request against the hardware first, so
    # a GPU request on a CPU-only box arrives as use_gpu=False and the run simply proceeds on CPU.
    use_gpu = resolve_compute(str2bool(args.use_gpu)).device.startswith("cuda")

    try:
        tune_alpha_li = parse_alphas(args.tune_alpha_li)
    except ValueError as e:
        _fail(str(e))

    if args.pixel_size is not None and not args.pixel_size > 0:
        _fail(f"pixel_size={args.pixel_size} must be a positive number of coordinate units.")
    if binning_on and args.bin_step <= 0:
        _fail(f"bin_step={args.bin_step} must be a positive number of coordinate units.")

    np.random.seed(args.seed)

    st_h5ad = os.path.abspath(args.st_h5ad)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    _log(f"Loading AnnData from {st_h5ad}")
    if not os.path.exists(st_h5ad):
        msg = f"Input file not found: {st_h5ad}"
        _log(msg)
        WorkerOutput.emit_error("st_gears", msg, task="3d_reconstruction")
        sys.exit(1)

    try:
        adata = ad.read_h5ad(st_h5ad)
    except Exception as e:
        _log("Failed to read h5ad:")
        _log("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        WorkerOutput.emit_error("st_gears", f"failed to read h5ad: {type(e).__name__}: {e}", task="3d_reconstruction")
        sys.exit(1)

    if args.slice_key not in adata.obs:
        msg = f"obs column '{args.slice_key}' not found in AnnData.obs"
        _log(msg)
        WorkerOutput.emit_error("st_gears", msg, task="3d_reconstruction")
        sys.exit(1)

    if args.group_key not in adata.obs:
        msg = f"obs column '{args.group_key}' not found in AnnData.obs (required for grouping)"
        _log(msg)
        WorkerOutput.emit_error("st_gears", msg, task="3d_reconstruction")
        sys.exit(1)

    if "spatial" not in adata.obsm:
        _fail(f"obsm['spatial'] not found in AnnData. Available obsm keys: {list(adata.obsm.keys())}")

    n_spots_input, n_genes = int(adata.n_obs), int(adata.n_vars)

    # Background glass (obs['in_tissue'] == 0; CELLxGENE Visium exports carry every array spot, labelled
    # e.g. 'unknown') is not tissue: aligned, it anchors the sections on the capture area rather than on
    # the tissue. The shared rule leaves it out and says how many. A section left with no spot at all
    # is refused rather than silently removed from the stack (it would shift start_idx / end_idx).
    sections_before = section_id_set(adata.obs[args.slice_key].to_numpy(dtype=object))
    try:
        adata, _, n_off_tissue = keep_in_tissue(adata, "spots")
    except ValueError as e:
        _fail(str(e))
    if n_off_tissue:
        _log(f"Left out {n_off_tissue} of {n_spots_input} spots with obs['in_tissue'] == 0 (background).")
        lost = sorted(sections_before - section_id_set(adata.obs[args.slice_key].to_numpy(dtype=object)))
        if lost:
            _fail(
                f"section(s) {lost} of obs['{args.slice_key}'] have no spot with obs['in_tissue'] == 1, so nothing "
                "of them is tissue to align. Fix their in_tissue flags, or remove those sections from the input."
            )

    # Missing section ids and missing labels: ST-GEARS has no notion of either. A NaN section id
    # selects no spots at all, and a NaN label crashed binning (a bin of only-NaN labels) or
    # serial_align (IndexError) with nothing saying why.
    try:
        keep_sid, n_no_sid = drop_unlabeled(
            adata.obs[args.slice_key].to_numpy(), allow_drop, what=f"spots (obs['{args.slice_key}'], the section id)"
        )
    except ValueError as e:
        _fail(str(e))
    if n_no_sid:
        adata = adata[keep_sid].copy()
        _log(f"Dropped {n_no_sid} spots with no section id in obs['{args.slice_key}'] (drop_unlabeled=True).")

    # Split into slices list, in serial order
    slice_ids, slice_order = order_slice_ids(adata.obs[args.slice_key].to_numpy())
    _log(f"Found {len(slice_ids)} serial sections in obs['{args.slice_key}'] ({slice_order} order): {slice_ids}")

    sid_text = adata.obs[args.slice_key].astype(str).to_numpy()
    slicesl = []
    for sid in slice_ids:
        sub = adata[sid_text == str(sid)].copy()
        _log(f"Section {sid}: {sub.n_obs} spots, {sub.n_vars} genes")
        slicesl.append(sub)

    if len(slicesl) < 2:
        msg = "ST-GEARS expects at least 2 serial sections; found < 2."
        _log(msg)
        WorkerOutput.emit_error("st_gears", msg, task="3d_reconstruction")
        sys.exit(1)

    # Determine alignment range indices
    start_i = 0 if args.start_idx is None else args.start_idx
    end_i = (len(slicesl) - 1) if args.end_idx is None else args.end_idx
    if start_i < 0 or end_i >= len(slicesl) or start_i >= end_i:
        msg = (
            f"Invalid start/end indices: start_idx={start_i}, end_idx={end_i}, n_slices={len(slicesl)}. "
            "They must satisfy 0 <= start_idx < end_idx <= n_slices - 1: ST-GEARS aligns at least two sections."
        )
        _log(msg)
        WorkerOutput.emit_error("st_gears", msg, task="3d_reconstruction")
        sys.exit(1)
    selected = list(range(start_i, end_i + 1))

    # Missing labels in the sections that will be aligned
    try:
        _, n_unlabeled = drop_unlabeled(
            np.concatenate([slicesl[i].obs[args.group_key].to_numpy(dtype=object) for i in selected]),
            allow_drop,
            what=f"spots in the sections being aligned (obs['{args.group_key}'])",
        )
    except ValueError as e:
        _fail(str(e))
    if n_unlabeled:
        for i in selected:
            keep, _ = drop_unlabeled(slicesl[i].obs[args.group_key].to_numpy(dtype=object), True)
            slicesl[i] = slicesl[i][keep].copy()
        _log(f"Dropped {n_unlabeled} spots with no label in obs['{args.group_key}'] (drop_unlabeled=True).")
    for i in selected:
        if slicesl[i].n_obs == 0:
            _fail(f"section {slice_ids[i]} has no spots left to align.")

    # ST-GEARS indexes a third coordinate column; give it one and remember the two-column input.
    try:
        original_xy = dict(zip(selected, add_ordinal_z([slicesl[i] for i in selected], [float(i) for i in selected])))
    except (KeyError, ValueError) as e:
        _fail(str(e))
    z_column_added = any(o is not None for o in original_xy.values())
    if z_column_added:
        _log("obsm['spatial'] has 2 columns; ST-GEARS reads a third, so its working copy carries the slice ordinal.")

    # Binning / granularity adjusting (only the sections being aligned)
    _log(f"Binning on: {binning_on}, step={args.bin_step}")
    slice_srk_li = list(slicesl)
    bins_per_section = {}
    if binning_on:
        # The binning cost is set by spots x genes, not by bin_step, so the one knob is binning_on --
        # and turning it off moves the cost to serial_align over every spot. Say what that would need.
        unbinned_align = max(
            estimate_align_bytes(slicesl[a].n_obs, slicesl[b].n_obs, slicesl[a].n_vars, _itemsize(slicesl[a].X))
            for a, b in zip(selected[:-1], selected[1:])
        )
        for i in selected:  # every section is costed before any is binned
            try:
                check_memory(
                    f"st_gears.binning on section {slice_ids[i]} ({slicesl[i].n_obs} spots x {slicesl[i].n_vars} "
                    f"genes; it densifies X and sums it in an object-typed pandas frame, ~{BINNING_BYTES_PER_ELEMENT} "
                    "bytes per spot x gene)",
                    estimate_binning_bytes(slicesl[i].n_obs, slicesl[i].n_vars),
                    "bin_step does not change this cost (every spot x gene is densified before the bins are summed). "
                    "binning_on=False skips binning; serial_align then works on every spot directly, which needs "
                    f"an estimated {unbinned_align / _GIB:.1f} GiB for the largest adjacent pair here (dense n_A x n_B "
                    "matrices).",
                )
            except MemoryError as e:
                _fail(str(e))
        for i in selected:
            slice_srk_li[i] = _upstream(
                st_gears.binning, binning_input(slicesl[i], args.group_key), args.group_key, args.bin_step
            )
            bins_per_section[str(slice_ids[i])] = int(slice_srk_li[i].n_obs)
            _log(f"Section {slice_ids[i]}: {slicesl[i].n_obs} spots -> {slice_srk_li[i].n_obs} bins")

    # Pixel size of the elastic field
    if args.pixel_size is not None:
        pixel_size = float(args.pixel_size)
        pixel_size_source = "given"
    else:
        try:
            pixel_size = median_spot_spacing([slice_srk_li[i] for i in selected])
        except ValueError as e:
            _fail(str(e))
        pixel_size_source = "median nearest-neighbour spacing of the " + ("bins" if binning_on else "spots")
        _log(f"pixel_size not given; using the {pixel_size_source}: {pixel_size:g}")

    for a, b in zip(selected[:-1], selected[1:]):
        n_a, n_b = slice_srk_li[a].n_obs, slice_srk_li[b].n_obs
        try:
            check_memory(
                f"serial_align on sections {slice_ids[a]} and {slice_ids[b]} ({n_a} and {n_b} "
                f"{'bins' if binning_on else 'spots'}, {slice_srk_li[a].n_vars} genes; dense spots x genes copies and "
                "n_A x n_B float64 matrices)",
                estimate_align_bytes(n_a, n_b, slice_srk_li[a].n_vars, _itemsize(slice_srk_li[a].X)),
                "With binning_on=True, a larger bin_step aggregates the spots into fewer bins; every spot is kept "
                "and the result is interpolated back to each one.",
            )
        except MemoryError as e:
            _fail(str(e))

    # Compute anchors
    _log("Computing anchors via st_gears.helper.gen_anncell_cid_from_all ...")
    anncell_cid = st_gears.helper.gen_anncell_cid_from_all([slice_srk_li[i] for i in selected], args.group_key)

    _log(
        f"Running serial_align(start_i={start_i}, end_i={end_i}, "
        f"alphas={tune_alpha_li}, numItermax={args.num_itermax}, "
        f"uniform_weight={uniform_weight}, filter_by_label={filter_by_label}, use_gpu={use_gpu})"
    )

    try:
        pili, tyscoreli, alphali, regis_ilist, ali, bli = _upstream(
            st_gears.serial_align,
            slice_srk_li,
            anncell_cid,
            label_col=args.group_key,
            start_i=start_i,
            end_i=end_i,
            tune_alpha_li=tune_alpha_li,
            numItermax=args.num_itermax,
            dissimilarity_val="kl",
            dissimilarity_weight_val="kl",
            uniform_weight=uniform_weight,
            map_method_dis2wei="logistic",
            filter_by_label=filter_by_label,
            use_gpu=use_gpu,
            verbose=True,
        )
    except Exception as e:
        _log("serial_align failed:")
        _log("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        WorkerOutput.emit_error("st_gears", f"serial_align failed: {type(e).__name__}: {e}", task="3d_reconstruction")
        sys.exit(1)

    regis_ilist = [int(i) for i in regis_ilist]
    _log(f"serial_align finished; regis_ilist = {regis_ilist}")

    # Rigid registration. It returns the list it was given -- the sections in regis_ilist, in that
    # order -- so from here on everything is indexed by position in that list, never by the
    # original section index again (a start_idx > 0 used to index past its end).
    _log(f"Running stack_slices_pairwise_rigid with fil_pc={args.fil_pc} ...")
    try:
        registered = _upstream(
            st_gears.stack_slices_pairwise_rigid,
            [slice_srk_li[i] for i in regis_ilist],
            pili,
            label_col=args.group_key,
            fil_pc=args.fil_pc,
            filter_by_label=filter_by_label,
        )
    except Exception as e:
        _log("stack_slices_pairwise_rigid failed:")
        _log("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        WorkerOutput.emit_error(
            "st_gears", f"stack_slices_pairwise_rigid failed: {type(e).__name__}: {e}", task="3d_reconstruction"
        )
        sys.exit(1)

    # The elastic field is a dense grid over each section at pixel_size, triangulated whole.
    for pos, i in enumerate(regis_ilist):
        shape, n_points = field_grid_points(registered[pos].obsm["spatial_rigid"], pixel_size)
        try:
            check_memory(
                f"The elastic field for section {slice_ids[i]} ({shape[0]} x {shape[1]} = {n_points} grid points at "
                f"pixel_size={pixel_size:g}; scipy griddata triangulates every point, ~{FIELD_BYTES_PER_POINT} bytes "
                "each)",
                n_points * FIELD_BYTES_PER_POINT,
                "pixel_size is in the coordinates' own units and upstream asks for 'a rough average of spots "
                "distance'; leave it unset to use the measured spot spacing, or pass a larger value.",
            )
        except MemoryError as e:
            _fail(str(e))

    # Elastic registration
    _log(
        f"Running stack_slices_pairwise_elas_field with pixel_size={pixel_size}, "
        f"sigma={args.sigma}, fil_pc={args.fil_pc} ..."
    )
    try:
        registered = _upstream(
            st_gears.stack_slices_pairwise_elas_field,
            registered,
            pili,
            label_col=args.group_key,
            pixel_size=pixel_size,
            fil_pc=args.fil_pc,
            filter_by_label=filter_by_label,
            sigma=args.sigma,
        )
    except Exception as e:
        _log("stack_slices_pairwise_elas_field failed:")
        _log("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        WorkerOutput.emit_error(
            "st_gears", f"stack_slices_pairwise_elas_field failed: {type(e).__name__}: {e}", task="3d_reconstruction"
        )
        sys.exit(1)

    # Interpolate back to original resolution -- the originals of exactly the registered sections,
    # position for position.
    originals_regis = [slicesl[i] for i in regis_ilist]
    _log("Interpolating back to original resolution ...")
    try:
        if binning_on:
            slicesl_aligned = _upstream(st_gears.interpolate, registered, originals_regis)
        else:
            slicesl_aligned = registered
    except Exception as e:
        _log("interpolate failed:")
        _log("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        WorkerOutput.emit_error("st_gears", f"interpolate failed: {type(e).__name__}: {e}", task="3d_reconstruction")
        sys.exit(1)

    # The 3D frame. Corrected 2026-09-21: what this block wrote was not ST-GEARS's answer.
    #
    # ST-GEARS writes its result to obsm['spatial_rigid'], then obsm['spatial_elas'], and to
    # obsm['spatial_elas_reuse'] when binning ran (which is the default). The string
    # "st_gears_xyz" appears nowhere in the upstream package -- it was this worker's own
    # construction, built from obs['x'] and obs['y'], which are INPUT columns. On a slide carrying
    # them it therefore recorded the UNALIGNED coordinates with a rank z under a name that reads
    # like the alignment; on a slide without them it silently wrote nothing, inside a try that only
    # logged. A key that looks like the answer and is not is worse than an absent one.
    #
    # The z is still the slice ordinal, because ST-GEARS is given no physical spacing and cannot
    # invent one. It is labelled as an ordinal in the payload so a consumer can refuse to build a
    # metric 3D neighbour graph on it.
    #
    # Built per section, before the concat, because the opt-in nearest fill needs that section's
    # bins. Spots linear interpolation could not place stay NaN unless the caller allowed the fill.
    aligned_key = ""
    unaligned_by_section = {}
    filled_by_section = {}
    for pos, i in enumerate(regis_ilist):
        sl = slicesl_aligned[pos]
        xyz, source, aligned_key = build_frame(
            sl, float(i), com=registered[pos] if binning_on else None, allow_nearest=allow_nearest
        )
        sl.obsm["spatial_3d_aligned"] = xyz
        # Kept, and holding the same values, because something may already read it.
        sl.obsm["st_gears_xyz"] = xyz.copy()
        sl.obs["st_gears_z"] = np.full(sl.n_obs, float(i))
        sl.obs["st_gears_xy_source"] = source.astype(str)
        unaligned_by_section[str(slice_ids[i])] = int((source == "none").sum())
        filled_by_section[str(slice_ids[i])] = int((source == "nearest").sum())

    restore_xy(slicesl_aligned, [original_xy.get(i) for i in regis_ilist])

    # Concatenate aligned slices into one AnnData
    _log("Concatenating aligned sections into a single AnnData ...")
    aligned_ids = [str(slice_ids[i]) for i in regis_ilist]
    try:
        aligned = ad.concat(
            slicesl_aligned,
            merge="first",
            label=args.slice_key,
            keys=aligned_ids,
        )
    except Exception as e:
        # No silent substitute: the old fallback stacked X and obs only and so wrote a file with no
        # aligned coordinates at all, under status "ok".
        _log("anndata.concat failed:")
        _log("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        WorkerOutput.emit_error(
            "st_gears",
            f"anndata.concat of the aligned sections failed: {type(e).__name__}: {e}",
            task="3d_reconstruction",
        )
        sys.exit(1)
    aligned.obs["st_gears_xy_source"] = aligned.obs["st_gears_xy_source"].astype("category")

    n_unaligned = int(sum(unaligned_by_section.values()))
    n_filled = int(sum(filled_by_section.values()))
    n_aligned_spots = int(aligned.n_obs)
    z_note = (
        f"aligned coordinates read from obsm[{aligned_key!r}], ST-GEARS's own output key. "
        f"The third column is the slice ORDINAL, not a distance: ST-GEARS is given no "
        f"section spacing and cannot supply one, so a metric 3D neighbour graph must not "
        f"be built on it without a declared spacing."
    )
    _log(z_note)

    # Save outputs
    output_h5ad = os.path.join(output_dir, "st_gears_aligned.h5ad")
    meta_json = os.path.join(output_dir, "st_gears_run_metadata.json")

    _log(f"Writing aligned AnnData to {output_h5ad}")
    _write_atomic_h5ad(aligned, output_h5ad)

    out = WorkerOutput("st_gears", task="3d_reconstruction")
    if z_note:
        out.add_warning(z_note)
    record_in_tissue(out, n_spots_input, n_off_tissue)
    method = "ST-GEARS serial_align + stack_slices_pairwise_rigid + stack_slices_pairwise_elas_field"
    if binning_on:
        method += f" on bin_step={args.bin_step} bins, interpolated back to spots (st_gears.interpolate, linear)"
    if n_filled:
        record_method(
            out,
            method + f"; {n_filled} spots outside the bins' convex hull placed by the nearest bin's displacement",
            used_fallback=True,
            why="allow_nearest_fallback=True; obs['st_gears_xy_source'] == 'nearest' marks them",
        )
    else:
        record_method(out, method)
    if n_unaligned:
        if aligned_key == "spatial_elas_reuse":
            nan_cause = (
                "they lie outside the convex hull of the binned grid, where ST-GEARS's linear interpolation "
                "(st_gears.interpolate) is undefined."
                + ("" if allow_nearest else " allow_nearest_fallback=True gives them the nearest bin's displacement.")
            )
        else:
            # No interpolation ran (binning off), so the NaN came from ST-GEARS's own field or stacking step,
            # and the nearest-bin fill has no bins to draw on.
            nan_cause = f"ST-GEARS itself wrote NaN for them in obsm[{aligned_key!r}] (binning was off)."
        out.add_warning(
            f"{n_unaligned} of {n_aligned_spots} spots have no aligned coordinate (NaN rows in spatial_3d_aligned, "
            f"st_gears_xyz and {aligned_key}; obs['st_gears_xy_source'] == 'none'), per section "
            f"{unaligned_by_section}: " + nan_cause
        )
    if z_column_added:
        out.add_warning(
            "obsm['spatial'] had 2 columns and ST-GEARS indexes a third (recons.py:150, granularity_adjusting.py:102), "
            "so its working copy carried the slice ordinal as z; the third column of "
            + ("spatial_elas_reuse" if binning_on else "spatial_rigid / spatial_elas")
            + " in the output is that ordinal. obsm['spatial'] in the output is the 2-column input, unchanged."
        )
    if allow_nearest and not binning_on:
        record_ignored(
            out,
            "allow_nearest_fallback",
            "binning_on=False: no spot is interpolated from bins, so there are no bins to fill from",
        )
    if n_no_sid or n_unlabeled:
        out.add_warning(
            f"drop_unlabeled=True left out {n_no_sid} spots with no section id and {n_unlabeled} spots with no "
            f"'{args.group_key}' label; they are not in the output."
        )
    out.add_param("aligned_obsm_key", aligned_key or "")
    out.set_data(n_slices=len(slice_ids), n_spots=n_spots_input, n_genes=n_genes)
    out.add_output_files(
        {
            "aligned_h5ad": output_h5ad,
            "run_metadata_json": meta_json,
        }
    )
    out.add_params(
        {
            "input_h5ad": st_h5ad,
            "slice_key": args.slice_key,
            "group_key": args.group_key,
            "binning_on": binning_on,
            "bin_step": args.bin_step,
            "uniform_weight": uniform_weight,
            "filter_by_label": filter_by_label,
            "tune_alpha_li": tune_alpha_li,
            "numItermax": args.num_itermax,
            "fil_pc": args.fil_pc,
            "pixel_size": pixel_size,
            "pixel_size_source": pixel_size_source,
            "sigma": args.sigma,
            "start_i": start_i,
            "end_i": end_i,
            "use_gpu": use_gpu,
            "seed": args.seed,
            "drop_unlabeled": allow_drop,
            "allow_nearest_fallback": allow_nearest,
            "z_column_added": z_column_added,
            "slice_order": slice_order,
        }
    )
    out.set_summary(
        slice_ids=[str(s) for s in slice_ids],
        regis_ilist=list(map(int, regis_ilist)),
        slice_ids_aligned=aligned_ids,
        n_slices_aligned=len(aligned_ids),
        n_spots_aligned=n_aligned_spots,
        n_spots_dropped_unlabeled=int(n_no_sid + n_unlabeled),
        n_spots_off_tissue_dropped=int(n_off_tissue),
        n_spots_without_aligned_xy=n_unaligned,
        n_spots_filled_nearest=n_filled,
        unaligned_by_section=unaligned_by_section,
        bins_per_section=bins_per_section,
        alpha_per_pair=[float(a) for a in alphali],
    )

    skipped = [str(s) for j, s in enumerate(slice_ids) if j not in set(regis_ilist)]
    analysis = (
        f"ST-GEARS 3D reconstruction completed for {len(aligned_ids)} of {len(slice_ids)} sections "
        f"({', '.join(aligned_ids)}); registration pairs: {len(pili)}. "
        f"{n_aligned_spots} spots written"
    )
    if n_unaligned:
        analysis += f", {n_unaligned} of them without an aligned coordinate" + (
            " (outside the binned grid)" if aligned_key == "spatial_elas_reuse" else f" (NaN in {aligned_key})"
        )
    if n_filled:
        analysis += f", {n_filled} placed by the nearest bin's displacement (allow_nearest_fallback)"
    analysis += "."
    if n_off_tissue:
        analysis += (
            f" {n_off_tissue} of the {n_spots_input} supplied spots have obs['in_tissue'] == 0 (background) and "
            "were left out."
        )
    if skipped:
        analysis += f" Sections outside start_idx..end_idx were not aligned and are not in the output: {skipped}."
    if binning_on:
        analysis += f" Binning (bin_step={args.bin_step}) turned the spots into {bins_per_section} bins per section."
    analysis += f" Elastic field pixel_size={pixel_size:g} ({pixel_size_source}). Output: {output_h5ad}."
    out.set_analysis(analysis)

    _write_atomic_json(out.to_dict(), meta_json)

    out.emit()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # safety net
        _log("UNCAUGHT EXCEPTION in st_gears_worker:")
        _log("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        WorkerOutput.emit_error("st_gears", str(e), task="3d_reconstruction")
        sys.exit(1)
