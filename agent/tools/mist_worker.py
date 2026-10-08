#!/usr/bin/env python
"""
MIST / ReST worker script for SpatialOmicsLab MCP integration.

This script runs inside the ReST environment (/opt/conda/envs/ReST) and does the
heavy lifting: loading an AnnData .h5ad spatial transcriptomics file, converting
it to the input format expected by MIST (ReST), running preprocessing,
region extraction, and optional imputation, and writing results to disk.

What MIST needs from the coordinates
------------------------------------
ReST builds its spot graph with ``neighbors.construct_graph(radius=2)``: two spots are
neighbours when their Euclidean distance in *coordinate units* is below 2. That is written for
the Visium array grid (``obs['array_row']`` / ``obs['array_col']``), where the hexagonal
neighbours sit at distance sqrt(2). Pixel coordinates (``obsm['spatial']``, tens to hundreds of
pixels apart) put every spot outside every other spot's radius, so region detection labels
every spot ``isolated`` and spKNN imputation has no neighbours to average. This worker therefore
reads the array grid first, checks the spacing of whatever it chose, and refuses coordinates
MIST cannot connect unless the caller opts in to a rescale (``--allow-coordinate-rescale-fallback``).

What runs
---------
Region detection is ReST's ``extract_regions``. Imputation is ReST's ``impute`` default, spKNN: each
zero is replaced by the mean of the spot's grid neighbours. MIST's region-wise ensemble imputer
(MIST2) is not run, so ``n_cores`` / ``n_experts`` -- which only it reads -- are accepted and
reported under ``params.ignored``, as is ``filter_spot`` (ReST's QC has no switch). Spots flagged
``obs['in_tissue'] == 0`` (background glass, which CELLxGENE Visium exports carry beside the tissue)
are left out before ReST sees the slide, and counted in ``params.in_tissue_filter``. Everything the
tables cover is after that and after ReST's QC, which drops spots with ``pct_counts_mt >= 25`` or no
counts; ``data.n_spots`` is the count supplied and ``data.n_spots_used`` says how many are left.

ReST.preprocess filters spots by their total count and CPM-normalises + log2-transforms the matrix it
is handed, i.e. it treats it as counts. The matrix is checked first (the shared rule of
``worker_utils.choose_counts_matrix``): a negative or non-finite matrix (scaled / z-scored data) is
refused, naming ``--use-raw-counts`` when ``adata.raw`` holds the counts; a non-negative non-integer one
(already normalised or log-transformed) runs as before with a warning; ``--use-raw-counts`` hands ReST
``adata.raw.X``. ``params.expression_source`` / ``params.x_matrix_kind`` say which matrix ran.

``seed`` is accepted and listed under ``params.ignored``: nothing on the path this worker runs
(``preprocess`` -- scanpy's PCA at its own fixed random_state 0 --, ``extract_regions``, spKNN
``impute``) draws from numpy's global random state, and the only seeded helpers in ReST's ``utils``
hard-code their own seeds (2021, 2022).

IMPORTANT:
- All progress / logging is printed to stderr (prefixed with [mist-worker]).
- The ONLY thing printed to stdout is a single line of JSON with a summary
  of the run, so that the FastMCP wrapper can parse it safely.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import traceback

import numpy as np
import pandas as pd
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    choose_counts_matrix,
    describe_reduction,
    expression_matrix_kind,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    spatial_coords,
    unsupported_choice_msg,
)

try:
    import anndata as ad  # type: ignore
    from scipy import sparse as sp  # type: ignore
except Exception as e:  # pragma: no cover
    print("[mist-worker] ERROR: anndata and scipy are required inside the ReST environment.", file=sys.stderr)
    print(str(e), file=sys.stderr)
    # Emit JSON error and exit
    WorkerOutput.emit_error("mist", f"anndata/scipy import failed: {e}")
    sys.exit(1)


def log(msg: str) -> None:
    """Log to stderr with a consistent prefix."""
    print(f"[mist-worker] {msg}", file=sys.stderr)


# -----------------------------------------------------------------------------
# Make sure the cloned MIST repo is importable
# Expected layout in your setup:
#   tools/third_party/MIST/ReST.py
# -----------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MIST_REPO_DIR: str | None = None
_candidate_dirs = [
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "MIST")),
    os.path.abspath(os.path.join(SCRIPT_DIR, "MIST")),
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "tools", "MIST")),
    os.path.join(SCRIPT_DIR, "third_party", "MIST"),
]
# MIST_SRC wins over all of them: the clone-relative entries only find a checkout laid out beside
# this repo, and the two absolutes are where it sat on the machine this was written on.
if os.environ.get("MIST_SRC"):
    _candidate_dirs.insert(0, os.environ["MIST_SRC"])

for d in _candidate_dirs:
    if os.path.isdir(d):
        if d not in sys.path:
            sys.path.insert(0, d)
        MIST_REPO_DIR = d
        break

if MIST_REPO_DIR is None:
    MIST_REPO_DIR = "<not_found>"

log(f"Using MIST repo dir: {MIST_REPO_DIR}")

try:
    # ReST's modules print their progress (and, at import, package versions) with bare print(); the
    # worker's stdout carries only the JSON result, so every ReST call runs with stdout on stderr.
    with contextlib.redirect_stdout(sys.stderr):
        from ReST import ReST  # type: ignore
except Exception as e:  # pragma: no cover
    log("ERROR: Could not import ReST from MIST repo.")
    log(f"Searched repo dir: {MIST_REPO_DIR}")
    log(str(e))
    WorkerOutput.emit_error("mist", f"ReST import failed: {e}")
    sys.exit(1)


# ReST.extract_regions(radius=2) and imputers.spKNN -> neighbors.construct_graph(radius=2): a
# neighbour is a spot at Euclidean distance strictly below this, in coordinate units.
NEIGHBOR_RADIUS = 2.0
# Where the opt-in rescale puts the median nearest-neighbour distance: the spacing of the Visium
# array grid's diagonal neighbours, which is the neighbourhood MIST was written against. Spots at
# that distance fall inside the radius; the second ring (sqrt(3) further out, ~2.45) falls outside.
GRID_TARGET_SPACING = float(np.sqrt(2.0))
VALID_SPECIES = ("Human", "Mouse")
# The imputer this worker runs. ReST.impute's default, and the only one it has ever called: a
# baseline that fills each zero with the mean of the spot's grid neighbours (radius 2, the spot
# itself included). MIST's region-wise rank-minimisation ensemble (MIST2) is NOT what runs, so the
# payload names spKNN and n_cores / n_experts -- which only MIST2 reads -- are reported as ignored.
IMPUTE_METHOD = "spKNN"
ISOLATED = "isolated"
REGION_COLUMN = "region_ind"  # the obs column ReST.extract_regions writes (ReST.py, adata.obs['region_ind'])
# The expression matrix is handed to ReST as float32: anndata 0.8 (the ReST env) casts an AnnData X
# to float32 on construction, which is what the counts-DataFrame constructor this worker used to
# call produced, so the numbers are unchanged and no second float64 copy is made.
EXPRESSION_DTYPE = np.float32
# Dense copies ReST itself makes, none of which this wrapper can avoid (see _rest_dense_bytes).
SPOTS_X_GENES_COPIES = 4
SPOTS_X_SPOTS_COPIES = 4
DEFAULT_SPATIAL_KEY = "spatial"


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _load_adata(path: str) -> ad.AnnData:
    """Load AnnData from path; raise on error with helpful message."""
    log(f"Loading AnnData from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"h5ad file not found: {path}")
    adata = ad.read_h5ad(path)
    log(f"Loaded AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    return adata


def _write_atomic_csv(df: pd.DataFrame, path: str, **to_csv_kwargs) -> None:
    """Write ``df`` to ``path`` through a ``.partial`` sibling so a killed run leaves no half file."""
    tmp = path + ".partial"
    df.to_csv(tmp, **to_csv_kwargs)
    os.replace(tmp, path)


def _write_atomic_text(text: str, path: str) -> None:
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _median_nn_distance(xy: np.ndarray) -> float:
    """Median distance from each point to its nearest other point (inf for fewer than 2 points)."""
    xy = np.asarray(xy, dtype=float)
    if xy.shape[0] < 2:
        return float("inf")
    from scipy.spatial import cKDTree  # type: ignore

    dist, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(dist[:, 1]))


def _grid_frame(x: np.ndarray, y: np.ndarray, index) -> pd.DataFrame:
    """The integer ``array_row`` / ``array_col`` frame ReST reads (x = column, y = row).

    Values must be ints so that when MIST concatenates them like "x x y" and later does
    ``int(c.split("x")[0])`` it does not see '8346.0'.
    """
    return pd.DataFrame(
        {
            "array_row": np.round(np.asarray(y, dtype=float)).astype(int),
            "array_col": np.round(np.asarray(x, dtype=float)).astype(int),
        },
        index=index,
    )


def _refuse_unconnected(source: str, median_nn: float) -> ValueError:
    return ValueError(
        f"MIST cannot connect these spots: the median nearest-neighbour distance of {source} is "
        f"{median_nn:.3g} coordinate units, and ReST only links spots closer than {NEIGHBOR_RADIUS:g} "
        "(neighbors.construct_graph radius=2, written for the Visium array grid where neighbours sit "
        "at sqrt(2)). Every spot would be 'isolated', no region would be found and spKNN would have "
        "nothing to average. Supply the array grid in obs['array_row'] / obs['array_col'] (Visium "
        "tissue_positions), or pass allow_coordinate_rescale_fallback=True to have the worker divide "
        f"the coordinates so that the median neighbour distance becomes {GRID_TARGET_SPACING:.3g} and "
        "round them to integers -- the rescale is recorded in params.coordinate_source and "
        "params.used_fallback."
    )


def _finalize_grid(
    x: np.ndarray,
    y: np.ndarray,
    index,
    source: str,
    allow_rescale: bool,
) -> tuple:
    """Round a coordinate pair to MIST's integer grid, checking that MIST can connect it.

    Returns ``(frame, info)`` where ``info`` carries ``coordinate_source``, ``median_nn_distance``
    (of the coordinates handed to MIST), ``coordinate_scale`` (None unless rescaled) and
    ``used_fallback``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_bad = int((~np.isfinite(x) | ~np.isfinite(y)).sum())
    if n_bad:
        raise ValueError(
            f"{n_bad} of {len(x)} spots have no finite coordinate in {source} (NaN or inf). Rounding "
            "them to an integer grid position would invent one; drop those spots or supply their "
            "positions."
        )
    df = _grid_frame(x, y, index)
    median_nn = _median_nn_distance(df[["array_col", "array_row"]].values)
    info = {
        "coordinate_source": source,
        "median_nn_distance": median_nn if np.isfinite(median_nn) else None,
        "coordinate_scale": None,
        "used_fallback": False,
    }
    if np.isfinite(median_nn) and median_nn >= NEIGHBOR_RADIUS:
        if not allow_rescale:
            raise _refuse_unconnected(source, median_nn)
        raw_nn = _median_nn_distance(np.c_[x, y])
        if not raw_nn > 0:
            raise _refuse_unconnected(source, median_nn)
        scale = GRID_TARGET_SPACING / raw_nn
        x_s = (x - x.min()) * scale
        y_s = (y - y.min()) * scale
        df = _grid_frame(x_s, y_s, index)
        median_nn_after = _median_nn_distance(df[["array_col", "array_row"]].values)
        log(
            f"Rescaled {source} by {scale:.6g} (median nearest-neighbour distance {raw_nn:.4g} -> "
            f"{median_nn_after:.4g} after integer rounding) because the caller allowed it."
        )
        info = {
            "coordinate_source": (
                f"{source} rescaled x{scale:.6g} so the median neighbour distance is "
                f"{GRID_TARGET_SPACING:.3g}, then rounded to integers"
            ),
            "median_nn_distance": median_nn_after,
            "coordinate_scale": float(scale),
            "used_fallback": True,
        }
        if median_nn_after >= NEIGHBOR_RADIUS:
            raise _refuse_unconnected(info["coordinate_source"], median_nn_after)
    n_dup = int(df.duplicated(subset=["array_row", "array_col"]).sum())
    if n_dup:
        raise ValueError(
            f"{n_dup} of {len(df)} spots share a grid position after integer rounding of "
            f"{info['coordinate_source']}. MIST keys every spot on the string 'col x row', so those "
            "spots would be merged into one and their results silently attributed to the wrong "
            "barcode. Supply coordinates that are distinct as integers."
        )
    return df, info


def _extract_spatial_coords(
    adata: ad.AnnData,
    spatial_key: str = DEFAULT_SPATIAL_KEY,
    allow_rescale: bool = False,
) -> tuple:
    """
    The integer grid MIST reads, and where it came from.

    Priority:
    1. adata.obs[['array_row', 'array_col']] -- the Visium array grid MIST's radius-2 graph was
       written for. Preferred whenever both columns are present.
    2. adata.obsm[spatial_key] -- only when the grid columns are absent. Pixel coordinates are
       refused (see ``_refuse_unconnected``) unless ``allow_rescale`` is True.

    Returns a DataFrame with index = obs_names and columns ['array_row', 'array_col'], plus the
    info dict of :func:`_finalize_grid`.
    """
    # 1) obs['array_row'] / obs['array_col'] branch
    if "array_row" in adata.obs.columns and "array_col" in adata.obs.columns:
        log("Reading coordinates from obs['array_row'], obs['array_col'] (the Visium array grid MIST expects).")
        return _finalize_grid(
            np.asarray(adata.obs["array_col"], dtype=float),
            np.asarray(adata.obs["array_row"], dtype=float),
            adata.obs_names,
            "obs['array_row','array_col']",
            allow_rescale,
        )

    # 2) obsm[spatial_key] branch
    if spatial_key in adata.obsm:
        # MIST keys its spots on the literal string "x x y", so a third axis has nowhere to go:
        # two sections at the same x/y and different z would collide on one key and silently merge.
        coords, _ = spatial_coords(adata, spatial_key, want=2, tool="MIST")
        log(
            f"obs['array_row'/'array_col'] are absent; reading coordinates from obsm['{spatial_key}'] "
            f"with shape {coords.shape}."
        )
        return _finalize_grid(coords[:, 0], coords[:, 1], adata.obs_names, f"obsm['{spatial_key}']", allow_rescale)

    raise ValueError(
        "Could not find spatial coordinates: neither obs['array_row'/'array_col'] nor "
        f"obsm['{spatial_key}'] are present. Available obsm keys: {list(adata.obsm.keys())}."
    )


def _missing_layer(adata: ad.AnnData, layer_key: str) -> KeyError:
    return KeyError(
        f"layer_key='{layer_key}' is not in adata.layers (available layers: "
        f"{list(adata.layers.keys())}). Pass one of them, or omit layer_key to use adata.X."
    )


def _choose_counts(adata: ad.AnnData, layer_key: str | None, use_raw_counts: bool) -> tuple:
    """``(adata, info)``: the object whose counts ReST will read, checked before ReST normalises them.

    ReST.preprocess drops spots below a total count, CPM-normalises and log2-transforms whatever it is
    handed. A scaled matrix (negative values) failed opaquely inside ReST, and a log-normalised one was
    normalised a second time with status ok. X and ``adata.raw`` follow ``worker_utils.choose_counts_matrix``
    (``use_raw_counts=True`` swaps in ``adata.raw.X``); a ``layer_key`` layer is held to the same rule by
    hand. ``use_raw_counts`` and ``layer_key`` both name the matrix, so the pair is refused rather than
    one silently winning.
    """
    if not layer_key:  # None or "" means adata.X, as in the portal (which forwards only a non-empty key)
        return choose_counts_matrix(adata, use_raw_counts)
    if use_raw_counts:
        raise ValueError(
            f"use_raw_counts=True and layer_key='{layer_key}' both name the matrix ReST should read "
            "(adata.raw.X and adata.layers['" + str(layer_key) + "']). Pass one of them."
        )
    if layer_key not in adata.layers:
        raise _missing_layer(adata, layer_key)
    kind = expression_matrix_kind(adata.layers[layer_key])
    where = f"layers['{layer_key}']"
    if kind in ("negative", "nonfinite"):
        what = "negative values (scaled or z-scored data)" if kind == "negative" else "NaN or infinite values"
        raise ValueError(
            f"adata.{where} holds {what}, not counts, and ReST normalises the matrix it is given as counts "
            "(CPM, then log2). Pass a layer of raw counts, or omit layer_key to use X (or adata.raw with "
            "use_raw_counts=True)."
        )
    warning = None
    if kind == "nonnegative_noninteger":
        warning = (
            f"adata.{where} holds non-integer values (normalised or log-transformed data?), and ReST normalises "
            "the matrix it is given as counts (CPM, then log2), so the result was computed on a matrix normalised "
            "twice. Pass a layer of raw counts for a single normalisation."
        )
    return adata, {"expression_source": where, "x_matrix_kind": kind, "warning": warning}


def _extract_expression(adata: ad.AnnData, layer_key: str | None = None) -> tuple:
    """
    The expression matrix ReST is given, as CSR, and the name of where it came from.

    ``layer_key`` must name an existing layer: a missing one is an error, not a silent switch to
    ``adata.X`` (the caller asked for a layer and would otherwise read results about another one).
    """
    if layer_key:  # None or "" means adata.X, as in the portal (which forwards only a non-empty key)
        if layer_key not in adata.layers:
            raise _missing_layer(adata, layer_key)
        X = adata.layers[layer_key]
        source = f"layers['{layer_key}']"
    else:
        X = adata.X
        source = "X"
    log(f"Using adata.{source} as expression matrix.")
    # ReST(adata=...) does csr_matrix(adata.X) itself; handing it a CSR avoids the dense spots x
    # genes DataFrame its counts constructor needed.
    if sp.issparse(X):
        X = sp.csr_matrix(X, dtype=EXPRESSION_DTYPE)
    else:
        X = sp.csr_matrix(np.asarray(X, dtype=EXPRESSION_DTYPE))
    return X, source


def _build_rest(coords_df: pd.DataFrame, X, var_names) -> ReST:
    """The ReST object, built through its AnnData constructor so X stays sparse."""
    gene_df = pd.DataFrame({"gene": list(var_names)}, index=pd.Index(var_names))
    rest_adata = ad.AnnData(X=X, obs=coords_df, var=gene_df)
    log(f"Constructing ReST object from a sparse {rest_adata.n_obs} x {rest_adata.n_vars} matrix...")
    return ReST(adata=rest_adata)


def _rest_dense_bytes(n_obs: int, n_vars: int, run_impute: bool) -> int:
    """Peak bytes of the dense matrices ReST allocates itself, for a spots x genes input.

    None of these can be avoided from the wrapper -- they are ReST's own code:

    * ``preprocess``: ``sc.pp.scale`` densifies X (spots x genes), the Spearman similarity
      ``raw_weights`` is a float64 spots x spots matrix, and above 10,000 genes the gene filter
      densifies X once more before it runs.
    * ``extract_regions``: ``adata.copy()`` (another dense X and ``raw_weights``) and a dense
      highly-variable slice.
    * ``impute`` (spKNN): ``shallow_copy`` (X and ``raw_weights`` again), the CPM layer densified,
      its copy, a copy of ``raw_weights`` and a float64 spots x spots neighbour matrix.

    Counted as ``SPOTS_X_GENES_COPIES`` float32 spots x genes plus ``SPOTS_X_SPOTS_COPIES`` float64
    spots x spots live at once (two fewer of each without imputation), with ``n_vars`` the genes
    supplied -- ReST's QC can only lower it.
    """
    n, g = int(n_obs), int(n_vars)
    k_ng = SPOTS_X_GENES_COPIES if run_impute else SPOTS_X_GENES_COPIES - 2
    k_nn = SPOTS_X_SPOTS_COPIES if run_impute else SPOTS_X_SPOTS_COPIES - 2
    steady = k_ng * n * g * np.dtype(EXPRESSION_DTYPE).itemsize + k_nn * n * n * 8
    gene_filter = n * g * (np.dtype(EXPRESSION_DTYPE).itemsize + 1) if g > 10000 else 0
    return int(max(steady, gene_filter + 2 * n * n * 8))


def _available_memory_bytes():
    """Memory this process can still allocate, or None when nothing can be read.

    The fleet's one reader (``worker_utils.available_memory_bytes``): the smaller of MemAvailable and
    the ROOM under the cgroup limit, page cache counted as reclaimable. The reader this replaces took
    the cgroup limit itself as what was available, so a container already using most of its limit was
    told the whole limit was free and the run was OOM-killed instead of refused with the numbers.
    """
    return available_memory_bytes()


def _check_memory(n_obs: int, n_vars: int, run_impute: bool, task: str, available=None) -> tuple:
    """Refuse, naming the numbers, when ReST's own dense matrices cannot fit. Never subsamples."""
    need = _rest_dense_bytes(n_obs, n_vars, run_impute)
    if available is None:
        available = _available_memory_bytes()
    if available is not None and need > available:
        gib = float(1 << 30)
        knob = (
            " task='regions' skips the imputation and its copies, which lowers the estimate to "
            f"about {_rest_dense_bytes(n_obs, n_vars, False) / gib:.1f} GiB."
            if run_impute
            else ""
        )
        raise MemoryError(
            f"MIST/ReST densifies this input itself: {n_obs} spots x {n_vars} genes needs about "
            f"{need / gib:.1f} GiB of dense matrices for task='{task}' (float32 spots x genes copies and "
            f"float64 {n_obs} x {n_obs} spot-similarity and neighbour matrices), but about "
            f"{available / gib:.1f} GiB is available here (MemAvailable / cgroup limit).{knob} This "
            "worker never subsamples spots; run it where that much memory is available, or one "
            "section at a time if the input holds several."
        )
    return need, available


def _region_labels(adata_rest, obs_cols_before) -> pd.Series:
    """The region column ReST.extract_regions writes, as strings. Never a pre-existing column.

    ReST writes ``obs['region_ind']`` (``'isolated'`` for a spot in no region). The earlier
    heuristic -- "any new column with 2..n-1 distinct values, else any existing column named
    region/cluster/domain" -- found nothing when every spot was isolated and let the run finish
    without a region column at all.
    """
    new_cols = [c for c in adata_rest.obs.columns if c not in obs_cols_before]
    log(f"New obs columns after region extraction: {new_cols}")
    if REGION_COLUMN not in adata_rest.obs.columns:
        raise RuntimeError(
            f"ReST.extract_regions did not write obs['{REGION_COLUMN}'] (new obs columns: {new_cols}); "
            "the installed MIST checkout is not the API this worker was written against."
        )
    return adata_rest.obs[REGION_COLUMN].astype(str)


def _zero_fraction(matrix) -> float:
    """Fraction of exactly-zero entries, without densifying a sparse matrix."""
    if sp.issparse(matrix):
        total = float(matrix.shape[0]) * float(matrix.shape[1])
        return float(1.0 - matrix.count_nonzero() / total) if total else 0.0
    arr = np.asarray(matrix)
    return float(1.0 - np.count_nonzero(arr) / arr.size) if arr.size else 0.0


def _canonical_species(species: str) -> str:
    """'human' / 'HUMAN' -> 'Human'. ReST compares the exact string and otherwise crashes in QC."""
    for valid in VALID_SPECIES:
        if str(species).strip().lower() == valid.lower():
            return valid
    raise ValueError(
        unsupported_choice_msg(
            "species", species, VALID_SPECIES, "ReST.preprocess only knows the MT-/mt- gene prefixes of these two."
        )
    )


def run_mist(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = DEFAULT_SPATIAL_KEY,
    layer_key: str | None = None,
    species: str = "Human",
    hvg_prop: float = 0.8,
    n_pcs: int = 10,
    filter_spot: bool = True,
    min_sim: float = 0.1,
    min_size: int = 20,
    gap: float = 0.02,
    n_cores: int = 1,
    n_experts: int = 3,
    task: str = "all",
    seed: int = 0,
    allow_coordinate_rescale_fallback: bool = False,
    use_raw_counts: bool = False,
) -> dict:
    """
    Core driver: load AnnData, run ReST pipeline, save outputs, emit the JSON payload and
    return it as a dict.

    The counts matrix is chosen and checked before ReST normalises it (:func:`_choose_counts`).
    ``seed`` has no effect on this path (see the module docstring) and is reported as ignored.
    """
    species = _canonical_species(species)
    run_regions = task in ("all", "regions", "regions_impute")
    run_impute = task in ("all", "impute", "regions_impute")
    if not (run_regions or run_impute):
        raise ValueError(unsupported_choice_msg("task", task, ("all", "regions", "impute", "regions_impute")))

    os.makedirs(output_dir, exist_ok=True)
    # No np.random.seed(seed) here: nothing ReST runs below draws from numpy's global random state, so
    # the call changed nothing it was said to control. The seed is reported under params.ignored.

    out = WorkerOutput("mist", task=task)

    adata = _load_adata(st_h5ad)
    n_spots_in = int(adata.n_obs)
    # Background spots (obs['in_tissue'] == 0) are not tissue: ReST would link them into regions,
    # average them into the spKNN imputation of their tissue neighbours and pay n^2 memory for them.
    adata, _, n_off_tissue = keep_in_tissue(adata, "spots")
    n_spots_in_tissue = int(adata.n_obs)
    # ReST normalises the matrix it is given as counts; check it (or swap in adata.raw.X) first. The
    # names are made unique after the swap, so a raw var with duplicate symbols is counted too.
    adata, counts_info = _choose_counts(adata, layer_key, use_raw_counts)
    log(f"Expression matrix: adata.{counts_info['expression_source']} ({counts_info['x_matrix_kind']})")
    n_genes_in = int(adata.n_vars)
    renamed = make_names_unique_and_report(adata)
    if n_off_tissue:
        log(
            f"Left out {n_off_tissue} of {n_spots_in} spots with obs['in_tissue'] == 0 (background); "
            f"{n_spots_in_tissue} in-tissue spots go to ReST."
        )
    record_in_tissue(out, n_spots_in, n_off_tissue)
    coords_df, coord_info = _extract_spatial_coords(
        adata, spatial_key=spatial_key, allow_rescale=allow_coordinate_rescale_fallback
    )
    X, _ = _extract_expression(adata, layer_key=layer_key)
    mem_need, mem_available = _check_memory(n_spots_in_tissue, n_genes_in, run_impute, task)
    log(
        f"Estimated peak of ReST's dense matrices: {mem_need / float(1 << 30):.2f} GiB"
        + (f" (available: {mem_available / float(1 << 30):.1f} GiB)." if mem_available else ".")
    )

    mt_prefix = "MT-" if species == "Human" else "mt-"
    n_mt = int(pd.Index(adata.var_names).astype(str).str.startswith(mt_prefix).sum())
    if n_mt == 0:
        out.add_warning(
            f"no var_names start with '{mt_prefix}', so ReST's mitochondrial QC (spots with "
            "pct_counts_mt >= 25 removed) could not act. If var_names are Ensembl IDs, MIST reads "
            "gene symbols from var_names only."
        )

    with contextlib.redirect_stdout(sys.stderr):
        rd = _build_rest(coords_df, X, adata.var_names)

    # ReST.preprocess has no spot-filter switch: its QC (pct_counts_mt < 25) always runs.
    log(f"Running rd.preprocess(species='{species}', hvg_prop={hvg_prop}, n_pcs={n_pcs})")
    with contextlib.redirect_stdout(sys.stderr):
        rd.preprocess(species=species, hvg_prop=hvg_prop, n_pcs=n_pcs)
    n_spots_used, n_genes_used = int(rd.adata.n_obs), int(rd.adata.n_vars)
    log(f"After ReST QC: {n_spots_used} of {n_spots_in_tissue} spots and {n_genes_used} of {n_genes_in} genes remain.")
    spot_cuts = []
    if n_off_tissue:
        spot_cuts.append(
            f"the in-tissue filter ({n_off_tissue} spots with obs['in_tissue'] == 0, background outside the tissue)"
        )
    if n_spots_used < n_spots_in_tissue:
        spot_cuts.append(
            f"ReST QC ({n_spots_in_tissue - n_spots_used} spots with pct_counts_mt >= 25, or with no counts at all, "
            "removed)"
        )
    spot_note = describe_reduction("spots", n_spots_in, n_spots_used, " and ".join(spot_cuts))
    gene_note = describe_reduction(
        "genes",
        n_genes_in,
        n_genes_used,
        "ReST QC (with more than 10000 genes, genes with a count above 1 in at most one spot are removed)",
    )
    for note in (spot_note, gene_note):
        if note:
            out.add_warning(note.strip())
    if n_spots_used == 0:
        raise RuntimeError(
            f"ReST QC removed every one of the {n_spots_in_tissue} in-tissue spots (pct_counts_mt >= 25, or no "
            "counts); nothing is left to analyse."
        )

    region_col: str | None = None
    n_regions: int | None = None
    n_isolated: int | None = None
    region_sizes: dict | None = None
    region_csv = os.path.join(output_dir, "mist_region_assignments.csv")
    stats_txt = os.path.join(output_dir, "mist_region_stats.txt")
    imputed_csv = os.path.join(output_dir, "mist_imputed_expression.csv")

    # --- Region extraction ---
    if run_regions:
        log(f"Running rd.extract_regions(min_sim={min_sim}, min_size={min_size}, gap={gap})")
        obs_cols_before = list(rd.adata.obs.columns)
        with contextlib.redirect_stdout(sys.stderr):
            rd.extract_regions(min_sim=min_sim, min_size=min_size, gap=gap)
        labels = _region_labels(rd.adata, obs_cols_before)
        region_col = REGION_COLUMN
        counts = labels.value_counts()
        n_isolated = int(counts.get(ISOLATED, 0))
        region_sizes = {str(k): int(v) for k, v in counts.items() if str(k) != ISOLATED}
        n_regions = len(region_sizes)
        if n_regions == 0:
            raise RuntimeError(
                f"MIST found no spatial regions: all {n_spots_used} QC-passing spots were '{ISOLATED}' "
                f"(no connected component of at least min_size={min_size} spots survived the similarity "
                f"search between min_sim={min_sim} and 0.96 in steps of gap={gap}, on "
                f"{coord_info['coordinate_source']} with median neighbour distance "
                f"{coord_info['median_nn_distance']}). Lower min_size or min_sim, or check the coordinates."
            )
        if n_regions == 1:
            out.add_warning(
                f"MIST found a single region ({list(region_sizes.values())[0]} spots) plus {n_isolated} "
                "isolated spots; one region is not a partition of the tissue. Consider raising min_sim "
                "or lowering min_size."
            )

        df_region = coords_df.loc[rd.adata.obs.index].copy()
        df_region.insert(0, "spot_id", rd.adata.obs.index.astype(str))
        df_region["region"] = labels.values
        _write_atomic_csv(df_region, region_csv, index=False)
        log(f"Saved region assignments to {region_csv}")

        stats_lines = [
            f"st_h5ad: {st_h5ad}",
            f"region_column: {region_col}",
            f"coordinate_source: {coord_info['coordinate_source']}",
            f"n_spots_input: {n_spots_in}",
            f"n_spots_off_tissue_dropped: {n_off_tissue}",
            f"n_spots_after_qc: {n_spots_used}",
            f"n_regions: {n_regions}",
            f"n_isolated: {n_isolated}",
            "region_counts:",
            counts.to_string(),
            "",
        ]
        _write_atomic_text("\n".join(stats_lines), stats_txt)
        log(f"Wrote region stats to {stats_txt}")

    # --- Imputation ---
    imputed_shape: tuple | None = None
    before_sparsity: float | None = None
    after_sparsity: float | None = None
    n_nan_entries = 0

    if run_impute:
        log(f"Running rd.impute(method='{IMPUTE_METHOD}')")
        with contextlib.redirect_stdout(sys.stderr):
            imputed = rd.impute(method=IMPUTE_METHOD)

        # spKNN returns a frame indexed by obs_names (imputers.Imputer builds it from layers['CPM']).
        if isinstance(imputed, pd.DataFrame):
            missing = pd.Index(rd.adata.obs_names).difference(imputed.index)
            if len(missing):
                raise RuntimeError(
                    f"the imputed table has no row for {len(missing)} of the {rd.adata.n_obs} QC-passing "
                    f"spots (e.g. {list(missing[:3])}); its rows cannot be matched to the spot barcodes."
                )
            imputed_df = imputed.loc[rd.adata.obs_names]
        else:
            imputed_array = np.asarray(imputed, dtype=float)
            if imputed_array.shape != (rd.adata.n_obs, rd.adata.n_vars):
                raise RuntimeError(
                    f"rd.impute returned shape {imputed_array.shape}, not the {rd.adata.n_obs} x "
                    f"{rd.adata.n_vars} QC-passing matrix; the rows cannot be named."
                )
            imputed_df = pd.DataFrame(imputed_array, index=rd.adata.obs_names, columns=rd.adata.var_names)

        imputed_shape = (int(imputed_df.shape[0]), int(imputed_df.shape[1]))
        # spKNN averages over the neighbours whose similarity is above 0, the spot itself included; a
        # spot left with none (a NaN similarity row) gets NaN for every zero it had.
        nan_mask = imputed_df.isna().values
        n_nan_entries = int(nan_mask.sum())
        if n_nan_entries:
            out.add_warning(
                f"{n_nan_entries} imputed values in {int(nan_mask.any(axis=1).sum())} spots are NaN: spKNN "
                "found no neighbour with a positive similarity (not even the spot itself) to average. "
                "They are written as empty cells in mist_imputed_expression.csv."
            )
        # spKNN reads layers['CPM'] (imputers.py), so the honest "before" is the zero fraction of
        # that layer -- not of rd.adata.X, which sc.pp.scale has zero-centred.
        before_sparsity = _zero_fraction(rd.adata.layers["CPM"])
        after_sparsity = _zero_fraction(imputed_df.values)

        _write_atomic_csv(imputed_df, imputed_csv)
        log(f"Saved imputed expression matrix to {imputed_csv}")
        log(f"Zero fraction of the CPM layer before imputation: {before_sparsity:.4f}, after: {after_sparsity:.4f}")

    out.set_data(n_spots=n_spots_in, n_genes=n_genes_in, n_spots_used=n_spots_used, n_genes_used=n_genes_used)
    out.add_params(
        {
            "spatial_key": spatial_key,
            "layer_key": layer_key,
            "species": species,
            "hvg_prop": hvg_prop,
            "n_pcs": n_pcs,
            "min_sim": min_sim,
            "min_size": min_size,
            "gap": gap,
            "seed": seed,
            "allow_coordinate_rescale_fallback": allow_coordinate_rescale_fallback,
            "coordinate_source": coord_info["coordinate_source"],
            "median_nn_distance": coord_info["median_nn_distance"],
            "coordinate_scale": coord_info["coordinate_scale"],
            "neighbor_radius": NEIGHBOR_RADIUS,
            "use_raw_counts": bool(use_raw_counts),
        }
    )
    # params.expression_source names the matrix of the input file ReST read: "X", "raw.X"
    # (use_raw_counts) or "layers['<key>']" -- plus params.x_matrix_kind, what that matrix held.
    record_expression_source(out, counts_info)
    out.add_params(identifier_rename_params(renamed))
    method_parts = []
    if run_regions:
        method_parts.append("ReST.extract_regions")
    if run_impute:
        method_parts.append(f"ReST.impute(method='{IMPUTE_METHOD}')")
    record_method(
        out,
        "MIST/ReST: " + " + ".join(method_parts) + f" on {coord_info['coordinate_source']}",
        used_fallback=bool(coord_info["used_fallback"]),
        why=(
            "the coordinates had to be rescaled to MIST's integer grid before it could connect any spots"
            if coord_info["used_fallback"]
            else ""
        ),
    )
    record_ignored(
        out,
        ["filter_spot"],
        "ReST.preprocess has no spot-filter switch; its QC (pct_counts_mt < 25) always runs.",
    )
    record_ignored(
        out,
        ["n_cores", "n_experts"],
        (
            f"no imputation ran for task='{task}'."
            if not run_impute
            else "the imputation is ReST's spKNN, which is single-threaded and has no experts; both "
            "belong to MIST's region-wise ensemble imputer (MIST2), which this tool does not run."
        ),
    )
    record_ignored(
        out,
        ["seed"],
        "nothing on the path this tool runs draws from a random stream it could seed: ReST.preprocess (scanpy's "
        "PCA runs at its own fixed random_state=0), extract_regions and the spKNN imputation are deterministic, "
        "and the seeded helpers in ReST's utils hard-code their own seeds.",
    )
    if coord_info["coordinate_source"].startswith("obs[") and spatial_key != DEFAULT_SPATIAL_KEY:
        record_ignored(
            out,
            ["spatial_key"],
            "obs['array_row'/'array_col'] are present and are the grid MIST reads; obsm is read only without them.",
        )

    # Only the files THIS run wrote: a region table left by an earlier run in the same directory is
    # not this run's result.
    output_files = {}
    if run_regions:
        output_files["region_assignments_csv"] = region_csv
        output_files["region_stats_txt"] = stats_txt
    if run_impute:
        output_files["imputed_expression_csv"] = imputed_csv
    out.add_output_files(output_files)
    out.set_summary(
        region_column=region_col,
        n_regions=n_regions,
        n_isolated=n_isolated,
        region_sizes=region_sizes,
        imputed_shape=imputed_shape,
        impute_method=IMPUTE_METHOD if run_impute else None,
        before_sparsity=before_sparsity,
        after_sparsity=after_sparsity,
        n_nan_imputed=n_nan_entries if run_impute else None,
    )
    analysis_parts = []
    if n_regions is not None:
        analysis_parts.append(
            f"MIST identified {n_regions} spatial region(s) covering {n_spots_used - (n_isolated or 0)} of the "
            f"{n_spots_used} QC-passing spots; {n_isolated} spots are '{ISOLATED}' (in no region) and the "
            f"'{ISOLATED}' label is not a region."
        )
    if before_sparsity is not None and after_sparsity is not None:
        analysis_parts.append(
            f"ReST's spKNN imputation (each zero replaced by the mean of the spot's grid neighbours; not "
            f"MIST's ensemble imputer) changed the zero fraction of the CPM layer from {before_sparsity:.4f} "
            f"to {after_sparsity:.4f} over {imputed_shape[0]} spots x {imputed_shape[1]} genes."
        )
    if counts_info["expression_source"] == "raw.X":
        analysis_parts.append("ReST was handed the counts in adata.raw.X (use_raw_counts=True).")
    if counts_info.get("warning"):
        analysis_parts.append(counts_info["warning"])
    analysis = " ".join(analysis_parts) + spot_note + gene_note + identifier_rename_note(renamed)
    out.set_analysis(analysis)
    out.emit()
    return out.to_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MIST / ReST worker for SpatialOmicsLab MCP. "
            "Converts AnnData to ReST inputs, runs region detection and/or "
            "imputation, and writes outputs."
        )
    )
    parser.add_argument(
        "--st-h5ad",
        required=True,
        help="Path to spatial transcriptomics AnnData .h5ad file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write MIST outputs (will be created if missing).",
    )
    parser.add_argument(
        "--spatial-key",
        default="spatial",
        help=(
            "Key in adata.obsm read only when obs['array_row'/'array_col'] are absent (default: 'spatial'). "
            "MIST needs its integer grid; pixel coordinates are refused unless --allow-coordinate-rescale-fallback."
        ),
    )
    parser.add_argument(
        "--layer-key",
        default=None,
        help="Optional key in adata.layers to use as expression matrix; it must exist. Omit to use adata.X.",
    )
    parser.add_argument(
        "--species",
        default="Human",
        help="Species argument passed to rd.preprocess: 'Human' or 'Mouse', any case (default: Human).",
    )
    parser.add_argument(
        "--hvg-prop",
        type=float,
        default=0.8,
        help="Proportion of genes to keep as HVGs in rd.preprocess (default: 0.8).",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=10,
        help="Number of PCs for rd.preprocess (default: 10).",
    )
    parser.add_argument(
        "--no-filter-spot",
        action="store_true",
        help=(
            "Accepted and ignored: ReST.preprocess has no spot-filter switch (its pct_counts_mt < 25 QC "
            "always runs). The payload lists filter_spot under params.ignored."
        ),
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.1,
        help="Minimum region similarity for rd.extract_regions (default: 0.1).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=20,
        help="Minimum region size (number of spots) for rd.extract_regions (default: 20).",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.02,
        help="Gap parameter for rd.extract_regions controlling merging (default: 0.02).",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=1,
        help=(
            "Accepted and ignored: the imputation is ReST's single-threaded spKNN (MIST's ensemble "
            "imputer, which reads it, does not run). Listed under params.ignored (default: 1)."
        ),
    )
    parser.add_argument(
        "--n-experts",
        type=int,
        default=3,
        help=(
            "Accepted and ignored: spKNN has no experts (MIST's ensemble imputer, which reads it, does "
            "not run). Listed under params.ignored (default: 3)."
        ),
    )
    parser.add_argument(
        "--allow-coordinate-rescale-fallback",
        action="store_true",
        help=(
            "Allow coordinates MIST cannot connect (median neighbour distance >= 2, e.g. pixels) to be "
            "rescaled so the median neighbour distance is sqrt(2), then rounded to integers. Off by default: "
            "the run refuses instead. Recorded in params.coordinate_source and params.used_fallback."
        ),
    )
    parser.add_argument(
        "--task",
        choices=["all", "regions", "impute", "regions_impute"],
        default="all",
        help=(
            "What to run: 'regions' (only region detection), "
            "'impute' (only imputation), "
            "'regions_impute' or 'all' (both; same behavior)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Accepted and ignored: nothing ReST runs here draws from a seedable random stream (its PCA uses "
            "scanpy's fixed random_state=0). Listed under params.ignored (default: 0)."
        ),
    )
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        default=False,
        help=(
            "Hand ReST adata.raw.X instead of X (for a CELLxGENE-style h5ad whose X is normalised or scaled and "
            "whose counts sit in adata.raw). Cannot be combined with --layer-key. Without it, a negative or "
            "non-finite matrix is refused and a non-integer one runs with a warning."
        ),
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()

        filter_spot = not args.no_filter_spot

        run_mist(
            st_h5ad=args.st_h5ad,
            output_dir=args.output_dir,
            spatial_key=args.spatial_key,
            layer_key=args.layer_key,
            species=args.species,
            hvg_prop=args.hvg_prop,
            n_pcs=args.n_pcs,
            filter_spot=filter_spot,
            min_sim=args.min_sim,
            min_size=args.min_size,
            gap=args.gap,
            n_cores=args.n_cores,
            n_experts=args.n_experts,
            task=args.task,
            seed=args.seed,
            allow_coordinate_rescale_fallback=args.allow_coordinate_rescale_fallback,
            use_raw_counts=args.use_raw_counts,
        )

    except Exception as e:
        log("ERROR in mist_worker: " + str(e))
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("mist", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
