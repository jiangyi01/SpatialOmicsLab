#!/usr/bin/env python
"""
SpatialScope worker for SpatialOmicsLab MCP.

Runs SpatialScope's CPU-runnable Cell-Type Identification (CTI / WarmStart)
pipeline — the same path the official benchmark uses for spot-level
deconvolution. Stage-2 diffusion decomposition is GPU-only and out of
scope here; CTI alone produces the standard per-spot proportions.

Pipeline:
  1. Load scRNA-seq reference and spatial AnnData; rename duplicated
     barcodes / genes (worker_utils.make_names_unique_and_report, published
     as params n_*_renamed); leave out background spots (obs['in_tissue'] ==
     0, reported in params.in_tissue_filter); harmonize gene IDs and
     intersect the genes.
  2. Choose each input's matrix (worker_utils.choose_counts_matrix): a
     matrix with negative or non-finite values -- scaled / z-scored data,
     neither counts nor log1p -- is refused, naming ``--use-raw-counts``
     when adata.raw holds counts; ``--use-raw-counts`` reads adata.raw.X
     (the spatial input's is required, the reference's is read when it has
     one). Log1p data is accepted by design (upstream un-logs it), so a
     non-integer matrix is not refused. Then bring both matrices to the
     count scale utils_pyRCTD models (``--input-scale`` auto | counts |
     log1p) and REPORT what was done: upstream's LoadData silently applies
     exp(X)-1 to any matrix whose maximum is below 30; this worker used to
     copy that rule without a log line or a payload key. A non-integer
     matrix taken as counts (max >= 30, or input_scale='counts') is said
     in a warning.
  3. Cost the dense genes x spots and genes x cells frames utils_pyRCTD
     requires and refuse, with the numbers, when they do not fit in the
     memory available (worker_utils.available_memory_bytes: MemAvailable,
     capped by a cgroup memory limit). Nothing is subsampled.
  4. Build SpatialRNA + Reference objects (upstream utils_pyRCTD).
  5. create_RCTD(...) and run_RCTD(..., doublet_mode='full').
  6. Extract per-spot weights, set the solver's small negative weights to 0 (as the manual
     benchmark runner does; the count is reported), normalize, save prediction CSV + h5ad.
     RCTD scores a spot only when its UMI total is at least UMI_min = min(100, UMI_min_sigma)
     (UMI_min_sigma picks the spots its noise parameter is fitted on); the floor is published
     as params.UMI_min and the unscored spots are counted in a warning.

status='dep_missing' covers every way the upstream method can be
unavailable: the source checkout is not found; it is found but
``utils_pyRCTD`` (or one of its own imports -- ray, qpsolvers, psutil,
scanpy) cannot be imported; or its ``extdata/`` likelihood tables are
missing. In each case the worker fails LOUDLY instead of silently
falling back to NNLS — silent NNLS produces misleading "SpatialScope"
outputs that look like the real method but aren't. An explicit
--allow-nnls-fallback flag lets users opt into the degraded path if they
accept it; the payload then says so in ``params.method`` /
``params.used_fallback``.

Environment: /opt/conda/envs/spatialscope_env
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

# Ensure worker_utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    build_deconv_analysis,
    cell_type_rename_note,
    cell_type_rename_params,
    choose_counts_matrix,
    cpu_budget,
    default_output_dir,
    expression_matrix_kind,
    gene_id_harmonization_note,
    gene_id_harmonization_params,
    harmonize_gene_ids,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    preflight_check,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    sanitize_cell_type_names,
    unsupported_choice_msg,
)
from worker_utils import (
    # Imported under another name: ``_run_spatialscope`` has a ``drop_unlabeled`` *parameter* (the shared
    # vocabulary), which would shadow the helper inside that function and turn every call into
    # ``TypeError: 'bool' object is not callable``.
    drop_unlabeled as _split_unlabeled,
)

#: Accepted values of ``--input-scale``.
INPUT_SCALES = ("auto", "counts", "log1p")

#: Upstream ``Cell_Type_Identification.LoadData`` un-logs a matrix whose maximum is below this.
UNLOG_MAX = 30.0

#: RCTD's scoring floor is ``UMI_min``: ``create_RCTD`` keeps a spot only when its UMI total is at least
#: ``UMI_min`` (and at most ``UMI_MAX``, upstream's 20,000,000 default). The worker fixes it at
#: ``min(UMI_MIN_CAP, UMI_min_sigma)``; ``UMI_min_sigma`` itself only picks the spots ``choose_sigma_c``
#: fits the noise parameter on.
UMI_MIN_CAP = 100
UMI_MAX = 20_000_000

#: What runs when the upstream method runs.
METHOD_NAME = (
    "SpatialScope CTI WarmStart (utils_pyRCTD create_RCTD + run_RCTD, doublet_mode='full', CPU; Stage-2 not run)"
)
#: What runs when the caller allowed the substitute. Keeps the label the portal and config promise.
FALLBACK_METHOD_NAME = (
    "SpatialScope (NNLS fallback): scipy.optimize.nnls per-spot regression on cell-type mean signatures; "
    "upstream CTI/WarmStart did not run"
)


def _log(msg: str) -> None:
    """Log to stderr so stdout stays clean for JSON output."""
    print(f"[spatialscope] {msg}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """Redirect stdout to stderr to capture training progress bars."""
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


class _SpatialScopeUnavailable(RuntimeError):
    """Raised when the upstream method cannot run for an environmental reason -- the source
    checkout is missing, ``utils_pyRCTD`` cannot be imported, or its likelihood tables are absent --
    and the caller did not opt into the NNLS fallback. The CLI translates this into
    status='dep_missing' so callers don't get a silently-degraded NNLS result that pretends to be
    SpatialScope, and so the agent can tell an environment problem from a data problem."""


def _load_h5ad(path: str):
    """Load an h5ad file."""
    try:
        import anndata as ad

        return ad.read_h5ad(path)
    except Exception:
        import scanpy as sc

        return sc.read_h5ad(path)


def _find_spatialscope_src() -> str | None:
    """
    Locate the SpatialScope source's `src/` directory (containing
    utils_pyRCTD.py). The checkout -- the upstream repo with this repo's patched
    ``src/utils_pyRCTD.py`` -- lives at ``tools/third_party/SpatialScope`` (moved in 2026-10-06 from the
    manual runner's env, which was removed).
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "SpatialScope"),
        "/opt/SpatialScope",
    ]
    # Allow override via env var
    env_override = os.environ.get("SPATIALSCOPE_SRC")
    if env_override:
        candidates.insert(0, env_override)

    for repo_root in candidates:
        src_dir = os.path.join(repo_root, "src")
        utils_path = os.path.join(src_dir, "utils_pyRCTD.py")
        if os.path.isfile(utils_path):
            return repo_root

    return None


def _import_upstream(ss_root: str):
    """Import ``utils_pyRCTD`` from the checkout, turning any import failure into ``dep_missing``.

    ``utils_pyRCTD`` imports ray, qpsolvers, psutil, scanpy, matplotlib and seaborn at module level,
    so a checkout that *exists* can still be unusable. That used to surface as a generic
    status='error' with a bare ImportError message, while the docstring promised 'dep_missing' for
    exactly this case; the agent then read an environment problem as a data problem.
    """
    src_dir = os.path.join(ss_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        import utils_pyRCTD
    except Exception as e:  # ImportError, or whatever a half-installed dependency raises at import
        raise _SpatialScopeUnavailable(
            f"SpatialScope source was found at {ss_root} but its src/utils_pyRCTD.py could not be imported "
            f"({type(e).__name__}: {e}). The checkout or its environment is incomplete: utils_pyRCTD needs ray, "
            "qpsolvers, psutil, scanpy, matplotlib and seaborn. Repair the environment or point SPATIALSCOPE_SRC "
            "at a working checkout."
        ) from e
    return utils_pyRCTD, src_dir


def _ray_init_kwargs(n_cpus: int | None, src_dir: str) -> dict[str, Any]:
    """Settings for the CTI ray pool: one worker per usable CPU, each with BLAS pinned to 1 thread.

    ``utils_pyRCTD.decompose_batch`` submits one ``@ray.remote`` task **per spot**, so ray keeps
    ``num_cpus`` worker processes running. Every one of those processes imports numpy/scipy/osqp for
    its IRWLS solve, and OpenBLAS/OpenMP/MKL each read their thread count once, at import, defaulting
    to the whole machine -- so an unpinned pool asks for ``num_cpus x n_cores`` threads (94 x 96 =
    9024 on a 96-core box) to run per-spot solves on a few hundred rows. The threads cost more than
    the arithmetic: this is the same thrash that made RCTD stall in its init phase and time out
    (commit d1689ee, load average ~328). Pin each worker to a single thread so ray's own parallelism
    is the ONLY parallelism.

    Two deliberate details:

    * The pin travels on ray's ``runtime_env`` env_vars, which are applied to a worker process before
      it imports anything -- and, unlike d1689ee's R subprocess, this leaves the driver's own
      ``os.environ`` untouched. That matters: a process-wide thread variable would turn
      ``prost_worker.py``'s ``os.environ.setdefault`` pin into a no-op and silently cost PROST the
      reproducibility it depends on.
    * An operator who set one of these keeps their value; only unset ones are pinned.

    The pool is sized from :func:`worker_utils.cpu_budget`, i.e. from the CPUs this process may
    actually use. ``os.cpu_count()`` reports the machine, so under a cgroup quota or an affinity mask
    it re-creates the oversubscription from the other direction.
    """
    if n_cpus is None:
        n_cpus = cpu_budget(reserve=2, minimum=2)
    env_vars = {var: os.environ.get(var, "1") for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    env_vars["PYTHONPATH"] = src_dir + ":" + os.environ.get("PYTHONPATH", "")
    return {
        "num_cpus": int(n_cpus),
        "_temp_dir": "/tmp/ray",
        "include_dashboard": False,
        "ignore_reinit_error": True,
        "runtime_env": {"env_vars": env_vars},
    }


def _load_likelihood_tables(ss_root: str):
    """Load the RCTD likelihood Q-matrix tables shipped with SpatialScope.
    Mirrors `load_likelihood()` from the manual run_spatialscope.py runner.

    A checkout without ``extdata/`` is as unusable as one without ``utils_pyRCTD``, so a missing
    table is reported as ``dep_missing`` rather than as a FileNotFoundError from deep in the run.
    """
    import gzip

    extdata = os.path.join(ss_root, "extdata")

    try:
        with gzip.open(os.path.join(extdata, "Q_mat_1_1.txt.gz"), "rt") as f:
            lines = f.readlines()
        with gzip.open(os.path.join(extdata, "Q_mat_1_2.txt.gz"), "rt") as f:
            lines += f.readlines()
        with gzip.open(os.path.join(extdata, "Q_mat_2_1.txt.gz"), "rt") as f:
            lines2 = f.readlines()
        with gzip.open(os.path.join(extdata, "Q_mat_2_2.txt.gz"), "rt") as f:
            lines2 += f.readlines()
        with open(os.path.join(extdata, "X_vals.txt")) as f:
            lines_X = f.readlines()
    except OSError as e:
        raise _SpatialScopeUnavailable(
            f"SpatialScope source at {ss_root} has no complete extdata/ likelihood tables "
            f"(Q_mat_1_1.txt.gz, Q_mat_1_2.txt.gz, Q_mat_2_1.txt.gz, Q_mat_2_2.txt.gz, X_vals.txt): {e}. "
            "The checkout is incomplete; restore extdata/ or point SPATIALSCOPE_SRC at a full checkout."
        ) from e

    Q1 = {}
    for i in range(len(lines)):
        if i == 61:
            Q1[str(72)] = np.reshape(np.array(lines[i].split(" ")).astype(np.float64), (2536, 103)).T
        elif i == 62:
            Q1[str(74)] = np.reshape(np.array(lines[i].split(" ")).astype(np.float64), (2536, 103)).T
        else:
            Q1[str(i + 10)] = np.reshape(np.array(lines[i].split(" ")).astype(np.float64), (2536, 103)).T

    Q2 = {}
    for i in range(len(lines2)):
        Q2[str(int(i * 2 + 76))] = np.reshape(np.array(lines2[i].split(" ")).astype(np.float64), (2536, 103)).T

    Q_mat_all = dict(Q1)
    Q_mat_all.update(Q2)
    X_vals_loc = np.array([float(x.strip()) for x in lines_X])
    return Q_mat_all, X_vals_loc


# --------------------------------------------------------------------------- #
# input scale
# --------------------------------------------------------------------------- #


def _matrix_max(X) -> float:
    """Largest value in ``X`` without densifying a sparse matrix (an implicit zero counts)."""
    from scipy.sparse import issparse

    if issparse(X):
        if X.nnz == 0:
            return 0.0
        mx = float(X.data.max())
        if X.nnz < int(X.shape[0]) * int(X.shape[1]):
            mx = max(mx, 0.0)
        return mx
    arr = np.asarray(X)
    return float(arr.max()) if arr.size else 0.0


def _expm1_in_place(adata) -> None:
    """Undo log1p without densifying: scipy sparse matrices carry their own ``expm1``."""
    from scipy.sparse import issparse

    X = adata.X
    adata.X = X.expm1() if issparse(X) else np.expm1(np.asarray(X))


def _raw_counts_hint(raw_holds_counts: bool) -> str:
    return " adata.raw holds raw counts: pass use_raw_counts=True to run on them." if raw_holds_counts else ""


def _apply_input_scale(adata_st, adata_sc, input_scale: str, raw_holds_counts: dict | None = None) -> dict[str, Any]:
    """Bring both matrices to the count scale utils_pyRCTD models, and say exactly what was done.

    Upstream ``LoadData`` applies ``exp(X) - 1`` to any matrix whose maximum is below 30 and then
    re-normalises the reference with ``sc.pp.normalize_total`` -- silently. This worker copied the
    rule without a log line or a payload key, so a low-count raw panel (a targeted assay whose
    deepest spot holds 12 UMIs of one gene) would have been exponentiated into garbage and reported
    as a clean run.

    ``input_scale='auto'`` keeps upstream's rule but never un-logs a matrix whose every value is a
    whole number (counts are integers; log1p values are not) and reports the decision per matrix.
    ``'counts'`` switches the heuristic off. ``'log1p'`` forces it, and refuses a matrix whose
    maximum says it cannot be log1p-scaled. The returned report is what the payload publishes.

    Every stored value is read (``worker_utils.expression_matrix_kind``). A matrix with negative or
    non-finite values is neither counts nor log1p -- a z-scored X would reach RCTD as UMIs, its row
    sums as nUMI and its floored values as likelihood-table rows -- and is refused whatever
    ``input_scale`` says. A non-integer matrix taken as counts (maximum 30 or more in 'auto', or on
    request in 'counts') runs with a warning. ``raw_holds_counts`` (role -> bool) adds the
    ``use_raw_counts`` remedy to both messages when that input's ``adata.raw`` holds counts.
    """
    if input_scale not in INPUT_SCALES:
        raise ValueError(unsupported_choice_msg("input_scale", input_scale, INPUT_SCALES))
    raw_holds_counts = raw_holds_counts or {}
    report: dict[str, Any] = {"input_scale": input_scale, "warnings": []}
    for role, adata in (("spatial", adata_st), ("reference", adata_sc)):
        hint = _raw_counts_hint(bool(raw_holds_counts.get(role)))
        kind = expression_matrix_kind(adata.X)
        if kind in ("negative", "nonfinite"):
            what = "negative values (scaled or z-scored data)" if kind == "negative" else "NaN or infinite values"
            raise ValueError(
                f"The {role} matrix holds {what}: it is neither counts nor log1p, and SpatialScope's CTI (RCTD) "
                f"reads every value as a UMI count, whatever input_scale says (input_scale={input_scale!r})." + hint
            )
        integer = kind in ("counts", "empty")
        mx = _matrix_max(adata.X)
        if input_scale == "auto":
            scale = "counts" if (mx >= UNLOG_MAX or integer) else "log1p"
        else:
            scale = input_scale
        if scale == "log1p" and mx >= UNLOG_MAX:
            raise ValueError(
                f"input_scale='log1p' but the {role} matrix has a maximum of {mx:g}; a log1p-scaled expression "
                f"matrix never reaches {UNLOG_MAX:g}, and expm1 of these values would not be counts. Pass "
                "input_scale='counts' if the matrix holds counts, or input_scale='auto' to let the max<30 rule "
                "decide per matrix."
            )
        report[f"{role}_max"] = mx
        report[f"{role}_scale"] = scale
        report[f"{role}_unlogged"] = scale == "log1p"
        report[f"{role}_matrix_kind"] = kind
        if scale == "log1p":
            _log(f"{role}: X.max()={mx:g}, taken as log1p; applying expm1 (input_scale={input_scale!r})")
            _expm1_in_place(adata)
            if input_scale == "auto":
                report["warnings"].append(
                    f"{role} expression (max {mx:g}, non-integer values) was taken as log1p and un-logged with "
                    "expm1 before deconvolution, following upstream LoadData's max<30 rule; pass input_scale='counts' "
                    "if it already held counts."
                )
        elif not integer:
            if input_scale == "counts":
                why = "on request (input_scale='counts')"
                if mx < UNLOG_MAX:
                    why += "; upstream's max<30 rule would have un-logged it"
            else:
                why = f"because its maximum {mx:g} is {UNLOG_MAX:g} or more, so it cannot be log1p"
            report["warnings"].append(
                f"{role} expression holds non-integer values (normalised data?) and was taken as counts {why}. "
                "RCTD reads each value as a UMI count and each row sum as the spot's nUMI, so the result was computed "
                "on a matrix that is not counts." + hint
            )
        elif mx < UNLOG_MAX and input_scale == "auto":
            report["warnings"].append(
                f"{role} expression has a maximum of only {mx:g} but every value is a whole number, so it was "
                "kept as counts; upstream's max<30 rule would have exponentiated it. Pass input_scale='log1p' "
                "to force un-logging."
            )
    report["reference_renormalized"] = False
    if report["reference_unlogged"]:
        import scanpy as sc

        # Upstream LoadData re-normalises the reference after un-logging it; the spatial side is left as is.
        sc.pp.normalize_total(adata_sc, inplace=True)
        report["reference_renormalized"] = True
    return report


def _choose_counts(adata, role: str, use_raw_counts: bool, raw_required: bool):
    """``(adata, info, note, raw_holds_counts)`` -- the matrix one input runs on, chosen by the shared rule.

    ``worker_utils.choose_counts_matrix`` refuses negative or non-finite values (naming
    ``use_raw_counts`` when ``adata.raw`` holds counts) and, with ``use_raw_counts``, returns
    ``adata.raw.X``. For the spatial input (``raw_required``) a missing ``adata.raw`` is then an error,
    as in every tool with this switch; the reference is read from ``adata.raw`` only when it has one,
    and ``note`` says when it did not. Its non-integer warning is dropped here: log1p input is accepted
    by design (upstream un-logs it), so what a non-integer matrix means is decided and said by
    :func:`_apply_input_scale`. ``raw_holds_counts`` feeds that function's remedy.
    """
    has_raw = getattr(adata, "raw", None) is not None
    read_raw = bool(use_raw_counts) and (raw_required or has_raw)
    try:
        chosen, info = choose_counts_matrix(adata, read_raw)
    except ValueError as e:
        raise ValueError(f"The {role} h5ad: {e}") from e
    note = None
    if use_raw_counts and not read_raw:
        note = f"use_raw_counts=True, but the {role} h5ad has no adata.raw, so its X was used."
    raw_holds_counts = (not read_raw) and has_raw and expression_matrix_kind(adata.raw.X) == "counts"
    return chosen, dict(info, warning=None), note, bool(raw_holds_counts)


def _umi_scoring_floor(UMI_min_sigma: int) -> int:
    """``UMI_min`` handed to ``create_RCTD``: spots below it are not scored at all (see :data:`UMI_MIN_CAP`)."""
    return min(UMI_MIN_CAP, int(UMI_min_sigma))


# --------------------------------------------------------------------------- #
# dense-frame budget
# --------------------------------------------------------------------------- #


def _memory_budget_bytes() -> int | None:
    """Memory this run may take for its dense frames, or None when the platform cannot say.

    ``worker_utils.available_memory_bytes``: the smaller of ``MemAvailable`` and the room under a
    cgroup memory limit (page cache counted reclaimable). An earlier revision read ``MemAvailable``
    alone, which inside a memory-limited container reports the *host*, so the promised refusal never
    fired and the run was OOM-killed part-way instead.
    """
    budget = available_memory_bytes()
    return None if budget is None else int(budget)


def _dense_frame_bytes(n_genes: int, n_spots: int, n_cells: int, itemsize: int) -> dict[str, int]:
    """Bytes of the dense frames ``utils_pyRCTD`` requires, and the peak the run reaches with them.

    The upstream API takes pandas DataFrames -- genes x spots for ``SpatialRNA``, genes x cells for
    ``Reference`` -- and indexes them with ``.loc``; there is no sparse path, so these two frames are
    intrinsic to the method. ``create_RCTD`` copies the spatial frame twice (``restrict_counts`` on
    the UMI floor, then on the bulk gene list) and the reference once (``create_downsampled_data``),
    and ``get_cell_type_info`` slices one cell type at a time, so the peak is about three spatial
    frames plus two reference frames.
    """
    itemsize = max(int(itemsize), 4)
    spatial = int(n_genes) * int(n_spots) * itemsize
    reference = int(n_genes) * int(n_cells) * itemsize
    return {"spatial": spatial, "reference": reference, "peak": 3 * spatial + 2 * reference}


def _check_dense_budget(adata_st, adata_sc) -> dict[str, Any]:
    """Refuse, with the numbers, before ``to_df()`` allocates a frame that cannot fit. Never subsamples."""
    itemsize = max(int(np.dtype(adata_st.X.dtype).itemsize), int(np.dtype(adata_sc.X.dtype).itemsize))
    est = _dense_frame_bytes(adata_st.n_vars, adata_st.n_obs, adata_sc.n_obs, itemsize)
    budget = _memory_budget_bytes()
    gb = 1e9
    if budget is not None and est["peak"] > budget:
        raise MemoryError(
            f"SpatialScope's utils_pyRCTD takes dense pandas frames: genes x spots ({adata_st.n_vars} x "
            f"{adata_st.n_obs}, {est['spatial'] / gb:.1f} GB) and genes x cells ({adata_sc.n_vars} x "
            f"{adata_sc.n_obs}, {est['reference'] / gb:.1f} GB). With the copies create_RCTD makes the run peaks "
            f"near {est['peak'] / gb:.1f} GB and {budget / gb:.1f} GB is available (MemAvailable, capped by any "
            "cgroup memory limit). The data is not subsampled, and "
            f"no parameter of this tool shrinks these frames: both scale with the shared gene count "
            f"({adata_st.n_vars} here after gene-ID harmonisation and intersection) and with every spot and every "
            "reference cell. Run on a machine with more free memory. allow_nnls_fallback=True would instead run "
            "the sparse NNLS substitute, reported as such (params.used_fallback=True), not SpatialScope."
        )
    return {
        "spatial_frame_gb": round(est["spatial"] / gb, 3),
        "reference_frame_gb": round(est["reference"] / gb, 3),
        "estimated_peak_gb": round(est["peak"] / gb, 3),
        "memory_available_gb": None if budget is None else round(budget / gb, 3),
    }


# --------------------------------------------------------------------------- #
# the method
# --------------------------------------------------------------------------- #

#: upstream ``utils_pyRCTD.Reference``'s own per-type cap (RCTD's ``n_max_cells``).
UPSTREAM_N_MAX_CELLS = 10000


def _reference_cell_cap(cell_types_df) -> int:
    """The ``n_max_cells`` that makes upstream ``Reference`` keep every cell of every type.

    ``Reference`` -> ``create_downsampled_data`` draws ``min(n_max_cells, n_type)`` cells per type
    without a seed. A cap at least as large as the biggest type turns that draw into a permutation of
    all of them, so nothing is subsampled; upstream's 10000 is kept as the floor so a small reference
    is handled exactly as before.
    """
    counts = pd.Series(np.asarray(cell_types_df).ravel()).value_counts()
    largest = int(counts.max()) if len(counts) else 0
    return max(UPSTREAM_N_MAX_CELLS, largest)


def _coords_frame(spatial, index) -> pd.DataFrame:
    """``obsm['spatial']`` as the coords frame ``SpatialRNA`` stores, at whatever width it has.

    CTI never reads the coordinates (``utils_pyRCTD`` only carries them alongside the counts), so a
    3-column ``obsm['spatial']`` -- the shape the 3D coordinate contract and ST-GEARS/STARmap write --
    is passed through whole. The old code named exactly two columns and died on such input with
    pandas' "Shape of passed values is (n, 3), indices imply (n, 2)".
    """
    arr = spatial.toarray() if hasattr(spatial, "toarray") else spatial
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            f"obsm['spatial'] has shape {tuple(arr.shape)}; SpatialScope CTI needs at least two coordinate columns."
        )
    names = ["x", "y", "z"][: arr.shape[1]] if arr.shape[1] <= 3 else [f"coord_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=index, columns=names)


def _run_spatialscope_warmstart(
    adata_sc,
    adata_st,
    cell_type_key: str,
    ss_root: str,
    UMI_min_sigma: int = 300,
    n_cpus: int | None = None,
    report: dict | None = None,
) -> pd.DataFrame:
    """
    Run SpatialScope CTI (WarmStart) on CPU and return per-spot proportions.
    Mirrors lines 137-200 of the manual runner. Both matrices must already be on the count scale
    (see :func:`_apply_input_scale`); this function no longer un-logs anything.

    ``report`` (optional dict) receives the dense-frame cost, the number of reference cells used and
    what :func:`_clip_negative_weights` clipped, so the payload can publish them.
    """
    import logging

    utils_pyRCTD, src_dir = _import_upstream(ss_root)
    Reference = utils_pyRCTD.Reference
    SpatialRNA = utils_pyRCTD.SpatialRNA
    create_RCTD = utils_pyRCTD.create_RCTD
    run_RCTD = utils_pyRCTD.run_RCTD

    if "spatial" not in adata_st.obsm:
        raise ValueError("Spatial AnnData has no obsm['spatial']; SpatialScope CTI requires spot coordinates.")

    # Read the likelihood tables before anything expensive: a checkout without extdata/ is a
    # dep_missing, and it used to surface only after the dense frames, create_RCTD and the ray pool.
    Q_mat_all, X_vals_loc = _load_likelihood_tables(ss_root)

    # The upstream API is DataFrame-only, so the two dense frames below are intrinsic; cost them first.
    cost = _check_dense_budget(adata_st, adata_sc)
    if report is not None:
        report["dense_frames"] = cost

    UMI_min = _umi_scoring_floor(UMI_min_sigma)

    counts = adata_st.to_df().T  # genes × spots -- the one dense copy of the spatial matrix this worker makes
    coords = _coords_frame(adata_st.obsm["spatial"], counts.columns)

    # np.asarray().flatten() handles both a scipy np.matrix column and a dense 1-D sum.
    x_arr = np.asarray(adata_st.X.sum(-1)).flatten()
    nUMI = pd.DataFrame(x_arr, index=adata_st.obs.index)
    puck = SpatialRNA(coords, counts, nUMI)

    sc_counts = adata_sc.to_df().T  # genes × cells -- the one dense copy of the reference this worker makes
    cell_types_df = pd.DataFrame(adata_sc.obs[cell_type_key])
    # Per-cell UMI straight from X (sparse-aware); the old code densified the reference a second time for this sum.
    sc_nUMI = pd.DataFrame(np.asarray(adata_sc.X.sum(-1)).flatten(), index=adata_sc.obs.index)

    loggings = logging.getLogger("ss")
    loggings.addHandler(logging.NullHandler())
    loggings.setLevel(logging.WARNING)

    # upstream Reference() keeps at most n_max_cells (default 10000) cells per type, chosen with an
    # unseeded np.random.choice, and says so only through a logger this worker silences. Every
    # labelled cell is used instead, so the cell-type means are the means of the reference that was
    # supplied rather than of a random draw that changes between runs. The dense frame above already
    # holds every cell, so this costs no memory beyond what _check_dense_budget priced.
    n_max_cells = _reference_cell_cap(cell_types_df)
    reference = Reference(sc_counts, cell_types_df, sc_nUMI, n_max_cells=n_max_cells, loggings=loggings)
    if report is not None:
        report["n_reference_cells_used"] = int(reference["counts"].shape[1])

    import ray

    try:
        ray.shutdown()
    except Exception:
        pass
    ray_kwargs = _ray_init_kwargs(n_cpus, src_dir)
    n_cpus = ray_kwargs["num_cpus"]
    _log(
        f"ray pool: {n_cpus} workers, BLAS pinned to {ray_kwargs['runtime_env']['env_vars']['OMP_NUM_THREADS']}/worker"
    )
    ray.init(**ray_kwargs)

    myRCTD = create_RCTD(
        puck,
        reference,
        max_cores=n_cpus,
        UMI_min=UMI_min,
        UMI_min_sigma=UMI_min_sigma,
        loggings=loggings,
    )
    myRCTD = run_RCTD(myRCTD, Q_mat_all, X_vals_loc, doublet_mode="full", loggings=loggings)

    # Extract per-spot weights
    results = myRCTD["results"]
    if isinstance(results, dict) and "weights" in results and isinstance(results["weights"], pd.DataFrame):
        weights = results["weights"]
    elif isinstance(results, pd.DataFrame):
        weights = results
    else:
        raise RuntimeError(f"SpatialScope returned unexpected results type: {type(results)}")

    weights, clipped = _clip_negative_weights(weights)
    if report is not None:
        report["negative_weights_clipped"] = clipped

    # Normalize per spot to sum to 1
    row_sums = weights.sum(axis=1).replace(0.0, 1.0)
    weights = weights.div(row_sums, axis=0).fillna(0.0)
    return weights


#: A spot whose clipped negative weight is at least this share of its positive weight is warned about.
NEGATIVE_MASS_WARN = 0.01


def _clip_negative_weights(weights: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Set the solver's small negative weights to 0 before the per-spot normalisation, and count them.

    ``fitPixels(doublet_mode='full')`` solves each spot's IRWLS with OSQP. The ``w >= 0`` constraint
    holds only to OSQP's tolerance, and the last iterate is returned unclamped, so the weights carry
    small negatives (on the SpinalCord Visium smoke, 44% of entries, down to -5e-4). Normalised as
    they were, they became negative *proportions* in the CSV and in ``obsm`` -- a proportion matrix
    that breaks any log- or divergence-based comparison. The manual benchmark runner this worker
    mirrors clips at 0 before normalising (``weights.fillna(0.0).clip(lower=0.0)``); so does this,
    and the payload says how much was clipped.
    """
    weights = weights.fillna(0.0)
    values = weights.to_numpy(dtype=float)
    negative = np.minimum(values, 0.0)
    n_negative = int((negative < 0).sum())
    positive_mass = np.maximum(values, 0.0).sum(axis=1)
    negative_mass = -negative.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(positive_mass > 0, negative_mass / positive_mass, np.where(negative_mass > 0, np.inf, 0.0))
    report = {
        "n_entries": n_negative,
        "n_spots": int((negative_mass > 0).sum()),
        "max_share_of_spot": float(share.max()) if share.size else 0.0,
        "min_weight": float(values.min()) if values.size else 0.0,
    }
    if n_negative:
        _log(
            f"clipped {n_negative} negative CTI weight(s) to 0 in {report['n_spots']} spot(s); largest clipped share "
            f"of a spot's weight {report['max_share_of_spot']:.3g}, most negative weight {report['min_weight']:.3g}"
        )
    return weights.clip(lower=0.0), report


# --------------------------------------------------------------------------- #
# the opt-in substitute
# --------------------------------------------------------------------------- #


def _build_signature_matrix(adata_sc, cell_type_key: str) -> pd.DataFrame:
    """Mean expression per cell type (types x genes), computed with an indicator product so a sparse
    reference stays sparse; the old ``to_df``/``groupby`` path densified the whole reference."""
    from scipy.sparse import csr_matrix, issparse

    cat = pd.Categorical(adata_sc.obs[cell_type_key])
    codes = np.asarray(cat.codes, dtype=np.int64)
    labelled = codes >= 0  # a NaN label is not a class; drop_unlabeled decided upstream whether any remain
    n_types = len(cat.categories)
    idx = np.flatnonzero(labelled)
    indicator = csr_matrix(
        (np.ones(idx.shape[0], dtype=np.float64), (codes[idx], idx)), shape=(n_types, adata_sc.n_obs)
    )
    sums = indicator @ adata_sc.X
    sums = np.asarray(sums.todense()) if issparse(sums) else np.asarray(sums, dtype=np.float64)
    n_per_type = np.bincount(codes[idx], minlength=n_types).astype(np.float64)
    used = n_per_type > 0
    means = sums[used] / n_per_type[used][:, None]
    return pd.DataFrame(means, index=list(np.asarray(cat.categories)[used]), columns=adata_sc.var_names)


def _nnls_deconv(adata_sc, adata_st, cell_type_key: str) -> pd.DataFrame:
    """NNLS-based deconvolution fallback (only behind --allow-nnls-fallback). One spot row at a time,
    so a sparse slide is never densified whole."""
    from scipy.optimize import nnls
    from scipy.sparse import issparse

    _log("Running NNLS fallback deconvolution...")

    sig_matrix = _build_signature_matrix(adata_sc, cell_type_key)
    sig_np = sig_matrix.values.T  # genes x celltypes

    X = adata_st.X
    sparse = issparse(X)
    if sparse and X.format != "csr":
        X = X.tocsr()  # row access below; a CSC slide would re-scan every column per spot
    n_spots = adata_st.n_obs
    proportions = np.zeros((n_spots, sig_matrix.shape[0]))

    for i in range(n_spots):
        row = X[i].toarray().ravel() if sparse else np.asarray(X[i]).ravel()
        coef, _ = nnls(sig_np, np.asarray(row, dtype=np.float64))
        total = coef.sum()
        if total > 0:
            proportions[i, :] = coef / total

    prop_df = pd.DataFrame(
        proportions,
        index=adata_st.obs_names,
        columns=sig_matrix.index,
    )
    _log("NNLS fallback deconvolution complete.")
    return prop_df


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #


def _run_spatialscope(
    sc_h5ad_path: str,
    spatial_h5ad_path: str,
    output_dir: str = default_output_dir(),
    cell_type_key: str = "cell_type",
    UMI_min_sigma: int = 300,
    n_cpus: int | None = None,
    allow_nnls_fallback: bool = False,
    input_scale: str = "auto",
    drop_unlabeled: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Core SpatialScope pipeline. Returns a WorkerOutput dict.

    By default this requires the upstream SpatialScope source; if it is
    not found, cannot be imported, or lacks its likelihood tables, the
    worker raises ``_SpatialScopeUnavailable`` which the CLI translates
    into a `dep_missing` error. Set ``allow_nnls_fallback=True`` to opt
    into the cheap NNLS approximation instead — the resulting output is
    clearly labelled ``method_name='SpatialScope (NNLS fallback)'`` and
    ``params.used_fallback=True`` so callers can tell what they actually
    got.
    """
    allow_drop_unlabeled = bool(drop_unlabeled)

    if input_scale not in INPUT_SCALES:
        raise ValueError(unsupported_choice_msg("input_scale", input_scale, INPUT_SCALES))

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load data ----
    _log(f"Reading scRNA-seq reference: {sc_h5ad_path}")
    adata_sc = _load_h5ad(sc_h5ad_path)
    _log(f"Reading spatial data: {spatial_h5ad_path}")
    adata_st = _load_h5ad(spatial_h5ad_path)

    _log(f"scRNA: {adata_sc.n_obs} cells x {adata_sc.n_vars} genes")
    _log(f"Spatial: {adata_st.n_obs} spots x {adata_st.n_vars} genes")

    # utils_pyRCTD keys everything by name: a duplicated gene raised pandas' "Reindexing only valid
    # with uniquely valued Index objects" and a duplicated barcode SpatialRNA's misleading "barcodes
    # ... were not mutually shared". Duplicates are renamed with the shared helper and the renames
    # are published (params n_*_renamed, analysis note).
    renamed_sc = make_names_unique_and_report(adata_sc)
    renamed_st = make_names_unique_and_report(adata_st)

    # Background spots (obs['in_tissue'] == 0, as CELLxGENE Visium exports carry them) are glass,
    # not tissue: they are left out before anything is priced or deconvolved, and reported.
    adata_st, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata_st)
    if n_spots_off_tissue:
        _log(f"Leaving out {n_spots_off_tissue} of {n_spots_supplied} spots with in_tissue == 0 (background).")

    # The matrix each input runs on: a scaled X (negative values) is refused, naming use_raw_counts when
    # adata.raw holds counts; use_raw_counts reads adata.raw.X (the spatial input's is required).
    adata_st, st_counts, st_counts_note, st_raw_counts = _choose_counts(
        adata_st, "spatial", use_raw_counts, raw_required=True
    )
    adata_sc, sc_counts, sc_counts_note, sc_raw_counts = _choose_counts(
        adata_sc, "reference", use_raw_counts, raw_required=False
    )
    for role, info, renamed in (("spatial", st_counts, renamed_st), ("reference", sc_counts, renamed_sc)):
        if info["expression_source"] != "X":
            _log(f"use_raw_counts: the {role} input runs on adata.raw.X")
            make_names_unique_and_report(adata_st if role == "spatial" else adata_sc, into=renamed)

    # ---- 2. Validate ----
    if cell_type_key not in adata_sc.obs.columns:
        raise ValueError(
            f"cell_type_key='{cell_type_key}' not found in scRNA obs. Available keys: {list(adata_sc.obs.columns)}"
        )

    # A NaN label is not a class: upstream's np.unique over the labels dies on a str/float comparison,
    # and the NNLS path used to drop such cells without a word.
    keep_mask, n_unlabeled_dropped = _split_unlabeled(
        adata_sc.obs[cell_type_key], allow_drop_unlabeled, what=f"reference cells (obs['{cell_type_key}'])"
    )
    if n_unlabeled_dropped:
        _log(f"Dropping {n_unlabeled_dropped} reference cells with no label in obs['{cell_type_key}'].")
        adata_sc = adata_sc[np.asarray(keep_mask)].copy()

    if not hasattr(adata_sc.obs[cell_type_key], "cat"):
        adata_sc.obs[cell_type_key] = adata_sc.obs[cell_type_key].astype("category")
    else:
        adata_sc.obs[cell_type_key] = adata_sc.obs[cell_type_key].cat.remove_unused_categories()

    cell_types = list(adata_sc.obs[cell_type_key].cat.categories)
    _log(f"Found {len(cell_types)} cell types: {cell_types}")

    # Harmonize gene IDs and intersect genes
    _log("Harmonizing gene IDs between scRNA and spatial data...")
    gene_id_report: dict = {}
    harmonize_gene_ids(adata_sc, adata_st, report=gene_id_report)
    shared_genes = sorted(set(adata_sc.var_names) & set(adata_st.var_names))
    if len(shared_genes) == 0:
        raise ValueError(
            "No shared genes between scRNA and spatial data even after gene ID harmonization. Check gene ID formats."
        )
    _log(f"{len(shared_genes)} genes shared between scRNA and spatial")

    adata_sc = adata_sc[:, shared_genes].copy()
    adata_st_full = adata_st.copy()
    adata_st = adata_st[:, shared_genes].copy()

    # ---- 3. Bring both matrices to the count scale, and remember what was done ----
    # Applied once, here, so the real method and the opt-in NNLS substitute see the same matrices, and so
    # the decision reaches the payload. adata_st_full keeps the caller's X for the written h5ad.
    scale_report = _apply_input_scale(
        adata_st, adata_sc, input_scale, raw_holds_counts={"spatial": st_raw_counts, "reference": sc_raw_counts}
    )

    # ---- 4. Run real SpatialScope (CTI WarmStart on CPU) ----
    prop_df = None
    used_fallback = False
    fallback_reason = ""
    warm_report: dict[str, Any] = {}
    ss_root = _find_spatialscope_src()

    if ss_root is not None:
        _log(f"Using SpatialScope source at {ss_root}")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
        try:
            prop_df = _run_spatialscope_warmstart(
                adata_sc=adata_sc,
                adata_st=adata_st,
                cell_type_key=cell_type_key,
                ss_root=ss_root,
                UMI_min_sigma=UMI_min_sigma,
                n_cpus=n_cpus,
                report=warm_report,
            )
        except Exception as e:
            _log(f"SpatialScope WarmStart failed: {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stderr)
            if not allow_nnls_fallback:
                raise
            fallback_reason = f"SpatialScope WarmStart failed: {type(e).__name__}: {e}"
    else:
        msg = (
            "SpatialScope source not found. Looked under "
            "$SPATIALSCOPE_SRC, "
            "tools/third_party/SpatialScope and /opt/SpatialScope for 'src/utils_pyRCTD.py'."
        )
        if not allow_nnls_fallback:
            raise _SpatialScopeUnavailable(msg)
        _log(f"WARNING: {msg}")
        fallback_reason = msg

    if prop_df is None:
        # Only reachable when allow_nnls_fallback=True and native run failed.
        _log("Using NNLS fallback (allow_nnls_fallback=True).")
        prop_df = _nnls_deconv(adata_sc, adata_st, cell_type_key)
        used_fallback = True

    # ---- 5. Align the result to the slide ----
    # SpatialScope's WarmStart is RCTD underneath, and RCTD drops spots below its UMI floor
    # (UMI_min / UMI_min_sigma). So prop_df can cover only part of the slide, indexed by the
    # barcodes RCTD kept, in RCTD's own order. Match on barcode; positional assignment is used
    # only when the returned index carries no slide barcode at all.
    obs_names = pd.Index([str(x) for x in adata_st_full.obs_names])
    prop_df.index = pd.Index([str(x) for x in prop_df.index])
    known = prop_df.index.isin(obs_names)
    positional = not known.any()
    if positional:
        if len(prop_df) != len(obs_names):
            raise ValueError(
                f"SpatialScope returned {len(prop_df)} rows for {len(obs_names)} spots, and none of them "
                "name a spot on the slide, so the two cannot be matched up."
            )
        _log("Result index carries no spot barcodes; assigning positionally.")
        prop_df.index = obs_names
    else:
        if not known.all():
            _log(f"WARNING: dropping {int((~known).sum())} result row(s) naming spots that are not on the slide.")
            prop_df = prop_df[known]
        if prop_df.index.has_duplicates:
            _log(f"WARNING: {int(prop_df.index.duplicated().sum())} duplicate spot id(s) in the result; keeping first.")
            prop_df = prop_df[~prop_df.index.duplicated(keep="first")]

    # Sanitize column names
    safe_cols, ct_renames = sanitize_cell_type_names(prop_df.columns)
    prop_df.columns = safe_cols

    # Normalize if needed
    row_sums = prop_df.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.05):
        _log("Normalizing proportions to sum to 1...")
        prop_df = prop_df.div(row_sums.replace(0, 1), axis=0)

    prop_csv_path = output_path / "spatialscope_proportions.csv"
    prop_csv_partial = output_path / "spatialscope_proportions.csv.partial"
    prop_df.to_csv(prop_csv_partial)
    os.replace(prop_csv_partial, prop_csv_path)
    _log(f"Saved proportions to {prop_csv_path}")

    # Store in spatial AnnData. Reindexed onto the slide's own barcodes, so a spot can only ever
    # receive its own row; spots SpatialScope did not score are left empty rather than filled with
    # a neighbour's proportions.
    n_spots = int(adata_st_full.n_obs)
    n_spots_scored = int(len(prop_df))
    n_spots_unscored = max(n_spots - n_spots_scored, 0)
    # The floor create_RCTD was given; the NNLS substitute has none and scores every spot.
    umi_floor = None if used_fallback else _umi_scoring_floor(UMI_min_sigma)
    if n_spots_unscored:
        _log(
            f"{n_spots_scored} of {n_spots} spots were scored; the remaining "
            f"{n_spots_unscored} fell below SpatialScope's UMI floor (UMI_min={umi_floor}) and are left unscored."
        )
    aligned = prop_df if positional else prop_df.reindex(obs_names)
    adata_st_full.obsm["spatialscope_proportions"] = aligned.to_numpy(dtype=float)
    for ct in safe_cols:
        adata_st_full.obs[ct] = aligned[ct].to_numpy(dtype=float)

    out_h5ad = output_path / "spatialscope_spatial.h5ad"
    out_h5ad_partial = output_path / "spatialscope_spatial.h5ad.partial"
    _log(f"Saving annotated spatial AnnData to {out_h5ad}")
    adata_st_full.write(out_h5ad_partial)
    os.replace(out_h5ad_partial, out_h5ad)

    # ---- 6. Build output ----
    cell_type_names = list(prop_df.columns)
    n_celltypes = len(cell_type_names)

    try:
        dominant_ct = prop_df.idxmax(axis=1)
        dominant_counts = dict(Counter(dominant_ct))
    except Exception:
        dominant_counts = None

    method_label = "SpatialScope (NNLS fallback)" if used_fallback else "SpatialScope"

    out = WorkerOutput("spatialscope", task="deconvolution")
    out.set_data(
        n_cells_sc=int(adata_sc.n_obs),
        n_genes_sc=int(adata_sc.n_vars),
        n_spots=n_spots,
        n_spots_scored=n_spots_scored,
        n_genes=int(adata_st_full.n_vars),
    )
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    record_expression_source(out, st_counts)
    out.add_params(
        {
            "use_raw_counts": bool(use_raw_counts),
            "reference_expression_source": sc_counts["expression_source"],
            "reference_x_matrix_kind": sc_counts["x_matrix_kind"],
            "spatial_matrix_kind": scale_report["spatial_matrix_kind"],
            "reference_matrix_kind": scale_report["reference_matrix_kind"],
        }
    )
    for note in (st_counts_note, sc_counts_note):
        if note:
            out.add_warning(note)
    if umi_floor is not None:
        # The effective scoring floor, beside the requested UMI_min_sigma (the sigma-fit threshold).
        out.add_params({"UMI_min": umi_floor})
    if n_spots_unscored and umi_floor is not None:
        out.add_warning(
            f"{n_spots_unscored} of the {n_spots} spots were not scored: SpatialScope's CTI (RCTD's create_RCTD) keeps a "
            f"spot only when its UMI total over the {len(shared_genes)} shared genes is at least UMI_min={umi_floor} "
            f"(fixed at min({UMI_MIN_CAP}, UMI_min_sigma)) and at most {UMI_MAX:,}. They are not in "
            "spatialscope_proportions.csv and are left empty (NaN) in spatialscope_spatial.h5ad."
        )
    out.add_output_files(
        {
            "proportions_csv": str(prop_csv_path),
            "spatial_h5ad": str(out_h5ad),
        }
    )
    out.add_params(
        {
            "cell_type_key": cell_type_key,
            "n_shared_genes": len(shared_genes),
            "spatialscope_repo": ss_root or "not_found",
            "UMI_min_sigma": UMI_min_sigma,
            "used_nnls_fallback": used_fallback,
            "input_scale": scale_report["input_scale"],
            "spatial_scale": scale_report["spatial_scale"],
            "reference_scale": scale_report["reference_scale"],
            "spatial_unlogged": scale_report["spatial_unlogged"],
            "reference_unlogged": scale_report["reference_unlogged"],
            "reference_renormalized": scale_report["reference_renormalized"],
            "spatial_max": scale_report["spatial_max"],
            "reference_max": scale_report["reference_max"],
            "drop_unlabeled": allow_drop_unlabeled,
            "n_unlabeled_dropped": int(n_unlabeled_dropped),
        }
    )
    if warm_report.get("dense_frames"):
        out.add_params({"dense_frames": warm_report["dense_frames"]})
    clipped = warm_report.get("negative_weights_clipped")
    if clipped is not None:
        out.add_params({"negative_weights_clipped": clipped})
        if clipped["n_entries"] and clipped["max_share_of_spot"] >= NEGATIVE_MASS_WARN:
            out.add_warning(
                f"{clipped['n_entries']} negative CTI weight(s) in {clipped['n_spots']} spot(s) were set to 0 before "
                f"normalising (OSQP solves w >= 0 only to its tolerance); in the worst spot they were "
                f"{clipped['max_share_of_spot']:.1%} of its weight."
            )
    if "n_reference_cells_used" in warm_report:
        # Every labelled reference cell; upstream's 10000-per-type random draw is not applied.
        out.add_params({"n_reference_cells_used": int(warm_report["n_reference_cells_used"])})
    record_method(out, FALLBACK_METHOD_NAME if used_fallback else METHOD_NAME, used_fallback, fallback_reason)
    if used_fallback:
        record_ignored(
            out,
            ["UMI_min_sigma"] + (["n_cpus"] if n_cpus is not None else []),
            "the NNLS substitute ran instead of SpatialScope's CTI: it has no UMI floor, no sigma fit and no ray pool, "
            "and scores every spot",
        )
    out.add_warnings(scale_report["warnings"])
    if n_unlabeled_dropped:
        out.add_warning(
            f"{n_unlabeled_dropped} reference cells had no label in obs['{cell_type_key}'] and were left out "
            "(drop_unlabeled=True)."
        )
    out.add_params(gene_id_harmonization_params(gene_id_report))
    out.add_params(cell_type_rename_params(ct_renames))
    out.add_params(identifier_rename_params(renamed_st))
    out.add_params(identifier_rename_params(renamed_sc, suffix="sc"))
    out.set_summary(
        n_cell_types=n_celltypes,
        cell_type_names=cell_type_names,
        dominant_counts=dominant_counts,
    )
    scale_note = ""
    if n_spots_unscored:
        scale_note += (
            f" {n_spots_unscored} of the {n_spots} spots were not scored"
            + (f" (UMI total below the UMI_min={umi_floor} floor)" if umi_floor is not None else "")
            + "; the proportions cover the scored spots only."
        )
    if n_spots_off_tissue:
        scale_note += (
            f" {n_spots_off_tissue} of the {n_spots_supplied} spots supplied have obs['in_tissue'] == 0 (background) "
            "and were left out; they are not in the proportions CSV or the written h5ad."
        )
    raw_inputs = [
        r for r, info in (("spatial", st_counts), ("reference", sc_counts)) if info["expression_source"] != "X"
    ]
    if raw_inputs:
        scale_note += f" The {' and '.join(raw_inputs)} input ran on adata.raw.X (use_raw_counts=True)."
    if scale_report["spatial_unlogged"] or scale_report["reference_unlogged"]:
        unlogged = [r for r in ("spatial", "reference") if scale_report[f"{r}_unlogged"]]
        scale_note += (
            f" The {' and '.join(unlogged)} expression was un-logged (expm1) before deconvolution "
            f"(input_scale={input_scale!r})."
        )
    out.set_analysis(
        build_deconv_analysis(
            n_celltypes=n_celltypes,
            dominant_counts=dominant_counts,
            # dominant_counts is tallied over the spots that were scored, so the sentence must
            # count those and not the whole slide.
            total_spots=n_spots_scored,
            method_name=method_label,
        )
        + scale_note
        + gene_id_harmonization_note(gene_id_report)
        + cell_type_rename_note(ct_renames)
        + identifier_rename_note(renamed_st, subject="spatial data")
        + identifier_rename_note(renamed_sc, subject="scRNA reference")
    )

    return out.to_dict()


def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="SpatialScope worker: Stage-1 Cell-Type Identification (CTI WarmStart) on CPU, giving per-spot "
        "cell-type proportions. Stage-2 single-cell decomposition is GPU-only and is not run."
    )
    parser.add_argument("--sc-h5ad", required=True, help="Path to scRNA-seq reference h5ad")
    parser.add_argument("--spatial-h5ad", required=True, help="Path to spatial h5ad")
    parser.add_argument("--output-dir", default=default_output_dir(), help="Output directory")
    parser.add_argument("--cell-type-key", default="cell_type", help="obs column with cell-type labels")
    parser.add_argument(
        "--umi-min-sigma",
        type=int,
        default=300,
        help="RCTD UMI_min_sigma (paper default: 300): the UMI total a spot needs to take part in fitting RCTD's noise "
        "parameter sigma. It is not the scoring floor: a spot is scored when its UMI total over the shared genes is at "
        f"least UMI_min = min({UMI_MIN_CAP}, UMI_min_sigma), published as params.UMI_min.",
    )
    parser.add_argument(
        "--n-cpus",
        type=int,
        default=None,
        help="ray workers to use (default: the CPUs this process may use -- affinity mask / cgroup quota -- minus 2, "
        "at least 2)",
    )
    parser.add_argument(
        "--allow-nnls-fallback",
        action="store_true",
        help="If SpatialScope cannot run (source missing, utils_pyRCTD or its likelihood tables unusable, or the "
        "CTI run itself raising), fall back to a scipy.optimize.nnls per-spot regression. Off by default — "
        "the fallback is a clearly inferior method and should "
        "not be presented as 'SpatialScope' silently; when it runs, params.method names it, "
        "params.used_fallback is true and a warning gives the reason.",
    )
    parser.add_argument(
        "--input-scale",
        default="auto",
        help="Scale of X in both h5ads: 'auto' (upstream's rule: a matrix whose maximum is below 30 is "
        "taken as log1p and un-logged, except that a whole-number matrix is always counts), 'counts' "
        "(never un-log) or 'log1p' (always un-log). A matrix with negative or NaN values is refused under every "
        "setting; a non-integer one taken as counts runs with a warning. Whatever was done is reported in params.",
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="Leave out reference cells whose cell-type label is missing (NaN/empty) instead of failing. "
        "Off by default; the count dropped is reported.",
    )
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        help="Run on adata.raw.X instead of X: required of the spatial h5ad (refused without an adata.raw holding "
        "counts), read for the reference when it has one. For CELLxGENE exports whose X is log-normalised or scaled.",
    )

    args = parser.parse_args()

    preflight_check(
        inputs={
            "sc_h5ad": args.sc_h5ad,
            "spatial_h5ad": args.spatial_h5ad,
        },
        output_dir=args.output_dir,
    )

    error_info = None
    error_status = "error"
    error_exc = None
    with _redirect_stdout_to_stderr():
        try:
            result = _run_spatialscope(
                sc_h5ad_path=args.sc_h5ad,
                spatial_h5ad_path=args.spatial_h5ad,
                output_dir=args.output_dir,
                cell_type_key=args.cell_type_key,
                UMI_min_sigma=args.umi_min_sigma,
                n_cpus=args.n_cpus,
                allow_nnls_fallback=args.allow_nnls_fallback,
                input_scale=args.input_scale,
                drop_unlabeled=args.drop_unlabeled,
                use_raw_counts=args.use_raw_counts,
            )
        except _SpatialScopeUnavailable as e:
            _log(f"DEP_MISSING: {e}")
            error_info = str(e)
            error_status = "dep_missing"
            error_exc = e
        except Exception as e:
            _log(f"ERROR: {e}")
            traceback.print_exc(file=sys.stderr)
            error_info = str(e)
            error_exc = e

    # stdout: JSON only (must be outside redirect block)
    if error_info is not None:
        WorkerOutput.emit_error("spatialscope", error_info, task="deconvolution", status=error_status, exc=error_exc)
        sys.exit(1)

    print(json.dumps(result, default=str))
    sys.stdout.flush()


if __name__ == "__main__":
    _cli_main()
