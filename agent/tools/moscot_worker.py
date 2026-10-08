#!/usr/bin/env python
"""
MOSCOT worker for SpatialOmicsLab MCP.

- Runs inside /opt/conda/envs/moscot
- Does the actual MOSCOT computation.
- All logs/progress go to stderr, prefixed with [moscot-worker].
- The ONLY thing printed to stdout is a single JSON object at the end.

Supported problem types:
  - temporal   : moscot.problems.time.TemporalProblem
  - alignment  : moscot.problems.space.AlignmentProblem
  - mapping    : moscot.problems.space.MappingProblem

What the worker does to the data before moscot sees it (every step is reported in
``params.preprocessing``):
  - ``normalize_total`` + ``log1p`` on X. This is applied unconditionally: an X that is already
    log-transformed is transformed a second time, and the payload warns whenever X did not hold
    integer counts.
  - A PCA with a SAFE number of components, n_comps = min(30, n_obs - 1, n_vars - 1), on
    ``var['highly_variable']`` when that column exists (scanpy's own default) and on every gene
    otherwise. It is stored in ``obsm["X_moscot_pca"]`` and passed as ``joint_attr``
    (temporal/alignment) or ``sc_attr`` (mapping, when the caller gives none), so moscot never runs
    its own fixed n_comps=30 PCA and hits the n_components error on small inputs.
  - Mapping: moscot builds the expression (linear) term from ONE joint PCA over the genes both
    AnnDatas share, so both X matrices get the same ``normalize_total(target_sum=1e4)`` +
    ``log1p``. Only the sc side used to be normalized, which set log values against raw counts
    inside that joint PCA. When the caller names ``sc_attr``, neither X is touched and the payload
    says so.

Background spots: a spatial input whose ``obs['in_tissue']`` marks spots as 0 (CELLxGENE Visium
exports carry the whole array) has those spots left out right after loading -- the input of
temporal and alignment, the spatial side of mapping -- and the payload reports how many
(``params.in_tissue_filter``), as every spot-level worker in this fleet does.

Device: ``--device`` is resolved once; the device ``solve()`` actually received is
``params.device_used`` (``params.device`` is the request), and a GPU request that ran on the CPU
is a warning, not only a line on stderr.

Labels: a spot/cell whose ``batch_key`` (alignment, mapping) or ``time_key`` (temporal) is missing
belongs to no subproblem. moscot leaves it out without a word and, for alignment, writes (0, 0) as
its warped coordinates. Such rows are refused unless ``--drop-unlabeled`` is given, in which case
they are dropped before anything is solved and counted in the payload.

Outputs (each written to ``<name>.partial`` and renamed into place, so a killed run leaves no torn
file): every problem type saves the solved problem as a pickle (the couplings live there). Mapping
also writes ``moscot_mapping_cell_to_spot.csv`` -- for every cell, the spot that receives most of its
transported mass -- which is the mapping's prediction; the two h5ads beside it are the inputs as
they entered the solver. Alignment writes ``adata_alignment_aligned.h5ad`` with the warped
coordinates in ``obsm['moscot_spatial_warp']``; ``params.batch_key`` names the obs column that
tells its sections apart and ``params.reference_batch_used`` the section held fixed.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from moscot.problems.space import AlignmentProblem, MappingProblem
from moscot.problems.time import TemporalProblem
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    default_output_dir,
    id_mismatch_msg,
    keep_in_tissue,
    record_ignored,
    record_in_tissue,
    record_method,
    resolve_compute,
    unsupported_choice_msg,
)
from worker_utils import drop_unlabeled as unlabeled_mask  # the run_* functions take a bool of that name

PCA_KEY = "X_moscot_pca"

# Mapping compares the two AnnDatas' X directly (one joint PCA over their shared genes), so both
# are scaled to the same total. scanpy's default -- each AnnData's own median total -- puts a Visium
# slide and a droplet reference on different scales before the log is even taken.
MAPPING_TARGET_SUM = 1e4

# Policies moscot_run can actually run. TemporalProblem also declares 'explicit', but that one
# needs the explicit list of time-point pairs (moscot's ``prepare(subset=...)``), which this tool
# does not take -- it used to be advertised and then fail inside moscot after the PCA.
TEMPORAL_POLICIES = ("sequential", "triu", "tril")
ALIGNMENT_POLICIES = ("sequential", "star")

# The mapping's per-cell table needs each solved coupling as one dense float32 matrix (and a host
# copy of it). The solve without ``batch_size`` already held a matrix that size; with ``batch_size``
# it did not, so the export checks the bytes against the memory actually available first.
CELL_TO_SPOT_CSV = "moscot_mapping_cell_to_spot.csv"
CELL_TO_SPOT_COLUMNS = ["cell", "batch", "spot", "transport_mass", "mass_fraction"]

METHOD_NAMES = {
    "temporal": "moscot TemporalProblem (entropic optimal transport between consecutive time points)",
    "alignment": "moscot AlignmentProblem (fused Gromov-Wasserstein between sections, warped onto a reference)",
    "mapping": "moscot MappingProblem (fused Gromov-Wasserstein from single cells onto spatial spots)",
}


def _moscot_device(request):
    """Translate the fleet's device vocabulary into the one moscot's own ``solve()`` declares.

    ``TemporalProblem``/``AlignmentProblem``/``MappingProblem.solve`` each take
    ``device: Device_t``, which moscot spells ``"cpu"``/``"gpu"``/``"tpu"`` (optionally suffixed
    ``":N"`` -- ``output.to()`` splits on the colon and indexes ``jax.devices()`` with it). That is
    *not* the string ``resolve_compute`` returns, which says ``"cuda"``/``"cuda:N"``, so the
    resolved value has to be translated rather than passed through; the card index survives.

    ``"tpu"`` has no torch equivalent and therefore no representation in ``resolve_compute``; it is
    forwarded untouched rather than silently downgraded to the CPU.
    """
    if str(request).strip().lower().startswith("tpu"):
        return str(request).strip().lower()
    resolved = resolve_compute(request).device
    return resolved.replace("cuda", "gpu", 1) if resolved.startswith("cuda") else "cpu"


#: Requests that do not ask for an accelerator: resolving any of them to the CPU is what was asked.
_CPU_OR_AUTO_REQUESTS = ("", "cpu", "auto", "default", "any", "none", "false", "no", "off", "-1")


def _device_note(request, used: str) -> str:
    """A warning when an accelerator was asked for and ``solve()`` got the CPU; '' otherwise.

    ``resolve_compute`` degrades such a request and says so on stderr only, which is not where a
    caller reads; the payload used to echo the request as ``params.device`` as though it ran.
    """
    asked = str(request).strip().lower()
    if used != "cpu" or asked in _CPU_OR_AUTO_REQUESTS:
        return ""
    return (
        f"device={request!r} was requested, but no GPU was available to this worker, so moscot's solve() "
        "ran on the CPU (params.device_used='cpu')."
    )


def log(msg: str) -> None:
    print(f"[moscot-worker] {msg}", file=sys.stderr)


def list_output_files(output_dir: str) -> list[str]:
    out_files: list[str] = []
    for root, _, files in os.walk(output_dir):
        for f in files:
            out_files.append(os.path.join(root, f))
    return sorted(out_files)


def _x_holds_counts(X, chunk: int = 1 << 22) -> bool:
    """True when every stored value of X is a non-negative integer, i.e. X looks like raw counts.

    Read in slices so a large dense X is never copied whole; every value is looked at.
    """
    import scipy.sparse as sp_sparse

    if sp_sparse.issparse(X):
        data = np.asarray(X.data).ravel()
        for start in range(0, data.size, chunk):
            d = data[start : start + chunk]
            if d.size and (np.nanmin(d) < 0 or not np.all(np.floor(d) == d)):
                return False
        return True
    n_rows = X.shape[0]
    n_cols = max(1, X.shape[1] if len(X.shape) > 1 else 1)
    step = max(1, chunk // n_cols)
    for start in range(0, n_rows, step):
        d = np.asarray(X[start : start + step])
        if d.size and (np.nanmin(d) < 0 or not np.all(np.floor(d) == d)):
            return False
    return True


def _normalize_log1p(adata: ad.AnnData, what: str, report: dict, target_sum: float | None = None) -> ad.AnnData:
    """``normalize_total`` + ``log1p`` on X, in place, and say what the input looked like.

    Applied whatever X holds. An X that did not hold integer counts gets a warning in
    ``report['warnings']``: if it was already log-transformed it is now log-transformed twice.
    ``target_sum`` None is scanpy's default (every observation scaled to the median total of this
    AnnData); mapping passes one fixed value to both sides so they land on the same scale.
    """
    counts = _x_holds_counts(adata.X)
    log(f"normalize_total(target_sum={target_sum}) + log1p on the {what} X (input holds integer counts: {counts})")
    sc.pp.normalize_total(adata, target_sum=target_sum, inplace=True)
    sc.pp.log1p(adata)

    # Safety: replace any NaN/Inf introduced by normalization (a spot with zero counts)
    import scipy.sparse as sp_sparse

    if sp_sparse.issparse(adata.X):
        adata.X.data = np.nan_to_num(adata.X.data, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)

    prep = report.setdefault("preprocessing", {})
    prep[what] = {
        "transform": "normalize_total+log1p",
        "target_sum": "median total of this AnnData" if target_sum is None else float(target_sum),
        "input_was_integer_counts": bool(counts),
    }
    if not counts:
        report.setdefault("warnings", []).append(
            f"the {what} X did not hold integer counts, and normalize_total + log1p was applied to it anyway; "
            "if it was already log-transformed it is now log-transformed twice. Pass raw counts in X."
        )
    return adata


def _ensure_normalized_pca(
    adata: ad.AnnData,
    pca_key: str = PCA_KEY,
    report: dict | None = None,
    what: str = "input",
    target_sum: float | None = None,
) -> ad.AnnData:
    """
    Normalize X and store a PCA with a SAFE number of PCs in ``adata.obsm[pca_key]``.

    - Applies normalize_total + log1p to X unconditionally (see ``_normalize_log1p``); it does NOT
      check whether that was already done.
    - Computes PCA with n_comps = min(30, n_obs-1, n_vars-1) (and >= 1), on
      ``var['highly_variable']`` when present (scanpy's default), otherwise on every gene.
    - Stores the embedding in adata.obsm[pca_key].
    """
    report = {} if report is None else report
    log(f"Ensuring normalized/log1p + PCA embedding for MOSCOT on the {what} AnnData...")
    adata = _normalize_log1p(adata, what, report, target_sum=target_sum)

    n_obs, n_vars = adata.n_obs, adata.n_vars
    max_comps = min(30, n_obs - 1, n_vars - 1)
    if max_comps < 2:
        # very tiny toy data; just use 1 component
        n_comps = max(1, max_comps)
    else:
        n_comps = max_comps

    hv = adata.var["highly_variable"] if "highly_variable" in adata.var.columns else None
    n_pca_genes = int(np.asarray(hv, dtype=bool).sum()) if hv is not None else int(n_vars)

    log(f"Running PCA with n_comps={n_comps} (n_obs={n_obs}, n_vars={n_vars}, genes used={n_pca_genes})...")
    sc.pp.pca(adata, n_comps=n_comps)

    # Scanpy stores PCA in obsm["X_pca"]; copy to our key
    adata.obsm[pca_key] = adata.obsm["X_pca"].copy()
    log(f"PCA embedding stored in obsm['{pca_key}'] with shape {adata.obsm[pca_key].shape}")

    prep = report.setdefault("preprocessing", {})
    prep.setdefault(what, {}).update(
        {
            "pca_key": pca_key,
            "pca_n_comps": int(n_comps),
            "pca_genes": "var['highly_variable']" if hv is not None else "all",
            "pca_n_genes": n_pca_genes,
        }
    )
    return adata


def _drop_unlabelled_obs(adata: ad.AnnData, key: str, allow_drop: bool, role: str, what: str, report: dict):
    """Refuse (or, with ``allow_drop``, drop) observations whose ``obs[key]`` is missing.

    moscot builds its subproblems from the categories of ``obs[key]``; a row with no value belongs
    to none of them and is left out without a word -- and ``AlignmentProblem.align`` then writes
    (0, 0) as its warped coordinates.
    """
    if key not in adata.obs.columns:
        cols = list(map(str, adata.obs.columns))
        raise ValueError(f"{role}={key!r} is not a column of obs. Available obs columns: {cols[:40]}")
    keep, n_dropped = unlabeled_mask(adata.obs[key].to_numpy(), allow_drop, what=f"{what} (obs[{key!r}])")
    report.setdefault("data", {})["n_dropped_unlabeled"] = int(n_dropped)
    if n_dropped:
        log(f"drop_unlabeled: leaving out {n_dropped} {what} with no obs[{key!r}]")
        report.setdefault("warnings", []).append(
            f"drop_unlabeled=True: {n_dropped} of {adata.n_obs} {what} had no obs[{key!r}] and were left out before "
            "solving; they appear in no output."
        )
        adata = adata[np.asarray(keep, dtype=bool)].copy()
    return adata, int(n_dropped)


def _as_string_category(adata: ad.AnnData, key: str, report: dict) -> None:
    """Store ``obs[key]`` as a categorical of strings, in moscot's own section order, none unused.

    ``AlignmentProblem.align`` needs exactly this, and the worker used to hand it anything else:

    * it reads ``obs[batch_key].cat.categories``, so an integer or plain-string column raised
      ``AttributeError`` -- after the whole optimal-transport solve;
    * its interpolation treats a non-``str`` reference as a list of references, so an integer
      section id raised ``TypeError: 'int' object is not iterable``;
    * the command line delivers ``reference_batch`` as a string, which moscot compares against the
      categories as they are: ``'0'`` against an integer column is "not in policy's categories".

    The order is kept: moscot orders sections by ``astype('category')`` of the original values
    (numeric order for numbers), and the 'sequential' policy pairs neighbours in that order, so the
    string labels are laid out in it rather than re-sorted as text ('10' before '2'). A section
    declared but holding no cell would be a subproblem with no cells, so unused categories go.
    """
    col = adata.obs[key]
    was = str(col.dtype)
    cats = (
        list(col.cat.categories)
        if isinstance(col.dtype, pd.CategoricalDtype)
        else list(col.astype("category").cat.categories)
    )
    counts = col.value_counts()
    used = [c for c in cats if int(counts.get(c, 0)) > 0]
    names = [str(c) for c in used]
    if len(set(names)) != len(names):
        raise ValueError(f"obs[{key!r}] has sections that are distinct values but print the same: {names[:50]}.")
    already = (
        isinstance(col.dtype, pd.CategoricalDtype) and all(isinstance(c, str) for c in cats) and len(used) == len(cats)
    )
    if already:
        return
    adata.obs[key] = pd.Categorical(col.astype(str), categories=names)
    unused = [str(c) for c in cats if int(counts.get(c, 0)) == 0]
    recast: dict = {"from": was, "to": "category of str"}
    if unused:
        recast["unused_categories_removed"] = unused
    report.setdefault("params", {})["batch_key_recast"] = recast
    log(f"obs[{key!r}] ({was}) stored as a categorical of strings in section order {names[:20]}")


def _match_level(levels: list, requested: Any, batch_key: str) -> Any:
    """Return the level of ``obs[batch_key]`` the caller named.

    Matched on its printed form, which after ``_as_string_category`` is the level itself.
    """
    by_name: dict = {}
    for lev in levels:
        by_name.setdefault(str(lev), lev)
    if str(requested) in by_name:
        return by_name[str(requested)]
    raise ValueError(
        f"reference_batch={requested!r} is not a level of obs[{batch_key!r}]. "
        f"Its levels are: {[str(v) for v in levels][:50]}."
    )


def _solver_default(problem_cls, name: str) -> Any:
    """The default moscot's own ``solve()`` uses for ``name`` (None when it declares none)."""
    try:
        param = inspect.signature(problem_cls.solve).parameters.get(name)
    except (TypeError, ValueError):
        return None
    if param is None or param.default is inspect.Parameter.empty:
        return None
    return param.default


def _describe_solutions(problem, report: dict) -> None:
    """Record, per solved subproblem, its size, whether the solver converged, and its cost."""
    rows = []
    not_converged = []
    for key, sol in dict(getattr(problem, "solutions", {}) or {}).items():
        row: dict = {"key": [str(k) for k in key] if isinstance(key, tuple) else str(key)}
        try:
            row["shape"] = [int(s) for s in sol.shape]
        except Exception:
            pass
        try:
            row["converged"] = bool(sol.converged)
        except Exception:
            row["converged"] = None
        try:
            row["cost"] = float(sol.cost)
        except Exception:
            row["cost"] = None
        if row.get("converged") is False:
            not_converged.append(row["key"])
        rows.append(row)
    summary = report.setdefault("summary", {})
    summary["subproblems"] = rows
    summary["n_subproblems"] = len(rows)
    summary["n_not_converged"] = len(not_converged)
    if not_converged:
        report.setdefault("warnings", []).append(
            f"the solver did not converge for {len(not_converged)} of {len(rows)} subproblem(s) {not_converged[:10]}; "
            "those couplings are not optimal transport plans and may hold NaN. A larger epsilon converges faster."
        )


def _mem_available_bytes() -> float | None:
    """Memory this worker can still allocate: the fleet's one reader (page cache counted reclaimable,
    the cgroup limit respected), not a private MemAvailable parse."""
    return available_memory_bytes()


def _write_csv_atomic(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".partial"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _write_h5ad_atomic(adata: ad.AnnData, path: str) -> None:
    tmp = path + ".partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def _save_problem_atomic(problem, path: str) -> None:
    """``problem.save`` (cloudpickle) through a ``.partial`` sibling, then rename into place."""
    tmp = path + ".partial"
    problem.save(tmp, overwrite=True)
    os.replace(tmp, path)


def _in_tissue(adata: ad.AnnData, report: dict) -> ad.AnnData:
    """Leave out background spots (``obs['in_tissue'] == 0``) and note how many, for the payload."""
    adata, n_supplied, n_off = keep_in_tissue(adata, "spots")
    if n_off:
        log(f"in_tissue: leaving out {n_off} of {n_supplied} spots marked obs['in_tissue'] == 0 (background)")
    report["in_tissue"] = {"n_supplied": int(n_supplied), "n_dropped": int(n_off)}
    return adata


def _export_cell_to_spot(problem, batch_key: str | None, path: str, report: dict) -> str | None:
    """Write, for every single cell, the spot that receives most of its transported mass.

    One row per (cell, spatial batch): with ``batch_key`` moscot solves one coupling per spatial
    batch against every cell. ``transport_mass`` is the coupling entry of that (spot, cell) pair and
    ``mass_fraction`` its share of all the mass the cell sends. Returns the path, or None when the
    coupling does not fit in memory (the payload then says so; the coupling is still in the pickle).
    """
    solutions = dict(getattr(problem, "solutions", {}) or {})
    need = 0
    for sol in solutions.values():
        n_src, n_tgt = (int(s) for s in sol.shape)
        need = max(need, n_src * n_tgt * 4 * 3)
    avail = _mem_available_bytes()
    if avail is not None and need > avail // 2:
        report.setdefault("warnings", []).append(
            f"{CELL_TO_SPOT_CSV} was not written: its largest coupling needs about {need / 1e9:.1f} GB as a dense "
            f"matrix and only {avail / 1e9:.1f} GB is available. The couplings are in the pickled problem "
            "(MappingProblem.load(...).solutions)."
        )
        return None

    frames = []
    for key, sol in solutions.items():
        sub = problem.problems[key]
        spots = np.asarray(sub.adata_src.obs_names).astype(str)
        cells = np.asarray(sub.adata_tgt.obs_names).astype(str)
        T = np.asarray(sol.transport_matrix)
        if T.shape != (len(spots), len(cells)):
            raise ValueError(
                f"coupling {key} has shape {T.shape}, but its subproblem holds {len(spots)} spots x {len(cells)} cells"
            )
        if not np.isfinite(T).all():
            report.setdefault("warnings", []).append(
                f"the coupling for {key} holds non-finite values; its rows in {CELL_TO_SPOT_CSV} are not meaningful"
            )
            T = np.where(np.isfinite(T), T, 0.0)
        best = np.argmax(T, axis=0)
        mass = np.asarray(T[best, np.arange(T.shape[1])], dtype=np.float64)
        total = np.asarray(T.sum(axis=0, dtype=np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(total > 0, mass / total, np.nan)
        batch = str(key[0]) if (batch_key and isinstance(key, tuple)) else ""
        frames.append(
            pd.DataFrame(
                {
                    "cell": cells,
                    "batch": batch,
                    "spot": spots[best],
                    "transport_mass": mass,
                    "mass_fraction": frac,
                },
                columns=CELL_TO_SPOT_COLUMNS,
            )
        )
        del T
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CELL_TO_SPOT_COLUMNS)
    _write_csv_atomic(table, path)
    report.setdefault("summary", {})["n_cell_to_spot_rows"] = int(len(table))
    return path


def run_temporal(
    adata_path: str,
    output_dir: str,
    time_key: str,
    policy: str,
    epsilon: float,
    batch_size: int,
    device: str,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    report: dict = {}
    log(f"Loading AnnData for TemporalProblem from: {adata_path}")
    adata = ad.read_h5ad(adata_path)
    report["data"] = {"n_spots": int(adata.n_obs), "n_genes": int(adata.n_vars)}
    adata = _in_tissue(adata, report)
    adata, _ = _drop_unlabelled_obs(adata, time_key, drop_unlabeled, "time_key", "observations", report)
    report["data"]["n_obs_used"] = int(adata.n_obs)

    # Precompute safe PCA and store in obsm["X_moscot_pca"]
    adata = _ensure_normalized_pca(adata, pca_key=PCA_KEY, report=report, what="input")

    log(
        f"Setting up TemporalProblem(time_key={time_key!r}, "
        f"policy={policy!r}, epsilon={epsilon}, batch_size={batch_size}, device={device!r}, "
        f"joint_attr='X_moscot_pca')"
    )

    tp = TemporalProblem(adata)
    tp = tp.prepare(
        time_key=time_key,
        policy=policy,
        joint_attr=PCA_KEY,
    )
    # device= was resolved, logged and recorded in params, but never reached the solver, so the
    # portal's documented "passed to the underlying JAX/XLA backend" was not true of any run.
    tp = tp.solve(epsilon=epsilon, batch_size=batch_size or None, device=_moscot_device(device))
    _describe_solutions(tp, report)

    problem_path = os.path.join(output_dir, "moscot_temporal_problem.pkl")
    log(f"Saving TemporalProblem to: {problem_path}")
    _save_problem_atomic(tp, problem_path)

    adata_out_path = os.path.join(output_dir, "adata_temporal_with_pca.h5ad")
    log(f"Saving AnnData (with PCA) to: {adata_out_path}")
    _write_h5ad_atomic(adata, adata_out_path)

    return {
        "problem_path": problem_path,
        "adata_path": adata_out_path,
        "facts": report,
    }


def run_alignment(
    adata_path: str,
    output_dir: str,
    batch_key: str,
    spatial_key: str,
    policy: str,
    reference_batch: str | None,
    epsilon: float,
    batch_size: int,
    device: str,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    report: dict = {}
    log(f"Loading AnnData for AlignmentProblem from: {adata_path}")
    adata = ad.read_h5ad(adata_path)
    report["data"] = {"n_spots": int(adata.n_obs), "n_genes": int(adata.n_vars)}
    adata = _in_tissue(adata, report)
    adata, _ = _drop_unlabelled_obs(adata, batch_key, drop_unlabeled, "batch_key", "spots", report)
    report["data"]["n_obs_used"] = int(adata.n_obs)
    _as_string_category(adata, batch_key, report)

    # The reference_batch trap, closed 2026-09-21 (D-063).
    #
    # ap.align() was called only when the caller named a reference, and the portal defaulted that
    # to None. So moscot_run(problem_type="alignment", ...) with the shipped defaults solved the
    # optimal-transport problem, wrote NO aligned AnnData, and returned status ok -- a silent
    # Phase-2 no-op that looks exactly like a successful alignment from the outside.
    #
    # An alignment needs a slice held fixed, and with none named the first level of batch_key is
    # the only defensible choice. It is taken, and said out loud in the payload rather than
    # assumed silently; a skipped align() is now a warning that names its cause instead of a bare
    # ok.
    #
    # The reference is resolved BEFORE prepare, not after the solve: policy='star' is a star
    # around that very section, and moscot's prepare() refuses a star with no reference -- so
    # 'star' failed on every call while the reference sat unused until ap.align(). The column is a
    # categorical of strings by now (_as_string_category), so '0' from the command line names an
    # integer section 0, and an unknown name is refused before the solve rather than after it.
    adata_aligned_path = None
    align_note = ""
    levels = list(dict.fromkeys(adata.obs[batch_key].tolist()))
    if len(levels) < 2:
        raise ValueError(
            f"alignment needs at least 2 sections in obs[{batch_key!r}]; found {len(levels)}: "
            f"{[str(v) for v in levels]}. Concatenate the sections into one AnnData with a column naming each."
        )
    reference = None
    if reference_batch is None:
        reference = levels[0]
        align_note = (
            f"no reference_batch was given, so the first level of obs[{batch_key!r}] "
            f"({str(reference)!r}) was held fixed. Every other section is warped onto it; naming a "
            f"different reference changes which section does not move."
        )
    else:
        reference = _match_level(levels, reference_batch, batch_key)

    # Precompute safe PCA and store in obsm["X_moscot_pca"]
    adata = _ensure_normalized_pca(adata, pca_key=PCA_KEY, report=report, what="input")

    log(
        f"Setting up AlignmentProblem(batch_key={batch_key!r}, spatial_key={spatial_key!r}, "
        f"policy={policy!r}, reference_batch={reference!r}, "
        f"epsilon={epsilon}, batch_size={batch_size}, device={device!r}, "
        f"joint_attr='X_moscot_pca')"
    )

    prepare_kwargs: dict = {
        "batch_key": batch_key,
        "spatial_key": spatial_key,
        "policy": policy,
        "joint_attr": PCA_KEY,
    }
    if policy == "star":
        # The hub of the star; ap.align() below warps onto the same section.
        prepare_kwargs["reference"] = reference
    ap = AlignmentProblem(adata=adata)
    ap = ap.prepare(**prepare_kwargs)
    ap = ap.solve(epsilon=epsilon, batch_size=batch_size or None, device=_moscot_device(device))
    _describe_solutions(ap, report)

    problem_path = os.path.join(output_dir, "moscot_alignment_problem.pkl")
    log(f"Saving AlignmentProblem to: {problem_path}")
    _save_problem_atomic(ap, problem_path)

    if reference is not None:
        log(f"Running ap.align(key_added='moscot_spatial_warp', reference={reference!r})")
        ap.align(key_added="moscot_spatial_warp", reference=reference)
        adata_aligned_path = os.path.join(output_dir, "adata_alignment_aligned.h5ad")
        log(f"Saving aligned AnnData to: {adata_aligned_path}")
        _write_h5ad_atomic(ap.adata, adata_aligned_path)
    if align_note:
        log(align_note)

    # Save input-with-PCA as well for completeness
    adata_in_path = os.path.join(output_dir, "adata_alignment_with_pca.h5ad")
    log(f"Saving AnnData (with PCA) to: {adata_in_path}")
    _write_h5ad_atomic(adata, adata_in_path)

    return {
        "problem_path": problem_path,
        "adata_aligned_path": adata_aligned_path,
        "reference_batch_used": None if reference is None else str(reference),
        "align_note": align_note,
        "adata_with_pca_path": adata_in_path,
        "facts": report,
    }


def run_mapping(
    adata_spatial_path: str,
    adata_sc_path: str,
    output_dir: str,
    batch_key: str | None,
    spatial_key: str,
    sc_attr: str | None,
    alpha: float,
    epsilon: float,
    batch_size: int,
    device: str,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    report: dict = {}
    log(f"Loading spatial AnnData for MappingProblem from: {adata_spatial_path}")
    adata_sp = ad.read_h5ad(adata_spatial_path)

    log(f"Loading single-cell AnnData for MappingProblem from: {adata_sc_path}")
    adata_sc = ad.read_h5ad(adata_sc_path)

    report["data"] = {
        "n_spots": int(adata_sp.n_obs),
        "n_genes": int(adata_sp.n_vars),
        "n_cells": int(adata_sc.n_obs),
        "n_sc_genes": int(adata_sc.n_vars),
    }
    adata_sp = _in_tissue(adata_sp, report)
    if batch_key:
        adata_sp, _ = _drop_unlabelled_obs(adata_sp, batch_key, drop_unlabeled, "batch_key", "spots", report)
    report["data"]["n_obs_used"] = int(adata_sp.n_obs)

    shared = adata_sp.var_names.intersection(adata_sc.var_names)
    report["data"]["n_shared_genes"] = int(len(shared))
    if len(shared) == 0:
        raise ValueError(
            id_mismatch_msg(
                "genes", "spatial var_names", adata_sp.var_names, "single-cell var_names", adata_sc.var_names
            )
        )

    # The expression (linear) term: with var_names unset, MappingProblem.prepare takes the shared
    # genes of BOTH X matrices and runs one joint PCA over them ('local-pca' on
    # vstack(adata_sp[:, shared].X, adata_sc[:, shared].X)). Only the sc side used to be
    # normalized, so that PCA set log values against raw counts. Both sides now get the same
    # transform; with sc_attr given, neither is touched.
    local_sc_attr = sc_attr
    if sc_attr is None:
        log("No sc_attr provided for MappingProblem; normalizing both AnnDatas and computing PCA for the sc one...")
        adata_sp = _normalize_log1p(adata_sp, "spatial", report, target_sum=MAPPING_TARGET_SUM)
        adata_sc = _ensure_normalized_pca(
            adata_sc, pca_key=PCA_KEY, report=report, what="sc", target_sum=MAPPING_TARGET_SUM
        )
        local_sc_attr = PCA_KEY
    else:
        prep = report.setdefault("preprocessing", {})
        untouched = "none: sc_attr was given, so X is used as supplied"
        sp_counts = _x_holds_counts(adata_sp.X)
        sc_counts = _x_holds_counts(adata_sc.X)
        prep["spatial"] = {"transform": untouched, "input_was_integer_counts": bool(sp_counts)}
        prep["sc"] = {"transform": untouched, "input_was_integer_counts": bool(sc_counts), "sc_attr": sc_attr}
        if sp_counts != sc_counts:
            report.setdefault("warnings", []).append(
                "the expression term compares X across both AnnDatas over their shared genes, and "
                f"{'the spatial' if sp_counts else 'the sc'} X holds integer counts while the other does not. "
                "With sc_attr given the worker does not normalize either; bring both to the same scale first."
            )

    log(
        f"Setting up MappingProblem(sc_attr={local_sc_attr!r}, batch_key={batch_key!r}, "
        f"spatial_key={spatial_key!r}, alpha={alpha}, epsilon={epsilon}, "
        f"batch_size={batch_size}, device={device!r})"
    )

    mp = MappingProblem(adata_sc, adata_sp)
    mp = mp.prepare(
        sc_attr=local_sc_attr,
        batch_key=batch_key or None,
        spatial_key=spatial_key,
    )
    mp = mp.solve(alpha=alpha, epsilon=epsilon, batch_size=batch_size or None, device=_moscot_device(device))
    _describe_solutions(mp, report)

    problem_path = os.path.join(output_dir, "moscot_mapping_problem.pkl")
    log(f"Saving MappingProblem to: {problem_path}")
    _save_problem_atomic(mp, problem_path)

    cell_to_spot_path = _export_cell_to_spot(mp, batch_key, os.path.join(output_dir, CELL_TO_SPOT_CSV), report)

    # The AnnDatas as they entered the solver (intermediate, not a prediction)
    adata_sp_out = os.path.join(output_dir, "adata_mapping_spatial.h5ad")
    adata_sc_out = os.path.join(output_dir, "adata_mapping_sc_with_pca.h5ad")
    log(f"Saving spatial AnnData to: {adata_sp_out}")
    log(f"Saving sc AnnData (with PCA if computed) to: {adata_sc_out}")
    _write_h5ad_atomic(adata_sp, adata_sp_out)
    _write_h5ad_atomic(adata_sc, adata_sc_out)

    return {
        "problem_path": problem_path,
        "adata_spatial_path": adata_sp_out,
        "adata_sc_path": adata_sc_out,
        "cell_to_spot_csv": cell_to_spot_path,
        "facts": report,
    }


def _ignored_parameters(args: argparse.Namespace, spatial_path: str | None) -> list[tuple[list[str], str]]:
    """Parameters the caller sent that the chosen problem type never reads, with the reason."""
    pt = args.problem_type
    groups: list[tuple[list[str], str]] = []
    if pt != "mapping":
        names = [
            n
            for n, v in (
                ("adata_spatial_path", args.adata_spatial_path),
                ("adata_sc_path", args.adata_sc_path),
                ("sc_attr", args.sc_attr),
            )
            if v is not None
        ]
        if names:
            groups.append((names, f"they apply to problem_type='mapping' only, not {pt!r}"))
        if pt == "alignment":
            groups.append(
                (
                    ["alpha"],
                    "moscot_run forwards alpha to MappingProblem.solve only; AlignmentProblem.solve ran with "
                    f"moscot's own alpha={_solver_default(AlignmentProblem, 'alpha')} (params.solver_alpha)",
                )
            )
        else:
            groups.append((["alpha"], "TemporalProblem is a linear problem and has no alpha"))
    if pt != "temporal" and args.time_key is not None:
        groups.append((["time_key"], f"it applies to problem_type='temporal' only, not {pt!r}"))
    if pt != "alignment" and args.reference_batch is not None:
        groups.append((["reference_batch"], f"it applies to problem_type='alignment' only, not {pt!r}"))
    if pt == "temporal":
        if args.batch_key is not None:
            groups.append((["batch_key"], "TemporalProblem pairs observations by time_key, not by batch"))
        groups.append((["spatial_key"], "TemporalProblem does not read spatial coordinates"))
    if pt == "mapping":
        if args.policy is not None:
            groups.append((["policy"], "MappingProblem has no policy; every spatial batch is mapped against all cells"))
        if (
            spatial_path is not None
            and args.adata_spatial_path
            and os.path.abspath(args.adata_spatial_path) != os.path.abspath(args.adata_path)
        ):
            groups.append((["adata_path"], "adata_spatial_path was given and is the spatial AnnData that was mapped"))
        if args.drop_unlabeled and not args.batch_key:
            groups.append((["drop_unlabeled"], "mapping reads no label column unless batch_key is given"))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="MOSCOT worker for SpatialOmicsLab MCP.")
    parser.add_argument(
        "--problem-type",
        required=True,
        choices=["temporal", "alignment", "mapping"],
        help="Which MOSCOT problem to run.",
    )
    parser.add_argument(
        "--adata-path",
        required=True,
        help=(
            "Path to AnnData (.h5ad) for temporal/alignment problems. "
            "For mapping, this is the spatial AnnData if --adata-spatial-path is not given."
        ),
    )
    parser.add_argument(
        "--adata-spatial-path",
        required=False,
        help="Spatial AnnData (.h5ad) for mapping problem. Defaults to --adata-path.",
    )
    parser.add_argument(
        "--adata-sc-path",
        required=False,
        help="Single-cell AnnData (.h5ad) for mapping problem.",
    )

    parser.add_argument("--time-key", required=False, help="obs key for time points.")
    parser.add_argument("--batch-key", required=False, help="obs key for batches/slides.")
    parser.add_argument(
        "--spatial-key",
        required=False,
        default="spatial",
        help="obsm key with spatial coordinates (default: 'spatial'). Not read by temporal problems.",
    )
    parser.add_argument(
        "--policy",
        required=False,
        help=(
            "Temporal: one of {'sequential','triu','tril'} ('explicit' needs time-point pairs this tool does not "
            "take); Alignment: one of {'sequential','star'} ('star' aligns every section to the reference). "
            "Default 'sequential'. Mapping has no policy."
        ),
    )
    parser.add_argument(
        "--reference-batch",
        required=False,
        help=(
            "For alignment: obs[batch_key] level held fixed. It is the hub of policy='star' "
            "(AlignmentProblem.prepare(reference=...)) and the target of AlignmentProblem.align(...). "
            "Omitted: the first level in order of appearance, reported as params.reference_batch_used."
        ),
    )
    parser.add_argument(
        "--sc-attr",
        required=False,
        default=None,
        help=(
            "For mapping problems: value passed to MappingProblem.prepare(sc_attr=...). "
            "If None, the worker normalizes both AnnDatas' X identically (normalize_total+log1p), computes "
            "a PCA for the sc AnnData and uses 'X_moscot_pca'. If given, neither X is normalized."
        ),
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        default=False,
        help=(
            "Drop observations whose time_key (temporal) or batch_key (alignment, mapping) is missing "
            "instead of refusing the run. moscot leaves such rows out of every subproblem, and alignment "
            "writes (0, 0) as their warped coordinates."
        ),
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Entropic regularization parameter for solve().",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="For mapping problems: alpha parameter in MappingProblem.solve(). "
        "Ignored for temporal and alignment problems (alignment runs with moscot's own alpha, 0.5); "
        "listed in params.ignored there.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=("Optional batch size passed to .solve(). 0 means 'let MOSCOT decide' (we pass None)."),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computations (e.g. 'cpu', 'gpu').",
    )

    parser.add_argument(
        "--output-dir",
        required=False,
        help=("Directory where all outputs will be written. If omitted, use /workspace/work/moscot_<out_tag>."),
    )
    parser.add_argument(
        "--out-tag",
        required=False,
        default="moscot_run",
        help="Tag used to derive default output_dir if not provided.",
    )

    args = parser.parse_args()

    # Resolve output_dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = default_output_dir(f"moscot_{args.out_tag}")
    os.makedirs(output_dir, exist_ok=True)

    log(f"Problem type: {args.problem_type}")
    log(f"Output directory: {output_dir}")

    try:
        spatial_path = None
        policy = None
        # Resolved once, here, so the device solve() receives is the one the payload reports.
        device_used = _moscot_device(args.device)
        if args.problem_type == "temporal":
            if args.time_key is None:
                raise ValueError("--time-key is required for problem_type='temporal'.")

            policy = args.policy or "sequential"
            if policy not in TEMPORAL_POLICIES:
                raise ValueError(
                    unsupported_choice_msg(
                        "policy",
                        policy,
                        list(TEMPORAL_POLICIES),
                        extra=(
                            "'explicit' needs the explicit list of time-point pairs (moscot's prepare(subset=...)), "
                            "which moscot_run does not take."
                            if policy == "explicit"
                            else ""
                        ),
                    )
                )
            res = run_temporal(
                adata_path=args.adata_path,
                output_dir=output_dir,
                time_key=args.time_key,
                policy=policy,
                epsilon=args.epsilon,
                batch_size=args.batch_size,
                device=device_used,
                drop_unlabeled=args.drop_unlabeled,
            )
            task_name = "temporal"

        elif args.problem_type == "alignment":
            if args.batch_key is None:
                raise ValueError("--batch-key is required for problem_type='alignment'.")

            policy = args.policy or "sequential"
            if policy not in ALIGNMENT_POLICIES:
                raise ValueError(unsupported_choice_msg("policy", policy, list(ALIGNMENT_POLICIES)))
            res = run_alignment(
                adata_path=args.adata_path,
                output_dir=output_dir,
                batch_key=args.batch_key,
                spatial_key=args.spatial_key,
                policy=policy,
                reference_batch=args.reference_batch,
                epsilon=args.epsilon,
                batch_size=args.batch_size,
                device=device_used,
                drop_unlabeled=args.drop_unlabeled,
            )
            task_name = "alignment"

        elif args.problem_type == "mapping":
            spatial_path = args.adata_spatial_path or args.adata_path
            if args.adata_sc_path is None:
                raise ValueError("--adata-sc-path is required for problem_type='mapping'.")

            res = run_mapping(
                adata_spatial_path=spatial_path,
                adata_sc_path=args.adata_sc_path,
                output_dir=output_dir,
                batch_key=args.batch_key,
                spatial_key=args.spatial_key,
                sc_attr=args.sc_attr,
                alpha=args.alpha,
                epsilon=args.epsilon,
                batch_size=args.batch_size,
                device=device_used,
                drop_unlabeled=args.drop_unlabeled,
            )
            task_name = "mapping"

        else:
            raise ValueError(
                unsupported_choice_msg("problem_type", args.problem_type, ["temporal", "alignment", "mapping"])
            )

        out = WorkerOutput("moscot", task=task_name)

        # These are facts about the run, not files. They are lifted out before the loop below,
        # which turns every remaining entry into an output path -- a note left in `res` would be
        # registered as an artefact that does not exist.
        align_note = res.pop("align_note", "") or ""
        reference_used = res.pop("reference_batch_used", None)
        facts = res.pop("facts", {}) or {}
        data = dict(facts.get("data", {}))
        n_spots = int(data.get("n_spots", 0))
        n_genes = int(data.get("n_genes", 0))
        out.set_data(**data)
        if align_note:
            out.add_warning(align_note)
        if reference_used is not None:
            out.add_param("reference_batch_used", reference_used)

        # Build output_files with semantic keys from res paths
        semantic_files = {}
        for key, path in res.items():
            if path is not None:
                # Convert keys like "problem_path" -> "problem_pkl", "adata_path" -> "adata_h5ad"
                sem_key = key.replace("_path", "")
                if sem_key.endswith("_pkl") or "problem" in sem_key:
                    sem_key = sem_key if sem_key.endswith("_pkl") else sem_key + "_pkl"
                elif str(path).endswith(".h5ad"):
                    sem_key = sem_key if sem_key.endswith("_h5ad") else sem_key + "_h5ad"
                semantic_files[sem_key] = path
        out.add_output_files(semantic_files)

        out.add_params(
            {
                "output_dir": output_dir,
                "problem_type": args.problem_type,
                "epsilon": args.epsilon,
                "batch_size": args.batch_size,
                # The request, and beside it what solve() received: resolve_compute turns a GPU request
                # on a box without one into the CPU and used to say so on stderr only.
                "device": args.device,
                "device_used": device_used,
                # The policy that ran (the omitted default used to be reported as None). Mapping has none.
                "policy": policy,
                "drop_unlabeled": bool(args.drop_unlabeled),
                "preprocessing": facts.get("preprocessing", {}),
            }
        )
        out.add_params(facts.get("params", {}))
        device_note = _device_note(args.device, device_used)
        if device_note:
            out.add_warning(device_note)
        in_tissue = facts.get("in_tissue") or {}
        record_in_tissue(out, int(in_tissue.get("n_supplied", n_spots)), int(in_tissue.get("n_dropped", 0)))
        # The label columns the run was split by. A reader of adata_alignment_aligned.h5ad needs the
        # section column to tell its sections apart: without it the merged object reads as one plane.
        if task_name == "temporal":
            out.add_param("time_key", args.time_key)
        elif args.batch_key is not None:
            out.add_param("batch_key", args.batch_key)
        if task_name != "temporal":
            out.add_param("spatial_key", args.spatial_key)
        if task_name == "mapping":
            # alpha reaches MappingProblem.solve only; echoing it for the other two said it was applied.
            out.add_param("alpha", args.alpha)
            out.add_param("sc_attr_used", args.sc_attr if args.sc_attr is not None else PCA_KEY)
        if task_name == "alignment":
            out.add_param("solver_alpha", _solver_default(AlignmentProblem, "alpha"))
        record_method(out, METHOD_NAMES[task_name])
        for names, why in _ignored_parameters(args, spatial_path):
            record_ignored(out, names, why)
        out.add_warnings(facts.get("warnings", []))

        summary = dict(facts.get("summary", {}))
        n_not_converged = int(summary.get("n_not_converged", 0))
        n_sub = int(summary.get("n_subproblems", 0))
        n_used = int(data.get("n_obs_used", n_spots))
        conv = (
            f" The solver did not converge for {n_not_converged} of {n_sub} subproblem(s)."
            if n_not_converged
            else f" All {n_sub} subproblem(s) converged."
        )
        if task_name == "temporal":
            message = (
                f"TemporalProblem solved {n_sub} time-point pair(s) over {n_used} of {n_spots} observations x "
                f"{n_genes} genes (policy={policy}, epsilon={args.epsilon}).{conv} The couplings live only in "
                "moscot_temporal_problem.pkl (TemporalProblem.load); no per-cell table is written."
            )
            roles = {"problem_pkl": "result: the solved couplings", "adata_h5ad": "intermediate: input with PCA"}
        elif task_name == "alignment":
            message = (
                f"AlignmentProblem solved {n_sub} section pair(s) over {n_used} of {n_spots} spots x {n_genes} genes "
                f"(policy={policy}, epsilon={args.epsilon}, alpha={_solver_default(AlignmentProblem, 'alpha')}, "
                f"reference={reference_used}).{conv} Warped coordinates for every section are in "
                "obsm['moscot_spatial_warp'] of adata_alignment_aligned.h5ad."
            )
            roles = {
                "problem_pkl": "the solved couplings",
                "adata_aligned_h5ad": "result: obsm['moscot_spatial_warp']",
                "adata_with_pca_h5ad": "intermediate: input with PCA",
            }
        else:
            wrote = semantic_files.get("cell_to_spot_csv")
            as_entered = (
                "normalize_total(target_sum=1e4) + log1p on both"
                if args.sc_attr is None
                else "an unmodified copy of the spatial input and the sc input as given"
            )
            message = (
                f"MappingProblem mapped {int(data.get('n_cells', 0))} cells onto {n_used} spots over "
                f"{int(data.get('n_shared_genes', 0))} shared genes "
                f"(alpha={args.alpha}, epsilon={args.epsilon}).{conv} "
                + (
                    f"{CELL_TO_SPOT_CSV} names, for each cell, the spot that receives most of its transported mass; "
                    if wrote
                    else f"{CELL_TO_SPOT_CSV} was NOT written (see warnings); "
                )
                + "the full coupling lives in moscot_mapping_problem.pkl. adata_mapping_spatial.h5ad and "
                f"adata_mapping_sc_with_pca.h5ad are the inputs as they entered the solver ({as_entered})."
            )
            roles = {
                "problem_pkl": "the solved couplings",
                "cell_to_spot_csv": "result: cell -> spot assignment",
                "adata_spatial_h5ad": "intermediate: spatial input as it entered the solver",
                "adata_sc_h5ad": "intermediate: sc input as it entered the solver",
            }
        summary["output_roles"] = {k: v for k, v in roles.items() if k in semantic_files}
        out.set_summary(problem_type=task_name, n_output_files=len(semantic_files), **summary)
        out.set_analysis(message)
        out.emit()

    except Exception as e:
        log("EXCEPTION during MOSCOT worker run:")
        import traceback

        tb_str = traceback.format_exc()
        log(tb_str)
        WorkerOutput.emit_error("moscot", str(e), task=args.problem_type)
        sys.exit(1)


if __name__ == "__main__":
    main()
