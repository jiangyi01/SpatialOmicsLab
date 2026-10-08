#!/usr/bin/env python3
"""
novoSpaRc worker for spatial gene expression reconstruction.

Runs inside /opt/conda/envs/novosparc conda env.

novoSpaRc reconstructs spatial gene expression from scRNA-seq data using
optimal transport (upstream ``novosparc.cm.Tissue``). Three modes, chosen by
the inputs and ``alpha`` and named in ``params.mode``:

  atlas-guided      ``st_h5ad`` given and ``alpha > 0``. The reference's
                    expression of the genes it shares with the scRNA-seq HVGs is
                    novoSpaRc's atlas: ``alpha`` weights that linear
                    (cell-to-location expression) cost against the structural
                    Gromov-Wasserstein cost. ``alpha = 1`` is atlas-only OT.
  reference geometry ``st_h5ad`` given and ``alpha = 0``. Only the reference
                    coordinates are used; the mapping is purely structural.
  de novo grid      no ``st_h5ad``. novoSpaRc's target grid, structure only.
                    There is no atlas, so ``alpha`` has nothing to weight: it is
                    reported under ``params.ignored`` (``alpha = 1`` is refused).

With ``st_h5ad``, reference spots whose ``obs['in_tissue']`` is 0 (background glass
around the section, carried by CELLxGENE Visium exports) are left out before the
reference is used -- they are neither target locations nor atlas rows -- and the
payload counts them (``params.in_tissue_filter``). Gene identifiers are harmonised
with ``worker_utils.harmonize_gene_ids`` (reported as ``params.gene_ids_*``).

The cost and coupling matrices are dense (cells x cells, locations x locations,
cells x locations) -- that is intrinsic to the method. The worker estimates their
bytes before allocating and refuses with the numbers when they do not fit. It
never subsamples cells or locations.

``params.converged`` comes from the solver's own record, not from a guess:
upstream's Gromov-Wasserstein loop is called with a log (Tissue.reconstruct passes
none) and every Sinkhorn solve it makes is watched. It is true only when the loop
reached its tolerance and the last Sinkhorn solve converged, false when either
stopped at its iteration limit, and null when the solver could not be traced.

Input: scRNA h5ad + optional reference spatial h5ad
Output: Reconstructed spatial expression h5ad (obs = location IDs), coupling CSV
(rows = cell IDs, columns = location IDs), spatial expression CSV (rows =
location IDs), and, when ``annotation_key`` names an obs column of the scRNA-seq
data, a locations x cell-types table of transported mass.

All logs go to stderr; stdout is JSON-only (final result).
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    drop_unlabeled,
    gene_id_harmonization_note,
    gene_id_harmonization_params,
    harmonize_gene_ids,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    record_ignored,
    record_in_tissue,
    record_method,
    spatial_coords,
)

#: The entropic-regularisation ladder. Upstream's own search (``search_epsilon=True``) looks for the
#: POT warning on captured *stdout*, but POT raises it through ``warnings.warn``, so upstream never
#: sees it; this worker walks the same ladder itself and watches the warnings module.
EPSILON_LADDER = (5e-4, 5e-3, 5e-2, 5e-1)

#: POT's Sinkhorn warnings, as ``ot.bregman.sinkhorn_knopp`` words them.
_NUMERICAL_ERROR_TEXT = "numerical errors"
_NOT_CONVERGED_TEXT = "did not converge"

#: Dense float64 arrays alive at novoSpaRc's two memory peaks, in units of (cells^2, locations^2,
#: cells*locations). Cost set-up: Tissue's placeholder ``np.ones`` costs are still held while
#: dijkstra's shortest-path matrix, its finite-value copy and the normalised cost exist. The
#: Gromov-Wasserstein loop: both structural costs, the marker cost and its normalised copy, T and
#: Tprev, the tensor product and its intermediate, the weighted sum, and Sinkhorn's K and Kp.
#: Checked against tracemalloc on the novosparc env (novosparc 0.4.4, POT 0.9.6): the model reads
#: 8-9% above the measured peak at 1200x800, 800x1200 and 1500x1500, with and without an atlas.
_COST_PHASE_UNITS = (4, 3, 1)
_GW_PHASE_UNITS = (1, 1, 10)

#: The one output this worker adds on top of the three it always wrote.
CELLTYPE_TABLE = "novosparc_celltype_location.csv"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def _ensure_dir(p: str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_dense(X) -> np.ndarray:
    """Convert sparse matrix to dense float32 if needed."""
    try:
        import scipy.sparse as sps

        if sps.issparse(X):
            return X.toarray().astype(np.float32)
    except Exception:
        pass
    return np.asarray(X, dtype=np.float32)


def _sample_values(X) -> np.ndarray:
    """The stored values of the first (up to) 200 rows, as a flat float array."""
    import scipy.sparse as sps

    head = X[: min(200, X.shape[0])]
    if sps.issparse(head):
        return np.asarray(head.data, dtype=np.float64)
    return np.asarray(head, dtype=np.float64).ravel()


def _data_is_preprocessed(X) -> bool:
    """True unless the matrix looks like raw counts (non-negative integers).

    Negative values mean centred/scaled data; fractional values mean normalised and/or
    log-transformed data. Only a matrix of non-negative integers gets ``normalize_total`` +
    ``log1p``.

    The old test was "negative, or more than half the entries nonzero". It sent a *sparse*
    log-normalised matrix -- the ordinary shape of processed scRNA-seq -- through ``log1p`` a second
    time, and handed a dense raw-count matrix (a targeted panel, deep full-length data) to the
    transport unnormalised.
    """
    values = _sample_values(X)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False
    if np.any(values < 0):
        return True
    return bool(np.any(np.abs(values - np.round(values)) > 1e-6))


def _data_is_log_transformed(X) -> bool:
    """Heuristic: check if data looks already log-transformed.

    Raw UMI counts have max in hundreds/thousands; log-transformed data
    typically has max < 15-20.  We also check that the minimum nonzero
    value is not close to an integer (log-transformed values like ln(2)
    are fractional).
    """
    import scipy.sparse as sps

    if sps.issparse(X):
        if X.nnz == 0:
            return False
        max_val = X.data.max()
        min_nz = X.data.min()
    else:
        arr = np.asarray(X)
        max_val = arr.max()
        positives = arr[arr > 0]
        min_nz = positives.min() if len(positives) > 0 else 0

    # If max is small and min nonzero is not integer-like, likely log-transformed
    return max_val < 20 and not np.isclose(min_nz, round(min_nz))


def _has_negative(X) -> bool:
    """True when any stored value is negative (centred/scaled data). Checks the whole matrix."""
    import scipy.sparse as sps

    data = X.data if sps.issparse(X) else np.asarray(X)
    if data.size == 0:
        return False
    return bool(np.nanmin(data) < 0)


def _rows_share_a_total(X, rtol: float = 1e-3) -> bool:
    """True when the sampled non-empty rows all sum to one total: a library-size-normalised matrix
    still on a linear scale (CPM, TPM over every gene, proportions). Log-transformed rows never do.
    """
    head = X[: min(200, X.shape[0])]
    sums = np.asarray(head.sum(axis=1), dtype=np.float64).ravel()
    sums = sums[np.isfinite(sums) & (sums > 0)]
    if sums.size < 2:
        return False
    return bool(sums.max() - sums.min() <= rtol * sums.max())


def _on_linear_scale(X) -> bool:
    """True for a non-negative matrix that still needs ``log1p``: counts, CPM/TPM, proportions, or
    fractional (e.g. ambient-corrected) counts. ``_data_is_log_transformed`` alone reads proportions
    (max < 1, fractional) as logged, so rows that share a total count as linear first.
    """
    return _rows_share_a_total(X) or not _data_is_log_transformed(X)


def _prepare_expression_data(adata, what: str = "scRNA-seq data"):
    """Return ``(AnnData suitable for HVG selection and OT, what was done to it)``.

    Handles these cases:
      1. Raw counts (non-negative integers) -> normalise + log1p
      2. Already preprocessed with .raw available -> use .raw (normalise + log1p unless it is
         already log-transformed)
      3. Already preprocessed without .raw:
         - negative values (centred/scaled) -> X as-is;
         - log-transformed -> X as-is;
         - still on a linear scale (CPM/TPM, proportions, fractional counts) -> normalise + log1p.

    Non-integer is not the same as logged: a CPM matrix is non-integer and linear, and handing it
    to HVG selection and the transport unlogged ranks genes by linear variance and lets a few
    highly expressed genes dominate the atlas cost.
    """
    import scanpy as sc

    if _data_is_preprocessed(adata.X):
        if adata.raw is None:
            if _has_negative(adata.X):
                eprint(f"[novoSpaRc] {what}: scaled (negative values) and no .raw; using X as-is")
                return adata.copy(), "X used as-is (scaled: negative values; no .raw)"
            if not _on_linear_scale(adata.X):
                eprint(f"[novoSpaRc] {what}: already log-transformed and no .raw; using X as-is")
                return adata.copy(), "X used as-is (already log-transformed; no .raw)"
            eprint(f"[novoSpaRc] {what}: non-integer but linear-scale and no .raw; applying normalize_total + log1p")
            adata_proc = adata.copy()
            sc.pp.normalize_total(adata_proc, target_sum=1e4)
            sc.pp.log1p(adata_proc)
            return adata_proc, "X normalize_total(target_sum=1e4) + log1p (non-integer, not log-transformed; no .raw)"
        eprint(f"[novoSpaRc] {what}: X already preprocessed; using the .raw layer")
        adata_proc = adata.raw.to_adata()
        if not _on_linear_scale(adata_proc.X):
            eprint(f"[novoSpaRc] {what}: .raw already log-transformed; skipping normalise/log1p")
            return adata_proc, ".raw used as-is (already log-transformed)"
        sc.pp.normalize_total(adata_proc, target_sum=1e4)
        sc.pp.log1p(adata_proc)
        return adata_proc, ".raw normalize_total(target_sum=1e4) + log1p"

    eprint(f"[novoSpaRc] {what}: raw counts; applying normalize_total + log1p")
    adata_proc = adata.copy()
    sc.pp.normalize_total(adata_proc, target_sum=1e4)
    sc.pp.log1p(adata_proc)
    return adata_proc, "X normalize_total(target_sum=1e4) + log1p (raw counts)"


def _prep_kind(how: str) -> str:
    """Collapse a ``_prepare_expression_data`` description to what the values now are."""
    if "log1p" in how or "log-transformed" in how:
        return "log-normalised"
    return "used as supplied"


def _gene_variance(X) -> np.ndarray:
    """Compute per-gene variance, handling both sparse and dense matrices."""
    import scipy.sparse as sps

    if sps.issparse(X):
        # Var = E[X^2] - E[X]^2
        mean = np.array(X.mean(axis=0)).flatten()
        mean_sq = np.array(X.power(2).mean(axis=0)).flatten()
        return mean_sq - mean**2
    return np.var(np.asarray(X), axis=0).flatten()


# ----------------------------------------------------------------------------- alpha and mode


def resolve_alpha(alpha: float, has_reference: bool):
    """``(alpha_linear passed to novoSpaRc, mode, reason alpha was ignored or "")``.

    ``alpha`` is novoSpaRc's ``alpha_linear``: the weight of the linear atlas (marker-expression)
    cost against the structural Gromov-Wasserstein cost. Without an atlas that linear cost is
    upstream's placeholder ``np.ones``; weighting a constant only rescales the entropic
    regularisation (``epsilon / (1 - alpha)``), and ``alpha = 1`` returned the uniform coupling at
    status ok. So with no reference ``alpha`` is not passed on, and ``1`` is refused.
    """
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError(
            f"alpha={alpha!r} is outside [0, 1]. alpha weights novoSpaRc's reference-atlas (marker "
            "expression) cost against its structural cost: 0 is structure only, 1 is the atlas only."
        )
    if has_reference:
        if alpha > 0.0:
            return alpha, "atlas-guided", ""
        return 0.0, "reference geometry (structure only)", ""
    if alpha >= 1.0:
        raise ValueError(
            "alpha=1 asks for a mapping driven only by the reference atlas, and no st_h5ad was "
            "given, so there is no atlas: the result would be the uniform coupling. Pass st_h5ad, or "
            "alpha < 1 for a structure-only (de novo) reconstruction."
        )
    reason = ""
    if alpha > 0.0:
        reason = (
            f"alpha={alpha:g} weights the reference-atlas cost, and no st_h5ad was given, so there is "
            "no atlas; the de novo reconstruction ran structure only (alpha_linear=0)"
        )
    return 0.0, "de novo grid (structure only)", reason


# ----------------------------------------------------------------------------- memory


def ot_peak_bytes(n_cells: int, n_locations: int, n_genes: int = 0, n_markers: int = 0) -> int:
    """Peak bytes novoSpaRc needs for this many cells and locations. See ``_COST_PHASE_UNITS``."""
    n, m = int(n_cells), int(n_locations)
    units = (n * n, m * m, n * m)
    cost_phase = sum(a * b for a, b in zip(_COST_PHASE_UNITS, units))
    gw_phase = sum(a * b for a, b in zip(_GW_PHASE_UNITS, units))
    dense_inputs = 4 * n * int(n_genes) + 8 * (n + m) * int(n_markers)
    return 8 * max(cost_phase, gw_phase) + dense_inputs


def check_ot_memory(n_cells: int, n_locations: int, n_genes: int, n_markers: int, de_novo: bool, available=None):
    """Refuse, naming both numbers, before novoSpaRc allocates matrices that cannot fit.

    Returns ``(needed_bytes, available_bytes_or_None)``. Never subsamples.
    """
    need = ot_peak_bytes(n_cells, n_locations, n_genes, n_markers)
    if available is None:
        available = available_memory_bytes()
    if available is not None and need > available:
        knob = (
            "Lower num_locations (the de novo grid size) or run"
            if de_novo
            else "The locations are the reference's spots, so no parameter of this tool shrinks them; run"
        )
        raise MemoryError(
            f"novoSpaRc holds dense float64 matrices of {n_cells} cells x {n_cells} cells, "
            f"{n_locations} locations x {n_locations} locations and several of {n_cells} x {n_locations} "
            f"(the coupling itself is one): about {need / 1e9:.1f} GB at peak, but about "
            f"{available / 1e9:.1f} GB is available here (MemAvailable / room under the cgroup limit, page cache "
            "counted as reclaimable). These matrices are "
            f"intrinsic to the method and this worker never subsamples cells or locations. {knob} where "
            "that much memory is available."
        )
    return need, available


# ----------------------------------------------------------------------------- the transport


def _solver_module(upstream):
    """Upstream's ``novosparc.rc._GWadjusted``, or None when this novosparc does not have it.

    ``Tissue.reconstruct`` looks up ``novosparc.rc._GWadjusted.gromov_wasserstein_adjusted_norm``
    at call time, and that function calls the module's own ``sinkhorn`` (imported from POT), so both
    can be watched for the length of one call without touching upstream's code.
    """
    gw_mod = getattr(getattr(upstream, "rc", None), "_GWadjusted", None)
    gw_fn = getattr(gw_mod, "gromov_wasserstein_adjusted_norm", None)
    if not callable(gw_fn) or not callable(getattr(gw_mod, "sinkhorn", None)):
        return None
    try:
        params = inspect.signature(gw_fn).parameters
    except (TypeError, ValueError):
        return None
    if not {"log", "tol", "max_iter"} <= set(params):
        return None
    return gw_mod


@contextlib.contextmanager
def traced_solver(upstream, caught: list, trace: dict):
    """Record, into ``trace``, what upstream's solver did during one ``tissue.reconstruct``.

    ``Tissue.reconstruct`` calls ``gromov_wasserstein_adjusted_norm(..., log=False)`` and keeps only
    the coupling, so whether the Gromov-Wasserstein loop reached ``tol`` or ran out at ``max_iter``
    never reaches the caller -- the old payload wrote ``converged: true`` for every GW run. Here the
    call is handed a log dict (upstream's ``log=True`` would fail: it indexes the flag itself), so
    its every-10th-iteration change ``||T - T_prev||`` comes back, and each Sinkhorn solve is
    checked against the warnings POT raised during it (``caught`` is the enclosing
    ``warnings.catch_warnings(record=True)`` list, so nothing is swallowed).

    ``trace`` gains ``n_sinkhorn_solves``, ``last_sinkhorn_converged`` and, for the GW loop,
    ``gw_changes`` / ``gw_tol`` / ``gw_max_iter``. When the module cannot be found it stays empty.
    """
    gw_mod = _solver_module(upstream)
    if gw_mod is None:
        yield
        return
    real_gw = gw_mod.gromov_wasserstein_adjusted_norm
    real_sinkhorn = gw_mod.sinkhorn
    signature = inspect.signature(real_gw)

    def sinkhorn(*args, **kwargs):
        start = len(caught)
        result = real_sinkhorn(*args, **kwargs)
        said = [str(w.message) for w in caught[start:]]
        trace["n_sinkhorn_solves"] = int(trace.get("n_sinkhorn_solves", 0)) + 1
        trace["last_sinkhorn_converged"] = not any(_NOT_CONVERGED_TEXT in m or _NUMERICAL_ERROR_TEXT in m for m in said)
        return result

    def gromov_wasserstein_adjusted_norm(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments.get("log"):
            return real_gw(*args, **kwargs)  # a caller that asked for its own log gets it untouched
        record = {"err": []}
        bound.arguments["log"] = record
        result = real_gw(*bound.args, **bound.kwargs)
        coupling = result[0] if isinstance(result, tuple) else result
        trace["gw_changes"] = [float(e) for e in record.get("err", [])]
        trace["gw_tol"] = float(bound.arguments["tol"])
        trace["gw_max_iter"] = int(bound.arguments["max_iter"])
        return coupling

    gw_mod.gromov_wasserstein_adjusted_norm = gromov_wasserstein_adjusted_norm
    gw_mod.sinkhorn = sinkhorn
    try:
        yield
    finally:
        gw_mod.gromov_wasserstein_adjusted_norm = real_gw
        gw_mod.sinkhorn = real_sinkhorn


def judge_convergence(trace: dict, single_solve: bool, n_not_converged: int) -> dict:
    """What the solver's record says about the published coupling.

    Returns ``converged`` (True / False / None when it cannot be known) plus the facts behind it.
    ``alpha = 1`` is one Sinkhorn solve on the atlas cost: that solve is the verdict. Otherwise the
    Gromov-Wasserstein loop must have reached ``tol`` before ``max_iter`` and the Sinkhorn solve that
    produced the last iterate must have converged. Without a trace a GW run is unknown (None), never
    assumed converged; a single solve is still judged by the warning it raised.
    """
    solves = int(trace.get("n_sinkhorn_solves", 0))
    facts: dict = {}
    if single_solve:
        if solves:
            converged = bool(trace.get("last_sinkhorn_converged"))
        else:
            converged = not n_not_converged
        facts["converged"] = converged
        return facts
    changes = trace.get("gw_changes")
    if not solves or changes is None:
        facts["converged"] = None
        return facts
    tol = float(trace.get("gw_tol", 0.0))
    loop_ok = bool(changes) and changes[-1] <= tol
    last_ok = bool(trace.get("last_sinkhorn_converged"))
    facts.update(
        {
            "converged": bool(loop_ok and last_ok),
            "gw_loop_converged": bool(loop_ok),
            "gw_iterations": solves,  # upstream makes one Sinkhorn solve per GW iteration
            "gw_max_iter": int(trace.get("gw_max_iter", 0)),
            "gw_tol": tol,
            "gw_last_change": float(changes[-1]) if changes else None,
            "last_sinkhorn_converged": last_ok,
        }
    )
    return facts


def reconstruct_searching_epsilon(tissue, alpha_linear: float, upstream=None) -> dict:
    """Run ``tissue.reconstruct`` up the epsilon ladder; return what converged, or raise.

    A coupling is accepted only when POT raised no "numerical errors" warning and the coupling is
    finite with positive mass. The old loop accepted whatever the last epsilon left behind -- a
    Sinkhorn that had bailed out, or a NaN coupling (``NaN == 0`` is False, so the all-zero check
    let it through) -- and published no epsilon at all.

    ``upstream`` is the imported ``novosparc`` package; its solver is traced (``traced_solver``) and
    the record of the accepted run is returned under ``trace``.
    """
    tried = []
    last_problem = ""
    for epsilon in EPSILON_LADDER:
        tried.append(float(epsilon))
        eprint(f"[novoSpaRc] Trying epsilon={epsilon:.0e} ...")
        trace: dict = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with traced_solver(upstream, caught, trace):
                tissue.reconstruct(alpha_linear=alpha_linear, epsilon=epsilon, search_epsilon=False)
        messages = [str(w.message) for w in caught]
        n_numerical = sum(_NUMERICAL_ERROR_TEXT in m for m in messages)
        gw = tissue.gw
        problems = []
        if n_numerical:
            problems.append(f"Sinkhorn reported numerical errors {n_numerical} time(s)")
        if gw is None:
            problems.append("no coupling was produced")
        else:
            gw = np.asarray(gw)
            if not np.all(np.isfinite(gw)):
                problems.append("the coupling holds non-finite values")
            elif not gw.sum() > 0:
                problems.append("the coupling is all zeros")
        if not problems:
            eprint(f"[novoSpaRc] Converged with epsilon={epsilon:.0e}")
            return {
                "epsilon_used": float(epsilon),
                "epsilon_tried": tried,
                "n_sinkhorn_not_converged": int(sum(_NOT_CONVERGED_TEXT in m for m in messages)),
                "trace": trace,
            }
        last_problem = "; ".join(problems)
        eprint(f"[novoSpaRc] epsilon={epsilon:.0e} rejected: {last_problem}")
    raise RuntimeError(
        "novoSpaRc's optimal transport produced no usable coupling at any epsilon tried "
        f"({', '.join(f'{e:.0e}' for e in tried)}); at the largest: {last_problem}. The result is "
        "not published because a coupling Sinkhorn gave up on is not a mapping. Check the inputs for "
        "non-finite values, or change alpha, num_neighbors_s or num_neighbors_t."
    )


def mean_row_entropy(coupling: np.ndarray, block: int = 1024) -> float:
    """Mean over cells of the entropy of each cell's distribution over locations (nats).

    Each row is normalised to sum to 1 first, so the uniform coupling scores ``ln(n_locations)``
    and a one-location assignment scores 0. The old summary summed ``-c log c`` over the raw rows,
    which sum to 1/n_cells, so the number depended on the cell count and read near 0 for any input.
    Computed in row blocks so no extra full-size copy of the coupling is made.
    """
    total = 0.0
    n = coupling.shape[0]
    for start in range(0, n, block):
        rows = np.asarray(coupling[start : start + block], dtype=np.float64)
        sums = rows.sum(axis=1, keepdims=True)
        sums[sums <= 0] = 1.0
        p = rows / sums
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
        total += float(h.sum())
    return total / max(n, 1)


def celltype_location_table(coupling: np.ndarray, labels, location_ids):
    """``(locations x cell-types DataFrame, n_unlabelled)`` -- each location's share of mass per type.

    Entry (j, t) is the fraction of location j's transported mass that comes from cells labelled t.
    Cells with no label (NaN/empty) are not a class: their mass is in no column, so a row sums to the
    labelled share of that location (1.0 when every cell is labelled), and their count is returned.
    """
    import scipy.sparse as sps

    keep, n_unlabelled = drop_unlabeled(labels, True, what="cells")
    values = np.asarray([str(v) for v in np.asarray(labels, dtype=object)], dtype=object)
    types = sorted(set(values[keep].tolist()))
    if not types:
        return None, n_unlabelled
    code = {t: i for i, t in enumerate(types)}
    rows = np.flatnonzero(keep)
    onehot = sps.csr_matrix(
        (np.ones(rows.size), ([code[values[i]] for i in rows], rows)),
        shape=(len(types), coupling.shape[0]),
    )
    mass = np.asarray(onehot @ coupling).T  # locations x types
    location_mass = np.asarray(coupling.sum(axis=0), dtype=np.float64).ravel()
    location_mass[location_mass <= 0] = np.nan
    table = mass / location_mass[:, None]
    return pd.DataFrame(table, index=pd.Index(location_ids), columns=types), n_unlabelled


def _sci(value) -> str:
    """``1.23e-08``, or ``n/a`` when the solver recorded no change at all."""
    return "n/a" if value is None else f"{float(value):.2e}"


def _convergence_phrase(convergence: dict, single_solve: bool) -> str:
    """The analysis clause that says whether the published coupling converged, and by what test."""
    converged = convergence.get("converged")
    if single_solve:
        return "" if converged else " (the single Sinkhorn solve stopped at POT's iteration limit, unconverged)"
    if converged is None:
        return " (whether the Gromov-Wasserstein loop converged could not be read from the solver)"
    if converged:
        return (
            f" (the Gromov-Wasserstein loop converged: last change {_sci(convergence['gw_last_change'])} <= tol "
            f"{convergence['gw_tol']:g} after {convergence['gw_iterations']} iterations)"
        )
    if not convergence.get("gw_loop_converged"):
        return (
            f" (the Gromov-Wasserstein loop stopped at its iteration limit of {convergence['gw_max_iter']} with "
            f"last change {_sci(convergence['gw_last_change'])} above tol {convergence['gw_tol']:g}, unconverged)"
        )
    return " (the last inner Sinkhorn solve stopped at POT's iteration limit, unconverged)"


# ----------------------------------------------------------------------------- writing


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp = Path(str(path) + ".partial")
    frame.to_csv(tmp)
    os.replace(tmp, path)


def _write_h5ad_atomic(adata, path: Path) -> None:
    tmp = Path(str(path) + ".partial")
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- the run


def run_novosparc(
    sc_h5ad: str,
    output_dir: str,
    st_h5ad: str = "",
    spatial_key: str = "spatial",
    annotation_key: str = "cell_type",
    num_locations: int = 1000,
    alpha: float = 0.5,
    n_hvg: int = 2000,
    num_neighbors_s: int = 5,
    num_neighbors_t: int = 5,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Run novoSpaRc spatial reconstruction."""

    import anndata as ad
    import novosparc
    import scanpy as sc

    has_reference = bool(st_h5ad and st_h5ad.strip())
    alpha_linear, mode, alpha_ignored_why = resolve_alpha(alpha, has_reference)
    atlas_guided = mode == "atlas-guided"
    if int(n_hvg) < 1:
        raise ValueError(f"n_hvg={n_hvg} must be at least 1 (a value above the gene count uses every gene).")
    for name, value in (("num_neighbors_s", num_neighbors_s), ("num_neighbors_t", num_neighbors_t)):
        if int(value) < 1:
            raise ValueError(f"{name}={value} must be at least 1.")
    if not has_reference and int(num_locations) < 1:
        raise ValueError(f"num_locations={num_locations} must be at least 1 for the de novo grid.")

    _ensure_dir(output_dir)
    np.random.seed(random_seed)
    outdir = Path(output_dir)

    # ---- Load scRNA-seq data ----
    eprint(f"[novoSpaRc] Loading scRNA-seq data: {sc_h5ad}")
    adata_sc = sc.read_h5ad(sc_h5ad)
    eprint(f"[novoSpaRc] Loaded scRNA: {adata_sc.n_obs} cells x {adata_sc.n_vars} genes")

    # ---- Preprocessing ----
    adata_proc, sc_how = _prepare_expression_data(adata_sc, "scRNA-seq data")
    # Gene axis only, and on the object whose genes key the outputs: the reconstructed h5ad and the
    # spatial-expression CSV are keyed by gene symbol, and when .raw was used its var_names are not
    # the ones on X. Cells are untouched.
    renamed = make_names_unique_and_report(adata_proc, axes=("var",))
    eprint(f"[novoSpaRc] Working data: {adata_proc.n_obs} cells x {adata_proc.n_vars} genes")

    # ---- Reference: coordinates always; expression only when it is the atlas ----
    adata_st = None
    st_proc = None
    st_how = ""
    renamed_st = None
    harmonization: dict = {}
    n_spots_supplied = 0
    n_spots_off_tissue = 0
    if has_reference:
        eprint(f"[novoSpaRc] Loading reference spatial data: {st_h5ad}")
        adata_st = sc.read_h5ad(st_h5ad)
        eprint(f"[novoSpaRc] Reference spatial: {adata_st.n_obs} spots x {adata_st.n_vars} genes")
        # Background spots (obs['in_tissue'] == 0) are glass, not tissue: as target locations they
        # would each receive 1/n_locations of every cell's mass under the uniform location marginal.
        adata_st, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata_st, "spots")
        if n_spots_off_tissue:
            eprint(
                f"[novoSpaRc] Left out {n_spots_off_tissue} of {n_spots_supplied} reference spots with "
                "obs['in_tissue'] == 0 (background)"
            )
        if spatial_key not in adata_st.obsm:
            raise KeyError(
                f"Spatial key '{spatial_key}' not found in reference h5ad obsm. Available: {list(adata_st.obsm.keys())}"
            )
        locations, _ = spatial_coords(adata_st, spatial_key, want=2, tool="novoSpaRc")
        location_ids = [str(x) for x in adata_st.obs_names]
        if atlas_guided:
            st_proc, st_how = _prepare_expression_data(adata_st, "reference spatial data")
            renamed_st = make_names_unique_and_report(st_proc, axes=("var",))
            harmonize_gene_ids(adata_proc, st_proc, harmonization)
        eprint(f"[novoSpaRc] Using {locations.shape[0]} reference locations from obsm['{spatial_key}']")
    else:
        eprint(f"[novoSpaRc] No reference spatial data. Constructing de novo grid for {num_locations} locations.")
        locations = np.asarray(novosparc.geometry.construct_target_grid(int(num_locations)))
        # construct_target_grid rounds to a full ratio-1.2 grid: 1000 requested -> 1015 built.
        location_ids = [str(i) for i in range(locations.shape[0])]
        eprint(f"[novoSpaRc] Constructed target grid: {locations.shape}")
    n_locations = int(locations.shape[0])
    n_bad = int((~np.isfinite(np.asarray(locations, dtype=np.float64))).any(axis=1).sum())
    if n_bad:
        raise ValueError(
            f"{n_bad} of {n_locations} locations in obsm['{spatial_key}'] have a missing or non-finite "
            "coordinate; novoSpaRc's location graph cannot place them. Fix or remove those spots first."
        )

    # ---- Genes: top-variance HVGs of the prepared expression ----
    # Direct variance ranking rather than scanpy's highly_variable_genes(), which can fail on
    # already-normalised data (NaN bin edges when data has been log1p'd or scaled).
    n_select = min(int(n_hvg), adata_proc.n_vars)
    gene_var = _gene_variance(adata_proc.X)
    top_idx = np.argsort(gene_var)[-n_select:]
    hvg_mask = np.zeros(adata_proc.n_vars, dtype=bool)
    hvg_mask[top_idx] = True
    gene_names = [str(g) for g in adata_proc.var_names[hvg_mask]]
    eprint(f"[novoSpaRc] Selected {len(gene_names)} highly variable genes")

    adata_hvg = adata_proc[:, hvg_mask].copy()
    # Dense on purpose: upstream Tissue keeps dataset.X as its dge and computes
    # ``np.dot(self.dge.T, gw)``. On a scipy sparse matrix np.dot does not multiply matrices -- it
    # returns an n_cells x n_locations object array holding one scaled copy of the whole sparse
    # matrix per entry, which exhausts memory on any real input. The atlas cost also indexes it.
    expression = _safe_dense(adata_hvg.X)
    adata_hvg.X = expression
    cell_ids = [str(x) for x in adata_hvg.obs_names]
    num_cells = expression.shape[0]
    num_genes_used = expression.shape[1]

    for name, value, count, what in (
        ("num_neighbors_s", num_neighbors_s, num_cells, "cells"),
        ("num_neighbors_t", num_neighbors_t, n_locations, "locations"),
    ):
        if int(value) > count:
            raise ValueError(f"{name}={value} exceeds the {count} {what}; the kNN graph needs {name} <= {count}.")

    # ---- The atlas: the reference's expression of the HVGs it shares ----
    markers_to_use = None
    atlas_matrix = None
    marker_genes: list = []
    if atlas_guided:
        st_pos = {str(g): i for i, g in enumerate(st_proc.var_names)}
        marker_idx = [j for j, g in enumerate(gene_names) if g in st_pos]
        if not marker_idx:
            raise ValueError(
                f"alpha={alpha:g} weights the reference-atlas cost, but none of the {len(gene_names)} scRNA-seq "
                f"HVGs is among the reference's {st_proc.n_vars} genes, so there is no atlas to weight (the two "
                f"objects share {harmonization.get('n_shared_genes', 0)} gene identifiers in all, after trying "
                "their gene-symbol columns). Pass alpha=0 for a structure-only mapping onto the reference "
                "coordinates, raise n_hvg, or supply a reference whose var_names use the scRNA-seq gene "
                "identifiers."
            )
        marker_genes = [gene_names[j] for j in marker_idx]
        markers_to_use = np.asarray(marker_idx, dtype=int)
        atlas_matrix = _safe_dense(st_proc.X[:, [st_pos[g] for g in marker_genes]])
        if not np.amax(atlas_matrix) > 0 or not np.amax(expression[:, markers_to_use]) > 0:
            raise ValueError(
                f"The {len(marker_genes)} marker genes shared with the reference carry no positive expression "
                "in the reference or in the scRNA-seq data, so the atlas cost is undefined (novoSpaRc scales "
                "each side by its maximum). Pass alpha=0 for a structure-only mapping."
            )
        eprint(f"[novoSpaRc] Atlas-guided: {len(marker_genes)} marker genes shared with the reference")

    check_ot_memory(num_cells, n_locations, num_genes_used, len(marker_genes), de_novo=not has_reference)

    # ---- Setup novoSpaRc tissue ----
    eprint("[novoSpaRc] Setting up novoSpaRc tissue object ...")
    tissue = novosparc.cm.Tissue(
        dataset=adata_hvg,
        locations=locations,
        output_folder=str(outdir),
    )

    eprint("[novoSpaRc] Computing cost matrices ...")
    tissue.setup_reconstruction(
        markers_to_use=markers_to_use,
        atlas_matrix=atlas_matrix,
        num_neighbors_s=num_neighbors_s,
        num_neighbors_t=num_neighbors_t,
    )

    # ---- Run reconstruction ----
    eprint(f"[novoSpaRc] Running optimal transport reconstruction ({mode}, alpha_linear={alpha_linear:g}) ...")
    search = reconstruct_searching_epsilon(tissue, alpha_linear, upstream=novosparc)
    eprint("[novoSpaRc] Reconstruction complete.")

    coupling = np.asarray(tissue.gw, dtype=np.float64)
    eprint(f"[novoSpaRc] Coupling matrix shape: {coupling.shape}")
    # Upstream's alpha_linear == 1 branch is one Sinkhorn solve on the atlas cost, with no GW loop:
    # when that solve hit POT's iteration limit, the published coupling is that unconverged solve.
    # Otherwise the GW loop's own record decides (judge_convergence); it used to be True regardless.
    single_solve = atlas_guided and alpha_linear >= 1.0
    convergence = judge_convergence(search["trace"], single_solve, search["n_sinkhorn_not_converged"])
    converged = convergence["converged"]

    # Reconstruct spatial expression: sdge = coupling^T @ expression (upstream's dge.T @ gw,
    # transposed). Each location's value is its mass-weighted expression sum; the coupling's columns
    # sum to 1/n_locations, so this is the location's mean expression divided by n_locations.
    sdge = coupling.T @ expression
    eprint(f"[novoSpaRc] Reconstructed spatial expression: {sdge.shape}")

    # ---- Save outputs (every table carries the identifiers it is about) ----
    coupling_csv = outdir / "novosparc_coupling.csv"
    _write_csv_atomic(pd.DataFrame(coupling, index=pd.Index(cell_ids), columns=location_ids), coupling_csv)
    eprint(f"[novoSpaRc] Saved coupling matrix to {coupling_csv}")

    adata_reconstructed = ad.AnnData(
        X=sdge,
        obs=pd.DataFrame(index=pd.Index(location_ids)),
        var=pd.DataFrame(index=pd.Index(gene_names)),
    )
    adata_reconstructed.obsm["spatial"] = np.asarray(locations)
    recon_h5ad = outdir / "novosparc_reconstructed.h5ad"
    _write_h5ad_atomic(adata_reconstructed, recon_h5ad)
    eprint(f"[novoSpaRc] Saved reconstructed h5ad to {recon_h5ad}")

    sdge_csv = outdir / "novosparc_spatial_expression.csv"
    _write_csv_atomic(pd.DataFrame(sdge, index=pd.Index(location_ids), columns=gene_names), sdge_csv)
    eprint(f"[novoSpaRc] Saved spatial expression CSV to {sdge_csv}")

    # ---- Cell types per location, when the reference names them ----
    celltype_csv = None
    n_unlabelled = 0
    n_cell_types = 0
    annotation_ignored_why = ""
    if annotation_key:
        if annotation_key not in adata_sc.obs.columns:
            shown = ", ".join(str(c) for c in list(adata_sc.obs.columns)[:20])
            annotation_ignored_why = (
                f"the scRNA-seq obs has no column {annotation_key!r} (columns: {shown}); no cell-type table was written"
            )
        else:
            table, n_unlabelled = celltype_location_table(
                coupling, adata_sc.obs[annotation_key].to_numpy(), location_ids
            )
            if table is None:
                annotation_ignored_why = (
                    f"no cell in obs[{annotation_key!r}] carries a label; no cell-type table was written"
                )
            else:
                celltype_csv = outdir / CELLTYPE_TABLE
                _write_csv_atomic(table, celltype_csv)
                n_cell_types = int(table.shape[1])
                eprint(f"[novoSpaRc] Saved cell-type x location table to {celltype_csv}")

    # ---- Build output ----
    out = WorkerOutput("novosparc", task="spatial_reconstruction")
    out.set_data(
        n_cells=int(num_cells),
        n_genes=int(adata_sc.n_vars),
        n_genes_used=int(num_genes_used),
        n_locations=int(n_locations),
    )
    if has_reference:
        out.set_data(n_reference_spots=int(n_spots_supplied))
    files = {
        "reconstructed_h5ad": str(recon_h5ad),
        "coupling_csv": str(coupling_csv),
        "spatial_expression_csv": str(sdge_csv),
    }
    if celltype_csv is not None:
        files["celltype_location_csv"] = str(celltype_csv)
    out.add_output_files(files)
    out.add_params(
        {
            "alpha": alpha,
            "n_hvg": n_hvg,
            "num_neighbors_s": num_neighbors_s,
            "num_neighbors_t": num_neighbors_t,
            "has_reference": has_reference,
            "random_seed": random_seed,
        }
    )
    out.add_params(
        {
            "mode": mode,
            "alpha_effective": float(alpha_linear),
            "n_markers": len(marker_genes),
            "epsilon_used": search["epsilon_used"],
            "epsilon_tried": search["epsilon_tried"],
            "converged": converged,
            "n_sinkhorn_not_converged": search["n_sinkhorn_not_converged"],
            "hvg_method": "top-variance genes of the prepared scRNA-seq expression",
            "sc_preprocessing": sc_how,
            "location_id_source": "reference obs_names" if has_reference else "de novo grid positions 0..n-1",
        }
    )
    out.add_params({k: v for k, v in convergence.items() if k != "converged"})
    if has_reference:
        out.add_params({"spatial_key": spatial_key})
        record_in_tissue(out, n_spots_supplied, n_spots_off_tissue, "spots")
    else:
        out.add_params({"num_locations": num_locations})
    if atlas_guided:
        out.add_params({"st_preprocessing": st_how})
        out.add_params(identifier_rename_params(renamed_st, suffix="st"))
        out.add_params(gene_id_harmonization_params(harmonization))
    if celltype_csv is not None:
        out.add_params({"annotation_key": annotation_key, "n_cells_unlabelled": int(n_unlabelled)})
    out.add_params(identifier_rename_params(renamed))

    if not atlas_guided:
        what_ran = "Gromov-Wasserstein, structural costs only"
    elif single_solve:
        # Upstream's alpha_linear == 1 branch is a single Sinkhorn on the atlas cost: no GW term.
        what_ran = f"entropic OT on the linear atlas cost alone over {len(marker_genes)} marker genes (no GW term)"
    else:
        what_ran = (
            f"Gromov-Wasserstein + linear atlas cost over {len(marker_genes)} marker genes shared with the reference"
        )
    method = f"novoSpaRc (upstream novosparc.cm.Tissue.reconstruct, alpha_linear={alpha_linear:g}): {what_ran}"
    record_method(out, method)
    if alpha_ignored_why:
        record_ignored(out, "alpha", alpha_ignored_why)
    if has_reference:
        record_ignored(out, "num_locations", f"the reference's {n_locations} in-tissue spots are the target locations")
    else:
        record_ignored(out, "spatial_key", "no st_h5ad was given, so no coordinates were read")
    if annotation_ignored_why:
        record_ignored(out, "annotation_key", annotation_ignored_why)
    record_ignored(
        out,
        "seed",
        "nothing random runs: the transport starts from the product coupling (random_ini=False), the de novo "
        "grid is not random, and the kNN graphs and shortest paths are exact",
    )
    if n_unlabelled:
        out.add_warning(
            f"{n_unlabelled} of {num_cells} cells have no label in obs[{annotation_key!r}]; their mass is in no "
            f"column of {CELLTYPE_TABLE}, so those rows sum to the labelled share of each location."
        )
    if search["n_sinkhorn_not_converged"] and single_solve:
        out.add_warning(
            f"The single atlas-only Sinkhorn solve (alpha=1: no Gromov-Wasserstein term) hit POT's iteration "
            f"limit at epsilon={search['epsilon_used']:.0e} (no numerical errors); the coupling is that "
            "unconverged solve, so params.converged is false."
        )
    elif search["n_sinkhorn_not_converged"]:
        out.add_warning(
            f"{search['n_sinkhorn_not_converged']} inner Sinkhorn solve(s) hit POT's iteration limit at "
            f"epsilon={search['epsilon_used']:.0e} (no numerical errors); the coupling is the Gromov-Wasserstein "
            "iterate those solves produced."
        )
    if not single_solve:
        if converged is None:
            out.add_warning(
                "Whether the Gromov-Wasserstein loop converged could not be read: this novosparc has no "
                "rc._GWadjusted solver to trace, so params.converged is null (unknown), not true."
            )
        elif not convergence["gw_loop_converged"]:
            out.add_warning(
                f"The Gromov-Wasserstein loop ran its {convergence['gw_iterations']} iterations (max_iter="
                f"{convergence['gw_max_iter']}) and its last change ||T - T_prev|| = {_sci(convergence['gw_last_change'])} "
                f"is above tol={convergence['gw_tol']:g}; the coupling is that unconverged iterate, so "
                "params.converged is false."
            )
        elif not convergence["last_sinkhorn_converged"]:
            out.add_warning(
                f"The Gromov-Wasserstein loop reached tol={convergence['gw_tol']:g} after "
                f"{convergence['gw_iterations']} iterations, but the Sinkhorn solve that produced the last iterate "
                "hit POT's iteration limit, so params.converged is false."
            )
    if atlas_guided and len(marker_genes) < 10:
        out.add_warning(
            f"Only {len(marker_genes)} marker gene(s) are shared with the reference; the atlas cost rests on them."
        )
    if atlas_guided and _prep_kind(sc_how) != _prep_kind(st_how):
        out.add_warning(
            f"The scRNA-seq data ({sc_how}) and the reference ({st_how}) were prepared differently; the atlas "
            "cost compares them after scaling each by its maximum."
        )

    uniform = math.log(n_locations) if n_locations > 0 else 0.0
    entropy = mean_row_entropy(coupling)
    out.set_summary(
        coupling_shape=list(coupling.shape),
        sdge_shape=list(sdge.shape),
        mean_entropy=entropy,
        uniform_entropy=float(uniform),
    )
    if single_solve:
        how = (
            f"Atlas only (alpha=1): the reference's expression of {len(marker_genes)} shared marker genes is the "
            "whole cost, in one Sinkhorn solve with no structural (Gromov-Wasserstein) term."
        )
    elif atlas_guided:
        how = (
            f"Atlas-guided: the reference's expression of {len(marker_genes)} shared marker genes weighted "
            f"alpha={alpha_linear:g} against the structural cost."
        )
    elif has_reference:
        how = "Structure only (alpha=0) onto the reference coordinates; the reference expression was not used."
    else:
        how = (
            f"Structure only onto a de novo grid of {n_locations} locations ({num_locations} requested; "
            "novoSpaRc builds a full grid)."
        )
        if alpha_ignored_why:
            how += f" alpha={alpha:g} was ignored: there is no atlas without a reference."
    if n_spots_off_tissue:
        how += (
            f" {n_spots_off_tissue} of the reference's {n_spots_supplied} spots have obs['in_tissue'] == 0 "
            f"(background) and were left out, so the {n_locations} locations are its in-tissue spots."
        )
    table_note = (
        f" Cell-type shares per location from obs[{annotation_key!r}] ({n_cell_types} types) are in {CELLTYPE_TABLE}."
        if celltype_csv is not None
        else ""
    )
    out.set_analysis(
        f"novoSpaRc reconstructed spatial expression for {num_genes_used} genes (top-variance HVGs) across "
        f"{n_locations} locations from {num_cells} cells. {how} Entropic regularisation epsilon="
        f"{search['epsilon_used']:.0e}"
        + _convergence_phrase(convergence, single_solve)
        + f". Mean per-cell mapping entropy {entropy:.3f} nats "
        f"(uniform = {uniform:.3f}). Expression values are the coupling-weighted sums (a location's mean "
        f"expression divided by the location count)."
        + table_note
        + identifier_rename_note(renamed, subject="scRNA-seq data")
        + (
            f" NOTE: {int(renamed_st.get('n_genes_renamed') or 0)} duplicate gene identifier(s) in the spatial "
            "reference were renamed to make them unique before its genes were matched to the scRNA-seq genes."
            if renamed_st and renamed_st.get("n_genes_renamed")
            else ""
        )
        + (gene_id_harmonization_note(harmonization) if atlas_guided else "")
    )

    return out.to_dict()


def main():
    import contextlib

    ap = argparse.ArgumentParser(description="novoSpaRc spatial reconstruction worker")
    ap.add_argument("--sc-h5ad", required=True, help="Path to scRNA-seq AnnData (.h5ad)")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--st-h5ad", default="", help="Path to reference spatial AnnData (.h5ad), optional")
    ap.add_argument("--spatial-key", default="spatial", help="obsm key for spatial coordinates in reference")
    ap.add_argument(
        "--annotation-key",
        default="cell_type",
        help="obs column of the scRNA-seq data whose labels are aggregated per location (empty: no table)",
    )
    ap.add_argument("--num-locations", type=int, default=1000, help="Number of target locations (if no reference)")
    ap.add_argument(
        "--alpha", type=float, default=0.5, help="Weight of the reference-atlas cost vs the structural cost (0-1)"
    )
    ap.add_argument("--n-hvg", type=int, default=2000, help="Number of highly variable genes to use")
    ap.add_argument("--num-neighbors-s", type=int, default=5, help="k-neighbors in source graph")
    ap.add_argument("--num-neighbors-t", type=int, default=5, help="k-neighbors in target graph")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    # Redirect all stdout to stderr during analysis to prevent library
    # prints (novosparc, scanpy) from polluting the JSON output.
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    result = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = run_novosparc(
                sc_h5ad=args.sc_h5ad,
                output_dir=args.output_dir,
                st_h5ad=args.st_h5ad,
                spatial_key=args.spatial_key,
                annotation_key=args.annotation_key,
                num_locations=args.num_locations,
                alpha=args.alpha,
                n_hvg=args.n_hvg,
                num_neighbors_s=args.num_neighbors_s,
                num_neighbors_t=args.num_neighbors_t,
                random_seed=args.seed,
            )
    except Exception as e:
        eprint(f"[novoSpaRc] ERROR: {e}")
        traceback.print_exc(file=sys.stderr)
        # Restore stdout before emitting error JSON
        sys.stdout = _real_stdout
        WorkerOutput.emit_error("novosparc", str(e), task="spatial_reconstruction", exc=e)
        sys.exit(1)
    finally:
        sys.stdout = _real_stdout

    if result is None:
        WorkerOutput.emit_error("novosparc", "Worker produced no result", task="spatial_reconstruction")
        sys.exit(1)

    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
