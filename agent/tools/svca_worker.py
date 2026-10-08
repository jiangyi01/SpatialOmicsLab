#!/usr/bin/env python
"""
SVCA worker: spatial variance component analysis.

Implements the SVCA per-gene spatial-vs-noise variance decomposition via the
FaST-LMM (Lippert et al. 2011) closed-form REML estimator. Eigendecomposition
of the RBF spatial covariance K = U diag(s) U^T is performed ONCE, then each
gene's spatial fraction reduces to a 1-D scalar optimisation over
delta = sigma_e / sigma_s. The result, sigma_s mean(s) / (sigma_s mean(s) + sigma_e),
goes into the 'intrinsic' column the downstream pipeline ranks against.

This replaces the prior 3-component (intrinsic/environmental/noise) L-BFGS-B
fit, which routinely converged to the [0.333, 0.333, 0.333] starting point on
sparse Visium genes (the optimiser had insufficient gradient signal in
log-variance space) and made the downstream rank-by-intrinsic ordering
arbitrary -> F1 ~ 0.23 on visium_svg.

The manual benchmark reference at
/workspace/hands_by_myself/runners/svca_variance_decomposition.py uses the
same FaST-LMM algorithm and scores F1 ~ 0.31 on visium_svg.

- Runs in the spatialomicsgym_e1 environment (or its pre-rebrand alias from
  ``constants.LEGACY_ENV_ALIASES`` on an upgraded box), not the legacy ``svca`` env: the worker needs only numpy/scipy/pandas/h5py, and the legacy
  env's BLAS returned NaN eigenvector columns on these kernels.
- All logs/progress go to stderr.
- stdout is reserved for a single final JSON line.

CSV schema is preserved for compatibility with the agent benchmark
output_standardizer (which reads `gene` + `intrinsic`):
    gene, intrinsic, environmental, noise, total_variance, converged, top_decile_by_intrinsic
where `intrinsic` is the FaST-LMM spatial fraction, `environmental` is written EMPTY (NaN): a
single-kernel model has no third component, so the environmental / cell-cell-interaction term of
the published SVCA model is not estimated. `noise` is 1 - intrinsic. Downstream rank-by-intrinsic
is the same as rank-by-spatial-frac.

How much of the slide is analysed
---------------------------------
Every in-tissue spot, by default. Spots with ``obs['in_tissue'] == 0`` are background glass and are
left out first, by the shared ``worker_utils.keep_in_tissue`` rule, and reported
(``params.in_tissue_filter``, a warning and the ``analysis`` sentence): CELLxGENE Visium exports carry
every array spot, and on the library's four such samples 56-70% of them are background that still
holds counts, so the kernel and every gene's spatial fraction measured the tissue edge. A file with
no ``in_tissue`` column is analysed whole. ``--max-spots N`` is an explicit opt-in that analyses a
random N-spot subset of the in-tissue spots (drawn with ``--seed``); the payload then reports the
subsample in ``params`` and in the ``analysis`` sentence. There used to be a default cap of 2500,
which silently decomposed a random fraction of every larger slide.

The estimator is exact and its cost is intrinsic: it eigendecomposes the dense n x n RBF kernel,
which needs about ``N_BY_N_COPIES`` float64 n x n matrices at once (24 n^2 bytes) and time that
grows as n^3. The worker estimates that memory from the spot count BEFORE it reads the expression
matrix and refuses, naming both numbers, when it cannot fit. It does not switch to a low-rank
approximation (that would be a different method) and it never subsamples on its own.

The expression matrix stays sparse through loading and normalisation; only blocks of genes over
the analysed spots are densified, for U^T X.

Which matrix is decomposed
--------------------------
SVCA library-size normalises and log1p-transforms the matrix, so it has to hold counts. ``X`` is read
when the file has one, ``raw/X`` when it has no ``X``, and ``raw/X`` whenever ``--use-raw-counts`` is
passed (the shared ``worker_utils.choose_counts_matrix`` rule, applied here by hand because the worker
reads the h5ad with h5py). Every stored value of the analysed spots is classified with
``worker_utils.expression_matrix_kind``: negative or non-finite values (scaled / z-scored data) are
refused, naming ``use_raw_counts`` when ``raw/X`` holds counts -- normalising them turned 7.27M stored
values of a CELLxGENE Skin slide into NaN and left 2 of its 35,477 genes to decompose -- and a
non-negative non-integer matrix (log-normalised data) runs as before with a warning that it was
normalised a second time. ``params.expression_source`` (``X`` / ``raw.X``) and ``params.x_matrix_kind``
say which matrix ran and what ``X`` held.

``total_variance`` is the variance (ddof=1) of the gene's normalised expression over the analysed
spots. It used to be the variance of the eigen-rotated vector U^T y, which is about mean^2 + var:
25.4 for a gene of variance 0.87 and mean 5.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Make worker_utils importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h5py
import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    expression_matrix_kind,
    keep_in_tissue,
    read_obsm_matrix,
    record_ignored,
    record_in_tissue,
    record_method,
    spatial_coords,
)

METHOD = "FaST-LMM REML (Lippert 2011)"

#: Dense n x n float64 matrices alive at once at the eigendecomposition, the peak of the run: the
#: kernel K, LAPACK's working copy of it (scipy passes a C-ordered array, which f2py copies), and
#: the eigenvector matrix U. Measured: eigh on a 4000 x 4000 kernel raised the peak RSS by two
#: matrices over the kernel it was given.
N_BY_N_COPIES = 3

#: Bytes per analysed non-zero while the expression matrix is normalised and summarised: the
#: float64 CSR that run_svca still holds (8 + 4), the normalised CSC copy (8 + 4), and the per-entry
#: column index, deviation and squared deviation used for the per-gene variance (3 x 8). Measured
#: with tracemalloc over read_expression + _normalise_and_filter on 3e6 stored entries: 48.1.
BYTES_PER_NONZERO = 48

#: Largest dense spots x genes block materialised for U^T X (the block and its product both exist).
GENE_BLOCK_BYTES = 256 * 1024 * 1024

#: The spot count the old default cap used; kept only as the reference point for the n^3 time note.
_TIME_REFERENCE_SPOTS = 2500

_GIB = float(1 << 30)


def log(msg):
    print(f"[svca-worker] {msg}", file=sys.stderr)


# ----------------------------------------------------------------------------- memory


def estimate_peak_bytes(n_spots, nnz_used=0):
    """Peak bytes of an exact SVCA run over ``n_spots`` spots holding ``nnz_used`` non-zeros.

    Dominated by the ``N_BY_N_COPIES`` dense n x n float64 matrices at the eigendecomposition
    (8 n^2 bytes each), plus the sparse expression matrix and two dense gene blocks.
    """
    n = int(n_spots)
    kernel = N_BY_N_COPIES * 8 * n * n
    sparse_part = BYTES_PER_NONZERO * int(nnz_used)
    blocks = 2 * GENE_BLOCK_BYTES
    return int(kernel + sparse_part + blocks)


def check_memory(n_spots, nnz_used=0, n_spots_total=None, max_spots=None, available=None, n_off_tissue=0):
    """Refuse, naming the numbers and the knob, when the exact kernel cannot fit. Never subsamples.

    Returns ``(needed_bytes, available_bytes_or_None)``. ``available`` is read from the machine when
    not given, with the shared ``worker_utils.available_memory_bytes``: the smaller of MemAvailable and
    the room under the cgroup memory limit, the cgroup's page cache (``active_file`` +
    ``inactive_file``) counted as reclaimable, so a container sitting at its limit on cache alone is
    not refused. When nothing can be read the check passes and says so in the log.

    ``n_spots_total`` is the number of spots the analysis draws from: the in-tissue spots when
    ``n_off_tissue`` background spots (``obs['in_tissue'] == 0``) were left out before it.
    """
    n = int(n_spots)
    need = estimate_peak_bytes(n, nnz_used)
    if available is None:
        available = available_memory_bytes()
    time_factor = (float(n) / _TIME_REFERENCE_SPOTS) ** 3
    if available is None:
        log(f"memory check skipped: available memory cannot be read here (estimated need {need / _GIB:.1f} GiB)")
        return need, None
    if need > available:
        total = int(n_spots_total) if n_spots_total is not None else n
        kind = "in-tissue spots" if n_off_tissue else "spots"
        if max_spots is None:
            scope = f"all {total} {kind} of the slide (no max_spots was set, so none were left out)"
        else:
            scope = f"{n} of the slide's {total} {kind} (max_spots={int(max_spots)})"
        if n_off_tissue:
            scope += f"; the {int(n_off_tissue)} background spots with obs['in_tissue'] == 0 are already excluded"
        raise MemoryError(
            f"SVCA's exact estimator eigendecomposes a dense {n} x {n} float64 spatial kernel and holds "
            f"{N_BY_N_COPIES} such matrices at once ({N_BY_N_COPIES} x 8 x {n}^2 bytes = "
            f"{N_BY_N_COPIES * 8.0 * n * n / _GIB:.1f} GiB, plus {BYTES_PER_NONZERO * int(nnz_used) / _GIB:.1f} GiB "
            f"for the sparse expression matrix): about {need / _GIB:.1f} GiB in total for {scope}, but about "
            f"{available / _GIB:.1f} GiB is available here (MemAvailable / cgroup limit, page cache counted as "
            f"reclaimable). Its time also grows as "
            f"n^3, about {time_factor:,.0f}x a {_TIME_REFERENCE_SPOTS}-spot run. Nothing was computed. This "
            "worker never subsamples on its own and does not replace the exact eigendecomposition with an "
            "approximation; run it where that much memory is free. The only parameter that changes the "
            "spot count is max_spots, which restricts the analysis to a random subset of that many spots "
            "and is reported as such."
        )
    log(
        f"memory estimate: {need / _GIB:.2f} GiB of {available / _GIB:.1f} GiB available; "
        f"eigendecomposition time ~{time_factor:.2f}x a {_TIME_REFERENCE_SPOTS}-spot run (grows as n^3)"
    )
    return need, available


# ----------------------------------------------------------------------------- reading


def _text_array(values):
    """HDF5 strings (bytes, vlen str, fixed-width) as a numpy str array, decoded as UTF-8."""
    out = []
    for v in np.asarray(values).ravel().tolist():
        out.append(v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v))
    return np.asarray(out, dtype=str)


def _attr_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return None if value is None else str(value)


def _read_index(node):
    """The index of an obs/var frame, whatever it is named, or None when the frame has none.

    AnnData names the index dataset in the frame's ``_index`` attribute; it is ``_index`` only by
    default. Real library files use ``barcode``, ``feature_name`` and ``gene_ids``, which the old
    reader missed -- and its fallback then called ``.shape`` on a sparse ``X`` group and crashed.
    Pre-0.7 files store the frame as one compound dataset with an ``index`` field.
    """
    if node is None:
        return None
    if isinstance(node, h5py.Group):
        candidates = []
        named = _attr_text(node.attrs.get("_index"))
        if named:
            candidates.append(named)
        candidates.extend(["_index", "index"])
        for key in candidates:
            if key in node and isinstance(node[key], h5py.Dataset):
                return _text_array(node[key][:])
        return None
    names = getattr(node.dtype, "names", None) or ()
    for key in ("index", "_index"):
        if key in names:
            return _text_array(node[key])
    return None


def _plain_values(values):
    """A dataset's values as numbers/booleans, or as decoded text when HDF5 stored strings."""
    values = np.asarray(values).ravel()
    if values.dtype.kind in ("S", "O", "U"):
        return _text_array(values).astype(object)
    return values


def _categories_of(node, obs_node):
    """The categories a pre-0.8 categorical column refers to through its ``categories`` attribute."""
    ref = node.attrs.get("categories")
    if ref is None:
        return None
    return _plain_values(obs_node.file[ref][()])


def _read_obs_column(obs_node, name):
    """One obs column's values as a 1-D array (categories decoded, missing as None), or None if absent.

    The worker reads the h5ad with h5py, so it decodes the encodings an obs column can take itself: a
    plain dataset (numbers, booleans, strings), a categorical group (``codes`` + ``categories``,
    anndata >= 0.8), a nullable integer/boolean group (``values`` + ``mask``), a pre-0.8 categorical
    (codes with a ``categories`` reference), and a field of the pre-0.7 compound frame.
    """
    if obs_node is None:
        return None
    if isinstance(obs_node, h5py.Dataset):
        names = getattr(obs_node.dtype, "names", None) or ()
        return _plain_values(obs_node[name]) if name in names else None
    if name not in obs_node:
        return None
    node = obs_node[name]
    if isinstance(node, h5py.Group):
        if "codes" in node and "categories" in node:
            codes = np.asarray(node["codes"][()]).ravel().astype(np.int64)
            categories = _plain_values(node["categories"][()])
            out = np.empty(codes.size, dtype=object)
            present = codes >= 0
            out[present] = categories[codes[present]]
            out[~present] = None
            return out
        if "values" in node:
            out = np.asarray(node["values"][()]).ravel().astype(object)
            if "mask" in node:
                out[np.asarray(node["mask"][()]).ravel().astype(bool)] = None
            return out
        raise ValueError(
            f"obs[{name!r}] is stored in an encoding this reader does not know "
            f"({_attr_text(node.attrs.get('encoding-type'))!r}, members {sorted(node.keys())}); re-save the "
            "h5ad with a current anndata so the column can be read."
        )
    categories = _categories_of(node, obs_node)
    if categories is not None:
        codes = np.asarray(node[()]).ravel().astype(np.int64)
        out = np.empty(codes.size, dtype=object)
        present = codes >= 0
        out[present] = categories[codes[present]]
        out[~present] = None
        return out
    return _plain_values(node[()])


class _ObsFrame:
    """Just enough of an AnnData for ``worker_utils.keep_in_tissue``: ``obs``, ``n_obs``, row selection.

    The expression matrix is never loaded whole, so the shared in-tissue rule is applied to this
    stand-in for the obs frame, and the row positions it keeps index the matrix.
    """

    def __init__(self, obs):
        self.obs = obs
        self.n_obs = int(len(obs))

    def __getitem__(self, keep):
        return _ObsFrame(self.obs[np.asarray(keep, dtype=bool)])

    def copy(self):
        return _ObsFrame(self.obs.copy())


def in_tissue_rows(in_tissue):
    """``(rows, n_off_tissue)`` for the ``obs['in_tissue']`` flag read by :func:`read_layout`.

    ``rows`` are the sorted positions of the in-tissue spots, or None when every spot is kept (no
    column, or a column that is 1 everywhere). The rule is ``worker_utils.keep_in_tissue``'s: a spot
    is tissue when its flag reads 1 (or True), and a column that marks no spot as tissue is refused.
    """
    if in_tissue is None:
        return None, 0
    frame = _ObsFrame(pd.DataFrame({"in_tissue": np.asarray(in_tissue, dtype=object)}))
    kept, _n_supplied, n_dropped = keep_in_tissue(frame, "spots")
    if not n_dropped:
        return None, 0
    return np.asarray(kept.obs.index, dtype=np.int64), int(n_dropped)


def _matrix_layout(node, n_obs_hint=None, n_var_hint=None):
    """``(format, shape, nnz)`` of an expression matrix node without reading its values."""
    if isinstance(node, h5py.Dataset):
        if len(node.shape) != 2:
            raise ValueError(f"the expression matrix has {len(node.shape)} dimensions, not 2")
        shape = (int(node.shape[0]), int(node.shape[1]))
        return "dense", shape, shape[0] * shape[1]
    if not (isinstance(node, h5py.Group) and "data" in node and "indices" in node and "indptr" in node):
        raise ValueError("Cannot parse expression matrix from h5ad")
    encoding = (
        _attr_text(node.attrs.get("encoding-type")) or _attr_text(node.attrs.get("h5sparse_format")) or ""
    ).lower()
    fmt = "csc" if "csc" in encoding else "csr"
    n_ptr = int(node["indptr"].shape[0]) - 1
    if "shape" in node.attrs:
        shape = tuple(int(s) for s in node.attrs["shape"])
    elif fmt == "csr":
        shape = (n_ptr, int(n_var_hint) if n_var_hint is not None else 0)
    else:
        shape = (int(n_obs_hint) if n_obs_hint is not None else 0, n_ptr)
    return fmt, shape, int(node["data"].shape[0])


def read_layout(h5ad_path, spatial_key="spatial", use_raw_counts=False):
    """Everything but the expression values: names, coordinates, the matrix's location and size.

    The matrix is ``X``, or ``raw/X`` when the file has no ``X`` or when ``use_raw_counts`` asks for it
    (refused when the file has no ``raw/X``). ``raw_path`` names a ``raw/X`` that was not chosen, so a
    refusal of a non-count ``X`` can say whether the file holds counts elsewhere.
    """
    log(f"Reading layout: {h5ad_path}")
    with h5py.File(h5ad_path, "r") as f:
        has_raw = "raw" in f and "X" in f["raw"]
        if use_raw_counts:
            if not has_raw:
                raise ValueError("use_raw_counts=True runs on adata.raw, and this h5ad has no adata.raw.")
            x_path = "raw/X"
            var_node = f["raw"]["var"] if "var" in f["raw"] else None
        elif "X" in f:
            x_path, var_node = "X", f["var"] if "var" in f else None
        elif has_raw:
            # raw.X has its own gene axis; its names live in raw/var, not var.
            x_path = "raw/X"
            var_node = f["raw"]["var"] if "var" in f["raw"] else None
        else:
            raise ValueError("No expression matrix found in h5ad file (checked X and raw/X)")
        # The matrix NOT chosen, when there is one: X under use_raw_counts, raw/X otherwise.
        other_path = None
        if x_path == "raw/X" and "X" in f:
            other_path = "X"
        elif x_path == "X" and has_raw:
            other_path = "raw/X"

        obs_names = _read_index(f["obs"]) if "obs" in f else None
        in_tissue = _read_obs_column(f["obs"] if "obs" in f else None, "in_tissue")
        var_names = _read_index(var_node)
        fmt, shape, nnz = _matrix_layout(
            f[x_path],
            None if obs_names is None else len(obs_names),
            None if var_names is None else len(var_names),
        )
        n_obs, n_var = shape

        if "obsm" not in f:
            raise ValueError("No 'obsm' group found in h5ad file")
        if spatial_key not in f["obsm"]:
            raise KeyError(
                f"spatial_key={spatial_key!r} is not in obsm. Available keys: {list(f['obsm'].keys())}. "
                "Pass the key that holds this slide's coordinates; the worker does not substitute another one."
            )
        coords = read_obsm_matrix(f, spatial_key)
        coords, _ = spatial_coords(coords, spatial_key, want=2, tool="SVCA")

    warnings = []
    if obs_names is None:
        obs_names = np.asarray([f"spot_{i}" for i in range(n_obs)], dtype=str)
    if var_names is None:
        var_names = np.asarray([f"gene_{i}" for i in range(n_var)], dtype=str)
        warnings.append(
            f"the h5ad has no gene index for {x_path}; genes are reported as gene_<column number> "
            "(gene_0, gene_1, ...), which match no external annotation."
        )
    if len(var_names) != n_var:
        raise ValueError(f"{x_path} has {n_var} genes but its gene index has {len(var_names)} names")
    if coords.shape[0] != n_obs:
        raise ValueError(f"obsm[{spatial_key!r}] has {coords.shape[0]} rows but {x_path} has {n_obs} spots")
    if in_tissue is not None and len(in_tissue) != n_obs:
        raise ValueError(f"obs['in_tissue'] has {len(in_tissue)} values but {x_path} has {n_obs} spots")

    return {
        "x_path": x_path,
        # The other matrix of the file (X or raw/X), or None; see read_layout's docstring.
        "other_path": other_path,
        "format": fmt,
        "shape": (int(n_obs), int(n_var)),
        "nnz": int(nnz),
        "obs_names": obs_names,
        "var_names": var_names,
        "coords": coords,
        # None when the file has no in_tissue column; see in_tissue_rows.
        "in_tissue": in_tissue,
        "warnings": warnings,
    }


def read_expression(h5ad_path, layout, rows=None):
    """The expression matrix as float64 CSR over ``rows`` (all spots when None). Never densified."""
    fmt, shape, x_path = layout["format"], layout["shape"], layout["x_path"]
    with h5py.File(h5ad_path, "r") as f:
        node = f[x_path]
        if fmt == "dense":
            # Stored dense: read in row blocks and keep it sparse from here on.
            n_obs, n_var = shape
            step = max(1, int(GENE_BLOCK_BYTES // max(1, 8 * n_var)))
            pieces = []
            for r0 in range(0, n_obs, step):
                r1 = min(n_obs, r0 + step)
                if rows is None:
                    block = np.asarray(node[r0:r1], dtype=np.float64)
                else:
                    wanted = rows[(rows >= r0) & (rows < r1)]
                    if wanted.size == 0:
                        continue
                    block = np.asarray(node[r0:r1], dtype=np.float64)[wanted - r0]
                pieces.append(sp.csr_matrix(block))
            X = sp.vstack(pieces, format="csr") if pieces else sp.csr_matrix((0, n_var))
            return X.astype(np.float64, copy=False)
        data = node["data"][:]
        indices = node["indices"][:]
        indptr = node["indptr"][:]
    if fmt == "csc":
        X = sp.csc_matrix((data, indices, indptr), shape=shape).tocsr()
    else:
        X = sp.csr_matrix((data, indices, indptr), shape=shape)
    if rows is not None:
        X = X[rows]
    X = X.astype(np.float64)
    # Duplicate entries add up when densified; normalising them one by one would not.
    X.sum_duplicates()
    return X


def stored_matrix_kind(h5ad_path, x_path, chunk=1 << 22):
    """``worker_utils.expression_matrix_kind`` of a whole stored matrix, read in chunks from the file.

    Used only for the matrix the run did NOT analyse (to say whether ``raw/X`` holds counts, or what
    ``X`` held under ``use_raw_counts``), so it never materialises a second matrix: a sparse node's
    stored values are streamed ``chunk`` at a time, a dense one a block of rows at a time.
    """
    kinds = []
    with h5py.File(h5ad_path, "r") as f:
        node = f[x_path]
        if isinstance(node, h5py.Dataset):
            n_rows = int(node.shape[0])
            step = max(1, int(chunk // max(1, int(node.shape[1]) if len(node.shape) > 1 else 1)))
            for r0 in range(0, n_rows, step):
                kinds.append(expression_matrix_kind(np.asarray(node[r0 : r0 + step], dtype=np.float64)))
        else:
            data = node["data"]
            n = int(data.shape[0])
            for start in range(0, n, chunk):
                values = np.asarray(data[start : start + chunk], dtype=np.float64).reshape(1, -1)
                kinds.append(expression_matrix_kind(values))
    for worst in ("nonfinite", "negative", "nonnegative_noninteger", "counts"):
        if worst in kinds:
            return worst
    return "empty"


def _matrix_label(x_path):
    """How the payload names a matrix of the file: ``X`` or ``raw.X`` (choose_counts_matrix's words)."""
    return "raw.X" if x_path == "raw/X" else "X"


def decide_counts(X, layout, h5ad_path, use_raw_counts=False):
    """``worker_utils.choose_counts_matrix``'s rule for the matrix already read, by hand.

    SVCA library-size normalises and log1p-transforms what it decomposes, so the matrix has to hold
    counts. Returns ``{"expression_source", "x_matrix_kind", "warning"}``; raises on a matrix that is
    not counts (negative or non-finite values), naming ``use_raw_counts`` when ``raw/X`` holds counts.
    ``X`` is the matrix over the analysed spots, so the verdict is about the spots that are decomposed.
    """
    x_path = layout["x_path"]
    kind = expression_matrix_kind(X)
    source = _matrix_label(x_path)
    other = layout.get("other_path")
    if use_raw_counts:
        if kind != "counts":
            raise ValueError(f"use_raw_counts=True, but adata.raw.X holds {kind.replace('_', ' ')} values, not counts.")
        x_kind = stored_matrix_kind(h5ad_path, other) if other == "X" else None
        return {"expression_source": source, "x_matrix_kind": x_kind, "warning": None}
    x_kind = kind if x_path == "X" else None
    if kind in ("negative", "nonfinite", "nonnegative_noninteger"):
        raw_kind = stored_matrix_kind(h5ad_path, other) if other == "raw/X" else None
        if raw_kind == "counts":
            hint = " adata.raw holds raw counts: pass use_raw_counts=True to run on them."
        elif x_path == "X":
            hint = " Supply an h5ad whose X (or adata.raw with use_raw_counts=True) holds raw counts."
        else:
            hint = " Supply an h5ad whose X or raw/X holds raw counts."
        if kind in ("negative", "nonfinite"):
            what = "negative values (scaled or z-scored data)" if kind == "negative" else "NaN or infinite values"
            raise ValueError(
                f"{source} holds {what}, not counts, and SVCA library-size normalises and log1p-transforms it "
                "as counts, which turns those values into NaN; nothing was decomposed." + hint
            )
        warning = (
            f"{source} holds non-integer values (normalised or log-transformed data?), and this tool normalises "
            f"{source} as counts, so the result was computed on a matrix normalised twice." + hint
        )
        return {"expression_source": source, "x_matrix_kind": x_kind, "warning": warning}
    return {"expression_source": source, "x_matrix_kind": x_kind, "warning": None}


def choose_spots(n_spots, max_spots, seed):
    """Sorted indices of the random ``max_spots`` subset, or None when every spot is analysed.

    Only an explicit ``max_spots`` below the slide's spot count draws a subset. The draw is the one
    the capped runs always made (``default_rng(seed).choice(..., replace=False)``, sorted), so an
    explicit ``max_spots=2500`` reproduces a run recorded under the old default cap.
    """
    if max_spots is None or n_spots <= max_spots:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_spots, size=max_spots, replace=False)
    idx.sort()
    return idx


# ----------------------------------------------------------------------------- method


def mark_top_decile(results_df):
    """Flag the genes this run calls spatially variable: the top 10% by intrinsic fraction.

    This used to be written as ``pvalue``, set to 0.01 on these rows and 1.0 on the rest. Nothing
    computed it -- there is no permutation, no null and no likelihood ratio anywhere in this worker
    -- so a rank cutoff was reaching users wearing a statistic's name. It got quoted back verbatim:
    asked for numbers for a figure legend, the agent proposed "intrinsic variance fractions
    0.997-0.9999; all p=0.01".

    The selection itself is unchanged, and deliberately so: ``standardize_svg_output`` turns this
    column into the ``significant`` set that every recorded SVCA F1 was scored on. Renaming the flag
    keeps those genes identical; dropping it would fall through to ``score != 0.0`` and call all 647
    genes significant. What changes is only that the column now says what it is.

    Expects ``results_df`` already sorted by ``intrinsic`` descending with a reset index, which is
    how the caller builds it.
    """
    n_top = max(1, int(np.ceil(len(results_df) * 0.10)))
    results_df["top_decile_by_intrinsic"] = results_df.index < n_top
    return results_df


def _column_variance(Xc):
    """Population variance (ddof=0) of every column of a CSC matrix, without densifying it.

    Two-pass (mean first, then squared deviations of the stored entries plus the implicit zeros),
    so it does not lose precision the way E[x^2] - E[x]^2 does.
    """
    n, g = Xc.shape
    counts = np.diff(Xc.indptr)
    col = np.repeat(np.arange(g), counts)
    mean = np.bincount(col, weights=Xc.data, minlength=g) / float(n)
    dev = Xc.data - mean[col]
    ss = np.bincount(col, weights=dev * dev, minlength=g)
    return (ss + (n - counts) * mean * mean) / float(n)


def _normalise_and_filter(X, var_names):
    """Library-size + log1p normalise the sparse matrix; drop near-constant genes.

    Returns ``(Xc, var_names_kept, var_kept, keep_columns)`` where ``Xc`` is the normalised matrix
    in CSC form over ALL genes and ``keep_columns`` indexes the genes that survive. The arithmetic
    is the dense version's, applied to the stored entries (zeros stay zero under x / l * m and
    log1p), so nothing is densified here. It is not done in place: the normalised values go into
    new arrays while the caller still holds the matrix it passed, so this phase holds the caller's
    CSR, the normalised CSC and the per-gene variance work arrays at once -- ``BYTES_PER_NONZERO``
    bytes per stored entry, which is what the memory check counts.
    """
    X = sp.csr_matrix(X)
    if X.dtype != np.float64:
        X = X.astype(np.float64)
    X.sum_duplicates()
    libsizes = np.asarray(X.sum(axis=1)).ravel()
    libsizes[libsizes == 0] = 1.0
    med = np.median(libsizes)
    X.data = X.data / np.repeat(libsizes, np.diff(X.indptr)) * med
    np.log1p(X.data, out=X.data)
    Xc = X.tocsc()
    del X
    n_nonfinite = int(np.count_nonzero(~np.isfinite(Xc.data)))
    if n_nonfinite:
        # decide_counts refuses the inputs that produce these; a NaN gene must never be dropped as
        # "invariant" (NaN > 1e-8 is False) and the rest described as the genes that vary.
        raise ValueError(
            f"library-size + log1p normalisation left {n_nonfinite} non-finite value(s); the matrix does not "
            "hold counts SVCA can normalise. Nothing was decomposed."
        )
    var = _column_variance(Xc)
    keep = var > 1e-8
    return Xc, np.asarray(var_names)[keep], var[keep], np.flatnonzero(keep)


def describe_spot_coverage(n_spots_total: int, n_spots_used: int, n_off_tissue: int = 0) -> str:
    """Phrase how much of the slide was analysed, so a partial run cannot read as a whole one.

    Every in-tissue spot is analysed unless ``--max-spots`` is set explicitly, in which case the
    analysis covers a random subset of them. The counts were always in the JSON payload, but the
    ``analysis`` sentence -- the one the agent quotes to the user -- once said only "across 2500
    spots", and 3.2% of a 78329-spot MERFISH slide reads exactly like all of it.

    ``n_off_tissue`` background spots (``obs['in_tissue'] == 0``) left out before the analysis are
    named separately: that cut is not random, and calling it a subsample would misdescribe it.

    A slide analysed whole gets no qualifier at all: a warning on a complete run trains the
    reader to skip the one that matters.
    """
    n_off_tissue = int(n_off_tissue or 0)
    n_eligible = n_spots_total - n_off_tissue
    background = ""
    if n_off_tissue:
        background = (
            f" ({n_off_tissue} of the file's {n_spots_total} spots have obs['in_tissue'] == 0 and were left out "
            "as background)"
        )
    if n_spots_used >= n_eligible:
        if n_off_tissue:
            return f"across all {n_eligible} in-tissue spots{background}"
        return f"across all {n_spots_total} spots"
    pct = 100.0 * n_spots_used / n_eligible if n_eligible else 0.0
    kind = "in-tissue spots" if n_off_tissue else "spots"
    return f"across a random subsample of {n_spots_used} of the slide's {n_eligible} {kind} ({pct:.1f}%){background}"


def describe_gene_coverage(n_genes_total: int, n_genes_used: int, ranked: bool) -> str:
    """Phrase which genes were analysed, and say whether they were picked by rank.

    The companion to :func:`describe_spot_coverage`, and the more dangerous of the two. Spot
    subsampling is random, so a capped run is noisier but unbiased. ``--n-genes N`` keeps the N
    highest-variance genes, so every statistic averaged over that subset is biased upward by
    construction: a live run capped at 100 genes reported a mean intrinsic fraction of 0.38 where
    the same slide over all 649 genes gives 0.127, and the user was quoting it in a figure legend.

    ``ranked`` has to be passed in rather than inferred from the counts, because two unrelated things
    shrink the gene count and only one of them biases the mean. Every run drops invariant genes
    (``var > 1e-8``) as quality control; only an explicit ``--n-genes`` sorts on variance and takes a
    head. Inferring from ``n_genes_used < n_genes_total`` conflates them, which is how an uncapped run
    over a real MERFISH slide came to describe its 635-of-649 QC filter as "the 635 highest-variance".
    An invented caveat is not a safe default: a reader who discounts a mean for a bias that never
    happened has still been misled.

    A run that analysed every gene says so plainly and gets no qualifier either way.
    """
    if n_genes_used >= n_genes_total:
        return f"all {n_genes_total} genes"
    if ranked:
        return f"the {n_genes_used} highest-variance of the slide's {n_genes_total} genes"
    return f"{n_genes_used} of the slide's {n_genes_total} genes (the rest have no variance across the analysed spots)"


def _median_offdiagonal_distance(D2):
    """``np.median(np.sqrt(D2[np.triu_indices(n, k=1)]))`` without the two n^2/2 index arrays.

    The same values in the same order, so the median -- and the kernel bandwidth -- is identical;
    the index arrays alone were 8 n^2 bytes, as much as the kernel itself.
    """
    n = D2.shape[0]
    m = n * (n - 1) // 2
    vals = np.empty(m, dtype=np.float64)
    pos = 0
    for i in range(n - 1):
        row = D2[i, i + 1 :]
        vals[pos : pos + row.size] = row
        pos += row.size
    np.sqrt(vals, out=vals)
    return float(np.median(vals, overwrite_input=True))


def _build_rbf_kernel(coords):
    """RBF kernel with median-pairwise-distance bandwidth. Returns (K, length_scale).

    Built in place so that no more than two n x n matrices exist at once; every value is computed
    with the same floating-point operations, in the same order, as the expression it replaces,
    ``exp(-(sq[:, None] + sq[None, :] - 2.0 * coords @ coords.T) / (2 l^2))``. The product is kept
    as ``(2.0 * coords) @ coords.T``: ``coords @ coords.T`` shares one buffer, which numpy hands to
    BLAS ``syrk`` instead of ``gemm``, and that rounds differently in the last bit.
    """
    sq = np.sum(coords**2, axis=1)
    D2 = sq[:, None] + sq[None, :]
    G = (2.0 * coords) @ coords.T
    D2 -= G
    del G
    np.maximum(D2, 0.0, out=D2)
    med = _median_offdiagonal_distance(D2)
    length_scale = med if med > 0 else 1.0
    np.negative(D2, out=D2)
    D2 /= 2.0 * length_scale * length_scale
    np.exp(D2, out=D2)
    return D2, length_scale


def _fastlmm_spatial_fraction(uy_col, eigvals, Uone, n_spots):
    """Closed-form REML per-gene FaST-LMM spatial fraction (Lippert 2011)."""
    yvar = float(uy_col.var(ddof=1))
    if yvar <= 1e-12 or not np.isfinite(yvar):
        return 0.0, False, 0.0
    uy = uy_col / np.sqrt(yvar)

    def negll(log_delta):
        delta = np.exp(log_delta)
        Sd = eigvals + delta
        if (Sd <= 0).any():
            return 1e30
        num = float((Uone * uy / Sd).sum())
        den = float((Uone * Uone / Sd).sum())
        if den <= 0 or not np.isfinite(den):
            return 1e30
        beta = num / den
        res = uy - beta * Uone
        sig2_s = float((res * res / Sd).sum() / (n_spots - 1))
        if sig2_s <= 0 or not np.isfinite(sig2_s):
            return 1e30
        ll = (
            -0.5 * (n_spots - 1) * np.log(2.0 * np.pi)
            - 0.5 * np.sum(np.log(Sd))
            - 0.5 * (n_spots - 1) * np.log(sig2_s)
            - 0.5 * np.log(den)
        )
        return -float(ll)

    res = minimize_scalar(
        negll,
        bounds=(-10.0, 10.0),
        method="bounded",
        options={"xatol": 1e-3, "maxiter": 50},
    )
    delta = float(np.exp(res.x))
    mean_eig = float(eigvals.mean())
    frac = mean_eig / (mean_eig + delta)
    return frac, bool(res.success), yvar


def _atomic_write(path, write):
    """Write through ``<name>.partial`` and rename, so a crash never leaves a truncated result."""
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    try:
        write(partial)
        os.replace(str(partial), str(path))
    finally:
        if partial.exists():
            partial.unlink()


def run_svca(h5ad_path, output_dir, spatial_key="spatial", n_genes=None, max_spots=None, seed=0, use_raw_counts=False):
    log("Starting SVCA variance decomposition (FaST-LMM)")
    if max_spots is not None and int(max_spots) < 3:
        raise ValueError(
            f"max_spots={max_spots} cannot be used: the estimator needs at least 3 spots. Leave max_spots "
            "unset to analyse every spot."
        )
    if n_genes is not None and int(n_genes) < 1:
        raise ValueError(f"n_genes={n_genes} cannot be used: leave it unset to analyse every gene.")
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    use_raw_counts = bool(use_raw_counts)
    layout = read_layout(h5ad_path, spatial_key, use_raw_counts=use_raw_counts)
    n_spots_total, n_genes_total = layout["shape"]
    log(f"Slide: {n_spots_total} spots, {n_genes_total} genes ({layout['x_path']}, {layout['format']})")

    # Background first (obs['in_tissue'] == 0, the shared keep_in_tissue rule), then the opt-in random
    # subset of what is left. Both are row positions into the file's spot axis.
    tissue_rows, n_off_tissue = in_tissue_rows(layout["in_tissue"])
    n_spots_eligible = n_spots_total - n_off_tissue
    if n_off_tissue:
        log(f"Leaving out {n_off_tissue} of {n_spots_total} spots with obs['in_tissue'] == 0 (background)")
    if n_spots_eligible < 3:
        where = (
            f"{n_spots_eligible} in-tissue spot(s) of {n_spots_total}" if n_off_tissue else f"{n_spots_total} spot(s)"
        )
        raise ValueError(f"the slide has {where}; the estimator needs at least 3.")

    picked = choose_spots(n_spots_eligible, max_spots, seed)
    if tissue_rows is None:
        rows = picked
    else:
        rows = tissue_rows if picked is None else tissue_rows[picked]
    n_spots = n_spots_eligible if picked is None else int(picked.size)
    if picked is not None:
        log(f"max_spots={max_spots}: analysing a random {n_spots} of {n_spots_eligible} spots (seed={seed})")

    # Refuse before anything large is read: the kernel's size is fixed by the spot count alone. A
    # sparse X states its non-zero count in the file; a dense one does not, and n x g would overstate
    # it, so its expression term is checked once the matrix has been read (sparsely).
    stored_dense = layout["format"] == "dense"
    if stored_dense:
        nnz_used = 0
    else:
        nnz_used = layout["nnz"] if rows is None else int(np.ceil(layout["nnz"] * float(n_spots) / n_spots_total))
    check_memory(n_spots, nnz_used, n_spots_total=n_spots_eligible, max_spots=max_spots, n_off_tissue=n_off_tissue)

    coords = layout["coords"] if rows is None else layout["coords"][rows]
    X = read_expression(h5ad_path, layout, rows)
    log(f"Loaded expression (sparse): {X.shape[0]} spots x {X.shape[1]} genes, {X.nnz} stored entries")
    if stored_dense:
        check_memory(n_spots, X.nnz, n_spots_total=n_spots_eligible, max_spots=max_spots, n_off_tissue=n_off_tissue)
    # Counts or refuse, over the spots that are decomposed (the shared choose_counts_matrix rule).
    counts_info = decide_counts(X, layout, h5ad_path, use_raw_counts=use_raw_counts)

    Xc, var_names, var_kept, keep_cols = _normalise_and_filter(X, layout["var_names"])
    del X
    n_genes_used = int(keep_cols.size)
    log(f"After normalise+filter: {n_spots} spots, {n_genes_used} genes")
    if n_genes_used == 0:
        raise ValueError(
            f"none of the {n_genes_total} genes varies across the {n_spots} analysed spots "
            "(variance <= 1e-8 after library-size + log1p normalisation); there is nothing to decompose."
        )

    # Optional HVG pre-restriction. Default is None == use all variance-filtered
    # genes (preferred for SVG benchmarking).
    # Recorded so the disclosure can distinguish this ranked cut from the invariant-gene filter
    # above; the two are indistinguishable from the counts alone.
    ranked_gene_subset = False
    gene_cols = keep_cols
    # Population variance of each analysed gene's normalised expression, aligned with gene_cols.
    gene_var = var_kept
    if n_genes is not None and n_genes < n_genes_used:
        log(f"Restricting to top {n_genes} HVGs by variance...")
        top_idx = np.argsort(var_kept)[::-1][:n_genes]
        gene_cols = keep_cols[top_idx]
        var_names = var_names[top_idx]
        gene_var = var_kept[top_idx]
        n_genes_used = int(gene_cols.size)
        ranked_gene_subset = True

    log("Building RBF spatial kernel...")
    K, length_scale = _build_rbf_kernel(coords)
    log(f"RBF length_scale = {length_scale:.3f}")

    t0 = time.time()
    # Match the manual reference exactly: scipy.linalg.eigh (handles NaN-column
    # bug present in some older numpy/BLAS combos), retain ALL eigvecs, then
    # only clip the numerical-jitter negatives to 0. Pruning the non-positive
    # eigvecs caused the FaST-LMM REML to converge to a single delta for every
    # gene (the residual mass was concentrated in dropped low-frequency modes
    # which then forced delta -> +infty for every gene).
    eigvals, U = sla.eigh(K)
    del K
    n_neg = int((eigvals < 0).sum())
    eigvals = np.clip(eigvals, 0.0, None)
    if n_neg:
        log(f"Clipped {n_neg} small-magnitude negative eigvals to 0 (numerical jitter)")
    # If any U columns are non-finite (older numpy/openblas combo), drop them.
    finite_cols = np.isfinite(U).all(axis=0)
    n_nan = int((~finite_cols).sum())
    if n_nan:
        U = U[:, finite_cols]
        eigvals = eigvals[finite_cols]
        log(f"Dropped {n_nan} NaN-column eigvecs (LAPACK numerical-zero fallout)")
    log(
        f"eigh done in {time.time() - t0:.1f}s; n_eig={eigvals.size} "
        f"min={eigvals.min():.3g} max={eigvals.max():.3g} mean={eigvals.mean():.3g}"
    )

    UT = U.T
    Uone = UT @ np.ones(n_spots)
    block = max(1, int(GENE_BLOCK_BYTES // max(1, 8 * n_spots)))

    spatial_frac = np.zeros(n_genes_used, dtype=np.float64)
    converged_arr = np.zeros(n_genes_used, dtype=bool)
    # The gene's own variance (ddof=1) over the analysed spots. The rotated vector's variance that
    # _fastlmm_spatial_fraction scales by is var(U^T y) ~ mean^2 + var, not the gene's variance.
    total_var_arr = np.asarray(gene_var, dtype=np.float64) * (float(n_spots) / max(1.0, float(n_spots - 1)))
    t_loop = time.time()
    n_nonfinite = 0
    for b0 in range(0, n_genes_used, block):
        cols = gene_cols[b0 : b0 + block]
        # The only densification: this block of genes over the analysed spots, for U^T X.
        UY = UT @ Xc[:, cols].toarray()
        n_nonfinite += int((~np.isfinite(UY)).any(axis=0).sum())
        for k in range(UY.shape[1]):
            j = b0 + k
            frac, ok, _rotated_var = _fastlmm_spatial_fraction(UY[:, k], eigvals, Uone, n_spots)
            spatial_frac[j] = frac
            converged_arr[j] = ok
            if (j + 1) % 2000 == 0:
                log(f"FaST-LMM {j + 1}/{n_genes_used} genes; elapsed={time.time() - t_loop:.0f}s")
    del U, UT, Xc
    if n_nonfinite:
        log(f"{n_nonfinite} gene(s) had non-finite U^T x values")
    log(
        f"FaST-LMM loop done in {time.time() - t_loop:.1f}s "
        f"(max_frac={spatial_frac.max():.3f}, median={np.median(spatial_frac):.3f})"
    )

    results_df = pd.DataFrame(
        {
            "gene": var_names,
            "intrinsic": spatial_frac,
            # NaN, not 0.0: this worker fits one spatial random effect against i.i.d. residual
            # noise, so there is no third component to report. The literal zero that used to sit
            # here was quoted back to a user as "Mean environmental fraction: 0.0", which reads as
            # a finding -- no environmental or cell-cell-interaction contribution in this tissue --
            # rather than as the absence of a measurement. An empty cell cannot be misread that way.
            "environmental": np.nan,
            "noise": 1.0 - spatial_frac,
            "total_variance": total_var_arr,
            "converged": converged_arr,
        }
    )
    results_df = results_df.sort_values("intrinsic", ascending=False).reset_index(drop=True)

    n_top = max(1, int(np.ceil(n_genes_used * 0.10)))
    results_df = mark_top_decile(results_df)

    decomp_path = out_dir / "svca_variance_decomposition.csv"
    _atomic_write(decomp_path, lambda p: results_df.to_csv(p, index=False))
    log(f"Saved {decomp_path}")

    top_spatial = results_df.head(n_top)
    summary_path = out_dir / "svca_summary.csv"
    _atomic_write(summary_path, lambda p: top_spatial.to_csv(p, index=False))
    log(f"Saved {summary_path}  ({n_top} top genes)")

    # Standardised SVG artifact for the agent evaluator (mirrors manual runner).
    predicted_genes = [
        str(g)
        for g in top_spatial["gene"].tolist()
        if not str(g).startswith("Blank-") and not str(g).startswith("Blank")
    ]
    pg_path = out_dir / "predicted_genes.json"

    def _write_predicted(p):
        with open(p, "w") as fp:
            json.dump({"predicted_genes": predicted_genes}, fp)

    _atomic_write(pg_path, _write_predicted)
    log(f"Saved {pg_path} with {len(predicted_genes)} SVG predictions")

    n_converged = int(converged_arr.sum())
    mean_intrinsic = float(results_df["intrinsic"].mean())
    n_spatial_genes = int((results_df["intrinsic"] > 0.3).sum())
    top_gene_names = list(top_spatial["gene"].values[:10])

    log(
        f"Converged: {n_converged}/{n_genes_used}; mean intrinsic={mean_intrinsic:.3f}; "
        f"genes with intrinsic>0.3: {n_spatial_genes}"
    )

    # The opt-in random subset only: leaving out the background is not a subsample.
    subsampled = bool(n_spots < n_spots_eligible)
    out = WorkerOutput("svca", task="variance_decomposition")
    out.set_data(
        n_spots=n_spots_total,
        n_spots_used=n_spots,
        n_genes_total=n_genes_total,
        n_genes_analyzed=n_genes_used,
    )
    out.add_output_files(
        {
            "variance_decomposition_csv": str(decomp_path),
            "summary_csv": str(summary_path),
            "predicted_genes_json": str(pg_path),
        }
    )
    out.add_params(
        {
            # The key the coordinates were read from: the requested one, or the run refused.
            "spatial_key": spatial_key,
            "expression_matrix": layout["x_path"],
            "use_raw_counts": use_raw_counts,
            "n_genes": n_genes,
            "length_scale": float(length_scale),
            # None = no cap: every spot analysed. A number is an explicit request for a random subset.
            "max_spots": max_spots,
            # Not a formality here: this seed picks *which* spots were analysed when the slide is
            # larger than max_spots, so without it the analysed subset cannot be recovered.
            "seed": seed,
            "subsampled_spots": subsampled,
            # Ranked, not random: recorded separately from the spot subsample because it biases
            # every averaged statistic upward rather than merely adding noise. Taken from the branch
            # that actually ranked, not from the counts -- the invariant-gene filter shrinks them too.
            "restricted_to_top_variance_genes": ranked_gene_subset,
        }
    )
    record_method(out, METHOD)
    record_in_tissue(out, n_spots_total, n_off_tissue)
    # params.expression_source / params.x_matrix_kind and the not-counts warning, in the words
    # worker_utils.record_expression_source gives every other worker.
    out.add_params(
        {"expression_source": counts_info["expression_source"], "x_matrix_kind": counts_info["x_matrix_kind"]}
    )
    if counts_info["warning"]:
        out.add_warning(counts_info["warning"])
    for message in layout["warnings"]:
        out.add_warning(message)
    if not subsampled and seed != 0:
        record_ignored(
            out,
            ["seed"],
            "the seed only chooses the random spot subset drawn when max_spots is below the slide's spot "
            "count; no subset was drawn, every spot was analysed, and the estimator itself is deterministic.",
        )
    if subsampled:
        kind = "in-tissue spots" if n_off_tissue else "spots"
        out.add_warning(
            f"max_spots={max_spots}: a random {n_spots} of the slide's {n_spots_eligible} {kind} "
            f"({100.0 * n_spots / n_spots_eligible:.1f}%, seed={seed}) were analysed; the rest of the slide was not."
        )
    out.set_summary(
        n_converged=n_converged,
        mean_intrinsic=mean_intrinsic,
        # A string rather than a number or NaN: this key is read by an LLM that will quote it, and
        # `null` invites "environmental variance was zero" just as readily as 0.0 did.
        mean_environmental="not estimated",
        mean_noise=float(1.0 - mean_intrinsic),
        n_spatial_genes=n_spatial_genes,
        top_spatial_genes=top_gene_names,
    )
    out.set_analysis(
        f"SVCA (FaST-LMM REML) decomposed variance of "
        f"{describe_gene_coverage(n_genes_total, n_genes_used, ranked_gene_subset)} "
        f"{describe_spot_coverage(n_spots_total, n_spots, n_off_tissue)}. "
        f"{n_converged}/{n_genes_used} models converged. "
        # Said in the sentence the agent quotes, not only in the docstring: the column named
        # `intrinsic` is the fraction of variance taken by the spatial random effect, and the
        # environmental / cell-cell-interaction term of the published SVCA model is not fitted here.
        f"This is a two-component fit (spatial vs. residual noise); the environmental / "
        f"cell-cell-interaction component of the published SVCA model is not estimated. "
        f"Mean intrinsic (spatial) fraction: {mean_intrinsic:.3f}. "
        f"{n_spatial_genes} genes show >30% intrinsic spatial variance. "
        f"Top spatial genes: {', '.join(top_gene_names[:5])}."
    )
    return out.to_dict()


def main():
    parser = argparse.ArgumentParser(description="SVCA worker: FaST-LMM spatial variance fractions.")
    parser.add_argument("--h5ad-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument(
        "--n-genes",
        type=int,
        default=None,
        help="Number of top HVGs to analyze. None (default) analyses ALL genes (preferred for SVG benchmarking).",
    )
    parser.add_argument(
        "--max-spots",
        type=int,
        default=None,
        help=(
            "Opt-in: analyse a random subset of at most this many spots (drawn with --seed). Default: none, "
            "every spot is analysed; the run refuses up front, with the numbers, if the exact n x n kernel "
            "cannot fit in memory."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        default=False,
        help=(
            "Decompose adata.raw.X (raw/X) instead of X. Use it when X is log-normalised or scaled and the "
            "counts sit in adata.raw (CELLxGENE exports); refused when the file has no raw/X or it is not counts."
        ),
    )
    args = parser.parse_args()

    try:
        result = run_svca(
            h5ad_path=args.h5ad_path,
            output_dir=args.output_dir,
            spatial_key=args.spatial_key,
            n_genes=args.n_genes,
            max_spots=args.max_spots,
            seed=args.seed,
            use_raw_counts=args.use_raw_counts,
        )
        print(json.dumps(result, default=str))
        sys.stdout.flush()
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("svca", str(e), task="variance_decomposition")
        sys.exit(1)


if __name__ == "__main__":
    main()
