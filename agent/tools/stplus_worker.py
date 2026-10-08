#!/usr/bin/env python
"""
stPlus worker for SpatialOmicsLab MCP.

Runs the stPlus spatial gene imputation pipeline:
  1. Load spatial and scRNA-seq data (h5ad, or a delimited text matrix -- .csv/.tsv/.txt,
     optionally gzipped, with the separator read from the file rather than its name).
  2. Load the list of genes to impute (one per line, or the first column of a table).
  3. Put both matrices on the scale stPlus is written for. Upstream's docstring asks for
     "normalized and logarithmized" spatial and reference data; with ``normalize='auto'`` a matrix
     of raw counts is log1p(normalize_total(target_sum=1e4))'d, anything else is used as given,
     and the payload says which happened to each side.
  4. Append up to ``top_k`` (upstream default 2000) highly variable reference genes that are
     neither anchors nor targets -- the augmentation stPlus's own ``top_k`` step makes.
  5. Apply runtime .cuda() -> .to(device) monkey-patch for CPU compatibility.
  6. Run stPlus with its checkpoints in a fresh per-run directory (upstream writes them to
     ``./stPlus-5min*.pt`` in the working directory and averages every such file it finds).
  7. Save the imputed expression, keyed by the spot barcodes, as CSV and annotated h5ad.

Environment: /opt/conda/envs/stplus_env

stPlus reference:
  Shengquan Chen et al., "stPlus: a reference-based method for the accurate
  enhancement of spatial transcriptomics", Bioinformatics, 2021.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

# Ensure worker_utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    default_output_dir,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    preflight_check,
    record_in_tissue,
    record_method,
    resolve_compute,
    sniff_tabular_sep,
    unsupported_choice_msg,
)

#: ``normalize`` values. ``auto`` log-normalises a matrix only when it holds raw counts.
NORMALIZE_CHOICES = ("auto", "always", "never")
#: Library size every spot/cell is scaled to before ``log1p`` when the worker normalises.
TARGET_SUM = 1e4
#: What a normalised matrix holds, as the payload spells it.
LOG_NORMALIZED = "log1p(normalize_total(target_sum=1e4))"
#: Above this, a non-integer matrix is not on a log scale (log1p of a 1e4-scaled count stays < 10).
LOG_SCALE_CEILING = 50.0
#: Upstream's own defaults, passed explicitly so the payload can report them as the values used.
TOP_K_DEFAULT = 2000  # stPlus(top_k=2000): highly variable reference genes appended to the model
T_MIN = 5  # stPlus(t_min=5): the lowest-loss epochs whose checkpoints are averaged at prediction
CONVERGE_RATIO = 0.004  # stPlus(converge_ratio=0.004): training stops once the loss moves less than this
#: Minimum anchors (measured genes shared with the reference and not being imputed).
MIN_ANCHOR_GENES = 10
#: Peak bytes per (row x model gene) inside ``stPlus()``: the spatial rows are held as float64 in
#: ``spatial_df_appended`` and again in the float64 ``np.vstack`` that feeds the float32 training
#: tensor (8 + 8 + 4), the reference rows as float32 in its own frame, the vstack and the tensor
#: (4 + 8 + 4). Measured against upstream model.py; the dense matrices are the method's inputs.
PEAK_BYTES_PER_SPOT_GENE = 20
PEAK_BYTES_PER_CELL_GENE = 16


def _log(msg: str) -> None:
    """Log to stderr so stdout stays clean for JSON output."""
    print(f"[stplus] {msg}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """Redirect stdout to stderr to capture training progress."""
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


# ---------------------------------------------------------------------------
# Device-agnostic monkey-patch for stPlus .cuda() calls
# ---------------------------------------------------------------------------


def _patch_stplus_for_device(device: str):
    """
    Monkey-patch stPlus internals to replace hardcoded .cuda() with
    device-agnostic .to(device) calls, so the caller -- not the hardware -- picks the device.

    stPlus source has .cuda() in stPlus.py (the main module) and model.py
    (``net = VAE(...).cuda()`` at model.py:116, ``train_x = train_x.cuda()`` at :135).
    This patch intercepts torch.Tensor.cuda and nn.Module.cuda and redirects both to ``device``.

    Applied **unconditionally**. It used to return early whenever ``torch.cuda.is_available()``,
    which meant that on any box with a GPU those ``.cuda()`` calls were the real thing and there was
    no flag, portal parameter or code path by which stPlus could be run on the CPU -- not to leave a
    shared card alone, not to work around an OOM, not to sidestep a driver problem. On a CPU-only
    host the resolved device is ``cpu`` and this behaves exactly as it always did; the difference is
    only visible where a GPU exists.
    """
    import torch

    _log(f"Applying .cuda() -> .to({device!r}) monkey-patch")

    def _tensor_cuda_patch(self, *args, **kwargs):
        """Redirect tensor.cuda() to tensor.to(device)."""
        return self.to(device)

    def _module_cuda_patch(self, *args, **kwargs):
        """Redirect module.cuda() to module.to(device)."""
        return self.to(device)

    torch.Tensor.cuda = _tensor_cuda_patch
    torch.nn.Module.cuda = _module_cuda_patch

    # stPlus calls torch.set_default_tensor_type('torch.cuda.FloatTensor'),
    # which forces every newly-created tensor onto CUDA and triggers a
    # "Found no NVIDIA driver" error on CPU-only hosts *before* any .cuda()
    # redirect can help. Neutralize it: force CPU float tensors regardless of
    # the (cuda) type stPlus requests.
    _orig_set_default = torch.set_default_tensor_type

    def _set_default_tensor_type_patch(_t=None):
        return _orig_set_default(torch.FloatTensor)

    torch.set_default_tensor_type = _set_default_tensor_type_patch

    _log(f"Monkey-patch applied: .cuda() -> .to({device!r}); set_default_tensor_type -> CPU")


def _load_batches_in_process() -> bool:
    """Make upstream's two ``DataLoader(..., num_workers=4)`` load their batches in this process.

    ``stPlus.model`` builds its training and validation loaders with four worker processes, and a
    worker hands every batch back through ``/dev/shm``. A container's ``/dev/shm`` is often small (64 MB
    on this project's box), and one batch is ``batch_size`` x model genes float32 -- about 10 MB on a
    whole-transcriptome slide at the default 512 -- so with the prefetched batches of four workers the
    run died with "DataLoader worker ... killed by signal: Bus error ... out of shared memory". The
    dataset is an in-memory tensor, so the workers only moved batches between processes; the batches
    and their order are the same with ``num_workers=0`` (the sampler runs in the main process either
    way). Returns True when upstream's module was patched.
    """
    model = sys.modules.get("stPlus.model")
    loader = getattr(model, "DataLoader", None) if model is not None else None
    if loader is None or getattr(loader, "_sog_in_process", False):
        return False

    def _in_process_loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        return loader(*args, **kwargs)

    _in_process_loader._sog_in_process = True
    model.DataLoader = _in_process_loader
    _log("stPlus's DataLoaders load batches in this process (num_workers=0 in place of upstream's 4)")
    return True


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


#: The suffixes read as HDF5 (an h5ad, or a 10x Cell Ranger / Space Ranger ``.h5`` matrix).
HDF5_SUFFIXES = (".h5ad", ".h5")


def _is_hdf5_input(path: str) -> bool:
    """True for a path the loader reads as HDF5 rather than as delimited text."""
    return Path(path).suffix.lower() in HDF5_SUFFIXES


def _hdf5_layout(path: str) -> str:
    """``'h5ad'`` or ``'10x_h5'``: which HDF5 layout ``path`` holds, read off its top-level groups.

    A 10x ``filtered_feature_bc_matrix.h5`` (the file every Space Ranger sample folder ships) keeps
    its matrix under ``matrix/`` (Cell Ranger >= 3) or under one group per genome holding
    ``barcodes``/``genes``/``data`` (Cell Ranger 2). An h5ad keeps ``X``/``obs``/``var`` at the top.
    Anything else is handed to anndata, which names what it could not read.
    """
    import h5py

    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        if {"X", "obs", "var"} <= keys:
            return "h5ad"
        if "matrix" in keys and isinstance(f["matrix"], h5py.Group):
            return "10x_h5"
        for key in keys:
            group = f[key]
            if isinstance(group, h5py.Group) and {"barcodes", "data", "indices", "indptr"} <= set(group.keys()):
                return "10x_h5"
    return "h5ad"


def _read_anndata(path: str):
    """``(adata, layout)`` for an HDF5 input: an h5ad through anndata, a 10x .h5 through scanpy.

    Both used to go to ``anndata.read_h5ad``, which dies on a 10x file with "__init__() got an
    unexpected keyword argument 'matrix'" -- naming neither the format nor what to pass instead.
    ``scanpy.read_10x_h5`` reads the gene-expression features with gene symbols as var_names (the
    Ensembl IDs stay in ``var['gene_ids']``) and the barcodes as obs_names.
    """
    layout = _hdf5_layout(path)
    if layout == "10x_h5":
        try:
            import scanpy as sc
        except ImportError as exc:
            raise ValueError(
                f"{path} is a 10x .h5 matrix, not an h5ad, and scanpy (which reads it) is not importable here "
                f"({exc}). Pass the sample's .h5ad instead."
            ) from exc
        return sc.read_10x_h5(path), layout
    import anndata

    return anndata.read_h5ad(path), layout


#: How each input layout is named in the payload.
INPUT_FORMATS = {"h5ad": "h5ad", "10x_h5": "10x .h5 matrix (scanpy.read_10x_h5)", "text": "delimited text"}


def _load_expression_data(
    path: str,
    label: str,
    renamed: dict | None = None,
    tissue: dict | None = None,
    source: dict | None = None,
):
    """
    Load expression data from h5ad, a 10x .h5 matrix, or a delimited text file.

    For h5ad / 10x .h5: extracts .X as a pandas DataFrame with var_names as columns
    and obs_names as index. ``source``, when given, gets ``format`` ('h5ad', '10x_h5' or 'text').

    ``tissue``, when a caller supplies a dict (the spatial side), applies the fleet's background rule
    (``worker_utils.keep_in_tissue``) right after reading: spots with ``obs['in_tissue'] == 0`` are
    left out before anything is densified, and the dict gets ``n_supplied`` / ``n_dropped``. A
    CELLxGENE Visium h5ad carries every array spot (56-70% background glass on the library's four),
    and stPlus used to train its autoencoder on them and write imputed rows for the glass. A 10x
    filtered matrix and a text table carry no flag; nothing is dropped from them.

    For text: the separator is read from the file's first line, not from its name, and the
    compression pandas can see through (.gz/.bz2/.xz) is handled for us.

    ``renamed``, when a caller supplies a dict, is filled in with how many gene symbols this load
    had to rename to make them unique -- gene axis only, because the columns of every table this
    worker writes are gene symbols while the rows are left exactly as the user supplied them. Each
    caller passes its own dict so the spatial and the scRNA-seq counts stay apart in the payload.

    Both branches invent names, by different mechanisms and with different spellings: anndata's
    ``var_names_make_unique`` writes ``GAPDH-1``, pandas' duplicate-column handling writes
    ``GAPDH.1``. Counting only the h5ad branch would make the payload's ``n_genes_renamed: 0``
    an affirmative false statement for a delimited-text input, so both are counted.

    Returns a pandas DataFrame (cells/spots x genes).
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse

    if _is_hdf5_input(path):
        adata, layout = _read_anndata(path)
        _log(f"Loading {label} from {INPUT_FORMATS[layout]}: {path}")
        if source is not None:
            source["format"] = layout
        if tissue is not None:
            adata, n_supplied, n_dropped = keep_in_tissue(adata, "spots")
            tissue.update({"n_supplied": int(n_supplied), "n_dropped": int(n_dropped)})
            if n_dropped:
                _log(f"  {label}: left out {n_dropped} of {n_supplied} spots with obs['in_tissue'] == 0 (background)")
        make_names_unique_and_report(adata, into=renamed, axes=("var",))

        # Extract dense matrix. stPlus takes pandas DataFrames, so the worker holds it densely; say
        # so with the numbers before allocating rather than be OOM-killed with no message at all.
        X = adata.X
        if scipy.sparse.issparse(X):
            itemsize = np.dtype(X.dtype).itemsize
            _check_dense_budget(
                int(X.shape[0]) * int(X.shape[1]) * (itemsize + (0 if X.dtype == np.float32 else 4)),
                f"the {label} matrix ({X.shape[0]} x {X.shape[1]}) as a dense float32 table",
                "stPlus takes dense pandas DataFrames, so every spot/cell and gene of this file is held in memory",
            )
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
        _log(f"  {label}: {df.shape[0]} samples x {df.shape[1]} genes")
        return df
    else:
        # `pd.read_csv` with no `sep` hardcodes a comma. A tab-delimited matrix parsed into a
        # single column, which `index_col=0` then consumed, so the caller got every spot and zero
        # genes -- with nothing raised. The empty frame reached the gene-overlap checks below and
        # the user was told to check their naming conventions, which were never the problem.
        # The h5ad branch above deliberately keeps reading raw `Path.suffix`: anndata is handed a
        # path and does not decompress one, so routing `x.h5ad.gz` by its inner suffix would swap
        # one bad error for a worse one.
        sep = sniff_tabular_sep(path)
        _log(f"Loading {label} from delimited text (sep={sep!r}): {path}")
        if source is not None:
            source["format"] = "text"
        df = pd.read_csv(path, index_col=0, sep=sep)
        if tissue is not None:
            tissue.update({"n_supplied": int(df.shape[0]), "n_dropped": 0})
        if renamed is not None:
            # pandas has already renamed any duplicate column labels by the time it returns
            # (GAPDH, GAPDH -> GAPDH, GAPDH.1), and since pandas 2.0 there is no switch to stop it.
            # Re-reading the header row as data recovers the symbols as written: pandas mangles
            # column labels, never cell values, and it sees through .gz/.bz2/.xz here as above.
            header = [str(v) for v in pd.read_csv(path, sep=sep, header=None, nrows=1).iloc[0].tolist()]
            start = max(0, len(header) - df.shape[1])  # skip whatever labelled the index column
            genes = header[start:]
            n_dupes = len(genes) - len(set(genes))
            renamed["n_genes_renamed"] = renamed.get("n_genes_renamed", 0) + n_dupes
            renamed.setdefault("n_cells_renamed", 0)
            if n_dupes:
                _log(f"  {label}: renamed {n_dupes} duplicate gene symbol(s) to make them unique")
        _log(f"  {label}: {df.shape[0]} samples x {df.shape[1]} genes")
        return df


def _load_gene_list(path: str) -> list[str]:
    """
    Load gene list from a text file (one gene per line) or a delimited table's first column.

    Optionally gzipped, bzipped or xz'd -- pandas decompresses on its own.
    """
    import pandas as pd

    # Both spellings are the same file: a gene name in the first field of every line. Branching on
    # `Path.suffix == ".csv"` and otherwise opening as plain text sent `genes.csv.gz` -- suffix
    # `.gz` -- down the text path, where the gzip framing surfaced as a UnicodeDecodeError naming a
    # byte offset rather than the compression. It also returned "GAPDH\tglyceraldehyde-3-phosphate
    # dehydrogenase" as a gene name for any annotated list, which then matched nothing in the
    # reference and was reported to the user as a naming-convention mismatch.
    sep = sniff_tabular_sep(path)
    df = pd.read_csv(path, header=None, sep=sep)
    genes = df.iloc[:, 0].astype(str).tolist()
    _log(f"Loaded {len(genes)} genes to impute from {path}")
    return genes


# ---------------------------------------------------------------------------
# Input scale, gene selection, memory
# ---------------------------------------------------------------------------


def _memory_available_bytes():
    """Memory this process can still allocate, or None when the platform cannot say.

    The fleet's one reader (``worker_utils.available_memory_bytes``): the smaller of MemAvailable and
    the room under the cgroup memory limit, page cache counted as reclaimable. This used to read
    MemAvailable alone -- the whole host's figure -- so in a memory-limited container every check below
    let through a matrix the cgroup then OOM-killed with no message. Kept under this name so the checks
    read it here.
    """
    return available_memory_bytes()


def _check_dense_budget(need_bytes: int, what: str, why: str, knobs: str = "") -> int:
    """Refuse, with the numbers, a dense allocation this machine reports it cannot hold."""
    available = _memory_available_bytes()
    if available is not None and need_bytes > available:
        gib = 1024.0**3
        raise MemoryError(
            f"Holding {what} needs about {need_bytes / gib:.1f} GiB and this machine reports "
            f"{available / gib:.1f} GiB available. {why}."
            + (f" {knobs}" if knobs else " Run on a machine with more memory; every spot and gene is kept.")
        )
    return need_bytes


def _scan_matrix(values, chunk: int = 5_000_000) -> dict:
    """One chunked pass over a dense matrix: is it raw counts, its range, and any non-finite entry.

    Raw counts are every value a finite, non-negative integer. Read in row blocks so the check never
    holds a second full-size temporary.
    """
    import numpy as np

    arr = np.asarray(values)
    info = {"looks_like_counts": arr.size > 0, "min": None, "max": None, "n_nonfinite": 0}
    if arr.size == 0:
        return info
    rows = max(1, chunk // max(1, arr.shape[1] if arr.ndim == 2 else 1))
    vmin = None
    vmax = None
    for start in range(0, arr.shape[0], rows):
        block = np.asarray(arr[start : start + rows], dtype=np.float64)
        finite = np.isfinite(block)
        n_bad = int(block.size - int(finite.sum()))
        info["n_nonfinite"] += n_bad
        good = block[finite] if n_bad else block
        if good.size == 0:
            continue
        bmin, bmax = float(good.min()), float(good.max())
        vmin = bmin if vmin is None else min(vmin, bmin)
        vmax = bmax if vmax is None else max(vmax, bmax)
        if info["looks_like_counts"] and (bmin < 0 or not np.all(np.equal(np.mod(good, 1), 0))):
            info["looks_like_counts"] = False
    if info["n_nonfinite"]:
        info["looks_like_counts"] = False
    info["min"], info["max"] = vmin, vmax
    return info


def _normalizes_in_place(df) -> bool:
    """True when ``df`` is one writable float32 block that ``_log_normalize`` can scale where it lies.

    Every h5ad frame the loader builds is (it wraps the float32 array it made), and so is a float32
    text table; a text table of integer or float64 counts is not, and needs a float32 copy.
    """
    import numpy as np

    if not all(dt == np.float32 for dt in df.dtypes):
        return False
    values = df.to_numpy()
    return bool(values.dtype == np.float32 and values.flags.writeable)


def _log_normalize(df, in_place: bool = False):
    """``log1p(normalize_total(target_sum=1e4))`` of every row, over all of the file's genes.

    Returns a float32 frame with the same index and columns. A row whose total is 0 stays all-zero.
    With ``in_place`` and a frame ``_normalizes_in_place`` accepts, the frame's own array is scaled
    and logged where it lies and the returned frame wraps that array: the caller's frame is changed
    and no second full-size matrix is allocated. Otherwise the input is left alone and a float32 copy
    is made -- a second full-size matrix beside the first, which the caller budgets for.
    """
    import numpy as np
    import pandas as pd

    if in_place and _normalizes_in_place(df):
        X = df.to_numpy()
    else:
        X = np.array(df.to_numpy(), dtype=np.float32, copy=True)
    totals = X.sum(axis=1, dtype=np.float64)
    n_empty = int((totals == 0).sum())
    scale = np.divide(TARGET_SUM, totals, out=np.zeros_like(totals), where=totals > 0).astype(np.float32)
    X *= scale[:, None]
    np.log1p(X, out=X)
    return pd.DataFrame(X, index=df.index, columns=df.columns, copy=False), n_empty


def _prepare_input_scale(df, label: str, mode: str, in_place: bool = False):
    """Put one matrix on the scale stPlus expects, and say what was done.

    Upstream's docstring asks for "normalized and logarithmized" spatial and reference data: the
    reconstruction loss is a sum of squared errors and the transfer is a cosine KNN, both of which
    raw counts hand to the few most highly expressed genes. ``auto`` normalises a matrix of raw
    counts and leaves anything else alone; ``always`` normalises regardless; ``never`` uses the
    matrix as given. Returns ``(frame, info, warnings)``.

    ``in_place`` says the caller does not need ``df`` as supplied afterwards, so it may be normalised
    where it lies. The normalisation used to copy every matrix while the caller still held the
    original: a second full-size float32 matrix no memory check covered (the loader budgets one copy,
    the training check runs after this), so a machine with less than twice the room was OOM-killed
    with no JSON. A copy that is still needed -- the caller keeps the original, or the frame is not
    float32 -- is now checked against the free memory before it is made.
    """
    scan = _scan_matrix(df.to_numpy())
    if scan["n_nonfinite"]:
        raise ValueError(
            f"The {label} matrix holds {scan['n_nonfinite']} NaN/inf value(s); stPlus trains on every entry and a "
            "single non-finite value makes the whole loss NaN. Fill or drop them before imputing."
        )
    warnings = []
    info = {
        "looked_like_counts": bool(scan["looks_like_counts"]),
        "value_min": scan["min"],
        "value_max": scan["max"],
        "n_rows_with_zero_total": 0,
    }
    apply = mode == "always" or (mode == "auto" and scan["looks_like_counts"])
    if apply:
        if scan["min"] is not None and scan["min"] < 0:
            raise ValueError(
                f"normalize={mode!r} cannot log-normalise the {label} matrix: it holds negative values "
                f"(min {scan['min']:.4g}), so it is already scaled or centred, not counts. Pass normalize='never' to "
                "use it as given."
            )
        can_scale_in_place = _normalizes_in_place(df)
        in_place = bool(in_place) and can_scale_in_place
        if not in_place:
            n_rows, n_cols = (int(v) for v in df.shape)
            reason = (
                "the matrix as supplied is still needed afterwards"
                if can_scale_in_place
                else "it is not held as one float32 matrix"
            )
            _check_dense_budget(
                n_rows * n_cols * 4,
                f"a log-normalised float32 copy of the {label} matrix ({n_rows} x {n_cols}) beside the matrix as loaded",
                f"stPlus is written for log-normalised input, and this matrix cannot be normalised where it lies "
                f"({reason})",
                "Pass normalize='never' with a matrix you have log-normalised yourself, or run on a machine with more "
                "memory; every spot, cell and gene is kept.",
            )
        df, n_empty = _log_normalize(df, in_place=in_place)
        info["normalization"] = LOG_NORMALIZED
        info["n_rows_with_zero_total"] = n_empty
        if n_empty:
            warnings.append(
                f"{n_empty} {label} row(s) have zero total expression; they stay all-zero after normalisation."
            )
        return df, info, warnings

    info["normalization"] = "none (used as given)"
    if scan["looks_like_counts"]:
        warnings.append(
            f"The {label} matrix holds raw integer counts and normalize='never' passed it to stPlus as given; "
            "stPlus is written for normalized, log-transformed input (its reconstruction loss is dominated by the "
            "most highly expressed genes on a count scale)."
        )
    elif scan["max"] is not None and scan["max"] > LOG_SCALE_CEILING:
        warnings.append(
            f"The {label} matrix is not integer counts but reaches {scan['max']:.4g}, which is not a log scale; "
            "stPlus expects normalized, log-transformed input. If it is normalized but not logged, pass "
            "normalize='always'."
        )
    if scan["min"] is not None and scan["min"] < 0:
        warnings.append(
            f"The {label} matrix holds negative values (min {scan['min']:.4g}); stPlus's decoder ends in a ReLU "
            "and cannot reconstruct negative expression."
        )
    return df, info, warnings


def _top_variable_genes(scrna_df, other_genes, k: int, chunk_cols: int = 512) -> list:
    """The ``k`` highest-variance reference genes among ``other_genes`` -- upstream's selection.

    stPlus's ``select_top_variable_genes`` is ``np.argpartition(np.var(mtx, axis=0), -k)[-k:]`` over
    ``np.setdiff1d(reference genes, shared + targets)``. It raises when ``0 < len(other) < k``, and
    it takes a full copy of the reference to compute a per-gene variance; this computes the same
    variances a block of columns at a time and makes the same argpartition.
    """
    import numpy as np

    other = np.asarray(other_genes, dtype=object)
    k = min(int(k), int(other.size))
    if k <= 0:
        return []
    var = np.concatenate(
        [
            np.var(scrna_df[list(other[start : start + chunk_cols])].to_numpy(), axis=0)
            for start in range(0, other.size, chunk_cols)
        ]
    )
    ind = np.argpartition(var, -k)[-k:]
    return [str(g) for g in other[ind]]


class _TrainingLog:
    """A stdout stand-in that passes every write through and reads stPlus's epoch lines.

    Upstream trains ``for e in range(max_epoch_num)`` and breaks once the relative loss change is
    below ``converge_ratio``; the only record of how far it got is the ``\\t[<epoch>] recon_loss:``
    line it prints per epoch.
    """

    _EPOCH = re.compile(r"^\s*\[(\d+)\]\s+recon_loss:")

    def __init__(self, stream):
        self._stream = stream
        self._pending = ""
        self.epochs_run = 0

    def write(self, text):
        self._stream.write(text)
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            match = self._EPOCH.match(line)
            if match:
                self.epochs_run = max(self.epochs_run, int(match.group(1)))
        return len(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        if name == "_stream":  # only reachable before __init__ ran; never recurse
            raise AttributeError(name)
        return getattr(self._stream, name)


@contextlib.contextmanager
def _tap_stdout():
    """Route stdout through a :class:`_TrainingLog` for the duration of the ``stPlus()`` call."""
    tap = _TrainingLog(sys.stdout)
    old = sys.stdout
    sys.stdout = tap
    try:
        yield tap
    finally:
        sys.stdout = old


def _write_csv_atomic(df, path) -> None:
    """``<path>.partial`` then ``os.replace``: a reader never sees a half-written table."""
    tmp = f"{path}.partial"
    df.to_csv(tmp)
    os.replace(tmp, str(path))


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _run_stplus(
    spatial_path: str,
    scrna_path: str,
    genes_path: str,
    output_dir: str = default_output_dir(),
    n_neighbors: int = 50,
    n_epochs: int = 200,
    batch_size: int = 512,
    device: str = "auto",
    normalize: str = "auto",
    top_k: int = TOP_K_DEFAULT,
) -> dict[str, Any]:
    """Core stPlus pipeline. Returns a WorkerOutput dict."""

    import numpy as np
    import pandas as pd

    mode = str(normalize).strip().lower()
    if mode not in NORMALIZE_CHOICES:
        raise ValueError(unsupported_choice_msg("normalize", normalize, NORMALIZE_CHOICES))
    if int(n_epochs) < 1:
        raise ValueError(
            f"n_epochs={n_epochs}: stPlus needs at least one training epoch to save a model to predict with."
        )
    if int(batch_size) < 1:
        raise ValueError(f"batch_size={batch_size} must be at least 1.")
    if int(top_k) < 0:
        raise ValueError(f"top_k={top_k} must be 0 (no highly variable reference genes appended) or more.")
    if int(n_neighbors) < 2:
        raise ValueError(
            f"n_neighbors={n_neighbors}: stPlus weights the neighbours by 1 - d/sum(d) divided by (n - 1), which is "
            "0/0 for a single neighbour, so every imputed value would be NaN. Use n_neighbors >= 2."
        )

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    warnings: list = []

    # ---- 1. Load data ----
    # One accumulator per file: they are independent objects and the payload reports each one's
    # renames under its own key, so a reader can tell which side the invented symbols came from.
    renamed_st: dict = {}
    renamed_sc: dict = {}
    # Background spots (obs['in_tissue'] == 0) are left out of the spatial side as it is read.
    tissue_st: dict = {}
    source_st: dict = {}
    source_sc: dict = {}
    spatial_df = _load_expression_data(spatial_path, "spatial", renamed=renamed_st, tissue=tissue_st, source=source_st)
    scrna_df = _load_expression_data(scrna_path, "scRNA-seq", renamed=renamed_sc, source=source_sc)
    genes_listed = _load_gene_list(genes_path)
    # A gene named twice is one gene: upstream selects it twice from a reference that then has it
    # twice, and its per-spot prediction no longer fits the one slot it was given.
    genes_to_impute = list(dict.fromkeys(genes_listed))
    n_duplicate_genes = len(genes_listed) - len(genes_to_impute)
    if n_duplicate_genes:
        warnings.append(
            f"{n_duplicate_genes} gene(s) were listed more than once in the gene list; each is imputed once."
        )

    # ---- 2. Validate genes ----
    # Genes to impute must be in scRNA-seq data
    scrna_genes = set(scrna_df.columns)
    valid_impute_genes = [g for g in genes_to_impute if g in scrna_genes]
    missing_genes = [g for g in genes_to_impute if g not in scrna_genes]

    if missing_genes:
        _log(
            f"WARNING: {len(missing_genes)} genes not found in scRNA-seq: "
            f"{missing_genes[:10]}{'...' if len(missing_genes) > 10 else ''}"
        )

    if not valid_impute_genes:
        raise ValueError(
            f"None of the {len(genes_to_impute)} requested imputation genes "
            f"were found in the scRNA-seq reference. Check gene naming "
            f"conventions (symbols vs Ensembl IDs)."
        )

    _log(f"Valid genes to impute: {len(valid_impute_genes)}/{len(genes_to_impute)}")

    # ---- 3. Find shared genes (anchors) ----
    shared_genes = sorted(set(spatial_df.columns) & set(scrna_df.columns))
    _log(f"Shared anchor genes between spatial and scRNA: {len(shared_genes)}")

    # Remove imputation targets from the anchor set (they should not overlap)
    impute_set = set(valid_impute_genes)
    anchor_genes = [g for g in shared_genes if g not in impute_set]
    spatial_genes = set(spatial_df.columns)
    also_measured = [g for g in valid_impute_genes if g in spatial_genes]
    _log(f"Anchor genes (shared, excluding imputation targets): {len(anchor_genes)}")

    # The anchors are what stPlus aligns on, so they are what the minimum applies to: counting the
    # shared genes before the targets are set aside let a run through with no anchor at all.
    if len(anchor_genes) < MIN_ANCHOR_GENES:
        raise ValueError(
            f"Only {len(anchor_genes)} anchor genes: {len(shared_genes)} genes are shared between the spatial and "
            f"scRNA data and {len(shared_genes) - len(anchor_genes)} of them are in the list to impute, which "
            f"sets them aside. stPlus needs at least {MIN_ANCHOR_GENES} measured genes as anchors. Check that gene "
            "naming conventions match, or impute fewer of the measured genes."
        )

    n_cells = int(scrna_df.shape[0])
    if int(n_neighbors) > n_cells:
        raise ValueError(
            f"n_neighbors={n_neighbors} is more than the {n_cells} cells in the scRNA-seq reference; stPlus predicts "
            "each spot from its n_neighbors nearest reference cells. Lower n_neighbors."
        )

    # ---- 4. Put both matrices on the scale stPlus is written for ----
    spatial_is_h5ad = _is_hdf5_input(spatial_path)
    spatial_as_supplied = None if spatial_is_h5ad else spatial_df
    # Normalised where it lies whenever nothing reads the matrix as supplied again: the reference
    # never, the spatial matrix unless it came from text (stplus_spatial.h5ad is then built from it;
    # an h5ad is re-read from its file instead).
    spatial_df, norm_st, warn_st = _prepare_input_scale(spatial_df, "spatial", mode, in_place=spatial_is_h5ad)
    scrna_df, norm_sc, warn_sc = _prepare_input_scale(scrna_df, "scRNA-seq reference", mode, in_place=True)
    warnings.extend(warn_st + warn_sc)
    _log(f"Input scale: spatial {norm_st['normalization']}; reference {norm_sc['normalization']}")

    # ---- 5. The highly variable reference genes stPlus appends ----
    # Upstream appends the top_k most variable reference genes that are neither anchors nor targets.
    # Handing it only anchors + targets made that set empty on every run, and handing it the whole
    # reference crashes upstream whenever 0 < (other genes) < top_k. The worker makes upstream's
    # selection itself and passes exactly those genes, with top_k = their number, so upstream's own
    # selection step keeps all of them.
    reserved = np.array(anchor_genes + valid_impute_genes, dtype=object)
    other_genes = np.setdiff1d(np.asarray(scrna_df.columns.values, dtype=object), reserved)
    hvg_genes = _top_variable_genes(scrna_df, other_genes, int(top_k))
    upstream_top_k = len(hvg_genes) if hvg_genes else TOP_K_DEFAULT
    _log(f"Highly variable reference genes appended: {len(hvg_genes)} of {len(other_genes)} (top_k={top_k})")

    n_spots = int(spatial_df.shape[0])
    width = len(anchor_genes) + len(valid_impute_genes) + len(hvg_genes)
    _check_dense_budget(
        width * (PEAK_BYTES_PER_SPOT_GENE * n_spots + PEAK_BYTES_PER_CELL_GENE * n_cells),
        f"stPlus's training matrix ({n_spots} spots + {n_cells} cells x {width} genes: {len(anchor_genes)} anchors, "
        f"{len(valid_impute_genes)} to impute, {len(hvg_genes)} highly variable reference genes)",
        "stPlus trains its autoencoder on this dense matrix and holds float64 copies of it while building it",
        "Lower top_k (the highly variable reference genes appended) or impute fewer genes; every spot and cell "
        "is kept.",
    )

    # ---- 6. Apply the device patch and run stPlus ----
    resolved_device = resolve_compute(device).device
    _log(f"Using device: {resolved_device}")
    _patch_stplus_for_device(resolved_device)

    # stPlus/model.py forces `torch.set_default_tensor_type('torch.cuda.FloatTensor')`
    # at module level (runs during the import above). On a CPU-only host that makes
    # every later tensor allocation try to init CUDA and crash. Reset the default to
    # CPU now that the import has happened (the .cuda() redirect handles the rest).
    import torch as _torch
    from stPlus import stPlus

    # Unconditionally, not `if not cuda.is_available()`. The guard used to skip the reset on exactly
    # the boxes where stPlus's module-level `set_default_tensor_type('torch.cuda.FloatTensor')`
    # actually takes effect, so a GPU host was left with a CUDA default tensor type for the rest of
    # the process -- every later bare `torch.zeros`/`torch.Tensor`, in this worker and in anything
    # it imports, silently allocating on GPU 0. On a CPU-only host this call is what it always was.
    _torch.set_default_tensor_type("torch.FloatTensor")
    loads_in_process = _load_batches_in_process()

    _log(
        f"Running stPlus imputation (n_neighbors={n_neighbors}, max_epochs={n_epochs}, batch_size={batch_size}, "
        f"{len(hvg_genes)} highly variable reference genes appended)..."
    )

    # stPlus expects:
    #   spatial_df: DataFrame (spots x genes) -- the anchors
    #   scrna_df: DataFrame (cells x genes) -- anchors, targets and the genes it may add as HVGs
    #   genes_to_predict: list of gene names to predict
    # It returns a DataFrame of imputed values (spots x imputed_genes) with a 0..n-1 RangeIndex.
    #
    # Checkpoints: upstream saves the t_min lowest-loss epochs to `<save_path_prefix>-5min<i>.pt`
    # (default prefix './stPlus', i.e. the MCP server's working directory) and, to predict, loads
    # every such file that EXISTS -- including another run's, of another width (a load_state_dict
    # crash) or of the same width (silently averaged in). A directory made fresh for this run is
    # the only place where "exists" means "this run wrote it".
    ckpt_dir = tempfile.mkdtemp(prefix=".stplus_checkpoints_", dir=str(output_path))
    save_path_prefix = os.path.join(ckpt_dir, "stPlus")
    try:
        with _tap_stdout() as training_log:
            stPlus_result = stPlus(
                spatial_df[anchor_genes],
                scrna_df[anchor_genes + valid_impute_genes + hvg_genes],
                valid_impute_genes,
                save_path_prefix=save_path_prefix,
                top_k=upstream_top_k,
                t_min=T_MIN,
                converge_ratio=CONVERGE_RATIO,
                verbose=True,
                n_neighbors=n_neighbors,
                max_epoch_num=n_epochs,
                batch_size=batch_size,
            )
        n_checkpoints = len([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    epochs_run = training_log.epochs_run or None

    if n_checkpoints == 0:
        raise RuntimeError(
            "stPlus saved no model to predict with: no epoch's training loss went below its starting bound (5e20), which "
            "happens when the loss is NaN, inf or astronomically large. Check the inputs for extreme values and the scale "
            f"(spatial: {norm_st['normalization']}; reference: {norm_sc['normalization']})."
        )

    # Upstream's frame carries a 0..n-1 RangeIndex, not the spot barcodes. Kept as is, the imputed
    # CSV was indexed 0..n-1 and the combined table aligned barcode rows against integer rows: 2n
    # rows, measured genes NaN on one half and imputed genes NaN on the other. The rows are
    # upstream's spatial rows in the order given, so they take the spatial index positionally.
    if isinstance(stPlus_result, pd.DataFrame):
        missing_cols = [g for g in valid_impute_genes if g not in stPlus_result.columns]
        if missing_cols:
            raise RuntimeError(
                f"stPlus returned no column for {len(missing_cols)} requested gene(s): {missing_cols[:10]}"
            )
        values = stPlus_result[valid_impute_genes].to_numpy()
    else:
        values = np.asarray(stPlus_result)
    if values.shape != (n_spots, len(valid_impute_genes)):
        raise RuntimeError(
            f"stPlus returned a {values.shape} result for {n_spots} spots x {len(valid_impute_genes)} genes; "
            "it returns 0 when a requested gene is absent from the reference it was given."
        )
    imputed_df = pd.DataFrame(values, index=spatial_df.index, columns=valid_impute_genes)

    n_nonfinite = int(values.size - int(np.isfinite(values).sum()))
    if n_nonfinite:
        warnings.append(
            f"{n_nonfinite} of {values.size} imputed values are NaN/inf: upstream weights a spot's neighbours by "
            "1 - d/sum(d) over those closer than cosine distance 1, divided by (their number - 1), which is 0/0 "
            "when only one neighbour is that close."
        )

    _log(f"Imputation complete: {imputed_df.shape[0]} spots x {imputed_df.shape[1]} genes")

    # What the numbers are: a KNN-weighted average of the reference's values for each gene, so they
    # sit on the scale the reference was handed to stPlus on.
    value_scale = (
        f"{LOG_NORMALIZED} of the scRNA-seq reference"
        if norm_sc["normalization"] == LOG_NORMALIZED
        else "the scRNA-seq reference's values as supplied"
    )

    # ---- 7. Save outputs ----
    # CSV with imputed gene expression
    imputed_csv = output_path / "stplus_imputed.csv"
    _write_csv_atomic(imputed_df, imputed_csv)
    _log(f"Saved imputed expression to {imputed_csv}")

    # Combined table: the measured genes as stPlus saw them beside the imputed genes, one scale.
    combined_csv = output_path / "stplus_combined.csv"
    try:
        combined_df = pd.concat([spatial_df, imputed_df], axis=1)
        if combined_df.shape[0] != n_spots:
            raise RuntimeError(f"the combined table has {combined_df.shape[0]} rows for {n_spots} spots")
        _write_csv_atomic(combined_df, combined_csv)
        del combined_df
        _log(f"Saved combined (original + imputed) to {combined_csv}")
    except Exception as e:
        _log(f"WARNING: Could not save combined CSV: {e}")
        warnings.append(f"stplus_combined.csv was not written: {e}")
        combined_csv = None
    if also_measured and combined_csv is not None:
        warnings.append(
            f"{len(also_measured)} imputed gene(s) are also measured in the spatial data "
            f"({', '.join(also_measured[:5])}{'...' if len(also_measured) > 5 else ''}); stplus_combined.csv carries "
            "each of them twice, the measured column first and the imputed column second."
        )

    # Annotated h5ad: original spatial + imputed genes in obsm
    try:
        import anndata

        # Reload original spatial data if h5ad (or a 10x .h5): the same reader as the first load,
        # and the same background rule, so the rows line up with the imputed ones.
        if spatial_is_h5ad:
            if source_st.get("format") == "10x_h5":
                adata_spatial, _ = _read_anndata(spatial_path)
            else:
                adata_spatial = anndata.read_h5ad(spatial_path)
            adata_spatial, _, _ = keep_in_tissue(adata_spatial, "spots")
            # Not counted: this re-reads the file `_load_expression_data` already loaded above, so
            # `renamed_st` already holds these renames. Counting them here would double the number.
            adata_spatial.var_names_make_unique()
            if list(map(str, adata_spatial.obs_names)) != list(map(str, imputed_df.index)):
                raise RuntimeError(
                    f"re-reading {spatial_path} gave {adata_spatial.n_obs} spots that do not line up with the "
                    f"{imputed_df.shape[0]} imputed rows"
                )
        else:
            adata_spatial = anndata.AnnData(
                X=spatial_as_supplied.values,
                obs=pd.DataFrame(index=spatial_as_supplied.index),
                var=pd.DataFrame(index=spatial_as_supplied.columns),
            )

        # Store imputed expression in obsm; its columns are named in uns, since obsm cannot carry them.
        adata_spatial.obsm["stplus_imputed"] = imputed_df.values
        adata_spatial.uns["stplus_imputed_genes"] = [str(g) for g in imputed_df.columns]
        adata_spatial.uns["stplus_imputed_scale"] = value_scale

        # Save annotated h5ad
        out_h5ad = output_path / "stplus_spatial.h5ad"
        adata_spatial.write_h5ad(f"{out_h5ad}.partial")
        os.replace(f"{out_h5ad}.partial", str(out_h5ad))
        _log(f"Saved annotated h5ad to {out_h5ad}")
    except Exception as e:
        _log(f"WARNING: Could not save h5ad output: {e}")
        warnings.append(f"stplus_spatial.h5ad was not written: {e}")
        out_h5ad = None

    # ---- 8. Build output ----
    out = WorkerOutput("stplus", task="gene_imputation")
    n_spots_supplied = int(tissue_st.get("n_supplied", n_spots))
    n_spots_off_tissue = int(tissue_st.get("n_dropped", 0))
    out.set_data(
        n_spots=n_spots,
        n_spots_supplied=n_spots_supplied,
        n_spatial_genes=int(spatial_df.shape[1]),
        n_scrna_genes=int(scrna_df.shape[1]),
        n_scrna_cells=n_cells,
        n_anchor_genes=len(anchor_genes),
    )

    output_files = {"imputed_csv": str(imputed_csv)}
    if out_h5ad is not None:
        output_files["spatial_h5ad"] = str(out_h5ad)
    if combined_csv is not None:
        output_files["combined_csv"] = str(combined_csv)
    out.add_output_files(output_files)

    stopped_early = bool(epochs_run is not None and epochs_run < int(n_epochs))
    record_method(out, "stPlus (upstream stPlus.stPlus)")
    out.add_params(
        {
            "n_neighbors": n_neighbors,
            # n_epochs is the maximum the caller allowed; epochs_run is how many stPlus trained.
            "n_epochs": n_epochs,
            "max_epochs": n_epochs,
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "converge_ratio": CONVERGE_RATIO,
            "t_min": T_MIN,
            "n_checkpoints_averaged": n_checkpoints,
            "batch_size": batch_size,
            "device": resolved_device,
            "normalize": mode,
            "normalization_spatial": norm_st["normalization"],
            "normalization_scrna": norm_sc["normalization"],
            "spatial_input_looked_like_counts": norm_st["looked_like_counts"],
            "scrna_input_looked_like_counts": norm_sc["looked_like_counts"],
            "imputed_value_scale": value_scale,
            "top_k": int(top_k),
            "n_hvg_genes_added": len(hvg_genes),
            "n_reference_genes_eligible_as_hvg": int(len(other_genes)),
            "n_model_genes": width,
            "n_genes_requested": len(genes_to_impute),
            "n_genes_listed_twice": n_duplicate_genes,
            "n_genes_imputed": len(valid_impute_genes),
            "n_genes_missing": len(missing_genes),
            "n_genes_imputed_also_measured": len(also_measured),
            "n_imputed_values_nonfinite": n_nonfinite,
            "spatial_input_format": INPUT_FORMATS[source_st.get("format", "text")],
            # Upstream's DataLoaders ask for 4 worker processes; they ran with 0 (see _load_batches_in_process).
            "data_loader_workers": 0 if loads_in_process else "upstream default (4)",
            "scrna_input_format": INPUT_FORMATS[source_sc.get("format", "text")],
        }
    )
    # params.in_tissue_filter and a warning, when background spots were left out.
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    out.add_params(identifier_rename_params(renamed_st))
    out.add_params(identifier_rename_params(renamed_sc, suffix="sc"))
    out.add_warnings(warnings)

    # Summary statistics on imputed values
    mean_expr = imputed_df.mean(axis=0)
    top_imputed = mean_expr.sort_values(ascending=False).head(10)
    finite_values = values[np.isfinite(values)]
    out.set_summary(
        n_imputed_genes=len(valid_impute_genes),
        top_imputed_genes=list(top_imputed.index),
        top_imputed_mean_expr=[round(float(v), 4) for v in top_imputed.values],
        imputed_value_range=[
            round(float(finite_values.min()), 4) if finite_values.size else None,
            round(float(finite_values.max()), 4) if finite_values.size else None,
        ],
        missing_genes=missing_genes[:20] if missing_genes else [],
        epochs_run=epochs_run,
    )

    epochs_text = (
        f"trained {epochs_run} of at most {n_epochs} epochs"
        + (f" (stopped once the loss changed by less than {CONVERGE_RATIO:g})" if stopped_early else "")
        if epochs_run is not None
        else f"trained for at most {n_epochs} epochs (the epoch count was not reported)"
    )
    tissue_text = (
        f" ({n_spots_off_tissue} of the {n_spots_supplied} spots supplied have obs['in_tissue'] == 0 -- background "
        f"outside the tissue -- and were left out of training and imputation)"
        if n_spots_off_tissue
        else ""
    )
    out.set_analysis(
        f"stPlus imputed {len(valid_impute_genes)} genes across "
        f"{n_spots} spatial spots{tissue_text} using {len(anchor_genes)} "
        f"anchor genes from scRNA-seq reference ({n_cells} cells), plus {len(hvg_genes)} highly variable "
        f"reference genes (top_k={top_k}); it {epochs_text} and averaged {n_checkpoints} lowest-loss "
        f"checkpoint(s). Spatial input: {norm_st['normalization']}; reference: {norm_sc['normalization']}. "
        f"Imputed values are on the scale of {value_scale}. "
        f"Top imputed genes by mean expression: "
        f"{', '.join(list(top_imputed.index)[:5])}."
        + identifier_rename_note(renamed_st, subject="spatial data")
        + identifier_rename_note(renamed_sc, subject="scRNA-seq reference")
    )

    return out.to_dict()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="stPlus worker: spatial gene imputation via semi-supervised autoencoder"
    )
    parser.add_argument("--spatial-path", required=True, help="Path to spatial data (.h5ad or .csv)")
    parser.add_argument("--scrna-path", required=True, help="Path to scRNA-seq reference (.h5ad or .csv)")
    parser.add_argument("--genes-path", required=True, help="Path to gene list file (.txt or .csv)")
    parser.add_argument("--output-dir", default=default_output_dir(), help="Output directory")
    parser.add_argument("--n-neighbors", type=int, default=50, help="Number of nearest neighbors (default: 50)")
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=200,
        help="Maximum training epochs; stPlus stops earlier once the loss converges (default: 200)",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (default: 512)")
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device: 'auto', 'cpu', 'gpu'/'cuda', or 'cuda:N' for a specific GPU.",
    )
    parser.add_argument(
        "--normalize",
        default="auto",
        help="Input scale: 'auto' (log1p(normalize_total(1e4)) a matrix of raw counts, use anything else as "
        "given), 'always', or 'never' (default: auto)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_DEFAULT,
        help=f"Highly variable reference genes stPlus appends to the model; 0 appends none (default: {TOP_K_DEFAULT})",
    )

    args = parser.parse_args()

    preflight_check(
        inputs={
            "spatial_data": args.spatial_path,
            "scrna_data": args.scrna_path,
            "genes_to_impute": args.genes_path,
        },
        output_dir=args.output_dir,
        packages=["torch", "stPlus"],
    )

    error_info = None
    error_exc = None
    with _redirect_stdout_to_stderr():
        try:
            result = _run_stplus(
                spatial_path=args.spatial_path,
                scrna_path=args.scrna_path,
                genes_path=args.genes_path,
                output_dir=args.output_dir,
                n_neighbors=args.n_neighbors,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                device=args.device,
                normalize=args.normalize,
                top_k=args.top_k,
            )
        except Exception as e:
            _log(f"ERROR: {e}")
            traceback.print_exc(file=sys.stderr)
            error_info = str(e)
            error_exc = e

    # stdout: JSON only (must be outside redirect block)
    if error_info is not None:
        WorkerOutput.emit_error("stplus", error_info, task="gene_imputation", exc=error_exc)
        sys.exit(1)

    print(json.dumps(result, default=str))
    sys.stdout.flush()


if __name__ == "__main__":
    _cli_main()
