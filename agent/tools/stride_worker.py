#!/usr/bin/env python
# /workspace/SpatialOmicsGym/tools/stride_worker.py
from __future__ import annotations  # allow PEP 585/604 annotations (list[int] | None) on the Py3.8 stride env

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from worker_utils import (
    WorkerOutput,
    build_deconv_analysis,
    describe_reduction,
    env_bin,
    id_mismatch_msg,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    record_in_tissue,
    record_method,
    unsupported_choice_msg,
)
from worker_utils import drop_unlabeled as _split_unlabeled  # the parameter of the same name shadows it

METHOD_NAME = "STRIDE deconvolve (LDA topic model trained on the scRNA-seq reference)"

# The small-panel variance ranking materialises at most this many bytes of the count matrix at once.
_VARIANCE_BLOCK_BYTES = 256 * 1024 * 1024

# What STRIDE trains on when no gene list is in force (Deconvolution.py: MarkerFind(..., ntop=200)).
_STRIDE_MARKERS = "STRIDE's own marker search (MarkerFind: top 200 Wilcoxon markers per reference cell type)"


def log(msg: str) -> None:
    """Print log messages to stderr (SpatialOmicsLab-friendly)."""
    print(f"[stride-worker] {msg}", file=sys.stderr, flush=True)


def get_count_matrix(adata):
    """
    Prefer raw counts if available, otherwise use X.

    Priority:
    1. adata.layers["counts"] if present
    2. adata.raw.X if present, realigned to adata.var_names
    3. adata.X

    The realignment is not optional. AnnData deliberately leaves ``.raw`` alone when the object is
    sliced on the var axis, so after ``adata[:, common]`` the raw matrix still carries every original
    gene in the original order. Returned as-is it is a matrix whose columns no longer correspond to
    ``adata.var_names`` -- and the caller labels its output rows from exactly that list, so every
    row comes out attributed to the wrong gene without anything raising.
    """
    return count_matrix_and_source(adata)[0]


def count_matrix_and_source(adata):
    """``(matrix, source)``: what :func:`get_count_matrix` returns, and where it came from.

    ``source`` is ``"layers['counts']"``, ``"raw.X"`` or ``"X"`` and is published as
    ``params.expression_source`` / ``params.expression_source_sc``: the priority below picks a
    counts matrix on the caller's behalf, so the payload says which one STRIDE was handed.
    """
    if "counts" in adata.layers:
        return adata.layers["counts"], "layers['counts']"
    if adata.raw is not None:
        names = list(adata.var_names)
        if list(adata.raw.var_names) == names:
            return adata.raw.X, "raw.X"
        absent = [g for g in names if g not in set(adata.raw.var_names)]
        if absent:
            # .raw normally holds a superset, but a reference assembled some other way may not.
            # Guessing an alignment would put us back where we started; X is the matrix that is
            # certain to match var_names.
            log(
                f"{len(absent)} of {len(names)} genes are absent from .raw "
                f"(e.g. {', '.join(absent[:3])}); using X rather than a raw matrix that cannot be aligned."
            )
            return adata.X, "X"
        log(f"Realigning .raw ({adata.raw.n_vars} genes) to the {len(names)} genes in var_names.")
        return adata.raw[:, names].X, "raw.X"
    return adata.X, "X"


def _rounded_count_csr(X, what: str = "count matrix", report: dict | None = None):
    """``X`` (obs x genes) as a CSR float32 matrix of counts rounded to the nearest integer.

    Sparse input stays sparse, and dense input is converted to CSR, whose size is set by the
    non-zeros rather than by obs x genes. The caller's matrix is never modified: the rounded values
    are a new array, and the index arrays are copied only when rounding created zeros that have to
    be dropped. Rounding is the text writer's: ``np.rint`` and Python's ``round()`` both round half
    to even, on the input's own precision, before the float32 cast STRIDE applies to every matrix.

    A value STRIDE's topic model cannot read as a count stops the run here with its count: NaN and
    inf (the text writer died on these with a bare ``cannot convert float NaN to integer``) and
    negative values (a scaled matrix, not counts).

    Non-negative values that are not whole numbers (a normalised or log-transformed matrix) are
    rounded, as they always were, but no longer in silence: ``report`` (when given) receives
    ``n_values``, ``n_non_integer`` and an ``example`` so the caller can warn that STRIDE was handed
    rounded expression rather than counts.
    """
    counts = sparse.csr_matrix(X) if sparse.issparse(X) else sparse.csr_matrix(np.asarray(X))
    if not counts.has_canonical_format:
        # Duplicate entries add up, as they would in the dense matrix the text form wrote out.
        counts = counts.copy()
        counts.sum_duplicates()
    data = counts.data
    n_non_integer = 0
    example = None
    if not np.issubdtype(data.dtype, np.integer):
        n_bad = int(np.count_nonzero(~np.isfinite(data)))
        if n_bad:
            raise ValueError(
                f"the {what} holds {n_bad} NaN/inf values; STRIDE needs raw counts (in X, layers['counts'] or .raw)."
            )
        rounded_values = np.rint(data)
        off = rounded_values != data
        n_non_integer = int(np.count_nonzero(off))
        if n_non_integer:
            example = float(data[np.flatnonzero(off)[0]])
        data = rounded_values
    if report is not None:
        report.update({"n_values": int(data.size), "n_non_integer": n_non_integer, "example": example})
    n_negative = int(np.count_nonzero(data < 0))
    if n_negative:
        raise ValueError(
            f"the {what} holds {n_negative} negative values, so it is not a count matrix (a scaled or "
            "centred matrix?); STRIDE needs raw counts (in X, layers['counts'] or .raw)."
        )
    rounded = np.array(data, dtype=np.float32)  # always a new array: the caller's values stay as they were
    if np.any(rounded == 0):
        out = sparse.csr_matrix((rounded, counts.indices.copy(), counts.indptr.copy()), shape=counts.shape)
        out.eliminate_zeros()
        return out
    return sparse.csr_matrix((rounded, counts.indices, counts.indptr), shape=counts.shape)


def _encode_names(names) -> np.ndarray:
    """Fixed-width UTF-8 bytes, the string form 10x files use and STRIDE's reader decodes."""
    encoded = [str(n).encode("utf-8") for n in names]
    width = max([len(b) for b in encoded] + [1])
    return np.array(encoded, dtype=f"S{width}")


def _write_10x_h5(counts, obs_names, var_names, out_path: str) -> None:
    """Write an obs x genes CSR matrix in the 10x HDF5 layout STRIDE's ``read_10X_h5`` loads.

    10x stores genes x barcodes as CSC. The CSC arrays of that matrix *are* the CSR arrays of its
    obs x genes transpose, so ``data``/``indices``/``indptr`` are written as they are: no transpose,
    no densifying. Written to ``<path>.partial`` and moved into place, so an interrupted write never
    leaves a truncated file under the name STRIDE reads.
    """
    n_obs, n_genes = counts.shape
    partial = out_path + ".partial"
    try:
        with h5py.File(partial, "w") as f:
            mat = f.create_group("matrix")
            mat.create_dataset("barcodes", data=_encode_names(obs_names))
            mat.create_dataset("data", data=counts.data)
            mat.create_dataset("indices", data=counts.indices)
            mat.create_dataset("indptr", data=counts.indptr)
            mat.create_dataset("shape", data=np.array([n_genes, n_obs], dtype=np.int64))
            features = mat.create_group("features")
            names = _encode_names(var_names)
            features.create_dataset("id", data=names)
            features.create_dataset("name", data=names)
            features.create_dataset("feature_type", data=np.array([b"Gene Expression"] * n_genes))
        os.replace(partial, out_path)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise


def write_gene_by_obs_counts(X, obs_names, var_names, out_path: str, what: str = "count matrix", report=None):
    """
    Write a gene x obs count matrix in a form the STRIDE CLI reads; the extension picks the form.

    ``.h5``: the sparse 10x HDF5 layout (``matrix/{data,indices,indptr,shape,barcodes}`` and
    ``matrix/features/{id,name}``) that STRIDE's ``read_10X_h5`` loads straight into a CSC matrix.
    This is what ``run_stride_deconvolution`` stages.

    Anything else: tab-delimited text. Rows: genes. Columns: observations (cells or spots). First
    row: header "gene" + obs_names. STRIDE's text reader (``utility/IO.read_count``) takes the file
    in with ``readlines()`` and builds one Python float per entry of the *dense* matrix before it
    converts to sparse -- about 9e9 objects for a 507,684-bin Visium HD slide on an 18k-gene panel --
    so the worker no longer stages this form; it stays for a caller who wants a readable table.

    Both forms round values to the nearest integer. The ``.h5`` form returns the rounded obs x
    genes CSR matrix it wrote (the counts STRIDE reads, used for its scale factor and for the spots
    with no count) and fills ``report`` as ``_rounded_count_csr`` describes; the text form returns
    None.
    """
    log(f"Writing count matrix to {out_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # The matrix and its gene labels arrive as independent arguments and are trusted to correspond.
    # That trust is what let a .raw/var_names mismatch mislabel every row of every STRIDE run in
    # silence: the label list is the shorter of the two, so indexing never overruns and no error
    # surfaces. A disagreement here is always a bug upstream; refuse rather than emit a file that
    # looks right.
    n_cols = X.shape[1] if hasattr(X, "shape") and len(X.shape) == 2 else None
    if n_cols is not None and n_cols != len(var_names):
        raise ValueError(
            f"count matrix has {n_cols} gene columns but {len(var_names)} gene labels were supplied -- "
            "the matrix and the labels describe different gene sets, so every written row would be "
            "attributed to the wrong gene."
        )

    if str(out_path).endswith(".h5"):
        counts = _rounded_count_csr(X, what=what, report=report)
        if counts.nnz == 0:
            raise ValueError(
                f"the {what} has no non-zero count in the {len(var_names)} genes it was restricted to; "
                "there is nothing for STRIDE to deconvolve."
            )
        _write_10x_h5(counts, obs_names, var_names, out_path)
        log(f"  wrote {counts.shape[1]} genes x {counts.shape[0]} observations ({counts.nnz} non-zero) as 10x HDF5")
        return counts

    is_sparse = sparse.issparse(X)
    if is_sparse:
        # X shape: (n_obs, n_genes); convert to CSC so columns are genes
        X = X.tocsc()
    else:
        X = np.asarray(X)

    with open(out_path, "w") as f:
        # header
        f.write("gene\t" + "\t".join(map(str, obs_names)) + "\n")

        n_genes = len(var_names)
        for gi, gene in enumerate(var_names):
            if is_sparse:
                col = X[:, gi].toarray().ravel()
            else:
                # X shape: (n_obs, n_genes); take column gi
                col = X[:, gi]

            # Convert to integer counts (STRIDE expects count-like input). Round rather than
            # truncate: int() floors, so a reference that arrives normalised instead of as raw
            # counts loses every value below 1.0 -- an expression of 0.99 is handed to STRIDE as
            # "gene not detected". On genuine integer counts the two are identical.
            row_vals = "\t".join(str(int(round(float(v)))) for v in col)
            f.write(f"{gene}\t{row_vals}\n")

            if (gi + 1) % 1000 == 0 or gi == n_genes - 1:
                log(f"  wrote {gi + 1}/{n_genes} genes")


def column_variance(X) -> np.ndarray:
    """Per-column population variance (``np.var``, ddof=0) of an obs x genes matrix, in float64.

    The small-panel ranking used to call ``X.toarray()`` on the whole slide and take ``np.var`` of
    that. Here one block of genes is densified at a time (at most ``_VARIANCE_BLOCK_BYTES``), so the
    peak is a slice of the slide rather than all of it. Accumulating in float64 can move a float32
    near-tie in the last bit, nothing more.
    """
    n_obs, n_genes = X.shape
    X = X.tocsc() if sparse.issparse(X) else np.asarray(X)
    block = max(1, int(_VARIANCE_BLOCK_BYTES // max(1, n_obs * 8)))
    parts = []
    for start in range(0, n_genes, block):
        chunk = X[:, start : min(n_genes, start + block)]
        chunk = chunk.toarray() if sparse.issparse(chunk) else np.asarray(chunk)
        parts.append(np.var(chunk, axis=0, dtype=np.float64))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def write_celltype_file(adata_sc, annotation_key: str, out_path: str) -> None:
    """
    Write a 2-column cell-type annotation file for STRIDE:

    <cell_id> <tab> <cell_type>

    No header row.
    """
    log(f"Writing cell type file to {out_path}")
    if annotation_key not in adata_sc.obs:
        raise ValueError(
            f"annotation_key='{annotation_key}' not found in scRNA obs. Available keys: {list(adata_sc.obs.keys())}"
        )

    df = pd.DataFrame(
        {
            "cell": adata_sc.obs_names.astype(str),
            "cell_type": adata_sc.obs[annotation_key].astype(str).values,
        }
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False, header=False)


def subset_to_common_genes(sc_adata, st_adata):
    """
    Subset both AnnData objects to common genes, same order.
    """
    common = np.intersect1d(sc_adata.var_names, st_adata.var_names)
    if common.size == 0:
        raise ValueError(id_mismatch_msg("genes", "scRNA", sc_adata.var_names, "spatial", st_adata.var_names))
    log(f"Found {common.size} common genes between scRNA and spatial data.")
    sc_sub = sc_adata[:, common].copy()
    st_sub = st_adata[:, common].copy()
    return sc_sub, st_sub, common


def _topic_spot_prefix(outprefix: str) -> str:
    return outprefix + "_topic_spot_mat_"


def _dominant_celltype_file(output_dir: str, outprefix: str) -> str:
    """The per-spot summary this worker writes after STRIDE exits. STRIDE itself never writes it."""
    return os.path.join(output_dir, f"{outprefix}_dominant_celltype_per_spot.csv")


def earlier_topic_matrices_of_this_tool(output_dir: str, outprefix: str) -> list:
    """Name the topic x spot matrices an earlier run of this tool left here; refuse any other.

    STRIDE's ``SpatialDeconvolve`` loads ``<outprefix>_topic_spot_mat_<k>.npz`` when the file already
    exists instead of recomputing it -- a cache meant for ``--model-dir`` reuse. This worker always
    trains a fresh LDA model, and LDA topics come out numbered differently on every unseeded run, so
    a second run into the same output_dir and outprefix that selects the same k would multiply the
    earlier run's topic weights by the new model's topic-to-cell-type table: fractions that describe
    neither run, reported as a success.

    An earlier run of this tool is recognised by ``<outprefix>_dominant_celltype_per_spot.csv``, which
    only this worker writes, after STRIDE has exited: every topic matrix that run wrote (or, before
    this check existed, silently reused) is no newer than it. Those are this tool's own output; they
    are returned so that the caller removes them just before STRIDE starts and STRIDE recomputes the
    matrix for the model it has just trained. A matrix with no such file beside it, or newer than it,
    was written by something else (a STRIDE run outside this tool, a copy): it is refused here and
    left untouched.
    """
    if not os.path.isdir(output_dir):
        return []
    prefix = _topic_spot_prefix(outprefix)
    cached = sorted(fn for fn in os.listdir(output_dir) if fn.startswith(prefix) and fn.endswith(".npz"))
    if not cached:
        return []
    result_file = _dominant_celltype_file(output_dir, outprefix)
    result_name = os.path.basename(result_file)
    if os.path.isfile(result_file):
        finished_ns = os.stat(result_file).st_mtime_ns
        foreign = [fn for fn in cached if os.stat(os.path.join(output_dir, fn)).st_mtime_ns > finished_ns]
        why = (
            f"they are newer than {result_name}, which this tool wrote when its last run here finished, so "
            "something else wrote them after it"
        )
    else:
        foreign = cached
        why = f"there is no {result_name} beside them, the file this tool writes at the end of every run"
    if foreign:
        raise ValueError(
            f"output_dir {output_dir} already holds {', '.join(foreign)} with outprefix='{outprefix}', and not "
            f"from an earlier run of this tool: {why}. STRIDE loads an existing <outprefix>_topic_spot_mat_<k>.npz "
            "instead of recomputing it, so this run would combine that topic x spot matrix with a newly trained "
            "topic model whose topics are numbered differently, giving fractions that match neither. This tool "
            "replaces only the topic matrices its own earlier runs left and does not delete anything else: use a "
            "new output_dir or another outprefix, or remove those files yourself if they are not needed."
        )
    return cached


def replace_earlier_topic_matrices(output_dir: str, cached: list) -> list:
    """Remove this tool's own earlier topic matrices (and their .txt twins) just before STRIDE starts.

    ``cached`` comes from ``earlier_topic_matrices_of_this_tool``. The removed names are returned for
    ``params.stale_topic_matrices_removed``.
    """
    removed = []
    for npz in cached:
        for name in (npz, npz[: -len(".npz")] + ".txt"):
            path = os.path.join(output_dir, name)
            if not os.path.lexists(path):  # the readable .txt twin may be missing
                continue
            try:
                os.remove(path)
            except OSError as exc:
                raise RuntimeError(
                    f"could not remove {path}, a topic x spot matrix an earlier run of this tool left in "
                    f"output_dir ({exc}). STRIDE would load it instead of recomputing the matrix for this run's "
                    "model; use a new output_dir or another outprefix."
                ) from exc
            removed.append(name)
    return removed


def _missing_gene_list_msg(gene_use: str) -> str:
    return (
        f"gene_use_file {gene_use} is not an existing file. STRIDE does not stop on a missing gene "
        'list: it prints "The gene file doesn\'t exist" and trains on its own marker search (top 200 '
        "per cell type) instead, while the run would still name your file. Pass a readable file with "
        "one gene symbol per line, the literal 'All' for every shared gene, or leave gene_use_file "
        "unset to let STRIDE pick markers."
    )


def _read_gene_list(gene_use: str) -> set:
    try:
        with open(gene_use) as f:
            return {line.strip() for line in f if line.strip()}
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"gene_use_file {gene_use} exists but could not be read ({exc}); STRIDE would fail on it too."
        ) from exc


# What an all-zero scaled matrix does on each side of STRIDE.
_EMPTY_DOCUMENTS = {
    "st_scale_factor": "gensim answers each with the same topic prior, published as every spot's composition",
    "sc_scale_factor": "the topic model would be trained on no words at all",
}


def stride_default_scale_factor(totals) -> tuple:
    """``(q75, factor)``: STRIDE's own default scale factor for these per-observation count totals.

    ``STRIDE/ModelTrain.py`` (``scProcess`` :58, ``stProcess`` :94) sets, when no factor is passed,
    ``np.round(np.quantile(count_per_obs, 0.75) / 1000, 0) * 1000`` -- the 75th percentile of the
    counts per cell/spot over the staged genes, rounded to the nearest 1000. Mirrored here verbatim.
    """
    q75 = float(np.quantile(np.asarray(totals, dtype=np.float64), 0.75))
    return q75, float(np.round(q75 / 1000, 0) * 1000)


def resolve_scale_factor(totals, requested, knob: str, noun: str, n_genes: int) -> dict:
    """The scale factor STRIDE will normalise ``noun`` counts with, and whether it must be passed.

    STRIDE scales every observation to ``count / total * factor`` and builds each one's LDA document
    from the non-zero scaled values. Its default factor rounds the 75th percentile of the totals to
    the nearest 1000, so any slide or reference whose q75 is below 500 counts over the staged genes
    -- a Xenium panel (q75 316 on the library's tonsil), a VisiumHD bin (302) -- gets factor 0:
    every scaled count is 0, every document is empty, and gensim answers each one with the same
    uniform topic prior, which STRIDE publishes as every spot's composition with exit code 0.

    * ``requested`` (the ``knob`` the caller set): used as given; it must be a finite number > 0.
    * otherwise STRIDE's default, when it is above 0: nothing is passed (STRIDE computes the same).
    * otherwise the unrounded 75th percentile, passed explicitly -- the value STRIDE's own help
      text promises ("the 75% quantile of nCount") -- with a warning that names both numbers.
    * a 75th percentile of 0 (three quarters of the observations have no count in the staged
      genes) stops the run with the numbers and the knob.

    Returns ``{"value", "passed", "source", "warning", "q75", "stride_default"}``.
    """
    q75, stride_default = stride_default_scale_factor(totals)
    what = f"the counts per {noun} over the {n_genes} staged genes"
    if requested is not None:
        return {
            "value": float(requested),
            "passed": True,
            "source": f"the {knob} you supplied",
            "warning": "",
            "q75": q75,
            "stride_default": stride_default,
        }
    if stride_default > 0:
        return {
            "value": stride_default,
            "passed": False,
            "source": f"STRIDE's default: the 75th percentile of {what} ({q75:g}), rounded to the nearest 1000",
            "warning": "",
            "q75": q75,
            "stride_default": stride_default,
        }
    if q75 > 0:
        return {
            "value": q75,
            "passed": True,
            "source": (
                f"the 75th percentile of {what}, unrounded: STRIDE's default rounds it to the nearest 1000, which is 0"
            ),
            "warning": (
                f"STRIDE's default {knob} is the 75th percentile of {what} rounded to the nearest 1000: "
                f"{q75:g} rounds to 0, which would scale every count to 0 and leave every {noun}'s topic document "
                f"empty ({_EMPTY_DOCUMENTS[knob]}). The unrounded value {q75:g} was passed instead "
                f"(--{knob.replace('_', '-')}); set {knob} to choose another."
            ),
            "q75": q75,
            "stride_default": stride_default,
        }
    totals = np.asarray(totals)
    n_zero = int(np.count_nonzero(totals == 0))
    raise ValueError(
        f"{n_zero} of {totals.size} {noun}s have no count in the {n_genes} genes staged for STRIDE, so the 75th "
        f"percentile of {what} is 0 and STRIDE's scale factor would be 0: every scaled count would be 0 and every "
        f"{noun}'s topic document empty ({_EMPTY_DOCUMENTS[knob]}). Pass {knob} (a number > 0, e.g. 10000) to set "
        f"the factor yourself; the {noun}s with no count in those genes still carry no information."
    )


def _check_scale_factor(value, knob: str):
    """A scale factor the caller set must be a finite number above 0 (STRIDE reads 0 as 'unset')."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{knob}={value!r} is not a number; pass a scale factor > 0 (e.g. 10000).") from None
    if not np.isfinite(number) or number <= 0:
        raise ValueError(
            f"{knob}={value!r} is not a scale factor STRIDE can use: it must be a finite number > 0 "
            "(STRIDE treats 0 as unset and scales every count to 0). Leave it unset for STRIDE's default."
        )
    return number


_GENE_DICTIONARY = "Gene_dict.txt"


def _file_signature(path: str):
    """``(inode, size, mtime_ns)`` of ``path``, or None when it does not exist."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def read_gene_dictionary(path: str, signature_before=None):
    """The genes of the topic model STRIDE trained in this run, from its ``Gene_dict.txt``, or None.

    ``scLDA`` saves its gensim ``Dictionary`` there (``save_as_text``: a document-count line, then
    ``id<TAB>token<TAB>docfreq`` per gene), and ``SpatialDeconvolve`` builds each spot's document
    from exactly these genes. ``signature_before`` is the file's :func:`_file_signature` before
    STRIDE started: a file this run did not write (absent now, or unchanged) is not this run's.
    """
    after = _file_signature(path)
    if after is None or after == signature_before:
        return None
    genes = []
    try:
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                parts = line.rstrip("\n").split("\t")
                if i == 0 and len(parts) == 1:
                    continue  # the document count gensim writes first
                if len(parts) < 2 or not parts[1]:
                    return None
                genes.append(parts[1])
    except (OSError, UnicodeDecodeError):
        return None
    return genes or None


def spots_without_counts(counts, var_names, genes) -> np.ndarray:
    """Boolean mask over the rows of ``counts`` (obs x genes CSR): no count in any of ``genes``."""
    position = {str(g): i for i, g in enumerate(var_names)}
    idx = sorted({position[str(g)] for g in genes if str(g) in position})
    if not idx:
        return np.ones(counts.shape[0], dtype=bool)
    totals = np.asarray(counts[:, idx].sum(axis=1)).ravel()
    return totals == 0


def _rounding_note(report: dict, what: str, source: str) -> str:
    """A warning when staging rounded values that were not whole numbers (see ``_rounded_count_csr``)."""
    if not report or not report.get("n_non_integer"):
        return ""
    return (
        f"{report['n_non_integer']} of the {report['n_values']} stored values of the {what} (read from {source}) "
        f"are not whole numbers (e.g. {report['example']:g}). STRIDE's topic model reads counts, so they were "
        "rounded to the nearest integer; a normalised or log-transformed matrix gives compositions that do not "
        "describe counts. Supply raw counts in layers['counts'], adata.raw or X."
    )


def _write_csv_atomic(frame, path: str, **kwargs) -> None:
    """``frame.to_csv(path)`` through ``<path>.partial`` and a rename: a killed run leaves no half file."""
    partial = path + ".partial"
    try:
        frame.to_csv(partial, **kwargs)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def run_stride_deconvolution(
    sc_h5ad: str,
    spatial_h5ad: str,
    output_dir: str,
    annotation_key: str,
    outprefix: str,
    normalize: bool,
    gene_use: str | None,
    ntopics: list[int] | None = None,
    drop_unlabeled: bool = False,
    st_scale_factor: float | None = None,
    sc_scale_factor: float | None = None,
) -> dict:
    """
    Core STRIDE deconvolution pipeline for SpatialOmicsLab worker.

    Spots with ``obs['in_tissue'] == 0`` (background glass in a CELLxGENE export) are left out
    before anything is staged and reported (``params.in_tissue_filter``). STRIDE's scale factors
    are resolved by :func:`resolve_scale_factor` and published as ``params.st_scale_factor`` /
    ``params.sc_scale_factor``; spots with no count in the genes STRIDE's topic model used get only
    the topic prior from STRIDE, so they are counted (``data.n_spots_without_counts``) and given no
    dominant cell type.
    """
    st_scale_factor = _check_scale_factor(st_scale_factor, "st_scale_factor")
    sc_scale_factor = _check_scale_factor(sc_scale_factor, "sc_scale_factor")
    os.makedirs(output_dir, exist_ok=True)

    log("Task = deconvolution")
    log(f"sc_h5ad      = {sc_h5ad}")
    log(f"spatial_h5ad = {spatial_h5ad}")
    log(f"output_dir   = {output_dir}")
    log(f"annotation_key = {annotation_key}")
    log(f"outprefix      = {outprefix}")
    log(f"normalize      = {str(normalize)}")
    if gene_use:
        log(f"gene_use       = {gene_use}")

    # Before any work: a topic matrix that STRIDE would load instead of recomputing is either this
    # tool's own from an earlier run here (replaced just before STRIDE starts, so a run that stops
    # earlier leaves it in place) or someone else's (refused now). A gene list that is not there would
    # be silently swapped for STRIDE's own (checked in full below).
    earlier_topic_matrices = earlier_topic_matrices_of_this_tool(output_dir, outprefix)
    if gene_use and gene_use != "All" and not os.path.isfile(gene_use):
        raise ValueError(_missing_gene_list_msg(gene_use))

    # 1. Load data
    sc_adata = sc.read_h5ad(sc_h5ad)
    st_adata = sc.read_h5ad(spatial_h5ad)

    # Background spots (obs['in_tissue'] == 0 in a CELLxGENE export) are glass, not tissue: ambient
    # counts deconvolved as if they were cells, or -- when empty -- STRIDE's uniform topic prior
    # published as a composition. They are left out before anything is staged, and counted.
    st_adata, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(st_adata, "spots")
    if n_spots_off_tissue:
        log(f"Leaving out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0 (background)")

    renamed_sc = make_names_unique_and_report(sc_adata)
    renamed_st = make_names_unique_and_report(st_adata)

    log(
        f"Loaded scRNA: n_cells={sc_adata.n_obs}, n_genes={sc_adata.n_vars}; "
        f"spatial: n_spots={st_adata.n_obs}, n_genes={st_adata.n_vars}"
    )

    # The subset below rebinds both objects, so the panels the caller supplied have to be captured
    # first -- otherwise n_genes, n_genes_sc and n_overlap_genes are three labels for one number.
    n_genes_st_supplied = int(st_adata.n_vars)
    n_genes_sc_supplied = int(sc_adata.n_vars)
    n_cells_sc_supplied = int(sc_adata.n_obs)

    # Labels are checked before anything is staged. A missing label is not a cell type: cast to str
    # it becomes a class called "nan" that STRIDE trains a topic for and reports a fraction of.
    if annotation_key not in sc_adata.obs:
        raise ValueError(
            f"annotation_key='{annotation_key}' not found in scRNA obs. Available keys: {list(sc_adata.obs.keys())}"
        )
    keep, n_unlabeled = _split_unlabeled(
        sc_adata.obs[annotation_key].values, bool(drop_unlabeled), what=f"reference cells in obs['{annotation_key}']"
    )
    if n_unlabeled:
        if n_unlabeled == n_cells_sc_supplied:
            raise ValueError(f"all {n_cells_sc_supplied} reference cells have no label in obs['{annotation_key}'].")
        log(f"Dropping {n_unlabeled} reference cells with no label in obs['{annotation_key}'] (drop_unlabeled=True)")
        sc_adata = sc_adata[keep].copy()

    # 2. Restrict to common genes
    sc_adata, st_adata, common_genes = subset_to_common_genes(sc_adata, st_adata)
    n_genes_shared = int(len(common_genes))

    # 3. Prepare matrices
    sc_X, sc_source = count_matrix_and_source(sc_adata)
    st_X, st_source = count_matrix_and_source(st_adata)

    # Handle small gene sets: STRIDE internally calls scanpy.pp.calculate_qc_metrics()
    # with percent_top that can exceed n_vars, causing IndexError.
    # Pre-compute marker genes so STRIDE skips its internal QC on small datasets.
    #
    # Whichever branch runs below, the gene list handed to STRIDE -- not the shared panel -- is the
    # basis the deconvolution is computed on, so it is tracked here and published as n_genes_used.
    # It stays None when no list is in force: STRIDE then selects internally and we cannot know.
    #
    # The gene list is settled before the count matrices are staged, so a bad gene_use_file stops
    # the run in seconds rather than after the slide has been written out.
    auto_marker_path = None
    n_genes_used = None
    genes_in_force = None  # the genes of the list STRIDE is handed, when there is one
    gene_use_note = ""
    gene_selection = _STRIDE_MARKERS
    if gene_use is None and st_adata.n_vars < 500:
        log(f"Small gene set ({st_adata.n_vars} genes) — pre-computing marker genes to avoid STRIDE QC IndexError")
        # Ranked on the count matrix STRIDE is handed (the "raw variance" the note below names), one
        # block of genes at a time rather than a dense copy of the whole slide.
        gene_var = column_variance(st_X)
        n_top = min(50, st_adata.n_vars)
        top_genes = st_adata.var_names[np.argsort(gene_var)[-n_top:]].tolist()
        marker_path = os.path.join(output_dir, "stride_marker_genes.txt")
        with open(marker_path, "w") as f:
            f.write("\n".join(top_genes))
        gene_use = marker_path
        auto_marker_path = marker_path
        n_genes_used = len(top_genes)
        genes_in_force = list(top_genes)
        gene_selection = f"top {n_genes_used} raw-variance genes of the shared panel (small-panel workaround)"
        gene_use_note = describe_reduction(
            "shared genes",
            n_genes_shared,
            n_genes_used,
            "STRIDE's small-panel workaround, which ranks the shared panel by raw (unnormalised) "
            "variance and hands STRIDE only the top "
            f"{n_genes_used} as --gene-use; this substitution was not requested, and supplying "
            "your own gene_use list declines it",
        )
        log(f"Wrote {len(top_genes)} marker genes to {marker_path}")
    elif gene_use:
        # STRIDE checks os.path.exists() first, then the literal "All", and treats anything else as
        # "no list": it prints "The gene file doesn't exist. Identifying markers..." and trains on
        # its own markers while the caller's path is still echoed back. Mirror its order, and stop
        # where it would silently substitute.
        if os.path.isfile(gene_use):
            listed = _read_gene_list(gene_use)
            shared = set(map(str, st_adata.var_names))
            if not listed:
                raise ValueError(f"gene_use_file {gene_use} lists no genes (one gene symbol per line is expected).")
            n_genes_used = len(listed & shared)
            genes_in_force = sorted(listed & shared)
            if n_genes_used == 0:
                raise ValueError(
                    id_mismatch_msg("genes", "gene_use_file", sorted(listed), "the shared panel", sorted(shared))
                    + " STRIDE would train its topic model on no genes."
                )
            n_absent = len(listed) - n_genes_used
            absent_note = (
                f" ({n_absent} of its {len(listed)} entries are absent from the shared panel and were ignored)"
                if n_absent
                else ""
            )
            gene_selection = "the gene_use list you supplied"
            gene_use_note = describe_reduction(
                "shared genes",
                n_genes_shared,
                n_genes_used,
                f"the gene_use list you supplied{absent_note}",
            )
        elif gene_use == "All":
            n_genes_used = n_genes_shared
            genes_in_force = [str(g) for g in st_adata.var_names]
            gene_selection = "every shared gene (gene_use='All')"
        else:
            raise ValueError(_missing_gene_list_msg(gene_use))

    # Staged as sparse 10x HDF5: STRIDE reads a .h5 straight into a CSC matrix, whereas its text
    # reader expands the whole dense matrix into Python floats first.
    sc_counts_path = os.path.join(output_dir, "stride_sc_gene_count.h5")
    st_counts_path = os.path.join(output_dir, "stride_st_gene_count.h5")
    sc_celltype_txt = os.path.join(output_dir, "stride_sc_celltype.txt")

    sc_rounding: dict = {}
    st_rounding: dict = {}
    sc_counts = write_gene_by_obs_counts(
        sc_X,
        obs_names=sc_adata.obs_names,
        var_names=sc_adata.var_names,
        out_path=sc_counts_path,
        what="scRNA-seq reference count matrix",
        report=sc_rounding,
    )
    st_counts = write_gene_by_obs_counts(
        st_X,
        obs_names=st_adata.obs_names,
        var_names=st_adata.var_names,
        out_path=st_counts_path,
        what="spatial count matrix",
        report=st_rounding,
    )
    write_celltype_file(sc_adata, annotation_key, sc_celltype_txt)

    # STRIDE normalises every spot (cell) to count / total * scale factor over the staged genes; a
    # default that rounds to 0 would empty every document. Resolved on the counts STRIDE reads.
    st_scale = resolve_scale_factor(
        np.asarray(st_counts.sum(axis=1)).ravel(), st_scale_factor, "st_scale_factor", "spot", n_genes_shared
    )
    sc_scale = resolve_scale_factor(
        np.asarray(sc_counts.sum(axis=1)).ravel(), sc_scale_factor, "sc_scale_factor", "reference cell", n_genes_shared
    )
    for side, resolved in (("spatial", st_scale), ("reference", sc_scale)):
        log(f"{side} scale factor = {resolved['value']:g} ({resolved['source']})")

    # 4. Run the STRIDE CLI installed alongside the interpreter running this worker, so a
    #    relocated conda root resolves rather than raising FileNotFoundError (see env_bin).
    cmd = [
        env_bin("STRIDE"),
        "deconvolve",
        "--sc-count",
        sc_counts_path,
        "--sc-celltype",
        sc_celltype_txt,
        "--st-count",
        st_counts_path,
        "--outdir",
        output_dir,
        "--outprefix",
        outprefix,
    ]
    if normalize:
        cmd.append("--normalize")
    if gene_use:
        cmd.extend(["--gene-use", gene_use])
    if ntopics:
        cmd.extend(["--ntopics"] + [str(n) for n in ntopics])
    if st_scale["passed"]:
        cmd.extend(["--st-scale-factor", repr(float(st_scale["value"]))])
    if sc_scale["passed"]:
        cmd.extend(["--sc-scale-factor", repr(float(sc_scale["value"]))])

    stale_removed = replace_earlier_topic_matrices(output_dir, earlier_topic_matrices)
    cache_note = ""
    if stale_removed:
        cache_note = (
            f"Replaced the topic x spot matrices an earlier run of this tool left in output_dir with "
            f"outprefix '{outprefix}' ({', '.join(stale_removed)}): STRIDE loads an existing "
            "<outprefix>_topic_spot_mat_<k>.npz instead of recomputing it, and would have paired that earlier "
            "matrix with the topic model trained in this run. They were removed before STRIDE started, so the "
            "topic matrix and the fractions reported here are this run's."
        )
        log(cache_note)

    log("Running STRIDE:")
    log("  " + " ".join(cmd))

    # scLDA saves the topic model's gene dictionary here; its signature before the run tells this
    # run's file from one an earlier run left.
    gene_dictionary_path = os.path.join(output_dir, _GENE_DICTIONARY)
    gene_dictionary_before = _file_signature(gene_dictionary_path)
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log(f"STRIDE ran for {time.time() - started:.1f} s")
    log(f"STRIDE finished with return code {proc.returncode}")
    if proc.stdout:
        log("=== STRIDE STDOUT ===")
        for line in proc.stdout.splitlines():
            log(line)
    if proc.stderr:
        log("=== STRIDE STDERR ===")
        for line in proc.stderr.splitlines():
            log(line)

    if proc.returncode != 0:
        raise RuntimeError(f"STRIDE deconvolve failed with code {proc.returncode}")

    # 5. Collect outputs
    frac_file = os.path.join(output_dir, f"{outprefix}_spot_celltype_frac.txt")
    if not os.path.exists(frac_file):
        raise RuntimeError(
            f"STRIDE exited 0 but wrote no {os.path.basename(frac_file)} in {output_dir}; there is no "
            "deconvolution result to report."
        )

    # The .npz is the matrix this run computed (an earlier one was removed or refused above); its .txt
    # twin is the readable copy, and k in the name is the topic number STRIDE selected.
    topic_file = None
    ntopics_selected = None
    topic_prefix = _topic_spot_prefix(outprefix)
    for fn in sorted(os.listdir(output_dir)):
        if fn.startswith(topic_prefix) and fn.endswith(".npz"):
            k = fn[len(topic_prefix) : -len(".npz")]
            txt = os.path.join(output_dir, topic_prefix + k + ".txt")
            topic_file = txt if os.path.exists(txt) else None
            ntopics_selected = int(k) if k.isdigit() else None
            break

    # A spot with no count in the genes STRIDE's topic model uses is an empty LDA document, and
    # gensim answers every empty document with the same topic prior: STRIDE writes that prior as
    # the spot's composition. Those spots are counted, over the model's own genes when this run's
    # Gene_dict.txt can be read, and get no dominant cell type.
    model_genes = read_gene_dictionary(gene_dictionary_path, gene_dictionary_before)
    if model_genes is not None:
        zero_basis = f"the {len(model_genes)} genes of STRIDE's topic model"
    elif genes_in_force is not None:
        model_genes = genes_in_force
        zero_basis = f"the {len(model_genes)} genes of the gene list STRIDE was handed"
    else:
        model_genes = [str(g) for g in st_adata.var_names]
        zero_basis = (
            f"the {len(model_genes)} shared genes (STRIDE's own marker list could not be read, so spots with counts "
            "only outside its markers are not counted here)"
        )
    no_count_mask = spots_without_counts(st_counts, st_adata.var_names, model_genes)
    n_spots_without_counts = int(no_count_mask.sum())
    no_count_spots = {str(s) for s in st_adata.obs_names[no_count_mask]}
    n_spots_used = int(st_adata.n_obs)
    no_count_note = ""
    if n_spots_without_counts == n_spots_used:
        raise ValueError(
            f"none of the {n_spots_used} spots has a count in {zero_basis}, so STRIDE gave every spot the same "
            "topic prior and there is no composition to report. Check that the reference and the slide measure the "
            "same genes, or pass gene_use_file='All' to train on every shared gene."
        )
    if n_spots_without_counts:
        no_count_note = (
            f"{n_spots_without_counts} of {n_spots_used} spots have no count in {zero_basis}. STRIDE gives such a "
            "spot an empty topic document, whose composition is the model's topic prior -- the same for every such "
            "spot, not an estimate -- so their rows in the fraction table carry that prior and they have no "
            "dominant cell type."
        )
        log(no_count_note)

    dominant_csv = None
    celltypes = None
    dominant_counts: dict = {}
    post_warnings = []
    try:
        frac_df = pd.read_csv(frac_file, sep="\t", index_col=0)
        celltypes = list(frac_df.columns)
        dominant = frac_df.idxmax(axis=1).astype(object)
        if no_count_spots:
            dominant[frac_df.index.astype(str).isin(no_count_spots)] = np.nan
        dominant_counts = {str(k): int(v) for k, v in dominant.dropna().value_counts().items()}
        dominant_csv = _dominant_celltype_file(output_dir, outprefix)
        _write_csv_atomic(dominant.to_frame("dominant_celltype"), dominant_csv)
        log(f"Saved dominant cell-type per spot to {dominant_csv} (n_spots={dominant.shape[0]})")
    except Exception as e:
        dominant_csv = None
        post_warnings.append(
            f"STRIDE's fraction table {frac_file} could not be summarised ({str(e)}); the dominant cell type "
            "per spot and the counts below are missing, the fraction table itself is unchanged."
        )
        log(f"Warning: failed to post-process frac file: {str(e)}")

    n_celltypes = len(celltypes) if celltypes else 0

    st_gene_note = describe_reduction(
        "spatial genes",
        n_genes_st_supplied,
        n_genes_shared,
        "restricting the analysis to the genes the single-cell reference also measured",
    )
    sc_gene_note = describe_reduction(
        "single-cell genes",
        n_genes_sc_supplied,
        n_genes_shared,
        "restricting the analysis to the genes the spatial slide also measured",
    )
    sc_cell_note = describe_reduction(
        "reference cells",
        n_cells_sc_supplied,
        int(sc_adata.n_obs),
        f"drop_unlabeled=True: they had no label in obs['{annotation_key}']",
    )
    spot_note = describe_reduction(
        "spots",
        n_spots_supplied,
        n_spots_used,
        "leaving out the spots with obs['in_tissue'] == 0 (background outside the tissue)",
    )
    rounding_notes = [
        _rounding_note(st_rounding, "spatial count matrix", st_source),
        _rounding_note(sc_rounding, "scRNA-seq reference count matrix", sc_source),
    ]
    scale_notes = [st_scale["warning"], sc_scale["warning"]]

    out = WorkerOutput("stride", task="deconvolution")
    counts = {
        "n_spots": n_spots_supplied,
        "n_spots_used": n_spots_used,
        "n_spots_without_counts": n_spots_without_counts,
        "n_genes": n_genes_st_supplied,
        "n_cells_sc": n_cells_sc_supplied,
        "n_cells_sc_used": int(sc_adata.n_obs),
        "n_genes_sc": n_genes_sc_supplied,
        "n_overlap_genes": n_genes_shared,
    }
    if n_genes_used is not None:
        counts["n_genes_used"] = n_genes_used
    out.set_data(**counts)
    out.add_output_files(
        {
            "spot_celltype_fraction_file": frac_file,
            "topic_spot_matrix_file": topic_file,
            "dominant_celltype_per_spot": dominant_csv,
            "marker_gene_file": auto_marker_path,
        }
    )
    out.add_params(
        {
            "annotation_key": annotation_key,
            "outprefix": outprefix,
            "normalize": normalize,
            "gene_use": gene_use,
            "ntopics": [int(n) for n in ntopics] if ntopics else None,
            "ntopics_selected": ntopics_selected,
            "gene_selection": gene_selection,
            "drop_unlabeled": bool(drop_unlabeled),
            "n_reference_cells_dropped_unlabeled": int(n_unlabeled),
            "staged_count_format": "10x HDF5 (sparse)",
            "stale_topic_matrices_removed": stale_removed,
            # The factors STRIDE normalised with, whether it computed them or this run passed them.
            "st_scale_factor": st_scale["value"],
            "st_scale_factor_source": st_scale["source"],
            "sc_scale_factor": sc_scale["value"],
            "sc_scale_factor_source": sc_scale["source"],
            "expression_source": st_source,
            "expression_source_sc": sc_source,
        }
    )
    record_method(out, METHOD_NAME)
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    out.add_params(identifier_rename_params(renamed_st))
    out.add_params(identifier_rename_params(renamed_sc, suffix="sc"))
    out.set_summary(
        n_celltypes=n_celltypes,
        celltypes=celltypes if celltypes else [],
        dominant_counts=dominant_counts,
    )
    # A successful run never carries stderr into the payload (base_mcp attaches stderr_tail only on
    # a non-zero exit), so the panel STRIDE actually scored has to travel in the payload itself.
    out.add_warnings([n.strip() for n in (st_gene_note, sc_gene_note, gene_use_note, sc_cell_note) if n])
    out.add_warnings([cache_note] if cache_note else [])
    out.add_warnings([n for n in scale_notes + rounding_notes if n])
    out.add_warnings([no_count_note] if no_count_note else [])
    out.add_warnings(post_warnings)
    extra = "".join(" " + n for n in scale_notes + [no_count_note] + rounding_notes if n)
    out.set_analysis(
        build_deconv_analysis(
            n_celltypes,
            dominant_counts,
            total_spots=n_spots_used - n_spots_without_counts,
            method_name="STRIDE",
        )
        + spot_note
        + extra
        + st_gene_note
        + sc_gene_note
        + gene_use_note
        + sc_cell_note
        + identifier_rename_note(renamed_st, subject="spatial data")
        + identifier_rename_note(renamed_sc, subject="scRNA-seq reference")
    )
    return out.to_dict()


def main():
    parser = argparse.ArgumentParser(description="STRIDE worker for SpatialOmicsLab MCP (two-layer: wrapper + worker).")
    parser.add_argument(
        "--task",
        default="deconvolution",
        choices=["deconvolution"],
        help="Currently only 'deconvolution' is supported.",
    )
    parser.add_argument(
        "--sc-h5ad",
        required=True,
        help="Path to scRNA reference AnnData (.h5ad).",
    )
    parser.add_argument(
        "--spatial-h5ad",
        required=True,
        help="Path to spatial transcriptomics AnnData (.h5ad).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store STRIDE outputs.",
    )
    parser.add_argument(
        "--annotation-key",
        default="CellType",
        help="obs column in sc_h5ad containing cell-type labels (e.g. 'CellType').",
    )
    parser.add_argument(
        "--outprefix",
        default="stride_run",
        help="Prefix for STRIDE output files (spot_celltype_frac, topic_spot_mat, etc.).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Pass --normalize to STRIDE deconvolve.",
    )
    parser.add_argument(
        "--gene-use",
        default=None,
        help="Marker gene list file for STRIDE --gene-use (must exist), or the literal 'All'.",
    )
    parser.add_argument(
        "--ntopics",
        nargs="*",
        type=int,
        default=None,
        help="Topic numbers to evaluate. If not set, STRIDE auto-ranges.",
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="Leave out reference cells with a missing (NaN/empty) label instead of stopping.",
    )
    parser.add_argument(
        "--st-scale-factor",
        type=float,
        default=None,
        help=(
            "STRIDE --st-scale-factor (> 0). Unset: STRIDE's default (75th percentile of counts per spot rounded to "
            "the nearest 1000), or the unrounded percentile when that rounds to 0."
        ),
    )
    parser.add_argument(
        "--sc-scale-factor",
        type=float,
        default=None,
        help="STRIDE --sc-scale-factor (> 0), the same rule over counts per reference cell.",
    )

    args = parser.parse_args()

    try:
        if args.task == "deconvolution":
            result = run_stride_deconvolution(
                sc_h5ad=args.sc_h5ad,
                spatial_h5ad=args.spatial_h5ad,
                output_dir=args.output_dir,
                annotation_key=args.annotation_key,
                outprefix=args.outprefix,
                normalize=args.normalize,
                gene_use=args.gene_use,
                ntopics=args.ntopics,
                drop_unlabeled=args.drop_unlabeled,
                st_scale_factor=args.st_scale_factor,
                sc_scale_factor=args.sc_scale_factor,
            )
        else:
            raise ValueError(unsupported_choice_msg("task", args.task, ["deconvolution"]))

        # IMPORTANT: only JSON to stdout (for MCP wrapper to parse)
        print(json.dumps(result, default=str))
        sys.stdout.flush()
    except Exception as e:
        log("ERROR:")
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("stride", str(e), task="deconvolution")
        sys.exit(1)


if __name__ == "__main__":
    main()
