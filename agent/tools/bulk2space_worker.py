#!/usr/bin/env python
"""
run_bulk2space worker for SpatialOmicsLab MCP: NNLS deconvolution on reference cell-type means.

WHAT RUNS HERE, HONESTLY. Upstream Bulk2Space (ZJUFanLab/bulk2space) is a two-stage method: a
beta-VAE trained on the scRNA-seq reference generates single cells whose type proportions are solved
from a *bulk* RNA-seq profile (``train_vae_and_generate``), and the generated cells are then placed
onto spots (``train_df_and_spatial_deconvolution``). This worker runs none of that. Earlier revisions
called ``Bulk2Space().fit(sc_count=..., st_count=...)``, a method the installed class does not have;
the ``AttributeError`` was caught by a broad ``except``, a non-negative least-squares deconvolution
ran instead, and the payload still said "Bulk2Space" with ``used_nnls_fallback=False``. Every run was
NNLS -- after first densifying both matrices into CSV files that nothing read.

So NNLS is the method, and the payload names it (``METHOD_NAME``):

  1. Load the scRNA-seq reference and the spatial AnnData. Background spots (``obs['in_tissue'] ==
     0``, every array spot of a CELLxGENE Visium export) are left out and counted
     (``params.in_tissue_filter``, ``data.n_spots_supplied``); they have no row in the outputs.
  2. Harmonise gene identifiers and intersect the two gene panels (duplicate gene names are made
     unique first and the renames reported, because a duplicated symbol cannot be indexed).
  3. Gene selection: keep the ``max_genes`` shared genes with the highest variance across reference
     cells (default 1000, as every earlier run did; ``max_genes=0`` keeps every shared gene). The
     selection is reported in ``params`` (``max_genes``, ``gene_selection``, ``n_genes_used``) and in
     the analysis text. No cells or spots are ever subsampled.
  4. Signature: mean expression per cell type over the reference, computed with a sparse indicator
     product, so a sparse reference is never densified.
  5. Per spot, NNLS of the spot's expression on the signature; the coefficients are normalised to sum
     to 1. A spot with no fit (all coefficients zero) keeps an all-zero row, is counted in
     ``n_spots_unassigned`` and is left out of ``dominant_counts``.
  6. Save ``bulk2space_proportions.csv`` and ``bulk2space_spatial.h5ad`` (atomically).

``params.method`` names NNLS and ``params.used_fallback`` is False: nothing else is attempted, so
NNLS is not a substitute here, it is the implementation. The legacy key ``params.used_nnls_fallback``
is kept for readers that already check it and is True on every run -- the proportions come from
NNLS, not from upstream Bulk2Space. ``output_files.csv_input_dir`` is kept and is None: no CSV
inputs are staged, because no upstream code reads them.

Both matrices are used as supplied (``X``), so give the reference and the slide on the same scale.
NNLS never normalises ``X`` as counts, so normalised data is accepted by design and not refused.
Spatial coordinates are not read: NNLS scores each spot on its own expression. Missing reference
labels are refused unless ``drop_unlabeled`` is set, and the number dropped is reported.

MEMORY. Nothing is densified whole. Reference variances are accumulated over row chunks, spots are
densified a bounded chunk at a time, and the only dense state is intrinsic to the output -- the
cell-type signature and the proportion table -- whose bytes are estimated against the memory this
process can still allocate (``worker_utils.available_memory_bytes``: MemAvailable, bounded by the room
under a cgroup memory limit) before they are allocated.

Environment: /opt/conda/envs/bulk2space_env (Python 3.8 with older numpy/pandas pins).
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
    default_output_dir,
    describe_reduction,
    gene_id_harmonization_note,
    gene_id_harmonization_params,
    harmonize_gene_ids,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    preflight_check,
    record_in_tissue,
    record_method,
    sanitize_cell_type_names,
)
from worker_utils import drop_unlabeled as _split_unlabeled  # the parameter of the same name shadows it

#: What this worker runs. Not "Bulk2Space": see the module docstring.
METHOD_NAME = (
    "non-negative least squares (scipy.optimize.nnls) of each spot on the reference's per-cell-type mean "
    "expression (upstream Bulk2Space not run: no VAE generation, no bulk profile, no cell-to-spot mapping)"
)
#: The short form the analysis sentence leads with.
METHOD_SHORT = "NNLS deconvolution on reference cell-type means (upstream Bulk2Space not run)"

#: Genes kept by the top-variance selection when the caller does not say otherwise. 1000 is what
#: every earlier run used (then unreported); 0 keeps every shared gene.
DEFAULT_MAX_GENES = 1000

#: Upper bound on the bytes of one densified row chunk (variance accumulation, NNLS spot blocks).
CHUNK_BYTES = 256 * 1024 * 1024


def _log(msg: str) -> None:
    """Log to stderr so stdout stays clean for JSON output."""
    print(f"[bulk2space] {msg}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """Redirect stdout to stderr so nothing a library prints can corrupt the JSON on stdout."""
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


def _load_h5ad(path: str):
    """Load an h5ad file, with fallback for different anndata versions."""
    try:
        import anndata as ad

        return ad.read_h5ad(path)
    except Exception:
        import scanpy as sc

        return sc.read_h5ad(path)


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` as ``<path>.partial`` and rename over ``path`` once it is complete."""
    tmp = str(path) + ".partial"
    df.to_csv(tmp)
    os.replace(tmp, str(path))


def _write_h5ad_atomic(adata, path: Path) -> None:
    """Write the AnnData as ``<path>.partial`` and rename over ``path`` once it is complete."""
    tmp = str(path) + ".partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, str(path))


def _chunk_rows(n_cols: int) -> int:
    """Rows per densified chunk so one float64 chunk stays within ``CHUNK_BYTES``."""
    return max(1, int(CHUNK_BYTES // (8 * max(1, int(n_cols)))))


def _as_csr(X):
    """CSR for a sparse matrix (row slicing a CSC matrix is slow); dense input is returned as is."""
    import scipy.sparse as sps

    if sps.issparse(X) and X.format != "csr":
        return X.tocsr()
    return X


def _column_variance(X, chunk_rows: int | None = None) -> np.ndarray:
    """Population variance (ddof=0) of each column over the rows, accumulated over row chunks.

    ``X.multiply(X)`` on a whole sparse reference materialises a second matrix the size of its
    non-zeros (the earlier code did exactly that); a dense reference's ``.var(axis=0)`` allocates a
    full-size temporary. Chunks keep the extra memory bounded by ``CHUNK_BYTES``.
    """
    import scipy.sparse as sps

    X = _as_csr(X)
    n_rows, n_cols = X.shape
    if chunk_rows is None:
        chunk_rows = _chunk_rows(n_cols)
    total = np.zeros(n_cols, dtype=np.float64)
    total_sq = np.zeros(n_cols, dtype=np.float64)
    sparse = sps.issparse(X)
    for start in range(0, n_rows, chunk_rows):
        block = X[start : start + chunk_rows]
        if sparse:
            block = block.astype(np.float64)
            total += np.asarray(block.sum(axis=0)).ravel()
            total_sq += np.asarray(block.multiply(block).sum(axis=0)).ravel()
        else:
            block = np.asarray(block, dtype=np.float64)
            total += block.sum(axis=0)
            total_sq += (block * block).sum(axis=0)
    if n_rows == 0:
        return total
    mean = total / n_rows
    return total_sq / n_rows - mean**2


def _top_variance_positions(X, max_genes: int) -> np.ndarray | None:
    """Sorted column positions of the ``max_genes`` highest-variance columns, or None for "keep all".

    None when ``max_genes`` is 0 or the matrix has no more columns than that. The ranking is the one
    every earlier run used (``np.argsort(var)[-max_genes:]``); what changed is that it is reported.
    """
    n_cols = int(X.shape[1])
    if not max_genes or n_cols <= max_genes:
        return None
    gene_var = _column_variance(X)
    return np.sort(np.argsort(gene_var)[-int(max_genes) :])


def _signature_matrix(X, labels) -> tuple[np.ndarray, list]:
    """Mean expression per cell type (types x genes) via a sparse indicator product.

    Cell types come out in the order the ``groupby(...).mean()`` this replaces gave them: a categorical
    label column keeps its category order, any other column is sorted. A category no cell carries is
    left out (it would be an all-NaN signature row, which NNLS refuses). The earlier code got that for
    free from anndata, whose ``adata[:, genes].copy()`` drops unused categories; the labels here are
    read from the loaded object, so the drop is done explicitly.
    """
    import scipy.sparse as sps

    cat = pd.Categorical(labels)
    codes = np.asarray(cat.codes, dtype=np.int64)
    n_types = len(cat.categories)
    idx = np.flatnonzero(codes >= 0)
    indicator = sps.csr_matrix(
        (np.ones(idx.shape[0], dtype=np.float64), (codes[idx], idx)),
        shape=(n_types, int(X.shape[0])),
    )
    sums = indicator @ X
    sums = sums.toarray() if sps.issparse(sums) else np.asarray(sums, dtype=np.float64)
    counts = np.bincount(codes[idx], minlength=n_types).astype(np.float64)
    used = counts > 0
    means = sums[used] / counts[used][:, None]
    names = list(np.asarray(cat.categories, dtype=object)[used])
    return means, names


def _nnls_proportions(X_spots, signature: np.ndarray) -> np.ndarray:
    """Per-spot NNLS on ``signature`` (types x genes); rows normalised to sum 1, all-zero if no fit.

    Spots are densified one bounded chunk at a time, so a sparse slide is never densified whole.
    """
    import scipy.sparse as sps
    from scipy.optimize import nnls

    A = np.ascontiguousarray(np.asarray(signature, dtype=np.float64).T)  # genes x types
    X_spots = _as_csr(X_spots)
    sparse = sps.issparse(X_spots)
    n_spots = int(X_spots.shape[0])
    proportions = np.zeros((n_spots, A.shape[1]), dtype=np.float64)
    step = _chunk_rows(A.shape[0])
    for start in range(0, n_spots, step):
        block = X_spots[start : start + step]
        block = block.toarray() if sparse else np.asarray(block)
        block = np.asarray(block, dtype=np.float64)
        for j in range(block.shape[0]):
            coef, _ = nnls(A, block[j])
            total = coef.sum()
            if total > 0:
                proportions[start + j, :] = coef / total
    return proportions


def _dense_bytes(n_spots: int, n_types: int, n_genes_used: int) -> int:
    """Bytes of the dense state intrinsic to the output, plus one chunk.

    The signature twice (types x genes and its transposed float64 copy for NNLS), the proportion
    table three times (the array, the obs columns written into the h5ad, and the CSV frame), and one
    densified spot chunk.
    """
    return int(8 * (2 * n_types * n_genes_used + 3 * n_spots * n_types) + min(CHUNK_BYTES, 8 * n_spots * n_genes_used))


def _memory_budget_bytes():
    """Memory this process can still allocate, or None when the platform cannot say.

    The shared reader (``worker_utils.available_memory_bytes``): the smaller of MemAvailable and the
    room under the cgroup memory limit, page cache counted as reclaimable. This used to read
    MemAvailable alone, which in a memory-limited container is the host's figure, so the check
    below could not refuse and an oversized run was OOM-killed with no payload.
    """
    return available_memory_bytes()


def _check_dense_state_fits(n_spots: int, n_types: int, n_genes_used: int) -> int:
    """Refuse, with the numbers, before allocating dense state the machine cannot hold."""
    need = _dense_bytes(n_spots, n_types, n_genes_used)
    budget = _memory_budget_bytes()
    if budget is not None and need > budget:
        raise MemoryError(
            f"NNLS deconvolution needs ~{need / 1e9:.1f} GB of dense state (a {n_types} cell types x "
            f"{n_genes_used} genes signature and a {n_spots} spots x {n_types} cell types proportion table) "
            f"but ~{budget / 1e9:.1f} GB is available (MemAvailable / room under the cgroup limit). The "
            "proportion table is the output itself; lower max_genes to shrink the signature, or run on a "
            "machine with more memory (the data is not subsampled)."
        )
    return need


def _run_bulk2space(
    sc_h5ad_path: str,
    spatial_h5ad_path: str,
    output_dir: str = default_output_dir(),
    cell_type_key: str = "cell_type",
    max_genes: int = DEFAULT_MAX_GENES,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """NNLS deconvolution of each spot on the reference's cell-type means. Returns a WorkerOutput dict."""
    allow_drop_unlabeled = bool(drop_unlabeled)
    max_genes = int(max_genes)
    if max_genes < 0:
        raise ValueError(f"max_genes={max_genes}: pass a positive gene count, or 0 to keep every shared gene.")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load data ----
    _log(f"Reading scRNA-seq reference: {sc_h5ad_path}")
    adata_sc = _load_h5ad(sc_h5ad_path)
    _log(f"Reading spatial data: {spatial_h5ad_path}")
    adata_st = _load_h5ad(spatial_h5ad_path)
    # Background spots (obs['in_tissue'] == 0) are not tissue. A CELLxGENE Visium export carries every
    # array spot -- 56-70% glass on the library's four such slides -- and NNLS fitted each one into
    # cell-type proportions that went into the table, the h5ad and every count below. They are left
    # out and counted; NNLS is per spot, so the in-tissue proportions are what they were.
    adata_st, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata_st, "spots")

    n_cells_sc_supplied = int(adata_sc.n_obs)
    n_genes_sc_supplied = int(adata_sc.n_vars)
    n_spots = int(adata_st.n_obs)
    n_genes_st_supplied = int(adata_st.n_vars)
    _log(f"scRNA: {n_cells_sc_supplied} cells x {n_genes_sc_supplied} genes")
    _log(f"Spatial: {n_spots} spots x {n_genes_st_supplied} genes")
    if n_spots_off_tissue:
        _log(f"Left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0 (background)")

    # Validate cell_type_key
    if cell_type_key not in adata_sc.obs.columns:
        raise ValueError(
            f"cell_type_key='{cell_type_key}' not found in scRNA obs. Available keys: {list(adata_sc.obs.columns)}"
        )

    # A missing label is not a class. It used to crash in sorted() with a TypeError that named nothing.
    try:
        keep, n_unlabeled = _split_unlabeled(
            adata_sc.obs[cell_type_key].values, allow_drop_unlabeled, what="reference cells"
        )
    except ValueError as exc:
        raise ValueError(f"cell_type_key='{cell_type_key}': {exc}") from exc
    if n_unlabeled:
        _log(f"Dropping {n_unlabeled} reference cells with no label in obs['{cell_type_key}'] (drop_unlabeled=True)")
        adata_sc = adata_sc[keep].copy()
    n_cells_sc_used = int(adata_sc.n_obs)

    # ---- 2. Harmonize gene IDs, make gene names unique, intersect ----
    _log("Harmonizing gene IDs between scRNA and spatial data...")
    gene_id_report: dict = {}
    harmonize_gene_ids(adata_sc, adata_st, report=gene_id_report)
    # A duplicated gene name cannot be indexed (pandas InvalidIndexError on the subset below); make
    # the names unique and say so, rather than crash on an ordinary 10x symbol table.
    renamed_sc = make_names_unique_and_report(adata_sc, axes=("var",))
    renamed_st = make_names_unique_and_report(adata_st, axes=("var",))
    shared_genes = sorted(set(adata_sc.var_names) & set(adata_st.var_names))
    if len(shared_genes) == 0:
        raise ValueError(
            "No shared genes between scRNA and spatial data even after gene ID harmonization. Check gene ID formats."
        )
    n_shared = len(shared_genes)
    _log(f"{n_shared} genes shared between scRNA and spatial")

    labels = adata_sc.obs[cell_type_key].values
    X_sc = adata_sc[:, shared_genes].X

    # ---- 3. Gene selection (reported, never silent) ----
    positions = _top_variance_positions(X_sc, max_genes)
    if positions is None:
        genes_used = list(shared_genes)
        gene_selection = "all shared genes"
    else:
        genes_used = [shared_genes[i] for i in positions]
        X_sc = _as_csr(X_sc)[:, positions]
        gene_selection = "top-variance (population variance across reference cells, X as supplied)"
        _log(f"Gene selection: top {max_genes} reference-variance genes of {n_shared} shared (max_genes={max_genes})")
    n_genes_used = len(genes_used)
    X_st = adata_st[:, genes_used].X

    # ---- 4. Signature matrix + per-spot NNLS ----
    signature, cell_types = _signature_matrix(X_sc, labels)
    del X_sc
    if not cell_types:
        raise ValueError(
            f"no labelled reference cells remain in obs['{cell_type_key}'] ({n_cells_sc_supplied} supplied, "
            f"{n_unlabeled} unlabelled), so there is no cell-type signature to deconvolve against."
        )
    dense_bytes = _check_dense_state_fits(n_spots, len(cell_types), n_genes_used)
    _log(f"Signature matrix: {len(cell_types)} cell types x {n_genes_used} genes")
    _log(f"Running NNLS on {n_spots} spots...")
    proportions = _nnls_proportions(X_st, signature)
    del X_st
    _log("NNLS deconvolution complete.")

    prop_df = pd.DataFrame(proportions, index=adata_st.obs_names, columns=cell_types)

    # ---- 5. Save results ----
    safe_cols, ct_renames = sanitize_cell_type_names(prop_df.columns)
    if len(set(safe_cols)) != len(safe_cols):
        clashes = sorted({c for c in safe_cols if safe_cols.count(c) > 1})
        raise ValueError(
            f"cell-type labels in obs['{cell_type_key}'] collide once '/' and ' ' become '_': {clashes}. "
            "Each would become the same proportion column; relabel them so they stay distinct."
        )
    prop_df.columns = safe_cols

    prop_csv_path = output_path / "bulk2space_proportions.csv"
    _write_csv_atomic(prop_df, prop_csv_path)
    _log(f"Saved proportions to {prop_csv_path}")

    # Store in spatial AnnData
    adata_st.obsm["bulk2space_proportions"] = prop_df.values
    for ct in safe_cols:
        adata_st.obs[ct] = prop_df[ct].values

    out_h5ad = output_path / "bulk2space_spatial.h5ad"
    _log(f"Saving annotated spatial AnnData to {out_h5ad}")
    _write_h5ad_atomic(adata_st, out_h5ad)

    # ---- 6. Build output ----
    cell_type_names = list(prop_df.columns)
    n_celltypes = len(cell_type_names)

    assigned = proportions.sum(axis=1) > 0
    n_unassigned = int((~assigned).sum())
    # A spot with no fit has an all-zero row; idxmax on it names the first column, which is not a finding.
    dominant_counts = dict(Counter(prop_df.loc[assigned].idxmax(axis=1))) if assigned.any() else {}

    out = WorkerOutput("bulk2space", task="deconvolution")
    out.set_data(
        n_cells_sc=n_cells_sc_supplied,
        n_cells_sc_used=n_cells_sc_used,
        n_genes_sc=n_genes_sc_supplied,
        n_genes_sc_used=n_genes_used,
        n_spots=n_spots,
        n_spots_supplied=n_spots_supplied,
        n_spots_off_tissue_dropped=n_spots_off_tissue,
        n_spots_unassigned=n_unassigned,
        n_genes=n_genes_st_supplied,
        n_genes_used=n_genes_used,
    )
    out.add_output_files(
        {
            "proportions_csv": str(prop_csv_path),
            "spatial_h5ad": str(out_h5ad),
            # Kept so the payload shape does not change; nothing is staged, because nothing reads it.
            "csv_input_dir": None,
        }
    )
    out.add_params(
        {
            "cell_type_key": cell_type_key,
            "n_shared_genes": n_shared,
            "max_genes": max_genes,
            "gene_selection": gene_selection,
            "n_genes_used": n_genes_used,
            "drop_unlabeled": allow_drop_unlabeled,
            "n_cells_sc_unlabeled_dropped": int(n_unlabeled),
            "n_spots_unassigned": n_unassigned,
            "dense_bytes_estimated": int(dense_bytes),
            # Upstream Bulk2Space is never imported, so there is no checkout to report.
            "bulk2space_repo": "not_used",
            # Legacy key, now true: every run's proportions are NNLS, never upstream Bulk2Space.
            "used_nnls_fallback": True,
        }
    )
    record_method(out, METHOD_NAME, used_fallback=False)
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    out.add_params(gene_id_harmonization_params(gene_id_report))
    out.add_params(identifier_rename_params(renamed_sc, suffix="sc"))
    out.add_params(identifier_rename_params(renamed_st))
    out.add_params(cell_type_rename_params(ct_renames))
    if n_unlabeled:
        out.add_warning(
            f"{n_unlabeled} reference cells had no label in obs['{cell_type_key}'] and were left out "
            "(drop_unlabeled=True)."
        )
    unassigned_note = ""
    if n_unassigned:
        unassigned_note = (
            f" NOTE: {n_unassigned} of {n_spots} spots had no NNLS fit on the {n_genes_used} analysed genes "
            "(every coefficient zero); their rows in the proportion table are all zero and they are left "
            "out of the dominant cell-type counts."
        )
        out.add_warning(unassigned_note.strip())
    out.set_summary(
        n_cell_types=n_celltypes,
        cell_type_names=cell_type_names,
        dominant_counts=dominant_counts,
    )
    gene_note = describe_reduction(
        "shared genes",
        n_shared,
        n_genes_used,
        f"the top-{max_genes} reference-variance gene selection (max_genes={max_genes}; max_genes=0 "
        "keeps every shared gene)",
    )
    cells_note = describe_reduction(
        "reference cells",
        n_cells_sc_supplied,
        n_cells_sc_used,
        f"drop_unlabeled=True, which leaves out cells whose '{cell_type_key}' label is missing",
    )
    spots_note = describe_reduction(
        "spots",
        n_spots_supplied,
        n_spots,
        "leaving out the obs['in_tissue'] == 0 background spots (they have no row in the proportion table)",
    )
    out.set_analysis(
        build_deconv_analysis(
            n_celltypes=n_celltypes,
            dominant_counts=dominant_counts,
            total_spots=n_spots,
            method_name=METHOD_SHORT,
        )
        + spots_note
        + gene_note
        + cells_note
        + unassigned_note
        + gene_id_harmonization_note(gene_id_report)
        + identifier_rename_note(renamed_sc, subject="scRNA reference")
        + identifier_rename_note(renamed_st, subject="spatial data")
        + cell_type_rename_note(ct_renames)
    )

    return out.to_dict()


def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "run_bulk2space worker: NNLS deconvolution of each spot on the reference's cell-type means "
            "(upstream Bulk2Space is not run)"
        )
    )
    parser.add_argument("--sc-h5ad", required=True, help="Path to scRNA-seq reference h5ad")
    parser.add_argument("--spatial-h5ad", required=True, help="Path to spatial h5ad")
    parser.add_argument("--output-dir", default=default_output_dir(), help="Output directory")
    parser.add_argument("--cell-type-key", default="cell_type", help="obs column with cell-type labels")
    parser.add_argument(
        "--max-genes",
        type=int,
        default=DEFAULT_MAX_GENES,
        help="Keep the N shared genes with the highest reference variance for NNLS (0 = every shared gene)",
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="Leave out reference cells whose label is missing instead of refusing the run",
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
    error_exc = None
    with _redirect_stdout_to_stderr():
        try:
            result = _run_bulk2space(
                sc_h5ad_path=args.sc_h5ad,
                spatial_h5ad_path=args.spatial_h5ad,
                output_dir=args.output_dir,
                cell_type_key=args.cell_type_key,
                max_genes=args.max_genes,
                drop_unlabeled=args.drop_unlabeled,
            )
        except Exception as e:
            _log(f"ERROR: {e}")
            traceback.print_exc(file=sys.stderr)
            error_info = str(e)
            error_exc = e

    # stdout: JSON only (must be outside redirect block)
    if error_info is not None:
        WorkerOutput.emit_error("bulk2space", error_info, task="deconvolution", exc=error_exc)
        sys.exit(1)

    print(json.dumps(result, default=str))
    sys.stdout.flush()


if __name__ == "__main__":
    _cli_main()
