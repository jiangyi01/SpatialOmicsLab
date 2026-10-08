#!/usr/bin/env python
"""
scresolve_worker.py

Worker behind the ``run_scresolve`` portal.

**What runs is not upstream scResolve.** scResolve (chenhcs/scResolve, an xfuse fork) recovers
single-cell expression from a Visium/ST slide *together with its paired histology image*. This
portal is given two h5ad files and no image, so that model cannot run here, and it never did: the
old code imported ``scresolve.run`` only to set a flag that was ``False`` on both branches. What
runs -- the tool's only implementation, so it is the method rather than a fallback -- is
reference marker-gene scoring at the input spot resolution:

1. Markers per reference cell type: ``scanpy.tl.rank_genes_groups`` (Wilcoxon, top 20) on the
   genes shared with the spatial data, after ``normalize_total(1e4)`` + ``log1p``, with
   ``use_raw=False`` so ``adata.raw`` never replaces the matrix that was just normalised.
2. One ``score_<cell type>`` obs column per cell type on the spatial spots
   (``scanpy.tl.score_genes``, same normalisation, ``use_raw=False``).

The output keeps exactly one row per input spot: no resolution enhancement happens (ratio 1.0).
Spots whose ``obs['in_tissue']`` is 0 (background glass, carried by CELLxGENE Visium exports) are
left out first and counted (``params.in_tissue_filter``), as every spot-scoring worker does.
The file keeps its historical name ``scresolve_enhanced.h5ad`` and payload key ``enhanced_h5ad``
because other code keys on them; the payload says what the file holds.

When the spatial genes are Ensembl IDs and the reference's are symbols, the spatial var_names are
mapped through the first gene-symbol column the object has (``SYMBOL``, then CELLxGENE's
``feature_name``, then ``gene_symbols`` / ``gene_name`` / ``GeneName``), published as
``params.gene_symbol_column``.

No source tree is fetched: the worker used to ``git clone`` scResolve from GitHub at run time
(120 s timeout, ``check=True``) for a model it never ran.

- Executed inside the scresolve conda env: /opt/conda/envs/scresolve
- All logs and progress go to stderr.
- Stdout contains exactly one line of JSON at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import (
    WorkerOutput,
    describe_reduction,
    drop_unlabeled,
    id_mismatch_msg,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    preflight_check,
    record_in_tissue,
    record_method,
)

#: Markers taken per reference cell type (``rank_genes_groups(n_genes=...)``).
N_MARKERS = 20

#: The method that actually runs, published as ``params.method``. It is the tool's only
#: implementation, so ``params.used_fallback`` is always False.
METHOD_NAME = (
    "reference marker-gene scoring at the input spot resolution "
    "(scanpy rank_genes_groups wilcoxon top-20 markers per cell type + score_genes); "
    "upstream scResolve super-resolution not run, enhancement ratio 1.0"
)

#: Where the per-spot scores' provenance is recorded in the output h5ad (an added key).
UNS_KEY = "scresolve_marker_scoring"

#: var columns that hold gene symbols beside Ensembl var_names, in the order they are tried.
#: ``SYMBOL`` (Space Ranger-derived objects) keeps its old priority; ``feature_name`` is where the
#: CELLxGENE exports keep them -- it was missing, so every CELLxGENE slide with a symbol-keyed
#: reference stopped with "No matching genes". The rest are the other names harmonize_gene_ids tries.
SYMBOL_COLUMNS = ("SYMBOL", "feature_name", "gene_symbols", "gene_name", "GeneName")

#: An Ensembl gene ID of any species: ENSG (human), ENSMUSG (mouse), ENSDARG (zebrafish), ...
_ENSEMBL_GENE = re.compile(r"^ENS[A-Z]*G\d{6,}")


def log(msg: str) -> None:
    sys.stderr.write(f"[scresolve-worker] {msg}\n")
    sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "run_scresolve worker: reference marker-gene scoring at the input spot resolution "
            "(upstream scResolve is not run)."
        )
    )
    parser.add_argument("--spatial-h5ad", type=str, required=True, help="Path to spatial AnnData (.h5ad).")
    parser.add_argument("--sc-h5ad", type=str, required=True, help="Path to scRNA-seq AnnData (.h5ad).")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--cell-type-key", type=str, default="cell_type", help="obs column for cell types.")
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        default=False,
        help="Drop reference cells whose label is missing (NaN/empty) instead of stopping.",
    )
    return parser.parse_args()


def _shared_genes(st_names, sc_names) -> list:
    """Genes present in both objects, in the spatial object's order.

    The old ``list(set(a) & set(b))`` ordered them by string hash, which changes with
    ``PYTHONHASHSEED``; ``score_genes`` draws its control genes from bins in that order, so the
    same inputs gave different scores from one process to the next.
    """
    sc_set = {str(g) for g in sc_names}
    return [str(g) for g in st_names if str(g) in sc_set]


def _looks_ensembl(names) -> bool:
    """True when most of the first 20 identifiers are Ensembl gene IDs (any species)."""
    head = [str(n) for n in list(names)[:20]]
    return bool(head) and sum(bool(_ENSEMBL_GENE.match(n)) for n in head) * 2 > len(head)


def _valid_symbols(values):
    """``(symbols as str, mask of usable ones)``: empty, NaN and None are not symbols."""
    import numpy as np

    symbols = np.asarray(values).astype(str)
    stripped = np.char.strip(symbols)
    valid = (stripped != "") & (stripped != "nan") & (stripped != "None")
    return symbols, valid


def _symbol_column(adata) -> str | None:
    """The first of ``SYMBOL_COLUMNS`` in ``adata.var`` that holds at least one usable symbol."""
    for col in SYMBOL_COLUMNS:
        if col in adata.var.columns and _valid_symbols(adata.var[col].values)[1].any():
            return col
    return None


def _labelled_reference(adata_sc, cell_type_key: str, allow_drop: bool):
    """The reference restricted to labelled cells, with the label as a string categorical.

    Returns ``(adata_sc, cell_types, n_dropped)``. A NaN label is not a class: by default it
    stops the run (``drop_unlabeled``); scanpy used to exclude those cells silently while the
    payload still counted "nan" as a cell type. Integer labels are cast to strings (scanpy
    crashed with "Can only use .cat accessor").
    """
    import pandas as pd

    if cell_type_key not in adata_sc.obs.columns:
        raise ValueError(
            f"Cell type key '{cell_type_key}' not found in scRNA-seq obs. Available: {list(adata_sc.obs.columns)}"
        )
    keep, n_dropped = drop_unlabeled(adata_sc.obs[cell_type_key].values, allow_drop=allow_drop, what="reference cells")
    if n_dropped:
        adata_sc = adata_sc[keep].copy()
        log(f"Dropped {n_dropped} reference cells with no '{cell_type_key}' label (drop_unlabeled=True)")
    labels = pd.Categorical(adata_sc.obs[cell_type_key].astype(str).values)
    adata_sc.obs[cell_type_key] = labels
    counts = pd.Series(labels).value_counts()
    cell_types = [str(c) for c in labels.categories]
    if len(cell_types) < 2:
        raise ValueError(
            f"The reference has {len(cell_types)} labelled cell type(s) in '{cell_type_key}'; marker genes are "
            "ranked one type against the rest, which needs at least 2."
        )
    small = sorted(str(k) for k, v in counts.items() if v < 2)
    if small:
        raise ValueError(
            f"Cell type(s) {small} have fewer than 2 reference cells in '{cell_type_key}'; the Wilcoxon marker "
            "test needs at least 2 per type. Merge or relabel them in the reference."
        )
    return adata_sc, cell_types, int(n_dropped)


def _score_columns(cell_types) -> tuple[dict, list]:
    """``{cell type: obs column}`` and the list of renames forced by a collision.

    ``/`` and ``\\`` are replaced by ``_`` because h5py reads them as a path. That can map two
    types onto one column (``A/B`` and ``A_B``); the second used to overwrite the first in silence.
    A colliding name now gets a numeric suffix and the rename is reported.
    """
    columns: dict = {}
    used: set = set()
    collided: list = []
    for ct in cell_types:
        base = "score_" + str(ct).replace("/", "_").replace("\\", "_")
        col = base
        k = 2
        while col in used:
            col = f"{base}_{k}"
            k += 1
        if col != base:
            collided.append((str(ct), col))
        used.add(col)
        columns[ct] = col
    return columns, collided


def _x_looks_like_counts(X) -> bool:
    """True when every stored value is a non-negative integer (read in row blocks, not densified)."""
    import numpy as np
    import scipy.sparse as sp

    if sp.issparse(X):
        blocks = [np.asarray(X.data)]
    else:
        arr = np.asarray(X)
        blocks = (arr[i : i + 2048] for i in range(0, arr.shape[0], 2048))
    for block in blocks:
        if block.size and (np.any(block < 0) or np.any(np.abs(block - np.rint(block)) > 1e-6)):
            return False
    return True


def _atomic_write_h5ad(adata, path: str) -> None:
    """Write ``<path>.partial`` and rename it over ``path`` once complete."""
    tmp = path + ".partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def run_scresolve_pipeline(
    spatial_h5ad: str,
    sc_h5ad: str,
    output_dir: str,
    cell_type_key: str,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """Score every spatial spot for each reference cell type's marker signature.

    Upstream scResolve is not run (it needs the slide's histology image); see the module docstring.
    """
    import matplotlib
    import numpy as np
    import scanpy as sc

    matplotlib.use("Agg")

    preflight_check(
        inputs={"spatial_h5ad": spatial_h5ad, "sc_h5ad": sc_h5ad},
        output_dir=output_dir,
    )

    warnings: list = []

    # Load data
    log(f"Loading spatial AnnData from {spatial_h5ad}")
    adata_st = sc.read_h5ad(spatial_h5ad)
    # Background spots (obs['in_tissue'] == 0) are glass, not tissue: they are not scored.
    adata_st, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata_st, "spots")
    if n_spots_off_tissue:
        log(f"Left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0 (background)")
    renamed_st = make_names_unique_and_report(adata_st)
    n_genes_st_input = int(adata_st.n_vars)

    log(f"Loading scRNA-seq AnnData from {sc_h5ad}")
    adata_sc = sc.read_h5ad(sc_h5ad)
    renamed_sc = make_names_unique_and_report(adata_sc)

    # Harmonize gene names: spatial Ensembl IDs against a symbol-keyed reference are mapped through
    # the spatial object's gene-symbol column (SYMBOL_COLUMNS; CELLxGENE keeps them in feature_name).
    n_genes_without_symbol = 0
    symbol_column = None
    if _looks_ensembl(adata_st.var_names) and not _looks_ensembl(adata_sc.var_names):
        symbol_column = _symbol_column(adata_st)
        if symbol_column is not None:
            log(f"Mapping spatial Ensembl IDs to gene symbols from var[{symbol_column!r}]")
            symbols, valid = _valid_symbols(adata_st.var[symbol_column].values)
            n_genes_without_symbol = int((~valid).sum())
            adata_st = adata_st[:, valid].copy()
            adata_st.var_names = symbols[valid]
            # The pass most likely to invent a name: two Ensembl IDs mapping to one symbol.
            make_names_unique_and_report(adata_st, into=renamed_st)

    n_spots = int(adata_st.n_obs)
    n_genes_st = int(adata_st.n_vars)
    log(f"Spatial data: n_spots={n_spots}, n_genes={n_genes_st}")
    n_cells = int(adata_sc.n_obs)
    n_genes_sc = int(adata_sc.n_vars)
    log(f"scRNA-seq data: n_cells={n_cells}, n_genes={n_genes_sc}")
    if n_spots == 0:
        raise ValueError(f"The spatial data at {spatial_h5ad} has no spots (obs is empty).")

    adata_sc, cell_types, n_unlabeled_dropped = _labelled_reference(adata_sc, cell_type_key, drop_unlabeled)
    n_celltypes = len(cell_types)
    n_cells_used = int(adata_sc.n_obs)
    log(f"Found {n_celltypes} cell types: {cell_types[:10]}...")

    log("Scoring reference marker signatures per spot (upstream scResolve is not run; no resolution enhancement)")

    common_genes = _shared_genes(adata_st.var_names, adata_sc.var_names)
    n_shared = len(common_genes)
    log(f"Common genes: {n_shared}")
    if n_shared < 10:
        raise ValueError(
            id_mismatch_msg(
                "genes",
                "spatial",
                adata_st.var_names,
                "scRNA-seq",
                adata_sc.var_names,
                n_common=n_shared,
            )
            + " Cannot score marker signatures."
        )

    # Subset to common genes
    adata_st_sub = adata_st[:, common_genes].copy()
    adata_sc_sub = adata_sc[:, common_genes].copy()

    for label, sub in (("spatial", adata_st_sub), ("scRNA-seq reference", adata_sc_sub)):
        if not _x_looks_like_counts(sub.X):
            warnings.append(
                f"The {label} X holds non-integer or negative values, so it does not look like raw counts; it "
                "was normalised (normalize_total 1e4) and log1p-transformed again, as for counts."
            )

    # Normalize
    sc.pp.normalize_total(adata_sc_sub, target_sum=1e4)
    sc.pp.log1p(adata_sc_sub)

    # Markers per cell type, on the normalised shared genes -- never on adata.raw, which scanpy
    # would otherwise pick by default and which holds other genes and unnormalised counts.
    sc.tl.rank_genes_groups(adata_sc_sub, groupby=cell_type_key, method="wilcoxon", n_genes=N_MARKERS, use_raw=False)
    ranked = adata_sc_sub.uns["rank_genes_groups"]["names"]
    shared_set = set(common_genes)
    marker_dict: dict = {}
    for ct in cell_types:
        marker_dict[ct] = [str(g) for g in ranked[ct][:N_MARKERS] if str(g) in shared_set]
    log(f"Found marker genes for {sum(1 for m in marker_dict.values() if m)} cell types")

    # Score spatial data with cell type signatures
    sc.pp.normalize_total(adata_st_sub, target_sum=1e4)
    sc.pp.log1p(adata_st_sub)

    columns, collided = _score_columns(cell_types)
    for ct, col in collided:
        warnings.append(f"cell type {ct!r} would share a score column with another type; it was written to {col!r}")
    overwritten = [columns[ct] for ct in cell_types if columns[ct] in adata_st.obs.columns]
    if overwritten:
        warnings.append(f"the spatial input already had obs column(s) {overwritten}; they were overwritten")

    scored: list = []
    unscored: list = []
    nan_scored: list = []
    for ct in cell_types:
        markers = marker_dict[ct]
        if not markers:
            unscored.append(ct)
            continue
        try:
            sc.tl.score_genes(adata_st_sub, gene_list=markers, score_name=columns[ct], use_raw=False)
        except (IndexError, RuntimeError) as exc:
            # scanpy draws control genes from expression bins of the non-marker genes; on a small
            # shared panel no bin has any, and 1.9.x fails with a bare IndexError about indices.
            raise ValueError(
                f"scanpy score_genes could not score cell type {ct!r}: {len(markers)} markers on only {n_shared} "
                f"shared genes leave no control genes to compare against ({type(exc).__name__}: {exc}). "
                "Use a spatial object and reference that share more genes."
            ) from exc
        values = adata_st_sub.obs[columns[ct]].values
        if not np.isfinite(np.asarray(values, dtype=float)).any():
            nan_scored.append(ct)
        adata_st.obs[columns[ct]] = values
        scored.append(ct)
    if unscored:
        warnings.append(f"no marker genes were found for cell type(s) {unscored}; they have no score column")
    if nan_scored:
        warnings.append(f"the score of cell type(s) {nan_scored} is NaN on every spot")
    score_cols = [columns[ct] for ct in scored]

    adata_st.uns[UNS_KEY] = {
        "method": METHOD_NAME,
        "cell_type_key": cell_type_key,
        "cell_types": [str(ct) for ct in scored],
        "score_columns": score_cols,
        "markers": {columns[ct]: list(marker_dict[ct]) for ct in scored},
    }

    # Save outputs (historical file name; it holds the input spots plus the score columns)
    enhanced_path = os.path.join(output_dir, "scresolve_enhanced.h5ad")
    _atomic_write_h5ad(adata_st, enhanced_path)
    n_enhanced = int(adata_st.n_obs)
    n_genes_out = int(adata_st.n_vars)
    log(f"Saved spot-level AnnData with marker scores: n_obs={n_enhanced}, n_vars={n_genes_out}")

    out = WorkerOutput("scresolve", task="resolution")
    out.set_data(
        n_spots=int(n_spots),
        n_spots_supplied=int(n_spots_supplied),
        n_genes_spatial=int(n_genes_st),
        n_cells=int(n_cells),
        n_genes_sc=int(n_genes_sc),
    )
    out.add_output_files(
        {
            "enhanced_h5ad": enhanced_path,
        }
    )
    out.add_params(
        {
            "cell_type_key": cell_type_key,
            "n_celltypes": n_celltypes,
            "drop_unlabeled": bool(drop_unlabeled),
            "n_unlabeled_dropped": int(n_unlabeled_dropped),
            "n_markers_per_type": N_MARKERS,
            "n_genes_without_symbol_dropped": int(n_genes_without_symbol),
            "gene_symbol_column": symbol_column,
        }
    )
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue, "spots")
    record_method(out, METHOD_NAME, used_fallback=False)
    out.add_params(identifier_rename_params(renamed_st))
    out.add_params(identifier_rename_params(renamed_sc, suffix="sc"))
    out.set_summary(
        n_enhanced_obs=int(n_enhanced),
        n_output_obs=int(n_enhanced),
        n_genes_output=int(n_genes_out),
        n_celltypes=int(n_celltypes),
        n_celltypes_scored=len(scored),
        n_reference_cells_used=int(n_cells_used),
        n_shared_genes=int(n_shared),
        score_columns=score_cols,
        resolution_enhanced=False,
        enhancement_ratio=round(n_enhanced / n_spots, 2),
    )
    out.add_warnings(warnings)
    out.set_analysis(
        f"Marker-gene scoring, not scResolve: {n_spots} spots were scored against {len(scored)} of {n_celltypes} "
        f"reference cell-type signatures (top {N_MARKERS} Wilcoxon markers per type, from {n_cells_used} "
        f"reference cells on the {n_shared} genes shared with the spatial data). The scores are the obs columns "
        f"'score_<cell type>' of scresolve_enhanced.h5ad. No resolution enhancement was performed: the output keeps "
        f"one row per input spot (ratio 1.0). Upstream scResolve super-resolves a slide from its paired histology "
        f"image, which this tool does not take, and was not run."
        + (
            f" {n_spots_off_tissue} of the {n_spots_supplied} spots supplied have obs['in_tissue'] == 0 "
            "(background) and were left out; they are not in the output."
            if n_spots_off_tissue
            else ""
        )
        + (
            f" Spatial Ensembl IDs were mapped to gene symbols from var[{symbol_column!r}]."
            if symbol_column is not None
            else ""
        )
        + (
            f" {n_unlabeled_dropped} reference cells with no '{cell_type_key}' label were dropped (drop_unlabeled)."
            if n_unlabeled_dropped
            else ""
        )
        + (f" Cell types with no score column (no markers): {unscored}." if unscored else "")
        + (
            describe_reduction(
                "spatial genes",
                n_genes_st_input,
                n_genes_st,
                f"the Ensembl-to-symbol mapping (no symbol in var[{symbol_column!r}] for the id); they are also "
                "absent from the output h5ad",
            )
            if n_genes_without_symbol
            else ""
        )
        + identifier_rename_note(renamed_st, subject="spatial data")
        + identifier_rename_note(renamed_sc, subject="scRNA-seq reference")
    )
    return out.to_dict()


def main() -> None:
    args = parse_args()

    orig_stdout = sys.stdout
    sys.stdout = sys.stderr
    error_exc = None
    try:
        try:
            result = run_scresolve_pipeline(
                spatial_h5ad=args.spatial_h5ad,
                sc_h5ad=args.sc_h5ad,
                output_dir=args.output_dir,
                cell_type_key=args.cell_type_key,
                drop_unlabeled=args.drop_unlabeled,
            )
        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc(file=sys.stderr)
            result = None
            error_msg = str(e)
            error_exc = e
    finally:
        sys.stdout = orig_stdout

    if result is None:
        WorkerOutput.emit_error("scresolve", error_msg, task="resolution", exc=error_exc)
        sys.exit(1)
    else:
        print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
