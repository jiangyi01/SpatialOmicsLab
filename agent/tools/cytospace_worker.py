#!/usr/bin/env python
"""
cytospace_worker.py

Worker script for CytoSPACE cell-to-spot assignment.

- Executed inside the cytospace conda env: /opt/conda/envs/cytospace_env
- All logs and progress go to stderr.
- Stdout contains exactly one line of JSON at the end.

NOTE: CytoSPACE expects tab-delimited text files as input, not h5ad.
This worker converts h5ad to the required format before running.

Two choices CytoSPACE makes are made here instead, and the payload names both:

* **The tissue-wide cell-type composition.** CytoSPACE fixes how many cells of each type it places
  from a fractions table (``-ctfep``). Upstream estimates that table from the spatial data with a
  Seurat R script; this environment has no R, so this wrapper has never run that step. By default
  the table is the scRNA-seq reference's own label frequencies, which makes the assigned
  composition an input equal to the reference's, not a finding. ``cell_type_fractions_path``
  supplies fractions estimated from the slide instead (for example from a deconvolution run).
* **Cells per location.** Without ``--single-cell`` CytoSPACE treats every location as a
  multi-cell Visium spot and places about ``mean_cell_numbers`` (upstream default 5) cells in
  each. ``single_cell=True`` places exactly one per location, which is what segmented-cell and
  <=8 um bin data need.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    describe_reduction,
    drop_unlabeled,
    id_mismatch_msg,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    preflight_check,
    record_ignored,
    record_in_tissue,
    record_method,
    sanitize_cell_type_names,
    sniff_tabular_sep,
    spatial_coords,
)

#: Upstream ``--mean-cell-numbers`` default ("appropriate for Visium"), and the portal's default.
UPSTREAM_MEAN_CELL_NUMBERS = 5
#: The limit this worker has always put on the CytoSPACE subprocess; ``--timeout-s`` now reaches it.
DEFAULT_TIMEOUT_S = 1100
#: Upstream ``--number-of-selected-spots`` and ``--number-of-processors`` defaults: in single-cell
#: mode the spots are solved in partitions of this many, this many partitions at a time.
_UPSTREAM_SPOTS_PER_PARTITION = 10000
_UPSTREAM_PROCESSORS = 4
#: Where the reference-frequency default comes from, in the payload's own words.
FRACTIONS_FROM_REFERENCE = "reference_label_frequency"
FRACTIONS_SUPPLIED = "user_supplied"
#: Visium HD bin barcodes, e.g. ``s_008um_00301_00321-1``.
_VISIUM_HD_BIN = re.compile(r"^s_0*(\d+)um_")
#: obs columns only a per-cell segmentation writes (Xenium, CosMx, MERSCOPE).
_SEGMENTATION_COLUMNS = ("cell_area", "nucleus_area", "x_centroid", "center_x", "CenterX_global_px")
#: var columns holding gene symbols beside ENSEMBL var_names, first match wins (CELLxGENE: feature_name).
_SPATIAL_SYMBOL_COLUMNS = ("SYMBOL", "feature_name", "gene_symbols", "gene_name")
#: Column names that hold spot identifiers in a proportions table, wherever the column sits (the
#: inspector's list, spark/spotlight's R spellings). SPOTlight and CARD write theirs LAST.
_SPOT_ID_COLUMNS = ("spot", "barcode", "barcodes", "spot_id", "spotid", "cell", "cell_id", "cellid", "index")


def log(msg: str) -> None:
    sys.stderr.write(f"[cytospace-worker] {msg}\n")
    sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CytoSPACE worker: cell-to-spot assignment.")
    parser.add_argument("--sc-h5ad", type=str, required=True, help="Path to scRNA-seq AnnData (.h5ad).")
    parser.add_argument("--spatial-h5ad", type=str, required=True, help="Path to spatial AnnData (.h5ad).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for outputs.")
    parser.add_argument("--cell-type-key", type=str, default="cell_type", help="obs column for cell types.")
    parser.add_argument(
        "--n-cells",
        type=int,
        default=0,
        help="Optional cap on reference cells. 0 (default) stages every cell; a positive value keeps that "
        "many, drawn at random with --seed.",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=5000,
        help="Match on the top N shared genes ranked by raw spatial variance (0 = every shared gene).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--single-cell",
        action="store_true",
        help="Place exactly one reference cell per location (CytoSPACE --single-cell), for segmented-cell or "
        "<=8 um bin data. Default: Visium spot mode.",
    )
    parser.add_argument(
        "--mean-cell-numbers",
        type=int,
        default=UPSTREAM_MEAN_CELL_NUMBERS,
        help="Spot mode only: mean number of cells per location (CytoSPACE -mcn; 5 suits Visium).",
    )
    parser.add_argument(
        "--cell-type-fractions-path",
        type=str,
        default="",
        help="Table of cell-type fractions estimated from the spatial data. Empty (default): the reference's "
        "own label frequencies are used.",
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="Leave out reference cells whose label is missing instead of stopping.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help="Seconds the CytoSPACE subprocess may run (0 = no limit).",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------- staging helpers


def _atomic_to_csv(frame, path: str, **kwargs) -> None:
    """``frame.to_csv(path)`` through ``path.partial`` + ``os.replace``: a killed run leaves no half file."""
    partial = path + ".partial"
    frame.to_csv(partial, **kwargs)
    os.replace(partial, path)


def _write_features_by_obs_tsv(X, obs_names, var_names, path: str, max_block_bytes: int = 256 << 20) -> None:
    """Write ``X`` (obs x features) as the features x obs TSV CytoSPACE reads, a block of features at a time.

    Byte-for-byte what ``pd.DataFrame(X.toarray(), index=obs, columns=var).T.to_csv(sep="\\t")``
    wrote, without ever holding the whole matrix dense: only ``block x n_obs`` values exist at once.
    A 280,000-cell reference over 18,000 genes is ~20 GB dense in float32; the text file it becomes
    is intrinsic to CytoSPACE's input format, a dense copy of it in this process is not.
    """
    import numpy as np
    import pandas as pd
    from scipy.sparse import issparse

    n_obs, n_var = X.shape
    if issparse(X):
        X = X.tocsc()
    step = max(1, int(max_block_bytes // (max(1, n_obs) * 8)))
    columns = pd.Index(obs_names)
    partial = path + ".partial"
    with open(partial, "w") as fh:
        start = 0
        while True:
            stop = min(n_var, start + step)
            block = X[:, start:stop]
            block = block.toarray() if issparse(block) else np.asarray(block)
            frame = pd.DataFrame(block.T, index=var_names[start:stop], columns=columns)
            frame.to_csv(fh, sep="\t", header=(start == 0))
            start = stop
            if start >= n_var:
                break
    os.replace(partial, path)


def _column_variance(X):
    """Population variance (ddof=0, as ``np.var``) of every column, in float64, without densifying.

    Sparse: sums of x and x^2 per column from the stored entries. Dense: ``np.var`` over column
    blocks, so the temporary is one block rather than a second copy of the matrix.
    """
    import numpy as np
    from scipy.sparse import issparse

    n_rows, n_cols = X.shape
    if n_rows == 0:
        return np.zeros(n_cols, dtype=np.float64)
    if issparse(X):
        X = X.tocsr()
        if not X.has_canonical_format:
            X = X.copy()
            X.sum_duplicates()
        data = np.asarray(X.data, dtype=np.float64)
        s1 = np.bincount(X.indices, weights=data, minlength=n_cols)
        s2 = np.bincount(X.indices, weights=data * data, minlength=n_cols)
        mean = s1 / n_rows
        return np.maximum(s2 / n_rows - mean * mean, 0.0)
    X = np.asarray(X)
    out = np.empty(n_cols, dtype=np.float64)
    step = max(1, int((64 << 20) // (n_rows * 8)))
    for start in range(0, n_cols, step):
        out[start : start + step] = np.var(X[:, start : start + step], axis=0, dtype=np.float64)
    return out


def _single_cell_resolution_hint(adata_st) -> str:
    """Why the slide looks like one cell per location, or "" when nothing says so."""
    names = [str(n) for n in adata_st.obs_names[:50]]
    matches = [_VISIUM_HD_BIN.match(n) for n in names]
    if names and all(matches):
        bin_um = int(matches[0].group(1))
        if bin_um <= 8:
            return f"its obs_names are Visium HD {bin_um} um bins (e.g. '{names[0]}')"
    seg = [c for c in _SEGMENTATION_COLUMNS if c in adata_st.obs.columns]
    if seg:
        return f"its obs carries per-cell segmentation columns ({', '.join(seg)})"
    return ""


def _load_cytospace_inputs(
    sc_h5ad: str,
    spatial_h5ad: str,
    cell_type_key: str,
    n_cells: int,
    n_top_genes: int = 5000,
    seed: int = 0,
    drop_unlabeled_cells: bool = False,
) -> dict[str, Any]:
    """Load both inputs and make every cut CytoSPACE's input will carry, in memory: nothing is written.

    Returns the cut AnnData objects, the label and coordinate frames, and the counts the payload and
    the memory check need. Everything :func:`check_cytospace_memory` reads is known here, so the
    pipeline can refuse a run that cannot fit before it converts tens of GB to text: on a Visium HD
    slide in spot mode the refusal is certain from the shapes alone, and it used to come only after
    ~7.5e9 values had been written out (most of an hour).

    Four reductions can run before CytoSPACE sees anything: spatial locations with
    ``obs['in_tissue'] == 0`` (background glass) are left out, reference cells with no label are
    dropped (only with ``drop_unlabeled_cells``; otherwise they stop the run), the reference is
    subsampled to ``n_cells`` (only when the caller sets a positive cap -- by default every cell is
    staged), and the shared panel is cut to the top ``n_top_genes`` by raw spatial variance. Each
    dimension therefore travels back twice -- ``n_*`` is what the user supplied and ``n_*_used`` is
    what CytoSPACE actually saw. Reporting only the survivors would describe a smaller experiment
    than the one that was asked for.

    The reference is staged with all of its genes, before the panel cut: CytoSPACE matches on the
    genes it shares with the (cut) spatial table, and writes each assigned cell's expression over
    every gene of the staged reference. ``n_genes_matched`` is the panel the match used.

    ``seed`` governs the optional reference subsample. It is the same seed the solver is given, so a
    user varying it varies the whole run and not just its last stage.
    """
    import numpy as np
    import pandas as pd
    import scanpy as sc

    log("Loading the h5ad inputs...")
    if n_cells < 0:
        raise ValueError(
            f"n_cells={n_cells} is not a cell count. Pass 0 (the default) to stage every reference cell, "
            "or a positive cap."
        )

    # --- scRNA-seq reference ---
    adata_sc = sc.read_h5ad(sc_h5ad)
    renamed_sc = make_names_unique_and_report(adata_sc)
    # Read before any cut, so the payload can name the reference the user actually handed in.
    n_sc_cells_supplied = int(adata_sc.n_obs)
    n_sc_genes_supplied = int(adata_sc.n_vars)
    log(f"Loaded scRNA-seq: n_cells={adata_sc.n_obs}, n_genes={adata_sc.n_vars}")

    if cell_type_key not in adata_sc.obs.columns:
        raise ValueError(
            f"Cell type key '{cell_type_key}' not found in scRNA-seq obs. Available: {list(adata_sc.obs.columns)}"
        )

    # A missing label is not a class: written out it becomes an empty field (or the string "nan"),
    # which CytoSPACE either samples as a cell type of its own or silently never places.
    keep, n_unlabeled = drop_unlabeled(
        adata_sc.obs[cell_type_key].values, drop_unlabeled_cells, what=f"reference cells (obs['{cell_type_key}'])"
    )
    if n_unlabeled:
        log(f"Dropping {n_unlabeled} reference cells with no '{cell_type_key}' label (drop_unlabeled=True)")
        adata_sc = adata_sc[keep].copy()

    # Optional cap. Off by default: every cell the user supplied is a candidate for placement.
    if n_cells > 0 and adata_sc.n_obs > n_cells:
        log(f"Subsampling scRNA-seq from {adata_sc.n_obs} to {n_cells} cells (seed={seed}), as n_cells asks")
        # Draw from the caller's seed, not a literal: --seed is advertised as governing
        # reproducibility, and this draw decides which cells the solver is ever shown.
        rng = np.random.RandomState(seed)
        idx = rng.choice(adata_sc.n_obs, n_cells, replace=False)
        adata_sc = adata_sc[idx].copy()
    n_sc_genes_staged = int(adata_sc.n_vars)

    # Cell type labels: cell_barcode -> cell_type
    labels_df = adata_sc.obs[[cell_type_key]].copy()
    labels_df.columns = ["CellType"]

    # --- Spatial data ---
    adata_st = sc.read_h5ad(spatial_h5ad)
    renamed_st = make_names_unique_and_report(adata_st)
    # Same reason as the reference above: the background, ENSEMBL and panel cuts all follow.
    n_st_spots_supplied = int(adata_st.n_obs)
    n_st_genes_supplied = int(adata_st.n_vars)
    log(f"Loaded spatial: n_spots={adata_st.n_obs}, n_genes={adata_st.n_vars}")
    # Background glass is not a location a cell can be placed in. CELLxGENE Visium exports carry
    # every array spot with obs['in_tissue'] 0/1 (56-70% background on the library's samples); spot
    # mode placed ~5 reference cells in each of them.
    adata_st, _, n_st_spots_off_tissue = keep_in_tissue(adata_st, what="spots")
    if n_st_spots_off_tissue:
        log(
            f"Left out {n_st_spots_off_tissue} of {n_st_spots_supplied} spots with obs['in_tissue'] == 0 "
            f"(background); {adata_st.n_obs} remain"
        )
    resolution_hint = _single_cell_resolution_hint(adata_st)

    # Spatial coordinates, read before anything is staged: a slide without them is refused here.
    if "spatial" in adata_st.obsm:
        coords, _ = spatial_coords(adata_st, "spatial", want=2, tool="CytoSPACE")
        coords_df = pd.DataFrame(coords, index=adata_st.obs_names, columns=["x", "y"])
    else:
        # See the note in bulk2space_worker: this used to fabricate y = x in barcode order and
        # keep going. st_coordinates.txt is a required CytoSPACE input and its assignment of cells
        # to spots is coordinate-aware, so the fabricated line becomes the answer, and the warning
        # is invisible to the model on a run that exits 0.
        raise ValueError(
            f"Spot coordinates not found: adata.obsm['spatial'] is missing. "
            f"Available obsm keys: {list(adata_st.obsm.keys())}"
        )

    # Harmonize gene names: if spatial uses ENSEMBL and carries a symbol column, convert. CELLxGENE
    # files keep the symbols in var['feature_name'], not var['SYMBOL'].
    spatial_symbol_column = ""
    symbol_col = next((c for c in _SPATIAL_SYMBOL_COLUMNS if c in adata_st.var.columns), None)
    if symbol_col is not None:
        st_is_ensembl = any(str(n).startswith("ENSG") for n in adata_st.var_names[:20])
        sc_is_symbol = not any(str(n).startswith("ENSG") for n in adata_sc.var_names[:20])
        if st_is_ensembl and sc_is_symbol:
            log(f"Mapping spatial ENSEMBL IDs to gene symbols from var['{symbol_col}'] for compatibility")
            spatial_symbol_column = symbol_col
            symbols = adata_st.var[symbol_col].values.astype(str)
            valid = (symbols != "") & (symbols != "nan") & (symbols != "None")
            adata_st = adata_st[:, valid].copy()
            adata_st.var_names = adata_st.var[symbol_col].values.astype(str)
            # The pass most likely to invent a name: two ENSEMBL IDs mapping to one symbol.
            make_names_unique_and_report(adata_st, into=renamed_st)
            log(f"  After mapping: n_genes={adata_st.n_vars}")

    # sorted(), not list(): top_idx indexes back into this list, so an unordered set would
    # let a variance tie at the cut-off resolve differently between runs of the same seed.
    common_genes = sorted(set(adata_sc.var_names) & set(adata_st.var_names))
    if not common_genes:
        # CytoSPACE would otherwise run on an empty gene panel and still exit 0, with an assignment
        # no gene informed (its log: "Number of genes used for mapping: 0").
        raise ValueError(
            id_mismatch_msg("genes", "scRNA-seq reference", adata_sc.var_names, "spatial data", adata_st.var_names)
        )
    n_sc_genes_used = int(adata_sc.n_vars)
    n_genes_matched = len(common_genes)

    # --- Cut the shared panel to the top genes by raw spatial variance (reduces the LP problem) ---
    if n_top_genes > 0:
        if len(common_genes) > n_top_genes:
            log(f"Subsetting from {len(common_genes)} common genes to top {n_top_genes} by raw spatial variance")
            # Rank on the spatial data (most informative for the assignment). The variance is taken
            # on the stored (sparse) matrix: densifying every shared gene to rank them used to cost
            # more memory than the whole CytoSPACE run on a Visium HD slide.
            positions = adata_st.var_names.get_indexer(common_genes)
            gene_var = _column_variance(adata_st.X)[positions]
            top_idx = np.argsort(gene_var, kind="stable")[-n_top_genes:]
            top_genes = [common_genes[i] for i in top_idx]
            n_sc_genes_used = int(adata_sc.var_names.isin(top_genes).sum())
            adata_st = adata_st[:, adata_st.var_names.isin(top_genes)].copy()
            n_genes_matched = len(top_genes)
            log(f"  After subsetting: sc={n_sc_genes_used} genes, st={adata_st.n_vars} genes")
        else:
            log(f"Only {len(common_genes)} common genes (<= {n_top_genes}), skipping gene subsetting")

    return {
        "adata_sc": adata_sc,
        "adata_st": adata_st,
        "labels_df": labels_df,
        "coords_df": coords_df,
        # Supplied, then analysed.
        "n_sc_cells": n_sc_cells_supplied,
        "n_sc_cells_used": int(adata_sc.n_obs),
        "n_sc_cells_unlabeled": int(n_unlabeled),
        "n_sc_genes": n_sc_genes_supplied,
        "n_sc_genes_used": n_sc_genes_used,
        "n_sc_genes_staged": n_sc_genes_staged,
        "n_st_spots": n_st_spots_supplied,
        "n_st_spots_used": int(adata_st.n_obs),
        "n_st_spots_off_tissue": int(n_st_spots_off_tissue),
        "n_st_genes": n_st_genes_supplied,
        "n_st_genes_used": int(adata_st.n_vars),
        "n_genes_matched": int(n_genes_matched),
        "resolution_hint": resolution_hint,
        "spatial_symbol_column": spatial_symbol_column,
        # The staged TSVs are keyed by these identifiers, and so is every assignment the solver
        # writes back. The caller builds the payload, so the counts have to travel with the file
        # paths -- there is no other channel between here and it.
        "renamed_sc": renamed_sc,
        "renamed_st": renamed_st,
    }


def _write_cytospace_inputs(loaded: dict, output_dir: str) -> dict[str, Any]:
    """Write what :func:`_load_cytospace_inputs` holds as CytoSPACE's four tab-delimited files.

    Returns the file paths and every count of ``loaded`` -- not the AnnData objects, so the caller
    can let them go before the CytoSPACE subprocess starts.
    """
    adata_sc = loaded["adata_sc"]
    adata_st = loaded["adata_st"]

    # scRNA-seq expression: genes x cells (tab-delimited), written block by block.
    sc_expr_path = os.path.join(output_dir, "sc_expression.txt")
    _write_features_by_obs_tsv(adata_sc.X, adata_sc.obs_names, adata_sc.var_names, sc_expr_path)
    log(f"Saved scRNA-seq expression: ({adata_sc.n_vars}, {adata_sc.n_obs})")

    labels_path = os.path.join(output_dir, "sc_labels.txt")
    _atomic_to_csv(loaded["labels_df"], labels_path, sep="\t")
    log(f"Saved cell type labels: {loaded['labels_df'].shape}")

    # Spatial expression: genes x spots (tab-delimited), written block by block.
    st_expr_path = os.path.join(output_dir, "st_expression.txt")
    _write_features_by_obs_tsv(adata_st.X, adata_st.obs_names, adata_st.var_names, st_expr_path)
    log(f"Saved spatial expression: ({adata_st.n_vars}, {adata_st.n_obs})")

    coords_path = os.path.join(output_dir, "st_coordinates.txt")
    _atomic_to_csv(loaded["coords_df"], coords_path, sep="\t")
    log(f"Saved spatial coordinates: {loaded['coords_df'].shape}")

    info = {k: v for k, v in loaded.items() if k not in ("adata_sc", "adata_st", "labels_df", "coords_df")}
    info.update(
        {
            "sc_expression": sc_expr_path,
            "sc_labels": labels_path,
            "st_expression": st_expr_path,
            "st_coordinates": coords_path,
        }
    )
    return info


def _h5ad_to_cytospace_inputs(
    sc_h5ad: str,
    spatial_h5ad: str,
    output_dir: str,
    cell_type_key: str,
    n_cells: int,
    n_top_genes: int = 5000,
    seed: int = 0,
    drop_unlabeled_cells: bool = False,
) -> dict[str, Any]:
    """Convert h5ad files to tab-delimited text files for CytoSPACE: load and cut, then write.

    Returns the staged file paths and the counts for each object -- the caller builds the payload,
    so anything it has to report has to travel back with the paths. See
    :func:`_load_cytospace_inputs` for the cuts; the pipeline itself calls the two halves with the
    memory check in between.
    """
    log("Converting h5ad inputs to CytoSPACE tab-delimited format...")
    loaded = _load_cytospace_inputs(
        sc_h5ad,
        spatial_h5ad,
        cell_type_key,
        n_cells,
        n_top_genes=n_top_genes,
        seed=seed,
        drop_unlabeled_cells=drop_unlabeled_cells,
    )
    return _write_cytospace_inputs(loaded, output_dir)


# ----------------------------------------------------------------------------- cell-type fractions


def _reference_label_fractions(labels):
    """The reference's own label frequencies, as a ``pd.Series`` indexed by cell type (str).

    ``labels`` is the staged ``sc_labels.txt`` path, or the labels frame that will be written as it:
    the frame goes through the same tab-separated text round trip, so a label reads back exactly as
    CytoSPACE will read it from the file (a numeric-looking label included), before anything is
    written to disk.
    """
    import io

    import pandas as pd

    if not isinstance(labels, (str, bytes, os.PathLike)):
        buffer = io.StringIO()
        labels.to_csv(buffer, sep="\t")
        buffer.seek(0)
        labels = buffer
    labels_df = pd.read_csv(labels, sep="\t", index_col=0)
    counts = labels_df["CellType"].astype(str).value_counts()
    return counts / counts.sum()


def _fractions_id_column(frame, reference):
    """The column of a fractions table that holds row identifiers, or None when it has none.

    Found by what it is, not by where it sits. ``index_col=0`` is right for the pandas convention and
    wrong for R's: SPOTlight and CARD ``write.csv`` their proportions with ``row.names = FALSE`` and a
    ``spot`` column LAST, so the first cell type became the index and ``spot`` a column of text, and
    every SPOTlight table was refused as "non-numeric". The identifier column is, in order: a column
    named like a spot identifier (anywhere), else the first column when its header cell is empty
    (pandas' unnamed index, read back as "Unnamed: 0") or its values are not numbers and its name is
    not a reference cell type. A header one field short of its rows (R's ``write.table``) has already
    had its identifiers made the index by pandas, and matches none of these.
    """
    import pandas as pd

    lowered = {str(c).strip().lower(): c for c in frame.columns}
    for name in _SPOT_ID_COLUMNS:
        if name in lowered and str(lowered[name]) not in reference:
            return lowered[name]
    first = frame.columns[0]
    if str(first).startswith("Unnamed: "):
        return first
    if str(first) in reference:
        return None
    values = frame[first]
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric[values.notna()].isna().any():
        return first
    return None


def _supplied_fractions(path: str, reference_types) -> tuple:
    """Read ``cell_type_fractions_path`` into fractions over the reference's cell types.

    Three layouts, decided by where the reference's cell-type names appear:

    * one row, cell types as columns -- upstream's own ``Seurat_weights.txt``;
    * spots x cell types (a deconvolution's per-spot proportions) -- summed per type and
      normalised, which is exactly how upstream turns its per-spot Seurat scores into fractions;
    * one column, cell types as the row index.

    The identifier column is found wherever it sits (:func:`_fractions_id_column`). A row with no
    value at all is a spot the tool that wrote the table left without a composition (TACCO leaves
    zero-count spots empty, SPOTlight writes NA where every type fell below min_prop): it adds
    nothing to any sum, so it is left out and counted rather than refused. A row with some values
    missing, a value that is not a number, a negative value, or a type the reference does not have
    is refused -- CytoSPACE stops on a type it has no cells for.

    Deconvolution tools rewrite '/' (and most of them ' ') in a label to '_' before they publish it
    (``sanitize_cell_type_names``), so SPOTlight's table names the reference's 'Treg/Tfr' as
    'Treg_Tfr'. A name that is not a reference type but is the rewrite of exactly one reference type
    (one the table does not also name as it is) is matched to it, and the match is returned for the
    payload; any other unknown name is refused.

    Returns ``(fractions, layout, details)`` with ``details`` holding ``n_rows_without_values`` and
    ``names_matched`` (``{name in the table: reference type}``).
    """
    import numpy as np
    import pandas as pd

    if not os.path.isfile(path):
        raise FileNotFoundError(f"cell_type_fractions_path={path!r} does not exist.")
    frame = pd.read_csv(path, sep=sniff_tabular_sep(path))
    if frame.shape[1] == 0:
        raise ValueError(
            f"cell_type_fractions_path={path!r} parsed into 0 columns -- the whole first line became the "
            f"index name ({frame.index.name!r}). Check the file's field separator."
        )
    reference = {str(t) for t in reference_types}
    id_column = _fractions_id_column(frame, reference)
    if id_column is not None:
        frame = frame.set_index(id_column)
    if frame.shape[1] == 0:
        raise ValueError(f"cell_type_fractions_path={path!r} holds identifiers and no fractions.")
    columns = [str(c) for c in frame.columns]
    rows = [str(i) for i in frame.index]
    if set(columns) & reference:
        raw = frame.copy()
        raw.columns = columns
        transposed = False
    elif frame.shape[1] == 1 and set(rows) & reference:
        raw = frame.iloc[:, [0]].T
        raw.columns = rows
        transposed = True
    else:
        raise ValueError(
            f"cell_type_fractions_path={path!r}: none of its column or row names is a cell type of the "
            f"reference (e.g. {sorted(reference)[:8]}). Expected cell types as columns (one row of "
            "fractions, or one row per spot) or as the row index of a single fractions column."
        )
    values = raw.apply(pd.to_numeric, errors="coerce")
    # Text where a number belongs is refused; an empty field is a missing value, handled below.
    not_numeric = [c for c in values.columns if bool((values[c].isna() & raw[c].notna()).any())]
    if not_numeric:
        raise ValueError(f"cell_type_fractions_path={path!r}: non-numeric values under {not_numeric[:8]}.")
    empty_rows = values.isna().all(axis=1)
    n_rows_without_values = 0 if transposed else int(empty_rows.sum())
    if n_rows_without_values:
        if n_rows_without_values == len(values):
            raise ValueError(f"cell_type_fractions_path={path!r}: every one of its {len(values)} rows is empty.")
        values = values[~empty_rows.to_numpy()]
    bad = [c for c in values.columns if not np.all(np.isfinite(values[c].to_numpy(dtype=float)))]
    if bad:
        n_partial = int(values[bad].isna().any(axis=1).sum()) if not transposed else 1
        raise ValueError(
            f"cell_type_fractions_path={path!r}: missing or non-finite values under {bad[:8]} in {n_partial} "
            "row(s) that carry other values. A row with no value at all is left out; a row with some missing "
            "has no composition to sum."
        )
    negative = [c for c in values.columns if (values[c].to_numpy(dtype=float) < 0).any()]
    if negative:
        raise ValueError(f"cell_type_fractions_path={path!r}: negative values under {negative[:8]}.")
    names_matched = {}
    supplied = set(values.columns)
    for name in sorted(supplied - reference):
        candidates = [
            ref
            for ref in sorted(reference)
            if ref not in supplied
            and name
            in (
                sanitize_cell_type_names([ref])[0][0],
                sanitize_cell_type_names([ref], replace_space=False)[0][0],
            )
        ]
        if len(candidates) == 1:
            names_matched[name] = candidates[0]
    # Two names of the table that both rewrite to one reference type would be summed into it: no match.
    targets = list(names_matched.values())
    names_matched = {k: v for k, v in names_matched.items() if targets.count(v) == 1}
    if names_matched:
        values = values.rename(columns=names_matched)
    unknown = sorted(set(values.columns) - reference)
    if unknown:
        raise ValueError(
            f"cell_type_fractions_path={path!r} names cell types the staged reference has no cells of: "
            f"{unknown[:8]}. CytoSPACE can only place types it can draw cells from; the reference has "
            f"{sorted(reference)[:8]}{' ...' if len(reference) > 8 else ''}."
        )
    if transposed:
        layout = "one column indexed by cell type"
    elif len(values) == 1 and not n_rows_without_values:
        layout = "one row of fractions"
    else:
        layout = f"{len(values)} rows summed per cell type"
        if n_rows_without_values:
            layout += f"; {n_rows_without_values} rows with no values left out"
    if id_column is not None and not str(id_column).startswith("Unnamed: "):
        layout += f"; row identifiers read from column {str(id_column)!r}"
    per_type = values.sum(axis=0)
    per_type = per_type.groupby(level=0).sum()
    total = float(per_type.sum())
    if not total > 0:
        raise ValueError(f"cell_type_fractions_path={path!r}: the fractions sum to {total}, not a composition.")
    details = {"n_rows_without_values": n_rows_without_values, "names_matched": names_matched}
    return per_type / total, layout, details


def _write_fractions(fractions, path: str) -> None:
    """CytoSPACE's fraction-file layout: cell types as the header, one row of fractions."""
    import pandas as pd

    frame = pd.DataFrame([fractions.values], columns=fractions.index)
    frame.index = ["fractions"]
    _atomic_to_csv(frame, path, sep="\t")


# ----------------------------------------------------------------------------- memory


def cytospace_peak_bytes(
    n_ref_cells: int,
    n_ref_genes: int,
    n_spots: int,
    n_st_genes: int,
    n_matched_genes: int,
    n_placed: int,
    single_cell: bool,
):
    """``(peak_bytes, phase, {phase: bytes})`` for the dense float64 frames CytoSPACE holds.

    CytoSPACE reads both tables into dense pandas frames and keeps the whole staged reference for
    its output. In spot mode it then solves one ``n_placed x n_placed`` assignment (the cost matrix,
    its jittered copy and the jitter itself), where ``n_placed`` is about ``mean_cell_numbers`` x
    spots. In single-cell mode it solves partitions of at most 10,000 locations, four at a time.
    A coarse, upper-leaning estimate: it exists to refuse a run that cannot fit, not to size one.
    """
    f8 = 8
    reference = n_ref_genes * n_ref_cells * f8
    spatial = n_st_genes * n_spots * f8
    read = 2 * reference + 2 * spatial
    downsample = reference + 3 * n_matched_genes * n_ref_cells * f8 + spatial
    if single_cell:
        part = min(n_spots, _UPSTREAM_SPOTS_PER_PARTITION)
        workers = min(int(math.ceil(n_spots / float(_UPSTREAM_SPOTS_PER_PARTITION))), _UPSTREAM_PROCESSORS)
        solve = max(1, workers) * (3 * part * part * f8 + 2 * n_matched_genes * part * f8)
    else:
        solve = 3 * n_placed * n_placed * f8 + n_spots * n_placed * f8
    assign = reference + 2 * n_matched_genes * (n_placed + n_spots) * f8 + solve
    phases = {
        "reading the staged tables": read,
        "downsampling the reference": downsample,
        "building and solving the assignment": assign,
    }
    phase = max(phases, key=phases.get)
    return phases[phase], phase, phases


def check_cytospace_memory(file_info: dict, single_cell: bool, mean_cell_numbers: int, available=None):
    """Refuse, naming the numbers, before CytoSPACE allocates frames that cannot fit. Never subsamples."""
    n_spots = int(file_info["n_st_spots_used"])
    n_placed = n_spots if single_cell else int(mean_cell_numbers) * n_spots
    need, phase, _ = cytospace_peak_bytes(
        n_ref_cells=int(file_info["n_sc_cells_used"]),
        n_ref_genes=int(file_info.get("n_sc_genes_staged", file_info["n_sc_genes"])),
        n_spots=n_spots,
        n_st_genes=int(file_info["n_st_genes_used"]),
        n_matched_genes=int(file_info.get("n_genes_matched", file_info["n_st_genes_used"])),
        n_placed=n_placed,
        single_cell=single_cell,
    )
    if available is None:
        available = available_memory_bytes()
    if available is not None and need > available:
        mode_hint = ""
        hint = "" if single_cell else (file_info.get("resolution_hint") or "")
        if hint:
            # The knob that changes the problem, when the data says spot mode is the wrong one.
            one_cell_need, _, _ = cytospace_peak_bytes(
                n_ref_cells=int(file_info["n_sc_cells_used"]),
                n_ref_genes=int(file_info.get("n_sc_genes_staged", file_info["n_sc_genes"])),
                n_spots=n_spots,
                n_st_genes=int(file_info["n_st_genes_used"]),
                n_matched_genes=int(file_info.get("n_genes_matched", file_info["n_st_genes_used"])),
                n_placed=n_spots,
                single_cell=True,
            )
            mode_hint = (
                f" The slide looks like one cell per location ({hint}): single_cell=True places one cell per "
                f"location and solves it in partitions of {_UPSTREAM_SPOTS_PER_PARTITION} (about "
                f"{one_cell_need / 1e9:.1f} GB at its peak)."
            )
        if single_cell:
            placed = f"{n_placed} cells (one per location, solved in partitions of {_UPSTREAM_SPOTS_PER_PARTITION})"
        else:
            placed = (
                f"about {n_placed} cells (mean_cell_numbers={mean_cell_numbers} x {n_spots} locations, "
                f"one {n_placed} x {n_placed} assignment)"
            )
        raise MemoryError(
            f"CytoSPACE holds its inputs as dense float64 frames: the staged reference is "
            f"{file_info.get('n_sc_genes_staged', file_info['n_sc_genes'])} genes x {file_info['n_sc_cells_used']} "
            f"cells, the spatial table {file_info['n_st_genes_used']} genes x {n_spots} locations, and it places "
            f"{placed}. That needs about {need / 1e9:.1f} GB at its peak ({phase}), but about "
            f"{available / 1e9:.1f} GB is available here (MemAvailable, or the room left under the cgroup memory "
            "limit). Nothing has been staged yet. These frames are how "
            "CytoSPACE works and this worker never subsamples cells or locations: n_top_genes narrows only the "
            "matched panel, and mean_cell_numbers sets how many cells spot mode places per location. Run where "
            "that much memory is available." + mode_hint
        )
    return need, available


# ----------------------------------------------------------------------------- the run


def _cytospace_command(
    file_info: dict,
    fractions_path: str,
    cytospace_output_dir: str,
    seed: int,
    single_cell: bool,
    mean_cell_numbers: int,
) -> list:
    """The CytoSPACE command line, from the binary in this interpreter's own env."""
    # Determine the cytospace CLI binary path (same env as worker python)
    env_bin_dir = os.path.dirname(sys.executable)
    cytospace_bin = os.path.join(env_bin_dir, "cytospace")
    if os.path.exists(cytospace_bin):
        head = [cytospace_bin]
    else:
        # Same program, reached through its entry-point function when the console script is absent.
        head = [sys.executable, "-c", "from cytospace.cytospace import run_cytospace; run_cytospace()"]
    cmd = head + [
        "--scRNA-path",
        file_info["sc_expression"],
        "--cell-type-path",
        file_info["sc_labels"],
        "--st-path",
        file_info["st_expression"],
        "--coordinates-path",
        file_info["st_coordinates"],
        "--cell-type-fraction-estimation-path",
        fractions_path,
        "-o",
        cytospace_output_dir,
        "--seed",
        str(seed),
    ]
    if single_cell:
        cmd.append("--single-cell")
    else:
        cmd += ["--mean-cell-numbers", str(int(mean_cell_numbers))]
    return cmd


def run_cytospace_pipeline(
    sc_h5ad: str,
    spatial_h5ad: str,
    output_dir: str,
    cell_type_key: str,
    n_cells: int,
    n_top_genes: int,
    seed: int,
    single_cell: bool = False,
    mean_cell_numbers: int = UPSTREAM_MEAN_CELL_NUMBERS,
    cell_type_fractions_path: str = "",
    drop_unlabeled_cells: bool = False,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run the CytoSPACE cell-to-spot assignment pipeline."""
    import subprocess

    import pandas as pd

    if not single_cell and int(mean_cell_numbers) < 1:
        raise ValueError(
            f"mean_cell_numbers={mean_cell_numbers}: spot mode needs at least 1 cell per location "
            f"(CytoSPACE's default is {UPSTREAM_MEAN_CELL_NUMBERS}, for Visium)."
        )
    if int(timeout_s) < 0:
        raise ValueError(f"timeout_s={timeout_s}: pass a number of seconds, or 0 for no limit.")

    inputs = {"sc_h5ad": sc_h5ad, "spatial_h5ad": spatial_h5ad}
    if cell_type_fractions_path:
        inputs["cell_type_fractions_path"] = cell_type_fractions_path
    preflight_check(inputs=inputs, output_dir=output_dir)

    # Load both inputs and make every cut, in memory. Nothing is written until the fractions table
    # has been read and the memory check has passed: both can refuse the run, and a refusal that
    # comes after tens of GB of text have been staged may never reach the caller.
    loaded = _load_cytospace_inputs(
        sc_h5ad,
        spatial_h5ad,
        cell_type_key,
        n_cells,
        n_top_genes,
        seed,
        drop_unlabeled_cells=drop_unlabeled_cells,
    )

    # The tissue-wide composition CytoSPACE will place. Upstream estimates it from the spatial data
    # with get_cellfracs_seuratv3.R, which needs R + Seurat; this env has neither, so that step has
    # never run here. Without a supplied table the reference's own label frequencies stand in, and
    # the payload says so: they make the assigned composition an input, not a finding.
    reference_fractions = _reference_label_fractions(loaded["labels_df"])
    n_fraction_rows_without_values = 0
    fraction_names_matched = {}
    if cell_type_fractions_path:
        fractions, layout, fraction_details = _supplied_fractions(cell_type_fractions_path, reference_fractions.index)
        n_fraction_rows_without_values = fraction_details["n_rows_without_values"]
        fraction_names_matched = fraction_details["names_matched"]
        fraction_source = FRACTIONS_SUPPLIED
        log(f"Cell type fractions from {cell_type_fractions_path} ({layout}): {len(fractions)} types")
    else:
        fractions, layout = reference_fractions, ""
        fraction_source = FRACTIONS_FROM_REFERENCE
        log("Cell type fractions = the scRNA-seq reference's label frequencies (no spatial estimate)")

    # Refuse, with the numbers, before a run that cannot fit is staged or launched. Every count the
    # estimate reads is known from the loaded objects.
    need_bytes, _available = check_cytospace_memory(loaded, single_cell, mean_cell_numbers)

    # Stage the tab-delimited text CytoSPACE reads, then let the AnnData objects go: the
    # CytoSPACE subprocess needs the memory, not this process.
    file_info = _write_cytospace_inputs(loaded, output_dir)
    del loaded
    fractions_path = os.path.join(output_dir, "cell_type_fractions.txt")
    _write_fractions(fractions, fractions_path)
    log(f"Saved cell type fractions: {len(fractions)} types -> {fractions_path}")

    # Run CytoSPACE via its console_scripts entry point (cytospace binary)
    cytospace_output_dir = os.path.join(output_dir, "cytospace_results")
    os.makedirs(cytospace_output_dir, exist_ok=True)
    cmd = _cytospace_command(file_info, fractions_path, cytospace_output_dir, seed, single_cell, mean_cell_numbers)
    mode = "single-cell mode" if single_cell else "spot mode"
    log(f"Running CytoSPACE ({mode}): {' '.join(cmd)}")
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=(int(timeout_s) or None))
    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"CytoSPACE was still running when this worker's timeout_s={timeout_s} s limit expired "
            f"({mode}; {file_info['n_sc_cells_used']} reference cells, {file_info['n_st_spots_used']} locations, "
            f"{file_info['n_genes_matched']} matched genes). It was stopped, so anything it wrote under "
            f"{cytospace_output_dir} is incomplete. Raise timeout_s (0 = no limit); the agent's per-call budget "
            "(STCoscientist(timeout_seconds=...), --timeout, SOG_TIMEOUT_SECONDS) bounds the call as well. "
            "Do not subsample the input to fit the limit."
        ) from None
    if proc.stderr:
        log(f"CytoSPACE stderr: {proc.stderr[-2000:]}")
    if proc.returncode != 0:
        raise RuntimeError(f"CytoSPACE exited with code {proc.returncode}. stderr: {proc.stderr[-2000:]}")

    # Collect this run's output files. A file older than the launch was left by an earlier run into
    # the same folder (a spot-mode run writes tables a single-cell run does not): not this run's.
    output_files = {}
    stale = []
    for fname in sorted(os.listdir(cytospace_output_dir)):
        fpath = os.path.join(cytospace_output_dir, fname)
        if os.path.isfile(fpath):
            if os.path.getmtime(fpath) + 1.0 < started:
                stale.append(fname)
                continue
            output_files[fname] = fpath
    if stale:
        log(f"Not reporting {stale}: left in {cytospace_output_dir} by an earlier run")

    assignment_path = None
    for key in ["assigned_locations.csv", "cell_to_spot_assignments.csv"]:
        if key in output_files:
            assignment_path = output_files[key]
            break
    if assignment_path is None:
        # Any other CSV here is a per-spot table, not one row per assigned cell; counting its rows
        # as "assigned cells" would publish a wrong number at status ok.
        raise RuntimeError(
            f"CytoSPACE exited 0 but wrote no assigned_locations.csv in {cytospace_output_dir} "
            f"(this run wrote: {sorted(output_files)})."
        )
    assignments = pd.read_csv(assignment_path)
    assigned_cells = len(assignments)
    if "SpotID" in assignments.columns:
        assigned_spots = assignments["SpotID"].nunique()
    elif len(assignments.columns) >= 2:
        assigned_spots = assignments.iloc[:, 1].nunique()
    else:
        assigned_spots = 0

    fractions_rounded = {str(k): round(float(v), 6) for k, v in fractions.items()}
    out = WorkerOutput("cytospace", task="cell_assignment")
    out.set_data(
        n_sc_cells=file_info["n_sc_cells"],
        n_sc_cells_used=file_info["n_sc_cells_used"],
        n_sc_cells_unlabeled=file_info["n_sc_cells_unlabeled"],
        n_sc_genes=file_info["n_sc_genes"],
        n_sc_genes_used=file_info["n_sc_genes_used"],
        n_st_spots=file_info["n_st_spots"],
        n_st_spots_used=file_info["n_st_spots_used"],
        n_st_genes=file_info["n_st_genes"],
        n_st_genes_used=file_info["n_st_genes_used"],
        n_genes_matched=file_info["n_genes_matched"],
        cell_type_fractions=fractions_rounded,
        estimated_peak_memory_gb=round(need_bytes / 1e9, 2),
    )
    out.add_output_files(output_files)
    out.add_output_file("assignment_csv", assignment_path)
    out.add_output_file("cell_type_fractions", fractions_path)
    out.add_params(
        {
            "cell_type_key": cell_type_key,
            "n_cells": n_cells,
            "n_top_genes": n_top_genes,
            "seed": seed,
            "single_cell": bool(single_cell),
            "mean_cell_numbers": int(mean_cell_numbers),
            "cell_type_fractions_path": cell_type_fractions_path,
            "drop_unlabeled": bool(drop_unlabeled_cells),
            "timeout_s": int(timeout_s),
            "assignment_mode": "single_cell" if single_cell else "spot",
            "cell_type_fraction_source": fraction_source,
            "spatial_gene_symbol_column": file_info.get("spatial_symbol_column") or None,
        }
    )
    if cell_type_fractions_path:
        out.add_params(
            {
                "cell_type_fraction_rows_without_values": int(n_fraction_rows_without_values),
                "cell_type_fraction_names_matched": dict(fraction_names_matched),
            }
        )
    record_in_tissue(out, file_info["n_st_spots"], file_info["n_st_spots_off_tissue"])
    if single_cell:
        method = "CytoSPACE (lapjv; single-cell mode: one reference cell per location)"
    else:
        method = f"CytoSPACE (lapjv; spot mode: about mean_cell_numbers={int(mean_cell_numbers)} cells per location)"
    if fraction_source == FRACTIONS_FROM_REFERENCE:
        method += "; cell-type fractions = the scRNA-seq reference's label frequencies"
    else:
        method += "; cell-type fractions from cell_type_fractions_path"
    # The reference-frequency table is this wrapper's only built-in source (upstream's Seurat
    # estimate needs R), so it is the method, not a fallback -- named as such rather than flagged.
    record_method(out, method, used_fallback=False)
    if single_cell and int(mean_cell_numbers) != UPSTREAM_MEAN_CELL_NUMBERS:
        record_ignored(
            out,
            ["mean_cell_numbers"],
            "single_cell=True places exactly one cell per location, so a mean number of cells per spot does not apply",
        )
    out.add_params(identifier_rename_params(file_info.get("renamed_st")))
    out.add_params(identifier_rename_params(file_info.get("renamed_sc"), suffix="sc"))
    out.set_summary(
        assigned_cells=int(assigned_cells),
        assigned_spots=int(assigned_spots),
    )

    # Every cut below happens before CytoSPACE is invoked, and the worker's own log() lines
    # announcing them go to stderr -- which base_mcp drops on a successful run. Say it here or
    # it is not said at all.
    cell_reasons = []
    if file_info["n_sc_cells_unlabeled"]:
        cell_reasons.append(f"dropping {file_info['n_sc_cells_unlabeled']} cells with no label (drop_unlabeled=True)")
    if n_cells > 0 and file_info["n_sc_cells_used"] < file_info["n_sc_cells"] - file_info["n_sc_cells_unlabeled"]:
        cell_reasons.append(f"a random draw of n_cells={n_cells} cells (seed={seed})")
    cell_note = describe_reduction(
        "reference cells",
        file_info["n_sc_cells"],
        file_info["n_sc_cells_used"],
        reason=" and ".join(cell_reasons),
    )
    sc_gene_note = describe_reduction(
        "reference genes",
        file_info["n_sc_genes"],
        file_info["n_sc_genes_used"],
        reason=f"the top-{n_top_genes} shared-panel cut, ranked by raw spatial variance",
    )
    st_gene_note = describe_reduction(
        "spatial genes",
        file_info["n_st_genes"],
        file_info["n_st_genes_used"],
        reason=f"the top-{n_top_genes} shared-panel cut, after any unmapped ENSEMBL IDs were dropped",
    )
    for note in (cell_note, sc_gene_note, st_gene_note):
        if note:
            out.add_warnings(note.strip())

    if fraction_source == FRACTIONS_FROM_REFERENCE:
        fraction_note = (
            " The number of cells of each type was fixed in advance to the scRNA-seq reference's own label "
            f"frequencies ({len(fractions)} types): CytoSPACE's spatial estimate of the composition needs R/Seurat, "
            "which this environment does not have, so the assigned composition mirrors the reference and is not a "
            "finding about the tissue. Pass cell_type_fractions_path (fractions estimated from this slide, e.g. by "
            "a deconvolution) to set it from the spatial data."
        )
        out.add_warning(fraction_note.strip())
    else:
        fraction_note = (
            f" The number of cells of each type was fixed in advance to the fractions in {cell_type_fractions_path} "
            f"({layout}, {len(fractions)} types)."
        )
        if fraction_names_matched:
            shown = ", ".join(f"{k!r} -> {v!r}" for k, v in list(fraction_names_matched.items())[:5])
            out.add_warning(
                f"{len(fraction_names_matched)} cell-type name(s) in {cell_type_fractions_path} are not reference labels "
                f"but the '/'->'_' rewrite deconvolution tools apply to one, and were matched to it: {shown}."
            )
        if n_fraction_rows_without_values:
            out.add_warning(
                f"{n_fraction_rows_without_values} row(s) of {cell_type_fractions_path} carry no value at all "
                "(e.g. spots the deconvolution left without a composition) and were left out of the per-type sums."
            )
    if single_cell:
        mode_note = " Single-cell mode: exactly one reference cell was placed at each location."
    else:
        mode_note = (
            f" Spot mode: each location's cell count was estimated from its RNA content, scaled so that the "
            f"average is about mean_cell_numbers={int(mean_cell_numbers)}."
        )
        hint = file_info.get("resolution_hint") or ""
        if hint:
            hint_note = (
                f" The spatial data looks like one cell per location ({hint}), yet spot mode placed about "
                f"{int(mean_cell_numbers)} reference cells in each; pass single_cell=True for such data."
            )
            mode_note += hint_note
            out.add_warning(hint_note.strip())
    off_tissue_note = ""
    if file_info["n_st_spots_off_tissue"]:
        off_tissue_note = (
            f" {file_info['n_st_spots_off_tissue']} of the {file_info['n_st_spots']} spatial locations supplied were "
            "background (obs['in_tissue'] == 0) and were left out."
        )
    out.set_analysis(
        f"CytoSPACE assigned {assigned_cells} cells to {assigned_spots} spatial spots "
        f"using {file_info['n_sc_cells_used']} reference cells and "
        f"{file_info['n_st_spots_used']} spatial locations, matched on {file_info['n_genes_matched']} shared genes."
        + off_tissue_note
        + mode_note
        + fraction_note
        + (
            f" Spatial ENSEMBL IDs were matched by the symbols in var['{file_info['spatial_symbol_column']}']."
            if file_info.get("spatial_symbol_column")
            else ""
        )
        + cell_note
        + sc_gene_note
        + st_gene_note
        + identifier_rename_note(file_info.get("renamed_st"), subject="spatial data")
        + identifier_rename_note(file_info.get("renamed_sc"), subject="scRNA-seq reference")
    )
    return out.to_dict()


def main() -> None:
    args = parse_args()

    orig_stdout = sys.stdout
    sys.stdout = sys.stderr
    error_exc = None
    try:
        try:
            result = run_cytospace_pipeline(
                sc_h5ad=args.sc_h5ad,
                spatial_h5ad=args.spatial_h5ad,
                output_dir=args.output_dir,
                cell_type_key=args.cell_type_key,
                n_cells=args.n_cells,
                n_top_genes=args.n_top_genes,
                seed=args.seed,
                single_cell=args.single_cell,
                mean_cell_numbers=args.mean_cell_numbers,
                cell_type_fractions_path=args.cell_type_fractions_path,
                drop_unlabeled_cells=args.drop_unlabeled,
                timeout_s=args.timeout_s,
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
        WorkerOutput.emit_error("cytospace", error_msg, task="cell_assignment", exc=error_exc)
        sys.exit(1)
    else:
        print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
