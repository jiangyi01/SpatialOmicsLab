#!/usr/bin/env python3
"""
Cell-type neighbourhood enrichment on a spatial k-NN graph, served under the DeepLinc tool name.

Runs inside the /opt/conda/envs/deeplinc conda env.

WHAT RUNS. Every cell (spot) is joined to its ``n_neighbors`` nearest neighbours by position; the
graph is made undirected (an edge exists when either end lists the other). For every pair of cell
types (A, B) the worker counts the undirected edges joining an A cell to a B cell and divides that
by the count expected if the labels were placed at random on the same graph:

    E[edges A-B] = 2 * E * n_A * n_B / (n * (n - 1))        for A != B
    E[edges A-A] =     E * n_A * (n_A - 1) / (n * (n - 1))  for A == A

so 1.0 means "as often as random placement", above 1.0 enriched, below 1.0 avoided. Significance
is a one-sided label-permutation test on the same graph (100 permutations), with the +1
correction, p = (1 + #{permuted count >= observed count}) / (1 + 100), so no p-value is 0.

WHAT DOES NOT RUN. DeepLinc proper -- the variational graph auto-encoder installed at
/opt/conda/envs/deeplinc/DeepLinc/DeepLinc.py, trained on an adjacency plus expression -- is never
called, and nothing here imports TensorFlow. The tool keeps the DeepLinc name because callers know
it by that name; the payload says what ran (``params.method``, ``params.deeplinc_model_run``).
Gene expression is not used: in h5ad mode the matrix is never read, and in CSV mode the counts file
only supplies the cell IDs and the gene count. ``n_hvg`` is accepted for compatibility and reported
in ``params.ignored``.

Input: h5ad file with spatial coordinates and cell type annotations,
       or separate CSV files (counts.csv, coord.csv, cell_type.csv). coord.csv may be Space Ranger's
       headed tissue_positions.csv or its headerless pre-2.0 tissue_positions_list.csv (recognised from
       the first line and read positionally, as 10x documents it).
Background: spots with in_tissue == 0 (obs['in_tissue'] of the h5ad, or an in_tissue column of
       coord.csv) are glass outside the tissue, not a cell type; they are left out before the graph is
       built and counted in params.in_tissue_filter and data.n_spots_off_tissue_dropped.
Output: deeplinc_interaction_scores.csv (observed/expected, types x types), deeplinc_pvalues.csv,
        deeplinc_significant_interactions.csv (always with its header), the graph as an edge list
        (deeplinc_adjacency_edges.csv: cell_a, cell_b, one row per undirected edge) and, up to
        DENSE_ADJACENCY_MAX_CELLS cells, the same graph as a dense n x n deeplinc_adjacency.csv.

Sections: the graph is two-dimensional by design, so ``--dims 3`` is refused, and coordinates are read
        through ``worker_utils.spatial_frame``: a file holding two or more sections (an obs section column,
        or a ``z``/section-like column of coord.csv) is refused unless ``--section-key`` names the column.
        With it the run is per section (``worker_utils.per_section``): each section's files in
        ``section_<label>/`` and one long deeplinc_significant_interactions.csv at the top level with a
        leading ``section`` column; ``params.mode`` is ``per-section-2d`` (``2d`` for one plane).

All logs go to stderr; stdout is JSON-only (final result).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from worker_utils import (
    MAX_SECTION_LEVELS,
    STACK_COLUMN_NAMES,
    TISSUE_POSITIONS_COLUMNS,
    Frame,
    WorkerOutput,
    drop_unlabeled,
    id_mismatch_msg,
    keep_in_tissue,
    per_section,
    read_tissue_positions,
    record_ignored,
    record_in_tissue,
    record_method,
    resolve_coord_columns,
    sniff_tabular_sep,
    spatial_frame,
)

#: What actually runs, in the words ``params.method`` publishes. Not DeepLinc's VGAE.
METHOD_NAME = "knn_edge_type_enrichment_label_permutation"

SCORE_DEFINITION = (
    "observed / expected undirected k-NN edges between two cell types; expected under random "
    "relabelling of the same graph = 2*E*n_a*n_b/(n*(n-1)) for a != b and E*n_a*(n_a-1)/(n*(n-1)) "
    "for a == b, so 1.0 is random placement"
)
PVALUE_DEFINITION = (
    "one-sided label-permutation p = (1 + #{permuted edge count >= observed edge count}) / (1 + n_permutations)"
)
N_PERMUTATIONS = 100
SIGNIFICANCE_ALPHA = 0.05
SIGNIFICANT_COLUMNS = ["type_a", "type_b", "score", "pvalue"]

#: The refusal for ``dims=3``: the k-NN graph here is two-dimensional by design.
TWO_D_ONLY = "DeepLinc builds its neighbourhood in two dimensions; run per section with `dims=2, section_key=<column>`."

#: The dense n x n deeplinc_adjacency.csv is a convenience copy of the edge list, written only up to
#: this many cells: at 20,000 it is already 4e8 entries (~1.6 GB of text), and on a Visium HD slide
#: (~500k bins) it would be ~1 TB. The edge list carries the same graph at every size.
DENSE_ADJACENCY_MAX_CELLS = 20000

#: Rows of the dense adjacency materialised at once while it is streamed to disk (~100 MB float32).
_DENSE_BLOCK_ENTRIES = 25000000


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def _ensure_dir(p: str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_to_csv(df: pd.DataFrame, path, **kwargs) -> None:
    """``df.to_csv`` to ``<path>.partial``, then rename: an interrupted write never sits at ``path``."""
    tmp = f"{path}.partial"
    df.to_csv(tmp, **kwargs)
    os.replace(tmp, str(path))


def _build_spatial_adjacency(coords: np.ndarray, n_neighbors: int = 10):
    """Undirected binary k-NN graph as a sparse CSR matrix (n x n, no self-loops).

    Each cell is joined to its ``n_neighbors`` nearest other cells, then the graph is symmetrised
    (an edge exists when either end lists the other). A cell's own index is removed wherever the
    tree returns it, so duplicate coordinates cannot turn into a self-loop, and nothing n x n is
    ever materialised.
    """
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    n_cells = int(coords.shape[0])
    k = int(n_neighbors)
    if k < 1:
        raise ValueError(f"n_neighbors={k} must be at least 1.")
    if n_cells <= k:
        raise ValueError(
            f"n_neighbors={k} needs more than {k} cells to build a k-NN graph, and the input has {n_cells}. "
            "Lower n_neighbors."
        )
    if not np.isfinite(coords).all():
        bad = int((~np.isfinite(coords)).any(axis=1).sum())
        raise ValueError(
            f"{bad} of {n_cells} cells have a missing or non-finite spatial coordinate; a k-NN graph cannot place "
            "them. Remove those cells or fix their coordinates first."
        )

    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    idx = np.asarray(idx).reshape(n_cells, k + 1)
    rows = np.arange(n_cells)
    # A stable sort on "is this me?" moves the self-hit to the end of its row without reordering
    # the neighbours, so the first k entries are the k nearest OTHER cells.
    order = np.argsort(idx == rows[:, None], axis=1, kind="stable")
    nbrs = np.take_along_axis(idx, order, axis=1)[:, :k]
    r = np.repeat(rows, k)
    c = nbrs.ravel()
    adj = coo_matrix((np.ones(r.size, dtype=np.float32), (r, c)), shape=(n_cells, n_cells)).tocsr()
    adj = adj.maximum(adj.T).tocsr()
    adj.data[:] = 1.0
    adj.eliminate_zeros()
    return adj


def _edge_endpoints(adj):
    """Each undirected edge once, as (u, v) index arrays with u < v."""
    from scipy.sparse import triu

    upper = triu(adj, k=1).tocoo()
    return upper.row.astype(np.int64), upper.col.astype(np.int64)


def _edge_type_counts(u: np.ndarray, v: np.ndarray, codes: np.ndarray, n_types: int) -> np.ndarray:
    """Symmetric types x types count of undirected edges: [a, b] = edges joining an a to a b.

    The diagonal counts each within-type edge ONCE. The old loop added 2 to [a, a] per edge while
    its expectation counted the edge once, and its off-diagonal expectation left out the factor 2
    for the two ways an a-b edge can be oriented -- so random placement scored ~2.0 everywhere.
    """
    flat = codes[u] * n_types + codes[v]
    half = np.bincount(flat, minlength=n_types * n_types).reshape(n_types, n_types).astype(np.float64)
    counts = half + half.T
    np.fill_diagonal(counts, np.diag(half))
    return counts


def _expected_edge_counts(type_sizes, n_edges: int, n_cells: int) -> np.ndarray:
    """Expected undirected edge counts per type pair under random relabelling of the same graph."""
    sizes = np.asarray(type_sizes, dtype=np.float64)
    pairs = float(n_cells) * float(n_cells - 1)
    if pairs <= 0:
        return np.zeros((sizes.size, sizes.size), dtype=np.float64)
    expected = 2.0 * float(n_edges) * np.outer(sizes, sizes) / pairs
    np.fill_diagonal(expected, float(n_edges) * sizes * (sizes - 1.0) / pairs)
    return expected


def _scores_from_counts(counts: np.ndarray, expected: np.ndarray) -> np.ndarray:
    scores = np.zeros_like(counts, dtype=np.float64)
    ok = expected > 0
    scores[ok] = counts[ok] / expected[ok]
    return scores


def _compute_interaction_scores(adj, cell_types, unique_types: list[str]) -> pd.DataFrame:
    """Observed / expected undirected edge counts for every pair of cell types (1.0 = random).

    ``adj`` may be sparse or dense; only its upper triangle is read. See SCORE_DEFINITION.
    """
    import scipy.sparse as sps

    if not sps.issparse(adj):
        adj = sps.csr_matrix(np.asarray(adj))
    type_to_idx = {ct: i for i, ct in enumerate(unique_types)}
    codes = np.array([type_to_idx[ct] for ct in cell_types], dtype=np.int64)
    n_types = len(unique_types)
    u, v = _edge_endpoints(adj)
    counts = _edge_type_counts(u, v, codes, n_types)
    sizes = np.bincount(codes, minlength=n_types)
    expected = _expected_edge_counts(sizes, len(u), len(codes))
    return pd.DataFrame(_scores_from_counts(counts, expected), index=unique_types, columns=unique_types)


def _permutation_pvalues(
    u: np.ndarray, v: np.ndarray, codes: np.ndarray, n_types: int, observed_counts: np.ndarray, n_perms: int
) -> np.ndarray:
    """One-sided permutation p-values with the +1 correction (PVALUE_DEFINITION).

    Counts, not ratios, are compared: the expectation is the same for every relabelling, so the
    two orderings agree, and integers cannot disagree by a rounding error. The permutations are
    drawn from numpy's global generator, which ``run_deeplinc`` seeds.
    """
    at_least = np.zeros((n_types, n_types), dtype=np.int64)
    for _ in range(n_perms):
        perm = np.random.permutation(codes)
        at_least += _edge_type_counts(u, v, perm, n_types) >= observed_counts
    return (1.0 + at_least) / (1.0 + n_perms)


def _significant_pairs(scores: np.ndarray, pvalues: np.ndarray, unique_types: list[str]) -> list[dict]:
    rows = []
    n_types = len(unique_types)
    for i in range(n_types):
        for j in range(i, n_types):
            if pvalues[i, j] < SIGNIFICANCE_ALPHA and scores[i, j] > 1.0:
                rows.append(
                    {
                        "type_a": unique_types[i],
                        "type_b": unique_types[j],
                        "score": float(scores[i, j]),
                        "pvalue": float(pvalues[i, j]),
                    }
                )
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def _write_significant(rows: list[dict], path) -> None:
    """Always with the four-column header, so "none significant" is not an empty, headerless file."""
    _atomic_to_csv(pd.DataFrame(rows, columns=SIGNIFICANT_COLUMNS), path, index=False)


def _write_edge_list(u: np.ndarray, v: np.ndarray, cell_names: list, path) -> None:
    names = np.asarray([str(n) for n in cell_names], dtype=object)
    _atomic_to_csv(pd.DataFrame({"cell_a": names[u], "cell_b": names[v]}), path, index=False)


def _write_dense_adjacency(adj, cell_names: list, path) -> None:
    """The n x n 0/1 matrix as CSV, streamed in row blocks so it is never dense in memory whole."""
    n = adj.shape[0]
    block = max(1, min(n, _DENSE_BLOCK_ENTRIES // max(n, 1)))
    names = [str(c) for c in cell_names]
    tmp = f"{path}.partial"
    for start in range(0, n, block):
        stop = min(n, start + block)
        dense = adj[start:stop].toarray().astype(np.float32)
        frame = pd.DataFrame(dense, index=names[start:stop], columns=names)
        frame.to_csv(tmp, mode="w" if start == 0 else "a", header=start == 0)
    os.replace(tmp, str(path))


def _labels_or_raise(labels, allow_drop: bool, source: str):
    """(keep_mask, n_dropped) for a label vector; NaN is never a cell type."""
    keep, n_dropped = drop_unlabeled(labels, allow_drop, what=f"cells in {source}")
    if n_dropped:
        eprint(f"[DeepLinc] Dropped {n_dropped} cell(s) with no label in {source} (drop_unlabeled=True)")
    return keep, n_dropped


def _load_h5ad(st_h5ad: str, spatial_key: str, annotation_key: str, section_key=None):
    """Coordinates, raw labels, cell names, gene count, in-tissue counts, sections and frame. X is not read.

    Returns ``(coords, labels, cell_names, n_genes, (n_supplied, n_off_tissue), sections, frame)``, where
    ``sections`` is each cell's ``obs[section_key]`` label (``None`` without a section key) and ``frame`` is
    ``worker_utils.spatial_frame``'s record of the coordinates read. The coordinates are read through
    ``spatial_frame`` in two dimensions, so a file holding several sections is refused unless
    ``section_key`` says the run is per section. Spots with
    ``obs['in_tissue'] == 0`` are left out here (``worker_utils.keep_in_tissue``): a CELLxGENE Visium
    export carries every array spot, 56-70% of them background glass labelled 'unknown', and kept they
    joined the k-NN graph and the expected counts as a cell type -- on Heart Fetal12W the cardiac
    muscle-endothelial score was 2.31 on all spots and 0.94 on the in-tissue ones.
    """
    import anndata as ad

    eprint(f"[DeepLinc] Loading spatial h5ad (obs/obsm only; expression is not used): {st_h5ad}")
    adata = ad.read_h5ad(st_h5ad, backed="r")
    try:
        eprint(f"[DeepLinc] Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
        if spatial_key not in adata.obsm:
            raise KeyError(f"Spatial key '{spatial_key}' not found in adata.obsm. Available: {list(adata.obsm.keys())}")
        if annotation_key not in adata.obs:
            raise ValueError(
                f"annotation_key='{annotation_key}' not found in adata.obs. Available keys: {list(adata.obs.columns)}"
            )
        # obs and the one obsm key in memory, without X: a backed object cannot be subset by copy.
        light = ad.AnnData(obs=adata.obs.copy())
        light.obsm[spatial_key] = np.asarray(
            adata.obsm[spatial_key].toarray()
            if hasattr(adata.obsm[spatial_key], "toarray")
            else adata.obsm[spatial_key]
        )
        if "spatial_3d" in adata.uns:
            # The frame declarations (units, slice order) travel with the light copy.
            light.uns["spatial_3d"] = adata.uns["spatial_3d"]
        n_genes = int(adata.n_vars)
    finally:
        try:
            adata.file.close()
        except Exception:
            pass
    light, n_supplied, n_off_tissue = keep_in_tissue(light, "spots")
    if n_off_tissue:
        eprint(f"[DeepLinc] Left out {n_off_tissue} of {n_supplied} spots with obs['in_tissue'] == 0 (background)")
    coords, frame = spatial_frame(light, spatial_key, 2, section_key, "DeepLinc", False)
    labels = np.asarray(light.obs[annotation_key].astype(object), dtype=object)
    cell_names = [str(n) for n in light.obs_names]
    sections = light.obs[section_key].astype(str).to_numpy() if section_key else None
    return coords, labels, cell_names, n_genes, (n_supplied, n_off_tissue), sections, frame


#: A first field that names the identifier column, so the line holding it is a header.
_ID_COLUMN_LABELS = (
    "",
    "barcode",
    "barcodes",
    "spot",
    "spot_id",
    "spotid",
    "cell",
    "cell_id",
    "cellid",
    "sample",
    "sample_id",
    "index",
)


def _is_number(text) -> bool:
    try:
        return np.isfinite(float(str(text).strip()))
    except (TypeError, ValueError):
        return False


def _coord_file_is_headerless(fields) -> bool:
    """Whether a coordinate file's first line is a spot rather than column names.

    The rule the R workers use (``spotsweeper_worker.R``'s ``read_coords_csv``): an identifier that is
    not an identifier-column label, followed by nothing but numbers. Names that are exactly 0, 1, 2, ...
    (a pandas RangeIndex written as a header) keep reading as a header when there are four or more
    fields; no Space Ranger row reads 0,1,2,3,4.
    """
    fields = [str(f).strip() for f in fields]
    if len(fields) < 3 or fields[0].lower() in _ID_COLUMN_LABELS:
        return False
    if not all(_is_number(f) for f in fields[1:]):
        return False
    range_index = len(fields) >= 4 and fields[1:] == [str(i) for i in range(len(fields) - 1)]
    return not range_index


def _read_coord_table(coord_csv: str, out: WorkerOutput) -> pd.DataFrame:
    """``coord_csv`` indexed by cell ID, whether or not its first line names the columns.

    Space Ranger before 2.0 writes ``spatial/tissue_positions_list.csv`` with no header, and every
    Visium folder in the library ships it beside the headed ``tissue_positions.csv``. Read with a header
    row assumed, its first spot became the column names ('0', '0.1', '0.2', '4034', '3524'), no named
    pair matched, and the resolver fell back to in_tissue and array_row as x and y -- a k-NN graph on a
    constant axis, at exit code 0. A headerless six-column file is read as 10x documents it
    (``worker_utils.read_tissue_positions``), a headerless three-column one as barcode, x, y, and any
    other headerless layout is refused: which two columns are the axes would be a guess.
    """
    first = pd.read_csv(
        coord_csv, sep=sniff_tabular_sep(coord_csv), header=None, nrows=1, dtype=str, keep_default_na=False
    )
    fields = [str(v).strip() for v in first.iloc[0]] if len(first) else []
    if not _coord_file_is_headerless(fields):
        out.add_param("coord_header", "present")
        return pd.read_csv(coord_csv, index_col=0, sep=sniff_tabular_sep(coord_csv))
    shown = ",".join(fields)
    shown = shown if len(shown) <= 120 else shown[:120] + "..."
    if len(fields) == len(TISSUE_POSITIONS_COLUMNS):
        frame = read_tissue_positions(coord_csv)
        flag = pd.to_numeric(frame["in_tissue"], errors="coerce")
        if not flag.isin([0, 1]).all():
            raise ValueError(
                f"coord_csv {coord_csv} has no header row (its first line, {shown}, is an identifier followed only "
                "by numbers) and six columns, but its second column holds values other than 0 and 1, so it is not "
                "Space Ranger's in_tissue flag and the file is not tissue_positions_list.csv. Give the file a header "
                "row naming its two coordinate columns (imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, "
                "array_row/array_col, row/col or x/y)."
            )
        header = (
            "absent: read as Space Ranger's tissue_positions_list.csv (" + ", ".join(TISSUE_POSITIONS_COLUMNS) + ")"
        )
        frame = frame.set_index("barcode")
    elif len(fields) == 3:
        frame = pd.read_csv(coord_csv, sep=sniff_tabular_sep(coord_csv), header=None, names=["barcode", "x", "y"])
        frame["barcode"] = frame["barcode"].astype(str)
        frame = frame.set_index("barcode")
        header = "absent: read as barcode, x, y"
    else:
        raise ValueError(
            f"coord_csv {coord_csv} has no header row that names its columns: its first line ({shown}) is an "
            f"identifier followed only by numbers, so it reads as a spot, and the file has {len(fields)} columns, "
            "which is neither Space Ranger's six-column tissue_positions_list.csv layout ("
            + ", ".join(TISSUE_POSITIONS_COLUMNS)
            + ") nor barcode, x, y. Give the file a header row naming its two coordinate columns (imagerow/imagecol, "
            "pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col, row/col or x/y)."
        )
    eprint(f"[DeepLinc] coord_csv has no header row; {header}")
    out.add_param("coord_header", header)
    return frame


def _duplicates(index) -> list:
    idx = pd.Index(index)
    return list(idx[idx.duplicated()].unique()[:3])


def _csv_sections(coord_df: pd.DataFrame, section_key, coord_csv: str):
    """Each cell's section label from ``coord_csv`` (``None`` without ``section_key``), or the stack refusal.

    The CSV path reads two named coordinate columns. A third axis it cannot use -- a ``z`` column, or a
    section-like column (``worker_utils.STACK_COLUMN_NAMES``) with 2..``MAX_SECTION_LEVELS`` levels --
    used to be dropped without a word, laying every section on one plane. Without ``section_key`` that
    is now refused with the sentence ``worker_utils.spatial_frame`` uses for an h5ad; with it, the run
    is per section by that column.
    """
    if section_key:
        if section_key not in coord_df.columns:
            raise ValueError(
                f"DeepLinc: section_key '{section_key}' is not a column of coord_csv {coord_csv}. Columns: "
                f"{[str(c) for c in coord_df.columns]}. Name the column that holds each cell's section label."
            )
        return coord_df[section_key].astype(str).to_numpy()
    by_lower = {str(c).lower(): c for c in coord_df.columns}
    candidates = [c for c in STACK_COLUMN_NAMES if c in coord_df.columns]
    if "z" in by_lower:
        candidates.append(by_lower["z"])
    for column in candidates:
        n = int(coord_df[column].astype(str).nunique())
        if 2 <= n <= MAX_SECTION_LEVELS or (str(column).lower() == "z" and n >= 2):
            raise ValueError(
                f"DeepLinc: this file holds {n} sections in coord_csv column '{column}'; a 2D run would overlay "
                f"them. Choose per-section 2D with `dims=2, section_key='{column}'`; DeepLinc is two-dimensional, "
                "so a 3D run is not offered."
            )
    return None


def _load_csvs(counts_csv: str, coord_csv: str, cell_type_csv: str, out: WorkerOutput, section_key=None):
    """Coordinates, raw labels, cell names, gene count, in-tissue counts and sections from the three CSVs.

    Returns ``(coords, labels, cell_names, n_genes, (n_supplied, n_off_tissue), sections)``; ``coord_csv``
    is read by :func:`_read_coord_table`, so Space Ranger's headerless positions file works as it is, and
    ``sections`` comes from :func:`_csv_sections` (a third axis is refused, never dropped).

    Only the counts file's row index and header are read: they name the cells and the genes, and
    the values are not used. Its orientation is decided by which axis carries the cell IDs the
    coordinate and cell-type files use -- the documented cells x genes, or the genes x cells that
    convert_h5ad_to_csv writes by default -- never by which axis is longer.
    """
    eprint(f"[DeepLinc] Loading from CSVs: counts={counts_csv}, coord={coord_csv}, cell_type={cell_type_csv}")
    # Three separate files, each of which may have come from a different pipeline, so each is
    # sniffed on its own. Read with a hardcoded comma, a tab-delimited file yields one index of
    # whole lines; the intersection below then finds no shared cells, which reads as a
    # barcode-convention mismatch -- the one thing it is not.
    counts_rows = pd.read_csv(counts_csv, index_col=0, usecols=[0], sep=sniff_tabular_sep(counts_csv)).index
    counts_cols = pd.read_csv(counts_csv, index_col=0, nrows=0, sep=sniff_tabular_sep(counts_csv)).columns
    coord_df = _read_coord_table(coord_csv, out)
    ct_df = pd.read_csv(cell_type_csv, index_col=0, sep=sniff_tabular_sep(cell_type_csv))
    if ct_df.shape[1] < 1:
        raise ValueError(f"cell_type_csv {cell_type_csv} has no label column after its index column.")

    coord_df.index = coord_df.index.astype(str)
    ct_df.index = ct_df.index.astype(str)
    row_ids = pd.Index(counts_rows.astype(str))
    col_ids = pd.Index(counts_cols.astype(str))
    for label, ids in (("coord_csv", coord_df.index), ("cell_type_csv", ct_df.index)):
        dup = _duplicates(ids)
        if dup:
            raise ValueError(
                f"{label} lists some cell IDs more than once (e.g. {dup}); each cell must appear once so its "
                "coordinates and label line up."
            )

    labelled = coord_df.index.intersection(ct_df.index)
    if len(labelled) == 0:
        raise ValueError(
            id_mismatch_msg("cell IDs", "coord_csv", list(coord_df.index), "cell_type_csv", list(ct_df.index))
        )
    by_rows = row_ids.intersection(labelled)
    by_cols = col_ids.intersection(labelled)
    if len(by_rows) == 0 and len(by_cols) == 0:
        raise ValueError(
            id_mismatch_msg("cell IDs", "counts_csv (rows or columns)", list(row_ids), "coord_csv", list(labelled))
        )
    if len(by_cols) > len(by_rows):
        orientation = "genes_x_cells (cell IDs are the counts columns)"
        cell_ids, n_genes = col_ids, len(row_ids)
    else:
        orientation = "cells_x_genes"
        cell_ids, n_genes = row_ids, len(col_ids)
    dup = _duplicates(cell_ids)
    if dup:
        raise ValueError(f"counts_csv lists some cell IDs more than once (e.g. {dup}).")
    out.add_param("counts_orientation", orientation)
    eprint(f"[DeepLinc] counts_csv orientation: {orientation}")

    common = cell_ids.intersection(coord_df.index).intersection(ct_df.index)
    n_in = {"counts": len(cell_ids), "coords": len(coord_df.index), "cell_types": len(ct_df.index)}
    eprint(
        "[DeepLinc] Common cells across files: {} (counts {}, coords {}, cell types {})".format(
            len(common), n_in["counts"], n_in["coords"], n_in["cell_types"]
        )
    )
    out.set_data(
        n_cells_in_counts=n_in["counts"], n_cells_in_coords=n_in["coords"], n_cells_in_cell_types=n_in["cell_types"]
    )
    left_out = {k: n - len(common) for k, n in n_in.items() if n > len(common)}
    if left_out:
        out.add_warning(
            "Only the {} cells present in all three files were analysed; left out: {}.".format(
                len(common), ", ".join(f"{n} from {k}" for k, n in left_out.items())
            )
        )

    coord_df = coord_df.loc[common]
    ct_df = ct_df.loc[common]
    # A coordinate file that carries Space Ranger's in_tissue flag (tissue_positions*.csv, or the
    # converter's metadata.csv of a CELLxGENE export) marks the background; it is left out here as in
    # h5ad mode. Counted among the cells shared by all three files, so a positions file that lists every
    # array spot beside counts of the tissue alone reports nothing dropped.
    n_supplied, n_off_tissue = len(common), 0
    if "in_tissue" in coord_df.columns:
        import anndata as ad

        flags = ad.AnnData(obs=pd.DataFrame({"in_tissue": coord_df["in_tissue"].values}, index=list(coord_df.index)))
        flags, n_supplied, n_off_tissue = keep_in_tissue(flags, "spots")
        if n_off_tissue:
            kept = pd.Index(flags.obs_names)
            coord_df = coord_df.loc[kept]
            ct_df = ct_df.loc[kept]
            common = kept
            eprint(
                f"[DeepLinc] Left out {n_off_tissue} of {n_supplied} spots with in_tissue == 0 in coord_csv (background)"
            )
    # By name, not by position: this file is one the user already had. Space Ranger's own
    # tissue_positions.csv leads with in_tissue, array_row, so the first two columns are a
    # constant and a lattice index -- a k-NN graph on those carries no spatial information, and
    # every score is a function of that graph. The run would still exit 0.
    x_col, y_col = resolve_coord_columns(coord_df.columns, coord_csv)
    eprint(f"[DeepLinc] Coordinates read from columns: {x_col}, {y_col}")
    out.add_param("coord_columns_used", f"{x_col},{y_col}")
    coords = coord_df[[x_col, y_col]].values.astype(np.float64)
    sections = _csv_sections(coord_df, section_key, coord_csv)
    labels = np.asarray(ct_df.iloc[:, 0].astype(object), dtype=object)
    return coords, labels, [str(c) for c in common], int(n_genes), (n_supplied, n_off_tissue), sections


def section_dir_name(label) -> str:
    """``section_<label>`` with every character a path cannot safely carry replaced by ``_``."""
    return "section_" + re.sub(r"[^A-Za-z0-9._-]+", "_", str(label))


def _section_dirs(labels) -> dict:
    """``{label: folder name}``, refusing two labels that would share one folder."""
    names = {}
    for label in labels:
        name = section_dir_name(label)
        clash = [other for other, used in names.items() if used == name]
        if clash:
            raise ValueError(
                f"DeepLinc: sections '{clash[0]}' and '{label}' would both be written to the folder {name}/; "
                "rename one of them in the section column."
            )
        names[label] = name
    return names


def _analyse(coords, cell_types, cell_names, n_neighbors: int, outdir: Path, out: WorkerOutput, where: str = ""):
    """Score one plane: the k-NN graph, observed/expected per type pair, the permutation test, the files.

    ``where`` prefixes messages with the section they are about (empty for a whole-file run). Returns a
    dict of what the payload reports.
    """
    unique_types = sorted(set(cell_types.tolist()))
    n_types = len(unique_types)
    n_cells = len(cell_types)
    eprint(f"[DeepLinc] {where}Found {n_types} cell types across {n_cells} cells")

    # ---- Build spatial adjacency (sparse) ----
    eprint(f"[DeepLinc] {where}Building spatial k-NN graph with {n_neighbors} neighbors ...")
    try:
        adj = _build_spatial_adjacency(coords, n_neighbors=n_neighbors)
    except ValueError as exc:
        raise ValueError(f"{where}{exc}") from exc
    u, v = _edge_endpoints(adj)
    n_edges = int(len(u))
    eprint(f"[DeepLinc] {where}Spatial graph: {n_cells} nodes, {n_edges} undirected edges")

    # ---- Observed / expected per type pair ----
    type_to_idx = {ct: i for i, ct in enumerate(unique_types)}
    codes = np.array([type_to_idx[ct] for ct in cell_types], dtype=np.int64)
    observed_counts = _edge_type_counts(u, v, codes, n_types)
    expected = _expected_edge_counts(np.bincount(codes, minlength=n_types), n_edges, n_cells)
    observed = _scores_from_counts(observed_counts, expected)
    interaction_scores = pd.DataFrame(observed, index=unique_types, columns=unique_types)

    # ---- Permutation test ----
    eprint(f"[DeepLinc] {where}Running label-permutation test ({N_PERMUTATIONS} permutations) ...")
    pvalues = _permutation_pvalues(u, v, codes, n_types, observed_counts, N_PERMUTATIONS)
    pval_df = pd.DataFrame(pvalues, index=unique_types, columns=unique_types)
    sig_interactions = _significant_pairs(observed, pvalues, unique_types)

    # ---- Save outputs ----
    edges_csv = outdir / "deeplinc_adjacency_edges.csv"
    _write_edge_list(u, v, cell_names, edges_csv)
    eprint(f"[DeepLinc] Saved {n_edges} edges to {edges_csv}")

    adj_csv = outdir / "deeplinc_adjacency.csv"
    dense_written = n_cells <= DENSE_ADJACENCY_MAX_CELLS
    if dense_written:
        _write_dense_adjacency(adj, cell_names, adj_csv)
        eprint(f"[DeepLinc] Saved dense adjacency matrix to {adj_csv}")
    else:
        n_entries = float(n_cells) * n_cells
        out.add_warning(
            f"{where}deeplinc_adjacency.csv was not written: a dense {n_cells} x {n_cells} matrix is "
            f"{n_entries:.3g} entries (the dense copy is written up to {DENSE_ADJACENCY_MAX_CELLS} cells). The "
            "same graph is in deeplinc_adjacency_edges.csv (cell_a, cell_b; one row per undirected edge)."
        )

    scores_csv = outdir / "deeplinc_interaction_scores.csv"
    _atomic_to_csv(interaction_scores, scores_csv)
    eprint(f"[DeepLinc] Saved interaction scores to {scores_csv}")

    pval_csv = outdir / "deeplinc_pvalues.csv"
    _atomic_to_csv(pval_df, pval_csv)
    eprint(f"[DeepLinc] Saved p-values to {pval_csv}")

    sig_csv = outdir / "deeplinc_significant_interactions.csv"
    _write_significant(sig_interactions, sig_csv)
    eprint(f"[DeepLinc] Saved {len(sig_interactions)} significant interactions to {sig_csv}")

    files = {
        "adjacency_edges_csv": str(edges_csv),
        "interaction_scores_csv": str(scores_csv),
        "pvalues_csv": str(pval_csv),
        "significant_interactions_csv": str(sig_csv),
    }
    if dense_written:
        files["adjacency_csv"] = str(adj_csv)
    return {
        "unique_types": unique_types,
        "n_types": n_types,
        "n_cells": n_cells,
        "n_edges": n_edges,
        "significant": sig_interactions,
        "dense_written": bool(dense_written),
        "files": files,
    }


def run_deeplinc(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    annotation_key: str = "cell_type",
    n_neighbors: int = 10,
    n_hvg: int | None = None,
    random_seed: int = 0,
    counts_csv: str = "",
    coord_csv: str = "",
    cell_type_csv: str = "",
    drop_unlabeled: bool = False,
    dims: int = 2,
    section_key: str | None = None,
) -> dict[str, Any]:
    """k-NN edge-type enrichment with a label-permutation test (the DeepLinc VGAE is not run).

    Two-dimensional by design: ``dims=3`` is refused (:data:`TWO_D_ONLY`). With ``section_key`` the run is
    per section -- one ``section_<label>/`` folder of the usual files per section, plus one long
    ``deeplinc_significant_interactions.csv`` at the top level with a leading ``section`` column.
    """
    if int(dims) == 3:
        raise ValueError(TWO_D_ONLY)
    if int(dims) != 2:
        raise ValueError(f"DeepLinc: dims must be 2 (or 3, which is refused), not {dims}.")
    section_key = section_key or None

    _ensure_dir(output_dir)
    np.random.seed(random_seed)
    outdir = Path(output_dir)
    out = WorkerOutput("deeplinc", task="cell_interaction_network")

    # ---- Load data (background spots, obs/coord_csv in_tissue == 0, are left out here) ----
    if st_h5ad and st_h5ad.strip():
        coords, labels, cell_names, n_genes, tissue, sections, frame = _load_h5ad(
            st_h5ad, spatial_key, annotation_key, section_key
        )
        label_source = f"obs['{annotation_key}']"
        tissue_source = "obs['in_tissue']"
    elif counts_csv and coord_csv and cell_type_csv:
        coords, labels, cell_names, n_genes, tissue, sections = _load_csvs(
            counts_csv, coord_csv, cell_type_csv, out, section_key
        )
        order = [str(s) for s in pd.unique(sections)] if sections is not None else None
        frame = Frame("coord_csv", 2, (None, None), None, section_key, order)
        label_source = "cell_type_csv"
        tissue_source = "coord_csv's in_tissue column"
    else:
        raise ValueError("Must provide either --st-h5ad or all of --counts-csv, --coord-csv, --cell-type-csv")
    n_supplied, n_off_tissue = tissue
    record_in_tissue(out, n_supplied, n_off_tissue)

    n_input = len(labels)
    keep, n_dropped = _labels_or_raise(labels, drop_unlabeled, label_source)
    if n_dropped:
        coords = coords[keep]
        labels = labels[keep]
        cell_names = [c for c, k in zip(cell_names, keep) if k]
        if sections is not None:
            sections = sections[keep]
    cell_types = np.array([str(x) for x in labels], dtype=object)

    per_result: dict = {}
    if section_key:
        import anndata as ad

        light = ad.AnnData(
            obs=pd.DataFrame(
                {"_sog_type": cell_types, "_sog_section": sections}, index=pd.Index(cell_names, dtype=object)
            )
        )
        light.obsm["_sog_xy"] = np.asarray(coords, dtype=np.float64)
        labels_in_order = frame.sections or [str(s) for s in pd.unique(sections)]
        dirs = _section_dirs(labels_in_order)

        def run_one(sub, label):
            folder = _ensure_dir(str(outdir / dirs[label]))
            np.random.seed(random_seed)  # each section's permutations do not depend on the others
            res = _analyse(
                np.asarray(sub.obsm["_sog_xy"]),
                np.asarray(sub.obs["_sog_type"].to_numpy(), dtype=object),
                [str(n) for n in sub.obs_names],
                n_neighbors,
                folder,
                out,
                where=f"section '{label}': ",
            )
            res["folder"] = str(folder)
            per_result[label] = res
            return pd.DataFrame(res["significant"], columns=SIGNIFICANT_COLUMNS)

        long = per_section(light, "_sog_section", run_one, "DeepLinc")
        sig_csv = outdir / "deeplinc_significant_interactions.csv"
        _atomic_to_csv(long, sig_csv, index=False)
        eprint(f"[DeepLinc] Saved {len(long)} significant interactions over {len(per_result)} sections to {sig_csv}")
        section_order = [s for s in labels_in_order if s in per_result]
        sig_interactions = [
            {"section": str(r["section"]), **{c: r[c] for c in SIGNIFICANT_COLUMNS}} for r in long.to_dict("records")
        ]
        sig_interactions.sort(key=lambda x: x["score"], reverse=True)
        unique_types = sorted({t for r in per_result.values() for t in r["unique_types"]})
        n_cells = int(sum(r["n_cells"] for r in per_result.values()))
        n_edges = int(sum(r["n_edges"] for r in per_result.values()))
        dense_written = all(r["dense_written"] for r in per_result.values())
        files = {
            "significant_interactions_csv": str(sig_csv),
            "section_dirs": [per_result[s]["folder"] for s in section_order],
        }
        mode = "per-section-2d"
    else:
        res = _analyse(coords, cell_types, cell_names, n_neighbors, outdir, out)
        sig_interactions = res["significant"]
        unique_types, n_cells, n_edges = res["unique_types"], res["n_cells"], res["n_edges"]
        dense_written = res["dense_written"]
        files = res["files"]
        section_order = None
        mode = "2d"
    n_types = len(unique_types)

    # ---- Build output ----
    out.set_data(
        n_cells=n_cells,
        n_genes=n_genes,
        n_cell_types=n_types,
        n_edges=n_edges,
        n_cells_input=int(n_supplied),
        n_spots_off_tissue_dropped=int(n_off_tissue),
        n_cells_dropped_unlabeled=int(n_dropped),
        dense_adjacency_written=bool(dense_written),
    )
    if section_order is not None:
        out.set_data(
            n_sections=len(section_order),
            n_cells_per_section={s: int(per_result[s]["n_cells"]) for s in section_order},
            n_edges_per_section={s: int(per_result[s]["n_edges"]) for s in section_order},
        )
    out.add_output_files(files)
    out.add_params(
        {
            "spatial_key": spatial_key,
            "coords_key": spatial_key if st_h5ad and st_h5ad.strip() else "coord_csv",
            "dims": 2,
            "section_key": section_key,
            "mode": mode,
            "sections": section_order,
            "frame": frame.to_dict(),
            "annotation_key": annotation_key,
            "n_neighbors": n_neighbors,
            "n_permutations": N_PERMUTATIONS,
            "random_seed": random_seed,
            "drop_unlabeled": bool(drop_unlabeled),
            "deeplinc_model_run": False,
            "uses_expression": False,
            "score_definition": SCORE_DEFINITION,
            "pvalue_definition": PVALUE_DEFINITION,
            "significance": f"pvalue < {SIGNIFICANCE_ALPHA} and score > 1.0",
        }
    )
    record_method(out, METHOD_NAME, used_fallback=False)
    if n_hvg is not None:
        record_ignored(
            out,
            ["n_hvg"],
            "this tool scores the spatial graph and the cell-type labels only; no expression is read, "
            "so there are no genes to select",
        )
    if n_dropped:
        out.add_warning(
            f"{n_dropped} of {n_input} cells had no label in {label_source} and were left out (drop_unlabeled=True)."
        )
    summary = {
        "n_significant_interactions": len(sig_interactions),
        "cell_types": unique_types,
        "top_interactions": sig_interactions[:10],
    }
    if section_order is not None:
        summary["n_significant_interactions_per_section"] = {
            s: len(per_result[s]["significant"]) for s in section_order
        }
    out.set_summary(**summary)

    def _pair(i):
        text = "{}-{}".format(i["type_a"], i["type_b"])
        return f"{text} ({i['section']})" if "section" in i else text

    top_pairs = [_pair(i) for i in sig_interactions[:5]]
    dropped_note = f" {n_dropped} of {n_input} cells had no label and were left out." if n_dropped else ""
    if n_off_tissue:
        dropped_note = (
            f" {n_off_tissue} of the {n_supplied} spots supplied are background ({tissue_source} == 0) and were "
            "left out before the graph was built." + dropped_note
        )
    where = (
        " Run per section in 2D ({} sections of {}: {}); each section has its own graph, scores and test, "
        "and no edge joins two sections.".format(len(section_order), section_key, ", ".join(section_order))
        if section_order is not None
        else ""
    )
    out.set_analysis(
        "kNN edge-type enrichment with a label-permutation test (the DeepLinc VGAE was not run; gene "
        "expression is not used): {} cells, {} undirected spatial edges ({} nearest neighbours, "
        "symmetrised), {} cell types.{}{} Score = observed/expected edges between two types, where 1.0 "
        "is random placement. Found {} enriched pairs (score > 1.0 and permutation p < {}, {} "
        "permutations). Top pairs: {}.".format(
            n_cells,
            n_edges,
            n_neighbors,
            n_types,
            where,
            dropped_note,
            len(sig_interactions),
            SIGNIFICANCE_ALPHA,
            N_PERMUTATIONS,
            ", ".join(top_pairs) if top_pairs else "none identified",
        )
    )

    return out.to_dict()


def main():
    ap = argparse.ArgumentParser(
        description="k-NN cell-type edge enrichment with a label-permutation test (served as DeepLinc; "
        "the DeepLinc VGAE is not run)"
    )
    ap.add_argument("--st-h5ad", default="", help="Path to spatial AnnData (.h5ad)")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--spatial-key", default="spatial", help="obsm key for spatial coordinates")
    ap.add_argument(
        "--coords-key", default=None, help="obsm key for spatial coordinates (same as --spatial-key; wins when given)"
    )
    ap.add_argument("--dims", type=int, default=2, choices=[2, 3], help="2 (the only one this tool runs); 3 is refused")
    ap.add_argument(
        "--section-key",
        default=None,
        help="obs column (h5ad) or coord_csv column naming each cell's section; the run is then per section",
    )
    ap.add_argument("--annotation-key", default="cell_type", help="obs column for cell type labels")
    ap.add_argument("--n-neighbors", type=int, default=10, help="Number of spatial neighbors for graph")
    ap.add_argument(
        "--n-hvg",
        type=int,
        default=None,
        help="Accepted for compatibility and ignored (no expression is read); reported in params.ignored",
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--counts-csv", default="", help="Path to counts CSV (alternative input)")
    ap.add_argument("--coord-csv", default="", help="Path to coordinates CSV (alternative input)")
    ap.add_argument("--cell-type-csv", default="", help="Path to cell type CSV (alternative input)")
    ap.add_argument(
        "--drop-unlabeled",
        action="store_true",
        default=False,
        help="Leave out cells whose label is missing (NaN/empty) instead of failing",
    )
    args = ap.parse_args()

    try:
        result = run_deeplinc(
            st_h5ad=args.st_h5ad,
            output_dir=args.output_dir,
            spatial_key=args.coords_key or args.spatial_key,
            annotation_key=args.annotation_key,
            n_neighbors=args.n_neighbors,
            n_hvg=args.n_hvg,
            random_seed=args.seed,
            counts_csv=args.counts_csv,
            coord_csv=args.coord_csv,
            cell_type_csv=args.cell_type_csv,
            drop_unlabeled=args.drop_unlabeled,
            dims=args.dims,
            section_key=args.section_key,
        )
        print(json.dumps(result, default=str))

    except Exception as e:
        eprint(f"[DeepLinc] ERROR: {e}")
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("deeplinc", str(e), task="cell_interaction_network")
        sys.exit(1)


if __name__ == "__main__":
    main()
