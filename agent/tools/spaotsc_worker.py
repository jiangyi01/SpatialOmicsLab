#!/usr/bin/env python
"""
SpaOTsc worker (layer 2; runs in /opt/conda/envs/spaotsc).

Runs the upstream ``spaotsc`` package (``SpaOTsc.spatial_sc``). Inputs are either a folder with the
standard SpaOTsc tutorial filenames, explicit matrix paths, or -- when those are missing -- raw
scRNA-seq and spatial data (h5ad / h5 / csv / txt / tsv), from which this worker builds the four
matrices SpaOTsc takes (upstream ships no preprocessing of its own):

  - sc expression: CPM to 1e4 then log2(x + 1), on the genes the two inputs share
  - sc_dmat: exp(1 - Pearson r) between cells on up to 40 principal components
  - cost_matrix: exp(1 - Pearson r) between per-gene binarised profiles, where a cell/spot counts as
    expressing a gene when it is strictly above that gene's 70th percentile (a zero never does)
  - is_dmat: Euclidean distance between the spot coordinates (obsm['spatial'] or raw_is_coord_path;
    there is no substitute when neither exists -- the run stops)

Background spots are not mapped: spots with ``obs['in_tissue'] == 0`` in an h5ad raw_is_path (a CELLxGENE
Visium export carries every array spot), or flagged 0 by an ``in_tissue`` column of raw_is_coord_path, are
left out before any matrix is built and counted (``params.in_tissue_filter``, ``data.n_spots_off_tissue_dropped``).
The preprocessing treats X as counts, so which matrix is read follows ``worker_utils.choose_counts_matrix``:
X by default, ``adata.raw.X`` of each h5ad input that has one with ``--use-raw-counts``; a negative or
non-finite matrix is refused, a non-integer one runs with a warning (``params.expression_source``,
``params.x_matrix_kind``).

Then, in this order, each step needing the one before it:

  - mapping: ``transport_plan`` (cells x spots)
  - cell-cell distance: ``cell_cell_distance`` -- needs the mapping. Upstream solves one Sinkhorn
    problem per *pair* of cells, n_sc * (n_sc - 1) / 2 of them, so it dominates the run time.
  - clustering: ``clustering`` -- needs the cell-cell distance
  - signaling: ``spatial_signaling_ot`` -- needs the cell-cell distance and at least one ligand and
    one receptor that are in the shared gene panel. With downstream genes (ds_up / ds_down) it also
    runs ``nonspatial_correlation`` + ``infer_signal_range_ml`` over effect ranges 10 / 50 / 100 (in
    the units of is_dmat); without them the range step is not run and the payload says so.

A requested step that cannot run stops the run with an error; nothing is skipped in silence.

SpaOTsc is dense by construction: n_sc x n_sc, n_sc x n_is and n_is x n_is float64 matrices. The
worker estimates their peak before allocating them and refuses, with the numbers, when they cannot
fit. It never subsamples.

Outputs are written under --output-dir (``output_files`` lists only what THIS run wrote):
  - mapping/transport_plan.[npy|csv]
  - ccd/cell_cell_distance.[npy|csv]
  - clustering/labels.csv       (cell, cluster = spatial subcluster "i_j", expression_cluster = i)
  - signaling/signaling_scores.[npy|csv], signaling/inferred_range.[npy|csv]
  - precomputed/*               (only when built from raw input)
  - meta/meta_inputs.json, meta/run_summary.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
from worker_utils import (
    MAX_SECTION_LEVELS,
    STACK_COLUMN_NAMES,
    Frame,
    WorkerOutput,
    available_memory_bytes,
    choose_counts_matrix,
    expression_matrix_kind,
    id_mismatch_msg,
    keep_in_tissue,
    make_names_unique_and_report,
    read_indexed_table,
    read_tissue_positions,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    resolve_coord_columns,
    sniff_tabular_sep,
    spatial_frame,
)

#: ``params.mode`` values: a 3D distance in the aligned frame, or one plane. SpaOTsc builds ONE spot-to-spot
#: distance matrix over the whole input, so it has no per-section mode: a stack in 2D is refused.
MODE_3D = "3d"
MODE_2D = "2d"

#: What runs. The matrices may be built by this worker, but every step is the upstream package.
METHOD_NAME = "SpaOTsc (upstream spaotsc.SpaOTsc.spatial_sc)"

#: How this worker builds SpaOTsc's inputs from raw data; published whenever it does.
PREPROCESSING = (
    "built by this worker from raw input: expression CPM to 1e4 then log2(x+1) on the shared gene panel; "
    "sc_dmat = exp(1 - Pearson r) between cells on up to 40 principal components; cost_matrix = "
    "exp(1 - Pearson r) between binarised profiles (expressed = strictly above the gene's 70th percentile); "
    "is_dmat = Euclidean distance between spot coordinates"
)

#: The binarisation rule behind the cost matrix. Strict: with ``>=`` every gene detected in fewer than
#: 30% of cells had a 70th percentile of 0, so every cell -- zero counts included -- "expressed" it.
BINARIZE_RULE = ">q70 per gene (strict; a zero never counts as expressed)"
BINARIZE_QUANTILE = 0.7

#: Spatial ranges ``infer_signal_range_ml`` scores, as the upstream README uses them, in is_dmat units.
EFFECT_RANGES = (10.0, 50.0, 100.0)

#: Upstream ``cell_cell_distance(n_landmark=100)``: with use_landmark it picks this many spots.
N_LANDMARK = 100

#: Upstream ``clustering`` crashes with its own default ``pca_n_components=None`` (it reads an
#: ``X_pca`` it only assigns when a component count is given), so one is always passed.
CLUSTERING_PCA_COMPONENTS = 40

#: Upstream ``clustering`` builds a 50-nearest-neighbour graph on the spatial cell-cell distance.
CLUSTERING_KNN = 50

#: Dense float64 work arrays alive at once during ``transport_plan`` (cost, weights, plan, the
#: Gromov-Wasserstein constants and products, the unbalanced-Sinkhorn temporaries), in n_sc x n_is units.
MAPPING_SC_IS_COPIES = 16


#: Mean normalised row entropy above which a transport plan is reported as not localising cells.
UNIFORM_PLAN_ENTROPY = 0.99


def _mean_row_entropy(plan: np.ndarray) -> float:
    """Mean over cells of the entropy of each cell's spot distribution, divided by log(n_spots)."""
    n_is = plan.shape[1]
    if n_is < 2:
        return 0.0
    total = 0.0
    n_rows = 0
    for start in range(0, plan.shape[0], 1024):  # row blocks: no second full-size copy of the plan
        block = np.asarray(plan[start : start + 1024], dtype=np.float64)
        sums = block.sum(axis=1, keepdims=True)
        keep = sums[:, 0] > 0
        r = block[keep] / sums[keep]
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.where(r > 0, r * np.log(r), 0.0).sum(axis=1)
        total += float(h.sum())
        n_rows += int(keep.sum())
    return total / max(n_rows, 1) / float(np.log(n_is))


def log(msg: str) -> None:
    print(f"[spaotsc-worker] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- atomic writes


def _atomic_np_save(path: str, arr) -> str:
    tmp = path + ".partial"
    with open(tmp, "wb") as fh:  # a file handle, so np.save does not append ".npy" to the temp name
        np.save(fh, arr)
    os.replace(tmp, path)
    return path


def _atomic_to_csv(df: pd.DataFrame, path: str, **kwargs) -> str:
    tmp = path + ".partial"
    df.to_csv(tmp, **kwargs)
    os.replace(tmp, path)
    return path


def _atomic_savetxt(path: str, arr) -> str:
    tmp = path + ".partial"
    np.savetxt(tmp, arr)
    os.replace(tmp, path)
    return path


def _atomic_json(path: str, obj) -> str:
    tmp = path + ".partial"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)
    return path


def _save_array(out_path: str, arr: np.ndarray) -> list[str]:
    """Write ``arr`` as ``<name>.npy`` and ``<name>.csv``; return both paths."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    npy = _atomic_np_save(os.path.splitext(out_path)[0] + ".npy", arr)
    csv = _atomic_to_csv(pd.DataFrame(arr), out_path, index=False)
    return [npy, csv]


# ----------------------------------------------------------------------------- readers


def _load_matrix(path: str) -> np.ndarray:
    p = path.lower()
    if p.endswith(".npy"):
        return np.load(path)
    # np.savetxt (how dm_is.txt is written) separates with single spaces; the sniffer names that.
    try:
        return pd.read_csv(path, sep=sniff_tabular_sep(path), header=None).values
    except Exception:
        return np.loadtxt(path)


def _load_table(path: str) -> pd.DataFrame:
    """A cells x genes table whose first column is the cell ID, separator sniffed."""
    return read_indexed_table(path, what="sc expression table (cells x genes)")


def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _expect(msg: str, cond: bool, missing: list[str], key: str) -> None:
    if not cond:
        missing.append(key)
        log(f"Missing: {msg}")


def _frame_args(args) -> dict:
    """The three coordinate arguments of a run, as ``_coords_from_frame`` takes them."""
    return {
        "coords_key": getattr(args, "coords_key", None) or "spatial",
        "dims": int(getattr(args, "dims", 2) or 2),
        "section_key": getattr(args, "section_key", None) or None,
    }


def _coords_from_frame(adata, path: str, coords_key="spatial", dims=2, section_key=None) -> pd.DataFrame | None:
    """``obsm[coords_key]`` read through ``worker_utils.spatial_frame``, as a frame on obs_names.

    is_dmat is a Euclidean distance over every column returned: two in 2D, three in 3D, where the frame is
    converted to micrometres and its z must be measured or registered (``spatial_frame`` refuses a rank z, an
    undeclared frame and a three-column ``spatial``). A missing default ``spatial`` returns None so the
    caller can say there are no coordinates; a missing named key is ``spatial_frame``'s KeyError.

    A 2D run over a stack of sections is refused by ``spatial_frame(per_section_ok=False)``, with a
    ``section_key`` or without: SpaOTsc builds ONE spot-to-spot distance matrix over the whole input, so a
    per-section run would still put the sections on one plane in it.

    ``.attrs`` carry ``frame`` (the Frame) and ``mode`` ("3d" / "2d") for the run's provenance.
    """
    if coords_key == "spatial" and "spatial" not in adata.obsm:
        return None
    # SpaOTsc does not run per section (ruling): per_section_ok=False refuses a 2D section_key over two or
    # more sections, and every refusal offers the 3D frame or a file subset to one section instead.
    coords, frame = spatial_frame(adata, coords_key, dims, section_key, "spaotsc", per_section_ok=False)
    names = ["x", "y", "z"] if coords.shape[1] == 3 else ["x", "y"]
    out = pd.DataFrame(coords, index=pd.Index([str(i) for i in adata.obs_names]), columns=names)
    out.attrs["coord_columns_used"] = f"obsm['{coords_key}'] ({coords.shape[1]} axes)"
    out.attrs["coord_alignment"] = "row order of the AnnData"
    out.attrs["frame"] = frame
    out.attrs["mode"] = MODE_3D if dims == 3 else MODE_2D
    return out


def _frame_params(frame, mode: str) -> dict:
    """``params.mode`` / ``params.frame`` for the run's provenance; ``frame`` None for a coordinate table."""
    if frame is None:
        frame = Frame("raw_is_coord_path", 2, (None, None), None, None, None)
    return {
        "mode": mode,
        "frame": frame.to_dict(),
        "dims": frame.dims,
        "section_key": frame.section_key,
    }


def _is_dmat_units(frame) -> str:
    """What one unit of is_dmat (and of the effect ranges) is: micrometres when every axis was converted."""
    if frame is not None and all(u is not None for u in frame.units_per_axis):
        return "um"
    return "the coordinates' own units (not declared)"


def _counts_of_anndata(adata, role: str, use_raw_counts: bool):
    """``(adata, info)``: the matrix this worker's CPM + log2 preprocessing runs on, for one AnnData input.

    ``worker_utils.choose_counts_matrix`` makes the choice: X, or ``adata.raw.X`` with ``use_raw_counts``;
    a negative or non-finite X is refused (it is not counts, and CPM + log2 of it is NaN), naming
    ``use_raw_counts`` when ``adata.raw`` holds counts; a non-negative non-integer X runs with a warning.
    ``use_raw_counts`` is applied per input, as celldart applies it: an input without ``adata.raw`` keeps
    X, and ``info['note']`` says so.
    """
    want_raw = bool(use_raw_counts)
    note = ""
    if want_raw and getattr(adata, "raw", None) is None:
        want_raw = False
        note = f"use_raw_counts=True, but the {role} h5ad has no adata.raw, so its X was used."
    try:
        adata, info = choose_counts_matrix(adata, want_raw)
    except ValueError as exc:
        raise ValueError(f"{role}: {exc}") from exc
    info = dict(info)
    info["note"] = note
    return adata, info


def _counts_of_table(X, role: str, use_raw_counts: bool) -> dict:
    """The same decision for a table input (csv/tsv/txt or a pandas HDF table), which has no ``adata.raw``."""
    X = np.asarray(X)
    if X.dtype.kind not in "biuf":
        raise ValueError(
            f"{role}: the expression table holds non-numeric values (dtype {X.dtype}); every column after the "
            "first (the cell/spot ID) must be a gene of numbers."
        )
    kind = expression_matrix_kind(X)
    if kind in ("negative", "nonfinite"):
        what = "negative values (scaled or z-scored data)" if kind == "negative" else "NaN or infinite values"
        raise ValueError(
            f"{role}: the expression table holds {what}, not counts, and this worker CPM-normalises and "
            "log2-transforms it as counts. Supply raw counts (a table has no adata.raw to read them from)."
        )
    warning = None
    if kind == "nonnegative_noninteger":
        warning = (
            "X holds non-integer values (normalised or log-transformed data?), and this tool normalises X as counts, "
            "so the result was computed on a matrix normalised twice. Supply raw counts."
        )
    note = (
        f"use_raw_counts=True, but {role} is a table with no adata.raw, so it was used as given."
        if use_raw_counts
        else ""
    )
    return {"expression_source": "table", "x_matrix_kind": kind, "warning": warning, "note": note}


def _read_raw(
    path: str,
    role: str = "input",
    spatial: bool = False,
    use_raw_counts: bool = False,
    report=None,
    frame_args=None,
):
    """``(X, obs_names, var_names, coords_or_None, n_genes_renamed)`` with X as stored.

    A sparse matrix stays sparse here: only the shared gene panel is ever densified (SpaOTsc takes a
    dense frame), so a 30k-gene reference is not materialised in full to keep 5k of its columns.

    - .h5ad: AnnData; coordinates from obsm[coords_key] through ``_coords_from_frame`` when
      ``frame_args`` (``_frame_args``) is given, else none are read.
    - .h5/.hdf5: AnnData, else 10x HDF5, else a pandas HDF table (no coordinates).
    - text (csv/tsv/txt): rows = cells/spots, columns = genes, first column = ID; no coordinates.

    ``role`` names the input in messages (``raw_sc_path`` / ``raw_is_path``). For the spatial input
    (``spatial=True``) an AnnData's ``obs['in_tissue'] == 0`` spots -- the background glass a CELLxGENE
    Visium export carries -- are left out right after loading (``worker_utils.keep_in_tissue``). Which
    matrix is read, and whether it holds counts, is decided by ``_counts_of_anndata`` /
    ``_counts_of_table``. ``report`` (a dict, when given) receives ``counts`` (that decision) and
    ``in_tissue`` (``n_supplied``, ``n_dropped``).
    """
    import scanpy as sc  # type: ignore

    low = path.lower()
    rep = {} if report is None else report

    def _from_anndata(adata, with_coords):
        n_supplied, n_off = int(adata.n_obs), 0
        if spatial:
            adata, n_supplied, n_off = keep_in_tissue(adata, "spots")
        rep["in_tissue"] = {"n_supplied": int(n_supplied), "n_dropped": int(n_off)}
        rep["stack"] = _stack_in(adata.obs) if spatial else (0, "")
        adata, rep["counts"] = _counts_of_anndata(adata, role, use_raw_counts)
        renamed = make_names_unique_and_report(adata, axes=("var",))
        coords = _coords_from_frame(adata, path, **frame_args) if with_coords and frame_args is not None else None
        return adata.X, list(adata.obs_names), list(adata.var_names), coords, int(renamed["n_genes_renamed"])

    def _from_table(X, obs_names, var_names):
        rep["in_tissue"] = {"n_supplied": len(obs_names), "n_dropped": 0}
        rep["counts"] = _counts_of_table(X, role, use_raw_counts)
        return X, obs_names, var_names, None, 0

    if low.endswith(".h5ad"):
        return _from_anndata(sc.read_h5ad(path), True)

    if low.endswith(".h5") or low.endswith(".hdf5"):
        # Format detection, not a method fallback: the three readers accept disjoint layouts. Only the
        # read is guarded, so a problem in a file that did parse (a malformed obsm['spatial']) is reported
        # as itself rather than sending the file on to the next reader.
        for reader, with_coords in ((sc.read_h5ad, True), (sc.read_10x_h5, False)):
            try:
                adata = reader(path)
            except Exception:
                continue
            return _from_anndata(adata, with_coords)
        try:
            df = pd.read_hdf(path)
        except Exception as e:
            raise RuntimeError(f"Could not parse raw matrix h5/hdf5 file: {path}: {e}") from e
        return _from_table(df.values, list(df.index), list(df.columns))

    df = read_indexed_table(path, what="expression matrix (rows = cells/spots, columns = genes)")
    return _from_table(df.values, list(df.index), list(df.columns))


def _tissue_mask(values) -> np.ndarray:
    """``in_tissue`` values as a boolean mask, by worker_utils.keep_in_tissue's rule (True/"1"/1 = in tissue)."""
    text = pd.Series(np.asarray(values, dtype=object)).astype(str).str.strip().str.lower()
    flag = pd.to_numeric(text.replace({"true": "1", "false": "0"}), errors="coerce")
    return np.asarray(flag == 1)


def _dense_subset(X, obs_names, var_names, genes: list[str]) -> pd.DataFrame:
    """The ``genes`` columns of X as a dense float64 frame -- the one densification SpaOTsc needs."""
    pos = pd.Index([str(v) for v in var_names]).get_indexer(genes)
    if (pos < 0).any():
        raise KeyError(f"genes missing from the matrix: {[g for g, p in zip(genes, pos) if p < 0][:5]}")
    sub = X[:, pos]
    sub = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
    return pd.DataFrame(np.asarray(sub, dtype=np.float64), index=[str(o) for o in obs_names], columns=list(genes))


def _looks_like_a_header(values) -> bool:
    """Row 0 is a header when any field after the first does not parse as a number."""
    for value in list(values)[1:]:
        try:
            float(str(value).strip())
        except (TypeError, ValueError):
            return True
    return False


def _is_numeric(series: pd.Series) -> bool:
    return bool(np.issubdtype(series.dtype, np.number))


def _load_raw_coords(path: str, index: pd.Index | None = None) -> pd.DataFrame:
    """Spot coordinates from a csv/tsv/txt file, aligned to ``index`` by spot ID.

    An ``in_tissue`` column is never read as a coordinate; its flag comes back in
    ``.attrs['in_tissue_mask']`` (one boolean per returned row; None without the column).

    The coordinate columns are chosen by *name* (``resolve_coord_columns``: pixel columns before the
    array lattice, then row/col, then x/y), not by position: a Space Ranger ``tissue_positions`` file or
    a converter ``metadata.csv`` leads with ``in_tissue, array_row``, and taking its first two numeric
    columns gave a tissue flag and a lattice index as "x/y". Rows are matched to ``index`` by the ID
    column (the first column, when it is an ID); a file without one is aligned by position and only
    when its row count equals the number of spots. A missing ID is an error, never a silent shift.

    Returns a frame indexed like ``index`` (when given); ``.attrs`` says which columns were read and
    how the rows were aligned.
    """
    sep = sniff_tabular_sep(path)
    head = pd.read_csv(path, sep=sep, header=None, nrows=1)
    has_header = _looks_like_a_header(head.iloc[0]) if head.shape[1] > 1 else False
    frame = pd.read_csv(path, sep=sep, header=0 if has_header else None)

    if not has_header and frame.shape[1] >= 6 and not _is_numeric(frame.iloc[:, 0]):
        # Pre-2.0 Space Ranger tissue_positions_list.csv: headerless, positional, documented by 10x.
        frame = read_tissue_positions(path)
        has_header = True

    if has_header:
        first = str(frame.columns[0]).strip().lower()
        id_col = frame.columns[0] if (not _is_numeric(frame.iloc[:, 0]) or first.startswith("unnamed")) else None
        if id_col is None and first in (
            "barcode",
            "barcodes",
            "spot",
            "spot_id",
            "spotid",
            "cell",
            "cell_id",
            "id",
            "index",
        ):
            id_col = frame.columns[0]
        candidates = [c for c in frame.columns if c != id_col]
        cx, cy = resolve_coord_columns(candidates, path)
        cols = [cx, cy]
        if str(cx).strip().lower() == "x" and "z" in [str(c).strip().lower() for c in candidates]:
            cols.append(next(c for c in candidates if str(c).strip().lower() == "z"))
    else:
        id_col = frame.columns[0] if not _is_numeric(frame.iloc[:, 0]) else None
        cols = [c for c in frame.columns if c != id_col]
        if len(cols) not in (2, 3):
            raise ValueError(
                f"{path} has no header and {len(cols)} numeric column(s) besides the spot ID; without names "
                "the coordinates cannot be told apart from other columns. Add a header naming them "
                "(imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col, row/col or x/y)."
            )
    for c in cols:
        if not _is_numeric(frame[c]):
            raise ValueError(f"{path}: coordinate column {c!r} is not numeric (first values: {list(frame[c][:3])}).")

    coords = frame[cols].astype(np.float64)
    coords.columns = [str(c) for c in cols]
    used = [str(c) for c in cols]
    # Space Ranger's tissue_positions and the converter's metadata.csv carry the in_tissue flag. It is read
    # beside the coordinates, aligned the same way, and handed back in .attrs: never a coordinate column.
    flag_col = next((c for c in frame.columns if c != id_col and str(c).strip().lower() == "in_tissue"), None)
    tissue = None if flag_col is None else pd.Series(frame[flag_col].values, index=coords.index)

    if id_col is not None:
        ids = frame[id_col].astype(str)
        if ids.duplicated().any():
            dup = ids[ids.duplicated()].unique()[:3].tolist()
            raise ValueError(
                f"{path}: spot IDs repeat in column {id_col!r} (e.g. {dup}); rows cannot be matched to spots."
            )
        coords.index = pd.Index(ids.values)
        if tissue is not None:
            tissue.index = coords.index
        alignment = f"by spot ID (column {str(id_col)!r})"
        if index is not None:
            want = pd.Index([str(i) for i in index])
            present = want.isin(coords.index)
            if not present.all():
                raise ValueError(
                    id_mismatch_msg(
                        "spot IDs",
                        "raw_is_path",
                        list(want),
                        "raw_is_coord_path",
                        list(coords.index),
                        n_common=int(present.sum()),
                    )
                    + f" {int((~present).sum())} of {len(want)} spots have no coordinates."
                )
            coords = coords.loc[want]
            if tissue is not None:
                tissue = tissue.loc[want]
    else:
        alignment = "by row position (the file has no spot-ID column)"
        if index is not None:
            if len(index) != coords.shape[0]:
                raise ValueError(
                    f"{path} has {coords.shape[0]} coordinate rows and no spot-ID column, and raw_is_path has "
                    f"{len(index)} spots; rows can only be matched by position when the counts agree. Add a "
                    "spot-ID column as the first column."
                )
            coords.index = pd.Index([str(i) for i in index])
    coords.attrs["coord_columns_used"] = used
    coords.attrs["coord_alignment"] = alignment
    # None when the file has no in_tissue column; else one boolean per row of ``coords``, in its order.
    coords.attrs["in_tissue_mask"] = None if tissue is None else _tissue_mask(tissue.values)
    coords.attrs["in_tissue_column"] = None if flag_col is None else str(flag_col)
    coords.attrs["stack"] = _stack_in(frame)
    return coords


def _stack_in(table) -> tuple:
    """``(n, column)`` of the first section-like column (``worker_utils.STACK_COLUMN_NAMES``) of ``table`` with
    2..``MAX_SECTION_LEVELS`` levels, or ``(0, "")``: the same rule ``spatial_frame`` applies to an h5ad's obs."""
    columns = [str(c) for c in getattr(table, "columns", [])]
    for name in STACK_COLUMN_NAMES:
        if name in columns:
            n = int(table[name].astype(str).nunique())
            if 2 <= n <= MAX_SECTION_LEVELS:
                return n, name
    return 0, ""


def _refuse_a_stacked_coordinate_table(raw_is_coord_path: str, coords, is_read: dict) -> None:
    """A coordinate table is two columns on one plane; a stack of sections behind it is refused, never overlaid.

    The sections are read from the table's own section-like column, then from the h5ad raw_is_path's obs (the
    table's rows are that file's spots). SpaOTsc does not run per section, so the ways out are the 3D frame of
    an h5ad or a single-section subset -- as DeepLinc's CSV path refuses the same file.
    """
    for (n, column), where in (
        (coords.attrs.get("stack") or (0, ""), "raw_is_coord_path column"),
        (is_read.get("stack") or (0, ""), "raw_is_path obs column"),
    ):
        if n >= 2:
            raise ValueError(
                f"spaotsc: this file holds {n} sections in {where} '{column}'; a 2D run would overlay them, because "
                f"raw_is_coord_path ({raw_is_coord_path}) is read as one plane. Choose 3D with an h5ad raw_is_path "
                f"and `coords_key=<aligned frame>, dims=3`, or run with `dims=2` on files subset to one '{column}' "
                "label; spaotsc does not run per section."
            )


# ----------------------------------------------------------------------------- matrices


def _normalize_log2(df: pd.DataFrame) -> pd.DataFrame:
    """CPM normalize to 1e4 and log2-transform."""
    lib = df.sum(axis=1)
    lib.replace(0, np.nan, inplace=True)
    df_cpm = df.div(lib, axis=0) * 1e4
    df_cpm = df_cpm.fillna(0.0)
    return np.log2(df_cpm + 1.0)


def _binarize_quantile(df: pd.DataFrame, q: float = BINARIZE_QUANTILE) -> pd.DataFrame:
    """1 where a value is strictly above its gene's q-quantile (see ``BINARIZE_RULE``)."""
    thr = df.quantile(q, axis=0)
    return (df.gt(thr, axis=1)).astype(int)


def _pearson_corr_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute row-wise Pearson correlation between A (n×g) and B (m×g),
    returns (n×m).
    """
    A_mean = A.mean(axis=1, keepdims=True)
    B_mean = B.mean(axis=1, keepdims=True)
    A_std = A.std(axis=1, keepdims=True) + 1e-8
    B_std = B.std(axis=1, keepdims=True) + 1e-8

    A_norm = (A - A_mean) / A_std
    B_norm = (B - B_mean) / B_std
    # correlation = (A_norm @ B_norm.T) / g  (population std above, so this is Pearson r)
    g = A.shape[1]
    return (A_norm @ B_norm.T) / float(g)


# ----------------------------------------------------------------------------- memory


# ``available_memory_bytes`` is worker_utils' reader: the smaller of MemAvailable and the room left under the
# cgroup memory limit (the limit minus the cgroup's working set, its page cache counted as reclaimable). This
# worker used to carry its own copy that took the cgroup LIMIT itself as the room, so a container already
# using most of its limit passed the preflight and was OOM-killed building the matrices it budgets.


def _gib(n_bytes) -> float:
    return float(n_bytes) / float(1 << 30)


def dense_peak_bytes(
    n_sc: int,
    n_is: int,
    n_genes: int,
    run_ccd: bool = False,
    run_clustering: bool = False,
    run_signaling: bool = False,
    run_signal_range: bool = False,
    build_from_raw: bool = False,
):
    """``(peak_bytes, phase, {phase: bytes})`` for the dense float64 state SpaOTsc holds.

    Counts what the upstream code allocates, phase by phase: the matrices that persist (sc_dmat,
    cost matrix, is_dmat, the sc expression table, the plan, the cell-cell distance) plus each step's
    own work arrays. The phases run one after another, so the peak is the largest of them.
    """
    n_sc, n_is, n_genes = int(n_sc), int(n_is), int(n_genes)
    sc_sc = n_sc * n_sc * 8
    sc_is = n_sc * n_is * 8
    is_is = n_is * n_is * 8
    expr_sc = n_sc * n_genes * 8
    phases = {}
    if build_from_raw:
        # raw, normalised, binarised and z-scored copies of both panels; corrcoef + exp; MCC + exp +
        # the product; the pairwise spot distances
        phases["building the matrices from raw input"] = 4 * (n_sc + n_is) * n_genes * 8 + 2 * sc_sc + 3 * sc_is + is_is
    persistent = sc_sc + sc_is + is_is + expr_sc
    phases["mapping (transport_plan)"] = persistent + 2 * sc_sc + MAPPING_SC_IS_COPIES * sc_is + 2 * is_is
    held = persistent + sc_is  # + the transport plan
    if run_ccd:
        phases["cell_cell_distance"] = held + sc_sc + 2 * sc_is
        held += sc_sc
    if run_clustering:
        phases["clustering"] = held + 2 * sc_sc
    if run_signaling:
        phases["signaling (spatial_signaling_ot)"] = held + 5 * sc_sc
    if run_signal_range:
        phases["signal range (nonspatial_correlation + infer_signal_range_ml)"] = (
            held + 3 * sc_sc + 2 * n_genes * n_genes * 8 + 3 * n_genes * n_sc * 8
        )
    phase = max(phases, key=phases.get)
    return phases[phase], phase, phases


def check_dense_budget(n_sc, n_is, n_genes, available=None, held=0, **steps):
    """Refuse, with the numbers, before allocating dense matrices the machine cannot hold.

    ``available`` defaults to ``available_memory_bytes()`` (worker_utils: MemAvailable, or the room left
    under the cgroup memory limit when that is smaller). ``held`` is the bytes of the matrices this run has
    already loaded and that ``dense_peak_bytes`` counts again as part of the peak; they are added back to
    the room, since the room was measured with them already allocated.
    """
    peak, phase, phases = dense_peak_bytes(n_sc, n_is, n_genes, **steps)
    if available is None:
        room = available_memory_bytes()
        available = None if room is None else room + int(held)
    if available is not None and peak > available:
        mapping = phases["mapping (transport_plan)"]
        loaded = f", plus the {_gib(held):.2f} GiB of input matrices already loaded" if held else ""
        raise MemoryError(
            f"SpaOTsc works on dense float64 matrices by construction: {n_sc} x {n_sc} over the single cells "
            f"(sc_dmat, the spatial cell-cell distance, the signaling scores), {n_sc} x {n_is} cells x spots "
            f"(cost matrix, transport plan and its Gromov-Wasserstein work arrays) and {n_is} x {n_is} over the "
            f"spots (is_dmat). For {n_sc} cells, {n_is} spots and {n_genes} shared genes this run needs about "
            f"{_gib(peak):.1f} GiB at its peak ({phase}), but about {_gib(available):.1f} GiB is available here "
            f"(MemAvailable, or the room left under the cgroup memory limit when that is smaller{loaded}). "
            f"The mapping alone needs about {_gib(mapping):.1f} GiB; "
            "run_cellcell_distance, run_clustering and run_signaling each add n_sc x n_sc work matrices and "
            "can be switched off. The data is never subsampled: run where that much memory is available."
        )
    return peak, available


# ----------------------------------------------------------------------------- building from raw


def _build_precomputed_from_raw(
    raw_sc_path: str,
    raw_is_path: str,
    raw_is_coord_path: str | None,
    precomp_dir: str,
    sel_genes_path: str | None = None,
    steps: dict | None = None,
    report: dict | None = None,
    use_raw_counts: bool = False,
    frame_args: dict | None = None,
) -> tuple[str, str, str, str, str | None]:
    """
    Build sc_expr, is_dmat, sc_dmat, cost_matrix from raw inputs (see ``PREPROCESSING``).

    Returns paths:
      sc_expr_path, is_dmat_path, sc_dmat_path, cost_matrix_path, selected_genes_path

    ``report`` (a dict, when given) is filled with what was read and written: ``written`` (every
    file this call wrote), ``coord_columns_used``, ``coord_alignment``, gene counts, rename counts,
    ``counts`` (per input: the matrix read and what it holds) and ``in_tissue`` (spots supplied and
    background spots left out).

    Background spots are left out of the spatial input before anything is built: ``obs['in_tissue'] == 0``
    in an AnnData raw_is_path, and spots a raw_is_coord_path with an ``in_tissue`` column flags 0.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances

    report = {} if report is None else report
    written = report.setdefault("written", [])
    os.makedirs(precomp_dir, exist_ok=True)

    log("Building precomputed matrices from raw data:")
    log(f"  raw_sc_path      = {raw_sc_path}")
    log(f"  raw_is_path      = {raw_is_path}")
    log(f"  raw_is_coord_path= {raw_is_coord_path}")
    if sel_genes_path:
        log(f"  selected_genes   = {sel_genes_path}")

    sc_read: dict = {}
    is_read: dict = {}
    X_sc, sc_obs, sc_var, _, sc_renamed = _read_raw(  # cells × genes
        raw_sc_path, role="raw_sc_path", use_raw_counts=use_raw_counts, report=sc_read
    )
    # A coordinates table, when given, is the coordinates: the h5ad's obsm is then not read at all.
    X_is, is_obs, is_var, coords_auto, is_renamed = _read_raw(  # spots × genes, background left out
        raw_is_path,
        role="raw_is_path",
        spatial=True,
        use_raw_counts=use_raw_counts,
        report=is_read,
        frame_args=None if raw_is_coord_path else (frame_args or {}),
    )
    report["n_genes_renamed_sc"] = sc_renamed
    report["n_genes_renamed_is"] = is_renamed
    report["counts"] = {"sc": sc_read["counts"], "spatial": is_read["counts"]}
    tissue = {
        "n_supplied": is_read["in_tissue"]["n_supplied"],
        "n_dropped": is_read["in_tissue"]["n_dropped"],
        "sources": ["obs['in_tissue'] of raw_is_path"] if is_read["in_tissue"]["n_dropped"] else [],
    }
    report["in_tissue"] = tissue

    # Coordinates first: without them there is no is_dmat, and nothing below is worth computing.
    if raw_is_coord_path:
        coords = _load_raw_coords(raw_is_coord_path, index=pd.Index([str(o) for o in is_obs]))
        _refuse_a_stacked_coordinate_table(raw_is_coord_path, coords, is_read)
    elif coords_auto is not None:
        coords = coords_auto
    else:
        raise ValueError(
            f"raw_is_path ({raw_is_path}) has no spatial coordinates, and SpaOTsc's spatial distance matrix "
            "(is_dmat) is built from them. Pass raw_is_coord_path (a csv/tsv with a spot-ID column and named "
            "coordinate columns) or an h5ad with obsm['spatial']. There is no substitute: an identity matrix "
            "would put every spot at distance 0 from every other."
        )
    report["coord_columns_used"] = coords.attrs.get("coord_columns_used")
    report["coord_alignment"] = coords.attrs.get("coord_alignment")
    report["frame"] = coords.attrs.get("frame")
    report["mode"] = coords.attrs.get("mode") or MODE_2D

    # A coordinates file with an in_tissue column (tissue_positions*.csv, the converter's metadata.csv of a
    # CELLxGENE export) marks the background too: those spots leave the expression, the coordinates and
    # every matrix built below, as obs['in_tissue'] == 0 spots already have.
    mask = coords.attrs.get("in_tissue_mask")
    if mask is not None and not mask.all():
        if not mask.any():
            raise ValueError(
                f"The in_tissue column of raw_is_coord_path ({raw_is_coord_path}) marks none of the {len(mask)} "
                "spots as in tissue; fix the column so in-tissue spots are 1, or remove it if every spot is tissue."
            )
        flag_name = coords.attrs.get("in_tissue_column") or "in_tissue"
        keep = np.flatnonzero(mask)
        X_is = X_is[keep]
        is_obs = [is_obs[i] for i in keep]
        coords = coords.iloc[keep]
        tissue["n_dropped"] += int((~mask).sum())
        tissue["sources"].append(f"the {flag_name} column of raw_is_coord_path")
    if tissue["n_dropped"]:
        log(
            f"Left out {tissue['n_dropped']} of {tissue['n_supplied']} spots with in_tissue == 0 "
            f"({' and '.join(tissue['sources'])}); {len(is_obs)} in-tissue spots are mapped."
        )

    # Determine gene set
    shared = {str(g) for g in sc_var}.intersection(str(g) for g in is_var)
    if sel_genes_path:
        # A user-supplied gene panel is one column of names as often as it is a delimited table,
        # and a single-column file sniffs back as the comma this always assumed. Read as a comma, a
        # tab-delimited panel makes every entry a whole line, the filter below keeps none of them,
        # and the run dies on "SpaOTsc needs a shared gene panel" -- naming the user's identifiers
        # for a mismatch that is entirely in the read. The panel this worker writes below is
        # one column, so it is unaffected.
        genes_list = (
            pd.read_csv(sel_genes_path, header=None, sep=sniff_tabular_sep(sel_genes_path))
            .iloc[:, 0]
            .astype(str)
            .tolist()
        )
        genes = [g for g in dict.fromkeys(genes_list) if g in shared]
        report["n_selected_genes_supplied"] = len(genes_list)
    else:
        genes = sorted(shared)
        sel_genes_path = os.path.join(precomp_dir, "selected_genes.txt")
        written.append(_atomic_to_csv(pd.Series(genes), sel_genes_path, index=False, header=False))
    report["n_genes_used"] = len(genes)

    if not genes:
        raise ValueError(
            id_mismatch_msg("genes", "scRNA", sc_var, "spatial", is_var) + " SpaOTsc needs a shared gene panel."
        )

    # Refuse before densifying anything the machine cannot hold.
    check_dense_budget(len(sc_obs), len(is_obs), len(genes), build_from_raw=True, **(steps or {}))

    df_sc = _dense_subset(X_sc, sc_obs, sc_var, genes)
    df_is = _dense_subset(X_is, is_obs, is_var, genes)
    del X_sc, X_is

    # Normalize + log transform (CPM 1e4 + log2), as described in implementation docs
    df_sc_norm = _normalize_log2(df_sc)
    df_is_norm = _normalize_log2(df_is)
    del df_sc, df_is

    sc_expr_path = os.path.join(precomp_dir, "dm_sc_normalized.txt")
    written.append(_atomic_to_csv(df_sc_norm, sc_expr_path, sep="\t"))

    # --- sc_dmat from PCA + Pearson correlation between cells ---
    log("Building sc_dmat from PCA + Pearson correlation between cells...")
    X = df_sc_norm.values
    n_components = min(40, X.shape[1], max(2, X.shape[0] - 1))
    pcs = PCA(n_components=n_components, svd_solver="auto").fit_transform(X)  # (n_sc × n_components)
    sc_dmat = np.exp(1.0 - np.corrcoef(pcs))  # (n_sc × n_sc), similarity -> distance-like
    sc_dmat_path = os.path.join(precomp_dir, "dm_scanpy_pca40_pcc.npy")
    written.append(_atomic_np_save(sc_dmat_path, sc_dmat))
    report["sc_dmat_n_pcs"] = int(n_components)
    del sc_dmat

    # --- cost_matrix from binarized normalized matrices ---
    log("Building cost_matrix from binarized normalized matrices...")
    A = _binarize_quantile(df_sc_norm).values  # n_sc × G
    B = _binarize_quantile(df_is_norm).values  # n_is × G
    report["n_genes_never_expressed_sc"] = int((A.sum(axis=0) == 0).sum())
    report["n_genes_never_expressed_spatial"] = int((B.sum(axis=0) == 0).sum())
    cost_matrix = np.exp(1.0 - _pearson_corr_rows(A, B))  # n_sc × n_is, SpaOTsc convention
    cost_matrix_path = os.path.join(precomp_dir, "dm_sc_is_mcc.npy")
    written.append(_atomic_np_save(cost_matrix_path, cost_matrix))
    del A, B, cost_matrix

    # --- is_dmat from coordinates ---
    log("Building is_dmat...")
    is_dmat = pairwise_distances(coords.loc[df_is_norm.index].values, metric="euclidean")
    is_dmat_path = os.path.join(precomp_dir, "dm_is.txt")
    written.append(_atomic_savetxt(is_dmat_path, is_dmat))

    log("Precomputed inputs built and saved.")
    return sc_expr_path, is_dmat_path, sc_dmat_path, cost_matrix_path, sel_genes_path


# ----------------------------------------------------------------------------- steps


def _validate_steps(args) -> None:
    """The prerequisites upstream SpaOTsc assumes and never checks, checked before any work.

    Each of these used to surface late -- an AttributeError on ``gamma_mapping`` or
    ``sc_dmat_spatial`` -- or, for signaling, not at all: the step was skipped or its exception
    swallowed while the payload still said it ran.
    """
    if args.run_ccd and not args.run_mapping:
        raise ValueError(
            "run_cellcell_distance=True needs run_mapping=True: SpaOTsc computes the spatial cell-cell "
            "distance from this run's transport plan."
        )
    if args.run_clustering and not args.run_ccd:
        raise ValueError(
            "run_clustering=True needs run_cellcell_distance=True: SpaOTsc's spatial subclustering runs on "
            "the spatial cell-cell distance."
        )
    if args.run_signaling:
        if not args.ligand or not args.receptor:
            raise ValueError(
                "run_signaling=True needs at least one ligand and one receptor "
                f"(got ligands={list(args.ligand)}, receptors={list(args.receptor)})."
            )
        if not args.run_ccd:
            raise ValueError(
                "run_signaling=True needs run_cellcell_distance=True: spatial_signaling_ot transports ligand "
                "mass over the spatial cell-cell distance that step computes."
            )


def _check_signaling_genes(args, genes: pd.Index) -> None:
    """Every signaling gene must be a column of the expression table SpaOTsc was given."""
    wanted = list(args.ligand) + list(args.receptor) + list(args.ds_up) + list(args.ds_down)
    missing = [g for g in dict.fromkeys(wanted) if g not in genes]
    if missing:
        raise ValueError(
            f"run_signaling: {missing} are not in the SpaOTsc expression table, which holds {len(genes)} genes "
            "(when built from raw input: the genes shared by the scRNA and spatial data, restricted to "
            "selected_genes_path if given). Ligands, receptors and ds_genes_up/ds_genes_down must be among them."
        )


@contextlib.contextmanager
def _louvain_seed_compat():
    """Let upstream ``clustering`` seed louvain on a louvain that has no ``set_rng_seed``.

    SpaOTsc passes its seed as ``find_partition(seed=...)`` only for louvain 0.7.0/0.7.1 and calls
    ``louvain.set_rng_seed(seed)`` for every other version -- which louvain 0.8 removed (it takes
    ``seed=`` instead), so on the 0.8.x this env ships every ``run_clustering`` died with an
    AttributeError. For the duration of the call, ``set_rng_seed`` records the seed and
    ``find_partition`` receives it as ``seed=``: the same upstream clustering, seeded with the value
    upstream chose. Nothing is patched when louvain still has ``set_rng_seed``. Yields a note for the
    payload (None when no shim was needed).
    """
    try:
        import louvain  # type: ignore
    except ImportError:
        yield None
        return
    if hasattr(louvain, "set_rng_seed") or str(getattr(louvain, "__version__", "")) in ("0.7.0", "0.7.1"):
        yield None
        return
    state = {"seed": None}
    original = louvain.find_partition

    def set_rng_seed(seed):
        state["seed"] = int(seed)

    def find_partition(*args, **kwargs):
        if kwargs.get("seed") is None and state["seed"] is not None:
            kwargs["seed"] = state["seed"]
        return original(*args, **kwargs)

    louvain.set_rng_seed = set_rng_seed
    louvain.find_partition = find_partition
    try:
        yield (
            f"louvain {getattr(louvain, '__version__', '?')} has no set_rng_seed; the seed SpaOTsc's clustering "
            "sets was passed to louvain.find_partition(seed=...) instead"
        )
    finally:
        del louvain.set_rng_seed
        louvain.find_partition = original


def _clustering_labels(spsc, cell_ids) -> tuple[pd.DataFrame, int]:
    """Per-cell labels from what upstream ``clustering()`` stores (it returns None).

    ``clustering_partition_org[i]`` lists the cells of expression cluster i; ``clustering_partition_inds``
    maps ``(i, j)`` to the cells of spatial subcluster j within it. Upstream keeps a subcluster only
    when it has more than ``min_n`` (3) members, so a cell of a smaller spatial group has an expression
    cluster and no subcluster: its ``cluster`` is left empty and counted, not invented.
    """
    parts = getattr(spsc, "clustering_partition_inds", None)
    org = getattr(spsc, "clustering_partition_org", None)
    if parts is None or org is None:
        raise RuntimeError("SpaOTsc.clustering() left no clustering_partition_inds / clustering_partition_org.")
    n = len(cell_ids)
    sub = np.full(n, "", dtype=object)
    for key in sorted(parts):
        idx = np.asarray(parts[key], dtype=int).reshape(-1)
        if (sub[idx] != "").any():
            raise RuntimeError(f"SpaOTsc subcluster {key} overlaps another subcluster.")
        sub[idx] = f"{key[0]}_{key[1]}"
    expr = np.full(n, -1, dtype=int)
    for i in range(len(org)):
        expr[np.asarray(list(org[i]), dtype=int)] = i
    if (expr < 0).any():
        raise RuntimeError(f"{int((expr < 0).sum())} cells are in no SpaOTsc expression cluster.")
    labels = pd.DataFrame({"cell": [str(c) for c in cell_ids], "cluster": sub, "expression_cluster": expr})
    return labels, int((sub == "").sum())


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-dir", default=None)

    # explicit precomputed files
    ap.add_argument("--sc-expr", dest="sc_expr", default=None)
    ap.add_argument("--is-dmat", dest="is_dmat", default=None)
    ap.add_argument("--sc-dmat", dest="sc_dmat", default=None)
    ap.add_argument("--cost-matrix", dest="cost_matrix", default=None)
    ap.add_argument("--selected-genes", dest="sel_genes", default=None)

    # raw data paths (for auto-building precomputed matrices)
    ap.add_argument("--raw-sc", dest="raw_sc", default=None)
    ap.add_argument("--raw-is", dest="raw_is", default=None)
    ap.add_argument("--raw-is-coord", dest="raw_is_coord", default=None)

    # the coordinate frame of an h5ad raw_is_path (worker_utils.spatial_frame)
    ap.add_argument("--coords-key", dest="coords_key", default="spatial")
    ap.add_argument("--dims", type=int, choices=(2, 3), default=2)
    ap.add_argument("--section-key", dest="section_key", default=None)

    # which analyses
    ap.add_argument("--run-mapping", action="store_true")
    ap.add_argument("--run-ccd", action="store_true")
    ap.add_argument("--run-clustering", action="store_true")
    ap.add_argument("--run-signaling", action="store_true")

    # signaling lists (repeatable)
    ap.add_argument("--ligand", action="append", default=[])
    ap.add_argument("--receptor", action="append", default=[])
    ap.add_argument("--ds-up", action="append", default=[])
    ap.add_argument("--ds-down", action="append", default=[])

    ap.add_argument("--use-landmark", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    # Read each raw h5ad input's counts from adata.raw (an input without one keeps X); see _counts_of_anndata.
    ap.add_argument("--use-raw-counts", dest="use_raw_counts", action="store_true")
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> WorkerOutput:
    """The whole pipeline. Raises on any failure; returns the payload for a run that did what was asked."""
    outdir = os.path.abspath(args.output_dir)
    os.makedirs(outdir, exist_ok=True)
    out = WorkerOutput("spaotsc", task="optimal_transport")
    record_method(out, METHOD_NAME)
    written: list[str] = []  # absolute paths of every file THIS run wrote

    _validate_steps(args)
    frame_args = _frame_args(args)
    if frame_args["dims"] == 3 and args.raw_is_coord:
        raise ValueError(
            "dims=3 reads the aligned 3D frame of an h5ad raw_is_path (obsm[coords_key], declared in "
            "uns['spatial_3d']['frames']), and raw_is_coord_path is a two-column coordinate table with no frame. "
            "Drop raw_is_coord_path and pass an h5ad raw_is_path with `coords_key=<aligned frame>, dims=3`, or run "
            "2D with the table."
        )
    signal_range_wanted = bool(args.run_signaling and (args.ds_up or args.ds_down))
    steps = {
        "run_ccd": bool(args.run_ccd),
        "run_clustering": bool(args.run_clustering),
        "run_signaling": bool(args.run_signaling),
        "run_signal_range": signal_range_wanted,
    }
    if not args.run_signaling:
        given = [
            name
            for name, value in (
                ("ligands", args.ligand),
                ("receptors", args.receptor),
                ("ds_genes_up", args.ds_up),
                ("ds_genes_down", args.ds_down),
            )
            if value
        ]
        record_ignored(out, given, "run_signaling=False, so no signaling step used them")

    import random

    from spaotsc import SpaOTsc  # type: ignore

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Precomputed defaults from standard tutorial layout
    std = {}
    if args.data_dir:
        dd = os.path.abspath(args.data_dir)
        std = {
            "sc_expr": os.path.join(dd, "dm_sc_normalized.txt"),
            "is_dmat_1": os.path.join(dd, "dm_is.txt"),
            "is_dmat_2": os.path.join(dd, "dm_pos_pos_geodesic_mgeom.npy"),
            "sc_dmat": os.path.join(dd, "dm_scanpy_pca40_pcc.npy"),
            "cost_matrix": os.path.join(dd, "dm_sc_is_mcc.npy"),
            "sel_genes": os.path.join(dd, "selected_genes.txt"),
        }

    # A path the caller named must exist: a typo used to be replaced in silence (by every shared gene
    # for the panel, by a rebuild from raw input for the matrices).
    for name, value in (
        ("selected_genes_path", args.sel_genes),
        ("sc_expr_path", args.sc_expr),
        ("is_dmat_path", args.is_dmat),
        ("sc_dmat_path", args.sc_dmat),
        ("cost_matrix_path", args.cost_matrix),
    ):
        if value and not os.path.exists(value):
            raise FileNotFoundError(f"{name} does not exist: {value}")

    sc_expr_path = args.sc_expr or std.get("sc_expr")
    is_dmat_path = args.is_dmat or _first_existing([std.get("is_dmat_1", ""), std.get("is_dmat_2", "")])
    sc_dmat_path = args.sc_dmat or std.get("sc_dmat")
    cost_matrix_path = args.cost_matrix or std.get("cost_matrix")
    sel_genes_path = args.sel_genes or _first_existing([std.get("sel_genes", "")])

    supplied = {
        "sc_expr_path": sc_expr_path,
        "is_dmat_path": is_dmat_path,
        "sc_dmat_path": sc_dmat_path,
        "cost_matrix_path": cost_matrix_path,
    }
    present = {name: bool(p) and os.path.exists(p) for name, p in supplied.items()}

    built_from_raw = False
    build_report: dict = {}
    if not all(present.values()) and args.raw_sc and args.raw_is:
        # All four are rebuilt together: they must describe the same cells, spots and genes.
        superseded = []
        for name, flag in (
            ("sc_expr_path", args.sc_expr),
            ("is_dmat_path", args.is_dmat),
            ("sc_dmat_path", args.sc_dmat),
            ("cost_matrix_path", args.cost_matrix),
        ):
            if flag and present[name]:
                superseded.append(name)
        if args.data_dir and any(present[n] for n in present if n not in superseded):
            superseded.append("data_dir")
        missing_names = [n for n, ok in present.items() if not ok]
        record_ignored(
            out,
            superseded,
            f"{', '.join(missing_names)} were missing, so all four matrices were rebuilt from raw_sc_path/"
            "raw_is_path (they must describe the same cells, spots and genes)",
        )
        (
            sc_expr_path,
            is_dmat_path,
            sc_dmat_path,
            cost_matrix_path,
            sel_genes_path,
        ) = _build_precomputed_from_raw(
            raw_sc_path=os.path.abspath(args.raw_sc),
            raw_is_path=os.path.abspath(args.raw_is),
            raw_is_coord_path=os.path.abspath(args.raw_is_coord) if args.raw_is_coord else None,
            precomp_dir=os.path.join(outdir, "precomputed"),
            sel_genes_path=sel_genes_path,
            steps=steps,
            report=build_report,
            use_raw_counts=bool(getattr(args, "use_raw_counts", False)),
            frame_args=frame_args,
        )
        written += build_report.get("written", [])
        built_from_raw = True
    else:
        unused = [
            name
            for name, value in (
                ("raw_sc_path", args.raw_sc),
                ("raw_is_path", args.raw_is),
                ("raw_is_coord_path", args.raw_is_coord),
            )
            if value
        ]
        frame_given = [
            name
            for name, value, default in (
                ("coords_key", frame_args["coords_key"], "spatial"),
                ("dims", frame_args["dims"], 2),
                ("section_key", frame_args["section_key"], None),
            )
            if value != default
        ]
        record_ignored(
            out,
            frame_given,
            "no matrix was built from raw input, so no coordinates were read; is_dmat was used as given",
        )
        if all(present.values()):
            record_ignored(out, unused, "the precomputed matrices were complete, so the raw inputs were not read")
            if args.sel_genes:
                record_ignored(
                    out,
                    ["selected_genes_path"],
                    "precomputed matrices were used as given; the panel applies only when building from raw input",
                )
        if getattr(args, "use_raw_counts", False):
            record_ignored(
                out,
                ["use_raw_counts"],
                "no matrix was built from raw input, so no adata.raw was read; it applies to raw_sc_path/raw_is_path",
            )

    # Final readiness check
    missing: list[str] = []
    _expect(
        "sc expression (cells×genes) table",
        bool(sc_expr_path) and os.path.exists(sc_expr_path),
        missing,
        "sc_expr",
    )
    _expect("spatial IS distance matrix", bool(is_dmat_path) and os.path.exists(is_dmat_path), missing, "is_dmat")
    _expect(
        "sc distance/dissimilarity matrix",
        bool(sc_dmat_path) and os.path.exists(sc_dmat_path),
        missing,
        "sc_dmat",
    )
    _expect(
        "sc↔spatial dissimilarity (cost matrix)",
        bool(cost_matrix_path) and os.path.exists(cost_matrix_path),
        missing,
        "cost_matrix",
    )
    if missing:
        raise ValueError(
            f"Data readiness check failed. Missing: {missing}. Pass raw_sc_path (the scRNA-seq reference) and "
            "raw_is_path (the spatial data) to build them, or data_dir / sc_expr_path, is_dmat_path, "
            "sc_dmat_path and cost_matrix_path for precomputed matrices."
        )

    meta_inputs = {
        "sc_expr": sc_expr_path,
        "is_dmat": is_dmat_path,
        "sc_dmat": sc_dmat_path,
        "cost_matrix": cost_matrix_path,
        "selected_genes": sel_genes_path,
        "raw_sc": args.raw_sc,
        "raw_is": args.raw_is,
        "raw_is_coord": args.raw_is_coord,
    }

    # Load precomputed data
    log("Loading precomputed matrices...")
    df_sc = _load_table(sc_expr_path)  # cells×genes
    is_dmat = _load_matrix(is_dmat_path)  # (#is × #is)
    sc_dmat = _load_matrix(sc_dmat_path)  # (#sc × #sc)
    cost_mat = _load_matrix(cost_matrix_path)  # (#sc × #is)

    meta_shapes = {
        "df_sc": [df_sc.shape[0], df_sc.shape[1]],
        "is_dmat": list(is_dmat.shape),
        "sc_dmat": list(sc_dmat.shape),
        "cost_matrix": list(cost_mat.shape),
    }

    # Basic shape sanity
    n_sc = df_sc.shape[0]
    n_is = is_dmat.shape[0]
    n_genes = df_sc.shape[1]
    if is_dmat.shape[0] != is_dmat.shape[1]:
        raise ValueError(f"is_dmat must be square; got {is_dmat.shape}")
    if sc_dmat.shape != (n_sc, n_sc):
        raise ValueError(f"sc_dmat must be (n_sc, n_sc) = ({n_sc},{n_sc}); got {sc_dmat.shape}")
    if cost_mat.shape != (n_sc, n_is):
        raise ValueError(f"cost_matrix shape must be (n_sc, n_is) = ({n_sc},{n_is}); got {cost_mat.shape}")

    # Everything a requested step needs, checked before the first expensive call.
    if args.run_signaling:
        _check_signaling_genes(args, df_sc.columns)
    if args.run_ccd and args.use_landmark and n_is < N_LANDMARK:
        raise ValueError(
            f"use_landmark=True makes SpaOTsc's cell_cell_distance pick {N_LANDMARK} landmark spots, and the "
            f"spatial input has {n_is}. Pass use_landmark=False (exact distances over all {n_is} spots)."
        )
    if args.run_clustering and n_sc <= CLUSTERING_KNN:
        raise ValueError(
            f"run_clustering needs more than {CLUSTERING_KNN} single cells: SpaOTsc's clustering builds a "
            f"{CLUSTERING_KNN}-nearest-neighbour graph on the spatial cell-cell distance, and there are {n_sc}."
        )
    held = int(sc_dmat.nbytes + cost_mat.nbytes + is_dmat.nbytes + df_sc.values.nbytes)
    peak_bytes, _ = check_dense_budget(n_sc, n_is, n_genes, held=held, **steps)

    # Initialize SpaOTsc object (minimal example from README)
    log("Initializing SpaOTsc.spatial_sc(...)")
    spsc = SpaOTsc.spatial_sc(sc_data=df_sc, is_dmat=is_dmat, sc_dmat=sc_dmat)

    os.makedirs(os.path.join(outdir, "meta"), exist_ok=True)
    written.append(
        _atomic_json(os.path.join(outdir, "meta", "meta_inputs.json"), {"inputs": meta_inputs, "shapes": meta_shapes})
    )

    summary_kwargs = {"shapes": meta_shapes}
    ran = {"mapping": False, "cell_cell_distance": False, "clustering": False, "signaling": False}
    notes: list[str] = []
    clustering_info = None
    signal_range = "not requested (run_signaling=False)"

    # Mapping / transport plan
    if args.run_mapping:
        log("Computing transport plan (mapping)...")
        tp = spsc.transport_plan(cost_mat)
        if tp is None:
            tp = getattr(spsc, "gamma_mapping", None)
        if tp is None:
            raise RuntimeError("SpaOTsc.transport_plan returned no plan and set no gamma_mapping.")
        tp = np.asarray(tp)
        written += _save_array(os.path.join(outdir, "mapping", "transport_plan.csv"), tp)
        summary_kwargs["transport_plan_shape"] = list(tp.shape)
        summary_kwargs["transport_plan_sparsity"] = round(float((tp == 0).sum()) / tp.size, 4)
        entropy = _mean_row_entropy(tp)
        summary_kwargs["transport_plan_mean_row_entropy"] = round(entropy, 4)
        if entropy > UNIFORM_PLAN_ENTROPY:
            out.add_warning(
                f"The transport plan is near-uniform (mean normalised row entropy {entropy:.3f}; 1.0 = every cell "
                "spread evenly over every spot), so it does not localise cells. SpaOTsc's transport_plan runs at its "
                f"defaults (epsilon=1.0), and the cost matrix here spans only {float(np.min(cost_mat)):.2f}-"
                f"{float(np.max(cost_mat)):.2f}."
            )
        ran["mapping"] = True

    # Spatial cell–cell distance
    if args.run_ccd:
        n_pairs = n_sc * (n_sc - 1) // 2
        width = min(N_LANDMARK, n_is) if args.use_landmark else n_is
        log(
            f"Computing spatial cell-cell distance: upstream solves {n_pairs} Sinkhorn problems (one per cell "
            f"pair) on a {width} x {width} cost; this is the slow step."
        )
        ccd = spsc.cell_cell_distance(use_landmark=bool(args.use_landmark))
        if ccd is None:
            ccd = getattr(spsc, "sc_dmat_spatial", None)
        if ccd is None:
            raise RuntimeError("SpaOTsc.cell_cell_distance returned nothing and set no sc_dmat_spatial.")
        written += _save_array(os.path.join(outdir, "ccd", "cell_cell_distance.csv"), np.asarray(ccd))
        ran["cell_cell_distance"] = True

    # Clustering
    if args.run_clustering:
        n_pcs = int(min(CLUSTERING_PCA_COMPONENTS, n_genes, n_sc))
        log(f"Running SpaOTsc clustering (pca_n_components={n_pcs})...")
        with _louvain_seed_compat() as louvain_note:
            spsc.clustering(pca_n_components=n_pcs)
        labels, n_unassigned = _clustering_labels(spsc, df_sc.index)
        os.makedirs(os.path.join(outdir, "clustering"), exist_ok=True)
        written.append(_atomic_to_csv(labels, os.path.join(outdir, "clustering", "labels.csv"), index=False))
        n_sub = len(set(labels["cluster"]) - {""})
        clustering_info = {
            "pca_n_components": n_pcs,
            "n_expression_clusters": int(labels["expression_cluster"].nunique()),
            "n_spatial_subclusters": n_sub,
            "n_cells_without_spatial_subcluster": n_unassigned,
        }
        if louvain_note:
            clustering_info["louvain_seed_compat"] = louvain_note
        summary_kwargs["n_spatial_subclusters"] = n_sub
        ran["clustering"] = True
        if n_unassigned:
            notes.append(
                f"{n_unassigned} of {n_sc} cells belong to a spatial group of 3 or fewer cells, which SpaOTsc does "
                "not keep as a subcluster; their 'cluster' is empty and 'expression_cluster' still names them."
            )

    # Signaling
    if args.run_signaling:
        ligands, receptors = list(args.ligand), list(args.receptor)
        ds_up, ds_down = list(args.ds_up), list(args.ds_down)
        log(f"Running signaling: ligands={ligands}, receptors={receptors}, DS_up={ds_up}, DS_down={ds_down}")
        os.makedirs(os.path.join(outdir, "signaling"), exist_ok=True)
        # Upstream defaults these to [] and calls len() on them; None was a TypeError.
        sig = spsc.spatial_signaling_ot(ligands, receptors, DSgenes_up=ds_up, DSgenes_down=ds_down)
        if sig is None:
            raise RuntimeError("SpaOTsc.spatial_signaling_ot returned no scores.")
        if isinstance(sig, np.ndarray):
            if not np.isfinite(sig).all():
                raise ValueError(
                    f"spatial_signaling_ot produced non-finite scores for ligands={ligands}, receptors={receptors}: "
                    "the ligand or receptor(+downstream) weight is zero in every cell, so there is no signal to "
                    "transport. Pick genes expressed in this reference."
                )
            written += _save_array(os.path.join(outdir, "signaling", "signaling_scores.csv"), sig)
        else:
            written.append(
                _atomic_to_csv(
                    pd.DataFrame(sig), os.path.join(outdir, "signaling", "signaling_scores_table.csv"), index=False
                )
            )
        ran["signaling"] = True

        if ds_up or ds_down:
            log("Inferring signal range (nonspatial_correlation + infer_signal_range_ml)...")
            spsc.nonspatial_correlation()  # sets gene_cor_scc, which infer_signal_range_ml reads
            rng, _ = spsc.infer_signal_range_ml(
                ligands, receptors, ds_up + ds_down, effect_ranges=np.asarray(EFFECT_RANGES, dtype=float)
            )
            if rng is None:
                raise RuntimeError("SpaOTsc.infer_signal_range_ml returned no result.")
            if isinstance(rng, np.ndarray):
                written += _save_array(os.path.join(outdir, "signaling", "inferred_range.csv"), rng)
                if not np.isfinite(rng).all():
                    out.add_warning(
                        "infer_signal_range_ml returned NaN strengths: the downstream genes' effect strength did "
                        f"not vary across the ranges {list(EFFECT_RANGES)} (is_dmat units), so it cannot be scaled."
                    )
            else:
                written.append(
                    _atomic_to_csv(
                        pd.DataFrame(rng), os.path.join(outdir, "signaling", "inferred_range_table.csv"), index=False
                    )
                )
            signal_range = f"ran over effect ranges {list(EFFECT_RANGES)} (is_dmat units)"
        else:
            signal_range = "not run: requires ds_genes_up/down"
            notes.append(
                "The signal-range step (infer_signal_range_ml) was not run: it scores downstream genes, and "
                "ds_genes_up/ds_genes_down were empty."
            )

    rel = [os.path.relpath(p, outdir) for p in written]
    rel_summary = os.path.join("meta", "run_summary.json")
    rel.append(rel_summary)

    out.set_data(n_sc=n_sc, n_is=n_is, n_spots=n_is)
    out.add_output_files({f: os.path.join(outdir, f) for f in rel})
    out.add_params(
        {
            "output_dir": outdir,
            "inputs": meta_inputs,
            "seed": args.seed,
            "ran": ran,
            "use_landmark": bool(args.use_landmark),
            "built_from_raw": built_from_raw,
            "signal_range": signal_range,
            "dense_peak_bytes_estimated": int(peak_bytes),
        }
    )
    if args.run_signaling:
        out.add_params(
            {
                "ligands": list(args.ligand),
                "receptors": list(args.receptor),
                "ds_genes_up": list(args.ds_up),
                "ds_genes_down": list(args.ds_down),
            }
        )
    if clustering_info is not None:
        out.add_params({"clustering": clustering_info})
    if built_from_raw:
        out.add_params(
            {
                "preprocessing": PREPROCESSING,
                "binarize": BINARIZE_RULE,
                "coord_columns_used": build_report.get("coord_columns_used"),
                "coord_alignment": build_report.get("coord_alignment"),
                "is_dmat_units": _is_dmat_units(build_report.get("frame")),
                "n_genes_used": build_report.get("n_genes_used"),
                "sc_dmat_n_pcs": build_report.get("sc_dmat_n_pcs"),
                "n_genes_renamed_sc": build_report.get("n_genes_renamed_sc", 0),
                "n_genes_renamed_is": build_report.get("n_genes_renamed_is", 0),
                "n_genes_never_binarised_expressed": {
                    "sc": build_report.get("n_genes_never_expressed_sc"),
                    "spatial": build_report.get("n_genes_never_expressed_spatial"),
                },
            }
        )
        if "n_selected_genes_supplied" in build_report:
            out.add_params({"n_selected_genes_supplied": build_report["n_selected_genes_supplied"]})
            dropped = build_report["n_selected_genes_supplied"] - build_report["n_genes_used"]
            if dropped:
                out.add_warning(
                    f"{dropped} of {build_report['n_selected_genes_supplied']} genes in selected_genes_path are not "
                    "in both inputs (or repeat) and were not used."
                )
        # Which matrix each input was read from and what it holds (worker_utils.choose_counts_matrix's rule).
        counts = build_report.get("counts") or {}
        names = {"sc": "raw_sc_path", "spatial": "raw_is_path"}
        record_expression_source(
            out,
            {
                "expression_source": {k: v["expression_source"] for k, v in counts.items()},
                "x_matrix_kind": {k: v["x_matrix_kind"] for k, v in counts.items()},
                "warning": " ".join(f"{names[k]}: {v['warning']}" for k, v in counts.items() if v.get("warning"))
                or None,
            },
        )
        out.add_params({"use_raw_counts": bool(getattr(args, "use_raw_counts", False))})
        out.add_warnings([v["note"] for v in counts.values() if v.get("note")])
        tissue = build_report.get("in_tissue") or {}
        if tissue.get("n_dropped"):
            record_in_tissue(out, tissue["n_supplied"], tissue["n_dropped"])
            out.set_data(
                n_spots_supplied=int(tissue["n_supplied"]),
                n_spots_off_tissue_dropped=int(tissue["n_dropped"]),
            )
            notes.append(
                f"{tissue['n_dropped']} of the {tissue['n_supplied']} spots supplied have in_tissue == 0 "
                f"({' and '.join(tissue['sources'])}; background outside the tissue) and were left out before the "
                f"matrices were built; the {n_is} in-tissue spots were mapped."
            )
    if built_from_raw:
        out.add_params(_frame_params(build_report.get("frame"), build_report.get("mode") or MODE_2D))
        if build_report.get("mode") == MODE_3D:
            frame = build_report["frame"]
            notes.append(
                f"The spot-to-spot distances (is_dmat) are 3D, in micrometres in obsm['{frame.key}'] "
                f"(z from {frame.z_source}); a mapping or signal between sections is inferred cross-section "
                "communication, not a measured one."
            )
    out.set_summary(**summary_kwargs)

    analyses_run = [k for k, v in ran.items() if v]
    analysis = (
        f"SpaOTsc optimal transport completed for {n_sc} single cells mapped to {n_is} spatial locations. "
        f"Analyses performed: {', '.join(analyses_run) if analyses_run else 'none'}. "
        f"Produced {len(rel)} output files."
    )
    if notes:
        analysis += " " + " ".join(notes)
    out.set_analysis(analysis)

    # Also save summary to disk (listed above, written here, before the payload is emitted)
    _atomic_json(os.path.join(outdir, rel_summary), out.to_dict())
    return out


def main(argv=None) -> None:
    args = _parse_args(argv)
    try:
        # Upstream prints progress (a line per cell in cell_cell_distance, whole matrices in
        # nonspatial_correlation) to stdout, which carries this worker's JSON payload.
        with contextlib.redirect_stdout(sys.stderr):
            out = run(args)
    except Exception as e:
        log("EXCEPTION during SpaOTsc worker run:")
        log(traceback.format_exc())
        WorkerOutput.emit_error("spaotsc", str(e), task="optimal_transport", exc=e)
        sys.exit(1)
    out.emit()


if __name__ == "__main__":
    main()
