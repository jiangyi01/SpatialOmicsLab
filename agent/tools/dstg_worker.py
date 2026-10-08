#!/usr/bin/env python
"""
dstg_worker.py

Worker script for DSTG-style spatial deconvolution with a TensorFlow 1.x
graph convolutional network.

WHAT RUNS: this worker's own two-layer GCN, trained on a kNN graph over the
spatial spots and the reference cells, propagating the reference cell-type
labels onto the spots. It does NOT run the upstream DSTG pipeline (its R
pseudo-spot synthesis and ``DSTG.models.DSTG``): ``DSTG.utils.load_data``
needs files that pipeline writes and nothing here produces, so the old
``try: load_data(...) except: <own GCN>`` always took the second branch
while the payload said "DSTG". The payload now names the method that ran.

- Called by the FastMCP wrapper (dstg_mcp_server.py).
- Must be executed inside the DSTG conda env: /opt/conda/envs/dstg_env
  (Python 3.7, TensorFlow 1.15.5)
- All logs go to stderr; stdout only prints a single JSON line at the end.
- Reference cells with no label (NaN/empty, or a categorical code of -1) are refused unless
  ``--drop-unlabeled`` is given; a missing label is never fitted as a cell type.
- Spots whose ``obs['in_tissue']`` is 0 (background glass in CELLxGENE Visium exports) are left out
  of the graph and the outputs, and the payload says how many (``params.in_tissue_filter``).
- Every file written here goes to ``<name>.partial`` first and is renamed into place.

Example (manual test):

  (dstg_env) python /workspace/epic-fermat/agent/tools/dstg_worker.py \
      --spatial-data /workspace/work/spatial_input/V1_Human_Lymph_Node.h5ad \
      --sc-data /workspace/spatial_demo_data/sc_annotation_ref/sc.h5ad \
      --output-dir /workspace/work/dstg_V1_LN \
      --n-clusters 7 \
      --learning-rate 0.01 \
      --epochs 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any

# Ensure worker_utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add DSTG repo to path (need both: parent for package import, subdir for bare imports within DSTG)
# <TOOL>_SRC seam (as spatialscope_worker.py does): this checkout exists only where it was
# installed, and a sys.path entry that does not exist fails silently -- the run dies later in an
# ImportError naming an upstream module, with no way to redirect it. The literal stays the default.
_DSTG_SRC = os.environ.get("DSTG_SRC") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "third_party", "dstg_repo"
)
sys.path.insert(0, _DSTG_SRC)
sys.path.insert(0, os.path.join(_DSTG_SRC, "DSTG"))

from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    build_deconv_analysis,
    keep_in_tissue,
    preflight_check,
    read_indexed_table,
    read_obsm_matrix,
    record_in_tissue,
    record_method,
    spatial_coords,
)
from worker_utils import drop_unlabeled as split_unlabeled  # the parameter of the same name shadows it


def log(msg: str) -> None:
    """Print log messages to stderr with a prefix."""
    sys.stderr.write(f"[dstg-worker] {msg}\n")
    sys.stderr.flush()


#: What this worker runs. Not "DSTG": the upstream pseudo-spot pipeline is not executed here.
METHOD_NAME = "DSTG-style GCN label propagation (SpatialOmicsLab reimplementation; upstream DSTG not run)"


def _dense_bytes(n_total, hidden_dim, n_classes):
    """Bytes of the dense float32 tensors the GCN materialises (activations, labels, logits)."""
    per_node = hidden_dim + 3 * n_classes + 1  # H1, labels, logits, softmax, mask
    return int(n_total) * per_node * 4 * 2  # x2: forward values + gradients


def _memory_budget_bytes():
    """Memory this run can still allocate, or None when the platform cannot say.

    ``worker_utils.available_memory_bytes``: ``MemAvailable``, and the room under a cgroup memory
    limit with the cgroup's page cache counted as reclaimable. ``MemAvailable`` alone ignored a
    container's limit, so a run that could not fit there was not refused but killed.
    """
    return available_memory_bytes()


def _write_csv_atomic(df, path, **kwargs):
    """Write ``df`` to ``<path>.partial`` and rename it over ``path`` once it is complete."""
    tmp = str(path) + ".partial"
    df.to_csv(tmp, **kwargs)
    os.replace(tmp, str(path))


class _ObsView:
    """What ``worker_utils.keep_in_tissue`` reads of an AnnData, over one obs column read with h5py.

    dstg_env has no anndata, so the shared in-tissue rule is applied to this stand-in: ``obs`` holds
    the ``in_tissue`` column, ``n_obs`` its length, and selecting rows keeps ``rows`` -- the positions
    of the kept spots in the file -- in step.
    """

    def __init__(self, obs, rows):
        self.obs = obs
        self.rows = rows

    @classmethod
    def of(cls, in_tissue_values):
        import numpy as np
        import pandas as pd

        values = list(in_tissue_values)
        return cls(pd.DataFrame({"in_tissue": pd.Series(values, dtype=object)}), np.arange(len(values)))

    @property
    def n_obs(self):
        return len(self.rows)

    def __getitem__(self, keep):
        return _ObsView(self.obs[keep].reset_index(drop=True), self.rows[keep])

    def copy(self):
        return self


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DSTG worker: spatial deconvolution via TF1 graph neural network.")
    parser.add_argument(
        "--spatial-data",
        type=str,
        required=True,
        help="Path to spatial data: .h5ad file or directory with DSTG-formatted files.",
    )
    parser.add_argument(
        "--sc-data",
        type=str,
        required=True,
        help="Path to single-cell data: .h5ad file or directory with DSTG-formatted files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save DSTG deconvolution outputs.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=7,
        help="k of the kNN graph over spots + reference cells that the GCN propagates on (the name is historical).",
    )
    parser.add_argument(
        "--cell-type-key",
        type=str,
        default="",
        help="obs column of the single-cell reference holding the cell-type labels. Empty: try the conventional "
        "names (cell_type, CellType, celltype, cell_type_key, annotation) and fail if none is present.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for DSTG GCN training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        default=False,
        help="Leave out reference cells whose label is missing (NaN/empty) instead of refusing the reference; "
        "the count is reported.",
    )
    return parser.parse_args(argv)


#: Above this many matrix cells the genes x obs count CSV is not written: a 507,684 x 18,085 Visium
#: HD slide is 9e9 cells of text. The matrix stays sparse in memory either way; the CSV is a
#: convenience copy in DSTG's own input format, and skipping it is recorded in the payload.
COUNT_CSV_MAX_CELLS = 200_000_000

LABEL_KEY_CANDIDATES = ("cell_type", "CellType", "celltype", "cell_type_key", "annotation")


def _decode_labels(obs_group, key):
    """Labels of one obs column, whatever anndata layout wrote it.

    Three layouts are live in the library: the current categorical Group (``codes`` + ``categories``),
    a plain string Dataset, and the legacy categorical -- an integer Dataset of codes beside
    ``obs/__categories/<key>`` holding the names (the Tonsil reference's ``CellType``). The legacy
    layout used to be read as a string Dataset, so the "cell types" came out as '0'..'43'.

    A categorical code of -1 is anndata's missing value and comes back as None -- not the string
    'NA', which the GCN then fitted and reported as a cell type.
    """
    import h5py

    label_data = obs_group[key]

    def _s(x):
        return x.decode("utf-8") if isinstance(x, bytes) else str(x)

    if isinstance(label_data, h5py.Group) and "categories" in label_data and "codes" in label_data:
        categories = [_s(x) for x in label_data["categories"][:]]
        codes = label_data["codes"][:]
        return [categories[c] if c >= 0 else None for c in codes]
    values = label_data[:]
    if values.dtype.kind in "iu":
        legacy = obs_group.get("__categories")
        if legacy is not None and key in legacy:
            categories = [_s(x) for x in legacy[key][:]]
            return [categories[c] if c >= 0 else None for c in values]
        raise ValueError(
            f"obs[{key!r}] holds integer codes but the file carries no categories for it; refusing to treat "
            "the codes as cell-type names"
        )
    return [_s(x) for x in values]


def _read_in_tissue(obs_group):
    """``obs['in_tissue']`` as stored (0/1 integers, booleans, or a categorical / string column), or None.

    anndata writes it as a plain integer Dataset (the CELLxGENE Visium exports), a nullable Group
    (``values`` + ``mask``; a masked entry is None), or a categorical; each is handed back as values
    ``worker_utils.keep_in_tissue`` can read.
    """
    import h5py

    if "in_tissue" not in obs_group:
        return None
    data = obs_group["in_tissue"]
    if isinstance(data, h5py.Group) and "values" in data:
        values = list(data["values"][:])
        if "mask" in data:
            mask = data["mask"][:]
            values = [None if m else v for v, m in zip(values, mask)]
        return values
    if isinstance(data, h5py.Dataset) and data.dtype.kind in "iub":
        legacy = obs_group.get("__categories")
        if data.dtype.kind != "b" and legacy is not None and "in_tissue" in legacy:
            return _decode_labels(obs_group, "in_tissue")
        return list(data[:])
    return _decode_labels(obs_group, "in_tissue")


def _h5ad_to_dstg_format(h5ad_path, output_dir, prefix, cell_type_key=""):
    """
    Read an AnnData .h5ad with h5py into what the GCN needs, writing DSTG-format side files.

    Uses h5py directly to avoid dependency on anndata/scanpy (not available
    in the dstg_env which runs Python 3.7 + TF 1.15).

    Returns a dict with the sparse ``X`` (obs x var, CSR), ``obs_names``, ``var_names``,
    ``n_obs``/``n_vars``, and the side files it wrote:
      - {prefix}_count.csv (genes x obs) unless the matrix is above COUNT_CSV_MAX_CELLS
      - {prefix}_coord.csv (spot coordinates, if obsm['spatial'] exists)
      - {prefix}_labels.csv + ``cell_types`` (reference labels, from ``cell_type_key`` or the
        conventional names; absent when the file has none). ``label_values`` keeps a missing
        label as None; ``cell_types`` lists only the labels that are present.
    and ``in_tissue`` (the obs column's values, or None when the file has no such column).
    Every side file is written to ``<name>.partial`` and renamed into place.
    """
    import h5py
    import pandas as pd
    from scipy.sparse import csc_matrix, csr_matrix

    log(f"Reading {h5ad_path} via h5py...")
    # `with`, not a bare open/close: `f.close()` used to sit 90 lines below, so any raise in
    # between -- the missing-`X` RuntimeError, a malformed sparse `X`, a full disk under
    # `to_csv` -- left the handle and HDF5's lock on the input alive for as long as the caller
    # held the traceback (`main` at :578 holds it across all of its error handling).
    with h5py.File(h5ad_path, "r") as f:

        def _read_index(group):
            """Read the index of an obs/var dataframe group following anndata encoding."""
            # anndata stores the index column name in attrs['_index']
            idx_col = None
            if "_index" in group.attrs:
                idx_col = group.attrs["_index"]
                if isinstance(idx_col, bytes):
                    idx_col = idx_col.decode("utf-8")
            # Try the named index column first, then fallback to _index/index datasets
            for key in ([idx_col] if idx_col else []) + ["_index", "index"]:
                if key and key in group:
                    ds = group[key]
                    if isinstance(ds, h5py.Dataset):
                        return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in ds[:]]
            return None

        # Read observation names (barcodes)
        obs_names = _read_index(f["obs"]) if "obs" in f else None

        # Read variable names (genes)
        var_names = _read_index(f["var"]) if "var" in f else None

        n_obs = len(obs_names) if obs_names else 0
        n_vars = len(var_names) if var_names else 0
        log(f"h5ad: {n_obs} obs x {n_vars} vars")

        # Read expression matrix X, kept sparse (CSR, obs x var). The old code called .toarray() here
        # and then wrote the dense transpose to CSV, which is what made Visium HD infeasible.
        if "X" in f:
            x_group = f["X"]
            if isinstance(x_group, h5py.Group):
                data = x_group["data"][:]
                indices = x_group["indices"][:]
                indptr = x_group["indptr"][:]
                shape = (n_obs, n_vars)
                encoding = x_group.attrs.get("encoding-type", "csr_matrix")
                encoding = encoding.decode() if isinstance(encoding, bytes) else str(encoding)
                if encoding.startswith("csc"):
                    X = csc_matrix((data, indices, indptr), shape=shape).tocsr()
                else:
                    X = csr_matrix((data, indices, indptr), shape=shape)
            else:
                X = csr_matrix(x_group[:])
        else:
            raise RuntimeError(f"No 'X' dataset found in {h5ad_path}")

        files = {"X": X, "obs_names": obs_names, "var_names": var_names}

        # Expression matrix (genes x cells/spots) in DSTG's own CSV layout, when it is small enough to
        # be worth writing as text.
        n_cells = int(n_obs) * int(n_vars)
        if n_cells <= COUNT_CSV_MAX_CELLS:
            count_df = pd.DataFrame(X.T.toarray(), index=var_names, columns=obs_names)
            count_path = os.path.join(output_dir, f"{prefix}_count.csv")
            log(f"Writing {prefix}_count.csv ({count_df.shape})...")
            _write_csv_atomic(count_df, count_path)
            files["count"] = count_path
            del count_df
        else:
            files["count_csv_skipped_reason"] = (
                f"{n_obs} x {n_vars} = {n_cells} matrix cells exceeds COUNT_CSV_MAX_CELLS={COUNT_CSV_MAX_CELLS}; "
                "the matrix is used sparse in memory instead"
            )
            log(f"Skipping {prefix}_count.csv: {files['count_csv_skipped_reason']}")

        # Spatial coordinates (for spatial data)
        if "obsm" in f and "spatial" in f["obsm"]:
            coords, _ = spatial_coords(read_obsm_matrix(f, "spatial"), "spatial", want=2, tool="DSTG")
            coord_df = pd.DataFrame(
                coords,
                index=obs_names,
                columns=["x", "y"],
            )
            coord_path = os.path.join(output_dir, f"{prefix}_coord.csv")
            _write_csv_atomic(coord_df, coord_path)
            files["coord"] = coord_path

        # Cell type labels (for single-cell data): the named column, else the conventional names.
        if "obs" in f:
            obs_group = f["obs"]
            obs_columns = [k for k in obs_group.keys() if k not in ("_index", "__categories")]
            if cell_type_key:
                if cell_type_key not in obs_group:
                    raise KeyError(
                        f"cell_type_key={cell_type_key!r} is not an obs column of {h5ad_path}. Available: {obs_columns}"
                    )
                label_key = cell_type_key
            else:
                label_key = next((k for k in LABEL_KEY_CANDIDATES if k in obs_group), None)
            if label_key is not None:
                labels = _decode_labels(obs_group, label_key)
                labels_df = pd.DataFrame({"label": labels}, index=obs_names)
                labels_path = os.path.join(output_dir, f"{prefix}_labels.csv")
                _write_csv_atomic(labels_df, labels_path)
                files["labels"] = labels_path
                files["label_key"] = label_key
                files["label_values"] = labels
                # A missing label is not a class: the shared rule decides what counts as missing.
                present, _ = split_unlabeled(labels, True)
                files["cell_types"] = sorted({str(v) for v, ok in zip(labels, present) if ok})
            files["obs_columns"] = obs_columns
            files["in_tissue"] = _read_in_tissue(obs_group)

    files["n_obs"] = n_obs
    files["n_vars"] = n_vars

    return files


def _labels_for_count_columns(labels_df, count_cells, labels_path):
    """(the label of each count column, in column order; how they were paired) for a pre-formatted dir.

    ``sc_labels.csv`` is indexed by cell. When its index names every column of ``sc_count.csv`` the
    labels are taken by name, so a file sorted differently cannot hand one cell another's type;
    otherwise, if it has exactly one label per column, they are paired by position as before (and
    the payload says so). Anything else is refused with the counts instead of failing inside the GCN.
    Missing labels stay missing (None/NaN) for the caller's drop_unlabeled rule.
    """
    column = labels_df.iloc[:, 0]
    index = [str(i) for i in labels_df.index]
    cells = [str(c) for c in count_cells]
    if index == cells:
        return column.tolist(), "name"
    if len(set(index)) == len(index) and set(cells) <= set(index):
        by_name = dict(zip(index, column.tolist()))
        return [by_name[c] for c in cells], "name"
    if len(index) == len(cells):
        return column.tolist(), "position"
    shared = len(set(cells) & set(index))
    raise ValueError(
        f"{labels_path} has {len(index)} labels but the single-cell counts have {len(cells)} cells "
        f"({shared} of them named in the labels' index); give one label per count column"
    )


def run_dstg_deconvolution(
    spatial_data: str,
    sc_data: str,
    output_dir: str,
    n_clusters: int,
    learning_rate: float,
    epochs: int,
    cell_type_key: str = "",
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    DSTG-style label propagation: a two-layer GCN over a kNN graph of spots + reference cells,
    trained on the reference labels, read out on the spots as per-type probabilities.

    ``drop_unlabeled`` leaves out reference cells whose label is missing (NaN/empty/code -1)
    instead of refusing the reference. Background spots (``obs['in_tissue'] == 0``) of an h5ad
    input are left out of the graph and the outputs; both counts are reported.
    """
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix, vstack

    os.makedirs(output_dir, exist_ok=True)

    log(f"spatial_data   = {spatial_data}")
    log(f"sc_data        = {sc_data}")
    log(f"output_dir     = {output_dir}")
    log(f"n_clusters     = {n_clusters}")
    log(f"learning_rate  = {learning_rate}")
    log(f"epochs         = {epochs}")
    log(f"drop_unlabeled = {drop_unlabeled}")

    # Preflight checks
    preflight_check(
        inputs={"spatial_data": spatial_data, "sc_data": sc_data},
        output_dir=output_dir,
    )

    # Convert h5ad to DSTG format if needed
    spatial_is_h5ad = spatial_data.endswith(".h5ad")
    sc_is_h5ad = sc_data.endswith(".h5ad")

    data_dir = os.path.join(output_dir, "dstg_input")
    os.makedirs(data_dir, exist_ok=True)

    def _table_to_sparse(df):
        # DSTG-format CSVs are genes x obs; the GCN works obs x genes.
        return csr_matrix(df.to_numpy(dtype=np.float32).T), list(df.columns), list(df.index)

    if spatial_is_h5ad:
        log("Reading spatial .h5ad...")
        sp_files = _h5ad_to_dstg_format(spatial_data, data_dir, "mix")
        mix_X, spot_names, mix_genes = sp_files["X"], sp_files["obs_names"], sp_files["var_names"]
    else:
        log("Using pre-formatted spatial data directory.")
        sp_files = {"count": os.path.join(spatial_data, "mix_count.csv")}
        # A pre-formatted directory is one the user built, so its delimiter is read from the file.
        # `n_spots` comes straight off the parsed shape: a tab-delimited mix_count.csv parsed with
        # a hardcoded comma reported 0 spots and the run continued on it. The directories this
        # worker writes itself are comma-delimited and sniff back as commas, so nothing moves for
        # the h5ad path above.
        mix_X, spot_names, mix_genes = _table_to_sparse(read_indexed_table(sp_files["count"], "spatial counts"))
    n_spots, n_genes_spatial = mix_X.shape
    if spot_names is None or len(spot_names) != n_spots:
        raise ValueError(f"spatial input has {n_spots} spots but {0 if spot_names is None else len(spot_names)} names")
    # Background glass (obs['in_tissue'] == 0) is not tissue: the shared rule leaves it out of the
    # graph, the proportions and the dominant-type table, and the payload says how many.
    n_spots_supplied, n_spots_off_tissue = int(n_spots), 0
    if sp_files.get("in_tissue") is not None:
        kept, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(_ObsView.of(sp_files["in_tissue"]), "spots")
        if n_spots_off_tissue:
            mix_X = mix_X[kept.rows]
            spot_names = [spot_names[i] for i in kept.rows]
            n_spots = int(mix_X.shape[0])
            log(f"Left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0 (background)")

    if sc_is_h5ad:
        log("Reading single-cell .h5ad...")
        sc_files = _h5ad_to_dstg_format(sc_data, data_dir, "sc", cell_type_key=cell_type_key)
        sc_X, _, sc_genes = sc_files["X"], sc_files["obs_names"], sc_files["var_names"]
        sc_labels = sc_files.get("label_values")
        label_key = sc_files.get("label_key")
        labels_matched_by = "file"  # one h5ad: the labels are the obs rows of the matrix
        if sc_labels is None:
            raise ValueError(
                "the single-cell reference has no cell-type labels: none of "
                f"{list(LABEL_KEY_CANDIDATES)} is an obs column and no cell_type_key was given. "
                f"Available obs columns: {sc_files.get('obs_columns')}"
            )
    else:
        log("Using pre-formatted single-cell data directory.")
        sc_files = {
            "count": os.path.join(sc_data, "sc_count.csv"),
            "labels": os.path.join(sc_data, "sc_labels.csv"),
        }
        sc_X, sc_cells, sc_genes = _table_to_sparse(read_indexed_table(sc_files["count"], "single-cell counts"))
        # The label column is the whole reference annotation. A hardcoded comma on a tab-delimited
        # file left `labels_df` with no columns at all, and `.iloc[:, 0]` then raised IndexError
        # out of a line that mentions neither the file nor its delimiter. The shared reader sniffs
        # the separator and, if a file still parses into no data columns, names it.
        labels_df = read_indexed_table(sc_files["labels"], "cell type labels")
        label_key = str(labels_df.columns[0])
        sc_labels, labels_matched_by = _labels_for_count_columns(labels_df, sc_cells, sc_files["labels"])
    n_sc_cells_input = int(sc_X.shape[0])
    # A missing label (NaN, empty, anndata's code -1) is not a cell type: refused by default, left
    # out and counted with drop_unlabeled. The pre-formatted path used to cast NaN to 'nan' and the
    # h5ad path decoded code -1 to 'NA'; both were then fitted and reported as a class.
    keep_labelled, n_unlabeled = split_unlabeled(
        sc_labels, drop_unlabeled, f"reference cells (labels in {label_key!r})"
    )
    if n_unlabeled:
        sc_X = sc_X[np.flatnonzero(keep_labelled)]
        sc_labels = [v for v, ok in zip(sc_labels, keep_labelled) if ok]
        log(f"Left out {n_unlabeled} reference cells with no label in {label_key!r} (drop_unlabeled)")
    sc_labels = [str(v) for v in sc_labels]
    n_sc_cells, n_genes_sc = sc_X.shape
    cell_types = sorted(set(sc_labels))
    n_celltypes = len(cell_types)
    if n_celltypes < 2:
        raise ValueError(
            f"the reference labels in {label_key!r} hold {n_celltypes} class(es); deconvolution needs at least two"
        )
    log(f"Spatial: {n_spots} spots, {n_genes_spatial} genes")
    log(f"SC ref: {n_sc_cells} cells, {n_genes_sc} genes, {n_celltypes} cell types from obs[{label_key!r}]")

    # Suppress TF warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Build the graph inputs. Upstream DSTG's ``load_data`` reads files its R pseudo-spot step
    # writes (Pseudo_ST1.csv, Real_ST2.csv, Infor_Data/*) that nothing here produces, so it is not
    # attempted: this worker builds its own kNN graph over spots + reference cells and trains its
    # own GCN on it. That is what the payload reports.
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import LabelBinarizer

    # Set / dict lookups: `g in set(sc_genes)` rebuilt the set for every spatial gene and
    # `mix_genes.index(g)` scanned the list for every shared one -- quadratic on a 36k-gene panel.
    sc_gene_set = set(sc_genes)
    shared_genes = [g for g in mix_genes if g in sc_gene_set]
    log(f"Shared genes: {len(shared_genes)}")
    if len(shared_genes) < 2:
        raise ValueError(
            f"spatial and reference share {len(shared_genes)} gene name(s); check that both use the same "
            "identifiers (symbols vs Ensembl ids)"
        )
    mix_first = {}
    for i, g in enumerate(mix_genes):
        mix_first.setdefault(g, i)  # list.index semantics: the first occurrence
    mix_idx = [mix_first[g] for g in shared_genes] if len(shared_genes) < len(mix_genes) else None
    sc_pos = {g: i for i, g in enumerate(sc_genes)}
    sc_idx = [sc_pos[g] for g in shared_genes]
    mix_shared = mix_X[:, mix_idx] if mix_idx is not None else mix_X
    sc_shared = sc_X[:, sc_idx]
    features = vstack([mix_shared, sc_shared]).tocsr().astype(np.float32)
    n_total = features.shape[0]

    # kNN adjacency on the shared-gene expression, symmetrised. Sparse throughout: the old code
    # densified this to an (n_spots + n_cells)^2 float32 matrix for a dense tf.placeholder.
    k = int(min(max(n_clusters, 1), n_total - 1))
    log(f"Building kNN graph (k={k}) over {n_total} nodes; pairwise search is O(n^2), sparse result")
    adj = kneighbors_graph(features, k, mode="connectivity", include_self=False)
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj = csr_matrix(adj)

    lb = LabelBinarizer()
    sc_onehot = lb.fit_transform(sc_labels)
    if sc_onehot.ndim == 1 or sc_onehot.shape[1] == 1:  # LabelBinarizer collapses 2 classes to one column
        sc_onehot = np.column_stack([1 - sc_onehot.reshape(-1), sc_onehot.reshape(-1)])
    cell_types = [str(c) for c in lb.classes_]
    n_classes = sc_onehot.shape[1]
    labels_onehot = np.vstack([np.zeros((n_spots, n_classes), dtype=np.float32), sc_onehot.astype(np.float32)])

    train_mask = np.zeros(n_total, dtype=bool)
    train_mask[n_spots:] = True

    # Memory the run will actually need for its dense parts (the sparse operands are cheap).
    hidden_dim = 32
    dense_bytes = _dense_bytes(n_total, hidden_dim, n_classes)
    budget = _memory_budget_bytes()
    if budget is not None and dense_bytes > budget:
        raise MemoryError(
            f"the GCN's dense activations need ~{dense_bytes / 1e9:.1f} GB for {n_total} nodes x "
            f"({hidden_dim} hidden + {n_classes} classes) but ~{budget / 1e9:.1f} GB is available; "
            "run on a machine with more memory (the data is not subsampled)"
        )

    # Row-normalise features (what upstream preprocess_features does), kept sparse.
    log("Row-normalising features...")
    row_sums = np.asarray(features.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    features = csr_matrix(features.multiply(1.0 / row_sums[:, None])).astype(np.float32)

    # TF1 session-based training
    log("Setting up TensorFlow 1.x session for DSTG training...")
    import tensorflow as tf

    if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
        tf = tf.compat.v1
    tf.disable_eager_execution() if hasattr(tf, "disable_eager_execution") else None

    from scipy.sparse import coo_matrix
    from scipy.sparse import eye as speye

    # Normalize adjacency
    def normalize_adj(adj_matrix):
        """Symmetrically normalize adjacency matrix: D^(-1/2) A D^(-1/2)."""
        adj_coo = coo_matrix(adj_matrix)
        rowsum = np.array(adj_matrix.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        from scipy.sparse import diags

        d_mat = diags(d_inv_sqrt)
        return adj_coo.dot(d_mat).T.dot(d_mat).tocoo()

    adj_norm = normalize_adj(adj + speye(adj.shape[0]))
    adj_norm = adj_norm.astype(np.float32)

    n_features = features.shape[1]
    log(f"n_total={n_total}, n_features={n_features}, n_classes={n_classes}")

    def _sparse_tensor_value(m):
        m = coo_matrix(m)
        idx = np.vstack([m.row, m.col]).T.astype(np.int64)
        return tf.SparseTensorValue(idx, m.data.astype(np.float32), m.shape)

    adj_value = _sparse_tensor_value(adj_norm)
    feat_value = _sparse_tensor_value(features)

    # Build simple GCN model in TF1
    tf.reset_default_graph()

    # Placeholders: adjacency and features stay sparse end to end
    ph_features = tf.sparse_placeholder(tf.float32, shape=[n_total, n_features], name="features")
    ph_adj = tf.sparse_placeholder(tf.float32, shape=[n_total, n_total], name="adj")
    ph_labels = tf.placeholder(tf.float32, shape=[n_total, n_classes], name="labels")
    ph_mask = tf.placeholder(tf.float32, shape=[n_total], name="mask")

    # Two-layer GCN
    W1 = tf.Variable(tf.glorot_uniform_initializer()([n_features, hidden_dim]), name="W1")
    W2 = tf.Variable(tf.glorot_uniform_initializer()([hidden_dim, n_classes]), name="W2")

    XW1 = tf.sparse_tensor_dense_matmul(ph_features, W1)
    H1 = tf.nn.relu(tf.sparse_tensor_dense_matmul(ph_adj, XW1))
    logits = tf.sparse_tensor_dense_matmul(ph_adj, tf.matmul(H1, W2))
    predictions = tf.nn.softmax(logits, axis=1)

    # Masked cross-entropy loss
    loss_per_node = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=logits)
    masked_loss = tf.reduce_sum(loss_per_node * ph_mask) / tf.maximum(tf.reduce_sum(ph_mask), 1.0)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(masked_loss)

    # Training
    log(f"Training DSTG GCN for {epochs} epochs (lr={learning_rate})...")
    train_mask_f = train_mask.astype(np.float32)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        feed = {
            ph_features: feat_value,
            ph_adj: adj_value,
            ph_labels: labels_onehot,
            ph_mask: train_mask_f,
        }

        for epoch in range(epochs):
            _, loss_val = sess.run([train_op, masked_loss], feed_dict=feed)
            if (epoch + 1) % 50 == 0 or epoch == 0:
                log(f"  Epoch {epoch + 1}/{epochs}, loss={loss_val:.4f}")

        # Get predictions for spatial spots
        preds = sess.run(predictions, feed_dict=feed)

    log("GCN training completed.")

    # Proportions for the spatial spots only, indexed by their real names (read with the same
    # attrs-aware index reader as the counts; the old code re-opened the file, looked only at
    # obs/_index or obs/index, and swallowed every failure into integer spot ids).
    spot_proportions = preds[:n_spots]
    col_names = cell_types
    proportions_df = pd.DataFrame(spot_proportions, columns=col_names, index=[str(x) for x in spot_names])

    # Save proportions CSV
    proportions_csv = os.path.join(output_dir, "dstg_proportions.csv")
    log(f"Writing cell-type proportions to {proportions_csv}")
    _write_csv_atomic(proportions_df, proportions_csv, index_label="spot")

    # Compute dominant cell type per spot
    dominant_ct = proportions_df.idxmax(axis=1)
    dominant_csv = os.path.join(output_dir, "dstg_dominant_celltype.csv")
    _write_csv_atomic(dominant_ct.to_frame(name="dominant_celltype"), dominant_csv, index_label="spot")

    from collections import Counter

    dominant_counts = dict(Counter(dominant_ct.values))

    # anndata is not available in dstg_env (Python 3.7 + TF 1.15), so no annotated h5ad is written;
    # the proportions CSV is the output.
    n_celltypes_final = len(col_names)

    out = WorkerOutput("dstg", task="deconvolution")
    out.set_data(
        n_spots=int(n_spots),
        n_spots_supplied=int(n_spots_supplied),
        n_reference_cells=int(n_sc_cells),
        n_reference_cells_input=int(n_sc_cells_input),
        n_classes=n_celltypes_final,
        n_shared_genes=len(shared_genes),
    )
    output_files_dict = {
        "proportions_csv": proportions_csv,
        "dominant_celltype_csv": dominant_csv,
        "output_dir": output_dir,
    }
    out.add_output_files(output_files_dict)
    out.add_params(
        {
            "n_clusters": n_clusters,
            "knn_k": k,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "cell_type_key": cell_type_key,
            "label_key_used": label_key,
            "labels_matched_by": labels_matched_by,
            "drop_unlabeled": bool(drop_unlabeled),
            "n_reference_cells_dropped_unlabeled": int(n_unlabeled),
            "hidden_dim": hidden_dim,
        }
    )
    # The GCN is the only implementation this worker has, so it is the method, not a fallback.
    record_method(out, METHOD_NAME)
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    if n_unlabeled:
        out.add_warning(
            f"{n_unlabeled} of {n_sc_cells_input} reference cells had no label in {label_key!r} and were left out "
            "(drop_unlabeled=True)."
        )
    if labels_matched_by == "position":
        out.add_warning(
            f"the index of {sc_files['labels']} does not name every column of {sc_files['count']}, so its "
            f"{n_sc_cells_input} labels were paired with the cells by position"
        )
    for skipped in (sp_files.get("count_csv_skipped_reason"), sc_files.get("count_csv_skipped_reason")):
        if skipped:
            out.add_warning(f"count CSV not written: {skipped}")
    out.set_summary(
        n_cell_types=n_celltypes_final,
        cell_type_names=col_names,
        dominant_counts=dominant_counts,
    )
    out.set_analysis(
        build_deconv_analysis(
            n_celltypes=n_celltypes_final,
            dominant_counts=dominant_counts,
            total_spots=int(n_spots),
            method_name=METHOD_NAME,
        )
        + (
            f" {n_spots_off_tissue} of {n_spots_supplied} spots were background (obs['in_tissue'] == 0) and were "
            f"left out; the {n_spots} in-tissue spots were deconvolved."
            if n_spots_off_tissue
            else ""
        )
        + (
            f" {n_unlabeled} of {n_sc_cells_input} reference cells had no label and were left out (drop_unlabeled)."
            if n_unlabeled
            else ""
        )
    )

    return out.to_dict()


def main() -> None:
    args = parse_args()

    # Redirect all stdout during heavy work to stderr
    orig_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        try:
            result = run_dstg_deconvolution(
                spatial_data=args.spatial_data,
                sc_data=args.sc_data,
                output_dir=args.output_dir,
                n_clusters=args.n_clusters,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                cell_type_key=args.cell_type_key,
                drop_unlabeled=args.drop_unlabeled,
            )
        except Exception as e:
            log("ERROR while running the DSTG-style GCN:")
            traceback.print_exc(file=sys.stderr)
            result = WorkerOutput.error("dstg", str(e), task="deconvolution")
    finally:
        sys.stdout = orig_stdout

    # Final JSON to stdout
    print(json.dumps(result))


if __name__ == "__main__":
    main()
