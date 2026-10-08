#!/usr/bin/env python
"""
celloscope_worker.py

Worker script for Celloscope probabilistic deconvolution.

- Executed inside the celloscope conda env: /opt/conda/envs/celloscope_env
- All logs and progress go to stderr.
- Stdout contains exactly one line of JSON at the end.

What runs, and what this wrapper supplies in its place
-------------------------------------------------------
The sampler is Celloscope's own (``code/impl.py`` of the upstream checkout, one chain). Celloscope's
manual prepares its three inputs by hand; here they are derived, and the payload says so:

* the marker matrix B -- from the single-cell reference: for each cell type, the genes whose
  library-size-normalised mean is highest in that type, ranked by fold change over the next-highest
  type (at most ``n_markers_per_type`` each), among genes counted at least once on the slide.
  Celloscope's own procedure starts from curated candidate lists instead. B always ends with the
  all-zero "dummy type" column Celloscope requires: ``impl.py`` forces the last column's indicator
  to 0, so without it the last real cell type would be held at the "absent" prior in every spot;
* the per-spot cell numbers -- ``n_cells_csv`` when supplied, otherwise one constant
  (``n_cells_per_spot``, default 5) for every spot. In Celloscope's ``ASPRIORS`` mode that value is
  the centre of each spot's prior on its cell number.

The result is ``celloscope_proportions.csv``: spots x reference cell types, each row summing to 1,
from Celloscope's point estimate ``chain01/thetas_est.csv`` (types x spots, unnormalised). The dummy
type's share is removed and each spot renormalised over the reference types; the shares including it
are published beside it. ``result_h.csv`` is the MCMC trace -- one flattened spots x types row per
drop -- and is kept only as such.

Memory
------
Only the marker genes (at most ``n_markers_per_type`` per reference type) ever reach Celloscope, so
neither counts table is held whole. Each is parsed a block of rows at a time (``STREAM_BLOCK_BYTES``
of values per block): the spatial table twice -- per-gene totals first, then the marker columns --
and the reference twice -- its cell IDs and library sizes first, then per-type sums over the
candidate genes. What stays in memory is the spot and cell IDs, per-gene and per-type sums, and the
marker genes x spots matrix Celloscope models (8 bytes a value: ~1.6 GB for 400 markers on a
507,684-bin VisiumHD slide), where reading each table whole took ~73 GB for that slide, plus a copy.

NOTE: Celloscope uses sys.argv at module level in some scripts, so we must
wrap it via subprocess or careful import handling.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import (
    WorkerOutput,
    build_deconv_analysis,
    describe_reduction,
    drop_unlabeled,
    id_mismatch_msg,
    preflight_check,
    read_indexed_table,
    record_ignored,
    record_method,
    sniff_tabular_sep,
)

# Default location for the Celloscope clone: this clone's own tools/third_party, not a fixed box.
# Nothing sets CELLOSCOPE_REPO -- the setup resolver only rebases *_WORKER/*_PYTHON -- so an absolute
# default here is the sole value the worker sees, and it must follow whatever checkout is running it.
CELLOSCOPE_REPO = os.environ.get(
    "CELLOSCOPE_REPO",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "Celloscope"),
)

METHOD_NAME = (
    "Celloscope MCMC (upstream code/impl.py, 1 chain) on a marker matrix derived by this wrapper from the "
    "single-cell reference (not Celloscope's curated-marker selection procedure)"
)
DEFAULT_N_MARKERS_PER_TYPE = 20
DEFAULT_N_CELLS_PER_SPOT = 5
#: ``"number of cells prior strength"`` in params.txt: the SD of the Normal prior on each spot's cell number.
N_CELLS_PRIOR_SD = 2
#: impl.py asserts ``burn in < number of iterations``; the burn-in below is ``max(5, n // 2)``.
MIN_ITERATIONS = 6
#: Name given to Celloscope's trailing all-zero column of B, as its own report labels it.
DUMMY_TYPE = "Dummy type"
#: Library size the reference cells are scaled to before the per-type means are taken.
MARKER_SCALE = 1e4
#: Above this, the estimate still looks like the sampler's random start. Calibrated: a run that
#: recovered known proportions (r=0.997 per type, 40 synthetic spots, 400 iterations) measured 0.18;
#: runs whose proportions tracked nothing measured 0.80-0.91 (library Visium, 200-700 iterations).
START_CORRELATION_WARN = 0.5
PROPORTIONS_FILE = "celloscope_proportions.csv"
H_WITH_DUMMY_FILE = "celloscope_h_with_dummy_type.csv"
#: The counts tables are parsed a block of rows at a time, about this many bytes of values (8 bytes
#: each) per block, so no table is ever held whole. pandas' tokeniser holds several times the block
#: while it parses one: measured on a 2,514 x 36,601 table, 64 MB blocks peaked at 432 MB resident and
#: 256 MB blocks at 1.3 GB, at the same speed (a whole read peaked at 2.1 GB).
STREAM_BLOCK_BYTES = 64 * 1024**2


def log(msg: str) -> None:
    sys.stderr.write(f"[celloscope-worker] {msg}\n")
    sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Celloscope worker: probabilistic deconvolution.")
    parser.add_argument("--spatial-counts", type=str, required=True, help="Spatial counts CSV (spots x genes).")
    parser.add_argument("--sc-counts", type=str, required=True, help="Single-cell counts CSV (cells x genes).")
    parser.add_argument("--cell-type-labels", type=str, required=True, help="Cell type labels CSV.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--n-iterations", type=int, default=1000, help="MCMC iterations (at least 6).")
    parser.add_argument(
        "--n-markers-per-type",
        type=int,
        default=DEFAULT_N_MARKERS_PER_TYPE,
        help="At most this many marker genes per cell type in the marker matrix B.",
    )
    parser.add_argument(
        "--n-cells-per-spot",
        type=int,
        default=DEFAULT_N_CELLS_PER_SPOT,
        help="Constant cell-number estimate for every spot, used when --n-cells-csv is not given.",
    )
    parser.add_argument(
        "--n-cells-csv",
        type=str,
        default="",
        help="Per-spot cell-number estimates (spot IDs, then one count column or a 'cellCount' column).",
    )
    return parser.parse_args()


def _ensure_repo() -> str:
    """Ensure Celloscope repo is cloned and return the repo path."""
    if os.path.isdir(CELLOSCOPE_REPO) and (
        os.path.exists(os.path.join(CELLOSCOPE_REPO, "celloscope"))
        or os.path.exists(os.path.join(CELLOSCOPE_REPO, "code"))
        or os.path.exists(os.path.join(CELLOSCOPE_REPO, "code", "impl.py"))
    ):
        log(f"Celloscope repo found at {CELLOSCOPE_REPO}")
        return CELLOSCOPE_REPO

    log(f"Cloning Celloscope to {CELLOSCOPE_REPO}...")
    os.makedirs(os.path.dirname(CELLOSCOPE_REPO), exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/szczurek-lab/Celloscope.git", CELLOSCOPE_REPO],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    log("Celloscope repo cloned successfully")
    return CELLOSCOPE_REPO


def _mcmc_schedule(n_iterations: int) -> dict[str, int]:
    """The sampler settings derived from ``n_iterations``, and how many draws the estimate averages.

    impl.py averages ``thetas`` over iterations ``i >= burn_in`` with ``i % thinning == 0``; that
    count is what the point estimate rests on, so it is reported rather than left to be inferred.
    """
    if n_iterations < MIN_ITERATIONS:
        raise ValueError(
            f"n_iterations={n_iterations} is too few for Celloscope: the burn-in is max(5, n_iterations // 2) "
            f"and Celloscope requires it to be smaller than n_iterations, so n_iterations must be at least "
            f"{MIN_ITERATIONS}."
        )
    burn_in = max(5, n_iterations // 2)
    thinning = max(1, n_iterations // 10)
    # Multiples of ``thinning`` in [burn_in, n_iterations - 1].
    n_samples = (n_iterations - 1) // thinning - (burn_in - 1) // thinning
    return {
        "burn_in": burn_in,
        "thinning": thinning,
        "how_often_update_step_size": max(3, n_iterations // 5),
        "how_often_drop": max(1, n_iterations // 10),
        "n_posterior_samples": n_samples,
    }


def _reference_labels(labels_df, sc_index):
    """The first labels column matched to the reference cells by ID; ``None`` where a cell has no label.

    Returns ``(labels, n_unlabelled)``. Cells are matched by ID, not by position: a labels file
    listing the same cells in another order is read correctly, and one that shares no ID with the
    counts is refused with both ID sets named.
    """
    import numpy as np

    # object dtype before the reindex below: reindexing an integer column onto cells it does not
    # list promotes it to float, and cluster ids 0/1/2 would then read '0.0'/'1.0'/'2.0' here while
    # the cell types (read from the file's own column) read '0'/'1'/'2' -- no type would match.
    labels = labels_df[labels_df.columns[0]].astype(object)
    if not labels.index.equals(sc_index):
        if not sc_index.is_unique or not labels.index.is_unique:
            raise ValueError(
                "cells are matched to their labels by ID, and the single-cell counts "
                f"({int(sc_index.duplicated().sum())} duplicated) or the cell type labels "
                f"({int(labels.index.duplicated().sum())} duplicated) repeat an ID while listing the cells "
                "differently. Give every cell one unique ID in both files."
            )
        n_shared = int(sc_index.isin(labels.index).sum())
        if n_shared == 0:
            raise ValueError(
                id_mismatch_msg("cell IDs", "single-cell counts", sc_index, "cell type labels", labels.index)
            )
        labels = labels.reindex(sc_index)
    values = labels.to_numpy()
    keep, n_unlabelled = drop_unlabeled(values, allow_drop=True, what="reference cells")
    out = np.array([str(values[i]) if keep[i] else None for i in range(len(values))], dtype=object)
    return out, int(n_unlabelled)


def _table_blocks(path: str, what: str):
    """``path`` as successive blocks of rows, indexed by its first column, the separator sniffed.

    The parse :func:`read_indexed_table` makes -- ``pd.read_csv(sep=<sniffed>, index_col=0)`` -- a
    block at a time: about ``STREAM_BLOCK_BYTES`` of values at once, never the whole table. The first
    block is one row (it tells how wide a row is); a header-only table yields one empty block, so
    its columns are still known. A table that parses into no data column is refused as
    ``read_indexed_table`` refuses it.
    """
    import pandas as pd

    reader = pd.read_csv(path, sep=sniff_tabular_sep(path), index_col=0, iterator=True)
    try:
        block = reader.get_chunk(1)
        if block.shape[1] == 0:
            raise ValueError(
                f"{path}: read as {what}, but it parsed into 0 data columns -- the whole first line "
                f"became the index name ({block.index.name!r}). Check the file's field separator; a "
                f"{what} file needs an ID column and at least one data column."
            )
        yield block
        rows = max(1, int(STREAM_BLOCK_BYTES // (8 * block.shape[1])))
        while True:
            try:
                block = reader.get_chunk(rows)
            except StopIteration:
                break
            yield block
    finally:
        reader.close()


def _numeric_positions(block):
    """Positions of the numeric columns of ``block``, and the names of the others."""
    from pandas.api.types import is_numeric_dtype

    numeric, other = [], []
    for i, (name, dtype) in enumerate(block.dtypes.items()):
        if is_numeric_dtype(dtype):
            numeric.append(i)
        else:
            other.append(str(name))
    return numeric, other


def _scan_table(path: str, what: str, axis: int):
    """One pass over ``path``: its row IDs, its columns, and its sums along ``axis`` (NaN skipped).

    ``axis=0`` sums each column over every row (a gene's total over the spots); ``axis=1`` sums each
    row over every column (a cell's library size). Returns ``(index, columns, sums, non_numeric)``,
    where ``non_numeric`` names the columns holding something other than numbers in any block; they
    are left out of the sums (``NaN`` for an ``axis=0`` total) and the caller decides whether they
    matter.
    """
    import numpy as np
    import pandas as pd

    ids, parts = [], []
    columns = None
    non_numeric = set()
    col_sums = None
    index_name = None
    for block in _table_blocks(path, what):
        if columns is None:
            columns = list(block.columns)
            index_name = block.index.name
            col_sums = np.zeros(len(columns), dtype=float)
        ids.append(block.index)
        numeric, other = _numeric_positions(block)
        non_numeric.update(other)
        values = block.iloc[:, numeric].to_numpy(dtype=float) if other else block.to_numpy(dtype=float)
        if axis == 0:
            col_sums[numeric] += np.nansum(values, axis=0)
        else:
            parts.append(np.nansum(values, axis=1))
    index = ids[0].append(ids[1:]) if len(ids) > 1 else ids[0]
    if len({str(ix.dtype) for ix in ids}) > 1:
        # Each block infers its own ID type: "1", "2", ..., "X1" reads as integers in one block and as
        # text in another. A whole read would have made every ID text, so the IDs are made text here too.
        index = index.astype(str)
    index.name = index_name
    if axis == 0:
        sums = pd.Series(col_sums, index=columns)
        if non_numeric:
            sums[sorted(non_numeric)] = np.nan
    else:
        sums = np.concatenate(parts) if parts else np.zeros(0, dtype=float)
    return index, columns, sums, sorted(non_numeric)


def _refuse_non_numeric(path: str, what: str, names) -> None:
    if names:
        raise ValueError(
            f"{path}: read as {what}, but {len(names)} column(s) hold values that are not numbers, e.g. "
            f"{list(names)[:5]}. A counts table holds numbers only, with the IDs in its first column."
        )


def _type_means(sc_counts: str, genes, labels, cell_types, library_sizes):
    """Per-type mean of library-size-normalised reference expression, genes x cell types.

    Each cell is scaled to ``MARKER_SCALE`` counts over all its genes first (``library_sizes``, in
    the file's row order), so a cell type sequenced deeper than the others does not win every gene
    on depth alone. The reference is read a block of cells at a time and only ``genes`` are taken
    from each block: the per-type sums accumulate, the cells x genes matrix is never built.
    """
    import numpy as np
    import pandas as pd

    genes = list(genes)
    scale_all = np.zeros_like(library_sizes, dtype=float)
    np.divide(MARKER_SCALE, library_sizes, out=scale_all, where=library_sizes > 0)
    sums = np.zeros((len(genes), len(cell_types)), dtype=float)
    counts = np.zeros(len(cell_types), dtype=np.int64)
    start = 0
    for block in _table_blocks(sc_counts, "single-cell counts"):
        stop = start + block.shape[0]
        X = block[genes].to_numpy(dtype=float)
        if X.size and np.nanmin(X) < 0:
            raise ValueError(
                "the single-cell reference holds negative values; marker genes are chosen from per-type means of "
                "expression, which needs counts (or another non-negative expression scale)."
            )
        block_labels = labels[start:stop]
        scale = scale_all[start:stop]
        for j, ct in enumerate(cell_types):
            mask = block_labels == ct
            n = int(mask.sum())
            if n:
                sums[:, j] += scale[mask] @ X[mask]
                counts[j] += n
        start = stop
    if start != len(labels):
        raise RuntimeError(
            f"{sc_counts} changed while it was read: {start} cells on the second pass, {len(labels)} on the first."
        )
    means = np.zeros_like(sums)
    np.divide(sums, counts, out=means, where=counts > 0)
    return pd.DataFrame(means, index=genes, columns=list(cell_types))


def _marker_columns(spatial_counts: str, marker_genes, index):
    """The spatial table's ``marker_genes`` columns (spots x markers), read a block of spots at a time."""
    import pandas as pd

    parts = [block[list(marker_genes)] for block in _table_blocks(spatial_counts, "spatial counts")]
    frame = pd.concat(parts, axis=0) if len(parts) > 1 else parts[0]
    if frame.shape[0] != len(index):
        raise RuntimeError(
            f"{spatial_counts} changed while it was read: {frame.shape[0]} spots on the second pass, "
            f"{len(index)} on the first."
        )
    frame.index = index
    return frame


def _marker_matrix(mean_expr, n_markers_per_type: int):
    """Binary genes x cell types marker matrix B from per-type means (genes x cell types).

    A gene is a candidate marker only for the type whose mean is highest, and only when that mean is
    strictly above every other type's; candidates are ranked by ``log1p(best) - log1p(next best)``
    and the top ``n_markers_per_type`` kept. A gene expressed alike in every type -- a housekeeping
    gene -- is therefore no type's marker. Rows keep ``mean_expr``'s order; genes that are no type's
    marker are dropped. Returns ``(B, {cell type: number of markers})``.
    """
    import numpy as np
    import pandas as pd

    values = mean_expr.to_numpy(dtype=float)
    n_genes, n_types = values.shape
    B = pd.DataFrame(0, index=mean_expr.index, columns=mean_expr.columns, dtype=int)
    per_type = {str(ct): 0 for ct in mean_expr.columns}
    if n_genes == 0 or n_types < 2:
        return B.iloc[:0], per_type
    order = np.argsort(values, axis=1, kind="stable")
    rows = np.arange(n_genes)
    best = values[rows, order[:, -1]]
    runner_up = values[rows, order[:, -2]]
    score = np.log1p(best) - np.log1p(runner_up)
    top = order[:, -1]
    for j, ct in enumerate(mean_expr.columns):
        idx = np.where((top == j) & (score > 0))[0]
        idx = idx[np.argsort(-score[idx], kind="stable")][:n_markers_per_type]
        B.iloc[np.sort(idx), j] = 1
        per_type[str(ct)] = int(len(idx))
    return B.loc[B.sum(axis=1) > 0], per_type


def _dummy_column(cell_types) -> str:
    name = DUMMY_TYPE
    while name in cell_types:
        name = f"_{name}_"
    return name


def _require_counts(C_gs, path: str) -> None:
    """Celloscope's likelihood is negative binomial and impl.py asserts integer counts; say so first."""
    import numpy as np

    values = C_gs.to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    finite = np.where(missing, 0.0, values)
    negative = finite < 0
    fractional = finite != np.round(finite)
    n_bad = int(missing.sum() + negative.sum() + fractional.sum())
    if n_bad:
        raise ValueError(
            f"{path}: Celloscope models raw counts (negative binomial) and needs whole, non-negative numbers. "
            f"Across the {C_gs.shape[0]} marker genes it would read, {int(fractional.sum())} values are "
            f"fractional, {int(negative.sum())} negative and {int(missing.sum())} missing. Supply raw counts, "
            "not normalised or log-transformed values."
        )


def _read_n_cells(path: str, spot_ids) -> Any:
    """Per-spot cell numbers from ``path``, in ``spot_ids`` order, as whole non-negative numbers.

    Accepts spot IDs then one count column, or Celloscope's own ``n_cells.csv`` layout (row number,
    ``spotId``, ``cellCount``).
    """
    import numpy as np
    import pandas as pd

    frame = read_indexed_table(path, "per-spot cell counts")
    if "spotId" in frame.columns:
        frame = frame.set_index("spotId")
    if "cellCount" in frame.columns:
        column = "cellCount"
    elif frame.shape[1] == 1:
        column = frame.columns[0]
    else:
        raise ValueError(
            f"{path}: cannot tell which column holds the cell counts (columns {list(frame.columns)}). Give the "
            "spot IDs and one count column, or name the count column 'cellCount'."
        )
    frame.index = frame.index.astype(str)
    wanted = [str(s) for s in spot_ids]
    if not frame.index.is_unique:
        raise ValueError(f"{path}: {int(frame.index.duplicated().sum())} spot IDs appear more than once.")
    n_shared = int(pd.Index(wanted).isin(frame.index).sum())
    if n_shared < len(wanted):
        raise ValueError(
            f"n_cells_csv covers {n_shared} of the {len(wanted)} spots; every spot needs an estimate. "
            + id_mismatch_msg("spot IDs", "spatial counts", wanted, "n_cells_csv", frame.index, n_common=n_shared)
        )
    counts = pd.to_numeric(frame[column], errors="coerce").reindex(wanted).to_numpy(dtype=float)
    bad = ~np.isfinite(counts) | (counts < 0) | (counts != np.round(counts))
    if bad.any():
        raise ValueError(
            f"{path}: {int(bad.sum())} of {len(wanted)} cell counts are not whole non-negative numbers "
            "(Celloscope asserts integer cell numbers)."
        )
    return counts.astype(int)


def _proportions_from_thetas(thetas_path: str, spot_ids, cell_types, dummy: str):
    """``(proportions, h_with_dummy)`` from Celloscope's point estimate ``thetas_est.csv``.

    impl.py writes ``pd.DataFrame(theta_est).T``: types (B's column order, dummy last) x spots (C's
    column order), headerless and unnormalised. ``h_with_dummy`` is each spot's share over every
    component, as Celloscope's own report computes it; ``proportions`` drops the dummy type and
    renormalises over the reference cell types.
    """
    import numpy as np
    import pandas as pd

    raw = pd.read_csv(thetas_path, header=None)
    expected = (len(cell_types) + 1, len(spot_ids))
    if raw.shape != expected:
        raise RuntimeError(
            f"{thetas_path} is {raw.shape[0]} x {raw.shape[1]}; Celloscope writes its estimate as types x spots, "
            f"so {expected[0]} x {expected[1]} ({len(cell_types)} cell types + the dummy type, {len(spot_ids)} "
            "spots) was expected."
        )
    thetas = raw.to_numpy(dtype=float).T
    if not np.isfinite(thetas).all() or (thetas < 0).any() or (thetas.sum(axis=1) <= 0).any():
        raise RuntimeError(f"{thetas_path} holds non-finite, negative or all-zero estimates; nothing to normalise.")
    h = thetas / thetas.sum(axis=1, keepdims=True)
    h_all = pd.DataFrame(h, index=spot_ids, columns=list(cell_types) + [dummy])
    real = h_all[list(cell_types)]
    return real.div(real.sum(axis=1), axis=0), h_all


def _write_csv_atomic(frame, path: str, index: bool = True) -> None:
    partial = path + ".partial"
    frame.to_csv(partial, index=index)
    os.replace(partial, path)


def _start_correlation(trace_path: str, h_all) -> Any:
    """Correlation between the sampler's first recorded draw and the published estimate, or None.

    impl.py starts every spot's thetas at a random Gamma(1) draw and moves them by a random walk
    with step 0.1, and the first row of ``result_h.csv`` is recorded after one iteration. An
    estimate that still correlates strongly with that row has not left its random start, whatever
    the number of iterations says. ``None`` when the trace is absent or not the expected length.
    """
    import numpy as np
    import pandas as pd

    if not os.path.isfile(trace_path):
        return None
    try:
        first = pd.read_csv(trace_path, header=None, nrows=1).to_numpy(dtype=float).reshape(-1)
    except (ValueError, pd.errors.EmptyDataError):
        return None
    estimate = h_all.to_numpy(dtype=float).reshape(-1)
    if first.shape != estimate.shape or np.std(first) == 0 or np.std(estimate) == 0:
        return None
    return float(np.corrcoef(first, estimate)[0, 1])


def run_celloscope_pipeline(
    spatial_counts: str,
    sc_counts: str,
    cell_type_labels: str,
    output_dir: str,
    n_iterations: int,
    n_markers_per_type: int = DEFAULT_N_MARKERS_PER_TYPE,
    n_cells_per_spot: int = DEFAULT_N_CELLS_PER_SPOT,
    n_cells_csv: str = "",
) -> dict[str, Any]:
    """Run Celloscope deconvolution.

    Celloscope expects specific input files:
    - C_gs.csv: gene-by-spot count matrix
    - matB.csv: gene-by-celltype marker gene indicator matrix
    - n_cells.csv: estimated number of cells per spot
    - params.txt: JSON parameter file

    We prepare these from the user-provided counts + labels (see the module docstring for how).
    """
    import numpy as np
    import pandas as pd

    inputs = {
        "spatial_counts": spatial_counts,
        "sc_counts": sc_counts,
        "cell_type_labels": cell_type_labels,
    }
    if n_cells_csv:
        inputs["n_cells_csv"] = n_cells_csv
    preflight_check(inputs=inputs, output_dir=output_dir)
    schedule = _mcmc_schedule(int(n_iterations))
    if n_markers_per_type < 1:
        raise ValueError(f"n_markers_per_type={n_markers_per_type}: at least one marker per cell type is needed.")
    if not n_cells_csv and n_cells_per_spot < 1:
        raise ValueError(f"n_cells_per_spot={n_cells_per_spot}: a spot's cell-number estimate must be at least 1.")

    repo_path = _ensure_repo()
    out = WorkerOutput("celloscope", task="deconvolution")

    # Load input data. Neither counts table is read whole (see "Memory" in the module docstring): the
    # first pass over each keeps only its IDs, its columns and the sums marker selection needs.
    log("Loading input data...")
    spot_ids, st_genes, spatial_gene_totals, st_non_numeric = _scan_table(spatial_counts, "spatial counts", axis=0)
    cell_ids, sc_genes_list, library_sizes, sc_non_numeric = _scan_table(sc_counts, "single-cell counts", axis=1)
    _refuse_non_numeric(sc_counts, "single-cell counts", sc_non_numeric)
    labels_df = read_indexed_table(cell_type_labels, "cell type labels")
    n_spots = len(spot_ids)
    n_genes_st = len(st_genes)
    n_cells_total = len(cell_ids)

    # Determine cell types from labels. read_indexed_table guarantees at least one data column, so
    # there is no frame here without one -- the branch that used to handle that case called
    # `.unique()` on a DataFrame and raised AttributeError naming neither the file nor its
    # separator, which is exactly the state a tab-delimited labels file arrived in.
    cell_type_col = labels_df.columns[0]
    cell_types = labels_df[cell_type_col].unique().tolist()
    cell_types = [str(ct) for ct in cell_types if str(ct) not in ("nan", "None", "")]
    n_celltypes = len(cell_types)
    log(f"Data: n_spots={n_spots}, n_genes={n_genes_st}, n_cells={n_cells_total}, n_celltypes={n_celltypes}")

    # Prepare Celloscope input directory
    data_dir = os.path.join(output_dir, "celloscope_data")
    results_dir = os.path.join(output_dir, "celloscope_results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Genes on both sides, in the slide's own order. A set's order changes with the interpreter's
    # string-hash seed, and B's row order is the order Celloscope draws its per-gene variables in,
    # so an unordered intersection gave a different estimate on every run of the same input.
    sc_genes = set(sc_genes_list)
    common_genes = [g for g in st_genes if g in sc_genes]
    if len(common_genes) < 10:
        raise ValueError(
            id_mismatch_msg("genes", "spatial", st_genes, "scRNA-seq", sc_genes_list, n_common=len(common_genes))
        )
    log(f"Common genes for marker selection: {len(common_genes)}")
    # A shared gene that is not a number on the slide cannot be summed or modelled; any other
    # non-numeric column of the slide never reaches Celloscope, as before.
    _refuse_non_numeric(spatial_counts, "spatial counts", [g for g in common_genes if g in set(st_non_numeric)])

    # Reference cells and their labels, matched by ID.
    cell_labels, n_unlabelled = _reference_labels(labels_df, cell_ids)
    present = {v for v in cell_labels if v is not None}
    ref_types = [ct for ct in cell_types if ct in present]
    absent_types = [ct for ct in cell_types if ct not in present]
    if n_unlabelled:
        out.add_warning(
            f"{n_unlabelled} of {n_cells_total} reference cells have no label and were left out of marker selection."
        )
    if absent_types:
        out.add_warning(
            f"{len(absent_types)} cell type(s) in the labels file have no cell in the single-cell counts and are not "
            f"modelled: {absent_types[:10]}"
        )
    if len(ref_types) < 2:
        raise ValueError(
            f"Celloscope needs at least two reference cell types with cells in the single-cell counts; found "
            f"{len(ref_types)} ({ref_types})."
        )

    # A gene with no count in any spot cannot be a marker: Celloscope's per-gene prior is the mean of
    # its non-zero counts, which is NaN for such a gene, and one NaN row makes every spot's likelihood
    # NaN -- no proposal is ever accepted and the "estimate" is the sampler's random starting point.
    candidate_genes = [g for g in common_genes if spatial_gene_totals[g] > 0]
    n_genes_unexpressed = len(common_genes) - len(candidate_genes)

    # 2. matB.csv: gene x celltype marker indicator (binary)
    mean_expr = _type_means(sc_counts, candidate_genes, cell_labels, ref_types, library_sizes)
    matB, markers_per_type = _marker_matrix(mean_expr, int(n_markers_per_type))
    if matB.shape[0] == 0:
        raise ValueError(
            f"no gene qualifies as a marker: of {len(candidate_genes)} genes shared with the reference and counted "
            "on the slide, none is expressed more highly in one reference cell type than in all others."
        )
    types_without_markers = [ct for ct in ref_types if markers_per_type.get(ct, 0) == 0]
    if types_without_markers:
        out.add_warning(
            f"{len(types_without_markers)} cell type(s) have no marker gene, so Celloscope cannot tell them from "
            f"the dummy type's background: {types_without_markers[:10]}"
        )
    dummy = _dummy_column(ref_types)
    matB[dummy] = 0
    marker_genes = matB.index.tolist()

    # 1. C_gs.csv: gene x spot count matrix (genes as rows, spots as columns), marker genes only --
    # the second pass over the slide, which keeps only these columns of each block.
    C_gs = _marker_columns(spatial_counts, marker_genes, spot_ids).T
    _require_counts(C_gs, spatial_counts)
    _write_csv_atomic(C_gs.round().astype(np.int64), os.path.join(data_dir, "C_gs.csv"))
    log(f"Saved C_gs.csv: {C_gs.shape}")
    # Written in place, not via .partial: an input staging file impl.py reads after this run returns, and
    # test/test_marker_tables_are_not_deconvolution_predictions.py pins this exact line.
    matB.to_csv(os.path.join(data_dir, "matB.csv"))
    log(f"Saved matB.csv: {matB.shape} ({int(matB.to_numpy().sum())} marker entries, last column {dummy!r})")

    # 3. n_cells.csv: estimated number of cells per spot
    if n_cells_csv:
        cell_numbers = _read_n_cells(n_cells_csv, spot_ids)
        n_cells_source = "n_cells_csv"
        if n_cells_per_spot != DEFAULT_N_CELLS_PER_SPOT:
            record_ignored(out, "n_cells_per_spot", "n_cells_csv supplies a cell-number estimate for every spot")
    else:
        cell_numbers = np.full(n_spots, int(n_cells_per_spot), dtype=int)
        n_cells_source = "constant"
        out.add_warning(
            f"no per-spot cell counts were supplied: every spot's cell number is assumed to be {n_cells_per_spot} "
            f"(the centre of a Normal({n_cells_per_spot}, {N_CELLS_PRIOR_SD}) prior). Pass n_cells_csv with "
            "per-spot estimates, e.g. from nuclei segmentation, or set n_cells_per_spot."
        )
    _write_csv_atomic(pd.DataFrame({"cellCount": cell_numbers}), os.path.join(data_dir, "n_cells.csv"), index=False)

    # 4. params.txt: JSON parameters
    params = {
        "number of iterations": int(n_iterations),
        "burn in": schedule["burn_in"],
        "mode number of cells": "ASPRIORS",
        "a": 10,
        "b": 1,
        "a_0": 0.1,
        "b_0": 1,
        "alpha": 8,
        "step size thetas": 0.1,
        "step size number of cells": 2.01,
        "step size lambda_0": 0.05,
        "step size p_g": 0.1,
        "thinning_parameter": schedule["thinning"],
        "number of cells prior strength": N_CELLS_PRIOR_SD,
        "how often update step size": schedule["how_often_update_step_size"],
        "how often drop": schedule["how_often_drop"],
    }
    params_path = os.path.join(data_dir, "params.txt")
    with open(params_path + ".partial", "w") as f:
        json.dump(params, f)
    os.replace(params_path + ".partial", params_path)
    log(f"Saved params.txt: {params}")

    # Run Celloscope via direct Python import
    impl_path = os.path.join(repo_path, "code", "impl.py")
    if not os.path.exists(impl_path):
        raise FileNotFoundError(f"Celloscope impl.py not found at {impl_path}")

    # Celloscope's impl.py has module-level code that reads sys.argv[1:3]
    # and calls Celloscope() directly. We must temporarily set sys.argv
    # before loading the module so the module-level code runs correctly.
    log(f"Running Celloscope with {n_iterations} iterations...")
    saved_argv = sys.argv[:]
    try:
        sys.argv = ["impl.py", data_dir, results_dir, "1"]
        import importlib.util

        spec = importlib.util.spec_from_file_location("celloscope_impl", impl_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # The module-level code already calls Celloscope() via sys.argv
    finally:
        sys.argv = saved_argv
    log("Celloscope completed")

    # Collect output files
    output_files = {}
    for root, _dirs, files in os.walk(results_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            output_files[fname] = fpath

    # The point estimate, not the trace: result_h.csv holds one flattened spots x types row per drop.
    chain_dir = os.path.join(results_dir, "chain01")
    thetas_path = os.path.join(chain_dir, "thetas_est.csv")
    if not os.path.isfile(thetas_path):
        raise RuntimeError(f"Celloscope finished without writing its point estimate {thetas_path}.")
    proportions, h_all = _proportions_from_thetas(thetas_path, spot_ids, ref_types, dummy)
    proportions.index.name = spot_ids.name or "spot"
    h_all.index.name = proportions.index.name
    proportions_path = os.path.join(output_dir, PROPORTIONS_FILE)
    h_all_path = os.path.join(output_dir, H_WITH_DUMMY_FILE)
    _write_csv_atomic(proportions, proportions_path)
    _write_csv_atomic(h_all, h_all_path)
    log(f"Saved {PROPORTIONS_FILE}: {proportions.shape}")

    dominant = proportions.idxmax(axis=1).value_counts()
    dominant_counts = {str(k): int(v) for k, v in dominant.items()}
    dummy_share = h_all[dummy]
    n_dummy_dominant = int((h_all.idxmax(axis=1) == dummy).sum())
    mean_dummy_share = float(dummy_share.mean())
    trace_path = os.path.join(chain_dir, "result_h.csv")
    start_corr = _start_correlation(trace_path, h_all)
    unmixed = start_corr is not None and start_corr > START_CORRELATION_WARN
    if unmixed:
        out.add_warning(
            f"the published estimate still correlates r={start_corr:.2f} with the sampler's first recorded draw, "
            f"which is a random start: after {n_iterations} iterations the chain has not moved far from where it "
            "began, so these proportions mostly reflect that random start. Raise n_iterations (Celloscope's own "
            "example runs 15000, with burn in 10000); a run that recovered known proportions measured r=0.18."
        )

    out.set_data(
        n_spots=int(n_spots),
        n_genes=int(n_genes_st),
        n_cells=int(n_cells_total),
        n_celltypes=int(len(ref_types)),
        n_genes_common=int(len(common_genes)),
        n_genes_used=int(len(marker_genes)),
        n_genes_unexpressed_on_slide=int(n_genes_unexpressed),
        n_reference_cells_used=int(n_cells_total - n_unlabelled),
        n_reference_cells_unlabelled=int(n_unlabelled),
        n_spots_dummy_type_dominant=n_dummy_dominant,
        mcmc_start_correlation=None if start_corr is None else round(start_corr, 4),
    )
    out.add_output_files(output_files)
    out.add_output_file("proportions_csv", proportions_path)
    out.add_output_file("h_with_dummy_type_csv", h_all_path)
    out.add_output_file("thetas_est_csv", thetas_path)
    out.add_output_file("mcmc_trace_h_csv", trace_path)
    record_method(out, METHOD_NAME)
    out.add_params(
        {
            "n_iterations": int(n_iterations),
            "n_markers_per_type": int(n_markers_per_type),
            "n_cells_per_spot": int(n_cells_per_spot),
            "n_cells_csv": n_cells_csv,
            "cell_types": ref_types,
            "n_chains": 1,
            "burn_in": schedule["burn_in"],
            "thinning": schedule["thinning"],
            "n_posterior_samples": schedule["n_posterior_samples"],
            "n_cells_source": n_cells_source,
            "n_cells_mode": "ASPRIORS",
            "n_cells_prior_sd": N_CELLS_PRIOR_SD,
            "marker_selection": (
                "per cell type, genes whose library-size-normalised reference mean (per "
                f"{int(MARKER_SCALE)} counts) is highest in that type, ranked by log1p fold change over the "
                f"next-highest type, at most {int(n_markers_per_type)} per type, among genes counted on the slide"
            ),
            "n_marker_genes": int(len(marker_genes)),
            "markers_per_type": markers_per_type,
            "dummy_type_column": dummy,
            "proportions_normalisation": (
                "posterior-mean thetas normalised per spot; the dummy type's share removed and each spot "
                "renormalised over the reference cell types"
            ),
        }
    )
    out.set_summary(
        n_celltypes=int(len(ref_types)),
        dominant_counts=dominant_counts,
        mean_dummy_type_share=round(mean_dummy_share, 4),
    )
    analysis = build_deconv_analysis(
        n_celltypes=len(ref_types),
        dominant_counts=dominant_counts if dominant_counts else None,
        total_spots=n_spots,
        method_name="Celloscope",
    )
    analysis += (
        f" Marker matrix: {len(marker_genes)} marker genes chosen from the reference (at most {n_markers_per_type} "
        f"per type) plus Celloscope's dummy type; the dummy type is the largest component in {n_dummy_dominant} "
        f"spots (mean share {mean_dummy_share:.3f}) and is left out of the proportions, which are renormalised "
        f"over the reference types. Posterior mean of {schedule['n_posterior_samples']} draws after "
        f"{schedule['burn_in']} burn-in iterations, one chain."
    )
    if unmixed:
        analysis += (
            f" WARNING: the estimate still correlates r={start_corr:.2f} with the sampler's random start; the chain "
            "has not mixed, so the proportions above mostly reflect that start. Raise n_iterations."
        )
    if n_cells_source == "constant":
        analysis += f" No per-spot cell counts were supplied; every spot's prior cell number was {n_cells_per_spot}."
    analysis += describe_reduction("genes", int(n_genes_st), int(len(marker_genes)), "marker selection")
    out.set_analysis(analysis)
    return out.to_dict()


def main() -> None:
    args = parse_args()

    orig_stdout = sys.stdout
    sys.stdout = sys.stderr
    error_exc = None
    try:
        try:
            result = run_celloscope_pipeline(
                spatial_counts=args.spatial_counts,
                sc_counts=args.sc_counts,
                cell_type_labels=args.cell_type_labels,
                output_dir=args.output_dir,
                n_iterations=args.n_iterations,
                n_markers_per_type=args.n_markers_per_type,
                n_cells_per_spot=args.n_cells_per_spot,
                n_cells_csv=args.n_cells_csv,
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
        WorkerOutput.emit_error("celloscope", error_msg, task="deconvolution", exc=error_exc)
        sys.exit(1)
    else:
        print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
