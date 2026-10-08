#!/usr/bin/env python
"""
stdgcn_worker.py

Worker script for running STdGCN spatial deconvolution via graph
convolutional network.

- Called by the FastMCP wrapper (stdgcn_mcp_server.py).
- Must be executed inside the STdGCN conda env: /opt/conda/envs/stdgcn_env
- All logs go to stderr; stdout only prints a single JSON line at the end.

STdGCN expects TSV/CSV files on disk (not AnnData objects):
  sc_path/sc_data.tsv   - expression matrix (cells x genes)
  sc_path/sc_label.tsv  - cell type labels (barcode, label)
  ST_path/ST_data.tsv   - expression matrix (spots x genes)
  ST_path/coordinates.csv - spatial coordinates (barcode, x, y)

This worker converts h5ad inputs to the required file formats.

Example (manual test):

  (stdgcn_env) python /workspace/epic-fermat/agent/tools/stdgcn_worker.py \
      --spatial-h5ad /workspace/work/spatial_input/V1_Human_Lymph_Node.h5ad \
      --sc-h5ad /workspace/spatial_demo_data/sc_annotation_ref/sc.h5ad \
      --output-dir /workspace/work/stdgcn_V1_LN \
      --cell-type-key CellType \
      --n-epochs 200 \
      --device CPU
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
# Add STdGCN repo to path
# <TOOL>_SRC seam (as spatialscope_worker.py does): this checkout exists only where it was
# installed, and a sys.path entry that does not exist fails silently -- the run dies later in an
# ImportError naming an upstream module, with no way to redirect it. The literal stays the default.
sys.path.insert(
    0,
    os.environ.get("STDGCN_SRC")
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "stdgcn_repo"),
)

from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    build_deconv_analysis,
    cpu_budget,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    preflight_check,
    record_in_tissue,
    record_method,
    resolve_compute,
    sanitize_cell_type_names,
    sniff_tabular_sep,
    spatial_coords,
)
from worker_utils import (
    drop_unlabeled as _split_unlabeled,
)

# --- STdGCN's spatial graph -------------------------------------------------------------------
#
# ``intra_dist_adj`` (STdGCN/adjacency_matrix.py:85-124) links each spot to those of its 27 nearest
# neighbours that lie closer than ``space_dist_threshold`` and, with ``link_method='soft'``, weighs
# the link 1/distance. Both the cut and the weights are written for the unit of the tutorial's
# coordinates, which are array positions one spot pitch apart (data/ST_data/coordinates.csv:
# ``spot338,22,8``): threshold 2 keeps the ring of immediate neighbours, and a neighbour one pitch
# away weighs 1.
#
# This worker handed STdGCN obsm['spatial'] -- pixels or microns, ~138 px between Visium spots --
# and kept the threshold of 2. No two spots on any library sample are closer than 2 px, so the
# adjacency was all zeros, ``adj_sp`` (STdGCN.py:191) became the identity, and the "graph
# convolutional" deconvolution ran with no spatial graph at all, status ok, nothing reported.
#
# The coordinates are therefore put into STdGCN's own unit before it reads them: divided by the
# median nearest-neighbour distance, so the spot pitch is 1 whatever the platform's unit, and the
# payload reports the pitch that was measured and the number of spatial edges the graph has.
#
# The cut is 1.9 pitches, not the tutorial's literal 2. On the tutorial's integer grid, 2 links the
# first ring (1) and the diagonals (1.414) and, being a strict '<', nothing at exactly 2. Measured
# coordinates are not integers: a Visium lattice has a ring at exactly 2 pitches (and at 1.732), and
# after rescaling those distances land a rounding error either side of 2, so a cut AT 2 links an
# arbitrary part of that ring -- on a clean 8 x 8 hex lattice it linked 28 of its pairs and not the
# rest. 1.9 keeps the tutorial's neighbourhood (every ring below 2, none at 2) with a margin on both
# sides, on square and hexagonal lattices alike.
SPACE_DIST_THRESHOLD = 1.9
#: Hard-coded in ``intra_dist_adj(space_dist_neighbors=27)``; run_STdGCN does not pass it.
SPACE_DIST_NEIGHBORS = 27
SPATIAL_LINK_METHOD = "soft"

#: STdGCN builds its graphs as dense float64 matrices over every real spot plus every pseudo-spot
#: (``A_intra_transfer`` and ``inter_adj`` allocate ``np.zeros((n, n))``; STdGCN.py:141-191 holds
#: four of them at once before the torch copies are made). This is intrinsic to the method, so the
#: worker estimates it and refuses with the numbers rather than letting the kernel kill the process.
DENSE_GRAPH_MATRICES_AT_ONCE = 4

#: run_STdGCN is asked for 10 pseudo-spots per real spot, at most this many.
PSEUDO_SPOTS_PER_SPOT = 10
MAX_PSEUDO_SPOTS = 30000

#: numpy's BLAS threads while run_STdGCN runs. The OpenBLAS numpy bundles in stdgcn_env
#: (0.3.23.dev) dies with SIGSEGV in a multi-threaded dsyrk -- the kernel numpy uses for ``x @ x.T``
#: -- once the product is about 25k square: measured under a 3-core affinity mask, a 26000 x 435 and
#: a 28120 x 435 ``x @ x.T`` crash at 2 and 3 threads and run at 1, and a general gemm of the same
#: shape does not crash. STdGCN computes exactly that product: ``cosine_similarity(pseudo, pseudo)``
#: in ``find_mutual_nn`` (adjacency_matrix.py:19) over its pseudo-spots, 10 per real spot up to
#: 30000. So a slide of about 2,500 spots or more killed the worker with no JSON at all -- the
#: SpinalCord Visium sample (2812 spots, 28120 pseudo-spots) did, three runs out of three. One BLAS
#: thread for the duration of run_STdGCN; torch's own thread pool, which trains the GCN, is not a
#: BLAS pool threadpoolctl limits here. The single-threaded product took 13 s on that sample.
STDGCN_BLAS_THREADS = 1


def log(msg: str) -> None:
    """Print log messages to stderr with a prefix."""
    sys.stderr.write(f"[stdgcn-worker] {msg}\n")
    sys.stderr.flush()


def stdgcn_device_token(device: str) -> str:
    """Normalise any device spelling to the exact word STdGCN compares against.

    STdGCN takes a word, not a torch device string, and tests it with a case-sensitive ``==``:
    ``GCN.py:165`` is ``if GCN_device == 'CPU': <cpu> else: <cuda>`` and ``autoencoder.py:46`` is
    ``if device == 'GPU': <cuda>``. The two disagree about which side an unmatched value falls on, so
    the lower-case spellings every other tool in the fleet uses both land wrong: ``'cpu'`` takes the
    GCN's else-branch and trains on the GPU, while ``'gpu'`` never matches the autoencoder's test and
    trains that half on the CPU. Resolve the request once, here, and hand the library one of the two
    words it can actually recognise.
    """
    return "GPU" if resolve_compute(device).device.startswith("cuda") else "CPU"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STdGCN worker: spatial deconvolution via graph convolutional network."
    )
    parser.add_argument(
        "--spatial-h5ad",
        type=str,
        required=True,
        help="Path to spatial transcriptomics AnnData (.h5ad).",
    )
    parser.add_argument(
        "--sc-h5ad",
        type=str,
        required=True,
        help="Path to annotated single-cell reference AnnData (.h5ad).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save STdGCN deconvolution outputs.",
    )
    parser.add_argument(
        "--cell-type-key",
        type=str,
        default="cell_type",
        help="Column name in sc_h5ad.obs with cell-type labels.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=200,
        help="Number of training epochs for STdGCN.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Compute device: 'CPU' or 'GPU'.",
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="Leave out reference cells whose label is missing (NaN/empty) instead of refusing the run.",
    )
    return parser.parse_args()


def _n_pseudo_spots(n_spots: int) -> int:
    """The pseudo-spots run_STdGCN is asked to simulate: 10 per real spot, at most 30000."""
    return min(MAX_PSEUDO_SPOTS, int(n_spots) * PSEUDO_SPOTS_PER_SPOT)


def _write_obs_by_features_tsv(X, obs_names, var_names, path: str, max_block_bytes: int = 256 << 20) -> None:
    """Write ``X`` (obs x features) as the TSV STdGCN reads, a block of rows at a time, atomically.

    Byte-for-byte what ``pd.DataFrame(X.toarray(), index=obs, columns=var).to_csv(path, sep="\t")``
    wrote, without holding the whole matrix dense in this process: only ``block x n_features`` values
    exist at once. The text file is STdGCN's input format; a dense copy of a Visium HD slide (~37 GB)
    or a large reference in the worker before STdGCN even starts is not. Written through
    ``path.partial`` + ``os.replace``, so a killed run leaves no half file under the name STdGCN reads.
    """
    import numpy as np
    import pandas as pd
    from scipy.sparse import issparse

    n_obs, n_var = X.shape
    if issparse(X):
        X = X.tocsr()
    step = max(1, int(max_block_bytes // (max(1, n_var) * 8)))
    columns = pd.Index(var_names)
    partial = path + ".partial"
    try:
        with open(partial, "w") as fh:
            start = 0
            while True:
                stop = min(n_obs, start + step)
                block = X[start:stop]
                block = block.toarray() if issparse(block) else np.asarray(block)
                frame = pd.DataFrame(block, index=obs_names[start:stop], columns=columns)
                frame.to_csv(fh, sep="\t", header=(start == 0))
                start = stop
                if start >= n_obs:
                    break
        os.replace(partial, path)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise


def _h5ad_to_stdgcn_format(
    spatial_h5ad: str,
    sc_h5ad: str,
    cell_type_key: str,
    output_dir: str,
    drop_unlabeled: bool = False,
) -> dict:
    """
    Convert h5ad files to STdGCN-compatible TSV/CSV format.

    Returns dict with keys: sc_path, ST_path, cell_types, cell_types_library_order,
    n_spots, n_sc_cells, n_sc_cells_dropped_unlabeled, n_pseudo_spots, memory, etc.

    The spatial object is read first, and everything that can refuse the run is decided before a
    byte is staged: its background spots (``obs['in_tissue'] == 0``) are left out, the dense-graph
    memory check runs on the spot count that is left (it needs nothing else), and the coordinates are
    read. The refusal on a Visium HD or Xenium slide (~9 TB of graph) used to come only after both
    matrices had been densified and written out as text -- hours, and tens of GB, later.

    coordinates.csv is written in the file's own units; ``_spatial_graph_in_spot_pitch`` puts it
    into STdGCN's unit afterwards, as a separate and reported step.
    """
    import pandas as pd
    import scanpy as sc

    # --- The spatial slide first: every refusal it can cause comes before any staging ---
    log("Loading spatial AnnData...")
    adata_st = sc.read_h5ad(spatial_h5ad)
    # Genes only, as before; the count is published rather than the rename happening in silence.
    renamed_st = make_names_unique_and_report(adata_st, axes=("var",))
    log(f"Spatial: {adata_st.n_obs} spots x {adata_st.n_vars} genes")
    adata_st, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata_st)
    if n_spots_off_tissue:
        log(
            f"Left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0 (background); "
            f"{adata_st.n_obs} in-tissue spots remain"
        )
    n_spots = int(adata_st.n_obs)
    n_pseudo_spots = _n_pseudo_spots(n_spots)
    memory = _check_dense_graph_memory(n_spots, n_pseudo_spots)

    if "spatial" in adata_st.obsm:
        coords, _ = spatial_coords(adata_st, "spatial", want=2, tool="STdGCN")
        coord_df = pd.DataFrame(
            coords,
            index=adata_st.obs_names,
            columns=["x", "y"],
        )
    else:
        # STdGCN is a graph convolution over the spatial adjacency graph built from this very file,
        # so fabricating y = x in barcode order does not degrade the result, it replaces it: the
        # "spatial" neighbours become barcode neighbours. Refuse, as for a missing cell-type column.
        raise ValueError(
            f"Spot coordinates not found: adata.obsm['spatial'] is missing. "
            f"Available obsm keys: {list(adata_st.obsm.keys())}"
        )

    # --- The reference ---
    log("Loading single-cell reference AnnData...")
    adata_sc = sc.read_h5ad(sc_h5ad)
    log(f"Reference: {adata_sc.n_obs} cells x {adata_sc.n_vars} genes")

    if cell_type_key not in adata_sc.obs.columns:
        raise ValueError(
            f"Cell type key '{cell_type_key}' not found in sc_h5ad.obs. Available columns: {list(adata_sc.obs.columns)}"
        )

    # A missing label is not a class. sorted() below used to die on it with a TypeError comparing
    # float and str, naming neither the column nor the count; had it got past, STdGCN would have
    # read the blank label back as a cell type of its own.
    try:
        keep, n_unlabeled = _split_unlabeled(
            adata_sc.obs[cell_type_key].astype(object).values, bool(drop_unlabeled), what="reference cells"
        )
    except ValueError as exc:
        raise ValueError(f"cell_type_key='{cell_type_key}': {exc}") from exc
    if n_unlabeled:
        log(f"Dropping {n_unlabeled} reference cells with no label in obs['{cell_type_key}'] (drop_unlabeled=True)")
        adata_sc = adata_sc[keep].copy()

    cell_types = sorted(adata_sc.obs[cell_type_key].unique().tolist())
    # STdGCN reads sc_label.tsv back and takes .unique() on the label column, so every labelled
    # output it produces is in first-appearance order (STdGCN.py:43 -> :45-46 -> utils.py:201 ->
    # the CSV header at :309). The same rows are staged below, so that order is recomputable here
    # rather than guessable: .unique() is first-appearance for object and categorical dtype alike.
    cell_types_library_order = adata_sc.obs[cell_type_key].unique().tolist()
    log(f"Found {len(cell_types)} cell types")

    # --- Filter to common genes before writing TSVs ---
    # STdGCN internally filters to common genes after pseudo-spot generation,
    # but pseudo-spot generation on 36K genes is the bottleneck. Pre-filtering
    # reduces TSV sizes and speeds up pseudo-spot generation dramatically.
    n_genes_sc_orig = adata_sc.n_vars
    n_genes_st_orig = adata_st.n_vars
    common_genes = sorted(set(adata_sc.var_names) & set(adata_st.var_names))
    if len(common_genes) < 100:
        raise ValueError(
            f"Only {len(common_genes)} common genes between sc and ST data. Check that gene naming conventions match."
        )
    log(f"Filtering to {len(common_genes)} common genes (sc={n_genes_sc_orig}, ST={n_genes_st_orig})")
    adata_sc = adata_sc[:, common_genes].copy()
    adata_st = adata_st[:, common_genes].copy()

    # Create data directories
    sc_path = os.path.join(output_dir, "sc_data")
    st_path = os.path.join(output_dir, "ST_data")
    os.makedirs(sc_path, exist_ok=True)
    os.makedirs(st_path, exist_ok=True)

    # --- Write sc_data.tsv (cells x genes), a block of cells at a time ---
    sc_data_path = os.path.join(sc_path, "sc_data.tsv")
    log(f"Writing sc_data.tsv ({adata_sc.n_obs}, {adata_sc.n_vars})...")
    _write_obs_by_features_tsv(adata_sc.X, adata_sc.obs_names, adata_sc.var_names, sc_data_path)

    # Write sc_label.tsv (barcode, label)
    sc_label = adata_sc.obs[[cell_type_key]].copy()
    sc_label.columns = ["label"]
    sc_label_path = os.path.join(sc_path, "sc_label.tsv")
    _atomic_to_csv(sc_label, sc_label_path, sep="\t")
    log(f"Written sc_label.tsv ({len(sc_label)} cells)")

    # --- Write ST_data.tsv (spots x genes), a block of spots at a time ---
    st_data_path = os.path.join(st_path, "ST_data.tsv")
    log(f"Writing ST_data.tsv ({adata_st.n_obs}, {adata_st.n_vars})...")
    _write_obs_by_features_tsv(adata_st.X, adata_st.obs_names, adata_st.var_names, st_data_path)

    # Write coordinates.csv (barcode, x, y)
    coord_path = os.path.join(st_path, "coordinates.csv")
    _atomic_to_csv(coord_df, coord_path)
    log(f"Written coordinates.csv ({len(coord_df)} spots)")

    return {
        "sc_path": sc_path,
        "ST_path": st_path,
        "cell_types": cell_types,
        "cell_types_library_order": cell_types_library_order,
        "n_spots": n_spots,
        "n_spots_supplied": int(n_spots_supplied),
        "n_spots_off_tissue_dropped": int(n_spots_off_tissue),
        "n_pseudo_spots": n_pseudo_spots,
        "memory": memory,
        "n_genes_st": n_genes_st_orig,
        "n_sc_cells": int(adata_sc.n_obs),
        "n_genes_sc": n_genes_sc_orig,
        "n_genes_common": len(common_genes),
        "n_sc_cells_dropped_unlabeled": int(n_unlabeled),
        "spot_names": list(adata_st.obs_names),
        "renamed_st": renamed_st,
    }


def _atomic_to_csv(frame, path: str, **kwargs) -> None:
    """Write ``frame`` to ``path`` through a ``.partial`` sibling, so a reader never sees half a file."""
    partial = path + ".partial"
    frame.to_csv(partial, **kwargs)
    os.replace(partial, path)


def _spatial_graph_in_spot_pitch(st_path: str) -> dict:
    """Rewrite ``coordinates.csv`` in STdGCN's unit (one spot pitch) and count the graph it will build.

    The file's own coordinates are kept beside it as ``coordinates_input_units.csv``. The pitch is
    the median distance from each spot to its nearest neighbour. The edge count applies
    ``intra_dist_adj``'s own rule -- 27 nearest neighbours, linked when closer than the threshold --
    to the rescaled coordinates, so it is the number of spatial links STdGCN will have, measured
    before hours of training rather than inferred afterwards.
    """
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors

    coord_path = os.path.join(st_path, "coordinates.csv")
    coords = pd.read_csv(coord_path, index_col=0)
    xy = coords[["x", "y"]].to_numpy(dtype=np.float64)
    n = int(len(xy))
    if n < 2:
        raise ValueError(f"STdGCN needs at least two spots to build a spatial graph; {coord_path} has {n}.")
    if not np.isfinite(xy).all():
        bad = int((~np.isfinite(xy)).any(axis=1).sum())
        first = str(coords.index[np.flatnonzero(~np.isfinite(xy).all(axis=1))[0]])
        raise ValueError(
            f"{bad} of {n} spots have a missing or non-finite coordinate in obsm['spatial'] (first: '{first}'); "
            "STdGCN cannot place them in its spatial graph."
        )

    k = min(SPACE_DIST_NEIGHBORS, n - 1)
    nearest = NearestNeighbors(n_neighbors=1).fit(xy).kneighbors()[0][:, 0]
    positive = nearest[nearest > 0]
    if positive.size == 0:
        raise ValueError(
            f"All {n} spots in obsm['spatial'] sit at the same position, so there is no spot spacing to "
            "build STdGCN's spatial graph from."
        )
    pitch = float(np.median(positive))

    # The edges are counted on exactly the coordinates STdGCN will read, with its own search
    # (NearestNeighbors, 27 neighbours, minkowski), so the count is the graph it builds.
    scaled_xy = xy / pitch
    dist, ind = NearestNeighbors(n_neighbors=k, metric="minkowski").fit(scaled_xy).kneighbors()
    linked = dist < SPACE_DIST_THRESHOLD
    coincident = linked & (dist == 0)
    if coincident.any():
        i = int(np.flatnonzero(coincident.any(axis=1))[0])
        j = int(ind[i][np.flatnonzero(coincident[i])[0]])
        n_coincident = int(coincident.sum())
        raise ValueError(
            f"{n_coincident} neighbour pair(s) in obsm['spatial'] share one position (first: '{coords.index[i]}' "
            f"and '{coords.index[j]}'). STdGCN's '{SPATIAL_LINK_METHOD}' spatial link weighs a pair by "
            "1/distance, so a pair at distance 0 gets an infinite weight and every prediction comes back NaN. "
            "Give each spot its own position."
        )
    rows = np.repeat(np.arange(n), k)[linked.ravel()]
    cols = ind.ravel()[linked.ravel()]
    pairs = np.unique(np.stack([np.minimum(rows, cols), np.maximum(rows, cols)], axis=1), axis=0)
    n_edges = int(len(pairs))
    if n_edges == 0:
        raise ValueError(
            f"No two of the {n} spots are within {SPACE_DIST_THRESHOLD:g} spot pitches of each other, so "
            "STdGCN's spatial graph would be empty and its spatial branch the identity."
        )
    degree = np.bincount(pairs.ravel(), minlength=n)

    input_units = os.path.join(st_path, "coordinates_input_units.csv")
    _atomic_to_csv(coords, input_units)
    in_pitch = coords.copy()
    in_pitch[["x", "y"]] = scaled_xy
    _atomic_to_csv(in_pitch, coord_path)
    log(
        f"Spatial graph: spot pitch {pitch:.6g} in the input's units; coordinates.csv rescaled to pitch 1; "
        f"{n_edges} spatial edges at threshold {SPACE_DIST_THRESHOLD:g} (mean {degree.mean():.2f} neighbours, "
        f"{int((degree == 0).sum())} spots with none)"
    )
    return {
        "spot_pitch": pitch,
        "n_spatial_edges": n_edges,
        "mean_spatial_neighbours": float(degree.mean()),
        "n_spots_without_spatial_neighbour": int((degree == 0).sum()),
        "input_units_csv": input_units,
    }


def _available_memory_bytes():
    """Memory this process can still allocate: worker_utils.available_memory_bytes().

    The smaller of MemAvailable and the room under the cgroup memory limit (page cache counted as
    reclaimable). Reading MemAvailable alone let a run inside a memory-limited container pass the
    check and meet the OOM killer instead. Kept under this name so the check has one seam.
    """
    return available_memory_bytes()


def _check_dense_graph_memory(n_spots: int, n_pseudo_spots: int) -> dict:
    """Refuse, with the numbers, a run whose dense graph matrices cannot fit in memory."""
    n_nodes = int(n_spots) + int(n_pseudo_spots)
    per_matrix = 8 * n_nodes * n_nodes
    needed = DENSE_GRAPH_MATRICES_AT_ONCE * per_matrix
    available = _available_memory_bytes()
    gib = float(1 << 30)
    if available is not None and needed > available:
        raise MemoryError(
            f"STdGCN builds its graphs as dense float64 matrices over every spot and pseudo-spot: "
            f"{n_spots} spots + {n_pseudo_spots} pseudo-spots = {n_nodes} nodes, {per_matrix / gib:.1f} GiB per "
            f"matrix, and it holds at least {DENSE_GRAPH_MATRICES_AT_ONCE} of them at once "
            f"({needed / gib:.1f} GiB). This machine has {available / gib:.1f} GiB available (MemAvailable, or "
            "the room left under the cgroup memory limit). The dense graph is how STdGCN works; run it on a "
            "machine with more memory."
        )
    return {"n_graph_nodes": n_nodes, "dense_graph_bytes_lower_bound": int(needed)}


def _restore_input_coordinates(result_adata, input_units_csv: str) -> bool:
    """Put the input's own coordinates back into the published ``obs['coor_X'/'coor_Y']``.

    STdGCN copies coordinates.csv into those two columns; they are in spot pitches only because the
    graph needed that unit. The published file keeps the unit the caller supplied.
    """
    import pandas as pd

    obs = getattr(result_adata, "obs", None)
    if obs is None or not {"coor_X", "coor_Y"} <= set(obs.columns):
        return False
    original = pd.read_csv(input_units_csv, index_col=0)
    # A barcode that looks like a number is read back as one; compare as text on both sides.
    original.index = original.index.astype(str)
    original = original.reindex([str(name) for name in result_adata.obs_names])
    if original[["x", "y"]].isna().any().any():
        return False
    result_adata.obs["coor_X"] = original["x"].to_numpy()
    result_adata.obs["coor_Y"] = original["y"].to_numpy()
    return True


def _h5ad_safe_obs_columns(result_adata) -> dict:
    """Give every obs column a name an .h5ad file can hold; return ``{old: new}`` for the renamed ones.

    STdGCN adds one obs column per reference cell type, named with the label itself (its ground-truth
    slots, STdGCN.py:122-127). HDF5 reads '/' in a key as a group separator, so with the benchmark's
    own reference -- which has both 'Treg' and 'Treg/Tfr' -- write_h5ad died with "Incompatible
    object (Dataset) already exists" after training had finished and the proportion table had been
    written, and the run reported an error. A '/' label with no such neighbour was written as a nested
    group rather than a column, and a label that is not a string (a numeric cluster id, read back from
    sc_label.tsv as a number) is refused by anndata outright.

    The rename applies to the h5ad's obs only: '/' becomes '_', the spelling
    ``sanitize_cell_type_names`` gives every deconvolution worker, and a name already taken gets a
    trailing '_'. The proportion CSVs keep the reference's own labels. The caller reports the mapping.
    """
    obs = getattr(result_adata, "obs", None)
    if obs is None:
        return {}
    taken = {c for c in obs.columns if isinstance(c, str) and "/" not in c}
    renamed = {}
    columns = []
    for col in obs.columns:
        if isinstance(col, str) and "/" not in col:
            columns.append(col)
            continue
        new = sanitize_cell_type_names([str(col)], replace_space=False)[0][0]
        while new in taken:
            new += "_"
        taken.add(new)
        renamed[str(col)] = new
        columns.append(new)
    if renamed:
        obs.columns = columns
    return renamed


def _write_h5ad_atomically(adata, path: str) -> None:
    """``adata.write_h5ad(path)`` through a ``.partial`` sibling, so a failed write leaves no torn file."""
    partial = path + ".partial"
    try:
        adata.write_h5ad(partial)
        os.replace(partial, path)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise


def _extract_predicted_proportions(result_adata, cell_types_library_order, output_dir):
    """Recover STdGCN's spot x cell-type prediction from whatever it returned."""
    import numpy as np
    import pandas as pd

    if result_adata is not None and hasattr(result_adata, "obs"):
        # STdGCN publishes its prediction in predict_result.csv and obsm['predict_result'].
        # It does NOT put it in obs: the per-cell-type obs columns are ground-truth slots,
        # filled only when run_STdGCN(load_test_groundtruth=True) -- which this worker never
        # sets. Reading them yielded an all-zero table that was written out as the result
        # (and idxmax then assigned every spot to the first cell type). Had they ever been
        # filled it would have published the answer key as the prediction instead.
        published_csv = os.path.join(output_dir, "predict_result.csv")
        obsm = getattr(result_adata, "obsm", {}) or {}
        # Ground-truth slots carry no signal, but their order is STdGCN's own cell-type
        # order -- which is how the unlabelled obsm array gets its column names.
        slot_cols = [c for c in result_adata.obs.columns if c in cell_types_library_order]
        proportions_df = None

        if os.path.exists(published_csv):
            candidate = pd.read_csv(published_csv, index_col=0)
            # Guard against a stale file from an earlier run in a reused output dir.
            if set(candidate.index) == set(result_adata.obs_names):
                proportions_df = candidate.reindex(result_adata.obs_names)
                log(f"Loaded STdGCN prediction from {published_csv}: {proportions_df.shape}")
            else:
                log(f"Ignoring {published_csv}: index does not match the returned result")

        if proportions_df is None and "predict_result" in obsm:
            values = np.asarray(obsm["predict_result"])
            # The array is unlabelled. STdGCN labels its CSV with
            # pseudo_adata_norm.obs.columns[:-2], which is the reference's own first-appearance
            # order, never sorted -- pairing the array with a sorted list would rename every cell
            # type to whichever one sorts into its position, leaving the shape, the row sums and
            # the names all plausible and the attribution wrong.
            if len(slot_cols) == values.shape[1]:
                columns = slot_cols
            elif len(cell_types_library_order) == values.shape[1]:
                log("No cell-type slots in obs; labelling the prediction with the reference's own cell-type order")
                columns = list(cell_types_library_order)
            else:
                raise RuntimeError(
                    f"Cannot label obsm['predict_result'] with {values.shape[1]} columns: "
                    f"{len(slot_cols)} obs slots, {len(cell_types_library_order)} cell types"
                )
            proportions_df = pd.DataFrame(values, index=result_adata.obs_names, columns=columns)
            log(f"Extracted proportions from obsm['predict_result']: {proportions_df.shape}")

        if proportions_df is None and "proportions" in obsm:
            proportions_df = pd.DataFrame(
                result_adata.obsm["proportions"],
                index=result_adata.obs_names,
            )
            log(f"Extracted proportions from obsm: {proportions_df.shape}")

        if proportions_df is None:
            # Save the full result and try to find proportions
            result_h5ad = os.path.join(output_dir, "stdgcn_raw_result.h5ad")
            _h5ad_safe_obs_columns(result_adata)
            _write_h5ad_atomically(result_adata, result_h5ad)
            log(f"Saved raw result to {result_h5ad}")
            log(f"Result obs columns: {list(result_adata.obs.columns)}")
            log(f"Result obsm keys: {list(obsm.keys())}")
            raise RuntimeError(
                f"Could not find cell-type proportions in STdGCN result. obs columns: {list(result_adata.obs.columns)}"
            )
    elif isinstance(result_adata, pd.DataFrame):
        proportions_df = result_adata
    else:
        # Check if STdGCN wrote output files directly
        pred_file = os.path.join(output_dir, "STdGCN_predictions.csv")
        if os.path.exists(pred_file):
            proportions_df = pd.read_csv(pred_file, index_col=0)
            log(f"Loaded predictions from {pred_file}")
        else:
            # Fallback: scan output directory for any CSV containing proportions
            import glob

            proportions_df = None
            proportion_patterns = [
                "*predict*",
                "*proportion*",
                "*result*",
                "*deconv*",
                "*weight*",
            ]
            for pattern in proportion_patterns:
                matches = glob.glob(os.path.join(output_dir, "**", pattern), recursive=True)
                csv_matches = [m for m in matches if m.endswith(".csv") or m.endswith(".tsv")]
                if csv_matches:
                    # The suffix is a name, not a fact about the bytes -- this scan reaches whatever
                    # the STdGCN release happened to write, and a ``.csv`` holding tabs parses into
                    # zero columns here. The suffix stays as the answer for a header with nothing to
                    # split on, which is where the old rule was right.
                    sep = sniff_tabular_sep(csv_matches[0], default="\t" if csv_matches[0].endswith(".tsv") else ",")
                    proportions_df = pd.read_csv(csv_matches[0], index_col=0, sep=sep)
                    log(f"Found proportions via fallback scan: {csv_matches[0]}")
                    break
            if proportions_df is None:
                raise RuntimeError(f"STdGCN did not return a recognizable result. Return type: {type(result_adata)}")

    return proportions_df


def run_stdgcn_deconvolution(
    spatial_h5ad: str,
    sc_h5ad: str,
    output_dir: str,
    cell_type_key: str,
    n_epochs: int,
    device: str,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Core STdGCN deconvolution pipeline.
    """
    import matplotlib

    matplotlib.use("Agg")

    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    # STdGCN compares this word with a case-sensitive '==' (see stdgcn_device_token), so normalise
    # before anything else and report what the model will actually run on, not what was asked for.
    device_token = stdgcn_device_token(device)

    log(f"spatial_h5ad  = {spatial_h5ad}")
    log(f"sc_h5ad       = {sc_h5ad}")
    log(f"output_dir    = {output_dir}")
    log(f"cell_type_key = {cell_type_key}")
    log(f"n_epochs      = {n_epochs}")
    log(f"device        = {device} -> {device_token}")
    log(f"drop_unlabeled = {bool(drop_unlabeled)}")

    # Preflight checks
    preflight_check(
        inputs={"spatial_h5ad": spatial_h5ad, "sc_h5ad": sc_h5ad},
        output_dir=output_dir,
    )

    # Convert h5ad to STdGCN-compatible file format
    log("Converting h5ad files to STdGCN format...")
    data_info = _h5ad_to_stdgcn_format(spatial_h5ad, sc_h5ad, cell_type_key, output_dir, drop_unlabeled=drop_unlabeled)

    cell_types = data_info["cell_types"]
    # STdGCN's own column order, for labelling the unlabelled prediction array it returns.
    cell_types_library_order = data_info["cell_types_library_order"]
    n_celltypes = len(cell_types)
    n_spots = data_info["n_spots"]
    n_sc_cells = data_info["n_sc_cells"]

    # STdGCN's spatial graph is sized in spot pitches; the file's own units left it empty.
    graph = _spatial_graph_in_spot_pitch(data_info["ST_path"])

    # Set up STdGCN paths and parameters
    paths = {
        "sc_path": data_info["sc_path"],
        "ST_path": data_info["ST_path"],
        "output_path": output_dir,
    }

    find_marker_genes_paras = {
        "preprocess": True,
        "normalize": True,
        "log": True,
        "highly_variable_genes": False,
        "highly_variable_gene_num": None,
        "regress_out": False,
        "PCA_components": 30,
        "marker_gene_method": "logreg",
        "top_gene_per_type": 100,
        "filter_wilcoxon_marker_genes": True,
        "pvals_adj_threshold": 0.10,
        "log_fold_change_threshold": 1,
        "min_within_group_fraction_threshold": None,
        "max_between_group_fraction_threshold": None,
    }

    # Checked in _h5ad_to_stdgcn_format, before anything was staged.
    n_pseudo_spots = data_info["n_pseudo_spots"]
    memory = data_info["memory"]

    pseudo_spot_simulation_paras = {
        "spot_num": n_pseudo_spots,
        "min_cell_num_in_spot": 8,
        "max_cell_num_in_spot": 12,
        "generation_method": "celltype",
        "max_cell_types_in_spot": min(4, n_celltypes),
    }

    data_normalization_paras = {
        "normalize": True,
        "log": True,
        "scale": False,
    }

    integration_for_adj_paras = {
        "batch_removal_method": None,
        "dim": 30,
        "dimensionality_reduction_method": "PCA",
        "scale": True,
    }

    inter_exp_adj_paras = {
        "find_neighbor_method": "MNN",
        "dist_method": "cosine",
        "corr_dist_neighbors": 20,
    }

    # In spot pitches: coordinates.csv was rescaled by _spatial_graph_in_spot_pitch above.
    spatial_adj_paras = {
        "link_method": SPATIAL_LINK_METHOD,
        "space_dist_threshold": SPACE_DIST_THRESHOLD,
    }

    real_intra_exp_adj_paras = {
        "find_neighbor_method": "MNN",
        "dist_method": "cosine",
        "corr_dist_neighbors": 10,
        "PCA_dimensionality_reduction": False,
        "dim": 50,
    }

    pseudo_intra_exp_adj_paras = {
        "find_neighbor_method": "MNN",
        "dist_method": "cosine",
        "corr_dist_neighbors": 20,
        "PCA_dimensionality_reduction": False,
        "dim": 50,
    }

    integration_for_feature_paras = {
        "batch_removal_method": None,
        "dimensionality_reduction_method": None,
        "dim": 80,
        "scale": True,
    }

    GCN_paras = {
        "epoch_n": n_epochs,
        "dim": 80,
        "common_hid_layers_num": 1,
        "fcnn_hid_layers_num": 1,
        "dropout": 0,
        "learning_rate_SGD": 2e-1,
        "weight_decay_SGD": 3e-4,
        "momentum": 0.9,
        "dampening": 0,
        "nesterov": True,
        "early_stopping_patience": 20,
        "clip_grad_max_norm": 1,
        "print_loss_epoch_step": max(1, n_epochs // 10),
    }

    # Patch tqdm to avoid notebook widget errors in CLI environment
    import tqdm
    import tqdm.notebook

    tqdm.notebook.tqdm = tqdm.tqdm  # force notebook tqdm to use console tqdm

    # Patch pandas DataFrame.append (removed in pandas 2.0, used by STdGCN)
    if not hasattr(pd.DataFrame, "append"):

        def _df_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
            return pd.concat([self, other], ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort)

        pd.DataFrame.append = _df_append
        log("Patched pandas DataFrame.append for STdGCN compatibility")

    # Import and run STdGCN
    log("Importing STdGCN...")
    from STdGCN.STdGCN import run_STdGCN

    log(f"Running STdGCN deconvolution (epochs={n_epochs}, device={device_token})...")
    from threadpoolctl import threadpool_limits

    log(f"numpy BLAS limited to {STDGCN_BLAS_THREADS} thread(s) while STdGCN runs (see STDGCN_BLAS_THREADS)")
    with threadpool_limits(limits=STDGCN_BLAS_THREADS, user_api="blas"):
        result_adata = run_STdGCN(
            paths,
            find_marker_genes_paras=find_marker_genes_paras,
            pseudo_spot_simulation_paras=pseudo_spot_simulation_paras,
            data_normalization_paras=data_normalization_paras,
            integration_for_adj_paras=integration_for_adj_paras,
            inter_exp_adj_paras=inter_exp_adj_paras,
            spatial_adj_paras=spatial_adj_paras,
            real_intra_exp_adj_paras=real_intra_exp_adj_paras,
            pseudo_intra_exp_adj_paras=pseudo_intra_exp_adj_paras,
            integration_for_feature_paras=integration_for_feature_paras,
            GCN_paras=GCN_paras,
            load_test_groundtruth=False,
            use_marker_genes=True,
            external_genes=False,
            generate_new_pseudo_spots=True,
            fraction_pie_plot=False,
            cell_type_distribution_plot=False,
            # NOT -1: STdGCN reads that as multiprocessing.cpu_count() and forks one process per
            # *machine* core, each holding a copy of the single-cell AnnData. Inside a cgroup quota or
            # under an affinity mask the machine count is not this process's allowance, so a container
            # given four CPUs still forks one process per physical core and thrashes or OOMs.
            # Uncapped: on an unconstrained host cpu_budget() equals the cpu_count() that -1 already
            # meant, so this only ever removes over-forking -- it never adds parallelism.
            n_jobs=cpu_budget(),
            GCN_device=stdgcn_device_token(device),
        )
    log("STdGCN completed.")

    # Extract proportions from result
    proportions_df = _extract_predicted_proportions(result_adata, cell_types_library_order, output_dir)

    # Save proportions CSV
    proportions_csv = os.path.join(output_dir, "stdgcn_proportions.csv")
    log(f"Writing cell-type proportions to {proportions_csv}")
    _atomic_to_csv(proportions_df, proportions_csv, index_label="spot")

    # Compute dominant cell type per spot
    dominant_ct = proportions_df.idxmax(axis=1)
    dominant_csv = os.path.join(output_dir, "stdgcn_dominant_celltype.csv")
    _atomic_to_csv(dominant_ct.to_frame(name="dominant_celltype"), dominant_csv, index_label="spot")

    from collections import Counter

    dominant_counts = dict(Counter(dominant_ct.values))

    # Save annotated h5ad
    annotated_h5ad = os.path.join(output_dir, "stdgcn_deconvolution.h5ad")
    obs_renamed = {}
    if result_adata is not None and hasattr(result_adata, "write_h5ad"):
        if not _restore_input_coordinates(result_adata, graph["input_units_csv"]):
            log("obs['coor_X'/'coor_Y'] left as STdGCN wrote them (spot-pitch units)")
        obs_renamed = _h5ad_safe_obs_columns(result_adata)
        if obs_renamed:
            log(f"Renamed {len(obs_renamed)} obs column(s) an .h5ad cannot hold: {obs_renamed}")
        log(f"Writing annotated AnnData to {annotated_h5ad}")
        _write_h5ad_atomically(result_adata, annotated_h5ad)
    else:
        annotated_h5ad = None

    # Build output
    out = WorkerOutput("stdgcn", task="deconvolution")
    out.set_data(
        n_spots=n_spots,
        n_genes_st=data_info["n_genes_st"],
        n_reference_cells=n_sc_cells,
        n_reference_genes=data_info["n_genes_sc"],
    )
    output_files = {
        "proportions_csv": proportions_csv,
        "dominant_celltype_csv": dominant_csv,
        "output_dir": output_dir,
    }
    if annotated_h5ad:
        output_files["annotated_h5ad"] = annotated_h5ad
    out.add_output_files(output_files)
    record_method(out, "STdGCN")
    n_dropped = int(data_info.get("n_sc_cells_dropped_unlabeled", 0))
    out.add_params(
        {
            "cell_type_key": cell_type_key,
            "n_epochs": n_epochs,
            "device": device_token,
            "device_requested": device,
            "drop_unlabeled": bool(drop_unlabeled),
            "n_reference_cells_dropped_unlabeled": n_dropped,
            # The spatial graph as STdGCN built it. Coordinates come from obsm['spatial'] and are
            # divided by spot_pitch, so the threshold is in spot pitches, as in STdGCN's tutorial.
            "coordinate_source": "obsm['spatial']",
            "coordinate_unit": "spot pitch (obsm['spatial'] / spot_pitch)",
            "spot_pitch": graph["spot_pitch"],
            "space_dist_threshold": SPACE_DIST_THRESHOLD,
            "space_dist_neighbors": SPACE_DIST_NEIGHBORS,
            "spatial_link_method": SPATIAL_LINK_METHOD,
            "n_pseudo_spots": n_pseudo_spots,
            "numpy_blas_threads": STDGCN_BLAS_THREADS,
            # obs columns of the annotated h5ad renamed because HDF5 cannot hold the label as a key
            # ('/' is its group separator); the proportion CSVs keep the reference's own labels.
            "annotated_h5ad_obs_columns_renamed": obs_renamed,
        }
    )
    out.add_params(identifier_rename_params(data_info.get("renamed_st")))
    record_in_tissue(out, data_info["n_spots_supplied"], data_info["n_spots_off_tissue_dropped"])
    if obs_renamed:
        first_old, first_new = next(iter(obs_renamed.items()))
        out.add_warning(
            f"{len(obs_renamed)} obs column(s) of stdgcn_deconvolution.h5ad were renamed so HDF5 can hold them "
            f"({first_old!r} is {first_new!r} there); stdgcn_proportions.csv keeps the reference's labels."
        )
    out.set_summary(
        n_cell_types=n_celltypes,
        cell_type_names=cell_types,
        dominant_counts=dominant_counts,
        n_spatial_edges=graph["n_spatial_edges"],
        mean_spatial_neighbours=round(graph["mean_spatial_neighbours"], 3),
        n_spots_without_spatial_neighbour=graph["n_spots_without_spatial_neighbour"],
        n_graph_nodes=memory["n_graph_nodes"],
    )
    analysis = build_deconv_analysis(
        n_celltypes=n_celltypes,
        dominant_counts=dominant_counts,
        total_spots=n_spots,
        method_name="STdGCN",
    )
    analysis += (
        f" Spatial graph: {graph['n_spatial_edges']} edges between spots within {SPACE_DIST_THRESHOLD:g} spot "
        f"pitches (pitch {graph['spot_pitch']:.4g} in obsm['spatial'] units; mean "
        f"{graph['mean_spatial_neighbours']:.1f} neighbours per spot)."
    )
    if n_dropped:
        analysis += f" {n_dropped} reference cells with no '{cell_type_key}' label were left out (drop_unlabeled=True)."
    if data_info["n_spots_off_tissue_dropped"]:
        analysis += (
            f" {data_info['n_spots_off_tissue_dropped']} of {data_info['n_spots_supplied']} spots were background "
            "(obs['in_tissue'] == 0) and were left out."
        )
    analysis += identifier_rename_note(data_info.get("renamed_st"), subject="spatial data")
    out.set_analysis(analysis)

    return out.to_dict()


def main() -> None:
    args = parse_args()

    # Redirect all stdout during heavy work to stderr
    orig_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        try:
            result = run_stdgcn_deconvolution(
                spatial_h5ad=args.spatial_h5ad,
                sc_h5ad=args.sc_h5ad,
                output_dir=args.output_dir,
                cell_type_key=args.cell_type_key,
                n_epochs=args.n_epochs,
                device=args.device,
                drop_unlabeled=bool(args.drop_unlabeled),
            )
        except Exception as e:
            log("ERROR while running STdGCN:")
            traceback.print_exc(file=sys.stderr)
            result = WorkerOutput.error("stdgcn", str(e), task="deconvolution")
    finally:
        sys.stdout = orig_stdout

    # Final JSON to stdout
    print(json.dumps(result))


if __name__ == "__main__":
    main()
