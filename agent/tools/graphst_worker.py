#!/usr/bin/env python

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

# ⚠️关键：完全按官方教程的方式导入
# from GraphST import GraphST，然后用 GraphST.GraphST(...)
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from GraphST import GraphST
from GraphST import utils as _graphst_utils
from GraphST.preprocess import filter_with_overlap_gene
from GraphST.utils import clustering
from sklearn import metrics
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    build_cluster_analysis,
    build_deconv_analysis,
    choose_counts_matrix,
    describe_reduction,
    ensure_r_home,
    expression_matrix_kind,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    resolve_compute,
    unsupported_choice_msg,
)

# Under a private name: run_deconvolution takes a ``drop_unlabeled`` parameter, which shadowed the
# helper and turned the call on the reference labels into a call on a bool.
from worker_utils import drop_unlabeled as _drop_unlabeled

#: GraphST.preprocess ranks this many seurat_v3 HVGs on each side; deconvolution trains on the
#: intersection of the two sets. It is not a parameter of upstream ``preprocess`` and so not one here.
HVG_N_TOP_GENES = 3000


def _mclust_R_fixed(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2020):
    """
    Drop-in replacement for GraphST.utils.mclust_R that avoids the rpy2
    numpy2ri dimnames bug with mclust >= 6.  Runs Mclust entirely inside
    R via robjects.r() so no numpy-to-R automatic conversion interferes.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects

    robjects.r.library("mclust")
    robjects.r["set.seed"](random_seed)

    # Pass embedding to R as a plain matrix (column-major)
    emb = adata.obsm[used_obsm]
    robjects.globalenv["bmni.emb"] = robjects.r["matrix"](
        robjects.FloatVector(emb.T.flatten()),
        nrow=emb.shape[0],
        ncol=emb.shape[1],
    )
    robjects.globalenv["bmni.G"] = robjects.IntVector([int(num_cluster)])
    robjects.globalenv["bmni.mn"] = robjects.StrVector([modelNames])

    robjects.r("bmni.res <- Mclust(bmni.emb, G=bmni.G, modelNames=bmni.mn)")
    mclust_res = np.array(robjects.r("bmni.res$classification"))

    # Cast to string before categorical so scanpy's rank_genes_groups /
    # dendrogram (which call ",".join(categories)) do not TypeError on ints.
    adata.obs["mclust"] = pd.Categorical(mclust_res.astype(int).astype(str))
    return adata


# Monkey-patch GraphST.utils.mclust_R so that clustering() uses our fixed version
_graphst_utils.mclust_R = _mclust_R_fixed


def _get_device(device_str: str) -> torch.device:
    """Resolve any of the fleet's device spellings to a device that exists on this box.

    Shared with every other worker (worker_utils.resolve_compute): 'GPU' no longer raises
    RuntimeError inside torch.device(), and a CUDA request on a CPU-only box degrades instead
    of dying at model construction.
    """
    return torch.device(resolve_compute(device_str).device)


def _require_cluster_backend(cluster_tool: str) -> None:
    """Refuse, before anything trains, a clustering this environment cannot run.

    GraphST's ``clustering(method='louvain')`` calls ``sc.tl.louvain`` (flavor 'vtraag'), which imports
    Traag's ``louvain`` package; the GraphST env does not ship it, so a 'louvain' run trained the whole
    model and then failed (hunt 2026-09-30, u38b-specs-a-9). Nothing is substituted: leiden is a
    different algorithm. Mirrors spiral_worker._require_cluster_backend.
    """
    if cluster_tool != "louvain":
        return
    try:
        import louvain  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "cluster_tool='louvain' runs scanpy's sc.tl.louvain (flavor 'vtraag'), which needs the 'louvain' "
            f"package, and it cannot be imported in this environment ({type(exc).__name__}: {exc}). Install "
            "louvain into the GraphST env, or pass cluster_tool='leiden' or 'mclust'."
        ) from exc


def _set_r_home_if_needed(cluster_tool: str, r_home_arg: str | None = None) -> None:
    """
    For mclust, R_HOME must be set.
    Priority: CLI arg > GRAPHST_R_HOME env > auto-detect from conda prefix > R RHOME.

    Shared with the other R-calling workers (worker_utils.ensure_r_home). The order is the one
    this function always had; what the shared version adds is an ``isdir`` check on the CLI arg
    and the env var. Both used to be taken verbatim, so a ``GRAPHST_R_HOME`` left pointing at the
    build host's env -- which is exactly what a captured ``conda_clone`` spec carries, see
    ``sog_install/mcp_resolver._rebase_source_env_extras`` -- shadowed the correct ``sys.prefix/lib/R``
    below it and hard-failed mclust on a tool that would otherwise have worked.

    The raise stays here rather than moving into the shared helper: it is right for graphst, which
    has a flag and a variable to name in the message, and wrong for a worker with neither, where
    leaving R_HOME unset lets rpy2's own detection have its turn.
    """
    if cluster_tool != "mclust":
        return
    if os.environ.get("R_HOME"):
        print(f"[graphst-worker] Using existing R_HOME={os.environ['R_HOME']}", file=sys.stderr)
        return
    r_home = ensure_r_home(explicit=r_home_arg, env_var="GRAPHST_R_HOME")
    if r_home:
        print(f"[graphst-worker] Using R_HOME={r_home}", file=sys.stderr)
    else:
        raise RuntimeError(
            "cluster_tool='mclust' requires R but R_HOME is not set and could not be "
            "auto-detected. Set GRAPHST_R_HOME env var or pass --r-home."
        )


# --------------------------------------------------------------------------- memory

#: Spot-by-spot float64 matrices alive at GraphST's peak, per task. Clustering: ``construct_interaction``
#: keeps the distance matrix, the kNN indicator and the symmetrised adjacency in ``obsm`` (3), and
#: ``GraphST.__init__`` then builds ``graph_neigh.copy() + np.eye`` and ``normalize_adj(...).toarray()
#: + np.eye`` beside them (3 transient). Deconvolution: the worker builds those three matrices itself,
#: the overlap-gene view is made actual by ``get_feature`` and ``GraphST.__init__`` copies it again (3
#: each), and the contrastive loss of ``train_map`` keeps several spot-by-spot float32 tensors in the
#: autograd graph. Measured 2026-09-29 as peak RSS above baseline on a synthetic 6,000-spot slide
#: (epochs=1): 6.4 (clustering) and 16.7 (deconvolution) n^2 float64. The values below sit under
#: those measurements on purpose -- see ``_dense_budget_bytes``.
_DENSE_SQUARE_MATRICES = {"clustering": 6, "deconvolution": 14}

#: Dense float32 copies of the n_cells x n_spots mapping matrix alive at the peak of ``train_map``
#: (deconvolution only): the ``Encoder_map`` Parameter ``M``, its gradient, Adam's two moment buffers,
#: the per-epoch ``softmax(M)`` kept for backward and its gradient, then the final softmax ``.numpy()``.
#: Measured 2026-09-30 as peak RSS above baseline around ``GraphST.GraphST(..., deconvolution=True)
#: .train_map()`` (GraphST env, CPU, epochs=3): 7.98 copies at 1,000 spots x 60,000 cells and 9.92 at
#: 2,000 x 30,000 (the second carrying ~2 copies' worth of spot-by-spot matrices). The value sits under
#: those measurements on purpose, like ``_DENSE_SQUARE_MATRICES``. It is the dominant term for a large
#: reference: 279,609 cells x 6,487 spots is 6.8 GiB per copy.
_MAPPING_MATRIX_COPIES = 6

#: Dense float32 copies of the reference's cells x overlap-genes matrix ``GraphST.__init__`` makes for
#: ``feat_sc`` (``X.toarray()``, the ``fillna`` frame and the torch tensor).
_REFERENCE_FEATURE_COPIES = 3


def _memory_available_bytes():
    """Memory this process can still allocate, or None when the platform cannot say.

    The fleet's one reader (``worker_utils.available_memory_bytes``): the smaller of MemAvailable and
    the room under the cgroup limit, page cache counted as reclaimable. MemAvailable alone is the
    whole host's figure, so in a memory-limited container this let through a slide the cgroup then
    OOM-killed mid-training, with no message. Kept under this name so the budget check reads it here.
    """
    return available_memory_bytes()


def _dense_budget_bytes(n_spots, n_features, task="clustering", n_cells=0, n_features_sc=0):
    """A lower bound on the bytes of the dense intermediates GraphST materialises for ``n_spots``.

    Intrinsic to the method, not to this wrapper: upstream builds its spot graph as dense
    ``n_spots x n_spots`` numpy arrays (``construct_interaction``, ``preprocess_adj``) and the 10X
    code path trains on them densely. The count per task is ``_DENSE_SQUARE_MATRICES``; the feature
    matrix and its permuted copy (numpy and torch) are added. For deconvolution with ``n_cells``
    reference cells, ``train_map`` also optimises a dense float32 ``n_cells x n_spots`` mapping
    matrix (``_MAPPING_MATRIX_COPIES`` of it alive at once) beside the dense reference features
    (``_REFERENCE_FEATURE_COPIES`` of ``n_cells x n_features_sc``) -- for a large reference that is
    most of the peak, and leaving it out let a run the machine could not hold pass the check and be
    OOM-killed in ``train_map`` with no payload. It is deliberately not an upper bound: a slide this
    refuses cannot run, and one it lets through may still be tight.
    """
    n = int(n_spots)
    c = int(n_cells or 0)
    spots = n * n * 8 * _DENSE_SQUARE_MATRICES[task] + n * int(n_features) * 4 * 4
    mapping = c * n * 4 * _MAPPING_MATRIX_COPIES + c * int(n_features_sc or 0) * 4 * _REFERENCE_FEATURE_COPIES
    return spots + mapping


def _check_dense_budget(n_spots, n_features, what="the slide", task="clustering", n_cells=0, n_features_sc=0):
    """Refuse up front, with the numbers, a slide whose dense matrices cannot fit in memory.

    Without this the run dies inside ``ot.dist`` or ``np.zeros`` with a bare MemoryError, or is
    OOM-killed mid-training with no message at all, after the reference has been read and
    preprocessed. With ``n_cells`` (deconvolution, once the reference is read and its unlabelled
    cells dropped) the n_cells x n_spots mapping matrix ``train_map`` optimises is counted too. No
    parameter of this tool lowers the footprint, and neither the slide nor the reference is ever cut
    down here: the message says what is needed and what the machine has.
    """
    available = _memory_available_bytes()
    if available is None:
        return None
    need = _dense_budget_bytes(n_spots, n_features, task=task, n_cells=n_cells, n_features_sc=n_features_sc)
    if need > available:
        gib = 1024.0**3
        spots_part = _dense_budget_bytes(n_spots, n_features, task=task)
        mapping_note = ""
        if n_cells:
            mapping_note = (
                f" and a dense {int(n_cells)}x{n_spots} float32 cell-to-spot mapping matrix that train_map optimises "
                f"(the parameter, its gradient, two Adam moments and the per-epoch softmax: about "
                f"{_MAPPING_MATRIX_COPIES} copies) beside the dense {int(n_cells)}x{int(n_features_sc)} reference "
                f"features -- {(need - spots_part) / gib:.1f} GiB for the {int(n_cells)} reference cells on top of "
                f"{spots_part / gib:.1f} GiB for the spots"
            )
        raise MemoryError(
            f"GraphST {task} builds dense {n_spots}x{n_spots} spot-by-spot matrices (distance, kNN graph, "
            f"normalised adjacency){mapping_note} for {what}: at least {need / gib:.1f} GiB, and this "
            f"machine reports {available / gib:.1f} GiB available. No parameter of this tool lowers that "
            "footprint; run it on a machine with more memory. The slide and the reference are analysed whole."
        )
    return need


# --------------------------------------------------------------------------- io


def _write_csv_atomic(df, path) -> None:
    """``<path>.partial`` then ``os.replace``: a reader never sees a half-written table."""
    tmp = f"{path}.partial"
    df.to_csv(tmp)
    os.replace(tmp, path)


def _write_h5ad_atomic(adata, path) -> None:
    tmp = f"{path}.partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- projection


def _validate_retain_percent(retain_percent) -> float:
    """``retain_percent`` is the fraction of reference cells kept per spot; only (0, 1] means anything.

    At 0 every cell is masked and every spot comes back all-zero; above 1 the mask keeps everything
    and the value has no effect. Both would be reported in ``params`` as if applied.
    """
    value = float(retain_percent)
    if not (0.0 < value <= 1.0):
        raise ValueError(
            f"retain_percent={retain_percent!r} must be in (0, 1]: it is the fraction of reference cells "
            "kept per spot when the mapping matrix is projected onto cell types (0.15 keeps the top 15%)."
        )
    return value


def _cells_retained_per_spot(n_cells, retain_percent) -> int:
    """How many reference cells the top-fraction mask keeps in every spot.

    Upstream keeps rank ``r`` (0-based, ascending) when ``r >= n_cells - retain_percent * n_cells``;
    for integer ranks that is ``n_cells - ceil(n_cells - top_k)`` cells, computed with the same
    floating-point arithmetic the mask uses so the two can never disagree.
    """
    n_cells = int(n_cells)
    top_k = float(retain_percent) * n_cells
    return max(0, min(n_cells, n_cells - int(math.ceil(n_cells - top_k))))


def _top_fraction_mask(map_matrix, retain_percent):
    """Upstream ``extract_top_value`` with ``retain_percent`` actually applied: per spot, keep the
    ``retain_percent * n_cells`` highest-probability cells (rank >= n_cells - top_k, ranks from a
    double argsort exactly as upstream computes them), zero the rest. Same dtype as the input.
    """
    map_matrix = np.asarray(map_matrix)
    n_cells = map_matrix.shape[1]
    top_k = float(retain_percent) * n_cells
    ranks = np.argsort(np.argsort(map_matrix, axis=1), axis=1)
    return map_matrix * (ranks >= n_cells - top_k)


def _project_cells_to_spots(map_matrix, cell_labels, retain_percent, spot_names, chunk_rows=2048):
    """Spot x cell-type abundance table from GraphST's spot x cell mapping matrix.

    The math of upstream ``GraphST.utils.project_cell_to_spot`` -- top-fraction mask, one-hot
    matmul, row normalisation -- with two differences that are the point of doing it here:
    ``retain_percent`` reaches the mask (upstream accepts the argument and then calls
    ``extract_top_value(map_matrix)`` with its own default of 0.1), and the result is returned as a
    table instead of being written straight into ``obs``, where a cell type named like an existing
    column overwrote it and then vanished from the CSV.

    Columns are every label of the reference as ``str``, sorted (upstream's order), whether or not
    any spot retained a cell of that type; a spot that retained no cell at all is all-zero, as
    upstream's ``fillna(0)`` left it. Row-chunked: the double argsort holds two int64 arrays the
    shape of its block, and the masked matrix is never materialised whole.
    """
    retain_percent = _validate_retain_percent(retain_percent)
    map_matrix = np.asarray(map_matrix)
    labels = [str(x) for x in np.asarray(cell_labels, dtype=object)]
    if map_matrix.ndim != 2 or map_matrix.shape[1] != len(labels):
        raise ValueError(
            f"map_matrix is {map_matrix.shape} (spots x cells) but the reference has {len(labels)} labelled cells"
        )
    cell_types = sorted(set(labels))
    onehot = pd.get_dummies(pd.Categorical(labels, categories=cell_types)).to_numpy(dtype=np.float64)
    projection = np.zeros((map_matrix.shape[0], len(cell_types)), dtype=np.float64)
    step = max(1, int(chunk_rows))
    for start in range(0, map_matrix.shape[0], step):
        block = map_matrix[start : start + step]
        projection[start : start + step] = _top_fraction_mask(block, retain_percent).dot(onehot)
    index = spot_names if isinstance(spot_names, pd.Index) else pd.Index(list(spot_names))
    df = pd.DataFrame(projection, index=index, columns=cell_types)
    return df.div(df.sum(axis=1), axis=0).fillna(0)


def _h5ad_obs_keys(cell_types, existing_columns):
    """The obs key under which each cell type's abundance is stored in the output h5ad.

    The name itself, unless HDF5 cannot hold it beside the other columns. h5py reads ``/`` in a key as
    a path, so ``'Treg/Tfr'`` needs a group called ``'Treg'``; when ``'Treg'`` is also a column -- on
    the Tonsil reference it is another cell type -- anndata fails the whole write with "Incompatible
    object (Dataset) already exists", after training and before the CSV. Such a name (a ``/`` path
    whose prefix is, or which is itself the prefix of, another column, or one with an empty segment)
    is stored with ``/`` replaced by ``_``, suffixed until unique. A ``/`` name with no such clash is
    kept, as it always wrote. The CSV and the payload keep every original name.
    """
    names = [str(c) for c in cell_types]
    taken = set(names) | {str(c) for c in existing_columns}

    def clashes(name):
        if "/" not in name:
            return False
        parts = name.split("/")
        if any(part == "" for part in parts):
            return True
        if {"/".join(parts[:i]) for i in range(1, len(parts))} & taken:
            return True
        return any(other.startswith(name + "/") for other in taken)

    keys = {}
    for name in names:
        if not clashes(name):
            keys[name] = name
            continue
        base = name.replace("/", "_")
        key, i = base, 2
        while key in taken:
            key, i = f"{base}_{i}", i + 1
        taken.add(key)
        keys[name] = key
    return keys


def _keep_in_tissue_spots(adata):
    """``(adata, n_spots_supplied, n_spots_off_tissue)``: background spots left out before GraphST.

    ``obs['in_tissue'] == 0`` marks array spots outside the tissue. CELLxGENE Visium exports carry
    them (56-70% of the spots on the library's four such samples, and not empty), and GraphST built
    its graph over them, trained on them and gave them domains or abundances -- using up the
    requested n_clusters on glass and paying n^2 memory for it. The shared rule leaves them out and
    counts them.
    """
    adata, n_supplied, n_off = keep_in_tissue(adata, "spots")
    if n_off:
        print(
            f"[graphst-worker] Left out {n_off} of {n_supplied} spots with obs['in_tissue'] == 0 (background); "
            f"{adata.n_obs} in-tissue spots are analysed.",
            file=sys.stderr,
        )
    return adata, int(n_supplied), int(n_off)


def _in_tissue_note(n_supplied, n_off) -> str:
    """The analysis sentence for spots the in-tissue filter left out ('' when none were)."""
    return describe_reduction(
        "spots",
        int(n_supplied),
        int(n_supplied) - int(n_off),
        "the in-tissue filter (obs['in_tissue'] == 0 marks background outside the tissue)",
    )


def _choose_counts(adata, use_raw_counts, what, normalised_by_graphst=True):
    """``(adata, info)``: the matrix ``GraphST.preprocess`` will normalise as counts, checked first.

    ``GraphST.preprocess`` runs seurat_v3 HVG selection, ``normalize_total`` and ``log1p`` on X as
    stored. A scaled X (negative values) became NaN and died inside skmisc's loess ("Extrapolation not
    allowed with blending"), and a log-normalised X was normalised a second time with status ok. The
    shared rule (``worker_utils.choose_counts_matrix``): a negative or non-finite X is refused, naming
    ``use_raw_counts`` when ``adata.raw`` holds counts; a non-negative non-integer X runs with a
    warning; ``use_raw_counts=True`` runs on ``adata.raw.X``. ``what`` names the input in a refusal.

    ``normalised_by_graphst=False`` is the clustering run on an input that already flags
    ``var['highly_variable']``: GraphST then skips its own preprocessing and trains on X as stored, so
    a normalised or scaled X is what that path accepts by design and is not refused; its kind is
    still reported.
    """
    if not use_raw_counts and not normalised_by_graphst:
        return adata, {"expression_source": "X", "x_matrix_kind": expression_matrix_kind(adata.X), "warning": None}
    try:
        return choose_counts_matrix(adata, use_raw_counts)
    except ValueError as exc:
        raise ValueError(f"{what}: {exc}") from exc


def _unlabeled_mask(labels):
    """Spots/cells whose label is missing (NaN/None/''/'nan'), by the fleet-wide definition."""
    keep, n_missing = _drop_unlabeled(labels, allow_drop=True)
    return np.asarray(keep, dtype=bool), int(n_missing)


# --------------------------------------------------------------------------- clustering


def run_spatial_clustering(
    st_h5ad: str,
    output_dir: str,
    n_clusters: int,
    cluster_tool: str,
    radius: int,
    device: str,
    label_key: str | None,
    r_home: str | None,
    use_raw_counts: bool = False,
) -> dict:
    """
    对单个 ST h5ad 做 GraphST 空间聚类（spatial domain）。

    GraphST's own preprocessing normalises X as counts, so the matrix is chosen and checked first
    (:func:`_choose_counts`): X by default, ``adata.raw.X`` with ``use_raw_counts=True``.
    """
    print("[graphst-worker] Task = clustering", file=sys.stderr)
    print(f"[graphst-worker] ST h5ad = {st_h5ad}", file=sys.stderr)
    print(f"[graphst-worker] output_dir = {output_dir}", file=sys.stderr)

    if cluster_tool not in ("mclust", "leiden", "louvain"):
        raise ValueError(unsupported_choice_msg("cluster_tool", cluster_tool, ["mclust", "leiden", "louvain"]))
    if int(n_clusters) < 1:
        raise ValueError(f"n_clusters={n_clusters!r} must be at least 1.")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    dev = _get_device(device)
    print(f"[graphst-worker] device = {dev}", file=sys.stderr)

    _set_r_home_if_needed(cluster_tool, r_home)
    _require_cluster_backend(cluster_tool)

    domain_csv = None
    ari_value = None
    ari_n_scored = None
    ari_n_unlabeled = None
    refinement_applied = cluster_tool == "mclust"
    refinement_note = ""
    gt_labels = None
    label_key_missing = False
    label_key_overwritten = False

    # GraphST / scanpy 的所有正常输出都重定向到 stderr
    with contextlib.redirect_stdout(sys.stderr):
        # 读取已经预处理好的 AnnData（你现在就是这个情况）
        adata = sc.read_h5ad(st_h5ad)
        adata, n_spots_supplied, n_off_tissue = _keep_in_tissue_spots(adata)
        # GraphST normalises X as counts unless the input already flags var['highly_variable'] (then it
        # trains on X as stored). use_raw_counts swaps in adata.raw.X, whose var decides that anew.
        adata, counts_info = _choose_counts(
            adata, use_raw_counts, "st_h5ad", normalised_by_graphst="highly_variable" not in adata.var
        )
        # Duplicate gene symbols are made unique, as before, and now counted in the payload.
        renamed = make_names_unique_and_report(adata, axes=("var",))

        # Ensure spatial coordinates are float64 for POT compatibility
        if "spatial" in adata.obsm:
            adata.obsm["spatial"] = np.array(adata.obsm["spatial"], dtype=np.float64)

        print(
            f"[graphst-worker] Loaded ST data: n_spots={adata.n_obs}, n_genes={adata.n_vars}, "
            f"expression=adata.{counts_info['expression_source']} ({counts_info['x_matrix_kind']})",
            file=sys.stderr,
        )
        n_spots = int(adata.n_obs)
        if cluster_tool == "mclust" and not (1 <= int(radius) < n_spots):
            # refine_label takes the ``radius`` nearest spots of every spot; 0 leaves it nothing to vote
            # on (max() of an empty list) and n_spots or more runs off the end -- after training.
            raise ValueError(
                f"radius={radius!r} must be between 1 and n_spots-1 ({n_spots - 1}): it is the number of "
                "nearest spots that vote on each spot's domain in the mclust refinement step."
            )
        _check_dense_budget(n_spots, min(int(adata.n_vars), HVG_N_TOP_GENES), what=Path(st_h5ad).name)

        # The ground truth is read NOW, before GraphST writes 'domain' (and 'mclust'/'leiden'/'louvain')
        # into obs: a label_key naming one of those columns used to be scored against the predictions
        # themselves, ARI 1.0.
        if label_key:
            if label_key in adata.obs:
                gt_labels = np.asarray(adata.obs[label_key].to_numpy(), dtype=object).copy()
                label_key_overwritten = label_key in ("domain", "mclust", cluster_tool)
            else:
                label_key_missing = True
                print(
                    f"[graphst-worker] WARNING: label_key='{label_key}' is not an obs column; ARI not computed",
                    file=sys.stderr,
                )

        # GraphST.__init__ runs its own preprocess (seurat_v3 top-3000 HVGs, normalize_total, log1p,
        # scale) only when var has no 'highly_variable'. An input that already carries that column is
        # trained on X as stored, restricted to the genes it flags -- which the payload has to say.
        input_hvg = "highly_variable" in adata.var
        n_input_hvg = int(np.asarray(adata.var["highly_variable"], dtype=bool).sum()) if input_hvg else None

        # === 完全按 Tutorial 1 的写法来 ===
        # from GraphST import GraphST
        # model = GraphST.GraphST(adata, device=device)
        # adata = model.train()
        model = GraphST.GraphST(adata, device=dev)
        adata = model.train()

        print(
            f"[graphst-worker] Clustering with tool={cluster_tool}, n_clusters={n_clusters}, radius={radius}",
            file=sys.stderr,
        )

        # Tutorial 1 的聚类代码：
        #   from GraphST.utils import clustering
        #   clustering(adata, n_clusters, radius=..., method=..., ...)
        if cluster_tool == "mclust":
            clustering(
                adata,
                n_clusters,
                radius=radius,
                method="mclust",
                refinement=True,
            )
        else:
            # Upstream applies ``radius`` only inside ``refine_label``, which runs when
            # refinement=True; the leiden/louvain path never smooths, so the value has no effect.
            try:
                clustering(
                    adata,
                    n_clusters,
                    radius=radius,
                    method=cluster_tool,
                    start=0.1,
                    end=2.0,
                    increment=0.01,
                    refinement=False,
                )
            except AssertionError as exc:
                # upstream search_res: "Resolution is not found. Please try bigger range or smaller step!"
                # -- neither is a parameter of this tool.
                raise RuntimeError(
                    f"cluster_tool='{cluster_tool}' found no resolution in [0.1, 2.0) (step 0.01) that gives "
                    f"exactly n_clusters={n_clusters} clusters on the GraphST embedding. That search range is "
                    "fixed in this tool; cluster_tool='mclust' fits exactly n_clusters components."
                ) from exc

        # Sanity check: mclust with G=K should always produce K components, but
        # the spatial refinement step can absorb tiny clusters into neighbors,
        # collapsing the visible K. If that happens, publish the raw
        # pre-refinement mclust labels so downstream evaluation sees the
        # requested K -- and SAY SO in params.method and the warnings, because the
        # smoothing the caller asked for (radius) did not make it into the output.
        post_k = int(adata.obs["domain"].nunique()) if "domain" in adata.obs else None
        pre_k = int(adata.obs["mclust"].nunique()) if "mclust" in adata.obs else None
        print(
            f"[graphst-worker] requested_K={n_clusters} mclust_K={pre_k} domain_K_after_refinement={post_k}",
            file=sys.stderr,
        )
        if (
            cluster_tool == "mclust"
            and post_k is not None
            and pre_k is not None
            and post_k < n_clusters
            and pre_k == n_clusters
        ):
            refinement_note = (
                f"spatial refinement (radius={radius}) collapsed {pre_k}->{post_k} domains (requested {n_clusters}); "
                "the pre-refinement mclust labels are published instead, so no spatial smoothing was applied"
            )
            print(f"[graphst-worker] WARNING: {refinement_note}", file=sys.stderr)
            adata.obs["domain"] = adata.obs["mclust"].astype(str)
            refinement_applied = False
        elif cluster_tool == "mclust" and pre_k is not None and pre_k != n_clusters:
            print(
                f"[graphst-worker] WARNING: requested K={n_clusters} but mclust "
                f"produced K={pre_k}. This usually means n_clusters was passed "
                f"incorrectly upstream; mclust with G=K should yield exactly K "
                f"components.",
                file=sys.stderr,
            )

        # 可选：如果用户在 obs[label_key] 里给了 GT，就算 ARI。A missing label (NaN, '', 'nan')
        # is not a class: those spots are left out of the score and their count is reported.
        if gt_labels is not None and "domain" in adata.obs:
            mask, ari_n_unlabeled = _unlabeled_mask(gt_labels)
            ari_n_scored = int(mask.sum())
            if ari_n_scored > 0:
                ari_value = float(
                    metrics.adjusted_rand_score(
                        np.asarray(adata.obs["domain"].values, dtype=object)[mask].astype(str),
                        gt_labels[mask].astype(str),
                    )
                )
                adata.uns[f"ARI_{label_key}"] = ari_value
                print(
                    f"[graphst-worker] ARI ({label_key}) = {ari_value:.4f} on {ari_n_scored} labelled spots "
                    f"({ari_n_unlabeled} unlabelled left out)",
                    file=sys.stderr,
                )

        # 保存结果
        out_h5ad = output_dir_path / "graphst_clustering_output.h5ad"
        _write_h5ad_atomic(adata, out_h5ad)
        print(f"[graphst-worker] Saved h5ad to {out_h5ad}", file=sys.stderr)

        if "domain" in adata.obs:
            domain_csv = output_dir_path / "graphst_domain.csv"
            _write_csv_atomic(adata.obs[["domain"]], domain_csv)
            print(
                f"[graphst-worker] Saved domain CSV to {domain_csv}",
                file=sys.stderr,
            )

    # Build cluster summary
    cluster_sizes: dict = {}
    if "domain" in adata.obs:
        cluster_sizes = adata.obs["domain"].value_counts().to_dict()

    n_hvg = int(adata.var["highly_variable"].sum()) if "highly_variable" in adata.var else None
    if input_hvg:
        hvg_source = "input var['highly_variable'] (GraphST preprocessing skipped)"
    else:
        hvg_source = f"GraphST.preprocess (seurat_v3, top {HVG_N_TOP_GENES})"

    out = WorkerOutput("graphst", task="clustering")
    out.set_data(
        n_spots=n_spots_supplied,
        n_spots_used=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        n_hvg=n_hvg,
        hvg_source=hvg_source,
    )
    record_in_tissue(out, n_spots_supplied, n_off_tissue)
    out.add_output_files(
        {
            "output_h5ad": str(out_h5ad),
            "domain_csv": str(domain_csv) if domain_csv is not None else None,
        }
    )
    out.add_params(
        {
            "n_clusters": int(n_clusters),
            "cluster_tool": cluster_tool,
            "radius": int(radius),
            "device": str(dev),
            "refinement_applied": bool(refinement_applied),
            "use_raw_counts": bool(use_raw_counts),
            **identifier_rename_params(renamed),
        }
    )
    record_expression_source(out, counts_info)
    if cluster_tool == "mclust":
        if refinement_applied:
            method = f"GraphST embedding + mclust with spatial refinement (radius={int(radius)})"
        else:
            method = "GraphST embedding + mclust, pre-refinement labels (refinement collapsed the domain count)"
            out.add_warning(refinement_note)
    else:
        method = (
            f"GraphST embedding + {cluster_tool} (resolution searched for {int(n_clusters)} domains; no refinement)"
        )
        record_ignored(
            out,
            "radius",
            f"GraphST applies radius only in the spatial refinement step, which runs for cluster_tool='mclust'; "
            f"cluster_tool='{cluster_tool}' publishes the unrefined labels",
        )
    record_method(out, method, used_fallback=False)
    out.set_summary(
        n_clusters=len(cluster_sizes),
        cluster_sizes=cluster_sizes,
    )
    if ari_value is not None:
        out.set_summary(
            ARI=ari_value,
            ground_truth_key=label_key,
            ari_n_spots_scored=ari_n_scored,
            ari_n_unlabeled_excluded=ari_n_unlabeled,
        )
    elif label_key and ari_n_scored == 0:
        out.add_warning(f"label_key='{label_key}' has no labelled spots (all NaN/empty); ARI not computed")
    if label_key_missing:
        record_ignored(
            out,
            "label_key",
            f"label_key='{label_key}' is not an obs column of the input (columns: {list(adata.obs.columns)[:20]}); "
            "ARI not computed",
        )
    if label_key_overwritten:
        out.add_warning(
            f"label_key='{label_key}' is also a column GraphST writes: ARI was scored against the input's values, "
            f"read before clustering, but obs['{label_key}'] in the output h5ad now holds GraphST's labels"
        )
    if input_hvg:
        out.add_warning(
            f"the input already had var['highly_variable'] ({n_input_hvg} genes): GraphST trained on those genes "
            "with X as stored and skipped its own seurat_v3 HVG selection, normalize_total, log1p and scale"
        )
    analysis = build_cluster_analysis(
        cluster_sizes,
        cluster_key="domain",
        total_spots=int(adata.n_obs),
        n_requested=int(n_clusters),
    )
    if input_hvg:
        analysis += (
            f" Embedding trained on the {n_input_hvg} genes the input flagged in var['highly_variable'] (of "
            f"{int(adata.n_vars)}), on X as stored: GraphST skips its own normalisation when that column exists."
        )
    elif n_hvg is not None:
        analysis += f" Embedding trained on the top {n_hvg} seurat_v3 highly variable genes of {int(adata.n_vars)}."
    if counts_info["expression_source"] == "raw.X":
        analysis += " The expression matrix was adata.raw.X (use_raw_counts=True)."
    if counts_info.get("warning"):
        analysis += " " + counts_info["warning"]
    if refinement_note:
        analysis += " " + refinement_note[0].upper() + refinement_note[1:] + "."
    analysis += _in_tissue_note(n_spots_supplied, n_off_tissue)
    analysis += identifier_rename_note(renamed)
    out.set_analysis(analysis)

    return out.to_dict()


# --------------------------------------------------------------------------- deconvolution


def run_deconvolution(
    st_h5ad: str,
    scrna_h5ad: str,
    output_dir: str,
    celltype_key: str,
    epochs: int,
    retain_percent: float,
    device: str,
    drop_unlabeled: bool = False,
    use_raw_counts: bool = False,
) -> dict:
    """
    按 Tutorial 2 实现的 deconvolution（scRNA+ST）。

    The mapping matrix comes from upstream ``train_map``; its projection onto cell types is done
    here (``_project_cells_to_spots``) so that ``retain_percent`` is the fraction that actually
    runs and so that no cell-type column can silently overwrite a spatial ``obs`` column.

    ``GraphST.preprocess`` normalises both inputs as counts, so each matrix is chosen and checked
    first (:func:`_choose_counts`). ``use_raw_counts=True`` reads ``adata.raw.X`` of each input that
    has an ``adata.raw``; an input without one is read from X (checked the same way) with a warning,
    and a run where neither input has one is refused.

    Memory is checked twice: on the spots alone before the reference is read, and again once the
    reference is read and its unlabelled cells dropped, with the dense n_cells x n_spots mapping
    matrix counted (:func:`_dense_budget_bytes`).
    """
    print("[graphst-worker] Task = deconvolution", file=sys.stderr)
    print(f"[graphst-worker] ST h5ad = {st_h5ad}", file=sys.stderr)
    print(f"[graphst-worker] scRNA h5ad = {scrna_h5ad}", file=sys.stderr)
    print(f"[graphst-worker] output_dir = {output_dir}", file=sys.stderr)

    retain_percent = _validate_retain_percent(retain_percent)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    dev = _get_device(device)
    print(f"[graphst-worker] device = {dev}", file=sys.stderr)

    celltype_csv = None
    allow_drop = bool(drop_unlabeled)

    with contextlib.redirect_stdout(sys.stderr):
        # === Reading ST data ===
        adata = sc.read_h5ad(st_h5ad)
        adata, n_spots_supplied, n_off_tissue = _keep_in_tissue_spots(adata)
        st_has_raw = getattr(adata, "raw", None) is not None
        adata, st_counts_info = _choose_counts(adata, bool(use_raw_counts) and st_has_raw, "st_h5ad")
        renamed = make_names_unique_and_report(adata, axes=("var",))

        # Ensure spatial coordinates are float64 for POT compatibility
        if "spatial" in adata.obsm:
            adata.obsm["spatial"] = np.array(adata.obsm["spatial"], dtype=np.float64)

        n_spots = int(adata.n_obs)
        n_genes_supplied = int(adata.n_vars)
        st_obs_columns = [str(c) for c in adata.obs.columns]
        print(
            f"[graphst-worker] Loaded ST data: n_spots={n_spots}, n_genes={n_genes_supplied}",
            file=sys.stderr,
        )
        _check_dense_budget(
            n_spots, min(n_genes_supplied, HVG_N_TOP_GENES), what=Path(st_h5ad).name, task="deconvolution"
        )

        # === Reading reference scRNA data ===
        adata_sc = sc.read(scrna_h5ad)
        sc_has_raw = getattr(adata_sc, "raw", None) is not None
        if use_raw_counts and not (st_has_raw or sc_has_raw):
            raise ValueError(
                "use_raw_counts=True reads adata.raw of each input that has one, and neither st_h5ad nor "
                "scrna_h5ad has an adata.raw. Leave use_raw_counts off to run on their X."
            )
        adata_sc, sc_counts_info = _choose_counts(adata_sc, bool(use_raw_counts) and sc_has_raw, "scrna_h5ad")
        renamed_sc = make_names_unique_and_report(adata_sc, axes=("var",))
        n_cells_supplied = int(adata_sc.n_obs)
        n_genes_sc_supplied = int(adata_sc.n_vars)
        print(
            f"[graphst-worker] Loaded scRNA data: n_cells={n_cells_supplied}, n_genes={n_genes_sc_supplied}, "
            f"expression=adata.{sc_counts_info['expression_source']} ({sc_counts_info['x_matrix_kind']})",
            file=sys.stderr,
        )

        # 确保 celltype 标签在 obs['cell_type']
        # A celltype_key the reference does not have is refused here, BEFORE train_map(): letting it
        # through either deconvolves against whatever 'cell_type' happens to hold (a wrong answer
        # that looks right) or dies on a column the caller never named, once the whole training
        # run has been paid for.
        if celltype_key not in adata_sc.obs:
            raise ValueError(
                f"celltype_key='{celltype_key}' not found in scRNA obs. Available keys: {list(adata_sc.obs.keys())}"
            )

        # A NaN / empty label is not a cell type. ``astype(str)`` used to turn it into a class
        # called 'nan' that then got an abundance column; now it is an error unless the caller
        # asked, by name, for those cells to be left out.
        keep, n_cells_dropped = _drop_unlabeled(
            adata_sc.obs[celltype_key].values, allow_drop=allow_drop, what=f"reference cells (obs['{celltype_key}'])"
        )
        if n_cells_dropped:
            adata_sc = adata_sc[np.asarray(keep, dtype=bool)].copy()
            print(
                f"[graphst-worker] drop_unlabeled=True: left out {n_cells_dropped} reference cells with no label",
                file=sys.stderr,
            )
        cell_labels = [str(x) for x in np.asarray(adata_sc.obs[celltype_key].values, dtype=object)]
        cell_types = sorted(set(cell_labels))
        if len(cell_types) < 2:
            raise ValueError(
                f"obs['{celltype_key}'] holds {len(cell_types)} cell type after dropping unlabelled cells; "
                "deconvolution needs at least 2."
            )
        n_retained_per_spot = _cells_retained_per_spot(len(cell_labels), retain_percent)
        if n_retained_per_spot < 1:
            raise ValueError(
                f"retain_percent={retain_percent:g} of {len(cell_labels)} reference cells keeps no cell in any spot "
                "(the mask keeps floor(retain_percent x n_cells)); every abundance row would be zero. Raise "
                "retain_percent."
            )
        # Now that the reference is read and its unlabelled cells are gone, the whole peak can be
        # estimated: train_map optimises a dense n_cells x n_spots mapping matrix, which for a large
        # reference dwarfs the spot-by-spot matrices checked above. Refused here, before any
        # preprocessing or training, rather than OOM-killed inside train_map with no payload.
        _check_dense_budget(
            n_spots,
            min(n_genes_supplied, HVG_N_TOP_GENES),
            what=f"{Path(st_h5ad).name} with {int(adata_sc.n_obs)} reference cells from {Path(scrna_h5ad).name}",
            task="deconvolution",
            n_cells=int(adata_sc.n_obs),
            n_features_sc=min(int(adata_sc.n_vars), HVG_N_TOP_GENES),
        )

        # The abundance columns are the reference's labels verbatim, and they are written into the
        # spatial obs. Upstream did that with ``obs[columns] = df`` and a label equal to an existing
        # column replaced it, after which the "new columns" scan lost that cell type from the CSV.
        obs_keys = _h5ad_obs_keys(cell_types, st_obs_columns)
        collisions = sorted(ct for ct in cell_types if obs_keys[ct] == ct and ct in st_obs_columns)
        if collisions:
            raise ValueError(
                f"{len(collisions)} cell type name(s) in obs['{celltype_key}'] equal existing spatial obs "
                f"column(s): {collisions}. Rename those spatial columns (or the reference labels) first; "
                "the abundance table is written under the cell-type names and must not overwrite them."
            )

        if celltype_key != "cell_type":
            adata_sc.obs["cell_type"] = pd.Series(cell_labels, index=adata_sc.obs_names)
            print(
                f"[graphst-worker] Copied obs['{celltype_key}'] -> obs['cell_type']",
                file=sys.stderr,
            )

        # === Pre-processing for ST data (Tutorial 2) ===
        # GraphST.preprocess(adata)
        # GraphST.construct_interaction(adata)
        # GraphST.add_contrastive_label(adata)
        GraphST.preprocess(adata)
        GraphST.construct_interaction(adata)
        GraphST.add_contrastive_label(adata)

        # === Pre-processing for reference data ===
        GraphST.preprocess(adata_sc)

        n_hvg_st = int(adata.var["highly_variable"].sum()) if "highly_variable" in adata.var else None
        n_hvg_sc = int(adata_sc.var["highly_variable"].sum()) if "highly_variable" in adata_sc.var else None

        # === Overlap genes ===
        adata, adata_sc = filter_with_overlap_gene(adata, adata_sc)
        n_overlap_genes = int(adata.n_vars)
        print(
            f"[graphst-worker] After overlap filter: ST genes={adata.n_vars}, scRNA genes={adata_sc.n_vars}",
            file=sys.stderr,
        )
        if n_overlap_genes == 0:
            raise ValueError(
                f"No genes shared between the spatial HVGs ({n_hvg_st}) and the reference HVGs ({n_hvg_sc}). "
                "Check that both files use the same gene identifiers (symbols vs Ensembl IDs)."
            )

        # === Extract features for ST data ===
        GraphST.get_feature(adata)

        # === Implementing GraphST for deconvolution (Tutorial 2) ===
        # model = GraphST.GraphST(adata, adata_sc, epochs=1200, random_seed=50, device=device, deconvolution=True)
        # adata, adata_sc = model.train_map()
        model = GraphST.GraphST(
            adata,
            adata_sc,
            epochs=int(epochs),
            random_seed=50,
            device=dev,
            deconvolution=True,
        )
        adata, adata_sc = model.train_map()

        # === Project cells into spatial space ===
        # Tutorial 2 calls ``project_cell_to_spot(adata, adata_sc, retain_percent=0.15)``; the installed
        # function accepts the argument and then masks with its own default (0.1), and writes the
        # table into obs by label name. Both are done here instead, with the value the caller chose.
        df_projection = _project_cells_to_spots(
            adata.obsm["map_matrix"],
            adata_sc.obs[celltype_key].values,
            retain_percent,
            adata.obs_names,
        )
        celltype_columns = list(df_projection.columns)

        # 保存 cell-type abundance CSV -- first: it is the result, and it carries every original name.
        celltype_csv = output_dir_path / "graphst_celltype_abundance.csv"
        _write_csv_atomic(df_projection, celltype_csv)
        print(
            f"[graphst-worker] Saved cell-type abundance CSV to {celltype_csv}",
            file=sys.stderr,
        )

        renamed_obs_keys = {ct: obs_keys[ct] for ct in celltype_columns if obs_keys[ct] != ct}
        for col in celltype_columns:
            adata.obs[obs_keys[col]] = df_projection[col].to_numpy()
        adata.uns["graphst_celltypes"] = np.array(celltype_columns, dtype=object)
        adata.uns["graphst_celltype_obs_keys"] = np.array([obs_keys[c] for c in celltype_columns], dtype=object)

        # 保存 h5ad
        out_h5ad = output_dir_path / "graphst_deconvolution_output.h5ad"
        _write_h5ad_atomic(adata, out_h5ad)
        print(f"[graphst-worker] Saved h5ad to {out_h5ad}", file=sys.stderr)

    # Build deconv summary
    n_celltypes = len(celltype_columns)
    has_abundance = df_projection.sum(axis=1) > 0
    n_spots_without_abundance = int((~has_abundance).sum())
    dominant_counts: dict = {}
    if has_abundance.any():
        dominant_counts = df_projection[has_abundance].idxmax(axis=1).value_counts().to_dict()

    out = WorkerOutput("graphst", task="deconvolution")
    out.set_data(
        n_spots=n_spots_supplied,
        n_spots_used=n_spots,
        n_genes=n_genes_supplied,
        n_cells_sc=int(adata_sc.n_obs),
        n_cells_sc_supplied=n_cells_supplied,
        n_cells_dropped_unlabeled=int(n_cells_dropped),
        n_genes_sc=n_genes_sc_supplied,
        n_hvg_st=n_hvg_st,
        n_hvg_sc=n_hvg_sc,
        n_overlap_genes=n_overlap_genes,
        n_cells_retained_per_spot=n_retained_per_spot,
        n_spots_without_abundance=n_spots_without_abundance,
    )
    record_in_tissue(out, n_spots_supplied, n_off_tissue)
    if renamed_obs_keys:
        out.set_data(h5ad_obs_keys_renamed=renamed_obs_keys)
        out.add_warning(
            f"{len(renamed_obs_keys)} cell type name(s) cannot be HDF5 keys beside the other columns ('/' is a path "
            f"separator): in graphst_deconvolution_output.h5ad they are obs columns {renamed_obs_keys} (mapping in "
            "uns['graphst_celltypes'] / uns['graphst_celltype_obs_keys']); graphst_celltype_abundance.csv and "
            "summary.celltypes keep the original names"
        )
    out.add_output_files(
        {
            "output_h5ad": str(out_h5ad),
            "celltype_abundance_csv": str(celltype_csv),
        }
    )
    out.add_params(
        {
            "celltype_key": celltype_key,
            "epochs": int(epochs),
            "retain_percent": float(retain_percent),
            "device": str(dev),
            "drop_unlabeled": allow_drop,
            "use_raw_counts": bool(use_raw_counts),
            "sc_expression_source": sc_counts_info["expression_source"],
            "sc_x_matrix_kind": sc_counts_info["x_matrix_kind"],
            **identifier_rename_params(renamed),
            **identifier_rename_params(renamed_sc, suffix="sc"),
        }
    )
    record_expression_source(out, st_counts_info)
    if sc_counts_info.get("warning"):
        out.add_warning("scRNA reference: " + sc_counts_info["warning"])
    for has_raw, label in ((st_has_raw, "st_h5ad"), (sc_has_raw, "scrna_h5ad")):
        if use_raw_counts and not has_raw:
            out.add_warning(f"use_raw_counts=True, but {label} has no adata.raw, so its X was used.")
    record_method(
        out,
        f"GraphST deconvolution (train_map mapping matrix; per-spot top {retain_percent:g} fraction of reference "
        "cells projected onto cell types in the worker)",
        used_fallback=False,
    )
    if n_cells_dropped:
        out.add_warning(
            f"drop_unlabeled=True: {n_cells_dropped} of {n_cells_supplied} reference cells had no label in "
            f"obs['{celltype_key}'] and were left out"
        )
    if n_spots_without_abundance:
        out.add_warning(
            f"{n_spots_without_abundance} spots retained no reference cell at retain_percent={retain_percent:g} "
            "and have an all-zero abundance row"
        )
    out.set_summary(
        n_celltypes=n_celltypes,
        celltypes=celltype_columns,
        dominant_counts=dominant_counts,
    )
    analysis = build_deconv_analysis(
        n_celltypes,
        dominant_counts,
        total_spots=n_spots,
        method_name="GraphST",
    )
    analysis += (
        f" Trained on the {n_overlap_genes} genes shared by the {n_hvg_st} spatial and {n_hvg_sc} reference"
        f" seurat_v3 HVGs (top {HVG_N_TOP_GENES} per side, of {n_genes_supplied} and {n_genes_sc_supplied}"
        f" genes supplied); per spot the top {retain_percent:g} fraction of the {int(adata_sc.n_obs)} reference"
        f" cells ({n_retained_per_spot} cells) was projected onto cell types."
    )
    raw_inputs = [
        name
        for name, info in (("spatial", st_counts_info), ("reference", sc_counts_info))
        if info["expression_source"] == "raw.X"
    ]
    if raw_inputs:
        analysis += (
            f" Counts were read from adata.raw.X for the {' and '.join(raw_inputs)} input (use_raw_counts=True)."
        )
    if st_counts_info.get("warning"):
        analysis += " Spatial input: " + st_counts_info["warning"]
    if sc_counts_info.get("warning"):
        analysis += " scRNA reference: " + sc_counts_info["warning"]
    analysis += _in_tissue_note(n_spots_supplied, n_off_tissue)
    analysis += identifier_rename_note(renamed, "spatial input") + identifier_rename_note(renamed_sc, "reference")
    out.set_analysis(analysis)
    return out.to_dict()


def main():
    parser = argparse.ArgumentParser(description="GraphST worker for SpatialOmicsLab MCP (CLI mode)")
    parser.add_argument(
        "--task",
        required=True,
        choices=["clustering", "deconvolution"],
        help="Type of GraphST task to run",
    )
    parser.add_argument("--st-h5ad", required=True, help="Path to spatial transcriptomics .h5ad file")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store outputs (e.g. /workspace/work/graphst_xxx)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device string, e.g. 'auto', 'cpu', 'cuda:0'",
    )

    # clustering-specific
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=7,
        help="Number of spatial domains (clustering task only)",
    )
    parser.add_argument(
        "--cluster-tool",
        default="mclust",
        choices=["mclust", "leiden", "louvain"],
        help="Clustering backend for spatial domains (default: mclust)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=50,
        help="Neighbours considered by the spatial refinement step (mclust only; leiden/louvain ignore it)",
    )
    parser.add_argument(
        "--label-key",
        default=None,
        help="Optional obs key for ground truth labels to compute ARI",
    )
    parser.add_argument(
        "--r-home",
        default=None,
        help="Optional R_HOME path (for mclust); otherwise use GRAPHST_R_HOME env",
    )

    # deconvolution-specific
    parser.add_argument(
        "--scrna-h5ad",
        default=None,
        help="Path to scRNA reference data (.h5ad or other scanpy-readable)",
    )
    parser.add_argument(
        "--celltype-key",
        default="cell_type",
        help="obs column in scRNA data that stores cell-type labels",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1200,
        help="Training epochs for deconvolution task",
    )
    parser.add_argument(
        "--retain-percent",
        type=float,
        default=0.15,
        help="Fraction (0, 1] of reference cells kept per spot when the mapping matrix is projected onto cell types",
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        default=False,
        help="Leave out reference cells whose label is missing (NaN/empty) instead of failing on them",
    )
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        default=False,
        help=(
            "Run on adata.raw.X instead of X (clustering: the spatial input, which must have adata.raw; "
            "deconvolution: each input that has an adata.raw). Without it, a negative or non-finite X is refused "
            "and a non-integer X runs with a warning."
        ),
    )

    args = parser.parse_args()

    try:
        if args.task == "clustering":
            result = run_spatial_clustering(
                st_h5ad=args.st_h5ad,
                output_dir=args.output_dir,
                n_clusters=args.n_clusters,
                cluster_tool=args.cluster_tool,
                radius=args.radius,
                device=args.device,
                label_key=args.label_key,
                r_home=args.r_home,
                use_raw_counts=args.use_raw_counts,
            )
        else:
            if not args.scrna_h5ad:
                raise ValueError("--scrna-h5ad is required when --task=deconvolution")
            result = run_deconvolution(
                st_h5ad=args.st_h5ad,
                scrna_h5ad=args.scrna_h5ad,
                output_dir=args.output_dir,
                celltype_key=args.celltype_key,
                epochs=args.epochs,
                retain_percent=args.retain_percent,
                device=args.device,
                drop_unlabeled=args.drop_unlabeled,
                use_raw_counts=args.use_raw_counts,
            )

        # stdout 只输出 JSON
        print(json.dumps(result, default=str))
        sys.stdout.flush()

    except Exception as e:
        print("[graphst-worker] ERROR:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("graphst", str(e), task=args.task)
        sys.exit(1)


if __name__ == "__main__":
    main()
