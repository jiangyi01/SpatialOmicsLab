#!/usr/bin/env python
"""
SPIRAL worker: spatial transcriptomics integration and coordinate alignment.

- Runs inside /opt/conda/envs/spiral
- All logs/progress go to stderr.
- stdout is reserved for a single final JSON line.

Supported tasks:
  - integrate: batch correction of multiple spatial slices with upstream
    ``spiral.main.SPIRAL_integration``, then clustering of the joint embedding.
  - align: the same integration for exactly two slices, then THIS WORKER'S OWN coordinate
    mapping (see ``ALIGN_METHOD``). SPIRAL ships ``spiral.CoordAlignment``; this worker has never
    called it, and the payload says so.

Key implementation notes:
  - SPIRAL expects CSV file inputs, NOT h5ad -- worker converts from h5ad. ``X`` is used as given
    (no normalisation, no gene selection here); SPIRAL min-max scales each spot itself. The dense
    spots x genes table is intrinsic to SPIRAL (it reads CSVs into pandas), so its size is checked
    against the memory this box has before anything is allocated.
  - Must monkey-patch torch.Tensor.cuda() before importing SPIRAL (see --device below)
  - rpy2 + R packages (mclust) available at /opt/conda/envs/spiral/lib/R
  - Clustering: ``mclust`` fits exactly ``n_clusters`` components; ``leiden``/``louvain`` are
    resolution-driven and never see ``n_clusters`` (it is recorded under ``params.ignored``).
    ``louvain`` is scanpy's ``sc.tl.louvain`` (flavor 'vtraag', the ``louvain`` package); where that
    package cannot be imported the run is refused BEFORE any CSV is written or SPIRAL trains -- it
    used to fail only after the whole training. No other clustering is substituted for it.
  - Background spots: a slice whose ``obs['in_tissue']`` marks spots as 0 has them left out as it is
    read (``worker_utils.keep_in_tissue``); ``params.in_tissue_filter`` and a warning count them.
  - Every output file is written to ``<name>.partial`` and renamed into place.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

# ---- Worker utilities: imported first, because the device has to be settled before SPIRAL is ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import (  # noqa: I001
    WorkerOutput,
    available_memory_bytes,
    build_cluster_analysis,
    ensure_r_home,
    graph_batch_size,
    id_mismatch_msg,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    record_ignored,
    record_in_tissue,
    record_method,
    resolve_compute,
    spatial_coords,
)

# ---- CUDA monkey-patch: must happen BEFORE any SPIRAL imports ----
import torch


def _device_request_from_argv(argv):
    """The ``--device`` value as it appears in raw ``argv``, or ``None`` for "follow the hardware".

    SPIRAL's own layers call ``.cuda()`` unconditionally, so the redirect below has to be installed
    before ``from spiral.main import ...`` runs -- which is before ``argparse`` has had a chance to.
    Hence reading the request straight out of ``argv``, and hence handling *both* spellings argparse
    itself accepts: recognising only ``--device cpu`` would make ``--device=cpu`` silently mean
    "whatever hardware is present", which is the failure this argument exists to remove.
    """
    for i, token in enumerate(argv):
        if token == "--device":
            return argv[i + 1] if i + 1 < len(argv) else None
        if token.startswith("--device="):
            return token.split("=", 1)[1]
    return None


DEVICE_REQUEST = _device_request_from_argv(sys.argv[1:])
DEVICE = torch.device(resolve_compute(DEVICE_REQUEST).device)

# Redirect SPIRAL's hardcoded .cuda() calls onto the device that was actually resolved.
# Unconditional, not "only when CUDA is missing": on a box that *has* a GPU, an explicit
# `--device cpu` -- to leave a shared card alone, or to work around an OOM -- is exactly the
# request that used to have nowhere to go, because this branch never ran and .cuda() stayed real.
if DEVICE.type != "cuda":
    torch.Tensor.cuda = lambda self, *a, **kw: self.to(DEVICE)

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import sklearn.neighbors

# ---- SPIRAL imports (after monkey-patch) ----
from spiral.main import SPIRAL_integration
from spiral.utils import layer_map, mclust_R


def log(msg):
    print(f"[spiral-worker] {msg}", file=sys.stderr)


#: What ``integrate`` runs. The integration is upstream SPIRAL's own class, unmodified.
INTEGRATE_METHOD = "SPIRAL_integration (upstream spiral.main)"

#: What ``align`` runs after that integration. SPIRAL's ``spiral.CoordAlignment`` picks the slice
#: with the most clusters as reference, scores expression with squared-Euclidean distance, and
#: places the spots of clusters found in one slice only with a Procrustes fit (R vegan). This
#: worker never called it: it maps slice_1 onto slice_0 one SHARED cluster at a time with POT's
#: ``fused_gromov_wasserstein`` (Euclidean expression cost) and a transport-weighted mean, and has
#: no Procrustes step -- so a spot outside every shared cluster is left unplaced (NaN) rather than
#: presented in a frame it is not in. Named here so every surface can say what actually ran.
ALIGN_METHOD = (
    "SPIRAL_integration (upstream) + this worker's per-shared-cluster fused Gromov-Wasserstein "
    "mapping (POT ot.gromov.fused_gromov_wasserstein, Euclidean costs, transport-weighted mean) of "
    "slice_1 onto slice_0; upstream spiral.CoordAlignment (most-clusters reference, "
    "squared-Euclidean expression cost, Procrustes placement of slice-specific clusters) not run"
)

#: The hidden layers are ``hidden_dim * 16`` wide: the default 32 gives 512, the width of
#: AEdims=[N,[512],32] / GSdims=[512,32] in every upstream SPIRAL demo.
HIDDEN_WIDTH_FACTOR = 16

#: Dimensions of the SPIRAL embedding that encode the batch (``params.znoise_dim``); excluded from
#: clustering and alignment, as upstream does.
ZNOISE_DIM = 4

#: Bytes per value of the spots x genes feature table at SPIRAL's peak. Upstream ``load_data``
#: reads each CSV into float64 pandas (8), ``prepare_data`` min-max scales it into a second float64
#: copy (8), the trainer holds a float32 tensor of it (4), the embedding pass copies it to float64
#: and to a tensor again (12), and this worker decodes a same-sized corrected matrix (4) into a
#: DataFrame. Rounded up to 40.
DENSE_BYTES_PER_VALUE = 40

_GPU_REQUEST_WORDS = ("gpu", "cuda", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Memory: the dense intermediates SPIRAL cannot do without
# ---------------------------------------------------------------------------


def _available_memory_bytes():
    """Memory this worker can still allocate, or None when the platform cannot say.

    The fleet's one reader (``worker_utils.available_memory_bytes``): the smaller of MemAvailable and
    the room under the cgroup limit, with page cache counted as reclaimable -- a private MemAvailable
    parse ignored the container limit.
    """
    return available_memory_bytes()


def _replace_into(path, write):
    """Call ``write(<path>.partial)`` and rename the result onto ``path``: no torn file on a kill."""
    tmp = str(path) + ".partial"
    write(tmp)
    os.replace(tmp, str(path))


def _gib(n_bytes):
    return float(n_bytes) / float(1 << 30)


#: What the "available" figure in a memory refusal is: ``worker_utils.available_memory_bytes``.
_AVAILABLE_MEANS = (
    "the smaller of MemAvailable and the room under the cgroup memory limit, the cgroup's page cache "
    "counted as reclaimable"
)

#: The one kind of spot this worker leaves out (``keep_in_tissue``, as each slice is read).
_BACKGROUND = "background spots (obs['in_tissue'] == 0)"


def _check_feature_budget(n_spots, n_genes, available=None, n_background=0):
    """Refuse, with the numbers, a feature table SPIRAL could not hold. Never subsamples.

    ``n_spots`` is counted after the background was left out; ``n_background`` is how many
    background spots that was, so the refusal can explain a count below the files' ``n_obs``
    (the error payload does not carry the in-tissue warning). Returns ``(needed_bytes, available_bytes_or_None)``.
    """
    need = int(n_spots) * int(n_genes) * DENSE_BYTES_PER_VALUE
    if available is None:
        available = _available_memory_bytes()
    if available is not None and need > available:
        left_out = ""
        if int(n_background or 0) > 0:
            left_out = (
                f"The {int(n_spots)} spots are those left after {int(n_background)} {_BACKGROUND} were "
                "left out as the slices were read. "
            )
        raise MemoryError(
            f"SPIRAL holds its features as a dense spots x genes table (upstream load_data reads the "
            f"CSVs into pandas, min-max scales them and copies them to a tensor): {int(n_spots)} spots x "
            f"{int(n_genes)} shared genes need about {_gib(need):.1f} GiB at peak, but about "
            f"{_gib(available):.1f} GiB is available here ({_AVAILABLE_MEANS}). {left_out}"
            f"Apart from {_BACKGROUND}, this worker drops no spot, and no parameter of this tool "
            "shrinks the gene axis: run it where that much memory is available, or pass slices that "
            "already hold fewer genes (for example the highly variable genes, selected before calling "
            "this tool)."
        )
    return need, available


def _fgw_bytes(n0, n1):
    """Bytes of one cluster's fused Gromov-Wasserstein problem, all dense float64.

    The two intra-slice distance matrices and POT's ``init_matrix`` terms (about 3 x n^2 per
    slice), and the cost, plan, gradient, line-search and constant terms (about 12 x n0 x n1).
    """
    n0, n1 = int(n0), int(n1)
    return 8 * (12 * n0 * n1 + 3 * (n0 * n0 + n1 * n1))


def _check_fgw_budget(block_sizes, available=None):
    """Refuse up front when the largest per-cluster transport problem cannot fit."""
    if not block_sizes:
        return 0, available
    clust, n0, n1 = max(block_sizes, key=lambda t: _fgw_bytes(t[1], t[2]))
    need = _fgw_bytes(n0, n1)
    if available is None:
        available = _available_memory_bytes()
    if available is not None and need > available:
        raise MemoryError(
            f"The alignment solves one dense fused Gromov-Wasserstein problem per shared cluster. "
            f"The largest, cluster {clust!r} ({n0} x {n1} spots), needs about {_gib(need):.1f} GiB, "
            f"but about {_gib(available):.1f} GiB is available here ({_AVAILABLE_MEANS}). More, "
            "smaller clusters shrink every block: raise resolution (cluster_method 'leiden'/'louvain') "
            f"or n_clusters (cluster_method 'mclust'). No tissue spot is dropped to make a block fit; "
            f"the only spots left out are {_BACKGROUND}, as each slice was read."
        )
    return need, available


# ---------------------------------------------------------------------------
# h5ad -> SPIRAL CSV conversion utilities
# ---------------------------------------------------------------------------


def _cal_spatial_net_knn(adata, k_cutoff=6, coords=None):
    """Build the KNN spatial neighbour graph on ``coords``; store it in adata.uns['Spatial_Net'].

    ``coords`` are the 2D coordinates the worker also writes to the slice's coord CSV, so the graph
    SPIRAL trains on and the frame the outputs are reported in are the same one.
    """
    if coords is None:
        coords, _ = spatial_coords(adata, "spatial", want=2, tool="SPIRAL")
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    knn_df = pd.DataFrame(
        {
            "Cell1": np.repeat(np.arange(n), indices.shape[1]),
            "Cell2": indices.ravel(),
            "Distance": distances.ravel(),
        }
    )
    spatial_net = knn_df.loc[knn_df["Distance"] > 0].copy()
    names = np.asarray(adata.obs_names)
    spatial_net["Cell1"] = names[spatial_net["Cell1"].to_numpy()]
    spatial_net["Cell2"] = names[spatial_net["Cell2"].to_numpy()]
    adata.uns["Spatial_Net"] = spatial_net
    return spatial_net


def _slice_coordinates(adata, slice_idx):
    """``(coords, note)``: one slice's (x, y), read through the shared width guard.

    SPIRAL takes one file per slice. A column past the second that is CONSTANT within the file is
    that slice's own z (a slice index or a section depth): the slice is one plane, x/y are all of
    it, and the note says what was set aside. A column that VARIES is a stack inside one file, and
    taking two columns would lay it on one plane -- ``spatial_coords`` refuses that. Both the KNN
    graph and the coordinate CSV use what this returns, so they cannot disagree.
    """
    raw = adata.obsm["spatial"]
    arr = np.asarray(raw.toarray() if hasattr(raw, "toarray") else raw, dtype=np.float64)
    planar = bool(
        arr.ndim == 2
        and arr.shape[1] > 2
        and arr.shape[0] > 0
        and np.all(np.nanmax(arr[:, 2:], axis=0) == np.nanmin(arr[:, 2:], axis=0))
    )
    coords, note = spatial_coords(adata, "spatial", want=2, tool="SPIRAL", project=planar)
    if note:
        note = (
            f"slice_{slice_idx}: obsm['spatial'] has {arr.shape[1]} columns and every column past "
            "the second is constant within the slice (one plane), so x and y were used and that "
            "constant was set aside."
        )
    return coords, note


def _read_slice(h5ad_path, slice_idx, renamed, tissue=None):
    """Load one slice: background left out, identifiers made unique (and counted), coordinates
    read, names prefixed. ``tissue`` accumulates what ``keep_in_tissue`` dropped, per slice."""
    log(f"Reading {h5ad_path} (slice {slice_idx})")
    adata = sc.read_h5ad(h5ad_path)
    if "spatial" not in adata.obsm:
        raise ValueError(f"Slice {h5ad_path} is missing obsm['spatial'].")
    adata, n_supplied, n_off = keep_in_tissue(adata, f"spots of slice {slice_idx}")
    if tissue is not None:
        tissue.setdefault("n_supplied", []).append(int(n_supplied))
        tissue.setdefault("n_dropped", []).append(int(n_off))
    if n_off:
        log(f"  slice {slice_idx}: left out {n_off} of {n_supplied} spots with obs['in_tissue'] == 0 (background)")
    # Duplicate gene symbols collide in the feature table; duplicate barcodes break SPIRAL's
    # name-keyed graph (upstream maps edge endpoints through a {name: row} dict).
    make_names_unique_and_report(adata, into=renamed)
    coords, note = _slice_coordinates(adata, slice_idx)
    # Prefix obs_names with slice index to ensure uniqueness across slices
    adata.obs_names = [f"s{slice_idx}_{name}" for name in adata.obs_names]
    return adata, coords, note


def write_spiral_csvs(adata, coords, out_dir, slice_idx, genes, knn=6):
    """
    Write the CSV files SPIRAL reads for one slice:
      - features CSV (cells x ``genes``; dense, as SPIRAL requires)
      - edge file (KNN spatial graph on ``coords``)
      - meta CSV (celltype, batch columns; SPIRAL trains on ``batch`` only)
      - coord CSV (x, y)

    Returns (feat_path, edge_path, meta_path, coord_path).
    """
    log(f"Writing SPIRAL CSVs for slice {slice_idx}")
    sub = adata[:, list(genes)]
    X = sub.X
    dense = X.toarray() if sp.issparse(X) else np.asarray(X)
    feat_df = pd.DataFrame(dense, index=adata.obs_names, columns=list(genes))

    # Build spatial KNN graph
    spatial_net = _cal_spatial_net_knn(adata, k_cutoff=knn, coords=coords)

    # Meta DataFrame in SPIRAL's own layout. Upstream reads only 'batch'; 'celltype' is carried
    # for that layout and never reaches training or any output this worker reports.
    celltype_col = None
    for candidate in ["celltype", "cell_type", "CellType", "cluster", "annotation", "label"]:
        if candidate in adata.obs.columns:
            celltype_col = candidate
            break
    meta_dict = {"batch": [f"slice_{slice_idx}"] * adata.n_obs}
    if celltype_col is not None:
        meta_dict["celltype"] = adata.obs[celltype_col].values
    else:
        meta_dict["celltype"] = ["unknown"] * adata.n_obs
    meta_df = pd.DataFrame(meta_dict, index=adata.obs_names)

    coord_df = pd.DataFrame(coords, index=adata.obs_names, columns=["x", "y"])

    prefix = os.path.join(out_dir, f"slice_{slice_idx}")
    feat_path = prefix + "_features.csv"
    edge_path = prefix + "_edges.csv"
    meta_path = prefix + "_meta.csv"
    coord_path = prefix + "_coord.csv"

    edges = spatial_net[["Cell1", "Cell2"]].to_numpy()
    _replace_into(feat_path, feat_df.to_csv)
    _replace_into(edge_path, lambda tmp: np.savetxt(tmp, edges, fmt="%s"))
    _replace_into(meta_path, meta_df.to_csv)
    _replace_into(coord_path, coord_df.to_csv)

    log(f"  Slice {slice_idx}: n_cells={adata.n_obs}, n_genes={len(genes)}, n_edges={len(spatial_net)}")
    return feat_path, edge_path, meta_path, coord_path


# ---------------------------------------------------------------------------
# Clustering of the joint embedding
# ---------------------------------------------------------------------------


def _cluster(ann, cluster_method, n_clusters, resolution):
    """Cluster ``ann.obsm['spiral']``; return ``(ann, cluster_key)``.

    mclust fits exactly ``n_clusters`` Gaussian components; leiden/louvain run at ``resolution``
    and never see ``n_clusters``.
    """
    with contextlib.redirect_stdout(sys.stderr):
        if cluster_method == "mclust":
            # Resolve R from this box, not from the one this file was written on: an R_HOME the
            # caller already set is kept, and if nothing resolves the variable is left unset so
            # rpy2's own detection still gets its turn.
            ensure_r_home()
            ann = mclust_R(ann, used_obsm="spiral", num_cluster=n_clusters)
            return ann, "mclust"
        sc.pp.neighbors(ann, use_rep="spiral")
        if cluster_method == "louvain":
            # flavor='vtraag' is scanpy's default and the only one that honours resolution
            # (flavor='igraph' ignores it); _require_cluster_backend checked the package up front.
            sc.tl.louvain(ann, resolution=resolution, flavor="vtraag")
            return ann, "louvain"
        sc.tl.leiden(ann, resolution=resolution)
        return ann, "leiden"


def _require_cluster_backend(cluster_method):
    """Refuse, before anything is written or trained, a clustering this environment cannot run.

    ``sc.tl.louvain`` (flavor 'vtraag', the only flavour that takes ``resolution``) imports Traag's
    ``louvain`` package, which the SPIRAL environment does not ship. The call came after the whole
    training and after the embedding CSVs were written, so a 'louvain' run spent its training and
    then failed. Nothing is substituted: leiden is a different algorithm, and flavor 'igraph'
    would drop ``resolution``.
    """
    if cluster_method != "louvain":
        return
    try:
        import louvain  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "cluster_method='louvain' runs scanpy's sc.tl.louvain (flavor 'vtraag'), which needs the "
            f"'louvain' package, and it cannot be imported in this environment ({type(exc).__name__}: {exc}). "
            "Install it into the SPIRAL environment (pip install louvain), or pass cluster_method='leiden' "
            "(also resolution-driven) or 'mclust' (exactly n_clusters). Nothing was trained or written."
        ) from exc


def _record_clustering(out, cluster_method, n_clusters, resolution):
    """Say which of ``n_clusters``/``resolution`` shaped the clusters; return the count requested.

    ``None`` for leiden/louvain: nothing asked them for a count, so the analysis must not frame
    their answer as the caller's request even when the numbers happen to agree.
    """
    if cluster_method == "mclust":
        record_ignored(
            out,
            ["resolution"],
            "cluster_method='mclust' fits exactly n_clusters components; resolution applies to leiden/louvain only",
        )
        return int(n_clusters)
    record_ignored(
        out,
        ["n_clusters"],
        f"cluster_method={cluster_method!r} is resolution-driven: it ran at resolution={resolution} and "
        "found as many clusters as that resolution gives. Use cluster_method='mclust' to ask for exactly "
        "n_clusters, or change resolution.",
    )
    return None


def _cluster_description(cluster_method, n_clusters, resolution):
    if cluster_method == "mclust":
        return f"mclust (R, EEE) with n_clusters={n_clusters}"
    return f"{cluster_method} at resolution={resolution}"


def _record_device(out):
    out.add_params({"device_used": str(DEVICE)})
    request = str(DEVICE_REQUEST or "").strip().lower()
    wanted_gpu = request.startswith(_GPU_REQUEST_WORDS) or (request.isdigit() and int(request) >= 0)
    if wanted_gpu and DEVICE.type != "cuda":
        out.add_warning(
            f"device={DEVICE_REQUEST!r} was requested but no CUDA device is available; SPIRAL ran on {DEVICE}."
        )


# ---------------------------------------------------------------------------
# Task: integration
# ---------------------------------------------------------------------------


def _integrate(
    h5ad_paths,
    output_dir,
    n_epochs=200,
    hidden_dim=32,
    latent_dim=32,
    knn=6,
    batch_size=1024,
    n_clusters=7,
    cluster_method="leiden",
    resolution=0.8,
):
    """Run SPIRAL integration and clustering; write its files and return what the payload needs."""
    if cluster_method != "mclust" and not float(resolution) > 0:
        raise ValueError(f"resolution must be > 0 for cluster_method={cluster_method!r}; got {resolution}.")
    _require_cluster_backend(cluster_method)
    log(f"Task = integrate, {len(h5ad_paths)} slices")
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "spiral_input_csvs"
    csv_dir.mkdir(exist_ok=True)

    warnings = []
    info = []
    renamed = {"n_genes_renamed": 0, "n_cells_renamed": 0}
    tissue = {}
    adatas = []
    slice_coords = []
    for i, h5ad_path in enumerate(h5ad_paths):
        adata, coords, note = _read_slice(h5ad_path, i, renamed, tissue)
        adatas.append(adata)
        slice_coords.append(coords)
        if note:
            info.append(note)

    # Find common genes across all slices
    common_genes = set(adatas[0].var_names)
    for ad in adatas[1:]:
        common_genes &= set(ad.var_names)
    common_genes = sorted(common_genes)
    log(f"Common genes across {len(h5ad_paths)} slices: {len(common_genes)}")

    if len(common_genes) == 0:
        # Naming an arbitrary pair would be misleading with more than two slices, so find the
        # slice that actually emptied the intersection. Error path only -- the loop above is
        # untouched.
        running, culprit = set(adatas[0].var_names), 1
        for _j, _ad in enumerate(adatas[1:], start=1):
            if not (running & set(_ad.var_names)):
                culprit = _j
                break
            running &= set(_ad.var_names)
        raise ValueError(
            id_mismatch_msg("genes", "slice 1", adatas[0].var_names, f"slice {culprit + 1}", adatas[culprit].var_names)
        )

    total_cells = sum(ad.n_obs for ad in adatas)
    _check_feature_budget(total_cells, len(common_genes), n_background=sum(tissue.get("n_dropped", [])))

    # Feature CSVs hold only the shared genes, in one order: SPIRAL concatenates them row-wise.
    feat_files = []
    edge_files = []
    meta_files = []
    coord_files = []
    for i, adata in enumerate(adatas):
        feat_p, edge_p, meta_p, coord_p = write_spiral_csvs(
            adata, slice_coords[i], str(csv_dir), i, common_genes, knn=knn
        )
        feat_files.append(feat_p)
        edge_files.append(edge_p)
        meta_files.append(meta_p)
        coord_files.append(coord_p)

    # Determine model dimensions
    n_features = len(common_genes)
    n_slices = len(h5ad_paths)
    if n_slices == 2:
        M = 1
    else:
        M = n_slices

    # Build argparse-like namespace for SPIRAL params
    from spiral.layers import MeanAggregator

    class SpiralParams:
        pass

    hidden_width = int(hidden_dim) * HIDDEN_WIDTH_FACTOR
    params = SpiralParams()
    params.seed = 0
    params.AEdims = [n_features, [hidden_width], latent_dim]
    params.AEdimsR = [latent_dim, [hidden_width], n_features]
    params.GSdims = [hidden_width, latent_dim]
    params.zdim = latent_dim
    params.znoise_dim = ZNOISE_DIM
    params.CLdims = [4, [], M]
    params.DIdims = [latent_dim - 4, [latent_dim, latent_dim // 2], M]
    params.beta = 1.0
    params.agg_class = MeanAggregator
    params.num_samples = knn
    params.N_WALKS = knn
    params.WALK_LEN = 1
    params.N_WALK_LEN = knn
    params.NUM_NEG = knn
    params.Q = 10
    params.epochs = n_epochs
    # Cap batch_size so DataLoader (drop_last=True) yields at least one batch, and so no batch
    # holds every node -- SPIRAL's node extension asserts the extended set is a *strict* superset.
    effective_batch_size = graph_batch_size(batch_size, total_cells)
    log(f"Effective batch_size={effective_batch_size} (requested={batch_size}, total_cells={total_cells})")
    if effective_batch_size != int(batch_size):
        warnings.append(
            f"batch_size={batch_size} was not usable with {total_cells} spots (a batch must hold at least "
            f"one spot and fewer than all of them, so SPIRAL's neighbour extension can grow it); training "
            f"used batch_size={effective_batch_size} (params.batch_size_effective)."
        )
    params.batch_size = effective_batch_size
    params.lr = 1e-3
    params.weight_decay = 5e-4
    params.alpha1 = float(n_features)
    params.alpha2 = 1.0
    params.alpha3 = 1.0
    params.alpha4 = 1.0
    params.lamda = 1.0

    log("Initializing SPIRAL_integration model...")
    with contextlib.redirect_stdout(sys.stderr):
        spii = SPIRAL_integration(params, feat_files, edge_files, meta_files)
        log("Training SPIRAL model...")
        spii.train()

    # Extract embeddings
    log("Extracting embeddings...")
    import torch.nn as nn

    spii.model.eval()
    with contextlib.redirect_stdout(sys.stderr):
        all_idx = np.arange(spii.feat.shape[0])
        all_layer, all_mapping = layer_map(all_idx.tolist(), spii.adj, len(params.GSdims))
        all_rows = spii.adj.tolil().rows[all_layer[0]]
        all_feature = torch.Tensor(spii.feat.iloc[all_layer[0], :].values).float().to(DEVICE)
        all_embed, ae_out, clas_out, disc_out = spii.model(
            all_feature,
            all_layer,
            all_mapping,
            all_rows,
            params.lamda,
            spii.de_act,
            spii.cl_act,
        )
        [ae_embed, gs_embed, embed] = all_embed
        embed = embed.cpu().detach()

    names = [f"SPIRAL_{i}" for i in range(embed.shape[1])]
    embed_df = pd.DataFrame(np.array(embed), index=spii.feat.index, columns=names)
    embed_file = str(out_dir / "spiral_embeddings.csv")
    _replace_into(embed_file, embed_df.to_csv)
    log(f"Saved embeddings to {embed_file}")

    # Batch-corrected expression
    with contextlib.redirect_stdout(sys.stderr):
        embed_no_noise = torch.cat(
            (
                torch.zeros((embed.shape[0], params.znoise_dim)),
                embed[:, params.znoise_dim :],
            ),
            dim=1,
        )
        xbar = np.array(spii.model.agc.ae.de(embed_no_noise.to(DEVICE), nn.Sigmoid())[1].cpu().detach())
    corrected_df = pd.DataFrame(xbar, index=spii.feat.index, columns=spii.feat.columns)
    corrected_file = str(out_dir / "spiral_corrected_expression.csv")
    _replace_into(corrected_file, corrected_df.to_csv)
    log(f"Saved corrected expression to {corrected_file}")

    # Clustering on the embedding (excluding noise dims)
    log(f"Clustering with method={cluster_method}, n_clusters={n_clusters}, resolution={resolution}")
    import anndata

    ann = anndata.AnnData(spii.feat)
    ann.obsm["spiral"] = embed_df.iloc[:, params.znoise_dim :].values
    ann, cluster_key = _cluster(ann, cluster_method, n_clusters, resolution)

    # Add batch info and spatial coords
    ann.obs["batch"] = spii.meta.loc[:, "batch"].values

    # Combine coordinates
    all_coords = []
    for i in range(len(coord_files)):
        c = pd.read_csv(coord_files[i], index_col=0)
        all_coords.append(c)
    coords_combined = pd.concat(all_coords)
    coords_combined.columns = ["x", "y"]
    ann.obsm["spatial"] = coords_combined.loc[ann.obs_names, :].values

    # Save results
    cluster_csv = str(out_dir / "spiral_clusters.csv")
    _replace_into(cluster_csv, ann.obs[[cluster_key, "batch"]].to_csv)
    log(f"Saved clusters to {cluster_csv}")

    out_h5ad = str(out_dir / "spiral_integrated.h5ad")
    ann.obsm["spiral_embedding"] = embed_df.values
    _replace_into(out_h5ad, ann.write_h5ad)
    log(f"Saved integrated h5ad to {out_h5ad}")

    info.append(
        "spiral_corrected_expression.csv (and X of spiral_integrated.h5ad) are in SPIRAL's per-spot "
        "min-max scaled [0, 1] space, not counts. obsm['spatial'] of spiral_integrated.h5ad is each "
        "slice's own input coordinates stacked -- the slices are not registered to one frame "
        "(spiral_align does that)."
    )

    return {
        "out_dir": out_dir,
        "ann": ann,
        "cluster_key": cluster_key,
        "cluster_sizes": ann.obs[cluster_key].value_counts().to_dict(),
        "total_spots": int(total_cells),
        "slice_sizes": [int(ad.n_obs) for ad in adatas],
        "n_common_genes": len(common_genes),
        "embedding_dim": int(embed_df.shape[1]),
        "files": {
            "integrated_h5ad": out_h5ad,
            "embeddings_csv": embed_file,
            "corrected_expression_csv": corrected_file,
            "clusters_csv": cluster_csv,
        },
        "batch_size_effective": int(effective_batch_size),
        "hidden_width": hidden_width,
        "renamed": renamed,
        "in_tissue": {
            "n_supplied": int(sum(tissue.get("n_supplied", []))),
            "n_dropped": int(sum(tissue.get("n_dropped", []))),
            "per_slice_dropped": list(tissue.get("n_dropped", [])),
        },
        "warnings": warnings,
        "info": info,
    }


def _record_integration(out, run, n_clusters, cluster_method, resolution):
    """The payload lines integrate and align share: effective settings, notes, renamed identifiers."""
    out.add_params(
        {
            "batch_size_effective": run["batch_size_effective"],
            "hidden_width": run["hidden_width"],
            "cluster_key": run["cluster_key"],
        }
    )
    out.add_params(identifier_rename_params(run["renamed"]))
    tissue = run.get("in_tissue") or {}
    record_in_tissue(out, int(tissue.get("n_supplied", 0)), int(tissue.get("n_dropped", 0)))
    if tissue.get("n_dropped"):
        out.add_params({"in_tissue_dropped_per_slice": tissue.get("per_slice_dropped", [])})
    _record_device(out)
    for message in run["warnings"]:
        out.add_warning(message)
    for message in run["info"]:
        out.add_info(message)
    return _record_clustering(out, cluster_method, n_clusters, resolution)


def run_integration(
    h5ad_paths,
    output_dir,
    n_epochs=200,
    hidden_dim=32,
    latent_dim=32,
    knn=6,
    batch_size=1024,
    n_clusters=7,
    cluster_method="leiden",
    resolution=0.8,
):
    """Integrate multiple spatial slices using SPIRAL."""
    run = _integrate(
        h5ad_paths,
        output_dir,
        n_epochs=n_epochs,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        knn=knn,
        batch_size=batch_size,
        n_clusters=n_clusters,
        cluster_method=cluster_method,
        resolution=resolution,
    )
    cluster_sizes = run["cluster_sizes"]
    total_spots = run["total_spots"]

    out = WorkerOutput("spiral", task="integration")
    out.set_data(
        n_slices=len(h5ad_paths),
        n_spots_total=int(total_spots),
        n_common_genes=run["n_common_genes"],
        slice_sizes=run["slice_sizes"],
    )
    out.add_output_files(run["files"])
    out.add_params(
        {
            "n_epochs": n_epochs,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "knn": knn,
            "batch_size": batch_size,
            "n_clusters": n_clusters,
            "cluster_method": cluster_method,
            "resolution": resolution,
        }
    )
    record_method(out, f"{INTEGRATE_METHOD}; domains: {_cluster_description(cluster_method, n_clusters, resolution)}")
    n_requested = _record_integration(out, run, n_clusters, cluster_method, resolution)
    out.set_summary(
        n_clusters=len(cluster_sizes),
        cluster_sizes=cluster_sizes,
        embedding_dim=run["embedding_dim"],
    )
    out.set_analysis(
        build_cluster_analysis(
            cluster_sizes, cluster_key="domain", total_spots=int(total_spots), n_requested=n_requested
        )
        + identifier_rename_note(run["renamed"], "input slices")
    )
    return out.to_dict()


# ---------------------------------------------------------------------------
# Task: alignment
# ---------------------------------------------------------------------------


def _align_shared_clusters(c0, c1, clusters_0, clusters_1, embed_0, embed_1, names_0, names_1, alpha, out_dir):
    """Map slice_1 spots into slice_0's frame, one shared cluster at a time.

    Returns ``(new_coords_1, placed, pi_files, shared, unplaced_why)``. ``new_coords_1`` is NaN for
    every slice_1 spot this step did not place, and ``placed`` says which ones it did: a spot whose
    cluster is absent from slice_0, whose cluster has fewer than 2 spots in either slice, or which
    received no transport mass. Those spots were never moved, so writing their input coordinates
    beside moved ones would present a point in slice_1's frame as a point in slice_0's.
    """
    import ot

    clusters_0 = np.asarray(clusters_0)
    clusters_1 = np.asarray(clusters_1)
    shared = sorted(set(clusters_0.tolist()) & set(clusters_1.tolist()))
    log(f"Shared clusters for alignment: {shared}")

    new_coords_1 = np.full(c1.shape, np.nan, dtype=np.float64)
    placed = np.zeros(c1.shape[0], dtype=bool)
    pi_files = {}
    unplaced_why = {}

    for clust in sorted(set(clusters_1.tolist()) - set(shared)):
        unplaced_why[str(clust)] = "cluster absent from slice_0"

    blocks = []
    for clust in shared:
        n0 = int((clusters_0 == clust).sum())
        n1 = int((clusters_1 == clust).sum())
        if n0 >= 2 and n1 >= 2:
            blocks.append((clust, n0, n1))
    _check_fgw_budget(blocks)

    with contextlib.redirect_stdout(sys.stderr):
        for clust in shared:
            mask_0 = clusters_0 == clust
            mask_1 = clusters_1 == clust

            if mask_0.sum() < 2 or mask_1.sum() < 2:
                log(f"  Skipping cluster {clust}: too few spots")
                unplaced_why[str(clust)] = (
                    f"fewer than 2 spots in a slice ({int(mask_0.sum())} in slice_0, {int(mask_1.sum())} in slice_1)"
                )
                continue

            e0 = np.asarray(embed_0)[mask_0].astype(np.float64)
            e1 = np.asarray(embed_1)[mask_1].astype(np.float64)
            sc0 = c0[mask_0]
            sc1 = c1[mask_1]

            # Spatial distance matrices
            D1 = ot.dist(sc0, sc0, metric="euclidean")
            D2 = ot.dist(sc1, sc1, metric="euclidean")

            # Expression distance
            M = ot.dist(e0, e1, metric="euclidean")

            # Uniform distributions
            d1 = np.ones(e0.shape[0]) / e0.shape[0]
            d2 = np.ones(e1.shape[0]) / e1.shape[0]

            # Fused Gromov-Wasserstein
            log(f"  GW alignment for cluster {clust} ({e0.shape[0]} vs {e1.shape[0]} spots)")
            pi = np.asarray(
                ot.gromov.fused_gromov_wasserstein(M, D1, D2, d1, d2, loss_fun="square_loss", alpha=alpha),
                dtype=np.float64,
            )

            pi_path = str(Path(out_dir) / f"spiral_align_pi_cluster_{clust}.csv")
            pi_df = pd.DataFrame(
                pi,
                index=np.asarray(names_0)[mask_0],
                columns=np.asarray(names_1)[mask_1],
            )
            _replace_into(pi_path, pi_df.to_csv)
            pi_files[f"pi_cluster_{clust}"] = pi_path

            # Map slice_1 coordinates to slice_0 reference frame: transport-weighted mean
            mass = pi.sum(axis=0)
            ok = np.isfinite(mass) & (mass > 0)
            rows = np.where(mask_1)[0]
            if ok.any():
                new_coords_1[rows[ok]] = (pi[:, ok] / mass[ok]).T @ sc0
                placed[rows[ok]] = True
            if not ok.all():
                unplaced_why[str(clust)] = f"{int((~ok).sum())} spot(s) received no transport mass"

    return new_coords_1, placed, pi_files, shared, unplaced_why


def run_alignment(
    h5ad_paths,
    output_dir,
    n_epochs=200,
    hidden_dim=32,
    latent_dim=32,
    knn=6,
    batch_size=1024,
    n_clusters=7,
    cluster_method="leiden",
    alpha=0.5,
    resolution=0.8,
):
    """Align two spatial slices: SPIRAL integration, then this worker's per-cluster FGW mapping."""
    if len(h5ad_paths) != 2:
        raise ValueError("spiral_align requires exactly two h5ad files.")

    log("Task = align, 2 slices")
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: the integration (the same code path spiral_integrate runs)
    run = _integrate(
        h5ad_paths,
        output_dir,
        n_epochs=n_epochs,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        knn=knn,
        batch_size=batch_size,
        n_clusters=n_clusters,
        cluster_method=cluster_method,
        resolution=resolution,
    )

    # Step 2: Load integration results for alignment
    log("Starting coordinate alignment via fused Gromov-Wasserstein OT...")
    embed_df = pd.read_csv(str(out_dir / "spiral_embeddings.csv"), index_col=0)
    cluster_df = pd.read_csv(str(out_dir / "spiral_clusters.csv"), index_col=0)
    cluster_col = [c for c in cluster_df.columns if c != "batch"][0]

    # Load coordinates per slice
    csv_dir = out_dir / "spiral_input_csvs"
    coord0 = pd.read_csv(str(csv_dir / "slice_0_coord.csv"), index_col=0)
    coord1 = pd.read_csv(str(csv_dir / "slice_1_coord.csv"), index_col=0)

    # Determine batch labels
    batch_labels = cluster_df["batch"].values
    unique_batches = np.unique(batch_labels)
    idx_0 = cluster_df.index[cluster_df["batch"] == unique_batches[0]]
    idx_1 = cluster_df.index[cluster_df["batch"] == unique_batches[1]]

    # Embedding subsets (exclude noise dims)
    embed_0 = embed_df.loc[idx_0].iloc[:, ZNOISE_DIM:].values
    embed_1 = embed_df.loc[idx_1].iloc[:, ZNOISE_DIM:].values

    # Coordinates
    c0 = coord0.loc[idx_0, ["x", "y"]].values.astype(np.float64)
    c1 = coord1.loc[idx_1, ["x", "y"]].values.astype(np.float64)

    clusters_0 = cluster_df.loc[idx_0, cluster_col].values
    clusters_1 = cluster_df.loc[idx_1, cluster_col].values

    new_coords_1, placed, pi_files, shared_clusters, unplaced_why = _align_shared_clusters(
        c0, c1, clusters_0, clusters_1, embed_0, embed_1, idx_0, idx_1, alpha, out_dir
    )
    n_placed = int(placed.sum())
    n_unplaced = int(len(idx_1) - n_placed)
    if n_placed == 0:
        raise ValueError(
            f"Nothing was aligned: no cluster of the {cluster_method} clustering holds at least 2 spots in "
            f"both slices ({len(shared_clusters)} shared cluster(s); why each slice_1 cluster was left out: "
            f"{unplaced_why}). The integration outputs in {out_dir} are complete. Clusters shared by both "
            "slices are what this alignment maps through: change resolution (leiden/louvain) or use "
            "cluster_method='mclust' with n_clusters."
        )

    # Save aligned coordinates
    aligned_coord_df = pd.DataFrame(new_coords_1, index=idx_1, columns=["x", "y"])
    ref_coord_df = pd.DataFrame(c0, index=idx_0, columns=["x", "y"])
    combined_aligned = pd.concat([ref_coord_df, aligned_coord_df])
    combined_aligned["batch"] = ["slice_0"] * len(idx_0) + ["slice_1"] * len(idx_1)

    aligned_csv = str(out_dir / "spiral_aligned_coordinates.csv")
    _replace_into(aligned_csv, combined_aligned.to_csv)
    log(f"Saved aligned coordinates to {aligned_csv}")

    # Update the integrated h5ad with aligned coords
    integrated_h5ad = str(out_dir / "spiral_integrated.h5ad")
    ann = sc.read_h5ad(integrated_h5ad)
    ann.obsm["spatial_aligned"] = combined_aligned.loc[ann.obs_names, ["x", "y"]].values
    placed_by_name = pd.Series(
        np.concatenate([np.ones(len(idx_0), dtype=bool), placed]),
        index=list(idx_0) + list(idx_1),
    )
    ann.obs["spiral_aligned"] = placed_by_name.loc[ann.obs_names].to_numpy(dtype=bool)
    aligned_h5ad = str(out_dir / "spiral_aligned.h5ad")
    _replace_into(aligned_h5ad, ann.write_h5ad)
    log(f"Saved aligned h5ad to {aligned_h5ad}")

    # Build output
    out = WorkerOutput("spiral", task="alignment")
    out.set_data(
        n_slices=2,
        n_spots_slice_0=len(idx_0),
        n_spots_slice_1=len(idx_1),
        n_shared_clusters=len(shared_clusters),
        n_spots_aligned_slice_1=n_placed,
        n_spots_unaligned_slice_1=n_unplaced,
    )
    out.add_output_files(
        {
            "aligned_h5ad": aligned_h5ad,
            "aligned_coordinates_csv": aligned_csv,
            "embeddings_csv": str(out_dir / "spiral_embeddings.csv"),
            "clusters_csv": str(out_dir / "spiral_clusters.csv"),
            **pi_files,
        }
    )
    out.add_params(
        {
            "n_epochs": n_epochs,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "knn": knn,
            "batch_size": batch_size,
            "n_clusters": n_clusters,
            "cluster_method": cluster_method,
            "resolution": resolution,
            "alpha": alpha,
            "reference_slice": "slice_0",
        }
    )
    record_method(out, ALIGN_METHOD)
    _record_integration(out, run, n_clusters, cluster_method, resolution)
    if n_unplaced:
        out.add_warning(
            f"{n_unplaced} of {len(idx_1)} slice_1 spots were not placed in slice_0's frame "
            f"({unplaced_why}). Their rows in spiral_aligned_coordinates.csv and obsm['spatial_aligned'] are "
            "NaN and obs['spiral_aligned'] is False: they were never moved, so their input coordinates "
            "are in slice_1's own frame. SPIRAL's CoordAlignment places such spots with a Procrustes fit; "
            "this worker does not."
        )
    out.set_summary(
        n_clusters_aligned=len(pi_files),
        shared_clusters=sorted(shared_clusters),
        n_spots_unaligned_slice_1=n_unplaced,
    )
    out.set_analysis(
        f"Mapped slice_1 onto slice_0 ({len(idx_0)} + {len(idx_1)} spots) with this worker's per-cluster "
        f"fused Gromov-Wasserstein transport (POT, alpha={alpha}) on SPIRAL embeddings, through "
        f"{len(pi_files)} cluster(s) shared by both slices. {n_placed} of {len(idx_1)} slice_1 spots were "
        f"placed; {n_unplaced} were not and are NaN. SPIRAL's own CoordAlignment was not run."
        + identifier_rename_note(run["renamed"], "input slices")
    )
    return out.to_dict()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SPIRAL worker: spatial integration & alignment.")
    parser.add_argument(
        "--task",
        required=True,
        choices=["integrate", "align"],
        help="Task to run: 'integrate' for batch correction, 'align' for coordinate alignment.",
    )
    parser.add_argument(
        "--h5ad",
        dest="h5ad_paths",
        action="append",
        required=True,
        help="Path to a spatial AnnData .h5ad file. Use multiple times for multiple slices.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--n-epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Hidden size unit: the hidden layers are hidden_dim*16 wide (512 at the default).",
    )
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension.")
    parser.add_argument("--knn", type=int, default=6, help="K nearest neighbors for spatial graph.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size.")
    parser.add_argument("--n-clusters", type=int, default=7, help="Number of clusters (cluster_method 'mclust' only).")
    parser.add_argument(
        "--cluster-method",
        default="leiden",
        choices=["leiden", "louvain", "mclust"],
        help="Clustering method.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.8,
        help="Resolution for leiden/louvain (ignored by mclust).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="GW alpha (alignment only): tradeoff between expression and spatial.",
    )
    # Declared so it appears in --help and is not rejected as unknown; the value was already read
    # out of argv by _device_request_from_argv() above, because DEVICE had to exist before the
    # SPIRAL import at module scope.
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device: 'auto', 'cpu', 'gpu'/'cuda', or 'cuda:N' for a specific GPU.",
    )

    args = parser.parse_args()
    log(f"Using device: {DEVICE}")

    try:
        if args.task == "integrate":
            if len(args.h5ad_paths) < 2:
                raise ValueError("Integration requires at least 2 h5ad files.")
            result = run_integration(
                h5ad_paths=args.h5ad_paths,
                output_dir=args.output_dir,
                n_epochs=args.n_epochs,
                hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim,
                knn=args.knn,
                batch_size=args.batch_size,
                n_clusters=args.n_clusters,
                cluster_method=args.cluster_method,
                resolution=args.resolution,
            )
        else:  # align
            if len(args.h5ad_paths) != 2:
                raise ValueError("Alignment requires exactly 2 h5ad files.")
            result = run_alignment(
                h5ad_paths=args.h5ad_paths,
                output_dir=args.output_dir,
                n_epochs=args.n_epochs,
                hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim,
                knn=args.knn,
                batch_size=args.batch_size,
                n_clusters=args.n_clusters,
                cluster_method=args.cluster_method,
                alpha=args.alpha,
                resolution=args.resolution,
            )

        print(json.dumps(result, default=str))
        sys.stdout.flush()

    except Exception as e:
        log("ERROR: Exception during SPIRAL run.")
        traceback.print_exc(file=sys.stderr)
        task = args.task if hasattr(args, "task") else "unknown"
        WorkerOutput.emit_error("spiral", str(e), task=task)
        sys.exit(1)


if __name__ == "__main__":
    main()
