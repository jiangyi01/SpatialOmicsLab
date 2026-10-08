#!/usr/bin/env python
"""
sedr_worker.py

Worker script for running SEDR spatial embedding and clustering via
variational graph autoencoder.

- Called by the FastMCP wrapper (sedr_mcp_server.py).
- Must be executed inside the SEDR conda env: /opt/conda/envs/sedr_env
- All logs go to stderr; stdout only prints a single JSON line at the end.

What runs, honestly: this worker's OWN preprocessing (not the upstream tutorial's -- see
``run_sedr_clustering``), SEDR's variational graph autoencoder for the embedding, and then
KMeans on that embedding for the domain labels. ``using_dec`` only decides whether SEDR's
Deep-Embedding-Clustering step refines the embedding before KMeans runs; every label the
worker writes is a KMeans label. Spots with ``obs['in_tissue'] == 0`` (background glass, which
CELLxGENE Visium exports ship beside the tissue) are left out before anything runs and counted
under ``params.in_tissue_filter`` with a warning; ``data.n_spots`` is the supplied count and
``data.n_spots_used`` the analysed one.

The pipeline normalises X as counts (normalize_total + log1p), so the matrix is checked first
(``worker_utils.choose_counts_matrix``): a negative or non-finite X (scaled / z-scored data) is
refused, naming ``--use-raw-counts`` when ``adata.raw`` holds the counts; a non-negative non-integer X
(already normalised or log-transformed) runs as before with a warning; ``--use-raw-counts`` runs on
``adata.raw.X``. ``params.expression_source`` / ``params.x_matrix_kind`` say which matrix ran.

Example (manual test):

  (sedr_env) python /workspace/epic-fermat/agent/tools/sedr_worker.py \
      --spatial-h5ad /workspace/work/spatial_input/V1_Human_Lymph_Node.h5ad \
      --output-dir /workspace/work/sedr_V1_LN \
      --n-clusters 7 \
      --using-dec True \
      --hvg-flavor seurat \
      --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from typing import Any

# Ensure worker_utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add SEDR repo to path
# <TOOL>_SRC seam (as spatialscope_worker.py does): this checkout exists only where it was
# installed, and a sys.path entry that does not exist fails silently -- the run dies later in an
# ImportError naming an upstream module, with no way to redirect it. The literal stays the default.
sys.path.insert(
    0,
    os.environ.get("SEDR_SRC") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "sedr_repo"),
)

from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    build_cluster_analysis,
    choose_counts_matrix,
    describe_reduction,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    preflight_check,
    record_expression_source,
    record_in_tissue,
    record_method,
    require_hvg_flavor,
    resolve_compute,
    spatial_coords,
    unsupported_choice_msg,
)

# --- Fixed settings of this worker's pipeline. Reported in ``params`` so a reader can tell what
# --- the run was set to; none is a caller-facing knob (the portal exposes hvg_flavor only).
GENE_MIN_CELLS = 3  # sc.pp.filter_genes(min_cells=...) before anything else
HVG_N_TOP_GENES = 3000  # highly-variable genes kept (or every detected gene when fewer survive)
KNN_K = 12  # neighbours in SEDR.graph_construction's spatial kNN graph
# DEC's KL target is built from this many KMeans centroids -- upstream SEDR_module's own default
# (``dec_clsuter_n=10``). It is NOT the requested domain count: upstream's tutorial trains DEC at
# this default and clusters the latent afterwards, and so does this worker.
DEC_CLUSTER_N_DEFAULT = 10
# Upstream's tutorial calls ``SEDR.fix_seed(2023)``. The seed fixes the random streams (weight init,
# the non-neighbour mask); it makes a run bit-reproducible only on ONE CPU thread
# (OMP_NUM_THREADS=1): two 1-thread runs on SpinalCord were byte-identical, while two 3-thread runs
# agreed at ARI 0.86 (measured 2026-09-29) -- parallel floating-point reductions differ run to run and
# can move spots between domains. The payload records ``torch_num_threads`` beside the seed. KMeans
# keeps its own long-standing random_state below.
RANDOM_SEED = 2023
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10
# Peak bytes per n_spots^2 of upstream ``SEDR.graph_construction``: a float64 distance matrix, the
# float64 kNN indicator and their symmetrised sum are alive at once (3 x 8). Measured as the ru_maxrss
# delta in sedr_env: 24.11 at n=8k and 24.02 at n=16k (and 24.1/24.0/24.0 at 10k/20k/30k).
GRAPH_BYTES_PER_SPOT_PAIR = 24
# Scanpy flavours this worker will run. ``seurat_v3`` needs scikit-misc and models raw counts, so
# it is run on the counts layer stashed before normalisation (as upstream does with layer='count').
HVG_FLAVORS = ("seurat", "cell_ranger", "seurat_v3")
DEFAULT_HVG_FLAVOR = "seurat"
# Where the raw counts wait for seurat_v3. A private name, so a ``counts`` layer the caller supplied
# is neither overwritten nor dropped from the output file.
HVG_COUNTS_LAYER = "_sedr_hvg_counts"
# Spot diameter (coordinate units) for the domain plot of a slide whose uns['spatial'] states none;
# a library that states its own ``spot_diameter_fullres`` is drawn at that size (_spatial_plot_kwargs).
PLOT_SPOT_SIZE = 150


def log(msg: str) -> None:
    """Print log messages to stderr with a prefix."""
    sys.stderr.write(f"[sedr-worker] {msg}\n")
    sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SEDR worker: spatial embedding and clustering via variational graph autoencoder."
    )
    parser.add_argument(
        "--spatial-h5ad",
        type=str,
        required=True,
        help="Path to spatial transcriptomics AnnData (.h5ad).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save SEDR outputs.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=7,
        help="Number of spatial domains / clusters (the k of the KMeans run on the SEDR embedding).",
    )
    parser.add_argument(
        "--using-dec",
        type=str,
        default="True",
        help="Whether SEDR's Deep Embedding Clustering step refines the embedding before KMeans (True/False).",
    )
    parser.add_argument(
        "--hvg-flavor",
        type=str,
        default=DEFAULT_HVG_FLAVOR,
        help=(
            f"Scanpy highly_variable_genes flavour for the top-{HVG_N_TOP_GENES} gene selection: 'seurat' "
            "(default, dispersion on log data), 'cell_ranger', or 'seurat_v3' (raw counts; needs scikit-misc -- "
            "when it is missing the run stops and says so, it does not switch flavour)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        default=False,
        help=(
            "Run on adata.raw.X instead of X (for a CELLxGENE-style h5ad whose X is normalised or scaled and whose "
            "counts sit in adata.raw). Without it, a negative or non-finite X is refused and a non-integer X runs "
            "with a warning."
        ),
    )
    return parser.parse_args()


def str_to_bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y"}


def resolve_device(device_str: str) -> str:
    """Resolve any device spelling, falling back to cpu if cuda is not available.

    Shared with every other worker (worker_utils.resolve_compute), which also covers the
    spellings this used to pass through untouched -- 'auto', 'gpu' and 'GPU' all reached
    torch.device() as literals and raised.
    """
    return resolve_compute(device_str).device


def _seed_everything(seed: int) -> None:
    """Upstream's ``SEDR.fix_seed`` without the CUDA-only calls that fail on a CPU torch."""
    import random

    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        if cudnn is not None:
            cudnn.deterministic = True
            cudnn.benchmark = False


def _write_atomically(final_path: str, write) -> None:
    """``write(<final>.partial)``, then ``os.replace`` onto the final name.

    A reader never sees a half-written file under the final name, and a write that fails takes its
    ``.partial`` with it instead of leaving it beside the outputs.
    """
    partial = final_path + ".partial"
    try:
        write(partial)
        os.replace(partial, final_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(partial)
        raise


def _memory_available_bytes():
    """Bytes this run can still allocate, or None when the platform cannot say.

    The shared reader (:func:`worker_utils.available_memory_bytes`): the smaller of MemAvailable and
    the room under a cgroup memory limit, with the cgroup's file LRU pages (page cache the kernel
    reclaims before it OOM-kills anything) counted as free. This used to compute the cgroup room as
    ``memory.max - memory.current``, and ``memory.current`` counts that page cache -- so a
    memory-limited container that had merely read files sat "full" and a slide that fitted was refused.
    """
    available = available_memory_bytes()
    return None if available is None else int(available)


def _matrix_nbytes(matrix) -> int:
    """In-memory bytes of a dense array or a scipy sparse matrix (its data and index arrays)."""
    nbytes = getattr(matrix, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    return sum(
        int(getattr(getattr(matrix, part, None), "nbytes", 0) or 0)
        for part in ("data", "indices", "indptr", "row", "col")
    )


def _dense_budget_bytes(n_spots: int, n_features: int) -> int:
    """Peak bytes of the dense intermediates a SEDR run materialises -- intrinsic to upstream.

    Two stages, one after the other, so the peak is the larger of the two (not their sum):

    * ``SEDR.graph_construction`` builds its kNN graph through ``generate_adj_mat``, which holds an
      n x n float64 distance matrix (``sklearn.metrics.pairwise_distances``), an n x n float64
      indicator and their symmetrised sum at once: ``GRAPH_BYTES_PER_SPOT_PAIR`` (24) bytes x n^2,
      measured, with no margin added. Beside them sits the dense n x g HVG matrix, budgeted at
      float64 (8 bytes; the usual float32 input needs half).
    * ``SEDR.Sedr`` then copies that matrix and wraps it in a float32 tensor, and training holds a
      reconstruction of it: budgeted as three n x g float64 copies. The n^2 matrices are gone by
      then (``graph_construction`` returns sparse tensors), so this stage only leads when g > n.
    """
    n = int(n_spots)
    g = int(n_features)
    graph_stage = GRAPH_BYTES_PER_SPOT_PAIR * n * n + 8 * n * g
    sedr_stage = 3 * 8 * n * g
    return max(graph_stage, sedr_stage)


def _check_dense_budget(n_spots: int, n_features: int, released_bytes: int = 0):
    """Refuse up front, with the numbers, a slide whose n^2 graph matrices cannot fit in memory.

    Without this the run dies inside upstream's ``pairwise_distances``/``np.zeros`` with a bare
    MemoryError, or is OOM-killed with no payload at all, after the whole preprocessing has run.
    No parameter of this tool lowers the footprint and the slide is never cut down here, so the
    message says what is needed and what the machine has. ``released_bytes`` is the loaded input
    matrix: MemAvailable is read while it is resident, and preprocessing replaces it (with the HVG
    matrix the budget already counts) before either stage runs, so it is room the run will have.
    Returns the estimate in bytes; when the platform cannot say how much memory is free, the run
    proceeds unchecked.
    """
    need = _dense_budget_bytes(n_spots, n_features)
    available = _memory_available_bytes()
    released = max(0, int(released_bytes or 0))
    if available is not None and need > available + released:
        gib = 1024.0**3
        raise MemoryError(
            f"SEDR builds its spatial graph from dense {n_spots}x{n_spots} spot-by-spot matrices (upstream "
            "SEDR.graph_construction holds a float64 distance matrix, the kNN indicator and their symmetrised "
            f"sum at once: {GRAPH_BYTES_PER_SPOT_PAIR} bytes x n_spots^2, measured) beside a dense "
            f"{n_spots}x{n_features} expression matrix: about {need / gib:.1f} GiB at peak for {n_spots} spots. "
            f"This machine reports {available / gib:.1f} GiB available, plus {released / gib:.1f} GiB the loaded "
            "input matrix frees during preprocessing. No parameter of this tool lowers that footprint; run it on "
            "a machine with more memory. The slide is analysed whole -- it is not subsampled."
        )
    return need


def _check_spatial_coordinates(adata) -> int:
    """Fail before preprocessing, with the shared helper's message, when obsm['spatial'] is missing
    or malformed. Returns the coordinate width.

    SEDR's kNN graph is built from every coordinate column (a 3-column key gives a 3-D graph, which
    is right for a stack), so the width is required to be >= 2 and is not forced down to 2.
    """
    raw = adata.obsm["spatial"] if "spatial" in adata.obsm else None
    width = int(raw.shape[1]) if raw is not None and getattr(raw, "ndim", 0) == 2 else 2
    coords, _ = spatial_coords(adata, "spatial", want=max(2, width), tool="sedr")
    return int(coords.shape[1])


def _torch_num_threads():
    """Threads torch uses for CPU ops -- a seeded run is bit-reproducible only at 1. None if unknown."""
    try:
        import torch

        return int(torch.get_num_threads())
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
        return None


def _spatial_plot_kwargs(adata) -> dict:
    """The ``library_id``/``spot_size`` arguments ``sc.pl.spatial`` needs to draw this slide as it is.

    * ``library_id``: scanpy refuses to guess once ``uns['spatial']`` has more than one key ("Found
      multiple possible libraries"). CELLxGENE Visium exports carry exactly that shape -- the one library
      (a dict of images and scale factors) beside a boolean ``is_single`` flag -- so every such slide lost
      its plot. When exactly one entry is a library it is named; with several, nothing is passed and
      scanpy's refusal is reported in the payload, not hidden.
    * ``spot_size``: the library's own ``spot_diameter_fullres`` when it states one (scanpy reads it when
      ``spot_size`` is None). The fixed ``PLOT_SPOT_SIZE`` of 150 coordinate units is only for a slide
      that states no diameter: on the library's Visium slides it drew each spot 5-8x its real size
      (SpinalCord 29.8, Heart Fetal12W 19.3), so neighbouring domains painted over one another.
    """
    from collections.abc import Mapping

    kwargs = {"spot_size": PLOT_SPOT_SIZE}
    spatial = adata.uns.get("spatial") if hasattr(adata, "uns") else None
    if not isinstance(spatial, Mapping) or not spatial:
        return kwargs
    libraries = [key for key, value in spatial.items() if isinstance(value, Mapping)]
    if len(libraries) != 1:
        return kwargs
    if len(spatial) > 1:
        kwargs["library_id"] = libraries[0]
    scalefactors = spatial[libraries[0]].get("scalefactors")
    if isinstance(scalefactors, Mapping) and scalefactors.get("spot_diameter_fullres"):
        kwargs["spot_size"] = None
    return kwargs


def _validate_n_clusters(n_clusters: int, n_spots: int) -> None:
    """Refuse, before training, a KMeans k that no partition of ``n_spots`` spots can have.

    KMeans only sees the request after SEDR has trained (and after DEC has refined), so an ``n_clusters``
    of 0 or one above the spot count used to cost the whole run before sklearn rejected it.
    """
    if int(n_clusters) < 1 or int(n_clusters) > int(n_spots):
        raise ValueError(
            f"n_clusters={n_clusters} cannot be reached: KMeans on {n_spots} analysed spots makes between 1 and "
            f"{n_spots} domains. Set n_clusters to the number of spatial domains expected in the tissue."
        )


def select_highly_variable_genes(adata, hvg_flavor: str, n_top_genes: int) -> str:
    """Mark ``adata.var['highly_variable']`` with the flavour the caller chose, or stop.

    ``seurat_v3`` models raw counts, so it runs on ``layers[HVG_COUNTS_LAYER]`` (stashed by the
    caller before normalisation) and needs scikit-misc: :func:`worker_utils.require_hvg_flavor` checks the
    import first and names the package if it is absent. Nothing here switches flavour on failure --
    a substitute flavour changes which 3,000 genes SEDR sees, and the old silent switch meant every
    run in this environment (which has no scikit-misc) reported one flavour and ran another.
    """
    import scanpy as sc

    if hvg_flavor not in HVG_FLAVORS:
        raise ValueError(unsupported_choice_msg("hvg_flavor", hvg_flavor, HVG_FLAVORS))
    hvg_flavor = require_hvg_flavor(hvg_flavor)
    hvg_kwargs = {"layer": HVG_COUNTS_LAYER} if hvg_flavor == "seurat_v3" else {}
    try:
        sc.pp.highly_variable_genes(adata, flavor=hvg_flavor, n_top_genes=n_top_genes, **hvg_kwargs)
    except Exception as e:
        # Broad on purpose: scikit-misc can be present but ABI-broken against this env's numpy
        # ("numpy.dtype size changed"), which raises ValueError, not ImportError. Whatever the cause,
        # the caller's flavour did not run, so the run stops here -- it never switches flavour.
        if hvg_flavor == "seurat_v3":
            raise RuntimeError(
                f"hvg_flavor='seurat_v3' could not run ({type(e).__name__}: {e}). It needs a working "
                "scikit-misc (skmisc.loess): if the error above names skmisc, loess or a numpy dtype size, "
                "that install is unusable here -- reinstall scikit-misc against this environment's numpy, or "
                "pass hvg_flavor='seurat' explicitly. The flavour is not switched for you."
            ) from e
        raise
    return hvg_flavor


def run_sedr_clustering(
    spatial_h5ad: str,
    output_dir: str,
    n_clusters: int,
    using_dec: bool,
    device: str,
    hvg_flavor: str = DEFAULT_HVG_FLAVOR,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    SEDR embedding + KMeans domains, with this worker's own preprocessing.

    The preprocessing normalises the matrix as counts, so it is chosen and checked first by
    ``worker_utils.choose_counts_matrix``: X by default (a negative or non-finite X is refused, a
    non-integer one runs with a warning), ``adata.raw.X`` with ``use_raw_counts=True``.

    The preprocessing is NOT the upstream SEDR tutorial's, and the payload says so under
    ``params.preprocessing``: the tutorial filters genes at min_cells=50/min_counts=10, normalises
    to 1e6 without log, keeps 2,000 seurat_v3 HVGs on counts, scales and feeds a 200-component PCA
    to Sedr. This worker filters at min_cells=3, normalises to 1e4 + log1p, keeps the top-3,000
    HVGs by ``hvg_flavor``, scales (no centering, clip 10) and feeds the full HVG matrix to Sedr --
    so Sedr's input dimensionality is the HVG count (``params.sedr_input_dim``), not 200. The HVG
    matrix is densified because ``SEDR.Sedr`` wraps its input in ``torch.FloatTensor``; that is
    intrinsic to the method.

    Clustering: KMeans(k=n_clusters) on the SEDR latent, always. ``using_dec=True`` runs upstream's
    ``train_with_dec`` (DEC refines the embedding toward ``dec_cluster_n`` KMeans centroids --
    upstream's default of 10, not ``n_clusters``) instead of ``train_without_dec``; SEDR writes no
    labels of its own, so there is no DEC label to publish and the payload names KMeans as the
    method in both cases.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import scanpy as sc
    from sklearn.cluster import KMeans

    os.makedirs(output_dir, exist_ok=True)

    log(f"spatial_h5ad = {spatial_h5ad}")
    log(f"output_dir   = {output_dir}")
    log(f"n_clusters   = {n_clusters}")
    log(f"using_dec    = {using_dec}")
    log(f"hvg_flavor   = {hvg_flavor}")
    log(f"device       = {device}")
    log(f"use_raw_counts = {use_raw_counts}")

    # Preflight checks
    preflight_check(
        inputs={"spatial_h5ad": spatial_h5ad},
        output_dir=output_dir,
    )
    # The HVG flavour and its dependency are checked before the slide is read, so a run that cannot
    # do what it was asked stops in milliseconds instead of after loading and filtering.
    if hvg_flavor not in HVG_FLAVORS:
        raise ValueError(unsupported_choice_msg("hvg_flavor", hvg_flavor, HVG_FLAVORS))
    require_hvg_flavor(hvg_flavor)

    # Load data
    log("Loading spatial AnnData...")
    adata = sc.read_h5ad(spatial_h5ad)
    # Background spots (obs['in_tissue'] == 0 -- CELLxGENE Visium exports ship every array spot, and
    # 56-70% of the library's four such slides are glass) are left out before anything is computed,
    # and counted: clustered, they became domains of their own and used up n_clusters.
    adata, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata, "spots")
    if n_spots_off_tissue:
        log(
            f"Left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0 (background); "
            f"{adata.n_obs} in-tissue spots are analysed"
        )
    # normalize_total + log1p below treat the matrix as counts: a scaled X (negative values) became NaN
    # and died in the HVG binning, and a log-normalised X was normalised a second time without a word.
    adata, counts_info = choose_counts_matrix(adata, use_raw_counts)
    log(f"Expression matrix: adata.{counts_info['expression_source']} ({counts_info['x_matrix_kind']} X)")
    # Gene symbols only, as before; the spot axis is left exactly as supplied. A renamed gene is
    # declared in the payload instead of published as if the user had supplied it.
    renamed = make_names_unique_and_report(adata, axes=("var",))
    log(f"Loaded: {adata.n_obs} spots x {adata.n_vars} genes")
    _validate_n_clusters(n_clusters, int(adata.n_obs))
    # What the user handed us, captured before the preprocessing below cuts the gene axis. The
    # payload has to report both, or a caller reads the survivor count as their own panel.
    n_genes_supplied = int(adata.n_vars)

    # Coordinates are checked up front, with the message the shared helper writes (which keys exist,
    # how many columns) instead of upstream graph_construction's bare AssertionError after the
    # whole preprocessing has run.
    spatial_dims = _check_spatial_coordinates(adata)

    # Upstream's graph construction is O(n^2) in memory; refuse a slide that cannot fit before any
    # work is done, naming the numbers. The feature width is its upper bound (the HVG cap). The
    # loaded matrix is still resident while MemAvailable is read, and is freed by preprocessing.
    dense_budget = _check_dense_budget(
        int(adata.n_obs), min(HVG_N_TOP_GENES, n_genes_supplied), released_bytes=_matrix_nbytes(adata.X)
    )
    log(f"Estimated peak of SEDR's dense intermediates: {dense_budget / 1024.0**3:.2f} GiB")

    _seed_everything(RANDOM_SEED)
    torch_num_threads = _torch_num_threads()
    log(f"torch CPU threads = {torch_num_threads} (a seeded run is bit-reproducible only at 1)")

    # Preprocessing (this worker's own; see the docstring for how it differs from the tutorial)
    log("Preprocessing: filtering, normalizing, log1p, HVG, scaling...")
    sc.pp.filter_genes(adata, min_cells=GENE_MIN_CELLS)
    if hvg_flavor == "seurat_v3":
        # seurat_v3 models raw counts; upstream runs it on layer='count'. Stashed only when asked
        # for, so the default run's memory and output file are exactly what they were.
        adata.layers[HVG_COUNTS_LAYER] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_top_genes = min(HVG_N_TOP_GENES, int(adata.n_vars))
    hvg_flavor = select_highly_variable_genes(adata, hvg_flavor, n_top_genes)
    if HVG_COUNTS_LAYER in adata.layers:
        del adata.layers[HVG_COUNTS_LAYER]
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, zero_center=False, max_value=10)
    log(f"After preprocessing: {adata.n_obs} spots x {adata.n_vars} genes")

    # Ensure X is dense (SEDR wraps X in torch.FloatTensor, which needs a dense array)
    from scipy.sparse import issparse

    if issparse(adata.X):
        log("Converting sparse X to dense for SEDR compatibility...")
        adata.X = adata.X.toarray()

    # Build spatial graph adjacency matrix
    log(f"Building spatial kNN graph (k={KNN_K}) from obsm['spatial'] coordinates...")
    import SEDR

    graph_dict = SEDR.graph_construction(adata, KNN_K)
    log("Graph constructed.")

    # Initialize SEDR model
    sedr_input_dim = int(adata.n_vars)
    log(f"Initializing SEDR model (input_dim={sedr_input_dim})...")
    sedr_net = SEDR.Sedr(
        adata.X,
        graph_dict,
        mode="clustering",
        device=device,
    )

    # Train SEDR
    dec_cluster_n = None
    if using_dec:
        # The model reports how many centroids its DEC layer really has; the constant is only the
        # expectation (upstream's default) for a double that carries no model.
        dec_cluster_n = int(getattr(getattr(sedr_net, "model", None), "dec_cluster_n", DEC_CLUSTER_N_DEFAULT))
        log(f"Training SEDR with Deep Embedding Clustering (DEC, dec_cluster_n={dec_cluster_n})...")
        # ``train_with_dec``'s ``N`` is a mask multiplier, unused in its body -- it was being passed
        # n_clusters, which no code read.
        sedr_net.train_with_dec()
    else:
        log("Training SEDR (without DEC)...")
        sedr_net.train_without_dec()

    # Get embedding
    log("Extracting SEDR embedding...")
    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm["SEDR"] = sedr_feat
    log(f"SEDR embedding shape: {sedr_feat.shape}")

    # Clustering: KMeans on the latent, in both modes. SEDR never writes domain labels itself.
    clustering_method = (
        f"KMeans(k={int(n_clusters)}, random_state={KMEANS_RANDOM_STATE}, n_init={KMEANS_N_INIT}) "
        "on the SEDR latent embedding"
    )
    log(f"Running {clustering_method}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
    clusters = kmeans.fit_predict(sedr_feat)
    adata.obs["sedr_cluster"] = pd.Categorical(clusters.astype(str))
    cluster_key = "sedr_cluster"

    # Summary stats
    unique_clusters = adata.obs[cluster_key].unique()
    n_found = len(unique_clusters)
    cluster_sizes = {str(k): int(v) for k, v in adata.obs[cluster_key].value_counts().items()}
    log(f"SEDR found {n_found} clusters: {list(cluster_sizes.keys())}")

    # Save outputs (each written beside its final name, then moved into place)
    annotated_h5ad = os.path.join(output_dir, "sedr_clustering.h5ad")
    log(f"Writing annotated AnnData to {annotated_h5ad}")
    _write_atomically(annotated_h5ad, adata.write_h5ad)

    clusters_csv = os.path.join(output_dir, "sedr_clusters_per_spot.csv")
    log(f"Writing per-spot cluster assignments to {clusters_csv}")
    _write_atomically(clusters_csv, lambda tmp: adata.obs[[cluster_key]].to_csv(tmp, index_label="spot"))

    embedding_csv = os.path.join(output_dir, "sedr_embedding.csv")
    log(f"Writing SEDR embedding to {embedding_csv}")
    _write_atomically(
        embedding_csv, lambda tmp: pd.DataFrame(sedr_feat, index=adata.obs_names).to_csv(tmp, index_label="spot")
    )

    # Plot spatial domains
    plot_path = os.path.join(output_dir, "sedr_spatial_domains.png")
    log(f"Plotting spatial domains to {plot_path}")
    plot_error = ""
    try:
        sc.pl.spatial(
            adata,
            color=[cluster_key],
            frameon=False,
            title="SEDR spatial domains",
            save=None,
            show=False,
            **_spatial_plot_kwargs(adata),
        )
        _write_atomically(plot_path, lambda tmp: plt.savefig(tmp, format="png", bbox_inches="tight", dpi=300))
    except Exception as e:
        # The domains are already written; a figure that cannot be drawn does not fail the run, but
        # the payload says so (stderr never reaches the caller on a successful run).
        plot_error = f"{type(e).__name__}: {e}"
        if os.path.exists(plot_path):
            # Left in place (it is not this run's to delete), but named, so it is not read as this run's.
            plot_error += f"; the {os.path.basename(plot_path)} already in output_dir is from an earlier run"
        log(f"WARNING: Failed to generate spatial plot: {plot_error}")
        plot_path = None
    finally:
        plt.close("all")

    n_spots = int(adata.n_obs)
    n_genes_used = int(adata.n_vars)

    spot_note = describe_reduction(
        "spots", n_spots_supplied, n_spots, "the in-tissue filter (obs['in_tissue'] == 0 marks background spots)"
    )
    gene_note = describe_reduction(
        "genes",
        n_genes_supplied,
        n_genes_used,
        f"this worker's preprocessing, which drops genes detected in fewer than {GENE_MIN_CELLS} spots "
        f"and then keeps the {HVG_N_TOP_GENES:,} most highly variable (scanpy flavor '{hvg_flavor}')",
    )

    preprocessing = [
        f"filter_genes(min_cells={GENE_MIN_CELLS})",
        "normalize_total(target_sum=1e4)",
        "log1p",
        f"highly_variable_genes(flavor='{hvg_flavor}', n_top_genes={n_top_genes}"
        + (
            f", layer='{HVG_COUNTS_LAYER}' (raw counts stashed before normalize_total)"
            if hvg_flavor == "seurat_v3"
            else ""
        )
        + ")",
        "scale(zero_center=False, max_value=10)",
        f"dense HVG matrix ({n_spots} x {sedr_input_dim}) as the Sedr input; no PCA",
    ]
    if counts_info["expression_source"] == "raw.X":
        preprocessing.insert(0, "counts read from adata.raw.X (use_raw_counts=True)")
    if n_spots_off_tissue:
        preprocessing.insert(
            0, f"keep obs['in_tissue'] == 1 ({n_spots_off_tissue} of {n_spots_supplied} background spots left out)"
        )
    embedding_method = (
        f"SEDR variational graph autoencoder (spatial kNN graph k={KNN_K}"
        + (f"; DEC refinement toward {dec_cluster_n} KMeans centroids" if using_dec else "; no DEC")
        + ")"
    )
    method = f"{embedding_method} + {clustering_method}"

    # Build output
    out = WorkerOutput("sedr", task="clustering")
    out.set_data(n_spots=n_spots_supplied, n_spots_used=n_spots, n_genes=n_genes_supplied, n_genes_used=n_genes_used)
    out.add_output_files(
        {
            "annotated_h5ad": annotated_h5ad,
            "clusters_csv": clusters_csv,
            "embedding_csv": embedding_csv,
            "spatial_plot_png": plot_path,
            "output_dir": output_dir,
        }
    )
    out.add_params(
        {
            "n_clusters": n_clusters,
            "using_dec": using_dec,
            "dec_cluster_n": dec_cluster_n,
            "device": device,
            "hvg_flavor": hvg_flavor,
            "n_top_genes": n_top_genes,
            "knn_k": KNN_K,
            "spatial_dims": spatial_dims,
            "random_seed": RANDOM_SEED,
            "torch_num_threads": torch_num_threads,
            "clustering_method": clustering_method,
            "preprocessing": preprocessing,
            "sedr_input_dim": sedr_input_dim,
            "use_raw_counts": bool(use_raw_counts),
            **identifier_rename_params(renamed),
        }
    )
    record_expression_source(out, counts_info)
    # KMeans is the only clustering this worker has ever run, so it is the method, not a fallback.
    record_method(out, method, used_fallback=False)
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    out.set_summary(
        n_clusters=n_found,
        cluster_key=cluster_key,
        cluster_sizes=cluster_sizes,
        embedding_dim=int(sedr_feat.shape[1]) if len(sedr_feat.shape) > 1 else 0,
    )
    # A successful run never carries stderr into the payload (base_mcp attaches stderr_tail only on a
    # non-zero exit), so a cut this large has to travel in the payload itself.
    if gene_note:
        out.add_warning(gene_note.strip())
    if plot_error:
        out.add_warning(f"the spatial-domain plot was not written ({plot_error}); the domain outputs are complete.")
    method_note = (
        f" Method: {method}. Every domain label is a KMeans label"
        + (
            "; DEC refined the embedding and assigned nothing -- its centroid count is upstream's "
            f"default ({dec_cluster_n}), not n_clusters."
            if using_dec
            else "."
        )
        + f" Preprocessing is this worker's own (not the SEDR tutorial's): {'; '.join(preprocessing)}."
    )
    out.set_analysis(
        build_cluster_analysis(cluster_sizes, cluster_key="domain", total_spots=n_spots, n_requested=int(n_clusters))
        + method_note
        + (" " + counts_info["warning"] if counts_info.get("warning") else "")
        + spot_note
        + gene_note
        + identifier_rename_note(renamed)
    )

    return out.to_dict()


def main() -> None:
    args = parse_args()

    # Redirect all stdout during heavy work to stderr
    orig_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        try:
            device = resolve_device(args.device)
            result = run_sedr_clustering(
                spatial_h5ad=args.spatial_h5ad,
                output_dir=args.output_dir,
                n_clusters=args.n_clusters,
                using_dec=str_to_bool(args.using_dec),
                device=device,
                hvg_flavor=args.hvg_flavor,
                use_raw_counts=bool(args.use_raw_counts),
            )
        except Exception as e:
            log("ERROR while running SEDR:")
            traceback.print_exc(file=sys.stderr)
            result = WorkerOutput.error("sedr", str(e), task="clustering")
    finally:
        sys.stdout = orig_stdout

    # Final JSON to stdout
    print(json.dumps(result))


if __name__ == "__main__":
    main()
