#!/usr/bin/env python
"""
prost_worker.py

Worker for PROST spatial transcriptomics analysis.

- Task "index":   PROST Index (SVG identification)
- Task "domains": PROST PNN (spatial domains)

The official PROST API lives at the *top level* of the package -- ``PROST.prepare_for_PI``,
``PROST.cal_PI``, ``PROST.feature_selection``, ``PROST.run_PNN``. Earlier revisions of this worker
looked for ``PROST.PNN`` and ``PROST.PI`` submodules, which do not exist in any release, so PROST
was never actually called: every run silently produced generic spectral clustering under the PROST
name.

A spatial-coordinate-only spectral substitute still exists for both tasks, but it runs ONLY when
the caller passes ``allow_spectral_fallback=True`` (``--allow-spectral-fallback``). By default a
PROST failure is an error that carries PROST's own exception text, so nothing that is not PROST is
ever reported as PROST unless it was asked for. When the substitute does run, ``params["method"]``
names it, ``params["used_fallback"]`` is True, ``params["used_prost"]`` is False and the analysis
text opens with the warning.

What PROST is given: the expression matrix named by ``layer_key`` (``adata.X`` when empty; a layer
that is not there is an error, never a silent switch to ``X``) and the coordinates named by
``spatial_key``, written to ``obsm['spatial']`` of a fresh AnnData because that is the slot PROST
reads for its PNN cell graph. PROST's ``'visium'``/``'ST'`` presets rasterise ``obs['array_row']``/
``obs['array_col']`` for the gene image when those columns exist (its own Visium convention), and
the ``spatial_key`` coordinates otherwise; ``params["coordinate_source"]`` says which.

Spots flagged ``obs['in_tissue'] == 0`` (background glass, which CELLxGENE Visium exports carry beside
the tissue) are left out right after loading, as ``scanpy_spatial`` always has: PROST would otherwise
score and cluster them as tissue. ``data.n_spots`` is the count supplied, ``data.n_spots_used`` the count
analysed, and ``params.in_tissue_filter`` plus a warning say how many were left out.

Domains follow the published DLPFC pipeline: prepare_for_PI -> cal_PI -> normalize_total + log1p
-> feature_selection(by='prost') -> run_PNN. The normalisation step is controlled by
``pnn_preprocessing`` and expects raw counts; a non-integer matrix is refused with a message
naming the knob rather than silently normalised twice.

PNN's memory grows with the SQUARE of the spot count, and that is intrinsic to PROST's code:
``run_PNN`` densifies the kNN graph (``adj.toarray()``) and trains a dense graph-attention layer on
it (about 56 bytes per spot pair at the peak, measured). PROST also ships ``run_PNN_sparse`` -- same
init, n_clusters and k_neighbors, a sparse graph and a sparse attention layer in the forward pass,
though its backward pass still builds one n x n float32 gradient (about 4-5 bytes per pair).
``pnn_sparse=True`` selects it (off by default: it is a different attention formula, so the default
keeps the published run_PNN). Both are estimated against ``worker_utils.available_memory_bytes()``
BEFORE prepare_for_PI starts, and a run that cannot fit stops there with the numbers and the knob,
instead of failing an hour later, after the whole PI stage, as "install PROST".

This script is meant to be called ONLY by the MCP wrapper:
  /workspace/SpatialOmicsGym/tools/prost_mcp_server.py

All logs go to STDERR, final JSON goes to STDOUT.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import traceback

# Same seed, same input, same box, two runs of the real PROST PNN: exact domain-label agreement
# 0.1718, ARI 0.3453. run_PNN seeds torch/numpy/random and then does dense linear algebra -- PCA of
# the top-gene matrix, Laplacian smoothing against a dense adjacency, KMeans centroids, a torch
# training loop. A multi-threaded BLAS sums each product in whatever order the cores finish, so the
# embedding differs in its last bits between runs; that moves the centroids, and the training
# amplifies it into a different partition. The seed fixes the draws, not the arithmetic.
#
# One thread fixes the summation order, and with it the embedding and the labels: the two runs above
# become byte-identical (ARI 1.0), on any core count. This must happen before numpy/scipy/torch load,
# because OpenBLAS/MKL read the count once at load time. setdefault so an operator profiling
# throughput can still opt out.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    build_cluster_analysis,
    build_svg_analysis,
    choose_counts_matrix,
    describe_reduction,
    expression_matrix_kind,
    keep_in_tissue,
    make_names_unique_and_report,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    spatial_coords,
)

logging.basicConfig(
    level=logging.INFO,
    format="[prost-worker] %(message)s",
    stream=sys.stderr,
)


# ---------------------------------------------------------------------
# Talking to the real PROST
# ---------------------------------------------------------------------

# Verified against the installed PROST 1.1.2: all four are module-level names in the ``PROST``
# package. Requiring the whole set up front means a partial or renamed install is caught here,
# with a message naming what is missing, rather than halfway through an 80-second PI computation.
_PROST_REQUIRED_API = ("prepare_for_PI", "cal_PI", "feature_selection", "run_PNN")

# PROST's own run_PNN default is init="leiden", and with leiden the n_clusters argument is ignored
# -- the cluster count comes from `res` instead. A caller asking for N domains therefore needs an
# init that honours a count. kmeans is pure sklearn; mclust matches the PROST paper but needs R.
_PNN_INITS_HONOURING_A_COUNT = ("kmeans", "mclust")

# The expression preprocessing PROST's own DLPFC tutorial runs between cal_PI and
# feature_selection/run_PNN (``sc.pp.normalize_total(adata); sc.pp.log1p(adata)``). run_PNN's PCA
# reads adata.X directly, so without this step the embedding is a PCA of raw counts. 'none' is for
# a matrix that is already normalised and log-transformed.
_PNN_PREPROCESSING_TUTORIAL = "normalize_log1p"
_PNN_PREPROCESSING_NONE = "none"
_PNN_PREPROCESSING_CHOICES = (_PNN_PREPROCESSING_TUTORIAL, _PNN_PREPROCESSING_NONE)

# PROST's ``platform`` argument is not a description of the tissue -- it selects a code path.
# prepare_for_PI reads the spot locations (obs[['array_row','array_col']], falling back to
# obsm['spatial'] under a bare ``except:``), shifts by one when the minimum is exactly zero, and
# hands them to make_image, which branches on ``platform == "visium"`` alone and rasterises them
# onto an integer lattice:
#
#     xloc = np.round(locates[:, 0]).astype(int); maxx = np.max(xloc)
#     image = np.zeros((maxy, maxx)); image[temp_y - 1, temp_x - 1] = temp_value
#
# which needs every rounded coordinate to be at least 1. np.zeros raises "negative dimensions are
# not allowed" when the maximum is negative, and any single spot rounding to zero or less is
# written to the opposite edge of the gene image instead of its own position. Every other platform
# name takes the second branch, which interpolates onto a regular grid with griddata and is fine
# with negative coordinates. Upstream offers the argument as "'visium', 'Slide-seq', 'Stereo-seq',
# 'osmFISH', 'SeqFISH' or other platform that generate irregular spots", and no PROST stage looks
# the name up -- every use is an ``==`` comparison -- so an unlisted name is sanctioned and means
# "not a lattice".
_PLATFORM_LATTICE = "visium"
_PLATFORM_IRREGULAR = "irregular"

# The lattice preset also needs neighbouring spots to be neighbouring PIXELS. cal_PI smooths each
# gene image with gaussian_filter(sigma=1, truncate=2) -- a 2-pixel radius -- and get_sub closes
# it with a 5-pixel kernel before labelling connected foreground. On a Visium array lattice
# (obs['array_row']/['array_col']) the nearest spots are 1-2 apart; full-resolution pixel
# coordinates put them tens to hundreds apart, so no two spots ever share a neighbourhood, every
# spot is its own component, and the index has no spatial structure left to measure. Anything
# above this spacing is provably not the lattice the preset was written for.
_LATTICE_MAX_SPACING = 4.0

_METHOD_PROST_PNN = "PROST PNN"
_METHOD_PROST_PNN_SPARSE = (
    "PROST PNN, sparse variant (PROST.run_PNN_sparse: sparse kNN graph and sparse graph-attention layer)"
)

# Peak resident memory of PROST 1.1.2's PNN per spot PAIR, measured on its own env (torch 1.8.1, CPU,
# kmeans init, 60 genes; n = 4,000 .. 12,000 spots, where the per-pair figure had settled):
#   run_PNN         ~56 bytes/pair: the kNN graph densified to a float64 numpy array (8), again as a
#                   float32 torch tensor (4), and the dense attention layer's n x n score, mask,
#                   softmax and dropout tensors kept for backward -- twice, because the training loop
#                   holds the previous forward's graph while it runs the next one;
#   run_PNN_sparse  ~4-5 bytes/pair: the forward pass is sparse, but the backward pass of its
#                   SpecialSpmmFunction (pyGAT's) materialises one n x n float32 gradient.
# At 507,684 spots (the library's VisiumHD sample) that is ~13 TiB and ~1.2 TiB.
_PNN_DENSE_BYTES_PER_PAIR = 56
_PNN_SPARSE_BYTES_PER_PAIR = 5
# Linear part: run_PNN's PCA densifies the selected-gene matrix (adata.X.A) and centres a copy.
_PNN_PCA_DENSE_COPIES = 3
_METHOD_PROST_INDEX = "PROST Index (PI)"
_METHOD_SUBSTITUTE_DOMAINS = (
    "spatial-coordinate-only spectral clustering (KMeans on the eigenvectors of a kNN-graph "
    "Laplacian built from the spot coordinates; gene expression not used; PROST substitute)"
)
_METHOD_SUBSTITUTE_SVG = (
    "spectral SVG score, SpaGFT-style (fraction of each gene's energy in the low-frequency "
    "eigenvectors of a kNN-graph Laplacian built from the spot coordinates; PROST substitute)"
)


class ProstUnavailable(RuntimeError):
    """Raised when PROST cannot be driven and the caller did not allow a substitute."""


class ProstMemoryError(MemoryError):
    """PROST cannot fit in memory (PNN at this spot count, or a stage that ran out).

    The message carries the numbers and the knob; never "install PROST".
    """


def _available_memory_bytes():
    """Memory this process can still allocate, or None when nothing can be read.

    The fleet's one reader (``worker_utils.available_memory_bytes``): the smaller of MemAvailable and the
    room left under the cgroup memory limit, page cache counted as reclaimable. A module-level seam so a
    test can stand in for the box.
    """
    return available_memory_bytes()


def _gib(n_bytes) -> str:
    return f"{float(n_bytes) / float(1 << 30):,.1f} GiB"


def _pnn_memory_estimate(n_spots: int, n_genes: int) -> dict:
    """Peak bytes of PROST's two PNN variants for ``n_spots`` spots and at most ``n_genes`` genes."""
    pairs = float(n_spots) * float(n_spots)
    linear = float(_PNN_PCA_DENSE_COPIES) * float(n_spots) * float(max(int(n_genes), 1)) * 8.0
    return {
        "run_PNN": int(_PNN_DENSE_BYTES_PER_PAIR * pairs + linear),
        "run_PNN_sparse": int(_PNN_SPARSE_BYTES_PER_PAIR * pairs + linear),
    }


def _check_pnn_memory(n_spots: int, n_genes: int, pnn_sparse: bool) -> dict:
    """Estimate PNN's peak before any PROST stage runs; raise :class:`ProstMemoryError` if it cannot fit.

    Returns the record published as ``params.pnn_memory``. ``None`` available (nothing readable) skips
    the refusal: the estimate is still reported.
    """
    estimate = _pnn_memory_estimate(n_spots, n_genes)
    function = "run_PNN_sparse" if pnn_sparse else "run_PNN"
    available = _available_memory_bytes()
    record = {
        "function": function,
        "n_spots": int(n_spots),
        "estimated_bytes": int(estimate[function]),
        "run_PNN_estimated_bytes": int(estimate["run_PNN"]),
        "run_PNN_sparse_estimated_bytes": int(estimate["run_PNN_sparse"]),
        "available_bytes": None if available is None else int(available),
    }
    if available is None or estimate[function] <= available:
        return record
    if not pnn_sparse:
        sparse = estimate["run_PNN_sparse"]
        alternative = (
            f"pnn_sparse=True runs PROST's own run_PNN_sparse instead (the same init, n_domains and "
            f"pnn_k_neighbors on a sparse graph and a sparse attention layer; its backward pass still builds "
            f"one {n_spots} x {n_spots} float32 gradient), estimated at {_gib(sparse)}, which "
            + ("fits." if sparse <= available else "does not fit either; run on a machine with more memory.")
        )
        raise ProstMemoryError(
            f"PROST PNN (run_PNN) cannot fit in memory for {n_spots} in-tissue spots: it densifies the "
            f"{n_spots} x {n_spots} spot graph (a float64 numpy copy and a float32 torch copy) and trains a "
            f"dense graph-attention layer whose n x n tensors are kept for the backward pass -- about "
            f"{_gib(estimate['run_PNN'])} at the peak (~{_PNN_DENSE_BYTES_PER_PAIR} bytes per spot pair, "
            f"measured on PROST 1.1.2), and {_gib(available)} is available. This was checked before "
            f"prepare_for_PI, so nothing was computed. {alternative}"
        )
    raise ProstMemoryError(
        f"PROST PNN (run_PNN_sparse, pnn_sparse=True) cannot fit in memory for {n_spots} in-tissue spots: its "
        f"sparse graph-attention layer's backward pass builds one {n_spots} x {n_spots} float32 gradient -- "
        f"about {_gib(estimate['run_PNN_sparse'])} at the peak (~{_PNN_SPARSE_BYTES_PER_PAIR} bytes per spot "
        f"pair, measured on PROST 1.1.2), and {_gib(available)} is available. This was checked before "
        f"prepare_for_PI, so nothing was computed. Run on a machine with more memory (the dense run_PNN, "
        f"pnn_sparse=False, needs about {_gib(estimate['run_PNN'])})."
    )


def _looks_out_of_memory(exc: BaseException) -> bool:
    """numpy's MemoryError, or torch's CPU allocator failure (a RuntimeError naming the allocation)."""
    if isinstance(exc, MemoryError):
        return True
    text = str(exc)
    return "can't allocate memory" in text or "Cannot allocate memory" in text or "not enough memory" in text


def _import_prost():
    """Return ``(PROST module, None)`` if its real API is present, else ``(None, reason)``.

    Reports *which* names are missing. The defect this replaces was a silent ``getattr(PROST,
    "PNN", None)`` returning None forever, so the diagnostic matters more than the brevity. The
    reason travels with the result so a refusal can quote it instead of a generic "unavailable".
    """
    try:
        import PROST  # type: ignore
    except Exception as exc:  # ImportError, but a broken install can raise anything
        reason = f"PROST is not importable in this environment ({type(exc).__name__}: {exc})"
        logging.warning(reason)
        return None, reason

    missing = [name for name in _PROST_REQUIRED_API if not callable(getattr(PROST, name, None))]
    if missing:
        reason = (
            f"PROST imported but is missing the top-level callables {missing}; the official workflow cannot be driven"
        )
        logging.warning(reason)
        return None, reason
    version = str(getattr(PROST, "__version__", "unknown")).strip()
    logging.info(f"PROST {version} found with the full top-level API.")
    return PROST, None


def _refuse_or_warn(
    allow_spectral_fallback: bool, what: str, why: str, out_of_memory: bool = False, remedy: str = ""
) -> None:
    """Either raise with PROST's own failure text, or log loudly that a substitute is about to run.

    A run that failed for lack of memory is refused as a ``MemoryError`` whose remedy is the memory knob
    (``remedy``), not "install PROST": the package was there and running when it ran out.
    """
    if not allow_spectral_fallback and out_of_memory:
        raise ProstMemoryError(
            f"PROST ran out of memory for {what}: {why}. allow_spectral_fallback is False (the default), so no "
            "substitute ran and nothing is reported under the PROST name. "
            + (remedy or "Run on a machine with more memory.")
        )
    if not allow_spectral_fallback:
        raise ProstUnavailable(
            f"PROST could not be run for {what}: {why}. allow_spectral_fallback is False (the default), "
            "so no substitute ran and nothing is reported under the PROST name. Fix the cause above "
            "(install PROST in this worker's environment, or correct the input it rejected), or pass "
            "allow_spectral_fallback=True to accept a clearly labelled spatial-coordinate-only spectral "
            "substitute in place of PROST."
        )
    logging.warning(
        "SUBSTITUTE ALGORITHM: PROST did not run for %s (%s). allow_spectral_fallback=True, so the results "
        "below come from a generic spectral method on the spot coordinates, not from PROST, and are "
        "labelled as such in params['method'] and params['used_fallback'].",
        what,
        why,
    )


def _substitute_note(real_method: str, substitute: str, why: str) -> str:
    """One sentence, front-loaded, for the analysis text the agent quotes to the user."""
    return (
        f"IMPORTANT: {real_method} did NOT run ({why}). These results were produced by a substitute "
        f"algorithm -- {substitute} -- which the caller permitted with allow_spectral_fallback=True. "
        f"Do not report them as {real_method} output. "
    )


def _prost_lattice_indices(adata) -> np.ndarray | None:
    """The image indices PROST's Visium branch would write to, or ``None`` if it can read none.

    Transcribed from PROST 1.1.2 (``calculate_PI.prepare_for_PI`` -> ``utils.make_image``) so the
    guard below cannot drift from the arithmetic it guards. Non-finite rows are dropped first: a
    NaN coordinate is a different failure, with its own message, and rounding one to an integer
    would otherwise read as a huge negative index here.
    """
    if {"array_row", "array_col"} <= set(adata.obs.columns):
        locates = adata.obs[["array_row", "array_col"]].values
    elif "spatial" in adata.obsm:
        locates = adata.obsm["spatial"]
        locates = locates.values if isinstance(locates, pd.DataFrame) else locates
    else:
        # prepare_for_PI's lattice branch would raise reading obsm['spatial']; that is a separate
        # failure and not ours to pre-empt.
        return None
    try:
        locates = np.asarray(locates, dtype=float)
    except (TypeError, ValueError):
        return None  # non-numeric obs columns; PROST would raise on its own terms
    if locates.ndim != 2 or locates.shape[1] < 2:
        return None
    locates = locates[np.isfinite(locates).all(axis=1)]
    if locates.shape[0] == 0:
        return None
    if np.min(locates) == 0:  # prepare_for_PI's own one-based shift
        locates = locates + 1
    return np.round(locates[:, :2]).astype(int) - 1


def _lattice_spacing(idx: np.ndarray) -> float:
    """Median distance from each rasterised spot to its nearest other spot, in image pixels."""
    if idx.shape[0] < 2:
        return 0.0
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(idx.astype(float))
    distances, _ = nbrs.kneighbors(idx.astype(float))
    return float(np.median(distances[:, 1]))


def _resolve_platform(adata, platform: str) -> tuple[str, str | None]:
    """Return the platform to drive PROST with, and a note when it is not the one asked for.

    Deliberately narrow: a non-lattice platform the caller named is theirs to choose and is never
    second-guessed, and ``'visium'`` -- the default, which an explicit ``'visium'`` cannot be told
    apart from -- is replaced only where the lattice path is provably unusable: a spot landing at a
    negative image index, which either raises inside ``np.zeros`` or writes that spot to the
    opposite edge of the image; or spots so far apart in pixels that PROST's 2-pixel-radius
    smoothing and 5-pixel closing can never join two of them, which leaves the index nothing
    spatial to measure (and, on full-resolution pixel coordinates, allocates gene images of
    hundreds of millions of pixels each). The note returned says which, for ``analysis``.
    """
    if platform != _PLATFORM_LATTICE:
        return platform, None
    idx = _prost_lattice_indices(adata)
    if idx is None:
        return platform, None
    if int(idx.min()) < 0:
        n_off = int((idx < 0).any(axis=1).sum())
        note = (
            f"Platform: this slide cannot form the integer lattice PROST's default "
            f"platform='{_PLATFORM_LATTICE}' preset requires -- {n_off} of {idx.shape[0]} spots map to "
            f"a negative image index (lowest {int(idx.min())}), where PROST either raises 'negative "
            f"dimensions are not allowed' or writes the spot to the opposite edge of the gene image. "
            f"PROST was run with platform='{_PLATFORM_IRREGULAR}' instead, which interpolates the "
            f"expression onto a regular grid (grid_size spacing) rather than rasterising it. The "
            f"'{_PLATFORM_LATTICE}' preset is replaced on these coordinates even when passed explicitly, "
            f"because it cannot run on them; every other platform name takes the same interpolating path."
        )
        return _PLATFORM_IRREGULAR, note
    spacing = _lattice_spacing(idx)
    if spacing > _LATTICE_MAX_SPACING:
        note = (
            f"Platform: PROST's default platform='{_PLATFORM_LATTICE}' preset rasterises each spot onto one "
            f"pixel of an integer lattice and then smooths every gene image with a 1-pixel Gaussian and a "
            f"5-pixel closing kernel; on this slide the nearest spots are {spacing:.0f} pixels apart after "
            f"rounding (a Visium array lattice puts them 1-2 apart), so no two spots would ever share a "
            f"neighbourhood and the index would measure no spatial structure. PROST was run with "
            f"platform='{_PLATFORM_IRREGULAR}' instead, which interpolates the expression onto a regular "
            f"grid (grid_size spacing). The '{_PLATFORM_LATTICE}' preset is replaced on these coordinates "
            f"even when passed explicitly, and every other platform name takes the same interpolating path; "
            f"to use the lattice preset on a Visium slide, supply obs['array_row']/obs['array_col'], the "
            f"array coordinates it expects."
        )
        return _PLATFORM_IRREGULAR, note
    return platform, None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _load_adata(path: str) -> ad.AnnData:
    logging.info(f"Loading AnnData from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"st_h5ad not found: {path}")
    adata = ad.read_h5ad(path)
    # A ranked gene list is this tool's whole deliverable, so it must not name one gene twice.
    # Two Ensembl IDs collapsing onto one HGNC symbol is routine in a CellRanger matrix, and
    # PROST scores both rows independently -- a top-N would otherwise return fewer than N
    # distinct genes, with one symbol carrying two different indices. The helper records the
    # count in .uns so the payload can say which names this run generated.
    make_names_unique_and_report(adata)
    logging.info(f"Loaded AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    return adata


def _load_in_tissue(path: str):
    """``(adata, n_spots_supplied, n_spots_off_tissue)``: the slide with background spots left out.

    ``obs['in_tissue'] == 0`` marks array spots outside the tissue. On the library's four CELLxGENE
    Visium samples they are 56-70% of the spots and not empty (Heart: median 968 counts), so PROST
    turned them into gene-image pixels and PNN domains and they used up the requested n_domains. They
    are left out by default, and counted, exactly as every other spot-level worker now does.
    """
    adata = _load_adata(path)
    adata, n_supplied, n_off = keep_in_tissue(adata, "spots")
    if n_off:
        logging.info(
            f"Left out {n_off} of {n_supplied} spots with obs['in_tissue'] == 0 (background); "
            f"{adata.n_obs} in-tissue spots are analysed."
        )
    return adata, int(n_supplied), int(n_off)


def _in_tissue_note(n_supplied: int, n_off: int) -> str:
    """The analysis sentence for spots the in-tissue filter left out ('' when none were)."""
    return describe_reduction(
        "spots",
        n_supplied,
        n_supplied - n_off,
        "the in-tissue filter (obs['in_tissue'] == 0 marks background outside the tissue)",
    )


def _seeded_start(n: int, seed: int) -> np.ndarray:
    """ARPACK's start vector, drawn from ``seed``.

    ``eigsh`` without ``v0`` starts from ARPACK's own internal random vector, which numpy's seed does
    not reach -- so the ``np.random.seed(seed)`` the substitute used to call changed nothing it
    computed, while ``params.seed`` said it had. Passing the start vector makes the seed the one that
    decides it.
    """
    return np.random.RandomState(int(seed)).uniform(-1.0, 1.0, size=int(n))


def _expression_matrix(adata: ad.AnnData, layer_key: str | None):
    """``(matrix, source)`` -- the expression the caller named, sparse or dense as stored.

    Never densified here: PROST densifies inside its own stages (``adata.X.A``), which is intrinsic
    to its image-based index, and a second dense copy held by the worker for the whole run doubled
    the footprint for nothing. A ``layer_key`` that is not there is an error naming the layers the
    file does have -- switching to ``X`` behind the caller's back reported a result from a matrix
    they did not choose.
    """
    if layer_key:
        if layer_key not in adata.layers:
            raise KeyError(
                f"layer_key='{layer_key}' is not in adata.layers (available: {list(adata.layers.keys())}). "
                "Name a layer the file has, or leave layer_key empty to use adata.X."
            )
        logging.info(f"Using adata.layers['{layer_key}'] as expression")
        return adata.layers[layer_key], f"layers['{layer_key}']"
    if adata.X is None:
        raise ValueError(
            f"adata.X is empty; pass layer_key naming an expression layer (available: {list(adata.layers.keys())})."
        )
    logging.info("Using adata.X as expression")
    return adata.X, "X"


def _detection_fraction(X) -> np.ndarray:
    """Fraction of spots with expression > 0 per gene, without densifying a sparse matrix."""
    if sp.issparse(X):
        return np.asarray((X > 0).sum(axis=0)).ravel() / float(X.shape[0])
    return np.asarray((np.asarray(X) > 0).mean(axis=0)).ravel()


def _first_non_count(X) -> float | None:
    """The first stored value of ``X`` that is not a non-negative integer, or ``None``.

    Scanned in row blocks and stopped at the first hit, so checking a whole-transcriptome dense
    matrix costs one block of scratch memory rather than two more copies of the matrix.
    """
    if sp.issparse(X):
        values = np.asarray(X.data)
        blocks = [values[i : i + 1_000_000] for i in range(0, values.size, 1_000_000)]
    else:
        arr = np.asarray(X)
        step = max(1, 1_000_000 // max(1, arr.shape[1] if arr.ndim == 2 else 1))
        blocks = [arr[i : i + step] for i in range(0, arr.shape[0], step)]
    for block in blocks:
        block = np.asarray(block, dtype=np.float64).ravel()
        bad = (block != np.floor(block)) | (block < 0) | ~np.isfinite(block)
        if bad.any():
            return float(block[np.flatnonzero(bad)[0]])
    return None


def _require_counts(X, source: str, layers, raw=None) -> None:
    """Refuse to log-normalise a matrix that is not raw counts.

    ``pnn_preprocessing='normalize_log1p'`` reproduces the tutorial's ``normalize_total`` +
    ``log1p`` on raw counts. Applied to a matrix that is already normalised it would normalise
    twice and log a log; guessing which case a non-integer matrix is would be the silent switch
    this worker exists to avoid, so the caller is asked. ``raw`` is the file's ``adata.raw``: when it
    holds counts the message names ``use_raw_counts=True`` (CELLxGENE exports keep their counts there,
    where no ``layer_key`` can reach them).
    """
    example = _first_non_count(X)
    if example is None:
        return
    raw_hint = ""
    if raw is not None and expression_matrix_kind(raw.X) == "counts":
        raw_hint = " adata.raw holds raw counts: pass use_raw_counts=True to run on them."
    raise ValueError(
        f"pnn_preprocessing='{_PNN_PREPROCESSING_TUTORIAL}' runs PROST's tutorial preprocessing "
        f"(sc.pp.normalize_total + sc.pp.log1p) on raw counts, but the expression source {source} holds values "
        f"that are not non-negative integers (for example {example:g}), so it is not a raw count matrix. Point "
        f"layer_key at a raw-count layer (available layers: {list(layers)}), or pass "
        f"pnn_preprocessing='{_PNN_PREPROCESSING_NONE}' to give PNN the matrix exactly as stored (use that when it "
        "is already normalised and log-transformed; for non-integer values that are not yet logged, e.g. "
        "volume-normalised MERFISH, write normalize_total + log1p into a layer yourself and pass it as "
        f"layer_key with pnn_preprocessing='{_PNN_PREPROCESSING_NONE}')." + raw_hint
    )


def _get_spatial_coords(adata: ad.AnnData, spatial_key: str) -> np.ndarray:
    # Through the shared guard rather than a bare read: PROST's kNN graph and its rasterisation
    # both take the first two columns of whatever this returns, so a three-column key would put
    # every section of a serial stack at the same plane and the run would report spatially
    # variable genes on a tissue several sections thick.
    coords, _ = spatial_coords(adata, spatial_key, want=2, tool="PROST")
    coords = coords.astype(np.float32)
    if coords.shape[0] != adata.n_obs:
        raise ValueError(f"Spatial coordinates have shape {coords.shape}, but n_obs={adata.n_obs}")
    return coords


def _prost_input(adata: ad.AnnData, X, coords: np.ndarray, var=None) -> ad.AnnData:
    """The AnnData PROST is driven with: the named expression in ``X``, the named coordinates in
    ``obsm['spatial']``.

    PROST hard-codes both slots -- ``pre_process``/``run_PNN`` read ``adata.X``, ``get_adj`` reads
    ``obsm['spatial']`` -- so ``layer_key`` and ``spatial_key`` can only be honoured by building the
    object it expects. ``obs`` is carried whole because the ``'visium'``/``'ST'`` presets rasterise
    ``obs['array_row']``/``obs['array_col']`` when present (PROST's own Visium convention; those are
    the lattice indices the preset needs, where ``obsm['spatial']`` is usually pixels). Both slots are
    private copies: ``prepare_for_PI`` does ``locates += 1`` in place on ``obsm['spatial']`` when the
    minimum is zero, and the caller's object is written back out as the annotated result. ``var`` is
    the gene table of ``X`` when it is not ``adata``'s own (``adata.raw.var`` with use_raw_counts).
    """
    pin = ad.AnnData(X=X.copy(), obs=adata.obs.copy(), var=(adata.var if var is None else var).copy())
    pin.obsm["spatial"] = np.array(coords, dtype=np.float64)
    return pin


def _coordinate_source(adata: ad.AnnData, spatial_key: str, prost_platform: str, pnn: bool) -> dict:
    """Which coordinates fed which PROST stage, for ``params`` -- the two can legitimately differ.

    The PROST Index reads coordinates only to build its gene images; PNN additionally builds its
    cell graph from ``obsm['spatial']`` (``get_adj``), which is always the ``spatial_key`` array.
    """
    graph = f"obsm['{spatial_key}']"
    lattice_cols = {"array_row", "array_col"} <= set(adata.obs.columns)
    if prost_platform in (_PLATFORM_LATTICE, "ST") and lattice_cols:
        image = "obs['array_row','array_col'] (PROST's Visium lattice columns, read by its lattice presets)"
    else:
        image = graph
    return {"pnn_graph": graph, "gene_image": image} if pnn else {"gene_image": image}


def _atomic_write(path: str, write) -> None:
    """``write(tmp)`` then ``os.replace``: a reader never sees a half-written result file."""
    tmp = path + ".partial"
    write(tmp)
    os.replace(tmp, path)


def _build_knn_graph(coords: np.ndarray, n_neighbors: int):
    """
    Build a symmetric kNN graph (no self-loops) and its normalized Laplacian.
    """
    n = coords.shape[0]
    logging.info(f"Building kNN graph with n_neighbors={n_neighbors}, n={n}")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="kd_tree")
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # Exclude self (index 0 in each row)
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = indices[:, 1:].reshape(-1)
    vals = np.ones_like(cols, dtype=np.float64)

    W = coo_matrix((vals, (rows, cols)), shape=(n, n))
    W = (W + W.T).tocsr()  # symmetrize

    d = np.array(W.sum(axis=1)).flatten()
    d_safe = np.clip(d, 1e-12, None)
    D_inv_sqrt = diags(1.0 / np.sqrt(d_safe))

    L = diags(np.ones(n)) - D_inv_sqrt @ W @ D_inv_sqrt
    return W, L


def _check_pnn_init(pnn_init: str) -> None:
    if pnn_init not in _PNN_INITS_HONOURING_A_COUNT:
        raise ValueError(
            f"pnn_init={pnn_init!r} is not supported: PROST's run_PNN ignores n_clusters unless init is one of "
            f"{list(_PNN_INITS_HONOURING_A_COUNT)}, so n_domains would be silently discarded. Use 'kmeans' or "
            "'mclust'."
        )


def _check_pnn_preprocessing(pnn_preprocessing: str) -> None:
    if pnn_preprocessing not in _PNN_PREPROCESSING_CHOICES:
        raise ValueError(
            f"pnn_preprocessing={pnn_preprocessing!r} is not one of {list(_PNN_PREPROCESSING_CHOICES)}: "
            f"'{_PNN_PREPROCESSING_TUTORIAL}' runs sc.pp.normalize_total + sc.pp.log1p on raw counts as PROST's "
            f"tutorial does; '{_PNN_PREPROCESSING_NONE}' feeds the matrix to PNN as stored."
        )


def _pnn_preprocess(adata_pnn, pnn_preprocessing: str):
    """The tutorial's expression preprocessing between cal_PI and feature_selection. ``(adata, steps)``."""
    if pnn_preprocessing == _PNN_PREPROCESSING_NONE:
        logging.info("pnn_preprocessing='none': PNN reads the expression matrix as stored.")
        return adata_pnn, []
    import scanpy as sc

    logging.info("PNN preprocessing (PROST tutorial): sc.pp.normalize_total + sc.pp.log1p.")
    sc.pp.normalize_total(adata_pnn)
    sc.pp.log1p(adata_pnn)
    return adata_pnn, ["sc.pp.normalize_total", "sc.pp.log1p"]


# ---------------------------------------------------------------------
# PROST Index (SVG) -- real PROST; a labelled spectral substitute only when allowed
# ---------------------------------------------------------------------


def run_prost_index(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    layer_key: str | None = None,
    n_neighbors: int = 20,
    n_eigs: int = 50,
    low_freq_fraction: float = 0.1,
    min_detected_frac: float = 0.05,
    n_top_genes: int = 200,
    seed: int = 0,
    platform: str = "visium",
    allow_spectral_fallback: bool = False,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    adata, n_spots_supplied, n_off_tissue = _load_in_tissue(st_h5ad)
    X, expression_source = _expression_matrix(adata, layer_key)
    coords = _get_spatial_coords(adata, spatial_key)

    # Detection mask for min_detected_frac. It shapes the spectral substitute's matrix ONLY: the
    # official branch below hands prepare_for_PI the whole matrix, and PROST runs its own gene
    # selection instead -- sc.pp.filter_genes(min_cells=50), an absolute spot COUNT rather than
    # our fraction. Measured against PROST 1.1.2 on a 400-spot slide with a smooth detection
    # gradient, min_detected_frac=0.5 excluded 122 genes that PROST scored anyway (0.8 -> 213).
    # So neither the "retained" log nor the empty-mask abort belongs here; both live in the
    # substitute now, where the mask is actually consumed. Putting the abort here also refused
    # runs PROST could have completed, on a threshold that would never have reached it.
    det_frac = _detection_fraction(X)
    keep = det_frac >= min_detected_frac
    genes_keep = np.asarray(adata.var_names)[keep]

    used_prost = False
    df_scores = None
    method = _METHOD_PROST_INDEX
    prost_failure = None
    prost_out_of_memory = False
    coordinate_source = None

    # ----- Official PROST Index -----
    # prepare_for_PI populates uns['nor_counts'] / ['subregions'] / ['del_index'] and subsets genes;
    # cal_PI then runs minmax_scaler -> gau_filter -> get_binary -> get_sub -> cal_prost_index and
    # writes var['PI'], var['SEP'], var['SIG']. Calling cal_prost_index directly raises
    # KeyError: 'nor_counts' -- confirmed on the real 1.1.2 package.
    prost, prost_failure = _import_prost()
    # Assigned unconditionally: params and the analysis text below read them on every route.
    prost_platform, platform_note = platform, None
    if prost is not None:
        pin = _prost_input(adata, X, coords)
        # platform threads on into cal_PI's gau_filter/get_binary/get_sub, so both calls have to
        # be given the same answer.
        prost_platform, platform_note = _resolve_platform(pin, platform)
        if platform_note:
            logging.warning(platform_note)
        coordinate_source = _coordinate_source(adata, spatial_key, prost_platform, pnn=False)
        try:
            logging.info(f"Running official PROST Index (platform={prost_platform}).")
            adata_pi = prost.prepare_for_PI(pin, platform=prost_platform)
            adata_pi = prost.cal_PI(adata_pi, platform=prost_platform)
            if "PI" not in adata_pi.var:
                raise RuntimeError("PROST cal_PI did not write var['PI']")
            df_scores = pd.DataFrame(
                {
                    "gene": np.asarray(adata_pi.var_names),
                    "prost_index": np.asarray(adata_pi.var["PI"], dtype=float),
                }
            )
            # SEP (separability) and SIG (significance) are the two factors PI is built from; they
            # are what a reader needs to judge a borderline gene, so carry them when present.
            for extra in ("SEP", "SIG"):
                if extra in adata_pi.var:
                    df_scores[extra.lower()] = np.asarray(adata_pi.var[extra], dtype=float)
            df_scores = df_scores.sort_values("prost_index", ascending=False).reset_index(drop=True)
            used_prost = True
            logging.info(f"PROST Index scored {len(df_scores)} genes.")
            logging.info(
                "min_detected_frac=%.3f was NOT applied: PROST's own prepare_for_PI gene selection "
                "chose those %d genes. The threshold governs the spectral substitute only.",
                min_detected_frac,
                len(df_scores),
            )
        except Exception as e:
            prost_failure = f"official PROST Index failed with {type(e).__name__}: {e}"
            prost_out_of_memory = _looks_out_of_memory(e)
            logging.warning("Official PROST Index failed (%s: %s).", type(e).__name__, e)
            traceback.print_exc(file=sys.stderr)
            df_scores = None

    if not used_prost:
        _refuse_or_warn(
            allow_spectral_fallback, "SVG identification", str(prost_failure), out_of_memory=prost_out_of_memory
        )
        method = _METHOD_SUBSTITUTE_SVG

    # ----- Substitute (only when allowed): spectral SVG score (SpaGFT-style) -----
    if not used_prost:
        logging.info("Running the spectral SVG substitute (SpaGFT-style).")

        # The substitute is the one consumer of the mask, so this is where the threshold is real --
        # and where an empty mask leaves nothing to score. Reached either because PROST is not
        # importable or because the official branch above raised, so it cannot be hoisted.
        logging.info(f"Filtering genes with detected_fraction >= {min_detected_frac:.3f}")
        if int(keep.sum()) == 0:
            raise RuntimeError("No genes pass the min_detected_frac filter; please lower the threshold.")
        X_keep = X[:, np.flatnonzero(keep)]
        logging.info(f"Retained {X_keep.shape[1]} genes out of {X.shape[1]} after detection filtering.")

        W, L = _build_knn_graph(coords, n_neighbors=n_neighbors)

        logging.info(f"Computing {n_eigs} smallest eigenpairs of normalized Laplacian (start vector seed={seed}).")
        evals, evecs = eigsh(L, k=n_eigs, which="SM", v0=_seeded_start(L.shape[0], seed))
        logging.info("First 10 eigenvalues: %s", evals[:10])

        K = max(1, int(low_freq_fraction * n_eigs))
        logging.info(f"Using K={K} low-frequency components out of n_eigs={n_eigs}")

        # Project the centred gene vectors onto the eigenvectors without densifying:
        # evecs.T @ (X - 1 mean^T) == evecs.T @ X - (evecs.T @ 1) mean^T.
        mean = np.asarray(X_keep.mean(axis=0), dtype=np.float64).ravel()
        if sp.issparse(X_keep):
            proj = np.asarray((X_keep.T @ evecs).T, dtype=np.float64)  # shape: (n_eigs, n_genes_keep)
        else:
            proj = evecs.T @ np.asarray(X_keep, dtype=np.float64)
        proj = proj - np.outer(evecs.sum(axis=0), mean)
        energy = (proj**2).sum(axis=0) + 1e-12
        low_energy = (proj[:K] ** 2).sum(axis=0)
        score = low_energy / energy

        df_scores = pd.DataFrame(
            {
                "gene": genes_keep,
                "prost_svg_score": score,
                "detected_fraction": det_frac[keep],
            }
        ).sort_values("prost_svg_score", ascending=False)

        # Save Laplacian spectrum for debugging/inspection
        spec_path = os.path.join(output_dir, "prost_laplacian_eigenvalues.csv")
        _atomic_write(spec_path, lambda p: pd.DataFrame({"eigenvalue": evals}).to_csv(p, index=False))
        logging.info(f"Saved Laplacian eigenvalues to {spec_path}")

    # ----- Save outputs (both routes converge here) -----
    # Detect score column
    score_cols = [c for c in df_scores.columns if "prost" in c.lower() or "score" in c.lower() or "svg" in c.lower()]
    if not score_cols:
        # Fallback: assume first numeric column after "gene"
        num_cols = [c for c in df_scores.columns if c != "gene" and np.issubdtype(df_scores[c].dtype, np.number)]
        if not num_cols:
            raise RuntimeError("Could not identify a numeric score column in PROST output.")
        score_col = num_cols[0]
    else:
        score_col = score_cols[0]

    all_scores_path = os.path.join(output_dir, "prost_index_all_gene_scores.csv")
    top_scores_path = os.path.join(output_dir, "prost_top_svg_genes.csv")

    _atomic_write(all_scores_path, lambda p: df_scores.to_csv(p, index=False))
    logging.info(f"Saved all gene scores to {all_scores_path}")

    df_top = df_scores.sort_values(score_col, ascending=False).head(n_top_genes)
    df_top = df_top.copy()
    df_top["rank"] = np.arange(1, df_top.shape[0] + 1)
    _atomic_write(top_scores_path, lambda p: df_top.to_csv(p, index=False))
    logging.info(f"Saved top {df_top.shape[0]} SVG genes to {top_scores_path}")

    # Extract top gene names
    gene_col = "gene" if "gene" in df_top.columns else df_top.columns[0]
    top_gene_names = df_top[gene_col].astype(str).tolist()

    out = WorkerOutput("prost", task="svg_identification")
    out.set_data(
        n_spots=n_spots_supplied,
        n_spots_used=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        n_genes_used=int(df_scores.shape[0]),
    )
    record_in_tissue(out, n_spots_supplied, n_off_tissue)
    out.add_output_files(
        {
            "all_scores_csv": all_scores_path,
            "top_scores_csv": top_scores_path,
        }
    )
    record_method(out, method, used_fallback=not used_prost, why=str(prost_failure) if not used_prost else "")
    out.add_params(
        {
            "used_prost": bool(used_prost),
            "allow_spectral_fallback": bool(allow_spectral_fallback),
            "score_column": score_col,
            # The preset PROST was actually driven with, which is not always the one asked for.
            "platform": prost_platform,
            "expression_source": expression_source,
            "min_detected_frac": min_detected_frac,
            # Published beside the threshold because the threshold alone is misleading: only the
            # spectral substitute consumes it. On the official path PROST's own gene selection
            # decides, so a reader who sees min_detected_frac here would otherwise credit their
            # setting for a gene set it had no part in choosing.
            "min_detected_frac_applied": not used_prost,
            "seed": seed,
        }
    )
    if used_prost:
        out.add_param("coordinate_source", coordinate_source)
        # These size the substitute's graph and its score; PROST's own stages read none of them. The
        # seed too: prepare_for_PI -> minmax_scaler -> gau_filter -> get_binary -> get_sub ->
        # cal_prost_index draws no random number, so the index is the same whatever it is.
        record_ignored(
            out,
            ["n_neighbors", "n_eigs", "low_freq_fraction", "min_detected_frac", "seed"],
            "they configure the spectral substitute only (PROST's Index is deterministic and reads no seed), "
            "and PROST Index ran instead",
        )
    else:
        out.add_params(
            {
                "n_neighbors": n_neighbors,
                "n_eigs": n_eigs,
                "low_freq_fraction": low_freq_fraction,
                "coordinate_source": {"knn_graph": f"obsm['{spatial_key}']"},
            }
        )
        record_ignored(out, ["platform"], "the spectral substitute has no platform preset; PROST did not run")
    # PROST's index is a continuous score with no null model, so nothing here is "significant" --
    # df_top is just the requested top-k. Reporting its size as n_significant made the discovery
    # rate track the caller's n_top_genes (5 -> 1.4%, 50 -> 13.8% on the same slide).
    out.set_summary(
        n_significant=None,
        n_top_reported=int(df_top.shape[0]),
        top_genes=top_gene_names,
    )
    analysis = build_svg_analysis(
        int(df_scores.shape[0]),
        None,
        top_gene_names,
        method_name=method,
        n_genes_renamed=int(adata.uns.get("identifier_renames", {}).get("n_genes_renamed", 0)),
    )
    if not used_prost:
        analysis = _substitute_note(_METHOD_PROST_INDEX, _METHOD_SUBSTITUTE_SVG, str(prost_failure)) + analysis
    else:
        # Which selection produced the gene list is the first thing a reader needs in order to
        # reproduce it, and the parameter they set is not the answer.
        analysis += (
            f" Gene selection: PROST's own prepare_for_PI chose these {int(df_scores.shape[0])} genes; "
            f"the min_detected_frac={min_detected_frac:g} setting was not applied on this path."
        )
    if platform_note:
        analysis += " " + platform_note
    analysis += _in_tissue_note(n_spots_supplied, n_off_tissue)
    out.set_analysis(analysis)
    return out.to_dict()


# ---------------------------------------------------------------------
# PROST PNN (domains) -- real PROST; a labelled spectral substitute only when allowed
# ---------------------------------------------------------------------


def run_prost_domains(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    layer_key: str | None = None,
    n_neighbors: int = 20,
    n_eigs: int = 30,
    n_domains: int = 6,
    seed: int = 0,
    platform: str = "visium",
    pnn_init: str = "kmeans",
    pnn_n_top_genes: int = 3000,
    # PROST's own run_PNN default. Deliberately NOT n_neighbors, whose default (20) belongs to the
    # spectral substitute's kNN graph: reusing it here would run the published method with a graph
    # three times denser than its authors' default and quietly change what "PROST" means.
    pnn_k_neighbors: int = 7,
    allow_spectral_fallback: bool = False,
    pnn_preprocessing: str = _PNN_PREPROCESSING_TUTORIAL,
    # PROST's run_PNN_sparse instead of run_PNN. Off by default: its attention layer is a different
    # formula (pyGAT's sparse one), so the default keeps the published run_PNN.
    pnn_sparse: bool = False,
    # Run on adata.raw.X instead of X or a layer: CELLxGENE exports keep their counts there, where no
    # layer_key reaches them. The fleet's shared rule (worker_utils.choose_counts_matrix).
    use_raw_counts: bool = False,
) -> dict:
    # Argument errors first, before anything is loaded: neither is a PROST failure, so neither may
    # be caught below and turned into a substitute run.
    _check_pnn_init(pnn_init)
    _check_pnn_preprocessing(pnn_preprocessing)
    if use_raw_counts and layer_key:
        raise ValueError(
            f"use_raw_counts=True runs on adata.raw.X and layer_key='{layer_key}' names another matrix; "
            "pass one of them."
        )
    os.makedirs(output_dir, exist_ok=True)

    adata, n_spots_supplied, n_off_tissue = _load_in_tissue(st_h5ad)
    counts_info = None
    gene_table = None  # adata.var, unless the matrix PROST is given has its own genes
    if use_raw_counts:
        # The written h5ad stays the caller's object; only the matrix PROST is handed changes.
        analysed, counts_info = choose_counts_matrix(adata, use_raw_counts=True)
        make_names_unique_and_report(analysed, axes=("var",))
        X, expression_source, gene_table = analysed.X, "raw.X", analysed.var
    else:
        X, expression_source = _expression_matrix(adata, layer_key)
    n_genes_supplied = int(X.shape[1])
    coords = _get_spatial_coords(adata, spatial_key)

    used_prost = False
    domain_labels = None
    method = _METHOD_PROST_PNN_SPARSE if pnn_sparse else _METHOD_PROST_PNN
    prost_failure = None
    prost_out_of_memory = False
    coordinate_source = None
    # Filled in only on the PROST path, so params record the knobs that actually shaped the result.
    pnn_params: dict = {}
    pnn_function = "run_PNN_sparse" if pnn_sparse else "run_PNN"
    pnn_memory = None
    # The PROST stage running when a failure hit, so a refusal names it (an out-of-memory in cal_PI is
    # not one pnn_sparse can fix).
    pnn_stage = "prepare_for_PI"

    # ----- Official PROST PNN -----
    # The published pipeline is prepare_for_PI -> cal_PI -> normalize_total + log1p ->
    # feature_selection(by='prost') -> run_PNN. feature_selection reads var['PI'], so cal_PI has to
    # run even when only domains are wanted; and it is what keeps run_PNN's dense PCA tractable
    # (33538 genes -> pnn_n_top_genes).
    prost, prost_failure = _import_prost()
    if prost is not None and pnn_sparse and not callable(getattr(prost, "run_PNN_sparse", None)):
        prost_failure = "pnn_sparse=True, but this PROST install has no top-level run_PNN_sparse"
        logging.warning(prost_failure)
        prost = None
    # Assigned unconditionally: params and the analysis text below read them on every route.
    prost_platform, platform_note = platform, None
    if prost is not None:
        if pnn_preprocessing == _PNN_PREPROCESSING_TUTORIAL:
            # An input the caller has to fix, so it is raised here and not caught as a PROST failure.
            _require_counts(X, expression_source, list(adata.layers.keys()), raw=None if use_raw_counts else adata.raw)
        # PNN's peak grows with the square of the spot count. Estimated here, BEFORE the private copy
        # PROST is handed and before prepare_for_PI / cal_PI (which take most of the run's time), so a
        # slide that cannot fit stops with the numbers and the knob rather than after the whole PI stage.
        # Not an input error: PROST cannot run, so a caller who allowed the substitute gets it, labelled;
        # everyone else gets the MemoryError.
        try:
            pnn_memory = _check_pnn_memory(
                int(adata.n_obs), min(int(pnn_n_top_genes), n_genes_supplied), bool(pnn_sparse)
            )
        except ProstMemoryError as err:
            if not allow_spectral_fallback:
                raise
            prost_failure = str(err)
            logging.warning(prost_failure)
            prost = None
    if prost is not None:
        pin = _prost_input(adata, X, coords, var=gene_table)
        prost_platform, platform_note = _resolve_platform(pin, platform)
        if platform_note:
            logging.warning(platform_note)
        coordinate_source = _coordinate_source(adata, spatial_key, prost_platform, pnn=True)
        try:
            logging.info(
                f"Running official PROST PNN (platform={prost_platform}, init={pnn_init}, "
                f"n_clusters={n_domains}, k_neighbors={pnn_k_neighbors}, preprocessing={pnn_preprocessing})."
            )
            adata_pnn = prost.prepare_for_PI(pin, platform=prost_platform)
            # prepare_for_PI keeps the genes detected in at least 10% of spots and then filter_genes
            # (min_cells=3 on the lattice presets, 50 otherwise); cal_PI keeps every one of them.
            n_genes_after_prepare = int(adata_pnn.n_vars)
            pnn_stage = "cal_PI"
            adata_pnn = prost.cal_PI(adata_pnn, platform=prost_platform)
            n_genes_pi_positive = (
                int((np.asarray(adata_pnn.var["PI"], dtype=float) > 0).sum()) if "PI" in adata_pnn.var else None
            )
            pnn_stage = "the PNN preprocessing"
            adata_pnn, preprocessing_steps = _pnn_preprocess(adata_pnn, pnn_preprocessing)
            pnn_stage = "feature_selection"
            adata_pnn = prost.feature_selection(adata_pnn, by="prost", n_top_genes=pnn_n_top_genes)
            # feature_selection keeps min(genes with PI > 0, n_top_genes) of what prepare_for_PI left,
            # so on a targeted panel PNN can cluster on far fewer genes than pnn_n_top_genes asks for.
            n_genes_used = int(adata_pnn.n_vars)
            pnn_stage = pnn_function
            if pnn_sparse:
                # Same init / n_clusters / k_neighbors; the graph stays a scipy sparse matrix and the
                # attention layer is PROST's SpGraphAttentionLayer. It has no adj_mode (always kNN).
                adata_pnn = prost.run_PNN_sparse(
                    adata_pnn,
                    SEED=seed,
                    init=pnn_init,
                    n_clusters=n_domains,
                    k_neighbors=pnn_k_neighbors,
                    key_added="PROST",
                    cuda=False,
                )
            else:
                adata_pnn = prost.run_PNN(
                    adata_pnn,
                    SEED=seed,
                    init=pnn_init,
                    n_clusters=n_domains,
                    adj_mode="neighbour",
                    k_neighbors=pnn_k_neighbors,
                    key_added="PROST",
                    cuda=False,
                )
            # run_PNN writes obs['clustering']; key_added names the obsm embedding, not the labels.
            if "clustering" not in adata_pnn.obs:
                raise RuntimeError("PROST run_PNN did not write obs['clustering']")
            labels = np.asarray(adata_pnn.obs["clustering"]).astype(int).reshape(-1)
            if labels.shape[0] != adata.n_obs:
                # feature_selection subsets genes, never spots, so a mismatch means the input was
                # filtered somewhere unexpected and the labels cannot be aligned to the slide.
                raise ValueError(f"PROST PNN returned {labels.shape[0]} labels, but n_obs={adata.n_obs}")
            domain_labels = labels
            used_prost = True
            pnn_params = {
                "pnn_init": pnn_init,
                "pnn_k_neighbors": pnn_k_neighbors,
                # The request; the three counts below are what it became.
                "pnn_n_top_genes": pnn_n_top_genes,
                "pnn_n_genes_after_prepare": n_genes_after_prepare,
                "pnn_n_genes_pi_positive": n_genes_pi_positive,
                "pnn_n_genes_used": n_genes_used,
                "pnn_preprocessing": pnn_preprocessing,
                "pnn_preprocessing_steps": preprocessing_steps,
                "pnn_sparse": bool(pnn_sparse),
                # The PROST function that clustered, and its memory estimate against what was free.
                "pnn_function": pnn_function,
                "pnn_memory": pnn_memory,
            }
            logging.info(f"PROST PNN produced {len(np.unique(labels))} domains.")
        except Exception as e:
            prost_failure = f"official PROST PNN failed in {pnn_stage} with {type(e).__name__}: {e}"
            prost_out_of_memory = _looks_out_of_memory(e)
            logging.warning("Official PROST PNN failed (%s: %s).", type(e).__name__, e)
            traceback.print_exc(file=sys.stderr)
            domain_labels = None
            used_prost = False

    if not used_prost:
        _refuse_or_warn(
            allow_spectral_fallback,
            "spatial domain identification",
            str(prost_failure),
            out_of_memory=prost_out_of_memory,
            remedy=(
                f"PNN was set to run as PROST.{pnn_function} on {int(adata.n_obs)} spots (estimated peak "
                + (_gib(pnn_memory["estimated_bytes"]) if pnn_memory else "unknown")
                + f"); the run ran out of memory in {pnn_stage}. "
                + (
                    "pnn_sparse=True runs PROST's run_PNN_sparse, which keeps the spot graph sparse, "
                    "or run on a machine with more memory."
                    if pnn_stage == "run_PNN"
                    else "Run on a machine with more memory."
                )
            ),
        )
        method = _METHOD_SUBSTITUTE_DOMAINS

    # ----- Substitute (only when allowed): spectral clustering of the spot coordinates -----
    if not used_prost:
        logging.info("Running the spatial-coordinate-only spectral substitute for spatial domains.")
        W, L = _build_knn_graph(coords, n_neighbors=n_neighbors)

        logging.info(f"Computing {n_eigs} smallest eigenpairs of normalized Laplacian (start vector seed={seed}).")
        evals, evecs = eigsh(L, k=n_eigs, which="SM", v0=_seeded_start(L.shape[0], seed))
        logging.info("First 10 eigenvalues: %s", evals[:10])

        # Use a few non-trivial eigenvectors as embedding
        k_embed = min(max(2, n_domains), n_eigs)
        U = evecs[:, 1:k_embed]  # skip trivial first eigenvector

        from sklearn.cluster import KMeans

        # A KMeans failure propagates: one domain for every spot is not a clustering result.
        logging.info(f"Running KMeans with n_clusters={n_domains}, embedding_dim={U.shape[1]}")
        km = KMeans(n_clusters=n_domains, random_state=seed, n_init=10)
        domain_labels = km.fit_predict(U)

    domain_labels = np.asarray(domain_labels).reshape(-1)
    if domain_labels.shape[0] != adata.n_obs:
        raise ValueError(f"domain_labels length {domain_labels.shape[0]} != n_obs={adata.n_obs}")

    # Save CSV with domain labels
    domains_csv = os.path.join(output_dir, "prost_domain_labels.csv")
    df_dom = pd.DataFrame(
        {
            "spot": np.asarray(adata.obs_names),
            "prost_domain": domain_labels.astype(int),
        }
    )
    _atomic_write(domains_csv, lambda p: df_dom.to_csv(p, index=False))
    logging.info(f"Saved domain labels to {domains_csv}")

    # Save annotated AnnData
    adata.obs["prost_domain"] = domain_labels.astype(str)
    domains_h5ad = os.path.join(output_dir, "prost_domains_annotated.h5ad")
    _atomic_write(domains_h5ad, lambda p: adata.write_h5ad(p))
    logging.info(f"Saved annotated AnnData to {domains_h5ad}")

    # Basic summary
    uniq, counts = np.unique(domain_labels, return_counts=True)
    cluster_summary = {int(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist())}

    out = WorkerOutput("prost", task="domains")
    out.set_data(
        n_spots=n_spots_supplied,
        n_spots_used=int(adata.n_obs),
        # The genes of the matrix PROST was given (adata.raw's with use_raw_counts).
        n_genes=n_genes_supplied,
    )
    if used_prost:
        # The genes PNN clustered on -- not the panel supplied, and not the pnn_n_top_genes requested.
        out.set_data(n_genes_used=int(pnn_params["pnn_n_genes_used"]))
        if pnn_params["pnn_n_genes_used"] < int(pnn_n_top_genes):
            out.add_warning(
                f"pnn_n_top_genes={int(pnn_n_top_genes)} was requested but PROST PNN clustered on "
                f"{pnn_params['pnn_n_genes_used']} genes: prepare_for_PI kept {pnn_params['pnn_n_genes_after_prepare']} "
                f"of the {n_genes_supplied} supplied and feature_selection keeps at most the genes with a positive "
                f"PROST Index ({pnn_params['pnn_n_genes_pi_positive']})"
            )
    record_in_tissue(out, n_spots_supplied, n_off_tissue)
    out.add_output_files(
        {
            "domains_csv": domains_csv,
            "domains_h5ad": domains_h5ad,
        }
    )
    record_method(out, method, used_fallback=not used_prost, why=str(prost_failure) if not used_prost else "")
    if counts_info is not None:
        record_expression_source(out, counts_info)
    out.add_params(
        {
            "used_prost": bool(used_prost),
            "allow_spectral_fallback": bool(allow_spectral_fallback),
            "use_raw_counts": bool(use_raw_counts),
            "n_domains": n_domains,
            "seed": seed,
            # The preset PROST was actually driven with, which is not always the one asked for.
            # Recorded on every route, including the substitute, so the reader can tell which
            # geometry was assumed.
            "platform": prost_platform,
            **pnn_params,
        }
    )
    if used_prost:
        out.add_params({"expression_source": expression_source, "coordinate_source": coordinate_source})
        # These size the substitute's graph and embedding; PROST's own stages read neither.
        record_ignored(
            out,
            ["n_neighbors", "n_eigs"],
            "they configure the spectral substitute only, and PROST PNN ran instead",
        )
    else:
        out.add_params(
            {
                "n_neighbors": n_neighbors,
                "n_eigs": n_eigs,
                "expression_source": "not used (the substitute clusters spot coordinates only)",
                "coordinate_source": {"knn_graph": f"obsm['{spatial_key}']"},
            }
        )
        ignored = ["pnn_init", "pnn_k_neighbors", "pnn_n_top_genes", "pnn_preprocessing", "pnn_sparse", "platform"]
        if layer_key:
            ignored.append("layer_key")
        if use_raw_counts:
            ignored.append("use_raw_counts")
        record_ignored(
            out,
            ignored,
            "the substitute clusters the spot coordinates alone; gene expression and PROST's PNN knobs were not used",
        )
    out.set_summary(
        n_domains=int(len(uniq)),
        cluster_sizes=cluster_summary,
    )
    analysis = build_cluster_analysis(
        cluster_summary,
        cluster_key="domain",
        total_spots=int(adata.n_obs),
        n_requested=n_domains,
    )
    if not used_prost:
        analysis = _substitute_note(_METHOD_PROST_PNN, _METHOD_SUBSTITUTE_DOMAINS, str(prost_failure)) + analysis
    else:
        analysis += (
            f" Expression source: {expression_source}; PNN graph coordinates: {coordinate_source['pnn_graph']}; "
            f"gene-image coordinates: {coordinate_source['gene_image']}. PNN preprocessing: "
            f"{' + '.join(pnn_params.get('pnn_preprocessing_steps') or []) or 'none (matrix used as stored)'}."
            f" Genes: PNN clustered on {pnn_params['pnn_n_genes_used']} of the {n_genes_supplied} genes supplied "
            f"(PROST's prepare_for_PI kept {pnn_params['pnn_n_genes_after_prepare']}, and feature_selection kept "
            f"the highest-PI genes with a positive index, at most pnn_n_top_genes={int(pnn_n_top_genes)})."
            f" PNN function: PROST.{pnn_function}"
            + (
                " (pnn_sparse=True: sparse spot graph and PROST's sparse graph-attention layer)."
                if pnn_sparse
                else " (dense spot graph and attention, PROST's published default)."
            )
        )
    if platform_note:
        analysis += " " + platform_note
    analysis += _in_tissue_note(n_spots_supplied, n_off_tissue)
    out.set_analysis(analysis)
    return out.to_dict()


# ---------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Worker for PROST (Index + PNN) in SpatialOmicsLab MCP.")
    parser.add_argument(
        "--task",
        required=True,
        choices=["index", "index_svg", "domains", "pnn"],
        help="Which PROST workflow to run.",
    )
    parser.add_argument("--st-h5ad", required=True, help="Input spatial AnnData (.h5ad).")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save PROST outputs.",
    )
    parser.add_argument(
        "--spatial-key",
        default="spatial",
        help=(
            "Key in adata.obsm for spatial coordinates. Written to obsm['spatial'] of the object PROST is given, "
            "so PROST's PNN graph reads these coordinates; its 'visium'/'ST' presets rasterise "
            "obs['array_row']/obs['array_col'] for the gene image when those columns exist."
        ),
    )
    parser.add_argument(
        "--layer-key",
        default="",
        help="Optional adata.layers key for expression; default uses adata.X. A layer that is absent is an error.",
    )

    # Substitute-only graph params
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=20,
        help="Number of neighbors for the spectral substitute's kNN graph (not read by PROST).",
    )
    parser.add_argument(
        "--n-eigs",
        type=int,
        default=50,
        help="Number of Laplacian eigenpairs the spectral substitute computes (not read by PROST).",
    )

    # Index-specific
    parser.add_argument(
        "--low-freq-fraction",
        type=float,
        default=0.1,
        help="Fraction of low-frequency components used in the substitute's SVG score.",
    )
    parser.add_argument(
        "--min-detected-frac",
        type=float,
        default=0.05,
        help="Minimum non-zero fraction to keep a gene (substitute only; PROST runs its own gene selection).",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=200,
        help="Number of top SVG genes to export.",
    )

    # Domain-specific
    parser.add_argument(
        "--n-domains",
        type=int,
        default=6,
        help="Number of spatial domains (clusters) for PNN / the substitute.",
    )

    parser.add_argument(
        "--platform",
        default="visium",
        help="PROST platform preset ('visium', 'ST', or a non-grid platform name).",
    )
    parser.add_argument(
        "--pnn-init",
        default="kmeans",
        choices=list(_PNN_INITS_HONOURING_A_COUNT),
        help=(
            "Centroid initialisation for PROST PNN. Only kmeans and mclust honour an exact "
            "n_domains; PROST's own default ('leiden') ignores it in favour of a resolution."
        ),
    )
    parser.add_argument(
        "--pnn-n-top-genes",
        type=int,
        default=3000,
        help="Genes kept by PROST feature_selection before PNN (PROST's own default is 3000).",
    )
    parser.add_argument(
        "--pnn-k-neighbors",
        type=int,
        default=7,
        help=(
            "Neighbours for PROST PNN's cell graph. Separate from --n-neighbors, which sizes the "
            "spectral substitute's graph; PROST's own default is 7."
        ),
    )
    parser.add_argument(
        "--pnn-preprocessing",
        default=_PNN_PREPROCESSING_TUTORIAL,
        choices=list(_PNN_PREPROCESSING_CHOICES),
        help=(
            "Expression preprocessing between cal_PI and feature_selection/run_PNN. 'normalize_log1p' runs "
            "sc.pp.normalize_total + sc.pp.log1p on raw counts, as PROST's tutorial does (a non-integer "
            "matrix is refused); 'none' feeds the matrix to PNN as stored."
        ),
    )
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        default=False,
        help=(
            "Domains only: give PROST adata.raw.X instead of X (refused when there is no adata.raw, when it is "
            "not counts, or together with --layer-key). For CELLxGENE-style files whose counts sit in adata.raw."
        ),
    )
    parser.add_argument(
        "--pnn-sparse",
        action="store_true",
        default=False,
        help=(
            "Cluster with PROST's run_PNN_sparse (sparse spot graph and attention layer) instead of run_PNN, "
            "whose memory grows with the square of the spot count. Off by default."
        ),
    )
    parser.add_argument(
        "--allow-spectral-fallback",
        action="store_true",
        default=False,
        help=(
            "Permit a clearly labelled spatial-coordinate-only spectral substitute when PROST cannot run. "
            "Off by default: a PROST failure is then an error carrying PROST's own message."
        ),
    )
    parser.add_argument(
        "--no-spectral-fallback",
        action="store_true",
        default=False,
        help=(
            "Refuse the substitute (the default behaviour; kept so existing command lines keep working). "
            "Wins over --allow-spectral-fallback when both are given."
        ),
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()

    layer_key: str | None = args.layer_key or None
    allow_spectral_fallback = bool(args.allow_spectral_fallback) and not bool(args.no_spectral_fallback)

    try:
        # PROST reports its progress with bare print() ("Normalize each geneing...", "Gaussian
        # filtering...") -- on stdout, which carries only the JSON result. Routed to stderr with the
        # rest of the log; the result is printed after the block, on the real stdout.
        with contextlib.redirect_stdout(sys.stderr):
            result = _run_task(args, layer_key, allow_spectral_fallback)

        print(json.dumps(result), file=sys.stdout)
        return 0

    except Exception as e:
        logging.error("ERROR in prost_worker: %s", e)
        logging.error(traceback.format_exc())
        task_name = "svg_identification" if args.task in ("index", "index_svg") else "domains"
        WorkerOutput.emit_error("prost", str(e), task=task_name)
        return 1


def _run_task(args, layer_key, allow_spectral_fallback) -> dict:
    """Dispatch the parsed command line to the task it names."""
    if args.task in ("index", "index_svg"):
        if args.use_raw_counts or args.pnn_sparse:
            # The PROST Index reads the matrix as stored (min-max scaled per gene) and runs no PNN; a flag
            # it cannot honour is refused rather than dropped.
            raise ValueError("--use-raw-counts and --pnn-sparse apply to --task domains only.")
        return run_prost_index(
            st_h5ad=args.st_h5ad,
            output_dir=args.output_dir,
            spatial_key=args.spatial_key,
            layer_key=layer_key,
            n_neighbors=args.n_neighbors,
            n_eigs=args.n_eigs,
            low_freq_fraction=args.low_freq_fraction,
            min_detected_frac=args.min_detected_frac,
            n_top_genes=args.n_top_genes,
            seed=args.seed,
            platform=args.platform,
            allow_spectral_fallback=allow_spectral_fallback,
        )
    # domains / pnn
    return run_prost_domains(
        st_h5ad=args.st_h5ad,
        output_dir=args.output_dir,
        spatial_key=args.spatial_key,
        layer_key=layer_key,
        n_neighbors=args.n_neighbors,
        n_eigs=args.n_eigs,
        n_domains=args.n_domains,
        seed=args.seed,
        platform=args.platform,
        pnn_init=args.pnn_init,
        pnn_n_top_genes=args.pnn_n_top_genes,
        pnn_k_neighbors=args.pnn_k_neighbors,
        allow_spectral_fallback=allow_spectral_fallback,
        pnn_preprocessing=args.pnn_preprocessing,
        pnn_sparse=args.pnn_sparse,
        use_raw_counts=args.use_raw_counts,
    )


if __name__ == "__main__":
    sys.exit(main())
