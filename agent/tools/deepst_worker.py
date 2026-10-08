#!/usr/bin/env python

"""
deepst_worker.py

Worker script for running DeepST spatial domain identification on a single
spatial transcriptomics AnnData (.h5ad).

- This script is called by the FastMCP wrapper (deepst_mcp_server.py).
- It must be executed inside the DeepST conda env: /opt/conda/envs/deepst-env
- All logs go to stderr; stdout only prints a single JSON line at the end.

Example (manual test):

  (deepst-env) python /workspace/epic-fermat/agent/tools/deepst_worker.py \
      --task identify_domains \
      --st-h5ad /workspace/work/spatial_input/V1_Human_Lymph_Node.h5ad \
      --output-dir /workspace/work/deepst_V1_LN \
      --n-domains 7 \
      --pre-epochs 300 \
      --epochs 300 \
      --pca-n-comps 200 \
      --spatial-type BallTree \
      --dist-type KDTree \
      --use-morphological False \
      --use-gpu auto \
      --seed 0
      # add --allow-resolution-fallback to accept Leiden at deepstkit's default
      # resolution 1.0 when no swept resolution yields exactly --n-domains

How the domain count is reached, and what this worker says about it
--------------------------------------------------------------------
``deepstkit`` clusters the learned embedding with Leiden. ``_priori_cluster`` sweeps the
resolution from 2.49 down to 0.10 in steps of 0.01 and stops at the first value whose Leiden
partition has exactly ``n_domains`` clusters. When no value does, upstream ``return 1.0``: Leiden
then runs at resolution 1.0 and the run finishes with whatever count that gives -- a different
clustering from the one asked for, produced without a word. This worker wraps that search
(:func:`install_resolution_search_recorder`) so the payload always carries
``params.resolution_used``, ``params.resolution_search_matched``, ``params.method`` and
``params.used_fallback``; the substitute resolution is refused unless the caller passes
``allow_resolution_fallback=True``. Spatial refinement (``DeepST_refine_domain``) can then merge
Leiden clusters, so ``summary.n_domains_leiden`` (before refinement) is reported beside
``summary.n_clusters`` (after).

Spots with ``obs['in_tissue'] == 0`` (background glass, which CELLxGENE Visium exports ship beside
the tissue) are left out before anything runs and counted under ``params.in_tissue_filter`` with a
warning; ``data.n_spots`` is the supplied count and ``data.n_spots_used`` the analysed one. When the
spatial plot cannot be drawn, ``output_files.spatial_plot_png`` is None and a warning says why.

deepstkit's ``_data_process`` normalises the augmented matrix as counts (``normalize_total`` +
``log1p``), so the matrix is checked first (``worker_utils.choose_counts_matrix``): a negative or
non-finite X (scaled / z-scored data) is refused, naming ``--use-raw-counts`` when ``adata.raw`` holds
the counts; a non-negative non-integer X (already normalised or log-transformed) runs as before with a
warning; ``--use-raw-counts`` runs on ``adata.raw.X``. ``params.expression_source`` /
``params.x_matrix_kind`` say which matrix ran.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from typing import Any

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
    record_expression_source,
    record_in_tissue,
    record_method,
    resolve_compute,
    unsupported_choice_msg,
)

# The bounds of deepstkit's own resolution sweep (``main.py::_priori_cluster``:
# ``sorted(np.arange(0.1, 2.5, 0.01), reverse=True)``) and the value it returns when the sweep
# fails. Named here so the messages below quote the real numbers rather than prose.
SWEEP_RESOLUTION_MIN = 0.10
SWEEP_RESOLUTION_MAX = 2.49
SWEEP_RESOLUTION_STEP = 0.01
UPSTREAM_FALLBACK_RESOLUTION = 1.0

# The obs column deepstkit writes its spatially refined labels to (``_get_cluster_data`` default
# ``output_key``) and the Leiden column it refines from (default ``key_added``). The refined one
# is the only column this worker publishes; nothing is guessed from the input's own obs.
REFINED_DOMAIN_KEY = "DeepST_refine_domain"
LEIDEN_DOMAIN_KEY = "DeepST_domain"

# The neighbour searches ``deepstkit.augment.cal_spatial_weight`` implements. Any other value
# reaches ``indices[:, 1:]`` with ``indices`` never assigned, so the caller got
# ``UnboundLocalError: local variable 'indices' referenced before assignment``. ``LinearRegress``
# is a separate branch of ``cal_weight_matrix`` that reads four obs columns instead of obsm.
SPATIAL_TYPES = ("BallTree", "KDTree", "NearestNeighbors", "LinearRegress")
LINEAR_REGRESS_OBS_COLUMNS = ("imagerow", "imagecol", "array_row", "array_col")

# ``cal_weight_matrix(use_morphological=True)`` reads this obsm key, which only deepstkit's
# ``_get_image_crop`` (ResNet50 features of H&E tiles, then PCA) writes. This worker never runs
# that step, so the features have to arrive in the input file.
MORPHOLOGY_FEATURE_KEY = "image_feat_pca"

# Spot diameter (coordinate units) for the domain plot of a slide whose uns['spatial'] states none;
# a library that states its own ``spot_diameter_fullres`` is drawn at that size (_spatial_plot_kwargs).
PLOT_SPOT_SIZE = 150


def log(msg: str) -> None:
    """Print log messages to stderr with a prefix."""
    sys.stderr.write(f"[deepst-worker] {msg}\n")
    sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepST worker: identify spatial domains on a spatial .h5ad.")

    parser.add_argument(
        "--task",
        type=str,
        default="identify_domains",
        help="Task name (reserved for future extension).",
    )
    parser.add_argument(
        "--st-h5ad",
        type=str,
        required=True,
        help="Path to spatial AnnData (.h5ad).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save DeepST outputs.",
    )
    parser.add_argument(
        "--n-domains",
        type=int,
        default=7,
        help=(
            "Target number of spatial domains. deepstkit sweeps the Leiden resolution from 2.49 down to 0.10 "
            "for a partition with exactly this many clusters; see --allow-resolution-fallback for what happens "
            "when none exists."
        ),
    )
    parser.add_argument(
        "--pre-epochs",
        type=int,
        default=500,
        help="Number of pretraining epochs for DeepST.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of main training epochs for DeepST.",
    )
    parser.add_argument(
        "--pca-n-comps",
        type=int,
        default=200,
        help=(
            "Number of PCs for DeepST._data_process (dimensionality reduction). Capped to "
            "min(n_obs, n_vars) - 1; the value that ran is reported as params.pca_n_comps beside "
            "params.pca_n_comps_requested."
        ),
    )
    parser.add_argument(
        "--spatial-type",
        type=str,
        default="BallTree",
        help="Spatial neighbor construction type for _get_augment (e.g. BallTree).",
    )
    parser.add_argument(
        "--dist-type",
        type=str,
        default="KDTree",
        help="Distance type for _get_graph (e.g. KDTree).",
    )
    parser.add_argument(
        "--use-morphological",
        type=str,
        default="False",
        help="Whether to use morphological (H&E) features in _get_augment (True/False). Usually False if no image.",
    )
    parser.add_argument(
        "--use-gpu",
        type=str,
        default="auto",
        help="GPU usage: 'auto', 'cpu', 'gpu'/'cuda', 'cuda:N', or an index ('-1' = CPU, '0' = GPU 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for DeepST (torch, numpy, etc.).",
    )
    parser.add_argument(
        "--allow-resolution-fallback",
        action="store_true",
        default=False,
        help=(
            "When no Leiden resolution in deepstkit's sweep (0.10-2.49) yields exactly --n-domains, accept "
            "deepstkit's substitute -- Leiden at resolution 1.0, however many clusters that gives -- instead of "
            "stopping. The payload then reports params.used_fallback=true and params.resolution_used=1.0. Off "
            "by default: the run fails with a message naming this flag."
        ),
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


def resolve_use_gpu(mode) -> bool:
    """``dt.main.run`` takes a bool, so collapse the fleet's device vocabulary down to one.

    This used to recognise exactly three words -- ``'gpu'``, ``'cpu'``, and anything-else-means-auto
    -- and answer ``torch.cuda.is_available()`` for that last case. Everywhere else in the fleet a
    device may also arrive as an int index or its string form (``-1`` = CPU, ``0`` = GPU 0, the
    convention documented at ``spaceflow_worker.py:240``), or as ``'cuda'``/``'cuda:1'``/``'CPU'``.
    All of those fell into the auto branch, so a caller writing ``-1`` to *avoid* the GPU was handed
    one. ``resolve_compute`` knows every spelling, and still degrades a GPU request on a CPU-only
    box, which is the behaviour ``deepstkit``'s own trainer re-checks for anyway.
    """
    return resolve_compute(mode).device.startswith("cuda")


def _atomic_replace(partial_path: str, final_path: str) -> None:
    """Move a finished ``.partial`` file onto its final name in one step."""
    os.replace(partial_path, final_path)


def validate_request(adata, n_domains: int, spatial_type: str, use_morphological: bool) -> None:
    """Refuse, before any training, a request deepstkit can only fail on -- and say why.

    Every check here used to fail late or opaquely: ``spatial_type`` outside deepstkit's set
    raised ``UnboundLocalError`` about a variable called ``indices``; ``use_morphological=True``
    raised ``KeyError: 'image_feat_pca'`` from inside the augmentation; and an ``n_domains`` no
    partition can have (below 1, above the spot count) trained for ``pre_epochs + epochs`` and
    then missed the resolution sweep.
    """
    n_spots = int(adata.n_obs)
    if int(n_domains) < 1 or int(n_domains) > n_spots:
        raise ValueError(
            f"n_domains={n_domains} cannot be reached: a partition of {n_spots} spots has between 1 and "
            f"{n_spots} domains. Set n_domains to the number of spatial domains expected in the tissue."
        )
    if spatial_type not in SPATIAL_TYPES:
        raise ValueError(
            unsupported_choice_msg(
                "spatial_type",
                spatial_type,
                SPATIAL_TYPES,
                extra="It picks the neighbour search deepstkit's augmentation uses to weight nearby spots.",
            )
        )
    if spatial_type == "LinearRegress":
        missing = [c for c in LINEAR_REGRESS_OBS_COLUMNS if c not in adata.obs]
        if missing:
            raise ValueError(
                f"spatial_type='LinearRegress' fits spot distances from obs columns "
                f"{list(LINEAR_REGRESS_OBS_COLUMNS)}, and this file lacks {missing}. Use spatial_type="
                f"'BallTree' (the default) to search neighbours on obsm['spatial'] instead."
            )
    if use_morphological and MORPHOLOGY_FEATURE_KEY not in adata.obsm:
        raise ValueError(
            f"use_morphological=True weights neighbours by H&E similarity read from "
            f"adata.obsm['{MORPHOLOGY_FEATURE_KEY}'] (deepstkit's _get_image_crop output: ResNet50 features of "
            f"the image tile under each spot, reduced by PCA). This worker does not extract image features, "
            f"and this file has no such key (obsm keys: {list(adata.obsm.keys())}). Add the features to the "
            f"input, or set use_morphological=False to run on expression and spatial proximity."
        )


def ensure_csr(adata) -> str:
    """Hand deepstkit a CSR (or dense) ``X``; return the original sparse format when it changed.

    ``augment.find_adjacent_spot`` accepts only ``csr_matrix``, ``ndarray`` or ``DataFrame`` and
    raises ``Unsupported data type`` for anything else, and ``augment_gene_data`` has no branch at
    all for other formats. A CSC-stored h5ad crashed there. Converting CSC to CSR keeps every
    value and stays sparse; deepstkit densifies later on its own terms.
    """
    import scipy.sparse as sp

    if sp.issparse(adata.X) and not isinstance(adata.X, sp.csr_matrix):
        original = type(adata.X).__name__
        adata.X = sp.csr_matrix(adata.X)
        log(f"Converted X from {original} to csr_matrix (same values; deepstkit's augmentation reads CSR only)")
        return original
    return ""


def _memory_available_bytes():
    """Bytes this run can still allocate, or None when the platform cannot say.

    The shared reader (:func:`worker_utils.available_memory_bytes`): the smaller of MemAvailable and
    the room under a cgroup memory limit, page cache counted as reclaimable. This used to read
    MemAvailable alone, which inside a memory-limited container is the *host's* free memory -- so a
    slide the container could not hold passed the check and was OOM-killed with no payload.
    """
    available = available_memory_bytes()
    return None if available is None else int(available)


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


def dense_budget_bytes(n_spots: int, n_genes: int) -> int:
    """A floor on the dense memory deepstkit holds at once for ``n_spots`` x ``n_genes``.

    Intrinsic to the method, not to this wrapper. The count is what is alive together during
    ``refine``, the last step, after all the training: ``obsm['weights_matrix_all']`` (float64,
    spot by spot, from the augmentation), the dense float32 ``adj_label`` the graph step built
    (still referenced by ``graph_dict``), and ``refine``'s own float64 ``cdist`` of every spot pair
    -- 20 bytes per spot pair -- beside three float64 expression matrices that stay attached to the
    AnnData (``obsm['adjacent_data']``, ``obsm['augment_gene_data']`` and the float64 ``X``
    ``_data_process`` installs) -- 24 bytes per spot and gene. The augmentation and the training
    peak higher than this, so a refusal is never a false alarm.
    """
    n = int(n_spots)
    return 20 * n * n + 24 * n * int(n_genes)


def check_dense_budget(n_spots: int, n_genes: int, what: str = "the slide"):
    """Refuse up front, with the numbers, a slide whose dense matrices cannot fit in memory.

    Without this a large slide is OOM-killed inside the augmentation or -- worse -- in the
    refinement after ``pre_epochs + epochs`` of training, with no message. No parameter of this
    tool lowers the footprint and the slide is never cut down here.
    """
    available = _memory_available_bytes()
    if available is None:
        return None
    need = dense_budget_bytes(n_spots, n_genes)
    if need > available:
        gib = 1024.0**3
        raise MemoryError(
            f"DeepST builds dense {n_spots}x{n_spots} spot-by-spot matrices (augmentation weights, graph "
            f"reconstruction target, refinement distances) and dense {n_spots}x{n_genes} expression copies for "
            f"{what}: at least {need / gib:.1f} GiB, and this machine reports {available / gib:.1f} GiB "
            "available. No parameter of this tool lowers that footprint; run it on a machine with more memory. "
            "The slide is analysed whole."
        )
    return need


def resolution_fallback_refused_message(
    n_domains: int, n_at_lowest_resolution: int, n_at_highest_resolution: int | None = None
) -> str:
    """The sentence a caller gets when the sweep misses ``n_domains`` and no fallback was allowed.

    ``n_at_lowest_resolution`` is the cluster count Leiden gave at the last resolution the sweep
    tried (0.10, the coarsest) and ``n_at_highest_resolution`` the count at its first (2.49, the
    finest), when known. Between them they bound what the embedding can reach, so the message says
    whether the request is below, above or inside that range instead of leaving the caller to guess
    a count that retraining might hit.
    """
    lo = f"the lowest resolution tried ({SWEEP_RESOLUTION_MIN:.2f}) gave {n_at_lowest_resolution} clusters"
    if n_at_lowest_resolution > n_domains:
        where = f"even {lo}, so {n_domains} is below the smallest count this embedding supports"
    elif n_at_highest_resolution is not None and n_at_highest_resolution < n_domains:
        where = (
            f"{lo} and the highest ({SWEEP_RESOLUTION_MAX:.2f}) gave {n_at_highest_resolution}, so {n_domains} is "
            f"above the largest count this embedding supports"
        )
    elif n_at_highest_resolution is not None:
        where = (
            f"{lo} and the highest ({SWEEP_RESOLUTION_MAX:.2f}) gave {n_at_highest_resolution}, but the counts "
            f"in between skip {n_domains}"
        )
    else:
        where = f"{lo} and no resolution up to {SWEEP_RESOLUTION_MAX:.2f} gave exactly {n_domains}"
    return (
        f"DeepST's Leiden resolution sweep ({SWEEP_RESOLUTION_MAX:.2f} down to {SWEEP_RESOLUTION_MIN:.2f}, "
        f"step {SWEEP_RESOLUTION_STEP:.2f}) found no resolution that yields n_domains={n_domains} on the "
        f"DeepST embedding: {where}. deepstkit would now run Leiden at its default resolution "
        f"{UPSTREAM_FALLBACK_RESOLUTION:.1f} and return however many domains that gives. Pass "
        f"allow_resolution_fallback=True to accept that substitute (the payload will then say "
        f"params.used_fallback=true and params.resolution_used={UPSTREAM_FALLBACK_RESOLUTION:.1f}), or set "
        f"n_domains to a count the embedding supports."
    )


def leiden_cluster_count(adata, resolution: float) -> int:
    """How many clusters deepstkit's own Leiden call gives at ``resolution`` on this neighbour graph.

    The arguments are the ones ``_priori_cluster`` passes (``random_state=0``, igraph, two
    iterations, undirected), so the count is the one its sweep saw at that resolution. Used only to
    tell a refused caller which counts the embedding supports; the probe column is removed again.
    """
    import scanpy as sc

    key = "_deepst_worker_resolution_probe"
    try:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            random_state=0,
            flavor="igraph",
            n_iterations=2,
            directed=False,
            key_added=key,
        )
        return int(adata.obs[key].nunique())
    finally:
        if key in adata.obs:
            del adata.obs[key]
        adata.uns.pop(key, None)


def install_resolution_search_recorder(
    deepst, allow_resolution_fallback: bool, search: dict, count_clusters=leiden_cluster_count
) -> None:
    """Make ``deepst._get_cluster_data`` report -- and, by default, refuse -- the substitute resolution.

    ``deepstkit.main.run._get_cluster_data`` calls ``self._priori_cluster(adata, n_domains)``, which
    sweeps Leiden resolutions and returns the first that yields ``n_domains`` clusters -- or ``1.0``
    when none does, with nothing to tell the two apart. Binding this wrapper on the *instance* puts
    it ahead of the class method in attribute lookup, so upstream's own sweep still runs unchanged
    and the wrapper reads the result: after the call ``adata.obs['leiden']`` holds the last Leiden
    partition tried, which is the matching one when the search succeeded and the 0.10 one when it
    did not. ``search`` receives ``resolution``, ``matched`` and ``n_at_last_resolution``.

    When the sweep failed and ``allow_resolution_fallback`` is False, the wrapper raises before
    upstream runs Leiden at 1.0 and the O(n_spots^2) refinement, so a refused run costs nothing more
    than one extra Leiden call (``count_clusters`` at 2.49) that tells the caller the reachable range.
    """
    original = deepst._priori_cluster

    def _priori_cluster_recorded(adata, n_domains):
        res = original(adata, n_domains)
        n_last = int(adata.obs["leiden"].nunique()) if "leiden" in adata.obs else -1
        matched = n_last == int(n_domains)
        search["resolution"] = round(float(res), 6)
        search["matched"] = bool(matched)
        search["n_at_last_resolution"] = n_last
        if matched:
            log(f"Resolution sweep matched n_domains={n_domains} at resolution {float(res):.2f}")
            return res
        log(
            f"Resolution sweep found no resolution giving n_domains={n_domains}; deepstkit returned "
            f"{float(res):.2f} (its default), Leiden at {SWEEP_RESOLUTION_MIN:.2f} gave {n_last} clusters"
        )
        if not allow_resolution_fallback:
            try:
                n_top = int(count_clusters(adata, SWEEP_RESOLUTION_MAX))
            except Exception as exc:  # the range is a courtesy; the refusal stands without it
                log(f"Could not count clusters at resolution {SWEEP_RESOLUTION_MAX:.2f} for the message: {exc}")
                n_top = None
            raise ValueError(resolution_fallback_refused_message(int(n_domains), n_last, n_top))
        log("allow_resolution_fallback=True: accepting Leiden at the default resolution")
        return res

    deepst._priori_cluster = _priori_cluster_recorded


def _domain_label_as_number(value: Any):
    """An integer-valued label (Python or numpy, int or float) as ``int``; anything else as text."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value) == int(value):
        return int(value)
    return str(value)


def run_deepst_identify_domains(
    st_h5ad: str,
    output_dir: str,
    n_domains: int,
    pre_epochs: int,
    epochs: int,
    pca_n_comps: int,
    spatial_type: str,
    dist_type: str,
    use_morphological: bool,
    use_gpu_flag: bool,
    seed: int,
    allow_resolution_fallback: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Core DeepST pipeline for identifying spatial domains on a single h5ad.

    deepstkit normalises the matrix as counts, so it is chosen and checked first by
    ``worker_utils.choose_counts_matrix``: X by default (a negative or non-finite X is refused, a
    non-integer one runs with a warning), ``adata.raw.X`` with ``use_raw_counts=True``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import deepstkit as dt
    import matplotlib.pyplot as plt
    import numpy as np
    import scanpy as sc

    os.makedirs(output_dir, exist_ok=True)

    log("Task        = identify_domains")
    log(f"st_h5ad     = {st_h5ad}")
    log(f"output_dir  = {output_dir}")
    log(f"n_domains   = {n_domains}")
    log(f"pre_epochs  = {pre_epochs}")
    log(f"epochs      = {epochs}")
    log(f"pca_n_comps = {pca_n_comps}")
    log(f"spatial_type= {spatial_type}")
    log(f"dist_type   = {dist_type}")
    log(f"use_morph   = {use_morphological}")
    log(f"use_gpu     = {use_gpu_flag}")
    log(f"seed        = {seed}")
    log(f"allow_resolution_fallback = {allow_resolution_fallback}")
    log(f"use_raw_counts = {use_raw_counts}")

    # Set seed
    dt.utils_func.seed_torch(seed=seed)

    # Load AnnData
    log("Loading spatial AnnData...")
    adata = sc.read_h5ad(st_h5ad)
    # Background spots (obs['in_tissue'] == 0 -- CELLxGENE Visium exports ship every array spot, and
    # 56-70% of the library's four such slides are glass) are left out before anything is computed,
    # and counted: kept, they became domains of their own, used up n_domains and entered the graph.
    adata, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata, "spots")
    if n_spots_off_tissue:
        log(
            f"Left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0 (background); "
            f"{adata.n_obs} in-tissue spots are analysed"
        )
    # deepstkit's _data_process runs normalize_total + log1p on the augmented X: a scaled X (negative
    # values) turned into NaN, and a log-normalised X was normalised a second time without a word.
    adata, counts_info = choose_counts_matrix(adata, use_raw_counts)
    log(f"Expression matrix: adata.{counts_info['expression_source']} ({counts_info['x_matrix_kind']} X)")
    renamed = make_names_unique_and_report(adata)
    log(f"Loaded AnnData: n_spots={adata.n_obs}, n_genes={adata.n_vars}")

    # This check used to sit below _get_augment, where it could never fire. deepstkit reads the
    # slot first: _get_augment hands the AnnData to augment_adata, which calls cal_weight_matrix,
    # whose else branch (every spatial_type except "LinearRegress", and the shipped default is
    # "BallTree") subscripts adata.obsm["spatial"] with no membership check and no try/except
    # anywhere in augment.py. So anndata's KeyError escaped first, main() stringified it, and str
    # of a KeyError is the repr of its key -- the model was handed the one quoted word 'spatial'.
    # Ask here, before DeepST is even constructed, and name what the file does have.
    if "spatial" not in adata.obsm:
        raise ValueError(
            f"Spot coordinates not found: adata.obsm['spatial'] is missing, and DeepST's "
            f"augmentation, graph construction and refinement all read it. "
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )
    validate_request(adata, n_domains, spatial_type, use_morphological)
    x_converted_from = ensure_csr(adata)
    check_dense_budget(adata.n_obs, adata.n_vars, what=os.path.basename(st_h5ad))

    # Initialize DeepST
    log("Initializing DeepST model (Identify_Domain)...")
    deepst = dt.main.run(
        save_path=output_dir,
        task="Identify_Domain",
        pre_epochs=pre_epochs,
        epochs=epochs,
        use_gpu=use_gpu_flag,
    )

    # Data augmentation (using coordinates in adata.obsm['spatial'])
    log("Running DeepST._get_augment (no H&E by default)...")
    adata = deepst._get_augment(
        adata,
        spatial_type=spatial_type,
        use_morphological=use_morphological,
    )

    # Graph construction
    log("Constructing spatial graph via DeepST._get_graph ...")
    graph_dict = deepst._get_graph(
        adata.obsm["spatial"],
        distType=dist_type,
    )

    # Dimensionality reduction
    # Cap n_components so PCA doesn't fail when n_components >= min(n_samples, n_features). The
    # capped value is what ran, so it is what params.pca_n_comps reports; the request is kept
    # beside it as params.pca_n_comps_requested.
    n_comps = min(pca_n_comps, adata.n_obs - 1, adata.n_vars - 1)
    pca_cap_note = ""
    if n_comps < pca_n_comps:
        pca_cap_note = (
            f"pca_n_comps was capped from {pca_n_comps} to {n_comps}: PCA needs fewer components than "
            f"min(n_obs={adata.n_obs}, n_vars={adata.n_vars}); params.pca_n_comps reports the {n_comps} that ran"
        )
        log(pca_cap_note)
    log("Running DeepST._data_process for PCA reduction...")
    data = deepst._data_process(
        adata,
        pca_n_comps=n_comps,
    )

    # Model training and embedding
    log("Fitting DeepST model (_fit) to obtain embeddings...")
    deepst_embed = deepst._fit(
        data=data,
        graph_dict=graph_dict,
    )
    adata.obsm["DeepST_embed"] = deepst_embed
    log(f"DeepST_embed shape: {deepst_embed.shape}")

    # Clustering into spatial domains. deepstkit's resolution sweep is watched so the payload can
    # say which resolution ran and whether it was the one that yields n_domains; a miss is refused
    # unless allow_resolution_fallback is set (see install_resolution_search_recorder).
    search: dict = {}
    install_resolution_search_recorder(deepst, allow_resolution_fallback, search)
    log("Running DeepST._get_cluster_data to call spatial domains...")
    adata = deepst._get_cluster_data(
        adata,
        n_domains=n_domains,
        priori=True,
    )

    if "resolution" not in search:
        # The installed deepstkit clustered without calling _priori_cluster, so nothing here saw
        # which resolution ran or could hold back the 1.0 substitute. Say so rather than guess.
        raise RuntimeError(
            "deepstkit's _get_cluster_data finished without calling _priori_cluster, the resolution sweep this "
            "worker watches, so it cannot say which Leiden resolution produced the domains or whether it yields "
            f"n_domains={n_domains}. The installed deepstkit does not match the one this worker wraps "
            "(deepstkit 2.0.x, main.run)."
        )

    # deepstkit writes its refined labels to DeepST_refine_domain and the Leiden labels it refined
    # from to DeepST_domain. Only the refined column is published. This used to fall through a
    # list ending in 'domain' and 'louvain' -- generic names an *input* file may carry -- so a
    # clustering step that wrote nothing would have published the caller's own column as DeepST's.
    if REFINED_DOMAIN_KEY not in adata.obs:
        raise RuntimeError(
            f"DeepST clustering finished but adata.obs['{REFINED_DOMAIN_KEY}'] -- the column "
            f"deepstkit._get_cluster_data writes its refined domains to -- is absent. obs columns present: "
            f"{list(adata.obs.columns)}. Nothing is published from a column DeepST did not write."
        )
    cluster_key = REFINED_DOMAIN_KEY
    n_domains_leiden = int(adata.obs[LEIDEN_DOMAIN_KEY].nunique()) if LEIDEN_DOMAIN_KEY in adata.obs else None
    log(f"Using cluster_key='{cluster_key}' for output.")

    # Save annotated AnnData (written to a .partial name and moved into place once complete, so a
    # killed run never leaves a truncated file under the final name)
    annotated_h5ad = os.path.join(output_dir, "deepst_clustering.h5ad")
    log(f"Writing annotated AnnData to {annotated_h5ad}")
    annotated_partial = annotated_h5ad + ".partial"
    adata.write_h5ad(annotated_partial)
    _atomic_replace(annotated_partial, annotated_h5ad)

    # Save per-spot domain assignments
    domain_series = adata.obs[cluster_key]
    # Convert to int if possible
    try:
        domains_int = domain_series.astype(int)
    except Exception:
        domains_int = domain_series

    domains_csv = os.path.join(output_dir, "deepst_domains_per_spot.csv")
    log(f"Writing per-spot domain assignments to {domains_csv}")
    domains_partial = domains_csv + ".partial"
    domains_int.to_frame(name=cluster_key).to_csv(domains_partial, index_label="spot")
    _atomic_replace(domains_partial, domains_csv)

    # Plot spatial domains
    plot_path = os.path.join(output_dir, "deepst_spatial_domains.png")
    plot_partial = plot_path + ".partial"
    log(f"Plotting spatial domains to {plot_path}")
    plot_error = ""
    try:
        sc.pl.spatial(
            adata,
            color=[cluster_key],
            frameon=False,
            title="DeepST spatial domains",
            save=None,
            show=False,
            **_spatial_plot_kwargs(adata),
        )
        plt.savefig(plot_partial, bbox_inches="tight", dpi=300, format="png")
        _atomic_replace(plot_partial, plot_path)
    except Exception as e:
        # The domains are already written; a figure that cannot be drawn does not fail the run. But
        # the payload used to name the PNG anyway -- a file this run never wrote, or one an earlier
        # run left in output_dir -- and stderr never reaches the caller on a successful run.
        plot_error = f"{type(e).__name__}: {e}"
        with contextlib.suppress(OSError):
            os.remove(plot_partial)
        if os.path.exists(plot_path):
            # Left in place (it is not this run's to delete), but named, so it is not read as this run's.
            plot_error += f"; the {os.path.basename(plot_path)} already in output_dir is from an earlier run"
        log(f"WARNING: Failed to generate spatial plot: {plot_error}")
        plot_path = None
    finally:
        plt.close("all")

    # Summaries
    unique_domains = np.unique(domains_int.values)
    n_clusters = len(unique_domains)
    cluster_sizes = {str(k): int(v) for k, v in adata.obs[cluster_key].value_counts().items()}
    log(f"DeepST finished. n_clusters={n_clusters}, domains={unique_domains}")

    n_spots = int(adata.n_obs)
    n_genes = int(adata.n_vars)

    resolution_used = search.get("resolution")
    search_matched = bool(search.get("matched", False))

    out = WorkerOutput("deepst", task="identify_domains")
    out.set_data(n_spots=n_spots_supplied, n_spots_used=n_spots, n_genes=n_genes)
    out.add_output_files(
        {
            "annotated_h5ad": annotated_h5ad,
            "domains_csv": domains_csv,
            "spatial_plot_png": plot_path,
        }
    )
    out.add_params(
        {
            "n_domains": n_domains,
            "pre_epochs": pre_epochs,
            "epochs": epochs,
            "pca_n_comps": int(n_comps),
            "pca_n_comps_requested": int(pca_n_comps),
            "spatial_type": spatial_type,
            "dist_type": dist_type,
            "use_morphological": use_morphological,
            "use_gpu": use_gpu_flag,
            "seed": seed,
            "allow_resolution_fallback": bool(allow_resolution_fallback),
            "resolution_used": resolution_used,
            "resolution_search_matched": search_matched,
            "use_raw_counts": bool(use_raw_counts),
        }
    )
    record_expression_source(out, counts_info)
    out.add_params(identifier_rename_params(renamed))
    if x_converted_from:
        out.add_param("x_converted_from", x_converted_from)
    if pca_cap_note:
        out.add_warning(pca_cap_note)
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    if plot_error:
        out.add_warning(f"the spatial-domain plot was not written ({plot_error}); the domain outputs are complete.")

    # What ran, in words a reader can check against the numbers above. The Leiden resolution was
    # either the swept value that yields n_domains, or -- only when the caller allowed it --
    # deepstkit's default 1.0 after the sweep missed. A refused miss never reaches this point.
    stage_note = ""
    if search_matched:
        method = (
            f"DeepST (deepstkit) embedding -> Leiden at resolution {resolution_used:.2f}, the swept value "
            f"that yields n_domains={n_domains} -> spatial refinement"
        )
        record_method(out, method, used_fallback=False)
    else:
        method = (
            f"DeepST (deepstkit) embedding -> Leiden at deepstkit's default resolution "
            f"{UPSTREAM_FALLBACK_RESOLUTION:.1f} because no swept resolution yields n_domains={n_domains} "
            f"-> spatial refinement"
        )
        record_method(
            out,
            method,
            used_fallback=True,
            why=(
                f"no resolution in [{SWEEP_RESOLUTION_MIN:.2f}, {SWEEP_RESOLUTION_MAX:.2f}] gave "
                f"n_domains={n_domains}; allow_resolution_fallback=True accepted Leiden at "
                f"{UPSTREAM_FALLBACK_RESOLUTION:.1f}, which gave {n_domains_leiden} clusters before refinement"
            ),
        )
        stage_note += (
            f" Leiden ran at deepstkit's default resolution {UPSTREAM_FALLBACK_RESOLUTION:.1f} because no "
            f"resolution in its sweep yields {n_domains} domains (accepted via allow_resolution_fallback=True)."
        )
    if n_domains_leiden is not None and n_domains_leiden != n_clusters:
        stage_note += (
            f" Spatial refinement left {n_clusters} of the {n_domains_leiden} Leiden clusters "
            f"(summary.n_domains_leiden counts before refinement, summary.n_clusters after)."
        )

    # deepstkit clusters with Leiden and only searches for a resolution that happens to yield
    # n_domains, falling back to 1.0 when none does -- so n_clusters can differ from what was
    # asked for. Report both, or the request looks honored when it was not.
    out.set_summary(
        n_clusters=int(n_clusters),
        n_domains_requested=int(n_domains),
        n_domains_leiden=n_domains_leiden,
        cluster_key=cluster_key,
        cluster_sizes=cluster_sizes,
        unique_domains=[_domain_label_as_number(x) for x in unique_domains],
    )
    out.set_analysis(
        build_cluster_analysis(
            cluster_sizes,
            cluster_key="domain",
            total_spots=n_spots,
            n_requested=int(n_domains),
        )
        + stage_note
        + (" " + counts_info["warning"] if counts_info.get("warning") else "")
        + (
            " The expression matrix was adata.raw.X (use_raw_counts=True)."
            if counts_info["expression_source"] == "raw.X"
            else ""
        )
        + describe_reduction(
            "spots",
            n_spots_supplied,
            n_spots,
            "the in-tissue filter (obs['in_tissue'] == 0 marks background spots)",
        )
        + identifier_rename_note(renamed)
    )
    return out.to_dict()


def main() -> None:
    args = parse_args()

    # Redirect *all* stdout during heavy work (DeepST / scanpy) to stderr,
    # so that only the final JSON goes to real stdout.
    orig_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        try:
            result = run_deepst_identify_domains(
                st_h5ad=args.st_h5ad,
                output_dir=args.output_dir,
                n_domains=args.n_domains,
                pre_epochs=args.pre_epochs,
                epochs=args.epochs,
                pca_n_comps=args.pca_n_comps,
                spatial_type=args.spatial_type,
                dist_type=args.dist_type,
                use_morphological=str_to_bool(args.use_morphological),
                use_gpu_flag=resolve_use_gpu(args.use_gpu),
                seed=args.seed,
                allow_resolution_fallback=bool(args.allow_resolution_fallback),
                use_raw_counts=bool(args.use_raw_counts),
            )
        except Exception as e:
            log("ERROR while running DeepST:")
            traceback.print_exc(file=sys.stderr)
            result = WorkerOutput.error("deepst", str(e), task="identify_domains")
    finally:
        # Restore real stdout
        sys.stdout = orig_stdout

    # Now print a single JSON line to stdout for the MCP wrapper to parse
    print(json.dumps(result))


if __name__ == "__main__":
    main()
