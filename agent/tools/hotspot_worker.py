#!/usr/bin/env python

"""
hotspot_worker.py

Worker script for running Hotspot spatial analysis on a single spatial
transcriptomics AnnData (.h5ad).

- Executed inside the Hotspot conda env: /opt/conda/envs/hotspot
- All logs and progress go to stderr.
- Stdout contains exactly one line of JSON at the end.

What the worker refuses to guess
--------------------------------
* ``layer_key`` names the matrix Hotspot models. Empty (the default) means ``adata.X``; a
  named layer that the file does not have is an error listing the layers it does have. An
  earlier revision copied ``X`` into a missing ``layers['counts']`` and said so only on stderr,
  which no successful run's payload carries.
* ``latent_obsm_key`` names the similarity space. A key absent from ``obsm`` is an error; it
  used to be swapped for ``'spatial'`` in silence, so a run asked for ``X_pca`` succeeded on
  tissue coordinates.
* The ``danb`` and ``bernoulli`` models are count models. hotspotsc 1.1.3 runs them on
  log-normalised or scaled values without complaint, so the worker checks the analysed matrix
  first: finite and non-negative for both, integer-valued for ``danb`` (and for ``bernoulli``
  unless the per-cell UMI totals come from ``obs``, since Bernoulli reads only detection from
  the matrix but uses the totals as trial counts). ``round_counts=True`` rounds near-integer
  floats; anything else needs ``model='normal'`` / ``'none'`` or a raw-count ``layer_key``.
* The zero-variance gene filter runs on the matrix Hotspot receives, sparse-aware. The old
  filter densified the whole of ``X`` (OOM on VisiumHD / Xenium) and looked at ``X`` even when
  a layer was analysed.

* Background spots flagged ``obs['in_tissue'] == 0`` (CELLxGENE Visium exports carry every array
  spot, most of them glass with ambient counts) are left out right after loading, as scanpy_spatial,
  BSP and SpaGFT do: tested beside the tissue, the tissue/background edge is itself a spatially
  autocorrelated "pattern" in every gene. They are reported in ``params.in_tissue_filter``,
  ``data.n_spots_off_tissue_dropped``, a warning and the analysis, and the annotated h5ad holds the
  analysed (in-tissue) spots.

* ``use_raw_counts=True`` models ``adata.raw.X`` (``worker_utils.choose_counts_matrix``): CELLxGENE
  exports keep the counts there beside a log-normalised or scaled X. It is refused when adata.raw is
  absent or does not hold counts, and together with a ``layer_key``. A count-model refusal of X says
  whether adata.raw holds counts. ``params.expression_source`` is ``X``, ``raw.X`` or
  ``layers['<key>']``. The ``normal`` / ``none`` models take normalised values by design and are not
  refused for them.
* Module maps: ``sc.pl.spatial`` is given the one library of a CELLxGENE ``uns['spatial']`` (which holds
  it beside a scalar ``is_single`` key, so scanpy refused to guess) and that library's spot diameter.
  Every map asked for (``n_top_modules``; 0 means none) is written, or named in a warning and the
  analysis with the reason.

Every count the payload reports is a count of what was analysed; the input gene and spot counts and
the numbers the filters removed are reported beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any

from worker_utils import (
    WorkerOutput,
    choose_counts_matrix,
    describe_reduction,
    expression_matrix_kind,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    unsupported_choice_msg,
)

# Hotspot's null models built on UMI counts; 'normal' and 'none' are not.
COUNT_MODELS = ("danb", "bernoulli")
VALID_MODELS = ("danb", "bernoulli", "normal", "none")

# Stored values examined per step by the integer check and the zero-variance filter. A VisiumHD
# slide has ~1e9 nonzeros; a whole-array ``np.rint`` (or scipy's CSC conversion for a column
# min/max) would allocate a temporary of that size beside the matrix itself.
VALUE_CHUNK = 20_000_000

# Spot diameter (coordinate units) for a module map of a slide whose uns['spatial'] states none. A
# library that states spot_diameter_fullres is drawn at that size instead (_spatial_plot_kwargs).
PLOT_SPOT_SIZE = 150


def log(msg: str) -> None:
    """Print log messages to stderr with a prefix."""
    sys.stderr.write(f"[hotspot-worker] {msg}\n")
    sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hotspot worker: spatial gene modules on a spatial AnnData (.h5ad).")

    parser.add_argument(
        "--task",
        type=str,
        default="spatial_modules",
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
        help="Directory to save Hotspot outputs.",
    )
    parser.add_argument(
        "--layer-key",
        type=str,
        default="",
        help=(
            "AnnData layer holding the matrix Hotspot models. Empty (default) analyses adata.X. "
            "A named layer that is absent is an error; the worker never substitutes X for it."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="danb",
        help=(
            "Hotspot background model: danb / bernoulli / normal / none. danb needs non-negative integer "
            "counts; bernoulli needs non-negative values, integer unless obs holds the UMI totals. Checked "
            "before the run."
        ),
    )
    parser.add_argument(
        "--round-counts",
        action="store_true",
        help=(
            "Round the analysed matrix to the nearest integer before a count model (danb/bernoulli). "
            "Off by default: non-integer values are then an error, not a warning."
        ),
    )
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        help=(
            "Model adata.raw.X instead of adata.X (CELLxGENE exports keep the counts there beside a processed "
            "X). Refused when adata.raw is absent or not counts, and together with a layer key."
        ),
    )
    parser.add_argument(
        "--latent-obsm-key",
        type=str,
        default="spatial",
        help=(
            "AnnData .obsm key defining the metric space (typically 'spatial' for Visium-like data). "
            "A key absent from obsm is an error; there is no fallback to 'spatial'."
        ),
    )
    parser.add_argument(
        "--umi-counts-obs-key",
        type=str,
        default="total_counts",
        help=(
            "AnnData .obs key containing per-cell UMI counts. If absent, totals are the row sums of the "
            "analysed matrix (Hotspot's own default) and the payload says so."
        ),
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
        help="Number of neighbors for hs.create_knn_graph.",
    )
    parser.add_argument(
        "--autocorr-fdr",
        type=float,
        default=0.05,
        help=("FDR threshold when selecting informative genes from hs.compute_autocorrelations results."),
    )
    parser.add_argument(
        "--min-gene-threshold",
        type=int,
        default=30,
        help="min_gene_threshold for hs.create_modules.",
    )
    parser.add_argument(
        "--module-fdr-threshold",
        type=float,
        default=0.05,
        help="fdr_threshold for hs.create_modules.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of parallel jobs for autocorrelations/local correlations.",
    )
    parser.add_argument(
        "--n-top-modules",
        type=int,
        default=3,
        help="Number of top modules to plot spatially (0: none; a negative value is refused).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


# --------------------------------------------------------------------------- matrix helpers


def _atomic_to_csv(df: Any, path: str, **kwargs: Any) -> None:
    """Write ``df`` as ``<path>.partial`` and rename over ``path`` once it is complete."""
    tmp = path + ".partial"
    df.to_csv(tmp, **kwargs)
    os.replace(tmp, path)


def _atomic_write_h5ad(adata: Any, path: str) -> None:
    """Write the AnnData as ``<path>.partial`` and rename over ``path`` once it is complete."""
    tmp = path + ".partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def _where_counts_could_be(adata: Any) -> str:
    """The places in this object a count matrix could live, for an error message."""
    layers = list(adata.layers.keys())
    where = f"available layers: {layers}" if layers else "this file has no layers"
    raw = getattr(adata, "raw", None)
    if raw is not None:
        try:
            kind = expression_matrix_kind(raw.X)
        except Exception:
            kind = "unreadable"
        if kind == "counts":
            where += (
                f"; adata.raw is present ({raw.n_vars} genes) and holds raw counts -- pass use_raw_counts=True "
                "(with layer_key='') to model them"
            )
        else:
            where += (
                f"; adata.raw is present ({raw.n_vars} genes) but holds {kind.replace('_', ' ')} values, not counts"
            )
    return where


def _analysed_matrix(adata: Any, layer_key: str, x_label: str = "X"):
    """The matrix Hotspot will model and the name the payload calls it: ``(matrix, source)``.

    ``layer_key == ""`` means ``adata.X`` (upstream's ``layer_key=None``). A named layer that is
    absent is refused with the layers the object does have; substituting ``X`` for it would change
    which numbers the model is fitted to without anything in the payload saying so. ``x_label`` is
    what the payload calls ``adata.X`` -- ``"raw.X"`` once use_raw_counts has put adata.raw there.
    """
    if layer_key:
        if layer_key not in adata.layers:
            raise KeyError(
                f"layer_key='{layer_key}' is not a layer of this AnnData ({_where_counts_could_be(adata)}). "
                "Pass layer_key='' (empty) to analyse adata.X, or name one of the layers listed."
            )
        return adata.layers[layer_key], f"layers['{layer_key}']"
    if adata.X is None:
        raise ValueError(
            "adata.X is empty and layer_key is '' (use X). Pass layer_key naming a layer that holds the "
            f"expression matrix ({_where_counts_could_be(adata)})."
        )
    return adata.X, x_label


def _spatial_plot_kwargs(adata: Any) -> dict:
    """The ``library_id``/``spot_size`` arguments ``sc.pl.spatial`` needs to draw this slide as it is.

    * ``library_id``: scanpy refuses to guess once ``uns['spatial']`` has more than one key ("Found
      multiple possible libraries"). CELLxGENE Visium exports carry exactly that shape -- the one library
      (a dict of images and scale factors) beside a scalar ``is_single`` flag -- so on Heart Fetal12W,
      Muscle, Skin and Thymus every module map failed and the run still said ok. When exactly one entry
      is a library it is named; with several, nothing is passed and scanpy's refusal is reported in the
      payload, not hidden.
    * ``spot_size``: the library's own ``spot_diameter_fullres`` when it states one (scanpy reads it when
      ``spot_size`` is None); the fixed ``PLOT_SPOT_SIZE`` only for a slide that states no diameter.
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


def _as_1d(values: Any):
    """A sparse or dense axis reduction as a flat numpy array."""
    import numpy as np

    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values).ravel()


def _nonconstant_genes(M: Any, chunk: int = VALUE_CHUNK):
    """Mask of genes (columns) whose values are not all identical, without densifying ``M``.

    Hotspot raises on a zero-variance gene, so these must go before it runs. A sparse matrix is
    reduced in row blocks of about ``chunk`` stored values: scipy's column min/max converts its
    operand to CSC, and doing that once for a whole VisiumHD slide would copy every nonzero. The
    dense path is only taken when the matrix already is dense, and reduces it in place.
    """
    import numpy as np
    import scipy.sparse as sp

    if not sp.issparse(M):
        dense = np.asarray(M)
        return dense.max(axis=0) != dense.min(axis=0)
    if getattr(M, "format", "") == "csc":
        return _as_1d(M.max(axis=0)) != _as_1d(M.min(axis=0))
    R = M if getattr(M, "format", "") == "csr" else M.tocsr()
    n_rows = int(R.shape[0])
    per_row = max(1.0, float(R.nnz) / max(1, n_rows))
    step = max(1, int(chunk / per_row))
    col_max = col_min = None
    for start in range(0, n_rows, step):
        block = R[start : start + step]
        bmax = _as_1d(block.max(axis=0))
        bmin = _as_1d(block.min(axis=0))
        col_max = bmax if col_max is None else np.maximum(col_max, bmax)
        col_min = bmin if col_min is None else np.minimum(col_min, bmin)
    if col_max is None:  # no rows: nothing varies
        return np.zeros(int(R.shape[1]), dtype=bool)
    return col_max != col_min


def _count_matrix_report(M: Any, chunk: int = VALUE_CHUNK) -> dict:
    """Value statistics of the stored entries of ``M``, computed in chunks.

    Returns ``min``, ``max`` (over finite values), ``n_values`` (stored entries), ``n_non_finite``,
    ``n_non_integer`` and ``max_integer_deviation``. Implicit zeros of a sparse matrix count
    towards ``min`` and ``max``.
    """
    import numpy as np
    import scipy.sparse as sp

    if sp.issparse(M):
        values = M.data
        n_total = int(M.shape[0]) * int(M.shape[1])
        has_implicit_zero = values.shape[0] < n_total
    else:
        values = np.asarray(M).ravel()
        has_implicit_zero = False
    n = int(values.shape[0])
    vmin = 0.0 if has_implicit_zero else float("inf")
    vmax = 0.0 if has_implicit_zero else float("-inf")
    n_non_finite = 0
    n_non_int = 0
    max_dev = 0.0
    integral_dtype = np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_)
    for start in range(0, n, chunk):
        block = values[start : start + chunk]
        if not integral_dtype:
            finite = np.isfinite(block)
            n_bad_finite = int(block.size - np.count_nonzero(finite))
            if n_bad_finite:
                n_non_finite += n_bad_finite
                block = block[finite]
        if block.size == 0:
            continue
        vmin = min(vmin, float(block.min()))
        vmax = max(vmax, float(block.max()))
        if not integral_dtype:
            dev = np.abs(block - np.rint(block))
            n_bad = int(np.count_nonzero(dev > 0))
            if n_bad:
                n_non_int += n_bad
                max_dev = max(max_dev, float(dev.max()))
    if vmin == float("inf"):  # no finite value at all
        vmin, vmax = 0.0, 0.0
    return {
        "min": vmin,
        "max": vmax,
        "n_values": n,
        "n_non_finite": n_non_finite,
        "n_non_integer": n_non_int,
        "max_integer_deviation": max_dev,
    }


def _round_in_place(M: Any) -> None:
    """Round the stored values of ``M`` to the nearest integer, in place."""
    import numpy as np
    import scipy.sparse as sp

    if sp.issparse(M):
        M.data[:] = np.rint(M.data)
        M.eliminate_zeros()
    else:
        dense = np.asarray(M)
        np.rint(dense, out=dense)


def _check_count_model_input(
    M: Any, model: str, source: str, where: str, round_counts: bool, umi_from_obs: bool
) -> dict:
    """Make sure a count model is fitted to counts; round on request; refuse otherwise.

    ``danb`` fits a negative binomial to the values, so they must be non-negative integers.
    ``bernoulli`` reads only detection (value > 0) from the matrix, so non-integer non-negative
    values are acceptable -- but only while the per-cell UMI totals come from ``obs``; computed
    totals would be row sums of the same non-count values, and Bernoulli uses them as the number
    of trials.

    Returns the value report of the matrix the model will see (after rounding, if any) with the
    extra fields ``n_values_rounded`` and ``integer_valued``.
    """
    report = _count_matrix_report(M)
    if report["n_non_finite"]:
        raise ValueError(
            f"model='{model}' is a count model, but {source} holds {report['n_non_finite']} NaN / infinite "
            f"values of {report['n_values']} stored. Counts are finite; point layer_key at a raw-count layer "
            f"({where}) or clean the matrix first."
        )
    if report["min"] < 0:
        raise ValueError(
            f"model='{model}' is a count model and needs non-negative counts, but {source} has a minimum "
            f"of {report['min']:g}; this looks like scaled or centred data. Point layer_key at a raw-count "
            f"layer ({where}), or pass model='normal' / model='none' for values that are not counts."
        )
    n_values_rounded = 0
    if report["n_non_integer"] > 0:
        detection_only = model == "bernoulli" and umi_from_obs
        if round_counts:
            n_values_rounded = int(report["n_non_integer"])
            log(
                f"round_counts=True: rounding {n_values_rounded} non-integer values of {source} "
                f"(max deviation {report['max_integer_deviation']:.3g}) to the nearest integer."
            )
            _round_in_place(M)
            report = _count_matrix_report(M)
        elif not detection_only:
            why = (
                "Bernoulli reads detection from any non-negative matrix, but with no per-cell UMI column in obs "
                "the totals it uses as trial counts would be row sums of these same non-count values. "
                if model == "bernoulli"
                else ""
            )
            raise ValueError(
                f"model='{model}' is a count model and needs integer counts, but {source} holds "
                f"{report['n_non_integer']} of {report['n_values']} stored values that are not integers "
                f"(max deviation from an integer {report['max_integer_deviation']:.3g}; range "
                f"{report['min']:g}..{report['max']:g}). {why}Point layer_key at a raw-count layer ({where}), "
                "pass model='normal' or model='none' for normalized / standardized values, or pass "
                "round_counts=True if these are counts stored as near-integer floats."
            )
    report["n_values_rounded"] = n_values_rounded
    report["integer_valued"] = report["n_non_integer"] == 0
    return report


# --------------------------------------------------------------------------- the pipeline


def run_hotspot_spatial_modules(
    st_h5ad: str,
    output_dir: str,
    layer_key: str,
    model: str,
    latent_obsm_key: str,
    umi_counts_obs_key: str,
    n_neighbors: int,
    autocorr_fdr: float,
    min_gene_threshold: int,
    module_fdr_threshold: float,
    n_jobs: int,
    n_top_modules: int,
    seed: int,
    round_counts: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Core Hotspot pipeline for spatial gene modules on a spatial h5ad.
    """

    import matplotlib
    import numpy as np
    import pandas as pd  # noqa: F401  # (kept in case of future use)
    import scanpy as sc

    matplotlib.use("Agg")
    import hotspot
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    layer_key = layer_key or ""

    log("Task                = spatial_modules")
    log(f"st_h5ad             = {st_h5ad}")
    log(f"output_dir          = {output_dir}")
    log(f"layer_key           = {layer_key!r} ({'adata.X' if not layer_key else 'adata.layers'})")
    log(f"model               = {model}")
    log(f"round_counts        = {round_counts}")
    log(f"use_raw_counts      = {use_raw_counts}")
    log(f"latent_obsm_key     = {latent_obsm_key}")
    log(f"umi_counts_obs_key  = {umi_counts_obs_key}")
    log(f"n_neighbors         = {n_neighbors}")
    log(f"autocorr_fdr        = {autocorr_fdr}")
    log(f"min_gene_threshold  = {min_gene_threshold}")
    log(f"module_fdr_threshold= {module_fdr_threshold}")
    log(f"n_jobs              = {n_jobs}")
    log(f"n_top_modules       = {n_top_modules}")
    log(f"seed                = {seed}")

    if model not in VALID_MODELS:
        raise ValueError(unsupported_choice_msg("model", model, VALID_MODELS))
    if int(n_top_modules) < 0:
        raise ValueError(
            f"n_top_modules must be 0 or more (got {n_top_modules}); it is how many of the top modules get a "
            "spatial map PNG (0: none)."
        )
    if use_raw_counts and layer_key:
        raise ValueError(
            f"use_raw_counts=True models adata.raw, and layer_key='{layer_key}' names a layer: they name two "
            "different matrices. Pass one of them (layer_key='' with use_raw_counts=True, or the layer alone)."
        )

    # Set seed
    np.random.seed(seed)

    # Load AnnData
    log("Loading spatial AnnData...")
    adata = sc.read_h5ad(st_h5ad)
    renamed = make_names_unique_and_report(adata)
    n_genes_input = int(adata.n_vars)
    log(f"Loaded AnnData: n_cells={adata.n_obs}, n_genes={adata.n_vars}")
    # Background spots are glass, not tissue; left out before the value scan, the gene filter and
    # the KNN graph, and counted (the payload says how many and why).
    adata, n_spots_input, n_spots_off_tissue = keep_in_tissue(adata, "spots")
    if n_spots_off_tissue:
        log(
            f"Left out {n_spots_off_tissue} of {n_spots_input} spots with obs['in_tissue'] == 0 (background); "
            f"{adata.n_obs} in-tissue spots are analysed"
        )

    out = WorkerOutput("hotspot", task="spatial_modules")

    # use_raw_counts: the counts CELLxGENE exports keep in adata.raw beside a processed X. choose_counts_matrix
    # refuses an absent adata.raw or one that is not counts; raw.var is a second gene index, so it is
    # deduplicated (and counted) again, keeping the barcode count from the load.
    counts_info = None
    x_label = "X"
    if use_raw_counts:
        adata, counts_info = choose_counts_matrix(adata, True)
        renamed = make_names_unique_and_report(
            adata, into={"n_genes_renamed": 0, "n_cells_renamed": int(renamed.get("n_cells_renamed", 0))}, axes=("var",)
        )
        n_genes_input = int(adata.n_vars)
        x_label = "raw.X"
        log(f"use_raw_counts=True: modelling adata.raw.X ({adata.n_vars} genes)")

    # The matrix Hotspot models -- named, never substituted.
    M, expression_source = _analysed_matrix(adata, layer_key, x_label)
    log(f"Hotspot models {expression_source}")

    # The similarity space is the one asked for, or nothing. Checked before the value scan so a
    # wrong key costs no pass over the matrix.
    if latent_obsm_key not in adata.obsm:
        raise KeyError(
            f"latent_obsm_key='{latent_obsm_key}' is not in adata.obsm (available keys: "
            f"{list(adata.obsm.keys())}). Name one of those; the worker does not substitute 'spatial'."
        )

    # A count model is fitted to counts. Checked here because hotspotsc does not check.
    umi_from_obs = bool(umi_counts_obs_key) and umi_counts_obs_key in adata.obs
    count_report = None
    if model in COUNT_MODELS:
        count_report = _check_count_model_input(
            M, model, expression_source, _where_counts_could_be(adata), round_counts, umi_from_obs
        )
        log(
            f"{expression_source}: {count_report['n_values']} stored values in "
            f"{count_report['min']:g}..{count_report['max']:g}, "
            + ("integer-valued" if count_report["integer_valued"] else "non-integer (bernoulli reads detection only)")
        )
        if not count_report["integer_valued"]:
            out.add_info(
                f"model='bernoulli' on {expression_source}, which holds {count_report['n_non_integer']} non-integer "
                f"values: Bernoulli uses only detection (value > 0) from the matrix and obs['{umi_counts_obs_key}'] "
                "as the per-cell UMI totals."
            )
    elif round_counts:
        record_ignored(out, "round_counts", f"model='{model}' does not model integer counts; nothing was rounded")

    # Filter zero-variance genes on the matrix Hotspot receives, without densifying it.
    keep = _nonconstant_genes(M)
    n_removed = int(adata.n_vars - int(keep.sum()))
    if n_removed:
        log(f"Filtered {n_removed} zero-variance genes")
        adata = adata[:, keep].copy()
        M, expression_source = _analysed_matrix(adata, layer_key, x_label)
    if adata.n_vars == 0:
        raise ValueError(
            f"Every one of the {n_genes_input} genes is constant in {expression_source}; Hotspot has nothing to model."
        )

    # UMI counts per cell: the obs column if present, else Hotspot's own default (row sums).
    if umi_from_obs:
        umi_counts_source = f"obs['{umi_counts_obs_key}']"
    else:
        log(f"obs['{umi_counts_obs_key}'] not found. Computing total counts as row sums of {expression_source}.")
        total_counts = _as_1d(M.sum(axis=1))
        adata.obs[umi_counts_obs_key] = total_counts.astype(float)
        umi_counts_source = f"computed: row sums of {expression_source} over the {adata.n_vars} analysed genes"

    # Initialize Hotspot (AnnData-based API, >= v1.0)
    log("Initializing Hotspot object...")
    hs = hotspot.Hotspot(
        adata,
        layer_key=layer_key if layer_key else None,
        model=model,
        latent_obsm_key=latent_obsm_key,
        umi_counts_obs_key=umi_counts_obs_key,
    )

    # Build KNN graph
    log("Creating KNN graph with hs.create_knn_graph...")
    hs.create_knn_graph(weighted_graph=False, n_neighbors=n_neighbors)

    # Compute per-gene autocorrelations
    log("Computing gene autocorrelations...")
    hs_results = hs.compute_autocorrelations(jobs=n_jobs)
    autocorr_csv = os.path.join(output_dir, "hotspot_gene_autocorrelations.csv")
    _atomic_to_csv(hs_results, autocorr_csv)
    log(f"Saved autocorrelations to {autocorr_csv}")

    # Select informative genes
    if 0 < autocorr_fdr < 1:
        sel_genes = hs_results.loc[hs_results["FDR"] < autocorr_fdr].index
        gene_selection = f"autocorrelation FDR < {autocorr_fdr}"
        log(f"Selected {len(sel_genes)} informative genes with FDR < {autocorr_fdr}.")
    else:
        sel_genes = hs_results.index
        gene_selection = f"all genes (autocorr_fdr={autocorr_fdr} is outside (0, 1), so no FDR cut was applied)"
        log(f"autocorr_fdr={autocorr_fdr} not in (0,1). Using all {len(sel_genes)} genes for local correlations.")

    if len(sel_genes) == 0:
        raise RuntimeError(f"No genes passed the autocorrelation FDR threshold ({autocorr_fdr}).")

    # Compute local correlations
    log("Computing gene-gene local correlations...")
    local_cor = hs.compute_local_correlations(sel_genes, jobs=n_jobs)
    local_cor_csv = os.path.join(output_dir, "hotspot_local_correlations_z.csv")
    _atomic_to_csv(local_cor, local_cor_csv)
    log(f"Saved local correlations to {local_cor_csv}")

    # Gene modules
    log("Creating gene modules with hs.create_modules...")
    modules = hs.create_modules(
        min_gene_threshold=min_gene_threshold,
        core_only=True,
        fdr_threshold=module_fdr_threshold,
    )
    modules_csv = os.path.join(output_dir, "hotspot_gene_modules.csv")
    _atomic_to_csv(modules, modules_csv, header=["module"])
    log(f"Saved gene modules to {modules_csv}")

    # Per-cell module scores
    log("Computing per-cell module scores...")
    module_scores = hs.calculate_module_scores()
    module_scores_csv = os.path.join(output_dir, "hotspot_module_scores_per_cell.csv")
    _atomic_to_csv(module_scores, module_scores_csv, index_label="cell")
    log(f"Saved module scores to {module_scores_csv}")

    # Annotate AnnData and save
    log("Annotating AnnData with module info and scores...")

    # Map gene->module into var
    adata.var["hotspot_module"] = modules.reindex(adata.var_names).fillna(-1).astype(int)

    # Ensure module_scores index matches obs_names
    module_scores = module_scores.loc[adata.obs_names]
    adata.obsm["hotspot_module_scores"] = module_scores.values
    adata.uns["hotspot_module_names"] = list(module_scores.columns)

    annotated_h5ad = os.path.join(output_dir, "hotspot_annotated.h5ad")
    log(f"Writing annotated AnnData to {annotated_h5ad}")
    _atomic_write_h5ad(adata, annotated_h5ad)

    # Spatial maps for the top modules. Every map promised (n_top_modules of them, or fewer when fewer
    # modules exist) is either written or named in a warning with the reason: a failed map used to be
    # logged to stderr only, so a run on a CELLxGENE slide said ok with spatial_module_plots={}.
    spatial_plots: dict[str, str] = {}
    plot_failures: dict[str, str] = {}
    top_modules: list = []
    if int(n_top_modules) > 0 and module_scores.shape[1] > 0:
        # Rank modules by mean absolute score
        mean_abs = module_scores.abs().mean(axis=0)
        sorted_modules = list(mean_abs.sort_values(ascending=False).index)
        top_modules = sorted_modules[: min(int(n_top_modules), len(sorted_modules))]
        log(f"Top modules for plotting: {top_modules}")

    if top_modules and "spatial" not in adata.obsm:
        for m in top_modules:
            plot_failures[str(m)] = "adata.obsm['spatial'] is absent, so there are no coordinates to draw it on"
    elif top_modules:
        plot_kwargs = _spatial_plot_kwargs(adata)
        if "library_id" in plot_kwargs:
            log(f"Drawing on uns['spatial'] library {plot_kwargs['library_id']!r}")
        for m in top_modules:
            col_name = f"hotspot_module_score_{m}"
            adata.obs[col_name] = module_scores[m]

            plot_path = os.path.join(output_dir, f"hotspot_spatial_module_{m}.png")
            log(f"Plotting spatial pattern for module {m} -> {plot_path}")
            try:
                sc.pl.spatial(
                    adata,
                    color=[col_name],
                    frameon=False,
                    title=f"Hotspot module {m}",
                    save=None,
                    show=False,
                    **plot_kwargs,
                )
                plt.savefig(plot_path + ".partial", format="png", bbox_inches="tight", dpi=300)
                plt.close()
                os.replace(plot_path + ".partial", plot_path)
                spatial_plots[str(m)] = plot_path
            except Exception as e:
                plt.close("all")
                plot_failures[str(m)] = f"{type(e).__name__}: {' '.join(str(e).split())[:300]}"
                log(f"WARNING: Failed to plot module {m}: {e}")

    # Summary stats
    valid_modules = [int(x) for x in sorted(set(modules.values)) if int(x) >= 0]
    n_modules = len(valid_modules)

    log(f"Hotspot finished. n_modules={n_modules}, modules={valid_modules}")

    data = {
        "n_spots": int(adata.n_obs),
        "n_spots_input": int(n_spots_input),
        "n_spots_off_tissue_dropped": int(n_spots_off_tissue),
        "n_genes": int(adata.n_vars),
        "n_genes_input": n_genes_input,
        "n_genes_zero_variance_removed": n_removed,
    }
    if count_report is not None:
        data["n_values_rounded"] = int(count_report["n_values_rounded"])
    out.set_data(**data)
    out.add_output_files(
        {
            "annotated_h5ad": annotated_h5ad,
            "gene_autocorrelations_csv": autocorr_csv,
            "local_correlations_csv": local_cor_csv,
            "gene_modules_csv": modules_csv,
            "module_scores_csv": module_scores_csv,
            # A list of paths, top module first (the module id is in each file name and in
            # summary.plotted_modules). The dict this used to pass reached the payload as its Python
            # repr -- one string no reader could split into paths -- because add_output_file str()s
            # anything that is not a list.
            "spatial_module_plots": list(spatial_plots.values()),
        }
    )
    out.add_params(
        {
            "layer_key": layer_key,
            "expression_source": expression_source,
            "model": model,
            "round_counts": bool(round_counts),
            "latent_obsm_key": latent_obsm_key,
            "umi_counts_obs_key": umi_counts_obs_key,
            "umi_counts_source": umi_counts_source,
            "n_neighbors": n_neighbors,
            "autocorr_fdr": autocorr_fdr,
            "gene_selection": gene_selection,
            "min_gene_threshold": min_gene_threshold,
            "module_fdr_threshold": module_fdr_threshold,
            "n_jobs": n_jobs,
            "n_top_modules": n_top_modules,
            "seed": seed,
        }
    )
    out.add_params(identifier_rename_params(renamed))
    out.add_params({"use_raw_counts": bool(use_raw_counts)})
    if counts_info is not None:
        record_expression_source(out, counts_info)
    elif expression_source == "X":
        if count_report is not None:
            integer_input = count_report["integer_valued"] and not count_report["n_values_rounded"]
            x_kind = "counts" if integer_input else "nonnegative_noninteger"
        else:
            x_kind = expression_matrix_kind(M)
        record_expression_source(out, {"expression_source": "X", "x_matrix_kind": x_kind, "warning": None})
    record_method(out, f"Hotspot (hotspotsc) {model} model on {expression_source}", used_fallback=False)
    record_in_tissue(out, n_spots_input, n_spots_off_tissue)
    if count_report is not None and count_report["n_values_rounded"]:
        out.add_warning(
            f"round_counts=True: {count_report['n_values_rounded']} stored values of {expression_source} were not "
            f"integers and were rounded to the nearest integer before the {model} model was fitted."
        )
    plot_note = ""
    if plot_failures:
        failed = ", ".join(f"module {m} ({why})" for m, why in plot_failures.items())
        plot_note = (
            f" NOTE: {len(plot_failures)} of the {len(top_modules)} module map PNG(s) asked for (n_top_modules="
            f"{n_top_modules}) were not written: {failed}. The module scores are in module_scores_csv and "
            "obsm['hotspot_module_scores']."
        )
        out.add_warning(plot_note.strip())
    out.set_summary(
        n_modules=int(n_modules),
        modules=valid_modules,
        n_informative_genes=int(len(sel_genes)),
        n_module_plots=len(spatial_plots),
        plotted_modules=[int(m) if str(m).lstrip("-").isdigit() else str(m) for m in spatial_plots],
    )
    out.set_analysis(
        f"Hotspot ({model} model on {expression_source}) identified {n_modules} gene modules from {len(sel_genes)} "
        f"genes selected by {gene_selection}, across {adata.n_obs} spots. "
        f"Module IDs: {valid_modules}."
        + describe_reduction(
            "spots",
            int(n_spots_input),
            int(adata.n_obs),
            "leaving out the background spots flagged obs['in_tissue'] == 0",
        )
        + describe_reduction("genes", n_genes_input, int(adata.n_vars), "the zero-variance gene filter")
        + identifier_rename_note(renamed)
        + plot_note
    )
    return out.to_dict()


def main() -> None:
    args = parse_args()

    # Redirect stdout to stderr during processing so logs don't pollute JSON
    orig_stdout = sys.stdout
    sys.stdout = sys.stderr
    error_exc = None
    try:
        try:
            result = run_hotspot_spatial_modules(
                st_h5ad=args.st_h5ad,
                output_dir=args.output_dir,
                layer_key=args.layer_key,
                model=args.model,
                latent_obsm_key=args.latent_obsm_key,
                umi_counts_obs_key=args.umi_counts_obs_key,
                n_neighbors=args.n_neighbors,
                autocorr_fdr=args.autocorr_fdr,
                min_gene_threshold=args.min_gene_threshold,
                module_fdr_threshold=args.module_fdr_threshold,
                n_jobs=args.n_jobs,
                n_top_modules=args.n_top_modules,
                seed=args.seed,
                round_counts=args.round_counts,
                use_raw_counts=args.use_raw_counts,
            )
        except Exception as e:
            log("ERROR while running Hotspot:")
            traceback.print_exc(file=sys.stderr)
            result = None
            error_msg = str(e)
            error_exc = e
    finally:
        # Restore stdout
        sys.stdout = orig_stdout

    # Print a single JSON line to stdout
    if result is None:
        WorkerOutput.emit_error("hotspot", error_msg, task="spatial_modules", exc=error_exc)
        sys.exit(1)
    else:
        print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
