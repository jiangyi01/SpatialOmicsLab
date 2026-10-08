#!/usr/bin/env python
"""
cell2location_worker.py -- the cell2location pipeline, run inside /opt/conda/envs/cell2loc_env.

What runs (nothing here is a substitute; every step is the upstream cell2location API):
  1) ``RegressionModel`` (negative-binomial regression) fitted on the scRNA reference gives the
     per-cell-type expression signatures (``inf_aver``).
  2) ``Cell2location`` maps those signatures onto the spatial AnnData, giving per-location cell
     abundances (``obsm['q05_cell_abundance_w_sf']``).
  3) Downstream add-on: KNN + Leiden on the q05 abundance gives coarse spatial domains
     (``obs['region_cluster']``). This is scanpy, not cell2location.
  4) QC / diagnostic PNGs.

Input contract (checked before anything is trained):
  * ``X`` of both files must hold raw integer counts. Every stored value is checked; a
    non-integer matrix is refused unless ``round_counts`` is set, in which case it is rounded to
    the nearest integer (``np.rint``) and the payload says so. Rounding is never silent.
  * ``labels_key`` must be an obs column of the reference; missing (NaN/empty) labels are refused
    unless ``drop_unlabeled`` is set, and the number dropped is reported.
  * ``batch_key`` (default ``"Sample"``, as in the portal) names the reference's batch column; a
    named column must exist and be fully populated. ``None`` / ``"none"`` / ``""`` fits the
    reference without a batch covariate.
  * ``X`` must not hold negative or NaN/inf values; ``round_counts`` does not change that.
  * Spots whose ``obs['in_tissue']`` is 0 (background glass; CELLxGENE Visium exports carry every
    array spot) are left out before anything is fitted, and the payload says how many
    (``params.in_tissue_filter``).

Memory: with ``batch_size_map`` unset the spatial model trains full-batch, and upstream then copies
the whole spots x genes matrix to one dense tensor (scvi's ``DeviceBackedDataSplitter``); every
training step's intermediates are that size too, and the posterior export keeps 1000 samples of every
spot's local variables until it summarises them. Before the reference model is trained the worker
estimates that peak (measured constants below), compares it with
``worker_utils.available_memory_bytes()`` and refuses with the numbers and the knob --
``batch_size_map`` -- rather than being OOM-killed hours in. With ``batch_size_map`` set, training and
the export run in minibatches of that many spots and the export summarises each batch as it goes
(``summary_frequency=1``). Nothing is ever subsampled.

Outputs: ``inf_aver.csv``, ``sp.h5ad`` and every PNG are written beside their final path and renamed
into place.

CLI (what the portal ``tools/cell2location_mcp_server.py`` calls): stdout carries only the final
JSON payload; all logs and training progress go to stderr. Run without arguments the module serves
the same tool over MCP stdio, for testing inside the env.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# Non-interactive backend so PNGs can be saved on a headless server.
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    build_deconv_analysis,
    cell_type_rename_note,
    cell_type_rename_params,
    default_output_dir,
    keep_in_tissue,
    record_in_tissue,
    record_method,
    sanitize_cell_type_names,
)
from worker_utils import drop_unlabeled as _split_unlabeled  # the parameter of the same name shadows it

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cell2location.models import Cell2location as Cell2locationModel
from cell2location.models import RegressionModel
from cell2location.utils.filtering import filter_genes
from matplotlib import rc_context
from mcp.server.fastmcp import FastMCP

# MCP instance (only used when this module is run as a server inside cell2loc_env).
mcp = FastMCP("cell2location_mcp")

METHOD_NAME = "cell2location (RegressionModel reference signatures + Cell2location spatial mapping)"

# Paper-grade training lengths (cell2location tutorial values). One source of truth for the
# pipeline signature, the in-env MCP tool and the CLI: the CLI used to default to 50/2000 while the
# portal advertised 250/30000, so a caller who passed None for either got 5x/15x less training
# than the payload's documentation promised.
DEFAULT_MAX_EPOCHS_REF = 250
DEFAULT_MAX_EPOCHS_MAP = 30000
DEFAULT_LABELS_KEY = "CellType"
# The same default as the portal. "none" (or "", or None where the caller can pass it) means
# "fit the reference without a batch covariate"; before, there was no way to ask for that at all.
DEFAULT_BATCH_KEY = "Sample"

# Spellings that mean "no batch covariate" when they arrive as a string (the CLI cannot carry None).
_NONE_SPELLINGS = ("", "none", "null")

# RegressionModel.export_posterior names its varm columns f"means_per_cluster_mu_fg_{label}"; the
# tutorial renames them back to the labels, this worker never did. The written column names are
# kept (inf_aver.csv columns, sp.h5ad obs columns and the standardizer all use them); the payload
# carries the reference's own labels beside them (summary.cell_type_labels), and the analysis text
# names cell types by the written column name without the prefix.
_SIGNATURE_COLUMN_PREFIX = "means_per_cluster_mu_fg_"

# How many stored values to examine per chunk in the integer-count check: bounded memory on a
# matrix with hundreds of millions of non-zeros, while still reading every value.
_INTEGER_CHECK_CHUNK = 1 << 24

# ---- the spatial model's memory, measured (2026-09-30, cell2loc_env: cell2location 0.1.5, scvi-tools
# 1.3.3, torch 2.9 on CPU; synthetic counts, 20k-80k spots x 500-4k genes, 10 cell types, peak RSS
# sampled every 20 ms). An "entry" is one (spot, gene) pair; a "spot site" is one per-spot value of the
# model's local variables (2 x cell types + n_groups + 3 per spot). Fitted slopes, rounded up ~10%.
# Full batch (batch_size_map unset): 66 bytes per entry of the WHOLE spots x genes matrix -- the dense
# copy of X plus mu, the Gamma-Poisson terms and their gradients, all that size, every step -- and
# ~160 bytes per spot site (fits 1.88 / 2.96 / 5.82 GB at 40k x 500 / 20k x 2k / 40k x 2k).
_TRAIN_FULL_BATCH_BYTES_PER_ENTRY = 72
_TRAIN_FULL_BATCH_BYTES_PER_SPOT_SITE = 176
# Minibatches: 168 bytes per entry of ONE batch (batch_size_map x genes), ~41 per spot site of every
# spot (the guide's per-spot parameters and their optimiser state) and the batch's own spot sites at
# the full-batch rate (fits 0.47 / 0.60 / 1.43 / 1.48 GB).
_TRAIN_MINIBATCH_BYTES_PER_ENTRY = 185
_TRAIN_MINIBATCH_BYTES_PER_SPOT_SITE = 45
# What either mode adds regardless of size (Lightning, the optimiser, the model graph).
_TRAIN_FIXED_BYTES = 100_000_000
# Posterior export: ~14 bytes per entry of one sampling batch (the dense batch and one sample's mu) ...
_EXPORT_BYTES_PER_BATCH_ENTRY = 16
# ... plus the samples of the per-spot sites, kept until they are summarised: num_samples x spots x
# (2 x cell types + n_groups + 3) float32 values, held ~2.4x over while they are concatenated and
# their quantiles taken (measured 2.37x at 200 samples x 40k spots).
_EXPORT_SAMPLE_OVERHEAD = 2.5
_POSTERIOR_NUM_SAMPLES = 1000
# Cell2location's default n_groups (never changed here): z_sr_groups_factors has that many per spot.
_C2L_N_GROUPS = 50
# The minibatch sizes a refusal may suggest, largest first.
_SUGGESTED_BATCH_SIZES = (20000, 10000, 5000, 2500, 1000, 500, 100)


# ============ helpers ============


def _log(msg: str) -> None:
    """Log one progress line to stderr (stdout is reserved for the JSON payload)."""
    print(f"[cell2location] {msg}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """
    Inside the block everything written to sys.stdout goes to sys.stderr, so Lightning / tqdm
    progress bars cannot corrupt the JSON on stdout that MCP / the portal read.
    """
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


def _none_if_unset(value: Any) -> str | None:
    """Map the CLI spellings of "no value" (None, "", "none", "null") to None; keep anything else."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _NONE_SPELLINGS:
        return None
    return text


def _validate_obs_key(adata: sc.AnnData, key: str, param_name: str, what: str) -> None:
    """Refuse a key that is not an obs column, naming the parameter and listing what is there."""
    if key in adata.obs.columns:
        return
    cols = [str(c) for c in adata.obs.columns]
    shown = ", ".join(cols[:40]) + (f", ... ({len(cols)} columns)" if len(cols) > 40 else "")
    raise ValueError(
        f"{param_name}='{key}' is not an obs column of {what}. Available columns: [{shown}]. "
        f"Pass the right {param_name}"
        + (" or batch_key=None / 'none' to fit without a batch covariate." if param_name == "batch_key" else ".")
    )


def _stored_values(X: Any) -> np.ndarray:
    """The values a matrix actually stores: ``.data`` of a sparse matrix, the flat array of a dense one."""
    import scipy.sparse as sp

    if sp.issparse(X):
        return X.data
    return np.asarray(X).ravel()


def _count_report(X: Any) -> dict[str, Any]:
    """Scan EVERY stored value of ``X`` and count what raw counts cannot contain.

    Returns ``{"n_values", "n_non_integer", "n_negative", "n_non_finite", "example"}``. Read in
    chunks so the temporaries stay bounded; nothing is sampled. An integer dtype is whole-numbered
    by construction, so only its sign is checked.
    """
    values = _stored_values(X)
    report: dict[str, Any] = {
        "n_values": int(values.size),
        "n_non_integer": 0,
        "n_negative": 0,
        "n_non_finite": 0,
        "example": None,
    }
    if values.size == 0:
        return report
    if values.dtype.kind in ("i", "u", "b"):
        if values.dtype.kind == "i":
            for start in range(0, values.size, _INTEGER_CHECK_CHUNK):
                report["n_negative"] += int(np.count_nonzero(values[start : start + _INTEGER_CHECK_CHUNK] < 0))
        return report
    for start in range(0, values.size, _INTEGER_CHECK_CHUNK):
        chunk = values[start : start + _INTEGER_CHECK_CHUNK]
        finite = np.isfinite(chunk)
        report["n_non_finite"] += int(chunk.size - np.count_nonzero(finite))
        report["n_negative"] += int(np.count_nonzero(chunk < 0))
        fractional = finite & (chunk != np.rint(chunk))
        n_here = int(np.count_nonzero(fractional))
        if n_here and report["example"] is None:
            report["example"] = float(chunk[fractional][0])
        report["n_non_integer"] += n_here
    return report


def _ensure_integer_counts(adata: sc.AnnData, name: str, round_counts: bool) -> bool:
    """Refuse non-integer ``X`` unless ``round_counts`` allows rounding it; return whether it was rounded.

    cell2location's models are Gamma-Poisson likelihoods over raw counts. A normalised or
    log-transformed matrix fed to them gives numbers that look like a result and are not one, so
    the check reads every stored value (the old code looked at the first 100 non-zeros, then applied
    ``ceil`` while logging "rounding" and told the payload nothing) and the transform is opt-in and
    reported, never applied on its own. Negative or non-finite values are refused whatever
    ``round_counts`` says: rounding cannot turn a scaled or corrupted matrix into counts.
    """
    import scipy.sparse as sp

    X = adata.X
    rep = _count_report(X)
    layers = ", ".join(str(k) for k in adata.layers.keys()) or "<none>"
    where = f"Put raw counts in X (layers present: [{layers}]; raw: {'present' if adata.raw is not None else 'absent'})"
    if rep["n_negative"] or rep["n_non_finite"]:
        raise ValueError(
            f"{name} X is not counts: {rep['n_negative']} negative and {rep['n_non_finite']} NaN/inf values among "
            f"{rep['n_values']} stored values. cell2location models raw counts, and round_counts cannot repair a "
            f"scaled or corrupted matrix. {where}."
        )
    n_bad = rep["n_non_integer"]
    if n_bad == 0:
        return False
    if not round_counts:
        raise ValueError(
            f"{name} X is not integer counts: {n_bad} of {rep['n_values']} stored values are not whole numbers "
            f"(e.g. {rep['example']!r}). cell2location models raw counts; a normalised or log-transformed matrix "
            f"would be fitted as if it were counts. {where} or pass round_counts=True to round every value to the "
            "nearest integer (np.rint) -- that transform is then reported in the payload."
        )
    _log(
        f"{name} X: {n_bad} of {rep['n_values']} stored values are not whole numbers; rounding with np.rint "
        "(round_counts=True)."
    )
    if sp.issparse(X):
        rounded = X.copy()  # stays sparse: only the stored values change
        rounded.data = np.rint(rounded.data)
        if hasattr(rounded, "eliminate_zeros"):
            rounded.eliminate_zeros()
        adata.X = rounded
    else:
        adata.X = np.rint(np.asarray(X)).astype(np.float32)
    return True


def _reference_label(column: str) -> str:
    """The reference label a signature column stands for (the export prefix removed)."""
    column = str(column)
    return column[len(_SIGNATURE_COLUMN_PREFIX) :] if column.startswith(_SIGNATURE_COLUMN_PREFIX) else column


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = Path(str(path) + ".partial")
    df.to_csv(tmp)
    os.replace(str(tmp), str(path))


def _write_h5ad_atomic(adata: sc.AnnData, path: Path) -> None:
    tmp = Path(str(path) + ".partial")
    adata.write_h5ad(tmp)
    os.replace(str(tmp), str(path))


def _savefig_atomic(save, out: Path) -> None:
    """Render a figure beside its final path with ``save(path)``, then rename it into place.

    The partial file keeps the final file's name inside a hidden ``.partial`` directory next to it
    (the same filesystem, so ``os.replace`` is atomic): matplotlib still infers PNG from the name,
    and a reader never finds a half-written figure under the final one. The directory is removed.
    """
    out = Path(out)
    tmp_dir = out.parent / ".partial"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / out.name
    try:
        save(str(tmp))
        os.replace(str(tmp), str(out))
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
        with contextlib.suppress(OSError):
            tmp_dir.rmdir()


def _mapping_memory_estimate(n_spots: int, n_genes: int, n_cell_types: int, batch_size_map: int | None) -> dict:
    """Peak bytes the spatial model needs to train and export, from the measured constants above.

    Training and the export run one after the other, so the peak is the larger of the two. Full batch
    (``batch_size_map`` None) holds the whole spots x genes matrix dense, and the export keeps every
    spot's samples; minibatches hold one batch, and the export summarises each batch as it goes.
    """
    n_spots, n_genes, n_cell_types = int(n_spots), int(n_genes), int(n_cell_types)
    sites = 2 * n_cell_types + _C2L_N_GROUPS + 3  # w_sf, u_sf_mRNA_factors, z_sr, n_s, b_s, detection_y_s
    if batch_size_map is None:
        rows = n_spots
        train = (
            _TRAIN_FULL_BATCH_BYTES_PER_ENTRY * n_spots * n_genes
            + _TRAIN_FULL_BATCH_BYTES_PER_SPOT_SITE * n_spots * sites
        )
        held = n_spots  # every spot's samples, until the end
    else:
        rows = min(int(batch_size_map), n_spots)
        train = (
            _TRAIN_MINIBATCH_BYTES_PER_ENTRY * rows * n_genes
            + _TRAIN_MINIBATCH_BYTES_PER_SPOT_SITE * n_spots * sites
            + _TRAIN_FULL_BATCH_BYTES_PER_SPOT_SITE * rows * sites
        )
        held = rows  # summary_frequency=1: one batch's samples at a time
    train += _TRAIN_FIXED_BYTES
    samples = int(_POSTERIOR_NUM_SAMPLES * held * sites * 4 * _EXPORT_SAMPLE_OVERHEAD)
    summaries = 4 * n_spots * sites * 4  # means, stds, q05, q95 of every spot
    export = _EXPORT_BYTES_PER_BATCH_ENTRY * rows * n_genes + samples + summaries
    return {
        "train_bytes": int(train),
        "export_bytes": int(export),
        "posterior_sample_bytes": int(samples),
        "peak_bytes": int(max(train, export)),
    }


def _check_mapping_memory_fits(n_spots: int, n_genes: int, n_cell_types: int, batch_size_map: int | None) -> dict:
    """Refuse, with the numbers and the knob, a spatial model the machine cannot hold.

    Called before the reference model is trained, so a run that cannot finish stops in seconds, not
    after hours of training followed by the OOM killer. Returns the estimate and the memory that was
    available (None when it cannot be read, in which case nothing is refused).
    """
    est = _mapping_memory_estimate(n_spots, n_genes, n_cell_types, batch_size_map)
    available = available_memory_bytes()
    est["available_bytes"] = None if available is None else int(available)
    if available is None or est["peak_bytes"] <= available:
        return est
    gb = 1e9
    fits = [
        b
        for b in _SUGGESTED_BATCH_SIZES
        if b < n_spots and _mapping_memory_estimate(n_spots, n_genes, n_cell_types, b)["peak_bytes"] <= available
    ]
    if batch_size_map is None:
        mode = (
            f"With batch_size_map unset the model trains full-batch: upstream copies the whole {n_spots} x {n_genes} "
            f"matrix to one dense tensor and every training step's intermediates are that size "
            f"(~{est['train_bytes'] / gb:.1f} GB), and the posterior export keeps {_POSTERIOR_NUM_SAMPLES} samples of "
            f"every spot's local variables until the end (~{est['posterior_sample_bytes'] / gb:.1f} GB)."
        )
    else:
        mode = (
            f"With batch_size_map={batch_size_map} training needs ~{est['train_bytes'] / gb:.1f} GB and the "
            f"posterior export ~{est['export_bytes'] / gb:.1f} GB; a smaller batch_size_map needs less."
        )
    if fits:
        best = fits[0]
        remedy = (
            f"Pass batch_size_map={best} (spots per minibatch; ~"
            f"{_mapping_memory_estimate(n_spots, n_genes, n_cell_types, best)['peak_bytes'] / gb:.1f} GB) to train "
            "and export in minibatches"
        )
    else:
        smallest = _SUGGESTED_BATCH_SIZES[-1]
        remedy = (
            f"Even batch_size_map={smallest} needs ~"
            f"{_mapping_memory_estimate(n_spots, n_genes, n_cell_types, smallest)['peak_bytes'] / gb:.1f} GB, because "
            "per-spot state still scales with every spot; run it where more memory is free"
        )
    raise MemoryError(
        f"cell2location's spatial model needs ~{est['peak_bytes'] / gb:.1f} GB at peak for {n_spots} spots x "
        f"{n_genes} shared genes x {n_cell_types} cell types, but ~{available / gb:.1f} GB is available. {mode} "
        f"{remedy}. The data is never subsampled."
    )


def _filter_genes_robust(
    adata_ref: sc.AnnData,
    cell_count_cutoff: int,
    cell_percentage_cutoff2: float,
    nonz_mean_cutoff: float,
) -> np.ndarray:
    """
    Normalise what ``cell2location.filter_genes`` returns across versions:

    - an AnnData (already filtered)
    - a bool mask (len == n_vars)
    - an array-like / pandas.Index of gene names

    Returns the selected gene names as ``np.ndarray[str]``. ``filter_genes`` also draws its
    selection histogram on a new figure and never closes it; that figure is closed here so it
    cannot become the canvas of a later plot.
    """
    _log("Filtering genes in reference for NB regression (cell2location defaults)...")
    selected = filter_genes(
        adata_ref,
        cell_count_cutoff=cell_count_cutoff,
        cell_percentage_cutoff2=cell_percentage_cutoff2,
        nonz_mean_cutoff=nonz_mean_cutoff,
    )
    plt.close("all")

    # 1) AnnData -> its var_names
    if isinstance(selected, sc.AnnData):
        genes_filtered = np.array(selected.var_names)
        _log(f"filter_genes returned AnnData with {genes_filtered.size} genes.")
        return genes_filtered

    # 2) pandas.Index -> gene names
    if isinstance(selected, pd.Index):
        genes_filtered = np.intersect1d(adata_ref.var_names, np.asarray(selected, dtype=str))
        _log(
            f"filter_genes returned pandas.Index with {selected.size} items; "
            f"{genes_filtered.size} of them matched adata_ref.var_names."
        )
        return genes_filtered

    # 3) numpy array / list / tuple
    if isinstance(selected, (np.ndarray, list, tuple)):
        arr = np.array(selected)
        # Bool mask
        if arr.dtype == bool and arr.shape[0] == adata_ref.n_vars:
            genes_filtered = np.array(adata_ref.var_names[arr])
            _log(f"filter_genes returned bool mask with {genes_filtered.size} True values.")
            return genes_filtered
        # otherwise: gene names
        genes_filtered = np.intersect1d(adata_ref.var_names, arr.astype(str))
        _log(
            f"filter_genes returned array-like with {arr.size} items; "
            f"{genes_filtered.size} of them matched adata_ref.var_names."
        )
        return genes_filtered

    # 4) anything else is an error
    raise TypeError(
        f"Unexpected return type from filter_genes: {type(selected)}. "
        f"Expected AnnData, pandas.Index, array-like of bool, or array-like of gene names."
    )


def _guess_img_key(adata_vis: sc.AnnData) -> str | None:
    """Pick an img_key from adata_vis.uns['spatial'] (prefer 'hires'); None when there is no image."""
    if "spatial" not in adata_vis.uns or len(adata_vis.uns["spatial"]) == 0:
        return None
    sample_key = list(adata_vis.uns["spatial"].keys())[0]
    imgs = adata_vis.uns["spatial"][sample_key].get("images", {})
    if not isinstance(imgs, dict) or len(imgs) == 0:
        return None
    if "hires" in imgs:
        return "hires"
    # otherwise the first key
    return list(imgs.keys())[0]


def _upstream_panels(draw) -> list:
    """Run an upstream cell2location plotting call and return one figure per panel it drew.

    cell2location's plotting helpers draw on the *current* axes (``plt.gca()`` / ``plt.hist2d``) and
    call ``plt.show()`` between panels, which in a notebook flushes the figure. Headless (Agg),
    ``show`` does nothing: every panel of ``RegressionModel.plot_QC`` landed on one axes, and the
    first history plot landed on the ``filter_genes`` histogram that was still the open figure --
    so ``regression_history.png`` was an ELBO curve drawn over the gene-filter histogram. Here each
    call starts on a fresh figure and each ``show`` opens the next one.

    Other helpers build a figure of their own instead: ``Cell2location.plot_spatial_QC_across_batches``
    does ``fig, axs = plt.subplots(...)`` and returns ``fig``, leaving the fresh figure empty. Those
    figures are panels too -- the one ``draw`` returns and any other it opened -- so they are kept,
    in the order they were created; only figures nothing was drawn on are dropped.
    """
    from matplotlib.figure import Figure

    plt.close("all")
    panels = [plt.figure()]
    original_show = plt.show

    def _next_panel(*_args, **_kwargs):
        panels.append(plt.figure())

    plt.show = _next_panel
    try:
        returned = draw()
    finally:
        plt.show = original_show

    candidates = list(panels)
    for number in plt.get_fignums():  # figures the upstream call opened itself (plt.subplots / plt.figure)
        fig = plt.figure(number)
        if fig not in candidates:
            candidates.append(fig)
    if isinstance(returned, Figure) and returned not in candidates:
        candidates.append(returned)  # returned, but already closed by the call that built it
    # figure numbers grow in creation order, so this is the order the panels were drawn in
    candidates.sort(key=lambda f: getattr(f, "number", float("inf")))
    drawn = [f for f in candidates if f.axes]
    for f in candidates:
        if f not in drawn:
            plt.close(f)
    return drawn


def _save_panels(panels: list, fig_dir: Path, names: list[str], figures: dict[str, str]) -> None:
    """Save panel i as ``names[i].png`` (extra panels as ``<names[0]>_<i+1>.png``) and record each."""
    for i, fig in enumerate(panels):
        name = names[i] if i < len(names) else f"{names[0]}_{i + 1}"
        out = fig_dir / f"{name}.png"
        _savefig_atomic(lambda path: fig.savefig(path, dpi=150, bbox_inches="tight"), out)  # noqa: B023
        plt.close(fig)
        figures[name] = str(out)


def _generate_plots(
    adata_ref: sc.AnnData,
    adata_vis: sc.AnnData,
    reg_model: RegressionModel,
    c2l_model: Cell2locationModel,
    ref_run_dir: Path,
    map_run_dir: Path,
    cell_type_names: list[str],
) -> dict[str, str]:
    """
    Save the tutorial's key QC / result plots as PNGs; return {name: path}.

    Every figure is drawn on a figure of its own (see ``_upstream_panels``). The reference QC is two
    panels upstream: ``regression_QC.png`` (reconstruction accuracy) and
    ``regression_QC_signatures.png`` (estimated signatures vs. per-cluster mean expression).
    """
    figures: dict[str, str] = {}

    fig_dir_ref = ref_run_dir / "figures"
    fig_dir_map = map_run_dir / "figures"
    fig_dir_ref.mkdir(parents=True, exist_ok=True)
    fig_dir_map.mkdir(parents=True, exist_ok=True)

    # ---------- 1. RegressionModel: training history ----------
    try:
        _log("Plotting RegressionModel training history...")
        # skip the first 20 epochs; ax=gca() of the fresh figure, never the filter_genes histogram
        panels = _upstream_panels(lambda: reg_model.plot_history(20, ax=plt.gca()))
        _save_panels(panels, fig_dir_ref, ["regression_history"], figures)
    except Exception as e:
        _log(f"Warning: could not plot RegressionModel history: {e}")

    # ---------- 2. RegressionModel: QC (two panels upstream) ----------
    try:
        _log("Plotting RegressionModel QC...")
        panels = _upstream_panels(reg_model.plot_QC)
        _save_panels(panels, fig_dir_ref, ["regression_QC", "regression_QC_signatures"], figures)
    except Exception as e:
        _log(f"Warning: could not plot RegressionModel QC: {e}")

    # ---------- 3. Cell2location: training history ----------
    try:
        _log("Plotting Cell2location training history...")
        panels = _upstream_panels(lambda: c2l_model.plot_history(50, ax=plt.gca()))  # skip the first 50 steps
        _save_panels(panels, fig_dir_map, ["mapping_history"], figures)
    except Exception as e:
        _log(f"Warning: could not plot Cell2location history: {e}")

    # ---------- 4. Cell2location: QC ----------
    try:
        _log("Plotting Cell2location QC...")
        panels = _upstream_panels(c2l_model.plot_QC)
        _save_panels(panels, fig_dir_map, ["mapping_QC"], figures)
    except Exception as e:
        _log(f"Warning: could not plot Cell2location QC: {e}")

    # ---------- 5. Cell2location: spatial QC across batches (when the version has it) ----------
    try:
        if hasattr(c2l_model, "plot_spatial_QC_across_batches"):
            _log("Plotting Cell2location spatial QC across batches...")
            panels = _upstream_panels(c2l_model.plot_spatial_QC_across_batches)
            _save_panels(panels, fig_dir_map, ["mapping_spatial_QC_across_batches"], figures)
    except Exception as e:
        _log(f"Warning: could not plot spatial QC across batches: {e}")

    # ---------- 6. Visualising cell abundance in spatial coordinates ----------
    try:
        _log("Plotting spatial abundance of selected cell types...")
        # the first sample
        samples = adata_vis.obs["sample"].astype(str).unique().tolist()
        sample_id = samples[0]
        slide = adata_vis[adata_vis.obs["sample"].astype(str) == sample_id].copy()

        img_key = _guess_img_key(adata_vis)

        # the first 8 cell types
        types_to_plot = list(cell_type_names)[:8]
        if len(types_to_plot) > 0:
            with rc_context({"axes.facecolor": "black", "figure.figsize": [4.5, 5]}):
                sc.pl.spatial(
                    slide,
                    color=types_to_plot,
                    ncols=min(4, len(types_to_plot)),
                    size=1.3,
                    img_key=img_key,
                    vmin=0,
                    vmax="p99.2",
                    show=False,
                )
                out = fig_dir_map / "spatial_cell_abundance.png"
                _savefig_atomic(lambda path: plt.savefig(path, dpi=150, bbox_inches="tight"), out)
                plt.close()
                figures["spatial_cell_abundance"] = str(out)
    except Exception as e:
        _log(f"Warning: could not plot spatial cell abundance: {e}")

    # ---------- 7. UMAP & region_cluster / sample ----------
    try:
        _log("Computing UMAP on cell2location abundance and plotting regions/sample...")
        if "X_umap" not in adata_vis.obsm:
            sc.tl.umap(adata_vis, min_dist=0.3, spread=1.0)

        # UMAP: region_cluster
        with rc_context({"axes.facecolor": "white", "figure.figsize": [8, 8]}):
            sc.pl.umap(
                adata_vis,
                color=["region_cluster"],
                size=30,
                legend_loc="on data",
                legend_fontsize=10,
                show=False,
            )
            out = fig_dir_map / "umap_region_cluster.png"
            _savefig_atomic(lambda path: plt.savefig(path, dpi=150, bbox_inches="tight"), out)
            plt.close()
            figures["umap_region_cluster"] = str(out)

        # UMAP: sample
        with rc_context({"axes.facecolor": "white", "figure.figsize": [8, 8]}):
            sc.pl.umap(
                adata_vis,
                color=["sample"],
                size=30,
                legend_fontsize=10,
                show=False,
            )
            out = fig_dir_map / "umap_sample.png"
            _savefig_atomic(lambda path: plt.savefig(path, dpi=150, bbox_inches="tight"), out)
            plt.close()
            figures["umap_sample"] = str(out)

        # spatial: region_cluster
        img_key = _guess_img_key(adata_vis)
        with rc_context({"axes.facecolor": "black", "figure.figsize": [4.5, 5]}):
            sc.pl.spatial(
                adata_vis,
                color=["region_cluster"],
                size=1.3,
                img_key=img_key,
                alpha=0.5,
                show=False,
            )
            out = fig_dir_map / "spatial_region_cluster.png"
            _savefig_atomic(lambda path: plt.savefig(path, dpi=150, bbox_inches="tight"), out)
            plt.close()
            figures["spatial_region_cluster"] = str(out)

    except Exception as e:
        _log(f"Warning: could not plot UMAP/spatial region clusters: {e}")

    return figures


# ============ Main pipeline ============


def _run_cell2location(
    sc_h5ad_path: str,
    spatial_h5ad_path: str,
    results_dir: str = default_output_dir(),
    labels_key: str = DEFAULT_LABELS_KEY,
    batch_key: str | None = DEFAULT_BATCH_KEY,
    # cell2location hyper-parameters, the tutorial's recommended values
    n_cells_per_location: int = 30,
    detection_alpha: float = 20.0,
    # Training epochs -- paper-grade defaults (cell2location tutorial values).
    # 30000 ST epochs on CPU is slow but required for posterior convergence.
    max_epochs_ref: int = DEFAULT_MAX_EPOCHS_REF,
    max_epochs_map: int = DEFAULT_MAX_EPOCHS_MAP,
    # Reference gene filter thresholds for the NB regression (from the tutorial)
    cell_count_cutoff: int = 5,
    cell_percentage_cutoff2: float = 0.03,
    nonz_mean_cutoff: float = 1.12,
    # Downstream Leiden "spatial domain" parameters
    n_neighbors: int = 15,
    leiden_resolution: float = 1.1,
    # Input-contract switches (default off: refuse rather than transform / drop silently)
    round_counts: bool = False,
    drop_unlabeled: bool = False,
    # Spatial-model minibatch size; None = full batch (upstream's default, and this worker's)
    batch_size_map: int | None = None,
) -> dict[str, Any]:
    """
    Run the full cell2location pipeline and return the payload dict.

    The payload carries:
    - output_files.reference_inf_aver_csv: the reference signatures (inf_aver) as CSV
    - output_files.spatial_h5ad: spatial mapping + region_cluster (sp.h5ad)
    - output_files.figures: a list of PNG paths
    - data: n_reference_cells, n_reference_genes, n_spots, n_genes (after filtering), the input sizes
    - params: every effective setting, params.method, what was rounded / dropped, the gene selection,
      the spatial model's batch size, the posterior-export sample kwargs and the memory estimate
    - summary.cell_type_names: the written column names (export prefix + HDF5-safe rewrite), unchanged
    - summary.cell_type_labels: the same cell types as the reference spelled them (prefix removed, not
      sanitised), in the same order; the analysis text names cell types in the rewritten spelling
      without the prefix, as its rename NOTE says
    """
    # ---------- paths and output directories ----------
    sc_path = Path(sc_h5ad_path).expanduser().resolve()
    spatial_path = Path(spatial_h5ad_path).expanduser().resolve()

    if not sc_path.exists():
        raise FileNotFoundError(f"sc_h5ad_path not found: {sc_path}")
    if not spatial_path.exists():
        raise FileNotFoundError(f"spatial_h5ad_path not found: {spatial_path}")

    labels_key = "" if labels_key is None else str(labels_key).strip()
    if not labels_key:
        raise ValueError(
            "labels_key is required: name the reference obs column that holds the cell-type labels "
            f"(the default is '{DEFAULT_LABELS_KEY}'). cell2location's RegressionModel learns one signature per label."
        )
    batch_key = _none_if_unset(batch_key)
    if batch_size_map is not None:
        if isinstance(batch_size_map, bool) or int(batch_size_map) != batch_size_map or int(batch_size_map) < 1:
            raise ValueError(
                f"batch_size_map={batch_size_map!r}: give a whole number of spots per minibatch (1 or more), or "
                "leave it unset to train the spatial model full-batch."
            )
        batch_size_map = int(batch_size_map)

    results_dir_path = Path(results_dir).expanduser().resolve()
    ref_run_dir = results_dir_path / "reference_signatures"
    map_run_dir = results_dir_path / "cell2location_map"
    for d in (results_dir_path, ref_run_dir, map_run_dir):
        d.mkdir(parents=True, exist_ok=True)

    _log(f"Using results_dir: {results_dir_path}")
    _log(f"Reference output dir : {ref_run_dir}")
    _log(f"Spatial output dir    : {map_run_dir}")

    # ---------- 1. load ----------
    _log(f"Reading reference AnnData: {sc_path}")
    adata_ref = sc.read_h5ad(sc_path)

    _log(f"Reading spatial AnnData:   {spatial_path}")
    adata_vis = sc.read_h5ad(spatial_path)

    n_ref_cells_input = int(adata_ref.n_obs)
    n_ref_genes_input = int(adata_ref.n_vars)
    n_vis_genes_input = int(adata_vis.n_vars)
    _log(f"Reference: {adata_ref.n_obs} cells x {adata_ref.n_vars} genes")
    _log(f"Spatial:   {adata_vis.n_obs} locations x {adata_vis.n_vars} genes")

    # Background glass (obs['in_tissue'] == 0) is not tissue: it is left out before anything is
    # checked or fitted, and the payload says how many (params.in_tissue_filter + a warning).
    adata_vis, n_spots_input, n_spots_off_tissue = keep_in_tissue(adata_vis, "spots")
    if n_spots_off_tissue:
        _log(
            f"Left out {n_spots_off_tissue} of {n_spots_input} spots with obs['in_tissue'] == 0 (background); "
            f"mapping the {adata_vis.n_obs} in-tissue spots."
        )

    # ---------- 1b. the label / batch columns, before any gene is filtered or anything is trained ----------
    _validate_obs_key(adata_ref, labels_key, "labels_key", "the reference")
    if batch_key is not None:
        _validate_obs_key(adata_ref, batch_key, "batch_key", "the reference")
        n_batch_missing = int(pd.isna(adata_ref.obs[batch_key]).sum())
        if n_batch_missing:
            raise ValueError(
                f"batch_key='{batch_key}': {n_batch_missing} of {adata_ref.n_obs} reference cells have no value in "
                "that column. Fill it, choose another column, or pass batch_key=None / 'none' to fit without a "
                "batch covariate."
            )

    keep_mask, n_dropped_unlabeled = _split_unlabeled(
        adata_ref.obs[labels_key].values, drop_unlabeled, f"reference cells (labels_key='{labels_key}')"
    )
    if n_dropped_unlabeled:
        adata_ref = adata_ref[keep_mask].copy()
        _log(f"Dropped {n_dropped_unlabeled} reference cells with no label in '{labels_key}' (drop_unlabeled=True).")

    # A categorical column can carry categories no cell uses (a subset reference keeps the parent's
    # categories); scvi-tools registers every category as a class, so cell2location would fit and
    # report a signature for a cell type with zero cells. Re-derive the classes from the cells present.
    label_values = adata_ref.obs[labels_key]
    n_declared = int(len(label_values.cat.categories)) if hasattr(label_values, "cat") else None
    adata_ref.obs[labels_key] = pd.Categorical(label_values.astype(str).values)
    label_classes = list(adata_ref.obs[labels_key].cat.categories)
    n_unused_categories = (n_declared - len(label_classes)) if n_declared is not None else 0
    if n_unused_categories > 0:
        _log(f"Dropped {n_unused_categories} unused categories of '{labels_key}' (no cells carry them).")
    if len(label_classes) < 2:
        raise ValueError(
            f"labels_key='{labels_key}' has {len(label_classes)} class after removing missing labels; deconvolution "
            "needs at least two cell types. Check the reference labels."
        )

    # ---------- 1c. integer counts (GammaPoisson likelihoods) -- checked over every stored value ----------
    rounded_inputs: list[str] = []
    for adata_tmp, name in [(adata_ref, "reference"), (adata_vis, "spatial")]:
        if _ensure_integer_counts(adata_tmp, name.capitalize(), round_counts):
            rounded_inputs.append(name)

    # ---------- 2. gene filtering + alignment ----------
    genes_filtered = _filter_genes_robust(
        adata_ref,
        cell_count_cutoff=cell_count_cutoff,
        cell_percentage_cutoff2=cell_percentage_cutoff2,
        nonz_mean_cutoff=nonz_mean_cutoff,
    )
    _log(f"{genes_filtered.size} genes passed filter in reference.")

    # intersect with the spatial genes
    shared = np.intersect1d(genes_filtered, adata_vis.var_names)
    if shared.size == 0:
        ref_example = ", ".join(str(g) for g in genes_filtered[:3]) or "<none>"
        vis_example = ", ".join(str(g) for g in adata_vis.var_names[:3]) or "<none>"
        raise ValueError(
            f"No shared genes between reference and spatial after filtering: "
            f"{genes_filtered.size} reference genes look like [{ref_example}], "
            f"{adata_vis.n_vars} spatial genes look like [{vis_example}]. "
            "Check that both files name genes the same way -- Ensembl IDs on one side and gene "
            "symbols on the other is the usual cause -- and convert one side before rerunning."
        )
    _log(f"{shared.size} genes shared between reference and spatial after filtering.")

    adata_ref = adata_ref[:, shared].copy()
    adata_vis = adata_vis[:, shared].copy()

    _log(f"Reference (filtered&shared): {adata_ref.n_obs} x {adata_ref.n_vars}")
    _log(f"Spatial   (filtered&shared): {adata_vis.n_obs} x {adata_vis.n_vars}")

    # ---------- 2b. will the spatial model fit? (before hours of reference training, not after) ----------
    memory = _check_mapping_memory_fits(adata_vis.n_obs, adata_vis.n_vars, len(label_classes), batch_size_map)
    _log(
        f"Spatial model memory estimate: ~{memory['peak_bytes'] / 1e9:.2f} GB at peak "
        f"({'full batch' if batch_size_map is None else f'minibatches of {batch_size_map} spots'}); "
        + (
            "available memory unknown."
            if memory["available_bytes"] is None
            else f"~{memory['available_bytes'] / 1e9:.1f} GB available."
        )
    )

    # ---------- 3. reference NB RegressionModel ----------
    _log(f"Setting up RegressionModel with labels_key='{labels_key}', batch_key={batch_key!r}.")
    RegressionModel.setup_anndata(
        adata=adata_ref,
        batch_key=batch_key,
        labels_key=labels_key,
    )

    reg_model = RegressionModel(adata_ref)
    _log(f"Training RegressionModel (max_epochs={max_epochs_ref}) on reference data...")
    reg_model.train(
        max_epochs=max_epochs_ref,
        batch_size=2500,
        train_size=1.0,
    )
    _log("RegressionModel training finished.")

    _log("Exporting reference posterior (only 'means' to varm, no q05/q95 to avoid HDF5 issues)...")
    adata_ref = reg_model.export_posterior(
        adata_ref,
        sample_kwargs={"num_samples": 1000, "batch_size": 2500},
        add_to_varm=["means"],  # only the means: the q05/q95 keys are not HDF5-safe
        use_quantiles=False,
    )

    # inf_aver (gene x cell_type)
    if "means_per_cluster_mu_fg" in adata_ref.varm:
        inf_aver_df = adata_ref.varm["means_per_cluster_mu_fg"].copy()
    else:
        # older layout: one var column per cell type
        cols = [c for c in adata_ref.var.columns if c.startswith("means_per_cluster_mu_fg_")]
        if not cols:
            raise RuntimeError("Cannot find 'means_per_cluster_mu_fg' in adata_ref.varm or adata_ref.var.")
        inf_aver_df = adata_ref.var[cols].copy()
        new_cols = [c.replace("means_per_cluster_mu_fg_", "") for c in cols]
        inf_aver_df.columns = new_cols

    factor_names_orig = list(inf_aver_df.columns)
    _log("Original cell type names from reference model:")
    _log(", ".join(factor_names_orig))

    # ---------- 4. sanitise cell type names ('/' and friends are not HDF5-safe) ----------
    safe_names, ct_renames = sanitize_cell_type_names(factor_names_orig)
    if ct_renames:
        _log(f"Sanitised {len(ct_renames)} cell type name(s) (for HDF5 safety):")
        for o, s in ct_renames.items():
            _log(f"  '{o}' -> '{s}'")

    inf_aver_df.columns = safe_names
    inf_aver = inf_aver_df  # gene x cell_type

    # reference signatures as CSV (no reference h5ad: HDF5 key problems)
    inf_aver_out = ref_run_dir / "inf_aver.csv"
    _log(f"Writing reference signatures (inf_aver) to {inf_aver_out}")
    _write_csv_atomic(inf_aver, inf_aver_out)

    # ---------- 5. spatial mapping (Cell2location) ----------
    # the spatial object needs a 'sample' column
    if "sample" not in adata_vis.obs.columns:
        if "spatial" in adata_vis.uns and len(adata_vis.uns["spatial"]) > 0:
            sample_id = list(adata_vis.uns["spatial"].keys())[0]
        else:
            sample_id = "sample_0"
        _log(f"Spatial AnnData missing obs['sample'], creating single sample: {sample_id}")
        adata_vis.obs["sample"] = sample_id

    _log("Setting up Cell2locationModel on spatial data (batch_key='sample').")
    Cell2locationModel.setup_anndata(
        adata=adata_vis,
        batch_key="sample",
    )

    c2l_model = Cell2locationModel(
        adata_vis,
        cell_state_df=inf_aver,  # gene x cell_type
        N_cells_per_location=n_cells_per_location,
        detection_alpha=detection_alpha,
    )

    # batch_size_map=None is full batch, as it always was (upstream then copies the whole matrix to one
    # dense tensor -- the estimate above refused it if that cannot fit). A number trains in minibatches
    # of that many spots; one epoch is then n_spots / batch_size_map steps.
    _log(
        f"Training Cell2location spatial model (max_epochs={max_epochs_map}, "
        f"batch_size={'full batch' if batch_size_map is None else batch_size_map})..."
    )
    c2l_model.train(
        max_epochs=max_epochs_map,
        batch_size=batch_size_map,
        train_size=1.0,
    )
    _log("Cell2location spatial model training finished.")

    # Unset: one sampling batch of every spot, as before. Set: batches of that size, each summarised as
    # soon as it is sampled (summary_frequency=1), so the export never holds every spot's 1000 samples.
    posterior_sample_kwargs: dict[str, Any] = {
        "num_samples": _POSTERIOR_NUM_SAMPLES,
        "batch_size": adata_vis.n_obs if batch_size_map is None else batch_size_map,
    }
    if batch_size_map is not None:
        posterior_sample_kwargs["summary_frequency"] = 1
    _log(f"Exporting spatial posterior (cell abundance) with {posterior_sample_kwargs}...")
    adata_vis = c2l_model.export_posterior(
        adata_vis,
        sample_kwargs=dict(posterior_sample_kwargs),
    )

    if "q05_cell_abundance_w_sf" not in adata_vis.obsm:
        present = ", ".join(sorted(adata_vis.obsm.keys())) or "<empty>"
        raise RuntimeError(
            "q05_cell_abundance_w_sf not found in adata_vis.obsm after export_posterior; "
            f"obsm holds [{present}]. Check the installed cell2location version and the "
            "export_posterior call -- other versions write the abundance under a different key."
        )

    # ---------- 6. downstream: Leiden spatial domains ----------
    _log("Computing spatial domains via Leiden on q05_cell_abundance_w_sf...")

    # one obs column per cell type, for plotting
    adata_vis.obs[safe_names] = adata_vis.obsm["q05_cell_abundance_w_sf"]

    sc.pp.neighbors(
        adata_vis,
        use_rep="q05_cell_abundance_w_sf",
        n_neighbors=n_neighbors,
    )
    sc.tl.leiden(adata_vis, resolution=leiden_resolution)
    adata_vis.obs["region_cluster"] = adata_vis.obs["leiden"].astype("category")

    # ---------- 7. plots (reference + mapping + downstream) ----------
    figures = _generate_plots(
        adata_ref=adata_ref,
        adata_vis=adata_vis,
        reg_model=reg_model,
        c2l_model=c2l_model,
        ref_run_dir=ref_run_dir,
        map_run_dir=map_run_dir,
        cell_type_names=safe_names,
    )

    # ---------- 8. the spatial result ----------
    sp_out_path = map_run_dir / "sp.h5ad"
    _log(f"Writing spatial AnnData (with region_cluster) to {sp_out_path} ...")
    _write_h5ad_atomic(adata_vis, sp_out_path)

    if not sp_out_path.exists():
        raise RuntimeError(f"Spatial output file not found after write: {sp_out_path}")

    _log("cell2location pipeline completed successfully.")

    n_cell_types = int(len(inf_aver.columns))
    cell_type_names_list = list(inf_aver.columns)
    # Two spellings of each cell type, both in column order:
    #  * summary.cell_type_labels -- the label exactly as the reference spelled it (export prefix
    #    removed, NOT sanitised: 'Lymph/Endo' stays 'Lymph/Endo'), so a caller can map a column back
    #    to the reference's own label;
    #  * the analysis text -- the written column name without the export prefix, i.e. the rewritten
    #    (HDF5-safe) spelling, which is the spelling the rename NOTE below says the names use.
    cell_type_labels = [_reference_label(n) for n in factor_names_orig]
    column_labels = [_reference_label(n) for n in cell_type_names_list]
    # the rename NOTE quotes one rename; say it in labels, not in the export-prefixed column names
    label_renames = {_reference_label(o): _reference_label(r) for o, r in ct_renames.items()}
    n_spatial = int(adata_vis.n_obs)
    n_regions = int(adata_vis.obs["region_cluster"].nunique())

    # dominant cell type per spot, for the analysis text
    try:
        abundance = adata_vis.obsm["q05_cell_abundance_w_sf"]
        dominant_idx = np.argmax(abundance.values if hasattr(abundance, "values") else abundance, axis=1)
        dominant_ct = [column_labels[i] for i in dominant_idx]
        from collections import Counter

        dominant_counts = dict(Counter(dominant_ct))
    except Exception:
        dominant_counts = None

    out = WorkerOutput("cell2location", task="deconvolution")
    out.set_data(
        n_reference_cells=int(adata_ref.n_obs),
        n_reference_genes=int(adata_ref.n_vars),
        n_spots=n_spatial,
        n_spots_input=int(n_spots_input),
        n_genes=int(adata_vis.n_vars),
        n_reference_cells_input=n_ref_cells_input,
        n_reference_genes_input=n_ref_genes_input,
        n_genes_input=n_vis_genes_input,
    )
    out.add_output_files(
        {
            "reference_inf_aver_csv": str(inf_aver_out),
            "spatial_h5ad": str(sp_out_path),
            # a list of paths, as every other worker publishes figures: a dict here went through
            # add_output_file's str() and reached the payload as one Python-repr string, not paths
            "figures": list(figures.values()),
        }
    )
    out.add_params(
        {
            "n_cells_per_location": n_cells_per_location,
            "detection_alpha": detection_alpha,
            "max_epochs_ref": max_epochs_ref,
            "max_epochs_map": max_epochs_map,
            "cell_count_cutoff": cell_count_cutoff,
            "cell_percentage_cutoff2": cell_percentage_cutoff2,
            "nonz_mean_cutoff": nonz_mean_cutoff,
            "n_neighbors": n_neighbors,
            "leiden_resolution": leiden_resolution,
            "labels_key": labels_key,
            "batch_key": batch_key,
            "round_counts": round_counts,
            "rounded_inputs": rounded_inputs,
            "drop_unlabeled": drop_unlabeled,
            "n_reference_cells_dropped_unlabeled": int(n_dropped_unlabeled),
            "n_unused_label_categories_dropped": int(n_unused_categories),
            "n_genes_passed_reference_filter": int(genes_filtered.size),
            "n_genes_shared": int(shared.size),
            "batch_size_map": batch_size_map,
            "posterior_sample_kwargs": dict(posterior_sample_kwargs),
            "memory_estimate": dict(memory),
        }
    )
    out.add_params(cell_type_rename_params(ct_renames))
    record_method(out, METHOD_NAME)
    record_in_tissue(out, n_spots_input, n_spots_off_tissue)
    for name in rounded_inputs:
        out.add_warning(
            f"{name} X held non-integer values and was rounded to the nearest integer (np.rint) because "
            "round_counts=True; cell2location was fitted on the rounded matrix."
        )
    if n_dropped_unlabeled:
        out.add_warning(
            f"{n_dropped_unlabeled} of {n_ref_cells_input} reference cells had no label in '{labels_key}' and were "
            "left out (drop_unlabeled=True)."
        )
    if n_unused_categories > 0:
        out.add_warning(
            f"{n_unused_categories} categories of '{labels_key}' had no cells and were not fitted as cell types."
        )
    out.set_summary(
        n_cell_types=n_cell_types,
        cell_type_names=cell_type_names_list,
        cell_type_labels=cell_type_labels,
        n_regions=n_regions,
    )
    gene_selection_note = (
        f" Gene selection: {genes_filtered.size} of {n_ref_genes_input} reference genes passed cell2location's "
        f"filter_genes (cell_count_cutoff={cell_count_cutoff}, cell_percentage_cutoff2={cell_percentage_cutoff2}, "
        f"nonz_mean_cutoff={nonz_mean_cutoff}); {shared.size} of those are present among the {n_vis_genes_input} "
        "spatial genes and were used for both models."
    )
    out.set_analysis(
        build_deconv_analysis(
            n_celltypes=n_cell_types,
            dominant_counts=dominant_counts,
            total_spots=n_spatial,
            method_name="Cell2location",
        )
        + gene_selection_note
        + (
            f" {n_spots_off_tissue} of {n_spots_input} spots were background (obs['in_tissue'] == 0) and were left "
            f"out; the {n_spatial} in-tissue spots were mapped."
            if n_spots_off_tissue
            else ""
        )
        + cell_type_rename_note(label_renames)
        + (
            " summary.cell_type_labels gives each cell type as the reference spelled it, in the order of "
            "summary.cell_type_names."
            if label_renames
            else ""
        )
    )
    return out.to_dict()


# ============ MCP tool (when this module is run as a server inside cell2loc_env) ============


@mcp.tool()
def run_cell2location(
    sc_h5ad_path: str,
    spatial_h5ad_path: str,
    results_dir: str = default_output_dir(),
    labels_key: str = DEFAULT_LABELS_KEY,
    batch_key: str | None = DEFAULT_BATCH_KEY,
    n_cells_per_location: int = 30,
    detection_alpha: float = 20.0,
    max_epochs_ref: int = DEFAULT_MAX_EPOCHS_REF,
    max_epochs_map: int = DEFAULT_MAX_EPOCHS_MAP,
    cell_count_cutoff: int = 5,
    cell_percentage_cutoff2: float = 0.03,
    nonz_mean_cutoff: float = 1.12,
    n_neighbors: int = 15,
    leiden_resolution: float = 1.1,
    round_counts: bool = False,
    drop_unlabeled: bool = False,
    batch_size_map: int | None = None,
) -> dict[str, Any]:
    """
    MCP tool: run ``_run_cell2location`` with stdout redirected to stderr for the whole run (so
    progress bars and warnings cannot break JSON-RPC) and every exception returned as a JSON error.
    """
    with _redirect_stdout_to_stderr():
        try:
            return _run_cell2location(
                sc_h5ad_path=sc_h5ad_path,
                spatial_h5ad_path=spatial_h5ad_path,
                results_dir=results_dir,
                labels_key=labels_key,
                batch_key=batch_key,
                n_cells_per_location=n_cells_per_location,
                detection_alpha=detection_alpha,
                max_epochs_ref=max_epochs_ref,
                max_epochs_map=max_epochs_map,
                cell_count_cutoff=cell_count_cutoff,
                cell_percentage_cutoff2=cell_percentage_cutoff2,
                nonz_mean_cutoff=nonz_mean_cutoff,
                n_neighbors=n_neighbors,
                leiden_resolution=leiden_resolution,
                round_counts=round_counts,
                drop_unlabeled=drop_unlabeled,
                batch_size_map=batch_size_map,
            )
        except Exception as e:
            import traceback as tb_mod

            tb = tb_mod.format_exc()
            _log("==== Exception inside run_cell2location MCP tool ====")
            _log(str(e))
            _log(tb)
            return WorkerOutput.error("cell2location", str(e), task="deconvolution")


# ============ CLI entry point (what the portal calls) ============


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run cell2location (reference NB regression + mapping + spatial domains + plots)"
    )
    parser.add_argument("--sc-h5ad", required=True, help="Path to single-cell reference h5ad")
    parser.add_argument("--spatial-h5ad", required=True, help="Path to Visium/spatial h5ad")
    parser.add_argument(
        "--results-dir",
        default=default_output_dir(),
        help="Directory to save results (reference_signatures / cell2location_map)",
    )
    parser.add_argument(
        "--labels-key",
        default=DEFAULT_LABELS_KEY,
        help="Reference obs column with the cell-type labels (required to exist).",
    )
    parser.add_argument(
        "--batch-key",
        default=DEFAULT_BATCH_KEY,
        help=(
            "Reference obs column with the batch/sample id (default 'Sample', as in the portal); "
            "pass 'none' to fit without a batch covariate."
        ),
    )
    parser.add_argument("--max-epochs-ref", type=int, default=DEFAULT_MAX_EPOCHS_REF)
    parser.add_argument("--max-epochs-map", type=int, default=DEFAULT_MAX_EPOCHS_MAP)
    parser.add_argument("--n-cells-per-location", type=int, default=30)
    parser.add_argument("--detection-alpha", type=float, default=20.0)
    parser.add_argument("--cell-count-cutoff", type=int, default=5)
    parser.add_argument("--cell-percentage-cutoff2", type=float, default=0.03)
    parser.add_argument("--nonz-mean-cutoff", type=float, default=1.12)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--leiden-resolution", type=float, default=1.1)
    parser.add_argument(
        "--round-counts",
        action="store_true",
        default=False,
        help="Round a non-integer X to the nearest integer (np.rint) instead of refusing it; reported in the payload.",
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        default=False,
        help="Leave out reference cells whose label is missing instead of refusing the reference; count reported.",
    )
    parser.add_argument(
        "--batch-size-map",
        type=int,
        default=None,
        help="Spots per minibatch for the spatial model's training and posterior export. Omitted: full batch "
        "(the whole spots x genes matrix is held dense; refused up front, with the numbers, if it cannot fit).",
    )
    return parser


def _cli_main() -> None:
    args = _build_parser().parse_args()

    # In CLI mode too, training / progress output goes to stderr so stdout holds only the final JSON.
    with _redirect_stdout_to_stderr():
        try:
            summary = _run_cell2location(
                sc_h5ad_path=args.sc_h5ad,
                spatial_h5ad_path=args.spatial_h5ad,
                results_dir=args.results_dir,
                labels_key=args.labels_key,
                batch_key=args.batch_key,
                n_cells_per_location=args.n_cells_per_location,
                detection_alpha=args.detection_alpha,
                max_epochs_ref=args.max_epochs_ref,
                max_epochs_map=args.max_epochs_map,
                cell_count_cutoff=args.cell_count_cutoff,
                cell_percentage_cutoff2=args.cell_percentage_cutoff2,
                nonz_mean_cutoff=args.nonz_mean_cutoff,
                n_neighbors=args.n_neighbors,
                leiden_resolution=args.leiden_resolution,
                round_counts=args.round_counts,
                drop_unlabeled=args.drop_unlabeled,
                batch_size_map=args.batch_size_map,
            )
        except Exception as e:
            _log(f"ERROR in CLI mode: {e}")
            summary = WorkerOutput.error("cell2location", str(e), task="deconvolution")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    # With no CLI arguments, serve the tool over MCP stdio (local testing inside cell2loc_env).
    if len(sys.argv) == 1:
        _log("Starting cell2location MCP server on stdio...")
        mcp.run(transport="stdio")
    else:
        _cli_main()
