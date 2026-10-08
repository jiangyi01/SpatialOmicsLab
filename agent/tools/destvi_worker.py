#!/usr/bin/env python
"""
DestVI worker for SpatialOmicsLab MCP.

Runs the DestVI (scvi-tools) deconvolution pipeline:
  1. Load scRNA-seq reference and spatial AnnData; leave out spots with ``obs['in_tissue'] == 0``
     (background glass, reported in ``params.in_tissue_filter``). Both models fit their matrix as
     counts, so each input's matrix is checked (``worker_utils.choose_counts_matrix``): negative or
     non-finite values are refused, a fractional matrix runs with a warning, and ``use_raw_counts``
     reads ``adata.raw`` of each input that carries one.
  2. Select highly variable genes shared between datasets (scanpy flavour ``hvg_flavor``; a flavour
     that cannot run stops the run -- no other flavour is substituted).
  3. Train CondSCVI on the scRNA-seq reference. It is always trained without a batch covariate:
     DestVI.from_rna_model cannot load a batch-aware decoder, so ``batch_key`` is accepted for
     compatibility only and reported under ``params.ignored`` when it could have mattered.
  4. Train DestVI on the spatial data using the trained CondSCVI model.
  5. Extract cell-type proportions via model.get_proportions().
  6. Save proportions CSV and annotated spatial h5ad.

Environment: /opt/conda/envs/destvi_env
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

# Ensure worker_utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import scanpy as sc
from worker_utils import (
    WorkerOutput,
    build_deconv_analysis,
    cell_type_rename_note,
    cell_type_rename_params,
    choose_counts_matrix,
    default_output_dir,
    gene_id_harmonization_note,
    gene_id_harmonization_params,
    harmonize_gene_ids,
    keep_in_tissue,
    preflight_check,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    require_hvg_flavor,
    sanitize_cell_type_names,
    unsupported_choice_msg,
)

# Under a private name: _run_destvi takes a ``drop_unlabeled`` parameter, which would shadow it.
from worker_utils import drop_unlabeled as _split_unlabeled

METHOD_NAME = "DestVI (scvi-tools CondSCVI on the reference, then DestVI on the slide)"

# scanpy flavours this worker can rank genes with. seurat_v3 ranks raw counts and needs
# scikit-misc; seurat and cell_ranger rank log-normalised data. The default is "seurat" because
# that is what every run in destvi_env actually did: scikit-misc is not installed there, and the
# worker used to try seurat_v3, catch the failure and run seurat without saying so.
HVG_FLAVORS = ("seurat", "seurat_v3", "cell_ranger")
DEFAULT_HVG_FLAVOR = "seurat"

# The portal's default batch_key. A column of that name in the reference is reported as ignored;
# so is any other value the caller chose, since none of them reaches the model.
DEFAULT_BATCH_KEY = "batch"
BATCH_KEY_IGNORED_WHY = (
    "CondSCVI is always trained without a batch covariate, because DestVI.from_rna_model cannot load "
    "a batch-aware decoder, so no batch correction was applied"
)


def _log(msg: str) -> None:
    """Log to stderr so stdout stays clean for JSON output."""
    print(f"[destvi] {msg}", file=sys.stderr, flush=True)


def _atomic_to_csv(df: Any, path: Path) -> None:
    """Write ``df`` as ``<path>.partial`` and rename over ``path`` once it is complete."""
    tmp = str(path) + ".partial"
    df.to_csv(tmp)
    os.replace(tmp, str(path))


def _atomic_write_h5ad(adata: Any, path: Path) -> None:
    """Write the AnnData as ``<path>.partial`` and rename over ``path`` once it is complete."""
    tmp = str(path) + ".partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, str(path))


def _select_hvg(adata_sc: Any, hvg_flavor: str, n_top_genes: int) -> Any:
    """The reference's raw counts restricted to its ``n_top_genes`` most variable genes.

    ``seurat_v3`` ranks the raw counts themselves (and needs scikit-misc); ``seurat`` and
    ``cell_ranger`` rank log-normalised data, so they rank a normalised copy and the mask is applied
    to the raw counts CondSCVI trains on. A flavour that cannot run stops the run with the reason --
    no other flavour is substituted.
    """
    if hvg_flavor not in HVG_FLAVORS:
        raise ValueError(unsupported_choice_msg("hvg_flavor", hvg_flavor, HVG_FLAVORS))
    require_hvg_flavor(hvg_flavor)  # a missing scikit-misc stops here, naming the package
    if hvg_flavor == "seurat_v3":
        ranked = adata_sc
    else:
        ranked = adata_sc.copy()
        sc.pp.normalize_total(ranked, target_sum=1e4)
        sc.pp.log1p(ranked)
    try:
        sc.pp.highly_variable_genes(
            ranked, n_top_genes=min(n_top_genes, adata_sc.n_vars), subset=False, flavor=hvg_flavor
        )
    except Exception as e:
        # scikit-misc can be installed yet ABI-broken against this env's numpy ("numpy.dtype size
        # changed"), which raises ValueError here rather than ImportError above.
        raise RuntimeError(
            f"highly_variable_genes(flavor={hvg_flavor!r}) failed and no other flavour was substituted "
            f"({type(e).__name__}: {e}). "
            + (
                "seurat_v3 needs a working scikit-misc (import name skmisc) in destvi_env; repair it, or pass "
                "hvg_flavor='seurat'."
                if hvg_flavor == "seurat_v3"
                else f"Valid hvg_flavor values: {', '.join(repr(f) for f in HVG_FLAVORS)}."
            )
        ) from e
    mask = np.asarray(ranked.var["highly_variable"], dtype=bool)
    return adata_sc[:, mask].copy()


def _counts_matrix(adata: Any, what: str, use_raw_counts: bool):
    """``(adata, info)``: the matrix one input's model fits as counts, and what was decided about it.

    CondSCVI and DestVI both fit their input's X as counts, and ran on whatever it held: a scaled X
    (negative values) or a log-normalised one went into a count likelihood without a word. This is
    ``worker_utils.choose_counts_matrix`` for each of the two inputs, with the input named in its
    refusal and warning. ``use_raw_counts`` reads ``adata.raw`` of each input that carries one, as
    CellDART's does; an input without one keeps X, and ``info['note']`` says so.
    """
    has_raw = getattr(adata, "raw", None) is not None
    try:
        chosen, info = choose_counts_matrix(adata, bool(use_raw_counts) and has_raw)
    except ValueError as e:
        raise ValueError(f"The {what}: {e}") from None
    info = dict(info)
    if info.get("warning"):
        info["warning"] = f"The {what}: {info['warning']}"
    info["has_raw"] = has_raw
    info["note"] = (
        f"use_raw_counts=True, but the {what} has no adata.raw, so its X was used."
        if use_raw_counts and not has_raw
        else None
    )
    return chosen, info


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """Redirect stdout to stderr to capture training progress bars."""
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


def _run_destvi(
    sc_h5ad_path: str,
    spatial_h5ad_path: str,
    output_dir: str = default_output_dir(),
    cell_type_key: str = "cell_type",
    batch_key: str = "batch",
    max_epochs_sc: int = 100,
    max_epochs_st: int = 2000,
    n_top_genes: int = 2000,
    hvg_flavor: str = DEFAULT_HVG_FLAVOR,
    drop_unlabeled: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Core DestVI pipeline. Returns a WorkerOutput dict.

    ``use_raw_counts`` reads each input's counts from its ``adata.raw`` when it has one (see
    :func:`_counts_matrix`); spots with ``obs['in_tissue'] == 0`` are left out and reported.
    """

    import os

    # Set scvi/lightning default_root_dir to output_dir to prevent
    # checkpoint conflicts from stale lightning_logs in working directory
    os.environ["SCVI_DATA_DIR"] = output_dir

    # Checked before anything is loaded or trained, so a flavour that cannot run costs nothing.
    if hvg_flavor not in HVG_FLAVORS:
        raise ValueError(unsupported_choice_msg("hvg_flavor", hvg_flavor, HVG_FLAVORS))
    require_hvg_flavor(hvg_flavor)

    import scvi
    import scvi.model._destvi as _destvi_mod
    import torch as _torch
    from scvi.model import CondSCVI, DestVI
    from scvi.model._condscvi import CondSCVI as _CondSCVIcls

    _orig_from_rna = _destvi_mod.DestVI.from_rna_model
    _orig_get_vamp_prior = _CondSCVIcls.get_vamp_prior

    def _vamp_prior_as_tensors(self, *a, **kw):
        res = _orig_get_vamp_prior(self, *a, **kw)
        try:
            conv = {}
            for k, v in res.items():
                if isinstance(v, _torch.Tensor):
                    conv[k] = v.detach().float()
                else:
                    conv[k] = _torch.from_numpy(np.asarray(v)).float()
            return conv
        except Exception:
            return res

    _CondSCVIcls.get_vamp_prior = _vamp_prior_as_tensors

    @classmethod
    def _safe_from_rna_model(cls, st_adata, sc_model, *a, **kw):
        try:
            ip = sc_model.init_params_
            kw_dict = ip.setdefault("kwargs", {}).setdefault("module_kwargs", {})
            kw_dict.setdefault("prior", getattr(sc_model.module, "prior", "normal"))
        except Exception:
            pass
        return _orig_from_rna.__func__(cls, st_adata, sc_model, *a, **kw)

    _destvi_mod.DestVI.from_rna_model = _safe_from_rna_model
    DestVI.from_rna_model = _safe_from_rna_model

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load data ----
    _log(f"Reading scRNA-seq reference: {sc_h5ad_path}")
    adata_sc = sc.read_h5ad(sc_h5ad_path)
    _log(f"Reading spatial data: {spatial_h5ad_path}")
    adata_st = sc.read_h5ad(spatial_h5ad_path)

    # Background spots (obs['in_tissue'] == 0; 56-70% of the library's CELLxGENE Visium exports) hold
    # ambient counts, not cells. DestVI trained its amortised model on them and published a proportions
    # row for each. Left out and reported, as the other spot-level tools do.
    adata_st, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata_st, "spots")
    if n_spots_off_tissue:
        _log(f"Left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0 (background)")

    # Which matrix each model fits as counts: X, or adata.raw with use_raw_counts (see _counts_matrix).
    adata_st, st_counts = _counts_matrix(adata_st, "spatial input (spatial_h5ad_path)", use_raw_counts)
    adata_sc, sc_counts = _counts_matrix(adata_sc, "scRNA reference (sc_h5ad_path)", use_raw_counts)
    for info in (st_counts, sc_counts):
        for message in (info["note"], info["warning"]):
            if message:
                _log(message)

    adata_sc.var_names_make_unique()
    adata_st.var_names_make_unique()

    _log(f"scRNA: {adata_sc.n_obs} cells x {adata_sc.n_vars} genes")
    _log(f"Spatial: {adata_st.n_obs} spots x {adata_st.n_vars} genes")

    # ---- 2. Validate cell_type_key ----
    if cell_type_key not in adata_sc.obs.columns:
        raise ValueError(
            f"cell_type_key='{cell_type_key}' not found in scRNA obs. Available keys: {list(adata_sc.obs.columns)}"
        )

    # A missing label is not a class. scvi-tools refuses NaN with a message that names two identical
    # category lists; a "nan" string would be trained as a cell type. Refuse (default) or drop.
    n_cells_sc_input = int(adata_sc.n_obs)
    labels = adata_sc.obs[cell_type_key]
    unused_categories: list = []
    if isinstance(labels.dtype, pd.CategoricalDtype):
        present = set(labels.dropna().unique().tolist())
        unused_categories = [str(c) for c in labels.cat.categories if c not in present]
    keep_mask, n_dropped_unlabeled = _split_unlabeled(
        labels.to_numpy(), drop_unlabeled, f"reference cells (cell_type_key='{cell_type_key}')"
    )
    if n_dropped_unlabeled:
        adata_sc = adata_sc[keep_mask].copy()
        _log(f"Dropped {n_dropped_unlabeled} reference cells with no label in '{cell_type_key}' (drop_unlabeled=True).")
    # scvi-tools registers every category of a categorical column as a class, so a category no cell
    # carries (a subset reference keeps its parent's categories) would be published as a proportions
    # column for a cell type with no reference cells. Drop those; the order of the rest is kept.
    if isinstance(adata_sc.obs[cell_type_key].dtype, pd.CategoricalDtype):
        adata_sc.obs[cell_type_key] = adata_sc.obs[cell_type_key].cat.remove_unused_categories()

    # ---- 3. Harmonize gene IDs ----
    _log("Harmonizing gene IDs between scRNA and spatial data...")
    gene_id_report: dict = {}
    n_shared_pre = harmonize_gene_ids(adata_sc, adata_st, report=gene_id_report)
    _log(f"Shared genes after harmonization: {n_shared_pre}")
    if n_shared_pre == 0:
        raise ValueError(
            "No shared genes between scRNA reference and spatial data even after "
            "attempting gene ID harmonization. Check that gene IDs (symbols vs "
            "Ensembl) are consistent between datasets."
        )

    # ---- 4. Select highly variable genes ----
    _log(f"Selecting top {n_top_genes} highly variable genes from scRNA reference (flavor={hvg_flavor!r})...")
    adata_sc = _select_hvg(adata_sc, hvg_flavor, n_top_genes)
    n_hvg = int(adata_sc.n_vars)
    _log(f"scRNA after HVG selection: {n_hvg} genes")

    # Intersect with spatial genes
    shared_genes = np.intersect1d(adata_sc.var_names, adata_st.var_names)
    if shared_genes.size == 0:
        raise ValueError(
            "No shared genes between scRNA reference and spatial data after HVG selection. "
            "Check that gene IDs (symbols vs Ensembl) are consistent between datasets."
        )
    _log(f"{shared_genes.size} genes shared between scRNA and spatial after HVG selection")

    adata_sc = adata_sc[:, shared_genes].copy()
    adata_st = adata_st[:, shared_genes].copy()

    # A spot with no counts in the genes the model uses cannot be fitted: DestVI's rate is the spot's
    # library size times its expression, so a zero library turns the likelihood NaN and the first epoch
    # stops on "Expected value argument ... within the support (Real())" -- one such spot among 4,992
    # killed the whole run. Those spots are left out of the fit; the outputs keep one row per spot, and
    # theirs is all zeros (not a composition), counted and named.
    st_umi = np.asarray(adata_st.X.sum(axis=1)).ravel()
    has_counts = st_umi > 0
    no_count_spots = [str(s) for s in adata_st.obs_names[~has_counts]]
    if len(no_count_spots) == adata_st.n_obs:
        raise ValueError(
            f"None of the {adata_st.n_obs} spots has a count in the {int(shared_genes.size)} genes shared by the "
            "reference's highly variable genes and the slide, so there is nothing to deconvolve. Check that X of the "
            "spatial h5ad holds raw counts, or raise n_top_genes to share more genes."
        )
    if no_count_spots:
        _log(f"{len(no_count_spots)} spots have no counts in the {int(shared_genes.size)} model genes")

    # ---- 5. Train CondSCVI on scRNA reference ----
    _log("Setting up CondSCVI on scRNA reference...")

    # NOTE: DestVI.from_rna_model does NOT support batch-corrected decoders.
    # When CondSCVI is trained with batch_key, the decoder includes batch dims
    # that cause a shape mismatch in DestVI. So we always train without batch_key.
    use_batch_key = None
    batch_key_in_reference = bool(batch_key) and batch_key in adata_sc.obs.columns
    if batch_key_in_reference:
        _log(
            f"batch_key='{batch_key}' found in scRNA obs, but DestVI does not "
            "support batch-corrected decoders. Training CondSCVI without batch correction."
        )

    scvi.model.CondSCVI.setup_anndata(
        adata_sc,
        labels_key=cell_type_key,
    )

    try:
        sc_model = CondSCVI(adata_sc, prior="normal")
    except TypeError:
        sc_model = CondSCVI(adata_sc)
    _log(f"Training CondSCVI (max_epochs={max_epochs_sc})...")
    sc_model.train(max_epochs=max_epochs_sc, train_size=1.0)
    try:
        mk = sc_model.init_params_.setdefault("kwargs", {}).setdefault("module_kwargs", {})
        if "prior" not in mk:
            mk["prior"] = "normal"
    except Exception:
        pass
    _log("CondSCVI training complete.")

    # ---- 6. Train DestVI on spatial data ----
    _log("Setting up DestVI on spatial data...")
    adata_fit = adata_st[has_counts].copy() if no_count_spots else adata_st
    DestVI.setup_anndata(adata_fit)

    st_model = DestVI.from_rna_model(adata_fit, sc_model)
    _log(f"Training DestVI (max_epochs={max_epochs_st})...")
    st_model.train(max_epochs=max_epochs_st, train_size=1.0)
    _log("DestVI training complete.")

    # ---- 6. Extract cell-type proportions ----
    _log("Extracting cell-type proportions...")
    proportions = st_model.get_proportions()

    if isinstance(proportions, pd.DataFrame):
        prop_df = proportions
    else:
        # If returns numpy array, build DataFrame. The columns are the classes scvi-tools registered,
        # in its order -- the obs column itself may not be categorical.
        cell_types = [str(c) for c in sc_model.adata_manager.get_state_registry("labels").categorical_mapping]
        prop_df = pd.DataFrame(proportions, index=adata_fit.obs_names, columns=cell_types)
    if no_count_spots:
        # One row per spot, as before; the spots left out of the fit get all-zero rows. By position,
        # since get_proportions keeps the order of the spots it was given.
        full = np.zeros((int(adata_st.n_obs), prop_df.shape[1]), dtype=float)
        full[has_counts] = prop_df.to_numpy(dtype=float)
        prop_df = pd.DataFrame(full, index=adata_st.obs_names, columns=prop_df.columns)

    # Sanitize column names for HDF5 safety
    safe_cols, ct_renames = sanitize_cell_type_names(prop_df.columns)
    if ct_renames:
        _log(f"Sanitized {len(ct_renames)} cell-type names for HDF5 safety")
    prop_df.columns = safe_cols

    # Save proportions to CSV
    prop_csv_path = output_path / "destvi_proportions.csv"
    _atomic_to_csv(prop_df, prop_csv_path)
    _log(f"Saved proportions to {prop_csv_path}")

    # Store in spatial AnnData
    adata_st.obsm["destvi_proportions"] = prop_df
    for ct in safe_cols:
        adata_st.obs[ct] = prop_df[ct].values

    # ---- 7. Save annotated spatial h5ad ----
    out_h5ad = output_path / "destvi_spatial.h5ad"
    _log(f"Saving annotated spatial AnnData to {out_h5ad}")
    _atomic_write_h5ad(adata_st, out_h5ad)

    # ---- 8. Build output ----
    cell_type_names = list(prop_df.columns)
    n_celltypes = len(cell_type_names)
    # The in-tissue spots DestVI deconvolved -- the rows of the outputs. n_spots_supplied is the slide.
    n_spots = int(adata_st.n_obs)

    # Compute dominant cell type per spot, over the spots with an estimate (an all-zero row's idxmax is
    # just its first column).
    try:
        dominant_ct = prop_df[has_counts].idxmax(axis=1)
        dominant_counts = dict(Counter(dominant_ct))
    except Exception:
        dominant_counts = None

    out = WorkerOutput("destvi", task="deconvolution")
    out.set_data(
        # Supplied, then trained on: they differ only when drop_unlabeled=True left cells out.
        n_cells_sc=n_cells_sc_input,
        n_cells_sc_used=int(adata_sc.n_obs),
        n_genes_sc=int(adata_sc.n_vars),
        n_spots=n_spots,
        n_spots_supplied=int(n_spots_supplied),
        n_genes=int(adata_st.n_vars),
        # Spots left out of the fit, whose rows are all zeros (see the warning), and the spots fitted.
        n_spots_no_counts_in_model_genes=len(no_count_spots),
        n_spots_fitted=int(adata_fit.n_obs),
    )
    out.add_output_files(
        {
            "proportions_csv": str(prop_csv_path),
            "spatial_h5ad": str(out_h5ad),
        }
    )
    out.add_params(
        {
            "cell_type_key": cell_type_key,
            "batch_key": batch_key,
            "batch_key_used": use_batch_key is not None,
            "max_epochs_sc": max_epochs_sc,
            "max_epochs_st": max_epochs_st,
            "n_top_genes": n_top_genes,
            # The flavour that ranked the genes, and how many it kept -- before this the payload
            # carried n_top_genes only, while every run in destvi_env silently used "seurat".
            "hvg_flavor": hvg_flavor,
            "n_hvg_selected": n_hvg,
            "drop_unlabeled": bool(drop_unlabeled),
            "n_reference_cells_dropped_unlabeled": int(n_dropped_unlabeled),
            "n_unused_label_categories_dropped": len(unused_categories),
            # Counted AFTER hvg selection, so this is what the model was trained on -- a smaller
            # number than the gene_ids_shared_after_harmonization added just below, which is the
            # intersection harmonization itself produced.
            "n_shared_genes": int(shared_genes.size),
            "use_raw_counts": bool(use_raw_counts),
            # The reference's matrix; the slide's is params.expression_source / x_matrix_kind below.
            "reference_expression_source": sc_counts["expression_source"],
            "reference_x_matrix_kind": sc_counts["x_matrix_kind"],
        }
    )
    out.add_params(gene_id_harmonization_params(gene_id_report))
    out.add_params(cell_type_rename_params(ct_renames))
    record_method(out, METHOD_NAME)
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    record_expression_source(out, st_counts)
    if sc_counts["warning"]:
        out.add_warning(sc_counts["warning"])
    if use_raw_counts and not (st_counts["has_raw"] or sc_counts["has_raw"]):
        record_ignored(out, "use_raw_counts", "neither input has an adata.raw, so both models fitted X as supplied")
    else:
        for info in (st_counts, sc_counts):
            if info["note"]:
                out.add_warning(info["note"])
    if batch_key_in_reference or (batch_key and batch_key != DEFAULT_BATCH_KEY):
        where = "is a column of the reference, but " if batch_key_in_reference else ""
        record_ignored(out, "batch_key", f"batch_key={batch_key!r} {where}{BATCH_KEY_IGNORED_WHY}.")
    if n_dropped_unlabeled:
        out.add_warning(
            f"{n_dropped_unlabeled} of {n_cells_sc_input} reference cells had no label in '{cell_type_key}' and were "
            "left out (drop_unlabeled=True)."
        )
    if no_count_spots:
        shown = ", ".join(no_count_spots[:5])
        more = f", ... ({len(no_count_spots)} in all)" if len(no_count_spots) > 5 else ""
        out.add_warning(
            f"{len(no_count_spots)} of {n_spots} spots have no counts in the {int(shared_genes.size)} genes the model "
            f"used ({shown}{more}). DestVI cannot fit a spot with no counts (its likelihood turns NaN and training "
            "stops), so they were left out of the fit; their rows in destvi_proportions.csv are all zeros, not a "
            "composition."
        )
    if unused_categories:
        shown = ", ".join(repr(c) for c in unused_categories[:10])
        more = f" and {len(unused_categories) - 10} more" if len(unused_categories) > 10 else ""
        out.add_warning(
            f"{len(unused_categories)} categories of '{cell_type_key}' had no cells in the reference ({shown}{more}) "
            "and were not fitted as cell types."
        )
    hvg_input = "raw counts" if hvg_flavor == "seurat_v3" else "log-normalised counts"
    fitted = (
        "X as supplied"
        if st_counts["expression_source"] == sc_counts["expression_source"] == "X"
        else f"the reference's {sc_counts['expression_source']} and the slide's {st_counts['expression_source']}"
    )
    hvg_note = (
        f" Genes: the {n_hvg} most variable reference genes by scanpy flavor '{hvg_flavor}' (ranked on "
        f"{hvg_input}), {int(shared_genes.size)} of them shared with the slide; both models were trained on "
        f"{fitted}, which they read as raw counts."
    )
    tissue_note = (
        f" {n_spots_off_tissue} of the {n_spots_supplied} spots have obs['in_tissue'] == 0 (background) and were "
        f"left out; the {n_spots} in-tissue spots were deconvolved."
        if n_spots_off_tissue
        else ""
    )
    if no_count_spots:
        tissue_note += (
            f" {len(no_count_spots)} of the {n_spots} spots have no counts in the {int(shared_genes.size)} model genes "
            "and were left out of the fit; their rows are all zeros, not a composition."
        )
    dropped_note = (
        f" {n_dropped_unlabeled} of {n_cells_sc_input} reference cells had no label and were left out."
        if n_dropped_unlabeled
        else ""
    )
    out.set_summary(
        n_cell_types=n_celltypes,
        cell_type_names=cell_type_names,
        dominant_counts=dominant_counts,
    )
    out.set_analysis(
        build_deconv_analysis(
            n_celltypes=n_celltypes,
            dominant_counts=dominant_counts,
            total_spots=n_spots,
            method_name="DestVI",
        )
        + hvg_note
        + tissue_note
        + dropped_note
        + gene_id_harmonization_note(gene_id_report)
        + cell_type_rename_note(ct_renames)
    )

    return out.to_dict()


def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="DestVI worker: scvi-tools CondSCVI + DestVI deconvolution pipeline")
    parser.add_argument("--sc-h5ad", required=True, help="Path to scRNA-seq reference h5ad")
    parser.add_argument("--spatial-h5ad", required=True, help="Path to spatial h5ad")
    parser.add_argument("--output-dir", default=default_output_dir(), help="Output directory")
    parser.add_argument("--cell-type-key", default="cell_type", help="obs column with cell-type labels")
    parser.add_argument(
        "--batch-key",
        default=DEFAULT_BATCH_KEY,
        help="Accepted for compatibility only: CondSCVI is always trained without a batch covariate",
    )
    parser.add_argument("--max-epochs-sc", type=int, default=100, help="Max epochs for CondSCVI")
    parser.add_argument("--max-epochs-st", type=int, default=2000, help="Max epochs for DestVI")
    parser.add_argument("--n-top-genes", type=int, default=2000, help="Number of HVGs to select")
    parser.add_argument(
        "--hvg-flavor",
        default=DEFAULT_HVG_FLAVOR,
        help=f"scanpy HVG flavour, one of {', '.join(HVG_FLAVORS)} (seurat_v3 needs scikit-misc)",
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="Leave out reference cells with no label instead of refusing the run",
    )
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        help="Read each input's counts from adata.raw when it has one, instead of X",
    )

    args = parser.parse_args()

    preflight_check(
        inputs={
            "sc_h5ad": args.sc_h5ad,
            "spatial_h5ad": args.spatial_h5ad,
        },
        output_dir=args.output_dir,
        packages=["scvi"],
    )

    error_info = None
    error_exc = None
    with _redirect_stdout_to_stderr():
        try:
            result = _run_destvi(
                sc_h5ad_path=args.sc_h5ad,
                spatial_h5ad_path=args.spatial_h5ad,
                output_dir=args.output_dir,
                cell_type_key=args.cell_type_key,
                batch_key=args.batch_key,
                max_epochs_sc=args.max_epochs_sc,
                max_epochs_st=args.max_epochs_st,
                n_top_genes=args.n_top_genes,
                hvg_flavor=args.hvg_flavor,
                drop_unlabeled=args.drop_unlabeled,
                use_raw_counts=args.use_raw_counts,
            )
        except Exception as e:
            _log(f"ERROR: {e}")
            traceback.print_exc(file=sys.stderr)
            error_info = str(e)
            error_exc = e

    # stdout: JSON only (must be outside redirect block)
    if error_info is not None:
        WorkerOutput.emit_error("destvi", error_info, task="deconvolution", exc=error_exc)
        sys.exit(1)

    print(json.dumps(result, default=str))
    sys.stdout.flush()


if __name__ == "__main__":
    _cli_main()
