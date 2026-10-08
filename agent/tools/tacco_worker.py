#!/usr/bin/env python
"""
TACCO worker: transfer single-cell annotations to spatial transcriptomics.

- Runs entirely inside the TACCO_env environment.
- All logs/progress go to stderr.
- stdout is reserved for a single final JSON line.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import tacco as tc
from worker_utils import (
    WorkerOutput,
    build_deconv_analysis,
    cell_type_rename_note,
    cell_type_rename_params,
    choose_counts_matrix,
    describe_reduction,
    id_mismatch_msg,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    sanitize_cell_type_names,
    unsupported_choice_msg,
)

# Under a private name: run_tacco_annotation takes a ``drop_unlabeled`` parameter, which would shadow it.
from worker_utils import drop_unlabeled as _split_unlabeled

# tc.tl.annotate(method=<name>) dispatches to one of these functions (tacco/tools/_annotate.py,
# annotation_method). Every keyword annotate() does not declare itself is forwarded to that
# function, and a function that does not take it raises TypeError -- which is how `lamb`, an
# argument of the OT method alone, used to crash every other method.
METHOD_FUNCTIONS = {
    "OT": ("_OT", "_annotate_OT"),
    "nnls": ("_nnls", "_annotate_nnls"),
    "projection": ("_projection", "_annotate_projection"),
    "svm": ("_svm", "_annotate_svm"),
    "NMFreg": ("_NMFreg", "annotate_NMFreg"),
    "WOT": ("_wot", "_annotate_wot"),
    "novosparc": ("_novosparc", "_annotate_novosparc"),
    "tangram": ("_tangram", "_annotate_tangram"),
    "RCTD": ("_RCTD", "annotate_RCTD"),
    "SingleR": ("_SingleR", "annotate_SingleR"),
}

# TACCO runs these in a separate conda environment that it locates through its own `conda_env`
# argument, whose default is a path on the TACCO authors' cluster. This wrapper does not expose
# `conda_env`, so these methods cannot run through it; refusing up front beats upstream's error,
# which tells the caller to pass an argument they have no way to pass.
EXTERNAL_ENV_METHODS = ("tangram", "RCTD", "SingleR")

# These import an optional package inside TACCO; without it the method cannot run.
OPTIONAL_PACKAGE_METHODS = {"novosparc": "novosparc", "WOT": "wot"}


def _package_missing(package):
    try:
        return importlib.util.find_spec(package) is None
    except (ImportError, ValueError):
        return True


def runnable_methods():
    """The method names this wrapper can hand to tc.tl.annotate in the current environment."""
    return [
        m
        for m in METHOD_FUNCTIONS
        if m not in EXTERNAL_ENV_METHODS
        and not (m in OPTIONAL_PACKAGE_METHODS and _package_missing(OPTIONAL_PACKAGE_METHODS[m]))
    ]


def check_method(method):
    """Refuse a method name TACCO does not have, or one that cannot run through this wrapper."""
    runnable = runnable_methods()
    listed = ", ".join(repr(m) for m in runnable)
    if method not in METHOD_FUNCTIONS:
        same_letters = [m for m in METHOD_FUNCTIONS if m.lower() == str(method).lower()]
        hint = f" Did you mean {same_letters[0]!r}?" if same_letters else ""
        raise ValueError(
            unsupported_choice_msg("method", method, runnable, "TACCO method names are case-sensitive." + hint)
        )
    if method in EXTERNAL_ENV_METHODS:
        raise ValueError(
            f"method={method!r} is run by TACCO in a separate conda environment that it locates through its "
            "conda_env argument (upstream default: a path on the TACCO authors' cluster). This tool does not "
            f"expose conda_env, so {method!r} cannot run here. Methods that run in this environment: {listed}."
        )
    if method in OPTIONAL_PACKAGE_METHODS and _package_missing(OPTIONAL_PACKAGE_METHODS[method]):
        raise ImportError(
            f"method={method!r} needs the {OPTIONAL_PACKAGE_METHODS[method]!r} package, which is not installed in "
            f"this environment. Methods that run in this environment: {listed}."
        )
    return method


def method_keywords(method):
    """The keyword names the function behind ``method`` declares by name (not via ``**kwargs``).

    A ``**kwargs`` catch-all does not count: svm forwards it to sklearn's LinearSVC, which raises on
    an unknown name just as a function without ``**kwargs`` does.
    """
    module_name, func_name = METHOD_FUNCTIONS[method]
    fn = getattr(importlib.import_module("tacco.tools." + module_name), func_name)
    named = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return {name for name, p in inspect.signature(fn).parameters.items() if p.kind in named}


def effective_bisections(method, bisections):
    """What tc.tl.annotate runs with (tacco/tools/_annotate.py: `if bisections is None`)."""
    if bisections is not None:
        return int(bisections)
    return 4 if method == "OT" else 0


def effective_platform_iterations(method, platform_iterations):
    """What tc.tl.annotate runs with (tacco/tools/_annotate.py: `if platform_iterations is None`)."""
    if platform_iterations is not None:
        return int(platform_iterations)
    return 0 if method in ("OT", "projection") else -1


def prepare_reference_labels(sc_adata, annotation_key, drop_unlabeled):
    """Keep the reference cells that carry a label, and the label categories some cell carries.

    A missing label is not a class. TACCO crashed on a categorical NaN ("type_prior contains na!")
    and scored the string "nan" as a cell type. A category no cell carries -- common once a
    reference has been subset -- crashed TACCO's multi-center k-means with "n_clusters ... Got 0".
    Returns ``(sc_adata, n_dropped_unlabeled, unused_categories)``.
    """
    labels = sc_adata.obs[annotation_key]
    keep, n_dropped = _split_unlabeled(
        labels.to_numpy(), drop_unlabeled, f"reference cells (annotation_key='{annotation_key}')"
    )
    if n_dropped:
        sc_adata = sc_adata[keep].copy()
        print(
            f"[tacco-worker] Dropped {n_dropped} reference cells with no label in '{annotation_key}' "
            "(drop_unlabeled=True).",
            file=sys.stderr,
        )
    if sc_adata.n_obs == 0:
        raise ValueError(f"No reference cell carries a label in '{annotation_key}'; there is nothing to transfer.")
    unused = []
    column = sc_adata.obs[annotation_key]
    if isinstance(column.dtype, pd.CategoricalDtype):
        present = set(column.dropna().tolist())
        unused = [str(c) for c in column.cat.categories if c not in present]
        if unused:
            sc_adata.obs[annotation_key] = column.cat.remove_unused_categories()
            print(
                f"[tacco-worker] Removed {len(unused)} label categor(ies) no reference cell carries: {unused[:5]}",
                file=sys.stderr,
            )
    return sc_adata, int(n_dropped), unused


def counts_matrix_for(adata, use_raw_counts, what, lenient_without_raw=False):
    """``(adata, info)``: the matrix TACCO is handed for one input, by ``worker_utils.choose_counts_matrix``.

    TACCO treats X as counts: it checks the first stored values of the slide for integers and
    builds the reference profiles from counts. The shared rule applies (negative or non-finite X is
    refused, naming ``use_raw_counts`` when ``adata.raw`` holds counts; non-integer X runs with a
    warning; ``use_raw_counts`` runs on ``adata.raw``), with its messages prefixed by ``what`` so the
    reader knows which input they are about. ``lenient_without_raw`` (the reference) keeps X, with a
    warning, when ``use_raw_counts`` is set and that input has no ``adata.raw``; the slide is strict.
    """
    note = ""
    if use_raw_counts and lenient_without_raw and getattr(adata, "raw", None) is None:
        use_raw_counts = False
        note = f"use_raw_counts=True, but the {what} has no adata.raw, so its X was used."
    try:
        chosen, info = choose_counts_matrix(adata, use_raw_counts)
    except ValueError as exc:
        raise ValueError(f"{what}: {exc}") from exc
    info = dict(info)
    parts = [note] if note else []
    if info.get("warning"):
        parts.append(f"{what}: {info['warning']}")
    info["warning"] = " ".join(parts) or None
    return chosen, info


def _is_tacco_counts_refusal(exc) -> bool:
    """TACCO's own ``check_counts_validity`` refusal (tacco/preprocessing/_qc.py)."""
    text = str(exc)
    return "counts" in text and ("dont look like integers" in text or "counts are negative" in text)


def dominant_labels(comp_df):
    """Per-spot argmax over the rows TACCO annotated; an unannotated (all-NaN) row stays empty."""
    unannotated = comp_df.isna().all(axis=1)
    dominant = pd.Series(np.nan, index=comp_df.index, dtype=object)
    if (~unannotated).any():
        dominant.loc[~unannotated] = comp_df.loc[~unannotated].idxmax(axis=1).astype(object)
    return dominant, unannotated


def _write_atomic(path, write):
    """Write through ``<name>.partial`` and rename, so a killed run never leaves a half file."""
    tmp = path.with_name(path.name + ".partial")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _sanitize_uns_for_h5ad(adata):
    """
    anndata cannot write pandas.Series directly into .uns.
    Convert any Series in .uns (and shallow nested dicts) into DataFrame
    so that write_h5ad works.
    """
    print("[tacco-worker] Sanitizing adata.uns for HDF5 (convert Series -> DataFrame)...", file=sys.stderr)

    # top-level keys
    for k, v in list(adata.uns.items()):
        if isinstance(v, pd.Series):
            print(f"[tacco-worker] Converting uns['{k}'] (Series) -> DataFrame", file=sys.stderr)
            adata.uns[k] = v.to_frame(name="value")
        elif isinstance(v, dict):
            # shallow sanitize nested dict
            for subk, subv in list(v.items()):
                if isinstance(subv, pd.Series):
                    print(
                        f"[tacco-worker] Converting uns['{k}']['{subk}'] (Series) -> DataFrame",
                        file=sys.stderr,
                    )
                    v[subk] = subv.to_frame(name="value")
            adata.uns[k] = v


def run_tacco_annotation(
    sc_h5ad,
    spatial_h5ad,
    output_dir,
    annotation_key,
    result_key,
    method="OT",
    multi_center=3,
    lamb=1e-3,
    bisections=None,
    bisection_divisor=3,
    platform_iterations=None,
    drop_unlabeled=False,
    use_raw_counts=False,
):
    print("[tacco-worker] Starting TACCO annotation", file=sys.stderr)
    print(f"[tacco-worker] sc_h5ad      = {sc_h5ad}", file=sys.stderr)
    print(f"[tacco-worker] spatial_h5ad = {spatial_h5ad}", file=sys.stderr)
    print(f"[tacco-worker] output_dir   = {output_dir}", file=sys.stderr)
    print(f"[tacco-worker] annotation_key = {annotation_key}", file=sys.stderr)
    print(f"[tacco-worker] result_key     = {result_key or '(auto from annotation_key)'}", file=sys.stderr)
    print(f"[tacco-worker] method={method}, multi_center={multi_center}, lamb={lamb}", file=sys.stderr)

    # Before any file is read: a method that cannot run here should not cost a load.
    check_method(method)

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    spatial_annotated_h5ad = out_dir / "tacco_spatial_with_annotations.h5ad"
    composition_csv = out_dir / "tacco_composition.csv"
    dominant_label_csv = out_dir / "tacco_dominant_label_per_spot.csv"

    # Ensure result_key
    if not result_key:
        result_key = f"tacco_{annotation_key}"

    bisections_used = effective_bisections(method, bisections)
    platform_iterations_used = effective_platform_iterations(method, platform_iterations)

    with contextlib.redirect_stdout(sys.stderr):
        # Load data
        sc_adata = sc.read_h5ad(sc_h5ad)
        sp_adata = sc.read_h5ad(spatial_h5ad)

        print(
            f"[tacco-worker] Loaded scRNA: n_cells={sc_adata.n_obs}, n_genes={sc_adata.n_vars}",
            file=sys.stderr,
        )
        print(
            f"[tacco-worker] Loaded spatial: n_spots={sp_adata.n_obs}, n_genes={sp_adata.n_vars}",
            file=sys.stderr,
        )
        n_cells_sc_input = int(sc_adata.n_obs)

        # Background spots (obs['in_tissue'] == 0 in a CELLxGENE export) carry ambient counts, not
        # cells: TACCO would give each a composition and a dominant label. Left out, and counted.
        sp_adata, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(sp_adata, "spots")
        if n_spots_off_tissue:
            print(
                f"[tacco-worker] Left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0",
                file=sys.stderr,
            )
        n_spots_used = int(sp_adata.n_obs)

        # TACCO treats both matrices as counts; the shared rule decides which matrix each input hands it.
        sp_adata, sp_counts_info = counts_matrix_for(sp_adata, use_raw_counts, "spatial slide")
        sc_adata, sc_counts_info = counts_matrix_for(
            sc_adata, use_raw_counts, "single-cell reference", lenient_without_raw=True
        )

        # Gene names only: spot and cell identifiers are left exactly as supplied, as before.
        renamed_sc = make_names_unique_and_report(sc_adata, axes=("var",))
        renamed_sp = make_names_unique_and_report(sp_adata, axes=("var",))

        overlap_genes = len(set(sc_adata.var_names) & set(sp_adata.var_names))
        print(f"[tacco-worker] Overlap genes between sc and spatial: {overlap_genes}", file=sys.stderr)

        # Check annotation_key exists
        if annotation_key not in sc_adata.obs:
            raise ValueError(
                f"annotation_key='{annotation_key}' not found in scRNA obs. Available keys: {list(sc_adata.obs.keys())}"
            )
        if overlap_genes == 0:
            # TACCO itself only says "`multi_center_annotation_method` needs per observation data".
            raise ValueError(
                id_mismatch_msg("genes", "sc_h5ad", sc_adata.var_names, "spatial_h5ad", sp_adata.var_names)
            )

        sc_adata, n_dropped_unlabeled, unused_categories = prepare_reference_labels(
            sc_adata, annotation_key, drop_unlabeled
        )

        # Prepare multi_center param
        if multi_center is not None and multi_center <= 0:
            mc = None
        else:
            mc = multi_center

        # lamb is an argument of the OT method only; handed to any other method it raised TypeError.
        lamb_applied = "lamb" in method_keywords(method)
        method_kwargs = {"lamb": lamb} if lamb_applied else {}

        print("[tacco-worker] Running tc.tl.annotate ...", file=sys.stderr)
        try:
            tc.tl.annotate(
                sp_adata,
                sc_adata,
                annotation_key=annotation_key,
                result_key=result_key,
                method=method,
                bisections=bisections,
                bisection_divisor=bisection_divisor,
                platform_iterations=platform_iterations,
                multi_center=mc,
                **method_kwargs,
            )
        except ValueError as exc:
            if not _is_tacco_counts_refusal(exc):
                raise
            # Upstream's message ends by offering assume_valid_counts=True, which this tool does not
            # expose; say what the matrix holds and which knob does exist.
            first = str(exc).strip().splitlines()[0]
            source = sp_counts_info["expression_source"]
            held = (
                f"X, which holds {str(sp_counts_info['x_matrix_kind']).replace('_', ' ')} values"
                if source == "X"
                else source
            )
            raise ValueError(
                f"TACCO refused the spatial matrix it was handed ({held}): {first} TACCO needs raw counts; pass "
                "use_raw_counts=True when adata.raw holds them, or supply an h5ad whose X holds raw counts."
            ) from exc

        if result_key not in sp_adata.obsm:
            raise RuntimeError(f"TACCO annotate finished but .obsm['{result_key}'] not found on spatial AnnData.")

        comp = sp_adata.obsm[result_key]
        if isinstance(comp, pd.DataFrame):
            comp_df = comp.copy()
        else:
            categories = list(pd.unique(sc_adata.obs[annotation_key]))
            comp_df = pd.DataFrame(comp, index=sp_adata.obs_names, columns=categories)

        # Sanitize column names to avoid HDF5 path issues (e.g., "Treg/Tfr")
        orig_cols = list(comp_df.columns)
        # replace_space=False: tacco has only ever folded "/", and the composition CSVs it has
        # already published carry the spaces. Folding them now would rename published columns.
        safe_cols, ct_renames = sanitize_cell_type_names(orig_cols, replace_space=False)

        if ct_renames:
            print(
                "[tacco-worker] Detected '/' in annotation names; replacing with '_' for HDF5 safety.",
                file=sys.stderr,
            )
            comp_df.columns = safe_cols
            sp_adata.uns[result_key + "_colmap"] = {
                "original": orig_cols,
                "sanitized": safe_cols,
            }

        # TACCO drops spots with no counts on the genes it keeps (remove_zero_cells) and reindexes its
        # result to every spot, so those rows come back all-NaN. They keep their (empty) row in both
        # CSVs; they are not counted as annotated and they get no dominant label.
        dominant, unannotated = dominant_labels(comp_df)
        n_spots_total = int(sp_adata.n_obs)
        n_spots_unannotated = int(unannotated.sum())
        n_spots_annotated = n_spots_total - n_spots_unannotated
        if n_spots_annotated == 0:
            raise ValueError(
                f"TACCO returned no composition for any of the {n_spots_total} spots: every spot has zero counts "
                f"on the {overlap_genes} genes shared with the reference that TACCO kept."
            )

        # Save composition matrix into obsm and CSV
        sp_adata.obsm[result_key] = comp_df
        _write_atomic(composition_csv, comp_df.to_csv)
        print(f"[tacco-worker] Saved composition CSV to {composition_csv}", file=sys.stderr)

        # Dominant label per spot
        sp_adata.obs[result_key + "_max"] = pd.Categorical(dominant)
        _write_atomic(dominant_label_csv, lambda p: dominant.to_csv(p, header=["dominant_label"]))
        print(f"[tacco-worker] Saved dominant labels to {dominant_label_csv}", file=sys.stderr)

        # anndata cannot write a pandas.Series held in .uns; convert them to DataFrames first.
        _sanitize_uns_for_h5ad(sp_adata)

        # Save annotated spatial h5ad
        _write_atomic(spatial_annotated_h5ad, sp_adata.write_h5ad)
        print(f"[tacco-worker] Saved annotated spatial h5ad to {spatial_annotated_h5ad}", file=sys.stderr)

    # Build JSON-friendly result summary using WorkerOutput
    celltypes = list(comp_df.columns)
    n_celltypes = len(celltypes)
    dominant_counts = {str(k): int(v) for k, v in dominant.value_counts().items()}

    out = WorkerOutput("tacco", task="cell_type_annotation")
    out.set_data(
        n_spots=n_spots_supplied,
        n_spots_used=n_spots_used,
        n_spots_annotated=n_spots_annotated,
        n_spots_unannotated=n_spots_unannotated,
        n_genes=int(sp_adata.n_vars),
        n_cells_sc=n_cells_sc_input,
        n_cells_sc_used=int(sc_adata.n_obs),
        n_genes_sc=int(sc_adata.n_vars),
        n_overlap_genes=int(overlap_genes),
    )
    out.add_output_files(
        {
            "spatial_with_annotations_h5ad": str(spatial_annotated_h5ad),
            "composition_csv": str(composition_csv),
            "dominant_label_csv": str(dominant_label_csv),
        }
    )
    out.add_params(
        {
            "annotation_key": annotation_key,
            "result_key": result_key,
            "multi_center": mc,
            "lamb": lamb,
            # The values TACCO ran with; the *_requested keys keep what the caller sent (None = unset).
            "bisections": bisections_used,
            "bisections_requested": bisections,
            "bisection_divisor": bisection_divisor,
            "platform_iterations": platform_iterations_used,
            "platform_iterations_requested": platform_iterations,
            "drop_unlabeled": bool(drop_unlabeled),
            "n_reference_cells_dropped_unlabeled": n_dropped_unlabeled,
            "unused_label_categories_removed": unused_categories,
            "use_raw_counts": bool(use_raw_counts),
            # Which matrix of the reference TACCO read (the slide's is params.expression_source).
            "expression_source_sc": sc_counts_info["expression_source"],
            "x_matrix_kind_sc": sc_counts_info["x_matrix_kind"],
        }
    )
    record_method(out, method)
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    record_expression_source(out, sp_counts_info)
    if sc_counts_info.get("warning"):
        out.add_warning(sc_counts_info["warning"])
    if not lamb_applied:
        record_ignored(
            out, "lamb", f"lamb is an argument of TACCO's 'OT' method only; method={method!r} does not take it."
        )
    if bisections_used <= 0:
        record_ignored(
            out,
            "bisection_divisor",
            f"bisections resolved to {bisections_used} for method={method!r}, so no bisection round ran and the "
            "divisor was never read.",
        )
    out.add_params(identifier_rename_params(renamed_sp))
    out.add_params(identifier_rename_params(renamed_sc, suffix="sc"))
    out.add_params(cell_type_rename_params(ct_renames))

    unannotated_note = ""
    if n_spots_unannotated:
        unannotated_note = (
            f"{n_spots_unannotated} of {n_spots_total} spots got no composition: TACCO leaves out spots with zero "
            "counts on the genes it keeps (shared with the reference, not constant), so their rows in "
            "tacco_composition.csv are empty and they have no dominant label."
        )
        out.add_warning(unannotated_note)
    if n_dropped_unlabeled:
        out.add_warning(
            f"{n_dropped_unlabeled} of {n_cells_sc_input} reference cells had no label in '{annotation_key}' and were "
            "left out (drop_unlabeled=True)."
        )
    if unused_categories:
        out.add_warning(
            f"{len(unused_categories)} label categor(ies) of '{annotation_key}' carried by no reference cell were "
            f"removed before annotating: {unused_categories[:10]}."
        )

    out.set_summary(
        n_celltypes=n_celltypes,
        celltypes=celltypes,
        dominant_counts=dominant_counts,
        n_spots_annotated=n_spots_annotated,
        n_spots_unannotated=n_spots_unannotated,
    )
    lamb_text = f", lamb={lamb}" if lamb_applied else ""
    out.set_analysis(
        build_deconv_analysis(
            n_celltypes,
            dominant_counts,
            total_spots=n_spots_annotated,
            method_name="TACCO",
        )
        + f" Method {method!r} ran with bisections={bisections_used}, "
        + f"platform_iterations={platform_iterations_used}, multi_center={mc}{lamb_text}."
        + describe_reduction(
            "spots",
            n_spots_supplied,
            n_spots_used,
            "leaving out the spots with obs['in_tissue'] == 0 (background outside the tissue)",
        )
        + (" " + unannotated_note if unannotated_note else "")
        + (
            f" {n_dropped_unlabeled} of {n_cells_sc_input} reference cells had no label and were left out."
            if n_dropped_unlabeled
            else ""
        )
        + cell_type_rename_note(ct_renames)
        + identifier_rename_note(renamed_sp, "spatial data")
        + identifier_rename_note(renamed_sc, "reference")
    )
    return out.to_dict()


def main():
    parser = argparse.ArgumentParser(description="TACCO worker: transfer scRNA annotations to spatial transcriptomics.")
    parser.add_argument("--sc-h5ad", required=True, help="Path to single-cell reference AnnData (.h5ad)")
    parser.add_argument("--spatial-h5ad", required=True, help="Path to spatial AnnData (.h5ad)")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    parser.add_argument(
        "--annotation-key",
        required=True,
        help="obs column in scRNA AnnData containing categorical cell-type labels (e.g. 'CellType')",
    )
    parser.add_argument(
        "--result-key",
        default="",
        help=(
            "obsm key in spatial AnnData to store TACCO compositions (default: 'tacco_<annotation_key>'); "
            "dominant labels go to obs['<result_key>_max']"
        ),
    )
    parser.add_argument(
        "--method",
        default="OT",
        help=(
            "TACCO annotation method (default: 'OT'). Also runnable here: 'nnls', 'projection', 'svm', 'NMFreg'. "
            "'tangram', 'RCTD' and 'SingleR' need an external conda env this tool does not expose and are refused; "
            "'novosparc' and 'WOT' need their package installed."
        ),
    )
    parser.add_argument(
        "--multi-center",
        type=int,
        default=3,
        help="Multi-center parameter for TACCO (recommended 3; <=0 disables multi-center).",
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default=1e-3,
        help="Regularization strength lambda of TACCO's OT method; ignored (and reported so) by every other method.",
    )
    parser.add_argument(
        "--bisections",
        type=int,
        default=None,
        help="Number of bisections for boosted annotation (None = TACCO default).",
    )
    parser.add_argument(
        "--bisection-divisor",
        type=int,
        default=3,
        help="Bisection divisor for boosted annotation (default: 3).",
    )
    parser.add_argument(
        "--platform-iterations",
        type=int,
        default=None,
        # TACCO runs normalize_platform() whenever this is > -1 (tacco/tools/_annotate.py), so 0 is
        # the correcting path, not the off switch -- only a negative value skips the correction.
        # Omitting the flag passes None through, which is how TACCO is asked to choose per method.
        help=(
            "Platform normalization iterations. 0 normalizes once and does not iterate; a "
            "positive value repeats the correction using the previous round's annotation; a "
            "negative value skips platform normalization entirely. Omit to take TACCO's own "
            "method-dependent default, which is 0 for the OT and projection methods and -1 "
            "for every other method."
        ),
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help=(
            "Leave out reference cells whose annotation_key label is missing (NaN/empty/'nan'). Without it a "
            "missing label stops the run; a missing label is not a class."
        ),
    )
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        help=(
            "Run on adata.raw of the spatial h5ad (refused when it has none, or it does not hold counts), and of "
            "the reference when it has one (else its X, with a warning). Without it X is used."
        ),
    )

    args = parser.parse_args()

    try:
        result = run_tacco_annotation(
            sc_h5ad=args.sc_h5ad,
            spatial_h5ad=args.spatial_h5ad,
            output_dir=args.output_dir,
            annotation_key=args.annotation_key,
            result_key=args.result_key,
            method=args.method,
            multi_center=args.multi_center,
            lamb=args.lamb,
            bisections=args.bisections,
            bisection_divisor=args.bisection_divisor,
            platform_iterations=args.platform_iterations,
            drop_unlabeled=args.drop_unlabeled,
            use_raw_counts=args.use_raw_counts,
        )
        # Single JSON line to stdout
        print(json.dumps(result, default=str))
        sys.stdout.flush()
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("tacco", str(e), task="cell_type_annotation")
        sys.exit(1)


if __name__ == "__main__":
    main()
