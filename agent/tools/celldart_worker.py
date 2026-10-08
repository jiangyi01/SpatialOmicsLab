#!/usr/bin/env python
"""
CellDART worker for SpatialOmicsLab MCP.

Runs the CellDART domain-adaptation deconvolution pipeline:
  1. Load scRNA-seq reference and spatial AnnData; leave out spots with ``obs['in_tissue'] == 0``
     (background glass, reported); drop reference cells with no usable label.
  2. Read counts from X, or from ``adata.raw`` of each input that has one when ``use_raw_counts``
     is set, and check them: CellDART normalises its input as counts (``normalize_total`` then
     ``log1p``), so a negative or NaN value is refused (log1p of a value below -1 is NaN, and the
     spot's fractions came out NaN at status ok) and a fractional matrix is reported.
  3. Harmonise gene identifiers between the two objects (no genes are cut here).
  4. Run CellDART's pred_cellf_celldart, which ranks the reference with a Wilcoxon test, keeps the
     union of each cell type's top ``num_markers`` genes that are on the spatial panel, cuts BOTH
     objects to that marker panel, and trains and predicts on it.
  5. Save proportions CSV, the marker panel CSV and the annotated spatial h5ad, each atomically.

CellDART uses adversarial domain adaptation to transfer cell-type
information from single-cell to spatial resolution.

Environment: /opt/conda/envs/celldart_env
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
    default_output_dir,
    describe_reduction,
    gene_id_harmonization_note,
    gene_id_harmonization_params,
    harmonize_gene_ids,
    keep_in_tissue,
    preflight_check,
    record_in_tissue,
    record_method,
    sanitize_cell_type_names,
)


def _log(msg: str) -> None:
    """Log to stderr so stdout stays clean for JSON output."""
    print(f"[celldart] {msg}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """Redirect stdout to stderr to capture training progress bars."""
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


def _import_pred_cellf():
    """
    Import CellDART's high-level pred_cellf_celldart pipeline.

    The high-level API matches the official tutorial / paper recipe:
    pred_cellf_celldart(adata_sp, adata_sc, ...) — runs marker selection,
    pseudo-spot synthesis, adversarial domain adaptation, and prediction
    in one call. It returns a copy of the spatial AnnData with one obs column
    '<cell type>_cellf' per cell type (nothing in obsm) and writes the same
    columns to cellfraction.csv in outdir.
    """
    # CELLDART_SRC first: the candidates below are where the clone happened to sit on the machine
    # this was written on, not anywhere the installer creates.
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "CellDART"),
        "/opt/CellDART",
    ]
    env_override = os.environ.get("CELLDART_SRC")
    if env_override:
        candidates.insert(0, env_override)
    # Each insert(0, ...) lands in front of the one before it, so walking the list forwards left
    # the LAST candidate first on sys.path and CELLDART_SRC last: an operator's override lost to
    # any stale clone at a hard-coded path. Walk it backwards so sys.path reads in list order.
    for candidate_path in reversed(candidates):
        if not os.path.isdir(candidate_path):
            continue
        if candidate_path in sys.path:
            sys.path.remove(candidate_path)
        sys.path.insert(0, candidate_path)
    try:
        from CellDART.pred_cellf_celldart import pred_cellf_celldart

        return pred_cellf_celldart
    except ImportError as e:
        raise ImportError(
            "Could not import CellDART.pred_cellf_celldart.pred_cellf_celldart. Looked under "
            f"$CELLDART_SRC and {', '.join(candidates)}. Install CellDART into the celldart_env "
            "conda environment, or set CELLDART_SRC to the directory holding your CellDART clone."
        ) from e


#: What ran, for params.method. There is no substitute path: pred_cellf_celldart runs or the run fails.
METHOD_NAME = (
    "CellDART pred_cellf_celldart: Wilcoxon marker panel, pseudo-spots mixed from the reference, "
    "adversarial domain adaptation to the spatial spots"
)

#: What pred_cellf_celldart trains on, in its own terms. Kept beside the code that reports it.
GENE_SELECTION = (
    "CellDART marker panel: the union of each cell type's top num_markers genes by a Wilcoxon "
    "rank_genes_groups on the reference, kept where the gene is on the spatial panel; both objects "
    "are cut to it before pseudo-spot mixing and training"
)


@contextlib.contextmanager
def _record_marker_ranking(sink: dict):
    """Keep the ranking pred_cellf_celldart picks its marker genes from.

    CellDART does not train on the genes it is handed. It ranks the reference with
    ``sc.tl.rank_genes_groups(single_all, celltype, method='wilcoxon')``, keeps the union of each
    cell type's top ``num_markers`` names that are on the spatial panel, and cuts both objects to
    that set (``pred_cellf_celldart.py``: ``res_genes.extend(df_genelists.head(num_markers)...)``,
    ``adata_final[:, inter_genes_comb]``, ``spatial_all[:, inter_genes_comb]``). None of that
    reaches its return value, so the payload could only report the full panels. The call looks
    ``rank_genes_groups`` up on ``scanpy.tl`` when it runs, so for the length of the call that
    attribute records the ``names`` table the vendor then reads; the ranking itself is untouched.
    """
    tools_module = sc.tl
    real = tools_module.rank_genes_groups

    def recording(adata, groupby, *args, **kwargs):
        result = real(adata, groupby, *args, **kwargs)
        target = result if result is not None else adata
        key = kwargs.get("key_added") or "rank_genes_groups"
        try:
            names = target.uns[key]["names"]
        except Exception:  # an unexpected layout means "not recorded", never a failed run
            names = None
        if names is not None:
            sink["names"] = pd.DataFrame.from_records(names)
            sink["groupby"] = groupby
            sink["method"] = kwargs.get("method")
        return result

    tools_module.rank_genes_groups = recording
    try:
        yield sink
    finally:
        tools_module.rank_genes_groups = real


def _marker_panel(ranking: pd.DataFrame, num_markers: int, spatial_genes) -> tuple[dict, list]:
    """Rebuild CellDART's marker panel from the ranking it used, the way it builds it.

    Returns ``(marker_for, panel)``: every gene in some cell type's top ``num_markers`` mapped to
    the cell types it ranks for, and the sorted subset of those on the spatial panel -- the genes
    both objects were cut to.
    """
    top = ranking.head(int(num_markers))
    marker_for: dict = {}
    for column in top:
        for gene in top[column].tolist():
            types_for_gene = marker_for.setdefault(str(gene), [])
            if str(column) not in types_for_gene:
                types_for_gene.append(str(column))
    on_panel = {str(g) for g in spatial_genes}
    panel = sorted(g for g in marker_for if g in on_panel)
    return marker_for, panel


def _write_marker_csv(path: Path, marker_for: dict, panel: list) -> None:
    """One row per ranked marker, flagged by whether CellDART trained on it; written atomically."""
    used = set(panel)
    table = pd.DataFrame(
        {
            "gene": sorted(marker_for),
            "marker_for": [";".join(marker_for[g]) for g in sorted(marker_for)],
            "used_for_training": [g in used for g in sorted(marker_for)],
        }
    )
    partial = path.with_name(path.name + ".partial")
    table.to_csv(partial, index=False)
    os.replace(partial, path)


#: Rows of X the counts check reads at a time, so it never builds a dense copy of the matrix.
COUNT_CHECK_BLOCK_ROWS = 1024

#: What CellDART does to both inputs before anything else (``pred_cellf_celldart.py``: ``normalize_total``
#: to 1e4, ``log1p`` for the marker ranking, ``np.log1p`` again inside ``log_minmaxscale``).
COUNTS_NOTE = "CellDART normalises its input as counts (normalize_total to 1e4, then log1p)"


def _count_report(X, rows=None) -> dict:
    """Check every stored value of ``X`` (only the rows in ``rows``, when given) for what counts cannot hold.

    Returns the number of stored values read and how many of them are negative, NaN/inf, or not whole
    numbers, with the first fractional value as an example and the largest finite value. Sparse-aware:
    ``COUNT_CHECK_BLOCK_ROWS`` rows are read at a time, through a sparse matrix's stored values.
    """
    import scipy.sparse as sp

    rep = {"n_values": 0, "n_negative": 0, "n_non_finite": 0, "n_non_integer": 0, "example": None, "max": None}
    n_rows = int(X.shape[0])
    index = np.arange(n_rows) if rows is None else np.flatnonzero(np.asarray(rows, dtype=bool))
    for start in range(0, len(index), COUNT_CHECK_BLOCK_ROWS):
        block = X[index[start : start + COUNT_CHECK_BLOCK_ROWS]]
        values = np.asarray(block.data if sp.issparse(block) else block).ravel()
        if values.size == 0:
            continue
        rep["n_values"] += int(values.size)
        if values.dtype == bool:
            continue
        if np.issubdtype(values.dtype, np.integer):
            finite = values
        else:
            keep = np.isfinite(values)
            rep["n_non_finite"] += int(values.size - np.count_nonzero(keep))
            finite = values[keep]
            fractional = finite != np.rint(finite)
            n_fractional = int(np.count_nonzero(fractional))
            if n_fractional and rep["example"] is None:
                rep["example"] = float(finite[fractional][0])
            rep["n_non_integer"] += n_fractional
        rep["n_negative"] += int(np.count_nonzero(finite < 0))
        if finite.size:
            top = float(finite.max())
            rep["max"] = top if rep["max"] is None else max(rep["max"], top)
    return rep


def _holds_counts(rep: dict) -> bool:
    """True when a :func:`_count_report` found only non-negative whole numbers."""
    return not (rep["n_negative"] or rep["n_non_finite"] or rep["n_non_integer"])


def _count_problems(rep: dict) -> str:
    """What a :func:`_count_report` found, as a clause: ``of its N stored values, a are ..., b are ...``."""
    parts = []
    if rep["n_non_integer"]:
        parts.append(f"{rep['n_non_integer']} are not whole numbers (e.g. {rep['example']!r})")
    if rep["n_negative"]:
        parts.append(f"{rep['n_negative']} are negative")
    if rep["n_non_finite"]:
        parts.append(f"{rep['n_non_finite']} are NaN or infinite")
    return f"of its {rep['n_values']} stored values, " + ", ".join(parts)


def _read_counts(adata, what: str, use_raw_counts: bool):
    """``(adata, source, note)``: the matrix CellDART will read for one input, and where it came from.

    CellDART's own ``count_from_raw=True`` reads ``.raw`` for BOTH inputs and fails on one without it,
    so the worker makes the choice per input instead: with ``use_raw_counts`` an input that carries an
    ``adata.raw`` is read from it (obs and obsm come along), and one without keeps X, with a note.
    """
    if not use_raw_counts:
        return adata, "X", ""
    if adata.raw is None:
        return (
            adata,
            "X",
            f"use_raw_counts=True, but the {what} h5ad has no adata.raw, so its X was used.",
        )
    _log(f"use_raw_counts=True: reading the {what}'s counts from adata.raw ({adata.raw.n_vars} genes)")
    return adata.raw.to_adata(), "raw", ""


def _check_counts(adata, what: str, source: str) -> str:
    """Refuse a matrix CellDART turns into NaN; describe one that is not integer counts. Returns the warning.

    ``pred_cellf_celldart`` runs ``normalize_total`` on each input and ``log1p`` on the result, so a
    scaled matrix (negative values) gives NaN rows, which its network predicts as NaN fractions --
    and the run used to report ok. Those are refused, naming what ``adata.raw`` holds. A matrix of
    fractional non-negative values still runs (CPM-like input normalises the same way), but a
    log-normalised one is logged a second time, so the payload says so.
    """
    rep = _count_report(adata.X)
    if _holds_counts(rep):
        return ""
    matrix = "adata.raw" if source == "raw" else "X"
    hint = ""
    if source == "raw":
        hint = "It was read from adata.raw because use_raw_counts=True."
    elif adata.raw is None:
        hint = f"The {what} h5ad has no adata.raw to read counts from."
    else:
        raw_rep = _count_report(adata.raw.X)
        if _holds_counts(raw_rep):
            hint = (
                f"adata.raw of the {what} h5ad holds non-negative integer counts ({adata.raw.n_vars} genes): pass "
                "use_raw_counts=True to read the counts from it."
            )
        else:
            hint = f"adata.raw of the {what} h5ad is not counts either ({_count_problems(raw_rep)})."
    if rep["n_negative"] or rep["n_non_finite"]:
        raise ValueError(
            f"The {what}'s {matrix} is not counts: {_count_problems(rep)}. {COUNTS_NOTE}, and log1p of a value "
            f"below -1 is NaN, so the fractions would come out NaN. {hint} Otherwise supply raw counts in X."
        )
    largest = f", largest {rep['max']!r}" if rep["max"] is not None else ""
    return (
        f"The {what}'s {matrix} is not integer counts: {rep['n_non_integer']} of {rep['n_values']} stored values "
        f"are fractional (e.g. {rep['example']!r}{largest}). {COUNTS_NOTE}, so a log-normalised matrix is logged a "
        f"second time. {hint}".rstrip()
    )


def _refuse_colliding_names(labels, cell_type_key: str) -> None:
    """Refuse cell-type labels that become the same column once ``/`` and spaces become ``_``.

    ``'T cell'`` and ``'T_cell'`` are two cell types to CellDART and one column name after
    ``sanitize_cell_type_names``: the proportions CSV was written with a duplicated header and the run
    then died on ``obs[ct] = prop_df[ct].values`` (a two-column frame). Checked on the labels before
    training, so hours of training are not spent first, and on the columns again before anything is written.
    """
    labels = [str(label) for label in labels]
    safe, _ = sanitize_cell_type_names(labels)
    if len(set(safe)) == len(safe):
        return
    groups: dict = {}
    for original, new in zip(labels, safe):
        groups.setdefault(new, []).append(original)
    clashes = "; ".join(f"{originals} -> {new!r}" for new, originals in sorted(groups.items()) if len(originals) > 1)
    raise ValueError(
        f"cell-type labels in obs['{cell_type_key}'] collide once '/' and ' ' become '_': {clashes}. Each group "
        "would become one proportion column; relabel them so they stay distinct."
    )


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    df.to_csv(partial)
    os.replace(partial, path)


def _write_h5ad_atomic(adata, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    try:
        adata.write_h5ad(partial)
        os.replace(partial, path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def _run_celldart(
    sc_h5ad_path: str,
    spatial_h5ad_path: str,
    output_dir: str = default_output_dir(),
    cell_type_key: str = "cell_type",
    num_markers: int = 20,
    nmix: int = 20,
    npseudo: int = 20000,
    alpha: float = 0.6,
    alpha_lr: int = 5,
    emb_dim: int = 64,
    batch_size: int = 64,
    n_iterations: int = 3000,
    init_train_epoch: int = 10,
    seed_num: int = 0,
    gpu: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Core CellDART pipeline. Returns a WorkerOutput dict.

    Uses the high-level pred_cellf_celldart() pipeline. npseudo, alpha, alpha_lr, emb_dim,
    n_iterations and init_train_epoch default to the CellDART tutorial's values; nmix=20 and
    batch_size=64 are this wrapper's own defaults (the tutorial uses nmix=8, batch_size=512, and
    pred_cellf_celldart's own defaults are 10 and 512). ``use_raw_counts`` reads each input's
    counts from its ``adata.raw`` when it has one (see :func:`_read_counts`).
    """

    pred_cellf_celldart = _import_pred_cellf()

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load data ----
    _log(f"Reading scRNA-seq reference: {sc_h5ad_path}")
    adata_sc = sc.read_h5ad(sc_h5ad_path)
    _log(f"Reading spatial data: {spatial_h5ad_path}")
    adata_st = sc.read_h5ad(spatial_h5ad_path)

    # Background spots (obs['in_tissue'] == 0, 56-70% of a CELLxGENE Visium export) hold ambient counts,
    # not cells: deconvolving them published a composition for bare glass and put them in the target
    # domain the network is adapted to. Left out and reported, as the other spot tools do.
    n_spots_supplied = int(adata_st.n_obs)
    adata_st, _, n_off_tissue = keep_in_tissue(adata_st, "spots")
    if n_off_tissue:
        _log(f"Left out {n_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0")

    # Which matrix each input is read from: X, or adata.raw when use_raw_counts is set.
    adata_sc, sc_counts_source, sc_note = _read_counts(adata_sc, "reference", use_raw_counts)
    adata_st, st_counts_source, st_note = _read_counts(adata_st, "spatial input", use_raw_counts)
    counts_notes = [note for note in (sc_note, st_note) if note]

    adata_sc.var_names_make_unique()
    adata_st.var_names_make_unique()

    _log(f"scRNA: {adata_sc.n_obs} cells x {adata_sc.n_vars} genes")
    _log(f"Spatial: {adata_st.n_obs} spots x {adata_st.n_vars} genes")

    # ---- 2. Validate cell_type_key + drop NA labels ----
    if cell_type_key not in adata_sc.obs.columns:
        raise ValueError(
            f"cell_type_key='{cell_type_key}' not found in scRNA obs. Available keys: {list(adata_sc.obs.columns)}"
        )

    # The mask below rebinds adata_sc to a subset, so this is the last point at which n_obs still
    # means "the reference the caller handed us". Capture it here; the payload reports both counts.
    n_cells_sc_supplied = int(adata_sc.n_obs)

    mask = adata_sc.obs[cell_type_key].notna()
    mask &= adata_sc.obs[cell_type_key].astype(str) != "nan"
    mask &= adata_sc.obs[cell_type_key].astype(str) != ""
    adata_sc = adata_sc[mask].copy()

    n_cells_sc_used = int(adata_sc.n_obs)
    reference_note = describe_reduction(
        "reference cells",
        n_cells_sc_supplied,
        n_cells_sc_used,
        f"an unusable value in the '{cell_type_key}' column (missing, 'nan' or empty)",
    )
    if reference_note:
        _log(reference_note.strip())

    if not hasattr(adata_sc.obs[cell_type_key], "cat"):
        adata_sc.obs[cell_type_key] = adata_sc.obs[cell_type_key].astype("category")

    cell_types = list(adata_sc.obs[cell_type_key].cat.categories)
    _log(f"Found {len(cell_types)} cell types: {cell_types}")
    # The vendor names its columns '<label>_cellf' for each label present; refuse labels that would share
    # a column before training rather than after.
    _refuse_colliding_names(sorted(adata_sc.obs[cell_type_key].astype(str).unique()), cell_type_key)

    # ---- 2b. What CellDART will normalise as counts ----
    counts_warnings = []
    for adata, what, source in (
        (adata_sc, "reference", sc_counts_source),
        (adata_st, "spatial input", st_counts_source),
    ):
        warning = _check_counts(adata, what, source)
        if warning:
            _log(warning)
            counts_warnings.append(warning)

    # ---- 3. Harmonize gene IDs ----
    _log("Harmonizing gene IDs between scRNA and spatial data...")
    gene_id_report: dict = {}
    n_shared_pre = harmonize_gene_ids(adata_sc, adata_st, report=gene_id_report)
    _log(f"Shared genes after harmonization: {n_shared_pre}")
    if n_shared_pre == 0:
        raise ValueError(
            "No shared genes between scRNA and spatial data even after gene ID harmonization. Check gene ID formats."
        )

    # ---- 4. Run CellDART's high-level pipeline (nmix / batch_size are this wrapper's defaults) ----
    _log(
        f"Running pred_cellf_celldart: num_markers={num_markers}, "
        f"nmix={nmix}, npseudo={npseudo}, alpha={alpha}, "
        f"n_iterations={n_iterations}, init_train_epoch={init_train_epoch}, "
        f"batch_size={batch_size}, gpu={gpu}"
    )

    ranking: dict = {}
    with _record_marker_ranking(ranking):
        adata_st_out = pred_cellf_celldart(
            adata_sp=adata_st,
            adata_sc=adata_sc,
            # Always False: the worker has already put the chosen counts in X of each input (_read_counts).
            # The vendor's True reads .raw of BOTH inputs and fails on one that has none.
            count_from_raw=False,
            gpu=gpu,
            celltype=cell_type_key,
            num_markers=num_markers,
            seed_num=seed_num,
            nmix=nmix,
            npseudo=npseudo,
            alpha=alpha,
            alpha_lr=alpha_lr,
            emb_dim=emb_dim,
            batch_size=batch_size,
            n_iterations=n_iterations,
            init_train_epoch=init_train_epoch,
            outdir=str(output_path),
            return_anndata=True,
        )

    # The genes CellDART actually trained on. Read off the ranking it made, not recomputed.
    marker_csv_path = None
    n_marker_genes = None
    n_genes_used = None
    panel_warning = ""
    if ranking.get("names") is not None:
        marker_for, panel = _marker_panel(ranking["names"], num_markers, adata_st.var_names)
        n_marker_genes = len(marker_for)
        n_genes_used = len(panel)
        marker_csv_path = output_path / "celldart_marker_genes.csv"
        _write_marker_csv(marker_csv_path, marker_for, panel)
        _log(f"CellDART trained on {n_genes_used} marker genes ({n_marker_genes} ranked, rest off the spatial panel)")
    else:
        panel_warning = (
            "CellDART's marker ranking was not seen during the run (sc.tl.rank_genes_groups was not "
            "called the way pred_cellf_celldart calls it), so the genes it trained on are not "
            "reported; n_genes_used is null. CellDART trains on a marker panel, not on the full "
            "panels n_genes_sc / n_genes describe."
        )
        _log(panel_warning)

    # ---- 5. Recover proportions: a named obsm['cellfraction'] table if one is returned, else the CSV ----
    # Only a table that carries its own cell-type names is read off the returned object. A bare array
    # has none, and pred_cellf_celldart numbers its cell types in the iteration order of a Python set,
    # so no order can be assumed for its columns: the worker used to label such an array with the
    # SORTED type names -- or with invented ct_0, ct_1, ... when the counts differed -- which could put
    # every fraction under the wrong cell type. cellfraction.csv names each column itself.
    prop_df = None
    if adata_st_out is not None and hasattr(adata_st_out, "obsm") and "cellfraction" in adata_st_out.obsm:
        prop = adata_st_out.obsm["cellfraction"]
        if isinstance(prop, pd.DataFrame):
            prop_df = prop.copy()
            prop_df.index = prop_df.index.astype(str)
        else:
            _log("obsm['cellfraction'] carries no cell-type names; reading the named cellfraction.csv instead")

    if prop_df is None:
        cf_csv = output_path / "cellfraction.csv"
        if not cf_csv.exists():
            raise RuntimeError(f"CellDART produced no cellfraction (obsm or {cf_csv})")
        prop_df = pd.read_csv(cf_csv, index_col=0)
        prop_df.columns = [c[:-6] if c.endswith("_cellf") else c for c in prop_df.columns]
        prop_df.index = prop_df.index.astype(str)

    # Re-align rows to original spatial obs index. A spot CellDART returned no fraction for becomes an
    # all-zero row -- not a composition -- so it is counted and named instead of passing for one.
    spot_index = adata_st.obs.index.astype(str)
    returned = set(prop_df.index)
    missing_spots = [spot for spot in spot_index if spot not in returned]
    prop_df = prop_df.reindex(spot_index, fill_value=0.0)

    # A NaN fraction is not a composition. The row normalisation below keeps NaN, so such a run used to
    # publish NaN rows at status ok; refuse it before anything is written.
    finite_rows = np.isfinite(prop_df.to_numpy(dtype=float)).all(axis=1)
    if not finite_rows.all():
        bad = prop_df.index[~finite_rows]
        raise RuntimeError(
            f"CellDART returned NaN or infinite cell-type fractions for {len(bad)} of {len(prop_df)} spots "
            f"(e.g. {bad[0]!r}), so no proportions were published. The inputs passed the counts check; the "
            "network itself produced them (e.g. a diverged training run)."
        )

    # Sanitize column names for HDF5 safety; two labels that meet in one name would make one column of two.
    _refuse_colliding_names(prop_df.columns, cell_type_key)
    safe_cols, ct_renames = sanitize_cell_type_names(prop_df.columns)
    prop_df.columns = safe_cols

    # Normalize rows to sum to 1 if they don't already
    row_sums = prop_df.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.05):
        _log("Normalizing proportions to sum to 1...")
        prop_df = prop_df.div(row_sums.replace(0, 1.0), axis=0)

    # Save proportions CSV (canonical name)
    prop_csv_path = output_path / "celldart_proportions.csv"
    _write_csv_atomic(prop_df, prop_csv_path)
    _log(f"Saved proportions to {prop_csv_path}")

    # Store in spatial AnnData
    adata_st.obsm["celldart_proportions"] = prop_df.values
    for ct in safe_cols:
        adata_st.obs[ct] = prop_df[ct].values

    out_h5ad = output_path / "celldart_spatial.h5ad"
    _log(f"Saving annotated spatial AnnData to {out_h5ad}")
    _write_h5ad_atomic(adata_st, out_h5ad)

    # ---- 6. Build output ----
    cell_type_names = list(prop_df.columns)
    n_celltypes = len(cell_type_names)
    # n_spots is the slide the caller supplied; n_spots_used the in-tissue spots CellDART deconvolved.
    n_spots = int(adata_st.n_obs)

    try:
        # Over the spots that have an estimate: an all-zero row's idxmax is just its first column.
        dominant_ct = prop_df.drop(index=missing_spots).idxmax(axis=1)
        dominant_counts = dict(Counter(dominant_ct))
    except Exception:
        dominant_counts = None

    out = WorkerOutput("celldart", task="deconvolution")
    out.set_data(
        # House polarity: bare key = what the caller supplied, `_used` = what CellDART trained on.
        # The label filter above can drop a large slice of the reference; reading n_obs here reported
        # the survivors as though they were the reference. n_genes_sc / n_genes are the full panels
        # handed to CellDART; n_genes_used is the marker panel it cut BOTH objects to (one number,
        # because the two are cut to the same genes), and n_marker_genes the ranked markers before
        # the spatial-panel intersection.
        n_cells_sc=n_cells_sc_supplied,
        n_cells_sc_used=n_cells_sc_used,
        n_genes_sc=int(adata_sc.n_vars),
        n_spots=n_spots_supplied,
        n_spots_used=n_spots,
        n_genes=int(adata_st.n_vars),
        n_marker_genes=n_marker_genes,
        n_genes_used=n_genes_used,
        n_spots_without_estimate=len(missing_spots),
    )
    out.add_output_files(
        {
            "proportions_csv": str(prop_csv_path),
            "spatial_h5ad": str(out_h5ad),
        }
    )
    if marker_csv_path is not None:
        out.add_output_file("marker_genes_csv", str(marker_csv_path))
    out.add_params(
        {
            "cell_type_key": cell_type_key,
            "num_markers": num_markers,
            "nmix": nmix,
            "npseudo": npseudo,
            "alpha": alpha,
            "alpha_lr": alpha_lr,
            "emb_dim": emb_dim,
            "batch_size": batch_size,
            "n_iterations": n_iterations,
            "init_train_epoch": init_train_epoch,
            "seed_num": seed_num,
            "gpu": gpu,
            # n_genes_sc/n_genes above are the full panels, not what the two datasets have in common;
            # this is the shared count after harmonisation, before CellDART's own marker cut.
            "n_shared_genes": int(n_shared_pre),
            "gene_selection": GENE_SELECTION,
            "use_raw_counts": bool(use_raw_counts),
            # The matrix each input was read from: "X", or "raw" (adata.raw, with use_raw_counts).
            "counts_source": {"reference": sc_counts_source, "spatial": st_counts_source},
        }
    )
    out.add_params(gene_id_harmonization_params(gene_id_report))
    out.add_params(cell_type_rename_params(ct_renames))
    record_method(out, METHOD_NAME, used_fallback=False)
    record_in_tissue(out, n_spots_supplied, n_off_tissue, "spots")
    if counts_notes or counts_warnings:
        out.add_warnings(counts_notes + counts_warnings)
    out.set_summary(
        n_cell_types=n_celltypes,
        cell_type_names=cell_type_names,
        dominant_counts=dominant_counts,
    )
    # A successful run never carries stderr into the payload (base_mcp attaches stderr_tail only on a
    # non-zero exit), so the reference cells we dropped have to travel in the payload itself.
    if reference_note:
        out.add_warnings([reference_note.strip()])
    if panel_warning:
        out.add_warning(panel_warning)
    if missing_spots:
        out.add_warning(
            f"CellDART returned no cell-type fractions for {len(missing_spots)} of {n_spots} spots "
            f"(e.g. {missing_spots[0]!r}); their rows in celldart_proportions.csv are all zeros, not a composition."
        )
    if n_genes_used is not None:
        panel_note = (
            f" CellDART trained on {n_genes_used} marker genes -- the union of each cell type's top "
            f"{num_markers} Wilcoxon markers in the reference that are on the spatial panel "
            f"({n_marker_genes} ranked) -- not on the {int(adata_sc.n_vars)} reference / "
            f"{int(adata_st.n_vars)} spatial genes supplied."
        )
    else:
        panel_note = " CellDART trained on a marker panel whose size was not recorded for this run."
    raw_inputs = [
        name for name, src in (("reference", sc_counts_source), ("spatial", st_counts_source)) if src == "raw"
    ]
    if raw_inputs:
        panel_note += (
            f" Counts were read from adata.raw for the {' and '.join(raw_inputs)} input (use_raw_counts=True)."
        )
    if n_off_tissue:
        panel_note += (
            f" {n_off_tissue} of the {n_spots_supplied} spots have obs['in_tissue'] == 0 and were left out; "
            f"the {n_spots} in-tissue spots were deconvolved."
        )
    out.set_analysis(
        build_deconv_analysis(
            n_celltypes=n_celltypes,
            dominant_counts=dominant_counts,
            total_spots=n_spots,
            method_name="CellDART",
        )
        + panel_note
        + reference_note
        + gene_id_harmonization_note(gene_id_report)
        + cell_type_rename_note(ct_renames)
    )

    return out.to_dict()


def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="CellDART worker: domain-adaptation deconvolution pipeline")
    parser.add_argument("--sc-h5ad", required=True, help="Path to scRNA-seq reference h5ad")
    parser.add_argument("--spatial-h5ad", required=True, help="Path to spatial h5ad")
    parser.add_argument("--output-dir", default=default_output_dir(), help="Output directory")
    parser.add_argument("--cell-type-key", default="cell_type", help="obs column with cell-type labels")
    parser.add_argument("--num-markers", type=int, default=20, help="Markers per cell type")
    parser.add_argument(
        "--nmix", type=int, default=20, help="Cells per pseudo-spot (wrapper default; the CellDART tutorial uses 8)"
    )
    parser.add_argument("--npseudo", type=int, default=20000, help="Pseudo-spots to synthesize")
    parser.add_argument("--alpha", type=float, default=0.6, help="Adversarial loss weight")
    parser.add_argument("--alpha-lr", type=int, default=5, help="Domain LR multiplier")
    parser.add_argument("--emb-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size (wrapper default; the CellDART tutorial uses 512)"
    )
    parser.add_argument("--n-iterations", type=int, default=3000, help="Adversarial training iterations")
    parser.add_argument("--init-train-epoch", type=int, default=10, help="Pre-training epochs")
    parser.add_argument("--seed-num", type=int, default=0, help="Random seed")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
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
    )

    error_info = None
    error_exc = None
    with _redirect_stdout_to_stderr():
        try:
            result = _run_celldart(
                sc_h5ad_path=args.sc_h5ad,
                spatial_h5ad_path=args.spatial_h5ad,
                output_dir=args.output_dir,
                cell_type_key=args.cell_type_key,
                num_markers=args.num_markers,
                nmix=args.nmix,
                npseudo=args.npseudo,
                alpha=args.alpha,
                alpha_lr=args.alpha_lr,
                emb_dim=args.emb_dim,
                batch_size=args.batch_size,
                n_iterations=args.n_iterations,
                init_train_epoch=args.init_train_epoch,
                seed_num=args.seed_num,
                gpu=args.gpu,
                use_raw_counts=args.use_raw_counts,
            )
        except Exception as e:
            _log(f"ERROR: {e}")
            traceback.print_exc(file=sys.stderr)
            error_info = str(e)
            error_exc = e

    # stdout: JSON only (must be outside redirect block)
    if error_info is not None:
        WorkerOutput.emit_error("celldart", error_info, task="deconvolution", exc=error_exc)
        sys.exit(1)

    print(json.dumps(result, default=str))
    sys.stdout.flush()


if __name__ == "__main__":
    _cli_main()
