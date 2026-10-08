#!/usr/bin/env python3
"""
SpatialPrompt SpatialOmicsLab worker (runs inside /opt/conda/envs/spatialpromptENV).

Design goals:
- Accept multiple spatial input modes (Visium 10x dir+counts_h5, or .h5ad).
- Accept single-cell reference (.h5ad) with a specified cell-type label column.
- Expose two high-level tools:
    1) spatialprompt_deconvolution: cell-type proportion inference for spatial spots
    2) spatialprompt_cluster: downstream clustering/labeling of spots (optional)

The upstream SpatialPrompt API takes plain arrays, never an AnnData -- both entry points
call np.array() on the matrix they are handed. As installed (0.0.6):
- SpatialDeconvolution().predict_cell_prop(sc_array, st_array, sc_genes, st_genes,
      sc_labels, x_cord, y_cord, n_hvgs=1000, min_cell=10, max_cell=15, return_prop=True,
      spot_ratio=[0.33, 0.33, 0.33], n_neighbor=45, n_itr=3)  -> per-spot proportions
- SpatialCluster().fit_predict(st_array, x_cord, y_cord, n_neighbor=20, n_itr=3,
      n_cluster="auto", W=0.4, n_hvgs=1000)  -> pd.Categorical, one label per spot

Both are called by keyword, and test/test_spatialprompt_calls_the_library_it_ships_with.py
holds the signatures above against the installed package. Note that SpatialCluster does not
consume cell-type proportions: upstream pairs a deconvolution with a clustering through
SpatialDeconvolution().cluster_cell_annotation(ct_deconv, cluster_annot, threshold=0.30).

This worker writes artifacts into output_dir and prints a single JSON payload to stdout.
All logs go to stderr.

spatialprompt_cluster therefore takes a proportions table as an option, not a requirement:
without one it segments the slide and stops, and with one it goes on to name each cluster
through cluster_cell_annotation. The table is never an input to fit_predict itself.

What each entry point analyses, and what it reports (data.* counts are supplied / used):
- Spots: a clean 0/1 obs['in_tissue'] flag is honoured in both modes -- in_tissue == 0 spots are
  background and are left out before anything runs (data.n_spots_out_of_tissue_excluded, a
  warning, and a sentence in the analysis). visium_10x mode carries the flag over from the
  positions file, so a raw_feature_bc_matrix.h5 and a filtered one give the same spots, and h5ad
  mode (the only mode for a CELLxGENE export) now agrees with both.
- Genes, deconvolution: min_counts (total counts over the analysed spots) is applied whatever
  max_genes is, then max_genes keeps the most variable spatial genes; both cuts are reported
  (data.n_genes / n_genes_below_min_counts / n_genes_used) before SpatialPrompt picks its own
  n_hvgs=1000 reference HVGs among the genes shared with the reference (data.n_genes_modelled).
- Genes, clustering: SpatialCluster keeps the n_hvgs=1000 genes with the highest raw-count
  variance as its first step, so only those columns are densified -- chosen sparsely here with
  the same rule -- instead of the whole spots x genes matrix (a VisiumHD slide densified over
  18,085 genes is ~37 GB before upstream copies it). data.n_genes_used says how many it clustered.
- Dense matrices are intrinsic to the library (it takes plain arrays and copies them into
  pandas frames), so their size is estimated first against the memory this process can still
  allocate (worker_utils.available_memory_bytes: MemAvailable, or the room left under the cgroup
  memory limit when that is smaller) and a run that cannot fit stops with the numbers rather than
  being killed part-way.
- Counts: both entry points TPM-normalise what they are handed, so the matrix is chosen by
  worker_utils.choose_counts_matrix (params.expression_source): negative or non-finite X is refused,
  non-integer X runs with a warning, and use_raw_counts runs on adata.raw.
- stdout carries only the JSON payload: the library prints progress bars and timings, so every call
  into it runs under contextlib.redirect_stdout(sys.stderr). Output files are written through a
  ``.partial`` file and renamed.

Direct-invocation extra: that table normally arrives as the cell_type_prop_csv path the
MCP portal sends. A caller driving this worker by hand may instead put it inline under the
payload key "cell_type_prop" (anything pd.DataFrame() accepts, indexed to match the
spatial spots). The portal has no parameter for that, so it is a bare --json convenience
only, and no error message offers it.
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
    TISSUE_POSITIONS_NAMES,
    WorkerOutput,
    available_memory_bytes,
    build_cluster_analysis,
    build_deconv_analysis,
    choose_counts_matrix,
    describe_reduction,
    drop_unlabeled,
    find_tissue_positions,
    id_mismatch_msg,
    identifier_rename_note,
    identifier_rename_params,
    make_names_unique_and_report,
    read_indexed_table,
    read_tissue_positions,
    record_ignored,
    record_method,
    unsupported_choice_msg,
)

# The share of a cluster a cell type must hold to be named in that cluster's annotation.
# spatialprompt 0.0.6 defaults cluster_cell_annotation(threshold=0.30) and steps down to
# 0.20 for any cluster nothing clears; we pass its own default rather than inventing one,
# and do not expose it -- the portal has no parameter for it, so a caller could not set it.
ANNOTATION_THRESHOLD = 0.30

# spatialprompt 0.0.6's own n_hvgs defaults, passed explicitly so the counts this worker reports
# are the counts that ran: fit_predict(n_hvgs=1000) keeps the 1000 genes with the highest raw
# variance (SpatialCluster.__hvg_detect__), and predict_cell_prop(n_hvgs=1000) keeps the 1000
# most variable reference genes among those shared with the slide (when at least 1000 are).
CLUSTER_N_HVGS = 1000
DECONV_N_HVGS = 1000

# Bytes per spots x genes element alive at once inside fit_predict: our float32 matrix, the
# float32 HVG slice upstream keeps, its TPM/min-max frame, the neighbourhood average, and the
# float32 hstack of the last two min-max scaled again -- about eight float32 copies.
CLUSTER_BYTES_PER_ELEMENT = 32
# predict_cell_prop copies each matrix into a DataFrame, casts it (a copy), drops duplicated
# columns (a copy) and selects the shared genes (a copy) before its HVG cut: four float32 copies.
DECONV_BYTES_PER_ELEMENT = 16


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _json_out(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, indent=2))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _read_json_arg(s: str) -> dict[str, Any]:
    try:
        return json.loads(s)
    except Exception as e:
        raise ValueError(f"--json must be valid JSON. error={e!r}") from e


def _import_spatialprompt():
    """
    Import SpatialPrompt with friendly error if not installed.
    """
    try:
        import spatialprompt  # type: ignore

        return spatialprompt
    except Exception as e:
        raise RuntimeError(
            "Failed to import 'spatialprompt'. Ensure SpatialPrompt is installed in /opt/conda/envs/spatialpromptENV."
        ) from e


def _memory_budget_bytes():
    """The memory this process can still allocate, or None where the platform cannot say.

    ``worker_utils.available_memory_bytes``: the smaller of ``MemAvailable`` and the room left under
    the cgroup memory limit (page cache counted reclaimable). ``MemAvailable`` alone reports the
    host's memory inside a memory-limited container, so a run the container cannot hold passed the
    check and was OOM-killed with no JSON. Kept under this name as the seam the check reads.
    """
    return available_memory_bytes()


def _check_dense_fits(need_bytes: int, what: str, remedy: str) -> None:
    """Refuse, with the numbers, before allocating dense state the machine cannot hold.

    SpatialPrompt takes plain arrays and copies them into pandas frames, so a dense matrix is
    intrinsic to the method. Estimating it up front turns an OOM kill part-way through (an empty
    stdout and a SIGKILL) into a message that names both sizes and what can be changed.
    """
    budget = _memory_budget_bytes()
    if budget is not None and need_bytes > budget:
        raise MemoryError(
            f"SpatialPrompt needs ~{need_bytes / 1e9:.1f} GB of dense state for {what}, but ~{budget / 1e9:.1f} GB "
            f"is available (MemAvailable, or the room left under the cgroup memory limit when that is smaller). "
            f"{remedy}"
        )


def _in_tissue_mask(obs):
    """``(keep_mask or None, note)`` for the Space Ranger ``in_tissue`` flag in ``obs``.

    None when there is no usable flag: no column, or values that are not a clean 0/1 flag -- then
    ``note`` says so and no spot is left out on a guess. A flag that marks every spot off the
    tissue is an error: there would be nothing left to analyse.
    """
    import numpy as np
    import pandas as pd

    if "in_tissue" not in obs.columns:
        return None, ""
    flag = pd.to_numeric(pd.Series(np.asarray(obs["in_tissue"], dtype=object)), errors="coerce")
    values = set(flag.dropna().unique().tolist())
    if flag.isna().any() or not values <= {0, 1}:
        return None, (
            "obs['in_tissue'] is present but is not a 0/1 flag, so no spot was left out by it; every spot was "
            "analysed as tissue."
        )
    keep = flag.to_numpy() == 1
    if not keep.any():
        raise ValueError(
            f"obs['in_tissue'] marks all {len(keep)} spots as off the tissue (in_tissue == 0), so nothing is left "
            "to analyse. Fix the flag, or drop the column if every spot is tissue."
        )
    return keep, ""


def _keep_in_tissue(adata):
    """``(adata, n_off_tissue, notes)`` -- the spots on the tissue, and how many were not.

    Off-tissue spots hold ambient counts, not cells: on the CELLxGENE Heart Fetal12W slide 3,009 of
    4,992 spots are in_tissue == 0, and clustering them spends domains on background. Space Ranger's
    filtered matrix already leaves them out, so this is also what makes an h5ad export of a slide
    and its Space Ranger directory analyse the same spots.
    """
    keep, note = _in_tissue_mask(adata.obs)
    notes = [note] if note else []
    if keep is None or bool(keep.all()):
        return adata, 0, notes
    n_off = int((~keep).sum())
    eprint(f"[SpatialPrompt] Leaving out {n_off} of {adata.n_obs} spots with obs['in_tissue'] == 0")
    notes.append(
        f"obs['in_tissue'] marks {n_off} of {adata.n_obs} spots as off the tissue (in_tissue == 0); they were "
        f"left out, so the results cover the {adata.n_obs - n_off} on-tissue spots."
    )
    return adata[keep].copy(), n_off, notes


def _in_tissue_params(n_supplied: int, n_off: int) -> dict:
    """``params.in_tissue_filter`` in the shape ``worker_utils.record_in_tissue`` gives every other worker.

    Written here by hand because :func:`_keep_in_tissue` already puts its own warning on the payload;
    the shared helper would add a second one for the same spots.
    """
    if not n_off:
        return {}
    return {
        "in_tissue_filter": {
            "n_spots_supplied": int(n_supplied),
            "n_spots_off_tissue_dropped": int(n_off),
            "n_spots_used": int(n_supplied - n_off),
        }
    }


def _counts_matrix_for(adata, use_raw_counts: bool, what: str, lenient_without_raw: bool = False):
    """``(adata, info)``: the matrix SpatialPrompt is handed for one input (``worker_utils.choose_counts_matrix``).

    SpatialPrompt TPM-normalises whatever it gets (``__normalisation1/2/3__``, ``__norm_hvg__``), so X
    is treated as counts: negative or non-finite X is refused (its row sums make the TPM meaningless),
    naming ``use_raw_counts`` when ``adata.raw`` holds counts; non-integer X runs as before with a
    warning; ``use_raw_counts`` runs on ``adata.raw``. Messages are prefixed with ``what``.
    ``lenient_without_raw`` (the reference) keeps X, with a warning, when ``use_raw_counts`` is set and
    that input has no ``adata.raw``; the slide is strict.
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


def _spatial_counts(adata, use_raw_counts: bool, input_mode: str):
    """``(adata, info, ignored_why)`` for the slide; a Space Ranger matrix has no ``adata.raw`` to switch to."""
    if input_mode == "visium_10x" and use_raw_counts:
        adata, info = _counts_matrix_for(adata, False, "spatial slide")
        return (
            adata,
            info,
            "input_mode='visium_10x' reads Space Ranger's count matrix, which is counts and has no adata.raw; "
            "use_raw_counts applies to an h5ad whose X is processed.",
        )
    adata, info = _counts_matrix_for(adata, use_raw_counts, "spatial slide")
    return adata, info, ""


def _write_atomic(path: str, write) -> None:
    """``write(<path>.partial)`` then rename, so a killed run never leaves a half-written output."""
    partial = path + ".partial"
    try:
        write(partial)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def _column_variance(X, chunk: int = 1 << 24):
    """Population variance (ddof=0) of every column of ``X``, in float64, without densifying it.

    Sparse input is walked over its stored values in chunks of ``chunk`` entries and dense input
    over row blocks, so the only allocation that scales with the matrix is one chunk.
    """
    import numpy as np
    import scipy.sparse as sp

    n_obs, n_vars = X.shape
    s1 = np.zeros(n_vars, dtype=np.float64)
    s2 = np.zeros(n_vars, dtype=np.float64)
    if n_obs == 0:
        return s1
    if sp.issparse(X):
        if X.format not in ("csr", "csc"):
            X = X.tocsr()
        if not X.has_canonical_format:
            # Duplicate entries sum, so their squares must be taken after they are summed.
            X = X.copy()
            X.sum_duplicates()
        nnz = int(X.nnz)
        for lo in range(0, nnz, chunk):
            hi = min(lo + chunk, nnz)
            if X.format == "csr":
                cols = np.asarray(X.indices[lo:hi])
            else:
                cols = np.searchsorted(X.indptr, np.arange(lo, hi), side="right") - 1
            vals = np.asarray(X.data[lo:hi], dtype=np.float64)
            s1 += np.bincount(cols, weights=vals, minlength=n_vars)
            s2 += np.bincount(cols, weights=vals * vals, minlength=n_vars)
    else:
        step = max(1, chunk // max(1, n_vars))
        for r0 in range(0, n_obs, step):
            block = np.asarray(X[r0 : r0 + step], dtype=np.float64)
            s1 += block.sum(axis=0)
            s2 += np.square(block).sum(axis=0)
    mean = s1 / n_obs
    return np.maximum(s2 / n_obs - mean * mean, 0.0)


def _top_variance_columns(X, n_top: int):
    """Indices of the ``n_top`` columns with the highest variance, highest first; None if all fit.

    This is SpatialCluster.__hvg_detect__'s rule -- ``np.argsort(df_array.var(axis=0))[::-1][0:N]``
    on the raw matrix -- applied before densifying instead of after, so only the columns the
    library keeps are ever made dense. The variances are float64 here and float32 upstream, so two
    genes whose variances agree to float32 rounding may be ranked either way; exact ties go to the
    earlier column. Upstream then re-ranks these N columns itself and keeps all of them.
    """
    import numpy as np

    n_vars = int(X.shape[1])
    if n_top <= 0 or n_vars <= n_top:
        return None
    var = _column_variance(X)
    return np.argsort(-var, kind="stable")[:n_top]


def _filter_spatial_genes(adata, max_genes: int, min_counts):
    """``(adata, n_below_min_counts, n_cut_by_max_genes)`` after the deconvolution's two gene cuts.

    ``min_counts`` (total counts over the analysed spots) is applied on its own, whatever
    ``max_genes`` is -- it used to run only inside the variance cut, so ``max_genes=0`` or a panel
    already at or under ``max_genes`` skipped it without a word. ``max_genes`` then keeps the genes
    with the highest spatial variance; the ranking arithmetic is the one this worker always used, so
    a run that cut before keeps the same genes now.
    """
    import numpy as np
    import scipy.sparse as sp

    X = adata.X
    n_vars = int(adata.n_vars)
    keep = np.ones(n_vars, dtype=bool)
    if min_counts is not None and float(min_counts) > 0:
        gene_sum = np.asarray(X.sum(axis=0)).ravel()
        keep = gene_sum >= float(min_counts)
    n_below = int((~keep).sum())
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        raise ValueError(
            f"min_counts={min_counts} removed all {n_vars} spatial genes: none has that many total counts over the "
            f"{adata.n_obs} spots analysed. Lower min_counts (0 disables it), or check that X holds raw counts."
        )
    n_cut = 0
    if max_genes is not None and int(max_genes) > 0 and idx.size > int(max_genes):
        if sp.issparse(X):
            # scipy reduces each column on its own, so ranking the full matrix and indexing gives the
            # same numbers as ranking the filtered copy the old helper made -- without the copy.
            mean = np.asarray(X.mean(axis=0)).ravel()[idx]
            # E[X^2] - mean^2
            ex2 = np.asarray(X.power(2).mean(axis=0)).ravel()[idx]
            var = ex2 - mean**2
        else:
            # numpy's float reductions are not column-independent bit for bit, so a dense matrix is
            # ranked on the filtered columns exactly as the old helper ranked it.
            dense = np.asarray(X)
            var = (dense[:, idx] if idx.size < n_vars else dense).var(axis=0)
        order = np.argsort(-var)[: int(max_genes)]
        n_cut = int(idx.size - order.size)
        idx = idx[order]
    if n_below == 0 and n_cut == 0:
        return adata, 0, 0
    return adata[:, idx].copy(), n_below, n_cut


def _count_threshold(value):
    """``min_counts`` as a number, integral values kept as int so the payload echoes what was sent."""
    number = float(value)
    return int(number) if number.is_integer() else number


def _load_sc_h5ad(sc_h5ad: str):
    import anndata as ad

    if not os.path.exists(sc_h5ad):
        raise FileNotFoundError(f"sc_h5ad not found: {sc_h5ad}")
    return ad.read_h5ad(sc_h5ad)


def _load_spatial_h5ad(spatial_h5ad: str):
    import anndata as ad

    if not os.path.exists(spatial_h5ad):
        raise FileNotFoundError(f"spatial_h5ad not found: {spatial_h5ad}")
    return ad.read_h5ad(spatial_h5ad)


# Both portal functions declare the slide under three names -- spatial_h5ad, st_h5ad,
# h5ad_path -- and collapse them before sending, so over MCP the spelling never reaches
# here. A direct --json caller's does: the smoke drivers and any recipe that shells this
# worker build the payload by hand. Both entry points resolve through this one place so
# they cannot drift apart again, and so a payload with no slide in it is told which keys
# would have supplied one instead of being handed back the name of a key it never saw.
_SPATIAL_H5AD_KEYS = ("spatial_h5ad", "st_h5ad", "h5ad_path")


def _resolve_spatial_h5ad(params: dict[str, Any]) -> str:
    for key in _SPATIAL_H5AD_KEYS:
        path = params.get(key)
        if path:
            return str(path)
    accepted = " or ".join(repr(key) for key in _SPATIAL_H5AD_KEYS)
    raise ValueError(f"h5ad mode requires the spatial slide in params, under {accepted}.")


def _load_visium_10x(counts_h5: str, spatial_dir: str):
    """
    Load Visium counts from 10x H5 and attach the spot coordinates into adata.obsm['spatial'].

    Supports both Space Ranger v1 'tissue_positions_list.csv' and v2 'tissue_positions.csv'.
    """
    import scanpy as sc

    if not os.path.exists(counts_h5):
        raise FileNotFoundError(f"counts_h5 not found: {counts_h5}")
    if not os.path.isdir(spatial_dir):
        raise FileNotFoundError(f"spatial_dir not found: {spatial_dir}")

    eprint(f"[SpatialPrompt] Reading 10x h5: {counts_h5}")
    adata = sc.read_10x_h5(counts_h5)
    # Unique gene names, reported: the counts land in adata.uns["identifier_renames"], which the
    # deconvolution carries into its payload (a renamed symbol exists in no reference).
    make_names_unique_and_report(adata, axes=("var",))

    # locate positions file
    pos_path = find_tissue_positions(spatial_dir)
    if pos_path is None:
        raise FileNotFoundError(
            f"Could not find {' or '.join(TISSUE_POSITIONS_NAMES)} under {spatial_dir} or its spatial/ subdirectory"
        )

    # Space Ranger v1 has no header; v2 has one, and renamed the file at the same release. The
    # shared reader handles both, and decides which it is from the row rather than from the name.
    df = read_tissue_positions(pos_path).set_index("barcode", drop=True)

    # align by barcodes (adata.obs_names are barcodes)
    common = df.index.intersection(adata.obs_names)
    if len(common) == 0:
        raise ValueError(
            id_mismatch_msg("barcodes", "counts", adata.obs_names, "positions", df.index) + f" positions={pos_path}"
        )
    adata = adata[common].copy()
    df = df.loc[common]

    # choose coordinates: prefer array_row/array_col if exist else the full-resolution pixels
    if "array_row" in df.columns and "array_col" in df.columns:
        coords = df[["array_row", "array_col"]].to_numpy().astype(float)
    elif "pxl_row_in_fullres" in df.columns and "pxl_col_in_fullres" in df.columns:
        coords = df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy().astype(float)
    else:
        raise ValueError(f"Positions file missing expected coordinate columns: {pos_path}")

    adata.obsm["spatial"] = coords
    # Store for convenience
    adata.obs["x"] = coords[:, 0]
    adata.obs["y"] = coords[:, 1]
    # Carried over so a raw_feature_bc_matrix.h5 is held to the same spots as a filtered one:
    # the positions file lists every barcode on the array, on the tissue or not.
    adata.obs["in_tissue"] = df["in_tissue"].to_numpy()

    # auto-detect image path (optional; not required by SpatialPrompt core)
    img_candidates = [
        os.path.join(spatial_dir, "spatial", "tissue_hires_image.png"),
        os.path.join(spatial_dir, "spatial", "tissue_lowres_image.png"),
    ]
    auto_he = next((p for p in img_candidates if os.path.exists(p)), None)
    eprint(f"[SpatialPrompt] Loaded spatial: obs={adata.n_obs}, vars={adata.n_vars}, auto_he={auto_he}")
    return adata, auto_he


def _extract_xy_from_spatial_adata(adata) -> tuple[Any, Any]:
    """Extract x/y coordinates from an AnnData.

    Priority:
      1) adata.obsm['spatial'] (Nx2)
      2) obs columns among: ('x','y'), ('pxl_col_in_fullres','pxl_row_in_fullres'),
         ('array_col','array_row')
    Returns:
      x_cord, y_cord arrays (length n_obs)
    """
    import numpy as np

    if hasattr(adata, "obsm") and "spatial" in adata.obsm:
        arr = adata.obsm["spatial"]
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 0], arr[:, 1]

    obs = adata.obs
    candidates = [
        ("x", "y"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("array_col", "array_row"),
        ("imagecol", "imagerow"),
    ]
    for xc, yc in candidates:
        if xc in obs.columns and yc in obs.columns:
            return obs[xc].to_numpy(), obs[yc].to_numpy()

    raise KeyError(
        "Could not find spatial coordinates in adata.obsm['spatial'] or common obs columns "
        "(x/y, pxl_col_in_fullres/pxl_row_in_fullres, array_col/array_row)."
    )


def tool_spatialprompt_deconvolution(params: dict[str, Any]) -> dict[str, Any]:
    """Run SpatialPrompt deconvolution (Tutorial 1).

    The upstream tutorial uses the `spatialprompt` python package (import name: spatialprompt).
    The key API is:
        deconv = spatialprompt.SpatialDeconvolution()
        cell_type_prop = deconv.predict_cell_prop(
            sc_array=..., st_array=..., sc_genes=..., st_genes=...,
            sc_labels=..., x_cord=..., y_cord=...
        )

    This worker supports:
      - spatial input_mode=visium_10x (10x h5 + spatial_dir)
      - spatial input_mode=h5ad (.h5ad with spatial coords in obsm['spatial'] or obs columns)
      - sc reference as .h5ad with labels in obs[sc_label_key]

    Before the library runs: off-tissue spots (a 0/1 obs['in_tissue'] == 0) are left out, spatial
    genes with fewer than min_counts total counts are removed, and max_genes (when > 0) keeps the
    most variable of the rest. Every cut is counted in data.* and named in the analysis.
    """
    sp = _import_spatialprompt()
    import numpy as np
    import pandas as pd

    output_dir = _ensure_dir(params["output_dir"])

    # ----------------------------
    # Load spatial dataset
    # ----------------------------
    input_mode = params.get("input_mode", "visium_10x")
    max_genes = int(params.get("max_genes", 2000))
    min_counts = _count_threshold(params.get("min_counts", 1))
    random_seed = int(params.get("random_seed", 0))
    allow_drop_unlabeled = bool(params.get("drop_unlabeled", False))
    use_raw_counts = bool(params.get("use_raw_counts", False))

    # predict_cell_prop takes no seed of its own and builds its whole simulated training
    # set from the global numpy RNG -- __random_spot_generator__ draws both how many cells
    # go into each simulated spot and which ones, and the three scenarios above it do the
    # same. Unseeded, two runs of one slide disagreed by up to 0.074 in proportion and
    # flipped the dominant cell type on 0.6% of spots; seeded alike they are identical.
    # The singleton is the only lever there is, so set it before the library is reached.
    np.random.seed(random_seed)

    if input_mode == "visium_10x":
        adata_sp, auto_he = _load_visium_10x(params["counts_h5"], params["spatial_dir"])
    elif input_mode == "h5ad":
        adata_sp = _load_spatial_h5ad(_resolve_spatial_h5ad(params))
    else:
        raise ValueError(f"Unsupported input_mode={input_mode!r}. Use 'visium_10x' or 'h5ad'.")

    # Gene renames made by the 10x loader, if any; deduplication itself happens once the matrix is chosen.
    renamed_sp = {"n_genes_renamed": 0, "n_cells_renamed": 0}
    if input_mode == "visium_10x":
        renamed_sp.update(dict(adata_sp.uns.get("identifier_renames") or {}))

    n_spots_supplied = int(adata_sp.n_obs)
    warnings_out: list[str] = []

    # Off-tissue spots are background: deconvolving them reports cell types for ambient counts,
    # and upstream's neighbourhood step averages them into every edge spot on the tissue.
    adata_sp, n_off_tissue, tissue_notes = _keep_in_tissue(adata_sp)
    warnings_out.extend(tissue_notes)
    n_spots_used = int(adata_sp.n_obs)

    # SpatialPrompt TPM-normalises the slide as counts; the shared rule picks the matrix it gets.
    adata_sp, sp_counts_info, raw_ignored_why = _spatial_counts(adata_sp, use_raw_counts, input_mode)
    if sp_counts_info.get("warning"):
        warnings_out.append(sp_counts_info["warning"])
    # Duplicate gene symbols (common in 10x features) become X, X-1, ...: names no reference carries,
    # so a renamed gene can only match by accident. Counted, not silent.
    make_names_unique_and_report(adata_sp, into=renamed_sp, axes=("var",))
    n_genes_supplied = int(adata_sp.n_vars)

    # min_counts first, whatever max_genes is; then max_genes keeps the most variable genes.
    adata_sp, n_below_min_counts, n_cut_by_max_genes = _filter_spatial_genes(adata_sp, max_genes, min_counts)
    n_genes_used = int(adata_sp.n_vars)
    if n_below_min_counts or n_cut_by_max_genes:
        eprint(
            f"[SpatialPrompt] Spatial genes: {n_genes_supplied} supplied, {n_below_min_counts} below "
            f"min_counts={min_counts}, {n_cut_by_max_genes} beyond max_genes={max_genes}, {n_genes_used} kept"
        )

    # Coordinates
    x_cord, y_cord = _extract_xy_from_spatial_adata(adata_sp)

    # ----------------------------
    # Load sc reference
    # ----------------------------
    sc_h5ad = params["sc_h5ad"]
    sc_label_key = params.get("sc_label_key", "cell_type")
    eprint(f"[SpatialPrompt] Loading sc reference: {sc_h5ad}")
    adata_sc = _load_sc_h5ad(sc_h5ad)

    if sc_label_key not in adata_sc.obs.columns:
        raise KeyError(
            f"sc_label_key={sc_label_key!r} not found in sc.obs. Available: {list(adata_sc.obs.columns)[:50]}"
        )
    n_cells_supplied = int(adata_sc.n_obs)
    # A missing label is not a class: .astype(str) used to turn NaN into the cell type "nan",
    # which upstream then simulated spots from and reported a proportion for.
    keep_cells, n_unlabeled = drop_unlabeled(
        adata_sc.obs[sc_label_key].to_numpy(), allow_drop_unlabeled, what=f"cells in sc.obs['{sc_label_key}']"
    )
    if n_unlabeled:
        eprint(f"[SpatialPrompt] Leaving out {n_unlabeled} reference cells with no '{sc_label_key}' label")
        adata_sc = adata_sc[keep_cells].copy()
        warnings_out.append(
            f"{n_unlabeled} of {n_cells_supplied} reference cells have no '{sc_label_key}' label and were left out "
            f"(drop_unlabeled=True); the reference used has {adata_sc.n_obs} cells."
        )
    # The reference is TPM-normalised as counts too; without an adata.raw it keeps X under use_raw_counts.
    adata_sc, sc_counts_info = _counts_matrix_for(
        adata_sc, use_raw_counts, "single-cell reference", lenient_without_raw=True
    )
    if sc_counts_info.get("warning"):
        warnings_out.append(sc_counts_info["warning"])
    renamed_sc = make_names_unique_and_report(adata_sc, axes=("var",))
    sc_labels = adata_sc.obs[sc_label_key].astype(str).to_numpy()

    # ----------------------------
    # Harmonize gene space (intersection)
    # ----------------------------
    st_genes = adata_sp.var_names.astype(str)
    sc_genes = adata_sc.var_names.astype(str)

    gene_inter = np.intersect1d(st_genes, sc_genes)
    if gene_inter.size == 0:
        raise RuntimeError(
            "No shared genes between spatial and sc reference after loading. Check gene IDs (symbols vs Ensembl)."
        )

    # predict_cell_prop copies both matrices into dense frames over the shared genes before it
    # narrows them to its HVGs, so that size is the one to check.
    _check_dense_fits(
        DECONV_BYTES_PER_ELEMENT * (n_spots_used + int(adata_sc.n_obs)) * int(gene_inter.size),
        f"{n_spots_used} spots and {adata_sc.n_obs} reference cells over {gene_inter.size} shared genes",
        "SpatialPrompt takes these as dense arrays; lower max_genes to shrink the shared gene panel, or run on a "
        "machine with more memory.",
    )

    # Reindex both to intersection in the same order
    adata_sp_i = adata_sp[:, gene_inter].copy()
    adata_sc_i = adata_sc[:, gene_inter].copy()

    # Dense arrays for SpatialPrompt (many implementations expect numpy arrays)
    st_array = adata_sp_i.X
    sc_array = adata_sc_i.X
    if not isinstance(st_array, np.ndarray):
        st_array = st_array.toarray()
    if not isinstance(sc_array, np.ndarray):
        sc_array = sc_array.toarray()

    # Convert to float32 for speed/memory
    st_array = np.asarray(st_array, dtype=np.float32)
    sc_array = np.asarray(sc_array, dtype=np.float32)

    # ----------------------------
    # Run SpatialPrompt deconvolution
    # ----------------------------
    eprint(
        "[SpatialPrompt] Running SpatialDeconvolution().predict_cell_prop(sc_array, st_array, sc_genes, st_genes, "
        f"sc_labels, x_cord, y_cord, n_hvgs={DECONV_N_HVGS}) on {n_spots_used} spots x {gene_inter.size} genes"
    )
    deconv = sp.SpatialDeconvolution()

    # Call with keywords to be robust to positional changes. The library prints its progress bars
    # and timings to stdout, which carries only this worker's JSON: they go to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        cell_type_prop = deconv.predict_cell_prop(
            sc_array=sc_array,
            st_array=st_array,
            sc_genes=np.asarray(gene_inter, dtype=object),
            st_genes=np.asarray(gene_inter, dtype=object),
            sc_labels=sc_labels,
            x_cord=np.asarray(x_cord, dtype=np.float32),
            y_cord=np.asarray(y_cord, dtype=np.float32),
            n_hvgs=DECONV_N_HVGS,
        )

    # Normalize output into a DataFrame
    if isinstance(cell_type_prop, pd.DataFrame):
        prop_df = cell_type_prop
    else:
        try:
            prop_df = pd.DataFrame(cell_type_prop)
        except Exception as e:
            raise RuntimeError(
                f"Unsupported output type from predict_cell_prop: {type(cell_type_prop)}; cannot convert to DataFrame: {e!r}"
            ) from e

    # One row per analysed spot, in input order. Anything else cannot be keyed back to the slide,
    # and a CSV indexed 0..n-1 would read as a table of spots that are not these.
    if prop_df.shape[0] != adata_sp.n_obs:
        raise RuntimeError(
            f"predict_cell_prop returned {prop_df.shape[0]} rows for {adata_sp.n_obs} spots, so they cannot be keyed "
            "back to the slide."
        )
    prop_df.index = adata_sp.obs_names

    # ----------------------------
    # Write outputs
    # ----------------------------
    out_csv = os.path.join(output_dir, "cell_type_proportions.csv")
    _write_atomic(out_csv, prop_df.to_csv)

    # Build summary stats
    celltypes = list(prop_df.columns.astype(str))
    n_celltypes = len(celltypes)
    dominant = prop_df.idxmax(axis=1)
    dominant_counts = dominant.value_counts().to_dict()

    # upstream __preprocessing__: all shared genes when fewer than n_hvgs, else its n_hvgs HVGs.
    n_overlap = int(len(gene_inter))
    n_genes_modelled = n_overlap if n_overlap < DECONV_N_HVGS else DECONV_N_HVGS

    gene_cuts = []
    if n_below_min_counts:
        gene_cuts.append(f"min_counts={min_counts} ({n_below_min_counts} genes with fewer total counts)")
    if n_cut_by_max_genes:
        gene_cuts.append(f"max_genes={max_genes} ({n_cut_by_max_genes} less variable genes)")
    gene_note = describe_reduction("spatial genes", n_genes_supplied, n_genes_used, " and ".join(gene_cuts))
    spot_note = describe_reduction(
        "spots", n_spots_supplied, n_spots_used, "the in_tissue flag, which marks spots off the tissue"
    )
    if gene_note:
        warnings_out.append(gene_note.strip())

    out = WorkerOutput("spatialprompt", task="deconvolution")
    out.set_data(
        n_spots=n_spots_supplied,
        n_spots_used=n_spots_used,
        n_spots_out_of_tissue_excluded=n_off_tissue,
        n_genes=n_genes_supplied,
        n_genes_below_min_counts=n_below_min_counts,
        n_genes_used=n_genes_used,
        n_cells_sc=n_cells_supplied,
        n_cells_sc_used=int(adata_sc.n_obs),
        n_cells_dropped_unlabeled=int(n_unlabeled),
        n_genes_sc=int(adata_sc.n_vars),
        n_overlap_genes=n_overlap,
        n_genes_modelled=n_genes_modelled,
    )
    out.add_output_files(
        {
            "cell_type_proportions_csv": out_csv,
        }
    )
    out.add_params(
        {
            "sc_label_key": sc_label_key,
            "input_mode": input_mode,
            "max_genes": max_genes,
            "min_counts": min_counts,
            "random_seed": random_seed,
            "drop_unlabeled": allow_drop_unlabeled,
            "use_raw_counts": use_raw_counts,
            "expression_source": sp_counts_info["expression_source"],
            "x_matrix_kind": sp_counts_info["x_matrix_kind"],
            "expression_source_sc": sc_counts_info["expression_source"],
            "x_matrix_kind_sc": sc_counts_info["x_matrix_kind"],
        }
    )
    out.add_params(_in_tissue_params(n_spots_supplied, n_off_tissue))
    out.add_params(identifier_rename_params(renamed_sp))
    out.add_params(identifier_rename_params(renamed_sc, suffix="sc"))
    record_method(out, "SpatialPrompt SpatialDeconvolution.predict_cell_prop")
    if raw_ignored_why:
        record_ignored(out, "use_raw_counts", raw_ignored_why)
    if warnings_out:
        out.add_warnings(warnings_out)
    out.set_summary(
        n_celltypes=n_celltypes,
        celltypes=celltypes,
        dominant_counts=dominant_counts,
    )
    out.set_analysis(
        build_deconv_analysis(
            n_celltypes,
            dominant_counts,
            total_spots=n_spots_used,
            method_name="SpatialPrompt",
        )
        + spot_note
        + gene_note
        + f" SpatialPrompt modelled {n_genes_modelled} genes: "
        + (
            f"all {n_overlap} genes shared with the reference."
            if n_genes_modelled == n_overlap
            else f"the {n_genes_modelled} most variable in the reference among the {n_overlap} shared with it."
        )
        + identifier_rename_note(renamed_sp, "spatial data")
        + identifier_rename_note(renamed_sc, "reference")
    )
    return out.to_dict()


def tool_spatialprompt_cluster(params: dict[str, Any]) -> dict[str, Any]:
    spatialprompt = _import_spatialprompt()
    import numpy as np
    import pandas as pd

    output_dir = _ensure_dir(params["output_dir"])
    input_mode = params.get("input_mode", "visium_10x")
    random_seed = _safe_int(params.get("random_seed", 0), 0)
    use_raw_counts = bool(params.get("use_raw_counts", False))

    # Same reason as the deconvolution half: fit_predict has no seed formal, and the
    # TruncatedSVD and KMeans inside __kmean_clus__ are built without a random_state, so
    # both fall through sklearn's check_random_state(None) to this same singleton. Left
    # unseeded the partition was stable on the slide measured but the label integers
    # permuted between runs, which is enough to break anything keyed by cluster id.
    np.random.seed(random_seed)

    # Load spatial
    auto_he = None
    if input_mode == "visium_10x":
        adata_sp, auto_he = _load_visium_10x(params["counts_h5"], params["spatial_dir"])
    elif input_mode == "h5ad":
        adata_sp = _load_spatial_h5ad(_resolve_spatial_h5ad(params))
        if "spatial" not in adata_sp.obsm and {"x", "y"} <= set(adata_sp.obs.columns):
            adata_sp.obsm["spatial"] = np.vstack([adata_sp.obs["x"].to_numpy(), adata_sp.obs["y"].to_numpy()]).T
    else:
        raise ValueError(unsupported_choice_msg("input_mode", input_mode, ["visium_10x", "h5ad"]))

    # The proportions table is not an input to the segmentation. fit_predict takes
    # (st_array, x_cord, y_cord, n_neighbor, n_itr, n_cluster, W, n_hvgs) and has no slot
    # for cell-type composition, so requiring one gated the cheapest way to get spatial
    # domains behind a deconvolution that could not move a single label. Where the table
    # is upstream's business is one method further on -- cluster_cell_annotation, below,
    # names each finished cluster after the cell types that dominate it -- so it is read
    # here and used after the clustering, or not supplied at all.
    #
    # A direct --json caller may pass the table inline under "cell_type_prop" instead. The
    # portal has no parameter for that, so no message offers it.
    prop_csv = params.get("cell_type_prop_csv")
    prop_df = None
    if prop_csv:
        if not os.path.exists(prop_csv):
            raise FileNotFoundError(f"cell_type_prop_csv not found: {prop_csv}")
        prop_df = read_indexed_table(prop_csv, "cell type proportions")
    elif params.get("cell_type_prop") is not None:
        # Positional against the slide as supplied, so it is built before any spot is left out.
        prop_df = pd.DataFrame(params["cell_type_prop"], index=adata_sp.obs_names)

    # Off-tissue spots are background with ambient counts: clustering them spends domains on
    # the empty part of the array. Space Ranger's filtered matrix already leaves them out, so
    # this is also what makes h5ad mode and visium_10x mode cluster the same spots.
    n_spots_supplied = int(adata_sp.n_obs)
    adata_sp, n_off_tissue, notes = _keep_in_tissue(adata_sp)
    n_spots_used = int(adata_sp.n_obs)

    # SpatialCluster TPM-normalises its HVG columns as counts; the shared rule picks the matrix.
    adata_sp, sp_counts_info, raw_ignored_why = _spatial_counts(adata_sp, use_raw_counts, input_mode)
    if sp_counts_info.get("warning"):
        notes.append(sp_counts_info["warning"])

    # Align the table to the slide, never the slide to the table. A table that matches no
    # spot at all is still an abort: the caller named a file and not one of its rows is a
    # spot here, which is a mistake worth stopping for. Partial coverage is not -- every
    # spot is clustered either way, and only the annotation is withheld.
    prop_aligned = None
    n_covered = 0
    if prop_df is not None:
        common = adata_sp.obs_names.intersection(prop_df.index)
        n_covered = len(common)
        if n_covered == 0:
            raise ValueError(
                id_mismatch_msg("spot IDs", "spatial data", adata_sp.obs_names, "cell_type proportions", prop_df.index)
            )
        prop_aligned = prop_df.reindex(adata_sp.obs_names)

    n_clust = _safe_int(params.get("n_clust", 8), 8)
    clust_label = params.get("clust_label", "spatialprompt_cluster")

    # fit_predict takes the spots x genes matrix and the two coordinate arrays -- its
    # first act is np.array() on st_array, so an AnnData raises there. Its second is
    # __hvg_detect__, which keeps the n_hvgs columns with the highest raw variance and drops
    # the rest, so densifying every gene first only built a matrix the library discards: a
    # VisiumHD slide (507,684 x 18,085) is ~37 GB dense before upstream copies it. Pick those
    # columns sparsely by the same rule and densify only them.
    X = adata_sp.X
    n_genes_supplied = int(adata_sp.n_vars)
    top = _top_variance_columns(X, CLUSTER_N_HVGS)
    if top is not None:
        X = X[:, top]
    n_genes_used = int(X.shape[1])
    _check_dense_fits(
        CLUSTER_BYTES_PER_ELEMENT * n_spots_used * n_genes_used,
        f"{n_spots_used} spots x {n_genes_used} genes (SpatialCluster's own top-{CLUSTER_N_HVGS} variance genes)",
        "SpatialCluster works on a dense matrix of every analysed spot; run on a machine with more memory.",
    )
    st_array = X if isinstance(X, np.ndarray) else (X.toarray() if hasattr(X, "toarray") else np.asarray(X))
    st_array = np.asarray(st_array, dtype=np.float32)
    x_cord, y_cord = _extract_xy_from_spatial_adata(adata_sp)

    eprint(
        f"[SpatialPrompt] Running SpatialCluster().fit_predict(n_cluster={n_clust}, n_hvgs={CLUSTER_N_HVGS}) on "
        f"{n_spots_used} spots x {n_genes_used} of {n_genes_supplied} genes"
    )
    clust = spatialprompt.SpatialCluster()
    # Upstream 0.0.6: fit_predict(st_array, x_cord, y_cord, n_neighbor=20, n_itr=3,
    # n_cluster="auto", W=0.4, n_hvgs=1000). By keyword, for the reason the sibling call
    # gives: a positional shift upstream must not silently rebind the coordinates.
    # Leaving n_cluster unset would discard n_clust -- upstream then picks k itself.
    # The library's progress bars and timings go to stderr; stdout carries only the JSON.
    with contextlib.redirect_stdout(sys.stderr):
        cluster_assignment = clust.fit_predict(
            st_array=st_array,
            x_cord=np.asarray(x_cord, dtype=np.float32),
            y_cord=np.asarray(y_cord, dtype=np.float32),
            n_cluster=n_clust,
            n_hvgs=CLUSTER_N_HVGS,
        )

    # A pd.Categorical of one label per spot, in input order -- not an annotated object.
    cluster_assignment = np.asarray(cluster_assignment)
    if cluster_assignment.shape[0] != adata_sp.n_obs:
        raise RuntimeError(
            f"SpatialCluster returned {cluster_assignment.shape[0]} labels for {adata_sp.n_obs} "
            f"spots, so they cannot be keyed back to the slide."
        )
    labels = pd.Series(cluster_assignment, index=adata_sp.obs_names).astype(str)

    clust_csv = os.path.join(output_dir, "spatialprompt_spot_clusters.csv")
    _write_atomic(clust_csv, labels.to_frame(name=clust_label).to_csv)

    # This is what the proportions table is for. cluster_cell_annotation averages each
    # cluster's per-spot proportions and names the cluster after whichever cell types clear
    # the threshold, which is the step that turns an anonymous partition into a domain map.
    #
    # Upstream reads ct_deconv positionally against cluster_annot and keys its result by
    # range(len(np.unique(cluster_annot))), so it is only correct on integer labels that are
    # exactly 0..k-1 and cover every spot in slide order. Anything else is skipped with a
    # note rather than mis-annotated: a wrong cell-type name on a domain is worse than none.
    cluster_cell_types = None
    annot_csv = None
    codes = None
    distinct: list[int] = []
    if prop_aligned is not None:
        if n_covered < adata_sp.n_obs:
            notes.append(
                f"The proportions table covered {n_covered} of {adata_sp.n_obs} spots, so no "
                f"per-cluster cell-type annotation was produced. Every spot was still clustered. "
                f"cluster_cell_annotation averages a cluster's proportions over its spots, so a "
                f"table with gaps would name each cluster from whichever of its spots happened "
                f"to be covered."
            )
        else:
            try:
                codes = cluster_assignment.astype(int)
                if not np.array_equal(codes, cluster_assignment):
                    codes = None
            except (TypeError, ValueError):
                codes = None
            distinct = sorted({int(v) for v in codes}) if codes is not None else []
            if codes is None or distinct != list(range(len(distinct))):
                notes.append(
                    f"No per-cluster cell-type annotation was produced: SpatialCluster returned "
                    f"labels {sorted(set(labels))[:8]}, and cluster_cell_annotation indexes its "
                    f"result by range(n_clusters), so it needs the integers 0..k-1. Every spot "
                    f"was still clustered and the proportions table is unchanged."
                )
                codes = None
    if codes is not None:
        try:
            eprint(
                f"[SpatialPrompt] Naming {len(distinct)} clusters with "
                f"SpatialDeconvolution().cluster_cell_annotation(threshold={ANNOTATION_THRESHOLD})"
            )
            with contextlib.redirect_stdout(sys.stderr):
                mapping, _annotated = spatialprompt.SpatialDeconvolution().cluster_cell_annotation(
                    prop_aligned,
                    codes,
                    threshold=ANNOTATION_THRESHOLD,
                )
            cluster_cell_types = {str(k): str(v) for k, v in dict(mapping).items()}
        except Exception as e:
            notes.append(
                f"cluster_cell_annotation failed, so the clusters are unnamed: {e!r}. The "
                f"clustering itself is unaffected."
            )
            cluster_cell_types = None

    # Written beside the clusters rather than into them: the spot-cluster CSV and the h5ad
    # obs column are read downstream as "the clustering", and an extra column of cell-type
    # strings in either is routinely mistaken for the labels themselves.
    if cluster_cell_types:
        annot_csv = os.path.join(output_dir, "spatialprompt_cluster_cell_types.csv")
        annot_frame = pd.DataFrame(
            {
                "cluster": list(cluster_cell_types),
                "cell_types": [cluster_cell_types[k] for k in cluster_cell_types],
            }
        )
        _write_atomic(annot_csv, lambda p: annot_frame.to_csv(p, index=False))

    # The payload advertises an annotated object under spatial_h5ad_with_clusters, and
    # fit_predict returns bare labels, so attach them here rather than publishing a null.
    out_h5ad = os.path.join(output_dir, "spatialprompt_spatial_with_clusters.h5ad")
    try:
        adata_sp.obs[clust_label] = pd.Categorical(labels.to_numpy())
        _write_atomic(out_h5ad, adata_sp.write_h5ad)
    except Exception as e:
        eprint(f"[SpatialPrompt] Could not write {out_h5ad}: {e!r}")
        out_h5ad = None

    # Build cluster summary
    cluster_sizes = labels.value_counts().to_dict()

    out = WorkerOutput("spatialprompt", task="cluster")
    out.set_data(
        n_spots=n_spots_supplied,
        n_spots_used=n_spots_used,
        n_spots_out_of_tissue_excluded=n_off_tissue,
        n_genes=n_genes_supplied,
        n_genes_used=n_genes_used,
    )
    out.add_output_files(
        {
            "spot_clusters_csv": clust_csv,
            "spatial_h5ad_with_clusters": out_h5ad,
            "cluster_cell_types_csv": annot_csv,
        }
    )
    # dim is deliberately absent: SpatialCluster reduces with a TruncatedSVD(n_components=50)
    # that upstream 0.0.6 hardcodes, so there is no latent-dimension setting to send it to.
    # Listing it here would say it was applied.
    out.add_params(
        {
            "n_clust": n_clust,
            "random_seed": random_seed,
            "clust_label": clust_label,
            "use_raw_counts": use_raw_counts,
            "expression_source": sp_counts_info["expression_source"],
            "x_matrix_kind": sp_counts_info["x_matrix_kind"],
        }
    )
    out.add_params(_in_tissue_params(n_spots_supplied, n_off_tissue))
    record_method(out, f"SpatialPrompt SpatialCluster.fit_predict (KMeans, n_cluster={n_clust})")
    if raw_ignored_why:
        record_ignored(out, "use_raw_counts", raw_ignored_why)
    if "dim" in params:
        # The portal stopped offering this, so only a hand-built --json payload can carry it.
        # Warn on presence rather than on a value: the old check fired only when the value
        # differed from the advertised 30, which made the default the one silent case.
        record_ignored(
            out,
            ["dim"],
            f"dim={params['dim']} was not applied. SpatialCluster reduces with a fixed "
            f"TruncatedSVD(n_components=50) that spatialprompt hardcodes, so there is no "
            f"latent-dimension setting in the library for this value to reach.",
        )
    if notes:
        out.add_warnings(notes)
    out.set_summary(
        n_clusters=len(cluster_sizes),
        cluster_sizes=cluster_sizes,
        cluster_cell_types=cluster_cell_types,
    )
    spot_note = describe_reduction(
        "spots", n_spots_supplied, n_spots_used, "the in_tissue flag, which marks spots off the tissue"
    )
    gene_note = (
        f" SpatialCluster clustered on the {n_genes_used} genes with the highest raw-count variance "
        f"(its fixed n_hvgs={CLUSTER_N_HVGS}); the other {n_genes_supplied - n_genes_used} of the "
        f"{n_genes_supplied} genes supplied did not enter the clustering."
        if n_genes_used < n_genes_supplied
        else ""
    )
    out.set_analysis(
        build_cluster_analysis(
            cluster_sizes,
            cluster_key="cluster",
            total_spots=n_spots_used,
            n_requested=n_clust,
        )
        + spot_note
        + gene_note
    )
    return out.to_dict()


TOOL_DISPATCH = {
    "spatialprompt_deconvolution": tool_spatialprompt_deconvolution,
    "spatialprompt_cluster": tool_spatialprompt_cluster,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json", required=True, help="JSON string payload. Must contain '__tool__' and tool-specific params."
    )
    args = ap.parse_args()

    try:
        params = _read_json_arg(args.json)
        tool = params.get("__tool__")
        if not tool:
            raise ValueError("Missing '__tool__' in JSON payload.")
        if tool not in TOOL_DISPATCH:
            raise ValueError(f"Unknown tool: {tool}. Allowed: {sorted(TOOL_DISPATCH.keys())}")
        payload = TOOL_DISPATCH[tool](params)
        _json_out(payload)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        # Derive task from the __tool__ name if available
        task_name = None
        try:
            p = _read_json_arg(args.json)
            t = p.get("__tool__", "")
            if "deconv" in t:
                task_name = "deconvolution"
            elif "cluster" in t:
                task_name = "cluster"
        except Exception:
            pass
        WorkerOutput.emit_error("spatialprompt", str(e), task=task_name)
        sys.exit(1)


if __name__ == "__main__":
    main()
