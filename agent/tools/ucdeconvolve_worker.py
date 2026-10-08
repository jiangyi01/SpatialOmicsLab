#!/usr/bin/env python3
"""
ucdeconvolve worker for SpatialOmicsLab MCP (runs in /opt/conda/envs/ucdenv).

SpatialOmicsLab protocol:
- stdout: final JSON only
- stderr: logs/progress/tracebacks

Implements:
- UCDBase via ucdeconvolve.tl.base(): UCD's pre-trained model, run on the UCD cloud API. It needs a
  token, and the expression matrix it reads is uploaded to UCD's service.
- Optional top-celltype assignment via ucdeconvolve.utils.assign_top_celltypes(). Upstream writes
  the label to obs["<pred_key_added>_<key_added>"] ("..._sm<k>" when smoothed over k neighbours);
  summary.pred_celltype_key names the column that was actually written.
- Flexible input loaders for "any spatial transcriptomics":
  * visium_h5_spatial: 10x Visium H5 + spatial/tissue_positions_list.csv
  * spaceranger_outs: Space Ranger outs/
  * h5ad: prebuilt AnnData
  * generic_counts_coords: counts h5ad + coords csv (barcode,x,y[,in_tissue])

Measured locally before anything is uploaded, and reported in params:
- counts_source: the matrix tl.base reads -- adata.raw when use_raw is on and the input has a raw,
  otherwise adata.X (upstream's own rule). Nothing is copied into a stand-in adata.raw.
- counts_integer_valued / n_blocks_read_as_log1p: tl.base preprocesses 256-row blocks and treats a
  block whose maximum is below 80 as log1p data (expm1 first); integer counts caught by that rule
  are exponentiated, and the payload warns.
- n_genes_matched: how many input genes map onto UCD's fixed input vocabulary. An input that
  matches none is refused instead of deconvolving an all-zero matrix.
- Background spots: obs['in_tissue'] == 0 spots (every array spot of a CELLxGENE Visium export,
  or a coords_csv / raw matrix that carries them) are left out right after loading, so they are
  never uploaded; params.in_tissue_filter, data.n_spots_supplied and the analysis count them.
- A matrix holding negative or non-finite values (scaled / z-scored data) is refused: UCD accepts
  counts and log1p data by design (it un-logs and normalises itself), and nothing else.
- Knobs set away from their default whose step did not run (assign_* / groupby / knnsmooth_* with
  assign_top_celltypes off, n_neighbors with compute_neighbors off, coord_type outside the Visium
  modes) are still echoed and are listed in params.ignored with a warning.

A post-processing step the caller asked for (compute_neighbors, assign_top_celltypes) that fails
stops the run with an error. Inputs it needs are checked before the upload; a failure after the
upload still writes the annotated h5ad and the prediction CSVs first, and the error names them.

References:
- UCD base signature/params (split/sort/propagate/use_raw/key_added)
- Utilities read_results / assign_top_celltypes
- Spatial tutorial showing authenticate + tl.base usage
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from worker_utils import (
    TISSUE_POSITIONS_NAMES,
    WorkerOutput,
    describe_reduction,
    find_tissue_positions,
    id_mismatch_msg,
    keep_in_tissue,
    read_coords_csv,
    read_tissue_positions,
    record_ignored,
    record_in_tissue,
    record_method,
    unsupported_choice_msg,
)

METHOD_NAME = "UCDBase (ucdeconvolve.tl.base on the UCD cloud API)"

#: The input modes whose loader builds obsm['spatial'] from Space Ranger's positions table, and so
#: the only ones that read ``coord_type``.
VISIUM_MODES = ("visium_h5_spatial", "spaceranger_outs")
COORD_TYPES = ("array", "pixel")

#: The portal defaults of the knobs that only shape a post-processing step. A knob that differs from
#: its default while its step does not run is reported in ``params.ignored`` (it is still echoed).
POSTPROCESS_DEFAULTS = {
    "pred_key_added": "ucd_pred_celltype",
    "assign_category": None,
    "groupby": "",
    "knnsmooth_neighbors": None,
    "knnsmooth_cycles": 1,
    "n_neighbors": 10,
    "coord_type": "array",
}

#: ``tl.base`` -> ``get_preprocessed_anndata`` preprocesses the matrix in blocks of this many rows.
UCD_BLOCK_ROWS = 256
#: ``preprocess_expression``: a block whose maximum is below this is taken to be log1p data and
#: un-logged with ``expm1`` before total-count normalisation.
UCD_LOG_DETECT_MAX = 80.0
#: The categories ``split=True`` adds beside ``raw`` (UCD docs: primary / cell lines / cancer).
SPLIT_CATEGORIES = ("primary", "lines", "cancer")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def emit_json(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.flush()


def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _load_visium_h5_spatial(visium_h5_path: str, visium_spatial_dir: str, coord_type: str = "array"):
    """
    Build AnnData from 10x filtered_feature_bc_matrix.h5 + spatial metadata.
    """
    import scanpy as sc

    adata = sc.read_10x_h5(visium_h5_path)
    adata.var_names_make_unique()

    tp_path = find_tissue_positions(visium_spatial_dir)
    if tp_path is None:
        raise FileNotFoundError(f"Cannot find {' or '.join(TISSUE_POSITIONS_NAMES)} under: {visium_spatial_dir}")
    tp = read_tissue_positions(tp_path)

    # align to barcodes
    common = np.intersect1d(adata.obs_names.astype(str), tp["barcode"].values)
    if common.size == 0:
        raise ValueError(
            id_mismatch_msg("barcodes", "count matrix", adata.obs_names, Path(tp_path).name, tp["barcode"].values)
        )
    adata = adata[common].copy()
    tp = tp.set_index("barcode").loc[adata.obs_names.astype(str)].copy()

    # populate obs fields expected by scanpy/visium conventions
    adata.obs["in_tissue"] = tp["in_tissue"].astype(int).values
    adata.obs["array_row"] = tp["array_row"].astype(int).values
    adata.obs["array_col"] = tp["array_col"].astype(int).values

    # spatial coords in obsm
    if coord_type == "pixel":
        coords = np.vstack([tp["pxl_col_in_fullres"].values, tp["pxl_row_in_fullres"].values]).T.astype(float)
    else:
        # array coords: (col,row) is typical for plotting-like conventions
        coords = np.vstack([tp["array_col"].values, tp["array_row"].values]).T.astype(float)
    adata.obsm["spatial"] = coords

    # store spatial assets if present (optional)
    adata.uns["spatial"] = {}
    sf = Path(visium_spatial_dir) / "scalefactors_json.json"
    if sf.exists():
        try:
            adata.uns["spatial"]["scalefactors_json"] = json.loads(sf.read_text())
        except Exception:
            pass

    return adata


def _load_spaceranger_outs(spaceranger_dir: str, coord_type: str = "array"):
    outs = Path(spaceranger_dir)
    h5 = outs / "filtered_feature_bc_matrix.h5"
    sp = outs / "spatial"
    if not h5.exists() or not sp.exists():
        raise FileNotFoundError("Space Ranger outs/ must contain filtered_feature_bc_matrix.h5 and spatial/")

    return _load_visium_h5_spatial(str(h5), str(sp), coord_type=coord_type)


def _load_h5ad(h5ad_path: str):
    import anndata as ad

    return ad.read_h5ad(h5ad_path)


def _in_tissue_ones(values: Any) -> Any:
    """Boolean array: which entries of an in_tissue column mark tissue, by ``keep_in_tissue``'s rule.

    1 / "1" / True (any case) is tissue; 0, False, empty and anything unreadable is not.
    """
    import pandas as pd

    series = pd.Series(np.asarray(values, dtype=object))
    flag = pd.to_numeric(
        series.astype(str).str.strip().str.lower().replace({"true": "1", "false": "0"}), errors="coerce"
    )
    return np.asarray(flag == 1)


def _load_generic_counts_coords(counts_h5ad_path: str, coords_csv: str):
    import anndata as ad

    adata = ad.read_h5ad(counts_h5ad_path)

    df = read_coords_csv(coords_csv).set_index("barcode")

    common = np.intersect1d(adata.obs_names.astype(str), df.index.values)
    if common.size == 0:
        raise ValueError(
            id_mismatch_msg("barcodes", "counts_h5ad_path", adata.obs_names, "coords_csv.barcode", df.index)
        )
    adata = adata[common].copy()
    df = df.loc[adata.obs_names.astype(str)].copy()

    # read_coords_csv fills in_tissue with 1 when the coords file has no such column, and that default
    # used to overwrite the counts h5ad's own flag (a CELLxGENE export carries one), turning its
    # background spots into tissue. A spot either file marks as background stays background.
    in_tissue = _in_tissue_ones(df["in_tissue"].values)
    if "in_tissue" in adata.obs.columns:
        in_tissue = in_tissue & _in_tissue_ones(adata.obs["in_tissue"].values)
    adata.obs["in_tissue"] = in_tissue.astype(int)
    adata.obsm["spatial"] = np.vstack([df["x"].values, df["y"].values]).T.astype(float)
    return adata


def _atomic_to_csv(df: Any, path: str) -> None:
    """Write ``df`` as ``<path>.partial`` and rename over ``path`` once it is complete."""
    tmp = path + ".partial"
    df.to_csv(tmp)
    os.replace(tmp, path)


def _atomic_write_h5ad(adata: Any, path: str) -> None:
    """Write the AnnData as ``<path>.partial`` and rename over ``path`` once it is complete."""
    tmp = path + ".partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def _ucd_input_matrix(adata: Any, use_raw: Any) -> tuple[Any, Any, str]:
    """``(matrix, var_names, source)``: what ``ucd.tl.base`` will read, by upstream's own rule.

    ``try_get_raw`` in ucdeconvolve reads ``adata.raw`` when ``use_raw`` is truthy and a raw exists,
    otherwise ``adata.X``. This used to be preceded by ``_maybe_make_raw``, which set
    ``adata.raw = adata`` whenever the input had none: UCD read the same numbers either way, but the
    alias wrote a copy of X into the annotated h5ad as ``.raw`` -- telling every later reader the
    file carried raw counts, log-normalised X included -- and nothing said which matrix was sent.
    """
    raw = getattr(adata, "raw", None)
    if use_raw and raw is not None:
        return raw.X, raw.var_names, "adata.raw"
    return adata.X, adata.var_names, "adata.X"


def _scan_ucd_blocks(matrix: Any) -> dict[str, Any]:
    """One pass over the matrix in UCD's own row blocks: value range, integrality, log detection,
    and how many values are negative or not finite (the caller refuses those before the upload).

    Sparse-aware: a CSR/CSC block is inspected through its stored values only, so nothing is
    densified. The per-block maximum is over every gene, which bounds from above the maximum over
    the genes UCD keeps, so ``n_blocks_read_as_log1p`` is a lower bound on the blocks it un-logs.
    """
    import scipy.sparse as sp

    n_rows = int(matrix.shape[0])
    integer_valued = True
    vmax = None
    n_blocks = 0
    n_low = 0
    n_negative = 0
    n_nonfinite = 0
    for start in range(0, n_rows, UCD_BLOCK_ROWS):
        block = matrix[start : start + UCD_BLOCK_ROWS]
        if sp.issparse(block):
            values = np.asarray(block.data)
            implicit_zero = block.nnz < block.shape[0] * block.shape[1]
        else:
            values = np.asarray(block).ravel()
            implicit_zero = False
        finite = np.isfinite(values)
        if not bool(finite.all()):
            n_nonfinite += int((~finite).sum())
            values = values[finite]
        n_negative += int((values < 0).sum())
        bmax = float(values.max()) if values.size else 0.0
        if implicit_zero:
            bmax = max(bmax, 0.0)
        if integer_valued and values.size and not bool(np.all(np.mod(values, 1) == 0)):
            integer_valued = False
        vmax = bmax if vmax is None else max(vmax, bmax)
        n_blocks += 1
        if bmax < UCD_LOG_DETECT_MAX:
            n_low += 1
    return {
        "integer_valued": bool(integer_valued and not n_nonfinite),
        "max": float(vmax or 0.0),
        "n_blocks": n_blocks,
        "n_blocks_read_as_log1p": n_low,
        "n_negative": n_negative,
        "n_nonfinite": n_nonfinite,
    }


def _ucd_gene_overlap(var_names: Any) -> tuple[Any, Any, str]:
    """``(n_used, n_vocabulary, why_unknown)``: how many of these genes tl.base feeds the model.

    Mirrors ``tl.base``: ``match_to_gene`` maps each name (symbol, alias or Ensembl id) to a symbol,
    unmatched names and repeats are dropped, and only symbols in UCD's fixed input vocabulary
    (``metadata['target_genes']``) get a column -- every other model input is zero. Computed
    locally, before the upload.
    """
    try:
        from ucdeconvolve import _utils as ucd_utils
        from ucdeconvolve._data import metadata

        match_to_gene = ucd_utils.match_to_gene
        vocabulary = set(metadata["target_genes"])
    except (ImportError, AttributeError, KeyError, TypeError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    seen = set()
    n_used = 0
    for symbol in match_to_gene([str(v) for v in var_names]):
        if symbol is None or symbol in seen:
            continue
        seen.add(symbol)
        if symbol in vocabulary:
            n_used += 1
    return n_used, len(vocabulary), ""


def _check_postprocessing_inputs(
    adata: Any,
    assign_top: bool,
    groupby: str,
    knnsmooth_neighbors: Any,
    knnsmooth_cycles: int,
    compute_neighbors: bool,
    n_neighbors: int,
) -> None:
    """Refuse, before anything is uploaded, a post-processing request that cannot succeed."""
    if not assign_top:
        return
    if groupby and groupby not in adata.obs.columns:
        cols = [str(c) for c in adata.obs.columns]
        raise ValueError(
            f"groupby={groupby!r} is not an obs column of this input, so assign_top_celltypes cannot "
            f"give each group one label. obs columns: {cols[:30]}{' ...' if len(cols) > 30 else ''}. "
            "Pass one of them, or leave groupby empty for a label per spot."
        )
    if knnsmooth_neighbors is None:
        return
    if knnsmooth_neighbors < 1:
        raise ValueError(f"knnsmooth_neighbors={knnsmooth_neighbors} must be a positive neighbour count (or unset).")
    if knnsmooth_cycles < 1:
        raise ValueError(f"knnsmooth_cycles={knnsmooth_cycles} must be at least 1 when knnsmooth_neighbors is set.")
    if compute_neighbors:
        # sc.pp.neighbors stores n_neighbors - 1 neighbours per spot (itself excluded).
        if knnsmooth_neighbors > n_neighbors - 1:
            raise ValueError(
                f"knnsmooth_neighbors={knnsmooth_neighbors} needs that many stored neighbours per spot, and "
                f"compute_neighbors with n_neighbors={n_neighbors} stores {n_neighbors - 1}. Raise n_neighbors "
                f"to at least {knnsmooth_neighbors + 1}, or lower knnsmooth_neighbors."
            )
    elif "distances" not in adata.obsp:
        raise ValueError(
            f"knnsmooth_neighbors={knnsmooth_neighbors} smooths over adata.obsp['distances'], which this input "
            "does not carry. Set compute_neighbors=True (n_neighbors sets its k), or pass an input that "
            "already has a neighbour graph."
        )


def _check_neighbour_graph(adata: Any, knnsmooth_neighbors: int) -> None:
    """Every spot must have at least ``knnsmooth_neighbors`` stored neighbours to smooth over."""
    import scipy.sparse as sp

    graph = sp.csr_matrix(adata.obsp["distances"])
    per_row = np.diff(graph.indptr)
    fewest = int(per_row.min()) if per_row.size else 0
    if fewest < knnsmooth_neighbors:
        raise ValueError(
            f"knnsmooth_neighbors={knnsmooth_neighbors}, but adata.obsp['distances'] stores as few as {fewest} "
            "neighbours for a spot. Lower knnsmooth_neighbors, or rebuild the graph with compute_neighbors=True "
            "and a larger n_neighbors."
        )


def _result_headers(adata: Any, key: str) -> dict[str, Any]:
    """``adata.uns[key]['headers']`` -- the prediction categories UCD attached, by name."""
    from collections.abc import Mapping

    entry = adata.uns[key] if key in adata.uns else None
    headers = entry.get("headers") if isinstance(entry, Mapping) else None
    return dict(headers) if isinstance(headers, Mapping) else {}


def _export_ucd_results(ucd: Any, adata: Any, key: str, output_dir: str) -> tuple[dict[str, str], dict[str, str]]:
    """Export one CSV per prediction category UCD attached: ``(files, failures)``.

    The categories are read from what UCD returned (``adata.uns[key]['headers']``) rather than
    assumed. The old fixed list (``raw`` + primary/lines/cancer) swallowed every export failure, so
    a run could report ok with no prediction CSV at all.
    """
    out: dict[str, str] = {}
    failed: dict[str, str] = {}
    for cat in _result_headers(adata, key):
        try:
            df = ucd.utils.read_results(adata, key=key, category=cat)
            fp = str(Path(output_dir) / f"{key}_{cat}_predictions.csv")
            _atomic_to_csv(df, fp)
            out[f"{cat}_predictions_csv"] = fp
        except Exception as e:
            failed[str(cat)] = f"{type(e).__name__}: {e}"
            log(f"[UCD worker] export of category={cat!r} failed: {type(e).__name__}: {e}")
    return out, failed


def _assign_top_celltypes(
    ucd: Any,
    adata: Any,
    key: str,
    category: Any,
    groupby: Any,
    pred_key_added: str,
    knnsmooth_neighbors: Any,
    knnsmooth_cycles: int,
) -> tuple[str, str]:
    """Run ``ucd.utils.assign_top_celltypes``: ``(obs column it actually wrote, category it read)``.

    Upstream writes ``obs[f"{key_added}_{key}"]``, with ``key`` suffixed ``_sm<k>`` when smoothing --
    never ``obs[pred_key_added]``, which is the name the summary used to report. With no category,
    upstream's ``read_results`` reads ``primary`` when present, else ``all``; a category UCD did not
    return is refused here by name rather than surfacing as upstream's bare ``KeyError``.
    """
    headers = _result_headers(adata, key)
    if category is not None and category not in headers:
        raise ValueError(
            f"assign_category={category!r} is not a category UCD returned for this run; it returned "
            f"{sorted(headers)}. (primary/lines/cancer exist only with split=True.)"
        )
    used = category if category is not None else ("primary" if "primary" in headers else "all")
    if used not in headers:
        raise ValueError(
            f"assign_category is unset, and ucdeconvolve's default category {used!r} is not among the "
            f"categories UCD returned for this run ({sorted(headers)}). Pass assign_category= one of them."
        )
    before = {str(c) for c in adata.obs.columns}
    ucd.utils.assign_top_celltypes(
        adata,
        key=key,
        category=category,
        groupby=groupby,
        inplace=True,
        key_added=pred_key_added,
        knnsmooth_neighbors=knnsmooth_neighbors,
        knnsmooth_cycles=knnsmooth_cycles,
    )
    expected = f"{pred_key_added}_{key}" + (f"_sm{knnsmooth_neighbors}" if knnsmooth_neighbors else "")
    if expected in adata.obs.columns:
        return expected, used
    added = [str(c) for c in adata.obs.columns if str(c) not in before]
    if len(added) == 1:
        return added[0], used
    raise RuntimeError(f"assign_top_celltypes returned without writing obs[{expected!r}] (new obs columns: {added}).")


def _optional_int(value: Any) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return int(value)


def _ignored_knobs(
    input_mode: str,
    coord_type: str,
    assign_top: bool,
    pred_key_added: str,
    assign_category: Any,
    groupby: str,
    knnsmooth_neighbors: Any,
    knnsmooth_cycles: int,
    compute_neighbors: bool,
    n_neighbors: int,
) -> list:
    """``[(names, why), ...]``: knobs set away from their default whose step did not run.

    They stay echoed in ``params`` as they were passed; this says, beside them, that they had no
    effect (``params.ignored`` and a warning), rather than letting the echo read as applied.
    """
    d = POSTPROCESS_DEFAULTS
    groups = []
    if not assign_top:
        names = [
            name
            for name, value in (
                ("pred_key_added", pred_key_added),
                ("assign_category", assign_category),
                ("groupby", groupby),
                ("knnsmooth_neighbors", knnsmooth_neighbors),
                ("knnsmooth_cycles", knnsmooth_cycles),
            )
            if value != d[name]
        ]
        if names:
            groups.append((names, "they shape the top-celltype label, and assign_top_celltypes=False skips that step"))
    elif knnsmooth_neighbors is None and knnsmooth_cycles != d["knnsmooth_cycles"]:
        groups.append(
            (["knnsmooth_cycles"], "it repeats the neighbour smoothing, and knnsmooth_neighbors is unset, so none ran")
        )
    if not compute_neighbors and n_neighbors != d["n_neighbors"]:
        groups.append((["n_neighbors"], "it sets k for compute_neighbors, which is off, so no graph was built"))
    if input_mode not in VISIUM_MODES and coord_type != d["coord_type"]:
        source = "the h5ad's own obsm" if input_mode == "h5ad" else "the x/y columns of coords_csv"
        groups.append(
            (
                ["coord_type"],
                f"it chooses array or pixel positions for the Visium loaders; input_mode={input_mode!r} "
                f"takes obsm['spatial'] from {source}",
            )
        )
    return groups


def run_ucd_base(payload: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()

    output_dir = ensure_dir(str(payload["output_dir"]))
    sample_id = str(payload.get("sample_id") or "sample")

    input_mode = str(payload.get("input_mode", "visium_h5_spatial"))
    coord_type = str(payload.get("coord_type", "array"))
    # The wrapper resolves this in the agent environment; resolve again here because the worker is
    # also run directly (the smoke harness and `python tools/ucdeconvolve_worker.py` both do), and
    # because the cache lives on this box rather than in the payload. `resolve_or_raise` carries
    # the sentence that tells an operator which of the four fixes is cheapest for them.
    token = str(payload.get("token") or "")
    if not token.strip():
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        # ucd_token reads the .env and the cache directory through sog_install, which is under install/.
        sys.path.append(str(Path(__file__).resolve().parents[2] / "install"))
        from tools.ucd_token import resolve_or_raise

        found = resolve_or_raise()
        token = found.token
        print(f"[ucd] token from {found.source} (fingerprint {found.fingerprint})", file=sys.stderr)
    elif payload.get("token_source"):
        print(f"[ucd] token from {payload['token_source']}", file=sys.stderr)

    # UCDBase params
    key_added = str(payload.get("key_added", "ucdbase"))
    split = bool(payload.get("split", True))
    sort = bool(payload.get("sort", True))
    propagate = bool(payload.get("propagate", True))
    use_raw = payload.get("use_raw", True)  # bool or tuple in upstream API
    verbosity = payload.get("verbosity", None)

    # optional post-processing
    assign_top = bool(payload.get("assign_top_celltypes", True))
    assign_category = payload.get("assign_category", None)  # e.g. "raw" or "primary"
    assign_category = (str(assign_category).strip() if assign_category is not None else None) or None
    groupby = str(payload.get("groupby", "") or "").strip()  # optional cluster label (e.g. "leiden")
    pred_key_added = str(payload.get("pred_key_added", "ucd_pred_celltype"))
    # 0/None both mean "no smoothing" upstream (`if knnsmooth_neighbors:`); pass None for either.
    knnsmooth_neighbors = _optional_int(payload.get("knnsmooth_neighbors", None)) or None
    knnsmooth_cycles = int(payload.get("knnsmooth_cycles", 1))

    # optional scanpy neighbors for smoothing
    compute_neighbors = bool(payload.get("compute_neighbors", False))
    n_neighbors = int(payload.get("n_neighbors", 10))

    log(f"[UCD worker] input_mode={input_mode} sample_id={sample_id} split={split} propagate={propagate} sort={sort}")

    # The Visium loaders used to read any coord_type other than "pixel" as "array", so a typo ran on
    # the lattice without a word; it is refused where it is read, and reported as ignored elsewhere.
    if input_mode in VISIUM_MODES and coord_type not in COORD_TYPES:
        raise ValueError(unsupported_choice_msg("coord_type", coord_type, list(COORD_TYPES)))

    # ---- load adata
    if input_mode == "visium_h5_spatial":
        adata = _load_visium_h5_spatial(
            visium_h5_path=str(payload["visium_h5_path"]),
            visium_spatial_dir=str(payload["visium_spatial_dir"]),
            coord_type=coord_type,
        )
    elif input_mode == "spaceranger_outs":
        adata = _load_spaceranger_outs(
            spaceranger_dir=str(payload["spaceranger_dir"]),
            coord_type=coord_type,
        )
    elif input_mode == "h5ad":
        adata = _load_h5ad(str(payload["h5ad_path"]))
    elif input_mode == "generic_counts_coords":
        adata = _load_generic_counts_coords(
            counts_h5ad_path=str(payload["counts_h5ad_path"]),
            coords_csv=str(payload["coords_csv"]),
        )
    else:
        raise ValueError(
            unsupported_choice_msg(
                "input_mode", input_mode, ["visium_h5_spatial", "spaceranger_outs", "h5ad", "generic_counts_coords"]
            )
        )

    # ---- background spots (obs['in_tissue'] == 0) are not tissue, and are never uploaded
    # A CELLxGENE Visium h5ad carries every array spot (56-70% glass on the library's four such
    # slides); a generic coords file or a raw Space Ranger matrix can carry them too. The filtered
    # Space Ranger matrix holds in-tissue spots only, so there this is a no-op.
    adata, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata, "spots")
    if n_spots_off_tissue:
        log(
            f"[UCD worker] left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0 "
            "(background); they are not uploaded"
        )

    # ---- everything that can be checked locally is checked before the upload
    _check_postprocessing_inputs(
        adata, assign_top, groupby, knnsmooth_neighbors, knnsmooth_cycles, compute_neighbors, n_neighbors
    )

    neighbor_graph = None
    if compute_neighbors:
        # Independent of the UCD result, so it runs before the upload: a failure costs no API run.
        # Expression-space neighbours (scanpy on X, via PCA when X is wide) -- not spatial ones.
        import scanpy as sc

        try:
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        except ImportError as exc:
            # ucdenv ships umap-learn without tqdm, which umap imports: this used to be swallowed, and
            # the smoothing that needed the graph then failed silently too.
            missing = getattr(exc, "name", None) or str(exc)
            raise ImportError(
                f"compute_neighbors=True could not build the neighbour graph: scanpy's neighbour backend "
                f"failed to import ({missing}). Install {missing} in the ucdeconvolve environment, or pass "
                "compute_neighbors=False with an input that already carries adata.obsp['distances']."
            ) from exc
        neighbor_graph = f"sc.pp.neighbors(n_neighbors={n_neighbors}) on adata.X (expression space)"
    elif assign_top and knnsmooth_neighbors:
        neighbor_graph = "input adata.obsp['distances']"
    if assign_top and knnsmooth_neighbors:
        _check_neighbour_graph(adata, knnsmooth_neighbors)

    matrix, ucd_var_names, counts_source = _ucd_input_matrix(adata, use_raw)
    scan = _scan_ucd_blocks(matrix)
    if scan["n_nonfinite"] or scan["n_negative"]:
        # ucdeconvolve's preprocessing un-logs (expm1) and total-count normalises: it accepts counts and
        # log1p data by design, but a scaled (z-scored) or NaN-bearing matrix reaches the model as NaN.
        # Refused here, before anything is uploaded.
        what = (
            f"{scan['n_nonfinite']} NaN or infinite values"
            if scan["n_nonfinite"]
            else f"{scan['n_negative']} negative values (scaled or z-scored data?)"
        )
        if counts_source == "adata.X" and getattr(adata, "raw", None) is not None:
            other = " The input carries adata.raw: use_raw=True reads it instead."
        elif counts_source == "adata.raw":
            other = " use_raw=False reads adata.X instead."
        else:
            other = " Supply raw counts, or log1p-normalised data, in X or adata.raw."
        raise ValueError(
            f"{counts_source} holds {what}. ucdeconvolve reads counts or log1p data (it un-logs and "
            "normalises them itself), and this matrix is neither, so it was not uploaded." + other
        )
    n_genes_input = int(len(ucd_var_names))
    n_matched, n_vocabulary, overlap_unknown = _ucd_gene_overlap(ucd_var_names)
    if n_matched == 0:
        if counts_source == "adata.raw":
            other = " use_raw=False reads adata.X instead of adata.raw."
        elif getattr(adata, "raw", None) is not None:
            other = " use_raw=True reads adata.raw instead."
        else:
            other = ""
        shown = [str(v) for v in list(ucd_var_names[:5])]
        raise ValueError(
            f"None of the {n_genes_input} genes in {counts_source} match UCD's {n_vocabulary}-gene input "
            f"vocabulary (first names: {shown}), so UCD would deconvolve an all-zero matrix. var_names must hold "
            "gene symbols or Ensembl ids." + other
        )
    log(
        f"[UCD worker] UCD reads {counts_source}: {int(matrix.shape[0])} x {n_genes_input}, "
        f"integer={scan['integer_valued']} max={scan['max']:.3g}, genes matched={n_matched}/{n_vocabulary}"
    )

    # ---- run UCD
    import ucdeconvolve as ucd

    # Authenticate once; docs show this workflow
    ucd.api.authenticate(token)

    # Execute UCDBase; signature in docs
    ucd.tl.base(
        adata,
        token=token,
        split=split,
        sort=sort,
        propagate=propagate,
        return_results=False,
        key_added=key_added,
        use_raw=use_raw,
        verbosity=verbosity,
    )

    # ---- outputs first, so a failure below still leaves the paid-for predictions on disk
    exported, export_failures = _export_ucd_results(ucd, adata, key_added, output_dir)

    pred_col = None
    category_used = None
    post_error = None
    if assign_top:
        try:
            pred_col, category_used = _assign_top_celltypes(
                ucd,
                adata,
                key=key_added,
                category=assign_category,
                groupby=groupby or None,
                pred_key_added=pred_key_added,
                knnsmooth_neighbors=knnsmooth_neighbors,
                knnsmooth_cycles=knnsmooth_cycles,
            )
        except Exception as e:  # re-raised below, after the annotated h5ad is written
            post_error = e

    out_h5ad = str(Path(output_dir) / "ucdeconvolve_annotated.h5ad")
    _atomic_write_h5ad(adata, out_h5ad)

    if not exported:
        detail = (
            "; ".join(f"{c}: {why}" for c, why in export_failures.items())
            if export_failures
            else f"adata.uns[{key_added!r}] carries no 'headers'"
        )
        raise RuntimeError(
            f"UCD returned no prediction matrix that could be exported ({detail}). "
            f"The annotated h5ad was written to {out_h5ad}."
        )
    if post_error is not None:
        raise RuntimeError(
            f"assign_top_celltypes failed: {type(post_error).__name__}: {post_error}. The UCD predictions were "
            f"written before the failure ({out_h5ad} and {len(exported)} prediction CSV(s) in {output_dir}). "
            "Fix assign_category/groupby/knnsmooth_neighbors, or pass assign_top_celltypes=False."
        ) from post_error

    runtime = round(time.time() - t0, 3)
    categories = [k[: -len("_predictions_csv")] for k in exported]

    out = WorkerOutput("ucdeconvolve", task="deconvolution")
    out.set_data(
        n_spots=int(getattr(adata, "n_obs", adata.shape[0])),
        n_genes=n_genes_input,
        n_spots_supplied=int(n_spots_supplied),
        n_spots_off_tissue_dropped=int(n_spots_off_tissue),
    )
    out.add_output_files({"annotated_h5ad": out_h5ad, **exported})
    out.add_params(
        {
            "sample_id": sample_id,
            "input_mode": input_mode,
            "coord_type": coord_type,
            "key_added": key_added,
            "split": bool(split),
            "propagate": bool(propagate),
            "sort": bool(sort),
            "use_raw": use_raw,
            "verbosity": verbosity,
            "counts_source": counts_source,
            "counts_integer_valued": scan["integer_valued"],
            "counts_max": scan["max"],
            "n_blocks": scan["n_blocks"],
            "n_blocks_read_as_log1p": scan["n_blocks_read_as_log1p"],
            "n_genes_input": n_genes_input,
            "n_genes_matched": n_matched,
            "n_genes_vocabulary": n_vocabulary,
            "prediction_categories": categories,
            "assign_top_celltypes": bool(assign_top),
            "pred_key_added": pred_key_added,
            "assign_category": assign_category,
            "assign_category_used": category_used,
            "groupby": groupby,
            "knnsmooth_neighbors": knnsmooth_neighbors,
            "knnsmooth_cycles": knnsmooth_cycles,
            "compute_neighbors": bool(compute_neighbors),
            "n_neighbors": n_neighbors,
            "neighbor_graph": neighbor_graph,
            "output_dir": output_dir,
        }
    )
    record_method(out, METHOD_NAME)
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    for names, why in _ignored_knobs(
        input_mode,
        coord_type,
        assign_top,
        pred_key_added,
        assign_category,
        groupby,
        knnsmooth_neighbors,
        knnsmooth_cycles,
        compute_neighbors,
        n_neighbors,
    ):
        record_ignored(out, names, why)

    for cat, why in export_failures.items():
        out.add_warning(f"prediction category {cat!r} was returned by UCD but could not be exported: {why}")
    if split:
        missing = [c for c in SPLIT_CATEGORIES if c not in categories]
        if missing:
            out.add_warning(f"split=True, but no {', '.join(missing)} prediction CSV was exported (got {categories}).")
    if overlap_unknown:
        out.add_warning(f"could not compute how many genes match UCD's input vocabulary ({overlap_unknown}).")
    if scan["n_blocks_read_as_log1p"]:
        if scan["integer_valued"]:
            out.add_warning(
                f"{counts_source} holds integer counts, but at least {scan['n_blocks_read_as_log1p']} of "
                f"{scan['n_blocks']} blocks of {UCD_BLOCK_ROWS} spots have a maximum below "
                f"{UCD_LOG_DETECT_MAX:g}; ucdeconvolve's preprocessing reads such a block as log1p data and "
                "applies expm1 to it, so those spots' counts were exponentiated rather than normalised."
            )
        elif scan["n_blocks_read_as_log1p"] < scan["n_blocks"]:
            out.add_warning(
                f"{counts_source} is not integer-valued and UCD treated its blocks of {UCD_BLOCK_ROWS} spots "
                f"differently: at least {scan['n_blocks_read_as_log1p']} of {scan['n_blocks']} were read as "
                f"log1p (maximum below {UCD_LOG_DETECT_MAX:g}, un-logged with expm1) and the rest as counts."
            )

    out.set_summary(
        pred_celltype_key=pred_col,
        runtime_sec=runtime,
        counts_source=counts_source,
        n_genes_matched=n_matched,
        prediction_categories=categories,
    )
    matched = (
        f"{n_matched} of UCD's {n_vocabulary} input genes matched"
        if n_matched is not None
        else "gene overlap with UCD's vocabulary not computed"
    )
    values = "integer counts" if scan["integer_valued"] else "non-integer values"
    analysis = (
        f"UCDBase (UCD cloud API) predicted cell-type fractions for {int(adata.n_obs)} spots in {runtime:.1f}s, "
        f"reading {counts_source} ({n_genes_input} genes, {values}; {matched}). "
        f"Prediction categories exported: {', '.join(categories)}."
    )
    if pred_col:
        analysis += f" Top cell type per {'group of ' + repr(groupby) if groupby else 'spot'} in obs[{pred_col!r}]."
    analysis += describe_reduction(
        "spots",
        n_spots_supplied,
        int(adata.n_obs),
        "leaving out the obs['in_tissue'] == 0 background spots, which were not uploaded",
    )
    out.set_analysis(analysis)
    return out.to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    payload = json.loads(args.json)
    if not isinstance(payload, dict):
        raise ValueError("--json must be a JSON object")
    if payload.get("__tool__") != "ucdeconvolve_base":
        raise ValueError(f"Unsupported __tool__: {payload.get('__tool__')}")

    # stdout carries the JSON only: anything upstream prints (ucdeconvolve's API client, scanpy)
    # goes to stderr while the run is in progress.
    with contextlib.redirect_stdout(sys.stderr):
        result = run_ucd_base(payload)
    emit_json(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[UCD worker] ERROR: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        WorkerOutput.emit_error("ucdeconvolve", str(e), task="deconvolution")
        sys.exit(1)
