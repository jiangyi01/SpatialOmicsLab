#!/usr/bin/env python
"""
XFuse worker script for SpatialOmicsLab MCP integration.

Supports two modes:

  1) --config / --save-path : Run xfuse directly from a pre-built TOML config.
  2) --json <payload>       : End-to-end pipeline: convert h5ad → xfuse h5 → TOML → run.

This script runs *inside* the XFuse environment (/opt/conda/envs/xfuse).

Conventions:
- All progress / debug / error logs go to stderr with a '[xfuse-worker]' prefix.
- stdout has exactly one line of JSON (the result).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from typing import Any

# worker_utils is in the same directory
sys.path.insert(0, os.path.dirname(__file__))
from worker_utils import (
    WorkerOutput,
    choose_counts_matrix,
    describe_reduction,
    expression_matrix_kind,
    identifier_rename_note,
    identifier_rename_params,
    make_names_unique_and_report,
    record_expression_source,
    record_in_tissue,
    record_method,
    spatial_coords,
    unsupported_choice_msg,
)

#: Upstream ``xfuse convert visium`` numbers the spots in an ``int16`` label image
#: (``convert/visium.py``: ``label = np.zeros(image.shape[:2]).astype(np.int16)``, then
#: ``labels_from_spots`` writes 1..n). Spot 32,768 wraps to -32,768 under numpy's silent overflow, so
#: its counts land on the wrong pixels. One slide can therefore carry at most this many spots.
XFUSE_MAX_SPOTS_PER_SLIDE = 32767


def log(msg: str) -> None:
    print(f"[xfuse-worker] {msg}", file=sys.stderr, flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Utility: collect analysis outputs
# ═══════════════════════════════════════════════════════════════════════════


#: A file this run wrote is never older than the run's start by more than this, on a filesystem
#: that stores modification times coarsely.
MTIME_SLACK_S = 2.0


def collect_analyses_outputs(save_path: str, since: float | None = None) -> dict[str, Any]:
    """The files under ``<save_path>/analyses``; with ``since``, only those written at or after it.

    ``save_path`` is reused by a re-run in the same output_dir and nothing clears it, so listing the
    whole tree reported analyses an earlier run left there (the gene maps of a run with
    enable_gene_maps on, say) as this run's (hunt 2026-09-30, u30-uncovered-mcp-14). Earlier files are
    left in place, not listed, and counted in ``n_earlier_files``.
    """
    analyses_dir = os.path.join(save_path, "analyses")
    outputs: dict[str, Any] = {
        "analyses_dir": analyses_dir if os.path.isdir(analyses_dir) else None,
        "tree": {},
        "n_earlier_files": 0,
    }
    if not os.path.isdir(analyses_dir):
        return outputs
    for root, _dirs, files in os.walk(analyses_dir):
        rel_root = os.path.relpath(root, save_path)
        if since is None:
            outputs["tree"].setdefault(rel_root, [])
        for f in sorted(files):
            if since is not None:
                try:
                    earlier = os.path.getmtime(os.path.join(root, f)) < since - MTIME_SLACK_S
                except OSError:
                    earlier = True
                if earlier:
                    outputs["n_earlier_files"] += 1
                    continue
            outputs["tree"].setdefault(rel_root, []).append(f)
    if since is not None and not outputs["tree"]:
        outputs["analyses_dir"] = None
    return outputs


def earlier_analyses_warning(outputs: dict[str, Any], save_path: str) -> str | None:
    """The warning for analyses an earlier run left in ``save_path``, which this run did not write."""
    n = int(outputs.get("n_earlier_files") or 0)
    if not n:
        return None
    return (
        f"{n} file(s) under {os.path.join(save_path, 'analyses')} predate this run (an earlier run in the same "
        "directory wrote them); they are left in place and are not listed in output_files or analyses_tree. Use a "
        "fresh output directory to keep runs apart."
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1: Convert h5ad → xfuse data.h5
# ═══════════════════════════════════════════════════════════════════════════


def _spatial_library(uns_spatial: Any) -> tuple[str | None, dict]:
    """The one library entry of ``uns['spatial']`` that is a mapping: ``(library_id, library)``.

    CELLxGENE exports put a scalar ``is_single`` beside the library dict. h5py hands keys back in
    name order, so on the Muscle, Skin FaceTemple and Thymus samples ``is_single`` came first, the
    converter took it as the library, and ``numpy.bool_.get('images')`` raised AttributeError --
    although each file carries a valid hires image under its real library key. Only mapping-valued
    entries are candidates (the same rule as ``seurat_mcp_server._spatial_library``). None found is
    returned as ``(None, {})`` so the image guard below can say what it looked for; more than one is
    refused by name, because XFuse converts one section per slide and nothing says which library the
    spots belong to.
    """
    from collections.abc import Mapping

    try:
        items = list(uns_spatial.items())
    except AttributeError:
        return None, {}
    libs = [(key, value) for key, value in items if isinstance(value, Mapping)]
    if not libs:
        return None, {}
    if len(libs) > 1:
        raise ValueError(
            f"uns['spatial'] holds {len(libs)} libraries ({sorted(str(k) for k, _ in libs)}); XFuse converts one "
            f"section per slide and this h5ad does not say which library its spots belong to. Pass an h5ad "
            f"that carries one library."
        )
    lib_id, lib = libs[0]
    return str(lib_id), lib


def _in_tissue_mask(adata: Any) -> tuple[Any, bool]:
    """``(mask, column_present)``: which spots ``obs['in_tissue']`` places on tissue.

    Nonzero is on tissue, the rule upstream's own ``--mask`` path applies (``np.where(in_tissue)``).
    A value that is neither a number nor a boolean is refused rather than guessed: the mask decides
    which spots XFuse is fitted on.
    """
    import numpy as np
    import pandas as pd

    if "in_tissue" not in adata.obs:
        return np.ones(adata.n_obs, dtype=bool), False
    column = adata.obs["in_tissue"]
    values = np.asarray(column, dtype=object)
    if values.size and all(isinstance(v, (bool, np.bool_)) for v in values):
        return values.astype(bool), True
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.isna().any():
        spelled = pd.Series(values).astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0})
        numeric = numeric.fillna(spelled)
    bad = numeric.isna().to_numpy()
    if bad.any():
        example = values[np.nonzero(bad)[0][0]]
        raise ValueError(
            f"adata.obs['in_tissue'] must be 0/1 (or boolean): {int(bad.sum())} of {adata.n_obs} values are "
            f"neither (e.g. {example!r}). It decides which spots XFuse models, so it is not guessed."
        )
    return numeric.to_numpy() != 0, True


def _integer_counts(X: Any, round_counts: bool, where: str) -> tuple[Any, int]:
    """Return ``(csr, n_values_rounded)`` for a count matrix XFuse can be fitted on.

    XFuse's likelihood is a negative binomial over counts, and the 10x file the converter writes
    stores ``int32``: a normalised or log value used to be truncated there (0.7 -> 0, 2.9 -> 2) and
    XFuse trained on the result without a word. Every stored value is checked. Negative or
    non-finite values are refused whatever ``round_counts`` says -- rounding cannot turn a scaled
    matrix into counts; non-integer values are refused unless ``round_counts`` allows ``np.rint``,
    which is then reported. Sparse in, sparse out: nothing is densified.
    """
    import numpy as np
    from scipy.sparse import csr_matrix, issparse

    Xc = X.tocsr(copy=True) if issparse(X) else csr_matrix(np.asarray(X))
    Xc.sum_duplicates()
    Xc.sort_indices()
    values = Xc.data
    n_values = int(values.size)
    if values.dtype.kind in ("i", "u", "b"):
        n_negative = int(np.count_nonzero(values < 0)) if values.dtype.kind == "i" else 0
        if n_negative:
            raise ValueError(
                f"adata.X is not counts: {n_negative} of {n_values} stored values are negative. XFuse models raw "
                f"counts, and round_counts cannot repair a scaled matrix. {where}."
            )
        return Xc, 0
    finite = np.isfinite(values)
    n_non_finite = int(n_values - np.count_nonzero(finite))
    n_negative = int(np.count_nonzero(values[finite] < 0))
    if n_non_finite or n_negative:
        raise ValueError(
            f"adata.X is not counts: {n_negative} negative and {n_non_finite} NaN/inf values among {n_values} "
            f"stored values. XFuse models raw counts, and round_counts cannot repair a scaled or corrupted "
            f"matrix. {where}."
        )
    fractional = values != np.rint(values)
    n_bad = int(np.count_nonzero(fractional))
    if n_bad == 0:
        return Xc, 0
    if not round_counts:
        example = float(values[fractional][0])
        raise ValueError(
            f"adata.X is not integer counts: {n_bad} of {n_values} stored values are not whole numbers "
            f"(e.g. {example!r}). XFuse fits a count likelihood and the converter stores integers, so a "
            f"normalised or log matrix would be truncated (0.7 -> 0, 2.9 -> 2) and modelled as if it were "
            f"counts. {where}, or pass round_counts=True to round every value to the nearest integer (np.rint) "
            f"if these are counts stored as near-integer floats -- the rounding is then reported."
        )
    log(f"round_counts=True: rounding {n_bad} of {n_values} non-integer values of X with np.rint.")
    Xc.data = np.rint(values)
    Xc.eliminate_zeros()
    return Xc, n_bad


def _raw_holds_counts(adata: Any) -> bool:
    """True when ``adata.raw`` exists and every stored value of ``adata.raw.X`` is a count."""
    raw = getattr(adata, "raw", None)
    return raw is not None and expression_matrix_kind(raw.X) == "counts"


#: Said beside a refused or rounded X when the counts are in ``adata.raw`` (CELLxGENE's layout).
USE_RAW_COUNTS_HINT = "adata.raw holds raw counts: pass use_raw_counts=True to fit XFuse on them."


def convert_h5ad_to_xfuse_h5(
    st_h5ad: str,
    output_dir: str,
    round_counts: bool = False,
    report: dict | None = None,
    use_raw_counts: bool = False,
) -> str:
    """
    Convert a spatial AnnData h5ad (Visium-style with uns['spatial'] images)
    into the xfuse-native data.h5 format.

    Spots with ``obs['in_tissue'] == 0`` are left out of the count data (the whole image is still
    converted). ``report``, when given, is filled with what the conversion did: the library and
    image it used, the spot counts before and after the tissue mask, the matrix the counts came
    from, and any rounding or renaming.

    ``use_raw_counts`` fits ``adata.raw.X`` instead of ``adata.X`` (obs, obsm and uns kept), through
    ``worker_utils.choose_counts_matrix``: CELLxGENE exports keep a log-normalised or scaled X with
    the integer counts in ``adata.raw`` (the library's Muscle Gastrocnemius and Skin FaceTemple
    samples), and XFuse fits counts. Without it, an X that is not counts is refused as before, and
    the refusal names ``use_raw_counts`` when ``adata.raw`` holds counts.

    Returns the path to the generated data.h5.
    """

    import anndata as ad
    import h5py
    import numpy as np
    import pandas as pd
    from PIL import Image

    if report is None:
        report = {}

    log(f"Loading h5ad: {st_h5ad}")
    adata = ad.read_h5ad(st_h5ad)
    counts_info = None
    if use_raw_counts:
        # Refuses by name when there is no adata.raw or it does not hold counts.
        adata, counts_info = choose_counts_matrix(adata, use_raw_counts=True)
        log(f"use_raw_counts=True: fitting adata.raw.X ({adata.n_obs} x {adata.n_vars}) instead of adata.X")
    n_spots_input = int(adata.n_obs)
    n_genes_input = int(adata.n_vars)
    # Barcodes as well as genes: upstream looks every barcode up in the positions table with
    # ``.loc``, so a duplicated barcode would match two rows and shift every spot after it.
    renamed = make_names_unique_and_report(adata)

    # The spot coordinates are read below by literal name, twice. A file that keeps them somewhere
    # else -- obsm['X_spatial'] after a Seurat conversion, obs['array_row']/['array_col'] straight
    # off Space Ranger -- is ordinary, not malformed, and used to die on anndata's own
    # KeyError('spatial'). main() emits str(e), and str() of a KeyError is the repr of its key, so
    # all the caller got back was the word 'spatial'. Say what was looked for and what is there.
    # ValueError, not KeyError, for the same reason: str(KeyError(msg)) would re-quote the sentence.
    if "spatial" not in adata.obsm:
        raise ValueError(
            f"Spot coordinates not found: adata.obsm['spatial'] is missing. "
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )

    # ── Extract image from uns['spatial'] ────────────────────────────────
    image_array = None
    image_key = None
    images = {}
    hires_sf = None
    spot_diameter_fullres = 100.0
    lib_id = None
    uns_keys: list = []

    # Scale factors the file does not carry, and the value used in their place. Both decide where
    # the spots are drawn on the image (the coordinates are multiplied by the image's scale factor,
    # the spot diameter sets each spot's radius), so an assumed value is reported, not kept quiet.
    scalefactors_assumed: dict = {}

    if "spatial" in adata.uns:
        uns_keys = sorted(str(k) for k in getattr(adata.uns["spatial"], "keys", lambda: [])())
        lib_id, sp_meta = _spatial_library(adata.uns["spatial"])
        images = sp_meta.get("images", {}) or {}
        scalefactors = sp_meta.get("scalefactors", {}) or {}
        if "spot_diameter_fullres" not in scalefactors:
            scalefactors_assumed["spot_diameter_fullres"] = 100.0
        spot_diameter_fullres = scalefactors.get("spot_diameter_fullres", 100.0)

        # Prefer hires, fall back to lowres
        if "hires" in images:
            image_key = "hires"
            image_array = images["hires"]
            if "tissue_hires_scalef" not in scalefactors:
                scalefactors_assumed["tissue_hires_scalef"] = 1.0
            hires_sf = scalefactors.get("tissue_hires_scalef", 1.0)
            log(f"Using hires image of library {lib_id!r}: {image_array.shape}, scale_factor={hires_sf}")
        elif "lowres" in images:
            image_key = "lowres"
            image_array = images["lowres"]
            if "tissue_lowres_scalef" not in scalefactors:
                scalefactors_assumed["tissue_lowres_scalef"] = 1.0
            hires_sf = scalefactors.get("tissue_lowres_scalef", 1.0)
            log(f"Using lowres image of library {lib_id!r}: {image_array.shape}, scale_factor={hires_sf}")

    if image_array is None:
        # This used to build np.ones((H, W, 3)) * 0.9 -- a uniform grey plate sized from the
        # coordinate range -- stage it as tissue_image.png and hand it to xfuse convert. XFuse
        # infers sub-spot expression by fusing the counts with a pixel-resolution H&E image, so a
        # constant image carries no spatial information: every pixel inside a spot gets the same
        # value and the published "super-resolution" is within-spot smoothing of the expression
        # alone. Two ordinary files reach here -- any non-Visium slide, which has no uns['spatial']
        # at all, and a Visium h5ad whose images dict holds neither key. The admission went to
        # stderr, and base_mcp._parse_result drops stderr_tail when the worker exits 0, so it never
        # reached the model. Refuse, in the same shape as the coordinate guard thirty lines above.
        raise ValueError(
            f"Histology image not found: XFuse infers sub-spot expression from the H&E image, so a "
            f"run without one is not super-resolution. Looked for "
            f"adata.uns['spatial'][<library>]['images']['hires'] or ['lowres']. "
            f"uns['spatial'] present: {'spatial' in adata.uns}; entries in it: {uns_keys}; "
            f"library used: {lib_id!r}; image keys found: {sorted(images)}"
        )

    # ── Tissue mask ──────────────────────────────────────────────────────
    # The worker converts with --no-mask, and upstream reads in_tissue only inside ``if mask:``, so
    # writing the column into the positions table used to change nothing: on the CELLxGENE Heart
    # Fetal12W sample 3,009 of 4,992 spots are off tissue and all of them were modelled as tissue.
    # Apply the documented mask here instead, and say how many spots it removed.
    on_tissue, in_tissue_present = _in_tissue_mask(adata)
    n_in_tissue = int(on_tissue.sum())
    n_off_tissue = int(n_spots_input - n_in_tissue)
    if n_in_tissue == 0:
        raise ValueError(
            f"adata.obs['in_tissue'] marks none of the {n_spots_input} spots as on tissue, so there is nothing "
            f"for XFuse to model."
        )
    if n_in_tissue > XFUSE_MAX_SPOTS_PER_SLIDE:
        raise ValueError(
            f"XFuse cannot model this slide: it has {n_in_tissue} on-tissue spots, and upstream 'xfuse convert "
            f"visium' numbers spots in an int16 label image, so one slide holds at most "
            f"{XFUSE_MAX_SPOTS_PER_SLIDE}. Spot numbers above that wrap to negative values and the counts would "
            f"be attached to the wrong pixels."
        )
    if n_off_tissue:
        log(f"obs['in_tissue']: leaving out {n_off_tissue} of {n_spots_input} spots that are off tissue.")
        adata = adata[on_tissue].copy()

    # ── Counts ───────────────────────────────────────────────────────────
    layers = ", ".join(str(k) for k in adata.layers.keys()) or "<none>"
    matrix_name = "adata.raw.X" if use_raw_counts else "adata.X"
    where = (
        f"Put raw counts in adata.X (layers present: [{layers}]; adata.raw: "
        f"{'present' if adata.raw is not None else 'absent'})"
        if not use_raw_counts
        else "use_raw_counts=True fits adata.raw.X, so that is where the raw counts must be"
    )
    round_warning = None
    if round_counts and not use_raw_counts and adata.raw is not None:
        # round_counts is for counts stored as near-integer floats. Rounding a normalised or log X
        # (0.69 -> 1, 1.94 -> 2) makes numbers that look like counts and are not -- while the real
        # counts sit in adata.raw. It still runs (the caller asked for it), and says so.
        if expression_matrix_kind(adata.X) == "nonnegative_noninteger" and _raw_holds_counts(adata):
            round_warning = (
                "round_counts=True rounded an adata.X whose values are not near-integer (normalised or "
                "log-transformed data?), so the rounded values XFuse was fitted on are not counts. "
                + USE_RAW_COUNTS_HINT
            )
    try:
        X_csr, n_values_rounded = _integer_counts(adata.X, bool(round_counts), where)
    except ValueError as exc:
        if use_raw_counts:
            raise ValueError(str(exc).replace("adata.X", matrix_name, 1)) from None
        if _raw_holds_counts(adata):
            raise ValueError(f"{exc} {USE_RAW_COUNTS_HINT}") from None
        raise
    if counts_info is None:
        counts_info = {
            "expression_source": "X",
            "x_matrix_kind": "counts" if not n_values_rounded else "nonnegative_noninteger",
            "warning": round_warning,
        }
    if X_csr.nnz and float(X_csr.data.max()) > np.iinfo(np.int32).max:
        raise ValueError(
            f"adata.X holds a count of {float(X_csr.data.max()):.0f}, above the int32 range the 10x matrix stores."
        )

    # Convert image to uint8 if float
    if image_array.dtype in (np.float32, np.float64):
        img_uint8 = (image_array * 255).clip(0, 255).astype(np.uint8)
    else:
        img_uint8 = image_array

    # ── Prepare intermediate files for xfuse convert ─────────────────────
    staging = os.path.join(output_dir, "_xfuse_staging")
    os.makedirs(staging, exist_ok=True)

    # Image
    image_path = os.path.join(staging, "tissue_image.png")
    Image.fromarray(img_uint8).save(image_path)
    log(f"Saved image: {img_uint8.shape} → {image_path}")

    # Spatial coordinates (scaled to image space)
    coords_fullres, _ = spatial_coords(adata, "spatial", want=2, tool="XFuse")
    coords_img = coords_fullres * hires_sf  # map to image pixel space

    # Tissue positions CSV
    tissue_path = os.path.join(staging, "tissue_positions_list.csv")
    array_row = adata.obs["array_row"].values.astype(int) if "array_row" in adata.obs else np.arange(adata.n_obs)
    array_col = adata.obs["array_col"].values.astype(int) if "array_col" in adata.obs else np.arange(adata.n_obs)
    in_tissue = np.ones(adata.n_obs, dtype=int)  # every spot left is on tissue

    tpos = pd.DataFrame(
        {
            0: adata.obs_names,
            1: in_tissue,
            2: array_row,
            3: array_col,
            4: coords_img[:, 1].astype(int),  # pxl_row = y
            5: coords_img[:, 0].astype(int),  # pxl_col = x
        }
    )
    tpos.to_csv(tissue_path, header=False, index=False)

    # 10x h5 barcode matrix. The 10x layout is CSC over genes x barcodes, which has exactly the
    # arrays of CSR over barcodes x genes -- and upstream reads it back as that CSR
    # (convert/visium.py: csr_matrix((data, indices, indptr), shape=(n_barcodes, n_features))).
    # So the CSR arrays are written as they are; densifying first (the old X.toarray()) cost
    # n_spots x n_genes floats for nothing.
    h5_path = os.path.join(staging, "filtered_feature_bc_matrix.h5")
    index_dtype = np.int32 if X_csr.nnz < np.iinfo(np.int32).max else np.int64
    with h5py.File(h5_path, "w") as f:
        grp = f.create_group("matrix")
        grp.create_dataset("data", data=X_csr.data.astype(np.int32))
        grp.create_dataset("indices", data=X_csr.indices.astype(np.int32))
        grp.create_dataset("indptr", data=X_csr.indptr.astype(index_dtype))
        grp.create_dataset("shape", data=np.array([adata.n_vars, adata.n_obs], dtype=np.int32))
        grp.create_dataset("barcodes", data=np.array(adata.obs_names.values, dtype="S"))
        feat = grp.create_group("features")
        feat.create_dataset("name", data=np.array(adata.var_names.values, dtype="S"))
        feat.create_dataset("id", data=np.array(adata.var_names.values, dtype="S"))
        feat.create_dataset("feature_type", data=np.array(["Gene Expression"] * adata.n_vars, dtype="S"))
        feat.create_dataset("genome", data=np.array(["GRCh38"] * adata.n_vars, dtype="S"))

    # Scale factors JSON (image-space values)
    sf_path = os.path.join(staging, "scalefactors_json.json")
    sf_dict = {
        "tissue_hires_scalef": 1.0,
        "tissue_lowres_scalef": 1.0,
        "fiducial_diameter_fullres": spot_diameter_fullres * hires_sf * 1.5,
        "spot_diameter_fullres": spot_diameter_fullres * hires_sf,
    }
    with open(sf_path, "w") as f:
        json.dump(sf_dict, f)

    report.update(
        {
            "library_id": lib_id,
            "image_used": image_key,
            "n_spots_input": n_spots_input,
            "n_genes_input": n_genes_input,
            "in_tissue_column": bool(in_tissue_present),
            "n_spots_in_tissue": n_in_tissue,
            "n_spots_out_of_tissue_excluded": n_off_tissue,
            "n_values_rounded": int(n_values_rounded),
            "renamed": dict(renamed),
            "counts_info": counts_info,
            "scalefactors_assumed": dict(scalefactors_assumed),
        }
    )

    # ── Run xfuse convert visium ─────────────────────────────────────────
    converted_dir = os.path.join(output_dir, "_xfuse_converted")
    os.makedirs(converted_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "xfuse",
        "convert",
        "visium",
        "--image",
        image_path,
        "--bc-matrix",
        h5_path,
        "--tissue-positions",
        tissue_path,
        "--scale-factors",
        sf_path,
        "--no-mask",
        "--no-rotate",
        "--save-path",
        converted_dir,
    ]
    log(f"Converting: {' '.join(cmd[:8])} ...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stderr:
        for line in proc.stderr.strip().splitlines():
            log(f"  [convert] {line}")
    if proc.returncode != 0:
        raise RuntimeError(f"xfuse convert failed (rc={proc.returncode}): " + (proc.stderr or proc.stdout)[-500:])

    data_h5 = os.path.join(converted_dir, "data.h5")
    if not os.path.isfile(data_h5):
        raise FileNotFoundError(f"Expected {data_h5} after conversion, but not found")

    log(f"Conversion successful: {data_h5}")
    return data_h5


def modelled_sizes(
    data_h5: str,
    gene_regex: str = ".*",
    min_counts: float = 1,
    slide_min_counts: float = 0,
    always_keep: tuple = (),
) -> dict[str, int]:
    """What XFuse will actually fit, read off the converted ``data.h5``.

    ``n_spots`` / ``n_genes`` used to be re-read from the input h5ad after the run, so they ignored
    everything between the input and the model: the spots ``xfuse convert`` drops (a spot drawn
    over by a later one, or outside the cropped image), the gene filter ``xfuse run`` applies
    (``run.py``: summed counts below ``min_counts`` or no ``re.match(gene_regex)``), and the spot
    mask ``STSlide`` builds from the slide's ``min_counts`` (every label whose summed counts fall
    below it, except those in ``always_keep``). This applies the same three rules to the same file.
    """
    import re

    import h5py
    import numpy as np

    with h5py.File(data_h5, "r") as f:
        counts = f["counts"]
        data = np.asarray(counts["data"][()], dtype=float)
        indices = np.asarray(counts["indices"][()])
        indptr = np.asarray(counts["indptr"][()])
        columns = [c.decode() if isinstance(c, bytes) else str(c) for c in counts["columns"][()]]
        n_rows = int(counts["index"].shape[0])

    gene_totals = np.bincount(indices, weights=data, minlength=len(columns)) if data.size else np.zeros(len(columns))
    cumulative = np.concatenate([[0.0], np.cumsum(data)])
    spot_totals = cumulative[indptr[1:]] - cumulative[indptr[:-1]]
    keep = {int(x) for x in always_keep}
    masked = {int(i) + 1 for i in np.nonzero(spot_totals < slide_min_counts)[0]} - keep

    dropped = {g for g, total in zip(columns, gene_totals) if total < min_counts}
    dropped |= {g for g in columns if not re.match(gene_regex, g)}
    return {
        "n_spots_converted": n_rows,
        "n_spots_masked_by_slide_min_counts": len(masked),
        "n_spots_used": n_rows - len(masked),
        "n_genes_converted": len(set(columns)),
        "n_genes_used": len(set(columns) - dropped),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2: Generate TOML configuration
# ═══════════════════════════════════════════════════════════════════════════


def _toml_str(value: Any) -> str:
    """``value`` as a TOML basic string.

    The config used to be written with ``f'gene_regex = "{gene_regex}"'``. A regex is the one value
    here that routinely carries backslashes, and in a TOML basic string ``\\d`` or ``\\w`` is an
    invalid escape (the file no longer parses) while ``\\b`` is a valid one that means backspace (the
    file parses and the regex matches nothing). A double quote ended the string early. Escape the
    backslash, the quote and the control characters TOML forbids; everything else is written as is.
    """
    out = []
    for ch in str(value):
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _toml_key(name: Any) -> str:
    """A table-key segment: bare when TOML allows it, quoted otherwise (``[slides."my slide"]``)."""
    import re

    text = str(name)
    return text if re.fullmatch(r"[A-Za-z0-9_-]+", text) else _toml_str(text)


def _slide_always_keep(data_h5: str) -> tuple[tuple, str]:
    """``(always_keep, basis)`` for one slide of a build_config config, read off its ``data.h5``.

    XFuse's template writes ``always_keep = [1]`` because ``xfuse convert``'s ``--mask`` path
    (``convert/utility.py`` ``mask_tissue``) shifts every spot up by one and puts the area outside the
    tissue under label 1 with zero counts; the entry keeps that background label out of the slide's
    ``min_counts`` spot filter. A ``--no-mask`` conversion -- the one xfuse_spatial_analysis writes --
    has no background label, so label 1 is its first real spot, and ``[1]`` exempted that one spot
    from the filter. build_config wrote ``[1]`` for every slide whatever the file held.

    The file says which it is: label 1 is the first row of ``counts``, and a zero-count label is what
    upstream itself treats as background (``write_data``: ``counts.index[counts.sum(1) == 0]``). A file
    that cannot be read here gets ``()``, the value for the conversion xfuse_spatial_analysis writes,
    and the basis says it was not read.
    """
    if not os.path.isfile(data_h5):
        return (), f"not read ({data_h5} does not exist yet); [] assumes a --no-mask conversion"
    try:
        import h5py
        import numpy as np

        with h5py.File(data_h5, "r") as f:
            counts = f["counts"]
            indptr = counts["indptr"]
            n_rows = int(indptr.shape[0]) - 1
            if n_rows < 1:
                return (), "the slide's counts table holds no label, so there is no label 1 to keep"
            start, stop = (int(v) for v in indptr[0:2])
            first_total = float(np.asarray(counts["data"][start:stop], dtype=float).sum()) if stop > start else 0.0
    except Exception as exc:  # not an XFuse data.h5, or not readable from this process
        return (), f"not read ({data_h5}: {exc}); [] assumes a --no-mask conversion"
    if first_total == 0:
        return (1,), (
            "label 1 has zero counts: the background label 'xfuse convert' writes with its tissue mask, kept "
            "out of min_counts as XFuse's template keeps it"
        )
    return (), (
        f"label 1 is a spot with {first_total:g} counts (a --no-mask conversion, as xfuse_spatial_analysis "
        f"writes), so it is filtered like every other spot"
    )


def _annotation_layers(data_h5: str) -> list[str] | None:
    """Names under ``annotation/`` in an XFuse ``data.h5``; None when the file cannot be read here."""
    if not os.path.isfile(data_h5):
        return None
    try:
        import h5py

        with h5py.File(data_h5, "r") as f:
            group = f.get("annotation")
            return sorted(str(k) for k in group.keys()) if group is not None else []
    except Exception:  # not an XFuse file, or not readable from this process
        return None


#: Why the end-to-end pipeline refuses ``enable_prediction`` and build_config warns about it.
PREDICTION_NEEDS_A_LAYER = (
    "XFuse's prediction analysis sums the predicted expression over the regions of a named annotation layer "
    "stored in the slide's data.h5 (upstream analyze/prediction.py -> STSlide.annotation), and fails with "
    "'Annotation layer \"<name>\" is missing' when that layer is absent -- after the whole training run, "
    "because analyses run once training has finished."
)


#: Why the gene maps of a --no-mask conversion are not masked to tissue, said wherever they are enabled.
GENE_MAPS_UNMASKED = (
    "XFuse masks its gene maps with the slide's zero-count labels (upstream analyze/gene_maps.py), and only an "
    "'xfuse convert --mask' conversion writes the background as one; this pipeline converts with --no-mask (the "
    "spots come from obs['in_tissue'], not from XFuse's image-based tissue detection), so the maps cover the "
    "whole image, background included, and the config says mask_tissue = false."
)


def generate_toml_config(
    data_h5_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 1,
    patch_size: int = 512,
    network_depth: int = 5,
    network_width: int = 16,
    learning_rate: float = 3e-4,
    gene_regex: str = ".*",
    min_counts: int = 1,
    slide_name: str = "section0",
    slide_min_counts: int = 0,
    enable_metagenes: bool = True,
    enable_prediction: bool = False,
    enable_gene_maps: bool = False,
    always_keep: tuple = (),
) -> str:
    """Generate a TOML config file. Returns the path to the config.

    ``always_keep`` is empty here, unlike XFuse's template's ``[1]``: that entry exempts label 1
    from the slide's spot filter because ``xfuse convert``'s ``--mask`` path makes label 1 the
    zero-count background. This pipeline converts with ``--no-mask``, so label 1 is an ordinary
    spot, and ``[1]`` exempted the first spot of every slide from ``slide_min_counts``.

    The gene-maps section says ``mask_tissue = false`` for the same reason. Upstream's gene-map
    tissue mask (``analyze/gene_maps.py``) is every pixel whose label is not a zero-count label, and
    only a ``--mask`` conversion writes one for the background; under ``--no-mask`` the background
    is label 0, which that rule keeps, so ``true`` masked nothing but the odd zero-count spot while
    the config said the maps were masked to tissue. See ``GENE_MAPS_UNMASKED``.
    """
    # data path must be relative to the config file location for xfuse
    config_dir = output_dir
    os.makedirs(config_dir, exist_ok=True)
    data_rel = os.path.relpath(data_h5_path, config_dir)

    lines = _render_config_lines(
        slides=[(slide_name, data_rel, slide_min_counts)],
        epochs=epochs,
        batch_size=batch_size,
        patch_size=patch_size,
        network_depth=network_depth,
        network_width=network_width,
        learning_rate=learning_rate,
        gene_regex=gene_regex,
        min_counts=min_counts,
        enable_metagenes=enable_metagenes,
        enable_prediction=enable_prediction,
        enable_gene_maps=enable_gene_maps,
        always_keep=always_keep,
        gene_maps_mask_tissue=False,
    )

    config_path = os.path.join(config_dir, "xfuse_config.toml")
    _write_text_atomic(config_path, "\n".join(lines))

    log(f"Generated config: {config_path}")
    return config_path


def _write_text_atomic(path: str, text: str) -> None:
    """Write ``text`` to ``path`` through ``<path>.partial`` + ``os.replace``."""
    partial = path + ".partial"
    with open(partial, "w") as fh:
        fh.write(text)
    os.replace(partial, path)


def _render_config_lines(
    slides: list[tuple[str, str, int]],
    epochs: int = 50,
    batch_size: int = 1,
    patch_size: int = 512,
    network_depth: int = 5,
    network_width: int = 16,
    learning_rate: float = 3e-4,
    gene_regex: str = ".*",
    min_counts: int = 1,
    enable_metagenes: bool = True,
    enable_prediction: bool = False,
    enable_gene_maps: bool = False,
    annotation_layer: str = "",
    always_keep: tuple = (),
    gene_maps_mask_tissue: bool = True,
) -> list[str]:
    """The TOML body, as lines. ``slides`` is a list of (name, data path, min_counts), each optionally
    followed by that slide's own ``always_keep`` labels.

    ``gene_maps_mask_tissue`` is the gene-maps ``mask_tissue`` option: XFuse's own default (true)
    for build_config, whose slides may be ``--mask`` conversions that carry a background label, and
    false for the end-to-end pipeline, whose ``--no-mask`` conversion carries none.

    Shared by the end-to-end pipeline, which always has exactly one slide, and by build_config,
    which takes however many the caller passes. Each slide gets its index as its ``section``
    covariate -- that is what xfuse uses to tell slides apart, so a shared value would merge them.
    Every string value goes through ``_toml_str`` and every slide name through ``_toml_key``.
    ``always_keep`` applies to a slide that does not carry its own; it defaults to none (see
    ``_slide_always_keep`` for why XFuse's template ``[1]`` is right only for a masked conversion).
    """
    lines = []
    lines.append("[xfuse]")
    lines.append(f"network_depth = {network_depth}")
    lines.append(f"network_width = {network_width}")
    lines.append(f"gene_regex = {_toml_str(gene_regex)}")
    lines.append(f"min_counts = {min_counts}")
    lines.append("")
    lines.append("[settings]")
    lines.append("cache_data = true")
    lines.append("data_workers = 0")
    lines.append("")
    lines.append("[expansion_strategy]")
    lines.append('type = "DropAndSplit"')
    lines.append("purge_interval = 1000")
    lines.append("")
    lines.append("[expansion_strategy.DropAndSplit]")
    lines.append("max_metagenes = 50")
    lines.append("")
    lines.append("[optimization]")
    lines.append(f"batch_size = {batch_size}")
    lines.append(f"epochs = {epochs}")
    lines.append(f"learning_rate = {learning_rate}")
    lines.append(f"patch_size = {patch_size}")
    lines.append("")
    lines.append("[analyses]")

    if enable_metagenes:
        lines.append("")
        lines.append("[analyses.analysis-metagenes]")
        lines.append('type = "metagenes"')
        lines.append("[analyses.analysis-metagenes.options]")
        lines.append('method = "pca"')

    if enable_prediction:
        lines.append("")
        lines.append("[analyses.analysis-prediction]")
        lines.append('type = "prediction"')
        lines.append("[analyses.analysis-prediction.options]")
        lines.append(f"annotation_layer = {_toml_str(annotation_layer)}")
        lines.append("num_samples = 1")
        lines.append("genes_per_batch = 10")
        lines.append("predict_mean = true")

    if enable_gene_maps:
        lines.append("")
        lines.append("[analyses.analysis-gene-maps]")
        lines.append('type = "gene_maps"')
        lines.append("[analyses.analysis-gene-maps.options]")
        lines.append('gene_regex = ".*"')
        lines.append("num_samples = 10")
        lines.append("genes_per_batch = 10")
        lines.append("predict_mean = true")
        lines.append(f"mask_tissue = {'true' if gene_maps_mask_tissue else 'false'}")
        lines.append("scale = 1.0")
        lines.append('writer = "image"')

    lines.append("")
    lines.append("[slides]")
    for index, slide in enumerate(slides):
        name, data_rel, slide_min_counts = slide[0], slide[1], slide[2]
        slide_keep = slide[3] if len(slide) > 3 else always_keep
        keep = ", ".join(str(int(x)) for x in slide_keep)
        key = _toml_key(name)
        lines.append("")
        lines.append(f"[slides.{key}]")
        lines.append(f"data = {_toml_str(data_rel)}")
        lines.append(f"[slides.{key}.covariates]")
        lines.append(f'section = "{index}"')
        lines.append(f"[slides.{key}.options]")
        lines.append(f"min_counts = {slide_min_counts}")
        lines.append("always_filter = []")
        lines.append(f"always_keep = [{keep}]")
    lines.append("")

    return lines


def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


def build_config(payload: dict[str, Any]) -> None:
    """Write a TOML config from slide descriptions alone -- no data conversion, no training.

    The output of this is the input to ``xfuse_run``, and nothing else in the system produces one.
    """
    output_config_path = payload.get("output_config_path")
    if not output_config_path:
        raise ValueError("build_config requires output_config_path (the file to write).")

    raw_slides = payload.get("slides")
    if not raw_slides:
        raise ValueError("build_config requires slides: a non-empty list of {'name': ..., 'data': ...} entries.")
    if not isinstance(raw_slides, list):
        raise ValueError(
            f"slides must be a list of {{'name': ..., 'data': ...}} entries, got {type(raw_slides).__name__}."
        )

    config_dir = os.path.dirname(os.path.abspath(output_config_path))
    os.makedirs(config_dir, exist_ok=True)

    slides: list[tuple[str, str, int, tuple]] = []
    keep_basis: dict[str, str] = {}
    unread: list[str] = []
    for position, slide in enumerate(raw_slides):
        if not isinstance(slide, dict):
            raise ValueError(f"slides[{position}] must be a dict with 'name' and 'data', got {type(slide).__name__}.")
        missing = [key for key in ("name", "data") if not slide.get(key)]
        if missing:
            raise ValueError(f"slides[{position}] is missing {', '.join(missing)}; each slide needs 'name' and 'data'.")
        # xfuse resolves slide data paths relative to the config file, so store them that way. A
        # relative 'data' that names a file from the working directory -- as output_config_path is
        # read, and as xfuse_spatial_analysis reports data_h5 -- is that file. Kept verbatim it was
        # re-read against the config's directory, and xfuse_run failed on a missing slide
        # (hunt 2026-09-30, u30-uncovered-mcp-7). Otherwise it stays as written, config-relative.
        data = os.path.expanduser(str(slide["data"]))
        if not os.path.isabs(data) and os.path.exists(data):
            # Both readings name a file, and different ones: which one was meant cannot be told, so
            # it is asked for rather than the working directory's silently winning (hunt 2026-09-30,
            # rp-u30 xfuse minor).
            beside_config = os.path.join(config_dir, data)
            if os.path.exists(beside_config) and not _same_file(data, beside_config):
                raise ValueError(
                    f"slides[{position}] data {slide['data']!r} names two different files: "
                    f"{os.path.abspath(data)} (from the working directory) and "
                    f"{os.path.normpath(beside_config)} (from the config's directory). Pass the absolute "
                    "path of the one you mean."
                )
            data = os.path.abspath(data)
        data_rel = os.path.relpath(data, config_dir) if os.path.isabs(data) else data
        name = str(slide["name"])
        slide_keep, basis = _slide_always_keep(os.path.normpath(os.path.join(config_dir, data_rel)))
        keep_basis[name] = basis
        if basis.startswith("not read"):
            unread.append(name)
        slides.append((name, data_rel, int(slide.get("min_counts", 0)), slide_keep))

    enable_prediction = bool(payload.get("enable_prediction", False))
    enable_gene_maps = bool(payload.get("enable_gene_maps", False))
    annotation_layer = str(payload.get("annotation_layer") or "")
    warnings: list[str] = []
    if enable_gene_maps:
        # mask_tissue = true (XFuse's default) masks the zero-count background a --mask conversion
        # writes under label 1; a slide whose label 1 is a spot has no such label to mask.
        unmasked = [name for name, basis in keep_basis.items() if basis.startswith("label 1 is a spot")]
        if unmasked:
            warnings.append(
                f"enable_gene_maps=True, but slide(s) {', '.join(unmasked)} carry no zero-count background label "
                "(label 1 is a spot: a --no-mask conversion, as xfuse_spatial_analysis writes), so XFuse's "
                "mask_tissue has nothing to mask there and their gene maps cover the whole image."
            )
    if unread:
        warnings.append(
            f"{len(unread)} slide data file(s) could not be read here ({', '.join(unread)}), so their always_keep "
            "is written as [] -- right for the --no-mask conversion xfuse_spatial_analysis writes, where label 1 is "
            "a spot. A slide converted by 'xfuse convert' with its default tissue mask carries a zero-count "
            "background under label 1, which XFuse's template keeps with always_keep = [1]."
        )
    if enable_prediction:
        # A slide file that is already on disk can be asked whether it has the layer; a missing layer
        # there is a certain failure after training, so it is refused now. A path that is not there
        # yet cannot be checked, and an empty name can never match -- both are said out loud.
        for name, data_rel, _, _ in slides:
            layers = _annotation_layers(os.path.join(config_dir, data_rel))
            if layers is not None and annotation_layer not in layers:
                raise ValueError(
                    f"enable_prediction=True with annotation_layer={annotation_layer!r}, but slide {name!r} "
                    f"({data_rel}) carries annotation layers {layers}. {PREDICTION_NEEDS_A_LAYER} Name one of "
                    f"its layers in annotation_layer, or leave enable_prediction off."
                )
        if not annotation_layer:
            warnings.append(
                "enable_prediction=True with no annotation_layer: the config's prediction section names the "
                f"layer '' and no data.h5 can carry a layer with an empty name. {PREDICTION_NEEDS_A_LAYER} "
                "Set annotation_layer to a layer that every slide's data.h5 carries (xfuse convert "
                "--annotation writes them; the h5ad conversion in xfuse_spatial_analysis writes none), or "
                "turn enable_prediction off."
            )

    lines = _render_config_lines(
        slides=slides,
        epochs=int(payload.get("epochs", 50)),
        batch_size=int(payload.get("batch_size", 1)),
        patch_size=int(payload.get("patch_size", 512)),
        network_depth=int(payload.get("network_depth", 5)),
        network_width=int(payload.get("network_width", 16)),
        learning_rate=float(payload.get("learning_rate", 3e-4)),
        gene_regex=str(payload.get("gene_regex", ".*")),
        min_counts=int(payload.get("min_counts", 1)),
        enable_metagenes=bool(payload.get("enable_metagenes", True)),
        enable_prediction=enable_prediction,
        enable_gene_maps=enable_gene_maps,
        annotation_layer=annotation_layer,
    )
    _write_text_atomic(output_config_path, "\n".join(lines))
    log(f"Generated config: {output_config_path}")

    out = WorkerOutput("xfuse", task="build_config")
    out.add_output_file("config_toml", output_config_path)
    out.add_params(
        {
            "slides": [slide[0] for slide in slides],
            "epochs": int(payload.get("epochs", 50)),
            # Per slide: the labels exempted from its min_counts spot filter, and why.
            "always_keep": {slide[0]: [int(x) for x in slide[3]] for slide in slides},
            "always_keep_basis": keep_basis,
        }
    )
    if enable_prediction:
        out.add_params({"enable_prediction": True, "annotation_layer": annotation_layer})
    for message in warnings:
        out.add_warning(message)
    analysis = (
        f"Wrote an XFuse TOML config for {len(slides)} slide(s) to {output_config_path}. "
        f"Pass it to xfuse_run as config_path to train."
    )
    if warnings:
        analysis += " WARNING: " + " ".join(warnings)
    out.set_analysis(analysis)
    out.emit()


# ═══════════════════════════════════════════════════════════════════════════
#  Step 3: Run xfuse training
# ═══════════════════════════════════════════════════════════════════════════


def run_xfuse(config_path: str, save_path: str, session_path: str | None = None) -> int:
    """Run xfuse training. Returns the process return code."""
    cmd = [sys.executable, "-m", "xfuse", "run", config_path, "--save-path", save_path]
    if session_path:
        cmd.extend(["--session", session_path])

    log(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.stdout:
        for line in proc.stdout.strip().splitlines():
            log(f"[xfuse stdout] {line}")
    if proc.stderr:
        for line in proc.stderr.strip().splitlines()[-30:]:
            log(f"[xfuse stderr] {line}")

    log(f"XFuse finished with return code {proc.returncode}")
    return proc.returncode


# ═══════════════════════════════════════════════════════════════════════════
#  Mode A: --config / --save-path (legacy direct run)
# ═══════════════════════════════════════════════════════════════════════════


def run_direct(args):
    config_path = os.path.abspath(args.config)
    save_path = os.path.abspath(args.save_path)
    session_path = args.session

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    os.makedirs(save_path, exist_ok=True)
    t_train = time.time()
    rc = run_xfuse(config_path, save_path, session_path)

    # Only this run's analyses: a reused save_path keeps an earlier run's (u30-uncovered-mcp-14).
    outputs = collect_analyses_outputs(save_path, since=t_train)
    if rc != 0:
        WorkerOutput.emit_error("xfuse", f"xfuse run exited with code {rc}", task="run")
        sys.exit(rc)

    out = WorkerOutput("xfuse", task="run")
    out.add_params({"config_path": config_path, "save_path": save_path, "session_path": session_path})
    if outputs.get("analyses_dir"):
        out.add_output_file("analyses_dir", outputs["analyses_dir"])
    earlier = earlier_analyses_warning(outputs, save_path)
    if earlier:
        out.add_warning(earlier)
    out.set_summary(return_code=rc, analyses_tree=outputs.get("tree", {}))
    out.set_analysis(f"XFuse run completed successfully. Results in {save_path}.")
    out.emit()


# ═══════════════════════════════════════════════════════════════════════════
#  Mode B: --json (end-to-end pipeline from h5ad)
# ═══════════════════════════════════════════════════════════════════════════


def run_pipeline(payload: dict[str, Any]):
    import re

    t0 = time.time()

    st_h5ad = payload["st_h5ad"]
    # Absolute, so the data_h5 / config_toml / save_path this reports name the same files from any
    # working directory -- a relative data_h5 handed on to xfuse_build_config with a config elsewhere
    # pointed XFuse at a file that is not there (hunt 2026-09-30, u30-uncovered-mcp-7).
    output_dir = os.path.abspath(os.path.expanduser(payload["output_dir"]))

    epochs = int(payload.get("epochs", 50))
    batch_size = int(payload.get("batch_size", 1))
    patch_size = int(payload.get("patch_size", 512))
    network_depth = int(payload.get("network_depth", 5))
    network_width = int(payload.get("network_width", 16))
    learning_rate = float(payload.get("learning_rate", 3e-4))
    gene_regex = str(payload.get("gene_regex", ".*"))
    min_counts = int(payload.get("min_counts", 1))
    slide_name = str(payload.get("slide_name", "section0"))
    slide_min_counts = int(payload.get("slide_min_counts", 0))
    enable_metagenes = bool(payload.get("enable_metagenes", True))
    enable_prediction = bool(payload.get("enable_prediction", False))
    enable_gene_maps = bool(payload.get("enable_gene_maps", False))
    session_path = payload.get("session_path")
    round_counts = bool(payload.get("round_counts", False))
    use_raw_counts = bool(payload.get("use_raw_counts", False))

    # Refused before anything is converted or trained. The prediction section used to be written
    # with annotation_layer = "", the conversion below never writes an annotation layer, and upstream
    # looks the layer up by name -- so every enable_prediction=True run trained to the end and then
    # died on 'Annotation layer "" is missing', throwing the training away with it.
    if enable_prediction:
        raise ValueError(
            "enable_prediction=True cannot run on this path. "
            + PREDICTION_NEEDS_A_LAYER
            + " The h5ad conversion here writes no annotation layer, so the run would train and then fail. "
            "Leave enable_prediction off (enable_metagenes and enable_gene_maps still run). To predict per "
            "annotated region, convert with 'xfuse convert visium --annotation', build the config with "
            "xfuse_build_config(annotation_layer=...) and train with xfuse_run."
        )
    try:
        re.compile(gene_regex)
    except re.error as exc:
        raise ValueError(f"gene_regex={gene_regex!r} is not a valid regular expression: {exc}") from exc

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(st_h5ad):
        raise FileNotFoundError(f"Input h5ad not found: {st_h5ad}")

    # ── Convert ──────────────────────────────────────────────────────────
    log("=== Step 1: Converting h5ad to xfuse format ===")
    conversion: dict[str, Any] = {}
    data_h5 = convert_h5ad_to_xfuse_h5(
        st_h5ad, output_dir, round_counts=round_counts, report=conversion, use_raw_counts=use_raw_counts
    )

    # What the model will be fitted on, from the converted file and this run's own filters. A
    # resumed session brings its own gene list (upstream run.py reads it from the session before it
    # would filter), so the gene count is only this config's when training starts fresh.
    sizes: dict[str, Any] = {}
    size_warnings: list[str] = []
    try:
        sizes = modelled_sizes(data_h5, gene_regex, min_counts, slide_min_counts, always_keep=())
    except Exception as exc:  # the sizes are a report; the run itself does not depend on them
        size_warnings.append(f"could not read the modelled spot/gene counts from {data_h5}: {exc}")
    if sizes and not session_path:
        if sizes["n_genes_used"] == 0:
            raise ValueError(
                f"No gene survives XFuse's gene filter: of {sizes['n_genes_converted']} genes in the converted "
                f"data, none has summed counts >= min_counts={min_counts} and matches gene_regex={gene_regex!r}. "
                f"Loosen gene_regex or min_counts."
            )
        if sizes["n_spots_used"] == 0:
            raise ValueError(
                f"No spot survives XFuse's spot filter: all {sizes['n_spots_converted']} converted spots have summed "
                f"counts below slide_min_counts={slide_min_counts}. Lower slide_min_counts."
            )

    # ── Config ───────────────────────────────────────────────────────────
    log("=== Step 2: Generating TOML config ===")
    config_path = generate_toml_config(
        data_h5_path=data_h5,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        patch_size=patch_size,
        network_depth=network_depth,
        network_width=network_width,
        learning_rate=learning_rate,
        gene_regex=gene_regex,
        min_counts=min_counts,
        slide_name=slide_name,
        slide_min_counts=slide_min_counts,
        enable_metagenes=enable_metagenes,
        enable_prediction=enable_prediction,
        enable_gene_maps=enable_gene_maps,
        always_keep=(),
    )

    # ── Train ────────────────────────────────────────────────────────────
    log("=== Step 3: Running xfuse training ===")
    save_path = os.path.join(output_dir, "results")
    os.makedirs(save_path, exist_ok=True)
    t_train = time.time()
    rc = run_xfuse(config_path, save_path, session_path)

    elapsed = time.time() - t0
    outputs = collect_analyses_outputs(save_path, since=t_train)

    if rc != 0:
        WorkerOutput.emit_error("xfuse", f"xfuse run exited with code {rc}", task="spatial_super_resolution")
        sys.exit(rc)

    # ── Output ───────────────────────────────────────────────────────────
    # n_spots / n_genes are what was supplied (as before); the *_used counts are what XFuse was
    # fitted on, after the tissue mask, the conversion, the gene filter and the spot filter.
    out = WorkerOutput("xfuse", task="spatial_super_resolution")
    n_spots_input = conversion.get("n_spots_input")
    n_genes_input = conversion.get("n_genes_input")
    data_kwargs: dict[str, Any] = {
        "n_spots": n_spots_input,
        "n_genes": n_genes_input,
        "n_spots_in_tissue": conversion.get("n_spots_in_tissue"),
        "n_spots_out_of_tissue_excluded": conversion.get("n_spots_out_of_tissue_excluded"),
    }
    if sizes:
        data_kwargs.update(
            {
                "n_spots_converted": sizes["n_spots_converted"],
                "n_spots_masked_by_slide_min_counts": sizes["n_spots_masked_by_slide_min_counts"],
                "n_spots_used": sizes["n_spots_used"],
                # A resumed session fits the gene list stored in it, which this run did not choose.
                "n_genes_used": None if session_path else sizes["n_genes_used"],
            }
        )
    out.set_data(**data_kwargs)
    out.add_output_files(
        {
            "config_toml": config_path,
            "data_h5": data_h5,
            "save_path": save_path,
        }
    )
    if outputs.get("analyses_dir"):
        out.add_output_file("analyses_dir", outputs["analyses_dir"])

    record_method(out, "XFuse (xfuse convert visium --no-mask, then xfuse run)")
    out.add_params(
        {
            "st_h5ad": st_h5ad,
            "epochs": epochs,
            "batch_size": batch_size,
            "patch_size": patch_size,
            "network_depth": network_depth,
            "network_width": network_width,
            "learning_rate": learning_rate,
            "gene_regex": gene_regex,
            "min_counts": min_counts,
            "slide_min_counts": slide_min_counts,
            "enable_metagenes": enable_metagenes,
            "enable_prediction": enable_prediction,
            "enable_gene_maps": enable_gene_maps,
            "session_path": session_path,
            "round_counts": round_counts,
            "use_raw_counts": use_raw_counts,
            "rounded_to_integers": bool(conversion.get("n_values_rounded")),
            "n_values_rounded": int(conversion.get("n_values_rounded") or 0),
            "library_id": conversion.get("library_id"),
            "image_used": conversion.get("image_used"),
            "tissue_mask": "obs['in_tissue']" if conversion.get("in_tissue_column") else "none (no in_tissue column)",
        }
    )
    if enable_gene_maps:
        # The conversion is --no-mask, so there is no background label for XFuse to mask the maps with.
        out.add_params({"gene_maps_masked": False})
    if conversion.get("scalefactors_assumed"):
        out.add_params({"scalefactors_assumed": conversion["scalefactors_assumed"]})
    out.add_params(identifier_rename_params(conversion.get("renamed")))
    if conversion.get("counts_info"):
        # params.expression_source ('X' or 'raw.X') and params.x_matrix_kind, the fleet's keys.
        record_expression_source(out, conversion["counts_info"])
    # params.in_tissue_filter, the key every spot-level worker reports the background spots under.
    record_in_tissue(out, int(n_spots_input or 0), int(conversion.get("n_spots_out_of_tissue_excluded") or 0))

    notes: list[str] = []  # analysis prose
    cautions: list[str] = []  # the same facts where a reader of warnings looks
    if conversion.get("n_spots_out_of_tissue_excluded"):
        notes.append(
            f"{conversion['n_spots_out_of_tissue_excluded']} of {n_spots_input} spots have obs['in_tissue'] == 0 "
            f"and were left out of the count data."
        )
    if use_raw_counts:
        notes.append("The counts are adata.raw.X (use_raw_counts=True); adata.X was not used.")
    if conversion.get("n_values_rounded"):
        cautions.append(
            f"round_counts=True: {conversion['n_values_rounded']} non-integer values of "
            f"{'adata.raw.X' if use_raw_counts else 'X'} were rounded to the nearest integer (np.rint) before "
            f"conversion."
        )
    if enable_gene_maps:
        cautions.append("The gene maps are not masked to tissue: " + GENE_MAPS_UNMASKED)
    if conversion.get("scalefactors_assumed"):
        missing = sorted(conversion["scalefactors_assumed"])
        assumed = ", ".join(f"{k}={conversion['scalefactors_assumed'][k]:g}" for k in missing)
        cautions.append(
            f"uns['spatial'] of library {conversion.get('library_id')!r} carries no {', '.join(missing)}; the "
            f"conversion assumed {assumed}, which sets where (and how large) the spots are drawn on the image."
        )
    if sizes and n_spots_input:
        cautions.append(
            describe_reduction(
                "spots",
                int(n_spots_input),
                int(sizes["n_spots_used"]),
                "the obs['in_tissue'] mask, xfuse convert and slide_min_counts",
            ).strip()
        )
    if sizes and n_genes_input and not session_path:
        cautions.append(
            describe_reduction(
                "genes",
                int(n_genes_input),
                int(sizes["n_genes_used"]),
                f"XFuse's gene filter (min_counts={min_counts}, gene_regex={gene_regex!r})",
            ).strip()
        )
    cautions.append(earlier_analyses_warning(outputs, save_path) or "")
    cautions = [c for c in cautions if c]
    out.add_warnings(size_warnings)
    out.add_warnings(cautions)

    out.set_summary(
        return_code=rc,
        runtime_sec=round(elapsed, 1),
        analyses_tree=outputs.get("tree", {}),
    )
    if sizes:
        fitted = (
            f"XFuse was fitted on {sizes['n_spots_used']} spots"
            + (
                f" x {sizes['n_genes_used']} genes"
                if not session_path
                else " (genes: the list stored in the resumed session)"
            )
            + f" of the {n_spots_input} x {n_genes_input} supplied."
        )
    else:
        fitted = f"{n_spots_input} spots x {n_genes_input} genes were supplied; the modelled counts could not be read."
    out.set_analysis(
        f"XFuse spatial super-resolution pipeline completed successfully in {elapsed:.0f}s. {fitted} "
        + " ".join(notes + cautions)
        + identifier_rename_note(conversion.get("renamed"))
        + f" Results in {save_path}."
    )
    out.emit()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="SpatialOmicsLab XFuse worker.")
    parser.add_argument("--config", help="Path to XFuse TOML config (direct mode).")
    parser.add_argument("--save-path", help="Output directory (direct mode).")
    parser.add_argument("--session", default=None, help="Resume from .session file.")
    parser.add_argument("--json", default=None, help="JSON payload for end-to-end pipeline mode.")
    args = parser.parse_args()

    try:
        if args.json:
            payload = json.loads(args.json)
            if not isinstance(payload, dict):
                raise ValueError("--json must be a JSON object")
            # The portal sends `action` to pick a job; without this the build_config call landed in
            # run_pipeline and died on st_h5ad, a parameter that tool does not have.
            action = payload.get("action") or "spatial_analysis"
            if action == "build_config":
                build_config(payload)
            elif action == "spatial_analysis":
                run_pipeline(payload)
            else:
                raise ValueError(unsupported_choice_msg("action", action, ["spatial_analysis", "build_config"]))
        elif args.config and args.save_path:
            run_direct(args)
        else:
            parser.error("Provide either --json <payload> or --config + --save-path")
    except Exception as e:
        log(f"EXCEPTION: {e}")
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("xfuse", str(e), task="run")
        sys.exit(1)


if __name__ == "__main__":
    main()
