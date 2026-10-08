#!/usr/bin/env python3
"""
BayesTME worker (runs in /opt/conda/envs/bayestme)

Rules:
- Input via --json
- stdout: JSON ONLY (final result)
- stderr: logs / progress / errors

Supports multiple ST input layouts:
- spaceranger_outs
- visium_h5_spatial (h5 + spatial/tissue_positions_list.csv)
- h5ad
- generic_counts_coords (counts + coords CSV)

What runs is BayesTME's SVI deconvolution (``bayestme.svi.deconvolution.deconvolve``), then,
optionally, its ``select_marker_genes`` CLI. Three facts about it the payload states, because none
of them can be read off the parameters:

- BayesTME builds the adjacency its spatial prior smooths over by itself, from the in-tissue
  positions and ``uns['layout']`` (HEX/SQUARE: lattice neighbours; IRREGULAR: a fixed 5-nearest-
  neighbour graph). It never reads ``obsp['connectivities']``, so ``knn_k`` shapes only that saved
  graph and is listed in ``params.ignored``.
- Only in-tissue spots are deconvolved. Out-of-tissue rows of the published obsm matrices are zeros,
  and ``data.n_spots_used`` counts the spots the model saw.
- ``deconvolution_result.h5`` leaves out the derived ``reads_trace`` (samples x spots x genes x
  components, hundreds of GB on a whole-transcriptome Visium slide). ``DeconvolutionResult.read_h5``
  never reads it, and its ``reads_trace`` property recomputes it from the four traces that are stored.
- BayesTME's likelihood is ``pyro`` ``Poisson(obs=counts)`` with validation on, so the matrix it fits
  must hold non-negative whole numbers. An h5ad whose X is log-normalised or scaled (CELLxGENE Visium
  exports keep the counts in ``adata.raw``) is refused before the SVI run, by name, instead of dying
  on pyro's support check; ``use_raw_counts=True`` fits ``adata.raw`` instead (``params.counts_source``).
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import re
import sys
import time
import traceback
from typing import Any

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from worker_utils import (
    TISSUE_POSITIONS_NAMES,
    WorkerOutput,
    env_bin,
    find_tissue_positions,
    id_mismatch_msg,
    read_coords_csv,
    read_tissue_positions,
    record_ignored,
    record_in_tissue,
    record_method,
    unsupported_choice_msg,
)

#: ``BayesTME_VI.__init__(rho=0.5)``: the library's own spatial smoothing strength. Its entry point
#: ``deconvolve(rho=None)`` forwards ``None`` over it, and the spatial guide then multiplies a tensor
#: by ``None`` on the first SVI step. Filled in here when the caller leaves ``deconvolution.rho`` unset.
BAYESTME_VI_DEFAULT_RHO = 0.5

#: The lattices ``bayestme.common.Layout`` defines; ``uns['layout']`` is read back as ``Layout[name]``.
LAYOUTS = ("HEX", "SQUARE", "IRREGULAR")

#: ``bayestme.marker_genes.MarkerGeneMethod`` values, the only ones ``select_marker_genes`` accepts.
MARKER_GENE_METHODS = ("TIGHT", "BEST_AVAILABLE", "FALSE_DISCOVERY_RATE")

#: The methods that apply ``alpha`` as a cutoff (TIGHT: ``omega > 1 - alpha``; FALSE_DISCOVERY_RATE:
#: ``fdr <= alpha``). BEST_AVAILABLE ranks every gene and never reads ``alpha``.
CUTOFF_MARKER_GENE_METHODS = ("TIGHT", "FALSE_DISCOVERY_RATE")

#: ``select_marker_genes``' own defaults for ``--n-marker-genes`` and ``--alpha`` (its argparse), used
#: when ``marker_genes`` leaves the key out or sets it to null.
MARKER_GENES_DEFAULT_N = 10
MARKER_GENES_DEFAULT_ALPHA = 0.05

#: ``bayestme.gene_filtering.RIBOSOME_GENE_NAME_PATTERN`` (``re.match`` against a gene name), read from the
#: module when it defines it; this copy only stands in for a module that does not.
RIBOSOME_GENE_NAME_PATTERN = "[Rr][Pp][SsLl]"

#: var columns read as gene symbols when var_names are Ensembl IDs, in ``worker_utils.harmonize_gene_ids``'
#: order. CELLxGENE exports keep Ensembl var_names and the symbols in ``var['feature_name']``.
SYMBOL_COLUMNS = ("SYMBOL", "gene_symbols", "gene_name", "GeneName", "GeneName-2", "feature_name")

_ENSEMBL_GENE_ID = re.compile(r"^ENS[A-Z]*G\d{6,}")

#: The input modes that read a user's h5ad, the only ones with an ``adata.raw`` to read counts from.
#: The other two read a 10x count matrix, which holds counts already.
RAW_READING_MODES = ("h5ad", "generic_counts_coords")

#: Rows of X the counts check reads at a time, so it never builds a dense copy of the matrix.
COUNT_CHECK_BLOCK_ROWS = 1024


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _build_knn_connectivities(coords: np.ndarray, k: int = 6) -> sparse.csr_matrix:
    """
    Build a sparse boolean adjacency matrix (N x N) using kNN in coordinate space.
    BayesTME expects adata.obsp['connectivities'] to indicate neighbor relationships.  See https://bayestme.readthedocs.io/en/latest/data_format.html
    """
    n = coords.shape[0]
    if n == 0:
        raise ValueError("No spots/cells found (coords empty).")
    k_eff = max(2, min(int(k), n))
    tree = cKDTree(coords.astype(float))
    dists, idxs = tree.query(coords, k=k_eff)  # includes self at position 0
    rows = np.repeat(np.arange(n), k_eff - 1)
    cols = idxs[:, 1:].reshape(-1)  # skip self
    data = np.ones(rows.shape[0], dtype=bool)
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    # symmetrize
    mat = (mat + mat.T).astype(bool)
    mat.setdiag(False)
    mat.eliminate_zeros()
    return mat


def _tissue_mask(values) -> np.ndarray:
    """Read an ``in_tissue`` column as booleans by value, whatever dtype the file stored it in.

    ``np.asarray(values, dtype=bool)`` is right for bools and 0/1 integers and silently wrong for
    text. anndata stores a string column as a categorical, and a categorical of ``'0'``/``'1'`` (or
    ``'False'``/``'True'``) converts to all-True, so every out-of-tissue spot was deconvolved as
    tissue. Numbers (including numeric text) are read as ``!= 0``; true/false spellings by name;
    anything else, and a missing value, is refused with what was found.
    """
    import pandas as pd

    series = pd.Series(np.asarray(values, dtype=object).ravel())
    n_missing = int(series.isna().sum())
    if n_missing:
        raise ValueError(
            f"in_tissue has {n_missing} missing value(s) of {len(series)}; every spot needs 1/0 (or true/false) "
            "to say whether BayesTME should deconvolve it."
        )
    numeric = pd.to_numeric(series, errors="coerce")
    if bool(numeric.notna().all()):
        return numeric.to_numpy(dtype=float) != 0
    text = series.astype(str).str.strip().str.lower()
    truthy = text.isin(("1", "1.0", "true", "t", "yes", "y"))
    falsy = text.isin(("0", "0.0", "false", "f", "no", "n"))
    unknown = sorted(set(text[~(truthy | falsy)]))
    if unknown:
        raise ValueError(
            f"in_tissue must hold 1/0 or true/false; found {unknown[:5]}"
            + (f" and {len(unknown) - 5} more" if len(unknown) > 5 else "")
            + "."
        )
    return truthy.to_numpy(dtype=bool)


def _flag(value, name: str, default: bool) -> bool:
    """A switch from the JSON payload, read by value.

    ``bool(value)`` turned the text ``"false"`` into True, so ``{"use_spatial_guide": "false"}`` ran the
    spatial guide the caller had switched off. JSON booleans, 0/1 and true/false text are read as
    meant; a missing or null value is the default; anything else is refused by name.
    """
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off", ""):
        return False if text else bool(default)
    raise ValueError(f"{name} must be true or false, got {value!r}.")


def _count_report(X, rows=None) -> dict:
    """Check every stored value of ``X`` (only the rows in ``rows``, when given) for what counts cannot hold.

    Returns the number of stored values read and how many of them are negative, NaN/inf, or not whole
    numbers, with the first fractional value as an example and the largest finite value. Sparse-aware:
    ``COUNT_CHECK_BLOCK_ROWS`` rows are read at a time, through a sparse matrix's stored values, so the
    check never builds a dense copy of a whole-transcriptome matrix.
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


def _counts_from_raw(adata, source: str):
    """``adata.raw`` as the AnnData to fit (``use_raw_counts=True``); obs, obsm and uns come with it."""
    if adata.raw is None:
        raise ValueError(
            f"use_raw_counts=True fits the counts in adata.raw, and {source} has no adata.raw. Leave "
            "use_raw_counts off to fit X, or supply an h5ad whose adata.raw holds the raw counts."
        )
    log(f"[BayesTME] use_raw_counts=True: fitting adata.raw ({adata.raw.n_vars} genes) instead of X")
    return adata.raw.to_adata()


def _require_counts(adata, tissue, counts_source: str, input_mode: str) -> dict:
    """Refuse a matrix BayesTME's Poisson likelihood cannot fit, before the SVI run, and say what would fit.

    BayesTME fits ``pyro.sample("obs", Poisson(...), obs=counts)`` with pyro's validation on, so a
    log-normalised X (non-integer) or a scaled one (negative) died on the first SVI step with pyro's
    ``Expected value argument ... to be within the support (IntegerGreaterThan(lower_bound=0))``, which
    names nothing the caller controls. CELLxGENE Visium exports are exactly that: a processed X and the
    integer counts in ``adata.raw``, and with no Space Ranger folder beside them h5ad mode is the only
    way in. Only the in-tissue rows are checked, because only they reach the model.
    """
    rep = _count_report(adata.X, rows=tissue)
    if _holds_counts(rep):
        return rep
    n_in = int(np.count_nonzero(np.asarray(tissue, dtype=bool)))
    where = "adata.raw" if counts_source == "raw" else "X"
    msg = (
        f"BayesTME fits a Poisson likelihood to raw counts, and the matrix it would fit ({where} over the {n_in} "
        f"in-tissue spots) is not counts: {_count_problems(rep)}. "
    )
    if counts_source == "raw":
        msg += "It was read from adata.raw because use_raw_counts=True; supply raw counts there or in X."
    elif input_mode in RAW_READING_MODES:
        raw = getattr(adata, "raw", None)
        if raw is None:
            msg += "The h5ad has no adata.raw to read counts from; supply the raw counts in X."
        else:
            raw_rep = _count_report(raw.X, rows=tissue)
            if _holds_counts(raw_rep):
                msg += (
                    f"adata.raw holds non-negative integer counts for the same spots ({raw.n_vars} genes): pass "
                    "use_raw_counts=True to fit those instead of X."
                )
            else:
                msg += f"adata.raw is not counts either ({_count_problems(raw_rep)}); supply the raw counts in X."
    else:
        msg += "Supply the raw count matrix of the same Space Ranger run."
    raise ValueError(msg)


def _ensure_bayestme_fields(adata, layout: str, spatial_coords: np.ndarray, in_tissue: np.ndarray, knn_k: int) -> None:
    """
    Ensures BayesTME-required AnnData fields exist.  See https://bayestme.readthedocs.io/en/latest/data_format.html

    ``obsp['connectivities']`` is written because BayesTME's data format carries it, but the SVI
    deconvolution never reads it: ``BayesTME_VI`` rebuilds its own edges from the in-tissue positions
    and ``uns['layout']``. ``knn_k`` therefore shapes only this saved graph.
    """
    # Required fields per BayesTME docs: X, obsm['spatial'], obs['in_tissue'], uns['layout'], obsp['connectivities']  See https://bayestme.readthedocs.io/en/latest/data_format.html
    adata.obsm["spatial"] = np.asarray(spatial_coords, dtype=float)
    adata.obs["in_tissue"] = _tissue_mask(in_tissue)
    adata.uns["layout"] = str(layout).upper()
    adata.obsp["connectivities"] = _build_knn_connectivities(adata.obsm["spatial"], k=knn_k)


def _load_visium_h5_spatial(visium_h5_path: str, visium_spatial_dir: str, coord_type: str, knn_k: int):
    """
    Load a Visium-style dataset where the user has:
      - filtered_feature_bc_matrix.h5 (possibly renamed)
      - spatial/tissue_positions_list.csv (older format)
    and build a BayesTME-compliant AnnData.

    tissue_positions_list.csv columns per 10x:
      barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres  See https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/spatial-outputs
    """
    import scanpy as sc

    positions_path = find_tissue_positions(visium_spatial_dir)
    if positions_path is None:
        raise FileNotFoundError(f"Missing {' or '.join(TISSUE_POSITIONS_NAMES)} under: {visium_spatial_dir}")

    # Read counts from 10x H5
    log(f"[BayesTME] Reading 10x H5 counts: {visium_h5_path}")
    adata = sc.read_10x_h5(visium_h5_path)
    adata.var_names_make_unique()

    # Read spot positions
    pos = read_tissue_positions(positions_path).set_index("barcode")

    # Align barcodes
    # scanpy uses barcodes as obs_names
    common = adata.obs_names.intersection(pos.index)
    if len(common) == 0:
        raise ValueError(
            id_mismatch_msg("barcodes", "counts (10x H5)", adata.obs_names, "tissue_positions_list.csv", pos.index)
            + " Check that both come from the same sample and are not modified."
        )
    adata = adata[common].copy()
    pos = pos.loc[common].copy()

    in_tissue = (pos["in_tissue"].astype(int) == 1).values

    coord_type = (coord_type or "array").lower()
    if coord_type == "array":
        # lattice coords (good for Visium neighbor graph)
        coords = np.vstack([pos["array_row"].values, pos["array_col"].values]).T
        layout = "HEX"  # Visium is a hex/offset lattice in general  See https://bayestme.readthedocs.io/en/latest/data_format.html
    elif coord_type == "pixel":
        # pixel coords (fullres image space)
        # 10x defines row=y, col=x  See https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/spatial-outputs
        coords = np.vstack([pos["pxl_row_in_fullres"].values, pos["pxl_col_in_fullres"].values]).T
        layout = "IRREGULAR"
    else:
        raise ValueError("coord_type must be 'array' or 'pixel'")

    _ensure_bayestme_fields(adata, layout=layout, spatial_coords=coords, in_tissue=in_tissue, knn_k=int(knn_k))
    return adata


def _load_generic_counts_coords(counts_h5ad_path: str, coords_csv: str, knn_k: int, use_raw_counts: bool = False):
    """
    Generic loader: counts AnnData + a coordinates CSV with:
      barcode,x,y[,in_tissue]
    Produces BayesTME-compliant AnnData.  See https://bayestme.readthedocs.io/en/latest/data_format.html
    ``use_raw_counts`` fits the h5ad's ``adata.raw`` instead of its X.
    """
    import anndata as ad

    adata = ad.read_h5ad(counts_h5ad_path)
    if use_raw_counts:
        adata = _counts_from_raw(adata, "counts_h5ad_path")
    df = read_coords_csv(coords_csv).set_index("barcode")

    common = adata.obs_names.intersection(df.index)
    if len(common) == 0:
        raise ValueError(id_mismatch_msg("barcodes", "counts_h5ad", adata.obs_names, "coords_csv", df.index))
    adata = adata[common].copy()
    df = df.loc[common].copy()

    coords = df[["x", "y"]].values.astype(float)
    # read_coords_csv always supplies in_tissue, defaulting to 1 where the file omits it. Read by
    # value: ``astype(bool)`` turns the text "false" into True.
    in_tissue = _tissue_mask(df["in_tissue"].values)

    _ensure_bayestme_fields(adata, layout="IRREGULAR", spatial_coords=coords, in_tissue=in_tissue, knn_k=int(knn_k))
    return adata


def _require_n_components(dc: dict) -> int:
    """Validate the one deconvolution knob BayesTME cannot infer.

    ``deconvolve(..., n_components=None)`` passes ``K=None`` straight into pyro, which fails with a
    ``torch.ones()`` signature error naming nothing the caller controls. Refuse it here instead,
    while the parameter still has a name. See test/test_bayestme_requires_n_components.py.
    """
    raw = dc.get("n_components")
    if raw is None:
        raise ValueError(
            "deconvolution.n_components is required: BayesTME cannot infer how many cell types to "
            "deconvolve into. Pass deconvolution={'n_components': <int>} -- e.g. the number of cell "
            "types in your reference annotation, or the expected number in the tissue."
        )
    not_whole = f"deconvolution.n_components must be a whole number of cell types, got {raw!r}."
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise ValueError(not_whole) from None
    if isinstance(raw, float) and n != raw:
        raise ValueError(not_whole)
    if n < 1:
        raise ValueError(f"deconvolution.n_components must be at least 1 cell type, got {raw!r}.")
    return n


def _resolve_rho(dc: dict):
    """The spatial smoothing strength BayesTME will use, and where it came from.

    ``deconvolve(rho=None)`` does not mean "use the default": ``BayesTME_VI`` stores the ``None``
    over its own ``rho=0.5`` and the spatial guide computes ``Tensor * None`` on the first SVI step
    (``TypeError: unsupported operand type(s) for *: 'Tensor' and 'NoneType'``). So an unset
    ``deconvolution.rho`` -- the documented default -- crashed every spatial run. Fill in the
    library's own value instead, and say so. A negative rho would reward differences between
    neighbouring spots rather than penalise them, so it is refused.
    """
    raw = dc.get("rho")
    if raw is None:
        return float(BAYESTME_VI_DEFAULT_RHO), "BayesTME_VI default (deconvolution.rho not given)"
    bad = f"deconvolution.rho must be a non-negative number (the spatial smoothing strength), got {raw!r}."
    if isinstance(raw, bool):
        raise ValueError(bad)
    try:
        rho = float(raw)
    except (TypeError, ValueError):
        raise ValueError(bad) from None
    if not math.isfinite(rho) or rho < 0:
        raise ValueError(bad)
    return rho, "deconvolution.rho"


def _require_positive_int(dc: dict, key: str, default: int, section: str = "deconvolution") -> int:
    """``<section>[key]`` as a whole number >= 1 (5.0 reads as 5), refused by name before the SVI run starts."""
    raw = dc.get(key)
    if raw is None:
        return int(default)
    bad = f"{section}.{key} must be a whole number >= 1, got {raw!r}."
    if isinstance(raw, bool):
        raise ValueError(bad)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(bad) from None
    if (isinstance(raw, float) and value != raw) or value < 1:
        raise ValueError(bad)
    return value


def _require_spatial_positions(positions) -> None:
    """Refuse the spatial guide when the in-tissue spots carry no spatial information at all.

    This used to switch the spatial guide off on its own (any axis with near-zero variance) and say
    so only on stderr, blaming the ``Tensor * None`` crash on degenerate coordinates. That crash was
    ``rho=None`` (see :func:`_resolve_rho`). What degenerate coordinates really do is give the
    spatial prior arbitrary neighbours: every spot is at distance 0 from every other. A line of
    spots (one constant axis) is not degenerate -- its neighbours are real -- so only positions
    that are all the same point are refused, and the caller chooses the non-spatial model.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[0] < 2 or bool(np.all(np.ptp(pos, axis=0) < 1e-9)):
        where = pos[0].tolist() if pos.ndim == 2 and pos.shape[0] else []
        raise ValueError(
            f"BayesTME's spatial guide needs in-tissue spots at different positions; all {pos.shape[0]} "
            f"in-tissue spot(s) sit at {where}, so it would smooth over arbitrary neighbours. Supply the real "
            "spot coordinates, or pass deconvolution={'use_spatial_guide': False} to run BayesTME without its "
            "spatial prior."
        )


def _resolve_layout(requested, file_layout):
    """The lattice for ``input_mode='h5ad'``: ``(layout, source, note_or_None)``.

    The ``layout`` parameter wins when it is given. When it is empty (the portal's default), the
    h5ad's own ``uns['layout']`` is used -- the old docs told callers to put it there, and the old
    code overwrote it with HEX, which builds a lattice graph over irregular positions. A file with
    no ``uns['layout']`` gets HEX, as before. Validated here because BayesTME reads the value back as
    ``Layout[adata.uns["layout"]]``, so a misspelling would surface as a bare KeyError far from the
    parameter that caused it.
    """
    if isinstance(file_layout, bytes):
        file_layout = file_layout.decode("utf-8", "replace")
    in_file = str(file_layout).strip().upper() if file_layout is not None else ""
    wanted = str(requested or "").strip().upper()
    if wanted:
        if wanted not in LAYOUTS:
            raise ValueError(unsupported_choice_msg("layout", requested, list(LAYOUTS)))
        note = None
        if in_file and in_file != wanted:
            note = (
                f"layout='{wanted}' replaced the h5ad's own uns['layout']={file_layout!r}; leave layout empty to use "
                "the file's."
            )
        return wanted, "layout parameter", note
    if in_file:
        if in_file not in LAYOUTS:
            raise ValueError(
                unsupported_choice_msg(
                    "uns['layout']",
                    file_layout,
                    list(LAYOUTS),
                    extra="It was read from the h5ad because no layout parameter was given; pass layout= to override it.",
                )
            )
        return in_file, "h5ad uns['layout']", None
    return "HEX", "default (no layout parameter and no uns['layout'] in the h5ad)", None


def _resolve_marker_method(marker_cfg: dict) -> str:
    """``marker_genes.method``, checked before the SVI run rather than after it, by the CLI's argparse."""
    raw = marker_cfg.get("method") or "BEST_AVAILABLE"
    method = str(raw).strip().upper()
    if method not in MARKER_GENE_METHODS:
        raise ValueError(unsupported_choice_msg("marker_genes.method", raw, list(MARKER_GENE_METHODS)))
    return method


def _resolve_marker_counts(marker_cfg: dict):
    """``(n_marker_genes, alpha)`` for ``select_marker_genes``, checked before the SVI run.

    The CLI reads ``--n-marker-genes`` with argparse ``type=int`` and ``--alpha`` with ``type=float``,
    and it runs AFTER the deconvolution. The worker used to hand it ``str(value)`` unchecked, so a JSON
    null (``'None'``), an LLM-style ``5.0`` (``'5.0'``) or a word failed argparse only once the whole
    SVI run was done, and the run ended as an error. A null is the CLI's own default (10 / 0.05), a
    whole float is its integer, and anything else is refused here, by name. ``alpha`` is the marker
    cutoff of the two cutoff methods (``omega > 1 - alpha`` for TIGHT, ``fdr <= alpha`` for
    FALSE_DISCOVERY_RATE), so it must lie in (0, 1]; BEST_AVAILABLE does not read it.
    """
    n_markers = _require_positive_int(marker_cfg, "n_marker_genes", MARKER_GENES_DEFAULT_N, section="marker_genes")
    raw = marker_cfg.get("alpha")
    if raw is None:
        return n_markers, float(MARKER_GENES_DEFAULT_ALPHA)
    bad = f"marker_genes.alpha must be a number in (0, 1] (the marker-gene cutoff), got {raw!r}."
    if isinstance(raw, bool):
        raise ValueError(bad)
    try:
        alpha = float(raw)
    except (TypeError, ValueError):
        raise ValueError(bad) from None
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError(bad)
    return n_markers, alpha


def _require_marker_method_can_finish(method: str, alpha: float) -> None:
    """Refuse, before the SVI run, a marker method the ``select_marker_genes`` CLI cannot finish.

    After choosing the markers, the CLI writes ``marker_genes.csv`` through ``create_top_gene_lists``
    with ``n_marker_genes=stdata.n_gene``: it labels ``n_gene`` rows per cell type but has only the
    genes that passed the cutoff, so pandas stops on ``Length of values (N) does not match length of
    index (M)`` and nothing is saved. BEST_AVAILABLE passes every gene and finishes. The two cutoff
    methods finish only when every gene passes: FALSE_DISCOVERY_RATE does at alpha=1 (``fdr <= 1``
    always holds) and at no smaller alpha one can count on; TIGHT (``omega > 1 - alpha``) did not even
    at alpha=1, because many genes have omega 0. Measured on Heart Fetal12W after 10 SVI steps: TIGHT
    passed 1,381 (alpha 0.05) and 12,835 (alpha 1) of 35,367 genes, FALSE_DISCOVERY_RATE 1,534
    (alpha 0.05), and each of those CLI runs stopped. The failure used to come after the whole SVI run.
    """
    if method == "TIGHT" or (method in CUTOFF_MARKER_GENE_METHODS and alpha < 1.0):
        raise ValueError(
            f"marker_genes.method={method!r} (alpha={alpha}) cannot finish: BayesTME's select_marker_genes CLI "
            'writes its gene list for every gene of the slide and stops ("Length of values ... does not match '
            'length of index") as soon as one gene fails the cutoff'
            + (
                ", which TIGHT's omega > 1 - alpha does even at alpha=1 (genes with omega 0)"
                if method == "TIGHT"
                else ", which fdr <= alpha does below alpha=1"
            )
            + ". Use marker_genes.method='BEST_AVAILABLE' (ranks every gene; alpha is not used), or "
            "marker_genes={'run': False} to skip marker selection."
        )


def _require_spaceranger_raw_matrix(spaceranger_dir: str) -> str:
    """Refuse a Space Ranger directory ``read_spaceranger`` cannot read, while we can still say why.

    ``SpatialExpressionDataset.read_spaceranger`` reads the RAW feature-barcode matrix and nothing
    else: it globs the directory for ``*raw_feature_bc_matrix.h5``, falls back to
    ``*raw_feature_bc_matrix`` for the mtx form, and otherwise raises ``No raw count matrix found in
    spaceranger directoryexpected None or None`` -- a message missing a space, whose two ``None``s
    are the variables it has just proved are ``None``, and which names no file to go and fetch.

    BayesTME is alone in this. SpaceFlow and Starfysh both read a filtered-only ``outs/`` tree
    through their own ``spaceranger_outs`` modes, and our portal advertises the mode here without
    saying it means something narrower.

    The two globs below are upstream's, so this accepts exactly what upstream accepts: widening it
    would wave a doomed run through under a friendly message, and narrowing it would refuse a
    directory that runs today. Returns the path it found. See
    test/test_bayestme_says_which_visium_matrix_it_needs.py.
    """
    if not os.path.isdir(spaceranger_dir):
        raise ValueError(
            f"input_mode=spaceranger_outs needs the Space Ranger output directory itself, and "
            f"{spaceranger_dir} is not a directory."
        )
    for pattern in ("*raw_feature_bc_matrix.h5", "*raw_feature_bc_matrix"):
        found = sorted(glob.glob(os.path.join(spaceranger_dir, pattern)))
        if found:
            return found[0]
    raise ValueError(
        f"BayesTME's spaceranger_outs mode reads the RAW feature-barcode matrix, and "
        f"{spaceranger_dir} has none: read_spaceranger looks there for raw_feature_bc_matrix.h5 or a "
        "raw_feature_bc_matrix/ directory (either may carry a sample prefix) and never reads the "
        "filtered matrix. Either add the raw matrix from the same Space Ranger run, or use "
        "input_mode=visium_h5_spatial (visium_h5_path + visium_spatial_dir) or "
        "input_mode=generic_counts_coords, both of which accept a filtered-only directory."
    )


def _load_input_as_spatial_expression_dataset(payload: dict[str, Any], report: dict | None = None):
    """
    Returns a bayestme.data.SpatialExpressionDataset from supported inputs.

    ``report`` (optional) receives ``input_mode``, ``layout``, ``layout_source`` and a ``notes`` list,
    so the payload can say which lattice the spatial prior was built on and why.
    """
    from bayestme.data import SpatialExpressionDataset

    if report is None:
        report = {}
    report.setdefault("notes", [])
    input_mode = (payload.get("input_mode") or "spaceranger_outs").lower()
    report["input_mode"] = input_mode
    knn_k = int(payload.get("knn_k", 6))
    # Which matrix is fitted: X of the file, or its adata.raw. The two 10x modes read a count matrix.
    use_raw_counts = _flag(payload.get("use_raw_counts"), "use_raw_counts", False)
    report["counts_source"] = "raw" if use_raw_counts and input_mode in RAW_READING_MODES else "X"

    if input_mode == "spaceranger_outs":
        spaceranger_dir = payload.get("spaceranger_dir")
        if not spaceranger_dir:
            raise ValueError("input_mode=spaceranger_outs requires spaceranger_dir")
        raw_matrix = _require_spaceranger_raw_matrix(spaceranger_dir)
        log(f"[BayesTME] Loading via SpatialExpressionDataset.read_spaceranger: {spaceranger_dir}")
        log(f"[BayesTME] Raw feature-barcode matrix: {raw_matrix}")
        stdata = SpatialExpressionDataset.read_spaceranger(
            spaceranger_dir
        )  # helper mentioned in docs  See https://bayestme.readthedocs.io/en/latest/data_format.html
        report["layout"] = str(stdata.adata.uns.get("layout", "HEX"))
        report["layout_source"] = "read_spaceranger (Visium lattice)"
        return stdata

    if input_mode == "visium_h5_spatial":
        visium_h5_path = payload.get("visium_h5_path")
        visium_spatial_dir = payload.get("visium_spatial_dir")
        coord_type = payload.get("coord_type", "array")  # 'array' or 'pixel'
        if not visium_h5_path or not visium_spatial_dir:
            raise ValueError("input_mode=visium_h5_spatial requires visium_h5_path and visium_spatial_dir")
        adata = _load_visium_h5_spatial(visium_h5_path, visium_spatial_dir, coord_type=coord_type, knn_k=knn_k)
        report["layout"] = str(adata.uns["layout"])
        report["layout_source"] = f"coord_type={str(coord_type or 'array').lower()!r}"
        return SpatialExpressionDataset(adata)

    if input_mode == "h5ad":
        import anndata as ad

        h5ad_path = payload.get("h5ad_path")
        if not h5ad_path:
            raise ValueError("input_mode=h5ad requires h5ad_path")
        log(f"[BayesTME] Reading h5ad: {h5ad_path}")
        adata = ad.read_h5ad(h5ad_path)
        if use_raw_counts:
            adata = _counts_from_raw(adata, "h5ad_path")
        # The other three input modes all refuse to start without a coordinate source. This one used
        # to substitute np.zeros((n_obs, 2)) instead, so an h5ad whose coordinates sit under another
        # name -- obsm['X_spatial'] after a Seurat conversion -- was stacked onto the origin. The
        # zeros then became state: _ensure_bayestme_fields writes them into obsm['spatial'] and
        # builds obsp['connectivities'] from them, main() notices the zero variance and turns the
        # spatial guide off, and the run finishes ok with BayesTME's spatial prior unused. Say what
        # was looked for and what is there. The in_tissue fallback below is a different case and is
        # left alone: a non-Visium h5ad legitimately has no tissue mask.
        if "spatial" not in adata.obsm:
            raise ValueError(
                f"Spot coordinates not found: adata.obsm['spatial'] is missing. "
                f"Available obsm keys: {list(adata.obsm.keys())}"
            )
        # The other three loaders derive the layout from the input (Visium array coordinates are a
        # hex lattice; pixel coordinates and a generic coords CSV are irregular). Only this branch
        # has nothing to derive it from: the caller's ``layout`` wins, then the file's own
        # uns['layout'], then HEX (a Visium h5ad is the common case). See _resolve_layout.
        layout, layout_source, layout_note = _resolve_layout(payload.get("layout"), adata.uns.get("layout"))
        report["layout"] = layout
        report["layout_source"] = layout_source
        if layout_note:
            report["notes"].append(layout_note)
        _ensure_bayestme_fields(
            adata,
            layout=layout,
            spatial_coords=adata.obsm["spatial"],
            in_tissue=adata.obs["in_tissue"].values if "in_tissue" in adata.obs else np.ones(adata.n_obs, dtype=bool),
            knn_k=knn_k,
        )
        return SpatialExpressionDataset(adata)

    if input_mode == "generic_counts_coords":
        counts_h5ad_path = payload.get("counts_h5ad_path")
        coords_csv = payload.get("coords_csv")
        if not counts_h5ad_path or not coords_csv:
            raise ValueError("input_mode=generic_counts_coords requires counts_h5ad_path and coords_csv")
        adata = _load_generic_counts_coords(counts_h5ad_path, coords_csv, knn_k=knn_k, use_raw_counts=use_raw_counts)
        report["layout"] = "IRREGULAR"
        report["layout_source"] = "generic coordinates are free positions"
        return SpatialExpressionDataset(adata)

    raise ValueError(
        unsupported_choice_msg(
            "input_mode", input_mode, ["spaceranger_outs", "visium_h5_spatial", "h5ad", "generic_counts_coords"]
        )
    )


def _ribosome_name_source(adata):
    """``(names, matched_on, ensembl)``: the names the ribosomal-gene pattern is matched against, where from,
    and whether most var_names are Ensembl IDs.

    Upstream's ``filter_ribosome_genes`` matches ``[Rr][Pp][SsLl]`` against ``var_names``. On a
    CELLxGENE h5ad those are Ensembl IDs (``ENSG...``) and the symbols sit in ``var['feature_name']``,
    so the filter removed nothing and the run recorded ``n_genes_before == n_genes_after`` with no
    word (0 of 35,476 genes on the library's Heart Fetal12W sample, 109 by ``feature_name``). When
    most var_names are Ensembl IDs and a symbol column exists, that column is matched instead;
    otherwise ``var_names`` are, exactly as upstream does. ``matched_on`` is ``"var_names"`` or
    ``"var['<column>']"``.
    """
    names = np.asarray(adata.var_names, dtype=object).astype(str)
    n_ensembl = sum(1 for g in names if _ENSEMBL_GENE_ID.match(g))
    if not len(names) or n_ensembl * 2 <= len(names):
        return names, "var_names", False
    for col in SYMBOL_COLUMNS:
        if col not in adata.var.columns:
            continue
        symbols = np.asarray(adata.var[col], dtype=object)
        text = np.array(["" if s is None or s != s else str(s).strip() for s in symbols], dtype=object)
        usable = [t for t in text if t and t.lower() not in ("nan", "none") and not _ENSEMBL_GENE_ID.match(t)]
        if usable:
            return text, f"var[{col!r}]", True
    return names, "var_names", True


def _filter_ribosomal_genes(stdata, gene_filtering, notes: list):
    """``(stdata, step)``: upstream's ribosomal-gene filter, matched against the gene symbols.

    With symbol var_names this is upstream's ``filter_ribosome_genes`` itself. With Ensembl var_names
    and a symbol column (see :func:`_ribosome_name_source`) the same pattern, taken from
    ``bayestme.gene_filtering``, is matched against that column and the genes are cut the way upstream
    cuts them. With Ensembl var_names and no symbol column nothing can match, and ``notes`` says so.
    """
    before = int(stdata.adata.n_vars)
    names, matched_on, ensembl = _ribosome_name_source(stdata.adata)
    log(f"[BayesTME] Filtering ribosomal genes (matching {matched_on})")
    if matched_on == "var_names":
        stdata = gene_filtering.filter_ribosome_genes(stdata)
    else:
        pattern = getattr(gene_filtering, "RIBOSOME_GENE_NAME_PATTERN", RIBOSOME_GENE_NAME_PATTERN)
        keep = ~np.array([bool(re.match(pattern, g)) for g in names], dtype=bool)
        stdata = type(stdata)(stdata.adata[:, keep].copy())
    after = int(stdata.adata.n_vars)
    if matched_on != "var_names":
        notes.append(
            f"gene_filtering.filter_ribosomal_genes matched the ribosomal-gene pattern against {matched_on}, "
            f"because var_names are Ensembl IDs, which the pattern can never match; {before - after} of "
            f"{before} genes were removed."
        )
    elif after == before and ensembl:
        notes.append(
            "gene_filtering.filter_ribosomal_genes removed no gene: var_names are Ensembl IDs, which the "
            "ribosomal-gene pattern ([Rr][Pp][SsLl]) can never match, and var has no gene-symbol column ("
            + ", ".join(SYMBOL_COLUMNS)
            + ") to match instead."
        )
    step = {
        "step": "filter_ribosomal_genes",
        "matched_on": matched_on,
        "n_genes_before": before,
        "n_genes_after": after,
    }
    return stdata, step


def _filter_genes(stdata, gf: dict, gene_filtering, utils, notes: list | None = None):
    """Apply the requested BayesTME gene filters, in upstream's order; return ``(stdata, steps)``.

    ``steps`` records each filter with the gene count before and after it, so the payload can say
    what the model was fitted on. ``spot_threshold`` is upstream's ``filter_genes_by_spot_threshold``,
    which KEEPS genes detected in at most ``int(spot_threshold * n_in_tissue_spots)`` spots -- it
    removes near-ubiquitous genes, not rare ones (0.95 drops genes seen in more than 95% of spots).
    A key present with a null value is treated as not given. ``notes`` (optional) receives sentences
    the payload should carry as warnings (see :func:`_filter_ribosomal_genes`).
    """
    steps = []
    if notes is None:
        notes = []

    def n_genes(ds) -> int:
        return int(ds.adata.n_vars)

    if _flag(gf.get("filter_ribosomal_genes"), "gene_filtering.filter_ribosomal_genes", False):
        stdata, step = _filter_ribosomal_genes(stdata, gene_filtering, notes)
        steps.append(step)

    if gf.get("spot_threshold") is not None:
        raw_thr = gf["spot_threshold"]
        try:
            thr = float(raw_thr) if not isinstance(raw_thr, bool) else float("nan")
        except (TypeError, ValueError):
            thr = float("nan")
        if not (0.0 < thr <= 1.0):
            raise ValueError(
                f"gene_filtering.spot_threshold must be in (0, 1], got {gf['spot_threshold']!r}: it is the largest "
                "fraction of in-tissue spots a gene may be detected in and still be kept."
            )
        n_in = int(np.asarray(stdata.adata.obs["in_tissue"], dtype=bool).sum())
        max_spots = int(n_in * thr)
        before = n_genes(stdata)
        log(
            f"[BayesTME] Dropping genes detected in more than {max_spots} of {n_in} in-tissue spots "
            f"(spot_threshold={thr})"
        )
        stdata = gene_filtering.filter_genes_by_spot_threshold(stdata, thr)
        steps.append(
            {
                "step": "spot_threshold",
                "value": thr,
                "kept_genes_detected_in_at_most_spots": max_spots,
                "n_genes_before": before,
                "n_genes_after": n_genes(stdata),
            }
        )

    if gf.get("n_top_by_standard_deviation") is not None:
        n = _require_positive_int(gf, "n_top_by_standard_deviation", 1, section="gene_filtering")
        before = n_genes(stdata)
        log(f"[BayesTME] Selecting top {n} genes by stddev")
        reads = np.asarray(stdata.counts)
        # By position, not by name: np.isin on names keeps every copy of a duplicated gene name, so
        # an h5ad with repeated var_names kept more than n genes.
        order = np.asarray(utils.get_stddev_ordering(reads))[:n]
        mask = np.zeros(before, dtype=bool)
        mask[order] = True
        stdata.adata = stdata.adata[:, mask].copy()
        steps.append(
            {
                "step": "n_top_by_standard_deviation",
                "value": n,
                "n_genes_before": before,
                "n_genes_after": n_genes(stdata),
            }
        )

    if steps and n_genes(stdata) == 0:
        raise ValueError(f"gene_filtering left no genes to deconvolve: {steps}")
    return stdata, steps


def _save_deconvolution_result(result, path: str) -> None:
    """Write a ``DeconvolutionResult`` the way ``DeconvolutionResult.save`` does, minus ``reads_trace``.

    ``save`` also writes ``f["reads_trace"] = self.reads_trace``, a property that materialises a dense
    samples x spots x genes x components array: about 787 GB for a 10,878-spot, 18,085-gene Visium
    slide at 10 components and the default 100 samples, so a run that finished SVI died saving it.
    The array is derived -- ``DeconvolutionResult.read_h5`` never reads it, and its ``reads_trace``
    property recomputes it from the traces written here -- so it is left out and never computed.
    Written as ``<path>.partial`` and renamed, so an interrupted write never sits at ``path``.
    """
    import h5py

    tmp = path + ".partial"
    try:
        with h5py.File(tmp, "w") as f:
            f["cell_prob_trace"] = result.cell_prob_trace
            f["expression_trace"] = result.expression_trace
            f["beta_trace"] = result.beta_trace
            f["cell_num_total_trace"] = result.cell_num_total_trace
            if result.losses is not None:
                f["losses"] = result.losses
            f.attrs["lam2"] = result.lam2
            f.attrs["n_components"] = result.n_components
            f.attrs["omitted_derived_datasets"] = "reads_trace"
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _write_h5ad_atomic(adata, path: str) -> None:
    """Write the AnnData as ``<path>.partial`` and rename over ``path`` once it is complete."""
    tmp = path + ".partial"
    try:
        adata.write_h5ad(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _knn_k_ignored_why(input_mode: str, layout: str) -> str:
    """Why ``knn_k`` did not reach the model, in terms of what did."""
    built = "a fixed 5-nearest-neighbour graph" if layout == "IRREGULAR" else f"{layout} lattice neighbours"
    saved = (
        "read_spaceranger builds obsp['connectivities'] itself, so knn_k is not used at all in this mode"
        if input_mode == "spaceranger_outs"
        else "knn_k only shapes obsp['connectivities'] in the saved h5ad, which the deconvolution never reads"
    )
    return f"BayesTME's spatial prior uses its own adjacency over the in-tissue spots ({built}); {saved}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    t0 = time.time()
    try:
        payload = json.loads(args.json)
        if payload.get("__tool__") != "bayestme_deconvolution":
            raise ValueError("Invalid __tool__ value")

        out_dir = payload["output_dir"]
        mkdir(out_dir)

        dc = payload.get("deconvolution", {}) or {}
        gf = payload.get("gene_filtering", {}) or {}
        marker_cfg = payload.get("marker_genes", {}) or {}

        # ---------- Validate every knob before the (hours-long) SVI run ----------
        n_components = _require_n_components(dc)
        rho, rho_source = _resolve_rho(dc)
        n_svi_steps = _require_positive_int(dc, "n_svi_steps", 10_000)
        n_samples = _require_positive_int(dc, "n_samples", 100)
        seed = int(dc.get("seed") if dc.get("seed") is not None else 0)
        use_spatial = _flag(dc.get("use_spatial_guide"), "deconvolution.use_spatial_guide", True)
        run_markers = _flag(marker_cfg.get("run"), "marker_genes.run", True)
        marker_method = _resolve_marker_method(marker_cfg) if run_markers else None
        # Checked here, not by the CLI's argparse after the SVI run (see _resolve_marker_counts).
        n_marker_genes, marker_alpha = _resolve_marker_counts(marker_cfg) if run_markers else (None, None)
        if run_markers:
            _require_marker_method_can_finish(marker_method, marker_alpha)
        use_raw_counts = _flag(payload.get("use_raw_counts"), "use_raw_counts", False)

        # ---------- Load input robustly ----------
        load_report: dict = {}
        stdata = _load_input_as_spatial_expression_dataset(payload, report=load_report)
        input_mode = load_report.get("input_mode", payload.get("input_mode"))
        layout = str(stdata.adata.uns.get("layout", load_report.get("layout", "")))

        n_spots = int(stdata.adata.n_obs)
        n_genes_input = int(stdata.adata.n_vars)
        tissue = np.asarray(stdata.adata.obs["in_tissue"], dtype=bool)
        n_spots_used = int(tissue.sum())
        if n_spots_used == 0:
            raise ValueError(
                f"No spot is marked in tissue (obs['in_tissue'] is false for all {n_spots} spots); BayesTME "
                "deconvolves in-tissue spots only."
            )

        # ---------- The Poisson likelihood needs counts: refuse anything else by name ----------
        counts_source = load_report.get("counts_source", "X")
        _require_counts(stdata.adata, tissue, counts_source, str(input_mode))

        # ---------- The spatial guide needs positions that carry information ----------
        if use_spatial:
            _require_spatial_positions(np.asarray(stdata.adata.obsm["spatial"])[tissue])

        # ---------- Optional gene filtering ----------
        from bayestme import gene_filtering, utils

        stdata, gene_steps = _filter_genes(stdata, gf, gene_filtering, utils, notes=load_report["notes"])
        n_genes_used = int(stdata.adata.n_vars)

        # ---------- Deconvolution ----------
        from bayestme.svi.deconvolution import deconvolve

        rng = np.random.default_rng(seed)

        log(
            f"[BayesTME] Running deconvolution: {n_spots_used} in-tissue spots x {n_genes_used} genes, "
            f"{n_components} components, rho={rho} ({rho_source}), spatial guide={'on' if use_spatial else 'off'}"
        )
        # stdout carries the JSON payload only; anything the library prints goes to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            result = deconvolve(
                stdata,
                n_components=n_components,
                rho=rho,
                n_svi_steps=n_svi_steps,
                n_samples=n_samples,
                use_spatial_guide=use_spatial,
                expression_truth=None,
                rng=rng,
            )

        # ---------- Save outputs ----------
        deconv_h5 = os.path.join(out_dir, "deconvolution_result.h5")
        log(f"[BayesTME] Saving DeconvolutionResult (without the derived reads_trace): {deconv_h5}")
        _save_deconvolution_result(result, deconv_h5)

        from bayestme.data import add_deconvolution_results_to_dataset

        add_deconvolution_results_to_dataset(stdata, result)

        adata_path = os.path.join(out_dir, "bayestme_deconvolved.h5ad")
        _write_h5ad_atomic(stdata.adata, adata_path)

        # ---------- Marker genes (CLI) ----------
        # CLI documented args (adata, adata-output, deconvolution-result, etc.)  See https://bayestme.readthedocs.io/en/latest/command_line_interface.html
        final_adata = adata_path

        if run_markers:
            out_adata = os.path.join(out_dir, "bayestme_with_markers.h5ad")
            out_partial = out_adata + ".partial"
            cmd = [
                env_bin("select_marker_genes"),
                "--adata",
                adata_path,
                "--adata-output",
                out_partial,
                "--deconvolution-result",
                deconv_h5,
                "--n-marker-genes",
                str(n_marker_genes),
                "--alpha",
                repr(float(marker_alpha)),
                "--marker-gene-method",
                str(marker_method),
            ]
            if _flag(marker_cfg.get("verbose"), "marker_genes.verbose", True):
                cmd.append("-v")

            log("[BayesTME] Running select_marker_genes: " + " ".join(cmd))
            import subprocess

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.stdout.strip():
                log("[select_marker_genes stdout]\n" + proc.stdout)
            if proc.stderr.strip():
                log("[select_marker_genes stderr]\n" + proc.stderr)
            if proc.returncode != 0:
                if os.path.exists(out_partial):
                    os.remove(out_partial)
                tail = [ln for ln in (proc.stderr or "").strip().splitlines() if ln.strip()][-5:]
                raise RuntimeError(
                    f"select_marker_genes failed (exit {proc.returncode})"
                    + (": " + " | ".join(tail) if tail else "")
                    + ". The deconvolution itself finished and is saved; pass marker_genes={'run': False} to skip "
                    "this step."
                )
            os.replace(out_partial, out_adata)
            final_adata = out_adata

        runtime = time.time() - t0
        k_used = int(result.n_components)
        n_out = n_spots - n_spots_used

        out = WorkerOutput("bayestme", task="deconvolution")
        out.set_data(
            n_spots=n_spots,
            n_spots_used=n_spots_used,
            n_spots_out_of_tissue=n_out,
            n_genes=n_genes_input,
            n_genes_used=n_genes_used,
        )
        out.add_output_files(
            {
                "deconvolution_h5": deconv_h5,
                "adata_h5ad": final_adata,
            }
        )
        out.add_params(
            {
                "input_mode": input_mode,
                "output_dir": out_dir,
                "n_components": n_components,
                "rho": rho,
                "rho_source": rho_source,
                "n_svi_steps": n_svi_steps,
                "n_samples": n_samples,
                "use_spatial_guide": use_spatial,
                "seed": seed,
                "layout": layout,
                "layout_source": load_report.get("layout_source"),
                "use_raw_counts": use_raw_counts,
                "counts_source": counts_source,
                "gene_filtering": gene_steps,
                "marker_genes_run": run_markers,
                "marker_gene_method": marker_method,
                # What select_marker_genes was run with (null when it did not run).
                "marker_n_marker_genes": n_marker_genes,
                # The cutoff that applied: null under BEST_AVAILABLE, which never reads alpha.
                "marker_alpha": marker_alpha if marker_method in CUTOFF_MARKER_GENE_METHODS else None,
                "reads_trace_saved": False,
            }
        )
        record_method(
            out,
            "BayesTME SVI deconvolution "
            + ("with the spatial guide" if use_spatial else "without the spatial guide (use_spatial_guide=False)")
            + (" + select_marker_genes" if run_markers else ""),
        )
        record_ignored(out, ["knn_k"], _knn_k_ignored_why(str(input_mode), layout))
        # BayesTME's own tissue mask: the out-of-tissue spots were never deconvolved (their obsm rows are zeros).
        record_in_tissue(out, n_spots, n_out)
        if run_markers and marker_method not in CUTOFF_MARKER_GENE_METHODS and marker_cfg.get("alpha") is not None:
            record_ignored(
                out,
                ["marker_genes.alpha"],
                f"marker_genes.method={marker_method!r} ranks every gene and applies no cutoff; only TIGHT and "
                "FALSE_DISCOVERY_RATE read alpha",
            )
        if not use_spatial and dc.get("rho") is not None:
            record_ignored(
                out, ["deconvolution.rho"], "use_spatial_guide=False runs the guide without the spatial regularizer"
            )
        if input_mode != "h5ad" and str(payload.get("layout") or "").strip():
            record_ignored(
                out, ["layout"], f"only input_mode='h5ad' reads it; {input_mode} takes the lattice from its input"
            )
        if input_mode != "visium_h5_spatial" and str(payload.get("coord_type") or "array").lower() != "array":
            record_ignored(out, ["coord_type"], f"only input_mode='visium_h5_spatial' reads it, not {input_mode}")
        if use_raw_counts and input_mode not in RAW_READING_MODES:
            record_ignored(
                out,
                ["use_raw_counts"],
                f"{input_mode} reads a 10x count matrix, which has no adata.raw; only input_mode='h5ad' and "
                "'generic_counts_coords' read one",
            )
        for note in load_report.get("notes", []):
            out.add_warning(note)
        out.set_summary(runtime_sec=round(runtime, 2), n_components=k_used)

        genes = f"{n_genes_used} genes"
        if gene_steps:
            genes += (
                f" (gene_filtering kept {n_genes_used} of {n_genes_input}: "
                + ", ".join(f"{s['step']} {s['n_genes_before']}->{s['n_genes_after']}" for s in gene_steps)
                + ")"
            )
        outside = ""
        if n_out:
            outside = (
                f" The other {n_out} of the {n_spots} spots are out of tissue and were not deconvolved; their rows in "
                "obsm['bayestme_cell_type_probabilities'] and obsm['bayestme_cell_type_counts'] are zeros."
            )
        source = " The counts fitted are adata.raw's (use_raw_counts=True), not X." if counts_source == "raw" else ""
        out.set_analysis(
            f"BayesTME SVI deconvolution into {k_used} components "
            f"({'with' if use_spatial else 'without'} the spatial guide on a {layout} layout, rho={rho}) "
            f"completed in {runtime:.1f}s over {n_svi_steps} SVI steps and {n_samples} posterior samples. "
            f"Deconvolved {n_spots_used} in-tissue spots x {genes}.{source}{outside} Results saved to {out_dir}."
        )
        out.emit()

    except Exception as e:
        log("[BayesTME][ERROR]")
        log(traceback.format_exc())
        WorkerOutput.emit_error("bayestme", str(e), task="deconvolution")
        sys.exit(1)


if __name__ == "__main__":
    main()
