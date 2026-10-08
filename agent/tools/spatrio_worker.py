#!/usr/bin/env python3
"""
SpaTrio worker: map single cells onto the spots of one spatial slice with optimal transport.

Runs inside /opt/conda/envs/spatrio conda env.

SpaTrio (``spatrio.ot_alignment``) solves a fused Gromov-Wasserstein optimal-transport problem
between the spots of a spatial transcriptomics slice (``rna_h5ad``) and the cells of a single-cell
(multi-omics) dataset (``other_h5ad``):

* the expression cost is a Euclidean distance on the features both objects SHARE -- SpaTrio
  inner-joins them, then runs ``normalize_total(1e4) + log1p + scale`` itself -- so both inputs
  must carry raw RNA counts under the same gene identifiers;
* the spot graph is a kNN graph on ``obsm['spatial']``; the cell graph is a kNN graph on
  ``obsm['reduction']`` of ``other_h5ad``, which is where an embedding of the second modality
  (ATAC LSI, ADT PCA, ...) belongs. When none is supplied, the worker computes a PCA of the
  ``other_h5ad`` RNA counts and says so;
* ``obs['type']`` makes the graph distances type-aware; types come only from ``annotation_key``;
* spots with ``obs['in_tissue'] == 0`` (background glass, which CELLxGENE Visium exports carry)
  are left out of the plan and counted in ``params.in_tissue_filter``.

Output: ``spatrio_aligned.csv``, the transport plan in long format -- one ``(spot, cell, value)``
row per spot-cell pair, n_spots x n_cells rows. It is a mapping table, not aligned coordinates.

All logs go to stderr; stdout is JSON-only (final result).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    record_ignored,
    record_in_tissue,
    record_method,
    spatial_coords,
)
from worker_utils import drop_unlabeled as _drop_unlabeled  # aliased: run_spatrio has a bool of that name

#: What actually runs, for ``params.method``. The upstream function is called; nothing substitutes it.
METHOD_NAME = "SpaTrio ot_alignment (fused Gromov-Wasserstein optimal transport, scaled_euc expression cost)"

#: The expression cost is computed on the features both inputs share. Below this many the run
#: continues but carries a warning: a cost built on a handful of accidental matches is noise.
MIN_SHARED_FEATURES_WARN = 50

#: SpaTrio's own default ``n_components``; the cap on the PCA the worker computes for obsm['reduction'].
MAX_REDUCTION_COMPS = 30

#: Upper edge of a "whole number" test on stored values (float32 counts round-trip exactly; this only
#: absorbs representation noise).
_INTEGER_TOL = 1e-3

#: Column names that usually hold labels, used only to suggest an ``annotation_key`` -- never read.
_LABEL_HINTS = ("type", "annot", "label", "cluster", "class")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def _ensure_dir(p: str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- inputs


def _matrix_profile(X, block: int = 1 << 22) -> dict[str, Any]:
    """min, max and whether every value is a finite whole number -- without densifying ``X``.

    Sparse matrices are read through their stored values (plus the implicit zeros); dense arrays
    through a flat view, in blocks, so no temporary the size of the matrix is ever allocated.
    """
    import scipy.sparse as sps

    if sps.issparse(X):
        values = np.asarray(X.data).ravel()
        n_total = int(X.shape[0]) * int(X.shape[1])
        implicit_zero = int(X.nnz) < n_total
    else:
        values = np.asarray(X).ravel(order="K")
        implicit_zero = False
    lo = 0.0 if implicit_zero else None
    hi = 0.0 if implicit_zero else None
    finite = True
    integer = True
    for start in range(0, values.size, block):
        chunk = values[start : start + block]
        if not np.all(np.isfinite(chunk)):
            finite = False
            integer = False
            chunk = chunk[np.isfinite(chunk)]
            if chunk.size == 0:
                continue
        c_lo = float(chunk.min())
        c_hi = float(chunk.max())
        lo = c_lo if lo is None else min(lo, c_lo)
        hi = c_hi if hi is None else max(hi, c_hi)
        if integer and not np.all(np.abs(chunk - np.rint(chunk)) <= _INTEGER_TOL):
            integer = False
    return {
        "min": 0.0 if lo is None else lo,
        "max": 0.0 if hi is None else hi,
        "finite": finite,
        "integer": integer,
    }


def _raw_counts(adata, layer: str, label: str):
    """The matrix SpaTrio receives for one input, and where it came from.

    SpaTrio normalises what it is given (``normalize_total(1e4) + log1p + scale`` on the merged
    shared features), so it must be given raw counts. Anything else would be normalised twice --
    which is what the old ``max > 50`` heuristic did to every count matrix it recognised.
    """
    if layer and layer in adata.layers:
        mat = adata.layers[layer]
        source = f"layers['{layer}']"
    else:
        mat = adata.X
        source = "X"
    prof = _matrix_profile(mat)
    if prof["finite"] and prof["integer"] and prof["min"] >= 0:
        return mat, source, prof
    kind = "non-finite" if not prof["finite"] else ("negative" if prof["min"] < 0 else "non-integer")
    layers = [str(k) for k in adata.layers.keys()]
    hint = (
        f" This file has layers {layers}: name the one that holds raw counts with layer=... ."
        if layers
        else " This file has no layers to read counts from."
    )
    raise ValueError(
        f"{label} {source} is not raw counts ({kind} values; min={prof['min']:g}, max={prof['max']:g}). SpaTrio "
        "normalises its input itself (normalize_total to 1e4, log1p and scaling on the shared features), so it must "
        "receive raw integer counts: normalised or log data would be normalised a second time. Put raw counts in X, "
        "or name the layer that holds them with layer=..." + hint
    )


def _label_candidates(adata) -> list[str]:
    return [str(c) for c in adata.obs.columns if any(h in str(c).lower() for h in _LABEL_HINTS)][:6]


def _resolve_types(adata, annotation_key: str, label: str, placeholder: str, allow_drop: bool):
    """Set ``obs['type']`` from ``annotation_key`` only, and say what happened.

    Returns ``(adata, info)``; ``adata`` is a row subset when ``allow_drop`` dropped unlabelled rows.
    ``info`` carries ``source``, ``n_types``, ``n_dropped``, ``typed`` and an optional ``warning``.
    A side without the key runs untyped: one placeholder type, which makes SpaTrio's type-aware
    graph distance identical to the plain one -- so the caller turns the aware flag off for it
    instead of paying for the n^2 melt that changes nothing. A pre-existing ``obs['type']`` is not
    read unless ``annotation_key='type'``, and no other column is guessed.
    """
    key = (annotation_key or "").strip()
    had_type = "type" in adata.obs.columns
    info: dict[str, Any] = {"source": "none", "n_types": 1, "n_dropped": 0, "typed": False, "warning": None}
    if key and key in adata.obs.columns:
        col = adata.obs[key]
        # NaN, None and pandas' <NA> (nullable string columns) all become None: the shared helper
        # recognises None and float NaN, but str(pd.NA) is '<NA>', which it would keep as a class.
        labels = col.astype(object).where(col.notna(), None).to_numpy()
        keep, n_dropped = _drop_unlabeled(labels, allow_drop, what=f"{label} observations (obs['{key}'])")
        if n_dropped:
            adata = adata[np.asarray(keep)].copy()
        types = adata.obs[key].astype(str).to_numpy()
        n_types = int(len(pd.unique(types)))
        info.update(source=f"obs['{key}']", n_types=n_types, n_dropped=int(n_dropped), typed=n_types > 1)
        notes = []
        if had_type and key != "type":
            notes.append(f"its existing obs['type'] column was replaced by obs['{key}'] for SpaTrio")
        if n_types < 2:
            notes.append(f"obs['{key}'] holds a single type, so type-aware distances have no effect on this side")
        if n_dropped:
            notes.append(f"{n_dropped} unlabelled observation(s) were left out (drop_unlabeled=True)")
        if notes:
            info["warning"] = f"{label}: " + "; ".join(notes) + "."
        adata.obs["type"] = types
        return adata, info
    adata.obs["type"] = placeholder
    if key:
        msg = (
            f"annotation_key '{key}' is not a column of {label}.obs, so every observation there got the single type "
            f"'{placeholder}' and SpaTrio's type-aware distances were off for that side."
        )
        if had_type:
            msg += " Its existing obs['type'] column was NOT used; pass annotation_key='type' to use it."
        cands = [c for c in _label_candidates(adata) if c != "type"]
        if cands:
            msg += f" Columns that may hold labels: {cands}."
        info["warning"] = msg
        info["source"] = f"none (annotation_key '{key}' absent)"
    else:
        info["source"] = "none (annotation_key empty: untyped by request)"
    return adata, info


def _shared_features(rna_vars, other_vars) -> pd.Index:
    """The features SpaTrio's inner join will keep; zero is refused before the join makes a 0/0 cost."""
    rna_idx = pd.Index(rna_vars)
    other_idx = pd.Index(other_vars)
    shared = rna_idx.intersection(other_idx)
    if len(shared) == 0:
        raise ValueError(
            f"rna_h5ad ({len(rna_idx)} features) and other_h5ad ({len(other_idx)} features) share no feature "
            "names. SpaTrio's expression cost is computed on the features both objects share (it inner-joins "
            f"them), so there is nothing to compare. rna_h5ad features look like {list(map(str, rna_idx[:5]))}; "
            f"other_h5ad features look like {list(map(str, other_idx[:5]))}. other_h5ad must be single-cell data "
            "whose X holds RNA counts under the same gene identifiers as rna_h5ad (both symbols, or both Ensembl "
            "IDs); a protein or ATAC-peak matrix shares none -- keep RNA counts in X and put that modality's "
            "embedding in obsm['reduction']."
        )
    return shared


def _other_reduction(adata_other, counts, random_seed: int):
    """``obsm['reduction']`` for the cell graph: the caller's, if supplied; otherwise one PCA of counts.

    The supplied embedding is SpaTrio's intended input (the second modality) and is never replaced.
    The computed one is normalised exactly once from raw counts -- the old code normalised a copy
    of an already-normalised matrix.
    """
    import scipy.sparse as sps

    if "reduction" in adata_other.obsm:
        val = adata_other.obsm["reduction"]
        arr = val.toarray() if sps.issparse(val) else np.asarray(val)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != adata_other.n_obs or arr.shape[1] < 1:
            raise ValueError(
                f"other_h5ad obsm['reduction'] has shape {arr.shape}; SpaTrio needs a 2-D embedding with one row "
                f"per cell ({adata_other.n_obs})."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("other_h5ad obsm['reduction'] contains NaN/inf; SpaTrio's kNN graph cannot use it.")
        frame = pd.DataFrame(arr, index=adata_other.obs_names.copy())
        return frame, f"supplied obsm['reduction'] ({arr.shape[1]} dims)"

    import anndata as ad
    import scanpy as sc

    n_comps = int(min(MAX_REDUCTION_COMPS, adata_other.n_vars - 1, adata_other.n_obs - 1))
    if n_comps < 2:
        raise ValueError(
            f"other_h5ad is {adata_other.n_obs} x {adata_other.n_vars}: too small for the PCA the worker computes "
            "as obsm['reduction'] when none is supplied. Supply obsm['reduction']."
        )
    proc = ad.AnnData(
        X=counts.copy(),
        obs=pd.DataFrame(index=adata_other.obs_names.copy()),
        var=pd.DataFrame(index=adata_other.var_names.copy()),
    )
    sc.pp.normalize_total(proc, target_sum=1e4)
    sc.pp.log1p(proc)
    sc.tl.pca(proc, n_comps=n_comps, random_state=random_seed)
    frame = pd.DataFrame(np.asarray(proc.obsm["X_pca"]), index=adata_other.obs_names.copy())
    return frame, f"PCA ({n_comps} comps) of other_h5ad RNA counts after one normalize_total(1e4) + log1p"


# ----------------------------------------------------------------------------- memory


def _ot_bytes_lower_bound(n_spots: int, n_cells: int, n_features: int, typed_spots: bool, typed_cells: bool) -> int:
    """A LOWER bound on SpaTrio's peak memory, from the dense intermediates the method itself builds.

    Intrinsic to ``ot_alignment``, and all alive at once: the scaled merged expression matrix
    ((n_spots + n_cells) x shared features, float32 at least), the cost ``M`` (n_spots x n_cells,
    float64), the dense geodesic distances ``Cx`` (n_spots^2) and ``Cy`` (n_cells^2), then either
    the melted (id1, id2, value) table a typed side builds of every pair, or the transport plan and
    its melted (spot, cell, value) table. The real peak is higher (pandas merges, solver
    temporaries); this is only what cannot be avoided.
    """
    n1, n2, f8 = int(n_spots), int(n_cells), 8
    base = 4 * (n1 + n2) * int(n_features) + f8 * n1 * n2
    cx_phase = f8 * n1 * n1 + (3 * f8 * n1 * n1 if typed_spots else 0)
    cy_phase = f8 * (n1 * n1 + n2 * n2) + (3 * f8 * n2 * n2 if typed_cells else 0)
    out_phase = f8 * (n1 * n1 + n2 * n2) + 4 * f8 * n1 * n2
    return int(base + max(cx_phase, cy_phase, out_phase))


def _available_memory_bytes() -> float | None:
    """Memory this process can still allocate, in bytes; None when nothing can be read.

    The shared ``worker_utils.available_memory_bytes``: the smaller of MemAvailable and the room under
    the cgroup memory limit, with the cgroup's page cache counted as reclaimable. This worker used to
    compute ``memory.max - memory.current`` itself; ``memory.current`` counts the page cache, so a
    memory-limited container at its limit on cache alone (reading the two h5ads is enough) was refused
    problems that fit. Kept as a module-level name so the check has one seam.
    """
    return available_memory_bytes()


def _check_memory(n_spots: int, n_cells: int, n_features: int, typed_spots: bool, typed_cells: bool) -> int:
    need = _ot_bytes_lower_bound(n_spots, n_cells, n_features, typed_spots, typed_cells)
    have = _available_memory_bytes()
    if have is not None and need > have:
        gb = 1024.0**3
        raise MemoryError(
            f"SpaTrio needs at least {need / gb:.1f} GB for this input and {have / gb:.1f} GB is available. Its fused "
            f"Gromov-Wasserstein solve holds dense {n_spots} x {n_spots} spot and {n_cells} x {n_cells} cell "
            f"distance matrices and a {n_spots} x {n_cells} transport plan; these are intrinsic to the method and no "
            "parameter of this tool shrinks them (n_hvg is not used, and alpha/numItermax do not change the "
            "problem size). annotation_key='' avoids the extra per-pair table a typed side builds. Otherwise run "
            "on a host with that much free memory."
        )
    return need


# ----------------------------------------------------------------------------- run


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    frame.to_csv(partial)
    os.replace(partial, path)


def run_spatrio(
    rna_h5ad: str,
    other_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    annotation_key: str = "cell_type",
    n_hvg: int = 2000,
    alpha: float = 0.1,
    numItermax: int = 200,
    random_seed: int = 0,
    drop_unlabeled: bool = False,
    layer: str = "",
) -> dict[str, Any]:
    """Run SpaTrio's spot-to-cell optimal-transport mapping."""

    import anndata as ad
    import scanpy as sc
    import spatrio

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha={alpha}: SpaTrio's alpha weighs the graph term against the expression term, 0..1.")

    _ensure_dir(output_dir)
    np.random.seed(random_seed)
    outdir = Path(output_dir)

    # ---- Load both inputs ----
    eprint(f"[SpaTrio] Loading spatial RNA data: {rna_h5ad}")
    adata_rna = sc.read_h5ad(rna_h5ad)
    renamed_rna = make_names_unique_and_report(adata_rna)
    eprint(f"[SpaTrio] Loaded RNA: {adata_rna.n_obs} spots x {adata_rna.n_vars} genes")
    # Background glass (obs['in_tissue'] == 0; CELLxGENE Visium exports carry every array spot) is not
    # tissue. SpaTrio's spot marginal is uniform, so every background spot would receive its share of
    # the cells' mass in the plan. The shared rule leaves it out and says how many.
    n_rna_in = int(adata_rna.n_obs)
    adata_rna, _, n_spots_off_tissue = keep_in_tissue(adata_rna, "spots")
    if n_spots_off_tissue:
        eprint(f"[SpaTrio] Left out {n_spots_off_tissue} of {n_rna_in} spots with obs['in_tissue'] == 0 (background).")

    eprint(f"[SpaTrio] Loading single-cell data: {other_h5ad}")
    adata_other = sc.read_h5ad(other_h5ad)
    renamed_other = make_names_unique_and_report(adata_other)
    eprint(f"[SpaTrio] Loaded other: {adata_other.n_obs} cells x {adata_other.n_vars} features")
    n_other_in = int(adata_other.n_obs)

    if spatial_key not in adata_rna.obsm:
        raise KeyError(
            f"Spatial key '{spatial_key}' not found in RNA h5ad obsm. Available: {list(adata_rna.obsm.keys())}"
        )
    layer = (layer or "").strip()
    if layer and layer not in adata_rna.layers and layer not in adata_other.layers:
        raise KeyError(
            f"layer='{layer}' is in neither input. rna_h5ad layers: {list(adata_rna.layers.keys())}; "
            f"other_h5ad layers: {list(adata_other.layers.keys())}."
        )

    # ---- Types: from annotation_key only, reported per side ----
    adata_rna, types_rna = _resolve_types(adata_rna, annotation_key, "rna_h5ad", "spot", drop_unlabeled)
    adata_other, types_other = _resolve_types(adata_other, annotation_key, "other_h5ad", "cell", drop_unlabeled)
    aware_spatial = bool(types_rna["typed"])
    aware_multi = bool(types_other["typed"])

    # ---- Raw counts: SpaTrio normalises them itself, once ----
    counts_rna, counts_src_rna, _ = _raw_counts(adata_rna, layer, "rna_h5ad")
    counts_other, counts_src_other, _ = _raw_counts(adata_other, layer, "other_h5ad")

    shared = _shared_features(adata_rna.var_names, adata_other.var_names)
    n_shared = int(len(shared))
    eprint(f"[SpaTrio] {n_shared} features shared by both inputs (the expression cost is computed on these)")

    # ---- Coordinates and the cell-graph embedding ----
    spatial_arr, _ = spatial_coords(adata_rna, spatial_key, want=2, tool="SpaTrio")
    reduction, reduction_source = _other_reduction(adata_other, counts_other, random_seed)
    eprint(f"[SpaTrio] Cell graph embedding: {reduction_source}")

    # SpaTrio reads X, obs.index, obs['type'], obsm['spatial'] (adata1) and obsm['reduction'] (adata2),
    # and concatenates the two objects. Handing it exactly those -- and nothing the user's files also
    # carry -- is what keeps that concatenate from failing on mismatched obs/obsm columns.
    lean_rna = ad.AnnData(
        X=counts_rna,
        obs=pd.DataFrame({"type": adata_rna.obs["type"].astype(str).to_numpy()}, index=adata_rna.obs_names.copy()),
        var=pd.DataFrame(index=adata_rna.var_names.copy()),
    )
    lean_rna.obsm["spatial"] = pd.DataFrame(
        np.asarray(spatial_arr, dtype=float), columns=["x", "y"], index=lean_rna.obs_names
    )
    lean_other = ad.AnnData(
        X=counts_other,
        obs=pd.DataFrame({"type": adata_other.obs["type"].astype(str).to_numpy()}, index=adata_other.obs_names.copy()),
        var=pd.DataFrame(index=adata_other.var_names.copy()),
    )
    lean_other.obsm["reduction"] = reduction
    n1, n2 = int(lean_rna.n_obs), int(lean_other.n_obs)

    need = _check_memory(n1, n2, n_shared, aware_spatial, aware_multi)
    eprint(f"[SpaTrio] Dense intermediates need at least {need / 1024.0**3:.2f} GB")

    # ---- Run SpaTrio ----
    eprint(
        f"[SpaTrio] Running ot_alignment (alpha={alpha}, aware_spatial={aware_spatial}, aware_multi={aware_multi}) ..."
    )
    result = spatrio.ot_alignment(
        adata1=lean_rna,
        adata2=lean_other,
        alpha=alpha,
        dissimilarity="scaled_euc",
        numItermax=numItermax,
        aware_spatial=aware_spatial,
        aware_multi=aware_multi,
    )
    eprint("[SpaTrio] Alignment complete.")

    if not isinstance(result, pd.DataFrame) or not {"spot", "cell", "value"} <= set(map(str, result.columns)):
        raise TypeError(
            f"spatrio.ot_alignment returned {type(result).__name__}, not the (spot, cell, value) table it documents; "
            "refusing to publish it."
        )
    n_pairs = int(len(result))

    # ---- Save: the long-format plan, same layout as always (leading row index, then spot,cell,value) ----
    alignment_csv = outdir / "spatrio_aligned.csv"
    _atomic_to_csv(pd.DataFrame(result), alignment_csv)
    eprint(f"[SpaTrio] Saved the {n1} x {n2} transport plan as {n_pairs} (spot, cell, value) rows to {alignment_csv}")

    values = pd.to_numeric(result["value"], errors="coerce").to_numpy(dtype=float)
    n_nonzero = int(np.count_nonzero(values > 0))

    # ---- Build output ----
    out = WorkerOutput("spatrio", task="spatial_alignment")
    out.set_data(
        n_spots_rna=n1,
        n_genes_rna=int(adata_rna.n_vars),
        n_spots_other=n2,
        n_cells=n2,
        n_features_other=int(adata_other.n_vars),
        n_shared_features=n_shared,
        # Kept names, honest values: the plan covers n1 spots, and its cost used the shared features.
        # (They used to hold the rows (n1*n2) and columns (3) of the long table.)
        n_aligned_spots=n1,
        n_aligned_features=n_shared,
        n_pairs=n_pairs,
        n_unlabeled_dropped_rna=int(types_rna["n_dropped"]),
        n_unlabeled_dropped_other=int(types_other["n_dropped"]),
        n_spots_rna_input=n_rna_in,
        n_cells_other_input=n_other_in,
        **identifier_rename_params(renamed_rna, "rna"),
        **identifier_rename_params(renamed_other, "other"),
    )
    # ``alignment_csv`` is the file's real role. ``aligned_h5ad`` is kept as a deprecated alias for
    # callers that key on it: it has always pointed at this CSV, never at an h5ad.
    out.add_output_files({"alignment_csv": str(alignment_csv), "aligned_h5ad": str(alignment_csv)})
    out.add_params(
        {
            "spatial_key": spatial_key,
            "annotation_key": annotation_key,
            "alpha": alpha,
            "numItermax": numItermax,
            "n_hvg": n_hvg,
            "random_seed": random_seed,
            "drop_unlabeled": bool(drop_unlabeled),
            "layer": layer,
            "dissimilarity": "scaled_euc",
            "aware_spatial": aware_spatial,
            "aware_multi": aware_multi,
            "type_source_rna": types_rna["source"],
            "type_source_other": types_other["source"],
            "n_types_rna": int(types_rna["n_types"]),
            "n_types_other": int(types_other["n_types"]),
            "counts_source_rna": counts_src_rna,
            "counts_source_other": counts_src_other,
            "reduction_source": reduction_source,
            "preprocessing": (
                f"inside SpaTrio, once: inner join on the {n_shared} shared features, then normalize_total(1e4) + "
                "log1p + scale on the merged raw counts"
            ),
        }
    )
    record_method(out, METHOD_NAME, used_fallback=False)
    record_in_tissue(out, n_rna_in, n_spots_off_tissue)
    record_ignored(
        out,
        ["n_hvg"],
        "SpaTrio's expression cost uses every feature the two inputs share; no highly-variable-gene selection runs",
    )
    for info in (types_rna, types_other):
        if info["warning"]:
            out.add_warning(info["warning"])
    if n_shared < MIN_SHARED_FEATURES_WARN:
        out.add_warning(
            f"only {n_shared} features are shared by rna_h5ad and other_h5ad (fewer than {MIN_SHARED_FEATURES_WARN}); "
            "the expression cost rests on them alone. Check that both use the same gene identifiers."
        )
    if not reduction_source.startswith("supplied"):
        out.add_warning(
            "other_h5ad has no obsm['reduction'], so the cell graph was built from a PCA of its RNA counts; "
            "SpaTrio's intended input there is an embedding of the second modality (e.g. ATAC LSI, ADT PCA)."
        )
    out.set_summary(
        coupling_shape=[n1, n2],
        coupling_format="long table: one (spot, cell, value) row per spot-cell pair",
        n_pairs=n_pairs,
        n_nonzero_pairs=n_nonzero,
        total_mass=float(np.nansum(values)),
    )
    typed_txt = (
        f"Spot types from {types_rna['source']}: {types_rna['n_types']} (aware_spatial={aware_spatial}); cell types "
        f"from {types_other['source']}: {types_other['n_types']} (aware_multi={aware_multi})."
    )
    tissue_txt = (
        f" {n_spots_off_tissue} of the {n_rna_in} supplied spots have obs['in_tissue'] == 0 (background) and were "
        "left out of the plan."
        if n_spots_off_tissue
        else ""
    )
    out.set_analysis(
        f"SpaTrio (fused Gromov-Wasserstein optimal transport) mapped {n2} single cells onto {n1} spatial spots "
        f"(alpha={alpha}), with an expression cost on the {n_shared} features both inputs share, normalised once "
        f"inside SpaTrio from raw counts. The result is the OT plan over {n1} spots x {n2} cells, written as "
        f"{n_pairs} (spot, cell, value) rows ({n_nonzero} non-zero) to spatrio_aligned.csv -- a spot-to-cell "
        f"mapping table, not aligned coordinates. {typed_txt} Cell graph: {reduction_source}.{tissue_txt}"
        + identifier_rename_note(renamed_rna, "spatial RNA input")
        + identifier_rename_note(renamed_other, "single-cell input")
    )
    return out.to_dict()


def _str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def main():
    ap = argparse.ArgumentParser(description="SpaTrio spot-to-cell optimal-transport worker")
    ap.add_argument("--rna-h5ad", required=True, help="Path to spatial RNA AnnData (.h5ad), raw counts")
    ap.add_argument("--other-h5ad", required=True, help="Path to single-cell (multi-omics) AnnData, raw RNA counts")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--spatial-key", default="spatial", help="obsm key for spatial coordinates")
    ap.add_argument("--annotation-key", default="cell_type", help="obs column for types ('' = untyped)")
    ap.add_argument("--n-hvg", type=int, default=2000, help="Accepted, not used (reported in params.ignored)")
    ap.add_argument("--alpha", type=float, default=0.1, help="FGW trade-off between graph and expression, 0..1")
    ap.add_argument("--numItermax", type=int, default=200, help="Max iterations for OT solver")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (numpy and the reduction PCA)")
    ap.add_argument(
        "--drop-unlabeled",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Leave out observations whose annotation_key label is NaN/empty instead of refusing",
    )
    ap.add_argument("--layer", default="", help="Layer holding raw counts, read instead of X where present")
    args = ap.parse_args()

    # Redirect stdout to stderr to keep stdout clean for JSON output
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        result = run_spatrio(
            rna_h5ad=args.rna_h5ad,
            other_h5ad=args.other_h5ad,
            output_dir=args.output_dir,
            spatial_key=args.spatial_key,
            annotation_key=args.annotation_key,
            n_hvg=args.n_hvg,
            alpha=args.alpha,
            numItermax=args.numItermax,
            random_seed=args.seed,
            drop_unlabeled=args.drop_unlabeled,
            layer=args.layer,
        )
    except Exception as e:
        eprint(f"[SpaTrio] ERROR: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.stdout = _real_stdout
        WorkerOutput.emit_error("spatrio", str(e), task="spatial_alignment")
        sys.exit(1)
    finally:
        sys.stdout = _real_stdout

    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
