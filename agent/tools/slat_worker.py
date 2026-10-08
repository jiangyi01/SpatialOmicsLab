#!/usr/bin/env python
"""
scSLAT worker: spatial alignment of two tissue slices via graph neural networks.

- Runs inside /opt/conda/envs/slat
- All logs/progress go to stderr.
- stdout is reserved for a single final JSON line.

Pipeline:
  1. Load two h5ad slices (raw counts expected in X)
  2. Restrict both slices to the genes they share
  3. Build spatial neighbor graphs (Cal_Spatial_Net)
  4. Extract features via load_anndatas -- scSLAT normalises, selects genes and reduces there
     itself, from the counts; nothing is preprocessed beforehand
  5. Train the SLAT model with run_SLAT
  6. Match every spot of the smaller slice to its most similar spot in the larger one
     (spatial_match, cosine nearest neighbour)
  7. Save matching, embeddings, and visualization

Two things this worker used to do and no longer does, because each produced a wrong result that
the payload did not mention:

* It ran ``scanpy_workflow`` (normalize_total, log1p, scale) on each slice before
  ``load_anndatas``. The DPCA and PCA feature steps then ran scanpy's ``seurat_v3`` gene selection
  on scaled, negative values and failed ("Extrapolation not allowed with blending") on every
  input, and a silent fallback chain ended on the dense all-gene matrix while the payload still
  said "DPCA".
* It wrote spatial_match's result as ``slice1_idx = arange, slice2_idx = best``. spatial_match
  indexes the larger slice and queries it with the smaller, so whenever slice 1 had at least as
  many spots as slice 2 the two columns held each other's contents.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback

# Ensure worker_utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    record_method,
)

# Force non-interactive matplotlib backend
os.environ["MPLBACKEND"] = "Agg"

import numpy as np

try:
    import anndata as ad  # noqa: F401
    import scanpy as sc
except Exception as e:
    print("[slat-worker] ERROR: Failed to import scanpy/anndata.", file=sys.stderr)
    print(str(e), file=sys.stderr)
    WorkerOutput.emit_error("slat", "Failed to import scanpy/anndata: " + str(e), task="import")
    sys.exit(1)

try:
    from scSLAT.model import Cal_Spatial_Net, load_anndatas, run_SLAT, spatial_match
except Exception as e:
    print("[slat-worker] ERROR: Failed to import scSLAT.", file=sys.stderr)
    print(str(e), file=sys.stderr)
    WorkerOutput.emit_error("slat", "Failed to import scSLAT: " + str(e), task="import")
    sys.exit(1)


FEATURE_TYPES = ("DPCA", "PCA", "HVG")

# What allow_feature_fallback lets the run continue on when the requested feature type fails.
# Only consulted when the caller opted in; by default the requested type runs or the run stops.
FEATURE_FALLBACKS = {
    "DPCA": ["PCA", "raw"],
    "PCA": ["raw"],
    "HVG": ["PCA", "raw"],
}

FEATURE_DESCRIPTIONS = {
    "DPCA": "dual PCA (50 dims) over up to 12,000 seurat_v3 highly variable genes",
    "PCA": "joint PCA (50 dims) over up to 2,500 seurat_v3 highly variable genes, normalised and scaled",
    "HVG": "the scaled expression of up to 2,500 seurat_v3 highly variable genes",
    "raw": "the expression matrix X as stored, every shared gene, dense",
}

# Hardcoded in scSLAT's load_anndatas(feature='DPCA'): sc.pp.highly_variable_genes(n_top_genes=12000).
DPCA_TOP_GENES = 12000
# load_anndatas' n_top_genes default, handed to scanpy_workflow for 'PCA' and 'HVG'. seurat_v3 marks
# exactly min(n_top_genes, n_genes) genes, and sc.tl.pca reads that mask by default.
PCA_TOP_GENES = 2500

# How many of the shared genes each feature type selects before building features (None = all).
FEATURE_TOP_GENES = {"DPCA": DPCA_TOP_GENES, "PCA": PCA_TOP_GENES, "HVG": PCA_TOP_GENES, "raw": None}


def _genes_used(feature, n_shared):
    """The number of shared genes scSLAT builds ``feature`` from: its seurat_v3 selection, or all."""
    cap = FEATURE_TOP_GENES.get(feature)
    return int(n_shared) if cap is None else min(int(cap), int(n_shared))


UPSTREAM_HVG_BUG = (
    "scSLAT's load_anndatas(feature='HVG') loops `for i in len(adatas)` and raises TypeError on "
    "every input (scSLAT 0.3.0), so feature_type='HVG' cannot run. Use feature_type='DPCA' (the "
    "scSLAT default) or 'PCA'."
)


def _log(msg):
    print(f"[slat-worker] {msg}", file=sys.stderr)


def _gib(n_bytes):
    return f"{n_bytes / float(1 << 30):.2f} GiB"


def _memory_available_bytes():
    """Memory this process can still allocate, in bytes, or None when it cannot be read.

    The shared ``worker_utils.available_memory_bytes``: the smaller of MemAvailable and the room under
    the cgroup memory limit, with the cgroup's page cache counted as reclaimable. This worker used to
    read ``memory.max - memory.current`` itself; ``memory.current`` counts the page cache, so a
    memory-limited container sitting at its limit on cache alone (reading the two h5ads is enough)
    was refused runs that fit. Kept as a module-level name so the floor check has one seam.
    """
    return available_memory_bytes()


def _dense_floor_bytes(feature, n1, n2, n_genes):
    """A lower bound on the dense intermediates scSLAT allocates for one feature type.

    These matrices are intrinsic to the method (scSLAT densifies them itself), so the worker does
    not avoid them; it refuses before allocating when even this floor does not fit.

    * raw: ``X.todense()`` plus its float32 tensor copy, per slice.
    * DPCA: both slices scaled dense over up to 12,000 genes, copied to tensors, plus the
      n1 x n2 cross-product ``X @ Y.T`` that dual PCA decomposes.
    * PCA / HVG: ``scanpy_workflow`` scales the concatenated slices dense over every shared gene.
    """
    n1, n2, n_genes = int(n1), int(n2), int(n_genes)
    if feature == "raw":
        return (n1 + n2) * n_genes * 8
    if feature == "DPCA":
        return (n1 + n2) * min(n_genes, DPCA_TOP_GENES) * 8 + n1 * n2 * 4
    if feature in ("PCA", "HVG"):
        return (n1 + n2) * n_genes * 4
    return 0


def _check_dense_fits(feature, n1, n2, n_genes):
    need = _dense_floor_bytes(feature, n1, n2, n_genes)
    avail = _memory_available_bytes()
    if avail is None or need <= avail:
        return
    if feature == "DPCA":
        what = (
            f"two dense scaled matrices over up to {DPCA_TOP_GENES} genes and the {n1} x {n2} "
            "cross-product that dual PCA decomposes"
        )
        knob = "feature_type='PCA' has no n1 x n2 matrix"
    elif feature == "raw":
        what = f"the dense {n1} x {n_genes} and {n2} x {n_genes} expression matrices"
        knob = "feature_type='DPCA' or 'PCA' builds 50-dimensional features instead"
    else:
        what = f"the concatenated ({n1} + {n2}) x {n_genes} matrix, scaled dense"
        knob = f"feature_type='DPCA' caps the genes at {DPCA_TOP_GENES}"
    raise MemoryError(
        f"scSLAT '{feature}' features need at least {_gib(need)} for {what}; {_gib(avail)} is "
        f"available. {knob}, or run on a machine with more memory."
    )


def _background_spots(adata, which):
    """How many spots of slice ``which`` ``obs['in_tissue']`` marks as background (0: none, or no column).

    Counted with the shared ``keep_in_tissue`` rule, applied to the obs column alone. They are not
    dropped here: ``slat_matching.csv`` holds 0-based row positions of the two INPUT files, one row per
    spot of the smaller one, and the 3D adapter (``spatialomicsgym.spatial3d.adapters._read_slat``)
    refuses a matching whose row count is not ``min(n1, n2)`` of those files. So they are embedded and
    matched like tissue, and the payload says how many there were.
    """
    if "in_tissue" not in adata.obs.columns:
        return 0
    import anndata

    probe = anndata.AnnData(obs=adata.obs[["in_tissue"]].copy())
    try:
        _, _, n_off = keep_in_tissue(probe, what=f"spots of slice {which}")
    except ValueError:  # the column marks no spot of this slice as in tissue
        return int(adata.n_obs)
    return int(n_off)


def _coords(adata, which):
    """obsm['spatial'] as a float array, written back so scSLAT and the plot read the same thing."""
    if "spatial" not in adata.obsm:
        raise KeyError(
            f"slice {which} has no obsm['spatial']; scSLAT builds its neighbour graph from the spot "
            f"coordinates. obsm keys present: {list(adata.obsm.keys())}"
        )
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"slice {which}: obsm['spatial'] has shape {coords.shape}, not (n_spots, 2)")
    n_bad = int((~np.isfinite(coords)).any(axis=1).sum())
    if n_bad:
        raise ValueError(
            f"slice {which}: {n_bad} of {coords.shape[0]} spots have non-finite coordinates in "
            "obsm['spatial']; the neighbour graph cannot place them."
        )
    adata.obsm["spatial"] = coords
    return coords


def _counts_problem(X, chunk_rows=4096):
    """Why X does not look like raw counts, or '' when it does. Reads every value, chunked."""
    import scipy.sparse as sp

    if sp.issparse(X):
        data = X.data
        step = 10_000_000
        blocks = (data[i : i + step] for i in range(0, data.shape[0], step))
    else:
        blocks = (np.asarray(X[i : i + chunk_rows]) for i in range(0, X.shape[0], chunk_rows))
    for block in blocks:
        if block.size == 0:
            continue
        if np.any(block < 0):
            return "X holds negative values"
        if np.any(block != np.floor(block)):
            return "X holds non-integer values"
    return ""


def _extract_features(adata1, adata2, feature_type, allow_feature_fallback):
    """Run load_anndatas for the requested feature type, and for the fallbacks only when allowed.

    Returns ``(feature_used, edges, features, failures)`` where ``failures`` lists
    ``(feature, error)`` for every type tried before the one that ran.
    """
    attempts = [feature_type]
    if allow_feature_fallback:
        attempts += FEATURE_FALLBACKS.get(feature_type, [])
    failures = []
    for ft in attempts:
        _log(f"  Building '{ft}' features...")
        try:
            _check_dense_fits(ft, adata1.n_obs, adata2.n_obs, adata1.n_vars)
            edges, features = load_anndatas([adata1, adata2], feature=ft, check_order=False)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            _log(f"  '{ft}' features failed: {reason}")
            failures.append((ft, reason))
            continue
        return ft, edges, features, failures

    first_ft, first_reason = failures[0]
    msg = f"scSLAT could not build '{first_ft}' features: {first_reason}."
    if first_ft == "HVG" and first_reason.startswith("TypeError"):
        msg += " " + UPSTREAM_HVG_BUG
    elif first_ft in ("DPCA", "PCA") and not first_reason.startswith("MemoryError"):
        msg += " DPCA and PCA select genes with scanpy's seurat_v3 flavour (scikit-misc), which needs raw counts in X."
    if allow_feature_fallback:
        tried = ", ".join(f"'{ft}' ({reason})" for ft, reason in failures[1:])
        msg += f" The fallbacks allowed by allow_feature_fallback=True failed too: {tried}."
    else:
        chain = FEATURE_FALLBACKS.get(first_ft, [])
        msg += (
            " No other feature type was tried. Pass allow_feature_fallback=True to let the run continue "
            f"on {' then '.join(repr(c) for c in chain)}; the payload then names the one that ran."
        )
    raise RuntimeError(msg)


def _to_numpy(t):
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _match(embd1, embd2, n1, n2):
    """Match each spot of the smaller slice to its most similar spot in the larger slice.

    scSLAT's own rule (spatial_match with reorder=True) is to index the larger slice and query it
    with the smaller; with equal sizes slice 1 is indexed. The rule is applied here explicitly,
    with reorder=False, so which slice the rows enumerate is decided in this file and labelled
    accordingly rather than inferred from upstream's internals.

    Returns ``(match_df, query_slice, reference_slice)``; the frame has one row per query spot.
    """
    import pandas as pd

    if n1 < n2:
        query, reference, n_query, n_ref = 1, 2, n1, n2
        embds = [embd2, embd1]  # index slice 2, query with slice 1
    else:
        query, reference, n_query, n_ref = 2, 1, n2, n1
        embds = [embd1, embd2]  # index slice 1, query with slice 2
    best, _order, _similarity = spatial_match(embds, reorder=False, smooth=False)
    best = np.asarray(best)
    if best.ndim != 1 or best.shape[0] != n_query:
        raise RuntimeError(
            f"spatial_match returned a matching of shape {best.shape}; expected one entry per spot of "
            f"slice {query} ({n_query} spots)"
        )
    best = best.astype(np.int64)
    if best.size and (best.min() < 0 or best.max() >= n_ref):
        raise RuntimeError(
            f"spatial_match returned indices outside slice {reference} (0..{n_ref - 1}): "
            f"min {int(best.min())}, max {int(best.max())}"
        )
    enum = np.arange(n_query, dtype=np.int64)
    if query == 2:
        cols = {"slice1_idx": best, "slice2_idx": enum}
    else:
        cols = {"slice1_idx": enum, "slice2_idx": best}
    return pd.DataFrame(cols, columns=["slice1_idx", "slice2_idx"]), query, reference


def _write_csv_atomic(df, path):
    partial = path + ".partial"
    df.to_csv(partial, index=False)
    os.replace(partial, path)


def _write_npz_atomic(path, **arrays):
    partial = path + ".partial"
    # A file handle, not a name: np.savez appends ".npz" to a name that lacks it.
    with open(partial, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(partial, path)


def _plot(adata1, adata2, match_df, plot_path):
    import matplotlib.pyplot as plt

    coords1 = adata1.obsm["spatial"]
    coords2 = adata2.obsm["spatial"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].scatter(coords1[:, 0], coords1[:, 1], s=1, c="blue", alpha=0.5)
    axes[0].set_title(f"Slice 1 ({adata1.n_obs} spots)")
    axes[0].invert_yaxis()
    axes[1].scatter(coords2[:, 0], coords2[:, 1], s=1, c="red", alpha=0.5)
    axes[1].set_title(f"Slice 2 ({adata2.n_obs} spots)")
    axes[1].invert_yaxis()

    # Draw at most 200 of the pairs so the lines stay legible; the CSV holds every pair.
    n_pairs = len(match_df)
    n_show = min(200, n_pairs)
    rng = np.random.default_rng(0)
    shown = rng.choice(n_pairs, n_show, replace=False) if n_pairs > n_show else np.arange(n_pairs)
    i1 = match_df["slice1_idx"].to_numpy()[shown]
    i2 = match_df["slice2_idx"].to_numpy()[shown]
    for a, b in zip(i1, i2):
        axes[2].plot(
            [coords1[a, 0], coords2[b, 0]],
            [coords1[a, 1], coords2[b, 1]],
            "gray",
            alpha=0.1,
            linewidth=0.5,
        )
    axes[2].scatter(coords1[:, 0], coords1[:, 1], s=1, c="blue", alpha=0.3, label="slice 1")
    axes[2].scatter(coords2[:, 0], coords2[:, 1], s=1, c="red", alpha=0.3, label="slice 2")
    axes[2].set_title(f"Matching ({n_show} of {n_pairs} pairs drawn)")
    axes[2].legend(markerscale=5)
    axes[2].invert_yaxis()
    plt.tight_layout()
    partial = plot_path + ".partial"
    fig.savefig(partial, dpi=150, bbox_inches="tight", format="png")
    plt.close(fig)
    os.replace(partial, plot_path)


def run_align_slices(
    h5ad_path_1, h5ad_path_2, output_dir, k_cutoff, feature_type, epochs, allow_feature_fallback=False
):
    """Full scSLAT alignment pipeline for two spatial slices."""
    _log("Starting scSLAT alignment pipeline")
    _log(f"h5ad_path_1 = {h5ad_path_1}")
    _log(f"h5ad_path_2 = {h5ad_path_2}")
    _log(f"output_dir  = {output_dir}")
    _log(
        f"k_cutoff={k_cutoff}, feature_type={feature_type}, epochs={epochs}, "
        f"allow_feature_fallback={allow_feature_fallback}"
    )
    if feature_type not in FEATURE_TYPES:
        raise ValueError(f"feature_type={feature_type!r} is not one of {list(FEATURE_TYPES)}")

    out_dir = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Output paths
    matching_csv = os.path.join(out_dir, "slat_matching.csv")
    embeddings_npz = os.path.join(out_dir, "slat_embeddings.npz")
    plot_path = os.path.join(out_dir, "slat_alignment_plot.png")

    warnings_out = []

    # Redirect stdout to stderr during computation (scSLAT prints progress)
    with contextlib.redirect_stdout(sys.stderr):
        # Step 1: Load data
        _log("Loading slice 1...")
        adata1 = sc.read_h5ad(h5ad_path_1)
        _log(f"Slice 1: {adata1.n_obs} cells/spots, {adata1.n_vars} genes")

        _log("Loading slice 2...")
        adata2 = sc.read_h5ad(h5ad_path_2)
        _log(f"Slice 2: {adata2.n_obs} cells/spots, {adata2.n_vars} genes")

        # Unique names on both axes. Spot names matter here: Cal_Spatial_Net stores the graph by
        # obs name and load_anndatas maps names back to rows, so a duplicated barcode handed every
        # edge of its first occurrence to its last. The outputs are positional, so nothing the
        # user joins on is renamed.
        renamed1 = make_names_unique_and_report(adata1)
        renamed2 = make_names_unique_and_report(adata2)
        n_genes_1, n_genes_2 = int(adata1.n_vars), int(adata2.n_vars)

        # Background glass (obs['in_tissue'] == 0) is counted, not dropped: see _background_spots.
        n_off_tissue = {1: _background_spots(adata1, 1), 2: _background_spots(adata2, 2)}
        for which, adata in ((1, adata1), (2, adata2)):
            if n_off_tissue[which]:
                warnings_out.append(
                    f"slice {which}: {n_off_tissue[which]} of {adata.n_obs} spots have obs['in_tissue'] == 0 "
                    "(background outside the tissue) and were embedded and matched like tissue spots. They are "
                    "kept because slat_matching.csv indexes the rows of the input files, one row per spot of the "
                    "smaller slice; remove them from the h5ad beforehand to align the tissue alone."
                )

        _coords(adata1, 1)
        _coords(adata2, 2)

        # Step 2: the genes both slices measure, in one order. DPCA and PCA inner-join the slices by
        # name themselves; doing it here as well gives 'raw' features the same columns, in the same
        # order, on both sides.
        shared = adata1.var_names.intersection(adata2.var_names)
        overlap_genes = int(len(shared))
        if overlap_genes == 0:
            raise ValueError(
                "the two slices share no gene names (slice 1 e.g. "
                f"{list(adata1.var_names[:3])}, slice 2 e.g. {list(adata2.var_names[:3])}); scSLAT "
                "builds its features from the genes both slices measure"
            )
        if not adata1.var_names.equals(shared):
            adata1 = adata1[:, shared].copy()
        if not adata2.var_names.equals(shared):
            adata2 = adata2[:, shared].copy()
        _log(f"Using the {overlap_genes} genes the slices share")

        for which, adata in ((1, adata1), (2, adata2)):
            problem = _counts_problem(adata.X)
            if problem:
                warnings_out.append(
                    f"slice {which}: {problem}. scSLAT's feature steps normalise and log-transform X "
                    "themselves and select genes with seurat_v3, which expects raw counts; an X that "
                    "is already normalised is transformed twice."
                )

        # Step 3: Build spatial neighbor graphs
        _log(f"Building spatial graph for slice 1 (k_cutoff={k_cutoff})...")
        Cal_Spatial_Net(adata1, k_cutoff=k_cutoff, model="KNN")

        _log(f"Building spatial graph for slice 2 (k_cutoff={k_cutoff})...")
        Cal_Spatial_Net(adata2, k_cutoff=k_cutoff, model="KNN")

        # Step 4: Extract features from the counts; scSLAT preprocesses inside load_anndatas.
        _log(f"Extracting features (feature_type={feature_type})...")
        feature_used, edges, features, failures = _extract_features(
            adata1, adata2, feature_type, allow_feature_fallback
        )
        _log(f"Feature dimensions: {tuple(features[0].shape)}, {tuple(features[1].shape)}")
        _log(f"Edge counts: {[tuple(e.shape) for e in edges]}")

        # Step 5: Build and train model using run_SLAT (high-level API)
        feature_dim = int(features[0].shape[1])
        _log(f"Feature dim={feature_dim}, training SLAT for {epochs} epochs...")
        embd0, embd1, run_time = run_SLAT(features, edges, epochs=epochs)
        _log(f"Embedding shapes: embd0={tuple(embd0.shape)}, embd1={tuple(embd1.shape)}, time={run_time:.1f}s")

        # Step 6: Spatial matching
        _log("Computing spatial matching from embeddings...")
        match_df, query_slice, reference_slice = _match(embd0, embd1, int(adata1.n_obs), int(adata2.n_obs))

    _write_csv_atomic(match_df, matching_csv)
    _log(f"Saved matching to {matching_csv}")

    _write_npz_atomic(
        embeddings_npz,
        embedding_slice1=_to_numpy(embd0),
        embedding_slice2=_to_numpy(embd1),
    )
    _log(f"Saved embeddings to {embeddings_npz}")

    try:
        _plot(adata1, adata2, match_df, plot_path)
        _log(f"Saved alignment plot to {plot_path}")
    except Exception as plot_err:
        _log(f"Warning: alignment plot failed: {plot_err}")
        warnings_out.append(f"alignment plot was not written: {type(plot_err).__name__}: {plot_err}")
        plot_path = None

    n_matched = int(len(match_df))
    ref_col = "slice1_idx" if reference_slice == 1 else "slice2_idx"
    n_unique_ref = int(match_df[ref_col].nunique())
    used_fallback = feature_used != feature_type
    n_genes_used = _genes_used(feature_used, overlap_genes)
    if n_genes_used < overlap_genes:
        genes_phrase = (
            f"the {n_genes_used} most variable (seurat_v3, selected by scSLAT) of the {overlap_genes} "
            "genes the slices share"
        )
    else:
        genes_phrase = f"all {overlap_genes} genes the slices share"

    out = WorkerOutput("slat", task="align_slices")
    out.set_data(
        n_cells_slice1=int(adata1.n_obs),
        n_cells_slice2=int(adata2.n_obs),
        n_genes_slice1=n_genes_1,
        n_genes_slice2=n_genes_2,
        n_overlap_genes=overlap_genes,
    )
    out.add_output_files(
        {
            "matching_csv": matching_csv,
            "embeddings_npz": embeddings_npz,
            "alignment_plot_png": str(plot_path) if plot_path else None,
        }
    )
    out.add_params(
        {
            "h5ad_path_1": h5ad_path_1,
            "h5ad_path_2": h5ad_path_2,
            "k_cutoff": k_cutoff,
            "feature_type": feature_used,
            "feature_type_requested": feature_type,
            "epochs": epochs,
            "allow_feature_fallback": bool(allow_feature_fallback),
            "n_genes_shared": overlap_genes,
            "n_genes_used": n_genes_used,
            "matching_query_slice": query_slice,
            "matching_reference_slice": reference_slice,
            "n_spots_off_tissue_slice1": int(n_off_tissue[1]),
            "n_spots_off_tissue_slice2": int(n_off_tissue[2]),
        }
    )
    out.add_params(identifier_rename_params(renamed1, "slice1"))
    out.add_params(identifier_rename_params(renamed2, "slice2"))
    method = (
        f"scSLAT run_SLAT on '{feature_used}' features ({FEATURE_DESCRIPTIONS[feature_used]}), "
        "matched by spatial_match (cosine nearest neighbour, no spatial smoothing)"
    )
    why = ""
    if used_fallback:
        why = "; ".join(f"'{ft}' failed: {reason}" for ft, reason in failures)
        out.add_param("feature_failures", [{"feature_type": ft, "error": reason} for ft, reason in failures])
    record_method(out, method, used_fallback=used_fallback, why=why)
    out.add_warnings(warnings_out)
    out.set_summary(
        n_matched_pairs=n_matched,
        n_unique_targets=n_unique_ref,
        query_slice=query_slice,
        reference_slice=reference_slice,
        embedding_dim=int(embd0.shape[1]) if hasattr(embd0, "shape") else None,
        feature_dim=feature_dim,
    )
    size_rule = (
        "the smaller slice"
        if int(adata1.n_obs) != int(adata2.n_obs)
        else "the slices are the same size, and scSLAT then enumerates slice 2"
    )
    analysis = (
        f"scSLAT embedded {adata1.n_obs} spots (slice 1) and {adata2.n_obs} spots (slice 2) from "
        f"'{feature_used}' features ({feature_dim} dims, built from {genes_phrase}) with {epochs} "
        f"training epochs. Each of the {n_matched} spots of slice {query_slice} "
        f"({size_rule}) was matched to its most similar spot in slice {reference_slice}; "
        f"{n_unique_ref} distinct slice-{reference_slice} spots were used."
    )
    if n_off_tissue[1] or n_off_tissue[2]:
        analysis += (
            f" Background spots (obs['in_tissue'] == 0) were aligned as tissue: {n_off_tissue[1]} in slice 1 and "
            f"{n_off_tissue[2]} in slice 2."
        )
    if used_fallback:
        analysis += (
            f" The requested '{feature_type}' features could not be built, so "
            f"'{feature_used}' ran instead (allow_feature_fallback=True): {why}."
        )
    analysis += identifier_rename_note(renamed1, "slice 1") + identifier_rename_note(renamed2, "slice 2")
    out.set_analysis(analysis)
    return out.to_dict()


def main():
    parser = argparse.ArgumentParser(
        description="scSLAT worker: spatial alignment of tissue slices via graph neural networks."
    )
    parser.add_argument("--h5ad-path-1", required=True, help="Path to first spatial AnnData (.h5ad)")
    parser.add_argument("--h5ad-path-2", required=True, help="Path to second spatial AnnData (.h5ad)")
    parser.add_argument("--output-dir", required=True, help="Directory to store outputs")
    parser.add_argument(
        "--k-cutoff",
        type=int,
        default=10,
        help="Number of neighbors for spatial graph (default 10).",
    )
    parser.add_argument(
        "--feature-type",
        default="DPCA",
        choices=list(FEATURE_TYPES),
        help="Feature type for load_anndatas (default DPCA). HVG fails in scSLAT 0.3.0 (upstream bug).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs for the GAN model (default 80).",
    )
    parser.add_argument(
        "--allow-feature-fallback",
        action="store_true",
        help=(
            "When the requested feature type fails, continue on the next one (DPCA -> PCA -> raw, "
            "PCA -> raw, HVG -> PCA -> raw) and name it in params.method. Off by default: the run stops."
        ),
    )

    args = parser.parse_args()

    try:
        result = run_align_slices(
            h5ad_path_1=args.h5ad_path_1,
            h5ad_path_2=args.h5ad_path_2,
            output_dir=args.output_dir,
            k_cutoff=args.k_cutoff,
            feature_type=args.feature_type,
            epochs=args.epochs,
            allow_feature_fallback=args.allow_feature_fallback,
        )
        # Single JSON line to stdout
        print(json.dumps(result, default=str))
        sys.stdout.flush()
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("slat", str(e), task="align_slices")
        sys.exit(1)


if __name__ == "__main__":
    main()
