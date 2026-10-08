#!/usr/bin/env python
"""
SpatialGlue worker: multi-omics spatial integration.

- Runs inside /opt/conda/envs/spatialglue
- All logs/progress go to stderr.
- stdout is reserved for a single final JSON line.

Workflow (upstream SpatialGlue ``Train_SpatialGlue``; the preprocessing is the upstream tutorials'):
  1. Load the two modalities. The same path for both means one combined 10x file (for example a
     CytAssist h5ad with Gene Expression and Antibody Capture in one X); it is split by
     ``var['feature_types']``. A file carrying other feature types keeps only its own modality.
  2. Pair spots by barcode -- never by row position -- and seed numpy/torch/random (``fix_seed``).
  3. RNA: filter_genes(min_cells=10), 3000 seurat_v3 HVGs on counts, normalize_total(1e4), log1p,
     scale, PCA. Protein: CLR, scale, PCA(n_proteins - 1). ATAC/epigenome: LSI.
  4. Construct spatial and feature neighbour graphs (upstream ``construct_neighbor_graph``).
  5. Train with the datatype's upstream weight preset and the caller's epochs.
  6. Cluster on the combined embedding; save results.

Before any data is read, the chosen ``cluster_method`` is run once on a 40-point toy embedding
through the same clustering call the real run makes: 'louvain' needs the ``louvain`` package
(scanpy's default 'vtraag' flavour) and 'mclust' needs rpy2 and an R with the mclust package. A
method that cannot run in this environment is refused up front, naming what is missing, instead of
failing after the full training run.

Matrices: the RNA recipe (seurat_v3 on counts, normalize_total, log1p) and protein CLR read counts.
``worker_utils.choose_counts_matrix`` decides which matrix of each input is read: with
``use_raw_counts`` an input that carries ``adata.raw`` is read from it (refused when that is not
counts) and one without keeps X, with a note; otherwise a negative or non-finite X is refused naming
``use_raw_counts`` when ``adata.raw`` holds counts. SpatialGlue is stricter than the shared rule in
one place, by design: a non-negative non-integer X is refused too (its recipes need integer counts),
with the same hint. The epigenome modality (LSI, or a precomputed ``obsm['X_lsi']``) is not a counts
recipe and its X is read as supplied.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import (  # noqa: I001
    WorkerOutput,
    available_memory_bytes,
    build_cluster_analysis,
    choose_counts_matrix,
    ensure_r_home,
    expression_matrix_kind,
    id_mismatch_msg,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    require_hvg_flavor,
    resolve_compute,
    unsupported_choice_msg,
)

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from SpatialGlue.preprocess import clr_normalize_each_cell, construct_neighbor_graph, fix_seed, lsi, pca
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue

#: What actually runs: the upstream trainer, not a reimplementation.
METHOD_NAME = "SpatialGlue (upstream Train_SpatialGlue)"

#: Upstream ``Train_SpatialGlue`` applies a datatype preset (epochs + loss weight factors) only on an
#: exact key match: 'SPOTS', 'Stereo-CITE-seq', '10x', 'Spatial-epigenome-transcriptome'
#: (SpatialGlue_pyG.py). The documented default here was '10X', which matched none of them, so every
#: default run trained with the constructor's generic weights [1, 5, 1, 1] while the docs promised
#: the 10x preset. Matching is case-insensitive onto the upstream key; anything else is refused.
DATATYPES = {
    "10x": "10x",
    "spots": "SPOTS",
    "stereo-cite-seq": "Stereo-CITE-seq",
    "spatial-epigenome-transcriptome": "Spatial-epigenome-transcriptome",
}

#: What the second modality is under each preset. The upstream tutorials CLR-normalise protein and
#: run LSI on the epigenome/ATAC modality; CLR on peaks was never the method.
OMICS2_KIND = {
    "10x": "protein",
    "SPOTS": "protein",
    "Stereo-CITE-seq": "protein",
    "Spatial-epigenome-transcriptome": "epigenome",
}

#: 10x ``var['feature_types']`` vocabulary. A combined file carries several of these in one X.
RNA_FEATURE_TYPES = ("Gene Expression",)
OMICS2_FEATURE_TYPES = {"protein": ("Antibody Capture",), "epigenome": ("Peaks",)}
TENX_FEATURE_TYPES = frozenset(
    {"Gene Expression", "Antibody Capture", "Peaks", "CRISPR Guide Capture", "Multiplexing Capture", "Custom"}
)

#: The upstream tutorials' RNA recipe.
RNA_MIN_CELLS = 10
RNA_N_TOP_GENES = 3000
RNA_TARGET_SUM = 1e4
#: The epigenome tutorial reduces RNA to 50 PCs and ATAC to 50 LSI components (51, first dropped).
EPIGENOME_N_COMPS = 50

#: Upstream ``construct_graph_by_feature`` links every spot to k=20 feature neighbours; it is not
#: a parameter upstream, so ``n_neighbors`` never reached it.
FEATURE_GRAPH_K = 20

#: The clustering methods offered, and what each needs beyond scanpy (named when it cannot run).
CLUSTER_METHOD_NEEDS = {
    "leiden": "the leidenalg package (scanpy's sc.tl.leiden)",
    "louvain": "the louvain package (scanpy's sc.tl.louvain, default 'vtraag' flavour)",
    "mclust": "rpy2 and an R installation with the mclust package (SpatialGlue.utils.mclust_R)",
}
#: Size of the toy embedding the chosen clustering method is tried on before any data is read.
CLUSTER_PROBE_POINTS = 40

#: ``adjacent_matrix_preprocessing`` densifies both spatial graphs (float64 n x n each), symmetrises
#: them (another n x n float64 per graph) and turns both feature graphs into dense float32 tensors
#: through a float64 ``toarray()``: about four float64 n x n matrices at the peak.
DENSE_SPOT_PAIR_BYTES = 4 * 8


def log(msg):
    print(f"[spatialglue-worker] {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- inputs


def canonical_datatype(datatype) -> str:
    """The upstream preset key for ``datatype``; refuse a value no preset answers to."""
    key = str(datatype).strip().lower()
    if key not in DATATYPES:
        raise ValueError(
            unsupported_choice_msg(
                "datatype",
                datatype,
                # Read off the table the lookup uses, so the offer cannot drift from what is accepted.
                list(dict.fromkeys(DATATYPES.values())),
                extra="Matching is case-insensitive ('10X' selects upstream's '10x' preset). Any other value "
                "used to train with no preset at all while the payload named one.",
            )
        )
    return DATATYPES[key]


def _feature_types(adata):
    if "feature_types" not in adata.var.columns:
        return None
    return adata.var["feature_types"].astype(str)


def select_modality(adata, wanted, arg: str):
    """Keep the features of one modality when ``var['feature_types']`` says the file holds several.

    Returns ``(adata, dropped)`` where ``dropped`` is ``{feature_type: n_features}`` left out. A file
    with no ``feature_types`` column, or with only this modality's types, is returned as is. A file
    whose types are all 10x types and none of them this modality's is refused: it is the other
    modality, and integrating it would pair a matrix with a relabelled copy of itself.
    """
    ft = _feature_types(adata)
    if ft is None:
        return adata, {}
    counts = ft.value_counts()
    present = [str(t) for t in counts.index]
    hits = [t for t in present if t in wanted]
    others = [t for t in present if t not in wanted]
    if not others:
        return adata, {}
    if hits:
        keep = ft.isin(hits).to_numpy()
        return adata[:, keep].copy(), {t: int(counts[t]) for t in others}
    if len(present) > 1 or all(t in TENX_FEATURE_TYPES for t in present):
        # Either the other modality (all 10x types, none of ours) or a mixed file whose blocks this
        # worker cannot tell apart. A single label of the user's own is trusted as this modality.
        found = {t: int(counts[t]) for t in present}
        raise ValueError(
            f"{arg} holds no {list(wanted)} features: its var['feature_types'] is {found}. Pass the file of "
            f"that modality, or pass one combined 10x file as both rna_h5ad and protein_h5ad to have it split."
        )
    return adata, {}


def read_counts_input(adata, arg: str, use_raw_counts: bool, counts_recipe: bool = True):
    """``(adata, info)``: the matrix of one input file SpatialGlue reads, and what was decided.

    ``worker_utils.choose_counts_matrix`` applies the shared rule. As celldart does with the same
    parameter, ``use_raw_counts`` reads ``adata.raw`` of an input that has one (refused when it is not
    counts) and keeps X of one that has none, with ``info['note']`` saying so. ``info['raw_hint']`` is
    the advice the stricter non-integer refusals in :func:`rna_features` / :func:`protein_features`
    append. ``counts_recipe=False`` (the epigenome modality: LSI or a precomputed ``obsm['X_lsi']``)
    reads X as supplied unless ``use_raw_counts`` asks for ``adata.raw``.
    """
    raw = getattr(adata, "raw", None)
    if use_raw_counts and raw is None:
        info = {"expression_source": "X", "x_matrix_kind": None, "warning": None, "raw_hint": ""}
        info["note"] = f"use_raw_counts=True, but {arg} has no adata.raw, so its X was read."
        return adata, info
    if not counts_recipe and not use_raw_counts:
        return adata, {"expression_source": "X", "x_matrix_kind": None, "warning": None, "raw_hint": "", "note": None}
    try:
        chosen, info = choose_counts_matrix(adata, use_raw_counts=use_raw_counts)
    except ValueError as exc:
        raise ValueError(f"{arg}: {exc}") from exc
    info = dict(info, note=None, raw_hint="")
    if info["expression_source"] == "X" and info["x_matrix_kind"] != "counts" and raw is not None:
        if expression_matrix_kind(raw.X) == "counts":
            info["raw_hint"] = " Its adata.raw holds raw counts: pass use_raw_counts=True to read them."
    return chosen, info


def load_modalities(rna_h5ad: str, protein_h5ad: str, omics2_kind: str, use_raw_counts: bool = False):
    """Read both modalities, splitting a combined file. Returns ``(omics1, omics2, report)``.

    ``report['counts_omics1']`` / ``report['counts_omics2']`` are :func:`read_counts_input`'s decisions.
    """
    same_file = os.path.realpath(str(rna_h5ad)) == os.path.realpath(str(protein_h5ad))
    report = {}
    if same_file:
        combined = sc.read_h5ad(rna_h5ad)
        combined, counts = read_counts_input(combined, "rna_h5ad (= protein_h5ad)", use_raw_counts)
        report["counts_omics1"] = report["counts_omics2"] = counts
        if _feature_types(combined) is None:
            raise ValueError(
                f"rna_h5ad and protein_h5ad both name {rna_h5ad}, and it has no var['feature_types'] column "
                "to split it by. SpatialGlue would integrate that matrix with a copy of itself. Pass one file "
                "per modality, or a combined 10x file whose var['feature_types'] marks 'Gene Expression' and "
                "'Antibody Capture' (or 'Peaks')."
            )
        omics1 = combined
        omics2 = combined
        report["input_mode"] = "one combined file, split by var['feature_types']"
    else:
        omics1, report["counts_omics1"] = read_counts_input(sc.read_h5ad(rna_h5ad), "rna_h5ad", use_raw_counts)
        omics2, report["counts_omics2"] = read_counts_input(
            sc.read_h5ad(protein_h5ad), "protein_h5ad", use_raw_counts, counts_recipe=(omics2_kind == "protein")
        )
        report["input_mode"] = "two files"
    n_features_file1, n_features_file2 = int(omics1.n_vars), int(omics2.n_vars)

    omics1, dropped1 = select_modality(omics1, RNA_FEATURE_TYPES, "rna_h5ad")
    omics2, dropped2 = select_modality(omics2, OMICS2_FEATURE_TYPES[omics2_kind], "protein_h5ad")
    if same_file and not (dropped1 and dropped2):
        raise ValueError(
            f"rna_h5ad and protein_h5ad both name {rna_h5ad}, and its var['feature_types'] does not separate a "
            f"'Gene Expression' block from a {list(OMICS2_FEATURE_TYPES[omics2_kind])} block, so there is "
            "nothing to split. Pass one file per modality."
        )
    if omics1 is omics2 or list(map(str, omics1.var_names)) == list(map(str, omics2.var_names)):
        raise ValueError(
            "rna_h5ad and protein_h5ad hold the same features, so the 'second modality' is a copy of the first. "
            "Pass one file per modality, or one combined 10x file as both paths to have it split by "
            "var['feature_types']."
        )
    report.update(
        {
            "n_features_rna_h5ad": n_features_file1,
            "n_features_protein_h5ad": n_features_file2,
            "omics1_feature_types_dropped": dropped1,
            "omics2_feature_types_dropped": dropped2,
        }
    )
    return omics1, omics2, report


def leave_out_background(omics1, omics2):
    """Leave out of both modalities every spot either one marks ``obs['in_tissue'] == 0``.

    ``worker_utils.keep_in_tissue`` decides what is background -- the rule the spot-analysing workers
    share: background is left out by default and reported. A CELLxGENE-style export keeps every array
    spot, and those spots were integrated and clustered as tissue. A spot flagged in one file and not
    in the other is still background, so it leaves both, and the pairing below does not count it as a
    barcode the other modality lacks. Returns ``(omics1, omics2, report)``; the counts are per modality.
    """
    background = []
    sources = []
    for arg, adata in (("rna_h5ad", omics1), ("protein_h5ad", omics2)):
        kept, _, n_dropped = keep_in_tissue(adata, "spots")
        if n_dropped:
            sources.append(arg)
            background.extend(map(str, adata.obs_names[~adata.obs_names.isin(kept.obs_names)]))
    report = {
        "n_supplied_omics1": int(omics1.n_obs),
        "n_dropped_omics1": 0,
        "n_dropped_omics2": 0,
        "flag_source": sources,
    }
    if not background:
        return omics1, omics2, report
    off = pd.Index(background).unique()
    in1 = np.asarray(omics1.obs_names.isin(off))
    in2 = np.asarray(omics2.obs_names.isin(off))
    if in1.all():
        raise ValueError(
            f"every one of the {omics1.n_obs} spots of rna_h5ad is marked obs['in_tissue'] == 0 in "
            f"{' or '.join(sources)}; fix the column so in-tissue spots are 1, or remove it if every spot is tissue."
        )
    report["n_dropped_omics1"] = int(in1.sum())
    report["n_dropped_omics2"] = int(in2.sum())
    if in1.any():
        omics1 = omics1[~in1].copy()
    if in2.any():
        omics2 = omics2[~in2].copy()
    return omics1, omics2, report


def pair_by_barcode(omics1, omics2):
    """Align both modalities on shared barcodes, in omics1's order. Returns ``(omics1, omics2, report)``.

    Upstream pairs the two feature matrices row by row. The old code intersected barcodes only when
    the row counts differed, so two files with equal counts in a different order were paired by
    position -- every spot integrated with some other spot's protein.
    """
    for arg, adata in (("rna_h5ad", omics1), ("protein_h5ad", omics2)):
        if not adata.obs_names.is_unique:
            dup = adata.obs_names[adata.obs_names.duplicated()]
            raise ValueError(
                f"{arg} has {len(dup)} duplicated barcode(s) (e.g. {list(map(str, dup[:3]))}); spots are paired "
                "between the two modalities by barcode, so every barcode must be unique."
            )
    n1, n2 = int(omics1.n_obs), int(omics2.n_obs)
    common = omics1.obs_names[omics1.obs_names.isin(omics2.obs_names)]
    if len(common) == 0:
        raise ValueError(id_mismatch_msg("cell/spot barcodes", "omics1", omics1.obs_names, "omics2", omics2.obs_names))
    omics2_order = omics2.obs_names[omics2.obs_names.isin(common)]
    reordered = not omics2_order.equals(common)
    if len(common) < n1:
        omics1 = omics1[common].copy()
    if len(common) < n2 or reordered:
        omics2 = omics2[common].copy()
    report = {
        "spot_pairing": "by barcode",
        "n_spots_omics1_supplied": n1,
        "n_spots_omics2_supplied": n2,
        "n_spots_paired": int(len(common)),
        "omics2_rows_reordered": bool(reordered),
    }
    return omics1, omics2, report


def count_violation(X, chunk_rows: int = 2000):
    """None when ``X`` holds finite, non-negative integers; else what is wrong with it. Sparse-aware."""
    import scipy.sparse as sp

    for start in range(0, int(X.shape[0]), chunk_rows):
        block = X[start : start + chunk_rows]
        values = np.asarray(block.data if sp.issparse(block) else block).ravel()
        if values.size == 0:
            continue
        if not np.all(np.isfinite(values)):
            return "non-finite values"
        if np.any(values < 0):
            return f"negative values (min {float(values.min()):.4g})"
        off = np.abs(values - np.round(values)) > 1e-6
        if np.any(off):
            return f"non-integer values (e.g. {float(values[off][0]):.4g})"
    return None


def _memory_available_bytes():
    """Memory this process can still allocate, or None when the platform cannot say.

    ``worker_utils.available_memory_bytes``: the smaller of MemAvailable and the room under the cgroup
    limit, page cache counted as reclaimable. This used to read MemAvailable alone, which in a container
    reports the host's memory rather than the limit the run is killed at.
    """
    return available_memory_bytes()


def omics2_dense_width(adata, omics2_kind: str) -> int:
    """Columns of the second modality that are ever held dense, one row per spot.

    Protein: every feature -- ``protein_features`` densifies X for CLR. Epigenome: a precomputed
    ``obsm['X_lsi']`` is used as is, and upstream ``lsi`` keeps a sparse peak matrix sparse through
    tfidf (``X.multiply``), l1 normalisation and log1p, so only its LSI components are ever dense. A
    peak matrix that is already dense is copied densely by each of those steps, so every peak counts.
    The budget used to count every peak for the epigenome kind whatever the input, which put a typical
    9,215-spot x 121,068-peak slide at about 19.6 GiB where its dense footprint is about 2.9 GiB.
    """
    import scipy.sparse as sp

    if omics2_kind == "protein":
        return int(adata.n_vars)
    if "X_lsi" in adata.obsm:
        return int(adata.obsm["X_lsi"].shape[1])
    if sp.issparse(adata.X):
        return int(min(EPIGENOME_N_COMPS + 1, int(adata.n_vars)))
    return int(adata.n_vars)


def dense_budget_bytes(n_spots: int, n_features: int) -> int:
    """Bytes of the dense intermediates upstream SpatialGlue materialises for ``n_spots`` spots.

    Intrinsic to the method: the n x n adjacency matrices of ``adjacent_matrix_preprocessing`` (see
    ``DENSE_SPOT_PAIR_BYTES``), plus the dense feature matrices ``pca`` and ``scale`` work on
    (``n_features`` columns over both modalities: the RNA HVGs and ``omics2_dense_width``).
    """
    n = int(n_spots)
    return DENSE_SPOT_PAIR_BYTES * n * n + 16 * n * int(n_features)


def check_dense_budget(n_spots: int, n_features: int, available=None):
    """Refuse up front, with the numbers, a slide whose dense n x n graphs cannot fit in memory."""
    if available is None:
        available = _memory_available_bytes()
    need = dense_budget_bytes(n_spots, n_features)
    if available is not None and need > available:
        gib = float(1 << 30)
        raise MemoryError(
            f"SpatialGlue densifies its spot-by-spot graphs ({n_spots}x{n_spots} spatial and feature adjacency "
            f"for both modalities) and {n_features} feature columns per spot: about {need / gib:.1f} GiB for "
            f"{n_spots} paired spots, and this machine reports {available / gib:.1f} GiB available. No parameter "
            "of this tool lowers that footprint; run it on a machine with more memory. The slide is analysed "
            "whole, never subsampled."
        )
    return need


# --------------------------------------------------------------------------- features


def rna_features(adata, n_comps_target: int, raw_hint: str = ""):
    """The upstream tutorial's RNA recipe on a working copy. Returns ``(features, report)``.

    The old code ran PCA straight on raw counts, up to 3000 components. The tutorial selects 3000
    seurat_v3 HVGs on counts, normalises, log-transforms and scales, then keeps a handful of PCs.
    Scaling is per gene, so scaling the HVG subset equals the tutorial's scale-then-subset without
    densifying the other genes. ``adata`` itself keeps its counts: it is the integrated h5ad.
    """
    problem = count_violation(adata.X)
    if problem is not None:
        hint = " A layers['counts'] is present: put it in X." if "counts" in adata.layers else ""
        raise ValueError(
            f"rna_h5ad X holds {problem}, not raw counts. SpatialGlue's RNA recipe selects seurat_v3 highly "
            f"variable genes on counts and then normalises and log-transforms, so it needs raw counts in X.{hint}"
            f"{raw_hint}"
        )
    work = adata.copy()
    n_genes_supplied = int(work.n_vars)
    sc.pp.filter_genes(work, min_cells=RNA_MIN_CELLS)
    n_after_filter = int(work.n_vars)
    if n_after_filter == 0:
        raise ValueError(f"no gene of rna_h5ad is detected in {RNA_MIN_CELLS} or more spots; nothing to embed.")
    n_top = min(RNA_N_TOP_GENES, n_after_filter)
    # No hvg_flavor parameter on this portal (hunt 2026-09-30, u29a-mcp-transport-15).
    require_hvg_flavor("seurat_v3", alternative=None)
    try:
        sc.pp.highly_variable_genes(work, flavor="seurat_v3", n_top_genes=n_top)
    except Exception as exc:
        # Narrow on purpose, and no flavour switch: seurat_v3 on counts is the recipe. It imports
        # scikit-misc (skmisc.loess) lazily, and a build of it compiled against another numpy
        # raises ValueError there rather than ImportError -- so say which package, and stop.
        raise RuntimeError(
            f"highly_variable_genes(flavor='seurat_v3') failed ({type(exc).__name__}: {exc}). That flavour needs "
            "scikit-misc (import name skmisc) importable in this environment; SpatialGlue's RNA recipe does not "
            "switch to another flavour."
        ) from exc
    sc.pp.normalize_total(work, target_sum=RNA_TARGET_SUM)
    sc.pp.log1p(work)
    work = work[:, work.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(work)
    n_comps = min(int(n_comps_target), int(work.n_vars), int(work.n_obs) - 1)
    feat = np.asarray(pca(work, n_comps=n_comps))
    report = {
        "omics1_preprocessing": [
            f"filter_genes(min_cells={RNA_MIN_CELLS})",
            f"highly_variable_genes(flavor='seurat_v3', n_top_genes={n_top}) on counts",
            f"normalize_total(target_sum={RNA_TARGET_SUM:g})",
            "log1p",
            "scale (HVG subset)",
            f"PCA(n_comps={n_comps})",
        ],
        "n_genes_omics1_supplied": n_genes_supplied,
        "n_genes_omics1_after_filter": n_after_filter,
        "n_hvg_omics1": int(work.n_vars),
        "n_comps_omics1": int(n_comps),
        "n_comps_omics1_target": int(n_comps_target),
    }
    return feat, report


def protein_features(adata, raw_hint: str = ""):
    """CLR, scale, PCA(n_proteins - 1), as the upstream protein tutorials do. Returns ``(features, report)``.

    ``adata.X`` is densified first: upstream ``clr_normalize_each_cell`` returns a dense matrix anyway,
    and its sparse branch calls ``X.A``, which scipy >= 1.14 removed -- so every sparse input used to die
    there with ``'csr_matrix' object has no attribute 'A'``.
    """
    import scipy.sparse as sp

    problem = count_violation(adata.X)
    if problem is not None:
        raise ValueError(
            f"protein_h5ad X holds {problem}, not raw antibody counts. SpatialGlue CLR-normalises the protein "
            "modality, so a matrix that was already normalised would be transformed twice. Supply raw ADT counts."
            + raw_hint
        )
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    adata.X = np.asarray(X, dtype=np.float64)
    adata = clr_normalize_each_cell(adata)
    sc.pp.scale(adata)
    n_comps = min(int(adata.n_vars) - 1, int(adata.n_obs) - 1)
    if n_comps < 2:
        raise ValueError(
            f"protein_h5ad has {adata.n_vars} feature(s); SpatialGlue embeds the protein modality with "
            "PCA(n_proteins - 1) and its feature graph needs at least 2 components, so at least 3 proteins."
        )
    feat = np.asarray(pca(adata, n_comps=n_comps))
    report = {
        "omics2_preprocessing": ["CLR (per spot)", "scale", f"PCA(n_comps={n_comps})"],
        "n_comps_omics2": int(n_comps),
    }
    return adata, feat, report


def epigenome_features(adata):
    """LSI, as the upstream epigenome tutorial does (a precomputed obsm['X_lsi'] is used as is)."""
    if "X_lsi" in adata.obsm:
        feat = np.asarray(adata.obsm["X_lsi"])
        return (
            adata,
            feat,
            {"omics2_preprocessing": ["precomputed obsm['X_lsi']"], "n_comps_omics2": int(feat.shape[1])},
        )
    problem = count_violation(adata.X)
    if problem is not None and not problem.startswith("non-integer"):
        raise ValueError(f"protein_h5ad (the epigenome modality) X holds {problem}; LSI needs non-negative counts.")
    n_components = min(EPIGENOME_N_COMPS + 1, int(adata.n_obs) - 1, int(adata.n_vars) - 1)
    lsi(adata, use_highly_variable=False, n_components=n_components)
    feat = np.asarray(adata.obsm["X_lsi"])
    report = {
        "omics2_preprocessing": [f"LSI(n_components={n_components}, first component dropped)"],
        "n_comps_omics2": int(feat.shape[1]),
    }
    return adata, feat, report


def _effective_spatial_k(adata):
    """Spatial neighbours per spot that upstream actually used, read off the graph it built."""
    adj = adata.uns.get("adj_spatial")
    if adj is None or not int(adata.n_obs):
        return None
    return int(round(len(adj) / float(adata.n_obs)))


def _effective_feature_k(adata):
    """Feature-graph neighbours per spot, read off the kNN graph upstream built (k=20 upstream)."""
    graph = adata.obsm["adj_feature"] if "adj_feature" in adata.obsm else None
    if graph is None or not hasattr(graph, "nnz") or not int(adata.n_obs):
        return None
    return int(round(graph.nnz / float(adata.n_obs)))


# --------------------------------------------------------------------------- clustering


def cluster_embedding(adata, cluster_method: str, n_clusters: int, resolution: float, use_rep: str = "SpatialGlue"):
    """Cluster ``adata.obsm[use_rep]`` with ``cluster_method``; returns ``(adata, cluster_key)``.

    The one clustering call site: the real run and :func:`clustering_probe_failure` both go through it,
    so the up-front check exercises exactly what the run will call.
    """
    sc.pp.neighbors(adata, use_rep=use_rep)
    if cluster_method == "mclust":
        # The literal that used to be here named a directory this env does not ship, and
        # rpy2 only falls back to its own detection when R_HOME is *unset* -- so the guess
        # actively suppressed the one thing that would have worked.
        ensure_r_home()
        from SpatialGlue.utils import mclust_R

        adata = mclust_R(adata, used_obsm=use_rep, num_cluster=n_clusters)
        return adata, "mclust"
    if cluster_method == "louvain":
        sc.tl.louvain(adata, resolution=resolution)
        return adata, "louvain"
    sc.tl.leiden(adata, resolution=resolution)
    return adata, "leiden"


def canonical_cluster_method(cluster_method) -> str:
    """``cluster_method`` lower-cased; refuse a method this worker does not offer."""
    key = str(cluster_method).strip().lower()
    if key not in CLUSTER_METHOD_NEEDS:
        raise ValueError(unsupported_choice_msg("cluster_method", cluster_method, list(CLUSTER_METHOD_NEEDS)))
    return key


def clustering_probe_failure(cluster_method: str, resolution: float, seed: int = 2022):
    """None when ``cluster_method`` runs here, else why not.

    Runs :func:`cluster_embedding` on a toy embedding of two well-separated groups: the same imports
    (louvain, rpy2 + R mclust, leidenalg) the real call makes, before any data is read or trained on.
    """
    rng = np.random.default_rng(int(seed))
    half = CLUSTER_PROBE_POINTS // 2
    emb = np.vstack([rng.standard_normal((half, 2)), rng.standard_normal((half, 2)) + 8.0])
    toy = sc.AnnData(X=np.zeros((emb.shape[0], 1), dtype=np.float32))
    toy.obsm["probe"] = emb
    try:
        with contextlib.redirect_stdout(sys.stderr):
            cluster_embedding(toy, cluster_method, n_clusters=2, resolution=resolution, use_rep="probe")
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def check_cluster_method(cluster_method: str, resolution: float, seed: int = 2022) -> None:
    """Refuse, before loading or training, a clustering method that cannot run in this environment."""
    why = clustering_probe_failure(cluster_method, resolution, seed)
    if why is not None:
        raise RuntimeError(
            f"cluster_method={cluster_method!r} cannot run in this environment: it needs "
            f"{CLUSTER_METHOD_NEEDS[cluster_method]}, and the same clustering call failed on a "
            f"{CLUSTER_PROBE_POINTS}-point probe ({why}). The run was stopped before any data was read or any "
            "model trained. Install what is missing in the SpatialGlue environment, or pass a cluster_method "
            f"whose dependencies are installed (one of {list(CLUSTER_METHOD_NEEDS)})."
        )


# --------------------------------------------------------------------------- outputs


def _write_csv_atomic(df, path) -> None:
    """``<path>.partial`` then ``os.replace``: a reader never sees a half-written table."""
    tmp = f"{path}.partial"
    df.to_csv(tmp)
    os.replace(tmp, str(path))


def _write_h5ad_atomic(adata, path) -> None:
    tmp = f"{path}.partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, str(path))


# --------------------------------------------------------------------------- run


def run_spatialglue_integration(
    rna_h5ad,
    protein_h5ad,
    output_dir,
    datatype="10X",
    epochs=600,
    dim_output=64,
    n_clusters=7,
    n_neighbors=3,
    cluster_method="leiden",
    resolution=1.0,
    device="auto",
    seed=2022,
    use_raw_counts=False,
):
    """Run SpatialGlue multi-omics integration."""
    log("Task = multi-omics integration")
    log(f"RNA h5ad      = {rna_h5ad}")
    log(f"Protein h5ad  = {protein_h5ad}")
    log(f"Output dir    = {output_dir}")
    preset = canonical_datatype(datatype)
    omics2_kind = OMICS2_KIND[preset]
    log(f"datatype={datatype} (upstream preset {preset!r}, omics2 = {omics2_kind}), epochs={epochs}, seed={seed}")
    # A clustering method whose packages are missing used to fail only after the full training run.
    cluster_method = canonical_cluster_method(cluster_method)
    check_cluster_method(cluster_method, resolution, seed)

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with contextlib.redirect_stdout(sys.stderr):
        # ---- Load data ----
        adata_omics1, adata_omics2, input_report = load_modalities(
            rna_h5ad, protein_h5ad, omics2_kind, use_raw_counts=bool(use_raw_counts)
        )
        counts1 = input_report.pop("counts_omics1")
        counts2 = input_report.pop("counts_omics2")
        renamed1 = make_names_unique_and_report(adata_omics1, axes=("var",))
        renamed2 = make_names_unique_and_report(adata_omics2, axes=("var",))

        log(f"omics1 (RNA): n_spots={adata_omics1.n_obs}, n_genes={adata_omics1.n_vars}")
        log(f"omics2 ({omics2_kind}): n_spots={adata_omics2.n_obs}, n_features={adata_omics2.n_vars}")

        if "spatial" not in adata_omics1.obsm:
            raise ValueError(f"RNA h5ad {rna_h5ad} is missing obsm['spatial'].")
        if "spatial" not in adata_omics2.obsm:
            raise ValueError(f"Protein h5ad {protein_h5ad} is missing obsm['spatial'].")

        # ---- Background spots (obs['in_tissue'] == 0) are left out and reported ----
        adata_omics1, adata_omics2, background = leave_out_background(adata_omics1, adata_omics2)
        if background["n_dropped_omics1"] or background["n_dropped_omics2"]:
            log(
                f"Left out {background['n_dropped_omics1']} omics1 and {background['n_dropped_omics2']} omics2 "
                f"background spot(s) (obs['in_tissue'] == 0 in {background['flag_source']})"
            )

        # ---- Pair spots by barcode, always ----
        adata_omics1, adata_omics2, pairing = pair_by_barcode(adata_omics1, adata_omics2)
        n_paired = pairing["n_spots_paired"]
        log(f"Paired {n_paired} spots by barcode (omics2 reordered: {pairing['omics2_rows_reordered']})")
        if n_paired <= FEATURE_GRAPH_K:
            raise ValueError(
                f"only {n_paired} spots are shared by the two modalities; SpatialGlue's feature graph links every "
                f"spot to {FEATURE_GRAPH_K} neighbours (fixed upstream), so it needs more than {FEATURE_GRAPH_K}."
            )
        check_dense_budget(
            n_paired,
            min(RNA_N_TOP_GENES, int(adata_omics1.n_vars)) + omics2_dense_width(adata_omics2, omics2_kind),
        )

        # ---- Seed before anything random (PCA solver, model init, training) ----
        fix_seed(int(seed))

        # ---- Preprocessing ----
        if omics2_kind == "protein":
            adata_omics2, feat2, omics2_report = protein_features(adata_omics2, raw_hint=counts2["raw_hint"])
            n_comps1_target = int(adata_omics2.n_vars) - 1
        else:
            adata_omics2, feat2, omics2_report = epigenome_features(adata_omics2)
            n_comps1_target = EPIGENOME_N_COMPS
        feat1, omics1_report = rna_features(adata_omics1, n_comps1_target, raw_hint=counts1["raw_hint"])
        adata_omics1.obsm["feat"] = feat1
        adata_omics2.obsm["feat"] = feat2
        log(f"Features: omics1 {feat1.shape[1]} PCs of {omics1_report['n_hvg_omics1']} HVGs; omics2 {feat2.shape[1]}")

        # ---- Construct neighbor graphs ----
        log(f"Constructing neighbor graphs (datatype={preset}, n_neighbors={n_neighbors})")
        data = construct_neighbor_graph(
            adata_omics1,
            adata_omics2,
            datatype=preset,
            n_neighbors=n_neighbors,
        )
        spatial_k = _effective_spatial_k(adata_omics1)
        feature_k = _effective_feature_k(adata_omics1)

        # ---- Train SpatialGlue ----
        # Was `torch.device("cpu")`, hardcoded: this GNN training loop could not use a GPU on any
        # hardware, ever, whatever the caller asked for. `resolve_compute` still returns "cpu" on a
        # box without CUDA, so the CPU-only behaviour is unchanged where there is nothing else.
        resolved = resolve_compute(device).device
        log(f"Using device: {resolved}")
        torch_device = torch.device(resolved)
        model = Train_SpatialGlue(
            data,
            datatype=preset,
            device=torch_device,
            random_seed=int(seed),
            epochs=int(epochs),
            dim_input=int(feat1.shape[1]),
            dim_output=dim_output,
        )
        # Upstream's preset overwrites `epochs` inside __init__ (10x 200, SPOTS 600, Stereo-CITE-seq
        # 1500, Spatial-epigenome-transcriptome 1600). The caller's value is the documented contract,
        # so it is put back; the preset's value is published beside it.
        epochs_preset = int(getattr(model, "epochs", epochs))
        model.epochs = int(epochs)
        weight_factors = [float(w) for w in getattr(model, "weight_factors", [])]
        log(f"Training SpatialGlue (epochs={epochs}, preset would train {epochs_preset}; weights={weight_factors})")
        output = model.train()

    # ---- Extract results ----
    emb_combined = output["SpatialGlue"]
    alpha_weights = output.get("alpha", None)

    log(f"Combined embedding shape: {emb_combined.shape}")

    # Store embedding in the RNA AnnData for clustering
    adata_omics1.obsm["SpatialGlue"] = emb_combined

    # Save embedding CSV
    emb_df = pd.DataFrame(
        emb_combined,
        index=adata_omics1.obs_names,
        columns=[f"SG_{i}" for i in range(emb_combined.shape[1])],
    )
    emb_csv = str(out_dir / "spatialglue_embedding.csv")
    _write_csv_atomic(emb_df, emb_csv)
    log(f"Saved embedding to {emb_csv}")

    # Save attention weights if available
    alpha_csv = None
    if alpha_weights is not None:
        alpha_df = pd.DataFrame(
            alpha_weights,
            index=adata_omics1.obs_names,
            columns=["alpha_omics1", "alpha_omics2"],
        )
        alpha_csv = str(out_dir / "spatialglue_attention_weights.csv")
        _write_csv_atomic(alpha_df, alpha_csv)
        log(f"Saved attention weights to {alpha_csv}")

    # ---- Clustering ----
    log(f"Clustering with method={cluster_method}, n_clusters={n_clusters}, resolution={resolution}")

    with contextlib.redirect_stdout(sys.stderr):
        adata_omics1, cluster_key = cluster_embedding(
            adata_omics1, cluster_method, n_clusters=n_clusters, resolution=resolution
        )

    # Save clusters
    cluster_csv = str(out_dir / "spatialglue_clusters.csv")
    _write_csv_atomic(adata_omics1.obs[[cluster_key]], cluster_csv)
    log(f"Saved clusters to {cluster_csv}")

    # Save integrated h5ad
    out_h5ad = str(out_dir / "spatialglue_integrated.h5ad")
    _write_h5ad_atomic(adata_omics1, out_h5ad)
    log(f"Saved integrated h5ad to {out_h5ad}")

    # ---- Build worker output ----
    cluster_sizes = adata_omics1.obs[cluster_key].value_counts().to_dict()
    n_spots = int(adata_omics1.n_obs)

    out = WorkerOutput("spatialglue", task="multi_omics_integration")
    out.set_data(
        n_spots=n_spots,
        n_genes_omics1=int(adata_omics1.n_vars),
        n_features_omics2=int(adata_omics2.n_vars),
    )
    output_files = {
        "integrated_h5ad": out_h5ad,
        "embedding_csv": emb_csv,
        "clusters_csv": cluster_csv,
    }
    if alpha_csv is not None:
        output_files["attention_weights_csv"] = alpha_csv
    out.add_output_files(output_files)
    out.add_params(
        {
            "datatype": preset,
            "datatype_requested": str(datatype),
            "omics2_kind": omics2_kind,
            "epochs": int(epochs),
            "epochs_preset": epochs_preset,
            "weight_factors": weight_factors,
            "dim_output": dim_output,
            "n_clusters": n_clusters,
            "n_neighbors": n_neighbors,
            "n_neighbors_spatial_effective": spatial_k,
            "feature_graph_k": feature_k,
            "cluster_method": cluster_method,
            "resolution": resolution,
            "seed": int(seed),
            "device": resolved,
            "use_raw_counts": bool(use_raw_counts),
            "expression_source_omics2": counts2["expression_source"],
        }
    )
    record_expression_source(out, counts1)
    # A combined file shares one decision between the two modalities; say it once.
    for note in dict.fromkeys(c["note"] for c in (counts1, counts2) if c.get("note")):
        out.add_warning(note)
    out.add_params(input_report)
    out.add_params(pairing)
    out.add_params(omics1_report)
    out.add_params(omics2_report)
    out.add_params(identifier_rename_params(renamed1))
    out.add_params(identifier_rename_params(renamed2, suffix="omics2"))
    record_method(out, METHOD_NAME)
    record_in_tissue(out, background["n_supplied_omics1"], background["n_dropped_omics1"])
    out.set_data(n_spots_off_tissue_dropped=background["n_dropped_omics1"])
    if background["n_dropped_omics1"] or background["n_dropped_omics2"]:
        out.add_params(
            {
                "in_tissue_flag_source": background["flag_source"],
                "n_spots_omics2_off_tissue_dropped": background["n_dropped_omics2"],
            }
        )

    if spatial_k is not None and int(spatial_k) != int(n_neighbors):
        record_ignored(
            out,
            "n_neighbors",
            f"upstream construct_neighbor_graph fixes the spatial graph at {spatial_k} neighbours for "
            f"datatype={preset!r}; the feature graph always uses k={FEATURE_GRAPH_K}",
        )
    if cluster_method == "mclust":
        record_ignored(
            out, "resolution", "mclust fits n_clusters components; resolution is read only by leiden/louvain"
        )
    else:
        record_ignored(
            out,
            "n_clusters",
            f"cluster_method={cluster_method!r} runs at the fixed resolution={resolution}; n_clusters is read only "
            "by mclust",
        )
    if epochs_preset != int(epochs):
        out.add_info(
            f"upstream's {preset!r} preset trains {epochs_preset} epochs; this run trained {int(epochs)}, as requested"
        )
    for side, dropped in (
        ("rna_h5ad", input_report["omics1_feature_types_dropped"]),
        ("protein_h5ad", input_report["omics2_feature_types_dropped"]),
    ):
        if dropped:
            out.add_warning(f"{side}: left out features of another modality by var['feature_types']: {dropped}")
    n_left_out = (pairing["n_spots_omics1_supplied"] - n_paired, pairing["n_spots_omics2_supplied"] - n_paired)
    if any(n_left_out):
        out.add_warning(
            f"{n_left_out[0]} omics1 spot(s) and {n_left_out[1]} omics2 spot(s) have no barcode in the other "
            f"modality and were left out; {n_paired} spots were paired by barcode."
        )
    out.set_summary(
        n_clusters=len(cluster_sizes),
        cluster_sizes=cluster_sizes,
        embedding_dim=emb_combined.shape[1],
    )

    # Mean attention weights per modality if available
    if alpha_weights is not None:
        mean_alpha = alpha_weights.mean(axis=0)
        out.set_summary(
            mean_attention_omics1=float(mean_alpha[0]),
            mean_attention_omics2=float(mean_alpha[1]),
        )

    method_note = (
        f" Method: {METHOD_NAME}, datatype preset {preset!r} (loss weights {weight_factors}), {int(epochs)} epochs, "
        f"seed {int(seed)}. {n_paired} spots paired by barcode. RNA features: {omics1_report['n_hvg_omics1']} "
        f"seurat_v3 highly variable genes of the {omics1_report['n_genes_omics1_after_filter']} detected in "
        f"{RNA_MIN_CELLS}+ spots, normalised, log-transformed and scaled, "
        f"reduced to {omics1_report['n_comps_omics1']} PCs; {omics2_kind} features: "
        f"{', '.join(omics2_report['omics2_preprocessing'])}."
    )
    if background["n_dropped_omics1"]:
        method_note += (
            f" NOTE: {background['n_dropped_omics1']} of the {background['n_supplied_omics1']} RNA spots are marked "
            "obs['in_tissue'] == 0 (background outside the tissue) and were left out."
        )
    raw_read = [
        side for side, counts in (("RNA", counts1), (omics2_kind, counts2)) if counts["expression_source"] != "X"
    ]
    if raw_read:
        method_note += f" Counts were read from adata.raw for the {' and '.join(dict.fromkeys(raw_read))} input."

    requested = int(n_clusters) if cluster_method == "mclust" else None
    out.set_analysis(
        build_cluster_analysis(cluster_sizes, cluster_key="domain", total_spots=n_spots, n_requested=requested)
        + method_note
        + identifier_rename_note(renamed1, "RNA genes")
        + identifier_rename_note(renamed2, f"{omics2_kind} features")
    )
    return out.to_dict()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SpatialGlue worker: multi-omics spatial integration.")
    parser.add_argument(
        "--rna-h5ad",
        required=True,
        help="Path to RNA/transcriptomics AnnData (.h5ad); raw counts in X.",
    )
    parser.add_argument(
        "--protein-h5ad",
        required=True,
        help="Path to protein/second-modality AnnData (.h5ad). The same path as --rna-h5ad splits one "
        "combined 10x file by var['feature_types'].",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store outputs.",
    )
    parser.add_argument(
        "--datatype",
        default="10X",
        help="Upstream preset, case-insensitive: '10x' (also '10X'), 'SPOTS', 'Stereo-CITE-seq', "
        "'Spatial-epigenome-transcriptome'.",
    )
    parser.add_argument("--epochs", type=int, default=600, help="Training epochs (always honoured).")
    parser.add_argument("--dim-output", type=int, default=64, help="Output embedding dimension.")
    parser.add_argument("--n-clusters", type=int, default=7, help="Number of clusters (mclust only).")
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=3,
        help="Spatial-graph neighbours (upstream forces 6 for Stereo-CITE-seq and "
        "Spatial-epigenome-transcriptome; the feature graph is fixed at k=20).",
    )
    parser.add_argument(
        "--cluster-method",
        default="leiden",
        choices=["leiden", "louvain", "mclust"],
        help="Clustering method.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution for leiden/louvain clustering.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device: 'auto', 'cpu', 'gpu'/'cuda', or 'cuda:N' for a specific GPU.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2022,
        help="Seed for numpy/torch/random (PCA, model initialisation, training) via SpatialGlue fix_seed.",
    )
    parser.add_argument(
        "--use-raw-counts",
        dest="use_raw_counts",
        action="store_true",
        default=False,
        help="Read each input's counts from adata.raw when it has one, instead of X.",
    )

    args = parser.parse_args()

    try:
        result = run_spatialglue_integration(
            rna_h5ad=args.rna_h5ad,
            protein_h5ad=args.protein_h5ad,
            output_dir=args.output_dir,
            datatype=args.datatype,
            epochs=args.epochs,
            dim_output=args.dim_output,
            n_clusters=args.n_clusters,
            n_neighbors=args.n_neighbors,
            cluster_method=args.cluster_method,
            resolution=args.resolution,
            device=args.device,
            seed=args.seed,
            use_raw_counts=args.use_raw_counts,
        )
        print(json.dumps(result, default=str))
        sys.stdout.flush()

    except Exception as e:
        log("ERROR: Exception during SpatialGlue run.")
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("spatialglue", str(e), task="multi_omics_integration")
        sys.exit(1)


if __name__ == "__main__":
    main()
