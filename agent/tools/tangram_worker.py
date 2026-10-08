#!/usr/bin/env python
"""Tangram worker: map a labelled single-cell reference onto a spatial slide.

What this file does that Tangram itself leaves to the caller:

* **Gene projection follows the mapping mode.** ``mode='clusters'`` maps cluster aggregates, so the
  map's rows are the clusters ('0'..'k-1'), not the reference cells. ``tg.project_genes`` has to be
  told the same ``cluster_label`` to aggregate the reference the same way; called without it, it
  compares the cluster map against the per-cell reference and raises "The two AnnDatas need to have
  same `obs` index." -- which is what the default mode did on every run, after the probability CSV
  had been written and before either h5ad was.
* **The projection is optional and costed.** ``tg.project_genes`` builds a dense spots x reference
  genes matrix (float64 in clusters mode; in cells mode it also densifies the reference). That
  matrix is intrinsic to the projected-genes output and is not needed for the cell-type mapping, so
  ``project_genes`` (default True, as before) can switch it off, and before training starts its size
  is estimated against the memory this process can use: a projection that cannot fit stops the run
  with the numbers and the knob instead of failing after the mapping has been trained. In
  ``mode='cells'`` the mapping itself is a dense cells x spots matrix that training holds several
  float32 copies of; it is costed the same way, whatever ``project_genes`` says -- against host
  memory on the CPU, and on CUDA against the GPU's free memory (with the host's smaller share against
  host memory) -- and a mapping that cannot fit names ``mode='clusters'`` -- skipping the projection
  would not save it.
* **The probability CSV is a composition.** ``tg.project_cell_annotations`` returns ``M^T @
  one_hot``: in clusters mode each column is one cluster's distribution over the spots (columns sum
  to 1, rows to ~1/n_spots), and in cells mode each row counts the reference cells mapped to the
  spot. Clusters mode trains the map against cluster *sums* and a spot density of
  ``cluster_density @ M``, so the cells it places on a spot are proportional to
  ``cluster_density[c] * M[c, spot]``; row-normalising the unweighted map instead gives every
  cluster the same prior and inflates rare types several-fold. ``tangram_celltype_probabilities.csv``
  (and ``obsm['tangram_ct_proportions']``) therefore hold each spot's cell-type composition, rows
  summing to 1: weighted by ``cluster_density`` in clusters mode, row-normalised counts in cells
  mode. ``obsm['tangram_ct_pred']`` keeps Tangram's own unweighted scores.
* **Background spots are not tissue.** Spots with ``obs['in_tissue'] == 0`` are left out of the
  mapping and the outputs, and the payload says how many (``params.in_tissue_filter``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import tangram as tg
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    build_deconv_analysis,
    cell_type_rename_note,
    cell_type_rename_params,
    id_mismatch_msg,
    keep_in_tissue,
    record_in_tissue,
    record_method,
    resolve_compute,
    sanitize_cell_type_names,
)
from worker_utils import drop_unlabeled as _split_unlabeled  # the parameter of the same name shadows it


def _get_device_str(device_str: str) -> str:
    """
    Resolve any of the fleet's device spellings ('auto', 'cpu', 'CPU', 'gpu', 'GPU', 'cuda:0', ...)
    to a string Tangram can hand to torch.device(). Shared with every other worker via
    worker_utils.resolve_compute, so 'GPU' no longer reaches torch as a literal and a CUDA request
    on a CPU-only box degrades instead of raising.
    """
    return resolve_compute(device_str).device


def _available_bytes():
    """Memory this process can still allocate, in bytes, or None if unknown.

    The fleet's one reader, ``worker_utils.available_memory_bytes``: the smaller of MemAvailable and
    the room under a cgroup memory limit, with the cgroup's page cache counted as reclaimable. This
    worker used to take ``memory.max - memory.current``, and ``memory.current`` counts cached file
    pages -- after reading two h5ad files a memory-limited container sits near its limit on cache
    alone, and the checks below refused mappings and projections that would have fit.
    """
    return available_memory_bytes()


#: Float32 copies of the cells x spots mapping ``tg.map_cells_to_space`` holds at its peak while it
#: trains in ``mode='cells'``: the parameter M, softmax(M), the entropy term's log(softmax) (computed
#: even at lambda_r=0), M's gradient and the backward temporaries, and Adam's two moment buffers.
#: Measured with tangram 1.0.4 / torch 2.8 on CPU (peak-RSS growth per extra float32 copy of the map,
#: 3,000-12,000 cells x 5,000 spots): 6.8-7.0.
_MAP_TRAINING_FLOAT32_COPIES = 7


def _mapping_training_bytes(n_cells, n_spots):
    """Bytes ``tg.map_cells_to_space(mode='cells')`` holds at its peak for the cells x spots map."""
    return _MAP_TRAINING_FLOAT32_COPIES * 4 * int(n_cells) * int(n_spots)


#: Float32-equivalents of the cells x spots map the HOST still holds when training runs on a CUDA
#: device (tangram 1.0.4 mapping_optimizer.Mapper). The parameter, its softmax, the gradient and Adam's
#: buffers -- the ``_MAP_TRAINING_FLOAT32_COPIES`` -- live on the GPU, but the initial draw is made on
#: the host: ``np.random.normal`` returns a float64 array (two float32-equivalents) that
#: ``torch.tensor(..., dtype=torch.float32)`` converts on the host before copying it over (one more).
#: ``train`` hands the map back as one float32 host array (``softmax(M).cpu().numpy()``), by which time
#: the float64 draw is gone.
_MAP_HOST_FLOAT32_COPIES_ON_CUDA = 3


def _gpu_free_bytes(device):
    """Free memory on a CUDA device, in bytes, or None when it is not a CUDA device or torch cannot say."""
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        free, _total = torch.cuda.mem_get_info(torch.device(device))
        return int(free)
    except Exception:
        return None


def _mapping_preflight(mode, n_spots, n_cells, n_clusters, device="cpu"):
    """Cost a cells-mode map where training will hold it, refuse one that cannot fit, and say what was checked.

    Runs whatever ``project_genes`` says: the mapping is trained either way, and a cells x spots map
    that does not fit is OOM-killed inside torch with no JSON at all. The remedy is
    ``mode='clusters'`` (one row per label instead of one per cell) or more memory -- not
    ``project_genes=False``, which only drops the projection that would have come after it. In
    clusters mode the map is labels x spots, a few dozen rows, smaller than the slide already in
    memory, and is not costed separately.

    Where the map is held follows the device. On the CPU (or any device that is not CUDA) training
    holds all ``_MAP_TRAINING_FLOAT32_COPIES`` in host memory, checked against ``_available_bytes()``.
    On CUDA those copies are held in GPU memory, checked against the device's free memory; the host
    then holds only ``_MAP_HOST_FLOAT32_COPIES_ON_CUDA`` of them, checked against host memory -- so a
    map that fits the GPU is not refused for host memory it would never use.

    Returns ``{"estimated_bytes", "checked_against", "host_estimated_bytes"}``: the training peak
    (None in clusters mode), the memory it was compared with (``'host'``, ``'cuda'``, or None when
    that memory could not be read and nothing was compared), and on CUDA the host's share.
    """
    report = {"estimated_bytes": None, "checked_against": None, "host_estimated_bytes": None}
    if mode != "cells":
        return report
    n_cells, n_spots, n_clusters = int(n_cells), int(n_spots), int(n_clusters)
    need = _mapping_training_bytes(n_cells, n_spots)
    report["estimated_bytes"] = need
    budget = _available_bytes()
    alternative = f"mode='clusters' maps one row per annotation_key label instead ({n_clusters} x {n_spots})"
    not_the_projection = (
        "skipping the gene projection would not help, because the mapping itself does not fit. Nothing was trained."
    )
    if not str(device).startswith("cuda"):
        if budget is not None and need > budget:
            raise MemoryError(
                f"mode='cells' maps every one of the {n_cells} reference cells onto the {n_spots} spots: "
                f"tg.map_cells_to_space trains a dense {n_cells} x {n_spots} mapping and holds about "
                f"{_MAP_TRAINING_FLOAT32_COPIES} float32 copies of it while it trains (the parameter, its softmax, the "
                f"entropy term, the gradient and Adam's two moment buffers): ~{need / 1e9:.1f} GB, but "
                f"~{budget / 1e9:.1f} GB is available. {alternative}, or run on a machine with more memory; "
                + not_the_projection
            )
        report["checked_against"] = "host" if budget is not None else None
        return report

    host_need = _MAP_HOST_FLOAT32_COPIES_ON_CUDA * 4 * n_cells * n_spots
    report["host_estimated_bytes"] = host_need
    gpu_free = _gpu_free_bytes(device)
    if gpu_free is not None and need > gpu_free:
        if budget is not None and need <= budget:
            on_the_cpu = f", or device='cpu' trains it in host memory, where ~{budget / 1e9:.1f} GB is available"
        elif budget is not None:
            on_the_cpu = (
                f"; device='cpu' would need the same ~{need / 1e9:.1f} GB of host memory, and only "
                f"~{budget / 1e9:.1f} GB is available there"
            )
        else:
            on_the_cpu = f", or device='cpu' trains it in host memory (~{need / 1e9:.1f} GB)"
        raise MemoryError(
            f"mode='cells' maps every one of the {n_cells} reference cells onto the {n_spots} spots: "
            f"tg.map_cells_to_space trains a dense {n_cells} x {n_spots} mapping on {device} and holds about "
            f"{_MAP_TRAINING_FLOAT32_COPIES} float32 copies of it in GPU memory while it trains (the parameter, its "
            f"softmax, the entropy term, the gradient and Adam's two moment buffers): ~{need / 1e9:.1f} GB of GPU "
            f"memory, but {device} has ~{gpu_free / 1e9:.1f} GB free. {alternative}{on_the_cpu}; " + not_the_projection
        )
    if budget is not None and host_need > budget:
        raise MemoryError(
            f"mode='cells' maps every one of the {n_cells} reference cells onto the {n_spots} spots. Training runs on "
            f"{device}, but the host still holds about {_MAP_HOST_FLOAT32_COPIES_ON_CUDA} float32 copies of the dense "
            f"{n_cells} x {n_spots} mapping (Tangram draws its starting value as float64 on the host and converts it "
            "to float32 there before moving it to the GPU, and hands the trained map back as a host array): "
            f"~{host_need / 1e9:.1f} GB of host memory, but ~{budget / 1e9:.1f} GB is available. {alternative}, or "
            "run on a machine with more host memory; " + not_the_projection
        )
    report["checked_against"] = "cuda" if gpu_free is not None else None
    return report


def _check_mapping_fits(mode, n_spots, n_cells, n_clusters, device="cpu"):
    """Refuse, with the numbers and the knob, a cells-mode mapping this machine cannot train.

    ``_mapping_preflight`` does the costing (host memory on the CPU, GPU memory plus the host's share
    on CUDA). Returns the training-peak estimate in bytes (None in clusters mode).
    """
    return _mapping_preflight(mode, n_spots, n_cells, n_clusters, device=device)["estimated_bytes"]


def _projection_bytes(mode, n_spots, n_cells, n_genes, n_clusters, sc_dtype, sc_is_sparse):
    """Bytes ``tg.project_genes`` allocates on top of what the run already holds.

    Upstream (tangram 1.0.4 utils.project_genes): in clusters mode the reference is aggregated into a
    float64 clusters x genes matrix (mapping_utils.adata_to_cluster_expression) and the product
    ``adata_map.X.T @ adata_sc.X`` is a dense float64 spots x genes matrix. In cells mode a sparse
    reference is densified in place (cells x genes) and the product takes the promoted dtype of the
    float32 map and the reference.
    """
    n_spots, n_cells, n_genes, n_clusters = int(n_spots), int(n_cells), int(n_genes), int(n_clusters)
    if mode == "clusters":
        return 8 * n_clusters * n_genes + 8 * n_spots * n_genes
    itemsize = int(np.dtype(sc_dtype).itemsize)
    out_itemsize = int(np.result_type(np.float32, np.dtype(sc_dtype)).itemsize)
    dense_reference = n_cells * n_genes * itemsize if sc_is_sparse else 0
    return dense_reference + n_spots * n_genes * out_itemsize


def _check_projection_fits(mode, n_spots, n_cells, n_genes, n_clusters, sc_dtype, sc_is_sparse):
    """Refuse, with the numbers and the knob, a gene projection this machine cannot hold.

    Run before training, so an infeasible projection costs seconds rather than a trained mapping. The
    projection runs while the float32 mapping it projects from (cells or clusters x spots) is still
    held, so that is counted against the budget too. Returns the projection's own estimate (bytes)
    so the payload can report it.
    """
    need = _projection_bytes(mode, n_spots, n_cells, n_genes, n_clusters, sc_dtype, sc_is_sparse)
    n_rows = int(n_clusters) if mode == "clusters" else int(n_cells)
    held = 4 * n_rows * int(n_spots)
    budget = _available_bytes()
    if budget is not None and need + held > budget:
        raise MemoryError(
            f"tg.project_genes would build a dense {int(n_spots)} spots x {int(n_genes)} reference genes expression "
            f"matrix (mode='{mode}'"
            + (", after densifying the reference" if mode == "cells" and sc_is_sparse else "")
            + f"): ~{need / 1e9:.1f} GB, beside the ~{held / 1e9:.1f} GB mapping it projects from, but "
            f"~{budget / 1e9:.1f} GB is available. Pass project_genes=False to skip the gene projection -- the "
            "cell-type mapping and tangram_celltype_probabilities.csv do not need it -- or run on a machine with "
            "more memory. Nothing was trained."
        )
    return need


def _composition(scores, map_obs, annotation_key, mode):
    """``(composition, weighting, n_spots_without_mass)`` from ``tg.project_cell_annotations`` scores.

    ``scores`` is ``obsm['tangram_ct_pred']`` as Tangram leaves it, spots x labels, columns named by
    the labels of ``map_obs[annotation_key]`` (upstream ``one_hot_encoding``). Clusters mode:
    ``scores[s, c] = M[c, s]``, cluster c's share of its cells placed on spot s, so the cells of type c
    at s are ``cluster_density[c] * M[c, s]`` (``map_cells_to_space`` trains against cluster sums and
    a spot density of ``d_source @ M`` with ``d_source = cluster_density``); those are weighted and
    then row-normalised. Cells mode: ``scores[s, t]`` already counts the type-t cells mapped to s, and
    is row-normalised. A spot onto which nothing was mapped keeps an all-zero row and is counted.
    """
    raw = scores.astype(float)
    if mode == "clusters":
        if "cluster_density" not in map_obs.columns:
            raise ValueError(
                "Tangram's cluster map carries no obs['cluster_density'] (each cluster's share of the reference), "
                "so its per-cluster scores cannot be turned into a per-spot composition."
            )
        density = {}
        for label, share in zip(map_obs[annotation_key].tolist(), map_obs["cluster_density"].astype(float).tolist()):
            density[label] = density.get(label, 0.0) + share
        unknown = [c for c in raw.columns if c not in density]
        if unknown:
            raise ValueError(
                f"Tangram scored labels {unknown[:5]} that its cluster map does not carry in obs['{annotation_key}']."
            )
        raw = raw.mul(np.array([density[c] for c in raw.columns], dtype=float), axis=1)
        weighting = "cluster_density"
    else:
        weighting = "none"
    totals = raw.sum(axis=1)
    n_non_finite = int((~np.isfinite(raw.to_numpy())).any(axis=1).sum())
    if n_non_finite:
        raise ValueError(
            f"Tangram's projected cell-type scores are NaN or infinite on {n_non_finite} of {len(raw)} spots, so the "
            "training produced no usable mapping; no composition was written."
        )
    has_mass = np.asarray(totals > 0)
    composition = raw.copy()
    composition.loc[has_mass] = raw.loc[has_mass].div(totals[has_mass], axis=0)
    return composition, weighting, int((~has_mass).sum())


def _replace_atomically(write, final_path: Path) -> None:
    """Write through ``write(tmp_path)`` to ``<final>.partial``, then rename over ``final_path``.

    A run that dies half-way leaves the previous file (or none) rather than a truncated one.
    """
    final_path = Path(final_path)
    tmp = final_path.with_name(final_path.name + ".partial")
    try:
        write(tmp)
        os.replace(str(tmp), str(final_path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def run_tangram_mapping(
    sc_h5ad: str,
    spatial_h5ad: str,
    output_dir: str,
    annotation_key: str,
    mode: str,
    n_markers_per_class: int,
    num_epochs: int,
    density_prior: str,
    device: str,
    project_genes: bool = True,
    drop_unlabeled: bool = False,
) -> dict:
    """
    Core Tangram pipeline:
      1. Load scRNA and spatial AnnData; leave out background spots (obs['in_tissue'] == 0); refuse
         (or, with drop_unlabeled, drop) unlabelled cells.
      2. Rank marker genes per annotation group and select top N.
      3. tg.pp_adatas for preprocessing & alignment of gene space.
      4. Cost, before training: the cells-mode mapping always, the dense projection if project_genes.
      5. tg.map_cells_to_space to learn mapping (cells or clusters).
      6. tg.project_cell_annotations to get cell-type scores per spot, turned into a per-spot composition
         (weighted by cluster_density in clusters mode, then row-normalised).
      7. If project_genes, tg.project_genes with the mapping's own cluster_label (clusters mode).
      8. Save all results under output_dir.
    """
    print("[tangram-worker] Task = map_sc_to_spatial", file=sys.stderr)
    print(f"[tangram-worker] sc_h5ad = {sc_h5ad}", file=sys.stderr)
    print(f"[tangram-worker] spatial_h5ad = {spatial_h5ad}", file=sys.stderr)
    print(f"[tangram-worker] output_dir = {output_dir}", file=sys.stderr)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    device_str = _get_device_str(device)
    print(f"[tangram-worker] device = {device_str}", file=sys.stderr)
    print(f"[tangram-worker] mode = {mode}", file=sys.stderr)
    print(
        f"[tangram-worker] annotation_key = {annotation_key}, n_markers_per_class = {n_markers_per_class}",
        file=sys.stderr,
    )

    # output paths
    map_h5ad = output_dir_path / "tangram_mapping_sc_to_spatial.h5ad"
    spatial_annotated_h5ad = output_dir_path / "tangram_spatial_with_annotations.h5ad"
    genes_h5ad = output_dir_path / "tangram_projected_genes.h5ad"
    ct_prob_csv = output_dir_path / "tangram_celltype_probabilities.csv"

    with contextlib.redirect_stdout(sys.stderr):
        # ---------- Load data ----------
        ad_sc = sc.read_h5ad(sc_h5ad)
        ad_sp = sc.read_h5ad(spatial_h5ad)

        ad_sc.var_names_make_unique()
        ad_sp.var_names_make_unique()

        # Background glass (obs['in_tissue'] == 0; CELLxGENE Visium exports carry every array spot) is
        # not tissue. Mapped onto, it takes a share of every cluster's cells and gets a composition of
        # its own in the CSV. The shared rule leaves it out and says how many.
        ad_sp, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(ad_sp, "spots")
        if n_spots_off_tissue:
            print(
                f"[tangram-worker] Left out {n_spots_off_tissue} of {n_spots_supplied} spots with "
                "obs['in_tissue'] == 0 (background).",
                file=sys.stderr,
            )

        print(
            f"[tangram-worker] Loaded scRNA: n_cells={ad_sc.n_obs}, n_genes={ad_sc.n_vars}",
            file=sys.stderr,
        )
        print(
            f"[tangram-worker] Loaded spatial: n_spots={ad_sp.n_obs}, n_genes={ad_sp.n_vars}",
            file=sys.stderr,
        )

        # ---------- Check annotation key ----------
        if annotation_key not in ad_sc.obs:
            raise ValueError(
                f"annotation_key='{annotation_key}' not found in scRNA obs. Available keys: {list(ad_sc.obs.columns)}"
            )

        # A missing label is not a class. Left in, cells mode scored a cell type named NaN (an
        # all-zero column in the probability CSV) and then crashed writing the h5ad ("nan of type
        # float is an invalid key"), and clusters mode dropped those cells without a word
        # (value_counts skips NaN).
        n_cells_input = int(ad_sc.n_obs)
        keep_mask, n_dropped_unlabeled = _split_unlabeled(
            ad_sc.obs[annotation_key].values,
            drop_unlabeled,
            f"reference cells (annotation_key='{annotation_key}')",
        )
        if n_dropped_unlabeled:
            ad_sc = ad_sc[keep_mask].copy()
            print(
                f"[tangram-worker] Dropped {n_dropped_unlabeled} reference cells with no label in "
                f"'{annotation_key}' (drop_unlabeled=True).",
                file=sys.stderr,
            )
        # A category no cell carries (a subset reference keeps its parent's categories) is not a cell
        # type this reference can speak for. Left in, sc.tl.rank_genes_groups stopped the run with
        # "Could not calculate statistics for groups D since they only contain one sample" -- for a
        # group with none -- and Tangram's cluster aggregation (value_counts keeps empty categories)
        # would have mapped it as an all-zero cluster.
        n_unused_categories = 0
        label_col = ad_sc.obs[annotation_key]
        if hasattr(label_col, "cat"):
            n_declared = len(label_col.cat.categories)
            ad_sc.obs[annotation_key] = label_col.cat.remove_unused_categories()
            n_unused_categories = int(n_declared - len(ad_sc.obs[annotation_key].cat.categories))
        n_label_classes = int(ad_sc.obs[annotation_key].nunique())

        # ---------- Rank marker genes (per annotation group) ----------
        print(
            f"[tangram-worker] Ranking marker genes by '{annotation_key}' ...",
            file=sys.stderr,
        )
        sc.tl.rank_genes_groups(ad_sc, groupby=annotation_key, use_raw=False)
        markers_df = pd.DataFrame(ad_sc.uns["rank_genes_groups"]["names"]).iloc[0:n_markers_per_class, :]
        markers = list(np.unique(markers_df.melt().value.values))
        print(
            f"[tangram-worker] Selected {len(markers)} unique marker genes "
            f"({n_markers_per_class} per {annotation_key} group).",
            file=sys.stderr,
        )

        # ---------- Preprocess with Tangram ----------
        print("[tangram-worker] Running tg.pp_adatas ...", file=sys.stderr)
        sc_genes_supplied = list(ad_sc.var_names)
        sp_genes_supplied = list(ad_sp.var_names)
        tg.pp_adatas(ad_sc, ad_sp, genes=markers)

        n_training = len(ad_sc.uns.get("training_genes", []))
        n_overlap = len(ad_sc.uns.get("overlap_genes", []))
        print(
            f"[tangram-worker] Tangram training_genes={n_training}, overlap_genes={n_overlap}",
            file=sys.stderr,
        )
        # With no training gene the cosine term has nothing to compare and its gradient is zero, so
        # the map is trained by the density prior alone and a composition comes out at status ok
        # that no gene supports.
        if n_overlap == 0:
            raise ValueError(
                id_mismatch_msg("genes", "scRNA reference", sc_genes_supplied, "spatial slide", sp_genes_supplied)
                + " Tangram compares them case-insensitively; with none shared there is nothing to map on."
            )
        if n_training == 0:
            raise ValueError(
                f"None of the {len(markers)} marker genes ranked for '{annotation_key}' "
                f"({n_markers_per_class} per label) is among the {n_overlap} genes the reference and the slide "
                "share, so Tangram would train on no gene at all. Raise n_markers_per_class, or use a slide "
                "whose panel covers the reference's markers."
            )

        # ---------- Cost the mapping and the gene projection before anything is trained ----------
        mapping_fit = _mapping_preflight(
            mode, n_spots=ad_sp.n_obs, n_cells=ad_sc.n_obs, n_clusters=n_label_classes, device=device_str
        )
        mapping_bytes = mapping_fit["estimated_bytes"]
        if mapping_bytes is not None:
            checked = mapping_fit["checked_against"]
            print(
                f"[tangram-worker] Cells-mode mapping estimated at ~{mapping_bytes / 1e9:.2f} GB while training"
                + (
                    f", checked against {checked} memory."
                    if checked
                    else f"; the free memory of {device_str} could not be read, so it was not checked."
                ),
                file=sys.stderr,
            )
        projection_bytes = None
        if project_genes:
            projection_bytes = _check_projection_fits(
                mode,
                n_spots=ad_sp.n_obs,
                n_cells=ad_sc.n_obs,
                n_genes=ad_sc.n_vars,
                n_clusters=n_label_classes,
                sc_dtype=ad_sc.X.dtype,
                sc_is_sparse=hasattr(ad_sc.X, "toarray"),
            )
            print(
                f"[tangram-worker] Gene projection estimated at ~{projection_bytes / 1e9:.2f} GB.",
                file=sys.stderr,
            )

        # ---------- Map scRNA to spatial ----------
        print("[tangram-worker] Running tg.map_cells_to_space ...", file=sys.stderr)
        map_kwargs = {
            "mode": mode,
            "density_prior": density_prior,
            "num_epochs": int(num_epochs),
            "device": device_str,
        }
        # Tangram's mode='clusters' aggregates SC cells by `cluster_label` before
        # mapping (paper-grade default for proportion estimation). The library
        # raises ValueError("A cluster_label must be specified ...") if omitted.
        # Reuse annotation_key as the cluster column since the SC reference is
        # already labeled by it.
        if mode == "clusters":
            map_kwargs["cluster_label"] = annotation_key
        ad_map = tg.map_cells_to_space(
            ad_sc,
            ad_sp,
            **map_kwargs,
        )

        print(
            f"[tangram-worker] Mapping done. ad_map: n_obs(cells)={ad_map.n_obs}, n_vars(spots)={ad_map.n_vars}",
            file=sys.stderr,
        )

        # ---------- Project cell annotations ----------
        print(
            f"[tangram-worker] Projecting annotation '{annotation_key}' to spatial ...",
            file=sys.stderr,
        )
        tg.project_cell_annotations(ad_map, ad_sp, annotation=annotation_key)

        # ct_prob_df: the composition written to the CSV. ct_renames: the payload builder reads it
        # whether or not the branch below runs.
        ct_prob_df = None
        ct_renames: dict = {}
        proportion_weighting = None
        n_spots_without_mass = 0
        if "tangram_ct_pred" in ad_sp.obsm:
            ct_pred = ad_sp.obsm["tangram_ct_pred"]

            # 建立 DataFrame（无论 Tangram 返回的是 ndarray 还是 DataFrame）
            if isinstance(ct_pred, pd.DataFrame):
                ct_scores_df = ct_pred.copy()
            else:
                # Upstream names the columns by the MAP's labels in order of appearance
                # (one_hot_encoding(adata_map.obs[annotation])); in clusters mode that is not the
                # reference's order.
                ct_scores_df = pd.DataFrame(
                    ct_pred,
                    index=ad_sp.obs_names,
                    columns=list(pd.unique(ad_map.obs[annotation_key])),
                )
            ct_prob_df, proportion_weighting, n_spots_without_mass = _composition(
                ct_scores_df, ad_map.obs, annotation_key, mode
            )

            # ---- 关键修复：把包含 "/" 的列名改成 HDF5 安全的名字 ----
            orig_cols = list(ct_prob_df.columns)
            # replace_space=False: tangram has only ever folded "/", and the probability CSVs it
            # has already published carry the spaces. Folding them now would rename them.
            safe_cols, ct_renames = sanitize_cell_type_names(orig_cols, replace_space=False)

            if ct_renames:
                print(
                    "[tangram-worker] Detected '/' in some cell-type names; "
                    "replacing with '_' to make HDF5-safe column names.",
                    file=sys.stderr,
                )
                ct_prob_df.columns = safe_cols
                ct_scores_df.columns = safe_cols
                # 记录一个 mapping，方便将来需要恢复原名字
                ad_sp.uns["tangram_ct_pred_colmap"] = {
                    "original": orig_cols,
                    "sanitized": safe_cols,
                }

            # 把安全列名的表回写进 obsm: Tangram's own scores under its own key, the composition
            # the CSV holds beside them.
            ad_sp.obsm["tangram_ct_pred"] = ct_scores_df
            ad_sp.obsm["tangram_ct_proportions"] = ct_prob_df
            _replace_atomically(ct_prob_df.to_csv, ct_prob_csv)
            print(
                f"[tangram-worker] Saved per-spot cell-type composition (weighting: {proportion_weighting}) "
                f"to {ct_prob_csv}",
                file=sys.stderr,
            )
        else:
            print(
                "[tangram-worker] WARNING: 'tangram_ct_pred' not found in ad_sp.obsm.",
                file=sys.stderr,
            )

        # ---------- Project genes (new spatial data) ----------
        ad_ge = None
        if project_genes:
            print("[tangram-worker] Projecting genes with tg.project_genes ...", file=sys.stderr)
            project_kwargs = {"adata_map": ad_map, "adata_sc": ad_sc}
            if mode == "clusters":
                # The map's rows are the clusters, so the reference has to be aggregated the same
                # way: same cluster_label, and scale=True, map_cells_to_space's own default (the
                # mapping above does not override it).
                project_kwargs["cluster_label"] = annotation_key
                project_kwargs["scale"] = True
            ad_ge = tg.project_genes(**project_kwargs)
            print(
                f"[tangram-worker] New spatial data ad_ge: n_spots={ad_ge.n_obs}, n_genes={ad_ge.n_vars}",
                file=sys.stderr,
            )
        else:
            print("[tangram-worker] project_genes=False: gene projection skipped.", file=sys.stderr)

        # ---------- Save AnnData outputs ----------
        _replace_atomically(ad_map.write_h5ad, map_h5ad)
        print(f"[tangram-worker] Saved mapping AnnData to {map_h5ad}", file=sys.stderr)

        _replace_atomically(ad_sp.write_h5ad, spatial_annotated_h5ad)
        print(
            f"[tangram-worker] Saved spatial AnnData with annotations to {spatial_annotated_h5ad}",
            file=sys.stderr,
        )

        if ad_ge is not None:
            _replace_atomically(ad_ge.write_h5ad, genes_h5ad)
            print(f"[tangram-worker] Saved projected genes AnnData to {genes_h5ad}", file=sys.stderr)

    # --- Compute summary stats for deconv analysis ---
    n_celltypes = 0
    celltypes = []
    dominant_counts: dict = {}
    if ct_prob_df is not None:
        celltypes = list(ct_prob_df.columns)
        n_celltypes = len(celltypes)
        # From the composition, and only over spots something was mapped onto: an all-zero row has
        # no dominant type (idxmax would name the first column).
        with_mass = ct_prob_df[np.asarray(ct_prob_df.sum(axis=1) > 0)]
        dominant = with_mass.idxmax(axis=1)
        dominant_counts = {str(k): int(v) for k, v in dominant.value_counts().items()}

    out = WorkerOutput("tangram", task="sc_to_spatial_mapping")
    out.set_data(
        n_spots=int(ad_sp.n_obs),
        n_genes=int(ad_sp.n_vars),
        n_cells_sc=int(ad_sc.n_obs),
        n_cells_sc_input=n_cells_input,
        n_genes_sc=int(ad_sc.n_vars),
        n_overlap_genes=int(len(ad_sc.uns.get("overlap_genes", []))),
    )
    out.add_output_files(
        {
            "mapping_h5ad": str(map_h5ad),
            "spatial_with_annotations_h5ad": str(spatial_annotated_h5ad),
            # Listed only when this run wrote it: with project_genes=False a file of that name in
            # output_dir is some earlier run's, and pointing at it would pass it off as this one's.
            "projected_genes_h5ad": str(genes_h5ad) if ad_ge is not None else None,
            "celltype_probabilities_csv": str(ct_prob_csv) if ct_prob_df is not None else None,
        }
    )
    out.add_params(
        {
            "annotation_key": annotation_key,
            "mode": mode,
            "density_prior": density_prior,
            "device": device_str,
            "num_epochs": int(num_epochs),
            "n_markers_per_class": int(n_markers_per_class),
            "n_training_genes": int(len(ad_sc.uns.get("training_genes", markers))),
            "project_genes": bool(project_genes),
            "projection_estimated_bytes": int(projection_bytes) if projection_bytes is not None else None,
            "mapping_estimated_bytes": int(mapping_bytes) if mapping_bytes is not None else None,
            "mapping_memory_checked_against": mapping_fit["checked_against"],
            "mapping_host_estimated_bytes": mapping_fit["host_estimated_bytes"],
            "proportion_weighting": proportion_weighting,
            "drop_unlabeled": bool(drop_unlabeled),
            "n_reference_cells_dropped_unlabeled": int(n_dropped_unlabeled),
            "n_unused_label_categories_dropped": int(n_unused_categories),
        }
    )
    out.add_params(cell_type_rename_params(ct_renames))
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    if n_spots_without_mass:
        out.add_params({"n_spots_without_mapped_cells": int(n_spots_without_mass)})
        out.add_warning(
            f"{n_spots_without_mass} of {int(ad_sp.n_obs)} spots received no mapped cells; their rows in "
            "tangram_celltype_probabilities.csv are all zero and they count toward no dominant type."
        )
    method = f"Tangram map_cells_to_space (mode='{mode}')"
    if ad_ge is not None:
        method += " + project_genes" + (f" (cluster_label='{annotation_key}')" if mode == "clusters" else "")
    record_method(out, method)
    if n_dropped_unlabeled:
        out.add_warning(
            f"{n_dropped_unlabeled} of {n_cells_input} reference cells had no label in '{annotation_key}' and were "
            "left out (drop_unlabeled=True)."
        )
    if n_unused_categories:
        out.add_warning(
            f"{n_unused_categories} categories of '{annotation_key}' had no cells and were not mapped as cell types."
        )
    out.set_summary(
        n_celltypes=n_celltypes,
        celltypes=celltypes,
        dominant_counts=dominant_counts,
    )
    projection_note = (
        " Gene projection skipped (project_genes=False); tangram_projected_genes.h5ad was not written."
        if ad_ge is None
        else f" Projected {int(ad_ge.n_vars)} reference genes onto {int(ad_ge.n_obs)} spots."
    )
    dropped_note = (
        f" {n_dropped_unlabeled} unlabelled reference cells were left out (drop_unlabeled=True)."
        if n_dropped_unlabeled
        else ""
    )
    if proportion_weighting == "cluster_density":
        composition_note = (
            " tangram_celltype_probabilities.csv holds each spot's cell-type composition (rows sum to 1): the "
            "cluster map weighted by each cluster's share of the reference (cluster_density), as Tangram's density "
            "term trains it, then row-normalised; obsm['tangram_ct_pred'] keeps Tangram's unweighted scores."
        )
    elif proportion_weighting == "none":
        composition_note = (
            " tangram_celltype_probabilities.csv holds each spot's cell-type composition (rows sum to 1): the "
            "reference cells mapped to the spot, counted per type and row-normalised; obsm['tangram_ct_pred'] keeps "
            "Tangram's counts."
        )
    else:
        composition_note = ""
    tissue_note = (
        f" {n_spots_off_tissue} of {n_spots_supplied} spots were background (obs['in_tissue'] == 0) and were left "
        "out of the mapping and the outputs."
        if n_spots_off_tissue
        else ""
    )
    out.set_analysis(
        build_deconv_analysis(
            n_celltypes,
            dominant_counts,
            total_spots=int(ad_sp.n_obs),
            method_name="Tangram",
        )
        + composition_note
        + cell_type_rename_note(ct_renames)
        + projection_note
        + dropped_note
        + tissue_note
    )

    return out.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Tangram worker for SpatialOmicsLab MCP (CLI mode)")

    parser.add_argument(
        "--task",
        required=False,
        default="map",
        choices=["map"],
        help="Currently only 'map' is supported (map scRNA to spatial).",
    )
    parser.add_argument(
        "--sc-h5ad",
        required=True,
        help="Path to single-cell AnnData (.h5ad).",
    )
    parser.add_argument(
        "--spatial-h5ad",
        required=True,
        help="Path to spatial AnnData (.h5ad), e.g. Visium.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store Tangram outputs (e.g. /workspace/work/tangram_xxx).",
    )
    parser.add_argument(
        "--annotation-key",
        default="cell_type",
        help="obs column in scRNA AnnData containing cell-type labels.",
    )
    parser.add_argument(
        "--mode",
        default="clusters",
        choices=["cells", "clusters"],
        help="Tangram mapping mode: 'clusters' (paper-grade) or 'cells'.",
    )
    parser.add_argument(
        "--n-markers-per-class",
        type=int,
        default=100,
        help="Top N marker genes per annotation group used as training genes.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1000,
        help="Number of training epochs for tg.map_cells_to_space (paper: 1000).",
    )
    parser.add_argument(
        "--density-prior",
        default="rna_count_based",
        choices=["rna_count_based", "uniform"],
        help="Density prior for Tangram ('rna_count_based' recommended for Visium).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device string: 'auto', 'cpu', or 'cuda:0'.",
    )
    parser.add_argument(
        "--no-project-genes",
        dest="project_genes",
        action="store_false",
        default=True,
        help="Skip tg.project_genes (the dense spots x reference-genes projection); the mapping is unaffected.",
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        default=False,
        help="Leave out reference cells whose annotation_key label is missing instead of refusing the run.",
    )

    args = parser.parse_args()

    try:
        result = run_tangram_mapping(
            sc_h5ad=args.sc_h5ad,
            spatial_h5ad=args.spatial_h5ad,
            output_dir=args.output_dir,
            annotation_key=args.annotation_key,
            mode=args.mode,
            n_markers_per_class=args.n_markers_per_class,
            num_epochs=args.num_epochs,
            density_prior=args.density_prior,
            device=args.device,
            project_genes=args.project_genes,
            drop_unlabeled=args.drop_unlabeled,
        )

        # stdout 只输出 JSON
        print(json.dumps(result, default=str))
        sys.stdout.flush()

    except Exception as e:
        print("[tangram-worker] ERROR:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("tangram", str(e), task="sc_to_spatial_mapping")
        sys.exit(1)


if __name__ == "__main__":
    main()
