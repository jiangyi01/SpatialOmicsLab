#!/usr/bin/env python

"""
stlearn_worker.py

Run stLearn stSME-based spatial clustering on a Visium-like h5ad.
This script is called by the MCP wrapper via subprocess.

- Logs & progress  -> stderr  (prefixed with [stlearn-worker])
- Final result     -> stdout (single JSON line)

What runs, and when it refuses
------------------------------
stSME is stLearn's method: tile the histology image around every spot, extract CNN features from
the tiles, and adjust each spot's expression PCA by its morphologically similar neighbours within
``radius`` image pixels (``st.spatial.morphology.adjust``); Louvain then clusters the adjusted
embedding. It needs a histology image, so an input without one -- or without the image named by
``use_quality`` -- is refused with the reason. So is any failure inside the stSME steps.
``allow_pca_fallback=True`` instead clusters the expression PCA alone, which uses no image and no
spatial information; the payload then says so in ``params.method`` / ``params.used_fallback`` and a
warning, and lists ``radius``/``crop_size`` under ``params.ignored``.

The library is the first ``uns['spatial']`` entry that is a Visium library (a mapping with
``images`` or ``scalefactors``). stLearn itself takes the first key, whatever it is, and CELLxGENE
files put a scalar ``is_single`` there. The other entries are set aside while stLearn runs and put
back before the annotated h5ad is written.

Scaling keeps a sparse ``X`` sparse: genes are scaled to unit variance and the centring is left to
the PCA, which centres a sparse matrix implicitly. That is the same principal subspace a
zero-centred dense scale produces, without allocating ``n_spots x n_genes`` floats.

Spots and matrix
----------------
Spots with ``obs['in_tissue'] == 0`` are background outside the tissue (CELLxGENE Visium exports
carry every array spot); they are left out right after loading (``worker_utils.keep_in_tissue``)
and reported in ``params.in_tissue_filter``, a warning and the analysis. ``data.n_spots`` is the
number clustered, ``data.n_spots_input`` the number supplied.

stLearn normalises ``X`` as counts (normalize_total, log1p). ``worker_utils.choose_counts_matrix``
decides what it runs on: a negative or non-finite ``X`` (scaled data) is refused, naming
``use_raw_counts`` when ``adata.raw`` holds counts; a non-negative non-integer ``X`` (already
normalised) runs with a warning; ``use_raw_counts=True`` runs on ``adata.raw.X``.
``params.expression_source`` says which matrix ran.

stLearn prints its progress messages (normalisation, log1p, scaling, PCA, feature extraction) to
stdout; the worker sends them to stderr so its stdout carries only the JSON result.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from collections.abc import Mapping

from worker_utils import (
    WorkerOutput,
    build_cluster_analysis,
    choose_counts_matrix,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    spatial_coords,
)

TOOL = "stlearn"

STSME_METHOD = "stSME (stLearn morphology-adjusted PCA, X_pca_morphology) + Louvain"
PCA_ONLY_METHOD = "PCA + Louvain on expression alone (stSME not run; no image or spatial information used)"

# Parameters that exist only for the stSME steps; they have no effect when stSME does not run.
STSME_ONLY_PARAMS = ("radius", "crop_size")


def log(msg: str) -> None:
    sys.stderr.write(f"[stlearn-worker] {msg}\n")
    sys.stderr.flush()


def _is_library(entry) -> bool:
    """A Visium library entry of ``uns['spatial']``: a mapping holding images or scale factors."""
    return isinstance(entry, Mapping) and ("images" in entry or "scalefactors" in entry)


def pick_spatial_library(adata):
    """``(library_id, other_keys)``: the first ``uns['spatial']`` entry that is a Visium library.

    ``other_keys`` names every other entry (scalars such as CELLxGENE's ``is_single``, and any
    further libraries). ``library_id`` is None when there is no library at all.
    """
    spatial = adata.uns.get("spatial", None)
    if not isinstance(spatial, Mapping):
        return None, []
    library_id = None
    others = []
    for key, entry in spatial.items():
        if library_id is None and _is_library(entry):
            library_id = key
        else:
            others.append(str(key))
    return library_id, others


def has_histology_images(adata) -> bool:
    """Whether the Visium library in ``adata.uns['spatial']`` holds at least one image."""
    library_id, _ = pick_spatial_library(adata)
    if library_id is None:
        return False
    images = adata.uns["spatial"][library_id].get("images", None)
    return isinstance(images, Mapping) and len(images) > 0


def stsme_blocker(adata, library_id, use_quality: str) -> str:
    """Why stSME cannot run on this object, or ``""`` when it can."""
    spatial = adata.uns.get("spatial", None)
    if library_id is None:
        if spatial is None:
            return "the input has no uns['spatial'], so there is no histology image to tile"
        keys = list(spatial.keys()) if isinstance(spatial, Mapping) else type(spatial).__name__
        return f"uns['spatial'] holds no Visium library (no entry with 'images' or 'scalefactors'); its keys are {keys}"
    lib = spatial[library_id]
    images = lib.get("images", None)
    if not isinstance(images, Mapping) or not len(images):
        return f"library '{library_id}' in uns['spatial'] stores no histology image"
    if use_quality not in images:
        return (
            f"library '{library_id}' has image(s) {sorted(str(k) for k in images)} but no "
            f"use_quality='{use_quality}' image"
        )
    if use_quality != "fulres":
        scalefactors = lib.get("scalefactors", None)
        key = f"tissue_{use_quality}_scalef"
        if not isinstance(scalefactors, Mapping) or key not in scalefactors:
            return (
                f"library '{library_id}' has no scalefactors['{key}'], so the spots cannot be placed "
                f"on its '{use_quality}' image"
            )
    if "spatial" not in adata.obsm:
        return "the input has no obsm['spatial'] spot coordinates to place on the image"
    return ""


def refusal_message(blocker: str) -> str:
    return (
        f"stLearn's stSME step cannot run on this input: {blocker}. stSME needs a histology image in "
        "uns['spatial'][<library>]['images'][use_quality] (and, for 'hires'/'lowres', the matching "
        "scalefactors['tissue_<quality>_scalef']). Pass the use_quality the image is stored under, or pass "
        "allow_pca_fallback=True to cluster the expression PCA alone -- no image and no spatial information "
        "then enter the clustering, and the payload records it as a fallback (params.method, "
        "params.used_fallback)."
    )


def isolate_library(adata, library_id):
    """Leave only the chosen library in ``uns['spatial']`` while stLearn runs.

    stLearn's ``convert_scanpy`` and ``tiling`` take the first key, and its plot refuses a mapping
    with more than one. Returns the original mapping (for :func:`restore_library`), or None when
    there was nothing to set aside.
    """
    spatial = adata.uns.get("spatial", None)
    if library_id is None or not isinstance(spatial, Mapping) or list(spatial.keys()) == [library_id]:
        return None
    original = dict(spatial)
    adata.uns["spatial"] = {library_id: spatial[library_id]}
    return original


def restore_library(adata, original, library_id) -> None:
    """Put the set-aside ``uns['spatial']`` entries back, in their original order."""
    if original is None:
        return
    current = adata.uns.get("spatial", {}).get(library_id, original[library_id])
    adata.uns["spatial"] = {key: (current if key == library_id else value) for key, value in original.items()}


def spots_without_neighbours(adata, radius):
    """Boolean mask of spots with no other spot within ``radius`` in the image frame stSME uses.

    Same neighbourhood as ``st.spatial.morphology.adjust`` (``cKDTree.query_ball_point`` on
    ``obs[['imagecol', 'imagerow']]``, distance <= radius).
    """
    import numpy as np
    from scipy.spatial import cKDTree

    coor = np.asarray(adata.obs[["imagecol", "imagerow"]].values, dtype=float)
    tree = cKDTree(coor)
    counts = np.asarray(tree.query_ball_point(coor, r=float(radius), return_length=True))
    return counts <= 1


def nearest_spacing(adata) -> float:
    """Median distance from a spot to its nearest other spot, in the image frame stSME uses."""
    import numpy as np
    from scipy.spatial import cKDTree

    coor = np.asarray(adata.obs[["imagecol", "imagerow"]].values, dtype=float)
    if coor.shape[0] < 2:
        return float("nan")
    dist, _ = cKDTree(coor).query(coor, k=2)
    return float(np.median(dist[:, 1]))


def adjust_by_morphology(st, adata, radius, isolated) -> None:
    """``st.spatial.morphology.adjust`` on ``X_pca``, with spots that have no neighbour kept out of it.

    Upstream's loop gives a spot with no neighbour inside ``radius`` the value it computed for the
    *previous* spot (its ``else`` branch reuses the last iteration's variable) and raises
    UnboundLocalError when the first spot is such a spot. The value it means to give them is their
    own embedding. A spot with no neighbour is nobody's neighbour, so running adjust on the other
    spots leaves their results exactly as a full call would, and the isolated rows keep ``X_pca``.
    """
    import numpy as np

    if not isolated.any():
        st.spatial.morphology.adjust(adata, use_data="X_pca", radius=radius, method="mean")
        return
    import anndata

    keep = ~isolated
    sub = anndata.AnnData(obs=adata.obs.loc[keep, ["imagecol", "imagerow"]].copy())
    sub.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"])[keep]
    sub.obsm["X_morphology"] = np.asarray(adata.obsm["X_morphology"])[keep]
    st.spatial.morphology.adjust(sub, use_data="X_pca", radius=radius, method="mean")
    adjusted_kept = np.asarray(sub.obsm["X_pca_morphology"])
    adjusted = np.array(adata.obsm["X_pca"], dtype=adjusted_kept.dtype, copy=True)
    adjusted[keep] = adjusted_kept
    adata.obsm["X_pca_morphology"] = adjusted


def scale_for_pca(st, adata) -> bool:
    """Scale genes to unit variance; returns whether ``X`` was zero-centred here.

    A sparse ``X`` is scaled without centring (zero-centring densifies it: ``n_spots x n_genes``
    floats) and the PCA that follows centres it implicitly, which gives the same components. A
    dense ``X`` is z-scored in place, as before.
    """
    from scipy import sparse

    if sparse.issparse(adata.X):
        st.pp.scale(adata, zero_center=False)
        return False
    st.pp.scale(adata)
    return True


def add_image_coordinates(st, adata, library_id, use_quality: str) -> None:
    """On the PCA-only path, still write ``obs['imagecol'/'imagerow']`` where ``convert_scanpy`` can.

    The clustering does not read them; they are what earlier runs of this path wrote to the
    annotated h5ad whenever the library had the scale factor, so they stay.
    """
    if library_id is None or "spatial" not in adata.obsm:
        return
    try:
        spatial_coords(adata, "spatial", want=2, tool=TOOL)
        lib = adata.uns["spatial"][library_id]
        scalefactors = lib.get("scalefactors", None)
        if use_quality == "fulres" or (
            isinstance(scalefactors, Mapping) and f"tissue_{use_quality}_scalef" in scalefactors
        ):
            st.convert_scanpy(adata, use_quality=use_quality)
    except Exception as e:  # informational columns only
        log(f"Not adding obs['imagecol'/'imagerow']: {e}")


def _atomic_write_csv(df, path: str) -> None:
    tmp = path + ".partial"
    df.to_csv(tmp, index=True)
    os.replace(tmp, path)


def _atomic_write_h5ad(adata, path: str) -> None:
    tmp = path + ".partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def run_stlearn_clustering(
    st_h5ad: str,
    output_dir: str,
    n_pcs: int = 50,
    n_neighbors: int = 25,
    radius: int = 50,
    crop_size: int = 40,
    resolution: float = 1.0,
    random_state: int = 0,
    use_quality: str = "hires",
    allow_pca_fallback: bool = False,
    use_raw_counts: bool = False,
):
    import numpy as np
    import scanpy as sc
    import stlearn as st

    os.makedirs(output_dir, exist_ok=True)
    run_warnings = []

    log(f"Loading spatial AnnData from: {st_h5ad}")
    adata = sc.read_h5ad(st_h5ad)
    log(f"Loaded ST data: n_spots={adata.n_obs}, n_genes={adata.n_vars}")

    # Background spots (obs['in_tissue'] == 0) are not tissue: left out before anything is computed.
    adata, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata, "spots")
    if n_spots_off_tissue:
        log(
            f"Left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0 (background); "
            f"{adata.n_obs} in-tissue spots remain."
        )

    # stLearn normalises X as counts: a scaled X is refused, a normalised one flagged, adata.raw on request.
    adata, counts_info = choose_counts_matrix(adata, use_raw_counts)
    log(f"Expression matrix: {counts_info['expression_source']} ({counts_info['x_matrix_kind']})")
    n_genes_input = int(adata.n_vars)

    # Make indices unique & convert to stLearn format
    renamed = make_names_unique_and_report(adata)

    library_id, other_keys = pick_spatial_library(adata)
    n_libraries = 0
    if library_id is not None:
        n_libraries = sum(1 for entry in adata.uns["spatial"].values() if _is_library(entry))
    blocker = stsme_blocker(adata, library_id, use_quality)
    if blocker:
        if not allow_pca_fallback:
            raise ValueError(refusal_message(blocker))
        log(f"WARNING: stSME cannot run ({blocker}); allow_pca_fallback=True, clustering expression PCA alone.")
    if library_id is not None and other_keys:
        log(f"Using uns['spatial'] library '{library_id}'; set aside while stLearn runs: {other_keys}")
    if n_libraries > 1:
        run_warnings.append(
            f"uns['spatial'] holds {n_libraries} libraries; stLearn used library '{library_id}' for every spot."
        )
    set_aside = isolate_library(adata, library_id)

    isolated = None
    if not blocker:
        # Refuses a 3-column obsm['spatial'] (a stack) instead of letting convert_scanpy drop the z.
        spatial_coords(adata, "spatial", want=2, tool=TOOL)
        log(f"Converting Scanpy AnnData to stLearn format (use_quality='{use_quality}')")
        adata = st.convert_scanpy(adata, use_quality=use_quality)
        isolated = spots_without_neighbours(adata, radius)
        if bool(isolated.all()):
            raise ValueError(
                f"radius={radius} is smaller than the spot spacing on the '{use_quality}' image (median "
                f"nearest-neighbour distance {nearest_spacing(adata):.1f} image pixels): no spot has another "
                "spot within radius, so stSME would adjust nothing. Pass a larger radius (it is measured in "
                f"pixels of the '{use_quality}' image)."
            )
    else:
        add_image_coordinates(st, adata, library_id, use_quality)

    # Raw count layer
    if "raw_count" not in adata.layers:
        log("Adding layers['raw_count'] from adata.X")
        adata.layers["raw_count"] = adata.X.copy()

    # Basic preprocessing
    log("Filtering genes, normalize_total, log1p, scale")
    st.pp.filter_genes(adata, min_cells=3)
    st.pp.normalize_total(adata)
    st.pp.log1p(adata)
    adata.raw = adata
    zero_centred = scale_for_pca(st, adata)

    # PCA (centres implicitly when X was left sparse)
    log(f"Running PCA (n_pcs={n_pcs}, random_state={random_state})")
    np.random.seed(random_state)
    st.em.run_pca(adata, n_comps=n_pcs, random_state=random_state, zero_center=True)

    use_rep = "X_pca"
    tile_dir = None
    cluster_plot_png = None
    fallback_why = blocker

    if not blocker:
        # Tiling & feature extraction based on histology
        tiles_to = os.path.join(output_dir, "tiling")
        os.makedirs(tiles_to, exist_ok=True)
        log("Histology image found, running stSME pipeline.")
        try:
            log(f"Tiling histology image to: {tiles_to} (crop_size={crop_size})")
            st.pp.tiling(adata, out_path=tiles_to, library_id=library_id, crop_size=crop_size)
            log("Extracting CNN-based image features")
            st.pp.extract_feature(adata)
            log(f"Applying morphology.adjust (radius={radius}, use_data='X_pca')")
            adjust_by_morphology(st, adata, radius, isolated)
            use_rep = "X_pca_morphology"
            tile_dir = tiles_to
        except Exception as e:
            why = f"stSME failed ({type(e).__name__}: {e})"
            if not allow_pca_fallback:
                raise RuntimeError(
                    f"{why}. The run stops rather than cluster without the image; pass allow_pca_fallback=True "
                    "to accept expression-PCA-only clustering instead (recorded as a fallback in the payload)."
                ) from e
            log(f"WARNING: {why}; allow_pca_fallback=True, clustering expression PCA alone.")
            fallback_why = why
    else:
        log("stSME not run; using PCA-only features (allow_pca_fallback=True).")

    stsme_ran = use_rep == "X_pca_morphology"
    n_isolated = int(isolated.sum()) if (stsme_ran and isolated is not None) else 0
    if n_isolated:
        run_warnings.append(
            f"{n_isolated} of {adata.n_obs} spots have no other spot within radius={radius} pixels of the "
            f"'{use_quality}' image; stSME left their expression PCA unadjusted."
        )

    # Neighbors & clustering
    log(f"Computing neighbors (n_neighbors={n_neighbors}, use_rep='{use_rep}')")
    st.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        random_state=random_state,
    )

    log(f"Running Louvain clustering (resolution={resolution}, random_state={random_state})")
    st.tl.clustering.louvain(
        adata,
        resolution=resolution,
        random_state=random_state,
    )

    cluster_key = "louvain"
    log(f"Cluster labels stored in adata.obs['{cluster_key}']")

    # Save cluster labels CSV
    cluster_csv = os.path.join(output_dir, "stlearn_clusters_per_spot.csv")
    log(f"Saving cluster labels to: {cluster_csv}")
    _atomic_write_csv(adata.obs[cluster_key].astype(str).to_frame(name=cluster_key), cluster_csv)

    # Optional cluster plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cluster_plot_png = os.path.join(output_dir, "stlearn_cluster_plot.png")
        log(f"Saving cluster plot to: {cluster_plot_png}")
        st.pl.cluster_plot(
            adata,
            use_label=cluster_key,
            image_alpha=1.0,
            size=7,
            fname=cluster_plot_png,
            dpi=150,
            # stlearn's knob is show_plot, not show. `show=False` raised TypeError before the figure
            # was ever saved, and the except below turned that into a warning, so every run reported
            # cluster_plot_png=None.
            show_plot=False,
        )
        plt.close("all")
    except Exception as e:
        log(f"WARNING: Failed to generate cluster plot: {e}")
        run_warnings.append(f"cluster plot not written: {type(e).__name__}: {e}")
        cluster_plot_png = None

    # Save annotated h5ad, with the uns['spatial'] entries that were set aside put back
    restore_library(adata, set_aside, library_id)
    annotated_h5ad = os.path.join(output_dir, "stlearn_clustering.h5ad")
    log(f"Writing annotated AnnData to: {annotated_h5ad}")
    _atomic_write_h5ad(adata, annotated_h5ad)

    n_spots = int(adata.n_obs)
    n_genes = int(adata.n_vars)
    cluster_sizes = {str(k): int(v) for k, v in adata.obs[cluster_key].value_counts().items()}
    n_clusters = len(cluster_sizes)

    out = WorkerOutput("stlearn", task="clustering")
    out.set_data(
        n_spots=n_spots,
        n_spots_input=int(n_spots_supplied),
        n_spots_used=n_spots,
        n_genes=n_genes,
        n_genes_input=n_genes_input,
    )
    out.add_output_files(
        {
            "annotated_h5ad": annotated_h5ad,
            "cluster_csv": cluster_csv,
            "cluster_plot_png": cluster_plot_png,
            "tiling_dir": tile_dir,
        }
    )
    params = {
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "random_state": random_state,
        "use_quality": use_quality,
        "allow_pca_fallback": bool(allow_pca_fallback),
        "use_raw_counts": bool(use_raw_counts),
        "use_rep": use_rep,
        "library_id": None if library_id is None else str(library_id),
        "uns_spatial_other_keys": other_keys,
        "scale_zero_center": bool(zero_centred),
        "min_cells": 3,
    }
    if stsme_ran:
        params["radius"] = radius
        params["crop_size"] = crop_size
        params["n_spots_without_neighbours_in_radius"] = n_isolated
    out.add_params(params)
    out.add_params(identifier_rename_params(renamed))
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    record_expression_source(out, counts_info)
    if stsme_ran:
        record_method(out, STSME_METHOD)
    else:
        record_method(out, PCA_ONLY_METHOD, used_fallback=True, why=fallback_why)
        record_ignored(out, list(STSME_ONLY_PARAMS), f"stSME did not run ({fallback_why})")
    for message in run_warnings:
        out.add_warning(message)
    out.set_summary(
        n_clusters=n_clusters,
        cluster_key=cluster_key,
        cluster_sizes=cluster_sizes,
    )
    if stsme_ran:
        lead = (
            f"Method: stSME -- expression PCA adjusted by histology morphology (library '{library_id}', "
            f"'{use_quality}' image, radius={radius}), then Louvain. "
        )
    else:
        lead = (
            f"FALLBACK (allow_pca_fallback=True): stSME did not run ({fallback_why}). These clusters are "
            "Louvain on expression PCA alone; no image and no spatial information entered them. "
        )
    spots_note = ""
    if n_spots_off_tissue:
        spots_note = (
            f" {n_spots_off_tissue} of the {n_spots_supplied} spots supplied are marked obs['in_tissue'] == 0 "
            f"(background outside the tissue) and were left out; the {n_spots} in-tissue spots were clustered."
        )
    if counts_info["expression_source"] != "X":
        spots_note += " Expression was read from adata.raw (use_raw_counts=True)."
    out.set_analysis(
        lead
        + build_cluster_analysis(cluster_sizes, cluster_key="cluster", total_spots=n_spots)
        + spots_note
        + identifier_rename_note(renamed)
    )
    return out.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="stLearn worker for spatial clustering (SME + Louvain).")
    parser.add_argument(
        "--task",
        choices=["clustering"],
        default="clustering",
        help="Currently only 'clustering' is supported.",
    )
    parser.add_argument(
        "--st-h5ad",
        dest="st_h5ad",
        required=True,
        help="Path to spatial AnnData .h5ad.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        help="Directory for all outputs (h5ad, CSV, plots, tiles).",
    )
    parser.add_argument(
        "--n-pcs",
        dest="n_pcs",
        type=int,
        default=50,
        help="Number of principal components.",
    )
    parser.add_argument(
        "--n-neighbors",
        dest="n_neighbors",
        type=int,
        default=25,
        help="Number of neighbors for kNN graph.",
    )
    parser.add_argument(
        "--radius",
        dest="radius",
        type=int,
        default=50,
        help="Radius (pixels of the use_quality image) for morphology.adjust in stSME.",
    )
    parser.add_argument(
        "--crop-size",
        dest="crop_size",
        type=int,
        default=40,
        help="Tile size (pixels) for st.pp.tiling.",
    )
    parser.add_argument(
        "--resolution",
        dest="resolution",
        type=float,
        default=1.0,
        help="Louvain resolution parameter.",
    )
    parser.add_argument(
        "--random-state",
        dest="random_state",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--use-quality",
        dest="use_quality",
        type=str,
        default="hires",
        choices=["hires", "lowres", "fulres"],
        help="Which image quality from uns['spatial'] to use.",
    )
    parser.add_argument(
        "--allow-pca-fallback",
        dest="allow_pca_fallback",
        action="store_true",
        default=False,
        help=(
            "When stSME cannot run (no histology image, no use_quality image, or a failure in tiling / "
            "feature extraction / morphology.adjust), cluster the expression PCA alone instead of stopping. "
            "Recorded as a fallback in params.method / params.used_fallback."
        ),
    )
    parser.add_argument(
        "--use-raw-counts",
        dest="use_raw_counts",
        action="store_true",
        default=False,
        help=(
            "Run on adata.raw.X instead of X (for an h5ad whose X is normalised or scaled and whose counts "
            "sit in adata.raw). Refused when there is no adata.raw or it does not hold counts."
        ),
    )

    args = parser.parse_args()

    log(f"Task           = {args.task}")
    log(f"ST h5ad        = {args.st_h5ad}")
    log(f"output_dir     = {args.output_dir}")
    log(f"n_pcs          = {args.n_pcs}")
    log(f"n_neighbors    = {args.n_neighbors}")
    log(f"radius         = {args.radius}")
    log(f"crop_size      = {args.crop_size}")
    log(f"resolution     = {args.resolution}")
    log(f"random_state   = {args.random_state}")
    log(f"use_quality    = {args.use_quality}")
    log(f"allow_pca_fallback = {args.allow_pca_fallback}")
    log(f"use_raw_counts = {args.use_raw_counts}")

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.task == "clustering":
            # stLearn prints its progress messages to stdout; stdout carries only the JSON.
            with contextlib.redirect_stdout(sys.stderr):
                result = run_stlearn_clustering(
                    st_h5ad=args.st_h5ad,
                    output_dir=args.output_dir,
                    n_pcs=args.n_pcs,
                    n_neighbors=args.n_neighbors,
                    radius=args.radius,
                    crop_size=args.crop_size,
                    resolution=args.resolution,
                    random_state=args.random_state,
                    use_quality=args.use_quality,
                    allow_pca_fallback=args.allow_pca_fallback,
                    use_raw_counts=args.use_raw_counts,
                )
        else:
            raise ValueError(f"Unsupported task: {args.task}")

        # ONLY JSON to stdout
        print(json.dumps(result))
    except Exception as e:
        tb = traceback.format_exc()
        log("ERROR:")
        log(tb)
        WorkerOutput.emit_error("stlearn", str(e), task="clustering")
        sys.exit(1)


if __name__ == "__main__":
    main()
