#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from worker_utils import (
    WorkerOutput,
    build_cluster_analysis,
    describe_reduction,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    record_ignored,
    record_in_tissue,
    record_method,
    require_hvg_flavor,
    resolve_compute,
    spatial_coords,
    unsupported_choice_msg,
)

#: What actually runs: STAGATE_pyG's graph attention auto-encoder, then KMeans on its embedding.
#: (mclust, which the STAGATE tutorials use for the last step, is not run by this wrapper.)
METHOD_NAME = "STAGATE (STAGATE_pyG graph attention auto-encoder) embedding, clustered with KMeans"

#: HVG flavours the worker can select genes with. ``seurat_v3`` ranks raw counts (scored before
#: normalisation) and needs scikit-misc; the other two rank log-normalised dispersion (scored after).
HVG_FLAVORS = ("seurat_v3", "seurat", "cell_ranger")

#: rad_cutoff=0 derives the radius from the data: AUTO_RADIUS_SLACK x the median distance from a spot
#: to its AUTO_RADIUS_NEIGHBOURS-th nearest spot. On a Visium hexagonal grid that is the six-spot
#: first ring (the second ring sits at sqrt(3) x the spacing), which is the graph STAGATE's own
#: Visium setting builds: rad_cutoff=150 on the DLPFC full-resolution pixel coordinates, whose spot
#: spacing is ~137, links each spot to the same six neighbours.
AUTO_RADIUS_NEIGHBOURS = 6
AUTO_RADIUS_SLACK = 1.2

#: A graph that leaves more than this fraction of the spots with no neighbour is refused: most of the
#: slide would be embedded from its own expression only, which is not a spatial method.
MAX_ISOLATED_FRACTION = 0.5

# Device spellings that ask for an accelerator (worker_utils.resolve_compute's vocabulary).
_GPU_REQUEST = re.compile(r"^(?:gpu|cuda|true|yes|on|(?:gpu|cuda)[:_ ]?\d+|\d+)$")


def log(msg: str) -> None:
    """Log to stderr with a consistent prefix."""
    print(f"[stagate-worker] {msg}", file=sys.stderr, flush=True)


def _write_h5ad_atomic(adata: ad.AnnData, path: str) -> None:
    tmp = f"{path}.partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def _write_csv_atomic(frame, path: str) -> None:
    tmp = f"{path}.partial"
    frame.to_csv(tmp, index=True)
    os.replace(tmp, path)


def _save_npy_atomic(array, path: str) -> None:
    # Through an open handle: np.save(<name>) appends '.npy' to a name that does not end in it.
    tmp = f"{path}.partial"
    with open(tmp, "wb") as fh:
        np.save(fh, np.asarray(array))
    os.replace(tmp, path)


def resolve_device(device_str: str):
    """Resolve any device spelling ('auto', 'cpu', 'CPU', 'gpu', 'GPU', 'cuda[:id]') into a torch.device.

    Shared with every other worker (worker_utils.resolve_compute). Two behaviours this used to get
    wrong: 'GPU' fell through to the unrecognised branch and ran on the CPU even when a GPU was
    present, and 'cuda' was honoured without checking that CUDA exists.
    """
    import torch

    return torch.device(resolve_compute(device_str).device)


def device_note(requested, resolved: str) -> str:
    """A sentence for the payload when an accelerator was asked for and the run used the CPU.

    ``resolve_compute`` degrades such a request to 'cpu' and says so on stderr only, which a
    successful run never carries into its payload.
    """
    text = str(requested).strip().lower()
    if resolved == "cpu" and _GPU_REQUEST.match(text):
        return f"device='{requested}' was requested but no CUDA device is available here; STAGATE trained on the CPU."
    return ""


def non_count_note(matrix, source: str) -> str:
    """A sentence for the payload when the expression matrix does not hold raw counts.

    The preprocessing below normalises every spot to 10,000 and log-transforms, so a matrix that is
    already normalised or logged is transformed a second time. Every stored value is checked,
    in fixed-size slices so the check allocates no copy of a large matrix.
    """
    import scipy.sparse as sp

    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()
    step = 10_000_000
    for start in range(0, int(values.size), step):
        part = np.asarray(values[start : start + step])
        bad = (part < 0) | (part != np.floor(part))
        if bad.any():
            example = float(part[bad][0])
            return (
                f"{source} is not a raw-count matrix (it holds non-integer or negative values, e.g. {example:.4g}). "
                "STAGATE's preprocessing here normalises each spot to 10,000 and log-transforms whatever it is "
                "given, so an already-normalised matrix is transformed twice. Pass layer_key naming a raw-count "
                "layer if the file has one."
            )
    return ""


def _normalise_and_log(adata: ad.AnnData) -> None:
    """normalize_total(1e4) + log1p over every gene ``adata`` holds, in place."""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def preprocess_adata(adata: ad.AnnData, n_hvg: int = 3000, hvg_flavor: str = "seurat_v3") -> ad.AnnData:
    """STAGATE's tutorial preprocessing: detection filter, HVG flags, normalise + log1p, then the HVG subset.

    The order is the tutorial's (``highly_variable_genes``; ``normalize_total``; ``log1p``; then
    ``train_STAGATE`` takes the highly-variable columns itself): each spot is scaled by its library
    size over EVERY detected gene, and only then are the HVG columns kept. Subsetting first -- as this
    wrapper used to -- scaled each spot by its counts in the n_hvg genes alone, whose share of a spot's
    library varies ~4x across a slide (0.13-0.55 on V1_Human_Lymph_Node at 3,000 seurat_v3 genes), so
    every value STAGATE trained on was off by that spot's factor.

    ``seurat_v3`` ranks the raw counts, so it is scored before normalisation; ``seurat`` and
    ``cell_ranger`` rank log-normalised dispersion, so they are scored after it. ``hvg_flavor`` is the
    only flavour that runs: when scikit-misc (seurat_v3) is missing or broken the run stops and says so.
    It used to switch to ``flavor='seurat'`` and report nothing.
    """
    log(f"Preprocessing AnnData: min_cells=3 gene filter, {n_hvg} HVGs (flavor={hvg_flavor}), normalize, log1p")
    sc.pp.filter_genes(adata, min_cells=3)
    n_top = min(int(n_hvg), int(adata.n_vars))
    ranks_raw_counts = hvg_flavor == "seurat_v3"
    if not ranks_raw_counts:
        _normalise_and_log(adata)
    try:
        sc.pp.highly_variable_genes(adata, flavor=hvg_flavor, n_top_genes=n_top)
    except Exception as exc:
        require_hvg_flavor(hvg_flavor)  # raises ImportError naming scikit-misc when that is the cause
        raise RuntimeError(
            f"highly_variable_genes(flavor='{hvg_flavor}') failed and no other flavour was substituted: {exc}. "
            "Pass hvg_flavor='seurat' to rank log-normalised dispersion instead."
        ) from exc
    if ranks_raw_counts:
        _normalise_and_log(adata)
    if "highly_variable" in adata.var:
        adata._inplace_subset_var(adata.var["highly_variable"].values)
    return adata


def measure_spatial_graph(coords, rad_cutoff: float = 0.0, k_cutoff: int = 0, where: str = "obsm['spatial']") -> dict:
    """Decide the spatial graph STAGATE will get, and measure it before anything is trained.

    Uses the rule upstream ``Cal_Spatial_Net`` uses (sklearn neighbours; a spot is linked to every
    other spot within the radius, boundary included, or to its k nearest; zero-distance pairs are
    dropped), so the counts are the graph's own. ``k_cutoff > 0`` selects the KNN graph;
    otherwise a radius graph is built with ``rad_cutoff``, or -- when ``rad_cutoff`` is 0 -- with a
    radius derived from the spot spacing (see ``AUTO_RADIUS_NEIGHBOURS``).

    Raises when the graph would leave most spots with no neighbour: upstream adds a self-loop to
    every spot, so an empty graph trains without complaint into a non-spatial auto-encoder.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=np.float64)
    n = int(coords.shape[0])
    if n < 2:
        raise ValueError(f"STAGATE needs at least 2 spots to build a spatial graph; the input has {n}.")
    finite = np.isfinite(coords).all(axis=1)
    if not finite.all():
        raise ValueError(
            f"{int((~finite).sum())} of {n} spots have a missing or infinite coordinate in {where}; "
            "a spatial graph cannot place them. Fix or remove those spots before running STAGATE."
        )
    rad = float(rad_cutoff)
    if not rad >= 0.0:
        raise ValueError(
            f"rad_cutoff must be 0 (derive the radius from the data) or a positive radius; got {rad_cutoff}."
        )
    k = int(k_cutoff)
    if k < 0:
        raise ValueError(f"k_cutoff must be 0 (use a radius graph) or a positive neighbour count; got {k_cutoff}.")
    if k >= n:
        raise ValueError(f"k_cutoff={k} asks for {k} neighbours per spot but the input has only {n} spots.")

    ring_k = min(AUTO_RADIUS_NEIGHBOURS, n - 1)
    dist, _ = NearestNeighbors(n_neighbors=ring_k + 1).fit(coords).kneighbors(coords)
    nearest = float(np.median(dist[:, 1]))
    ring = float(np.median(dist[:, ring_k]))
    derived = AUTO_RADIUS_SLACK * ring

    if k > 0:
        model, radius, source = "KNN", None, None
        kdist, _ = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)
        per_spot = (kdist > 0).sum(axis=1)
    else:
        model = "Radius"
        radius = rad if rad > 0 else derived
        source = "caller" if rad > 0 else "derived"
        per_spot = np.zeros(n, dtype=np.int64)
        if radius > 0:
            nbrs = NearestNeighbors(radius=radius).fit(coords)
            step = 4096  # bounded memory: the neighbour lists of one slice of spots at a time
            for start in range(0, n, step):
                rows = nbrs.radius_neighbors(coords[start : start + step], return_distance=True)[0]
                per_spot[start : start + len(rows)] = [int((row > 0).sum()) for row in rows]

    graph = {
        "model": model,
        "rad_cutoff": None if radius is None else float(radius),
        "rad_cutoff_source": source,
        "k_cutoff": k if k > 0 else None,
        "n_spots": n,
        "median_nn_distance": nearest,
        "median_ring_distance": ring,
        "derived_rad_cutoff": float(derived),
    }
    graph.update(_edge_counts(per_spot, n))
    refuse_degenerate_graph(graph, where)
    return graph


def _edge_counts(per_spot, n: int) -> dict:
    n_edges = int(np.sum(per_spot))
    return {
        "n_edges": n_edges,
        "mean_neighbours": float(n_edges) / float(n) if n else 0.0,
        "n_isolated_spots": int(n - np.count_nonzero(per_spot)),
    }


def graph_from_spatial_net(net, obs_names, graph: dict) -> dict:
    """Re-count the graph from the ``Spatial_Net`` table upstream actually built, when it built one."""
    import pandas as pd

    names = pd.Index([str(x) for x in obs_names])
    per_spot = net["Cell1"].astype(str).value_counts().reindex(names, fill_value=0).to_numpy()
    measured = dict(graph)
    measured.update(_edge_counts(per_spot, len(names)))
    return measured


def refuse_degenerate_graph(graph: dict, where: str) -> None:
    """Stop, with the numbers and the knobs, when the graph cannot make the method spatial."""
    n = graph["n_spots"]
    isolated = graph["n_isolated_spots"]
    if graph["n_edges"] > 0 and isolated <= MAX_ISOLATED_FRACTION * n:
        return
    if graph["model"] == "KNN":
        what = f"k_cutoff={graph['k_cutoff']}"
    elif graph["rad_cutoff_source"] == "derived":
        what = f"rad_cutoff=0 (derived radius {graph['rad_cutoff']:.4g})"
    else:
        what = f"rad_cutoff={graph['rad_cutoff']:.4g}"
    if graph["median_ring_distance"] <= 0:
        raise ValueError(
            f"{what} cannot build a spatial graph: the coordinates in {where} do not separate the spots (the "
            f"median distance from a spot to its {AUTO_RADIUS_NEIGHBOURS}th-nearest spot is 0). Check that "
            "spatial_key names real per-spot coordinates."
        )
    knn = f"k_cutoff={AUTO_RADIUS_NEIGHBOURS} for a {AUTO_RADIUS_NEIGHBOURS}-nearest-neighbour graph"
    if graph["model"] == "KNN":
        remedy = "Spots that share a coordinate are not linked (zero-distance pairs are dropped); check spatial_key."
    elif graph["rad_cutoff_source"] == "derived":
        remedy = f"Pass a larger rad_cutoff, or {knn}."
    else:
        remedy = (
            f"Pass rad_cutoff=0 to derive the radius from the data (here {graph['derived_rad_cutoff']:.4g}, "
            f"{AUTO_RADIUS_SLACK} x the median distance to the {AUTO_RADIUS_NEIGHBOURS}th-nearest spot), a larger "
            f"rad_cutoff, or {knn}."
        )
    raise ValueError(
        f"{what} leaves {isolated} of {n} spots ({100.0 * isolated / n:.1f}%) with no neighbour in the spatial "
        f"graph ({graph['n_edges']} edges). The median distance from a spot to its nearest spot in {where} is "
        f"{graph['median_nn_distance']:.4g} (same units). STAGATE adds a self-loop to every spot, so it would "
        f"train anyway, as a non-spatial auto-encoder; the run was stopped instead. {remedy}"
    )


def describe_graph(graph: dict) -> str:
    """One sentence naming the graph the embedding was trained on."""
    if graph["model"] == "KNN":
        how = f"each spot linked to its {graph['k_cutoff']} nearest spots"
    elif graph["rad_cutoff_source"] == "derived":
        how = (
            f"radius {graph['rad_cutoff']:.4g} derived from the data ({AUTO_RADIUS_SLACK} x the median distance "
            f"to the {AUTO_RADIUS_NEIGHBOURS}th-nearest spot, {graph['median_ring_distance']:.4g})"
        )
    else:
        how = (
            f"radius {graph['rad_cutoff']:.4g} as requested (median nearest-spot distance "
            f"{graph['median_nn_distance']:.4g})"
        )
    isolated = graph["n_isolated_spots"]
    lonely = (
        f"; {isolated} spot(s) had no neighbour and were embedded from their own expression only" if isolated else ""
    )
    return (
        f" STAGATE trained on a spatial neighbour graph with {how}: {graph['n_edges']} edges, "
        f"{graph['mean_neighbours']:.1f} neighbours per spot on average{lonely}."
    )


def run_stagate_domains(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    rad_cutoff: float = 0.0,
    k_cutoff: int = 0,
    n_epochs: int = 1000,
    n_clusters: int = 6,
    device: str = "auto",
    seed: int = 0,
    n_hvg: int = 3000,
    layer_key: str = "",
    hvg_flavor: str = "seurat_v3",
) -> dict:
    """
    Run STAGATE (pyG) for spatial domain identification on a single ST dataset.

    Returns a small JSON-serializable dict summarizing the run.
    """
    np.random.seed(seed)
    log(f"Loading AnnData from {st_h5ad}")
    adata = ad.read_h5ad(st_h5ad)
    log(f"Loaded ST data: n_spots={adata.n_obs}, n_genes={adata.n_vars}")
    # What the user handed us, captured before preprocess_adata() cuts the gene axis below.
    n_genes_supplied = int(adata.n_vars)

    # Select the expression source. When a layer is requested, copy it into adata.X so the
    # downstream HVG/normalize/log1p operate on it. The portal documents `layer_key` and emits
    # `--layer-key`; without the matching argparse flag + this branch the whole run hard-failed
    # ("unrecognized arguments: --layer-key") whenever a caller set it. Validate the layer up
    # front so a bad key fails fast with a clear message (before the heavy STAGATE import).
    if layer_key:
        if layer_key not in adata.layers:
            raise ValueError(
                f"layer_key='{layer_key}' not found in adata.layers. Available layers: {list(adata.layers.keys())}"
            )
        log(f"Using adata.layers['{layer_key}'] as the expression matrix")
        adata.X = adata.layers[layer_key].copy()

    if hvg_flavor not in HVG_FLAVORS:
        raise ValueError(unsupported_choice_msg("hvg_flavor", hvg_flavor, HVG_FLAVORS))

    # Background spots (obs['in_tissue'] == 0: the glass around a CELLxGENE Visium export, 56-70% of
    # the spots on the library's such samples) are not tissue. They are left out before the graph is
    # built and reported, as every spot-clustering tool here does.
    adata, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata, "spots")
    if n_spots_off_tissue:
        log(f"Leaving out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0")

    # Ensure spatial coordinates are available under 'spatial'. Read through the shared guard on
    # every key, 'spatial' included: a three-column obsm['spatial'] used to reach upstream
    # Cal_Spatial_Net, which died on a pandas column-length mismatch.
    if spatial_key not in adata.obsm:
        raise ValueError(
            f"spatial_key='{spatial_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}"
        )
    coords, _ = spatial_coords(adata, spatial_key, want=2, tool="STAGATE")
    if spatial_key != "spatial":
        log(f"Copying coordinates from obsm['{spatial_key}'] to obsm['spatial']")
        adata.obsm["spatial"] = coords

    # STAGATE keys its graph by obs_names: Cal_Spatial_Net writes names, Transfer_pytorch_Data maps
    # them back with a dict, so a duplicated barcode wires every copy's edges onto its last copy.
    renamed = make_names_unique_and_report(adata)

    # Measure the graph before any training. rad_cutoff=150 was the default, and in obsm units it
    # linked no two spots on many Visium slides; upstream adds self-loops and trains regardless.
    where = f"obsm['{spatial_key}']"
    graph = measure_spatial_graph(coords, rad_cutoff=rad_cutoff, k_cutoff=k_cutoff, where=where)
    log(
        f"Spatial graph plan: model={graph['model']} rad_cutoff={graph['rad_cutoff']} k_cutoff={graph['k_cutoff']} "
        f"-> {graph['n_edges']} edges, {graph['mean_neighbours']:.2f} neighbours/spot, "
        f"{graph['n_isolated_spots']} isolated"
    )

    counts_note = non_count_note(adata.X, f"adata.layers['{layer_key}']" if layer_key else "adata.X")

    # Preprocess (HVGs, normalization, log1p)
    adata = preprocess_adata(adata, n_hvg=n_hvg, hvg_flavor=hvg_flavor)

    # STAGATE_pyG package name. Imported here (after input validation + preprocessing) so a bad
    # layer_key/spatial_key fails fast with a clear message instead of after a heavy torch import.
    import STAGATE_pyG as STAGATE  # type: ignore

    # Build the spatial neighbour graph. Upstream's model defaults to 'Radius' and reads k_cutoff only
    # under model='KNN', so k_cutoff used to have no effect at all.
    if graph["model"] == "KNN":
        log(f"Building spatial neighbor graph: KNN with k_cutoff={graph['k_cutoff']}")
        STAGATE.Cal_Spatial_Net(adata, k_cutoff=graph["k_cutoff"], model="KNN")
    else:
        log(f"Building spatial neighbor graph: Radius with rad_cutoff={graph['rad_cutoff']}")
        STAGATE.Cal_Spatial_Net(adata, rad_cutoff=graph["rad_cutoff"])
    net = adata.uns.get("Spatial_Net")
    if net is not None and hasattr(net, "columns") and "Cell1" in net.columns:
        # The table upstream built is the graph that trains; count it, and refuse it if it is empty.
        graph = graph_from_spatial_net(net, adata.obs_names, graph)
        refuse_degenerate_graph(graph, where)
    STAGATE.Stats_Spatial_Net(adata)

    # Train STAGATE
    torch_device = resolve_device(device)
    log(f"Training STAGATE on device '{torch_device}' for {n_epochs} epochs")
    STAGATE.train_STAGATE(
        adata,
        n_epochs=n_epochs,
        random_seed=seed,
        device=torch_device,
        save_loss=False,
        save_reconstrction=False,
    )

    if "STAGATE" not in adata.obsm:
        raise RuntimeError("STAGATE embedding not found in adata.obsm['STAGATE'] after training")

    emb = adata.obsm["STAGATE"]
    log(f"STAGATE embedding shape: {emb.shape}")

    # Cluster in the STAGATE embedding space
    log(f"Running KMeans with n_clusters={n_clusters} and seed={seed}")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(emb)

    cluster_key = "stagate_domain"
    # Cast via string so pandas Categorical has object-dtype categories.
    # scanpy's rank_genes_groups / dendrogram internally call ",".join(categories)
    # which TypeErrors on int categories.
    adata.obs[cluster_key] = cluster_labels.astype(int).astype(str)
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

    # Optionally compute neighbors/UMAP for visualization
    umap_note = ""
    try:
        log("Computing neighbors/UMAP for visualization (use_rep='STAGATE')")
        sc.pp.neighbors(adata, use_rep="STAGATE")
        sc.tl.umap(adata, min_dist=0.3, spread=1.0)
    except Exception as e:
        log(f"UMAP computation failed (non-fatal): {str(e)}")
        umap_note = (
            f"The visualisation UMAP failed ({e}); the h5ad carries no obsm['X_umap']. The domains are unaffected."
        )

    # Write outputs
    os.makedirs(output_dir, exist_ok=True)
    h5ad_out = os.path.join(output_dir, "stagate_domains.h5ad")
    csv_out = os.path.join(output_dir, "stagate_domain_assignments.csv")
    emb_out = os.path.join(output_dir, "stagate_embedding.npy")

    # Each output is written to '<name>.partial' and renamed onto its final name, so a run killed
    # mid-write (the per-call budget) leaves no truncated file under a name the payload keys on.
    log(f"Writing annotated AnnData to {h5ad_out}")
    _write_h5ad_atomic(adata, h5ad_out)

    log(f"Writing cluster assignments to {csv_out}")
    _write_csv_atomic(adata.obs[[cluster_key]], csv_out)

    log(f"Saving STAGATE embedding to {emb_out}")
    _save_npy_atomic(emb, emb_out)

    # Summarize clusters
    counts = adata.obs[cluster_key].value_counts().sort_index()
    cluster_sizes = {int(k): int(v) for k, v in counts.items()}

    n_spots = int(adata.n_obs)
    n_genes_used = int(adata.n_vars)

    gene_note = describe_reduction(
        "genes",
        n_genes_supplied,
        n_genes_used,
        f"STAGATE's tutorial preprocessing (run by this tool), which drops genes detected in fewer than 3 spots "
        f"and then keeps the {n_hvg} most highly variable",
    )

    torch_device_str = str(torch_device)
    out = WorkerOutput("stagate", task="domains")
    # House polarity: n_spots / n_genes are what was supplied, *_used what STAGATE analysed.
    out.set_data(n_spots=n_spots_supplied, n_spots_used=n_spots, n_genes=n_genes_supplied, n_genes_used=n_genes_used)
    out.add_output_files(
        {
            "h5ad": h5ad_out,
            "clusters_csv": csv_out,
            "embedding_npy": emb_out,
        }
    )
    out.add_params(
        {
            # The radius that built the graph (derived from the data when rad_cutoff=0 was passed);
            # None when k_cutoff selected the KNN graph. What was passed is rad_cutoff_requested.
            "rad_cutoff": graph["rad_cutoff"],
            "rad_cutoff_requested": rad_cutoff,
            "k_cutoff": k_cutoff,
            "graph_model": graph["model"],
            "n_epochs": n_epochs,
            "n_clusters": n_clusters,
            # The device that trained, not the spelling that was asked for.
            "device": torch_device_str,
            "device_requested": device,
            "seed": seed,
            # The cap that was REQUESTED. What was actually selected is data.n_genes_used, which is
            # smaller whenever the slide has fewer detected genes than the cap.
            "n_hvg": n_hvg,
            "hvg_flavor": hvg_flavor,
            "spatial_key": spatial_key,
        }
    )
    out.add_params(identifier_rename_params(renamed))
    record_method(out, METHOD_NAME, used_fallback=False)
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    if k_cutoff and k_cutoff > 0 and rad_cutoff and rad_cutoff > 0:
        record_ignored(out, "rad_cutoff", f"k_cutoff={k_cutoff} builds a k-nearest-neighbour graph; no radius is used")
    out.set_summary(
        # The number of domains the labels hold; KMeans can return fewer than asked on degenerate input.
        n_clusters=int(len(cluster_sizes)),
        n_clusters_requested=int(n_clusters),
        cluster_key=cluster_key,
        embedding_key="STAGATE",
        cluster_sizes=cluster_sizes,
        spatial_graph={
            "model": graph["model"],
            "rad_cutoff": graph["rad_cutoff"],
            "rad_cutoff_source": graph["rad_cutoff_source"],
            "k_cutoff": graph["k_cutoff"],
            "n_edges": graph["n_edges"],
            "mean_neighbours": round(graph["mean_neighbours"], 3),
            "n_isolated_spots": graph["n_isolated_spots"],
            "median_nn_distance": graph["median_nn_distance"],
        },
    )
    # A successful run never carries stderr into the payload (base_mcp attaches stderr_tail only on a
    # non-zero exit), so a cut this large has to travel in the payload itself.
    if gene_note:
        out.add_warning(gene_note.strip())
    if graph["n_isolated_spots"]:
        out.add_warning(
            f"{graph['n_isolated_spots']} of {n_spots} spots have no neighbour in the spatial graph and were "
            "embedded from their own expression only; their domain labels draw on no spatial context."
        )
    for note in (device_note(device, torch_device_str), counts_note, umap_note):
        if note:
            out.add_warning(note)
    rename_note = identifier_rename_note(renamed)
    if rename_note:
        out.add_warning(rename_note.strip())
    tissue_sentence = ""
    if n_spots_off_tissue:
        tissue_sentence = (
            f" {n_spots_off_tissue} of the {n_spots_supplied} spots were left out because obs['in_tissue'] marks "
            f"them as background outside the tissue; the domains cover the {n_spots} in-tissue spots."
        )
    out.set_analysis(
        build_cluster_analysis(cluster_sizes, cluster_key="domain", total_spots=n_spots, n_requested=int(n_clusters))
        + describe_graph(graph)
        + tissue_sentence
        + gene_note
        + rename_note
    )
    return out.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STAGATE (pyG) worker for SpatialOmicsLab MCP - spatial domain identification"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="domains",
        choices=["domains"],
        help="Task to run (currently only 'domains' is supported).",
    )
    parser.add_argument(
        "--st-h5ad",
        dest="st_h5ad",
        type=str,
        required=True,
        help="Path to spatial transcriptomics AnnData (.h5ad).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        required=True,
        help="Directory to store STAGATE outputs.",
    )
    parser.add_argument(
        "--spatial-key",
        dest="spatial_key",
        type=str,
        default="spatial",
        help="Key in adata.obsm containing spatial coordinates (will be copied to 'spatial' if different).",
    )
    parser.add_argument(
        "--rad-cutoff",
        dest="rad_cutoff",
        type=float,
        default=0.0,
        help=(
            "Radius of the spatial neighbour graph, in obsm coordinate units. 0 (default) derives it from the "
            f"data: {AUTO_RADIUS_SLACK} x the median distance to the {AUTO_RADIUS_NEIGHBOURS}th-nearest spot. "
            "Not used when --k-cutoff > 0."
        ),
    )
    parser.add_argument(
        "--k-cutoff",
        dest="k_cutoff",
        type=int,
        default=0,
        help="K-nearest-neighbour graph with this k when > 0 (upstream model='KNN'); 0 => radius graph.",
    )
    parser.add_argument(
        "--n-epochs",
        dest="n_epochs",
        type=int,
        default=1000,
        help="Number of training epochs for STAGATE.",
    )
    parser.add_argument(
        "--n-clusters",
        dest="n_clusters",
        type=int,
        default=6,
        help="Number of spatial domains (clusters) for KMeans.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="auto",
        help="Computation device: 'auto', 'cpu', or 'cuda[:id]'.",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n-hvg",
        dest="n_hvg",
        type=int,
        default=3000,
        help="Number of highly variable genes to keep for STAGATE training.",
    )
    parser.add_argument(
        "--hvg-flavor",
        dest="hvg_flavor",
        type=str,
        default="seurat_v3",
        help=(
            f"scanpy HVG flavour, one of {', '.join(HVG_FLAVORS)}. seurat_v3 ranks raw counts and needs "
            "scikit-misc; a missing package stops the run rather than switching flavour."
        ),
    )
    parser.add_argument(
        "--layer-key",
        dest="layer_key",
        type=str,
        default="",
        help="Optional adata.layers key to use as the expression matrix (copied into adata.X before preprocessing).",
    )

    args = parser.parse_args()

    try:
        log(f"Task = {args.task}")
        if args.task != "domains":
            raise ValueError(f"Unsupported task '{args.task}'; only 'domains' is currently implemented.")

        result = run_stagate_domains(
            st_h5ad=args.st_h5ad,
            output_dir=args.output_dir,
            spatial_key=args.spatial_key,
            rad_cutoff=args.rad_cutoff,
            k_cutoff=args.k_cutoff,
            n_epochs=args.n_epochs,
            n_clusters=args.n_clusters,
            device=args.device,
            seed=args.seed,
            n_hvg=args.n_hvg,
            layer_key=args.layer_key,
            hvg_flavor=args.hvg_flavor,
        )

        # Final JSON result to stdout (SpatialOmicsLab wrapper 解析这个)
        print(json.dumps(result), flush=True)
    except Exception as e:
        # Log traceback to stderr, emit JSON error to stdout
        log(f"ERROR in stagate_worker: {str(e)}")
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("stagate", str(e), task="domains")
        sys.exit(1)


if __name__ == "__main__":
    main()
