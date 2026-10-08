#!/usr/bin/env python
"""
Worker script: Scanpy Leiden clustering of a spatial slide, published as "spatial domains".

What runs: normalize_total -> log1p -> highly-variable-gene selection -> scale -> PCA -> kNN
graph on the PCA -> Leiden -> UMAP. The kNN graph is built from expression alone; the spot
coordinates in obsm['spatial'] are read only to draw the spatial figure. The payload says so
(params.method), because "spatial domain" otherwise reads as a spatially-aware method.

sc.pp.scale densifies the HVG-subset matrix (n_spots x n_top_genes float32). That is intrinsic
to centring and scaling; the HVG selection is what keeps it small. The working set is estimated
against worker_utils.available_memory_bytes() -- MemAvailable and the room under a container's
cgroup limit, whichever is smaller -- before it is allocated.

What it reads: spots with obs['in_tissue'] == 0 (background outside the tissue) are left out first
(worker_utils.keep_in_tissue, reported in params.in_tissue_filter). normalize_total + log1p treat
the matrix as counts, so the matrix is checked first (worker_utils.choose_counts_matrix): negative or
non-finite values (scaled / z-scored data) are refused -- they turned into NaN in log1p and crashed
in the HVG step with "'[nan] not in index'" -- naming use_raw_counts when adata.raw holds counts;
non-integer non-negative values run with a warning; use_raw_counts=True clusters adata.raw.X.

Environment: /opt/conda/envs/SpaGCN (has scanpy, anndata, matplotlib)
Usage (example):
  conda run -p /opt/conda/envs/SpaGCN python scanpy_spatial_worker.py \
    --data-path /path/to/spatial.h5ad \
    --output-dir /path/to/results \
    --resolution 0.6

--data-path is required. --output-dir defaults to worker_utils.default_output_dir(), which
resolves SOG_WORK_DIR -> a writable /workspace/work -> ./work, so a run on a box without the
author's directories still lands somewhere writable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import scanpy as sc

# The honest name of what this worker runs. It is the only implementation here, so it is the
# method and never a fallback (record_method(..., used_fallback=False)).
METHOD_NAME = (
    "Leiden on an expression PCA kNN graph (spatial coordinates are not used in clustering, only for plotting)"
)

# scanpy.pp.calculate_qc_metrics' default percent_top tiers.
_QC_PERCENT_TOP = (50, 100, 200, 500)

# Dense n_spots x n_genes copies alive at the peak of scale + PCA (the scaled matrix, the clip mask
# rounded up, and the centred copy PCA makes).
_DENSE_SCALE_COPIES = 3


def _binary_search_leiden(
    adata, target_k: int, low: float = 0.05, high: float = 5.0, max_iter: int = 25, report: dict | None = None
):
    """Binary search Leiden resolution to land on target_k clusters.

    Mirrors the manual runner's _binary_search_leiden (hands_by_myself/runners/
    run_scanpy_spatial_domain.py) so MCP-driven and manual runs converge on the
    same clustering objective.

    Returns the obs key of the closest trial (``None`` only when ``max_iter < 1``). The search
    hands back its closest miss whether or not ``target_k`` was reached -- for many slides no
    resolution gives exactly k -- so the caller must not read the return value as a hit. That
    return shape is pinned (test/test_cluster_count_mismatch_is_reported.py); everything else
    the caller needs is written into ``report`` when a dict is passed::

        resolution_used   the resolution behind the returned key
        n_clusters        the cluster count that resolution gave
        matched           True only when n_clusters == target_k
        target_n_clusters the request
        trials            [{"resolution": r, "n_clusters": k}, ...] in the order tried
        trial_keys        every obs column the search wrote, so the caller can remove them

    Every trial leaves a ``leiden_<res>`` column in ``adata.obs``; they are the search's scratch
    space, not results, and the caller drops them once the chosen one has been copied out.
    """
    best_key = None
    best_diff = 10**9
    best_resolution = None
    best_k = None
    trials = []
    trial_keys = []
    for _ in range(max_iter):
        mid = (low + high) / 2
        key = f"leiden_{mid:.4f}"
        if key in trial_keys:
            # The interval has shrunk below the key's precision; another run would overwrite the
            # column the best key points at with a different partition.
            break
        sc.tl.leiden(adata, resolution=mid, key_added=key, random_state=0)
        k = int(adata.obs[key].nunique())
        trial_keys.append(key)
        trials.append({"resolution": float(mid), "n_clusters": k})
        diff = abs(k - target_k)
        if diff < best_diff:
            best_diff = diff
            best_key = key
            best_resolution = float(mid)
            best_k = k
        if k == target_k:
            break
        if k < target_k:
            low = mid
        else:
            high = mid
        if high - low < 1e-3:
            break
    if report is not None:
        report.update(
            {
                "target_n_clusters": int(target_k),
                "resolution_used": best_resolution,
                "n_clusters": best_k,
                "matched": bool(best_k == target_k) if best_k is not None else False,
                "trials": trials,
                "trial_keys": list(trial_keys),
            }
        )
    return best_key


def _keep_only_the_chosen_leiden_record(adata, search: dict, chosen_key: str) -> None:
    """Leave the written h5ad's ``uns`` describing the partition it publishes, not the last trial.

    sc.tl.leiden records its parameters in ``uns`` on every call: scanpy >= 1.10 under
    ``uns[key_added]`` (one ``leiden_<res>`` entry per trial), scanpy < 1.10 -- the SpaGCN env ships
    1.9.8 -- under ``uns["leiden"]``, overwritten each time, so after a search that missed its target
    that record names the last resolution tried rather than the one ``obs['spatial_domain']`` holds.
    """
    trials = search.get("trials") or []
    chosen_record = adata.uns.get(chosen_key)
    for key in search.get("trial_keys") or []:
        adata.uns.pop(key, None)
    if isinstance(chosen_record, dict):
        # scanpy >= 1.10: the chosen trial's own record, under the column name it now lives in, as
        # the target_n_clusters=0 path (key_added="spatial_domain") writes it.
        adata.uns["spatial_domain"] = chosen_record
    record = adata.uns.get("leiden")
    params = record.get("params") if isinstance(record, dict) else None
    if isinstance(params, dict) and trials and params.get("resolution") == trials[-1]["resolution"]:
        # scanpy < 1.10: the record the trials overwrote; only corrected when the trials wrote it.
        params["resolution"] = search["resolution_used"]


def _available_memory_bytes():
    """Memory this process can still allocate, in bytes, or None where nothing can be read.

    The fleet's one reader, ``worker_utils.available_memory_bytes``: the smaller of MemAvailable and
    the room left under the cgroup memory limit, page cache counted as reclaimable. The reader this
    replaces took MemAvailable alone, which inside a memory-limited container is the HOST's free
    memory -- so a working set the container could not hold passed the check and the kernel killed
    the run with no payload. Kept as a module-level seam so a test can stand in for the box.
    """
    from worker_utils import available_memory_bytes

    return available_memory_bytes()


def _check_dense_scale_fits(n_obs: int, n_vars: int, itemsize: int, n_top_genes: int) -> int:
    """Estimate the dense scale/PCA working set and stop, with the numbers, if it cannot fit.

    sc.pp.scale centres every gene, so the HVG-subset matrix becomes dense, and the PCA that
    follows centres a copy of it: about three dense n_spots x n_genes copies are alive at the peak.
    That densification is intrinsic to the method, so it stays -- but a run that cannot fit fails
    here, before the allocation, rather than being killed by the kernel half-way through.
    Returns the estimate in bytes.
    """
    need = int(_DENSE_SCALE_COPIES * int(n_obs) * int(n_vars) * int(itemsize))
    available = _available_memory_bytes()
    if available is not None and need > available:
        raise MemoryError(
            f"scaling and PCA densify the {n_obs} spots x {n_vars} highly variable genes matrix, which needs "
            f"about {need / 2**30:.2f} GiB ({_DENSE_SCALE_COPIES} dense copies of {itemsize}-byte values); "
            f"{available / 2**30:.2f} GiB is available. Lower n_top_genes (currently {n_top_genes}) to shrink "
            "the gene axis, or run on a machine with more memory."
        )
    return need


def _atomic_savefig(path: Path, dpi: int = 150) -> None:
    """Save the current figure beside ``path`` as ``<name>.partial`` and rename it into place."""
    tmp = path.with_suffix(path.suffix + ".partial")
    plt.savefig(str(tmp), dpi=dpi, bbox_inches="tight", format="png")
    plt.close()
    os.replace(str(tmp), str(path))


def run_scanpy_spatial_domain(
    data_path: str,
    output_dir: str,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    min_counts: int = 0,
    target_n_clusters: int = 0,
    n_top_genes: int = 2000,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Cluster the slide and return the payload.

    A run that fails keeps what it had already found out about its input: the exception leaves with the
    payload's warnings so far as ``worker_warnings``, and ``main`` publishes them in the error payload. A
    sentence such as "X holds non-integer values" explains a crash further down better than the crash's
    own text does, and before this it was kept in a local list that only a successful run ever emitted.
    """
    from worker_utils import WorkerOutput

    out = WorkerOutput("scanpy_spatial", task="spatial_domain_clustering")
    try:
        return _cluster_slide(
            out,
            data_path=data_path,
            output_dir=output_dir,
            resolution=resolution,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            min_counts=min_counts,
            target_n_clusters=target_n_clusters,
            n_top_genes=n_top_genes,
            use_raw_counts=use_raw_counts,
        )
    except Exception as exc:
        try:
            exc.worker_warnings = [str(w) for w in (out._extra.get("warnings") or [])]
        except AttributeError:  # an exception type that takes no attributes
            pass
        raise


def _cluster_slide(
    out,
    data_path: str,
    output_dir: str,
    resolution: float,
    n_neighbors: int,
    n_pcs: int,
    min_counts: int,
    target_n_clusters: int,
    n_top_genes: int,
    use_raw_counts: bool,
) -> dict[str, Any]:
    from worker_utils import (
        build_cluster_analysis,
        choose_counts_matrix,
        describe_reduction,
        identifier_rename_note,
        identifier_rename_params,
        keep_in_tissue,
        make_names_unique_and_report,
        record_expression_source,
        record_ignored,
        record_in_tissue,
        record_method,
    )

    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata_out_path = output_dir / "scanpy_spatial_domains.h5ad"
    domains_csv_path = output_dir / "spatial_domains_summary.csv"
    umap_png_path = output_dir / "umap_spatial_domains.png"
    spatial_png_path = output_dir / "spatial_spatial_domains.png"

    sc.settings.figdir = str(output_dir)
    sc.settings.autoshow = False

    if not data_path.exists():
        raise FileNotFoundError(f"Input h5ad not found: {data_path}")

    adata = sc.read_h5ad(data_path)

    # Every count in the payload is read after the cuts below, so record what the caller
    # actually handed us before the first one bites -- otherwise the survivors get published
    # as the input (a 33,538-gene slide reported as a 2,000-gene one).
    n_spots_supplied = int(adata.n_obs)
    spot_cuts = []

    # Background spots first (the fleet rule): obs['in_tissue'] == 0 is glass outside the tissue, and
    # CELLxGENE Visium exports carry it beside the tissue. This worker has always left it out; the
    # shared helper also reads True/"1" as in tissue and the count is published in params.
    try:
        adata, _, n_off_tissue = keep_in_tissue(adata, "spots")
    except ValueError as exc:
        raise ValueError(f"the in_tissue filter kept 0 of {n_spots_supplied} spots: {exc}") from exc
    if n_off_tissue:
        spot_cuts.append("the in-tissue filter")
    record_in_tissue(out, n_spots_supplied, n_off_tissue)

    # normalize_total + log1p treat the matrix as counts. A scaled / z-scored X (negative values) turned
    # into NaN in log1p and died in the HVG step with "'[nan] not in index'"; it is refused here instead,
    # naming use_raw_counts when adata.raw holds the counts. A non-integer non-negative X still runs, as
    # before, with a warning.
    adata, counts_info = choose_counts_matrix(adata, use_raw_counts=bool(use_raw_counts))
    if counts_info.get("warning"):
        counts_info = dict(counts_info, warning="X does not look like raw counts. " + counts_info["warning"])
    record_expression_source(out, counts_info)

    # Only the gene axis is deduplicated, as this worker always has -- on the matrix that is clustered
    # (adata.raw's own gene list with use_raw_counts) -- and the count is published rather than the
    # rename happening in silence.
    renamed = make_names_unique_and_report(adata, axes=("var",))
    n_genes_supplied = int(adata.n_vars)

    if min_counts > 0:
        if "total_counts" not in adata.obs.columns:
            # scanpy's default percent_top=(50, 100, 200, 500) raises "Positions outside range of
            # features" on any panel under 500 genes (MERFISH, Xenium, targeted Visium). Keep the
            # tiers the panel can fill, so a whole-transcriptome slide gets the same columns as before.
            tiers = [n for n in _QC_PERCENT_TOP if n <= int(adata.n_vars)]
            sc.pp.calculate_qc_metrics(adata, percent_top=tiers or None, inplace=True)
        n_before = int(adata.n_obs)
        adata = adata[adata.obs["total_counts"] >= min_counts].copy()
        if int(adata.n_obs) == 0:
            raise ValueError(
                f"min_counts={min_counts} dropped every one of the {n_before} spots; lower min_counts "
                "(0 applies no filter)."
            )
        if int(adata.n_obs) < n_before:
            spot_cuts.append(f"the min_counts={min_counts} filter")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes, subset=True)

    smallest_axis = min(int(adata.n_obs), int(adata.n_vars))
    if int(n_pcs) >= smallest_axis:
        raise ValueError(
            f"n_pcs={n_pcs} is not below the smaller matrix axis after filtering "
            f"({adata.n_obs} spots x {adata.n_vars} genes); PCA needs fewer components than that. "
            f"Lower n_pcs, or raise n_top_genes (currently {n_top_genes}) if the gene axis is the limit."
        )

    # Intrinsic dense step: scaling centres every gene, so the HVG-subset matrix is densified here.
    # Estimate it first so a run that cannot fit stops with the numbers instead of being OOM-killed.
    x_itemsize = int(getattr(getattr(adata.X, "dtype", None), "itemsize", 0) or 8)
    _check_dense_scale_fits(adata.n_obs, adata.n_vars, x_itemsize, n_top_genes)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pcs)

    # use_rep="X_pca" is explicit because scanpy's default is not: with <= 50 genes left after the
    # HVG cut (settings.N_PCS) sc.pp.neighbors silently builds the graph on the scaled X and ignores
    # both the PCA and n_pcs. The graph is always the n_pcs-component expression PCA, as documented.
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep="X_pca")
    requested = int(target_n_clusters) if target_n_clusters and target_n_clusters > 0 else None
    search = None
    if requested is not None:
        search = {}
        chosen_key = _binary_search_leiden(adata, target_k=requested, report=search)
        if chosen_key is None:
            # Unreachable with max_iter=25 (the first trial always becomes best_key); kept as a
            # loud stop rather than a silent run at `resolution` the caller did not ask for.
            raise RuntimeError(
                f"the Leiden resolution search for target_n_clusters={requested} ran no trial; "
                "set target_n_clusters=0 to cluster at `resolution` instead."
            )
        adata.obs["spatial_domain"] = adata.obs[chosen_key].values
        resolution_used = float(search["resolution_used"])
        # The trial columns are the search's scratch space. Left in obs they reach the written
        # h5ad, where a name-based label picker can grab a leiden_* column instead of the result.
        trial_columns = [c for c in search["trial_keys"] if c in adata.obs.columns]
        if trial_columns:
            adata.obs.drop(columns=trial_columns, inplace=True)
        _keep_only_the_chosen_leiden_record(adata, search, chosen_key)
    else:
        sc.tl.leiden(adata, resolution=resolution, key_added="spatial_domain", random_state=0)
        resolution_used = float(resolution)
    sc.tl.umap(adata)

    counts_per_domain = (
        adata.obs["spatial_domain"]
        .value_counts()
        .sort_index()
        .rename_axis("spatial_domain")
        .reset_index(name="n_spots")
    )
    csv_tmp = domains_csv_path.with_suffix(domains_csv_path.suffix + ".partial")
    counts_per_domain.to_csv(csv_tmp, index=False)
    os.replace(str(csv_tmp), str(domains_csv_path))

    h5ad_tmp = adata_out_path.with_suffix(adata_out_path.suffix + ".partial")
    adata.write(h5ad_tmp)
    os.replace(str(h5ad_tmp), str(adata_out_path))

    sc.pl.umap(
        adata,
        color="spatial_domain",
        legend_loc="on data",
        title="UMAP – Scanpy spatial domains",
        show=False,
    )
    _atomic_savefig(umap_png_path)

    try:
        sc.pl.spatial(
            adata,
            color="spatial_domain",
            title="Spatial – Scanpy spatial domains",
            show=False,
        )
        _atomic_savefig(spatial_png_path)
    except (ValueError, KeyError):
        # Non-Visium datasets (MERFISH, Slide-seq) may lack .uns['spatial']
        # metadata required by sc.pl.spatial. Draw a plain scatter of the stored
        # coordinates instead; this is the figure only, the clustering is unchanged.
        import numpy as np

        plt.close()
        coords = None
        if "spatial" in adata.obsm:
            coords = adata.obsm["spatial"]
        elif "x" in adata.obs.columns and "y" in adata.obs.columns:
            coords = np.column_stack([adata.obs["x"].values, adata.obs["y"].values])

        if coords is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            categories = adata.obs["spatial_domain"].astype(str)
            for cat in sorted(categories.unique(), key=lambda x: int(x) if x.isdigit() else x):
                mask = categories == cat
                ax.scatter(coords[mask, 0], coords[mask, 1], s=4, label=cat, alpha=0.7)
            ax.set_title("Spatial – Scanpy spatial domains")
            ax.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
            _atomic_savefig(spatial_png_path)
        else:
            spatial_png_path = None  # no spatial plot possible

    n_spots_used = int(adata.n_obs)
    n_genes_used = int(adata.n_vars)
    n_domains = int(counts_per_domain["spatial_domain"].nunique())
    cluster_sizes = {str(row["spatial_domain"]): int(row["n_spots"]) for _, row in counts_per_domain.iterrows()}

    spot_note = describe_reduction("spots", n_spots_supplied, n_spots_used, " and ".join(spot_cuts))
    gene_note = describe_reduction(
        "genes",
        n_genes_supplied,
        n_genes_used,
        f"the highly-variable-gene selection (n_top_genes={n_top_genes})",
    )

    out.set_data(
        n_spots=n_spots_supplied,
        n_genes=n_genes_supplied,
        n_spots_used=n_spots_used,
        n_genes_used=n_genes_used,
    )
    reduction_notes = [note.strip() for note in (spot_note, gene_note) if note]
    if reduction_notes:
        out.add_warnings(reduction_notes)
    out.add_output_files(
        {
            "annotated_h5ad": str(adata_out_path),
            "domains_csv": str(domains_csv_path),
            "umap_png": str(umap_png_path),
            "spatial_png": str(spatial_png_path) if spatial_png_path else None,
        }
    )
    # `resolution` is what the caller passed; `resolution_used` is what Leiden actually ran at.
    # They differ whenever target_n_clusters > 0, because then the binary search picks it.
    out.add_params(
        {
            "resolution": resolution,
            "resolution_used": resolution_used,
            "n_neighbors": n_neighbors,
            "n_pcs": n_pcs,
            "min_counts": min_counts,
            "target_n_clusters": target_n_clusters,
            "n_top_genes": n_top_genes,
            "use_raw_counts": bool(use_raw_counts),
        }
    )
    out.add_params(identifier_rename_params(renamed))
    record_method(out, METHOD_NAME)
    resolution_note = ""
    if search is not None:
        n_trials = len(search["trials"])
        out.add_params(
            {
                "resolution_search": {
                    "target_n_clusters": requested,
                    "resolution_used": resolution_used,
                    "n_clusters_achieved": int(search["n_clusters"]),
                    "matched": bool(search["matched"]),
                    "n_trials": n_trials,
                    "trials": search["trials"],
                }
            }
        )
        record_ignored(
            out,
            "resolution",
            f"target_n_clusters={requested} selects the Leiden resolution by binary search; the resolution "
            f"that ran is params.resolution_used={resolution_used:.4f}",
        )
        if search["matched"]:
            resolution_note = (
                f" Leiden resolution used: {resolution_used:.4f}, chosen by a binary search over {n_trials} "
                f"trial(s) for target_n_clusters={requested}, which it reached."
            )
        else:
            achieved = int(search["n_clusters"])
            out.add_warning(
                f"target_n_clusters={requested} was not reached: none of the {n_trials} Leiden resolutions "
                f"tried gave exactly {requested} clusters. The closest, resolution {resolution_used:.4f}, "
                f"gave {achieved} and is what obs['spatial_domain'] holds (params.resolution_search lists "
                "every trial)."
            )
            resolution_note = (
                f" Leiden resolution used: {resolution_used:.4f}, the closest of {n_trials} trial(s) to "
                f"target_n_clusters={requested}; it gave {achieved} clusters, not {requested}."
            )
    else:
        resolution_note = f" Leiden resolution used: {resolution_used:.4f} (the `resolution` parameter)."
    method_note = (
        " Method: Leiden clustering of a kNN graph built on the expression PCA "
        f"(n_pcs={n_pcs}, n_neighbors={n_neighbors}); spatial coordinates were not used in clustering, "
        "only to draw the spatial figure."
    )
    # _binary_search_leiden keeps the closest resolution it saw and returns it whether or not
    # that resolution actually yields target_n_clusters -- and for many datasets no resolution
    # does (a live run asking for 4 domains saw 5, 3 or 2, never 4). Report the request next to
    # the result so a miss is visible; 0 means the caller asked for no particular count.
    out.set_summary(
        n_domains=n_domains,
        n_domains_requested=requested,
        cluster_key="spatial_domain",
        cluster_sizes=cluster_sizes,
    )
    # The domain tally really does span the analysed spots, so that is the total the sentence
    # is built from; the notes carry what the caller supplied so the two are never confused.
    out.set_analysis(
        build_cluster_analysis(
            cluster_sizes,
            cluster_key="domain",
            total_spots=n_spots_used,
            n_requested=requested,
        )
        + resolution_note
        + method_note
        + spot_note
        + gene_note
        + identifier_rename_note(renamed)
        + (
            " Expression: adata.raw.X (use_raw_counts=True), normalised and log-transformed here as counts."
            if counts_info["expression_source"] == "raw.X"
            else ""
        )
    )

    return out.to_dict()


def main():
    from worker_utils import WorkerOutput, default_output_dir

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the spatial .h5ad to cluster.",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir("scanpy_spatial_domain_results"),
        help="Where to write the clustered .h5ad, the domain table and the figures.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Leiden resolution. Ignored when --target-n-clusters is > 0 (the search picks it).",
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--min-counts", type=int, default=0)
    parser.add_argument(
        "--target-n-clusters",
        type=int,
        default=0,
        help="Binary-search the Leiden resolution for the count closest to this; 0 clusters at --resolution.",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=2000,
        help="Number of highly variable genes to keep before clustering. The rest are discarded.",
    )
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        default=False,
        help=(
            "Cluster adata.raw.X instead of X (refused when there is no adata.raw or it is not counts). Use it "
            "when X is log-normalised or scaled and the counts sit in adata.raw, as in CELLxGENE exports."
        ),
    )

    args = parser.parse_args()

    # Redirect all stdout to stderr during analysis to prevent library
    # prints (scanpy, SpaGCN) from polluting the JSON output.
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        try:
            result = run_scanpy_spatial_domain(
                data_path=args.data_path,
                output_dir=args.output_dir,
                resolution=args.resolution,
                n_neighbors=args.n_neighbors,
                n_pcs=args.n_pcs,
                min_counts=args.min_counts,
                target_n_clusters=args.target_n_clusters,
                n_top_genes=args.n_top_genes,
                use_raw_counts=args.use_raw_counts,
            )
        except Exception as e:
            sys.stdout = _real_stdout
            error = WorkerOutput.error("scanpy_spatial", str(e), task="spatial_domain_clustering", exc=e)
            # What the run had already reported about its input before it failed (a counts warning, the
            # spots left out) -- the part of the diagnosis the exception text alone does not carry.
            collected = getattr(e, "worker_warnings", None)
            if collected:
                error["warnings"] = list(collected)
            sys.stdout.write("\n" + json.dumps(error, default=str) + "\n")
            sys.stdout.flush()
            return
    finally:
        sys.stdout = _real_stdout

    # Only the final JSON goes to real stdout for the MCP server to capture.
    # Must be single-line JSON so _parse_result() can find it as one line.
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
