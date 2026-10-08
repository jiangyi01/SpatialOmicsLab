#!/usr/bin/env python3
"""STAGATE spatial domain MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "stagate"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "STAGATE",
    "/opt/conda/envs/stagate_pyg/bin/python",
    "/workspace/epic-fermat/agent/tools/stagate_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def stagate_spatial_domains(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    layer_key: str = "",
    rad_cutoff: float = 0.0,
    k_cutoff: int = 0,
    n_epochs: int = 2000,
    n_clusters: int = 6,
    device: str = "auto",
    seed: int = 0,
    n_hvg: int = 3000,
    hvg_flavor: str = "seurat_v3",
) -> dict[str, Any]:
    """
    Run STAGATE to learn a spatially informed embedding and cluster spots
    into spatial domains.

    What runs: STAGATE_pyG's graph attention auto-encoder on a spatial neighbour graph, then
    KMeans (n_clusters) on its embedding. mclust, which the STAGATE tutorials use for the last
    step, is not run.

    Spots with obs['in_tissue'] == 0 (background outside the tissue) are left out before the
    graph is built; the payload reports them under params.in_tissue_filter with a warning, and
    data.n_spots / data.n_spots_used give the supplied and analysed spot counts.

    STAGATE trains on a subset of the genes you supply. The preprocessing follows STAGATE's
    tutorial: drop every gene detected in fewer than 3 spots, flag the n_hvg most highly
    variable (3,000 by default), normalise each spot to 10,000 over all the remaining genes and
    log-transform, and only then keep the flagged genes. The payload reports the supplied panel
    as n_genes and the analysed panel as n_genes_used, and carries a warning naming the cut when
    the two differ. X (or layer_key) should hold raw counts: the worker normalises and
    log-transforms it, and warns when it does not look like counts.

    The spatial graph is measured before training and reported in summary.spatial_graph
    (edges, mean neighbours per spot, isolated spots). A graph that leaves more than half the
    spots with no neighbour is refused with the spot spacing and the knobs: upstream adds a
    self-loop to every spot, so an empty graph would train as a non-spatial auto-encoder.

    Parameters
    ----------
    st_h5ad:
        Path to the spatial AnnData (.h5ad) input.
    output_dir:
        Directory where STAGATE outputs will be saved: stagate_domains.h5ad,
        stagate_domain_assignments.csv and stagate_embedding.npy.
    spatial_key:
        Key in adata.obsm with spatial coordinates.
    layer_key:
        Optional key in adata.layers to use as expression.
    rad_cutoff:
        Radius of the spatial neighbour graph, in the units of obsm[spatial_key]. 0 (default)
        derives it from the data: 1.2 x the median distance from a spot to its 6th-nearest
        spot, which on a Visium grid is the six-spot first ring (the graph STAGATE's Visium
        tutorial value of 150 builds on its own coordinates). A fixed value depends on the
        coordinate units: 150 links no two spots on slides whose spot spacing exceeds 150.
        The radius used is reported as params.rad_cutoff. Not used when k_cutoff > 0.
    k_cutoff:
        When > 0, build a k-nearest-neighbour graph with this k instead of a radius graph
        (upstream model='KNN'); 0 => radius graph.
    n_epochs:
        Number of training epochs for STAGATE.
    n_clusters:
        Number of spatial domains KMeans is asked for; summary.n_clusters is the number the
        labels hold and summary.n_clusters_requested this value.
    device:
        "auto", "cpu", or a CUDA device string. params.device reports the device that
        trained (a CUDA request on a machine without CUDA trains on the CPU and says so).
    seed:
        Random seed.
    n_hvg:
        How many highly variable genes STAGATE trains on, chosen after the
        detected-in-fewer-than-3-spots filter. 3,000 is the upstream tutorial value. Raise it
        toward the panel size to train on more of the slide, or lower it on a targeted panel
        where 3,000 exceeds the genes available.
    hvg_flavor:
        scanpy flavour for choosing the n_hvg genes: 'seurat_v3' (default; ranks raw counts,
        needs scikit-misc in the STAGATE env), 'seurat' or 'cell_ranger' (log-normalised
        dispersion). A missing package stops the run; no other flavour is substituted.
    """
    args = [
        "--task",
        "domains",
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--spatial-key",
        spatial_key,
        "--rad-cutoff",
        str(rad_cutoff),
        "--k-cutoff",
        str(k_cutoff),
        "--n-epochs",
        str(n_epochs),
        "--n-clusters",
        str(n_clusters),
        "--device",
        device,
        "--seed",
        str(seed),
        "--n-hvg",
        str(n_hvg),
        "--hvg-flavor",
        hvg_flavor,
    ]
    if layer_key:
        args.extend(["--layer-key", layer_key])

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
