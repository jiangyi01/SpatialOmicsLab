#!/usr/bin/env python3
"""GraphST spatial transcriptomics MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "graphst"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "GRAPHST",
    "/opt/conda/envs/GraphST/bin/python",
    "/workspace/epic-fermat/agent/tools/graphst_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def graphst_spatial_clustering(
    st_h5ad: str,
    output_dir: str,
    n_clusters: int,
    cluster_tool: str = "mclust",
    radius: int = 50,
    device: str = "auto",
    label_key: str | None = None,
    r_home: str | None = None,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Run GraphST spatial clustering / domain identification on a single ST dataset.

    ``radius`` is the number of nearest spots that vote in the spatial refinement step, which runs
    only for ``cluster_tool='mclust'``; leiden/louvain accept it and report it under
    ``params.ignored``. Spots with ``obs['in_tissue'] == 0`` (background outside the tissue) are left
    out before the graph is built and get no domain; ``data.n_spots`` is the count supplied,
    ``data.n_spots_used`` the count clustered, and ``params.in_tissue_filter`` with a warning says how
    many were left out.

    GraphST's own preprocessing (seurat_v3 HVGs, normalize_total, log1p, scale) treats X as raw
    counts, so X is checked first: a negative or non-finite X (scaled or z-scored data) is refused,
    and the error says whether ``adata.raw`` holds counts; a non-negative non-integer X (already
    normalised or log-transformed) runs as before but is normalised a second time, with a warning.
    An input whose var already has ``highly_variable`` is trained on X as stored (GraphST skips its
    preprocessing), so that path accepts normalised data by design and is not refused.
    ``use_raw_counts=True`` runs on ``adata.raw.X`` instead (refused when the file has no adata.raw,
    or when adata.raw does not hold counts), and GraphST then preprocesses those counts itself.
    ``params.expression_source`` and ``params.x_matrix_kind`` say which matrix ran. Duplicate gene
    symbols are made unique and counted under ``params.n_genes_renamed``.
    """
    args = [
        "--task",
        "clustering",
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--n-clusters",
        str(n_clusters),
        "--cluster-tool",
        cluster_tool,
        "--radius",
        str(radius),
        "--device",
        device,
    ]
    if label_key is not None:
        args += ["--label-key", label_key]
    if r_home is not None:
        args += ["--r-home", r_home]
    if use_raw_counts:
        args.append("--use-raw-counts")

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def graphst_deconvolution(
    st_h5ad: str,
    scrna_h5ad: str,
    output_dir: str,
    celltype_key: str = "cell_type",
    epochs: int = 1200,
    retain_percent: float = 0.15,
    device: str = "auto",
    drop_unlabeled: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Run GraphST scRNA-ST deconvolution (cell-type mapping).

    ``retain_percent`` (0, 1] is the fraction of reference cells kept per spot when the learned
    mapping matrix is projected onto cell types. A reference cell with no label in
    ``obs[celltype_key]`` is an error unless ``drop_unlabeled=True`` leaves it out (the count is
    reported). Spatial spots with ``obs['in_tissue'] == 0`` (background outside the tissue) are left
    out before training and get no abundance row; ``data.n_spots`` is the count supplied,
    ``data.n_spots_used`` the count deconvolved, and ``params.in_tissue_filter`` with a warning says how
    many were left out.

    ``GraphST.preprocess`` normalises both inputs as raw counts, so each X is checked first: a
    negative or non-finite X is refused (the error names the input and says whether its adata.raw
    holds counts); a non-negative non-integer X runs as before with a warning.
    ``use_raw_counts=True`` reads ``adata.raw.X`` of each input that has an adata.raw (an input
    without one is read from X, with a warning; a run where neither has one is refused).
    ``params.expression_source`` / ``params.x_matrix_kind`` describe the spatial matrix and
    ``params.sc_expression_source`` / ``params.sc_x_matrix_kind`` the reference's.

    Memory: GraphST learns a dense n_cells x n_spots mapping matrix (float32; the parameter, its
    gradient, two Adam moments and the per-epoch softmax are alive together) beside dense
    spot-by-spot matrices. Once the reference is read, a run whose estimate exceeds the memory
    available (a cgroup limit included) is refused with the numbers before any training; neither
    the slide nor the reference is ever cut down.
    """
    args = [
        "--task",
        "deconvolution",
        "--st-h5ad",
        st_h5ad,
        "--scrna-h5ad",
        scrna_h5ad,
        "--output-dir",
        output_dir,
        "--celltype-key",
        celltype_key,
        "--epochs",
        str(epochs),
        "--retain-percent",
        str(retain_percent),
        "--device",
        device,
    ]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    if use_raw_counts:
        args.append("--use-raw-counts")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
