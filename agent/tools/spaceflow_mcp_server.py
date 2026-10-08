#!/usr/bin/env python3
"""SpaceFlow spatial domains MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_json

TOOL_NAME = "spaceflow"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPACEFLOW",
    "/opt/conda/envs/spaceflow_env/bin/python",
    "/workspace/epic-fermat/agent/tools/spaceflow_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def spaceflow_spatial_domains(
    output_dir: str,
    input_mode: str = "h5ad",
    h5ad_path: str | None = None,
    visium_h5_path: str | None = None,
    visium_spatial_dir: str | None = None,
    spaceranger_dir: str | None = None,
    counts_h5ad_path: str | None = None,
    coords_csv: str | None = None,
    sample_id: str = "sample",
    coord_type: str = "array",
    n_top_genes: int = 3000,
    spatial_regularization_strength: float = 0.1,
    z_dim: int = 50,
    lr: float = 1e-3,
    epochs: int = 1000,
    max_patience: int = 50,
    min_stop: int = 100,
    random_seed: int = 42,
    gpu: int = 0,
    regularization_acceleration: bool = True,
    edge_subset_sz: int = 1000000,
    seg_n_neighbors: int = 50,
    seg_resolution: float = 1.0,
    target_n_clusters: int = 0,
    run_psm: bool = False,
    psm_n_neighbors: int = 20,
    psm_resolution: float = 1.0,
    allow_preprocessing_fallback: bool = False,
    hvg_flavor: str = "seurat",
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Run SpaceFlow: preprocess, train embedding, spatial domain segmentation, optional pSM.

    If target_n_clusters > 0, seg_resolution is binary-searched so SpaceFlow's
    Leiden segmentation lands on that exact cluster count. Pass the registry's
    metadata.n_clusters here for benchmark datasets — otherwise seg_resolution
    is used as-is and the realized k is data-dependent. When the search cannot hit
    the count, the payload warns and reports the count it produced.

    Only the top `n_top_genes` highly variable genes (default 3000) are used to train
    the embedding; every other gene is discarded before the method runs. The payload
    reports the panel you supplied as `n_genes` and the panel actually trained on as
    `n_genes_used`, and warns when the two differ. The matrix is handed to SpaceFlow as
    supplied (a sparse matrix stays sparse); SpaceFlow densifies only the selected genes
    to train, and a slide whose dense spots x genes matrix cannot fit in memory is refused
    up front with the numbers.

    Spots with obs['in_tissue'] == 0 (background outside the tissue) are left out and reported
    under params.in_tissue_filter with a warning. In generic_counts_coords mode the flag is the
    coords_csv's in_tissue column when it has one, the counts h5ad's own obs['in_tissue'] when only
    it has one, and both together (a spot is tissue only where both say so) when both do
    (params.in_tissue_source); spots of the counts h5ad with no coords_csv row are left out and
    counted in params.n_spots_without_coordinates, and data.n_spots is every spot of the counts h5ad.

    SpaceFlow normalises the matrix as counts (normalize_total, log1p): an X with negative or NaN
    values (scaled data) is refused, and the error says whether adata.raw holds counts; a fractional
    X (already log-normalised) runs with a warning; use_raw_counts=True trains on adata.raw.X instead
    (h5ad and generic_counts_coords modes; the two Space Ranger modes read the count matrix itself and
    list it under params.ignored). params.expression_source and params.x_matrix_kind report which.

    Genes with no counts in any remaining spot are
    left out before SpaceFlow sees the matrix (params.n_genes_without_counts_dropped): they cannot
    be highly variable, and their shared mean of 0 collapses cell_ranger's mean bins ("Bin edges
    must be unique"), which stopped SpaceFlow's own preprocessing on most whole-transcriptome slides.

    SpaceFlow's own preprocessing selects genes with scanpy flavor='cell_ranger' and builds
    an alpha-complex spatial graph. If cell_ranger still fails because many of the remaining
    genes share one mean expression, the run stops with
    that error, unless allow_preprocessing_fallback=True: only then does this tool's substitute
    run -- highly variable genes chosen by hvg_flavor ('seurat' default; 'cell_ranger'; or
    'seurat_v3', which ranks raw counts and needs scikit-misc, absent from the SpaceFlow env,
    so it stops the run rather than switching flavour) and a k-nearest-neighbour spatial graph
    in place of the alpha complex. params.method and params.used_fallback say which ran;
    hvg_flavor has no effect when SpaceFlow's own preprocessing succeeds.

    spatial_regularization_strength weights SpaceFlow's spatial consistency penalty, which
    pulls together the embeddings of spots that are close in space. regularization_acceleration/
    edge_subset_sz are SpaceFlow.train()'s own knobs for it: with acceleration on (the default)
    the penalty is estimated each epoch from `edge_subset_sz` random spot pairs instead of all of
    them. Turning acceleration off gives the exact penalty only on slides of 5000 spots or fewer —
    above that upstream samples anyway, whatever this flag says, and the payload warns. SpaceFlow
    1.0.4 pairs each sampled spot's coordinates with themselves on that sampled path, which leaves
    the penalty with no spatial information; this tool corrects that index in upstream's own
    train() before training, and also restores the lowest-loss weights (upstream restored the last
    epoch's). params.train_corrections lists the corrections applied.

    gpu is a CUDA device index; -1 means CPU. SpaceFlow trains on the CPU when no CUDA device
    exists, and params.device / params.gpu report where training actually ran.

    run_psm computes SpaceFlow's pseudo-Spatiotemporal Map: diffusion pseudotime on the embedding
    from the spot most distant from all others (chosen over every spot; params.psm_root names it).
    psm_resolution has no effect on it and is reported under params.ignored.

    Modes (input_mode, default "h5ad"):
    - h5ad: h5ad_path (must carry obsm["spatial"], used as stored)
    - visium_h5_spatial: visium_h5_path + visium_spatial_dir
    - spaceranger_outs: spaceranger_dir
    - generic_counts_coords: counts_h5ad_path + coords_csv (barcode, x, y[, in_tissue]; x, y used as
      stored)
    coord_type ('array' or 'pixel') picks the columns of a Space Ranger tissue_positions file, so it
    applies to the visium_h5_spatial and spaceranger_outs modes only; elsewhere it is reported under
    params.ignored. sample_id is recorded in the payload and in the annotated h5ad's
    uns['spaceflow_run'].
    """
    payload: dict[str, Any] = {
        "__tool__": "spaceflow_spatial_domains",
        "output_dir": output_dir,
        "input_mode": input_mode,
        "sample_id": sample_id,
        "coord_type": coord_type,
        "n_top_genes": n_top_genes,
        "spatial_regularization_strength": spatial_regularization_strength,
        "z_dim": z_dim,
        "lr": lr,
        "epochs": epochs,
        "max_patience": max_patience,
        "min_stop": min_stop,
        "random_seed": random_seed,
        "gpu": gpu,
        "regularization_acceleration": regularization_acceleration,
        "edge_subset_sz": edge_subset_sz,
        "seg_n_neighbors": seg_n_neighbors,
        "seg_resolution": seg_resolution,
        "target_n_clusters": int(target_n_clusters),
        "run_psm": run_psm,
        "psm_n_neighbors": psm_n_neighbors,
        "psm_resolution": psm_resolution,
        "allow_preprocessing_fallback": bool(allow_preprocessing_fallback),
        "hvg_flavor": hvg_flavor,
        "use_raw_counts": bool(use_raw_counts),
    }
    if h5ad_path:
        payload["h5ad_path"] = h5ad_path
    if visium_h5_path:
        payload["visium_h5_path"] = visium_h5_path
    if visium_spatial_dir:
        payload["visium_spatial_dir"] = visium_spatial_dir
    if spaceranger_dir:
        payload["spaceranger_dir"] = spaceranger_dir
    if counts_h5ad_path:
        payload["counts_h5ad_path"] = counts_h5ad_path
    if coords_csv:
        payload["coords_csv"] = coords_csv
    return run_worker_json(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, payload)


if __name__ == "__main__":
    mcp.run()
