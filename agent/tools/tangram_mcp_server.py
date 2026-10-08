#!/usr/bin/env python3
"""Tangram sc-to-spatial mapping MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "tangram"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "TANGRAM",
    "/opt/conda/envs/tangram-env/bin/python",
    "/workspace/epic-fermat/agent/tools/tangram_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def tangram_map_sc_to_spatial(
    sc_h5ad: str,
    spatial_h5ad: str | None = None,
    output_dir: str = default_output_dir("tangram_output"),
    annotation_key: str | None = "cell_type",
    mode: str | None = "clusters",
    n_markers_per_class: int | None = 100,
    num_epochs: int | None = 1000,
    density_prior: str | None = "rna_count_based",
    device: str | None = "auto",
    project_genes: bool = True,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Map single-cell RNA-seq data onto spatial transcriptomics data using Tangram.

    tangram_celltype_probabilities.csv (spots x cell types) holds each spot's cell-type composition,
    rows summing to 1. In 'clusters' mode it is Tangram's cluster map weighted by each cluster's share
    of the reference (obs['cluster_density'], the weights Tangram's density term trains the map with)
    and row-normalised; in 'cells' mode it is the reference cells mapped to the spot, counted per type
    and row-normalised. params.proportion_weighting says which ('cluster_density' or 'none'), and
    obsm['tangram_ct_pred'] of tangram_spatial_with_annotations.h5ad keeps Tangram's unweighted scores
    beside the composition in obsm['tangram_ct_proportions']. Spots with obs['in_tissue'] == 0
    (background) are left out of the mapping and every output, and params.in_tissue_filter says how
    many.

    Parameters
    ----------
    sc_h5ad:
        Path to single-cell AnnData (.h5ad) with cell-type labels in obs[annotation_key]. REQUIRED.
    spatial_h5ad:
        Path to spatial AnnData (.h5ad), e.g., 10x Visium; expression in X (the mapping does not use
        coordinates), gene identifiers of the same kind as the reference's. Required at runtime —
        if omitted, the wrapper returns a clean error rather than letting
        Pydantic raise a validation error in the MCP framework.
    output_dir:
        Output directory where all Tangram outputs will be saved: tangram_celltype_probabilities.csv,
        tangram_mapping_sc_to_spatial.h5ad, tangram_spatial_with_annotations.h5ad, and (when
        project_genes is True) tangram_projected_genes.h5ad.
    annotation_key:
        Column in scRNA obs with cell-type labels. Cells with no label (NaN/empty) are refused unless
        drop_unlabeled=True.
    mode:
        Tangram mapping mode: 'clusters' (paper-grade default) or 'cells'.
        'clusters' aggregates cells by annotation_key for stabler proportion
        estimation; 'cells' maps every individual single cell. The gene projection
        aggregates the reference by the same annotation_key in 'clusters' mode. 'cells' trains a
        dense cells x spots map (about 7 float32 copies of it at the peak); before training, whatever
        project_genes says, that size is checked against GPU memory when training runs on CUDA (the
        host's share, about 3 copies, against host memory) and against host memory on the CPU, and
        a map that cannot fit stops the run naming mode='clusters' (on CUDA also device='cpu' when
        host memory can hold it). params.mapping_memory_checked_against says which memory it was
        checked against.
    n_markers_per_class:
        Top N marker genes per annotation group used as training genes.
    num_epochs:
        Number of epochs for tg.map_cells_to_space (paper: 1000).
    density_prior:
        Tangram density prior ('rna_count_based' or 'uniform').
    device:
        'auto', 'cpu', or 'cuda:0'.
    project_genes:
        Run tg.project_genes and write tangram_projected_genes.h5ad (default True). The projection is
        a dense spots x reference-genes matrix (float64 in 'clusters' mode; 'cells' mode also
        densifies the reference); its size is checked against available memory before training, and a
        projection that cannot fit stops the run naming this knob. False skips it; the cell-type
        mapping and tangram_celltype_probabilities.csv do not depend on it.
    drop_unlabeled:
        If True, reference cells whose annotation_key label is missing are left out (the count is
        reported); if False (default) such a reference is refused with the count.
    """
    if spatial_h5ad is None:
        return {
            "status": "error",
            "tool": TOOL_NAME,
            "error": "spatial_h5ad is required: provide the path to the spatial transcriptomics AnnData (.h5ad).",
        }

    args = [
        "--task",
        "map",
        "--sc-h5ad",
        sc_h5ad,
        "--spatial-h5ad",
        spatial_h5ad,
        "--output-dir",
        output_dir,
    ]
    if annotation_key is not None:
        args += ["--annotation-key", annotation_key]
    if mode is not None:
        args += ["--mode", mode]
    if n_markers_per_class is not None:
        args += ["--n-markers-per-class", str(n_markers_per_class)]
    if num_epochs is not None:
        args += ["--num-epochs", str(num_epochs)]
    if density_prior is not None:
        args += ["--density-prior", density_prior]
    if device is not None:
        args += ["--device", device]
    if not project_genes:
        args.append("--no-project-genes")
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
