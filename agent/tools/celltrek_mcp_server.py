#!/usr/bin/env python3
"""
CellTrek MCP server — spatial cell mapping via co-embedding.

Calls celltrek_worker.R inside /opt/conda/envs/celltrek.
"""

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "celltrek"
DEFAULT_RSCRIPT = "/opt/conda/envs/celltrek/bin/Rscript"
DEFAULT_WORKER = "/workspace/epic-fermat/agent/tools/celltrek_worker.R"

RSCRIPT, WORKER_R = get_worker_paths("CELLTREK", DEFAULT_RSCRIPT, DEFAULT_WORKER)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def celltrek_spatial_mapping(
    sc_data: str,
    st_data: str,
    output_dir: str,
    celltype_col: str = "cell_type",
    n_components: int = 30,
    reduction: str = "pca",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Map single cells to spatial coordinates using CellTrek co-embedding.

    Both objects are log-normalised (Seurat LogNormalize) by the worker when they carry no
    normalised data layer of their own -- every .rds convert_h5ad_to_seurat_rds writes is counts
    only -- and a supplied data layer is used as given; ``params.normalisation`` says which
    happened on each side. Single-cell names are rewritten with R's make.names() (the form CellTrek
    keys cells by; ``summary.n_sc_cells_renamed`` counts them). CellTrek places a cell at up to 5
    spots: ``summary.n_mapped_cells`` counts distinct cells, ``summary.n_placements`` the rows of
    celltrek_mapped_coords.csv, whose ``coord_x`` is the image row and ``coord_y`` the image column.
    A cell is left unplaced (``summary.n_sc_cells_unmapped``) by CellTrek's mutual nearest-neighbour
    pruning alone: each cell keeps its 5 nearest spots or interpolated points, each of those its 5
    nearest cells, and a pair survives only when both keep it. CellTrek's distance cut
    (ntree x dist_thresh = 1000 x 0.55 = 550) lies above every random-forest distance, which
    randomForestSRC returns in [0, 1], so it removes nothing; ``params.distance_cut`` and a warning
    say so. The random-forest step holds an N x N distance matrix (N = cells + spots + 10,000 interpolated
    points); the worker refuses up front, with the numbers, when that cannot fit in free memory.
    The two objects are co-embedded on the genes both measure, so they must name genes the same way
    (both symbols or both Ensembl IDs): a pair with no gene in common is refused, and
    ``data.n_shared_genes`` counts them.

    Parameters
    ----------
    sc_data:
        Path to scRNA-seq Seurat RDS object.
    st_data:
        Path to spatial transcriptomics Seurat RDS object. Spots whose in_tissue metadata is 0
        (background glass: convert_h5ad_to_seurat_rds carries obs['in_tissue'] across, and a
        CELLxGENE Visium h5ad holds every array spot) are left out before mapping, so no cell is
        charted onto them; params.in_tissue_filter and a warning count them, data.n_st_spots is the
        spots mapped onto and data.n_st_spots_supplied the object as given. Spot positions come from its own
        Visium image, else from the 'spatial' reduction or x_coord/y_coord metadata that
        convert_h5ad_to_seurat_rds writes (AnnData's obsm['spatial'] order: x = image column,
        y = image row). An own image is matched to the spots by name, and it must cover every
        spot: an object whose first image holds only some of them (several sections) is refused.
    output_dir:
        Directory for output files.
    celltype_col:
        Column in scRNA-seq metadata with cell type labels.
    n_components:
        Number of components of ``reduction`` CellTrek's random forest is trained on: 2..50 for
        'pca', exactly 2 for 'umap'. Values outside that range are refused before the run starts.
    reduction:
        Which of the two embeddings CellTrek's traint() builds on the joint object to use: 'pca'
        (50 components) or 'umap' (2 components). Any other name is refused.
    seed:
        Random seed for CellTrek's own random steps: the interpolated points, the random forest
        and the charting. traint()'s Seurat steps (CCA, PCA, UMAP) use Seurat's fixed seed 42
        whatever this says. A multi-threaded random forest is not bit-reproducible, so placements
        still differ between runs at one seed unless the worker runs with RF_CORES=1;
        ``params.seed_scope`` and ``params.random_forest_threads`` say which applied.
    """
    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--sc-h5ad",
        sc_data,
        "--st-h5ad",
        st_data,
        "--output-dir",
        output_dir,
        "--celltype-col",
        celltype_col,
        "--n-components",
        str(n_components),
        "--reduction",
        reduction,
        "--seed",
        str(seed),
    ]
    return run_worker_cli(TOOL_NAME, RSCRIPT, WORKER_R, args)


if __name__ == "__main__":
    mcp.run()
