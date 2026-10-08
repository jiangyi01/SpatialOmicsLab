#!/usr/bin/env python3
"""novoSpaRc spatial reconstruction MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "novosparc"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "NOVOSPARC",
    "/opt/conda/envs/novosparc/bin/python",
    "/workspace/epic-fermat/agent/tools/novosparc_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def novosparc_reconstruct_spatial(
    sc_h5ad: str,
    output_dir: str,
    st_h5ad: str = "",
    spatial_key: str = "spatial",
    annotation_key: str = "cell_type",
    num_locations: int = 1000,
    alpha: float = 0.5,
    n_hvg: int = 2000,
    num_neighbors_s: int = 5,
    num_neighbors_t: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Reconstruct spatial gene expression from scRNA-seq using novoSpaRc (optimal transport).

    novoSpaRc maps single cells to locations with upstream ``novosparc.cm.Tissue``: a
    Gromov-Wasserstein transport that matches the cells' expression graph to the locations'
    physical graph, optionally plus a linear "atlas" cost from reference expression. The payload's
    ``params.mode`` names which of three modes ran:

    * atlas-guided -- ``st_h5ad`` given and ``alpha > 0``: the reference's expression of the genes
      it shares with the scRNA-seq HVGs is the atlas (``params.n_markers`` of them);
    * reference geometry -- ``st_h5ad`` given and ``alpha = 0``: only the reference coordinates
      are used;
    * de novo grid -- no ``st_h5ad``: novoSpaRc's own target grid, structure only.

    The cost and coupling matrices are dense (cells x cells, locations x locations, cells x
    locations); that is intrinsic to the method. The worker estimates their memory first and fails
    with the numbers when it does not fit. Cells and locations are never subsampled. The entropic
    regularisation is searched over 5e-4, 5e-3, 5e-2, 5e-1; the first that runs without numerical
    errors is published as ``params.epsilon_used``, and if none does the run fails instead of
    publishing a coupling Sinkhorn gave up on. ``params.converged`` is read from the solver itself:
    true only when the Gromov-Wasserstein loop reached its tolerance (``params.gw_tol``, after
    ``params.gw_iterations`` of ``params.gw_max_iter`` iterations) and the last Sinkhorn solve
    converged; false when either stopped at its iteration limit; null when it could not be read.

    Parameters
    ----------
    sc_h5ad:
        Path to scRNA-seq AnnData (.h5ad) file.
    output_dir:
        Directory where novoSpaRc outputs will be saved.
    st_h5ad:
        Optional path to reference spatial AnnData (.h5ad). Its ``obsm[spatial_key]`` coordinates
        are the target locations and its spot barcodes label them; with ``alpha > 0`` its
        expression is also the atlas. Spots with ``obs['in_tissue'] == 0`` (background, as in
        CELLxGENE Visium exports) are left out first and counted in ``params.in_tissue_filter``.
        Its gene identifiers are matched to the scRNA-seq ones directly or through a gene-symbol
        var column (reported as ``params.gene_ids_*``). If empty, a de novo target grid is
        constructed.
    spatial_key:
        obsm key for spatial coordinates in the reference h5ad (unused without ``st_h5ad``).
    annotation_key:
        obs column of the scRNA-seq data holding cell-type labels. When present, each location's
        share of transported mass per cell type is written to ``novosparc_celltype_location.csv``
        (locations x cell types); unlabelled cells are counted, not treated as a class. When the
        column is absent the run still succeeds and ``annotation_key`` is listed in
        ``params.ignored``; pass an empty string to skip the table.
    num_locations:
        Number of target locations when no reference is provided. novoSpaRc builds a full grid
        (1000 requested gives 1015); ``data.n_locations`` reports the number built. Unused with
        ``st_h5ad``.
    alpha:
        novoSpaRc's ``alpha_linear`` (0-1): the weight of the reference-atlas (marker expression)
        cost against the structural Gromov-Wasserstein cost. 0 is structure only; 1 uses the atlas
        alone. Without ``st_h5ad`` there is no atlas, so ``alpha`` is not applied (listed in
        ``params.ignored``) and ``alpha = 1`` is refused.
    n_hvg:
        Number of highly variable genes (top variance of the prepared scRNA-seq expression) to use
        for reconstruction. Counts and linear-scale data (CPM/TPM, proportions, fractional counts)
        are normalize_total + log1p'd first; log-transformed or scaled data is used as supplied.
        ``params.sc_preprocessing`` says which.
    num_neighbors_s:
        Number of neighbors in the source (expression) graph.
    num_neighbors_t:
        Number of neighbors in the target (spatial) graph.
    seed:
        Accepted for compatibility; it has no effect. Nothing random runs: the transport starts from
        the product coupling (no random initialisation), the de novo grid is not random, and the kNN
        graphs are exact. It is listed in ``params.ignored`` (``params.random_seed`` still echoes it).

    Outputs
    -------
    ``novosparc_coupling.csv`` (rows: scRNA-seq cell IDs; columns: location IDs -- the reference
    barcodes, or grid positions 0..n-1), ``novosparc_spatial_expression.csv`` and
    ``novosparc_reconstructed.h5ad`` (one row per location, keyed by the same location IDs; values
    are coupling-weighted expression sums), and the optional cell-type table above.
    """
    args = [
        "--sc-h5ad",
        sc_h5ad,
        "--output-dir",
        output_dir,
        "--st-h5ad",
        st_h5ad,
        "--spatial-key",
        spatial_key,
        "--annotation-key",
        annotation_key,
        "--num-locations",
        str(num_locations),
        "--alpha",
        str(alpha),
        "--n-hvg",
        str(n_hvg),
        "--num-neighbors-s",
        str(num_neighbors_s),
        "--num-neighbors-t",
        str(num_neighbors_t),
        "--seed",
        str(seed),
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
