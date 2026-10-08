#!/usr/bin/env python3
"""SpaTrio spot-to-cell optimal-transport mapping MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "spatrio"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPATRIO",
    "/opt/conda/envs/spatrio/bin/python",
    "/workspace/epic-fermat/agent/tools/spatrio_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def spatrio_align_multiomics(
    rna_h5ad: str,
    other_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    annotation_key: str = "cell_type",
    n_hvg: int = 2000,
    alpha: float = 0.1,
    numItermax: int = 200,
    seed: int = 0,
    drop_unlabeled: bool = False,
    layer: str = "",
) -> dict[str, Any]:
    """
    Map single cells onto the spots of one spatial slice with SpaTrio (optimal transport).

    SpaTrio's ``ot_alignment`` solves a fused Gromov-Wasserstein optimal-transport problem between
    the spots of a spatial transcriptomics slice and the cells of a single-cell (multi-omics)
    dataset. Its expression cost is computed on the genes BOTH inputs share (SpaTrio inner-joins
    them and normalises them itself), so ``other_h5ad`` must hold RNA counts under the same gene
    identifiers as ``rna_h5ad``; a protein-only or ATAC-peak-only matrix shares no features and is
    refused. The second modality enters through ``other_h5ad.obsm['reduction']``, the embedding its
    cell graph is built on. It is not a serial-section aligner and writes no coordinates.

    Output: ``spatrio_aligned.csv``, the transport plan in long format -- one ``spot,cell,value``
    row per spot-cell pair (n_spots x n_cells rows, after a leading row-index column). It is in
    ``output_files['alignment_csv']``; ``output_files['aligned_h5ad']`` is a deprecated alias of the
    same CSV (it never was an h5ad). ``summary.coupling_shape`` is ``[n_spots, n_cells]``;
    ``data.n_shared_features`` is the size of the feature set the cost used.

    Parameters
    ----------
    rna_h5ad:
        Path to the spatial RNA AnnData (.h5ad) with spatial coordinates in obsm[spatial_key] and
        raw integer counts in X (or in ``layer``). Spots with obs['in_tissue'] == 0 (background
        glass, as CELLxGENE Visium exports carry) are left out of the plan and counted in
        params.in_tissue_filter; data.n_spots_rna_input is the count supplied.
    other_h5ad:
        Path to the single-cell (multi-omics) AnnData (.h5ad) whose cells are mapped: raw RNA counts
        in X (or in ``layer``) sharing gene names with rna_h5ad, and optionally obsm['reduction'],
        an embedding of its second modality (ATAC LSI, ADT PCA, ...). Without one, the worker
        builds the cell graph from a PCA (up to 30 comps) of these RNA counts and says so in
        params.reduction_source.
    output_dir:
        Directory where SpaTrio outputs will be saved.
    spatial_key:
        obsm key for spatial coordinates.
    annotation_key:
        obs column giving each spot / cell a type for SpaTrio's type-aware graph distances, read
        from whichever input has it. An input without the column runs untyped and the run says so
        in its warnings and in params.type_source_rna / params.type_source_other; no other column
        (not ``obs['type']``, not ``CellType``) is read in its place. Empty string = untyped by
        request.
    n_hvg:
        Accepted for compatibility and not used: SpaTrio's expression cost uses every feature the
        two inputs share. Listed in params.ignored.
    alpha:
        SpaTrio's fused Gromov-Wasserstein trade-off between the graph (structure) term and the
        expression term, 0..1.
    numItermax:
        Maximum iterations for the OT solver.
    seed:
        Random seed (numpy, and the PCA computed for obsm['reduction'] when none is supplied).
    drop_unlabeled:
        An observation whose annotation_key label is NaN/empty is not a type. False (default):
        such an input is refused with the count. True: those spots/cells are left out of the
        mapping and data.n_unlabeled_dropped_rna / _other say how many.
    layer:
        Name of a layer holding raw counts, read instead of X in each input that has it. Empty
        reads X. Either way the matrix must be raw integer counts: SpaTrio normalises it itself,
        so normalised or log data is refused rather than normalised twice.
    """
    args = [
        "--rna-h5ad",
        rna_h5ad,
        "--other-h5ad",
        other_h5ad,
        "--output-dir",
        output_dir,
        "--spatial-key",
        spatial_key,
        "--annotation-key",
        annotation_key,
        "--n-hvg",
        str(n_hvg),
        "--alpha",
        str(alpha),
        "--numItermax",
        str(numItermax),
        "--seed",
        str(seed),
    ]
    if drop_unlabeled:
        args += ["--drop-unlabeled", "true"]
    if layer:
        args += ["--layer", layer]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
