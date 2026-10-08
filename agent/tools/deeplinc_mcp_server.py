#!/usr/bin/env python3
"""Cell-type neighbourhood enrichment MCP wrapper for SpatialOmicsLab, served under the DeepLinc name.

What runs is a k-NN edge-type enrichment with a label-permutation test, not the DeepLinc VGAE; see
``tools/deeplinc_worker.py`` for the definition of the score and the p-value.
"""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "deeplinc"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "DEEPLINC",
    "/opt/conda/envs/deeplinc/bin/python",
    "/workspace/epic-fermat/agent/tools/deeplinc_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def deeplinc_cell_interactions(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    annotation_key: str = "cell_type",
    n_neighbors: int = 10,
    n_hvg: int = 2000,
    seed: int = 0,
    drop_unlabeled: bool = False,
    coords_key: str = "spatial",
    dims: int = 2,
    section_key: str | None = None,
) -> dict[str, Any]:
    """
    Score which cell types sit next to which, on a spatial k-NN graph with a label-permutation test.

    What runs (the tool keeps the DeepLinc name, but the DeepLinc VGAE is NOT run): each cell is
    joined to its n_neighbors nearest cells by position (the graph is symmetrised), and for every
    pair of cell types the number of edges joining them is divided by the number expected if the
    labels were placed at random on the same graph -- so 1.0 is random placement, above 1.0
    enriched, below 1.0 avoided. Significance is a one-sided test over 100 label permutations,
    p = (1 + #{permuted >= observed}) / 101. Gene expression is not used; only positions and labels.
    The payload names the method in params.method and sets params.deeplinc_model_run = false.

    Outputs: deeplinc_interaction_scores.csv, deeplinc_pvalues.csv,
    deeplinc_significant_interactions.csv (type_a, type_b, score, pvalue; header even when empty),
    deeplinc_adjacency_edges.csv (cell_a, cell_b; the graph, one row per undirected edge) and, up to
    20,000 cells, the same graph as a dense n x n deeplinc_adjacency.csv.

    Two-dimensional by design: the k-NN graph is built on two columns, so dims=3 is refused, and a file
    holding two or more sections is refused unless section_key names them (a 2D run would overlay them).
    With section_key the run is per section: each section's files in output_dir/section_<label>/, and
    one long deeplinc_significant_interactions.csv at the top level with a leading `section` column.
    params.mode is "per-section-2d" (or "2d" for one plane), params.sections lists the sections and
    params.frame the coordinates read; the same is written to sog_run_provenance.json.

    Parameters
    ----------
    st_h5ad:
        Path to spatial AnnData (.h5ad) with spatial coordinates in
        obsm[spatial_key] and cell type annotations in obs[annotation_key]. The expression matrix is
        not read. Spots with obs['in_tissue'] == 0 (the background glass a CELLxGENE Visium export
        carries, labelled 'unknown') are left out before the graph is built, so they are neither a
        cell type nor neighbours; the count is in data.n_spots_off_tissue_dropped and
        params.in_tissue_filter.
    output_dir:
        Directory where the outputs will be saved.
    spatial_key:
        obsm key for spatial coordinates (coords_key, when not 'spatial', takes its place).
    annotation_key:
        obs column containing cell type annotations.
    n_neighbors:
        Number of spatial neighbors for building the interaction graph.
    n_hvg:
        Accepted for compatibility and ignored: no expression is read, so there are no genes to
        select. Listed in params.ignored.
    seed:
        Random seed for the label permutations.
    drop_unlabeled:
        If true, leave out cells whose label is missing (NaN/empty) and report how many. If false
        (default), a missing label stops the run with the count -- it is never scored as a type.
    coords_key, dims, section_key:
        coords_key: the obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
        'spatial_3d_aligned'). dims: 2 or 3; 3 builds the graph in the aligned frame in micrometres and
        needs a frame with recorded units and a measured or registered z. section_key: the obs column
        naming sections; required for a 2D run on a multi-section file (the run is per section) and for
        the cross-section edge count of a 3D run.
        For DeepLinc: dims=3 is refused ("DeepLinc builds its neighbourhood in two dimensions; run per
        section with `dims=2, section_key=<column>`.").
    """
    args = [
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--spatial-key",
        spatial_key,
        "--dims",
        str(dims),
        "--annotation-key",
        annotation_key,
        "--n-neighbors",
        str(n_neighbors),
        "--n-hvg",
        str(n_hvg),
        "--seed",
        str(seed),
    ]
    if coords_key != "spatial":
        args += ["--coords-key", coords_key]  # the worker's own flag for it; it wins over --spatial-key
    if section_key:
        args += ["--section-key", section_key]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def deeplinc_cell_interactions_csv(
    counts_csv: str,
    coord_csv: str,
    cell_type_csv: str,
    output_dir: str,
    n_neighbors: int = 10,
    seed: int = 0,
    drop_unlabeled: bool = False,
    dims: int = 2,
    section_key: str | None = None,
) -> dict[str, Any]:
    """
    Same k-NN cell-type edge enrichment as deeplinc_cell_interactions, from three CSV files.

    The DeepLinc VGAE is NOT run; see deeplinc_cell_interactions for what is. Only cells present in
    all three files are analysed, and the payload says how many each file lost.

    Parameters
    ----------
    counts_csv:
        Path to a counts CSV with an index column. Only its cell IDs and gene count are read; the
        values are not used. Cells x genes is the documented layout; a genes x cells table (what
        convert_h5ad_to_csv writes by default) is recognised by which axis carries the cell IDs of
        the other two files, and params.counts_orientation says which was found.
    coord_csv:
        Path to coordinates CSV (cells as rows, cell names in the first column). Axis columns are matched
        by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col, row/col
        or x/y -- so Space Ranger's tissue_positions.csv can be passed as it is. The headerless
        tissue_positions_list.csv of Space Ranger before 2.0 is recognised from its first line and read
        with 10x's column names (params.coord_header says how the file was read); a headerless file of
        another shape is refused. When the file has an in_tissue column, spots with in_tissue == 0 are
        left out and counted (data.n_spots_off_tissue_dropped). Only two axis columns are read: a file that
        also holds a `z` column or a section-like column (`section`, `slice_id`, ...) with two or more
        values is refused unless section_key names the column, rather than having that axis dropped.
    cell_type_csv:
        Path to cell type CSV (cells as rows, cell names in the first column, labels in the next).
    output_dir:
        Directory where the outputs will be saved.
    n_neighbors:
        Number of spatial neighbors for building the interaction graph.
    seed:
        Random seed for the label permutations.
    drop_unlabeled:
        If true, leave out cells whose label is missing (NaN/empty) and report how many. If false
        (default), a missing label stops the run with the count -- it is never scored as a type.
    dims:
        2; dims=3 is refused ("DeepLinc builds its neighbourhood in two dimensions; run per section with
        `dims=2, section_key=<column>`.").
    section_key:
        A column of coord_csv naming each cell's section; the run is then per section, as in
        deeplinc_cell_interactions.
    """
    args = [
        "--counts-csv",
        counts_csv,
        "--coord-csv",
        coord_csv,
        "--cell-type-csv",
        cell_type_csv,
        "--output-dir",
        output_dir,
        "--n-neighbors",
        str(n_neighbors),
        "--seed",
        str(seed),
        "--dims",
        str(dims),
    ]
    if section_key:
        args += ["--section-key", section_key]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
