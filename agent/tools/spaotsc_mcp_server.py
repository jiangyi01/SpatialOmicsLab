#!/usr/bin/env python3
"""SpaOTsc MCP wrapper for SpatialOmicsLab."""

import os
from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "spaotsc"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPAOTSC",
    "/opt/conda/envs/spaotsc/bin/python",
    "/workspace/epic-fermat/agent/tools/spaotsc_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def spaotsc_run(
    out_tag: str = "demo",
    output_dir: str | None = None,
    data_dir: str | None = None,
    sc_expr_path: str | None = None,
    is_dmat_path: str | None = None,
    sc_dmat_path: str | None = None,
    cost_matrix_path: str | None = None,
    selected_genes_path: str | None = None,
    raw_sc_path: str | None = None,
    raw_is_path: str | None = None,
    raw_is_coord_path: str | None = None,
    run_mapping: bool = True,
    run_cellcell_distance: bool = True,
    run_clustering: bool = False,
    run_signaling: bool = False,
    ligands: list[str] | None = None,
    receptors: list[str] | None = None,
    ds_genes_up: list[str] | None = None,
    ds_genes_down: list[str] | None = None,
    seed: int = 1234,
    use_landmark: bool = True,
    use_raw_counts: bool = False,
    coords_key: str = "spatial",
    dims: int = 2,
    section_key: str | None = None,
) -> dict[str, Any]:
    """Run the upstream SpaOTsc package: map scRNA-seq cells onto spatial spots by optimal transport,
    then optionally derive spatial cell-cell distances, spatial subclusters and ligand-receptor signaling.

    Inputs. A single-cell reference is REQUIRED: either raw_sc_path (scRNA-seq h5ad/h5/csv/tsv/txt,
    cells x genes, raw counts) plus raw_is_path (spatial h5ad/h5/csv/tsv/txt), or precomputed matrices
    via data_dir (dm_sc_normalized.txt, dm_is.txt or dm_pos_pos_geodesic_mgeom.npy,
    dm_scanpy_pca40_pcc.npy, dm_sc_is_mcc.npy, optional selected_genes.txt) or the explicit
    sc_expr_path / is_dmat_path / sc_dmat_path / cost_matrix_path. A path you name must exist. When any
    precomputed matrix is missing and both raw inputs are given, all four are rebuilt from raw input
    (CPM 1e4 + log2 on the shared genes, or on selected_genes_path; PCA-40 Pearson sc_dmat; a cost
    matrix from profiles binarised strictly above each gene's 70th percentile; Euclidean is_dmat) and
    any matrix you supplied is reported under params.ignored. Spot coordinates come from
    obsm['spatial'] or raw_is_coord_path (a csv/tsv whose coordinate columns are found by name --
    pxl_row_in_fullres/pxl_col_in_fullres, imagerow/imagecol, array_row/array_col, row/col or x/y --
    and whose rows are matched by spot ID); without coordinates the run stops. Background spots are not
    mapped: spots with obs['in_tissue'] == 0 in an h5ad raw_is_path (a CELLxGENE Visium export carries
    every array spot), or flagged 0 by an in_tissue column of raw_is_coord_path, are left out before the
    matrices are built and counted in data.n_spots_off_tissue_dropped and params.in_tissue_filter.
    The raw build normalises X as counts: a raw input whose matrix holds negative or NaN values (scaled
    data) is refused, one with non-integer values runs with a warning, and use_raw_counts=True reads the
    counts from adata.raw instead (params.expression_source, params.x_matrix_kind).

    coords_key: the obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
    'spatial_3d_aligned'). dims: 2 or 3; 3 builds the graph in the aligned frame in micrometres and
    needs a frame with recorded units and a measured or registered z. section_key: the obs column
    naming sections; required for a 2D run on a multi-section file (the run is per section) and for
    the cross-section edge count of a 3D run.
    For SpaOTsc the "graph" is is_dmat, the spot-to-spot distance matrix, and the frame is read from an
    h5ad raw_is_path only: with dims=3 is_dmat (and the 10/50/100 signal ranges) are in micrometres, and
    dims=3 with raw_is_coord_path is refused. SpaOTsc builds one is_dmat over the whole input, so a 2D run
    on a multi-section file is refused even with section_key: run 3D, or once per section on a file
    subset to one section. params.mode ("3d" or "2d") and params.frame (coords_key, dims, units_per_axis_um,
    z_source, section_key, sections) say which ran.

    Steps, each needing the one before it (a requested step that cannot run is an error, never a
    silent skip):
      run_mapping            transport plan, cells x spots (mapping/transport_plan.*)
      run_cellcell_distance  needs run_mapping. Upstream solves one Sinkhorn problem per PAIR of cells,
                             n_sc*(n_sc-1)/2 of them (about 4.5 million for 3,000 cells), so this is by
                             far the slowest step (ccd/cell_cell_distance.*). use_landmark (default True)
                             solves each on 100 landmark spots and needs at least 100 spots.
      run_clustering         needs run_cellcell_distance and more than 50 cells (clustering/labels.csv:
                             cell, cluster = spatial subcluster "i_j", expression_cluster = i; a cell in a
                             spatial group of 3 or fewer has an empty cluster and is counted in params)
      run_signaling          needs run_cellcell_distance and at least one ligand and one receptor, all in
                             the shared gene panel (signaling/signaling_scores.*: an n_sc x n_sc score
                             matrix, sender row -> receiver column). With ds_genes_up/ds_genes_down the
                             signal range is also inferred over 10/50/100 is_dmat units
                             (signaling/inferred_range.*); without them params.signal_range says it was
                             not run.

    Memory: SpaOTsc is dense by construction (n_sc x n_sc, n_sc x n_is and n_is x n_is float64). The
    worker estimates the peak before allocating and refuses with the numbers when it cannot fit (available
    memory is MemAvailable, or the room left under a container's cgroup memory limit when that is
    smaller); it never subsamples. output_files lists only files this run wrote.

    Args:
        out_tag: Tag for the default output directory (spaotsc_<out_tag>) when output_dir is unset.
        output_dir: Output directory.
        data_dir: Folder with the SpaOTsc tutorial filenames above.
        sc_expr_path: Precomputed sc expression table (cells x genes, first column = cell ID).
        is_dmat_path: Precomputed spot x spot distance matrix.
        sc_dmat_path: Precomputed cell x cell dissimilarity matrix.
        cost_matrix_path: Precomputed cells x spots dissimilarity matrix.
        selected_genes_path: Gene panel (one name per line) used when building from raw input.
        raw_sc_path: Raw scRNA-seq reference (cells x genes, counts).
        raw_is_path: Raw spatial data (spots x genes, counts); coordinates from obsm['spatial'].
        raw_is_coord_path: Spot coordinates with a spot-ID column (overrides obsm['spatial']).
        run_mapping: Compute the transport plan.
        run_cellcell_distance: Compute the spatial cell-cell distance (needs run_mapping).
        run_clustering: SpaOTsc spatial subclustering (needs run_cellcell_distance).
        run_signaling: Ligand-receptor signaling (needs run_cellcell_distance, ligands and receptors).
        ligands: Ligand gene symbols for run_signaling.
        receptors: Receptor gene symbols for run_signaling.
        ds_genes_up: Downstream genes up-regulated by the signal (enables the signal-range step).
        ds_genes_down: Downstream genes down-regulated by the signal (enables the signal-range step).
        seed: Seeds Python's random module and numpy.
        use_landmark: Solve the cell-cell distance on 100 landmark spots instead of every spot.
        use_raw_counts: Read the counts from adata.raw instead of X, for each h5ad raw input that carries an
            adata.raw (an input without one keeps X, with a warning); default False reads X. Use it for a
            CELLxGENE-style h5ad whose X is processed and whose counts are in adata.raw. Only the raw build
            reads it; with precomputed matrices it is listed in params.ignored.
        coords_key: The obsm key of raw_is_path holding the coordinates (default 'spatial'; an aligned 3D
            frame such as 'spatial_3d_aligned').
        dims: 2 or 3; 3 builds is_dmat in the aligned frame in micrometres and needs a frame with recorded
            units and a measured or registered z.
        section_key: The obs column naming sections of raw_is_path; recorded in params.frame of a 3D run.
    """
    ligands = ligands or []
    receptors = receptors or []
    ds_genes_up = ds_genes_up or []
    ds_genes_down = ds_genes_down or []

    if output_dir:
        outdir = os.path.abspath(output_dir)
    else:
        outdir = os.path.abspath(default_output_dir(f"spaotsc_{out_tag}"))
    os.makedirs(outdir, exist_ok=True)

    def _abs(p: str | None) -> str | None:
        return os.path.abspath(p) if p else None

    args = ["--output-dir", outdir, "--seed", str(seed)]

    if data_dir:
        args += ["--data-dir", os.path.abspath(data_dir)]
    if sc_expr_path:
        args += ["--sc-expr", _abs(sc_expr_path)]
    if is_dmat_path:
        args += ["--is-dmat", _abs(is_dmat_path)]
    if sc_dmat_path:
        args += ["--sc-dmat", _abs(sc_dmat_path)]
    if cost_matrix_path:
        args += ["--cost-matrix", _abs(cost_matrix_path)]
    if selected_genes_path:
        args += ["--selected-genes", _abs(selected_genes_path)]
    if raw_sc_path:
        args += ["--raw-sc", _abs(raw_sc_path)]
    if raw_is_path:
        args += ["--raw-is", _abs(raw_is_path)]
    if raw_is_coord_path:
        args += ["--raw-is-coord", _abs(raw_is_coord_path)]

    if run_mapping:
        args.append("--run-mapping")
    if run_cellcell_distance:
        args.append("--run-ccd")
    if run_clustering:
        args.append("--run-clustering")
    if run_signaling:
        args.append("--run-signaling")
    if use_landmark:
        args.append("--use-landmark")
    if use_raw_counts:
        args.append("--use-raw-counts")
    args += ["--dims", str(int(dims))]
    if coords_key and coords_key != "spatial":
        args += ["--coords-key", coords_key]
    if section_key:
        args += ["--section-key", section_key]

    for g in ligands:
        args += ["--ligand", g]
    for g in receptors:
        args += ["--receptor", g]
    for g in ds_genes_up:
        args += ["--ds-up", g]
    for g in ds_genes_down:
        args += ["--ds-down", g]

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
