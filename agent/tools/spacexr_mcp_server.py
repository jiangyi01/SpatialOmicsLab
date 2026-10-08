#!/usr/bin/env python3
"""spacexr RCTD cell type deconvolution MCP wrapper for SpatialOmicsLab."""

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli
from worker_utils import cpu_budget

TOOL_NAME = "spacexr"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "SPACEXR",
    "/opt/conda/envs/spacexr/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/spacexr_worker.R",
)

mcp = create_mcp(TOOL_NAME)

# The doublet_mode values spacexr's run.RCTD accepts.
VALID_MODES = ("doublet", "full", "multi")
# create.RCTD's CELL_MIN_INSTANCE: RCTD refuses a reference cell type with fewer cells.
CELL_MIN_INSTANCE = 25


@mcp.tool()
def spacexr_rctd_deconvolution(
    spatial_counts_csv: str,
    spatial_coords_csv: str,
    ref_counts_csv: str,
    ref_celltypes_csv: str,
    output_dir: str,
    mode: str = "doublet",
    max_cores: int = 0,
    gene_cutoff: float = 0.000125,
    fc_cutoff: float = 0.5,
    gene_cutoff_reg: float = 0.0002,
    fc_cutoff_reg: float = 0.75,
    UMI_min: int = 100,
    UMI_min_sigma: int = 100,
    seed: int = 0,
    n_max_cells: int = 0,
    ref_UMI_min: int = 100,
) -> dict[str, Any]:
    """
    Run RCTD cell type deconvolution on spatial transcriptomics data using spacexr.

    RCTD (Robust Cell Type Decomposition) assigns cell type proportions to each
    spatial spot by leveraging a single-cell RNA-seq reference.

    Parameters
    ----------
    spatial_counts_csv:
        Path to spatial gene expression counts CSV (genes x spots).
    spatial_coords_csv:
        Path to spatial coordinates CSV (spots as rows, spot names in the first column). Axis columns are
        matched by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col,
        row/col or x/y -- so Space Ranger's tissue_positions.csv can be passed as it is, and so can the
        headerless tissue_positions_list.csv of Space Ranger 1 (a first line that is a spot is read as
        one). When the file has an in_tissue column, counts spots it marks 0 (background) are left out
        and counted in data.n_spots_off_tissue_dropped and params.in_tissue_filter.
    ref_counts_csv:
        Path to scRNA-seq reference counts CSV (genes x cells).
    ref_celltypes_csv:
        Path to reference cell type annotation CSV (cell barcode as index). A one-label-column file
        is read as it is; a wider metadata table must have a column named celltype, cell_type,
        cell.type, annotation, annot, cluster or label (any case), and the column used is reported
        in data.celltype_column. Cells whose label is NA, empty, 'nan', 'none', 'na' or '<NA>' (any
        case) are left out and counted in data.n_ref_cells_unlabeled. RCTD rejects '/' in a cell
        type name, so '/' and whitespace become '_' in every output; summary.renamed_cell_types maps
        the new names back, and two labels that would become the same name stop the run.
    output_dir:
        Directory for RCTD output files. The worker writes:
          - proportions.csv (canonical prediction; rows=spots, cols=cell types, per-row normalized).
            In every mode this is RCTD's unconstrained full-model fit (results$weights in doublet
            mode, which is the same fit full mode returns), so any number of cell types can be
            non-zero in a spot; data.proportions_source says so in the payload.
          - spacexr_weights.csv (legacy schema for backwards compat): full mode, the proportions
            plus a spot column; doublet mode, spot / first_type / second_type / spot_class;
            multi mode, spot / cell_types / n_types / confident_types.
          - spacexr_rctd_doublet_weights.csv (doublet mode only): the doublet call as a spots x
            cell types matrix with at most 2 non-zero types per spot (a singlet is 1 for its
            first_type; a reject spot is NA).
          - spacexr_rctd_multi_weights.csv (multi mode only): the multi-type decomposition, at
            most 4 cell types per spot.
          - spacexr_rctd.rds (full RCTD object for re-analysis; written as soon as the fit ends).
    mode:
        RCTD mode: 'doublet' (classifies each spot as singlet / doublet / reject and fits at most 2
        types per spot, written to spacexr_rctd_doublet_weights.csv), 'full' (all types weighted),
        or 'multi' (up to 4 types per spot, written to spacexr_rctd_multi_weights.csv). Any other
        value is refused before R starts. proportions.csv is the full-model fit in all three.
    max_cores:
        Number of parallel R workers for RCTD. 0 (default) auto-scales to the box
        (capped for memory). Each worker's BLAS is pinned to a single thread so the
        workers don't oversubscribe the cores (which otherwise thrashes and can time
        RCTD out on a many-core server).
    gene_cutoff:
        Minimum normalized expression for gene filtering.
    fc_cutoff:
        Minimum log-fold-change for gene filtering.
    gene_cutoff_reg:
        Minimum normalized expression for regression gene filtering. Must be >=
        gene_cutoff.
    fc_cutoff_reg:
        Minimum log-fold-change for regression gene filtering. Must be >= fc_cutoff.
        RCTD restricts the spatial matrix to the genes the (gene_cutoff, fc_cutoff)
        pair selects, then indexes that same matrix with the genes the _reg pair
        selects; a lower _reg threshold selects genes that are no longer there and
        the run dies with 'subscript out of bounds'. The defaults satisfy this and
        are what every recorded Visium and Slide-seqV2 run used successfully. To
        loosen the regression filter, lower gene_cutoff/fc_cutoff by as much.
    UMI_min:
        Minimum total UMI count per spot (spots below are excluded).
    UMI_min_sigma:
        Minimum UMI for sigma estimation (RCTD's choose_sigma_c). For low-UMI
        platforms like Slide-seqV2, set to 100 (default for RCTD is 300, which
        filters all beads on Slide-seq and causes "N_fit of 0" failure).
    seed:
        Random seed for reproducibility (also picks the cells an n_max_cells cap keeps).
    n_max_cells:
        Opt-in cap on reference cells per cell type. 0 (default) uses every reference cell. A
        positive value (at least 25, RCTD's minimum per type) is passed to RCTD's Reference(),
        which keeps that many cells drawn at random from each larger type; the number left out is
        reported in data.n_ref_cells_downsampled. RCTD's own default was a silent 10,000.
    ref_UMI_min:
        Minimum UMI count, over the genes the reference shares with the spatial data, for a
        reference cell to be used (RCTD Reference()'s min_UMI; default 100, its own default).
        Cells below it are left out and counted in data.n_ref_cells_below_umi. After this filter a
        cell type with fewer than 25 cells is dropped, named in summary.dropped_cell_types, and
        absent from every output; summary.n_ref_cells is the number of cells RCTD actually used.

    Both count CSVs are read straight into a sparse matrix, a block of rows at a time, so a large
    slide is never held dense while it is read. RCTD's per-pixel fit does hold a dense spots x
    regression-genes matrix (plus one copy per parallel worker); a run whose lower-bound need
    exceeds the available memory stops before the fit, naming max_cores and the _reg cutoffs.
    """
    # RCTD builds two gene lists from the same reference: gene_list_bulk from (gene_cutoff,
    # fc_cutoff) and gene_list_reg from (gene_cutoff_reg, fc_cutoff_reg). create.RCTD restricts the
    # puck to gene_list_bulk (restrict_counts assigns puck@counts <- puck@counts[gene_list, keep]),
    # and choose_sigma_c later indexes that restricted puck by gene_list_reg. get_de_genes selects
    # on `logFC > fc_thresh & expr > expr_thresh`, so the reg list is a subset of the bulk list
    # exactly when both _reg thresholds are at least their twins. Below that, R dies mid-run with a
    # bare "subscript out of bounds" that names neither parameter and no line of ours.
    #
    # This is not hypothetical: our own docstring used to prescribe gene_cutoff_reg=5e-5 for
    # Slide-seqV2 against a 1.25e-4 gene_cutoff. All 9 recorded runs that took that advice died
    # here; all 8 that held the constraint finished, including 2 on Slide-seqV2. Refuse up front
    # with a message that names the constraint, rather than paying for an R crash to learn it.
    for reg_name, reg_val, bulk_name, bulk_val in (
        ("gene_cutoff_reg", gene_cutoff_reg, "gene_cutoff", gene_cutoff),
        ("fc_cutoff_reg", fc_cutoff_reg, "fc_cutoff", fc_cutoff),
    ):
        if reg_val < bulk_val:
            return {
                "status": "error",
                "error": (
                    f"{reg_name}={reg_val!r} is below {bulk_name}={bulk_val!r}. RCTD requires "
                    f"{reg_name} >= {bulk_name}; below it the run dies inside choose_sigma_c with "
                    f"'subscript out of bounds'. Either leave both at their defaults (which work on "
                    f"Visium and Slide-seqV2) or lower {bulk_name} to {reg_val!r} as well."
                ),
            }

    # run.RCTD accepts exactly these three, but only rejects anything else after the reference and
    # the slide have been read and create.RCTD has selected its genes. Refuse it before R starts.
    if mode not in VALID_MODES:
        return {
            "status": "error",
            "error": f"mode={mode!r} is not an RCTD mode; use one of {', '.join(VALID_MODES)}.",
        }
    if n_max_cells < 0 or 0 < n_max_cells < CELL_MIN_INSTANCE:
        return {
            "status": "error",
            "error": (
                f"n_max_cells={n_max_cells!r}: use 0 (keep every reference cell, the default) or a cap of at "
                f"least {CELL_MIN_INSTANCE} cells per type -- RCTD refuses a cell type with fewer."
            ),
        }
    if ref_UMI_min < 0:
        return {"status": "error", "error": f"ref_UMI_min={ref_UMI_min!r}: use a non-negative UMI count."}

    os.makedirs(output_dir, exist_ok=True)

    # RCTD spawns `max_cores` parallel R workers; if EACH also lets its BLAS use every core, that is
    # max_cores x n_cores threads (e.g. 4 x 96 = 384 on a 96-core box) -> the cores thrash and RCTD runs
    # far SLOWER, timing out on big servers (measured: load average ~328 vs ~3, RCTD stalls in init).
    # Cap BLAS/OMP to a single thread PER worker so RCTD's own parallelism is the ONLY parallelism. The R
    # subprocess inherits os.environ and OpenBLAS/MKL read these at R startup (a Sys.setenv inside R is too
    # late), so set them here. setdefault respects an explicit operator override.
    for _thr_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_thr_var, "1")

    # Auto-scale RCTD's own parallelism to the box (max_cores<=0 => auto). With BLAS pinned to 1 thread
    # per worker (above), N workers == N cores with no oversubscription, so more workers is safe on any
    # box and result-neutral (RCTD fits each spot independently). Without this the default of 4 cores
    # left RCTD impractically slow on a big server (it could exceed the agent's turn timeout). Capped so
    # per-worker memory (each loads the reference signatures) stays bounded.
    #
    # cpu_budget(), not os.cpu_count(): the latter reports the machine, so under a cgroup quota or a
    # CPU affinity mask (container, Slurm, k8s) a 96-core host that granted us 4 CPUs still read 96 and
    # started 16 R workers on 4 cores -- re-creating, from the other direction, the oversubscription
    # this block exists to prevent.
    if max_cores <= 0:
        max_cores = cpu_budget(cap=16)

    args = [
        "--spatial-counts-csv",
        spatial_counts_csv,
        "--spatial-coords-csv",
        spatial_coords_csv,
        "--ref-counts-csv",
        ref_counts_csv,
        "--ref-celltypes-csv",
        ref_celltypes_csv,
        "--output-dir",
        output_dir,
        "--mode",
        mode,
        "--max-cores",
        str(max_cores),
        "--gene-cutoff",
        str(gene_cutoff),
        "--fc-cutoff",
        str(fc_cutoff),
        "--gene-cutoff-reg",
        str(gene_cutoff_reg),
        "--fc-cutoff-reg",
        str(fc_cutoff_reg),
        "--umi-min",
        str(UMI_min),
        "--umi-min-sigma",
        str(UMI_min_sigma),
        "--seed",
        str(seed),
        "--n-max-cells",
        str(n_max_cells),
        "--ref-umi-min",
        str(ref_UMI_min),
    ]

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
