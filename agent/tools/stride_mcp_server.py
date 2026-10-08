#!/usr/bin/env python3
"""STRIDE deconvolution MCP wrapper for SpatialOmicsLab."""

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "stride"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "STRIDE",
    "/opt/conda/envs/stride/bin/python",
    "/workspace/epic-fermat/agent/tools/stride_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def stride_deconvolution(
    sc_h5ad: str,
    spatial_h5ad: str,
    output_dir: str,
    annotation_key: str,
    outprefix: str = "stride",
    normalize: bool = True,
    ntopics: list[int] | None = None,
    gene_use_file: str | None = None,
    task: str = "deconvolution",
    drop_unlabeled: bool = False,
    st_scale_factor: float | None = None,
    sc_scale_factor: float | None = None,
) -> dict[str, Any]:
    """
    Run STRIDE deconvolution on spatial transcriptomics data.

    STRIDE is run on the genes the two objects share. If fewer than 500 genes are shared -- which
    is normal for a targeted panel (MERFISH, Xenium, CosMx, seqFISH) and for any thin reference
    match -- STRIDE's own QC crashes, so the tool works around it by ranking the shared panel by
    raw variance and deconvolving on the top 50 genes only. Pass gene_use_file to decline that
    substitution and choose the panel yourself. The payload reports the supplied panels as n_genes
    (spatial) and n_genes_sc (single-cell), the shared panel as n_overlap_genes, and the genes the
    deconvolution was actually computed on as n_genes_used, with a warning naming each cut.

    Both count matrices are rounded to the nearest integer and handed to STRIDE as sparse 10x HDF5
    (STRIDE's plain-text reader would expand the whole dense matrix into Python floats). A reference
    cell with no label (NaN/empty) stops the run unless drop_unlabeled is true: a missing label is not
    a cell type. params.gene_selection names the gene list the topic model was trained on and
    params.ntopics_selected the topic number STRIDE chose. Counts are read from layers['counts'],
    then adata.raw, then X (params.expression_source / expression_source_sc say which); values that
    are not whole numbers are rounded with a warning.

    Spots with obs['in_tissue'] == 0 (background glass in a CELLxGENE export) are left out before
    anything runs: data.n_spots is the slide supplied, data.n_spots_used the spots deconvolved, and
    params.in_tissue_filter counts the rest. A spot with no count in the genes STRIDE's topic model
    uses gets only the model's topic prior from STRIDE -- the same row for every such spot -- so it
    is counted in data.n_spots_without_counts, named in a warning, and given no dominant cell type.

    STRIDE scales each spot (cell) to count / total * scale factor. Its default factor is the 75th
    percentile of counts per spot rounded to the nearest 1000, which is 0 on a slide or reference
    whose 75th percentile is under 500 (Xenium, VisiumHD bins): every scaled count would be 0 and
    every spot would get the same prior-only composition. The tool then passes the unrounded 75th
    percentile instead, with a warning; params.st_scale_factor / sc_scale_factor are the factors
    used and *_scale_factor_source says where each came from.

    Parameters
    ----------
    sc_h5ad:
        Path to single-cell AnnData (.h5ad) with cell type labels.
    spatial_h5ad:
        Path to spatial AnnData (.h5ad) to be deconvolved.
    output_dir:
        Directory for STRIDE outputs.
    annotation_key:
        Column in sc_h5ad.obs with cell type labels.
    outprefix:
        Prefix for STRIDE output files. STRIDE loads an existing <outprefix>_topic_spot_mat_<k>.npz
        instead of recomputing it, which would pair an earlier topic x spot matrix with the newly
        trained model. A matrix an earlier run of this tool left in output_dir (no newer than the
        <outprefix>_dominant_celltype_per_spot.csv that run wrote) is replaced: it is removed just
        before STRIDE starts and listed in params.stale_topic_matrices_removed, with a warning. Any
        other such matrix stops the run and is left untouched; use a new output_dir or another
        outprefix.
    normalize:
        If True, pass the "--normalize" flag to STRIDE CLI.
    ntopics:
        List of topic numbers to evaluate. If None, STRIDE auto-ranges
        from N_celltypes to 3*N_celltypes. Providing a focused list
        (e.g. [44, 55, 66, 77, 88]) dramatically speeds up model selection.
    gene_use_file:
        Path to a plain-text file with one gene symbol per line, restricting the deconvolution to
        those genes. Leave unset to let STRIDE choose -- except on a slide sharing fewer than 500
        genes with the reference, where leaving it unset means the tool substitutes its own
        50-gene raw-variance panel (see above). Genes not in the shared panel are ignored. The
        literal 'All' trains on every shared gene. A path that is not a readable file, or a list
        with no gene in the shared panel, stops the run: STRIDE itself would silently replace the
        list with its own marker search.
    task:
        STRIDE worker task. 'deconvolution' is the only supported value; any other is rejected.
    drop_unlabeled:
        Leave out reference cells whose label in obs[annotation_key] is missing (NaN/empty) and
        report the count in params.n_reference_cells_dropped_unlabeled. Default False: such cells
        stop the run with their count, because a missing label is not a cell type.
    st_scale_factor:
        Scale factor STRIDE normalises each spot to (count / total * factor), a number > 0, e.g.
        10000. Unset: STRIDE's default (75th percentile of counts per spot over the shared genes,
        rounded to the nearest 1000), or the unrounded percentile when that rounds to 0. A slide
        where three quarters of the spots have no count in the shared genes stops the run unless
        this is set.
    sc_scale_factor:
        The same for the reference cells (counts per cell). Unset: the same rule as st_scale_factor.
    """
    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--task",
        task,
        "--sc-h5ad",
        sc_h5ad,
        "--spatial-h5ad",
        spatial_h5ad,
        "--output-dir",
        output_dir,
        "--annotation-key",
        annotation_key,
        "--outprefix",
        outprefix,
    ]
    if normalize:
        args.append("--normalize")
    if gene_use_file:
        args.extend(["--gene-use", gene_use_file])
    if ntopics:
        args.extend(["--ntopics"] + [str(n) for n in ntopics])
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    # None has no CLI spelling: omitting the flag leaves the worker to resolve STRIDE's default.
    if st_scale_factor is not None:
        args.extend(["--st-scale-factor", str(st_scale_factor)])
    if sc_scale_factor is not None:
        args.extend(["--sc-scale-factor", str(sc_scale_factor)])

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
