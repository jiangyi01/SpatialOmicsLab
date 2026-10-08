#!/usr/bin/env python3
"""DestVI (scvi-tools) MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "destvi"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "DESTVI",
    "/opt/conda/envs/destvi_env/bin/python",
    "/workspace/epic-fermat/agent/tools/destvi_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_destvi(
    sc_h5ad_path: str,
    spatial_h5ad_path: str,
    output_dir: str = default_output_dir(),
    cell_type_key: str = "cell_type",
    batch_key: str = "batch",
    max_epochs_sc: int = 100,
    max_epochs_st: int = 2000,
    n_top_genes: int = 2000,
    hvg_flavor: str = "seurat",
    drop_unlabeled: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run DestVI deconvolution pipeline via scvi-tools.

    Trains a CondSCVI model on a scRNA-seq reference and then fits a DestVI
    model on spatial transcriptomics data to infer cell-type proportions
    per spot. Returns proportions as CSV and annotated spatial h5ad.

    CondSCVI is always trained without a batch covariate (DestVI.from_rna_model cannot load a
    batch-aware decoder), so ``batch_key`` has no effect. The payload names the HVG flavour that ran
    (``params.hvg_flavor``) and the method (``params.method``); reference cells with no label stop
    the run unless ``drop_unlabeled=True``, and label categories no cell carries are not fitted.
    Spots with no counts in the genes the model uses cannot be fitted (DestVI's likelihood turns NaN
    on them and training stops), so they are left out of the fit; their proportions rows are all
    zeros, not a composition. They are counted in ``data.n_spots_no_counts_in_model_genes`` (the
    spots fitted in ``data.n_spots_fitted``) and named in ``warnings``.

    Spatial spots with ``obs['in_tissue'] == 0`` (background outside the tissue) are left out and
    reported (``params.in_tissue_filter`` and a warning); ``data.n_spots_supplied`` is the slide
    supplied and ``data.n_spots`` the in-tissue spots deconvolved, which are the rows of the outputs.

    CondSCVI and DestVI fit their input as counts. An input whose matrix holds negative or NaN values
    (a scaled X) is refused, saying whether its adata.raw holds counts; a fractional matrix (e.g.
    log-normalised) runs with a warning. The matrix each input was read from is reported
    (``params.expression_source`` / ``x_matrix_kind`` for the slide, ``params.reference_expression_source``
    / ``reference_x_matrix_kind`` for the reference).

    Parameters
    ----------
    sc_h5ad_path:
        Path to single-cell reference AnnData (.h5ad) with cell-type labels.
    spatial_h5ad_path:
        Path to spatial transcriptomics AnnData (.h5ad), e.g. 10x Visium.
    output_dir:
        Directory where all DestVI outputs will be saved.
    cell_type_key:
        obs column in scRNA AnnData containing cell-type labels.
    batch_key:
        Accepted for compatibility only; it never reaches the model. CondSCVI is always trained
        without a batch covariate because DestVI.from_rna_model cannot load a batch-aware decoder,
        so no batch correction is applied whether or not the column exists. When the column exists
        in the reference (or another value is passed) it is listed under ``params.ignored``.
    max_epochs_sc:
        Max training epochs for CondSCVI on single-cell data.
    max_epochs_st:
        Max training epochs for DestVI on spatial data.
    n_top_genes:
        Number of highly variable genes to select for training.
    hvg_flavor:
        scanpy flavour that ranks the reference genes: 'seurat' (default; log-normalised
        dispersion, what every earlier run in destvi_env used), 'cell_ranger', or 'seurat_v3'
        (raw counts; needs scikit-misc, which destvi_env does not ship -- a missing package stops
        the run rather than switching flavour). CondSCVI always trains on raw counts.
    drop_unlabeled:
        Default False: reference cells whose cell_type_key label is missing (NaN, empty, 'nan')
        stop the run with their count. True leaves them out and reports how many
        (``params.n_reference_cells_dropped_unlabeled``, ``data.n_cells_sc_used``).
    use_raw_counts:
        Default False reads X. True reads the counts from adata.raw of each input that carries one
        (an input without one keeps X, with a warning; when neither has one the parameter is listed
        under ``params.ignored``). Use it for a CELLxGENE-style h5ad whose X is processed and whose
        counts are in adata.raw.
    """
    sc_path = str(Path(sc_h5ad_path).expanduser())
    spatial_path = str(Path(spatial_h5ad_path).expanduser())
    out_dir = str(Path(output_dir).expanduser())

    args = [
        "--sc-h5ad",
        sc_path,
        "--spatial-h5ad",
        spatial_path,
        "--output-dir",
        out_dir,
        "--cell-type-key",
        cell_type_key,
        "--batch-key",
        # A None from a direct Python caller would reach subprocess as a non-string argv item.
        batch_key if batch_key is not None else "",
        "--max-epochs-sc",
        str(max_epochs_sc),
        "--max-epochs-st",
        str(max_epochs_st),
        "--n-top-genes",
        str(n_top_genes),
        "--hvg-flavor",
        hvg_flavor,
    ]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    if use_raw_counts:
        args.append("--use-raw-counts")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
