#!/usr/bin/env python3
"""CellDART MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "celldart"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "CELLDART",
    "/opt/conda/envs/celldart_env/bin/python",
    "/workspace/epic-fermat/agent/tools/celldart_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_celldart(
    sc_h5ad_path: str,
    spatial_h5ad_path: str | None = None,
    output_dir: str = default_output_dir("celldart_output"),
    cell_type_key: str | None = "cell_type",
    num_markers: int | None = 20,
    nmix: int | None = 20,
    npseudo: int | None = 20000,
    alpha: float | None = 0.6,
    alpha_lr: int | None = 5,
    emb_dim: int | None = 64,
    batch_size: int | None = 64,
    n_iterations: int | None = 3000,
    init_train_epoch: int | None = 10,
    seed_num: int | None = 0,
    gpu: bool | None = False,
    use_raw_counts: bool | None = False,
) -> dict[str, Any]:
    """
    Run CellDART domain-adaptation deconvolution on spatial transcriptomics data.

    CellDART uses adversarial domain adaptation to transfer cell-type labels
    from a scRNA-seq reference to spatial spots. Wraps the high-level
    pred_cellf_celldart() pipeline. npseudo, alpha, alpha_lr, emb_dim, n_iterations and
    init_train_epoch default to the values of the CellDART tutorial and of pred_cellf_celldart
    itself; nmix=20 and batch_size=64 are this wrapper's own choices, not the published recipe
    (the tutorial uses nmix=8 and batch_size=512; pred_cellf_celldart's defaults are nmix=10 and
    batch_size=512).

    Reference cells whose cell_type_key value is unusable -- missing, the literal string
    'nan', or empty -- are dropped before training. The payload reports the supplied count
    as n_cells_sc and the trained-on count as n_cells_sc_used, and carries a warning naming
    the cut when it fires.

    CellDART does not train on the full gene panels. It ranks the reference with a Wilcoxon
    test, keeps the union of each cell type's top num_markers genes that are also on the
    spatial panel, and cuts BOTH objects to that marker panel before mixing pseudo-spots and
    training. The payload reports the supplied panels as n_genes_sc / n_genes, the ranked
    markers as n_marker_genes and the genes trained on as n_genes_used; the panel itself is
    written to celldart_marker_genes.csv (gene, the cell types it marks, used_for_training).

    Spatial spots with obs['in_tissue'] == 0 (background outside the tissue) are left out and
    reported (params.in_tissue_filter and a warning); n_spots is the slide supplied and
    n_spots_used the in-tissue spots deconvolved, which are the rows of the outputs.

    CellDART normalises both inputs as counts (normalize_total, then log1p). An input whose
    matrix holds negative or NaN values (a scaled X) is refused, because its fractions would come
    out NaN; a fractional matrix (e.g. log-normalised) runs and is named in a warning. Each
    message says whether adata.raw holds counts. params.counts_source says which matrix each
    input was read from. Cell-type labels that become one name once '/' and ' ' turn into '_'
    ('T cell' and 'T_cell') are refused before training.

    Parameters
    ----------
    sc_h5ad_path:
        Path to single-cell reference AnnData (.h5ad) with cell-type labels. REQUIRED.
    spatial_h5ad_path:
        Path to spatial transcriptomics AnnData (.h5ad). Required at runtime —
        if omitted, the wrapper returns a clean error rather than letting
        Pydantic raise a validation error in the MCP framework.
    output_dir:
        Directory where all CellDART outputs will be saved.
    cell_type_key:
        obs column in scRNA AnnData containing cell-type labels.
    num_markers:
        Top Wilcoxon marker genes kept per cell type. CellDART trains on the union of these
        that are on the spatial panel (at most num_markers x the number of cell types genes),
        reported as n_genes_used.
    nmix:
        Number of single cells mixed per synthetic pseudo-spot. Default 20 is this wrapper's choice;
        the CellDART tutorial uses 8 and pred_cellf_celldart defaults to 10.
    npseudo:
        Total number of pseudo-spots synthesized for training (paper: 20000).
    alpha:
        Adversarial domain-classifier loss weight (paper: 0.6).
    alpha_lr:
        Learning-rate multiplier on the domain classifier (paper: 5).
    emb_dim:
        Embedding dimension of the shared feature extractor.
    batch_size:
        Mini-batch size for adversarial training. Default 64 is this wrapper's choice; the CellDART
        tutorial and pred_cellf_celldart use 512.
    n_iterations:
        Adversarial training iterations (paper: 3000). Lower values
        (e.g. 500) produce systematically degraded predictions.
    init_train_epoch:
        Pre-training epochs on source-only classifier (paper: 10).
    seed_num:
        Seed for CellDART's pseudo-spot mixing (which cells go into each pseudo-spot). CellDART
        does not seed the network's initial weights or its minibatch draws, and it orders the
        marker genes through a Python set, so two runs with the same seed_num can differ.
    gpu:
        Whether to use GPU for training (default False = CPU only).
    use_raw_counts:
        Read counts from adata.raw instead of X, for each input that carries an adata.raw (an input
        without one keeps X, with a warning); default False reads X. Use it for a CELLxGENE-style
        h5ad whose X is processed and whose counts are in adata.raw.
    """
    if spatial_h5ad_path is None:
        return {
            "status": "error",
            "tool": TOOL_NAME,
            "error": "spatial_h5ad_path is required: provide the path to the spatial transcriptomics AnnData (.h5ad).",
        }

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
    ]
    if cell_type_key is not None:
        args += ["--cell-type-key", cell_type_key]
    if num_markers is not None:
        args += ["--num-markers", str(num_markers)]
    if nmix is not None:
        args += ["--nmix", str(nmix)]
    if npseudo is not None:
        args += ["--npseudo", str(npseudo)]
    if alpha is not None:
        args += ["--alpha", str(alpha)]
    if alpha_lr is not None:
        args += ["--alpha-lr", str(alpha_lr)]
    if emb_dim is not None:
        args += ["--emb-dim", str(emb_dim)]
    if batch_size is not None:
        args += ["--batch-size", str(batch_size)]
    if n_iterations is not None:
        args += ["--n-iterations", str(n_iterations)]
    if init_train_epoch is not None:
        args += ["--init-train-epoch", str(init_train_epoch)]
    if seed_num is not None:
        args += ["--seed-num", str(seed_num)]
    if gpu:
        args.append("--gpu")
    if use_raw_counts:
        args.append("--use-raw-counts")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
