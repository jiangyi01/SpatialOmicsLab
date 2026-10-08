#!/usr/bin/env python3
"""STAGE MCP wrapper for SpatialOmicsLab."""

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "stage"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "STAGE",
    "/opt/conda/envs/stage_env/bin/python",
    "/workspace/epic-fermat/agent/tools/stage_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def stage_run(
    st_h5ad: str,
    output_dir: str,
    data_type: str = "10x",
    experiment: str = "generation",
    coord_sf: float | None = None,
    down_ratio: float | None = None,
    train_epoch: int = 10000,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    w_recon: float | None = None,
    w_w: float | None = None,
    w_l1: float | None = None,
    use_gpu: bool = False,
    device: str = "",
    seed: int = 1234,
    sec_name: str | None = None,
    select_section: str | None = None,
    gap: float | None = None,
    relu: bool = True,
    hvg_flavor: str = "seurat_v3",
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run upstream STAGE (``STAGE.STAGE``) once on a spatial AnnData.

    ``data_type`` / ``experiment`` pairs STAGE accepts: "10x" with "generation" or "recovery",
    "ST_KTH" with "generation", "Slide-seq" with "3d_model". Any other pair is refused before any
    work is done.

    Coordinates (``obsm['coord']``): used as supplied when present. Otherwise, for 10x and ST_KTH the
    worker builds them from ``obs[['array_col', 'array_row']]``, the layout the STAGE tutorials use.
    ``generation`` places each new spot a fixed fraction of an array-grid step from a measured one,
    so it needs array-grid units: pixel coordinates (``obsm['spatial']``) are refused rather than
    turned into copies of the measured spots. ``recovery`` and ``3d_model`` also accept
    ``obsm['spatial']`` (three columns x, y, z for ``3d_model``). Generation runs under pandas
    Copy-on-Write (upstream's coordinate shifts otherwise alias one buffer under pandas 2 and
    generate nothing), and a run that generated no new positions is reported as an error. The
    installed STAGE 1.0.1 cannot run ``recovery`` (its model.py never sets ``X_dim`` for that
    branch); such a call returns an error that says so, and no other method is substituted.

    ``coord_sf`` left unset: the tutorial value (10x 77, ST_KTH 35, Slide-seq 6000) when it scales
    the largest coordinate to about 1, otherwise the largest |x|,|y|. ``params.coord_sf`` and
    ``params.coord_sf_source`` say which.

    Genes: unless ``var['highly_variable']`` is present, the worker selects highly variable genes
    with ``hvg_flavor`` ("seurat_v3" on counts, as in the tutorials; "seurat" or "cell_ranger" run
    on normalised log data) -- the top 5000 for 10x generation and Slide-seq, 3000 otherwise -- then
    applies normalize_total and log1p. STAGE models only those genes (``data.n_genes_used``).
    "seurat_v3" needs scikit-misc; when it cannot run the call stops and says so instead of
    switching flavour.

    Spots with ``obs['in_tissue'] == 0`` (background outside the tissue, which CELLxGENE Visium
    exports carry) are left out before anything runs: STAGE neither trains on them nor generates
    positions around them. ``params.in_tissue_filter`` and a warning say how many; ``data.n_spots``
    counts the measured spots modelled and ``data.n_spots_input`` those supplied.

    Matrix: when the worker selects the genes it also normalises X as counts, so a negative or
    non-finite X (scaled data) is refused, naming ``use_raw_counts`` when ``adata.raw`` holds counts,
    and a non-negative non-integer X runs with a warning. ``use_raw_counts=True`` reads
    ``adata.raw.X`` and runs the tutorial preprocessing on it. An input that carries its own
    ``var['highly_variable']`` is used as supplied (normalised data is accepted there).
    ``params.expression_source`` says which matrix was read.

    Memory: STAGE holds its training matrix dense (in-tissue spots x modelled genes) and decodes
    every output position into a dense matrix; that is upstream's design. A run whose estimated peak
    exceeds the memory available (under a cgroup limit, the room left once what the container
    already uses is counted) is refused with the numbers before training; no in-tissue spot is ever
    dropped.

    ``down_ratio`` applies to recovery only; ``sec_name``, ``select_section`` and ``gap`` to
    3d_model only. Passed to another experiment they are listed in ``params.ignored``. For 3d_model
    a ``train_epoch`` left at 10000 is read as unset and the 3D tutorial's 1000 is used.

    Outputs go to ``<output_dir>/stage_run/``: ``adata_stage.h5ad`` (plus ``adata_sample.h5ad``
    for recovery, and ``adata_simu.h5ad`` and ``adata_all.h5ad`` for 3d_model) and STAGE's own
    files (encoder/decoder weights, original/generated/recovered h5ad). ``output_files`` lists only
    files this call wrote. STAGE does not cluster; ``adata_stage.h5ad`` carries no domain labels.

    Set ``device`` to pick the compute device explicitly -- "cpu", "gpu"/"cuda", or "cuda:N" for a
    particular card on a multi-GPU host. It overrides ``use_gpu``, which can only say "some GPU".

    ``use_raw_counts`` (default False): read ``adata.raw.X`` instead of X, for an h5ad whose X is
    normalised or scaled and whose counts sit in ``adata.raw``; refused when there is no
    ``adata.raw`` or it does not hold counts.
    """
    st_h5ad = os.path.abspath(st_h5ad)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--data-type",
        data_type,
        "--experiment",
        experiment,
        "--train-epoch",
        str(train_epoch),
        "--seed",
        str(seed),
        "--hvg-flavor",
        hvg_flavor,
    ]
    if coord_sf is not None:
        args += ["--coord-sf", str(coord_sf)]
    if down_ratio is not None:
        args += ["--down-ratio", str(down_ratio)]
    if batch_size is not None:
        args += ["--batch-size", str(batch_size)]
    if learning_rate is not None:
        args += ["--learning-rate", str(learning_rate)]
    if w_recon is not None:
        args += ["--w-recon", str(w_recon)]
    if w_w is not None:
        args += ["--w-w", str(w_w)]
    if w_l1 is not None:
        args += ["--w-l1", str(w_l1)]
    if sec_name is not None:
        args += ["--sec-name", sec_name]
    if select_section is not None:
        args += ["--select-section", select_section]
    if gap is not None:
        args += ["--gap", str(gap)]
    if use_gpu:
        args.append("--use-gpu")
    if device:
        args += ["--device", device]
    if not relu:
        args.append("--no-relu")
    if use_raw_counts:
        args.append("--use-raw-counts")

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
