#!/usr/bin/env python3
"""SpatialScope MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "spatialscope"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPATIALSCOPE",
    "/opt/conda/envs/spatialscope_env/bin/python",
    "/workspace/epic-fermat/agent/tools/spatialscope_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_spatialscope(
    sc_h5ad_path: str,
    spatial_h5ad_path: str | None = None,
    output_dir: str = default_output_dir("spatialscope_output"),
    cell_type_key: str | None = "cell_type",
    UMI_min_sigma: int | None = 300,
    n_cpus: int | None = None,
    allow_nnls_fallback: bool | None = False,
    input_scale: str | None = "auto",
    drop_unlabeled: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run SpatialScope's CPU-runnable Cell-Type Identification (CTI) pipeline
    for spot-level spatial deconvolution.

    Internally calls the upstream WarmStart (`create_RCTD` + `run_RCTD`)
    flow with `doublet_mode='full'`, identical to the official benchmark
    runner. The output is a per-spot cell-type proportion matrix
    normalized to row-sum 1; the small negative weights the OSQP solver
    leaves (it holds w >= 0 only to its tolerance) are set to 0 first, as
    the benchmark runner does, and counted in
    `params.negative_weights_clipped`. Spots whose UMI total over the shared
    genes is below RCTD's scoring floor UMI_min = min(100, UMI_min_sigma)
    (published as `params.UMI_min`) are not scored: they are counted in
    `data.n_spots_scored` and in a warning, absent from the proportions CSV
    and left empty in the written h5ad. Background spots (spatial
    obs['in_tissue'] == 0, as CELLxGENE Visium exports carry them) are left
    out before the run and reported in `params.in_tissue_filter` and
    `warnings`; `data.n_spots` counts the in-tissue spots. Duplicated
    barcodes or gene names are renamed to be unique and reported
    (`params.n_cells_renamed` / `n_genes_renamed`, `_sc` for the reference).
    A 3-column obsm['spatial'] is accepted (CTI does not use coordinates).

    Counts: CTI (RCTD) reads every value as a UMI count. A matrix with
    negative or NaN values (scaled / z-scored, as some CELLxGENE exports
    hold) is refused, and the error says whether adata.raw holds counts;
    `use_raw_counts=True` runs on adata.raw.X. log1p input is accepted by
    design (see `input_scale`), and a non-integer matrix that is taken as
    counts is named in `warnings`. `params.expression_source` /
    `params.reference_expression_source` say which matrix each input ran on.

    Important: SpatialScope's diffusion-based Stage-2 decomposition is
    GPU-only and is NOT exercised here. The agent's CPU-only environment
    runs Stage 1 (CTI) only — the same scope as upstream's published
    deconvolution benchmark.

    If the SpatialScope source repository is not found, its `utils_pyRCTD`
    module (or one of that module's own imports) cannot be imported, or its
    extdata/ likelihood tables are missing, this tool fails LOUDLY with
    status='dep_missing' rather than silently substituting an inferior NNLS
    approximation. Set `allow_nnls_fallback=True` to opt into the degraded
    path explicitly; outputs in that case are clearly labelled
    'SpatialScope (NNLS fallback)', and the payload's `params.method` and
    `params.used_fallback` say which method ran.

    Memory: upstream `utils_pyRCTD` takes dense pandas frames (shared genes x
    spots and shared genes x reference cells), so those two dense copies are
    intrinsic to the method. The worker prices them before allocating and
    refuses, with the numbers, when they do not fit in the available memory
    (MemAvailable, capped by the room under a cgroup memory limit).
    Nothing is subsampled: every spot and every labelled reference cell is
    used (upstream's 10000-cells-per-type random draw is not applied).

    Parameters
    ----------
    sc_h5ad_path:
        Path to single-cell reference AnnData (.h5ad) with cell-type labels. REQUIRED.
    spatial_h5ad_path:
        Path to spatial transcriptomics AnnData (.h5ad), e.g. 10x Visium.
        Required at runtime — if omitted, the wrapper returns a clean error
        rather than letting Pydantic raise a validation error in the MCP framework.
        Must have obsm['spatial'] coordinates.
    output_dir:
        Directory where all SpatialScope outputs will be saved.
    cell_type_key:
        obs column in scRNA AnnData containing cell-type labels.
    UMI_min_sigma:
        The UMI total a spot needs to take part in fitting RCTD's noise
        parameter sigma (paper default: 300; lower it on low-UMI platforms,
        where no spot may reach 300). It is not the scoring floor: a spot is
        scored when its UMI total is at least UMI_min = min(100,
        UMI_min_sigma), published as `params.UMI_min`.
    n_cpus:
        Number of ray workers. Default leaves 2 CPUs of headroom on whatever this
        process is actually allowed (affinity mask / cgroup quota, not the machine's
        core count). Each worker's BLAS is pinned to one thread, so the workers do not
        oversubscribe the cores.
    allow_nnls_fallback:
        If True, fall back to scipy.optimize.nnls per-spot regression
        when SpatialScope cannot run -- the source or its likelihood tables
        are missing, `utils_pyRCTD` cannot be imported, or the CTI run itself
        raises (the reason is in `warnings`). Off by default — enabling this
        changes the reported method name to 'SpatialScope (NNLS fallback)'
        and sets `params.used_fallback` so downstream consumers can tell the
        difference. Recommended: leave off.
    input_scale:
        Scale of `X` in both h5ads. 'auto' (default) follows upstream
        SpatialScope's rule -- a matrix whose maximum is below 30 is taken as
        log1p and un-logged with expm1 (the reference is then re-normalised
        with `normalize_total`) -- except that a matrix whose values are all
        whole numbers is always kept as counts. 'counts' never un-logs;
        'log1p' always un-logs and refuses a matrix whose maximum is 30 or
        more. A non-integer matrix taken as counts (maximum 30 or more under
        'auto', or on request under 'counts') runs with a warning; a matrix
        with negative or NaN values is refused under every setting. The
        decision for each matrix is reported in `params` (`spatial_scale`,
        `reference_scale`, `spatial_unlogged`, `reference_unlogged`,
        `spatial_matrix_kind`, `reference_matrix_kind`) and in `warnings`.
    drop_unlabeled:
        Default False: reference cells with a missing cell-type label
        (NaN/empty) stop the run with an error naming the count. True leaves
        them out and reports how many were dropped (`params.n_unlabeled_dropped`).
    use_raw_counts:
        Default False. True runs on adata.raw.X instead of X: the spatial
        h5ad must carry an adata.raw holding counts (refused otherwise), and
        the reference is read from its adata.raw when it has one (a warning
        says when it has none). For CELLxGENE exports whose X is
        log-normalised or scaled.
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
    if UMI_min_sigma is not None:
        args += ["--umi-min-sigma", str(UMI_min_sigma)]
    if n_cpus is not None:
        args += ["--n-cpus", str(int(n_cpus))]
    if allow_nnls_fallback:
        args.append("--allow-nnls-fallback")
    if input_scale is not None:
        args += ["--input-scale", str(input_scale)]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    if use_raw_counts:
        args.append("--use-raw-counts")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
