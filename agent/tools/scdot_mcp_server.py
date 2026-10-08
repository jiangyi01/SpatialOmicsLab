#!/usr/bin/env python3
"""scDOT-based cell-spot mapping MCP wrapper for SpatialOmicsLab (entropic OT on a cosine cost)."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "scdot"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SCDOT",
    "/opt/conda/envs/scdot/bin/python",
    "/workspace/epic-fermat/agent/tools/scdot_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def scdot_map_cells_to_spots(
    sc_h5ad: str,
    st_h5ad: str,
    output_dir: str,
    annotation_key: str = "cell_type",
    spatial_key: str = "spatial",
    n_hvg: int = 2000,
    ot_gamma: float = 1.0,
    ot_maxiters: int = 1000,
    ot_eps: float = 1e-6,
    device: str = "cpu",
    seed: int = 0,
    hvg_flavor: str = "seurat_v3",
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Map single cells to spatial spots by entropic optimal transport on a cosine cost, using
    scDOT's OptimalTransportLayer only.

    WHAT RUNS. Upstream scDOT trains two modules jointly -- an NNLS deconvolution of each spot
    against per-cell-type marker profiles, and a differentiable OT layer whose cost matrix is
    tuned by Adam so the transport plan agrees with the NNLS proportions. This tool runs ONLY the
    OT half, once, on the fixed cosine distance between log-normalised spot and cell profiles: no
    markers, no NNLS, no training. That is plain Sinkhorn (entropy-regularised OT), and the payload
    says so in ``params.method``; ``params.used_fallback`` is False because nothing else ever ran.
    Do not report the numbers as "scDOT" results.

    OUTPUTS. ``scdot_coupling.csv`` (spots x cells transport plan, each cell's column summing to
    1), ``scdot_deconvolution.csv`` (per-spot cell-type proportions = coupling @ one-hot labels,
    row-normalised) and ``scdot_spatial_annotated.h5ad`` with ``obsm['deconvolution']``. The
    payload's ``params.sinkhorn_marginal_error`` is the Sinkhorn plan's largest deviation from its
    marginals (each spot 1/n_spots, each cell 1/n_cells, summed in float64) before its columns are
    rescaled to 1 for the CSV, and ``params.sinkhorn_converged`` is True only when that is at or
    below `ot_eps` itself -- the absolute tolerance upstream's Sinkhorn loop stops on. A run that
    is not converged carries a warning saying why: either `ot_maxiters` ran out, or `ot_eps` is
    below what a float32 plan resolves (about 5e-7 times the larger marginal, 1/min(n_spots,
    n_cells)), which no number of iterations reaches.

    MEMORY. The transport plan is a dense n_spots x n_cells matrix, intrinsic to the method. The
    worker holds at most about 12 bytes per spot-cell pair (float64 cost + its float32 copy, then
    float32 cost + two Sinkhorn temporaries), checks that figure against the memory it can still
    allocate (MemAvailable and the room under a cgroup limit, page cache counted as reclaimable)
    before building anything that size, and fails with the numbers if it does not fit; it never
    subsamples. Everything else stays sparse or n_cells x n_types.

    BACKGROUND SPOTS. When the spatial file's ``obs['in_tissue']`` marks background spots (0), only
    the in-tissue spots are mapped: every output row is an in-tissue spot, ``data.n_spots`` counts
    them, ``data.n_spots_supplied`` counts the file, and ``params.in_tissue_filter`` plus a warning
    say how many were left out.

    REFUSALS. A spot or reference cell with no counts on the analysed panel has no cosine
    distance (0/0) and would turn the whole plan into NaN; the run stops and names them. A plan
    that comes back non-finite (a very large `ot_gamma` underflows the kernel) is refused too.

    The reference is mapped on a subset of BOTH panels you supply: it filters and HVG-selects each
    modality separately (single cell: drop genes under 10 counts, drop genes named 'MT-*' --
    upstream's human mitochondrial prefix -- and keep the top `n_hvg` by `hvg_flavor`; spatial:
    drop genes detected in fewer than 3 spots), then keeps only the genes both sides retained. The
    payload reports the supplied panels as n_genes (spatial) and n_genes_sc (single-cell), and the
    shared analysed panel as n_genes_used and n_genes_sc_used -- equal by construction -- and
    carries a warning naming each cut.

    Parameters
    ----------
    sc_h5ad:
        Path to scRNA-seq AnnData (.h5ad) with cell-type labels in obs[annotation_key].
    st_h5ad:
        Path to spatial AnnData (.h5ad) with expression in X. Spatial coordinates are NOT read:
        the mapping compares expression profiles only, so no obsm key is required.
    output_dir:
        Directory where the outputs will be saved.
    annotation_key:
        obs column in the scRNA data containing cell-type labels. A NaN/empty label is refused
        unless `drop_unlabeled` is True.
    spatial_key:
        Accepted for compatibility only. The method never reads coordinates, so this value has no
        effect; every run lists it under ``params.ignored`` with a warning.
    n_hvg:
        Number of highly variable genes to select on the single-cell reference. The
        analysed panel is the intersection of that selection with the spatial slide's
        surviving genes, so it is normally smaller than n_hvg -- see n_genes_used in
        the payload.
    ot_gamma:
        Regularization parameter for entropy-regularized OT (higher = sparser).
    ot_maxiters:
        Maximum iterations for the Sinkhorn algorithm.
    ot_eps:
        Convergence tolerance for Sinkhorn: the absolute marginal error at which it stops, and the
        one ``params.sinkhorn_converged`` is judged against (see OUTPUTS).
    device:
        Compute device: "cpu", "gpu"/"cuda", "cuda:N" for a particular card, or "auto" to follow
        the hardware. A GPU request degrades to the CPU when no CUDA device is present; the
        payload's ``params.device`` is the device the solve actually ran on.
    seed:
        Random seed for reproducibility (Sinkhorn on a fixed cost is deterministic; the seed is
        set for completeness).
    hvg_flavor:
        scanpy flavour for the reference HVG selection: "seurat_v3" (ranks raw counts; needs the
        scikit-misc package), "seurat" or "cell_ranger" (rank the log-normalised matrix). The
        chosen flavour is the only one that runs: if its dependency is missing the run fails
        naming the package instead of silently switching flavour, as this tool once did.
    drop_unlabeled:
        Drop reference cells whose label is NaN/empty instead of refusing them (default False).
        The payload reports how many were dropped in ``params.n_cells_sc_unlabeled_dropped`` and
        ``data.n_cells_sc_used``.
    """
    args = [
        "--sc-h5ad",
        sc_h5ad,
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--annotation-key",
        annotation_key,
        "--spatial-key",
        spatial_key,
        "--n-hvg",
        str(n_hvg),
        "--hvg-flavor",
        hvg_flavor,
        "--ot-gamma",
        str(ot_gamma),
        "--ot-maxiters",
        str(ot_maxiters),
        "--ot-eps",
        str(ot_eps),
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
