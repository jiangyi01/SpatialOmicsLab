#!/usr/bin/env python3
"""NCEM-style neighbourhood Ridge cell-communication MCP wrapper for SpatialOmicsLab.

The tool keeps the NCEM name because callers address it as ``ncem_cell_communication``, but what runs
is a closed-form sklearn Ridge regression on neighbourhood composition written in the worker; the
``ncem`` package is never imported. The payload says so in ``params.method``.
"""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "ncem"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "NCEM",
    "/opt/conda/envs/ncem/bin/python",
    "/workspace/epic-fermat/agent/tools/ncem_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def ncem_cell_communication(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    annotation_key: str = "cell_type",
    n_neighbors: int = 10,
    max_epochs: int = 50,
    learning_rate: float = 1e-3,
    n_hvg: int = 2000,
    model_type: str = "linear",
    seed: int = 0,
    drop_unlabeled: bool = False,
    use_raw_counts: bool = False,
    coords_key: str = "spatial",
    dims: int = 2,
    section_key: str | None = None,
) -> dict[str, Any]:
    """
    Estimate how neighbouring cell types shift gene expression with an NCEM-style linear model.

    What runs is NOT the ncem package (installed, never imported): for every spot, the fraction of
    each cell type among its ``n_neighbors`` nearest spatial neighbours is computed, and an sklearn
    ``Ridge(alpha=1.0)`` regression of the normalised, log1p-transformed expression of the top
    ``n_hvg`` highly variable genes on ``[one-hot own cell type | neighbourhood composition]`` is
    solved in closed form. The neighbourhood coefficients form a gene x cell-type communication
    matrix; because neighbour fractions sum to 1, each gene's effects are relative to the average
    neighbour type. ``params.method`` names this model and ``params.used_fallback`` is False (it is
    the only implementation).

    Spots with ``obs['in_tissue'] == 0`` (off-tissue background in Space Ranger / CELLxGENE exports)
    are left out before anything is computed, and the count is reported (``params.in_tissue_filter``).
    Nothing is subsampled. The regression targets are a dense spots x HVG matrix (intrinsic to
    sklearn's Ridge); the worker estimates its size first and refuses with the numbers if it cannot fit
    (available memory is MemAvailable, or the room left under a container's cgroup memory limit when
    that is smaller).

    Outputs (in ``output_dir``): ``ncem_communication_matrix.csv`` (genes x neighbour cell types,
    signed effects), ``ncem_communication_strength.csv`` (mean |effect| per neighbour type),
    ``ncem_neighborhood_composition.csv`` (spots x cell types: fractions of each spot's NEIGHBOURS,
    not of the spot) and ``ncem_top_genes_per_type.csv``.

    Two-dimensional by design: the neighbourhood is built on two columns, so dims=3 is refused, and a
    file holding two or more sections is refused unless section_key names them (a 2D run would overlay
    them). With section_key the model is fitted per section: the four files per section in
    ``output_dir/section_<label>/`` (the gene x cell-type matrix is per section only), and one long
    ``ncem_communication_strength.csv`` at the top level with a leading ``section`` column.
    ``params.mode`` is "per-section-2d" (or "2d" for one plane), ``params.sections`` lists the sections
    and ``params.frame`` the coordinates read; the same is written to sog_run_provenance.json.

    Parameters
    ----------
    st_h5ad:
        Path to spatial AnnData (.h5ad) with raw counts in X, spatial coordinates in
        obsm[spatial_key] and cell type annotations in obs[annotation_key]. An X with negative or
        NaN/inf values (scaled data, which normalisation + log1p turns into NaN) is refused with the
        count before anything is normalised, naming adata.raw and use_raw_counts when adata.raw exists;
        X is never swapped for adata.raw unless use_raw_counts=True. A non-negative non-integer X
        (log-normalised) runs with a warning that it was transformed twice.
    output_dir:
        Directory where the outputs will be saved.
    spatial_key:
        obsm key for spatial coordinates (two columns; coords_key, when not 'spatial', takes its place).
    annotation_key:
        obs column containing cell type annotations; at least two cell types are needed.
    n_neighbors:
        Number of nearest spatial neighbours (the spot itself excluded) whose cell types form each
        spot's neighbourhood composition.
    max_epochs:
        Accepted for compatibility and IGNORED: the Ridge fit is closed-form, nothing is trained.
        Listed under ``params.ignored``.
    learning_rate:
        Accepted for compatibility and IGNORED: the Ridge fit is closed-form, nothing is trained.
        Listed under ``params.ignored``.
    n_hvg:
        Number of highly variable genes (Seurat flavour, on log-normalised data) used as regression
        targets.
    model_type:
        Only 'linear' is implemented (the neighbourhood Ridge model above).
    seed:
        Accepted and IGNORED: nothing in the method is random (KD-tree neighbours, HVG ranking and the
        closed-form solve are deterministic). Listed under ``params.ignored``.
    drop_unlabeled:
        A spot with no label (NaN/empty) in obs[annotation_key] stops the run with the count by
        default. Pass True to leave those spots out instead; the number dropped is reported in
        ``data.n_cells_dropped_unlabeled``, ``params`` and the analysis text.
    use_raw_counts:
        Normalise adata.raw.X instead of X (default False reads X). Use it for a CELLxGENE-style h5ad
        whose X is processed and whose counts are in adata.raw; refused when there is no adata.raw or
        it does not hold counts. ``params.expression_source`` says which matrix was used.
    coords_key, dims, section_key:
        coords_key: the obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
        'spatial_3d_aligned'). dims: 2 or 3; 3 builds the graph in the aligned frame in micrometres and
        needs a frame with recorded units and a measured or registered z. section_key: the obs column
        naming sections; required for a 2D run on a multi-section file (the run is per section) and for
        the cross-section edge count of a 3D run.
        For NCEM: dims=3 is refused ("NCEM builds its neighbourhood in two dimensions; run per section
        with `dims=2, section_key=<column>`.").
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
        "--max-epochs",
        str(max_epochs),
        "--learning-rate",
        str(learning_rate),
        "--n-hvg",
        str(n_hvg),
        "--model-type",
        model_type,
        "--seed",
        str(seed),
    ]
    if coords_key != "spatial":
        args += ["--coords-key", coords_key]  # the worker's own flag for it; it wins over --spatial-key
    if section_key:
        args += ["--section-key", section_key]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    if use_raw_counts:
        args.append("--use-raw-counts")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
