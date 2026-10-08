#!/usr/bin/env python3
"""SpatialDE spatially variable genes MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_json

TOOL_NAME = "spatialde"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPATIALDE",
    "/opt/conda/envs/SpatialDE/bin/python",
    "/workspace/epic-fermat/agent/tools/spatialde_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def spatialde_run_svg(
    input_mode: str,
    output_dir: str,
    counts_h5: str | None = None,
    spatial_dir: str | None = None,
    h5ad_path: str | None = None,
    spatial_key: str = "spatial",
    layer: str | None = None,
    matrix_path: str | None = None,
    coords_path: str | None = None,
    matrix_orientation: str = "cells_by_genes",
    id_col: str = "barcode",
    x_col: str = "x",
    y_col: str = "y",
    regress_formula: str = "~np.log(total_counts)",
    top_k_genes: int = 20,
    hvg_top_n: int = 2000,
    hvg_flavor: str = "variance",
    run_aeh: bool = False,
    aeh_C: int = 8,
    aeh_l: float = 1000.0,
    random_seed: int = 0,
    allow_hvg_fallback: bool = False,
    round_counts: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Run SpatialDE (v1; SpatialDE 1.1.3 in the tool env) to identify spatially variable genes.

    The worker calls ``SpatialDE.anndata.spatialde_test`` -- the Svensson et al. 2018 Gaussian-process
    likelihood-ratio test (NaiveDE variance stabilisation + regress-out + ``base.run``) -- and names
    the version that ran under ``params.method`` / ``params.spatialde_version``. Before the test:
      - Gene prefilter: keeps the top ``hvg_top_n`` genes ranked by ``hvg_flavor`` (default 2000).
        Pass ``hvg_top_n=0`` to test every gene. The default ``hvg_flavor`` is ``variance``: that is
        the ranking every default run has actually used (the old ``seurat_v3`` default could never run
        in the SpatialDE env and fell back to it silently), so default results are unchanged and now
        labelled truthfully. ``hvg_flavor`` is one of ``variance`` (the worker's own raw-count
        variance ranking), ``seurat_v3`` (scanpy; needs scikit-misc, which the SpatialDE env does not
        ship -- without it the run STOPS and names the package, nothing is substituted),
        ``cell_ranger`` (scanpy dispersion on counts; scanpy fails with "Bin edges must be unique"
        when 15% or more of the genes are zero in every tested spot, which is most whole-transcriptome
        Visium samples -- the run then stops and says so) or ``seurat`` (scanpy's log-data ranking:
        it applies expm1 first, and this worker's matrix is raw counts, so it is refused, never run).
        A flavour that cannot run stops the run with the reason; ``allow_hvg_fallback=True`` lets
        variance ranking stand in instead, and the payload then carries
        ``params.used_fallback=True`` and ``params.hvg_method_used='variance'``. The cut is reported
        in ``data.n_genes`` vs ``data.n_genes_used``, ``warnings`` and ``analysis``; genes cut here
        are absent from the output table, not ranked last in it. When nothing is cut,
        ``hvg_method_used`` is ``'none'`` and ``hvg_flavor`` is listed in ``params.ignored``.
      - Spots are never subsampled. Background spots flagged ``obs['in_tissue'] == 0`` (CELLxGENE
        Visium exports carry every array spot; the flag is also read from the tissue-positions file in
        visium_10x mode and from an ``in_tissue`` column of coords_path) are left out before the test,
        because their ambient counts would make the tissue/background edge read as a spatial pattern;
        the payload reports them (``params.in_tissue_filter``, ``data.n_spots_off_tissue_dropped``, a
        warning and the analysis). ``data.n_spots`` is the count supplied and ``data.n_spots_used`` the
        count tested. SpatialDE v1 keeps ten dense n_spots x n_spots float64 kernels in memory at once;
        a preflight estimates that against the memory available and refuses, naming both numbers, when
        it cannot fit (``params.kernel_memory_estimate_gib``). No parameter shrinks the kernel.
      - Counts: every stored value is checked. NaN/inf or negative values are an error. Non-integer
        values (normalised data) are an error that names the count and ``round_counts`` unless
        ``round_counts=True``, which rounds to the nearest integer and records it
        (``params.rounded_to_integers``, a warning). Pass ``layer`` to point at a raw-count layer
        instead, or ``use_raw_counts=True`` (h5ad mode, no ``layer``) to test ``adata.raw.X`` --
        CELLxGENE exports keep the counts there beside a processed X. A negative or non-finite X is
        refused, and the message says whether ``adata.raw`` holds counts. The matrix tested is
        ``params.expression_source`` (``X``, ``raw.X`` or ``layers['<name>']``); ``use_raw_counts``
        is listed in ``params.ignored`` in the visium_10x and matrix_with_coords modes, which have no
        adata.raw, and is an error together with ``layer``. With the default ``regress_formula``, a
        spot with zero total counts among the spots tested is an error: remove empty spots first
        (background spots flagged in_tissue == 0 are already left out).
      - ``top_k_genes`` (0 or more; a negative value is refused) is how many of the most significant
        genes are listed in top_genes.tsv / ``summary.top_genes`` and plotted.
      - Writes spatialde_results.csv with all tested genes (columns g, pval, qval, FSV, ...); the
        standardizer applies qval<0.05 to mark significant rows for downstream F1.
      - ``random_seed`` seeds numpy; the test itself is deterministic, so it only affects the random
        initialisation of AEH (``run_aeh=True``) and is listed in ``params.ignored`` otherwise. AEH is
        fitted to the qval<0.05 genes, as upstream documents; its patterns (a spots x C table) are
        written as ``aeh_patterns.pkl`` (output key ``aeh_patterns``) and ``aeh_patterns.csv``
        (``aeh_patterns_csv``).

    Modes (input_mode, required):
    - visium_10x: counts_h5 + spatial_dir
    - h5ad: h5ad_path (+ optional spatial_key, layer or use_raw_counts)
    - matrix_with_coords: matrix_path + coords_path (+ matrix_orientation, id_col, x_col, y_col)
    """
    payload = {
        "input_mode": input_mode,
        "output_dir": output_dir,
        "counts_h5": counts_h5,
        "spatial_dir": spatial_dir,
        "h5ad_path": h5ad_path,
        "spatial_key": spatial_key,
        "layer": layer,
        "matrix_path": matrix_path,
        "coords_path": coords_path,
        "matrix_orientation": matrix_orientation,
        "id_col": id_col,
        "x_col": x_col,
        "y_col": y_col,
        "regress_formula": regress_formula,
        "top_k_genes": top_k_genes,
        "hvg_top_n": hvg_top_n,
        "hvg_flavor": hvg_flavor,
        "run_aeh": run_aeh,
        "aeh_C": aeh_C,
        "aeh_l": aeh_l,
        "random_seed": random_seed,
        "allow_hvg_fallback": allow_hvg_fallback,
        "round_counts": round_counts,
        "use_raw_counts": use_raw_counts,
    }
    return run_worker_json(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, payload)


if __name__ == "__main__":
    mcp.run()
