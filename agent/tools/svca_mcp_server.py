#!/usr/bin/env python3
"""SVCA spatial variance component analysis MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "svca"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SVCA",
    # Run in spatialomicsgym_e1 (on a box that kept the pre-rebrand name, base_mcp maps the legacy
    # alias via constants.LEGACY_ENV_ALIASES) rather than the legacy svca env. The legacy svca env's openblas64 has a bug where
    # eigh on dense RBF kernels produces NaN columns + ~half negative eigvals; spatialomicsgym_e1's
    # newer BLAS handles the same matrix cleanly. The worker no longer needs the legacy `svca`
    # Python package (it implements FaST-LMM REML directly with numpy/scipy/h5py), so this switch
    # is safe.
    "/opt/conda/envs/spatialomicsgym_e1/bin/python",
    "/workspace/epic-fermat/agent/tools/svca_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def svca_variance_decomposition(
    h5ad_path: str,
    output_dir: str,
    spatial_key: str = "spatial",
    n_genes: int | None = None,
    max_spots: int | None = None,
    seed: int = 0,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Decompose gene expression variance into spatial components using SVCA.

    SVCA (Spatial Variance Component Analysis) decomposes gene expression
    variance into spatially structured and residual components using
    Gaussian process models. This reveals which genes have spatially
    structured expression patterns.

    This implementation fits a TWO-component model per gene: `intrinsic`
    (the spatial/covariance fraction) and `noise` (residual), which sum
    to 1. The published SVCA's third component -- `environmental`
    (cell-cell interaction) -- is NOT estimated: that column is reported
    as NaN in the output so downstream readers cannot mistake a
    placeholder zero for a measured absence. Do not promise an
    environmental/cell-cell estimate from this tool.

    Spots with obs['in_tissue'] == 0 are background outside the tissue
    (CELLxGENE Visium exports carry every array spot, and 56-70% of them
    are background on the library's samples): they are left out before
    the fit and reported in params.in_tissue_filter, a warning and the
    analysis text. A file without an in_tissue column is analysed whole.

    Parameters
    ----------
    h5ad_path:
        Path to the spatial transcriptomics AnnData (.h5ad) file. Must
        contain spatial coordinates in obsm[spatial_key] and expression
        data in X or raw.X. An optional obs['in_tissue'] 0/1 flag marks
        the background spots that are left out. SVCA library-size
        normalises and log1p-transforms the matrix, so it must hold
        counts: X is used (raw.X when the file has no X); a matrix with
        negative or non-finite values (scaled / z-scored data) is refused,
        naming use_raw_counts when raw.X holds counts, and a non-integer
        (already normalised) one runs with a warning that it was
        normalised twice. params.expression_source and
        params.x_matrix_kind say which matrix ran and what X held.
    output_dir:
        Directory where SVCA outputs will be written:
          - svca_variance_decomposition.csv (per-gene variance components)
          - svca_summary.csv (top spatially variable genes)
    spatial_key:
        Key in adata.obsm storing spatial coordinates as an (n_spots, 2)
        array. Default is 'spatial'. A key that is not in obsm is an error
        that lists the keys the file has; no other key is substituted.
    n_genes:
        Number of top highly variable genes to analyze. Leave as None to
        analyze all genes; this is the right choice for almost every
        request, not just for benchmarking. The subset is chosen by
        *ranking* genes on variance, so any figure reporting a mean or a
        range over the analysed genes is biased upward when this is set:
        capping a 649-gene MERFISH slide at 100 genes reported a mean
        intrinsic fraction of 0.38 where the whole slide gives 0.127.
        Setting it also lowers recall and F1 against curated SVG ground
        truth. Only set it when runtime is genuinely the bottleneck, and
        then use 500+ rather than a small subset. A value below 100 is not
        honoured: every gene is analysed instead, and the payload lists
        n_genes under params.ignored with a warning.
    max_spots:
        Optional cap on the number of spots analysed. Leave as None (the
        default) to analyse every spot of the slide that is in tissue. When
        set below that count, a random subset of this many in-tissue spots
        (drawn with `seed`) is analysed instead, and the payload says so in
        params.subsampled_spots, a warning and the `analysis` text, which
        names the spot count and percentage because a random 3.2% of a
        slide reads exactly like all of it. The estimator is exact: it
        eigendecomposes a dense spot-by-spot kernel, holding about three
        n x n float64 matrices at once (24 n^2 bytes: ~0.6 GB at 5000
        spots, ~9.6 GB at 20000) and taking time that grows as n^3. Before
        reading the expression matrix the worker estimates that memory and,
        if it exceeds what is available, refuses with both numbers; it
        never subsamples on its own.
    seed:
        Seed for the spot subsample drawn when max_spots is set below the
        slide's spot count. On such a run it decides *which* spots are
        analysed, so two runs with different seeds decompose different
        tissue and the subset cannot be recovered without it. It has no
        effect when every spot is analysed (the default); a non-zero seed
        on such a run is listed under params.ignored. Default 0; the
        payload reports it back.
    use_raw_counts:
        Decompose adata.raw.X instead of X. Use it when X is log-normalised
        or scaled and the integer counts sit in adata.raw (CELLxGENE Visium
        exports). Refused when the file has no adata.raw or it does not hold
        counts. Default False (X, or raw.X when the file has no X).

    The output column total_variance is the variance (ddof=1) of each
    gene's normalised expression over the analysed spots.
    """
    effective_n_genes = n_genes
    if n_genes is not None and n_genes < 100:
        effective_n_genes = None

    args = [
        "--h5ad-path",
        h5ad_path,
        "--output-dir",
        output_dir,
        "--spatial-key",
        spatial_key,
        "--seed",
        str(seed),
    ]
    if effective_n_genes is not None:
        args.extend(["--n-genes", str(effective_n_genes)])
    # Forward nothing when unset, so the worker's own default -- no cap, every spot -- stays the
    # single source of truth.
    if max_spots is not None:
        args.extend(["--max-spots", str(max_spots)])
    if use_raw_counts:
        args.append("--use-raw-counts")

    result = run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)
    if n_genes is not None and n_genes < 100 and isinstance(result, dict):
        why = (
            f"n_genes={n_genes} was overridden to None (analyze all genes). "
            "Values < 100 drop SVG recall against curated ground truth."
        )
        result["n_genes_override_note"] = why
        # The same keys worker_utils.record_ignored writes, so a reader of params.ignored / warnings
        # sees the discarded value the way it sees every other tool's.
        params = result.get("params")
        if not isinstance(params, dict):
            params = {}
            result["params"] = params
        ignored = params.get("ignored")
        if not isinstance(ignored, list):
            ignored = []
            params["ignored"] = ignored
        if "n_genes" not in ignored:
            ignored.append("n_genes")
        warnings = result.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
            result["warnings"] = warnings
        warnings.append(f"ignored parameter(s) n_genes: {why}")
    return result


if __name__ == "__main__":
    mcp.run()
