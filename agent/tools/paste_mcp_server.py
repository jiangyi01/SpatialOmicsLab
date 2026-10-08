#!/usr/bin/env python3
"""PASTE pairwise alignment MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "paste"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "PASTE",
    "/opt/conda/envs/paste_env/bin/python",
    "/workspace/epic-fermat/agent/tools/paste_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def paste_pairwise_align(
    slice_h5ads: list[str],
    output_dir: str,
    alpha: float = 0.1,
    use_gpu: bool = False,
    device: str = "",
    random_seed: int = 0,
    z_spacing: float = 0.0,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run PASTE pairwise alignment on a list of spatial slices.

    Spots with obs['in_tissue'] == 0 (background glass, carried by CELLxGENE Visium exports) are
    left out of every slice before alignment and counted in params.in_tissue_filter and
    params.in_tissue_dropped_per_slice; the aligned slices hold the in-tissue spots only. PASTE
    compares spots by a KL divergence over their expression, so X is read as counts: negative or
    non-finite X is refused, non-integer X runs with a warning. Two consecutive slices that share no
    gene name (gene symbols against Ensembl IDs) are refused; the genes each pair shares are in
    data.n_common_genes_per_pair.

    Parameters
    ----------
    slice_h5ads:
        List of paths to .h5ad files (at least 2 required).
    output_dir:
        Directory to store PASTE outputs.
    alpha:
        Tradeoff between spatial and expression terms.
    use_gpu:
        Whether to use GPU acceleration.
    device:
        Compute device: "cpu", "gpu"/"cuda", or "cuda:N" for a particular card on a multi-GPU
        host. Overrides ``use_gpu``, which can only say "some GPU".
    random_seed:
        Accepted and ignored in this mode: pst.pairwise_align is deterministic (uniform initial
        plan, exact EMD), so the seed changes nothing. Reported under params.ignored. It does
        reach paste_center_align, whose NMF is random.
    use_raw_counts:
        Align on adata.raw.X (raw counts) of every slice instead of X. Refused when a slice has no
        adata.raw or it does not hold counts. Default False aligns on X.
    z_spacing:
        Physical distance between consecutive sections, in the coordinates' own units. 0.0 means
        "not declared", not "zero apart": PASTE aligns in plane and cannot supply a z itself.
        Given, the aligned coordinates are a three-column ``obsm['spatial_3d_aligned']``; absent,
        two columns under ``obsm['spatial_aligned']``. Either way ``obsm['spatial']`` keeps what
        PASTE was given, without which the alignment cannot be validated.
    """
    if not slice_h5ads or len(slice_h5ads) < 2:
        return {
            "status": "error",
            "error": "paste_pairwise_align requires at least two slice_h5ads.",
        }

    args: list[str] = [
        "--mode",
        "pairwise",
        "--output-dir",
        output_dir,
        "--alpha",
        str(alpha),
        "--random-seed",
        str(random_seed),
        "--z-spacing",
        str(z_spacing),
    ]
    if use_gpu:
        args.append("--use-gpu")
    if device:
        args += ["--device", device]
    if use_raw_counts:
        args.append("--use-raw-counts")
    for p in slice_h5ads:
        args.extend(["--slice-h5ad", p])

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def paste_center_align(
    slice_h5ads: list[str],
    output_dir: str,
    alpha: float = 0.1,
    use_gpu: bool = False,
    device: str = "",
    random_seed: int = 0,
    z_spacing: float = 0.0,
    use_spatial_init: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run PASTE center alignment on multiple spatial slices.

    The worker leaves out background spots (obs['in_tissue'] == 0, counted in
    params.in_tissue_filter and params.in_tissue_dropped_per_slice), filters for
    common genes across slices (none shared is refused; the count is in
    data.n_common_genes), optionally builds spatial heuristic initial mappings,
    then calls center_align to infer a consensus center slice and per-slice
    couplings.  Finally it stacks slices around the center using
    stack_slices_center and saves aligned h5ad files and .npy coupling matrices.
    PASTE's KL divergence and NMF read X as counts: negative or non-finite X is
    refused, non-integer X runs with a warning.

    Parameters
    ----------
    slice_h5ads:
        List of paths to .h5ad files (at least 2 required).
    output_dir:
        Directory to store PASTE center-alignment outputs.
    alpha:
        Tradeoff between spatial and expression terms.
    use_gpu:
        Whether to use GPU acceleration.
    device:
        Compute device: "cpu", "gpu"/"cuda", or "cuda:N" for a particular card on a multi-GPU
        host. Overrides ``use_gpu``, which can only say "some GPU".
    random_seed:
        Seed for center_align's NMF (its random initialisation), for reproducibility.
    z_spacing:
        Physical distance between consecutive sections, in the coordinates' own units. 0.0 means
        "not declared", not "zero apart": PASTE aligns in plane and cannot supply a z itself.
        Given, the aligned coordinates are a three-column ``obsm['spatial_3d_aligned']``; absent,
        two columns under ``obsm['spatial_aligned']``. Either way ``obsm['spatial']`` keeps what
        PASTE was given, without which the alignment cannot be validated.
    use_raw_counts:
        Align on adata.raw.X (raw counts) of every slice instead of X. Refused when a slice has no
        adata.raw or it does not hold counts. Default False aligns on X.
    use_spatial_init:
        If true, build initial mappings from the slices' coordinates (obsm['spatial'], two
        columns) using match_spots_using_spatial_heuristic, from the first slice to each slice,
        and pass them as pis_init to center_align. The payload records the source in
        params.spatial_init_source.
    """
    if not slice_h5ads or len(slice_h5ads) < 2:
        return {
            "status": "error",
            "error": "paste_center_align requires at least two slice_h5ads.",
        }

    args: list[str] = [
        "--mode",
        "center",
        "--output-dir",
        output_dir,
        "--alpha",
        str(alpha),
        "--random-seed",
        str(random_seed),
        "--z-spacing",
        str(z_spacing),
    ]
    if use_gpu:
        args.append("--use-gpu")
    if device:
        args += ["--device", device]
    if use_spatial_init:
        args.append("--use-spatial-init")
    if use_raw_counts:
        args.append("--use-raw-counts")
    for p in slice_h5ads:
        args.extend(["--slice-h5ad", p])

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
