#!/usr/bin/env python3
"""stLearn spatial clustering MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "stlearn"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "STLEARN",
    "/opt/conda/envs/stlearn/bin/python",
    "/workspace/epic-fermat/agent/tools/stlearn_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def stlearn_spatial_clustering(
    st_h5ad: str,
    output_dir: str,
    n_pcs: int = 50,
    n_neighbors: int = 25,
    radius: int = 50,
    crop_size: int = 40,
    resolution: float = 1.0,
    random_state: int = 0,
    use_quality: str = "hires",
    allow_pca_fallback: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run stLearn stSME spatial clustering (PCA adjusted by histology morphology, then Louvain) on a
    spatial transcriptomics AnnData (.h5ad).

    stSME needs a histology image: ``uns['spatial'][<library>]['images'][use_quality]`` and, for
    'hires'/'lowres', ``scalefactors['tissue_<use_quality>_scalef']``. The library is the first
    ``uns['spatial']`` entry that holds images or scale factors (a scalar such as CELLxGENE's
    ``is_single`` is skipped, and kept in the output). Without that image, or when tiling / CNN
    feature extraction / morphology.adjust fails, the call stops with the reason unless
    ``allow_pca_fallback=True``, which clusters the expression PCA alone (no image, no spatial
    information) and says so in ``params.method``, ``params.used_fallback`` and a warning.

    Spots with ``obs['in_tissue'] == 0`` (background outside the tissue, which CELLxGENE Visium
    exports carry) are left out before anything runs and reported in ``params.in_tissue_filter``
    and a warning; ``data.n_spots`` is the number clustered and ``data.n_spots_input`` the number
    supplied. stLearn normalises X as counts: a negative or non-finite X (scaled data) is refused,
    naming ``use_raw_counts`` when ``adata.raw`` holds counts; a non-negative non-integer X runs
    with a warning; ``params.expression_source`` says which matrix ran.

    Parameters
    ----------
    st_h5ad:
        Path to spatial AnnData (.h5ad) with counts in X (or in adata.raw, with use_raw_counts=True)
        and obsm['spatial'].
    output_dir:
        Output directory for all results.
    n_pcs:
        Number of principal components for PCA. A sparse X is scaled to unit variance without
        densifying it; the PCA centres it implicitly.
    n_neighbors:
        Number of neighbors for kNN graph.
    radius:
        Radius, in pixels of the use_quality image, for st.spatial.morphology.adjust in stSME.
        Spots with no other spot inside it keep their unadjusted PCA (counted in the payload); a
        radius below the spot spacing, where no spot has a neighbour, is refused.
    crop_size:
        Tile size in pixels for st.pp.tiling to extract CNN features.
    resolution:
        Louvain resolution.
    random_state:
        Random seed.
    use_quality:
        Image quality key in uns['spatial'][library]['images'] ('hires', 'lowres' or 'fulres').
        A key the library does not hold is refused with the keys it does hold.
    allow_pca_fallback:
        Default False. True lets a run that cannot do stSME cluster the expression PCA alone
        instead of stopping; radius and crop_size are then listed under params.ignored.
    use_raw_counts:
        Default False. True runs on adata.raw.X instead of X, for an h5ad whose X is normalised or
        scaled and whose counts sit in adata.raw; refused when there is no adata.raw or it does not
        hold counts.
    """
    args = [
        "--task",
        "clustering",
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--n-pcs",
        str(n_pcs),
        "--n-neighbors",
        str(n_neighbors),
        "--radius",
        str(radius),
        "--crop-size",
        str(crop_size),
        "--resolution",
        str(resolution),
        "--random-state",
        str(random_state),
        "--use-quality",
        use_quality,
    ]
    if allow_pca_fallback:
        args.append("--allow-pca-fallback")
    if use_raw_counts:
        args.append("--use-raw-counts")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
