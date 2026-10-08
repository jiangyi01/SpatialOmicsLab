#!/usr/bin/env python3
"""CellPie intNMF MCP wrapper for SpatialOmicsLab: a reference-free topic model of a spatial slide."""

from __future__ import annotations

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "cellpie"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "CELLPIE",
    "/opt/conda/envs/cellpie_env/bin/python3.9",
    "/workspace/epic-fermat/agent/tools/cellpie_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_cellpie(
    spatial_h5ad_path: str,
    output_dir: str,
    sc_h5ad_path: str = "",
    n_components: int = 10,
    cell_type_key: str = "cell_type",
    epochs: int = 20,
    allow_expression_only_fallback: bool = False,
    allow_pca_image_fallback: bool = False,
    hvg_flavor: str = "seurat",
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run CellPie intNMF, a reference-free topic model of a spatial transcriptomics slide.

    CellPie jointly factorises the spot x gene expression matrix and a spot x image-feature
    matrix (``obsm['features']`` in the spatial h5ad) into ``n_components`` non-negative topics.
    The output ``cellpie_proportions.csv`` has one row per spot and columns ``topic_0..topic_{n-1}``
    (row-normalised loadings). Topics are latent expression programmes learned from the slide alone;
    they are NOT cell types, and no scRNA-seq reference takes part in the factorisation.

    Image modality, decided in this order and recorded in ``params.method`` / ``params.image_modality``:

    1. ``obsm['features']`` present in the spatial h5ad -> upstream's joint mode (``mod1_skew=1``).
    2. ``allow_expression_only_fallback=True`` -> upstream's expression-only mode (``mod1_skew=2``). A
       constant 1.0 one-column image matrix fills the slot upstream reads; its value enters upstream's
       first-epoch theta rescaling, so it is fixed and published as ``params.expression_only_image_constant``.
    3. ``allow_pca_image_fallback=True`` -> |PCA of log-normalised expression| stands in for image
       features (the expression is then factorised twice). A fallback, labelled as one.
    4. Otherwise the run stops and the error names these two switches.

    Both fallbacks set ``params.used_fallback=True``.

    Spots: background spots (``obs['in_tissue'] == 0``, as CELLxGENE Visium exports carry them) are
    left out before fitting and reported in ``params.in_tissue_filter`` and a warning;
    ``data.n_spots`` counts the in-tissue spots factorised (the rows of the outputs) and
    ``data.n_spots_supplied`` the spots in the file.

    Counts: the expression matrix is normalised with upstream's own recipe (normalize_total + log1p)
    before fitting, so X must hold counts. An X with negative or NaN values (scaled / z-scored) is
    refused, and the error says whether ``adata.raw`` holds counts; a non-integer X (log-normalised)
    runs with a warning; ``use_raw_counts=True`` factorises ``adata.raw.X``. ``params.expression_source``
    and ``params.x_matrix_kind`` say which matrix ran.

    ``cellpie_annotated.h5ad`` keeps the topic loadings in ``obsm['cellpie_proportions']`` and names
    the method in ``uns['cellpie_method']``; a fallback's stand-in image matrix is not left in
    ``obsm['features']``.

    Parameters
    ----------
    spatial_h5ad_path:
        Path to the spatial AnnData (.h5ad); expression in X. Optional ``obsm['features']``
        (spots x image features, dense or sparse, e.g. from CellPie's ``extract_features``) enables
        the joint mode; it is written back to the annotated h5ad exactly as supplied.
    output_dir:
        Directory to write CellPie outputs.
    sc_h5ad_path:
        Optional and accepted for compatibility only. CellPie is reference-free: the file is not
        read and it does not shape the topics. When given it is reported under ``params.ignored``.
    n_components:
        Number of NMF topics (columns of the proportions table).
    cell_type_key:
        Accepted for compatibility only; no reference is read, so it has no effect
        (reported under ``params.ignored`` when a reference is passed or the value is changed).
    epochs:
        intNMF optimisation epochs (upstream default 20).
    allow_expression_only_fallback:
        If True and the h5ad has no ``obsm['features']``, run upstream's expression-only mode
        (``mod1_skew=2``) instead of stopping. Off by default.
    allow_pca_image_fallback:
        If True and the h5ad has no ``obsm['features']``, let |PCA of the log-normalised expression|
        stand in for image features. Off by default; expression-only takes precedence when both are on.
    hvg_flavor:
        HVG flavour used only by the PCA image-feature fallback: 'seurat' (default, log-data
        dispersion) or 'seurat_v3' (needs scikit-misc in the cellpie env; a missing package stops
        the run rather than switching flavour).
    use_raw_counts:
        Default False. True factorises the counts in ``adata.raw.X`` instead of X (a CELLxGENE h5ad
        whose X is log-normalised or scaled keeps its integer counts there); an h5ad without
        ``adata.raw``, or whose raw is not counts, is refused.
    """
    args = [
        "--spatial-h5ad",
        spatial_h5ad_path,
        "--output-dir",
        output_dir,
        "--n-components",
        str(n_components),
        "--cell-type-key",
        cell_type_key,
        "--epochs",
        str(epochs),
        "--hvg-flavor",
        hvg_flavor,
    ]
    if sc_h5ad_path:
        args += ["--sc-h5ad", sc_h5ad_path]
    if allow_expression_only_fallback:
        args.append("--allow-expression-only-fallback")
    if allow_pca_image_fallback:
        args.append("--allow-pca-image-fallback")
    if use_raw_counts:
        args.append("--use-raw-counts")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
