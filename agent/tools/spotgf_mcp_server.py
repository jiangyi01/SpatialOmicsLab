#!/usr/bin/env python3
"""SpotGF denoising MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "spotgf"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPOTGF",
    "/opt/conda/envs/SpotGF/bin/python",
    "/workspace/epic-fermat/agent/tools/spotgf_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def spotgf_denoise(
    gem_path: str,
    output_dir: str,
    binsize: int = 70,
    proportion: float = 0.5,
    lower: float = 0.0,
    upper: float = 100000.0,
    max_iterations: int = 10000,
    auto_threshold: bool = True,
    visualize: bool = True,
    spot_size: int = 5,
    alpha: float = 0.0,
) -> dict[str, Any]:
    """
    Run the full SpotGF workflow on a GEM-format spatial transcriptomics file.

    SpotGF (``SpotGF.py``) bins the GEM at ``binsize``, scores every gene seen in more than 10 GEM rows by
    the optimal-transport distance between its spatial distribution and a uniform spread over the tissue
    outline (upstream grid-summarises a gene present in more than 5000 bins to 5000 points for this), and
    writes denoised GEMs that keep the highest-scoring genes. Genes in 10 rows or fewer are not scored and are
    kept in every denoised GEM. The payload lists only the files this run wrote: a same-named file left in
    ``output_dir`` by an earlier run is reported in a warning, not as an output.

    Parameters
    ----------
    gem_path:
        Path to the input GEM/txt/csv file (tab- or comma-separated; the delimiter is sniffed). A ``.gz``
        file is decompressed to a scratch file in ``output_dir`` first, because SpotGF itself cannot read
        one; the scratch file is removed after the run.
    output_dir:
        Directory for SpotGF outputs: ``SpotGF_scores.txt``, ``SpotGF_proportion_<proportion>.gem``,
        ``SpotGF_auto_threshold.gem`` (when ``auto_threshold``), ``alpha_shape.png`` (always) and the
        figures listed under ``visualize``.
    binsize:
        Denoising resolution: GEM coordinates are divided by it (integer bins) before scoring. 1 uses the
        cell-bin columns ``cen_x``/``cen_y`` when present.
    proportion:
        Share (0-1] of the scored genes the proportion GEM keeps, by SpotGF score. The unscored genes
        (10 GEM rows or fewer) are added on top, so the GEM holds more than this share of all genes; the
        kept count is in ``summary.n_genes_kept_proportion``. SpotGF keeps ``int(n_scored * proportion)``
        genes, so a proportion that keeps none is refused before the run (the message gives the minimum).
    lower, upper:
        Bounds of SpotGF's search for the tissue-outline alpha (``alphashape.optimizealpha``), which runs
        only when ``alpha`` is 0. With any other ``alpha`` they have no effect and are listed in
        ``params.ignored`` with a warning.
    max_iterations:
        Maximum iterations of that same alpha search; like ``lower``/``upper`` it has no effect (and is
        listed in ``params.ignored``) unless ``alpha`` is 0.
    auto_threshold:
        If True, also write ``SpotGF_auto_threshold.gem`` (genes above an automatic score threshold) and the
        two violin plots comparing the denoised GEMs with the raw data. False skips both. SpotGF builds the
        violins from its spatial figures, which it draws only for a denoised GEM with more than 200 genes, so
        with a smaller GEM it stops in that last step, after the scores and every GEM are written: the run
        is then reported with those files, ``summary.upstream_completed`` False and a warning, and no violin
        plot.
    visualize:
        If True, save ``Spatial_proportion.png`` (and ``Spatial_automatic.png`` with ``auto_threshold``).
        SpotGF draws a spatial figure only for a denoised GEM with more than 200 genes. SpotGF cannot run
        its auto threshold without these figures, so with ``auto_threshold=True`` they are always drawn
        and ``visualize=False`` is reported in ``params.ignored``; pass ``auto_threshold=False`` as well to
        run without figures. ``alpha_shape.png`` is written either way.
    spot_size:
        Marker size for spatial expression figures.
    alpha:
        Alpha parameter for tissue boundary detection. 0 (the default) searches for it within
        ``lower``/``upper`` for at most ``max_iterations`` iterations; any other value is used as the alpha
        directly. ``params.alpha_searched`` says which happened.
    """
    # Convert kwargs to CLI flags: key -> --key (snake_case -> kebab-case)
    kw = {
        "gem_path": gem_path,
        "output_dir": output_dir,
        "binsize": binsize,
        "proportion": proportion,
        "lower": lower,
        "upper": upper,
        "max_iterations": max_iterations,
        "auto_threshold": auto_threshold,
        "visualize": visualize,
        "spot_size": spot_size,
        "alpha": alpha,
    }
    args = []
    for k, v in kw.items():
        if v is None:
            continue
        flag = f"--{k.replace('_', '-')}"
        args.extend([flag, str(v)])

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
