#!/usr/bin/env python3
"""XFuse MCP wrapper for SpatialOmicsLab.

Exposes three tools:
  1) xfuse_spatial_analysis  — End-to-end: h5ad → convert → config → train → results
  2) xfuse_run               — Advanced: run from a pre-built TOML config
  3) xfuse_build_config      — Advanced: write a TOML config from slide paths

Only (1) accepts an h5ad. Both advanced tools need data already in XFuse's native .h5 format,
which nothing but (1) produces, so all three must stay registered in mcp_config.yaml for XFuse to
be reachable from a standard input file.
"""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli, run_worker_json

TOOL_NAME = "xfuse"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "XFUSE",
    "/opt/conda/envs/xfuse/bin/python",
    "/workspace/epic-fermat/agent/tools/xfuse_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def xfuse_spatial_analysis(
    st_h5ad: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 1,
    patch_size: int = 512,
    network_depth: int = 5,
    network_width: int = 16,
    learning_rate: float = 3e-4,
    gene_regex: str = ".*",
    min_counts: int = 1,
    slide_min_counts: int = 0,
    enable_metagenes: bool = True,
    enable_prediction: bool = False,
    enable_gene_maps: bool = False,
    session_path: str | None = None,
    round_counts: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run XFuse spatial super-resolution analysis end-to-end from a standard h5ad file.

    Automatically handles data conversion, configuration, and training.

    The input h5ad should contain:
    - adata.X: raw integer counts. XFuse fits a count likelihood and the converter stores integers,
      so a normalised or log matrix is refused (it used to be truncated: 0.7 -> 0, 2.9 -> 2).
      Negative or NaN/inf values are refused whatever round_counts says. When the counts are in
      adata.raw instead (CELLxGENE exports), pass use_raw_counts=True; the refusal names it.
    - adata.obsm['spatial']: spatial coordinates
    - adata.uns['spatial']: Visium-style metadata with an H&E image (hires, else lowres). The library
      is the one mapping-valued entry, so a scalar beside it (CELLxGENE's 'is_single') is skipped;
      an h5ad with more than one library is refused.
    - adata.obs['in_tissue']: tissue mask (optional). Spots with in_tissue == 0 are left out of the
      count data (the whole image is still converted) and the payload reports how many; without
      the column every spot is modelled.

    XFuse numbers spots in an int16 label image, so a slide with more than 32,767 on-tissue spots is
    refused (a Visium HD slide binned at 8 or 16 um typically has more).

    The payload reports data.n_spots / data.n_genes as supplied, and data.n_spots_used /
    data.n_genes_used as fitted: after the tissue mask, the conversion, XFuse's gene filter
    (min_counts, gene_regex) and its spot filter (slide_min_counts).

    Parameters
    ----------
    st_h5ad:
        Path to spatial transcriptomics AnnData h5ad file.
    output_dir:
        Directory for all outputs (converted data, config, training results).
    epochs:
        Number of training epochs (default: 50; use 3-10 for quick test).
    batch_size:
        Training batch size (default: 1; increase if GPU memory allows).
    patch_size:
        Size of image patches for training (default: 512; use 256 for speed).
    network_depth:
        XFuse neural network depth (default: 5).
    network_width:
        XFuse neural network width (default: 16).
    learning_rate:
        Optimizer learning rate (default: 3e-4).
    gene_regex:
        Regex to select genes (default: '.*' for all genes;
        use '^(?!RPS|RPL|MT-).*' to exclude ribosomal/mitochondrial).
    min_counts:
        Minimum reads per gene across all spots (default: 1). This is a GENE filter.
    slide_min_counts:
        Minimum reads per SPOT for that spot to be modelled (default: 0, keep every spot).
        XFuse masks out every spot below this. Its own project template uses 100, which drops
        background and near-empty spots; raise it if the slide has a lot of off-tissue area.
    enable_metagenes:
        Run metagene analysis after training (default: true).
    enable_prediction:
        Must stay false here (default: false); true is refused before anything runs. XFuse's
        prediction analysis sums predicted expression over the regions of a named annotation layer
        in the slide's data.h5, and this h5ad conversion writes none, so the run would train to the
        end and then fail. To predict per annotated region, convert with
        'xfuse convert visium --annotation', then use xfuse_build_config(annotation_layer=...) and
        xfuse_run.
    enable_gene_maps:
        Run gene map imputation after training (default: false). The maps are not masked to tissue:
        XFuse masks them with a zero-count background label that only its own image-based tissue
        detection ('xfuse convert --mask') writes, and this pipeline converts with --no-mask (the
        spots come from obs['in_tissue']). They cover the whole image, background included; the
        config says mask_tissue = false and the payload reports params.gene_maps_masked = false.
    session_path:
        Optional path to an '.session' file from an earlier run, to resume training instead of
        starting over. Leave unset for a fresh run.
    round_counts:
        Round adata.X to the nearest integer (np.rint) before conversion (default: false). Off, a
        non-integer X is an error rather than silently truncated. Use it only for counts stored as
        near-integer floats; the payload reports params.n_values_rounded and a warning (a pointed
        one when adata.raw holds counts: use_raw_counts is then the right switch).
    use_raw_counts:
        Fit adata.raw.X instead of adata.X (default: false), keeping obs, obsm and uns. For h5ads
        whose X is log-normalised or scaled with the integer counts in adata.raw (CELLxGENE
        exports). Refused when there is no adata.raw or it does not hold counts; the payload
        reports params.expression_source ('X' or 'raw.X').
    """
    payload = {
        "st_h5ad": st_h5ad,
        "output_dir": output_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "patch_size": patch_size,
        "network_depth": network_depth,
        "network_width": network_width,
        "learning_rate": learning_rate,
        "gene_regex": gene_regex,
        "min_counts": min_counts,
        "slide_min_counts": slide_min_counts,
        "enable_metagenes": enable_metagenes,
        "enable_prediction": enable_prediction,
        "enable_gene_maps": enable_gene_maps,
        "session_path": session_path,
        "round_counts": round_counts,
        "use_raw_counts": use_raw_counts,
    }
    return run_worker_json(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, payload)


@mcp.tool()
def xfuse_run(
    config_path: str,
    save_path: str,
    session_path: str | None = None,
) -> dict[str, Any]:
    """
    Run an XFuse analysis using a pre-built TOML configuration file.

    Use xfuse_spatial_analysis for end-to-end runs from h5ad files.
    Use this tool only when you have a custom TOML config.

    Parameters
    ----------
    config_path:
        Path to the XFuse TOML config file.
    save_path:
        Directory for XFuse outputs.
    session_path:
        Optional path to an '.session' file to resume a previous run.
    """
    args = ["--config", config_path, "--save-path", save_path]
    if session_path:
        args.extend(["--session", session_path])
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def xfuse_build_config(
    output_config_path: str,
    slides: list,
    epochs: int = 50,
    batch_size: int = 1,
    patch_size: int = 512,
    network_depth: int = 5,
    network_width: int = 16,
    learning_rate: float = 3e-4,
    gene_regex: str = ".*",
    min_counts: int = 1,
    enable_metagenes: bool = True,
    enable_prediction: bool = False,
    enable_gene_maps: bool = False,
    annotation_layer: str = "",
) -> dict[str, Any]:
    """
    Build a basic XFuse TOML configuration file from simple inputs
    (slides, optimization settings, and which analyses to enable).
    This tool writes a ready-to-use config file and returns its path.

    Parameters
    ----------
    output_config_path:
        Path where the TOML configuration file should be written.
    slides:
        List of slide dicts, each with 'name' and 'data' keys, and optionally 'min_counts' (a
        per-slide SPOT filter: spots whose summed counts fall below it are masked out; default 0).
        Each slide's always_keep -- the labels exempted from that filter -- is read off its data.h5:
        [1] when label 1 has zero counts (the background 'xfuse convert' writes with its default
        tissue mask), [] otherwise (a --no-mask conversion such as xfuse_spatial_analysis writes,
        where label 1 is a real spot), and [] with a warning when the file cannot be read yet.
        params.always_keep and params.always_keep_basis say what each slide got and why.
    epochs:
        Number of training epochs.
    batch_size:
        Training batch size.
    patch_size:
        Size of image patches for training.
    network_depth:
        XFuse neural network depth.
    network_width:
        XFuse neural network width.
    learning_rate:
        Optimizer learning rate.
    gene_regex:
        Regular expression selecting which genes XFuse models. Default ".*" keeps every gene.
        XFuse's own template uses "^(?!RPS|RPL|MT-).*" to exclude ribosomal and mitochondrial
        genes, which otherwise dominate the likelihood.
    min_counts:
        Drop genes whose total counts across the slide fall below this. Default 1 keeps every
        gene that was detected at all. This is a GENE filter, and is not the per-slide
        "min_counts" carried in `slides`, which filters low-quality spots.
    enable_metagenes:
        Include metagene analysis section in the config.
    enable_prediction:
        Include prediction analysis section in the config. XFuse's prediction analysis sums
        predicted expression over the regions of the annotation layer named by annotation_layer,
        and fails after training when a slide's data.h5 does not carry that layer. A slide file
        that already exists is checked now and a missing layer is refused; with no annotation_layer
        the config is written with a warning, because an empty name matches no layer.
    enable_gene_maps:
        Include gene map imputation section in the config, with XFuse's default mask_tissue = true.
        That mask is the slide's zero-count background label, which only a --mask conversion
        writes; a slide whose label 1 is a spot (a --no-mask conversion, as xfuse_spatial_analysis
        writes) has none, so its maps cover the whole image, and a warning names such slides.
    annotation_layer:
        Name of the annotation layer the prediction analysis reads (default: "", which no data.h5
        can carry). Layers are written by 'xfuse convert ... --annotation'; the h5ad conversion in
        xfuse_spatial_analysis writes none. Only used when enable_prediction is true.
    """
    payload = {
        "action": "build_config",
        "output_config_path": output_config_path,
        "slides": slides,
        "epochs": epochs,
        "batch_size": batch_size,
        "patch_size": patch_size,
        "network_depth": network_depth,
        "network_width": network_width,
        "learning_rate": learning_rate,
        "gene_regex": gene_regex,
        "min_counts": min_counts,
        "enable_metagenes": enable_metagenes,
        "enable_prediction": enable_prediction,
        "enable_gene_maps": enable_gene_maps,
        "annotation_layer": annotation_layer,
    }
    return run_worker_json(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, payload)


if __name__ == "__main__":
    mcp.run()
