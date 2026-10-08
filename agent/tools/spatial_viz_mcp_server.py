"""MCP portal: draw a figure from a dataset or from a result another tool wrote.

Every function here writes three things into the output directory: the figure, a record beside it
saying exactly how it was drawn, and a post-analysis manifest declaring it. The manifest is the
part that is easy to leave out and impossible to do without -- a directory holding only images is
never picked up for review, so the tool would succeed, the image would be on disk, and the answer
would show nothing at all.

Only functions whose pipeline is implemented and driven against real data are registered here.
A function that is declared and does nothing is worse than an absent one, because the model will
choose it.

The worker runs on the agent environment's own interpreter: matplotlib, scanpy and anndata are
already there, so there is no environment to build. The heavier families -- segmentation
boundaries, SpatialData scenes, very large point clouds -- live on a separate portal with its own
recipe, and each of them reports honestly when that environment is absent rather than quietly
drawing something simpler.
"""

from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "spatial-viz"

WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPATIAL_VIZ",
    "/opt/conda/envs/spatialomicsgym_env/bin/python",
    "/workspace/epic-fermat/agent/tools/spatial_viz_worker.py",
)

mcp = create_mcp(TOOL_NAME)

# An empty output_dir resolves to the work directory the config advertises -- ./work/spatial_viz,
# SOG_WORK_DIR honoured -- and is resolved per call, not at import. It used to reach the producers as
# "", which they read as this process's own working directory: the repository tree from a CLI, one
# shared directory for every portal conversation (hunt 2026-09-30, u20a-viz-pipelines-23). compose,
# export and update keep "" on purpose: for them it means "beside the figure's own record".

#: One argument crosses the process boundary as a single argv element, and the kernel caps that
#: at 128 KB. A refusal that names the limit keeps a working tool usable; an E2BIG from a healthy
#: interpreter reads as a broken installation and makes the model give the tool up.
INLINE_LIST_MAX_CHARS = 40_000


def _too_long(name: str, value: str) -> dict[str, Any] | None:
    if len(value or "") <= INLINE_LIST_MAX_CHARS:
        return None
    return {
        "status": "error",
        "tool": TOOL_NAME,
        "error": (
            f"{name} is {len(value):,} characters, above the {INLINE_LIST_MAX_CHARS:,}-character "
            "limit for one argument. Draw fewer at a time, or point at a file instead."
        ),
        "diagnostic": "This tool is installed correctly and does not need reprovisioning.",
    }


@mcp.tool()
def plot_spatial_expression(
    data_path: str,
    genes: str = "",
    obs_key: str = "",
    obsm_key: str = "",
    output_dir: str = "",
    layer: str = "",
    use_raw: bool = False,
    library_id: str = "",
    image_key: str = "",
    image_alpha: float = 1.0,
    spot_alpha: float = 1.0,
    point_size: float = 0.0,
    color_map: str = "",
    vmin: str = "",
    vmax: str = "",
    normalize: str = "auto",
    share_scale: bool = False,
    ncols: int = 3,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Paint continuous values on the tissue, optionally over the histology image."""
    refusal = _too_long("genes", genes)
    if refusal:
        return refusal
    args = [
        "--task",
        "spatial_expression",
        "--data-path",
        data_path,
        "--genes",
        genes,
        "--obs-key",
        obs_key,
        "--obsm-key",
        obsm_key,
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--layer",
        layer,
        "--library-id",
        library_id,
        "--image-key",
        image_key,
        "--image-alpha",
        str(float(image_alpha)),
        "--spot-alpha",
        str(float(spot_alpha)),
        "--point-size",
        str(float(point_size)),
        "--color-map",
        color_map,
        "--vmin",
        vmin,
        "--vmax",
        vmax,
        "--normalize",
        normalize,
        "--ncols",
        str(int(ncols)),
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    if use_raw:
        args.append("--use-raw")
    if share_scale:
        args.append("--share-scale")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def plot_spatial_annotation(
    data_path: str,
    obs_key: str = "",
    output_dir: str = "",
    library_id: str = "",
    image_key: str = "",
    image_alpha: float = 1.0,
    spot_alpha: float = 1.0,
    point_size: float = 0.0,
    split_panels: bool = False,
    ncols: int = 3,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Paint a categorical annotation -- domain, cell type or cluster -- on the tissue."""
    args = [
        "--task",
        "spatial_annotation",
        "--data-path",
        data_path,
        "--obs-key",
        obs_key,
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--library-id",
        library_id,
        "--image-key",
        image_key,
        "--image-alpha",
        str(float(image_alpha)),
        "--spot-alpha",
        str(float(spot_alpha)),
        "--point-size",
        str(float(point_size)),
        "--ncols",
        str(int(ncols)),
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    if split_panels:
        args.append("--split-panels")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def plot_embedding(
    data_path: str,
    color: str = "",
    basis: str = "auto",
    output_dir: str = "",
    layer: str = "",
    use_raw: bool = False,
    groupby: str = "",
    legend_loc: str = "right margin",
    point_size: float = 0.0,
    color_map: str = "",
    vmin: str = "",
    vmax: str = "",
    normalize: str = "auto",
    share_scale: bool = False,
    ncols: int = 3,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Scatter a stored embedding, coloured by genes or metadata, optionally split by a grouping."""
    refusal = _too_long("color", color)
    if refusal:
        return refusal
    args = [
        "--task",
        "embedding",
        "--data-path",
        data_path,
        "--color",
        color,
        "--basis",
        basis,
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--layer",
        layer,
        "--groupby",
        groupby,
        "--legend-loc",
        legend_loc,
        "--point-size",
        str(float(point_size)),
        "--color-map",
        color_map,
        "--vmin",
        vmin,
        "--vmax",
        vmax,
        "--normalize",
        normalize,
        "--ncols",
        str(int(ncols)),
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    if use_raw:
        args.append("--use-raw")
    if share_scale:
        args.append("--share-scale")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def generate_qc_report(
    data_path: str,
    output_dir: str = "",
    groupby: str = "",
    compute_if_missing: bool = True,
    on_tissue: bool = True,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Counts, detected genes and mitochondrial fraction, as distributions and on the tissue."""
    args = [
        "--task",
        "qc",
        "--data-path",
        data_path,
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--groupby",
        groupby,
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    if compute_if_missing:
        args.append("--compute-if-missing")
    if on_tissue:
        args.append("--on-tissue")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def plot_marker_expression(
    data_path: str,
    genes: str = "",
    groupby: str = "",
    output_dir: str = "",
    kind: str = "dotplot",
    layer: str = "",
    use_raw: bool = False,
    normalize: str = "auto",
    standardize: bool = False,
    top_n: int = 5,
    split_by: str = "",
    color_map: str = "",
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Marker genes across groups as a dot plot, violins or a heatmap, or group composition."""
    refusal = _too_long("genes", genes)
    if refusal:
        return refusal
    args = [
        "--task",
        "markers",
        "--data-path",
        data_path,
        "--genes",
        genes,
        "--groupby",
        groupby,
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--kind",
        kind,
        "--layer",
        layer,
        "--normalize",
        normalize,
        "--top-n",
        str(int(top_n)),
        "--split-by",
        split_by,
        "--color-map",
        color_map,
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    if use_raw:
        args.append("--use-raw")
    if standardize:
        args.append("--standardize")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def plot_differential_expression(
    data_path: str,
    group: str = "",
    output_dir: str = "",
    kind: str = "volcano",
    top_n: int = 15,
    fc_threshold: float = 1.0,
    p_threshold: float = 0.05,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Draw a stored differential-expression result, or refuse and say what is missing."""
    args = [
        "--task",
        "differential",
        "--data-path",
        data_path,
        "--group",
        group,
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--kind",
        kind,
        "--top-n",
        str(int(top_n)),
        "--fc-threshold",
        str(float(fc_threshold)),
        "--p-threshold",
        str(float(p_threshold)),
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def plot_deconvolution(
    data_path: str,
    obsm_key: str = "",
    proportions_csv: str = "",
    output_dir: str = "",
    kind: str = "maps",
    library_id: str = "",
    ncols: int = 3,
    top_n: int = 9,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Cell-type proportions from a deconvolution: per-type maps, the dominant type, or a summary."""
    args = [
        "--task",
        "deconvolution",
        "--data-path",
        data_path,
        "--obsm-key",
        obsm_key,
        "--proportions-csv",
        proportions_csv,
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--kind",
        kind,
        "--library-id",
        library_id,
        "--ncols",
        str(int(ncols)),
        "--top-n",
        str(int(top_n)),
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def plot_pathway_results(
    results_path: str = "",
    data_path: str = "",
    output_dir: str = "",
    kind: str = "enrichment",
    collection: str = "",
    method: str = "",
    pathways: str = "",
    obsm_key: str = "",
    top_n: int = 15,
    fdr_threshold: float = 0.05,
    library_id: str = "",
    ncols: int = 3,
    color_map: str = "",
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Enriched terms from a stored enrichment table, or per-spot pathway activity on the tissue."""
    args = [
        "--task",
        "pathway",
        "--results-path",
        results_path,
        "--data-path",
        data_path,
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--kind",
        kind,
        "--collection",
        collection,
        "--method",
        method,
        "--pathways",
        pathways,
        "--obsm-key",
        obsm_key,
        "--top-n",
        str(int(top_n)),
        "--fdr-threshold",
        str(float(fdr_threshold)),
        "--library-id",
        library_id,
        "--ncols",
        str(int(ncols)),
        "--color-map",
        color_map,
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def plot_spatial_statistics(
    data_path: str = "",
    results_path: str = "",
    output_dir: str = "",
    kind: str = "graph",
    obs_key: str = "",
    n_neighbors: int = 6,
    top_n: int = 20,
    library_id: str = "",
    color_map: str = "",
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Tissue organisation: the neighbour graph, or a statistic the spatial-statistics tool wrote."""
    args = [
        "--task",
        "organisation",
        "--data-path",
        data_path,
        "--results-path",
        results_path,
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--kind",
        kind,
        "--obs-key",
        obs_key,
        "--n-neighbors",
        str(int(n_neighbors)),
        "--top-n",
        str(int(top_n)),
        "--library-id",
        library_id,
        "--color-map",
        color_map,
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def plot_trajectory(
    data_path: str,
    output_dir: str = "",
    kind: str = "pseudotime",
    pseudotime_key: str = "",
    basis: str = "auto",
    genes: str = "",
    groupby: str = "",
    layer: str = "",
    use_raw: bool = False,
    normalize: str = "auto",
    n_bins: int = 20,
    library_id: str = "",
    point_size: float = 0.0,
    color_map: str = "",
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """A stored pseudotime on an embedding, on the tissue, or as gene trends along the ordering."""
    refusal = _too_long("genes", genes)
    if refusal:
        return refusal
    args = [
        "--task",
        "trajectory",
        "--data-path",
        data_path,
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--kind",
        kind,
        "--pseudotime-key",
        pseudotime_key,
        "--basis",
        basis,
        "--genes",
        genes,
        "--groupby",
        groupby,
        "--layer",
        layer,
        "--normalize",
        normalize,
        "--n-bins",
        str(int(n_bins)),
        "--library-id",
        library_id,
        "--point-size",
        str(float(point_size)),
        "--color-map",
        color_map,
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    if use_raw:
        args.append("--use-raw")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def plot_cell_communication(
    results_path: str,
    output_dir: str = "",
    kind: str = "interactions",
    source_column: str = "",
    target_column: str = "",
    score_column: str = "",
    top_n: int = 20,
    color_map: str = "",
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """An inferred ligand-receptor result as a sender-receiver matrix or a ranking of pairs."""
    args = [
        "--task",
        "communication",
        "--results-path",
        results_path,
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--kind",
        kind,
        "--source-column",
        source_column,
        "--target-column",
        target_column,
        "--score-column",
        score_column,
        "--top-n",
        str(int(top_n)),
        "--color-map",
        color_map,
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def compose_figure(
    figure_specs: str,
    output_dir: str = "",
    ncols: int = 2,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Place several finished figures on one contact sheet that indexes them."""
    refusal = _too_long("figure_specs", figure_specs)
    if refusal:
        return refusal
    args = [
        "--task",
        "compose",
        "--figure-specs",
        figure_specs,
        "--output-dir",
        output_dir,
        "--ncols",
        str(int(ncols)),
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def export_visualization(
    figure_spec: str,
    output_dir: str = "",
    data_path: str = "",
    export_format: str = "svg",
    dpi: int = 0,
    include_values: bool = True,
    bundle: bool = False,
) -> dict[str, Any]:
    """Re-render a figure for publication and ship the numbers it was drawn from beside it."""
    args = [
        "--task",
        "export",
        "--figure-spec",
        figure_spec,
        "--output-dir",
        output_dir,
        "--data-path",
        data_path,
        "--export-format",
        export_format,
        "--dpi",
        str(int(dpi)),
    ]
    if include_values:
        args.append("--include-values")
    if bundle:
        args.append("--bundle")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def run_visualization_pipeline(
    data_path: str,
    output_dir: str = "",
    depth: str = "overview",
    question: str = "",
    figure_format: str = "",
    dpi: int = 0,
) -> dict[str, Any]:
    """Inspect a dataset, draw the few figures worth drawing, and name what was left out."""
    # Empty is "not given": png, or svg at depth='publication'. The default used to be 'png', which
    # the worker then rewrote to "" to detect "not given", so an explicit png was impossible to ask
    # for at publication depth (hunt 2026-09-30, u20a-viz-pipelines-35).
    args = [
        "--task",
        "pipeline",
        "--data-path",
        data_path,
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--depth",
        depth,
        "--question",
        question,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def update_visualization(
    figure_spec: str,
    set_params: str = "",
    unset_params: str = "",
    output_dir: str = "",
    data_path: str = "",
    allow_reread: bool = True,
) -> dict[str, Any]:
    """Change an existing figure's appearance or layout without redoing the analysis behind it."""
    refusal = _too_long("set_params", set_params)
    if refusal:
        return refusal
    args = [
        "--task",
        "update",
        "--figure-spec",
        figure_spec,
        "--set-params",
        set_params,
        "--unset-params",
        unset_params,
        "--output-dir",
        output_dir,
        "--data-path",
        data_path,
    ]
    if allow_reread:
        args.append("--allow-reread")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def plot_spatial_3d(
    data_path: str,
    view: str = "scatter",
    genes: str = "",
    obs_key: str = "",
    layer: str = "",
    use_raw: bool = False,
    coords_key: str = "",
    section_key: str = "",
    z_spacing: float = 0.0,
    axis: str = "z",
    n_bins: int = 20,
    output_dir: str = "",
    point_size: float = 0.0,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Draw a reconstructed stack of serial sections in three dimensions.

    ``view`` chooses the figure: ``scatter`` is the volume from three fixed angles, ``depth`` is
    cells and mean value per z plane, ``axis`` bins a value along x, y or z.

    The z is read from a three-column coordinate key -- ``spatial_3d_aligned`` first, then
    ``spatial_3d_raw`` -- and only failing that built from the section axis. It is never invented:
    a stack drawn at z = 0, 1, 2 when the real sections are 98.5 microns apart and irregular is a
    picture of something that does not exist, and it renders perfectly.
    """
    args = [
        "--task",
        "spatial_3d",
        "--data-path",
        data_path,
        "--view",
        view,
        "--genes",
        genes,
        "--obs-key",
        obs_key,
        "--layer",
        layer,
        "--coords-key",
        coords_key,
        "--section-key",
        section_key,
        "--z-spacing",
        str(float(z_spacing)),
        "--axis",
        axis,
        "--n-bins",
        str(int(n_bins)),
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--point-size",
        str(float(point_size)),
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    if use_raw:
        args.append("--use-raw")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def plot_section_grid(
    data_path: str,
    genes: str = "",
    obs_key: str = "",
    layer: str = "",
    use_raw: bool = False,
    section_key: str = "",
    ncols: int = 4,
    output_dir: str = "",
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """One panel per section, on one shared colour scale.

    Use this on any object holding more than one section. The tissue-map functions panel per GENE,
    so a merged serial-section object comes out of them as every section overlaid in a single
    frame -- which looks like a plot and is not one.
    """
    args = [
        "--task",
        "section_grid",
        "--data-path",
        data_path,
        "--genes",
        genes,
        "--obs-key",
        obs_key,
        "--layer",
        layer,
        "--section-key",
        section_key,
        "--ncols",
        str(int(ncols)),
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    if use_raw:
        args.append("--use-raw")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def plot_alignment_qc(
    data_path: str,
    mode: str = "before_after",
    coords_key: str = "",
    before_key: str = "",
    after_key: str = "",
    section_key: str = "",
    pair: str = "",
    output_dir: str = "",
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """One adjacent pair of sections, side by side across two frames or overlaid in one.

    ``mode='before_after'`` refuses unless both frames are present and distinct, and reports the
    centroid offset of that one pair in each. Read the pair, not the stack: an alignment that
    improves the median while collapsing one pair is the failure this figure exists to make visible.

    ``mode='overlay'`` draws the pair at full size in a single named frame and needs only one, which
    makes it the Phase-1 figure: it answers whether two adjacent sections sit on top of each other
    before any alignment has been run.
    """
    args = [
        "--task",
        "alignment_qc",
        "--data-path",
        data_path,
        "--mode",
        mode,
        "--coords-key",
        coords_key,
        "--before-key",
        before_key,
        "--after-key",
        after_key,
        "--section-key",
        section_key,
        "--pair",
        pair,
        "--output-dir",
        output_dir or default_output_dir("spatial_viz"),
        "--title",
        title,
        "--figure-format",
        figure_format,
        "--dpi",
        str(int(dpi)),
        "--figure-id",
        figure_id,
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
