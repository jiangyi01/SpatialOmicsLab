#!/usr/bin/env python3
"""TACCO cell-type deconvolution MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "tacco"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "TACCO",
    "/opt/conda/envs/TACCO_env/bin/python",
    "/workspace/epic-fermat/agent/tools/tacco_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def tacco_annotate(
    sc_h5ad: str,
    spatial_h5ad: str,
    output_dir: str,
    annotation_key: str,
    result_key: str | None = None,
    method: str = "OT",
    multi_center: int = 3,
    lamb: float = 0.001,
    bisections: int | None = None,
    bisection_divisor: int = 3,
    platform_iterations: int | None = None,
    drop_unlabeled: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run TACCO to transfer cell-type annotations / compositions from
    single-cell reference to spatial transcriptomics data.

    Parameters
    ----------
    sc_h5ad:
        Path to the single-cell AnnData (.h5ad) with cell type labels.
    spatial_h5ad:
        Path to the spatial AnnData (.h5ad) to be annotated / deconvolved.
    output_dir:
        Directory where TACCO outputs will be written, e.g.:
          - tacco_composition.csv
          - tacco_dominant_label_per_spot.csv
          - tacco_spatial_with_annotations.h5ad (the spatial AnnData with the
            composition in .obsm and the dominant label in .obs)
    annotation_key:
        Column in sc_h5ad.obs used as annotation (e.g. "CellType").
    result_key:
        Optional .obsm key under which the per-spot composition matrix is stored
        in the annotated spatial AnnData; the dominant label per spot goes to
        .obs["<result_key>_max"]. Empty or None uses "tacco_<annotation_key>".
    method:
        TACCO annotation method. Default "OT". Also runnable in this
        environment: "nnls", "projection", "svm", "NMFreg". "tangram", "RCTD"
        and "SingleR" run in an external conda env that this tool does not
        expose and are refused; "novosparc" and "WOT" need their package
        installed. The payload records the method in params.method.
    multi_center:
        Number of sub-clusters (k-means centres) each reference cell type is
        split into before annotating (TACCO's multi_center); applies to every
        method. <=0 disables it.
    lamb:
        Regularization parameter (lambda) of the "OT" method. No other method
        takes it: for them it is not passed to TACCO and is listed in
        params.ignored with a warning.
    bisections:
        Number of recursive bisections of the annotation. Above 0 TACCO runs
        the chosen method repeatedly, each round assigning a fraction of the
        counts still unassigned, which sharpens compositions on spots holding
        several cell types at the cost of that many extra rounds. Leave unset
        to take TACCO's own method-dependent default, which is 4 for the "OT"
        method used here by default and 0 for every other method.
    bisection_divisor:
        Size of the fraction assigned per bisection round: each round assigns
        1/bisection_divisor of the counts still unassigned, and the total
        number of typing rounds is bisections + bisection_divisor - 1 -- so
        the default pair (OT's 4 bisections, divisor 3) is six rounds. Only
        read once bisections resolves above 0, and must then be at least 2;
        TACCO raises on anything smaller. Default 3.
    platform_iterations:
        Platform-normalization iterations run before annotating, correcting
        the systematic per-gene efficiency difference between the reference
        and the slide. 0 normalizes once and does not iterate; a positive
        value repeats the correction using the previous round's annotation;
        a negative value skips platform normalization entirely, which is only
        appropriate when reference and slide come from the same assay. Leave
        unset to take TACCO's own method-dependent default, which is 0 for the
        "OT" and "projection" methods and -1 -- no normalization at all -- for
        every other method.
    drop_unlabeled:
        Reference cells whose annotation_key label is missing (NaN, empty or
        the string "nan") stop the run by default, because a missing label is
        not a class. True leaves those cells out instead; the payload reports
        how many (params.n_reference_cells_dropped_unlabeled,
        data.n_cells_sc_used). Default False.
    use_raw_counts:
        TACCO treats expression as counts. Default False reads X of both
        inputs; X holding negative or NaN/inf values (scaled data) is refused,
        and non-integer X (normalised data) runs with a warning -- TACCO itself
        refuses a slide whose values do not look like integers. True runs on
        adata.raw of the spatial h5ad (refused when it has none or it does not
        hold counts) and of the reference when it has one (otherwise its X,
        with a warning); the annotated h5ad then carries those counts as X.
        params.expression_source / expression_source_sc say which matrix was read.

    Spots with obs['in_tissue'] == 0 (background glass in a CELLxGENE export)
    are left out before TACCO runs: data.n_spots is the slide supplied,
    data.n_spots_used the spots annotated, params.in_tissue_filter counts the
    rest, and only in-tissue spots appear in the outputs.

    The payload reports the values TACCO actually ran with: params.bisections
    and params.platform_iterations are the resolved values (the *_requested keys
    keep what was sent), and data.n_spots_annotated / n_spots_unannotated count
    the spots TACCO returned a composition for. TACCO leaves out spots with zero
    counts on the genes it keeps; their rows in tacco_composition.csv are empty.
    """
    args = [
        "--sc-h5ad",
        sc_h5ad,
        "--spatial-h5ad",
        spatial_h5ad,
        "--output-dir",
        output_dir,
        "--annotation-key",
        annotation_key,
        "--method",
        method,
        "--multi-center",
        str(multi_center),
        "--lamb",
        str(lamb),
        "--bisection-divisor",
        str(bisection_divisor),
    ]
    if result_key is not None and result_key != "":
        args.extend(["--result-key", result_key])
    # None has no CLI spelling: omitting the flag is what leaves TACCO's own default in place.
    # Both of these resolve per method upstream (bisections to 4 for "OT" and 0 otherwise;
    # platform_iterations to 0 for "OT"/"projection" and -1 otherwise), so sending a fixed
    # number would silently override that choice for every other method.
    if bisections is not None:
        args.extend(["--bisections", str(bisections)])
    if platform_iterations is not None:
        args.extend(["--platform-iterations", str(platform_iterations)])
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    if use_raw_counts:
        args.append("--use-raw-counts")

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
