#!/usr/bin/env python3
"""UCD (ucdeconvolve) MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_json

TOOL_NAME = "ucdeconvolve"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "UCD",
    "/opt/conda/envs/ucdenv/bin/python",
    "/workspace/epic-fermat/agent/tools/ucdeconvolve_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def ucdeconvolve_base(
    output_dir: str,
    input_mode: str = "visium_h5_spatial",
    visium_h5_path: str | None = None,
    visium_spatial_dir: str | None = None,
    spaceranger_dir: str | None = None,
    h5ad_path: str | None = None,
    counts_h5ad_path: str | None = None,
    coords_csv: str | None = None,
    sample_id: str = "sample",
    coord_type: str = "array",
    token: str | None = None,
    key_added: str = "ucdbase",
    split: bool = True,
    sort: bool = True,
    propagate: bool = True,
    use_raw: bool = True,
    verbosity: int | None = None,
    assign_top_celltypes: bool = True,
    pred_key_added: str = "ucd_pred_celltype",
    assign_category: str | None = None,
    groupby: str = "",
    compute_neighbors: bool = False,
    n_neighbors: int = 10,
    knnsmooth_neighbors: int | None = None,
    knnsmooth_cycles: int = 1,
) -> dict[str, Any]:
    """Run UniCell Deconvolve (UCD) base model on spatial/bulk/sc data.

    UCDBase is UCD's pre-trained, reference-free model and it runs on the UCD cloud API: it needs a
    UCD token, and the expression matrix it reads is uploaded to UCD's service.

    Modes (input_mode, default "visium_h5_spatial"):
    - visium_h5_spatial: visium_h5_path + visium_spatial_dir
    - spaceranger_outs: spaceranger_dir
    - h5ad: h5ad_path
    - generic_counts_coords: counts_h5ad_path + coords_csv

    Background spots (obs["in_tissue"] == 0 -- every array spot of a CELLxGENE Visium h5ad, or the
    in_tissue column of coords_csv) are left out right after loading and never uploaded;
    params.in_tissue_filter and data.n_spots_supplied count them. coord_type ("array" or "pixel")
    is read by the two Visium modes only; any other value is refused there.

    Outputs in output_dir: ucdeconvolve_annotated.h5ad, plus one
    <key_added>_<category>_predictions.csv per prediction category UCD returned (raw; with
    split=True also primary, lines and cancer). A run that exports no prediction CSV is an error.

    use_raw/verbosity are passed straight to ucd.tl.base beside split/sort/propagate.
    use_raw=True reads counts from adata.raw when one is present, otherwise adata.X;
    params.counts_source names the matrix UCD read, and nothing is copied into a stand-in
    adata.raw. params also reports whether that matrix holds integer counts and how many of its
    genes match UCD's fixed input vocabulary; an input matching none is refused before upload.
    UCD un-logs and normalises what it reads, so counts and log1p data are both accepted; a matrix
    with negative or NaN/infinite values (scaled data) is refused before upload.

    The assign_* knobs shape the top-celltype label (only when assign_top_celltypes is on).
    ucdeconvolve writes it to obs["<pred_key_added>_<key_added>"] (with "_sm<k>" appended when
    smoothed), and summary.pred_celltype_key names that column. assign_category restricts the
    call to one prediction category UCD returned, groupby names an obs column so a whole cluster
    gets one label, and knnsmooth_neighbors/knnsmooth_cycles average the call over each spot's
    nearest neighbours in adata.obsp["distances"] -- an expression-space graph, not a spatial
    one. That graph exists only if one was built: set compute_neighbors=True (sc.pp.neighbors on
    X; n_neighbors sets its k and must exceed knnsmooth_neighbors) unless your input already
    carries one. These inputs are checked before anything is uploaded; a post-processing step
    that fails after the upload stops the run with an error, once the predictions are on disk.

    A knob set away from its default whose step does not run -- pred_key_added, assign_category,
    groupby and knnsmooth_* with assign_top_celltypes off, knnsmooth_cycles with no
    knnsmooth_neighbors, n_neighbors with compute_neighbors off, coord_type in the h5ad and
    generic_counts_coords modes -- is still echoed in params and is listed in params.ignored with a
    warning.
    """
    payload: dict[str, Any] = {
        "__tool__": "ucdeconvolve_base",
        "output_dir": output_dir,
        "input_mode": input_mode,
        "sample_id": sample_id,
        "coord_type": coord_type,
        "key_added": key_added,
        "split": split,
        "sort": sort,
        "propagate": propagate,
        "use_raw": use_raw,
        "verbosity": verbosity,
        "assign_top_celltypes": assign_top_celltypes,
        "pred_key_added": pred_key_added,
        "assign_category": assign_category,
        "groupby": groupby,
        "compute_neighbors": compute_neighbors,
        "n_neighbors": n_neighbors,
        "knnsmooth_neighbors": knnsmooth_neighbors,
        "knnsmooth_cycles": knnsmooth_cycles,
    }
    # UCD requires an API token. This used to be `token or $UCD_TOKEN`, which meant every run had
    # to be handed one: `ucdeconvolve` keeps its token in memory only, and each run is a fresh
    # worker process, so nothing carried over. `ucd_token.resolve` walks argument -> $UCD_TOKEN ->
    # project .env -> local cache, so a token acquired once is found by every later run with no
    # argument at all. It never prompts and never raises; a miss here is reported by the worker,
    # which is where the actionable sentence lives.
    try:
        from tools import ucd_token as _ucd_token
    except ImportError:  # running from the tools/ directory rather than the repo root
        import ucd_token as _ucd_token  # type: ignore[no-redef]
    _found = _ucd_token.resolve(token)
    if _found is not None:
        payload["token"] = _found.token
        # The source, never the value -- this string reaches logs.
        payload["token_source"] = _found.source
    if visium_h5_path:
        payload["visium_h5_path"] = visium_h5_path
    if visium_spatial_dir:
        payload["visium_spatial_dir"] = visium_spatial_dir
    if spaceranger_dir:
        payload["spaceranger_dir"] = spaceranger_dir
    if h5ad_path:
        payload["h5ad_path"] = h5ad_path
    if counts_h5ad_path:
        payload["counts_h5ad_path"] = counts_h5ad_path
    if coords_csv:
        payload["coords_csv"] = coords_csv
    return run_worker_json(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, payload)


if __name__ == "__main__":
    mcp.run()
