#!/usr/bin/env python3
"""SpatialPrompt MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_json

TOOL_NAME = "spatialprompt"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPATIALPROMPT",
    "/opt/conda/envs/spatialpromptENV/bin/python",
    "/workspace/epic-fermat/agent/tools/spatialprompt_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def spatialprompt_deconvolution(
    input_mode: str,
    output_dir: str,
    sc_h5ad: str,
    sc_label_key: str = "cell_type",
    counts_h5: str | None = None,
    spatial_dir: str | None = None,
    spatial_h5ad: str | None = None,
    st_h5ad: str | None = None,
    h5ad_path: str | None = None,
    max_genes: int = 2000,
    min_counts: int = 1,
    random_seed: int = 0,
    drop_unlabeled: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Estimate per-spot cell-type proportions with SpatialPrompt.

    Pass the spatial h5ad as `spatial_h5ad=...` (aliases `st_h5ad`/`h5ad_path`
    are also accepted for cross-tool kwarg-name consistency).

    Modes (input_mode, required):
    - h5ad: spatial_h5ad + sc_h5ad + sc_label_key
    - visium_10x: counts_h5 + spatial_dir + sc_h5ad + sc_label_key

    What reaches SpatialPrompt, in order (every cut is counted in the payload's `data`
    and named in `analysis`):
    - Spots: when obs['in_tissue'] is a 0/1 flag (h5ad, or the Space Ranger positions
      file in visium_10x mode), in_tissue == 0 spots are background and are left out;
      data.n_spots is the slide supplied, data.n_spots_used the spots deconvolved.
    - Genes: `min_counts` removes spatial genes with fewer total counts over those spots
      (0 disables it), then `max_genes` keeps the most variable of the rest (0 keeps all).
      data.n_genes is the panel supplied, data.n_genes_used what survived both cuts. With
      the default max_genes=2000 a whole-transcriptome slide is cut to 2000 genes before
      the reference is consulted, so SpatialPrompt's own reference-HVG choice (its
      n_hvgs=1000, among the genes shared with the reference) sees only those 2000.
    - Reference: a cell with no `sc_label_key` label is an error unless
      `drop_unlabeled=True`, which leaves such cells out and counts them.
    - Counts: SpatialPrompt TPM-normalises both matrices, so it reads them as counts.
      X holding negative or NaN/inf values (scaled data) is refused; non-integer X
      (normalised data) runs with a warning. `use_raw_counts=True` runs on adata.raw of
      the spatial h5ad (refused when it has none or it does not hold counts) and of the
      reference when it has one (otherwise its X, with a warning); in visium_10x mode it
      is ignored (a Space Ranger matrix is counts). params.expression_source /
      expression_source_sc say which matrix was read.

    Output: cell_type_proportions.csv under output_dir (spots x cell types, indexed by
    spot barcode). SpatialPrompt holds both matrices dense, so the worker estimates that
    size against available memory (MemAvailable, or the room under the cgroup memory
    limit when that is smaller) first and stops with the numbers if it cannot fit.
    """
    spatial_h5ad = spatial_h5ad or st_h5ad or h5ad_path
    payload: dict[str, Any] = {
        "__tool__": "spatialprompt_deconvolution",
        "input_mode": input_mode,
        "output_dir": output_dir,
        "sc_h5ad": sc_h5ad,
        "sc_label_key": sc_label_key,
        "max_genes": max_genes,
        "min_counts": min_counts,
        "random_seed": random_seed,
        "drop_unlabeled": drop_unlabeled,
        "use_raw_counts": use_raw_counts,
    }
    if input_mode == "visium_10x":
        payload["counts_h5"] = counts_h5
        payload["spatial_dir"] = spatial_dir
    elif input_mode == "h5ad":
        payload["spatial_h5ad"] = spatial_h5ad
    return run_worker_json(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, payload)


@mcp.tool()
def spatialprompt_cluster(
    input_mode: str,
    output_dir: str,
    cell_type_prop_csv: str | None = None,
    counts_h5: str | None = None,
    spatial_dir: str | None = None,
    spatial_h5ad: str | None = None,
    st_h5ad: str | None = None,
    h5ad_path: str | None = None,
    n_clust: int = 8,
    clust_label: str = "spatialprompt_cluster",
    random_seed: int = 0,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Cluster spatial spots into domains using SpatialPrompt's SpatialCluster.

    Pass spatial h5ad as `spatial_h5ad=...` (aliases `st_h5ad`/`h5ad_path`
    are accepted for cross-tool kwarg-name consistency).

    Modes (input_mode, required):
    - h5ad: spatial_h5ad
    - visium_10x: counts_h5 + spatial_dir

    SpatialCluster reads expression and coordinates only, so no reference and no
    deconvolution are needed. `cell_type_prop_csv` is optional: supply the CSV from
    spatialprompt_deconvolution on the same slide and each domain is additionally
    named after the cell types that dominate it.

    There is no latent-dimension setting: SpatialPrompt reduces at a fixed 50 components
    internally, so the only clustering knob is `n_clust`, which is handed to KMeans as the
    exact number of clusters.

    What is clustered: when obs['in_tissue'] is a 0/1 flag (h5ad, or the Space Ranger
    positions file in visium_10x mode), in_tissue == 0 spots are background and are left
    out -- data.n_spots is the slide supplied, data.n_spots_used the spots clustered, and
    only those appear in the outputs. SpatialCluster itself clusters on the 1000 genes
    with the highest raw-count variance (data.n_genes_used of data.n_genes); only those
    columns are made dense, and that dense size is checked against available memory
    (MemAvailable, or the room under the cgroup memory limit when that is smaller) first.

    SpatialCluster TPM-normalises the slide, so it reads X as counts: negative or NaN/inf
    values (scaled data) are refused and non-integer values (normalised data) run with a
    warning. `use_raw_counts=True` clusters adata.raw instead (refused when the h5ad has
    none or it does not hold counts; ignored in visium_10x mode, whose matrix is counts);
    params.expression_source says which matrix was read.

    Outputs: spatialprompt_spot_clusters.csv and spatialprompt_spatial_with_clusters.h5ad
    (labels in obs[clust_label]), plus spatialprompt_cluster_cell_types.csv when a
    proportions table covering every clustered spot is supplied.
    """
    spatial_h5ad = spatial_h5ad or st_h5ad or h5ad_path
    payload: dict[str, Any] = {
        "__tool__": "spatialprompt_cluster",
        "input_mode": input_mode,
        "output_dir": output_dir,
        "cell_type_prop_csv": cell_type_prop_csv,
        "n_clust": n_clust,
        "clust_label": clust_label,
        "random_seed": random_seed,
        "use_raw_counts": use_raw_counts,
    }
    if input_mode == "visium_10x":
        payload["counts_h5"] = counts_h5
        payload["spatial_dir"] = spatial_dir
    elif input_mode == "h5ad":
        payload["spatial_h5ad"] = spatial_h5ad
    return run_worker_json(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, payload)


if __name__ == "__main__":
    mcp.run()
