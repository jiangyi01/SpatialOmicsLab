#!/usr/bin/env python3
"""MIST (ReST) MCP wrapper for SpatialOmicsLab."""

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli
from worker_utils import pin_blas_threads

TOOL_NAME = "mist"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "MIST",
    "/opt/conda/envs/ReST/bin/python",
    "/workspace/epic-fermat/agent/tools/mist_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def mist_regions_impute(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    layer_key: str | None = None,
    species: str = "Human",
    hvg_prop: float = 0.8,
    n_pcs: int = 10,
    filter_spot: bool = True,
    min_sim: float = 0.1,
    min_size: int = 20,
    gap: float = 0.02,
    n_cores: int = 1,
    n_experts: int = 3,
    task: str = "all",
    seed: int = 0,
    allow_coordinate_rescale_fallback: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run the MIST (ReST) pipeline on a spatial AnnData file: ReST QC and preprocessing, region
    detection (ReST.extract_regions) and/or imputation (ReST.impute with its default imputer,
    spKNN: each zero is replaced by the mean of the spot's grid neighbours -- MIST's region-wise
    ensemble imputer is not run).

    Coordinates. MIST links two spots only when they are closer than 2 coordinate units
    (its graph radius, written for the Visium array grid where neighbours sit at sqrt(2)). The
    worker therefore reads obs['array_row'] / obs['array_col'] whenever both are present and
    falls back to obsm[spatial_key] only when they are absent. Pixel coordinates -- whose
    neighbours are tens of pixels apart -- would leave every spot 'isolated', so the worker
    refuses them unless allow_coordinate_rescale_fallback=True; the coordinates actually used
    are reported in params.coordinate_source (with params.used_fallback and
    params.coordinate_scale when a rescale ran).

    Outputs in output_dir: 'mist_region_assignments.csv' (spot_id, array_row, array_col,
    region -- the ReST region_ind per QC-passing spot; the value 'isolated' marks a spot in no
    region and is not a region), 'mist_region_stats.txt', and 'mist_imputed_expression.csv'
    (spKNN-imputed CPM matrix, QC-passing spots x genes). Spots with obs['in_tissue'] == 0
    (background outside the tissue) are left out before ReST sees the slide (counted in
    params.in_tissue_filter), and ReST's QC then removes spots with pct_counts_mt >= 25 (or no
    counts): data.n_spots is the input count, data.n_spots_used the count the tables cover, and a
    warning names both reductions. A run whose region detection finds no region at all is an
    error, not an ok payload without a region column. ReST densifies the data itself (spots x genes, and spots x spots similarity and
    neighbour matrices); a run whose estimate exceeds the available memory is refused with the
    numbers before it starts.

    Raw counts: ReST's QC and preprocessing drop spots by total count and CPM-normalise and
    log2-transform the matrix they are given, i.e. they treat it as counts. The matrix is checked
    first: a negative or non-finite one (scaled or z-scored data) is refused, and the error says
    whether adata.raw holds counts; a non-negative non-integer one (already normalised or
    log-transformed) runs as before but is normalised a second time, with a warning;
    use_raw_counts=True hands ReST adata.raw.X instead. params.expression_source ("X", "raw.X"
    or "layers['<key>']") and params.x_matrix_kind say which matrix ran.

    Parameters
    ----------
    st_h5ad:
        Path to spatial AnnData (.h5ad) with raw counts in X (or in a layer, or in adata.raw
        with use_raw_counts=True) and the Visium array grid in obs['array_row'] /
        obs['array_col'] (obsm[spatial_key] is read only when the grid columns are absent).
    output_dir:
        Directory for all outputs.
    spatial_key:
        obsm key read only when obs['array_row'/'array_col'] are absent.
    layer_key:
        Optional layers key for expression (else adata.X). It must exist: a missing layer is an
        error, not a silent switch to X. The layer is checked like X (raw counts expected). The
        source used is reported in params.expression_source. Cannot be combined with
        use_raw_counts=True.
    species:
        "Human" or "Mouse", any case (decides the MT-/mt- prefix of ReST's mitochondrial QC).
    hvg_prop:
        Proportion of genes ReST.preprocess keeps as highly variable for its PCA.
    n_pcs:
        Number of PCs in rd.preprocess.
    filter_spot:
        Accepted and IGNORED: ReST.preprocess has no spot-filter switch (its pct_counts_mt < 25
        QC always runs). Kept so existing callers keep working; listed in params.ignored.
    min_sim, min_size, gap:
        Region extraction parameters (rd.extract_regions).
    n_cores, n_experts:
        Accepted and IGNORED: they belong to MIST's ensemble imputer, which this tool does not
        run; the spKNN imputation that does run is single-threaded and has no experts. Listed in
        params.ignored.
    task:
        "all", "regions", "impute", or "regions_impute".
    seed:
        Accepted and IGNORED: nothing on the path this tool runs draws from a random stream it
        could seed (ReST's PCA uses scanpy's fixed random_state=0; region extraction and spKNN
        are deterministic). Listed in params.ignored.
    allow_coordinate_rescale_fallback:
        Default False: coordinates MIST cannot connect (median nearest-neighbour distance >= 2,
        i.e. pixels) are refused with a message naming this switch. True: they are shifted to
        the origin, divided so the median neighbour distance becomes sqrt(2), and rounded to
        integers; params.used_fallback is then True and params.coordinate_scale holds the factor.
    use_raw_counts:
        False (default): hand ReST adata.X (or the layer_key layer). True: hand it the counts in
        adata.raw.X; an h5ad without adata.raw, or whose adata.raw does not hold counts, is refused,
        and so is combining it with layer_key.
    """
    # ReST's dense linear algebra (sc.pp.scale / PCA, the spot-similarity matrix) runs on BLAS;
    # letting each worker use every core has thrashed a shared box before (commit d1689ee measured
    # load average ~328 and a stalled RCTD on a 96-core box). Pin BLAS to one thread per worker.
    # Must precede the worker launch: OpenBLAS reads the count once, when it is loaded, and the
    # subprocess inherits os.environ.
    pin_blas_threads()

    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--spatial-key",
        spatial_key,
        "--species",
        species,
        "--hvg-prop",
        str(hvg_prop),
        "--n-pcs",
        str(n_pcs),
        "--min-sim",
        str(min_sim),
        "--min-size",
        str(min_size),
        "--gap",
        str(gap),
        "--n-cores",
        str(n_cores),
        "--n-experts",
        str(n_experts),
        "--task",
        task,
        "--seed",
        str(seed),
    ]
    if layer_key is not None and layer_key != "":
        args.extend(["--layer-key", layer_key])
    if not filter_spot:
        args.append("--no-filter-spot")
    if allow_coordinate_rescale_fallback:
        args.append("--allow-coordinate-rescale-fallback")
    if use_raw_counts:
        args.append("--use-raw-counts")

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
