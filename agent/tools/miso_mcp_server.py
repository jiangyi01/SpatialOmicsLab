#!/usr/bin/env python3
"""MISO spatial domain clustering MCP wrapper for SpatialOmicsLab."""

import os
from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "miso"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "MISO",
    "/opt/conda/envs/miso/bin/python",
    "/workspace/epic-fermat/agent/tools/miso_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_miso(
    h5ad_path: str,
    histology_image_path: str = "",
    n_clusters: int = 6,
    output_dir: str = "",
    device: str = "auto",
    sparse: bool = False,
    neighbors: int = 100,
    image_resolution: str = "auto",
    scalefactors_json: str = "",
    pixel_size_raw: float = 0.0,
    spot_diameter_microns: float = 0.0,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """
    Run MISO clustering on a spatial transcriptomics .h5ad file.

    MISO reads no spot coordinates: its affinity graph is built from each modality's features, so
    spatial information enters the model only through histology image features sampled at each spot.
    With the RNA modality alone (no histology_image_path) the result is expression clusters, and the
    payload says so (params.uses_spatial_coordinates).

    MISO clusters a subset of what you supply. Spots with in_tissue=0 are dropped first
    (params.in_tissue_filter), and miso.utils.preprocess then drops every gene detected in
    fewer than 10 spots -- a threshold fixed inside the MISO library, which no parameter here
    can change. The payload reports the supplied counts as n_spots/n_genes and the analysed
    counts as n_spots_used/n_genes_used, and carries a warning naming each cut that fired.

    Counts: miso.utils.preprocess log1p's the matrix as counts. An X with negative or NaN values
    (scaled data) is refused, and the error says whether adata.raw holds counts; a fractional X
    (already log-normalised) runs with a warning; use_raw_counts=True runs on adata.raw.X
    (params.expression_source, params.x_matrix_kind).

    summary.n_clusters is the number of clusters the labels hold and summary.n_clusters_requested
    the n_clusters asked for; KMeans can return fewer on a degenerate embedding, with a warning.

    Memory: MISO holds the spots x genes matrix dense (intrinsic to the library). With sparse=False
    it also holds an N x N affinity per modality and a dense N x N distance matrix every epoch. The
    peak is estimated before anything is allocated; a run that cannot fit stops with the numbers
    (and names sparse=True when that would fit). Nothing is subsampled.

    Parameters
    ----------
    h5ad_path:
        Absolute path to a spatial transcriptomics .h5ad file with raw counts in .X (or in
        adata.raw, with use_raw_counts=True).
        obsm['spatial'] (full-resolution pixel coordinates) is required with a histology image and
        is otherwise only copied into the output CSV.
    histology_image_path:
        Optional path to the matching H&E image: the full-resolution TIF, or Space Ranger's
        tissue_hires_image.png / tissue_lowres_image.png. If provided, MISO computes image features
        via miso.hist_features.get_features and uses them as a second modality. Every spot's
        window must fall inside the image, or the run stops (an empty window is a NaN feature).
    n_clusters:
        Number of clusters for Miso.cluster() (KMeans on the MISO embedding).
    output_dir:
        Directory to store CSV and annotated h5ad. If empty, the portal's default output
        directory for MISO is used (base_mcp.default_output_dir("miso")) -- never the folder
        the input h5ad sits in.
    device:
        Compute device: "auto" (follow the hardware), "cpu", "gpu"/"cuda", or "cuda:N".
    sparse:
        False (default, the MISO tutorial's setting): MISO's dense Gaussian affinity over every
        pair of spots. True: MISO's own k-nearest-neighbour affinity (Miso(sparse=True)), which
        needs memory linear in the spot count. A different affinity; the payload records which ran.
    neighbors:
        k of the sparse kNN affinity (MISO's own default 100). Used only with sparse=True; with
        sparse=False a non-default value is reported in params.ignored.
    image_resolution:
        Which pixel frame the histology image is in: "fullres" (obsm['spatial'] used as-is),
        "hires" / "lowres" (coordinates multiplied by tissue_hires_scalef / tissue_lowres_scalef),
        or "auto" (default: hires/lowres when the file name or an exact size match with an image in
        uns['spatial'] says so, otherwise fullres). Needs a histology image.
    scalefactors_json:
        Path to Space Ranger's scalefactors_json.json. Empty (default): the scalefactors in
        adata.uns['spatial'][<library>] when the file holds one library (scalar markers beside
        it, such as CELLxGENE's is_single, are not libraries), then a scalefactors_json.json
        beside the image. Needs a histology image.
    pixel_size_raw:
        Microns per pixel of the supplied image. 0 (default): derived from the scalefactors
        (microns_per_pixel, or the physical spot spot_diameter_fullres measures -- bin_size_um on
        Visium HD, otherwise the 55 um Visium spot -- over spot_diameter_fullres, divided by the
        image's scale factor); only with no scalefactors at all is the MISO tutorial's 0.2544 used,
        with a warning. For a platform whose spot is not 55 um, pass pixel_size_raw. Needs a
        histology image.
    spot_diameter_microns:
        Diameter in microns of the image window each spot's features are averaged over; it sets the
        window radius only, never the image scale (params.histology.spot_diameter_um / spot_radius_px).
        0 (default): bin_size_um from the scalefactors (Visium HD), otherwise the 55 um Visium spot.
        Needs a histology image.
    use_raw_counts:
        False (default): MISO runs on X. True: on the counts in adata.raw.X, for an h5ad whose X is
        log-normalised or scaled and whose raw counts sit in adata.raw (a file without adata.raw, or
        whose raw is not counts, is refused).

    Returns
    -------
    JSON dict summarizing outputs (paths, cluster counts, etc.).
    """
    if not output_dir:
        # Never next to the input: a library sample folder is read-only data (found 2026-09-30).
        output_dir = default_output_dir("miso")
    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--h5ad",
        h5ad_path,
        "--output_dir",
        output_dir,
        "--n_clusters",
        str(int(n_clusters)),
        "--device",
        device,
        "--neighbors",
        str(int(neighbors)),
        "--image_resolution",
        str(image_resolution or "auto"),
    ]
    if histology_image_path:
        args.extend(["--histology_tif", histology_image_path])
    if sparse:
        args.append("--sparse")
    if scalefactors_json:
        args.extend(["--scalefactors_json", scalefactors_json])
    # 0 means "derive it"; any other value is forwarded, so a negative one is refused by name.
    if pixel_size_raw:
        args.extend(["--pixel_size_raw", repr(float(pixel_size_raw))])
    if spot_diameter_microns:
        args.extend(["--spot_diameter_microns", repr(float(spot_diameter_microns))])
    if use_raw_counts:
        args.append("--use_raw_counts")

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
