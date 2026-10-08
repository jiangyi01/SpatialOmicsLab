#!/usr/bin/env python3
"""Starfysh deconvolution MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_json

TOOL_NAME = "starfysh"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "STARFYSH",
    "/opt/conda/envs/Starfysh/bin/python",
    "/workspace/epic-fermat/agent/tools/starfysh_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def starfysh_deconvolution(
    output_dir: str,
    input_mode: str = "visium_h5_spatial",
    sample_id: str = "V1_Human_Lymph_Node",
    count_h5: str | None = None,
    spatial_dir: str | None = None,
    outs_dir: str | None = None,
    counts_h5ad: str | None = None,
    coords_csv: str | None = None,
    hires_image: str | None = None,
    scalefactors_json: str | None = None,
    n_genes: int = 2000,
    use_poe: bool = True,
    hchannel: bool = False,
    signature_mode: str = "auto",
    signature_csv: str | None = None,
    aa_r: int = 30,
    aa_n_markers: int = 100,
    n_anchors: int = 30,
    window_size: float = 3.0,
    patch_r: int = 16,
    sig_version: str = "gene_score",
    n_repeats: int = 3,
    lr: float = 1e-3,
    epochs: int = 200,
    patience: int = 50,
    device: str = "auto",
    use_raw_counts: bool = False,
    allow_poe_fallback: bool = False,
) -> dict[str, Any]:
    """Run Starfysh deconvolution (reference-free; signatures provided or derived by Archetypal Analysis).

    Modes (input_mode, default "visium_h5_spatial"). There is no plain h5ad mode --
    Starfysh needs Visium-style tissue positions and scalefactors alongside the counts:
    - visium_h5_spatial: count_h5 + spatial_dir
    - spaceranger_outs: outs_dir (read in place; nothing in it is ever modified or deleted)
    - generic_counts_coords: counts_h5ad + coords_csv (+ hires_image, scalefactors_json)
    Each mode reads only its own inputs: hires_image and scalefactors_json (like counts_h5ad and
    coords_csv) are read only in generic_counts_coords; the Visium modes take the H&E image and
    scalefactors from their spatial/ folder. A supplied input the mode does not read is listed in
    params["ignored"]. The first and last modes stage the input under
    <output_dir>/_starfysh_input/<sample_id>, which is rebuilt from scratch on every run (the worker's
    own directory; nothing outside it is removed).

    Spots: in generic_counts_coords mode, background spots (in_tissue == 0 in the counts' obs, else in
    coords_csv) are left out before Starfysh runs (params["in_tissue_filter"] + a warning). Starfysh's
    own preprocessing then drops every spot whose pct_counts_mt is not below 100 -- zero-count and
    all-mitochondrial spots; data["n_spots_input"] and data["n_spots_dropped_by_starfysh_preprocess"]
    report it beside data["n_spots"] (the spots deconvolved).

    Counts: Starfysh normalises X as counts (normalize_total + log1p) and fits a count model, so the
    matrix must hold counts. In generic_counts_coords mode a counts_h5ad whose X has negative or NaN
    values (scaled / z-scored, as CELLxGENE exports can be) is refused, and the error says whether
    adata.raw holds counts; a non-integer X (log-normalised) runs with a warning; use_raw_counts=True
    (default False) runs on adata.raw.X instead. params["expression_source"] ("X" or "raw.X") and
    params["x_matrix_kind"] say what ran. The Visium modes read a 10x count matrix, which has no
    adata.raw, so use_raw_counts is listed in params["ignored"] there.

    Images: a Visium spatial_dir / outs_dir without tissue_hires_image.png and tissue_lowres_image.png
    (VisiumHD exports often ship neither) is read without images -- the counts, tissue positions and
    scalefactors are all Starfysh needs for the standard model -- and params["visium_images_missing"]
    names what was absent.

    Signatures: signature_csv is used when signature_mode="provided", or "auto" with a signature_csv
    given: one column per cell type / state, its marker genes listed down it (lengths may differ).
    A leading index column is optional -- the first column is the row index when its header is blank
    or none of its values is a gene of the counts, and a signature otherwise (a named first column is
    reported in the warnings). Every signature must share a gene with the counts. Otherwise
    Archetypal Analysis (AA) derives one marker set per archetype; the proportion columns are then
    arch_<i> factors, not named cell types. Provided signature names are kept exactly as given
    (numeric headers such as cluster ids included). If AA fails the run fails -- no substitute
    signature set is generated. params["signature_source"] records "provided" or "archetypal_analysis".

    aa_r is forwarded to ArchetypalAnalysis.compute_archetypes(cn=aa_r): the FisherS conditional number
    that chooses how many PCs feed the intrinsic-dimension estimate (the lower bound on the archetype
    count). It is NOT the number of archetypes. Must be a positive integer; 30 is the upstream default
    (0 raised ZeroDivisionError in skdim). The name is historical and kept for compatibility.

    PoE (use_poe=True, the default) is Starfysh's histology-integrated model and needs the paired H&E
    image: spatial/tissue_hires_image.png in the spatial_dir / outs_dir of a Visium or spaceranger
    input, or hires_image=<path> in generic_counts_coords mode (hires_image has no effect in the
    other two modes). Without an image a use_poe=True run is refused before any data is loaded, with
    a message saying how to supply one in the mode used. Pass use_poe=False to run standard Starfysh
    (which needs no image in any mode), or allow_poe_fallback=True (default False) to let the run
    degrade to standard Starfysh; params["method"] names the model that ran and
    params["used_fallback"] is true only when that degradation happened.

    patience is accepted for compatibility but has no effect: the installed utils.run_starfysh has no
    early stopping and every run trains for the full `epochs`. It is reported in params["ignored"].
    sig_version accepts "gene_score" (library default) or "norm"; other spellings are rejected up
    front. Outputs: starfysh_annotated.h5ad, cell_type_proportions.csv (obsm['qc_m']),
    gene_signatures_used.csv, anchor_spots.csv, training_losses.csv (one row per epoch of the best
    restart: loss = total, plus loss_<component>), starfysh_model_state.pt.
    """
    payload: dict[str, Any] = {
        "__tool__": "starfysh_deconvolution",
        "output_dir": output_dir,
        "input_mode": input_mode,
        "sample_id": sample_id,
        "n_genes": n_genes,
        "use_poe": use_poe,
        "allow_poe_fallback": allow_poe_fallback,
        "hchannel": hchannel,
        "signature_mode": signature_mode,
        "aa_r": aa_r,
        "aa_n_markers": aa_n_markers,
        "n_anchors": n_anchors,
        "window_size": window_size,
        "patch_r": patch_r,
        "sig_version": sig_version,
        "n_repeats": n_repeats,
        "lr": lr,
        "epochs": epochs,
        "patience": patience,
        "device": device,
        "use_raw_counts": use_raw_counts,
    }
    if count_h5:
        payload["count_h5"] = count_h5
    if spatial_dir:
        payload["spatial_dir"] = spatial_dir
    if outs_dir:
        payload["outs_dir"] = outs_dir
    if counts_h5ad:
        payload["counts_h5ad"] = counts_h5ad
    if coords_csv:
        payload["coords_csv"] = coords_csv
    if hires_image:
        payload["hires_image"] = hires_image
    if scalefactors_json:
        payload["scalefactors_json"] = scalefactors_json
    if signature_csv:
        payload["signature_csv"] = signature_csv
    return run_worker_json(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, payload)


if __name__ == "__main__":
    mcp.run()
