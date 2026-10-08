#!/usr/bin/env python3
"""BIDCell cell segmentation MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "bidcell"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "BIDCELL",
    "/opt/conda/envs/bidcell_env/bin/python",
    "/workspace/epic-fermat/agent/tools/bidcell_worker.py",
)

mcp = create_mcp(TOOL_NAME)


def _path(value: str) -> str:
    return str(Path(value).expanduser())


@mcp.tool()
def run_bidcell(
    input_dir: str,
    output_dir: str,
    config_path: str = "",
    model_type: str = "default",
    sc_reference_csv: str = "",
    pos_markers_csv: str = "",
    neg_markers_csv: str = "",
    dapi_path: str = "",
    transcripts_path: str = "",
    elongated_cell_types: str = "",
    total_steps: int = 4000,
    patch_size: int = 48,
    shift_to_origin: bool = False,
) -> dict[str, Any]:
    """
    Run BIDCell cell segmentation on subcellular spatial transcriptomics data.

    BIDCell trains a deep network on the sample itself, from a DAPI image (nuclei are segmented
    first with Cellpose) and a table of detected transcripts, guided by a single-cell reference and
    positive/negative marker tables. It writes a cell label image and a cell x gene count matrix.
    A single-cell reference is required: BIDCell has no default for it.

    Without ``config_path`` the worker writes ``<output_dir>/bidcell_auto_config.yaml`` from
    BIDCell's own Xenium template (transcript columns x_location / y_location / feature_name,
    0.2125 um DAPI pixels, transcript coordinates in um) and BIDCell writes everything under
    ``<output_dir>/bidcell_data/``. With ``config_path`` BIDCell reads every input and setting from
    that file and writes under that file's ``files.data_dir``; the auto-config options below are
    then ignored and listed in ``params.ignored``. The payload names the files where BIDCell wrote
    them, and a run that wrote no segmentation is an error.

    Parameters
    ----------
    input_dir:
        Directory holding the DAPI image (a .tif/.tiff whose name contains "dapi" or "morphology")
        and the transcript table (a .csv/.csv.gz/.tsv/.tsv.gz whose name contains "transcript").
        Read only. Not used when ``config_path`` is given.
    output_dir:
        Directory for the auto-config and BIDCell's working data: the cell label image is
        ``bidcell_data/model_outputs/<run>/test_output/epoch_1_step_<total_steps>_connected.tif`` and
        the cell x gene matrix ``bidcell_data/cell_gene_matrices/<run>/expr_mat.csv``. Use a new
        directory per run: BIDCell would reuse an earlier run's nuclei matrix found there.
    config_path:
        Optional BIDCell YAML config file. When given, it is used as is (inputs, reference, training
        settings and data_dir). A path that is not a file is an error, not a switch to the auto-config.
    model_type:
        Segmentation network: "default" (BIDCell's own network with the auto-config; the config
        file's model_params.name with config_path), "custom" (BIDCell's own network; needs
        patch_size divisible by 16), or a segmentation_models_pytorch encoder name such as
        "resnet34" (a U-Net with that encoder, untrained weights; needs patch_size divisible by 32).
        Sets model_params.name, also over a config_path file.
    sc_reference_csv:
        BIDCell single-cell reference CSV (fp_ref): one row per cell type, gene columns, then
        ct_idx, cell_type and atlas. Required without config_path.
    pos_markers_csv:
        Positive-marker table (fp_pos_markers), cell types x genes. Required without config_path.
    neg_markers_csv:
        Negative-marker table (fp_neg_markers), cell types x genes. Required without config_path.
    dapi_path:
        DAPI image to use; default "" finds the single matching image in input_dir. Several
        candidates (e.g. Xenium's morphology.ome.tif and morphology_mip.ome.tif) are an error that
        asks for this parameter. The image must be 2-D (BIDCell reads its first two axes as height
        and width): a Z-stack such as Xenium's morphology.ome.tif is refused with its shape, found
        or given; use the maximum projection morphology_mip.ome.tif.
    transcripts_path:
        Transcript table to use; default "" finds the single matching table in input_dir.
    elongated_cell_types:
        Comma-separated reference cell types that BIDCell treats as elongated (model_params.elongated),
        e.g. "Endothelial,Fibroblasts,Myofibroblasts,SMC". Each must be a cell_type in the reference.
        Default "" treats none as elongated.
    total_steps:
        Training steps (BIDCell's default 4000); prediction uses the checkpoint at this step
        (test_step = total_steps). At most 1000 or a multiple of 1000, and fewer than the image's
        training patches, or no checkpoint exists at that step and the run stops with the counts.
    patch_size:
        Side of the transcript-map patches the network sees (model_params.patch_size; 48 in
        BIDCell's Xenium template). BIDCell's own network needs a multiple of 16 and a
        segmentation_models_pytorch encoder a multiple of 32; any other value is refused before
        preprocessing starts.
    shift_to_origin:
        Shift transcript coordinates so their minimum is (0, 0). Leave False for a whole Xenium
        slide; set True when the DAPI image is a crop that starts at the first transcript.
    """
    args = [
        "--input-dir",
        _path(input_dir),
        "--output-dir",
        _path(output_dir),
        "--model-type",
        model_type,
        "--total-steps",
        str(int(total_steps)),
        "--patch-size",
        str(int(patch_size)),
    ]
    if config_path:
        args += ["--config-path", _path(config_path)]
    for flag, value in (
        ("--sc-reference-csv", sc_reference_csv),
        ("--pos-markers-csv", pos_markers_csv),
        ("--neg-markers-csv", neg_markers_csv),
        ("--dapi-path", dapi_path),
        ("--transcripts-path", transcripts_path),
    ):
        if value:
            args += [flag, _path(value)]
    if elongated_cell_types:
        args += ["--elongated-cell-types", elongated_cell_types]
    if shift_to_origin:
        args.append("--shift-to-origin")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
