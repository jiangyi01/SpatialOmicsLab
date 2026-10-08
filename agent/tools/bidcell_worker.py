#!/usr/bin/env python
"""
bidcell_worker.py

Worker script for running BIDCell cell segmentation on subcellular spatial
transcriptomics data: a DAPI image plus a table of detected transcripts, together
with a single-cell reference and its positive/negative marker tables.

- Called by the FastMCP wrapper (bidcell_mcp_server.py).
- Must be executed inside the BIDCell conda env: /opt/conda/envs/bidcell_env
- All logs go to stderr; stdout only prints a single JSON line at the end.

Two ways to configure a run
---------------------------
* ``--config-path``: a BIDCell YAML config you wrote. BIDCell reads every input, the
  reference, the training settings and ``files.data_dir`` from that file; the worker's
  auto-config options are then ignored (and listed in ``params.ignored``). ``--model-type``
  other than ``default`` still overrides ``model_params.name``.
* No ``--config-path``: the worker writes ``<output_dir>/bidcell_auto_config.yaml`` from
  BIDCell's own Xenium template (``bidcell/example_params/xenium.yaml``: transcript columns
  x_location / y_location / feature_name, 0.2125 um DAPI pixels, transcript coordinates in um,
  4000 training steps -- BIDCell's default). It fills in the DAPI image and transcript table
  (found in ``--input-dir`` or given explicitly), the reference and marker CSVs, and the options
  below. Other platforms need their own config file.

Where the outputs are
---------------------
BIDCell writes everything under its config's ``files.data_dir``:
``model_outputs/<experiment>/test_output/epoch_E_step_S_connected.tif`` (the cell label image),
``cell_gene_matrices/<experiment>/expr_mat.csv`` (cells x genes), plus its intermediates. The
auto-config sets ``data_dir`` to ``<output_dir>/bidcell_data``; with ``--config-path`` it is
whatever that file says. The payload lists the files where BIDCell actually wrote them, and a run
that did not write its segmentation is an error, not a success.

Example (manual test):

  (bidcell_env) python /workspace/epic-fermat/agent/tools/bidcell_worker.py \
      --input-dir /workspace/work/xenium_outs \
      --output-dir /workspace/work/bidcell_output \
      --sc-reference-csv /workspace/work/refs/sc_breast.csv \
      --pos-markers-csv /workspace/work/refs/sc_breast_markers_pos.csv \
      --neg-markers-csv /workspace/work/refs/sc_breast_markers_neg.csv
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import traceback
from typing import Any

# Ensure worker_utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import WorkerOutput, cpu_budget, preflight_check, record_ignored, record_method

# ---------------------------------------------------------------------------------------------- #
# Auto-config: BIDCell's own Xenium template, not the 60-step toy ``small_example.yaml``.
# ---------------------------------------------------------------------------------------------- #
AUTO_TEMPLATE = "xenium.yaml"
AUTO_CONFIG_NAME = "bidcell_auto_config.yaml"
AUTO_DATA_SUBDIR = "bidcell_data"
DEFAULT_TOTAL_STEPS = 4000  # bidcell.config.TrainingParams.total_steps / TestingParams.test_step
DEFAULT_PATCH_SIZE = 48  # model_params.patch_size in BIDCell's Xenium template
# train() saves a checkpoint every min(training_params.model_freq, test_step) steps and predict()
# loads the one at test_step; 1000 is TrainingParams.model_freq's default.
MODEL_FREQ_DEFAULT = 1000
# segmentation_models_pytorch U-Nets (depth 5) need inputs divisible by 2**5.
SMP_OUTPUT_STRIDE = 32
# BIDCell's own network (model_params.name='custom', bidcell/model/model/model.py) is a UNet3+ that halves
# the patch with four MaxPool2d(2) steps and concatenates the scales back together, so its patch side must
# be divisible by 2**4. Measured in bidcell_env on SegmentationModel(n_channels=4): 16, 32, 48, 64, 80 and
# 96 run; 24, 40 and 56 stop in torch.cat with "Sizes of tensors must match".
CUSTOM_OUTPUT_STRIDE = 16
DAPI_EXTENSIONS = (".tif", ".tiff")
TRANSCRIPT_EXTENSIONS = (".csv", ".csv.gz", ".tsv", ".tsv.gz")

# Options that only shape the auto-config. With --config-path they have no effect, so a value
# other than the default is reported in params.ignored instead of being echoed as if applied.
AUTO_ONLY_DEFAULTS = {
    "sc_reference_csv": "",
    "pos_markers_csv": "",
    "neg_markers_csv": "",
    "dapi_path": "",
    "transcripts_path": "",
    "elongated_cell_types": "",
    "total_steps": DEFAULT_TOTAL_STEPS,
    "patch_size": DEFAULT_PATCH_SIZE,
    "shift_to_origin": False,
}


def log(msg: str) -> None:
    """Print log messages to stderr with a prefix."""
    sys.stderr.write(f"[bidcell-worker] {msg}\n")
    sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BIDCell worker: cell segmentation from a DAPI image and a transcript table."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory holding the DAPI image and transcript table (read-only; nothing is written here).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for the auto-config and, without --config-path, BIDCell's data_dir (bidcell_data/).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="",
        help="Optional BIDCell YAML config. When given, BIDCell reads all inputs and settings from it.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="default",
        help="'default' (BIDCell's own network in the auto-config; the config's model_params.name otherwise), "
        "'custom', or a segmentation_models_pytorch encoder name such as 'resnet34'.",
    )
    parser.add_argument("--sc-reference-csv", type=str, default="", help="BIDCell single-cell reference CSV (fp_ref).")
    parser.add_argument("--pos-markers-csv", type=str, default="", help="Positive-marker CSV (fp_pos_markers).")
    parser.add_argument("--neg-markers-csv", type=str, default="", help="Negative-marker CSV (fp_neg_markers).")
    parser.add_argument("--dapi-path", type=str, default="", help="DAPI image; default: the one in --input-dir.")
    parser.add_argument(
        "--transcripts-path", type=str, default="", help="Transcript table; default: the one in --input-dir."
    )
    parser.add_argument(
        "--elongated-cell-types",
        type=str,
        default="",
        help="Comma-separated reference cell types BIDCell treats as elongated (model_params.elongated).",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=DEFAULT_TOTAL_STEPS,
        help="Training steps; also the step whose checkpoint is used for prediction (test_step).",
    )
    parser.add_argument(
        "--patch-size", type=int, default=DEFAULT_PATCH_SIZE, help="model_params.patch_size of the auto-config."
    )
    parser.add_argument(
        "--shift-to-origin",
        action="store_true",
        help="Shift transcript coordinates so their minimum is (0, 0) -- for a DAPI crop of a larger slide.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------------------------- #
# Small helpers
# ---------------------------------------------------------------------------------------------- #
def _parse_cell_types(text: str) -> list:
    """``"Endothelial, Fibroblasts"`` -> ``["Endothelial", "Fibroblasts"]`` (empty entries dropped)."""
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _resolve_model_name(model_type: str):
    """Map ``model_type`` onto BIDCell's ``model_params.name``; ``None`` means "leave it as configured".

    BIDCell's train()/predict() build their own ``Network`` for ``"custom"`` and a
    ``segmentation_models_pytorch`` U-Net with that encoder for any other name, so any other name
    must be a real smp encoder -- an unknown one would only fail after preprocessing.
    """
    value = str(model_type or "").strip()
    if value in ("", "default"):
        return None
    if value == "custom":
        return "custom"
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            f"model_type={value!r} names a segmentation_models_pytorch encoder, but that package is not "
            f"importable here ({exc}). Use model_type='default' or 'custom' for BIDCell's own network."
        ) from exc
    names = list(smp.encoders.get_encoder_names())
    if value not in names:
        raise ValueError(
            f"model_type={value!r} is neither 'default', 'custom' nor a segmentation_models_pytorch encoder. "
            f"Examples of valid encoders: {', '.join(names[:8])} ({len(names)} in total)."
        )
    return value


def _describe_model(name: str) -> str:
    if name == "custom":
        return "BIDCell (its own segmentation network, model_params.name='custom')"
    return f"BIDCell (segmentation_models_pytorch U-Net with the {name} encoder)"


def _check_patch_size(name: str, patch_size: int, config_source: str) -> None:
    """Both networks reject a patch side they cannot halve evenly -- but only once training starts.

    That is after the whole preprocessing (Cellpose nuclei, expression maps, patches,
    pre-annotation), so it is checked here, before any of it runs.
    """
    patch_size = int(patch_size)
    where = "patch_size" if config_source == "auto" else "model_params.patch_size in your config_path file"
    if name == "custom":
        if patch_size % CUSTOM_OUTPUT_STRIDE == 0:
            return
        raise ValueError(
            f"BIDCell's own network (model_params.name='custom', what model_type='default' runs with the auto-config) "
            f"halves the patch four times and joins the scales back together, so it needs a patch side divisible by "
            f"{CUSTOM_OUTPUT_STRIDE}; the patch size is {patch_size}. Set {where} to a multiple of "
            f"{CUSTOM_OUTPUT_STRIDE} (e.g. 48, BIDCell's Xenium default, or 64)."
        )
    if patch_size % SMP_OUTPUT_STRIDE == 0:
        return
    raise ValueError(
        f"model_type={name!r} builds a segmentation_models_pytorch U-Net, which needs image patches whose side is "
        f"divisible by {SMP_OUTPUT_STRIDE}; the patch size is {patch_size}. Set {where} to a multiple of "
        f"{SMP_OUTPUT_STRIDE} (e.g. 64), or use model_type='default' for BIDCell's own network."
    )


def _dapi_shape(path: str):
    """``(shape, axes)`` of the DAPI image as ``tifffile.imread`` -- BIDCell's reader -- returns it.

    Read from the TIFF header (``TiffFile.series[0]``, the array ``imread`` builds), not the pixels.
    ``(None, "")`` when tifffile is absent or cannot parse the file: BIDCell then reads it itself.
    """
    try:
        import tifffile
    except ImportError:
        return None, ""
    try:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            return tuple(int(n) for n in series.shape), str(getattr(series, "axes", "") or "")
    except Exception as exc:
        log(f"Could not read the TIFF header of {path} to check it is 2-D ({exc}); BIDCell reads it itself.")
        return None, ""


def _check_dapi_is_2d(path: str, where: str) -> None:
    """BIDCell's nuclei step reads ``dapi.shape[0]`` / ``shape[1]`` as height / width, with no projection.

    A Z-stack (Xenium's ``morphology.ome.tif``) or any other image that is not 2-D is therefore sliced
    along the wrong axes after the run has started, so it is refused here with its shape.
    """
    shape, axes = _dapi_shape(path)
    if shape is None or len(shape) == 2:
        return
    raise ValueError(
        f"The DAPI image {path} is {len(shape)}-D: tifffile, which BIDCell reads it with, gives shape {shape}"
        + (f" (axes {axes})" if axes else "")
        + ". BIDCell segments nuclei on a 2-D image and takes its first two axes as height and width. Set "
        f"{where} to a 2-D DAPI image (for Xenium, the maximum projection morphology_mip.ome.tif; a Z-stack such as "
        "morphology.ome.tif has to be projected to 2-D first)."
    )


def _checkpoint_is_reachable(test_step: int, model_freq: int = MODEL_FREQ_DEFAULT) -> bool:
    """Whether train()'s save rhythm includes the step predict() loads (enough patches permitting).

    train() saves at every step divisible by ``min(training_params.model_freq, test_step)``.
    """
    test_step = int(test_step)
    if test_step < 1:
        return False
    return test_step % min(int(model_freq), test_step) == 0


def _has_extension(name: str, extensions) -> bool:
    lower = name.lower()
    return any(lower.endswith(ext) for ext in extensions)


def _discover(input_dir: str, kind: str) -> str:
    """Find the one DAPI image or transcript table in ``input_dir``; refuse to guess between several."""
    if kind == "dapi":
        words, extensions, knob = ("dapi", "morphology"), DAPI_EXTENSIONS, "dapi_path"
    else:
        words, extensions, knob = ("transcript",), TRANSCRIPT_EXTENSIONS, "transcripts_path"
    found = sorted(
        name
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
        and any(word in name.lower() for word in words)
        and _has_extension(name, extensions)
    )
    if not found:
        if kind == "dapi":
            raise FileNotFoundError(
                f"No DAPI image found for BIDCell (fp_dapi unset): {input_dir} has no .tif/.tiff whose name contains "
                f"'dapi' or 'morphology'. Pass dapi_path, or a config_path file."
            )
        raise FileNotFoundError(
            f"No transcript table found for BIDCell: {input_dir} has no file whose name contains 'transcript' and "
            f"ends in {', '.join(TRANSCRIPT_EXTENSIONS)}. Pass transcripts_path, or a config_path file."
        )
    if len(found) > 1:
        raise ValueError(
            f"{len(found)} candidate {'DAPI images' if kind == 'dapi' else 'transcript tables'} in {input_dir}: "
            f"{', '.join(found)}. Name the one to use with {knob} (for Xenium, the 2-D morphology_mip image)."
        )
    return os.path.join(input_dir, found[0])


def _read_header(path: str) -> list:
    """Column names of a CSV/TSV, gzipped or not, read from its first line only."""
    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        first = fh.readline().rstrip("\r\n")
    sep = "\t" if ".tsv" in os.path.basename(path).lower() else ","
    return [col.strip().strip('"').strip("'") for col in first.split(sep)]


def _reference_cell_types(path: str) -> list:
    """The ``cell_type`` column of a BIDCell reference (one row per cell type; the file is small)."""
    import pandas as pd

    header = _read_header(path)
    missing = [col for col in ("ct_idx", "cell_type") if col not in header]
    if missing:
        raise ValueError(
            f"sc_reference_csv {path} is not a BIDCell reference: it has no {', '.join(missing)} column(s). BIDCell "
            f"expects one row per cell type, gene columns, then ct_idx, cell_type and atlas (see its sc_breast.csv)."
        )
    return [str(v) for v in pd.read_csv(path, usecols=["cell_type"])["cell_type"].tolist()]


def _count_rows(path: str) -> int:
    """Data rows in a CSV written by pandas (header line excluded); no DataFrame is built."""
    with open(path, "rb") as fh:
        n_lines = sum(1 for line in fh if line.strip())
    return max(0, n_lines - 1)


def _subdirs(path: str) -> set:
    if not os.path.isdir(path):
        return set()
    return {name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))}


def _write_yaml_atomic(cfg: dict, path: str) -> None:
    import yaml

    partial = path + ".partial"
    with open(partial, "w") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False)
    os.replace(partial, path)


def _require_cellpose_class() -> None:
    """BIDCell 1.0.x segments nuclei with ``cellpose.models.Cellpose(model_type='cyto')``.

    cellpose 4 removed that class, and the failure otherwise surfaces as a bare AttributeError from
    inside BIDCell's preprocessing. Say what is wrong with the environment before starting.
    """
    try:
        import cellpose
        from cellpose import models as cellpose_models
    except ImportError as exc:
        raise ImportError(f"BIDCell needs cellpose for its nuclei step, and it is not importable: {exc}") from exc
    if hasattr(cellpose_models, "Cellpose"):
        return
    version = getattr(cellpose, "__version__", "") or getattr(cellpose, "version", "") or "unknown"
    try:
        from importlib.metadata import version as dist_version

        version = dist_version("cellpose")
    except Exception:
        pass
    raise RuntimeError(
        f"module 'cellpose.models' has no attribute 'Cellpose': BIDCell calls cellpose.models.Cellpose("
        f"model_type='cyto') for its nuclei step, and cellpose {version} (installed in this env) removed that class "
        "in cellpose 4. This is an environment problem, not an input problem: install cellpose<4 (e.g. "
        "cellpose==3.1.1.2) in the BIDCell env."
    )


# ---------------------------------------------------------------------------------------------- #
# Auto-config
# ---------------------------------------------------------------------------------------------- #
def build_auto_config(
    input_dir: str,
    output_dir: str,
    model_name,
    sc_reference_csv: str = "",
    pos_markers_csv: str = "",
    neg_markers_csv: str = "",
    dapi_path: str = "",
    transcripts_path: str = "",
    elongated_cell_types: str = "",
    total_steps: int = DEFAULT_TOTAL_STEPS,
    patch_size: int = DEFAULT_PATCH_SIZE,
    shift_to_origin: bool = False,
    template_path: str = "",
) -> dict:
    """A complete BIDCell config dict from its Xenium template, or an error naming what is missing.

    Everything is checked here, before BIDCell's pydantic validation or its preprocessing and training
    could fail on it: the three reference files BIDCell requires (there is no default for them), the
    DAPI image and transcript table, the transcript columns, the elongated types against the
    reference, and the step count against the checkpoint predict() will load.
    """
    import yaml

    named = {
        "sc_reference_csv": sc_reference_csv,
        "pos_markers_csv": pos_markers_csv,
        "neg_markers_csv": neg_markers_csv,
    }
    absent = [name for name, value in named.items() if not value]
    if absent:
        raise ValueError(
            "BIDCell needs a single-cell reference and its positive/negative marker tables, and has no default "
            f"for them. Missing: {', '.join(absent)}. Pass sc_reference_csv, pos_markers_csv and neg_markers_csv "
            "(in BIDCell's reference format), or a config_path file that names them."
        )
    for name, value in named.items():
        if not os.path.isfile(value):
            raise FileNotFoundError(f"{name}={value!r} is not a file.")

    fp_dapi = dapi_path or _discover(input_dir, "dapi")
    fp_transcripts = transcripts_path or _discover(input_dir, "transcripts")
    for name, value in (("dapi_path", fp_dapi), ("transcripts_path", fp_transcripts)):
        if not os.path.isfile(value):
            raise FileNotFoundError(f"{name}={value!r} is not a file.")
    _check_dapi_is_2d(fp_dapi, "dapi_path" if dapi_path else f"dapi_path (this image was the one found in {input_dir})")

    if not template_path:
        import bidcell

        template_path = os.path.join(os.path.dirname(bidcell.__file__), "example_params", AUTO_TEMPLATE)
    if not os.path.isfile(template_path):
        raise FileNotFoundError(
            f"BIDCell's Xenium config template is missing at {template_path}; pass a config_path file instead."
        )
    with open(template_path) as fh:
        cfg = yaml.safe_load(fh)

    total_steps = int(total_steps)
    if total_steps < 1:
        raise ValueError(f"total_steps must be a positive number of training steps, got {total_steps}.")
    model_freq = int(cfg.get("training_params", {}).get("model_freq", MODEL_FREQ_DEFAULT))
    if not _checkpoint_is_reachable(total_steps, model_freq):
        raise ValueError(
            f"total_steps={total_steps}: BIDCell saves a checkpoint every {model_freq} steps once training runs past "
            f"{model_freq}, and predicts from the one at step total_steps, so total_steps must be at most "
            f"{model_freq} or a multiple of it."
        )
    patch_size = int(patch_size)
    if patch_size < 1:
        raise ValueError(f"patch_size must be positive, got {patch_size}.")
    _check_patch_size(model_name or "custom", patch_size, "auto")

    columns = cfg["transcripts"]
    wanted = [columns.get("x_col", "x_location"), columns.get("y_col", "y_location"), columns.get("gene_col")]
    header = _read_header(fp_transcripts)
    lacking = [col for col in wanted if col not in header]
    if lacking:
        raise ValueError(
            f"The transcript table {fp_transcripts} has no {', '.join(lacking)} column(s) (it has: "
            f"{', '.join(header[:12])}{' ...' if len(header) > 12 else ''}). The auto-config uses BIDCell's Xenium "
            f"layout ({', '.join(wanted)}); for another platform pass a config_path file."
        )

    elongated = _parse_cell_types(elongated_cell_types)
    ref_types = _reference_cell_types(sc_reference_csv)
    unknown = [ct for ct in elongated if ct not in ref_types]
    if unknown:
        raise ValueError(
            f"elongated_cell_types names {', '.join(repr(u) for u in unknown)}, which the reference "
            f"{sc_reference_csv} does not have (its cell types: {', '.join(ref_types)}). BIDCell looks each one up "
            "in the reference and stops on a miss."
        )

    data_dir = os.path.join(output_dir, AUTO_DATA_SUBDIR)
    stale = os.path.join(data_dir, "cell_gene_matrices", "nuclei", "expr_mat.csv")
    if os.path.exists(stale):
        raise FileExistsError(
            f"{output_dir} already holds a previous BIDCell run ({stale}). BIDCell reuses that nuclei matrix instead "
            "of recomputing it, so a second run into this output_dir could mix two runs' inputs. Use a new output_dir."
        )
    os.makedirs(data_dir, exist_ok=True)

    cfg["cpus"] = int(cpu_budget())
    cfg["files"] = {
        "data_dir": os.path.abspath(data_dir),
        "fp_dapi": os.path.abspath(fp_dapi),
        "fp_transcripts": os.path.abspath(fp_transcripts),
        "fp_ref": os.path.abspath(sc_reference_csv),
        "fp_pos_markers": os.path.abspath(pos_markers_csv),
        "fp_neg_markers": os.path.abspath(neg_markers_csv),
    }
    cfg["transcripts"]["shift_to_origin"] = bool(shift_to_origin)
    cfg["model_params"]["name"] = model_name or "custom"
    cfg["model_params"]["patch_size"] = patch_size
    cfg["model_params"]["elongated"] = elongated
    cfg["training_params"]["total_steps"] = total_steps
    cfg["testing_params"]["test_step"] = total_steps
    return {"config": cfg, "template": template_path}


def _auto_only_values_set(values: dict) -> list:
    return [name for name, default in AUTO_ONLY_DEFAULTS.items() if values.get(name, default) != default]


# ---------------------------------------------------------------------------------------------- #
# Outputs, where BIDCell really wrote them
# ---------------------------------------------------------------------------------------------- #
def _experiment_id(config, before: set) -> str:
    """The model_outputs/<id> predict() used, resolved the way BIDCellModel.predict() resolves it."""
    exp_root = os.path.join(config.files.data_dir, "model_outputs")
    dir_id = config.experiment_dirs.dir_id
    if dir_id and dir_id != "last":
        return dir_id
    new = sorted(_subdirs(exp_root) - set(before))
    if len(new) == 1:
        return new[0]
    candidates = sorted(_subdirs(exp_root))
    if not candidates:
        raise RuntimeError(f"BIDCell finished but wrote no experiment directory under {exp_root}.")
    return candidates[-1]


def collect_outputs(config, experiment_id: str) -> dict:
    """Paths of the files BIDCell wrote for ``experiment_id``; a missing segmentation is an error."""
    data_dir = config.files.data_dir
    stem = f"epoch_{config.testing_params.test_epoch}_step_{config.testing_params.test_step}"
    exp_dir = os.path.join(data_dir, "model_outputs", experiment_id)
    test_dir = os.path.join(exp_dir, config.experiment_dirs.test_output_dir)
    seg = os.path.join(test_dir, stem + "_connected.tif")
    cgm = os.path.join(data_dir, config.files.dir_cgm, experiment_id, config.files.fp_expr)
    if not os.path.isfile(seg):
        listing = sorted(os.listdir(test_dir))[:12] if os.path.isdir(test_dir) else []
        raise RuntimeError(
            f"BIDCell finished without writing its cell segmentation {seg} (in {test_dir}: {listing or 'nothing'})."
        )
    if not os.path.isfile(cgm):
        raise RuntimeError(f"BIDCell wrote {seg} but not its cell x gene matrix {cgm}; the run did not complete.")

    files: dict = {
        "segmentation_masks": [seg],
        "cell_segmentation_tif": seg,
        "cell_stats_csv": cgm,
        "cell_gene_matrix_csv": cgm,
    }
    optional = {
        "raw_prediction_tif": os.path.join(test_dir, stem + ".tif"),
        "model_checkpoint": os.path.join(exp_dir, config.experiment_dirs.model_dir, stem + ".pth"),
        "nuclei_mask_tif": os.path.join(data_dir, config.files.fp_nuclei),
        "nuclei_gene_matrix_csv": os.path.join(data_dir, config.files.dir_cgm, "nuclei", config.files.fp_expr),
        "nuclei_cell_types_h5": os.path.join(data_dir, config.files.fp_nuclei_anno),
    }
    for key, path in optional.items():
        if os.path.isfile(path):
            files[key] = path
    learning_rate_png = os.path.join(exp_dir, "LR.png")
    if os.path.isfile(learning_rate_png):
        files["images"] = [learning_rate_png]
    patch_dir = os.path.join(test_dir, stem + "_connected")
    if os.path.isdir(patch_dir):
        files["postprocess_patch_dir"] = patch_dir
    files["experiment_dir"] = exp_dir
    files["data_dir"] = data_dir
    return files


def _count_training_patches(config):
    """How many transcript-map patches BIDCell cut the image into (it trains on 80% of them); None if unknown."""
    files = config.files
    patch = int(config.model_params.patch_size)
    folder = os.path.join(
        files.data_dir,
        getattr(files, "dir_out_maps", "expr_maps"),
        f"{getattr(files, 'dir_patches', 'expr_maps_input_patches_')}{patch}x{patch}_shift_0",
    )
    if not os.path.isdir(folder):
        return None
    return sum(1 for name in os.listdir(folder) if name.endswith(".hdf5"))


def _check_checkpoint(config, experiment_id: str) -> None:
    """After train(): is the checkpoint predict() will load really there?"""
    if config.testing_params.test_epoch < 0:
        return
    stem = f"epoch_{config.testing_params.test_epoch}_step_{config.testing_params.test_step}"
    models_dir = os.path.join(config.files.data_dir, "model_outputs", experiment_id, config.experiment_dirs.model_dir)
    wanted = os.path.join(models_dir, stem + ".pth")
    if os.path.isfile(wanted):
        return
    saved = sorted(n for n in os.listdir(models_dir) if n.endswith(".pth")) if os.path.isdir(models_dir) else []
    test_step = config.testing_params.test_step
    if not _checkpoint_is_reachable(test_step, config.training_params.model_freq):
        why = (
            f"train() saves every min(model_freq={config.training_params.model_freq}, test_step) steps, and "
            f"{test_step} is not one of them. Set test_step to a multiple of model_freq."
        )
    else:
        n_patches = _count_training_patches(config)
        counted = (
            f"this image gave {n_patches} patches, {int(0.8 * n_patches)} of them for training, "
            if n_patches is not None
            else ""
        )
        why = (
            f"Training takes one step per training patch (80% of the image's patches): {counted}so training ended "
            f"before step {test_step}. Lower total_steps (the auto-config predicts from test_step = total_steps) "
            "below the number of training patches, or lower testing_params.test_step in config_path."
        )
    raise RuntimeError(
        f"BIDCell trained but saved no checkpoint at step {test_step} ({wanted}), which predict() loads; it saved: "
        f"{', '.join(saved) or 'none'}. {why}"
    )


# ---------------------------------------------------------------------------------------------- #
# Main pipeline
# ---------------------------------------------------------------------------------------------- #
def run_bidcell_segmentation(
    input_dir: str,
    output_dir: str,
    config_path: str,
    model_type: str,
    sc_reference_csv: str = "",
    pos_markers_csv: str = "",
    neg_markers_csv: str = "",
    dapi_path: str = "",
    transcripts_path: str = "",
    elongated_cell_types: str = "",
    total_steps: int = DEFAULT_TOTAL_STEPS,
    patch_size: int = DEFAULT_PATCH_SIZE,
    shift_to_origin: bool = False,
) -> dict[str, Any]:
    """Run BIDCell (preprocess -> train -> predict) and report what it wrote, where it wrote it."""
    import matplotlib

    matplotlib.use("Agg")

    os.makedirs(output_dir, exist_ok=True)

    log(f"input_dir   = {input_dir}")
    log(f"output_dir  = {output_dir}")
    log(f"config_path = {config_path or '(auto-config)'}")
    log(f"model_type  = {model_type}")

    # Preflight checks
    preflight_check(
        inputs={"input_dir": input_dir},
        output_dir=output_dir,
        packages=["bidcell"],
    )

    out = WorkerOutput("bidcell", task="segmentation")
    requested_name = _resolve_model_name(model_type)
    auto_values = {
        "sc_reference_csv": sc_reference_csv,
        "pos_markers_csv": pos_markers_csv,
        "neg_markers_csv": neg_markers_csv,
        "dapi_path": dapi_path,
        "transcripts_path": transcripts_path,
        "elongated_cell_types": elongated_cell_types,
        "total_steps": total_steps,
        "patch_size": patch_size,
        "shift_to_origin": bool(shift_to_origin),
    }

    template = ""
    if config_path:
        # A config_path that is not a file is a typo, not a request for the auto-config.
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"config_path={config_path!r} is not a file. Fix the path, or leave config_path empty to use the "
                "auto-config (which then needs sc_reference_csv, pos_markers_csv and neg_markers_csv)."
            )
        effective_config = os.path.abspath(config_path)
        config_source = "config_path"
        record_ignored(
            out,
            _auto_only_values_set(auto_values) + ["input_dir"],
            "config_path was given, so BIDCell read its inputs, reference and training settings from that file",
        )
        log(f"Using config from: {effective_config}")
    else:
        built = build_auto_config(input_dir, output_dir, requested_name, **auto_values)
        effective_config = os.path.abspath(os.path.join(output_dir, AUTO_CONFIG_NAME))
        _write_yaml_atomic(built["config"], effective_config)
        template = built["template"]
        config_source = "auto"
        log(f"Auto-generated config at: {effective_config} (template {template})")

    _require_cellpose_class()

    log("Importing BIDCell...")
    from bidcell import BIDCellModel

    log("Initializing BIDCell model...")
    model = BIDCellModel(config_file=effective_config)
    config = model.config
    if requested_name is not None:
        config.model_params.name = requested_name
    model_name = config.model_params.name
    _check_patch_size(model_name, config.model_params.patch_size, config_source)
    if config_source == "config_path" and getattr(config.files, "fp_dapi", None):
        _check_dapi_is_2d(config.files.fp_dapi, "files.fp_dapi in your config_path file")

    exp_root = os.path.join(config.files.data_dir, "model_outputs")
    before = _subdirs(exp_root)

    # BIDCellModel.run_pipeline() is exactly these three calls; running them one by one lets the
    # checkpoint predict() needs be checked before prediction starts.
    log("BIDCell: preprocessing (nuclei, expression maps, patches, pre-annotation)...")
    model.preprocess()
    log("BIDCell: training...")
    model.train()
    new_runs = sorted(_subdirs(exp_root) - before)
    if new_runs and config.experiment_dirs.dir_id in ("", "last"):
        _check_checkpoint(config, new_runs[-1])
    log("BIDCell: predicting and post-processing...")
    model.predict()
    log("BIDCell pipeline completed.")

    experiment_id = _experiment_id(config, before)
    files = collect_outputs(config, experiment_id)
    n_cells = _count_rows(files["cell_gene_matrix_csv"])
    nuclei_csv = files.get("nuclei_gene_matrix_csv", "")
    n_nuclei = _count_rows(nuclei_csv) if nuclei_csv else None

    out.set_data(
        input_dir=input_dir,
        fp_dapi=config.files.fp_dapi,
        fp_transcripts=config.files.fp_transcripts,
    )
    out.add_output_files(files)
    out.add_output_file("config_yaml", effective_config)
    out.add_output_file("output_dir", output_dir)
    record_method(out, _describe_model(model_name))
    out.add_params(
        {
            "model_type": model_type,
            "model_name": model_name,
            "config_path": effective_config,
            "config_source": config_source,
            "config_template": template or None,
            "data_dir": config.files.data_dir,
            "fp_ref": config.files.fp_ref,
            "fp_pos_markers": config.files.fp_pos_markers,
            "fp_neg_markers": config.files.fp_neg_markers,
            "elongated_cell_types": list(config.model_params.elongated),
            "patch_size": config.model_params.patch_size,
            "total_steps": config.training_params.total_steps,
            "test_epoch": config.testing_params.test_epoch,
            "test_step": config.testing_params.test_step,
            "shift_to_origin": config.transcripts.shift_to_origin,
            "cpus": config.cpus,
            "experiment_id": experiment_id,
        }
    )
    n_file_entries = sum(
        len(v) if isinstance(v, list) else 1
        for k, v in files.items()
        if k not in ("experiment_dir", "data_dir", "postprocess_patch_dir")
    )
    out.set_summary(
        n_cells_segmented=n_cells,
        n_nuclei_detected=n_nuclei,
        n_mask_files=len(files["segmentation_masks"]),
        output_file_count=n_file_entries,
    )

    analysis_parts = [
        f"{_describe_model(model_name)} segmented {n_cells} cells"
        + (f" from {n_nuclei} Cellpose nuclei" if n_nuclei is not None else "")
        + f" ({config.training_params.total_steps} training steps, checkpoint at step "
        f"{config.testing_params.test_step}).",
        f"Cell label image: {files['cell_segmentation_tif']}; cell x gene counts: {files['cell_gene_matrix_csv']}.",
    ]
    if config_source == "config_path":
        analysis_parts.append(
            f"Configured by {effective_config}; BIDCell wrote its outputs under that file's data_dir "
            f"({config.files.data_dir}), not under output_dir."
        )
    else:
        analysis_parts.append(
            f"Configured by the auto-config (BIDCell's {AUTO_TEMPLATE} template) written to {effective_config}."
        )
    if n_cells == 0:
        out.add_warning("BIDCell produced a segmentation with no cells.")
    out.set_analysis(" ".join(analysis_parts))

    return out.to_dict()


def main() -> None:
    args = parse_args()

    # Redirect all stdout during heavy work to stderr
    orig_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        try:
            result = run_bidcell_segmentation(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                config_path=args.config_path,
                model_type=args.model_type,
                sc_reference_csv=args.sc_reference_csv,
                pos_markers_csv=args.pos_markers_csv,
                neg_markers_csv=args.neg_markers_csv,
                dapi_path=args.dapi_path,
                transcripts_path=args.transcripts_path,
                elongated_cell_types=args.elongated_cell_types,
                total_steps=args.total_steps,
                patch_size=args.patch_size,
                shift_to_origin=args.shift_to_origin,
            )
        except SystemExit as e:
            # BIDCell stops with sys.exit() in a dozen places (e.g. preannotate() on a gene mismatch
            # between reference and transcripts, with no message at all). SystemExit is not an
            # Exception, so without this the worker died with no JSON.
            log("BIDCell called sys.exit():")
            traceback.print_exc(file=sys.stderr)
            detail = "" if e.code is None or e.code == 0 else f": {e.code}"
            result = WorkerOutput.error(
                "bidcell",
                f"BIDCell stopped with sys.exit(){detail} (see the traceback for the step; with no message this is "
                "usually preannotate() finding genes in the transcripts that the reference lacks, or vice versa)",
                task="segmentation",
                exc=e,
            )
        except Exception as e:
            log("ERROR while running BIDCell:")
            traceback.print_exc(file=sys.stderr)
            result = WorkerOutput.error("bidcell", str(e), task="segmentation", exc=e)
    finally:
        sys.stdout = orig_stdout

    # Final JSON to stdout
    print(json.dumps(result))


if __name__ == "__main__":
    main()
