#!/usr/bin/env python3
"""Seurat MCP wrapper for SpatialOmicsLab (delegates to R worker).

Supports flexible input formats — auto-converts to what the R worker expects:
  - .h5ad  → exported to 10x MTX directory via scanpy
  - .h5    → exported to 10x MTX directory via scanpy; for seurat_spatial_qc_cluster, the
             filtered_feature_bc_matrix.h5 (or <sample>_filtered_feature_bc_matrix.h5) of a Space
             Ranger folder has that folder staged instead
  - .csv/.tsv/.txt → exported to 10x MTX directory via scanpy
  - .loom  → exported to 10x MTX directory via scanpy
  - .rds   → passed directly to R worker
  - 10x directory (matrix.mtx + barcodes + features) → passed directly
  - Visium Space Ranger directory → re-staged with one positions file for Load10X_Spatial

The text matrices may be gzipped (`.csv.gz`, `.tsv.gz`, `.txt.gz`), which is how GEO distributes
them. The binary formats may not: h5py and loompy are handed a path and neither decompresses one.

A counts directory built here for the single-cell modes holds the Gene Expression features only: an
.h5ad or .h5 that also carries other feature types (a CytAssist protein run's 35 Antibody Capture
features) has them left out and reported (``params.feature_types_dropped`` and a warning), because
Seurat would otherwise cluster them as genes. The spatial mode keeps them, as assays of their own. A
feature with no type (``var['feature_types']`` empty or NaN, as an outer-join concat leaves it) is a
gene, exactly as a matrix with no ``feature_types`` column is.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from base_mcp import create_mcp, resolve_worker_script, run_worker_cli
from worker_utils import keep_in_tissue, sniff_tabular_sep, uncompressed_suffix

TOOL_NAME = "seurat"
# Every other worker-launching portal resolves both paths through ``get_worker_paths``, which reads
# ``{PREFIX}_PYTHON``. seurat's interpreter override is spelled ``SEURAT_RSCRIPT`` -- recorded that
# way in ``install/recipes/tool_specs/seurat.yaml`` because ``sog_install.capture`` parses it from these two lines,
# and written back by ``sog_install.wiring`` -- so the prefix form would rename a live override. The
# worker path still needs ``get_worker_paths``' repair for a ``SEURAT_WORKER`` that names another
# machine's checkout, so it is applied here directly.
WORKER_RSCRIPT = os.environ.get("SEURAT_RSCRIPT", "/opt/conda/envs/seurat_env/bin/Rscript")
WORKER_SCRIPT = resolve_worker_script(
    os.environ.get("SEURAT_WORKER", os.path.join(os.path.dirname(os.path.abspath(__file__)), "seurat_worker.R"))
)

mcp = create_mcp(TOOL_NAME)


# ---------------------------------------------------------------------------
# Data preparation: auto-convert various formats to 10x MTX directory
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    sys.stderr.write(f"[seurat-mcp] {msg}\n")
    sys.stderr.flush()


def _is_10x_dir(path: str) -> bool:
    """Check if a path is a 10x counts directory (matrix.mtx + barcodes + features)."""
    p = Path(path)
    if not p.is_dir():
        return False
    contents = {f.name.lower() for f in p.iterdir()}
    has_matrix = any(f.startswith("matrix.mtx") for f in contents)
    has_barcodes = any(f.startswith("barcodes.tsv") for f in contents)
    has_features = any(f.startswith("features.tsv") or f.startswith("genes.tsv") for f in contents)
    return has_matrix and has_barcodes and (has_features or has_barcodes)


def _is_visium_dir(path: str) -> bool:
    """Check if a path is a Visium Space Ranger output directory the R loader can read.

    ``Load10X_Spatial`` reads ``filtered_feature_bc_matrix.h5`` through ``Read10X_h5`` and nothing
    else, so a folder holding only the MTX directory is not a Visium dir for this purpose -- it
    used to be accepted here and then died in R with ``File not found``.
    """
    p = Path(path)
    if not p.is_dir():
        return False
    return (p / "spatial").is_dir() and (p / "filtered_feature_bc_matrix.h5").is_file()


POSITIONS_FILES = ("tissue_positions.csv", "tissue_positions_list.csv")


def _clear_staging_dir(path: str) -> None:
    """Remove this tool's own staging folder without ever following a link out of it.

    ``shutil.rmtree`` unlinks symlinks it finds inside the tree rather than descending into them;
    a staging path that is itself a link is unlinked, not emptied.
    """
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def _replace_file(path: str, write) -> None:
    """Write ``path`` via ``<path>.partial`` + ``os.replace`` so an existing link is replaced, never written through."""
    tmp = path + ".partial"
    if os.path.lexists(tmp):
        os.remove(tmp)
    write(tmp)
    os.replace(tmp, path)


def _stage_visium_dir(source: str, output_dir: str, counts_h5: str | None = None) -> dict[str, Any]:
    """Re-stage a Space Ranger folder so Seurat's loader can read it.

    Seurat 5.3.1's ``Read10X_Image`` globs ``*tissue_positions*`` and then tests
    ``file_ext(filename) == "parquet"`` on the result. Every standard library folder ships BOTH
    ``tissue_positions.csv`` and ``tissue_positions_list.csv``, so the glob returns two paths and
    R fails with "the condition has length > 1". The staged folder holds a symlink to the counts
    matrix and COPIES of the small spatial files (scalefactors, images, exactly one positions file
    -- the headed ``tissue_positions.csv`` when present). They are copies, not links, because a later
    ``.h5ad`` run into the same ``output_dir`` rebuilds this folder: a link there would let that run
    write through it onto the library sample's own files (found 2026-09-30). The matrix link is safe:
    nothing opens it for writing, and ``_write_10x_h5`` replaces a path with ``os.replace``.

    ``counts_h5`` names the counts file when the caller was handed the file rather than the folder
    (``_stage_visium_h5``); the link is made to that file under the name the loader opens.
    """
    src = Path(source).resolve()
    counts = Path(counts_h5).resolve() if counts_h5 else src / "filtered_feature_bc_matrix.h5"
    spatial_src = src / "spatial"
    positions = [n for n in POSITIONS_FILES if (spatial_src / n).is_file()]
    if not positions:
        return {"error": f"{source}/spatial holds neither {' nor '.join(POSITIONS_FILES)}"}
    if not (spatial_src / "tissue_lowres_image.png").is_file():
        return {
            "error": (
                f"{source}/spatial has no tissue_lowres_image.png; Seurat's Visium loader (png::readPNG) needs it. "
                "Run the tool on the .h5ad instead if it carries the image in uns['spatial']."
            )
        }
    staged = Path(output_dir) / "_seurat_visium_input"
    _clear_staging_dir(str(staged))
    (staged / "spatial").mkdir(parents=True)
    os.symlink(counts, staged / "filtered_feature_bc_matrix.h5")
    for name in ("scalefactors_json.json", "tissue_lowres_image.png", "tissue_hires_image.png", positions[0]):
        if (spatial_src / name).is_file():
            shutil.copyfile(spatial_src / name, staged / "spatial" / name)
    return {
        "data_dir": str(staged),
        "format_detected": "visium_spaceranger",
        "conversion": "staged_single_positions_file",
        "positions_file": positions[0],
        "positions_files_in_source": positions,
    }


SPATIAL_INPUTS = (
    "a Visium Space Ranger folder holding filtered_feature_bc_matrix.h5 and spatial/, that folder's "
    "filtered_feature_bc_matrix.h5 (the folder around it is staged), or an .h5ad with obsm['spatial'] and a "
    "tissue image in uns['spatial']"
)


def _stage_visium_h5(h5_path: str, output_dir: str) -> dict[str, Any]:
    """A Space Ranger counts ``.h5`` handed to the spatial tool in place of its folder.

    Seurat's Visium loader needs the folder: the counts matrix AND ``spatial/`` beside it. The ``.h5``
    route used to run a full MTX conversion that no spatial run reads and then fail with "Ensure the
    input has spatial coordinates", while the tool's own docstring advertised ".h5 counts + spatial/".
    The folder around a ``filtered_feature_bc_matrix.h5`` (10x downloads prefix the name with the
    sample) is staged exactly as the folder itself would be, with the link made to the file given. A
    ``raw_feature_bc_matrix.h5`` holds every barcode on the slide, including the background the loader
    assumes is gone, and any other ``.h5`` cannot be told from one; both are refused before any work.
    """
    p = Path(h5_path)
    folder = os.path.dirname(os.path.abspath(h5_path))
    name = p.name.lower()
    if "raw_feature_bc_matrix" in name:
        return {
            "error": (
                f"{p.name} is Space Ranger's raw matrix: it holds every barcode on the slide, including the "
                "off-tissue background that Seurat's Visium loader assumes was filtered out. Pass the "
                f"filtered_feature_bc_matrix.h5 or its folder. seurat_spatial_qc_cluster reads {SPATIAL_INPUTS}."
            )
        }
    if not name.endswith("filtered_feature_bc_matrix.h5") or not os.path.isdir(os.path.join(folder, "spatial")):
        return {
            "error": (
                f"{p.name} is a counts file with no Visium folder to stage around it (a spatial/ folder beside a "
                f"file named *filtered_feature_bc_matrix.h5). seurat_spatial_qc_cluster reads {SPATIAL_INPUTS}."
            )
        }
    staged = _stage_visium_dir(folder, output_dir, counts_h5=h5_path)
    if "error" not in staged:
        staged["format_detected"] = "10x_h5_in_visium_folder"
    return staged


GENE_EXPRESSION = "Gene Expression"
# What an unset label reads as once it is text: a missing value ('nan', pandas' '<NA>', 'None') or
# nothing at all. An outer-join concat leaves these for the features only some inputs labelled.
_UNLABELLED = frozenset({"", "nan", "none", "<na>"})


def _is_unlabelled(label: Any) -> bool:
    return label is None or (isinstance(label, float) and label != label) or str(label).strip().lower() in _UNLABELLED


def _feature_type_label(label: Any) -> str:
    """One feature's type, with an unlabelled feature read as a gene.

    A feature with no type is a gene, exactly as a matrix with no ``feature_types`` column is. The
    library's Tonsil BCellRepertoire single-cell reference (an outer-join concat) labels 10,230 of its
    10,237 genes 'nan'; read as a type of its own, 'nan' left Seurat 7 features.
    """
    if isinstance(label, bytes):
        label = label.decode("utf-8", "replace")
    return GENE_EXPRESSION if _is_unlabelled(label) else str(label).strip()


def _normalised_feature_types(var) -> list[str]:
    """Each feature's type from ``var['feature_types']`` (all genes when there is no such column).

    Every writer of a feature-type field here (features.tsv.gz, the staged ``.h5``) and the gene-only
    filter read the types through this, so no 'nan' label reaches Read10X, which would split it into a
    matrix of its own.
    """
    if "feature_types" not in var.columns:
        return [GENE_EXPRESSION] * len(var)
    return [_feature_type_label(v) for v in var["feature_types"].tolist()]


def _gene_expression_only(adata) -> tuple[Any, dict[str, int]]:
    """``(adata restricted to its Gene Expression features, {other feature type: n left out})``.

    An .h5ad written from a multi-modal Space Ranger run keeps every feature type in one matrix and
    names them in ``var['feature_types']``. The MTX this module writes for the single-cell modes used
    to label every feature "Gene Expression", so a CytAssist protein sample's antibodies were
    normalised, scaled and clustered as genes -- the defect the R worker refuses by name for a counts
    directory. No ``feature_types`` column, or no label on a feature, means a gene, as before; only a
    feature with a real other label (Antibody Capture, Peaks, CRISPR Guide Capture) is left out.
    """
    from collections import Counter

    import numpy as np

    if "feature_types" not in adata.var.columns:
        return adata, {}
    n_unlabelled = sum(_is_unlabelled(v) for v in adata.var["feature_types"].tolist())
    if n_unlabelled:
        _log(f"{n_unlabelled} feature(s) have no var['feature_types'] label and are read as Gene Expression")
    types = _normalised_feature_types(adata.var)
    is_gex = np.array([t == GENE_EXPRESSION for t in types], dtype=bool)
    if is_gex.all():
        return adata, {}
    counts = dict(Counter(t for t, g in zip(types, is_gex) if not g))
    if not is_gex.any():
        raise ValueError(
            f"var['feature_types'] holds no Gene Expression feature ({counts}); Seurat's clustering modes need genes."
        )
    _log(f"Leaving out {int((~is_gex).sum())} feature(s) that are not Gene Expression: {counts}")
    return adata[:, is_gex].copy(), counts


def _h5_feature_type_counts(h5_path: str) -> tuple[dict[str, int], int]:
    """``({type: n features}, n unlabelled)`` for a 10x ``.h5`` (``({}, 0)`` for a v2 file, which has no types).

    An unlabelled feature is counted as Gene Expression (``_feature_type_label``).
    """
    from collections import Counter

    import h5py

    counts: Counter = Counter()
    n_unlabelled = 0
    with h5py.File(h5_path, "r") as f:
        for group in f.values():
            features = group.get("features") if isinstance(group, h5py.Group) else None
            if isinstance(features, h5py.Group) and "feature_type" in features:
                for raw in features["feature_type"][:]:
                    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
                    n_unlabelled += int(_is_unlabelled(text))
                    counts[_feature_type_label(text)] += 1
    return dict(counts), n_unlabelled


def _report_feature_types_dropped(result: dict[str, Any], dropped: dict[str, Any], where: str = "") -> None:
    """Publish what the converter left out of the counts directory (params, a warning, the analysis)."""
    if not dropped:
        return
    result.setdefault("params", {})["feature_types_dropped"] = dropped
    text = (
        f"the input{where} holds feature types other than Gene Expression and they were left out of the counts "
        f"directory Seurat clustered: {dropped}"
    )
    warnings = result.get("warnings")
    result["warnings"] = [*(warnings if isinstance(warnings, list) else []), text]
    if isinstance(result.get("analysis"), str):
        result["analysis"] = f"{result['analysis']} Note: {text}."


def _prepare_data(data_path: str, output_dir: str, mode: str = "sc") -> dict[str, Any]:
    """Auto-detect input format and convert to what the R worker needs.

    Returns dict with:
      - counts_dir: path to 10x MTX directory (for sc modes)
      - data_dir: path to Visium directory (for spatial modes)
      - format_detected: what was detected
      - conversion: what was done
    """
    p = Path(data_path)

    if not p.exists():
        return {"error": f"Path does not exist: {data_path}"}

    # Already a 10x counts directory
    if _is_10x_dir(data_path):
        return {"counts_dir": data_path, "format_detected": "10x_mtx_dir", "conversion": "none"}

    # Visium Space Ranger directory: never handed to R as-is (see _stage_visium_dir)
    if _is_visium_dir(data_path):
        if mode == "spatial":
            return _stage_visium_dir(data_path, output_dir)
        return {"data_dir": data_path, "format_detected": "visium_spaceranger", "conversion": "none"}

    # Directory containing filtered_feature_bc_matrix/
    if p.is_dir():
        mtx_subdir = p / "filtered_feature_bc_matrix"
        if mtx_subdir.is_dir() and _is_10x_dir(str(mtx_subdir)):
            return {"counts_dir": str(mtx_subdir), "format_detected": "10x_mtx_subdir", "conversion": "none"}

    suffix = p.suffix.lower()
    # `.csv.gz` answers `.gz` to `Path.suffix`, and that is the form nearly every GEO supplementary
    # matrix is distributed in — it used to fall through every branch to "Unsupported input format"
    # even though pandas reads it without being told. Only the text branch is allowed to look
    # through the compression: routing `.h5ad.gz` or `.loom.gz` by their inner suffix would hand
    # compressed bytes to h5py, trading an accurate refusal for "file signature not found".
    text_suffix = uncompressed_suffix(p).lower()

    # .rds file — pass directly
    if suffix == ".rds":
        return {"rds_path": data_path, "format_detected": "seurat_rds", "conversion": "none"}

    # .h5ad file — convert via scanpy
    if suffix == ".h5ad":
        return _convert_h5ad_to_10x(data_path, output_dir, mode)

    # .h5 file — 10x HDF5 counts. The spatial mode needs the folder around it, never an MTX copy.
    if suffix == ".h5":
        if mode == "spatial":
            return _stage_visium_h5(data_path, output_dir)
        return _convert_h5_to_10x(data_path, output_dir)

    # .csv/.tsv/.txt — expression matrix. The separator comes from the file's own header, not from
    # its name: `.txt` is accepted here and a `.txt` counts matrix is conventionally tab-delimited
    # (the GEO supplementary format), which read with a comma parses to zero columns rather than
    # raising. The suffix supplies only the fallback, for a header with nothing to split on.
    if text_suffix in (".csv", ".tsv", ".txt"):
        sep = sniff_tabular_sep(p, default="\t" if text_suffix in (".tsv", ".txt") else ",")
        return _convert_csv_to_10x(data_path, output_dir, sep=sep)

    # .loom file
    if suffix == ".loom":
        return _convert_loom_to_10x(data_path, output_dir)

    return {
        "error": (
            f"Unsupported input format: {p.name}. Accepted: .h5ad, .h5, .csv, .tsv, .txt, .loom, "
            f".rds, 10x directory, Visium directory. Text matrices may be gzipped "
            f"(.csv.gz, .tsv.gz, .txt.gz); the binary formats must be decompressed first."
        )
    }


def _convert_h5ad_to_10x(h5ad_path: str, output_dir: str, mode: str = "sc") -> dict[str, Any]:
    """Convert h5ad to 10x MTX directory."""
    import scanpy as sc
    import scipy.sparse as sp

    _log(f"Converting h5ad to 10x format: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    adata.var_names_make_unique()

    # Use raw counts if available
    if adata.raw is not None:
        adata = adata.raw.to_adata()

    # The single-cell modes cluster this MTX, so it holds genes only; the spatial mode reads the .h5
    # built below, which keeps every feature type for the R worker to separate and report.
    feature_types_dropped: dict[str, int] = {}
    mtx_adata = adata
    if mode != "spatial":
        try:
            mtx_adata, feature_types_dropped = _gene_expression_only(adata)
        except ValueError as exc:
            return {"error": str(exc)}

    # Ensure sparse integer matrix
    X = mtx_adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    mtx_dir = os.path.join(output_dir, "_seurat_10x_input")
    os.makedirs(mtx_dir, exist_ok=True)

    # Write 10x-format files. Each .gz lands through <name>.partial + os.replace, so a reader never
    # sees half a file and an existing link at that name is replaced rather than written through.
    import gzip

    from scipy.io import mmwrite

    def _gzip_into_place(plain: str) -> None:
        def _write(tmp: str) -> None:
            with open(plain, "rb") as f_in, gzip.open(tmp, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        _replace_file(plain + ".gz", _write)
        os.remove(plain)

    # matrix.mtx.gz
    mtx_path = os.path.join(mtx_dir, "matrix.mtx")
    mmwrite(mtx_path, X.T)  # 10x format is genes x cells
    _gzip_into_place(mtx_path)

    # barcodes.tsv.gz
    barcodes_path = os.path.join(mtx_dir, "barcodes.tsv")
    with open(barcodes_path, "w") as f:
        f.write("\n".join(mtx_adata.obs_names.tolist()) + "\n")
    _gzip_into_place(barcodes_path)

    # features.tsv.gz (gene_id \t gene_name \t feature_type)
    features_path = os.path.join(mtx_dir, "features.tsv")
    gene_ids = (
        mtx_adata.var.get("gene_ids", mtx_adata.var_names).tolist()
        if "gene_ids" in mtx_adata.var
        else mtx_adata.var_names.tolist()
    )
    gene_names = mtx_adata.var_names.tolist()
    feature_types = _normalised_feature_types(mtx_adata.var)
    with open(features_path, "w") as f:
        for gid, gname, ftype in zip(gene_ids, gene_names, feature_types, strict=True):
            f.write(f"{gid}\t{gname}\t{ftype}\n")
    _gzip_into_place(features_path)

    result = {
        "counts_dir": mtx_dir,
        "format_detected": "h5ad",
        "conversion": "h5ad_to_10x_mtx",
        "n_cells": mtx_adata.n_obs,
        "n_genes": mtx_adata.n_vars,
    }
    if feature_types_dropped:
        result["feature_types_dropped"] = feature_types_dropped

    # For spatial mode, also prepare Visium-like directory structure
    if mode == "spatial" and "spatial" in adata.obsm:
        visium_dir = os.path.join(output_dir, "_seurat_visium_input")
        try:
            built = _build_visium_dir_from_h5ad(adata, mtx_dir, visium_dir)
        except ValueError as exc:
            return {"error": str(exc)}
        result["data_dir"] = visium_dir
        result["conversion"] = "h5ad_to_visium_dir"
        result.update(built)

    return result


def _spatial_library(uns_spatial) -> tuple[str | None, dict]:
    """The one library entry of ``uns['spatial']`` that is a mapping.

    CELLxGENE exports add a scalar ``is_single`` beside the library dict; on the Thymus sample it
    comes first in key order, and ``"scalefactors" in lib_data`` on a ``numpy.bool_`` raised
    ``TypeError``. Only mapping-valued entries are candidates; more than one is refused by name.
    """
    from collections.abc import Mapping

    libs = {k: v for k, v in dict(uns_spatial).items() if isinstance(v, Mapping)}
    if not libs:
        return None, {}
    if len(libs) > 1:
        raise ValueError(
            f"uns['spatial'] holds {len(libs)} libraries ({sorted(libs)}); this converter handles one section"
        )
    ((lib_id, lib),) = libs.items()
    return str(lib_id), lib


def _write_10x_h5(adata, path: str) -> None:
    """Write counts as a 10x Genomics v3 HDF5 file (what ``Read10X_h5`` reads)."""
    import h5py
    import numpy as np
    import scipy.sparse as sp

    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    csc = sp.csc_matrix(X.T)  # genes x barcodes, column-compressed: the 10x layout
    n_genes, n_barcodes = csc.shape
    ids = adata.var["gene_ids"].astype(str).tolist() if "gene_ids" in adata.var else adata.var_names.tolist()
    # An unlabelled feature is written as a gene: Load10X_Spatial files each type as its own layer, and
    # the worker would cluster the labelled genes only and keep the rest as a 'nan' assay.
    feature_types = _normalised_feature_types(adata.var)
    integer_counts = bool(np.allclose(csc.data, np.round(csc.data)))
    partial = path + ".partial"
    with h5py.File(partial, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("barcodes", data=np.array(adata.obs_names.tolist(), dtype="S"))
        g.create_dataset("data", data=csc.data.astype(np.int32 if integer_counts else np.float32))
        g.create_dataset("indices", data=csc.indices.astype(np.int64))
        g.create_dataset("indptr", data=csc.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array([n_genes, n_barcodes], dtype=np.int32))
        feat = g.create_group("features")
        feat.create_dataset("id", data=np.array(ids, dtype="S"))
        feat.create_dataset("name", data=np.array(adata.var_names.tolist(), dtype="S"))
        feat.create_dataset("feature_type", data=np.array(feature_types, dtype="S"))
        feat.create_dataset("genome", data=np.array(["unknown"] * n_genes, dtype="S"))
        feat.create_dataset("_all_tag_keys", data=np.array(["genome"], dtype="S"))
    os.replace(partial, path)


def _build_visium_dir_from_h5ad(adata, mtx_dir: str, visium_dir: str) -> dict[str, Any]:
    """Build a Visium-compatible directory from an h5ad with spatial coords.

    Writes what ``Load10X_Spatial`` really opens: ``filtered_feature_bc_matrix.h5`` (the old code
    copied the MTX directory, which the loader never reads), one ``tissue_positions.csv`` with a
    header, ``scalefactors_json.json``, and the lowres image from ``uns['spatial']``. An h5ad with
    no lowres image is refused here rather than failing later inside ``png::readPNG``.

    A *filtered* matrix holds in-tissue spots only -- that is what Space Ranger writes and what
    Seurat's loader assumes. CELLxGENE exports carry every array spot with ``obs['in_tissue']``
    0/1, so the matrix is written for the in-tissue spots and the positions file keeps every spot
    with its flag, exactly as Space Ranger lays them out.
    """
    import numpy as np

    # Start from an empty folder: an earlier Space Ranger run into this output_dir may have left a
    # link to the library's matrix here, and nothing may ever be written through it.
    _clear_staging_dir(visium_dir)
    spatial_dir = os.path.join(visium_dir, "spatial")
    os.makedirs(spatial_dir, exist_ok=True)

    # The shared rule (worker_utils.keep_in_tissue): 1 / "1" / True / "true" are tissue, and a column
    # that marks no spot as tissue is refused (ValueError, returned as the error). The positions file
    # keeps every spot with its flag, as Space Ranger lays them out.
    tissue, n_supplied, n_out = keep_in_tissue(adata, "spots")
    in_mask = adata.obs_names.isin(tissue.obs_names) if n_out else np.ones(adata.n_obs, dtype=bool)
    _write_10x_h5(tissue, os.path.join(visium_dir, "filtered_feature_bc_matrix.h5"))

    coords = np.asarray(adata.obsm["spatial"])
    array_row = adata.obs.get("array_row", np.zeros(adata.n_obs, dtype=int))
    array_col = adata.obs.get("array_col", np.zeros(adata.n_obs, dtype=int))

    def _write_positions(tmp: str) -> None:
        with open(tmp, "w") as f:
            f.write("barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n")
            for i, barcode in enumerate(adata.obs_names):
                it = int(in_mask[i])
                ar = int(array_row.iloc[i]) if hasattr(array_row, "iloc") else int(array_row[i])
                ac = int(array_col.iloc[i]) if hasattr(array_col, "iloc") else int(array_col[i])
                px = int(coords[i, 0])
                py = int(coords[i, 1])
                f.write(f"{barcode},{it},{ar},{ac},{py},{px}\n")

    _replace_file(os.path.join(spatial_dir, "tissue_positions.csv"), _write_positions)

    sf = {
        "tissue_hires_scalef": 1.0,
        "tissue_lowres_scalef": 0.3,
        "fiducial_diameter_fullres": 144.5,
        "spot_diameter_fullres": 89.4,
    }
    lib_id, lib = _spatial_library(adata.uns.get("spatial", {}))
    images_written: list[str] = []
    if lib:
        if "scalefactors" in lib:
            sf.update({k: float(v) for k, v in dict(lib["scalefactors"]).items()})
        for quality, img_array in dict(lib.get("images", {})).items():
            from PIL import Image

            arr = np.asarray(img_array)
            if arr.dtype != np.uint8:
                scaled = arr * 255 if float(arr.max()) <= 1.0 else arr
                arr = np.clip(scaled, 0, 255).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]  # readPNG wants RGB
            _replace_file(
                os.path.join(spatial_dir, f"tissue_{quality}_image.png"),
                lambda tmp, arr=arr: Image.fromarray(arr).save(tmp, format="PNG"),
            )
            images_written.append(str(quality))
    if "lowres" not in images_written:
        if "hires" in images_written:
            # Seurat reads the lowres file by name; offer the hires one under that name and scale it.
            _replace_file(
                os.path.join(spatial_dir, "tissue_lowres_image.png"),
                lambda tmp: shutil.copyfile(os.path.join(spatial_dir, "tissue_hires_image.png"), tmp),
            )
            sf["tissue_lowres_scalef"] = sf.get("tissue_hires_scalef", sf["tissue_lowres_scalef"])
            images_written.append("lowres(from hires)")
        else:
            raise ValueError(
                "Seurat's Visium loader needs spatial/tissue_lowres_image.png and this h5ad carries no image in "
                "uns['spatial'][<library>]['images']. Give the tool the Space Ranger folder instead."
            )

    def _write_scalefactors(tmp: str) -> None:
        with open(tmp, "w") as f:
            json.dump(sf, f)

    _replace_file(os.path.join(spatial_dir, "scalefactors_json.json"), _write_scalefactors)
    return {
        "library_id": lib_id,
        "images": images_written,
        "n_spots_in_tissue": int(in_mask.sum()),
        "n_spots_out_of_tissue_excluded": n_out,
        "n_spots_supplied": n_supplied,
    }


def _convert_h5_to_10x(h5_path: str, output_dir: str) -> dict[str, Any]:
    """Convert 10x HDF5 counts file to 10x MTX directory."""
    import scanpy as sc

    os.makedirs(output_dir, exist_ok=True)
    _log(f"Converting 10x H5 to MTX: {h5_path}")
    # scanpy reads the Gene Expression features only (gex_only=True); count what that leaves out so the
    # payload can say so rather than the other types vanishing without a word. scanpy's filter keeps
    # the literal label only, so an .h5 with unlabelled genes is read whole and filtered here instead,
    # by the rule the .h5ad route uses (an unlabelled feature is a gene).
    type_counts, n_unlabelled = _h5_feature_type_counts(h5_path)
    if type_counts and GENE_EXPRESSION not in type_counts:
        return {"error": f"{os.path.basename(h5_path)} holds no Gene Expression feature ({type_counts})."}
    dropped = {t: n for t, n in type_counts.items() if t != GENE_EXPRESSION}
    adata = sc.read_10x_h5(h5_path, gex_only=not n_unlabelled)
    if n_unlabelled:
        adata, _ = _gene_expression_only(adata)
        adata.var["feature_types"] = _normalised_feature_types(adata.var)
    adata.var_names_make_unique()

    # Write as h5ad then use h5ad converter
    tmp_h5ad = os.path.join(output_dir, "_temp.h5ad")
    adata.write_h5ad(tmp_h5ad)
    result = _convert_h5ad_to_10x(tmp_h5ad, output_dir)
    os.remove(tmp_h5ad)
    result["format_detected"] = "10x_h5"
    if dropped and "error" not in result:
        result["feature_types_dropped"] = dropped
    return result


def _convert_csv_to_10x(csv_path: str, output_dir: str, sep: str = ",") -> dict[str, Any]:
    """Convert CSV/TSV expression matrix to 10x MTX directory."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    os.makedirs(output_dir, exist_ok=True)
    _log(f"Converting CSV to 10x: {csv_path}")
    df = pd.read_csv(csv_path, sep=sep, index_col=0)

    # A frame with no columns is not a matrix of zero genes, it is a matrix that did not parse: with
    # `index_col=0` every line's single field became the index. Nothing below notices -- the
    # orientation test is false, `.astype(float)` succeeds on an empty array, and a 10x directory
    # holding the cells and none of the genes is written and reported as a detected format. The
    # separator is sniffed from the header now, which covers tab and comma; a semicolon locale
    # export still lands here, and has to say so rather than be handed to Seurat.
    if df.shape[1] == 0 or df.shape[0] == 0:
        return {
            "error": (
                f"Parsed no data columns from {os.path.basename(csv_path)} using separator {sep!r}. "
                "The file is probably delimited by something else -- save it as comma- or "
                "tab-separated, or convert it to .h5ad."
            )
        }

    # Detect orientation: if more columns than rows, likely genes x cells → transpose
    if df.shape[0] < df.shape[1]:
        _log("Detected genes-as-rows, transposing to cells-as-rows")
        df = df.T

    adata = ad.AnnData(
        X=sp.csr_matrix(df.values.astype(float)),
        obs=pd.DataFrame(index=df.index.astype(str)),
        var=pd.DataFrame(index=df.columns.astype(str)),
    )

    tmp_h5ad = os.path.join(output_dir, "_temp.h5ad")
    adata.write_h5ad(tmp_h5ad)
    result = _convert_h5ad_to_10x(tmp_h5ad, output_dir)
    os.remove(tmp_h5ad)
    result["format_detected"] = "csv_matrix"
    return result


def _convert_loom_to_10x(loom_path: str, output_dir: str) -> dict[str, Any]:
    """Convert .loom file to 10x MTX directory."""
    import scanpy as sc

    os.makedirs(output_dir, exist_ok=True)
    _log(f"Converting loom to 10x: {loom_path}")
    adata = sc.read_loom(loom_path)
    adata.var_names_make_unique()

    tmp_h5ad = os.path.join(output_dir, "_temp.h5ad")
    adata.write_h5ad(tmp_h5ad)
    result = _convert_h5ad_to_10x(tmp_h5ad, output_dir)
    os.remove(tmp_h5ad)
    result["format_detected"] = "loom"
    return result


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def seurat_qc_cluster(
    data_path: str,
    output_dir: str,
    project: str = "SeuratQC",
    min_cells: int = 3,
    min_features: int = 200,
    n_hvgs: int = 2000,
    n_pcs: int = 30,
    resolution: float = 0.8,
    umap: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Run Seurat QC + clustering pipeline.

    Accepts flexible input: .h5ad, .h5, .csv/.tsv/.txt (optionally gzipped), .loom, or a 10x
    counts directory (not a processed .rds, and not a Visium Space Ranger folder: use
    seurat_spatial_qc_cluster for that). The data is automatically converted to what Seurat expects.
    A converted .h5ad / .h5 keeps its Gene Expression features only; any other feature type it holds
    (Antibody Capture, say) is left out and listed in ``params.feature_types_dropped`` with a warning. A
    feature with no type (empty or NaN) counts as a gene. A counts directory holding more than one
    feature type is refused by name.

    ``CreateSeuratObject`` drops every cell with fewer than ``min_features`` detected genes, then every
    gene detected in fewer than ``min_cells`` of the cells left. On sparse spatial input the default
    200 can remove most spots, so the payload reports the input size (``data.n_cells_input`` /
    ``n_genes_input``) and what the filters removed (``params.n_cells_dropped`` / ``n_genes_dropped``,
    plus a warning); set ``min_features=0`` to keep every cell. ``seed`` reaches FindClusters
    (``random.seed = seed``) and RunPCA / RunUMAP (``seed.use = 42 + seed``), so seed 0 reproduces
    Seurat's own defaults; ``params.seeds`` shows the value each step received.
    """
    os.makedirs(output_dir, exist_ok=True)
    prep = _prepare_data(data_path, output_dir, mode="sc")
    if "error" in prep:
        return {"status": "error", "error": prep["error"]}

    # If .rds, pass directly
    if prep.get("rds_path"):
        return {
            "status": "error",
            "error": "qc_cluster expects raw counts, not a processed .rds. Use find_markers or dimplot for .rds files.",
        }

    # `_prepare_data` has three success shapes and none of them carries an `error` key, so passing
    # the guard above does not mean a counts directory was produced. `.rds` is caught just above;
    # a Visium directory comes back as `data_dir` and used to reach `prep["counts_dir"]` as a
    # KeyError traceback out of the tool, on an input the module header advertises as accepted.
    counts_dir = prep.get("counts_dir")
    if not counts_dir:
        return {
            "status": "error",
            "error": (
                f"qc_cluster needs a counts directory, and {os.path.basename(data_path)} was detected as "
                f"{prep.get('format_detected')}, which is handed to the worker as-is rather than converted. "
                "Use seurat_spatial_qc_cluster for a Visium Space Ranger directory, or pass its "
                "filtered_feature_bc_matrix.h5 here."
            ),
        }

    args: list[str] = [
        "--mode",
        "qc_cluster",
        "--counts-dir",
        counts_dir,
        "--output-dir",
        output_dir,
        "--project",
        project,
        "--min-cells",
        str(min_cells),
        "--min-features",
        str(min_features),
        "--n-hvgs",
        str(n_hvgs),
        "--n-pcs",
        str(n_pcs),
        "--resolution",
        str(resolution),
        "--seed",
        str(seed),
    ]
    if umap:
        args.append("--umap")

    result = run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)
    if "params" not in result:
        result["params"] = {}
    result["params"]["input_format"] = prep.get("format_detected")
    result["params"]["input_conversion"] = prep.get("conversion")
    _report_feature_types_dropped(result, prep.get("feature_types_dropped") or {})
    return result


@mcp.tool()
def seurat_integrate_qc_cluster(
    data_paths: list[str],
    sample_ids: list[str],
    output_dir: str,
    project: str = "SeuratIntegration",
    n_hvgs: int = 2000,
    n_pcs: int = 30,
    resolution: float = 0.8,
    umap: bool = True,
    seed: int = 0,
    min_cells: int = 3,
    min_features: int = 200,
) -> dict[str, Any]:
    """Run Seurat integration on multiple samples.

    Each entry in data_paths can be .h5ad, .h5, .csv/.tsv/.txt (optionally gzipped), .loom, or a
    10x directory. Not .rds and not a Visium directory: integration needs a counts directory per
    sample, and those two formats are handed to the worker as-is rather than converted. A converted
    .h5ad / .h5 keeps its Gene Expression features only (a feature with no type counts as a gene); the
    other types are listed per sample in ``params.feature_types_dropped`` with a warning.

    Each sample is filtered by ``CreateSeuratObject`` before integration: cells with fewer than
    ``min_features`` detected genes, then genes detected in fewer than ``min_cells`` of the remaining
    cells (defaults 3 / 200, the values the worker always applied). The payload reports the input
    and dropped counts in total (``data.n_cells_input``, ``params.n_cells_dropped`` /
    ``n_genes_dropped``) and per sample (``summary.qc_per_sample``). ``data.n_genes`` counts the RNA
    assay; ``data.n_integration_features`` is the number of anchor features the integrated assay
    holds. The saved object keeps DefaultAssay "integrated"; seurat_find_markers tests its RNA assay.
    ``seed`` reaches FindClusters and RunPCA / RunUMAP as in seurat_qc_cluster (``params.seeds``).
    """
    if not data_paths:
        return {"status": "error", "error": "data_paths is empty."}
    if len(data_paths) != len(sample_ids):
        return {"status": "error", "error": "data_paths and sample_ids must have the same length."}

    os.makedirs(output_dir, exist_ok=True)
    counts_dirs = []
    feature_types_dropped: dict[str, Any] = {}
    for i, dp in enumerate(data_paths):
        sample_out = os.path.join(output_dir, f"_sample_{i}")
        os.makedirs(sample_out, exist_ok=True)
        prep = _prepare_data(dp, sample_out, mode="sc")
        if "error" in prep:
            return {"status": "error", "error": f"Failed to prepare sample {sample_ids[i]}: {prep['error']}"}

        # Same three-shapes problem as in qc_cluster, and here neither of the two shapes without a
        # counts directory was caught: a `.rds` and a Visium directory both left this loop as a
        # KeyError. Integration is the one mode that needs a staged counts dir for every sample.
        counts_dir = prep.get("counts_dir")
        if not counts_dir:
            hint = (
                "pass its filtered_feature_bc_matrix.h5 instead"
                if prep.get("data_dir")
                else "convert it to .h5ad or a 10x counts directory first"
            )
            return {
                "status": "error",
                "error": (
                    f"Failed to prepare sample {sample_ids[i]}: integration needs a counts directory per "
                    f"sample, and {os.path.basename(dp)} was detected as {prep.get('format_detected')}, "
                    f"which is handed to the worker as-is rather than converted -- {hint}."
                ),
            }
        counts_dirs.append(counts_dir)
        if prep.get("feature_types_dropped"):
            feature_types_dropped[str(sample_ids[i])] = prep["feature_types_dropped"]

    args: list[str] = [
        "--mode",
        "integrate_qc_cluster",
        "--output-dir",
        output_dir,
        "--project",
        project,
        "--min-cells",
        str(min_cells),
        "--min-features",
        str(min_features),
        "--n-hvgs",
        str(n_hvgs),
        "--n-pcs",
        str(n_pcs),
        "--resolution",
        str(resolution),
        "--seed",
        str(seed),
    ]
    if umap:
        args.append("--umap")
    for cd in counts_dirs:
        args.extend(["--counts-dir", cd])
    for sid in sample_ids:
        args.extend(["--sample-id", sid])
    result = run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)
    _report_feature_types_dropped(
        result, feature_types_dropped, where=" of sample(s) " + ", ".join(feature_types_dropped)
    )
    return result


@mcp.tool()
def seurat_find_markers(
    seurat_rds: str,
    output_dir: str,
    ident_key: str = "seurat_clusters",
    group_a: str | None = None,
    group_b: str | None = None,
    cluster: str | None = None,
    logfc_threshold: float = 0.25,
    min_pct: float = 0.1,
    test_use: str = "wilcox",
    assay: str | None = None,
) -> dict[str, Any]:
    """Run Seurat marker detection on a Seurat .rds object.

    The comparison is chosen from ``group_a`` / ``group_b`` / ``cluster``:

    * ``group_a`` and ``group_b``: ``group_a`` vs ``group_b`` (``seurat_markers_<a>_vs_<b>.csv``). A
      ``cluster`` given as well is not used and is listed in ``params.ignored``.
    * ``group_a`` alone, or ``cluster`` alone: that group vs all the others
      (``seurat_markers_cluster_<group>.csv``). ``group_a`` and a different ``cluster`` together are
      refused, as is ``group_b`` without ``group_a``.
    * none of them: FindAllMarkers, positive markers of every identity (``seurat_all_markers.csv``).

    ``params.comparison`` names the comparison that ran. ``assay`` picks the assay tested; left
    unset it is "RNA" when the object has one, else the object's DefaultAssay -- so an object from
    seurat_integrate_qc_cluster is tested on its RNA assay, not on the batch-corrected "integrated"
    values. Split per-sample layers are joined for the test (``params.layers_joined``).
    """
    args: list[str] = [
        "--mode",
        "find_markers",
        "--seurat-rds",
        seurat_rds,
        "--output-dir",
        output_dir,
        "--ident-key",
        ident_key,
        "--logfc-threshold",
        str(logfc_threshold),
        "--min-pct",
        str(min_pct),
        "--test-use",
        test_use,
    ]
    # An empty string is how an agent often spells "not given"; it is not an identity.
    group_a, group_b, cluster, assay = (v if v else None for v in (group_a, group_b, cluster, assay))
    if group_a is not None:
        args.extend(["--group-a", group_a])
    if group_b is not None:
        args.extend(["--group-b", group_b])
    if cluster is not None:
        args.extend(["--cluster", cluster])
    if assay is not None:
        args.extend(["--assay", assay])
    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


@mcp.tool()
def seurat_dimplot(
    seurat_rds: str,
    output_dir: str,
    reduction: str = "umap",
    group_by: str = "seurat_clusters",
    label: bool = True,
    allow_reduction_fallback: bool = False,
) -> dict[str, Any]:
    """Generate a DimPlot PNG from a Seurat object (.rds).

    A ``reduction`` the object does not hold is an error naming the ones it has. With
    ``allow_reduction_fallback=True`` the first of umap/tsne/pca present is drawn instead, and the
    payload says so (``params.reduction`` drawn, ``params.reduction_requested``,
    ``params.used_fallback``, a warning).
    """
    args: list[str] = [
        "--mode",
        "dimplot",
        "--seurat-rds",
        seurat_rds,
        "--output-dir",
        output_dir,
        "--reduction",
        reduction,
        "--group-by",
        group_by,
    ]
    if label:
        args.append("--label")
    if allow_reduction_fallback:
        args.append("--allow-reduction-fallback")
    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


@mcp.tool()
def seurat_spatial_qc_cluster(
    data_path: str,
    output_dir: str,
    project: str = "SeuratSpatial",
    min_cells: int = 3,
    min_features: int = 200,
    n_hvgs: int = 2000,
    n_pcs: int = 30,
    resolution: float = 0.6,
    umap: bool = True,
    spatial_var: bool = True,
    sv_assay: str = "Spatial",
    sv_selection_method: str = "markvariogram",
    sv_nfeatures: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Seurat spatial QC + clustering on Visium data.

    Accepts a Visium Space Ranger folder (filtered_feature_bc_matrix.h5 + spatial/), the
    ``filtered_feature_bc_matrix.h5`` inside such a folder (the folder around it is staged; a
    ``raw_feature_bc_matrix.h5`` or a counts .h5 with no spatial/ beside it is refused), or an .h5ad
    with obsm['spatial'] and a tissue image in uns['spatial'], from which a Visium folder is built.
    Spots with obs['in_tissue'] == 0 in an .h5ad are left out of the matrix and reported
    (``params.in_tissue_filter`` and a warning).

    A counts matrix holding several feature types (a CytAssist protein run's Antibody Capture
    features beside the genes) has only its Gene Expression features QC-filtered, normalised and
    clustered; the others are kept unprocessed as assays of their own in the saved object, and
    ``params.feature_types_input`` / ``other_feature_type_assays`` plus a warning say so.

    With ``spatial_var``, FindSpatiallyVariableFeatures ranks only the scaled features (the
    ``n_hvgs`` variable genes); the CSV holds the top ``sv_nfeatures`` of those, never unscored genes,
    and ``summary.n_sv_features_tested`` / ``n_sv_genes`` say how many were ranked and kept.
    ``seed`` reaches FindClusters and RunPCA / RunUMAP as in seurat_qc_cluster (``params.seeds``).
    """
    os.makedirs(output_dir, exist_ok=True)
    prep = _prepare_data(data_path, output_dir, mode="spatial")
    if "error" in prep:
        return {"status": "error", "error": prep["error"]}

    data_dir = prep.get("data_dir")
    if not data_dir:
        detected = prep.get("format_detected")
        why = " (this .h5ad has no obsm['spatial'])" if detected == "h5ad" else ""
        return {
            "status": "error",
            "error": (
                f"Could not create Visium directory from {data_path}: it was detected as {detected}{why}. "
                f"seurat_spatial_qc_cluster reads {SPATIAL_INPUTS}."
            ),
        }

    args: list[str] = [
        "--mode",
        "spatial_qc_cluster",
        "--data-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--project",
        project,
        "--min-cells",
        str(min_cells),
        "--min-features",
        str(min_features),
        "--n-hvgs",
        str(n_hvgs),
        "--n-pcs",
        str(n_pcs),
        "--resolution",
        str(resolution),
        "--sv-assay",
        sv_assay,
        "--sv-selection-method",
        sv_selection_method,
        "--sv-nfeatures",
        str(sv_nfeatures),
        "--seed",
        str(seed),
    ]
    if umap:
        args.append("--umap")
    if spatial_var:
        args.append("--spatial-var")

    result = run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)
    if "params" not in result:
        result["params"] = {}
    result["params"]["input_format"] = prep.get("format_detected")
    result["params"]["input_conversion"] = prep.get("conversion")
    for key in ("positions_file", "library_id", "images", "n_spots_in_tissue", "n_spots_out_of_tissue_excluded"):
        if prep.get(key) is not None:
            result["params"][key] = prep[key]
    n_off = int(prep.get("n_spots_out_of_tissue_excluded") or 0)
    if n_off:
        # worker_utils.record_in_tissue's keys and wording; the payload here is a dict, not a WorkerOutput.
        n_supplied = int(prep["n_spots_supplied"])
        result["params"]["in_tissue_filter"] = {
            "n_spots_supplied": n_supplied,
            "n_spots_off_tissue_dropped": n_off,
            "n_spots_used": n_supplied - n_off,
        }
        warnings = result.get("warnings")
        result["warnings"] = [
            *(warnings if isinstance(warnings, list) else []),
            f"{n_off} of {n_supplied} spots have obs['in_tissue'] == 0 (background outside the tissue) and were "
            f"left out; {n_supplied - n_off} in-tissue spots were analysed.",
        ]
    return result


@mcp.tool()
def seurat_spatial_feature_plot(
    seurat_rds: str,
    output_dir: str,
    features: list[str],
    image_alpha: float = 0.8,
    spot_size: float = 1.5,
    ncol: int = 1,
) -> dict[str, Any]:
    """Generate SpatialFeaturePlot PNG files for given genes, one PNG per gene.

    ``ncol`` is accepted for compatibility and lays out nothing (there is no multi-panel figure); a
    value other than 1 is listed in ``params.ignored`` with a warning.
    """
    if not features:
        return {"status": "error", "error": "features list is empty."}
    args: list[str] = [
        "--mode",
        "spatial_feature_plot",
        "--seurat-rds",
        seurat_rds,
        "--output-dir",
        output_dir,
        "--image-alpha",
        str(image_alpha),
        "--spot-size",
        str(spot_size),
        "--ncol",
        str(ncol),
    ]
    for f in features:
        args.extend(["--feature", f])
    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


@mcp.tool()
def seurat_spatial_dimplot(
    seurat_rds: str,
    output_dir: str,
    group_by: str = "seurat_clusters",
    image_alpha: float = 0.8,
    spot_size: float = 1.5,
    label: bool = True,
) -> dict[str, Any]:
    """Generate a SpatialDimPlot PNG colored by a meta.data column."""
    args: list[str] = [
        "--mode",
        "spatial_dimplot",
        "--seurat-rds",
        seurat_rds,
        "--output-dir",
        output_dir,
        "--group-by",
        group_by,
        "--image-alpha",
        str(image_alpha),
        "--spot-size",
        str(spot_size),
    ]
    if label:
        args.append("--label")
    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


@mcp.tool()
def seurat_spatial_variable_features(
    seurat_rds: str,
    output_dir: str,
    assay: str = "Spatial",
    selection_method: str = "markvariogram",
    nfeatures: int = 2000,
) -> dict[str, Any]:
    """Run FindSpatiallyVariableFeatures and save CSV of spatially variable genes.

    Seurat tests only the features in the assay's scale.data layer (the ``n_hvgs`` variable genes of
    an object from seurat_spatial_qc_cluster); an assay with no scale.data layer is refused. The CSV
    holds the top ``nfeatures`` of the tested features by rank -- the rows upstream flags
    ``variable`` -- so it has fewer rows when fewer were tested, and ``summary.n_features_tested`` /
    ``n_features_not_tested`` say how many of the assay's genes were ranked and how many never were.
    """
    args: list[str] = [
        "--mode",
        "spatial_variable_features",
        "--seurat-rds",
        seurat_rds,
        "--output-dir",
        output_dir,
        "--assay",
        assay,
        "--selection-method",
        selection_method,
        "--nfeatures",
        str(nfeatures),
    ]
    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
