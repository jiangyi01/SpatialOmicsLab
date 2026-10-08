#!/usr/bin/env python3
"""
Unified data format converter for spatial transcriptomics.

Supports bidirectional conversion between:
  - h5ad (AnnData / Python) ↔ RDS (Seurat / R)
  - h5ad ↔ CSV bundle (counts + coords + metadata)
  - SingleCellExperiment ↔ h5ad (via R subprocess)

Runs inside /opt/conda/envs/novosparc (has anndata, scanpy, scipy).
R operations run whichever Rscript this box resolves: DATA_CONVERTER_RSCRIPT if the deployment set
one, else the first shipped candidate that exists here (repaired across a renamed env or a relocated
conda root), else a bare "Rscript" on PATH.

All logs go to stderr; stdout is JSON-only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from worker_utils import (
    WorkerOutput,
    drop_unlabeled,
    expression_matrix_kind,
    keep_in_tissue,
    resolve_coord_columns,
    sniff_tabular_sep,
)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def _ensure_dir(p: str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _output_path(output_path: str, outdir: Path, default_name: str) -> str:
    """The absolute path a mode writes its one output file to.

    The tools document ``output_dir`` as where the file goes "unless output_rds gives a full path",
    but a bare ``sample.rds`` was used as given and landed in the portal's working directory --
    outside ``output_dir``, where the run's harvester never looks (hunt 2026-09-30,
    u29a-mcp-transport-12). A relative path is therefore taken relative to ``output_dir``.
    """
    path = os.path.expanduser(output_path or "")
    if not path:
        path = str(outdir / default_name)
    elif not os.path.isabs(path):
        path = str(outdir / path)
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _atomic_write_h5ad(adata, path: str) -> None:
    """``write_h5ad`` to ``<path>.partial``, then rename over ``path``; a failed write leaves neither.

    The rename is what keeps a budget kill from leaving a corrupt file at a name exists-therefore-done
    retry logic will trust. A write that RAISES used to leave the ``.partial`` behind beside the
    output (hunt 2026-09-30, u29a-mcp-transport-4), so it is removed before the error goes on.
    """
    tmp = f"{path}.partial"
    try:
        adata.write_h5ad(tmp)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


#: (x-axis name, y-axis name) pairs, most specific first. An equal copy of the agent door's
#: ``spatialomicsgym.tool.spatial_data_converter._COORD_NAME_PAIRS`` -- this worker runs in a tool env
#: that cannot import the package -- and
#: test/test_the_mcp_converter_puts_x_first_as_scanpy_and_the_agent_door_do.py holds the two equal.
_COORD_NAME_PAIRS = (
    ("imagecol", "imagerow"),
    ("pxl_col_in_fullres", "pxl_row_in_fullres"),
    ("x_centroid", "y_centroid"),
    ("center_x", "center_y"),
    ("xcoord", "ycoord"),
    ("x", "y"),
)


def _order_xy_by_name(columns) -> tuple[list, str]:
    """``([x column, y column], how)``: two coordinate columns ordered (x, y) by name.

    Seurat's exporter writes ``imagerow`` (y) first and Space Ranger writes ``pxl_row_in_fullres``
    first, so the file's order is (row, col). scanpy's ``read_visium`` and the agent-side converter both
    store ``obsm['spatial']`` as (col, row) = (x, y); taking the first two columns as they came
    shipped this door's objects mirrored across the diagonal against both, at status ok (hunt
    2026-09-30, u29a-mcp-transport-2). With no recognised pair the file's own order stands, and
    ``how`` says so.
    """
    names = list(columns)
    by_lower = {str(c).strip().lower(): c for c in names}
    for x_name, y_name in _COORD_NAME_PAIRS:
        if x_name in by_lower and y_name in by_lower:
            return [by_lower[x_name], by_lower[y_name]], f"matched by name: x={x_name}, y={y_name}"
    taken = names[:2]
    return taken, f"no recognised coordinate names among {[str(n) for n in names]}; file order {taken} taken as (x, y)"


def _x_matrix_note(adata, written_as: str) -> tuple[str, str]:
    """``(kind, warning)``: what X holds, and a sentence when it is not counts.

    Both export modes write X under a name that says counts -- counts.csv, Seurat's counts slot --
    and the R count-model tools downstream read it as such. A CELLxGENE X is often log-normalised or
    z-scored with the counts in ``adata.raw``, and the export said nothing about which it shipped
    (hunt 2026-09-30, u29a-mcp-transport-7). The matrix written is unchanged; this only says what it is.
    """
    if adata.X is None:
        return "", ""  # no matrix at all: each mode refuses that on its own terms
    kind = expression_matrix_kind(adata.X)
    if kind in ("counts", "empty"):
        return kind, ""
    what = {
        "nonnegative_noninteger": "non-integer values (normalised or log-transformed data?)",
        "negative": "negative values (scaled or z-scored data)",
        "nonfinite": "NaN or infinite values",
    }.get(kind, kind)
    raw = getattr(adata, "raw", None)
    if raw is not None and expression_matrix_kind(raw.X) == "counts":
        remedy = (
            " adata.raw holds raw counts: write them to X (adata.raw.to_adata(), keeping obs/obsm) and convert "
            "that h5ad if the tool reading this needs counts."
        )
    else:
        remedy = " Convert an h5ad whose X holds raw counts if the tool reading this needs counts."
    return kind, f"X holds {what}, not counts, and was exported as {written_as} unchanged.{remedy}"


def _atomic_to_csv(df: pd.DataFrame, path: str) -> None:
    """Write ``df`` beside ``path`` as ``<name>.partial``, then rename over ``path``.

    The counts table of a full-scale export runs to gigabytes and minutes of wall time, and this
    worker is killed from outside whenever the agent's tool budget expires mid-write. A plain
    ``to_csv(path)`` then leaves a truncated file AT THE CANONICAL NAME: a real 73k-cell reference
    died at 5,341 of 10,238 rows, ended on a clean line boundary, and parsed as a complete
    5,340-gene genome for the deconvolution step downstream. After the rename-on-completion, an
    interrupted write leaves only ``counts.csv.partial`` -- a spelling no consumer globs -- and the
    canonical name either holds the whole table or nothing.

    ``os.replace`` is atomic because the temp name is a sibling (same directory, same filesystem).
    """
    tmp = f"{path}.partial"
    df.to_csv(tmp)
    os.replace(tmp, path)


#: Matrix entries materialised at once while counts.csv is streamed out (~200 MB at float64).
_COUNTS_CSV_BLOCK_ENTRIES = 25_000_000


def _atomic_counts_csv(X, obs_names, var_names, path: str, transpose_counts: bool) -> None:
    """Write the expression matrix as counts.csv in row blocks, then rename into place.

    The file is dense text by nature, but the matrix need not be dense in memory all at once: the
    old ``X.toarray()`` of a Visium HD slide (~500k bins x 18k genes) asked for ~37 GB before the
    first byte was written. Each block is built exactly as the whole frame was -- same dtype, same
    index and header objects -- so the bytes on disk are unchanged. Rows are genes when
    ``transpose_counts`` (R convention), cells otherwise. Same ``.partial`` + ``os.replace`` contract
    as :func:`_atomic_to_csv`.
    """
    if transpose_counts:
        mat = X.tocsc() if sp.issparse(X) else X
        index, columns, n_rows = var_names, obs_names, X.shape[1]
    else:
        mat = X.tocsr() if sp.issparse(X) else X
        index, columns, n_rows = obs_names, var_names, X.shape[0]

    def rows(start: int, stop: int) -> np.ndarray:
        blk = mat[:, start:stop] if transpose_counts else mat[start:stop]
        blk = blk.toarray() if sp.issparse(blk) else np.asarray(blk)
        return blk.T if transpose_counts else blk

    block = max(1, _COUNTS_CSV_BLOCK_ENTRIES // max(len(columns), 1))
    tmp = f"{path}.partial"
    if n_rows == 0:
        pd.DataFrame(rows(0, 0), index=index[:0], columns=columns).to_csv(tmp)
    for start in range(0, n_rows, block):
        stop = min(n_rows, start + block)
        frame = pd.DataFrame(rows(start, stop), index=index[start:stop], columns=columns)
        frame.to_csv(tmp, mode="w" if start == 0 else "a", header=start == 0)
    os.replace(tmp, path)


#: Label columns tried, in order, when convert_h5ad_to_csv is not told which one to export.
_CELL_TYPE_CANDIDATES = ["cell_type", "CellType", "celltype", "cluster", "leiden", "louvain"]


# ---------------------------------------------------------------------------
# Rscript discovery
# ---------------------------------------------------------------------------
_RSCRIPT_CANDIDATES = [
    "/opt/conda/envs/seurat_env/bin/Rscript",
    "/opt/conda/envs/celltrek/bin/Rscript",
    "/opt/conda/envs/spatialomicsgym_e1/bin/Rscript",
]


def _repair(path: str) -> str:
    """This box's copy of ``path``, via the same repair every portal applies to its interpreter.

    ``base_mcp`` is importable here because ``tools/`` is on ``sys.path`` for any worker (that is how
    ``worker_utils`` resolves) and its module-level imports are stdlib only. If it is not -- a worker
    copied out of the tree -- the pinned path is returned and resolution degrades to what it was.
    """
    try:
        from base_mcp import _resolve_worker_python

        return _resolve_worker_python(path)
    except Exception:
        return path


def _find_rscript() -> str:
    """The Rscript to launch: the deployment's, else a shipped candidate, else the bare name.

    ``DATA_CONVERTER_RSCRIPT`` is the escape hatch, spelled the way the house spells an interpreter
    override, and honoured without an existence check exactly as ``get_worker_paths`` honours a
    ``{PREFIX}_PYTHON``. It is deliberately NOT in this tool's ``install/recipes/tool_specs`` ``override_vars``:
    that machinery means "the interpreter of the env THIS spec builds" and rebases every entry onto
    that env, which here is a python-only clone of ``novosparc`` -- measured, declaring it would wire
    ``<clone>/bin/Rscript``, a file no such env ships.

    Nothing has to set the variable on a normal box. The candidates below are build-box paths, so
    each is repaired before it is tested: an env renamed by the rebrand, or a conda root anywhere but
    ``/opt/conda``, misses all three otherwise. The bare name is the documented package-side
    fallback, not a failure -- ``utils/execution.py`` runs R that way too.
    """
    pinned = os.environ.get("DATA_CONVERTER_RSCRIPT", "").strip()
    if pinned:
        return _repair(pinned)
    for rs in _RSCRIPT_CANDIDATES:
        repaired = _repair(rs)
        if os.path.isfile(repaired):
            return repaired
    return "Rscript"


def _r_payload(stdout: str) -> dict[str, Any]:
    """The JSON object an R exporter printed, which may span many lines.

    r_to_h5ad_converter.R emits with ``pretty = TRUE``, so a per-line scan for a line that starts
    with an opening brace finds only the brace itself. Take everything from the first brace on and
    parse that instead.

    Anything unparseable -- no JSON at all, or a payload cut short by a killed process -- reads as
    "the R said nothing". This channel carries disclosures about a conversion that has otherwise
    already succeeded, so it must never be the thing that fails it.
    """
    if not stdout:
        return {}
    start = stdout.find("{")
    if start < 0:
        return {}
    try:
        parsed = json.loads(stdout[start:])
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _make_names_unique(adata) -> dict[str, int]:
    """Deduplicate gene and cell names, and say so.

    anndata renames a repeated identifier by joining a counter with ``-``: a gene symbol appearing
    twice comes back as ``GENE0`` and ``GENE0-1``, and a barcode shared by two concatenated samples
    comes back as ``AAACCC-1`` and ``AAACCC-1-1``. Every mode below then writes those names out --
    into barcodes.tsv.gz and the Seurat object R builds from it, or into the counts.csv header and
    the row index of its companion tables -- so the renamed strings are the deliverable, not an
    internal detail. A gene called ``GENE0-1`` joins against no annotation, and a barcode with a
    doubled suffix matches nothing in the run it came from; nothing downstream can tell an
    identifier we invented from one the caller supplied.

    Reports the counts to the payload and shows a worked example on stderr, so the rename is
    visible on both channels. Silent when there was nothing to rename.
    """
    renamed: dict[str, int] = {}
    for noun, attr in (("gene", "var_names"), ("cell", "obs_names")):
        before = [str(n) for n in getattr(adata, attr)]
        getattr(adata, attr + "_make_unique")()
        after = [str(n) for n in getattr(adata, attr)]
        # This worker's env is Python 3.9, where zip() has no strict= to catch a length mismatch.
        changed = [(before[i], after[i]) for i in range(len(before)) if before[i] != after[i]]
        renamed[f"n_{noun}s_renamed"] = len(changed)
        if changed:
            shown = "; ".join(f"{b} -> {a}" for b, a in changed[:3])
            more = ", ..." if len(changed) > 3 else ""
            eprint(
                f"[converter] WARNING: {len(changed)} of {len(before)} {noun} name(s) were not "
                f"unique and have been renamed ({shown}{more}). Every file written below carries "
                f"the renamed {noun} names, not the originals."
            )
    return renamed


def _rename_note(renamed: dict[str, int]) -> str:
    """The same disclosure as one sentence of prose, for the analysis field the model reads.

    Empty string when nothing was renamed, so a conversion that changed no identifier reads exactly
    as it did before. Deliberately free of the substring ``Spatial:``, which the analysis line
    already uses as a field marker.
    """
    parts = []
    for noun, key in (("gene", "n_genes_renamed"), ("cell", "n_cells_renamed")):
        count = int(renamed.get(key) or 0)
        if count:
            parts.append(f"{count} {noun} name(s)")
    if not parts:
        return ""
    return (
        f" NOTE: {' and '.join(parts)} were duplicated in the input and have been renamed with a "
        "numeric '-N' suffix; the outputs carry those renamed identifiers, not the originals."
    )


def _xy_for_export(adata, what: str):
    """``(coords or None, note)``: ``obsm['spatial']`` as the two columns (x, y) an exported coordinate file holds.

    Both exports write two coordinate columns -- coordinates.csv and the spatial_coords.csv the Seurat
    assembler reads -- and both used to take ``obsm['spatial'][:, :2]`` whatever its width. On a
    three-column key (the shape STARmap's converter and ST-GEARS write) that drops z and lays every
    section of a serial stack on one plane, and the export said nothing: has_spatial was True and a
    tool reading the file computed distances across a tissue that does not exist.

    So the width is checked (``worker_utils.spatial_coords`` is the rule for analysis tools; a converter
    has no spatial_key to point a caller at, so the advice here is its own):

    * two columns: returned as they are, no note;
    * more than two, every column past the second constant (one plane, e.g. a single section carrying
      its own z): the first two are returned and ``note`` says what was set aside -- nothing is lost;
    * more than two with a varying extra axis (a stack): refused, before any file is written;
    * no key: ``(None, "")``. Fewer than two columns, or not a matrix: ``(None, note)`` -- not coordinates.
    """
    if "spatial" not in adata.obsm:
        return None, ""
    raw = adata.obsm["spatial"]
    if hasattr(raw, "toarray"):
        raw = raw.toarray()
    elif hasattr(raw, "values"):
        raw = raw.values
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None, (
            f"obsm['spatial'] has shape {tuple(arr.shape)}, which is not two coordinate columns, so no "
            f"coordinates were written to {what}."
        )
    width = int(arr.shape[1])
    if width == 2:
        return arr, ""
    extra = arr[:, 2:]
    if arr.shape[0] and not np.all(np.nanmax(extra, axis=0) == np.nanmin(extra, axis=0)):
        n_planes = int(len(np.unique(extra[np.all(np.isfinite(extra), axis=1)], axis=0)))
        raise ValueError(
            f"obsm['spatial'] has {width} columns and the axis past the second varies across the {arr.shape[0]} "
            f"cells ({n_planes} distinct values: a stack of sections, not one plane). {what} holds two coordinate "
            "columns (x, y), so writing it would drop that axis and lay every section on one plane -- a tool "
            "reading the file would measure distances across a tissue that does not exist. Subset the h5ad to "
            "one section, or write the 2D original back to obsm['spatial'] (the 3D coordinate contract keeps "
            "obsm['spatial'] two-column; the spatial3d_inspector portal's inspect_3d_coordinates shows which "
            "frames the object holds) and convert again."
        )
    constant = [float(v) for v in extra[0]] if arr.shape[0] else []
    note = (
        f"obsm['spatial'] has {width} columns; every column past the second is constant across the cells "
        f"(one plane, value(s) {constant}), so {what} holds x, y and that constant was set aside."
    )
    return arr[:, :2], note


# ---------------------------------------------------------------------------
# h5ad → flat files (for R to read)
# ---------------------------------------------------------------------------
def _export_h5ad_to_flat(h5ad_path: str, export_dir: str) -> dict:
    """Export an h5ad file to flat files that R can read."""
    import anndata as ad

    eprint(f"[converter] Loading h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    # Cells need the same treatment as genes. barcodes.tsv.gz, metadata.csv, spatial_coords.csv
    # and every reduction_*.csv are all indexed off obs_names below, and h5ad_to_seurat.R looks
    # its cells up by name in each of them -- reading metadata.csv with row.names = 1, which
    # rejects a repeated key with "duplicate 'row.names' are not allowed". Concatenating two
    # samples without index_unique leaves exactly that, so without this the documented
    # conversion cannot run at all.
    renamed = _make_names_unique(adata)
    eprint(f"[converter] Shape: {adata.n_obs} cells x {adata.n_vars} genes")

    # Coordinates are checked first: a stack of sections is refused before anything is written.
    coords_xy, coords_note = _xy_for_export(adata, "the Seurat object's spatial coordinates")
    x_kind, x_note = _x_matrix_note(adata, "the Seurat object's counts layer")

    outdir = Path(export_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Expression matrix (genes x cells, sparse MTX)
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    # Convert to CSC for genes-x-cells orientation
    X_t = X.T.tocsc()

    mtx_path = outdir / "matrix.mtx.gz"
    with gzip.open(mtx_path, "wb") as f:
        sio.mmwrite(f, X_t)
    eprint(f"[converter] Wrote matrix: {X_t.shape}")

    # 2. Barcodes (cell names)
    bar_path = outdir / "barcodes.tsv.gz"
    with gzip.open(bar_path, "wt") as f:
        f.write("\n".join(adata.obs_names.astype(str)) + "\n")

    # 3. Features (gene names)
    feat_path = outdir / "features.tsv.gz"
    with gzip.open(feat_path, "wt") as f:
        f.write("\n".join(adata.var_names.astype(str)) + "\n")

    # 4. Metadata (obs)
    meta_path = outdir / "metadata.csv"
    adata.obs.to_csv(meta_path)
    eprint(f"[converter] Wrote metadata: {adata.obs.shape[1]} columns")

    # 5. Spatial coordinates (if present): two columns, checked by _xy_for_export above
    has_spatial = False
    if coords_xy is not None:
        coord_path = outdir / "spatial_coords.csv"
        coord_df = pd.DataFrame(coords_xy, index=adata.obs_names, columns=["x", "y"])
        coord_df.to_csv(coord_path)
        has_spatial = True
        eprint("[converter] Wrote spatial coordinates")
    if coords_note:
        eprint(f"[converter] WARNING: {coords_note}")

    # 6. Dimensionality reductions
    reductions_exported = []
    for key in adata.obsm:
        if key == "spatial":
            continue
        red_name = key.replace("X_", "").lower()
        red_path = outdir / f"reduction_{red_name}.csv"
        red_data = np.asarray(adata.obsm[key])
        if red_data.ndim == 2:
            cols = [f"{red_name}_{i + 1}" for i in range(red_data.shape[1])]
            red_df = pd.DataFrame(red_data, index=adata.obs_names, columns=cols)
            red_df.to_csv(red_path)
            reductions_exported.append(red_name)

    eprint(f"[converter] Exported reductions: {reductions_exported}")

    return {
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "has_spatial": has_spatial,
        "reductions": reductions_exported,
        "meta_columns": list(adata.obs.columns),
        # The caller spreads this dict into the payload's data block, so the rename counts reach a
        # reader without any change there.
        **renamed,
        # Said, not only logged: a set-aside constant axis or an obsm['spatial'] that holds no coordinates.
        "coordinates_note": coords_note,
        "x_matrix_kind": x_kind,
        "x_matrix_note": x_note,
    }


# ---------------------------------------------------------------------------
# Mode 1: h5ad → Seurat RDS
# ---------------------------------------------------------------------------
def convert_h5ad_to_rds(
    h5ad_path: str,
    output_path: str,
    output_dir: str,
    project: str = "SeuratProject",
    assay: str = "RNA",
) -> dict[str, Any]:
    """Convert h5ad to Seurat RDS via flat file interchange."""
    out = WorkerOutput("data_converter", task="h5ad_to_rds")
    outdir = _ensure_dir(output_dir)

    # Step 1: Export h5ad to flat files
    tmpdir = tempfile.mkdtemp(prefix="h5ad2rds_")
    try:
        info = _export_h5ad_to_flat(h5ad_path, tmpdir)
        coords_note = info.pop("coordinates_note", "")
        x_kind = info.pop("x_matrix_kind", "")
        x_note = info.pop("x_matrix_note", "")
        out.set_data(**info)
        if coords_note:
            out.add_warning(coords_note)
            out.add_params({"coordinates_note": coords_note})
        out.add_params({"x_matrix_kind": x_kind})
        if x_note:
            out.add_warning(x_note)

        # Step 2: Call R script to assemble Seurat and save RDS
        rscript = _find_rscript()
        r_worker = str(Path(__file__).parent / "h5ad_to_seurat.R")

        output_path = _output_path(output_path, outdir, "converted.rds")

        cmd = [
            rscript,
            r_worker,
            "--input-dir",
            tmpdir,
            "--output-rds",
            output_path,
            "--project",
            project,
            "--assay",
            assay,
        ]
        eprint(f"[converter] Running R assembler: {' '.join(cmd)}")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
            sys.stderr.flush()

        if proc.returncode != 0:
            raise RuntimeError(f"R assembler failed (exit {proc.returncode})")

        # Parse R JSON output
        stdout = proc.stdout.strip()
        lines = [l for l in stdout.split("\n") if l.strip().startswith("{")]
        r_summary: dict[str, Any] = {}
        if lines:
            r_result = json.loads(lines[-1])
            if r_result.get("status") == "error":
                raise RuntimeError(r_result.get("error", "Unknown R error"))
            if isinstance(r_result.get("summary"), dict):
                r_summary = r_result["summary"]

        out.add_output_file("seurat_rds", output_path)
        out.add_params(
            {
                "project": project,
                "assay": assay,
                "input": h5ad_path,
            }
        )
        rds_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        # info describes the intermediate CSVs we exported from the INPUT h5ad. What the caller
        # asked about is the object the assembler built, and the assembler reports that itself --
        # so prefer its answer, and only fall back to the export's view if it gave us none.
        if "reductions" in r_summary:
            # jsonlite's auto_unbox renders a length-1 character vector as a bare JSON string,
            # and one lone "spatial" is exactly what a coordinates-only conversion produces --
            # so list() on it would hand back one entry per letter.
            raw = r_summary["reductions"]
            attached = [raw] if isinstance(raw, str) else list(raw or [])
            has_spatial = "spatial" in attached
        else:
            attached = info["reductions"]
            has_spatial = info["has_spatial"]
        out.set_summary(
            rds_size_mb=round(rds_size / 1024 / 1024, 2),
            has_spatial=has_spatial,
            reductions=attached,
        )
        out.set_analysis(
            f"Converted h5ad ({info['n_cells']} cells x {info['n_genes']} genes) "
            f"to Seurat RDS ({rds_size / 1024 / 1024:.1f} MB). "
            f"Spatial: {'yes' if has_spatial else 'no'}, "
            f"Reductions: {attached or 'none'}." + (f" NOTE: {coords_note}" if coords_note else "") + _rename_note(info)
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return out.to_dict()


# ---------------------------------------------------------------------------
# Mode 2: Seurat RDS → h5ad
# ---------------------------------------------------------------------------
def _read_per_cell_table(path: str) -> pd.DataFrame:
    """Read one of the R exporter's per-cell CSVs, indexed by cell barcode.

    ``r_to_h5ad_converter.R`` appends the barcode as an ordinary column and writes
    ``row.names = FALSE`` (:179-180, :190-191, :251-252), so the barcode is the *last* column and
    there are no row names at all. Reading these with ``index_col=0`` indexes them by whatever
    happens to come first -- ``orig.ident`` for metadata, ``spatial_1`` for coordinates -- which
    matches no cell and quietly discards the whole table.
    """
    df = pd.read_csv(path, index_col=None)
    if "barcode" in df.columns:
        df = df.set_index("barcode")
        df.index = df.index.astype(str)
    return df


def _align_to_obs(df: pd.DataFrame, obs_names, source: str) -> pd.DataFrame | None:
    """Return ``df``'s numeric columns in ``obs_names`` order, or ``None`` if they cannot be aligned.

    Two alignment routes, in order of trust:

    * by barcode, when the frame carries one and every cell is present;
    * positionally, when it does not but the row count matches -- the historical behaviour, and
      correct for an export whose rows are in matrix order.

    Anything else returns ``None``. The alternative is what this function exists to prevent: a
    ``reindex`` that matches nothing still produces a correctly *shaped* all-NaN frame, which sails
    through a shape check and lands in ``obsm`` under a log line announcing success.
    """
    if "barcode" in df.columns:
        df = df.set_index("barcode")
        df.index = df.index.astype(str)
        aligned = df.reindex(obs_names) if pd.Index(obs_names).isin(df.index).all() else None
        if aligned is None:
            eprint(
                f"[converter] WARNING: {source} is indexed by barcode but does not cover every cell "
                f"({len(df.index.intersection(pd.Index(obs_names)))}/{len(obs_names)} matched); skipped"
            )
            return None
    elif len(df) == len(obs_names):
        aligned = df
    else:
        eprint(f"[converter] WARNING: {source} has {len(df)} rows for {len(obs_names)} cells; skipped")
        return None

    numeric = aligned.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        eprint(f"[converter] WARNING: {source} has no numeric columns; skipped")
        return None
    if numeric.isna().all().all():
        # Unreachable via the two routes above, which is the point: it is the last line of defence
        # against a future path that reintroduces a mismatched reindex.
        eprint(f"[converter] WARNING: {source} aligned to all-NaN; skipped rather than stored")
        return None
    return numeric


def _read_r_spatial_assets(export_dir: str) -> tuple[dict, str]:
    """``(uns['spatial'] entry, note)`` from the exporter's scalefactors.json and images/, or ``({}, "")``.

    The layout is the agent-side converter's (``_assemble_h5ad_from_r_export``): ``scalefactors`` as
    written, and each raster under ``images`` keyed by the resolution its FILE NAME states -- a Seurat
    image slot holds the lowres raster, never a hires one -- as float32 in [0, 1]. ``note`` says why a
    raster that was exported could not be attached.
    """
    entry: dict = {}
    notes = []
    sf_file = os.path.join(export_dir, "scalefactors.json")
    if os.path.isfile(sf_file):
        # A corrupt asset costs the image, not the conversion: before these assets were read at all,
        # the same export converted (review of u29a-mcp-transport-3).
        try:
            with open(sf_file, encoding="utf-8") as fh:
                sf = json.load(fh)
        except (OSError, ValueError) as exc:
            sf = None
            notes.append(f"The scale factors the R exporter wrote could not be read ({type(exc).__name__}: {exc}).")
        if isinstance(sf, dict) and sf:
            entry["scalefactors"] = sf
    img_dir = os.path.join(export_dir, "images")
    rasters = sorted(
        f
        for f in (os.listdir(img_dir) if os.path.isdir(img_dir) else [])
        if f.lower().endswith((".png", ".jpg", ".tif"))
    )
    if rasters:
        try:
            from PIL import Image
        except ImportError as exc:
            return (
                entry,
                f"The tissue image the R exporter wrote was not attached: PIL cannot be imported here ({exc}).",
            )
        images = {}
        for name in rasters:
            try:
                with Image.open(os.path.join(img_dir, name)) as im:
                    arr = np.asarray(im)
            except Exception as exc:  # PIL raises its own DecompressionBombError, OSError, SyntaxError...
                notes.append(
                    f"The tissue image {name} the R exporter wrote was not attached ({type(exc).__name__}: {exc})."
                )
                continue
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            images["hires" if "hires" in name.lower() else "lowres"] = arr
        if images:
            entry["images"] = images
    return entry, " ".join(notes)


def convert_rds_to_h5ad(
    rds_path: str,
    output_path: str,
    output_dir: str,
) -> dict[str, Any]:
    """Convert Seurat RDS to h5ad via flat file interchange."""
    import anndata as ad

    out = WorkerOutput("data_converter", task="rds_to_h5ad")
    outdir = _ensure_dir(output_dir)

    # Step 1: Call existing R exporter
    rscript = _find_rscript()
    r_exporter = str(Path(__file__).parent / "r_to_h5ad_converter.R")

    tmpdir = tempfile.mkdtemp(prefix="rds2h5ad_")
    try:
        cmd = [rscript, r_exporter, "--input", rds_path, "--output-dir", tmpdir]
        eprint(f"[converter] Running R exporter: {' '.join(cmd)}")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
            sys.stderr.flush()

        # The exporter's own account of what it did -- which axis it took to be the cells, and
        # whether it had to assume that. Forwarded below; a disclosure printed to a stdout nobody
        # reads is the same as no disclosure.
        r_info = _r_payload(proc.stdout)
        # Read before the exit code: the exporter says WHY it failed on stdout, and its stderr does
        # not. Checking only the code threw that sentence away -- an unsupported extension read "R
        # exporter failed (exit 1)", and an object it could not extract (exit 0 then) read "R export
        # missing matrix.mtx.gz in <a tmpdir already deleted>" (hunt 2026-09-30, u29a-mcp-transport-3).
        if r_info.get("status") == "error":
            raise RuntimeError(f"R exporter: {r_info.get('message') or 'reported an error without a message'}")
        if proc.returncode != 0:
            raise RuntimeError(f"R exporter failed (exit {proc.returncode})")

        # Step 2: Reassemble h5ad from flat files
        eprint("[converter] Reassembling h5ad from R export...")

        mtx_file = os.path.join(tmpdir, "matrix.mtx.gz")
        bar_file = os.path.join(tmpdir, "barcodes.tsv.gz")
        feat_file = os.path.join(tmpdir, "features.tsv.gz")
        meta_file = os.path.join(tmpdir, "metadata.csv")

        if not os.path.exists(mtx_file):
            raise FileNotFoundError(f"R export missing matrix.mtx.gz in {tmpdir}")

        mat = sio.mmread(mtx_file)
        mat = sp.csr_matrix(mat.T)  # genes x cells → cells x genes

        with gzip.open(bar_file, "rt") as f:
            barcodes = [l.strip() for l in f if l.strip()]
        with gzip.open(feat_file, "rt") as f:
            features_raw = [l.strip() for l in f if l.strip()]

        # Handle tab-separated features. The layout is 10x's: gene_id, gene_name, feature_type --
        # r_to_h5ad_converter.R:174 writes exactly that, with feature_type the constant
        # "Gene Expression". parts[-1] therefore named every gene "Gene Expression"; the symbol is
        # the second field.
        features = []
        gene_ids = []
        for line in features_raw:
            parts = line.split("\t")
            features.append(parts[1] if len(parts) > 1 else parts[0])
            gene_ids.append(parts[0])

        adata = ad.AnnData(X=mat)
        adata.obs_names = barcodes
        adata.var_names = features
        if gene_ids != features:
            adata.var["gene_ids"] = gene_ids

        # Add metadata (handle index mismatches gracefully)
        if os.path.exists(meta_file):
            meta = _read_per_cell_table(meta_file)
            common = meta.index.intersection(adata.obs_names)
            if len(common) > 0:
                meta_aligned = meta.reindex(adata.obs_names)
                for col in meta_aligned.columns:
                    adata.obs[col] = meta_aligned[col].values
                eprint(f"[converter] Added {len(meta_aligned.columns)} metadata columns")
            else:
                eprint(
                    f"[converter] WARNING: none of the {len(meta)} rows in metadata.csv match a cell "
                    f"barcode; no obs columns added"
                )

        # Add spatial coordinates, (x, y) by column name -- see _order_xy_by_name.
        coord_file = os.path.join(tmpdir, "spatial_coords.csv")
        coords_used: list = []
        coords_how = ""
        if os.path.exists(coord_file):
            coords = pd.read_csv(coord_file, index_col=None)
            coords_aligned = _align_to_obs(coords, adata.obs_names, "spatial_coords.csv")
            if coords_aligned is not None and coords_aligned.shape[1] >= 2:
                coords_used, coords_how = _order_xy_by_name(coords_aligned.columns)
                adata.obsm["spatial"] = coords_aligned[coords_used].values.astype(np.float64)
                eprint(f"[converter] Added spatial coordinates ({coords_how})")
        coords_dropped = bool(r_info.get("has_spatial")) and "spatial" not in adata.obsm

        # The Visium scale factors and tissue raster the exporter wrote beside the coordinates. Read
        # into uns['spatial'] exactly as the agent-side converter reads the same export; this door
        # used to leave them in the tmpdir, so sc.pl.spatial and every image-feature tool had no
        # image and no scale (hunt 2026-09-30, u29a-mcp-transport-2).
        spatial_uns, image_note = _read_r_spatial_assets(tmpdir)
        if spatial_uns:
            adata.uns["spatial"] = {"spatial_sample": spatial_uns}

        # Add reductions
        import glob

        for rf in glob.glob(os.path.join(tmpdir, "reduction_*.csv")):
            red_name = os.path.basename(rf).replace("reduction_", "").replace(".csv", "")
            red_df = pd.read_csv(rf, index_col=None)
            red_aligned = _align_to_obs(red_df, adata.obs_names, os.path.basename(rf))
            if red_aligned is None or red_aligned.shape[1] == 0:
                continue
            adata.obsm[f"X_{red_name}"] = red_aligned.values.astype(np.float64)
            eprint(f"[converter] Added reduction: X_{red_name}")

        output_path = _output_path(output_path, outdir, "converted.h5ad")
        # Same rename-on-completion as _atomic_to_csv: a budget kill mid-write must not leave a
        # corrupt file at a name that exists-therefore-done retry logic will trust.
        _atomic_write_h5ad(adata, output_path)
        eprint(f"[converter] Saved h5ad: {output_path}")

        h5ad_size = os.path.getsize(output_path)
        out.set_data(n_cells=adata.n_obs, n_genes=adata.n_vars)
        out.add_output_file("h5ad", output_path)
        params: dict[str, Any] = {"input": rds_path}
        if r_info.get("orientation"):
            params["orientation"] = r_info["orientation"]
        if coords_used:
            params["spatial_columns_used"] = ",".join(str(c) for c in coords_used)
            params["spatial_columns_how"] = coords_how
        out.add_params(params)
        out.add_warnings(r_info.get("warnings"))
        if coords_dropped:
            out.add_warning(
                "The R exporter wrote spatial coordinates, but they could not be matched to every cell of the "
                "matrix, so none were attached: the h5ad has no obsm['spatial'] (stderr names the mismatch)."
            )
        if image_note:
            out.add_warning(image_note)
        out.set_summary(
            h5ad_size_mb=round(h5ad_size / 1024 / 1024, 2),
            has_spatial="spatial" in adata.obsm,
            reductions=[k for k in adata.obsm if k.startswith("X_")],
            meta_columns=list(adata.obs.columns),
        )
        out.set_analysis(
            f"Converted Seurat RDS to h5ad: {adata.n_obs} cells x {adata.n_vars} genes "
            f"({h5ad_size / 1024 / 1024:.1f} MB)."
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return out.to_dict()


# ---------------------------------------------------------------------------
# Mode 3: h5ad → CSV bundle
# ---------------------------------------------------------------------------
#: The record, beside an export, of which h5ad it holds (FOLLOWUPS.md F-38). The bundle's readers look
#: for the four names below, and the harvester and the output inspector pass over ``.json``.
_CSV_EXPORT_RECORD = "csv_export_source.json"
_CSV_EXPORT_SCHEMA = "sog.csv_export_source/1"
#: Every file convert_h5ad_to_csv writes, in the order it writes them. Only these names are ever
#: compared, carried in a record or removed, whatever a record on disk lists.
_CSV_EXPORT_NAMES = ("counts.csv", "coordinates.csv", "metadata.csv", "celltypes.csv")


def _read_json_dict(path: Path) -> dict | None:
    """The JSON object stored at ``path``, or None: a record that cannot be read is no record."""
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _csv_export_names(values) -> list[str]:
    """The export file names among ``values``, once each, in their order; anything else is dropped."""
    names: list[str] = []
    for value in values if isinstance(values, (list, tuple)) else ():
        if isinstance(value, str) and value in _CSV_EXPORT_NAMES and value not in names:
            names.append(value)
    return names


#: Bytes hashed to tell two inputs of one name and size apart: 16 evenly spaced 64 KiB blocks, or the
#: whole file below 1 MiB. HDF5 allocates in fixed steps, so two h5ad files of one shape with different
#: values are the same size to the byte (measured: 6x4 and 9x4 both 21,912 bytes; two 600x400 slides
#: with different counts both 1,032,144) -- a name and a size alone would call two slides one input.
_SAMPLE_BLOCKS, _SAMPLE_BLOCK = 16, 1 << 16


def _sampled_sha256(path: str) -> str | None:
    """sha256 of the sampled blocks of ``path`` (at most 1 MiB read), or None when it cannot be read or is not a
    regular file.

    ``path`` can come from a provenance sidecar, which names whatever it was written to name: ``/dev/zero`` never
    ends a read and a FIFO blocks one for ever. So the file is opened without blocking, read only when ``fstat``
    calls it regular, and never past the sampled bytes, however large it has grown since. A regular file gives the
    digest it always did.
    """
    import hashlib
    import stat

    digest = hashlib.sha256()
    cap = _SAMPLE_BLOCKS * _SAMPLE_BLOCK
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0))
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None
        if info.st_size <= cap:
            digest.update(_read_at(fd, 0, cap))
        else:
            step = (info.st_size - _SAMPLE_BLOCK) // (_SAMPLE_BLOCKS - 1)
            for i in range(_SAMPLE_BLOCKS):
                digest.update(_read_at(fd, i * step, _SAMPLE_BLOCK))
    except OSError:
        return None
    finally:
        os.close(fd)
    return digest.hexdigest()


def _read_at(fd: int, offset: int, limit: int) -> bytes:
    """At most ``limit`` bytes of ``fd`` from ``offset``: fewer only at the end of the file."""
    parts = []
    while limit > 0:
        chunk = os.pread(fd, min(limit, _SAMPLE_BLOCK), offset)
        if not chunk:
            break
        parts.append(chunk)
        offset += len(chunk)
        limit -= len(chunk)
    return b"".join(parts)


def _csv_export_source(h5ad_path: str) -> dict[str, Any] | None:
    """The input as the record keeps it, or None when it cannot be stat'ed (the read then says why)."""
    try:
        st = os.stat(h5ad_path)
    except OSError:
        return None
    return {
        "path": os.path.realpath(h5ad_path),
        "name": os.path.basename(h5ad_path),
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
    }


def _earlier_csv_export(outdir: Path) -> dict[str, Any] | None:
    """What ``outdir`` records about the export it already holds, normalised; None when nothing does.

    The converter's own record first. A folder exported before that record existed still carries the
    provenance sidecar ``WorkerOutput.to_dict`` drops beside the CSVs -- ``params.input``,
    ``input_data.n_cells``/``n_genes`` and ``output_files`` -- which says the same minus the input's size
    and sampled bytes, so those are taken from the recorded path when the file is still there. Another
    tool's sidecar (a later run writing into the same folder replaces it) says nothing about these files.
    """
    rec = _read_json_dict(outdir / _CSV_EXPORT_RECORD)
    if rec is not None and rec.get("schema") == _CSV_EXPORT_SCHEMA and isinstance(rec.get("source"), dict):
        src = rec["source"]
        return {
            "path": str(src.get("path") or ""),
            "name": str(src.get("name") or ""),
            "size": src.get("size"),
            "sample_sha256": src.get("sample_sha256"),
            "n_obs": rec.get("n_obs"),
            "n_vars": rec.get("n_vars"),
            "files": _csv_export_names(rec.get("files")),
            "unfinished": rec.get("state") == "writing",
        }
    from worker_utils import PROVENANCE_FILENAME

    prov = _read_json_dict(outdir / PROVENANCE_FILENAME)
    if prov is None or prov.get("tool") != "data_converter" or prov.get("task") != "h5ad_to_csv":
        return None
    params = prov.get("params") if isinstance(prov.get("params"), dict) else {}
    data = prov.get("input_data") if isinstance(prov.get("input_data"), dict) else {}
    files = prov.get("output_files") if isinstance(prov.get("output_files"), dict) else {}
    path = str(params.get("input") or "")
    size = None
    if path:
        try:
            size = int(os.stat(path).st_size)
        except OSError:
            pass
    return {
        "path": path,
        "name": os.path.basename(path),
        "size": size,
        "sample_sha256": _sampled_sha256(path) if size is not None else None,
        "n_obs": data.get("n_cells"),
        "n_vars": data.get("n_genes"),
        "files": _csv_export_names([os.path.basename(v) for v in files.values() if isinstance(v, str)]),
        "unfinished": False,
    }


def _cells_x_genes(n_obs, n_vars) -> str:
    if isinstance(n_obs, int) and isinstance(n_vars, int):
        return f"{n_obs} cells x {n_vars} genes"
    return "cells x genes not recorded"


def _unrecorded_csv_export(present: list[str]) -> str:
    return (
        f"{', '.join(present)} with no record of the input they came from (no {_CSV_EXPORT_RECORD}, and no "
        "provenance from this converter)"
    )


def _csv_export_refusal(outdir: Path, h5ad_path: str, held: str) -> ValueError:
    """The one refusal both checks give: what the folder holds, the input, the folder to use instead."""
    stem = Path(h5ad_path).stem or "input"
    return ValueError(
        f"output_dir {outdir} already holds {held}. Exporting {h5ad_path} there would replace those files under "
        "the same names and leave the folder mixing two datasets. Convert each h5ad into its own output_dir, "
        f"e.g. {outdir / stem}; the files keep their names. Nothing was written."
    )


def _claim_csv_export_dir(
    outdir: Path, h5ad_path: str, source: dict[str, Any] | None, transpose_counts: bool
) -> tuple[list[str], bool]:
    """``(earlier files of this input, compare the counts header after the read)``, or a ValueError.

    Runs before the h5ad is read, so a refusal reads and writes nothing. The converter writes fixed
    names, and a slide exported and then a reference exported into one ``output_dir`` replaced the
    slide's counts.csv with no word and left its coordinates.csv beside the reference's counts; the
    model read the folder as one dataset and stopped (live run E3c, FOLLOWUPS.md F-38). So:

    * nothing of an export is here (a record whose files are gone included): go ahead;
    * a record names this input -- same real path, or a staged copy (same file name, size and sampled
      bytes): re-export in place, and the record's files this run does not rewrite are removed afterwards;
    * a record names another input: refused, naming it from the record (no read);
    * export files and no record of either kind: refused, unless the existing counts.csv header is the one
      this export writes -- its cell names, so only the genes x cells orientation can show that. Cells x
      genes heads it with the genes, which another input on the same panel shares.
    """
    present = [n for n in _CSV_EXPORT_NAMES if os.path.lexists(outdir / n)]
    if not present or source is None:
        return [], False  # nothing to overwrite; or no input to export, which the read reports
    earlier = _earlier_csv_export(outdir)
    if earlier is not None:
        same = bool(earlier["path"]) and os.path.realpath(earlier["path"]) == source["path"]
        if not same and earlier["name"] and (earlier["name"], earlier["size"]) == (source["name"], source["size"]):
            same = earlier["sample_sha256"] is not None and earlier["sample_sha256"] == _sampled_sha256(h5ad_path)
        if same:
            return earlier["files"], False
        files = [n for n in earlier["files"] if n in present] or present
        what = "an unfinished CSV export" if earlier["unfinished"] else "the CSV export"
        raise _csv_export_refusal(
            outdir,
            h5ad_path,
            f"{what} of {earlier['path'] or 'an unnamed input'} "
            f"({_cells_x_genes(earlier['n_obs'], earlier['n_vars'])}: {', '.join(files)}), a different input",
        )
    unknown = _unrecorded_csv_export(present)
    if "counts.csv" not in present:
        raise _csv_export_refusal(outdir, h5ad_path, f"{unknown} and no counts.csv to show which cells they hold")
    if not transpose_counts:
        raise _csv_export_refusal(
            outdir,
            h5ad_path,
            f"{unknown}, and with transpose_counts=False counts.csv is headed by the genes, not the cells, so it "
            "cannot show the files came from this input",
        )
    return [], True


def _counts_csv_header(index, columns) -> bytes:
    """The header line :func:`_atomic_counts_csv` writes for these names, built the way it builds it."""
    return pd.DataFrame(np.empty((0, len(columns))), index=index[:0], columns=columns).to_csv().encode("utf-8")


def _check_counts_csv_header(outdir: Path, h5ad_path: str, index, columns) -> None:
    """Refuse, before anything is written, an unrecorded counts.csv whose header this export would not write."""
    want = _counts_csv_header(index, columns)
    try:
        with open(outdir / "counts.csv", "rb") as fh:
            got = fh.read(len(want))
    except OSError:
        got = b""
    if got != want:
        present = [n for n in _CSV_EXPORT_NAMES if os.path.lexists(outdir / n)]
        raise _csv_export_refusal(
            outdir,
            h5ad_path,
            f"{_unrecorded_csv_export(present)}, and its counts.csv is not headed by the {len(columns)} cell names "
            "this export writes",
        )


def _drop_stale_csv_exports(outdir: Path, earlier_files: list[str], written: list[str]) -> list[str]:
    """Remove the files an earlier export of this input wrote and this one did not; their paths.

    A label column the h5ad no longer has, say: its celltypes.csv would otherwise sit beside counts.csv
    as if this export had written it. Only names an export writes, only in ``outdir``.
    """
    removed: list[str] = []
    for name in earlier_files:
        path = outdir / name
        if name in written or not os.path.lexists(path):
            continue
        try:
            os.unlink(path)
        except OSError as exc:
            eprint(f"[converter] WARNING: could not remove {name} left by the earlier export of this input: {exc}")
            continue
        removed.append(str(path))
    return removed


def _write_csv_export_record(outdir: Path, record: dict[str, Any]) -> None:
    """The record, written to ``<name>.partial`` and renamed into place. A failed write -- the dump or the rename --
    leaves the earlier record as it was and no ``.partial`` beside it."""
    path = str(outdir / _CSV_EXPORT_RECORD)
    tmp = f"{path}.partial"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def convert_h5ad_to_csv(
    h5ad_path: str,
    output_dir: str,
    transpose_counts: bool = True,
    cell_type_key: str = "",
) -> dict[str, Any]:
    """Export h5ad to CSV files (counts, coords, metadata, cell types).

    ``cell_type_key`` names the obs column exported as celltypes.csv (its single column is always
    headed ``cell_type``). Empty keeps the old behaviour -- the first of :data:`_CELL_TYPE_CANDIDATES`
    present -- and either way ``params.cell_type_column_used`` says which column it was.
    """
    import anndata as ad

    out = WorkerOutput("data_converter", task="h5ad_to_csv")
    # Absolute, so output_files names the files wherever the payload is read.
    outdir = Path(os.path.abspath(_ensure_dir(output_dir)))
    # Whose export the folder already holds is settled before the read (FOLLOWUPS.md F-38).
    source = _csv_export_source(h5ad_path)
    earlier_files, check_header = _claim_csv_export_dir(outdir, h5ad_path, source, transpose_counts)

    adata = ad.read_h5ad(h5ad_path)
    # Cells need the same treatment as genes. counts.csv writes obs_names as its column HEADER,
    # and pandas silently renames a repeated column on read -- so a two-sample concatenation came
    # back carrying cells named "<barcode>.1" that appear in no input, while coordinates.csv,
    # metadata.csv and celltypes.csv (row-indexed, not renamed) kept the raw repeat. The two halves
    # of one bundle then disagreed about what the cells are called, and convert_csv_to_h5ad below
    # died on our own output with "cannot reindex on an axis with duplicate labels".
    renamed = _make_names_unique(adata)
    eprint(f"[converter] Loaded: {adata.n_obs} x {adata.n_vars}")

    cell_type_key = (cell_type_key or "").strip()
    if cell_type_key and cell_type_key not in adata.obs.columns:
        # ValueError, not KeyError: str(KeyError) wraps the whole sentence in quotes on the payload.
        raise ValueError(
            f"cell_type_key={cell_type_key!r} is not an obs column of {h5ad_path}. "
            f"Available obs columns: {list(adata.obs.columns)}"
        )

    if adata.X is None:
        raise ValueError(f"{h5ad_path} has no expression matrix in X, so there is no counts.csv to write.")

    # Coordinates are checked before anything is written: a stack of sections in a three-column
    # obsm['spatial'] is refused rather than flattened into coordinates.csv (see _xy_for_export).
    coords_xy, coords_note = _xy_for_export(adata, "coordinates.csv")
    x_kind, x_note = _x_matrix_note(adata, "counts.csv")
    if x_note:
        out.add_warning(x_note)
    if check_header:  # an unrecorded export is this input's only if its counts.csv is headed by these cells
        _check_counts_csv_header(outdir, h5ad_path, adata.var_names, adata.obs_names)

    # The label column celltypes.csv takes: the one asked for, else the first known label column present.
    if cell_type_key:
        ct_col = cell_type_key
    else:
        ct_col = next((c for c in _CELL_TYPE_CANDIDATES if c in adata.obs.columns), "")

    # The record goes down before the first CSV, so a run killed mid-export still says whose files these
    # are. Until this run finishes it also lists the earlier export's files of this input still here.
    planned = ["counts.csv"]
    planned += ["coordinates.csv"] if coords_xy is not None else []
    planned += ["metadata.csv"] if adata.obs.shape[1] > 0 else []
    planned += ["celltypes.csv"] if ct_col else []
    record = {
        "schema": _CSV_EXPORT_SCHEMA,
        "state": "writing",
        "source": dict(source or {}, sample_sha256=_sampled_sha256(h5ad_path)),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "transpose_counts": bool(transpose_counts),
        "cell_type_column_used": ct_col,
        "files": planned + [n for n in earlier_files if n not in planned and os.path.lexists(outdir / n)],
    }
    _write_csv_export_record(outdir, record)
    written: list[str] = []

    # Counts CSV: genes x cells (R convention) or cells x genes (Python convention), streamed.
    counts_path = str(outdir / "counts.csv")
    _atomic_counts_csv(adata.X, adata.obs_names, adata.var_names, counts_path, transpose_counts)
    out.add_output_file("counts_csv", counts_path)
    written.append("counts.csv")

    # Coordinates CSV: x, y as _xy_for_export returned them
    if coords_xy is not None:
        coords_df = pd.DataFrame(coords_xy, index=adata.obs_names, columns=["x", "y"])
        coords_path = str(outdir / "coordinates.csv")
        _atomic_to_csv(coords_df, coords_path)
        out.add_output_file("coordinates_csv", coords_path)
        written.append("coordinates.csv")
    if coords_note:
        out.add_warning(coords_note)
        out.add_params({"coordinates_note": coords_note})

    # Metadata CSV
    if adata.obs.shape[1] > 0:
        meta_path = str(outdir / "metadata.csv")
        _atomic_to_csv(adata.obs, meta_path)
        out.add_output_file("metadata_csv", meta_path)
        written.append("metadata.csv")

    # Cell type CSV: obs[ct_col], chosen above.
    n_unlabeled = 0
    if ct_col:
        ct_path = str(outdir / "celltypes.csv")
        _atomic_to_csv(adata.obs[[ct_col]].rename(columns={ct_col: "cell_type"}), ct_path)
        out.add_output_file("celltypes_csv", ct_path)
        written.append("celltypes.csv")
        # Counted, not dropped: the export keeps every cell so the bundle stays aligned. The
        # consumer decides (deeplinc and friends take drop_unlabeled); this just says how many.
        _, n_unlabeled = drop_unlabeled(adata.obs[ct_col].astype(object).values, True, what="cells")
        if n_unlabeled:
            out.add_warning(
                f"{n_unlabeled} of {adata.n_obs} cells have no label in obs[{ct_col!r}] (NaN/empty); "
                "celltypes.csv keeps them as they are, so the tool that reads it must drop or label them."
            )
        eprint(f"[converter] Cell types exported from obs[{ct_col!r}] ({n_unlabeled} unlabelled)")
    else:
        eprint(
            f"[converter] No cell-type column exported: none of {_CELL_TYPE_CANDIDATES} is in obs "
            "and no cell_type_key was given"
        )

    # Every file is in place: what this input's earlier export wrote and this run did not goes, and the
    # record says what the folder now holds.
    removed = _drop_stale_csv_exports(outdir, earlier_files, written)
    if removed:
        out.add_params({"removed_stale_files": removed})
        eprint(f"[converter] Removed {removed}: the earlier export of this input wrote them, this one did not")
    left = [n for n in _CSV_EXPORT_NAMES if n not in written and os.path.lexists(outdir / n)]
    if left:
        out.add_warning(
            f"output_dir also holds {', '.join(left)}, which this export did not write; they were left as they "
            "were and are not part of it (output_files lists what is)."
        )
    record.update(state="complete", files=written)
    _write_csv_export_record(outdir, record)

    # Background spots: counted, not dropped -- the bundle keeps every spot, as it keeps unlabelled ones.
    # obs['in_tissue'] travels in metadata.csv only (coordinates.csv is x, y), so a tool that reads
    # coordinates.csv beside counts.csv cannot tell the glass from the tissue unless it is told.
    n_off_tissue = None
    tissue_note = ""
    if "in_tissue" in adata.obs.columns:
        try:
            _, _, n_off_tissue = keep_in_tissue(ad.AnnData(obs=adata.obs[["in_tissue"]].copy()), "spots")
        except ValueError as exc:  # no spot in tissue at all: still exported, and said so
            n_off_tissue = int(adata.n_obs)
            out.add_warning(str(exc))
        if n_off_tissue:
            tissue_note = (
                f" {n_off_tissue} of {adata.n_obs} spots have obs['in_tissue'] == 0 (background outside the tissue) "
                "and are exported with the rest; the flag is in metadata.csv, not coordinates.csv."
            )
            out.add_warning(
                f"{n_off_tissue} of {adata.n_obs} spots have obs['in_tissue'] == 0 (background outside the tissue) "
                "and every spot is exported. The flag travels in metadata.csv only -- coordinates.csv carries x, y -- "
                "so a tool reading coordinates.csv beside counts.csv analyses the background as tissue. Pass "
                "metadata.csv where a tool takes a coordinates file with an in_tissue column, or subset the h5ad to "
                "in-tissue spots before converting."
            )

    orientation = "genes_x_cells" if transpose_counts else "cells_x_genes"
    out.set_data(n_cells=adata.n_obs, n_genes=adata.n_vars, **renamed)
    if ct_col:
        out.set_data(n_cells_unlabeled=int(n_unlabeled))
    if n_off_tissue is not None:
        out.set_data(n_spots_off_tissue=int(n_off_tissue))
    out.add_params(
        {
            "input": h5ad_path,
            "transpose_counts": transpose_counts,
            "cell_type_key": cell_type_key,
            # What was asked for and what was read are two different facts.
            "cell_type_column_used": ct_col,
            "x_matrix_kind": x_kind,
        }
    )
    out.set_summary(
        counts_orientation=orientation,
        # coordinates.csv was written: an obsm['spatial'] that holds no coordinates is not "spatial".
        has_spatial=coords_xy is not None,
        has_celltypes=bool(ct_col),
        meta_columns=list(adata.obs.columns),
    )
    if ct_col:
        ct_note = f" Cell types: obs[{ct_col!r}] -> celltypes.csv" + (
            f" ({n_unlabeled} unlabelled)." if n_unlabeled else "."
        )
    else:
        ct_note = " No cell-type column found (pass cell_type_key to export one)."
    out.set_analysis(
        f"Exported h5ad to CSVs: {adata.n_obs} cells x {adata.n_vars} genes. "
        f"Counts orientation: {orientation}."
        + ct_note
        + tissue_note
        + (f" NOTE: {coords_note}" if coords_note else "")
        + _rename_note(renamed)
    )
    return out.to_dict()


# ---------------------------------------------------------------------------
# Mode 4: CSV bundle → h5ad
# ---------------------------------------------------------------------------
#: First names of the (row, col) pairs ``worker_utils.resolve_coord_columns`` returns row-first.
_ROW_FIRST_NAMES = ("imagerow", "pxl_row_in_fullres", "array_row", "row")


def _read_cell_table(path: str) -> pd.DataFrame:
    """A companion table of a CSV bundle (coordinates, metadata, cell types), indexed by its first column as text.

    The counts header is always read as text, but ``index_col=0`` parses integer-like cell IDs --
    ``0, 1, ...`` from MERFISH or an older Xenium ``cell_id``, and this converter's own round trip of
    such an object -- to int64, and an int64 index matches no text name. The reindex then filled every
    cell with NaN: coordinates came back 100% NaN at status ok, and a NaN label column made
    write_h5ad fail (hunt 2026-09-30, u29a-mcp-transport-4).
    """
    df = pd.read_csv(path, index_col=0, sep=sniff_tabular_sep(path))
    df.index = df.index.astype(str)
    return df


def _cells_matched(
    table: pd.DataFrame, obs_names, path: str, consequence: str, out: WorkerOutput, partial: str = ""
) -> int:
    """How many cells ``table`` has a row for, with a payload warning when that is not all of them.

    A table that matches no cell is not attached at all -- a correctly shaped all-NaN block is the
    thing this converter must never store under a log line announcing success -- and ``consequence``
    says so. ``partial``, when given, says what the cells without a row get instead.
    """
    n_cells = len(obs_names)
    n_matched = int(pd.Index(obs_names).isin(table.index).sum())
    msg = ""
    if not n_matched:
        msg = (
            f"None of the {len(table)} rows of {path} is named like a cell of the counts table (cells: "
            f"{[str(n) for n in list(obs_names)[:3]]}, rows: {[str(n) for n in list(table.index)[:3]]}), so "
            f"{consequence}."
        )
    elif partial and n_matched < n_cells:
        msg = f"{n_cells - n_matched} of {n_cells} cells have no row in {path}; {partial}."
    if msg:
        out.add_warning(msg)
        eprint(f"[converter] WARNING: {msg}")
    return n_matched


def convert_csv_to_h5ad(
    counts_csv: str,
    output_dir: str,
    coords_csv: str = "",
    metadata_csv: str = "",
    celltypes_csv: str = "",
    counts_transposed: bool = True,
    spatial_columns: str = "x,y",
) -> dict[str, Any]:
    """Build h5ad from CSV files."""
    import anndata as ad

    out = WorkerOutput("data_converter", task="csv_to_h5ad")
    outdir = _ensure_dir(output_dir)

    eprint(f"[converter] Reading counts: {counts_csv}")
    # All four files in a bundle are user-supplied paths, and none of them may be assumed to be
    # comma-delimited -- R's write.table and most of GEO ship tabs. Read with a hardcoded comma,
    # a tab-delimited counts table parsed to one column, which index_col=0 consumed, and the
    # transpose below turned the resulting (n_genes, 0) frame into an h5ad with zero cells. The
    # worker logged "Matrix: 0 x 3", saved the file and returned status ok.
    counts_df = pd.read_csv(counts_csv, index_col=0, sep=sniff_tabular_sep(counts_csv))

    # Names as text on both axes, as the companion tables below are read (_read_cell_table).
    if counts_transposed:
        # genes x cells → cells x genes
        X = sp.csr_matrix(counts_df.values.T.astype(np.float32))
        cell_names = [str(c) for c in counts_df.columns]
        gene_names = [str(g) for g in counts_df.index]
    else:
        X = sp.csr_matrix(counts_df.values.astype(np.float32))
        cell_names = [str(c) for c in counts_df.index]
        gene_names = [str(g) for g in counts_df.columns]

    adata = ad.AnnData(X=X)
    adata.obs_names = cell_names
    adata.var_names = gene_names
    eprint(f"[converter] Matrix: {adata.n_obs} x {adata.n_vars}")

    # Add coordinates. Align tolerantly with reindex (missing cells -> NaN) instead of a strict
    # .loc[obs_names], which raises KeyError and aborts the whole conversion whenever the coords CSV
    # does not cover every cell (a common real mismatch: filtered cells, or a superset metadata file).
    used_cols: list[str] = []
    if coords_csv and os.path.exists(coords_csv):
        coords = _read_cell_table(coords_csv)
        sp_cols = spatial_columns.split(",")
        if all(c in coords.columns for c in sp_cols):
            used_cols = sp_cols
        else:
            # The caller passed --coords-csv, so they asked for coordinates. Requiring the exact names
            # and then saying nothing meant the two files a user actually has -- Space Ranger's own
            # tissue_positions.csv, and a Seurat-side export naming the axes imagerow/imagecol -- both
            # produced an h5ad with no obsm["spatial"] at status ok and an empty stderr. Resolve by
            # name instead, the same policy the R workers use, and say which columns were read.
            try:
                used_cols = list(resolve_coord_columns(coords.columns, coords_csv))
            except ValueError as exc:
                eprint(f"[converter] WARNING: no spatial coordinates attached -- {exc}")
            if used_cols and not all(pd.api.types.is_numeric_dtype(coords[c]) for c in used_cols):
                eprint(
                    f"[converter] WARNING: no spatial coordinates attached -- {coords_csv} has no "
                    f"{spatial_columns} column and the closest pair ({', '.join(used_cols)}) is not numeric"
                )
                used_cols = []
            if used_cols and str(used_cols[0]).strip().lower() in _ROW_FIRST_NAMES:
                # The resolver answers in the file's (row, col) order; obsm['spatial'] is (x, y) =
                # (col, row), as scanpy and the agent-side converter store it. Taken as returned, a
                # Space Ranger or Seurat-named file came back mirrored across the diagonal (hunt
                # 2026-09-30, u29a-mcp-transport-2). Reordered here, not in the shared resolver,
                # whose (row, col) answer the R workers read.
                used_cols = [used_cols[1], used_cols[0]]
            if used_cols:
                eprint(
                    f"[converter] WARNING: {coords_csv} has no column named {spatial_columns}; "
                    f"reading coordinates from {', '.join(used_cols)} instead"
                )

        if used_cols and not _cells_matched(coords, adata.obs_names, coords_csv, "no coordinates were attached", out):
            used_cols = []
        if used_cols:
            aligned = coords.reindex(adata.obs_names)[used_cols]
            adata.obsm["spatial"] = aligned.values.astype(np.float64)
            n_missing = int(aligned.isna().any(axis=1).sum())
            if n_missing:
                # In the payload, not only on stderr: a NaN position is not a position.
                out.add_warning(
                    f"{n_missing} of {adata.n_obs} cells have no coordinates in {coords_csv} (no row, or an empty "
                    "value); their obsm['spatial'] is NaN."
                )
                out.set_data(n_cells_without_coordinates=n_missing)
                eprint(f"[converter] WARNING: {n_missing}/{adata.n_obs} cells missing spatial coords (NaN-filled)")
            eprint("[converter] Added spatial coordinates")
    elif coords_csv:
        # A path that is not there is the same silence as a name that does not match: the caller
        # asked for coordinates and the h5ad comes back without them.
        eprint(f"[converter] WARNING: no spatial coordinates attached -- coords file not found: {coords_csv}")

    # Add metadata (same tolerant reindex). Guard against clobbering an obs column already present —
    # the old `col in adata.obs_names` compared a column NAME against the cell barcodes (wrong axis),
    # so it never skipped a real collision and could silently DROP a metadata column named like a cell.
    if metadata_csv and os.path.exists(metadata_csv):
        meta = _read_cell_table(metadata_csv)
        if _cells_matched(
            meta, adata.obs_names, metadata_csv, "no metadata columns were added", out, "their metadata is NaN"
        ):
            meta = meta.reindex(adata.obs_names)
            for col in meta.columns:
                if col in adata.obs.columns:
                    continue
                adata.obs[col] = meta[col].values

    # Add cell types (same tolerant reindex)
    if celltypes_csv and os.path.exists(celltypes_csv):
        ct = _read_cell_table(celltypes_csv)
        if _cells_matched(ct, adata.obs_names, celltypes_csv, "no cell types were added", out, "their label is NaN"):
            ct = ct.reindex(adata.obs_names)
            for col in ct.columns:
                adata.obs[col] = ct[col].values

    # Compute QC
    adata.obs["total_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["n_genes_by_counts"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()

    output_path = str(outdir / "converted.h5ad")
    # Same rename-on-completion as _atomic_to_csv: a budget kill mid-write must not leave a
    # corrupt file at a name that exists-therefore-done retry logic will trust.
    _atomic_write_h5ad(adata, output_path)
    eprint(f"[converter] Saved: {output_path}")

    out.set_data(n_cells=adata.n_obs, n_genes=adata.n_vars)
    out.add_output_file("h5ad", output_path)
    out.add_params(
        {
            "input": counts_csv,
            "coords_csv": coords_csv or "",
            "metadata_csv": metadata_csv or "",
            "celltypes_csv": celltypes_csv or "",
            "counts_transposed": counts_transposed,
            "spatial_columns": spatial_columns,
            # What was asked for and what was read are two different facts. Recording only the
            # request made params a promise the file did not always keep.
            "spatial_columns_used": ",".join(str(c) for c in used_cols),
        }
    )
    out.set_summary(
        has_spatial="spatial" in adata.obsm,
        meta_columns=list(adata.obs.columns),
    )
    out.set_analysis(f"Built h5ad from CSVs: {adata.n_obs} cells x {adata.n_vars} genes.")
    return out.to_dict()


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Spatial data format converter")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["h5ad_to_rds", "rds_to_h5ad", "h5ad_to_csv", "csv_to_h5ad"],
        help="Conversion mode",
    )
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output", default="", help="Output file path (optional)")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    # h5ad_to_rds / rds_to_h5ad options
    parser.add_argument("--project", default="SeuratProject")
    parser.add_argument("--assay", default="RNA")

    # h5ad_to_csv options
    parser.add_argument("--transpose-counts", action="store_true", default=True)
    parser.add_argument("--no-transpose", dest="transpose_counts", action="store_false")
    parser.add_argument("--cell-type-key", default="", help="obs column to export as celltypes.csv")

    # csv_to_h5ad options
    parser.add_argument("--coords-csv", default="")
    parser.add_argument("--metadata-csv", default="")
    parser.add_argument("--celltypes-csv", default="")
    parser.add_argument("--counts-transposed", action="store_true", default=True)
    # Companion store_false so the portal can actually request "no transpose" (input already
    # cells x genes). Without this, --counts-transposed defaulted True with no way to unset it,
    # so a cells x genes CSV was silently transposed -> obs/var swapped. Mirrors --no-transpose.
    parser.add_argument("--no-counts-transposed", dest="counts_transposed", action="store_false")
    parser.add_argument("--spatial-columns", default="x,y")

    args = parser.parse_args()

    try:
        if args.mode == "h5ad_to_rds":
            result = convert_h5ad_to_rds(args.input, args.output, args.output_dir, args.project, args.assay)
        elif args.mode == "rds_to_h5ad":
            result = convert_rds_to_h5ad(args.input, args.output, args.output_dir)
        elif args.mode == "h5ad_to_csv":
            result = convert_h5ad_to_csv(args.input, args.output_dir, args.transpose_counts, args.cell_type_key)
        elif args.mode == "csv_to_h5ad":
            result = convert_csv_to_h5ad(
                args.input,
                args.output_dir,
                args.coords_csv,
                args.metadata_csv,
                args.celltypes_csv,
                args.counts_transposed,
                args.spatial_columns,
            )
        else:
            result = {"status": "error", "error": f"Unknown mode: {args.mode}"}

        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        sys.stdout.flush()

    except Exception as e:
        eprint(f"[converter] ERROR: {e}")
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("data_converter", str(e), task=args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
