#!/usr/bin/env python3
"""
Unified data format converter MCP server.

Provides bidirectional conversion between spatial transcriptomics data formats:
  - h5ad (AnnData/Python) ↔ RDS (Seurat/R)
  - h5ad ↔ CSV bundle (counts + coords + metadata)

Worker runs in novosparc env (has anndata, scanpy, scipy).
R operations use seurat_env via subprocess.
"""

import os
from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

MCP_NAME = "data_converter"
TOOL_NAME = "data_converter"

_default_python = "/opt/conda/envs/novosparc/bin/python"
_default_worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_converter_worker.py")
PYTHON_ENV, WORKER_PY = get_worker_paths("DATA_CONVERTER", _default_python, _default_worker)

mcp = create_mcp(MCP_NAME)


def _default_dir(name: str) -> str:
    """``<work root>/spatialomicsgym_csv_export/<name>`` -- never beside the input, never a fixed /tmp.

    The work root is ``base_mcp.default_output_dir`` (``SOG_WORK_DIR``, else a writable
    ``/workspace/work``, else ``./work``), the one fallback every portal shares.
    """
    return default_output_dir(os.path.join("spatialomicsgym_csv_export", name))


@mcp.tool()
def convert_h5ad_to_seurat_rds(
    h5ad_path: str,
    output_dir: str = "",
    output_rds: str = "",
    project: str = "SeuratProject",
    assay: str = "RNA",
) -> dict[str, Any]:
    """
    Convert an AnnData h5ad file to a Seurat RDS object.

    Transfers expression matrix, cell metadata, spatial coordinates,
    and dimensionality reductions (PCA, UMAP, etc.) from h5ad to Seurat.

    The spatial coordinates are two columns (x, y) from obsm['spatial']. A wider obsm['spatial']
    whose extra columns are constant (one plane) is transferred as x, y with the set-aside constant
    reported (params.coordinates_note, a warning); one whose extra axis varies (a stack of sections)
    is refused rather than flattened onto one plane.

    Parameters
    ----------
    h5ad_path:
        Path to input AnnData .h5ad file.
    output_dir:
        Directory for output files. If empty, uses spatialomicsgym_csv_export/<filename>/ under the
        portals' shared work directory (SOG_WORK_DIR when set).
    output_rds:
        Path for output .rds file. If empty, saves as <output_dir>/converted.rds.
    project:
        Seurat project name.
    assay:
        Seurat assay name (default: RNA).

    Returns
    -------
    Dictionary with status, output file path, and conversion summary.
    """
    if not output_dir:
        output_dir = _default_dir(os.path.basename(h5ad_path).replace(".h5ad", "_rds"))

    args = [
        "--mode",
        "h5ad_to_rds",
        "--input",
        h5ad_path,
        "--output-dir",
        output_dir,
        "--project",
        project,
        "--assay",
        assay,
    ]
    if output_rds:
        args += ["--output", output_rds]
    return run_worker_cli(TOOL_NAME, PYTHON_ENV, WORKER_PY, args)


@mcp.tool()
def convert_seurat_rds_to_h5ad(
    rds_path: str,
    output_dir: str = "",
    output_h5ad: str = "",
) -> dict[str, Any]:
    """
    Convert a Seurat RDS object to AnnData h5ad format.

    Extracts expression matrix, cell metadata, spatial coordinates,
    and dimensionality reductions from the Seurat object.

    Parameters
    ----------
    rds_path:
        Path to input Seurat .rds file.
    output_dir:
        Directory for output files. If empty, uses spatialomicsgym_csv_export/<filename>/ under the
        portals' shared work directory (SOG_WORK_DIR when set).
    output_h5ad:
        Path for output .h5ad file. If empty, saves as <output_dir>/converted.h5ad.

    Returns
    -------
    Dictionary with status, output file path, and conversion summary.
    """
    if not output_dir:
        output_dir = _default_dir(os.path.basename(rds_path).replace(".rds", "_h5ad"))

    args = [
        "--mode",
        "rds_to_h5ad",
        "--input",
        rds_path,
        "--output-dir",
        output_dir,
    ]
    if output_h5ad:
        args += ["--output", output_h5ad]
    return run_worker_cli(TOOL_NAME, PYTHON_ENV, WORKER_PY, args)


@mcp.tool()
def convert_h5ad_to_csv(
    h5ad_path: str,
    output_dir: str = "",
    transpose_counts: bool = True,
    cell_type_key: str = "",
) -> dict[str, Any]:
    """
    Export an AnnData h5ad file to CSV files for use with R tools.

    Produces:
      - counts.csv: Expression matrix (default: genes x cells for R)
      - coordinates.csv: Spatial coordinates (if present)
      - metadata.csv: Cell/spot metadata
      - celltypes.csv: Cell type annotations, one column headed cell_type (if a label column is
        named by cell_type_key or found among cell_type, CellType, celltype, cluster, leiden,
        louvain)

    Every spot is exported. Spots with obs['in_tissue'] == 0 (the background a CELLxGENE Visium
    export carries) are counted in data.n_spots_off_tissue with a warning: the flag is in
    metadata.csv, not coordinates.csv, so a tool reading coordinates.csv cannot leave them out.

    coordinates.csv holds two columns (x, y) from obsm['spatial']. A wider obsm['spatial'] whose extra
    columns are constant (one section carrying its own z) is written as x, y and the set-aside
    constant is reported (params.coordinates_note, a warning); one whose extra axis varies (a stack of
    sections) is refused before anything is written, since two columns would lay every section on one
    plane. An obsm['spatial'] with fewer than two columns writes no coordinates.csv and says so.

    Parameters
    ----------
    h5ad_path:
        Path to input .h5ad file.
    output_dir:
        Directory for output CSV files.
    transpose_counts:
        If true (default), output counts as genes x cells (R convention).
        If false, output as cells x genes (Python convention).
    cell_type_key:
        obs column to export as celltypes.csv (e.g. DeconvolutionLabel1); an absent column is an
        error that lists the obs columns. Empty (default) takes the first of cell_type, CellType,
        celltype, cluster, leiden, louvain that exists. params.cell_type_column_used names the
        column exported, and cells with no label in it are counted in data.n_cells_unlabeled.

    Returns
    -------
    Dictionary with status, output file paths, and conversion summary.
    """
    if not output_dir:
        # Never beside the input: the library and benchmark data directories are not written to.
        output_dir = _default_dir(os.path.basename(h5ad_path).replace(".h5ad", ""))

    args = [
        "--mode",
        "h5ad_to_csv",
        "--input",
        h5ad_path,
        "--output-dir",
        output_dir,
    ]
    if not transpose_counts:
        args.append("--no-transpose")
    if cell_type_key:
        args += ["--cell-type-key", cell_type_key]
    return run_worker_cli(TOOL_NAME, PYTHON_ENV, WORKER_PY, args)


@mcp.tool()
def convert_csv_to_h5ad(
    counts_csv: str,
    output_dir: str = "",
    coords_csv: str = "",
    metadata_csv: str = "",
    celltypes_csv: str = "",
    counts_transposed: bool = True,
    spatial_columns: str = "x,y",
) -> dict[str, Any]:
    """
    Build an AnnData h5ad file from CSV files.

    Parameters
    ----------
    counts_csv:
        Path to counts CSV. Default assumes genes x cells (R convention);
        set counts_transposed=false if cells x genes.
    output_dir:
        Directory for output .h5ad file.
    coords_csv:
        Optional path to a spatial coordinates CSV (cells as rows, cell names in the first column).
        Read on the spatial_columns names when the file has them; otherwise the axis pair is matched
        by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col,
        row/col or x/y -- and the columns actually used are reported in params.spatial_columns_used.
    metadata_csv:
        Optional path to cell metadata CSV.
    celltypes_csv:
        Optional path to cell type annotations CSV.
    counts_transposed:
        If true (default), counts CSV is genes x cells. If false, cells x genes.
    spatial_columns:
        Comma-separated names of the two coordinate columns to read from coords_csv (default "x,y").
        If the file does not have them the axis pair is matched by name instead; either way
        params.spatial_columns_used records what was actually read.

    Returns
    -------
    Dictionary with status, output file path, and conversion summary.
    """
    if not output_dir:
        output_dir = _default_dir(os.path.basename(counts_csv).replace(".csv", "_h5ad"))

    args = [
        "--mode",
        "csv_to_h5ad",
        "--input",
        counts_csv,
        "--output-dir",
        output_dir,
        "--spatial-columns",
        spatial_columns,
    ]
    if coords_csv:
        args += ["--coords-csv", coords_csv]
    if metadata_csv:
        args += ["--metadata-csv", metadata_csv]
    if celltypes_csv:
        args += ["--celltypes-csv", celltypes_csv]
    if not counts_transposed:
        # Counts are already cells x genes -> tell the worker to skip its default transpose.
        # (The bare --counts-transposed flag was a no-op: the worker argparse defaults it True
        #  with no store_false companion, so a cells x genes CSV got silently transposed.)
        args.append("--no-counts-transposed")
    return run_worker_cli(TOOL_NAME, PYTHON_ENV, WORKER_PY, args)


if __name__ == "__main__":
    mcp.run()
