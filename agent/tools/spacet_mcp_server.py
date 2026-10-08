#!/usr/bin/env python3
"""SpaCET cell-type deconvolution + cell-cell interaction MCP wrapper for SpatialOmicsLab."""

import os
import shutil
import tempfile
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli
from worker_utils import pin_blas_threads

TOOL_NAME = "spacet"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "SPACET",
    "/opt/conda/envs/spacet/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/spacet_worker.R",
)

mcp = create_mcp(TOOL_NAME)

# A Space Ranger folder carries its counts as a Matrix Market folder, as an HDF5 file, or both.
# The spacet env's R has no HDF5 reader (SpaCET reads the .h5 through Seurat::Read10X_h5, and
# neither Seurat nor hdf5r is installed there), so when only the .h5 exists this portal -- which
# runs in the agent env, where h5py is -- rewrites it as the Matrix Market folder Space Ranger
# would have written beside it, and hands R that.
MTX_DIR_NAME = "filtered_feature_bc_matrix"
H5_NAME = "filtered_feature_bc_matrix.h5"
MTX_NAMES = ("matrix.mtx.gz", "matrix.mtx")


def _has_mtx_dir(visium_path: str) -> bool:
    folder = os.path.join(visium_path, MTX_DIR_NAME)
    return any(os.path.isfile(os.path.join(folder, name)) for name in MTX_NAMES)


def _decode(values) -> list:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _read_10x_h5(path: str):
    """A 10x HDF5 matrix as (genes x barcodes CSC matrix, ids, names, feature types, barcodes).

    Reads both layouts Cell/Space Ranger have written: v3+ (one ``matrix`` group with a ``features``
    subgroup) and v2 (one group per genome with ``genes``/``gene_names``). The matrix stays sparse.
    """
    import h5py
    import scipy.sparse as sp

    with h5py.File(path, "r") as fh:
        if "matrix" in fh:
            grp = fh["matrix"]
            feats = grp["features"]
            ids = _decode(feats["id"][:])
            names = _decode(feats["name"][:])
            types = _decode(feats["feature_type"][:]) if "feature_type" in feats else ["Gene Expression"] * len(ids)
        else:
            groups = [key for key in fh.keys() if isinstance(fh[key], h5py.Group)]
            if len(groups) != 1:
                raise ValueError(
                    f"{path} has no 'matrix' group and {len(groups)} top-level groups ({groups}); a 10x HDF5 "
                    "matrix has either one 'matrix' group (v3+) or one group per genome (v2, one genome)"
                )
            grp = fh[groups[0]]
            ids = _decode(grp["genes"][:])
            names = _decode(grp["gene_names"][:])
            types = ["Gene Expression"] * len(ids)
        n_features, n_barcodes = (int(v) for v in grp["shape"][:2])
        mat = sp.csc_matrix(
            (grp["data"][:], grp["indices"][:], grp["indptr"][:]),
            shape=(n_features, n_barcodes),
        )
        barcodes = _decode(grp["barcodes"][:])
    if len(ids) != n_features or len(barcodes) != n_barcodes:
        raise ValueError(
            f"{path}: the matrix is {n_features} x {n_barcodes} but lists {len(ids)} features and "
            f"{len(barcodes)} barcodes"
        )
    return mat, ids, names, types, barcodes


def _write_atomic(path: str, write, binary: bool) -> None:
    import gzip

    partial = path + ".partial"
    with gzip.open(partial, "wb" if binary else "wt", compresslevel=1) as fh:
        write(fh)
    os.replace(partial, path)


def _stage_mtx(h5_path: str, stage_dir: str) -> str:
    """Write ``h5_path`` as a ``filtered_feature_bc_matrix/`` folder under ``stage_dir``; return it.

    Every feature is written with its type; the worker keeps the 'Gene Expression' rows and says
    how many others it left out, so the choice is made (and reported) in one place.
    """
    import numpy as np
    import scipy.io as sio

    mat, ids, names, types, barcodes = _read_10x_h5(h5_path)
    out_dir = os.path.join(stage_dir, MTX_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)
    field = "integer" if np.issubdtype(mat.dtype, np.integer) else "real"
    _write_atomic(os.path.join(out_dir, "matrix.mtx.gz"), lambda fh: sio.mmwrite(fh, mat, field=field), True)
    _write_atomic(
        os.path.join(out_dir, "features.tsv.gz"),
        lambda fh: fh.writelines(f"{i}\t{n}\t{t}\n" for i, n, t in zip(ids, names, types)),
        False,
    )
    _write_atomic(os.path.join(out_dir, "barcodes.tsv.gz"), lambda fh: fh.writelines(f"{b}\n" for b in barcodes), False)
    return out_dir


@mcp.tool()
def spacet_deconvolution(
    output_dir: str,
    visium_path: str | None = None,
    counts_csv: str | None = None,
    coords_csv: str | None = None,
    cancer_type: str = "BRCA",
    platform: str = "Visium",
    organism: str = "human",
    core_no: int = 1,
    run_cci: bool = False,
    seed: int = 0,
    allow_signature_fallback: bool = False,
) -> dict[str, Any]:
    """
    Run SpaCET cell-type deconvolution on spatial transcriptomics data.

    SpaCET performs hierarchical deconvolution of the tumor microenvironment,
    first inferring malignant cell fractions then decomposing non-malignant
    cells into immune and stromal subtypes. Optionally runs cell-cell
    interaction (CCI) colocalization analysis.

    Memory: SpaCET's first stage makes the genes x spots matrix dense at every slide size -- 8 bytes
    per entry, with at least three full-size copies alive at once (4.2 measured). Below 20,001 spots
    the genes are every gene expressed on the slide (~2.9 GB per copy for 18,000 genes x 20,000
    spots); from 20,001 spots SpaCET keeps only the genes in its reference profiles, still ~69 GB per
    copy on a 507,684-bin VisiumHD slide (17,050 genes). This is intrinsic to SpaCET. Both inputs are
    read sparse -- the Matrix Market folder as it is, counts_csv block-wise, one block of rows dense at
    a time -- so a run whose lower bound exceeds the available memory stops before SpaCET starts, with
    the numbers, on either path.

    Parameters
    ----------
    output_dir:
        Directory for SpaCET output files (spacet_proportions.csv, spacet_object.rds and, with
        run_cci, spacet_cci_colocalization.csv).
    visium_path:
        Path to a 10X Visium Space Ranger output folder: filtered_feature_bc_matrix/ or
        filtered_feature_bc_matrix.h5 beside spatial/ (tissue_positions.csv,
        tissue_positions_list.csv with or without a header row, or tissue_positions.parquet).
        A .h5 is read here and handed to R as a Matrix Market folder (the SpaCET env has no HDF5
        reader); only in-tissue spots are deconvolved, spots keep their barcodes, and only
        'Gene Expression' features are used. Use this OR counts_csv+coords_csv.
    counts_csv:
        Path to gene expression counts CSV (genes x spots, with row/col names), raw counts. Use with
        coords_csv as alternative to visium_path. Missing or negative values are refused; non-integer
        values are named in a warning, since SpaCET normalises its input itself.
    coords_csv:
        Path to spot coordinates CSV (spots as rows, spot names in the first column). Rows are
        matched to the counts columns by name, in any order: rows for spots the counts do not
        have (e.g. off-tissue spots) are unused, and counts spots with no row are left out and
        reported. Axis columns are matched by name -- imagerow/imagecol,
        pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col, row/col or x/y -- so Space
        Ranger's tissue_positions.csv can be passed as it is, and so can Space Ranger 1's headerless
        tissue_positions_list.csv (a first line that is a spot is read as one). When the file has an
        in_tissue column, counts spots it marks 0 (background) are left out, counted in
        data.n_spots_off_tissue and params.in_tissue_filter, and named in a warning.
    cancer_type:
        Cancer type whose signatures SpaCET uses to find the malignant spots (e.g., 'BRCA',
        'LIHC', 'PDAC', 'CRC', 'OV', 'PRAD', 'BLCA', 'UCEC', 'HNSC', 'LUAD'; 'PANCAN' for the
        pan-cancer expression signature). Its copy-number signature is tried first, then its
        expression signature; if neither marks a cluster malignant the run stops unless
        allow_signature_fallback is True. The signature used is reported as
        params.malignant_signature. From 20,000 spots SpaCET skips its clustering and uses the
        copy-number signature alone, so the type must have one (PANCAN does not); the run says so
        before it starts. The memory SpaCET needs is set by the slide, not by the signature (see
        Memory above); from 20,001 spots LIHC and CHOL narrow the genes further, to those also in
        SpaCET's normal-liver profiles.
    platform:
        Spatial platform: 'Visium', 'oldST', or 'hiresST'.
    organism:
        Organism: 'human' or 'mouse' (mouse genes are mapped to human by SpaCET).
    core_no:
        Number of CPU cores for parallel computation.
    run_cci:
        If True, also run cell-cell interaction colocalization analysis and write
        spacet_cci_colocalization.csv. If that step fails the deconvolution is still returned,
        with the error in summary.cci_error, a warning and the analysis.
    seed:
        Accepted and recorded, but it has no effect: SpaCET seeds its only random step itself
        (set.seed(123)), so the payload lists it under params.ignored.
    allow_signature_fallback:
        If True, let SpaCET's own cascade continue past the requested cancer type: the pan-cancer
        expression signature, and then -- when nothing matches, as on non-tumour tissue -- the 5%
        of spots with the most detected genes as the malignant reference ('seq_depth'). The
        payload names the signature that ran and sets params.used_fallback. It has no effect from
        20,000 spots, where SpaCET has no cascade (then listed under params.ignored).
    """
    # SpaCET.deconvolution spawns `core_no` parallel R workers; if EACH also lets its BLAS use every core that is
    # core_no x n_cores threads, and the cores thrash instead of computing (commit d1689ee measured
    # load average ~328 and a stalled RCTD on a 96-core box). Pin BLAS to one thread per worker so
    # SpaCET.deconvolution's own parallelism is the only parallelism. Must precede the Rscript launch: OpenBLAS
    # reads the count at R startup, and the subprocess inherits os.environ.
    pin_blas_threads()

    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--output-dir",
        output_dir,
        "--cancer-type",
        cancer_type,
        "--platform",
        platform,
        "--organism",
        organism,
        "--core-no",
        str(core_no),
        "--seed",
        str(seed),
    ]

    staged_from = None
    stage_dir = None
    if visium_path:
        if not os.path.isdir(visium_path):
            return {
                "status": "error",
                "error": (
                    f"visium_path {visium_path} is not a directory. It should be a Space Ranger output folder: "
                    f"{MTX_DIR_NAME}/ or {H5_NAME} beside a spatial/ folder. For a single file, pass "
                    "counts_csv + coords_csv instead."
                ),
            }
        args.extend(["--visium-path", visium_path])
        h5_path = os.path.join(visium_path, H5_NAME)
        if not _has_mtx_dir(visium_path) and os.path.isfile(h5_path):
            stage_dir = tempfile.mkdtemp(prefix=".spacet_counts_", dir=output_dir)
            try:
                args.extend(["--matrix-dir", _stage_mtx(h5_path, stage_dir)])
            except Exception as exc:
                shutil.rmtree(stage_dir, ignore_errors=True)
                return {
                    "status": "error",
                    "error": f"could not read {h5_path} as a 10x HDF5 matrix: {type(exc).__name__}: {exc}",
                }
            staged_from = h5_path
    elif counts_csv and coords_csv:
        args.extend(["--counts-csv", counts_csv, "--coords-csv", coords_csv])
    else:
        return {
            "status": "error",
            "error": "Provide either visium_path or both counts_csv and coords_csv",
        }

    if run_cci:
        args.append("--run-cci")
    if allow_signature_fallback:
        args.append("--allow-signature-fallback")

    try:
        result = run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)
    finally:
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)

    if staged_from is not None and isinstance(result, dict) and result.get("status") == "ok":
        # The worker read a staged copy that no longer exists; name the file the counts came from.
        data = result.get("data")
        if isinstance(data, dict):
            data["counts_source"] = staged_from
        params = result.get("params")
        if isinstance(params, dict):
            params["counts_conversion"] = (
                f"{H5_NAME} read by the portal (h5py) and handed to R as a Matrix Market folder, removed after the run"
            )
    return result


if __name__ == "__main__":
    mcp.run()
