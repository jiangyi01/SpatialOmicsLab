"""Spatial transcriptomics data format transformation pipeline.

Converts diverse spatial transcriptomics data formats into a standardized AnnData .h5ad
that is compatible with all SpatialOmicsLab MCP spatial analysis tools.

Target h5ad structure (MCP-compatible):
    adata.X                     - Raw count matrix (sparse CSR)
    adata.obsm['spatial']       - (n_obs, 2) spatial coordinates
    adata.obs['total_counts']   - UMI counts per spot/cell
    adata.obs['n_genes_by_counts'] - Genes detected per spot/cell
    adata.var_names             - Gene symbols (unique)
    adata.uns['spatial']        - Optional: Visium image metadata

Supported input formats:
    - 10x Visium (Space Ranger output)
    - 10x Visium H5 + spatial directory
    - 10x Xenium (cell-level transcripts)
    - MERFISH / Vizgen (cell_by_gene.csv + cell_metadata.csv)
    - Slide-seq / Slide-seqV2 (bead locations + DGE)
    - Stereo-seq (text GEM or lasso CSV; a binary GEF must be converted with `geftools gef2gem`)
    - seqFISH / seqFISH+ (cell positions + expression matrix)
    - CosMx / SMI (Nanostring flat files)
    - STARmap (3D spatial + expression matrix)
    - Generic CSV/TSV (expression matrix + coordinates)
    - Existing h5ad (validate and repair)
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# Every h5ad this module writes is the input re-encoded, never a result, and says so in its own
# ``uns`` -- see ``conversion_record`` for why post-analysis needs to be told.
from spatialomicsgym.tool.conversion_record import stamp_conversion

#: Every text-table extension this module can read, compressed forms included -- pandas decompresses
#: by extension, so a `.tsv.gz` needs no unpacking step. Public because `spatial_pipeline` decides
#: which files to diagnose as tables and has to agree with what is actually convertible: its own
#: list held `.csv .tsv .txt` plus a bolted-on `.csv.gz`, and since `Path("dge.txt.gz").suffix` is
#: `.gz`, every other compressed table was reported "unknown" and the user was told to unpack a
#: container that was really a readable table. `.txt.gz` is the standard Slide-seq DGE extension.
TABULAR_SUFFIXES = (".csv", ".tsv", ".tab", ".txt", ".csv.gz", ".tsv.gz", ".tab.gz", ".txt.gz")

# ---------------------------------------------------------------------------
# Validation & Repair
# ---------------------------------------------------------------------------


def validate_spatial_h5ad(h5ad_path: str) -> str:
    """Validate that an h5ad file meets MCP tool requirements.

    Checks for: count matrix in .X, spatial coordinates in obsm['spatial'],
    QC metrics (total_counts, n_genes_by_counts), and unique var_names.

    Args:
        h5ad_path: Path to the h5ad file to validate.

    Returns:
        str: JSON report with validation status and any issues found.

    """
    adata = sc.read_h5ad(h5ad_path)
    issues = []
    info = {"n_obs": adata.n_obs, "n_vars": adata.n_vars}

    # An object with no cells or no genes is a failed conversion, not a valid one. Every other check
    # below passes vacuously on an empty object — a (0, 2) coordinate array has the right shape and no
    # NaNs — so without this the caller is told an empty h5ad is fine and the real failure surfaces
    # several analysis steps later, far from its cause.
    if adata.n_obs == 0:
        issues.append("CRITICAL: 0 observations (conversion produced an empty object)")
    if adata.n_vars == 0:
        issues.append("CRITICAL: 0 variables/genes (conversion produced an empty object)")

    # Check spatial coordinates. Three of these used to be non-critical, so the file came back
    # "valid" -- and this is the readiness check the unknown-format recovery route ends on. Every
    # spatial tool reads obsm['spatial'] and raises KeyError without it, a 1-column array has no
    # y, and a NaN row breaks every neighbour graph; `_diagnose_h5ad` already called the first a
    # blocker, so the two verdicts contradicted each other (hunt 2026-09-30, u22-spatial-pipeline-4).
    if "spatial" not in adata.obsm:
        # Try to find coordinates in obs columns
        coord_cols = _detect_coordinate_columns(adata.obs)
        if coord_cols:
            issues.append(
                f"CRITICAL: spatial coords found in obs columns {coord_cols}, not in obsm['spatial'], which is "
                "where every spatial tool reads them (auto-fixable by repair_spatial_h5ad)"
            )
        else:
            issues.append("CRITICAL: no spatial coordinates found in obsm['spatial'] or obs columns")
    else:
        coords = np.asarray(adata.obsm["spatial"])
        info["spatial_shape"] = list(coords.shape)
        n_dims = int(coords.shape[1]) if coords.ndim == 2 else 0
        if n_dims < 2:
            issues.append(
                f"CRITICAL: obsm['spatial'] has {n_dims} coordinate column(s), expected >= 2 -- a spatial "
                "tool needs x and y, and repair_spatial_h5ad cannot add an axis"
            )
        else:
            try:
                n_nan = int(np.isnan(coords.astype(np.float64)).any(axis=1).sum())
            except (TypeError, ValueError):
                n_nan = 0
                issues.append("CRITICAL: obsm['spatial'] is not numeric")
            if n_nan:
                issues.append(
                    f"CRITICAL: {n_nan} of {adata.n_obs} observations have a NaN spatial coordinate "
                    "(auto-fixable by repair_spatial_h5ad, which drops them)"
                )

    # Check count matrix
    if adata.X is None:
        issues.append("CRITICAL: .X is None (no expression matrix)")
    else:
        x = adata.X
        if sp.issparse(x):
            x_sample = x[:100].toarray()
        else:
            x_sample = x[:100]
        if np.any(x_sample < 0):
            issues.append("WARNING: .X contains negative values (may not be raw counts)")
        if np.all(x_sample == x_sample.astype(int)):
            info["appears_integer"] = True
        else:
            info["appears_integer"] = False
            issues.append("WARNING: .X contains non-integer values (may be normalized, not raw counts)")

    # Check QC metrics
    if "total_counts" not in adata.obs:
        issues.append("missing obs['total_counts'] (will be computed on repair)")
    if "n_genes_by_counts" not in adata.obs:
        issues.append("missing obs['n_genes_by_counts'] (will be computed on repair)")

    # Check var_names uniqueness
    if not adata.var_names.is_unique:
        issues.append("var_names are not unique (will be deduplicated on repair)")

    # Check for Visium metadata
    if "spatial" in adata.uns:
        info["has_visium_metadata"] = True
    else:
        info["has_visium_metadata"] = False

    status = "valid" if not any("CRITICAL" in i for i in issues) else "invalid"
    return json.dumps({"status": status, "info": info, "issues": issues}, indent=2)


def repair_spatial_h5ad(h5ad_path: str, output_path: str | None = None) -> str:
    """Repair an h5ad file to meet MCP tool requirements.

    Fixes: moves coordinates from obs to obsm['spatial'], computes missing QC metrics,
    deduplicates var_names, and ensures sparse matrix format.

    Args:
        h5ad_path: Path to the input h5ad file.
        output_path: Path for the repaired h5ad. Defaults to ``<stem>_repaired.h5ad`` under the work
            root's ``repaired/`` folder (``$SOG_WORK_DIR``, else ``/workspace/work`` when writable, else
            ``./work``) -- never beside the input, which may sit in a shared data library.

    Returns:
        str: JSON report of repairs applied and output path.

    """
    adata = sc.read_h5ad(h5ad_path)
    repairs = []

    # Fix var_names
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
        repairs.append("deduplicated var_names")

    # Fix spatial coordinates
    if "spatial" not in adata.obsm:
        coord_cols = _detect_coordinate_columns(adata.obs)
        if coord_cols:
            coords = adata.obs[coord_cols].values.astype(np.float64)
            adata.obsm["spatial"] = coords
            repairs.append(f"moved coordinates from obs columns {coord_cols} to obsm['spatial']")
        else:
            # The one repair this function exists to perform is the one it cannot do here, and
            # `obsm['spatial']` is the single hard requirement of every spatial MCP tool. Reporting
            # `"status": "repaired"` with the failure buried in a `repairs` string the caller does
            # not read -- `auto_convert` and `spatial_pipeline` both dispatch on `status` alone --
            # sent the agent on to an analysis step that then died on a KeyError far from here.
            return _conversion_error(
                "repair",
                f"no spatial coordinates found in {Path(h5ad_path).name}: obsm has no 'spatial' key and "
                f"obs has no recognisable coordinate columns (obs columns: {_preview(list(adata.obs.columns))}; "
                f"obsm keys: {_preview(list(adata.obsm))}). Nothing was written -- a file without "
                "obsm['spatial'] is not usable by any spatial tool, so it must be converted from the "
                "original vendor output rather than repaired.",
                extra={"n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars), "repairs": repairs},
            )

    # A 1-column array is not something a repair can complete: there is no second axis to recover,
    # so writing the file would hand back a "repaired" h5ad every spatial tool still refuses.
    spatial = np.asarray(adata.obsm["spatial"])
    n_dims = int(spatial.shape[1]) if spatial.ndim == 2 else 0
    if n_dims < 2:
        return _conversion_error(
            "repair",
            f"obsm['spatial'] in {Path(h5ad_path).name} has {n_dims} coordinate column(s); a spatial tool needs "
            "x and y, and there is no second axis to recover here. Nothing was written -- convert it from the "
            "original vendor output.",
            extra={"n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars), "repairs": repairs},
        )

    # Remove NaN coordinates
    if "spatial" in adata.obsm and np.any(np.isnan(adata.obsm["spatial"])):
        mask = ~np.any(np.isnan(adata.obsm["spatial"]), axis=1)
        n_removed = (~mask).sum()
        adata = adata[mask].copy()
        repairs.append(f"removed {n_removed} spots with NaN coordinates")

    # Ensure sparse matrix
    if adata.X is not None and not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
        repairs.append("converted .X to sparse CSR matrix")

    # Compute QC metrics
    if "total_counts" not in adata.obs or "n_genes_by_counts" not in adata.obs:
        try:
            sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
        except (IndexError, ValueError):
            # Fallback: compute manually if scanpy's percent_top fails on sparse data
            if sp.issparse(adata.X):
                adata.obs["total_counts"] = np.asarray(adata.X.sum(axis=1)).flatten()
                adata.obs["n_genes_by_counts"] = np.asarray((adata.X > 0).sum(axis=1)).flatten()
            else:
                adata.obs["total_counts"] = np.asarray(adata.X.sum(axis=1)).flatten()
                adata.obs["n_genes_by_counts"] = np.asarray((adata.X > 0).sum(axis=1)).flatten()
        repairs.append("computed QC metrics (total_counts, n_genes_by_counts)")

    if output_path is None:
        # Never beside the input: a library h5ad got its repaired copy written into the shared library
        # folder (hunt 2026-09-30, u30-uncovered-mcp-10). The root every MCP portal writes to instead.
        from spatialomicsgym.paths import tool_output_root

        output_path = str(Path(tool_output_root("repaired")) / f"{Path(h5ad_path).stem}_repaired.h5ad")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Dropping every NaN-coordinate spot can empty the object; an empty h5ad is not a repaired one.
    if adata.n_obs == 0 or adata.n_vars == 0:
        return _conversion_error(
            "repair",
            f"repair produced an empty object ({adata.n_obs} obs x {adata.n_vars} vars) and was not "
            "written. Every spot in the input had a NaN spatial coordinate, or the input was already empty.",
            extra={"n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars), "repairs": repairs},
        )

    report: dict[str, Any] = {
        "status": "repaired",
        "repairs": repairs,
        "output_path": output_path,
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
    }
    in_place = _same_file(h5ad_path, output_path)
    _write_h5ad_atomically(adata, output_path)
    stamp_conversion(output_path, producer="repair_spatial_h5ad", source=h5ad_path)
    if in_place:
        report["warnings"] = [_in_place_note(output_path)]
    return json.dumps(report, indent=2)


def _same_file(a: str | Path, b: str | Path) -> bool:
    """True when the two paths name one file, through any symlink or hard link."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _in_place_note(output_path: str | Path) -> str:
    """What a repair whose output_path is its own input did to that name, and what it did not."""
    return (
        f"output_path {output_path} was the input: that name now holds a new file with the result. If it was "
        "a symlink, the link itself was replaced and the file it pointed to was not modified"
    )


def _write_h5ad_atomically(adata: ad.AnnData, output_path: str | Path) -> None:
    """Write beside ``output_path`` and ``os.replace`` onto it, never through it.

    ``write_h5ad`` truncates the path it is given and follows a symlink to do it. Pointed at its
    own input -- ``run_spatial_pipeline(p, p)`` on a staging link into the data library -- it
    rewrote the library copy in place, non-atomically, so a crash mid-write destroyed it (hunt
    2026-09-30, u22-spatial-pipeline-7). Replacing the directory entry changes only the name the
    caller gave; a link's target is left exactly as it was.
    """
    out = Path(output_path)
    tmp = out.with_name(f".{out.name}.partial-{os.getpid()}.h5ad")
    try:
        adata.write_h5ad(tmp)
        os.replace(tmp, out)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def _copy_atomically(src: str | Path, output_path: str | Path) -> bool:
    """Copy ``src`` to ``output_path`` the same way; False (nothing done) when they are one file.

    ``shutil.copy2`` raised ``SameFileError`` for an already-valid h5ad named as its own output, and
    a copy onto a symlink wrote into the link's target (u22-spatial-pipeline-7).
    """
    import shutil

    if _same_file(src, output_path):
        return False
    out = Path(output_path)
    tmp = out.with_name(f".{out.name}.partial-{os.getpid()}")
    try:
        shutil.copy2(str(src), str(tmp))
        os.replace(tmp, out)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
    return True


# ---------------------------------------------------------------------------
# Format Converters
# ---------------------------------------------------------------------------


def _read_visium_naming_the_slide_if_the_matrix_does_not(sr_path: Path, **kwargs):
    """`sc.read_visium`, retried with an explicit slide name when the counts h5 carries none.

    Real Space Ranger stamps a ``library_ids`` attribute into the counts .h5 and scanpy reads the
    slide's name from it. A matrix re-exported by a script (h5py- or scanpy-written fixtures,
    some GEO re-uploads) has no such attribute, and ``sc.read_visium`` dies on
    ``KeyError: 'library_ids'`` -- live round 4 hit exactly this on a Space Ranger directory the
    diagnosis had already declared convertible. The slide is still perfectly readable; only its
    name is missing, so retry naming it after the directory -- the same rule
    ``_attach_visium_spatial`` uses for the MTX route. Any other KeyError is a real failure and
    propagates untouched.
    """
    try:
        return sc.read_visium(sr_path, **kwargs)
    except KeyError as e:
        if e.args[:1] != ("library_ids",):
            raise
        return sc.read_visium(sr_path, library_id=sr_path.resolve().name or "spatial_sample", **kwargs)


#: The six columns of a Space Ranger positions table, in the order every version writes them.
_POSITION_COLUMNS = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]


def read_tissue_positions(spatial_dir: str | Path) -> pd.DataFrame | None:
    """A Space Ranger ``spatial/`` folder's spot positions, indexed by barcode, or ``None``.

    ``tissue_positions.csv`` (headered, Space Ranger 2), ``tissue_positions_list.csv`` (no header,
    Space Ranger 1) and ``tissue_positions.parquet`` (Space Ranger 3, and the only one a Visium HD
    bin carries). The parquet was not read at all, so an HD bin converted with no coordinates and
    the format probe named a reader that raises on it (hunt 2026-09-30, u15-validation-6). Columns
    come back in the file order the callers index positionally, whichever file it was.
    """
    folder = Path(spatial_dir)
    csv = folder / "tissue_positions.csv"
    listed = folder / "tissue_positions_list.csv"
    parquet = folder / "tissue_positions.parquet"
    if csv.exists() or listed.exists():
        pos_file = csv if csv.exists() else listed
        first_line = pos_file.read_text().split("\n")[0]
        if "barcode" in first_line.lower():
            return pd.read_csv(pos_file, index_col=0)
        return pd.read_csv(pos_file, header=None, names=_POSITION_COLUMNS, index_col=0)
    if parquet.exists():
        table = pd.read_parquet(parquet)
        missing = [c for c in _POSITION_COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(f"{parquet} has no {', '.join(missing)} column(s); cannot place the spots")
        return table[_POSITION_COLUMNS].set_index("barcode")
    return None


def convert_visium_spaceranger(spaceranger_dir: str, output_path: str) -> str:
    """Convert 10x Visium Space Ranger output directory to MCP-compatible h5ad.

    Reads the filtered_feature_bc_matrix and spatial/ directory from a standard
    Space Ranger output folder.

    Args:
        spaceranger_dir: Path to Space Ranger output directory (containing
            filtered_feature_bc_matrix.h5 or filtered_feature_bc_matrix/ and spatial/).
        output_path: Path where the output h5ad file will be saved.

    Returns:
        str: JSON report with conversion details and output path.

    """
    sr_path = Path(spaceranger_dir)
    # `sc.read_visium` reads exactly one file name and nothing else, so a run distributed without its
    # filtered .h5 — `spaceranger count --no-bam`, and most public GEO deposits, ship the MTX triplet
    # instead — reached it only to raise FileNotFoundError, after auto_convert had already told the
    # user the format was recognised. Prefer the filtered matrix, then raw, then the triplet.
    # The filtered-.h5 branch first tries plain `sc.read_visium` exactly as before (that path is
    # audited); only the previously-crashing no-`library_ids` case gains the named-slide retry.
    # Space Ranger 3 -- and every Visium HD bin -- writes the positions only as a parquet, which
    # `sc.read_visium` does not read (it raised on the missing CSV). The h5 is read directly then,
    # and the positions attached by the same routine the MTX branch uses (u15-validation-6).
    #
    # The matrix may carry the sample's name in front -- 10x ships every public dataset as
    # `<sample>_filtered_feature_bc_matrix.h5` beside `<sample>_spatial.tar.gz` -- and only the exact
    # name was looked for, so the layout the probe calls Visium at 95% was refused here and in
    # auto_convert (hunt 2026-09-30, u22-spatial-pipeline-1).
    spatial = sr_path / "spatial"
    try:
        counts, ambiguous = _spaceranger_counts(sr_path)
    except OSError as exc:
        return _conversion_error("visium_spaceranger", f"Could not list {sr_path}: {exc}")
    if ambiguous:
        return _conversion_error(
            "visium_spaceranger",
            f"{sr_path} holds more than one sample's count matrix ({_preview([p.name for p in ambiguous])}) beside "
            "one spatial/ folder, which belongs to only one of them. Put each sample's matrix in its own "
            "directory with its own spatial/ folder.",
        )
    if counts is None:
        return json.dumps(
            {
                "status": "error",
                "message": (
                    f"No count matrix in {sr_path}: expected filtered_feature_bc_matrix.h5, "
                    "raw_feature_bc_matrix.h5, or a filtered_feature_bc_matrix/ MTX directory "
                    "(a sample-name prefix in front of any of these is fine)."
                ),
            },
            indent=2,
        )
    # Every route below needs the positions table, and each used to raise its own bare exception
    # without it (OSError from scanpy, FileNotFoundError from the MTX route) out of a function
    # whose contract is a JSON report.
    if not any((spatial / n).exists() for n in _POSITION_FILES):
        return _conversion_error(
            "visium_spaceranger",
            f"No tissue_positions[_list].csv or tissue_positions.parquet in {spatial}: a count matrix carries "
            "no coordinates, so there is nothing to place the spots with. Provide the Visium spatial/ folder "
            "with its positions file.",
        )
    parquet_only = (spatial / "tissue_positions.parquet").exists() and not any(
        (spatial / n).exists() for n in ("tissue_positions.csv", "tissue_positions_list.csv")
    )
    # `sc.read_visium` also raises OSError when either rendering or the scale factors are missing,
    # although neither is needed to place a spot -- a trimmed deposit or an image-free export was
    # diagnosed convertible and then crashed here (hunt 2026-09-30, u22-spatial-pipeline skeptic
    # note 1). `_attach_visium_spatial` mirrors it and takes whatever of those is present.
    incomplete = not all(
        (spatial / n).exists() for n in ("scalefactors_json.json", "tissue_hires_image.png", "tissue_lowres_image.png")
    )
    is_raw = counts.name.lower().endswith(("raw_feature_bc_matrix.h5", "raw_feature_bc_matrix"))
    exact = counts.name in _SPACERANGER_COUNTS
    if counts.is_dir():
        adata = sc.read_10x_mtx(counts)
        _attach_visium_spatial(adata, sr_path)
        source = f"{counts.name}/ (MTX triplet)"
    elif parquet_only or incomplete:
        adata = sc.read_10x_h5(counts)
        _attach_visium_spatial(adata, sr_path)
        source = counts.name + (" (positions from spatial/tissue_positions.parquet)" if parquet_only else "")
    elif exact and counts.name == "filtered_feature_bc_matrix.h5":
        adata = _read_visium_naming_the_slide_if_the_matrix_does_not(sr_path)
        source = "filtered_feature_bc_matrix.h5"
    else:
        adata = _read_visium_naming_the_slide_if_the_matrix_does_not(sr_path, count_file=counts.name)
        source = counts.name

    # A raw matrix holds every barcode on the slide, on tissue or not; Space Ranger's filtered one is
    # the in-tissue subset. Read from raw, the background spots entered clustering and SVG calls as a
    # domain of their own, while `convert_visium_h5_spatial` filtered the same files to in_tissue == 1
    # -- so the result depended on which converter ran (hunt 2026-09-30, u22-spatial-pipeline-18).
    extra: dict[str, Any] = {"count_matrix_source": source}
    notes: list[str] = []
    if is_raw and "in_tissue" in adata.obs.columns:
        on_tissue = pd.to_numeric(adata.obs["in_tissue"], errors="coerce").to_numpy() == 1
        n_off = int((~on_tissue).sum())
        if n_off:
            notes.append(
                f"the count matrix is the raw one ({counts.name}), which holds every barcode on the slide: "
                f"{n_off} of {adata.n_obs} barcodes are not on tissue (in_tissue != 1) and were dropped, as "
                "Space Ranger's filtered matrix drops them"
            )
            adata = adata[on_tissue].copy()
        extra["off_tissue_barcodes_dropped"] = n_off

    return _finish_conversion(
        "visium_spaceranger",
        adata,
        output_path,
        extra=extra,
        notes=notes,
        empty_hint=("every barcode in the raw matrix has in_tissue != 1" if is_raw and adata.n_obs == 0 else None),
    )


#: What a Space Ranger output names its count matrix, in the order a conversion prefers them.
_SPACERANGER_COUNTS = (
    "filtered_feature_bc_matrix.h5",
    "raw_feature_bc_matrix.h5",
    "filtered_feature_bc_matrix",
    "raw_feature_bc_matrix",
)

#: The positions files ``read_tissue_positions`` reads.
_POSITION_FILES = ("tissue_positions.csv", "tissue_positions_list.csv", "tissue_positions.parquet")


def _spaceranger_counts(sr_path: Path) -> tuple[Path | None, list[Path]]:
    """The count matrix a Space Ranger directory holds, and any rival when the choice is not unique.

    Exact names first, in ``_SPACERANGER_COUNTS`` order, so every layout that converted before
    converts from the same file. Then a sample-prefixed spelling (``V1_Mouse_Brain_filtered_
    feature_bc_matrix.h5``), matched with the rule ``format_probe._visium_counts_name`` and
    ``spatial_pipeline._matches_marker`` use: the marker preceded by ``_``, ``-`` or ``.``. Two
    prefixed matrices of the same kind are two samples, and which one spatial/ belongs to is not
    something a filename can settle, so they come back as rivals rather than as a pick.
    """
    for name in _SPACERANGER_COUNTS:
        candidate = sr_path / name
        if candidate.is_dir() if "." not in name else candidate.exists():
            return candidate, []
    entries = sorted(sr_path.iterdir(), key=lambda e: (len(e.name), e.name))
    for name in _SPACERANGER_COUNTS:
        want_dir = "." not in name
        low = name.lower()
        hits = [
            e
            for e in entries
            if e.name.lower().endswith(low)
            and len(e.name) > len(low)
            and e.name.lower()[-len(low) - 1] in "_-."
            and (e.is_dir() if want_dir else e.is_file())
        ]
        if len(hits) > 1:
            return None, hits
        if hits:
            return hits[0], []
    return None, []


def convert_visium_h5_spatial(counts_h5: str, spatial_dir: str, output_path: str) -> str:
    """Convert 10x Visium H5 counts file + spatial directory to MCP-compatible h5ad.

    For cases where Space Ranger output is split: the filtered counts H5 and the
    spatial/ folder are provided separately.

    Args:
        counts_h5: Path to filtered_feature_bc_matrix.h5 file.
        spatial_dir: Path to the spatial/ directory with tissue_positions and images.
        output_path: Path where the output h5ad file will be saved.

    Returns:
        str: JSON report with conversion details and output path.

    """
    adata = sc.read_10x_h5(counts_h5)
    adata.var_names_make_unique()

    spatial_path = Path(spatial_dir)
    # Load tissue positions (v1 or v2 CSV, or Space Ranger 3's parquet)
    positions = read_tissue_positions(spatial_path)

    if positions is not None:
        # Align barcodes
        common = adata.obs_names.intersection(positions.index)
        adata = adata[common].copy()
        positions = positions.loc[common]

        adata.obs["in_tissue"] = positions["in_tissue"].values
        adata.obs["array_row"] = positions["array_row"].values
        adata.obs["array_col"] = positions["array_col"].values
        adata.obsm["spatial"] = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(np.float64)

        # Filter to in-tissue spots
        if "in_tissue" in adata.obs.columns:
            adata = adata[adata.obs["in_tissue"] == 1].copy()

    # Without a tissue_positions file the coordinate block above is skipped and obsm['spatial'] is never
    # set — don't write a "success" h5ad that spatial MCP tools will then KeyError on.
    if "spatial" not in adata.obsm:
        return json.dumps(
            {
                "status": "error",
                "message": f"No tissue_positions[_list].csv or tissue_positions.parquet found in {spatial_path}; "
                "obsm['spatial'] "
                "could not be set. Provide the Visium spatial/ folder with tissue positions.",
            },
            indent=2,
        )

    # Load images if available
    _load_visium_images(adata, spatial_path)
    return _finish_conversion(
        "visium_h5_spatial",
        adata,
        output_path,
        empty_hint=(
            "no barcode in the counts matrix matched tissue_positions, or every matched spot had in_tissue=0"
            if adata.n_obs == 0
            else None
        ),
    )


def convert_xenium(
    transcripts_path: str,
    output_path: str,
    cell_id_col: str = "cell_id",
    gene_col: str = "feature_name",
    x_col: str = "x_location",
    y_col: str = "y_location",
    min_counts: int = 5,
    min_qv: float | None = 20,
) -> str:
    """Convert 10x Xenium transcript-level data to cell-level MCP-compatible h5ad.

    Aggregates transcript-level data into a cell-by-gene count matrix with
    cell centroid coordinates.

    Args:
        transcripts_path: Path to transcripts.csv.gz or transcripts.parquet.
        output_path: Path where the output h5ad file will be saved.
        cell_id_col: Column name for cell IDs.
        gene_col: Column name for gene/feature names.
        x_col: Column name for x coordinates.
        y_col: Column name for y coordinates.
        min_counts: Minimum total counts per cell to keep.
        min_qv: Minimum decoding quality (the ``qv`` Phred score) a transcript needs to be counted.
            20 is the threshold Xenium Onboard Analysis applies when it builds its own
            cell_feature_matrix. ``None`` or 0 counts every decoded transcript. A table with no
            ``qv`` column is counted unfiltered, and the report says so.

    Returns:
        str: JSON report with conversion details and output path.

    """
    path = Path(transcripts_path)
    compression = "gzip" if str(path).endswith(".gz") else None
    wanted = list(dict.fromkeys([cell_id_col, gene_col, x_col, y_col]))
    # The quality column is read only when a threshold is in force and the table has one. XOA
    # builds cell_feature_matrix from transcripts with qv >= 20, and this counted every decoded
    # read, so totals, the min_counts cut and everything downstream differed from the vendor's
    # matrix for the same run with nothing in the report saying so (hunt 2026-09-30,
    # u22-spatial-pipeline-17). The header is read without pandas so the one pandas read below is
    # still the projected one.
    header = _table_columns(path)
    qv_filter = bool(min_qv) and "qv" not in wanted and "qv" in (header or ())
    if qv_filter:
        wanted.append("qv")

    # Read the four columns, not the table. XOA writes about a dozen -- transcript_id, qv,
    # z_location, fov_name, codeword_index and the rest -- and a run is hundreds of millions of
    # rows, so loading the ones nothing below touches is pure resident memory on the data this
    # converter exists for. The aggregation is unchanged, so the h5ad is what the full read
    # produced; only the peak footprint moves.
    #
    # The fallback is the wrong-column-name path: both engines raise ValueError there, and the
    # answer the caller should get is the documented JSON report naming what the file *does* have,
    # which needs the header. Re-reading in full costs no more than the old code always cost, and
    # only happens on the way to returning that error.
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path, columns=wanted)
        else:
            df = pd.read_csv(path, compression=compression, usecols=wanted)
    except (ValueError, KeyError):
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, compression=compression)

    # These four names are parameters because a transcripts table from anything but XOA uses other
    # ones. Check them before touching the data so a wrong guess arrives as the documented JSON
    # report naming the columns the file really has, not as `KeyError: 'feature_name'`.
    err = _require_columns(df, wanted, source_format="xenium", label=f"transcripts file {path.name}")
    if err:
        return err

    # Older XOA parquet exports store cell_id and feature_name as binary. `str(b"NegControlProbe_1")`
    # is "b'NegControlProbe_1'", so no control prefix below matched and negative controls were
    # counted as genes, inflating total_counts (hunt 2026-09-30, u22-spatial-pipeline-24).
    for col in dict.fromkeys([cell_id_col, gene_col]):
        df[col] = _decoded(df[col])

    notes: list[str] = []
    extra: dict[str, Any] = {"min_qv": min_qv if qv_filter else None}
    if qv_filter:
        quality = pd.to_numeric(df["qv"], errors="coerce")
        passed = (quality >= float(min_qv)).to_numpy()
        n_low = int((~passed).sum())
        extra["transcripts_below_min_qv_dropped"] = n_low
        if n_low:
            notes.append(
                f"{n_low} of {len(df)} transcripts have qv < {min_qv} (or no qv) and were not counted, as Xenium "
                "Onboard Analysis leaves them out of its cell_feature_matrix; pass min_qv=None to count them"
            )
            df = df[passed]
    elif min_qv:
        notes.append(
            f"no decoding-quality filter (min_qv={min_qv}) was applied: "
            + (
                "the transcripts table has no 'qv' column"
                if header is not None
                else "the table's header could not be read ahead of the load to find a 'qv' column"
            )
        )

    # Filter out unassigned transcripts. XOA v2 writes the string "UNASSIGNED"; XOA v1.x writes an
    # INTEGER cell_id using -1 as the sentinel, so the string comparison alone matches nothing there
    # and every off-cell transcript in the run is aggregated into an observation literally named
    # '-1' — typically the highest-count "cell" in the object, which then survives min_counts,
    # distorts normalisation and clustering, and sits at the mean position of the whole slide.
    # The -1 rule is gated on dtype: a platform with genuine string ids that happen to read "-1"
    # is not covered by the XOA integer convention.
    df = df[df[cell_id_col].notna()]
    if pd.api.types.is_numeric_dtype(df[cell_id_col]):
        df = df[df[cell_id_col] != -1]
    else:
        df = df[df[cell_id_col].astype(str) != "UNASSIGNED"]
    df[cell_id_col] = df[cell_id_col].astype(str)

    # Build count matrix, sparsely. `groupby().size().unstack(fill_value=0)` materialised the dense
    # cells x genes frame -- ~400k cells x a 5,000-gene panel is ~16 GB of int64, plus the
    # column-filtered copy -- before csr_matrix ever saw it (hunt 2026-09-30, u22-spatial-pipeline-16).
    # The codes reproduce its layout exactly: cells and genes in sorted order, and a categorical gene
    # column keeps every category in category order, as an unobserved groupby does.
    counted = df[df[gene_col].notna()]
    cell_codes, cell_index = pd.factorize(counted[cell_id_col], sort=True)
    genes = counted[gene_col]
    if isinstance(genes.dtype, pd.CategoricalDtype):
        gene_codes, gene_index = genes.cat.codes.to_numpy(), genes.cat.categories
    else:
        gene_codes, gene_index = pd.factorize(genes, sort=True)
    counts = sp.coo_matrix(
        (np.ones(len(counted), dtype=np.int64), (cell_codes, gene_codes)), shape=(len(cell_index), len(gene_index))
    ).tocsr()
    # Drop Xenium control features — counting them as genes inflates total_counts (perturbing the
    # min_counts filter below) and feeds negative-control noise to every downstream tool. The MERFISH
    # and CosMx converters filter their controls the same way.
    _control_prefixes = (
        "NegControlProbe",
        "NegControlCodeword",
        "antisense",
        "BLANK",
        "UnassignedCodeword",
        "DeprecatedCodeword",
    )
    _keep = [i for i, g in enumerate(gene_index) if not str(g).startswith(_control_prefixes)]
    counts = counts[:, _keep]
    cell_ids = [str(c) for c in cell_index]
    gene_names = [gene_index[i] for i in _keep]

    # Compute cell centroids
    centroids = df.groupby(cell_id_col)[[x_col, y_col]].mean()
    centroids = centroids.loc[cell_ids]

    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=cell_ids),
        var=pd.DataFrame(index=gene_names),
    )
    adata.obsm["spatial"] = centroids.values.astype(np.float64)

    # Filter low-count cells
    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
    n_before = adata.n_obs
    adata = adata[adata.obs["total_counts"] >= min_counts].copy()

    return _finish_conversion(
        "xenium",
        adata,
        output_path,
        extra=extra,
        notes=notes,
        empty_hint=_min_counts_hint(n_before, adata.n_obs, min_counts),
        count_filter=(n_before, min_counts),
    )


def _table_columns(path: Path) -> list[str] | None:
    """A transcripts table's column names, read without loading it; None when they cannot be read."""
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            return list(pq.read_schema(path).names)
        import csv
        import gzip

        opener = gzip.open if str(path).lower().endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:  # type: ignore[operator]
            return next(csv.reader([fh.readline().rstrip("\r\n")]))
    except Exception:
        return None


def _decoded(column: pd.Series) -> pd.Series:
    """``column`` with any bytes values decoded as UTF-8; unchanged when it holds none."""
    if isinstance(column.dtype, pd.CategoricalDtype):
        categories = column.cat.categories
        if any(isinstance(c, bytes) for c in categories):
            return column.cat.rename_categories([c.decode("utf-8") if isinstance(c, bytes) else c for c in categories])
        return column
    if column.dtype != object:
        return column
    first = column.dropna().head(1)
    if first.empty or not isinstance(first.iloc[0], bytes):
        return column
    return column.map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)


def convert_merfish(
    cell_by_gene_path: str,
    cell_metadata_path: str,
    output_path: str,
    x_col: str = "center_x",
    y_col: str = "center_y",
    cell_id_col: str | None = None,
    min_counts: int = 5,
) -> str:
    """Convert MERFISH / Vizgen data to MCP-compatible h5ad.

    Reads a cell-by-gene expression matrix and a cell metadata file with spatial coordinates.

    Args:
        cell_by_gene_path: Path to cell_by_gene.csv (cells x genes count matrix).
        cell_metadata_path: Path to cell_metadata.csv (with cell coordinates).
        output_path: Path where the output h5ad file will be saved.
        x_col: Column name for x coordinates in metadata.
        y_col: Column name for y coordinates in metadata.
        cell_id_col: Column to use as cell ID. If None, uses the first column or index.
        min_counts: Minimum total counts per cell to keep.

    Returns:
        str: JSON report with conversion details and output path.

    """
    expr = pd.read_csv(cell_by_gene_path, index_col=0)
    # Read the metadata unindexed and set the index afterwards: `index_col=<name>` raises a bare
    # ValueError when the name is absent, which tells the caller nothing about what the file does
    # have. Vizgen has shipped the id as `cell`, `cell_id` and `EntityID` across MERSCOPE releases,
    # so a wrong guess here is routine.
    meta = pd.read_csv(cell_metadata_path)
    meta_label = f"cell metadata file {Path(cell_metadata_path).name}"
    if cell_id_col is None:
        meta = meta.set_index(meta.columns[0])
    else:
        err = _require_columns(meta, [cell_id_col], source_format="merfish", label=meta_label)
        if err:
            return err
        meta = meta.set_index(cell_id_col)
    err = _require_columns(meta, [x_col, y_col], source_format="merfish", label=meta_label)
    if err:
        return err

    # Align indices
    expr, meta, notes, err = _align_on_shared_ids(
        expr,
        meta,
        source_format="merfish",
        left_label=f"cell_by_gene file {Path(cell_by_gene_path).name}",
        right_label=meta_label,
    )
    if err:
        return err

    # Remove Blank/Negative control probes
    gene_cols = [c for c in expr.columns if not str(c).startswith(("Blank", "blank", "NegControl", "negcontrol"))]
    expr = expr[gene_cols]

    adata = ad.AnnData(
        X=sp.csr_matrix(expr.values.astype(np.float32)),
        obs=pd.DataFrame(index=expr.index.astype(str)),
        var=pd.DataFrame(index=gene_cols),
    )
    adata.obsm["spatial"] = meta[[x_col, y_col]].values.astype(np.float64)

    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
    n_before = adata.n_obs
    adata = adata[adata.obs["total_counts"] >= min_counts].copy()

    return _finish_conversion(
        "merfish",
        adata,
        output_path,
        notes=notes,
        empty_hint=_min_counts_hint(n_before, adata.n_obs, min_counts),
        count_filter=(n_before, min_counts),
    )


def convert_slideseq(
    dge_path: str,
    bead_locations_path: str,
    output_path: str,
    min_counts: int = 5,
) -> str:
    """Convert Slide-seq / Slide-seqV2 data to MCP-compatible h5ad.

    Args:
        dge_path: Path to digital gene expression matrix (genes x beads, TSV/CSV).
        bead_locations_path: Path to bead locations file (bead_id, x, y).
        output_path: Path where the output h5ad file will be saved.
        min_counts: Minimum total counts per bead to keep.

    Returns:
        str: JSON report with conversion details and output path.

    """
    # Load DGE (typically genes x beads) and bead locations. Sniff the delimiter rather than assuming
    # tabs: `spatial_pipeline` resolves the DGE as MappedDGEForR.csv / DGE.csv before *.tsv, and the
    # stock Broad MappedDGEForR.csv is comma-separated. Reading a comma file with sep="\t" parses each
    # line as a single column, index_col=0 then consumes it, and the result is a 0-column frame that
    # silently becomes a 0x0 h5ad. `sep=None` requires engine="python", which runs csv.Sniffer.
    dge = pd.read_csv(dge_path, sep=None, engine="python", index_col=0)
    locs = pd.read_csv(bead_locations_path)
    if "bead_barcode" in locs.columns:
        locs = locs.set_index("bead_barcode")
    else:
        locs = locs.set_index(locs.columns[0])

    # Orient the DGE to beads x genes by testing which axis actually carries the bead barcodes. A pure
    # shape heuristic (rows<cols) mislabels a puck with fewer beads than genes (small/filtered puck, or a
    # DGE already stored beads x genes) and yields a silent 0-observation h5ad — fall back to the shape
    # rule only when neither axis overlaps the barcodes (e.g. mismatched barcode formats).
    idx_overlap = len(dge.index.intersection(locs.index))
    col_overlap = len(dge.columns.intersection(locs.index))
    if col_overlap > idx_overlap:
        dge = dge.T
    elif idx_overlap == 0 and col_overlap == 0 and dge.shape[0] < dge.shape[1]:
        dge = dge.T

    # Detect coordinate columns. The old fallback here was `[locs.columns[0], locs.columns[1]]` --
    # file order over *every* column, numeric or not. A deposited puck that carries a cluster
    # assignment or a QC flag in front of its coordinates then put that column into
    # `obsm['spatial']`, and a cluster label is numeric, so it casts to float exactly like a
    # coordinate: nothing raised, the h5ad opened, `validate_spatial_h5ad` called it valid, and every
    # neighbour graph downstream was built on an axis that is not a spatial axis.
    #
    # Restrict to numeric columns and order them through the helper `convert_starmap` already uses,
    # which returns a note recording whether the pair was matched by name or guessed from position.
    coord_cols = _detect_coordinate_columns(locs)
    coord_note = None
    if not coord_cols:
        num_cols = [c for c in locs.columns if pd.api.types.is_numeric_dtype(locs[c])]
        if len(num_cols) < 2:
            return _conversion_error(
                "slideseq",
                f"bead locations file {Path(bead_locations_path).name} has fewer than 2 numeric columns "
                f"(numeric: {_preview(num_cols)}; all columns: {_preview(list(locs.columns))}). "
                "The first column is read as the bead barcode, so the file needs a barcode column "
                "plus an x and a y.",
            )
        coord_cols, coord_note = _order_coordinate_columns_by_name(num_cols)

    # Align
    dge, locs, notes, err = _align_on_shared_ids(
        dge,
        locs,
        source_format="slideseq",
        left_label=f"DGE file {Path(dge_path).name}",
        right_label=f"bead locations file {Path(bead_locations_path).name}",
    )
    if err:
        return err
    if coord_note:
        notes.append(coord_note)

    adata = ad.AnnData(
        X=sp.csr_matrix(dge.values.astype(np.float32)),
        obs=pd.DataFrame(index=dge.index.astype(str)),
        var=pd.DataFrame(index=dge.columns),
    )
    adata.obsm["spatial"] = locs[coord_cols].values.astype(np.float64)

    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
    n_before = adata.n_obs
    adata = adata[adata.obs["total_counts"] >= min_counts].copy()

    return _finish_conversion(
        "slideseq",
        adata,
        output_path,
        notes=notes,
        empty_hint=_min_counts_hint(n_before, adata.n_obs, min_counts),
        count_filter=(n_before, min_counts),
    )


# The tool that turns the binary half of the Stereo-seq format into the text half this reads.
# `spatial_pipeline` states the same remedy for the diagnosis path; it imports this module lazily
# inside functions, so the constant is not shared by import. `test/test_spatial_converter_ingest_
# honesty.py` pins the two spellings equal.
_STEREOSEQ_GEF_TOOL = "geftools gef2gem"


def convert_stereoseq(
    gem_path: str,
    output_path: str,
    bin_size: int = 50,
    min_counts: int = 5,
) -> str:
    """Convert a Stereo-seq text GEM file to MCP-compatible h5ad.

    Stereo-seq GEM files have columns: geneID, x, y, MIDCount (or UMICount).
    Data is binned into square bins of the specified size.

    A binary `.gef` is the other half of the format and is not readable here; SAW writes both,
    and `geftools gef2gem` produces the GEM from one.

    Args:
        gem_path: Path to the .gem or .gem.gz file.
        output_path: Path where the output h5ad file will be saved.
        bin_size: Bin size in coordinate units for spatial binning.
        min_counts: Minimum total counts per bin to keep.

    Returns:
        str: JSON report with conversion details and output path.

    """
    # A .gef is a binary HDF5 container, not a text GEM — read_csv would raise a raw ParserError/
    # UnicodeDecodeError. Reject it with a clear JSON error (auto_convert routes .gef here).
    #
    # Name the tool that produces a readable file. "Provide a .gem text file" is not an instruction
    # anyone can act on without already knowing what makes one, and a deposited chip is often
    # GEF-only. `diagnose_spatial_data` and `run_spatial_pipeline` already say this; the tool the
    # agent is then routed to was the one surface still handing back the dead end.
    if str(gem_path).lower().endswith(".gef"):
        return json.dumps(
            {
                "status": "error",
                "message": (
                    "GEF (binary HDF5) is not supported; it is the binary half of the Stereo-seq "
                    f"format. Run `{_STEREOSEQ_GEF_TOOL}` on this file to produce the text GEM "
                    "(SAW writes both, so one may already be beside it), then convert that."
                ),
            },
            indent=2,
        )
    # Read GEM file (skip comment lines starting with #). A GEM is tab-separated by specification,
    # but the diagnosis also routes a comma-separated `geneID,x,y,MIDCount` lasso `.csv` here, and
    # the fixed tab read that whole header as one column, so the run failed on columns the file has
    # (hunt 2026-09-30, u22-spatial-pipeline-12). The header line -- the first one after the '#'
    # preamble -- says which: a tab wherever it has one, as before.
    gem = pd.read_csv(
        gem_path,
        sep=_gem_delimiter(Path(gem_path)),
        comment="#",
        compression="gzip" if str(gem_path).endswith(".gz") else None,
    )

    # Normalize column names. Match the count column by PREFIX and case rather than against a list of
    # exact spellings: SAW has shipped MIDCount, MIDCounts, MIDCnt and UMICount across versions, and
    # any spelling the list missed previously escaped as a bare KeyError out of a function whose
    # documented contract is to return a JSON report. Coordinates are normalised for case too, so an
    # "X"/"Y" header does not surface as KeyError: 'x'.
    _count_prefixes = ("midcount", "midcnt", "umicount", "count", "expressioncount")
    col_map = {}
    for col in gem.columns:
        lower = str(col).strip().lower()
        if lower in ("geneid", "gene", "gene_id", "gene_name", "genename"):
            col_map[col] = "geneID"
        elif lower in ("x", "y"):
            col_map[col] = lower
        elif lower.startswith(_count_prefixes) and "MIDCount" not in col_map.values():
            col_map[col] = "MIDCount"
    gem = gem.rename(columns=col_map)

    missing = [c for c in ("geneID", "x", "y", "MIDCount") if c not in gem.columns]
    if missing:
        return json.dumps(
            {
                "status": "error",
                "message": (
                    f"GEM file is missing required column(s) {missing} after normalisation. "
                    f"Found columns: {list(gem.columns)}. Expected a gene column (geneID/gene), "
                    "x and y, and a count column (MIDCount/MIDCounts/UMICount)."
                ),
            },
            indent=2,
        )

    # Bin coordinates
    gem["x_bin"] = (gem["x"] // bin_size) * bin_size
    gem["y_bin"] = (gem["y"] // bin_size) * bin_size
    gem["bin_id"] = gem["x_bin"].astype(str) + "_" + gem["y_bin"].astype(str)

    # Aggregate counts per bin x gene, sparsely. `pivot(...).fillna(0)` built the dense bins x genes
    # frame first -- a bin50 full chip is ~160k bins x ~25k genes, tens of GB of float64 -- before
    # handing it to csr_matrix (hunt 2026-09-30, u22-spatial-pipeline-16). Sorted codes keep the
    # pivot's row and column order, and duplicate (bin, gene) entries sum exactly as groupby did.
    keep = gem["geneID"].notna().to_numpy()
    bin_codes, bins = pd.factorize(gem["bin_id"][keep], sort=True)
    gene_codes, genes = pd.factorize(gem["geneID"][keep], sort=True)
    matrix = sp.coo_matrix(
        (gem["MIDCount"][keep].fillna(0).to_numpy(dtype=np.float64), (bin_codes, gene_codes)),
        shape=(len(bins), len(genes)),
    ).tocsr()
    matrix.eliminate_zeros()

    # Compute bin centroids
    bin_coords = gem.groupby("bin_id")[["x_bin", "y_bin"]].first()
    bin_coords = bin_coords.loc[bins]

    adata = ad.AnnData(
        X=matrix.astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(bins, name="bin_id")),
        var=pd.DataFrame(index=pd.Index(genes, name="geneID")),
    )
    adata.obsm["spatial"] = bin_coords.values.astype(np.float64)

    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
    n_before = adata.n_obs
    adata = adata[adata.obs["total_counts"] >= min_counts].copy()

    return _finish_conversion(
        "stereoseq",
        adata,
        output_path,
        extra={"bin_size": bin_size},
        empty_hint=_min_counts_hint(n_before, adata.n_obs, min_counts),
        count_filter=(n_before, min_counts),
    )


def _is_gem_table(columns) -> bool:
    """Whether a table's columns (lower-cased) are a GEM's: a gene column beside ``x`` and ``y``.

    ``spatial_pipeline._diagnose_tabular_file`` applies this same rule to call a table
    ``stereoseq_gem``; ``auto_convert`` reads it from here so a file goes to one converter whichever
    tool the caller reached for.
    """
    cols = {str(c).lower() for c in columns}
    return {"x", "y"} <= cols and bool(cols & {"geneid", "gene"})


def _gem_delimiter(path: Path) -> str:
    """The delimiter of a GEM-shaped table, read off its header line (the first after any '#' lines).

    A tab whenever the header has one -- the GEM specification, and every file that read correctly
    before -- otherwise the first of comma, semicolon or pipe it contains. A header that cannot be
    read keeps the tab, so the read below fails exactly as it always did and names the columns.
    """
    import gzip
    import itertools

    try:
        opener = gzip.open if str(path).lower().endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:  # type: ignore[operator]
            line = next((row for row in itertools.islice(fh, 10_000) if row.strip() and not row.startswith("#")), "")
    except (OSError, EOFError, UnicodeError):
        return "\t"
    if not line or "\t" in line:
        return "\t"
    return next((c for c in (",", ";", "|") if c in line), "\t")


def convert_cosmx(
    expr_path: str,
    fov_positions_path: str,
    output_path: str,
    x_col: str = "CenterX_global_px",
    y_col: str = "CenterY_global_px",
    cell_id_col: str = "cell_ID",
    fov_col: str = "fov",
    min_counts: int = 5,
) -> str:
    """Convert Nanostring CosMx SMI data to MCP-compatible h5ad.

    Args:
        expr_path: Path to exprMat_file.csv (cells x genes expression matrix).
        fov_positions_path: Path to metadata_file.csv (cell positions and FOV info).
        output_path: Path where the output h5ad file will be saved.
        x_col: Column name for global x pixel coordinate.
        y_col: Column name for global y pixel coordinate.
        cell_id_col: Column name for cell IDs.
        fov_col: Column name for field of view IDs.
        min_counts: Minimum total counts per cell to keep.

    Returns:
        str: JSON report with conversion details and output path.

    """
    expr = pd.read_csv(expr_path)
    meta = pd.read_csv(fov_positions_path)
    expr_label = f"expression file {Path(expr_path).name}"
    meta_label = f"metadata file {Path(fov_positions_path).name}"
    notes: list[str] = []

    err = _require_columns(meta, [x_col, y_col], source_format="cosmx", label=meta_label)
    if err:
        return err

    # `cell_id_col` and `fov_col` are parameters because AtoMx has shipped `cell_ID`, `cell_id` and
    # `cell` across versions. Honour them on BOTH files: testing only `expr.columns` meant a name
    # missing from the expression file fell through to `set_index(expr.columns[0])`, and the first
    # column of a CosMx exprMat is `fov` -- so every cell in a field of view collapsed onto a single
    # label. When the name was missing from the metadata file instead, a bare KeyError escaped.
    in_expr, in_meta = cell_id_col in expr.columns, cell_id_col in meta.columns
    if in_expr != in_meta:
        return _require_columns(
            meta if in_expr else expr,
            [cell_id_col],
            source_format="cosmx",
            label=meta_label if in_expr else expr_label,
        )

    # Build unique cell ID from fov + cell_id
    if in_expr and in_meta:
        use_fov = fov_col in expr.columns and fov_col in meta.columns
        if use_fov:
            expr["uid"] = expr[fov_col].astype(str) + "_" + expr[cell_id_col].astype(str)
            meta["uid"] = meta[fov_col].astype(str) + "_" + meta[cell_id_col].astype(str)
        else:
            # CosMx numbers cells within a field of view, so without the FOV the id is only unique
            # if the export already made it so. `_align_on_shared_ids` refuses it if it is not.
            notes.append(
                f"no '{fov_col}' column in both files; used '{cell_id_col}' alone as the cell id and "
                "assumed it is unique across the whole run"
            )
            expr["uid"] = expr[cell_id_col].astype(str)
            meta["uid"] = meta[cell_id_col].astype(str)
        expr = expr.set_index("uid")
        meta = meta.set_index("uid")
    else:
        # Neither file has the requested id column. Falling back to the first column is only safe
        # when that column is actually an id: the first column of a CosMx exprMat is `fov`, so the
        # fallback used to collapse every cell in a field of view onto one label. Take it only when
        # both first columns are unique, and otherwise say which column was asked for and missing.
        expr_first, meta_first = expr.columns[0], meta.columns[0]
        if expr[expr_first].duplicated().any() or meta[meta_first].duplicated().any():
            return _conversion_error(
                "cosmx",
                f"neither file has a '{cell_id_col}' column, and the first column of each "
                f"('{expr_first}' in {expr_label}, '{meta_first}' in {meta_label}) repeats values, so it "
                "cannot be the cell id -- in a CosMx exprMat the first column is the FOV, and indexing on "
                "it would merge every cell in a field of view. Pass cell_id_col naming the real id column. "
                f"Expression columns: {_preview(list(expr.columns))}. Metadata columns: "
                f"{_preview(list(meta.columns))}.",
            )
        notes.append(
            f"neither file has a '{cell_id_col}' column; used the first column of each "
            f"('{expr_first}' and '{meta_first}') as the cell id"
        )
        expr = expr.set_index(expr_first)
        meta = meta.set_index(meta_first)

    # Identify gene columns (exclude metadata columns)
    non_gene_cols = {
        cell_id_col,
        fov_col,
        x_col,
        y_col,
        "uid",
        "Area",
        "Mean.PanCK",
        "Max.PanCK",
        "Mean.CD45",
        "Max.CD45",
        "Mean.DAPI",
        "Max.DAPI",
        "Mean.CD298_B2M",
        "Max.CD298_B2M",
    }
    # CosMx negative-control probes measure background, not expression. The real vendor names are
    # `NegPrb1..N` and `SystemControl1..N`; matching only "Negative" let them through into var_names,
    # where they inflate total_counts (shifting the min_counts filter below) and feed background
    # counts to normalisation and to every downstream tool. Matched case-insensitively because the
    # capitalisation varies across CosMx export versions.
    _control_prefixes = ("negprb", "systemcontrol", "negative")
    candidate_genes = [c for c in expr.columns if c not in non_gene_cols]
    gene_cols = [c for c in candidate_genes if not str(c).lower().startswith(_control_prefixes)]
    control_cols = [c for c in candidate_genes if c not in gene_cols]

    expr, meta, align_notes, err = _align_on_shared_ids(
        expr, meta, source_format="cosmx", left_label=expr_label, right_label=meta_label
    )
    if err:
        return err
    notes.extend(align_notes)
    expr = expr[gene_cols]

    adata = ad.AnnData(
        X=sp.csr_matrix(expr.values.astype(np.float32)),
        obs=pd.DataFrame(index=expr.index.astype(str)),
        var=pd.DataFrame(index=gene_cols),
    )
    adata.obsm["spatial"] = meta[[x_col, y_col]].values.astype(np.float64)
    # The schema advertises that the FOV is "kept in obs so per-FOV batch effects can be modelled
    # later", and it was not: a CosMx run images tens to hundreds of fields of view, each its own
    # exposure, so the FOV is the batch key for the whole experiment. Recovering it after conversion
    # means going back to the metadata CSV, which by then nothing downstream still has.
    if fov_col in meta.columns:
        adata.obs[fov_col] = pd.Categorical(meta[fov_col].astype(str).to_numpy())
    # Record what was dropped so the removal is auditable from the h5ad alone.
    adata.uns["control_probes_removed"] = list(control_cols)

    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
    n_before = adata.n_obs
    adata = adata[adata.obs["total_counts"] >= min_counts].copy()

    return _finish_conversion(
        "cosmx",
        adata,
        output_path,
        extra={"control_probes_removed": list(control_cols)},
        notes=notes,
        empty_hint=_min_counts_hint(n_before, adata.n_obs, min_counts),
        count_filter=(n_before, min_counts),
    )


def convert_starmap(
    expr_path: str,
    coords_path: str,
    output_path: str,
    min_counts: int = 5,
) -> str:
    """Convert STARmap data (expression matrix + 2D/3D coordinates) to MCP-compatible h5ad.

    Args:
        expr_path: Path to expression matrix (cells x genes CSV/TSV).
        coords_path: Path to coordinates file (cells x [x, y] or [x, y, z] CSV).
        output_path: Path where the output h5ad file will be saved.
        min_counts: Minimum total counts per cell to keep.

    Returns:
        str: JSON report with conversion details and output path.

    """
    expr = pd.read_csv(expr_path, index_col=0)
    coords = pd.read_csv(coords_path, index_col=0)

    expr, coords, notes, err = _align_on_shared_ids(
        expr,
        coords,
        source_format="starmap",
        left_label=f"expression file {Path(expr_path).name}",
        right_label=f"coordinates file {Path(coords_path).name}",
    )
    if err:
        return err

    # `iloc[:, :2]` took whatever two columns came first. A coordinates file written as (y, x) --
    # the order every image-derived table uses, because an image is indexed row-then-column -- was
    # then loaded transposed, and a transposed tissue still looks like a tissue: neighbourhoods,
    # Moran's I and every spatial-domain call come out plausible and wrong. Order by name instead,
    # and record in the report when there was no name to go on.
    num_cols = [c for c in coords.columns if pd.api.types.is_numeric_dtype(coords[c])]
    if len(num_cols) < 2:
        return _conversion_error(
            "starmap",
            f"coordinates file {Path(coords_path).name} has fewer than 2 numeric columns "
            f"(numeric: {_preview(num_cols)}; all columns: {_preview(list(coords.columns))}). "
            "The first column is read as the cell id, so the file needs an id column plus x and y.",
        )
    ordered, note = _order_coordinate_columns_by_name(num_cols)
    notes.append(note)

    adata = ad.AnnData(
        X=sp.csr_matrix(expr.values.astype(np.float32)),
        obs=pd.DataFrame(index=expr.index.astype(str)),
        var=pd.DataFrame(index=expr.columns),
    )
    adata.obsm["spatial"] = coords[ordered].values.astype(np.float64)
    # A third axis only by its NAME: every other numeric column used to be filed as depth, so a
    # coordinates table with cell_area, fov, volume or in_tissue beside x and y became a "3D" stack
    # whose z was a cell's area (hunt 2026-09-30, u15-validation-extra-10). Exactly one named z
    # column is used; anything else numeric is reported, never stacked.
    extras = [c for c in num_cols if c not in ordered]
    named_z = [c for c in extras if str(c).strip().lower() in _Z_NAMES]
    z_col = named_z[0] if len(named_z) == 1 else None
    if z_col is not None:
        adata.obsm["spatial_3d"] = coords[[*ordered, z_col]].values.astype(np.float64)
    elif named_z:
        notes.append(f"more than one column is named as a z axis ({_preview(named_z)}); none was used as depth")
    unused = [c for c in extras if c != z_col]
    if unused:
        notes.append(f"numeric columns not used as coordinates: {_preview(unused)}")

    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
    n_before = adata.n_obs
    adata = adata[adata.obs["total_counts"] >= min_counts].copy()

    return _finish_conversion(
        "starmap",
        adata,
        output_path,
        notes=notes,
        empty_hint=_min_counts_hint(n_before, adata.n_obs, min_counts),
        count_filter=(n_before, min_counts),
    )


def convert_seqfish(
    expr_path: str,
    coords_path: str,
    output_path: str,
    min_counts: int = 5,
) -> str:
    """Convert seqFISH / seqFISH+ data to MCP-compatible h5ad.

    Args:
        expr_path: Path to expression matrix (cells x genes CSV).
        coords_path: Path to cell positions file (cells x [x, y] CSV).
        output_path: Path where the output h5ad file will be saved.
        min_counts: Minimum total counts per cell to keep.

    Returns:
        str: JSON report with conversion details and output path.

    """
    # seqFISH uses same structure as STARmap
    return convert_starmap(expr_path, coords_path, output_path, min_counts=min_counts)


def convert_generic_csv(
    expr_path: str,
    coords_path: str | None = None,
    output_path: str = "",
    x_col: str = "x",
    y_col: str = "y",
    transpose: bool = False,
    sep: str = ",",
    min_counts: int = 5,
) -> str:
    """Convert generic CSV expression matrix (+ optional coordinate file) to MCP-compatible h5ad.

    Handles any tabular spatial transcriptomics data where expression is in a matrix
    and coordinates are either in a separate file or embedded as columns.

    Args:
        expr_path: Path to expression matrix CSV/TSV (cells x genes, or genes x cells with transpose=True).
        coords_path: Path to coordinates CSV. If None, looks for x/y columns in the expression file.
        output_path: Path where the output h5ad file will be saved.
        x_col: Column name for x coordinates.
        y_col: Column name for y coordinates.
        transpose: If True, transpose the expression matrix (genes x cells -> cells x genes).
        sep: Delimiter character.
        min_counts: Minimum total counts per cell to keep.

    Returns:
        str: JSON report with conversion details and output path.

    """
    # output_path carries a "" default only so coords_path can precede it with a default (Python
    # requires no non-default arg after a defaulted one); it is genuinely required.
    #
    # The tool schema lists output_path as the second parameter, so an agent following it writes
    # convert_generic_csv("x.csv", "out.h5ad") -- binding the h5ad to coords_path. An .h5ad is never
    # a coordinates table, so that call is read the way its author meant it, and the report says so;
    # a missing output_path is a JSON error like every other refusal here, not a raise (hunt
    # 2026-09-30, u22-spatial-pipeline-23).
    notes: list[str] = []
    if str(coords_path or "").lower().endswith(".h5ad") and not str(output_path or "").lower().endswith(".h5ad"):
        coords_path, output_path = (output_path or None), str(coords_path)
        notes.append(
            "the second positional argument named an .h5ad, so it was read as output_path (the order the tool "
            "schema lists); pass coords_path= and output_path= by keyword to be explicit"
        )
    if not output_path:
        return _conversion_error(
            "generic_csv", "output_path is required: name the .h5ad to write, e.g. output_path='converted.h5ad'"
        )
    expr = pd.read_csv(expr_path, sep=sep, index_col=0)
    if transpose:
        expr = expr.T
    if coords_path:
        coords = pd.read_csv(coords_path, sep=sep, index_col=0)
        err = _require_columns(
            coords, [x_col, y_col], source_format="generic_csv", label=f"coordinates file {Path(coords_path).name}"
        )
        if err:
            return err
        expr, coords, align_notes, err = _align_on_shared_ids(
            expr,
            coords,
            source_format="generic_csv",
            left_label=f"expression file {Path(expr_path).name}",
            right_label=f"coordinates file {Path(coords_path).name}",
        )
        if err:
            return err
        notes.extend(align_notes)
        coord_vals = coords[[x_col, y_col]].values.astype(np.float64)
        gene_cols = expr.columns.tolist()
    elif x_col in expr.columns and y_col in expr.columns:
        coord_vals = expr[[x_col, y_col]].values.astype(np.float64)
        gene_cols = [c for c in expr.columns if c not in (x_col, y_col)]
        expr = expr[gene_cols]
    else:
        return _conversion_error(
            "generic_csv",
            f"no coordinates file was given and the expression file {Path(expr_path).name} has no "
            f"'{x_col}'/'{y_col}' columns. Columns present: {_preview(list(expr.columns))}. Pass "
            "coords_path, or x_col/y_col matching this file. (If the table is genes x cells, "
            "pass transpose=True.)",
        )

    adata = ad.AnnData(
        X=sp.csr_matrix(expr.values.astype(np.float32)),
        obs=pd.DataFrame(index=expr.index.astype(str)),
        var=pd.DataFrame(index=gene_cols),
    )
    adata.obsm["spatial"] = coord_vals

    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
    n_before = adata.n_obs
    adata = adata[adata.obs["total_counts"] >= min_counts].copy()

    return _finish_conversion(
        "generic_csv",
        adata,
        output_path,
        notes=notes,
        empty_hint=_min_counts_hint(n_before, adata.n_obs, min_counts),
        count_filter=(n_before, min_counts),
    )


# --------------------------------------------------------------------------- #
# Rscript discovery
# --------------------------------------------------------------------------- #
# The same three envs, in the same order, that ``tools/data_converter_worker.py`` looks in. Both are
# doors onto one script -- ``tools/r_to_h5ad_converter.R`` -- so both need an R that has Seurat, and
# a box where only one of them can find it converts differently through the portal than through the
# Python API. They stay two copies because the worker runs inside a per-tool env that cannot import
# this package; they are pinned against each other by
# ``test/test_the_agent_side_converter_finds_its_rscript_on_a_relocated_box.py``.
_RSCRIPT_CANDIDATES = [
    "/opt/conda/envs/seurat_env/bin/Rscript",
    "/opt/conda/envs/celltrek/bin/Rscript",
    "/opt/conda/envs/spatialomicsgym_e1/bin/Rscript",
]


def _repair(path: str) -> str:
    """This box's copy of ``path``, via the repair every launcher in the system already applies.

    The candidates above are the *build* box's paths. A deployment that renamed the shared env, or
    that keeps conda somewhere other than ``/opt/conda``, has the same R under a different absolute
    path -- and an unrepaired walk misses it and falls through to the bare name, which is **system**
    R and need not have Seurat. The failure then happens inside R, as a package-not-found error
    about the user's data.

    Import is function-level: ``sog_install.__init__`` runs ``ensure_repo_importable()``
    at import time, which this module has no reason to trigger. Any failure to resolve leaves the
    configured path alone -- the operator is still told what was actually tried.
    """
    try:
        from sog_install.constants import interpreter_on_this_box

        return interpreter_on_this_box(path)
    except Exception:
        return path


def _find_rscript(rscript_path: str | None = None) -> str:
    """The Rscript to launch: the caller's, else a shipped candidate this box has, else ``Rscript``.

    A caller-supplied ``rscript_path`` outranks discovery entirely -- it is repaired, exactly as a
    shipped candidate is, but never replaced by one, so naming an interpreter always selects it.

    The bare terminal fallback is deliberate and is the package-side contract: ``utils/execution.py``
    runs R as ``["Rscript", ...]`` too, so a deployment whose R is on ``PATH`` and nowhere
    conda-shaped is a supported box. What changed is that it is now reached only when this box
    genuinely has none of the candidates, rather than whenever their build-box spelling is stale.
    """
    if rscript_path:
        return _repair(rscript_path)
    for candidate in _RSCRIPT_CANDIDATES:
        repaired = _repair(candidate)
        if Path(repaired).is_file():
            return repaired
    return "Rscript"


def _find_r_converter_script() -> Path | None:
    """Locate ``tools/r_to_h5ad_converter.R``: the checkout sibling first, then the platform copy.

    The sibling walk is byte-identical to the old inline derivation and answers every checkout;
    on a pip-only install it lands in site-packages, where the platform rungs (a seeded
    SOG_HOME, the wheel's read-only ``_platform`` copy) carry the same ``tools/`` layout.
    ``None`` when neither has the script, so the caller's error can say where it looked.
    """
    checkout_copy = Path(__file__).parent.parent.parent / "tools" / "r_to_h5ad_converter.R"
    if checkout_copy.exists():
        return checkout_copy
    from spatialomicsgym import platform_root

    platform_tools = platform_root.tools_dir()
    if platform_tools is not None:
        candidate = platform_tools / "r_to_h5ad_converter.R"
        if candidate.exists():
            return candidate
    return None


def convert_r_object(
    r_file_path: str,
    output_path: str,
    assay: str = "RNA",
    slot: str = "counts",
    object_name: str | None = None,
    rscript_path: str | None = None,
    timeout: int | None = None,
) -> str:
    """Convert R objects (.rds, .rda, .RData) to MCP-compatible h5ad.

    Supports Seurat objects (v3/v4/v5), dgCMatrix sparse matrices, dense matrices,
    and data.frames. For Seurat objects, extracts counts, metadata, spatial coordinates,
    dimensionality reductions, and images. Uses R subprocess for extraction then
    assembles h5ad in Python.

    Args:
        r_file_path: Path to .rds or .rda/.RData file.
        output_path: Path where the output h5ad file will be saved.
        assay: Seurat assay to extract (default 'RNA'; auto-falls back to 'Spatial' or first available).
        slot: Seurat slot/layer to extract ('counts', 'data', or 'scale.data').
        object_name: For .rda files with multiple objects, name of the object to extract.
        rscript_path: Path to an Rscript binary that has Seurat. If None, the R environments this
            project provisions are tried in turn and then the bare name on PATH.
        timeout: Seconds the R extraction may run before it is stopped. If None, the
            ``SOG_R_CONVERT_TIMEOUT_SECONDS`` environment variable, else 300. A large Seurat object
            (100k+ cells) can need more.

    Returns:
        str: JSON report with conversion details and output path.

    """
    import subprocess
    import tempfile

    from spatialomicsgym.utils.execution import _r_child_env

    caller_supplied = bool(rscript_path)
    rscript_path = _find_rscript(rscript_path)
    # The limit was a literal 300 with no parameter, and the TimeoutExpired it raised is not an
    # OSError, so it escaped the JSON contract naming no knob -- and an object needing longer could
    # not be converted on this path at all (hunt 2026-09-30, u22-spatial-pipeline-13).
    try:
        limit = _r_convert_timeout(timeout)
    except ValueError as exc:
        return json.dumps({"status": "error", "message": str(exc)}, indent=2)

    converter_script = _find_r_converter_script()
    if converter_script is None:
        from spatialomicsgym import platform_root

        return json.dumps(
            {
                "status": "error",
                "message": f"R converter script tools/r_to_h5ad_converter.R not found: {platform_root.describe_search()}",
            }
        )

    # Create temp directory for intermediate files
    tmp_dir = tempfile.mkdtemp(prefix="r_to_h5ad_")
    try:
        # Run R converter
        cmd = [
            rscript_path,
            str(converter_script),
            "--input",
            r_file_path,
            "--output-dir",
            tmp_dir,
            "--assay",
            assay,
            "--slot",
            slot,
        ]
        if object_name:
            cmd.extend(["--object-name", object_name])

        try:
            # Under a C ctype, jsonlite <xx>-escapes every non-ASCII byte in the summary JSON this
            # function parses (metadata columns, reduction names). Same locale pin as run_r_code
            # and base_mcp._worker_env; reused from utils.execution rather than copied.
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=limit, env=_r_child_env())
        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "status": "error",
                    "message": (
                        f"The R extraction of {Path(r_file_path).name} did not finish within {limit} s and was "
                        "stopped; nothing was written. Raise the limit with convert_r_object(..., timeout=<seconds>) "
                        f"or the {_R_TIMEOUT_ENV} environment variable."
                    ),
                    "timeout_seconds": limit,
                },
                indent=2,
            )
        except OSError as exc:
            # An interpreter that cannot be launched at all. Without this the errno escaped the
            # function -- the `finally` below is the only handler -- and surfaced through callers'
            # generic except as a bare "[Errno 2] ... 'Rscript'", which names no step. The adjacent
            # non-zero-exit branch already returns a diagnosed payload; this is the same courtesy
            # for the failure one moment earlier.
            origin = (
                "supplied as rscript_path"
                if caller_supplied
                else "chosen by auto-detection (no provisioned R environment was found on this "
                "machine, so the bare name was tried on PATH)"
            )
            return json.dumps(
                {
                    "status": "error",
                    "message": (
                        f"R interpreter could not be launched: {rscript_path} ({origin}). "
                        f"Install R with Seurat, or pass rscript_path pointing at an Rscript that has it."
                    ),
                    "rscript_path": rscript_path,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
            )

        # Parse R output. The success payload is `toJSON(..., pretty = TRUE)` — indented across many
        # lines — so a line-at-a-time parse matches nothing and leaves r_info empty (and can even
        # match a lone array element, yielding a bare string). Take the last complete JSON object in
        # the stream instead; log_msg() writes to stderr, so stdout is the payload plus whatever
        # R may have printed before it.
        r_info = _parse_last_json_object(proc.stdout)

        # The exporter's own error comes first: a failed extraction now exits 1 AND prints its
        # {status: error, message} (u29a-mcp-transport-3), and checking the exit code first replaced
        # that message with a bare "R converter failed (exit 1)" (review of u29a-mcp-transport-3).
        if r_info.get("status") == "error":
            if proc.stderr and "stderr" not in r_info:
                r_info["stderr"] = proc.stderr[-2000:]
            return json.dumps(r_info, indent=2)

        if proc.returncode != 0:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"R converter failed (exit {proc.returncode})",
                    "stderr": proc.stderr[-2000:] if proc.stderr else "",
                },
                indent=2,
            )

        # Assemble h5ad from exported files
        adata = _assemble_h5ad_from_r_export(tmp_dir)

        # has_spatial/has_images describe the h5ad that was just written, so measure them from it
        # rather than repeating the R object's own claim about itself.
        images = next(iter(adata.uns.get("spatial", {}).values()), {}).get("images", {})

        # Everything the R script had to decide for itself rather than read off the object -- an
        # assumed orientation, invented barcodes, invented gene names. It writes them to a stdout
        # nobody downstream reads, so they have to be carried onto the report; the worker route
        # (data_converter_worker.convert_rds_to_h5ad) already does exactly this. A payload from
        # another process is untrusted input: accept jsonlite's unboxed single string, and treat
        # anything else as no notes rather than losing a conversion that succeeded.
        r_notes = r_info.get("warnings")
        if isinstance(r_notes, str):
            r_notes = [r_notes]
        elif not isinstance(r_notes, list):
            r_notes = []

        return _finish_conversion(
            "r_object",
            adata,
            output_path,
            extra={
                "r_class": r_info.get("class", "unknown"),
                "assay_used": r_info.get("assay_used"),
                "has_images": bool(images),
                "image_resolutions": sorted(images),
                "reductions": r_info.get("reductions", []),
                "spatial_coords_provenance": adata.uns.get("spatial_coords_provenance"),
            },
            notes=[str(n) for n in r_notes],
        )
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


_R_TIMEOUT_ENV = "SOG_R_CONVERT_TIMEOUT_SECONDS"
_R_TIMEOUT_DEFAULT = 300


def _r_convert_timeout(timeout: int | None) -> int:
    """The caller's limit, else the environment's, else 300 s. A value that is set and unusable is
    a ValueError naming where it came from, never quietly replaced by the default."""
    for source, candidate in (("timeout", timeout), (_R_TIMEOUT_ENV, os.environ.get(_R_TIMEOUT_ENV))):
        if candidate is None or (isinstance(candidate, str) and not candidate.strip()):
            continue
        # OverflowError too: int(float("inf")) raises it, and it escaped convert_r_object's JSON
        # contract exactly as the TimeoutExpired this knob was added for did (hunt 2026-09-30,
        # u22-spatial-pipeline-13).
        try:
            value = int(float(candidate))
        except (TypeError, ValueError, OverflowError):
            value = 0
        if value <= 0:
            raise ValueError(f"{source}={candidate!r} is not a positive number of seconds")
        return value
    return _R_TIMEOUT_DEFAULT


def _parse_last_json_object(stdout: str) -> dict:
    """Return the last complete top-level JSON *object* in `stdout`, or {} if there is none.

    Scans from each line that starts an object and takes the longest successful parse, so a
    pretty-printed multi-line payload is recovered whether or not other output precedes it.
    """
    text = stdout.strip()
    if not text:
        return {}
    # Fast path: the whole stream is the payload (what the R converter emits on success).
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1] if "{" in text and "}" in text else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed

    # Fall back to scanning for the last brace-balanced object in the stream.
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for start in reversed(starts):
        for end in range(len(text), start, -1):
            if text[end - 1] != "}":
                continue
            try:
                parsed = json.loads(text[start:end])
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def _assemble_h5ad_from_r_export(export_dir: str) -> ad.AnnData:
    """Assemble an AnnData from R converter's exported flat files."""
    from scipy.io import mmread

    export_path = Path(export_dir)

    # Read matrix
    mtx_file = export_path / "matrix.mtx.gz"
    if not mtx_file.exists():
        raise FileNotFoundError(f"matrix.mtx.gz not found in {export_dir}")

    mat = mmread(str(mtx_file))
    mat = sp.csc_matrix(mat).T  # R writes genes x cells, we need cells x genes

    # Read barcodes
    barcodes_file = export_path / "barcodes.tsv.gz"
    barcodes = pd.read_csv(barcodes_file, sep="\t", header=None, compression="gzip")
    cell_names = barcodes.iloc[:, 0].astype(str).tolist()

    # Read features
    features_file = export_path / "features.tsv.gz"
    features = pd.read_csv(features_file, sep="\t", header=None, compression="gzip")
    gene_names = (
        features.iloc[:, 1].astype(str).tolist() if features.shape[1] > 1 else features.iloc[:, 0].astype(str).tolist()
    )
    gene_ids = features.iloc[:, 0].astype(str).tolist()

    # Create AnnData
    adata = ad.AnnData(
        X=sp.csr_matrix(mat.astype(np.float32)),
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names),
    )
    if gene_ids != gene_names:
        adata.var["gene_ids"] = gene_ids

    # Read metadata
    meta_file = export_path / "metadata.csv"
    if meta_file.exists():
        meta = pd.read_csv(meta_file, index_col=None)
        if "barcode" in meta.columns:
            meta = meta.set_index("barcode")
        # Align to adata
        common = adata.obs_names.intersection(meta.index)
        if len(common) > 0:
            for col in meta.columns:
                adata.obs[col] = meta.loc[adata.obs_names, col].values if len(common) == adata.n_obs else np.nan

    # Read spatial coordinates
    spatial_file = export_path / "spatial_coords.csv"
    if spatial_file.exists():
        coords = pd.read_csv(spatial_file, index_col=None)
        if "barcode" in coords.columns:
            coords = coords.set_index("barcode")
        # Get numeric columns for coordinates
        num_cols = coords.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            # Order the two axes by column NAME. Seurat hands back `imagerow` (the row, i.e. y)
            # first, so taking num_cols[:2] in file order silently transposes every Visium object.
            ordered, note = _order_coordinate_columns_by_name(num_cols)
            # Take the barcode-aligned .loc branch only when every obs name is actually in coords.index —
            # a same-length coords with a non-barcode index (e.g. a default RangeIndex) makes
            # .loc[adata.obs_names] raise KeyError. Otherwise align positionally.
            coord_vals = (
                coords.loc[adata.obs_names, ordered].values.astype(np.float64)
                if adata.obs_names.isin(coords.index).all()
                else coords[ordered].values[: adata.n_obs].astype(np.float64)
            )
            adata.obsm["spatial"] = coord_vals
            adata.uns["spatial_coords_provenance"] = note

    # Read reductions
    for red_file in export_path.glob("reduction_*.csv"):
        red_name = red_file.stem.replace("reduction_", "")
        red_df = pd.read_csv(red_file, index_col=None)
        if "barcode" in red_df.columns:
            red_df = red_df.set_index("barcode")
        num_cols = red_df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols and len(red_df) == adata.n_obs:
            adata.obsm[f"X_{red_name}"] = red_df[num_cols].values.astype(np.float64)

    # Read scalefactors and images
    sf_file = export_path / "scalefactors.json"
    img_dir = export_path / "images"

    if sf_file.exists() or img_dir.exists():
        spatial_dict: dict = {}
        lib_id = "spatial_sample"

        if sf_file.exists():
            with open(sf_file) as f:
                sf = json.load(f)
            spatial_dict["scalefactors"] = sf

        if img_dir.exists():
            from PIL import Image as PILImage

            images_dict = {}
            for img_file in sorted(img_dir.iterdir()):
                if img_file.suffix.lower() not in (".png", ".jpg", ".tif"):
                    continue
                img_arr = np.array(PILImage.open(str(img_file)))
                if img_arr.dtype == np.uint8:
                    img_arr = img_arr.astype(np.float32) / 255.0
                # Resolution comes from the filename, never from iteration order. A Seurat image slot
                # holds the raster Read10X_Image loaded, and that defaults to tissue_lowres_image.png
                # — Seurat keeps no hires raster — so an unqualified name is lowres, not hires.
                # Labelling it 'hires' pairs it with tissue_hires_scalef and puts every spot overlay
                # off-image by the ratio of the two scale factors.
                name = img_file.stem.lower()
                key = "hires" if "hires" in name else "lowres"
                images_dict[key] = img_arr
            if images_dict:
                spatial_dict["images"] = images_dict

        if spatial_dict:
            adata.uns["spatial"] = {lib_id: spatial_dict}

    return adata


def auto_convert(input_path: str, output_path: str, **kwargs) -> str:
    """Auto-detect spatial transcriptomics data format and convert to MCP-compatible h5ad.

    Examines the input path (file or directory) and dispatches to the appropriate converter.

    Args:
        input_path: Path to input file or directory.
        output_path: Path where the output h5ad file will be saved.
        **kwargs: Additional arguments passed to the detected converter. A keyword that converter
            does not take is reported under ``ignored_parameters`` rather than raised.

    Returns:
        str: JSON report with detected format, conversion details, and output path.

    """
    p = Path(input_path)
    # Every suffix test below reads from here. They used to be a mix of case-sensitive `p.suffix`
    # comparisons and one `.lower()` call, so a `.CSV` off a Windows export and a `.H5AD` off a
    # collaborator both fell through to "Could not auto-detect format" -- and the compressed set knew
    # only `.csv.gz`, although a GEO-deposited count matrix is nearly always `.tsv.gz`. pandas
    # decompresses by extension, so those files convert unaided once they are dispatched at all.
    lname = p.name.lower()

    # Case 1: Already an h5ad file. Nothing here takes a keyword, so any that came are named.
    if lname.endswith(".h5ad"):
        report = json.loads(validate_spatial_h5ad(str(p)))
        if report["status"] == "valid" and not report["issues"]:
            # Already valid: copy it -- or nothing at all when the output IS the input, which
            # raised SameFileError for a file that needed no work (u22-spatial-pipeline-7).
            copied = _copy_atomically(p, output_path)
            if copied:  # never stamp the user's own file: output_path IS the input when not copied
                stamp_conversion(output_path, producer="auto_convert", source=str(p))
            result = {
                "status": "success",
                "format_detected": "h5ad_valid",
                "action": "copied" if copied else "none (output_path is the input, which is already valid)",
                "output_path": output_path,
            }
        else:
            result = json.loads(repair_spatial_h5ad(str(p), output_path))
            result["format_detected"] = "h5ad_repaired"
        return _with_ignored(json.dumps(result, indent=2), sorted(kwargs), "an existing h5ad")

    # Case 2: Space Ranger directory. Any of the four count-matrix layouts Space Ranger can leave
    # behind counts as one, under the sample-name prefix 10x downloads carry too (u22-spatial-
    # pipeline-1) — convert_visium_spaceranger resolves them through the same helper — so
    # detection and conversion accept the same set and a detected directory always converts.
    if p.is_dir():
        spatial_dir = p / "spatial"
        try:
            counts, rivals = _spaceranger_counts(p)
        except OSError:
            counts, rivals = None, []
        if spatial_dir.exists() and (counts is not None or rivals):
            return _dispatch(convert_visium_spaceranger, str(p), output_path, **kwargs)

    # Case 2b: a 10x HDF5 counts file whose spatial/ directory is named alongside it. Case 2 needs
    # both under one parent AND the counts named the way Space Ranger names them, so a run
    # re-exported as `mybrain.h5` beside `spatial/`, or split across two folders, matches neither.
    # `convert_visium_h5_spatial` has existed for exactly this case and nothing ever dispatched to
    # it; the caller names the pair, this line routes it.
    if lname.endswith(".h5") and "spatial_dir" in kwargs:
        # This converter takes exactly three paths; any other keyword is reported, not forwarded.
        return _dispatch(convert_visium_h5_spatial, str(p), str(kwargs.pop("spatial_dir")), output_path, **kwargs)

    # Every branch below goes through `_dispatch`. Forwarding **kwargs unfiltered turned the
    # schema's own optional parameters into TypeErrors -- min_counts into convert_r_object,
    # bin_size into convert_generic_csv -- and on the h5ad and Space Ranger branches they vanished
    # without a word (hunt 2026-09-30, u22-spatial-pipeline-9).

    # Case 3: R objects (.rds, .rda, .RData)
    if lname.endswith((".rds", ".rda", ".rdata")):
        return _dispatch(convert_r_object, str(p), output_path, **kwargs)

    # Case 4: GEM/GEF file (Stereo-seq)
    if lname.endswith((".gem", ".gef", ".gem.gz")):
        return _dispatch(convert_stereoseq, str(p), output_path, **kwargs)

    # Case 4: Parquet file (likely Xenium transcripts)
    if lname.endswith(".parquet"):
        return _dispatch(convert_xenium, str(p), output_path, **kwargs)

    # Case 5: CSV/TSV - try to detect type
    if lname.endswith(TABULAR_SUFFIXES):
        # A companion path is the caller STATING what the layout is; a column sniff is this
        # function GUESSING it. The statement is honoured first, and before the file is opened: a
        # Slide-seq DGE is genes x beads, so "read five rows" is five rows of forty thousand
        # columns, spent only to reach a branch the keyword had already decided. Each converter
        # below reads its own two files with its own rules and accepts neither `sep` nor
        # `coords_path` nor `x_col`, so nothing here may fall through into the generic reader.
        if "cell_metadata_path" in kwargs:  # Vizgen MERFISH: cell_by_gene.csv + cell_metadata.csv
            return _dispatch(convert_merfish, str(p), kwargs.pop("cell_metadata_path"), output_path, **kwargs)
        if "fov_positions_path" in kwargs:  # Nanostring CosMx: exprMat_file.csv + metadata_file.csv
            # The column sniff that used to guard this required `fov` or `centerx_global_px` among
            # the EXPRESSION matrix's columns, where neither lives -- both are columns of the
            # metadata file this branch is being handed. So the guard could only be satisfied by
            # accident, and a caller who named both CosMx files got a generic-CSV coordinates
            # lecture about the one that is not the coordinates.
            return _dispatch(convert_cosmx, str(p), kwargs.pop("fov_positions_path"), output_path, **kwargs)
        if "bead_locations_path" in kwargs:  # Slide-seq / Slide-seqV2: DGE + bead locations
            return _dispatch(convert_slideseq, str(p), kwargs.pop("bead_locations_path"), output_path, **kwargs)
        if "starmap_coords_path" in kwargs:  # STARmap / seqFISH: matrix + 2D or 3D coordinates
            # Its own keyword rather than `coords_path`, because the two routes differ in what they
            # keep: `convert_generic_csv` takes exactly two coordinate columns, `convert_starmap`
            # stores a third as `obsm["spatial_3d"]` as well. A z axis dropped in silence is a loss
            # nothing downstream can notice, so this choice is stated by the caller, never inferred
            # from a shape that a 2D generic pair shares.
            return _dispatch(convert_starmap, str(p), kwargs.pop("starmap_coords_path"), output_path, **kwargs)

        # Detection and conversion must read the file the same way. The head read used tabs for a
        # `.tsv`, matched on the real column names, and then handed the file to `convert_generic_csv`,
        # whose default `sep` is a comma -- the whole line parsed as a single column, so the user was
        # told the coordinate columns were missing from a file that plainly has them.
        sep = _table_delimiter(p, kwargs.get("sep"))
        df_head = pd.read_csv(p, nrows=5, sep=sep)
        cols_lower = [c.lower() for c in df_head.columns]

        # Xenium transcripts
        if "cell_id" in cols_lower and "feature_name" in cols_lower and "x_location" in cols_lower:
            return _dispatch(convert_xenium, str(p), output_path, **kwargs)

        # A GEM-shaped table -- a Stereo-seq lasso export `geneID,x,y,MIDCount`, one row per gene per
        # position. The diagnosis routes it to `convert_stereoseq`; this branch sent it to the generic
        # reader, which made geneID the index and MIDCount the one "gene" (hunt 2026-09-30,
        # u22-spatial-pipeline-12). One rule now, `_is_gem_table`, for both entry points.
        if _is_gem_table(cols_lower):
            return _dispatch(convert_stereoseq, str(p), output_path, **kwargs)

        # Generic CSV with coordinates. Only this branch takes `sep`; the vendor converters above
        # read their own fixed formats and would reject the keyword.
        coords_path = kwargs.pop("coords_path", None)
        if kwargs.get("sep") is None:
            kwargs["sep"] = sep
        return _dispatch(convert_generic_csv, str(p), coords_path, output_path, **kwargs)

    return _with_ignored(
        json.dumps({"status": "error", "message": f"Could not auto-detect format for: {input_path}"}),
        sorted(kwargs),
        "an unrecognised input",
    )


def _dispatch(converter, *args, **kwargs) -> str:
    """Call ``converter`` with the keywords it takes, and name the ones it does not in its report."""
    import inspect

    params = inspect.signature(converter).parameters
    accepted = {k: v for k, v in kwargs.items() if k in params}
    ignored = sorted(k for k in kwargs if k not in params)
    return _with_ignored(converter(*args, **accepted), ignored, converter.__name__)


def _with_ignored(report: str, ignored: list[str], taker: str) -> str:
    """``report`` with ``ignored_parameters`` and a warning naming them, when there are any."""
    if not ignored:
        return report
    try:
        payload = json.loads(report)
    except ValueError:
        return report
    if not isinstance(payload, dict):
        return report
    payload["ignored_parameters"] = list(ignored)
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        warnings = [] if warnings is None else [str(warnings)]
    warnings.append(f"{taker} takes no {', '.join(ignored)}: passed but not used")
    payload["warnings"] = warnings
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


# (x-axis name, y-axis name) pairs, most specific first. `imagecol`/`imagerow` is Seurat's Visium
# naming, `pxl_col_in_fullres`/`pxl_row_in_fullres` is Space Ranger's. Both put the ROW (y) first in
# the file, which is exactly why these must be selected by name and never by position.
#: Names a z column goes by, lowercase, matched case-insensitively: the third-axis counterpart of
#: ``_COORD_NAME_PAIRS``. ``sog_portal/ingest.py`` keeps an equal copy (it may not import this module at
#: load time), and a test holds the two equal.
_Z_NAMES: tuple[str, ...] = (
    "z",
    "z_um",
    "z_microns",
    "z_location",
    "z_position",
    "z_centroid",
    "zcoord",
    "z_coord",
    "global_z",
    "centroid_z",
    "center_z",
)

_COORD_NAME_PAIRS: tuple[tuple[str, str], ...] = (
    ("imagecol", "imagerow"),
    ("pxl_col_in_fullres", "pxl_row_in_fullres"),
    ("x_centroid", "y_centroid"),
    ("center_x", "center_y"),
    # Slide-seq's own naming, in `BeadLocationsForR.csv` and `Puck_*_bead_locations.csv`. Its absence
    # is why every stock puck reached the positional fallback below and was placed correctly only
    # because the barcode happens to be column 0 and x happens to precede y.
    ("xcoord", "ycoord"),
    ("x", "y"),
)


def _order_coordinate_columns_by_name(num_cols: list[str]) -> tuple[list[str], str]:
    """Order two coordinate columns as (x, y) by name, falling back to file order.

    Returns the ordered column names and a provenance note recording how the choice was made, so a
    positional fallback -- which may well be (y, x) -- is visible to the caller rather than silent.
    """
    by_lower = {str(c).strip().lower(): c for c in num_cols}
    for x_name, y_name in _COORD_NAME_PAIRS:
        if x_name in by_lower and y_name in by_lower:
            return [by_lower[x_name], by_lower[y_name]], f"columns matched by name: x={x_name}, y={y_name}"
    fallback = list(num_cols[:2])
    return fallback, (
        f"WARNING: no recognised coordinate column names among {list(num_cols)}; "
        f"used file column order {fallback} as (x, y), which may be transposed"
    )


def _preview(values, limit: int = 25) -> str:
    """Render a column/index listing for an error message without pasting a 1000-gene header into it."""
    vals = [str(v) for v in values]
    if len(vals) <= limit:
        return str(vals)
    return f"{vals[:limit]} ... (+{len(vals) - limit} more)"


def _conversion_error(source_format: str, message: str, extra: dict | None = None) -> str:
    """Build the documented failure payload.

    Every converter in this module documents "Returns: JSON report", and `spatial_pipeline`, the MCP
    portal and the agent all dispatch on `status`. A bare `KeyError` escaping one of these functions
    reaches them as an opaque traceback naming neither the column nor the file at fault, so a wrong
    column name is indistinguishable from a broken tool.
    """
    payload: dict[str, Any] = {"status": "error", "source_format": source_format, "message": message}
    if extra:
        payload.update(extra)
    return json.dumps(payload, indent=2)


def _require_columns(frame: pd.DataFrame, wanted, *, source_format: str, label: str) -> str | None:
    """Return a JSON error naming the missing *and* the available columns, or None if all are present.

    The available list is the actionable half: a caller who guessed `center_x` on a file that says
    `x` can only fix it if the report says what the file really has.
    """
    missing = [c for c in dict.fromkeys(wanted) if c is not None and c not in frame.columns]
    if not missing:
        return None
    return _conversion_error(
        source_format,
        f"{label} has no column(s) {missing}. Columns present: {_preview(list(frame.columns))}. "
        "Pass the matching column name(s) explicitly.",
    )


def _align_on_shared_ids(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    source_format: str,
    left_label: str,
    right_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str | None]:
    """Restrict two indexed frames to the ids they share, refusing to guess when they do not line up.

    Returns ``(left, right, notes, error_json)``. Two failure modes are refused outright rather than
    converted:

    * **Duplicate ids.** ``.loc`` on a duplicated label returns *every* matching row, so one file's
      counts get paired with another row's coordinates. A multi-slide CosMx export restarts
      ``fov``/``cell_ID`` numbering on each slide, which makes ``fov_cell_ID`` non-unique.
    * **No overlap at all.** The intersection is then empty and the conversion yields zero cells --
      the usual causes being a genes x cells matrix and a barcode-suffix mismatch, both of which are
      fixable once named.

    A *partial* overlap is legal (a coordinate table filtered to in-tissue beads, say) but is
    recorded in ``notes``: losing part of a tissue changes every number computed from it.
    """
    for frame, label in ((left, left_label), (right, right_label)):
        if not frame.index.is_unique:
            dups = [str(v) for v in frame.index[frame.index.duplicated()].unique()[:5]]
            return (
                left,
                right,
                [],
                _conversion_error(
                    source_format,
                    f"{label} has duplicate cell ids (e.g. {dups}). Aligning on a duplicated label pairs "
                    "every copy with every other, so expression and coordinates would be mismatched "
                    "cell for cell. Multi-slide runs restart fov/cell numbering -- make the id unique "
                    "(prefix it with the slide) before converting.",
                ),
            )

    common = left.index.intersection(right.index)
    if len(common) == 0:
        return (
            left,
            right,
            [],
            _conversion_error(
                source_format,
                f"no cell ids are shared between {left_label} ({len(left.index)} ids, e.g. "
                f"{_preview(list(left.index[:3]), 3)}) and {right_label} ({len(right.index)} ids, e.g. "
                f"{_preview(list(right.index[:3]), 3)}). Check that both files use the same id format, "
                "and that the expression matrix is stored cells x genes rather than genes x cells.",
            ),
        )

    notes = []
    for frame, label in ((left, left_label), (right, right_label)):
        dropped = len(frame.index) - len(common)
        if dropped > 0:
            notes.append(f"dropped {dropped} of {len(frame.index)} rows from {label}: no matching id in the other file")
    return left.loc[common], right.loc[common], notes, None


def _min_counts_hint(n_before: int, n_after: int, min_counts: int) -> str | None:
    """Explain an object emptied by the count filter, naming the threshold that emptied it."""
    if n_after == 0 and n_before > 0:
        return (
            f"all {n_before} observations were removed by the min_counts={min_counts} filter; lower "
            "min_counts, or check that .X holds raw counts rather than normalised values"
        )
    return None


def _min_counts_note(n_before: int, n_after: int, min_counts: int) -> str | None:
    """Report a count filter that dropped *some* observations, which the report otherwise hides.

    Only the emptied case used to be reported. A run that goes in with 4,992 spots and comes out
    with 2,310 reports ``n_obs: 2310`` and nothing else, so the person who uploaded the slide has no
    way to tell a sparse section from a filter they did not know was running. This is the same class
    of silence ``_conversion_report``'s ``notes`` exists for: everything the conversion decided on
    its own belongs in the report."""
    if n_before > 0 and 0 < n_after < n_before:
        dropped = n_before - n_after
        return (
            f"{dropped} of {n_before} observations were dropped by the min_counts={min_counts} "
            f"filter, leaving {n_after}; lower min_counts to keep low-count spots"
        )
    return None


def _finish_conversion(
    source_format: str,
    adata: ad.AnnData,
    output_path: str,
    *,
    extra: dict | None = None,
    notes: list[str] | None = None,
    empty_hint: str | None = None,
    count_filter: tuple[int, int] | None = None,
) -> str:
    """Standardise, refuse to write an empty object, write it, and report on what was written.

    An object with no cells or no genes is a *failed* conversion. Reporting it as
    ``"status": "success", "n_obs": 0`` is the worst outcome this module can produce: the agent reads
    only the report, so it moves on to the next analysis step, and the failure surfaces far from its
    cause -- or not at all. ``validate_spatial_h5ad`` already calls such a file ``"invalid"``, so
    without this guard the module contradicts itself.

    The empty file is not written, because anything left at ``output_path`` will eventually be picked
    up by a later step that never saw the report.
    """
    adata = _standardize_adata(adata)
    notes = list(notes or [])

    # ``count_filter`` is ``(n_before, min_counts)`` from the caller that ran ``sc.pp`` filtering.
    if count_filter is not None:
        dropped = _min_counts_note(int(count_filter[0]), int(adata.n_obs), int(count_filter[1]))
        if dropped:
            notes.append(dropped)

    if adata.n_obs and "spatial" in adata.obsm:
        try:
            coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
            n_nan = int(np.isnan(coords).any(axis=1).sum()) if coords.ndim == 2 else 0
        except (TypeError, ValueError):
            n_nan = 0
        if n_nan:
            notes.append(
                f"{n_nan} of {adata.n_obs} observations have a NaN spatial coordinate: those cells are "
                "not placed anywhere, and distance-based tools will either fail or ignore them"
            )

    if adata.n_obs == 0 or adata.n_vars == 0:
        reason = empty_hint or f"the conversion yielded {adata.n_obs} observations x {adata.n_vars} genes"
        return _conversion_error(
            source_format,
            f"conversion produced an empty object and was not written: {reason}",
            extra={"n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars), "warnings": notes},
        )

    # Beside the output and renamed onto it, never through it: a symlinked output_path used to be
    # written through into whatever it pointed at (u22-spatial-pipeline-7).
    _write_h5ad_atomically(adata, output_path)
    stamp_conversion(output_path, producer=f"spatial_data_converter ({source_format})")
    return _conversion_report(source_format, adata, output_path, extra=extra, notes=notes)


def _sniff_delimiter(path: Path, default: str = ",") -> str:
    """Best-effort delimiter for a text table whose extension does not say."""
    import csv
    import gzip

    try:
        opener = gzip.open if str(path).lower().endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:  # type: ignore[operator]
            sample = fh.read(8192)
    except (OSError, EOFError, UnicodeError):
        return default
    if not sample.strip():
        return default
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except (csv.Error, UnicodeError):
        header = sample.splitlines()[0] if sample.splitlines() else ""
        return next((c for c in ("\t", ",", ";", "|") if c in header), default)


def _header_fields(path: Path, delimiter: str) -> int:
    """How many fields the first line of `path` parses into under `delimiter`.

    Parsed with `csv.reader` rather than counted with `str.count`, so a quoted field containing the
    delimiter does not inflate the answer.
    """
    import csv
    import gzip

    try:
        opener = gzip.open if str(path).lower().endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:  # type: ignore[operator]
            header = fh.readline()
    except (OSError, EOFError, UnicodeError):
        return 0
    if not header.strip():
        return 0
    try:
        return len(next(csv.reader([header.rstrip("\r\n")], delimiter=delimiter)))
    except (csv.Error, StopIteration):
        return 0


def _table_delimiter(path: Path, override: str | None = None) -> str:
    """The delimiter a text table must be read with, so detection and conversion see the same columns.

    `auto_convert` used to sniff a `.tsv` header with tabs, match on the real column names, then hand
    the file to `convert_generic_csv`, whose default is a comma -- the whole line parsed as one
    column and the user was told the coordinate columns were missing from a file that has them.
    Making the extension decisive closed that case and left its mirror image open: a `.csv` that is
    really semicolon- or tab-separated (locale exports, and the many tools that write `.csv` with
    tabs) was still forced through a comma and produced the same one-column read, while
    `_diagnose_tabular` -- which sniffs -- reported the coordinate columns by name. One run told the
    user the file was fine and then that its columns were missing.

    So the extension is preferred, not decisive: it wins wherever it parses. Only a file whose name
    yields a SINGLE field, which is the signature of a delimiter that is not there, is sniffed --
    and the sniffed answer is taken only if it does better. A file that reads correctly today keeps
    the delimiter it has, including a genuinely one-column table.
    """
    if override:
        return override
    name = str(path).lower()
    if name.endswith((".tsv", ".tsv.gz", ".tab", ".tab.gz")):
        by_name = "\t"
    elif name.endswith((".csv", ".csv.gz")):
        by_name = ","
    else:
        return _sniff_delimiter(path)
    if _header_fields(path, by_name) > 1:
        return by_name
    sniffed = _sniff_delimiter(path, default=by_name)
    return sniffed if _header_fields(path, sniffed) > 1 else by_name


def _detect_coordinate_columns(df: pd.DataFrame) -> list[str] | None:
    """Detect spatial coordinate columns from a DataFrame, as ``[x, y]`` in the frame's own spelling.

    The hand-kept list here lacked two pairs ``_COORD_NAME_PAIRS`` -- this module's other list --
    already knew: Seurat's ``imagecol``/``imagerow`` and Slide-seq's ``xcoord``/``ycoord``. An h5ad
    carrying either in obs was diagnosed "missing spatial coordinates", and repair refused it as
    needing the original vendor output (hunt 2026-09-30, u22-spatial-pipeline-21). The pairs this
    list held keep their order, so a frame that matched before matches the same columns; the rest
    of ``_COORD_NAME_PAIRS`` follows, and names match case-insensitively, exact spelling first.
    """
    candidates = [
        ("x", "y"),
        ("X", "Y"),
        ("x_centroid", "y_centroid"),
        ("center_x", "center_y"),
        ("CenterX_global_px", "CenterY_global_px"),
        ("x_location", "y_location"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("x_bin", "y_bin"),
    ]
    candidates += [pair for pair in _COORD_NAME_PAIRS if pair not in candidates]
    by_lower: dict[str, Any] = {}
    for column in df.columns:
        by_lower.setdefault(str(column).strip().lower(), column)
    # `array_col`/`array_row` are deliberately absent. They are Visium's 0-77 / 0-127 capture-grid
    # indices, not pixels or microns: promoting them yields a unit-pitch lattice on which every
    # neighbour graph, Moran's I and spatially-variable-gene call is computed against a geometry the
    # tissue does not have — and, because the values are finite and 2-D, nothing downstream detects
    # it. An h5ad carrying only these has no usable coordinates and must be reported as such.
    # `spatial_pipeline._diagnose_h5ad` already refuses them; this is the same rule.
    for x_col, y_col in candidates:
        if x_col in df.columns and y_col in df.columns:
            return [x_col, y_col]
    for x_col, y_col in candidates:
        x_hit, y_hit = by_lower.get(x_col.lower()), by_lower.get(y_col.lower())
        if x_hit is not None and y_hit is not None:
            return [x_hit, y_hit]
    return None


def _attach_visium_spatial(adata: ad.AnnData, sr_path: Path) -> None:
    """Attach a Space Ranger `spatial/` folder to an MTX-loaded AnnData, in place.

    Deliberately mirrors `scanpy.read_visium` step for step so a run without its `.h5` produces the
    same object as one with it -- same `obsm['spatial']` ordering, same `uns['spatial']` layout, same
    `obs` columns. Any divergence here would make results depend on which files a deposit happened to
    include, which is exactly the class of silent inconsistency this module exists to prevent.
    """
    spatial_dir = sr_path / "spatial"
    positions = read_tissue_positions(spatial_dir)
    if positions is None:
        raise FileNotFoundError(f"No tissue_positions[_list].csv or tissue_positions.parquet in {spatial_dir}")
    # scanpy renames positionally, and its 4th/5th labels are swapped relative to 10x's file order;
    # selecting ["pxl_row_in_fullres", "pxl_col_in_fullres"] from *these* labels is what yields the
    # true (x, y). Reproduced verbatim so both paths agree.
    positions.columns = ["in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]
    adata.obs = adata.obs.join(positions, how="left")
    adata.obsm["spatial"] = adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()
    adata.obs.drop(columns=["pxl_row_in_fullres", "pxl_col_in_fullres"], inplace=True)

    library_id = sr_path.resolve().name or "spatial_sample"
    entry: dict[str, Any] = {"images": {}}
    scalefactors_file = spatial_dir / "scalefactors_json.json"
    if scalefactors_file.exists():
        entry["scalefactors"] = json.loads(scalefactors_file.read_bytes())

    from matplotlib.image import imread

    for res in ("hires", "lowres"):
        img = spatial_dir / f"tissue_{res}_image.png"
        if img.exists():
            entry["images"][res] = imread(str(img))
    adata.uns["spatial"] = {library_id: entry}


def _load_visium_images(adata: ad.AnnData, spatial_dir: Path):
    """Load Visium H&E images and scale factors into adata.uns['spatial']."""
    scalefactors_file = spatial_dir / "scalefactors_json.json"
    hires_image = spatial_dir / "tissue_hires_image.png"
    lowres_image = spatial_dir / "tissue_lowres_image.png"

    if not scalefactors_file.exists():
        return

    import json as _json

    with open(scalefactors_file) as f:
        scalefactors = _json.load(f)

    sample_id = "spatial_sample"
    adata.uns["spatial"] = {sample_id: {"scalefactors": scalefactors, "images": {}}}

    try:
        from matplotlib.image import imread

        if hires_image.exists():
            adata.uns["spatial"][sample_id]["images"]["hires"] = imread(str(hires_image))
        if lowres_image.exists():
            adata.uns["spatial"][sample_id]["images"]["lowres"] = imread(str(lowres_image))
    except ImportError:
        pass


def _standardize_adata(adata: ad.AnnData) -> ad.AnnData:
    """Standardize an AnnData object for MCP compatibility."""
    adata.var_names_make_unique()

    if adata.X is not None and not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    if "total_counts" not in adata.obs or "n_genes_by_counts" not in adata.obs:
        sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)

    return adata


def _conversion_report(
    source_format: str,
    adata: ad.AnnData,
    output_path: str,
    extra: dict | None = None,
    notes: list[str] | None = None,
) -> str:
    """Build a standardized JSON conversion report.

    `extra` carries per-format annotations (bin size, the R object's class, ...). It is merged
    *underneath* the measured fields: those are read off the AnnData that was just written and are
    the ground truth about the file. `result.update(extra)` let a caller's default -- notably an
    empty R payload's `has_spatial: False` -- overwrite a measured `True`, so the report contradicted
    the h5ad it had produced.

    `notes` carries everything the conversion silently decided or discarded -- rows dropped for want
    of a matching id, coordinates that came out NaN. It is written last so nothing can overwrite it.
    """
    measured: dict[str, Any] = {
        "status": "success",
        "source_format": source_format,
        "output_path": output_path,
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "has_spatial": "spatial" in adata.obsm,
        "has_qc_metrics": "total_counts" in adata.obs,
        "has_visium_metadata": "spatial" in adata.uns,
        "sparse_matrix": sp.issparse(adata.X) if adata.X is not None else False,
    }
    result: dict[str, Any] = dict(extra) if extra else {}
    result.update(measured)
    if notes:
        result["warnings"] = list(notes)
    return json.dumps(result, indent=2)
