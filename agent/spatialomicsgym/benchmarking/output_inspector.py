"""Output Inspector — examines tool outputs and extracts prediction information.

This is the core of the benchmarking skill. It:
  1. Scans the output directory for files
  2. Matches files against tool-specific output profiles
  3. Validates prediction data is present and well-formed
  4. Returns a structured report for the evaluator

Must be called BEFORE evaluation when benchmarking mode is active.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


# Directories that hold a tool's *inputs*, or artifacts derived from a previous inspection. A file
# under one of these is never a prediction, and scoring one produces a real-looking number with no
# warning anywhere: four recorded DLPFC cells (run_iris and precast_spatial_clustering, across three
# models) all report ARI 0.3619 because all four scored the same staged `csv_inputs/metadata.csv`,
# whose `Cluster` column is 10x's own shipped clustering -- with the `layer_guess` ground truth
# sitting three columns away in the same file.
#
# This list used to be copy-pasted into each inspection branch, and the copies drifted to 9, 6, 10
# and 16 entries: every name was added in response to one incident, and only the branch involved in
# that incident learned it. Keep it here, once.
#
#   - converted_sc / converted_spatial / sc_csv / spatial_csv: data_converter h5ad->csv stagings
#   - csv_spatial / csv_reference / csv_filtered / csv_export(s) / csv_input(s) / converted_csv:
#     per-tool input CSV stagings
#   - _starfysh_input: the Visium-like tree starfysh_worker.py builds for Starfysh
#   - prep: coordinate/barcode stagings (e.g. prep/coords_barcode_xy.csv)
#   - celloscope_data: Celloscope's input, including matB.csv -- genes x cell types of marker flags,
#     whose column names are real cell types, so it reads as a plausible proportions matrix
#   - reference_signatures: cell2location reference output (intermediate)
#   - standardized: written by output_standardizer; re-picking it feeds a stale or malformed
#     derived CSV into the next standardizer pass
STAGED_INPUT_DIRS = (
    "converted_sc",
    "converted_spatial",
    "sc_csv",
    "spatial_csv",
    "csv_spatial",
    "csv_reference",
    "csv_filtered",
    "csv_export",
    "csv_exports",
    "csv_input",
    "csv_inputs",
    "converted_csv",
    "_starfysh_input",
    "prep",
    "reference_signatures",
    "celloscope_data",
    "standardized",
)

#: The tail of a per-tool scratch directory. See :func:`_is_staged_input_path` for why this sits
#: beside the enumeration rather than in it, and for why the qualifier in front of the tail is what
#: is required rather than a leading underscore.
_STAGING_DIR_SUFFIXES = ("_staging", "_input", "_inputs", "_input_csvs", "_converted")

#: The head of a per-tool CSV conversion directory, for the scratch names written the other way
#: round. See :func:`_is_staged_input_path` for why a prefix needs a guard a tail does not.
_STAGING_DIR_PREFIXES = ("csv_",)


def _is_staged_input_path(path: Path | str, root: Path | str | None = None) -> bool:
    """Whether `path` lies inside a staging or derived-artifact directory.

    Pass ``root`` to confine the component scan to the components *below* it. Every marker here
    describes a scratch directory a worker creates *inside* a run's tree -- nothing about it is
    evidence against the tree itself, and without the confinement the name of the directory the
    caller explicitly handed over (or of any ancestor on the absolute path) vetoes every file
    beneath it. That is not hypothetical: a live agent, told to convert a raw 10x .h5 and re-run
    clustering, naturally wrote the second attempt into ``scanpy_on_converted/`` -- which ends in
    the ``_converted`` staging tail, so post-analysis dropped the run's own ``*.h5ad`` and domains
    CSV as "staged inputs" and published no deliverable at all. The benchmark inspector's own call
    sites pass no ``root`` on purpose: their behaviour against the recorded trees is measured, and
    the paths they scan are tool-controlled rather than caller-named.

    Matching is per path *component*, and both halves of that are load-bearing on real data. A bare
    equality test would miss `standardized.bak_20260501_231156_visium_dup/`, a recorded directory
    name holding a stale standardized CSV, so a component that starts with a marker and continues
    with a separator counts too. Conversely the previous rule searched the whole path string for each
    marker, which let the shortest entry (`prep`) reject any output whose path merely mentioned it --
    `preprocessed/` among them -- and over-rejection silently marks a finished run incomplete.

    `STAGED_INPUT_DIRS` is an enumeration of names, so it only ever knows the staging directories
    someone has already written down. `_starfysh_input` is on it; `_xfuse_staging` is not, and the
    recorded xfuse run therefore offered its staged headerless Visium `tissue_positions_list.csv`
    as a result -- read with one spot's pixel coordinates as the column names, and published as
    "column means". `_STAGING_DIR_SUFFIXES` adds the shape those names share instead of the next
    name, so the next worker's scratch directory needs no edit here.

    The shape is a *qualifier* in front of one of the staging tails -- `dstg_input`, `_xfuse_staging`
    -- and the qualifier is the whole of it. A bare `input/` or `staging/` is untouched, because a
    tool told to write into a directory of that name is writing results there and over-rejection
    silently marks a finished run incomplete.

    This used to require a *leading underscore* as well, on the stated grounds that "only the `_`
    form is scratch". The recorded data says otherwise: five staging directories are written without
    one -- `spiral_input_csvs`, `dstg_input`, `bulk2space_csv_input`, `matrix_input` (105 MB of
    staged MERFISH counts) and `pipeline_precast_input` -- and none is on the enumeration either, so
    everything under them was offered as tool output. The recorded SPIRAL run, which wrote nothing at
    all, had all eight of its files read as results: it was published as a `deconvolution` because
    `slice_0_features.csv` -- its own 200 x 500 input feature matrix, transposed on the way in --
    "is a non-negative spot x cell-type matrix", it named that input as its `source_outputs`, and it
    reported 8 "Files written by the tool" against a true figure of zero.

    Requiring the qualifier rather than the underscore is what keeps the widening safe, and it was
    measured rather than assumed: across `smoke_outputs/` and `benchmarks/results/` it newly rejects
    exactly those five directories and no other component, no file name anywhere ends in a staging
    tail, and `_archive`, `_logs`, `_plots`, `_rerun_via_manual`, eight `_tune_res_*`,
    `cell2location_map`, `celloscope_results` and `assigned_expression` are all still output.

    `_STAGING_DIR_PREFIXES` is the same move for the names written head-first. The enumeration
    carries seven `csv_*` directories; five recorded components share their shape and are on it
    nowhere -- `csv_sc`, `csv_sc.bak_1777746574_old_naming`, `csv_for_stdeconvolve`,
    `csv_conversion` and `csv_from_h5ad`. `csv_sc/` is where a deconvolution tool's single-cell
    reference is staged, and the recorded spacexr Visium run was typed `spatial_clustering` off its
    `celltypes.csv` -- "one low-cardinality label per spot", that being the reference's own cell-type
    column -- while `spacexr_weights.csv`, the spot x cell-type matrix the tool wrote, sat unread
    beside it. A bare `csv/` is untouched for the reason a bare `input/` is: `run_bass` and
    `precast_spatial_clustering` write their results into one.

    The prefix clause skips a final component that carries a suffix, and the tail clause needs no
    such guard, because an extension lives at the end: `foo_inputs.csv` does not end in `_inputs`,
    but `csv_export.csv` does start with `csv_`, and rejecting an output file by its own name is the
    over-rejection this function exists to avoid. Measured the same way as the tail rule: across
    `smoke_outputs/` and `benchmarks/results/` the clause newly rejects exactly those five
    components, no file anywhere is named `csv_*`, and `inspect_tool_output`'s pick and status are
    unchanged on every run that has one of these directories.
    """
    p = Path(path)
    parts = p.parts
    if root is not None:
        try:
            parts = p.relative_to(Path(root)).parts
        except ValueError:
            # Not under root (or mixed absolute/relative): fall back to the full path, which is
            # the stricter reading -- over-rejection is preferable to trusting an unrelated tree.
            pass
    for i, part in enumerate(parts):
        for marker in STAGED_INPUT_DIRS:
            if part == marker or (part.startswith(marker) and part[len(marker) : len(marker) + 1] in (".", "_", "-")):
                return True
        if part.endswith(_STAGING_DIR_SUFFIXES):
            return True
        if i == len(parts) - 1 and Path(part).suffix:
            continue
        if any(part.startswith(prefix) and len(part) > len(prefix) for prefix in _STAGING_DIR_PREFIXES):
            return True
    return False


@dataclass
class OutputInspectionResult:
    """Result of inspecting a tool's output directory."""

    tool_name: str
    task_type: str
    output_dir: str
    status: str = "unknown"  # "ready", "incomplete", "invalid", "empty"

    # File inventory
    files_found: list[str] = field(default_factory=list)
    h5ad_files: list[str] = field(default_factory=list)
    csv_files: list[str] = field(default_factory=list)
    other_files: list[str] = field(default_factory=list)

    # Prediction info
    prediction_file: str | None = None
    prediction_format: str = ""  # h5ad, csv, etc.
    prediction_key: str = ""  # obs column or CSV column
    prediction_type: str = ""  # cluster_labels, gene_list_with_pvalues, proportions_matrix
    n_predictions: int = 0

    # For clustering: extracted labels
    cluster_labels: list[str] | None = None
    n_clusters: int = 0

    # For SVG: extracted gene list
    svg_genes: list[str] | None = None
    n_svg_genes: int = 0

    # For deconvolution: extracted proportions info
    proportions_shape: tuple[int, int] | None = None
    cell_types: list[str] | None = None

    # Validation
    conversion_needed: bool = False
    conversion_notes: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Completeness check results (populated by validate_output_completeness)
    completeness_check: dict[str, Any] | None = None

    def is_ready(self) -> bool:
        return self.status == "ready" and len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "task_type": self.task_type,
            "status": self.status,
            "prediction_file": self.prediction_file,
            "prediction_format": self.prediction_format,
            "prediction_key": self.prediction_key,
            "prediction_type": self.prediction_type,
            "n_predictions": self.n_predictions,
            "n_clusters": self.n_clusters,
            "n_svg_genes": self.n_svg_genes,
            # The list itself, not only its length: benchmark_runner reads ``svg_genes`` from this dict
            # (the post-execution gate stores it), always got None, and fell back to its own file scan,
            # whose list -- not this staging-aware, FDR-first one -- fed the pre-evaluation gate and the
            # standardizer's gene-list fallback (hunt 2026-09-30, u31-benchmarking-8).
            "svg_genes": self.svg_genes,
            "proportions_shape": self.proportions_shape,
            "cell_types": self.cell_types,
            "files_found": self.files_found,
            "conversion_needed": self.conversion_needed,
            "completeness_check": self.completeness_check,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def scan_for_misplaced_outputs(
    expected_output_dir: str,
    search_dirs: list[str] | None = None,
    task_type: str = "",
) -> list[dict[str, str]]:
    """Scan for output files that may have been written outside the expected directory.

    Common failure: tools write to CWD or a default directory instead of output_dir.

    Returns list of dicts with 'path', 'type', and 'reason' for each candidate.
    """
    expected = Path(expected_output_dir).resolve()
    candidates: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    # Build the list of directories to scan
    dirs_to_scan: list[Path] = []
    if search_dirs:
        dirs_to_scan.extend(Path(d).resolve() for d in search_dirs if Path(d).resolve().is_dir())

    # Always check the parent of expected_output_dir
    parent = expected.parent
    if parent.is_dir():
        dirs_to_scan.append(parent)

    # Always check current working directory
    cwd = Path(os.getcwd()).resolve()
    if cwd.is_dir():
        dirs_to_scan.append(cwd)

    # Deduplicate scan dirs
    unique_dirs: list[Path] = []
    seen_dirs: set[str] = set()
    for d in dirs_to_scan:
        ds = str(d)
        if ds not in seen_dirs:
            unique_dirs.append(d)
            seen_dirs.add(ds)

    # Output file extensions that tools commonly produce
    output_extensions = {".h5ad", ".csv", ".tsv"}

    # Strings that indicate a file is an input, not an output
    input_indicators = (
        "input",
        "original",
        "raw",
        "reference",
        "ground_truth",
        "prepared",
        "converted",
        "download",
        "source",
        "template",
    )

    # Threshold: files modified in the last 30 minutes
    cutoff = time.time() - 30 * 60

    for scan_dir in unique_dirs:
        try:
            for entry in scan_dir.iterdir():
                if not entry.is_file():
                    continue
                if entry.suffix not in output_extensions:
                    continue

                resolved = entry.resolve()
                resolved_str = str(resolved)

                # Skip files that are inside the expected output directory
                try:
                    resolved.relative_to(expected)
                    continue
                except ValueError:
                    pass

                # Skip already seen
                if resolved_str in seen_paths:
                    continue
                seen_paths.add(resolved_str)

                # Skip files that look like inputs
                stem_lower = entry.stem.lower()
                if any(ind in stem_lower for ind in input_indicators):
                    continue

                # Check modification time
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                if mtime < cutoff:
                    continue

                # Skip empty files
                try:
                    if entry.stat().st_size == 0:
                        continue
                except OSError:
                    continue

                # Build reason string
                if scan_dir == parent:
                    reason = f"Found in parent directory of expected output_dir ({scan_dir})"
                elif scan_dir == cwd:
                    reason = f"Found in current working directory ({scan_dir})"
                else:
                    reason = f"Found in search directory ({scan_dir})"

                candidates.append(
                    {
                        "path": resolved_str,
                        "type": entry.suffix.lstrip("."),
                        "reason": reason,
                    }
                )
        except PermissionError:
            continue

    return candidates


def validate_output_completeness(
    prediction_file: str,
    task_type: str,
    prediction_key: str = "",
    unassigned_labels: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Deep validation of prediction output completeness.

    Checks beyond basic existence:
    - Not all NaN
    - Not all identical values
    - Reasonable number of predictions
    - No truncation indicators
    - File is not a partial write (size > expected minimum)
    """
    result: dict[str, Any] = {
        "valid": True,
        "issues": [],
        "checks_performed": [],
    }

    pred_path = Path(prediction_file)
    if not pred_path.exists():
        result["valid"] = False
        result["issues"].append(f"Prediction file does not exist: {prediction_file}")
        return result

    # Minimum file size check (catch partial writes)
    file_size = pred_path.stat().st_size
    result["file_size_bytes"] = file_size

    # Each validator claims its own check, once it has established it can read the file. Claiming
    # here instead named checks that never examined anything -- see _claim_check.
    if task_type == "spatial_clustering":
        _validate_clustering_completeness(result, prediction_file, prediction_key, unassigned_labels)
    elif task_type == "svg_detection":
        _validate_svg_completeness(result, prediction_file, prediction_key)
    elif task_type == "deconvolution":
        _validate_deconvolution_completeness(result, prediction_file, prediction_key)
    else:
        result["checks_performed"].append("generic_size_check")
        if file_size < 100:
            result["valid"] = False
            result["issues"].append(f"File suspiciously small ({file_size} bytes)")

    return result


def _validate_clustering_completeness(
    result: dict[str, Any],
    prediction_file: str,
    prediction_key: str,
    unassigned_labels: Iterable[str] | None = None,
) -> None:
    """Validate clustering predictions for completeness."""
    pred_path = Path(prediction_file)
    if not _claim_check(result, pred_path, "clustering_completeness"):
        return

    try:
        if pred_path.suffix == ".h5ad":
            import anndata as ad

            adata = ad.read_h5ad(prediction_file)
            if prediction_key not in adata.obs.columns:
                result["valid"] = False
                result["issues"].append(f"Key '{prediction_key}' not found in obs columns")
                return
            series = adata.obs[prediction_key]
        else:  # a delimited text table -- _claim_check admitted nothing else
            import pandas as pd

            sep = _tabular_sep(pred_path)
            df = pd.read_csv(prediction_file, sep=sep)
            if prediction_key not in df.columns:
                # One value per line, no header — treat the only column as the labels
                if df.shape[1] == 1:
                    series = df.iloc[:, 0]
                else:
                    result["valid"] = False
                    result["issues"].append(f"Key '{prediction_key}' not found in CSV columns")
                    return
            else:
                series = df[prediction_key]

        non_nan = series.dropna()
        # A label the profile declares unassigned (mist's 'isolated') is not a cluster here either: this
        # check reads the file on its own, kept counting it, and passed one region plus isolated spots
        # as two clusters (hunt 2026-09-30, u31-benchmarking-11 review).
        unassigned = {str(v) for v in (unassigned_labels or ())}
        if unassigned:
            non_nan = non_nan[~non_nan.astype(str).isin(unassigned)]
        result["total_values"] = len(series)
        result["non_nan_values"] = len(non_nan)
        result["unique_values"] = int(non_nan.nunique())

        # Check: not all NaN
        if len(non_nan) == 0:
            result["valid"] = False
            result["issues"].append("All values are NaN")
            return

        # Check: enough non-NaN values
        if len(non_nan) <= 10:
            result["valid"] = False
            result["issues"].append(f"Only {len(non_nan)} non-NaN predictions (need >10)")

        # Check: more than 1 unique label
        if non_nan.nunique() <= 1:
            result["valid"] = False
            result["issues"].append(
                f"Only {non_nan.nunique()} unique cluster label(s) found; clustering should produce >1 cluster"
            )

        # Check: not all the same label
        if non_nan.nunique() == 1:
            result["issues"].append(f"All predictions are '{non_nan.iloc[0]}' — likely a failed run")

    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"Error reading prediction file: {e}")


def _validate_svg_completeness(
    result: dict[str, Any],
    prediction_file: str,
    prediction_key: str,
) -> None:
    """Validate SVG detection predictions for completeness."""
    pred_path = Path(prediction_file)

    # Valid gene identifier pattern: starts with a letter, contains only
    # alphanumeric characters, hyphens, dots, or underscores.
    gene_pattern = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")
    if not _claim_check(result, pred_path, "svg_completeness"):
        return

    try:
        if _is_tabular_prediction(pred_path):
            import pandas as pd

            sep = _tabular_sep(pred_path)
            df = pd.read_csv(prediction_file, sep=sep)
            if prediction_key not in df.columns:
                # Check if gene names are in the index (Hotspot, SVGbit, etc.)
                idx = df.index
                if idx.dtype == object or (hasattr(idx, "inferred_type") and idx.inferred_type == "string"):
                    sample = [str(x) for x in idx[:5]]
                    if any(not s.replace(".", "").replace("-", "").replace("_", "").isdigit() for s in sample):
                        df = df.reset_index()
                        if df.columns[0] == prediction_key or prediction_key not in df.columns:
                            # Use first column after reset as the gene column
                            prediction_key = df.columns[0]
                # Check for 'Unnamed: 0' column with gene names
                if prediction_key not in df.columns:
                    for col in df.columns:
                        if str(col).startswith("Unnamed"):
                            sample = [str(x) for x in df[col].dropna().head(5)]
                            if sample and any(
                                not s.replace(".", "").replace("-", "").replace("_", "").isdigit() for s in sample
                            ):
                                prediction_key = col
                                break
                if prediction_key not in df.columns:
                    result["valid"] = False
                    result["issues"].append(f"Key '{prediction_key}' not found in CSV columns")
                    return
            raw_genes = df[prediction_key].dropna().astype(str).tolist()
        else:  # .h5ad -- _claim_check admitted nothing else
            import anndata as ad

            adata = ad.read_h5ad(prediction_file)
            if prediction_key not in adata.var.columns:
                raw_genes = []
            elif adata.var[prediction_key].dtype == bool:
                raw_genes = list(adata.var_names[adata.var[prediction_key]])
            else:
                # `prediction_key` does not name the same thing in the two branches. For a CSV it is
                # the gene column, which is why `df[prediction_key]` above is right; for an h5ad it
                # is the *significance* column the reader settled on (_inspect_svg records its own
                # choice there). Reading its values as gene names read the scores -- "0.99",
                # "0.985" -- and then reported that none of them look like a gene, about names it
                # never opened. Since inspect_tool_output turns an invalid completeness check into
                # status "incomplete" and run_formal_benchmark refuses to score anything that is not
                # ready, that misreading withheld the whole run. The genes an h5ad prediction is
                # about are its var_names.
                raw_genes = [str(g) for g in adata.var_names]

        # Filter blanks and empty strings
        genes = [g.strip() for g in raw_genes if g.strip() and not g.startswith("Blank")]
        result["total_genes_raw"] = len(raw_genes)
        result["genes_after_filtering"] = len(genes)

        # Check: at least some genes
        if len(genes) == 0:
            result["valid"] = False
            result["issues"].append("No genes remaining after filtering blanks and empty strings")
            return

        # Check: genes look like valid identifiers.
        # Both sides of the comparison must count the same population. Sampling the first
        # 100 genes but testing against half of *all* of them made the check unsatisfiable
        # for any list of 200+ (len(invalid_genes) cannot exceed 100), which is 5 of the 9
        # SVG outputs in the real smoke corpus -- long lists, the ones least likely to be
        # eyeballed. The message below already names the intended denominator.
        sampled = genes[:100]
        invalid_genes = [g for g in sampled if not gene_pattern.match(g)]
        result["invalid_gene_names_sample"] = invalid_genes[:10]
        if len(invalid_genes) > len(sampled) * 0.5:
            result["valid"] = False
            result["issues"].append(
                f"More than half of gene names ({len(invalid_genes)}/{len(sampled)} sampled) "
                "do not look like valid gene identifiers"
            )

    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"Error reading prediction file: {e}")


def _validate_deconvolution_completeness(
    result: dict[str, Any],
    prediction_file: str,
    prediction_key: str,
) -> None:
    """Validate deconvolution predictions for completeness."""
    pred_path = Path(prediction_file)
    if not _claim_check(result, pred_path, "deconvolution_completeness"):
        return

    try:
        if _is_tabular_prediction(pred_path):
            import pandas as pd

            sep = _tabular_sep(pred_path)
            # The spot IDs are not always in the first column -- R's write.csv puts an explicit
            # `spot` column LAST, and CARD is a shipped tool that does exactly that. Reading at 0
            # then succeeds one column off: the first cell type becomes the index and the barcode
            # column is dropped as non-numeric, so every count and row sum below is taken against
            # the matrix minus one real cell type. Measured on all four recorded CARD runs.
            # See _proportions_index_col, which the sibling readers below already use; it answers 0
            # whenever there is no such column to find, which is the common case.
            index_col = _proportions_index_col(pred_path, sep)
            df = pd.read_csv(prediction_file, sep=sep, index_col=index_col)
            # Spots on the rows before any per-spot check (hunt 2026-09-30, u31-benchmarking-9).
            from spatialomicsgym.benchmarking.output_standardizer import _orient_proportions

            numeric_df, _ = _orient_proportions(df.select_dtypes(include="number"), wide_only=True)
        else:  # .h5ad -- _claim_check admitted nothing else
            import anndata as ad
            import pandas as pd

            adata = ad.read_h5ad(prediction_file)
            if prediction_key and prediction_key in adata.obsm:
                mat = adata.obsm[prediction_key]
                numeric_df = pd.DataFrame(mat)
            else:
                result["valid"] = False
                result["issues"].append(f"obsm key '{prediction_key}' not found in h5ad")
                return

        n_spots, n_cell_types = numeric_df.shape
        result["n_spots"] = n_spots
        result["n_cell_types"] = n_cell_types

        # Check: more than 1 cell type
        if n_cell_types <= 1:
            result["valid"] = False
            result["issues"].append(
                f"Only {n_cell_types} cell type column(s) found; deconvolution should produce >1 cell type"
            )

        # Check: no all-NaN rows
        all_nan_rows = int(numeric_df.isna().all(axis=1).sum())
        result["all_nan_rows"] = all_nan_rows
        if all_nan_rows > 0:
            nan_frac = all_nan_rows / max(n_spots, 1)
            if nan_frac > 0.5:
                result["valid"] = False
                result["issues"].append(f"{all_nan_rows}/{n_spots} rows are entirely NaN ({nan_frac:.0%})")
            elif all_nan_rows > 0:
                result["issues"].append(f"{all_nan_rows} rows are entirely NaN")

        # Check: the matrix actually says something. A prediction in which every value is
        # identical carries no information about any spot, and no rescaling can recover
        # any: all-zero is the case seen in practice (STdGCN published a (200, 5) table of
        # zeros and it was scored), all-0.3 is the same emptiness sitting inside the
        # row-sum band below. This is the "Not all identical values" check
        # validate_output_completeness already documents.
        import numpy as np

        values = numeric_df.to_numpy(dtype="float64", na_value=np.nan)
        finite = values[np.isfinite(values)]
        if finite.size and float(finite.min()) == float(finite.max()):
            result["valid"] = False
            result["issues"].append(
                f"All {finite.size} values are identical ({float(finite.min()):.4g}); the prediction carries no signal"
            )

        # Check: row sums are roughly 1 (proportions should sum to ~1)
        row_sums = numeric_df.dropna(how="all").sum(axis=1)
        if len(row_sums) > 0:
            mean_sum = float(row_sums.mean())
            result["mean_row_sum"] = round(mean_sum, 4)
            # Allow sums between 0.5 and 1.5 as "roughly 1"
            if not (0.5 <= mean_sum <= 1.5):
                result["issues"].append(
                    f"Mean row sum is {mean_sum:.4f} (expected ~1.0 for proportions); values may not be normalized"
                )

    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"Error reading prediction file: {e}")


def _inspect_with_fallback(
    result: OutputInspectionResult,
    out_path: Path,
    all_files: list[Path],
    profile,
    task_type: str,
    ground_truth_path: str | None = None,
) -> None:
    """Try tool-specific profile first, then generic detection, with better error messages.

    This wrapper routes to the task-specific inspector and, if the profile-based
    detection fails, retries with a None profile so generic heuristics are used.
    """
    inspector_map = {
        "spatial_clustering": _inspect_clustering,
        "svg_detection": _inspect_svg,
        "deconvolution": _inspect_deconvolution,
    }

    inspector = inspector_map.get(task_type)
    if inspector is None:
        result.status = "invalid"
        result.errors.append(f"Unknown task type: {task_type}")
        return
    if task_type == "spatial_clustering" and ground_truth_path:
        # The input / ground-truth h5ad, so a label column the dataset shipped with is not read as the
        # tool's (hunt 2026-09-30, u31-benchmarking-1).
        import functools

        inspector = functools.partial(_inspect_clustering, reference_path=ground_truth_path)

    # Withhold files the profile itself declares under a non-prediction role, from BOTH
    # passes. The generic pass deliberately drops the profile so unmatched patterns can
    # still be recovered by heuristics — but dropping it also discarded what the profile
    # said files are *not*, and the candidate scanner then promoted them. STdeconvolve:
    # the declared prediction (theta) is correctly rejected by the archetype guard because
    # its topics are unnamed integers, the generic pass then picked stdeconvolve_beta.csv
    # — declared role="metadata", a topic x gene loading table — and reported it as a
    # (3, 34) proportions matrix with gene symbols as cell types, on all six deconvolution
    # datasets. Positive knowledge is fallible (patterns drift); negative knowledge is not.
    # When that leaves nothing, "no prediction here" is the truthful verdict — restoring
    # the disclaimed files as a safety net is precisely the behaviour being fixed.
    if profile is not None:
        all_files = [f for f in all_files if not _is_declared_non_prediction(f, profile)]

    def _the_tool_answered_nothing() -> bool:
        # (hunt 2026-09-30, u31-benchmarking-12) ``empty`` with a ``prediction_file`` is not a miss:
        # the reader found the tool's own table and a real threshold selected nothing from it, and it
        # says "stop looking". Treating it as a miss filed a false "did not match any output", ran
        # the generic pass -- which takes CSVs newest first, so an undeclared side table with a gene
        # column and no p-value came back ``ready`` with every gene, recall 1.0 over the tool's honest
        # "no SVGs" -- and, failing that, sent the gate off to adopt another directory's file.
        if result.status != "empty" or not result.prediction_file:
            return False
        name = Path(result.prediction_file).name
        if _is_declared_prediction(Path(result.prediction_file), profile):
            result.errors.append(
                f"{name} was read and selects no {task_type} prediction: the tool's answer is that "
                "nothing passed, not that no output was found (see warnings)"
            )
        else:
            # A table the profile does not declare is not the tool's word, so it is not called its
            # answer -- and post_execution_gate, asking the same question, still searches for the
            # declared file elsewhere (hunt 2026-09-30, u31-benchmarking-12 review).
            whose = (
                "this tool has no registry profile to say whether it is the tool's prediction"
                if profile is None
                else "it is not a file this tool's profile declares as its prediction"
            )
            result.errors.append(f"{name} was read and selects no {task_type} prediction (see warnings), but {whose}")
        return True

    # First pass: use the tool-specific profile
    if profile is not None:
        inspector(result, out_path, all_files, profile)
        if result.status == "ready":
            return
        if _the_tool_answered_nothing():
            return
        # Profile-based detection did not succeed; record what we tried
        profile_name = getattr(profile, "tool_name", str(profile))
        result.warnings.append(
            f"Tool profile '{profile_name}' did not match any output; falling back to generic detection"
        )
        # Preserve first-pass errors as diagnostics, then reset for second pass
        if result.errors:
            result.warnings.extend(f"Profile pass: {e}" for e in result.errors)
        result.status = "unknown"
        result.errors.clear()

    # Second pass: generic detection (no profile)
    inspector(result, out_path, all_files, None)

    if result.status != "ready" and not _the_tool_answered_nothing():
        result.errors.append(
            f"Neither profile-based nor generic detection found valid {task_type} output. "
            f"Files present: {[f.name for f in all_files[:10]]}"
        )


def _drop_our_own_report(result: OutputInspectionResult, out_path: Path, all_files: list[Path]) -> list[Path]:
    """Remove this system's own post-analysis output from the candidate set.

    The post-analysis engine writes its report to ``<tool output dir>/post_analysis/`` by default --
    inside the directory this function is pointed at. Its ``tables/`` are derived summaries whose
    column names read as data, and the inspector accepted them: ``mean_composition.csv`` is a 5 x 3
    table over ``mean_proportion`` / ``max_proportion`` / ``spots_above_10pct``, returned as a
    proportions matrix whose *cell types* were those three strings. Measured over this repo's own
    recorded outputs, 36 of the 92 run directories holding such a report had the prediction picked
    from under ``post_analysis/`` -- deconvolution runs at ``mean_composition.csv``, SVG runs at
    ``gene_ranking.csv``, clustering runs at ``domain_sizes.csv``.

    The rule for what is ours is not re-derived here. It lives in
    :mod:`spatialomicsgym.generated_report`, which belongs to neither package precisely so that this
    one can consult it: a directory is ours when it holds a report manifest we wrote, decided by
    reading the file. That is what makes a custom results directory caught and a tool's own
    ``manifest.json`` left alone.

    The skip is stated in ``warnings`` rather than made silently. A directory that plainly holds
    twelve files must not inspect as holding four with no account of the difference -- and "there is
    a previous report in your output directory" is worth an operator knowing either way.

    One layout cannot be dropped by subtree: the results directory is a caller-supplied argument, so
    our report may be written *into* the directory being inspected rather than beneath it.
    "Everything under our report" is then the whole run, and removing it takes the tool's prediction
    with it -- the run inspects as ``empty``, which is a louder wrong answer than the one being
    fixed. There, and only there, the manifest's own record of what it wrote is what is ours;
    :func:`~spatialomicsgym.generated_report.split_report_files` handles both layouts.
    """
    try:
        from spatialomicsgym.generated_report import split_report_files
    except Exception:  # pragma: no cover - ships with this package
        return all_files

    kept, ours = split_report_files(out_path, all_files)
    if not ours:
        return all_files

    names = ", ".join(sorted({f.parent.name for f in ours if f.name == "manifest.json"}))
    result.warnings.append(
        f"ignored {len(ours)} file(s) under this system's own post-analysis report ({names}): "
        "those are derived summaries of the tool's output, never the tool's prediction"
    )
    return kept


def inspect_tool_output(
    tool_name: str,
    task_type: str,
    output_dir: str,
    ground_truth_path: str | None = None,
) -> OutputInspectionResult:
    """Inspect a tool's output directory and extract prediction information.

    This is the MANDATORY first step in benchmarking. It must be called
    before any evaluation to verify outputs are valid and identify the
    correct prediction files.
    """
    result = OutputInspectionResult(tool_name=tool_name, task_type=task_type, output_dir=output_dir)

    out_path = Path(output_dir)
    if not out_path.exists():
        result.status = "empty"
        result.errors.append(f"Output directory does not exist: {output_dir}")
        return result

    # Inventory all files
    all_files = list(out_path.rglob("*"))
    all_files = [f for f in all_files if f.is_file() and f.stat().st_size > 0]
    all_files = _drop_our_own_report(result, out_path, all_files)
    result.files_found = [str(f.relative_to(out_path)) for f in all_files]
    result.h5ad_files = [str(f) for f in all_files if f.suffix == ".h5ad"]
    result.csv_files = [str(f) for f in all_files if f.suffix in (".csv", ".tsv")]
    result.other_files = [str(f) for f in all_files if f.suffix not in (".h5ad", ".csv", ".tsv")]

    if not all_files:
        result.status = "empty"
        result.errors.append("No output files found")
        return result

    # Get tool-specific profile
    from spatialomicsgym.benchmarking.tool_output_registry import get_profile

    profile = get_profile(tool_name)

    # Route to task-specific inspector with fallback
    _inspect_with_fallback(result, out_path, all_files, profile, task_type, ground_truth_path=ground_truth_path)

    # Whatever route picked the prediction, ask whether it predates this run. The check used to sit on
    # the deconvolution CSV route alone, while the pre-run cleanup leaves SVG tables, clustering files
    # and nested h5ads in place -- a rerun whose worker died was scored on the previous run's file
    # with no warning (hunt 2026-09-30, u31-benchmarking-7). This system's own derived files are not
    # siblings: a later re-standardization is not evidence that the tool's file is old.
    if result.prediction_file:
        siblings = [f for f in all_files if not _is_our_derived_artifact(f, out_path)]
        stale_w = _stale_file_warning(Path(result.prediction_file), siblings)
        if stale_w:
            result.warnings.append(stale_w)

    # Run completeness validation when a prediction file was found
    if result.prediction_file and result.status == "ready":
        completeness = validate_output_completeness(
            prediction_file=result.prediction_file,
            task_type=task_type,
            prediction_key=result.prediction_key,
            unassigned_labels=_unassigned_labels(profile),
        )
        result.completeness_check = completeness
        if not completeness["valid"]:
            result.status = "incomplete"
            for issue in completeness["issues"]:
                result.errors.append(f"Completeness: {issue}")
        else:
            # Issues recorded without failing the check used to stop here, reachable only
            # by digging into completeness_check. They are the reason the checks exist:
            # cell2location and SpaCET report absolute abundances (mean row sums 4.24 and
            # 7.86), which the standardizer rescales -- worth saying, not worth failing.
            for issue in completeness["issues"]:
                result.warnings.append(f"Completeness: {issue}")

    return result


def _match_prediction_files(files: list[Path], pattern):
    """`_match_files`, minus anything under a staged-input directory.

    The profile path is consulted *before* the generic keyword scanners and its matches are moved to
    the front of the candidate list, so a pattern loose enough to match a staged file would win over
    the tool's real output. No pattern in ``TOOL_PROFILES`` is that loose today, and
    ``register_dynamic_profile`` -- the hook through which a user tool's patterns would arrive
    unaudited -- has no caller yet; the filter is there for when it does. ``_match_files`` itself is
    left permissive: it also validates explicitly supplied paths, where the caller has already chosen
    the file.
    """
    return [f for f in _match_files(files, pattern) if not _is_staged_input_path(f)]


def _match_files(files: list[Path], pattern):
    """Match files against a glob pattern OR a list of glob alternatives.

    pattern may be:
      - a single glob string (e.g. ``"spotlight_proportions.csv"``)
      - a list of glob strings (e.g. ``["spotlight_proportions.csv", "*spotlight*props*.csv"]``)

    When given a list, files matching ANY pattern are returned, with earlier
    patterns prioritised in the result order. Used by the deconvolution /
    clustering / SVG inspectors so a worker that has shipped multiple
    historical filename variants (or whose filename depends on params) is
    still picked up by profile-based detection instead of falling through
    to the generic keyword scanner.
    """
    if isinstance(pattern, (list, tuple)):
        matched: list[Path] = []
        seen: set[Path] = set()
        for p in pattern:
            for f in _match_files(files, p):
                if f not in seen:
                    matched.append(f)
                    seen.add(f)
        return matched

    matched = []
    for f in files:
        if fnmatch.fnmatch(f.name, pattern):
            matched.append(f)
    # Also try matching against relative path
    if not matched:
        for f in files:
            if fnmatch.fnmatch(str(f), f"*{pattern}"):
                matched.append(f)
    return matched


def _is_declared_non_prediction(path: Path, profile) -> bool:
    """True when the profile declares ``path`` under a role other than ``prediction``.

    A profile listing ``OutputSpec("x_beta.csv", role="metadata")`` is asserting that the
    file is not a prediction. Generic candidate scanning must honour that, otherwise a
    guard that rejects the real prediction silently promotes the metadata file in its
    place (STdeconvolve: a topic x gene loading table scored as spot proportions, with
    gene symbols reported as cell types).

    A file matched by *any* prediction spec stays eligible even if another spec also names
    it — declaring it a prediction anywhere wins.

    Matched with ``PurePath.match``, i.e. the way the registry writes the declaration: anchored
    at the right, ``/`` a component boundary, ``*`` never crossing one. ``_match_files`` is not
    usable here — its second pass is ``fnmatch(str(f), "*" + pattern)`` against the ABSOLUTE
    path, where ``*`` does cross ``/``, so ``*rctd*weights*.csv`` (spacexr, role=metadata) was
    satisfied by the run directory ``spacexr_rctd_deconvolution/`` and withheld
    ``spacexr_weights.csv`` there but not under ``_live_out/deconv/``, and a bare
    ``weights.csv`` declaration withheld ``rctd_weights.csv``. Negative knowledge decided by a
    parent directory the tool does not control is not knowledge. Positive (prediction) matching
    keeps ``_match_files``: its permissiveness is load-bearing there.
    """
    if profile is None:
        return False
    declared_other = False
    for spec in getattr(profile, "outputs", None) or []:
        patterns = spec.filename_pattern
        patterns = [patterns] if isinstance(patterns, str) else list(patterns or ())
        if not any(path.match(p) for p in patterns if p):
            continue
        if spec.role == "prediction":
            return False
        declared_other = True
    return declared_other


def _declared_patterns(path: Path, profile, attr: str) -> bool:
    """True when ``path`` matches a pattern in ``attr`` of one of the profile's *prediction* specs.

    Matched with ``PurePath.match``, as ``_is_declared_non_prediction`` matches, so a declaration is
    satisfied by the file and never by the name of a directory above it.
    """
    if profile is None:
        return False
    for spec in getattr(profile, "outputs", None) or []:
        if spec.role != "prediction":
            continue
        patterns = getattr(spec, attr, None)
        patterns = [patterns] if isinstance(patterns, str) else list(patterns or ())
        if any(path.match(p) for p in patterns if p):
            return True
    return False


def _is_declared_prediction(path: Path, profile) -> bool:
    """True when the profile names ``path`` as the tool's prediction (a prediction spec's filename_pattern)."""
    return _declared_patterns(path, profile, "filename_pattern")


def _is_declared_selection(path: Path, profile) -> bool:
    """True when the profile names ``path`` as the tool's own selection (``OutputSpec.selection_patterns``)."""
    return _declared_patterns(path, profile, "selection_patterns")


# Formats a profile may declare for a delimited text prediction table. json/h5ad are
# deliberately absent: those specs are handled by their own readers, not the CSV path.
_TABULAR_FORMATS = ("csv", "tsv", "txt", "table")


def _tabular_sep(path: Path) -> str:
    """Field separator to read a tabular prediction file with.

    ``.csv`` and ``.tsv`` keep their historical hard-coded separators, so files that were
    already being read are read identically. Any other extension can only reach a reader
    when a tool profile names it explicitly, and those conventions vary (SpotGF writes a
    tab-separated ``.txt``), so sniff the header rather than assuming commas.

    The sniff itself lives in ``utils.file_io`` and is shared with the diagnosis path, which
    trusts no extension because it feeds no recorded score. The two trusted here are passed
    explicitly for that reason.
    """
    from spatialomicsgym.utils.file_io import BENCHMARK_TRUSTED_EXTENSIONS, sniff_tabular_sep

    return sniff_tabular_sep(path, trust_extension=BENCHMARK_TRUSTED_EXTENSIONS)


def _is_tabular_prediction(path: Path) -> bool:
    """Does this path name a delimited text table?

    Judged on the uncompressed suffix, so ``proportions.csv.gz`` is a table and not an unknown
    format. The set is ``_TABULAR_FORMATS`` itself -- the formats a profile may declare for a
    prediction -- because a file the resolver is willing to turn into a prediction is a file the
    completeness guards have to be able to read.
    """
    from spatialomicsgym.utils.file_io import uncompressed_suffix

    return uncompressed_suffix(path).lstrip(".") in _TABULAR_FORMATS


def _claim_check(result: dict[str, Any], pred_path: Path, check: str) -> bool:
    """Record ``check`` as performed, but only if the validator below can read this file at all.

    ``checks_performed`` is the only record of which guards ran, and it used to be written before
    the dispatch, so a validator that returned at its ``else`` still reported its check as done.
    Two shipped tools write a ``.txt`` prediction -- SpotGF's ``SpotGF_scores.txt`` and STRIDE's
    ``*_spot_celltype_frac.txt``, both named as *authoritative* by their profiles -- and for those
    the whole battery (all-NaN, single-label, all-identical, row sums) was skipped under a green
    claim. An all-zero STRIDE table reached status ``ready``.
    """
    if pred_path.suffix == ".h5ad" or _is_tabular_prediction(pred_path):
        result["checks_performed"].append(check)
        return True
    result["issues"].append(
        f"No completeness reader for '{pred_path.suffix or pred_path.name}'; the prediction was not checked"
    )
    return False


# Column names a tool uses for the spot/cell identifier of a proportion matrix. Matched
# case-insensitively; the same four names ``output_standardizer`` searches for (`:730`).
_SPOT_ID_COLUMNS = ("spot", "barcode", "spot_id", "cell_id")


def _proportions_index_col(path: Path, sep: str):
    """Which column of a proportion matrix holds the spot IDs -- a name, or ``0`` for "the first".

    ``index_col=0`` is right for the Python convention and wrong for the R one. ``write.csv`` on a
    data frame carrying an explicit ``spot`` column puts the IDs LAST, and CARD is a shipped tool
    that does exactly that: the recorded ``card_proportions.csv`` has 45 fields with ``spot`` at the
    end. Reading it at ``index_col=0`` does not fail -- it succeeds one column off, naming the first
    cell type as the index and the barcode column as a cell type. The reported ``cell_types`` then
    both loses a real type and gains a fake one, and the deconvolution gate adds up what it was
    handed, comes up short by exactly the dropped column, and reports "proportions don't sum to 1"
    against a matrix whose rows sum to 1.

    Only the header is read, so a file whose header is one field short of its rows -- R's
    ``write.csv(row.names=TRUE)``, which pandas already resolves by making column 0 the index --
    matches nothing here and keeps that resolution. ``output_standardizer`` searches for the same
    names inline before its own fallback (``:728-738``), which is why it reads that same file
    correctly today; this is the inspector agreeing with it.
    """
    import pandas as pd

    try:
        header = pd.read_csv(path, sep=sep, nrows=0).columns
    except Exception:
        return 0
    lowered = {str(c).strip().lower(): c for c in header}
    for name in _SPOT_ID_COLUMNS:
        if name in lowered:
            return lowered[name]
    return 0


def _is_self_relation_table(df) -> bool:
    """True when both axes name the same entities, so the table relates cell types to each other.

    A colocalization or correlation matrix over cell types passes every other test the deconvolution
    candidate loop applies -- two or more numeric columns, real cell-type names, no archetype
    indices -- and two recorded runs were scored on one. spacexr's Slide-seqV2 RCA puck was published
    as ``(7, 7)`` for 12,645 beads, with 1.0 down the diagonal and -0.4287 off it; destvi's Visium
    section as ``(44, 44)``. In both, the run's real prediction was unavailable -- withheld by the
    ``role="metadata"`` rule for spacexr, present only as a truncated ``.bak`` for destvi -- so there
    was nothing to find and the scanner manufactured something.

    The columns cannot tell these apart: ``spatialscope_proportions.csv`` carries the identical 44
    cell-type column names as destvi's correlation matrix. Nor can the values -- the obvious guard,
    rejecting negatives, would have discarded three correct SpatialScope predictions whose genuine
    output carries solver noise down to -0.0010, to fix two wrong ones. The row labels separate them
    categorically over every deconvolution run of both corpora, with no threshold to choose: a spot
    barcode is never a cell-type name.

    Containment, not set equality, because the loop decides on ``nrows=5`` and a 44 x 44 shows only
    five of its row labels there. Two rows are required: one label appearing among the columns is a
    coincidence, and a single-row frame relates nothing to anything.
    """
    rows = [str(i) for i in df.index]
    if len(rows) < 2:
        return False
    columns = {str(c) for c in df.columns}
    return all(row in columns for row in rows)


def _stale_file_warning(picked: Path, all_files: list[Path], grace_seconds: int = 3600) -> str | None:
    """Detect when ``picked`` is significantly older than the youngest sibling.

    A common contamination mode: a worker fails this run but a prior run left a
    same-named prediction file behind. The inspector silently picks the stale
    file and the standardizer + evaluator process garbage data.

    Returns a human-readable warning string when the picked file's mtime trails
    the most recent sibling by more than ``grace_seconds``; ``None`` otherwise.

    The check is best-effort: missing files / broken stat() returns None.
    """
    try:
        picked_m = picked.stat().st_mtime
        siblings = [f for f in all_files if f != picked]
        if not siblings:
            return None
        youngest = max(f.stat().st_mtime for f in siblings)
        gap = youngest - picked_m
        if gap > grace_seconds:
            from datetime import datetime

            picked_iso = datetime.fromtimestamp(picked_m).isoformat(timespec="seconds")
            youngest_iso = datetime.fromtimestamp(youngest).isoformat(timespec="seconds")
            hours = gap / 3600.0
            return (
                f"Picked prediction '{picked.name}' has mtime {picked_iso}, "
                f"but the freshest sibling in the same directory has mtime "
                f"{youngest_iso} ({hours:.1f}h newer). This often means the "
                f"current run did not produce a prediction file and a stale "
                f"file from a prior run is being used silently."
            )
    except Exception:
        return None
    return None


def _is_our_derived_artifact(path: Path, root: Path) -> bool:
    """A file this system writes into a run directory after inspecting it: ``standardized/``, ``metrics.json``."""
    try:
        parts = Path(path).relative_to(root).parts
    except ValueError:
        parts = Path(path).parts
    return Path(path).name == "metrics.json" or any(
        part == "standardized" or part.startswith(("standardized.", "standardized_")) for part in parts[:-1]
    )


def _resolve_prediction_key(container, key) -> str | None:
    """Resolve a prediction_key against a container that supports `in` and `[]`.

    `container` may be:
      - an AnnData (treated as `adata.obsm` for deconvolution; pass `adata.obs`
        / `adata.obsm` / `adata.var` / `adata.uns` directly for other lookups)
      - a Mapping / dict-like
      - a pandas DataFrame (uses .columns)

    `key` may be a single string or a list of candidate strings. The first
    candidate that resolves is returned; otherwise None. List ordering is the
    priority: more-specific keys first, fallbacks last.
    """
    candidates: list[str]
    if isinstance(key, (list, tuple)):
        candidates = [str(k) for k in key if k]
    elif key:
        candidates = [str(key)]
    else:
        return None

    # Build a "names" view: for AnnData-like, prefer obsm keys; for DataFrame, columns
    if hasattr(container, "columns"):
        names = list(container.columns)
    elif hasattr(container, "keys"):
        try:
            names = list(container.keys())
        except Exception:
            names = []
    else:
        names = []

    for cand in candidates:
        if cand in names:
            return cand

    # Case-insensitive fallback
    name_lower = {str(n).lower(): n for n in names}
    for cand in candidates:
        if cand.lower() in name_lower:
            return name_lower[cand.lower()]
    return None


# Words that mark an obsm entry as a per-spot abundance matrix, for an h5ad whose key nobody named.
_ABUNDANCE_OBSM_KEYWORDS = ("abundance", "proportion", "deconv", "cell_type", "q05")


def _abundance_obsm_key(obsm, prediction_key=None) -> str | None:
    """Which ``obsm`` entry of a deconvolution h5ad holds the per-spot abundances.

    The key the caller already settled on wins -- the inspector records it, the benchmark runner
    forwards it. The standardizer and the pre-evaluation gate used to ignore it and take the first
    keyword hit in ``obsm`` order, which ``read_h5ad`` returns alphabetically: cell2location exports
    ``means_``, ``q05_``, ``q95_`` and ``stds_cell_abundance_w_sf``, so the inspector scored q05 and
    the standardizer extracted the posterior *means* under column names no cleanup recognised
    (hunt 2026-09-30, u31-benchmarking-2). With no key to honour, a ``q05`` entry -- the abundance
    cell2location's own profile names first -- comes before the alphabetical keyword scan, so a
    profile-less read of the same export lands on the same matrix.
    """
    resolved = _resolve_prediction_key(obsm, prediction_key) if prediction_key else None
    if resolved:
        return resolved
    keys = [str(k) for k in obsm.keys()]
    for key in keys:
        if "q05" in key.lower():
            return key
    for key in keys:
        if any(kw in key.lower() for kw in _ABUNDANCE_OBSM_KEYWORDS):
            return key
    return None


def _unassigned_labels(profile) -> set[str]:
    """Label values the profile's prediction specs declare as "no cluster" (``OutputSpec.unassigned_labels``)."""
    if profile is None:
        return set()
    return {str(v) for s in profile.outputs if s.role == "prediction" for v in (s.unassigned_labels or ())}


def _assigned_labels(series, unassigned: set[str]) -> list[str]:
    """A label column as strings, without unassigned spots: NaN, and any value the profile declares unassigned."""
    labels = series.dropna().astype(str)
    if unassigned:
        labels = labels[~labels.isin(unassigned)]
    return labels.tolist()


def _declared_cluster_keys(profile) -> set[str]:
    """The label columns a profile names itself -- its own word, never second-guessed below."""
    keys: set[str] = set()
    if profile is None:
        return keys
    declared = [profile.prediction_key] + [s.prediction_key for s in profile.outputs if s.role == "prediction"]
    for key in declared:
        keys.update(str(k) for k in (key if isinstance(key, (list, tuple)) else [key]) if k)
    return keys


def _reference_obs(path):
    """The ``obs`` table of the input / ground-truth h5ad at ``path``, or ``None`` when there is none to read."""
    if not path or Path(path).suffix != ".h5ad" or not Path(path).is_file():
        return None
    try:
        from spatialomicsgym.utils.file_io import read_h5ad_backed

        with read_h5ad_backed(str(path)) as ref:
            return ref.obs.copy()
    except Exception:
        return None


#: Column names a table uses for its spot IDs, matched case-insensitively. ``obs_names`` is what the
#: clustering standardizer adds to ``_SPOT_ID_COLUMNS`` (``output_standardizer``'s CSV branch).
_CLUSTER_SPOT_ID_COLUMNS = (*_SPOT_ID_COLUMNS, "obs_names")


def _shipped_with_the_input(labels, reference_obs, key: str) -> bool:
    """Whether ``labels`` (a series indexed by spot ID) is the input's own ``key`` column, value for value.

    A tool's output is very often the AnnData it was handed plus one column of its own, so every
    annotation the dataset shipped with rides along -- and the generic candidate list knows those
    names as well as the tool's. Both Slide-seqV2 domain inputs carry the authors' Seurat clustering
    as ``obs['SCT_snn_res.0.5']``; whenever the declared key was missing (a SpaceFlow run recorded
    before its labels were attached, labels under a non-default ``cluster_key``) the inspector
    returned ``ready`` on it, and the ARI was computed on the dataset's own clustering rather than
    the tool's (hunt 2026-09-30, u31-benchmarking-1). A column identical to the input's is not
    something the tool said. Compared on the spots both tables carry, as strings, so a QC-filtered
    output and a categorical dtype still compare.
    """
    if reference_obs is None or key not in reference_obs.columns or not labels.index.is_unique:
        return False
    ref = reference_obs[key]
    if not ref.index.is_unique:
        return False
    common = labels.index.intersection(ref.index)
    if len(common) == 0:
        return False
    mine = labels.loc[common].astype(str).to_numpy()
    theirs = ref.loc[common].astype(str).to_numpy()
    return bool((mine == theirs).all())


def _csv_labels_by_spot(df, col):
    """``df[col]`` indexed by the table's spot IDs, or ``None`` when it does not identify its spots.

    The IDs are a column named for them, else the first column when its values are unique -- the
    ``write.csv`` / ``to_csv`` row-name convention.
    """
    lowered = {str(c).strip().lower(): c for c in df.columns}
    id_col = next((lowered[n] for n in _CLUSTER_SPOT_ID_COLUMNS if n in lowered and lowered[n] != col), None)
    if id_col is None and df.columns[0] != col and df[df.columns[0]].astype(str).is_unique:
        id_col = df.columns[0]
    if id_col is None:
        return None
    labels = df[col].copy()
    labels.index = df[id_col].astype(str).to_numpy()
    return labels


def _inspect_clustering(
    result: OutputInspectionResult,
    out_path: Path,
    all_files: list[Path],
    profile,
    reference_path: str | None = None,
):
    """Inspect spatial clustering output.

    ``reference_path`` is the input / ground-truth h5ad, when the caller has one: a *generic* label
    column found identical to the input's own is refused rather than scored (see
    :func:`_shipped_with_the_input`). A key the profile declares is the tool's word and is not checked.
    """
    import anndata as ad

    declared_keys = _declared_cluster_keys(profile)
    unassigned = _unassigned_labels(profile)
    _reference: list = []  # loaded on first need, at most once

    def _came_with_the_input(labels, key: str, source: str) -> bool:
        if not reference_path or key in declared_keys or labels is None:
            return False
        if not _reference:
            _reference.append(_reference_obs(reference_path))
        if not _shipped_with_the_input(labels, _reference[0], key):
            return False
        result.warnings.append(
            f"'{key}' in {source} is the input's own column, value for value -- it came with the data, "
            "not from the tool; not read as its prediction"
        )
        return True

    def _pick_cluster_column_not_from_the_input(head, path: Path, candidates) -> str | None:
        refused: list = []
        full = None
        while True:
            col = _pick_cluster_column(head, candidates, exclude=refused)
            if col is None or not reference_path or str(col) in declared_keys:
                return col
            if full is None:
                import pandas as pd

                full = pd.read_csv(str(path))
            if not _came_with_the_input(_csv_labels_by_spot(full, col), str(col), path.name):
                return col
            refused.append(col)

    # ── Priority 0: If profile specifies a CSV prediction file, check that first ──
    # Tools like BASS, PRECAST, IRIS, Seurat write CSV, not h5ad.
    if profile:
        for out_spec in profile.outputs:
            if out_spec.role == "prediction" and out_spec.format == "csv":
                matched = _match_prediction_files(all_files, out_spec.filename_pattern)
                if matched:
                    csv_path = matched[0]
                    try:
                        import pandas as pd

                        df = pd.read_csv(str(csv_path))
                        pred_key = out_spec.prediction_key
                        # An unassigned spot is not a cluster. ``.astype(str)`` renders NaN as the
                        # string "nan", which then counts once in ``set(labels)`` and once in
                        # ``len(labels)`` below -- so a K-cluster result with one unassigned spot
                        # publishes K+1 clusters over N spots instead of K over N-1. Tools emit
                        # these routinely: mclust returns NA for spots it will not assign and R
                        # writers leave the field empty. The h5ad route at the bottom of this
                        # function is the reference -- it has always done ``.dropna()`` -- and
                        # ``validate_output_completeness`` does the same on its own read. Only one
                        # reader compensates today (``workflow_gates:437`` filters "nan" back out of
                        # ``cluster_labels``), and four lines above it ``:430`` reads the inflated
                        # ``n_clusters`` raw, so the two contradict each other.
                        # Try the profile-specified key
                        #
                        # A value the spec declares unassigned (mist's 'isolated': ReST placed the spot
                        # in no region) is dropped the same way (hunt 2026-09-30, u31-benchmarking-11).
                        if pred_key and pred_key in df.columns:
                            labels = _assigned_labels(df[pred_key], unassigned)
                        else:
                            # Fallback: search for any cluster-like column
                            cluster_key_candidates = _get_cluster_key_candidates(profile)
                            found_col = None
                            for c in cluster_key_candidates:
                                if c in df.columns:
                                    found_col = c
                                    break
                            if found_col is None:
                                for c in df.columns:
                                    if any(
                                        kw in c.lower() for kw in ["cluster", "domain", "leiden", "louvain", "pred"]
                                    ):
                                        found_col = c
                                        break
                            if found_col:
                                labels = _assigned_labels(df[found_col], unassigned)
                                pred_key = found_col
                            else:
                                continue  # Skip this CSV, no cluster column found

                        if len(labels) > 0 and len(set(labels)) > 1:
                            result.prediction_file = str(csv_path)
                            result.prediction_format = "csv"
                            result.prediction_key = pred_key
                            result.prediction_type = "cluster_labels"
                            result.cluster_labels = labels
                            result.n_clusters = len(set(labels))
                            result.n_predictions = len(labels)
                            result.status = "ready"
                            return
                    except Exception:
                        pass

    # ── Strategy 1: find h5ad with cluster labels ──
    # The `excluded` list below tests the file *name*, so a staged input called `adata.h5ad` passes
    # it. _is_staged_input_path is the path-*component* test, and it is what the CSV route further
    # down already applies -- a tool's staged input is not its output on either path.
    h5ad_files = sorted(
        [f for f in all_files if f.suffix == ".h5ad" and f.stat().st_size >= 1024 and not _is_staged_input_path(f)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Filter out obvious non-output files (input preparations, data copies, etc.)
    excluded = (
        "input",
        "prepared",
        "original",
        "converted",
        "repaired",
        "mcp_ready",
        "fixed",
        "ready",
        "spatial_transcriptomics",
    )
    h5ad_candidates = [f for f in h5ad_files if not any(ex in f.stem.lower() for ex in excluded)]

    # If tool profile exists, try matching its pattern first
    if profile:
        for out_spec in profile.outputs:
            if out_spec.role == "prediction" and out_spec.format == "h5ad":
                matched = _match_prediction_files(all_files, out_spec.filename_pattern)
                if matched:
                    h5ad_candidates = matched + [f for f in h5ad_candidates if f not in matched]
                    break

    if not h5ad_candidates:
        # Fallback: only allow unexcluded h5ad files matching the profile pattern
        # Do NOT fall back to input-like h5ad files (that defeats the exclusion logic)
        if profile:
            for out_spec in profile.outputs:
                if out_spec.role == "prediction" and out_spec.format == "h5ad":
                    matched = _match_prediction_files(h5ad_files, out_spec.filename_pattern)
                    if matched:
                        h5ad_candidates = matched
                        break

    if not h5ad_candidates:
        # Last resort: check CSV files for cluster labels
        csv_files = [f for f in all_files if f.suffix == ".csv"]
        # Exclude CSVs that are clearly input exports (counts, coordinates), and anything under a
        # staged-input or derived-artifact directory (STAGED_INPUT_DIRS).
        input_csv_indicators = ("counts", "coordinates", "coords", "csv_export")
        csv_files = [
            f
            for f in csv_files
            if not any(ind in f.stem.lower() for ind in input_csv_indicators) and not _is_staged_input_path(f)
        ]
        # Prioritize CSVs with metadata/cluster/domain in filename
        metadata_csvs = [
            f for f in csv_files if any(kw in f.stem.lower() for kw in ("metadata", "cluster", "clusters", "domain"))
        ]
        csv_files = metadata_csvs + [f for f in csv_files if f not in metadata_csvs]
        # Also use the cluster key candidates for column matching
        cluster_key_candidates = _get_cluster_key_candidates(profile)
        for cf in csv_files:
            try:
                import pandas as pd

                df = pd.read_csv(str(cf), nrows=5)
                # Known candidates first, in the priority order they were ranked in — then the
                # keyword and Seurat fallbacks. See _pick_cluster_column. A column the input shipped
                # with is passed over for the next one (hunt 2026-09-30, u31-benchmarking-1).
                cluster_col = _pick_cluster_column_not_from_the_input(df, cf, cluster_key_candidates)
                if cluster_col:
                    df_full = pd.read_csv(str(cf))
                    # An unassigned spot is not a cluster -- see the profile branch above.
                    labels = _assigned_labels(df_full[cluster_col], unassigned)
                    result.prediction_file = str(cf)
                    result.prediction_format = "csv"
                    result.prediction_key = cluster_col
                    result.prediction_type = "cluster_labels"
                    result.cluster_labels = labels
                    result.n_clusters = len(set(labels))
                    result.n_predictions = len(labels)
                    result.status = "ready"
                    return
            except Exception:
                continue

        result.status = "incomplete"
        result.errors.append("No h5ad or CSV with cluster labels found")
        return

    # Try each h5ad candidate
    prediction_key_candidates = _get_cluster_key_candidates(profile)

    for h5ad_path in h5ad_candidates:
        try:
            adata = ad.read_h5ad(str(h5ad_path))
            for key in prediction_key_candidates:
                if key in adata.obs.columns:
                    labels = _assigned_labels(adata.obs[key], unassigned)
                    if len(labels) > 0 and len(set(labels)) > 1:
                        if _came_with_the_input(adata.obs[key], key, h5ad_path.name):
                            continue
                        result.prediction_file = str(h5ad_path)
                        result.prediction_format = "h5ad"
                        result.prediction_key = key
                        result.prediction_type = "cluster_labels"
                        result.cluster_labels = labels
                        result.n_clusters = len(set(labels))
                        result.n_predictions = len(labels)
                        result.status = "ready"
                        return
        except Exception as e:
            result.warnings.append(f"Could not read {h5ad_path.name}: {e}")

    # h5ad search failed — try CSV fallback (e.g. Seurat writes clusters.csv)
    csv_files = [f for f in all_files if f.suffix == ".csv"]
    # Exclude input export CSVs and anything under a staged-input directory (STAGED_INPUT_DIRS).
    input_csv_kw = ("counts", "coordinates", "coords", "csv_export")
    csv_files = [
        f for f in csv_files if not any(ind in f.stem.lower() for ind in input_csv_kw) and not _is_staged_input_path(f)
    ]
    metadata_csvs = [
        f for f in csv_files if any(kw in f.stem.lower() for kw in ("metadata", "cluster", "clusters", "domain"))
    ]
    csv_files = metadata_csvs + [f for f in csv_files if f not in metadata_csvs]
    cluster_key_candidates_csv = _get_cluster_key_candidates(profile)
    for cf in csv_files:
        try:
            import pandas as pd

            df = pd.read_csv(str(cf), nrows=5)
            cluster_col = _pick_cluster_column_not_from_the_input(df, cf, cluster_key_candidates_csv)
            if cluster_col:
                df_full = pd.read_csv(str(cf))
                # An unassigned spot is not a cluster -- see the profile branch above.
                labels = _assigned_labels(df_full[cluster_col], unassigned)
                result.prediction_file = str(cf)
                result.prediction_format = "csv"
                result.prediction_key = cluster_col
                result.prediction_type = "cluster_labels"
                result.cluster_labels = labels
                result.n_clusters = len(set(labels))
                result.n_predictions = len(labels)
                result.status = "ready"
                return
        except Exception:
            continue

    # TSV/plain-text domain file fallback (SpaceFlow writes domains.tsv)
    tsv_files = [f for f in all_files if f.suffix == ".tsv" and "domain" in f.stem.lower()]
    for tf in tsv_files:
        try:
            with open(str(tf)) as fh:
                lines = [line.strip() for line in fh if line.strip()]
            # Plain text: one domain label per line, no header
            if lines and all(len(l) < 20 for l in lines[:10]):
                # Check if they look like cluster labels (integers or short strings)
                labels = [label for label in lines if label not in unassigned]
                n_unique = len(set(labels))
                if n_unique > 1:
                    result.prediction_file = str(tf)
                    result.prediction_format = "tsv"
                    result.prediction_key = "domain"
                    result.prediction_type = "cluster_labels"
                    result.cluster_labels = labels
                    result.n_clusters = n_unique
                    result.n_predictions = len(labels)
                    result.status = "ready"
                    result.warnings.append(f"Domain labels read from plain TSV: {tf.name}")
                    return
        except Exception:
            continue

    result.status = "incomplete"
    result.errors.append(
        f"h5ad files found but no cluster label column detected. Searched keys: {prediction_key_candidates[:5]}"
    )


def _pick_cluster_column(df, candidates, exclude=()) -> str | None:
    """Which column of a CSV carries the cluster labels. The priority list decides, not the file.

    ``candidates`` comes from :func:`_get_cluster_key_candidates` and is ordered: the profile's own
    ``prediction_key`` first, then tool-specific names, then generic ones. Both CSV branches used to
    select with ``[c for c in df.columns if c in candidates][0]``, which reads the *file's* column
    order instead -- a table carrying both ``spatial_domain`` (candidate #1) and ``leiden`` (near the
    tail) resolved to whichever the header happened to list first. ``benchmark_runner`` forwards the
    chosen key into ``standardize_clustering_output``, so on a run that writes both, the wrong
    column's labels reach the final ARI/NMI. The h5ad branch never had this bug -- it iterates the
    candidates -- and this is that same loop, now shared so the two cannot drift apart again.

    The two fallbacks below are unchanged and still ordered after it: a keyword match, for a column
    name no profile predicted, then Seurat's ``<assay>_snn_res.<n>``. Both still go by the file's own
    order, which is all there is to go on once the priority list has nothing to say.

    ``exclude`` names columns already refused, so a caller can ask for the next choice.
    """
    columns = [c for c in df.columns if c not in exclude]
    for key in candidates:
        if key in columns:
            return key
    for c in columns:
        if any(kw in str(c).lower() for kw in ("cluster", "domain", "leiden", "louvain", "pred")):
            return c
    for c in columns:
        if re.match(r"(RNA|SCT)_snn_res\.\d", str(c)):
            return c
    return None


def _get_cluster_key_candidates(profile) -> list[str]:
    """Get ordered list of cluster label column candidates.

    ``prediction_key`` is contractually ``str | list[str]`` -- ``tool_output_registry`` says so, 14
    shipped profiles are lists, and the sibling ``_resolve_prediction_key`` reads both. Appending it
    whole treated the list as ONE candidate, which the ``set``-based dedup below then rejected as an
    ``unhashable type: 'list'``: the TypeError escaped ``inspect_tool_output`` and discarded a run
    that had produced valid output. A list of alternatives is a list of candidates, in its own order.
    """
    candidates: list[str] = []
    if profile and profile.prediction_key:
        key = profile.prediction_key
        candidates.extend(str(k) for k in (key if isinstance(key, (list, tuple)) else [key]) if k)

    # Generic candidates in priority order — from audit of all worker scripts
    candidates.extend(
        [
            # Tool-specific keys (exact names from worker source code)
            "spatial_domain",  # scanpy_spatial_worker
            "domain",  # graphst_worker clustering task
            "stagate_domain",  # stagate_worker
            "DeepST_refine_domain",  # deepst_worker (priority 1)
            "DeepST_domain",  # deepst_worker (priority 2)
            "prost_domain",  # prost_worker domains task
            "miso_cluster",  # miso_worker
            "sedr_dec_cluster",  # sedr_worker (DEC mode)
            "sedr_cluster",  # sedr_worker (KMeans fallback)
            "cluster_cellcharter",  # cellcharter_worker
            "spicemix_factor",  # spicemix_worker (dominant factor)
            "splane_cluster",  # spacel_worker
            "spaceflow_domain",  # spaceflow_worker
            "spatialprompt_cluster",  # spatialprompt_worker
            "iris_cluster",  # iris_spatial_worker (obs key when loaded from CSV)
            "bass_cluster",  # bass_worker (obs key when loaded from CSV)
            "mclust",  # stagate/graphst with mclust backend
            # Seurat-specific keys
            "seurat_clusters",  # seurat_qc_cluster
            # Generic keys
            "louvain",  # stlearn, spiral
            "leiden",  # scanpy fallback, spiral
            # Seurat's per-resolution columns, after louvain/leiden: no profile writes them as its
            # prediction (the Seurat profiles name seurat_clusters), and datasets ship with them --
            # both Slide-seqV2 domain inputs carry the authors' SCT_snn_res.0.5, which outranked a
            # tool's own leiden labels sitting beside it (hunt 2026-09-30, u31-benchmarking-1).
            "RNA_snn_res.0.8",  # Seurat default resolution
            "SCT_snn_res.0.8",  # Seurat SCTransform resolution
            "RNA_snn_res.0.5",  # Seurat alternative resolution
            "SCT_snn_res.0.5",  # Seurat SCTransform alternative
            "domains",
            "cluster",
            "clusters",
            "pred",
            "prediction",
            "predicted",
            "cell_type_pred",
            "predicted_cell_type",
            "phenotype",
            "class",
            "label",
            "type",
        ]
    )
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


# An SVG reader finds its column by name -- anything matching "svg", "spatially_variable",
# "significant" or "hotspot" -- and then keeps the entries *below* 0.05. That rule is only correct
# for a p-value. Three of those four names routinely spell a spatial *score* instead: Moran's I,
# SpatialDE/SOMDE's FSV and SpaGFT's gft_score are all large when a gene IS spatially variable, so
# thresholding them returns precisely the genes the tool rejected. The value distribution cannot
# tell the two apart -- Moran's I and a p-value both live in [0, 1] -- so the column name is the
# only sound signal.
_PVALUE_COLUMN_RE = re.compile(r"fdr|[pq][-_ ]?(?:val|value|values|adj)|adjusted[-_ ]?p|p[-_ ]?adj", re.IGNORECASE)
_SCORE_COLUMN_RE = re.compile(
    r"moran|geary|fsv|gft|llr|intrinsic|statistic|[-_ ]?score|z[-_ ]?score|ranking", re.IGNORECASE
)

# The third thing those four keywords spell, and the one with no ordering at all: a label. Hotspot
# writes `hotspot_module` -- gene module IDs, `-1` for the genes it could not assign -- and the same
# vocabulary covers a cluster, domain or component ID. A p-value orders genes and a score orders
# them the other way; an identifier orders nothing, so neither `< 0.05` nor `nlargest` means
# anything over it. `< 0.05` in particular keeps modules `-1` and `0`: mostly the tool's own
# rejects, returned as its spatially-variable genes.
#
# Checked *after* the two above so a statistic word still wins: `module_fdr` is a p-value per
# module and `cluster_score` is a score, and both keep the behaviour they have today. `id` is
# anchored to a word boundary because bare `id` is a substring of too much ordinary English.
_ID_COLUMN_RE = re.compile(
    r"module|cluster|domain|component|segment|assignment|label|group|(?:^|[^a-z])id(?:$|[^a-z])",
    re.IGNORECASE,
)

# How many genes a bare score column is worth when the tool never thresholded it. There is no
# defensible cutoff for a ranking, so the reader takes the head of it and says so in a warning.
_SCORE_COLUMN_TOP_N = 100


def _svg_column_kind(name: str) -> str:
    """Classify an SVG ``var``/CSV column by name: ``pvalue``, ``score``, ``identifier``, ``unknown``."""
    if _PVALUE_COLUMN_RE.search(name):
        return "pvalue"
    if _SCORE_COLUMN_RE.search(name):
        return "score"
    if _ID_COLUMN_RE.search(name):
        return "identifier"
    return "unknown"


def _inspect_svg(
    result: OutputInspectionResult,
    out_path: Path,
    all_files: list[Path],
    profile,
):
    """Inspect SVG detection output."""
    import pandas as pd

    gene_col_candidates = ["Gene", "gene", "gene_name", "g", "feature", "var_name", "genes"]
    # Shared with the standardizer rather than copied: the two used to keep separate lists, both
    # written adjusted-first, and both iterated in the *file's* column order, so neither ordering
    # did anything and both thresholded on whichever p-value the tool wrote first. Importing here
    # (not at module scope) keeps the existing inspector <- standardizer import direction.
    from spatialomicsgym.benchmarking.output_standardizer import _svg_calls
    from spatialomicsgym.benchmarking.tool_output_registry import get_profile

    # Which files are the tool's own top-k, asked of its profile in both passes: the generic pass drops
    # ``profile`` to widen the search, not to forget what a file the tool wrote is.
    declared = profile if profile is not None else get_profile(result.tool_name)

    # Try CSV/TSV files first (most SVG tools output CSV), skipping staged inputs and derived
    # artifacts (STAGED_INPUT_DIRS).
    csv_files = sorted(
        [
            f
            for f in all_files
            if f.suffix in (".csv", ".tsv") and f.stat().st_size >= 100 and not _is_staged_input_path(f)
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # If profile has a pattern, prioritize matching files.
    # The generic scan above only collects .csv/.tsv, so a tool that writes its score table
    # under another extension (SpotGF writes a tab-separated SpotGF_scores.txt) is invisible
    # to it. A profile naming that file explicitly is authoritative, so honour any tabular
    # format here. json/h5ad prediction specs stay excluded, as they were under `== "csv"`.
    if profile:
        for out_spec in profile.outputs:
            if out_spec.role == "prediction" and out_spec.format in _TABULAR_FORMATS:
                matched = [f for f in _match_prediction_files(all_files, out_spec.filename_pattern) if f.is_file()]
                if matched:
                    csv_files = matched + [f for f in csv_files if f not in matched]
                    if out_spec.prediction_key:
                        keys = out_spec.prediction_key
                        keys = [keys] if isinstance(keys, str) else list(keys)
                        gene_col_candidates = keys + gene_col_candidates
                    break

    for csv_path in csv_files:
        try:
            sep = _tabular_sep(csv_path)
            df = pd.read_csv(str(csv_path), sep=sep)

            gene_col = None
            for col in df.columns:
                if col in gene_col_candidates or col.lower() in [c.lower() for c in gene_col_candidates]:
                    gene_col = col
                    break

            # Check if index looks like gene names (handles Hotspot, SVGbit, etc.)
            if gene_col is None:
                idx = df.index
                if idx.dtype == object or (hasattr(idx, "inferred_type") and idx.inferred_type == "string"):
                    sample = [str(x) for x in idx[:5]]
                    if any(not s.replace(".", "").replace("-", "").replace("_", "").isdigit() for s in sample):
                        df = df.reset_index()
                        gene_col = df.columns[0]
                        result.warnings.append(
                            f"Gene names inferred from CSV index as column '{gene_col}' in {csv_path.name}"
                        )

            # Check for 'Unnamed: 0' column containing gene names (CSV where first col is unnamed)
            if gene_col is None:
                for col in df.columns:
                    if str(col).startswith("Unnamed"):
                        sample = [str(x) for x in df[col].dropna().head(5)]
                        if sample and any(
                            not s.replace(".", "").replace("-", "").replace("_", "").isdigit() for s in sample
                        ):
                            gene_col = col
                            result.warnings.append(
                                f"Gene names found in unnamed column '{gene_col}' in {csv_path.name}"
                            )
                            break

            if gene_col is None:
                continue

            # Extract genes by the call rule the standardizer applies to the same file -- a stated
            # selection, else p < 0.05, else the top of a score-only ranking, else every gene -- so
            # the two layers call one set (hunt 2026-09-30, u31-benchmarking-3/-4/-5/-18). Before,
            # this route thresholded p-values only: a score-only table (svgbit's AI.csv) came back
            # as every gene, while the h5ad route below took the top of the same ranking.
            #
            # The columns are read the way the picker validated them, not the way pandas boxed them:
            # an adjusted-p vector that underflowed to 0 and saturated at 1 reads back int64, and one
            # carrying ``n.s.`` reads back object. Gating on the container's dtype skipped the
            # threshold for both and reported every gene, with status='ready' and no warning.
            calls = _svg_calls(df, gene_col, selected=_is_declared_selection(csv_path, declared))
            df = calls.frame
            sig_col = calls.pval_col
            thresholded = bool(calls.rule)
            genes = df.loc[calls.significant, gene_col].astype(str).str.strip().tolist()
            notes = [f"{csv_path.name}: {note}" for note in calls.notes]

            # Filter blank probes
            genes = [g for g in genes if not g.startswith("Blank")]

            if not genes and thresholded:
                # A real threshold ran and selected nothing: that IS the tool's answer. Falling
                # back to the whole gene table here (as this did) scores recall 1.0 against any
                # ground truth, and hides the empty result from the workflow gate that exists to
                # block it. Record it, leave the status non-ready, and stop looking.
                result.prediction_file = str(csv_path)
                result.prediction_format = "csv"
                result.prediction_key = gene_col
                result.prediction_type = "gene_list"
                result.svg_genes = []
                result.n_svg_genes = 0
                result.n_predictions = 0
                result.status = "empty"
                result.warnings.extend(notes)
                result.warnings.append(
                    f"{csv_path.name}: no gene passed {calls.rule} — the tool reported no spatially variable genes"
                )
                return

            # Reject obvious spot-barcode files masquerading as gene lists.
            # Visium/MERFISH/Slide-seq spot barcodes look like "AAACAAGTATCTCCCA-1"
            # (16 A/C/G/T chars + "-1") or similar pure-nucleotide patterns. If >50% of
            # the picked "genes" match this shape, the inspector grabbed a spot-metadata
            # file (e.g. converted_csv/metadata.csv) instead of a real SVG prediction.
            if genes:
                _bc_re = re.compile(r"^[ACGT]{12,20}(?:[-_]\d+)?$")
                bc_hits = sum(1 for g in genes[:200] if _bc_re.match(g))
                bc_frac = bc_hits / max(1, min(len(genes), 200))
                if bc_frac > 0.5:
                    result.warnings.append(
                        f"{csv_path.name}: {bc_frac:.0%} of 'genes' look like spot barcodes "
                        f"(e.g. {genes[0]!r}); rejecting as a non-SVG file"
                    )
                    continue

                # Reject pairwise/adjacency matrices masquerading as gene tables.
                # A cell-type x cell-type neighbourhood-enrichment matrix (squidpy) or a
                # gene x gene correlation matrix carries the SAME labels down the rows and
                # across the columns; a real SVG table has genes on one axis and statistic
                # names on the other. Without this, squidpy_nhood_enrichment_zscore.csv is
                # reported as a clean pass of "3 SVGs: TypeA, TypeB, TypeC".
                row_labels = set(df[gene_col].dropna().astype(str))
                other_cols = {str(c) for c in df.columns if c != gene_col}
                if len(row_labels) >= 2 and other_cols == row_labels:
                    result.warnings.append(
                        f"{csv_path.name}: row labels and column labels are the same set "
                        f"(e.g. {sorted(row_labels)[:3]!r}) — this is a pairwise matrix, not a "
                        "gene x statistic table; rejecting as a non-SVG file"
                    )
                    continue

            if genes:
                result.prediction_file = str(csv_path)
                result.prediction_format = "csv"
                result.prediction_key = gene_col
                result.prediction_type = "gene_list"
                result.svg_genes = genes
                result.n_svg_genes = len(genes)
                result.n_predictions = len(genes)
                result.status = "ready"
                result.warnings.extend(notes)
                if sig_col:
                    result.prediction_type = "gene_list_with_pvalues"
                elif not thresholded:
                    result.warnings.append(
                        f"No significance column found in {csv_path.name} — all genes treated as candidates"
                    )
                return

        except Exception as e:
            result.warnings.append(f"Could not parse {csv_path.name}: {e}")

    # Fallback: check h5ad var annotations
    # A tool's staged input is not its output. Every CSV route in this file filters it out; this
    # scan had no exclusion at all, so a run that wrote nothing could be answered from the input
    # prepared for it -- the SPIRAL failure _is_staged_input_path records, on the .h5ad path.
    h5ad_files = [
        f for f in all_files if f.suffix == ".h5ad" and f.stat().st_size >= 1024 and not _is_staged_input_path(f)
    ]
    for h5ad_path in h5ad_files:
        try:
            import anndata as ad

            adata = ad.read_h5ad(str(h5ad_path))
            for vc in adata.var.columns:
                if any(kw in vc.lower() for kw in ["svg", "spatially_variable", "significant", "hotspot"]):
                    warning = None
                    if adata.var[vc].dtype == bool:
                        # A bool holds a flag, never an ID, so the identifier fence below does not
                        # reach it: `in_significant_module` is a membership flag and reads correctly.
                        genes = list(adata.var_names[adata.var[vc]])
                    elif _svg_column_kind(vc) == "identifier":
                        # Numbers that label rather than rank. Declined rather than thresholded --
                        # and said out loud, because int64 already fell through the dtype gate below
                        # to a silent `continue`, which produced the right answer for a reason that
                        # was nobody's decision and that one `.astype(int)` away from being wrong.
                        result.warnings.append(
                            f"{vc} in {h5ad_path.name} looks like a module/cluster identifier, not a "
                            "significance column; it was not read as an SVG result"
                        )
                        continue
                    elif adata.var[vc].dtype in (float, "float64") or str(adata.var[vc].dtype) in (
                        "float64",
                        "float32",
                    ):
                        kind = _svg_column_kind(vc)
                        if kind == "score":
                            # Large means spatially variable here, so rank instead of thresholding.
                            # dropna first: nlargest is sort_values().head(n), and sort_values puts
                            # NaN *last* rather than discarding it -- so an all-NaN score column, a
                            # tool that ran and computed nothing, came back as a full top-N, which
                            # is the first N genes in file order wearing a ranking.
                            ranked = adata.var[vc].dropna().nlargest(_SCORE_COLUMN_TOP_N)
                            genes = [str(g) for g in ranked.index]
                            warning = (
                                f"{vc} is a spatial score, not a p-value: took its top "
                                f"{len(genes)} of {adata.n_vars} genes by rank because the tool "
                                "wrote no threshold"
                            )
                        else:
                            # p-value, or a name that says neither -- keep the established rule.
                            genes = list(adata.var_names[adata.var[vc] < 0.05])
                            if kind == "unknown":
                                warning = (
                                    f"{vc} was thresholded at 0.05 as a p-value, but its name does "
                                    "not say whether it holds one; verify the direction"
                                )
                    else:
                        continue
                    if genes:
                        if warning:
                            result.warnings.append(warning)
                        result.prediction_file = str(h5ad_path)
                        result.prediction_format = "h5ad"
                        result.prediction_key = vc
                        result.prediction_type = "gene_list"
                        result.svg_genes = genes
                        result.n_svg_genes = len(genes)
                        result.n_predictions = len(genes)
                        result.status = "ready"
                        return
        except Exception:
            continue

    result.status = "incomplete"
    result.errors.append("No SVG prediction file found")


def _inspect_deconvolution(
    result: OutputInspectionResult,
    out_path: Path,
    all_files: list[Path],
    profile,
):
    """Inspect deconvolution output."""
    import pandas as pd

    from spatialomicsgym.benchmarking.output_standardizer import _orient_proportions

    # Deconvolution outputs are typically CSV proportion matrices
    proportion_keywords = (
        "proportion",
        "probabilities",
        "composition",
        "celltype_frac",
        "cell_type_frac",
        "deconv",
        "abundance",
        "weights",
        "rctd",
        "tangram_celltype",
        "tacco",
        "graphst_celltype",
        "spotlight",
        "stride",
        "stdeconvolve",
        "destvi",
        "card",
    )

    # Exclude known non-output CSVs (input data, reference, coordinates, metadata, previews)
    exclude_keywords = (
        "counts",
        "coordinates",
        "coords_for",
        "spatial_coords",
        "metadata",
        "ref_celltype",
        "inf_aver",
        "preview",
        "prepared",
        "converted",
        "input",
    )
    # Staged inputs and derived artifacts are excluded by STAGED_INPUT_DIRS.
    csv_files = sorted(
        [
            f
            for f in all_files
            if f.suffix == ".csv"
            and f.stat().st_size >= 256
            and not any(kw in f.stem.lower() for kw in exclude_keywords)
            and not _is_staged_input_path(f)
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Prioritize files matching proportion keywords
    keyword_csvs = [f for f in csv_files if any(kw in f.stem.lower() for kw in proportion_keywords)]

    # If profile has a pattern, try it
    if profile:
        for out_spec in profile.outputs:
            if out_spec.role == "prediction":
                matched = _match_prediction_files(all_files, out_spec.filename_pattern)
                if matched:
                    keyword_csvs = matched + [f for f in keyword_csvs if f not in matched]
                    break

    # Try keyword-matched CSVs first, then all CSVs
    candidates = keyword_csvs + [f for f in csv_files if f not in keyword_csvs]

    for csv_path in candidates:
        try:
            # A profile may name a delimited table that is not comma-separated (STRIDE writes
            # a tab-separated <prefix>_spot_celltype_frac.txt); read it with its own separator
            # so it is not mis-parsed into a single column. .csv/.tsv are unaffected.
            sep = _tabular_sep(csv_path)
            # The spot IDs are not always in the first column -- CARD writes them last. See
            # _proportions_index_col; this is 0 whenever there is no such column to find.
            index_col = _proportions_index_col(csv_path, sep)
            df = pd.read_csv(str(csv_path), index_col=index_col, nrows=5, sep=sep)
            numeric_cols = df.select_dtypes(include="number").shape[1]
            if numeric_cols >= 2:
                # Reject columns that are all-numeric strings (e.g. "1", "6982.0",
                # "8346.0") or all in the form "factor_<n>" / "arch_<n>" — these
                # cannot map to any cell-type name in the aggregation YAML and
                # produce silent null Pearson. Symptom seen with starfysh when
                # adata.uns['cell_types'] is missing or polluted with index ints.
                col_strs = [str(c) for c in df.columns]
                bad_ct = sum(
                    1
                    for c in col_strs
                    if re.match(r"^\d+(\.\d+)?(\.\d+)?$", c)
                    or re.match(r"^(factor|arch|archetype|component|topic)[_\-]?\d+$", c, re.IGNORECASE)
                )
                if len(col_strs) and bad_ct / len(col_strs) > 0.5:
                    result.warnings.append(
                        f"{csv_path.name}: columns look like archetype/factor indices "
                        f"(e.g. {col_strs[:3]!r}), not cell-type names — rejecting "
                        "(re-run tool with proper signature_csv / cell_types annotation)"
                    )
                    continue
                # The same test one axis over: the column names above say what the values are, the
                # row labels say what they are *of*. A cell-type x cell-type colocalization matrix
                # has real cell-type names in both places and was scored as this run's per-spot
                # prediction on two recorded runs -- (7, 7) for a 12,645-bead puck, (44, 44) for a
                # 4,033-spot section. See _is_self_relation_table for why the values cannot decide
                # this and the columns cannot either.
                if _is_self_relation_table(df):
                    result.warnings.append(
                        f"{csv_path.name}: row labels are cell-type names drawn from its own columns "
                        f"(e.g. {[str(i) for i in df.index][:3]!r}), so it relates cell types to each "
                        "other rather than assigning them to spots — rejecting"
                    )
                    continue
                # This looks like a proportions matrix
                df_full = pd.read_csv(str(csv_path), index_col=index_col, sep=sep)
                # Spots on the rows, by the standardizer's own test: SpaCET writes cell types x spots,
                # which used to be reported as cell types named by spot barcodes (hunt 2026-09-30,
                # u31-benchmarking-9).
                oriented, transposed = _orient_proportions(df_full.select_dtypes(include="number"), wide_only=True)
                if transposed:
                    df_full = oriented
                    result.warnings.append(f"{csv_path.name}: {transposed}")
                result.prediction_file = str(csv_path)
                result.prediction_format = "csv"
                result.prediction_type = "proportions_matrix"
                result.proportions_shape = df_full.shape
                result.cell_types = list(df_full.columns)
                result.n_predictions = df_full.shape[0]
                result.status = "ready"
                return
        except Exception:
            continue

    # Fallback: check h5ad obsm for abundance matrices.
    # Priority order:
    #   (1) h5ads matched by the profile's prediction-h5ad OutputSpec
    #   (2) any h5ad in the directory
    # For each, first try the profile's prediction_key (str or list[str]);
    # then fall back to keyword-based key inference.
    # A tool's staged input is not its output. Every CSV route in this file filters it out; this
    # scan had no exclusion at all, so a run that wrote nothing could be answered from the input
    # prepared for it -- the SPIRAL failure _is_staged_input_path records, on the .h5ad path.
    h5ad_files = [
        f for f in all_files if f.suffix == ".h5ad" and f.stat().st_size >= 1024 and not _is_staged_input_path(f)
    ]
    profile_h5ad: list[Path] = []
    profile_keys: list = []
    if profile:
        for out_spec in profile.outputs:
            if out_spec.role == "prediction" and out_spec.format == "h5ad":
                m = _match_prediction_files(all_files, out_spec.filename_pattern)
                profile_h5ad.extend([f for f in m if f not in profile_h5ad])
                if out_spec.prediction_key:
                    profile_keys.append(out_spec.prediction_key)
        # Profile-level prediction_key as final fallback
        if profile.prediction_key:
            profile_keys.append(profile.prediction_key)
    ordered_h5ads = profile_h5ad + [f for f in h5ad_files if f not in profile_h5ad]
    for h5ad_path in ordered_h5ads:
        try:
            import anndata as ad

            adata = ad.read_h5ad(str(h5ad_path))
            chosen_key: str | None = None
            for k in profile_keys:
                resolved = _resolve_prediction_key(adata.obsm, k)
                if resolved:
                    chosen_key = resolved
                    break
            if chosen_key is None:
                chosen_key = _abundance_obsm_key(adata.obsm)
            if chosen_key is None:
                continue
            mat = adata.obsm[chosen_key]
            result.prediction_file = str(h5ad_path)
            result.prediction_format = "h5ad"
            result.prediction_key = chosen_key
            result.prediction_type = "cell_type_abundance_matrix"
            result.proportions_shape = mat.shape
            result.n_predictions = mat.shape[0]
            result.conversion_needed = True
            result.conversion_notes = f"Extract obsm['{chosen_key}'] and convert to proportions CSV"
            result.status = "ready"
            return
        except Exception:
            continue

    # Surface a visible warning when a profile existed but every prediction
    # OutputSpec missed: this is the silent-failure path that produced
    # phantom-shape predictions in past Wave-1 cells.
    if profile:
        attempted = [getattr(o, "filename_pattern", "?") for o in profile.outputs if o.role == "prediction"]
        result.errors.append(
            f"Profile-driven detection MISSED for {profile.tool_name}: "
            f"no file matched any of {attempted!r} and no obsm fallback hit "
            f"(searched {len(h5ad_files)} h5ad + {len(all_files)} total files)."
        )
    result.status = "incomplete"
    result.errors.append("No deconvolution proportion matrix found")
