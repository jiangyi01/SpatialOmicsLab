"""Mandatory workflow gates for the SpatialOmicsLab benchmark pipeline.

Three gates enforce data validity at each workflow stage:
  1. pre_execution_gate  — block tool execution on invalid inputs
  2. post_execution_gate — block evaluation on missing/malformed outputs
  3. pre_evaluation_gate — block metrics on ambiguous/incomplete predictions

All gates return a GateResult. Callers MUST check gate.passed before proceeding.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spatialomicsgym.generated_report import split_report_files
from spatialomicsgym.paths import work_output_roots

logger = logging.getLogger(__name__)

# This checkout. Derived, because the misplaced-output search below looks inside the repo: a tool
# launched from the repo root that ignores its output_dir writes there, which is exactly what the
# search is for. Spelled as an absolute constant it only matched on the machine it was written on.
# Two directories since the re-layout: the agent part (``<repo>/agent``, holding ``tools/``) and the
# repository root above it, which is where the portal and the CLI launch tools from.
_AGENT_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = _AGENT_DIR.parent


def _output_recovery_search_dirs() -> list[Path]:
    """Where a tool's output may have landed when it ignored the ``output_dir`` it was given.

    Two kinds of place, and only the first is specific to this module: the repo itself, because a
    tool launched from the repo root that ignores its ``output_dir`` writes *there*, which is
    exactly what this search is for.

    The rest is the portal scratch tree, and its branch set comes from
    :func:`spatialomicsgym.paths.work_output_roots` rather than being spelled again here. That
    resolver mirrors ``agent/tools/base_mcp.py::default_output_dir``, which cannot be imported (``tools/``
    is not on this package's ``sys.path`` and a library module must not put it there) and is pinned
    to it by ``test/test_output_roots_agree.py``. Held as a union, not a re-run of the resolver:
    the writer picks exactly one root, while this reader is hunting for a file that may have been
    written under a different setting, by an earlier run, or by a portal whose own default differs.
    """
    dirs = [_AGENT_DIR / "tools", _REPO_ROOT]
    dirs.extend(Path(root) for root in work_output_roots())
    unique: list[Path] = []
    for d in dirs:
        if d not in unique:
            unique.append(d)
    return unique


# Tools that require histology images to function
IMAGE_DEPENDENT_TOOLS = {
    "stlearn_spatial_clustering",
    "deepst_identify_domains",
    "run_miso",  # MISO can work without but has image mode
}

# Tools that strictly require images (won't produce output without them)
STRICT_IMAGE_TOOLS = {
    "stlearn_spatial_clustering",
}

# Platforms known to have no histology images
NO_IMAGE_PLATFORMS = {
    "MERFISH",
    "Slide-seqV2",
    "seqFISH",
    "STARmap",
}


@dataclass
class GateResult:
    """Result of a workflow gate check."""

    gate_name: str
    passed: bool
    status: str = "unknown"  # "passed", "blocked", "warning", "error"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate_name,
            "passed": self.passed,
            "status": self.status,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "details": self.details,
        }


def pre_execution_gate(
    tool_name: str,
    task_type: str,
    input_path: str,
    output_dir: str,
    sc_ref_path: str | None = None,
    **kwargs: Any,
) -> GateResult:
    """Validate inputs before tool execution.

    Checks:
    1. Input file exists and is readable
    2. Input format matches tool requirements (via data_validation)
    3. Output directory is writable
    4. Tool is compatible with the task type
    5. Required metadata exists (spatial coords, cell types, etc.)
    6. Reference data exists when needed (deconvolution)
    """
    gate = GateResult(gate_name="pre_execution", passed=True, status="passed")

    # 1. Input file exists
    inp = Path(input_path)
    if not inp.exists():
        gate.passed = False
        gate.status = "blocked"
        gate.reasons.append(f"Input file not found: {input_path}")
        return gate

    if inp.stat().st_size == 0:
        gate.passed = False
        gate.status = "blocked"
        gate.reasons.append(f"Input file is empty: {input_path}")
        return gate

    # 2. Output directory
    out = Path(output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        gate.passed = False
        gate.status = "blocked"
        gate.reasons.append(f"Cannot create output directory: {e}")
        return gate

    # 3. Data readiness check (delegates to data_validation module)
    try:
        from spatialomicsgym.agent.data_validation import check_data_readiness

        report = check_data_readiness(input_path, tool_name, sc_ref_path=sc_ref_path)
        gate.details["data_readiness"] = report

        if not report["is_ready"]:
            blocking_issues = [
                i
                for i in report["issues"]
                if i["status"] not in ("preferred_missing", "preferred_missing_but_alias_exists", "check_manually")
            ]
            if blocking_issues:
                if report.get("auto_fixable"):
                    # These warnings reach the scored agent's prompt (benchmark_runner appends them under
                    # "WARNINGS from input validation"). They used to say "auto_fix_data() should be
                    # called": a name the REPL does not bind, whose default rewrites the input h5ad in
                    # place. State what the data holds instead, and keep the input untouched (hunt
                    # 2026-09-30, uL4-honesty-5).
                    for issue in blocking_issues:
                        gate.warnings.append(
                            f"Input data: {issue['field']} is missing but recoverable from the file -- "
                            f"{issue['fix']}. Do this on a copy; do not overwrite the input file."
                        )
                    gate.details["auto_fixable"] = True
                else:
                    gate.passed = False
                    gate.status = "blocked"
                    for issue in blocking_issues:
                        gate.reasons.append(f"[{issue['field']}] {issue['fix']}")
    except Exception as e:
        gate.warnings.append(f"Data readiness check failed (non-blocking): {e}")

    # 4. Tool-task compatibility
    try:
        from spatialomicsgym.agent.data_validation import UNKNOWN_CATEGORY, resolve_category

        expected_category = resolve_category(tool_name)
        # Map task_type to category names
        task_to_category = {
            "spatial_clustering": "spatial_clustering",
            "svg_detection": "svg_detection",
            "deconvolution": "deconvolution",
            "cell_communication": "cell_communication",
            "spatial_alignment": "spatial_alignment",
            "image_segmentation": "image_segmentation",
        }
        # An unknown category is the absence of an opinion, not a contradicting one. Warning of a
        # "possible mismatch" there reports our own silence as evidence against the caller's
        # task_type, and it fires for every tool the category map does not cover.
        if task_type in task_to_category and expected_category != UNKNOWN_CATEGORY:
            if expected_category != task_to_category[task_type]:
                gate.warnings.append(
                    f"Tool '{tool_name}' is category '{expected_category}' "
                    f"but task_type is '{task_type}' — possible mismatch"
                )
    except Exception:
        pass

    # 5. Image-dependency check
    platform = kwargs.get("platform", "")
    if tool_name in STRICT_IMAGE_TOOLS and platform in NO_IMAGE_PLATFORMS:
        gate.passed = False
        gate.status = "blocked"
        gate.reasons.append(f"Tool '{tool_name}' requires histology images but platform '{platform}' has none")
        return gate
    elif tool_name in IMAGE_DEPENDENT_TOOLS and platform in NO_IMAGE_PLATFORMS:
        gate.warnings.append(
            f"Tool '{tool_name}' may need histology images; platform '{platform}' has none — results may be degraded"
        )

    # 5b. CSV input format check
    csv_tools = {"precast_spatial_clustering", "run_bass", "run_iris", "run_spatialdecon", "spvc_svg_detection"}
    if tool_name in csv_tools and str(input_path).endswith(".h5ad"):
        gate.warnings.append(
            f"Tool '{tool_name}' requires CSV input but received h5ad. STCoscientist agent should call convert_h5ad_to_csv first."
        )

    # 6. Deconvolution reference check
    if task_type == "deconvolution" and sc_ref_path is None:
        gate.warnings.append("Deconvolution task without single-cell reference — tool may fail")

    # 7. Required environment variables
    TOOL_ENV_REQUIREMENTS = {
        "ucdeconvolve_base": ["UCD_TOKEN"],
    }
    required_env = TOOL_ENV_REQUIREMENTS.get(tool_name, [])
    missing_env = [v for v in required_env if not os.environ.get(v)]
    if missing_env:
        gate.passed = False
        gate.status = "blocked"
        gate.reasons.append(
            f"Tool '{tool_name}' requires environment variable(s): {missing_env}. Set them before running."
        )
        return gate

    return gate


def _path_names_the_tool(path: Path, root: Path, tool_name: str) -> bool:
    """Whether ``path`` carries ``tool_name`` in the part of itself that lies below ``root``.

    Recovery copies a candidate into the tool's output directory and lets it be scored as that
    tool's prediction, so a candidate has to be attributable to the tool before it is offered. The
    extra-search-dir half of the same candidate list has always required this -- it globs
    ``*{tool_name}*`` -- and the sibling half required nothing at all.

    Matched below ``root`` rather than on the filename, because a tool that creates its own
    directory beside the one it was handed leaves its name on the directory and not on the file:
    ``<parent>/<tool>_run/proportions.csv`` is the ordinary shape, and filename matching alone
    would leave exactly that case unrecoverable.

    The name must land on a non-alphanumeric boundary. Plain substring matching -- what the
    ``*{tool_name}*`` glob does -- gives ``miso`` a claim on ``miso2``'s output, and this repo's own
    run directories contain ``miso``/``miso2``, ``mist``/``mistyr``, ``spatialde``/``spatialdecon``.
    ``_``, ``-``, ``.`` and the path separator all count as boundaries, so ``istar_debug/`` still
    reads as ``istar``'s, which is the intent of that name.
    """
    try:
        rel = Path(path).relative_to(root).as_posix()
    except ValueError:  # not below root at all; fall back to what we can attribute
        rel = Path(path).name
    pattern = rf"(?<![a-z0-9]){re.escape(tool_name.casefold())}(?![a-z0-9])"
    return re.search(pattern, rel.casefold()) is not None


def _attempt_output_recovery(
    output_dir: str,
    candidates: list[str],
    tool_name: str,
    task_type: str,
    ground_truth_path: str | None = None,
) -> dict[str, Any] | None:
    """Try to recover misplaced outputs by copying candidates into output_dir."""
    import shutil

    out = Path(output_dir)
    recovered = []

    for candidate_path in candidates[:3]:  # Only try top 3
        src = Path(candidate_path)
        if not src.exists():
            continue
        dst = out / src.name
        if dst.exists():
            continue
        try:
            shutil.copy2(str(src), str(dst))
            recovered.append(str(dst))
        except Exception:
            continue

    if not recovered:
        return None

    # Re-inspect after recovery
    try:
        from spatialomicsgym.benchmarking.output_inspector import inspect_tool_output

        inspection = inspect_tool_output(
            tool_name=tool_name,
            task_type=task_type,
            output_dir=output_dir,
            ground_truth_path=ground_truth_path,
        )
        if inspection.is_ready():
            return {
                "recovered_files": recovered,
                "inspection": inspection.to_dict(),
                "prediction_file": inspection.prediction_file,
                "prediction_key": inspection.prediction_key,
                "prediction_type": inspection.prediction_type,
            }
    except Exception:
        pass

    # Recovery did not succeed, so the copies are not this run's output: take them back out. Left in
    # place they were permanent residents of the tool's directory -- another run's files, with that
    # run's mtime preserved by copy2 so the stale-file check could not see them -- for a rerun or the
    # post-analysis review to pick up as this tool's result (hunt 2026-09-30, u31-benchmarking-17).
    for path in recovered:
        try:
            Path(path).unlink()
        except OSError:
            pass
    return None


def post_execution_gate(
    tool_name: str,
    task_type: str,
    output_dir: str,
    ground_truth_path: str | None = None,
    expected_n_clusters: int | None = None,
) -> GateResult:
    """Validate outputs after tool execution, before evaluation.

    Checks:
    1. Output directory exists and has files
    2. Output inspection identifies prediction files
    3. Predictions are not empty or degenerate
    4. Output is in expected directory (not misplaced)
    5. Output is final (not intermediate)

    Non-blocking quality warnings (appended to gate.warnings; do not fail the gate):
    - clustering: K_pred != expected_n_clusters (registry metadata.n_clusters)
    - deconvolution: dominant cell-type mean fraction > 0.5
    - deconvolution: proportions don't sum to 1 (mean |sum-1| > 0.05)
    """
    gate = GateResult(gate_name="post_execution", passed=True, status="passed")

    out = Path(output_dir)

    # 1. Output directory exists
    if not out.exists():
        gate.passed = False
        gate.status = "blocked"
        gate.reasons.append(f"Output directory does not exist: {output_dir}")
        return gate

    # 2. Has files
    all_files = [f for f in out.rglob("*") if f.is_file() and f.stat().st_size > 0]
    if not all_files:
        gate.passed = False
        gate.status = "blocked"
        gate.reasons.append("No output files produced")
        return gate

    # This system's own post-analysis report is written *inside* the tool's output directory, so a
    # plain scan of that directory counts our summaries as the tool's output. Measured over this
    # repo's recorded runs, 79 of the 92 directories holding a report published an inflated
    # ``n_files`` (``card``: 10 files, 8 of them ours) and 13 held nothing else at all -- every
    # number the gate reported about the tool was a number about us. The rule for what is ours is
    # shared with the inspector rather than re-derived (see ``generated_report``).
    all_files, our_report = split_report_files(out, all_files)
    if our_report:
        gate.warnings.append(
            f"Ignored {len(our_report)} file(s) belonging to this system's own post-analysis report; "
            "they are derived summaries of the tool's output, not the tool's output"
        )

    gate.details["n_files"] = len(all_files)
    gate.details["file_types"] = list({f.suffix for f in all_files})

    # 2b. An h5ad that will not open is not an output -- but it is still the user's file.
    #
    # This used to ``f.unlink()`` it. The trigger is not "corrupt", it is *unreadable right now*:
    # h5py says "file signature not found" for a file another process is still writing, and
    # "truncated file: eof = N, stored_eof = M" for one whose worker was killed mid-write. Deleting
    # either destroys a running job's result, or the partial output that is the only evidence of how
    # far a crashed run got -- on a box where one tool run costs hours.
    #
    # Skipping reaches the identical verdict, so no benchmark outcome moves: the file did not count
    # towards ``all_files`` when it was deleted and does not count now. It simply still exists.
    unreadable = set()
    for f in all_files:
        if f.suffix == ".h5ad":
            try:
                # Nothing holds the handle in ``ad.read_h5ad(...).file.close()``, so a read that
                # succeeded and then raised on the attribute chain left an HDF5 lock on a file this
                # gate has just decided *is* the run's output -- the next thing to touch it is the
                # standardizer. ``read_h5ad_backed`` hands it back whichever way the block is left.
                from spatialomicsgym.utils.file_io import read_h5ad_backed

                with read_h5ad_backed(str(f)):
                    pass
            except Exception as e:
                if "signature" in str(e).lower() or "truncated" in str(e).lower():
                    gate.warnings.append(f"Unreadable h5ad, not counted as output: {f.name}")
                    unreadable.add(f)

    all_files = [f for f in all_files if f not in unreadable]
    if not all_files and not our_report:
        gate.passed = False
        gate.status = "blocked"
        gate.reasons.append("No readable output files")
        return gate
    # A directory holding *only* our report deliberately falls through instead of returning here.
    # It is the case that most needs the recovery search below -- the tool wrote nothing where it
    # was told to, so its real output is either elsewhere or absent -- and returning early would
    # take that search away from exactly the 13 runs that were relying on it before this filter
    # existed. Inspection reports the directory as empty and blocks; nothing is passed silently.

    # 3. Run output inspector
    # Set when the inspector read the tool's own prediction and found that it selects nothing. That
    # is the run's answer, so there is nothing misplaced to recover (hunt 2026-09-30,
    # u31-benchmarking-12): the search below would otherwise copy another directory's file in and
    # pass the gate on it. "The tool's own" means a file its profile declares as its prediction: an
    # undeclared side table that selects nothing says nothing about where the declared file went, so
    # the search for it still runs (u31-benchmarking-12 review).
    answered_nothing = False
    try:
        from spatialomicsgym.benchmarking.output_inspector import _is_declared_prediction, inspect_tool_output
        from spatialomicsgym.benchmarking.tool_output_registry import get_profile

        inspection = inspect_tool_output(
            tool_name=tool_name,
            task_type=task_type,
            output_dir=output_dir,
            ground_truth_path=ground_truth_path,
        )
        gate.details["inspection"] = inspection.to_dict()
        answered_nothing = (
            inspection.status == "empty"
            and bool(inspection.prediction_file)
            and _is_declared_prediction(Path(inspection.prediction_file), get_profile(tool_name))
        )

        if not inspection.is_ready():
            gate.passed = False
            gate.status = "blocked"
            for err in inspection.errors:
                gate.reasons.append(f"Output inspection: {err}")
            for warn in inspection.warnings:
                gate.warnings.append(f"Output inspection: {warn}")
        else:
            gate.details["prediction_file"] = inspection.prediction_file
            gate.details["prediction_key"] = inspection.prediction_key
            gate.details["prediction_type"] = inspection.prediction_type
            gate.details["n_predictions"] = inspection.n_predictions

            # Check for degenerate predictions
            if task_type == "spatial_clustering" and inspection.n_clusters is not None:
                if inspection.n_clusters <= 1:
                    gate.passed = False
                    gate.status = "blocked"
                    gate.reasons.append(f"Degenerate clustering: only {inspection.n_clusters} cluster(s) found")

            if task_type == "spatial_clustering" and inspection.cluster_labels is not None:
                # Check for all-identical labels
                unique_labels = {str(l) for l in inspection.cluster_labels if str(l) not in ("nan", "None", "")}
                if len(unique_labels) == 1:
                    gate.passed = False
                    gate.status = "blocked"
                    gate.reasons.append(f"All spots assigned to single cluster label: {unique_labels}")

            if task_type == "svg_detection" and inspection.n_svg_genes is not None:
                if inspection.n_svg_genes == 0:
                    gate.passed = False
                    gate.status = "blocked"
                    gate.reasons.append("No SVG genes detected")

            if task_type == "deconvolution" and inspection.proportions_shape is not None:
                if inspection.proportions_shape[1] <= 1:
                    gate.passed = False
                    gate.status = "blocked"
                    gate.reasons.append(
                        f"Degenerate deconvolution: only {inspection.proportions_shape[1]} cell type(s)"
                    )

            if task_type == "deconvolution" and inspection.proportions_shape is not None:
                # Check for all-uniform proportions (no real deconvolution)
                if inspection.proportions_shape[0] > 0 and inspection.proportions_shape[1] > 1:
                    try:
                        import pandas as pd

                        if inspection.prediction_file:
                            # Re-read the file the inspector picked, the way the inspector read
                            # it. Hardcoding a comma and column 0 here made this gate disagree
                            # with the matrix it was judging: a tab-separated prediction (STRIDE,
                            # stereoscope) collapsed to one object column, leaving no numbers to
                            # measure, so the uniform-proportions check below became vacuous and
                            # a degenerate run passed -- while the row sums, now 0, warned that a
                            # normalized matrix "doesn't sum to 1" at |row_sum - 1| = 1.000. An
                            # R-convention matrix with the IDs last (CARD) lost one cell type to
                            # the index and warned by exactly that column's mean.
                            from spatialomicsgym.benchmarking.output_inspector import (
                                _proportions_index_col,
                                _tabular_sep,
                            )

                            pred_path = Path(inspection.prediction_file)
                            sep = _tabular_sep(pred_path)
                            pred = pd.read_csv(
                                inspection.prediction_file,
                                index_col=_proportions_index_col(pred_path, sep),
                                sep=sep,
                            )
                            # Spots on the rows, as the inspector reported them (hunt 2026-09-30,
                            # u31-benchmarking-9): on a cell type x spot table these per-spot checks
                            # were run per cell type.
                            from spatialomicsgym.benchmarking.output_standardizer import _orient_proportions

                            numeric, _ = _orient_proportions(pred.select_dtypes(include="number"), wide_only=True)
                            if numeric.shape[0] > 10:
                                row_std = numeric.std(axis=1)
                                if (row_std < 1e-6).mean() > 0.9:
                                    gate.passed = False
                                    gate.status = "blocked"
                                    gate.reasons.append(
                                        "Degenerate deconvolution: >90% of spots have uniform proportions"
                                    )
                                # Non-blocking quality warnings
                                col_means = numeric.mean(axis=0)
                                if len(col_means):
                                    max_frac = float(col_means.max())
                                    if max_frac > 0.5:
                                        dom = col_means.idxmax()
                                        gate.warnings.append(
                                            f"Deconvolution dominated by single cell type "
                                            f"'{dom}' (mean fraction {max_frac:.2f} > 0.5); "
                                            f"possible collapse onto majority class."
                                        )
                                row_sums = numeric.sum(axis=1)
                                sum_violation = float((row_sums - 1.0).abs().mean())
                                if sum_violation > 0.05:
                                    gate.warnings.append(
                                        f"Deconvolution proportions don't sum to 1 "
                                        f"(mean |row_sum - 1| = {sum_violation:.3f} > 0.05); "
                                        f"tool may have output absolute counts/scores instead of proportions."
                                    )
                    except Exception:
                        pass

            # Non-blocking K mismatch warning (clustering)
            if (
                task_type == "spatial_clustering"
                and inspection.n_clusters is not None
                and expected_n_clusters is not None
                and inspection.n_clusters != expected_n_clusters
            ):
                gate.warnings.append(
                    f"K mismatch: predicted {inspection.n_clusters} cluster(s), "
                    f"expected {expected_n_clusters} (registry metadata.n_clusters). "
                    f"Tool may have collapsed clusters or used wrong K."
                )
    except Exception as e:
        gate.passed = False
        gate.status = "error"
        gate.reasons.append(f"Output inspection failed: {e}")

    # 4. Attempt output recovery if gate blocked
    if not gate.passed and gate.status == "blocked" and not answered_nothing:
        parent = out.parent
        if parent.exists():
            # ``is_relative_to`` and not ``str.startswith``: a string prefix makes ``<out>_run/`` and
            # ``<out>2/`` read as being *inside* ``out``, so they were dropped from the search --
            # and a directory named after the one the tool was handed is the single most likely
            # place for its misplaced output to be. This repo's own run directories carry the shape
            # (``istar``/``istar_debug``, ``miso``/``miso2``), so the exclusion that exists to avoid
            # rediscovering the tool's own directory was instead hiding it.
            sibling_files = [
                f for f in parent.rglob("*") if f.is_file() and f.stat().st_size > 0 and not f.is_relative_to(out)
            ]
            # Before the suffix filter, not after: the marker that identifies a post-analysis report
            # is its ``manifest.json``, which ``relevant`` would already have discarded, leaving the
            # report's ``.csv`` tables looking like ordinary misplaced output. This is the path with
            # teeth -- ``_attempt_output_recovery`` does not merely *name* a candidate, it copies it
            # into the tool's output directory and re-inspects, and on success clears the block and
            # marks the gate passed. A neighbouring run's ``mean_composition.csv`` could therefore be
            # copied in and adopted as this tool's prediction. The copy is also why the inspector's
            # own filter is not enough here: it recognises our files by the manifest beside them, and
            # copying one out of its directory launders it. All 92 recorded runs holding a report had
            # its files in this candidate list.
            sibling_files, _ = split_report_files(parent, sibling_files)
            # Whose output this is has to be asked, and until now the sibling half of this list
            # never asked it. Every recorded run directory in this repo sits under one shared
            # parent, so this scan reached all 92 of them: gating ``bass`` offered 272 candidates,
            # none of whose paths mention bass, and the three that would have been copied in belong
            # to ``scanpy_spatial``. A blocked run could therefore be marked passed with
            # ``prediction_file`` pointing at another tool's prediction.
            relevant = [
                f
                for f in sibling_files
                if f.suffix in (".h5ad", ".csv", ".tsv") and _path_names_the_tool(f, parent, tool_name)
            ]
            # Also search common MCP output locations
            additional_search_dirs = _output_recovery_search_dirs()
            for search_dir in additional_search_dirs:
                if search_dir.exists() and search_dir != out:
                    extra_files = [
                        f
                        for f in search_dir.glob(f"*{tool_name}*")
                        if f.is_file() and f.suffix in (".h5ad", ".csv", ".tsv") and f.stat().st_size > 0
                    ]
                    if extra_files:
                        relevant.extend(extra_files[:3])

            if relevant:
                gate.warnings.append(
                    f"Found {len(relevant)} potential output file(s) outside expected dir: "
                    + ", ".join(f.name for f in relevant[:5])
                )
                gate.details["misplaced_candidates"] = [str(f) for f in relevant[:10]]

                # Attempt recovery
                recovery = _attempt_output_recovery(
                    output_dir, [str(f) for f in relevant], tool_name, task_type, ground_truth_path=ground_truth_path
                )
                if recovery:
                    gate.passed = True
                    gate.status = "passed"
                    gate.reasons.clear()
                    gate.warnings.append(f"Output recovered from misplaced location: {recovery['recovered_files']}")
                    gate.details.update(recovery)

    return gate


def pre_evaluation_gate(
    task_type: str,
    prediction_file: str,
    ground_truth_path: str,
    ground_truth_key: str = "",
    predicted_key: str = "",
    predicted_genes: list[str] | None = None,
) -> GateResult:
    """Validate that evaluation inputs are correct and complete.

    Checks:
    1. Prediction file exists and is readable
    2. Ground truth file exists and is readable
    3. Schema validation: prediction format matches task type
    4. For clustering: predicted and GT columns exist, have >1 unique value
    5. For SVG: gene list is non-empty, genes exist in expression matrix
    6. For deconvolution: proportions matrix is well-formed, cell types align
    7. Label/gene/cell-type alignment between prediction and ground truth
    """
    gate = GateResult(gate_name="pre_evaluation", passed=True, status="passed")

    # 1. Files exist
    pred_path = Path(prediction_file)
    gt_path = Path(ground_truth_path)

    if not pred_path.exists():
        gate.passed = False
        gate.status = "blocked"
        gate.reasons.append(f"Prediction file not found: {prediction_file}")
        return gate

    if not gt_path.exists():
        gate.passed = False
        gate.status = "blocked"
        gate.reasons.append(f"Ground truth file not found: {ground_truth_path}")
        return gate

    # 2. Task-specific validation
    if task_type == "spatial_clustering":
        gate = _validate_clustering_inputs(gate, pred_path, gt_path, ground_truth_key, predicted_key)
    elif task_type == "svg_detection":
        gate = _validate_svg_inputs(gate, pred_path, gt_path, predicted_genes)
    elif task_type == "deconvolution":
        gate = _validate_deconvolution_inputs(gate, pred_path, gt_path, ground_truth_key, predicted_key)
    else:
        gate.warnings.append(f"No pre-evaluation validation for task type: {task_type}")

    return gate


def _validate_clustering_inputs(
    gate: GateResult,
    pred_path: Path,
    gt_path: Path,
    ground_truth_key: str,
    predicted_key: str,
) -> GateResult:
    """Validate clustering evaluation inputs (supports h5ad, CSV, TSV predictions)."""
    try:
        import anndata as ad
        import pandas as pd

        # ── Read prediction file based on format ──
        pred_values = None
        pred_index = None
        n_obs = 0

        if pred_path.suffix == ".h5ad":
            adata = ad.read_h5ad(str(pred_path))
            obs_cols = list(adata.obs.columns)
            gate.details["prediction_obs_columns"] = obs_cols[:30]
            gate.details["prediction_shape"] = list(adata.shape)
            n_obs = adata.n_obs
            pred_index = adata.obs.index.astype(str)
            if predicted_key and predicted_key in obs_cols:
                pred_values = adata.obs[predicted_key].dropna().astype(str)

        elif pred_path.suffix in (".csv", ".tsv"):
            # The same one-sniffer rule the other two readers in this file already follow (:479,
            # :869). For the two suffixes this branch admits it is a provable no-op --
            # _tabular_sep trusts ".csv" as a comma and ".tsv" as a tab -- so it exists to keep
            # the rule in one place rather than to change any read.
            from spatialomicsgym.benchmarking.output_inspector import _tabular_sep

            sep = _tabular_sep(pred_path)
            df = pd.read_csv(str(pred_path), sep=sep)
            gate.details["prediction_format"] = pred_path.suffix
            gate.details["prediction_shape"] = list(df.shape)

            # Plain TSV: one domain label per line, no columns
            if pred_path.suffix == ".tsv" and df.shape[1] <= 1:
                pred_values = df.iloc[:, 0].dropna().astype(str)
                pred_index = pd.RangeIndex(len(df)).astype(str)
            else:
                obs_cols = list(df.columns)
                gate.details["prediction_columns"] = obs_cols[:20]
                # Find predicted column
                if predicted_key and predicted_key in obs_cols:
                    pred_values = df[predicted_key].dropna().astype(str)
                else:
                    cluster_kw = ["cluster", "domain", "leiden", "louvain", "pred", "label", "seurat"]
                    for c in obs_cols:
                        if any(kw in c.lower() for kw in cluster_kw):
                            pred_values = df[c].dropna().astype(str)
                            predicted_key = c
                            gate.warnings.append(f"Using column '{c}' as predicted labels")
                            break
                # Extract spot index
                for sc in ["spot", "barcode", "spot_id", "cell_id"]:
                    if sc in df.columns:
                        pred_index = df[sc].astype(str)
                        break
                if pred_index is None:
                    pred_index = df.index.astype(str)
            n_obs = len(df)

        else:
            gate.warnings.append(f"Unknown prediction format: {pred_path.suffix}")
            return gate

        # ── Validate predicted values (same checks for all formats) ──
        if pred_values is not None:
            n_unique = int(pred_values.nunique())
            gate.details["n_predicted_clusters"] = n_unique
            if len(pred_values) == 0:
                gate.passed = False
                gate.status = "blocked"
                gate.reasons.append(f"Predicted column '{predicted_key}' is all NaN")
                return gate
            if n_unique <= 1:
                gate.passed = False
                gate.status = "blocked"
                gate.reasons.append(
                    f"Predicted column '{predicted_key}' has only {n_unique} unique value(s) — degenerate"
                )
                return gate

        # ── Check ground truth (always h5ad) ──
        gt_adata = ad.read_h5ad(str(gt_path))
        gt_cols = list(gt_adata.obs.columns)
        if ground_truth_key and ground_truth_key not in gt_cols:
            found = False
            for c in gt_cols:
                if c.lower() == ground_truth_key.lower():
                    gate.warnings.append(
                        f"Ground truth key '{ground_truth_key}' not found exactly, "
                        f"but case-insensitive match '{c}' exists"
                    )
                    found = True
                    break
            if not found:
                gate.passed = False
                gate.status = "blocked"
                gate.reasons.append(f"Ground truth column '{ground_truth_key}' not found. Available: {gt_cols[:15]}")
        elif ground_truth_key:
            gt_vals = gt_adata.obs[ground_truth_key].dropna()
            n_gt_unique = gt_vals.nunique()
            gate.details["n_gt_clusters"] = int(n_gt_unique)
            if n_gt_unique <= 1:
                gate.passed = False
                gate.status = "blocked"
                gate.reasons.append(f"Ground truth column '{ground_truth_key}' has only {n_gt_unique} unique value(s)")

        # ── Check index alignment (for h5ad predictions vs h5ad GT) ──
        if pred_path.suffix == ".h5ad" and str(gt_path) != str(pred_path):
            common = pd.Index(pred_index).intersection(gt_adata.obs.index.astype(str))
            gate.details["n_common_indices"] = len(common)
            if len(common) == 0:
                gate.passed = False
                gate.status = "blocked"
                gate.reasons.append("No overlapping cell/spot indices between prediction and ground truth")
            elif len(common) < min(n_obs, gt_adata.n_obs) * 0.5:
                gate.warnings.append(
                    f"Only {len(common)}/{min(n_obs, gt_adata.n_obs)} indices overlap — partial alignment"
                )
    except Exception as e:
        gate.passed = False
        gate.status = "error"
        gate.reasons.append(f"Clustering validation failed: {e}")

    return gate


def _validate_svg_inputs(
    gate: GateResult,
    pred_path: Path,
    gt_path: Path,
    predicted_genes: list[str] | None,
) -> GateResult:
    """Validate SVG evaluation inputs."""
    # Check predicted genes
    if not predicted_genes:
        gate.passed = False
        gate.status = "blocked"
        gate.reasons.append("No predicted genes provided for SVG evaluation")
        return gate

    # Filter blanks
    clean_genes = [g for g in predicted_genes if g.strip() and not g.startswith("Blank")]
    gate.details["n_predicted_genes"] = len(clean_genes)

    if not clean_genes:
        gate.passed = False
        gate.status = "blocked"
        gate.reasons.append("All predicted genes are blank probes or empty")
        return gate

    # Check ground truth
    if gt_path.suffix == ".csv":
        try:
            import pandas as pd

            gt_df = pd.read_csv(str(gt_path))
            gt_genes = {str(g) for g in gt_df.iloc[:, 0].tolist()}
            gate.details["n_gt_genes"] = len(gt_genes)

            overlap = set(clean_genes) & gt_genes
            gate.details["n_overlap_genes"] = len(overlap)

            if len(overlap) == 0:
                gate.warnings.append(
                    "Zero overlap between predicted and ground truth gene sets — possible gene identifier mismatch"
                )
        except Exception as e:
            gate.warnings.append(f"Could not read ground truth CSV: {e}")

    return gate


def _validate_deconvolution_inputs(
    gate: GateResult,
    pred_path: Path,
    gt_path: Path,
    ground_truth_key: str,
    predicted_key: str = "",
) -> GateResult:
    """Validate deconvolution evaluation inputs (supports CSV and h5ad predictions)."""
    try:
        import pandas as pd

        # Read prediction based on format
        if pred_path.suffix == ".h5ad":
            import anndata as ad

            from spatialomicsgym.benchmarking.output_inspector import _abundance_obsm_key

            adata = ad.read_h5ad(str(pred_path))
            # The obsm key the inspector chose, as the standardizer now reads it -- the alphabetical
            # keyword scan this replaced judged cell2location's means, not the q05 that is scored
            # (hunt 2026-09-30, u31-benchmarking-2).
            prop_key = _abundance_obsm_key(adata.obsm, predicted_key)
            if prop_key:
                import numpy as np

                mat = adata.obsm[prop_key]
                if hasattr(mat, "columns"):
                    pred = pd.DataFrame(mat, index=adata.obs_names)
                else:
                    mat_arr = mat.toarray() if hasattr(mat, "toarray") else np.array(mat)
                    pred = pd.DataFrame(mat_arr, index=adata.obs_names)
                gate.details["prediction_obsm_key"] = prop_key
            else:
                gate.warnings.append("No deconvolution proportion matrix found in obsm")
                return gate
        else:
            # Not every prediction is comma-separated: STRIDE's authoritative output is the
            # tab-separated *_spot_celltype_frac.txt (tool_output_registry.py:942-945,
            # stride_worker.py:99). Read with the default comma it collapses to one object
            # column, and the numeric-column check below blocks the run with a reason that
            # blames the tool. _tabular_sep keeps "," for .csv and "\t" for .tsv, so files
            # that already pass this gate are read identically.
            # Nor is the spot column always the first one: CARD writes the IDs LAST, and at
            # index_col=0 the first cell type becomes the index while the barcode column joins
            # pred.columns -- so the cell-type alignment this gate exists to report loses a real
            # type and counts a fake one, on a prediction that named every type correctly.
            from spatialomicsgym.benchmarking.output_inspector import _proportions_index_col, _tabular_sep
            from spatialomicsgym.benchmarking.output_standardizer import _orient_proportions

            sep = _tabular_sep(pred_path)
            pred = pd.read_csv(str(pred_path), index_col=_proportions_index_col(pred_path, sep), sep=sep)
            oriented, transposed = _orient_proportions(pred.select_dtypes(include="number"), wide_only=True)
            if transposed:
                pred = oriented  # cell types on the columns, so the alignment below compares them
        # The names the standardizer will score, so the alignment reported below is about them.
        from spatialomicsgym.benchmarking.output_standardizer import _strip_cell2location_prefix

        cleaned = _strip_cell2location_prefix(pred.columns)
        if cleaned is not None:
            pred.columns = cleaned
        gate.details["prediction_shape"] = list(pred.shape)
        gate.details["prediction_columns"] = list(pred.columns)[:20]

        # Check proportions are valid
        numeric_cols = pred.select_dtypes(include="number")
        if numeric_cols.shape[1] < 2:
            gate.passed = False
            gate.status = "blocked"
            gate.reasons.append(
                f"Prediction has only {numeric_cols.shape[1]} numeric column(s) — need at least 2 cell types"
            )
            return gate

        # Check for NaN or negative values
        if numeric_cols.isna().all().any():
            gate.warnings.append("Some cell type columns are entirely NaN")

        if (numeric_cols < 0).any().any():
            gate.warnings.append("Negative values in proportion matrix")

        # Read ground truth
        import anndata as ad

        gt_adata = ad.read_h5ad(str(gt_path))

        # Find cell type column
        resolved_key = ground_truth_key if ground_truth_key in gt_adata.obs.columns else None
        if resolved_key is None:
            for candidate in [
                "cell_type",
                "CellType",
                "celltype",
                "subclass",
                "subclass_label",
                "class_label",
                "cluster_label",
                "annotation",
                "Cell_class",
                "author_cell_type",
                "label",
            ]:
                if candidate in gt_adata.obs.columns:
                    resolved_key = candidate
                    break

        if resolved_key is None:
            gate.passed = False
            gate.status = "blocked"
            gate.reasons.append(
                f"Ground truth cell type column not found. "
                f"Tried '{ground_truth_key}'. Available: {list(gt_adata.obs.columns)[:15]}"
            )
            return gate

        # Check cell type alignment
        gt_types = {str(t) for t in gt_adata.obs[resolved_key].dropna().unique()}
        pred_types = {str(c) for c in pred.columns}

        gate.details["n_gt_cell_types"] = len(gt_types)
        gate.details["n_pred_cell_types"] = len(pred_types)

        overlap = gt_types & pred_types
        gate.details["n_cell_type_overlap"] = len(overlap)

        if len(overlap) == 0:
            gate.warnings.append(
                f"No cell type overlap between prediction columns and ground truth labels. "
                f"GT types: {list(gt_types)[:5]}, Pred columns: {list(pred_types)[:5]}"
            )
    except Exception as e:
        gate.passed = False
        gate.status = "error"
        gate.reasons.append(f"Deconvolution validation failed: {e}")

    return gate
