"""Benchmark evaluation - connects tool outputs to eval_metrics.

Bridges the gap between the benchmark framework (benchmarks/) and the
metric functions in tools/eval_metrics.py, providing a simplified API
for evaluating spatial clustering, SVG detection, and deconvolution
tool outputs against ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add tools/ to path for eval_metrics import
_TOOLS_DIR = str(Path(__file__).resolve().parent.parent.parent / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

# Ordered list of candidate prediction column names for clustering.
# More specific names first, generic names last.
_CLUSTER_PRED_CANDIDATES = [
    "spatial_domain",
    "spatial_domains",
    "scanpy_spatial_domain",
    "domain",
    "domains",
    "pred",
    "prediction",
    "predicted",
    "cell_type_pred",
    "predicted_cell_type",
    "predicted_domain",
    "cluster",
    "clusters",
    "mclust",
    "leiden",
    "louvain",
    "phenotype",
    "class",
    "label",
    "type",
    "subtype",
    "graphst_cluster",
    "stagate_domain",
    "cellcharter_domain",
    "cluster_cellcharter",
    "sedr_cluster",
    "sedr_dec_cluster",
    "deepst_domain",
    "DeepST_refine_domain",
    "DeepST_domain",
    "miso_cluster",
    "spaceflow_domain",
    "precast_cluster",
    "prost_domain",
    "spicemix_factor",
    "splane_cluster",
    "spatialprompt_cluster",
    "bass_cluster",
    "iris_cluster",
    "stage_cluster",
    # Generic Seurat clusterings last. A dataset's own precomputed clustering is one of these (the
    # Slide-seqV2 inputs carry the authors' ``SCT_snn_res.0.5``), and a worker that copies the input
    # obs into its output carries it forward. It used to sit ahead of every tool-specific key, so a
    # search with no ``predicted_key`` scored the authors' clustering whatever the tool produced
    # (hunt 2026-09-30, u33b-bench-scoring-2).
    "seurat_clusters",
    "RNA_snn_res.0.8",
    "SCT_snn_res.0.8",
    "RNA_snn_res.0.5",
    "SCT_snn_res.0.5",
]

# Candidate ground truth column names for clustering.
_CLUSTER_GT_CANDIDATES = [
    "cell_type",
    "Cell_class",
    "layer_guess_reordered_short",
    "author_cell_type",
    "subclass",
    "annotation",
    "celltype",
    "CellType",
    "cluster",
    "ground_truth",
    "label",
    "region",
    "Cluster",
    "class_label",
    "layer_guess_reordered",
    "Layer",
]


def _read_proportions(path: str) -> Any:
    """Read a predicted proportions table the way the rest of the system reads one.

    Every read below used to hardcode ``index_col=0``. That is the Python convention --
    ``DataFrame.to_csv`` writes the row labels into an unnamed first field -- and it is not R's.
    ``write.csv(row.names = FALSE)`` writes no index field at all and puts the spot identifier in
    an ordinary named column, conventionally last. Reading such a file at ``index_col=0`` does not
    fail; it succeeds one column off, taking the first CELL TYPE as the index and admitting the
    barcode column to ``pred.columns`` as a cell type made of strings. Downstream that costs the
    eaten cell type (``reindex(columns=pred.columns)`` drops it from the ground truth), costs spot
    alignment (the index holds proportions, so it shares no ID with the ground truth), and then
    raises ``could not convert string to float`` on the barcodes. On a two-cell-type table the
    pre-evaluation gate gets there first and blocks the run for having one numeric column.

    CARD, SPOTlight and IRIS all write this way: ``card_proportions.csv`` for ``slideseqv2_aorta``
    has 8 fields, ``fibroblast`` first and ``spot`` last. No recorded score is affected -- those
    runs were scored through the standardizer's copy, which carries ``spot_id`` first -- but the
    raw fallthrough in ``evaluate_tool_output`` reaches these reads directly.

    Defined once and shared, because three sites asking the same question separately is how they
    drift apart. The question itself is already answered by ``_proportions_index_col``, added to
    the completeness checker for this defect; the output_standardizer searches the same four names
    inline. The separator comes from the same module so that the header the identifier is looked
    up in is parsed the same way as the body it must exist in.
    """
    import pandas as pd

    from spatialomicsgym.benchmarking.output_inspector import _proportions_index_col, _tabular_sep

    target = Path(path)
    sep = _tabular_sep(target)
    return pd.read_csv(path, index_col=_proportions_index_col(target, sep), sep=sep)


def validate_before_evaluation(
    task_type: str,
    output_path: str,
    ground_truth_path: str,
    ground_truth_key: str = "",
    predicted_key: str = "",
    predicted_genes: list[str] | None = None,
) -> dict[str, Any]:
    """Pre-evaluation validation gate. Must pass before computing metrics.

    Returns dict with:
        valid: bool — whether evaluation should proceed
        status: str — "valid", "blocked", "warning"
        reasons: list[str] — blocking reasons
        warnings: list[str] — non-blocking warnings
        details: dict — diagnostic info
    """
    reasons: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {"task_type": task_type}

    # ── Common: check output file ──
    output = Path(output_path)
    if not output.exists():
        reasons.append(f"Output file does not exist: {output_path}")
    elif not output.is_file():
        reasons.append(f"Output path is not a file: {output_path}")
    else:
        try:
            output.read_bytes()[:1]
        except PermissionError:
            reasons.append(f"Output file is not readable: {output_path}")

    # ── Common: check ground truth file ──
    gt = Path(ground_truth_path)
    if not gt.exists():
        reasons.append(f"Ground truth file does not exist: {ground_truth_path}")
    elif not gt.is_file():
        reasons.append(f"Ground truth path is not a file: {ground_truth_path}")
    else:
        try:
            gt.read_bytes()[:1]
        except PermissionError:
            reasons.append(f"Ground truth file is not readable: {ground_truth_path}")

    # ── Task-specific checks (only if files exist) ──
    if task_type == "spatial_clustering" and not reasons:
        out_suffix = Path(output_path).suffix.lower()
        if out_suffix in {".csv", ".tsv"}:
            # CSV-only clustering output (e.g., precast writes a metadata.csv with
            # cluster labels directly instead of an h5ad). Validate the CSV;
            # opening it as h5ad would raise a misleading "file signature" error.
            try:
                import pandas as pd

                sep = "\t" if out_suffix == ".tsv" else ","
                df = pd.read_csv(output_path, sep=sep)
                csv_cols = list(df.columns)
                details["csv_columns"] = csv_cols
                pred_col = _find_column(csv_cols, _CLUSTER_PRED_CANDIDATES, preferred=predicted_key)
                if pred_col is None:
                    reasons.append(f"No predicted cluster column found in CSV. Available columns: {csv_cols}")
                else:
                    details["predicted_column"] = pred_col
                    n_unique = int(df[pred_col].dropna().nunique())
                    details["predicted_n_unique"] = n_unique
                    if n_unique <= 1:
                        reasons.append(
                            f"Predicted column '{pred_col}' has {n_unique} unique non-NaN value(s); "
                            f"need >1 for clustering evaluation"
                        )
            except Exception as exc:
                reasons.append(f"Failed to read CSV for clustering validation: {exc}")

            # Ground truth is still h5ad — validate separately
            try:
                import anndata as ad

                gt_adata = ad.read_h5ad(ground_truth_path)
                gt_obs_cols = list(gt_adata.obs.columns)
                # Same rule as the scorers: a named key that is absent blocks (hunt 2026-09-30,
                # u33b-bench-scoring-4).
                gt_col, gt_error = _resolve_cluster_gt_column(gt_obs_cols, ground_truth_key)
                if gt_col is None:
                    reasons.append(gt_error)
                else:
                    details["ground_truth_column"] = gt_col
            except Exception as exc:
                reasons.append(f"Failed to read ground-truth h5ad for clustering validation: {exc}")
        else:
            try:
                import anndata as ad

                adata = ad.read_h5ad(output_path)
                obs_cols = list(adata.obs.columns)
                details["obs_columns"] = obs_cols

                # Check predicted column
                pred_col = _find_column(obs_cols, _CLUSTER_PRED_CANDIDATES, preferred=predicted_key)
                if pred_col is None:
                    reasons.append(f"No predicted cluster column found in output. Available obs columns: {obs_cols}")
                else:
                    details["predicted_column"] = pred_col
                    pred_values = adata.obs[pred_col].dropna()
                    n_unique = int(pred_values.nunique())
                    details["predicted_n_unique"] = n_unique
                    if n_unique <= 1:
                        reasons.append(
                            f"Predicted column '{pred_col}' has {n_unique} unique non-NaN value(s); "
                            f"need >1 for clustering evaluation"
                        )

                # Check ground truth column
                gt_adata = ad.read_h5ad(ground_truth_path) if ground_truth_path != output_path else adata
                gt_obs_cols = list(gt_adata.obs.columns)
                gt_col, gt_error = _resolve_cluster_gt_column(gt_obs_cols, ground_truth_key)
                if gt_col is None:
                    reasons.append(gt_error)
                else:
                    details["ground_truth_column"] = gt_col
            except Exception as exc:
                reasons.append(f"Failed to read h5ad for clustering validation: {exc}")

    elif task_type == "svg_detection" and not reasons:
        # Verify predicted_genes is non-empty after filtering blanks
        genes = predicted_genes or []
        clean = [g.strip() for g in genes if g.strip() and not g.startswith("Blank")]
        details["raw_gene_count"] = len(genes)
        details["clean_gene_count"] = len(clean)
        if not clean:
            reasons.append("predicted_genes is empty after filtering blanks and whitespace")

    elif task_type == "deconvolution" and not reasons:
        try:
            pred = _read_proportions(output_path)
            numeric_cols = pred.select_dtypes(include="number").columns.tolist()
            details["csv_shape"] = list(pred.shape)
            details["numeric_columns"] = len(numeric_cols)
            if len(numeric_cols) < 2:
                reasons.append(
                    f"Predicted proportions CSV has {len(numeric_cols)} numeric column(s); "
                    f"need >=2 for deconvolution evaluation"
                )
        except Exception as exc:
            reasons.append(f"Failed to read predicted CSV: {exc}")

        try:
            import anndata as ad

            gt_adata = ad.read_h5ad(ground_truth_path)
            gt_obs_cols = list(gt_adata.obs.columns)
            details["gt_obs_columns"] = gt_obs_cols
            # Check cell type column
            resolved = None
            if ground_truth_key and ground_truth_key in gt_obs_cols:
                resolved = ground_truth_key
            if resolved is None:
                for candidate in [
                    "CellType",
                    "cell_type",
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
                    if candidate in gt_obs_cols:
                        resolved = candidate
                        break
            if resolved is None:
                reasons.append(
                    f"Cell type column not found in ground truth h5ad. "
                    f"Tried '{ground_truth_key}' and common candidates. Available: {gt_obs_cols}"
                )
            else:
                details["celltype_column"] = resolved
        except Exception as exc:
            reasons.append(f"Failed to read ground truth h5ad for deconvolution validation: {exc}")

    # ── Build result ──
    if reasons:
        status = "blocked"
        valid = False
    elif warnings:
        status = "warning"
        valid = True
    else:
        status = "valid"
        valid = True

    return {
        "valid": valid,
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "details": details,
    }


def inspect_h5ad(path: str) -> dict[str, Any]:
    """Return a compact schema report for debugging benchmark evaluation."""
    from spatialomicsgym.utils.file_io import read_h5ad_backed

    # The context manager releases the HDF5 handle before returning. Dropping the local is not
    # enough: an input with a .raw slot is cyclic, so the file would stay locked until the garbage
    # collector next ran, and whatever rewrote that path in the meantime would fail with errno 11.
    with read_h5ad_backed(path) as adata:
        obs_cols = list(adata.obs.columns)
        report: dict[str, Any] = {
            "path": path,
            "shape": list(adata.shape),
            "obs_columns": obs_cols,
            "obsm_keys": list(adata.obsm.keys()),
            "uns_keys": list(adata.uns.keys())[:15],
            "layers": list(adata.layers.keys()) if adata.layers else [],
        }
        # Summarize candidate clustering columns
        candidates = {}
        for c in obs_cols:
            if any(
                kw in c.lower()
                for kw in ["cluster", "domain", "leiden", "louvain", "pred", "label", "class", "spatial"]
            ):
                try:
                    candidates[c] = {"n_unique": int(adata.obs[c].nunique()), "dtype": str(adata.obs[c].dtype)}
                except Exception:
                    candidates[c] = {"n_unique": "?", "dtype": str(adata.obs[c].dtype)}
        report["candidate_prediction_columns"] = candidates
        return report


def _find_column(obs_columns: list[str], candidates: list[str], preferred: str = "") -> str | None:
    """Find the best matching column from candidates list.

    Search order:
    1. The explicitly preferred name (if provided and present) -- registry-specified key
    2. Exact match from candidates list (preserving priority order)
    3. Case-insensitive match from candidates list
    Emits a diagnostic message to stderr when falling back to a non-preferred match.
    """
    col_set = set(obs_columns)
    col_lower_map = {c.lower(): c for c in obs_columns}

    # 1. Preferred name exact match (registry-specified key)
    if preferred and preferred in col_set:
        return preferred

    # 2. Exact match in priority order
    for cand in candidates:
        if cand in col_set:
            if preferred:
                print(
                    f"[evaluator] preferred column '{preferred}' not found; falling back to exact match '{cand}'",
                    file=sys.stderr,
                )
            return cand

    # 3. Case-insensitive match
    for cand in candidates:
        if cand.lower() in col_lower_map:
            matched = col_lower_map[cand.lower()]
            if preferred:
                print(
                    f"[evaluator] preferred column '{preferred}' not found; "
                    f"falling back to case-insensitive match '{matched}' (from candidate '{cand}')",
                    file=sys.stderr,
                )
            return matched

    return None


def _coverage_fields(
    n_scored: int, n_ground_truth: int, alignment: str = "spot_id", n_aligned: int | None = None
) -> dict[str, Any]:
    """How much of the ground truth a score was actually computed on.

    Every evaluator below aligns the prediction to the ground truth by intersecting their spot IDs
    and scoring what is left. That is the right thing to do -- a tool that ran its own QC has no
    prediction for the spots it dropped -- but each of them recorded at most the numerator, so no
    single metrics file could distinguish a tool scored on the whole slide from one scored on part
    of it, and the results tables put the two side by side.

    Aligning is not the same as scoring, and the gap between them is what this signature exists to
    keep straight. After the intersection each caller drops more rows -- the clustering paths drop
    every pair where either label is NaN, the deconvolution paths drop every ground-truth row that
    sums to zero -- so a slide whose annotation covers a fifth of its spots aligns 40 and scores 8.
    ``n_scored`` is the count the metric was actually averaged over and is what ``_spot_coverage``
    divides; ``n_aligned`` is the intersection, and defaults to ``n_scored`` for the callers where
    nothing is dropped in between. ``_n_common_spots`` keeps meaning the intersection, because that
    is what the already-recorded metrics files that carry it mean by it.

    Defined once and shared, because the same numbers were wanted at four alignment sites and a
    per-site copy is how they drift apart. ``alignment`` records *how* the rows were matched:
    ``positional`` is the legacy fallback for predictions written without barcodes, and it is the
    mode where a reordered prediction scores against the wrong spots, so a reader deserves to know
    which one produced the number.
    """
    n_ground_truth = int(n_ground_truth)
    n_scored = int(n_scored)
    n_aligned = n_scored if n_aligned is None else int(n_aligned)
    return {
        "_n_common_spots": n_aligned,
        "_n_spots_scored": n_scored,
        "_n_ground_truth_spots": n_ground_truth,
        "_spot_coverage": round(n_scored / n_ground_truth, 6) if n_ground_truth else 0.0,
        "_spot_alignment": alignment,
    }


def _is_barcodeless_index(index: Any) -> bool:
    """Whether a spot index is the one a table written WITHOUT spot IDs comes back with.

    ``read_csv`` gives such a file pandas' ``0, 1, 2, ...``; R's ``write.csv`` with its default row
    names writes ``1, 2, 3, ...``. Either way the labels carry no identity, so position is all there
    is to match on. Any other label is an identity, and a set of them sharing nothing with the
    ground truth is a table about other spots, not one to be read in order.
    """
    labels = [str(x) for x in index]
    n = len(labels)
    return labels == [str(i) for i in range(n)] or labels == [str(i) for i in range(1, n + 1)]


def _align_spot_tables(pred: Any, gt: Any) -> tuple[Any, Any, bool, str | None]:
    """Line a proportions table up with its ground truth, row for row.

    Returns ``(pred, gt, aligned_by_id, error)``. Matching row counts are not the same thing as
    corresponding rows: a tool that sorts its barcodes, groups spots by cluster, or round-trips
    through a dict writes an equal-length table in a different order, and a positional ``.values``
    comparison then scores every spot against a different spot.

    ``evaluate_deconvolution`` learned that and aligned whenever the indexes differ;
    ``evaluate_deconvolution_aggregated`` -- the path for three of the four full deconvolution
    datasets -- kept the old rule of aligning only when the counts differ, so a perfect prediction
    in reversed row order scored RMSE 0.68 there and 0.0 next door. One helper now, so the two
    cannot drift apart again (hunt 2026-09-30, u33b-bench-scoring-1).

    Position is used only where there is nothing else: equal counts, no shared ID, and a prediction
    index that is the barcode-less ``0..n-1`` (or R's ``1..n``) form. A prediction that carries its
    own IDs and shares none of them with the ground truth is refused, not read in order.
    """
    pred = pred.copy()
    gt = gt.copy()
    pred.index = pred.index.astype(str)
    gt.index = gt.index.astype(str)

    if pred.index.equals(gt.index):
        return pred, gt, True, None

    if pred.index.is_unique and gt.index.is_unique:
        common_idx = pred.index.intersection(gt.index)
        if len(common_idx) > 0:
            # Both sides are reindexed to the same list, so whatever order it comes back in, the
            # two tables agree row for row.
            return pred.loc[common_idx], gt.loc[common_idx], True, None

    pred_sample = list(pred.index[:3])
    gt_sample = list(gt.index[:3])
    if gt.shape[0] != pred.shape[0]:
        return (
            pred,
            gt,
            False,
            (
                f"No overlapping spot IDs between prediction ({pred.shape[0]} spots) "
                f"and ground truth ({gt.shape[0]} spots). "
                f"Pred IDs e.g. {pred_sample}, GT IDs e.g. {gt_sample}"
            ),
        )
    if not _is_barcodeless_index(pred.index):
        return (
            pred,
            gt,
            False,
            (
                f"Prediction spot IDs share none of the ground truth's ({pred.shape[0]} spots each); "
                f"a positional match is used only for a table written without spot IDs (0..n-1), "
                f"so this one is not scored. Pred IDs e.g. {pred_sample}, GT IDs e.g. {gt_sample}"
            ),
        )
    # Equal counts, a barcode-less prediction: nothing to align on, so position is the only thing
    # that can work. Long-standing behaviour for those tools, recorded as ``positional``.
    return pred, gt, False, None


def classify_evaluation(metrics: dict[str, Any]) -> tuple[str, Any]:
    """Is an ``evaluate_tool_output`` return a result, or a failure wearing one?

    Returns ``(status, error)``: ``("evaluated", None)`` when the run was scored, or
    ``("eval_failed", reason)`` when it was not.

    **Order matters, and it is the whole point of this function.** An ``"error"`` key means an
    evaluator above REFUSED to score the run, and several of those refusals carry a diagnostic
    integer beside the message so a reader can see why nothing was measurable -- ``n_total_genes``
    next to "No significant genes in standardized output", ``n_spots``/``n_spots_scored``/
    ``n_celltypes`` next to "No spot has a non-zero ground-truth row". Three callers each decided the
    status by scanning for "a real metric" with ``k != "error"`` in the comprehension, which excludes
    the error KEY from the scan but does not notice that an error is PRESENT. So those diagnostic
    integers read as scores: "nothing is measurable" was recorded ``evaluated``, printed ``[OK]``, and
    counted among the successes, and the failure counter never saw it.

    Ask about the error first; only then ask whether anything numeric came back at all. That second
    question is still load-bearing -- an evaluator that returns neither an error nor a number has also
    failed, just in a differently-shaped way -- and ``0.0`` is a measurement, not a failure.

    Defined once and shared for the same reason ``_coverage_fields`` is: a per-caller copy of a rule
    is how the callers stop agreeing about it. The reason is handed back exactly as the evaluator
    wrote it, not stringified, so that adopting this function anywhere changes only the wrong answers.
    """
    if "error" in metrics:
        return "eval_failed", metrics["error"]
    # A count is not a measurement. ``evaluate_svg`` always sets ``n_predicted_genes``, and its
    # Moran's I failure branches add ``n_predicted``/``n_in_adata``, so an SVG run with no curated
    # truth and no computable Moran's I -- nothing measured at all -- read as ``evaluated`` on a
    # gene count (hunt 2026-09-30, u33b-bench-scoring-10). Every diagnostic count the evaluators
    # write is spelled ``n_*``; no metric is.
    has_real = any(isinstance(v, (int, float)) and not k.startswith(("_", "n_")) for k, v in metrics.items())
    if has_real:
        return "evaluated", None
    return "eval_failed", "Evaluation returned no numeric metrics"


def _degenerate_partition(values: set[str], n_labelled: int, side: str, column: str) -> str | None:
    """Why this label set cannot support a clustering score, or None if it can.

    Four shapes force a metric to exactly 1.0 whatever the other side holds, because sklearn's
    definitions are then trivially satisfied. A single class makes every cluster pure, so
    homogeneity is 1.0; a class per spot makes every class undivided, so completeness is 1.0 -- and
    both statements hold with the sides swapped. When the two sides collapse the same way, ARI and
    NMI are 1.0 as well, so a run that predicted one cluster is published as a perfect
    reconstruction. Either shape also sits happily beside an ARI near zero, which reads as "the
    method over-split the tissue": a claim about the method that this data cannot support.

    ``evaluate_clustering`` refused a one-value *prediction* already and asked nothing of the
    ground truth; ``evaluate_clustering_from_standardized`` -- the path every recorded clustering
    cell went through -- asked nothing of either side. Shared rather than copied into each, for the
    reason ``_coverage_fields`` and ``classify_evaluation`` are shared: a per-site copy of a rule
    is how the sites stop agreeing about it. ``tools/eval_mcp_server.py`` warns about the same two
    ground-truth shapes, and this is deliberately its wording.

    Refusing rather than warning, because a warning nothing reads is how a run gets recorded as
    scored. Callers put the returned string in an ``"error"`` key, which ``classify_evaluation``
    maps to ``eval_failed`` with the reason handed through verbatim.

    Below three labelled spots the degenerate shapes and the real ones are the same shape, so no
    verdict is issued -- matching the MCP check rather than diverging from it.
    """
    if n_labelled < 3:
        return None
    n_distinct = len(values)
    noun = "Ground truth column" if side == "ground truth" else "Predicted column"
    other = "prediction" if side == "ground truth" else "ground truth"
    if n_distinct <= 1:
        forced = "homogeneity" if side == "ground truth" else "completeness"
        return (
            f"{noun} '{column}' puts all {n_labelled} spots in one cluster, so it defines no "
            f"partition to agree with -- {forced} is 1.0 by construction here, for any {other}"
        )
    if n_distinct == n_labelled:
        forced = "completeness" if side == "ground truth" else "homogeneity"
        return (
            f"{noun} '{column}' has one distinct label per spot ({n_distinct} labels for "
            f"{n_labelled} spots), so it defines no partition to agree with -- {forced} is 1.0 by "
            f"construction here, for any {other}"
        )
    return None


def _resolve_cluster_gt_column(columns: list[str], ground_truth_key: str) -> tuple[str | None, str]:
    """The ground-truth column to score against, or (None, why not).

    A key that was named and is absent is an error, not a cue to go looking: the fallback search
    used to answer the tuner's hard-coded ``cell_type`` on the MERFISH z-domain task with
    ``Cell_class`` -- 15 cell classes instead of 8 regions -- and the only trace was a stderr line
    (hunt 2026-09-30, u33b-bench-scoring-4). The candidate search is for a caller that named no key,
    and the column it lands on is recorded.
    """
    if ground_truth_key:
        if ground_truth_key in columns:
            return ground_truth_key, ""
        return None, (
            f"Ground truth column '{ground_truth_key}' not found. Available: {columns}. A named key "
            f"is not substituted with another column; pass the dataset's registry ground_truth_key"
        )
    found = _find_column(columns, _CLUSTER_GT_CANDIDATES)
    if found is None:
        return None, f"No ground truth column given and none of the known names found. Available: {columns}"
    return found, ""


def _columns_copied_from(obs: Any, source_obs: Any) -> list[str]:
    """Columns of ``obs`` that hold exactly the same labels as ``source_obs`` on every shared spot."""
    shared_spots = obs.index.intersection(source_obs.index)
    if len(shared_spots) == 0:
        return []
    copied = []
    for col in obs.columns:
        if col not in source_obs.columns:
            continue
        mine = obs.loc[shared_spots, col].astype(str).to_numpy()
        theirs = source_obs.loc[shared_spots, col].astype(str).to_numpy()
        if (mine == theirs).all():
            copied.append(col)
    return copied


def _read_clustering_table(path: str) -> Any:
    """A CSV/TSV clustering output as an obs-like frame indexed by spot ID, or an error string.

    The labels are found by the same candidate search the validator's CSV branch uses; the spot
    column by the standardizer's identifier search (one value per row, by keyword). A table with no
    spot identifier is refused rather than read in row order -- clustering aligns by ID only.
    """
    import pandas as pd

    from spatialomicsgym.benchmarking.output_standardizer import _find_spot_id_column

    sep = "\t" if Path(path).suffix.lower() == ".tsv" else ","
    try:
        frame = pd.read_csv(path, sep=sep)
    except Exception as exc:
        return f"Failed to read clustering table {Path(path).name}: {exc}"
    spot_col = _find_spot_id_column(frame)
    if spot_col is None and len(frame.columns) and str(frame.columns[0]).startswith("Unnamed: 0"):
        if frame.iloc[:, 0].astype(str).nunique() == len(frame):
            spot_col = frame.columns[0]
    if spot_col is None:
        return (
            f"Clustering table {Path(path).name} has no spot identifier column (columns: "
            f"{list(frame.columns)}), so its labels cannot be aligned to the ground truth"
        )
    frame = frame.set_index(spot_col)
    frame.index = frame.index.astype(str)
    return frame


def evaluate_clustering(
    output_h5ad: str,
    ground_truth_key: str,
    predicted_key: str = "",
    ground_truth_h5ad: str = "",
) -> dict[str, Any]:
    """Evaluate clustering results against ground truth labels.

    Searches for predicted cluster labels using an extensive candidate list
    with deterministic priority order. Reports diagnostics on failure.

    Args:
        output_h5ad: Path to the tool's output h5ad file.
        ground_truth_key: Column name in obs for ground truth labels.
        predicted_key: Preferred predicted column name (optional).
        ground_truth_h5ad: Separate ground truth h5ad if different from output.

    Returns:
        Dict with clustering metrics (ARI, NMI, etc.) plus diagnostics,
        or an error dict with diagnostic info if evaluation fails.
    """

    import anndata as ad
    import numpy as np

    separate_gt = bool(ground_truth_h5ad) and ground_truth_h5ad != output_h5ad

    # A CSV clustering output (PRECAST, BASS and IRIS all write one as their authoritative
    # prediction) passed ``validate_before_evaluation``'s CSV branch and then reached
    # ``ad.read_h5ad`` here, which raised "file signature not found" -- the phantom error the
    # validator's CSV branch was written to remove. Read it as the table it is, keyed by its spot
    # column, and score it through the same path (hunt 2026-09-30, u33b-bench-scoring-3).
    if Path(output_h5ad).suffix.lower() in {".csv", ".tsv"}:
        table = _read_clustering_table(output_h5ad)
        if isinstance(table, str):
            return {"error": table, "_output_file": output_h5ad}
        if not separate_gt:
            return {
                "error": "A CSV clustering output carries no ground truth; pass the ground-truth h5ad",
                "_output_file": output_h5ad,
            }
        obs, output_shape = table, [int(len(table)), 0]
    else:
        adata = ad.read_h5ad(output_h5ad)
        obs, output_shape = adata.obs, list(adata.shape)
    obs_cols = list(obs.columns)

    # The ground truth is read before the prediction column is chosen, so the search can tell the
    # tool's column from one it merely copied over from the input.
    gt_obs = ad.read_h5ad(ground_truth_h5ad).obs if separate_gt else obs

    # A worker that keeps the input obs carries the input's own columns -- on Slide-seqV2 the
    # authors' Seurat clustering -- into its output beside the one it added. A column identical to
    # the ground-truth file's is not something the tool produced, so the fallback search skips it.
    # An explicit ``predicted_key`` is the inspector's decision and is honoured as given
    # (hunt 2026-09-30, u33b-bench-scoring-2).
    copied_from_input: list[str] = []
    if separate_gt and not (predicted_key and predicted_key in obs_cols):
        copied_from_input = _columns_copied_from(obs, gt_obs)
    pred_col = _find_column(
        [c for c in obs_cols if c not in copied_from_input], _CLUSTER_PRED_CANDIDATES, preferred=predicted_key
    )

    diagnostics = {
        "_output_file": output_h5ad,
        "_output_shape": output_shape,
        "_obs_columns": obs_cols,
        "_predicted_column_searched": _CLUSTER_PRED_CANDIDATES[:10],
        "_predicted_column_matched": pred_col,
        "_ground_truth_column": ground_truth_key,
    }
    if copied_from_input:
        diagnostics["_predicted_columns_skipped_as_input_copies"] = copied_from_input

    if pred_col is None:
        return {
            "error": f"No predicted cluster column found. Available obs: {obs_cols}"
            + (f" (skipped as copies of the input's own columns: {copied_from_input})" if copied_from_input else ""),
            **diagnostics,
        }

    # Check for degenerate predictions (all NaN or all identical)
    pred_values_raw = obs[pred_col]
    pred_non_nan = pred_values_raw.dropna()
    if len(pred_non_nan) == 0:
        return {
            "error": f"Predicted column '{pred_col}' contains all NaN values",
            **diagnostics,
        }
    if pred_non_nan.nunique() <= 1:
        return {
            "error": (
                f"Predicted column '{pred_col}' has only one unique value "
                f"({pred_non_nan.unique().tolist()}); clustering may have failed"
            ),
            **diagnostics,
        }

    # Check for ambiguous predictions (too many clusters relative to sample size)
    n_predicted_clusters = int(pred_non_nan.nunique())
    if n_predicted_clusters > len(pred_non_nan) * 0.5:
        diagnostics["_warning_high_cluster_count"] = (
            f"Unusually high cluster count ({n_predicted_clusters}) relative to sample size ({len(pred_non_nan)})"
        )

    gt_col, gt_error = _resolve_cluster_gt_column(list(gt_obs.columns), ground_truth_key)
    if gt_col is None:
        return {"error": gt_error, **diagnostics}
    diagnostics["_ground_truth_column_matched"] = gt_col
    if not ground_truth_key:
        diagnostics["_ground_truth_column_auto_detected"] = True

    # The ground truth scored against itself is ARI 1.0 by construction. Nothing stopped a GT-like
    # candidate ('label', 'class', ...) or an explicit key from naming the truth column.
    if pred_col == gt_col:
        return {
            "error": (
                f"Predicted column '{pred_col}' is the ground-truth column itself; scoring it would "
                f"compare the ground truth with itself"
            ),
            **diagnostics,
        }

    # Align indices if ground truth is from a different file
    if separate_gt:
        common = obs.index.intersection(gt_obs.index)
        if len(common) == 0:
            return {"error": "No overlapping indices between output and ground truth", **diagnostics}
        pred_labels = obs.loc[common, pred_col].values
        true_labels = gt_obs.loc[common, gt_col].values
        n_common = int(len(common))
    else:
        pred_labels = obs[pred_col].values
        true_labels = obs[gt_col].values
        n_common = int(len(obs))

    # Drop NaN from both
    mask = ~(
        np.array([str(x) in ("nan", "None", "") for x in pred_labels])
        | np.array([str(x) in ("nan", "None", "") for x in true_labels])
    )
    pred_labels = pred_labels[mask]
    true_labels = true_labels[mask]

    # How much of the slide this score covers. Scoring the intersection is right -- a tool that ran
    # its own QC genuinely has no prediction for the spots it dropped -- but the number scored means
    # nothing without the number there were, and that denominator was never recorded. Two recorded
    # runs on ``visium_dlpfc_domain`` were scored on 3,481 of 4,226 spots and tabulated beside tools
    # scored on all 4,226, with nothing in either metrics file to say so.
    #
    # Below the NaN mask, not above it: the metric is averaged over the labelled pairs, so those are
    # what the coverage is of. A slide annotated on a fifth of its spots aligns all of them and
    # scores an eighth of them, and taking the alignment as the numerator published 1.0 beside this
    # function's own ``_n_samples_evaluated`` of 3.
    diagnostics.update(_coverage_fields(int(len(pred_labels)), int(len(gt_obs)), n_aligned=n_common))

    if len(pred_labels) == 0:
        return {"error": "No valid label pairs after filtering NaN", **diagnostics}

    pred_values = {str(x) for x in pred_labels}
    gt_values = {str(x) for x in true_labels}
    n_predicted_clusters = len(pred_values)
    n_gt_clusters = len(gt_values)
    n_labelled = int(len(pred_labels))
    diagnostics["_n_samples_evaluated"] = n_labelled
    diagnostics["_n_predicted_clusters"] = n_predicted_clusters
    diagnostics["_n_gt_clusters"] = n_gt_clusters
    diagnostics["_n_true_clusters"] = n_gt_clusters

    # The prediction was checked for a single value up at the top, before alignment; neither side
    # was ever checked for the shapes that make a metric 1.0 by construction. Ground truth first:
    # when both sides are degenerate, a truth file parsed with the wrong delimiter is the more
    # actionable of the two answers.
    for values, side, column in ((gt_values, "ground truth", gt_col), (pred_values, "prediction", pred_col)):
        reason = _degenerate_partition(values, n_labelled, side, column)
        if reason:
            return {"error": reason, **diagnostics}

    # Compute metrics
    try:
        from sklearn.metrics import (
            adjusted_rand_score,
            homogeneity_completeness_v_measure,
            normalized_mutual_info_score,
        )

        pred_str = [str(x) for x in pred_labels]
        true_str = [str(x) for x in true_labels]

        ari = adjusted_rand_score(true_str, pred_str)
        nmi = normalized_mutual_info_score(true_str, pred_str, average_method="arithmetic")
        h, c, v = homogeneity_completeness_v_measure(true_str, pred_str)

        return {
            "ARI": round(float(ari), 4),
            "NMI": round(float(nmi), 4),
            "homogeneity": round(float(h), 4),
            "completeness": round(float(c), 4),
            "v_measure": round(float(v), 4),
            **diagnostics,
        }
    except Exception as e:
        return {"error": f"Metric computation failed: {e}", **diagnostics}


def evaluate_clustering_from_standardized(
    standardized_csv: str,
    ground_truth_h5ad: str,
    ground_truth_key: str,
) -> dict[str, Any]:
    """Evaluate clustering from standardized CSV with [spot_id, predicted_label].

    The standardized CSV has columns: spot_id, predicted_label.
    Ground truth is in the h5ad obs column specified by ground_truth_key.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    pred_df = pd.read_csv(standardized_csv)
    if "spot_id" not in pred_df.columns or "predicted_label" not in pred_df.columns:
        return {"error": f"Standardized CSV missing required columns. Found: {list(pred_df.columns)}"}

    pred_df = pred_df.set_index("spot_id")
    pred_df.index = pred_df.index.astype(str)

    gt_adata = ad.read_h5ad(ground_truth_h5ad)

    # Find ground truth column -- a named key that is absent is refused, not substituted, and the
    # column actually scored is recorded: this path produced every recorded clustering cell and
    # wrote no trace of which column it had used (hunt 2026-09-30, u33b-bench-scoring-4).
    gt_col, gt_error = _resolve_cluster_gt_column(list(gt_adata.obs.columns), ground_truth_key)
    if gt_col is None:
        return {"error": gt_error}
    gt_trace = {"_ground_truth_column_matched": gt_col}
    if not ground_truth_key:
        gt_trace["_ground_truth_column_auto_detected"] = True

    # Align indices
    gt_adata.obs.index = gt_adata.obs.index.astype(str)
    common = pred_df.index.intersection(gt_adata.obs.index)
    if len(common) == 0:
        pred_sample = list(pred_df.index[:3])
        gt_sample = list(gt_adata.obs.index[:3])
        return {
            "error": (
                f"No overlapping spot IDs between prediction ({len(pred_df)} spots) "
                f"and ground truth ({len(gt_adata)} spots). "
                f"Pred IDs e.g. {pred_sample}, GT IDs e.g. {gt_sample}"
            ),
            **gt_trace,
        }

    pred_labels = pred_df.loc[common, "predicted_label"].values
    true_labels = gt_adata.obs.loc[common, gt_col].values

    # Drop NaN
    mask = ~(
        np.array([str(x) in ("nan", "None", "") for x in pred_labels])
        | np.array([str(x) in ("nan", "None", "") for x in true_labels])
    )
    pred_labels = pred_labels[mask]
    true_labels = true_labels[mask]

    if len(pred_labels) == 0:
        return {"error": "No valid label pairs after filtering NaN", **gt_trace}

    pred_values = {str(x) for x in pred_labels}
    gt_values = {str(x) for x in true_labels}
    n_labelled = int(len(pred_labels))

    # The counts that make a degenerate cell recognisable in a recorded metrics file. This function
    # produced every recorded clustering cell and wrote neither of them, which is why the recorded
    # corpus cannot be audited for the shapes below after the fact -- only re-run.
    counts = {
        "_n_samples_evaluated": n_labelled,
        "_n_predicted_clusters": len(pred_values),
        "_n_gt_clusters": len(gt_values),
        "_n_true_clusters": len(gt_values),
        **gt_trace,
    }

    for values, side, column in ((gt_values, "ground truth", gt_col), (pred_values, "prediction", "predicted_label")):
        reason = _degenerate_partition(values, n_labelled, side, column)
        if reason:
            return {"error": reason, "_source": "standardized_csv", **counts}

    from sklearn.metrics import (
        adjusted_rand_score,
        homogeneity_completeness_v_measure,
        normalized_mutual_info_score,
    )

    pred_str = [str(x) for x in pred_labels]
    true_str = [str(x) for x in true_labels]

    ari = adjusted_rand_score(true_str, pred_str)
    nmi = normalized_mutual_info_score(true_str, pred_str, average_method="arithmetic")
    h, c, v = homogeneity_completeness_v_measure(true_str, pred_str)

    return {
        "ARI": round(float(ari), 4),
        "NMI": round(float(nmi), 4),
        "homogeneity": round(float(h), 4),
        "completeness": round(float(c), 4),
        "v_measure": round(float(v), 4),
        "_source": "standardized_csv",
        "_n_samples": int(len(pred_labels)),
        **counts,
        # ``n_labelled``, not ``len(common)``: the scores above are computed over the pairs that
        # survived the NaN mask, and the coverage has to be the coverage of those.
        **_coverage_fields(n_labelled, gt_adata.n_obs, n_aligned=len(common)),
    }


# The canonical SVG comparison filters each prediction to the dataset's top 1500 Seurat HVGs
# (``svg_hvg_canonical_processor.py``); data prep (``prepare_svg_input_hvg``) uses the same recipe.
_SVG_HVG_N = 1500
_SVG_HVG_CACHE: dict[tuple[str, int, int], tuple[set[str], str, str]] = {}


def _svg_hvg_universe(spatial_h5ad: str) -> tuple[set[str], str, str]:
    """(HVG gene set, how it was obtained, the universe this h5ad itself represents).

    Read-only: the data-prep helper writes a cache next to the input, and the evaluator must not
    write into the benchmark data, so the set is computed in memory with the same recipe --
    ``normalize_total(1e4)``, ``log1p``, ``highly_variable_genes(flavor="seurat", n_top_genes=1500)``
    -- and a panel of no more than 1500 genes is its own universe, as in data prep.
    """
    import h5py

    path = Path(spatial_h5ad)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    if key in _SVG_HVG_CACHE:
        return _SVG_HVG_CACHE[key]

    with h5py.File(path, "r") as fh:
        uns = fh.get("uns")
        hvg_n = int(uns["_hvg_n"][()]) if uns is not None and "_hvg_n" in uns else None

    import anndata as ad

    if hvg_n is not None:
        from spatialomicsgym.utils.file_io import read_h5ad_backed

        with read_h5ad_backed(str(path)) as adata:
            genes = {str(g) for g in adata.var_names}
        found = (genes, f"hvg{hvg_n} (the input is the data-prep HVG cache)", f"hvg{hvg_n}")
    else:
        adata = ad.read_h5ad(path)
        n_vars = int(adata.n_vars)
        if n_vars <= _SVG_HVG_N:
            genes = {str(g) for g in adata.var_names}
            found = (genes, f"full panel ({n_vars} genes <= {_SVG_HVG_N}: no HVG step, as in data prep)", "full")
        else:
            import scanpy as sc

            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=_SVG_HVG_N, flavor="seurat")
            genes = {str(g) for g in adata.var_names[adata.var["highly_variable"].to_numpy()]}
            found = (genes, f"hvg{_SVG_HVG_N} (Seurat, computed from {path.name})", f"full ({n_vars} genes)")
        del adata
    _SVG_HVG_CACHE[key] = found
    return found


def _svg_hvg_overlap(clean_pred: set[str], gt_genes: set[str], spatial_h5ad: str | None) -> dict[str, Any]:
    """F1 on the HVG universe, beside the raw F1 on whatever universe the tool ran on.

    The raw ``f1`` is computed on the prediction as given, and the predictions come from three
    gene universes: the 1500-HVG input most tools are handed, the full transcriptome the
    significance-calibrated tools get, and a tool's own prefilter. The canonical comparison
    re-filters every prediction to the HVGs, so ``metrics.json`` carries that number too, plus the
    universe each one is on (hunt 2026-09-30, u33b-bench-scoring-11; user decision: f1_hvg beside
    the raw f1). The ground truth is not filtered, exactly as in the canonical processor.
    """
    if not spatial_h5ad or not Path(spatial_h5ad).exists():
        return {"_svg_hvg_error": "spatial h5ad not provided, so the HVG universe is unknown; no f1_hvg"}
    try:
        hvg, how, input_universe = _svg_hvg_universe(spatial_h5ad)
    except Exception as exc:  # the extra figure must never break the score it sits beside
        return {"_svg_hvg_error": f"HVG universe not computed: {str(exc)[:200]}"}
    from eval_metrics import eval_svg_overlap

    pred_hvg = clean_pred & hvg
    hvg_metrics = eval_svg_overlap(sorted(pred_hvg), sorted(gt_genes))
    return {
        "f1_hvg": hvg_metrics["f1"],
        "precision_hvg": hvg_metrics["precision"],
        "recall_hvg": hvg_metrics["recall"],
        "jaccard_hvg": hvg_metrics["jaccard"],
        "n_predicted_genes_hvg": len(pred_hvg),
        "_svg_gene_universe": input_universe,
        "_svg_hvg_universe": how,
    }


def evaluate_svg(
    predicted_genes: list[str],
    ground_truth_csv: str | None,
    spatial_h5ad: str | None = None,
) -> dict[str, Any]:
    """Evaluate SVG detection against curated gene list + Moran's I.

    Two-part evaluation:
      1. Overlap metrics (Jaccard, precision, recall, F1) vs curated list
      2. Moran's I spatial autocorrelation on predicted genes (requires h5ad)

    Args:
        predicted_genes: List of predicted spatially variable gene names.
        ground_truth_csv: Path to CSV file with curated SVG list (first column).
        spatial_h5ad: Path to spatial h5ad for Moran's I computation.

    Returns:
        Dict with overlap metrics + Moran's I metrics.
    """
    result: dict[str, Any] = {"n_predicted_genes": len(predicted_genes)}

    if not predicted_genes:
        result["error"] = "Empty predicted gene list"
        return result

    # ── Part 1: Overlap metrics ──
    # Only a gene table is a curated SVG ground truth. A dataset with no curated list
    # (slide_seqv2_aorta_svg: ground_truth_path null) had every driver substitute the spatial h5ad
    # here, and ``pd.read_csv`` on HDF5 raised "'utf-8' codec can't decode byte 0x89" -- no run on
    # that dataset could be scored, and the error named an encoding problem. Anything else is
    # the no-ground-truth case it really is: Moran's I only (hunt 2026-09-30, u33b-bench-scoring-9).
    gt_suffixes = {x.lower() for x in Path(ground_truth_csv).suffixes} if ground_truth_csv else set()
    is_gene_table = bool(gt_suffixes & {".csv", ".tsv", ".txt"})
    if ground_truth_csv and Path(ground_truth_csv).exists() and is_gene_table:
        import pandas as pd
        from eval_metrics import eval_svg_overlap

        gt_df = pd.read_csv(ground_truth_csv, sep="\t" if ".tsv" in gt_suffixes else ",")
        gt_genes = {str(g) for g in gt_df.iloc[:, 0].tolist()}
        # Clean: remove blank probes, whitespace
        clean_pred = [g.strip() for g in predicted_genes if g.strip() and not g.startswith("Blank")]
        overlap_metrics = eval_svg_overlap(list(set(clean_pred)), list(gt_genes))
        result.update(overlap_metrics)
        result.update(_svg_hvg_overlap(set(clean_pred), gt_genes, spatial_h5ad))
        # Recall above is measured against every curated gene, including ones this assay never
        # measured: a curated list scored against a targeted panel has a hard recall ceiling of
        # in-panel/total, and duplicate rows in the CSV silently shrink the denominator. Disclose
        # both so a low recall can be read as "unreachable truth" instead of "bad tool".
        result["n_gt_genes"] = len(gt_genes)
        if len(gt_df) != len(gt_genes):
            result["n_gt_rows"] = int(len(gt_df))
        if spatial_h5ad and Path(spatial_h5ad).exists():
            try:
                import anndata as ad

                adata = ad.read_h5ad(spatial_h5ad, backed="r")
                try:
                    panel = {str(v) for v in adata.var_names}
                finally:
                    if adata.file is not None:
                        adata.file.close()
                n_in_panel = len(gt_genes & panel)
                result["n_gt_in_panel"] = n_in_panel
                if n_in_panel < 0.5 * len(gt_genes):
                    ceiling = n_in_panel / len(gt_genes) if gt_genes else 0.0
                    result["gt_panel_warning"] = (
                        f"Only {n_in_panel} of {len(gt_genes)} curated SVG ground-truth genes exist in "
                        f"this dataset's {len(panel)}-gene panel, so recall cannot exceed "
                        f"{ceiling:.1%} for any tool. Overlap metrics reflect panel coverage more "
                        f"than tool quality; compare tools on this dataset against the "
                        f"{n_in_panel} reachable genes or re-curate a panel-matched ground truth."
                    )
            except Exception:  # disclosure must never break scoring
                pass
    else:
        result["evaluation_mode"] = "no_ground_truth"
        result["note"] = "No curated SVG ground truth available"
        if ground_truth_csv and Path(ground_truth_csv).exists():
            result["_ground_truth_ignored"] = (
                f"{Path(ground_truth_csv).name} is not a curated SVG gene list (.csv/.tsv/.txt); scored "
                f"without ground truth"
            )

    # ── Part 2: Moran's I on predicted SVGs ──
    if not spatial_h5ad or not Path(spatial_h5ad).exists():
        result["morans_i_warning"] = (
            "Moran's I not computed — spatial h5ad not provided. "
            "Moran's I is a required metric for SVG evaluation when spatial data is available."
        )
    if spatial_h5ad and Path(spatial_h5ad).exists():
        try:
            morans = _compute_morans_i_validated(spatial_h5ad, predicted_genes)
            result.update(morans)
        except Exception as e:
            result["morans_i_error"] = str(e)[:200]

    return result


def evaluate_svg_from_standardized(
    standardized_csv: str,
    ground_truth_csv: str | None,
    spatial_h5ad: str | None = None,
) -> dict[str, Any]:
    """Evaluate SVG from standardized CSV with [gene, score, pvalue, significant].

    Extracts significant genes and routes to evaluate_svg().
    """
    import pandas as pd

    df = pd.read_csv(standardized_csv)
    if "gene" not in df.columns:
        return {"error": f"Standardized SVG CSV missing 'gene' column. Found: {list(df.columns)}"}

    # Use significant column if available, otherwise use all genes
    if "significant" in df.columns:
        sig_genes = df[df["significant"] == True]["gene"].tolist()  # noqa: E712
    else:
        sig_genes = df["gene"].tolist()

    if not sig_genes:
        return {"error": "No significant genes in standardized output", "n_total_genes": len(df)}

    return evaluate_svg(sig_genes, ground_truth_csv, spatial_h5ad=spatial_h5ad)


def _compute_morans_i_validated(
    h5ad_path: str,
    gene_list: list[str],
    n_neighbors: int = 6,
) -> dict[str, Any]:
    """Compute Moran's I with strict validation.

    Validation steps:
      1. Check spatial coordinates exist and are valid
      2. Check genes exist in the expression matrix
      3. Filter zero-variance and all-zero genes
      4. Build spatial weights from coordinates
      5. Compute per-gene Moran's I
      6. Return structured results
    """
    import anndata as ad
    import numpy as np

    adata = ad.read_h5ad(h5ad_path)

    # ── Validate spatial coordinates ──
    if "spatial" not in adata.obsm:
        return {"morans_i_error": "No obsm['spatial'] — cannot compute Moran's I"}

    coords_raw = adata.obsm["spatial"]
    # Convert DataFrame/sparse to ndarray
    if hasattr(coords_raw, "values"):
        coords = np.asarray(coords_raw.values, dtype=np.float64)
    elif hasattr(coords_raw, "toarray"):
        coords = np.asarray(coords_raw.toarray(), dtype=np.float64)
    else:
        coords = np.asarray(coords_raw, dtype=np.float64)

    # Use only first 2 columns (x, y)
    if coords.ndim == 2 and coords.shape[1] > 2:
        coords = coords[:, :2]

    if coords.shape[0] != adata.n_obs:
        return {"morans_i_error": f"Coordinate mismatch: {coords.shape[0]} coords vs {adata.n_obs} obs"}

    # Check for NaN/inf coordinates
    if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
        return {"morans_i_error": "Invalid coordinates (NaN or Inf)"}

    # ── Validate genes ──
    clean_genes = [g.strip() for g in gene_list if g.strip() and not g.startswith("Blank")]
    valid_genes = [g for g in clean_genes if g in adata.var_names]
    if not valid_genes:
        return {
            "morans_i_error": "No predicted genes found in expression matrix",
            "n_predicted": len(clean_genes),
            "n_in_adata": 0,
        }

    # ── Get expression matrix for valid genes ──
    gene_idx = [list(adata.var_names).index(g) for g in valid_genes]
    if hasattr(adata.X, "toarray"):
        expr = adata.X[:, gene_idx].toarray()
    else:
        expr = np.array(adata.X[:, gene_idx])
    expr = np.asarray(expr, dtype=np.float64)

    # ── Filter zero-variance and all-zero genes ──
    gene_var = np.var(expr, axis=0)
    gene_nonzero = np.sum(expr != 0, axis=0)
    keep_mask = (gene_var > 1e-10) & (gene_nonzero >= 3)
    filtered_genes = [valid_genes[i] for i in range(len(valid_genes)) if keep_mask[i]]
    filtered_expr = expr[:, keep_mask]

    if not filtered_genes:
        return {
            "morans_i_error": "All genes have zero variance after filtering",
            "n_valid_genes": len(valid_genes),
            "n_after_filter": 0,
        }

    # ── Build spatial weights (KNN) ──
    from sklearn.neighbors import kneighbors_graph

    k = min(n_neighbors, adata.n_obs - 1)
    W = kneighbors_graph(coords, n_neighbors=k, mode="connectivity")
    # Row-normalize
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    W = W.multiply(1.0 / row_sums[:, np.newaxis])

    # ── Compute per-gene Moran's I ──
    n = adata.n_obs
    per_gene: dict[str, float] = {}
    for i, gene in enumerate(filtered_genes):
        x = filtered_expr[:, i]
        x_mean = np.mean(x)
        x_centered = x - x_mean
        denominator = np.sum(x_centered**2)
        if denominator < 1e-15:
            per_gene[gene] = 0.0
            continue
        # Moran's I = (n / S0) * (x'Wx / x'x) where S0 = sum of weights
        Wx = np.array(W.dot(x_centered)).flatten()
        numerator = np.sum(x_centered * Wx)
        S0 = W.sum()
        morans_i = (n / S0) * (numerator / denominator) if S0 > 0 else 0.0
        per_gene[gene] = round(float(morans_i), 6)

    values = list(per_gene.values())
    return {
        "mean_morans_i": round(float(np.mean(values)), 4) if values else None,
        "median_morans_i": round(float(np.median(values)), 4) if values else None,
        "n_genes_morans_i": len(values),
        "n_genes_filtered_out": len(valid_genes) - len(filtered_genes),
        "top_morans_i_genes": dict(sorted(per_gene.items(), key=lambda x: -x[1])[:10]),
    }


def evaluate_deconvolution(
    predicted_proportions_csv: str,
    ground_truth_h5ad: str,
    celltype_key: str = "CellType",
    ground_truth_key: str = "",
    reference_classes: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate deconvolution results against known cell type labels.

    Converts categorical ground truth labels from the h5ad into one-hot
    proportions and compares against predicted proportions.

    Args:
        predicted_proportions_csv: Path to CSV with predicted cell type proportions
            (rows=spots, columns=cell types).
        ground_truth_h5ad: Path to h5ad with ground truth cell type labels in obs.
        celltype_key: Column name in obs containing ground truth cell type labels.
            Used when ground_truth_key is not provided.
        ground_truth_key: Overrides celltype_key if provided and found in obs.
        reference_classes: The single-cell reference's cell-type labels, when known
            (``evaluate_deconvolution_auto`` reads them from the registry). A reference
            class the prediction left out is scored as a predicted 0 rather than
            dropped from the ground truth.

    Returns:
        Dict of deconvolution metrics (RMSE, MAE, Pearson, JSD, etc.) or error dict.
    """
    import anndata as ad
    import pandas as pd
    from eval_metrics import eval_deconvolution

    pred = _read_proportions(predicted_proportions_csv)

    # Safety: detect transposed predictions (rows=celltypes, cols=spots)
    # Heuristic: if columns look like barcodes (contain dashes, >15 chars) and
    # there are way more columns than rows, transpose
    sample_cols = [str(c) for c in pred.columns[:5]]
    if pred.shape[1] > pred.shape[0] * 2 and any("-" in c or len(c) > 15 for c in sample_cols):
        pred = pred.T

    adata = ad.read_h5ad(ground_truth_h5ad)

    # Resolve the cell type column: ground_truth_key overrides celltype_key
    resolved_key = None
    if ground_truth_key and ground_truth_key in adata.obs.columns:
        resolved_key = ground_truth_key
    elif celltype_key in adata.obs.columns:
        resolved_key = celltype_key

    # Extended search for cell type column
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
            "cell_annotation",
            "Cell_class",
            "author_cell_type",
            "label",
        ]:
            if candidate in adata.obs.columns:
                resolved_key = candidate
                break

    if resolved_key is None:
        return {
            "error": (
                f"Cell type column not found in h5ad obs. "
                f"Tried ground_truth_key='{ground_truth_key}' and celltype_key='{celltype_key}'. "
                f"Available columns: {list(adata.obs.columns)}"
            ),
        }

    # Convert categorical labels to one-hot proportions
    ct = adata.obs[resolved_key]
    gt_raw = pd.get_dummies(ct)

    # Try fuzzy cell type name matching (handles underscores, dashes, case)
    import re

    def _normalize_ct(name: str) -> str:
        # `/` folds with the rest. Cell-type names carry it -- the Allen subclasses `L2/3 IT`,
        # `L4/5 IT`, `L5/6 NP` -- and tools sanitize it away because it is a path separator, so the
        # prediction says `L2_3_IT` where the ground truth says `L2/3 IT`. Left unfolded, those are
        # different names and `reindex(fill_value=0)` below hands the evaluator an all-zero column
        # for a subclass that is really in the slide. `_normalize_label` (:955) already folds it for
        # this exact reason; the two normalizers were disagreeing about the same phenomenon.
        s = str(name).strip().lower()
        s = re.sub(r"[_\-\s/]+", " ", s)
        return s

    # A cell type the prediction leaves out used to leave the scoring with it: the ground truth is
    # cut to the prediction's columns below, so the spots of an omitted type became all-zero rows,
    # and ``eval_deconvolution`` excludes those from every metric. A tool predicting only the two
    # easy classes of a four-class slide was scored on half the spots and came back RMSE 0.0. That
    # exclusion exists for a ground-truth type ABSENT FROM THE SC REFERENCE (no tool could predict
    # it), and it stays; a reference class the prediction omitted is now a predicted 0 instead.
    # Every tool on a dataset is then scored over the same reference column set (hunt 2026-09-30,
    # u33b-bench-scoring-5; user decision: zero-fill from the registry's sc reference label set).
    zero_filled: list[str] = []
    if reference_classes:
        claimed_names = {_normalize_ct(c) for c in pred.columns}
        claimed_names |= {_normalize_ct(_strip_worker_prefix(c)) for c in pred.columns}
        named_by_prediction = {_normalize_ct(c) for c in pred.columns}
        gt_names = {_normalize_ct(c) for c in gt_raw.columns}
        if not (named_by_prediction & gt_names):
            # Zero-filling here would turn "names none of the ground truth's classes" (a misread
            # label column) into an all-zero prediction scored as a real, terrible one.
            return {
                "error": (
                    f"The prediction names none of the ground truth's cell types in obs['{resolved_key}'] "
                    f"(prediction e.g. {[str(c) for c in pred.columns[:5]]}, ground truth e.g. "
                    f"{[str(c) for c in gt_raw.columns[:5]]}) -- check which reference label column it "
                    f"was made on"
                ),
            }
        pred = pred.copy()
        for ref in dict.fromkeys(str(r) for r in reference_classes):
            if _normalize_ct(ref) not in claimed_names:
                pred[ref] = 0.0
                claimed_names.add(_normalize_ct(ref))
                zero_filled.append(ref)

    common_exact = set(gt_raw.columns) & set(pred.columns)

    # Normalized matching runs unconditionally, over only the columns exact matching left over.
    # It used to be gated on `len(common_exact) < 2`, which fired backwards: a prediction that
    # spelled enough cell-type names exactly right cleared the threshold, so the normalizer never
    # ran and its *remaining* separator-variant columns (MERFISH `L5_ET` vs GT `L5 ET`) were
    # reindexed against an all-zero ground truth -- the same numbers scored worse for being better
    # named. Restricting both sides to the unmatched columns, and letting each GT column be claimed
    # once, means this can only add matches: nothing already resolved is disturbed.
    gt_norm: dict[str, str] = {}
    for c in gt_raw.columns:
        if c not in common_exact:
            gt_norm.setdefault(_normalize_ct(c), c)

    rename_gt: dict[str, str] = {}
    claimed: set[str] = set()
    for pred_col in pred.columns:
        if pred_col in common_exact:
            continue
        gt_col = gt_norm.get(_normalize_ct(pred_col))
        if gt_col is not None and gt_col not in claimed:
            rename_gt[gt_col] = pred_col
            claimed.add(gt_col)

    if rename_gt:
        gt_raw = gt_raw.rename(columns=rename_gt)

    gt = gt_raw.reindex(columns=pred.columns, fill_value=0)

    # Align by spot ID -- see ``_align_spot_tables``, shared with the aggregated path below.
    n_gt_spots = int(gt.shape[0])
    pred, gt, aligned_by_id, align_error = _align_spot_tables(pred, gt)
    if align_error:
        return {"error": align_error}

    # Final shape check -- must match after alignment
    if pred.shape != gt.shape:
        return {
            "error": (
                f"Shape mismatch after alignment: pred={pred.shape}, gt={gt.shape}. "
                f"Prediction columns: {list(pred.columns)[:10]}"
            ),
        }

    metrics = eval_deconvolution(pred.values, gt.values, list(pred.columns))
    # The path that reported nothing at all about coverage: the check above compares the two tables
    # *after* both were reduced to their intersection, so it can never fail, and a prediction
    # overlapping the ground truth on three spots scored exactly like one covering the slide.
    #
    # ``eval_deconvolution`` then drops every ground-truth row that sums to zero -- an unannotated
    # spot one-hot-encodes to all zeros -- and averages over what is left, so its own
    # ``n_spots_scored`` is the numerator here, not the aligned row count.
    metrics.update(
        _coverage_fields(
            int(metrics.get("n_spots_scored", pred.shape[0])),
            n_gt_spots,
            alignment="spot_id" if aligned_by_id or pred.index.equals(gt.index) else "positional",
            n_aligned=pred.shape[0],
        )
    )
    if reference_classes:
        metrics["_zero_filled_reference_classes"] = zero_filled
    return metrics


def _normalize_label(name: str) -> str:
    """Lowercase + collapse [_-\\s/] to single space. Used for matching
    prediction column names to aggregation-table keys despite tool-side
    sanitization differences (e.g. 'Treg/Tfr' vs 'Treg_Tfr')."""
    import re

    return re.sub(r"[_\-\s/]+", " ", str(name).strip().lower())


# Worker-emitted prefixes that confound aggregation-key lookup.
# Order matters: longer prefixes first so we don't strip a substring of a longer one.
_WORKER_COL_PREFIXES = (
    "means_per_cluster_mu_fg_",  # cell2location means_per_cluster_mu_fg
    "q05_cell_abundance_w_sf_",  # cell2location preferred posterior
    "means_cell_abundance_w_sf_",  # cell2location alternative
    "cell_abundance_w_sf_",  # cell2location bare
    "stds_cell_abundance_w_sf_",  # cell2location std
    "proportion_",  # generic
    "proportions_",
    "weights_",  # spacexr
    "abundance_",
)


def _strip_worker_prefix(name: str) -> str:
    s = str(name)
    for p in _WORKER_COL_PREFIXES:
        if s.startswith(p):
            return s[len(p) :]
    return s


def evaluate_deconvolution_aggregated(
    predicted_proportions_csv: str,
    ground_truth_h5ad: str,
    aggregation_yaml: str,
    celltype_key: str = "CellType",
    ground_truth_key: str = "",
) -> dict[str, Any]:
    """Evaluate deconvolution after aggregating fine prediction labels into
    coarse GT classes via a YAML mapping table.

    Use when the GT obs column has fewer / coarser labels than the sc-reference
    cell types the tool predicted (e.g. Visium lymph-node GT has 4 zones but
    the sc reference has 44 fine clusters). For tools that already predict at
    the coarse level, the YAML can include identity entries (or simply omit
    them — unmapped columns get dropped with their mass redistributed only if
    they normalize-match an entry; see `unmapped_columns` in the result).

    Args:
        predicted_proportions_csv: spot × celltype predictions CSV.
        ground_truth_h5ad: spatial AnnData whose obs[celltype_key] holds the
            coarse GT label per spot.
        aggregation_yaml: path to YAML with keys `gt_classes` (list) and
            `mapping` (dict: fine_label → coarse_label).
        celltype_key, ground_truth_key: same as evaluate_deconvolution.

    Returns:
        Same shape as eval_deconvolution result, plus diagnostic fields
        `_aggregation_used`, `_n_pred_cols_in`, `_n_pred_cols_out`,
        `_unmapped_columns` (pred cols that had no normalize-match in the
        table — their mass is dropped before metric computation).
    """
    import anndata as ad
    import pandas as pd
    import yaml
    from eval_metrics import eval_deconvolution

    with open(aggregation_yaml) as f:
        cfg = yaml.safe_load(f)
    gt_classes: list[str] = list(cfg["gt_classes"])
    mapping_raw: dict[str, str] = dict(cfg["mapping"])

    # Normalize the mapping keys for fuzzy lookup
    norm_to_target: dict[str, str] = {_normalize_label(k): v for k, v in mapping_raw.items()}

    pred = _read_proportions(predicted_proportions_csv)
    sample_cols = [str(c) for c in pred.columns[:5]]
    if pred.shape[1] > pred.shape[0] * 2 and any("-" in c or len(c) > 15 for c in sample_cols):
        pred = pred.T

    # Aggregate pred columns into the GT class set
    aggregated = pd.DataFrame(0.0, index=pred.index, columns=gt_classes)
    unmapped: list[str] = []
    for col in pred.columns:
        target = norm_to_target.get(_normalize_label(col))
        if target is None:
            target = norm_to_target.get(_normalize_label(_strip_worker_prefix(col)))
        if target is None or target not in gt_classes:
            unmapped.append(str(col))
            continue
        aggregated[target] = aggregated[target] + pred[col].astype(float).values

    # Dropping one column is the table's design (the RCA YAML drops ``unknown`` on purpose);
    # dropping every column is a mapping failure. It used to be scored anyway: an all-zero
    # prediction against a valid coarse ground truth, recorded ``evaluated`` as a (bad) tool score.
    # The non-aggregated path refuses the same shape (hunt 2026-09-30, u33b-bench-scoring-6).
    unmapped_set = set(unmapped)
    numeric_pred = pred.apply(pd.to_numeric, errors="coerce").abs()
    total_mass = float(numeric_pred.sum().sum())
    unmapped_mass = float(numeric_pred[[c for c in pred.columns if str(c) in unmapped_set]].sum().sum())
    unmapped_mass_fraction = round(unmapped_mass / total_mass, 6) if total_mass > 0 else None
    if len(unmapped_set) == pred.shape[1] or float(aggregated.abs().sum().sum()) == 0.0:
        return {
            "error": (
                f"None of the prediction's {pred.shape[1]} cell-type columns maps onto the ground-truth "
                f"classes {gt_classes} through {Path(aggregation_yaml).name}, so there is nothing to "
                f"score -- check which reference label column the prediction was made on. "
                f"Columns e.g. {[str(c) for c in pred.columns[:5]]}"
            ),
            "_aggregation_used": aggregation_yaml,
            "_unmapped_columns": unmapped,
            "_unmapped_mass_fraction": unmapped_mass_fraction,
        }

    # Build GT one-hot at the coarse level
    adata = ad.read_h5ad(ground_truth_h5ad)
    resolved_key = ground_truth_key if (ground_truth_key and ground_truth_key in adata.obs.columns) else celltype_key
    if resolved_key not in adata.obs.columns:
        return {
            "error": f"Cell type column not found. Tried '{ground_truth_key}', '{celltype_key}'. "
            f"Available: {list(adata.obs.columns)}",
        }

    ct = adata.obs[resolved_key]
    gt_raw = pd.get_dummies(ct)
    # Normalize GT label columns onto gt_classes (case/whitespace/separator) before
    # the exact-match reindex -- mirrors the non-aggregated path. Without this, a
    # formatting mismatch between obs labels and the YAML gt_classes silently zeroes
    # a class and deflates mean_pearson_r. No-op when they already match exactly.
    _norm_gt_class = {_normalize_label(str(c)): c for c in gt_classes}
    _rename_gt = {
        col: _norm_gt_class[_normalize_label(str(col))]
        for col in gt_raw.columns
        if _normalize_label(str(col)) in _norm_gt_class and str(col) != _norm_gt_class[_normalize_label(str(col))]
    }
    if _rename_gt:
        gt_raw = gt_raw.rename(columns=_rename_gt)
    gt = gt_raw.reindex(columns=gt_classes, fill_value=0)

    # Align by spot ID whenever the indexes differ, not only when the counts do: an equal-length
    # table in another row order was scored spot-against-the-wrong-spot here (hunt 2026-09-30,
    # u33b-bench-scoring-1).
    n_gt_spots = int(gt.shape[0])
    aggregated, gt, aligned_by_id, align_error = _align_spot_tables(aggregated, gt)
    if align_error:
        return {"error": align_error}

    if aggregated.shape != gt.shape:
        return {"error": f"Shape mismatch: pred={aggregated.shape} gt={gt.shape}"}

    metrics = eval_deconvolution(aggregated.values, gt.values, list(aggregated.columns))
    metrics["_aggregation_used"] = aggregation_yaml
    metrics["_n_pred_cols_in"] = int(pred.shape[1])
    metrics["_n_pred_cols_out"] = int(aggregated.shape[1])
    metrics["_unmapped_columns"] = unmapped
    metrics["_unmapped_mass_fraction"] = unmapped_mass_fraction
    metrics.update(
        _coverage_fields(
            int(metrics.get("n_spots_scored", aggregated.shape[0])),
            n_gt_spots,
            alignment="spot_id" if aligned_by_id or aggregated.index.equals(gt.index) else "positional",
            n_aligned=aggregated.shape[0],
        )
    )
    return metrics


# The benchmark registry and the aggregation tables, as module attributes so a test can point them
# at a fixture of its own.
_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "benchmark_data" / "registry.yaml"
_AGGREGATION_DIR = Path(__file__).resolve().parent / "celltype_aggregation"
_REFERENCE_CLASS_CACHE: dict[tuple[str, int, str], list[str]] = {}


def _registry_entry(dataset_name: str | None, ground_truth_path: str | None, task_type: str) -> tuple[Any, str]:
    """The registry entry a deconvolution is being scored for, and how it was found.

    By name when the caller passed one. Otherwise by the ground-truth file: two of the three
    benchmark drivers never pass ``dataset_name``, so the dataset-specific scoring -- the
    aggregation table, the reference's label set -- silently switched off for them and the same
    tool on the same dataset scored differently by driver (hunt 2026-09-30, u33b-bench-scoring-7).
    A path match is accepted only when exactly one entry of this task type owns the file.
    """
    import yaml

    try:
        with open(_REGISTRY_PATH) as fh:
            raw = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return None, "registry unreadable"
    entries = [e for e in (raw.get("datasets") or []) if isinstance(e, dict)]
    if dataset_name:
        for entry in entries:
            if entry.get("name") == dataset_name:
                return entry, "dataset_name"
        return None, "dataset_name not in registry"
    if not ground_truth_path:
        return None, "no dataset_name and no ground-truth path"
    try:
        target = Path(ground_truth_path).resolve()
    except OSError:
        return None, "ground-truth path unresolvable"
    data_root = _REGISTRY_PATH.parent
    matches = []
    for entry in entries:
        if entry.get("task_type") != task_type:
            continue
        rel = entry.get("ground_truth_path") or entry.get("input_path")
        if rel and (data_root / rel).resolve() == target:
            matches.append(entry)
    if len(matches) == 1:
        return matches[0], "ground_truth_path"
    return None, f"{len(matches)} registry entries own this ground-truth file"


def _aggregation_yaml_for(entry: Any, dataset_name: str | None) -> Path | None:
    """The coarse-GT aggregation table for a dataset, or None.

    ``metadata.celltype_aggregation`` in the registry names it explicitly -- which is how the mini
    Visium entry shares the full entry's table instead of finding no ``mini_visium_deconv.yaml``
    and failing every 44-label prediction (hunt 2026-09-30, u33b-bench-scoring-8). Otherwise the
    long-standing convention, ``celltype_aggregation/<dataset_name>.yaml``.
    """
    meta = (entry or {}).get("metadata") or {}
    declared = meta.get("celltype_aggregation") if isinstance(meta, dict) else None
    if declared:
        path = _AGGREGATION_DIR / str(declared)
        return path if path.exists() else None
    name = (entry or {}).get("name") or dataset_name
    if name:
        path = _AGGREGATION_DIR / f"{name}.yaml"
        if path.exists():
            return path
    return None


def _reference_classes_for(entry: Any) -> tuple[list[str] | None, str]:
    """The sc reference's cell-type labels for a registry entry, or (None, why not)."""
    meta = (entry or {}).get("metadata") or {}
    if not isinstance(meta, dict):
        return None, "registry entry has no metadata"
    rel, key = meta.get("sc_reference_path"), meta.get("sc_reference_celltype_key")
    if not rel or not key:
        return None, "registry entry declares no sc_reference_path/sc_reference_celltype_key"
    path = _REGISTRY_PATH.parent / rel
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None, f"sc reference not found: {path}"
    cache_key = (str(path), stamp, str(key))
    if cache_key not in _REFERENCE_CLASS_CACHE:
        import pandas as pd

        from spatialomicsgym.utils.file_io import read_h5ad_backed

        try:
            with read_h5ad_backed(str(path)) as ref:
                if key not in ref.obs.columns:
                    return None, f"sc reference has no obs['{key}']"
                labels = ref.obs[key]
                _REFERENCE_CLASS_CACHE[cache_key] = [str(v) for v in pd.unique(labels.dropna().astype(str))]
        except Exception as exc:  # an unreadable reference is reported, not raised into the score
            return None, f"sc reference unreadable: {str(exc)[:200]}"
    return _REFERENCE_CLASS_CACHE[cache_key], str(path)


def evaluate_deconvolution_auto(
    predicted_proportions_csv: str,
    ground_truth_h5ad: str,
    dataset_name: str | None = None,
    celltype_key: str = "CellType",
    ground_truth_key: str = "",
) -> dict[str, Any]:
    """Auto-route to evaluate_deconvolution_aggregated() when an aggregation YAML
    exists for the dataset, otherwise fall through to evaluate_deconvolution().

    Lets the orchestrator stay dataset-agnostic — the registry entry (found by
    ``dataset_name``, or by the ground-truth file when no name is passed) decides
    whether the coarse GT / fine prediction adapter runs (``metadata.celltype_aggregation``,
    else ``celltype_aggregation/<dataset_name>.yaml``), and supplies the sc reference's
    label set the non-aggregated path zero-fills omitted classes from.
    """
    entry, found_by = _registry_entry(dataset_name, ground_truth_h5ad, "deconvolution")
    yaml_path = _aggregation_yaml_for(entry, dataset_name)
    if yaml_path is not None:
        result = evaluate_deconvolution_aggregated(
            predicted_proportions_csv,
            ground_truth_h5ad,
            str(yaml_path),
            celltype_key=celltype_key,
            ground_truth_key=ground_truth_key,
        )
    else:
        reference_classes, reference_source = _reference_classes_for(entry) if entry else (None, found_by)
        result = evaluate_deconvolution(
            predicted_proportions_csv,
            ground_truth_h5ad,
            celltype_key=celltype_key,
            ground_truth_key=ground_truth_key,
            reference_classes=reference_classes,
        )
        if entry is not None:
            result["_reference_classes_source"] = reference_source
    if entry is not None:
        result["_registry_dataset"] = entry.get("name")
        result["_registry_dataset_found_by"] = found_by
    return result


def evaluate_tool_output(
    task_type: str,
    output_path: str,
    ground_truth_path: str,
    ground_truth_key: str = "",
    predicted_key: str = "",
    standardized_file: str | None = None,
    dataset_name: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Dispatch evaluation based on task type.

    This is the main entry point for the benchmark evaluation pipeline.
    It routes to the appropriate evaluation function based on the task type
    defined in the dataset registry.

    When *standardized_file* is provided and the file exists on disk, the
    evaluator will prefer the standardized CSV path over the raw tool output.
    This allows the output_standardizer to produce uniform CSV files that
    the evaluator can consume directly.

    Args:
        task_type: One of 'spatial_clustering', 'svg_detection', 'deconvolution'.
        output_path: Path to the tool's output file (h5ad for clustering,
            CSV for deconvolution).
        ground_truth_path: Path to ground truth file (h5ad or CSV).
        ground_truth_key: Column name for ground truth labels. Defaults vary
            by task type if not provided.
        predicted_key: Column name for predicted labels (clustering only).
        standardized_file: Optional path to a standardized CSV produced by
            the output_standardizer. When provided and the file exists, the
            evaluator routes through the standardized evaluation functions
            instead of parsing raw tool outputs.
        **kwargs: Additional arguments, e.g. predicted_genes for SVG evaluation.

    Returns:
        Dict with metric names as keys and float values, plus diagnostic
        fields prefixed with '_'. Returns an error dict if evaluation fails.
    """
    result: dict[str, Any] = {"_task_type": task_type, "_output_path": output_path}

    # Route through standardized paths when available
    if standardized_file and Path(standardized_file).exists():
        if task_type == "spatial_clustering":
            metrics = evaluate_clustering_from_standardized(
                standardized_file, ground_truth_path, ground_truth_key or ""
            )
            result.update(metrics)
            return result
        elif task_type == "svg_detection":
            metrics = evaluate_svg_from_standardized(
                standardized_file,
                ground_truth_path,
                spatial_h5ad=kwargs.get("spatial_h5ad", ground_truth_path),
            )
            result.update(metrics)
            return result
        elif task_type == "deconvolution":
            # Deconvolution already uses CSV - evaluate directly (auto-route via dataset YAML if present)
            ct_key = ground_truth_key or "CellType"
            result.update(
                evaluate_deconvolution_auto(
                    standardized_file,
                    ground_truth_path,
                    dataset_name=dataset_name,
                    celltype_key=ct_key,
                    ground_truth_key=ground_truth_key,
                )
            )
            return result

    # Mandatory pre-evaluation gate
    validation = validate_before_evaluation(
        task_type,
        output_path,
        ground_truth_path,
        ground_truth_key=ground_truth_key,
        predicted_key=predicted_key,
        predicted_genes=kwargs.get("predicted_genes"),
    )
    result["_pre_eval_validation"] = validation
    if not validation["valid"]:
        result["error"] = "Pre-evaluation validation failed: " + "; ".join(validation["reasons"])
        result["_validation_status"] = validation["status"]
        return result

    if task_type == "spatial_clustering":
        gt_key = ground_truth_key or ""  # let _find_column auto-detect
        metrics = evaluate_clustering(
            output_path,
            gt_key,
            predicted_key=predicted_key,
            ground_truth_h5ad=ground_truth_path,
        )
        result.update(metrics)
    elif task_type == "svg_detection":
        genes = kwargs.get("predicted_genes", [])
        spatial_h5ad = kwargs.get("spatial_h5ad", ground_truth_path)
        if not genes:
            result["error"] = "No predicted_genes provided for SVG evaluation"
        else:
            result.update(evaluate_svg(genes, ground_truth_path, spatial_h5ad=spatial_h5ad))
    elif task_type == "deconvolution":
        ct_key = ground_truth_key or "CellType"
        result.update(
            evaluate_deconvolution_auto(
                output_path,
                ground_truth_path,
                dataset_name=dataset_name,
                celltype_key=ct_key,
                ground_truth_key=ground_truth_key,
            )
        )
    else:
        result["error"] = f"No evaluation implemented for task type: {task_type}"

    return result
