"""Objective functions for tuning evaluation.

Maps benchmark tasks to their metrics and provides proxy objectives for
non-benchmark tasks. Whether a configuration has any tuning signal to begin
with is decided upstream in ``mode_router.select_mode``, which falls back to
defaults for a task type it does not recognise.
"""

from __future__ import annotations

import logging
from pathlib import Path

from spatialomicsgym import platform_root
from spatialomicsgym.tuning.core import METRIC_LOWER_IS_BETTER, TASK_METRICS, TrialResult

logger = logging.getLogger(__name__)


def _tools_dir_for_eval_metrics() -> Path:
    """The ``tools/`` directory to put on ``sys.path`` so the evaluator can import eval_metrics.

    The checkout sibling first -- byte-identical to the old inline derivation whenever it
    exists -- then the platform rungs (seeded SOG_HOME, the wheel's ``_platform`` copy) for a
    pip-only install, where the sibling walk lands in site-packages. When every rung is dark
    the checkout-derived path is still returned: inserting a nonexistent path is what the old
    code did, and the ImportError that follows names the real problem.
    """
    checkout_tools = Path(__file__).parent.parent.parent / "tools"
    if checkout_tools.is_dir():
        return checkout_tools
    platform_tools = platform_root.tools_dir()
    if platform_tools is not None:
        return platform_tools
    return checkout_tools


def evaluate_benchmark_trial(
    task_type: str,
    output_dir: str,
    ground_truth_path: str,
    ground_truth_key: str = "cell_type",
    svg_ground_truth_path: str | None = None,
    tool_name: str = "",
) -> TrialResult:
    """Evaluate a single benchmark trial using the existing evaluator.

    Connects to benchmarks/evaluation/evaluator.py for scoring.

    *ground_truth_path* is the dataset the trial ran on. For clustering and deconvolution the ground
    truth is a column inside that file, so the two roles coincide. SVG is the exception: its ground
    truth is a curated gene list in a separate CSV, which is what *svg_ground_truth_path* carries.
    Leaving it unset does not degrade to an unscored trial -- ``evaluate_svg`` reads whatever it is
    given with ``pd.read_csv``, and an h5ad raises ``UnicodeDecodeError`` there -- so the SVG branch
    below declines up front instead.

    *tool_name* lets a deconvolution trial find its proportions table the way the benchmark does,
    through the tool's output profile, and a clustering trial score the column that profile names.
    """
    import sys
    import time

    # Add tools/ to path for eval_metrics
    tools_dir = str(_tools_dir_for_eval_metrics())
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    try:
        from benchmarks.evaluation.evaluator import evaluate_svg, evaluate_tool_output
    except ImportError:
        benchmarks_dir = str(Path(__file__).parent.parent.parent)
        if benchmarks_dir not in sys.path:
            sys.path.insert(0, benchmarks_dir)
        from benchmarks.evaluation.evaluator import evaluate_svg, evaluate_tool_output

    start = time.time()
    metrics_key = TASK_METRICS.get(task_type, {})
    primary_metric = metrics_key.get("primary", "ARI")
    secondary_metric = metrics_key.get("secondary", "NMI")

    try:
        output_path = Path(output_dir)

        # Find output files
        h5ad_files = sorted(
            output_path.rglob("*.h5ad"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        eval_file = h5ad_files[0] if h5ad_files else None
        no_score = float("inf") if METRIC_LOWER_IS_BETTER.get(primary_metric, False) else 0.0

        if task_type == "svg_detection":
            # An SVG trial is scored from the gene names the tool predicted, not from one output
            # file: 11 of the 14 svg_detection profiles in ``tool_output_registry`` declare no h5ad
            # at all, only a gene table, so requiring one rejects most SVG tools before anything is
            # read. ``_extract_svg_genes`` is the benchmark runner's reader for that same question,
            # reused rather than copied.
            from benchmarks.workflows.benchmark_runner import _extract_svg_genes

            if not svg_ground_truth_path:
                return TrialResult(
                    params={},
                    score=no_score,
                    status="failed",
                    error_message=(
                        "svg_detection needs a curated SVG gene list as ground truth; none was "
                        "configured (svg_ground_truth_path)"
                    ),
                    elapsed_seconds=time.time() - start,
                    output_files=[str(eval_file)] if eval_file else [],
                )
            metrics = evaluate_svg(
                _extract_svg_genes(output_path, eval_file),
                svg_ground_truth_path,
                spatial_h5ad=ground_truth_path,
            )
        elif task_type == "deconvolution":
            # A deconvolution prediction is a proportions table. The evaluator reads output_path as
            # that CSV, so handing it the newest h5ad failed its gate on the HDF5 bytes, and a tool
            # writing only the CSV -- spacexr, card, spotlight, stdeconvolve -- failed earlier on "No
            # output h5ad found": no deconvolution trial could ever score (hunt 2026-09-30,
            # u32-tuning-2). Found and standardised the way the benchmark does it, so an h5ad-only
            # result (cell2location's obsm) is read too.
            located = _locate_proportions(tool_name, output_path, ground_truth_path)
            if isinstance(located, str):
                return TrialResult(
                    params={},
                    score=no_score,
                    status="failed",
                    error_message=located,
                    elapsed_seconds=time.time() - start,
                )
            prediction_file, standardized_file = located
            eval_file = Path(prediction_file)
            metrics = evaluate_tool_output(
                task_type=task_type,
                output_path=prediction_file,
                ground_truth_path=ground_truth_path,
                ground_truth_key=ground_truth_key,
                standardized_file=standardized_file,
            )
        elif eval_file is None:
            return TrialResult(
                params={},
                score=no_score,
                status="failed",
                error_message="No output h5ad found",
                elapsed_seconds=time.time() - start,
            )
        else:
            # The tool's own column, by the name its output profile gives it. Without one the
            # evaluator walks its generic candidate list, where ``mclust`` or ``leiden`` -- a
            # by-product the worker also wrote -- can come before the tool's ``stagate_domain``,
            # so every trial was scored on the wrong column. The benchmark passes the inspector's
            # key; the tuner passed none (hunt 2026-09-30, u33b-bench-scoring-2).
            predicted_key = _profile_prediction_key(tool_name, eval_file) if task_type == "spatial_clustering" else ""
            metrics = evaluate_tool_output(
                task_type=task_type,
                output_path=str(eval_file),
                ground_truth_path=ground_truth_path,
                ground_truth_key=ground_truth_key,
                predicted_key=predicted_key,
            )
            if metrics.get("_ground_truth_column_auto_detected"):
                logger.info(
                    "No ground-truth column was named; the evaluator scored against %r",
                    metrics.get("_ground_truth_column_matched"),
                )

        if primary_metric not in metrics:
            # The evaluation produced no score at all: an error dict (no overlapping spot IDs, shape
            # mismatch), or a mode that cannot compute this metric -- ``evaluate_svg`` returns
            # ``evaluation_mode: no_ground_truth`` and no jaccard when the curated list is missing.
            # This used to fall through to ``metrics.get(primary_metric, 0.0)`` and come back
            # ``status="success"``, and for an error term 0.0 is not a neutral stand-in but the best
            # score there is -- an evaluation that failed out-ranked every one that succeeded. The
            # no-output path above already picks its no-score value by direction; so does this one.
            return TrialResult(
                params={},
                score=no_score,
                secondary_scores={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                status="failed",
                error_message=str(metrics.get("error") or f"evaluator returned no {primary_metric}"),
                elapsed_seconds=time.time() - start,
                output_files=[str(eval_file)] if eval_file else [],
            )

        score = metrics[primary_metric]
        secondary_scores = {}
        if secondary_metric in metrics:
            secondary_scores[secondary_metric] = metrics[secondary_metric]

        # Include all other metrics as secondary
        for k, v in metrics.items():
            if k not in (primary_metric,) and isinstance(v, (int, float)):
                secondary_scores[k] = v

        return TrialResult(
            params={},  # Filled by executor
            score=score,
            secondary_scores=secondary_scores,
            status="success",
            elapsed_seconds=time.time() - start,
            output_files=[str(eval_file)] if eval_file else [],
        )

    except Exception as e:
        logger.warning("Benchmark evaluation failed: %s", e)
        return TrialResult(
            params={},
            score=float("inf") if METRIC_LOWER_IS_BETTER.get(primary_metric, False) else 0.0,
            status="failed",
            error_message=str(e),
            elapsed_seconds=time.time() - start,
        )


def _profile_prediction_key(tool_name: str, output_file: Path) -> str:
    """The clustering column *tool_name*'s output profile names, as present in *output_file*; else ``""``.

    ``prediction_key`` is ``str | list[str]`` (alternatives, in priority order). ``""`` -- no
    profile, or none of its names in the output -- leaves the choice to the evaluator's own search.
    """
    if not tool_name:
        return ""
    from spatialomicsgym.benchmarking.tool_output_registry import get_profile

    profile = get_profile(tool_name)
    key = profile.prediction_key if profile else ""
    candidates = [str(k) for k in (key if isinstance(key, (list, tuple)) else [key]) if k]
    if not candidates:
        return ""
    present = set(_input_obs_columns(str(output_file)))
    return next((c for c in candidates if c in present), "")


def _locate_proportions(
    tool_name: str, output_dir: Path, ground_truth_path: str | None
) -> tuple[str, str | None] | str:
    """``(prediction_file, standardized_file_or_None)`` for a deconvolution trial, or why there is none.

    The benchmark's own reader: ``inspect_tool_output`` picks the prediction through the tool's
    output profile (a CSV, or an h5ad obsm block), and ``standardize_output`` turns it into the
    spot x cell-type CSV the evaluator scores. A failed standardisation leaves the raw file, which
    the evaluator's own gate then judges.
    """
    from spatialomicsgym.benchmarking.output_inspector import inspect_tool_output
    from spatialomicsgym.benchmarking.output_standardizer import standardize_output

    inspection = inspect_tool_output(tool_name, "deconvolution", str(output_dir), ground_truth_path)
    if not inspection.prediction_file or not inspection.is_ready():
        detail = "; ".join(inspection.errors) or f"status {inspection.status}"
        return f"No deconvolution proportions found in the trial output: {detail}"
    standardized = standardize_output(
        task_type="deconvolution",
        prediction_file=inspection.prediction_file,
        prediction_key=inspection.prediction_key,
        output_dir=str(output_dir),
        tool_name=tool_name,
        ground_truth_path=ground_truth_path,
    )
    std_file = standardized.get("standardized_file") if standardized.get("success") else None
    return inspection.prediction_file, std_file or None


def evaluate_proxy_trial(
    task_type: str,
    output_dir: str,
    dataset_path: str | None = None,
    seed: int = 42,
    tool_name: str = "",
) -> TrialResult:
    """Evaluate a trial using proxy metrics (no ground truth).

    Proxy signals:
    - spatial_clustering: silhouette score, cluster count reasonableness
    - svg_detection: number of SVGs found, Moran's I distribution
    - deconvolution: convergence (ELBO), proportion reasonableness

    ``dataset_path`` is the input the trial ran on. It tells the tool's own output columns from the
    ones that came with the data, which is the difference between scoring the search and scoring
    the dataset. ``seed`` is the run's seed, so a trial scores the same twice. ``tool_name`` lets a
    deconvolution trial with no h5ad find its proportions table through the tool's output profile.

    A proxy that finds nothing to score fails the trial, with its reason. It used to come back
    ``status="success"`` at 0.0, so a run that measured nothing passed the candidate filter and was
    persisted as a tuned config (hunt 2026-09-30, u32-tuning-18).
    """
    import time

    start = time.time()
    output_path = Path(output_dir)
    scores: dict[str, float] = {}
    primary_score = 0.0

    try:
        h5ad_files = sorted(
            output_path.rglob("*.h5ad"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        eval_file = h5ad_files[0] if h5ad_files else None

        # Only clustering is scored from an h5ad. An SVG trial is scored from its gene table, and
        # most SVG tools write no h5ad at all; most deconvolution tools write only a proportions CSV.
        # Demanding an h5ad first failed every adaptive trial of those tools (u32-tuning-11).
        if task_type == "svg_detection":
            primary_score, scores = _proxy_svg(str(output_dir))
        elif task_type == "spatial_clustering":
            if eval_file is None:
                scores = {"error": 1.0, "reason": "No output h5ad found"}
            else:
                primary_score, scores = _proxy_clustering(str(eval_file), dataset_path=dataset_path, seed=seed)
        elif task_type == "deconvolution":
            if eval_file is not None:
                primary_score, scores = _proxy_deconvolution(str(eval_file))
            if eval_file is None or "error" in scores:
                located = _locate_proportions(tool_name, output_path, dataset_path)
                if isinstance(located, str):
                    scores = {"error": 1.0, "reason": located}
                else:
                    eval_file = Path(located[1] or located[0])
                    primary_score, scores = _proxy_deconvolution(str(eval_file))

        if "error" in scores:
            return TrialResult(
                params={},
                score=0.0,
                status="failed",
                error_message=str(scores.get("reason") or "the proxy found nothing to score in the trial output"),
                elapsed_seconds=time.time() - start,
                output_files=[str(eval_file)] if eval_file else [],
            )

        return TrialResult(
            params={},
            score=primary_score,
            secondary_scores=scores,
            status="success",
            elapsed_seconds=time.time() - start,
            output_files=[str(eval_file)] if eval_file else [],
        )

    except Exception as e:
        logger.warning("Proxy evaluation failed: %s", e)
        return TrialResult(
            params={},
            score=0.0,
            status="failed",
            error_message=str(e),
            elapsed_seconds=time.time() - start,
        )


#: Substrings that make an obs column a candidate for "the clustering". Unchanged, and still the
#: only thing that makes a column eligible at all.
_CLUSTER_KEYWORDS = ("cluster", "domain", "leiden", "louvain", "pred")

#: Substrings that mark a candidate as a label the data *arrived with* rather than one this trial
#: produced. Consulted only when the input is not readable to say so directly, and only among names
#: the keyword scan already accepted -- so this never sees a bare "annotation", only shapes like
#: ``ground_truth_domain`` or ``manual_cluster``.
_ANNOTATION_MARKERS = ("ground_truth", "groundtruth", "gt_", "_gt", "true_", "_true", "manual", "expert")


def _input_obs_columns(dataset_path: str | None) -> tuple[str, ...]:
    """The obs column names of the dataset a trial started from; empty when they cannot be read.

    Backed, because only the names are wanted and the input can be multi-GB, and through
    ``read_h5ad_backed`` so the HDF5 handle is released -- a tuning run reads one input once per
    trial. ``dataset_path`` is not always an ``.h5ad`` (a Visium directory, a path that has since
    moved), and an input that cannot be read is no evidence rather than an error.
    """
    if not dataset_path:
        return ()
    try:
        from spatialomicsgym.utils.file_io import read_h5ad_backed

        with read_h5ad_backed(dataset_path) as adata:
            return tuple(str(c) for c in adata.obs.columns)
    except Exception as exc:  # any unreadable input means "no evidence", not an error
        logger.debug("Could not read input obs columns from %s: %s", dataset_path, exc)
        return ()


def _pick_prediction_column(obs_columns, input_columns=()) -> str | None:
    """The obs column holding *this trial's* clustering, out of everything the output carries.

    A spatial-clustering tool writes its labels into the AnnData it was handed and saves the whole
    object, so the output holds the input's obs columns first and the tool's appended after them.
    The keyword scan matches ``ground_truth_domain``, and a ``leiden`` left over from the user's own
    QC, exactly as readily as the tool's ``pred_domain`` -- and taking the first match scored that
    same unchanging column for every trial in the search. Identical silhouettes, and the "best"
    parameters are whichever configuration happened to be tried first.

    ``input_columns`` is what the trial started from, which the executor already had and never
    passed on: anything in it is by construction not this trial's output. A tool that overwrote a
    column in place leaves nothing after that subtraction, and then the full candidate list is all
    there is to go on -- the same answer as before, for the case where the input cannot break the
    tie.
    """
    candidates = [c for c in obs_columns if any(kw in str(c).lower() for kw in _CLUSTER_KEYWORDS)]
    if not candidates:
        return None

    known = set(input_columns or ())
    fresh = [c for c in candidates if c not in known]
    if fresh:
        candidates = fresh

    unannotated = [c for c in candidates if not any(m in str(c).lower() for m in _ANNOTATION_MARKERS)]
    if unannotated:
        candidates = unannotated

    return candidates[0]


def _proxy_clustering(
    h5ad_path: str, dataset_path: str | None = None, seed: int = 42
) -> tuple[float, dict[str, float]]:
    """Proxy metrics for clustering quality without ground truth.

    Primary: silhouette score (normalized to 0-1).
    Secondary: cluster count reasonableness, weighted combination.

    ``dataset_path`` is the input this trial ran on, used only to tell the tool's labels from the
    ones that came with the data (see :func:`_pick_prediction_column`). ``seed`` makes the
    subsample below reproducible, so one output scores one way.

    Handles edge cases:
    - No cluster column at all: an ``error`` entry, nothing to score (the trial fails)
    - Only 1 cluster: penalizes heavily (score=0.1)
    - Missing embeddings: falls back to cluster_score only
    - Silhouette computation failure: falls back gracefully
    """
    import anndata as ad
    import numpy as np

    adata = ad.read_h5ad(h5ad_path)
    scores: dict[str, float] = {}

    pred_col = _pick_prediction_column(adata.obs.columns, _input_obs_columns(dataset_path))

    if pred_col is None:
        return 0.0, {"error": 1.0, "reason": "no cluster column found"}

    labels = adata.obs[pred_col].astype(str)
    n_clusters = labels.nunique()
    scores["n_clusters"] = float(n_clusters)

    # Hard penalty for degenerate results (1 cluster = no clustering)
    if n_clusters < 2:
        scores["cluster_reasonableness"] = 0.0
        return 0.1, scores

    # Cluster count reasonableness
    if 2 <= n_clusters <= 30:
        cluster_score = 1.0
    elif n_clusters <= 50:
        cluster_score = 0.5
    else:
        cluster_score = 0.2
    scores["cluster_reasonableness"] = cluster_score

    # Silhouette score
    try:
        from sklearn.metrics import silhouette_score

        if "X_pca" in adata.obsm:
            embedding = np.asarray(adata.obsm["X_pca"][:, :30])
        elif "spatial" in adata.obsm:
            embedding = np.asarray(adata.obsm["spatial"])
        else:
            embedding = None

        if embedding is not None:
            # Subsample for speed -- from `seed`, not from numpy's global state. Above 5000 spots
            # the draw decides the silhouette, so one trial output scored two different ways and
            # the search ranked configurations on the difference between two subsamples.
            # TuningConfig has carried `seed` all along, and the executor holds the config where
            # the evaluator is called.
            n = min(5000, len(labels))
            if len(labels) > n:
                idx = np.random.default_rng(seed).choice(len(labels), n, replace=False)
            else:
                idx = np.arange(len(labels))
            sil = silhouette_score(embedding[idx], labels.values[idx])
            if np.isnan(sil) or np.isinf(sil):
                primary_score = cluster_score * 0.5
            else:
                scores["silhouette"] = float(sil)
                sil_norm = (sil + 1) / 2  # [-1,1] -> [0,1]
                primary_score = sil_norm * 0.7 + cluster_score * 0.3
        else:
            primary_score = cluster_score * 0.5
    except Exception:
        primary_score = cluster_score * 0.5

    return float(max(0, min(1, primary_score))), scores


def _proxy_svg(output_dir: str) -> tuple[float, dict[str, float]]:
    """Proxy metrics for SVG detection quality without ground truth.

    Uses number of SVGs found and their Moran's I values. An ``error`` entry when the output holds
    no gene table at all: nothing was measured, which is not the same as a table that names no
    significant gene -- that one is a real 0.
    """
    import pandas as pd

    from spatialomicsgym.benchmarking.output_standardizer import _pick_significance_column
    from spatialomicsgym.utils.file_io import sniff_tabular_sep

    scores: dict[str, float] = {}
    output_path = Path(output_dir)

    # Count SVGs from CSV output. The compressed patterns are last so the precedence between plain
    # files is the one this always had: a `.csv.gz` result was invisible to the glob, and every
    # trial of a gzipping tool scored the 0.1 floor -- the whole search flat (R80/T).
    csv_files = [path for pattern in ("*.csv", "*.tsv", "*.csv.gz", "*.tsv.gz") for path in output_path.rglob(pattern)]
    n_svgs = 0
    gene_table_seen = False

    for csv_path in csv_files:
        try:
            # From the bytes, not the name. A tool writing tab-separated content under a `.csv`
            # name is common enough that sniff_tabular_sep exists for it (R62, R80/F); read with
            # the wrong separator the header is one column, the gene-column check below misses,
            # and the trial scores as if it had produced nothing.
            df = pd.read_csv(csv_path, sep=sniff_tabular_sep(csv_path))
            has_gene_column = any(col.lower() in ("gene", "gene_name", "g") for col in df.columns)
            gene_table_seen = gene_table_seen or has_gene_column
            if len(df) > 0 and has_gene_column:
                # Which column says "significant" is a settled question -- _pick_significance_column
                # answers it for the benchmark standardizer, the output inspector and the
                # post-analysis reviewer, and this was a fifth, shorter, differently-ordered copy.
                # Two ways it scored the wrong number. It iterated the FILE's column order and
                # stopped at the first candidate with any hit, so a table writing raw `pval` before
                # `fdr` -- 45 of the recorded SVG files do -- was thresholded on the uncorrected
                # value. And eight of the sixteen canonical names were not in it at all (`qval`,
                # `p_adj`, `adjusted_pvalue`, squidpy's `pval_norm_fdr_bh`, ...), which fell through
                # to len(df): the gene list the tool screened, a number its parameters cannot move,
                # scored at the top of the range because min(1.0, n/500) saturates.
                #
                # The fallback to len(df) stays, for tables that carry no p-value at all, where
                # length is all there is to read.
                sig_col = _pick_significance_column(df)
                if sig_col is None:
                    n_svgs = len(df)
                else:
                    n_svgs = int((pd.to_numeric(df[sig_col], errors="coerce") < 0.05).sum())
                break
        except Exception:
            continue

    scores["n_svgs"] = float(n_svgs)

    if not gene_table_seen:
        # Nothing was measured. Since the proxy stopped demanding an h5ad (u32-tuning-11), a trial
        # that wrote no gene table -- an empty directory, a log, a worker that exited 0 having done
        # nothing -- came back "success" at 0.0, and a search of such trials persisted a 0.0
        # best_config as tuned. The error entry fails the trial instead (hunt 2026-09-30,
        # u32-tuning-18). A table that has the gene column and no significant gene is still a 0.
        scores["svg_reasonableness"] = 0.0
        scores["error"] = 1.0
        scores["reason"] = (
            "no SVG gene table in the trial output: none of its "
            f"{len(csv_files)} .csv/.tsv file(s) has a gene, gene_name or g column"
        )
        return 0.0, scores

    # Reasonableness: typical SVG counts are 100-3000. Higher is better and the search maximizes,
    # so the ramp must not be interrupted by a constant larger than the values it hands off to.
    # There used to be a third band -- `else: 0.1  # Too few` for n < 10 -- described as the bottom
    # of the range. It was not: the ramp below starts at 0.02 (n=10), so a trial detecting nothing
    # scored five times a trial detecting ten genes, and every count up to 49 lost to zero. #206
    # removed the same preference at the other end of the table (a zero-detection trial scoring the
    # MAXIMUM, via the count) and in doing so made this region reachable for the first time; the
    # floor itself dates to the module's first commit. The ramp already covers n < 10 continuously
    # and reaches 0.0 exactly where the trial found nothing, so the special case only inverted it.
    #
    # The `> 5000` penalty is left as it was. It still outranks the ramp for counts below 250 -- a
    # second, narrower inversion -- but lowering it is a calibration judgement no evidence here
    # settles, and raising it would make an over-permissive trial more attractive than it is today.
    if n_svgs > 5000:
        primary_score = 0.5  # Suspiciously many
    else:
        primary_score = min(1.0, n_svgs / 500)  # Saturates at 500 SVGs

    scores["svg_reasonableness"] = primary_score
    return primary_score, scores


def _proxy_deconvolution(h5ad_path: str) -> tuple[float, dict[str, float]]:
    """Proxy metrics for deconvolution quality without ground truth.

    Uses proportion distribution quality as proxy. *h5ad_path* is an h5ad carrying the proportions
    in obsm or obs, or a proportions table (spots x cell types) -- most deconvolution tools write
    only the table (u32-tuning-11); the parameter keeps its name. An ``error`` entry when it holds no
    proportions: nothing to score.
    """
    import numpy as np

    scores: dict[str, float] = {}

    if not str(h5ad_path).endswith(".h5ad"):
        import pandas as pd

        from spatialomicsgym.benchmarking.output_inspector import _proportions_index_col, _tabular_sep

        table = Path(h5ad_path)
        sep = _tabular_sep(table)
        frame = pd.read_csv(table, index_col=_proportions_index_col(table, sep), sep=sep)
        numeric = frame.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return 0.0, {"error": 1.0, "reason": f"{table.name} has fewer than two numeric columns"}
        proportions = numeric.to_numpy()
    else:
        import anndata as ad

        adata = ad.read_h5ad(h5ad_path)

        # Look for deconvolution results in obsm
        deconv_key = None
        for key in adata.obsm:
            if any(kw in key.lower() for kw in ["deconv", "proportion", "abundance", "cell_type"]):
                deconv_key = key
                break

        if deconv_key is None:
            # Check obs for proportion columns
            proportion_cols = [c for c in adata.obs.columns if "proportion" in c.lower() or "fraction" in c.lower()]
            if not proportion_cols:
                return 0.0, {"error": 1.0, "reason": "no proportions in obsm or obs"}
            proportions = adata.obs[proportion_cols].values
        else:
            proportions = adata.obsm[deconv_key]

    if hasattr(proportions, "toarray"):
        proportions = proportions.toarray()
    proportions = np.array(proportions, dtype=float)
    proportions = np.nan_to_num(proportions, nan=0.0, posinf=0.0, neginf=0.0)

    # Check proportions sum to ~1
    row_sums = proportions.sum(axis=1)
    sum_quality = 1.0 - np.mean(np.abs(row_sums - 1.0))
    scores["proportion_sum_quality"] = float(max(0, sum_quality))

    # Check for reasonable number of cell types
    n_celltypes = proportions.shape[1]
    scores["n_celltypes"] = float(n_celltypes)

    # Entropy of mean proportions (higher = more diverse, generally better)
    mean_props = proportions.mean(axis=0)
    mean_props = mean_props[mean_props > 0]
    entropy = float(-np.sum(mean_props * np.log(mean_props + 1e-10)))
    max_entropy = float(np.log(n_celltypes + 1e-10))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    scores["proportion_entropy"] = normalized_entropy

    # Non-negative check
    non_neg_frac = float(np.mean(proportions >= -0.01))
    scores["non_negative_fraction"] = non_neg_frac

    primary_score = sum_quality * 0.4 + normalized_entropy * 0.3 + non_neg_frac * 0.3
    return float(max(0, min(1, primary_score))), scores
