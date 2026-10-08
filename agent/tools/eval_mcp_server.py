#!/usr/bin/env python3
"""
Evaluation MCP server for SpatialOmicsLab.

Provides tools to:
1. Evaluate a single tool's output against gold standard or self-assessment
2. Collect results from multiple tool runs
3. Generate benchmark summary reports

This runs in the spatialomicsgym_env environment alongside other MCP servers.
"""

import traceback
from typing import Any

from base_mcp import create_mcp

mcp = create_mcp("spatialomicsgym-eval")


def _eval_error(task: str, error_msg: str) -> dict[str, Any]:
    """Return a standard error dict for the eval tool."""
    return {
        "status": "error",
        "tool": "eval",
        "task": task,
        "error": error_msg,
        "traceback": traceback.format_exc(),
    }


def _degenerate_clustering_gold_standard(metrics: dict[str, Any]) -> list[str]:
    """Warn when the gold standard has no partition in it for the prediction to agree with.

    Two shapes qualify, and both put a perfect-looking number on the scorecard. Every true class a
    singleton makes ``completeness`` 1.0 for any prediction whatsoever; a single true class makes
    ``homogeneity`` 1.0 the same way. Either one sits beside an ARI near zero, which reads as "the
    method over-split the tissue" -- a claim about the method that this gold standard cannot
    support. The usual cause is a truth file parsed with the wrong delimiter, but the check is on
    the shape rather than the cause, so it also catches a file that is simply the wrong file.

    Returns an empty list for the self-assessment branch, whose metrics carry no true-cluster count,
    and below three samples, where the degenerate shapes and the real ones are the same shape.
    """
    n_true = metrics.get("n_true_clusters")
    n_samples = metrics.get("n_samples")
    if not isinstance(n_true, int) or not isinstance(n_samples, int) or n_samples < 3:
        return []
    if n_true == n_samples:
        return [
            f"WARNING: the gold standard has one distinct label per sample ({n_true} labels for "
            f"{n_samples} samples), so it defines no partition to agree with -- completeness="
            f"{metrics.get('completeness')} is 1.0 by construction here, for any prediction. Check "
            f"the true-label file's delimiter and which column holds the labels."
        ]
    if n_true == 1:
        return [
            f"WARNING: the gold standard puts all {n_samples} samples in one cluster, so it defines "
            f"no partition to agree with -- homogeneity={metrics.get('homogeneity')} is 1.0 by "
            f"construction here, for any prediction. Check the true-label file's label column."
        ]
    return []


#: How many placeholder names ``use_mock`` mixes into the pool it samples the SVG gold standard from.
#: Named because the warning below states the arithmetic this number sets, and the two must not drift.
_SVG_MOCK_DECOYS = 200


def _self_referential_svg_gold_standard(metrics: dict[str, Any], n_predicted: int, n_mock_svg: int) -> list[str]:
    """Warn that a mock SVG gold standard was drawn from the prediction it is about to score.

    ``use_mock`` samples ``n_mock_svg`` names out of the tool's own predicted genes plus 200
    placeholders, so with G predicted genes the expected overlap is ``n_mock_svg * G/(G+200)``, hence
    ``recall ~= G/(G+200)`` and ``precision ~= n_mock_svg/(G+200)``. Both are functions of how many
    genes the tool named and of nothing else. Measured, three 500-gene lists sharing not one symbol
    score identically at jaccard 0.1321 / precision 0.1400 / recall 0.7000, and recall climbs 0.21 to
    0.88 across 50 to 2000 predicted genes -- the longer list wins.

    Unlike the clustering mock, whose random labels give an obviously failed ARI near zero, this one
    produces flattering mid-range numbers that read as a real benchmark, which is what makes silence
    about it publishable rather than merely useless.
    """
    if metrics.get("gold_standard_source") != "mock":
        return []
    return [
        "WARNING: no gold standard was given, so these overlap metrics were scored against a mock "
        f"one -- {n_mock_svg} names drawn from the tool's own {n_predicted} predicted genes plus "
        f"{_SVG_MOCK_DECOYS} placeholders. recall={metrics.get('recall')} is then a function of how "
        f"many genes the tool predicted (about {n_predicted}/{n_predicted + _SVG_MOCK_DECOYS}) and "
        "not of which genes it predicted: two different gene lists of this length score the same. "
        "Not a measurement of the method."
    ]


def _degenerate_deconvolution_input(metrics: dict[str, Any], n_celltypes: int) -> list[str]:
    """Warn when a proportions table has no cell types, or no metric could be computed at all."""
    warnings = []
    if n_celltypes == 0:
        warnings.append(
            "WARNING: the proportions table has no cell-type columns, so every number below was "
            "computed over an empty matrix. Check the file's delimiter and whether its first column "
            "really is the spot index."
        )
    error = metrics.get("error")
    if error:
        warnings.append(f"WARNING: no deconvolution metric could be computed -- {error}")
    # ``mean_entropy`` is reported as "how mixed the average spot is", and it is only that if each
    # row is a distribution. When it is not, the metric is withheld rather than published wrong, and
    # this is where the reader is told -- `analysis` is the field the ReAct observation shows.
    entropy_note = metrics.get("mean_entropy_note")
    if entropy_note:
        warnings.append(f"WARNING: {entropy_note}")
    return warnings


#: How many dropped cell-type names a warning spells out before it starts counting them instead. A
#: 22-cell-type MERFISH run drops as many as ten, and a wall of names buries the sentence about them.
_MAX_NAMED_CELLTYPES = 4


def _named_celltypes(names: list[str]) -> str:
    shown = ", ".join(repr(n) for n in names[:_MAX_NAMED_CELLTYPES])
    rest = len(names) - _MAX_NAMED_CELLTYPES
    return f"{shown} and {rest} more" if rest > 0 else shown


def _mean_over_a_shorter_denominator(metrics: dict[str, Any]) -> list[str]:
    """Warn when mean_pearson_r averaged over fewer cell types than the run has, and say why.

    A cell type leaves that average when either column is constant, and the two causes are not the
    same kind of thing. A constant ground truth is a property of the dataset -- unscoreable for
    everyone, so every tool on it loses the same column and the ranking between them stays fair. A
    constant prediction is a property of the tool: it captured none of that cell type's variation,
    and dropping the column takes its own worst case out of its own average.

    On the twelve recorded slideseqv2_aorta_deconvolution runs no ground-truth column is constant,
    so every drop there is the second kind. ``spatialomicsgym/agent/empirical_leaderboard.py`` ranks
    deconvolution tools on mean_pearson_r, and it puts the tool that averaged over 6 of the 7 cell
    types first, ahead of two that averaged over all 7.
    """
    n_ct = metrics.get("n_celltypes")
    n_corr = metrics.get("n_celltypes_correlated")
    if not isinstance(n_ct, int) or not isinstance(n_corr, int) or n_corr >= n_ct:
        return []

    by_tool = metrics.get("celltypes_constant_in_prediction") or []
    by_dataset = metrics.get("celltypes_constant_in_truth") or []
    parts = [f"WARNING: mean_pearson_r is an average over {n_corr} of the {n_ct} cell types, not all {n_ct}."]
    if by_tool:
        parts.append(
            f"Left out because this tool predicted a constant value: {_named_celltypes(by_tool)}. Those "
            "are cell types whose variation it captured none of, so its average excludes its own "
            "weakest columns and is not comparable with an average another tool took over more of."
        )
    if by_dataset:
        parts.append(
            "Constant in the ground truth, so no tool can be scored on them and every tool on this "
            f"dataset loses them alike: {_named_celltypes(by_dataset)}."
        )
    return [" ".join(parts)]


def _lead_with(warnings: list[str], analysis_text: str) -> str:
    """Put the caveats in front of the numbers -- `analysis` is the field a reader is shown."""
    return " ".join([*warnings, analysis_text]) if warnings else analysis_text


@mcp.tool()
def evaluate_clustering_result(
    annotated_h5ad: str,
    pred_key: str,
    true_key: str | None = None,
    true_labels_csv: str | None = None,
    n_mock_clusters: int = 7,
    use_mock: bool = False,
) -> dict[str, Any]:
    """
    Evaluate a spatial domain / clustering result.

    Provide either true_key (column in h5ad) or true_labels_csv (external CSV),
    or set use_mock=True for benchmarking with mock labels.

    Returns ARI, NMI, homogeneity, completeness, v_measure, cluster balance metrics.
    """
    try:
        import os
        import sys

        # Guarded: this server stays resident across every eval call of a benchmark sweep, so an
        # unconditional insert would append the same directory once per call, forever.
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        import anndata as ad
        import numpy as np
        import pandas as pd
        from eval_metrics import eval_clustering, mock_clustering_labels
        from worker_utils import sniff_tabular_sep

        adata = ad.read_h5ad(annotated_h5ad)
        pred = adata.obs[pred_key].values

        if true_key and true_key in adata.obs.columns:
            metrics = eval_clustering(pred, adata.obs[true_key].values)
            metrics["gold_standard_source"] = f"adata.obs['{true_key}']"
        elif true_labels_csv:
            # A tab-delimited annotation -- the ordinary spelling for a GEO supplementary table --
            # read with a hardcoded comma gave one column of whole lines, so every spot became its
            # own singleton "true cluster". On 30 spots whose labels matched the prediction exactly
            # that published ARI 0.0 beside completeness 1.0, at status ok, naming the file as the
            # gold standard. The reader concludes the method over-split; the file had tabs.
            df = pd.read_csv(true_labels_csv, sep=sniff_tabular_sep(true_labels_csv))
            true = df.iloc[:, -1].values
            metrics = eval_clustering(pred, true)
            metrics["gold_standard_source"] = true_labels_csv
        elif use_mock:
            true = mock_clustering_labels(len(pred), n_mock_clusters)
            metrics = eval_clustering(pred, true)
            metrics["gold_standard_source"] = "mock"
        else:
            # Self-assessment only
            from collections import Counter

            counts = Counter(pred)
            sizes = list(counts.values())
            total = sum(sizes)
            metrics = {
                "n_clusters": len(counts),
                "balance_ratio": round(max(sizes) / min(sizes), 2) if min(sizes) > 0 else float("inf"),
                "entropy": round(float(-sum((s / total) * np.log2(s / total) for s in sizes if s > 0)), 4),
                "gold_standard_source": "none (self-assessment)",
            }

        n_cells = adata.n_obs
        analysis_text = (
            f"Clustering evaluation on {n_cells} cells with pred_key='{pred_key}'. "
            f"Gold standard: {metrics.get('gold_standard_source', 'N/A')}."
        )
        # eval_clustering returns "ARI"/"NMI" in upper case, so the lower-case spellings this used to
        # test for were never present and the clause never appended: every scored clustering run's
        # prose ended at "Gold standard: ...", with the ARI reachable only in `summary`. `analysis` is
        # the field the ReAct observation and the CLI show, so the headline was effectively unpublished.
        if "ARI" in metrics:
            analysis_text += f" ARI={metrics['ARI']:.4f}, NMI={metrics.get('NMI', 'N/A')}."

        gold_warnings = _degenerate_clustering_gold_standard(metrics)

        return {
            "status": "ok",
            "tool": "eval",
            "task": "clustering",
            "warnings": gold_warnings,
            "data": {"n_cells": n_cells, "n_genes": adata.n_vars},
            "output_files": {},
            "params": {
                "annotated_h5ad": annotated_h5ad,
                "pred_key": pred_key,
                "true_key": true_key,
                "true_labels_csv": true_labels_csv,
                "use_mock": use_mock,
            },
            "summary": metrics,
            "analysis": _lead_with(gold_warnings, analysis_text),
        }
    except Exception as e:
        return _eval_error("clustering", str(e))


@mcp.tool()
def evaluate_svg_result(
    predicted_genes_csv: str,
    true_genes_csv: str | None = None,
    true_genes: list[str] | None = None,
    annotated_h5ad: str | None = None,
    use_mock: bool = False,
    n_mock_svg: int = 100,
) -> dict[str, Any]:
    """
    Evaluate a spatially variable gene (SVG) result.

    Provide true_genes (list) or true_genes_csv, or set use_mock=True.
    If annotated_h5ad is provided, also computes Moran's I.

    Returns: jaccard, precision, recall, F1, overlap genes, and optionally Moran's I.
    """
    try:
        import os
        import sys

        # Guarded: this server stays resident across every eval call of a benchmark sweep, so an
        # unconditional insert would append the same directory once per call, forever.
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        import pandas as pd
        from eval_metrics import eval_morans_i, eval_svg_overlap, mock_svg_gene_list
        from worker_utils import sniff_tabular_sep

        # Both sides are read the same way, and neither may assume a comma. A tab-delimited file on
        # either side parses to one column of whole lines, which overlaps nothing -- so an identical
        # ten-gene answer scored jaccard 0.0 while n_predicted and n_true both stayed correct. Every
        # count on that scorecard is plausible and every score on it is fabricated.
        df = pd.read_csv(predicted_genes_csv, sep=sniff_tabular_sep(predicted_genes_csv))
        gene_col = [c for c in df.columns if c.lower() in ("gene", "gene_name", "genes", "index")]
        pred_genes = df[gene_col[0]].tolist() if gene_col else df.iloc[:, 0].tolist()

        metrics: dict[str, Any] = {"n_predicted": len(pred_genes)}

        # Overlap metrics. Which gold standard produced them is recorded on every path, not only the
        # mock one: the clustering and deconvolution branches of this same file both name theirs, and
        # a jaccard whose truth set is unstated cannot be read at all.
        gold_genes = true_genes or []
        gold_source = f"true_genes (explicit list of {len(gold_genes)} genes)" if gold_genes else ""
        if not gold_genes and true_genes_csv:
            gdf = pd.read_csv(true_genes_csv, sep=sniff_tabular_sep(true_genes_csv))
            gold_genes = gdf.iloc[:, 0].tolist()
            if gold_genes:
                gold_source = true_genes_csv
        if not gold_genes and use_mock:
            all_genes = pred_genes + [f"MOCK_{i}" for i in range(_SVG_MOCK_DECOYS)]
            gold_genes = mock_svg_gene_list(all_genes, n_mock_svg)
            gold_source = "mock"
        metrics["gold_standard_source"] = gold_source or "none (no overlap metrics)"

        if gold_genes:
            metrics.update(eval_svg_overlap(pred_genes, gold_genes))

        # Moran's I
        if annotated_h5ad and os.path.isfile(annotated_h5ad):
            try:
                import anndata as ad

                adata = ad.read_h5ad(annotated_h5ad)
                morans = eval_morans_i(adata, pred_genes[:50])
                metrics["morans_i"] = morans
            except Exception as e:
                metrics["morans_i_error"] = str(e)

        analysis_text = (
            f"SVG evaluation: {len(pred_genes)} predicted genes. Gold standard: {metrics['gold_standard_source']}."
        )
        if "jaccard" in metrics:
            analysis_text += (
                f" Jaccard={metrics['jaccard']:.4f}, Precision={metrics.get('precision', 'N/A')}, "
                f"Recall={metrics.get('recall', 'N/A')}."
            )

        gold_warnings = _self_referential_svg_gold_standard(metrics, len(pred_genes), n_mock_svg)

        return {
            "status": "ok",
            "tool": "eval",
            "task": "svg",
            "warnings": gold_warnings,
            "data": {"n_predicted_genes": len(pred_genes)},
            "output_files": {},
            "params": {
                "predicted_genes_csv": predicted_genes_csv,
                "true_genes_csv": true_genes_csv,
                "annotated_h5ad": annotated_h5ad,
                "use_mock": use_mock,
            },
            "summary": metrics,
            "analysis": _lead_with(gold_warnings, analysis_text),
        }
    except Exception as e:
        return _eval_error("svg", str(e))


@mcp.tool()
def evaluate_deconvolution_result(
    predicted_proportions_csv: str,
    true_proportions_csv: str | None = None,
    use_mock: bool = False,
) -> dict[str, Any]:
    """
    Evaluate a deconvolution result.

    Provide true_proportions_csv or set use_mock=True.
    Returns RMSE, MAE, Pearson r, JSD.
    """
    try:
        import os
        import sys

        # Guarded: this server stays resident across every eval call of a benchmark sweep, so an
        # unconditional insert would append the same directory once per call, forever.
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        import pandas as pd
        from eval_metrics import eval_deconvolution, mean_row_entropy, mock_deconv_proportions
        from worker_utils import sniff_tabular_sep

        # `index_col=0` consumes the single column a hardcoded-comma read of a tab-delimited table
        # produces, so both sides come back (n_spots, 0). The shape check below then passes -- the
        # two frames agree, at zero cell types -- and the run is published at status ok with
        # n_celltypes 0. The self-assessment branch reports the same 0 with no comparison at all.
        pred_df = pd.read_csv(predicted_proportions_csv, index_col=0, sep=sniff_tabular_sep(predicted_proportions_csv))
        pred = pred_df.values

        if true_proportions_csv:
            true_df = pd.read_csv(true_proportions_csv, index_col=0, sep=sniff_tabular_sep(true_proportions_csv))
            true = true_df.values
            ct_names = list(true_df.columns)
            source = true_proportions_csv
        elif use_mock:
            true, ct_names = mock_deconv_proportions(pred.shape[0], pred.shape[1])
            source = "mock"
        else:
            # Self-assessment only
            n_spots, n_ct = pred.shape
            # Entropy is only defined over a distribution, and the tool's own CSV is not guaranteed
            # to be one -- rows that do not sum to 1 return a different quantity, and a negative
            # cell returns NaN. ``mean_row_entropy`` holds that rule for both copies of this code.
            entropy, entropy_note = mean_row_entropy(pred)
            metrics = {
                "n_spots": n_spots,
                "n_celltypes": n_ct,
                "mean_entropy": entropy,
                "gold_standard_source": "none (self-assessment)",
            }
            if entropy_note:
                metrics["mean_entropy_note"] = entropy_note
            self_warnings = _degenerate_deconvolution_input(metrics, n_ct)
            return {
                "status": "ok",
                "tool": "eval",
                "task": "deconvolution",
                "warnings": self_warnings,
                "data": {"n_spots": n_spots, "n_celltypes": n_ct},
                "output_files": {},
                "params": {
                    "predicted_proportions_csv": predicted_proportions_csv,
                    "use_mock": use_mock,
                },
                "summary": metrics,
                "analysis": _lead_with(
                    self_warnings, f"Deconvolution self-assessment: {n_spots} spots, {n_ct} cell types."
                ),
            }

        if pred.shape != true.shape:
            return _eval_error("deconvolution", f"Shape mismatch: pred={pred.shape}, true={true.shape}")

        metrics = eval_deconvolution(pred, true, ct_names)
        metrics["gold_standard_source"] = source

        analysis_text = (
            f"Deconvolution evaluation: {pred.shape[0]} spots, {pred.shape[1]} cell types. Gold standard: {source}."
        )
        # Same case drift as the clustering branch above: eval_deconvolution returns "RMSE". The test
        # is still needed -- eval_deconvolution returns an {"error": ...} dict with no RMSE when no
        # ground-truth row is non-zero -- but it has to name the key the producer actually writes.
        if "RMSE" in metrics:
            analysis_text += f" RMSE={metrics['RMSE']:.4f}."

        scored_warnings = _degenerate_deconvolution_input(metrics, int(pred.shape[1]))
        scored_warnings += _mean_over_a_shorter_denominator(metrics)

        return {
            "status": "ok",
            "tool": "eval",
            "task": "deconvolution",
            "warnings": scored_warnings,
            "data": {"n_spots": pred.shape[0], "n_celltypes": pred.shape[1]},
            "output_files": {},
            "params": {
                "predicted_proportions_csv": predicted_proportions_csv,
                "true_proportions_csv": true_proportions_csv,
                "use_mock": use_mock,
            },
            "summary": metrics,
            "analysis": _lead_with(scored_warnings, analysis_text),
        }
    except Exception as e:
        return _eval_error("deconvolution", str(e))


@mcp.tool()
def generate_benchmark_report(
    results_dir: str,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Generate a benchmark summary report from collected tool results.

    Reads *_result.json files from results_dir and produces a
    Markdown + JSON report with rankings and comparisons.
    """
    try:
        import json
        import os
        import sys

        # Guarded: this server stays resident across every eval call of a benchmark sweep, so an
        # unconditional insert would append the same directory once per call, forever.
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from result_collector import ResultCollector

        out_dir = output_dir or results_dir
        collector = ResultCollector(output_dir=out_dir)

        # Load existing results
        loaded = 0
        for fname in sorted(os.listdir(results_dir)):
            if fname.endswith("_result.json"):
                with open(os.path.join(results_dir, fname)) as f:
                    result = json.load(f)
                collector.results.append(result)
                loaded += 1

        if not loaded:
            return _eval_error("benchmark", f"No *_result.json files found in {results_dir}")

        report_path = collector.save_report()
        report = collector.generate_report()

        report_json_path = os.path.join(out_dir, "benchmark_report.json")
        analysis_text = (
            f"Benchmark report generated from {loaded} result files. "
            f"{report['n_success']} succeeded, {report['n_failed']} failed. "
            f"Task types: {', '.join(report['by_task_type'].keys())}."
        )

        return {
            "status": "ok",
            "tool": "eval",
            "task": "benchmark",
            "data": {"n_results_loaded": loaded},
            "output_files": {
                "report_json": report_json_path,
                "report_markdown": report_path,
            },
            "params": {
                "results_dir": results_dir,
                "output_dir": out_dir,
            },
            "summary": {
                "n_tools": report["n_tools_run"],
                "n_success": report["n_success"],
                "n_failed": report["n_failed"],
                "task_types": list(report["by_task_type"].keys()),
            },
            "analysis": analysis_text,
        }
    except Exception as e:
        return _eval_error("benchmark", str(e))


if __name__ == "__main__":
    mcp.run()
