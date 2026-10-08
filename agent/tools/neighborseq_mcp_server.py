#!/usr/bin/env python3
"""Neighborseq cell-cell interaction network MCP wrapper for SpatialOmicsLab."""

import os
import sys
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "neighborseq"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "NEIGHBORSEQ",
    "/opt/conda/envs/neighborseq/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/neighborseq_worker.R",
)

mcp = create_mcp(TOOL_NAME)

#: Neighbor-seq's cost, estimated before the worker starts (F-61: the Zhuang mini block, 2,586 cells x 17 domains,
#: took 1,729 s with nothing said before the run, and the 10-section object was not run for cost). FindAllMarkers on every cell, the predictions and the enrichment
#: shuffles grow with cells x clusters; the xgboost forest is grown per class (200 trees each) on 100 artificial
#: profiles per class, so it grows with classes^2, and the classes are every cluster and every pair of clusters,
#: k(k+3)/2 for k clusters. So a run costs about
#: ``FIXED + CELL_CLUSTER * cells * clusters + CLASS2 * classes^2`` seconds.
#: Calibrated 2026-10-06 on this host (Neighborseq 0.1.0, 96 cores, xgboost on all of them) by least squares on five
#: worker runs on real Zhuang counts (1,122 genes): the mini block in 8 (merged) and its own 17 domains, 2,589 and 2,586
#: cells: 58.9 and 138.8 s; 10,000 and 20,000 cells of the 10-section object with 17 (merged) domains: 189.9 and 252.2 s;
#: 10,000 cells with its 30 domains: 863.6 s. The model gives 55.8 / 142.5 / 189.3 / 252.5 / 863.2 s, and for the
#: 10-section object (309,599 cells x 30 domains) about 4,200 s. The record's 1,729 s on the mini block was not
#: reproduced: the same input ran in 137 s (Task H2) and 139-158 s here, so that run was most likely sharing the box.
NSEQ_FIXED_S = 42.0
NSEQ_CELL_CLUSTER_S = 3.7e-4
NSEQ_CLASS2_S = 2.9e-3
#: The budget a run's estimate must fit unless ``max_estimated_s`` or ``SOG_NEIGHBORSEQ_MAX_SECONDS`` says otherwise.
DEFAULT_MAX_ESTIMATED_S = 1800.0
BUDGET_ENV = "SOG_NEIGHBORSEQ_MAX_SECONDS"
_MISSING_LABELS = {"", "nan", "na", "none", "null"}


def neighborseq_classes(n_clusters: int) -> int:
    """Neighborseq's class count for ``n_clusters`` clusters: each cluster, and each pair (homotypic included)."""
    k = max(0, int(n_clusters))
    return k + k * (k + 1) // 2


def neighborseq_seconds(n_cells: int, n_clusters: int) -> float:
    """The estimated wall time, in seconds, of one Neighbor-seq run on ``n_cells`` cells in ``n_clusters`` clusters."""
    n, k = float(max(0, n_cells)), float(max(1, n_clusters))
    c = float(neighborseq_classes(int(k)))
    return NSEQ_FIXED_S + NSEQ_CELL_CLUSTER_S * n * k + NSEQ_CLASS2_S * c * c


def _budget(max_estimated_s) -> tuple[float, str]:
    """``(seconds, source)``: the argument, else ``SOG_NEIGHBORSEQ_MAX_SECONDS``, else the default. <= 0: no budget."""
    if max_estimated_s is not None:
        return float(max_estimated_s), "max_estimated_s"
    raw = (os.environ.get(BUDGET_ENV) or "").strip()
    if raw:
        try:
            return float(raw), BUDGET_ENV
        except ValueError:
            pass
    return DEFAULT_MAX_ESTIMATED_S, "default"


def _duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _cost_plan(clusters_csv: str, drop_singleton_clusters: bool, max_estimated_s) -> dict[str, Any]:
    """The run's estimate from the clusters table alone (cells with a label; a one-cell cluster dropped when the run
    drops them), before the worker reads the counts. A cell in the table but not in the counts is counted: an upper
    bound."""
    import pandas as pd

    table = pd.read_csv(clusters_csv, index_col=0, dtype=str, keep_default_na=False)
    if table.shape[1] < 1:
        raise ValueError("the clusters table has no label column")
    labels = table.iloc[:, 0].astype(str).str.strip()
    labels = labels[~labels.str.lower().isin(_MISSING_LABELS)]
    sizes = labels.value_counts()
    if drop_singleton_clusters:
        sizes = sizes[sizes > 1]
    n_cells, n_clusters = int(sizes.sum()), int(len(sizes))
    budget, source = _budget(max_estimated_s)
    return {
        "n_cells": n_cells,
        "n_clusters": n_clusters,
        "n_classes": neighborseq_classes(n_clusters),
        "estimated_s": round(neighborseq_seconds(n_cells, n_clusters), 1),
        "max_estimated_s": budget,
        "budget_source": source,
        "model": (
            f"seconds ~ {NSEQ_FIXED_S:g} + {NSEQ_CELL_CLUSTER_S:.3g} * cells * clusters + {NSEQ_CLASS2_S:.3g} * "
            "classes^2, classes = clusters * (clusters + 3) / 2 (calibrated on one 96-core host; xgboost uses every "
            "core)"
        ),
    }


def _over_budget(plan: dict[str, Any]) -> str | None:
    """The refusal when the estimate exceeds a positive budget, every number stated; else None."""
    budget = plan["max_estimated_s"]
    if budget <= 0 or plan["estimated_s"] <= budget:
        return None
    return (
        f"Neighbor-seq: this run is estimated at about {_duration(plan['estimated_s'])} ({plan['n_cells']:,} cells "
        f"in {plan['n_clusters']} clusters, so {plan['n_classes']} classes: every cluster and every pair). The marker "
        "panel, the predictions and the enrichment shuffles grow with cells x clusters, and the classifier with the "
        "square of the classes; the estimate is calibrated on one host and approximate. It is over the budget of "
        f"{_duration(budget)} (max_estimated_s, or {BUDGET_ENV} for the host), so nothing was run. To run it: fewer "
        "cells (a section or bounding-box subset of counts_csv and clusters_csv), fewer, coarser clusters in "
        "clusters_csv, or raise max_estimated_s."
    )


@mcp.tool()
def neighborseq_interaction_network(
    counts_csv: str,
    clusters_csv: str,
    output_dir: str,
    n_trees: int = 500,
    n_top: int = 20,
    pval_cutoff: float = 0.05,
    seed: int = 0,
    allow_hvg_panel_fallback: bool = False,
    drop_unlabeled: bool = False,
    drop_singleton_clusters: bool = False,
    max_estimated_s: float | None = None,
) -> dict[str, Any]:
    """
    Infer cell-cell interaction networks using Neighborseq (Neighbor-seq).

    Neighborseq 0.1.0 (``Neighborseq::neighborseq``) builds artificial doublets from the labelled
    cells, trains an xgboost random-forest classifier on them (``multiplet_rf``: 200 trees, one
    boosting round, max depth 20 -- fixed upstream), predicts a singlet or doublet class for every
    cell or spot, and tests each predicted cell-type pair for enrichment (Wilcoxon against 100
    shuffles; ``pval`` and ``padj`` per pair). The payload names what ran under ``params.method``.

    Input is two CSVs; an h5ad is not accepted (convert it with ``convert_h5ad_to_csv`` first).
    The gene panel is FindAllMarkers (Wilcoxon, logfc 0.25, min.pct 0.1, positive only) on every
    cell -- no per-cluster downsampling -- keeping the top 50 markers per cluster; a panel under
    10 genes is padded with 50 variable features, and ``params.gene_selection`` says which.

    Neighborseq splits class names on any non-alphanumeric character and matches cell types by
    regex substring, so labels are replaced by letters-only tokens before it runs and decoded
    back afterwards. Every output carries the original labels; ``neighborseq_label_map.csv``
    records the token for each label with its cell count.

    Parameters
    ----------
    counts_csv:
        Path to a comma-separated counts CSV with row and column headers: genes x cells, or cells x
        genes (the orientation is detected from the barcodes and reported in
        ``data.counts_orientation``). It is read in blocks of rows straight into a sparse matrix, so
        the whole table is never held dense; only the marker-panel genes are, for Neighborseq.
    clusters_csv:
        Path to a cluster assignments CSV: cell barcode in the first column (the index), cluster /
        cell-type label in the next. Cells in counts without a row here are left out and counted
        in ``data.n_cells_without_label_row``.
    output_dir:
        Directory for Neighborseq output files: ``neighborseq_interactions.csv`` (every tested pair,
        Cell_1/Cell_2 in the original labels), ``neighborseq_predictions.csv`` (class probabilities,
        one row per cell keyed by barcode), ``neighborseq_top_interactions.csv`` and
        ``neighborseq_label_map.csv``.
    n_trees:
        Accepted but not used: Neighborseq's ``multiplet_rf`` fixes the forest at 200 trees
        (xgboost ``num_parallel_tree = 200``) and no argument reaches it. Listed under
        ``params.ignored`` with a warning; the tree count that ran is ``params.num_parallel_tree``.
    n_top:
        Number of top interactions (lowest ``padj``) written to the top-interactions table.
    pval_cutoff:
        Accepted but not used: Neighborseq applies no p-value cutoff, and every tested pair is
        written with its ``pval`` and ``padj``. Listed under ``params.ignored`` with a warning.
    seed:
        Random seed (R ``set.seed``) for the artificial doublets, the train/test split and the
        enrichment shuffles.
    allow_hvg_panel_fallback:
        When FindAllMarkers finds no marker gene for any cluster the run stops by default. True
        trains on the top 100 variable features instead; the payload then says so in
        ``params.method``, ``params.used_fallback=True`` and a warning.
    drop_unlabeled:
        A cell whose label is missing (NA / empty / "nan") is an error by default. True leaves
        those cells out and reports the count in ``data.n_unlabeled_dropped``.
    drop_singleton_clusters:
        A cluster with exactly one cell is an error by default: Neighborseq samples a cluster's
        cells with ``base::sample``, which on a single index draws from ``1:index``, so the cluster
        would be trained on random other cells. True leaves those cells out and names the clusters
        in ``data.dropped_singleton_clusters``.
    max_estimated_s:
        The wall-time budget, in seconds, the run's estimate must fit. Neighbor-seq's cost grows with cells x
        clusters and with the square of its classes (every cluster and every pair: k(k+3)/2 for k clusters); the
        Zhuang mini block, 2,586 cells x 17 clusters, took minutes, and 309,599 cells x 30 clusters is hours. The
        cost is estimated from clusters_csv before the worker starts and reported as params.cost_estimate, and a
        run estimated over the budget is refused with the estimate and the ways to bring it down (fewer cells,
        fewer and coarser clusters). Default: the host's SOG_NEIGHBORSEQ_MAX_SECONDS, else 1800 s. 0 or less
        means no budget.
    """
    try:
        plan = _cost_plan(clusters_csv, drop_singleton_clusters, max_estimated_s)
    except Exception as exc:  # the worker reads the table itself and refuses what it cannot read, by name
        print(f"[neighborseq] cost estimate skipped: {exc}", file=sys.stderr, flush=True)
        plan = None
    if plan is not None:
        refusal = _over_budget(plan)
        if refusal:
            return {"status": "error", "tool": TOOL_NAME, "error": refusal, "params": {"cost_estimate": plan}}
    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--counts-csv",
        counts_csv,
        "--clusters-csv",
        clusters_csv,
        "--output-dir",
        output_dir,
        "--n-trees",
        str(n_trees),
        "--n-top",
        str(n_top),
        "--pval-cutoff",
        str(pval_cutoff),
        "--seed",
        str(seed),
    ]
    # The switches travel only when set, so a default call sends the argv it always sent.
    if allow_hvg_panel_fallback:
        args += ["--allow-hvg-panel-fallback", "true"]
    if drop_unlabeled:
        args += ["--drop-unlabeled", "true"]
    if drop_singleton_clusters:
        args += ["--drop-singleton-clusters", "true"]

    payload = run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)
    if plan is not None and isinstance(payload, dict) and payload.get("status") == "ok":
        payload.setdefault("params", {})["cost_estimate"] = plan
    return payload


if __name__ == "__main__":
    mcp.run()
