#!/usr/bin/env python3
"""
eval_metrics.py - Evaluation metrics for SpatialOmicsLab MCP tool outputs.

Supports three task categories:
  1. Spatial domain / clustering  →  ARI, NMI, homogeneity, completeness
  2. Spatially variable genes (SVG) →  gene overlap (Jaccard, precision, recall), Moran's I
  3. Deconvolution / cell-type mapping →  RMSE, Pearson/Spearman correlation, JSD

Each function returns a dict of metric_name -> value, suitable for inclusion
in the WorkerOutput "evaluation" field.

Mock gold-standard generators are provided for quick benchmarking when
real ground truth is unavailable.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# ============================================================================
# 1. Spatial Domain / Clustering Evaluation
# ============================================================================


def eval_clustering(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate clustering against ground truth.

    Returns: ARI, NMI, homogeneity, completeness, v_measure.
    """
    from sklearn.metrics import (
        adjusted_rand_score,
        homogeneity_completeness_v_measure,
        normalized_mutual_info_score,
    )

    pred = np.asarray(predicted_labels).ravel()
    true = np.asarray(true_labels).ravel()

    # Align lengths (drop NaN if any)
    mask = ~(np.array([str(x) == "nan" for x in pred]) | np.array([str(x) == "nan" for x in true]))
    pred, true = pred[mask], true[mask]

    ari = adjusted_rand_score(true, pred)
    nmi = normalized_mutual_info_score(true, pred, average_method="arithmetic")
    h, c, v = homogeneity_completeness_v_measure(true, pred)

    return {
        "ARI": round(float(ari), 4),
        "NMI": round(float(nmi), 4),
        "homogeneity": round(float(h), 4),
        "completeness": round(float(c), 4),
        "v_measure": round(float(v), 4),
        "n_samples": int(len(pred)),
        "n_predicted_clusters": int(len(set(pred))),
        "n_true_clusters": int(len(set(true))),
    }


def eval_clustering_from_adata(
    adata,
    pred_key: str,
    true_key: str,
) -> dict[str, float]:
    """Evaluate clustering from AnnData obs columns."""
    if pred_key not in adata.obs.columns:
        return {"error": f"predicted key '{pred_key}' not found in adata.obs"}
    if true_key not in adata.obs.columns:
        return {"error": f"ground truth key '{true_key}' not found in adata.obs"}
    return eval_clustering(
        adata.obs[pred_key].values,
        adata.obs[true_key].values,
    )


# ============================================================================
# 2. Spatially Variable Gene (SVG) Evaluation
# ============================================================================


def eval_svg_overlap(
    predicted_genes: list[str],
    true_genes: list[str],
) -> dict[str, float]:
    """
    Evaluate SVG prediction via set overlap metrics.

    Returns: jaccard, precision, recall, f1, n_overlap.
    """
    pred_set = set(predicted_genes)
    true_set = set(true_genes)

    overlap = pred_set & true_set
    union = pred_set | true_set

    n_overlap = len(overlap)
    jaccard = n_overlap / len(union) if union else 0.0
    precision = n_overlap / len(pred_set) if pred_set else 0.0
    recall = n_overlap / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "jaccard": round(jaccard, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_predicted": len(pred_set),
        "n_true": len(true_set),
        "n_overlap": n_overlap,
        "overlap_genes": sorted(overlap)[:50],  # cap at 50
    }


def eval_morans_i(
    adata,
    gene_list: list[str],
    spatial_key: str = "spatial",
    n_neighbors: int = 6,
) -> dict[str, Any]:
    """
    Compute Moran's I for a list of genes as a spatial autocorrelation metric.

    Higher Moran's I → stronger spatial pattern → better SVG candidate.
    Returns per-gene Moran's I and summary statistics.
    """
    import scanpy as sc
    from scipy.sparse import issparse

    # Build spatial graph if not present
    if "spatial_connectivities" not in adata.obsp:
        try:
            import squidpy as sq

            sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=n_neighbors)
        except ImportError:
            # Fallback: use scanpy neighbors on spatial coords
            from sklearn.neighbors import kneighbors_graph

            coords = adata.obsm[spatial_key]
            adj = kneighbors_graph(coords, n_neighbors=n_neighbors, mode="connectivity")
            adata.obsp["spatial_connectivities"] = adj

    # Filter to genes present in adata
    valid_genes = [g for g in gene_list if g in adata.var_names]
    if not valid_genes:
        return {"error": "No valid genes found in adata", "n_valid": 0}

    # Compute Moran's I against the spatial graph built above. scanpy's ``obsp=``/``use_graph=``
    # keywords do not reach it -- ``obsp=`` names the *values* to score, and ``use_graph=`` raises
    # NotImplementedError -- so the AnnData overload can only ever read obsp["connectivities"],
    # the transcriptomic neighbour graph, which spatial output does not have. The graph overload
    # takes the adjacency directly, with values shaped (n_genes, n_spots). It returns the scores;
    # it writes nothing to adata.var.
    expr = adata[:, valid_genes].X
    values_matrix = expr.T.tocsr() if issparse(expr) else np.asarray(expr).T
    scores = np.asarray(sc.metrics.morans_i(adata.obsp["spatial_connectivities"], values_matrix), dtype=float)

    # A gene with no variance across the tissue gives 0/0. Dropping it keeps the summary a number
    # and keeps the payload parseable -- NaN is not JSON.
    #
    # No ``zip(strict=)``: this module runs inside per-tool worker envs as old as Python 3.7,
    # where the keyword is a TypeError. The explicit check carries the same misalignment guard.
    if len(scores) != len(valid_genes):
        return {"error": f"morans_i returned {len(scores)} scores for {len(valid_genes)} genes"}
    morans = {g: round(float(v), 4) for g, v in zip(valid_genes, scores) if np.isfinite(v)}

    values = list(morans.values())
    result = {
        "per_gene_morans_i": morans,
        "mean_morans_i": round(float(np.mean(values)), 4) if values else None,
        "median_morans_i": round(float(np.median(values)), 4) if values else None,
        "n_genes_evaluated": len(values),
    }
    if len(values) < len(valid_genes):
        result["n_genes_undefined"] = len(valid_genes) - len(values)
    return result


# ============================================================================
# 3. Deconvolution Evaluation
# ============================================================================


def eval_deconvolution(
    predicted_proportions: np.ndarray,
    true_proportions: np.ndarray,
    cell_type_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate deconvolution results against ground truth proportions.

    Both arrays should be (n_spots, n_celltypes).
    Returns: RMSE, MAE, Pearson correlation (per cell type and overall), JSD.

    Every metric is computed over the spots whose ground truth carries mass, and the count of
    those spots is returned as ``n_spots_scored`` beside the ``n_spots`` handed in.
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import pearsonr

    pred = np.asarray(predicted_proportions, dtype=float)
    true = np.asarray(true_proportions, dtype=float)

    if pred.shape != true.shape:
        return {"error": f"Shape mismatch: pred={pred.shape}, true={true.shape}"}

    n_spots, n_ct = pred.shape
    ct_names = cell_type_names or [f"celltype_{i}" for i in range(n_ct)]

    # One cell type is not a composition. The JSD below renormalises each spot row -- ``p / p.sum()``
    # -- so a single column is ``[1.0]`` on both sides for every spot and the divergence is 0, the
    # best attainable value, whatever number the tool put there. Measured on 200 spots: a tool
    # writing 0.02 against a ground truth of 1.0 scores RMSE 0.98 and mean_JSD 0.0. In the case that
    # actually occurs the other two collapse as well, because proportions sum to 1 and a tool with
    # one cell type therefore writes 1.0 -- RMSE 0.0, MAE 0.0, mean_JSD 0.0, off a prediction
    # carrying no information. ``mean_pearson_r`` is already None here, since a one-hot ground truth
    # restricted to its own scorable rows is the constant 1.0 and has no variance to correlate.
    #
    # The shape is what a misread prediction table looks like -- transposed, read one column off,
    # or a ``dominant_celltype`` string column -- not a real experiment, and ``mean_JSD`` is
    # published straight into the comparison tables and ``jsd.png``. Refused rather than warned
    # about, for the reason ``_degenerate_partition`` refuses in the clustering evaluator: a warning
    # nothing reads is how a run gets recorded as scored.
    #
    # Zero columns is refused here too. It was already refused one check further down, by
    # "No spot has a non-zero ground-truth row" -- true, but it describes the ground truth when the
    # thing that went wrong is the prediction's delimiter, which is the wording ``eval_mcp_server``
    # uses for the same shape.
    if n_ct < 2:
        reason = (
            "Prediction has no cell-type columns at all, so there is nothing to score -- check the "
            "file's delimiter and whether its first column really is the spot index"
            if n_ct == 0
            else (
                f"Prediction has a single cell type '{ct_names[0]}', so there is no composition to "
                f"estimate -- every spot row renormalises to [1.0] on both sides and mean_JSD is "
                f"0.0 by construction here, for any prediction"
            )
        )
        return {
            "error": reason,
            "n_spots": n_spots,
            "n_spots_scored": 0,
            "n_celltypes": n_ct,
        }

    # A spot whose ground-truth row is all zeros has no ground truth. The caller aligns the two
    # tables on the prediction's columns, so a cell type present in the spatial reference but
    # absent from the single-cell one is dropped from the ground truth, and every spot labelled
    # with it is left at zero -- 817 of 4296 spots (19.0%) on the recorded MERFISH runs.
    #
    # The renormalisation below would turn each of those into a uniform 1/n_ct and score the
    # prediction against it, so a near-uniform output scored near-perfectly on a distribution
    # nobody measured (JSD 0.0241 on the fabricated rows against 0.7699 on the real ones, for the
    # same tool on the same dataset). They are excluded from every metric here rather than scored
    # against an invented target, and counted so the exclusion is visible in the result.
    scorable = true.sum(axis=1) > 0
    n_scored = int(scorable.sum())
    if n_scored == 0:
        return {
            "error": "No spot has a non-zero ground-truth row; nothing is measurable",
            "n_spots": n_spots,
            "n_spots_scored": 0,
            "n_celltypes": n_ct,
        }
    pred_s, true_s = pred[scorable], true[scorable]

    # Overall metrics
    rmse = float(np.sqrt(np.mean((pred_s - true_s) ** 2)))
    mae = float(np.mean(np.abs(pred_s - true_s)))

    # Per cell-type Pearson.
    #
    # A column leaves the average below for one of two very different reasons, and the result has to
    # record which. A constant GROUND-TRUTH column is a property of the dataset: it is unscoreable
    # for everyone, every tool on that dataset loses it, and the comparison between them stays fair.
    # A constant PREDICTION is a property of the tool: it captured none of that cell type's
    # variation, and dropping the column takes its own worst case out of its own average.
    #
    # On the twelve recorded slideseqv2_aorta_deconvolution runs not one of the seven ground-truth
    # columns is constant, so every drop there is the second kind -- and the tool ranked first by
    # empirical_leaderboard.py averages over six of the seven its eleven competitors average over.
    # Both causes are named so that a mean taken over a short denominator can be read as one.
    constant_in_truth: list[str] = []
    constant_in_prediction: list[str] = []
    per_ct_pearson = {}
    for i, ct in enumerate(ct_names):
        truth_varies = bool(np.std(true_s[:, i]) > 1e-10)
        pred_varies = bool(np.std(pred_s[:, i]) > 1e-10)
        if truth_varies and pred_varies:
            r, _ = pearsonr(true_s[:, i], pred_s[:, i])
            per_ct_pearson[ct] = round(float(r), 4)
            continue
        per_ct_pearson[ct] = None
        # Truth first: where both sides are flat nobody could have scored it, so blaming the tool
        # for a column no tool could win would be the same error in the other direction.
        (constant_in_truth if not truth_varies else constant_in_prediction).append(ct)

    valid_rs = [v for v in per_ct_pearson.values() if v is not None]
    mean_pearson = round(float(np.mean(valid_rs)), 4) if valid_rs else None

    # Per-spot JSD (average across spots)
    jsd_per_spot = []
    n_jsd_undefined = 0
    for j in range(n_scored):
        p = pred_s[j] + 1e-12
        q = true_s[j] + 1e-12
        p = p / p.sum()
        q = q / q.sum()
        value = float(jensenshannon(p, q))
        if not np.isfinite(value):
            # ``jensenshannon`` ends in ``sqrt(js / 2)``. Two distributions that agree to
            # floating-point noise leave a cancellation residue whose sign is not guaranteed, and
            # the sqrt of a tiny negative is NaN -- which ``np.mean`` then spreads to the dataset
            # metric. Six recorded metrics.json publish a bare ``NaN`` for mean_JSD, which is not
            # valid JSON either. Where the two rows really are equal to within floating point, the
            # divergence is 0 and that is what is recorded; anything else stays out of the mean and
            # is counted rather than quietly replaced by a number.
            if np.allclose(p, q):
                value = 0.0
            else:
                n_jsd_undefined += 1
                continue
        jsd_per_spot.append(value)
    mean_jsd = round(float(np.mean(jsd_per_spot)), 4) if jsd_per_spot else None

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "mean_pearson_r": mean_pearson,
        "per_celltype_pearson": per_ct_pearson,
        "mean_JSD": mean_jsd,
        "n_spots": n_spots,
        # How many of them carried a ground truth to score against, and how many cell types the
        # mean_pearson_r above was actually taken over -- constant columns leave that average, so
        # publishing n_celltypes alone beside it overstated the denominator on every recorded run.
        "n_spots_scored": n_scored,
        "n_celltypes": n_ct,
        "n_celltypes_correlated": len(valid_rs),
        "celltypes_constant_in_truth": constant_in_truth,
        "celltypes_constant_in_prediction": constant_in_prediction,
        "n_spots_jsd_undefined": n_jsd_undefined,
    }


def mean_row_entropy(values: np.ndarray) -> tuple[float | None, str]:
    """Mean Shannon entropy in bits over rows, each normalised to a distribution first.

    Returns the metric and a note; the note is empty when nothing needed saying, and the metric is
    ``None`` when the rows are not proportions at all.

    Entropy is only defined over a probability distribution. This used to be computed as
    ``-sum(p * log2(p + 1e-12))`` straight off the tool's CSV. Writing the row total as ``s`` and
    substituting ``p = s*q`` gives what that actually returns:

        s * H_true + s * log2(1 / s)

    -- the entropy only at ``s = 1``. Fifteen of forty-eight recorded proportion tables do not sum
    to one, and three of them have a single cell-type column, where there is one possible outcome
    per spot so the entropy is identically zero. Those three published 0.0707, 0.1121 and 0.3876.
    Normalising first is therefore not a refinement; it is the difference between the quantity and
    a different quantity. It is also why this is safe on everything already correct: at ``s = 1``
    the correction term is zero and the result is unchanged to the last decimal.

    Negative cells are the second half. ``log2`` of a negative is NaN, and one such cell takes the
    whole mean with it -- four recorded tables do this today, and the run is still returned at
    status ok. The recorded data separates the two causes by three orders of magnitude, which is
    what the 1% threshold below is drawn from rather than invented:

    - SpatialScope's NMF leaves residuals whose negative mass is at most 0.47% of the row's
      positive mass. That is rounding noise on a real deconvolution, and clipping it to zero is the
      ordinary treatment.
    - An IRIS spatial-domain result saved as ``*_proportions.csv`` carries negative mass equal to
      845% of its positive mass at the median. Clipping cannot turn that into a distribution, so
      the honest answer is to decline the metric and say why.

    A row that is entirely zero contributes nothing rather than a NaN: there is no distribution to
    measure, and dropping it is what lets a table with a few empty spots still report the metric for
    the spots it did resolve. If every row is empty there is nothing to average and the answer is
    ``None``.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.size == 0:
        return None, "the proportions table is empty, so there is no distribution to measure."

    finite = np.where(np.isfinite(array), array, 0.0)
    negative_mass = np.where(finite < 0, -finite, 0.0).sum(axis=1)
    positive_mass = np.where(finite > 0, finite, 0.0).sum(axis=1)
    # Compared per row against that row's own scale, so the rule does not depend on the units the
    # tool happened to write in.
    not_proportions = negative_mass > 0.01 * positive_mass
    if not_proportions.any():
        worst = float(finite.min())
        return None, (
            f"{int(not_proportions.sum())} of {finite.shape[0]} rows carry more than 1% negative "
            f"mass (most negative value {worst:.4g}), so they are not cell-type proportions and no "
            "entropy was computed for this table. Check that the file really is a proportions "
            "matrix and not another per-spot result saved under that name."
        )

    clipped = np.clip(finite, 0.0, None)
    totals = clipped.sum(axis=1, keepdims=True)
    measurable = (totals > 0).ravel()
    if not measurable.any():
        return None, "every row of the proportions table sums to zero, so no entropy is defined."

    q = np.divide(clipped, totals, out=np.zeros_like(clipped), where=totals > 0)
    # 0*log2(0) is 0 in the limit; computing it directly would be NaN and poison the mean.
    terms = np.where(q > 0, q * np.log2(np.where(q > 0, q, 1.0)), 0.0)
    # ``+ 0.0`` collapses the negative zero that comes out of negating an exact 0.0 sum -- a fully
    # resolved table really is zero entropy, and "-0.0" reads like a sign error to anyone seeing it.
    entropy = float(-terms.sum(axis=1)[measurable].mean()) + 0.0

    note = ""
    if not measurable.all():
        note = (
            f"{int((~measurable).sum())} of {measurable.size} spots have no assigned proportion at "
            "all and were left out of the mean entropy."
        )
    return round(entropy, 4), note


# ============================================================================
# 4. Mock Gold Standard Generators
# ============================================================================


def mock_clustering_labels(
    n_spots: int,
    n_clusters: int = 7,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate mock ground-truth clustering labels.
    Assigns spots to clusters with roughly equal proportions.
    """
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_clusters, size=n_spots)


def mock_svg_gene_list(
    all_genes: list[str],
    n_svg: int = 100,
    seed: int = 42,
) -> list[str]:
    """
    Generate a mock gold-standard SVG list by sampling from available genes.
    """
    rng = np.random.RandomState(seed)
    n = min(n_svg, len(all_genes))
    return list(rng.choice(all_genes, size=n, replace=False))


def mock_deconv_proportions(
    n_spots: int,
    n_celltypes: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate mock ground-truth deconvolution proportions.
    Returns (proportions_array, celltype_names).
    """
    rng = np.random.RandomState(seed)
    raw = rng.dirichlet(np.ones(n_celltypes), size=n_spots)
    names = [f"CellType_{i}" for i in range(n_celltypes)]
    return raw, names


# ============================================================================
# 5. High-level Evaluation Dispatcher
# ============================================================================


def evaluate_tool_output(
    tool_name: str,
    task_type: str,
    output_dir: str,
    output_files: dict[str, str],
    summary: dict[str, Any],
    gold_standard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    High-level dispatcher that runs appropriate evaluation for a tool output.

    Args:
        tool_name: Name of the tool (e.g. "deepst")
        task_type: One of "clustering", "svg", "deconvolution", "other"
        output_dir: Tool's output directory
        output_files: Dict of output file paths from WorkerOutput
        summary: Summary dict from WorkerOutput
        gold_standard: Optional dict with ground truth data:
            - For clustering: {"labels_csv": path} or {"labels": array}
            - For SVG: {"gene_list": [genes]} or {"gene_list_csv": path}
            - For deconv: {"proportions_csv": path} or {"proportions": array, "celltypes": [...]}

    Returns:
        Dict with "metrics", "interpretation", and "task_type" keys.
    """
    result: dict[str, Any] = {
        "tool": tool_name,
        "task_type": task_type,
        "eval_mode": "gold_standard" if gold_standard else "self_assessment",
    }

    try:
        if task_type == "clustering":
            result["metrics"] = _eval_clustering_output(output_files, summary, gold_standard)
        elif task_type == "svg":
            result["metrics"] = _eval_svg_output(output_files, summary, gold_standard)
        elif task_type == "deconvolution":
            result["metrics"] = _eval_deconv_output(output_files, summary, gold_standard)
        else:
            result["metrics"] = _eval_generic_output(output_files, summary)

        result["interpretation"] = _interpret_metrics(task_type, result["metrics"])

    except Exception as e:
        result["metrics"] = {"error": str(e)}
        result["interpretation"] = f"Evaluation failed: {e}"

    return result


def _eval_clustering_output(
    output_files: dict[str, str],
    summary: dict[str, Any],
    gold: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate clustering output."""
    import anndata as ad
    import pandas as pd
    from worker_utils import sniff_tabular_sep

    metrics: dict[str, Any] = {}

    # Self-assessment: cluster balance
    cluster_sizes = summary.get("cluster_sizes", {})
    if cluster_sizes:
        sizes = list(cluster_sizes.values())
        total = sum(sizes)
        max_s, min_s = max(sizes), min(sizes)
        metrics["n_clusters"] = len(sizes)
        metrics["balance_ratio"] = round(max_s / min_s, 2) if min_s > 0 else float("inf")
        metrics["entropy"] = round(float(-sum((s / total) * np.log2(s / total) for s in sizes if s > 0)), 4)
        metrics["max_cluster_pct"] = round(max_s / total * 100, 1)

    # Gold standard: ARI / NMI
    if gold:
        h5ad_path = output_files.get("annotated_h5ad")
        cluster_key = summary.get("cluster_key")
        if h5ad_path and cluster_key and os.path.isfile(h5ad_path):
            adata = ad.read_h5ad(h5ad_path)
            if "labels" in gold:
                true_labels = np.asarray(gold["labels"])
            elif "labels_csv" in gold and os.path.isfile(gold["labels_csv"]):
                # The caller names this file, so its delimiter is read from it. Parsed as commas, a
                # tab-delimited truth file has one column, ``iloc[:, -1]`` is the whole line, and
                # every spot becomes its own class -- a prediction that reproduces the truth exactly
                # then scores ARI 0.0.
                df = pd.read_csv(gold["labels_csv"], sep=sniff_tabular_sep(gold["labels_csv"]))
                true_labels = df.iloc[:, -1].values
            else:
                # Mock labels for benchmarking
                true_labels = mock_clustering_labels(adata.n_obs, summary.get("n_clusters", 7))
                metrics["gold_standard_source"] = "mock"

            pred_labels = adata.obs[cluster_key].values
            metrics.update(eval_clustering(pred_labels, true_labels))

    return metrics


def _eval_svg_output(
    output_files: dict[str, str],
    summary: dict[str, Any],
    gold: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate SVG output."""
    import pandas as pd
    from worker_utils import sniff_tabular_sep

    metrics: dict[str, Any] = {}

    # Get predicted genes from output
    top_csv = output_files.get("top_genes_csv") or output_files.get("top_scores_csv")
    pred_genes = summary.get("top_genes", [])

    if not pred_genes and top_csv and os.path.isfile(top_csv):
        df = pd.read_csv(top_csv, sep=sniff_tabular_sep(top_csv))
        gene_col = [c for c in df.columns if c.lower() in ("gene", "gene_name", "genes", "index")]
        if gene_col:
            pred_genes = df[gene_col[0]].tolist()
        elif df.shape[1] >= 1:
            pred_genes = df.iloc[:, 0].tolist()

    metrics["n_predicted_svg"] = len(pred_genes)

    # Gold standard overlap
    if gold:
        true_genes = gold.get("gene_list", [])
        if not true_genes and "gene_list_csv" in gold:
            # Parsed as commas, a tab-delimited gene list yields whole lines, none of which is a
            # gene symbol, so a method that recovered every true SVG scores f1 0.0. Nothing
            # downstream can tell that apart from a method that found nothing.
            df = pd.read_csv(gold["gene_list_csv"], sep=sniff_tabular_sep(gold["gene_list_csv"]))
            true_genes = df.iloc[:, 0].tolist()

        if not true_genes and pred_genes:
            # Mock: sample from predicted + random genes
            all_genes = list(pred_genes) + [f"MOCK_GENE_{i}" for i in range(200)]
            true_genes = mock_svg_gene_list(all_genes, n_svg=100)
            metrics["gold_standard_source"] = "mock"

        if true_genes and pred_genes:
            metrics.update(eval_svg_overlap(pred_genes, true_genes))

    # Moran's I self-assessment
    h5ad_path = output_files.get("annotated_h5ad") or output_files.get("scores_h5ad")
    if h5ad_path and os.path.isfile(h5ad_path) and pred_genes:
        try:
            import anndata as ad

            adata = ad.read_h5ad(h5ad_path)
            morans = eval_morans_i(adata, pred_genes[:50])
            metrics["morans_i"] = morans
        except Exception as e:
            metrics["morans_i_error"] = str(e)

    return metrics


def _eval_deconv_output(
    output_files: dict[str, str],
    summary: dict[str, Any],
    gold: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate deconvolution output."""
    import pandas as pd
    from worker_utils import sniff_tabular_sep

    metrics: dict[str, Any] = {}

    # Self-assessment: proportion distribution
    prop_csv = output_files.get("cell_type_proportions_csv") or output_files.get("composition_csv")
    if prop_csv and os.path.isfile(prop_csv):
        df = pd.read_csv(prop_csv, index_col=0, sep=sniff_tabular_sep(prop_csv))
        n_spots, n_ct = df.shape
        metrics["n_spots"] = n_spots
        metrics["n_celltypes"] = n_ct
        metrics["mean_entropy"], entropy_note = mean_row_entropy(df.to_numpy())
        if entropy_note:
            metrics["mean_entropy_note"] = entropy_note
        metrics["dominant_celltype_coverage"] = round(
            float((df.idxmax(axis=1).value_counts().iloc[0] / n_spots) * 100), 1
        )

    # Gold standard: RMSE / correlation
    if gold:
        if "proportions" in gold:
            true_props = np.asarray(gold["proportions"])
            ct_names = gold.get("celltypes")
        elif "proportions_csv" in gold and os.path.isfile(gold["proportions_csv"]):
            # Parsed as commas with ``index_col=0``, a tab-delimited truth table has zero columns.
            # The shape check below then never matches, ``eval_deconvolution`` is never called, and
            # the run comes back with no deconvolution metric at all rather than a bad one -- which
            # nothing downstream distinguishes from a tool that produced nothing to score.
            true_df = pd.read_csv(gold["proportions_csv"], index_col=0, sep=sniff_tabular_sep(gold["proportions_csv"]))
            true_props = true_df.values
            ct_names = list(true_df.columns)
        elif prop_csv and os.path.isfile(prop_csv):
            # Mock proportions
            df = pd.read_csv(prop_csv, index_col=0, sep=sniff_tabular_sep(prop_csv))
            true_props, ct_names = mock_deconv_proportions(df.shape[0], df.shape[1])
            metrics["gold_standard_source"] = "mock"
        else:
            return metrics

        if prop_csv and os.path.isfile(prop_csv):
            pred_df = pd.read_csv(prop_csv, index_col=0, sep=sniff_tabular_sep(prop_csv))
            pred_props = pred_df.values
            if pred_props.shape == true_props.shape:
                metrics.update(eval_deconvolution(pred_props, true_props, ct_names))

    return metrics


def _eval_generic_output(
    output_files: dict[str, str],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Basic self-assessment for tools that don't fit standard categories."""
    metrics: dict[str, Any] = {}

    # Check that output files exist
    files_exist = {}
    for k, v in output_files.items():
        if v and isinstance(v, str):
            files_exist[k] = os.path.isfile(v)
    metrics["output_files_exist"] = files_exist
    metrics["all_outputs_created"] = all(files_exist.values()) if files_exist else False

    return metrics


def _interpret_metrics(task_type: str, metrics: dict[str, Any]) -> str:
    """Generate human-readable interpretation of evaluation metrics."""
    if "error" in metrics:
        return f"Evaluation error: {metrics['error']}"

    parts = []

    if task_type == "clustering":
        if "ARI" in metrics:
            ari = metrics["ARI"]
            if ari > 0.8:
                parts.append(f"Excellent clustering agreement (ARI={ari}).")
            elif ari > 0.5:
                parts.append(f"Good clustering agreement (ARI={ari}).")
            elif ari > 0.2:
                parts.append(f"Moderate clustering agreement (ARI={ari}).")
            else:
                parts.append(f"Low clustering agreement (ARI={ari}).")

            if "NMI" in metrics:
                parts.append(f"NMI={metrics['NMI']}.")

        if "balance_ratio" in metrics:
            br = metrics["balance_ratio"]
            if br < 3:
                parts.append(f"Clusters are well-balanced (max/min={br}x).")
            elif br < 10:
                parts.append(f"Moderate cluster imbalance (max/min={br}x).")
            else:
                parts.append(f"Significant cluster imbalance (max/min={br}x).")

    elif task_type == "svg":
        if "f1" in metrics:
            parts.append(
                f"SVG overlap: F1={metrics['f1']}, precision={metrics['precision']}, "
                f"recall={metrics['recall']} ({metrics['n_overlap']} genes in common)."
            )
        if "morans_i" in metrics and isinstance(metrics["morans_i"], dict):
            mi = metrics["morans_i"].get("mean_morans_i")
            if mi is not None:
                if mi > 0.3:
                    parts.append(f"Strong spatial autocorrelation (mean Moran's I={mi}).")
                elif mi > 0.1:
                    parts.append(f"Moderate spatial autocorrelation (mean Moran's I={mi}).")
                else:
                    parts.append(f"Weak spatial autocorrelation (mean Moran's I={mi}).")

    elif task_type == "deconvolution":
        if "RMSE" in metrics:
            parts.append(f"Deconvolution RMSE={metrics['RMSE']}, MAE={metrics['MAE']}.")
        if "mean_pearson_r" in metrics:
            r = metrics["mean_pearson_r"]
            if r and r > 0.8:
                parts.append(f"Excellent correlation (mean Pearson r={r}).")
            elif r and r > 0.5:
                parts.append(f"Moderate correlation (mean Pearson r={r}).")
            elif r:
                parts.append(f"Weak correlation (mean Pearson r={r}).")
        if "mean_JSD" in metrics:
            parts.append(f"Mean JSD={metrics['mean_JSD']}.")

    return " ".join(parts) if parts else "Evaluation completed."
