#!/usr/bin/env python3
"""
BSP (scbsp) worker for spatially variable gene detection.

Runs inside /opt/conda/envs/bsp conda env.

``scbsp.granp(coords, counts)`` (single-cell big-small patch) scores each gene by how the variance of
its local means changes between a small and a large neighbourhood patch, fits one log-normal null to
those scores and returns a DataFrame of ``gene_names`` / ``p_values``.

Input: h5ad file with spatial coords in obsm[spatial_key] and expression in .X or a layer
Output: bsp_results.csv (every tested gene, ranked by p-value), bsp_top_genes.csv and
predicted_genes.json (the reported top-k, after the optional pvalue_cutoff filter)

Spots flagged ``obs['in_tissue'] == 0`` are background and are left out before the test (counted in
``data``, ``params`` and the analysis text). Genes detected in fewer than MIN_SPOTS_PER_GENE spots are
not tested (counted the same way). Nothing is subsampled.

The expression matrix stays sparse end to end: scbsp works on CSR throughout and converts any other
input to CSR itself, so a dense copy here is pure overhead (73 GB on a VisiumHD section).

Which matrix is tested: ``scbsp.granp`` documents its input as the raw expression matrix (it scales each
spot by its maximum and fits a log-normal null to variance ratios). ``adata.X`` is used, or the layer
named by ``--layer``, or ``adata.raw.X`` with ``--use-raw-counts`` (the shared
``worker_utils.choose_counts_matrix`` rule). Negative or non-finite values (scaled / z-scored data, as in
a CELLxGENE export whose counts sit in ``adata.raw``) are refused, naming ``use_raw_counts`` when
``adata.raw`` holds counts: a z-score is "detected" whenever it is above 0 and a negative spot maximum
flips the scaling. A non-negative non-integer matrix (normalised data) is tested as supplied, with a
warning. ``params.expression_source`` and ``params.x_matrix_kind`` say which matrix ran.

All logs go to stderr; stdout is JSON-only (final result): scbsp prints on import, so the import and
the test run with stdout redirected to stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from worker_utils import (
    WorkerOutput,
    build_svg_analysis,
    choose_counts_matrix,
    expression_matrix_kind,
    identifier_rename_params,
    make_names_unique_and_report,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    spatial_coords,
)

#: What runs, named for ``params.method``. There is no other implementation behind this tool.
METHOD_NAME = "scBSP big-small patch test (scbsp.granp)"

#: A gene must be detected (value > 0) in at least this many spots to be tested. scbsp fits ONE
#: log-normal null to the scores of every gene it is given, and a gene seen in a handful of spots
#: has a degenerate score (0 or NaN) that turns every p-value of the run into NaN.
MIN_SPOTS_PER_GENE = 10

#: The threshold ``summary.n_significant`` counts (p < 0.05) when ``pvalue_cutoff`` is left at 1.0,
#: i.e. when the caller asked for no significance filter on the reported list.
DEFAULT_SIGNIFICANCE = 0.05

#: Why ``seed`` cannot change a BSP result.
_NO_RANDOMNESS = (
    "scbsp.granp draws no random numbers (ball-tree neighbourhoods, sparse local means and a "
    "log-normal null fitted to the scores are all deterministic), so the seed cannot change the result"
)

_PVALUE_COLUMNS = ("p_values", "p-value", "pval", "p_value", "pvalue", "P-value", "Pvalue")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def _ensure_dir(p: str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _as_csr_float64(X):
    """The expression matrix as a canonical float64 CSR copy, never a dense one.

    ``scbsp.granp`` keeps a CSR input as it is and converts anything else to CSR itself
    (``input_exp_mat_raw if isspmatrix_csr(...) else csr_matrix(...)``), and every step after that is
    sparse. Densifying first cost ``n_spots x n_genes x 8`` bytes for nothing: 73 GB on a VisiumHD
    section, 26 GB on a Xenium one.

    Canonical form matters, not just the format: upstream's row scaling pairs ``.data`` with
    ``.nonzero()``, which skips explicitly stored zeros, so a CSR that carries any would be scaled
    out of alignment. Summing duplicates and dropping stored zeros gives exactly the matrix
    ``csr_matrix(dense)`` would have built. float64 because that is what the dense path fed it.
    """
    import scipy.sparse as sps

    if sps.issparse(X):
        m = sps.csr_matrix(X, dtype=np.float64, copy=True)
    else:
        m = sps.csr_matrix(np.asarray(X), dtype=np.float64)
    m.sum_duplicates()
    m.eliminate_zeros()
    return m


def _detected_spots_per_gene(counts) -> np.ndarray:
    """Spots with a value > 0, per gene, without a dense boolean copy of the matrix."""
    return np.asarray((counts > 0).sum(axis=0)).ravel().astype(np.int64)


def _in_tissue_mask(obs: pd.DataFrame):
    """``(keep_mask or None, why_not)`` for the Space Ranger ``in_tissue`` flag.

    Space Ranger / CELLxGENE exports can carry every array spot, and the off-tissue ones are
    background: an SVG test run over them finds the tissue edge (on the Fetal12W heart sample 3,009 of
    4,992 spots are off the tissue). None when there is no usable flag: no column, or values that are
    not a clean 0/1 flag (then ``why_not`` says so and nothing is dropped on a guess).
    """
    if "in_tissue" not in obs.columns:
        return None, ""
    flag = pd.to_numeric(pd.Series(np.asarray(obs["in_tissue"]), dtype=object), errors="coerce")
    values = set(flag.dropna().unique().tolist())
    if flag.isna().any() or not values <= {0, 1}:
        return None, (
            "obs['in_tissue'] is present but is not a 0/1 flag, so no spot was excluded by it; "
            "every spot was analysed as tissue"
        )
    keep = flag.to_numpy() == 1
    if not keep.any():
        raise ValueError(
            f"obs['in_tissue'] marks all {len(keep)} spots as off the tissue (in_tissue == 0), so nothing is left "
            "to analyse. Fix the flag, or drop the column if every spot is tissue."
        )
    return keep, ""


def _pvalue_column(df: pd.DataFrame):
    for candidate in _PVALUE_COLUMNS:
        if candidate in df.columns:
            return candidate
    p_cols = [c for c in df.columns if "p" in str(c).lower() and "val" in str(c).lower()]
    return p_cols[0] if p_cols else None


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = str(path) + ".partial"
    df.to_csv(tmp, index=False)
    os.replace(tmp, str(path))


def _write_text_atomic(text: str, path: Path) -> None:
    tmp = str(path) + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, str(path))


def _check_arguments(top_k_genes: Any, pvalue_cutoff: Any) -> float:
    """Refuse values that used to be quietly reinterpreted. Returns the cutoff as a float."""
    if int(top_k_genes) < 0:
        raise ValueError(
            f"top_k_genes={top_k_genes} is negative; it is the number of genes to report (0 or more). "
            "A negative value used to be read as 'every gene except the last N'."
        )
    cutoff = float(pvalue_cutoff)
    if math.isnan(cutoff) or cutoff <= 0.0:
        raise ValueError(
            f"pvalue_cutoff={pvalue_cutoff} admits no gene: it must be > 0 (e.g. 0.05), or 1.0 to report "
            "the top_k_genes ranking without a significance filter."
        )
    return cutoff


def _threshold_sentence(filter_applied: bool, cutoff: float, n_passing, n_reported: int) -> str:
    """Which threshold ``n_significant`` counted, and how the reported list relates to it.

    The cutoff is the caller's own value: any cutoff >= 1 applies no filter, and the sentence used to say
    "pvalue_cutoff=1.0" whatever was passed.
    """
    if not filter_applied:
        return (
            f"Significant means p < {DEFAULT_SIGNIFICANCE:g}; with pvalue_cutoff={cutoff:g} (>= 1) no significance "
            f"filter is applied, so the {n_reported} reported genes are the lowest p-values whether or not they "
            "pass it."
        )
    head = f"Significant means p < {cutoff:g} (pvalue_cutoff); "
    if not n_passing:
        return head + "no gene passes it, so no gene is reported (the full ranking is in bsp_results.csv)."
    if n_reported >= n_passing:
        return head + f"all {n_passing} genes that pass it are reported."
    return head + f"{n_passing} genes pass it and the {n_reported} with the lowest p-values are reported."


_NOT_COUNTS_BSP = (
    "{source} holds non-integer values (normalised or log-transformed data?); scbsp.granp documents its input "
    "as the raw expression matrix, and the test ran on these values as supplied.{hint}"
)


def _choose_matrix(adata, layer: str, use_raw_counts: bool, renamed: dict):
    """``(adata, counts_csr, info, ignored)``: the matrix scBSP tests, by the shared counts rule.

    ``use_raw_counts`` runs on ``adata.raw.X`` (``choose_counts_matrix``); a named layer then has no
    effect and is returned in ``ignored``. Otherwise the layer, or ``adata.X``: negative or non-finite
    values are refused (naming ``use_raw_counts`` when ``adata.raw`` holds counts) and a non-integer
    matrix is tested as supplied, with a warning. Nothing is densified.
    """
    layer = (layer or "").strip()
    raw = getattr(adata, "raw", None)
    raw_holds_counts = raw is not None and expression_matrix_kind(raw.X) == "counts"
    raw_hint = " adata.raw holds raw counts: pass use_raw_counts=True to run on them." if raw_holds_counts else ""
    if use_raw_counts:
        adata, info = choose_counts_matrix(adata, True)
        # raw.X has its own gene axis; deduplicate it the way the load deduplicated var. Only the
        # matrix that is tested is reported, so X's gene renames from the load are not added to it.
        renamed["n_genes_renamed"] = 0
        make_names_unique_and_report(adata, into=renamed, axes=("var",))
        eprint("[BSP] Using adata.raw.X for expression data (use_raw_counts)")
        return adata, _as_csr_float64(adata.X), info, (["layer"] if layer else [])
    if layer:
        if layer not in adata.layers:
            raise KeyError(f"Requested layer='{layer}' not found. Available: {list(adata.layers.keys())}")
        kind = expression_matrix_kind(adata.layers[layer])
        info = {"expression_source": f"layers['{layer}']", "x_matrix_kind": expression_matrix_kind(adata.X)}
        if kind in ("negative", "nonfinite"):
            what = "negative values (scaled or z-scored data)" if kind == "negative" else "NaN or infinite values"
            raise ValueError(
                f"layer '{layer}' holds {what}, not an expression matrix scbsp.granp can test (it scales each "
                "spot by its maximum and takes logs of the scores). Name a layer that holds counts." + raw_hint
            )
        info["warning"] = (
            _NOT_COUNTS_BSP.format(source=f"layer '{layer}'", hint="") if kind == "nonnegative_noninteger" else None
        )
        eprint(f"[BSP] Using layer '{layer}' for expression data")
        return adata, _as_csr_float64(adata.layers[layer]), info, []
    # choose_counts_matrix's rule, in scBSP's words: it tests X as supplied rather than normalising it.
    kind = expression_matrix_kind(adata.X)
    if kind in ("negative", "nonfinite"):
        what = "negative values (scaled or z-scored data)" if kind == "negative" else "NaN or infinite values"
        hint = (
            " adata.raw holds raw counts: pass use_raw_counts=True to run on them."
            if raw_holds_counts
            else " Supply an h5ad whose X (or adata.raw with use_raw_counts=True, or a layer) holds raw counts."
        )
        raise ValueError(
            f"X holds {what}, not counts. scbsp.granp tests the raw expression matrix: it scales each spot by "
            "its maximum and takes logs of variance ratios, and a gene counts as detected where its value is "
            "above 0, so scaled values give it meaningless scores; nothing was tested." + hint
        )
    warning = None
    if kind == "nonnegative_noninteger":
        hint = " adata.raw holds raw counts: pass use_raw_counts=True to run on them." if raw_holds_counts else ""
        warning = _NOT_COUNTS_BSP.format(source="X", hint=hint)
    info = {"expression_source": "X", "x_matrix_kind": kind, "warning": warning}
    eprint("[BSP] Using adata.X for expression data")
    return adata, _as_csr_float64(adata.X), info, []


def run_bsp(
    st_h5ad: str,
    output_dir: str,
    spatial_key: str = "spatial",
    layer: str = "",
    top_k_genes: int = 200,
    pvalue_cutoff: float = 1.0,
    random_seed: int = 0,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Run BSP spatially variable gene detection."""

    cutoff = _check_arguments(top_k_genes, pvalue_cutoff)
    top_k_genes = int(top_k_genes)
    filter_applied = cutoff < 1.0
    # summary.n_significant counts at the caller's cutoff when they set one, else at p < 0.05.
    threshold = cutoff if filter_applied else DEFAULT_SIGNIFICANCE

    import scanpy as sc

    with contextlib.redirect_stdout(sys.stderr):
        # scbsp prints "CUDA is not available" on stdout at import when torch is installed.
        import scbsp

    use_raw_counts = bool(use_raw_counts)
    _ensure_dir(output_dir)
    np.random.seed(random_seed)
    outdir = Path(output_dir)

    # ---- Load data ----
    eprint(f"[BSP] Loading spatial data: {st_h5ad}")
    adata = sc.read_h5ad(st_h5ad)
    renamed = make_names_unique_and_report(adata)
    eprint(f"[BSP] Loaded: {adata.n_obs} spots x {adata.n_vars} genes")

    # ---- Extract coordinates ----
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"Spatial key '{spatial_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}"
        )
    spatial_coords(adata, spatial_key, want=2, tool="BSP")  # refuse a 3-column key before any work

    run_warnings: list = []

    # ---- Off-tissue spots are background, not tissue ----
    n_spots_supplied = int(adata.n_obs)
    tissue_keep, tissue_note = _in_tissue_mask(adata.obs)
    if tissue_note:
        run_warnings.append(tissue_note)
    n_off_tissue = 0 if tissue_keep is None else int((~tissue_keep).sum())
    if n_off_tissue:
        eprint(f"[BSP] Leaving out {n_off_tissue} of {n_spots_supplied} spots with in_tissue == 0")
        adata = adata[tissue_keep].copy()

    coords, _ = spatial_coords(adata, spatial_key, want=2, tool="BSP")
    eprint(f"[BSP] Using spatial coordinates from obsm['{spatial_key}'], shape={coords.shape}")

    # ---- Select the matrix (counts or refuse; kept sparse) ----
    adata, counts, counts_info, matrix_ignored = _choose_matrix(adata, layer, use_raw_counts, renamed)

    # ---- Filter ultra-sparse genes (mirrors manual bsp_identify_svg.py) ----
    # scbsp.granp fits one log-normal null to the scores of all genes; a gene detected in fewer than
    # MIN_SPOTS_PER_GENE spots has a degenerate score that turns EVERY p-value into NaN, so these are
    # left out before the call, and the count is published (data, params, analysis).
    n_genes_supplied = int(adata.n_vars)
    detected = _detected_spots_per_gene(counts)
    keep_mask = detected >= MIN_SPOTS_PER_GENE
    n_kept = int(keep_mask.sum())
    if n_kept == 0:
        most = int(detected.max()) if detected.size else 0
        raise ValueError(
            f"No gene is detected (value > 0) in at least {MIN_SPOTS_PER_GENE} of the {adata.n_obs} spots; "
            f"the most widely detected of the {n_genes_supplied} genes is seen in {most}. scBSP fits one "
            "log-normal null to every gene's score and a gene this sparse has a degenerate score, so there "
            f"is nothing it can test here. Check that {counts_info['expression_source']} holds the expression "
            "values."
        )
    n_dropped = n_genes_supplied - n_kept
    var_names_all = np.asarray(adata.var_names.astype(str))
    if n_dropped > 0:
        counts = counts[:, np.flatnonzero(keep_mask)]
        eprint(
            f"[BSP] Pre-filter: dropped {n_dropped} ultra-sparse genes (<{MIN_SPOTS_PER_GENE} nonzero spots); "
            f"kept {counts.shape[1]}"
        )
    kept_var_names = [str(g) for g in var_names_all[keep_mask]]

    # ---- Run BSP ----
    eprint(f"[BSP] Running scbsp.granp on {adata.n_obs} spots x {counts.shape[1]} genes (sparse, nnz={counts.nnz})")
    with contextlib.redirect_stdout(sys.stderr):
        result = scbsp.granp(input_sp_mat=coords, input_exp_mat_raw=counts)
    eprint(f"[BSP] BSP completed. Result shape: {getattr(result, 'shape', None)}")

    # ---- Process results ----
    # scbsp.granp returns one row per column of the matrix it received, in that order, with
    # placeholder names ('Gene_0'..) in 'gene_names'. The matrix is post-filter, so row i is
    # kept_var_names[i] -- never adata.var_names[i].
    result_df = result.copy() if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
    if len(result_df) != n_kept:
        raise RuntimeError(
            f"scbsp.granp returned {len(result_df)} rows for the {n_kept} genes it was given; the rows "
            "cannot be matched back to gene names."
        )
    pval_col = _pvalue_column(result_df)
    if pval_col is None:
        raise RuntimeError(f"scbsp.granp returned no p-value column (columns: {list(result_df.columns)}).")
    result_df = result_df.reset_index(drop=True)
    for col in ("gene", "index"):
        if col in result_df.columns:
            result_df = result_df.drop(columns=[col])
    result_df.insert(0, "gene", kept_var_names)
    if "gene_names" in result_df.columns:
        # Upstream's 'Gene_<i>' indexes the post-filter matrix and names nothing the user has.
        result_df["gene_names"] = kept_var_names
    gene_col = "gene"
    result_df[pval_col] = pd.to_numeric(result_df[pval_col], errors="coerce").astype(float)

    pvals = result_df[pval_col]
    n_nan = int(pvals.isna().sum())
    if n_nan == len(result_df):
        # Nothing is written: a table of NaN p-values is not a result, and inventing one (the old
        # code stamped p=0.01 on every gene) makes every gene "significant".
        raise ValueError(
            f"scBSP returned no usable p-value: all {n_nan} genes tested came back NaN ({adata.n_obs} spots). "
            "Its null is a log-normal fitted to the scores below their 90th percentile, which needs roughly "
            "20 or more genes with finite, positive scores; with this gene set that fit is undefined. "
            "No p-value was invented. Test a larger gene panel, or check that the expression values are "
            "counts or normalised expression."
        )

    # Sort by p-value ascending (NaN last)
    result_df = result_df.sort_values(pval_col, ascending=True)
    pvals = result_df[pval_col]

    # Save full results
    result_csv = outdir / "bsp_results.csv"
    _write_csv_atomic(result_df, result_csv)
    eprint(f"[BSP] Saved full results to {result_csv}")

    # Significance pre-filter (matches manual /workspace/hands_by_myself/runners/bsp_identify_svg.py).
    # When pvalue_cutoff < 1.0, the reported pool is restricted to p < cutoff BEFORE taking top-K, and
    # an empty pool stays empty: substituting the unfiltered ranking would report genes that failed
    # the filter the caller asked for. Default pvalue_cutoff=1.0 disables the filter (top-K only).
    n_passing = int((pvals < cutoff).sum()) if filter_applied else None
    if filter_applied:
        pool = result_df[(pvals < cutoff).to_numpy()]
        eprint(f"[BSP] pvalue_cutoff={cutoff}: {n_passing}/{len(result_df)} genes pass significance filter")
        if n_passing == 0:
            run_warnings.append(
                f"0 of the {len(result_df)} genes tested have p < {cutoff:g} (pvalue_cutoff), so predicted_genes.json "
                "and bsp_top_genes.csv list no gene; the full ranking is in bsp_results.csv."
            )
    else:
        pool = result_df

    # Get top genes from the (optionally pre-filtered) pool
    top_k = min(top_k_genes, len(pool))
    top_df = pool.head(top_k)
    top_genes = top_df[gene_col].astype(str).tolist()
    top_csv = outdir / "bsp_top_genes.csv"
    _write_csv_atomic(top_df, top_csv)
    eprint(f"[BSP] Top {top_k} genes saved to {top_csv}")

    # Curated SVG-prediction artifact for benchmark evaluator (preferred over
    # bsp_results.csv, which is unfiltered and yields low-precision F1).
    pg_path = outdir / "predicted_genes.json"
    _write_text_atomic(json.dumps({"predicted_genes": top_genes}, indent=2), pg_path)
    eprint(f"[BSP] Wrote {pg_path} with {len(top_genes)} top genes")

    # Count significant genes: p < pvalue_cutoff when one was set, else the conventional p < 0.05.
    n_significant = int((pvals < threshold).sum())
    eprint(f"[BSP] Significant genes (p < {threshold:g}): {n_significant}")
    if n_nan:
        run_warnings.append(
            f"{n_nan} of the {len(result_df)} genes tested got a NaN p-value from scBSP; they are ranked last "
            "and never counted as significant."
        )

    # ---- Build output ----
    out = WorkerOutput("bsp", task="svg_identification")
    out.set_data(
        n_spots=int(adata.n_obs),
        n_spots_supplied=n_spots_supplied,
        n_spots_out_of_tissue_excluded=n_off_tissue,
        n_genes=n_genes_supplied,
        n_genes_tested=int(len(result_df)),
    )
    out.add_output_files(
        {
            "results_csv": str(result_csv),
            "top_genes_csv": str(top_csv),
            "predicted_genes_json": str(pg_path),
        }
    )
    out.add_params(
        {
            "spatial_key": spatial_key,
            "layer": layer if layer else "X",
            "use_raw_counts": use_raw_counts,
            "top_k_genes": top_k,
            "top_k_genes_requested": top_k_genes,
            "pvalue_cutoff": cutoff,
            "significance_filter_applied": bool(filter_applied),
            "significance_threshold": threshold,
            "min_spots_per_gene": MIN_SPOTS_PER_GENE,
            "n_genes_dropped_sparse": int(n_dropped),
            "n_spots_out_of_tissue_excluded": n_off_tissue,
            "random_seed": random_seed,
        }
    )
    out.add_params(identifier_rename_params(renamed))
    record_method(out, METHOD_NAME, used_fallback=False)
    record_in_tissue(out, n_spots_supplied, n_off_tissue)
    record_expression_source(out, counts_info)
    record_ignored(out, "seed", _NO_RANDOMNESS)
    if matrix_ignored:
        record_ignored(out, matrix_ignored, "use_raw_counts=True tests adata.raw.X, so the named layer was not read")
    out.add_warnings(run_warnings)
    out.set_summary(
        n_significant=n_significant,
        significance_threshold=threshold,
        n_passing_cutoff=n_passing,
        n_genes_reported=len(top_genes),
        n_nan_pvalues=n_nan,
        top_genes=top_genes,
    )

    analysis = build_svg_analysis(
        int(len(result_df)),
        n_significant,
        top_genes,
        method_name="BSP (scbsp)",
        n_genes_renamed=int(renamed.get("n_genes_renamed", 0)),
    )
    analysis += " " + _threshold_sentence(filter_applied, cutoff, n_passing, len(top_genes))
    if n_off_tissue:
        analysis += (
            f" {n_off_tissue} of the {n_spots_supplied} spots were left out because obs['in_tissue'] marks them "
            f"off the tissue; the test ran on the {adata.n_obs} tissue spots."
        )
    if n_dropped:
        analysis += (
            f" {n_dropped} of the {n_genes_supplied} genes supplied were not tested (detected in fewer than "
            f"{MIN_SPOTS_PER_GENE} spots), so a gene absent from this ranking may never have been tested."
        )
    out.set_analysis(analysis)

    return out.to_dict()


def main():
    ap = argparse.ArgumentParser(description="BSP (scbsp) spatially variable gene detection worker")
    ap.add_argument("--st-h5ad", required=True, help="Path to spatial AnnData (.h5ad)")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--spatial-key", default="spatial", help="obsm key for spatial coordinates")
    ap.add_argument("--layer", default="", help="AnnData layer to use (empty=adata.X)")
    ap.add_argument("--top-k-genes", type=int, default=20, help="Number of top SVGs to report")
    ap.add_argument(
        "--pvalue-cutoff",
        type=float,
        default=1.0,
        help="When <1.0, restrict the prediction pool to genes with p<cutoff before taking top-K "
        "(matches manual BSP runner default p<0.05); if no gene passes, the reported list is empty. "
        "Default 1.0 disables the filter.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Accepted and ignored: scbsp draws no random numbers")
    ap.add_argument(
        "--use-raw-counts",
        action="store_true",
        default=False,
        help="Test adata.raw.X instead of X (CELLxGENE exports keep the counts there); refused without adata.raw.",
    )
    args = ap.parse_args()

    try:
        result = run_bsp(
            st_h5ad=args.st_h5ad,
            output_dir=args.output_dir,
            spatial_key=args.spatial_key,
            layer=args.layer,
            top_k_genes=args.top_k_genes,
            pvalue_cutoff=args.pvalue_cutoff,
            random_seed=args.seed,
            use_raw_counts=args.use_raw_counts,
        )
        print(json.dumps(result, default=str))

    except Exception as e:
        eprint(f"[BSP] ERROR: {e}")
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("bsp", str(e), task="svg_identification")
        sys.exit(1)


if __name__ == "__main__":
    main()
