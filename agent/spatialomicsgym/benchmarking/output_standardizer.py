"""Output standardization — converts tool-specific outputs into unified benchmark format.

The benchmark pipeline flow is:
  1. Tool execution (produces raw output)
  2. Output inspection (finds prediction file, validates completeness)
  3. **Output standardization** (this module — normalizes to unified format)
  4. Evaluation (consumes only standardized format)

Standardized formats:
  - Clustering: CSV with [spot_id, predicted_label]
  - SVG: CSV with [gene, score, pvalue, significant]
  - Deconvolution: CSV with spot_id index, cell_type columns, proportion values
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

# Written to the standardized `pvalue` column when the tool reported no significance test at all
# (a boolean flag, a bare gene list). It used to be filled with a literal 0.01, which is a claim
# rather than a value: a reader of the standardized CSV, or of the exported trajectory dataset,
# could not tell a real p=0.01 from a gene list that was simply handed over. Selection is unaffected
# — these paths set `significant` directly and the evaluator reads that column, not this one.
_NO_PVALUE = float("nan")

# Same convention for the `score` column: a tool that reported no ranking statistic at all (BSP, for
# instance, publishes p-values only) gets an empty cell rather than 0.0. A literal 0.0 reads as a
# measured absence of spatial variance, which is a different claim from "not measured".
_NO_SCORE = float("nan")

# Ranking-statistic column names, in the order they are preferred. Lower-cased exact matches.
#
# This list is the whole reason SpatialDE, SOMDE, SpaGFT and squidpy standardized to score=0.0 for
# every gene: it held the names five tools use and none of the names those four use. Any name added
# here must be a per-gene "higher means more spatially variable" statistic — the `significant` column
# is derived from p-values wherever they exist, so this only affects the ranking, never selection.
_SVG_SCORE_COLUMNS = (
    "prost_svg_score",  # PROST, spectral fallback path
    "prost_index",  # ... and its official path: var['PI'], the statistic the package is named for
    "spagft_score",
    "gft_score",  # SpaGFT actually writes this one; "spagft_score" never appears in its output
    # SpotGF's per-gene statistic, the whole content of its SpotGF_scores.txt (SpotGF.py:249).
    # Position in this list is immaterial: no tool writes it alongside another name from the list.
    "spotgf_score",
    "fsv",  # SpatialDE / SOMDE fraction-of-spatial-variance: the published effect size
    "llr",  # ... their likelihood-ratio statistic, used only when FSV is absent
    "morans_i",
    "moran_i",
    "intrinsic",  # SVCA
    "lr_score",
    "ai",  # SVGbit
    "z",  # Hotspot
    "statistic",  # SpVC
    "score",
    "pi",
    "i",  # squidpy writes Moran's I as a bare "I"
    # PROST's var['SIG'], the significance *factor* PI is built from -- a [0, 1] score where large
    # means variable, not a p-value. Last, so `prost_index` wins whenever both are present (which
    # is always, on the official path); it stands alone only if a caller wrote a partial table.
    "sig",
)

#: Significance-column names, **most-corrected first**. Order is the contract: a tool that reports
#: both a raw and an adjusted p-value must be thresholded on the adjusted one, whichever way round
#: it happened to write the columns. Every SVG tool in the registry that reports both writes the raw
#: value first (spark, spvc, spagft, spatialde, somde, hotspot -- 45 recorded files, none the other
#: way round), so scanning the file's own column order picked the uncorrected value every time and
#: called ~15-30k genes at unadjusted p < 0.05. ``output_inspector._inspect_svg`` reads this same
#: tuple; the two used to keep private copies that both looked ordered and neither was.
_SVG_PVALUE_COLUMNS = (
    # Adjusted / FDR-corrected.
    "fdr",
    "padj",
    "p_adj",
    "qval",
    "qvalue",
    "q_value",
    "adjusted_pvalue",
    "adjusted_p_value",
    "pval_norm_fdr_bh",  # squidpy's BH-corrected Moran's I p-value, beside its raw `pval_norm`
    # Raw.
    "pval",
    "p_value",
    "p_values",
    "combined_pvalue",
    # CELINA's CombinedPvals: the Cauchy combination of its 11 kernel tests, per (cell_type, gene);
    # not corrected across genes. Added 2026-09-29 -- before it, a standardized CELINA run carried
    # no significance column at all.
    "combinedpvals",
    "pvalue",
    "pval_norm",
)


def _pick_significance_column(frame, candidates=_SVG_PVALUE_COLUMNS):
    """The most-corrected significance column ``frame`` both *carries* and *populates*.

    The second half of that is not pedantry. squidpy writes all three of its ``*_fdr_bh`` columns
    whether or not it computed them: the recorded ``squidpy_moranI.csv`` for visium_svg has
    ``pval_norm_fdr_bh`` present and NaN in all 32285 rows, beside a populated ``pval_norm``.
    Preferring a name without looking at the values would call zero genes there. An empty column is
    not a stricter answer, it is no answer, so it is skipped for the next candidate.

    Priority ordering is what creates this hazard -- scanning the file's own column order used to
    land on a populated column by accident -- so the guard belongs with the ordering.

    On a duplicate lower-cased name the leftmost column wins (hence ``reversed``), matching the
    score-column picker.
    """
    by_priority = {str(c).lower(): c for c in reversed(list(frame.columns))}
    for name in candidates:
        col = by_priority.get(name)
        if col is None:
            continue
        if _svg_numeric(frame[col]).notna().any():
            return col
    return None


def _svg_numeric(series):
    """A p-value or score column as numbers, reading R's bound notation the way R means it.

    ``format.pval`` writes ``<2e-16`` for the most significant genes, and agent-written tables carry
    tokens like ``n.s.``. The standardizer used to ``float()`` every cell and abort the whole run on
    the first such token, while the inspector coerced ``<2e-16`` to NaN and dropped exactly the most
    significant genes -- three layers, three call sets (hunt 2026-09-30, u31-benchmarking-18). A
    leading bound sign is stripped (``<2e-16`` -> 2e-16, still below any threshold); anything else
    that is not a number is NaN and is never called.
    """
    import pandas as pd

    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype(str).str.strip().str.lstrip("<>= ")
    return pd.to_numeric(series, errors="coerce")


def _pick_score_column(frame):
    """The highest-priority ``_SVG_SCORE_COLUMNS`` name ``frame`` carries *and* populates, or ``None``.

    Matched in the whitelist's own priority order, not in the table's column order: SpatialDE and
    SOMDE emit both FSV and LLR, and which one you get should not depend on how the tool happened to
    order its columns. A whitelisted name in the header is not yet a score -- the column has to hold
    numbers: a tool that declares a statistic column and leaves it empty would otherwise win the
    priority walk on the empty one and flatten every gene to 0.0, discarding a populated column
    beside it. Shared with ``output_inspector`` so both layers rank by the same column.
    """
    by_priority = {str(c).lower(): c for c in reversed(list(frame.columns))}
    return next(
        (
            by_priority[n]
            for n in _SVG_SCORE_COLUMNS
            if n in by_priority and _svg_numeric(frame[by_priority[n]]).notna().any()
        ),
        None,
    )


#: A tool that already states which genes it calls. Matched by exact name, so only the worker that
#: writes it is affected: svca used to encode this same selection as a constant 0.01 "pvalue".
_SVG_SELECTION_COLUMNS = ("top_decile_by_intrinsic",)

#: The column a per-cell-type long table names its groups by (CELINA: one block per cell type).
_SVG_GROUP_COLUMNS = ("cell_type", "celltype")

#: squidpy's own p-value columns. Their presence beside a bare ``I`` / ``C`` is squidpy's signature;
#: Hotspot also writes a ``C`` (a different statistic, large = autocorrelated) but names its p-values
#: ``Pval`` / ``FDR``.
_SQUIDPY_PVALUE_PREFIXES = ("pval_norm", "pval_sim", "pval_z_sim")


class _SvgCalls(NamedTuple):
    """Which rows of an SVG table are its calls, decided once for the inspector and the standardizer."""

    frame: Any  # the table, possibly reduced to one row per gene; RangeIndex
    significant: Any  # bool Series aligned to ``frame``
    pval_col: str | None
    score_col: str | None
    selection_col: str | None
    rule: str  # what selected the calls, for messages
    notes: list[str]


def _svg_calls(frame, gene_col, selected: bool = False) -> _SvgCalls:
    """Decide which genes an SVG table calls, by the one rule both layers apply.

    In order: a column in which the tool states its calls; else a p-value below 0.05; else, for a
    table that carries only a ranking score, the top ``_SCORE_COLUMN_TOP_N`` by that score -- or, when
    ``selected`` (the registry declares the table the tool's own top-k, ``OutputSpec.selection_patterns``),
    every gene in it with a non-zero score; else every gene (and the caller says so). The inspector and
    the standardizer used to keep separate rules and called different sets from one file -- the
    standardizer is what is scored.
    """
    import pandas as pd

    from spatialomicsgym.benchmarking.output_inspector import _SCORE_COLUMN_TOP_N

    frame = frame.reset_index(drop=True)
    notes: list[str] = []
    selection_col = next((c for c in frame.columns if str(c).lower() in _SVG_SELECTION_COLUMNS), None)
    pval_col = _pick_significance_column(frame)
    score_col = _pick_score_column(frame)
    for col in (pval_col, score_col):
        if col is not None:
            frame[col] = _svg_numeric(frame[col])

    genes = frame[gene_col].astype(str).str.strip()
    callable_gene = (genes != "") & (genes != "nan") & ~genes.str.startswith("Blank")

    # CELINA writes one block of rows per cell type, so a gene appears once per cell type with that
    # type's CombinedPvals. Keeping the first row per gene (the standardizer's dedup) silently scored
    # only the first cell type, while the inspector listed a gene once for every type it passed in.
    # Reduced per gene to its smallest p across cell types, in both layers -- the scoring decision
    # recorded for this finding (hunt 2026-09-30, u31-benchmarking-3).
    group_col = next((c for c in frame.columns if str(c).lower() in _SVG_GROUP_COLUMNS), None)
    if pval_col is not None and group_col is not None and genes[callable_gene].duplicated().any():
        n_rows, n_groups = len(frame), frame[group_col].nunique()
        frame = (
            frame.assign(_gene=genes)
            .sort_values(pval_col, na_position="last", kind="stable")
            .drop_duplicates(subset=["_gene"], keep="first")
            .sort_index()
            .drop(columns="_gene")
            .reset_index(drop=True)
        )
        genes = frame[gene_col].astype(str).str.strip()
        callable_gene = (genes != "") & (genes != "nan") & ~genes.str.startswith("Blank")
        notes.append(
            f"long table: {n_rows} rows over {n_groups} {group_col} values reduced to {len(frame)} genes by "
            f"each gene's smallest {pval_col} across {group_col}"
        )

    if selection_col is not None:
        flag = frame[selection_col]
        significant = flag.map(lambda v: bool(v) and str(v).strip().lower() not in ("false", "0", "nan", ""))
        rule = f"{selection_col} flag"
    elif pval_col is not None:
        significant = frame[pval_col] < 0.05
        rule = f"{pval_col} < 0.05"
        # squidpy folds its p one-tailed in whichever direction the statistic departs from its
        # expectation, so a strongly *negatively* autocorrelated gene gets a tiny p too and was called
        # spatially variable. A spatially variable gene is the positive direction, as squidpy_worker
        # filters its own predicted_genes.json: I above E[I] = -1/(n-1), or Geary's C below 1 (hunt
        # 2026-09-30, u31-benchmarking-5). The table does not carry n, so I > 0 stands in for
        # I > E[I]: in (E[I], 0] the z-score is at most |E[I]| / sd(I), far below the 1.645 a
        # one-tailed p < 0.05 needs, so beside the p threshold the two bounds call the same genes.
        if str(pval_col).lower().startswith(_SQUIDPY_PVALUE_PREFIXES):
            columns = {str(c): c for c in frame.columns}
            stat = "I" if "I" in columns else "C" if "C" in columns else None
            if stat is not None:
                values = _svg_numeric(frame[columns[stat]])
                positive = values > 0 if stat == "I" else values < 1
                dropped = int((significant & ~positive).sum())
                significant = significant & positive
                rule += f" with {stat} {'> 0' if stat == 'I' else '< 1'}"
                if dropped:
                    notes.append(
                        f"{dropped} gene(s) with p < 0.05 are negatively autocorrelated ({stat} on the wrong "
                        "side of its expectation) and were not called"
                    )
    elif score_col is not None and selected:
        # The table is already the tool's own selection -- PROST's prost_top_svg_genes.csv is the top
        # n_top_genes it was asked for, 200 in the scored prompt -- so the top-N cut below is not this
        # table's to make: it halved PROST's answer on every benchmarked dataset, a score change nobody
        # decided (hunt 2026-09-30, u31-benchmarking-4 review). Every gene in it is a call, as before
        # the cut existed, bar a score of exactly 0.
        scores = frame[score_col]
        significant = scores.notna() & (scores != 0)
        n_called = int((significant & callable_gene).sum())
        rule = f"{score_col} != 0 in the tool's own selection"
        notes.append(
            f"{score_col} is a spatial score with no threshold, and this table is the tool's own selection "
            f"(its top-k, declared in its registry profile): called its {n_called} genes with a non-zero "
            f"score, without the top-{_SCORE_COLUMN_TOP_N} cut a full score table gets"
        )
    elif score_col is not None:
        # A score with no threshold: the h5ad rule, now for tables too. ``significant = score != 0``
        # called every analysed gene -- svgbit's whole HVG input -- so recall was 1.0 by construction,
        # while the same tool's h5ad took the top of the ranking: one tool, two call sets by container
        # (hunt 2026-09-30, u31-benchmarking-4; scoring decision recorded for that finding). A score of
        # exactly 0 is still never a call -- PROST's PI = 0 means no spatial pattern at all.
        scores = frame.loc[callable_gene, score_col]
        ranked = scores[scores.notna() & (scores != 0)].nlargest(_SCORE_COLUMN_TOP_N)
        significant = pd.Series(frame.index.isin(ranked.index), index=frame.index)
        rule = f"top {len(ranked)} by {score_col}"
        notes.append(
            f"{score_col} is a spatial score, not a p-value: took its top {len(ranked)} of "
            f"{int(callable_gene.sum())} genes by rank because the tool wrote no threshold"
        )
    else:
        significant = pd.Series(True, index=frame.index)
        rule = ""
    significant = significant.fillna(False).astype(bool) & callable_gene
    return _SvgCalls(frame, significant, pval_col, score_col, selection_col, rule, notes)


#: Substrings that name a spot/cell identifier column, most specific first -- so a table carrying
#: both ``spotName`` and ``cell_type`` resolves to the former. Deliberately broader than the exact
#: names searched at the call sites: this list only ever runs after those have all missed.
_SPOT_ID_KEYWORDS = ("spot", "barcode", "obs_names", "cell")


def _find_spot_id_column(frame) -> str | None:
    """A column holding one identifier per row, found by keyword rather than by exact name.

    The exact-name searches (``spot``, ``barcode``, ``spot_id``, ``cell_id``, ``obs_names``) match a
    header or they do not, and a tool that spells it ``spotName`` or ``CellID`` falls through them
    all. Uniqueness is what makes the answer safe to widen this far: an identifier column has one
    value per row, so ``cell_type`` and ``n_cells`` -- which ``cell`` would otherwise match -- are
    rejected on their own contents rather than on a guess about their name.

    Returns ``None`` when nothing qualifies, which is the honest answer for a table that does not
    identify its spots at all.
    """
    n = len(frame)
    if n == 0:
        return None
    for keyword in _SPOT_ID_KEYWORDS:
        for col in frame.columns:
            if keyword in str(col).lower() and frame[col].astype(str).nunique() == n:
                return col
    return None


#: cell2location's export prefix: ``<stat>cell_abundance_w_sf_`` for each of the four statistics
#: ``export_posterior`` writes, then the reference signature's own ``means_per_cluster_mu_fg_`` when
#: the worker kept it (it does). Only the q05 spelling used to be cleaned, so any other statistic's
#: columns reached the evaluator as ``meanscell_abundance_w_sf_means_per_cluster_mu_fg_Astro`` and
#: matched no ground-truth cell type (hunt 2026-09-30, u31-benchmarking-2).
_C2L_COLUMN_PREFIX = re.compile(r"^(?:means|stds|q\d+)cell_abundance_w_sf_(?:means_per_cluster_mu_fg_)?")


def _strip_cell2location_prefix(columns) -> list[str] | None:
    """The cell-type names under cell2location's export prefix, or ``None`` if not every column has it."""
    names = [str(c) for c in columns]
    if names and all(_C2L_COLUMN_PREFIX.match(n) for n in names):
        return [_C2L_COLUMN_PREFIX.sub("", n, count=1) for n in names]
    return None


def standardize_output(
    task_type: str,
    prediction_file: str,
    prediction_key: str,
    output_dir: str,
    tool_name: str = "",
    ground_truth_path: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Dispatch standardization based on task type.

    Returns dict with:
        success: bool
        standardized_file: str — path to standardized output
        format: str — "clustering_csv", "svg_csv", "deconvolution_csv"
        n_predictions: int
        warnings: list[str]
        error: str (if failed)
    """
    result: dict[str, Any] = {
        "success": False,
        "standardized_file": "",
        "format": "",
        "n_predictions": 0,
        "warnings": [],
        "error": "",
    }

    std_dir = Path(output_dir) / "standardized"
    std_dir.mkdir(parents=True, exist_ok=True)

    try:
        if task_type == "spatial_clustering":
            return standardize_clustering_output(
                prediction_file, prediction_key, str(std_dir), tool_name, ground_truth_path=ground_truth_path, **kwargs
            )
        elif task_type == "svg_detection":
            return standardize_svg_output(prediction_file, prediction_key, str(std_dir), tool_name, **kwargs)
        elif task_type == "deconvolution":
            return standardize_deconvolution_output(
                prediction_file,
                prediction_key,
                str(std_dir),
                tool_name,
                ground_truth_path=ground_truth_path,
                **kwargs,
            )
        else:
            result["error"] = f"No standardization for task type: {task_type}"
            return result
    except Exception as e:
        result["error"] = f"Standardization failed: {e}"
        return result


def standardize_clustering_output(
    prediction_file: str,
    prediction_key: str,
    std_dir: str,
    tool_name: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Standardize clustering output to CSV with [spot_id, predicted_label].

    Handles both h5ad and CSV input formats.
    """
    import pandas as pd

    result: dict[str, Any] = {
        "success": False,
        "standardized_file": "",
        "format": "clustering_csv",
        "n_predictions": 0,
        "warnings": [],
        "error": "",
    }

    pred_path = Path(prediction_file)

    # Label values the tool's profile declares as "no cluster" (mist's 'isolated'), dropped like NaN
    # in every branch below, as the inspector drops them (hunt 2026-09-30, u31-benchmarking-11).
    from spatialomicsgym.benchmarking.output_inspector import _unassigned_labels
    from spatialomicsgym.benchmarking.tool_output_registry import get_profile

    unassigned = _unassigned_labels(get_profile(tool_name))

    # Handle Seurat RDS output — look for companion clusters.csv
    if pred_path.suffix == ".rds":
        clusters_csv = pred_path.parent / "clusters.csv"
        if not clusters_csv.exists():
            # Also check for any CSV with cluster in the name
            for csv_file in pred_path.parent.glob("*cluster*.csv"):
                clusters_csv = csv_file
                break
        if clusters_csv.exists():
            pred_path = clusters_csv
            result["warnings"].append(f"Seurat RDS detected, using companion CSV: {clusters_csv.name}")
        else:
            result["error"] = "Seurat RDS output found but no companion clusters.csv"
            return result

    if pred_path.suffix == ".h5ad":
        import anndata as ad

        adata = ad.read_h5ad(prediction_file)

        resolved_key = prediction_key
        if resolved_key not in adata.obs.columns:
            # Try fallback search using the inspector's candidate list
            from spatialomicsgym.benchmarking.output_inspector import (
                _declared_cluster_keys,
                _get_cluster_key_candidates,
                _reference_obs,
                _shipped_with_the_input,
            )

            profile = get_profile(tool_name)
            candidates = _get_cluster_key_candidates(profile)
            declared = _declared_cluster_keys(profile)
            # The inspector's rule, on the same reference: a generic column the input carries value for
            # value came with the data, not from the tool (hunt 2026-09-30, u31-benchmarking-1).
            reference = _reference_obs(kwargs.get("ground_truth_path"))
            found = None
            for c in candidates:
                if c in adata.obs.columns:
                    if c not in declared and _shipped_with_the_input(adata.obs[c], reference, c):
                        result["warnings"].append(f"'{c}' is the input's own column, value for value; not used")
                        continue
                    found = c
                    break
            # SpiceMix factor output — look for factor assignment columns
            if found is None and tool_name in ("run_spicemix",):
                for c in adata.obs.columns:
                    if "factor" in c.lower() or "component" in c.lower():
                        found = c
                        result["warnings"].append(f"Using factor column '{c}' as cluster labels")
                        break

            if found is None:
                result["error"] = f"No cluster column found. Tried: {prediction_key}, {candidates[:5]}"
                return result
            result["warnings"].append(f"Used fallback column '{found}' instead of '{prediction_key}'")
            resolved_key = found

        labels = adata.obs[resolved_key].dropna().astype(str)
        labels = labels[~labels.isin(unassigned)]
        df = pd.DataFrame(
            {
                "spot_id": labels.index.astype(str),
                "predicted_label": labels.values,
            }
        )

    elif pred_path.suffix == ".tsv":
        # Plain TSV domain file (SpaceFlow writes domains.tsv — one label per line)
        with open(prediction_file) as fh:
            labels_list = [line.strip() for line in fh if line.strip()]
        # Try to get spot_ids from an h5ad in the same directory
        h5ad_files = list(pred_path.parent.glob("*.h5ad"))
        spot_ids = [str(i) for i in range(len(labels_list))]
        for hf in h5ad_files:
            # ``except Exception: continue`` is the right behaviour here -- a candidate that cannot
            # be read is simply not the one -- but the two closes used to sit in the ``try`` body,
            # so a candidate that raised on ``n_obs`` (a truncated h5ad from a crashed run, which
            # workflow_gates already warns about) was skipped while its HDF5 lock was still held,
            # and the loop went on to the next file still holding it.
            try:
                from spatialomicsgym.utils.file_io import read_h5ad_backed

                with read_h5ad_backed(str(hf)) as a:
                    if a.n_obs == len(labels_list):
                        spot_ids = [str(s) for s in a.obs_names]
                        break
            except Exception:
                continue

        df = pd.DataFrame(
            {
                "spot_id": spot_ids[: len(labels_list)],
                "predicted_label": labels_list,
            }
        )
        # Positional, so the unassigned rows go after the spot IDs are paired, not before.
        df = df[~df["predicted_label"].isin(unassigned)]
        result["warnings"].append(f"Labels read from plain TSV ({len(labels_list)} entries)")

    elif pred_path.suffix == ".csv":
        raw = pd.read_csv(prediction_file, index_col=0)
        if prediction_key and prediction_key in raw.columns:
            col = prediction_key
        else:
            # Find cluster-like column
            cluster_kw = ["cluster", "domain", "leiden", "louvain", "pred", "label"]
            col = None
            for c in raw.columns:
                if any(kw in c.lower() for kw in cluster_kw):
                    col = c
                    break
            if col is None:
                col = raw.columns[0]
                result["warnings"].append(f"No cluster column found, using first column: {col}")

        # Determine spot_id: use 'spot' or 'barcode' column if present, else use index
        spot_col = None
        for sc in ["spot", "barcode", "spot_id", "cell_id", "obs_names"]:
            if sc in raw.columns:
                spot_col = sc
                break
        if spot_col is None and raw.index.astype(str).nunique() != len(raw):
            # The index came from ``index_col=0`` above, so falling back to it assumes the first
            # column of the file IS the barcodes. IRIS writes ``Slice,spotName,x,y,domain``: the
            # index is then the literal ``Slice1`` on every row, and all 600 spots standardize to
            # one ID while ``spotName`` -- the barcodes -- sits unused in ``raw.columns``. Nothing
            # downstream survives that: ``evaluate_clustering`` aligns strictly by ID with no
            # positional fallback, so a fully labelled answer is scored "No overlapping spot IDs".
            #
            # Only a non-unique index gets here, and a non-unique index cannot produce a correct
            # evaluation today by construction, so widening the search cannot change a run that
            # currently scores.
            spot_col = _find_spot_id_column(raw)
            if spot_col is None:
                result["warnings"].append(
                    f"No unique spot identifier found in {pred_path.name}; falling back to the "
                    f"index, whose {len(raw)} values are not unique. Evaluation aligns by spot_id "
                    "and will not match the ground truth."
                )
        spot_ids = raw[spot_col].astype(str).values if spot_col else raw.index.astype(str)

        # Drop unassigned spots together with their spot IDs. These two arrays used to be derived
        # independently -- the full spot column against a `.dropna()`-ed label column -- so a
        # single missing label (mclust's NA, a filtered spot, an empty field from an R writer)
        # made the lengths differ and pandas raised "All arrays must be of the same length",
        # discarding the entire run. Worse, had the lengths ever agreed the labels would have
        # shifted up across the gap and silently mislabelled every spot after it. The h5ad branch
        # above already keeps the index of the dropna()-ed series; this does the same.
        labels = raw[col]
        assigned = (labels.notna() & ~labels.astype(str).isin(unassigned)).to_numpy()
        n_unassigned = int(labels.size - assigned.sum())
        df = pd.DataFrame(
            {
                "spot_id": pd.Index(spot_ids)[assigned].to_numpy(),
                "predicted_label": labels[assigned].astype(str).to_numpy(),
            }
        )
        if n_unassigned:
            result["warnings"].append(f"Dropped {n_unassigned} unassigned spot(s) with no cluster label")
    else:
        result["error"] = f"Unsupported prediction format: {pred_path.suffix}"
        return result

    # Validate
    if len(df) == 0:
        result["error"] = "Empty prediction after standardization"
        return result

    n_unique = df["predicted_label"].nunique()
    if n_unique <= 1:
        result["error"] = f"Degenerate: only {n_unique} unique label(s)"
        return result

    # Write standardized output
    out_path = Path(std_dir) / f"{tool_name}_clustering_standardized.csv"
    df.to_csv(str(out_path), index=False)

    result["success"] = True
    result["standardized_file"] = str(out_path)
    result["n_predictions"] = len(df)
    result["n_unique_labels"] = int(n_unique)
    return result


def standardize_svg_output(
    prediction_file: str,
    prediction_key: str,
    std_dir: str,
    tool_name: str = "",
    svg_genes: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Standardize SVG output to CSV with [gene, score, pvalue, significant].

    Handles CSV with significance columns, h5ad var annotations, and gene lists.
    """
    import pandas as pd

    result: dict[str, Any] = {
        "success": False,
        "standardized_file": "",
        "format": "svg_csv",
        "n_predictions": 0,
        "warnings": [],
        "error": "",
    }

    pred_path = Path(prediction_file)
    rows: list[dict[str, Any]] = []
    had_pval_col = False  # Track whether any significance column was found
    # Whether the tool reported a per-gene ranking statistic. The other branches (a boolean h5ad
    # flag, a bare JSON gene list) write a constant placeholder score, which is not a ranking and
    # must not be judged as one.
    had_score_col = False

    if pred_path.suffix in (".csv", ".tsv", ".txt"):
        # ``.txt`` belongs here: SpotGF writes its per-gene ranking as a tab-separated
        # SpotGF_scores.txt (SpotGF.py:250), which the registry declares as format "txt" and
        # output_inspector already reads through the same helper. Matching no branch, it used to
        # fall through to the ``svg_genes`` fallback below and come back as one row per gene with a
        # constant score -- a manufactured table, indistinguishable from a real one.
        # _tabular_sep keeps "," for .csv and "\t" for .tsv, so the files that were already being
        # read here are read identically; only the new extension gets its header sniffed.
        from spatialomicsgym.benchmarking.output_inspector import _tabular_sep

        sep = "\t" if pred_path.suffix == ".tsv" else _tabular_sep(pred_path)
        raw = pd.read_csv(prediction_file, sep=sep)

        # Find gene column
        gene_col = None
        for c in raw.columns:
            if c.lower() in ("gene", "gene_name", "g", "feature", "var_name", "genes", "symbol"):
                gene_col = c
                break
        # Check if index looks like gene names
        if gene_col is None:
            idx = raw.index
            if idx.dtype == object or (hasattr(idx, "inferred_type") and idx.inferred_type == "string"):
                # Verify index values look like gene names (not numeric IDs)
                sample = [str(x) for x in idx[:5]]
                if any(not s.replace(".", "").replace("-", "").replace("_", "").isdigit() for s in sample):
                    raw = raw.reset_index()
                    gene_col = raw.columns[0]
                    result["warnings"].append(f"Gene names inferred from index as column '{gene_col}'")

        # Check for 'Unnamed: 0' column containing gene names (CSV where first col is unnamed)
        if gene_col is None:
            for c in raw.columns:
                if str(c).startswith("Unnamed"):
                    sample = [str(x) for x in raw[c].dropna().head(5)]
                    if sample and any(
                        not s.replace(".", "").replace("-", "").replace("_", "").isdigit() for s in sample
                    ):
                        gene_col = c
                        result["warnings"].append(f"Gene names found in unnamed column '{gene_col}'")
                        break

        # Last resort: the column the registry declared for this tool. Deliberately last, after the
        # name whitelist and both heuristics, so every file that resolves a gene column today
        # resolves the identical one -- this can only turn a failure into a success, never move an
        # existing result. It is what rescues SpotGF: `geneID` is in no whitelist, the index is a
        # RangeIndex so the index heuristic declines, and the unnamed first column holds row numbers
        # so the `Unnamed:` heuristic correctly declines too. The registry has always known the
        # answer (`prediction_key=["geneID", "gene", "Gene"]`) and output_inspector already resolves
        # it; this parameter was accepted here and never read.
        if gene_col is None and prediction_key:
            from spatialomicsgym.benchmarking.output_inspector import _resolve_prediction_key

            gene_col = _resolve_prediction_key(raw, prediction_key)

        if gene_col is None:
            result["error"] = "No gene column found in CSV"
            return result

        # BSP and similar tools may use a separate column for gene names
        # (e.g. "gene" column has numeric IDs while "gene_names" has the actual names)
        _needs_gene_index_map = False
        if gene_col is not None:
            sample_genes = [str(x) for x in raw[gene_col].dropna().head(5)]
            if all(s.isdigit() for s in sample_genes):
                # Gene column has numeric IDs — look for a real gene name column
                switched = False
                for alt_col in ("gene_names", "gene_name", "symbol", "feature_name"):
                    if alt_col in raw.columns:
                        # Check if alt_col has real gene names (not synthetic like "Gene_384")
                        alt_sample = [str(x) for x in raw[alt_col].dropna().head(5)]
                        if any(not s.startswith("Gene_") for s in alt_sample):
                            result["warnings"].append(
                                f"Gene column '{gene_col}' contains numeric IDs; using '{alt_col}' instead"
                            )
                            gene_col = alt_col
                            switched = True
                            break
                if not switched:
                    # Still have numeric IDs — need to map from input h5ad
                    _needs_gene_index_map = True

        # Which rows are calls: a stated selection, else p < 0.05, else the top of a score-only
        # ranking, else every gene -- decided by `_svg_calls`, the rule output_inspector applies to
        # the same file. The score and p-value columns are picked in their whitelists' own priority
        # order and must be populated (see `_pick_score_column` / `_pick_significance_column`).
        from spatialomicsgym.benchmarking.output_inspector import _is_declared_selection
        from spatialomicsgym.benchmarking.tool_output_registry import get_profile

        calls = _svg_calls(raw, gene_col, selected=_is_declared_selection(pred_path, get_profile(tool_name)))
        raw = calls.frame
        score_col, pval_col, selection_col = calls.score_col, calls.pval_col, calls.selection_col
        had_score_col = score_col is not None
        # A stated selection is significance information too, so the "no filtering applied" warning
        # below must not fire for a tool that did state its calls.
        had_pval_col = selection_col is not None or pval_col is not None
        result["warnings"].extend(calls.notes)

        # Map numeric gene IDs to real gene names from input h5ad
        _gene_index_map: dict[str, str] = {}
        if _needs_gene_index_map:
            spatial_h5ad = kwargs.get("spatial_h5ad") or kwargs.get("input_h5ad")
            if spatial_h5ad and Path(spatial_h5ad).exists():
                try:
                    from spatialomicsgym.utils.file_io import read_h5ad_backed

                    # Release the handle here rather than leaving it to the collector -- this runs
                    # mid-pipeline and the standardizers rewrite files in place.
                    with read_h5ad_backed(spatial_h5ad) as ref:
                        _gene_index_map = {str(i): name for i, name in enumerate(ref.var_names)}
                    result["warnings"].append(
                        f"Mapped {len(_gene_index_map)} numeric gene IDs to names from input h5ad"
                    )
                except Exception:
                    pass

        for idx, row in raw.iterrows():
            gene = str(row[gene_col]).strip()
            # Map numeric gene ID to real gene name if available
            if _needs_gene_index_map and gene in _gene_index_map:
                gene = _gene_index_map[gene]
            if not gene or gene.startswith("Blank") or gene == "nan" or gene.startswith("Gene_"):
                continue
            if not score_col:
                # No ranking column at all. _NO_SCORE, not 0.0: "not measured" is a different claim
                # from a measured absence of spatial variance.
                score = _NO_SCORE
            else:
                # A blank cell in a real score column is written as 0.0, as it always was; it is never
                # a call (`_svg_calls` ranks only populated cells).
                score = float(row[score_col]) if pd.notna(row.get(score_col)) else 0.0
            # No p-value column (or an empty cell) means the tool reported no significance test.
            # Record that as _NO_PVALUE, the module-wide convention explained at the top of this
            # file -- NOT a literal 1.0. A hardcoded 1.0 is a claim the tool never made, and paired
            # with a score-derived `significant` it produced rows reading "pvalue=1.0,
            # significant=True", which is self-contradictory to any reader of the standardized CSV or
            # the exported trajectory dataset.
            pval = float(row[pval_col]) if pval_col and pd.notna(row.get(pval_col)) else _NO_PVALUE
            significant = bool(calls.significant.loc[idx])
            rows.append({"gene": gene, "score": score, "pvalue": pval, "significant": significant})

        if not pval_col and not score_col and not selection_col and rows:
            result["warnings"].append(
                "No significance or score columns found — all output genes treated as SVG candidates"
            )

    elif pred_path.suffix == ".h5ad":
        import anndata as ad

        # The same three-way rule output_inspector applies, from the same helpers. This branch used
        # to re-scan `var` and decide from dtype alone -- the rule R74 replaced there and left here.
        # It matters because the two answers are not both discarded: the inspector's choice is
        # recorded in `prediction_key` and its genes in `svg_genes`, but evaluate_tool_output prefers
        # the CSV this function writes, so wherever they disagreed the unfenced answer won.
        from spatialomicsgym.benchmarking.output_inspector import _SCORE_COLUMN_TOP_N, _svg_column_kind

        adata = ad.read_h5ad(prediction_file)
        # The column the inspector already settled on, then the keyword matches -- in `var` order, so
        # a file it never inspected reads as before. `prediction_key` is only honoured when it names
        # a real column: for a CSV prediction it holds the *gene* column instead, and a caller that
        # passes one of those must not be able to make this branch read it as significance.
        candidates = [prediction_key] if prediction_key and prediction_key in adata.var.columns else []
        candidates += [
            vc
            for vc in adata.var.columns
            if vc not in candidates
            and any(kw in vc.lower() for kw in ["svg", "spatially_variable", "significant", "hotspot"])
        ]
        for vc in candidates:
            col = adata.var[vc]
            if col.dtype == bool:
                # A boolean flag says which genes were called, not how confidently. The p-value
                # stays empty rather than being filled with a plausible-looking 0.01 that no
                # test produced; `significant` is what the evaluator reads.
                for gene in adata.var_names[col]:
                    rows.append({"gene": str(gene), "score": 1.0, "pvalue": _NO_PVALUE, "significant": True})
            elif _svg_column_kind(vc) == "identifier":
                # Numbers that label rather than rank. Hotspot writes `hotspot_module`, -1 for the
                # genes it could not assign, and thresholding it at 0.05 called modules -1 and 0.
                result["warnings"].append(
                    f"{vc} in {pred_path.name} looks like a module/cluster identifier, not a "
                    "significance column; it was not read as an SVG result"
                )
                continue
            elif str(col.dtype) in ("float64", "float32"):
                kind = _svg_column_kind(vc)
                if kind == "score":
                    # Moran's I, SpatialDE/SOMDE's FSV, SpaGFT's gft_score: large means spatially
                    # variable, so `< 0.05` selected precisely the genes the tool rejected. Rank
                    # instead, and put the statistic in the column named for it.
                    # dropna first: nlargest sorts NaN to the end rather than discarding it, so an
                    # all-NaN score column -- a tool that ran and computed nothing -- comes back as
                    # a full top-N, which is the first N genes in file order wearing a ranking.
                    top = {str(g) for g in col.dropna().nlargest(_SCORE_COLUMN_TOP_N).index}
                    if top:
                        had_score_col = True
                        for gene, val in zip(adata.var_names, col.to_numpy(), strict=False):
                            rows.append(
                                {
                                    "gene": str(gene),
                                    "score": float(val),
                                    "pvalue": _NO_PVALUE,
                                    "significant": str(gene) in top,
                                }
                            )
                        result["warnings"].append(
                            f"{vc} is a spatial score, not a p-value: took its top {len(top)} of "
                            f"{adata.n_vars} genes by rank because the tool wrote no threshold"
                        )
                else:
                    # p-value, or a name that says neither -- keep the established rule.
                    had_pval_col = True
                    for gene in adata.var_names:
                        val = float(col[gene])
                        rows.append({"gene": str(gene), "score": 0.0, "pvalue": val, "significant": val < 0.05})
                    if kind == "unknown":
                        result["warnings"].append(
                            f"{vc} was thresholded at 0.05 as a p-value, but its name does not say "
                            "whether it holds one; verify the direction"
                        )
            else:
                continue
            # A candidate that produced nothing has not answered the question, so the search goes on
            # -- the unconditional `break` this replaces let an unusable first match hide a real
            # p-value column one position later.
            if rows:
                break

    elif pred_path.suffix == ".json":
        import json as _json

        with open(prediction_file) as fh:
            data = _json.load(fh)
        gene_list: list[str] = []
        if isinstance(data, list):
            gene_list = [str(g) for g in data if g]
        elif isinstance(data, dict):
            for key in (
                "predicted_genes",
                "top_genes",
                "svg_genes",
                "spatially_variable_genes",
                "genes",
                "significant_genes",
            ):
                if isinstance(data.get(key), list) and data[key]:
                    gene_list = [str(g) for g in data[key] if g]
                    break
            if not gene_list and isinstance(data.get("summary"), dict):
                for key in ("predicted_genes", "top_genes", "svg_genes", "spatially_variable_genes", "genes"):
                    if isinstance(data["summary"].get(key), list) and data["summary"][key]:
                        gene_list = [str(g) for g in data["summary"][key] if g]
                        break
        for g in gene_list:
            g = g.strip()
            if g and not g.startswith("Blank") and g != "nan":
                # A bare gene list carries no significance information; do not manufacture one.
                rows.append({"gene": g, "score": 1.0, "pvalue": _NO_PVALUE, "significant": True})

    # Fallback: use pre-extracted gene list
    if not rows and svg_genes:
        for gene in svg_genes:
            gene = str(gene).strip()
            if gene and not gene.startswith("Blank") and gene != "nan":
                rows.append({"gene": gene, "score": 0.0, "pvalue": _NO_PVALUE, "significant": True})

    if not rows:
        result["error"] = "No SVG genes found after standardization"
        return result

    df = pd.DataFrame(rows)
    # Deduplicate
    df = df.drop_duplicates(subset=["gene"], keep="first")

    out_path = Path(std_dir) / f"{tool_name}_svg_standardized.csv"
    df.to_csv(str(out_path), index=False)

    # A prediction whose genes are not in the assay is a prediction for a different dataset. Two
    # recorded MERFISH cells (somde_run, spatialde_run_svg) are Visium results filed under
    # merfish_svg -- 97.6% and 98.5% of their predicted genes do not exist in the 649-gene panel --
    # and both were scored anyway, because a Visium result is a perfectly well-formed SVG table and
    # nothing ever compared it against the dataset it was filed under.
    #
    # Reported, never enforced: this must not silently re-score a benchmark. And case-only
    # differences are excluded, because gene symbols are cased inconsistently across tools and that
    # is a naming difference, not a different dataset.
    ref_h5ad = kwargs.get("spatial_h5ad") or kwargs.get("input_h5ad")
    if ref_h5ad and Path(ref_h5ad).exists():
        try:
            from spatialomicsgym.utils.file_io import read_h5ad_backed

            # The un-named temporary this replaced could not be closed at all -- nothing held a
            # reference to hand back, so the handle survived until the collector reclaimed it.
            with read_h5ad_backed(ref_h5ad) as _ref_adata:
                ref = {str(g) for g in _ref_adata.var_names}
        except Exception:
            ref = set()
        if ref:
            ref_lower = {g.lower() for g in ref}
            off = [g for g in map(str, df["gene"]) if g not in ref and g.lower() not in ref_lower]
            if len(off) > 0.5 * len(df):
                result["warnings"].append(
                    f"{len(off)} of the {len(df)} predicted genes are not among the "
                    f"{len(ref)} genes in the input dataset — this prediction appears to come from a "
                    f"different dataset (e.g. {', '.join(off[:3])})"
                )

    sig_count = int(df["significant"].sum())

    # Validate standardized output
    if sig_count == 0:
        result["warnings"].append(
            "No significant genes after standardization — tool may have produced incomplete output"
        )
    if sig_count == len(df) and not had_pval_col:
        result["warnings"].append(
            "All genes marked significant — no p-value filtering applied, output may be incomplete"
        )
    # A score column with a single distinct value carries no ranking: every gene is tied, so any
    # top-N selection is really just input order and recall against a ground-truth set is 1.0 by
    # construction. `output_inspector` already rejects an all-identical *clustering* label column as
    # "likely a failed run"; SVG had no equivalent, which is how SVCA output where every gene scored
    # exactly 1/3 (the degenerate fixed point of the variance decomposition, reported with
    # converged=True) was standardized into "every gene is a significant SVG" and scored. Warn only
    # above a handful of genes — three genes that happen to tie is not evidence of anything.
    # ...and only when the tool actually reported one. Without the `had_score_col` guard this fired
    # on all-NaN score columns too, i.e. on tools that never claimed to rank anything, blaming them
    # for a failed run they did not have.
    if had_score_col and len(df) > 10 and df["score"].nunique(dropna=True) <= 1:
        result["warnings"].append(
            f"Every one of the {len(df)} genes received an identical score "
            f"({df['score'].iloc[0]!r}) — the tool produced no ranking; likely a failed or "
            "non-converged run rather than a result"
        )

    result["success"] = True
    result["standardized_file"] = str(out_path)
    result["n_predictions"] = len(df)
    result["n_significant"] = sig_count
    return result


#: Where SpaCET's lineage levels are read from when the caller does not pass them: the worker
#: payload's ``data.major_lineages`` / ``data.sub_lineages``, written beside ``spacet_proportions.csv``.
_SPACET_LINEAGE_LEVELS_FILE = "spacet_lineage_levels.json"


def _spacet_major_lineages(pred_path: Path, kwargs: dict[str, Any]) -> list[str] | None:
    """SpaCET's major-lineage names: ``major_lineages=`` from the caller, else the levels file beside the table."""
    levels = kwargs.get("major_lineages")
    if not levels:
        side = Path(pred_path).parent / _SPACET_LINEAGE_LEVELS_FILE
        try:
            import json

            data = json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}
        except (OSError, ValueError):
            data = {}
        if isinstance(data, dict) and isinstance(data.get("data"), dict):
            data = data["data"]  # the whole payload, not just its levels
        levels = data.get("major_lineages") if isinstance(data, dict) else None
    if isinstance(levels, str):
        levels = [levels]
    return [str(x) for x in levels] if isinstance(levels, (list, tuple)) and levels else None


def _orient_proportions(pred, wide_only: bool = False):
    """``pred`` with spots as rows, and the reason it was transposed (``None`` when it was not).

    Shared with ``output_inspector`` and ``workflow_gates`` so every reader of a proportions table
    puts its spots on the same axis. Only this module used to transpose: the inspector reported
    SpaCET's cell-type x spot ``propMat`` as 13 "cell types" named by spot barcodes, and the gates'
    row-sum and dominance checks ran on the wrong axis (hunt 2026-09-30, u31-benchmarking-9). Those
    readers pass ``wide_only``: the wide case is the one the numbers or the barcodes settle, while the
    tall case rests on a name heuristic alone, and they did not inherit it.
    """
    import numpy as np

    def _looks_like_barcodes(names: list[str]) -> bool:
        """Spot barcodes: contain dashes, are long alphanumeric, or match barcode pattern."""
        return any("-" in n or len(n) > 15 or (len(n) > 10 and n.isalnum()) for n in names)

    def _looks_like_celltype_names(names: list[str]) -> bool:
        """Cell type names: short, may have spaces/underscores, not purely numeric."""
        return all(len(n) < 30 and not n.isdigit() and "-" not in n for n in names) and not _looks_like_barcodes(names)

    def _sums_to_one(axis: int) -> bool:
        """Do the vectors along `axis` sum to 1, the way a proportion vector must?

        The orientation heuristics below read the index *labels*, which is how a correct
        6142 x 7 bulk2space matrix came to be stored as 7 x 6142: its spots were named
        `spot_0, spot_1, ...`, and "short, no dash, not all digits" is indistinguishable
        from a cell type name. The numbers do not have that ambiguity -- deconvolution
        proportions sum to 1 across cell types -- so ask them first.
        """
        sums = pred.sum(axis=axis).to_numpy(dtype=np.float64)
        sums = sums[np.isfinite(sums)]
        return bool(len(sums)) and bool(np.isclose(np.median(sums), 1.0, atol=0.02))

    sample_idx = [str(x) for x in pred.index[:5]]
    sample_cols = [str(x) for x in pred.columns[:5]]

    # Case 1: rows=celltypes, cols=spots (few rows, many cols)
    if pred.shape[1] > pred.shape[0] * 2:
        # Columns summing to 1 in a matrix this wide means the columns are the spots. Only
        # checked on wide matrices: Tangram and cell2location write *tall* matrices whose
        # columns also sum to 1 (each cell type's mapping distributed over spots), and those
        # are already the right way round.
        if (
            _sums_to_one(axis=0)
            or _looks_like_barcodes(sample_cols)
            or (_looks_like_celltype_names(sample_idx) and not _looks_like_barcodes(sample_idx))
        ):
            pred = pred.T
            return pred, f"Transposed: cols were spots, rows were cell types, now {pred.shape}"
    # Case 2: rows have many more entries than cols, and index doesn't look like barcodes
    #
    # Note this shape test is satisfied by *every* correctly-oriented deconvolution matrix --
    # there are always far more spots than cell types -- so the name heuristic is the only
    # thing standing between a correct file and a corrupted one. Rows that already sum to 1
    # settle it: they are proportion vectors, so the rows are the spots. Leave it alone.
    elif (
        not wide_only
        and pred.shape[0] > pred.shape[1] * 2
        and not _sums_to_one(axis=1)
        and _looks_like_celltype_names(sample_idx)
        and not _looks_like_barcodes(sample_idx)
    ):
        pred = pred.T
        return pred, f"Transposed: rows were celltype names, now {pred.shape}"
    return pred, None


def standardize_deconvolution_output(
    prediction_file: str,
    prediction_key: str,
    std_dir: str,
    tool_name: str = "",
    ground_truth_path: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Standardize deconvolution output to CSV with spot_id index and cell_type columns.

    Handles:
    - CSV proportion matrices (most tools)
    - h5ad obsm abundance matrices (Cell2Location)
    - Row normalization to sum to ~1
    - Index alignment with ground truth when needed
    """
    import numpy as np
    import pandas as pd

    result: dict[str, Any] = {
        "success": False,
        "standardized_file": "",
        "format": "deconvolution_csv",
        "n_predictions": 0,
        "warnings": [],
        "error": "",
    }

    pred_path = Path(prediction_file)

    if pred_path.suffix == ".csv":
        raw = pd.read_csv(prediction_file)
        # If 'spot' or 'barcode' column exists, use it as index
        spot_col = None
        for sc in ["spot", "barcode", "spot_id", "cell_id"]:
            if sc in raw.columns:
                spot_col = sc
                break
        if spot_col:
            raw = raw.set_index(spot_col)
        else:
            # Use first column as index (default)
            raw = pd.read_csv(prediction_file, index_col=0)

        # Ensure numeric columns only
        numeric = raw.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            result["error"] = f"Only {numeric.shape[1]} numeric columns — need at least 2 cell types"
            return result
        pred = numeric

        # Check if matrix is transposed
        pred, transposed = _orient_proportions(pred)
        if transposed:
            result["warnings"].append(transposed)

    elif pred_path.suffix == ".h5ad":
        import anndata as ad

        from spatialomicsgym.benchmarking.output_inspector import _abundance_obsm_key

        adata = ad.read_h5ad(prediction_file)
        # The obsm key the inspector picked, not the first keyword hit in alphabetical obsm order --
        # for cell2location that was the posterior means, not the q05 it scored (hunt 2026-09-30,
        # u31-benchmarking-2).
        found_key = _abundance_obsm_key(adata.obsm, prediction_key)
        if found_key is None:
            result["error"] = "No deconvolution matrix found in obsm"
            return result

        mat = adata.obsm[found_key]
        # If obsm value is a DataFrame, extract column names directly
        col_names = None
        if hasattr(mat, "columns"):
            col_names = [str(c) for c in mat.columns]
            mat = mat.values
        elif hasattr(mat, "toarray"):
            mat = mat.toarray()
        mat = np.array(mat, dtype=np.float64)

        # Fall back to uns lookup only if column names weren't extracted from DataFrame
        if col_names is None:
            col_names = [f"celltype_{i}" for i in range(mat.shape[1])]
            # Check uns for cell type names
            for uns_key in adata.uns:
                if "cell_type" in uns_key.lower() or "abundance" in uns_key.lower():
                    val = adata.uns[uns_key]
                    if hasattr(val, "tolist") and len(val) == mat.shape[1]:
                        col_names = [str(v) for v in val.tolist()]
                        break

        pred = pd.DataFrame(mat, index=adata.obs_names, columns=col_names)
        result["warnings"].append(f"Extracted obsm['{found_key}'] ({mat.shape})")

    elif pred_path.suffix in (".tsv", ".txt"):
        # A delimited text table is a table whether it is named .tsv or .txt — STRIDE
        # writes <prefix>_spot_celltype_frac.txt (tab-separated, 200 spots x 5 cell
        # types), which the inspector accepts and this branch used to refuse outright.
        # Sniff with the inspector's own helper so both sides parse the picked file
        # identically; otherwise the standardized artifact would describe a different
        # table than the one that was inspected. .tsv keeps its hard-coded tab.
        from spatialomicsgym.benchmarking.output_inspector import _tabular_sep

        sep = "\t" if pred_path.suffix == ".tsv" else _tabular_sep(pred_path)
        pred = pd.read_csv(prediction_file, sep=sep, index_col=0)
        numeric = pred.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            result["error"] = f"Only {numeric.shape[1]} numeric columns in {pred_path.suffix.lstrip('.').upper()}"
            return result
        pred = numeric

    else:
        result["error"] = f"Unsupported format: {pred_path.suffix}"
        return result

    # SpaCET's table is two lineage levels in one: the major lineages sum to 1 in every spot and the
    # sub-lineages each sum to their parent, so per-spot totals run 1.00-1.99. Normalising every
    # column by that total shrank every fraction and counted each sub-lineage cell twice (Malignant
    # 0.0835 -> 0.0646, majors summing to 0.77). Scoring decision recorded for this finding: SpaCET is
    # scored on its major lineages only, as the worker's payload names them, before normalisation --
    # and without those names a hierarchical table is refused rather than guessed at (hunt
    # 2026-09-30, u31-benchmarking-10).
    if tool_name == "spacet_deconvolution":
        major = _spacet_major_lineages(pred_path, kwargs)
        if major:
            missing = [m for m in major if m not in pred.columns]
            if missing:
                result["error"] = (
                    f"SpaCET's major lineages {missing[:5]} are not columns of {pred_path.name}; "
                    "the lineage levels describe a different table"
                )
                return result
            dropped = [str(c) for c in pred.columns if c not in major]
            pred = pred[major]
            if dropped:
                result["warnings"].append(
                    f"SpaCET: scored its {len(major)} major lineages; dropped {len(dropped)} sub-lineage "
                    f"column(s) that split them (e.g. {', '.join(dropped[:3])})"
                )
        else:
            totals = pred.sum(axis=1)
            if bool((totals > 1.02).any()):
                result["error"] = (
                    f"{pred_path.name} holds two SpaCET lineage levels (per-spot totals up to "
                    f"{float(totals.max()):.2f}; the major lineages alone sum to 1), and the names of the "
                    "major lineages are not available: pass major_lineages (the worker payload's "
                    f"data.major_lineages) or write {_SPACET_LINEAGE_LEVELS_FILE} beside the table. "
                    "Normalising both levels together counts every sub-lineage cell twice."
                )
                return result

    # Normalize rows to sum to ~1 (proportions)
    row_sums = pred.sum(axis=1)
    non_zero = row_sums > 0
    if non_zero.sum() < pred.shape[0]:
        n_zero = pred.shape[0] - int(non_zero.sum())
        result["warnings"].append(f"{n_zero} rows with zero sum (will be kept as-is)")

    pred_norm = pred.copy()
    pred_norm.loc[non_zero] = pred.loc[non_zero].div(row_sums[non_zero], axis=0)

    # Handle negative values
    if (pred_norm < 0).any().any():
        result["warnings"].append("Negative values clipped to 0 before normalization")
        pred_norm = pred_norm.clip(lower=0)
        row_sums2 = pred_norm.sum(axis=1)
        non_zero2 = row_sums2 > 0
        pred_norm.loc[non_zero2] = pred_norm.loc[non_zero2].div(row_sums2[non_zero2], axis=0)

    # Normalize cell type column names
    pred_norm.columns = [c.strip() for c in pred_norm.columns]

    # Clean Cell2Location-style verbose column names
    # e.g. 'q05cell_abundance_w_sf_means_per_cluster_mu_fg_Astro' -> 'Astro'
    cleaned = _strip_cell2location_prefix(pred_norm.columns)
    if cleaned is not None:
        pred_norm.columns = cleaned
        result["warnings"].append("Cleaned Cell2Location verbose column names")

    # Also handle other common prefixes
    for prefix_pattern in [r"^means_per_cluster_mu_fg_", r"^q\d+_nUMI_factors_per_groups_"]:
        if all(re.match(prefix_pattern, c) for c in pred_norm.columns):
            pred_norm.columns = [re.sub(prefix_pattern, "", c) for c in pred_norm.columns]
            result["warnings"].append(f"Cleaned column prefix pattern: {prefix_pattern}")

    # Align with ground truth if provided (handles Cell2Location off-tissue issue)
    if ground_truth_path:
        try:
            gt_path = Path(ground_truth_path)
            if gt_path.suffix == ".h5ad":
                import anndata as ad

                gt_adata = ad.read_h5ad(str(gt_path))
                gt_index = set(gt_adata.obs_names.astype(str))
                pred_index = set(pred_norm.index.astype(str))
                common = gt_index & pred_index
                if len(common) < len(pred_norm) and len(common) > 0:
                    result["warnings"].append(f"Aligned: {len(pred_norm)} pred spots -> {len(common)} common with GT")
                    pred_norm = pred_norm.loc[pred_norm.index.astype(str).isin(common)]
        except Exception as e:
            result["warnings"].append(f"Ground truth alignment skipped: {e}")

    if len(pred_norm) == 0:
        result["error"] = "Empty prediction after standardization"
        return result

    # Set index name
    pred_norm.index.name = "spot_id"

    out_path = Path(std_dir) / f"{tool_name}_deconvolution_standardized.csv"
    pred_norm.to_csv(str(out_path))

    result["success"] = True
    result["standardized_file"] = str(out_path)
    result["n_predictions"] = len(pred_norm)
    result["n_cell_types"] = int(pred_norm.shape[1])
    return result
