"""Spatially variable genes: a ranking, and whether the ranking distinguishes anything.

An SVG table is either a *significance* result (a p-value or an FDR to threshold at 0.05) or a
*ranking* result (Moran's I, an FSV, a GFT score -- monotone, with no threshold the tool would
defend). The two are reported differently on purpose: calling the top-N of a ranking "significant"
is how a tool with no test ends up compared against one that has one.

Two recorded failure modes are checked before anything is ranked:

* a **constant** score column -- every gene scores the same, so every gene "wins" and recall is 1.0
  by construction. Two tools in this repo have produced exactly that;
* a **degenerate** p-value column -- when >90% of the transcriptome passes, the threshold is not
  selecting anything, and that is a property of the result, not a finding about the tissue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialomicsgym.postanalysis import plots
from spatialomicsgym.postanalysis.detect import match_authoritative
from spatialomicsgym.postanalysis.manifest import step
from spatialomicsgym.postanalysis.sources import align_to_coordinates, find_coordinates, is_staged_input
from spatialomicsgym.postanalysis.tables import (
    column_kind,
    is_signal_free,
    is_tabular,
    looks_like_gene_table,
    numeric_frame,
    read_table,
)

if TYPE_CHECKING:
    from spatialomicsgym.postanalysis.context import AnalysisContext

#: The threshold every tool in this repo that reports a p-value reports it against.
_ALPHA = 0.05

#: With no p-value to threshold, this many top-ranked genes are reported as the shortlist.
_TOP_N = 50

#: Above this pass rate the threshold is not selecting anything.
_DEGENERATE_FRACTION = 0.9

#: Genes mapped in the spatial grid, when coordinates and an expression matrix are both available.
_MAX_MAPPED_GENES = 6


def run(ctx: AnalysisContext) -> None:
    picked = _pick_gene_table(ctx)
    if picked is None:
        ctx.manifest.fail(
            "No per-gene result table found in the tool output "
            f"({len(ctx.files)} files scanned, none ranked gene symbols by a numeric column)"
        )
        return

    frame, gene_key, score_column, source = picked
    ctx.analysed(source)

    import pandas as pd

    genes = frame.index.astype(str) if gene_key is None else frame[gene_key].astype(str)
    scores = frame[score_column]

    # Counted over unique symbols, not rows. A table with one row per gene *per slice* -- or one
    # that simply repeats a symbol, which several of these tools do -- reported 200 genes tested
    # for 50 genes, and every count derived from it inherited the wrong denominator. The unit of
    # this analysis is the gene; :func:`_rank` collapses the rows to match.
    symbols = pd.Index(genes)
    repeated = symbols[symbols.duplicated()].unique()

    # ``_rank`` is computed *here*, before anything is reported, because it is what decides how many
    # genes this analysis actually has: it drops every row whose score is not numeric. Counting the
    # symbols instead published two different totals for one quantity in one manifest -- a 400-gene
    # table with 100 unscored genes said ``n_genes: 400`` next to a 300-row ``gene_ranking.csv``, a
    # "300 genes" histogram caption and a ``significant_fraction`` over 300. ``n_genes`` is the
    # denominator every other number here is taken over, so it has to be that same 300; the genes
    # the tool listed without scoring are a *different* quantity and get their own key and label.
    is_pvalue = column_kind(score_column) == "pvalue"
    ranked = _rank(frame, genes, scores, score_column, ascending=is_pvalue)
    unscored = int(symbols.nunique() - len(ranked))

    ctx.find("n_genes", int(len(ranked)), "Genes tested")
    if unscored:
        ctx.find("n_genes_unscored", unscored, f"Genes listed with no numeric {score_column}")
    ctx.find("score_column", str(score_column), "Column the ranking was read from")
    if unscored:
        ctx.warn(
            f"{symbols.nunique()} genes are listed in {source.name}, but {unscored} of them carry no "
            f"numeric {score_column!r} and are not ranked. n_genes and every count below are over the "
            f"{len(ranked)} genes that do."
        )
    if len(repeated):
        shown = ", ".join(str(g) for g in repeated[:5])
        more = f" and {len(repeated) - 5} more" if len(repeated) > 5 else ""
        ctx.warn(
            f"{len(frame)} rows cover {symbols.nunique()} genes: {len(repeated)} symbol(s) appear more "
            f"than once ({shown}{more}). Each gene is ranked once, on its best-scoring row, so the "
            "counts below are per gene rather than per row."
        )

    signal_free = is_signal_free(scores.to_numpy())
    ctx.find("signal_free", signal_free, "Result carries no information")
    if signal_free:
        # ``len(ranked)``, not ``len(frame)`` and not ``symbols.nunique()``: on the per-slice table
        # the warning above has just finished explaining ("300 rows cover 100 genes"), a row count
        # here contradicts it inside one manifest -- and it is the same wrong denominator this
        # module was written to stop reporting. It has to be the number ``n_genes`` reports, which
        # is the genes that were scored, not the genes that were listed.
        ctx.warn(
            f"Signal-free result: {score_column!r} is constant across all {len(ranked)} genes, so "
            "every gene ranks equally. A constant score scores recall 1.0 by construction; the ranking "
            "was not used."
        )
        ctx.manifest.degrade()

    # Empty when the shortlist below is a genuine selection -- genes a test picked out. Otherwise
    # the parenthetical its titles carry, because "spatially variable gene" is a claim and
    # "best-scoring" is not. The two qualified cases do not read the same and must not: a test can
    # come back empty, or there can have been no test at all.
    note = ""

    if is_pvalue:
        significant = ranked[ranked[score_column] <= _ALPHA]
        fraction = len(significant) / max(len(ranked), 1)
        ctx.find("n_significant", int(len(significant)), f"Genes with {score_column} <= {_ALPHA}")
        ctx.find("significant_fraction", round(float(fraction), 4), "Share of genes passing the threshold")
        if fraction >= _DEGENERATE_FRACTION and not signal_free:
            ctx.warn(
                f"Degenerate threshold: {fraction:.0%} of {len(ranked)} genes pass {score_column} <= "
                f"{_ALPHA}. A test that passes almost everything is not selecting spatial genes; rank "
                "by effect size instead of treating this as a gene set."
            )
            ctx.manifest.degrade()
        if len(significant):
            shortlist = significant
        else:
            # The fallback is right -- an empty table helps nobody, and the branch below does the
            # same thing. What was missing is that the branch below *says* so, and this one did not:
            # two identical shortlists, one disclosed and one published under a title claiming a
            # significance call the tool never made. Degrading matches the mirror-image case just
            # above, where a threshold everything passes is also not a selection.
            ctx.warn(
                f"No gene passed {score_column} <= {_ALPHA}: of {len(ranked)} genes tested, none is "
                f"significant. The {min(len(ranked), _TOP_N)} best-scoring genes are listed below as a "
                "shortlist and not a significance call."
            )
            ctx.manifest.degrade()
            shortlist = ranked.head(_TOP_N)
            note = "none significant"
    else:
        ctx.warn(
            f"{score_column!r} is a ranking statistic, not a test: the tool reports no p-value, so the "
            f"top {_TOP_N} genes are a shortlist and not a significance call."
        )
        shortlist = ranked.head(_TOP_N)
        note = "ranking, not a test"

    _tables(ctx, ranked, shortlist, score_column, note=note)
    # ``ranked``, not ``scores``: one value per gene, the same collapse `gene_ranking.csv` and
    # `significant_fraction` are built on. Drawn from ``scores`` the histogram put one bar's worth
    # of weight on every *row*, so a gene measured on three slices counted three times -- and the
    # caption then called the row count a gene count. On a one-row-per-gene table, which is every
    # checked-in fixture, the two series hold the same values and nothing moves.
    _distribution(ctx, ranked[score_column], score_column, is_pvalue, signal_free)
    if not signal_free:
        _top_bar(ctx, ranked, shortlist, score_column, is_pvalue, note=note)
        _spatial(ctx, shortlist, gene_key, note=note)


# ---------------------------------------------------------------------------------------


def _pick_gene_table(ctx: AnalysisContext):
    """``(frame, gene_key_or_None, score_column, path)`` -- the tool's own output preferred.

    Every parseable gene table is read and the best one wins, rather than the first one found.
    "First" meant alphabetical whenever the registry could not name an authoritative output, which
    it cannot whenever ``tool_name`` is unset -- and on the recorded SpatialDE run that picked
    ``posthoc_top20_spatialde_svgs.csv`` (20 genes, every qval tied at 7.29e-17, a *derived excerpt*
    of the result) over ``spatialde_results.csv`` (2000 genes, 448 distinct qvals, the result). A
    tied score column is what :func:`is_signal_free` exists to catch, so a healthy run was reported
    signal-free and L2 called it suspicious. The tool had even left a
    ``posthoc_spatialde_summary.json`` naming the real file.

    Ranking, in order: the registry's authoritative output if there is one; then a score column that
    actually varies over a table that has been ranked down to nothing; then gene count. Registry
    first and not "most genes" first, because a tool that legitimately publishes a shortlist has
    said so in the registry, and a bigger neighbouring file must not outvote that.
    """
    candidates = [f for f in ctx.files if not is_staged_input(f)]
    tabular = [f for f in candidates if is_tabular(f)]
    preferred = set(match_authoritative(candidates, ctx.tool_name)) & set(tabular)

    best = None
    best_rank: tuple[int, int, int] = (-1, -1, -1)
    # Preferred first, as ``_pick_proportions`` and ``_pick_labels`` already visit them. The break
    # below can only fire on a preferred file, so in filename order it fires after everything that
    # sorts ahead of the answer has already been through ``pd.read_csv``: on the recorded hotspot run
    # that is ``hotspot_local_correlations_z.csv``, 2.97 GB of 12,428 x 12,428 gene-by-gene
    # correlations read in full to conclude it is not a gene ranking. Which files that hits is an
    # accident of spelling -- ``svca_variance_decomposition.csv`` is authoritative and sorts *after*
    # ``svca_summary.csv`` -- so the two big cases are safe only by luck.
    #
    # Partitioning ``tabular`` keeps the original order inside each group, which is what makes this
    # eval-neutral rather than merely faster: every preferred file outranks every non-preferred one
    # on ``rank[0]``, so moving a preferred file earlier cannot displace a winner, and the ties that
    # ``>`` resolves by visit order are between files that were already in this relative order.
    for path in [f for f in tabular if f in preferred] + [f for f in tabular if f not in preferred]:
        frame = read_table(path)
        if frame is None:
            continue
        is_genes, gene_key, score_column = looks_like_gene_table(frame)
        if not is_genes or not score_column:
            continue
        rank = (
            1 if path in preferred else 0,
            0 if is_signal_free(frame[score_column]) else 1,
            min(int(len(frame)), 10**9),
        )
        if rank > best_rank:
            best, best_rank = (frame, gene_key, score_column, path), rank
        # The registry named this file; nothing later can outrank it, so stop reading tables.
        if rank[0] and rank[1]:
            break
    return best


def _rank(frame, genes, scores, score_column: str, *, ascending: bool):
    import pandas as pd

    ranked = pd.DataFrame({"gene": list(genes), score_column: pd.to_numeric(scores, errors="coerce")})
    extra = [c for c in numeric_frame(frame).columns if c != score_column][:4]
    for column in extra:
        ranked[str(column)] = frame[column].to_numpy()
    # ``kind="mergesort"`` -- stable -- because ties are common here and the pick among them is
    # published. somde's recorded table holds one distinct qval across all 500 genes, and numpy's
    # introsort only falls back to a stable insertion sort on short arrays: at 500 rows, 40 of 40
    # shuffles of that table move ``top_gene`` to a different gene. Once the finding says "first of
    # 500 tied", "first" has to mean the first in the tool's own file order and not wherever
    # quicksort's pivot left it.
    ordered = ranked.dropna(subset=[score_column]).sort_values(score_column, ascending=ascending, kind="mergesort")
    # Sorted first, so ``keep="first"`` keeps each symbol's best row under either direction -- the
    # smallest p-value when ascending, the largest statistic when not. Without this a repeated
    # symbol occupied several slots of a top-50 shortlist that then listed fewer than 50 genes, and
    # ``significant_fraction`` was a fraction of rows while ``n_genes`` counted genes.
    return ordered.drop_duplicates(subset=["gene"], keep="first")


def _tied_at_top(ranked, shortlist, score_column: str) -> int:
    """How many genes share the best score. ``1`` means the ranking has a genuine winner.

    Counted over ``ranked`` -- every gene -- and not over ``shortlist``. Both are sorted and share a
    first row, so the best score is the same either way, but the shortlist is ``ranked.head(50)``
    whenever nothing clears the threshold, and a count taken there is capped at 50 by construction.
    somde's tie is 500 and spvc's is 186; both would report "first of 50", a number that looks
    measured and is only the shortlist's own length.
    """
    if not len(shortlist):
        return 0
    return int((ranked[score_column] == shortlist.iloc[0][score_column]).sum())


def _tables(ctx: AnalysisContext, ranked, shortlist, score_column: str, *, note: str = "") -> None:
    with step(ctx.manifest, "SVG shortlist"):
        top = shortlist.head(_TOP_N)
        n = len(top)
        title = f"Top {n} genes by score ({note})" if note else f"Top {n} spatially variable genes"
        # The chart below already qualifies its own cut with :func:`_tie_at_cut`; the table it sits
        # beside was cut the same way and said nothing. Fifty genes are the top fifty only when the
        # fiftieth score is the fiftieth score -- spagft's fdr at the cut is held by ten genes and
        # three are listed, so seven exactly as good are off the list and nothing anywhere says so.
        # (somde and spvc cut a tie too, but a signal-free warning and a bar caption happen to cover
        # theirs.) The rows keep their order and their contents; only the claim over them changes.
        # Same sentence as ``_top_bar``'s caption, word for word, so the two are one statement.
        title += _cut_caveat(ranked, top, score_column, shows="are listed")
        ctx.table(top, "top_svgs.csv", title=title, index=False)
        if len(shortlist):
            # The value stays: it is a real member of the best-scoring set, and withholding it would
            # be the worse bug. What cannot stand is "Highest ranked gene" over a pick from 500
            # equals -- somde's whole table is one qval, spatialde's best is held by 100 genes, spvc's
            # by 186. R66's repair, applied to the strongest single claim an SVG run makes.
            tied = _tied_at_top(ranked, shortlist, score_column)
            label = "Highest ranked gene" if tied <= 1 else f"First of {tied} genes tied at the best score"
            ctx.find("top_gene", str(shortlist.iloc[0]["gene"]), label)
            if tied > 1:
                # A key and not only prose, so the portal, the report and the export can act on it.
                ctx.find("n_tied_at_top", tied, "Genes sharing the best score")
    with step(ctx.manifest, "full ranking"):
        ctx.table(ranked, "gene_ranking.csv", title="All genes, ranked", index=False)


def _distribution(ctx: AnalysisContext, scores, score_column: str, is_pvalue: bool, signal_free: bool) -> None:
    import numpy as np

    with step(ctx.manifest, "score distribution"):
        values = scores.to_numpy()
        # ``plots.histogram`` drops non-finite values before drawing (``plots.py:275``), so
        # ``len(scores)`` captioned genes the picture does not contain. ``_rank`` removes NaN but not
        # ``Inf``, and ``Inf`` is exactly what R's ``write.csv`` emits for an infinite score -- most
        # of this repo's SVG tools are R -- so the gap is reachable from an ordinary output. Count
        # what was plotted, on the same test the plotter uses.
        drawn = int(np.isfinite(np.asarray(values, dtype="float64")).sum())
        ctx.figure(
            plots.histogram(
                values,
                title=f"Distribution of {score_column}",
                xlabel=str(score_column),
                vline=_ALPHA if is_pvalue else None,
            ),
            "score_distribution.png",
            title=f"Distribution of {score_column}",
            caption=(
                "Every gene has the same score; the distribution is a single spike" if signal_free else f"{drawn} genes"
            ),
            kind="histogram",
        )


def _tie_at_cut(ranked, top, score_column: str) -> tuple[int, int]:
    """``(genes sharing the score the chart was cut at, how many of them the chart shows)``.

    ``(0, 0)`` when every gene holding that score is on the chart, which is the ordinary case and
    needs no qualifying. Otherwise the twenty bars are not the top twenty: spvc's best
    ``adjusted_pvalue`` is held by 186 genes, so a hundred and sixty-six exactly as good as the ones
    drawn are off the chart -- and all twenty bars are the same height, which is the picture
    contradicting the caption.

    Over ``ranked`` and not ``shortlist``, for the reason in :func:`_tied_at_top`.
    """
    cut = top.iloc[-1][score_column]
    size = int((ranked[score_column] == cut).sum())
    shown = int((top[score_column] == cut).sum())
    return (size, shown) if size > shown else (0, 0)


def _cut_caveat(ranked, top, score_column: str, *, shows: str) -> str:
    """The clause qualifying a cut, or ``""`` when the cut needs no qualifying.

    Two ways a cut misleads, and :func:`_tie_at_cut` reports only the first::

        genes off the list hold the score it was cut at ... which N of them are shown is arbitrary
        every row on the list holds one score ............ the order within the list is arbitrary

    The second is a case ``_tie_at_cut`` answers ``(0, 0)`` for, because nothing tied is missing --
    and it is the more visible one. svca's recorded run is ten significant genes all at ``pvalue``
    0.01: ten bars of the same length, captioned "Top 10 genes by pvalue". The chart is not a
    ranking of ten, it is a set of ten in the order the file happened to list them.

    Both are said in ``ranked`` and not ``shortlist`` terms, for the reason in :func:`_tied_at_top`.
    When both hold, the first is the stronger statement and is the one made: it already says the
    order is arbitrary, and it also says how many equally good genes the reader is not seeing.

    ``shows`` completes the sentence for the artefact asking -- the table lists rows, the chart
    draws bars -- so the two stay one statement in two voices.
    """
    if len(top) < 2:
        return ""
    tie_size, tie_shown = _tie_at_cut(ranked, top, score_column)
    if tie_size:
        return (
            f"; the score at the cut is shared by {tie_size} genes, so which {tie_shown} of them {shows} is arbitrary"
        )
    if int(top[score_column].nunique(dropna=False)) == 1:
        return f"; all {len(top)} have the same {score_column}, so their order is arbitrary"
    return ""


def _top_bar(ctx: AnalysisContext, ranked, shortlist, score_column: str, is_pvalue: bool, *, note: str = "") -> None:
    with step(ctx.manifest, "top gene ranking"):
        top = shortlist.head(20)
        if not len(top):
            return
        values = top[score_column].to_numpy()
        label = f"-log10({score_column})" if is_pvalue else str(score_column)
        heading = f"Best-scoring genes ({note})" if note else "Top spatially variable genes"
        if is_pvalue:
            import numpy as np

            values = -np.log10(np.clip(values, 1e-300, None))
        ctx.figure(
            plots.bar_chart(
                top["gene"].tolist(),
                values,
                title=heading,
                ylabel=label,
                horizontal=True,
            ),
            "top_svgs.png",
            title=heading,
            caption=(
                f"Top {len(top)} genes by {score_column}"
                + _cut_caveat(ranked, top, score_column, shows="these bars show")
            ),
            kind="bar",
        )


def _spatial(ctx: AnalysisContext, shortlist, gene_key, *, note: str = "") -> None:
    """Map the top genes only when an expression matrix is actually present next to the result."""
    from spatialomicsgym.postanalysis.sources import open_anndata

    heading = f"Best-scoring genes ({note})" if note else "Top spatially variable genes"
    genes = [str(g) for g in shortlist["gene"].head(_MAX_MAPPED_GENES)]
    if not genes:
        return
    for path in [f for f in ctx.files if f.suffix == ".h5ad" and not is_staged_input(f)]:
        with open_anndata(path) as adata:
            if adata is None:
                continue
            present = [g for g in genes if g in {str(v) for v in adata.var_names}]
            if not present:
                continue
            with step(ctx.manifest, "top gene maps"):
                import numpy as np
                import pandas as pd

                coordinates, _ = find_coordinates([path], index=[str(i) for i in adata.obs_names])
                if coordinates is None:
                    return
                matrix = adata[:, present].X
                values = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
                frame = pd.DataFrame(values, columns=present, index=[str(i) for i in adata.obs_names])
                notes: list[str] = []
                aligned, coordinates = align_to_coordinates(frame, coordinates, notes=notes)
                for note in notes:
                    ctx.warn(note)
                if aligned is None:
                    return
                ctx.analysed(path)
                ctx.figure(
                    plots.spatial_grid(
                        coordinates["x"],
                        coordinates["y"],
                        aligned,
                        title=heading,
                        value_label="Expression",
                    ),
                    "top_svg_maps.png",
                    title=f"{heading} in space",
                    caption=f"Expression of {len(present)} top-ranked genes",
                    kind="spatial_map",
                )
            return
