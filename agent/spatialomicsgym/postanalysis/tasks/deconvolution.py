"""Deconvolution: a spot x cell-type proportion matrix, and everything that can be wrong with one.

The analysis is the easy half. The hard half is deciding whether the matrix in front of you is the
result at all, and this repo has recorded five ways it is not:

* it is written transposed, with spot IDs sitting where cell-type names go;
* it is all zeros, or every spot has the same composition -- a result with no information in it;
* its columns are LDA topics (``1 2 3``) or archetypes, not annotated cell types;
* it is not a proportions matrix but a marker table that happens to have numeric columns;
* it is not about cell types at all -- a counts, adjacency or coupling matrix whose cell-type axis
  is labelled with spot barcodes.

Each is checked before anything is plotted, reported as a finding L2 can act on, and -- when the
values are meaningless -- used to *skip* the analyses that would otherwise dress them up as a
result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialomicsgym.postanalysis import plots
from spatialomicsgym.postanalysis.detect import match_authoritative
from spatialomicsgym.postanalysis.manifest import step
from spatialomicsgym.postanalysis.sources import (
    align_to_coordinates,
    find_coordinates,
    is_staged_input,
    open_anndata,
    read_anndata,
)
from spatialomicsgym.postanalysis.tables import (
    celltype_columns,
    columns_are_the_gene_panel,
    duplicate_header_fields,
    implausible_as_cell_types,
    is_signal_free,
    is_tabular,
    looks_like_barcodes,
    looks_like_marker_table,
    looks_like_proportions,
    numeric_frame,
    orient_proportions,
    read_table,
    spot_evidence,
    topic_like_columns,
)

if TYPE_CHECKING:
    from pathlib import Path

    from spatialomicsgym.postanalysis.context import AnalysisContext

#: obsm keys a tool publishes its abundance matrix under, in the registry's own preference order.
_OBSM_KEYWORDS = ("abundance", "proportion", "deconv", "cell_type_prob", "cell_type_counts", "q05")

#: Panels in the per-cell-type spatial grid.
_MAX_MAPPED_TYPES = 9

#: What to call the column axis, in the three inflections this module's labels, titles and captions
#: use, keyed by whether the guard in :func:`run` accepted the column names as cell-type annotations.
#:
#: Every string built from the ``True`` row is byte-identical to what this module published before
#: R79/O, which is the point: the wording is the only thing that moves, and only on the runs the
#: guard has already ruled against. Sentence-initial uses go through ``.capitalize()`` rather than a
#: fourth field -- it title-cases "cell-type" and "columns" correctly and leaves the rest alone.
_AXIS_WORDS = {
    True: ("cell type", "cell types", "cell-type"),
    False: ("column", "columns", "column"),
}


def _axis_words(ctx: AnalysisContext) -> tuple[str, str, str]:
    """``(singular, plural, adjective)`` for the column axis of the matrix being analysed.

    ``run()`` establishes whether the columns are cell types, records it as ``celltype_names_resolved``
    and -- deliberately, see :func:`~spatialomicsgym.postanalysis.tables.topic_like_columns` -- carries
    on publishing the numbers anyway, because a panel really named ``Layer1..Layer6`` should cost a
    warning and not a lost result. What it must not also do is narrate those numbers in cell-type
    words. Measured over the recorded corpus, 20 of the 40 runs that carry the finding have it
    ``False`` and every one of them published ``Most abundant cell type: arch_0`` (Starfysh),
    ``slice1_idx`` (SLAT), ``alpha_omics1`` (SpatialGlue) or a bare integer, three lines under the
    warning saying those are not cell types. Same defect as R66, and it takes R66's repair: correct
    the labels, keep the measurements.

    Read back from the finding rather than threaded down as an argument so the wording and the
    published fact are one value and cannot drift apart -- and because ``_composition``, ``_dominant``,
    ``_colocalization`` and ``_spatial`` would each have had to remember to pass it. Absent means
    nobody ran the guard, so the wording stays exactly as it was.

    Only the prose moves. Finding keys, artifact filenames and every number are identical on both
    branches; the scoring path, the portal and the HuggingFace export read those, not this.
    """
    named = True
    for finding in ctx.manifest.findings:
        if finding["key"] == "celltype_names_resolved":
            named = bool(finding["value"])
            break
    return _AXIS_WORDS[named]


def run(ctx: AnalysisContext) -> None:
    picked = _pick_proportions(ctx)
    if picked is None:
        ctx.manifest.fail(
            "No cell-type proportion matrix found in the tool output "
            f"({len(ctx.files)} files scanned, none was a spot x cell-type matrix)"
        )
        return

    frame, source = picked
    ctx.analysed(source)

    frame, transposed, notes = orient_proportions(frame)
    for note in notes:
        ctx.warn(note)
    ctx.find("orientation_transposed", bool(transposed), "Proportion table was transposed on read")

    # Orientation is settled, so the columns are the axis being *claimed* as cell types -- and a
    # spot barcode is never one. Checking here and not in `looks_like_proportions` is the whole
    # point: a real composition written cell_types x spots has barcodes across its columns on disk,
    # and rescuing it is `orient_proportions`'s job.
    #
    # This is not a corner case. Over the checked-in smoke fixtures, eight tools published a
    # barcode-labelled `mean_composition.csv` at status=ok, and none of the eight files was a
    # deconvolution result: a spot x spot adjacency matrix, two raw counts matrices, a staged
    # mixture-count input, an imputed expression matrix, a cell x spot coupling matrix and a
    # decontaminated counts matrix. `topic_like_columns` was the only label check and it looks for
    # `arch_0`/`1`/`topic_3`, so every one of them passed as "annotated cell types".
    if looks_like_barcodes(frame.columns):
        ctx.find("celltype_names_resolved", False, "Columns are annotated cell types")
        ctx.manifest.fail(
            f"Columns {[str(c) for c in frame.columns[:3]]} are spot barcodes, not cell types, so "
            f"{source.name} is a spot x spot or spot x feature matrix and not a composition. No "
            "cell-type result was published from it. Pass tool_name= or task_type= if the real "
            "proportions are in another file."
        )
        return

    # The same question as the barcode guard, asked of the other axis a table can be mislabelled
    # with. An imputation tool writes spots x *genes*, and stPlus's was published at status=ok as a
    # composition of 510 cell types whose dominant one was `WNT4.1`. Content cannot separate that
    # from a real composition -- see `columns_are_the_gene_panel` -- but the dataset sitting in the
    # same directory can, because those names are its own `var_names`.
    panel_source = _gene_panel_source(ctx, frame.columns)
    if panel_source is not None:
        ctx.find("celltype_names_resolved", False, "Columns are annotated cell types")
        ctx.manifest.fail(
            f"Columns {[str(c) for c in frame.columns[:3]]} are gene names -- they are the "
            f"var_names of {panel_source.name} in the same directory -- so {source.name} is an "
            "expression matrix and not a cell-type composition. No cell-type result was published "
            "from it. Pass tool_name= or task_type= if the real proportions are in another file."
        )
        return

    # A fourth question about the columns, and the only one whose wrong answer this layer wrote
    # itself. ``read_csv`` cannot hold two columns under one name, so it renames the second:
    # ``spot,Bcell,Tcell,Bcell`` arrives as ``Bcell``, ``Tcell``, ``Bcell.1``. Both halves are then
    # published as cell types and neither is that cell type's share -- Bcell at 0.75 of the tissue
    # is reported at 0.35 and 0.40, and the larger half can top ``mean_composition.csv`` and name
    # the dominant type in every tissue map, under a label no reference contained.
    #
    # Only the header line can be asked, and only when the columns are the axis the header wrote:
    # after a transpose the repeated fields are row labels, where a repeat costs nothing this
    # function publishes. Reach over the recorded corpus: 5 of 535 tables repeat a header field, 0
    # published labels carry a mangle suffix -- three of the five are headerless numeric matrices
    # ``_header_is_data`` already re-reads, and the other two hold gene names in tables the handler
    # does not pick. Nothing published today moves.
    repeated = [] if transposed else duplicate_header_fields(source)
    if repeated:
        shown = ", ".join(repeated[:3]) + (f" and {len(repeated) - 3} more" if len(repeated) > 3 else "")
        ctx.warn(
            f"{source.name} gives more than one column the same column name ({shown}), so each of "
            f"those cell types has its share split across two columns and the reader renamed the "
            f"second -- {repeated[0]}.1 is this reader's label, not the tool's. Every number below "
            f"is a part of a cell type's share rather than the whole of it. Deduplicate the "
            f"reference labels and re-run before reading the composition."
        )
        ctx.manifest.degrade()

    named = not topic_like_columns(frame.columns)
    ctx.find("celltype_names_resolved", named, "Columns are annotated cell types")
    if not named:
        ctx.warn(
            f"Columns {[str(c) for c in frame.columns[:4]]} are positional labels -- latent topics, "
            "archetypes, or a stem and a counter -- not cell-type annotations. Re-run the tool with "
            "a cell-type-annotated reference, or map these labels to cell types, before reading "
            "them as a composition."
        )
        ctx.manifest.degrade()

    _, many, _ = _AXIS_WORDS[named]
    ctx.find("n_spots", int(frame.shape[0]), "Spots deconvolved")
    ctx.find("n_cell_types", int(frame.shape[1]), f"{many.capitalize()} resolved")

    signal_free = _is_signal_free(frame)
    ctx.find("signal_free", signal_free, "Result carries no spatial information")
    if signal_free:
        ctx.warn(
            "Signal-free result: every spot has an identical (or all-zero) composition, so the "
            "proportions carry no information. Composition-dependent analyses were skipped."
        )
        ctx.manifest.degrade()

    row_sums = frame.sum(axis=1)
    # The gate used to be ``not signal_free``, which skipped the rescaling for *every* signal-free
    # result -- and a signal-free result is not necessarily a zero one. A tool that assigns the same
    # composition to every spot on an abundance scale (60 spots x 250 counts) reached `_composition`
    # unscaled, where the labels all say proportion: ``mean_proportion 250.0``, ylabel "Mean
    # proportion", and `spots_above_10pct` applying a ``> 0.1`` threshold to counts, which is true
    # of every spot and every cell type by construction.
    #
    # What the old gate was protecting is narrower than what it said: an all-zero matrix has no
    # scale to rescale *to*. The division is already a no-op there (`replace(0, nan)` then
    # `fillna(0.0)`), but the sentences below would be about a rescaling that did not happen, and
    # `_composition`'s own comment records that the recorded all-zero incident must keep reporting
    # exactly what it reports today. So the guard asks that question directly.
    has_mass = bool(len(row_sums)) and bool(float(row_sums.abs().max()) > 0)
    if has_mass and not _sums_to_one(row_sums):
        median = float(row_sums.median())
        if median > 1.0:
            ctx.warn(
                f"Rows do not sum to 1 (median {median:.3g}); these are abundances, not fractions. "
                "Figures show the row-normalised values."
            )
        else:
            # A median *below* 1 is not what an abundance matrix looks like, and saying "these are
            # abundances" of a median of 0 states the opposite of what the number shows. What it
            # does look like is a matrix in which many spots were never fitted: half the rows are
            # empty, so the median falls to zero while the rows that do carry mass sum to 1.
            empty = int((row_sums.fillna(0.0) <= 0).sum())
            if empty:
                ctx.warn(
                    f"Rows do not sum to 1 (median {median:.3g}). A median below 1 is not an abundance "
                    f"scale: {empty} of {len(row_sums)} spots carry no cell-type mass at all. Those spots "
                    "are empty, not evenly mixed, and every proportion below is an average over the spots "
                    "that were fitted. Figures show the row-normalised values."
                )
            else:
                # ``empty`` is the number the sentence above rests on, and it printed it: two
                # recorded runs (novosparc, spaotsc) read "0 of 300 spots carry no cell-type mass at
                # all. Those spots are empty, not evenly mixed." No spot is empty in either. Every
                # row carries mass on a scale of 1/300 -- a matrix normalised over the whole grid
                # rather than over the cell types -- which is a different fact and a different repair.
                ctx.warn(
                    f"Rows do not sum to 1 (median {median:.3g}) and no spot is empty: every row "
                    "carries mass, on a scale below 1. The matrix is normalised over some axis other "
                    "than the cell types. Figures show the row-normalised values."
                )
        frame = frame.div(row_sums.replace(0, float("nan")), axis=0).fillna(0.0)

    _composition(ctx, frame)
    if not signal_free:
        _dominant(ctx, frame)
        _colocalization(ctx, frame)
    _spatial(ctx, frame, signal_free)


def _gene_panel_source(ctx: AnalysisContext, columns) -> Path | None:
    """The h5ad beside the result whose gene vocabulary ``columns`` *are*, or ``None``.

    Read backed -- ``var`` only, never ``X`` -- for the reason ``_coordinates_from_h5ad`` gives:
    a full read pulls the whole expression matrix through RAM to hand back a list of names.

    Staged inputs are deliberately *included* here, unlike everywhere else in this file. The dataset
    the tool was handed is the most direct statement of what this experiment's genes are, and it is
    the file most likely to still be sitting in the directory.

    Half the recorded result directories carry no h5ad at all. With nothing to compare against, the
    guard has nothing to say -- and saying nothing is the honest outcome, not a refusal.
    """
    for path in ctx.files:
        if path.suffix.lower() != ".h5ad":
            continue
        with open_anndata(path, backed=True) as adata:
            try:
                names = [] if adata is None else [str(v) for v in adata.var_names]
            except Exception:
                names = []
        if names and columns_are_the_gene_panel(columns, names):
            return path
    return None


# ---------------------------------------------------------------------------------------
# picking the matrix
# ---------------------------------------------------------------------------------------


def _pick_proportions(ctx: AnalysisContext):
    """The tool's own authoritative output first, then any table that reads as proportions."""
    candidates = [f for f in ctx.files if not is_staged_input(f)]
    tabular = [f for f in candidates if is_tabular(f)]
    preferred = [f for f in match_authoritative(candidates, ctx.tool_name) if f in tabular]

    rejected_marker_tables: list[str] = []
    normalised = None
    unnormalised = None
    unevidenced = None
    for path in preferred + [f for f in tabular if f not in preferred]:
        frame = read_table(path)
        if frame is None:
            continue
        if looks_like_marker_table(frame):
            rejected_marker_tables.append(path.name)
            continue
        if not looks_like_proportions(frame):
            continue
        # The matrix returned has to be the matrix that was just accepted. ``looks_like_proportions``
        # drops the spot coordinates before deciding; returning the unfiltered ``numeric_frame``
        # handed the analyses a table in which ``x`` and ``y`` were cell types -- ``y`` topped
        # ``mean_composition.csv`` at 50 %, and the coordinate magnitudes pushed the row sums to a
        # median of 1.08e+04, which produced the warning "these are abundances, not fractions" about
        # a table whose five real columns sum to exactly 1.
        accepted = celltype_columns(numeric_frame(frame)), path
        # "First acceptable table" is directory order, and a topic model writes two acceptable
        # tables: the spot x topic composition and the topic x gene expression profile. Whichever
        # sorted first won. When that was the profile the run reported its *genes* as the spots --
        # ``orient_proportions`` has no row sum to key on for a raw profile, so its shape tie-break
        # transposed the wider-than-tall matrix and 35 gene symbols became 35 spots.
        #
        # An axis that sums to 1 is the only positive evidence that a non-negative numeric table is
        # a composition, and it is the same evidence ``orient_proportions`` decides orientation by.
        # A file the registry names stays first past the post; the row sums only break the tie
        # between guesses, and when nothing sums to 1 the first acceptable table still wins.
        #
        # Summing to 1 is not enough on its own, because a topic model's *other* output sums to 1
        # too: ``beta`` is a distribution over genes, one row per topic. Both files pass and the tie
        # fell back to filename order a second time -- a (3 x 500) gene-named beta beat the (200 x 3)
        # composition beside it and the run published ``n_cell_types: 500``, ``top_cell_type:
        # 'IGKC'``. The axis that sums to 1 is the cell-type axis, so the tie goes to the candidate
        # whose normalised axis holds a *believable number of cell types* -- true whichever way
        # round the file was written, and 500 genes is not that number.
        if path in preferred:
            return accepted
        # Nothing names this table's rows and there are too few of them to take for spots on the
        # count (``tables.spot_evidence``) -- the content route declines exactly these, so a run it
        # typed from a table that has evidence must not be analysed from one that sorts earlier and
        # has none: a cluster x timepoint split written with its clusters as the unnamed index sums
        # to 1 per row over a believable number of columns and would win the tie-break below. Kept
        # as the last resort rather than refused, because a caller that names the task has said
        # what the table is, and the table does not say otherwise.
        if spot_evidence(frame) is None:
            if unevidenced is None:
                unevidenced = accepted
            continue
        if _is_normalised(accepted[0]):
            if _normalised_over_a_plausible_celltype_axis(accepted[0]):
                return accepted
            if normalised is None:
                normalised = accepted
        elif unnormalised is None:
            unnormalised = accepted
    if normalised is not None:
        return normalised
    if unnormalised is not None:
        return unnormalised

    for name in rejected_marker_tables:
        ctx.warn(f"{name} is a per-gene marker table, not a proportions matrix; it was not scored as one")

    for path in [f for f in candidates if f.suffix == ".h5ad"]:
        result = _from_h5ad(path)
        if result is not None:
            return result
    return unevidenced


def _is_normalised(numeric) -> bool:
    """True when one axis of the matrix already sums to 1, in either orientation.

    Same test, same tolerance, as ``orient_proportions``: whichever axis sums to ~1 is the
    cell-type axis, because that is what a proportion vector means.
    """
    import numpy as np

    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        return False
    for axis in (1, 0):
        sums = np.asarray(numeric.sum(axis=axis), dtype="float64")
        sums = sums[np.isfinite(sums)]
        if sums.size and bool(np.isclose(np.median(sums), 1.0, atol=0.02)):
            return True
    return False


def _normalised_over_a_plausible_celltype_axis(numeric) -> bool:
    """True when an axis sums to 1 and that axis could be the cell types -- i.e. a composition.

    A composition's proportion vector runs over cell types, so the normalised axis is the one this
    would name as cell types, and ``tables.implausible_as_cell_types`` is what says whether that
    naming is believable. A topic model's ``beta`` is normalised over its genes, of which there are
    thousands, and fails it. This is a tie-break, not a rejection: an unusual-but-possible count
    passes.
    """
    import numpy as np

    n_rows, n_cols = numeric.shape
    # ``axis=1`` sums each row *across the columns*, so it is the columns that are being summed
    # over and the columns that would be the cell types; ``axis=0`` is the mirror of that.
    for axis, summed_over, the_other, labels in (
        (1, n_cols, n_rows, numeric.columns),
        (0, n_rows, n_cols, numeric.index),
    ):
        sums = np.asarray(numeric.sum(axis=axis), dtype="float64")
        sums = sums[np.isfinite(sums)]
        if not sums.size or not bool(np.isclose(np.median(sums), 1.0, atol=0.02)):
            continue
        if not implausible_as_cell_types(summed_over, the_other, labels):
            return True
    return False


def _from_h5ad(path: Path):
    import pandas as pd

    adata = read_anndata(path)
    if adata is None:
        return None
    for key in adata.obsm:
        if not any(word in str(key).lower() for word in _OBSM_KEYWORDS):
            continue
        matrix = adata.obsm[key]
        if hasattr(matrix, "columns"):
            frame = pd.DataFrame(matrix)
        else:
            try:
                values = matrix.toarray() if hasattr(matrix, "toarray") else matrix
                frame = pd.DataFrame(values, columns=[f"celltype_{i}" for i in range(values.shape[1])])
            except Exception:
                continue
        frame.index = [str(i) for i in adata.obs_names]
        if frame.shape[1] >= 2:
            return frame, path
    return None


# ---------------------------------------------------------------------------------------
# the analyses
# ---------------------------------------------------------------------------------------


def _composition(ctx: AnalysisContext, frame) -> None:
    one, many, adj = _axis_words(ctx)
    # A spot the tool never fitted is a row of zeros, and averaging it in dilutes every proportion
    # by the fraction of spots that were never fitted -- 40 unfitted spots out of 100 published
    # `Astrocytes` at 0.300 where the fitted tissue is 0.500, and the column summed to 0.600 instead
    # of 1. The reader takes 0.300 as "30 % of the tissue". Same rule `_dominant` already applies,
    # and it is what `run()` already tells the reader it did ("every proportion below is an average
    # over the spots that were fitted").
    #
    # When *nothing* was fitted there is no subset to average over, and the recorded all-zero
    # incident must keep reporting exactly what it reports today, so the raw frame stands and the
    # signal-free warning carries it.
    usable, unfitted = _fitted_spots(frame)
    if not len(usable):
        usable, unfitted = frame, 0
    if unfitted:
        ctx.warn(
            f"{unfitted} of {len(frame)} spots carry no cell-type mass and were not averaged; "
            f"the composition below is over the {len(usable)} spots that have one."
        )
    with step(ctx.manifest, "mean composition"):
        # One denominator for every column. `mean` defaults to `skipna=True`, which gives each cell
        # type its *own* denominator -- the spots where it is not blank -- and a share divided by a
        # different number per column is not a share. Measured on 100 spots where `Rare` is written
        # in 5 of them at 0.50 and left blank in the rest, with every row summing to exactly 1: the
        # column read `Rare 0.5000, Astro 0.4915, Neuron 0.4835`, totalling 1.475, and
        # `top_cell_type` came back "Rare" -- the most abundant cell type in the sample, at twenty
        # times its real 2.5 %, beside its own `spots_above_10pct` of 5. The rows sum to 1 under the
        # same `skipna`, so `run()`'s rescaling branch -- whose `.fillna(0.0)` would have caught
        # this -- never fires.
        #
        # Local to the composition on purpose. `_colocalization` reads a blank as *not measured* and
        # keeps it out of `corr`, because imputing zero into a correlation manufactures the signal it
        # then reports. That is right for a correlation and wrong for a share.
        blank = int(usable.isna().to_numpy().sum())
        if blank:
            spots = int(usable.isna().any(axis=1).sum())
            ctx.warn(
                f"{blank} blank cell(s) across {spots} of {len(usable)} spots were read as zero mass "
                "for the composition below, so every cell type is averaged over the same spots. A "
                "blank in a proportion matrix is a cell type the tool assigned nothing to."
            )
        # ``kind="mergesort"``: with the label below able to read "first of five", "first" has to
        # name something, and a tie is the one case in which the sort's stability is observable.
        # Same reason `svg._rank` says it.
        mean = usable.fillna(0.0).mean(axis=0).sort_values(ascending=False, kind="mergesort")
        table = mean.rename("mean_proportion").to_frame()
        table["max_proportion"] = frame.max(axis=0)
        table["spots_above_10pct"] = (frame > 0.1).sum(axis=0)
        ctx.table(table, "mean_composition.csv", title=f"Mean {adj} composition")
        # "Most abundant" is a claim that one column beat the others, and nothing here asked whether
        # it did. stdgcn's proportion matrix is entirely zeros -- `signal_free` is `True` three
        # findings above and the warning says the composition carries no information -- so all five
        # of its cell types hold the top mean, and it published ``Most abundant cell type: CD14+
        # Monocytes``, which is the order pandas left them in. It is the only run of the 29 that
        # publish a composition where the top mean is shared; the other 28 have a strict winner and
        # read exactly as they did.
        #
        # Exact equality, no tolerance. novosparc's top two means differ by a relative 2.96e-11
        # (0.0050000000412593242 against 0.0050000000411113471) across 200 cell types, which is some
        # 670x the worst-case rounding of a 200-term float64 mean -- a real ordering, however
        # meaningless, and calling it a tie would be the false statement. Every other run's gap is
        # 1.67e-3 or wider, so nothing sits between the two readings.
        #
        # R66's repair, as `svg._tables` already applies it to `top_gene`: keep the pick, correct the
        # claim. The column named is a real member of the top-mean set and withholding it would be
        # the worse bug; what cannot stand is the word *the*.
        #
        # ``top_cell_type`` stays the first of the two: findings are consumed in order -- L2 quotes
        # three of them as its reasons and the follow-on prompt names four -- and a count of the tie
        # ahead of the thing tied would spend the run's headline slot on its own footnote.
        tied = int((mean == mean.iloc[0]).sum())
        label = f"Most abundant {one}" if tied <= 1 else f"First of {tied} {many} tied at the highest mean proportion"
        ctx.find("top_cell_type", str(mean.index[0]), label)
        if tied > 1:
            # A key and not only prose, so the portal, the report and the export can act on it.
            ctx.find("n_tied_at_top", tied, f"{many.capitalize()} sharing the highest mean proportion")
        ctx.figure(
            plots.bar_chart(
                mean.index,
                mean.to_numpy(),
                title=f"Average {adj} composition",
                ylabel="Mean proportion",
            ),
            "average_composition.png",
            title=f"Average {adj} composition",
            caption=f"Mean proportion across {usable.shape[0]} spots, {usable.shape[1]} {many}",
            kind="bar",
        )
    with step(ctx.manifest, "proportion spread"):
        spread_caption = f"Distribution of each {one}'s proportion over all spots"
        if unfitted:
            spread_caption = (
                f"Distribution of each {one}'s proportion over the {len(usable)} spots that carry a composition"
            )
        ctx.figure(
            plots.box_plot(usable, title="Per-spot proportion spread", ylabel="Proportion"),
            "proportion_spread.png",
            title="Per-spot proportion spread",
            caption=spread_caption,
            kind="boxplot",
        )


def _fitted_spots(frame):
    """``(rows that carry cell-type mass, how many do not)``.

    A spot with no numeric value at all has no dominant type, and asking `idxmax` for one raises a
    FutureWarning today and a ValueError in a later pandas. It is also a fact worth reporting
    rather than absorbing: a CSV with blank cells means those spots were not fitted.

    Blank is not the only way a spot arrives empty, and `dropna(how="all")` only caught that one.
    A tool that writes **zeros** for the spots it did not fit produces a row of `[0, 0, 0, 0]`,
    which `dropna` keeps and `idxmax` answers with the *first* column -- so every unfitted spot was
    credited to whichever cell type happened to sort first. Nor did the blank case survive `run()`:
    once more than half the spots are empty the median row sum falls below 1, the matrix is
    row-normalised, and `.fillna(0.0)` rewrites the blanks as zeros before `_dominant` sees them.
    Measured on 100 spots with 20 zeroed, `Astrocytes` went from 13 spots (4th of 4) to 33 (1st)
    and the order of `dominant_celltype.csv` inverted.
    """
    import numpy as np

    values = frame.to_numpy(dtype="float64", na_value=np.nan)
    mass = np.nansum(np.abs(values), axis=1)
    fitted = np.isfinite(mass) & (mass > 0)
    return frame[fitted], int((~fitted).sum())


def _strictly_dominant(frame):
    """The spots one column wins outright, and the count of those where the top is a tie.

    ``idxmax`` answers "which column holds the maximum" by returning the *first* one that does, so
    on a tied spot it reports a position in the file rather than a fact about the tissue. Reversing
    the columns of the same numbers moves the answer: on the recorded corpus celloscope's modal
    component flips from ``0`` (193 spots) to ``9`` (197), and slat -- where all 199 spots are
    two-way ties between two columns its own manifest already warns are constant -- publishes
    whichever of them is written first at ``fraction 1.0``.

    A tied spot is not dominated by anything, so it is excluded and counted, exactly as the
    unfitted spots above it are, and as :func:`~spatialomicsgym.postanalysis.tasks.svg._tied_at_top`
    already does for genes sharing a best score.

    Returns the winning column per strictly-dominated spot -- indexed by that spot, so the map can
    plot exactly the spots the table counts -- and how many spots were tied.
    """
    import numpy as np
    import pandas as pd

    values = frame.to_numpy(dtype="float64", na_value=np.nan)
    best = np.nanmax(values, axis=1)
    # NaN never equals itself, so a blank never counts towards the width of the tie -- which is
    # right: the column has no proportion at this spot, not a proportion equal to the best one.
    width = (values == best[:, None]).sum(axis=1)
    strict = width == 1
    winners = frame.columns.to_numpy()[np.nanargmax(values, axis=1)]
    return pd.Series(winners, index=frame.index)[strict], int((~strict).sum())


def _dominant(ctx: AnalysisContext, frame) -> None:
    with step(ctx.manifest, "dominant cell type"):
        one, many, adj = _axis_words(ctx)
        usable, empty = _fitted_spots(frame)
        if empty:
            ctx.warn(
                f"{empty} of {len(frame)} spots have no proportion for any cell "
                "type; they have no dominant type and were left out of the counts below."
            )
        if not len(usable):
            ctx.warn("dominant cell type skipped: no spot carries a proportion")
            return
        winners, tied = _strictly_dominant(usable)
        if tied:
            ctx.warn(
                f"{tied} of {len(usable)} spots are tied at the top, with two or more {many} sharing "
                f"the highest proportion; they have no dominant {one} and were left out of the counts below."
            )
        if not len(winners):
            ctx.warn(f"dominant {one} skipped: every spot is tied at the top, so nothing dominates")
            return
        counts = winners.value_counts()
        table = counts.rename("n_spots").to_frame()
        table["fraction"] = table["n_spots"] / max(len(winners), 1)
        ctx.table(table, "dominant_celltype.csv", title=f"Spots by dominant {one}")
        ctx.find("n_dominant_types", int(len(counts)), f"{many.capitalize()} dominating at least one spot")


def _colocalization(ctx: AnalysisContext, frame) -> None:
    with step(ctx.manifest, "co-localization"):
        if frame.shape[1] < 2:
            ctx.warn("co-localization skipped: only 1 cell type present")
            return
        # Same denominator as `_composition` and `_dominant`, for the same reason -- and here it
        # decides the *sign*, not just the magnitude. An unfitted spot is a row of zeros, which
        # sits at the origin, below the mean of both variables, so it adds positive covariance to
        # every pair: the unfitted spots do not dilute a correlation, they manufacture one. With
        # 40 of 100 spots unfitted, all six pairs of a four-type result published as co-localizing
        # (+0.34 to +0.58) when over the fitted tissue they exclude one another (-0.19 to -0.60).
        # Every pair had the wrong sign, under a caption that named no denominator at all.
        usable, unfitted = _fitted_spots(frame)
        if not len(usable):
            ctx.warn("co-localization skipped: no spot carries a proportion")
            return
        # A reference with more cell types than the tissue contains is the ordinary case, and the
        # absent type comes back as a column of zeros. `corr` cannot correlate a constant and
        # returns NaN for its whole row and column; the blanket `.fillna(0.0)` published those as
        # measured zeros -- "Dead does not co-localize with Neurons" where nothing was measured --
        # and put 0.0 on the diagonal, which is not a value a Pearson matrix can take.
        # ``dropna=False`` counted a blank as one of the distinct values, so a cell type the tool
        # measured on part of the tissue and gave the *same* proportion everywhere it did measure
        # scored 2 and passed this filter as varying. It has no correlation to report either, so it
        # walked into precisely the case described above: `corr` returned NaN for its whole row and
        # column, the guard below announced "too few measured spots" -- the wrong reason, for a pair
        # sharing 60 of 100 spots -- and `fillna(0.0)` published 0.0 against every other type and
        # 0.0 on its own diagonal. Counting only the measured values drops it here with the other
        # constants, which is also what makes the guard's "those columns are gone" true.
        varying = usable.loc[:, usable.nunique(dropna=True) > 1]
        dropped = [c for c in usable.columns if c not in set(varying.columns)]
        constant = [str(c) for c in dropped]
        if constant:
            shown = ", ".join(constant[:4]) + (f" and {len(constant) - 4} more" if len(constant) > 4 else "")
            # "across every spot" is a claim about the spots it has no value on. Only make it when
            # there are none: no recorded run has a blank in its proportion matrix, so slat -- the
            # one that reaches this sentence, constant on all 199 spots -- keeps it word for word.
            where = "wherever they were measured" if usable[dropped].isna().to_numpy().any() else "across every spot"
            ctx.warn(
                f"{len(constant)} cell type(s) are constant {where} ({shown}) and are not in "
                "the co-localization matrix: a constant has no correlation to report. They are still "
                "counted in the composition above."
            )
        if varying.shape[1] < 2:
            ctx.warn("co-localization skipped: fewer than 2 cell types vary across the spots")
            return
        correlation = varying.corr(method="pearson")
        if bool(correlation.isna().to_numpy().any()):
            # Not the constant case -- those columns are gone. This is a pair with too few spots
            # where both are measured. Shown as 0.0 as before, but no longer silently.
            ctx.warn(
                "Some cell-type pairs share too few measured spots to correlate; those cells of "
                "colocalization.csv are shown as 0.0 and are not measurements."
            )
            correlation = correlation.fillna(0.0)
        _, many, adj = _axis_words(ctx)
        caption = f"Pearson correlation of per-spot proportions between {many}"
        if unfitted:
            caption = (
                f"Pearson correlation of per-spot proportions between {many}, over the "
                f"{len(usable)} spots that carry a composition"
            )
        ctx.table(correlation, "colocalization.csv", title=f"{adj.capitalize()} co-localization (Pearson)")
        ctx.figure(
            plots.heatmap(correlation, title=f"{adj.capitalize()} co-localization"),
            "colocalization_heatmap.png",
            title=f"{adj.capitalize()} co-localization",
            caption=caption,
            kind="heatmap",
        )


def _spatial(ctx: AnalysisContext, frame, signal_free: bool) -> None:
    one, many, adj = _axis_words(ctx)
    notes: list[str] = []
    coordinates, source = find_coordinates(ctx.files, index=frame.index, notes=notes)
    for note in notes:
        ctx.warn(note)
    if coordinates is None:
        # An input that was never there is not an analysis that failed. Warn, do not degrade:
        # `partial` is L2's signal that something went wrong, and a coordinate-free CSV -- the
        # normal shape for half the deconvolution tools -- must not spend it.
        if not notes:
            # Only when the directory really holds no coordinates. `find_coordinates` fills `notes`
            # when it found one and rejected it, and telling that reader to "keep the spatial h5ad
            # in the output directory" is advice about a file already sitting next to the result.
            ctx.warn(
                "No spatial coordinates found next to the result, so the tissue maps were skipped. "
                "Keep the spatial h5ad (or a coordinates CSV) in the output directory to get them."
            )
        return
    ctx.analysed(source)
    notes = []
    aligned, coordinates = align_to_coordinates(frame, coordinates, notes=notes)
    for note in notes:
        ctx.warn(note)
    if aligned is None:
        # No sentence here: `align_to_coordinates` is the only place that knows which of its two
        # refusals fired, and it puts that reason in `notes`, which the loop above has already
        # turned into warnings. The line that used to sit here said "share no spot IDs" for both.
        ctx.manifest.degrade()
        return

    if not signal_free:
        with step(ctx.manifest, "dominant cell type map"):
            # Same rule as the table: a spot carrying no mass is not dominated by the first column,
            # so it is left off the map rather than painted with a cell type it does not contain.
            # A spot whose top proportion is shared by two columns is not dominated by the first of
            # them either, and is left off for the same reason -- see `_strictly_dominant`.
            fitted, empty = _fitted_spots(aligned)
            winners, tied = _strictly_dominant(fitted)
            if not len(fitted):
                ctx.warn("dominant cell type map skipped: no spot carries a proportion")
            elif not len(winners):
                ctx.warn(f"dominant {one} map skipped: every spot is tied at the top, so nothing dominates")
            else:
                caption = f"{len(winners)} spots, coloured by the {one} with the highest proportion"
                if int(winners.nunique()) == 1:
                    # The map is one colour. The caption counted the spots and said nothing about
                    # the picture, so scdot's and spatialscope's read as a tissue with structure in
                    # it. ``n_dominant_types`` carries the same fact three findings away; the figure
                    # travels into the report and the portal with this line and nothing else.
                    caption += (
                        f"; the same {one} ({winners.iloc[0]}) is on top at every spot shown, "
                        "so the map is a single colour"
                    )
                if empty:
                    caption += f"; {empty} spots with no {adj} mass are not shown"
                if tied:
                    caption += f"; {tied} spots tied at the top are not shown"
                ctx.figure(
                    plots.spatial_scatter(
                        coordinates.loc[winners.index, "x"],
                        coordinates.loc[winners.index, "y"],
                        winners.to_numpy(),
                        title=f"Dominant {one} per spot",
                        categorical=True,
                    ),
                    "dominant_celltype_spatial.png",
                    title=f"Dominant {one} per spot",
                    caption=caption,
                    kind="spatial_map",
                )
    with step(ctx.manifest, "per-cell-type maps"):
        # The caption says "by mean proportion", which is the quantity `_composition` publishes in
        # `mean_composition.csv`, so it has to be computed the same way: a blank read as zero mass,
        # one denominator for every column. `mean` defaults to `skipna=True` -- the per-column
        # denominator that function rejects in as many words ("every cell type is averaged over the
        # same spots"). The two rankings then disagree and the reader is shown a grid of the "top"
        # cell types that the report's own table says are not the top cell types. Measured on 100
        # spots x 12 types where `Rare` is fitted on 10 of them at 0.12: the grid put `Rare` first
        # and dropped a type present on every spot, while the table ranks `Rare` last of twelve at
        # 0.012.
        #
        # Unfitted spots -- whole rows of zeros, which `_composition` drops and this frame keeps --
        # are the other difference between the two, and they cannot reorder anything: they scale
        # every column's mean by the same factor.
        order = aligned.fillna(0.0).mean(axis=0).sort_values(ascending=False).index[:_MAX_MAPPED_TYPES]
        ctx.figure(
            plots.spatial_grid(
                coordinates["x"],
                coordinates["y"],
                aligned[order],
                title=f"{adj.capitalize()} proportions in space",
                value_label="Proportion",
            ),
            "celltype_spatial_maps.png",
            title=f"{adj.capitalize()} proportions in space",
            caption=f"Top {len(order)} {many} by mean proportion",
            kind="spatial_map",
        )


def _is_signal_free(frame) -> bool:
    """No information: every value identical, or every spot's composition identical."""
    if is_signal_free(frame.to_numpy()):
        return True
    spread = frame.std(axis=0, ddof=0)
    return bool((spread.fillna(0.0) <= 1e-12).all())


def _sums_to_one(row_sums) -> bool:
    import numpy as np

    values = np.asarray(row_sums, dtype="float64")
    values = values[np.isfinite(values)]
    return bool(values.size) and bool(np.isclose(np.median(values), 1.0, atol=0.05))
