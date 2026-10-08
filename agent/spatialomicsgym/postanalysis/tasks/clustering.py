"""Spatial clustering: which domain each spot was assigned to, and whether that assignment says anything.

Two things here are less obvious than they look.

**Picking the label column.** A clustered h5ad usually carries more than one categorical obs column
-- the tool's own domain labels *and* whatever annotation came in with the data. Reporting
``cell_type`` as "the domains" is silently wrong, so the registry's ``prediction_key`` for the tool
is asked first, then the canonical priority list benchmarking scores by
(:func:`_canonical_keys`), and only then a fixed preference order of generic substrings
(domain > cluster > leiden > ... > cell_type).

**Integer categories.** ``sc.tl.rank_genes_groups`` does ``','.join(categories)`` internally, so an
``int64`` label column raises ``TypeError: sequence item 0: expected str instance, int found``.
Tools write integer labels constantly. The labels are cast to ``str`` before scanpy sees them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from spatialomicsgym.postanalysis import plots
from spatialomicsgym.postanalysis.detect import match_authoritative
from spatialomicsgym.postanalysis.manifest import step
from spatialomicsgym.postanalysis.sources import (
    align_to_coordinates,
    conversion_record,
    find_coordinates,
    is_staged_input,
    read_anndata,
)
from spatialomicsgym.postanalysis.tables import is_tabular, read_table

if TYPE_CHECKING:
    from spatialomicsgym.postanalysis.context import AnalysisContext

#: Preference order for the label column when the registry does not name one. First match wins.
_KEY_PREFERENCE: tuple[str, ...] = (
    "spatial_domain",
    "domain",
    "spatial_cluster",
    "cluster",
    "leiden",
    "louvain",
    "mclust",
    "kmeans",
    "region",
    "label",
    "annotation",
    "celltype",
    "cell_type",
)

#: The tail of :data:`_KEY_PREFERENCE`. That list runs from names only a clustering tool writes to
#: names an annotation carries just as readily, and these are the ones on the wrong end of it: a
#: ``cell_type`` column is what :func:`_say_how_the_key_was_chosen` exists to be suspicious of.
_ANNOTATION_NAMES: tuple[str, ...] = ("region", "label", "annotation", "celltype", "cell_type")

#: More distinct values than this and the column is a measurement, not a domain assignment.
_MAX_DOMAINS = 60

#: Markers reported per domain.
_MARKERS_PER_DOMAIN = 5

#: Adjusted-p threshold below which a ranked gene may be called a marker of its domain.
_MARKER_FDR = 0.05

#: Widest the marker heatmap gets. Past this the columns are unreadable, so the picture is capped
#: and :func:`_marker_heatmap` says in its caption how many of the table's genes it left off.
_MAX_HEATMAP_GENES = 30

#: How many of the runners-up a warning names before it stops listing them.
_NAMES_IN_WARNING = 8


class _Labels(NamedTuple):
    """What :func:`_pick_labels` found -- and how much of it is evidence rather than inference.

    ``evidence`` is one of ``"registry"`` (the column the tool declares), ``"name"`` (it matched
    :func:`_canonical_keys` or :data:`_KEY_PREFERENCE`) or ``"shape"`` (nothing but its values
    qualified it). ``usable`` is
    every column that could have been picked and ``declared`` is what the registry asked for, so
    :func:`_say_how_the_key_was_chosen` can put the alternatives in front of the reader.
    """

    labels: object
    key: str
    source: Path
    adata: object | None
    unassigned: int
    evidence: str
    usable: list[str]
    declared: list[str]


def run(ctx: AnalysisContext) -> None:
    picked = _pick_labels(ctx)
    if picked is None:
        ctx.manifest.fail(
            "No domain assignment found in the tool output "
            f"({len(ctx.files)} files scanned, none carried a per-spot cluster label)"
        )
        return

    labels, key, source, adata, unassigned = picked[:5]
    ctx.analysed(source)
    ctx.find("cluster_key", str(key), "Column holding the domain assignment")
    _say_how_the_key_was_chosen(ctx, picked)
    _say_if_the_key_came_with_the_input(ctx, picked)

    if not len(labels):
        ctx.manifest.fail(
            f"{key!r} is the domain assignment but every one of its {unassigned} spots is empty; "
            "the tool assigned nothing"
        )
        return
    if unassigned:
        ctx.find("n_unassigned_spots", int(unassigned), "Spots the tool left unassigned")
        ctx.warn(
            f"{unassigned} of {unassigned + len(labels)} spots were left unassigned by the tool "
            f"({key!r} is empty for them). An unassigned spot is not a domain, so it is excluded "
            "from the domain counts, the map and the marker test."
        )

    sizes = labels.value_counts()
    n_domains = int(len(sizes))
    ctx.find("n_spots", int(len(labels)), "Spots assigned")
    ctx.find("n_domains", n_domains, "Spatial domains identified")
    ctx.find("largest_domain_frac", round(float(sizes.iloc[0]) / max(len(labels), 1), 4), "Largest domain share")

    signal_free = n_domains <= 1
    ctx.find("signal_free", signal_free, "Result carries no spatial information")
    if signal_free:
        ctx.warn(
            f"Signal-free result: every spot was assigned to a single domain ({sizes.index[0]!r}), so "
            "the clustering carries no information. Marker ranking was skipped."
        )
        ctx.manifest.degrade()
    elif n_domains > _MAX_DOMAINS:
        ctx.warn(f"{n_domains} domains is more than a spatial clustering usually means; check the label column")

    _domain_sizes(ctx, sizes, len(labels))
    _spatial(ctx, labels)
    if adata is not None and not signal_free:
        _markers(ctx, adata, labels, key)
    elif adata is None:
        # A missing input warns; only a failed analysis degrades. See the same note in
        # deconvolution._spatial -- `partial` has to keep meaning "something went wrong".
        ctx.warn(
            "Marker ranking needs the expression matrix; only a label table was found, so no "
            "per-domain markers were computed. Keep the clustered h5ad in the output directory."
        )


# ---------------------------------------------------------------------------------------
# picking the labels
# ---------------------------------------------------------------------------------------


def _pick_labels(ctx: AnalysisContext) -> _Labels | None:
    """h5ad first: only it can also give markers. See :class:`_Labels` for the fields."""
    candidates = [f for f in ctx.files if not is_staged_input(f)]
    preferred = match_authoritative(candidates, ctx.tool_name)
    ordered = preferred + [f for f in candidates if f not in preferred]
    declared = _registry_keys(ctx.tool_name)

    for path in [f for f in ordered if f.suffix == ".h5ad"]:
        adata = read_anndata(path)
        if adata is None:
            continue
        columns = [str(c) for c in adata.obs.columns]
        key = _choose_key(columns, ctx.tool_name, adata.obs)
        if key is not None:
            labels, unassigned = _label_series(adata.obs[key], index=[str(i) for i in adata.obs_names])
            usable = [c for c in columns if _is_label_column(adata.obs[c])]
            return _Labels(labels, key, path, adata, unassigned, _key_evidence(key, declared), usable, declared)

    for path in [f for f in ordered if is_tabular(f)]:
        frame = read_table(path)
        if frame is None:
            continue
        columns = [str(c) for c in frame.columns]
        key = _choose_key(columns, ctx.tool_name, frame)
        if key is not None:
            labels, unassigned = _label_series(frame[key])
            usable = [c for c in columns if _is_label_column(frame[c])]
            return _Labels(labels, key, path, None, unassigned, _key_evidence(key, declared), usable, declared)
    return None


def _key_evidence(column: str, declared: list[str]) -> str:
    """``"registry"``, ``"name"`` or ``"shape"`` -- what stands behind the pick. See :class:`_Labels`.

    A canonical name counts as a name match. Half of :func:`_canonical_keys` is invisible to
    :data:`_KEY_PREFERENCE` -- ``RNA_snn_res.0.8``, ``spicemix_factor``, ``pred`` -- so a column
    chosen *because* benchmarking names it would otherwise be reported as a shrug, and
    :func:`_say_how_the_key_was_chosen` would tell the reader nothing establishes it is the tool's
    output when the canonical priority list establishes exactly that.
    """
    low = str(column).lower()
    if any(low == str(k).lower() for k in declared):
        return "registry"
    if any(wanted in low for wanted in _KEY_PREFERENCE) or low in _canonical_keys():
        return "name"
    return "shape"


def _say_how_the_key_was_chosen(ctx: AnalysisContext, picked: _Labels) -> None:
    """Warn unless the column is the one the tool declares it writes.

    A clustered h5ad carries the tool's labels *and* whatever came in with the data, and
    :func:`_choose_key` tells them apart by name alone. Over this repo's 17 recorded clustering
    fixtures, six publish a column the tool did not write; for the three round-tripped from
    ``mini_spatial.h5ad`` the published column is element-wise identical to the input's own
    ``cluster`` / ``annotation``. ``spacel`` is the case that needs no heuristic to see: the registry
    declares ``splane_cluster``, no output file carries it, and a different column was substituted in
    silence, under ``cluster_key``, with domain counts, a map and markers computed from it.

    Nothing here changes which column is used -- only whether the run admits it is a guess. Refusing
    the guess would be worse: five of those six runs have no registered prediction column at all, and
    for a tool that writes ``pred`` the guess is right.

    Nor does every name match deserve the warning. A run whose output holds exactly one label-shaped
    column, named ``spatial_domain``, was not choosing between anything and the name is one only a
    clustering tool writes -- saying "this was chosen by its name" there is noise, and noise is what
    stops the six real cases from being read. The warning is kept for the three shapes that carry
    information: a declared column that is missing, a pick made *between* candidates, and a pick
    whose name (:data:`_ANNOTATION_NAMES`) is one an input annotation carries as readily as a result.
    All six recorded cases are still caught -- ``data_converter`` by the third of those, its lone
    candidate being ``cell_type``.
    """
    if picked.evidence == "registry":
        return

    others = [c for c in picked.usable if c != picked.key]
    annotation_ish = any(name in str(picked.key).lower() for name in _ANNOTATION_NAMES)
    if picked.evidence == "name" and not picked.declared and not others and not annotation_ish:
        return

    shown = ", ".join(others[:_NAMES_IN_WARNING]) + ("..." if len(others) > _NAMES_IN_WARNING else "")
    alongside = f" alongside {len(others)} other label-shaped column(s) ({shown})" if others else ""
    caveat = (
        "A column that arrived with the input has the same shape as one the tool wrote, so confirm "
        "this is the tool's own output before reading the domains below."
    )

    if picked.declared:
        ctx.warn(
            f"{ctx.tool_name or 'This tool'} declares its domain assignment in "
            f"{', '.join(repr(d) for d in picked.declared)}, and no output file carries that column. "
            f"{picked.key!r} was read instead{alongside}. {caveat}"
        )
    elif picked.evidence == "name":
        ctx.warn(
            f"No prediction column is registered for this run, so the column {picked.key!r} was chosen "
            f"by its name{alongside}. {caveat}"
        )
    else:
        ctx.warn(
            f"The column {picked.key!r} matches no known domain-assignment name and none is registered "
            f"for this run; it was read because its values are shaped like labels{alongside}. Nothing "
            "establishes that it holds the tool's assignment rather than some other per-spot attribute."
        )


def _say_if_the_key_came_with_the_input(ctx: AnalysisContext, picked: _Labels) -> None:
    """Say so, and lower the run's confidence, when the picked column's name was in the input already.

    :func:`_say_how_the_key_was_chosen` can only ask the reader to confirm, because a column the
    input carried "has the same shape as one the tool wrote". When the file descends from a
    conversion utility's output most of that can be answered: the conversion recorded the names of
    the obs columns it wrote (:mod:`spatialomicsgym.tool.conversion_record`), every one of them the
    input's own or a QC count computed from it. A column with a name on that list is the user's own
    annotation, whatever the tool added beside it -- the E-03 qc_01 runs published
    ``cell_type_annot`` as the tool's "13 spatial domains" and were reviewed ``ok`` -- unless a tool
    wrote over that column. The record keeps names, not values, so that one case reads the same,
    and the warning says "unless" rather than claim to know. The findings stand; the claim that
    they are a clustering this run produced is not one the run can make.
    """
    record = conversion_record(picked.source)
    if record is None or str(picked.key) not in record.obs_columns:
        return
    made_by = record.producer or "a conversion utility"
    made_from = f" from {Path(record.source).name}" if record.source else ""
    ctx.warn(
        f"{picked.key!r} is the name of a column the input arrived with: {made_by} wrote an obs "
        f"column of that name when it converted the data{made_from}, before any tool ran. Unless a "
        "tool wrote over that column, the domains below are the input's own labels, not a clustering "
        "this run produced."
    )
    ctx.manifest.degrade()


def _label_series(series, index=None):
    """``(assigned_labels_as_str, n_unassigned)``.

    A null label is not a domain. ``Series.astype(str)`` renders one as the four-character string
    ``"nan"``, and that string then counted as a domain: ``n_domains`` came back 5 for a 4-domain
    result, ``domain_sizes.csv`` grew a ``NaN`` row, ``spatial_domains.png`` grew a colour for it,
    and ``largest_domain_frac`` divided by the unassigned spots as well.

    An integral float keeps its integer spelling for the same reason. pandas types a label column
    with a gap in it ``float64``, and ``astype(str)`` then renames every domain the tool reported
    (``3`` -> ``3.0``).
    """
    import pandas as pd

    if index is not None:
        series = pd.Series(list(series), index=list(index))
    missing = int(series.isna().sum())
    kept = series.dropna()
    if getattr(getattr(kept, "dtype", None), "kind", "") == "f" and _is_integral(kept):
        kept = kept.astype("int64")
    return kept.astype(str), missing


def _choose_key(columns: list[str], tool_name: str | None, frame) -> str | None:
    """The registry's ``prediction_key``, then the canonical names, then the substring order.

    Only the first of those tiers is evidence that the column is the tool's output; the rest are a
    name match and a shrug. :func:`_key_evidence` says which one answered, and callers are expected
    to pass that on rather than present the pick as established.
    """
    usable = [c for c in columns if _is_label_column(frame[c])]
    if not usable:
        return None
    for candidate in _registry_keys(tool_name):
        for column in usable:
            if column.lower() == str(candidate).lower():
                return column
    lowered = {c: c.lower() for c in usable}
    for wanted in _canonical_keys():
        for column, low in lowered.items():
            if low == wanted:
                return column
    for wanted in _KEY_PREFERENCE:
        for column, low in lowered.items():
            if wanted in low:
                return column
    return usable[0]


@lru_cache(maxsize=1)
def _canonical_keys() -> tuple[str, ...]:
    """The lower-cased cluster-key names benchmarking ranks, in its priority order.

    :data:`_KEY_PREFERENCE` is thirteen generic **substrings**, and when two columns both match one
    of them the tie is broken by whichever the tool happened to write first -- ``lowered`` is in
    column order. That is precisely the bug ``output_inspector._pick_cluster_column`` documents
    having fixed on the benchmarking side. DeepST writes ``DeepST_domain`` and
    ``DeepST_refine_domain``; both contain ``"domain"``, the raw one comes first in ``obs``, so
    post-analysis published the raw assignment while the scored run used the refined one -- they
    agree on 99.50% of spots in the recorded DLPFC run, and every domain count, marker gene and map
    in the report came off the column the tool does not consider its answer.

    ``_get_cluster_key_candidates(None)`` is the same profile-less list the inspector itself falls
    back to: ~25 tool-specific names, most specific first. It is consulted here as an *exact*
    case-insensitive match, which is strictly narrower than the substring pass below -- so it can
    only replace an answer that pass had no way to prefer, never override one it did.

    Cached: the list is a constant, and :func:`_choose_key` runs once per output file.
    """
    try:
        from spatialomicsgym.benchmarking.output_inspector import _get_cluster_key_candidates

        return tuple(str(k).lower() for k in (_get_cluster_key_candidates(None) or []))
    except Exception:
        return ()


def _registry_keys(tool_name: str | None) -> list[str]:
    # Resolve the caller's spelling to the registry's key first (worker JSON self-reports
    # "scanpy_spatial"; the registry says "run_scanpy_spatial_domain"). Passing the alias through
    # unresolved made get_prediction_key_candidates fall back to its GENERIC candidate list, which
    # _say_how_the_key_was_chosen then reported as what "the tool declares" -- a warning about a
    # declaration the tool never made, on a run whose column matched its real declaration exactly.
    if not tool_name:
        return []
    try:
        from spatialomicsgym.benchmarking.tool_output_registry import get_prediction_key_candidates
        from spatialomicsgym.postanalysis.detect import resolve_profile

        profile = resolve_profile(tool_name)
        canonical = getattr(profile, "tool_name", None) or tool_name
        return [str(k) for k in (get_prediction_key_candidates(canonical) or [])]
    except Exception:
        return []


def _is_label_column(series) -> bool:
    try:
        distinct = int(series.nunique(dropna=True))
    except Exception:
        return False
    if distinct < 1 or distinct > _MAX_DOMAINS:
        return False
    kind = getattr(getattr(series, "dtype", None), "kind", "")
    if str(getattr(series, "dtype", "")) == "category" or kind in ("O", "b", "i", "u"):
        return distinct <= max(1, len(series) // 2)
    # ``float64`` is how pandas types an *integer* label column that has a gap in it: mclust
    # returns ``NA`` for spots it will not assign and R writes an empty field, so ``0,1,NA,2``
    # reads back float. Rejecting the dtype outright made the tool's real result invisible -- the
    # run either failed with "no domain assignment found" or, when the tool had written the spot
    # coordinates into the same CSV, reported the *x coordinate* as the domain assignment. The
    # discriminator is not the dtype but whether the values are whole numbers: a per-spot
    # measurement (``pct_counts_mt``, a pseudotime) is not, and stays rejected.
    if kind == "f":
        return _is_integral(series) and distinct <= max(1, len(series) // 2)
    return False


def _is_integral(series) -> bool:
    """True when every non-null value of a float column is a whole number."""
    import numpy as np

    try:
        values = np.asarray(series.dropna().to_numpy(), dtype="float64")
    except (TypeError, ValueError):
        return False
    values = values[np.isfinite(values)]
    return bool(values.size) and bool(np.all(values == np.round(values)))


# ---------------------------------------------------------------------------------------
# the analyses
# ---------------------------------------------------------------------------------------


def _domain_sizes(ctx: AnalysisContext, sizes, total: int) -> None:
    table = sizes.rename("n_spots").to_frame()
    table.index.name = "domain"
    table["fraction"] = table["n_spots"] / max(total, 1)
    small = table[table["n_spots"] < 10]
    if len(small):
        ctx.warn(
            f"{len(small)} domain(s) have fewer than 10 spots ({', '.join(str(i) for i in small.index[:5])}); "
            "they are unlikely to survive a marker test"
        )
    with step(ctx.manifest, "domain sizes"):
        ctx.table(table, "domain_sizes.csv", title="Spots per spatial domain")
        ctx.figure(
            plots.bar_chart(
                [str(i) for i in table.index],
                table["n_spots"].to_numpy(),
                title="Spots per spatial domain",
                xlabel="Domain",
                ylabel="Spots",
            ),
            "domain_sizes.png",
            title="Spots per spatial domain",
            caption=f"{len(table)} domains over {total} spots",
            kind="bar",
        )


def _spatial(ctx: AnalysisContext, labels) -> None:
    notes: list[str] = []
    coordinates, source = find_coordinates(ctx.files, index=labels.index, notes=notes)
    for note in notes:
        ctx.warn(note)
    if coordinates is None:
        if not notes:
            # See the same branch in `tasks/deconvolution._spatial`: a coordinate table that was
            # found and rejected has already said so, and must not also be reported as absent.
            ctx.warn(
                "No spatial coordinates found next to the result, so the domain map was skipped. Keep "
                "the clustered h5ad (or a coordinates CSV) in the output directory to get it."
            )
        return
    ctx.analysed(source)
    with step(ctx.manifest, "domain map"):
        frame = labels.to_frame(name="domain")
        notes = []
        aligned, coordinates = align_to_coordinates(frame, coordinates, notes=notes)
        for note in notes:
            ctx.warn(note)
        if aligned is None:
            # The reason is `align_to_coordinates`'s to give -- it is the only caller-visible place
            # that knows which refusal fired -- and the loop above has already warned with it.
            ctx.manifest.degrade()
            return
        ctx.figure(
            plots.spatial_scatter(
                coordinates["x"],
                coordinates["y"],
                aligned["domain"].to_numpy(),
                title="Spatial domains",
                categorical=True,
            ),
            "spatial_domains.png",
            title="Spatial domains",
            caption=f"{aligned['domain'].nunique()} domains over {len(aligned)} spots",
            kind="spatial_map",
        )


def _markers(ctx: AnalysisContext, adata, labels, key: str) -> None:
    with step(ctx.manifest, "marker ranking"):
        import scanpy as sc

        work = adata.copy()
        # scanpy joins the category names with ',' -- an int64 label column raises there. The
        # cast has to come *after* the missing rows are dropped: ``str(nan)`` is ``"nan"``, which
        # is a perfectly ``notna()`` string, so stringifying first made the filter below a no-op
        # and every spot the tool left unassigned came back as a domain called ``nan`` with its
        # own five marker genes.
        mapped = labels.reindex([str(i) for i in work.obs_names])
        work.obs["_sog_domain"] = mapped.to_numpy()
        work = work[work.obs["_sog_domain"].notna()].copy()
        work.obs["_sog_domain"] = work.obs["_sog_domain"].astype(str).astype("category")
        if work.obs["_sog_domain"].nunique() < 2:
            ctx.warn("Marker ranking needs at least 2 domains")
            return
        work = _one_matrix(ctx, work)
        if _needs_normalising(work):
            sc.pp.normalize_total(work, target_sum=1e4)
            sc.pp.log1p(work)
        # ``use_raw`` is pinned, not left at its default. See :func:`_one_matrix`: the default is
        # ``None``, which scanpy resolves to "use ``.raw`` if present", and that silently split the
        # guard above from the statistic below.
        sc.tl.rank_genes_groups(work, "_sog_domain", method="wilcoxon", use_raw=False)
        frame = sc.get.rank_genes_groups_df(work, group=None)
        top = (
            frame.sort_values(["group", "scores"], ascending=[True, False])
            .groupby("group", observed=True)
            .head(_MARKERS_PER_DOMAIN)
            .reset_index(drop=True)
        )
        # A ranking always has a first place. ``rank_genes_groups`` returns a top five whether or not
        # anything separated the domains, and the word *marker* asserts that something did: paste,
        # spacel, squidpy, st_gears and spaceflow each published fifteen "markers" whose best
        # adjusted p-value was 0.994 or worse, at verdict ok, with ``pvals_adj`` sitting unread in
        # the CSV. Keep the pick and correct the claim, as R66 established and ``svg._tables`` does.
        measured = _significant_markers(top)
        n_sig = None if measured is None else measured[0]
        marker = n_sig is None or n_sig > 0
        ctx.table(
            top,
            "markers.csv",
            title=(
                f"Top {_MARKERS_PER_DOMAIN} markers per domain"
                if marker
                else f"Top {_MARKERS_PER_DOMAIN} ranked genes per domain -- none significant at FDR {_MARKER_FDR}"
            ),
            index=False,
        )
        # ``len(top)`` counts ``(domain, gene)`` rows -- ``_MARKERS_PER_DOMAIN`` of them per domain --
        # and the label says genes. A gene enriched in two domains occupies two rows, which is the
        # ordinary case, and it put two numbers for one table in one manifest: `_marker_heatmap`
        # captions the *distinct* symbols from this same object one line down, so a 3-domain result
        # sharing markers read "Marker genes reported: 15" beside "10 marker genes over 3 domains".
        # The row count is not lost -- the table is right there, and its title says five per domain.
        ctx.find(
            "n_markers",
            int(top["names"].nunique()),
            "Marker genes reported"
            if marker
            else f"Top-ranked genes reported -- none significant at FDR {_MARKER_FDR}",
        )
        if measured is not None:
            # After ``n_markers``, never before: findings are consumed in order and the count of the
            # reported genes is the headline. The gap between the two -- graphst_cluster reports 15
            # and 11 of them clear the threshold -- is otherwise only in the CSV.
            ctx.find("n_markers_significant", n_sig, f"Marker genes below FDR {_MARKER_FDR}")
        if n_sig == 0:
            ctx.warn(
                "No gene in the marker table is enriched in the domain it was ranked for: the "
                f"smallest adjusted p-value anywhere in the table is {measured[1]:.3g}, against an "
                f"FDR threshold of {_MARKER_FDR}. They are the highest-ranked genes of each domain "
                "and not markers of it -- the domains this tool found are not separable in this "
                "expression matrix."
            )
        _marker_heatmap(ctx, work, top, marker)


def _significant_markers(top) -> tuple[int, float] | None:
    """Distinct published genes clearing :data:`_MARKER_FDR`, and the best adjusted p among them.

    ``None`` when ``pvals_adj`` is absent or entirely unreadable. Not measured is not the same as
    zero: answering "none of them is a marker" without having looked would be the same unbacked
    claim this guard exists to remove.

    Judged over the rows the table lists, because those are the ones it makes a claim about, and as
    a test of *enrichment* only -- scanpy's ``scores`` is a signed z and its ``pvals`` two-sided, so
    a strongly depleted gene can hold a smaller p-value than anything in the top five without being
    a marker of the domain it is missing from.
    """
    import pandas as pd

    if "pvals_adj" not in top.columns:
        return None
    padj = pd.to_numeric(top["pvals_adj"], errors="coerce")
    if not padj.notna().any():
        return None
    return int(top["names"][padj < _MARKER_FDR].nunique()), float(padj.min())


def _heatmap_genes(top) -> list[str]:
    """The heatmap's columns: the published markers, capped, with every domain kept in the picture.

    ``top`` is sorted by group, so its gene order runs domain by domain and a plain
    ``published[:_MAX_HEATMAP_GENES]`` did not drop the lowest-ranked genes -- it dropped whole
    domains off the end. The figure still drew those domains a row, composed of other domains'
    markers, where by construction they are low: a heatmap titled "Mean marker expression per
    domain" then showed a domain as having nothing distinctive, which is the reverse of the data.

    Measured on the three recorded clustering runs, all of which are over the cap::

        run       domains/genes   drawn with none of their own markers
        graphst          8 / 39   domain 7
        precast         15 / 49   OD Mature 3, OD Mature 4
        stlearn          9 / 35   -- (its tail happened to fall inside the last domain)

    Taken round-robin -- every domain's rank-1 marker, then every domain's rank-2 -- the same thirty
    columns reach every domain, and what the cap drops really is each domain's tail. On those three
    runs it swaps 6, 11 and 4 of the thirty columns respectively and leaves none unrepresented.

    Below the cap the order is left exactly as ``dict.fromkeys`` produced it, so a figure that never
    had to drop a gene is unchanged column for column.

    Past ``_MAX_HEATMAP_GENES`` *domains* no thirty columns can cover them all; the leftovers come
    back unrepresented and :func:`_marker_heatmap` narrows the caption to say so.
    """
    published = list(dict.fromkeys(top["names"].tolist()))
    if len(published) <= _MAX_HEATMAP_GENES:
        return published
    ranked = [list(dict.fromkeys(sub["names"].tolist())) for _, sub in top.groupby("group", observed=True, sort=False)]
    genes: list[str] = []
    seen: set[str] = set()
    for rank in range(max((len(names) for names in ranked), default=0)):
        for names in ranked:
            if rank >= len(names) or names[rank] in seen:
                continue
            seen.add(names[rank])
            genes.append(names[rank])
            if len(genes) == _MAX_HEATMAP_GENES:
                return genes
    return genes


def _unrepresented_domains(top, genes: list[str]) -> list[str]:
    """Domains the heatmap draws a row for without one of their own markers among its columns."""
    drawn = set(genes)
    grouped = top.groupby("group", observed=True, sort=False)
    return [str(domain) for domain, sub in grouped if not set(sub["names"]) & drawn]


def _marker_heatmap(ctx: AnalysisContext, work, top, marker: bool) -> None:
    import numpy as np
    import pandas as pd

    published = list(dict.fromkeys(top["names"].tolist()))
    genes = _heatmap_genes(top)
    if not genes:
        return
    matrix = work[:, genes].X
    values = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    frame = pd.DataFrame(values, columns=genes)
    frame["_g"] = list(work.obs["_sog_domain"].astype(str))
    mean = frame.groupby("_g", observed=True).mean()
    # ``marker`` is required and not defaulted: a future caller that has not established enrichment
    # must have to say so rather than inherit the claim.
    title = "Mean marker expression per domain" if marker else "Mean expression of the top-ranked genes per domain"
    # ``len(genes)`` is how many columns were drawn; ``n_markers`` one line up is how many the table
    # holds, and the cap between them was silent. stlearn publishes 33 and draws 30 -- two counts of
    # one set, three lines apart, with nothing saying which is the set. R66's repair: keep the
    # picture, correct the claim.
    noun = "marker genes" if marker else "top-ranked genes"
    drawn = f"{len(genes)} of the {len(published)} {noun}" if len(genes) < len(published) else f"{len(genes)} {noun}"
    caption = (
        f"{drawn} over {len(mean)} domains"
        if marker
        else f"{drawn} over {len(mean)} domains -- none significant at FDR {_MARKER_FDR}"
    )
    # With more domains than columns, no selection can reach them all: the rows that stay empty-handed
    # are drawn out of other domains' genes, so the caption has to stop implying otherwise.
    if orphans := _unrepresented_domains(top, genes):
        shown = ", ".join(orphans[:_NAMES_IN_WARNING])
        if len(orphans) > _NAMES_IN_WARNING:
            shown += f" and {len(orphans) - _NAMES_IN_WARNING} more"
        plural = ("domain", "has", "its") if len(orphans) == 1 else ("domains", "have", "their")
        caption += f"; {len(orphans)} {plural[0]} ({shown}) {plural[1]} no marker of {plural[2]} own among the columns"
    ctx.figure(
        plots.heatmap(mean, title=title, cmap="viridis", center_zero=False),
        "domain_markers.png",
        title=title,
        caption=caption,
        kind="heatmap",
    )


def _sample_values(adata):
    """Up to 50 rows of ``adata.X`` as finite dense values -- enough to characterise the matrix."""
    import numpy as np

    matrix = adata.X
    sample = matrix[: min(50, matrix.shape[0])]
    values = sample.toarray() if hasattr(sample, "toarray") else np.asarray(sample)
    return values[np.isfinite(values)]


def _one_matrix(ctx: AnalysisContext, work):
    """Resolve ``.raw`` away, so one matrix answers every question asked below.

    ``sc.tl.rank_genes_groups`` leaves ``use_raw`` at ``None``, which scanpy resolves to *"use
    ``.raw`` if present"*. Nothing else here knows that: :func:`_needs_normalising` reads ``X`` to
    decide whether to normalise, and :func:`_marker_heatmap` slices ``X`` by the gene names the
    ranking returned. So on ``adata.raw = adata`` -- the commonest scanpy recipe there is -- the
    guard vetted the log-normalised matrix, declined to normalise, and the statistic was then
    computed on raw counts. Measured on 200 spots: the published ``logfoldchanges`` came out
    ``inf`` where the vetted matrix gives ``4.57``, with scanpy printing "It seems you use
    rank_genes_groups on the raw count data" into a log nobody reads.

    Which matrix is the right one depends on what ``X`` holds:

    * ``X`` scaled (z-scores, so negative) and ``.raw`` holding something that is *not* scaled --
      ``.raw`` is the interpretable one, because ``logfoldchanges`` is
      ``log2(expm1(mean_group) / expm1(mean_rest))`` and ``expm1`` of a z-score means nothing. This
      is the one case the old default got right for the ranking, and it still got the guard and the
      heatmap wrong, so the promotion is explicit here instead of implicit three calls away.
    * ``X`` scaled with nothing unscaled to fall back on -- there is no repair to make, so the
      column of blanks that comes back is disclosed rather than presented as a result. Being
      handed a ``.raw`` does not by itself mean an unscaled matrix exists: stlearn's ``.raw`` is
      ``array_equal`` to its ``X``, and promoting it returned the same z-scores under a warning
      announcing that markers had been ranked on the unscaled matrix instead.
    * otherwise ``X`` -- which is what the guard already assumed, and what the heatmap slices.

    Either way the returned object has no ``.raw``, so ``use_raw=False`` is not merely correct but
    unambiguous, and the heatmap's gene names come from the same var space it indexes.
    """
    scaled = _is_scaled(work)

    if work.raw is not None:
        if scaled and not _is_scaled(work.raw):
            promoted = work.raw.to_adata()  # carries obs, so ``_sog_domain`` rides along
            ctx.warn(
                "The expression matrix is scaled (it has negative values), which no fold change can be "
                "computed from, so markers were ranked on the unscaled matrix in .raw instead."
            )
            return promoted
        del work.raw

    if scaled:
        ctx.warn(
            "The expression matrix is scaled (it has negative values) and the result carries no "
            "unscaled copy of it, so the logfoldchanges column of the marker table is empty. The "
            "gene ranking and its p-values come from a rank-based test and are reported as usual."
        )
    return work


def _is_scaled(holder) -> bool:
    """Whether the matrix has been centred, which is what makes ``expm1`` of its means meaningless.

    Takes anything carrying an ``.X`` -- an ``AnnData``, or the ``Raw`` hanging off one -- so the
    same question can be put to the matrix in hand and to the matrix offered as its replacement.
    """
    values = _sample_values(holder)
    return bool(values.size and float(values.min()) < 0.0)


def _needs_normalising(adata) -> bool:
    """Whether ``X`` is on a linear scale, so it has to be normalised and logged before ranking.

    ``logfoldchanges`` is ``log2(expm1(mean_group) / expm1(mean_rest))``, which inverts a ``log1p``
    that must already have been applied. Hand it a linear matrix and ``expm1`` of a group mean in
    the thousands overflows: the recorded deepst run published ``IGHG3 371.43``, ``IGHG4 inf`` and
    an empty ``IGKC`` from a matrix whose maximum is 5561.

    The test is the magnitude, and only the magnitude. ``log1p(x) > 30`` needs ``x`` above 1e13, so
    nothing already logged comes near the bar, while a library-size-normalised matrix clears it
    easily. This used to also require the values to be whole numbers, which recognised raw counts
    and missed exactly the case above -- same scale, same need to be logged, fractional values.
    A linear matrix whose maximum falls under the bar is still missed; no recorded run sits there.

    Negatives are excluded because a scaled matrix is not linear in the sense that matters here:
    ``normalize_total`` divides by a row sum that means nothing once the row holds z-scores, and
    scaled values are not bounded under the bar the way logged ones are -- a gene detected in one
    spot of n scores about sqrt(n) there. :func:`_one_matrix` is what handles that matrix instead.
    """
    values = _sample_values(adata)
    if values.size == 0:
        return False
    return bool(values.min() >= 0.0 and values.max() > 30)
