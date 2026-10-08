"""What the runs measured, laid out as a table -- and the one thing such a table must not become.

Every post-analysis run already records its numbers. :meth:`Manifest.add_finding` writes
``{key, value, label}`` into ``findings[]``, and across the 122 manifests on this box **121 carry at
least one**, so a page built on ``findings[]`` needs no schema change, no new writer and no backfill
of the runs that already exist. That is the whole reason this module reads that list and not a
``metrics`` block: a key added to the schema today is empty on every run written before today, and a
column that is blank for all of history is not a measurement, it is a promise.

What ``findings[]`` does not carry is **type**. A value arrives as ``0.991`` whether it is a share, a
count or a score, and ``0.991`` rendered raw is the same characters in all three cases. So this
module holds one table -- :data:`FINDING_REGISTRY` -- mapping the keys the shipped tools emit to a
unit and a sentence. An unrecognised key still renders: :func:`spec_for` derives a spec from the key
itself, so a tool that starts emitting ``n_niches`` next week appears in the table under its own
name rather than disappearing from it.

**There is no ranking here, and that is a finding rather than an omission.**

*  No ``findings[]`` key has a *direction*. A higher ``n_domains`` is not a better run -- the right
   number of domains is a property of the tissue. A ``significant_fraction`` of 0.991 is a red flag,
   not a win. Sorting any of these columns would invent an ordering the measurement does not have.
*  Scoring and post-analysis are mutually exclusive by contract
   (``docs/design/post_analysis_contract.md``, enforced in ``postanalysis/engine.py``): on this box
   486 ``metrics.json`` files and 122 ``manifest.json`` files share not one directory. So there is no
   accuracy number to rank *by*, even for a page that wanted one.
*  The manifest carries no dataset identifier, so "these two runs did the same job" is not a
   question this data can answer. Two runs of the same tool over different tissue compare as
   cleanly as two runs of different tools, which is to say not at all.

What *can* be ordered is the reviewer's own verdict, because L2 emits it from a fixed, ordered
vocabulary. :func:`standings` does exactly that and nothing more: it groups by ``task_type``, orders
by ``(verdict, share of checks passed, recency)``, and puts every run the reviewer did not judge in
a separate list instead of at the bottom of this one. A run with zero judgeable checks ranked last
with a score of 0 reads as "reviewed, and it failed"; it means "not reviewed".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Findings that are post-analysis counting its own scan rather than something a tool measured.
#: Read from the schema module so the two lists cannot drift; the literal is the fallback for a
#: partial tree, exactly as ``report.manifest`` does for ``VERDICTS``.
try:  # pragma: no cover - the fallback only fires on a broken/partial tree
    from spatialomicsgym.postanalysis.manifest import BOOKKEEPING_FINDINGS
except Exception:  # pragma: no cover
    BOOKKEEPING_FINDINGS = frozenset({"n_output_files", "n_staged_inputs"})

#: The reviewer's vocabulary **in its own order** -- best first. Taken from ``review.py``, which is
#: where the order lives; ``report.manifest.VERDICTS`` is a ``frozenset`` built from this same tuple
#: and has therefore lost the only property a ranking needs.
try:  # pragma: no cover - as above
    from spatialomicsgym.postanalysis.review import VERDICTS as VERDICT_ORDER
except Exception:  # pragma: no cover
    VERDICT_ORDER = ("ok", "suspicious", "unusable")

#: Units. ``COUNT`` is a whole number of things, ``FRACTION`` a share in ``[0, 1]`` shown as a
#: percentage, ``FLAG`` a yes/no, ``NAME`` an identifier the tool chose (a column, a gene, a cell
#: type) that must be shown verbatim because it is what the user will search their own data for.
COUNT = "count"
FRACTION = "fraction"
FLAG = "flag"
NAME = "name"
UNKNOWN = ""


#: How a unit is said on screen. The constants above are identifiers; these are what a reader sees
#: beside a column name, and they are phrased as what the number *is* rather than as a type name --
#: "a share, shown as a percentage" tells someone why the cell says 99.1% when the manifest says
#: 0.991, which "fraction" does not.
UNIT_WORDS: dict[str, str] = {
    COUNT: "a count",
    FRACTION: "a share, shown as a percentage",
    FLAG: "yes or no",
    NAME: "a name the tool chose",
}


def unit_word(unit: str) -> str:
    """``UNIT_WORDS`` with a safe answer for the unit this module could not derive."""
    return UNIT_WORDS.get(unit, "")


@dataclass(frozen=True)
class FindingSpec:
    """How to print one ``findings[]`` key, and what its number does not mean."""

    label: str
    unit: str
    note: str = ""
    #: True for the two keys post-analysis appends about its own scan. Kept out of the metrics
    #: table by default: ``n_output_files`` is present on 99 of the 122 runs here and would be the
    #: widest column on the page while saying nothing about the biology.
    bookkeeping: bool = False


#: The keys the shipped task runners emit, measured across every manifest in this checkout. ``label``
#: is what a column spanning several runs is called; the wording an individual run recorded is kept
#: separately by :func:`labels_of`, because it can be about that run alone ("First of 1522 genes tied
#: at the best score"). What only this table knows is the unit, and the sentence under ``note`` --
#: which is there because for most of these keys the reader's first instinct, that bigger is better,
#: is wrong.
FINDING_REGISTRY: dict[str, FindingSpec] = {
    # -- post-analysis' own bookkeeping ------------------------------------------------------
    "n_output_files": FindingSpec("Files written", COUNT, "Counted by the scan, not by the tool.", True),
    "n_staged_inputs": FindingSpec("Staged inputs", COUNT, "Files the tool was handed, not files it made.", True),
    # -- shared ------------------------------------------------------------------------------
    "signal_free": FindingSpec(
        "No spatial signal",
        FLAG,
        "Set when the result carries no spatial information at all. The one flag here that is unambiguously bad.",
    ),
    # -- spatial_clustering ------------------------------------------------------------------
    "cluster_key": FindingSpec("Domain column", NAME, "Where in the output the assignment was found."),
    "n_spots": FindingSpec("Spots assigned", COUNT, "The size of the slide, not the quality of the answer."),
    "n_domains": FindingSpec(
        "Domains",
        COUNT,
        "More is not better. The right number is a property of the tissue, and is usually the "
        "number that was asked for.",
    ),
    "largest_domain_frac": FindingSpec(
        "Largest domain",
        FRACTION,
        "A share near 1.0 means one domain swallowed the slide; near 1/n means the domains came "
        "out even. Neither is a score.",
    ),
    "n_markers": FindingSpec("Markers reported", COUNT, "Genes the marker step ranked, before any threshold."),
    "n_markers_significant": FindingSpec("Markers below FDR", COUNT, "Of those, the ones that passed the cutoff."),
    # -- deconvolution -----------------------------------------------------------------------
    "n_cell_types": FindingSpec("Cell types", COUNT, "Comes from the reference, not from the slide."),
    "top_cell_type": FindingSpec("Most abundant", NAME, "The reference's own name for it, verbatim."),
    "n_dominant_types": FindingSpec(
        "Types dominating a spot",
        COUNT,
        "How many of the reference's types win at least one spot. A small number over many types is worth a look.",
    ),
    "orientation_transposed": FindingSpec(
        "Table was transposed",
        FLAG,
        "The proportion table arrived cell-types-by-spots and was turned on read. Not an error, "
        "but it is the check that catches a spot id read as a cell type.",
    ),
    "celltype_names_resolved": FindingSpec(
        "Columns are named types",
        FLAG,
        "False means the columns are positional, so every name on this row is a guess.",
    ),
    # -- svg_detection -----------------------------------------------------------------------
    "n_genes": FindingSpec("Genes tested", COUNT, "The denominator for the two below."),
    "n_significant": FindingSpec("Genes significant", COUNT, "At the threshold the tool applied, not a fixed one."),
    "significant_fraction": FindingSpec(
        "Share significant",
        FRACTION,
        "High is a red flag, not a result: a test calling most of the transcriptome spatially "
        "variable is usually reading a raw p-value as if it were corrected.",
    ),
    "score_column": FindingSpec(
        "Ranked on",
        NAME,
        "Which column the ranking was read from. An FDR column and a raw p-value column produce "
        "very different lists from the same run.",
    ),
    "top_gene": FindingSpec("Top gene", NAME, "First by the score column above; see the tie count."),
    "n_tied_at_top": FindingSpec(
        "Tied at the top",
        COUNT,
        "How many genes share that best score. A large tie means the ranking is arbitrary within "
        "it, and 'top gene' names one of them.",
    ),
}

#: Derived unit when a key is not in the registry. Deliberately small: guessing a unit wrongly
#: prints a count as a percentage, which is worse than printing it plainly.
_COUNT_PREFIX = "n_"
_FRACTION_SUFFIXES = ("_frac", "_fraction", "_share", "_rate")


def _humanise(key: str) -> str:
    """``n_tied_at_top`` -> ``N tied at top``. Only ever a fallback for an unregistered key."""
    words = str(key or "").replace("-", "_").split("_")
    words = [w for w in words if w]
    if not words:
        return "Finding"
    head = words[0].upper() if len(words[0]) == 1 else words[0].capitalize()
    return " ".join([head, *words[1:]])


def spec_for(key: str) -> FindingSpec:
    """The spec for ``key``. Never ``None`` -- an unknown key gets one derived from its own name.

    Deliberately blind to the label the manifest recorded, even though that label is the tool's own
    and this one is not. A recorded label describes *one run*: ``top_gene`` arrives on this box
    labelled "First of 1522 genes tied at the best score", where 1522 is that run's transcriptome.
    As a cell caption that is better than anything written here; as the heading of a column spanning
    six runs it is simply false for five of them, and which run's wording won would depend on the
    order the rows happened to be listed in. So the column is named once, by this registry, and the
    recorded wording is shown beside the key where a reader can see whose words they are.
    """
    spec = FINDING_REGISTRY.get(key)
    if spec is None:
        unit = UNKNOWN
        if str(key).startswith(_COUNT_PREFIX):
            unit = COUNT
        elif any(str(key).endswith(s) for s in _FRACTION_SUFFIXES):
            unit = FRACTION
        spec = FindingSpec(_humanise(key), unit)
    return spec


#: ``report.manifest._json_scalar`` spells a non-finite float rather than dropping it, so these
#: three arrive here as strings and must not be formatted as if they were names.
_NON_FINITE = {"NaN", "Infinity", "-Infinity"}


def format_value(key: str, value: Any) -> str:
    """One cell, printed in its unit. Never raises, and never invents precision.

    A missing value is the empty string and the caller renders it as a blank cell -- never ``0``,
    which a reader takes for a measurement that came out zero.
    """
    if value is None or value == "":
        return ""
    if isinstance(value, str) and value in _NON_FINITE:
        return value
    unit = spec_for(key).unit
    if isinstance(value, bool):
        return "yes" if value else "no"
    if unit == FLAG:
        return "yes" if value else "no"
    if unit == FRACTION and isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    if unit == COUNT and isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        # Four places is what the tools record; trailing zeros are noise, not precision.
        return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def findings_of(manifest: dict, *, bookkeeping: bool = False) -> dict[str, Any]:
    """``{key: value}`` for one run. Bookkeeping keys are excluded unless asked for.

    Later entries win, matching ``add_finding``'s replace-by-key semantics -- a manifest hand-edited
    into carrying the same key twice reads here the way the writer would have left it.
    """
    out: dict[str, Any] = {}
    for entry in manifest.get("findings") or ():
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key") or "")
        if not key:
            continue
        if not bookkeeping and (key in BOOKKEEPING_FINDINGS or spec_for(key).bookkeeping):
            continue
        out[key] = entry.get("value")
    return out


def labels_of(manifest: dict) -> dict[str, str]:
    """``{key: label}`` as *this run* recorded them, which is not what the column is called.

    Kept apart from :func:`spec_for` on purpose, and the separation is the whole point: these are
    one run's words about one run's number, so they belong beside the key in a legend and never in a
    heading over six rows. See :func:`spec_for` for the measured example that settles it.
    """
    out: dict[str, str] = {}
    for entry in manifest.get("findings") or ():
        if isinstance(entry, dict) and entry.get("key"):
            out[str(entry["key"])] = str(entry.get("label") or "")
    return out


def columns(rows: list[dict[str, Any]]) -> list[str]:
    """The union of keys across ``rows``, registry order first, then unknown keys as first seen.

    Registry order rather than alphabetical, because the registry is grouped by task and a reader
    scanning a clustering table wants ``n_domains`` beside ``largest_domain_frac`` rather than
    between ``n_cell_types`` and ``n_genes``. Unknown keys go after the known ones rather than being
    interleaved, so a tool that starts emitting something new is visibly new.
    """
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    known = [k for k in FINDING_REGISTRY if k in seen]
    return known + [k for k in seen if k not in FINDING_REGISTRY]


# --------------------------------------------------------------------------- #
# review standings -- the only thing on this page that is ordered
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Standing:
    """One run's place in its task group, or no place at all."""

    row: dict
    verdict: str
    passed: int
    judged: int
    #: 1-based, or ``None`` -- for an unreviewed run, and for the sole member of a group. Ranking one
    #: run against nothing is a crown for turning up.
    rank: int | None = None

    @property
    def passed_fraction(self) -> float:
        return (self.passed / self.judged) if self.judged else 0.0


def review_score(review: Any) -> tuple[int, int]:
    """``(passed, judged)`` over the checks that reached an answer.

    ``passed`` is a tristate in the contract: ``None`` means *this layer could not read it*, which is
    neither a pass nor a failure and must not be counted as either. A review made entirely of those
    scores ``(0, 0)``, and :func:`standings` reads that as "not reviewed" rather than as zero.
    """
    if not isinstance(review, dict):
        return (0, 0)
    passed = judged = 0
    for check in review.get("checks") or ():
        if not isinstance(check, dict):
            continue
        state = check.get("passed")
        if state is None:
            continue
        judged += 1
        if state:
            passed += 1
    return (passed, judged)


def _rank_key(standing: Standing) -> tuple:
    row = standing.row
    return (
        VERDICT_ORDER.index(standing.verdict),
        -standing.passed_fraction,
        -float(row.get("mtime") or 0.0),
        str(row.get("dir") or ""),
    )


def standings(entries: list[tuple[dict, dict]]) -> tuple[list[Standing], list[Standing]]:
    """``(ranked, unreviewed)`` for one group of runs. ``entries`` is ``(list row, manifest)``.

    A run is *unreviewed* when L2 has not run on it, when it emitted a verdict outside the contract's
    vocabulary, or when every check it did emit came back unreadable. All three mean the same thing
    to a reader -- nobody has judged this -- and none of them is "it scored zero".
    """
    ranked: list[Standing] = []
    unreviewed: list[Standing] = []
    for row, manifest in entries:
        review = manifest.get("review") if isinstance(manifest, dict) else None
        verdict = str((review or {}).get("verdict") or "") if isinstance(review, dict) else ""
        passed, judged = review_score(review)
        standing = Standing(row=row, verdict=verdict, passed=passed, judged=judged)
        if verdict in VERDICT_ORDER and judged:
            ranked.append(standing)
        else:
            unreviewed.append(standing)
    ranked.sort(key=_rank_key)
    if len(ranked) < 2:
        # One run is not a standing. Left in the list so the page can still show its verdict and
        # its check tally; it simply carries no number and no position.
        ranked = [Standing(s.row, s.verdict, s.passed, s.judged, None) for s in ranked]
    else:
        ranked = [Standing(s.row, s.verdict, s.passed, s.judged, i + 1) for i, s in enumerate(ranked)]
    unreviewed.sort(key=lambda s: (-float(s.row.get("mtime") or 0.0), str(s.row.get("dir") or "")))
    return ranked, unreviewed


def group_by_task(entries: list[tuple[dict, dict]]) -> list[tuple[str, list[tuple[dict, dict]]]]:
    """``entries`` split by ``task_type``, the largest group first, ``unknown`` always last.

    By ``task_type`` **alone**. Grouping any finer would need a dataset identifier, and the manifest
    does not carry one -- so "the same job, done two ways" is not a grouping this data supports, and
    a page that pretended otherwise would be comparing a lymph node against a brain section.
    """
    buckets: dict[str, list[tuple[dict, dict]]] = {}
    for row, manifest in entries:
        task = str(row.get("task_type") or "unknown") or "unknown"
        buckets.setdefault(task, []).append((row, manifest))
    ordered = sorted(
        buckets.items(),
        key=lambda item: (item[0] == "unknown", -len(item[1]), item[0]),
    )
    return ordered
