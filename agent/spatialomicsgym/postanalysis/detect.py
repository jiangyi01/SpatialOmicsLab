"""What kind of task produced these files.

Three sources of truth, in falling order of confidence:

1. **The tool name.** ``benchmarking/tool_output_registry`` already records, per tool, what its
   output looks like and which task it belongs to. That knowledge is read, never re-stated here.
2. **The registry's filename patterns.** ``stdgcn_proportions.csv`` matches exactly one profile's
   ``authoritative_output``; so does ``SpotGF_scores.txt``. Only patterns specific enough to name a
   tool count -- ``*results*.csv`` and ``proportions.csv`` belong to several task types between
   them, and a guess from those is worse than looking at the data.
3. **The data.** Whether the numbers are a proportions matrix, a per-gene ranking or a label column
   is decided by the predicates in :mod:`.tables`, which check that values vary rather than that
   names look plausible.

Imputation is not content-detectable -- an imputed h5ad looks like any other h5ad -- so it is
reachable through the tool name or an explicit ``task_type`` only. That is stated in the returned
reason rather than guessed at.

Alignment WAS in that sentence until 2026-09-21, and it was true while the only evidence was a
CSV's column names. It stopped being true once aligned coordinates got an agreed home: a frame
under ``obsm`` or the ``uns['spatial_3d']`` provenance block is stronger evidence than a filename.
The branch sits before the clustering one on purpose -- an aligned object routinely carries a
cell-type column, being the same cells annotated, so clustering would otherwise claim it and the
run would be measured as a clustering.
"""

from __future__ import annotations

import fnmatch
import re
from typing import TYPE_CHECKING

from .sources import is_staged_input
from .tables import (
    MIN_UNNAMED_SPOTS,
    group_key,
    is_tabular,
    looks_like_composition_summary,
    looks_like_gene_table,
    looks_like_marker_table,
    looks_like_proportions,
    read_table,
    spot_evidence,
    tabular_sep,
)

if TYPE_CHECKING:
    from pathlib import Path

#: Rows read from each table before deciding what it is. Named because
#: :func:`~.tables.looks_like_composition_summary` has to be told the cap to distrust a sum taken
#: over a frame that filled it.
_CONTENT_ROWS = 200

#: Delimiter characters :func:`~.file_io.sniff_tabular_sep` does not offer, so a table written with
#: one parses into a single column whose lone *name* still holds the character. Their presence there
#: is what separates "this file is one column" from "this file was never split".
#:
#: Space is not among them, and the omission is the same one the sniffer makes: a column of labels
#: like ``B cells`` is a genuine one-column list, so reading a space as a missed delimiter would
#: hand the reader wrong advice about the one arrangement this codebase has documented as a hazard.
_UNOFFERED_SEPARATORS = (";", "|")

#: Literal tokens too common to identify a tool: several profiles' patterns contain them.
#:
#: These are *task* vocabulary. The test to apply before adding one is whether more than one profile
#: can claim it: ``svg`` is in two profiles' patterns (``prost_index_svg``, ``spagft_identify_svg``,
#: both publishing the identical ``*svg*.csv``), so it names a task and not a tool, and the recorded
#: SpatialDE directory came back named ``prost_index_svg`` while it was absent. ``bsp`` reads like
#: the same kind of word and is not: one profile publishes it, and it is that tool's actual name.
_GENERIC_TOKENS = frozenset(
    {
        "result",
        "results",
        "svg",
        "svgs",
        "output",
        "outputs",
        "proportion",
        "proportions",
        # Three profiles publish one (tacco, ncem, svca) and every deconvolution runner in the
        # corpus writes a mean-composition summary beside its result whether the registry says so or
        # not, so the word says which task this is. On the bare ``*composition*.csv`` it named tacco
        # for three recorded runs tacco had no part in, and tied with the real tool's pattern on
        # three more, leaving those unnamed. The names built around the word keep their own tokens.
        "composition",
        "score",
        "scores",
        "gene",
        "genes",
        "cluster",
        "clusters",
        "domain",
        "domains",
        "data",
        "matrix",
        "summary",
        "predicted",
        "annotated",
        "spatial",
        "cell",
        "celltype",
        "cell_type",
    }
)

_MIN_SPECIFICITY = 2

#: Word boundaries inside a filename pattern: glob wildcards *and* the separators people type.
#: Splitting on wildcards alone leaves ``predicted_genes`` as one token, which no list of single
#: words can ever catch.
_WORDS = re.compile(r"[*?_\-.\s]+")

#: obs columns that carry a domain/cluster label, beyond whatever the registry names for the tool.
_CLUSTER_OBS_KEYS = (
    "spatial_domain",
    "domain",
    "cluster",
    "leiden",
    "louvain",
    "mclust",
    "kmeans",
    "region",
    "annotation",
    "celltype",
    "cell_type",
)

_PSEUDOTIME_KEYS = ("pseudotime", "dpt_pseudotime", "latent_time", "ptime", "trajectory")

_COMMUNICATION_COLUMN_RE = re.compile(r"ligand|receptor|^s-|^r-|sender|receiver|interaction|lr_pair", re.I)

#: What a spatial *alignment* result names its columns. Two families, because the two recorded
#: shapes are different objects: the new position of every spot (``aligned_0``, ``aligned_1``), and
#: which cell in one section matched which in another (``slice1_idx``, ``slice2_idx``). Anchored on
#: separators so a gene called ``MISALIGN1`` or a cell type called ``Realigned macrophage`` cannot
#: count -- these fire on the word, not on the letters.
_ALIGNMENT_COLUMN_RE = re.compile(r"(^|[_\-.])align(ed|ment)?([_\-.]|\d|$)", re.I)
_SECTION_INDEX_RE = re.compile(r"^(slice|section|sample|batch|dataset)[_\-]?\d+[_\-]?(idx|index|id)$", re.I)

#: One matching column is a label on a table about something else; two are the table's subject.
_MIN_ALIGNMENT_COLUMNS = 2

#: Above this many distinct values, a column is a measurement, not a set of domain labels.
_MAX_DOMAIN_CARDINALITY = 60

#: How far past the chosen table to keep looking for a competing verdict. The pick itself always
#: stops at the first match; this bounds only the ambiguity survey, so a directory of 500 tables
#: cannot turn one detection into 500 table reads.
_AMBIGUITY_SCAN_LIMIT = 25

#: Opens the one note the content route emits, and marks it as a note about *confidence*: the file
#: analysed won its place in a ``sorted()`` call rather than on evidence. The engine imports this to
#: decide whether to degrade the manifest, instead of matching the sentence a second time, so the two
#: modules cannot come to disagree about which notes mean the answer was a tie-break.
AMBIGUOUS_OUTPUT_NOTE = "Ambiguous output: "

#: The one "no shape matched" reason that describes real, parsed output rather than a broken
#: artifact. The engine imports it (same arrangement as :data:`AMBIGUOUS_OUTPUT_NOTE`) to keep such
#: a run at ``partial``: the contract reserves ``failed`` -- and the ``unusable`` verdict review
#: turns it into -- for results that are wrong, and a file that read fine but fits no known task
#: shape is merely unclassified. The recorded live case is a converted Visium ``.h5ad``: the
#: conversion succeeded, and ``failed`` made the portal badge a good file unusable.
READABLE_BUT_UNRECOGNIZED = "read fine but matched no known task shape"


def task_type_for_tool(tool_name: str | None) -> str | None:
    """The registry's task type for ``tool_name``, including dynamically registered tools."""
    profile = resolve_profile(tool_name)
    return getattr(profile, "task_type", None) if profile is not None else None


def resolve_profile(tool_name: str | None):
    """The registry profile for ``tool_name``, tolerating the names a live caller actually holds.

    The registry is keyed by portal function name (``run_scanpy_spatial_domain``). A live caller
    rarely has that exact string: the worker's own result JSON self-reports ``"tool":
    "scanpy_spatial"``, the MCP server banner says ``scanpy-spatial``, and the prompt tells the
    model to pass "the tool you called". On the recorded lymph-node run the model-faithful
    spelling ``scanpy_spatial`` resolved to no profile, so the authoritative task-type route died
    and detection fell through to filename/content guessing -- which a staging heuristic then
    emptied, leaving the manifest ``failed`` beside four perfectly good result files.

    Resolution order: the exact key; the key with ``run_`` prefixed; the name with its own
    ``run_`` stripped; finally a unique registry key that extends the name at a ``_`` boundary
    (``scanpy_spatial`` -> ``run_scanpy_spatial_domain``). Hyphens are read as underscores and
    case is folded first. Ambiguity refuses rather than guesses: two keys extending the same name
    mean the caller's name does not identify a tool. Best-effort and never raises.
    """
    if not tool_name:
        return None
    name = str(tool_name).strip().lower().replace("-", "_")
    try:
        from spatialomicsgym.benchmarking.tool_output_registry import get_all_profiles, get_profile

        for candidate in (name, f"run_{name}", name[len("run_") :] if name.startswith("run_") else None):
            if not candidate:
                continue
            profile = get_profile(candidate)
            if profile is not None:
                return profile
        profiles = get_all_profiles()
    except Exception:
        return None
    stem = name[len("run_") :] if name.startswith("run_") else name
    hits = [k for k in profiles if k.startswith((f"{stem}_", f"run_{stem}_"))]
    return profiles[hits[0]] if len(hits) == 1 else None


def authoritative_patterns(tool_name: str | None) -> list[str]:
    """The registry's ``authoritative_output`` globs for ``tool_name``, most preferred first."""
    profile = resolve_profile(tool_name)
    if profile is None:
        return []
    return _as_list(getattr(profile, "authoritative_output", ""))


def match_authoritative(files: list[Path], tool_name: str | None) -> list[Path]:
    """Files matching the tool's authoritative output patterns, in pattern order."""
    ordered: list[Path] = []
    for pattern in authoritative_patterns(tool_name):
        for path in files:
            if path not in ordered and fnmatch.fnmatch(path.name, pattern):
                ordered.append(path)
    return ordered


def detect_task_type(
    files: list[Path], tool_name: str | None = None, *, notes: list[str] | None = None
) -> tuple[str | None, str]:
    """``(task_type, why)``. ``task_type`` may be a non-contract string when the registry says so.

    Pass ``notes`` to collect what the decision could not settle. Only the *content* route fills it,
    and only when the files support more than one task type: there the answer rests on a
    ``sorted()`` tie-break, which is not knowledge. The tool-name and registry-filename routes are
    authoritative and are never second-guessed. Task runners hand these to ``ctx.warn``, the same
    way they already do for :func:`sources.align_to_coordinates`.
    """
    from_tool = task_type_for_tool(tool_name)
    if from_tool:
        return from_tool, f"tool_output_registry profile for {tool_name!r}"

    task, tool = _from_registry_filenames(files)
    if task:
        # ``tool`` is deliberately ``None`` when several profiles claim the filename and agree only
        # on the task type -- see ``_from_registry_filenames``. This string was written when it was
        # always a name and rendered that refusal as "the registry profile for None", naming a
        # profile that does not exist; ``review.py`` carries a comment about the same mistake at its
        # own site. Three registry patterns are shared this way today, across three task types.
        return task, (
            f"output filename matches the registry profile for {tool!r}"
            if tool
            else "output filename matches the authoritative output of several registry tools, "
            f"which agree it is {task} and do not agree on which of them wrote it"
        )

    task, why = _from_content(files, notes=notes)
    if task:
        return task, why
    return None, why or "no tool name, no registry filename match, and the data matched no known task shape"


# ---------------------------------------------------------------------------------------


def tool_name_from_filenames(files: list[Path]) -> str | None:
    """The registry tool whose authoritative output is sitting in ``files``, or ``None``.

    :func:`detect_task_type` has always worked this out and then thrown it away into the ``why``
    prose. Nothing downstream could use it, so ``tool_name`` was null on 100% of recorded production
    manifests -- the only caller that ever supplies one is a test -- which left two things
    permanently inert that were written to depend on it: :func:`match_authoritative`, so every task
    runner picked its input file by suffix and alphabetical order, and L2's
    ``published_output_slot`` check, which needs a registry profile to have a slot to check against.

    Best-effort and never raises: a missing or unimportable registry means ``None``, the same
    already-handled state as "the caller named no tool".
    """
    try:
        return _from_registry_filenames(list(files))[1]
    except Exception:
        return None


def _from_registry_filenames(files: list[Path]) -> tuple[str | None, str | None]:
    """``(task_type, tool_name)`` from filenames alone, or ``(None, None)``.

    Staged inputs are dropped first, for the same reason :func:`_from_content` drops them and with
    more at stake: this route runs *before* the content route and is never second-guessed, so one
    derived artifact under ``standardized/`` or ``prep/`` decides the task type outright and the
    run's own output is never opened. ``sources.is_staged_input`` is asked rather than re-derived --
    it matches per path component *and* accepts a marker followed by a separator, which is what
    catches the recorded ``standardized.bak_20260501_231156_visium_dup/``.

    The two halves of the answer are decided separately, because the evidence for them is not the
    same strength. Several tools tying at the top score still agree on the *task type* often enough
    to be worth returning -- ``prost_index_svg`` and ``spagft_identify_svg`` both publish
    ``*svg*.csv`` and both do SVG detection, so a directory matching it is an SVG run whichever one
    wrote it. They do not agree on the *tool*, and returning ``hits[0]`` returned whichever came
    first in registry order: the recorded SpatialDE directory came back named ``prost_index_svg``.
    Alphabetical order is a tie-break, not a judgement. A tied tool name is dropped and the task
    type kept, which is exactly what the evidence supports.
    """
    files = [f for f in files if not is_staged_input(f)]
    if not files:
        return None, None
    try:
        from spatialomicsgym.benchmarking.tool_output_registry import get_all_profiles

        profiles = get_all_profiles()
    except Exception:
        return None, None

    best_score = 0
    hits: list[tuple[str, str]] = []  # (task_type, tool_name)
    for tool_name, profile in profiles.items():
        for pattern in _as_list(getattr(profile, "authoritative_output", "")):
            score = _pattern_specificity(pattern, tool_name)
            if score < _MIN_SPECIFICITY or score < best_score:
                continue
            if not any(fnmatch.fnmatch(f.name, pattern) for f in files):
                continue
            if score > best_score:
                best_score, hits = score, []
            hits.append((profile.task_type, tool_name))

    tasks = {task for task, _ in hits}
    if len(tasks) != 1:
        return None, None
    tools = {tool for _, tool in hits}
    return hits[0][0], (tools.pop() if len(tools) == 1 else None)


def _pattern_specificity(pattern: str, tool_name: str = "") -> int:
    """3 = an exact filename naming a tool, 2 = a glob naming one, 1 or 0 = too generic to trust.

    Being exact is not the same as being distinctive, and the generic-token test used to be applied
    only to globs -- so ``proportions.csv`` took the top score, outranking every glob that does name
    a tool. Three profiles claim that one exact name (``run_cell2location``, ``run_bulk2space``,
    ``spacexr_rctd_deconvolution``); ``result.csv``, ``AI.csv`` and ``sp.h5ad`` are the registry's
    other exact-but-generic patterns. A match on any of them says a deconvolution tool wrote the
    file, which the *data* says better, and names a tool, which nothing here knows: the recorded
    spacexr output was attributed to ``run_cell2location`` on this route. Exactness still counts --
    a distinctive exact name outranks a distinctive glob -- it just no longer substitutes for
    distinctiveness.

    The price is that ``AI.csv`` and ``sp.h5ad`` no longer name svgbit and cell2location either.
    Both still detect from content, with a reason that quotes what was read; svgbit additionally
    picks up an ambiguity note, because its gene x gene ``Di.csv`` is non-negative and clears
    ``looks_like_proportions``. That note is the content route's real uncertainty, previously hidden
    behind an attribution this function had not earned.

    **Words, not filenames.** The split used to be on glob wildcards only, so a *multi-word* generic
    name arrived as a single token and sailed past a list that holds the words one at a time:
    ``predicted_genes.json`` scored 3 on ``predicted_genes``, though both ``predicted`` and
    ``genes`` are listed. It is squidpy's, and the recorded SVCA run writes one -- so a real SVCA
    directory was attributed to ``squidpy_spatial_autocorr``. That cost nothing while the tool name
    was discarded; it costs a wrong claim now that the engine keeps it and L2 checks the output
    against the named tool's published slot. Separators are now split on too.

    **``tool_name`` rescues the tool's own short name.** ``bsp`` is three characters, so the length
    floor -- there to stop ``AI`` and ``sp`` from counting -- drops it, and splitting on ``_`` is
    what exposes that: ``bsp_results.csv`` used to hide behind the 11-character token
    ``bsp_results``. A token that appears in the profile's own key is distinctive by construction,
    whatever its length, so passing the key keeps ``bsp_identify_svg`` attributable. Being generic
    still wins over being owned: ``run_scanpy_spatial_domain``'s ``*spatial_domains.h5ad`` is made
    of two listed words, and every domain tool in the registry writes a file that matches it.
    """
    if not pattern:
        return 0
    exact = "*" not in pattern and "?" not in pattern
    stem = pattern.rsplit(".", 1)[0]
    tokens = [t for t in _WORDS.split(stem) if t]
    owned = {t.lower() for t in _WORDS.split(tool_name or "") if t}
    distinctive = [t for t in tokens if t.lower() not in _GENERIC_TOKENS and (len(t) >= 4 or t.lower() in owned)]
    if not distinctive:
        return 1
    return 3 if exact else 2


def _from_content(files: list[Path], notes: list[str] | None = None) -> tuple[str | None, str]:
    """``(task_type, why)``; on no match, ``why`` may still explain why there was nothing to match.

    Staged *inputs* are dropped before anything is read: what a tool was handed does not say what it
    did, and the files it was handed are exactly the ones most easily mistaken for a result.
    Celloscope's ``celloscope_data/matB.csv`` is genes x cell types under real cell-type column
    names, so it reads as a textbook proportions matrix. Detection was the last consumer of the file
    list still looking at these -- every task runner and :func:`sources.find_coordinates` filters
    them first -- which is how a healthy SpatialDE run sitting beside a staged marker matrix was
    routed to the deconvolution runner, which then found no proportions and marked the run
    ``failed``: wrong task type, wrong verdict, and the tool's real SVG table never opened.
    """
    candidates = [f for f in files if not is_staged_input(f)]
    if not candidates:
        return None, (
            f"none of the {len(files)} file(s) is tool output -- they are all staged inputs or "
            "derived artifacts (a converted reference, a standardized copy, a prep directory)"
        )

    tabular = [f for f in candidates if is_tabular(f)]
    opened: list[Path] = []
    blank: list[Path] = []
    unsplit: dict[Path, str] = {}
    grouped: dict[Path, tuple[str, ...]] = {}
    unnamed: list[Path] = []
    picked: tuple[str, str, Path] | None = None
    picked_index = 0
    for index, path in enumerate(tabular):
        # The window is measured from the file that was picked, not from the start of the
        # directory. Measured from the start, the files scanned *before* the pick spend the budget:
        # a directory whose first recognisable table is the 31st file had nothing left, so the
        # survey stopped on file 32 and the competing result sitting right beside the pick was
        # never opened -- the one arrangement the warning exists for.
        if picked is not None and (notes is None or index > picked_index + _AMBIGUITY_SCAN_LIMIT):
            break
        verdict = _content_verdict(path, opened=opened, blank=blank, unsplit=unsplit, grouped=grouped, unnamed=unnamed)
        if verdict is None:
            continue
        if picked is None:
            picked, picked_index = (*verdict, path), index
        elif verdict[0] != picked[0]:
            notes.append(
                f"{AMBIGUOUS_OUTPUT_NOTE}this directory holds results for more than one task type -- "
                f"{picked[2].name} reads as {picked[0]} and {path.name} reads as {verdict[0]}. "
                # Not "was analysed". This function picks a task *type*; the handler for that type
                # then runs its own selection over the whole directory and often lands somewhere
                # else -- ``_pick_proportions`` prefers a table whose rows sum to 1, detect prefers
                # the first table in filename order. The note fires on 4 of the 92 recorded runs and
                # on cytospace the two disagree: the note named
                # ``cell_type_assignments_by_spot.csv`` while ``source_outputs`` recorded
                # ``fractional_abundances_by_spot.csv`` and ``assigned_locations.csv``. What detect
                # can say about the file it did not pick is about the run, not the file: one task
                # type ran, and it was not that one.
                f"The run was typed {picked[0]} from {picked[2].name} because it comes first in "
                f"filename order, which is a tie-break and not a judgement; no {verdict[0]} "
                f"analysis was run. Pass tool_name= or task_type= to say which one is the result."
            )
            break
    if picked is not None:
        return picked[0], picked[1]

    annotated = [f for f in candidates if f.suffix == ".h5ad"]
    for path in annotated:
        task, why = _from_anndata(path, opened=opened)
        if task:
            return task, why
    return None, _no_shape_reason(candidates, tabular + annotated, opened, blank, unsplit, grouped, unnamed)


def _has_no_rows(path: Path) -> bool:
    """True when the file parsed to column names with nothing under them.

    ``read_table`` returns ``None`` for this and for a file it could not parse at all -- ``if
    df.empty`` and the ``except`` clause reach the same ``return`` -- so the two arrive here
    indistinguishable, and a header-only CSV is described to the reader as *"could not be opened or
    parsed ... may be truncated or written in a format this reader does not handle"*. It parsed. It
    is empty, which is a tool that created its output file and wrote no results into it.

    Re-reading here rather than widening ``read_table``'s return keeps the contract its eleven other
    callers depend on -- ``None`` means "no table to work with" -- and costs one read of a file that
    is, by construction, one line long. Only reachable on the failure path, where nothing in the
    directory was recognised and the sole remaining product is a sentence.

    The first line has to look like a header before the file is called empty rather than unreadable,
    because "no rows" is not on its own the distinction. A run of bytes with no newline in it is one
    line to ``read_csv``, which makes it one column name and zero rows -- so a genuinely unparseable
    blob parses to exactly the shape a header-only CSV does, and was described as one. Column names
    are printable; ``\\x00\\x01\\x02`` is not.
    """
    import pandas as pd

    try:
        with path.open("rb") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    try:
        first = head.decode("utf-8").splitlines()[0] if head else ""
    except (UnicodeDecodeError, IndexError):
        return False
    if not first.strip() or any(ch != "\t" and ord(ch) < 32 for ch in first):
        return False
    try:
        frame = pd.read_csv(path, sep=tabular_sep(path), nrows=1)
    except Exception:
        return False
    return bool(frame.shape[1]) and frame.empty


def _holds_no_values(frame) -> bool:
    """True when a table parsed with rows in it and not one cell of them holds a value."""
    return bool(len(frame)) and not bool(frame.notna().to_numpy().any())


def _leftover_separator(frame) -> str | None:
    """The delimiter still sitting in a one-column frame's column name, if there is one.

    A frame with a single column is not on its own evidence of anything -- a list of gene symbols is
    one column because it holds one column. What distinguishes it from a table nothing split is
    where the delimiter ended up: ``spot;Bcell;Tcell`` survives *as the column name* because no pass
    consumed it, and ``gene`` does not.

    Returns the character rather than a bool so the caller can name it. The evidence is reported as
    an observation and the inference hedged, because the same shape is written on purpose:
    ``sp|P04637|P53_HUMAN`` is a real accession, and a one-column list of them is a real file.
    """
    if frame.shape[1] != 1:
        return None
    name = str(frame.columns[0])
    return next((sep for sep in _UNOFFERED_SEPARATORS if sep in name), None)


def _no_shape_reason(
    candidates: list[Path],
    examined: list[Path],
    opened: list[Path],
    blank: list[Path] | None = None,
    unsplit: dict[Path, str] | None = None,
    grouped: dict[Path, tuple[str, ...]] | None = None,
    unnamed: list[Path] | None = None,
) -> str:
    """Why the content route came back empty -- which is eight different facts, not one.

    "The data matched no known task shape" was said for all of them, and it is only true of the
    first. It tells a reader the detector looked and the output is unusual, so the next move is to
    name the task type. Said about a directory of ``.npy`` masks and ``.png`` overlays -- cellpose,
    deepcell and stacker in the recorded corpus -- it describes an inspection that never happened,
    and said about a file that failed to parse it hides a broken artifact behind a shrug.

    Three more were folded into those three. A table holding a header and no rows was called
    unreadable (see :func:`_has_no_rows`), and a table whose every cell is missing was called an
    unrecognised shape -- when an all-``NaN`` ``proportions.csv`` is shaped exactly like a
    proportions matrix and simply has nothing in it. The reader here is a language model choosing
    what to do next, and each of these is a different next move.

    The sixth is a file that never split (see :func:`_leftover_separator`). A semicolon-delimited
    proportions matrix -- what a spreadsheet writes under a European locale -- parses to one string
    column, so the shape tests ran against that and the sentence said both that the read went fine
    and that the content is *not a proportions matrix*. It is one. Neither half was true, and the
    next move it prompts is the wrong one: the file does not need a ``task_type=``, it needs its
    delimiter read. The replacement names the character that is still sitting in the column name and
    hedges the inference, because a one-column list of ``sp|P04637|P53_HUMAN`` accessions has the
    same shape and is exactly what it looks like.

    The seventh is a table over groups (see :func:`~.tables.group_key`): a cluster x donor
    cross-tabulation a run writes about its clustering. It is refused on purpose, and "not a
    proportions matrix ... pass task_type=" says neither why nor what to do -- it reads as an
    invitation to name the task the table was just refused as. The replacement says what a row is,
    which is the whole reason, and points at the per-spot output instead.

    The eighth is a table that nothing names (see :func:`~.tables.spot_evidence`): numbered rows,
    too few to be a section's spots -- the same cross-tabulation with its key written as the unnamed
    index, or a parameter sweep. "Not a proportions matrix" would be false -- it is one, non-negative
    and numeric -- so the reason says what is missing instead.
    """
    unparsed = [f for f in examined if f not in opened]
    empty = [f for f in unparsed if is_tabular(f) and _has_no_rows(f)]
    unread = [f for f in unparsed if f not in empty]
    valueless = blank or []
    never_split = list(unsplit or {})
    if opened:
        if len(valueless) == len(opened):
            return (
                f"{_name_list(opened)} parsed, and every cell in the rows read is missing -- the "
                f"table has the shape of a result and no values in it. Check that the tool finished "
                f"writing, or pass task_type= to say what this output is."
            )
        counts = []
        if unread:
            counts.append(f"{len(unread)} further file(s) could not be parsed")
        if empty:
            counts.append(f"{len(empty)} held column names and no rows")
        if len(never_split) == len(opened):
            return (
                f"{_name_list(never_split)} parsed into a single column whose name still contains "
                f"{_quoted_chars(unsplit)} -- the delimiter was probably not recognised, so "
                f"nothing here was matched against a task shape. Re-read it with that separator, or "
                f"pass task_type= to say what this output is."
            )
        if never_split:
            counts.append(
                f"{len(never_split)} parsed into a single column and may not have been split on the right delimiter"
            )
        read_ok = [f for f in opened if f not in never_split]
        summaries = [f for f in read_ok if f in (grouped or {})]
        few = [f for f in read_ok if f in (unnamed or ()) and f not in summaries]
        declined = []
        if summaries:
            declined.append(
                (
                    summaries,
                    f"one row per {_quoted_keys(grouped, summaries)}, a summary over groups rather than a "
                    "result per spot or per gene",
                )
            )
        if few:
            declined.append(
                (
                    few,
                    f"rows numbered rather than named, and fewer than {MIN_UNNAMED_SPOTS} of them: too few to "
                    "take for a section's spots on the count alone",
                )
            )
        rest = [f for f in read_ok if f not in summaries and f not in few]
        if declined and not rest:
            also = f"; {' and '.join(counts)}" if counts else ""
            body = (
                declined[0][1] if len(declined) == 1 else " and ".join(f"{_name_list(f)} held {w}" for f, w in declined)
            )
            return (
                f"{_name_list(summaries + few)} {READABLE_BUT_UNRECOGNIZED} -- {body}{also}. Point "
                f"post-analysis at the per-spot output if there is one."
            )
        counts.extend(f"{_name_list(f)} held {w}" for f, w in declined)
        read_ok = rest
        also = f"; {' and '.join(counts)}" if counts else ""
        return (
            f"{_name_list(read_ok)} {READABLE_BUT_UNRECOGNIZED} -- not a proportions "
            f"matrix, not one label per spot, not a ranked gene table{also}. Pass task_type= to say "
            f"what this output is."
        )
    if empty and not unread:
        return (
            f"{_name_list(empty)} parsed, but held column names and no rows under them -- there was "
            f"nothing to recognise. Check that the tool finished writing, or pass task_type= to say "
            f"what this output is."
        )
    if unread:
        also = f" {_name_list(empty)} parsed but held no rows." if empty else ""
        return (
            f"{_name_list(unread)} could not be opened or parsed, so nothing here was read. The "
            f"output may be truncated or written in a format this reader does not handle.{also}"
        )
    kinds = sorted({f.suffix.lower() or "<no suffix>" for f in candidates})
    return (
        f"none of the {len(candidates)} file(s) here is a table or an h5ad, so nothing was opened "
        f"({', '.join(kinds)}). Pass task_type= to say what this output is."
    )


def _quoted_keys(grouped: dict[Path, tuple[str, ...]], paths: list[Path]) -> str:
    """``"value of 'leiden'"``, ``"value of 'cluster' or 'leiden'"``, ``"'ident' x 'time_str' pair"``.

    The row keys of ``paths``, deduplicated. "or", not "and": each file has one key, and two files
    keyed differently are two tables over groups, not one table keyed by both. A pair is one key --
    see :func:`~.tables.group_key` -- so it is named as a pair, after the single columns.
    """
    keys = sorted({grouped[p] for p in paths}, key=lambda k: (len(k), k))
    singles = [f"{k[0]!r}" for k in keys if len(k) == 1]
    pairs = [" x ".join(f"{c!r}" for c in k) + " pair" for k in keys if len(k) > 1]
    said = [f"value of {_or_list(singles)}"] if singles else []
    said += [_or_list(pairs)] if pairs else []
    return ", or per ".join(said)


def _or_list(items: list[str]) -> str:
    """``"a"``, ``"a or b"``, ``"a, b or c"``."""
    return " or ".join(items) if len(items) < 3 else f"{', '.join(items[:-1])} or {items[-1]}"


def _quoted_chars(unsplit: dict[Path, str] | None) -> str:
    """``"';'"`` or ``"';' and '|'"`` -- the delimiters found, deduplicated, in a stable order.

    Taken from what :func:`_leftover_separator` actually saw rather than re-derived from the file,
    so the character the sentence names cannot drift from the one the detection was based on.
    """
    seen = sorted(set((unsplit or {}).values()))
    quoted = [f"{char!r}" for char in seen]
    return " and ".join(quoted) if len(quoted) < 3 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _name_list(paths: list[Path], limit: int = 4) -> str:
    """``'a.csv, b.csv and 3 more'`` -- a directory can hold hundreds and the reason is one line."""
    names = [p.name for p in paths[:limit]]
    tail = f" and {len(paths) - limit} more" if len(paths) > limit else ""
    return f"{', '.join(names)}{tail}"


def _content_verdict(
    path: Path,
    *,
    opened: list[Path] | None = None,
    blank: list[Path] | None = None,
    unsplit: dict[Path, str] | None = None,
    grouped: dict[Path, tuple[str, ...]] | None = None,
    unnamed: list[Path] | None = None,
) -> tuple[str, str] | None:
    """``(task_type, why)`` for one table, or ``None``. The order of these tests is the decision.

    ``opened`` collects the files that parsed, so the caller can tell "read it and recognised
    nothing" from "never got it open" -- a distinction ``None`` on its own throws away. ``blank``
    collects the subset of those whose every cell is missing, which is a third thing again and was
    being reported as the first. ``unsplit`` collects a fourth: the ones that parsed into a single
    column whose name still holds a delimiter, which means the shape tests below ran against one
    string per row and their verdict is about that, not about the table in the file.

    All three are recorded and none short-circuits: the ladder below runs on an all-``NaN`` frame,
    and on a one-column frame, exactly as it did -- so whatever it recognises today it recognises
    still. The lists are read only by :func:`_no_shape_reason`, on the path where nothing was
    recognised at all.

    ``grouped`` is the exception, and it is one because it records a refusal rather than a
    circumstance: a table over groups is declined by the ladder (see the comment where it is
    tested), and this maps it to the column that made its rows groups, so the reason can say so.
    ``unnamed`` is the same refusal for a table with no key to name: see the comment at the
    proportions test.
    """
    frame = read_table(path, nrows=_CONTENT_ROWS)
    if frame is None:
        return None
    if opened is not None:
        opened.append(path)
    if blank is not None and _holds_no_values(frame):
        blank.append(path)
    leftover = _leftover_separator(frame) if unsplit is not None else None
    if leftover is not None:
        unsplit[path] = leftover
    if looks_like_marker_table(frame):
        return None
    # Before the proportions test, not after. ``looks_like_proportions`` is permissive by design --
    # any non-negative numeric table whose column names it does not recognise qualifies -- so a
    # ligand-receptor score matrix cleared it and was reported as a cell-type composition, naming
    # signalling pairs (``r-CCL19-CCR7``) as the tissue's cell types. Column names are the evidence
    # that separates them, and they are only evidence if they are consulted first: no cell type is
    # ever called ``receptor``. Measured over the 92 smoke-output fixtures, none of the 114 tables
    # that pass ``looks_like_proportions`` in a real deconvolution directory matches this pattern.
    if any(_COMMUNICATION_COLUMN_RE.search(str(c)) for c in frame.columns):
        return "cell_communication", f"{path.name} has ligand/receptor-style columns"
    # Also before the proportions test, and for the same reason. ``alignment`` is a contract task
    # type with a working handler and a sentence written for it, but the string never appeared in
    # this module and no registry profile carries it, so an alignment run could only ever be typed
    # wrongly. Both recorded ones are non-negative numeric tables under unrecognised column names --
    # ``looks_like_proportions``' definition of a composition matrix -- so gpsa's two coordinate axes
    # were published as this tissue's two cell types, ``aligned_1`` the more abundant, under a
    # heatmap correlating the x axis against the y axis; slat's cell-to-cell correspondence table
    # did the same with ``slice1_idx`` and ``slice2_idx``. Neither file measures the tissue at all.
    # Two matching columns are required, and that floor is what keeps this narrow: 2 of the corpus'
    # 286 non-staged tables clear it, and no other name among 19,625 recorded ones matches either
    # pattern even once.
    aligned = [str(c) for c in frame.columns if _ALIGNMENT_COLUMN_RE.search(str(c)) or _SECTION_INDEX_RE.search(str(c))]
    if len(aligned) >= _MIN_ALIGNMENT_COLUMNS:
        return "alignment", (
            f"{path.name} names section-alignment columns ({', '.join(aligned)}) -- where spots were "
            "moved to, or which matched which, rather than anything measured in the tissue"
        )
    # A table whose rows are clusters is a summary over groups, and every rung below needs rows that
    # are something else: spots for a composition or a label table, genes for a ranking. Refusing it
    # in ``looks_like_proportions`` alone is not enough, because the ladder then keeps going: the
    # recorded ``cluster_sample_fractions.csv`` (``leiden`` x eight donors, plus a
    # ``dominant_sample`` column of donor IDs) went from "a non-negative spot x cell-type matrix" to
    # "ranks gene symbols by 'D01361_NG02955'", because a donor ID is spelled like a gene. After the
    # two column-name tests on purpose: per-cluster ligand-receptor scores are still a communication
    # result, and their columns say so before the rows are asked about.
    key = group_key(frame)
    if key is not None:
        if grouped is not None:
            grouped[path] = key
        return None
    if looks_like_proportions(frame):
        # A matrix is taken for one row per spot only on some evidence that its rows are spots. A
        # cluster x donor table written with the clusters as its unnamed index has no key column for
        # the test above, and its 11 rows read back numbered ``0..10`` -- the only thing it shows is
        # a non-negative matrix, and that is what every table over groups also is. Declined, not
        # passed down: every rung below needs per-spot or per-gene rows too, and a table with no
        # text column naming each row has neither.
        if spot_evidence(frame) is not None:
            return "deconvolution", f"{path.name} is a non-negative spot x cell-type matrix"
        if unnamed is not None:
            unnamed.append(path)
        return None
    # Before the gene-table test, not after: a narrow table with a `domain` column and one
    # numeric value per spot is a clustering result, and its barcode index reads as a list of
    # gene symbols to any name-based heuristic.
    if _looks_like_label_table(frame):
        return "spatial_clustering", f"{path.name} assigns one low-cardinality label per spot"
    # Before the gene-table test, and after both tests above so a real matrix and a real domain
    # assignment are still decided by their own shape. A deconvolution runner's mean-composition
    # summary has one column, which ``looks_like_proportions`` cannot accept, so the gene branch took
    # it: 44 cell types were published as this tissue's spatially variable genes, ranked by
    # ``mean_proportion``. The reason says which of the two files this is, because "deconvolution"
    # alone sends a reader looking for one row per spot in a file that has one row per cell type.
    is_summary, total, n_values = looks_like_composition_summary(frame, row_cap=_CONTENT_ROWS)
    if is_summary:
        return "deconvolution", (
            f"{path.name} is a per-cell-type composition summary -- {n_values} values summing to "
            f"{total:.3f}, one per cell type rather than one per spot"
        )
    is_genes, _, score = looks_like_gene_table(frame)
    if is_genes and score:
        return "svg_detection", f"{path.name} ranks gene symbols by {score!r}"
    return None


def _from_anndata(path: Path, *, opened: list[Path] | None = None) -> tuple[str | None, str]:
    """Only ``obs``/``obsm`` metadata is read, so the file is opened backed and closed on exit."""
    from .sources import open_anndata

    with open_anndata(path, backed=True) as adata:
        if adata is None:
            return None, ""
        if opened is not None:
            opened.append(path)
        try:
            obs_columns = [str(c) for c in adata.obs.columns]
            obsm_keys = [str(k) for k in adata.obsm]
            for column in obs_columns:
                if any(key in column.lower() for key in _PSEUDOTIME_KEYS):
                    return "trajectory", f"{path.name} has obs[{column!r}]"
            for key in obsm_keys:
                if any(word in key.lower() for word in ("abundance", "proportion", "deconv", "cell_type_prob")):
                    return "deconvolution", f"{path.name} has obsm[{key!r}]"
            # Alignment, BEFORE clustering and for that exact reason. An aligned object routinely
            # carries a cell-type or cluster column -- it is the same cells, annotated -- so the
            # clustering branch below claims it first and the run is measured as a clustering. The
            # coordinate key is the subject of the file; a label column beside it is not.
            #
            # The module docstring used to say alignment was not content-detectable, which was true
            # while the only evidence was a CSV's column names. An obsm key is stronger evidence
            # than a filename, and `_MIN_ALIGNMENT_COLUMNS` is deliberately NOT applied here: that
            # floor exists so a two-column CSV is not read as an alignment by accident, and one
            # obsm key IS the thing the file is about.
            if "spatial_3d" in getattr(adata, "uns", {}):
                return "alignment", f"{path.name} has uns['spatial_3d'], the 3D coordinate provenance block"
            from spatialomicsgym.spatial3d.adapters import ALIGNED_OBSM_KEYS

            for key in obsm_keys:
                if key == "spatial":
                    continue
                if key in ALIGNED_OBSM_KEYS or _ALIGNMENT_COLUMN_RE.search(key):
                    return "alignment", f"{path.name} has obsm[{key!r}]"
            for column in obs_columns:
                if any(key in column.lower() for key in _CLUSTER_OBS_KEYS) and _is_label_series(adata.obs[column]):
                    return "spatial_clustering", f"{path.name} has obs[{column!r}]"
        except Exception:
            return None, ""
    return None, ""


def _looks_like_label_table(frame) -> bool:
    """One label column per spot: what a clustering tool writes when it writes a CSV."""
    if frame.shape[1] > 4 or frame.shape[0] < 5:
        return False
    for column in frame.columns:
        if _is_label_series(frame[column]) and any(key in str(column).lower() for key in _CLUSTER_OBS_KEYS):
            return True
    return False


def _is_label_series(series) -> bool:
    try:
        distinct = int(series.nunique(dropna=True))
    except Exception:
        return False
    if distinct < 1 or distinct > _MAX_DOMAIN_CARDINALITY:
        return False
    kind = getattr(getattr(series, "dtype", None), "kind", "")
    if str(getattr(series, "dtype", "")) == "category" or kind in ("O", "b", "i", "u"):
        return distinct <= max(2, len(series) // 2)
    return False


def _as_list(value) -> list[str]:
    """``authoritative_output`` is ``str | list[str]`` -- both spellings are live in the registry."""
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)
