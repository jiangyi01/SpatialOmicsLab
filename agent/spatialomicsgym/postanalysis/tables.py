"""Reading a tool's tabular output, and the guards that decide whether to believe it.

Every predicate here exists because a recorded run got it wrong:

* :func:`orient_proportions` -- a correct 6142 x 7 bulk2space matrix was stored as 7 x 6142 because
  its spots were named ``spot_0, spot_1, ...`` and "short, no dash, not all digits" is
  indistinguishable from a cell-type name. The numbers do not have that ambiguity: deconvolution
  proportions sum to 1 across cell types, so the row sums are asked first and the labels last.
* :func:`looks_like_marker_table` -- ``marker_genes.csv`` (gene_name / rank_in_cell_type /
  cell_type) was once read as a (1500, 2) proportions matrix whose "cell types" were the two
  bookkeeping columns.
* :func:`topic_like_columns` -- STdeconvolve's theta has columns ``1 2 3``; STRIDE writes an LDA
  topic x spot matrix next to its real result. Latent topics are not cell types.
* :func:`is_signal_free` -- a constant score column gives recall 1.0 by construction. Shape and
  plausible names prove nothing; the values have to vary.
* :func:`group_key` -- ``cluster_sample_crosstab.csv`` (``leiden`` x eight donors) was
  published as a deconvolution whose most abundant cell type was a donor. Its rows are clusters.
* :func:`spot_evidence` -- the same table written with its clusters as the unnamed index has no key
  to find, and nothing in it says its 11 numbered rows are spots.

Where ``benchmarking`` already answers one of these questions -- the separator sniffer and the SVG
column classifier -- it is imported rather than copied, so the two layers cannot drift apart. Where
it answers one *differently* on purpose (:func:`orient_proportions`, :func:`topic_like_columns`),
the docstring says so and says why the other one must not be changed.

Pandas is imported inside the functions: ``import spatialomicsgym.postanalysis`` must stay cheap.
The ``benchmarking`` imports at the bottom are stdlib-only at module level, which the
``test_resolving_the_entry_point_is_also_free`` test pins.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

#: Extensions worth trying to parse as a table, ignoring any compression on top of them.
TABULAR_SUFFIXES = frozenset({".csv", ".tsv", ".txt"})

#: Compression a table may arrive under. The *vocabulary* half of the pair whose other half is
#: ``file_io._TEXT_OPENERS``, which maps these same names to the decompressor that opens them.
#: Spelled twice on purpose -- this module is stdlib-only at import and ``file_io`` pulls in
#: ``requests`` and ``tqdm``, so importing it here to read one tuple would cost every consumer of
#: ``tables`` those two -- and pinned by a test that fails if the two spellings drift apart.
COMPRESSION_SUFFIXES: tuple[str, ...] = (".gz", ".bz2", ".xz")

#: Every filename ending that names a table, compressed or not: ``.csv``, ``.csv.gz``, and so on.
#: Callers matching on the whole name (``str.endswith``) want this; callers holding a ``Path`` want
#: :func:`is_tabular`, which is the same rule stated as a question about one file.
TABULAR_SUFFIXES_WITH_COMPRESSION: tuple[str, ...] = tuple(
    sorted(TABULAR_SUFFIXES) + [t + c for t in sorted(TABULAR_SUFFIXES) for c in COMPRESSION_SUFFIXES]
)


def is_tabular(path) -> bool:
    """True when *path* names a delimited table, gzipped or not.

    Ask this rather than ``path.suffix.lower() in TABULAR_SUFFIXES``, which is what every picker in
    this layer used to do and which answers ``.gz`` for ``result.csv.gz``. A tool that writes its
    output compressed then falls out of ``autorun``'s nomination list, ``sources``' tabular list and
    all four task runners at once, so the run gets no post-analysis at all and nothing says so.
    xfuse is the recorded case: its only real output is a 500 x 4
    ``ST-metagene-log2fold.csv.gz`` and every other file it writes is a log, a session or a config.

    The compression has to be stripped rather than accepted, because ``.gz`` alone does not mean
    table -- ``stacker`` writes ``stacker_warped.nii.gz``, a NIfTI volume. Keying on the last suffix
    would hand a binary image to ``pd.read_csv``, which is the failure ``_ANALYSABLE_SUFFIXES`` was
    narrowed to prevent.
    """
    from spatialomicsgym.utils.file_io import uncompressed_suffix

    return uncompressed_suffix(path) in TABULAR_SUFFIXES


#: Columns that carry the spot/cell identity rather than a measurement.
_ID_COLUMN_NAMES = ("spot", "spot_id", "barcode", "barcodes", "cell", "cell_id", "cellid", "index")

#: Column names that mark a table as a per-gene annotation, not a spot x cell-type matrix.
_MARKER_COLUMN_RE = re.compile(r"^(gene|gene_name|gene_id|feature|rank|rank_in_\w+|cell_type|celltype)$", re.I)

#: A column name that names a latent factor rather than an annotated cell type.
# ``celltype``/``cell_type`` is in here because :func:`_from_h5ad` invents ``celltype_0 ... _n`` for
# an ``obsm`` array that carries no column labels, and the guard below was then asked whether those
# names were real annotations. A check that certifies the placeholder written two lines earlier is
# checking nothing. A numbered cell type is not an annotated one whoever wrote it.
_TOPIC_COLUMN_RE = re.compile(
    r"^(topic|factor|arch|archetype|component|comp|cluster|k|celltype|cell_type)[_\-\s]?\d+$", re.I
)

#: One label split around its first run of digits: ``aligned_0`` -> ``('aligned_', 0, '')``,
#: ``slice1_idx`` -> ``('slice', 1, '_idx')``, ``CD14+ Monocytes`` -> ``('CD', 14, '+ Monocytes')``.
#: The number is allowed to sit anywhere, because a generated axis puts it wherever it likes.
_ENUMERATED_LABEL_RE = re.compile(r"^(?P<before>.*?)(?P<number>\d+)(?P<after>.*)$")

# Columns that ride along in the same CSV as a composition without being cell types: spot
# coordinates and per-spot QC. ``Slice, spotName, x, y, CD14+ Monocytes, ...`` is a recorded
# fixture, and keeping ``x``/``y`` made them the two dominant "cell types" at 50 % each while the
# five real ones sat at 0.002 %. Names are the only available signal -- a coordinate is numeric and
# non-negative exactly like a proportion. Anchored, so a cell type merely *containing* one of these
# words (``Cycling cells``, ``Row cells``) is untouched.
_NON_CELLTYPE_COLUMN_RE = re.compile(
    r"^(x|y|z|row|col|column|array_row|array_col|imagerow|imagecol"
    r"|pxl_row_in_fullres|pxl_col_in_fullres|pxl_row|pxl_col|x_coord|y_coord|xcoord|ycoord"
    r"|spatial[_\-]?[12xy]?|centroid_[xy]|center_[xy]"
    r"|n_counts|n_genes|total_counts|n_genes_by_counts|pct_counts_mt|ncount_\w+|nfeature_\w+"
    # A tally of the observations is the one thing a matrix indexed *by* observations cannot be
    # holding. ``n_genes`` and ``n_counts`` were here from the start and ``n_spots`` was not, which
    # is how ``spatial_domain,n_spots`` -- six rows, one per domain, written beside the ``.h5ad``
    # that holds the domains -- was published as a spot x cell-type matrix whose top cell type is
    # ``n_spots``, at ``status: ok``, on a clustering run.
    r"|n_spots|n_cells|n_obs"
    r"|in_tissue|fov|slice|z_slice|cluster|leiden|louvain|domain|region"
    # And the same words with the tool's name in front of them. Anchoring is what keeps a cell type
    # merely containing one of these words safe, and the cost of it is that ``domain`` is dropped
    # while ``stagate_domain``, ``IRIS_domain``, ``DeepST_refine_domain``, ``prost_domain``,
    # ``miso_cluster``, ``sedr_cluster`` and ``spatial_domain`` -- every namespaced label column in
    # the recorded corpus, and not one of them a cell type -- are kept. ``region`` is deliberately
    # not in this half: ``CA1_region`` is a place in a brain and a name a real annotation uses,
    # nothing in the corpus needed it, and a bare ``region`` is still caught by the line above.
    r"|\w+_(cluster|leiden|louvain|domain)s?"
    # The identity names too, as the net under :func:`_set_identifier_index`. That function
    # consumes at most one of them -- the first -- and only from a frame whose first column is
    # not already an unnamed index, so a long-form ``,spot,cell,value`` table keeps both, and a
    # duplicated barcode keeps its column. Whatever it leaves behind is still not a cell type and
    # still not a statistic: ``index`` was published as the dominant cell type at 0.97 from a
    # ``reset_index().to_csv(index=False)`` table whose real rows sum to 1, and the same column
    # was returned by :func:`pick_score_column`, which ranks ``top_genes.csv`` by row number.
    r"|index|spot|spot_id|spotname|spot_name|barcode|barcodes|cell|cell_id|cellid)$",
    re.I,
)
_NUMERIC_NAME_RE = re.compile(r"^\d+(\.\d+)*$")

_GENE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._\-]{1,24}$")

#: A 10x barcode (``TTCGCACTGTACGACA-1``) satisfies :data:`_GENE_NAME_RE` exactly -- it starts with a
#: letter and contains only letters, digits and a dash. Every barcode-indexed table with a numeric
#: column would otherwise read as a gene ranking. The inspector carries the same guard.
_BARCODE_RE = re.compile(r"^[ACGTN]{8,}(-\d+)?$", re.IGNORECASE)

#: A slice/sample prefix in front of a barcode (``s0_CATCAAACTGGCGCCC-1``) is what a multi-slice
#: tool writes, and it defeats the anchored :data:`_BARCODE_RE`. Stripped only inside
#: :func:`looks_like_barcodes`; ``_BARCODE_RE`` itself mirrors the inspector's guard and is left
#: spelled exactly as the inspector spells it.
_BARCODE_PREFIX_RE = re.compile(r"^[A-Za-z0-9]{1,8}[_.\-]")


def looks_like_barcodes(values) -> bool:
    """True when most of ``values`` are spot barcodes rather than names of anything.

    A majority test, not an any-test: one cell type whose name happens to spell out in ``ACGTN``
    must not sink a real result. Used on the *cell-type axis* of a deconvolution matrix, where a
    barcode proves the matrix is not a composition -- but only once the orientation is settled,
    because a composition written cell_types x spots legitimately has barcodes across its columns
    on disk and ``orient_proportions`` exists to flip exactly that.
    """
    sample = [str(v).strip() for v in list(values)[:50]]
    if not sample:
        return False
    hits = sum(1 for s in sample if _BARCODE_RE.match(s) or _BARCODE_RE.match(_BARCODE_PREFIX_RE.sub("", s, count=1)))
    return hits / len(sample) > 0.5


def _tabular_sep_fallback(path: Path) -> str:
    """The benchmarking sniffer's behaviour for an install without ``benchmarking``.

    Same shared implementation and same trusted extensions, so the fallback cannot drift away
    from the thing it stands in for -- which is the whole reason it is a delegation and not a
    copy of the rule.
    """
    from spatialomicsgym.utils.file_io import BENCHMARK_TRUSTED_EXTENSIONS, sniff_tabular_sep

    return sniff_tabular_sep(path, trust_extension=BENCHMARK_TRUSTED_EXTENSIONS)


try:  # The inspector's sniffer, not a copy of it. The fallback covers a benchmarking-less install.
    from spatialomicsgym.benchmarking.output_inspector import _tabular_sep as _inspector_tabular_sep
except Exception:
    _inspector_tabular_sep = _tabular_sep_fallback


def tabular_sep(path: Path) -> str:
    """Field separator for ``path``. ``.csv``/``.tsv`` are fixed; anything else is sniffed.

    Delegates to the inspector so both layers parse a file the same way: STRIDE writes a
    tab-separated ``.txt`` and SpotGF a tab-separated ``_scores.txt``, and reading either with
    commas yields one useless column.
    """
    return _inspector_tabular_sep(path)


def _header_is_data(columns) -> bool:
    """True when ``pd.read_csv`` turned a matrix's first row of values into its column names.

    Some tools write a bare matrix -- no header line, no index column. celloscope's declared source
    output is one: ``result_h.csv`` is 10 x 1000 floats, so the default read returns 9 x 1000, drops
    a whole deconvolution component and shifts the label of every component that survives. Nothing
    downstream can catch it, because the truncated frame is still rectangular, still numeric and
    still sums like a composition.

    Guessing wrong the other way is worse than the bug, so the test is deliberately narrow. Line 1
    of ``novosparc_coupling.csv`` and of spaotsc's ``transport_plan.csv`` is exactly ``0,1,...,199``,
    a real positional header, and integer gene or cluster IDs are equally real labels. What
    separates them across all 535 tabular files in the corpus is a fractional part: the headerless
    matrices carry 17 to 20 decimal places, machine-precision output no tool writes as a label,
    while every genuine numeric header is integral.

    A repeated label counts too, and for a stronger reason than precision: a header physically
    cannot name two columns the same, so when pandas de-duplicates ``55,54,48,54`` into
    ``55,54,48,54.1`` the suffix it appends is itself proof that line 1 was not a header.

    The residue is an all-integer matrix of distinct values, which no evidence in the file can tell
    from a header of integer labels. That one keeps the reading it has always had.
    """
    names = [str(c) for c in columns]
    if len(names) < 2:
        return False
    try:
        values = [float(n) for n in names]
    except ValueError:
        return False
    return any(math.isfinite(v) and v != int(v) for v in values)


def duplicate_header_fields(path: Path) -> list[str]:
    """Names the header line repeats, in first-seen order -- the ones ``read_csv`` renamed.

    A repeated column name is a shape pandas cannot represent, so it appends a counter:
    ``spot,Bcell,Tcell,Bcell`` parses to ``Bcell`` and ``Bcell.1``. By the time a caller holds the
    frame there is nothing left to notice -- ``Bcell.1`` is a well-formed label that no reference
    ever contained, and the two columns each hold *part* of that cell type's share, so a Bcell at
    0.75 of the tissue is published twice at 0.35 and 0.40. Only the header line knows.

    It is read twice to find out, once as names and once as data, both through pandas' own
    tokenizer, so a quoted field containing the separator (``"CD4, T cells"``) is not miscounted the
    way splitting the line on commas would.

    Files :func:`_header_is_data` claims are exempt. Their line 1 is a row of *values*,
    :func:`read_table` re-reads them with ``header=None``, and a repeated value there is not a
    repeated name -- it is, per that function, the evidence that line 1 was never a header at all.
    Three of the five files in the recorded corpus with a repeated first line are exactly this.
    """
    import pandas as pd

    path = Path(path)
    sep = tabular_sep(path)
    try:
        mangled = pd.read_csv(path, sep=sep, nrows=0).columns
        header_row = pd.read_csv(path, sep=sep, nrows=1, header=None)
    except Exception:
        return []
    if _header_is_data(mangled) or header_row.empty or header_row.shape[1] != len(mangled):
        return []
    names = [str(v) for v in header_row.iloc[0].tolist()]
    counts = Counter(names)
    return [n for n in dict.fromkeys(names) if counts[n] > 1]


def read_table(path: Path, nrows: int | None = None):
    """Read ``path`` as a DataFrame with the spot/gene identifier on the index, or ``None``.

    Three shapes are normalised here so no caller has to think about them again:
      * a file with no header line at all, whose first row of data pandas would eat as the column
        names (see ``_header_is_data``);
      * a leading unnamed index column (pandas calls it ``Unnamed: 0``) becomes the index;
      * an identifier column anywhere in the frame (STdeconvolve's theta puts ``spot`` *last*)
        becomes the index.
    """
    import pandas as pd

    path = Path(path)
    sep = tabular_sep(path)
    try:
        df = pd.read_csv(path, sep=sep, nrows=nrows)
    except Exception:
        return None
    if df.empty or df.shape[1] == 0:
        return None
    if _header_is_data(df.columns):
        try:
            reread = pd.read_csv(path, sep=sep, nrows=nrows, header=None)
        except Exception:
            return _set_identifier_index(df)
        if not reread.empty and reread.shape[1]:
            df = reread
    return _set_identifier_index(df)


def _set_identifier_index(df):
    """Move the spot/cell identity out of the data and into the index.

    The dtype test used to be ``== object``, which reads "identifiers are text". Barcodes are, but
    ``pd.DataFrame(...).reset_index().to_csv(index=False)`` is an ordinary way for a tool to write
    its result and it emits ``index,Astro,Neuron,Microglia`` with ``index`` holding ``0..n-1`` as
    int64. That column then went the whole way through as data: it dominated
    ``mean_composition.csv`` at 0.97 against Astro's real 0.50, and it inflated every row sum to
    ~100 so the run also warned that a table summing to exactly 1 held abundances.

    So an integer column counts too -- but only when its values are unique, because that is what
    separates an identifier from a measurement that happens to be whole. A float never counts: a
    proportion is a float, and ``cell`` is a name a real column can carry.
    """
    import pandas as pd

    first = str(df.columns[0])
    if first.startswith("Unnamed: 0") or first == "":
        df = df.set_index(df.columns[0])
        df.index.name = None
        return df
    for col in df.columns:
        if str(col).strip().lower() not in _ID_COLUMN_NAMES:
            continue
        values = df[col]
        if values.dtype == object or (pd.api.types.is_integer_dtype(values) and values.is_unique):
            out = df.set_index(col)
            out.index.name = None
            return out
    return df


def numeric_frame(df):
    """The numeric columns of ``df``, with all-NaN columns dropped."""
    numeric = df.select_dtypes(include="number")
    return numeric.dropna(axis=1, how="all")


def celltype_columns(df):
    """``df`` without the columns that are demonstrably not cell types.

    A deconvolution tool writing the spot coordinates it was given into the same CSV as the
    composition it produced is ordinary. Reading them as cell types is not: it puts ``y`` at the top
    of ``mean_composition.csv`` at 50 %, and it inflates every row sum, which is what produced the
    warning "rows do not sum to 1 (median 1.08e+04); these are abundances" about a table whose real
    rows sum to exactly 1. See :data:`_NON_CELLTYPE_COLUMN_RE`.
    """
    keep = [c for c in df.columns if not _NON_CELLTYPE_COLUMN_RE.match(str(c).strip())]
    return df[keep] if len(keep) != len(df.columns) else df


# ---------------------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------------------


def looks_like_marker_table(df) -> bool:
    """True when ``df`` is a per-gene annotation table rather than a measurement matrix.

    Recognised by its column *names*, which is the only signal available: ``rank_in_cell_type`` is
    numeric and ``cell_type`` is a label, so a shape/dtype test accepts it happily.
    """
    names = [str(c) for c in df.columns]
    if not names:
        return False
    marker_like = sum(1 for n in names if _MARKER_COLUMN_RE.match(n.strip()))
    return marker_like >= 2 and marker_like / len(names) >= 0.5


def _is_enumerated_axis(names: list[str]) -> bool:
    """True when most of ``names`` are one shared stem plus a counter running from 0 or 1, ungapped.

    The word list this backs up can only recognise stems it has already met. The shape is what the
    stems have in common, and a cell-type panel does not have it: annotations are names, and a
    generated axis is a stem and a position.

    Both halves of the stem have to match, so ``slice1_idx``/``slice2_idx`` group together while
    ``Astrocyte_1``/``Oligodendrocyte_2`` do not. The counter has to *start* at 0 or 1 and run
    without gaps, which is what keeps ``CD4 T cells``/``CD8 T cells`` out -- those share the stem
    ``CD``/`` T cells`` exactly, and are told apart from a counter only by starting at 4.
    """
    groups: dict[tuple[str, str], set[int]] = {}
    for name in names:
        match = _ENUMERATED_LABEL_RE.match(name)
        if match:
            groups.setdefault((match["before"].lower(), match["after"].lower()), set()).add(int(match["number"]))
    if not groups:
        return False
    numbers = max(groups.values(), key=len)
    if len(numbers) < 2 or len(numbers) / len(names) <= 0.5:
        return False
    first = min(numbers)
    return first in (0, 1) and numbers == set(range(first, first + len(numbers)))


def topic_like_columns(columns) -> bool:
    """True when the column labels name latent factors (``1``, ``topic_3``, ``arch_0``).

    Same policy as the inspector's inline archetype rejection (``output_inspector`` ~line 1548),
    widened by ``cluster_<n>``/``comp_<n>``/``k<n>``. It is inline there, inside a scoring path
    that must not change, so it cannot be imported; this is a deliberate second implementation
    with a deliberately wider net, and it only ever produces a warning.

    The word list is not enough on its own, because it can only hold stems that have already been
    seen. Three tools in the recorded smoke-output corpus name their axis something else and walked
    through the guard into ``status: ok``: GPSA's alignment axis ``aligned_0``/``aligned_1``, SLAT's
    ``slice1_idx``/``slice2_idx`` and SpatialGlue's ``alpha_omics1``/``alpha_omics2`` were each
    published as the tissue's cell types, with a coordinate, an index and an attention weight named
    as the dominant one. :func:`_is_enumerated_axis` catches the shape they share; the word list
    stays because it catches what the shape cannot -- Starfysh writes ``arch_0, arch_1, arch_5,
    arch_6``, an enumeration with holes in it.

    Measured over the recorded corpus, the shape test adds exactly those three frames and no
    others: no genuine composition in the fixtures is flagged by it. It can only ever warn and
    degrade the run to ``partial`` -- the numbers are still published -- so a panel that really is
    named ``Layer1..Layer6`` costs a warning asking for confirmation, not a lost result.
    """
    names = [str(c).strip() for c in columns]
    if not names:
        return False
    bad = sum(1 for n in names if _NUMERIC_NAME_RE.match(n) or _TOPIC_COLUMN_RE.match(n))
    return bad / len(names) > 0.5 or _is_enumerated_axis(names)


def is_signal_free(values) -> bool:
    """True when the values carry no information: constant, all-zero, or all-NaN.

    This is the check behind two recorded incidents -- an all-zero STdGCN proportion table and an
    SVG score column with one distinct value, which scores recall 1.0 by construction.
    """
    import numpy as np

    arr = np.asarray(values, dtype="float64").ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return True
    return bool(np.nanmin(arr) == np.nanmax(arr))


#: More cell types than any reference in this repo's benchmark suite annotates. It is the *count*
#: that has to be implausible, not merely larger than the other axis: a per-region or per-domain
#: mean composition is a genuine composition with a handful of rows, and 9 regions x 20 cell types
#: is an ordinary table whose rows sum to 1 over its longer axis. Reading "longer axis" alone as
#: disqualifying transposed every one of those into ``cell_types x spots``, which is the exact
#: failure this constraint was added to prevent, in a different shape window. Above this line the
#: reading stops being merely unusual: Tangram's 4035 "cell types" were the section's spots.
MAX_PLAUSIBLE_CELL_TYPES = 100


def why_not_cell_types(n_named: int, n_other: int, labels=None) -> str | None:
    """Why an axis of ``n_named`` entries cannot be the cell types, or ``None`` if it can be.

    The one place the "which axis can be the cell types" premise is written down.
    :func:`orient_proportions`, ``review._deconvolution_checks`` and
    ``tasks/deconvolution._normalised_over_a_plausible_celltype_axis`` all decide the same question
    and had to agree; three copies of a premise is three chances for one of them to be relaxed
    alone. It returns the *reason* rather than a bool so the three can also say the same thing to
    the reader -- a note that names the wrong disqualifier is a wrong sentence about a right call.

    Two disqualifiers, and they cover different scales:

    * the labels are spot barcodes. Decisive at any size, and the only signal that works on a small
      section: a 60 x 8 matrix whose columns sum to 1 is Tangram's shape at ROI scale, and no count
      test can tell 60 cell types from 60 spots.
    * the count is implausible *and* it is the longer axis. Covers the axes whose labels are not
      barcodes at all -- a topic model's 500 genes, an ``obsm`` array whose names post-analysis
      invented -- and needs both halves: an axis longer than the other is not suspicious by itself
      (that reading transposed genuine per-region compositions), and the ceiling alone would let a
      4035 x 4036 matrix through.

    ``labels`` is optional because a caller may hold the counts and not the axis; passing them is
    strictly better. Note what this never does: conclude an axis *is* cell types from its labels.
    That is the inference that turned a correct 6142 x 7 bulk2space matrix upside down, because
    ``spot_0`` reads as a cell-type name. Only disqualification is safe here.
    """
    if labels is not None and looks_like_barcodes(labels):
        return "those labels are spot barcodes"
    if int(n_named) > int(n_other) and int(n_named) > MAX_PLAUSIBLE_CELL_TYPES:
        return f"{int(n_named)} is more cell types than a reference carries"
    return None


def implausible_as_cell_types(n_named: int, n_other: int, labels=None) -> bool:
    """:func:`why_not_cell_types` as a predicate, for callers that only branch on it."""
    return why_not_cell_types(n_named, n_other, labels) is not None


def orient_proportions(df) -> tuple[Any, bool, list[str]]:
    """Return ``(spots x cell_types, was_transposed, notes)``.

    Decided by the numbers first: whichever axis sums to ~1 is the cell-type axis, because that is
    what a proportion vector means. Only when neither axis sums to 1 (abundances, not fractions)
    does shape break the tie -- there are always more spots than cell types.

    The sum test is *constrained* by :func:`implausible_as_cell_types`, which it did not used to be.
    A sum test names one axis the cell types, so an answer naming thousands of them is not an
    answer -- it is the tool having normalised over the other thing. Tangram writes exactly that
    file: 4035 spots x 44 cell types, each **column** summing to 1 because Tangram normalises each
    cell type's mapping across spots. Read without the constraint, "the columns sum to 1"
    transposed a correct matrix into 44 x 4035, and the barcode guard in ``tasks/deconvolution.py``
    then failed the run on the spot barcodes the transpose had just moved into the columns.

    **This deliberately disagrees with** ``benchmarking/output_standardizer.py`` (~line 683), which
    solves the same problem but keeps a label-reading fallback (``_looks_like_celltype_names`` /
    ``_looks_like_barcodes``) after its sum test. That fallback is what turned a correct 6142 x 7
    bulk2space matrix into a 7 x 6142 one, because ``spot_0`` reads as a cell-type name. It is not
    fixed here and must not be: the standardizer's verdicts are baked into recorded benchmark
    scores, and changing it would move them. This function never touches a scored path -- it
    decides what post-analysis plots -- so it can afford to answer "ambiguous" instead of guessing.
    """
    import numpy as np

    notes: list[str] = []
    if df.shape[0] < 2 or df.shape[1] < 2:
        return df, False, notes

    n_rows, n_cols = df.shape[0], df.shape[1]

    def _sums_to_one(axis: int) -> bool:
        sums = np.asarray(df.sum(axis=axis), dtype="float64")
        sums = sums[np.isfinite(sums)]
        return bool(sums.size) and bool(np.isclose(np.median(sums), 1.0, atol=0.02))

    def _why_not(axis: int) -> str | None:
        # ``axis=1`` sums each row across the columns, so the *columns* are the ones being summed
        # over and the columns are what this answer would name the cell types; ``axis=0`` mirrors
        # it.
        if axis == 1:
            return why_not_cell_types(n_cols, n_rows, df.columns)
        return why_not_cell_types(n_rows, n_cols, df.index)

    rows_sum_to_one, cols_sum_to_one = _sums_to_one(axis=1), _sums_to_one(axis=0)
    why_not_rows, why_not_cols = _why_not(axis=1), _why_not(axis=0)
    rows_are_spots = rows_sum_to_one and why_not_rows is None
    cols_are_spots = cols_sum_to_one and why_not_cols is None

    # ``(what sums to 1, what that would make the cell types, how many of each, and why not)``.
    implausible = [
        (vector, n_celltypes, celltypes, n_spots, spots, reason)
        for summed, vector, n_celltypes, celltypes, n_spots, spots, reason in (
            (rows_sum_to_one and not rows_are_spots, "row", n_cols, "columns", n_rows, "rows", why_not_rows),
            (cols_sum_to_one and not cols_are_spots, "column", n_rows, "rows", n_cols, "columns", why_not_cols),
        )
        if summed
    ]
    decided_on_shape = not (rows_are_spots or cols_are_spots)
    for vector, n_celltypes, celltypes, n_spots, spots, reason in implausible:
        # What is established is that the sum is not evidence of orientation. *Why* it sums to 1 is
        # not, and naming one cause would be a diagnosis this has not earned: both recorded shapes
        # reach here -- Tangram's per-cell-type normalisation across spots, and a topic model's
        # topic x gene beta, whose rows are distributions over genes. Only the second is not a
        # composition at all, and it is the positional-label guard downstream, not this note, that
        # is able to tell them apart.
        notes.append(
            f"Each {vector} sums to 1, which would make the {n_celltypes} {celltypes} the cell "
            f"types over {n_spots} {spots} -- but {reason}, so the sum runs over something other "
            "than the cell types: a per-cell-type normalisation across spots (Tangram writes one), "
            "or an axis that is not cell types at all (a topic model's gene distribution). It is "
            "not a per-spot composition either way."
            + (" Orientation was decided on shape instead." if decided_on_shape else "")
        )

    if rows_are_spots and not cols_are_spots:
        return df, False, notes
    if cols_are_spots and not rows_are_spots:
        notes.append(
            f"Transposed {df.shape} -> {df.shape[::-1]}: the columns summed to 1 and there are "
            f"{n_cols} of them against {n_rows} rows, so the columns were the spots. Orientation "
            "decided by row sums, not by the labels."
        )
        return df.T, True, notes
    if rows_are_spots and cols_are_spots:
        # Square-ish and both normalised: nothing to prefer, leave it and say so.
        notes.append("Both axes sum to 1; orientation is ambiguous and was left as written.")
        return df, False, notes

    # No axis is a credible composition -- abundances, or a sum that runs over something else.
    # Shape is all that is left. The reason has to be the true one: saying "neither axis sums to 1"
    # about the matrix whose rows sum to 1.000, two notes below the note that said they do, is a
    # contradiction a reader has to resolve against the data to know which half to believe.
    if df.shape[1] > df.shape[0] * 2:
        why = (
            "no axis sums to 1 over a credible number of cell types"
            if (rows_sum_to_one or cols_sum_to_one)
            else "neither axis sums to 1"
        )
        notes.append(
            f"Transposed {df.shape} -> {df.shape[::-1]}: {why}, and a matrix this wide has its spots on the columns."
        )
        return df.T, True, notes
    return df, False, notes


def looks_like_proportions(df) -> bool:
    """True when ``df`` could be a spot x cell-type matrix, before orientation is fixed.

    Deliberately permissive about the *values* (abundances are not fractions, and an all-zero
    matrix is still a proportions matrix -- a broken one, which the signal-free check reports).
    Strict about what it is *not*: a marker table, a gene table, a coordinate table, a table over
    groups.
    """
    if looks_like_marker_table(df):
        return False
    numeric = celltype_columns(numeric_frame(df))
    if numeric.shape[1] < 2 or numeric.shape[0] < 2:
        return False
    if _is_coordinate_frame(numeric):
        return False
    # An SVG table clears every test above: 500 genes x ``{FSV, qval}`` is non-negative with two
    # numeric columns, and one was duly reported as a composition of 65.9 % qval and 34.1 % FSV.
    # Shape and sign cannot separate the two; the column names can, because no cell type is ever
    # named ``qval``. Checked on the names rather than the index because a proportions table indexed
    # ``spot_0 ... spot_n`` reads as a list of gene symbols to any name-based identifier heuristic.
    if all(column_kind(str(c)) != "unknown" for c in numeric.columns):
        return False
    # A table whose rows are groups is not a table whose rows are spots, whatever its columns are
    # called. See :func:`group_key`; checked before the gene-table test because it reads one or two
    # columns and that test reads several.
    if group_key(df) is not None:
        return False
    # The third exclusion the docstring promises, and the one that was never written. Placed here
    # rather than beside the marker-table test because everything above the group-key test reads
    # column metadata only; that test reads one column and this one several, and there is no reason
    # to pay for either on a frame already refused.
    #
    # Narrow on purpose. "Carries a column of gene-shaped strings" is true of genuine compositions
    # too: iris writes a constant ``Slice1`` column that matches the gene-symbol pattern, and a tool
    # is free to name its spots ``S001A``. Three conditions together say "the rows of this frame are
    # genes": the identifiers sit in a column, because an index of symbols is how a counts matrix is
    # shaped and orientation is settled downstream; the column is not one this layer already knows
    # names spots, slices or coordinates; and it names each row exactly once, because a ranking
    # lists a gene once where a per-spot table repeats it. Drop that last condition and a
    # deconvolution carrying one marker gene per spot is refused.
    #
    # Measured over the 92 recorded runs before landing: 141 calls reach here, 8 refuse, and all 8
    # are istar's per-gene contrast tables and xfuse's metagene table -- runs that published a gene
    # count under "Spots deconvolved" and a statistic column under "Most abundant cell type".
    is_gene_table, gene_key, _score = looks_like_gene_table(df)
    if is_gene_table and gene_key is not None and gene_key in df.columns:
        ids = df[gene_key]
        named_once = len(ids) > 1 and bool(ids.notna().all()) and ids.nunique() == len(ids)
        if named_once and not _NON_CELLTYPE_COLUMN_RE.match(str(gene_key).strip()):
            return False
    import numpy as np

    arr = numeric.to_numpy(dtype="float64", na_value=np.nan)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    return bool(finite.min() >= 0.0)


#: The words a column naming a row's group is spelled with: a cluster, a domain, a niche, Seurat's
#: identity class. Matched as whole words of the name (see :func:`_names_a_group`), so the tool, the
#: embedding, the resolution and an id may sit anywhere around one: the recorded keys include
#: ``cluster_fallback``, ``cluster_mbkm_harmony``, ``leiden_harmony_r04``, ``leiden_r05`` and
#: ``ident``, and scanpy's own spelling is ``leiden_res_0.50``. ``region`` is not here for the reason it
#: is not in the namespaced half of :data:`_NON_CELLTYPE_COLUMN_RE`: ``CA1_region`` is a name a real
#: annotation uses. Cell-type words are not here either, and must not be: a composition written cell
#: types x spots has one row per cell type, and :func:`orient_proportions` exists to turn exactly that
#: table round.
_GROUP_WORDS = frozenset(
    "cluster clusters leiden louvain kmeans mclust domain domains niche niches group groups ident idents".split()
)

#: Words that make a name about groups something other than the label of one, anywhere in it: a
#: quantity (``cluster_size`` beside a key, ``max_single_timepoint_fraction``) or a pointer from a row
#: that is something else (``best_cluster``, on a recorded table whose rows are lineages).
_NOT_A_LABEL_WORDS = frozenset(
    (
        "count counts total size sizes frac fraction fractions prop proportion proportions pct percent percentage "
        "ratio score scores mean median max min std sd var entropy purity prob probability dist distance ari nmi "
        "silhouette rank best dominant top majority nearest"
    ).split()
)

#: The same, only as the first word: ``n_clusters`` counts clusters, where ``cluster_number`` is one.
_NOT_A_LABEL_PREFIXES = frozenset({"n", "num", "number", "is", "has"})

#: The words of a column name: ``Cluster ID`` -> ``Cluster, ID``; ``kmeans_k12`` -> ``kmeans, k, 12``.
_NAME_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")

#: Below this, a label that differs on every row is what any two spots in two domains look like.
_MIN_GROUP_ROWS = 3

#: A column that makes a long table's rows genes within a group rather than groups: scanpy's
#: ``rank_genes_groups`` writes ``group, names, scores``, and models write ``cluster, rank, gene``.
_PER_GENE_COLUMN_RE = re.compile(r"^(gene|genes|gene_name|gene_names|gene_id|gene_symbol|symbol|names|rank)$", re.I)

#: A table that nothing names -- rows numbered, no text column naming each one, no barcodes across the
#: columns -- is taken for one row per spot only when an axis is at least this long. Of the recorded
#: tables that nothing names and that would otherwise read as a composition, the longest whose rows
#: are not spots has 40 (a long-form marker table; the sweeps run to 24) and the shortest whose rows are
#: spots has 60 (starfysh's ``anchor_spots.csv``); the one real composition among them, celloscope's
#: ``thetas_est.csv``, has 200 spots across its columns. A clustering can have hundreds of clusters --
#: two recorded tables have 451 and 459 rows, keyed -- so this catches a small table over groups written
#: without its key, not a large one: the key is the evidence, and this is what is left when there is none.
MIN_UNNAMED_SPOTS = 50


def _names_a_group(name: str) -> bool:
    """True when ``name`` is spelled as the label of a cluster, domain or niche, not a count of them."""
    words = [w.lower() for w in _NAME_WORD_RE.findall(str(name))]
    if not any(w in _GROUP_WORDS for w in words) or words[0] in _NOT_A_LABEL_PREFIXES:
        return False
    return not any(w in _NOT_A_LABEL_WORDS for w in words)


def group_key_column(df) -> str | None:
    """The column that makes each row one group -- one cluster, one domain -- or ``None``.

    A deconvolution result has one row per spot, and spots share clusters: a per-spot cluster label
    repeats down the column (``miso_clusters.csv``: 200 spots, 3 clusters). A column of cluster
    labels that names every row exactly once says the rows *are* the clusters, and then the table is
    a summary over groups -- whatever its other columns hold. That is what a model writes when it
    tallies its clustering against the samples: ``cluster_sample_crosstab.csv`` (``leiden`` x eight
    donors), ``cluster_by_timepoint_fractions.csv`` (``cluster`` x eight timepoints, rows summing to
    1). Non-negative, numeric and under names nothing recognises, both cleared
    :func:`looks_like_proportions`, and the recorded post-analysis of the ``stcoscientist-v8p9`` and
    ``-L1`` trials published donor ``D01363_NG02952`` and timepoint ``4hr`` as the tissue's most
    abundant cell types, reviewed ``ok``.

    The column names cannot separate the two -- ``4hr`` and ``Immature`` are no less plausible as
    labels than ``CD14+ Monocytes`` -- and the row sums cannot either: a cluster x timepoint fraction
    table sums to 1 per row exactly as a composition does. The row key can, because it is the one
    place the table says what a row is.

    The key's name is matched word by word rather than against one grammar. The grammar this
    replaced -- a group word, optional digits, one of five suffixes -- missed the recorded
    ``cluster_mbkm_harmony`` and ``leiden_harmony_r04`` and scanpy's canonical ``leiden_res_0.50``,
    so a ``stcoscientist-v8src2`` run's cluster x timepoint table stayed a deconvolution whose top
    cell type was its own key.

    Integer or text labels only: a float column is a measurement, and a measurement is unique per
    row by accident. And only when nothing else names the rows: a table indexed by its spots has
    rows that are those spots, whatever a column beside them holds -- ``spatial_domain`` numbered
    0..199 down 200 barcodes is a degenerate labelling, not 200 groups. Every recorded table over
    groups carries a numbered index, because its key is a column.

    Measured over the 343 recorded post-analysis directories of every ``stcoscientist`` arm (299
    tables) and the 128 smoke-output runs (1,359): 227 and 64 name their rows this way -- the 64 are
    istar's per-cluster summaries (a tool with no registry profile) and scanpy's per-domain spot
    tally. Not one real deconvolution output carries such a column, unique or not.
    """
    if len(df) < _MIN_GROUP_ROWS or not _index_numbers_the_rows(df.index):
        return None
    for position, column in enumerate(df.columns):
        if not _names_a_group(str(column).strip()):
            continue
        values = df.iloc[:, position]
        if getattr(values.dtype, "kind", "") not in ("i", "u", "O"):
            continue
        if bool(values.notna().all()) and values.is_unique:
            return str(column)
    return None


def group_key(df) -> tuple[str, ...] | None:
    """The column, or the pair of columns, that makes each row one group -- or ``None``.

    :func:`group_key_column` first. Then the same table written long: one row per cluster *and*
    timepoint, ``time_str, ident, pct`` down 72 rows. No single column names a row there -- each
    cluster repeats once per timepoint -- but a cluster column and one other label column name every
    row once between them, and a per-spot table cannot do that: spots share a cluster *and* a
    sample, a batch, an array row. Recorded in the ``stcoscientist-v3`` arm as
    ``cluster_time_percentages.csv`` and ``harmony_cluster_time_percentages.csv``; the second was
    published as a deconvolution whose top cell type was ``pct``, reviewed ``ok``.

    Both columns must repeat, each value at least twice, and neither may be a float: that is what
    keeps a coordinate or a score from pairing with a cluster by accident. And the other column may
    not be a gene or a rank (:data:`_PER_GENE_COLUMN_RE`): ``cluster, rank, gene, score`` is one row
    per cluster and rank too, but it is a per-cluster gene ranking -- what scanpy's
    ``rank_genes_groups`` writes -- which the gene rung reads and this does not claim. The index gate
    and the row floor are :func:`group_key_column`'s.
    """
    key = group_key_column(df)
    if key is not None:
        return (key,)
    if len(df) < _MIN_GROUP_ROWS or not _index_numbers_the_rows(df.index):
        return None
    labels = []
    for position, column in enumerate(df.columns):
        values = df.iloc[:, position]
        if getattr(values.dtype, "kind", "") not in ("i", "u", "O") or not bool(values.notna().all()):
            continue
        counts = values.value_counts()
        if len(counts) >= 2 and int(counts.min()) >= 2:
            labels.append((position, str(column)))
    for g_position, group in labels:
        if not _names_a_group(group.strip()):
            continue
        for p_position, other in labels:
            if p_position == g_position or _PER_GENE_COLUMN_RE.match(other.strip()):
                continue
            if not df.iloc[:, [g_position, p_position]].duplicated().any():
                return (group, other)
    return None


def spot_evidence(df) -> str | None:
    """What in ``df`` lets its rows be spots, as a clause for a reader -- or ``None`` when nothing does.

    A cluster x donor table written ``pd.DataFrame(counts, index=clusters).to_csv()`` has no key
    column for :func:`group_key_column` to find: the clusters are the unnamed first column, which
    :func:`read_table` makes the index, and ``0..10`` reads back as a row numbering. What such a
    table lacks is anything a per-spot result has -- names for its rows, or enough of them. So a
    table is taken for one row per spot when something names the rows (an index that is not a
    numbering, a text column that names every row once, spot barcodes across the columns of one
    written cell types x spots) or an axis is :data:`MIN_UNNAMED_SPOTS` long. The two recorded
    tables over clusters written this way have 3 and 11 rows. Nine recorded parameter sweeps, 3 to
    24 rows, have no names either; the content route typed seven of them compositions -- ``count``
    the most abundant cell type of one, ``k`` of two -- and, once the first fix declined the table
    over groups beside it, an eighth.

    Evidence *for* the reading, not a proof of it, and asked only where nothing else vouches for the
    table: the content route, and the runner's choice among files the tool's registry profile does
    not name. A registry-named output is taken whatever its size.
    """
    if not _index_numbers_the_rows(df.index):
        return "the index names the rows rather than numbering them"
    for column in df.columns:
        values = df[column]
        if getattr(values.dtype, "kind", "") == "O" and bool(values.notna().all()) and values.is_unique:
            return f"{str(column)!r} names every row"
    if looks_like_barcodes(df.columns):
        return "the columns are spot barcodes"
    longest = max(int(df.shape[0]), int(df.shape[1]))
    if longest >= MIN_UNNAMED_SPOTS:
        return f"an axis {longest} long, enough to be a section's spots"
    return None


def _index_numbers_the_rows(index) -> bool:
    """True when the index only numbers the rows -- ``0..n-1`` or ``1..n`` in any order.

    Positions are what ``read_csv`` invents and what R writes as row names; in another order they
    are the same positions after a sort, and ``0, 1, 10, 11, 2, ...`` is a cluster label sorted as
    text. Anything else was put there as the rows' names, by :func:`read_table` or by the file.
    """
    if getattr(index.dtype, "kind", "") not in ("i", "u") or len(index) == 0 or not index.is_unique:
        return False
    low, high = int(index.min()), int(index.max())
    return low in (0, 1) and high - low == len(index) - 1


def looks_like_composition_summary(df, *, row_cap: int | None = None) -> tuple[bool, float | None, int]:
    """``(is_summary, total, n_values)`` for a per-cell-type mean composition.

    Every deconvolution runner writes one of these beside its spot-level result -- the mean share of
    each cell type over the whole tissue, one row per type. :func:`looks_like_proportions` cannot see
    it, because it declines a single-column frame outright, so it used to fall through to
    :func:`looks_like_gene_table`, which accepted ``B_plasma`` and ``Macrophages`` as gene symbols
    and ranked them by ``mean_proportion``.

    What is true here and of no gene ranking: one non-negative numeric column whose values sum to 1.
    500 FSV values in [0, 1] sum to about 250. Two guards keep that from being a coincidence:

    ``row_cap``
        The ``nrows`` the frame was read with, if it was capped. A sum over a truncated read is not
        the file's sum: spatrio's 40,001-row alignment table holds ``value=0.005`` throughout, so its
        first 200 rows sum to exactly 1.0000 and nothing about that is a composition. A frame that
        filled the cap is refused.

    the row floor
        Two numbers adding to 1 are a fraction and its complement, a precision/recall pair, or
        stlearn's ``evaluation_metrics.csv`` -- ``ARI`` and ``NMI``, which summed to 1.0789 in the
        recorded corpus and were the only near miss in 518 tables.
    """
    import numpy as np

    numeric = numeric_frame(df)
    if numeric.shape[1] != 1 or numeric.shape[0] < _MIN_COMPOSITION_ROWS:
        return False, None, numeric.shape[0]
    if row_cap is not None and len(df) >= row_cap:
        return False, None, numeric.shape[0]
    column = numeric.iloc[:, 0].to_numpy(dtype="float64", na_value=np.nan)
    finite = column[np.isfinite(column)]
    if finite.size < _MIN_COMPOSITION_ROWS or float(finite.min()) < 0.0:
        return False, None, int(finite.size)
    total = float(finite.sum())
    return abs(total - 1.0) <= _COMPOSITION_TOLERANCE, total, int(finite.size)


#: Fewer rows than this and "they sum to 1" is arithmetic, not evidence. See the docstring above.
_MIN_COMPOSITION_ROWS = 3

#: Written fractions are rounded, and the recorded summaries land between 0.9911 and 1.0000.
_COMPOSITION_TOLERANCE = 0.02


def _is_coordinate_frame(numeric) -> bool:
    names = {str(c).strip().lower() for c in numeric.columns}
    return numeric.shape[1] <= 3 and bool(names & {"x", "y"} == {"x", "y"} or names & {"row", "col"} == {"row", "col"})


def looks_like_gene_table(df) -> tuple[bool, str | None, str | None]:
    """``(is_gene_table, gene_key, score_column)``.

    ``gene_key`` is a column name or ``None`` meaning "the index". A gene table needs identifiers
    that look like gene symbols *and* at least one numeric column to rank them by; a bare list of
    strings is a gene list, handled separately.
    """
    gene_key = _gene_identifier(df)
    if gene_key is _NO_GENES:
        return False, None, None
    numeric = numeric_frame(df.drop(columns=[gene_key]) if gene_key is not None else df)
    if numeric.shape[1] == 0:
        return False, None, None
    return True, gene_key, pick_score_column(numeric.columns)


_NO_GENES = object()


def _gene_identifier(df):
    """The column holding gene symbols, ``None`` for the index, or ``_NO_GENES``.

    A frame whose index labels are drawn from its own column labels is a relation matrix and never a
    ranking, so the index is not offered as a gene axis. squidpy's neighbourhood enrichment is a
    square z-score matrix labelled ``TypeA,TypeB,TypeC`` both ways; three strings that look like
    symbols and three numeric columns to sort by were enough, and the run published "Genes tested: 3",
    "Highest ranked gene: TypeB" and a chart titled "Top spatially variable genes" whose bars are one
    column of a symmetric matrix -- at ``status: ok``.

    Containment of the *visible* row labels, not set equality, because callers disagree about how much
    of the file to read: ``detect``'s content route decides on ``nrows=200`` while ``tasks/svg`` reads
    the frame whole. Equality holds only on the whole file, so hotspot's 508 x 508 gene correlation
    matrix was a relation matrix to one caller and a ranking "by 'Myh11'" to the other. Its Visium
    twin is 12,428 x 12,428 and 2.8 GB, which settles which caller the guard has to work for: a test
    that needs the whole file is not available here.

    Still not *overlap*, which is what would cost something: every visible row label must be a column
    name, so a lone statistic column that happens to share a gene's name is as harmless as it was.
    Two distinct labels minimum -- one label relates nothing to anything. Columns are tried afterwards
    either way, so a frame carrying a genuine gene column beside a matching index is unaffected.
    """
    row_labels = {str(i) for i in df.index}
    label_square = len(row_labels) > 1 and len(df.columns) > 1 and row_labels <= {str(c) for c in df.columns}
    for candidate in (None, *df.columns):
        if candidate is None and label_square:
            continue
        values = df.index if candidate is None else df[candidate]
        # ``kind == "O"``, not ``dtype != object``: a numpy dtype compares equal to ``object`` but
        # is not it, so the identity form ruff asks for would silently reject every column.
        if candidate is not None and getattr(getattr(values, "dtype", None), "kind", "") != "O":
            continue
        sample = [str(v) for v in list(values)[:50]]
        if len(sample) < 3:
            continue
        if sum(1 for s in sample if _BARCODE_RE.match(s)) / len(sample) > 0.5:
            continue  # spot barcodes, not gene symbols
        hits = sum(1 for s in sample if _GENE_NAME_RE.match(s) and not _NUMERIC_NAME_RE.match(s))
        if hits / len(sample) >= 0.8:
            return candidate
    return _NO_GENES


#: Below this many entries, a vocabulary is not evidence of anything. The guard below asks whether a
#: claimed cell-type axis is really the dataset's gene panel, and it answers by comparing against the
#: ``var_names`` of an h5ad found beside it -- but some tools publish their *result* as an h5ad whose
#: ``var_names`` are the cell types, and comparing a correct composition against that vocabulary
#: scores 1.00 and rejects a correct result. Size is what separates the two, with room to spare: in
#: the recorded corpus the gene panels run 206-900 entries and the largest cell-type panel is 35.
_MIN_GENE_PANEL = 100


def columns_are_the_gene_panel(columns, gene_names) -> bool:
    """True when ``columns`` are drawn from ``gene_names`` -- an expression matrix, not a composition.

    The other guards in this module infer from a table's own content, and for this defect content is
    provably not enough: stPlus's imputed expression (200 spots x 10 genes, non-negative,
    unnormalised) and IRIS's proportions (200 x 8, non-negative, unnormalised) are the same table to
    any test that only looks at the numbers. Measured three ways -- a gene-symbol name test scores an
    acronym panel (``OPC, VLMC, AST, NK``) at 1.00, identical to real genes; axis sizes overlap; and
    genuine compositions publish abundances so they do not sum to 1 either.

    The evidence is not in the table, it is next to it. These are the *dataset's own* genes, so this
    is direct evidence rather than a heuristic, and the acronym problem disappears with it: ``AST``
    would have to appear in that specific dataset's ``var_names``, and then only as 1 column of N.
    Measured over the recorded corpus, stPlus scores 0.87 and every genuine composition whose
    directory carries an h5ad scores exactly 0.00 against 209-900 genes.

    ``WNT4.1`` is what pandas names the second ``WNT4`` column, and the recorded stPlus run reported
    exactly that as the tissue's dominant cell type, so the suffix has to be seen through. The raw
    spelling is tried first, because a gene really named ``MT.1`` must still match itself.
    """
    genes = {str(g).strip().lower() for g in gene_names}
    if len(genes) < _MIN_GENE_PANEL:
        return False
    names = [str(c).strip() for c in columns]
    if not names:
        return False
    return sum(1 for n in names if _matches_gene(n, genes)) / len(names) > 0.5


def _matches_gene(name: str, genes: set[str]) -> bool:
    lowered = name.lower()
    if lowered in genes:
        return True
    stem, _, tail = lowered.rpartition(".")
    return bool(stem) and tail.isdigit() and stem in genes


# The inspector already classifies SVG columns, and a copy of its regexes here would drift. They
# are imported, with a stdlib-only fallback so post-analysis still runs if benchmarking is absent
# (the same defensive pattern detect.py uses for the registry). The fallback is a literal copy and
# the test suite pins the two spellings equal.
try:  # pragma: no cover -- exercised by the import itself
    from spatialomicsgym.benchmarking.output_inspector import (
        _PVALUE_COLUMN_RE,
        _SCORE_COLUMN_RE,
    )
except Exception:
    _PVALUE_COLUMN_RE = re.compile(r"fdr|[pq][-_ ]?(?:val|value|values|adj)|adjusted[-_ ]?p|p[-_ ]?adj", re.IGNORECASE)
    _SCORE_COLUMN_RE = re.compile(
        r"moran|geary|fsv|gft|llr|intrinsic|statistic|[-_ ]?score|z[-_ ]?score|ranking", re.IGNORECASE
    )

#: What post-analysis recognises *beyond* the inspector, kept separate so the delta is visible
#: rather than buried in a forked pattern. The inspector reads a tool's own thresholded output and
#: sees spelled-out column names; here an arbitrary table has to be classified, and the bare
#: statistic symbols are what SPARK-X (``AI``/``PI``), Moran (``I``) and Geary (``C``) actually
#: write. Anchored so they match only a column named exactly that.
_EXTRA_SCORE_COLUMN_RE = re.compile(r"autocorr|^(AI|PI|I|C|Z)$", re.IGNORECASE)

#: The same delta for p-values. The inspector's separator class is ``[-_ ]``, which excludes ``.``,
#: so the dotted R spellings (``p.value``, ``p.adj``, limma's ``adj.P.Val``) came back ``unknown``.
#: This used to be documented as a gap not worth closing on the grounds that no packaged tool writes
#: one -- true, and beside the point, because :func:`pick_score_column` selected such a column
#: anyway and then :func:`column_kind` declined to call it a p-value, so an SVG table was sorted
#: *descending* by an adjusted p-value and reported its least significant gene as the top hit.
#: Agent-authored analysis code is where these names come from, and an agent writing the analysis
#: code is what this system does. Kept here rather than in ``output_inspector`` for the same reason
#: as the score delta above: that regex sits on a scored path whose recorded numbers must not move.
_EXTRA_PVALUE_COLUMN_RE = re.compile(r"[pq]\.(?:val|value|values|adj)|adjusted\.p|p\.adj", re.IGNORECASE)


def column_kind(name: str) -> str:
    """``"pvalue"``, ``"score"`` or ``"unknown"``.

    The inspector's classification, plus :data:`_EXTRA_SCORE_COLUMN_RE` and
    :data:`_EXTRA_PVALUE_COLUMN_RE`. It never reclassifies a p-value as a score, so a column the
    inspector calls ``"pvalue"`` is called ``"pvalue"`` here.
    """
    text = str(name)
    if _PVALUE_COLUMN_RE.search(text) or _EXTRA_PVALUE_COLUMN_RE.search(text):
        return "pvalue"
    if _SCORE_COLUMN_RE.search(text) or _EXTRA_SCORE_COLUMN_RE.search(text):
        return "score"
    return "unknown"


def pick_score_column(columns) -> str | None:
    """The column an SVG ranking should use: an adjusted p-value, then a raw one, then a score.

    Preferring the adjusted p-value matters -- the alternative is ranking by a raw p-value the tool
    itself would not have thresholded on.

    Every rule here filters through :func:`column_kind` first, because the caller asks
    :func:`column_kind` which direction to sort in. The two used to disagree: rule 1 matched the
    bare substring ``adj``, which selected ``p.adj`` (that ``column_kind`` then called ``unknown``,
    inverting the ranking) and equally selected ``adj_r2``, an R-squared, in preference to a real
    ``pval`` sitting in the same table. Whatever this returns has to survive being classified.
    """
    names = [str(c) for c in columns]
    pvalues = [n for n in names if column_kind(n) == "pvalue"]
    adjusted = [n for n in pvalues if re.search(r"fdr|qval|adj", n, re.I)]
    if adjusted:
        return adjusted[0]
    if pvalues:
        return pvalues[0]
    scores = [n for n in names if column_kind(n) == "score" and not _is_never_a_statistic(n)]
    if scores:
        return scores[0]
    # Last resort, and it has to be a column that could be a measurement at all. Returning a bare
    # ``names[0]`` returned whatever sorted first, which for a table of spot positions is ``x`` --
    # and nothing downstream questions the answer, so two tools' plain coordinate files came out of
    # the smoke fixtures as finished SVG results with ``status=ok``: 902 "genes" named ``UCID794``
    # ranked by ``x``, and 200 named ``s0_CATCAAACTGGCGCCC-1`` ranked by ``x``. The only caveat
    # emitted was "'x' is a ranking statistic, not a test", which concedes the significance point
    # and asserts the false half.
    #
    # Narrowed rather than removed: a tool writing ``gene,spatial_variability`` under a name no
    # pattern here knows must still be ranked. :data:`_NON_CELLTYPE_COLUMN_RE` is the list of names
    # that are never a measurement, and is already what keeps coordinates out of a composition.
    # ``I``/``C``/``Z`` overlap that alphabet and are returned by the score rule above, before this.
    measured = [n for n in names if not _is_never_a_statistic(n)]
    return measured[0] if measured else None


def _is_never_a_statistic(name: str) -> bool:
    """A name that carries position, QC or identity rather than a measured quantity.

    Applied only inside :func:`pick_score_column`, deliberately, and not folded into
    :func:`column_kind`: that function also gates :func:`looks_like_proportions`, where demoting a
    name to ``"unknown"`` *admits* a table rather than rejecting one, so the same edit there would
    let a Visium ``tissue_positions`` file through as a composition.

    :data:`_EXTRA_SCORE_COLUMN_RE` wins, because the bare statistic symbols ``AI``/``PI``/``I``/
    ``C``/``Z`` share an alphabet with the coordinate names. The two names this actually rescues are
    ``pxl_row_in_fullres`` and ``pxl_col_in_fullres``: the inspector's score pattern matches
    substrings and ``llr`` is a substring of ``fullres``, so the Visium full-resolution pixel
    coordinates were classified as spatial statistics and ranked genes by a pixel row. That pattern
    is imported from ``benchmarking/output_inspector.py`` and sits on a scored path whose recorded
    numbers must not move, so it is corrected here rather than there.
    """
    text = str(name).strip()
    return bool(_NON_CELLTYPE_COLUMN_RE.match(text)) and not _EXTRA_SCORE_COLUMN_RE.match(text)
