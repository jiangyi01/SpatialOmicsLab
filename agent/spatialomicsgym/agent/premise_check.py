"""Deterministic check that the tissue a user names matches the tissue the data shows.

Motivated by a real session: a human DLPFC (brain) Visium slide was described as "my human breast
tumour Visium sample", and the agent answered with a confident ER+ call on "the tumor region",
citing an ESR1 enrichment with an FDR. Every number was correctly computed; the premise was false,
and nothing in the pipeline had any way to notice. ESR1 ranks 5939th of 33538 genes by total counts
on that slide. MBP, PLP1, GFAP and SNAP25 are in its top 200.

The whole design is built around one asymmetry: **a false positive here is worse than the bug**,
because it means telling a scientist their correctly-labelled sample is something else. So the
check declines wherever it is not certain -- no tissue named, two tissues named, no raw counts, no
dominant signature, a weak margin -- and emits at most one short line when it does fire.

Because it is silent unless the prompt names a tissue *and* the data contradicts it, it cannot
touch a benchmark: those prompts name no tissue at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# What each tissue looks like from the top of a count matrix.
#
# These are NOT "marker genes" in the differential-expression sense. They are transcripts that are
# among the most ABUNDANT ones in that tissue, which is a much stronger and much rarer property --
# and it is the property the failed session lacked. ESR1 is a perfectly good breast marker and is
# nowhere near the top of a breast slide either, which is exactly why keying on specificity rather
# than abundance is what let the fabricated call through.
#
# Housekeeping genes (ACTB, GAPDH, FTL, B2M, ribosomal, mitochondrial) are deliberately absent:
# they top every tissue and would make every class match everything.
# ---------------------------------------------------------------------------------------------
_TISSUE_MARKERS: dict[str, frozenset[str]] = {
    "brain": frozenset(
        {
            "MBP",
            "PLP1",
            "MOBP",
            "MOG",
            "CNP",
            "MAG",
            "SNAP25",
            "SYT1",
            "NEFL",
            "NEFM",
            "STMN2",
            "CAMK2A",
            "GRIN1",
            "NRGN",
            "GFAP",
            "AQP4",
            "SLC1A2",
            "SLC1A3",
            "OLIG1",
            "OLIG2",
        }
    ),
    "liver": frozenset(
        {
            "ALB",
            "APOA1",
            "APOA2",
            "APOC1",
            "APOC3",
            "APOH",
            "HP",
            "HPX",
            "FGA",
            "FGB",
            "FGG",
            "SERPINA1",
            "TF",
            "ORM1",
            "CYP2E1",
            "AMBP",
        }
    ),
    "kidney": frozenset(
        {"UMOD", "ALDOB", "SLC34A1", "LRP2", "NPHS2", "AQP2", "SLC12A1", "MIOX", "GATM", "CUBN", "SLC12A3"}
    ),
    "pancreas": frozenset(
        {
            "PRSS1",
            "PRSS2",
            "CTRB1",
            "CTRB2",
            "CELA3A",
            "CELA2A",
            "CPA1",
            "CPB1",
            "PNLIP",
            "CLPS",
            "CTRC",
            "INS",
            "GCG",
            "SPINK1",
        }
    ),
    "heart": frozenset({"MYH7", "TNNT2", "NPPA", "NPPB", "MYL2", "MYL3", "ACTC1", "MYBPC3", "TNNI3", "DES", "MB"}),
    "skeletal muscle": frozenset({"ACTA1", "MYH1", "MYH2", "CKM", "MYBPC1", "TNNT3", "TNNC2", "MYLPF", "NEB", "TTN"}),
    "lung": frozenset({"SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "SCGB1A1", "SCGB3A2", "NAPSA", "AGER", "SLC34A2"}),
    "prostate": frozenset({"KLK3", "KLK2", "MSMB", "ACP3", "ACPP", "NKX3-1", "TGM4", "KLK4"}),
    "breast": frozenset(
        {"SCGB2A2", "SCGB1D2", "MUCL1", "PIP", "LALBA", "ANKRD30A", "TFF1", "TFF3", "KRT18", "KRT8", "CSN2", "CSN3"}
    ),
    "intestine": frozenset(
        {"FABP1", "OLFM4", "DEFA5", "DEFA6", "MUC2", "LGALS4", "CEACAM7", "PIGR", "GUCA2A", "GUCA2B", "CA1", "AQP8"}
    ),
    "stomach": frozenset({"PGA3", "PGA4", "PGA5", "PGC", "LIPF", "CBLIF", "GIF", "TFF2", "ATP4A", "ATP4B", "MUC5AC"}),
    "skin": frozenset({"KRT1", "KRT2", "KRT10", "KRT14", "KRT5", "FLG", "LOR", "DCD", "KRTDAP", "SBSN"}),
    "testis": frozenset({"PRM1", "PRM2", "TNP1", "PGK2", "SPATA3", "ODF1", "AKAP4"}),
    "thyroid": frozenset({"TG", "TPO", "IYD", "TSHR", "PAX8", "DUOX1"}),
    "placenta": frozenset({"CSH1", "CSH2", "CGA", "CGB3", "PSG1", "PSG3", "PSG4", "HSD3B1"}),
    "retina": frozenset({"RHO", "SAG", "PDE6A", "PDE6B", "RCVRN", "GNAT1", "NRL", "CRX"}),
}

# Words that name a tissue, mapped to the class above. Deliberately conservative:
#
#  * bare "cortex" is missing -- renal, adrenal and cerebral cortex all use it, and guessing wrong
#    is precisely the mistake this module exists to prevent;
#  * "melanoma" is missing -- it arises in skin but a metastasis carries no epidermal signature;
#  * "muscle" is missing -- cardiac, smooth and skeletal muscle are three different answers.
#
# Every bare word is anchored. Unanchored, "renal" fires on ad-RENAL, "liver" on de-LIVER-ed,
# "lung" on p-LUNG-er, "colon" on COLON-y and "thyroid" on para-THYROID -- each one a confident
# wrong tissue call, which is the failure mode this module is supposed to remove rather than add.
#
# And the phrases that name a tissue without claiming one are excluded (hunt 2026-09-30,
# u15-validation-3): the SUBventricular zone is brain, hepatocyte / epidermal / placental GROWTH
# FACTOR are proteins, a brainstorm is not a brain, "at the heart of" is an idiom and a
# colon-separated list is punctuation. ``\w*+`` is possessive so a lookahead cannot be dodged by
# giving back letters ("epiderma" + "l growth factor").
_GROWTH_FACTOR = r"(?![\s-]+growth[\s-]+factor)"
_TISSUE_WORDS: dict[str, str] = {
    r"\bbrain(?!storm)\w*|\bcerebral\b|cerebell\w*|hippocamp\w*|\bdlpfc\b|prefrontal|neocort\w*|cortical layer|"
    r"\bwhite matter\b|\bstriatum\b|glioma|glioblastoma|\bgbm\b|astrocytoma|olfactory bulb": "brain",
    r"\bbreast\b|\bmammary\b|\bdcis\b|\bductal carcinoma\b": "breast",
    r"\bliver\b|\bhepatic\b|hepatocellular|hepatocyte\w*+" + _GROWTH_FACTOR + r"|\bhcc\b": "liver",
    r"\bkidney\w*|(?<![a-z])renal\b|\bnephron\w*|\bnephrectomy\b|glomerul\w*": "kidney",
    r"\blung\w*|\bpulmonary\b|\balveolar\b|\bnsclc\b|\bluad\b|\blusc\b|bronchi\w*": "lung",
    r"\bpancrea\w*|\bpdac\b|\bislet\w*": "pancreas",
    r"\bheart\b(?!\s+of\b)|\bcardiac\b|myocard\w*|(?<![a-z])ventricul\w*+(?![\s-]+zone)|\batrial\b": "heart",
    r"\bskeletal muscle\b|quadricep\w*|\bgastrocnemius\b|\bsoleus\b|\btibialis\b": "skeletal muscle",
    r"\bprostate\w*|\bprostatic\b": "prostate",
    r"\bintestin\w*|\bcolon\b(?![\s-]+separated)|colonoscop\w*|\bcolorectal\b|\bcrc\b|\bileum\b|\bjejunum\b|\bduodenum\b|"
    r"\brectal mucosa\b": "intestine",
    r"\bstomach\b|\bgastric\b|\bantrum\b|pylor\w*": "stomach",
    r"\bskin\b|epiderm\w*+" + _GROWTH_FACTOR + r"|(?<![a-z])cutaneous\b|psoria\w*|\bdermis\b": "skin",
    r"\btestis\b|\btesticular\b|\bseminiferous\b": "testis",
    r"(?<![a-z])thyroid\b": "thyroid",
    r"\bplacenta\w*+" + _GROWTH_FACTOR + r"|trophoblast\w*|\bchorionic villi\b": "placenta",
    r"\bretina\w*|\bphotoreceptor\w*": "retina",
}

_CLAIM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.I), tissue) for pattern, tissue in _TISSUE_WORDS.items()
]

# Genes excluded from the ranking before markers are looked for. They are the most abundant
# transcripts in every tissue and carry no identity, so leaving them in would just push real
# markers out of the window.
_UNINFORMATIVE = re.compile(r"^(MT-|MT\d|MTRNR|RP[LS]\d|RPLP|RPSA|RN7S|MALAT1$|NEAT1$)", re.I)

# How many of the highest-count genes to look in. Wide enough that a real signature clears the
# housekeeping crowd (brain scores 8/20 on the DLPFC slide), narrow enough that a stray marker in
# the long tail does not count as evidence.
_TOP_N = 200

# A class needs this many of its markers in that window before it is called at all...
_MIN_HITS = 3
# ...and this much of a lead over the claimed tissue before the claim is contradicted.
#
# Not "the claimed tissue has almost no hits": SCGB2A2 and SCGB1D2 (both breast) are genuinely in
# the top 200 of the DLPFC slide -- SCGB2A2 is 9th -- so a rule that required the claim to score
# zero would have stayed silent on the exact case this module was written for. Brain scores 8 there
# against breast's 2, and it is the margin that carries the conclusion.
_MIN_MARGIN = 3


@dataclass(frozen=True)
class TissueMismatch:
    """A named tissue that the data's most abundant transcripts do not support."""

    claimed: str
    observed: str
    evidence: tuple[str, ...]

    def note(self) -> str:
        """One line for the agent. Short on purpose -- a paragraph here gets skimmed."""
        genes = ", ".join(self.evidence[:5])
        return (
            f"[Data check] You described this as {self.claimed}, but the most abundant transcripts "
            f"in the file are {self.observed} markers ({genes}). Confirm what the sample is before "
            f"interpreting, and say so in your answer if the data disagrees with the description."
        )


def claimed_tissue(prompt: str) -> str | None:
    """The one tissue a prompt names, or None if it names none or several.

    Several is treated the same as none: "compare my breast and brain slides" says nothing about
    which file is in front of us.
    """
    if not prompt:
        return None
    found = {tissue for pattern, tissue in _CLAIM_PATTERNS if pattern.search(prompt)}
    return found.pop() if len(found) == 1 else None


#: ``(abspath, st_mtime_ns, st_size)`` -> the ``_gene_totals`` answer, None included. Same shape and
#: reasoning as ``prompt_builder._SPATIAL_DIAGNOSIS_MEMO``: the file's identity, not its path, so a
#: rewritten file is re-read. Measured before this existed: a 3.9 GB h5ad cost 6.6 s and a 4.1 GB
#: memory peak on EVERY turn whose prompt named a tissue, ahead of the first model call.
_GENE_TOTALS_MEMO: dict[tuple[str, int, int], object] = {}
_GENE_TOTALS_MEMO_CAP = 8

#: Entries streamed per HDF5 read. Bounds memory to a few hundred MB whatever the file's size.
_STREAM_CHUNK = 1 << 24


def forget_gene_totals() -> None:
    """Drop every memoised answer. Called beside ``forget_spatial_diagnoses`` on an account change:
    the memo carries no owner, for the same reason that one does not."""
    _GENE_TOTALS_MEMO.clear()


def _gene_totals(path: Path):
    """Per-gene total counts and gene names, or None if the matrix is not raw counts.

    Refusing normalised data is not caution for its own sake. The processed output of the failed
    session was HVG-filtered and log-scaled, and its top-"expressed" gene was MRM2 -- an artefact
    of scaling. Ranking genes on a matrix like that yields confident nonsense.

    Streamed from the HDF5 file rather than loaded, and memoised on the file's identity. Any file
    the streaming reader does not recognise goes through :func:`_gene_totals_in_memory`, the
    original, so an unusual encoding costs time and never a wrong answer.

    "Does not recognise" means ANY failure of the streamed read, not only :class:`_NotStreamable`.
    The stream re-implements a slice of anndata's reader over raw HDF5, so what trips it is whatever
    that slice leaves out, and that cannot be listed in advance. Measured while only
    ``_NotStreamable`` fell back: a pre-0.7 file whose ``var`` is a compound dataset raised
    ``AttributeError`` from the name read, and anndata 0.10 raised ``ModuleNotFoundError`` on every
    file from the import; ``observed_tissue`` swallows both, so the check went silent on files the
    loader answers.
    """
    import os

    key = None
    try:
        st = os.stat(path)
        key = (os.path.abspath(str(path)), st.st_mtime_ns, st.st_size)
    except OSError:
        pass
    if key is not None and key in _GENE_TOTALS_MEMO:
        return _GENE_TOTALS_MEMO[key]
    try:
        answer = _gene_totals_streamed(path)
    except Exception:
        answer = _gene_totals_in_memory(path)
    if key is not None:
        if len(_GENE_TOTALS_MEMO) >= _GENE_TOTALS_MEMO_CAP:
            _GENE_TOTALS_MEMO.pop(next(iter(_GENE_TOTALS_MEMO)))
        _GENE_TOTALS_MEMO[key] = answer
    return answer


class _NotStreamable(Exception):
    """The file uses an encoding the streaming reader does not handle; load it instead."""


def _gene_totals_streamed(path: Path):
    """:func:`_gene_totals_in_memory`'s answer, read a chunk at a time from the HDF5 file.

    Same candidates in the same order (``X``, ``layers/counts``, ``raw/X``), the same counts test
    over the same evenly spaced row blocks, and totals accumulated in float64. Raises
    :class:`_NotStreamable` for anything that is not a dense dataset or a CSR/CSC group, and when
    nothing it read is counts but the file keeps a pre-0.7 dotted ``raw.X``, which it does not read:
    returning None there would memoise "not raw counts" for a file whose raw counts the loader finds.
    """
    import h5py
    import numpy as np

    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11, which pyproject still admits (``anndata>=0.10``)
        from anndata.experimental import read_elem

    from spatialomicsgym.agent.data_validation import _count_sample_blocks

    def encoding(obj) -> str:
        enc = obj.attrs.get("encoding-type", "")
        return enc.decode() if isinstance(enc, bytes) else str(enc)

    def sampled_values(obj, n_obs: int):
        blocks = _count_sample_blocks(n_obs)
        if isinstance(obj, h5py.Dataset):
            return np.concatenate([np.asarray(obj[a:b]).ravel() for a, b in blocks]) if blocks else np.array([])
        indptr = obj["indptr"]
        if encoding(obj) == "csr_matrix":
            chunks = []
            for a, b in blocks:
                lo, hi = int(indptr[a]), int(indptr[b])
                chunks.append(np.asarray(obj["data"][lo:hi]))
            return np.concatenate(chunks) if chunks else np.array([])
        # CSC: rows are scattered through every column, so stream the entries and keep the sampled rows.
        wanted = np.zeros(n_obs, dtype=bool)
        for a, b in blocks:
            wanted[a:b] = True
        nnz, out = int(indptr[-1]), []
        for lo in range(0, nnz, _STREAM_CHUNK):
            rows = np.asarray(obj["indices"][lo : lo + _STREAM_CHUNK])
            out.append(np.asarray(obj["data"][lo : lo + _STREAM_CHUNK])[wanted[rows]])
        return np.concatenate(out) if out else np.array([])

    def counts_like(obj, n_obs: int) -> bool:
        values = sampled_values(obj, n_obs)
        finite = values[np.isfinite(values) & (values != 0)]
        return bool(finite.size and finite.min() >= 0 and np.all(finite == np.floor(finite)))

    def totals(obj, n_obs: int, n_var: int):
        if isinstance(obj, h5py.Dataset):
            acc = np.zeros(n_var, dtype=np.float64)
            step = max(1, _STREAM_CHUNK // max(1, n_var))
            for a in range(0, n_obs, step):
                acc += np.asarray(obj[a : a + step], dtype=np.float64).sum(axis=0)
            return acc
        data, indices, indptr = obj["data"], obj["indices"], np.asarray(obj["indptr"][:], dtype=np.int64)
        nnz = int(indptr[-1])
        if encoding(obj) == "csr_matrix":
            acc = np.zeros(n_var, dtype=np.float64)
            for lo in range(0, nnz, _STREAM_CHUNK):
                cols = np.asarray(indices[lo : lo + _STREAM_CHUNK])
                acc += np.bincount(
                    cols, weights=np.asarray(data[lo : lo + _STREAM_CHUNK], dtype=np.float64), minlength=n_var
                )
            return acc
        # CSC: column j is data[indptr[j]:indptr[j+1]]; prefix sums at every boundary, streamed.
        prefix = np.zeros(len(indptr), dtype=np.float64)
        running = 0.0
        for lo in range(0, nnz, _STREAM_CHUNK):
            chunk = np.cumsum(np.asarray(data[lo : lo + _STREAM_CHUNK], dtype=np.float64)) + running
            hi = lo + len(chunk)
            first, last = np.searchsorted(indptr, lo, side="right"), np.searchsorted(indptr, hi, side="right")
            prefix[first:last] = chunk[indptr[first:last] - 1 - lo]
            running = float(chunk[-1]) if len(chunk) else running
        return np.diff(prefix)

    with h5py.File(path, "r") as f:
        for element, var_key in (("X", "var"), ("layers/counts", "var"), ("raw/X", "raw/var")):
            obj = f.get(element)
            if obj is None:
                continue
            if isinstance(obj, h5py.Group) and encoding(obj) not in ("csr_matrix", "csc_matrix"):
                raise _NotStreamable(f"{element}: {encoding(obj) or 'unknown encoding'}")
            shape = tuple(int(x) for x in (obj.shape if isinstance(obj, h5py.Dataset) else obj.attrs["shape"]))
            if len(shape) != 2:
                raise _NotStreamable(f"{element}: shape {shape}")
            if counts_like(obj, shape[0]):
                names = np.asarray([str(n) for n in read_elem(f[var_key]).index])
                return totals(obj, shape[0], shape[1]), names
        if "raw.X" in f:
            raise _NotStreamable("raw.X: the pre-0.7 dotted raw layout")
    return None


def _gene_totals_in_memory(path: Path):
    """Per-gene total counts and gene names, or None if the matrix is not raw counts -- by loading
    the whole file. The fallback for an encoding :func:`_gene_totals_streamed` does not recognise.

    Refusing normalised data is not caution for its own sake. The processed output of the failed
    session was HVG-filtered and log-scaled, and its top-"expressed" gene was MRM2 -- an artefact
    of scaling. Ranking genes on a matrix like that yields confident nonsense.
    """
    from spatialomicsgym.utils.file_io import read_h5ad_backed

    # Backed, and summed over row chunks: this runs before the model starts on every turn that
    # names a tissue, and it read the whole file -- X, layers, raw and the image stack of a
    # multi-GB Xenium slide -- into the serving process each time (hunt 2026-09-30,
    # u15-validation-5). Backed mode keeps X and raw.X on disk.
    with read_h5ad_backed(path) as adata:
        return _gene_totals_of(adata)


def _column_totals(matrix, chunk: int = 4096):
    """Per-column sums of ``matrix`` -- in memory or backed -- one block of rows at a time."""
    import numpy as np

    rows = int(matrix.shape[0])
    total = np.zeros(int(matrix.shape[1]), dtype=np.float64)
    for start in range(0, rows, chunk):
        block = matrix[start : min(rows, start + chunk)]
        total += np.asarray(block.sum(axis=0), dtype=np.float64).ravel()
    return total


def _gene_totals_of(adata):
    import numpy as np
    from scipy import sparse

    from spatialomicsgym.agent.data_validation import _count_sample_blocks

    def counts_like(matrix) -> bool:
        if matrix is None:
            return False
        # Evenly spaced blocks, and nonzeros only. Both come from the canonical raw-counts sample in
        # ``data_validation``, and for the reason its own comment gives: a Visium file LEADS with
        # zeros, because barcodes are stored in array order so the off-tissue rim comes first (a
        # filtered-not-subset object and a concatenation lead with zeros too). This read a
        # contiguous ``matrix[:500]``, and zero is a non-negative whole number in every matrix ever
        # written -- so a slide with a 520-spot rim answered "raw counts" from 500 rows of zeros,
        # and this function then ranked genes on a log-normalised matrix, which is the one thing its
        # docstring exists to prevent. Restricting to nonzeros is what turns an all-empty sample
        # into "cannot tell"; here that has to be a refusal, so the loop below moves on to
        # ``layers["counts"]`` and ``raw.X``, which is where the counts actually are.
        chunks = []
        for start, stop in _count_sample_blocks(int(matrix.shape[0])):
            block = matrix[start:stop]
            dense = block.toarray() if sparse.issparse(block) else np.asarray(block)
            chunks.append(np.asarray(dense).ravel())
        if not chunks:
            return False
        values = np.concatenate(chunks)
        finite = values[np.isfinite(values) & (values != 0)]
        return bool(finite.size and finite.min() >= 0 and np.all(finite == np.floor(finite)))

    for matrix, names in (
        (adata.X, adata.var_names),
        (adata.layers.get("counts") if adata.layers else None, adata.var_names),
        (adata.raw.X if adata.raw is not None else None, adata.raw.var_names if adata.raw is not None else None),
    ):
        if matrix is not None and counts_like(matrix):
            return _column_totals(matrix), np.asarray([str(n) for n in names])
    return None


def observed_tissue(path: str | Path) -> tuple[str | None, tuple[str, ...]]:
    """The tissue the file's most abundant transcripts support, with the genes that say so.

    ``(None, ())`` whenever the data does not make a clear case -- unreadable, not raw counts, no
    class reaching ``_MIN_HITS``, or two classes too close to separate.
    """
    path = Path(path)
    if not path.exists() or path.suffix != ".h5ad":
        return None, ()
    try:
        st = path.stat()
        key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    except OSError:
        return None, ()
    if key in _OBSERVED:
        return _OBSERVED[key]
    answer = _observed_tissue_uncached(path)
    if len(_OBSERVED) >= _OBSERVED_MAX:
        _OBSERVED.pop(next(iter(_OBSERVED)))
    _OBSERVED[key] = answer
    return answer


#: ``observed_tissue`` per file identity: every follow-up that names the tissue asks again about the
#: same slide, and the answer only changes when the file does.
_OBSERVED: dict[tuple[str, int, int], tuple[str | None, tuple[str, ...]]] = {}
_OBSERVED_MAX = 16


def _observed_tissue_uncached(path: Path) -> tuple[str | None, tuple[str, ...]]:
    try:
        totals = _gene_totals(path)
    # Bare except on purpose: a premise check must never be the thing that breaks a run.
    except Exception:
        return None, ()
    if totals is None:
        return None, ()

    import numpy as np

    values, names = totals
    informative = np.array([not _UNINFORMATIVE.match(n) for n in names])
    ranked = np.argsort(-np.where(informative, values, -np.inf))[:_TOP_N]
    top = {names[i].upper(): int(rank) for rank, i in enumerate(ranked) if values[i] > 0}

    scores = {tissue: sorted(markers & top.keys(), key=lambda g: top[g]) for tissue, markers in _TISSUE_MARKERS.items()}
    best = max(scores, key=lambda t: len(scores[t]))
    if len(scores[best]) < _MIN_HITS:
        return None, ()
    runner_up = max((len(v) for t, v in scores.items() if t != best), default=0)
    if len(scores[best]) - runner_up < _MIN_MARGIN:
        return None, ()
    return best, tuple(scores[best])


def check_tissue_premise(prompt: str, data_path: str | Path | None) -> TissueMismatch | None:
    """The whole check. Returns None -- meaning "say nothing" -- in every uncertain case."""
    if data_path is None:
        return None
    claimed = claimed_tissue(prompt)
    if claimed is None:
        return None
    observed, evidence = observed_tissue(data_path)
    if observed is None or observed == claimed:
        return None
    return TissueMismatch(claimed=claimed, observed=observed, evidence=evidence)
