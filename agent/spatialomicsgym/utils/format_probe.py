"""Evidence-based detection of data formats nobody declared.

Every ingest path in this repository dispatches on the *filename*: `spatial_pipeline` branches on
``p.suffix == ".h5ad"``, its tabular branch picks the separator from ``.tsv`` vs ``.csv``, and the
``data_converter`` portal trusts whatever the caller passed. None of them reads the first eight
bytes. A user-supplied ``counts.dat`` that is a perfectly good h5ad is therefore reported as
"Unrecognized file format", and a CSV re-saved as ``data.h5ad`` reaches ``anndata.read_h5ad`` and
dies with an OSError the agent has to guess its way out of.

This module answers a different question -- *what is this, actually?* -- from the bytes on disk,
the group layout inside a container, and the shape of a directory. Three rules govern it:

1. **Magic bytes before extensions.** The extension is recorded as a cross-check and, when it
   disagrees with the content, surfaced as a warning. It never drives the verdict.
2. **Say what you saw.** Every verdict carries the evidence that produced it. For a recognised
   format that is a provenance trail; for an ``unknown`` it is the entire value of the call --
   "HDF5 container, root keys ['Measurement', 'Calibration'], no h5ad/loom/10x marker" is
   actionable in a way that "unknown" is not.
3. **An honest ``unknown`` is a correct answer.** Guessing "csv" for a binary blob sends the agent
   into a doomed ``pd.read_csv``; refusing sends it somewhere useful.

Dependencies: the standard library only, at import time. ``spatialomicsgym.utils`` is imported by
the agent core on every run in the 1.6 GB minimal env, so h5py -- the one non-stdlib thing the
prober can use -- is imported lazily inside the function that needs it, and its absence degrades
the HDF5 verdict rather than breaking the import.

Also here: :func:`resolve_orientation`, which decides genes-x-cells vs cells-x-genes from evidence
and **refuses when the evidence does not separate the two axes**. That refusal is the point. This
codebase has been bitten twice by orientation heuristics that always produced an answer -- a
correct 6142 x 7 deconvolution matrix transposed to 7 x 6142 because ``spot_0`` reads as a cell
type name, and a silently transposed conversion reported as ``status: ok``. Correct shape and
plausible names prove nothing.
"""

from __future__ import annotations

import codecs
import csv
import gzip
import json
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "FormatVerdict",
    "OrientationVerdict",
    "probe_format",
    "resolve_orientation",
]

# ---------------------------------------------------------------------------
# Magic byte signatures. Offset 0 unless stated.
# ---------------------------------------------------------------------------

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_GZIP_MAGIC = b"\x1f\x8b"
_ZIP_MAGIC = b"PK\x03\x04"
_BZIP2_MAGIC = b"BZh"
_XZ_MAGIC = b"\xfd7zXZ\x00"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_PARQUET_MAGIC = b"PAR1"
_TIFF_LE_MAGIC = b"II\x2a\x00"
_TIFF_BE_MAGIC = b"MM\x00\x2a"
_BIGTIFF_LE_MAGIC = b"II\x2b\x00"
_BIGTIFF_BE_MAGIC = b"MM\x00\x2b"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_RDS_GZIP_INNER = b"X\n"  # serialized R object, XDR format, after gunzip
_NUMPY_MAGIC = b"\x93NUMPY"
_SQLITE_MAGIC = b"SQLite format 3\x00"
_MATLAB_MAGIC = b"MATLAB"
_PDF_MAGIC = b"%PDF"
# The OLE2 compound-file header, which starts every pre-2007 Office file. Telling .xls from .doc
# from .msg means walking the compound directory, so the extension decides -- but the family is
# named either way, because "no known magic-byte signature" was untrue of these eight bytes.
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
# tar keeps its magic at byte 257 of the first 512-byte header block, not at offset 0, because
# bytes 0-99 are the first member's filename. POSIX writes "ustar\0"; GNU writes "ustar  \0". Both
# start with the same five bytes, which is all that is matched here.
_TAR_MAGIC = b"ustar"
_TAR_MAGIC_OFFSET = 257
_TAR_BLOCK = 512
#: How many member names to walk out of the header blocks. The names are evidence, not an index --
#: enough to recognise a Space Ranger or Xenium layout and stop.
_TAR_MAX_MEMBERS = 40

# How much of a file we are ever willing to touch. A 4 GB Stereo-seq GEM must be probeable.
_HEAD_BYTES = 64 * 1024
_TEXT_SAMPLE_BYTES = 256 * 1024
# How much *compressed* input to feed the decompressor per pass, and the wbits that tell zlib to
# expect a gzip header and trailer rather than a bare deflate stream. See ``_decompress_head``.
_GZIP_CHUNK_BYTES = 64 * 1024
_GZIP_WBITS = 16 + zlib.MAX_WBITS
_MAX_SAMPLE_BYTES = 1 << 20

_DELIMITER_CANDIDATES = ("\t", ",", ";", "|", " ")

# ---------------------------------------------------------------------------
# Directory layout markers
# ---------------------------------------------------------------------------

_MTX_NAMES = ("matrix.mtx", "matrix.mtx.gz")
_BARCODE_NAMES = ("barcodes.tsv", "barcodes.tsv.gz", "barcodes.csv")
_FEATURE_NAMES = ("features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz", "features.csv")
_VISIUM_COUNTS = ("filtered_feature_bc_matrix.h5", "raw_feature_bc_matrix.h5", "filtered_feature_bc_matrix")
_VISIUM_POSITIONS = ("tissue_positions.csv", "tissue_positions_list.csv", "tissue_positions.parquet")
_XENIUM_MARKERS = (
    "experiment.xenium",
    "transcripts.parquet",
    "transcripts.csv.gz",
    "cell_feature_matrix.h5",
    "cells.parquet",
    "cells.csv.gz",
)
_MERFISH_MARKERS = ("cell_by_gene.csv", "cell_metadata.csv", "detected_transcripts.csv")
_COSMX_MARKERS = ("exprmat_file.csv", "metadata_file.csv", "tx_file.csv", "fov_positions_file.csv")
# Deposited pucks carry the puck id as a prefix -- `Puck_200115_08_bead_locations.csv` -- so the
# stock names, matched whole, recognise a spelling no published dataset uses. These are matched as
# endings instead. The bead locations are what make it Slide-seq: a DGE on its own is Drop-seq,
# with no coordinates in it anywhere, so it is evidence and never the deciding marker.
_SLIDESEQ_LOCATION_SUFFIXES = ("_bead_locations.csv", "beadlocationsforr.csv", "_barcode_matching.txt")
_SLIDESEQ_COUNT_SUFFIXES = (".digital_expression.txt.gz", ".digital_expression.txt", "mappeddgeforr.csv")

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".svs", ".ndpi")


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


@dataclass
class FormatVerdict:
    """What the bytes say this is, how sure we are, and why.

    Attributes:
        path: The probed path, as a string.
        kind: ``file``, ``directory``, ``missing`` or ``other`` (FIFO, socket, ...).
        format: Canonical format id, or ``unknown``. Never a guess -- see module docstring.
        container: Physical envelope: ``hdf5``, ``gzip``, ``zip``, ``parquet``, ``tiff``, ``text``,
            ``binary``, ``directory``. Present even when ``format`` is unknown.
        confidence: 0.0-1.0. Above 0.9 means a signature or a decisive group layout was matched.
        evidence: Human-readable observations that produced the verdict. Populated even -- and
            especially -- for ``unknown``.
        reader: Suggested reader call, e.g. ``anndata.read_h5ad(path)``. ``None`` when unknown.
        details: Format-specific facts a caller needs to actually read the thing: delimiter,
            header presence, HDF5 root keys, directory members, matrix shape.
        warnings: Things that are wrong but not fatal -- most importantly an extension that
            disagrees with the content.

    """

    path: str
    kind: str
    format: str = "unknown"
    container: str | None = None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    reader: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping. Every consumer returns this to the agent through an MCP portal."""
        return {
            "path": self.path,
            "kind": self.kind,
            "format": self.format,
            "container": self.container,
            "confidence": round(float(self.confidence), 3),
            "evidence": list(self.evidence),
            "reader": self.reader,
            "details": json.loads(json.dumps(self.details, default=str)),
            "warnings": list(self.warnings),
        }

    @property
    def is_known(self) -> bool:
        """True when the probe committed to a format it can name a reader for."""
        return self.format not in {"unknown", "missing", "empty_file", "empty_directory"} and self.confidence >= 0.5


@dataclass
class OrientationVerdict:
    """genes-x-cells vs cells-x-genes, or an explicit refusal.

    Attributes:
        orientation: ``cells_x_genes``, ``genes_x_cells`` or ``ambiguous``.
        transpose_needed: True/False when resolved, ``None`` when refused. The tri-state is
            deliberate: a caller that reads ``False`` out of a refusal has been misled.
        confidence: 0.0-1.0.
        refused: True when the evidence did not separate the axes.
        evidence: Every signal considered, including the ones that abstained.
        signals: Per-signal breakdown ``{name: {"vote": ..., "strength": ..., "detail": ...}}``.

    """

    orientation: str
    transpose_needed: bool | None
    confidence: float = 0.0
    refused: bool = False
    evidence: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping."""
        return {
            "orientation": self.orientation,
            "transpose_needed": self.transpose_needed,
            "confidence": round(float(self.confidence), 3),
            "refused": bool(self.refused),
            "evidence": list(self.evidence),
            "signals": json.loads(json.dumps(self.signals, default=str)),
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def probe_format(path: str | Path) -> FormatVerdict:
    """Identify the format of a file or directory from its content.

    Reads magic bytes first and the extension only as a cross-check. For HDF5 containers it
    inspects the group layout with h5py (imported lazily) to tell h5ad, loom and 10x apart. For
    directories it matches known layouts (10x MTX triplet, Visium Space Ranger, Xenium, MERFISH,
    CosMx, Slide-seq, Zarr). For delimited text it samples the head of the file -- never the whole
    file -- to detect the delimiter, whether there is a header row, and whether the first column is
    an index.

    Never raises for content reasons: an unreadable, truncated or unrecognised input comes back as
    a low-confidence verdict carrying the evidence gathered.

    Args:
        path: File or directory to probe.

    Returns:
        FormatVerdict: detected format, confidence, evidence, suggested reader and per-format
        details. Use ``.to_dict()`` to hand it to an agent.

    """
    p = Path(path)
    try:
        if not p.exists():
            return FormatVerdict(
                path=str(p),
                kind="missing",
                format="missing",
                confidence=0.0,
                evidence=[f"path does not exist: {p}"],
            )
        if p.is_dir():
            return _probe_directory(p)
        if p.is_file():
            return _probe_file(p)
    except OSError as exc:
        return FormatVerdict(
            path=str(p),
            kind="other",
            format="unreadable",
            confidence=0.0,
            evidence=[f"could not stat/read path: {exc}"],
        )
    return FormatVerdict(
        path=str(p),
        kind="other",
        format="unknown",
        confidence=0.0,
        evidence=["path is neither a regular file nor a directory (FIFO, socket or device?)"],
    )


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _probe_file(p: Path) -> FormatVerdict:
    size = p.stat().st_size
    v = FormatVerdict(path=str(p), kind="file")
    v.details["size_bytes"] = size

    if size == 0:
        v.format = "empty_file"
        v.evidence.append("file is zero bytes")
        return v

    try:
        with p.open("rb") as fh:
            head = fh.read(_HEAD_BYTES)
    except OSError as exc:
        v.format = "unreadable"
        v.evidence.append(f"could not read file: {exc}")
        return v

    _classify_bytes(v, p, head, size)
    _crosscheck_extension(v, p)
    return v


def _classify_bytes(v: FormatVerdict, p: Path, head: bytes, size: int) -> None:
    """Dispatch on the leading bytes. Populates ``v`` in place."""
    if head.startswith(_HDF5_MAGIC):
        v.container = "hdf5"
        v.evidence.append("HDF5 signature \\x89HDF\\r\\n\\x1a\\n at offset 0")
        _probe_hdf5_layout(v, p)
        return

    if head.startswith(_GZIP_MAGIC):
        v.container = "gzip"
        v.evidence.append("gzip signature \\x1f\\x8b at offset 0")
        _probe_gzip_member(v, p)
        return

    if head.startswith(_ZIP_MAGIC):
        v.container = "zip"
        v.evidence.append("zip local-file-header signature PK\\x03\\x04 at offset 0")
        _probe_zip_members(v, p)
        return

    if head.startswith(_PARQUET_MAGIC) or _has_parquet_footer(p, size):
        v.container = "parquet"
        v.format = "parquet"
        v.confidence = 0.97
        v.reader = "pandas.read_parquet(path)  # or pyarrow.parquet.read_table"
        v.evidence.append("Parquet magic 'PAR1' at file start and/or footer")
        return

    if head[:4] in (_TIFF_LE_MAGIC, _TIFF_BE_MAGIC, _BIGTIFF_LE_MAGIC, _BIGTIFF_BE_MAGIC):
        _classify_tiff(v, head)
        return

    if head.startswith(_PNG_MAGIC) or head.startswith(_JPEG_MAGIC):
        v.container = "binary"
        v.format = "image"
        v.confidence = 0.95
        kind = "PNG" if head.startswith(_PNG_MAGIC) else "JPEG"
        v.details["image_type"] = kind.lower()
        v.reader = "PIL.Image.open(path)"
        v.evidence.append(f"{kind} signature at offset 0")
        return

    if head.startswith(_NUMPY_MAGIC):
        v.container = "binary"
        v.format = "numpy_npy"
        v.confidence = 0.97
        v.reader = "numpy.load(path)"
        v.evidence.append("NumPy .npy magic \\x93NUMPY at offset 0")
        return

    if head.startswith(_SQLITE_MAGIC):
        v.container = "binary"
        v.format = "sqlite"
        v.confidence = 0.97
        v.reader = "sqlite3.connect(path)"
        v.evidence.append("SQLite 3 header string at offset 0")
        return

    if head[:6] == _MATLAB_MAGIC:
        v.container = "binary"
        v.format = "matlab_mat"
        v.confidence = 0.9
        v.reader = "scipy.io.loadmat(path)"
        v.evidence.append("MATLAB level-5 header text at offset 0")
        return

    if head.startswith(_BZIP2_MAGIC) or head.startswith(_XZ_MAGIC) or head.startswith(_ZSTD_MAGIC):
        which = "bzip2" if head.startswith(_BZIP2_MAGIC) else ("xz" if head.startswith(_XZ_MAGIC) else "zstd")
        v.container = which
        v.format = f"{which}_compressed"
        v.confidence = 0.85
        v.evidence.append(f"{which} signature at offset 0; inner format not inspected")
        v.reader = f"decompress with {which} first, then re-probe the member"
        return

    if head.startswith(_RDS_GZIP_INNER) and p.suffix.lower() in (".rds", ".rda", ".rdata"):
        v.container = "binary"
        v.format = "r_object"
        v.confidence = 0.85
        v.reader = "R: readRDS(path)  (no pure-Python reader)"
        v.evidence.append("uncompressed R serialization header 'X\\n'")
        return

    if head.startswith(_PDF_MAGIC):
        v.container = "binary"
        v.format = "pdf"
        v.confidence = 0.95
        v.evidence.append("PDF signature at offset 0")
        return

    if head.startswith(_OLE2_MAGIC):
        _classify_ole2(v, p)
        return

    # Checked after every offset-0 signature and before the text fallback, because a tar has no
    # signature at offset 0 -- it opens with its first member's *filename*, and its magic sits at
    # byte 257. Anything with a real leading signature has already returned above.
    if _looks_like_tar(head):
        _describe_tar(v, head, compressed=False)
        return

    text = _decode_text(head)
    if text is None:
        v.container = "binary"
        v.format = "unknown"
        v.confidence = 0.1
        v.evidence.append(f"leading bytes (hex): {_hexdump(head[:16])}")
        v.evidence.append("no known magic-byte signature; content is not decodable as text")
        # The hex stays -- a signature that is only partly printable is still read from it. This is
        # a second reading of the same bytes: the one a search engine can match. See
        # :func:`_printable_magic`.
        token = _printable_magic(head)
        if token:
            v.details["magic_ascii"] = token
            v.evidence.append(
                f"the leading bytes spell {token!r} in ASCII, which is how a magic number is "
                "usually written down -- searchable as-is"
            )
        v.details["search_terms"] = _format_search_terms(token, [p.suffix.lower()])
        return

    v.container = "text"
    # `len(head)`, not `len(text)`: this is the only place that still holds the bytes, and the
    # question downstream is a byte question -- did the fixed-size read stop short of the file?
    # Comparing `size` to a decoded character count, as the callees used to, called every file
    # with a non-ASCII byte in it truncated.
    _classify_text(v, p, text, sampled_bytes=len(head), truncated=len(head) < size)


def _classify_tiff(v: FormatVerdict, head: bytes) -> None:
    little = head[:2] == b"II"
    big_tiff = head[2:4] in (b"\x2b\x00", b"\x00\x2b")
    v.container = "tiff"
    v.details["byte_order"] = "little" if little else "big"
    v.details["bigtiff"] = big_tiff
    v.reader = "tifffile.imread(path)"
    v.evidence.append(f"TIFF magic {'II' if little else 'MM'} ({'BigTIFF' if big_tiff else 'classic'})")
    # OME-TIFF carries its OME-XML in the ImageDescription tag, which sits in the first IFD --
    # in practice within the first few KB. Substring search is enough and costs nothing.
    if b"<OME" in head or b"openmicroscopy.org" in head:
        v.format = "ome_tiff"
        v.confidence = 0.9
        v.evidence.append("OME-XML found in the header block (ImageDescription tag)")
    else:
        v.format = "tiff"
        v.confidence = 0.9
        v.evidence.append("no OME-XML in the first 64 KB; treating as a plain TIFF")


def _has_parquet_footer(p: Path, size: int) -> bool:
    """Parquet also stamps PAR1 at the very end. Cheap and catches truncated-header writes."""
    if size < 8:
        return False
    try:
        with p.open("rb") as fh:
            fh.seek(-4, 2)
            return fh.read(4) == _PARQUET_MAGIC
    except OSError:
        return False


def _probe_hdf5_layout(v: FormatVerdict, p: Path) -> None:
    """Tell h5ad / loom / 10x apart by group layout. Degrades honestly without h5py."""
    try:
        import h5py
    except Exception:
        h5py = None

    if h5py is None:
        v.format = "hdf5_unknown"
        v.confidence = 0.4
        v.evidence.append("h5py is not importable here, so the HDF5 group layout could not be inspected")
        v.warnings.append("install h5py to distinguish h5ad / loom / 10x inside this container")
        v.reader = "unknown HDF5 layout; install h5py and re-probe"
        return

    try:
        with h5py.File(str(p), "r") as f:
            root_keys = sorted(f.keys())
            root_attrs = {k: _scalar(f.attrs[k]) for k in f.attrs}
            v.details["root_keys"] = root_keys
            v.details["root_attrs"] = root_attrs

            # h5ad: anndata stamps the encoding on the root group. Decisive.
            if str(root_attrs.get("encoding-type", "")) == "anndata":
                v.format = "h5ad"
                v.confidence = 0.99
                v.reader = "anndata.read_h5ad(path)"
                v.evidence.append("root attribute encoding-type='anndata' (written by anndata itself)")
                _describe_h5ad(v, f)
                return
            if {"X", "obs", "var"}.issubset(set(root_keys)):
                v.format = "h5ad"
                v.confidence = 0.92
                v.reader = "anndata.read_h5ad(path)"
                v.evidence.append("root groups X/obs/var present (AnnData layout, no encoding attribute)")
                _describe_h5ad(v, f)
                return

            # loom
            if "matrix" in root_keys and {"row_attrs", "col_attrs"}.issubset(set(root_keys)):
                v.format = "loom"
                v.confidence = 0.95
                v.reader = "anndata.read_loom(path)  # or loompy.connect(path)"
                v.evidence.append("root layout matrix + row_attrs + col_attrs (loom specification)")
                if "LOOM_SPEC_VERSION" in root_attrs:
                    v.evidence.append(f"LOOM_SPEC_VERSION={root_attrs['LOOM_SPEC_VERSION']}")
                shape = _h5_shape(f, "matrix")
                if shape:
                    v.details["shape"] = shape
                    v.details["axis_order"] = "genes_x_cells (loom stores rows=genes)"
                return

            # 10x CellRanger v3: CSC arrays under /matrix with a features group.
            if "matrix" in root_keys and isinstance(f.get("matrix"), h5py.Group):
                sub = set(f["matrix"].keys())
                if {"data", "indices", "indptr", "shape", "barcodes"}.issubset(sub):
                    v.format = "10x_h5"
                    v.confidence = 0.96
                    v.reader = "scanpy.read_10x_h5(path)"
                    v.evidence.append(
                        "/matrix holds CSC arrays data/indices/indptr/shape plus barcodes (CellRanger v3)"
                    )
                    v.details["matrix_keys"] = sorted(sub)
                    shape = _read_shape_dataset(f, "matrix/shape")
                    if shape:
                        v.details["shape"] = shape
                        v.details["axis_order"] = "genes_x_cells (10x stores features x barcodes)"
                    if "features" in sub:
                        v.details["feature_keys"] = sorted(f["matrix"]["features"].keys())
                    return

            # 10x CellRanger v2: the same arrays one level down under a genome name.
            for key in root_keys:
                node = f.get(key)
                if isinstance(node, h5py.Group) and {"data", "indices", "indptr", "shape", "barcodes"}.issubset(
                    set(node.keys())
                ):
                    v.format = "10x_h5"
                    v.confidence = 0.9
                    v.reader = "scanpy.read_10x_h5(path)"
                    v.evidence.append(f"CSC arrays under genome group '/{key}' (CellRanger v2 layout)")
                    v.details["genome"] = key
                    shape = _read_shape_dataset(f, f"{key}/shape")
                    if shape:
                        v.details["shape"] = shape
                    return

            # Stereo-seq ships the binary half of its format as `.gef`: bin-level counts under
            # `geneExp`, cell-level under `cellBin`. `spatial_pipeline` already lists `.gef` among
            # its Stereo-seq extensions, so reporting `hdf5_unknown` withheld a name the system had.
            gef_root = next((k for k in ("geneExp", "cellBin") if k in root_keys), None)
            if gef_root is not None:
                _classify_stereoseq_gef(v, f, root_keys, gef_root)
                return

            v.format = "hdf5_unknown"
            v.confidence = 0.35
            v.evidence.append(f"HDF5 container with root keys {root_keys} -- no h5ad, loom or 10x marker found")
            v.reader = "h5py.File(path) and inspect the groups listed in details.root_keys"
    except OSError as exc:
        # A half-copied h5ad keeps its magic bytes and loses its superblock. This is the common
        # real failure and it must not surface as a traceback.
        v.format = "hdf5_unreadable"
        v.confidence = 0.3
        v.evidence.append(f"HDF5 signature present but the file could not be opened (truncated or corrupt): {exc}")
        v.warnings.append("file looks truncated -- compare its size against the source before re-reading")
        v.reader = None
    except Exception as exc:
        v.format = "hdf5_unreadable"
        v.confidence = 0.3
        v.evidence.append(f"HDF5 signature present but the layout could not be read: {type(exc).__name__}: {exc}")
        v.reader = None


def _describe_h5ad(v: FormatVerdict, f: Any) -> None:
    """Record the parts of an h5ad a spatial caller needs, without loading X."""
    try:
        obsm_keys = sorted(f["obsm"].keys()) if "obsm" in f else []
        v.details["obsm_keys"] = obsm_keys
        v.details["has_spatial"] = "spatial" in obsm_keys
        if "obs" in f:
            v.details["obs_keys"] = sorted(f["obs"].keys())[:25]
        if "uns" in f:
            v.details["uns_keys"] = sorted(f["uns"].keys())[:25]
        shape = _h5_shape(f, "X")
        if shape:
            v.details["shape"] = shape
        elif "X" in f:
            attrs = dict(f["X"].attrs) if hasattr(f["X"], "attrs") else {}
            raw = attrs.get("shape")
            if raw is not None:
                v.details["shape"] = [int(x) for x in raw]
        if v.details.get("has_spatial"):
            v.evidence.append("obsm['spatial'] present -- coordinates already attached")
    except Exception as exc:
        v.warnings.append(f"h5ad opened but its sub-groups could not be summarised: {exc}")


def _h5_shape(f: Any, key: str) -> list[int] | None:
    node = f.get(key)
    if node is None:
        return None
    shape = getattr(node, "shape", None)
    if shape is not None and len(shape) == 2:
        return [int(shape[0]), int(shape[1])]
    return None


def _read_shape_dataset(f: Any, key: str) -> list[int] | None:
    try:
        return [int(x) for x in f[key][()]]
    except Exception:
        return None


def _scalar(value: Any) -> Any:
    """h5py attributes come back as numpy scalars/bytes; make them JSON-safe."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", ()) == ():
        try:
            out = item()
            return out.decode("utf-8", "replace") if isinstance(out, bytes) else out
        except Exception:
            return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    return value


def _decompress_head(p: Path, want: int) -> tuple[bytes, bool]:
    """Decompress up to ``want`` bytes, keeping whatever decodes if the stream breaks.

    Returns ``(bytes, cut_short)``, where ``cut_short`` means the input ran out before the
    end-of-stream marker -- an interrupted download, a killed rsync, a job that died mid-write.

    ``gzip.open(...).read(n)`` cannot do this. It raises `EOFError` on a truncated member and hands
    back nothing, so every byte that did decode is lost with the exception; and `EOFError` descends
    from `Exception`, not `OSError`, so it escaped the caller's handler and came straight out of
    `probe_format` -- the one thing this module promises never to do. Feeding the raw file to a
    decompressor instead means truncation is not an error at all: the output simply stops, and
    ``eof`` says whether it stopped because the stream ended or because the file did.
    """
    out = bytearray()
    decomp = zlib.decompressobj(_GZIP_WBITS)
    cut_short = False
    with p.open("rb") as raw:
        while len(out) < want:
            block = raw.read(_GZIP_CHUNK_BYTES)
            if not block:
                cut_short = not decomp.eof
                break
            out += decomp.decompress(block, want - len(out))
            # `cat a.gz b.gz` and bgzip -- which every indexed genomics file uses -- both produce
            # gzip files of many members. Stopping at the first one would read a bgzipped table 64
            # KB at a time and call the rest truncated.
            while decomp.eof and len(out) < want and decomp.unused_data:
                decomp, tail = zlib.decompressobj(_GZIP_WBITS), decomp.unused_data
                out += decomp.decompress(tail, want - len(out))
    return bytes(out), cut_short


def _looks_like_tar(blob: bytes) -> bool:
    """True when these bytes open with a tar header block.

    A tar is the one archive format with nothing at offset 0 to recognise it by -- the first 100
    bytes are the first member's filename -- so every offset-0 check misses it and the file reads
    as binary noise whose leading bytes happen to spell a path.
    """
    return blob[_TAR_MAGIC_OFFSET : _TAR_MAGIC_OFFSET + len(_TAR_MAGIC)] == _TAR_MAGIC


def _tar_member_names(blob: bytes) -> tuple[list[str], bool]:
    """Member names read out of the header blocks in ``blob``, and whether that is all of them.

    Walks the archive's own block structure rather than calling ``tarfile.getnames()``, which
    streams to the end: on a 40 GB Xenium tarball that is the whole file decompressed to answer a
    question about its first few entries. Everything here is bounded by the sample already in hand,
    and a header that does not parse ends the walk rather than raising -- these names are evidence.

    The second return value is what keeps the bound honest. A 120 MB first member pushes the second
    header past the sample, so the walk sees one name in a two-member archive; reporting that as
    ``n_members: 1`` is a count of what was read presented as a count of what is there. True only
    when the archive's own end-of-file marker -- a zero block -- was reached.
    """
    names: list[str] = []
    offset = 0
    while offset + _TAR_BLOCK <= len(blob):
        header = blob[offset : offset + _TAR_BLOCK]
        if header[_TAR_MAGIC_OFFSET : _TAR_MAGIC_OFFSET + len(_TAR_MAGIC)] != _TAR_MAGIC:
            # A zero block is tar's end-of-archive marker and means the list is complete. Anything
            # else here is a header shape not parsed (a GNU long name, a PAX extension), which
            # stops the walk without telling us where the archive ends.
            return names, not any(header)
        name = header[:100].split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()
        try:
            # Octal, space- or NUL-terminated. A size that will not parse means the next block
            # boundary is unknown, so the walk cannot continue past this entry.
            size = int(header[124:136].split(b"\0", 1)[0].strip() or b"0", 8)
        except ValueError:
            return names, False
        if name and name != "pax_global_header":
            names.append(name)
        if len(names) >= _TAR_MAX_MEMBERS:
            return names, False
        offset += _TAR_BLOCK + ((size + _TAR_BLOCK - 1) // _TAR_BLOCK) * _TAR_BLOCK
    return names, False


def _describe_tar(v: FormatVerdict, blob: bytes, *, compressed: bool) -> None:
    """Record the archive and what is in it. Populates ``v`` in place.

    The member names are the entire diagnostic content of an archive -- `outs/spatial/` says Visium
    as plainly as the directory would -- so they are what the verdict carries. No search hint is
    attached: the format is identified, and offering the first member's filename as a magic number
    to look up is what the unrecognised path was doing wrong.
    """
    names, complete = _tar_member_names(blob)
    v.container = "tar"
    v.format = "tar_archive"
    v.confidence = 0.9
    v.reader = "extract the archive, then re-probe the extracted directory"
    v.evidence.append("gzip member is a tar archive" if compressed else "tar header magic 'ustar' at offset 257")
    if not names:
        return
    v.details["members"] = names
    v.details["members_complete"] = complete
    # Only a completed walk knows how many members there are. Naming the key `n_members` for a
    # partial read would be read as the archive's size by anything that consumes it.
    if complete:
        v.details["n_members"] = len(names)
    shown = ", ".join(names[:3])
    entries = "entry" if len(names) == 1 else "entries"
    caveat = "" if complete else f" (first {len(names)} {entries}; the sample stops before the end of the archive)"
    v.evidence.append(f"tar archive containing {shown}{caveat}")


def _probe_gzip_member(v: FormatVerdict, p: Path) -> None:
    """Decompress only the head of the member and re-classify it."""
    # One byte past the sample, then trimmed back: whether the member continues is answered by the
    # read that hit the limit. The compressed file's `st_size` says nothing about the length of
    # what comes out of it, and the previous caller passed `size=len(inner)` -- the sample's own
    # length -- so no gzipped file of any size has ever reported a truncated sample, and the
    # partial-final-line trim below never ran.
    try:
        inner, cut_short = _decompress_head(p, _TEXT_SAMPLE_BYTES + 1)
    except (OSError, zlib.error) as exc:
        v.format = "gzip_unreadable"
        v.confidence = 0.3
        v.evidence.append(f"gzip signature present but the member could not be decompressed: {exc}")
        return

    if cut_short and not inner:
        v.format = "gzip_unreadable"
        v.confidence = 0.3
        v.evidence.append("gzip signature present but the member ends before any data decompresses")
        return
    if cut_short:
        v.warnings.append(
            f"This gzip member is truncated: it ends before its end-of-stream marker. Everything "
            f"below is read from the {len(inner)} bytes that did decompress, so it describes the "
            f"start of the file and not the whole of it. Re-fetch the file before using it."
        )

    # A stream that broke is truncated whether or not it reached the sample limit, and the line it
    # broke in the middle of is partial either way -- which is exactly what this flag is for.
    inner_truncated = cut_short or len(inner) > _TEXT_SAMPLE_BYTES
    inner = inner[:_TEXT_SAMPLE_BYTES]

    if not inner:
        v.format = "empty_file"
        v.evidence.append("gzip member is empty")
        return

    if inner.startswith(_RDS_GZIP_INNER):
        v.format = "r_object"
        v.confidence = 0.85
        v.reader = "R: readRDS(path)  (no pure-Python reader)"
        v.evidence.append("gzip member starts with the R serialization header 'X\\n' (RDS)")
        return
    if inner.startswith(_HDF5_MAGIC):
        v.format = "hdf5_gzipped"
        v.confidence = 0.8
        v.evidence.append("gzip member is itself an HDF5 file; decompress before reading")
        return

    # `.tar.gz` is how a Space Ranger run arrives from GEO, and it used to land in the `unknown`
    # branch below with its evidence reading "leading bytes (hex): 6f 75 74 73 2f ..." -- which
    # spells `outs/` and is the answer, in the one encoding a search engine cannot match.
    if _looks_like_tar(inner):
        _describe_tar(v, inner, compressed=True)
        return

    text = _decode_text(inner)
    if text is None:
        v.format = "unknown"
        v.confidence = 0.15
        v.evidence.append(f"gzip member leading bytes (hex): {_hexdump(inner[:16])}")
        v.evidence.append("gzip member is binary and matches no known signature")
        return

    declares_gzip = p.name.lower().endswith(".gz")
    inner_name = p.name[:-3] if declares_gzip else p.name
    _classify_text(v, Path(inner_name), text, sampled_bytes=len(inner), truncated=inner_truncated)
    v.details["compression"] = "gzip"
    v.evidence.append("classification above is of the decompressed gzip member")

    v.reader = _reader_through_gzip(v.reader)
    if not declares_gzip:
        v.warnings.append(
            "This file is gzip-compressed but its name does not end in '.gz'. Every reader that "
            "infers compression from the filename -- pandas, scipy.io.mmread, scanpy's loaders -- "
            "will try to parse the compressed bytes as text and fail on them. The reader below "
            "names the compression explicitly; anything else reading this path needs telling too."
        )


# The member was re-classified, so every reader below was written for the *decompressed* content
# while the bytes on disk stay compressed. On a `.gz` name that went unnoticed because pandas infers
# compression from the extension; on a file gzipped in place it handed back a reader that raises
# `UnicodeDecodeError` on the file the probe had just read successfully. Naming the compression
# outright is correct either way, and does not depend on what the file happens to be called.
_GZIP_READERS = {
    "json.load(open(path))": "json.load(gzip.open(path, 'rt'))",
    "open(path).read()": "gzip.open(path, 'rt').read()",
    "scipy.io.mmread(path)": "scipy.io.mmread(gzip.open(path, 'rb'))",
    "Bio.SeqIO.parse(path, 'fasta')": "Bio.SeqIO.parse(gzip.open(path, 'rt'), 'fasta')",
}


def _reader_through_gzip(reader: str | None) -> str | None:
    """Rewrite a reader written for decompressed content so it opens the compressed file."""
    if reader is None:
        return None
    if reader.startswith("pandas.read_csv(") and reader.endswith(")"):
        return f"{reader[:-1]}, compression='gzip')"
    return _GZIP_READERS.get(reader, reader)


def _probe_zip_members(v: FormatVerdict, p: Path) -> None:
    import zipfile

    try:
        with zipfile.ZipFile(str(p)) as z:
            names = z.namelist()
    except Exception as exc:
        v.format = "zip_unreadable"
        v.confidence = 0.3
        v.evidence.append(f"zip signature present but the archive could not be listed: {exc}")
        return

    v.details["members"] = names[:50]
    v.details["n_members"] = len(names)
    v.evidence.append(f"zip archive with {len(names)} members, first: {names[:5]}")

    # Some formats *are* a zip with a required member, and stopping at "the bytes are a zip" is how
    # the commonest file a collaborator sends -- an .xlsx of markers or sample metadata -- came back
    # as an archive to unpack. The member list says which format it is, so ask before falling
    # through. Paths are matched against the archive root and against a single shared top-level
    # directory, because `zip -r store.zarr.zip store.zarr` nests everything one level down.
    rooted, prefix = _members_at_archive_root(names)
    if _detect_ooxml(v, rooted) or _detect_zipped_zarr(v, p, rooted, prefix):
        return

    v.format = "zip_archive"
    v.confidence = 0.85
    v.reader = "extract the archive, then re-probe the extracted directory"
    lowered = {n.lower().rsplit("/", 1)[-1] for n in names}
    if lowered & set(_MTX_NAMES) and lowered & set(_BARCODE_NAMES):
        v.details["inner_layout"] = "10x_mtx"
        v.evidence.append("archive contains a 10x MTX triplet")


def _classify_ole2(v: FormatVerdict, p: Path) -> None:
    """Name a pre-2007 Office file. GEO supplementary tables still ship as `.xls`."""
    v.container = "binary"
    v.confidence = 0.9
    v.evidence.append("OLE2 compound-file signature D0CF11E0 at offset 0 (pre-2007 Microsoft Office)")
    if p.suffix.lower() not in (".xls", ".xlt"):
        v.format = "ole2_compound"
        v.reader = None
        v.evidence.append(
            f"extension {p.suffix.lower()!r} does not say spreadsheet, and which Office format this is "
            "cannot be read from the header alone -- .doc, .ppt and .msg share it"
        )
        return
    v.format = "xls"
    v.reader = "pandas.read_excel(path)"
    v.warnings.append(
        "This is the legacy .xls format, not .xlsx. pandas reads it through xlrd, which is an "
        "optional dependency -- 'pip install xlrd' if read_excel raises ImportError."
    )


def _members_at_archive_root(names: list[str]) -> tuple[set[str], str]:
    """Member paths relative to what the archive actually holds, plus the prefix that was stripped.

    An archive of a store built by ``shutil.make_archive`` has its markers at the root; the same
    store zipped with ``zip -r`` has them under one directory named after it. Both are the same
    store, so a single shared top-level directory is stripped -- and only when it is shared by
    every member, which is what distinguishes "one thing was zipped" from "several were".

    The prefix comes back because a reader has to name it: zarr opens a nested store only when
    told the path *inside* the archive, so dropping it here would hand back a reader that raises.
    """
    # `removeprefix`, not `lstrip`: lstrip strips *characters*, so it ate the leading dot off
    # `.zgroup` and `.zattrs` and made every zarr v2 store invisible to the marker check below.
    cleaned = [n.removeprefix("./") for n in names if n and not n.endswith("/")]
    tops = {n.split("/", 1)[0] for n in cleaned if "/" in n}
    if len(tops) == 1 and all("/" in n for n in cleaned):
        top = tops.pop()
        return {n[len(top) + 1 :].lower() for n in cleaned}, f"{top}/"
    return {n.lower() for n in cleaned}, ""


# Every OOXML package declares its parts in `[Content_Types].xml`; which Office format it is shows
# in the one top-level directory that holds the payload. Naming a .docx is worth as much as naming
# a .xlsx -- a caller that was handed one instead of data finds out here rather than three steps on.
_OOXML_PARTS = {
    "xl/": ("xlsx", "an Excel workbook"),
    "word/": ("docx", "a Word document"),
    "ppt/": ("pptx", "a PowerPoint presentation"),
}


def _detect_ooxml(v: FormatVerdict, rooted: set[str]) -> bool:
    """Recognise the Office formats, which are zips with a content-types manifest."""
    if "[content_types].xml" not in rooted:
        return False
    for part, (fmt, label) in _OOXML_PARTS.items():
        if not any(n.startswith(part) for n in rooted):
            continue
        v.format = fmt
        v.confidence = 0.95
        v.evidence.append(f"'[Content_Types].xml' plus a '{part}' part -- {label}, not an archive of files")
        if fmt != "xlsx":
            v.reader = None
            v.warnings.append(
                f"This is {label}. It is not a data file: there is no table here to read into a "
                "DataFrame. Check whether the intended file was sent alongside it."
            )
            return True

        sheets = sum(1 for n in rooted if n.startswith("xl/worksheets/sheet"))
        v.details["n_sheets"] = sheets
        if sheets > 1:
            # `read_excel` with no `sheet_name` returns the first sheet and says nothing about the
            # others, so half a manifest reads as the whole of it.
            v.reader = "pandas.read_excel(path, sheet_name=None)  # dict of {sheet name: DataFrame}"
            v.warnings.append(
                f"This workbook has {sheets} sheets. pandas.read_excel(path) would return only the "
                "first and drop the rest silently; sheet_name=None returns all of them."
            )
        else:
            v.reader = "pandas.read_excel(path)"
        return True
    return False


def _detect_zipped_zarr(v: FormatVerdict, p: Path, rooted: set[str], prefix: str) -> bool:
    """Recognise a zarr store shipped as one `.zip` -- zarr reads it in place, through a ZipStore."""
    marker = next((m for m in (".zgroup", ".zarray", "zarr.json") if m in rooted), None)
    if marker is None:
        return False
    v.format = "zarr"
    v.confidence = 0.9
    v.details["zarr_format"] = 2 if marker != "zarr.json" else 3
    inner = f', path="{prefix.rstrip("/")}"' if prefix else ""
    v.reader = f'zarr.open(store=zarr.storage.ZipStore(path, mode="r"){inner})'
    v.evidence.append(f"Zarr store marker {marker!r} at the archive root -- a zipped store, not an archive of files")
    if prefix:
        v.details["store_path_in_archive"] = prefix.rstrip("/")
        v.evidence.append(f"the store sits under {prefix.rstrip('/')!r} inside the archive, and the reader says so")

    groups = sorted(g for g in _SPATIALDATA_GROUPS if any(n.startswith(f"{g}/") for n in rooted))
    if groups:
        v.details["spatialdata_groups"] = groups
        v.evidence.append(f"SpatialData group names present: {groups}")
    if _zipped_zarr_declares_spatialdata(p, f"{prefix}{marker}", f"{prefix}.zattrs"):
        v.evidence.append(f"the store's own {marker!r} carries 'spatialdata_attrs'")
        groups = groups or ["declared"]
    if groups:
        # spatialdata.read_zarr wants a store on disk, so the zip has to be unpacked for it --
        # unlike zarr itself. Saying which reader needs which is the point of the note.
        v.warnings.append(
            "This is a SpatialData store. The reader above opens it read-only through zarr; "
            "spatialdata.read_zarr needs the store unpacked to a directory first."
        )
    return True


def _zipped_zarr_declares_spatialdata(p: Path, *candidates: str) -> bool:
    """True when the zipped store's own root metadata says it is SpatialData.

    The candidates are full member paths, not basenames: every group in the store has a file by
    the same name, and matching on the basename would answer from whichever sub-group the archive
    happened to list first. `_json_declares_spatialdata` decides what counts as a declaration, so
    this and the on-disk probe cannot drift apart. Never raises -- unreadable metadata is one fewer
    clue, not a failure.
    """
    import zipfile

    try:
        with zipfile.ZipFile(str(p)) as z:
            actual = {n.removeprefix("./").lower(): n for n in z.namelist()}
            for candidate in candidates:
                name = actual.get(candidate.lower())
                if name is None:
                    continue
                with z.open(name) as fh:
                    if _json_declares_spatialdata(fh.read(_HEAD_BYTES)):
                        return True
    except (OSError, ValueError, KeyError, RuntimeError):
        return False
    return False


def _classify_text(v: FormatVerdict, p: Path, text: str, sampled_bytes: int, truncated: bool) -> None:
    """Recognise structured text formats, then fall back to delimiter sniffing.

    ``sampled_bytes`` is how many bytes went into ``text`` and ``truncated`` whether more content
    follows it. Both are facts about the *read*, so both are supplied by the caller that performed
    it. This function used to take the file's total size instead and derive truncation as
    ``size > len(text)`` -- a byte count against a character count, true for any file holding a
    non-ASCII byte however completely it had been read.
    """
    stripped = text.lstrip()

    if stripped.startswith("%%MatrixMarket"):
        v.format = "matrix_market"
        v.confidence = 0.97
        v.reader = "scipy.io.mmread(path)"
        banner = stripped.split("\n", 1)[0]
        v.evidence.append(f"Matrix Market banner: {banner.strip()}")
        dims = _matrix_market_dims(stripped)
        if dims:
            v.details["shape"] = dims[:2]
            v.details["n_entries"] = dims[2] if len(dims) > 2 else None
        return

    if stripped[:1] in "{[":
        try:
            # `stripped` unconditionally: `json.loads` skips leading whitespace itself, so the
            # `text`/`stripped` choice this used to make on `len(text) < size` picked between two
            # identical parses.
            json.loads(stripped)
            v.format = "json"
            v.confidence = 0.9
            v.reader = "json.load(open(path))"
            v.evidence.append("parses as JSON")
            return
        except ValueError:
            # A large JSON file is truncated by the sample, so failing to parse proves nothing.
            # A *complete* one that fails to parse is not JSON, and must fall through to the
            # sniffer rather than be reported as JSON that no reader can open.
            if truncated:
                v.format = "json"
                v.confidence = 0.6
                v.reader = "json.load(open(path))"
                v.evidence.append("starts with '{' or '[' (sample truncated, so not fully validated)")
                return

    if stripped.startswith("<?xml") or stripped.startswith("<OME"):
        v.format = "xml"
        v.confidence = 0.85
        v.evidence.append("XML declaration at the start of the file")
        return

    if stripped.startswith(">") and _looks_like_fasta(stripped):
        v.format = "fasta"
        v.confidence = 0.85
        v.reader = "Bio.SeqIO.parse(path, 'fasta')"
        v.evidence.append("FASTA '>' record header followed by sequence lines")
        return

    _sniff_delimited(v, p, text, sampled_bytes, truncated)


def _matrix_market_dims(text: str) -> list[int] | None:
    for line in text.split("\n")[1:]:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if all(part.isdigit() for part in parts) and len(parts) in (2, 3):
            return [int(x) for x in parts]
        return None
    return None


def _looks_like_fasta(text: str) -> bool:
    lines = [ln for ln in text.split("\n")[1:6] if ln.strip()]
    if not lines:
        return False
    alphabet = set("ACGTUNacgtunRYKMSWBDHVrykmswbdhv-*")
    return all(set(ln.strip()) <= alphabet for ln in lines[:3])


def _parse_delimited(lines: list[str], delim: str) -> list[list[str]]:
    """The fields each line yields under `delim`, with quoting honoured.

    `str.split` and `str.count` read a line as flat characters, so everything inside a quoted field
    -- the commas in a description, the spaces in a label -- is scored as structure. Parsing instead
    of scanning is what lets a quoted region be recognised as one field.
    """
    try:
        return list(csv.reader(lines, delimiter=delim))
    except csv.Error:
        # A NUL byte or a field past the 128 KB limit. The flat read is still a usable answer, and
        # this decision has a `< 0.6` consistency floor under it to catch the case where it is not.
        return [ln.split(delim) for ln in lines]


def _residual_structure(rows: list[list[str]], others: Iterable[str]) -> float:
    """The share of parsed rows in which a structural character survived inside a field.

    A character counts as structural when it cannot plausibly remain inside a field that was cut in
    the right place, and exactly two qualify:

    * `"`, for every candidate. `csv.reader` consumes the quotes of a field it parsed as quoted, so
      one still sitting in the output is direct evidence the split ran through a quoted region.

    * any other viable delimiter, but **only when the candidate being judged is the space**. Space
      is the one candidate that occurs freely inside ordinary field content -- gene descriptions,
      cell-type names, free text -- so a space split that leaves tabs stranded in its fields is
      subdividing another delimiter's columns rather than finding the table's own.

    That restriction is not squeamishness, it is what the reference implementations support. Asked
    about `a<TAB>b,c,d,e`, both `csv.Sniffer` and `pandas(sep=None)` answer comma, agreeing with the
    field-count tie-break already documented below -- so a rule that made a surviving tab disqualify
    the comma would overturn a decision two independent implementations get right. Asked about a
    `features.tsv` whose descriptions contain commas, `pandas` answers tab. Between two real
    delimiters the field count still decides; only the space is held to this.

    `others` is restricted by the caller to delimiters that are themselves viable for this file, so
    a stray tab in a file that is not tab-delimited accuses nobody.
    """
    if not rows:
        return 0.0
    marks = ('"', *others)
    return sum(1 for row in rows if any(m in cell for cell in row for m in marks)) / len(rows)


def _sniff_delimited(v: FormatVerdict, p: Path, text: str, sampled_bytes: int, truncated: bool) -> None:
    """Detect delimiter, header and index column from a sample of the head of the file.

    Deliberately does not use ``csv.Sniffer``: it raises on the single-column case and picks
    delimiters out of quoted gene descriptions. Counting consistent field splits is both cheaper
    and more predictable to explain in the evidence.
    """
    # A '#' preamble is the norm in this field -- Stereo-seq GEM, GTF, VCF, BED all open with one --
    # and letting those lines through corrupted two separate answers at once: they voted in the
    # delimiter count (a GTF's space-separated `#!genome-build` line outvoted its tabs), and line
    # one became the header row, so a GEM came back named `#FileFormat=GEMv0.1` with its real header
    # demoted to data. Excluded from both, then handed back in `details` and named in the reader.
    raw = text.split("\n")
    skiprows = _leading_comment_lines(raw)
    body = raw[skiprows:]
    if not any(ln.strip() for ln in body):
        # Nothing but preamble. Skipping it all would turn a describable file into `unknown`.
        skiprows, body = 0, raw

    lines = [ln for ln in body if ln.strip()][:200]
    # Reported, not reconstructed. `len(text.encode("utf-8"))` would be wrong for the latin-1
    # fallback in `_decode_text`, which is exactly the case where the count matters.
    v.details["bytes_sampled"] = sampled_bytes
    if not lines:
        v.format = "unknown"
        v.confidence = 0.1
        v.evidence.append("file decodes as text but contains no non-blank lines")
        return

    # A partial final line from a fixed-size read would skew the field counts. Only when the read
    # really did stop short: this used to discard the last row of any complete table that happened
    # to contain a non-ASCII character.
    #
    # `> 1`, not `> 2`. At exactly two sampled lines the fragment was kept and counted as a row, and
    # one whole row disagreeing with one partial one is 50% consistency -- under the 0.6 floor
    # below. So a 6001-column CSV whose first two lines straddle the 64 KB read came back
    # `plain_text`, `delimiter: None`, "single-column or free text", with `open(path).read()` handed
    # over as the reader: a table declared unreadable because of where the read happened to stop.
    # Dropping the fragment leaves the header, which is enough to describe the table.
    #
    # When the header itself is what got cut there is no whole line to fall back on, and the one
    # thing that cannot be recovered is how many columns the file has. Recorded here so the count
    # can be withheld rather than guessed from the fragment.
    header_truncated = truncated and len(lines) == 1
    if len(lines) > 1 and truncated:
        lines = lines[:-1]

    # Every candidate is parsed once, up front, and both the ranking below and the header decision
    # further down read that one parse. Counting separators and then splitting the line again is how
    # the report came to contradict itself: `n_columns` was the modal count over all sampled lines
    # and `column_names` a fresh split of line 0, so a four-column `quoted_xy.csv` whose body rows
    # carry a comma its header does not reported `n_columns=5` beside four `column_names`. Sharing
    # the parse makes the two agree by construction rather than by both happening to be right.
    parsed: dict[str, list[list[str]]] = {}
    counted: dict[str, tuple[float, int]] = {}
    for delim in _DELIMITER_CANDIDATES:
        rows = _parse_delimited(lines, delim)
        counts = [len(r) for r in rows]
        if not counts or max(counts) < 2:
            continue
        modal = max(set(counts), key=counts.count)
        if modal < 2:
            continue
        parsed[delim] = rows
        counted[delim] = (counts.count(modal) / len(counts), modal)

    # Whether the split left structure behind outranks both, because it is evidence about the split
    # itself rather than about how tidy the result looks. The two keys under it could not settle
    # these: a delimiter occurring only inside a quoted phrase is perfectly consistent, and "more
    # fields" then hands it the win -- a two-column CSV whose second column is a quoted phrase came
    # back `delimiter=space, n_columns=6, has_header=false`. Nor is that only about quoting: on the
    # `features.tsv` named below, tab, comma and space all fit every line, so the tie-break decided
    # it on field count alone and space won with seven spurious columns, in the exact file this
    # comment says the consistency rule fixed. Neither file leaves room for doubt -- the residue is
    # 0% for the right delimiter and 100% for the wrong ones.
    #
    # A file with no quotes, and one where the space is not in contention, scores 0.0 here for every
    # candidate, so the tuple degenerates to the two keys below and is decided exactly as before.
    best_rank: tuple[float, float, int] = (-1.0, 0.0, 0)
    best_delim, best_consistency, n_fields = None, 0.0, 1
    for delim, (consistency, modal) in counted.items():
        competing = [o for o in counted if o != delim] if delim == " " else []
        residue = _residual_structure(parsed[delim], competing)
        # Consistency dominates the field count; field count breaks ties between equally consistent
        # delimiters. Ranking on separate keys is what makes that true. Folding them into one number
        # -- `consistency * (1 + fields/100)` -- let the tie-break overturn the thing it is
        # documented to defer to: on a two-column `features.tsv` whose descriptions contain commas,
        # space fit 98% of lines and found eight spurious columns, which outscored the tab that fit
        # all of them. That same number was then printed as a share of lines, so the evidence read
        # "102% of 51 sampled lines". Rounding the consistency keeps a real tie a tie without making
        # float equality load-bearing; capping the field count preserves the old rule that past 50
        # fields, more fields stop helping.
        rank = (-round(residue, 2), round(consistency, 3), min(modal, 50))
        if rank > best_rank:
            best_rank, best_delim, best_consistency, n_fields = rank, delim, consistency, modal

    if best_delim is None or best_consistency < 0.6:
        v.format = "plain_text"
        v.confidence = 0.4
        v.details["delimiter"] = None
        v.details["n_lines_sampled"] = len(lines)
        v.evidence.append("text with no consistent delimiter across sampled lines (single-column or free text)")
        v.reader = "open(path).read()"
        return

    header_row = parsed[best_delim][0]
    # Its last field is the half of a name the read stopped inside. Everything before it is whole.
    if header_truncated and len(header_row) > 1:
        header_row = header_row[:-1]
    body_rows = parsed[best_delim][1:20] or [header_row]

    has_header = _detect_header(header_row, body_rows)
    index_column = _detect_index_column(header_row, body_rows, has_header)

    v.format = "delimited_text"
    v.confidence = 0.85 if best_consistency >= 0.9 else 0.7
    v.details.update(
        {
            "delimiter": best_delim,
            "delimiter_name": {"\t": "tab", ",": "comma", ";": "semicolon", "|": "pipe", " ": "space"}[best_delim],
            # Withheld when the only line sampled was a fragment: see the evidence line below.
            "n_columns": None if header_truncated else n_fields,
            "has_header": has_header,
            "index_column": index_column,
            "n_lines_sampled": len(lines),
            "column_names": [c.strip().strip('"') for c in header_row[:30]] if has_header else None,
            "truncated_sample": truncated,
        }
    )
    if skiprows:
        v.details["comment_lines"] = skiprows
        # Kept, not just counted: a GEM preamble carries OffsetX/OffsetY, and without those the
        # bin coordinates in the table cannot be placed on the chip at all.
        v.details["preamble"] = [ln.strip()[:200] for ln in raw[:skiprows] if ln.strip()][:20]
        v.evidence.append(
            f"{skiprows} '#' preamble line(s) precede the table; excluded from the delimiter and "
            "header decisions above, and kept in details['preamble']"
        )
    if header_truncated:
        # A matrix whose header alone runs past the sample. The fragment's field count used to be
        # published as `n_columns` with nothing marking it as partial, and it understates badly:
        # three shipped DepMap CSVs reported 4639 and 4789 columns against true widths of 17917 and
        # 19206, a factor of four. The delimiter is plain from 64 KB of a header and the names that
        # were read are whole, so those stand; only the total is withheld, because only the total is
        # unknown. The count is not worth a wider read -- the agent reading this verdict needs to
        # know the file is a wide matrix, which the missing count and this line say between them.
        v.evidence.append(
            f"delimiter {v.details['delimiter_name']!r} read from the first line, but that line runs "
            f"past the {sampled_bytes} bytes sampled, so the column count is not known from it "
            f"(at least {n_fields} columns; column_names holds those read whole)"
        )
    else:
        v.evidence.append(
            f"delimiter {v.details['delimiter_name']!r} gives a consistent {n_fields} fields on "
            f"{best_consistency:.0%} of {len(lines)} sampled lines"
        )
    v.evidence.append(
        f"header row {'present' if has_header else 'absent'}; "
        f"{'first column is a row index' if index_column == 0 else 'no index column detected'}"
    )
    if _header_evidence_is_absent(header_row, body_rows):
        # Nothing in an all-text table separates a header from a first data row, so `has_header`
        # above is a default rather than a finding. Say so, and hand over the row itself: a caller
        # that can see `gene_id, symbol, description` settles this instantly, and one that cannot
        # is no worse off than before.
        v.details["first_row"] = [c.strip().strip('"') for c in header_row[:30]]
        v.warnings.append(
            "Every sampled cell in the body of this table is text, so there is no numeric contrast "
            f"to tell a header row from a first data row: reported has_header={has_header} is a "
            "default, not a finding. The first row is in details['first_row'] -- if those are "
            + (
                "not column names, read with header=None; the reader string below assumes they are."
                if has_header
                else "column names, read with header=0; the reader string below assumes they are not."
            )
        )
    # `skiprows`, not `comment='#'`: skiprows counts physical lines off the top and cannot reach a
    # '#' that appears mid-row in somebody's data ("lot #3 of 9"), which `comment` would truncate.
    skip_arg = f", skiprows={skiprows}" if skiprows else ""
    index_arg = ", index_col=0" if index_column == 0 else ""
    header_arg = "" if has_header else ", header=None"
    v.reader = f"pandas.read_csv(path, sep={best_delim!r}{skip_arg}{index_arg}{header_arg})"


def _leading_comment_lines(raw: list[str]) -> int:
    """How many physical lines at the top of the file form a '#' preamble.

    Physical lines, blanks inside the block included, because that is what `skiprows` counts.
    Returns 0 unless at least one line in the block is actually a comment: a file that merely opens
    with a blank line needs no help, and saying `skiprows=1` for it would be noise dressed as a
    finding. Only a contiguous block at the very top counts -- a '#' further down is data.
    """
    n, seen_comment = 0, False
    for line in raw:
        stripped = line.strip()
        if stripped.startswith("#"):
            seen_comment = True
        elif stripped:
            break
        n += 1
    return n if seen_comment else 0


def _numeric_fraction(cells: list[str]) -> float:
    cells = [c for c in cells if c.strip() != ""]
    if not cells:
        return 0.0
    return sum(_is_numeric(c) for c in cells) / len(cells)


def _header_evidence_is_absent(header_row: list[str], body_rows: list[list[str]]) -> bool:
    """True when `_detect_header`'s rule had nothing to read.

    That rule is "a header row is non-numeric where the body is numeric". A body with no numeric
    cells in it -- a features table, a sample sheet, an all-categorical metadata export -- offers
    no contrast, so the rule cannot fire and returns `False` for want of evidence rather than
    because it judged. The caller uses this to hedge instead of asserting.
    """
    if not body_rows or body_rows == [header_row]:
        return False
    # A leading empty cell is the pandas index-name convention, which `_detect_header` reads as a
    # header for sure: that IS a finding, and hedging it told the agent to re-read with header=None
    # and turn the column names into a data row (u15-validation-8).
    if header_row and header_row[0].strip() in ("", '""'):
        return False
    return all(_numeric_fraction(r) == 0.0 for r in body_rows)


def _detect_header(header_row: list[str], body_rows: list[list[str]]) -> bool:
    """A header row is non-numeric where the body is numeric."""
    if not body_rows or body_rows == [header_row]:
        return not all(_is_numeric(c) for c in header_row)

    head_num = _numeric_fraction(header_row)
    body_num = sum(_numeric_fraction(r) for r in body_rows) / len(body_rows)
    # A leading empty cell ("",gene1,gene2) is the pandas index-name convention: header for sure.
    if header_row and header_row[0].strip() in ("", '""'):
        return True
    return head_num + 0.3 < body_num or (head_num < 0.2 and body_num > 0.5)


def _detect_index_column(header_row: list[str], body_rows: list[list[str]], has_header: bool) -> int | None:
    """The first column is an index when it is non-numeric and unique while the rest are numeric."""
    rows = body_rows if has_header else [header_row, *body_rows]
    rows = [r for r in rows if len(r) > 1]
    if not rows:
        return None
    first = [r[0].strip() for r in rows]
    rest_numeric = sum(_is_numeric(c) for r in rows for c in r[1:])
    rest_total = sum(len(r) - 1 for r in rows)
    if rest_total == 0:
        return None
    if all(_is_numeric(c) for c in first):
        # All-numeric first column: an index only if the header names it empty.
        if has_header and header_row and header_row[0].strip() in ("", '""'):
            return 0
        return None
    if rest_numeric / rest_total > 0.8 and len(set(first)) == len(first):
        return 0
    if has_header and header_row and header_row[0].strip() in ("", '""'):
        return 0
    return None


_NUMERIC_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _is_numeric(cell: str) -> bool:
    cell = cell.strip().strip('"')
    if cell in ("", "NA", "NaN", "nan", "null", "None"):
        return False
    return bool(_NUMERIC_RE.match(cell))


# A byte-order mark is the file stating its own encoding, and it is checked before anything else
# infers one. UTF-32-LE has to be tried ahead of UTF-16-LE because its mark *starts with* the
# UTF-16-LE mark, so testing the short one first would decode a UTF-32 file as UTF-16 rubbish.
_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)
_MIN_PRINTABLE_FRACTION = 0.85


def _decode_text(head: bytes) -> str | None:
    """Return decoded text, or None when the bytes are binary.

    The order matters and used to be wrong in three separate ways.

    A declared encoding comes first. A UTF-16 file -- Excel's "Unicode Text", and a fair number of
    Windows instrument exports -- is half NUL bytes, so the bare NUL search that used to run first
    dismissed an ordinary tab-delimited table as undecodable binary. The NUL rule is still the right
    backstop for everything that does *not* announce itself; it just cannot outrank the announcement.

    Then the decode, and only then the judgement of what came out. Judging printability by counting
    bytes under 128 meant judging UTF-8 by how much of it was ASCII: one umlaut and one em-dash per
    row is a quarter of the bytes, and an 800 KB CSV was called binary for containing German. C1
    control characters are what actually separate a decoded blob from decoded text.

    ``utf-8-sig`` is what strips a BOM, and plain ``utf-8`` was tried before it -- successfully,
    since U+FEFF is a perfectly legal character -- so the mark survived into the first column name.
    """
    for bom, encoding in _BOMS:
        if head.startswith(bom):
            text = _decode_sample(head, encoding)
            return text if text is not None and _looks_like_text(text) else None

    if b"\x00" in head:
        return None
    for encoding in ("utf-8", "latin-1"):
        text = _decode_sample(head, encoding)
        if text is not None and _looks_like_text(text):
            return text
    return None


def _decode_sample(data: bytes, encoding: str) -> str | None:
    """Decode a byte *sample*, tolerating a character the sample boundary cut in half.

    ``bytes.decode`` has no way to express "there may be more coming", so a fixed-size read that
    landed mid-sequence raised and fell through to ``latin-1``, turning every non-ASCII character in
    the sample into mojibake. An incremental decoder told the input is not final holds the partial
    tail back instead of raising.
    """
    try:
        return codecs.getincrementaldecoder(encoding)().decode(data, False)
    except UnicodeDecodeError:
        return None


def _looks_like_text(text: str) -> bool:
    """Whether decoded characters read as text rather than as a blob that happened to decode.

    ``latin-1`` maps every possible byte, so it never raises and cannot itself tell the two apart.
    What separates them is that arbitrary bytes land on C0/C1 control characters roughly a third of
    the time, and those are not printable.
    """
    sample = text[:8192]
    if not sample:
        return False
    printable = sum(1 for ch in sample if ch.isprintable() or ch in "\t\n\r")
    return printable / len(sample) >= _MIN_PRINTABLE_FRACTION


def _hexdump(chunk: bytes) -> str:
    return " ".join(f"{b:02x}" for b in chunk)


# A magic number has to be long enough that its appearance is not a coincidence. Three printable
# bytes ahead of binary noise is already common by chance in a numeric array; four is not, and every
# real signature this module knows of -- ARROW1, PAR1, MATLAB, SQLite format 3, \x93NUMPY -- clears
# it. Below the floor the answer is "there is nothing to search for", which is more useful to the
# caller than a token invented from two bytes.
_MIN_MAGIC_CHARS = 4


def _printable_magic(head: bytes) -> str:
    """The leading run of printable ASCII, when it is long enough to be a signature.

    Written for the branch that gives up: an unrecognised binary reports its first bytes as hex,
    and hex is the one encoding a search engine cannot match. ``41 52 52 4f 57 31`` spells
    ``ARROW1``, which resolves to the Arrow IPC / Feather-v2 spec in a single query. Recovering that
    string is what turns "unknown" from a verdict into a question the agent can go and answer.

    Only the *leading* run counts, and only from offset 0. A printable token found somewhere in the
    middle of a binary blob is far more likely to be a column name or an embedded path than a magic
    number, and sending the agent to search for one of those wastes a turn on noise.
    """
    run = bytearray()
    for byte in head[:32]:
        if 32 <= byte < 127:
            run.append(byte)
        else:
            break
    token = run.decode("ascii").strip()
    return token if len(token) >= _MIN_MAGIC_CHARS else ""


def _classify_stereoseq_gef(v: FormatVerdict, f: Any, root_keys: list[str], gef_root: str) -> None:
    """Name a Stereo-seq GEF and say what to do with it, which is not to read it here.

    `reader` stays None on purpose: `convert_stereoseq` refuses a `.gef` and asks for the text GEM,
    so a reader string would be a suggestion the rest of the system rejects. The search terms name
    the tool that produces the file it does want.
    """
    v.format = "stereoseq_gef"
    v.confidence = 0.9
    v.reader = None
    v.details["gef_level"] = "bin" if gef_root == "geneExp" else "cell"
    v.evidence.append(
        f"HDF5 root group '/{gef_root}' -- a Stereo-seq GEF holding "
        f"{'binned' if gef_root == 'geneExp' else 'cell-segmented'} counts (root keys {root_keys})"
    )
    try:
        bins = sorted(int(k[3:]) for k in f[gef_root].keys() if k.startswith("bin") and k[3:].isdigit())
    except (AttributeError, KeyError, TypeError, ValueError):
        bins = []
    if bins:
        # Which bin sizes the chip was written at decides what analysis is even possible on it.
        v.details["bin_sizes"] = bins
        v.evidence.append(f"bin groups present: {[f'bin{b}' for b in bins]}")
    v.warnings.append(
        "GEF is the binary Stereo-seq container. `convert_stereoseq` does not read it and asks for "
        "the text `.gem` instead; SAW writes both, and `geftools gef2gem` produces one from this file."
    )
    v.details["search_terms"] = [
        "Stereo-seq GEF file format specification geneExp cellBin",
        "stereopy read_gef python how to read a GEF file",
        "geftools gef2gem convert Stereo-seq GEF to GEM",
    ]


def _format_search_terms(token: str, extensions: Sequence[str] = ()) -> list[str]:
    """Queries that identify an unrecognised format, best first.

    The agent has ``search_google`` / ``advanced_web_search_claude``; what it lacks when the probe
    gives up is the string to put in them. A magic token is quoted so the engine treats it as a
    literal -- unquoted, ``ARROW1`` is a common enough substring to bury the spec -- and paired with
    the words that pull up a format description rather than a project README.
    """
    terms: list[str] = []
    if token:
        terms.append(f'"{token}" file format magic number specification')
        terms.append(f'"{token}" file format how to read python')
    for ext in list(dict.fromkeys(e for e in extensions if e))[:3]:
        terms.append(f'"{ext}" file format spatial transcriptomics how to read')
    return terms


def _crosscheck_extension(v: FormatVerdict, p: Path) -> None:
    """Record the extension and warn -- never act -- when it disagrees with the content."""
    name = p.name.lower()
    suffix = p.suffix.lower()
    v.details["extension"] = suffix
    expected = {
        ".h5ad": {"h5ad", "hdf5_unknown", "hdf5_unreadable"},
        ".loom": {"loom", "hdf5_unknown", "hdf5_unreadable"},
        ".h5": {"h5ad", "loom", "10x_h5", "hdf5_unknown", "hdf5_unreadable"},
        ".hdf5": {"h5ad", "loom", "10x_h5", "hdf5_unknown", "hdf5_unreadable"},
        ".csv": {"delimited_text", "plain_text"},
        ".tsv": {"delimited_text", "plain_text"},
        ".txt": {"delimited_text", "plain_text", "matrix_market", "fasta", "json"},
        ".mtx": {"matrix_market"},
        ".parquet": {"parquet"},
        ".json": {"json"},
        ".zarr": {"zarr"},
        ".npy": {"numpy_npy"},
        ".rds": {"r_object"},
        ".gz": None,  # the wrapper says nothing about the member
    }.get(suffix)

    if name.endswith((".csv.gz", ".tsv.gz", ".txt.gz")):
        expected = {"delimited_text", "plain_text"}
    if expected is None or v.format in {"unknown", "empty_file", "unreadable"}:
        return
    if v.format not in expected:
        v.warnings.append(
            f"extension {suffix!r} disagrees with the content: bytes say {v.format!r}. "
            "The content was used; the filename was ignored."
        )
        v.evidence.append(f"extension {suffix!r} would have selected the wrong reader")


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------


def _probe_directory(p: Path) -> FormatVerdict:
    v = FormatVerdict(path=str(p), kind="directory", container="directory")
    try:
        entries = sorted(x for x in p.iterdir() if not x.name.startswith("._"))
    except OSError as exc:
        v.format = "unreadable"
        v.evidence.append(f"directory could not be listed: {exc}")
        return v

    names = {x.name for x in entries}
    # Lower-cased name -> the spelling actually on disk. Matching markers case-insensitively is
    # deliberate, but a name the verdict *reports* -- or reopens to read a header -- has to be the
    # real one: `p / "matrix.mtx"` does not find `Matrix.mtx` on a case-sensitive filesystem, and a
    # caller handed a filename that is not there gets FileNotFoundError for a file it was just
    # told exists. Built from the sorted entries so a case-only collision resolves the same way twice.
    by_lower = {x.name.lower(): x.name for x in entries}
    subdirs = {x.name.lower() for x in entries if x.is_dir()}
    v.details["n_entries"] = len(entries)
    v.details["entries"] = sorted(names)[:40]

    if not entries:
        v.format = "empty_directory"
        v.evidence.append("directory is empty")
        return v

    matched = _run_layout_detectors(p, by_lower, subdirs, declined=v)

    if not matched:
        _describe_unknown_directory(v, p, entries, by_lower, subdirs)
        return v

    winner, alternatives = _rank_layouts(matched)
    v.format = winner.format
    v.confidence = winner.confidence
    v.reader = winner.reader
    v.evidence.extend(winner.evidence)
    v.warnings.extend(winner.warnings)
    v.details.update(winner.details)
    if alternatives:
        _record_alternative_layouts(v, alternatives)
    return v


def _layout_detectors() -> tuple[Callable[[FormatVerdict, Path, dict[str, str], set[str]], bool], ...]:
    """The detectors, in priority order. A function rather than a module constant because they are
    defined below this point in the file and a tuple up here would be built before they exist."""
    return (
        _detect_zarr,
        _detect_visium_hd,
        _detect_visium,
        _detect_mtx_triplet,
        _detect_xenium,
        _detect_merfish,
        _detect_cosmx,
        _detect_slideseq,
    )


def _run_layout_detectors(
    p: Path, by_lower: dict[str, str], subdirs: set[str], declined: FormatVerdict | None = None
) -> list[FormatVerdict]:
    """Every layout that fits this directory, in detector order.

    Every detector is asked, not just the ones before the first hit. Returning on the first match
    meant a second layout was never looked for, so nothing in the verdict could record that one
    existed -- and the first detector in the list is not reliably the right answer. See
    ``_rank_layouts``.

    ``declined`` collects what the detectors that said no still observed: "10x counts present but
    there is no spatial/" is the reason the caller is not getting Visium. Left ``None`` when
    matching a *child* directory, where those asides describe something the caller did not ask about.
    """
    matched: list[FormatVerdict] = []
    for detector in _layout_detectors():
        candidate = FormatVerdict(path=str(p), kind="directory", container="directory")
        if detector(candidate, p, by_lower, subdirs):
            matched.append(candidate)
        elif declined is not None:
            declined.evidence.extend(candidate.evidence)
            declined.warnings.extend(candidate.warnings)
    return matched


# A layout that says how the bytes are packed, not which instrument produced them. It is a real
# match and worth reporting, but it knows nothing about where the cells are, so when a platform
# layout fits the same directory the platform is the more complete answer.
_CONTAINER_LAYOUTS = frozenset({"10x_mtx"})


def _rank_layouts(matched: list[FormatVerdict]) -> tuple[FormatVerdict, list[FormatVerdict]]:
    """Pick between layouts that all fit, and hand back the ones that lost.

    Two rules, in order. A platform layout beats a container layout, because reading a MERSCOPE
    bundle as a bare MTX triplet drops the coordinates without erroring. Then the higher confidence
    wins -- each detector already grades itself by how many of its markers it saw, so a CosMx export
    matching three markers outranks ``cell_metadata.csv`` matching one, and no separate heuristic is
    needed. Ties fall back to the detector order, which ``max`` preserves by returning the first
    maximal element.
    """
    if len(matched) == 1:
        return matched[0], []
    platforms = [m for m in matched if m.format not in _CONTAINER_LAYOUTS]
    winner = max(platforms or matched, key=lambda m: m.confidence)
    return winner, [m for m in matched if m is not winner]


def _record_alternative_layouts(v: FormatVerdict, alternatives: list[FormatVerdict]) -> None:
    """Say that the directory fits more than one layout. That fact is the actionable part.

    Choosing a winner in silence just swaps one confident wrong answer for another; the caller
    needs to know this was a choice so it can check before reading rather than after.
    """
    v.details["also_matched"] = [
        {"format": o.format, "confidence": round(float(o.confidence), 3), "evidence": o.evidence[:1]}
        for o in alternatives
    ]
    v.warnings.append(
        f"This directory also matches {', '.join(o.format for o in alternatives)}. It is being read "
        f"as {v.format}; see details['also_matched'] for what else fits and why."
    )
    demoted = [o.format for o in alternatives if o.format in _CONTAINER_LAYOUTS]
    if demoted:
        v.warnings.append(
            f"{', '.join(demoted)} fits too, but it is a container layout with no coordinates in it. "
            f"Reading this directory that way would silently turn a spatial dataset into plain "
            f"expression -- the coordinates {v.format} carries would simply not be loaded."
        )


def _detect_zarr(v: FormatVerdict, p: Path, by_lower: dict[str, str], subdirs: set[str]) -> bool:
    marker = None
    if ".zgroup" in by_lower or ".zarray" in by_lower:
        marker = ".zgroup" if ".zgroup" in by_lower else ".zarray"
        v.details["zarr_format"] = 2
    elif "zarr.json" in by_lower:
        marker = "zarr.json"
        v.details["zarr_format"] = 3
    if marker is None:
        return False
    v.format = "zarr"
    v.confidence = 0.9
    v.reader = "zarr.open(path)  # SpatialData: spatialdata.read_zarr(path)"
    v.evidence.append(f"Zarr store marker {marker!r} at the directory root")

    groups = sorted(_SPATIALDATA_GROUPS & subdirs)
    declared = _zarr_declares_spatialdata(p, by_lower)
    if groups:
        v.details["spatialdata_groups"] = groups
        v.evidence.append(f"SpatialData group names present: {groups}")
    if declared:
        # The document that was actually read, not the one a v2 store would have had: a v3 store
        # has no `.zattrs`, and naming it anyway sends anyone checking the claim to a missing path.
        v.evidence.append(f"{declared!r} at the store root carries 'spatialdata_attrs'")
    if groups or declared:
        v.reader = "spatialdata.read_zarr(path)"
    return True


# The element groups a SpatialData store puts at its root. `tables` is plural since spatialdata
# 0.1, which renamed it when several tables became possible; `table` stays for stores written
# before that. `labels` is deliberately absent: OME-NGFF image pyramids put segmentation masks
# under that name too, so on its own it identifies nothing -- a store that really is SpatialData
# and has only labels in it says so in `.zattrs`, which is checked separately.
_SPATIALDATA_GROUPS = frozenset({"table", "tables", "images", "shapes", "points"})


def _json_declares_spatialdata(blob: bytes) -> bool:
    """True when this root-metadata document declares a SpatialData store.

    Zarr v2 keeps attributes at the top level of `.zattrs`; v3 nests them under `attributes` inside
    `zarr.json`. Both spellings are read here so the disk and the zipped path cannot answer the same
    question differently -- they did, and a v3 store recognised as SpatialData inside a `.zip` was
    handed a bare `zarr.open` once unpacked. Never raises: unparseable metadata is one fewer clue.
    """
    try:
        attrs = json.loads(blob.decode("utf-8", errors="replace"))
    except ValueError:
        return False
    if not isinstance(attrs, dict):
        return False
    nested = attrs.get("attributes")
    return "spatialdata_attrs" in attrs or (isinstance(nested, dict) and "spatialdata_attrs" in nested)


def _zarr_declares_spatialdata(p: Path, by_lower: dict[str, str]) -> str | None:
    """The root document that says this store is SpatialData, or None if neither does.

    Directory names are a guess; this is the format answering the question in writing. Reads both
    root documents, because which one exists is the store's zarr version rather than anything about
    its content, and returns the one that answered so the evidence can name a file that was really
    opened. Never raises -- unreadable metadata is one fewer clue, not a failed probe.
    """
    for basename in (".zattrs", "zarr.json"):
        name = by_lower.get(basename)
        if not name:
            continue
        try:
            # Bounded at the read, not after it: root metadata is a few hundred bytes in every store
            # that has ever existed, and slicing a `read_text()` would have pulled the whole file in
            # first. A v3 root also holds the group's `attributes`, which stays well inside this.
            with (p / name).open("rb") as fh:
                blob = fh.read(_HEAD_BYTES)
        except OSError:
            continue
        if _json_declares_spatialdata(blob):
            return name
    return None


# Space Ranger 3 writes a Visium HD run as one complete Space Ranger output per bin size, under
# `binned_outputs/square_<NNN>um/`. The bins are recognised perfectly on their own; the run root is
# what a user actually points at, and it is two levels up from any marker -- one further than the
# one-level child scan reaches.
_HD_BIN_RE = re.compile(r"^square_(\d+)um$")
_HD_DEFAULT_BIN_UM = 8


def _detect_visium_hd(v: FormatVerdict, p: Path, by_lower: dict[str, str], subdirs: set[str]) -> bool:
    """Recognise a Visium HD run from its bin directories, wherever the caller entered the tree."""
    for holder in ([p / by_lower["binned_outputs"]] if "binned_outputs" in subdirs else []) + [p]:
        bins = _visium_hd_bins(holder)
        if bins:
            break
    else:  # pragma: no cover - the loop body always runs at least once
        return False
    if not bins:
        return False

    v.format = "visium_hd"
    v.confidence = 0.95
    v.details["bin_sizes_um"] = sorted(bins)
    v.details["bins"] = [bins[um].name for um in sorted(bins)]
    # 8um is the bin the field analyses -- roughly one cell -- so it is the suggested one when it is
    # there. Otherwise the largest present, because it is the one that loads; every bin is listed
    # above, and picking 2um for someone by default would suggest the heaviest file in the run.
    chosen = _HD_DEFAULT_BIN_UM if _HD_DEFAULT_BIN_UM in bins else max(bins)
    target = bins[chosen]
    rel = target.relative_to(p).as_posix()
    # A bin's positions are a parquet, which scanpy.read_visium does not read (u15-validation-6).
    v.reader = (
        f"sc.read_10x_h5(path / '{rel}/filtered_feature_bc_matrix.h5') + "
        f"pd.read_parquet(path / '{rel}/spatial/tissue_positions.parquet')  # the {chosen}um bin"
    )
    v.evidence.append(
        f"Visium HD run: bin directories {[bins[um].name for um in sorted(bins)]} "
        f"(Space Ranger writes one full output per bin size)"
    )
    if len(bins) > 1:
        v.warnings.append(
            f"This run holds {len(bins)} bin sizes ({', '.join(f'{um}um' for um in sorted(bins))}) and they "
            f"are separate datasets, not parts of one. The reader above opens the {chosen}um bin; "
            "analysing more than one means reading each in turn."
        )
    return True


def _visium_hd_bins(holder: Path) -> dict[int, Path]:
    """Bin size in microns -> the bin directory, for bins that hold a Space Ranger output.

    The `spatial/` check is what stops an empty directory that merely *looks* named like a bin from
    promoting the whole run to HD and sending the reader somewhere with nothing in it.
    """
    try:
        children = sorted(holder.iterdir())
    except OSError:
        return {}
    found: dict[int, Path] = {}
    for child in children:
        match = _HD_BIN_RE.match(child.name.lower())
        if match and (child / "spatial").is_dir():
            found[int(match.group(1))] = child
    return found


def _visium_counts_name(by_lower: dict[str, str]) -> str | None:
    """The counts matrix's real name: the Space Ranger name, or a sample-prefixed spelling of it.

    10x distributes public runs as ``<sample>_filtered_feature_bc_matrix.h5`` beside
    ``<sample>_spatial.tar.gz``; extracted side by side that is a Visium directory, and an exact-name
    match called it unknown_directory with no hint (u15-validation-13). Exact names win.
    """
    for name in _VISIUM_COUNTS:
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    for name in _VISIUM_COUNTS:
        for low in sorted(by_lower):
            if _is_marker(low, name.lower()):
                return by_lower[low]
    return None


def _detect_visium(v: FormatVerdict, p: Path, by_lower: dict[str, str], subdirs: set[str]) -> bool:
    counts_name = _visium_counts_name(by_lower)
    has_counts = counts_name is not None
    if not has_counts:
        return False
    if "spatial" not in subdirs:
        # Counts without coordinates is not Visium for any purpose downstream cares about; saying
        # so sends the agent to a converter that then cannot find the positions file.
        v.evidence.append("10x counts present but there is no spatial/ subdirectory -- not a Space Ranger output")
        return False

    spatial = p / "spatial"
    try:
        spatial_names = {x.name.lower() for x in spatial.iterdir()}
    except OSError:
        spatial_names = set()

    v.format = "visium_spaceranger"
    prefixed = counts_name.lower() not in {n.lower() for n in _VISIUM_COUNTS}
    v.reader = (
        f"scanpy.read_visium(path, count_file={counts_name!r})"
        if prefixed
        else "scanpy.read_visium(path)  # or squidpy.read.visium(path)"
    )
    v.details["has_tissue_positions"] = bool(spatial_names & {n.lower() for n in _VISIUM_POSITIONS})
    # Space Ranger 3 and every Visium HD bin write the positions ONLY as a parquet, and
    # scanpy.read_visium reads the two CSV names and raises on anything else -- so naming it here, at
    # 0.95, handed the agent a reader that fails on a verdict scored decisive (u15-validation-6).
    parquet_only = "tissue_positions.parquet" in spatial_names and not (
        spatial_names & {"tissue_positions.csv", "tissue_positions_list.csv"}
    )
    v.details["positions_parquet_only"] = parquet_only
    if parquet_only:
        v.reader = (
            f"sc.read_10x_h5(path / {counts_name!r}) + "
            "pd.read_parquet(path / 'spatial/tissue_positions.parquet')  # scanpy.read_visium cannot read the parquet"
        )
        v.warnings.append(
            "positions are only in spatial/tissue_positions.parquet (Space Ranger 3), which scanpy.read_visium "
            "does not read: load the counts h5 and attach the parquet's pxl_col_in_fullres/pxl_row_in_fullres"
        )
    v.details["has_scalefactors"] = "scalefactors_json.json" in spatial_names
    # 0.95 was returned whatever `spatial/` turned out to hold, so the same verdict scored a
    # decisive layout on one line and warned that coordinates cannot be attached on the next. Above
    # 0.9 is documented on `FormatVerdict` to mean a signature or a decisive group layout was
    # matched, and a Space Ranger directory missing either of the two files `scanpy.read_visium`
    # opens is not one: the layout is Visium-shaped, but the reader named below cannot run on it.
    # Only the number moves -- the format and the reader stay, and 0.7 keeps `is_known` True, so
    # nothing that routes on this verdict reroutes.
    v.confidence = (
        0.95 if (v.details["has_tissue_positions"] and v.details["has_scalefactors"] and not parquet_only) else 0.7
    )
    v.details["images"] = sorted(n for n in spatial_names if n.endswith(_IMAGE_SUFFIXES))
    v.details["counts_file"] = counts_name
    v.evidence.append("10x counts matrix beside a spatial/ directory (Space Ranger output layout)")
    v.evidence.append(
        f"spatial/ holds tissue positions: {v.details['has_tissue_positions']}, "
        f"scalefactors: {v.details['has_scalefactors']}, images: {v.details['images'] or 'none'}"
    )
    if not v.details["has_tissue_positions"]:
        v.warnings.append("spatial/ has no tissue_positions file -- coordinates cannot be attached")
    if not v.details["has_scalefactors"]:
        v.warnings.append("spatial/ has no scalefactors_json.json -- image alignment will fail")
    return True


def _detect_mtx_triplet(v: FormatVerdict, p: Path, by_lower: dict[str, str], subdirs: set[str]) -> bool:
    # Two spellings of each name, and they are not interchangeable. The canonical key decides
    # whether the member is gzipped -- `.GZ` has to count -- while every name that leaves this
    # function, or is joined onto `p` to read a header, must be the one on disk.
    mtx_key = next((n for n in _MTX_NAMES if n in by_lower), None)
    barcodes_key = next((n for n in _BARCODE_NAMES if n in by_lower), None)
    features_key = next((n for n in _FEATURE_NAMES if n in by_lower), None)
    if not (mtx_key and barcodes_key and features_key):
        return False
    mtx, barcodes, features = by_lower[mtx_key], by_lower[barcodes_key], by_lower[features_key]
    gzipped = mtx_key.endswith(".gz")
    v.format = "10x_mtx"
    v.reader = "scanpy.read_10x_mtx(path)"
    v.details["gzipped"] = gzipped
    v.details["triplet"] = {"matrix": mtx, "barcodes": barcodes, "features": features}
    v.evidence.append(f"Matrix Market triplet present: {mtx}, {barcodes}, {features}")
    banner = _read_mtx_header(p / mtx, gzipped=gzipped)
    if banner:
        v.details["shape"] = banner
        v.details["axis_order"] = "genes_x_cells (10x writes features x barcodes)"
        v.evidence.append(f"Matrix Market header declares {banner[0]} rows x {banner[1]} columns")
        v.confidence = 0.95
    else:
        # The banner is the only content evidence a triplet offers, and it was already being read
        # and then discarded: three empty files with the right names -- what an interrupted export
        # leaves behind -- came back at 0.95 with `scanpy.read_10x_mtx(path)` and no warning at all.
        # Above 0.9 is documented to mean a signature was matched. This one was looked for and was
        # not there, so say that and score it as the name match it is.
        v.confidence = 0.7
        v.details["shape"] = None
        v.warnings.append(
            f"{mtx} carries no Matrix Market banner in its first 50 lines -- the names match a 10x "
            "triplet but the matrix is empty, truncated or some other format, and read_10x_mtx "
            "will fail on it"
        )
        v.evidence.append(f"no Matrix Market header found in {mtx}; verdict rests on the file names alone")
    return True


#: The longest line a Matrix-Market header sniff reads. A banner, a comment or the size line is
#: far shorter; anything longer is not a header this probe can use.
_MTX_HEADER_LINE_CHARS = 4096


def _read_mtx_header(path: Path, gzipped: bool) -> list[int] | None:
    try:
        opener = gzip.open if gzipped else open
        with opener(str(path), "rt", errors="replace") as fh:
            for _ in range(50):
                # Bounded per line, like every other sniff here: an unbounded readline() on a
                # newline-free file read all of it into the serving process -- a 300 KB gzip cost
                # ~600 MB of portal RSS (hunt 2026-09-30, uL3-resources-3).
                line = fh.readline(_MTX_HEADER_LINE_CHARS)
                if not line:
                    return None
                if len(line) >= _MTX_HEADER_LINE_CHARS and not line.endswith("\n"):
                    return None
                line = line.strip()
                if not line or line.startswith("%"):
                    continue
                parts = line.split()
                if len(parts) in (2, 3) and all(x.isdigit() for x in parts):
                    return [int(parts[0]), int(parts[1])]
                return None
    except OSError:
        return None
    return None


def _detect_xenium(v: FormatVerdict, p: Path, by_lower: dict[str, str], subdirs: set[str]) -> bool:
    hits = sorted(by_lower[n] for n in by_lower.keys() & set(_XENIUM_MARKERS))
    if not hits:
        return False
    v.format = "xenium_output"
    v.confidence = 0.9 if "experiment.xenium" in by_lower else 0.75
    v.reader = "spatialdata_io.xenium(path)  # or manual: read cells.parquet + cell_feature_matrix.h5"
    v.details["markers"] = hits
    v.details["images"] = sorted(name for key, name in by_lower.items() if key.endswith((".ome.tif", ".ome.tiff")))
    v.evidence.append(f"Xenium output markers present: {hits}")
    return True


def _detect_merfish(v: FormatVerdict, p: Path, by_lower: dict[str, str], subdirs: set[str]) -> bool:
    by_lower = _with_table_aliases(by_lower)
    hits = sorted(by_lower[n] for n in by_lower.keys() & set(_MERFISH_MARKERS))
    if not hits:
        return False
    has_expr = "cell_by_gene.csv" in by_lower
    has_meta = "cell_metadata.csv" in by_lower

    v.format = "merfish_vizgen"
    v.confidence = 0.85 if len(hits) > 1 else 0.7
    v.details["markers"] = hits
    v.details["has_expression"] = has_expr
    v.details["has_coordinates"] = has_meta
    v.evidence.append(f"Vizgen MERFISH file names present: {hits}")

    # `detected_transcripts.csv` on its own is enough to match, and a MERSCOPE run handed over at
    # transcript granularity is a normal thing to receive. The reader named `cell_by_gene.csv`
    # either way, so the verdict said `has_expression: False` on one line and told the caller to
    # open that file on the next. Named through `by_lower` so the spelling is the one on disk.
    if has_expr and has_meta:
        v.reader = (
            f"pandas.read_csv(path / '{by_lower['cell_by_gene.csv']}', index_col=0) + "
            f"pandas.read_csv(path / '{by_lower['cell_metadata.csv']}')  # center_x/center_y"
        )
    elif has_expr:
        v.reader = f"pandas.read_csv(path / '{by_lower['cell_by_gene.csv']}', index_col=0)  # no coordinates here"
    elif "detected_transcripts.csv" in by_lower:
        # Not nothing: the transcript table is readable and carries its own global coordinates. It
        # is a different granularity, and that is the fact worth handing back rather than a reader
        # for a file that is not here.
        v.reader = (
            f"pandas.read_csv(path / '{by_lower['detected_transcripts.csv']}')"
            "  # transcript-level (global_x/global_y), not cells"
        )
    else:
        v.reader = None

    if not has_meta:
        v.warnings.append("cell_metadata.csv is missing -- there are no cell coordinates in this bundle")
    if not has_expr:
        v.warnings.append(
            "cell_by_gene.csv is missing -- there is no cell-by-gene matrix here, so convert_merfish "
            "cannot run. Segmenting cells over the transcript table is what produces one (Vizgen's vpt)."
        )
    return True


def _detect_cosmx(v: FormatVerdict, p: Path, by_lower: dict[str, str], subdirs: set[str]) -> bool:
    by_lower = _with_table_aliases(by_lower)
    hits = _names_matching_marker(by_lower, _COSMX_MARKERS)
    if not hits:
        return False
    # Which of the two files `convert_cosmx` needs are actually here. Asserting both and naming
    # them in the reader regardless is how an export missing its coordinates was described as
    # complete -- the MERFISH detector below answers the same question by looking.
    expr = _names_matching_marker(by_lower, ("exprmat_file.csv",))
    meta = _names_matching_marker(by_lower, ("metadata_file.csv",))
    # Pair the two halves by slide, not by position. An AtoMx hand-over of several slides lands in
    # one directory often enough that `convert_cosmx` has a guard for what happens next: `fov` and
    # `cell_ID` restart per slide, so pooling them makes `fov_cell_ID` non-unique and the join goes
    # cartesian. Naming one slide's two files without saying the others are here is what sent the
    # caller into that.
    one_expr, one_meta, slides, paired = _pair_by_bundle(expr, ("exprmat_file.csv",), meta, ("metadata_file.csv",))

    v.format = "cosmx_output"
    v.confidence = 0.85 if len(hits) > 1 else 0.65
    v.details["markers"] = hits
    v.details["has_expression"] = bool(expr)
    v.details["has_coordinates"] = bool(meta)
    v.details["slides"] = slides
    v.evidence.append(f"Nanostring CosMx file names present: {hits}")

    # Named in full, because the slide prefix differs per export: a caller told only "read the
    # expression matrix and the metadata" has to go and look up what they are called here.
    if expr and meta and paired:
        v.reader = f"pandas.read_csv(path / '{one_expr}') + pandas.read_csv(path / '{one_meta}')  # CenterX/Y_global_px"
    elif expr and meta:
        # Both halves are present but not for the same slide. Handing back a reader that names one
        # of each would be a line the caller can run and get a wrong answer from, so there is none.
        v.reader = None
        v.warnings.append(
            f"'{one_expr}' and '{one_meta}' are from different slides -- no slide in this directory "
            "has both halves, so there is no pair of files convert_cosmx can be given as they stand"
        )
    elif expr:
        v.reader = f"pandas.read_csv(path / '{one_expr}')  # expression only; no coordinates in this bundle"
        v.warnings.append(
            "metadata_file.csv is missing -- there are no cell coordinates in this bundle, and "
            "convert_cosmx needs it as well as the expression matrix"
        )
    else:
        v.reader = None
        v.warnings.append(
            "exprMat_file.csv is missing -- this bundle carries CosMx side files but no expression matrix"
        )
    if len(slides) > 1:
        v.warnings.append(
            f"this directory holds {len(slides)} CosMx slides ({slides}); the reader above names one of them. "
            "convert_cosmx takes a single slide -- pooling them makes fov/cell_ID collide across slides"
        )
    return True


def _detect_slideseq(v: FormatVerdict, p: Path, by_lower: dict[str, str], subdirs: set[str]) -> bool:
    locations = _names_ending_with(by_lower, _SLIDESEQ_LOCATION_SUFFIXES)
    if not locations:
        return False
    counts = _names_ending_with(by_lower, _SLIDESEQ_COUNT_SUFFIXES)
    # By puck, not by position. The bead file and the DGE are deposited separately and are not
    # always deposited together, so sorting the two lists apart and taking [0] of each paired one
    # puck's beads with another's counts. Those two files share no barcode at all, and the
    # intersection that follows would just quietly produce an almost-empty object.
    one_counts, one_locations, pucks, paired = _pair_by_bundle(
        counts, _SLIDESEQ_COUNT_SUFFIXES, locations, _SLIDESEQ_LOCATION_SUFFIXES
    )

    v.format = "slideseq"
    v.confidence = 0.8
    v.details["markers"] = locations + counts
    v.details["bead_locations"] = one_locations
    v.details["pucks"] = pucks
    v.evidence.append(f"Slide-seq bead-location table present: {locations}")
    if counts and paired:
        v.details["expression"] = one_counts
        v.evidence.append(f"expression matrix alongside it: {counts}")
        # Naming the two files is the whole reader: their names carry a puck id that differs per
        # dataset, so a caller told only "read the DGE and the locations" has to go and look.
        v.reader = f"pandas.read_csv(path / '{one_counts}', sep=None) + pandas.read_csv(path / '{one_locations}')"
    elif counts:
        v.details["expression"] = one_counts
        v.evidence.append(f"expression matrix alongside it: {counts}")
        v.reader = None
        v.warnings.append(
            f"'{one_counts}' and '{one_locations}' are from different pucks -- their barcodes will not "
            "intersect. No puck in this directory has both its DGE and its bead locations."
        )
    else:
        v.reader = f"pandas.read_csv(path / '{one_locations}')  # bead locations; no expression matrix found here"
        v.warnings.append(
            f"Bead locations are here ({one_locations}) but no expression matrix is. Slide-seq ships the "
            "two separately and the counts are usually a '.digital_expression.txt.gz' beside them."
        )
    if len(pucks) > 1:
        v.warnings.append(
            f"this directory holds {len(pucks)} Slide-seq pucks ({pucks}); the reader above names one of them"
        )
    return True


def _names_ending_with(by_lower: dict[str, str], suffixes: tuple[str, ...]) -> list[str]:
    """The on-disk spellings of members whose lower-cased name ends with one of these."""
    return sorted(real for lower, real in by_lower.items() if lower.endswith(suffixes))


def _is_marker(lower_name: str, marker: str) -> bool:
    """True when a member is `marker`, or a vendor-prefixed spelling of it.

    Nanostring puts the slide name in front of every file in an export, so
    `Lung5_Rep1_exprMat_file.csv` is what ships and the bare name is what (almost) never does.
    Requiring a separator immediately before the marker is what keeps this a prefix rule rather
    than a substring search: `myexprMat_file.csv` is somebody else's file.

    `spatial_pipeline._matches_marker` is the same rule. It is restated here rather than imported
    because `utils` must not depend on `tool`, and that module already imports this one.
    """
    return lower_name == marker or (
        lower_name.endswith(marker) and len(lower_name) > len(marker) and lower_name[-len(marker) - 1] in "_-."
    )


def _names_matching_marker(by_lower: dict[str, str], markers: tuple[str, ...]) -> list[str]:
    """The on-disk spellings of members that are one of these markers, vendor prefix and all."""
    return sorted(real for lower, real in by_lower.items() if any(_is_marker(lower, m) for m in markers))


#: A compression suffix over a delimited table is an encoding, not a format: pandas reads
#: ``cell_by_gene.csv.gz`` as it reads ``cell_by_gene.csv``. ``spatial_pipeline._uncompressed_name``
#: is the same rule, restated because ``utils`` must not import ``tool``.
_TEXT_COMPRESSIONS = (".gz", ".bz2", ".xz")
_COMPRESSIBLE_TABLES = (".csv", ".tsv", ".tab", ".txt")


def _uncompressed(lower_name: str) -> str:
    for compression in _TEXT_COMPRESSIONS:
        if lower_name.endswith(compression):
            stem = lower_name[: -len(compression)]
            return stem if stem.endswith(_COMPRESSIBLE_TABLES) else lower_name
    return lower_name


def _with_table_aliases(by_lower: dict[str, str]) -> dict[str, str]:
    """``by_lower`` plus each compressed table under its uncompressed name, mapped to the real file.

    For the vendor detectors whose markers are exact table names: a Vizgen bundle of
    ``cell_by_gene.csv.gz`` + ``cell_metadata.csv.gz``, or an AtoMx ``*_exprMat_file.csv.gz`` export,
    probed as ``unknown_directory`` while ``spatial_pipeline`` recognised it (hunt 2026-09-30,
    u15-validation-7). Not applied to the 10x detectors, which decide gzip from the name themselves.
    """
    out = dict(by_lower)
    for lower, real in by_lower.items():
        out.setdefault(_uncompressed(lower), real)
    return out


def _bundle_key(real_name: str, tails: tuple[str, ...]) -> str:
    """Which slide/puck a file belongs to: its name with the shared tail removed.

    `Lung5_Rep1_exprMat_file.csv` and `Lung5_Rep1_metadata_file.csv` are two halves of one bundle;
    `Lung5_Rep2_*` is a different one. Empty string for the unprefixed stock names, which is a key
    like any other -- a directory of bare `exprMat_file.csv`/`metadata_file.csv` is one bundle.
    """
    lower = real_name.lower()
    if not any(lower.endswith(t) for t in tails):
        lower = _uncompressed(lower)  # `Run1_S1_exprMat_file.csv.gz` is slide `run1_s1`
    # Longest first: `.digital_expression.txt.gz` must win over `.digital_expression.txt`.
    for tail in sorted(tails, key=len, reverse=True):
        if lower.endswith(tail):
            return lower[: len(lower) - len(tail)].rstrip("_-.")
    return ""


def _pair_by_bundle(
    left: list[str], left_tails: tuple[str, ...], right: list[str], right_tails: tuple[str, ...]
) -> tuple[str | None, str | None, list[str], bool]:
    """Pick one file from each side belonging to the *same* bundle; list every bundle seen.

    Returns ``(left_pick, right_pick, bundles, paired)``. ``paired`` is False when no single bundle
    has both halves -- the picks are then each side's first and go together only by accident.

    Both sides used to be sorted independently and indexed ``[0]``. That is only right when the two
    lists line up, and they do not have to: a puck whose bead file was deposited without its DGE
    shifts one list and not the other, so the reader paired puck 08's beads with puck 09's counts --
    two files with no barcode in common. Nothing downstream could tell, because the join that
    follows drops non-matching rows silently.
    """
    left_by_key = {_bundle_key(n, left_tails): n for n in reversed(left)}
    right_by_key = {_bundle_key(n, right_tails): n for n in reversed(right)}
    bundles = sorted(set(left_by_key) | set(right_by_key))

    shared = sorted(set(left_by_key) & set(right_by_key))
    if shared:
        return left_by_key[shared[0]], right_by_key[shared[0]], bundles, True
    return (left[0] if left else None), (right[0] if right else None), bundles, False


def _describe_unknown_directory(
    v: FormatVerdict, p: Path, entries: list[Path], by_lower: dict[str, str], subdirs: set[str]
) -> None:
    """No layout matched. Say exactly what is in there -- that is the actionable part."""
    files = [x for x in entries if x.is_file()]
    by_probe: dict[str, list[str]] = {}
    # Probe up to a handful of files so the caller learns what the contents actually are, not just
    # what they are called. Bounded: a directory can hold 100k FOV tiles.
    for f in files[:12]:
        try:
            sub = _probe_file(f)
        except Exception:  # a helper must never be the thing that throws
            continue
        by_probe.setdefault(sub.format, []).append(f.name)

    v.format = "unknown_directory" if len(files) or subdirs else "empty_directory"
    v.confidence = 0.1
    v.details["member_formats"] = {k: sorted(names)[:5] for k, names in by_probe.items()}
    v.details["subdirectories"] = sorted(subdirs)[:20]
    v.details["n_files"] = len(files)
    v.evidence.append(
        f"no known layout matched: {len(files)} files, {len(subdirs)} subdirectories "
        f"({sorted(subdirs)[:5] if subdirs else 'none'})"
    )
    if by_probe:
        v.evidence.append(f"content probe of the first {min(len(files), 12)} files found: {sorted(by_probe)}")
    _describe_nested_datasets(v, p, entries)
    if "spatial" not in subdirs and any(f.endswith(".h5") for f in by_lower):
        v.evidence.append("an .h5 counts file is present but there is no spatial/ directory")

    # A directory has no magic bytes of its own, so the searchable fact is what its members are
    # called. An unfamiliar extension is the strongest handle -- `.gef`, `.cbor`, `.mtxb` each name
    # exactly one format -- and the members' own magic tokens back it up when the extensions are
    # generic. See :func:`_format_search_terms`.
    extensions = [e for e in dict.fromkeys(f.suffix.lower() for f in files) if e and e not in _GENERIC_SUFFIXES]
    member_token = next((sub for sub in (_member_magic(f) for f in files[:12]) if sub), "")
    v.details["search_terms"] = _format_search_terms(member_token, extensions)
    v.reader = None


def _describe_nested_datasets(v: FormatVerdict, p: Path, entries: list[Path]) -> None:
    """Name the datasets sitting one directory down, when this directory is not one itself.

    Unzip a delivery and you get ``handover/xenium_run/``; hand over a study and you get
    ``study/sample_0/``, ``sample_1/``, ... Pointing the agent at the outer directory is the normal
    thing to do, and it used to get "no known layout matched: 0 files, 1 subdirectories" at 0.1
    confidence -- because member *files* were probed and member *directories* were listed by name
    only. The layout was one `listdir` away the whole time.

    One level, no recursion: the detectors are run directly against each child rather than
    re-entering ``probe_format``, so a deep tree of unremarkable folders costs one listing each and
    cannot fan out. Bounded for the same reason the file probe is -- a study can hold thousands of
    sample folders.

    The bound is reported rather than hidden. A 40-sample series announced as "24 of its
    subdirectories are" states the number the scan reached as the number that exist, and a study
    whose samples sit behind 30 log directories used to return in silence -- which is this
    function's own failure mode, "reading this one will find no data and report that none exists",
    reproduced one level out.
    """
    children = [x for x in entries if x.is_dir()]
    scanned = children[:_MAX_CHILDREN_SCANNED]
    # Whether the cap bit, which every sentence below depends on: a count of what was examined is
    # not a count of what is there, and finding nothing in a bounded scan is not "there is nothing".
    complete = len(scanned) == len(children)
    found: list[dict[str, str]] = []
    for child in scanned:
        try:
            kids = sorted(child.iterdir())
            matched = _run_layout_detectors(
                child, {x.name.lower(): x.name for x in kids}, {x.name.lower() for x in kids if x.is_dir()}
            )
        except OSError:  # unreadable child: one fewer clue, not a failed probe
            continue
        if matched:
            found.append({"subdirectory": child.name, "format": _rank_layouts(matched)[0].format})

    v.details["subdirectories_scanned"] = len(scanned)
    if not found:
        # Nothing found in a scan that stopped early is the one case where saying nothing is wrong:
        # the caller reads an absent warning as "checked, and there is no data down there".
        if not complete:
            v.warnings.append(
                f"Only the first {len(scanned)} of this directory's {len(children)} subdirectories "
                f"were examined and none of them holds a dataset; {len(children) - len(scanned)} "
                "were not looked at. If the data is in one of those, list this directory and point "
                "the reader at the sample itself."
            )
        return

    v.details["datasets_in_subdirectories"] = found
    v.details["datasets_in_subdirectories_complete"] = complete
    named = ", ".join(f"{f['subdirectory']}/ ({f['format']})" for f in found[:8])
    more = f" and {len(found) - 8} more" if len(found) > 8 else ""
    count = ("one" if len(found) == 1 else str(len(found))) if complete else f"at least {len(found)}"
    verb = "is" if len(found) == 1 else "are"
    tail = (
        ""
        if complete
        else f" Only the first {len(scanned)} of {len(children)} subdirectories were examined, so there may be more."
    )
    v.warnings.append(
        f"This directory is not itself a dataset, but {count} of its subdirectories {verb}: "
        f"{named}{more}. Point the reader at the subdirectory rather than at this path -- reading "
        f"this one will find no data and report that none exists.{tail}"
    )


# One listing per child is cheap only while the children are thin. A child holding 2000 FOV tiles
# costs ~16 ms to list and sort, so 256 of them is 4 s of probe -- which is why this stays a cap
# rather than becoming a bigger number. What the cap must not do is stay invisible: see
# `_describe_nested_datasets`, which reports how far it got whenever the answer is partial.
_MAX_CHILDREN_SCANNED = 24


# Extensions that identify a container rather than a format: searching for them returns the
# container's spec, which the probe already understands, and not the thing it failed to name.
_GENERIC_SUFFIXES = frozenset({".txt", ".csv", ".tsv", ".dat", ".bin", ".gz", ".zip", ".json", ".log", ".md", ".xml"})


def _member_magic(f: Path) -> str:
    """The printable magic of one directory member, or ``""``. Never raises -- an unreadable
    member is one fewer clue, not a failed probe."""
    try:
        with f.open("rb") as fh:
            return _printable_magic(fh.read(32))
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Orientation resolution
# ---------------------------------------------------------------------------

_ENSEMBL_RE = re.compile(r"^ENS[A-Z]{0,4}[GTP]\d{6,}(\.\d+)?$", re.IGNORECASE)
_BARCODE_RE = re.compile(r"^[ACGTN]{12,}(-\d+)?$", re.IGNORECASE)
_SPOTID_RE = re.compile(r"^(spot|cell|bead|bc|obs|pixel|px)[_\-.]?\d+$", re.IGNORECASE)
_COORDID_RE = re.compile(r"^\d+[x,_:\-]\d+$")
_GENE_SYMBOL_RE = re.compile(r"^(?:[A-Z][A-Z0-9]{1,9}(?:[.\-]\d+)?|[A-Z][a-z0-9]{1,9})$")
_MT_GENE_RE = re.compile(r"^(mt|MT|Mt)[-_.]", re.IGNORECASE)

# A signal must clear this share of an axis to count as identifying it.
_CLASS_THRESHOLD = 0.6
_VOCAB_HIT = 0.5
_VOCAB_MISS = 0.15


def resolve_orientation(
    row_labels: Sequence[Any] | None = None,
    col_labels: Sequence[Any] | None = None,
    *,
    matrix: Any | None = None,
    gene_vocabulary: Iterable[str] | None = None,
    expect: str = "cells_x_genes",
) -> OrientationVerdict:
    """Decide whether a matrix is cells-x-genes or genes-x-cells, or refuse.

    Independent signals are collected -- membership in a supplied gene vocabulary, the *class* of
    each axis's labels (Ensembl ID / barcode / gene symbol / positional id), the all-zero-vector
    asymmetry between the two axes, and the shape ratio -- and each votes or abstains. An answer is
    returned only when a strong signal identifies **both** axes differently and no other signal
    contradicts it. Otherwise the verdict is ``ambiguous``.

    Refusing is the designed behaviour, not a failure mode. Two orientation heuristics in this
    codebase always produced an answer and both silently corrupted real data: a correct 6142 x 7
    deconvolution matrix was transposed because ``spot_0`` reads as a cell-type name, and a
    genes-x-cells CSV was converted to h5ad and reported as ``status: ok``. Shape never decides
    here -- a 200-gene MERFISH panel over 3000 cells and a 3000-gene panel over 200 cells are the
    same shape -- and neither does the absence of a disqualifier on one axis.

    Args:
        row_labels: Row index labels, if any.
        col_labels: Column labels, if any.
        matrix: Optional 2-D array-like (numpy/scipy) for the numeric signals. Only aggregate
            statistics are computed; nothing is densified.
        gene_vocabulary: Optional iterable of known gene names -- e.g. a reference's ``var_names``.
            When available this is the strongest evidence there is.
        expect: The orientation the caller wants, used only to fill ``transpose_needed``.

    Returns:
        OrientationVerdict: with ``transpose_needed`` set only when the question was answered.

    """
    evidence: list[str] = []
    signals: dict[str, Any] = {}
    votes: list[tuple[str, str, float]] = []  # (signal name, vote, strength)
    conflicts: list[str] = []

    rows = _as_labels(row_labels)
    cols = _as_labels(col_labels)

    _vote_label_class(rows, cols, votes, signals, evidence)
    _vote_vocabulary(rows, cols, gene_vocabulary, votes, signals, evidence)
    _crosscheck_vocabulary(rows, cols, gene_vocabulary, signals, conflicts)
    _vote_zero_vectors(matrix, votes, signals, evidence)
    _note_shape(rows, cols, matrix, signals, evidence)

    # An internal disagreement is decisive in the negative direction: whichever signal is wrong,
    # the caller needs to know before anything gets transposed.
    if conflicts:
        evidence.append("REFUSED: independent signals disagree -- " + "; ".join(conflicts))
        return OrientationVerdict(
            orientation="ambiguous",
            transpose_needed=None,
            confidence=0.0,
            refused=True,
            evidence=evidence,
            signals=signals,
        )

    strong = [(n, vote, s) for n, vote, s in votes if s >= 0.8]
    moderate = [(n, vote, s) for n, vote, s in votes if 0.4 <= s < 0.8]
    decisive = strong or []

    if not decisive:
        reason = (
            "no signal positively identified both axes"
            if not moderate
            else f"only weak signals available ({', '.join(n for n, _, _ in moderate)})"
        )
        evidence.append(f"REFUSED: {reason}. Shape and plausible-looking names are not evidence of orientation.")
        return OrientationVerdict(
            orientation="ambiguous",
            transpose_needed=None,
            confidence=0.0,
            refused=True,
            evidence=evidence,
            signals=signals,
        )

    chosen = {vote for _, vote, _ in decisive}
    contradicting = {vote for _, vote, s in votes if s >= 0.4} - chosen
    if len(chosen) > 1 or contradicting:
        names = ", ".join(f"{n}={vote}" for n, vote, s in votes if s >= 0.4)
        evidence.append(f"REFUSED: signals contradict each other ({names}). Something is wrong with this input.")
        return OrientationVerdict(
            orientation="ambiguous",
            transpose_needed=None,
            confidence=0.0,
            refused=True,
            evidence=evidence,
            signals=signals,
        )

    orientation = decisive[0][1]
    confidence = min(0.99, max(s for _, _, s in decisive) + 0.05 * (len(decisive) - 1))
    evidence.append(f"RESOLVED as {orientation} on: {', '.join(f'{n} ({s:.2f})' for n, _, s in decisive)}.")
    return OrientationVerdict(
        orientation=orientation,
        transpose_needed=orientation != expect,
        confidence=confidence,
        refused=False,
        evidence=evidence,
        signals=signals,
    )


def _as_labels(labels: Sequence[Any] | None) -> list[str] | None:
    if labels is None:
        return None
    out = [str(x) for x in labels]
    return out or None


def _vote_vocabulary(
    rows: list[str] | None,
    cols: list[str] | None,
    vocabulary: Iterable[str] | None,
    votes: list[tuple[str, str, float]],
    signals: dict[str, Any],
    evidence: list[str],
) -> None:
    if vocabulary is None or not (rows or cols):
        return
    vocab = {str(g).upper() for g in vocabulary}
    if not vocab:
        return
    row = _vocabulary_presence(rows, vocab)
    col = _vocabulary_presence(cols, vocab)
    signals["gene_vocabulary"] = {"rows": row, "columns": col, "vocabulary_size": len(vocab)}
    evidence.append(
        f"gene vocabulary ({len(vocab)} genes): rows hold {row['matches']} of them "
        f"({row['share_of_vocabulary']:.0%} of the vocabulary, {row['share_of_axis']:.0%} of the axis); "
        f"columns hold {col['matches']} ({col['share_of_vocabulary']:.0%} of the vocabulary, "
        f"{col['share_of_axis']:.0%} of the axis)"
    )

    row_genes, col_genes = _axis_carries_genes(row), _axis_carries_genes(col)
    if row_genes and _axis_is_free_of_genes(col):
        votes.append(("gene_vocabulary", "genes_x_cells", 0.95))
    elif col_genes and _axis_is_free_of_genes(row):
        votes.append(("gene_vocabulary", "cells_x_genes", 0.95))
    elif row_genes and col_genes:
        evidence.append("both axes match the gene vocabulary, so it carries no orientation information")
    elif not (row_genes or col_genes):
        evidence.append("neither axis matches the gene vocabulary -- wrong species, or these are not gene labels")


# The reverse measure below is a share of a set that the caller chose the size of, so a four-gene
# marker list covered 100% by any axis that happens to contain four of anything would otherwise
# resolve an orientation. Enough of the vocabulary has to actually turn up for its coverage to mean
# something; a panel of this many genes landing entirely on one axis is not chance.
_VOCAB_MIN_MATCHES = 25


def _vocabulary_presence(labels: list[str] | None, vocab: set[str]) -> dict[str, Any]:
    """How much of this axis is in the vocabulary, and how much of the vocabulary is on this axis.

    Both directions are needed because either side can be the smaller one, and only one of them was
    ever measured. A share *of the axis* cannot exceed ``len(vocab) / len(axis)``, so a 300-gene
    panel against a 20000-gene axis topped out at 2% -- under every threshold here -- and the
    strongest signal this resolver has abstained on the ordinary case of a focused reference: a
    MERFISH panel, a marker list, an HVG-selected reference's ``var_names``. Coverage answers that
    case: 300 of 300 panel genes on one axis and none on the other identifies both axes outright.

    ``matches`` counts distinct labels, so a duplicated index cannot inflate either share.
    """
    if not labels:
        return {"share_of_axis": 0.0, "share_of_vocabulary": 0.0, "matches": 0}
    present = {x.upper() for x in labels} & vocab
    return {
        "share_of_axis": sum(1 for x in labels if x.upper() in vocab) / len(labels),
        "share_of_vocabulary": len(present) / len(vocab),
        "matches": len(present),
    }


def _axis_carries_genes(presence: dict[str, Any]) -> bool:
    """True when this axis is made of genes -- by either measure, since either can be the small one."""
    return presence["share_of_axis"] >= _VOCAB_HIT or (
        presence["share_of_vocabulary"] >= _VOCAB_HIT and presence["matches"] >= _VOCAB_MIN_MATCHES
    )


def _axis_is_free_of_genes(presence: dict[str, Any]) -> bool:
    """True when this axis is clean of the vocabulary on *both* measures.

    Voting requires one axis identified and the other ruled out. Ruling out on the axis share alone
    would let a long cell axis hide the whole vocabulary inside its own length.
    """
    return presence["share_of_axis"] < _VOCAB_MISS and presence["share_of_vocabulary"] < _VOCAB_MISS


def _crosscheck_vocabulary(
    rows: list[str] | None,
    cols: list[str] | None,
    vocabulary: Iterable[str] | None,
    signals: dict[str, Any],
    conflicts: list[str],
) -> None:
    """Catch a vocabulary that claims a cell-barcode axis is made of genes.

    Either the vocabulary is contaminated or the labels are not what they look like. Both are the
    caller's problem to fix, and neither is something to transpose a matrix through.
    """
    if vocabulary is None or "label_class" not in signals:
        return
    vocab = {str(g).upper() for g in vocabulary}
    if not vocab:
        return
    for axis, labels in (("rows", rows), ("columns", cols)):
        axis_class = signals["label_class"][axis]["class"]
        if axis_class not in _CELL_CLASSES:
            continue
        hit = _overlap(labels, vocab)
        if hit >= _VOCAB_HIT:
            conflicts.append(
                f"{hit:.0%} of the {axis} axis is in the supplied gene vocabulary, but those labels are "
                f"cell barcodes ({axis_class}) -- the vocabulary and the labels cannot both be right"
            )


def _overlap(labels: list[str] | None, vocab: set[str]) -> float:
    if not labels:
        return 0.0
    return sum(1 for x in labels if x.upper() in vocab) / len(labels)


def _classify_axis(labels: list[str] | None) -> tuple[str, dict[str, float]]:
    """Assign an axis to a label class, or 'mixed'/'none'."""
    if not labels:
        return "none", {}
    sample = labels[:2000]
    fractions = {
        "ensembl": sum(bool(_ENSEMBL_RE.match(x)) for x in sample) / len(sample),
        "barcode": sum(bool(_BARCODE_RE.match(x)) for x in sample) / len(sample),
        "positional": sum(bool(_SPOTID_RE.match(x) or _COORDID_RE.match(x) or x.isdigit()) for x in sample)
        / len(sample),
        "gene_symbol": sum(bool(_GENE_SYMBOL_RE.match(x) or _MT_GENE_RE.match(x)) for x in sample) / len(sample),
    }
    # An Ensembl ID also matches the loose symbol pattern; the specific class wins.
    if fractions["ensembl"] >= _CLASS_THRESHOLD:
        return "ensembl", fractions
    if fractions["barcode"] >= _CLASS_THRESHOLD:
        return "barcode", fractions
    if fractions["positional"] >= _CLASS_THRESHOLD:
        return "positional", fractions
    if fractions["gene_symbol"] >= _CLASS_THRESHOLD:
        return "gene_symbol", fractions
    return "mixed", fractions


# Which side of the matrix a label class points at. `positional` is deliberately absent: `spot_0`
# and `type_0` are the same string shape, and reading one as a spot cost a whole benchmark run.
_GENE_CLASSES = {"ensembl", "gene_symbol"}
_CELL_CLASSES = {"barcode"}


def _vote_label_class(
    rows: list[str] | None,
    cols: list[str] | None,
    votes: list[tuple[str, str, float]],
    signals: dict[str, Any],
    evidence: list[str],
) -> None:
    row_class, row_fracs = _classify_axis(rows)
    col_class, col_fracs = _classify_axis(cols)
    signals["label_class"] = {
        "rows": {"class": row_class, "fractions": {k: round(v, 3) for k, v in row_fracs.items()}},
        "columns": {"class": col_class, "fractions": {k: round(v, 3) for k, v in col_fracs.items()}},
    }
    evidence.append(f"row labels classify as {row_class!r}, column labels as {col_class!r}")

    row_is_gene = row_class in _GENE_CLASSES
    col_is_gene = col_class in _GENE_CLASSES

    # Both axes must be positively identified, and identified *differently*.
    if row_class in _GENE_CLASSES and col_class in _CELL_CLASSES:
        strength = 0.92 if row_class == "ensembl" else 0.85
        votes.append(("label_class", "genes_x_cells", strength))
        evidence.append(f"rows are {row_class} identifiers and columns are cell barcodes")
        return
    if col_class in _GENE_CLASSES and row_class in _CELL_CLASSES:
        strength = 0.92 if col_class == "ensembl" else 0.85
        votes.append(("label_class", "cells_x_genes", strength))
        evidence.append(f"columns are {col_class} identifiers and rows are cell barcodes")
        return
    if row_class in _GENE_CLASSES and col_class in _GENE_CLASSES:
        evidence.append("both axes look like gene identifiers -- the labels carry no orientation information")
        return
    if row_class in _CELL_CLASSES and col_class in _CELL_CLASSES:
        evidence.append("both axes look like cell barcodes -- the labels carry no orientation information")
        return
    if row_is_gene or col_is_gene:
        # One axis identified, the other unreadable. Suggestive, never sufficient.
        vote = "genes_x_cells" if row_is_gene else "cells_x_genes"
        votes.append(("label_class", vote, 0.5))
        other = col_class if row_is_gene else row_class
        evidence.append(f"one axis looks like gene identifiers but the other is {other!r} -- suggestive, not decisive")
        return
    evidence.append("neither axis could be assigned to a known label class")


def _vote_zero_vectors(
    matrix: Any | None,
    votes: list[tuple[str, str, float]],
    signals: dict[str, Any],
    evidence: list[str],
) -> None:
    """Unexpressed genes are normal; all-zero cells are filtered out before release.

    That asymmetry is real but weak, so this signal never exceeds moderate strength and therefore
    can never resolve an orientation on its own. Note what is *not* here: global sparsity and
    integer-ness are identical under transposition and carry no orientation information at all.
    """
    stats = _matrix_stats(matrix)
    if stats is None:
        return
    signals["matrix"] = stats
    zr, zc = stats["zero_row_fraction"], stats["zero_col_fraction"]
    evidence.append(
        f"all-zero vectors: {zr:.1%} of rows, {zc:.1%} of columns "
        f"(counts-like: {stats['integer_valued']}, density {stats['density']:.3f})"
    )
    gap = abs(zr - zc)
    if gap < 0.02 or max(zr, zc) < 0.02:
        return
    vote = "genes_x_cells" if zr > zc else "cells_x_genes"
    strength = 0.45 if gap >= 0.05 else 0.3
    votes.append(("zero_vectors", vote, strength))
    evidence.append(
        f"the {'row' if zr > zc else 'column'} axis holds more all-zero vectors, which is how genes behave -- "
        "supporting evidence only"
    )


#: Values examined for the integer-ness check. Not walking every value is deliberate -- the signal
#: is weak enough that a 40 GB matrix must not be read to produce it. What matters is that the
#: sample is spread across the matrix rather than taken off one end of it: same budget, no corner.
_MATRIX_VALUE_SAMPLE = 100_000
_MATRIX_VALUE_SAMPLE_AXIS = 500


def _matrix_stats(matrix: Any) -> dict[str, Any] | None:
    if matrix is None:
        return None
    try:
        import numpy as np
    except Exception:
        return None
    try:
        if hasattr(matrix, "getnnz") and hasattr(matrix, "shape"):  # scipy sparse
            n_rows, n_cols = matrix.shape
            row_nnz = matrix.getnnz(axis=1)
            col_nnz = matrix.getnnz(axis=0)
            nnz = int(matrix.nnz)
            # Strided rather than the leading values: `.data` is in storage order, so a prefix is
            # the first rows. A raw block concatenated with a normalised one reads as pure counts
            # off its opening rows, and "counts-like: True" over a normalised matrix is the kind of
            # confident wrong answer that decides which methods get run.
            data = matrix.data[:: max(1, matrix.data.size // _MATRIX_VALUE_SAMPLE)]
            integer_valued = bool(np.all(np.equal(np.mod(data, 1), 0))) if data.size else True
        else:
            arr = np.asarray(matrix)
            if arr.ndim != 2 or arr.size == 0:
                return None
            n_rows, n_cols = arr.shape
            nonzero = arr != 0
            row_nnz = nonzero.sum(axis=1)
            col_nnz = nonzero.sum(axis=0)
            nnz = int(nonzero.sum())
            # Both axes strided for the same reason: `arr[:500, :500]` judged a 2000 x 2000 matrix
            # by one quadrant of its top-left corner. Same number of values, drawn from everywhere.
            sub = arr[
                :: max(1, n_rows // _MATRIX_VALUE_SAMPLE_AXIS),
                :: max(1, n_cols // _MATRIX_VALUE_SAMPLE_AXIS),
            ]
            integer_valued = bool(np.all(np.equal(np.mod(sub[np.isfinite(sub)], 1), 0)))
        return {
            "shape": [int(n_rows), int(n_cols)],
            "density": float(nnz) / float(max(n_rows * n_cols, 1)),
            "zero_row_fraction": float(np.mean(np.asarray(row_nnz) == 0)),
            "zero_col_fraction": float(np.mean(np.asarray(col_nnz) == 0)),
            "integer_valued": integer_valued,
        }
    except Exception:
        return None


def _note_shape(
    rows: list[str] | None,
    cols: list[str] | None,
    matrix: Any | None,
    signals: dict[str, Any],
    evidence: list[str],
) -> None:
    """Record the shape ratio as context. It never votes -- see the module docstring."""
    n_rows = len(rows) if rows else None
    n_cols = len(cols) if cols else None
    if n_rows is None or n_cols is None:
        stats = signals.get("matrix")
        if stats and stats.get("shape"):
            n_rows, n_cols = stats["shape"]
    if not n_rows or not n_cols:
        return
    ratio = n_rows / n_cols
    signals["shape"] = {"n_rows": n_rows, "n_cols": n_cols, "row_to_col_ratio": round(ratio, 3), "voted": False}
    evidence.append(
        f"shape is {n_rows} x {n_cols} (ratio {ratio:.2f}); shape does not vote -- a 200-gene panel over "
        "3000 cells and a 3000-gene panel over 200 cells are the same shape"
    )
