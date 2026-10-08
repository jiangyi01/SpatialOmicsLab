import contextlib
import math
import os
import pickle
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
import tqdm


def json_safe(value: Any) -> Any:
    """Return *value* with every non-finite float replaced by ``None``, recursively.

    ``json.dumps`` writes ``float('nan')`` as the bare token ``NaN``, and the infinities as
    ``Infinity``/``-Infinity``. RFC 8259 has no literals for any of the three, so what gets written
    is not JSON. The reason this hides so well is that Python's own ``json.loads`` accepts those
    tokens by default -- so no reader on the Python side of the project ever complains -- while
    ``JSON.parse``, ``jq``, R's ``jsonlite`` and Arrow all reject the **whole document**. One metric
    that could not be computed therefore takes an entire run's report down with it, and nine files
    under the recorded results tree are unreadable that way today.

    Passing ``default=`` is not a guard: ``default`` is only consulted for objects json does not
    know how to serialise, and a NaN *is* a float, so it is serialised -- to an invalid token.

    ``None`` is the honest encoding rather than ``0.0``: a non-finite metric already meant "this
    could not be computed", and a zero would be read back as a measurement, which is the specific
    misreport being avoided. Two other copies of this rule already exist in the project and agree
    on it -- ``postanalysis.manifest._jsonable`` and ``huggingface_data.build_dataset.json_safe``.
    Neither covers the benchmark writers, which is where all nine bad files came from.

    numpy floats are swept too, and must be: ``numpy.float64`` subclasses ``float``, but
    ``numpy.float32``/``float16`` do not, and they are not too exotic to reach a writer --
    ``adata.X.mean()`` on a float32 matrix is one. A non-finite one does not fail loudly either:
    every writer that passes ``default=`` re-opens the hole, because ``default`` turns it into
    exactly the bad output this function exists to stop (``default=str`` writes the string
    ``"nan"``, and any ``float(obj)`` default writes the bare ``NaN`` token). numpy is looked up
    in ``sys.modules`` rather than imported: a numpy scalar can only be in *value* if numpy is
    already loaded, so the lookup never misses when it matters and numpy-free callers pay nothing.
    """
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.floating):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


#: The extensions the benchmarking readers have always taken at their word. Reads that produced
#: recorded scores must keep producing them, so those two names win over the file's own bytes.
#: Everything else -- notably ``.txt``, whose convention varies by tool -- is sniffed.
BENCHMARK_TRUSTED_EXTENSIONS = {".csv": ",", ".tsv": "\t"}

#: Compression this module can see through, mapped to the opener that yields text. Only formats
#: whose decompressor is in the standard library are listed: the point is to read one header line,
#: not to take on a dependency. ``.zip`` and ``.zst`` are absent for that reason -- pandas can read
#: them, so a caller may still hand one to ``read_csv``, but this function would have to guess the
#: separator rather than sniff it, and guessing is the failure mode the docstring below describes.
_TEXT_OPENERS = {".gz": "gzip", ".bz2": "bz2", ".xz": "lzma"}

#: What reading one header line through :func:`_open_maybe_compressed` can raise when the file is
#: unreadable. OSError covers the filesystem and both gzip's and bz2's *header* checks
#: (``BadGzipFile`` is an OSError), EOFError covers truncation -- but a corrupt *body* surfaces the
#: decompressor's own exception, and neither ``zlib.error`` (gzip) nor ``lzma.LZMAError`` (.xz)
#: subclasses OSError. zlib is compiled into every CPython this project supports; lzma is optional
#: at interpreter build time, so its absence must cost nothing when no ``.xz`` is ever read.
try:
    import lzma as _lzma

    _LZMA_ERRORS = (_lzma.LZMAError,)
except ImportError:  # pragma: no cover - a CPython built without liblzma
    _LZMA_ERRORS = ()
import zlib as _zlib

HEADER_READ_ERRORS = (OSError, EOFError, ValueError, _zlib.error, *_LZMA_ERRORS)


def _open_maybe_compressed(path):
    """Open *path* as text, transparently decompressing the formats in :data:`_TEXT_OPENERS`.

    A compressed file is not a different kind of file, it is the same file behind an encoding, and
    the only reason this exists is that ``open()`` on one yields the compression framing instead of
    the header. Read as text with ``errors="replace"`` that framing does not raise -- it comes back
    as replacement characters -- so the caller counts delimiters in binary noise and returns an
    answer with the same confidence as a real one. On a comma-separated ``.csv.gz`` that answer was
    a tab, which parses the whole table into one column named after the entire header line.
    """
    suffix = path.suffix.lower()
    module = _TEXT_OPENERS.get(suffix)
    if module is None:
        return open(path, encoding="utf-8", errors="replace")
    import importlib

    return importlib.import_module(module).open(path, "rt", encoding="utf-8", errors="replace")


def uncompressed_suffix(path):
    """The suffix *path* would have with any recognised compression stripped, lowercased.

    ``a.csv.gz`` -> ``.csv``; ``a.csv`` -> ``.csv``; ``a.nii.gz`` -> ``.nii``. Callers that decide
    what a file *is* from its extension have to ask this rather than ``Path.suffix``, which answers
    ``.gz`` for the first and the last alike and so cannot tell a gzipped table from a gzipped
    image volume.
    """
    path = Path(path)
    if path.suffix.lower() in _TEXT_OPENERS:
        return Path(path.stem).suffix.lower()
    return path.suffix.lower()


#: What ``pd.read_csv`` wants for "split on runs of whitespace". Named rather than spelled at the
#: return because it is two characters where the other answers are one: every consumer of this
#: module's sniffer passes the value straight to ``read_csv(sep=...)``, and none writes with it.
WHITESPACE_SEP = r"\s+"

#: How much of the first line the sniffer is allowed to hold. It only counts two characters, so the
#: length of the line is incidental to the answer -- but ``readline()`` with no argument holds all
#: of it, and reads to EOF when there is no newline at all. That is not a contrived file: a count
#: matrix written cells-as-columns has one field per cell, and a 400,000-cell reference is an
#: ordinary deconvolution input. Measured on one: a 3.1 MB header cost 6.2 MB of allocation to
#: decide "comma", in a process also holding the AnnData it is about to profile.
#:
#: A megabyte holds tens of thousands of fields, so the counts below separate on the same evidence
#: they had before for any header this size or smaller -- and for a header shorter than the cap the
#: string returned is byte-identical to the unbounded one.
_HEADER_MAX_CHARS = 1 << 20


def _is_number(text):
    try:
        float(text)
    except ValueError:
        return False
    return True


def _looks_whitespace_delimited(header):
    """Whether ``header`` is the first line of a bare numeric matrix written with spaces.

    Both halves of the test carry weight. Requiring no tab and no comma keeps every file the two
    counts above can already answer out of this branch, so nothing that reads correctly today
    changes. Requiring every field to parse as a number is what separates a matrix from a
    one-column list of two-word labels: ``B cells`` splits into two fields exactly as ``0.1 0.2``
    does, and only one of the two is a table. Drop that half and the branch reintroduces the
    failure the space exclusion exists to prevent.
    """
    if "\t" in header or "," in header:
        return False
    fields = header.split()
    return len(fields) > 1 and all(_is_number(field) for field in fields)


def sniff_tabular_sep(path, trust_extension=None, default=","):
    """Field separator for a delimited text file, decided by its first line.

    Tab and comma are decided first, and the header's larger count wins. Space is not a peer of
    those two: STRIDE writes cell types like ``B cells`` and ``CD14+ Monocytes`` into a
    tab-separated file, so its header carries more spaces than tabs and a whitespace-preferring
    sniffer would split six columns into twelve.

    Runs of whitespace are offered last and only to a header holding neither of the other two, on
    the terms in :func:`_looks_whitespace_delimited`. That is the ``numpy.savetxt`` default and a
    shipped worker writes it -- ``spaotsc``'s ``precomputed/dm_is.txt`` is a square distance matrix
    with one space between fields, sitting in a directory the staged-input rule does not filter --
    and without the branch its 200 columns read as one, with the first row of values as the column
    name. STRIDE cannot reach the branch, because its header has tabs.

    ``trust_extension`` maps a suffix to the separator to use without looking, for callers that
    must not change how a file already on disk is read -- pass
    :data:`BENCHMARK_TRUSTED_EXTENSIONS` for the historical ``.csv``/``.tsv`` behaviour. A caller
    with no such constraint passes nothing and gets the bytes. ``default`` is the answer when the
    header holds nothing to split on (a scalar, a one-name-per-line list) or cannot be read at all;
    there is nothing to sniff in those, so the caller's convention stands.

    The first line is read up to :data:`_HEADER_MAX_CHARS`; see that constant for why a bound is
    needed and why it cannot change an answer.

    One implementation on purpose. The rule previously existed twice and, separately, as a
    derive-it-from-the-extension shortcut in the diagnosis path; that third variant reported a
    300-cell count matrix named ``.txt`` but written with commas as a one-column file.
    """
    path = Path(path)
    suffix = uncompressed_suffix(path)
    if trust_extension and suffix in trust_extension:
        return trust_extension[suffix]
    try:
        with _open_maybe_compressed(path) as fh:
            header = fh.readline(_HEADER_MAX_CHARS)
    except HEADER_READ_ERRORS:
        return default
    if header.count("\t") > header.count(","):
        return "\t"
    if header.count(",") > header.count("\t"):
        return ","
    if _looks_whitespace_delimited(header):
        return WHITESPACE_SEP
    return default


@contextlib.contextmanager
def read_h5ad_backed(path):
    """Open *path* as a backed AnnData and guarantee its HDF5 handle is released.

    ``anndata.read_h5ad(path, backed="r")`` is the right call for profiling -- it reads ``obs`` and
    ``var`` without materializing a multi-GB ``.X`` -- but it keeps the file open, and HDF5 takes a
    lock. Whatever writes that path next then fails: ``OSError: unable to truncate a file which is
    already open`` in-process, or ``BlockingIOError: [Errno 11] ... unable to lock file`` from a
    worker subprocess.

    Letting the handle fall out of scope is not enough. An input carrying a ``.raw`` slot -- the
    scanpy convention for stashing counts before normalizing, so nearly every single-cell reference
    -- is a reference cycle, which refcounting cannot collect. The file then stays locked until the
    cyclic collector next runs, making the failure intermittent and dependent on the host's
    filesystem locking behaviour rather than on anything in the data.

    Usage::

        with read_h5ad_backed(path) as adata:
            columns = list(adata.obs.columns)

    Reading errors propagate to the caller; only the release is handled here.
    """
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        yield adata
    finally:
        # A file already closed, or an object whose backing was never opened, must not turn a
        # successful read into an error on the way out.
        with contextlib.suppress(Exception):
            adata.file.close()


def load_pickle(file):
    import pickle

    with open(file, "rb") as f:
        return pickle.load(f)


def save_pkl(f, filename):
    with open(filename, "wb") as file:
        pickle.dump(f, file)


def load_pkl(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def check_or_create_path(path=None):
    # Set a default path if none is provided
    if path is None:
        path = os.path.join(os.getcwd(), "tmp_directory")

    # Check if the path exists
    if not os.path.exists(path):
        # If it doesn't exist, create the directory
        os.makedirs(path)
        print(f"Directory created at: {path}")
    else:
        print(f"Directory already exists at: {path}")

    return path


def download_and_unzip(url: str, dest_dir: str) -> str:
    """Download a zip file from a URL and extract it to the destination directory.

    Args:
        url: The URL to download the zip file from.
        dest_dir: The directory to extract the contents to.

    Returns:
        The path to the extracted directory, or an error message.

    """
    tmp_zip_path = None
    try:
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Downloading from {url} ...")
        # (connect, read) timeout so a stalled server socket can't hang this download forever.
        # download_and_unzip is exported public API with no run_with_timeout wrapper (mirrors the
        # download_with_progress fix). The read timeout is per-chunk, so a progressing download completes.
        with requests.get(url, stream=True, timeout=(10, 60)) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            chunk_size = 8192
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                # Capture the path up front: NamedTemporaryFile(delete=False) has already created the
                # file on disk, so a mid-download failure (below) must still be cleaned up in `finally`.
                tmp_zip_path = tmp_file.name
                with tqdm.tqdm(
                    total=total_size / (1024**3),
                    unit="GB",
                    unit_scale=True,
                    desc="Downloading",
                    ncols=80,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            tmp_file.write(chunk)
                            pbar.update(len(chunk) / (1024**3))
        print(f"Downloaded to {tmp_zip_path}. Extracting...")
        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"Extraction complete to {dest_dir}")
        return dest_dir
    except Exception as e:
        print(f"Error downloading or extracting zip: {e}")
        return f"Error: {e}"
    finally:
        # Always remove the temp zip: on success after extraction, AND on any download/extract
        # failure (a corrupt or partially-downloaded zip used to leak into the temp dir because the
        # unlink only ran on the success path). Sibling check_and_download_s3_files cleans up the
        # same way via finally.
        if tmp_zip_path and os.path.exists(tmp_zip_path):
            try:
                os.unlink(tmp_zip_path)
            except OSError:
                pass


# The original public release bucket. It is gone — the object *and* the bucket root both answer 404 —
# so any download aimed at it can only print an error. Kept as a named constant rather than deleted so
# the value can be *recognised and refused* when it turns up in someone's environment. Must stay equal
# to ``sog_install.constants.DEFAULT_PUBLIC_DATA_URL`` (pinned by a test); spelled here
# instead of imported so this light module does not pull in the setup package.
RETIRED_PUBLIC_DATA_URL = "https://spatialomicsgym-release.s3.amazonaws.com"


def configured_benchmark_mirror() -> str | None:
    """The base URL to fetch benchmark datasets from, or ``None`` when there is nothing worth trying.

    ``STCoscientist.__init__`` used to fetch from :data:`RETIRED_PUBLIC_DATA_URL` unconditionally, so
    every construction on a machine without the benchmark corpus printed a 404 — and, offline, first
    blocked on the connect timeout. The completeness marker is ``benchmark/hle`` (the Biomni-era
    Humanity's-Last-Exam corpus), which no SpatialOmicsLab user has, so that was every construction.

    The capability itself is worth keeping: an institution can host its own copy and point
    ``SOG_PUBLIC_DATA_URL`` at it. So the rule is simply that a download is attempted when, and only
    when, someone has named a mirror that might actually answer. The retired bucket is refused even
    when set explicitly, because the usual way it gets set is being copied out of an old example file.
    """
    url = (os.environ.get("SOG_PUBLIC_DATA_URL") or "").strip().rstrip("/")
    if not url or url == RETIRED_PUBLIC_DATA_URL.rstrip("/"):
        return None
    return url


def check_and_download_s3_files(
    s3_bucket_url: str, local_data_lake_path: str, expected_files: list[str], folder: str = "data_lake"
) -> dict[str, bool]:
    """Check for missing files in the local data lake and download them from S3 bucket.

    Args:
        s3_bucket_url: Base URL of the S3 bucket (e.g., "https://spatialomicsgym-release.s3.amazonaws.com")
        local_data_lake_path: Local path to the data lake directory
        expected_files: List of expected file names in the data lake
        folder: S3 folder name ("data_lake" or "benchmark")

    Returns:
        Dictionary mapping file names to download success status
    """

    os.makedirs(local_data_lake_path, exist_ok=True)
    download_results = {}

    def download_with_progress(url: str, file_path: str, desc: str) -> bool:
        """Download file with progress bar."""
        try:
            # (connect, read) timeout: a stalled S3 socket must not hang STCoscientist.__init__,
            # which calls this synchronously with no run_with_timeout wrapper. The read timeout is
            # per-chunk (not a total budget), so a slow-but-progressing download still completes.
            response = requests.get(url, stream=True, timeout=(10, 60))
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(file_path, "wb") as f:
                if total_size > 0:
                    with tqdm.tqdm(total=total_size, unit="B", unit_scale=True, desc=desc, ncols=80) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            print(f"\u2717 Failed to download {desc}: {e}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            return False

    def cleanup_file(file_path: str):
        """Clean up file if it exists."""
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

    # Handle benchmark folder (download as zip)
    if folder == "benchmark":
        print(f"Downloading entire {folder} folder structure...")
        s3_zip_url = urljoin(s3_bucket_url + "/", folder + ".zip")

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
            tmp_zip_path = tmp_zip.name

            if download_with_progress(s3_zip_url, tmp_zip_path, f"{folder}.zip"):
                print(f"Extracting {folder}.zip...")
                try:
                    with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
                        zip_ref.extractall(local_data_lake_path)
                    print(f"\u2713 Successfully downloaded and extracted {folder} folder")
                    download_results = dict.fromkeys(expected_files, True)
                except Exception as e:
                    print(f"\u2717 Error extracting {folder}.zip: {e}")
                    download_results = dict.fromkeys(expected_files, False)
                finally:
                    cleanup_file(tmp_zip_path)
            else:
                download_results = dict.fromkeys(expected_files, False)

        return download_results

    # Handle data_lake folder (download individual files)
    for filename in expected_files:
        local_file_path = os.path.join(local_data_lake_path, filename)

        if os.path.exists(local_file_path):
            download_results[filename] = True
            continue

        s3_file_url = urljoin(s3_bucket_url + "/" + folder + "/", filename)
        print(f"Downloading {filename} from {folder}...")

        if download_with_progress(s3_file_url, local_file_path, filename):
            print(f"\u2713 Successfully downloaded: {filename}")
            download_results[filename] = True
        else:
            download_results[filename] = False

    return download_results
