"""Reads out of an ``.h5ad`` through an open descriptor, with h5py and numpy and nothing heavier.

The explorer's reader process (``sog_portal.vizchild``) is handed the dataset as an inherited
descriptor and never learns its path, so everything here starts from ``h5py.File(io.FileIO(fd,
closefd=False), "r")``. Nothing here resolves ``/proc/self/fd`` or calls ``os.path.realpath``: through a
descriptor either one answers with the real absolute path, and that is exactly what the reader must not
know (and must never print).

What it reads, and how much:

* **obsm** -- one key at a time, as a column slice ``[:, :dims]``, so a 50-component ``X_pca`` costs two
  columns. A DataFrame-valued key is read column by column: a plain numeric array with h5py, any other
  column through ``anndata.io.read_elem`` (that column's element only, never the frame's index) -- if the
  address-space cap leaves :data:`FALLBACK_HEADROOM_BYTES` to load anndata in; else :class:`TooLarge` names
  the cap.
* **obs** -- one column at a time, whole: categorical codes and their categories, numbers (nullable ones
  with NaN where the mask says missing), booleans, strings. A free-text string column longer than
  :data:`FREE_TEXT_MAX_ROWS` is not offered at all -- reading it means a Python string per row.
* **obs names** -- only the rows asked for, gathered in blocks of :data:`GATHER_BLOCK` rows; or, for a join
  against the whole index, every name one block at a time (:meth:`H5AD.obs_name_blocks`).
* **one gene** -- from X, a layer or ``raw/X``, with the bytes the read will touch estimated FIRST and the
  read refused above the caller's ceiling. A CSC matrix answers with one ``indptr`` slice; a CSR matrix is
  stored by rows, so one gene's values are spread over every row and the cost is the drawn rows' share of
  the matrix; a dense matrix is read as one column.
* **obsp** -- listed with each matrix's stored count from ``indptr[-1]`` alone; one matrix read whole as CSR, refused
  above the caller's count before ``indices`` or ``data`` are touched (the communication panels, ``viz.ccc``).
* **uns** -- a small provenance block (:meth:`H5AD.uns`), or one numeric array or one data frame with its size checked
  against the caller's cap first (:meth:`H5AD.uns_array`, :meth:`H5AD.uns_frame`).

Files written by anndata older than 0.8 store categoricals differently and are refused with what to do
(:class:`Unsupported`); nothing here guesses at an old layout.

Every error is one of three exceptions whose message is prose without a path: :class:`Unreadable`,
:class:`Unsupported` and :class:`TooLarge` (which names the setting that would lift it).
"""

from __future__ import annotations

import io
import math
import resource
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

#: obs names are gathered in blocks of this many rows, so a request for a thousand names reads a
#: thousand names' worth of blocks rather than the whole index.
GATHER_BLOCK = 262_144
#: A string column (not categorical) longer than this is free text -- an id per row -- and is not offered:
#: reading it builds a Python string per row before a single level can be counted.
FREE_TEXT_MAX_ROWS = 2_000_000
#: A labelled column with more distinct values than this is not drawn as colours: an identifier, or a grouping too
#: fine for any palette (:class:`TooManyLevels` says which, and the explorer's describe names it). The explorer's
#: default; ``SOG_VIZ_MAX_LEVELS`` sets its own (``sog_portal.vizchild.max_levels``).
MAX_LEVELS = 4096
#: Rows ``obsm`` is probed on for the proportions rule -- :func:`spatialomicsgym.viz.profile.obsm_facts`'s.
PROPORTION_PROBE_ROWS = 200
#: The HDF5 data-sieve buffer: a column read out of a contiguous row-major dataset fetches this much per
#: row once the rows are longer than it.
_SIEVE_BYTES = 64 * 1024
#: Two row ranges of a CSR matrix closer than this many stored values are read as one, so a sample spread
#: over a file costs a bounded number of reads instead of two per row.
_CSR_GAP = 16_384
#: HDF5's chunk cache per dataset. The 1 MiB default is smaller than the chunks h5py picks for a compressed
#: matrix, and a chunk that does not fit is decompressed again on every read that touches it -- once per drawn
#: row of a CSR gene read. Rows are read in file order, so one cached chunk is decompressed once.
_CHUNK_CACHE_BYTES = 32 * 1024 * 1024
#: Address space anndata's reader is never loaded with less of, under a cap. Loading it maps about 155 MiB under
#: python 3.12 -- scipy's own OpenBLAS among it, whose constructor retries for ever when the cap refuses its buffer
#: (measured 2026-10-01: with 80-120 MiB left the import spun until killed) -- and about 35 MiB under 3.11.
FALLBACK_HEADROOM_BYTES = 256 << 20
#: What uns is read to: enough for ``uns['spatial_3d']``'s provenance block, never an image.
_UNS_MAX_DEPTH = 4
_UNS_MAX_ITEMS = 256
_UNS_MAX_ARRAY = 10_000

_RESAVE = (
    "re-save it with a current anndata (read it with anndata.read_h5ad and write it again with write_h5ad), "
    "then open the new file"
)


class ReadError(Exception):
    """A read that cannot be answered. ``detail`` is prose with no path; ``knob`` names a setting or is None."""

    kind = "unreadable"

    def __init__(self, detail: str, *, knob: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.knob = knob


class Unreadable(ReadError):
    """The bytes are not a readable ``.h5ad``, or a part of one is broken."""

    kind = "unreadable"


class Unsupported(ReadError):
    """Readable, but not in a form the explorer draws (an old layout, a sparse obsm, an unknown key)."""

    kind = "unsupported"


class TooManyLevels(Unsupported):
    """A labelled column with more distinct values than :data:`MAX_LEVELS`: ``n_levels`` of them, over the ``n_rows``
    rows that have one -- so a caller can tell an identifier (a different value on every row) from a grouping too
    fine for a palette, and say which it left out."""

    def __init__(self, detail: str, *, n_levels: int, n_rows: int, knob: str | None = None) -> None:
        super().__init__(detail, knob=knob)
        self.n_levels = int(n_levels)
        self.n_rows = int(n_rows)


class TooLarge(ReadError):
    """More than a limit allows; ``knob`` names the limit."""

    kind = "too_large"


@dataclass(frozen=True)
class ObsmKey:
    """One obsm entry: its position among the sorted keys, its shape, and how it is stored.

    ``kind`` is ``"array"`` (a numeric dataset), ``"dataframe"`` (a DataFrame-valued key, ``columns`` set),
    ``"sparse"`` or ``"other"`` -- the last two are listed and never drawn.
    """

    index: int
    key: str
    shape: tuple[int, ...]
    kind: str
    columns: tuple[str, ...] | None = None

    @property
    def n_cols(self) -> int:
        return int(self.shape[1]) if len(self.shape) == 2 else 0


@dataclass(frozen=True)
class ObspKey:
    """One obsp entry: its position among the sorted keys, its shape, its encoding (``csr_matrix`` or ``csc_matrix``)
    and how many values it stores (``indptr[-1]``, one scalar read -- the listing reads no matrix)."""

    index: int
    key: str
    shape: tuple[int, ...]
    encoding: str
    nnz: int


@dataclass(frozen=True)
class ObsColumn:
    """One obs column as stored: its position in ``column-order``, its encoding and its storage kind.

    ``kind`` is ``"categorical"`` (codes + categories), ``"numeric"`` (numbers, nullable ones included),
    ``"bool"``, ``"string"`` (one string per row), ``"nullable-bool"`` or ``"unsupported"``; ``reason`` says
    why an ``unsupported`` column is not offered.
    """

    index: int
    name: str
    encoding: str
    kind: str
    dtype: str
    reason: str = ""


@dataclass
class Labels:
    """A labelled read: ``codes`` (int32, ``-1`` = missing) into ``labels`` (str, one per code)."""

    codes: np.ndarray
    labels: list[str]


# --------------------------------------------------------------------------------------------------
# Opening


def open_fd(fd: int) -> H5AD:
    """The ``.h5ad`` behind an inherited descriptor, opened read-only. The descriptor is not closed."""
    try:
        handle = h5py.File(
            io.FileIO(int(fd), mode="r", closefd=False), "r", rdcc_nbytes=_CHUNK_CACHE_BYTES, rdcc_nslots=10007
        )
    except (OSError, ValueError) as exc:
        raise Unreadable(
            "This file could not be opened as an HDF5 (.h5ad) file. It may be damaged, still being written, or "
            "another format."
        ) from exc
    return H5AD(handle)


def _attr_str(obj: Any, name: str, default: str = "") -> str:
    try:
        value = obj.attrs.get(name, default)
    except MemoryError:
        raise  # the cap, not the file: a default here would read as an old layout
    except Exception:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
    return str(value) if value is not None else default


def _native(array: np.ndarray) -> np.ndarray:
    """``array`` in this machine's byte order. A file can store big-endian numbers, and pandas cannot gather
    from a big-endian buffer; the same values in native order compare, print and parse the same."""
    dtype = array.dtype
    if dtype.kind in "biufc" and not dtype.isnative:
        return array.astype(dtype.newbyteorder("="))
    return array


def _strings(dataset: h5py.Dataset, selection: Any = ()) -> np.ndarray:
    """A string dataset (variable or fixed length) as an object array of ``str``."""
    if dataset.dtype.kind in "iufb":
        raise Unreadable("A column that should hold text holds numbers.")
    # Text that is not valid UTF-8 is shown with replacement characters rather than refused: a name is for
    # reading, and the row it belongs to is carried by its index, never by the text.
    return np.asarray(dataset.asstr(errors="replace")[selection], dtype=object)


class H5AD:
    """An open ``.h5ad``: listings are cheap (metadata only), and every value read is explicit."""

    def __init__(self, handle: h5py.File) -> None:
        self.file = handle
        obs = handle.get("obs")
        if obs is None:
            raise Unsupported("This file has no obs table, so it is not an AnnData .h5ad file.")
        if isinstance(obs, h5py.Dataset):
            # anndata before 0.7 stored obs as one compound record array.
            raise Unsupported(
                f"This file was written by an anndata older than 0.8, in a layout the explorer does not read; {_RESAVE}."
            )
        self._check_layout(obs)
        self._obs = obs
        index_name = _attr_str(obs, "_index", "_index")
        index = obs.get(index_name)
        if not isinstance(index, h5py.Dataset) or index.ndim != 1:
            raise Unreadable("This file's obs table has no row index, so its rows cannot be named or counted.")
        self._obs_index = index
        self.n_obs = int(index.shape[0])
        var = handle.get("var")
        var_index = var.get(_attr_str(var, "_index", "_index")) if isinstance(var, h5py.Group) else None
        self._var_index = var_index if isinstance(var_index, h5py.Dataset) and var_index.ndim == 1 else None
        self.n_vars = int(self._var_index.shape[0]) if self._var_index is not None else 0
        self._columns: list[ObsColumn] | None = None
        self._obsm: list[ObsmKey] | None = None
        self._raw_same: bool | None = None
        self._raw_read = False

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass

    @staticmethod
    def _check_layout(obs: h5py.Group) -> None:
        """Refuse the pre-0.8 layouts by what they wrote, before any value is read as if it were current."""
        legacy = "__categories" in obs or _attr_str(obs, "encoding-type") != "dataframe"
        if not legacy and _attr_str(obs, "encoding-version") == "0.1.0":
            legacy = True
        if not legacy:
            for name in obs:
                item = obs[name]
                if isinstance(item, h5py.Dataset) and "categories" in item.attrs:
                    legacy = True
                    break
        if legacy:
            raise Unsupported(
                f"This file was written by an anndata older than 0.8, which stores categories in a layout the "
                f"explorer does not read; {_RESAVE}."
            )

    # ---------------------------------------------------------------------------------------------- obsm
    def obsm_keys(self) -> list[ObsmKey]:
        """Every obsm entry, in sorted key order (the order ids number them in).

        A key whose name is not valid UTF-8 is not listed (:func:`_names`).
        """
        if self._obsm is not None:
            return self._obsm
        group = self.file.get("obsm")
        out: list[ObsmKey] = []
        if isinstance(group, h5py.Group):
            for i, key in enumerate(_names(group)):
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    kind = "array" if item.dtype.kind in "biuf" and item.ndim == 2 else "other"
                    out.append(ObsmKey(i, str(key), tuple(int(v) for v in item.shape), kind))
                    continue
                encoding = _attr_str(item, "encoding-type")
                if encoding == "dataframe":
                    columns = tuple(str(c) for c in _column_order(item))
                    index = item.get(_attr_str(item, "_index", "_index"))
                    n = int(index.shape[0]) if isinstance(index, h5py.Dataset) else self.n_obs
                    out.append(ObsmKey(i, str(key), (n, len(columns)), "dataframe", columns))
                elif encoding in ("csr_matrix", "csc_matrix"):
                    shape = tuple(int(v) for v in np.asarray(item.attrs.get("shape", ())).tolist())
                    out.append(ObsmKey(i, str(key), shape, "sparse"))
                else:
                    out.append(ObsmKey(i, str(key), (), "other"))
        self._obsm = out
        return out

    def obsm_key(self, key: str) -> ObsmKey | None:
        return next((k for k in self.obsm_keys() if k.key == key), None)

    def obsm(self, key: str, dims: int | None = None, rows: slice | None = None) -> np.ndarray:
        """``obsm[key][rows, :dims]`` as a 2-D numeric array (floats keep their width; integers become float64).

        ``rows`` is a slice (the proportions probe reads the first rows); ``None`` reads every row.
        """
        info = self.obsm_key(key)
        if info is None:
            raise Unsupported("That obsm key is not in this file.")
        sel_rows = rows if rows is not None else slice(None)
        if info.kind == "array":
            dataset = self.file["obsm"][key]
            if info.shape[0] != self.n_obs:
                raise Unreadable(f"obsm[{key!r}] has {info.shape[0]:,} rows, but the file has {self.n_obs:,}.")
            width = info.n_cols if dims is None else min(int(dims), info.n_cols)
            array = _native(np.asarray(dataset[sel_rows, :width]))
        elif info.kind == "dataframe":
            array = self._obsm_frame(key, info, dims, sel_rows)
        elif info.kind == "sparse":
            raise Unsupported(f"obsm[{key!r}] is a sparse matrix, which the explorer does not draw.")
        else:
            raise Unsupported(f"obsm[{key!r}] is not a numeric matrix.")
        if array.ndim != 2:
            raise Unsupported(f"obsm[{key!r}] is not a two-dimensional matrix.")
        if array.dtype.kind not in "f":
            array = array.astype(np.float64)
        return array

    def obsm_column(self, key: str, column: int) -> np.ndarray:
        """One column of ``obsm[key]`` for every row, float64: a column slice, not the whole matrix."""
        info = self.obsm_key(key)
        if info is None or not 0 <= int(column) < info.n_cols:
            raise Unsupported("That obsm column is not in this file.")
        if info.kind == "array":
            values = _native(np.asarray(self.file["obsm"][key][:, int(column)]))
            return values.astype(np.float64, copy=False)
        if info.kind == "dataframe":
            name = (info.columns or ())[int(column)]
            item = self.file["obsm"][key].get(name)
            if isinstance(item, h5py.Dataset) and item.dtype.kind in "biuf" and item.ndim == 1:
                return _native(np.asarray(item[()])).astype(np.float64, copy=False)
            return self.obsm(key)[:, int(column)]
        return self.obsm(key)[:, int(column)]

    def _obsm_frame(self, key: str, info: ObsmKey, dims: int | None, rows: slice) -> np.ndarray:
        group = self.file["obsm"][key]
        columns = list(info.columns or ())[: (len(info.columns or ()) if dims is None else int(dims))]
        if not columns:
            return np.empty((self.n_obs if rows == slice(None) else 0, 0))
        plain = [self._plain_column(group, c) for c in columns]
        if not all(plain):
            # A column that is not a plain number (a categorical, a nullable integer) takes anndata's own reader.
            left = _address_space_left()
            if "anndata" not in sys.modules and left is not None and left < FALLBACK_HEADROOM_BYTES:
                raise TooLarge(
                    f"obsm[{key!r}] is a table whose columns are not all plain numbers, which takes anndata's "
                    f"reader; loading it needs about {_bytes(FALLBACK_HEADROOM_BYTES)} of memory and {_bytes(left)} "
                    "is left under SOG_VIZ_MEMORY_BYTES. Raising SOG_VIZ_MEMORY_BYTES allows it.",
                    knob="SOG_VIZ_MEMORY_BYTES",
                )
        parts = [
            _native(np.asarray(group[c][rows])).astype(np.float64, copy=False)
            if ok
            else self._frame_column(key, group, c, rows)
            for c, ok in zip(columns, plain, strict=True)
        ]
        return np.column_stack(parts)

    def _plain_column(self, group: h5py.Group, name: str) -> bool:
        """Whether column ``name`` of a DataFrame-valued obsm key is a plain number per row -- read by slicing its
        dataset, never by anndata's reader (a categorical, a nullable integer)."""
        item = group.get(name)
        return isinstance(item, h5py.Dataset) and item.dtype.kind in "biuf" and item.shape == (self.n_obs,)

    def obsm_frame_is_plain(self, key: str) -> bool:
        """Whether the DataFrame-valued obsm ``key`` has columns and every one is a plain number per row
        (:meth:`_plain_column`). Reads no values."""
        info = self.obsm_key(key)
        if info is None or info.kind != "dataframe" or not info.columns:
            return False
        group = self.file["obsm"][key]
        return all(self._plain_column(group, c) for c in info.columns)

    def _frame_column(self, key: str, group: h5py.Group, name: str, rows: slice) -> np.ndarray:
        """One column of a DataFrame-valued obsm key through ``anndata.io.read_elem``, as float64 (NaN missing).

        That column's element only: reading the whole frame would also read its index -- a string per row -- and
        every column nobody asked for, for a probe of the first rows.
        """
        try:
            import pandas as pd
            from anndata.io import read_elem

            values = read_elem(group[name])
            if len(values) != self.n_obs:
                raise ValueError("not one value per row")
            return np.asarray(pd.Series(values[rows]).to_numpy(dtype=np.float64, na_value=np.nan))
        except MemoryError:
            raise  # the cap, not the table: as "not numbers" the key would be dropped without a word
        except Exception as exc:
            raise Unsupported(f"obsm[{key!r}] is a table whose columns are not all numbers.") from exc

    # ----------------------------------------------------------------------------------------------- obs
    def obs_columns(self) -> list[ObsColumn]:
        """Every obs column in ``column-order``, with how it is stored. Reads no values."""
        if self._columns is not None:
            return self._columns
        out: list[ObsColumn] = []
        for i, name in enumerate(_column_order(self._obs)):
            name = str(name)
            item = self._obs.get(name)
            if item is None:
                out.append(ObsColumn(i, name, "", "unsupported", "", "listed in column-order but not stored"))
                continue
            encoding = _attr_str(item, "encoding-type")
            if isinstance(item, h5py.Dataset):
                kind_char = item.dtype.kind
                if item.ndim != 1 or int(item.shape[0]) != self.n_obs:
                    out.append(ObsColumn(i, name, encoding, "unsupported", str(item.dtype), "not one value per row"))
                elif kind_char == "b":
                    out.append(ObsColumn(i, name, encoding or "array", "bool", "bool"))
                elif kind_char in "iuf":
                    out.append(ObsColumn(i, name, encoding or "array", "numeric", str(item.dtype)))
                elif kind_char in "OSU" or h5py.check_string_dtype(item.dtype) is not None:
                    if self.n_obs > FREE_TEXT_MAX_ROWS:
                        reason = f"free text over {FREE_TEXT_MAX_ROWS:,} rows"
                        out.append(ObsColumn(i, name, encoding or "string-array", "unsupported", "str", reason))
                    else:
                        out.append(ObsColumn(i, name, encoding or "string-array", "string", "str"))
                else:
                    out.append(ObsColumn(i, name, encoding, "unsupported", str(item.dtype), "not numbers or text"))
                continue
            if encoding == "categorical":
                out.append(ObsColumn(i, name, encoding, "categorical", "category"))
            elif encoding == "nullable-integer":
                out.append(ObsColumn(i, name, encoding, "numeric", "Int"))
            elif encoding == "nullable-boolean":
                out.append(ObsColumn(i, name, encoding, "nullable-bool", "boolean"))
            elif encoding == "nullable-string-array":
                if self.n_obs > FREE_TEXT_MAX_ROWS:
                    reason = f"free text over {FREE_TEXT_MAX_ROWS:,} rows"
                    out.append(ObsColumn(i, name, encoding, "unsupported", "string", reason))
                else:
                    out.append(ObsColumn(i, name, encoding, "string", "string"))
            else:
                out.append(
                    ObsColumn(
                        i,
                        name,
                        encoding,
                        "unsupported",
                        "",
                        f"an encoding the explorer does not read ({encoding or 'none'})",
                    )
                )
        self._columns = out
        return out

    def obs_column(self, name: str) -> ObsColumn | None:
        return next((c for c in self.obs_columns() if c.name == name), None)

    def _obs_item(self, column: ObsColumn) -> Any:
        item = self._obs.get(column.name)
        if item is None:
            raise Unreadable(f"obs[{column.name!r}] is listed but not stored.")
        return item

    def obs_labels(self, column: ObsColumn, rows: np.ndarray | None = None) -> Labels:
        """A labelled column (categorical, bool, nullable bool, strings) as codes into labels.

        Labels are what ``series.astype(str)`` prints for each category, so a numeric category reads as the
        figures print it. Missing rows are ``-1``. ``rows`` (original row indices) picks rows after the read.
        """
        item = self._obs_item(column)
        if column.kind == "categorical":
            codes = _native(np.asarray(item["codes"][()])).astype(np.int32, copy=False)
            categories = item["categories"]
            labels = _category_labels(categories)
            if codes.shape != (self.n_obs,):
                raise Unreadable(f"obs[{column.name!r}] has codes for {codes.shape[0]:,} rows, not {self.n_obs:,}.")
            if codes.size and (int(codes.max()) >= len(labels) or int(codes.min()) < -1):
                raise Unreadable(f"obs[{column.name!r}] has codes outside its categories.")
            out = Labels(codes, labels)
        elif column.kind == "bool":
            values = np.asarray(item[()], dtype=bool)
            out = Labels(values.astype(np.int32), ["False", "True"])
        elif column.kind == "nullable-bool":
            values = np.asarray(item["values"][()], dtype=bool).astype(np.int32)
            values[np.asarray(item["mask"][()], dtype=bool)] = -1
            out = Labels(values, ["False", "True"])
        elif column.kind == "string":
            out = self._string_labels(column, item)
        else:
            raise Unsupported(f"obs[{column.name!r}] is not a labelled column.")
        if rows is not None:
            out = Labels(out.codes[rows], out.labels)
        return out

    def _string_labels(self, column: ObsColumn, item: Any) -> Labels:
        import pandas as pd

        if column.encoding == "nullable-string-array":
            values = _strings(item["values"])
            mask = np.asarray(item["mask"][()], dtype=bool)
        else:
            values = _strings(item)
            mask = None
        codes, uniques = pd.factorize(values, use_na_sentinel=True)
        codes = codes.astype(np.int32, copy=False)
        if mask is not None:
            codes[mask] = -1
        return Labels(codes, [str(u) for u in uniques])

    def obs_numbers(self, column: ObsColumn, rows: np.ndarray | None = None) -> np.ndarray:
        """A numeric column as float64, NaN where missing (a nullable column's mask included)."""
        item = self._obs_item(column)
        if column.kind != "numeric":
            raise Unsupported(f"obs[{column.name!r}] is not numeric.")
        if column.encoding == "nullable-integer":
            values = _native(np.asarray(item["values"][()])).astype(np.float64)
            values[np.asarray(item["mask"][()], dtype=bool)] = np.nan
        else:
            values = _native(np.asarray(item[()]))
            values = values.astype(np.float64, copy=False) if values.dtype != np.float64 else values
        return values if rows is None else values[rows]

    def obs_raw(self, column: ObsColumn) -> tuple[np.ndarray, np.ndarray | None]:
        """``(values as stored, missing mask or None)`` of a numeric column, in native byte order.

        The stored dtype is kept because it decides how a value prints -- a float32 ``0.1`` prints ``0.1``,
        its float64 cast does not -- and a level label must print as the figures print it.
        """
        item = self._obs_item(column)
        if column.kind != "numeric":
            raise Unsupported(f"obs[{column.name!r}] is not numeric.")
        if column.encoding == "nullable-integer":
            return _native(np.asarray(item["values"][()])), np.asarray(item["mask"][()], dtype=bool)
        return _native(np.asarray(item[()])), None

    def obs_series(self, name: str) -> Any:
        """One obs column as the pandas Series anndata would give, for the shared ``viz.layers`` helpers."""
        import pandas as pd

        column = self.obs_column(name)
        if column is None:
            raise KeyError(name)
        item = self._obs_item(column)
        if column.kind == "categorical":
            codes = _native(np.asarray(item["codes"][()]))
            categories = item["categories"]
            if isinstance(categories, h5py.Dataset) and categories.dtype.kind in "biuf":
                cats: Any = _native(np.asarray(categories[()]))
            else:
                cats = _strings(categories)
            ordered = bool(item.attrs.get("ordered", False))
            return pd.Series(pd.Categorical.from_codes(codes, categories=cats, ordered=ordered))
        if column.kind == "numeric":
            if column.encoding == "nullable-integer":
                values = _native(np.asarray(item["values"][()]))
                mask = np.asarray(item["mask"][()], dtype=bool)
                return pd.Series(pd.arrays.IntegerArray(values, mask))
            return pd.Series(_native(np.asarray(item[()])))
        if column.kind == "bool":
            return pd.Series(np.asarray(item[()], dtype=bool))
        if column.kind == "nullable-bool":
            values = np.asarray(item["values"][()], dtype=bool)
            mask = np.asarray(item["mask"][()], dtype=bool)
            return pd.Series(pd.arrays.BooleanArray(values, mask))
        if column.kind == "string":
            if column.encoding == "nullable-string-array":
                values = _strings(item["values"])
                values[np.asarray(item["mask"][()], dtype=bool)] = None
                return pd.Series(pd.array(values, dtype="string"))
            return pd.Series(_strings(item))
        raise KeyError(name)

    # ----------------------------------------------------------------------------------------- obs names
    def obs_names(self, rows: Sequence[int] | np.ndarray) -> list[str]:
        """The obs names of ``rows`` (original row indices), in the order given, repeats included."""
        wanted = np.asarray(rows, dtype=np.int64)
        if wanted.size == 0:
            return []
        if int(wanted.min()) < 0 or int(wanted.max()) >= self.n_obs:
            raise Unsupported("A row asked for is not in this file.")
        unique = np.unique(wanted)
        found: dict[int, str] = {}
        block = unique // GATHER_BLOCK
        for b in np.unique(block):
            in_block = unique[block == b]
            lo, hi = int(in_block[0]), int(in_block[-1]) + 1
            if (hi - lo) <= 64 * in_block.size:
                # Dense enough: one contiguous read of the span.
                span = _strings(self._obs_index, slice(lo, hi))
                picked = span[in_block - lo]
            else:
                # Sparse: a point selection, which HDF5 reads without decoding the rows between.
                picked = _strings(self._obs_index, in_block)
            found.update(zip(in_block.tolist(), (str(v) for v in picked), strict=True))
        return [found[int(r)] for r in wanted]

    def obs_name_blocks(self, block: int = GATHER_BLOCK) -> Iterator[tuple[int, np.ndarray]]:
        """Every obs name in file order, ``block`` rows at a time: ``(first row, object array of str)``.

        For a join against the whole index (the explorer's result tables): one pass over it, and never more
        than one block's names held at once -- a list of every name of a twenty-million-row file is over a
        gigabyte of Python strings.
        """
        step = max(1, int(block))
        for start in range(0, self.n_obs, step):
            yield start, _strings(self._obs_index, slice(start, min(self.n_obs, start + step)))

    def var_names(self) -> list[str]:
        """Every var name, in var order. Empty when the file has no var table."""
        if self._var_index is None or self.n_vars == 0:
            return []
        return [str(v) for v in _strings(self._var_index)]

    # ---------------------------------------------------------------------------------------------- genes
    def gene_slots(self) -> list[str]:
        """The matrices a gene can be read from: ``"X"``, ``"raw"``, and ``"layers:<name>"`` per layer.

        A layer whose name holds characters outside ``[A-Za-z0-9_-]`` is not listed: the slot travels inside
        an id the portal checks against that alphabet. A slot is listed only when its shape is
        (n_obs, n_vars) -- and raw only when its own var table holds exactly var's genes in var's order
        (:meth:`raw_has_own_genes`): a gene is found by searching var's names, so a raw with genes of its own
        would offer, under raw, genes it may not hold, and never the ones only it holds.
        """
        if self.n_vars == 0:
            return []
        slots: list[str] = []
        if self._matrix_shape(self.file.get("X")) == (self.n_obs, self.n_vars):
            slots.append("X")
        if self._raw_matches_var() is True:
            slots.append("raw")
        layers = self.file.get("layers")
        if isinstance(layers, h5py.Group):
            for name in _names(layers):
                if _slot_name_ok(name) and self._matrix_shape(layers[name]) == (self.n_obs, self.n_vars):
                    slots.append(f"layers:{name}")
        return slots

    def raw_has_own_genes(self) -> bool:
        """Whether the file has a raw matrix, with n_obs rows and a var table, whose genes are not var's genes in
        var's order -- the scanpy convention of ``.raw`` set before subsetting to the highly variable genes. Such a
        raw is not a gene slot (:meth:`gene_slots`), and describe says why."""
        return self._raw_matches_var() is False

    def _raw_matches_var(self) -> bool | None:
        """``True`` when raw.X has n_obs rows and n_vars columns and raw's var index is var's, name for name;
        ``False`` for a raw with a var table of its own; ``None`` when there is no such raw. Read once."""
        if self._raw_read:
            return self._raw_same
        same: bool | None = None
        raw = self.file.get("raw")
        if isinstance(raw, h5py.Group) and self._var_index is not None:
            shape = self._matrix_shape(raw.get("X"))
            raw_var = raw.get("var")
            if shape is not None and shape[0] == self.n_obs and isinstance(raw_var, h5py.Group):
                index = raw_var.get(_attr_str(raw_var, "_index", "_index"))
                try:
                    same = (
                        isinstance(index, h5py.Dataset)
                        and index.ndim == 1
                        and shape[1] == self.n_vars == int(index.shape[0])
                        and bool(np.array_equal(_strings(index), _strings(self._var_index)))
                    )
                except (Unreadable, OSError, KeyError, ValueError, TypeError):
                    same = None  # a raw whose genes cannot be read is no slot -- and no reason to refuse the file
        self._raw_same, self._raw_read = same, True
        return same

    @staticmethod
    def _matrix_shape(item: Any) -> tuple[int, int] | None:
        if isinstance(item, h5py.Dataset):
            return (int(item.shape[0]), int(item.shape[1])) if item.ndim == 2 and item.dtype.kind in "biuf" else None
        if isinstance(item, h5py.Group) and _attr_str(item, "encoding-type") in ("csr_matrix", "csc_matrix"):
            shape = np.asarray(item.attrs.get("shape", ())).tolist()
            return (int(shape[0]), int(shape[1])) if len(shape) == 2 else None
        return None

    def _slot_matrix(self, slot: str) -> Any:
        if slot == "X":
            return self.file.get("X")
        if slot == "raw":
            return self.file.get("raw/X")
        if slot.startswith("layers:"):
            layers = self.file.get("layers")
            return layers.get(slot[len("layers:") :]) if isinstance(layers, h5py.Group) else None
        return None

    def _slot_column(self, slot: str, j: int) -> int:
        """The column of ``slot``'s matrix that holds var ``j``.

        raw has its OWN var table -- the scanpy convention sets ``.raw`` before subsetting to the highly
        variable genes -- so var ``j`` is found in raw's table by name; a position taken from var and used on
        raw.X would read another gene's column (``viz.layers.gene_values`` says the same).
        """
        if not 0 <= j < self.n_vars:
            raise Unsupported("That gene is not in this file.")
        if slot != "raw" or self._raw_matches_var() is True:
            return j  # a raw whose genes are var's, in var's order, holds gene j at column j
        name = str(_strings(self._var_index, slice(j, j + 1))[0])
        raw_var = self.file["raw/var"]
        raw_index = raw_var.get(_attr_str(raw_var, "_index", "_index"))
        if not isinstance(raw_index, h5py.Dataset):
            raise Unreadable("raw has no var index.")
        names = _strings(raw_index)
        hits = np.flatnonzero(names == name)
        if hits.size == 0:
            raise Unsupported(f"{name!r} is not in .raw, which holds its own, different list of genes.")
        if hits.size > 1:
            raise Unsupported(f"{name!r} appears {hits.size} times in .raw, so which column to read is ambiguous.")
        return int(hits[0])

    def gene_cost(self, slot: str, j: int, rows: np.ndarray) -> tuple[int, str]:
        """``(bytes the read of var j for these rows touches, how the matrix is stored)``, before any read.

        Uncompressed bytes: a compressed file touches fewer on disk and decompresses the same.
        """
        matrix = self._slot_matrix(slot)
        if matrix is None:
            raise Unsupported("That matrix is not in this file.")
        column = self._slot_column(slot, j)
        return self._cost(matrix, column, rows)[:2]

    def _cost(self, matrix: Any, column: int, rows: np.ndarray) -> tuple[int, str, Any]:
        if isinstance(matrix, h5py.Dataset):
            n, m = int(matrix.shape[0]), int(matrix.shape[1])
            item = int(matrix.dtype.itemsize)
            if matrix.chunks:
                chunk_rows, chunk_cols = int(matrix.chunks[0]), int(matrix.chunks[1])
                cost = math.ceil(n / chunk_rows) * chunk_rows * chunk_cols * item
            else:
                # Contiguous: HDF5 fetches a sieve buffer per row once rows are longer than it, else the
                # whole matrix passes through the sieve.
                cost = min(n * m * item, n * _SIEVE_BYTES)
            return cost, "dense", None
        encoding = _attr_str(matrix, "encoding-type")
        data, indices, indptr = matrix["data"], matrix["indices"], matrix["indptr"]
        if encoding == "csc_matrix":
            lo, hi = (int(v) for v in indptr[column : column + 2])
            spans = [(lo, hi)] if hi > lo else []
            return _touched(data, spans) + _touched(indices, spans), "csc", (lo, hi)
        if encoding == "csr_matrix":
            pointers = _native(np.asarray(indptr[()])).astype(np.int64, copy=False)
            spans = _csr_spans(pointers, np.unique(np.asarray(rows, dtype=np.int64)))
            cost = _touched(data, spans) + _touched(indices, spans) + pointers.nbytes
            return cost, "csr", (pointers, spans)
        raise Unsupported("That matrix is stored in a form the explorer does not read.")

    def gene_values(self, slot: str, j: int, rows: np.ndarray, *, max_read_bytes: int) -> np.ndarray:
        """Var ``j`` of ``slot`` for ``rows`` (original row indices, in that order), float32.

        Refused with :class:`TooLarge` (``SOG_VIZ_GENE_MAX_READ_BYTES``) before reading when the bytes it
        would touch are above ``max_read_bytes``.
        """
        matrix = self._slot_matrix(slot)
        if matrix is None:
            raise Unsupported("That matrix is not in this file.")
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= self.n_obs):
            raise Unsupported("A row asked for is not in this file.")
        column = self._slot_column(slot, j)
        cost, layout, plan = self._cost(matrix, column, rows)
        if cost > int(max_read_bytes):
            raise TooLarge(
                _gene_refusal(cost, layout, int(max_read_bytes), rows.size), knob="SOG_VIZ_GENE_MAX_READ_BYTES"
            )
        if layout == "dense":
            values = _native(np.asarray(matrix[:, column]))
            return values[rows].astype(np.float32)
        data, indices = matrix["data"], matrix["indices"]
        if layout == "csc":
            lo, hi = plan
            full = np.zeros(self.n_obs, dtype=np.float32)
            where = _native(np.asarray(indices[lo:hi])).astype(np.int64, copy=False)
            if where.size and (int(where.min()) < 0 or int(where.max()) >= self.n_obs):
                raise Unreadable("This matrix has a row index outside the file.")
            # A non-canonical matrix can store one cell twice; scipy sums the two, and so does this.
            np.add.at(full, where, _native(np.asarray(data[lo:hi])).astype(np.float32))
            return full[rows]
        pointers, spans = plan
        # A merged span also covers rows nobody asked for; their values land in ``full`` and are never returned.
        full = np.zeros(self.n_obs, dtype=np.float32)
        for lo, hi in spans:
            hits = np.flatnonzero(_native(np.asarray(indices[lo:hi])) == column)
            if not hits.size:
                continue
            row_of = np.searchsorted(pointers, hits + lo, side="right") - 1
            np.add.at(full, row_of, _native(np.asarray(data[lo:hi]))[hits].astype(np.float32))
        return full[rows]

    # ------------------------------------------------------------------------------------------------ uns
    def uns(self, key: str) -> Any:
        """``uns[key]`` read to a small plain value: dicts, strings, numbers, short lists. ``None`` if absent.

        Bounded (depth, items, array length), because uns can hold an image or a whole result table, and the
        only reason to read it here is a provenance block of a few keys.
        """
        group = self.file.get("uns")
        if not isinstance(group, h5py.Group) or key not in group:
            return None
        return _read_small(group[key], 0)

    def uns_group(self, key: str) -> h5py.Group | None:
        """``uns[key]`` when it is a group (a dict or a data frame anndata wrote), else ``None``. Reads no values."""
        group = self.file.get("uns")
        item = group.get(key) if isinstance(group, h5py.Group) else None
        return item if isinstance(item, h5py.Group) else None

    def _uns_member(self, key: str, member: str | None) -> Any:
        group = self.file.get("uns")
        item = group.get(key) if isinstance(group, h5py.Group) else None
        if member is not None:
            item = item.get(member) if isinstance(item, h5py.Group) else None
        return item

    def uns_array(self, key: str, member: str | None, *, max_values: int) -> np.ndarray | None:
        """``uns[key][member]`` (``uns[key]`` when ``member`` is None) as a numeric array in native byte order.

        ``None`` when it is absent, not numbers, or more than three-dimensional -- squidpy's co-occurrence (K, K, bins)
        is the deepest a facet reads. Refused with :class:`TooLarge` above ``max_values`` values, before the read:
        uns can hold a whole simulation (``sims_stat``) or an image.
        """
        item = self._uns_member(key, member)
        if not isinstance(item, h5py.Dataset) or item.dtype.kind not in "biuf" or item.ndim > 3:
            return None
        size = int(np.prod(item.shape, dtype=np.int64)) if item.ndim else 1
        if size > int(max_values):
            raise TooLarge(
                f"uns[{key!r}] holds {size:,} values, more than the {int(max_values):,} this read allows.",
                # ``max_values`` is the caller's fixed cap, which no setting lifts: no knob is named.
                knob=None,
            )
        return _native(np.asarray(item[()]))

    def uns_frame(self, key: str, member: str | None, *, max_rows: int) -> dict[str, np.ndarray] | None:
        """``uns[key][member]`` -- a data frame anndata wrote -- as ``{column: array}`` in ``column-order``.

        Numeric columns as stored (native byte order), categoricals as their labels (what ``astype(str)`` prints, as
        :meth:`obs_labels` reads them; a missing value is ``None``), text as ``str``. ``None`` when it is absent or no
        data frame; a column in another encoding is left out. Refused with :class:`TooLarge` above ``max_rows`` rows,
        before a value is read.
        """
        item = self._uns_member(key, member)
        if not isinstance(item, h5py.Group) or _attr_str(item, "encoding-type") != "dataframe":
            return None
        index = item.get(_attr_str(item, "_index", "_index"))
        n = int(index.shape[0]) if isinstance(index, h5py.Dataset) and index.ndim == 1 else None
        if n is None:
            raise Unreadable(f"uns[{key!r}] is a table with no row index.")
        if n > int(max_rows):
            raise TooLarge(
                f"uns[{key!r}] is a table of {n:,} rows, more than the {int(max_rows):,} this read allows.",
                # ``max_rows`` is the caller's fixed cap (``_LIGREC_MAX_ROWS``, ``_UNS_MAX_VALUES``): no knob lifts it.
                knob=None,
            )
        out: dict[str, np.ndarray] = {}
        for name in _column_order(item):
            column = item.get(name)
            encoding = _attr_str(column, "encoding-type") if column is not None else ""
            if isinstance(column, h5py.Dataset) and column.ndim == 1 and int(column.shape[0]) == n:
                if column.dtype.kind in "biuf":
                    out[name] = _native(np.asarray(column[()]))
                elif column.dtype.kind in "OSU" or h5py.check_string_dtype(column.dtype) is not None:
                    out[name] = _strings(column)
            elif isinstance(column, h5py.Group) and encoding == "categorical":
                codes = _native(np.asarray(column["codes"][()])).astype(np.int64, copy=False)
                labels = np.asarray(_category_labels(column["categories"]), dtype=object)
                if codes.shape != (n,) or (codes.size and int(codes.max()) >= len(labels)):
                    raise Unreadable(f"uns[{key!r}] has a categorical column whose codes do not fit it.")
                values = np.empty(n, dtype=object)
                values[codes >= 0] = labels[codes[codes >= 0]]
                out[name] = values
        return out

    # ----------------------------------------------------------------------------------------------- obsp
    def obsp_keys(self) -> list[ObspKey]:
        """Every sparse obsp matrix (CSR or CSC), in sorted key order, with its stored count. Reads one scalar each."""
        group = self.file.get("obsp")
        out: list[ObspKey] = []
        if isinstance(group, h5py.Group):
            for key in _names(group):
                item = group[key]
                encoding = _attr_str(item, "encoding-type") if isinstance(item, h5py.Group) else ""
                if encoding not in ("csr_matrix", "csc_matrix") or not isinstance(item.get("indptr"), h5py.Dataset):
                    continue
                pointers = item["indptr"]
                nnz = int(pointers[pointers.shape[0] - 1]) if pointers.ndim == 1 and pointers.shape[0] else 0
                shape = tuple(int(v) for v in np.asarray(item.attrs.get("shape", ())).tolist())
                out.append(ObspKey(len(out), str(key), shape, encoding, nnz))
        return out

    def obsp_csr(self, key: str, *, max_nnz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """``obsp[key]`` as ``(indptr int64, indices int64, data float64, transposed)``.

        A CSR matrix is answered as stored. A CSC matrix stores its columns the way a CSR stores its rows, so read as
        stored it is the CSR of the TRANSPOSE: it is answered that way, ``transposed=True``, and the caller swaps rows
        for columns -- converting would mean a sort of every stored value. Refused with :class:`TooLarge` above
        ``max_nnz`` stored values (from ``indptr[-1]``, before ``indices`` or ``data`` are read), and
        :class:`Unreadable` when the matrix is not n_obs x n_obs or its pointers do not fit its values.
        """
        info = next((k for k in self.obsp_keys() if k.key == key), None)
        if info is None:
            raise Unsupported("That obsp key is not in this file.")
        if info.nnz > int(max_nnz):
            raise TooLarge(
                f"obsp[{key!r}] stores {info.nnz:,} values, more than the {int(max_nnz):,} the explorer reads from one "
                "matrix. Raising SOG_VIZ_MEMORY_BYTES allows more.",
                knob="SOG_VIZ_MEMORY_BYTES",
            )
        if info.shape != (self.n_obs, self.n_obs):
            raise Unreadable(f"obsp[{key!r}] is {info.shape}, not {self.n_obs:,} x {self.n_obs:,} like the file.")
        group = self.file["obsp"][key]
        indptr = _native(np.asarray(group["indptr"][()])).astype(np.int64, copy=False)
        indices = _native(np.asarray(group["indices"][()])).astype(np.int64, copy=False)
        data = _native(np.asarray(group["data"][()])).astype(np.float64, copy=False)
        if (
            indptr.shape != (self.n_obs + 1,)
            or indices.shape != data.shape
            or int(indptr[-1]) != indices.size
            or (indptr.size > 1 and bool(np.any(np.diff(indptr) < 0)))
            or (indices.size and (int(indices.min()) < 0 or int(indices.max()) >= self.n_obs))
        ):
            raise Unreadable(f"obsp[{key!r}] is a sparse matrix whose pointers do not fit its values.")
        return indptr, indices, data, info.encoding == "csc_matrix"


def _address_space_left() -> int | None:
    """Bytes this process can still map under its ``RLIMIT_AS``, or ``None`` when it has no cap (or cannot say)."""
    soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    if soft == resource.RLIM_INFINITY:
        return None
    try:
        with open("/proc/self/status", "rb") as handle:
            size = next((int(line.split()[1]) * 1024 for line in handle if line.startswith(b"VmSize:")), None)
    except (OSError, ValueError, IndexError):
        return None
    return None if size is None else int(soft) - size


def _names(group: h5py.Group) -> list[str]:
    """A group's member names, sorted. h5py hands a name that is not valid UTF-8 back as ``bytes``, which does not
    sort beside the ``str`` names (a TypeError for the whole listing); such a member is left out instead."""
    return sorted(name for name in group if isinstance(name, str))


def _slot_name_ok(name: str) -> bool:
    return 0 < len(name) <= 64 and all(ch.isascii() and (ch.isalnum() or ch in "_-") for ch in name)


def _column_order(frame: h5py.Group) -> list[str]:
    order = frame.attrs.get("column-order", ())
    values = np.asarray(order).ravel().tolist() if order is not None else []
    out = []
    for v in values:
        out.append(v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v))
    return out


def _category_labels(categories: Any) -> list[str]:
    """What ``series.astype(str)`` prints for each category: the string itself, or pandas' print of a number."""
    if not isinstance(categories, h5py.Dataset):
        raise Unsupported("A categorical column's categories are stored in a form the explorer does not read.")
    if categories.dtype.kind in "biuf":
        import pandas as pd

        values = _native(np.asarray(categories[()]))
        return [str(v) for v in pd.Series(values).astype(str).tolist()]
    return [str(v) for v in _strings(categories)]


def _csr_spans(pointers: np.ndarray, rows: np.ndarray) -> list[tuple[int, int]]:
    """The stored-value ranges that cover ``rows`` (sorted, distinct), neighbours closer than :data:`_CSR_GAP`
    merged into one read."""
    if rows.size == 0:
        return []
    starts = pointers[rows]
    stops = pointers[rows + 1]
    # A new read starts where the gap from the previous row's end is wider than _CSR_GAP.
    breaks = np.flatnonzero(starts[1:] - stops[:-1] > _CSR_GAP) + 1
    first = np.concatenate([[0], breaks])
    last = np.concatenate([breaks - 1, [rows.size - 1]])
    return [
        (int(starts[a]), int(stops[b]))
        for a, b in zip(first.tolist(), last.tolist(), strict=True)
        if stops[b] > starts[a]
    ]


def _touched(dataset: h5py.Dataset, spans: list[tuple[int, int]]) -> int:
    """Uncompressed bytes of ``dataset`` a read of ``spans`` (sorted, disjoint) passes through.

    A chunked dataset is read -- and, compressed, decompressed -- a whole chunk at a time, so the cost is the
    chunks the spans touch, each counted once; a contiguous one costs the spans themselves.
    """
    item = int(dataset.dtype.itemsize)
    if not spans:
        return 0
    if not dataset.chunks:
        return sum(hi - lo for lo, hi in spans) * item
    length = int(dataset.chunks[0])
    chunks, last = 0, -1
    for lo, hi in spans:
        first, end = lo // length, (hi - 1) // length
        first = max(first, last + 1)
        if end >= first:
            chunks += end - first + 1
            last = end
    return chunks * length * item


def _gene_refusal(cost: int, layout: str, ceiling: int, n_rows: int) -> str:
    size, limit = _bytes(cost), _bytes(ceiling)
    if layout == "csr":
        return (
            f"Colouring by this gene reads about {size} of the file, more than the {limit} the explorer reads for "
            f"one gene (SOG_VIZ_GENE_MAX_READ_BYTES). The matrix is stored by rows (CSR), so one gene's values are "
            f"spread across every one of the {n_rows:,} drawn rows and each of them has to be read. Fewer points "
            "read less; raising SOG_VIZ_GENE_MAX_READ_BYTES allows it."
        )
    if layout == "dense":
        return (
            f"Colouring by this gene reads about {size} of the file, more than the {limit} the explorer reads for "
            f"one gene (SOG_VIZ_GENE_MAX_READ_BYTES). The matrix is stored dense, so one gene's column is read "
            "for every row whatever the number of points; raising SOG_VIZ_GENE_MAX_READ_BYTES allows it."
        )
    return (
        f"Colouring by this gene reads about {size}, more than the {limit} the explorer reads for one gene "
        "(SOG_VIZ_GENE_MAX_READ_BYTES); raising it allows the read."
    )


def _bytes(n: float) -> str:
    for unit in ("bytes", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:,.0f} {unit}" if unit == "bytes" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} GiB"  # pragma: no cover - the loop returns


def _read_small(item: Any, depth: int) -> Any:
    if isinstance(item, h5py.Group):
        if depth >= _UNS_MAX_DEPTH:
            return None
        out: dict[str, Any] = {}
        for i, name in enumerate(item):
            if i >= _UNS_MAX_ITEMS:
                break
            value = _read_small(item[name], depth + 1)
            if value is not None:
                out[str(name)] = value
        return out
    if not isinstance(item, h5py.Dataset):
        return None
    try:
        if item.shape == ():
            value = item[()]
            if isinstance(value, bytes):
                return value.decode("utf-8", "replace")
            if isinstance(value, np.generic):
                return value.item()
            return value if isinstance(value, (str, int, float, bool)) else None
        if item.ndim != 1 or item.shape[0] > _UNS_MAX_ARRAY:
            return None
        if item.dtype.kind in "biuf":
            return _native(np.asarray(item[()])).tolist()
        return [str(v) for v in _strings(item)]
    except MemoryError:
        raise
    except Exception:
        return None


__all__ = [
    "FREE_TEXT_MAX_ROWS",
    "GATHER_BLOCK",
    "H5AD",
    "Labels",
    "MAX_LEVELS",
    "ObsColumn",
    "ObsmKey",
    "ObspKey",
    "ReadError",
    "TooLarge",
    "TooManyLevels",
    "Unreadable",
    "Unsupported",
    "open_fd",
]
