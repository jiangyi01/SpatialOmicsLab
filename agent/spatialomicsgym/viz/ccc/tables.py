"""A communication tool's CSV tables, read through an open descriptor, in blocks, under the explorer's table caps.

The same rules as the explorer's result tables (``sog_portal.vizchild._read_table``): never a path -- the descriptor is
wrapped, read from its start and left open for its owner --; refused before a byte is parsed when the file is over
``table_max_bytes`` (``SOG_VIZ_TABLE_MAX_BYTES``), and as soon as a block takes it past ``max_rows``
(``SOG_VIZ_MAX_ROWS``); parsed :data:`BLOCK_ROWS` rows at a time, so a 60 MB MISTy importance table (500,000 rows) can
be reduced block by block without ever holding more than a block of it.

Then the three shapes a communication table comes in (``detect.table_columns``): a **square** index-by-columns matrix,
a **long** table of one row per sender/receiver pair, and a **vector** of one value per label.
"""

from __future__ import annotations

import io
import os
import stat
from typing import TYPE_CHECKING, Any

import numpy as np

from .errors import CCCRefusal

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import pandas as pd

#: Rows per parsed block -- ``vizchild._TABLE_BLOCK_ROWS``.
BLOCK_ROWS = 262_144


def _human(n: float) -> str:
    for unit in ("bytes", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:,.0f} {unit}" if unit == "bytes" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TiB"  # pragma: no cover - the loop returns


def check(fd: int, limits: Mapping[str, int]) -> int:
    """The table's size in bytes, or a refusal: not a regular file, or over ``limits['table_max_bytes']``."""
    try:
        info = os.fstat(int(fd))
    except OSError:
        raise CCCRefusal("bad_request", "A table's descriptor was not handed to the reader.") from None
    if not stat.S_ISREG(info.st_mode):
        raise CCCRefusal("unreadable", "A table is not a regular file.")
    cap = int(limits["table_max_bytes"])
    if info.st_size > cap:
        raise CCCRefusal(
            "too_large",
            f"A table of this result is {_human(info.st_size)}; the explorer reads tables of at most {_human(cap)} "
            "(SOG_VIZ_TABLE_MAX_BYTES). Raising SOG_VIZ_TABLE_MAX_BYTES allows it.",
            "SOG_VIZ_TABLE_MAX_BYTES",
        )
    return int(info.st_size)


def _parse(fd: int, sep: str, **kwargs: Any) -> Any:
    import pandas as pd

    handle = io.BufferedReader(io.FileIO(int(fd), "r", closefd=False), buffer_size=1 << 20)
    handle.seek(0)
    try:
        return pd.read_csv(handle, sep=sep, encoding="utf-8", encoding_errors="replace", **kwargs), handle
    except BaseException:
        handle.close()
        raise


def _unreadable(exc: BaseException, sep: str) -> CCCRefusal:
    said = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ""
    what = "comma" if sep == "," else "tab"
    return CCCRefusal(
        "unreadable", f"A table could not be read as {what}-separated text" + (f": {said}" if said else ".")
    )


def header(fd: int, sep: str, limits: Mapping[str, int]) -> list[str]:
    """The table's column names (``[]`` for an empty file). Reads the first line only."""
    import pandas as pd

    if check(fd, limits) == 0:
        return []
    try:
        frame, handle = _parse(fd, sep, nrows=0)
        handle.close()
    except pd.errors.EmptyDataError:
        return []
    except (ValueError, UnicodeError) as exc:
        raise _unreadable(exc, sep) from None
    return [str(c) for c in frame.columns]


def blocks(
    fd: int, sep: str, limits: Mapping[str, int], *, usecols: list[str] | None = None, dtype: Any = None
) -> Iterator[pd.DataFrame]:
    """The table in blocks of :data:`BLOCK_ROWS` rows, refused as soon as the rows pass ``limits['max_rows']``.

    An empty file yields nothing. ``usecols`` reads only those columns -- a block of four of five columns is a fifth
    smaller, and a column that holds a path is never parsed at all.
    """
    import pandas as pd

    if check(fd, limits) == 0:
        return
    max_rows = int(limits["max_rows"])
    total = 0
    try:
        reader, handle = _parse(fd, sep, chunksize=BLOCK_ROWS, usecols=usecols, dtype=dtype)
    except pd.errors.EmptyDataError:
        return
    except (ValueError, UnicodeError) as exc:
        raise _unreadable(exc, sep) from None
    try:
        with reader:
            while True:
                try:
                    block = next(reader)
                except StopIteration:
                    return
                except (ValueError, UnicodeError) as exc:  # pandas' ParserError is a ValueError
                    raise _unreadable(exc, sep) from None
                total += len(block)
                if total > max_rows:
                    raise CCCRefusal(
                        "too_large",
                        f"A table of this result has more than {max_rows:,} rows, the most the explorer reads "
                        "(SOG_VIZ_MAX_ROWS).",
                        "SOG_VIZ_MAX_ROWS",
                    )
                yield block
    finally:
        handle.close()


def read(
    fd: int, sep: str, limits: Mapping[str, int], *, usecols: list[str] | None = None, dtype: Any = None
) -> pd.DataFrame:
    """The whole table (every block joined); an empty frame for an empty file."""
    import pandas as pd

    parts = list(blocks(fd, sep, limits, usecols=usecols, dtype=dtype))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


def numbers(values: Any) -> np.ndarray:
    """A column as float64, anything that is not a number NaN."""
    import pandas as pd

    return np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=np.float64)


def labels(values: Any) -> list[str]:
    """A column as text labels (a missing one is ``""``)."""
    return ["" if v is None or (isinstance(v, float) and v != v) else str(v) for v in list(values)]


def square(frame: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray]:
    """An index-by-columns matrix: ``(row labels, column labels, values)``, the first column being the row labels."""
    if frame.shape[1] < 2 or frame.shape[0] < 1:
        raise CCCRefusal("unsupported", "A table that should be a matrix has no rows or no value columns.")
    rows = labels(frame.iloc[:, 0])
    cols = [str(c) for c in frame.columns[1:]]
    values = np.column_stack([numbers(frame.iloc[:, j]) for j in range(1, frame.shape[1])])
    return rows, cols, values


def long_matrix(
    frame: pd.DataFrame, source: str, target: str, value: str, *, mirror: bool, order: list[str] | None = None
) -> tuple[list[str], np.ndarray]:
    """A long table's ``value`` column as a labels-by-labels matrix, sender rows and receiver columns.

    Labels in order of first appearance (sources and targets interleaved, row by row) unless ``order`` is given. A pair
    the table does not list is NaN. ``mirror`` fills a missing reverse pair from the one listed -- for tables of
    unordered pairs (neighbourhood enrichment, colocalisation), which list each pair once.
    """
    sources, targets = labels(frame[source]), labels(frame[target])
    if order is None:
        order = []
        seen: set[str] = set()
        for a, b in zip(sources, targets, strict=True):
            for name in (a, b):
                if name not in seen:
                    seen.add(name)
                    order.append(name)
    where = {name: i for i, name in enumerate(order)}
    out = np.full((len(order), len(order)), np.nan)
    values = numbers(frame[value])
    for a, b, v in zip(sources, targets, values.tolist(), strict=True):
        if a in where and b in where:
            out[where[a], where[b]] = v
    if mirror:
        missing = np.isnan(out) & ~np.isnan(out.T)
        out[missing] = out.T[missing]
    return order, out


__all__ = ["BLOCK_ROWS", "blocks", "check", "header", "labels", "long_matrix", "numbers", "read", "square"]
