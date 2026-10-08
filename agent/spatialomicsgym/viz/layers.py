"""The only module that reads values out of a dataset, and the only one that decides which slot.

Keeping the read in one place is what makes the honesty rules enforceable. A family that reached
into ``adata.X`` itself could draw a z-scored matrix and caption it as expression, and no check
above it would notice. Everything here returns the values **and** a record of where they came
from, and the renderer refuses to save a figure whose record is missing.

The slot is chosen by rule, never silently:

* a counts-shaped question reads counts -- ``layers['counts']`` first, then ``.raw``, then ``X``;
* a magnitude plot refuses a matrix with negative values unless a counts layer is reachable,
  because negative "expression" is not expression and a sequential colour map over it makes the
  most negative value look like the lowest expression;
* whichever slot is used is written into the figure's record and printed in its caption.

A gene symbol that appears twice is an ambiguity, not a first match. A real Visium object in this
repository has ten duplicated symbols, and taking the first row is a silent, unrepeatable choice.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class LayerError(Exception):
    """A value could not be read, with a sentence saying what to do about it."""

    def __init__(self, message: str, *, field: str = "", fix: str = "") -> None:
        super().__init__(message)
        self.field = field
        self.fix = fix


#: Above this in-memory size, reading the whole object into memory is refused. A figure is not
#: worth an out-of-memory kill. ``SOG_VIZ_MAX_IN_MEMORY_GB`` moves it, for a box that has the memory.
MAX_IN_MEMORY_GB = 12.0


def _max_in_memory_gb() -> float:
    import os

    raw = str(os.environ.get("SOG_VIZ_MAX_IN_MEMORY_GB", "")).strip()
    try:
        value = float(raw) if raw else MAX_IN_MEMORY_GB
    except ValueError:
        return MAX_IN_MEMORY_GB
    return value if value > 0 else MAX_IN_MEMORY_GB


def stored_gb(data_path: str) -> float | None:
    """What reading this file into memory costs, from the sizes of what it stores. ``None`` if unknown.

    ``n_obs * n_vars * 4`` is the footprint of a DENSE matrix, and ``read_h5ad`` keeps a sparse one
    sparse: a 600k-bin Visium HD object is ~40 GB by that sum and a few GB on read, so every family
    refused it, including the ones that never touch X (hunt 2026-09-30, u20b-viz-rest-22). Summing
    the uncompressed size of every stored dataset is what the read actually materialises, layers
    included, and it costs one metadata walk of the file.
    """
    try:
        import h5py
    except Exception:
        return None
    total = 0

    def _visit(_name: str, obj: Any) -> None:
        nonlocal total
        if isinstance(obj, h5py.Dataset):
            try:
                total += int(obj.size) * int(obj.dtype.itemsize)
            except Exception:
                pass

    try:
        with h5py.File(data_path, "r") as handle:
            handle.visititems(_visit)
    except Exception:
        return None
    return total / 1024**3


@contextmanager
def open_source(data_path: str, *, estimated_dense_gb: float = 0.0) -> Iterator[Any]:
    """Open a dataset **for reading values**, which is a different job from profiling it.

    Profiling reads metadata and a bounded sample, so it opens backed. Reading values cannot:
    a backed sparse matrix rejects column indexing outright -- ``adata[:, i].X`` raises
    ``AttributeError: 'backed_csr_matrix' object has no attribute '_validate_indices'``, and
    ``to_memory()`` on a variable slice comes back with no matrix at all. Both were driven on a
    real Visium object here before this was written.

    So this opens in memory, and guards the one case where that is the wrong trade: an object
    whose in-memory footprint would not fit. The full read of a 4,035 by 36,601 Visium object takes
    0.13 s, which is the common case and is not worth being clever about. ``estimated_dense_gb`` is
    used only when the stored size cannot be read.
    """
    import anndata as ad

    ceiling = _max_in_memory_gb()
    needed = stored_gb(data_path)
    if needed is None:
        needed = float(estimated_dense_gb or 0.0)
    if needed > ceiling:
        # The fix used to say "name a layer or a subset", which no producer can act on: none has a
        # subset parameter and the whole file is read whatever layer is named (u20b-viz-rest-22).
        raise LayerError(
            f"reading this object needs about {needed:.1f} GB of memory, above the {ceiling:.0f} GB "
            "ceiling for drawing a figure (SOG_VIZ_MAX_IN_MEMORY_GB)",
            field="X",
            fix=(
                "raise SOG_VIZ_MAX_IN_MEMORY_GB on a machine with that much free memory, or draw "
                "from a result table a tool already wrote rather than from the full object"
            ),
        )
    adata = ad.read_h5ad(data_path)
    try:
        yield adata
    finally:
        del adata


def resolve_expression_slot(
    adata: Any,
    *,
    layer: str = "",
    use_raw: bool = False,
    need_counts: bool = False,
    x_has_negative: bool = False,
) -> dict[str, Any]:
    """Which slot to read, and the record of why. Never silent, never guessed twice."""

    available = list(getattr(adata, "layers", {}) or {})
    if layer:
        if layer not in available:
            raise LayerError(
                f"this object has no layer named {layer!r}; it has {available or 'no layers at all'}",
                field=f"layers['{layer}']",
                fix="name one of the layers above, or leave the layer empty to read X",
            )
        return {"slot": "layers", "key": layer, "why": "named by the caller"}
    if use_raw:
        if adata.raw is None:
            raise LayerError(
                "this object has no .raw slot",
                field="raw",
                fix="leave use_raw off, or name a layer",
            )
        return {"slot": "raw.X", "key": "", "why": "named by the caller"}
    if need_counts and "counts" in available:
        return {"slot": "layers", "key": "counts", "why": "a counts-shaped question, and a counts layer exists"}
    if need_counts and x_has_negative and getattr(adata, "raw", None) is not None:
        # X has been scaled, so its values are standard deviations. A sequential colour map over
        # them makes the most negative value read as the lowest expression, which is the opposite
        # of what it means. `.raw` is the scanpy convention for what was kept before scaling.
        return {"slot": "raw.X", "key": "", "why": "X is scaled and .raw holds the pre-scaling values"}
    if need_counts and x_has_negative:
        # The rule in this module's docstring, which nothing implemented: a z-scored X fell through
        # to here and was painted with viridis, captioned "as stored (already transformed)" -- a
        # map of standard deviations presented as expression (hunt 2026-09-30, u20b-viz-rest-23).
        raise LayerError(
            "X holds negative values, so it has been scaled: its values are standard deviations, not "
            "expression, and this object has no layers['counts'] and no .raw to read expression from",
            field="X",
            fix=(
                "name a layer that holds counts or normalised expression with layer=, or keep the "
                "pre-scaling matrix in .raw or layers['counts'] before scaling"
            ),
        )
    return {"slot": "X", "key": "", "why": "the default slot"}


def _matrix(adata: Any, slot: dict[str, Any]) -> Any:
    if slot["slot"] == "layers":
        return adata.layers[slot["key"]]
    if slot["slot"] == "raw.X":
        return adata.raw.X
    return adata.X


def gene_values(
    adata: Any,
    symbols: Sequence[str],
    slot: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """``(values per symbol, not found, ambiguous)``.

    The two failure lists are kept apart because they need different answers: a symbol that is
    not there may be spelled differently or may be an Ensembl identifier, while a symbol that is
    there twice needs the caller to say which row they mean.
    """
    import numpy as np
    import scipy.sparse as sp

    # raw.X has its OWN columns: the standard scanpy object sets .raw before subsetting to the
    # highly variable genes, so adata.var_names is a different, shorter list in a different order.
    # Positions taken from it and used on raw.X read another gene's column, captioned "raw.X"
    # (hunt 2026-09-30, u20b-viz-rest-1).
    index = adata.raw.var_names if slot.get("slot") == "raw.X" else adata.var_names
    names = [str(v) for v in index]
    positions: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        positions.setdefault(name.upper(), []).append(i)

    found: dict[str, Any] = {}
    missing: list[str] = []
    ambiguous: list[str] = []
    matrix = _matrix(adata, slot)
    for symbol in symbols:
        hits = positions.get(str(symbol).upper(), [])
        if not hits:
            missing.append(str(symbol))
            continue
        if len(hits) > 1:
            ambiguous.append(str(symbol))
            continue
        column = matrix[:, hits[0]]
        if sp.issparse(column):
            column = np.asarray(column.todense()).ravel()
        else:
            column = np.asarray(column).ravel()
        found[str(symbol)] = column
    return found, missing, ambiguous


def obs_values(adata: Any, key: str, *, max_levels: int = 200) -> tuple[Any, bool]:
    """``(values, is_categorical)`` for one obs column."""
    import pandas as pd

    if key not in adata.obs.columns:
        raise LayerError(
            f"this object has no obs column named {key!r}",
            field=f"obs['{key}']",
            fix="name one of the columns the dataset inspection listed",
        )
    series = adata.obs[key]
    categorical = bool(
        isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object or str(series.dtype) == "bool"
    )
    if not categorical and pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        # An integer column with few levels is a cluster id. Drawing it on a continuous colour
        # bar implies an ordering the labels do not have.
        if values.dropna().nunique() <= 64 and (values.dropna() % 1 == 0).all():
            return series.astype(str).to_numpy(), True
        return values.to_numpy(dtype=float), False
    return series.astype(str).to_numpy(), True


def obsm_values(adata: Any, reference: str) -> tuple[Any, str]:
    """One column of an obsm matrix, addressed as ``matrix:column``. ``matrix:*`` is every column."""
    import numpy as np

    key, _, column = str(reference).partition(":")
    if key not in adata.obsm:
        raise LayerError(
            f"this object has no obsm matrix named {key!r}; it has {sorted(adata.obsm)}",
            field=f"obsm['{key}']",
            fix="name one of the matrices above",
        )
    matrix = adata.obsm[key]
    labels = None
    try:
        labels = [str(c) for c in matrix.columns]
    except Exception:
        labels = None
    array = np.asarray(matrix)
    if not column or column == "*":
        return array, key
    if labels and column in labels:
        return array[:, labels.index(column)], column
    try:
        return array[:, int(column)], f"{key}[{column}]"
    except (ValueError, IndexError) as exc:
        raise LayerError(
            f"{key!r} has no column named {column!r}"
            + (f"; it has {labels}" if labels else f" and {array.shape[1]} unnamed columns"),
            field=f"obsm['{key}']:{column}",
            fix="name one of the columns above, or use '*' for all of them",
        ) from exc


#: obs columns that name a section, in the order they are tried. ``library_id`` is first because
#: it is the Space Ranger word; ``brain_section_label`` and ``Bregma`` are here because the two
#: atlases this toolkit is driven against use them and nothing else would find them.
SECTION_COLUMNS = (
    "library_id",
    "slice_id",
    "section",
    "section_id",
    "brain_section_label",
    "Bregma",
    "z_index",
    "batch",
)

#: A section column with more levels than this is not a stack anyone will plot as panels; it is
#: more likely a continuous value that happens to be stored as a category.
MAX_SECTION_LEVELS = 200


# --------------------------------------------------------------------------- #
# Per-distinct-value reads
#
# The section and z readers below turn an obs column into strings (``astype(str)``) and, for a z, the
# strings into floats. Done row by row that is 8.7 s for a z column and 10 s for a section column at
# 10M rows (measured 2026-10-01), and the explorer reads both on every 3D dataset it opens. A section
# column holds a handful of distinct values, so the conversion runs once per distinct value here and
# is gathered back by an integer code. The conversion that runs is still the caller's own, on rows
# of the same column with the same dtype -- so the result is the row-by-row result, not a re-derivation
# of it. test/test_viz_layers_vectorized_reads_match_the_row_by_row_reads.py holds the row-by-row
# implementation as its oracle.
# --------------------------------------------------------------------------- #
def _identity_codes(series: Any) -> tuple[Any, int] | None:
    """``(codes, n_codes)``: rows with one code print the same under ``astype(str)``. ``None``: no cheap code.

    The identity is the stored BITS of a number -- so ``-0.0`` and ``0.0``, equal as numbers and
    printed differently, stay apart -- the category code of a categorical, and the value of a
    string. A code of ``-1`` marks a row that is converted on its own: a missing value in a string
    column, because ``None``, ``nan`` and ``pd.NA`` are all missing and print three ways. A mixed
    object column has no such identity (``1``, ``1.0`` and ``True`` are equal and print three ways),
    and only an object built in memory has one; it takes the row-by-row path.
    """
    import numpy as np
    import pandas as pd

    if not isinstance(series, pd.Series):
        return None
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        # pandas converts the categories and takes them by code, with ONE fill value for a missing
        # row -- so missing is one more code, not a row-by-row case.
        return series.cat.codes.to_numpy().astype(np.intp) + 1, len(dtype.categories) + 1
    if isinstance(dtype, np.dtype):
        if dtype.kind in "biuf" and dtype.itemsize in (1, 2, 4, 8):
            codes, uniques = pd.factorize(series.to_numpy().view(f"u{dtype.itemsize}"))
            return codes, len(uniques)
        if dtype.kind == "O":
            values = series.to_numpy()
            if pd.api.types.infer_dtype(values, skipna=True) not in ("string", "empty"):
                return None
            codes, uniques = pd.factorize(values)
            return codes, len(uniques)
        return None
    if isinstance(dtype, pd.StringDtype):
        codes, uniques = pd.factorize(series.array)
        return np.asarray(codes), len(uniques)
    array = series.array
    if isinstance(array, (pd.arrays.IntegerArray, pd.arrays.FloatingArray, pd.arrays.BooleanArray)):
        # A nullable column: the value's bits where it is present, and one more code for missing,
        # which every missing row prints the same way.
        data = np.ascontiguousarray(array.to_numpy(dtype=dtype.numpy_dtype, na_value=0))
        if data.dtype.itemsize not in (1, 2, 4, 8):
            return None
        codes, uniques = pd.factorize(data.view(f"u{data.dtype.itemsize}"))
        missing = np.asarray(array.isna(), dtype=bool)
        n_codes = len(uniques)
        if missing.any():
            codes[missing] = n_codes
            n_codes += 1
        return codes, n_codes
    return None


def _per_distinct(series: Any, convert: Any) -> tuple[Any, Any]:
    """``(convert(series), every distinct result)``, with ``convert`` run once per distinct value.

    ``convert`` must act element by element, as ``astype(str)`` and a float parse of it do; then one
    row of each code stands for every row of it. Rows coded ``-1`` are converted on their own. Where
    no code is available, or the pieces would not reassemble into the dtype the whole column gives,
    this is ``convert(series)`` itself -- slower, never different.
    """
    import numpy as np

    dtype = getattr(series, "dtype", None)
    if isinstance(dtype, np.dtype) and not dtype.isnative:
        # A big-endian column -- anndata keeps a file's byte order on read. pandas cannot gather rows
        # from a non-native buffer (``take`` raises "Big-endian buffer not supported"), and the
        # gather below is a ``take``; the row-by-row read never gathered, so it never failed. The
        # same values in this machine's byte order print and parse the same, so they are converted
        # instead: one copy of the column, not a string per row.
        series = series.astype(dtype.newbyteorder("="))
    found = _identity_codes(series)
    if found is None or len(series) == 0 or found[1] == 0:
        # No code, no rows, or no row with a code (an all-missing string column): the whole read.
        whole = np.asarray(convert(series))
        return whole, whole
    codes, n_codes = found
    rows = np.arange(len(series), dtype=np.intp)
    known = codes >= 0
    every = bool(known.all())
    # A code no row carries (an unused category) keeps row 0, which exists and is converted anyway.
    representative = np.zeros(n_codes, dtype=np.intp)
    if every:
        representative[codes] = rows
    else:
        representative[codes[known]] = rows[known]
    distinct = np.asarray(convert(series.iloc[representative]))
    if every:
        return distinct[codes], distinct
    lone = rows[~known]
    lone_values = np.asarray(convert(series.iloc[lone]))
    if lone_values.dtype != distinct.dtype:
        whole = np.asarray(convert(series))
        return whole, whole
    out = distinct[np.where(known, codes, 0)]
    out[lone] = lone_values
    return out, np.concatenate([distinct, lone_values])


def _obs_as_str(series: Any) -> tuple[Any, int]:
    """``(series.astype(str).to_numpy(), how many distinct strings that holds)``, per distinct value."""
    import numpy as np

    values, distinct = _per_distinct(series, lambda part: part.astype(str).to_numpy())
    return values, len(np.unique(distinct))


def _obs_as_float(series: Any, *, exact: bool = True) -> Any:
    """``np.asarray(series.astype(str), dtype=float)`` -- the z read -- without a string per row.

    A float64 or integer column needs no strings at all: ``str`` of a float64 is its shortest
    round-trip repr and of an integer its digits, so parsing either back is the plain cast (NaN comes
    back as the one NaN ``float('nan')`` is). float32 is NOT that: pandas prints it at float32
    precision, so ``0.1f`` reads back as ``0.1``, not ``0.10000000149``; it and everything else go
    through the per-distinct path. Raises what the row-by-row read raises, for the same columns.

    ``exact=False`` is for a decision about the values, never for a z that is drawn: a float32 or
    float16 column is cast too. That keeps which values are finite and which are equal -- the
    shortest repr is one-to-one with the value, and distinct decimals of at most nine digits stay
    distinct as float64 -- but not the bits the strings parse back to.
    """
    import numpy as np
    import pandas as pd

    if isinstance(series, pd.Series) and isinstance(series.dtype, np.dtype):
        kind = series.dtype.kind
        if kind in "iu" or (kind == "f" and (series.dtype.itemsize == 8 or not exact)):
            values = series.to_numpy().astype(np.float64)
            if kind == "f":
                values[np.isnan(values)] = np.nan
            return values
    values, _distinct = _per_distinct(series, lambda part: np.asarray(part.astype(str), dtype=float))
    return values


def section_labels(adata: Any, section_key: str = "") -> tuple[Any, str]:
    """``(labels, where they came from)``, or ``(None, "")`` when the object holds one section.

    Read from obs, never from ``uns['spatial']``. ``viz.profile`` derives ``library_ids`` from the
    keys of ``uns['spatial']``, which a merged serial-section object does not have at all -- so on
    every dataset this toolkit's 3D work is about, that list is empty and anything keyed off it
    silently does nothing. The section axis on real data is an obs column.
    """
    candidates = [section_key] if section_key else list(SECTION_COLUMNS)
    block = adata.uns.get("spatial_3d") if hasattr(adata, "uns") else None
    if isinstance(block, dict) and block.get("slice_key") and not section_key:
        candidates.insert(0, str(block["slice_key"]))
    for name in candidates:
        if not name or name not in adata.obs.columns:
            continue
        # ``adata.obs[name].astype(str).to_numpy()`` and its distinct count, once per distinct value.
        values, levels = _obs_as_str(adata.obs[name])
        if 2 <= levels <= MAX_SECTION_LEVELS:
            return values, f"obs[{name!r}]"
        if section_key:
            raise LayerError(
                f"obs[{name!r}] has {levels} distinct values, which is not a section axis",
                field=f"obs['{name}']",
                fix=f"name a column with between 2 and {MAX_SECTION_LEVELS} levels",
            )
    if section_key:
        # `"one of " + joined or "add one"` binds as `("one of " + joined) or ...`, which is always
        # truthy, so an object with no section column was told "one of " (hunt 2026-09-30,
        # u20b-viz-rest-32).
        present = ", ".join(repr(c) for c in SECTION_COLUMNS if c in adata.obs.columns)
        raise LayerError(
            f"there is no obs column named {section_key!r}",
            field=f"obs['{section_key}']",
            fix=f"one of {present}" if present else "add one",
        )
    return None, ""


def _spatial_libraries(adata: Any) -> list[str]:
    try:
        spatial = adata.uns.get("spatial")
    except Exception:
        return []
    return [str(k) for k in spatial] if isinstance(spatial, dict) else []


def library_mask(adata: Any, library_id: str) -> tuple[Any, str]:
    """Which observations ``library_id`` selects: ``(mask, where it was read)``, or ``(None, note)``
    when it selects every observation.

    The tools describe ``library_id`` as "which uns['spatial'] library the coordinates and image
    come from", and inspect_dataset reports those keys as ``library_ids``; the code resolved it
    only against an obs section column, so naming the one library of a plain Visium object was
    refused with "no section axis to select from" (hunt 2026-09-30, u20b-viz-rest-21). Both
    vocabularies resolve here: a section value in obs first, then a key of ``uns['spatial']``.
    """
    labels, source = section_labels(adata, library_id if library_id in adata.obs.columns else "")
    libraries = _spatial_libraries(adata)
    if labels is not None:
        mask = labels == str(library_id)
        if mask.any():
            return mask, source
        present = sorted(set(labels))[:12]
        raise LayerError(
            f"no section named {library_id!r}; {source} holds {present}"
            + (f", and uns['spatial'] holds {libraries[:12]}" if libraries else ""),
            field="library_id",
            fix="name one of the sections above",
        )
    if str(library_id) in libraries:
        if len(libraries) == 1:
            return None, f"uns['spatial'] holds the one library {library_id!r}, so every observation is in it"
        raise LayerError(
            f"uns['spatial'] holds {len(libraries)} libraries, but no obs column says which observations "
            f"belong to {library_id!r}",
            field="library_id",
            fix="add an obs column naming each observation's library (library_id), or draw every library",
        )
    raise LayerError(
        f"library_id={library_id!r} was given but this object has no section axis to select from"
        + (f"; uns['spatial'] holds {libraries[:12]}" if libraries else ""),
        field="library_id",
        fix="drop library_id, name one of the libraries above, or add an obs column naming each section",
    )


def select_library(adata: Any, library_id: str = "") -> tuple[Any, str]:
    """``(the observations library_id selects, what was selected)``. Every observation when empty.

    The VALUES a figure paints must be read from the same observations as its coordinates.
    ``spatial_coords(adata, library_id)`` restricts only the coordinates, so a caller that read its
    values from the whole object handed matplotlib 160 colours for 80 points (hunt 2026-09-30,
    u20b-viz-rest-3). Subsetting the object once, here, before anything is read, keeps every read
    aligned by construction.
    """
    if not library_id:
        return adata, ""
    mask, source = library_mask(adata, library_id)
    if mask is None:
        return adata, source
    return adata[mask], f"{source} == {library_id!r}"


def spatial_coords(adata: Any, library_id: str = "") -> tuple[Any, str]:
    """``(coordinates, where they came from)``. Falls back to an obs pair, and says so.

    ``library_id``, when given, selects ONE section. It was accepted and silently ignored by six
    plotting functions until 2026-09-22, so on a merged multi-section object every one of them
    drew all the sections overlaid in a single frame with nothing said about it. A caller that
    paints values must read them from :func:`select_library` too, or the two lengths differ.
    """
    import numpy as np

    if library_id:
        mask, source = library_mask(adata, library_id)
        if mask is None:
            coords, where = spatial_coords(adata)
            return coords, f"{where} ({source})"
        coords, where = spatial_coords(adata[mask])
        return coords, f"{where} restricted to {source} == {library_id!r}"

    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"], dtype=float), "obsm['spatial']"
    from spatialomicsgym.utils.obs_aliases import _find_spatial_coord_columns

    pair = _find_spatial_coord_columns(list(adata.obs.columns))
    if not pair:
        raise LayerError(
            "this object has no spatial coordinates: there is no obsm['spatial'] and no x/y pair in obs",
            field="obsm['spatial']",
            fix="convert the raw platform output with the data converter, which writes obsm['spatial']",
        )
    coords = np.column_stack([adata.obs[pair[0]].to_numpy(dtype=float), adata.obs[pair[1]].to_numpy(dtype=float)])
    return coords, f"obs['{pair[0]}'] and obs['{pair[1]}']"


#: obsm key prefixes that name coordinates. ``X_`` is scanpy's embedding prefix and is never one.
COORDINATE_KEY_PREFIXES = ("spatial", "moscot_spatial")


def is_coordinate_key(key: str, adata: Any = None) -> bool:
    """Does this obsm key hold coordinates? By its name, or by the 3D contract's own declaration.

    Shape is not the test: a three-column ``X_umap`` and a three-type ``proportions`` matrix have
    the shape of a 3D frame and are not one. ``viz.profile`` uses this same rule, so the catalogue
    and the reader cannot disagree about which keys are coordinates.
    """
    name = str(key)
    if name.lower().startswith("x_"):
        return False
    if name.startswith(COORDINATE_KEY_PREFIXES):
        return True
    try:
        block = adata.uns.get("spatial_3d") if adata is not None else None
    except Exception:
        block = None
    frames = block.get("frames") if isinstance(block, dict) else None
    return isinstance(frames, dict) and name in {str(k) for k in frames}


def spatial_coords_3d(
    adata: Any,
    *,
    coords_key: str = "",
    section_key: str = "",
    z_spacing: float = 0.0,
) -> tuple[Any, str, Any, str]:
    """``(xyz, where the coordinates came from, section labels, where those came from)``.

    Every 2D plot in this toolkit reads ``obsm['spatial']`` and nothing else, so writing aligned
    coordinates to a new key -- which is exactly what the 3D contract requires -- made them
    invisible to every figure. This is the seam that fixes that.

    The resolution order, and each step is recorded in the returned source string so a caption can
    say which one fired:

    1. ``coords_key`` when named, refused by name when absent;
    2. ``obsm['spatial_3d_aligned']``, then ``obsm['spatial_3d_raw']``, then ``obsm['spatial_3d']``
       -- the contract's own keys, aligned first because that is what a reader means by "the 3D
       coordinates" once an alignment has been run;
    3. ``obsm['spatial']`` itself when it has three columns (the STARmap and ST-GEARS shape);
    4. any other three-column key that names coordinates (:func:`is_coordinate_key`) -- never an
       embedding or a proportions matrix that happens to have three columns;
    5. two-column coordinates plus a z built from the section axis.

    **Step 5 never invents a spacing.** It reads a z column (:func:`z_column`: the contract's
    recorded ``z_key``, then ``slice_z``, ``z``, ``Bregma``), then ``uns['spatial_3d']['z_spacing']``,
    then ``z_spacing`` if the caller passed one, and otherwise refuses. Numeric section labels in any
    other column are never a z: a concat's ``batch`` "0"/"1" or a DLPFC ``library_id`` "151507" is an
    id, and reading it as a position drew two 2D samples as a stack at z = 0, 1. A stack drawn at
    z = 0, 1, 2 when the real spacing is 0.0985 mm and irregular is a picture of something that does
    not exist, and it renders perfectly.
    """
    import numpy as np

    obsm = getattr(adata, "obsm", {}) or {}
    labels, section_from = section_labels(adata, section_key)

    def _three(key: str) -> Any:
        arr = np.asarray(obsm[key], dtype=float)
        return arr if arr.ndim == 2 and arr.shape[1] >= 3 else None

    if coords_key:
        if coords_key not in obsm:
            raise LayerError(
                f"there is no obsm[{coords_key!r}]",
                field=f"obsm['{coords_key}']",
                fix="one of " + ", ".join(repr(k) for k in sorted(obsm)) if obsm else "add coordinates",
            )
        arr = _three(coords_key)
        if arr is None:
            raise LayerError(
                f"obsm[{coords_key!r}] has fewer than three columns, so it carries no z",
                field=f"obsm['{coords_key}']",
                fix="name a three-column key, or let the z come from the section axis",
            )
        return arr[:, :3], f"obsm[{coords_key!r}]", labels, section_from

    for key in ("spatial_3d_aligned", "spatial_3d_raw", "spatial_3d"):
        if key in obsm and (arr := _three(key)) is not None:
            return arr[:, :3], f"obsm[{key!r}]", labels, section_from

    if "spatial" in obsm and (arr := _three("spatial")) is not None:
        return arr[:, :3], "obsm['spatial'] (three columns)", labels, section_from

    # Only a key that names coordinates. Any three-column obsm was taken here, so a 3-component
    # UMAP or a three-type proportions matrix was drawn as "the stack in three dimensions", and was
    # preferred over a three-column obsm['spatial'] (hunt 2026-09-30, u20b-viz-rest-6).
    for key in sorted(obsm):
        if key == "spatial" or not is_coordinate_key(key, adata):
            continue
        if (arr := _three(key)) is not None:
            return arr[:, :3], f"obsm[{key!r}]", labels, section_from

    # Step 5: build a z from the sections, or refuse saying what would supply one.
    xy, xy_from = spatial_coords(adata)
    if labels is None:
        raise LayerError(
            "this object has no three-column coordinate key and no section axis, so there is no z",
            field="obsm",
            fix=(
                "run an aligner and adopt its output into obsm['spatial_3d_aligned'], or add an obs "
                "column naming each section"
            ),
        )
    z, z_from = _section_z(adata, labels, z_spacing, section_from)
    return (
        np.column_stack([np.asarray(xy, float)[:, :2], z]),
        f"{xy_from} plus a z from {z_from}",
        labels,
        section_from,
    )


#: obs columns whose values ARE a position along the stack, in the order the 3D contract reads them
#: (``spatial3d.pipelines`` and ``spatial3d.validate``: a recorded ``uns['spatial_3d']['z_key']``,
#: then these). Only these, a recorded spacing or a caller's spacing make a z.
Z_COLUMNS = ("slice_z", "z", "Bregma")


def z_column(adata: Any) -> str:
    """The obs column holding each observation's z, or ``""``. Numeric, by the contract's own order.

    One rule for the profile, which decides whether a 3D plot is offered, and for :func:`_section_z`,
    which builds the z the plot draws -- so the catalogue cannot offer a stack the reader then builds
    from something else (hunt 2026-09-30, u20b-viz-rest-7).
    """
    # Only the name is answered, so the values need not be the drawn z's bits: a per-cell float32 z
    # is decided by a cast instead of a string per distinct value (3 s per two million).
    return _z_column_values(adata, exact=False)[0]


def _z_column_values(adata: Any, *, exact: bool = True) -> tuple[str, Any]:
    """``(z_column(adata), that column read as floats)``, or ``("", None)``: the column is read once.

    The column chosen does not depend on ``exact``; with ``exact=False`` the values returned are
    only fit to decide with (:func:`_obs_as_float`).
    """
    import numpy as np

    try:
        columns = list(adata.obs.columns)
        block = adata.uns.get("spatial_3d") if hasattr(adata, "uns") else None
    except Exception:
        return "", None
    recorded = str(block.get("z_key") or "") if isinstance(block, dict) else ""
    for column in dict.fromkeys([recorded, *Z_COLUMNS]):
        if not column or column not in columns:
            continue
        try:
            values = _obs_as_float(adata.obs[column], exact=exact)
        except (TypeError, ValueError):
            continue
        # A constant z (a 2D converter's z = 0) places nothing; it is not a stack's z. "More than one
        # distinct finite value" is "some finite value differs from the first", without a sort.
        finite = values[np.isfinite(values)]
        if finite.size and bool((finite != finite.flat[0]).any()):
            return column, values
    return "", None


def _first_seen_codes(labels: Any) -> Any:
    """Each label's index in ``dict.fromkeys(labels)``, as float64: the section order, one pass.

    For strings -- every label :func:`section_labels` gives, a missing one reading ``'nan'`` --
    ``pd.factorize`` numbers values in order of first appearance and compares them by hash and
    equality, as a dict does. Anything else keeps the dict itself: a dict tells two NaN objects
    apart by identity and ``None`` from NaN, where factorize merges every missing value into one.
    """
    import numpy as np
    import pandas as pd

    values = np.asarray(labels, dtype=object)
    if values.ndim == 1 and pd.api.types.infer_dtype(values, skipna=False) in ("string", "empty"):
        codes, _uniques = pd.factorize(values)
        return codes.astype(np.float64)
    index = {name: i for i, name in enumerate(dict.fromkeys(labels))}
    return np.array([index[name] for name in labels], dtype=np.float64)


def _section_z(adata: Any, labels: Any, z_spacing: float, section_from: str = "") -> tuple[Any, str]:
    """``(a physical z per cell, where it came from)``: a z column, or a spacing. Never invented.

    A recorded spacing is read before the caller's, and neither before a z column, which is a
    measurement rather than an assumption. Numeric section labels are NOT a z unless their column is
    one of :data:`Z_COLUMNS`: they were, so a concat of two 2D samples (``batch`` "0"/"1", or DLPFC
    ``library_id`` "151507"/"151673") was drawn as "the stack in three dimensions" at those z values
    with nothing said (hunt 2026-09-30, u20b-viz-rest-7, the review of its first fix).
    """
    block = adata.uns.get("spatial_3d") if hasattr(adata, "uns") else None
    block = block if isinstance(block, dict) else {}

    column, z = _z_column_values(adata)
    if column:
        unused = f"; the z_spacing passed ({float(z_spacing):g}) was not used" if float(z_spacing or 0.0) > 0 else ""
        return z, f"obs[{column!r}]{unused}"

    # Each label's position in first-seen order -- ``dict.fromkeys(labels)`` -- as one hashed pass
    # rather than a dictionary lookup per row.
    order = _first_seen_codes(labels)

    recorded = block.get("z_spacing")
    if isinstance(recorded, (int, float)) and not isinstance(recorded, bool) and recorded > 0:
        z = order * float(recorded)
        return z, f"{section_from or 'the section order'} at the spacing recorded in uns['spatial_3d'] ({recorded:g})"

    spacing = float(z_spacing or 0.0)
    if spacing > 0:
        z = order * spacing
        return z, f"{section_from or 'the section order'} at the z_spacing passed ({spacing:g})"

    raise LayerError(
        "the sections carry no z: there is no z column (obs['slice_z'], obs['z'], obs['Bregma'] or the "
        "column uns['spatial_3d']['z_key'] names), no spacing is recorded in uns['spatial_3d'] and none "
        "was passed in. Numeric section labels are ids, not positions, unless their column is one of those",
        field="z",
        fix=(
            "pass z_spacing, or record it in uns['spatial_3d']['z_spacing'], or name the obs column that "
            "holds each section's position in uns['spatial_3d']['z_key']. Drawing a stack at z = 0, 1, 2 "
            "when the real spacing is irregular is a picture of something that does not exist, and it "
            "renders perfectly."
        ),
    )


def histology(adata: Any, library_id: str = "", image_key: str = "hires") -> tuple[Any, float, str]:
    """``(image, scalefactor, library)``. The pair is chosen together or not at all.

    Returns ``(None, 1.0, "")`` when there is no image, which is a fact about the dataset rather
    than an error: an imaging-based platform legitimately has coordinates and no image.
    """
    spatial = None
    try:
        spatial = adata.uns.get("spatial")
    except Exception:
        spatial = None
    if not isinstance(spatial, dict) or not spatial:
        return None, 1.0, ""
    libraries = sorted(str(k) for k in spatial)
    if not library_id and len(spatial) > 1:
        # It took next(iter(spatial)), so every spot of every library was drawn over the FIRST
        # library's image, status ok, nothing in the caption (hunt 2026-09-30, u20b-viz-rest-21).
        raise LayerError(
            f"this object holds {len(spatial)} libraries in uns['spatial'] ({libraries[:12]}), so "
            "there is no one image to draw under its spots",
            field="library_id",
            fix="name one of the libraries above as library_id, or leave the image out",
        )
    library = library_id or next(iter(spatial))
    entry = spatial.get(library)
    if not isinstance(entry, dict):
        if library_id:
            # A named section with no image used to come back as "no image", and the producer drew
            # a histology overlay with no histology in it (hunt 2026-09-30, u20b-viz-rest-21).
            raise LayerError(
                f"there is no image for library {library_id!r}; uns['spatial'] holds {libraries[:12]}",
                field="uns['spatial']",
                fix="name one of the libraries above as library_id, or leave the image out",
            )
        return None, 1.0, ""
    images = entry.get("images") or {}
    factors = entry.get("scalefactors") or {}
    if image_key not in images:
        available = sorted(images)
        raise LayerError(
            f"this object has no {image_key!r} image" + (f"; it has {available}" if available else ""),
            field=f"uns['spatial']['{library}']['images']",
            fix="ask for one of the images above, or leave the image out",
        )
    factor_key = f"tissue_{image_key}_scalef"
    if factor_key not in factors:
        # The pairing rule: an image without its own scalefactor cannot be registered, and
        # borrowing the other resolution's factor is a silent several-fold error.
        raise LayerError(
            f"the {image_key!r} image has no {factor_key} to register it against the coordinates",
            field=f"uns['spatial']['{library}']['scalefactors']",
            fix=(
                "re-read the platform output with the data converter, which writes the image and "
                "its scalefactor together; do not pair an image with another resolution's factor"
            ),
        )
    # No scalefactor applies to an array index. The catalogue refused every tissue map on index
    # coordinates until plain maps were allowed (hunt 2026-09-30, u20b-viz-rest-26); the overlay
    # stays refused, here, so no producer can register an image under an index lattice.
    try:
        from spatialomicsgym.viz.profile import coords_look_like_array_indices

        indices = coords_look_like_array_indices(spatial_coords(adata)[0])
    except LayerError:
        indices = False
    if indices:
        raise LayerError(
            "the coordinates are array row and column indices, not positions, so no scalefactor "
            "registers the image under them",
            field="obsm['spatial']",
            fix="leave the image out to draw the map on the index lattice, or re-read the platform "
            "output with the data converter, which stores pixel positions",
        )
    return images[image_key], float(factors[factor_key]), str(library)


def normalize_for_display(values: Any, *, mode: str, integral: bool | None) -> tuple[Any, str]:
    """``(values, what was done)``. Only an integer matrix is transformed under ``auto``."""
    import numpy as np

    array = np.asarray(values, dtype=float)
    mode = (mode or "auto").lower()
    if mode == "none":
        return array, "as stored"
    if mode == "log1p" or (mode == "auto" and integral is True):
        return np.log1p(array), "log1p of the stored counts"
    if mode == "auto":
        return array, "as stored (already transformed)"
    if mode == "cpm_log1p":
        total = float(np.nansum(array)) or 1.0
        return np.log1p(array / total * 1e6), "counts per million, log1p"
    return array, "as stored"
