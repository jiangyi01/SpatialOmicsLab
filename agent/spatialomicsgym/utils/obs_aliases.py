"""The obs column aliases and the spread-sample probe, shared by the readiness check and the viewer.

These lived in ``spatialomicsgym.agent.data_validation``, and every reader in ``spatialomicsgym.viz``
imported them from there. Importing anything under ``spatialomicsgym.agent`` runs its package
``__init__``, which builds ``STCoscientist`` -- langgraph and langchain_core, 0.89 s measured
2026-10-01 -- so a figure that only wanted to know whether ``obs`` has an ``x``/``y`` pair paid for
the whole agent, and the explorer's child process, which must never load the agent stack, could not
use the same rules the figures use at all.

So the definitions moved here and nothing else changed: the code below is the code that was there,
byte for byte. ``data_validation`` imports and re-exports every name under the name it always had,
because these feed the scored DATA-READINESS prompt and its behaviour must not drift by a character.
``test/test_obs_aliases_moved_out_of_the_agent.py`` pins that each re-export *is* the object here.

Stdlib only at module scope. ``numpy`` and ``scipy`` are imported inside the one function that reads
values, as they were before; importing this module costs nothing (``spatialomicsgym.utils`` resolves
its submodules lazily, PEP 562).
"""

from __future__ import annotations

import re
from typing import Any

# Column names that are semantically equivalent to ``cell_type``.
_CELL_TYPE_ALIASES: list[str] = [
    "cell_type",
    "CellType",
    "celltype",
    "cell_types",
    "annotation",
    "cell_annotation",
    "cluster_annotation",
    "labels",
]

# Potential obs column names that hold spatial coordinates when obsm is missing.
_SPATIAL_COORD_OBS_PAIRS: list[tuple[str, str]] = [
    ("x", "y"),
    ("X", "Y"),
    ("x_centroid", "y_centroid"),
    ("X_centroid", "Y_centroid"),
    ("spatial_x", "spatial_y"),
    ("xcoord", "ycoord"),
    ("array_row", "array_col"),
]


# The finder and its name rules came from ``program9/harness-and-core`` (merged 2026-10-02), where they were
# written in ``agent/data_validation``; they live here with the alias table they read.
def _normalise_obs_name(name: str) -> str:
    """``Cell_Type``, ``cell.type`` and ``celltype`` are one name: lower case, separators dropped."""
    return re.sub(r"[^0-9a-z]", "", str(name).lower())


#: The alias table above, normalised. Derived rather than restated, so the two cannot drift.
_CELL_TYPE_NAME_CORES: frozenset[str] = frozenset(_normalise_obs_name(a) for a in _CELL_TYPE_ALIASES)

#: Words a curated annotation column carries around a cell-type name, on either side:
#: ``celltype_plot`` (the Xenium kidney file the scored trials stage), ``cell_type_annot`` (the
#: MERFISH brain-aging file), ``predicted_labels`` (celltypist), ``author_cell_type``,
#: ``Annotation_Level1``. Longest first only for legibility; the search below tries them all.
_CELL_TYPE_NAME_QUALIFIERS: tuple[str, ...] = (
    "predicted",
    "consensus",
    "refined",
    "author",
    "manual",
    "coarse",
    "level1",
    "level2",
    "level3",
    "annot",
    "label",
    "broad",
    "final",
    "major",
    "minor",
    "fine",
    "lvl1",
    "lvl2",
    "lvl3",
    "main",
    "name",
    "plot",
    "pred",
    "l1",
    "l2",
    "l3",
)

#: At most this many qualifiers are peeled off one name. Three covers ``predicted_cell_type_l1_label``;
#: beyond that a name is a sentence, and the core it reduces to is a coincidence.
_CELL_TYPE_MAX_QUALIFIERS = 3

#: A level that is a whole number, written as text: ``"7"``, ``"-1"``, ``"3.0"``.
_INTEGER_TEXT_RE = re.compile(r"[+-]?\d+(?:\.0*)?")


def _cell_type_name_distance(name: str) -> int | None:
    """How many qualifiers separate *name* from a cell-type alias, or ``None`` if none reach one.

    Breadth-first over every order the qualifiers can be peeled in, so the answer is the fewest
    peels and does not depend on the order of the list: a greedy pass that took ``pred`` before
    ``predicted`` would turn ``predicted_labels`` into ``ictedlabels`` and stop there.
    """
    frontier = {_normalise_obs_name(name)}
    seen = set(frontier)
    for peeled in range(_CELL_TYPE_MAX_QUALIFIERS + 1):
        if frontier & _CELL_TYPE_NAME_CORES:
            return peeled
        following: set[str] = set()
        for stem in frontier:
            for word in _CELL_TYPE_NAME_QUALIFIERS:
                if len(stem) <= len(word):
                    continue
                if stem.startswith(word):
                    following.add(stem[len(word) :])
                if stem.endswith(word):
                    following.add(stem[: -len(word)])
        frontier = following - seen
        seen |= following
        if not frontier:
            return None
    return None


def _holds_cell_type_labels(series: Any) -> bool:
    """Could this obs column be a set of cell-type labels at all?

    Not "is it one" -- the name decided that. This only refuses what cannot be: a number (an id or
    a score, whatever it is called), whole numbers written as text (``"0"``..``"12"`` is a cluster
    id), a single level, and a level for nearly every observation (a barcode or cell id). Fewer
    than two observations per level on average is the line, because a label that names one cell
    is an identifier, not a type.
    """
    try:
        import pandas as pd

        dtype = series.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            if pd.api.types.is_numeric_dtype(dtype.categories.dtype):
                return False
        elif pd.api.types.is_numeric_dtype(dtype):  # booleans included: pandas counts them as numeric
            return False
        levels = [str(v).strip() for v in pd.Series(series).astype("object").dropna().unique()]
    except Exception:
        return False
    if len(levels) < 2 or len(levels) * 2 > len(series):
        return False
    return not all(_INTEGER_TEXT_RE.fullmatch(v) for v in levels)


def _find_cell_type_column(obs_columns: list[str], obs: Any = None) -> str | None:
    """Return the obs column that holds cell-type labels, or ``None``.

    An exact alias always wins and is decided from the names alone, exactly as it always was.

    Given ``obs`` as well -- the obs frame, so the values can be read -- a column the alias table
    does not spell is still found when its name reduces to an alias once case, separators and the
    qualifiers in :data:`_CELL_TYPE_NAME_QUALIFIERS` are set aside, AND its values pass
    :func:`_holds_cell_type_labels`. Both, because each alone is wrong: the name alone calls an
    integer ``cell_type_label`` code a cell type, and the content alone calls ``sample``,
    ``region``, ``timepoint`` and every Leiden partition one. The fewest qualifiers wins, then obs
    order.

    Calibrated on the ten h5ad files the scored trials stage, read backed. It finds
    ``celltype_plot`` in the Xenium kidney file, where the exact aliases found nothing, and leaves
    the viz profile of the other nine byte-identical, as it does the nineteen h5ad files the trial
    trees kept. The negatives among them, all left alone: ``ident``, ``region`` and ``time`` (both
    kidney files, one of them staged with its cell types removed) and ``CN`` (the other);
    ``sample``, ``SampleName`` and ``condition`` (the three dBit-seq files); ``cluster`` (the motif
    file, which stays a cluster); ``timepoint`` and ``barcode`` (both ovary files); ``donor_id``,
    ``tissue``, ``fov`` and ``clust_annot`` (MERFISH brain, which keeps its exact ``cell_type``).

    The readiness gate calls this with names only. Its recovery action copies the column it finds
    into the user's file as ``cell_type``, so widening what it accepts is a separate decision.
    """
    col_set = set(obs_columns)
    for alias in _CELL_TYPE_ALIASES:
        if alias in col_set:
            return alias
    if obs is None:
        return None
    ranked: list[tuple[int, int, str]] = []
    for position, column in enumerate(obs_columns):
        distance = _cell_type_name_distance(column)
        if distance is None:
            continue
        try:
            series = obs[column]
        except Exception:
            continue
        if _holds_cell_type_labels(series):
            ranked.append((distance, position, column))
    return min(ranked)[2] if ranked else None


def _find_spatial_coord_columns(obs_columns: list[str]) -> tuple[str, str] | None:
    """Return a (col_x, col_y) pair from *obs_columns* if spatial coordinates are present."""
    col_set = set(obs_columns)
    for cx, cy in _SPATIAL_COORD_OBS_PAIRS:
        if cx in col_set and cy in col_set:
            return (cx, cy)
    return None


#: How many rows the raw-counts check reads, and across how many evenly spaced blocks.
#: Blocks rather than a prefix, because leading empty spots are ordinary -- Visium writes barcodes
#: in array order so the off-tissue rim comes first, a filtered-not-subset object keeps its dropped
#: spots as zero rows in place, and a concatenation leads with whichever section was listed first.
#: A window of zeros satisfies "every value is a non-negative whole number" no matter what the rest
#: of the matrix holds, so reading only the top of the file answers a log1p matrix "raw counts" with
#: full confidence. Contiguous *within* a block because one backed-CSR row range is one contiguous
#: read of ``data``.
_COUNT_SAMPLE_ROWS = 100
_COUNT_SAMPLE_BLOCKS = 5


def _count_sample_blocks(n_obs: int) -> list[tuple[int, int]]:
    """Up to :data:`_COUNT_SAMPLE_ROWS` rows, as evenly spaced half-open ``(start, stop)`` ranges."""
    if n_obs <= _COUNT_SAMPLE_ROWS:
        return [(0, n_obs)]
    per_block = max(1, _COUNT_SAMPLE_ROWS // _COUNT_SAMPLE_BLOCKS)
    step = n_obs // _COUNT_SAMPLE_BLOCKS
    return [(i * step, min(i * step + per_block, n_obs)) for i in range(_COUNT_SAMPLE_BLOCKS)]


def _nonzero_value_sample(adata: Any) -> Any:
    """The **nonzero** values of a spread sample of ``adata.X``, or ``None`` if it cannot be read.

    Only nonzeros carry information here: zero is a non-negative whole number in every matrix ever
    written, normalized or not. Returning them alone also makes the sparse path free -- the values
    of a CSR row range are one contiguous slice of ``data``, with no need to reconstruct the matrix.
    """
    import numpy as np
    import scipy.sparse as sp

    X = adata.X
    blocks = _count_sample_blocks(int(adata.n_obs))
    chunks: list[Any] = []

    # Backed sparse datasets (anndata ``_CSRDataset``) do NOT support direct positional slicing:
    # ``X[:n]`` raises ``AttributeError`` (no ``_validate_indices``) on modern anndata, and the class
    # exposes ``to_memory()`` but not ``toarray``. ``to_memory()`` in turn loads the ENTIRE on-disk
    # matrix, which OOMs on a real Xenium/Slide-seqV2 file -- so read the backing h5py group
    # directly, and fall back to a size-capped ``to_memory`` only if that layout is not accessible.
    if hasattr(X, "to_memory"):
        grp = getattr(X, "group", None) or getattr(X, "_group", None)
        read = False
        if grp is not None and all(k in grp for k in ("indptr", "data")):
            try:
                for start, stop in blocks:
                    ptr = grp["indptr"][start : stop + 1]
                    lo, hi = int(ptr[0]), int(ptr[-1])
                    if hi > lo:
                        chunks.append(np.asarray(grp["data"][lo:hi]))
                read = True
            except Exception:
                chunks = []
        if not read:
            if adata.n_obs > 50000:
                return None  # too large to materialize safely; do not block on an unverifiable check
            mem = X.to_memory()
            for start, stop in blocks:
                block = mem[start:stop]
                chunks.append(block.data if sp.issparse(block) else np.asarray(block).ravel())
    elif sp.issparse(X):
        for start, stop in blocks:
            chunks.append(np.asarray(X[start:stop].data))
    else:
        for start, stop in blocks:
            chunks.append(np.asarray(X[start:stop]).ravel())

    chunks = [c for c in chunks if getattr(c, "size", 0)]
    if not chunks:
        return np.empty(0)
    values = np.concatenate(chunks)
    return values[values != 0]


#: Public names for the viewer and the figures. The SAME objects as the underscored ones above --
#: an alias, never a copy -- so there is one table and one rule, whichever name a caller reads.
CELL_TYPE_ALIASES = _CELL_TYPE_ALIASES
SPATIAL_COORD_OBS_PAIRS = _SPATIAL_COORD_OBS_PAIRS
find_cell_type_column = _find_cell_type_column
find_spatial_coord_columns = _find_spatial_coord_columns
nonzero_value_sample = _nonzero_value_sample

__all__ = [
    "CELL_TYPE_ALIASES",
    "SPATIAL_COORD_OBS_PAIRS",
    "find_cell_type_column",
    "find_spatial_coord_columns",
    "nonzero_value_sample",
]
