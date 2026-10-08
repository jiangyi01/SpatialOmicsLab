"""The communication panels' facets and every bound they are answered under. Standard library only.

The portal imports this beside :mod:`detect` (it never loads numpy, pandas or h5py), and the frontend names its views
by :data:`FACETS`. A bound is a promise about cost or size, so each is stated where it is enforced and repeated in the
answer that it shaped (``warnings``).

* ``field``      the signal sums per spot, as colours (listed by describe only; Task 2's colour families draw them)
* ``direction``  the per-spot direction of travel of a signal, binned into at most :data:`MAX_ARROWS` arrows
* ``matrix``     group by group: who signals to whom, with permutation z and p where they can be derived
* ``dotplot``    the strongest ligand-receptor pairs, group by group
* ``ranking``    a result table, sorted and capped at :data:`MAX_RANK_ROWS` rows
* ``curves``     squidpy's co-occurrence and Ripley curves, at most :data:`MAX_CURVE_BINS` points each
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

FACETS = ("field", "direction", "matrix", "dotplot", "ranking", "curves")

#: Groups beyond this many are pooled into one "other" group (the 63 largest keep their label): a 64 x 64 matrix is
#: the most a heatmap or chord diagram reads as anything.
MAX_GROUPS = 64
#: Ligand-receptor pairs a dot plot shows at most, and by default.
MAX_PAIRS = 50
DEFAULT_PAIRS = 20
#: Dots a dot plot answers at most: pairs x senders x receivers grows past what one answer line holds.
MAX_DOTPLOT_CELLS = 2000
#: Arrows a direction answer holds at most; the grid's side is ``floor(sqrt(MAX_ARROWS))`` = 64.
MAX_ARROWS = 4096
#: Partners per spot a direction is summed over (COMMOT's own plots use a handful).
MAX_K = 20
DEFAULT_K = 5
#: Label shuffles a permutation test runs at most, and by default.
MAX_PERMUTATIONS = 200
DEFAULT_PERMUTATIONS = 100
#: Links x permutations a permutation test may touch: each shuffle is one pass over the links (a bincount).
PERMUTE_BUDGET = 2_000_000_000
#: Rows a ranking answers at most.
MAX_RANK_ROWS = 500
#: Points per curve; a longer curve is averaged down in consecutive bins.
MAX_CURVE_BINS = 64
#: Rows a brushed subset may name.
MAX_ROWS_SUBSET = 100_000
#: A data request's whole-number parameters and their accepted ranges (the portal checks them, the child again).
INT_PARAMS = {
    "k": (1, MAX_K),
    "permutations": (0, MAX_PERMUTATIONS),
    "top": (1, MAX_PAIRS),
    "seed": (0, 2**32 - 1),
    "a": (0, MAX_GROUPS - 1),
}
#: Its text parameters (at most :data:`TEXT_MAX` characters): each adapter checks the value against its result.
#: ``focus`` is a group label: ``direction`` beside a ``group`` draws only that group's spots' links (COMMOT).
#: ``db`` is a COMMOT database: one h5ad can hold several, and the explorer's family names the one it colours by.
#: ``section`` is one section's label (a per-section set's, or a 3D result's section column's): every facet answers
#: for that section alone. ``links`` is ``"all"`` or ``"cross-section"`` -- only the links whose sender and receiver
#: lie in different sections, a 3D result's inferred cross-section communication (:data:`LINKS`).
TEXT_PARAMS = ("key", "role", "table", "view", "kind", "sample", "p", "stat", "focus", "db", "section", "links")
#: The values ``links`` takes.
LINKS = ("all", "cross-section")
#: A result's modes: a graph or distances built in the aligned 3D frame, one run per section stacked along z, one plane.
MODES = ("3d", "per-section-2d", "2d")
TEXT_MAX = 256
#: Stored values of one spot-by-spot matrix the reader loads at most, whatever the memory cap allows.
MAX_NNZ = 50_000_000
#: The reader child's answer rides in its header line, capped at 1 MiB with room for the envelope around it
#: (``sog_portal.vizchild._DESCRIBE_FIT_BYTES`` is the same number).
FIT_BYTES = 1024 * 1024 - 64 * 1024


def size_of(payload: Any) -> int:
    """The bytes ``payload`` takes as the child sends it: compact UTF-8 JSON of :func:`clean`'s copy.

    Measured on the cleaned copy because an adapter's payload is still raw when it is fitted -- a real table holds
    NaN (squidpy leaves a constant gene's p-value empty, a MISTy target can miss a measure) -- and the NaN only becomes
    ``None`` in ``child.serve``. Sizing the raw payload with ``allow_nan=False`` failed on the first NaN.
    """
    return len(json.dumps(clean(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def fit(payload: dict, trimmers: list[Callable[[dict], bool]]) -> list[str]:
    """Apply ``trimmers`` in order until ``payload`` is at most :data:`FIT_BYTES`; the warnings that says.

    Each trimmer edits the payload in place and returns whether it took anything out; its docstring's first line says
    what, and becomes the warning. Trimmers after the one that made it fit are never called. An answer that still does
    not fit is the caller's to refuse.
    """
    warnings: list[str] = []
    for trim in trimmers:
        if size_of(payload) <= FIT_BYTES:
            break
        if trim(payload):
            said = (trim.__doc__ or "").strip().splitlines()
            what = said[0].rstrip(".") if said else "part of it was left out"
            warnings.append(f"The answer was cut to fit the reader's answer line: {what}.")
    return warnings


def clean(value: Any) -> Any:
    """``value`` as plain JSON: NaN and infinities become ``None``, tuples become lists, numpy scalars plain numbers.

    Stdlib only, so a numpy scalar is recognised by its ``item`` method rather than by its type.
    """
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        return clean(item())
    return value


__all__ = [
    "DEFAULT_K",
    "DEFAULT_PAIRS",
    "DEFAULT_PERMUTATIONS",
    "FACETS",
    "FIT_BYTES",
    "INT_PARAMS",
    "LINKS",
    "MAX_ARROWS",
    "MAX_CURVE_BINS",
    "MAX_DOTPLOT_CELLS",
    "MAX_GROUPS",
    "MAX_K",
    "MAX_NNZ",
    "MAX_PAIRS",
    "MAX_PERMUTATIONS",
    "MAX_RANK_ROWS",
    "MAX_ROWS_SUBSET",
    "MODES",
    "PERMUTE_BUDGET",
    "TEXT_MAX",
    "TEXT_PARAMS",
    "clean",
    "fit",
    "size_of",
]
