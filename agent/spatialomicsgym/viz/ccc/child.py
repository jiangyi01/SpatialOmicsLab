"""The reader child's two communication operations, ``ccc_describe`` and ``ccc_data`` (Program 10).

``serve(spec, rows)`` is the whole of it: ``sog_portal.vizchild`` hands a request here before it opens any dataset (a
table-only bundle has no h5ad) and turns a :class:`~.errors.CCCRefusal` into its own refusal. The spec:

``op``                ``ccc_describe`` | ``ccc_data``
``fd``, ``ccc_fds``   the bundle's descriptors, the anchor's first (at most ``detect.MAX_FDS`` in all: a per-section
                      set of ``detect.MAX_SECTIONS`` sections of ``detect.MAX_BUNDLE`` files, and a domain table)
``ccc_backend``       one of ``detect.BACKENDS``
``ccc_roles``         ``[{"role", "sep", "fields"?}]`` aligned with ``[fd, *ccc_fds]``: each file's ``detect`` role,
                      ``","`` or ``"\\t"`` for a table (``None`` for the h5ad), and what ``detect`` read from its
                      name (``mode`` for a squidpy Ripley table, ``db`` for COMMOT's), handed on as ``Source.fields``;
                      a per-section set's members each carry their ``section`` (``detect.section_of``), and a role
                      may then be named once per section
                      One more role is accepted beside a backend's own: ``labels``, a domain table a run wrote, which
                      a ``{"source": "labels"}`` group is read from when the backend writes no ``labels`` of its own
                      (:func:`_joined_labels`)
``table_max_bytes``   ``SOG_VIZ_TABLE_MAX_BYTES``
``limits``            the child's limits (``memory_bytes``, ``max_rows``; ``max_levels`` optional)
``facet``             (data) one of ``facets.FACETS`` but ``field``, which is drawn through the colour families
``params``            (data) ``key``/``role``/``table``/``view``/``kind``/``sample``/``p``/``stat``/``focus``/``db``/
                      ``section``/``links`` (text; ``links`` is ``all`` or ``cross-section``) and
                      ``k`` (1-20), ``permutations`` (0-200), ``top`` (1-50), ``seed`` (0-2^32), ``a`` (a group index)
``group``             (data) ``{"source": "h5ad", "column": <obs position>}`` or ``{"source": "labels", "column": <j>}``
                      (the ``labels`` member's column ``j``; its first column names the rows)

``rows`` is the brushed subset (u32, at most ``facets.MAX_ROWS_SUBSET``) or ``None``.

Every answer is plain JSON (NaN as ``None``), at most ``facets.FIT_BYTES``, and carries no path: the files arrive as
descriptors and nothing here asks for a name.

Every describe answer says the result's ``mode`` (``"3d"`` | ``"per-section-2d"`` | ``"2d"``, ``None`` when its files
cannot tell), its ``sections`` (``[{label, z_um}]``), its ``frame`` and ``legacy_rank_z``. COMMOT reads a per-section
set itself (its sections are stacked into one index space); for every other backend a per-section set is answered
section by section here: describe from each section's files, data from the one ``section`` asked for (the first when
none is).
"""

from __future__ import annotations

import os
import stat
from typing import Any

import numpy as np

from spatialomicsgym.viz import h5lite

from . import detect
from .adapters import ADAPTERS
from .adapters.base import Source, cap_groups
from .errors import CCCRefusal
from .facets import FACETS, FIT_BYTES, LINKS, MAX_ROWS_SUBSET, clean, size_of
from .facets import INT_PARAMS as _INT_PARAMS
from .facets import TEXT_MAX as _TEXT_MAX
from .facets import TEXT_PARAMS as _TEXT_PARAMS

OPS = ("ccc_describe", "ccc_data")
_SEPS = (",", "\t")
#: The role of a domain table the portal adds to group a result by (``sog_portal.api.routers.viz``'s ``table.<record>`` groups).
LABELS_ROLE = "labels"
#: A joined table's names must be rows of the file for at least this share of its rows (``postanalysis.sources``'
#: ``_MIN_INDEX_OVERLAP``, the result tables' own rule).
_MIN_JOINED = 0.5


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _bad(detail: str) -> CCCRefusal:
    return CCCRefusal("bad_request", detail)


def _fields(entry: dict[str, Any]) -> dict[str, str]:
    """A role's ``fields`` (what ``detect`` read from the file's name): a few short texts, or ``bad_request``."""
    fields = entry.get("fields", {})
    if (
        not isinstance(fields, dict)
        or len(fields) > 4
        or not all(
            isinstance(k, str) and isinstance(v, str) and len(k) <= 32 and len(v) <= 64 for k, v in fields.items()
        )
    ):
        raise _bad("A file's fields are not a few short texts.")
    return dict(fields)


def _roles(spec: dict[str, Any], backend: str) -> list[tuple[int, str, str | None, dict[str, str]]]:
    """``[(fd, role, sep, fields)]`` for the bundle, or ``bad_request``."""
    fds = [spec.get("fd"), *(spec.get("ccc_fds") or [])] if isinstance(spec.get("ccc_fds", []), list) else []
    roles = spec.get("ccc_roles")
    if not fds or len(fds) > detect.MAX_FDS or not all(_strict_int(fd) and fd >= 0 for fd in fds):
        raise _bad(f"The request names no bundle of 1 to {detect.MAX_FDS} descriptors.")
    if not isinstance(roles, list) or len(roles) != len(fds):
        raise _bad("The request's roles do not line up with its descriptors.")
    known = (*detect.roles_of(backend), LABELS_ROLE)
    out = []
    for fd, entry in zip(fds, roles, strict=True):
        if not isinstance(entry, dict) or entry.get("role") not in known:
            raise _bad("The request names a file role this backend does not write.")
        role, sep = entry["role"], entry.get("sep")
        if (role == "h5ad") != (sep is None) or (sep is not None and sep not in _SEPS):
            raise _bad("A table's separator must be a comma or a tab, and the h5ad has none.")
        out.append((fd, role, sep, _fields(entry)))
    if len({(role, fields.get("section")) for _, role, _, fields in out}) != len(out):
        raise _bad("The request names one role twice.")
    sectioned = {fields.get("section") is not None for _, role, _, fields in out if role != LABELS_ROLE}
    if len(sectioned) > 1:
        raise _bad("A per-section set names a section for every one of its files.")
    if len({fields.get("section") for _, _, _, fields in out} - {None}) > detect.MAX_SECTIONS:
        raise _bad(f"A per-section set holds at most {detect.MAX_SECTIONS} sections.")
    return out


def _params(spec: dict[str, Any]) -> dict[str, Any]:
    params = spec.get("params", {})
    if not isinstance(params, dict) or len(params) > len(_INT_PARAMS) + len(_TEXT_PARAMS):
        raise _bad("The request's params are not an object of known parameters.")
    for name, value in params.items():
        if name in _INT_PARAMS:
            low, high = _INT_PARAMS[name]
            if not (_strict_int(value) and low <= value <= high):
                raise _bad(f"The request's {name} must be a whole number from {low} to {high:,}.")
        elif name in _TEXT_PARAMS:
            if not isinstance(value, str) or not 0 < len(value) <= _TEXT_MAX:
                raise _bad(f"The request's {name} must be a short text.")
        else:
            raise _bad("The request names a parameter no facet takes.")
    if params.get("role", "sender") not in ("sender", "receiver"):
        raise _bad("The request's role must be sender or receiver.")
    if params.get("links", "all") not in LINKS:
        raise _bad("The request's links must be 'all' or 'cross-section'.")
    return params


def _group(spec: dict[str, Any]) -> dict[str, Any] | None:
    group = spec.get("group")
    if group is None:
        return None
    if (
        not isinstance(group, dict)
        or group.get("source") not in ("h5ad", "labels")
        or not _strict_int(group.get("column"))
        or group["column"] < 0
    ):
        raise _bad('The request\'s group must be {"source": "h5ad" | "labels", "column": <a whole number>}.')
    return {"source": group["source"], "column": int(group["column"])}


def _limits(spec: dict[str, Any]) -> dict[str, int]:
    limits = spec.get("limits")
    cap = spec.get("table_max_bytes")
    if not isinstance(limits, dict) or not all(
        _strict_int(limits.get(k)) and limits[k] >= 1 for k in ("memory_bytes", "max_rows")
    ):
        raise _bad("The request's limits are missing or not whole numbers.")
    if not _strict_int(cap) or cap < 1:
        raise _bad("The request names no table size limit.")
    out = {"memory_bytes": int(limits["memory_bytes"]), "max_rows": int(limits["max_rows"]), "table_max_bytes": cap}
    if _strict_int(limits.get("max_levels")):
        out["max_levels"] = int(limits["max_levels"])
    return out


def serve(spec: dict[str, Any], rows: np.ndarray | None) -> dict:
    """One communication request's answer, or :class:`CCCRefusal`."""
    op = spec.get("op")
    if op not in OPS:
        raise _bad("The request names no communication operation.")
    backend = spec.get("ccc_backend")
    if backend not in ADAPTERS:
        raise _bad("The request names no communication backend this reader knows.")
    bundle = _roles(spec, backend)
    limits = _limits(spec)
    facet = params = group = None
    if op == "ccc_data":
        facet = spec.get("facet")
        if facet not in FACETS or facet == "field":
            raise _bad("The request names no facet with data of its own (the field is drawn as colours).")
        params, group = _params(spec), _group(spec)
    if rows is not None:
        rows = np.asarray(rows)
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or rows.size > MAX_ROWS_SUBSET:
            raise _bad(f"A brushed subset is at most {MAX_ROWS_SUBSET:,} row numbers.")
        rows = rows.astype(np.int64)
    sources = [Source(role, fd, sep, fields=fields) for fd, role, sep, fields in bundle]
    # A ``labels`` member beside a backend that writes none of its own is a domain table to group by, not a result
    # file: the adapter never sees it.
    joined = None
    if LABELS_ROLE not in detect.roles_of(backend):
        joined = next((s for s in sources if s.role == LABELS_ROLE), None)
        sources = [s for s in sources if s is not joined]
    opened: list[h5lite.H5AD] = []
    try:
        for source in sources:
            if source.role != "h5ad":
                continue
            try:
                info = os.fstat(source.fd)
            except OSError:
                raise _bad("The h5ad's descriptor was not handed to the reader.") from None
            if not stat.S_ISREG(info.st_mode):
                raise CCCRefusal("unreadable", "The h5ad is not a regular file.")
            source.h5 = h5lite.open_fd(source.fd)
            opened.append(source.h5)
        h5s = [source for source in sources if source.h5 is not None]
        if joined is not None and h5s and group is not None and group["source"] == LABELS_ROLE:
            group = _joined_labels(joined, h5s, group["column"], limits)
        adapter = ADAPTERS[backend]
        per_section = any(source.fields.get("section") for source in sources)
        if op == "ccc_describe":
            if per_section and backend != "commot":
                answer = _describe_sections(adapter, sources, limits)
            else:
                answer = adapter.describe(sources, limits)
            out = {
                "backend": backend,
                "label": answer["label"],
                "n_obs": answer["n_obs"],
                "unit": answer["unit"],
                "facets": answer["facets"],
                "groups": answer["groups"],
                "confirmed": True,
                "mode": answer.get("mode"),
                "sections": list(answer.get("sections") or []),
                "frame": answer.get("frame"),
                "legacy_rank_z": bool(answer.get("legacy_rank_z", False)),
                "undeclared_3d": bool(answer.get("undeclared_3d", False)),
                # False when no facet can be cut to one section (a pooled per-section squidpy enrichment): no scrub.
                "section_filterable": answer.get("section_filterable", True) is not False,
                "warnings": list(answer["warnings"]),
            }
            for name in ("cross_section_edge_fraction", "n_cross_section_edges"):
                if name in answer:
                    out[name] = answer[name]
        else:
            if params.get("links", "all") != "all" and backend != "commot":
                raise CCCRefusal(
                    "unsupported",
                    "No cross-section links: this result holds no spot-by-spot links to tell them by; only COMMOT's "
                    "3D results do.",
                )
            if per_section and backend != "commot":
                answer = _data_section(adapter, sources, facet, params, group, rows, limits)
            else:
                answer = adapter.data(sources, facet, params, group, rows, limits)
                said = answer["payload"].get("section") if isinstance(answer["payload"], dict) else None
                if params.get("section") is not None and said != params["section"]:
                    raise CCCRefusal("bad_request", "This result names no sections, so no section can be asked for.")
            out = {"facet": answer["facet"], "payload": answer["payload"], "warnings": list(answer["warnings"])}
    except h5lite.ReadError as exc:
        raise CCCRefusal(exc.kind, exc.detail, exc.knob or "") from None
    finally:
        for h5 in opened:
            h5.close()
    if op == "ccc_data" and out["facet"] == "matrix":
        out["warnings"] += cap_groups(out["payload"])
    out = clean(out)
    if size_of(out) > FIT_BYTES:
        raise CCCRefusal("too_large", "This answer is longer than the reader may send, even trimmed; ask for less.")
    return out


def _section_groups(sources: list[Source]) -> dict[str, list[Source]]:
    """A per-section set's sources by section, in label order (a domain table to group by is in none)."""
    out: dict[str, list[Source]] = {}
    for source in sources:
        label = source.fields.get("section")
        if label is not None:
            out.setdefault(label, []).append(source)
    return dict(sorted(out.items()))


def _describe_sections(adapter: Any, sources: list[Source], limits: dict[str, int]) -> dict:
    """A per-section set of a table backend, described section by section: the first section's facets (every
    section's files are the same tool's same outputs), the warnings of all of them once each, the sections in label
    order (no table gives a depth)."""
    by_section = _section_groups(sources)
    answers = {label: adapter.describe(members, limits) for label, members in by_section.items()}
    first = next(iter(answers.values()))
    warnings: list[str] = []
    for answer in answers.values():
        warnings += [w for w in answer["warnings"] if w not in warnings]
    n_obs = [a["n_obs"] for a in answers.values()]
    warnings.append(
        f"Per-section 2D: the tool ran on each of these {len(answers)} sections on its own, so nothing crosses a "
        "section; each facet answers for one section."
    )
    return {
        **first,
        "n_obs": sum(n_obs) if all(isinstance(n, int) for n in n_obs) else None,
        "mode": "per-section-2d",
        "sections": [{"label": label, "z_um": None} for label in answers],
        "frame": None,
        "legacy_rank_z": False,
        "undeclared_3d": False,
        "warnings": warnings,
    }


def _data_section(
    adapter: Any,
    sources: list[Source],
    facet: str,
    params: dict[str, Any],
    group: dict[str, Any] | None,
    rows: np.ndarray | None,
    limits: dict[str, int],
) -> dict:
    """One facet of one section of a per-section set: the ``section`` asked for, else the first (and said so)."""
    by_section = _section_groups(sources)
    asked = params.get("section")
    if asked is not None and asked not in by_section:
        known = ", ".join(list(by_section)[:12])
        raise CCCRefusal("bad_request", f"The request's section is not one of this result's ({known}).")
    label = asked if asked is not None else next(iter(by_section))
    rest = {k: v for k, v in params.items() if k != "section"}
    answer = adapter.data(by_section[label], facet, rest, group, rows, limits)
    warnings = list(answer["warnings"])
    if asked is None and len(by_section) > 1:
        warnings.insert(
            0,
            f"This per-section 2D result has {len(by_section)} sections; section {label} is shown. Choose another "
            "with section.",
        )
    payload = answer["payload"]
    if isinstance(payload, dict):
        # A curves answer's ``mode`` is Ripley's statistic (F, G or L): there the run's mode rides as ``per_section``.
        said = {"per_section": True} if answer["facet"] == "curves" else {"mode": "per-section-2d"}
        payload = {**payload, **said, "section": label}
    return {"facet": answer["facet"], "payload": payload, "warnings": warnings}


def _joined_labels(source: Source, h5s: list[Source], column: int, limits: dict[str, int]) -> dict[str, Any]:
    """The group each row of the bundle's h5ads has in a domain table a run wrote: ``group`` with ``codes`` (one per
    obs row of the first h5ad, ``-1`` for none), ``codes_by_fd`` (the same for every h5ad, by its descriptor: a
    per-section set is several), ``labels`` and ``name`` added, which the adapter groups by instead of an obs column.

    The table's first column names the rows and its column ``column`` holds their group; it is joined to the files'
    obs names as ``sog_portal.vizchild._join_table`` joins a result table -- one pass over the names, a block at a time,
    a name repeated in the table keeping its first row -- and refused ``no_join`` when under half of its names are
    rows of the result. A table that covers a whole stack does not join a block or one section of it, and the
    refusal says so: that is its commonest cause.
    """
    import pandas as pd

    from . import tables

    frame = tables.read(source.fd, source.sep or ",", limits, dtype=str)
    if not 1 <= column < frame.shape[1]:
        raise _bad("The group's column is not one of the grouping table's columns.")
    keep = (frame.iloc[:, 0].notna() & frame.iloc[:, column].notna()).to_numpy()
    names = frame.iloc[:, 0].to_numpy(dtype=object)[keep]
    first = ~pd.Index(names).duplicated()
    index = pd.Index(names[first], dtype=object)
    if not len(index):
        raise CCCRefusal("unsupported", "The grouping table has no row with both a name and a group.")
    codes, levels = pd.factorize(frame.iloc[:, column].to_numpy(dtype=object)[keep][first], sort=True)
    seen = np.zeros(len(index), dtype=bool)
    by_fd: dict[int, np.ndarray] = {}
    for member in h5s:
        h5 = member.h5
        out = np.full(h5.n_obs, -1, dtype=np.int64)
        for start, block in h5.obs_name_blocks():
            found = index.get_indexer(block)
            hit = found >= 0
            out[start : start + block.size][hit] = codes[found[hit]]
            seen[found[hit]] = True
        by_fd[member.fd] = out
    if seen.sum() < _MIN_JOINED * len(index):
        raise CCCRefusal(
            "no_join",
            f"Only {int(seen.sum()):,} of the grouping table's {len(index):,} names are rows of this result, under the "
            "half it takes to group the result by it: this table covers more cells than this result (a whole stack "
            "vs a block or one section) — group by a column of this result, or re-run on the cells the table covers.",
        )
    return {
        "source": LABELS_ROLE,
        "column": column,
        "codes": by_fd[h5s[0].fd],
        "codes_by_fd": by_fd,
        "labels": [str(v) for v in levels],
        "name": str(frame.columns[column]),
    }


__all__ = ["LABELS_ROLE", "OPS", "serve"]
