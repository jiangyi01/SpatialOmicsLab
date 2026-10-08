"""COMMOT: spot-level signal sums, and the spot-by-spot matrices every derived view is read from.

What the platform's worker writes (``tools/commot_worker.py``, ``spatial_communication`` only), and so what is here:

* ``obsm['commot-<db>-sum-sender' | '-sum-receiver']`` -- DataFrames, one column per signal: ``s-<LIG>-<REC>`` /
  ``r-...`` per ligand-receptor pair, ``s-<PATHWAY>`` per pathway, ``s-total-total`` for everything;
* ``obsp['commot-<db>-<key>']`` -- the same keys, sparse, sender rows by receiver columns;
* ``uns['commot-<db>-info']`` -- ``df_ligrec`` (ligand, receptor, pathway): the ONLY source of a key's kind. Nothing
  here knows a pathway's or a ligand's name;
* the two sums again as ``commot_<db>_sum_sender.csv`` / ``_sum_receiver.csv``, indexed by barcode.

Not written: a vector field (``communication_direction``) or a cluster matrix (``cluster_communication``). Both are
derived here (:mod:`..direction`, :mod:`..groups`) -- unless the file does hold ``obsm['commot_sender_vf-<db>-<key>']``,
which is then used as COMMOT computed it, and the answer says so.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from spatialomicsgym.viz.ccc import direction as dirn
from spatialomicsgym.viz.ccc import geometry as geo
from spatialomicsgym.viz.ccc import groups as grp
from spatialomicsgym.viz.ccc import h5 as h5x
from spatialomicsgym.viz.ccc import tables
from spatialomicsgym.viz.ccc.errors import CCCRefusal
from spatialomicsgym.viz.ccc.facets import (
    DEFAULT_K,
    DEFAULT_PAIRS,
    DEFAULT_PERMUTATIONS,
    MAX_ARROWS,
    MAX_DOTPLOT_CELLS,
    MAX_K,
    MAX_PAIRS,
    MAX_PERMUTATIONS,
    PERMUTE_BUDGET,
    fit,
)

from .base import Source, by_role, choice, halve_rows, matrix_payload, not_offered, ranking_payload

if TYPE_CHECKING:
    from collections.abc import Mapping

    from spatialomicsgym.viz.h5lite import H5AD

_NEEDS_H5AD = "direction, matrix and dotplot need the results .h5ad (commot_<db>_results.h5ad)"
_ROLES = ("sender", "receiver")
#: df_ligrec is a table of ligand-receptor pairs; more rows than this is no database COMMOT ships.
_LIGREC_MAX_ROWS = 100_000


class _Result:
    """One COMMOT h5ad, as describe and every data read see it: its database, keys, their kinds and their matrices.

    ``named`` is the database the file's name says (``commot_<db>_results.h5ad``, ``detect``'s ``db`` field); ``asked``
    the one a request names (the explorer's family's ``db``). The database read is the one asked -- refused when the
    file holds no such database --, else the one named, and only when the file holds neither its first (sorted), which
    :meth:`which` then says. One file can hold several databases; reading the first regardless drew one database's
    matrices under another's family.
    """

    def __init__(self, h5: H5AD, named: str | None = None, asked: str | None = None) -> None:
        self.h5 = h5
        dbs = h5x.commot_dbs(h5)
        if not dbs:
            raise CCCRefusal(
                "unsupported",
                "This file holds no COMMOT signal sums (obsm['commot-<db>-sum-sender']), so it is not "
                "a COMMOT result the explorer can read.",
            )
        if asked is not None and asked not in dbs:
            raise CCCRefusal(
                "bad_request", f"This file holds no COMMOT results for {asked!r}; it holds {', '.join(dbs)}."
            )
        self.named = named
        self.db = asked or (named if named in dbs else dbs[0])
        self.other_dbs = [d for d in dbs if d != self.db]
        self.sum_key = {role: f"commot-{self.db}-sum-{role}" for role in _ROLES}
        prefix = f"commot-{self.db}-"
        self.nnz = {k.key[len(prefix) :]: k.nnz for k in h5.obsp_keys() if k.key.startswith(prefix)}
        self.columns: dict[str, dict[str, int]] = {}
        for role in _ROLES:
            info = h5.obsm_key(self.sum_key[role])
            mark = f"{role[0]}-"
            names = list(info.columns or ()) if info is not None and info.kind == "dataframe" else []
            self.columns[role] = {c[len(mark) :]: j for j, c in enumerate(names) if c.startswith(mark)}
        self.kinds, self.has_ligrec = key_kinds(h5, self.db)
        listed = list(dict.fromkeys([*self.columns["sender"], *self.columns["receiver"], *self.nnz]))
        self.keys = sorted(listed, key=lambda k: (_KIND_ORDER.get(self.kind(k), 3), listed.index(k)))

    def which(self) -> str | None:
        """Which database is shown and why, when the file holds more than one; ``None`` when it holds one."""
        if not self.other_dbs:
            return None
        others = ", ".join(self.other_dbs)
        if self.named == self.db:
            why = "the file is named for it"
        elif self.named:
            why = f"the file is named for {self.named}, which it does not hold, so the first is shown"
        else:
            why = "the file's name names none, so the first is shown"
        return f"This file also holds COMMOT results for {others}; {self.db} is shown because {why}."

    def kind(self, key: str) -> str | None:
        return self.kinds.get(key, {}).get("kind")

    def key_entry(self, key: str) -> dict[str, Any]:
        return {"key": key, **self.kinds.get(key, {"kind": None})}

    def label(self, key: str) -> str:
        return key_label(key, self.kinds.get(key, {}))

    def matrix(self, key: str, limits: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """``(senders, receivers, values, nnz)`` of every link of ``obsp['commot-<db>-<key>']``."""
        indptr, indices, data, transposed = h5x.read_obsp(self.h5, f"commot-{self.db}-{key}", limits)
        rows = dirn.csr_rows(indptr)
        return (indices, rows, data, data.size) if transposed else (rows, indices, data, data.size)

    def sent(self, key: str, role: str, mask: np.ndarray | None) -> float | None:
        j = self.columns[role].get(key)
        if j is None:
            return None
        values = self.h5.obsm_column(self.sum_key[role], j)
        return float(np.nansum(values if mask is None else values[mask]))

    def column(self, key: str, role: str) -> np.ndarray | None:
        j = self.columns[role].get(key)
        return None if j is None else self.h5.obsm_column(self.sum_key[role], j)


_KIND_ORDER = {"total": 0, "pathway": 1, "pair": 2}


def _named_db(sources: list[Source]) -> str | None:
    """The database the bundle's h5ad is named for (``detect``'s ``db`` field, from ``commot_<db>_results.h5ad``)."""
    source = by_role(sources).get("h5ad")
    named = source.fields.get("db") if source is not None else None
    return named if isinstance(named, str) and named else None


def _asked_db(params: Mapping[str, Any]) -> str | None:
    """The database a request names (the explorer's family's ``db``), or ``None``."""
    asked = params.get("db")
    return asked if isinstance(asked, str) and asked else None


def key_kinds(h5: H5AD, db: str) -> tuple[dict[str, dict[str, Any]], bool]:
    """``({key: {kind, ...}}, whether the file has a df_ligrec)`` for COMMOT database ``db``.

    Read from ``uns['commot-<db>-info']['df_ligrec']`` alone -- nothing here knows a pathway's or a ligand's name: a
    ``<ligand>-<receptor>`` row is a ``pair`` (with its ``ligand``, ``receptor`` and ``pathway``), a value of the
    ``pathway`` column a ``pathway`` (with ``n_pairs``, its rows), and ``total-total`` is always the ``total``. The
    explorer's colour families (``sog_portal.vizchild``) read kinds by this same rule.
    """
    ligrec = h5.uns_frame(f"commot-{db}-info", "df_ligrec", max_rows=_LIGREC_MAX_ROWS) or {}
    kinds: dict[str, dict[str, Any]] = {}
    if {"ligand", "receptor"} <= set(ligrec):
        paths = ligrec.get("pathway", [None] * len(ligrec["ligand"]))
        for lig, rec, path in zip(ligrec["ligand"], ligrec["receptor"], paths, strict=True):
            entry = {"kind": "pair", "ligand": str(lig), "receptor": str(rec), "pathway": _text(path)}
            kinds.setdefault(f"{lig}-{rec}", entry)
        members = Counter(str(p) for p in paths if _text(p) is not None)
        for path, n_pairs in members.items():
            kinds.setdefault(path, {"kind": "pathway", "n_pairs": n_pairs})
    kinds["total-total"] = {"kind": "total"}
    return kinds, bool(ligrec)


def key_label(key: str, entry: Mapping[str, Any]) -> str:
    """A key in words, by its :func:`key_kinds` entry: ``CXCL12 → ACKR3``, ``CXCL pathway``, ``all pairs``; a key
    with no entry is its own name."""
    if entry.get("kind") == "pair":
        return f"{entry['ligand']} → {entry['receptor']}"
    if entry.get("kind") == "pathway":
        return f"{key} pathway"
    if entry.get("kind") == "total":
        return "all pairs"
    return key


def _text(value: Any) -> str | None:
    return None if value is None or (isinstance(value, float) and value != value) else str(value)


@dataclass
class _Part:
    """One h5ad of the bundle: the whole result, or one section of a per-section set. ``offset`` is where its spots
    start in the bundle's one index space (the sections one after another, in depth order)."""

    source: Source
    result: _Result
    section: str | None
    z_um: float | None
    offset: int = 0
    rank: int = 0
    _geometry: geo.Geometry | None = None

    @property
    def h5(self) -> H5AD:
        return self.result.h5

    @property
    def n(self) -> int:
        return int(self.result.h5.n_obs)

    @property
    def geometry(self) -> geo.Geometry:
        if self._geometry is None:
            self._geometry = geo.of(self.result.h5)
        return self._geometry


def _first_section(sources: list[Source], section: str | None = None) -> list[Source]:
    """A per-section set's sources of one section (``section``, else the first in label order); every source of a
    single result."""
    labels = sorted({s.fields["section"] for s in sources if s.fields.get("section")})
    if not labels:
        return sources
    if section is not None and section not in labels:
        raise CCCRefusal(
            "bad_request", f"The request's section is not one of this result's ({', '.join(labels[:12])})."
        )
    wanted = section if section is not None else labels[0]
    return [s for s in sources if s.fields.get("section") == wanted]


def _top_k(
    senders: np.ndarray, receivers: np.ndarray, values: np.ndarray, role: str, k: int, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(own, other, values)``: each of the role's own spots' ``k`` strongest links (senders for a sender field,
    receivers for a receiver field), ``direction.top_k_per_row`` on the links grouped by that end (a stable sort, so
    of equal values the first stored still wins)."""
    own = np.asarray(senders if role == "sender" else receivers, dtype=np.int64)
    other = np.asarray(receivers if role == "sender" else senders, dtype=np.int64)
    order = np.argsort(own, kind="stable")
    own, other, values = own[order], other[order], np.asarray(values, dtype=np.float64)[order]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(own, minlength=int(n)))]).astype(np.int64)
    return dirn.top_k_per_row(indptr, other, values, k)


def _h5_sources(sources: list[Source]) -> list[Source]:
    return [s for s in sources if s.role == "h5ad" and s.h5 is not None]


def _per_section(sources: list[Source]) -> bool:
    return any(s.fields.get("section") for s in sources)


def _parts(sources: list[Source], asked: str | None, section: str | None = None) -> list[_Part]:
    """The bundle's h5ads as parts, in depth order (label order when a depth is unknown); one section's alone when
    ``section`` names one of a per-section set."""
    h5s = _h5_sources(sources)
    if not _per_section(sources):
        return [_Part(h5s[0], _Result(h5s[0].h5, h5s[0].fields.get("db"), asked), None, None)]
    parts = []
    for source in h5s:
        label = source.fields.get("section")
        if not label:
            raise CCCRefusal("bad_request", "A per-section set names a section for every file.")
        parts.append(_Part(source, _Result(source.h5, source.fields.get("db"), asked), label, None))
    for part in parts:
        part.z_um = geo.depth_of_section(part.h5)
    if all(part.z_um is not None for part in parts):
        parts.sort(key=lambda part: (part.z_um, part.section))
    else:
        parts.sort(key=lambda part: part.section)
    for rank, part in enumerate(parts):
        part.rank = rank
    if section is not None:
        chosen = [part for part in parts if part.section == section]
        if not chosen:
            known = ", ".join(part.section for part in parts[:12])
            raise CCCRefusal("bad_request", f"The request's section is not one of this result's ({known}).")
        parts = chosen
    offset = 0
    for part in parts:
        part.offset = offset
        offset += part.n
    return parts


def _sections_of(parts: list[_Part]) -> list[dict[str, Any]]:
    return [{"label": part.section, "z_um": part.z_um} for part in parts]


def _part_keys(parts: list[_Part]) -> list[str]:
    """Every key any section's matrices hold, in the first section's order then the others'."""
    return list(dict.fromkeys(k for part in parts for k in part.result.nnz))


def _group(parts: list[_Part], group: dict | None, masks: list[np.ndarray | None], max_levels: int) -> tuple[Any, ...]:
    """``(codes, labels, pooled, group descriptor)`` over the parts' spots one after another -- the obs column asked
    for (by its position in the first file, by its name in the others), else the first one describe lists, or a
    domain table's groups the child already joined to the spots (``child._joined_labels``: ``codes`` given, one array
    per file by its descriptor). Spots outside a brushed row mask take no group; labels are joined by their text."""
    first = parts[0].h5
    source = "h5ad"
    if group is None:
        offered = h5x.obs_groups(first, max_levels)
        if not offered:
            raise CCCRefusal("bad_request", "This file has no categorical obs column to group the spots by.")
        column = int(offered[0]["id"].split(".")[1])
    elif group.get("source") == "labels" and group.get("codes") is not None:
        source, column = "labels", int(group["column"])
    elif group.get("source") != "h5ad":
        raise CCCRefusal("bad_request", "A COMMOT result is grouped by an obs column of its h5ad or a domain table.")
    else:
        column = int(group["column"])
    labels: list[str] = []
    chunks: list[np.ndarray] = []
    if source == "labels":
        labels, name = list(group["labels"]), str(group["name"])
        by_fd = group.get("codes_by_fd") or {}
        for part in parts:
            codes = by_fd.get(part.source.fd) if by_fd else group["codes"]
            chunks.append(np.asarray(codes, dtype=np.int64))
    else:
        found = next((c for c in first.obs_columns() if c.index == column), None)
        if found is None:
            raise CCCRefusal("bad_request", "The group asked for is not a labelled obs column of this file.")
        name = found.name
        for part in parts:
            other = part.h5.obs_column(name)
            if other is None:
                raise CCCRefusal("unreadable", f"A section of this result has no obs column {name!r} to group by.")
            codes, own, _name = h5x.group_codes(part.h5, other.index)
            remap = np.empty(len(own) + 1, dtype=np.int64)
            for i, label in enumerate(own):
                if label not in labels:
                    labels.append(label)
                remap[i] = labels.index(label)
            remap[-1] = -1
            chunks.append(np.where(codes >= 0, remap[np.clip(codes, 0, len(own))], -1))
    out = []
    for part, codes, mask in zip(parts, chunks, masks, strict=True):
        if codes.size != part.n:
            raise CCCRefusal("unreadable", "A grouping does not have one group per spot of this result.")
        out.append(codes if mask is None else np.where(mask, codes, -1))
    codes, labels, pooled = grp.pooled_codes(np.concatenate(out), labels)
    prefix = "obs" if source == "h5ad" else "labels"
    return codes, labels, pooled, {"id": f"{prefix}.{column}", "label": name, "source": source}


def _links(parts: list[_Part], key: str, limits: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(senders, receivers, values)`` of every part's ``obsp['commot-<db>-<key>']``, in the bundle's index space.
    The parts' links together are held to one matrix's caps."""
    total = sum(part.result.nnz.get(key, 0) for part in parts)
    h5x.preflight(total, limits)
    senders, receivers, values = [], [], []
    for part in parts:
        if key not in part.result.nnz:
            continue
        s, r, v, _ = part.result.matrix(key, limits)
        senders.append(np.asarray(s, dtype=np.int64) + part.offset)
        receivers.append(np.asarray(r, dtype=np.int64) + part.offset)
        values.append(np.asarray(v, dtype=np.float64))
    if not senders:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0)
    return np.concatenate(senders), np.concatenate(receivers), np.concatenate(values)


def _masks(parts: list[_Part], rows: np.ndarray | None) -> list[np.ndarray | None]:
    """The brushed rows as one mask per part. A per-section set is several files: a subset is taken only of the one
    section asked for (``section``), whose file the rows number."""
    if rows is None:
        return [None] * len(parts)
    if len(parts) > 1:
        raise CCCRefusal(
            "bad_request",
            "A brushed subset numbers the spots of one file; this per-section result is several. Choose a section "
            "first, then brush its spots.",
        )
    return [h5x.rows_mask(rows, parts[0].n)]


class _Scope:
    """What a link-based facet reads: the parts, the mode, the section asked for and the links kept, with the
    cross-section count every 3D answer carries."""

    def __init__(self, sources: list[Source], params: Mapping[str, Any]) -> None:
        self.links = geo.links_param(params)
        section = params.get("section")
        self.per_section = _per_section(sources)
        self.parts = _parts(sources, _asked_db(params), section if self.per_section else None)
        self.section = section
        self.mode = "per-section-2d" if self.per_section else self.parts[0].geometry.mode
        self.geometry = None if self.per_section else self.parts[0].geometry
        if self.links == "cross-section":
            geo.refuse_cross_section(self.geometry, self.per_section)
        self.section_code = None
        if section is not None and not self.per_section:
            if self.geometry is None or self.geometry.codes is None:
                raise CCCRefusal("bad_request", "This result names no sections, so no section can be asked for.")
            self.section_code = self.geometry.code_of(str(section))

    @property
    def n(self) -> int:
        return sum(part.n for part in self.parts)

    @property
    def three_d(self) -> bool:
        return self.geometry is not None and self.geometry.dims == 3

    def crossing(self, senders: np.ndarray, receivers: np.ndarray) -> np.ndarray | None:
        """The cross-section mask of these links, or ``None`` where cross-section cannot be told."""
        if not self.three_d or self.geometry.not_distances or self.geometry.codes is None:
            return None
        return dirn.cross_section_mask(senders, receivers, self.geometry.codes)

    def kept(self, senders: np.ndarray, receivers: np.ndarray) -> tuple[np.ndarray, int | None]:
        """``(links kept, cross-section count)``: a 3D section asked for keeps the links sent from its spots; then
        ``links: "cross-section"`` keeps those joining two sections. The count is over the links before the second
        cut (``None`` where cross-section cannot be told)."""
        keep = np.ones(senders.size, dtype=bool)
        if self.section_code is not None:
            keep &= self.geometry.codes[senders] == self.section_code
        crossing = self.crossing(senders, receivers)
        count = None if crossing is None else int(np.count_nonzero(crossing & keep))
        if self.links == "cross-section" and crossing is not None:
            keep &= crossing
        return keep, count

    def warnings(self, n_cross: int | None) -> list[str]:
        out: list[str] = []
        if self.three_d and self.geometry.not_distances:
            out.append(f"This is a {self.geometry.not_distances}.")
        elif self.three_d and n_cross == 0:
            out.append(geo.ZERO_CROSS)
        return out

    def fields(self, n_cross: int | None) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode, "section": self.section, "links": self.links}
        if self.three_d:
            out["n_cross_section_links"] = n_cross
        return out


class CommotAdapter:
    backend = "commot"

    # -------------------------------------------------------------------------------------------- describe
    def describe(self, sources: list[Source], limits: Mapping[str, int]) -> dict:
        if not _h5_sources(sources):
            out = self._describe_tables(_first_section(sources), limits)
            if _per_section(sources):
                labels = sorted({s.fields["section"] for s in sources if s.fields.get("section")})
                out["mode"], out["sections"] = "per-section-2d", [{"label": x, "z_um": None} for x in labels]
            return out
        parts = _parts(sources, None)
        first = parts[0]
        result = first.result
        warnings: list[str] = []
        which = result.which()
        if which:
            warnings.append(which)
        if not result.has_ligrec:
            warnings.append(
                "This file has no ligand-receptor table (uns['commot-<db>-info']['df_ligrec']), so which keys are "
                "pairs and which are pathways is not known."
            )
        nnz: dict[str, int] = {}
        for part in parts:
            for key, count in part.result.nnz.items():
                nnz[key] = nnz.get(key, 0) + int(count)
        keys = result.keys + [k for k in _part_keys(parts) if k not in result.keys]
        field_keys = [
            result.key_entry(k) for k in result.keys if k in result.columns["sender"] or k in result.columns["receiver"]
        ]
        linked = [{"key": k, "nnz": nnz[k], "kind": result.kind(k)} for k in keys if k in nnz]
        precomputed = sorted(
            k.key for k in first.h5.obsm_keys() if k.key.startswith(("commot_sender_vf-", "commot_receiver_vf-"))
        )
        facets: list[dict] = [
            {
                "id": "field",
                "roles": [r for r in _ROLES if result.columns[r]],
                "keys": field_keys,
                "source": "h5ad",
                "obsm": {r: result.sum_key[r] for r in _ROLES if result.columns[r]},
            }
        ]
        if linked:
            pairs = [k for k in keys if result.kind(k) == "pair" and k in nnz]
            facets += [
                {
                    "id": "direction",
                    "keys": linked,
                    "roles": list(_ROLES),
                    "k": {"default": DEFAULT_K, "max": MAX_K},
                    "precomputed": precomputed,
                },
                {
                    "id": "matrix",
                    "keys": linked,
                    "stat": "sum",
                    "permutations": {"default": DEFAULT_PERMUTATIONS, "max": MAX_PERMUTATIONS},
                },
                {"id": "dotplot", "n_pairs": len(pairs), "top": {"default": DEFAULT_PAIRS, "max": MAX_PAIRS}},
            ]
        else:
            warnings.append(
                "This file holds no spot-by-spot matrices (obsp), so direction, matrix and dotplot are not offered."
            )
        tables_offered = (
            [{"id": "pairs", "label": "Ligand-receptor pairs"}, {"id": "pathways", "label": "Pathways"}]
            if result.has_ligrec
            else [{"id": "signals", "label": "Signals"}]
        )
        facets.append({"id": "ranking", "tables": tables_offered})
        groups = h5x.obs_groups(first.h5, int(limits.get("max_levels", 4096)))
        if len(parts) > 1:
            fields = {
                "mode": "per-section-2d",
                "sections": _sections_of(parts),
                "frame": {**first.geometry.frame, "z_source": None},
                "legacy_rank_z": False,
            }
            warnings.append(
                f"Per-section 2D: COMMOT ran on each of these {len(parts)} sections on its own, so no signal crosses "
                "a section; the sections are stacked "
                + (
                    "at their depth in micrometres."
                    if all(p.z_um is not None for p in parts)
                    else "in their label order (no file gives a depth)."
                )
            )
        else:
            fields, said = geo.describe_fields(first.geometry)
            warnings += said
        out = {
            "label": f"COMMOT ({result.db})",
            "n_obs": sum(part.n for part in parts),
            "unit": "spot",
            "facets": facets,
            "groups": groups,
            "warnings": warnings,
            **fields,
        }
        out["warnings"] += fit(out, [_drop_pair_keys_from_views, _drop_pair_keys_from_field])
        return out

    def _sum_tables(self, sources: list[Source], limits: Mapping[str, int]) -> dict[str, list[str]]:
        """The keys each sum CSV holds, confirmed by its ``s-``/``r-`` header."""
        found: dict[str, list[str]] = {}
        for role, source in by_role(sources).items():
            if role not in ("sum_sender", "sum_receiver"):
                continue
            header = tables.header(source.fd, source.sep or ",", limits)
            mark = "s-" if role == "sum_sender" else "r-"
            keys = [c[len(mark) :] for c in header[1:] if c.startswith(mark)]
            if not keys:
                raise CCCRefusal(
                    "unsupported", f"A COMMOT {role.replace('_', ' ')} table has no {mark}<signal> columns."
                )
            found[role] = keys
        return found

    def _describe_tables(self, sources: list[Source], limits: Mapping[str, int]) -> dict:
        found = self._sum_tables(sources, limits)
        if not found:
            raise CCCRefusal("unsupported", "This COMMOT result has neither its .h5ad nor its sum tables.")
        keys = list(dict.fromkeys([*found.get("sum_sender", []), *found.get("sum_receiver", [])]))
        needs = {"needs": ["h5ad"]}
        return {
            "label": "COMMOT",
            "n_obs": None,
            "unit": "spot",
            "facets": [
                {
                    "id": "field",
                    "roles": [r for r in _ROLES if f"sum_{r}" in found],
                    "keys": [{"key": k, "kind": "total" if k == "total-total" else None} for k in keys],
                    "source": "tables",
                },
                {"id": "direction", **needs},
                {"id": "matrix", **needs},
                {"id": "dotplot", **needs},
                {"id": "ranking", "tables": [{"id": "signals", "label": "Signals"}]},
            ],
            "groups": [],
            "warnings": [
                f"Only the sum tables are here: {_NEEDS_H5AD}.",
                "Without the .h5ad's ligand-receptor table, which signals are pairs and which are pathways is not known.",
            ],
        }

    # -------------------------------------------------------------------------------------------- data
    def data(
        self,
        sources: list[Source],
        facet: str,
        params: Mapping[str, Any],
        group: dict | None,
        rows: np.ndarray | None,
        limits: Mapping[str, int],
    ) -> dict:
        if facet == "ranking":
            return self._ranking(sources, params, rows, limits)
        if facet not in ("direction", "matrix", "dotplot"):
            raise not_offered(facet, "COMMOT gives field, direction, matrix, dotplot and ranking.")
        if not _h5_sources(sources):
            raise not_offered(facet, f"{_NEEDS_H5AD}.")
        scope = _Scope(sources, params)
        masks = _masks(scope.parts, rows)
        if facet == "direction":
            return self._direction(scope, params, group, masks, limits)
        if facet == "matrix":
            return self._matrix(scope, params, group, masks, limits)
        return self._dotplot(scope, params, group, masks, limits)

    def _direction(
        self, scope: _Scope, params: Mapping[str, Any], group: dict | None, masks: list, limits: Mapping
    ) -> dict:
        keys = _part_keys(scope.parts)
        key = choice(params, "key", keys, "total-total" if "total-total" in keys else None)
        role = choice(params, "role", _ROLES, "sender")
        k = int(params.get("k", DEFAULT_K))
        if scope.per_section:
            return self._direction_stacked(scope, key, role, k, params, group, masks, limits)
        part = scope.parts[0]
        result, geometry = part.result, part.geometry
        xy = geometry.coords
        stored = f"commot_{role}_vf-{result.db}-{key}"
        n_cross: int | None = None
        if geometry.dims == 2 and scope.links == "all" and result.h5.obsm_key(stored) is not None:
            field = np.asarray(result.h5.obsm(stored, dims=2), dtype=np.float64)
            method = f"COMMOT's own {role} vector field (obsm['{stored}']), binned: each arrow is one grid cell's sum."
        else:
            indptr, indices, data, transposed = h5x.read_obsp(result.h5, f"commot-{result.db}-{key}", limits)
            senders, receivers = dirn.csr_rows(indptr), np.asarray(indices, dtype=np.int64)
            if transposed:
                senders, receivers = receivers, senders
            crossing = scope.crossing(senders, receivers)
            # The count is over the links this answer considers, as matrix and dotplot count them (``_Scope.kept``):
            # with a section asked, the links sent from that section's spots.
            _sent, n_cross = scope.kept(senders, receivers)
            if crossing is None:
                n_cross = None
            if scope.links == "cross-section" and crossing is not None:
                senders, receivers, data = senders[crossing], receivers[crossing], np.asarray(data)[crossing]
            own, other, values = _top_k(senders, receivers, data, role, k, part.n)
            senders, receivers = (own, other) if role == "sender" else (other, own)
            field = dirn.vector_field(xy, senders, receivers, values, role)
            towards = "to its" if role == "sender" else "from its"
            which = " joining two sections" if scope.links == "cross-section" else ""
            method = (
                f"COMMOT's direction, derived here from obsp['commot-{result.db}-{key}']: per spot the weight-sum of "
                f"unit vectors {towards} {k} strongest {'receivers' if role == 'sender' else 'senders'}{which}, "
                "pointing sender to receiver; each arrow is one grid cell's sum."
            )
        finite = np.isfinite(xy).all(axis=1)
        bounds = (xy[finite].min(axis=0), xy[finite].max(axis=0)) if finite.any() else None
        used = np.ones(part.n, dtype=bool) if masks[0] is None else masks[0]
        # ``focus`` (a label of ``group``, honoured only beside one): the arrows of that group's links alone -- the
        # network's selected group, drawn on the tissue. A spot's vector is the sum of the links at its ROLE side
        # (``vector_field``'s ``at``), so keeping only the group's spots keeps exactly the links whose sender (role
        # sender) or receiver (role receiver) is in the group, and the bins count only those spots. ``section`` keeps
        # that section's spots the same way.
        focus = params.get("focus") if group is not None else None
        kept = used
        if focus is not None:
            codes, labels, _pooled, _described = _group(scope.parts, group, masks, int(limits.get("max_levels", 4096)))
            if focus not in labels:
                raise CCCRefusal(
                    "bad_request", f"The focus is not one of the {len(labels):,} labels this grouping offers."
                )
            kept = used & (codes == labels.index(focus))
        if scope.section_code is not None:
            kept = kept & (geometry.codes == scope.section_code)
        binner = dirn.bin_arrows_3d if geometry.dims == 3 else dirn.bin_arrows
        binned = binner(xy[kept], field[kept], bounds=bounds)
        payload = {
            "facet": "direction",
            "key": key,
            "role": role,
            "k": k,
            "dims": geometry.dims,
            **binned,
            "y_down": geometry.dims == 2,
            "n_spots_with_signal": int(np.count_nonzero(np.linalg.norm(field[kept], axis=1) > 0)),
            "rows_used": int(used.sum()),
            "method": method,
            "frame": geometry.frame,
            **scope.fields(n_cross),
        }
        if focus is not None:
            payload["focus"] = focus
        warnings = scope.warnings(n_cross)
        return {"facet": "direction", "payload": payload, "warnings": warnings + fit(payload, [_keep_populated_arrows])}

    def _direction_stacked(
        self,
        scope: _Scope,
        key: str,
        role: str,
        k: int,
        params: Mapping[str, Any],
        group: dict | None,
        masks: list,
        limits: Mapping,
    ) -> dict:
        """A per-section set's direction: each section binned in its own 2D plane, its arrows stacked at its depth
        (``[x, y, z, dx, dy, 0, n, mag]``), the sections sharing the arrow cap."""
        focus = params.get("focus") if group is not None else None
        focus_codes = None
        if focus is not None:
            codes, labels, _pooled, _described = _group(scope.parts, group, masks, int(limits.get("max_levels", 4096)))
            if focus not in labels:
                raise CCCRefusal(
                    "bad_request", f"The focus is not one of the {len(labels):,} labels this grouping offers."
                )
            focus_codes = codes == labels.index(focus)
        h5x.preflight(sum(part.result.nnz.get(key, 0) for part in scope.parts), limits)
        side = max(1, math.isqrt(max(1, MAX_ARROWS // max(1, len(scope.parts)))))
        depth_known = all(part.z_um is not None for part in scope.parts)
        arrows: list[list[float]] = []
        rows_used = n_signal = 0
        for part, mask in zip(scope.parts, masks, strict=True):
            z = part.z_um if depth_known else float(part.rank)
            if key not in part.result.nnz:
                continue
            xy = part.geometry.coords[:, :2]
            indptr, indices, data, transposed = h5x.read_obsp(part.h5, f"commot-{part.result.db}-{key}", limits)
            senders, receivers = dirn.csr_rows(indptr), np.asarray(indices, dtype=np.int64)
            if transposed:
                senders, receivers = receivers, senders
            own, other, values = _top_k(senders, receivers, data, role, k, part.n)
            pair = (own, other) if role == "sender" else (other, own)
            field = dirn.vector_field(xy, pair[0], pair[1], values, role)
            used = np.ones(part.n, dtype=bool) if mask is None else mask
            kept = used
            if focus_codes is not None:
                kept = used & focus_codes[part.offset : part.offset + part.n]
            rows_used += int(used.sum())
            n_signal += int(np.count_nonzero(np.hypot(field[kept, 0], field[kept, 1]) > 0))
            binned = dirn.bin_arrows(xy[kept], field[kept], max_arrows=side * side)
            arrows += [[x, y, z, dx, dy, 0.0, n, mag] for x, y, dx, dy, n, mag in binned["arrows"]]
        payload = {
            "facet": "direction",
            "key": key,
            "role": role,
            "k": k,
            "dims": 3,
            "grid": [side, side, len(scope.parts)],
            "arrows": arrows,
            "max_mag": max((math.hypot(a[3], a[4]) for a in arrows), default=0.0),
            "stacked": True,
            "z_is": "um" if depth_known else "section order",
            "sections": _sections_of(scope.parts),
            "y_down": True,
            "n_spots_with_signal": n_signal,
            "rows_used": rows_used,
            "method": (
                f"COMMOT's direction per section, derived from each section's obsp['commot-<db>-{key}'] and binned "
                "in that section's own plane; each section's arrows are stacked at its depth"
                + (" in micrometres." if depth_known else " (its place in label order: no file gives a depth).")
            ),
            "frame": {**scope.parts[0].geometry.frame, "z_source": None},
            **scope.fields(None),
        }
        if focus is not None:
            payload["focus"] = focus
        return {"facet": "direction", "payload": payload, "warnings": fit(payload, [_keep_populated_arrows])}

    def _matrix(
        self, scope: _Scope, params: Mapping[str, Any], group: dict | None, masks: list, limits: Mapping
    ) -> dict:
        keys = _part_keys(scope.parts)
        key = choice(params, "key", keys, "total-total" if "total-total" in keys else None)
        wanted = int(params.get("permutations", DEFAULT_PERMUTATIONS))
        seed = int(params.get("seed", 0))
        codes, labels, pooled, described = _group(scope.parts, group, masks, int(limits.get("max_levels", 4096)))
        senders, receivers, values = _links(scope.parts, key, limits)
        keep, n_cross = scope.kept(senders, receivers)
        senders, receivers, values = senders[keep], receivers[keep], values[keep]
        nnz = int(values.size)
        k = len(labels)
        sums, n_links = grp.aggregate(senders, receivers, values, codes, k)
        n = np.bincount(codes[codes >= 0], minlength=k)
        warnings: list[str] = scope.warnings(n_cross)
        ran = grp.permutation_budget(nnz, wanted)
        z = p = None
        if 0 < ran < wanted:
            warnings.append(
                f"{ran} of the {wanted} permutations asked for were run: links x permutations is held under "
                f"{PERMUTE_BUDGET:,}, and this matrix has {nnz:,} links."
            )
        elif ran == 0 and wanted > 0:
            warnings.append(
                f"No permutation test was run: this matrix's {nnz:,} links are past the {PERMUTE_BUDGET:,} links x "
                "permutations a test may touch, so z and p are left out."
            )
        if ran > 0:
            strata = None
            if len(scope.parts) > 1:
                strata = np.concatenate([np.full(part.n, i, dtype=np.int64) for i, part in enumerate(scope.parts)])
                warnings.append("The permutation test shuffles the group labels within each section.")
            z, p = grp.permutation(senders, receivers, values, codes, k, n=ran, seed=seed, strata=strata)
        payload = matrix_payload(
            key=key,
            key_label=scope.parts[0].result.label(key),
            stat="sum",
            group=described,
            labels=labels,
            values=sums,
            n=n.tolist(),
            pooled=pooled,
            mean=grp.mean_per_pair(sums, n),
            z=z,
            p=p,
            permutations=ran,
            seed=seed,
            n_links=n_links,
            rows_used=int(
                sum(part.n if m is None else int(m.sum()) for part, m in zip(scope.parts, masks, strict=True))
            ),
            rows_total=scope.n,
        )
        payload.update(scope.fields(n_cross))
        if scope.links == "cross-section":
            payload["caption_text"] = "inferred cross-section communication"
        return {"facet": "matrix", "payload": payload, "warnings": warnings}

    def _dotplot(
        self, scope: _Scope, params: Mapping[str, Any], group: dict | None, masks: list, limits: Mapping
    ) -> dict:
        top = int(params.get("top", DEFAULT_PAIRS))
        codes, labels, pooled, described = _group(scope.parts, group, masks, int(limits.get("max_levels", 4096)))
        first = scope.parts[0].result
        keys = _part_keys(scope.parts)
        pairs = [k for k in keys if first.kind(k) == "pair"]
        totals: dict[str, float | None] = {}
        # A 3D section asked for ranks the pairs by what that section's spots sent, as its dots count only the links
        # sent from them (``_Scope.kept``); ranking over the whole block showed another section's strongest pairs.
        ranking_masks = masks
        if scope.section_code is not None:
            here = scope.geometry.codes == scope.section_code
            ranking_masks = [here if mask is None else mask & here for mask in masks]
        for key in pairs:
            sent = [
                part.result.sent(key, "sender", mask) for part, mask in zip(scope.parts, ranking_masks, strict=True)
            ]
            known = [v for v in sent if v is not None]
            totals[key] = float(sum(known)) if known else None
        ranked = sorted((k for k in pairs if totals[k] is not None), key=lambda k: -totals[k])[:top]
        warnings: list[str] = []
        if len(ranked) < len(pairs):
            warnings.append(f"The {len(ranked)} of {len(pairs)} pairs that sent the most are shown.")
        if scope.links == "cross-section":
            warnings.append(
                "Pairs are ranked by everything they sent; the dots sum only their inferred cross-section links."
            )
        k = len(labels)
        n = np.bincount(codes[codes >= 0], minlength=k)
        cells: list[list] = []
        n_cross_total: int | None = 0 if scope.three_d and scope.geometry.codes is not None else None
        for i, key in enumerate(ranked):
            senders, receivers, values = _links(scope.parts, key, limits)
            keep, n_cross = scope.kept(senders, receivers)
            if n_cross_total is not None and n_cross is not None:
                n_cross_total += n_cross
            sums, _ = grp.aggregate(senders[keep], receivers[keep], values[keep], codes, k)
            means = grp.mean_per_pair(sums, n)
            for s, r in zip(*np.nonzero(sums > 0), strict=True):
                cells.append([i, int(s), int(r), float(sums[s, r]), float(means[s, r])])
        truncated = len(cells) > MAX_DOTPLOT_CELLS
        if truncated:
            cells = sorted(cells, key=lambda c: -c[3])[:MAX_DOTPLOT_CELLS]
            warnings.append(f"The {MAX_DOTPLOT_CELLS:,} strongest of the dots are shown.")
        payload = {
            "facet": "dotplot",
            "group": {**described, "pooled": pooled},
            "labels": labels,
            "n": n.tolist(),
            "pairs": [
                {
                    "key": key,
                    "pathway": first.kinds[key].get("pathway"),
                    "ligand": first.kinds[key]["ligand"],
                    "receptor": first.kinds[key]["receptor"],
                    "total": totals[key],
                }
                for key in ranked
            ],
            "cells": cells,
            "ranked_by": "sum of s-<pair> over the rows used",
            "truncated": truncated,
            **scope.fields(n_cross_total),
        }
        return {"facet": "dotplot", "payload": payload, "warnings": scope.warnings(n_cross_total) + warnings}

    def _ranking(
        self,
        sources: list[Source],
        params: Mapping[str, Any],
        rows: np.ndarray | None,
        limits: Mapping,
    ) -> dict:
        per = _per_section(sources)
        section = params.get("section")
        if not _h5_sources(sources):
            chosen = _first_section(sources, section if per else None)
            if section is not None and not per:
                raise CCCRefusal("bad_request", "This result names no sections, so no section can be asked for.")
            answer = self._ranking_tables(chosen, params, rows, limits)
            if per:
                answer["payload"]["section"] = chosen[0].fields.get("section")
                answer["payload"]["mode"] = "per-section-2d"
            return answer
        parts = _parts(sources, _asked_db(params), section if per else None)
        masks = _masks(parts, rows)
        if section is not None and not per:
            geometry = parts[0].geometry
            if geometry.codes is None:
                raise CCCRefusal("bad_request", "This result names no sections, so no section can be asked for.")
            here = geometry.codes == geometry.code_of(str(section))
            masks = [here if masks[0] is None else masks[0] & here]
        result = parts[0].result
        offered = ["pairs", "pathways"] if result.has_ligrec else ["signals"]
        table = choice(params, "table", offered, offered[0])
        wanted = {"pairs": ("pair",), "pathways": ("pathway",), "signals": ("total", "pathway", "pair", None)}[table]
        listed = list(dict.fromkeys(k for part in parts for k in part.result.keys))
        keys = [k for k in listed if result.kind(k) in wanted]
        records = []
        for key in keys:
            sent = received = sending = None
            for part, mask in zip(parts, masks, strict=True):
                values = part.result.column(key, "sender")
                if values is not None:
                    used = values if mask is None else values[mask]
                    sent = (sent or 0.0) + float(np.nansum(used))
                    sending = (sending or 0) + int(np.count_nonzero(used > 0))
                got = part.result.sent(key, "receiver", mask)
                if got is not None:
                    received = (received or 0.0) + got
            records.append({"key": key, "sent": sent, "received": received, "spots sending": sending})
        records.sort(key=lambda r: -(r["sent"] if r["sent"] is not None else -np.inf))
        if table == "pairs":
            columns = [("pair", "text"), ("pathway", "text"), ("ligand", "text"), ("receptor", "text")]
            lead = [
                [r["key"], *(result.kinds[r["key"]].get(f) for f in ("pathway", "ligand", "receptor"))] for r in records
            ]
        elif table == "pathways":
            columns = [("pathway", "text"), ("pairs", "number")]
            lead = [[r["key"], result.kinds[r["key"]].get("n_pairs")] for r in records]
        else:
            columns = [("signal", "text")]
            lead = [[r["key"]] for r in records]
        columns += [("sent", "number"), ("received", "number"), ("spots sending", "number")]
        out_rows = [[*a, r["sent"], r["received"], r["spots sending"]] for a, r in zip(lead, records, strict=True)]
        label = {"pairs": "Ligand-receptor pairs", "pathways": "Pathways", "signals": "Signals"}[table]
        payload = ranking_payload(
            table=table,
            label=f"{label} by signal sent",
            columns=columns,
            rows=out_rows,
            n_total=len(records),
            sort_by="sent",
            desc=True,
        )
        payload["mode"] = "per-section-2d" if per else parts[0].geometry.mode
        payload["section"] = section
        return {"facet": "ranking", "payload": payload, "warnings": fit(payload, [halve_rows])}

    def _ranking_tables(
        self, sources: list[Source], params: Mapping[str, Any], rows: np.ndarray | None, limits: Mapping
    ) -> dict:
        found = self._sum_tables(sources, limits)
        choice(params, "table", ["signals"], "signals")
        warnings = []
        if rows is not None:
            warnings.append(
                "A brushed subset needs the .h5ad, which ties a row to a spot: these sums are over every spot."
            )
        totals: dict[str, dict[str, Any]] = {}
        for role, source in by_role(sources).items():
            if role not in found:
                continue
            frame = tables.read(source.fd, source.sep or ",", limits)
            mark = "s-" if role == "sum_sender" else "r-"
            for column in frame.columns[1:]:
                if not str(column).startswith(mark):
                    continue
                values = tables.numbers(frame[column])
                entry = totals.setdefault(str(column)[len(mark) :], {})
                entry["sent" if role == "sum_sender" else "received"] = float(np.nansum(values))
                if role == "sum_sender":
                    entry["spots sending"] = int(np.count_nonzero(values > 0))
        order = sorted(totals, key=lambda k: -totals[k].get("sent", totals[k].get("received", 0.0)))
        rows = [[k, totals[k].get("sent"), totals[k].get("received"), totals[k].get("spots sending")] for k in order]
        payload = ranking_payload(
            table="signals",
            label="Signals by signal sent",
            columns=[("signal", "text"), ("sent", "number"), ("received", "number"), ("spots sending", "number")],
            rows=rows,
            n_total=len(rows),
            sort_by="sent" if "sum_sender" in found else "received",
            desc=True,
        )
        return {"facet": "ranking", "payload": payload, "warnings": warnings + fit(payload, [halve_rows])}


def _drop_pair_keys_from_views(described: dict) -> bool:
    """listed only totals and pathways under direction and matrix (every pair is still in the field)"""
    changed = False
    for facet in described["facets"]:
        if facet["id"] in ("direction", "matrix") and any(k.get("kind") == "pair" for k in facet["keys"]):
            facet["keys"] = [k for k in facet["keys"] if k.get("kind") != "pair"]
            changed = True
    return changed


def _drop_pair_keys_from_field(described: dict) -> bool:
    """listed only totals and pathways in the field"""
    for facet in described["facets"]:
        if facet["id"] == "field" and any(k.get("kind") == "pair" for k in facet["keys"]):
            facet["keys"] = [k for k in facet["keys"] if k.get("kind") != "pair"]
            return True
    return False


def _keep_populated_arrows(payload: dict) -> bool:
    """kept the half of the arrows drawn from the most spots"""
    arrows = payload["arrows"]
    if len(arrows) < 2:
        return False
    # ``n`` is second to last in both shapes: [x, y, dx, dy, n, mag] and [x, y, z, dx, dy, dz, n, mag].
    payload["arrows"] = sorted(arrows, key=lambda a: -a[-2])[: len(arrows) // 2]
    return True


ADAPTER = CommotAdapter()
