"""Which communication tool wrote a file, told by its NAME and its header shape. Standard library only.

The portal imports this to decide whether a record opens the communication panels before any reader child starts,
and the portal process never loads numpy, pandas or h5py (``test/test_webui_stays_out_of_the_heavy_stack.py``).

What it does not look at, on purpose:

* **what the analysis found** -- a pathway, a ligand, a cell type. A list of known names is a list of the results
  someone happened to see; the next database's names are not on it.
* **a record's ``produced_by``** -- a scrubbed tool-name summary, not authoritative. ``sog_run_provenance.json``
  names the tool but is never registered as a record.

The names are the ones each worker writes (``tools/<tool>_worker.py``); the real outputs under ``test/test_data`` are
walked by ``test/test_the_explorer_tells_a_communication_result_by_its_files.py``, so a worker that starts writing a
new name shows up there. Inside an adapter a match is confirmed by keys and headers (COMMOT's obsm/obsp/uns keys,
deeplinc's square frame whose index is its header) before anything is drawn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

BACKENDS = ("commot", "squidpy", "neighborseq", "deeplinc", "spacet", "mistyr", "ncem", "spaotsc")

#: ``(backend, role, regex on the bare file NAME)``. spaotsc's ``labels.csv`` and ``transport_plan.csv`` are names any
#: tool could write: they count only beside its ``signaling_scores.csv`` (:data:`ANCHOR_ROLES`).
_PATTERNS = (
    ("commot", "h5ad", r"^commot_(?P<db>[^_]+)_results\.h5ad$"),
    ("commot", "sum_sender", r"^commot_(?P<db>[^_]+)_sum_sender\.csv$"),
    ("commot", "sum_receiver", r"^commot_(?P<db>[^_]+)_sum_receiver\.csv$"),
    (
        "squidpy",
        "h5ad",
        r"^squidpy_(nhood_enrichment|co_occurrence|ripley_[FGL]|spatial_autocorr_(moran|geary)|centrality_scores)\.h5ad$",
    ),
    ("squidpy", "zscore", r"^squidpy_nhood_enrichment_zscore\.csv$"),
    ("squidpy", "count", r"^squidpy_nhood_enrichment_count\.csv$"),
    ("squidpy", "cooc_mean", r"^squidpy_co_occurrence_mean\.csv$"),
    ("squidpy", "centrality", r"^squidpy_centrality_scores\.csv$"),
    ("squidpy", "moran", r"^squidpy_moranI\.csv$"),
    ("squidpy", "geary", r"^squidpy_gearyC\.csv$"),
    ("squidpy", "ripley", r"^squidpy_ripley_(?P<mode>[FGL])\.csv$"),
    ("neighborseq", "interactions", r"^neighborseq_interactions\.csv$"),
    ("neighborseq", "top", r"^neighborseq_top_interactions\.csv$"),
    ("neighborseq", "predictions", r"^neighborseq_predictions\.csv$"),
    ("neighborseq", "label_map", r"^neighborseq_label_map\.csv$"),
    ("deeplinc", "scores", r"^deeplinc_interaction_scores\.csv$"),
    ("deeplinc", "pvalues", r"^deeplinc_pvalues\.csv$"),
    ("deeplinc", "significant", r"^deeplinc_significant_interactions\.csv$"),
    ("deeplinc", "adjacency", r"^deeplinc_adjacency(_edges)?\.csv$"),
    ("spacet", "proportions", r"^spacet_proportions\.csv$"),
    ("spacet", "cci", r"^spacet_cci_colocalization\.csv$"),
    ("mistyr", "importances", r"^mistyr_importances\.csv$"),
    ("mistyr", "top", r"^mistyr_top_interactions\.csv$"),
    ("mistyr", "performance", r"^mistyr_performance\.csv$"),
    ("mistyr", "contributions", r"^mistyr_contributions\.csv$"),
    ("ncem", "matrix", r"^ncem_communication_matrix\.csv$"),
    ("ncem", "strength", r"^ncem_communication_strength\.csv$"),
    ("ncem", "composition", r"^ncem_neighborhood_composition\.csv$"),
    ("ncem", "top_genes", r"^ncem_top_genes_per_type\.csv$"),
    ("spaotsc", "signaling", r"^signaling_scores\.csv$"),
    ("spaotsc", "labels", r"^labels\.csv$"),
    ("spaotsc", "transport", r"^transport_plan\.csv$"),
)
_COMPILED = tuple((backend, role, re.compile(pattern)) for backend, role, pattern in _PATTERNS)

#: The roles a bundle may be opened from: each is a result on its own. The others (a p-value table, a label map)
#: only ever join one.
ANCHOR_ROLES = {
    "commot": ("h5ad", "sum_sender", "sum_receiver"),
    "squidpy": ("zscore", "count", "cooc_mean", "centrality", "moran", "geary", "ripley", "h5ad"),
    "neighborseq": ("interactions",),
    "deeplinc": ("scores",),
    "spacet": ("cci", "proportions"),
    "mistyr": ("importances", "top"),
    "ncem": ("matrix", "strength"),
    "spaotsc": ("signaling",),
}
#: The most files one bundle holds -- the reader child is handed one descriptor per member. For a per-section set
#: (:func:`bundle_of`) it is the cap PER SECTION: one file per role of each section's run.
MAX_BUNDLE = 8
#: The most sections one per-section set holds; a run with more is opened on its first :data:`MAX_SECTIONS` (in
#: label order) and says so (``Bundle.n_sections_found``).
MAX_SECTIONS = 32
#: The most descriptors one communication read is handed: a full per-section set, plus one domain table to group by.
MAX_FDS = MAX_BUNDLE * MAX_SECTIONS + 1
#: The folder a per-section run writes each section's outputs into (``worker_utils.per_section``'s callers:
#: ``section_<label>/``, the label with every character a path cannot carry replaced by ``_``).
_SECTION_DIR_RE = re.compile(r"^section_(?P<label>[^/\\]{1,64})$")

#: Column names (compared case-insensitively) a long table's sender, receiver, score, p-value and pair columns go by,
#: across the backends' own headers and the common CCC exports. A list is searched in its own order.
_SOURCE_NAMES = (
    "source",
    "sender",
    "ligand_cluster",
    "cell_type_sender",
    "celltype_a",
    "source_celltype",
    "cell_1",
    "type_a",
    "cell_type_1",
    "predictor",
)
_TARGET_NAMES = (
    "target",
    "receiver",
    "receptor_cluster",
    "cell_type_receiver",
    "celltype_b",
    "target_celltype",
    "cell_2",
    "type_b",
    "cell_type_2",
)
_SCORE_NAMES = (
    "score",
    "means",
    "mean",
    "magnitude",
    "lr_means",
    "weight",
    "strength",
    "prob",
    "enrichmentscore",
    "fraction_rho",
    "reference_rho",
    "importance",
)
_P_NAMES = ("p", "pval", "pvalue", "p_value", "padj", "fdr", "fraction_pv", "reference_pv")
_PAIR_NAMES = ("interaction", "ligand_receptor", "lr_pair", "pair")
#: The first header cell of an index-by-columns matrix as pandas and R write it: empty, or pandas' placeholder.
_CORNER = ("", "unnamed: 0")


@dataclass(frozen=True)
class Match:
    """One file name recognised: its backend, its role in that backend's output, and what the name carries (``db``
    for COMMOT, ``mode`` for a squidpy Ripley table)."""

    backend: str
    role: str
    fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Bundle:
    """The files one result is read from. ``members`` are ``(role, record id)``, the anchor first; ``primary`` is the
    anchor's record id; ``fields`` the anchor's (a COMMOT bundle holds one ``db``).

    A **per-section set** (one tool run per section, each section's files in ``section_<label>/``) is one bundle too:
    ``member_sections`` names each member's section (aligned with ``members``), ``sections`` the distinct labels in
    label order, ``n_sections_found`` how many the run had before :data:`MAX_SECTIONS` was applied. For a single
    result ``member_sections`` is all ``None`` and ``sections`` is empty."""

    backend: str
    primary: str
    members: tuple[tuple[str, str], ...]
    fields: dict[str, str] = field(default_factory=dict)
    member_sections: tuple[str | None, ...] = ()
    sections: tuple[str, ...] = ()
    n_sections_found: int = 0

    @property
    def per_section(self) -> bool:
        return bool(self.sections)


def match_name(name: str) -> Match | None:
    """The backend and role of a bare file name, or ``None``. A path is not a name and matches nothing."""
    if not isinstance(name, str) or "/" in name or "\\" in name:
        return None
    for backend, role, pattern in _COMPILED:
        found = pattern.match(name)
        if found:
            return Match(backend, role, {k: v for k, v in found.groupdict().items() if v is not None})
    return None


def roles_of(backend: str) -> tuple[str, ...]:
    """Every role a backend's files can have, in pattern order."""
    return tuple(role for b, role, _ in _PATTERNS if b == backend)


def is_anchor(match: Match) -> bool:
    """Whether a bundle may be opened from this file: a result on its own, not a table that only joins one."""
    return match.role in ANCHOR_ROLES.get(match.backend, ())


def section_of(path: str) -> tuple[str, str] | None:
    """``(label, run folder)`` when a run's file sits in ``<run folder>/section_<label>/``, else ``None``.

    Read from the path the run wrote the file at (a record's ``source_path``), never from what the file holds. The
    run folder -- the folder ABOVE ``section_<label>/`` -- is what keeps two per-section runs of one tool apart when
    they share a ``run_id`` (two output directories in one chat turn): a set holds one run folder's sections only."""
    if not isinstance(path, str) or not path:
        return None
    parts = path.replace("\\", "/").rstrip("/").rsplit("/", 2)
    if len(parts) < 2:
        return None
    found = _SECTION_DIR_RE.match(parts[-2])
    if not found:
        return None
    return found.group("label"), (parts[0] if len(parts) == 3 else "")


def _section_key(section: object) -> tuple[str, str | None] | None:
    """A section as ``(label, run folder)``: :func:`section_of`'s pair, or a bare label (no folder known)."""
    if section is None:
        return None
    if isinstance(section, tuple):
        return str(section[0]), (None if section[1] is None else str(section[1]))
    return str(section), None


def _sibling(entry: tuple) -> tuple[str, str, tuple[str, str | None] | None]:
    """``(record id, name, section)`` of a sibling given as ``(id, name)`` or ``(id, name, section)``."""
    if len(entry) == 3:
        return str(entry[0]), str(entry[1]), _section_key(entry[2])
    return str(entry[0]), str(entry[1]), None


def bundle_of(
    anchor_id: str,
    anchor_name: str,
    siblings: Iterable[tuple],
    *,
    anchor_section: str | tuple[str, str] | None = None,
) -> Bundle | None:
    """The bundle opened from ``anchor_name``, with the siblings that belong to it.

    ``siblings`` are ``(record id, name)`` or ``(record id, name, section)`` -- the section read by
    :func:`section_of` from where the run wrote the file (its ``(label, run folder)``, or a bare label). A sibling belongs when it matches the SAME backend -- and for
    COMMOT the same database, so a CellChat and a CellPhoneDB run in one folder stay two results. One file per role,
    the first given winning (the caller orders siblings newest first), the anchor's own role taken by the anchor; at
    most :data:`MAX_BUNDLE` files. ``None`` when the anchor is not a communication result's name.

    **A per-section set.** When the anchor sits in a ``section_<label>/`` folder (``anchor_section``), the bundle is
    the run's sections: for each section that holds a file of the anchor's role, one file per role (at most
    :data:`MAX_BUNDLE`), at most :data:`MAX_SECTIONS` sections in label order, the anchor's section always kept.
    A file outside any section folder never joins a per-section set, and a file inside one never fills a role of a
    single result: a per-section run's folders are not spare copies of its whole-file outputs. Only the sections of
    the anchor's own run folder join: a second per-section run in the same ``run_id`` is another result, and taking
    its files would pair one run's scores with the other's p-values.
    """
    anchor = match_name(anchor_name)
    if anchor is None or not is_anchor(anchor):
        return None

    def same(match: Match | None) -> bool:
        if match is None or match.backend != anchor.backend:
            return False
        return anchor.backend != "commot" or match.fields.get("db") == anchor.fields.get("db")

    if anchor_section is None:
        members: list[tuple[str, str]] = [(anchor.role, anchor_id)]
        taken = {anchor.role}
        for entry in siblings:
            if len(members) >= MAX_BUNDLE:
                break
            record_id, name, section = _sibling(entry)
            match = match_name(name)
            if section is not None or not same(match) or match.role in taken:
                continue
            members.append((match.role, record_id))
            taken.add(match.role)
        return Bundle(anchor.backend, anchor_id, tuple(members), dict(anchor.fields), (None,) * len(members))

    anchor_label, anchor_folder = _section_key(anchor_section)
    by_section: dict[str, dict[str, str]] = {anchor_label: {anchor.role: anchor_id}}
    for entry in siblings:
        record_id, name, section = _sibling(entry)
        match = match_name(name)
        if section is None or section[1] != anchor_folder or not same(match):
            continue
        roles = by_section.setdefault(section[0], {})
        if match.role not in roles and len(roles) < MAX_BUNDLE:
            roles[match.role] = record_id
    complete = sorted(label for label, roles in by_section.items() if anchor.role in roles)
    kept = [anchor_label, *[label for label in complete if label != anchor_label][: MAX_SECTIONS - 1]]
    kept = sorted(kept)
    members = [(anchor.role, anchor_id)]
    member_sections: list[str | None] = [anchor_label]
    for label in kept:
        for role, record_id in by_section[label].items():
            if record_id == anchor_id:
                continue
            members.append((role, record_id))
            member_sections.append(label)
    return Bundle(
        anchor.backend,
        anchor_id,
        tuple(members),
        dict(anchor.fields),
        tuple(member_sections),
        tuple(kept),
        len(complete),
    )


def backend_of_names(names: Iterable[str]) -> str | None:
    """The backend of the first name that is an anchor, or ``None`` when none is."""
    for name in names:
        match = match_name(name)
        if match is not None and is_anchor(match):
            return match.backend
    return None


_ROLE_NAMES = {
    "source": _SOURCE_NAMES,
    "target": _TARGET_NAMES,
    "score": _SCORE_NAMES,
    "p": _P_NAMES,
    "pair": _PAIR_NAMES,
}


def named_column(header: list[str], role: str) -> str | None:
    """The header cell a long table's ``role`` (source, target, score, p or pair) goes by, or ``None``.

    The first name of that role's list the header holds, compared case-insensitively, returned as the header spells
    it. One role on its own: a table naming a sender and a receiver but no score is still told apart from one naming
    neither, which :func:`table_columns` alone cannot say.
    """
    cells = [str(c) for c in header]
    lowered = [c.strip().lower() for c in cells]
    for wanted in _ROLE_NAMES[role]:
        if wanted in lowered:
            return cells[lowered.index(wanted)]
    return None


def table_columns(header: list[str], name: str = "") -> dict | None:
    """What shape a table's header says it has, and which columns play which part.

    ``{"shape": "long" | "square" | "vector", "source", "target", "score", "pair", "p"}`` (column names or ``None``):

    * **long** -- one row per sender/receiver pair: a source, a target and a score column, by the name lists above;
      ``pair`` and ``p`` when present.
    * **square** -- an index-by-columns matrix: an empty (or pandas ``Unnamed: 0``) first header cell and at least two
      other columns. spaotsc's ``signaling_scores.csv`` (``name``) is one too, though its header is the column numbers
      and it has no row labels: the same header under another name is not vouched for.
    * **vector** -- one label and one value: exactly two columns (the first may be an unnamed index).

    ``None`` when nothing fits, and when the header names a sender and a receiver but no score. From the header alone: whether a column holds numbers is the reader's to find.
    """
    cells = [str(c) for c in header]
    lowered = [c.strip().lower() for c in cells]
    source, target, score = (named_column(cells, role) for role in ("source", "target", "score"))
    if source and target and score:
        return {
            "shape": "long",
            "source": source,
            "target": target,
            "score": score,
            "pair": named_column(cells, "pair"),
            "p": named_column(cells, "p"),
        }
    if source and target:
        # A sender and a receiver with no score is a long table missing its score -- not a matrix, and not a
        # ``source,target`` vector whose receiver is read as the value. Every caller then refuses it as such.
        return None
    empty = {"source": None, "target": None, "score": None, "pair": None, "p": None}
    if len(cells) >= 3 and lowered[0] in _CORNER:
        return {"shape": "square", **empty}
    numbered = len(cells) >= 2 and cells == [str(i) for i in range(len(cells))]
    if numbered and (m := match_name(name)) is not None and (m.backend, m.role) == ("spaotsc", "signaling"):
        return {"shape": "square", **empty}
    if len(cells) == 2:
        return {"shape": "vector", **empty, "source": cells[0], "score": cells[1]}
    return None


__all__ = [
    "ANCHOR_ROLES",
    "BACKENDS",
    "MAX_BUNDLE",
    "MAX_FDS",
    "MAX_SECTIONS",
    "Bundle",
    "Match",
    "backend_of_names",
    "bundle_of",
    "is_anchor",
    "match_name",
    "named_column",
    "roles_of",
    "section_of",
    "table_columns",
]
