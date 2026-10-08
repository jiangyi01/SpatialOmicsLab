"""Spatial reference atlases: the Allen Brain Atlas and HuBMAP.

Both services answer the same shape of question from opposite ends. A spatial-omics result produces
a map -- clusters laid out in tissue coordinates -- and the map is only worth trusting if it agrees
with an independent one built by someone else:

* **Allen Brain Atlas** (``api.brain-map.org``) -- genome-wide in-situ hybridisation across the adult
  mouse brain, quantified per anatomical structure, plus human, macaque and developmental atlases.
  Use it for "is this gene really enriched in *that* structure", where the structure is a named
  brain region and the answer is a number, not a picture. ``StructureUnionize`` is the table that
  matters: for one ISH experiment it carries ``expression_energy`` and ``expression_density``
  aggregated over every annotated structure, which is the closest thing to ground truth for a
  region-level expression claim in mouse brain.
* **HuBMAP** (``*.api.hubmapconsortium.org``) -- the NIH Human BioMolecular Atlas Program's catalogue
  of published human tissue datasets: CODEX, Visium, Xenium, seqFISH, MIBI, GeoMx, single-cell and
  bulk, with donor and organ provenance attached. Use it to find a comparable human dataset to
  reuse or to check a result against, not to look a gene up.

The two are complementary, not interchangeable: Allen gives you a measured value per structure in a
reference space; HuBMAP gives you other people's experiments with enough provenance to judge whether
they are comparable to yours.

Four properties of these two APIs shape every function below. All four were measured, and three of
them produce a confident empty answer rather than an error -- the failure mode this module exists to
prevent.

**Allen's RMA query language is a string, and upstream builds it by interpolation.** A criteria
clause reads ``model::Structure,rma::criteria,[acronym$eq'CA1']``. An apostrophe in the caller's
value ends the literal early and the server answers ``success: false`` with a parser error naming
the column: measured, ``[acronym$eq'Ga'd1']`` returns ``line 1, column 92: mismatched character
<EOF>; expecting "'"``. Every value going into a criteria clause here is validated first, and when
the server does reject a query its own message is surfaced instead of a generic failure string.

**Allen answers an unknown acronym with ``success: true, total_rows: 0``.** Measured:
``[acronym$eq'NotAGene999']`` returns exactly that, which is byte-identical to a gene that exists and
has no data. Every lookup here that comes back empty re-queries with a partial match and returns an
error naming the closest real acronyms.

**An Allen acronym is not unique.** ``CA1`` matches **nine** structures across seven different
structure graphs -- adult mouse (id 382, "Field CA1"), adult human left/right/unsided (graph 10),
macaque (graph 8), and three developing atlases -- and upstream returns all nine with nothing to tell
them apart. Every structure row here carries the atlas and species it belongs to, and both are
filters.

**HuBMAP organ codes are two letters, are laterality-split, and do not mean what they look like.**
``LU`` is **Ureter (Left)**, not lung; lung is ``LL`` and ``RL``. Of the 47 live codes, 11 pairs are
one organ split across two sides, so a single code silently drops half the data -- lung is 769
datasets on the left and 906 on the right. This module takes an organ *name*, resolves it against the
live ontology, and expands it to every code that shares it.

Every outbound call goes through :mod:`spatialomicsgym.utils.http_client`, which enforces HTTPS, an
explicit host allowlist, one shared connection pool, a timeout and bounded retry. No function here
calls ``requests`` directly, and no function here reads an API key -- both services answer
anonymously (measured 2026-09-17).

Reference tables -- Allen's product and structure-graph catalogues, HuBMAP's organ map and assay
vocabulary -- are **read from the live API and cached for the life of the process**, never shipped as
a literal. That is a direct consequence of F28 and F33 below: both defects are a hard-coded table
that was wrong.

Return shape, uniform across the module: ``{"status": "success", "data": ..., ...}`` on success and
``{"status": "error", "error": "<what went wrong and what to do about it>"}`` on failure. Nothing
raises for an expected failure -- a network problem, an outage or a bad argument comes back as a
value, so one failed lookup inside a longer script does not abort the rest of it. The agent loop
recognises ``"status": "error"`` as a failed action, so a failure is never silently read as data.

These functions return a dict; ``print()`` the result (or the part of it you need) or it will not
appear in the observation.

------------------------------------------------------------------------------------------------
Adapted from ToolUniverse -- https://github.com/mims-harvard/ToolUniverse -- at commit
``f075c2a75e8b35ae5dbb220d48d4e87e980388b1``. Copyright [2025] [ToolUniverse team], licensed under
the Apache License, Version 2.0.

CHANGED BY SPATIALOMICSGYM, as Apache-2.0 section 4(b) requires this file to state. The endpoint
knowledge -- which RMA model answers which question, that ``StructureUnionize`` is where quantified
per-structure expression lives, HuBMAP's three API hosts and its Elasticsearch body shape -- is
upstream's; the code is not. Specifically:

* the ``BaseTool``/``register_tool``/config-driven dispatch machinery was not vendored, and each
  upstream tool is re-expressed here as a plain function;
* raw ``requests`` calls were replaced by our HTTP layer, which adds the host allowlist, the HTTPS
  floor and bounded retry that upstream's per-file ``timeout=30`` did not provide;
* every value that reaches a URL path segment is percent-encoded through ``_seg``, as in the other
  five vendored modules. Caller-supplied HuBMAP ids are additionally pinned to an anchored regex, so
  the encoding here is what covers the uuid this module reads back out of a HuBMAP *response* and
  then puts straight into the next request's path. Upstream interpolates both raw;
* ``allen_brain_list_products`` and ``hubmap_list_dataset_types`` are ours, not upstream's. They
  answer the discovery question that upstream answered with a hard-coded list -- wrongly, in both
  cases (F28, F36);
* **F28** -- ``AllenBrain_get_expression_datasets`` advertises ``1=Mouse Brain ISH, 2=Mouse Brain
  microarray``. Read live, ``model::Product`` has **64** products and id 2 is ``HumanMA`` / "Human
  Brain Microarray" -- the Allen *Human* Brain Atlas, a different species and a different assay. A
  model told it is asking for mouse microarray and given human microarray has no way to notice. Here
  the catalogue is read live (42 Mouse, 17 Human, 5 NHP) and every result echoes the resolved
  product's name and species, so a wrong id is visible in the answer rather than inferred from it.
* **F29** -- an unknown gene or structure acronym returns ``success: true, total_rows: 0`` and
  upstream passes the empty list through as a successful result. Measured on
  ``[acronym$eq'NotAGene999']``. Here an empty exact match is retried as a partial match and comes
  back as an error naming the closest real acronyms.
* **F30** -- RMA criteria are built by f-string interpolation of caller input, and upstream collapses
  every ``success: false`` into the string "Allen Brain Atlas query failed". Measured: a single
  apostrophe yields ``line 1, column 92: mismatched character <EOF>; expecting "'"``, which upstream
  discards. Here the characters that can break out of a criteria literal are rejected by name before
  the request, and a server-side rejection surfaces the server's own message.
* **F31** -- ``AllenBrain_search_structures`` returns every graph's match with no atlas context.
  Measured: ``CA1`` returns 9 rows spanning structure graphs 1, 4, 8, 10 (three rows), 13, 16 and 17
  -- mouse, human, macaque and three developmental atlases. Here each row carries its atlas name and
  species, resolved live from ``model::StructureGraph`` and ``model::Ontology``, and both are
  filters.
* **F32** -- ``AllenBrain_get_expression_datasets`` reports ``total_results: len(records)``, the
  length of the current page after QC filtering, while the API's own ``total_rows`` is the real
  total. A caller reading that number believes it has seen everything. Here the page count and the
  API total are separate fields and a truncated result says so.
* **F33** -- ``HuBMAP_search_datasets`` advertises ``'LU' for lung``. Live, ``LU`` is **Ureter
  (Left)**; lung is ``LL``/``RL``. ``LU`` matches no published dataset at all, so an agent asking for
  lung datasets receives an empty result with ``status: success``. Here the organ argument takes a
  name, resolves it against the live ``/organs`` ontology, and rejects an unknown one with the
  closest real organ names.
* **F34** -- 20 of HuBMAP's 47 organ codes are one organ split by laterality, in 10 pairs (Kidney
  LK/RK, Lung LL/RL, Eye LE/RE, Ovary LO/RO, Fallopian Tube LF/RF, Knee LN/RN, Tonsil LT/RT, Ureter
  LU/RU, Mammary Gland ML/MR, Main Bronchus LB/RB). A single code silently drops the other side:
  measured over published datasets, lung is 769 under ``LL`` and 906 under ``RL``, union 1,671 (four
  carry samples from both sides), so naming only ``LL`` misses 902 of 1,671 -- 54% -- while the code
  upstream actually advertises for lung, ``LU``, misses all of them. Here a name expands to every
  code sharing its ``category`` term, and the result says which codes it searched.
* **F35** -- ``hits.total`` is an Elasticsearch **capped estimate**. Measured, the unfiltered dataset
  query returns ``{'relation': 'gte', 'value': 10000}`` while the live published count is 10,688.
  Upstream reads ``.total.value`` and reports 10000 as if it were exact. Here the relation is carried
  through and an estimate is labelled as one.
* **F36** -- of the six ``dataset_type`` examples upstream advertises, two match **zero** datasets:
  ``snATACseq`` (the live vocabulary spells it ``ATACseq``, 901 datasets) and
  ``scRNAseq-10xGenomics-v3``. Because upstream matches the field as analysed full text, ``RNAseq``
  also silently spans three distinct assays -- measured, ``match`` returns 3,008 published datasets
  where the assay actually called ``RNAseq`` has 1,180. Here the query uses the ``.keyword``
  sub-field, so a named assay means that assay and nothing else, ``hubmap_list_dataset_types``
  returns the live vocabulary with counts, and an unmatched value is an error with suggestions
  rather than an empty result.
* **F37** -- ``HuBMAP_get_dataset`` returns each contact's e-mail address. That is third-party
  personal data copied into every agent transcript and log for no research benefit; the published
  record behind the DOI carries it for anyone who needs it. Name and affiliation are kept, the
  address is dropped.
* **F38** -- ``AllenBrain_get_structure_expression_values`` defaults to 50 rows out of ~2,400 per
  dataset (measured: 2,382 for SectionDataSet 480) in the API's own order, so "where is this gene
  expressed" answers with 50 arbitrary structures rather than the 50 with the most signal. Here the
  query carries ``rma::options[order$eq'expression_energy$desc']``, so the *server* ranks the whole
  result set before paging it, and the ordering is named in the result.
* **F39** -- ``HuBMAP_get_dataset_provenance`` calls the lineage "the ordered chain of ancestor
  entities", but ``/ancestors/{uuid}`` returns an unordered bag whose records carry no parent
  pointer (measured: ``immediate_ancestor_ids`` is ``None`` on every one). Upstream sorts it by a
  hand-written rank table, which cannot order two entities of the same kind -- a real lineage
  measured here runs Dataset -> Dataset -> suspension -> section -> block -> block -> Donor's organ
  -> Donor, and the rank table puts the two Datasets and the two blocks in arbitrary order. The
  search index *does* carry ``immediate_ancestor_ids``, so here the chain is rebuilt by following
  it, walked node by node, and labelled ``exact``. When the index does not hold every ancestor the
  result falls back to the rank ordering and says ``approximate`` instead of implying a chain it
  does not have.
* **F40** -- ``HuBMAP_get_dataset`` reports ``organ`` for **every** dataset as ``None``. It reads
  ``origin_samples`` off the entity API, and the entity record has no such field -- that field lives
  in the search index. The default in ``data.get("origin_samples", [{}])`` then yields one empty
  dict, the organ list comes out empty, and the failure is indistinguishable from a dataset with no
  organ recorded. Verified on three published datasets whose real organs are ``RK``, ``PA`` and
  ``PA``. Here the organ is read from the search index and reported with its ontology term.

See ``VENDORING.md`` for the file-by-file record and the re-sync procedure.
"""

from __future__ import annotations

import difflib
import logging
import re
import time
from urllib.parse import quote

from spatialomicsgym.utils.http_client import HttpError, request_json

logger = logging.getLogger(__name__)

_ALLOWED_HOSTS = (
    "api.brain-map.org",
    "search.api.hubmapconsortium.org",
    "ontology.api.hubmapconsortium.org",
    "entity.api.hubmapconsortium.org",
)

_ALLEN_QUERY_URL = "https://api.brain-map.org/api/v2/data/query.json"
_HUBMAP_SEARCH_URL = "https://search.api.hubmapconsortium.org/v3/search"
_HUBMAP_ORGANS_URL = "https://ontology.api.hubmapconsortium.org/organs"
_HUBMAP_ENTITY_URL = "https://entity.api.hubmapconsortium.org"

_DEFAULT_ROWS = 50
_MAX_ROWS = 2500
_HUBMAP_MAX_LIMIT = 50

_WS_RE = re.compile(r"\s+")

# Characters that terminate or redirect an RMA criteria clause. A value carrying any of them is
# rejected rather than escaped: RMA has no documented escape sequence, so there is nothing to escape
# to, and a silently mangled query is worse than a refused one. See F30.
_RMA_UNSAFE = "'\"[](),$:;\\"
_HUBMAP_ID_RE = re.compile(r"^(HBM\d{3}\.[A-Z]{4}\.\d{3}|[0-9a-f]{32})$", re.IGNORECASE)

# Read live and cached for the life of the process. Never a shipped literal -- F28 and F33 are both
# the bug that happens when a reference table is written down instead of looked up.
#
# MED-18 (hunt 2026-09-30, u22-spatial-pipeline-10): the success test was "the call did not fail",
# so one empty upstream answer was cached for the life of the portal process, and every later caller
# -- any account -- was told a real organ or assay "is not a HuBMAP organ/dataset_type" with an
# empty suggestion list. An entry is now stored only through `_remember`, which refuses an empty
# table, and read only through `_cached`, which expires it after `_CACHE_TTL_S`; the dict is capped.
_CACHE: dict[str, object] = {}
_CACHE_STORED_AT: dict[str, float] = {}
_CACHE_TTL_S = 3600.0
_CACHE_MAX_ENTRIES = 32


def _cached(key):
    """The cached value for ``key``, or None when it is absent or older than the TTL."""
    if key not in _CACHE:
        return None
    stored_at = _CACHE_STORED_AT.get(key)
    if stored_at is not None and time.monotonic() - stored_at > _CACHE_TTL_S:
        _CACHE.pop(key, None)
        _CACHE_STORED_AT.pop(key, None)
        return None
    return _CACHE[key]


def _remember(key, value, upstream):
    """Cache a reference table, or return the retryable error an empty one is.

    An empty table is not an answer about the world -- the upstream answered with nothing -- so it is
    reported as a fault naming ``upstream`` and nothing is stored: the next call asks again.
    """
    if not value:
        return _error(
            f"{upstream} answered with an empty table. That is an upstream fault, not an answer; nothing was "
            "cached, so retrying asks again.",
            retryable=True,
        )
    if key not in _CACHE and len(_CACHE) >= _CACHE_MAX_ENTRIES:
        oldest = min(_CACHE, key=lambda k: _CACHE_STORED_AT.get(k, 0.0))
        _CACHE.pop(oldest, None)
        _CACHE_STORED_AT.pop(oldest, None)
    _CACHE[key] = value
    _CACHE_STORED_AT[key] = time.monotonic()
    return None


def _error(message, **extra):
    """An error payload in the shape ``execution.py:_EXEC_ERROR_RE`` recognises as a failed action."""
    payload = {"status": "error", "error": message}
    payload.update(extra)
    return payload


def _ok(data, **extra):
    payload = {"status": "success", "data": data}
    payload.update(extra)
    return payload


def _missing(name, hint, *, plural=False):
    return _error(f"{name} {'are' if plural else 'is'} required. {hint}")


def _clean(text):
    return _WS_RE.sub(" ", str(text or "")).strip()


def _seg(value):
    """One percent-encoded URL path segment.

    ``safe=""`` so a slash inside an identifier stays part of the identifier instead of becoming
    path structure. ``_HUBMAP_ID_RE`` already pins caller-supplied ids to an anchored shape, but
    uuids taken out of an API *response* reach the same position without passing that regex, and an
    upstream host is not a trusted source of path structure. Upstream interpolates all of these raw.
    """
    return quote(str(value), safe="")


def _norm(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")


def _suggest(value, candidates, limit=5):
    """Closest known names for a value that did not match, so a typo is one hop from correct."""
    folded = {}
    for candidate in candidates:
        folded.setdefault(_norm(candidate), candidate)
    close = difflib.get_close_matches(_norm(value), list(folded), n=limit, cutoff=0.55)
    if not close:
        needle = _norm(value)
        close = [k for k in folded if needle and needle in k][:limit]
    return [folded[k] for k in close]


def _as_int(value, default=0):
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rows(size, default=_DEFAULT_ROWS, maximum=_MAX_ROWS):
    try:
        return max(1, min(int(size), maximum))
    except (TypeError, ValueError):
        return default


def _fetch_json(url, **kwargs):
    """``(payload, None)`` on success, ``(None, error_payload)`` on failure."""
    try:
        return request_json(url, allowed_hosts=_ALLOWED_HOSTS, **kwargs), None
    except HttpError as exc:
        return None, _error(exc.detail, retryable=bool(exc.status) and exc.status >= 500)


# ------------------------------------------------------------------------------------- Allen Brain


def _rma_literal(value, field):
    """``(cleaned, None)`` when ``value`` is safe inside an RMA criteria clause, else ``(None, error)``.

    F30. RMA criteria are a query language passed as one URL parameter, and upstream builds them by
    f-string interpolation of caller input. There is no documented escape sequence, so the only
    correct handling of a value containing a delimiter is to refuse it and say which character was
    the problem -- an apostrophe silently truncates the literal and the server answers with a parser
    error about a column the caller never mentioned.
    """
    text = _clean(value)
    if not text:
        return None, None
    bad = sorted({character for character in text if character in _RMA_UNSAFE})
    if bad:
        return None, _error(
            f"{field} cannot contain {' '.join(repr(character) for character in bad)} -- those "
            f"characters are delimiters in the Allen Brain query language and there is no way to "
            f"escape them. Pass the plain acronym or name, e.g. 'Gad1' or 'hippocampus'."
        )
    return text, None


def _rma(criteria, *, num_rows=_DEFAULT_ROWS, include=None, start_row=0, order=None):
    """``(payload, None)`` or ``(None, error)`` for one RMA query.

    Two failure shapes are folded in here so no caller has to know about them. ``success: false``
    carries the server's own parser message in ``msg`` as a *string* (not a list), and upstream
    discards it (F30). A transport failure comes back through ``_fetch_json``.

    ``order`` is ``(column, "asc"|"desc")`` and appends ``rma::options[order$eq'<column>$<dir>']``,
    which makes the *server* rank the whole result set before paging it. That is the difference
    between "the highest-expressing structures" and "the first fifty rows in storage order" (F38).
    """
    if order:
        criteria = f"{criteria},rma::options[order$eq'{order[0]}${order[1]}']"
    params = {"criteria": criteria, "num_rows": num_rows, "start_row": start_row}
    if include:
        params["include"] = include
    payload, err = _fetch_json(_ALLEN_QUERY_URL, params=params)
    if err:
        return None, err
    if not payload.get("success"):
        detail = payload.get("msg")
        detail = _clean(detail if isinstance(detail, str) else "no message")
        return None, _error(f"The Allen Brain Atlas rejected the query: {detail}")
    return payload, None


def _allen_table(key, criteria, fields):
    """One live reference table, projected to ``fields`` and cached (see ``_remember``)."""
    rows = _cached(key)
    if rows is not None:
        return rows, None
    payload, err = _rma(criteria, num_rows="all")
    if err:
        return None, err
    rows = [{name: row.get(name) for name in fields} for row in payload.get("msg") or []]
    err = _remember(key, rows, f"The Allen Brain Atlas ({criteria})")
    if err:
        return None, err
    return rows, None


def _allen_atlas_context():
    """``{graph_id: {"atlas": name, "species": name, "ontology_id": id}}``, read live.

    F31. A ``Structure`` row carries ``graph_id`` and ``ontology_id`` and nothing else to say which
    atlas it belongs to. ``model::StructureGraph`` names the atlas, ``model::Ontology`` links it to an
    organism, and ``model::Organism`` names the species. Three joins, done once per process, turn
    nine indistinguishable ``CA1`` rows into nine labelled ones.
    """
    context = _cached("atlas_context")
    if context is not None:
        return context, None
    graphs, err = _allen_table(
        "structure_graphs", "model::StructureGraph", ("id", "name", "ontology_id", "root_structure_id")
    )
    if err:
        return None, err
    ontologies, err = _allen_table(
        "ontologies", "model::Ontology", ("id", "abbreviation", "name", "organism_id", "has_atlas")
    )
    if err:
        return None, err
    organisms, err = _allen_table("organisms", "model::Organism", ("id", "name", "ncbi_taxonomy_id"))
    if err:
        return None, err
    species_by_id = {row["id"]: row["name"] for row in organisms}
    species_by_ontology = {row["id"]: species_by_id.get(row["organism_id"]) for row in ontologies}
    context = {
        row["id"]: {
            "atlas": row["name"],
            "species": species_by_ontology.get(row["ontology_id"]),
            "ontology_id": row["ontology_id"],
        }
        for row in graphs
    }
    err = _remember("atlas_context", context, "The Allen Brain Atlas (model::StructureGraph)")
    if err:
        return None, err
    return context, None


def _allen_products():
    """The live product catalogue, keyed by id. See F28 for why this is not a two-entry literal."""
    by_id = _cached("products_by_id")
    if by_id is not None:
        return by_id, None
    rows, err = _allen_table(
        "products", "model::Product", ("id", "abbreviation", "name", "species", "resource", "description")
    )
    if err:
        return None, err
    by_id = {row["id"]: row for row in rows}
    err = _remember("products_by_id", by_id, "The Allen Brain Atlas (model::Product)")
    if err:
        return None, err
    return by_id, None


# Internal bookkeeping columns every Allen record carries: a Sphinx search-index row id, a facet
# hash, and the absolute path of the image series inside Allen's own storage. None of it answers a
# biological question, and ``storage_directory`` puts a filesystem path into the transcript, so the
# projections below drop them rather than forwarding the raw record the way upstream does.
_ALLEN_NOISE = (
    "sphinx_id",
    "failed_facet",
    "structure_name_facet",
    "product_name_facet",
    "species_name_facet",
    "storage_directory",
    "neuro_name_structure_id_path",
)


def _allen_gene_row(row, species_by_id):
    return {
        "id": row.get("id"),
        "acronym": row.get("acronym"),
        "name": row.get("name"),
        "species": species_by_id.get(row.get("organism_id")),
        "entrez_id": row.get("entrez_id"),
        "ensembl_id": row.get("ensembl_id") or row.get("legacy_ensembl_gene_id"),
        "homologene_id": row.get("homologene_id"),
        "aliases": [alias for alias in _clean(row.get("alias_tags")).split(" ") if alias],
    }


def _allen_structure_row(row, context):
    """One ``Structure`` record with the atlas context F31 says it needs to be interpretable."""
    graph = context.get(row.get("graph_id")) or {}
    path = [part for part in str(row.get("structure_id_path") or "").split("/") if part]
    return {
        "id": row.get("id"),
        "acronym": row.get("acronym"),
        "name": row.get("name"),
        "atlas": graph.get("atlas"),
        "species": graph.get("species"),
        "graph_id": row.get("graph_id"),
        "depth": row.get("depth"),
        "parent_structure_id": row.get("parent_structure_id"),
        "ancestor_structure_ids": [_as_int(part) for part in path],
        "color_hex_triplet": row.get("color_hex_triplet"),
    }


_ALLEN_STRUCTURE_FIELDS = (
    "id",
    "acronym",
    "name",
    "graph_id",
    "ontology_id",
    "depth",
    "parent_structure_id",
    "structure_id_path",
    "color_hex_triplet",
    "hemisphere_id",
    "st_level",
)


#: Common species names, as Allen's product table spells them, to the binomial its organism table uses.
_COMMON_SPECIES = {
    "mouse": "Mus musculus",
    "human": "Homo Sapiens",
    "macaque": "Macaca mulatta",
    "rat": "Rattus norvegicus",
    "monkey": "Macaca mulatta",
}


def _allen_gene_suggestions(acronym, species_label=None):
    """The sentence that turns an empty Allen gene result into an actionable one (F29).

    Two different failures look identical in the response -- the acronym does not exist anywhere in
    Allen, and the acronym exists but the product being queried never measured it. One extra request
    tells them apart, and the advice differs: fix the spelling, or change the product. ``(hint,
    error)``; the hint is always a usable sentence even when the extra request fails.
    """
    exact, err = _rma(f"model::Gene,rma::criteria,[acronym$eq'{acronym}']", num_rows=25, include="organism")
    hits = [] if err else (exact.get("msg") or [])
    if hits:
        found = sorted(
            {(row.get("organism") or {}).get("name") for row in hits if isinstance(row.get("organism"), dict)} - {None}
        )
        carried = f"Allen carries it for {', '.join(found)}. " if found else ""
        # The product's species is a common name ('Mouse') and the organisms are binomials ('Mus
        # musculus'), and the two were never compared: "none of which is Mouse" was asserted about a
        # mouse gene on a mouse product, sending the caller to switch species for nothing (hunt
        # 2026-09-30, u22-spatial-pipeline-22). Compared through the same name map the species
        # filter uses; a label it cannot map is reported, not judged.
        names = {_norm(name) for name in found}
        binomial = _COMMON_SPECIES.get(_norm(species_label)) if species_label else None
        if species_label and _norm(species_label) in names:
            binomial = species_label  # already spelled the way the organism table spells it
        if binomial and _norm(binomial) in names:
            return (
                f"The acronym itself is real -- {carried}That is this product's species ({species_label}), so "
                f"the gene is not the problem: this product has no experiment for it. Allen's products assay "
                f"different gene sets -- run allen_brain_list_products() for the other {species_label} products.",
                None,
            )
        if species_label and found and not binomial:
            return (
                f"The acronym itself is real -- {carried}The species here is listed as '{species_label}', which "
                "this check cannot map to one of those organism names, so it does not say whether they match. If "
                "they do not, this is a species or product mismatch, not a spelling one: run "
                "allen_brain_list_products() and pick a product whose species carries the gene.",
                None,
            )
        mismatch = ""
        if binomial and found:
            named = species_label if _norm(binomial) == _norm(species_label) else f"{species_label} ({binomial})"
            mismatch = f"none of which is {named}. "
        return (
            f"The acronym itself is real -- {carried}{mismatch}"
            f"So this is a species or product mismatch, not a spelling one: an Allen product is "
            f"tied to one organism, and the gene record has to belong to that organism too. Run "
            f"allen_brain_list_products() and pick one whose species matches.",
            None,
        )
    near, near_err = _rma(f"model::Gene,rma::criteria,[acronym$li'*{acronym}*']", num_rows=25)
    near_rows = [] if near_err else (near.get("msg") or [])
    close = sorted({row.get("acronym") for row in near_rows if row.get("acronym")})
    if close:
        return f"Closest acronyms that do exist: {', '.join(close[:8])}.", None
    return (
        "Allen uses the source organism's capitalisation -- 'Gad1' in mouse, 'GAD1' in human -- so "
        "check the case, or search by gene name with allen_brain_search_genes(name_contains=...).",
        None,
    )


def _allen_species_filter(species):
    """``(organism_id, None)`` for a species name, or ``(None, error)`` naming the real ones.

    Allen spells them as binomials with its own capitalisation -- ``Homo Sapiens``, ``Mus
    musculus``, ``Macaca mulatta`` -- and carries 44 organisms, most of them cloning vectors and
    phages from the probe records. Matching is fold-insensitive so 'mouse' does not have to be
    spelled 'Mus musculus', but an unrecognised value is an error rather than an empty result.
    """
    wanted = _clean(species)
    if not wanted:
        return None, None
    organisms, err = _allen_table("organisms", "model::Organism", ("id", "name", "ncbi_taxonomy_id"))
    if err:
        return None, err
    wanted = _COMMON_SPECIES.get(_norm(wanted), wanted)
    for row in organisms:
        if _norm(row["name"]) == _norm(wanted):
            return row["id"], None
    names = [row["name"] for row in organisms]
    close = _suggest(wanted, names)
    return None, _error(
        f"'{species}' is not a species the Allen Brain Atlas indexes. "
        + (f"Closest: {', '.join(close)}." if close else "")
        + " The three with brain atlases are 'Mus musculus', 'Homo Sapiens' and 'Macaca mulatta'."
    )


def allen_brain_list_products(species=None):
    """List the Allen Brain Atlas data products, read live from the API.

    A "product" is one Allen study -- the adult mouse ISH atlas, the human microarray atlas, a
    developmental series, a connectivity project. Its id is what
    :func:`allen_brain_get_expression_datasets` filters on, and getting it wrong changes the species
    without any error: id 1 is the adult *mouse* ISH atlas and id 2 is the adult *human* microarray
    atlas, which upstream's tool description labels "Mouse Brain microarray" (F28).

    This function is ours, not upstream's. It exists so the product id is looked up rather than
    remembered.

    Parameters
    ----------
    species : str, optional
        Restrict to one species -- ``"Mouse"``, ``"Human"`` or ``"NHP"`` as Allen spells them on the
        product record. Default ``None``, every product.

    Returns
    -------
    dict
        ``data.products``, each with ``id``, ``abbreviation``, ``name``, ``species``, ``resource``
        and ``description``; ``data.by_species`` counts them.

    Examples
    --------
    >>> print(allen_brain_list_products(species="Mouse"))  # doctest: +SKIP
    """
    products, err = _allen_products()
    if err:
        return err
    rows = list(products.values())
    known = sorted({_clean(row.get("species")) for row in rows if row.get("species")})
    wanted = _clean(species)
    if wanted:
        matched = [row for row in rows if _norm(row.get("species")) == _norm(wanted)]
        if not matched:
            return _error(f"'{species}' is not a species on any Allen product. Known: {', '.join(known)}.")
        rows = matched
    counts: dict[str, int] = {}
    for row in rows:
        counts[_clean(row.get("species")) or "unspecified"] = (
            counts.get(_clean(row.get("species")) or "unspecified", 0) + 1
        )
    return _ok(
        {
            "products": sorted(rows, key=lambda row: row["id"]),
            "total": len(rows),
            "by_species": counts,
            "species_filter": wanted or "all",
        },
        note="Product ids are stable; the catalogue is read live because it grows.",
    )


def allen_brain_search_genes(acronym=None, name_contains=None, species=None, limit=_DEFAULT_ROWS):
    """Find genes in the Allen Brain Atlas by acronym or by a fragment of their name.

    Allen's gene ids are its own, not Ensembl's, and they are what every other Allen query keys on --
    so this is usually the first call in an Allen chain. Acronyms follow the source organism's
    convention: ``Gad1`` in mouse, ``GAD1`` in human.

    Parameters
    ----------
    acronym : str, optional
        Exact acronym match, e.g. ``"Gad1"``. One of ``acronym`` or ``name_contains`` is required.
    name_contains : str, optional
        Substring of the gene's full name, e.g. ``"glutamate decarboxylase"``.
    species : str, optional
        Restrict to one species. ``"mouse"``, ``"human"`` and ``"macaque"`` are understood as well
        as Allen's own binomials. Default ``None``, every species.
    limit : int, optional
        Maximum rows. Default 50.

    Returns
    -------
    dict
        ``data.genes``, each with the Allen ``id`` that other Allen calls need, plus the species,
        Entrez and HomoloGene ids and the alias list.

    Examples
    --------
    >>> print(allen_brain_search_genes(acronym="Gad1", species="mouse"))  # doctest: +SKIP
    """
    exact, err = _rma_literal(acronym, "acronym")
    if err:
        return err
    partial, err = _rma_literal(name_contains, "name_contains")
    if err:
        return err
    if not exact and not partial:
        return _missing(
            "acronym or name_contains",
            "Pass an exact acronym (e.g. 'Gad1') or a fragment of the gene's name (e.g. 'glutamate decarboxylase').",
        )
    organism_id, err = _allen_species_filter(species)
    if err:
        return err
    organisms, err = _allen_table("organisms", "model::Organism", ("id", "name", "ncbi_taxonomy_id"))
    if err:
        return err
    species_by_id = {row["id"]: row["name"] for row in organisms}

    clauses = [f"[acronym$eq'{exact}']"] if exact else [f"[name$li'*{partial}*']"]
    if organism_id is not None:
        clauses.append(f"[organism_id$eq{organism_id}]")
    payload, err = _rma(f"model::Gene,rma::criteria,{','.join(clauses)}", num_rows=_rows(limit))
    if err:
        return err
    rows = payload.get("msg") or []

    if not rows and exact:
        # F29. An unknown acronym is indistinguishable from a real one with no rows, so spend one
        # more request turning the empty answer into a list of acronyms that do exist.
        hint, _unused = _allen_gene_suggestions(exact, species)
        return _error(
            f"No Allen Brain Atlas gene has the acronym '{exact}'" + (f" in {species}" if species else "") + ". " + hint
        )
    if not rows:
        return _error(
            f"No Allen Brain Atlas gene has a name containing '{partial}'"
            + (f" in {species}" if species else "")
            + ". Try a shorter fragment, or search by acronym."
        )
    return _ok(
        {
            "query": {"acronym": exact, "name_contains": partial, "species": species or "any"},
            "total_matched": payload.get("total_rows"),
            "returned": len(rows),
            "genes": [_allen_gene_row(row, species_by_id) for row in rows],
        }
    )


def allen_brain_search_structures(acronym=None, name_contains=None, atlas=None, species=None, limit=_DEFAULT_ROWS):
    """Find brain structures by acronym or name, with the atlas each one belongs to.

    Allen's 13,709 structures span seventeen structure graphs -- adult mouse, adult human, macaque,
    mouse and human spinal cord, four developmental series, and several sampling ontologies -- and an
    acronym is unique *within* a graph, not across them. ``CA1`` matches nine structures in seven
    graphs (F31), so the atlas is part of the answer, not decoration: structure id 382 is mouse Field
    CA1 and id 4254 is human "CA1 field, left", and only the atlas name distinguishes them.

    The structure ``id`` returned here is what :func:`allen_brain_get_structure` and the
    per-structure expression table are keyed on.

    Parameters
    ----------
    acronym : str, optional
        Exact acronym match, e.g. ``"CA1"``, ``"VISp"``, ``"HIP"``. One of ``acronym`` or
        ``name_contains`` is required.
    name_contains : str, optional
        Substring of the structure's name, e.g. ``"hippocampus"``.
    atlas : str, optional
        Keep only structures whose atlas name contains this, e.g. ``"Mouse Brain Atlas"``. Call
        without it first to see which atlases matched.
    species : str, optional
        Keep only structures from atlases of this species -- ``"mouse"``, ``"human"``, ``"macaque"``.
    limit : int, optional
        Maximum rows *fetched*; filtering by atlas or species happens after. Default 50.

    Returns
    -------
    dict
        ``data.structures`` with ``id``, ``acronym``, ``name``, ``atlas``, ``species``, the parent id
        and the full ancestor id path; ``data.atlases_matched`` summarises the spread.

    Examples
    --------
    >>> print(allen_brain_search_structures(acronym="CA1", species="mouse"))  # doctest: +SKIP
    """
    exact, err = _rma_literal(acronym, "acronym")
    if err:
        return err
    partial, err = _rma_literal(name_contains, "name_contains")
    if err:
        return err
    if not exact and not partial:
        return _missing(
            "acronym or name_contains",
            "Pass an exact acronym (e.g. 'CA1') or a fragment of the structure's name (e.g. 'hippocampus').",
        )
    context, err = _allen_atlas_context()
    if err:
        return err
    criteria = (
        f"model::Structure,rma::criteria,[acronym$eq'{exact}']"
        if exact
        else f"model::Structure,rma::criteria,[name$li'*{partial}*']"
    )
    payload, err = _rma(criteria, num_rows=_rows(limit))
    if err:
        return err
    rows = [_allen_structure_row(row, context) for row in payload.get("msg") or []]

    if not rows and exact:
        near, near_err = _rma(f"model::Structure,rma::criteria,[acronym$li'*{exact}*']", num_rows=25)
        close = [row.get("acronym") for row in (near.get("msg") or [])] if not near_err else []
        return _error(
            f"No Allen Brain Atlas structure has the acronym '{exact}'. "
            + (
                f"Closest acronyms that do exist: {', '.join(sorted(set(close))[:8])}."
                if close
                else "Acronyms are atlas-specific and case-sensitive; search by name_contains instead."
            )
        )
    if not rows:
        return _error(
            f"No Allen Brain Atlas structure has a name containing '{partial}'. Try a shorter "
            "fragment -- names are spelled per atlas, e.g. 'Field CA1' in mouse and 'CA1 field' in "
            "human."
        )

    before = len(rows)
    if _clean(atlas):
        rows = [row for row in rows if _norm(atlas) in _norm(row.get("atlas"))]
    if _clean(species):
        organism_id, species_err = _allen_species_filter(species)
        if species_err:
            return species_err
        organisms, organism_err = _allen_table("organisms", "model::Organism", ("id", "name", "ncbi_taxonomy_id"))
        if organism_err:
            return organism_err
        wanted = {row["name"] for row in organisms if row["id"] == organism_id}
        rows = [row for row in rows if row.get("species") in wanted]
    if not rows:
        return _error(
            f"{before} structure(s) matched, but none in "
            f"{'atlas ' + repr(atlas) if _clean(atlas) else ''}"
            f"{' and ' if _clean(atlas) and _clean(species) else ''}"
            f"{'species ' + repr(species) if _clean(species) else ''}. "
            "Call again without the filter to see which atlases carry this acronym."
        )
    spread: dict[str, int] = {}
    for row in rows:
        label = f"{row.get('atlas')} ({row.get('species')})"
        spread[label] = spread.get(label, 0) + 1
    return _ok(
        {
            "query": {"acronym": exact, "name_contains": partial, "atlas": atlas, "species": species},
            "total_matched": payload.get("total_rows"),
            "returned": len(rows),
            "atlases_matched": spread,
            "structures": rows,
        },
        note=(
            "An acronym is unique within one structure graph, not across atlases -- check the "
            "'atlas' field before using an id."
        ),
    )


def allen_brain_get_structure(structure_id):
    """Look one brain structure up by its Allen id, with its atlas and its ancestor chain.

    Parameters
    ----------
    structure_id : int
        Allen structure id, e.g. ``382`` (mouse Field CA1), ``375`` (mouse Hippocampal region),
        ``315`` (mouse Isocortex), ``997`` (mouse root). Ids are atlas-specific --
        :func:`allen_brain_search_structures` reports which atlas an id came from.

    Returns
    -------
    dict
        ``data.structure``, plus ``data.ancestors`` naming every structure on the path to the root,
        which is what turns a bare id into an anatomical location.

    Examples
    --------
    >>> print(allen_brain_get_structure(382))  # doctest: +SKIP
    """
    identifier = _as_int(structure_id, default=-1)
    if identifier < 0:
        return _error(
            f"structure_id must be an integer Allen structure id, not {structure_id!r}. "
            "allen_brain_search_structures(acronym=...) returns one."
        )
    context, err = _allen_atlas_context()
    if err:
        return err
    payload, err = _rma(f"model::Structure,rma::criteria,[id$eq{identifier}]", num_rows=1)
    if err:
        return err
    rows = payload.get("msg") or []
    if not rows:
        return _error(
            f"No Allen Brain Atlas structure has id {identifier}. Ids are per-atlas integers; "
            "allen_brain_search_structures(acronym='CA1') lists the id in each atlas."
        )
    structure = _allen_structure_row(rows[0], context)
    ancestors = []
    path = [part for part in structure.get("ancestor_structure_ids") or [] if part != identifier]
    if path:
        joined = ",".join(str(part) for part in path)
        ancestor_payload, ancestor_err = _rma(f"model::Structure,rma::criteria,[id$in{joined}]", num_rows=len(path))
        if not ancestor_err:
            by_id = {row.get("id"): row for row in ancestor_payload.get("msg") or []}
            ancestors = [
                {
                    "id": part,
                    "acronym": (by_id.get(part) or {}).get("acronym"),
                    "name": (by_id.get(part) or {}).get("name"),
                }
                for part in path
            ]
    return _ok(
        {
            "structure": structure,
            "ancestors": ancestors,
            "path_root_to_structure": " > ".join(
                [entry["acronym"] or str(entry["id"]) for entry in ancestors]
                + [structure.get("acronym") or str(identifier)]
            ),
        }
    )


_ALLEN_UNIONIZE_MEASURES = (
    "expression_energy",
    "expression_density",
    "sum_expressing_pixels",
    "sum_expressing_pixel_intensity",
    "sum_pixels",
    "sum_pixel_intensity",
    "voxel_energy_mean",
    "voxel_energy_cv",
)


def allen_brain_get_expression_datasets(gene_acronym, product_id=1, limit=_DEFAULT_ROWS, include_failed=False):
    """List the Allen in-situ hybridisation experiments that measured one gene.

    Each row is a ``SectionDataSet`` -- one gene assayed across one brain, sectioned in one plane.
    Its ``id`` is what ``allen_brain_get_structure_expression_values`` needs to read the actual
    numbers out, so this is the discovery step in front of the quantitative one.

    Two things this reports that the raw endpoint does not. The **product** is echoed back by name
    and species, because ``product_id`` silently decides which organism you are asking about and the
    upstream documentation for it is wrong (F28): id 2 is the *Human* Brain Microarray, not a mouse
    product, so a caller who passes ``product_id=2`` with a mouse-cased symbol gets an empty answer
    and no explanation. Run ``allen_brain_list_products()`` for the live catalogue of 64. And the
    **count** is split into ``experiments_returned`` (this page) and ``experiments_total`` (the whole
    result set) rather than reporting the page length as if it were the total (F32).

    Gene symbols are case-sensitive per species as Allen stores them: ``Gad1`` is the mouse record
    and ``GAD1`` the human one. An acronym that matches nothing comes back as an error naming close
    symbols, not as an empty success (F29).

    Parameters
    ----------
    gene_acronym : str
        Gene symbol as Allen spells it, e.g. ``'Gad1'``, ``'Pvalb'``, ``'Sst'``, ``'Slc17a7'``.
    product_id : int, optional
        Which Allen product to search. 1 = Allen Mouse Brain Atlas ISH (default),
        2 = Allen Human Brain Microarray. ``allen_brain_list_products()`` lists all 64.
    limit : int, optional
        Experiments to return, 1-2500. Default 50.
    include_failed : bool, optional
        Keep experiments Allen flagged ``failed`` during QC. Default False, which is what you want
        unless you are auditing the pipeline itself.

    Returns
    -------
    dict
        ``{"status": "success", "data": {"gene": ..., "product": {...}, "experiments_returned": int,
        "experiments_total": int, "experiments": [...]}}`` or ``{"status": "error", ...}``.

    Examples
    --------
    >>> out = allen_brain_get_expression_datasets("Gad1")
    >>> out["data"]["experiments_total"]
    22
    >>> out["data"]["product"]["name"]
    'Mouse Brain'
    """
    acronym, err = _rma_literal(gene_acronym, "gene_acronym")
    if err:
        return err
    if not acronym:
        return _missing("gene_acronym", "Name the gene whose experiments you want, e.g. 'Gad1'.")
    product = _as_int(product_id, default=-1)
    if product < 0:
        return _error(f"product_id must be a whole number, not {product_id!r}. 1 is the Allen Mouse Brain Atlas.")

    products, err = _allen_products()
    if err:
        return err
    record = products.get(product)
    if record is None:
        known = ", ".join(f"{pid} ({products[pid]['abbreviation']})" for pid in sorted(products)[:8])
        return _error(
            f"{product} is not an Allen product id. The catalogue holds {len(products)}; the first few "
            f"are {known}. Run allen_brain_list_products() to see them with their species."
        )

    rows = _rows(limit)
    payload, err = _rma(
        f"model::SectionDataSet,rma::criteria,genes[acronym$eq'{acronym}'],products[id$eq{product}]",
        num_rows=rows,
        include="genes",
    )
    if err:
        return err
    records = payload.get("msg") or []
    if not records:
        hint, _ = _allen_gene_suggestions(acronym, record.get("species"))
        return _error(
            f"No {record['name']} experiment measured '{gene_acronym}'. " + hint,
            gene=gene_acronym,
            product={"id": product, "name": record.get("name"), "species": record.get("species")},
        )

    experiments = []
    for row in records:
        if row.get("failed") and not include_failed:
            continue
        genes = [_allen_gene_row(gene, {}) for gene in row.get("genes") or []]
        experiments.append(
            {
                "section_data_set_id": row.get("id"),
                "plane_of_section_id": row.get("plane_of_section_id"),
                "section_thickness_um": row.get("section_thickness"),
                "reference_space_id": row.get("reference_space_id"),
                "specimen_id": row.get("specimen_id"),
                "failed": row.get("failed"),
                "qc_date": row.get("qc_date"),
                "genes": [{key: gene[key] for key in ("id", "acronym", "name", "entrez_id")} for gene in genes],
            }
        )
    return _ok(
        {
            "gene": gene_acronym,
            "product": {
                "id": product,
                "abbreviation": record.get("abbreviation"),
                "name": record.get("name"),
                "species": record.get("species"),
            },
            "experiments_returned": len(experiments),
            "experiments_total": payload.get("total_rows"),
            "dropped_failed_qc": len(records) - len(experiments),
            "experiments": experiments,
            "next_step": "Pass a section_data_set_id to allen_brain_get_structure_expression_values() "
            "for the per-structure numbers.",
        }
    )


def allen_brain_get_structure_expression_values(
    section_data_set_id, limit=_DEFAULT_ROWS, include_structure=True, sort_by="expression_energy"
):
    """Read the per-structure expression numbers out of one Allen ISH experiment, ranked.

    A ``SectionDataSet`` is aggregated into one ``StructureUnionize`` row per anatomical structure --
    around 2,400 of them for a mouse experiment, one for every node of the reference ontology from
    ``root`` down to individual cortical layers. Each row carries ``expression_energy`` (intensity x
    density, the headline measure), ``expression_density`` (fraction of expressing pixels), the raw
    pixel sums behind both, and the across-voxel mean and coefficient of variation.

    **The ranking happens on the server, before paging** (F38). Asking for fifty of 2,382 rows in
    storage order answers "here are fifty structures" when the question was "where is this gene
    expressed"; the first row that way is whatever the pipeline wrote first. Sorted by
    ``expression_energy`` the same query answers the question that was asked.

    Note the hierarchy is fully present, so a parent structure and its layers both appear and their
    values are not independent. ``root`` is the whole brain.

    Parameters
    ----------
    section_data_set_id : int
        Experiment id from ``allen_brain_get_expression_datasets``, e.g. 480 (mouse Gad1).
    limit : int, optional
        Structures to return after ranking, 1-2500. Default 50.
    include_structure : bool, optional
        Attach each structure's acronym, name, depth and ontology path. Default True.
    sort_by : str, optional
        One of expression_energy (default), expression_density, sum_expressing_pixels,
        sum_expressing_pixel_intensity, sum_pixels, sum_pixel_intensity, voxel_energy_mean,
        voxel_energy_cv, or ``'none'`` for the API's own order. Always descending.

    Returns
    -------
    dict
        ``{"status": "success", "data": {"section_data_set_id": ..., "sorted_by": ...,
        "structures_returned": int, "structures_total": int, "values": [...]}}`` or an error dict.

    Examples
    --------
    >>> out = allen_brain_get_structure_expression_values(480, limit=5)
    >>> out["data"]["structures_total"]
    2382
    >>> out["data"]["values"][0]["structure"]["acronym"]
    'AOVE'
    """
    dataset = _as_int(section_data_set_id, default=-1)
    if dataset <= 0:
        return _error(
            f"section_data_set_id must be a positive experiment id, not {section_data_set_id!r}. "
            f"allen_brain_get_expression_datasets('Gad1') returns some."
        )
    column = _norm(sort_by) or "expression_energy"
    if column not in _ALLEN_UNIONIZE_MEASURES and column != "none":
        close = _suggest(column, list(_ALLEN_UNIONIZE_MEASURES))
        return _error(
            f"'{sort_by}' is not a StructureUnionize measure. "
            + (f"Closest: {', '.join(close)}. " if close else "")
            + f"Allen records {', '.join(_ALLEN_UNIONIZE_MEASURES)}, or pass 'none' for API order."
        )

    rows = _rows(limit)
    payload, err = _rma(
        f"model::StructureUnionize,rma::criteria,[section_data_set_id$eq{dataset}]",
        num_rows=rows,
        include="structure" if include_structure else None,
        order=None if column == "none" else (column, "desc"),
    )
    if err:
        return err
    records = payload.get("msg") or []
    if not records:
        return _error(
            f"Allen has no per-structure expression for SectionDataSet {dataset}. Either the id is "
            f"not an experiment, or it is one whose images were never unionized -- microarray and "
            f"failed-QC experiments have no StructureUnionize rows. Check the id against "
            f"allen_brain_get_expression_datasets()."
        )

    context, err = _allen_atlas_context() if include_structure else (None, None)
    if err:
        return err
    values = []
    for row in records:
        entry = {measure: row.get(measure) for measure in _ALLEN_UNIONIZE_MEASURES}
        entry["structure_id"] = row.get("structure_id")
        if include_structure and isinstance(row.get("structure"), dict):
            entry["structure"] = _allen_structure_row(row["structure"], context)
        values.append(entry)
    return _ok(
        {
            "section_data_set_id": dataset,
            "sorted_by": None if column == "none" else f"{column} (descending)",
            "structures_returned": len(values),
            "structures_total": payload.get("total_rows"),
            "measures": list(_ALLEN_UNIONIZE_MEASURES),
            "values": values,
            "note": "Structures nest: a parent and its layers both appear, so values are not independent. "
            "'root' is the whole brain.",
        }
    )


# ------------------------------------------------------------------------------------------ HuBMAP

# Fields pulled back for a dataset hit. Everything omitted is either bookkeeping (index_version,
# *_sub) or personal data the search index carries and a research answer does not need
# (created_by_user_email, last_modified_user_email) -- see F37.
_HUBMAP_HIT_SOURCE = (
    "hubmap_id",
    "uuid",
    "title",
    "dataset_type",
    "status",
    "group_name",
    "origin_samples.organ",
    "anatomy_0",
    "anatomy_1",
    "doi_url",
    "created_timestamp",
    "donor.mapped_metadata.sex",
    "donor.mapped_metadata.age_value",
)

_HUBMAP_LINEAGE_SOURCE = (
    "uuid",
    "hubmap_id",
    "entity_type",
    "sample_category",
    "organ",
    "dataset_type",
    "group_name",
    "immediate_ancestor_ids",
    "created_timestamp",
)


def _hubmap_search(must, *, size=0, source=None, aggs=None):
    """One search POST. ``(hits, total, error)`` where ``total`` is ``(relation, value)`` (F35)."""
    body = {"size": size, "query": {"bool": {"must": must}}}
    if source:
        body["_source"] = list(source)
    if aggs:
        body["aggs"] = aggs
    payload, err = _fetch_json(_HUBMAP_SEARCH_URL, method="POST", json_body=body)
    if err:
        return None, None, err
    return payload, _hubmap_total(payload), None


def _hubmap_total(payload):
    """``{"value": int, "relation": "exact"|"at least"}``.

    F35. Elasticsearch stops counting at 10,000 and reports ``{'relation': 'gte', 'value': 10000}``.
    Reading ``.value`` alone turns "at least ten thousand" into "ten thousand" -- measured against a
    live published count of 10,688.
    """
    total = ((payload or {}).get("hits") or {}).get("total") or {}
    relation = "exact" if total.get("relation") == "eq" else "at least"
    return {"value": _as_int(total.get("value")), "relation": relation}


def _hubmap_dataset_filter(status):
    """The two clauses every dataset query starts from, on the exact ``.keyword`` sub-fields."""
    must = [{"term": {"entity_type.keyword": "Dataset"}}]
    wanted = _clean(status)
    if wanted and _norm(wanted) not in ("any", "all"):
        must.append({"term": {"status.keyword": wanted}})
    return must


def _hubmap_organ_table():
    """The live organ ontology, keyed by two-letter code, plus a name index. See F33."""
    table = _cached("organs")
    if table is not None:
        return table, None
    rows, err = _fetch_json(_HUBMAP_ORGANS_URL, params={"application_context": "HUBMAP"})
    if err:
        return None, err
    table = {}
    for row in rows or []:
        code = _clean(row.get("rui_code")).upper()
        if not code:
            continue
        category = row.get("category") if isinstance(row.get("category"), dict) else None
        table[code] = {
            "code": code,
            "term": row.get("term"),
            "category": (category or {}).get("term") or row.get("term"),
            "laterality": row.get("laterality"),
            "organ_uberon": row.get("organ_uberon"),
            "organ_cui": row.get("organ_cui"),
            "rui_supported": row.get("rui_supported"),
        }
    err = _remember("organs", table, "The HuBMAP organ ontology")
    if err:
        return None, err
    return table, None


def _hubmap_resolve_organ(organ):
    """``(codes, label, None)`` for an organ name or code, or ``(None, None, error)``.

    F33 and F34 together. ``organ`` may be a two-letter code or an organ name; a name resolves to
    *every* code sharing its category, so 'lung' becomes ``['LL', 'RL']`` rather than one side of a
    pair, and a code that does not exist is an error naming real organs instead of an empty result.
    """
    wanted = _clean(organ)
    table, err = _hubmap_organ_table()
    if err:
        return None, None, err
    upper = wanted.upper()
    if upper in table:
        row = table[upper]
        return [upper], f"{row['term']} ({upper})", None
    key = _norm(wanted)
    codes = sorted(code for code, row in table.items() if key in (_norm(row["term"]), _norm(row["category"])))
    if codes:
        label = table[codes[0]]["category"]
        return codes, f"{label} ({', '.join(codes)})", None
    names = sorted({row["category"] for row in table.values()} | set(table))
    close = _suggest(wanted, names)
    return (
        None,
        None,
        _error(
            f"'{organ}' is not a HuBMAP organ. "
            + (f"Closest: {', '.join(close)}. " if close else "")
            + "Pass an organ name such as 'lung' or 'kidney' -- a name covers both sides of a paired "
            "organ, which a single two-letter code does not. hubmap_list_organs() lists every one."
        ),
    )


def _hubmap_dataset_row(source):
    organs = [
        sample.get("organ")
        for sample in source.get("origin_samples") or []
        if isinstance(sample, dict) and sample.get("organ")
    ]
    donor = source.get("donor") if isinstance(source.get("donor"), dict) else {}
    mapped = donor.get("mapped_metadata") if isinstance(donor.get("mapped_metadata"), dict) else {}
    return {
        "hubmap_id": source.get("hubmap_id"),
        "uuid": source.get("uuid"),
        "title": source.get("title"),
        "dataset_type": source.get("dataset_type"),
        "status": source.get("status"),
        # Upstream reports organs[0] and drops the rest; a dataset sampling both lungs has two.
        "organ_codes": organs,
        "anatomy": source.get("anatomy_0") or source.get("anatomy_1"),
        "group_name": source.get("group_name"),
        "doi_url": source.get("doi_url"),
        "donor_sex": _first(mapped.get("sex")),
        "donor_age": _first(mapped.get("age_value")),
    }


def _first(value):
    """The search index returns some mapped donor fields as one-element lists."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def hubmap_list_organs(with_dataset_counts=True):
    """List every organ HuBMAP indexes, with its code, UBERON id and live dataset count.

    Forty-seven codes, of which twenty are one organ split by laterality into ten pairs -- Kidney is
    ``LK``/``RK``, Lung is ``LL``/``RL``, and so on (F34). The ``category`` column is what groups a
    pair back together, and it is what ``hubmap_search_datasets`` expands an organ *name* into, so
    that asking for lung does not silently mean one lung.

    ``datasets`` is counted live, which matters because eighteen of the forty-seven codes have no
    published dataset at all. A code with a count of zero is a real ontology entry with nothing
    behind it, and knowing that before querying saves an empty result that looks like a failed
    search.

    Parameters
    ----------
    with_dataset_counts : bool, optional
        Count published datasets per organ. Default True; one extra request.

    Returns
    -------
    dict
        ``data.organs``, sorted by descending dataset count then code, each with ``code``, ``term``,
        ``category``, ``laterality``, ``organ_uberon``, ``rui_supported`` and ``datasets``.

    Examples
    --------
    >>> out = hubmap_list_organs()
    >>> out["data"]["total"]
    47
    """
    table, err = _hubmap_organ_table()
    if err:
        return err

    counts = {}
    if with_dataset_counts:
        payload, _total, err = _hubmap_search(
            _hubmap_dataset_filter("Published"),
            aggs={"organ": {"terms": {"field": "origin_samples.organ.keyword", "size": 200}}},
        )
        if err:
            return err
        buckets = ((payload.get("aggregations") or {}).get("organ") or {}).get("buckets") or []
        counts = {bucket["key"]: bucket["doc_count"] for bucket in buckets}

    organs = []
    for code, row in table.items():
        entry = dict(row)
        if with_dataset_counts:
            entry["datasets"] = counts.get(code, 0)
        organs.append(entry)
    organs.sort(key=lambda row: (-row.get("datasets", 0), row["code"]))

    pairs = sorted({row["category"] for row in table.values() if row.get("laterality")})
    return _ok(
        {
            "total": len(organs),
            "with_published_datasets": sum(1 for row in organs if row.get("datasets")) if with_dataset_counts else None,
            "paired_organs": pairs,
            "organs": organs,
            "note": "Pass the organ NAME to hubmap_search_datasets(); a name covers both codes of a "
            "paired organ, a single code covers one side.",
        }
    )


def hubmap_list_dataset_types(limit=60):
    """List the assay vocabulary HuBMAP actually uses, with live published counts.

    This is the companion to ``hubmap_search_datasets(dataset_type=...)`` and exists because the
    vocabulary cannot be guessed (F36). Fifty-two distinct values are in use; several are a raw assay
    and its processed derivative under near-identical names (``RNAseq`` and ``RNAseq [Salmon]``;
    ``CODEX`` and ``CODEX [Cytokit + SPRM]``), and the spellings that read as obvious -- ``snATACseq``,
    ``scRNAseq-10xGenomics-v3`` -- match nothing.

    Parameters
    ----------
    limit : int, optional
        How many of the most-used types to return, 1-200. Default 60, which is the whole vocabulary.

    Returns
    -------
    dict
        ``data.dataset_types``, each ``{"dataset_type": str, "datasets": int}``, most-used first.

    Examples
    --------
    >>> out = hubmap_list_dataset_types(limit=3)
    >>> out["data"]["dataset_types"][0]["datasets"] > 0
    True
    """
    size = max(1, min(_as_int(limit, default=60) or 60, 200))
    payload, _total, err = _hubmap_search(
        _hubmap_dataset_filter("Published"),
        aggs={"dtype": {"terms": {"field": "dataset_type.keyword", "size": size}}},
    )
    if err:
        return err
    buckets = ((payload.get("aggregations") or {}).get("dtype") or {}).get("buckets") or []
    if not buckets:
        return _error(
            "HuBMAP returned no dataset_type aggregation. The search index answered, so this is an "
            "upstream mapping change rather than an outage -- re-run hubmap_search_datasets() "
            "without a dataset_type filter to confirm the index is otherwise healthy."
        )
    return _ok(
        {
            "returned": len(buckets),
            "dataset_types": [{"dataset_type": bucket["key"], "datasets": bucket["doc_count"]} for bucket in buckets],
            "note": "These strings are matched exactly. A bracketed suffix such as '[Salmon]' is a "
            "processed derivative of the assay of the same name, counted separately.",
        }
    )


def hubmap_search_datasets(organ=None, dataset_type=None, query=None, status="Published", limit=10):
    """Search HuBMAP's published human tissue atlas datasets by organ, assay or free text.

    HuBMAP is the NIH Human BioMolecular Atlas Program: spatial and single-cell measurements of
    healthy human tissue, with donor and anatomical provenance attached to every dataset. It is the
    natural place to look for a human spatial reference next to a mouse Allen atlas.

    Three corrections to the obvious query are built in.

    ``organ`` takes a **name**, resolved live against HuBMAP's ontology, and a paired organ expands
    to both codes -- 'lung' searches ``LL`` and ``RL`` and reports 1,671 published datasets, where a
    single code finds 769 or 906 (F34). Two-letter codes are still accepted. An organ HuBMAP does
    not carry is an error naming close ones, not an empty success (F33).

    ``dataset_type`` is matched **exactly**, so 'RNAseq' means the assay called RNAseq and not the
    three different assays whose names contain it (F36). ``hubmap_list_dataset_types()`` is the
    vocabulary.

    ``total`` carries a ``relation``: Elasticsearch caps its count at 10,000, so a broad query
    reports ``{"value": 10000, "relation": "at least"}`` and says so rather than reporting a
    truncated count as exact (F35).

    Parameters
    ----------
    organ : str, optional
        Organ name (``'lung'``, ``'kidney'``, ``'placenta'``) or two-letter code (``'LL'``).
    dataset_type : str, optional
        Exact assay name, e.g. ``'RNAseq'``, ``'CODEX'``, ``'Visium (no probes)'``, ``'Xenium'``.
    query : str, optional
        Free text over title, description, dataset type and anatomy.
    status : str, optional
        ``'Published'`` (default), ``'Retracted'``, or ``'any'`` for both.
    limit : int, optional
        Datasets to return, 1-50. Default 10.

    Returns
    -------
    dict
        ``data`` with ``total`` (``value`` + ``relation``), ``returned``, ``filters`` naming the
        codes and assay actually searched, and ``datasets``.

    Examples
    --------
    >>> out = hubmap_search_datasets(organ="lung", dataset_type="Visium (no probes)", limit=3)
    >>> out["data"]["filters"]["organ_codes"]
    ['LL', 'RL']
    """
    must = _hubmap_dataset_filter(status)
    filters = {"status": _clean(status) or "any"}

    if _clean(organ):
        codes, label, err = _hubmap_resolve_organ(organ)
        if err:
            return err
        must.append({"terms": {"origin_samples.organ.keyword": codes}})
        filters["organ"] = label
        filters["organ_codes"] = codes

    if _clean(dataset_type):
        types, err = _hubmap_type_vocabulary()
        if err:
            return err
        exact = _clean(dataset_type)
        if exact not in types:
            fold = {_norm(name): name for name in types}
            if _norm(exact) in fold:
                exact = fold[_norm(exact)]
            else:
                close = _suggest(exact, sorted(types))
                return _error(
                    f"'{dataset_type}' is not a HuBMAP dataset_type. "
                    + (f"Closest: {', '.join(close)}. " if close else "")
                    + "The names are matched exactly and cannot be guessed -- "
                    "hubmap_list_dataset_types() returns every one with its count."
                )
        must.append({"term": {"dataset_type.keyword": exact}})
        filters["dataset_type"] = exact

    if _clean(query):
        must.append(
            {
                "multi_match": {
                    "query": _clean(query),
                    "fields": ["title", "description", "dataset_type", "anatomy_0", "anatomy_1"],
                }
            }
        )
        filters["query"] = _clean(query)

    if len(must) == 1:
        filters["note"] = "No filter given, so this is the whole index."

    size = max(1, min(_as_int(limit, default=10) or 10, _HUBMAP_MAX_LIMIT))
    payload, total, err = _hubmap_search(must, size=size, source=_HUBMAP_HIT_SOURCE)
    if err:
        return err
    hits = ((payload.get("hits") or {}).get("hits")) or []
    if not hits:
        return _error(
            "No HuBMAP dataset matches those filters: "
            + "; ".join(f"{key}={value}" for key, value in filters.items())
            + ". Each filter narrows independently -- drop one and re-run to see which is empty. "
            "hubmap_list_organs() reports the per-organ counts and hubmap_list_dataset_types() the "
            "per-assay ones, so an empty combination is visible before it is queried.",
            filters=filters,
        )
    return _ok(
        {
            "filters": filters,
            "total": total,
            "returned": len(hits),
            "datasets": [_hubmap_dataset_row(hit.get("_source") or {}) for hit in hits],
        }
    )


def _hubmap_type_vocabulary():
    """The live set of ``dataset_type`` values, cached for the process (F36)."""
    names = _cached("dataset_types")
    if names is not None:
        return names, None
    payload, _total, err = _hubmap_search(
        _hubmap_dataset_filter("Published"),
        aggs={"dtype": {"terms": {"field": "dataset_type.keyword", "size": 200}}},
    )
    if err:
        return None, err
    buckets = ((payload.get("aggregations") or {}).get("dtype") or {}).get("buckets") or []
    names = {bucket["key"] for bucket in buckets}
    err = _remember("dataset_types", names, "The HuBMAP search index (dataset_type aggregation)")
    if err:
        return None, err
    return names, None


def _hubmap_entity(entity_id, what):
    """``(record, None)`` for one entity, or ``(None, error)`` with the reason spelled out.

    The entity API answers a malformed HuBMAP id with 400 and an unknown-but-well-formed one with
    404. Those are different mistakes -- a typo in the shape versus an id that does not exist -- so
    the format is checked here first and each gets its own message.
    """
    wanted = _clean(entity_id)
    if not wanted:
        return None, _missing(what, "Pass a HuBMAP id like 'HBM527.MDNB.349', or an entity uuid.")
    if not _HUBMAP_ID_RE.match(wanted):
        return None, _error(
            f"'{entity_id}' is not a HuBMAP identifier. They take one of two shapes: a HuBMAP id "
            f"like 'HBM527.MDNB.349' (HBM, three digits, a dot, four letters, a dot, three digits) "
            f"or a 32-character hexadecimal uuid. hubmap_search_datasets() returns both."
        )
    record, err = _fetch_json(f"{_HUBMAP_ENTITY_URL}/entities/{_seg(wanted)}")
    if err:
        detail = _clean(err.get("error"))
        if "404" in detail:
            return None, _error(
                f"HuBMAP has no entity {wanted}. The id is well-formed, so it is either retracted, "
                f"not yet public, or from another consortium. hubmap_search_datasets(status='any') "
                f"will find it if HuBMAP holds it at all."
            )
        return None, err
    return record, None


def _hubmap_index_record(entity_id, fields):
    """One entity as the *search index* holds it. Returns ``{}`` when the index has no such doc.

    The entity API and the search index are two different views. The index carries provenance the
    entity record does not -- ``origin_samples.organ`` (F40) and ``immediate_ancestor_ids`` (F39) --
    so a few answers need both. A miss here is not an error: the entity may exist and simply not be
    indexed, and the caller degrades rather than fails.
    """
    field = "uuid.keyword" if len(entity_id) == 32 and "." not in entity_id else "hubmap_id.keyword"
    payload, _total, err = _hubmap_search([{"term": {field: entity_id}}], size=1, source=fields)
    if err:
        return {}
    hits = ((payload.get("hits") or {}).get("hits")) or []
    return (hits[0].get("_source") or {}) if hits else {}


def hubmap_get_dataset(hubmap_id):
    """Full metadata for one HuBMAP dataset: assay, organ, donor group, DOI and access level.

    Two things are worth knowing before reading the result.

    **The organ comes from the search index, not the dataset record.** The entity API simply does not
    carry ``origin_samples``, so reading the organ from it yields ``None`` for every dataset ever
    queried (F40). It is resolved here against the live organ ontology and reported as both the code
    and the anatomical term.

    **Contributor e-mail addresses are not returned.** HuBMAP publishes them and upstream forwards
    them; they are third-party personal data, they answer no research question, and once in a
    transcript they are in every downstream artefact that transcript feeds. Names, affiliations and
    ORCIDs are kept, which is what attribution actually needs (F37).

    ``data_access_level`` is the field to read before planning to download anything: ``'protected'``
    means the raw data needs dbGaP authorisation even though the metadata is public.

    Parameters
    ----------
    hubmap_id : str
        A HuBMAP id such as ``'HBM527.MDNB.349'``, or a 32-character entity uuid.

    Returns
    -------
    dict
        ``data`` with identity, assay, organ, donor group, DOI, access level and contributors.

    Examples
    --------
    >>> out = hubmap_get_dataset("HBM527.MDNB.349")
    >>> out["data"]["organ"]["code"]
    'RK'
    """
    record, err = _hubmap_entity(hubmap_id, "hubmap_id")
    if err:
        return err
    entity_type = _clean(record.get("entity_type"))
    if entity_type and entity_type != "Dataset":
        return _error(
            f"{record.get('hubmap_id') or hubmap_id} is a {entity_type}, not a Dataset. "
            f"hubmap_get_dataset_provenance() reads samples and donors; this function reads "
            f"datasets."
        )

    indexed = _hubmap_index_record(
        _clean(record.get("uuid")) or _clean(hubmap_id), ("origin_samples.organ", "uuid", "hubmap_id")
    )
    codes = [
        sample.get("organ")
        for sample in indexed.get("origin_samples") or []
        if isinstance(sample, dict) and sample.get("organ")
    ]
    organ = {"code": None, "term": None, "note": "not indexed"}
    if codes:
        table, table_err = _hubmap_organ_table()
        if table_err:
            return table_err
        row = table.get(codes[0]) or {}
        organ = {
            "code": codes[0],
            "term": row.get("term"),
            "category": row.get("category"),
            "organ_uberon": row.get("organ_uberon"),
            "all_codes": codes,
        }

    people = []
    for person in (record.get("contributors") or record.get("contacts") or [])[:15]:
        if not isinstance(person, dict):
            continue
        people.append(
            {
                "name": person.get("display_name")
                or " ".join(filter(None, [person.get("first_name"), person.get("last_name")])),
                "affiliation": person.get("affiliation"),
                "orcid": person.get("orcid"),
                "principal_investigator": person.get("is_principal_investigator") == "Yes",
            }
        )

    return _ok(
        {
            "hubmap_id": record.get("hubmap_id"),
            "uuid": record.get("uuid"),
            "entity_type": record.get("entity_type"),
            "dataset_type": record.get("dataset_type"),
            "status": record.get("status"),
            "title": record.get("title"),
            "description": record.get("description"),
            "organ": organ,
            "group_name": record.get("group_name"),
            "doi_url": record.get("doi_url"),
            "registered_doi": record.get("registered_doi"),
            "data_access_level": record.get("data_access_level"),
            "contains_human_genetic_sequences": record.get("contains_human_genetic_sequences"),
            "created_timestamp": record.get("created_timestamp"),
            "published_timestamp": record.get("published_timestamp"),
            "contributors": people,
            "note": "Contributor e-mail addresses are deliberately not returned. "
            "data_access_level 'protected' means the raw data needs dbGaP authorisation.",
        }
    )


_HUBMAP_SAMPLE_RANK = {"section": 1, "suspension": 1, "block": 2, "organ": 3}
_HUBMAP_TYPE_RANK = {"Dataset": 0, "Sample": 1, "Donor": 2, "Publication": 3}


def _hubmap_lineage_row(source):
    return {
        "entity_type": source.get("entity_type"),
        "hubmap_id": source.get("hubmap_id"),
        "uuid": source.get("uuid"),
        "sample_category": source.get("sample_category"),
        "organ": source.get("organ"),
        "dataset_type": source.get("dataset_type"),
        "group_name": source.get("group_name"),
    }


def hubmap_get_dataset_provenance(entity_id):
    """Walk a HuBMAP dataset back to the donor it came from, one step at a time.

    HuBMAP's value over a plain expression matrix is this chain: a dataset was processed from a
    suspension, cut from a section, cut from a block, taken from an organ, taken from a consented
    donor -- and every link is a registered entity with its own id. Reading it tells you whether two
    datasets share a donor, how many processing steps sit between the published matrix and the
    tissue, and which organ the tissue is actually from.

    **The chain here is walked, not guessed** (F39). The ancestors endpoint returns an unordered bag
    with no parent pointers, so sorting it by entity kind cannot tell two blocks or two datasets
    apart. The search index carries ``immediate_ancestor_ids``, so the parent link is followed node
    by node and the result is labelled ``lineage_quality: "exact"``. If the index does not hold every
    ancestor the result falls back to ordering by kind and says ``"approximate"`` -- which is a
    weaker claim, and saying so is the point.

    Parameters
    ----------
    entity_id : str
        A HuBMAP dataset id such as ``'HBM527.MDNB.349'``, or its 32-character uuid.

    Returns
    -------
    dict
        ``data`` with ``lineage`` ordered dataset-first, ``lineage_quality``, ``donor`` and
        ``organ`` pulled out, and ``ancestor_count``.

    Examples
    --------
    >>> out = hubmap_get_dataset_provenance("HBM527.MDNB.349")
    >>> out["data"]["lineage_quality"]
    'exact'
    """
    record, err = _hubmap_entity(entity_id, "entity_id")
    if err:
        return err
    uuid = _clean(record.get("uuid"))
    ancestors, err = _fetch_json(f"{_HUBMAP_ENTITY_URL}/ancestors/{_seg(uuid)}")
    if err:
        return err
    if not isinstance(ancestors, list) or not ancestors:
        return _error(
            f"{record.get('hubmap_id') or entity_id} has no registered ancestors. A Donor is the "
            f"root of every HuBMAP lineage and has none by definition; anything else with none is "
            f"an entity registered without provenance."
        )

    uuids = [uuid] + [_clean(entity.get("uuid")) for entity in ancestors if entity.get("uuid")]
    payload, _total, search_err = _hubmap_search(
        [{"terms": {"uuid.keyword": uuids}}], size=len(uuids) + 5, source=_HUBMAP_LINEAGE_SOURCE
    )
    indexed = {}
    if not search_err:
        for hit in ((payload.get("hits") or {}).get("hits")) or []:
            source = hit.get("_source") or {}
            if source.get("uuid"):
                indexed[source["uuid"]] = source

    lineage, quality = [], "approximate"
    if len(indexed) == len(set(uuids)):
        node, seen = uuid, set()
        while node and node in indexed and node not in seen:
            seen.add(node)
            source = indexed[node]
            lineage.append(_hubmap_lineage_row(source))
            parents = source.get("immediate_ancestor_ids") or []
            node = _clean(parents[0]) if parents else None
        if len(seen) == len(set(uuids)):
            quality = "exact"
        else:
            lineage = []

    if not lineage:
        ordered = sorted(
            [indexed.get(entity.get("uuid")) or entity for entity in [record] + ancestors],
            key=lambda entity: (
                _HUBMAP_TYPE_RANK.get(_clean(entity.get("entity_type")), 9),
                _HUBMAP_SAMPLE_RANK.get(_norm(entity.get("sample_category")), 0),
                -_as_int(entity.get("created_timestamp")),
            ),
        )
        lineage = [_hubmap_lineage_row(entity) for entity in ordered]

    donor = next((row for row in lineage if row["entity_type"] == "Donor"), None)
    organ_row = next((row for row in lineage if row.get("organ")), None)
    return _ok(
        {
            "hubmap_id": record.get("hubmap_id"),
            "uuid": uuid,
            "lineage_quality": quality,
            "lineage": lineage,
            "ancestor_count": len(lineage) - 1,
            "donor": donor,
            "organ": organ_row.get("organ") if organ_row else None,
            "note": "Ordered dataset-first, each entry the direct parent of the one above it."
            if quality == "exact"
            else "The search index did not hold every ancestor, so this is ordered by entity "
            "kind rather than by the parent link -- entities of the same kind may be in "
            "the wrong order relative to each other.",
        }
    )
