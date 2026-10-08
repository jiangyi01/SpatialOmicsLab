"""Ontology lookup against EMBL-EBI OLS4: terms, hierarchies, cross-references, disease IDs.

Spatial-omics work runs on controlled vocabularies whether or not anyone says so out loud. A cell
type is a CL term, a tissue is an UBERON term, a phenotype is an HP term, a disease is a MONDO
term, a pathway annotation is a GO term, and a public dataset's sample sheet is annotated in EFO.
Any time two datasets have to be compared, or a marker panel has to be mapped onto an atlas, the
question is the same: what is this label's identifier, what sits above it, what sits below it, and
what is it called in the other vocabulary.

The Ontology Lookup Service (OLS4, ``www.ebi.ac.uk/ols4``) answers all four for 280+ ontologies
from one endpoint, so this module is a thin, well-behaved client for it plus one convenience on
top:

* **Search** -- free text or genuinely exact, scoped to one ontology or across all of them.
* **Term records** -- label, definition, synonyms, obsolescence and, when a term is obsolete, the
  term that replaced it.
* **Hierarchy** -- direct children, ancestors, and the full descendant subtree.
* **Cross-references** -- the ``obo_xref`` block that maps a term onto DOID, ICD-10, OMIM, UMLS,
  MeSH, NCIt, SNOMED and MedDRA. This is the identifier-mapping workhorse.
* **Disease name to identifier** -- a reranked lookup, because OLS's own relevance order gets
  common disease abbreviations wrong.

Every outbound call goes through :mod:`spatialomicsgym.utils.http_client`, which enforces HTTPS, an
explicit host allowlist, one shared connection pool, a timeout and bounded retry. No function here
calls ``requests`` directly.

Return shape, uniform across the module: ``{"status": "success", "data": ..., "metadata": {...}}``
on success and ``{"status": "error", "error": "<what went wrong and what to do about it>"}`` on
failure. Nothing raises for an expected failure -- a network problem or a bad argument comes back as
a value, so one failed lookup inside a longer script does not abort the rest of it. The agent loop
recognises ``"status": "error"`` as a failed action, so a failure is never silently read as data.

These functions return a dict; ``print()`` the result (or the part of it you need) or it will not
appear in the observation.

------------------------------------------------------------------------------------------------
Adapted from ToolUniverse -- https://github.com/mims-harvard/ToolUniverse -- at commit
``f075c2a75e8b35ae5dbb220d48d4e87e980388b1``. Copyright [2025] [ToolUniverse team], licensed under
the Apache License, Version 2.0.

CHANGED BY SPATIALOMICSGYM, as Apache-2.0 section 4(b) requires this file to state. The endpoint
knowledge and two of the response-repair behaviours (constraining ``queryFields`` so ``exact=true``
means what it says; reranking a disease search onto an exact synonym) are upstream's; the code is
not. Specifically: the ``BaseTool``/``register_tool``/config-driven ``operation`` dispatch machinery
was not vendored and each upstream operation is re-expressed here as a plain function; the pydantic
response models were dropped in favour of plain dict projections that keep more of the record; raw
``requests`` calls were replaced by our HTTP layer; user-supplied values interpolated into URL
*paths* are percent-encoded, which upstream does not do; term IRIs are **resolved from the service**
rather than constructed from a CURIE, which fixes a defect that made every EFO-native hierarchy
query silently return an empty result; the disease-search rerank window was widened because
upstream's was too small to contain its own worked example; and the five EFO-scoped catalog entries
that differ from the generic tools only by a pre-filled ``ontology="efo"`` were not vendored as
separate tools. See ``VENDORING.md`` for the full record.
"""

import logging
import re
from urllib.parse import quote

from spatialomicsgym.utils.http_client import HttpError, request_json

logger = logging.getLogger(__name__)

# The only host this module is permitted to reach. Passed to the HTTP layer on every call, which
# refuses anything else -- so no argument, however malformed or adversarial, can redirect a request
# at a host that is not on this list. Cleared under the RL-1 origin review recorded in
# CHINA_EXCLUSION.md (EMBL-EBI, Hinxton, UK).
_ALLOWED_HOSTS = ("www.ebi.ac.uk",)

#: OLS4 runs two API generations side by side and they are not interchangeable. ``/api`` (v1) is
#: the one that answers term records by CURIE, carries ``obo_xref``/``has_children``/
#: ``term_replaced_by``, and serves the full descendant subtree. ``/api/v2`` is the one that serves
#: the class hierarchy (children/ancestors) and the ontology catalog. Each function below uses
#: whichever generation actually answers its question; the split is not an accident.
_OLS_V1 = "https://www.ebi.ac.uk/ols4/api"
_OLS_V2 = "https://www.ebi.ac.uk/ols4/api/v2"

#: A page ceiling, so a stray ``size=100000`` cannot paste a megabyte of ontology into the
#: observation and blow the context window.
_MAX_PAGE = 200

#: ``exact=true`` on its own restricts almost nothing: OLS matches across label, synonym,
#: description, iri, short_form and obo_id, so an "exact" hit on a *description* token drags in the
#: whole neighbourhood. Measured live against the API on 2026-09-17:
#:   q=fibroblast&ontology=cl&exact=true                       -> 167 hits
#:   q=fibroblast&ontology=cl&exact=true&queryFields=label,synonym -> 1 hit
#: Constraining ``queryFields`` is what makes ``exact`` mean what it says. Synonyms are included
#: because an exact hit on an alternative name is a real exact match ("T-lymphocyte" -> CL:0000084).
_EXACT_NAME_FIELDS = "label,synonym"

#: An identifier is not a name, and restricting it to label/synonym returns nothing at all, so an
#: identifier-shaped query is matched against the identifier fields instead.
_EXACT_IDENTIFIER_FIELDS = "obo_id,short_form,iri"

#: CURIE ("CL:0000084") or the OBO underscore form ("CL_0000084"). Whitespace is rejected so an
#: ordinary multi-word label such as "type 2 diabetes mellitus" is treated as a name.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.]*[:_][A-Za-z0-9._-]+$")

#: A disease search reranks onto an exact label/synonym match, and can only rerank over what it
#: fetched. Upstream fetches 10. Measured live on 2026-09-17, the exact match for "type 2 diabetes"
#: (MONDO:0005148) sits at rank 11 in the EFO-scoped result, i.e. one place outside that window, so
#: upstream answers with EFO:1001503 "type II diabetes mellitus with acanthosis nigricans" instead.
#: One request either way; the wider window is free.
_RERANK_FETCH = 25


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


def _seg(value):
    """Percent-encode a caller-supplied value going into a URL *path* segment."""
    return quote(str(value), safe="")


def _iri_seg(iri):
    """Double percent-encode an IRI for use as an OLS path segment.

    OLS puts the whole IRI in the path, so the slashes inside it have to survive one round of
    decoding by the web server before the application sees them. Encoding twice is what the API
    documents and what it requires.
    """
    return quote(quote(str(iri), safe=""), safe="")


def _fetch_json(url, **kwargs):
    """``(payload, None)`` on success, ``(None, error_payload)`` on failure."""
    try:
        return request_json(url, allowed_hosts=_ALLOWED_HOSTS, **kwargs), None
    except HttpError as exc:
        return None, _error(exc.detail)


def _page_size(size, default=20):
    try:
        return max(1, min(int(size), _MAX_PAGE))
    except (TypeError, ValueError):
        return default


def _exact_query_fields(query):
    candidate = str(query).strip()
    if candidate.lower().startswith(("http://", "https://")):
        return _EXACT_IDENTIFIER_FIELDS
    if _IDENTIFIER_RE.match(candidate):
        return _EXACT_IDENTIFIER_FIELDS
    return _EXACT_NAME_FIELDS


#: Prefixes whose OLS ontology id is not the prefix lower-cased: Orphanet's terms live in "ordo".
_ONTOLOGY_ID_FOR_PREFIX = {"orphanet": "ordo"}

#: Canonical spelling of the mixed-case prefixes, for a CURIE typed all in lower case.
_MIXED_CASE_PREFIXES = {
    name.lower(): name for name in ("NCBITaxon", "HsapDv", "MmusDv", "Orphanet", "FBbt", "FBdv", "WBbt", "ZFA")
}


def _infer_ontology(identifier):
    """Ontology id from a term identifier: "HP:0001903" -> "hp", and an IRI's last segment too."""
    text = str(identifier or "").strip()
    if not text:
        return ""
    if text.lower().startswith(("http://", "https://")):
        tail = text.rstrip("/").rsplit("/", 1)[-1]
        # An OBO PURL ends in PREFIX_LOCALID; an EFO IRI ends the same way.
        if "_" in tail:
            prefix = tail.split("_", 1)[0].lower()
            return _ONTOLOGY_ID_FOR_PREFIX.get(prefix, prefix)
        return ""
    for separator in (":", "_"):
        if separator in text:
            # "Orphanet" was inferred as "orphanet", an id OLS does not have, so the IRI lookup always
            # failed (hunt 2026-09-30, uT1-database-28).
            prefix = text.split(separator, 1)[0].lower()
            return _ONTOLOGY_ID_FOR_PREFIX.get(prefix, prefix)
    return ""


def _canonical_prefix(prefix):
    """The prefix as given, unless it is all lower case: then its canonical spelling, else upper case.

    Upper-casing every prefix turned NCBITaxon:9606, HsapDv:... and MmusDv:... -- the organism and
    stage IDs every CELLxGENE-schema h5ad carries -- into NCBITAXON:9606 and HSAPDV:..., which are
    not those terms' IDs: they fail schema validation and every case-sensitive join, and they are
    what the lookup sent (hunt 2026-09-30, uT1-database-28).
    """
    if prefix.islower():
        return _MIXED_CASE_PREFIXES.get(prefix, prefix.upper())
    return prefix


def _curie(identifier):
    """Normalise "CL_0000084" or an IRI to the colon CURIE "CL:0000084"; pass anything else through."""
    text = str(identifier or "").strip()
    if text.lower().startswith(("http://", "https://")):
        text = text.rstrip("/").rsplit("/", 1)[-1]
    if ":" in text:
        prefix, local = text.split(":", 1)
        return f"{_canonical_prefix(prefix)}:{local}"
    if "_" in text:
        prefix, local = text.split("_", 1)
        return f"{_canonical_prefix(prefix)}:{local}"
    return text


def _first_text(value):
    """OLS4 v2 returns ``label`` as a string on one endpoint and a list on another."""
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item
        return ""
    return value if isinstance(value, str) else ""


def _resolve_term(identifier, ontology):
    """``(record, ontology, None)`` for a term, or ``(None, ontology, error_payload)``.

    Upstream builds a term's IRI by string surgery on its CURIE -- every prefix is pasted onto
    ``http://purl.obolibrary.org/obo/``. That is wrong for EFO, whose terms live at
    ``http://www.ebi.ac.uk/efo/``, and the failure is silent: the hierarchy endpoints answer HTTP
    200 with an empty element list for an IRI they do not know, which reads exactly like a leaf
    term. (Measured live on 2026-09-17: the v2 class record for
    ``http://purl.obolibrary.org/obo/EFO_0000408`` is a 404 while
    ``http://www.ebi.ac.uk/efo/EFO_0000408`` is a 200.)

    So we do not construct IRIs. We ask OLS for the term record and read the ``iri`` it reports,
    which is correct for every ontology by construction and needs no table of special cases.
    """
    text = str(identifier or "").strip()
    ontology = str(ontology or "").strip().lower() or _infer_ontology(text)

    if text.lower().startswith(("http://", "https://")):
        if not ontology:
            return (
                None,
                ontology,
                _error(
                    f"Could not tell which ontology '{text}' belongs to. Pass ontology, for example ontology='efo'."
                ),
            )
        payload, error = _fetch_json(f"{_OLS_V1}/ontologies/{_seg(ontology)}/terms/{_iri_seg(text)}")
        if error is not None:
            return None, ontology, error
        if isinstance(payload, dict) and payload.get("iri"):
            return payload, ontology, None
        return None, ontology, _error(f"Term '{text}' was not found in ontology '{ontology}'.")

    curie = _curie(text)
    terms = None
    if ontology:
        # The ontology-scoped lookup returns the *canonical* record for that ontology. The unscoped
        # one can answer with a copy held by some importing ontology instead, which carries a
        # thinner annotation set.
        payload, error = _fetch_json(f"{_OLS_V1}/ontologies/{_seg(ontology)}/terms", params={"obo_id": curie})
        if error is None and isinstance(payload, dict):
            terms = (payload.get("_embedded") or {}).get("terms")
    if not terms:
        payload, error = _fetch_json(f"{_OLS_V1}/terms", params={"id": curie})
        if error is not None:
            return None, ontology, error
        if isinstance(payload, dict):
            terms = (payload.get("_embedded") or {}).get("terms")
    if not terms:
        return (
            None,
            ontology,
            _error(
                f"Term '{curie}' was not found in OLS. Check the identifier, or search for it by name "
                "with ols_search_terms first."
            ),
        )
    record = terms[0]
    return record, (ontology or record.get("ontology_name") or ""), None


def _term_summary(record):
    """Project an OLS v1 term record onto the fields worth carrying into an observation.

    ``term_replaced_by`` is kept because it is the difference between "that identifier is dead" and
    "that identifier is dead, here is the live one". EFO retired its disease branch to MONDO, so a
    disease CURIE copied out of an older paper is very often obsolete with a replacement recorded.
    """
    summary = {
        "iri": record.get("iri"),
        "obo_id": record.get("obo_id"),
        "short_form": record.get("short_form"),
        "label": _first_text(record.get("label")),
        "description": record.get("description") or [],
        "synonyms": record.get("synonyms") or [],
        "ontology_name": record.get("ontology_name"),
        "ontology_prefix": record.get("ontology_prefix"),
        "has_children": record.get("has_children"),
        "is_obsolete": bool(record.get("is_obsolete")),
        "is_root": record.get("is_root"),
    }
    replaced_by = record.get("term_replaced_by")
    if replaced_by:
        summary["term_replaced_by"] = replaced_by
        summary["term_replaced_by_curie"] = _curie(replaced_by)
    return summary


def _v2_element(item):
    """Project an OLS v2 class element. Returns ``None`` for an element with no identifier."""
    iri = item.get("iri") or item.get("@id") or item.get("id")
    if not iri:
        return None
    curie = item.get("curie") or item.get("shortForm") or ""
    return {
        "iri": iri,
        "obo_id": _curie(curie) if curie else None,
        "short_form": item.get("shortForm"),
        "label": _first_text(item.get("label")),
        "ontology_name": item.get("ontologyId"),
        "is_obsolete": bool(item.get("isObsolete")),
        "has_children": item.get("hasDirectChildren"),
    }


def _v2_collection(payload, size):
    """Normalise a v2 paged envelope to ``(terms, total)``; an empty page is a real answer."""
    elements = payload.get("elements") if isinstance(payload, dict) else None
    if not isinstance(elements, list):
        elements = []
    terms = [_v2_element(item) for item in elements[:size]]
    terms = [term for term in terms if term is not None]
    total = payload.get("totalElements") if isinstance(payload, dict) else None
    if total is None:
        total = len(terms)
    return terms, total


def _as_int(value):
    """OLS reports its counts as strings ("94312"); make them numbers or leave them alone."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _ontology_summary(item):
    return {
        "ontology_id": item.get("ontologyId"),
        "title": item.get("title"),
        "preferred_prefix": item.get("preferredPrefix"),
        "description": item.get("description"),
        "homepage": item.get("homepage"),
        "version": item.get("version"),
        "number_of_classes": _as_int(item.get("numberOfClasses")),
        "number_of_entities": _as_int(item.get("numberOfEntities")),
        "loaded": item.get("loaded"),
    }


def ols_search_terms(
    query,
    ontology=None,
    rows=10,
    exact_match=False,
    obsolete_only=False,
):
    """Search OLS4 for ontology terms by name, synonym or identifier.

    This is the entry point when you have a label and need an identifier: a cell-type name from a
    marker table, a tissue name from a sample sheet, a disease name from a paper.

    Parameters
    ----------
    query : str
        What to search for. A name ("T cell", "hepatocyte"), an identifier ("CL:0000084") or a
        full IRI all work.
    ontology : str, optional
        Restrict to one ontology by its OLS id, lower case: "cl" (cell types), "uberon" (anatomy),
        "hp" (phenotypes), "mondo" (disease), "go" (gene ontology), "efo" (experimental factors),
        "chebi" (chemistry). Leave unset to search all 280+.
    rows : int, optional
        How many hits to return, 1-200. Default 10.
    exact_match : bool, optional
        Require the query to equal a term's label or one of its exact synonyms in full, rather than
        merely appear somewhere in its record. Default False. Turning this on is what separates
        "fibroblast" (1 hit) from "anything whose definition mentions fibroblasts" (167 hits).
        If the query looks like an identifier rather than a name, the exact match is applied to the
        identifier fields instead, which returns that identifier in every ontology that declares
        it -- pass ontology to narrow that to one.
    obsolete_only : bool, optional
        Search the terms OLS has **retired** instead of the current ones. Default False. Note the
        name: OLS's search returns one set or the other, never both -- measured live, "asthma" in
        EFO matches 42 current terms or 13 retired ones, and the two lists are disjoint. Turn this
        on only when chasing an identifier from an older paper that no longer resolves; leave it off
        for every ordinary lookup.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "num_found": int, ...}``. Each hit carries iri,
        obo_id, label, description, synonyms and ontology_name. ``num_found`` is the total the
        service matched, which is usually larger than the page returned.

    Examples
    --------
    >>> hits = ols_search_terms(query="T cell", ontology="cl", exact_match=True)
    >>> print(hits["data"][0]["obo_id"], hits["data"][0]["label"])
    >>> print(ols_search_terms(query="spatial transcriptomics", ontology="efo", rows=5))
    >>> print(ols_search_terms(query="asthma", ontology="efo", obsolete_only=True)["data"])
    """
    if not query:
        return _missing(
            "query",
            "Pass the term name or identifier to look up, for example query='T cell'.",
        )

    rows = _page_size(rows, default=10)
    params = {
        "q": query,
        "rows": rows,
        "start": 0,
        "exact": bool(exact_match),
        # Named "obsoletes" upstream and documented there as "include obsolete terms", which is not
        # what it does. Measured live on 2026-09-17: q=asthma&ontology=efo returns 42 hits, and
        # adding obsoletes=true returns 13 -- a *disjoint* set of retired terms, not a superset. The
        # search docs do not even carry is_obsolete, so a caller who believed the upstream wording
        # would silently lose every current term and have no way to see it. Hence the parameter is
        # named for what it does.
        "obsoletes": bool(obsolete_only),
    }
    # Only constrain queryFields when exact matching was actually asked for; an unfiltered search
    # needs its full-text recall.
    exact_fields = _exact_query_fields(query) if exact_match else None
    if exact_fields:
        params["queryFields"] = exact_fields
    if ontology:
        params["ontology"] = str(ontology).strip().lower()

    payload, error = _fetch_json(f"{_OLS_V1}/search", params=params)
    if error is not None:
        return error

    response = payload.get("response") if isinstance(payload, dict) else None
    docs = (response or {}).get("docs") or []
    num_found = (response or {}).get("numFound", len(docs))

    # Upstream funnels these hits through a pydantic model that keeps six fields and discards the
    # rest. The discarded ones -- description, synonyms, ontology_prefix -- are the ones that let a
    # reader tell two similarly named terms apart without a second round trip, so we keep them.
    terms = [
        {
            "iri": doc.get("iri"),
            "obo_id": doc.get("obo_id"),
            "short_form": doc.get("short_form"),
            "label": doc.get("label"),
            "description": doc.get("description") or [],
            "exact_synonyms": doc.get("exact_synonyms") or [],
            "ontology_name": doc.get("ontology_name"),
            "ontology_prefix": doc.get("ontology_prefix"),
            "entity_type": doc.get("type"),
        }
        for doc in docs[:rows]
    ]

    filters = {
        "ontology": ontology,
        "exact_match": bool(exact_match),
        "obsolete_only": bool(obsolete_only),
    }
    if exact_fields:
        # State which fields the exact match was applied to, so the echoed exact_match flag is a
        # verifiable claim rather than one the reader has to take on trust.
        filters["exact_match_fields"] = exact_fields

    return _ok(terms, query=query, num_found=num_found, showing=len(terms), filters=filters)


def ols_get_term_info(term_id, ontology=None):
    """Fetch the full OLS record for one ontology term: label, definition, synonyms, status.

    Use this to confirm that an identifier means what a paper says it means, and to find out
    whether it is still current -- an obsolete term reports the identifier that replaced it.

    Parameters
    ----------
    term_id : str
        The term, as a CURIE ("HP:0001903", "MONDO:0005148"), the OBO underscore form
        ("HP_0001903") or a full IRI.
    ontology : str, optional
        The OLS ontology id to look in. Inferred from the identifier's prefix when omitted, which
        is right almost always; pass it explicitly to read a term as some other ontology imports
        it.

    Returns
    -------
    dict
        ``{"status": "success", "data": {...}}`` where data carries iri, obo_id, label,
        description, synonyms, has_children, is_obsolete and -- when the term is retired --
        term_replaced_by and term_replaced_by_curie.

    Examples
    --------
    >>> term = ols_get_term_info(term_id="HP:0001903")
    >>> print(term["data"]["label"], term["data"]["description"])
    >>> print(ols_get_term_info(term_id="EFO:0000270")["data"].get("term_replaced_by_curie"))
    """
    if not term_id:
        return _missing(
            "term_id",
            "Pass the ontology term identifier, for example term_id='HP:0001903'.",
        )

    record, ontology, error = _resolve_term(term_id, ontology)
    if error is not None:
        return error

    summary = _term_summary(record)
    metadata = {"query_id": _curie(term_id), "ontology": ontology}
    if summary["is_obsolete"]:
        metadata["obsolete_note"] = "This term is obsolete in OLS. " + (
            f"Use {summary['term_replaced_by_curie']} instead."
            if summary.get("term_replaced_by_curie")
            else "No replacement term is recorded; search by name for the current term."
        )
        logger.info("OLS term %s is obsolete", summary.get("obo_id") or term_id)
    return _ok(summary, metadata=metadata)


def ols_get_term_children(term_id, ontology=None, size=20, include_obsolete=False):
    """List the direct children (immediate subclasses) of an ontology term.

    One level down only. For the whole subtree use ols_get_term_descendants.

    Parameters
    ----------
    term_id : str
        The parent term, as a CURIE ("CL:0000084"), the underscore form, or a full IRI.
    ontology : str, optional
        The OLS ontology id. Inferred from the identifier's prefix when omitted.
    size : int, optional
        How many children to return, 1-200. Default 20.
    include_obsolete : bool, optional
        Include retired children. Default False.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "total": int}``. An empty list with total 0 is a
        real answer -- it means the term is a leaf, and the term record's has_children field
        agrees.

    Examples
    --------
    >>> kids = ols_get_term_children(term_id="CL:0000084", size=10)
    >>> print(kids["total"], [k["label"] for k in kids["data"]])
    """
    return _hierarchy(term_id, ontology, size, include_obsolete, "children")


def ols_get_term_ancestors(term_id, ontology=None, size=20, include_obsolete=False):
    """List the ancestors of an ontology term -- every class above it, up to the ontology root.

    This is how you place a term: whether a cell type sits under "lymphocyte" or under "myeloid
    leukocyte" is an ancestor question, and it is the check that catches a mis-mapped label before
    it propagates through an annotation.

    Parameters
    ----------
    term_id : str
        The term, as a CURIE ("HP:0001903"), the underscore form, or a full IRI.
    ontology : str, optional
        The OLS ontology id. Inferred from the identifier's prefix when omitted.
    size : int, optional
        How many ancestors to return, 1-200. Default 20.
    include_obsolete : bool, optional
        Include retired ancestors. Default False.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "total": int}``, ordered as OLS returns them --
        broadest first.

    Examples
    --------
    >>> up = ols_get_term_ancestors(term_id="HP:0001903")
    >>> print([a["label"] for a in up["data"]])
    """
    return _hierarchy(term_id, ontology, size, include_obsolete, "ancestors")


def _hierarchy(term_id, ontology, size, include_obsolete, relation):
    """Shared body of the two v2 hierarchy tools; they differ only by the endpoint they hit."""
    if not term_id:
        return _missing(
            "term_id",
            f"Pass the ontology term whose {relation} you want, for example term_id='CL:0000084'.",
        )

    record, ontology, error = _resolve_term(term_id, ontology)
    if error is not None:
        return error
    iri = record.get("iri")
    if not iri:
        return _error(f"OLS returned no IRI for term '{term_id}', so its {relation} cannot be read.")

    size = _page_size(size)
    payload, error = _fetch_json(
        f"{_OLS_V2}/ontologies/{_seg(ontology)}/classes/{_iri_seg(iri)}/{_seg(relation)}",
        params={"page": 0, "size": size, "includeObsoleteEntities": bool(include_obsolete)},
    )
    if error is not None:
        return error

    terms, total = _v2_collection(payload, size)
    return _ok(
        terms,
        total=total,
        showing=len(terms),
        metadata={
            "term_iri": iri,
            "obo_id": record.get("obo_id"),
            "label": _first_text(record.get("label")),
            "ontology": ontology,
            "relation": relation,
            "include_obsolete": bool(include_obsolete),
        },
    )


def ols_get_term_descendants(term_id, ontology=None, size=20):
    """List the full descendant subtree of an ontology term, not just its direct children.

    The difference matters: "T cell" has a handful of direct children and 171 descendants, and the
    question "is this label anywhere under T cell" is the second number, not the first.

    Parameters
    ----------
    term_id : str
        The term, as a CURIE ("CL:0000084"), the underscore form, or a full IRI.
    ontology : str, optional
        The OLS ontology id. Inferred from the identifier's prefix when omitted.
    size : int, optional
        How many descendants to return, 1-200. Default 20. The reported total is the size of the
        whole subtree regardless of how many are returned.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "total": int}``. A total of 0 carries a note,
        because it has more than one cause -- see below.

    Examples
    --------
    >>> sub = ols_get_term_descendants(term_id="CL:0000084", size=5)
    >>> print(sub["total"], [d["label"] for d in sub["data"]])
    """
    if not term_id:
        return _missing(
            "term_id",
            "Pass the ontology term whose subtree you want, for example term_id='CL:0000084'.",
        )

    record, ontology, error = _resolve_term(term_id, ontology)
    if error is not None:
        return error
    iri = record.get("iri")
    if not iri:
        return _error(f"OLS returned no IRI for term '{term_id}', so its descendants cannot be read.")

    size = _page_size(size)
    payload, error = _fetch_json(
        f"{_OLS_V1}/ontologies/{_seg(ontology)}/terms/{_iri_seg(iri)}/descendants",
        params={"size": size},
    )
    if error is not None:
        return error

    rows = ((payload.get("_embedded") or {}).get("terms") or []) if isinstance(payload, dict) else []
    terms = [
        {
            "iri": row.get("iri"),
            "obo_id": row.get("obo_id"),
            "short_form": row.get("short_form"),
            "label": _first_text(row.get("label")),
            "ontology_name": row.get("ontology_name"),
            "is_obsolete": bool(row.get("is_obsolete")),
        }
        for row in rows[:size]
    ]
    total = (payload.get("page") or {}).get("totalElements") if isinstance(payload, dict) else None
    if total is None:
        total = len(terms)

    result = _ok(
        terms,
        total=total,
        showing=len(terms),
        metadata={
            "term_iri": iri,
            "obo_id": record.get("obo_id"),
            "label": _first_text(record.get("label")),
            "ontology": ontology,
        },
    )
    if not total:
        # An empty subtree is ambiguous, and reading it as "no subtypes exist" is how a wrong
        # conclusion gets made. Say which causes are possible.
        result["no_results_note"] = (
            "No descendants were returned. The term may genuinely be a leaf (its record reports "
            f"has_children={record.get('has_children')!r}), it may be obsolete, or its subclasses "
            "may live in a different ontology -- EFO disease terms in particular were retired to "
            "MONDO, so query the replacement ontology instead."
        )
    return result


def ols_get_term_xrefs(term_id, ontology=None):
    """Map an ontology term onto its equivalents in other vocabularies.

    This is the identifier-translation tool for controlled vocabularies, the counterpart to
    TogoID/BridgeDb for genes. It reads the term's obo_xref block, which is how a MONDO disease
    reaches DOID, ICD-10, OMIM, UMLS, MeSH, NCIt, SNOMED and MedDRA -- the mapping you need when a
    cohort is coded in one vocabulary and an atlas in another.

    Parameters
    ----------
    term_id : str
        The term to translate, as a CURIE ("MONDO:0005148", "UBERON:0002107"), the underscore form,
        or a full IRI.
    ontology : str, optional
        The OLS ontology id. Inferred from the identifier's prefix when omitted.

    Returns
    -------
    dict
        ``{"status": "success", "data": {...}}`` where data carries the term's own identity plus
        an xrefs list; each entry has database, id, the combined curie, and a resolver url when
        OLS supplies one. An empty xrefs list is a real answer -- not every ontology records them
        (HP and GO largely do not).

    Examples
    --------
    >>> x = ols_get_term_xrefs(term_id="MONDO:0005148")
    >>> print([e["curie"] for e in x["data"]["xrefs"]])
    >>> print(ols_get_term_xrefs(term_id="UBERON:0002107")["metadata"]["xref_count"])
    """
    if not term_id:
        return _missing(
            "term_id",
            "Pass the ontology term to translate, for example term_id='MONDO:0005148'.",
        )

    record, ontology, error = _resolve_term(term_id, ontology)
    if error is not None:
        return error

    xrefs = []
    for entry in record.get("obo_xref") or []:
        if not isinstance(entry, dict):
            continue
        database = entry.get("database")
        local_id = entry.get("id")
        xrefs.append(
            {
                "database": database,
                "id": local_id,
                "curie": f"{database}:{local_id}" if database and local_id is not None else None,
                "url": entry.get("url"),
                "description": entry.get("description"),
            }
        )

    data = {
        "obo_id": record.get("obo_id") or _curie(term_id),
        "iri": record.get("iri"),
        "label": _first_text(record.get("label")),
        "ontology_name": record.get("ontology_name") or ontology,
        "is_obsolete": bool(record.get("is_obsolete")),
        "xrefs": xrefs,
    }
    result = _ok(
        data,
        metadata={
            "query_id": _curie(term_id),
            "ontology": ontology,
            "xref_count": len(xrefs),
            "databases": sorted({x["database"] for x in xrefs if x["database"]}),
        },
    )
    if not xrefs:
        result["no_results_note"] = (
            "This term records no cross-references. That is a property of the ontology, not a "
            "lookup failure -- HP and GO rarely carry obo_xref. Try the equivalent MONDO or NCIT "
            "term, which usually does."
        )
    return result


def ols_find_similar_terms(term_id, ontology, size=10):
    """Find terms in the same ontology whose names are close to a given term's name.

    Useful when a label almost matches: you have "CD8 T cell" from a marker table, the ontology
    calls it something else, and you want the neighbourhood to choose from.

    Parameters
    ----------
    term_id : str
        The reference term, as a CURIE ("CL:0000084"), the underscore form, or a full IRI.
    ontology : str
        The OLS ontology id to search within, lower case, for example "cl" or "efo".
    size : int, optional
        How many neighbours to return, 1-200. Default 10.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "source_label": str}``. The reference term itself is
        excluded from the list.

    Notes
    -----
    This is lexical similarity over the ontology's own labels and synonyms, not semantic
    similarity. OLS4 has no embedding endpoint, and the returned payload says so, so the result is
    never mistaken for a semantic neighbourhood.

    Examples
    --------
    >>> near = ols_find_similar_terms(term_id="CL:0000084", ontology="cl", size=5)
    >>> print(near["source_label"], [t["label"] for t in near["data"]])
    """
    if not term_id or not ontology:
        return _missing(
            "term_id and ontology",
            "Pass both, for example term_id='CL:0000084', ontology='cl'.",
            plural=True,
        )

    record, ontology, error = _resolve_term(term_id, ontology)
    if error is not None:
        return error

    label = _first_text(record.get("label"))
    if not label:
        return _error(
            f"OLS holds no label for term '{term_id}' in ontology '{ontology}', so there is nothing "
            "to match neighbours against. Check the identifier with ols_get_term_info."
        )

    size = _page_size(size, default=10)
    payload, error = _fetch_json(
        f"{_OLS_V1}/search",
        params={"q": label, "ontology": ontology, "type": "class", "rows": size + 1},
    )
    if error is not None:
        return error

    docs = ((payload.get("response") or {}).get("docs") or []) if isinstance(payload, dict) else []
    iri = record.get("iri")
    neighbours = [
        {
            "iri": doc.get("iri"),
            "obo_id": doc.get("obo_id"),
            "label": doc.get("label"),
            "description": doc.get("description") or [],
            "ontology_name": doc.get("ontology_name"),
        }
        for doc in docs
        if doc.get("iri") != iri
    ][:size]

    return _ok(
        neighbours,
        source_label=label,
        showing=len(neighbours),
        metadata={
            "term_iri": iri,
            "obo_id": record.get("obo_id"),
            "ontology": ontology,
            "method": "lexical search over ontology labels and synonyms; OLS4 exposes no semantic similarity endpoint",
        },
    )


def ols_get_ontology_info(ontology_id):
    """Fetch metadata for one ontology: title, description, version, size, homepage.

    Worth a call before trusting a hierarchy result, because "how many classes does this ontology
    actually contain" and "when was it last loaded" decide whether an empty answer means absence or
    staleness.

    Parameters
    ----------
    ontology_id : str
        The OLS ontology id, lower case: "efo", "mondo", "hp", "go", "cl", "uberon", "chebi".

    Returns
    -------
    dict
        ``{"status": "success", "data": {...}}`` with ontology_id, title, preferred_prefix,
        description, homepage, version, number_of_classes and number_of_entities.

    Examples
    --------
    >>> info = ols_get_ontology_info(ontology_id="cl")
    >>> print(info["data"]["title"], info["data"]["number_of_classes"])
    """
    if not ontology_id:
        return _missing(
            "ontology_id",
            "Pass the OLS ontology abbreviation, for example ontology_id='efo'.",
        )

    ontology_id = str(ontology_id).strip().lower()
    payload, error = _fetch_json(f"{_OLS_V2}/ontologies/{_seg(ontology_id)}")
    if error is not None:
        return error
    if not isinstance(payload, dict) or not payload.get("ontologyId"):
        return _error(
            f"OLS returned no ontology record for '{ontology_id}'. List the available ones with ols_search_ontologies."
        )
    return _ok(_ontology_summary(payload), metadata={"query_id": ontology_id})


def ols_search_ontologies(search=None, page=0, size=20):
    """List or search the ontologies OLS serves.

    Call this when you do not know which vocabulary owns a concept. There are 280+ of them and the
    right one is often not the obvious one -- anatomy is UBERON, not "anatomy".

    Parameters
    ----------
    search : str, optional
        Free-text filter over ontology titles and descriptions, for example "disease" or "cell".
        Omit to list everything, page by page.
    page : int, optional
        Zero-based page number. Default 0.
    size : int, optional
        Ontologies per page, 1-200. Default 20.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "total": int, "pagination": {...}}``.

    Examples
    --------
    >>> found = ols_search_ontologies(search="disease", size=10)
    >>> print([(o["ontology_id"], o["title"]) for o in found["data"]])
    """
    size = _page_size(size)
    try:
        page = max(0, int(page))
    except (TypeError, ValueError):
        page = 0

    params = {"page": page, "size": size}
    if search:
        params["search"] = search

    payload, error = _fetch_json(f"{_OLS_V2}/ontologies", params=params)
    if error is not None:
        return error

    elements = payload.get("elements") if isinstance(payload, dict) else None
    if not isinstance(elements, list):
        elements = []
    ontologies = [_ontology_summary(item) for item in elements]

    return _ok(
        ontologies,
        total=payload.get("totalElements", len(ontologies)),
        showing=len(ontologies),
        pagination={
            "page": page,
            "size": size,
            "total_pages": payload.get("totalPages"),
            "total_items": payload.get("totalElements"),
        },
        metadata={"search": search},
    )


def efo_id_for_disease_name(disease, rows=1):
    """Look up the ontology identifier for a disease named in plain English.

    Scoped to EFO, which is the vocabulary GWAS Catalog, Open Targets and ArrayExpress annotate
    with, and which imports MONDO for its disease branch -- so a search here answers with whichever
    of the two actually owns the term.

    Parameters
    ----------
    disease : str
        The disease name or abbreviation, for example "asthma", "PCOS", "type 2 diabetes".
    rows : int, optional
        How many candidates to return, 1-200. Default 1, which returns the single best match.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...]}`` -- always a list, even for rows=1, so the caller
        does not have to branch on the shape. Each entry carries efo_id, curie, label, ontology and
        exact_match.

    Notes
    -----
    OLS ranks hits on the whole indexed record, description text included, which puts the wrong
    term first for short names. Measured live: "PCOS" ranks EFO:0005187 "C-peptide measurement"
    -- whose *definition* mentions PCOS in passing -- above MONDO:0008487 "polycystic ovary
    syndrome", which lists PCOS as an exact synonym. So candidates are fetched several deep and any
    whose own label or exact synonym equals the query is promoted first, and the promotion is
    reported in the exact_match flag rather than applied silently.

    Examples
    --------
    >>> hit = efo_id_for_disease_name(disease="PCOS")
    >>> print(hit["data"][0]["curie"], hit["data"][0]["label"], hit["data"][0]["exact_match"])
    >>> print(efo_id_for_disease_name(disease="asthma", rows=5)["data"])
    """
    if not disease:
        return _missing(
            "disease",
            "Pass the disease name to look up, for example disease='type 2 diabetes'.",
        )

    rows = _page_size(rows, default=1)
    payload, error = _fetch_json(
        f"{_OLS_V1}/search",
        params={"ontology": "efo", "q": disease, "rows": max(rows, _RERANK_FETCH)},
    )
    if error is not None:
        return error

    docs = ((payload.get("response") or {}).get("docs") or []) if isinstance(payload, dict) else []
    if not docs:
        return _ok(
            [],
            no_results_note=(
                f"EFO has no term matching '{disease}'. Try a fuller clinical name, or search "
                "across every ontology with ols_search_terms."
            ),
            metadata={"query": disease},
        )

    target = str(disease).strip().lower()

    def _is_exact(doc):
        if str(doc.get("label") or "").strip().lower() == target:
            return True
        return any(str(s).strip().lower() == target for s in doc.get("exact_synonyms") or [])

    exact = [doc for doc in docs if _is_exact(doc)]
    rest = [doc for doc in docs if not _is_exact(doc)]
    ordered = (exact + rest)[:rows]

    results = [
        {
            "efo_id": doc.get("short_form"),
            "curie": doc.get("obo_id") or _curie(doc.get("short_form") or ""),
            "label": doc.get("label"),
            "ontology": doc.get("ontology_name"),
            "iri": doc.get("iri"),
            "exact_match": _is_exact(doc),
        }
        for doc in ordered
    ]

    result = _ok(
        results,
        showing=len(results),
        metadata={
            "query": disease,
            "candidates_considered": len(docs),
            "exact_matches_found": len(exact),
        },
    )
    if exact and docs and not _is_exact(docs[0]):
        # The rerank changed the answer. Say so -- a caller comparing this against a raw OLS query
        # should be able to see why the two disagree.
        result["rerank_note"] = (
            f"The service ranked '{docs[0].get('label')}' first, but "
            f"'{exact[0].get('label')}' matches '{disease}' exactly as a label or synonym, so it "
            "was promoted."
        )
        logger.info(
            "EFO disease rerank promoted %s over %s for %r",
            exact[0].get("short_form"),
            docs[0].get("short_form"),
            disease,
        )
    elif not exact:
        result["match_note"] = (
            f"No candidate's label or exact synonym equals '{disease}', so these are the service's "
            "own relevance ranking. Check the label before relying on the identifier."
        )
    return result
