"""Model-organism genetics across the Alliance of Genome Resources.

One API, ``www.alliancegenome.org``, in front of eight long-running organism databases that each
curate their own literature by hand: RGD (rat), MGI (mouse), ZFIN (zebrafish), FlyBase, WormBase,
SGD (yeast), Xenbase (frog) and HGNC (human gene nomenclature). The Alliance harmonises their
annotations onto shared ontologies -- Disease Ontology for disease, the species phenotype ontologies
for phenotype, MI for interaction evidence -- so a question can cross species without the caller
reconciling eight schemas.

What it is for, in this system: a spatial-omics result names genes, and the useful next question is
usually comparative rather than descriptive. Which orthologue carries this function in the organism
the experiment actually used? Has anyone already made a mutant and recorded a phenotype? Is the gene
annotated to the disease the tissue came from? Those are curated-literature questions, and this is
the catalogue that answers them across species rather than within one.

Use it for orthology, phenotype, allele/model and disease-gene questions that span organisms. Do not
use it for expression levels (``expression_atlases``), for identifier translation between namespaces
(``gene_identifiers``), or for anything human-only where HGNC or Ensembl is more direct.

Five properties of this API shape every function below. All five were measured against the live
service, and three of them return a confident wrong or empty answer rather than an error -- which is
the failure mode this module exists to prevent.

**F41 -- a bare number must not be guessed into a prefix.** Upstream's ``_normalize_gene_id`` maps
any bare numeric id of five digits or more to ``MGI:<n>``. Measured: ``620474`` is the Rat Genome
Database's *Sox9* -- ``RGD:620474`` resolves to ``Sox9``, while the guessed ``MGI:620474`` returns
HTTP 400. The convenience converts a resolvable identifier into an error, and because of F43 it can
convert one into a confident empty answer instead. No prefix is inferred here: an unprefixed id is
refused with the list of real prefixes and a pointer to the search call.

**F42 -- the server's explanation is worth more than its status code.** Alliance signals "no such
record" with HTTP **400**, not 404, and the body says exactly what was wrong:
``{"statusCode":400,"errors":["No gene found with ID: MGI:99999999"],"statusCodeName":"Bad
Request"}``. Upstream's handler reports only ``"Alliance API HTTP error: 400"``, discarding both the
reason and the distinction between a malformed id and an absent one. Every error here unwraps the
``errors`` list and reports what the server actually said.

**F43 -- a gene that does not exist produces an empty list, not an error.** Measured on every gene
sub-endpoint -- ``/phenotypes``, ``/orthologs``, ``/paralogs``, ``/molecular-interactions``,
``/alleles``, ``/models`` -- a nonexistent gene id returns HTTP **200** with ``total: 0`` and *no*
``results`` key at all. Upstream reports that as ``status: success`` with zero rows, which is
byte-identical to a real gene that simply has no annotations of that kind. Compounded with F41, a
bare RGD number becomes ``MGI:<n>`` and then reports "0 phenotypes" for a gene that exists. Every
sub-endpoint call here resolves the gene first (one ~0.3 s request) so "no such gene" and "no
annotations" are different answers.

**F44 -- ``moderate`` is an advertised stringency that never returns anything.** Upstream advertises
``stringency`` as one of ``stringent``/``moderate``/``all``. Measured across six genes spanning six
members, ``filter.stringency=moderate`` returns **zero** rows every time, and an unrecognised value
returns zero as well -- so a typo and a real empty result are indistinguishable. The row-level flag
``moderateFilter`` does exist and is true for 3 of *Pax6*'s 19 orthologues, so moderate-confidence
orthologues are reachable only by asking for ``all`` and filtering client-side, which is what
``model_organisms_get_gene_orthologs`` does. An unrecognised stringency is refused, not sent.

**F45 -- the autocomplete endpoint is capped at ten rows and ignores ``limit``.** Upstream searches
through ``/search_autocomplete`` and advertises ``limit`` from 1 to 50, computing a fivefold fetch
buffer so that client-side species filtering has headroom. Measured: the endpoint returns exactly ten
rows for ``limit`` of 5, 10, 50 and 200, and ignores ``page``, ``offset``, ``rows`` and ``size`` --
so the buffer has no headroom and any ``limit`` above ten is unreachable. The ``/search`` endpoint
honours both ``limit`` and ``offset``, reports a real total (2,461 for ``pax6`` against
autocomplete's ten), and populates ``species`` on every row, which autocomplete leaves null. This
module searches through ``/search``. It is slower -- roughly 6-8 s against autocomplete's 0.9 s, and
it returns 504 under concurrent load -- so the default limit is small and a gateway timeout is
reported as a retryable condition rather than as "no results".

**F46 -- the allele rows come back with no symbol and no identifier.** The shipped
``get_alleles_and_models`` tool reads each allele row as ``r.get("symbol") or r.get("symbolText")``
and its id as ``allele.get("curie") or r.get("id")``. Measured against 675 rows of mouse *Pax6*:
no row carries ``symbol``, ``symbolText`` or ``id`` at any level, and a curated allele carries
``primaryExternalId`` rather than ``curie``. The symbol is at
``row["allele"]["alleleSymbol"]["displayText"]``. A curated allele therefore returns with *both*
fields null -- an unidentifiable row. Upstream's own ``allele_detail`` handler reads
``allele.get("alleleSymbol", {}).get("displayText")`` correctly a few hundred lines away in the same
file, so this is one path out of step with its neighbour rather than a misread of the API.

**F47 -- dbSNP variants outrank curated alleles, and no filter separates them.** ``/gene/{id}/alleles``
mixes hand-curated named alleles with imported dbSNP variants and returns the variants first.
Measured: mouse *Pax6* has 675 rows, 607 of them bare ``rs`` accessions and 68 curated alleles;
human *TP53* has 2,920 rows and **not one** curated allele; human *BRCA1* has 12,596, also none;
mouse *Trp53* has 719 of which 362 are curated. So ``limit=20`` on *Pax6* returns twenty dbSNP
accessions and none of the alleles the caller meant. Six spellings of a server-side filter --
``filter.alterationType``, ``alterationType``, ``filter.alleleType``, ``filter.category``,
``sortBy`` -- are all accepted and all ignored, returning the identical first page.

The split can only be made client-side, and one measured property makes that affordable: the two
kinds are **block-sorted, variants first**, never interleaved. Checked on six genes across five
member databases -- *Pax6* (607 then 68), *Trp53* (357 then 362), *white* (129 then 932), *unc-54*
(124 then 383), *TP53* and *BRCA1* (all variants) -- the last variant always precedes the first
curated allele, and only variant rows carry ``alterationTypeSortOrder``. Curated alleles are
therefore the *tail* of the list, so ``include="alleles"`` reads the last page rather than the first
and never walks the middle. That turns *BRCA1* from a 12,596-row, ~90 MB scan into one page of 596.

**F48 -- the disease-gene relation is dropped, and the obvious substitute inverts curated negatives.**
Upstream reads ``r.get("associationType")``; that key is absent from all 500 rows measured, so the
field is always null. The relation is instead in ``generatedRelationString``, and it is the
scientifically load-bearing part of the row: of 500 human rows for DOID:9351, 268 are
``is_implicated_in``, 199 ``is_marker_for``, **32 ``is_not_implicated_in`` and 1 ``is_not_marker_for``**
-- curated statements that a gene is *not* involved. The neighbouring ``relation.name`` field reports
300 and 200, exactly the positive counts plus the negatives, so reading it silently reverses the sign
on 33 rows in 500. This module reads ``generatedRelationString``. Two further distinctions are
carried for the same reason: ``/disease/{id}/genes`` returns the whole Disease Ontology subtree --
only 19 of the first 200 rows for "diabetes mellitus" are annotated to DOID:9351 itself, 107 to type-2
and 55 to type-1 -- and ``viaOrthologyOrder`` marks rows inferred from a human annotation rather than
curated in the organism, which is 409 of the first 500 mouse rows.

Two upstream behaviours are *correct* and were kept rather than re-derived: ``_get_gene_detail``
already unwraps the record from the top-level ``gene`` key and already unwraps Alliance's
``{formatText, displayText}`` label objects, and ``_search_genes`` already documents that
``category=gene`` returns nothing and that the identifier moved from ``primaryKey`` to ``curie``.
Both are reproduced here with their reasoning intact.

There is no machine-readable schema to discover the member list from: ``/api/species``,
``/api/swagger.json``, ``/api/openapi.json`` and ``/api/release`` all return 404 (measured). The
prefix table below is therefore a constant, and ``model_organisms_list_members`` re-resolves one
representative gene per member on every call so the table is checked against the live service rather
than recited.

No credential is required and every call is a read-only GET.

Adapted from ToolUniverse -- https://github.com/mims-harvard/ToolUniverse -- at commit
``f075c2a75e8b35ae5dbb220d48d4e87e980388b1``, Apache-2.0, Copyright [2025] [ToolUniverse team].
CHANGED BY SPATIALOMICSGYM as Apache-2.0 section 4(b) requires: the eight tools of
``alliance_genome_tools.json`` were re-expressed as the functions below against our house
conventions; the ``AllianceGenomeTool`` class, its ``endpoint_type`` dispatch and its
``_normalize_gene_id`` heuristic were not carried across; ``model_organisms_list_members`` is ours,
not upstream's, and exists because F41's prefix trap had no discovery call. Every value that
reaches a URL path segment is percent-encoded through ``_seg``, as in the other five vendored
modules: the validators here check an identifier's *prefix*, not the rest of it, and an
unencoded ``/`` in the remainder would read as path structure rather than as part of the id.

See ``VENDORING.md`` for the file-by-file record and the re-sync procedure.
"""

from __future__ import annotations

import difflib
import logging
import re
from urllib.parse import quote

from spatialomicsgym.utils.http_client import HttpError, request_json

logger = logging.getLogger(__name__)

_ALLOWED_HOSTS = ("www.alliancegenome.org",)

_BASE = "https://www.alliancegenome.org/api"

#: Association endpoints answer in well under a second; ``/search`` is measurably slower (6-8 s
#: uncontended) and 504s under load, so it gets its own ceiling rather than the module default.
_TIMEOUT = 30.0
_SEARCH_TIMEOUT = 75.0

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100
_SEARCH_DEFAULT_LIMIT = 10
#: Measured: ``/search`` serves 50 rows uncontended and returns 504 above that under load. The cap
#: is the largest size observed to succeed, not the largest the server will accept.
_SEARCH_MAX_LIMIT = 50

_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")

#: Each member's identifier prefix, the organism it curates, and a gene that resolves today. The
#: representative ids are re-checked live by ``model_organisms_list_members`` rather than trusted.
#: ``Xenbase:`` is the only working frog prefix -- ``XB:XB-GENE-865965`` and the bare
#: ``XB-GENE-865965`` both return HTTP 400 (measured).
_MEMBERS = (
    ("HGNC", "Homo sapiens", "human", "HGNC:11998", "TP53"),
    ("MGI", "Mus musculus", "mouse", "MGI:97490", "Pax6"),
    ("RGD", "Rattus norvegicus", "rat", "RGD:620474", "Sox9"),
    ("ZFIN", "Danio rerio", "zebrafish", "ZFIN:ZDB-GENE-990415-8", "pax2a"),
    ("FB", "Drosophila melanogaster", "fruit fly", "FB:FBgn0003996", "w"),
    ("WB", "Caenorhabditis elegans", "roundworm", "WB:WBGene00006789", "unc-54"),
    ("SGD", "Saccharomyces cerevisiae", "budding yeast", "SGD:S000005739", "SAS5"),
    ("Xenbase", "Xenopus", "frog", "Xenbase:XB-GENE-865965", "lhx5.S"),
)

_PREFIXES = tuple(m[0] for m in _MEMBERS)

#: Accepted stringency values. ``moderate`` is accepted here but never sent as a query value -- see
#: F44: the server answers it with zero rows for every gene tested, so it is served by asking for
#: ``all`` and filtering on the row's own ``moderateFilter`` flag.
_STRINGENCIES = ("stringent", "moderate", "all")
_ALLELE_MODES = ("alleles", "variants", "all")
_ALLELE_SCAN = 1000  # rows read when curated alleles must be separated from dbSNP variants (F47)
_ORTHOLOG_SCAN = 1000  # 'all' rows read, page by page, when moderate orthologues are filtered client-side (F44)


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
    """One percent-encoded URL path segment.

    Alliance ids carry a colon (``MGI:97490``) and disease ids a prefix (``DOID:9351``), and the
    validators above check the prefix, not the rest of the string. Encoding the whole segment is
    what keeps a value the caller supplied from reading as path structure.
    """
    return quote(str(value), safe="")


def _clean(text):
    return _WS_RE.sub(" ", str(text or "")).strip()


def _norm(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")


def _strip_html(text):
    """Phenotype and disease statements carry markup for italicised gene symbols."""
    return _clean(_TAG_RE.sub("", str(text or "")))


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


def _limit(size, default=_DEFAULT_LIMIT, maximum=_MAX_LIMIT):
    """Clamp a caller's row count.

    The server has no ceiling of its own: measured, ``/disease/DOID:9351/genes?limit=20000`` returns
    all 12,740 annotations in one response, which would put a megabyte of rows into the agent's
    context. Upstream clamps at 100 and so does this.
    """
    try:
        return max(1, min(int(size), maximum))
    except (TypeError, ValueError):
        return default


def _text(obj):
    """Unwrap Alliance's ``{formatText, displayText}`` label objects to a plain string.

    Symbols, names and allele designations are all served as this two-field object so a client can
    choose between markup and plain text. A reader that treats one as a string gets ``None``.
    """
    if isinstance(obj, dict):
        return obj.get("displayText") or obj.get("formatText")
    return obj


def _alliance_get(path, params=None, *, timeout=_TIMEOUT):
    """``(payload, None)`` on success, ``(None, error_payload)`` on failure.

    F42. Alliance reports a missing record as HTTP 400 with a usable body -- ``{"errors":["No gene
    found with ID: ..."]}``. That message names the identifier it could not resolve, which is the
    one thing the caller needs, so it is unwrapped and surfaced instead of the status code.
    """
    try:
        return request_json(
            _BASE + path,
            allowed_hosts=_ALLOWED_HOSTS,
            params=params or None,
            headers={"Accept": "application/json"},
            timeout=timeout,
        ), None
    except HttpError as exc:
        detail = exc.detail
        server_said = _server_message(exc.body)
        if server_said:
            detail = server_said
        return None, _error(detail, retryable=bool(exc.status) and exc.status >= 500)


def _server_message(body):
    """The ``errors`` list out of an Alliance error body, joined, or ``None``."""
    if not body:
        return None
    match = re.search(r'"errors"\s*:\s*\[(.*?)\]', str(body), re.DOTALL)
    if not match:
        return None
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1))
    joined = "; ".join(p.replace('\\"', '"') for p in parts if p.strip())
    return joined or None


def _check_gene_id(gene_id):
    """``(cleaned, None)`` for a prefixed Alliance gene id, else ``(None, error)``.

    F41. Upstream guesses: any bare number of five digits or more becomes ``MGI:<n>``. Measured,
    ``620474`` is RGD's *Sox9*, so the guess turns a resolvable id into ``MGI:620474`` and an HTTP
    400 -- or, through F43, into a confident empty list. Nothing is inferred here.
    """
    text = _clean(gene_id)
    if not text:
        return None, _missing(
            "gene_id",
            "Pass a prefixed Alliance gene id such as 'MGI:97490' or 'HGNC:11998'; "
            "model_organisms_search_genes() finds one from a symbol.",
        )
    if ":" not in text:
        hint = ""
        if text.isdigit():
            hint = (
                f" A bare number is ambiguous: '{text}' could belong to any member, and guessing "
                "a prefix is how a resolvable id becomes an error -- RGD:620474 and MGI:620474 are "
                "not the same gene, and only one of them exists."
            )
        return None, _error(
            f"'{gene_id}' has no database prefix.{hint} Alliance ids are '<PREFIX>:<id>' where "
            f"PREFIX is one of {', '.join(_PREFIXES)}. Use model_organisms_search_genes() to get a "
            "prefixed id from a gene symbol, or model_organisms_list_members() to see each "
            "member's prefix and a worked example."
        )
    prefix = text.split(":", 1)[0]
    known = {p.lower(): p for p in _PREFIXES}
    if prefix.lower() not in known:
        close = _suggest(prefix, _PREFIXES)
        return None, _error(
            f"'{prefix}' is not an Alliance member prefix. "
            + (f"Closest: {', '.join(close)}. " if close else "")
            + f"The members are {', '.join(_PREFIXES)}. Note that frog ids use 'Xenbase:' -- "
            "'XB:' and a bare 'XB-GENE-...' are both rejected by the server."
        )
    return text, None


def _resolve_gene(gene_id):
    """``(record, None)`` for a gene that exists, else ``(None, error)``.

    F43. Every gene sub-endpoint answers a nonexistent id with HTTP 200, ``total: 0`` and no
    ``results`` key -- indistinguishable from a real gene with nothing annotated. Resolving the gene
    first costs one request of about 0.3 s and makes the two cases different answers.
    """
    checked, err = _check_gene_id(gene_id)
    if err:
        return None, err
    payload, err = _alliance_get(f"/gene/{_seg(checked)}")
    if err:
        return None, err
    record = (payload or {}).get("gene") or payload or {}
    if not isinstance(record, dict) or not record.get("primaryExternalId"):
        return None, _error(
            f"Alliance has no gene {checked}. The prefix is valid, so this is a wrong or retired "
            "identifier rather than a malformed one. Find the current id with "
            "model_organisms_search_genes()."
        )
    return record, None


def _gene_label(record):
    """The compact ``{gene_id, symbol, name, species}`` every function echoes back."""
    taxon = record.get("taxon") or {}
    return {
        "gene_id": record.get("primaryExternalId"),
        "symbol": _text(record.get("geneSymbol")),
        "name": _text(record.get("geneFullName")),
        "species": taxon.get("name"),
    }


def _sub_endpoint(gene_id, path, params, *, what):
    """Resolve the gene, then fetch one of its association endpoints.

    Returns ``(rows, total, label, None)`` or ``(None, None, None, error)``. The resolve step is
    what separates "no such gene" from "no {what}" -- see F43.
    """
    record, err = _resolve_gene(gene_id)
    if err:
        return None, None, None, err
    label = _gene_label(record)
    payload, err = _alliance_get(f"/gene/{_seg(label['gene_id'])}/{_seg(path)}", params)
    if err:
        return None, None, None, err
    data = payload or {}
    return (data.get("results") or []), _as_int(data.get("total"), 0), label, None


def _pubmed_ids(row):
    """PMIDs off an annotation row, which carries them as ``{referencedCurie, displayName}``."""
    out = []
    for pub in row.get("pubmedPublications") or []:
        curie = (pub or {}).get("referencedCurie") or (pub or {}).get("displayName")
        if curie and curie not in out:
            out.append(curie)
    return out


# ------------------------------------------------------------------------------------- discovery


def model_organisms_list_members(verify=True):
    """The eight member databases, their id prefixes, and a worked example for each.

    Every other function in this module is keyed on a prefixed identifier, and picking the wrong
    prefix is the single most common way to get a wrong answer rather than an error (F41, F43). This
    is the call that makes the prefix visible before it is guessed.

    Alliance serves no machine-readable member list -- ``/api/species``, ``/api/swagger.json``,
    ``/api/openapi.json`` and ``/api/release`` all return 404 (measured) -- so the table is a
    constant. To stop it from drifting into a comfortable fiction, ``verify`` re-resolves each
    member's representative gene against the live service and reports what came back.

    Parameters
    ----------
    verify : bool, optional
        Re-resolve every representative gene live and report whether each member is answering.
        Default True, and about 2-3 s for the eight calls. Pass False for the table alone.

    Returns
    -------
    dict
        ``data.members`` with ``prefix``, ``organism``, ``common_name``, ``example_gene_id`` and
        ``example_symbol``; when verified, each row also carries ``resolved`` and the ``symbol`` the
        service actually returned, and ``data.release`` names the data release that answered.

    Examples
    --------
    >>> print(model_organisms_list_members())  # doctest: +SKIP
    """
    members = [
        {
            "prefix": prefix,
            "organism": organism,
            "common_name": common,
            "example_gene_id": example,
            "example_symbol": symbol,
        }
        for prefix, organism, common, example, symbol in _MEMBERS
    ]

    data = {"members": members, "member_count": len(members), "verified": bool(verify)}

    # verify=False is documented as the offline table alone, so it returns before any request. The
    # releaseInfo fetch ran first, and offline that meant request_json's retries for a call the
    # caller had opted out of (hunt 2026-09-30, uT4-genomics-25).
    if not verify:
        data["note"] = (
            "Prefixes reported from the module's table without checking them. Pass verify=True to "
            "re-resolve each example gene against the live service."
        )
        return _ok(data)

    release, err = _alliance_get("/releaseInfo")
    if not err and isinstance(release, dict):
        data["release"] = {
            "version": release.get("releaseVersion"),
            "date": release.get("releaseDate"),
        }

    unresolved = []
    for row in members:
        record, err = _resolve_gene(row["example_gene_id"])
        if err:
            row["resolved"] = False
            row["error"] = err.get("error")
            unresolved.append(row["prefix"])
        else:
            row["resolved"] = True
            row["symbol"] = _text(record.get("geneSymbol"))
            row["species"] = (record.get("taxon") or {}).get("name")
    data["unresolved"] = unresolved
    if unresolved:
        data["note"] = (
            f"{len(unresolved)} of {len(members)} members did not resolve their example gene "
            f"({', '.join(unresolved)}). That is a change in the service or in this table, not a "
            "problem with your query -- the other members are unaffected."
        )
    return _ok(data)


# ---------------------------------------------------------------------------------------- search


def model_organisms_search_genes(query, limit=_SEARCH_DEFAULT_LIMIT, offset=0, species=None):
    """Find genes by symbol, name or synonym across all eight member databases.

    This is the way to turn a gene symbol into the prefixed identifier the rest of the module needs.
    A symbol usually matches once per species, so ``pax6`` returns the rat, human, mouse, frog and
    zebrafish genes together, each with its own prefixed id.

    Searches run against ``/search`` rather than ``/search_autocomplete`` (F45): autocomplete is
    hard-capped at ten rows and ignores ``limit``, ``page`` and ``offset`` entirely, while ``/search``
    honours ``limit`` and ``offset``, reports the real number of matches, and fills in ``species``,
    which autocomplete leaves null. The cost is latency -- roughly 6-8 s against autocomplete's 0.9 s.

    Alliance matches symbols and synonyms, not descriptive prose: ``"insulin"`` returns diseases and
    GO terms but no gene, while the symbol ``"INS"`` returns the genes. When nothing matched but
    other entity types did, the result says which, rather than looking like an empty database.

    Parameters
    ----------
    query : str
        Gene symbol, name fragment or synonym, e.g. ``"pax6"``, ``"TP53"``, ``"unc-54"``.
    limit : int, optional
        Genes to return, 1-50. Default 10. Above 50 the endpoint returns 504 under load.
    offset : int, optional
        Rows to skip, for paging through a large match set. Default 0.
    species : str, optional
        Keep only genes from this organism, matched against the binomial or the common name, e.g.
        ``"Mus musculus"`` or ``"mouse"``. Filtering happens after the fetch, so widen ``limit``
        when filtering hard.

    Returns
    -------
    dict
        ``data.genes`` with ``gene_id``, ``symbol``, ``name`` and ``species``; ``data.total_matched``
        is the service's own count of everything matching, which is usually far larger than the page
        returned.

    Examples
    --------
    >>> print(model_organisms_search_genes("pax6", species="mouse"))  # doctest: +SKIP
    """
    text = _clean(query)
    if not text:
        return _missing(
            "query",
            "Pass a gene symbol, name fragment or synonym, e.g. 'pax6'.",
        )
    rows = _limit(limit, default=_SEARCH_DEFAULT_LIMIT, maximum=_SEARCH_MAX_LIMIT)
    start = max(0, _as_int(offset, 0))

    payload, err = _alliance_get(
        "/search",
        {"q": text, "limit": rows, "offset": start},
        timeout=_SEARCH_TIMEOUT,
    )
    if err:
        if err.get("retryable"):
            err["error"] = (
                f"The Alliance search index did not answer in time for '{text}'. It is measurably "
                "slower than the rest of the API and returns a gateway error under load; this is a "
                "transient condition, not an empty result. Retry, or lower limit."
            )
        return err

    data = payload or {}
    results = data.get("results") or []
    # Upstream documents this and it is still true: category=gene returns zero results as a query
    # parameter, and the identifier lives in `curie` (it was `primaryKey`). The row's own category
    # field is the working discriminator.
    gene_rows = [r for r in results if (r or {}).get("category") == "gene_search_result"]

    genes = [
        {
            "gene_id": r.get("curie"),
            "symbol": r.get("symbol"),
            "name": r.get("name"),
            "species": r.get("species"),
        }
        for r in gene_rows
    ]

    wanted = _clean(species)
    if wanted:
        # The binomial and the common name both work, as documented: a needle naming a member by
        # either is widened to both before matching. A bare substring match on the row's species
        # (a binomial) refused 'mouse', 'human' and 'zebrafish', and let 'rat' through only because
        # it is inside 'rattus' (hunt 2026-09-30, uT4-genomics-24).
        needle = _norm(wanted)
        aliases = {needle}
        for _prefix, organism, common, _example, _symbol in _MEMBERS:
            if needle in (_norm(organism), _norm(common)):
                aliases |= {_norm(organism), _norm(common)}
        kept = [g for g in genes if any(a in _norm(g.get("species")) for a in aliases)]
        if not kept and genes:
            seen = sorted({g.get("species") for g in genes if g.get("species")})
            return _error(
                f"No '{wanted}' gene among the {len(genes)} matches for '{text}'. This page carried "
                + (f"{', '.join(seen)}. " if seen else "no species labels. ")
                + "Raise limit or drop species to see the rest; "
                "model_organisms_list_members() lists the eight organisms covered."
            )
        genes = kept

    out = {
        "query": text,
        "genes": genes,
        "returned": len(genes),
        "total_matched": _as_int(data.get("total"), 0),
        "offset": start,
    }
    if wanted:
        out["species_filter"] = wanted

    if not genes and results:
        other = sorted(
            {
                str((r or {}).get("category", "")).replace("_search_result", "")
                for r in results
                if (r or {}).get("category")
            }
        )
        out["note"] = (
            f"No gene matched '{text}', but {', '.join(other)} records did. Alliance matches gene "
            "symbols and synonyms rather than descriptive names -- search the symbol ('INS') rather "
            "than the concept ('insulin')."
        )
    return _ok(out)


# ------------------------------------------------------------------------------------------ gene


def model_organisms_get_gene(gene_id):
    """Full curated record for one gene: symbol, name, species, location, synonyms, cross-references.

    Parameters
    ----------
    gene_id : str
        Prefixed Alliance gene id, e.g. ``"MGI:97490"``, ``"HGNC:11998"``, ``"WB:WBGene00006789"``.
        No prefix is inferred from a bare number (F41); use
        :func:`model_organisms_search_genes` to get one from a symbol.

    Returns
    -------
    dict
        ``data`` with ``gene_id``, ``symbol``, ``name``, ``species``, ``gene_type``, ``synonyms``,
        ``genomic_location``, ``cross_references`` and ``data_provider`` -- the member database that
        curated the record, which is not always the one the prefix suggests.

    Examples
    --------
    >>> print(model_organisms_get_gene("MGI:97490"))  # doctest: +SKIP
    """
    record, err = _resolve_gene(gene_id)
    if err:
        return err

    locations = record.get("geneGenomicLocationAssociations") or []
    first = locations[0] if isinstance(locations, list) and locations else {}
    location = {}
    if isinstance(first, dict):
        assembly = first.get("genomeAssembly") or {}
        location = {
            "chromosome": (first.get("chromosome") or {}).get("name"),
            "start": first.get("start"),
            "end": first.get("end"),
            "strand": first.get("strand"),
            "assembly": assembly.get("primaryExternalId") if isinstance(assembly, dict) else None,
        }
        location = {k: v for k, v in location.items() if v is not None}

    provider = record.get("dataProvider") or {}
    out = dict(_gene_label(record))
    out.update(
        {
            "gene_type": _text((record.get("geneType") or {}).get("name"))
            or (record.get("geneType") or {}).get("name"),
            "synonyms": [_text(s) for s in (record.get("geneSynonyms") or []) if _text(s)],
            "secondary_ids": [_text(s) for s in (record.get("geneSecondaryIds") or []) if _text(s)],
            "genomic_location": location or None,
            "data_provider": {
                "abbreviation": provider.get("abbreviation"),
                "name": provider.get("fullName"),
            },
            "cross_references": [
                {
                    "curie": x.get("referencedCurie"),
                    "name": x.get("displayName"),
                }
                for x in (record.get("crossReferences") or [])[:15]
                if isinstance(x, dict)
            ],
        }
    )
    return _ok(out)


def model_organisms_get_gene_phenotypes(gene_id, limit=_DEFAULT_LIMIT, page=1):
    """Curated phenotype annotations for one gene, with the publications supporting each.

    These are hand-curated from the literature by the member database that owns the organism, which
    is what makes them worth more than a text search: an annotation means a curator read the paper
    and decided the phenotype was demonstrated.

    A gene that does not exist returns an error here rather than an empty list -- the API answers a
    bad id with ``total: 0`` on this endpoint (F43), so the gene is resolved first.

    Parameters
    ----------
    gene_id : str
        Prefixed Alliance gene id, e.g. ``"MGI:97490"``.
    limit : int, optional
        Annotations to return, 1-100. Default 20. *Pax6* alone has 184.
    page : int, optional
        1-based page number. Default 1. Page 0 is rejected by the server.

    Returns
    -------
    dict
        ``data.phenotypes`` with ``phenotype`` and ``pubmed_ids``; ``data.total_annotations`` is the
        count across all pages, and ``data.gene`` echoes the resolved gene so a mistaken id is
        visible in the result.

    Examples
    --------
    >>> print(model_organisms_get_gene_phenotypes("MGI:97490", limit=5))  # doctest: +SKIP
    """
    rows = _limit(limit)
    which = max(1, _as_int(page, 1))
    results, total, label, err = _sub_endpoint(
        gene_id, "phenotypes", {"limit": rows, "page": which}, what="phenotype annotations"
    )
    if err:
        return err

    phenotypes = [
        {
            "phenotype": _strip_html(r.get("phenotypeStatement")),
            "pubmed_ids": _pubmed_ids(r),
        }
        for r in results
        if isinstance(r, dict)
    ]
    out = {
        "gene": label,
        "phenotypes": phenotypes,
        "returned": len(phenotypes),
        "total_annotations": total,
        "page": which,
    }
    if not phenotypes:
        out["note"] = (
            f"{label['symbol']} ({label['gene_id']}) exists but has no curated phenotype "
            "annotations on this page. This is a real empty result, not an unrecognised id -- the "
            "gene was resolved before the query."
        )
    return _ok(out)


def model_organisms_get_gene_orthologs(gene_id, stringency="stringent", limit=_DEFAULT_LIMIT):
    """Orthologues and paralogues of one gene, with the prediction methods that support each call.

    This is the function that crosses species. Alliance runs a dozen orthology prediction methods --
    Ensembl Compara, PANTHER, OrthoFinder, InParanoid and others -- and reports how many agreed,
    which is the basis of the stringency filter: a *stringent* call is best-score in both directions
    with broad method agreement, while *all* includes single-method low-confidence calls.

    ``stringency="moderate"`` is served differently from the other two, and deliberately (F44). The
    server accepts ``moderate`` as a query value and answers it with zero rows for every gene tested,
    identically to how it answers a misspelt value -- so asking for it directly cannot be
    distinguished from a typo. Moderate-confidence orthologues are instead fetched as ``all`` and
    filtered on each row's own ``moderateFilter`` flag, which does carry the information.

    Parameters
    ----------
    gene_id : str
        Prefixed Alliance gene id, e.g. ``"MGI:97490"``.
    stringency : str, optional
        ``"stringent"`` (default), ``"moderate"`` or ``"all"``. Measured on mouse *Pax6*: stringent
        gives 7 orthologues, moderate 3, all 19.
    limit : int, optional
        Orthologues to return, 1-100. Default 20. Paralogues are capped at the same number.

    Returns
    -------
    dict
        ``data.orthologs`` with ``gene_id``, ``symbol``, ``species``, ``confidence``, ``stringency``,
        ``is_best_score`` and ``methods_matched``; ``data.paralogs`` with the same-species
        duplicates; ``data.ortholog_count`` and ``data.paralog_count`` are the totals before paging.
        For ``moderate``, ``data.ortholog_count_complete`` says whether every ``all`` row was scanned
        (up to 1,000 are).

    Examples
    --------
    >>> print(model_organisms_get_gene_orthologs("MGI:97490", stringency="all"))  # doctest: +SKIP
    """
    wanted = _norm(stringency) or "stringent"
    if wanted not in _STRINGENCIES:
        close = _suggest(stringency, _STRINGENCIES)
        return _error(
            f"'{stringency}' is not a stringency. "
            + (f"Closest: {', '.join(close)}. " if close else "")
            + "Use 'stringent' (best-score both ways, broad method agreement), 'moderate' "
            "(mid-confidence), or 'all'. An unrecognised value is not passed through, because the "
            "server answers one with zero rows and that is indistinguishable from a real empty."
        )

    # F44: 'moderate' as a query value returns nothing for every gene tested, so it is served by
    # fetching 'all' and keeping the rows the server itself flags as moderate.
    sent = "all" if wanted in ("all", "moderate") else "stringent"
    rows = _limit(limit)
    fetch = _MAX_LIMIT if wanted == "moderate" else rows

    record, err = _resolve_gene(gene_id)
    if err:
        return err
    label = _gene_label(record)

    # 'moderate' pages through the 'all' set, up to _ORTHOLOG_SCAN rows. It read page 1 only (100
    # rows) and then reported that window's moderate count as the total, so a gene with more 'all'
    # rows silently lost every moderate orthologue past row 100 (hunt 2026-09-30, uT4-genomics-26).
    ortho_rows, previous = [], None
    page = 1
    while True:
        payload, err = _alliance_get(
            f"/gene/{_seg(label['gene_id'])}/orthologs",
            {"limit": fetch, "page": page, "filter.stringency": sent},
        )
        if err:
            return err
        batch = (payload or {}).get("results") or []
        ortholog_total = _as_int((payload or {}).get("total"), 0)
        if batch == previous:  # a server that ignored `page` would hand back page 1 again
            break
        ortho_rows.extend(batch)
        previous = batch
        if wanted != "moderate" or not batch or len(ortho_rows) >= min(ortholog_total, _ORTHOLOG_SCAN):
            break
        page += 1
    scanned_all, all_total = len(ortho_rows), ortholog_total

    orthologs = []
    for r in ortho_rows:
        if not isinstance(r, dict):
            continue
        body = r.get("geneToGeneOrthologyGenerated") or {}
        if wanted == "moderate" and not body.get("moderateFilter"):
            continue
        other = body.get("objectGene") or {}
        methods = [
            m.get("name") for m in (body.get("predictionMethodsMatched") or []) if isinstance(m, dict) and m.get("name")
        ]
        orthologs.append(
            {
                "gene_id": other.get("primaryExternalId"),
                "symbol": _text(other.get("geneSymbol")),
                "species": (other.get("taxon") or {}).get("name"),
                "confidence": (body.get("confidence") or {}).get("name"),
                "stringency": r.get("stringencyFilter"),
                "is_best_score": (body.get("isBestScore") or {}).get("name"),
                "methods_matched": methods,
                "method_match_count": len(methods),
            }
        )
    if wanted == "moderate":
        ortholog_total = len(orthologs)
    orthologs = orthologs[:rows]

    paralogs = []
    paralog_total = 0
    para_payload, para_err = _alliance_get(f"/gene/{_seg(label['gene_id'])}/paralogs", {"limit": rows, "page": 1})
    if not para_err:
        paralog_total = _as_int((para_payload or {}).get("total"), 0)
        for r in (para_payload or {}).get("results") or []:
            if not isinstance(r, dict):
                continue
            body = r.get("geneToGeneParalogy") or {}
            other = body.get("objectGene") or {}
            paralogs.append(
                {
                    "gene_id": other.get("primaryExternalId"),
                    "symbol": _text(other.get("geneSymbol")),
                    "rank": body.get("rank"),
                    "length": body.get("length"),
                    "similarity": body.get("similarity"),
                    "identity": body.get("identity"),
                }
            )

    out = {
        "gene": label,
        "stringency": wanted,
        "orthologs": orthologs,
        "ortholog_count": ortholog_total,
        "paralogs": paralogs,
        "paralog_count": paralog_total,
    }
    if wanted == "moderate":
        out["note"] = (
            "Moderate-confidence orthologues are selected from the full set on each row's own "
            "moderateFilter flag. Asking the server for stringency='moderate' directly returns "
            "nothing for every gene tested, which is why this call does not do that."
        )
        out["ortholog_count_complete"] = scanned_all >= all_total
        if scanned_all < all_total:
            out["note"] += (
                f" Scanned the first {scanned_all} of {all_total} 'all' rows, so ortholog_count is the "
                "moderate count within those rows and may be short."
            )
    if para_err:
        out["paralog_note"] = f"Paralogues could not be fetched: {para_err.get('error')}"
    return _ok(out)


def model_organisms_get_molecular_interactions(gene_id, limit=_DEFAULT_LIMIT, page=1):
    """Molecular (physical) interaction partners of one gene, with detection method and source.

    Genetic interactions are a separate Alliance annotation class and are not queried here
    (hunt 2026-09-30, uT4-genomics-28).

    Interactions are aggregated from BioGRID and IMEx and carry MI-ontology terms for how each was
    detected -- ``pull down``, ``two hybrid``, ``affinity chromatography`` -- which is what separates
    a directly demonstrated interaction from a high-throughput screen hit.

    Parameters
    ----------
    gene_id : str
        Prefixed Alliance gene id, e.g. ``"MGI:97490"``.
    limit : int, optional
        Interactions to return, 1-100. Default 20.
    page : int, optional
        1-based page number. Default 1.

    Returns
    -------
    dict
        ``data.interactions`` with the ``partner`` gene, ``interaction_type``, ``detection_method``,
        ``source_database``, the two interactor roles and the source's own ``interaction_id``.

    Examples
    --------
    >>> print(model_organisms_get_molecular_interactions("MGI:97490", limit=5))  # doctest: +SKIP
    """
    rows = _limit(limit)
    which = max(1, _as_int(page, 1))
    results, total, label, err = _sub_endpoint(
        gene_id,
        "molecular-interactions",
        {"limit": rows, "page": which},
        what="molecular interactions",
    )
    if err:
        return err

    def _name(obj):
        return (obj or {}).get("name")

    interactions = []
    for r in results:
        if not isinstance(r, dict):
            continue
        body = r.get("geneMolecularInteraction") or {}
        subject = body.get("geneAssociationSubject") or {}
        partner = body.get("geneGeneAssociationObject") or {}
        # The endpoint reports the association from whichever side was curated, so the query gene is
        # sometimes the object. Report the other end, whichever that is.
        if partner.get("primaryExternalId") == label["gene_id"]:
            partner = subject
        interactions.append(
            {
                "partner_gene_id": partner.get("primaryExternalId"),
                "partner_symbol": _text(partner.get("geneSymbol")),
                "interaction_type": _name(body.get("interactionType")),
                "detection_method": _name(body.get("detectionMethod")),
                "source_database": _name(body.get("interactionSource")) or _name(body.get("aggregationDatabase")),
                "interactor_a_role": _name(body.get("interactorARole")),
                "interactor_b_role": _name(body.get("interactorBRole")),
                "relation": _name(body.get("relation")),
                "interaction_id": body.get("interactionId"),
            }
        )

    out = {
        "gene": label,
        "interactions": interactions,
        "returned": len(interactions),
        "total_interactions": total,
        "page": which,
    }
    if not interactions:
        out["note"] = (
            f"{label['symbol']} ({label['gene_id']}) exists but has no curated molecular "
            "interactions on this page. Genetic interactions are a separate annotation class and "
            "are frequently empty even where molecular ones are not."
        )
    return _ok(out)


def model_organisms_get_alleles_and_models(gene_id, include="alleles", limit=_DEFAULT_LIMIT):
    """Curated alleles, dbSNP variants and mutant strains carrying a gene.

    "Allele" means two different things at this endpoint and they are returned in one stream: a
    hand-curated named allele such as ``Pax6<Gt(OST128284)Lex>``, and an imported dbSNP variant that
    is nothing but an ``rs`` accession. The variants come first and usually outnumber the alleles by
    an order of magnitude -- mouse *Pax6* has 607 variants ahead of 68 alleles, human *TP53* has
    2,920 variants and no curated alleles at all -- so a plain page of results is not what a caller
    asking about alleles wants. No server-side filter separates them (F47). The two kinds are
    block-sorted with the variants first, measured across six genes, so ``include="alleles"`` reads
    the tail of the list -- one page of at most 1,000 rows, wherever the list ends -- and splits it
    here. ``include="variants"`` pages normally from the front and is cheaper still.

    Models are affected genomic models -- the mutant strains, genotypes and fish lines a member
    database maintains -- and each carries the phenotypes observed in it and the diseases it is
    curated as a model of.

    Parameters
    ----------
    gene_id : str
        Prefixed Alliance gene id, e.g. ``"MGI:97490"``.
    include : str, optional
        ``"alleles"`` (default) returns only curated named alleles, ``"variants"`` only dbSNP
        variants, ``"all"`` both in the server's own order.
    limit : int, optional
        Alleles and models each returned, 1-100. Default 20.

    Returns
    -------
    dict
        ``data.alleles`` with ``allele_id``, ``symbol``, ``alteration_type``, ``has_phenotype``,
        ``has_disease`` and up to three ``variant_locations``; ``data.models`` with ``model_id``,
        ``name``, ``subtype``, ``phenotypes`` and ``diseases_modelled``; and ``data.allele_breakdown``
        giving the curated/variant split so the F47 ratio is visible rather than implied.

    Examples
    --------
    >>> print(model_organisms_get_alleles_and_models("MGI:97490"))  # doctest: +SKIP
    >>> print(model_organisms_get_alleles_and_models("ZFIN:ZDB-GENE-990415-8"))  # doctest: +SKIP
    """
    mode = _norm(include) or "alleles"
    if mode not in _ALLELE_MODES:
        close = _suggest(include, _ALLELE_MODES)
        return _error(
            f"'{include}' is not an include mode. "
            + (f"Closest: {', '.join(close)}. " if close else "")
            + "Use 'alleles' for curated named alleles, 'variants' for dbSNP variants, or 'all'."
        )

    rows = _limit(limit)
    record, err = _resolve_gene(gene_id)
    if err:
        return err
    label = _gene_label(record)
    resolved = label["gene_id"]

    # F47: variants are block-sorted ahead of curated alleles, so the alleles are the tail. A
    # one-row probe gives the length, and the last page is read directly instead of walking to it.
    # The endpoint has no offset parameter -- four spellings were tried and all returned page one --
    # so the window is placed by choosing a page size that divides the list evenly. For fly *white*
    # that reads the last 530 rows instead of the ragged final 61 a fixed 1,000-row page would give.
    scan_page = 1
    scan = rows
    window_start = 0
    if mode == "alleles":
        probe, err = _alliance_get(f"/gene/{_seg(resolved)}/alleles", {"limit": 1, "page": 1})
        if err:
            return err
        probe_total = _as_int((probe or {}).get("total"), 0)
        pages = max(1, -(-probe_total // _ALLELE_SCAN))  # ceiling division
        scan = max(1, -(-probe_total // pages)) if probe_total else _ALLELE_SCAN
        scan_page = pages
        window_start = (pages - 1) * scan

    payload, err = _alliance_get(f"/gene/{_seg(resolved)}/alleles", {"limit": scan, "page": scan_page})
    if err:
        return err
    allele_rows = (payload or {}).get("results") or []
    allele_total = _as_int((payload or {}).get("total"), 0)
    scanned = len(allele_rows)

    breakdown = {}
    for row in allele_rows:
        if isinstance(row, dict):
            kind = row.get("alterationType") or "unknown"
            breakdown[kind] = breakdown.get(kind, 0) + 1

    def _is_variant(row):
        return (row.get("alterationType") or "") == "variant"

    # Where the two blocks meet tells us the real curated count. If a variant row is in the window,
    # the boundary is inside it and the count is exact; if the window is entirely curated and began
    # mid-list, all we honestly know is a lower bound.
    curated_exact = None
    curated_floor = 0
    if mode == "alleles":
        first_curated = next(
            (i for i, r in enumerate(allele_rows) if isinstance(r, dict) and not _is_variant(r)),
            None,
        )
        if first_curated is None:
            curated_exact = 0 if window_start == 0 or breakdown.get("variant", 0) else None
        elif first_curated > 0 or window_start == 0:
            curated_exact = allele_total - (window_start + first_curated)
        else:
            curated_floor = len(allele_rows)

    if mode == "alleles":
        allele_rows = [r for r in allele_rows if isinstance(r, dict) and not _is_variant(r)]
    elif mode == "variants":
        allele_rows = [r for r in allele_rows if isinstance(r, dict) and _is_variant(r)]

    alleles = []
    for row in allele_rows[:rows]:
        if not isinstance(row, dict):
            continue
        allele = row.get("allele") or {}
        locations = []
        for variant in (row.get("variantList") or [])[:3]:
            for place in ((variant or {}).get("curatedVariantGenomicLocations") or [])[:1]:
                if (place or {}).get("hgvs"):
                    locations.append(place["hgvs"])
        # F46: a curated allele has primaryExternalId and no curie; a variant has curie and no
        # primaryExternalId. The symbol is never on the row -- it is under allele.alleleSymbol.
        alleles.append(
            {
                "allele_id": allele.get("primaryExternalId") or allele.get("curie"),
                "symbol": _strip_html(_text(allele.get("alleleSymbol"))) or None,
                "alteration_type": row.get("alterationType"),
                "has_phenotype": row.get("hasPhenotype"),
                "has_disease": row.get("hasDisease"),
                "variant_locations": locations,
            }
        )

    models = []
    model_total = 0
    model_payload, model_err = _alliance_get(f"/gene/{_seg(resolved)}/models", {"limit": rows, "page": 1})
    if not model_err:
        model_total = _as_int((model_payload or {}).get("total"), 0)
        for row in (model_payload or {}).get("results") or []:
            if not isinstance(row, dict):
                continue
            model = row.get("model") or {}
            diseases = [
                entry.get("diseaseModel")
                for entry in (row.get("diseaseModels") or [])
                if isinstance(entry, dict) and entry.get("diseaseModel")
            ]
            models.append(
                {
                    "model_id": model.get("primaryExternalId"),
                    # agmFullName is a label object carrying <sup> markup for superscripted alleles.
                    "name": _strip_html(_text(model.get("agmFullName"))) or None,
                    "subtype": (model.get("subtype") or {}).get("name"),
                    "data_provider": (model.get("dataProvider") or {}).get("abbreviation"),
                    "phenotypes": [_strip_html(p) for p in (row.get("associatedPhenotype") or [])][:10],
                    # diseaseModels[n].disease is an empty object; the name is in diseaseModel.
                    "diseases_modelled": diseases,
                }
            )

    out = {
        "gene": label,
        "include": mode,
        "alleles": alleles,
        "allele_count_returned": len(alleles),
        "allele_count_total": allele_total,
        "allele_breakdown": breakdown,
        "models": models,
        "model_count_returned": len(models),
        "model_count_total": model_total,
    }
    if mode == "alleles":
        whole_list = scanned >= allele_total
        out["rows_scanned"] = scanned
        out["rows_scanned_from"] = "the whole list" if whole_list else f"the last {scanned} rows"
        out["curated_allele_count"] = curated_exact
        if curated_exact is None:
            out["curated_allele_count_at_least"] = curated_floor
        if not alleles:
            where = (
                f"all {allele_total} rows are"
                if whole_list
                else f"the last {scanned} of {allele_total} rows, which is where curated alleles would sit, are all"
            )
            out["note"] = (
                f"{label['symbol']} has no curated named alleles: {where} dbSNP variants. "
                "Pass include='variants' to see them."
            )
        elif curated_exact is not None:
            variants = allele_total - curated_exact
            tail = "" if whole_list else f", found by reading the last {scanned} rows of {allele_total}"
            out["note"] = (
                f"{curated_exact} curated alleles and no dbSNP variants{tail}."
                if not variants
                else f"{curated_exact} curated alleles sit behind {variants} dbSNP variants in one "
                f"unfilterable list{tail}."
            )
        else:
            out["note"] = (
                f"At least {curated_floor} curated alleles, counted in the last {scanned} rows of "
                f"{allele_total}; every row in that window is curated, so the block of variants "
                "ends somewhere earlier and the exact split is not known from one request."
            )
    if model_err:
        out["model_note"] = f"Models could not be fetched: {model_err.get('error')}"
    return _ok(out)


def model_organisms_get_disease(disease_id):
    """A Disease Ontology term with its definition and its place in the hierarchy.

    Use this to check that a DOID means what you think before asking for its genes, and to move up
    or down the ontology: ``parents`` generalises the query, ``children`` narrows it, and
    ``descendant_count`` says how much of the tree sits underneath.

    Parameters
    ----------
    disease_id : str
        Disease Ontology identifier, e.g. ``"DOID:9351"`` (diabetes mellitus).

    Returns
    -------
    dict
        ``data.name``, ``data.definition``, ``data.synonyms``, ``data.descendant_count``, and
        ``data.parents`` / ``data.children`` as lists of ``{disease_id, name}``.

    Examples
    --------
    >>> print(model_organisms_get_disease("DOID:9351"))  # doctest: +SKIP
    """
    wanted = _clean(disease_id)
    if not wanted:
        return _missing(
            "disease_id",
            "Pass a Disease Ontology identifier such as 'DOID:9351' (diabetes mellitus) or 'DOID:162' (cancer).",
        )
    if not wanted.upper().startswith("DOID:"):
        return _error(
            f"'{disease_id}' is not a Disease Ontology identifier. This endpoint is keyed by DOID "
            "and takes nothing else -- a disease name, an OMIM or MeSH id will not resolve. "
            "Identifiers look like 'DOID:9351'."
        )

    payload, err = _alliance_get(f"/disease/{_seg(wanted)}")
    if err:
        return err
    record = payload or {}
    term = record.get("doTerm") or {}
    if not term.get("curie"):
        return _error(
            f"Alliance returned no Disease Ontology term for {wanted}. The prefix is right, so this "
            "is a wrong or obsolete DOID rather than a malformed one."
        )

    def _terms(key):
        out = []
        for entry in record.get(key) or []:
            if isinstance(entry, dict) and entry.get("curie"):
                out.append({"disease_id": entry.get("curie"), "name": entry.get("name")})
        return out

    return _ok(
        {
            "disease_id": term.get("curie"),
            "name": term.get("name"),
            "definition": _strip_html(term.get("definition")) or None,
            "synonyms": [s for s in (term.get("synonyms") or []) if s],
            "descendant_count": _as_int(term.get("descendantCount"), 0),
            "parents": _terms("parents"),
            "children": _terms("children"),
        }
    )


def model_organisms_get_disease_genes(disease_id, limit=_DEFAULT_LIMIT, page=1, species=None):
    """Genes curated as associated with a disease, across every Alliance organism.

    Two things about this endpoint decide whether the answer means anything, and both are surfaced
    in the result rather than left in the payload.

    **The relation is the finding, and some of them are negative.** Each row states *how* the gene
    relates to the disease: ``is_implicated_in`` (evidence it contributes), ``is_marker_for``
    (associated without a causal claim), or the negated forms ``is_not_implicated_in`` and
    ``is_not_marker_for``, which are curated statements that a published association did *not* hold.
    Of 500 human rows for DOID:9351, 33 are negative. They are returned here with the relation
    intact; treating the list as a flat "disease gene list" turns those 33 into their opposite (F48).

    **The subtree comes back with the term.** Asking for DOID:9351 (diabetes mellitus) returns
    annotations to its descendants too -- type-2, type-1, and ten further terms in the first 200
    rows, of which only 19 are DOID:9351 itself. ``data.by_disease_term`` reports that breakdown.

    A third distinction matters when filtering by species: a row with ``via_orthology`` true was not
    curated in that organism at all, it was inferred from a human annotation through an orthology
    call. For mouse rows on DOID:9351 that is 409 of the first 500.

    Parameters
    ----------
    disease_id : str
        Disease Ontology identifier, e.g. ``"DOID:9351"``.
    limit : int, optional
        Annotations to return, 1-100. Default 20. DOID:9351 has 12,740 in total.
    page : int, optional
        1-based page number. Default 1.
    species : str, optional
        Restrict to one organism by binomial name, e.g. ``"Mus musculus"``. Alliance accepts this
        only as ``filter.species``; a plain ``species`` parameter is ignored and returns everything.

    Returns
    -------
    dict
        ``data.genes`` with ``gene_id``, ``symbol``, ``species``, ``relation``, ``negated``,
        ``via_orthology``, ``disease_id``, ``disease_name``, ``evidence_codes`` and ``pubmed_ids``;
        plus ``data.by_disease_term`` and ``data.by_relation`` over the returned page.

    Examples
    --------
    >>> print(model_organisms_get_disease_genes("DOID:9351", limit=10))  # doctest: +SKIP
    >>> print(model_organisms_get_disease_genes("DOID:9351", species="Mus musculus", limit=10))  # doctest: +SKIP
    """
    wanted = _clean(disease_id)
    if not wanted:
        return _missing("disease_id", "Pass a Disease Ontology identifier such as 'DOID:9351'.")
    if not wanted.upper().startswith("DOID:"):
        return _error(
            f"'{disease_id}' is not a Disease Ontology identifier. Identifiers look like "
            "'DOID:9351'; model_organisms_get_disease() confirms one before you query its genes."
        )

    rows = _limit(limit)
    which = max(1, _as_int(page, 1))
    params = {"limit": rows, "page": which}
    wanted_species = _clean(species)
    if wanted_species:
        params["filter.species"] = wanted_species

    payload, err = _alliance_get(f"/disease/{_seg(wanted)}/genes", params)
    if err:
        return err
    data = payload or {}
    results = data.get("results") or []
    total = _as_int(data.get("total"), 0)

    genes = []
    by_term = {}
    by_relation = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        subject = row.get("subject") or {}
        disease = row.get("object") or {}
        # F48: relation.name drops the negation; generatedRelationString keeps it.
        relation = row.get("generatedRelationString") or (row.get("relation") or {}).get("name")
        evidence = [
            code.get("abbreviation") or code.get("name")
            for code in (row.get("evidenceCodes") or [])
            if isinstance(code, dict)
        ]
        genes.append(
            {
                "gene_id": subject.get("primaryExternalId"),
                "symbol": _text(subject.get("geneSymbol")),
                "species": (subject.get("taxon") or {}).get("name"),
                "relation": relation,
                "negated": bool(relation and relation.startswith("is_not_")),
                "via_orthology": bool(_as_int(row.get("viaOrthologyOrder"), 0)),
                "disease_id": disease.get("curie"),
                "disease_name": disease.get("name"),
                "evidence_codes": [e for e in evidence if e],
                "pubmed_ids": _pubmed_ids(row),
            }
        )
        term_name = disease.get("name") or disease.get("curie")
        if term_name:
            by_term[term_name] = by_term.get(term_name, 0) + 1
        if relation:
            by_relation[relation] = by_relation.get(relation, 0) + 1

    out = {
        "disease_id": wanted,
        "species_filter": wanted_species or "all",
        "genes": genes,
        "returned": len(genes),
        "total_annotations": total,
        "page": which,
        "by_disease_term": by_term,
        "by_relation": by_relation,
        "negated_on_this_page": sum(1 for g in genes if g["negated"]),
        "via_orthology_on_this_page": sum(1 for g in genes if g["via_orthology"]),
    }
    if not genes:
        if wanted_species:
            out["note"] = (
                f"No annotations for {wanted} in '{wanted_species}'. Check the binomial spelling -- "
                "Alliance matches it exactly, and an unrecognised name returns an empty page rather "
                "than an error. model_organisms_list_members() lists the eight it carries."
            )
        else:
            out["note"] = (
                f"Alliance has no gene annotations for {wanted}. Confirm the term exists with "
                "model_organisms_get_disease(); an unknown DOID also answers with an empty page here."
            )
    return _ok(out)
