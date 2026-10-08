"""Omics-skills tools: literature search, impact assessment, DOI validation, and JGI Lakehouse access.

Adapted from omics-skills (https://github.com/fmschulz/omics-skills).
"""

import json

#: What ``CrossrefClient.fetch_doi`` reports when Crossref itself answered 404. Every other failure
#: it reports -- a transport error, a 429 or 5xx, a body that is not Crossref's -- means Crossref was
#: not asked successfully, which is no evidence that the DOI does not exist.
_CROSSREF_NOT_FOUND = "DOI not found in Crossref"

#: The styles ``format_citation`` has a branch for. Anything else used to fall silently into the
#: Chicago branch (hunt 2026-09-30, uT3-atlases-18).
_CITATION_STYLES = ("apa", "ama", "vancouver", "ieee", "chicago")


def _failed(error: Exception) -> str:
    """A call that could not run, as the JSON error the agent loop recognises -- not a traceback.

    The arXiv and bioRxiv clients raise RuntimeError for an HTTP or network failure and ValueError
    for a bad argument. assess_scientific_impact and the JGI wrappers already answered in JSON and
    these three let the exception escape (hunt 2026-09-30, uT3-atlases skeptic note). ``success:
    false`` keeps the research ledger's arXiv check reading it as "arXiv did not answer".
    """
    return json.dumps(
        {"success": False, "status": "error", "error": f"{type(error).__name__}: {error}"},
        indent=2,
    )


def _without_arxiv_errors(result: dict) -> dict:
    """Move arXiv's error entries out of ``results``.

    arXiv answers a malformed id or query with HTTP 200 and an Atom entry whose id is under
    ``http://arxiv.org/api/errors`` and whose title is "Error". Parsed as a paper, it let the
    research ledger "verify" a made-up id with the title "Error" (hunt 2026-09-30, uT3-atlases
    skeptic note). It is arXiv's refusal, not a record, so it is reported under ``errors``.
    """
    found = result.get("results")
    if not isinstance(found, list):
        return result
    errors = [r for r in found if isinstance(r, dict) and "arxiv.org/api/errors" in str(r.get("abs_url") or "")]
    if errors:
        result["results"] = [r for r in found if not any(r is e for e in errors)]
        result["result_count"] = len(result["results"])
        result["errors"] = [str(e.get("summary") or e.get("abs_url")) for e in errors]
    return result


def search_arxiv_advanced(
    query: str,
    max_results: int = 10,
    phrase: bool = False,
    category: str | None = None,
    days: int | None = None,
    sort: str = "relevance",
    start: int = 0,
) -> str:
    """Search arXiv with advanced filtering including category, date range, and pagination.

    This is an enhanced arXiv search that supports category filtering (e.g., 'q-bio.GN'),
    date filtering via days parameter, phrase search, and raw arXiv query syntax.
    Returns structured results with paper metadata including authors, abstract, URLs, and categories.

    Args:
        query: Search query (plain text or raw arXiv query syntax with ti:/au:/abs:/cat: prefixes).
        max_results: Maximum number of results (1-2000).
        phrase: If True, treat the query as a single phrase instead of individual terms.
        category: Restrict to an arXiv category (e.g., 'q-bio.GN', 'cs.LG').
        days: Only return papers published within the last N days.
        sort: Sort field - 'relevance', 'lastUpdatedDate', or 'submittedDate'.
        start: 0-based result offset for pagination.

    Returns:
        str: JSON string with search results including paper titles, authors, abstracts, URLs, and metadata.

    """
    from spatialomicsgym.tool.omics_scripts.arxiv_search import search

    try:
        result = search(
            query=query,
            max_results=max_results,
            phrase=phrase,
            category=category,
            days=days,
            sort=sort,
            start=start,
        )
    except Exception as e:
        return _failed(e)
    return json.dumps(_without_arxiv_errors(result), indent=2)


def fetch_arxiv_by_ids(ids: str) -> str:
    """Fetch arXiv papers by their IDs.

    Args:
        ids: Comma-separated arXiv paper IDs (e.g., '2301.00001,2301.00002').

    Returns:
        str: JSON string with paper metadata for the requested IDs.

    """
    from spatialomicsgym.tool.omics_scripts.arxiv_search import fetch_by_ids

    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    try:
        result = fetch_by_ids(id_list)
    except Exception as e:
        return _failed(e)
    return json.dumps(_without_arxiv_errors(result), indent=2)


def search_biorxiv(
    query: str | None = None,
    max_results: int = 10,
    days: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
    authors: str | None = None,
    doi: str | None = None,
    phrase: bool = False,
    scan_limit: int = 300,
) -> str:
    """Search bioRxiv preprints with local keyword and author filtering.

    Queries the bioRxiv API and applies local filtering on title, abstract, and author fields.
    Supports date range filtering, category filtering, and automatic deduplication of paper versions.

    Args:
        query: Keyword query for filtering (supports OR between groups, quoted phrases).
        max_results: Maximum results to return.
        days: Search the most recent N days of bioRxiv records.
        start_date: Explicit start date (YYYY-MM-DD). Use with end_date.
        end_date: Explicit end date (YYYY-MM-DD). Use with start_date.
        category: bioRxiv subject category (e.g., 'genomics', 'bioinformatics').
        authors: Semicolon-separated author names for filtering (e.g., 'Smith J; Doe Jane').
        doi: Fetch a specific bioRxiv DOI directly.
        phrase: Treat query as a single phrase.
        scan_limit: Maximum API records to inspect locally.

    Returns:
        str: JSON string with matched preprints including title, authors, abstract, DOI, and URLs.

    """
    from spatialomicsgym.tool.omics_scripts.biorxiv_search import search

    author_list = [a.strip() for a in (authors or "").split(";") if a.strip()] if authors else None
    try:
        result = search(
            query=query,
            max_results=max_results,
            phrase=phrase,
            days=days,
            start_date=start_date,
            end_date=end_date,
            category=category,
            authors=author_list,
            doi=doi,
            scan_limit=scan_limit,
        )
    except Exception as e:
        return _failed(e)
    return json.dumps(result, indent=2)


def assess_scientific_impact(
    doi: str | None = None,
    openalex_id: str | None = None,
    mailto: str | None = None,
) -> str:
    """Assess publication impact using OpenAlex citation data and Altmetric scores.

    Retrieves citation counts, citation percentiles, journal information from OpenAlex,
    and social media/news attention from Altmetric (if ALTMETRIC_API_KEY is set).

    Args:
        doi: The DOI of the publication to assess.
        openalex_id: OpenAlex work ID (e.g., 'W2741809807'). Alternative to DOI.
        mailto: Email for OpenAlex polite-pool (or set OPENALEX_MAILTO env var).

    Returns:
        str: JSON report with OpenAlex metrics (citations, percentiles, journal) and Altmetric scores.

    """
    from spatialomicsgym.tool.omics_scripts.impact_assessment import assess_impact

    # assess_impact is hardened against network/404 failures, but normalize_doi() runs first and raises
    # ValueError on a malformed DOI string — catch it so a bad DOI returns JSON, not a traceback
    # (validate_doi already degrades gracefully; match that).
    try:
        result = assess_impact(doi=doi, openalex_id=openalex_id, mailto=mailto)
    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)
    return json.dumps(result, indent=2)


def validate_doi(doi: str) -> str:
    """Validate a DOI using the Crossref REST API and retrieve its metadata.

    Checks whether a DOI exists in Crossref and returns the publication metadata
    including title, journal, authors, and year.

    Args:
        doi: The DOI to validate (e.g., '10.1038/nature12373').

    Returns:
        str: JSON string. ``valid`` is true, with the metadata and an APA citation, when Crossref has
        the record; false, with ``not_found``, when the string is not a DOI or Crossref answered that
        it has no such record; null, with ``status: error``, when Crossref gave no answer about it --
        an outage, a rate limit or a DOI another agency (DataCite) registered is not evidence that
        the DOI does not exist.

    """
    from spatialomicsgym.tool.omics_scripts.crossref_validator import (
        CrossrefClient,
        extract_year,
        first_string,
        format_citation,
        normalize_doi,
    )

    # Every failure used to come back as ``valid: false``, so a Crossref outage, a 429 or a 5xx
    # became a refutation, and the research ledger printed "no record of this DOI" under a real
    # paper (hunt 2026-09-30, uT3-atlases-1). Only a malformed DOI or Crossref's own 404 is a "no".
    if normalize_doi(str(doi or "")) is None:
        return json.dumps({"valid": False, "not_found": True, "doi": doi, "error": "invalid DOI format"})
    try:
        is_valid, metadata = CrossrefClient().fetch_doi(doi)
    except Exception as e:  # e.g. a 200 whose body is not JSON (a maintenance page)
        is_valid, metadata = None, {"error": f"{type(e).__name__}: {e}"}
    if not is_valid or metadata is None:
        metadata = metadata or {}
        detail = metadata.get("error") or "no reason given"
        if is_valid is False and not metadata.get("unchecked") and detail == _CROSSREF_NOT_FOUND:
            return json.dumps({"valid": False, "not_found": True, "doi": doi, "error": detail})
        unchecked = {
            "valid": None,
            "status": "error",
            "doi": doi,
            "error": f"Unchecked, not refuted: {detail}. This is no evidence that the DOI does not exist.",
        }
        if metadata.get("registration_agency"):
            unchecked["registration_agency"] = metadata["registration_agency"]
        return json.dumps(unchecked)
    result = {
        "valid": True,
        "doi": doi,
        "title": first_string(metadata.get("title")),
        "journal": first_string(metadata.get("container-title")),
        "year": extract_year(metadata),
        "citation_apa": format_citation(metadata, "apa"),
    }
    return json.dumps(result, indent=2)


def search_crossref_by_title(title: str, max_results: int = 5) -> str:
    """Search for publications by title using the Crossref REST API.

    Args:
        title: Title or partial title to search for.
        max_results: Maximum number of results to return.

    Returns:
        str: JSON string with matching publications including DOI, title, journal, and year.

    """
    from spatialomicsgym.tool.omics_scripts.crossref_validator import CrossrefClient, extract_year, first_string

    # A failed search printed as ``result_count: 0`` reads as "no paper has this title", which is the
    # cue to drop a real reference or write a DOI from memory (hunt 2026-09-30, uT3-atlases-3). A
    # search that could not run is an error the agent loop recognises, never an empty hit list.
    try:
        works = CrossrefClient().search_title(title, max_results=max_results)
    except Exception as e:
        works = e
    if not isinstance(works, list):
        reason = f"{type(works).__name__}: {works}" if isinstance(works, Exception) else repr(works)[:200]
        return json.dumps(
            {
                "status": "error",
                "error": f"The Crossref title search did not complete ({reason}). An empty result was not "
                "returned because none was observed: this says nothing about whether such a paper exists.",
            },
            indent=2,
        )
    results = []
    for work in works:
        results.append(
            {
                "doi": work.get("DOI", ""),
                "title": first_string(work.get("title")),
                "journal": first_string(work.get("container-title")),
                "year": extract_year(work),
            }
        )
    return json.dumps({"result_count": len(results), "results": results}, indent=2)


def format_citation_from_doi(doi: str, style: str = "apa") -> str:
    """Format a citation for a DOI in a given citation style.

    Args:
        doi: The DOI to format a citation for.
        style: Citation style - 'apa', 'vancouver', 'ama', 'ieee', or 'chicago'.

    Returns:
        str: Formatted citation string, or an error message if the DOI is invalid.

    """
    from spatialomicsgym.tool.omics_scripts.crossref_validator import CrossrefClient, format_citation

    key = str(style or "").strip().lower()
    if key not in _CITATION_STYLES:
        return f"Error: unknown citation style '{style}'. Supported: {', '.join(_CITATION_STYLES)}."
    try:
        is_valid, metadata = CrossrefClient().fetch_doi(doi)
    except Exception as e:  # e.g. a 200 whose body is not JSON
        is_valid, metadata = None, {"error": f"{type(e).__name__}: {e}"}
    if not is_valid or metadata is None:
        detail = (metadata or {}).get("error") or "no reason given"
        return f"Error: could not retrieve metadata for DOI {doi} ({detail})"
    return format_citation(metadata, key)


def query_jgi_lakehouse(sql: str, limit: int = 100, timeout: int = 300) -> str:
    """Execute a SQL query against the JGI Lakehouse (Dremio) and return results.

    Requires DREMIO_PAT environment variable and LBNL network access. Connects over verified HTTPS; for a
    Dremio without TLS the operator sets DREMIO_SCHEME=http and DREMIO_ALLOW_PLAINTEXT_TOKEN=1.
    Queries biological databases including GOLD, IMG, Mycocosm, and Phytozome.

    Args:
        sql: SQL query to execute.
        limit: Maximum rows to return.
        timeout: Maximum seconds to wait for query completion.

    Returns:
        str: JSON string with query result rows.

    """
    from spatialomicsgym.tool.omics_scripts.jgi_lakehouse import query

    # A missing DREMIO_PAT (ValueError) or off-LBNL/network failure (requests error) must reach the agent
    # as a JSON error it can reason about, not a raw traceback.
    try:
        rows = query(sql, limit=limit, timeout=timeout)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)
    return json.dumps({"row_count": len(rows), "rows": rows}, indent=2)


def list_jgi_lakehouse_schemas() -> str:
    """List all available schemas in the JGI Lakehouse (Dremio).

    Requires DREMIO_PAT environment variable and LBNL network access. Connects over verified HTTPS; for a
    Dremio without TLS the operator sets DREMIO_SCHEME=http and DREMIO_ALLOW_PLAINTEXT_TOKEN=1.

    Returns:
        str: JSON string with list of available database schemas.

    """
    from spatialomicsgym.tool.omics_scripts.jgi_lakehouse import show_schemas

    try:
        schemas = show_schemas()
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)
    return json.dumps({"schema_count": len(schemas), "schemas": schemas}, indent=2)
