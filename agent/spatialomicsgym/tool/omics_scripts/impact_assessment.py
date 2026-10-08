"""Scientific impact assessment via OpenAlex and Altmetric.

Adapted from omics-skills (https://github.com/fmschulz/omics-skills).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DOI_PATTERN = re.compile(r"^10\.\d{4,}/\S+$", re.IGNORECASE)


def normalize_doi(raw_doi: str) -> str:
    doi = raw_doi.strip()
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    if not DOI_PATTERN.match(doi):
        raise ValueError(f"Invalid DOI: {raw_doi}")
    return doi


def normalize_openalex_id(raw_id: str) -> str:
    candidate = raw_id.strip()
    if candidate.startswith("https://openalex.org/"):
        candidate = candidate.rsplit("/", 1)[-1]
    if not re.fullmatch(r"W\d+", candidate):
        raise ValueError(f"Invalid OpenAlex ID: {raw_id}")
    return candidate


def normalize_name(value: str) -> str:
    lowered = value.casefold()
    lowered = lowered.replace("&", " and ")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def fetch_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "spatialomicsgym-impact-assessment/1.0",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def fetch_openalex_work(doi: str | None, openalex_id: str | None, mailto: str | None) -> dict[str, Any]:
    base = "https://api.openalex.org/works"
    if doi:
        target = f"https://doi.org/{doi}"
        path = urllib.parse.quote(target, safe=":/")
    elif openalex_id:
        path = normalize_openalex_id(openalex_id)
    else:
        raise ValueError("Either doi or openalex_id is required")
    url = f"{base}/{path}"
    if mailto:
        url = f"{url}?{urllib.parse.urlencode({'mailto': mailto})}"
    return fetch_json(url)


def parse_openalex_work(payload: dict[str, Any]) -> dict[str, Any]:
    ids = payload.get("ids") or {}
    primary_source = (payload.get("primary_location") or {}).get("source") or {}
    doi = ids.get("doi") or payload.get("doi")
    normalized_doi = normalize_doi(doi) if doi else None
    percentile = payload.get("cited_by_percentile_year") or {}
    return {
        "openalex_id": ids.get("openalex") or payload.get("id"),
        "doi": normalized_doi,
        "title": payload.get("display_name"),
        "publication_year": payload.get("publication_year"),
        "cited_by_count": payload.get("cited_by_count"),
        "citation_percentile_min": percentile.get("min"),
        "citation_percentile_max": percentile.get("max"),
        "counts_by_year": payload.get("counts_by_year") or [],
        "journal_name": primary_source.get("display_name"),
        "journal_issn_l": primary_source.get("issn_l"),
        "type": payload.get("type"),
    }


def summarize_altmetric_payload(payload: dict[str, Any] | None, reason: str | None = None) -> dict[str, Any]:
    if payload is None:
        return {"status": "unavailable", "reason": reason or "not_requested"}
    return {
        "status": "available",
        "score": payload.get("score"),
        "details_url": payload.get("details_url") or payload.get("url"),
        "cited_by_posts_count": payload.get("cited_by_posts_count"),
        "cited_by_news_outlets_count": payload.get("cited_by_msm_count"),
        "cited_by_tweeters_count": payload.get("cited_by_tweeters_count"),
        "readers_count": payload.get("readers_count"),
    }


def fetch_altmetric_summary(doi: str | None, api_key: str | None) -> dict[str, Any]:
    if not doi:
        return summarize_altmetric_payload(None, reason="doi_required")
    if not api_key:
        return summarize_altmetric_payload(None, reason="no_api_key")
    encoded_doi = urllib.parse.quote(doi, safe="")
    url = f"https://api.altmetric.com/v1/doi/{encoded_doi}?{urllib.parse.urlencode({'key': api_key})}"
    try:
        payload = fetch_json(url)
    except urllib.error.HTTPError as exc:
        return summarize_altmetric_payload(None, reason=f"http_{exc.code}")
    except urllib.error.URLError:
        return summarize_altmetric_payload(None, reason="network_error")
    except (OSError, ValueError) as exc:
        # A read timeout or a dropped connection mid-body (OSError) escaped as a traceback, and a
        # body that is not JSON (ValueError) became an error -- either way the OpenAlex result
        # already in hand was thrown away (hunt 2026-09-30, uT6-literature-20).
        return summarize_altmetric_payload(None, reason=f"error: {type(exc).__name__}")
    return summarize_altmetric_payload(payload)


def assess_impact(
    doi: str | None = None,
    openalex_id: str | None = None,
    mailto: str | None = None,
    altmetric_api_key: str | None = None,
) -> dict[str, Any]:
    """Assess publication impact using OpenAlex and optionally Altmetric."""
    if doi:
        doi = normalize_doi(doi)
    if not doi and not openalex_id:
        raise ValueError("Either doi or openalex_id is required")

    mailto = mailto or os.environ.get("OPENALEX_MAILTO")
    altmetric_api_key = altmetric_api_key or os.environ.get("ALTMETRIC_API_KEY")

    # Guard the OpenAlex fetch the same way the Altmetric path already is: an un-indexed DOI returns 404
    # and fetch_json raises urllib.error.HTTPError — unguarded, that crashed the whole skill.
    try:
        openalex_payload = fetch_openalex_work(doi=doi, openalex_id=openalex_id, mailto=mailto)
        openalex_summary = parse_openalex_work(openalex_payload)
    except urllib.error.HTTPError as exc:
        openalex_summary = {"status": "unavailable", "reason": f"http_{exc.code}"}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # OSError, not only TimeoutError: a connection reset mid-body is one too (hunt 2026-09-30,
        # uT6-literature-20).
        openalex_summary = {"status": "unavailable", "reason": f"error: {type(exc).__name__}"}
    altmetric_summary = fetch_altmetric_summary(doi=doi, api_key=altmetric_api_key)

    return {
        "openalex": openalex_summary,
        "altmetric": altmetric_summary,
    }
