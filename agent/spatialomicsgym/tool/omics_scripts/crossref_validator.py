"""Crossref DOI validation, title search, and citation formatting.

Adapted from omics-skills (https://github.com/fmschulz/omics-skills).
"""

from __future__ import annotations

import re
import threading
import time
import urllib.parse
from typing import Any

import requests

CROSSREF_API_BASE = "https://api.crossref.org"
RATE_LIMIT_DELAY_SECONDS = 0.05
DOI_PATTERN = re.compile(r"^10\.\d{4,}/\S+$")


class CrossrefUnanswered(RuntimeError):
    """Crossref gave no answer either way: a transport failure, a 429/5xx, or a body that is not one.

    Raised where the return type has no room to say so -- a title search's list of hits, where an
    empty list already means "no such title".
    """


def _unchecked(error: str, **extra: Any) -> dict[str, Any]:
    """The ``fetch_doi`` payload for a lookup that says nothing about the DOI itself.

    ``unchecked`` is what separates it from the two answers that do -- Crossref's own 404 and a
    string that is not a DOI. Both used to share one ``(False, {"error": ...})`` shape with every
    outage, so ``validate_doi`` reported a rate limit as ``valid: false`` and a real paper as one
    that does not exist (hunt 2026-09-30, uT6-literature-2).
    """
    return {"error": error, "unchecked": True, **extra}


class CrossrefClient:
    #: One clock for every client in the process. ``omics_skills`` builds a fresh client per call,
    #: so a per-instance timestamp started at zero every time, never delayed anything, and a burst
    #: of lookups went out back to back -- which is what draws the 429s (hunt 2026-09-30,
    #: uT6-literature-2).
    _clock = threading.Lock()
    _last_request = 0.0

    def __init__(self, user_agent: str = "spatialomicsgym/1.0") -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    def _rate_limit(self) -> None:
        with CrossrefClient._clock:
            elapsed = time.time() - CrossrefClient._last_request
            if elapsed < RATE_LIMIT_DELAY_SECONDS:
                time.sleep(RATE_LIMIT_DELAY_SECONDS - elapsed)
            CrossrefClient._last_request = time.time()

    def fetch_doi(self, raw_doi: str) -> tuple[bool, dict[str, Any] | None]:
        """``(True, metadata)``, or ``(False, {"error": ...})``.

        Two errors are about the DOI and refute it: ``"invalid DOI format"`` and ``"DOI not found in
        Crossref"`` (callers match those two strings). Every other ``False`` carries
        ``"unchecked": True`` -- nobody answered, or the DOI is real but registered with another
        agency -- and is not evidence about the paper.
        """
        doi = normalize_doi(raw_doi)
        if doi is None:
            return False, {"error": "invalid DOI format"}
        self._rate_limit()
        # Percent-encode the DOI (keeping its path slash) so a DOI containing '#'/'?'/space doesn't
        # truncate the request path or inject a query param.
        doi_path = urllib.parse.quote(doi, safe="/")
        try:
            response = self._session.get(f"{CROSSREF_API_BASE}/works/{doi_path}", timeout=10)
        except requests.RequestException as exc:
            return False, _unchecked(f"request failed: {exc}")
        if response.status_code == 404:
            # Crossref holds only the DOIs its own members register. A Zenodo, figshare, Dryad or
            # arXiv DOI is DataCite's, so Crossref answers 404 for a dataset that plainly exists --
            # and that 404 was reported as "not found", refuting a real data-availability citation.
            # Crossref's agency route answers for every registration agency (skeptic note on
            # hunt 2026-09-30, uT6-literature-2).
            agency, failure = self._registration_agency(doi_path)
            if failure:
                return False, _unchecked(
                    f"DOI not found in Crossref, and its registration agency could not be read ({failure})"
                )
            if agency and agency.get("id") != "crossref":
                label = agency.get("label") or agency.get("id")
                return False, _unchecked(
                    f"DOI is registered with {label}, not Crossref, so Crossref holds no metadata for it",
                    registration_agency=label,
                )
            if agency:
                # doi.org names Crossref as the registrar, so the DOI exists and Crossref's works index
                # does not hold it yet. That 404 was returned as "not found" and refuted a registered
                # DOI (hunt 2026-09-30, uT6-literature-2 review).
                return False, _unchecked(
                    "DOI is registered with Crossref, but Crossref's works index has no record of it yet",
                    registration_agency=agency.get("label") or "Crossref",
                )
            return False, {"error": "DOI not found in Crossref"}
        if response.status_code != 200:
            return False, _unchecked(f"HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError:
            return False, _unchecked("Crossref answered with a body that is not JSON")
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            return False, _unchecked("unexpected Crossref response")
        return True, payload.get("message", {})

    def _registration_agency(self, doi_path: str) -> tuple[dict[str, Any] | None, str]:
        """``({"id", "label"}, "")`` for a registered DOI, ``(None, "")`` for one no agency knows,
        ``(None, reason)`` when the agency route itself did not answer."""
        self._rate_limit()
        try:
            response = self._session.get(f"{CROSSREF_API_BASE}/works/{doi_path}/agency", timeout=10)
        except requests.RequestException as exc:
            return None, f"request failed: {exc}"
        if response.status_code == 404:
            return None, ""
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"
        try:
            payload = response.json()
        except ValueError:
            return None, "a body that is not JSON"
        agency = ((payload or {}).get("message") or {}).get("agency") if isinstance(payload, dict) else None
        if not isinstance(agency, dict) or not agency.get("id"):
            return None, "no agency in the answer"
        return agency, ""

    def search_title(self, title: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Crossref's hits for ``title``. ``[]`` means Crossref answered and has none.

        Raises :class:`CrossrefUnanswered` when Crossref did not answer. Every such failure used to
        come back as ``[]``, which a caller cannot tell from "no paper has this title" -- the answer
        that sends a model off to invent a DOI (hunt 2026-09-30, uT6-literature-17).
        """
        self._rate_limit()
        try:
            response = self._session.get(
                f"{CROSSREF_API_BASE}/works",
                params={"query.title": title, "rows": max_results},
                timeout=10,
            )
        except requests.RequestException as exc:
            raise CrossrefUnanswered(f"Crossref title search did not answer: request failed: {exc}") from exc
        if response.status_code != 200:
            raise CrossrefUnanswered(
                f"Crossref title search did not answer: HTTP {response.status_code}. This says nothing "
                "about whether the paper exists; try again later."
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise CrossrefUnanswered("Crossref title search answered with a body that is not JSON") from exc
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise CrossrefUnanswered("Crossref title search returned an unexpected response")
        return payload.get("message", {}).get("items", [])


def normalize_doi(raw_doi: str) -> str | None:
    doi = raw_doi.strip()
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    if DOI_PATTERN.match(doi):
        return doi
    return None


def extract_year(metadata: dict[str, Any]) -> str:
    for field in ("published-print", "published-online", "issued"):
        # A date field can be present-but-null (-> AttributeError on .get) or have partial date-parts
        # like [[None]] (-> the literal year "None"); guard both.
        date_parts = (metadata.get(field) or {}).get("date-parts") or [[]]
        if date_parts and date_parts[0] and date_parts[0][0] is not None:
            return str(date_parts[0][0])
    return "n.d."


def first_string(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if value is None:
        return ""
    return str(value)


def format_authors(authors: list[dict[str, Any]], style: str) -> str:
    if not authors:
        return "Unknown"
    formatted: list[str] = []
    for author in authors[:6]:
        family = (author.get("family") or "").strip()
        given = (author.get("given") or "").strip()
        spaced_initials = " ".join(f"{part[0]}." for part in given.split() if part)
        initials = "".join(part[0] for part in given.split() if part)
        if not family:
            # A consortium is deposited with a single ``name`` and no family/given, so the citation
            # printed with its author blank: " (2019). T. Nature" (hunt 2026-09-30, uT6-literature-19).
            name = (author.get("name") or "").strip()
        elif style in {"apa", "chicago"}:
            name = f"{family}, {spaced_initials}".strip().rstrip(",")
        else:
            name = f"{family} {initials}".strip()
        if name:
            formatted.append(name)
    if not formatted:
        return "Unknown"
    if len(authors) > 6:
        formatted.append("et al.")
    elif len(formatted) > 1 and style in {"apa", "chicago"}:
        formatted[-1] = f"& {formatted[-1]}"
    elif len(formatted) > 1 and style == "ieee":
        formatted[-1] = f"and {formatted[-1]}"
    return ", ".join(formatted)


def format_citation(metadata: dict[str, Any], style: str = "apa") -> str:
    """Format a Crossref metadata record as a citation string."""
    authors = format_authors(metadata.get("author", []), style)
    title = first_string(metadata.get("title"))
    journal = first_string(metadata.get("container-title"))
    year = extract_year(metadata)
    volume = metadata.get("volume", "")
    issue = metadata.get("issue", "")
    pages = metadata.get("page", "")
    doi = metadata.get("DOI", "")

    if style == "apa":
        citation = f"{authors} ({year}). {title}. {journal}"
        if volume:
            citation += f", {volume}"
        if issue:
            citation += f"({issue})"
        if pages:
            citation += f", {pages}"
        if doi:
            citation += f". https://doi.org/{doi}"
        return citation

    if style in {"ama", "vancouver"}:
        citation = f"{authors} {title}. {journal}. {year}"
        if volume:
            citation += f";{volume}"
        if issue:
            citation += f"({issue})"
        if pages:
            citation += f":{pages}"
        if doi:
            citation += f". doi:{doi}"
        return citation

    if style == "ieee":
        citation = f'{authors}, "{title}," {journal}'
        if volume:
            citation += f", vol. {volume}"
        if issue:
            citation += f", no. {issue}"
        if pages:
            citation += f", pp. {pages}"
        if year:
            citation += f", {year}"
        if doi:
            citation += f". doi: {doi}"
        return citation

    # Chicago (default)
    citation = f'{authors} {year}. "{title}." {journal}'
    if volume:
        citation += f" {volume}"
    if issue:
        citation += f"({issue})"
    if pages:
        citation += f": {pages}"
    if doi:
        citation += f". https://doi.org/{doi}"
    return citation
