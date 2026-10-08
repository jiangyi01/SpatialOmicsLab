"""Scholarly literature: Europe PMC, OpenAlex and Crossref -- search, full text, citations, metadata.

A spatial-omics question almost always ends in the literature. Which paper first reported this
marker in this tissue? What does the methods section of the dataset's source paper actually say
about the panel? Who else has used this atlas, and what did they find? Is this DOI real, and what
does it point at? Three public services answer those, and each is better at a different part:

* **Europe PMC** (``www.ebi.ac.uk/europepmc``) -- biomedical search with abstracts, and the only one
  of the three that hands back **open-access full text**. This module can fetch that full text whole,
  pull keyword-centred snippets out of it, or return it split into the sections a methods question
  actually needs.
* **OpenAlex** (``api.openalex.org``) -- the open scholarly graph: works, authors, institutions and
  journals, with citation counts, concepts and open-access status. Use it for "who / where / how
  influential", and for author and venue disambiguation.
* **Crossref** (``api.crossref.org``) -- the DOI registration agency. Authoritative for what a DOI
  *is*: title, container, publication date, funders, licence, member and type. Use it to verify or
  resolve a reference rather than to discover one.

Full text arrives by a four-step fallback because no single source serves every article. In order:
Europe PMC ``fullTextXML``, then the NCBI PMC OAI record, then NCBI ``efetch``, then the PMC HTML
page. Every attempt is recorded in a ``trace`` on the result, so a partial or empty answer says
which sources were tried and what each returned instead of looking like an article with no content.

Every outbound call goes through :mod:`spatialomicsgym.utils.http_client`, which enforces HTTPS, an
explicit host allowlist, one shared connection pool, a timeout and bounded retry. No function here
calls ``requests`` directly, and no function here reads an API key -- all three services answer
anonymously (measured, see ``_ALLOWED_HOSTS`` and ``openalex_literature_search``).

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
knowledge, the four-step full-text fallback chain and the section-keyword map are upstream's; the
code is not. Specifically:

* the ``BaseTool``/``BaseRESTTool``/``register_tool``/config-driven dispatch machinery was not
  vendored, and each upstream tool is re-expressed here as a plain function;
* upstream's ``http_utils.request_with_retry`` was not vendored -- our HTTP layer supersedes it --
  and raw ``requests`` calls (some with **no timeout at all**, e.g. OpenAlex's own search) were
  replaced by it;
* **no API key is read.** Upstream's ``_with_api_key`` asserts in its docstring that anonymous
  OpenAlex "now returns HTTP 503" and requires ``OPENALEX_API_KEY``. Measured live 2026-09-17: an
  anonymous request with neither ``api_key`` nor ``mailto`` returned **HTTP 200**. One fewer secret
  to hold, so the requirement is dropped;
* **no ``mailto`` is sent.** Upstream hardcodes ``mailto=support@openalex.org`` -- a third party's
  address -- on every request. We send none rather than send someone else's, or the user's;
* **no User-Agent spoof.** Two upstream classes send a Chrome UA string with the comment that
  "NCBI/PMC return 403 for the default python-requests User-Agent". Measured live 2026-09-17: the
  OAI endpoint, ``efetch`` and the PMC HTML page all returned **200** with the default UA. Our HTTP
  layer sends one honest, identifying User-Agent, as NCBI's own usage policy asks for;
* caller-supplied values interpolated into URL **paths** are percent-encoded. Upstream's Europe PMC
  and OpenAlex classes use ``str.replace`` on the template and encode nothing (their Crossref base
  class does encode, so the behaviour was inconsistent even upstream). Verified not to break
  OpenAlex's DOI route, whose path legitimately contains ``https://doi.org/``;
* **F9** -- ``_extract_abstract_from_pmc_html`` is dead upstream: its three patterns are raw strings
  containing ``\\s``, which compiles to "a literal backslash then one or more ``s``" and can never
  match. The function always returns ``None``, so the HTML branch of abstract enrichment is
  unreachable. Re-written here so the branch works;
* **F10** -- the snippet loop computes ``truncated`` from the running character count, which stays
  ``False`` when the budget was hit by a snippet that was *skipped*. Matches were dropped and the
  caller was told nothing was. ``truncated`` here is set where the drop happens;
* **F11** -- upstream issues two searches per query (``resultType=core`` and ``resultType=lite``)
  solely to read a journal title. Measured live: ``core`` already carries it at
  ``journalInfo.journal.title``. One request, not two;
* **F12** -- OpenAlex abstracts are rebuilt into a fixed ``[""] * 500`` buffer, silently discarding
  every inverted-index position past 500. Measured live: work ``W2005501262`` has **4,055**
  positions, i.e. 87.7% of that abstract was dropped with no flag. The buffer here is sized from the
  data;
* **F13** -- the structured full-text parser uses bare ``root.find(".//abstract")``. JATS arrives in
  two namespace regimes: Europe PMC's ``fullTextXML`` has no namespace, while the NCBI OAI record
  wraps JATS in ``https://jats.nlm.nih.gov/ns/archiving/1.4/`` inside an OAI-PMH envelope. Against
  the OAI document every bare ``find()`` returns ``None``, so upstream answers ``status: "success"``
  with an empty title, empty abstract and zero sections -- precisely in the case the fallback exists
  to serve. Confirmed live on PMC13521112, where Europe PMC returns 500 and the OAI record returns a
  complete article. Parsing here is namespace-agnostic;
* **F14** -- see the key/mailto/User-Agent items above;
* **F15** -- every XML-to-text conversion upstream is ``"".join(element.itertext())`` (five sites:
  ``europe_pmc_tool.py`` 419, 783, 909, 1149 and ``_itertext`` at 1307). Joining on the empty string
  fuses the last word of one block element to the first word of the next, so ``<title>Abstract</
  title><title>Background</title><p>Ashwagandha is ...`` reads out as the single token
  ``AbstractBackgroundAshwagandha``. Measured on PMC7096075: **65** such fusions in one 42k-character
  article, including ``study.Exclusion criteriaParticipants``. It corrupts every consumer -- a phrase
  cannot match across a fusion, term counts drift, and the model reads joined-up words. Extraction
  here delimits block elements and leaves inline markup tight, which is the distinction that matters
  (``<italic>P</italic>. <italic>falciparum</italic>`` must stay one phrase); the same article now
  has **0** false fusions, the four remaining case boundaries being HeLa, DiLoreto and BioMed. Whole
  -article text also keeps its paragraphs rather than collapsing onto one 42k-character line;
* snippet search is scoped to ``<abstract>`` and ``<body>``. Upstream searches the whole document,
  so a term appearing only in a cited paper's title in ``<ref-list>`` counts as a hit and can spend
  the snippet budget on bibliography, while ``<front>`` yields publisher metadata -- upstream's
  first snippet for PMC7096075 began ``2757cureusCureusCureusCureus Inc.PMC7096075``;
* the Crossref paging ceiling is enforced as ``offset + rows <= 10000``, which is what the service's
  own error message states, rather than upstream's ``offset <= 10000``;
* ``extract_terms_from_fulltext`` was dropped from the search tool: it fetched and scanned the full
  text of **every** hit inline. ``europe_pmc_get_fulltext_snippets`` does the same job on an article
  the model has chosen. Recorded rather than silently narrowed -- see ``VENDORING.md``;
* supplying ``fulltext_terms`` does **not** silently force ``require_has_fulltext``/``HAS_FT:Y`` on,
  as upstream does. For Europe PMC the forcing is a no-op, because a ``BODY:`` clause can only match
  a record whose body is indexed (measured 2026-09-17: ``(spatial transcriptomics) AND
  (BODY:"Xenium")`` -> 1222 hits with and without ``HAS_FT:Y``). For OpenAlex it is not a no-op and
  it loses matches: ``fulltext.search:Xenium`` -> 3468 works, and adding ``has_fulltext:true`` ->
  1788, because OpenAlex's full-text search also covers title and abstract. Narrowing a result set
  by two thirds is the caller's decision to make, so the flag stays the caller's.
"""

import logging
import re
import time
import xml.etree.ElementTree as ElementTree
from html.parser import HTMLParser
from urllib.parse import quote

from spatialomicsgym.utils.http_client import HttpError, request_json, request_text

logger = logging.getLogger(__name__)

# The only hosts this module is permitted to reach. Passed to the HTTP layer on every call, which
# refuses anything else -- so no argument, however malformed or adversarial, can redirect a request
# at a host that is not on this list. Every one cleared under the RL-1 origin review recorded in
# CHINA_EXCLUSION.md:
#   www.ebi.ac.uk        -- EMBL-EBI, Hinxton, UK (Europe PMC's REST service)
#   www.ncbi.nlm.nih.gov -- US NIH / NCBI (the PMC OAI-PMH interface)
#   eutils.ncbi.nlm.nih.gov -- US NIH / NCBI (E-utilities efetch)
#   pmc.ncbi.nlm.nih.gov -- US NIH / NCBI (the PMC article page)
#   api.openalex.org     -- OurResearch, a US non-profit
#   api.crossref.org     -- Crossref, a US non-profit DOI registration agency
# doi.org appears inside one OpenAlex URL *path* and is never contacted as a host, so it is not
# listed here -- the allowlist names what we connect to, not what a URL mentions.
_ALLOWED_HOSTS = (
    "www.ebi.ac.uk",
    "www.ncbi.nlm.nih.gov",
    "eutils.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "api.openalex.org",
    "api.crossref.org",
)

_EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_OPENALEX = "https://api.openalex.org"
_CROSSREF = "https://api.crossref.org"

#: Page ceilings, so a stray ``limit=100000`` cannot paste a megabyte of records into the
#: observation and blow the context window.
_MAX_PAGE = 100
_MAX_CROSSREF_ROWS = 1000

#: Crossref refuses deep paging, and states the rule itself: "Offset specified as 10000 but for this
#: route and for rows = 1, offset must be a positive integer less than or equal to 9999. Use the
#: cursor parameter to page further into result sets." Measured live 2026-09-17: offset=9999&rows=1
#: succeeds, offset=10000 fails, and offset=9990&rows=20 also fails -- so the constraint is on the
#: *sum*, not on offset alone. (Upstream's constant says ``offset <= 10000`` with an inline comment
#: claiming offset=9999 fails, which is the opposite of what the service does.)
_CROSSREF_WINDOW = 10000

#: The bound on ``enrich_missing_abstract``: how many hits it fetches, and the wall clock after which
#: it starts no more (uT6-literature-31).
_ENRICH_MAX_HITS = 10
_ENRICH_BUDGET_SECONDS = 180.0

#: How much full text to hand back by default. Whole articles routinely exceed 200 kB, which is more
#: than an observation should ever carry; the cap is disclosed on the result rather than applied
#: silently.
_DEFAULT_MAX_CHARS = 200000

#: Section keywords, upstream's map, used to bucket JATS ``<sec>`` elements by their title when the
#: markup carries no ``sec-type``. Order matters: the first bucket whose keyword appears wins, so
#: "materials and methods" lands in methods rather than in results.
_SECTION_KEYWORDS = (
    ("abstract", ("abstract", "summary")),
    ("introduction", ("introduction", "background")),
    ("methods", ("method", "materials", "experimental", "procedure", "protocol")),
    ("results", ("result", "finding", "observation")),
    ("discussion", ("discussion", "interpretation")),
    ("conclusion", ("conclusion", "concluding", "perspective", "outlook")),
)

#: A PMCID, with or without its prefix, in any of the forms that turn up in metadata.
_PMCID_RE = re.compile(r"^(?:PMC)?(\d+)$", re.IGNORECASE)

#: Collapses runs of whitespace, including the newlines JATS uses freely inside a paragraph.
_WS_RE = re.compile(r"\s+")


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


def _page_size(size, default=10, maximum=_MAX_PAGE):
    try:
        return max(1, min(int(size), maximum))
    except (TypeError, ValueError):
        return default


def _as_int(value, default=0):
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean(text):
    """Collapse whitespace; JATS and HTML both wrap mid-sentence and the model reads the result."""
    return _WS_RE.sub(" ", str(text or "")).strip()


def _http_failure(exc):
    """The error payload for an ``HttpError``, carrying the service's own explanation.

    The HTTP layer keeps the response body apart from the message, and only the message was passed
    on -- so OpenAlex's "... is not a valid field" never reached the agent, which saw only "returned
    HTTP 403" and could not tell which filter clause was wrong. A 429 that outlived the retries was
    also marked not retryable (hunt 2026-09-30, uT6-literature-12).
    """
    message = exc.detail
    body = _clean(exc.body)[:300] if exc.body else ""
    if body:
        message = f"{message}: {body}"
    status = exc.status
    return _error(message, retryable=bool(status) and (status == 429 or status >= 500), http_status=status)


def _fetch_json(url, **kwargs):
    """``(payload, None)`` on success, ``(None, error_payload)`` on failure."""
    try:
        return request_json(url, allowed_hosts=_ALLOWED_HOSTS, **kwargs), None
    except HttpError as exc:
        return None, _http_failure(exc)


def _fetch_text(url, **kwargs):
    """``(body, None)`` on success, ``(None, error_payload)`` on failure."""
    try:
        return request_text(url, allowed_hosts=_ALLOWED_HOSTS, **kwargs), None
    except HttpError as exc:
        return None, _http_failure(exc)


def _as_terms(value):
    """A caller's term list. A bare string is one term, not a list of its characters.

    ``for item in "Xenium"`` iterates letters, so fulltext_terms="Xenium" became BODY:"X" OR BODY:"e"
    OR ... and voided the filter while reporting success (hunt 2026-09-30, uT6-literature-10).
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _local(tag):
    """The local name of an XML tag: ``{http://ns}article-title`` -> ``article-title``."""
    text = str(tag)
    return text.rsplit("}", 1)[-1] if "}" in text else text


def _find(root, name):
    """First descendant (or self) whose *local* name is ``name``, whatever namespace it carries.

    This is the whole of the F13 fix. JATS reaches us in two namespace regimes -- Europe PMC's
    ``fullTextXML`` has root ``<article>`` in no namespace, while the NCBI PMC OAI record wraps the
    same article in an OAI-PMH envelope with JATS under
    ``https://jats.nlm.nih.gov/ns/archiving/1.4/`` -- and a bare ``root.find(".//abstract")`` matches
    only the first. Matching on the local name handles both, and keeps handling the next namespace
    version NLM publishes.
    """
    if root is None:
        return None
    if _local(root.tag) == name:
        return root
    for element in root.iter():
        if _local(element.tag) == name:
            return element
    return None


def _find_all(root, name):
    """Every descendant whose local name is ``name``. See :func:`_find`."""
    if root is None:
        return []
    return [element for element in root.iter() if _local(element.tag) == name]


# JATS block-level elements. Text inside an *inline* element must stay glued to its neighbours --
# "<italic>P</italic>. <italic>falciparum</italic>" is one phrase -- but text across a *block*
# boundary must not be. This is the F15 fix: see the module docstring.
_BLOCK_TAGS = frozenset(
    {
        "abstract",
        "ack",
        "app",
        "article-title",
        "body",
        "boxed-text",
        "caption",
        "def",
        "def-item",
        "disp-formula",
        "disp-quote",
        "fig",
        "front",
        "label",
        "list",
        "list-item",
        "note",
        "p",
        "ref",
        "ref-list",
        "sec",
        "speech",
        "statement",
        "subtitle",
        "table",
        "table-wrap",
        "td",
        "th",
        "title",
        "tr",
        "trans-title",
        "verse-group",
        "verse-line",
    }
)


def _emit_text(element, out):
    """Append every text node under ``element`` to ``out``, newline-delimiting block elements."""
    block = _local(element.tag) in _BLOCK_TAGS
    if block:
        out.append("\n")
    if element.text:
        out.append(element.text)
    for child in element:
        _emit_text(child, out)
        if child.tail:
            out.append(child.tail)
    if block:
        out.append("\n")


def _text_of(element):
    """All text under ``element``, block-delimited then whitespace-collapsed. "" if missing."""
    if element is None:
        return ""
    out = []
    _emit_text(element, out)
    return _clean("".join(out))


def _structured_text(element):
    """As :func:`_text_of`, but keeping block boundaries as blank-line-separated paragraphs.

    Whole-article text is what the model actually reads; collapsing a 42,000-character body onto one
    unbroken line makes it materially harder to follow than the paragraphs the publisher wrote.
    """
    if element is None:
        return ""
    out = []
    _emit_text(element, out)
    lines = (_clean(line) for line in "".join(out).split("\n"))
    return "\n\n".join(line for line in lines if line)


def _paragraphs(element):
    """The paragraphs directly beneath ``element``, as one string, blank-line separated."""
    if element is None:
        return ""
    chunks = [_text_of(p) for p in _find_all(element, "p")]
    chunks = [chunk for chunk in chunks if chunk]
    return "\n\n".join(chunks)


class _TextExtractor(HTMLParser):
    """Strips tags from a PMC article page, dropping the parts that are not prose."""

    _SKIP = {"script", "style", "noscript", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if not self._depth and data.strip():
            self._parts.append(data)

    def text(self):
        return _clean(" ".join(self._parts))


def _strip_html(html):
    """Readable text from an HTML page, with a regex fallback if the markup defeats the parser."""
    try:
        parser = _TextExtractor()
        parser.feed(html)
        parser.close()
        text = parser.text()
        if text:
            return text
    except Exception as exc:  # a malformed page must not take the tool down
        logger.debug("HTML parse failed, falling back to regex strip: %s", exc)
    stripped = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    return _clean(re.sub(r"(?s)<[^>]+>", " ", stripped))


#: Abstract in a PMC page's meta tags. Upstream's equivalents are raw strings containing ``\s``,
#: which compiles to "backslash then one-or-more s" and matches nothing -- the F9 defect. These are
#: the same patterns with the escaping fixed, verified against a real ``<meta name="citation_abstract"
#: content="...">`` element on 2026-09-17.
_HTML_ABSTRACT_PATTERNS = (
    re.compile(r'(?is)<meta\s+name=["\']citation_abstract["\']\s+content=["\'](.*?)["\']\s*/?>'),
    re.compile(r'(?is)<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']\s*/?>'),
)

#: The abstract container, as an opening tag only: ``[^<>]`` stops at the next tag, so an unterminated
#: ``<div class=`` costs the distance to the next ``<`` rather than the rest of the page. The whole
#: container used to be one pattern, ``[^>]*`` then ``(.*?)</div>``, and both halves rescanned to
#: the end of the page from every start: 8000 unterminated tags took 5.9 s, quadratic in a page the
#: HTTP layer allows to be 8 MB (hunt 2026-09-30, uT6-literature-30).
_HTML_ABSTRACT_OPEN = re.compile(r'(?is)<(?:div|section)\b[^<>]*\bclass=["\'][^"\'<>]*abstract[^"\'<>]*["\'][^<>]*>')
_HTML_ABSTRACT_CLOSE = re.compile(r"(?i)</(?:div|section)\s*>")


def _abstract_from_html(html):
    """The abstract out of a PMC article page, or ``""``. See ``_HTML_ABSTRACT_PATTERNS`` (F9)."""
    html = html or ""
    for pattern in _HTML_ABSTRACT_PATTERNS:
        match = pattern.search(html)
        if match:
            text = _strip_html(match.group(1))
            if text:
                return text
    # Each opening tag runs to the first closing tag after it, as the old non-greedy pattern did. An
    # opening tag that starts before the last closing tag found shares that closing tag, so it is
    # skipped rather than searched again -- which is what keeps this linear.
    searched_to = -1
    for opening in _HTML_ABSTRACT_OPEN.finditer(html):
        if opening.start() < searched_to:
            continue
        closing = _HTML_ABSTRACT_CLOSE.search(html, opening.end())
        if closing is None:
            break  # no closing tag after this one, so none after any later one either
        searched_to = closing.end()
        text = _strip_html(html[opening.end() : closing.start()])
        if text:
            return text
    return ""


def _normalise_pmcid(value):
    """``"PMC7096075"`` or ``"7096075"`` -> ``("PMC7096075", "7096075")``; ``("", "")`` if unusable."""
    match = _PMCID_RE.match(str(value or "").strip())
    if not match:
        return "", ""
    digits = match.group(1)
    return f"PMC{digits}", digits


def _oai_error(xml_text):
    """OAI-PMH answers HTTP 200 for logical errors, so the envelope has to be read for one.

    ``<error code="idDoesNotExist">`` inside a 200 is how "no such article" arrives. Returns the
    message, or ``""`` when the record is real.
    """
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return "the OAI response was not well-formed XML"
    error = _find(root, "error")
    if error is None:
        return ""
    code = error.attrib.get("code", "unknown")
    return f"{code}: {_text_of(error) or 'no detail given'}"


#: PMC refuses bulk download for some publishers and says so in the body of an otherwise-200
#: ``efetch`` response. Detecting the sentence is what stops that prose being handed back as if it
#: were the article.
_EFETCH_REFUSAL = "does not allow downloading"


def _fetch_fulltext(digits, *, timeout=60):
    """Fetch an article's full text, trying four sources in turn.

    Returns ``(body, kind, trace)`` where ``kind`` is ``"jats"`` or ``"html"`` and ``body`` is ``""``
    if every source declined. ``trace`` is a list of ``{"source", "url", "status", "chars"}`` records
    -- one per attempt, including the failures, so an empty answer can be told apart from an article
    that genuinely has no body.
    """
    trace = []
    pmcid = f"PMC{digits}"

    attempts = (
        ("europe_pmc_fulltextxml", f"{_EUROPE_PMC}/{_seg(pmcid)}/fullTextXML", "jats"),
        (
            "ncbi_pmc_oai",
            "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
            f"?verb=GetRecord&metadataPrefix=pmc&identifier=oai:pubmedcentral.nih.gov:{_seg(digits)}",
            "jats",
        ),
        (
            "ncbi_efetch",
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={_seg(digits)}&retmode=xml",
            "jats",
        ),
        ("ncbi_pmc_html", f"https://pmc.ncbi.nlm.nih.gov/articles/{_seg(pmcid)}/", "html"),
    )

    for source, url, kind in attempts:
        body, failure = _fetch_text(url, timeout=timeout)
        if failure is not None:
            trace.append(
                {
                    "source": source,
                    "url": url,
                    "status": "failed",
                    "detail": failure["error"],
                    "http_status": failure.get("http_status"),
                }
            )
            continue
        body = body or ""
        if not body.strip():
            trace.append({"source": source, "url": url, "status": "empty", "chars": 0})
            continue
        if source == "ncbi_pmc_oai":
            oai_failure = _oai_error(body)
            if oai_failure:
                # OAI-PMH reports "no such record" inside an HTTP 200, so this branch is not dead
                # code -- without it the envelope itself would be parsed as if it were the article.
                trace.append({"source": source, "url": url, "status": "declined", "detail": oai_failure})
                continue
        if source == "ncbi_efetch" and _EFETCH_REFUSAL in body.lower():
            trace.append(
                {"source": source, "url": url, "status": "declined", "detail": "publisher blocks bulk download"}
            )
            continue
        trace.append({"source": source, "url": url, "status": "ok", "chars": len(body)})
        return body, kind, trace

    return "", "", trace


def _unanswered(step):
    """A ``_fetch_fulltext`` trace step where the source gave no answer: no connection, 429 or 5xx.

    A 404 is an answer about that one article, so it must not stop the enrichment of the others.
    """
    status = step.get("http_status")
    return step.get("status") == "failed" and (status is None or status == 429 or status >= 500)


def _resolve_pmcid(*, pmcid=None, pmid=None, article_id=None, source_db=None):
    """Work out the PMC digits to fetch from whichever identifier the caller had to hand.

    Returns ``(digits, error_payload_or_None)``. A PMID is resolved through Europe PMC's
    ``EXT_ID:`` search, because the PMC full-text endpoints are keyed on PMCID only.
    """
    # An article_id is a PMCID only when it says so -- source_db="PMC" or a "PMC" prefix. With no
    # source_db, bare digits were read as PMC digits, but a search hit's id for a MED record is its
    # PMID, so PMID 10592173 fetched PMC10592173, an unrelated article, and returned it as success
    # (hunt 2026-09-30, uT6-literature-11).
    source = str(source_db or "").strip().upper()
    article = str(article_id or "").strip()
    if source == "PPR":
        return "", _error(
            f"{article or 'This id'} is a Europe PMC preprint id (source_db='PPR'); the full-text tools read "
            "PubMed Central articles only. If the preprint was published, search for the article and pass "
            "its pmcid; otherwise europe_pmc_search_articles returns its abstract."
        )
    as_pmcid = article if source == "PMC" or (not source and article.upper().startswith("PMC")) else None
    for candidate in (pmcid, as_pmcid):
        _, digits = _normalise_pmcid(candidate)
        if digits:
            return digits, None

    as_pmid = article if source in ("MED", "PMID") or (not source and article.isdigit()) else None
    lookup = pmid or as_pmid
    if not lookup:
        return "", _missing(
            "pmcid",
            "Pass pmcid='PMC7096075', or pmid='32226684' and it will be resolved. "
            "europe_pmc_search_articles returns both on every hit.",
        )

    payload, failure = _fetch_json(
        f"{_EUROPE_PMC}/search",
        params={"query": f"EXT_ID:{lookup} AND SRC:MED", "resultType": "core", "format": "json", "pageSize": 1},
    )
    if failure is not None:
        return "", failure
    results = ((payload or {}).get("resultList") or {}).get("result") or []
    if not results:
        return "", _error(f"No Europe PMC record found for PMID {lookup}.")
    _, digits = _normalise_pmcid(results[0].get("pmcid"))
    if not digits:
        return "", _error(
            f"PMID {lookup} has no PMC identifier, so no open-access full text is available for it. "
            "The abstract is still reachable with europe_pmc_search_articles."
        )
    return digits, None


def _hit(record):
    """One Europe PMC search record, projected to the fields a literature question actually uses."""
    journal = ((record.get("journalInfo") or {}).get("journal") or {}).get("title")
    return {
        "id": record.get("id"),
        "source": record.get("source"),
        "pmid": record.get("pmid"),
        "pmcid": record.get("pmcid"),
        "doi": record.get("doi"),
        "title": _clean(record.get("title")),
        "authors": _clean(record.get("authorString")),
        # F11: upstream runs a second `resultType=lite` search purely to read a journal title.
        # `core` already carries it here -- measured live 2026-09-17, journalInfo.journal.title is
        # "The oncologist" for the record whose lite journalTitle is "Oncologist".
        "journal": _clean(journal) or _clean(record.get("journalTitle")),
        "year": record.get("pubYear"),
        "abstract": _clean(record.get("abstractText")),
        "is_open_access": record.get("isOpenAccess") == "Y",
        # Full text means Europe PMC holds the body or the record has a PMCID. hasTextMinedTerms is
        # set for abstract-only records too, so it made an abstract-only MED record "has_fulltext"
        # and sent the agent to fulltext tools that then had no PMCID to fetch (hunt 2026-09-30,
        # uT6-literature-33). It is still reported, under its own name.
        "has_fulltext": record.get("inEPMC") == "Y" or bool(record.get("pmcid")),
        "has_text_mined_terms": record.get("hasTextMinedTerms") == "Y",
        "cited_by_count": _as_int(record.get("citedByCount")),
    }


def europe_pmc_search_articles(
    query,
    limit=5,
    require_fulltext=False,
    fulltext_terms=None,
    enrich_missing_abstract=False,
):
    """Search Europe PMC for biomedical articles and return their metadata and abstracts.

    The first stop for "what has been published about X". Europe PMC indexes PubMed, PMC, preprints
    and patents, and is the only one of this module's three services that can also hand back full
    text (see :func:`europe_pmc_get_fulltext`).

    Parameters
    ----------
    query : str
        Europe PMC query syntax. Plain words work ("spatial transcriptomics tumour microenvironment")
        and so do field terms: ``AUTH:"Smith J"``, ``JOURNAL:"Nature Methods"``, ``PUB_YEAR:2024``,
        ``DOI:"10.1038/..."``, ``EXT_ID:32226684``, combined with AND/OR/NOT.
    limit : int, optional
        How many records to return, 1-100. Default 5.
    require_fulltext : bool, optional
        Restrict to articles whose full text Europe PMC can serve. Default False. Turn it on when
        the next step is reading methods rather than skimming abstracts.
    fulltext_terms : list of str, optional
        Terms that must appear in the article's **full text**, not merely its abstract -- useful for
        a method or reagent that is only ever named in the methods section. Several terms are OR-ed,
        so the hit needs any one of them. Only records whose full text Europe PMC has indexed can
        match, which is narrower than "has a free full-text link somewhere".
    enrich_missing_abstract : bool, optional
        For hits that have no abstract in the index but do have a PMCID, fetch the article and pull
        the abstract out of it. Default False, because it costs one extra request per such hit.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "num_found": int, "returned": int, "truncated": bool}``.
        ``num_found`` is Europe PMC's own total, so ``truncated`` says plainly when there is more
        behind the page you asked for.

    Examples
    --------
    >>> hits = europe_pmc_search_articles(query="Visium spatial transcriptomics glioma", limit=5)
    >>> for hit in hits["data"]:
    ...     print(hit["year"], hit["journal"], "|", hit["title"], "| PMCID", hit["pmcid"])
    >>> print(europe_pmc_search_articles(query='AUTH:"Regev A" AND PUB_YEAR:2024', limit=3))
    """
    if not query:
        return _missing(
            "query",
            "Pass what to search for, for example query='spatial transcriptomics tumour microenvironment'.",
        )

    clauses = [f"({query})"]
    if require_fulltext:
        clauses.append("(HAS_FT:Y)")
    # The indexed-full-text field is BODY. Europe PMC answers an unknown field name with HTTP 200
    # and hitCount 0 rather than an error, so a misspelt field is indistinguishable from "no such
    # paper" at the call site -- measured 2026-09-17: BODY:"Xenium" -> 1312 hits, while
    # FULL_TEXT:"Xenium" and NOT_A_FIELD:"Xenium" both -> 0. Terms are OR-ed, so several of them
    # mean "mentions any of these", which is what sweeping for a method or reagent wants.
    body_terms = [term for term in (_clean(item) for item in _as_terms(fulltext_terms)) if term]
    if body_terms:
        joined = " OR ".join('BODY:"{}"'.format(term.replace('"', '\\"')) for term in body_terms)
        clauses.append(f"({joined})")

    page_size = _page_size(limit, default=5)
    payload, failure = _fetch_json(
        f"{_EUROPE_PMC}/search",
        params={
            "query": " AND ".join(clauses),
            # F11: one request. Upstream issues this *and* a second resultType=lite search solely to
            # read a journal title that this response already carries.
            "resultType": "core",
            "format": "json",
            "pageSize": page_size,
        },
    )
    if failure is not None:
        return failure

    records = ((payload or {}).get("resultList") or {}).get("result") or []
    hits = [_hit(record) for record in records]

    enriched = 0
    skipped = 0
    stop_reason = ""
    if enrich_missing_abstract:
        # Bounded. Each enrichment is a four-source fetch chain, and it ran serially for every
        # abstract-less hit -- up to 100 -- with no overall budget, so an unreachable NCBI or Europe
        # PMC held the call for hours (hunt 2026-09-30, uT6-literature-31). At most
        # _ENRICH_MAX_HITS are tried, none is started past _ENRICH_BUDGET_SECONDS, and a hit whose
        # every source failed to answer stops the rest; abstracts_skipped counts what was not tried.
        started = time.monotonic()
        attempted = 0
        for hit in hits:
            if hit["abstract"] or not hit["pmcid"]:
                continue
            _, digits = _normalise_pmcid(hit["pmcid"])
            if not digits:
                continue
            if not stop_reason and attempted >= _ENRICH_MAX_HITS:
                stop_reason = f"at most {_ENRICH_MAX_HITS} abstracts are fetched per search"
            elif not stop_reason and time.monotonic() - started > _ENRICH_BUDGET_SECONDS:
                stop_reason = f"the {_ENRICH_BUDGET_SECONDS:.0f}s enrichment budget ran out"
            if stop_reason:
                skipped += 1
                continue
            attempted += 1
            body, kind, trace = _fetch_fulltext(digits)
            if not body:
                if trace and all(_unanswered(step) for step in trace):
                    stop_reason = "no full-text source answered, so the remaining hits were not tried"
                continue
            if kind == "html":
                # F9: upstream's HTML abstract patterns can never match, so this branch was dead and
                # an article only reachable as HTML was always left without an abstract.
                hit["abstract"] = _abstract_from_html(body)
            else:
                try:
                    hit["abstract"] = _paragraphs(_find(ElementTree.fromstring(body), "abstract"))
                except ElementTree.ParseError:
                    hit["abstract"] = ""
            if hit["abstract"]:
                hit["abstract_source"] = "fulltext"
                enriched += 1

    total = _as_int((payload or {}).get("hitCount"))
    result = _ok(
        hits,
        num_found=total,
        returned=len(hits),
        truncated=total > len(hits),
        query=" AND ".join(clauses),
    )
    if enrich_missing_abstract:
        result["abstracts_enriched"] = enriched
        result["abstracts_skipped"] = skipped
        if skipped:
            result["enrichment_note"] = (
                f"{skipped} hit(s) with a pmcid but no abstract were not enriched: {stop_reason}. Fetch one "
                "with europe_pmc_get_structured_fulltext if you need it."
            )
    return result


def _citation_page(article_id, source, page_size, page, relation):
    if not article_id:
        return _missing(
            "article_id",
            "Pass the article's identifier, for example article_id='32226684' with source='MED', "
            "or article_id='PMC7096075' with source='PMC'.",
        )
    source = (str(source or "MED").strip() or "MED").upper()
    payload, failure = _fetch_json(
        f"{_EUROPE_PMC}/{_seg(source)}/{_seg(article_id)}/{_seg(relation)}",
        params={"format": "json", "pageSize": _page_size(page_size, default=25), "page": max(1, _as_int(page, 1))},
    )
    if failure is not None:
        # Europe PMC takes these two endpoints down for maintenance independently of search, and
        # answers 503 with "This API is temporarily unavailable due to maintenance." That is worth
        # distinguishing from "this article has no citations", which is why the payload carries
        # retryable rather than an empty list.
        return failure

    container = (payload or {}).get(f"{relation[:-1]}List") or {}
    records = container.get(relation[:-1]) or []
    projected = [
        {
            "id": record.get("id"),
            "source": record.get("source"),
            "title": _clean(record.get("title")),
            "authors": _clean(record.get("authorString")),
            "journal": _clean(record.get("journalAbbreviation")),
            "year": record.get("pubYear"),
            "doi": record.get("doi"),
            "pmcid": record.get("pmcid"),
            "cited_by_count": _as_int(record.get("citedByCount")),
        }
        for record in records
    ]
    total = _as_int((payload or {}).get("hitCount"))
    return _ok(projected, num_found=total, returned=len(projected), truncated=total > len(projected))


def europe_pmc_get_citations(article_id, source="MED", page_size=25, page=1):
    """List the articles that cite a given article.

    Forward citation search: who built on this paper. Pair it with
    :func:`europe_pmc_get_references` to walk a citation graph in both directions.

    Parameters
    ----------
    article_id : str
        The article's identifier in ``source``: a PMID for ``MED`` ("32226684"), a PMCID for ``PMC``
        ("PMC7096075").
    source : str, optional
        Which database ``article_id`` belongs to. ``"MED"`` (PubMed, the default), ``"PMC"``,
        ``"PPR"`` (preprints), ``"AGR"``, ``"CBA"``, ``"PAT"``.
    page_size : int, optional
        Records per page, 1-100. Default 25.
    page : int, optional
        1-based page number. Default 1.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "num_found": int, "returned": int, "truncated": bool}``,
        each record carrying id, source, title, authors, journal, year, doi, pmcid and
        cited_by_count.

    Examples
    --------
    >>> citing = europe_pmc_get_citations(article_id="32226684", source="MED")
    >>> print(citing["num_found"], "citing articles")
    >>> for record in citing["data"][:5]:
    ...     print(record["year"], record["title"])
    """
    return _citation_page(article_id, source, page_size, page, "citations")


def europe_pmc_get_references(article_id, source="MED", page_size=25, page=1):
    """List the articles a given article cites -- its reference list.

    Backward citation search: what this paper was built on. The mirror of
    :func:`europe_pmc_get_citations`.

    Parameters
    ----------
    article_id : str
        The article's identifier in ``source``: a PMID for ``MED``, a PMCID for ``PMC``.
    source : str, optional
        ``"MED"`` (default), ``"PMC"``, ``"PPR"``, ``"AGR"``, ``"CBA"``, ``"PAT"``.
    page_size : int, optional
        Records per page, 1-100. Default 25.
    page : int, optional
        1-based page number. Default 1.

    Returns
    -------
    dict
        Same shape as :func:`europe_pmc_get_citations`. Note that Europe PMC takes this endpoint
        down for maintenance from time to time independently of the rest of the service; when it
        does, the result is ``{"status": "error", ..., "retryable": True}`` rather than an empty
        reference list, so an outage is never read as "this paper cites nothing".

    Examples
    --------
    >>> refs = europe_pmc_get_references(article_id="32226684", source="MED")
    >>> print(refs["num_found"], "references")
    >>> print([r["doi"] for r in refs["data"][:5]])
    """
    return _citation_page(article_id, source, page_size, page, "references")


def europe_pmc_get_fulltext(
    pmcid=None,
    pmid=None,
    article_id=None,
    source_db=None,
    output_format="text",
    max_chars=_DEFAULT_MAX_CHARS,
):
    """Fetch an open-access article's full text, trying four sources until one answers.

    Use this when the abstract is not enough -- a methods detail, a reagent, a parameter, a cohort
    size. Only open-access articles have retrievable full text; everything else returns an error
    saying so rather than an empty string.

    The sources, in order: Europe PMC ``fullTextXML``, the NCBI PMC OAI record, NCBI ``efetch``, and
    the PMC article page. Each attempt appears in the returned ``trace``, so an empty result can be
    told apart from an article with no body.

    Parameters
    ----------
    pmcid : str, optional
        ``"PMC7096075"`` or ``"7096075"``.
    pmid : str, optional
        A PubMed ID, resolved to a PMCID for you. Supply one of ``pmcid`` or ``pmid``.
    article_id : str, optional
        An identifier whose database is given by ``source_db`` -- the shape
        :func:`europe_pmc_search_articles` returns.
    source_db : str, optional
        ``"PMC"`` or ``"MED"``, saying which of those ``article_id`` is.
    output_format : str, optional
        ``"text"`` (default) for readable prose, or ``"raw"`` for the JATS XML / HTML as fetched.
    max_chars : int, optional
        Cap on the returned body. Default 200000. Truncation is reported on the result, never
        silent.

    Returns
    -------
    dict
        ``{"status": "success", "data": {"pmcid", "format", "source", "text", "chars",
        "truncated"}, "trace": [...]}``.

    Examples
    --------
    >>> article = europe_pmc_get_fulltext(pmcid="PMC7096075")
    >>> print(article["data"]["source"], article["data"]["chars"], "characters")
    >>> print(article["data"]["text"][:2000])
    >>> print([step["source"] + ":" + step["status"] for step in article["trace"]])
    """
    digits, failure = _resolve_pmcid(pmcid=pmcid, pmid=pmid, article_id=article_id, source_db=source_db)
    if failure is not None:
        return failure

    body, kind, trace = _fetch_fulltext(digits)
    if not body:
        return _error(
            f"No full text could be retrieved for PMC{digits}. All four sources were tried; see "
            "trace for what each returned. Most often this means the article is not open access.",
            trace=trace,
        )

    if str(output_format).lower() == "raw":
        text = body
    elif kind == "html":
        text = _strip_html(body)
    else:
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError as exc:
            return _error(f"The full text for PMC{digits} did not parse as XML: {exc}", trace=trace)
        # Namespace-agnostic, so this reads the OAI record as happily as the Europe PMC one (F13).
        sections = [_structured_text(_find(root, "abstract")), _structured_text(_find(root, "body"))]
        text = "\n\n".join(chunk for chunk in sections if chunk) or _structured_text(root)

    limit = max(1, _as_int(max_chars, _DEFAULT_MAX_CHARS))
    truncated = len(text) > limit
    used = next((step["source"] for step in trace if step.get("status") == "ok"), "")
    return _ok(
        {
            "pmcid": f"PMC{digits}",
            "format": "raw" if str(output_format).lower() == "raw" else "text",
            "source": used,
            "text": text[:limit],
            "chars": len(text[:limit]),
            "total_chars": len(text),
            "truncated": truncated,
        },
        trace=trace,
    )


def europe_pmc_get_fulltext_snippets(
    terms,
    pmcid=None,
    pmid=None,
    article_id=None,
    source_db=None,
    window_chars=220,
    max_snippets_per_term=3,
    max_total_chars=8000,
):
    """Pull the passages around given keywords out of an article's full text.

    This is the tool for "what does this paper say about X" when the article is long and only a few
    sentences matter -- a reagent, a parameter, a cell type, a software name. Far cheaper than
    reading the whole body through :func:`europe_pmc_get_fulltext`.

    Parameters
    ----------
    terms : list of str
        Words or phrases to find. Matching is case-insensitive and literal, not regex.
    pmcid : str, optional
        ``"PMC7096075"`` or ``"7096075"``.
    pmid : str, optional
        A PubMed ID, resolved for you. Supply one of ``pmcid`` or ``pmid``.
    article_id, source_db : str, optional
        The identifier-plus-database form :func:`europe_pmc_search_articles` returns.
    window_chars : int, optional
        Characters of context on each side of a match. Default 220.
    max_snippets_per_term : int, optional
        Cap per term, so one common word cannot crowd out the rest. Default 3.
    max_total_chars : int, optional
        Overall budget across every snippet. Default 8000.

    Returns
    -------
    dict
        ``{"status": "success", "data": {"pmcid", "snippets": {term: [...]}, "counts": {term: int}},
        "truncated": bool, "trace": [...]}``. ``counts`` is how many times each term occurs in the
        whole article, which is often the answer on its own. ``truncated`` is ``True`` whenever a
        match was found but not returned -- including when the budget ran out mid-article, which
        upstream reports as ``False``.

    Examples
    --------
    >>> found = europe_pmc_get_fulltext_snippets(terms=["Visium", "10x Genomics"], pmcid="PMC7096075")
    >>> print(found["data"]["counts"])
    >>> for term, snippets in found["data"]["snippets"].items():
    ...     for snippet in snippets:
    ...         print(term, "->", snippet)
    """
    wanted = [_clean(term) for term in _as_terms(terms) if _clean(term)]
    if not wanted:
        return _missing("terms", "Pass the words to look for, for example terms=['Visium', 'DAPI'].", plural=True)

    digits, failure = _resolve_pmcid(pmcid=pmcid, pmid=pmid, article_id=article_id, source_db=source_db)
    if failure is not None:
        return failure

    body, kind, trace = _fetch_fulltext(digits)
    if not body:
        return _error(
            f"No full text could be retrieved for PMC{digits}, so there is nothing to search. "
            "See trace for what each source returned.",
            trace=trace,
        )

    if kind == "html":
        text = _strip_html(body)
    else:
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError:
            text = _strip_html(body)
        else:
            # Abstract and body only. Upstream searches the whole document, so a term occurring
            # nowhere but a cited paper's title in <ref-list> counts as a hit and can fill the
            # snippet budget with bibliography; <front> contributes publisher metadata that reads
            # as noise ("2757cureusCureusCureus Inc.PMC7096075..." was a real first snippet).
            scoped = [_text_of(_find(root, "abstract")), _text_of(_find(root, "body"))]
            text = " ".join(chunk for chunk in scoped if chunk) or _text_of(root)

    haystack = text.lower()
    window = max(20, _as_int(window_chars, 220))
    per_term = max(1, _as_int(max_snippets_per_term, 3))
    budget = max(200, _as_int(max_total_chars, 8000))

    snippets = {}
    counts = {}
    spent = 0
    truncated = False
    for term in wanted:
        needle = term.lower()
        positions = []
        start = haystack.find(needle)
        while start != -1:
            positions.append(start)
            start = haystack.find(needle, start + len(needle))
        counts[term] = len(positions)
        collected = []
        for position in positions:
            if len(collected) >= per_term:
                # More matches exist than the per-term cap allows; say so rather than imply the
                # article contains only these.
                truncated = True
                break
            left = max(0, position - window)
            right = min(len(text), position + len(term) + window)
            snippet = _clean(text[left:right])
            if spent + len(snippet) > budget:
                # F10: upstream breaks here and then computes truncated from the running total,
                # which is still below the budget precisely because this snippet was skipped -- so
                # the caller is told nothing was dropped. Set it where the drop happens.
                truncated = True
                break
            collected.append(snippet)
            spent += len(snippet)
        snippets[term] = collected

    return _ok(
        {"pmcid": f"PMC{digits}", "snippets": snippets, "counts": counts, "chars_returned": spent},
        truncated=truncated,
        trace=trace,
    )


def _bucket_for(title, sec_type):
    """Which named bucket a JATS ``<sec>`` belongs in, from its ``sec-type`` or its title."""
    text = f"{sec_type or ''} {title or ''}".lower()
    for bucket, keywords in _SECTION_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return bucket
    return "other"


def europe_pmc_get_structured_fulltext(pmcid=None, pmid=None, max_section_chars=50000):
    """Fetch an article's full text already split into introduction, methods, results and discussion.

    The shape most literature questions actually want. "How was the tissue prepared?" is a methods
    question; handing the model the whole article to find that is wasteful and unreliable. This
    returns the body as named sections, so you can read one.

    Parameters
    ----------
    pmcid : str, optional
        ``"PMC7096075"`` or ``"7096075"``.
    pmid : str, optional
        A PubMed ID, resolved to a PMCID for you. Supply one of ``pmcid`` or ``pmid``.
    max_section_chars : int, optional
        Cap per section. Default 50000. Truncation is reported per section, never silent.

    Returns
    -------
    dict
        ``{"status": "success", "data": {"pmcid", "title", "abstract", "sections", "section_titles",
        "counts"}, "trace": [...]}``. ``sections`` is keyed by ``introduction``, ``methods``,
        ``results``, ``discussion``, ``conclusion``, ``other``, and ``unsectioned`` for body text
        that sits in no ``<sec>`` at all (a Brief Communication's whole main text); ``section_titles``
        keeps the article's own headings so nothing is lost to the bucketing.

    Notes
    -----
    Parsing is namespace-agnostic, which is not cosmetic. Europe PMC's ``fullTextXML`` has a
    namespace-free ``<article>`` root, while the NCBI OAI fallback wraps JATS in an OAI-PMH envelope
    under a JATS namespace. Upstream's bare ``find(".//abstract")`` matches only the first, so
    against the fallback it returns ``status: "success"`` with an empty title, empty abstract and no
    sections -- exactly in the case the fallback exists to cover. Confirmed live on PMC13521112,
    where Europe PMC answers 500 and the OAI record carries the complete article.

    Examples
    --------
    >>> article = europe_pmc_get_structured_fulltext(pmcid="PMC7096075")
    >>> print(article["data"]["title"])
    >>> print(article["data"]["counts"])
    >>> print(article["data"]["sections"]["methods"][:3000])
    """
    digits, failure = _resolve_pmcid(pmcid=pmcid, pmid=pmid)
    if failure is not None:
        return failure

    body_text, kind, trace = _fetch_fulltext(digits)
    if not body_text:
        return _error(
            f"No full text could be retrieved for PMC{digits}. See trace for what each source returned.",
            trace=trace,
        )
    if kind != "jats":
        return _error(
            f"Only the HTML article page was available for PMC{digits}, which carries no section "
            "structure. Use europe_pmc_get_fulltext for the prose, or "
            "europe_pmc_get_fulltext_snippets to search it.",
            trace=trace,
        )

    try:
        root = ElementTree.fromstring(body_text)
    except ElementTree.ParseError as exc:
        return _error(f"The full text for PMC{digits} did not parse as XML: {exc}", trace=trace)

    title_group = _find(root, "title-group")
    title = _text_of(_find(title_group, "article-title") if title_group is not None else _find(root, "article-title"))
    abstract = _paragraphs(_find(root, "abstract")) or _text_of(_find(root, "abstract"))

    body = _find(root, "body")
    limit = max(500, _as_int(max_section_chars, 50000))
    sections = {}
    section_titles = []
    truncated_sections = []
    if body is not None:
        for element in body:
            if _local(element.tag) != "sec":
                # Text straight under <body> -- the whole main text of a Brief Communication or a
                # Letter, and the lead-in of many articles -- was skipped, and the result still said
                # success with counts that showed no sign of it (hunt 2026-09-30, uT6-literature-7).
                bucket = "unsectioned"
                chunk = _text_of(element)
                if not chunk:
                    continue
            else:
                heading = _text_of(_find(element, "title"))
                section_titles.append(heading)
                bucket = _bucket_for(heading, element.attrib.get("sec-type"))
                chunk = _text_of(element)
            merged = f"{sections.get(bucket, '')}\n\n{chunk}".strip() if bucket in sections else chunk
            if len(merged) > limit:
                merged = merged[:limit]
                if bucket not in truncated_sections:
                    truncated_sections.append(bucket)
            sections[bucket] = merged

    return _ok(
        {
            "pmcid": f"PMC{digits}",
            "title": title,
            "abstract": abstract,
            "sections": sections,
            "section_titles": section_titles,
            "counts": {name: len(text) for name, text in sections.items()},
            "truncated_sections": truncated_sections,
        },
        trace=trace,
    )


# --------------------------------------------------------------------------------------------- #
# OpenAlex
# --------------------------------------------------------------------------------------------- #

#: An OpenAlex entity id is a letter and digits. It turns up bare ("W2005501262"), as a URL
#: ("https://openalex.org/W2005501262") and lower-cased; all three are accepted and normalised.
_OPENALEX_ID_RE = re.compile(r"([WAISCPFT]\d+)\s*$", re.IGNORECASE)


def _openalex_id(value):
    """``"https://openalex.org/w2005501262"`` -> ``"W2005501262"``; ``""`` if it is not an id."""
    match = _OPENALEX_ID_RE.search(str(value or "").strip())
    return match.group(1).upper() if match else ""


#: Every resolver spelling a reference list uses. The prefix list missed "doi: 10.x" (the space
#: survived and was sent as %20), http://dx.doi.org/, https://www.doi.org/ and a bare doi.org/, so a
#: real DOI was requested at a path no registry has and reported as unregistered (hunt 2026-09-30,
#: uT6-literature-3).
_DOI_PREFIX_RE = re.compile(r"^(?:doi:\s*|(?:https?://)?(?:www\.|dx\.)?doi\.org/)", re.IGNORECASE)


def _bare_doi(value):
    """Strip any resolver prefix from a DOI: ``"https://doi.org/10.1/x"`` -> ``"10.1/x"``."""
    text = str(value or "").strip()
    return _DOI_PREFIX_RE.sub("", text, count=1).strip()


def _invert_abstract(inverted_index):
    """Rebuild an abstract from OpenAlex's ``{word: [positions]}`` inverted index.

    F12: upstream writes into a fixed ``[""] * 500`` buffer and drops every position past 500 with
    no flag. Measured live on 2026-09-17, work ``W2005501262`` has 4,055 positions -- 87.7% of that
    abstract silently discarded. The buffer here is sized from the data, so the abstract is whole or
    it is absent.
    """
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    highest = -1
    for positions in inverted_index.values():
        for position in positions or []:
            index = _as_int(position, -1)
            if index > highest:
                highest = index
    if highest < 0:
        return ""
    words = [""] * (highest + 1)
    for word, positions in inverted_index.items():
        for position in positions or []:
            index = _as_int(position, -1)
            if 0 <= index <= highest:
                words[index] = word
    return _clean(" ".join(word for word in words if word))


def _openalex_get(path, params=None):
    """GET an OpenAlex path. No API key and no ``mailto`` -- see the module docstring (F14)."""
    return _fetch_json(f"{_OPENALEX}{path}", params=params or {})


def _work_summary(record, author_cap=25):
    """One OpenAlex work. ``author_cap=None`` keeps every author, for a single-record fetch.

    ``authors_total`` and ``authors_truncated`` say when the list is not the whole author list. It
    was cut at 25 everywhere with no sign, so a consortium paper's 200 authors came back as 25 and
    "is X an author" was answered no (hunt 2026-09-30, uT6-literature-14).
    """
    record = record or {}
    primary = record.get("primary_location") or {}
    source = primary.get("source") or {}
    authorships = record.get("authorships") or []
    authors = [_clean((a.get("author") or {}).get("display_name")) for a in authorships]
    institutions = sorted(
        {
            _clean(inst.get("display_name"))
            for a in authorships
            for inst in (a.get("institutions") or [])
            if inst.get("display_name")
        }
    )
    return {
        "openalex_id": _openalex_id(record.get("id")),
        "doi": _bare_doi(record.get("doi")),
        "title": _clean(record.get("display_name") or record.get("title")),
        "year": record.get("publication_year"),
        "date": record.get("publication_date"),
        "type": record.get("type"),
        "venue": _clean(source.get("display_name")),
        "issn_l": source.get("issn_l"),
        "authors": authors[:author_cap],
        "authors_total": len(authors),
        # OpenAlex cuts authorships at 100 in list responses and says so in is_authors_truncated.
        "authors_truncated": len(authors[:author_cap]) < len(authors) or bool(record.get("is_authors_truncated")),
        "institutions": institutions[:author_cap],
        "cited_by_count": _as_int(record.get("cited_by_count")),
        "is_open_access": bool((record.get("open_access") or {}).get("is_oa")),
        "oa_url": (record.get("open_access") or {}).get("oa_url"),
        "concepts": [_clean(c.get("display_name")) for c in (record.get("concepts") or [])][:10],
        "referenced_works_count": len(record.get("referenced_works") or []),
        "abstract": _invert_abstract(record.get("abstract_inverted_index")),
    }


#: OpenAlex filter grammar: "," separates clauses, "|" ORs values and a leading "!" negates. There is
#: no escape, so a term carrying one is refused by name rather than spliced in -- "10x Genomics, Visium"
#: became a second, malformed clause, "a|b" silently ORed and "!x" silently negated (hunt 2026-09-30,
#: uT6-literature-29, formerly LOW-10).
_OPENALEX_FILTER_SYNTAX = ",|!"


def _fulltext_filters(require_has_fulltext, fulltext_terms):
    """``(clauses, None)`` constraining a works query to searchable full text, or ``(None, error)``."""
    clauses = []
    if require_has_fulltext:
        clauses.append("has_fulltext:true")
    for term in _as_terms(fulltext_terms):
        cleaned = _clean(term)
        if not cleaned:
            continue
        bad = sorted({character for character in cleaned if character in _OPENALEX_FILTER_SYNTAX})
        if bad:
            return None, _error(
                f"fulltext_terms cannot contain {' '.join(repr(character) for character in bad)} -- those "
                f"characters are OpenAlex filter syntax and cannot be escaped. Pass each term on its own, "
                f"e.g. fulltext_terms=['10x Genomics', 'Visium'] (every term must match)."
            )
        clauses.append(f"fulltext.search:{cleaned}")
    return clauses, None


def openalex_literature_search(
    query,
    max_results=10,
    year_from=None,
    year_to=None,
    open_access=None,
    require_has_fulltext=False,
    fulltext_terms=None,
):
    """Search OpenAlex for scholarly works, with year, open-access and full-text filters.

    OpenAlex covers every discipline, not only biomedicine, and carries citation counts, concepts,
    author affiliations and open-access status on every record. Use it when the question is about
    influence, provenance or coverage; use :func:`europe_pmc_search_articles` when the question is
    biomedical and the next step is reading the paper.

    No API key is needed and none is read -- see the module docstring.

    Parameters
    ----------
    query : str
        Free-text search across title, abstract and full text where available.
    max_results : int, optional
        How many works to return, 1-100. Default 10.
    year_from, year_to : int, optional
        Inclusive publication-year bounds.
    open_access : bool, optional
        ``True`` for open-access works only, ``False`` for closed only, ``None`` (default) for both.
    require_has_fulltext : bool, optional
        Restrict to works whose full text OpenAlex has indexed. Default False.
    fulltext_terms : list of str, optional
        Terms that must appear in the indexed full text, not merely the abstract.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "num_found": int, "returned": int, "truncated": bool}``.
        Abstracts are rebuilt from OpenAlex's inverted index; many records simply have none, in which
        case ``abstract`` is ``""`` and Europe PMC is the better source.

    Examples
    --------
    >>> works = openalex_literature_search(query="spatial transcriptomics", year_from=2023, max_results=5)
    >>> for work in works["data"]:
    ...     print(work["cited_by_count"], work["year"], work["title"])
    >>> print(openalex_literature_search(query="Xenium in situ", open_access=True)["num_found"])
    """
    if not query:
        return _missing("query", "Pass what to search for, for example query='spatial transcriptomics'.")

    filters, failure = _fulltext_filters(require_has_fulltext, fulltext_terms)
    if failure is not None:
        return failure
    if year_from is not None:
        filters.append(f"from_publication_date:{_as_int(year_from, 1900)}-01-01")
    if year_to is not None:
        filters.append(f"to_publication_date:{_as_int(year_to, 2100)}-12-31")
    if open_access is not None:
        filters.append(f"is_oa:{'true' if open_access else 'false'}")

    params = {"search": query, "per-page": _page_size(max_results, default=10), "page": 1}
    if filters:
        params["filter"] = ",".join(filters)

    payload, failure = _openalex_get("/works", params)
    if failure is not None:
        return failure

    works = [_work_summary(record) for record in (payload or {}).get("results") or []]
    total = _as_int(((payload or {}).get("meta") or {}).get("count"))
    return _ok(works, num_found=total, returned=len(works), truncated=total > len(works))


def openalex_search_works(
    search=None,
    filter=None,  # OpenAlex's own parameter name; renaming it would obscure their docs
    per_page=10,
    page=1,
    sort=None,
    require_has_fulltext=False,
    fulltext_terms=None,
):
    """Query the OpenAlex works endpoint directly, with its own filter and sort syntax.

    The escape hatch beneath :func:`openalex_literature_search`: use it when you need a filter that
    the convenience wrapper does not expose, or a non-default sort.

    Parameters
    ----------
    search : str, optional
        Free-text search. Optional, because a pure ``filter`` query is legitimate.
    filter : str, optional
        OpenAlex filter syntax, comma-separated for AND:
        ``"institutions.country_code:us,publication_year:2024"``,
        ``"cited_by_count:>100"``, ``"authorships.author.id:A5023888391"``.
    per_page : int, optional
        Records per page, 1-100. Default 10.
    page : int, optional
        1-based page number. Default 1.
    sort : str, optional
        ``"cited_by_count:desc"``, ``"publication_date:desc"``, ``"relevance_score:desc"``.
    require_has_fulltext : bool, optional
        Add ``has_fulltext:true`` to the filter. Default False.
    fulltext_terms : list of str, optional
        Add ``fulltext.search:`` clauses to the filter.

    Returns
    -------
    dict
        Same shape as :func:`openalex_literature_search`.

    Examples
    --------
    >>> top = openalex_search_works(search="Visium", sort="cited_by_count:desc", per_page=5)
    >>> print([(w["cited_by_count"], w["title"]) for w in top["data"]])
    >>> print(openalex_search_works(filter="publication_year:2025,is_oa:true", search="MERFISH")["num_found"])
    """
    if not search and not filter:
        return _error(
            "Pass search, filter, or both. For example search='Visium', or filter='publication_year:2025,is_oa:true'."
        )

    clauses = [clause for clause in str(filter or "").split(",") if clause.strip()]
    fulltext_clauses, failure = _fulltext_filters(require_has_fulltext, fulltext_terms)
    if failure is not None:
        return failure
    clauses.extend(fulltext_clauses)

    params = {"per-page": _page_size(per_page, default=10), "page": max(1, _as_int(page, 1))}
    if search:
        params["search"] = search
    if clauses:
        params["filter"] = ",".join(clause.strip() for clause in clauses)
    if sort:
        params["sort"] = sort

    payload, failure = _openalex_get("/works", params)
    if failure is not None:
        return failure

    works = [_work_summary(record) for record in (payload or {}).get("results") or []]
    total = _as_int(((payload or {}).get("meta") or {}).get("count"))
    return _ok(works, num_found=total, returned=len(works), truncated=total > len(works))


def openalex_get_work(openalex_id):
    """Fetch one OpenAlex work by its OpenAlex identifier.

    Parameters
    ----------
    openalex_id : str
        ``"W2005501262"``, or the full ``"https://openalex.org/W2005501262"``.

    Returns
    -------
    dict
        ``{"status": "success", "data": {...}}`` with the same projection the search tools return,
        plus the reconstructed abstract.

    Examples
    --------
    >>> work = openalex_get_work(openalex_id="W2005501262")
    >>> print(work["data"]["title"], "|", work["data"]["cited_by_count"], "citations")
    >>> print(work["data"]["abstract"][:500])
    """
    identifier = _openalex_id(openalex_id)
    if not identifier:
        return _missing(
            "openalex_id",
            "Pass an OpenAlex work id, for example openalex_id='W2005501262'. The search tools "
            "return one on every hit.",
        )
    payload, failure = _openalex_get(f"/works/{_seg(identifier)}")
    if failure is not None:
        return failure
    return _ok(_work_summary(payload, author_cap=None))


def openalex_get_work_by_doi(doi):
    """Fetch one OpenAlex work by its DOI -- the usual way in from a reference list.

    Parameters
    ----------
    doi : str
        ``"10.1038/s41586-020-2649-2"``, with or without a ``https://doi.org/`` prefix.

    Returns
    -------
    dict
        ``{"status": "success", "data": {...}}``, same projection as :func:`openalex_get_work`.

    Examples
    --------
    >>> work = openalex_get_work_by_doi(doi="10.1038/s41586-020-2649-2")
    >>> print(work["data"]["title"], work["data"]["year"], work["data"]["venue"])
    """
    identifier = _bare_doi(doi)
    if not identifier:
        return _missing("doi", "Pass a DOI, for example doi='10.1038/s41586-020-2649-2'.")
    # OpenAlex's documented DOI route puts a resolver URL inside the path. The DOI itself is
    # percent-encoded -- upstream encodes nothing here, so a DOI containing a '#' or '?' would break
    # the request. Verified live that the encoded form answers 200.
    payload, failure = _openalex_get(f"/works/https://doi.org/{_seg(identifier)}")
    if failure is not None:
        return failure
    return _ok(_work_summary(payload, author_cap=None))


def _author_summary(record):
    record = record or {}
    last = record.get("last_known_institution") or (record.get("last_known_institutions") or [{}])[0] or {}
    return {
        "openalex_id": _openalex_id(record.get("id")),
        "name": _clean(record.get("display_name")),
        "orcid": record.get("orcid"),
        "alternate_names": [_clean(name) for name in (record.get("display_name_alternatives") or [])][:10],
        "works_count": _as_int(record.get("works_count")),
        "cited_by_count": _as_int(record.get("cited_by_count")),
        "h_index": _as_int((record.get("summary_stats") or {}).get("h_index")),
        "institution": _clean(last.get("display_name")),
        "institution_country": last.get("country_code"),
        "topics": [_clean(topic.get("display_name")) for topic in (record.get("topics") or [])][:10],
    }


def _institution_summary(record):
    record = record or {}
    return {
        "openalex_id": _openalex_id(record.get("id")),
        "name": _clean(record.get("display_name")),
        "ror": record.get("ror"),
        "country_code": record.get("country_code"),
        "type": record.get("type"),
        "homepage": record.get("homepage_url"),
        "works_count": _as_int(record.get("works_count")),
        "cited_by_count": _as_int(record.get("cited_by_count")),
        "alternate_names": [_clean(name) for name in (record.get("display_name_alternatives") or [])][:10],
    }


def _source_summary(record):
    record = record or {}
    publisher = record.get("host_organization_name")
    stats = record.get("summary_stats") or {}
    return {
        "openalex_id": _openalex_id(record.get("id")),
        "name": _clean(record.get("display_name")),
        "issn_l": record.get("issn_l"),
        "issn": record.get("issn") or [],
        "type": record.get("type"),
        "publisher": _clean(publisher),
        "country_code": record.get("country_code"),
        "is_open_access": bool(record.get("is_oa")),
        "is_in_doaj": bool(record.get("is_in_doaj")),
        "works_count": _as_int(record.get("works_count")),
        "cited_by_count": _as_int(record.get("cited_by_count")),
        "h_index": _as_int(stats.get("h_index")),
        "two_year_mean_citedness": stats.get("2yr_mean_citedness"),
        "homepage": record.get("homepage_url"),
    }


def _openalex_search(entity, search, per_page, page, project, hint):
    if not search:
        return _missing("search", hint)
    payload, failure = _openalex_get(
        f"/{_seg(entity)}",
        {"search": search, "per-page": _page_size(per_page, default=5), "page": max(1, _as_int(page, 1))},
    )
    if failure is not None:
        return failure
    records = [project(record) for record in (payload or {}).get("results") or []]
    total = _as_int(((payload or {}).get("meta") or {}).get("count"))
    return _ok(records, num_found=total, returned=len(records), truncated=total > len(records))


def _openalex_entity(entity, identifier, project, name, hint):
    resolved = _openalex_id(identifier)
    if not resolved:
        return _missing(name, hint)
    payload, failure = _openalex_get(f"/{_seg(entity)}/{_seg(resolved)}")
    if failure is not None:
        return failure
    return _ok(project(payload))


def openalex_search_authors(search, per_page=5, page=1):
    """Find researchers by name in OpenAlex, with their affiliation, output and citation counts.

    Author names are ambiguous and this is how you disambiguate one: the hits carry alternate name
    spellings, the last known institution, works count, citation count and h-index, so the right
    "J. Smith" is usually obvious from the affiliation and topic mix.

    Parameters
    ----------
    search : str
        The researcher's name, e.g. ``"Aviv Regev"``.
    per_page : int, optional
        Hits per page, 1-100. Default 5.
    page : int, optional
        1-based page number. Default 1.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "num_found": int, "returned": int, "truncated": bool}``.

    Examples
    --------
    >>> people = openalex_search_authors(search="Aviv Regev")
    >>> for person in people["data"]:
    ...     print(person["name"], "|", person["institution"], "| h-index", person["h_index"])
    """
    return _openalex_search(
        "authors", search, per_page, page, _author_summary, "Pass a researcher's name, for example search='Aviv Regev'."
    )


def openalex_get_author(author_id):
    """Fetch one OpenAlex author record by identifier.

    Parameters
    ----------
    author_id : str
        ``"A5023888391"``, or the full ``"https://openalex.org/A5023888391"``.
        :func:`openalex_search_authors` returns one on every hit.

    Returns
    -------
    dict
        ``{"status": "success", "data": {...}}`` with name, ORCID, institution, works and citation
        counts, h-index and topics.

    Examples
    --------
    >>> person = openalex_get_author(author_id="A5023888391")
    >>> print(person["data"]["name"], person["data"]["institution"], person["data"]["works_count"])
    """
    return _openalex_entity(
        "authors",
        author_id,
        _author_summary,
        "author_id",
        "Pass an OpenAlex author id, for example author_id='A5023888391'.",
    )


def openalex_search_institutions(search, per_page=5, page=1):
    """Find universities, hospitals and research institutes by name in OpenAlex.

    Useful for resolving an affiliation string from a paper onto a ROR identifier and a country, and
    for building the ``institutions.id`` filter that :func:`openalex_search_works` accepts.

    Parameters
    ----------
    search : str
        The institution's name, e.g. ``"Broad Institute"``.
    per_page : int, optional
        Hits per page, 1-100. Default 5.
    page : int, optional
        1-based page number. Default 1.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "num_found": int, "returned": int, "truncated": bool}``,
        each record carrying OpenAlex id, ROR, country code, type, works and citation counts.

    Examples
    --------
    >>> places = openalex_search_institutions(search="Broad Institute")
    >>> for place in places["data"]:
    ...     print(place["name"], place["country_code"], place["ror"])
    """
    return _openalex_search(
        "institutions",
        search,
        per_page,
        page,
        _institution_summary,
        "Pass an institution name, for example search='Broad Institute'.",
    )


def openalex_get_institution(institution_id):
    """Fetch one OpenAlex institution record by identifier.

    Parameters
    ----------
    institution_id : str
        ``"I4210109156"``, or the full ``"https://openalex.org/I4210109156"``.

    Returns
    -------
    dict
        ``{"status": "success", "data": {...}}`` with name, ROR, country, type, homepage and counts.

    Examples
    --------
    >>> place = openalex_get_institution(institution_id="I4210109156")
    >>> print(place["data"]["name"], place["data"]["ror"], place["data"]["works_count"])
    """
    return _openalex_entity(
        "institutions",
        institution_id,
        _institution_summary,
        "institution_id",
        "Pass an OpenAlex institution id, for example institution_id='I4210109156'.",
    )


def openalex_search_sources(search, per_page=5, page=1):
    """Find journals, conference series and repositories by name in OpenAlex.

    Answers "is this a real journal, who publishes it, is it open access, and how much is it cited" --
    the venue-quality question, with h-index and two-year mean citedness rather than a licensed
    impact factor.

    Parameters
    ----------
    search : str
        The venue's name, e.g. ``"Nature Methods"``.
    per_page : int, optional
        Hits per page, 1-100. Default 5.
    page : int, optional
        1-based page number. Default 1.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "num_found": int, "returned": int, "truncated": bool}``,
        each record carrying ISSN-L, publisher, open-access and DOAJ status, h-index and counts.

    Examples
    --------
    >>> venues = openalex_search_sources(search="Nature Methods")
    >>> for venue in venues["data"]:
    ...     print(venue["name"], venue["issn_l"], "h-index", venue["h_index"], "OA", venue["is_open_access"])
    """
    return _openalex_search(
        "sources", search, per_page, page, _source_summary, "Pass a journal name, for example search='Nature Methods'."
    )


def openalex_get_source(source_id):
    """Fetch one OpenAlex source (journal, repository, conference) record by identifier.

    Parameters
    ----------
    source_id : str
        ``"S4210194219"``, or the full ``"https://openalex.org/S4210194219"``.

    Returns
    -------
    dict
        ``{"status": "success", "data": {...}}`` with ISSNs, publisher, open-access status and
        citation statistics.

    Examples
    --------
    >>> venue = openalex_get_source(source_id="S4210194219")
    >>> print(venue["data"]["name"], venue["data"]["publisher"], venue["data"]["is_in_doaj"])
    """
    return _openalex_entity(
        "sources",
        source_id,
        _source_summary,
        "source_id",
        "Pass an OpenAlex source id, for example source_id='S4210194219'.",
    )


# --------------------------------------------------------------------------------------------- #
# Crossref
# --------------------------------------------------------------------------------------------- #


def _crossref_get(path, params=None):
    """GET a Crossref path and unwrap its ``message`` envelope.

    Crossref wraps everything in ``{"status": "ok", "message-type": ..., "message": {...}}``; the
    envelope is noise once the status has been checked, so callers here see the message.
    """
    payload, failure = _fetch_json(f"{_CROSSREF}{path}", params=params or {})
    if failure is not None:
        return None, failure
    if not isinstance(payload, dict):
        return None, _error("Crossref returned a response that was not a JSON object.")
    if payload.get("status") not in (None, "ok"):
        return None, _error(f"Crossref reported status {payload.get('status')!r} for {path}.")
    return payload.get("message", payload), None


def _crossref_window(offset, rows):
    """Clamp a Crossref page to the service's own ``offset + rows <= 10000`` ceiling.

    Measured live 2026-09-17: ``offset=9999&rows=1`` succeeds, ``offset=10000`` fails, and
    ``offset=9990&rows=20`` also fails -- the constraint is on the sum, which is what Crossref's own
    error text says. (Upstream's constant treats it as ``offset <= 10000`` and its inline comment
    claims offset=9999 fails, which is the opposite of the observed behaviour.) Deeper paging needs
    the cursor API, which these tools do not expose.

    Returns ``(offset, rows, note_or_empty)``.
    """
    rows = _page_size(rows, default=20, maximum=_MAX_CROSSREF_ROWS)
    offset = max(0, _as_int(offset, 0))
    if offset + rows <= _CROSSREF_WINDOW:
        return offset, rows, ""
    if offset >= _CROSSREF_WINDOW:
        return (
            _CROSSREF_WINDOW - rows,
            rows,
            (
                f"offset {offset} is past Crossref's {_CROSSREF_WINDOW}-record paging ceiling; "
                f"it was clamped. Narrow the query with filter= instead of paging deeper."
            ),
        )
    rows = _CROSSREF_WINDOW - offset
    return (
        offset,
        rows,
        (f"offset + rows exceeded Crossref's {_CROSSREF_WINDOW}-record paging ceiling; rows was reduced to {rows}."),
    )


def _joined(value):
    """Crossref returns titles and container titles as lists; take the first non-empty one."""
    if isinstance(value, list):
        for item in value:
            if _clean(item):
                return _clean(item)
        return ""
    return _clean(value)


def _crossref_date(block):
    """``{"date-parts": [[2024, 3, 7]]}`` -> ``"2024-03-07"``; partial dates stay partial."""
    parts = ((block or {}).get("date-parts") or [[]])[0] or []
    return "-".join(f"{_as_int(part):02d}" if index else str(_as_int(part)) for index, part in enumerate(parts))


def _crossref_work(record, author_cap=50):
    """One Crossref work. ``author_cap=None`` keeps every author, for a single-record fetch.

    ``author_details`` and ``funder_details`` carry the ORCIDs, affiliations and award numbers the
    tool descriptions promise, and ``issued`` the issue date beside ``published``; the projection
    returned name strings and one date only (hunt 2026-09-30, uT6-literature-13). ``authors`` and
    ``funders`` keep their plain-string shape for callers already reading them.
    """
    record = record or {}
    authors = []
    author_details = []
    for author in record.get("author") or []:
        name = _clean(f"{author.get('given', '')} {author.get('family', '')}") or _clean(author.get("name"))
        if name:
            authors.append(name)
            author_details.append(
                {
                    "name": name,
                    "orcid": author.get("ORCID"),
                    "affiliations": [
                        _clean(entry.get("name")) for entry in (author.get("affiliation") or []) if entry.get("name")
                    ],
                }
            )
    funder_details = [
        {"name": _clean(funder.get("name")), "doi": funder.get("DOI"), "awards": list(funder.get("award") or [])}
        for funder in (record.get("funder") or [])
    ]
    licences = [entry.get("URL") for entry in (record.get("license") or []) if entry.get("URL")]
    return {
        "doi": _bare_doi(record.get("DOI")),
        "title": _joined(record.get("title")),
        "container": _joined(record.get("container-title")),
        "type": record.get("type"),
        "publisher": _clean(record.get("publisher")),
        "published": _crossref_date(record.get("published") or record.get("issued")),
        "issued": _crossref_date(record.get("issued")),
        "authors": authors[:author_cap],
        "author_details": author_details[:author_cap],
        "authors_total": len(authors),
        "authors_truncated": len(authors[:author_cap]) < len(authors),
        "issn": record.get("ISSN") or [],
        "isbn": record.get("ISBN") or [],
        "volume": record.get("volume"),
        "issue": record.get("issue"),
        "page": record.get("page"),
        "url": record.get("URL"),
        "cited_by_count": _as_int(record.get("is-referenced-by-count")),
        "references_count": _as_int(record.get("references-count")),
        "subjects": record.get("subject") or [],
        "licenses": licences[:5],
        "funders": [_clean(funder.get("name")) for funder in (record.get("funder") or [])][:20],
        "funder_details": funder_details[:20],
        "abstract": _strip_html(record.get("abstract")) if record.get("abstract") else "",
    }


def _crossref_list(message, project, note):
    items = [project(item) for item in (message or {}).get("items") or []]
    total = _as_int((message or {}).get("total-results"))
    result = _ok(items, num_found=total, returned=len(items), truncated=total > len(items))
    if note:
        result["paging_note"] = note
    return result


def crossref_search_works(query, limit=10, offset=None, filter=None):  # Crossref's own name
    """Search Crossref's DOI registry for works by free text, with optional filters.

    Crossref is the authority on what a DOI *is*. Search it to resolve a half-remembered reference,
    to check that a citation is real, or to find every work a funder or publisher registered. For
    discovery of biomedical literature to *read*, prefer :func:`europe_pmc_search_articles`.

    Parameters
    ----------
    query : str
        Free-text search across titles, authors and container titles.
    limit : int, optional
        Records to return, 1-1000. Default 10.
    offset : int, optional
        Records to skip. Crossref caps ``offset + limit`` at 10000; past that the page is clamped
        and ``paging_note`` says so.
    filter : str, optional
        Crossref filter syntax, comma-separated:
        ``"from-pub-date:2024-01-01,type:journal-article"``, ``"has-full-text:true"``,
        ``"funder:10.13039/100000002"``.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "num_found": int, "returned": int, "truncated": bool}``.

    Examples
    --------
    >>> works = crossref_search_works(query="spatial transcriptomics benchmark", limit=5)
    >>> for work in works["data"]:
    ...     print(work["published"], work["container"], "|", work["title"], "|", work["doi"])
    >>> print(crossref_search_works(query="Visium", filter="from-pub-date:2025-01-01")["num_found"])
    """
    if not query:
        return _missing("query", "Pass what to search for, for example query='spatial transcriptomics benchmark'.")
    offset, rows, note = _crossref_window(offset, limit)
    params = {"query": query, "rows": rows, "offset": offset}
    if filter:
        params["filter"] = filter
    message, failure = _crossref_get("/works", params)
    if failure is not None:
        return failure
    return _crossref_list(message, _crossref_work, note)


def crossref_get_work(doi):
    """Fetch the registered metadata for one DOI -- the authoritative record.

    This is how you verify a citation. A DOI that resolves here is real and the record is what the
    publisher deposited: title, container, date, authors, funders, licence.

    Parameters
    ----------
    doi : str
        ``"10.1038/s41586-020-2649-2"``, with or without a resolver prefix.

    Returns
    -------
    dict
        ``{"status": "success", "data": {...}}``. A DOI that is not registered anywhere comes back as
        the 404 error (``http_status: 404``, no ``unchecked``), which is the answer to "is this citation
        real". A DOI registered with another agency -- DataCite for Zenodo, figshare, Dryad and arXiv --
        is real but has no Crossref record; it comes back as an error that says so, names the agency in
        ``registration_agency`` and carries ``unchecked: true``. So does a DOI Crossref registered but
        does not index yet, and one whose agency could not be looked up (a rate limit or an outage):
        an error with ``unchecked: true`` is not evidence that the citation is wrong.

    Examples
    --------
    >>> work = crossref_get_work(doi="10.1038/s41586-020-2649-2")
    >>> print(work["data"]["title"], "|", work["data"]["container"], "|", work["data"]["published"])
    >>> print(work["data"]["authors"][:5])
    """
    identifier = _bare_doi(doi)
    if not identifier:
        return _missing("doi", "Pass a DOI, for example doi='10.1038/s41586-020-2649-2'.")
    # A DOI legitimately contains '/', and a few contain characters that would otherwise terminate
    # the path. Upstream's base class encodes here; its Europe PMC and OpenAlex classes do not.
    message, failure = _crossref_get(f"/works/{_seg(identifier)}")
    if failure is not None:
        if failure.get("http_status") == 404:
            # Crossref holds only its members' DOIs, so a real dataset DOI answers 404 here, and the
            # contract above told the model to read that as "this citation is not real" (skeptic note
            # on hunt 2026-09-30, uT6-literature-2). Crossref's agency route answers for every agency.
            agency_message, agency_failure = _crossref_get(f"/works/{_seg(identifier)}/agency")
            if agency_failure is not None and agency_failure.get("http_status") == 404:
                return failure  # no agency registered it: the one answer that refutes the citation
            agency = (agency_message or {}).get("agency") if isinstance(agency_message, dict) else None
            if agency_failure is not None or not isinstance(agency, dict) or not agency.get("id"):
                # The agency lookup itself got no answer. Returning the plain 404 here told the model
                # "not registered anywhere" about a DOI nobody checked, where CrossrefClient.fetch_doi
                # already says unchecked (hunt 2026-09-30, uT6-literature-2 review).
                reason = (agency_failure or {}).get("error") or "the answer named no agency"
                transient = agency_failure is not None and (
                    bool(agency_failure.get("retryable")) or agency_failure.get("http_status") is None
                )
                return _error(
                    f"Crossref has no record of {identifier}, and which agency registered it could not be "
                    f"looked up ({reason}). That is not evidence the citation is wrong; try again later or "
                    "resolve it through doi.org.",
                    unchecked=True,
                    retryable=transient,
                )
            label = agency.get("label") or agency.get("id")
            if agency["id"] == "crossref":
                # doi.org says Crossref registered it; Crossref's works index just does not hold it yet.
                return _error(
                    f"{identifier} is registered with Crossref, but Crossref's works index has no record "
                    "of it yet. That is not evidence the citation is wrong; resolve it through doi.org.",
                    registration_agency=label,
                    unchecked=True,
                )
            return _error(
                f"{identifier} is a registered DOI, but it is registered with {label}, not Crossref, so "
                "Crossref has no record of it. That is not evidence the citation is wrong; look it up "
                f"through {label} or doi.org.",
                registration_agency=label,
                unchecked=True,
            )
        return failure
    return _ok(_crossref_work(message, author_cap=None))


def crossref_get_journal(issn):
    """Fetch a journal's Crossref record and deposit statistics by ISSN.

    Answers "does this journal exist, who publishes it, how much have they registered, and how
    complete is their metadata". For citation-based venue quality use
    :func:`openalex_search_sources` instead.

    Parameters
    ----------
    issn : str
        ``"1548-7091"``. Either the print or electronic ISSN works.

    Returns
    -------
    dict
        ``{"status": "success", "data": {"title", "publisher", "issn", "issn_types", "counts",
        "coverage", "flags", "last_status_check"}}``.

    Examples
    --------
    >>> journal = crossref_get_journal(issn="1548-7091")
    >>> print(journal["data"]["title"], "|", journal["data"]["publisher"])
    >>> print(journal["data"]["counts"])
    """
    if not issn:
        return _missing("issn", "Pass a journal ISSN, for example issn='1548-7091'.")
    message, failure = _crossref_get(f"/journals/{_seg(str(issn).strip())}")
    if failure is not None:
        return failure
    message = message or {}
    return _ok(
        {
            "title": _clean(message.get("title")),
            "publisher": _clean(message.get("publisher")),
            "issn": message.get("ISSN") or [],
            "issn_types": message.get("issn-type") or [],
            "counts": message.get("counts") or {},
            "coverage": message.get("coverage") or {},
            "flags": message.get("flags") or {},
            "last_status_check": message.get("last-status-check-time"),
        }
    )


def _funder_summary(record):
    """Summarise one funder record.

    Two shapes reach this. ``/funders/{id}`` returns the full record; ``/funders?query=`` returns
    list entries that carry no counts and no descendants at all (measured 2026-09-17: the list item
    has only id, name, alt-names, location, uri, replaces, replaced-by, tokens). Reading an absent
    count through a zero-default would report ``work_count: 0`` for every funder in a search result,
    which reads as "this funder has registered no works" -- a wrong answer rather than a missing
    one. Absent stays ``None``; :func:`crossref_get_funder` is where the counts live.

    ``descendants`` comes from the flat field, not from ``hierarchy``. Crossref roots ``hierarchy``
    at the TOP of the funder tree rather than at the funder asked for: ``/funders/100000002`` (NIH)
    returns a hierarchy keyed by ``100000016`` (HHS, its parent), so looking a funder up inside its
    own hierarchy finds nothing and reports no descendants -- exactly for the deeply nested funders
    whose descendant list is worth having. The flat field is authoritative and shape-independent:
    59 entries for NIH. ``hierarchy-names`` names only 39 of those 59, so a missing name is normal.
    """
    record = record or {}
    work_count = record.get("work-count")
    descendant_work_count = record.get("descendant-work-count")
    descendants = sorted(str(item) for item in (record.get("descendants") or []))
    names = record.get("hierarchy-names") or {}
    return {
        "funder_id": record.get("id"),
        "name": _clean(record.get("name")),
        "alt_names": [_clean(name) for name in (record.get("alt-names") or [])][:20],
        "uri": record.get("uri"),
        "location": _clean(record.get("location")),
        "work_count": None if work_count is None else _as_int(work_count),
        "descendant_work_count": (None if descendant_work_count is None else _as_int(descendant_work_count)),
        "descendants": descendants,
        "descendant_names": {key: _clean(names[key]) for key in descendants if key in names},
        "replaces": record.get("replaces") or [],
        "replaced_by": record.get("replaced-by") or [],
    }


def _member_summary(record):
    record = record or {}
    return {
        "member_id": record.get("id"),
        "name": _clean(record.get("primary-name")),
        "location": _clean(record.get("location")),
        "prefixes": record.get("prefixes") or [],
        "counts": record.get("counts") or {},
        "coverage": record.get("coverage") or {},
        "names": [_clean(name) for name in (record.get("names") or [])][:10],
    }


def crossref_list_funders(query=None, limit=20, offset=None):
    """Search the Crossref Funder Registry for grant-giving bodies.

    Use it to turn a funder name from an acknowledgements section into a funder DOI, which then
    becomes a ``funder:`` filter on :func:`crossref_search_works` -- that is how you enumerate the
    output of a grant programme.

    Parameters
    ----------
    query : str, optional
        Funder name to search for, e.g. ``"National Institutes of Health"``. Omit to list the
        registry from the top.
    limit : int, optional
        Records to return, 1-1000. Default 20.
    offset : int, optional
        Records to skip; ``offset + limit`` is capped at 10000.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "num_found": int, "returned": int, "truncated": bool}``,
        each record carrying funder_id, name, alternate names, location and URI. The registry's
        search endpoint reports **no counts and no descendants** -- those fields come back ``None``
        and ``[]`` here, and :func:`crossref_get_funder` is where they are actually populated.

    Examples
    --------
    >>> funders = crossref_list_funders(query="National Institutes of Health", limit=5)
    >>> for funder in funders["data"]:
    ...     print(funder["funder_id"], funder["name"], funder["location"])
    """
    offset, rows, note = _crossref_window(offset, limit)
    params = {"rows": rows, "offset": offset}
    if query:
        params["query"] = query
    message, failure = _crossref_get("/funders", params)
    if failure is not None:
        return failure
    return _crossref_list(message, _funder_summary, note)


def crossref_get_funder(funder_id):
    """Fetch one funder's Crossref registry record, including every funder beneath it.

    This is the endpoint that carries the counts and the descendant list;
    :func:`crossref_list_funders` finds the id but reports neither.

    Parameters
    ----------
    funder_id : str
        The funder's Crossref id -- the DOI suffix, e.g. ``"100000002"`` for the NIH, or the full
        ``"10.13039/100000002"``. :func:`crossref_list_funders` returns one on every hit.

    Returns
    -------
    dict
        ``{"status": "success", "data": {...}}`` with name, alternate names, location, work counts,
        ``descendants`` (every funder id beneath this one) and ``descendant_names`` for the subset
        Crossref names. ``work_count`` counts this funder alone; ``descendant_work_count`` counts
        the whole subtree, which for a parent agency is the larger and usually the intended number.

    Examples
    --------
    >>> funder = crossref_get_funder(funder_id="100000002")
    >>> print(funder["data"]["name"], funder["data"]["work_count"], "registered works")
    >>> for child in funder["data"]["descendants"][:10]:
    ...     print(child, funder["data"]["descendant_names"].get(child, "(unnamed in registry)"))
    """
    if not funder_id:
        return _missing("funder_id", "Pass a Crossref funder id, for example funder_id='100000002' (NIH).")
    identifier = str(funder_id).strip()
    if identifier.startswith("10.13039/"):
        identifier = identifier[len("10.13039/") :]
    message, failure = _crossref_get(f"/funders/{_seg(identifier)}")
    if failure is not None:
        return failure
    return _ok(_funder_summary(message))


def crossref_list_types():
    """List the work types Crossref recognises -- the vocabulary its ``type:`` filter accepts.

    Small, fixed and worth checking before writing a ``filter="type:..."`` clause, because a
    misspelled type silently matches nothing rather than erroring.

    Returns
    -------
    dict
        ``{"status": "success", "data": [{"id": "journal-article", "label": "Journal Article"}, ...],
        "returned": int}``.

    Examples
    --------
    >>> types = crossref_list_types()
    >>> print([entry["id"] for entry in types["data"]])
    """
    message, failure = _crossref_get("/types")
    if failure is not None:
        return failure
    items = [{"id": item.get("id"), "label": _clean(item.get("label"))} for item in (message or {}).get("items") or []]
    return _ok(items, returned=len(items), num_found=_as_int((message or {}).get("total-results"), len(items)))


def crossref_search_members(query, limit=20, offset=None):
    """Search Crossref members -- the publishers that register DOIs.

    Answers "who is this publisher, what DOI prefixes do they own, and how complete is their
    deposited metadata". The ``coverage`` block is the practical part: it says what fraction of a
    publisher's records carry abstracts, licences, ORCIDs and funders, which tells you what a search
    restricted to them can actually return.

    Parameters
    ----------
    query : str
        Publisher name, e.g. ``"Springer Nature"``.
    limit : int, optional
        Records to return, 1-1000. Default 20.
    offset : int, optional
        Records to skip; ``offset + limit`` is capped at 10000.

    Returns
    -------
    dict
        ``{"status": "success", "data": [...], "num_found": int, "returned": int, "truncated": bool}``.

    Examples
    --------
    >>> members = crossref_search_members(query="Springer Nature", limit=5)
    >>> for member in members["data"]:
    ...     print(member["member_id"], member["name"], member["prefixes"][:3])
    """
    if not query:
        return _missing("query", "Pass a publisher name, for example query='Springer Nature'.")
    offset, rows, note = _crossref_window(offset, limit)
    message, failure = _crossref_get("/members", {"query": query, "rows": rows, "offset": offset})
    if failure is not None:
        return failure
    return _crossref_list(message, _member_summary, note)


def crossref_get_member(member_id):
    """Fetch one Crossref member (publisher) record by its member id.

    Parameters
    ----------
    member_id : str
        The numeric Crossref member id, e.g. ``"297"`` for Springer.
        :func:`crossref_search_members` returns one on every hit.

    Returns
    -------
    dict
        ``{"status": "success", "data": {...}}`` with the publisher's name, location, DOI prefixes,
        deposit counts and metadata-coverage breakdown.

    Examples
    --------
    >>> member = crossref_get_member(member_id="297")
    >>> print(member["data"]["name"], member["data"]["counts"])
    """
    if not member_id:
        return _missing("member_id", "Pass a Crossref member id, for example member_id='297'.")
    message, failure = _crossref_get(f"/members/{_seg(str(member_id).strip())}")
    if failure is not None:
        return failure
    return _ok(_member_summary(message))
