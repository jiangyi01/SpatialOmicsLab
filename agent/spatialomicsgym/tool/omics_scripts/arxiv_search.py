"""arXiv search via the official API with advanced filtering.

Adapted from omics-skills (https://github.com/fmschulz/omics-skills).
"""

from __future__ import annotations

import datetime as dt
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
USER_AGENT = "spatialomicsgym-arxiv-search/1.0"
RAW_QUERY_HINTS = (
    "ti:",
    "au:",
    "abs:",
    "co:",
    "jr:",
    "cat:",
    "rn:",
    "all:",
    "submittedDate:",
    "AND",
    "OR",
    "ANDNOT",
    "(",
    ")",
    "[",
    "]",
)


def compact_whitespace(text: str) -> str:
    return " ".join(text.split())


#: Raw arXiv syntax, matched as syntax: a field prefix at a word start, a boolean operator standing
#: alone between spaces, or a bracket. ``RAW_QUERY_HINTS`` was tested as substrings, so "mTOR" and
#: "CORTEX" contained "OR" and "pattern:" contained "rn:", and those queries were sent raw with the
#: phrase flag silently ignored (hunt 2026-09-30, uT6-literature-18).
_RAW_QUERY_RE = re.compile(
    r"\b(?:ti|au|abs|co|jr|cat|rn|all|submittedDate):|(?:^|\s)(?:AND|OR|ANDNOT)(?:\s|$)|[()\[\]]"
)


def is_raw_query(query: str) -> bool:
    return _RAW_QUERY_RE.search(query) is not None


def quote_term(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace('"', '\\"')
    if re.search(r"\s", escaped):
        return f'"{escaped}"'
    return escaped


def compile_plain_query(query: str, phrase: bool) -> str:
    text = compact_whitespace(query)
    if not text:
        raise ValueError("query must not be empty")
    if phrase:
        return f"all:{quote_term(text)}"
    return " AND ".join(f"all:{quote_term(token)}" for token in text.split())


def parse_arxiv_timestamp(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.UTC)
    except ValueError:
        return None


def apply_local_days_filter(results: list[dict], days: int | None) -> tuple[list[dict], dict | None]:
    if days is None:
        return results, None
    if days <= 0:
        raise ValueError("days must be a positive integer")
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=days)
    filtered = []
    excluded = 0
    undated = 0
    for result in results:
        published = parse_arxiv_timestamp(result.get("published"))
        if published is None:
            undated += 1
            continue
        if published >= cutoff:
            filtered.append(result)
        else:
            excluded += 1
    metadata = {
        "mode": "local_published_date",
        "days": days,
        "cutoff_utc": cutoff.isoformat(timespec="seconds"),
        "pre_filter_result_count": len(results),
        "post_filter_result_count": len(filtered),
        "excluded_older_count": excluded,
        "excluded_undated_count": undated,
    }
    return filtered, metadata


def compile_search_query(query: str, phrase: bool = False, category: str | None = None) -> str:
    compiled = query if is_raw_query(query) else compile_plain_query(query, phrase)
    if category:
        compiled = f"{compiled} AND cat:{category}"
    return compiled


def fetch_feed(params: dict[str, str], timeout: int) -> tuple[str, str]:
    query_string = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    request_url = f"{API_URL}?{query_string}"
    request = urllib.request.Request(request_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return request_url, response.read().decode(charset)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from arXiv API: {body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        # HTTPError is a URLError subclass (caught above); this arm handles DNS/connection/read-timeout
        # failures that otherwise propagated as a raw URLError.
        raise RuntimeError(f"Network error from arXiv API: {getattr(exc, 'reason', exc)}") from exc


def text_or_none(parent: ET.Element, path: str) -> str | None:
    node = parent.find(path, ATOM_NS)
    if node is None or node.text is None:
        return None
    value = compact_whitespace(node.text)
    return value or None


def parse_result(entry: ET.Element) -> dict:
    categories = [
        category.attrib["term"] for category in entry.findall("atom:category", ATOM_NS) if category.attrib.get("term")
    ]
    pdf_url = None
    abs_url = text_or_none(entry, "atom:id")
    for link in entry.findall("atom:link", ATOM_NS):
        href = link.attrib.get("href")
        title = link.attrib.get("title")
        rel = link.attrib.get("rel")
        if title == "pdf" and href:
            pdf_url = href
        elif rel == "alternate" and href:
            abs_url = href
    return {
        "title": text_or_none(entry, "atom:title"),
        "summary": text_or_none(entry, "atom:summary"),
        "authors": [
            compact_whitespace(node.text) for node in entry.findall("atom:author/atom:name", ATOM_NS) if node.text
        ],
        "arxiv_id": (abs_url or "").rstrip("/").split("/")[-1] or None,
        "abs_url": abs_url,
        "pdf_url": pdf_url,
        "published": text_or_none(entry, "atom:published"),
        "updated": text_or_none(entry, "atom:updated"),
        "primary_category": (
            entry.find("arxiv:primary_category", ATOM_NS).attrib.get("term")
            if entry.find("arxiv:primary_category", ATOM_NS) is not None
            else None
        ),
        "categories": categories,
        "comment": text_or_none(entry, "arxiv:comment"),
        "journal_ref": text_or_none(entry, "arxiv:journal_ref"),
        "doi": text_or_none(entry, "arxiv:doi"),
    }


def parse_feed(xml_text: str) -> dict:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse arXiv Atom response: {exc}") from exc
    entries = [parse_result(entry) for entry in root.findall("atom:entry", ATOM_NS)]
    total_results = root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    start_index = root.findtext("{http://a9.com/-/spec/opensearch/1.1/}startIndex")
    items_per_page = root.findtext("{http://a9.com/-/spec/opensearch/1.1/}itemsPerPage")
    return {
        "feed_updated": text_or_none(root, "atom:updated"),
        "total_results": int(total_results) if total_results else None,
        "start_index": int(start_index) if start_index else None,
        "items_per_page": int(items_per_page) if items_per_page else None,
        "results": entries,
    }


def search(
    query: str,
    max_results: int = 10,
    phrase: bool = False,
    category: str | None = None,
    days: int | None = None,
    sort: str = "relevance",
    order: str = "descending",
    start: int = 0,
    timeout: int = 20,
) -> dict:
    """Search arXiv and return structured results."""
    compiled_query = compile_search_query(query, phrase=phrase, category=category)
    # When filtering to recent papers, sort the FETCH by submission date. With the default relevance
    # sort, the API returns the top-relevance papers and the local `days` filter keeps only the few that
    # are also recent (often 0-1), missing the actual recent submissions.
    effective_sort = "submittedDate" if (days is not None and sort == "relevance") else sort
    params = {
        "search_query": compiled_query,
        "start": str(start),
        "max_results": str(min(max(max_results, 1), 2000)),
        "sortBy": effective_sort,
        "sortOrder": order,
    }
    request_url, xml_text = fetch_feed(params, timeout)
    parsed = parse_feed(xml_text)
    filtered_results, days_filter = apply_local_days_filter(parsed["results"], days)
    return {
        "success": True,
        "compiled_query": compiled_query,
        "request_url": request_url,
        "total_results": parsed["total_results"],
        "days_filter": days_filter,
        "result_count": len(filtered_results),
        "results": filtered_results,
    }


def fetch_by_ids(ids: list[str], timeout: int = 20) -> dict:
    """Fetch arXiv papers by their IDs."""
    unique_ids = list(dict.fromkeys(compact_whitespace(i.strip().strip(",")) for i in ids if i.strip()))
    if not unique_ids:
        raise ValueError("at least one arXiv ID is required")
    params = {
        "id_list": ",".join(unique_ids),
        "start": "0",
        "max_results": str(len(unique_ids)),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    request_url, xml_text = fetch_feed(params, timeout)
    parsed = parse_feed(xml_text)
    return {
        "success": True,
        "request_url": request_url,
        "result_count": len(parsed["results"]),
        "results": parsed["results"],
    }
