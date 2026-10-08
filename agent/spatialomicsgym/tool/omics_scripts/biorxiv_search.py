"""bioRxiv search via the official API with local filtering and deduplication.

Adapted from omics-skills (https://github.com/fmschulz/omics-skills).
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shlex
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://api.biorxiv.org/details/biorxiv"
USER_AGENT = "spatialomicsgym-biorxiv-search/1.0"
VALID_FIELDS = ("title", "abstract", "authors")
PAGE_SIZE = 100


def compact_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_term(text: str) -> str:
    return compact_whitespace(text).strip().lower()


def parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def to_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def parse_authors(raw: str | None) -> list[str]:
    if not raw:
        return []
    if ";" in raw:
        parts = raw.split(";")
    else:
        parts = [raw]
    return [compact_whitespace(part) for part in parts if compact_whitespace(part)]


def expand_author_variants(author_filters: list[str]) -> list[str]:
    variants: set[str] = set()
    for author in author_filters:
        text = compact_whitespace(author)
        if not text:
            continue
        variants.add(normalize_term(text))
        tokens = [token.rstrip(".") for token in text.split()]
        if len(tokens) < 2:
            continue
        first = tokens[0]
        last = tokens[-1]
        variants.add(normalize_term(f"{first} {last}"))
        variants.add(normalize_term(f"{first[0]}. {last}"))
        if len(tokens) >= 3:
            middle = tokens[1]
            variants.add(normalize_term(f"{first} {middle[0]}. {last}"))
            variants.add(normalize_term(f"{first[0]}. {middle[0]}. {last}"))
    return sorted(variants)


def fetch_json(url: str, timeout: int) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(charset))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from bioRxiv API: {body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        # DNS/connection/read-timeout failures otherwise propagated as a raw URLError to the caller.
        raise RuntimeError(f"Network error from bioRxiv API: {getattr(exc, 'reason', exc)}") from exc


def normalize_record(raw: dict) -> dict:
    doi = compact_whitespace(str(raw.get("doi", "") or ""))
    title = compact_whitespace(str(raw.get("title", "") or ""))
    abstract = compact_whitespace(str(raw.get("abstract", "") or ""))
    authors_raw = compact_whitespace(str(raw.get("authors", "") or ""))
    version = to_int(raw.get("version"))
    biorxiv_url = None
    if doi:
        if version is not None:
            biorxiv_url = f"https://www.biorxiv.org/content/{doi}v{version}"
        else:
            biorxiv_url = f"https://www.biorxiv.org/search/{urllib.parse.quote(doi)}"
    return {
        "doi": doi or None,
        "title": title or None,
        "authors": parse_authors(authors_raw),
        "authors_text": authors_raw or None,
        "author_corresponding": compact_whitespace(str(raw.get("author_corresponding", "") or "")) or None,
        "date": compact_whitespace(str(raw.get("date", "") or "")) or None,
        "version": version,
        "category": compact_whitespace(str(raw.get("category", "") or "")) or None,
        "abstract": abstract or None,
        "published": compact_whitespace(str(raw.get("published", "") or "")) or None,
        "doi_url": f"https://doi.org/{doi}" if doi else None,
        "biorxiv_url": biorxiv_url,
    }


def parse_query_groups(query: str | None, phrase: bool) -> list[list[str]]:
    if not query:
        return []
    text = compact_whitespace(query)
    if not text:
        return []
    if phrase:
        return [[normalize_term(text)]]
    groups = []
    for chunk in re.split(r"\s+OR\s+", text, flags=re.IGNORECASE):
        piece = compact_whitespace(chunk)
        if not piece:
            continue
        try:
            tokens = shlex.split(piece)
        except ValueError:
            tokens = piece.split()
        terms = [normalize_term(token) for token in tokens if normalize_term(token)]
        if terms:
            groups.append(terms)
    return groups


def matches_groups(text: str, groups: list[list[str]]) -> bool:
    if not groups:
        return True
    haystack = normalize_term(text)
    if not haystack:
        return False
    return any(all(term in haystack for term in group) for group in groups)


def matches_author_filters(authors_text: str, author_variants: list[str]) -> bool:
    if not author_variants:
        return True
    haystack = normalize_term(authors_text)
    if not haystack:
        return False
    return any(variant in haystack for variant in author_variants)


def record_matches(record: dict, fields: list[str], query_groups: list[list[str]], author_variants: list[str]) -> bool:
    texts = []
    if "title" in fields:
        texts.append(str(record.get("title") or ""))
    if "abstract" in fields:
        texts.append(str(record.get("abstract") or ""))
    if "authors" in fields:
        texts.append(str(record.get("authors_text") or ""))
    combined = " ".join(texts)
    if not matches_groups(combined, query_groups):
        return False
    if not matches_author_filters(str(record.get("authors_text") or ""), author_variants):
        return False
    return True


def dedupe_latest(records: list[dict]) -> tuple[list[dict], int]:
    by_doi: dict[str, dict] = {}
    dropped = 0
    for record in records:
        doi = str(record.get("doi") or "")
        if not doi:
            key = f"__no_doi__::{record.get('title')}::{record.get('date')}"
            by_doi[key] = record
            continue
        existing = by_doi.get(doi)
        if existing is None:
            by_doi[doi] = record
            continue
        existing_version = to_int(existing.get("version")) or -1
        current_version = to_int(record.get("version")) or -1
        if current_version > existing_version:
            by_doi[doi] = record
        dropped += 1
    return list(by_doi.values()), dropped


def search(
    query: str | None = None,
    max_results: int = 10,
    phrase: bool = False,
    days: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
    authors: list[str] | None = None,
    doi: str | None = None,
    fields: str = "title,abstract,authors",
    scan_limit: int = 300,
    all_versions: bool = False,
    timeout: int = 30,
) -> dict:
    """Search bioRxiv with local filtering and deduplication."""
    # Parse fields
    field_list = [f.strip() for f in fields.split(",") if f.strip() in VALID_FIELDS]
    if not field_list:
        field_list = list(VALID_FIELDS)

    query_groups = parse_query_groups(query, phrase)
    author_variants = expand_author_variants(authors or [])

    # Build interval
    if doi:
        interval = ""
    elif start_date and end_date:
        interval = f"{start_date}/{end_date}"
    else:
        d = days or 30
        today = dt.datetime.now(dt.UTC).date()
        start = today - dt.timedelta(days=d)
        interval = f"{start.isoformat()}/{today.isoformat()}"

    matched: list[dict] = []
    records_scanned = 0
    # What the interval holds, as the API states it in messages[0].total. Without it a search that
    # stopped at scan_limit -- 300 records of a month that holds thousands -- returned result_count 0
    # reading as "no preprints on this in 30 days" (hunt 2026-09-30, uT6-literature-8).
    interval_total: int | None = None

    if doi:
        encoded_doi = urllib.parse.quote(doi, safe="/")
        url = f"{API_BASE}/{encoded_doi}/na/json"
        data = fetch_json(url, timeout)
        collection = data.get("collection", [])
        if isinstance(collection, list):
            for raw in collection:
                record = normalize_record(raw)
                if record_matches(record, field_list, query_groups, author_variants):
                    matched.append(record)
            records_scanned = len(collection)
    else:
        cursor = 0
        while records_scanned < scan_limit:
            # NOTE: the bioRxiv /details endpoint ignores a ?category= param, so the category filter is
            # applied CLIENT-SIDE below (previously it was tacked onto the URL and silently did nothing).
            url = f"{API_BASE}/{interval}/{cursor}/json"
            data = fetch_json(url, timeout)
            if interval_total is None:
                messages = data.get("messages")
                if isinstance(messages, list) and messages and isinstance(messages[0], dict):
                    interval_total = to_int(messages[0].get("total"))
            collection = data.get("collection", [])
            if not isinstance(collection, list) or not collection:
                break
            for raw in collection:
                records_scanned += 1
                record = normalize_record(raw)
                cat_ok = not category or normalize_term(str(record.get("category") or "")) == normalize_term(category)
                if cat_ok and record_matches(record, field_list, query_groups, author_variants):
                    matched.append(record)
                if records_scanned >= scan_limit:
                    break
            if len(collection) < PAGE_SIZE:
                break
            cursor += len(collection)

    deduped = matched
    versions_collapsed = 0
    if not all_versions:
        deduped, versions_collapsed = dedupe_latest(matched)

    # Sort by date descending
    def sort_key(r):
        d = parse_date(str(r.get("date") or "")) or dt.date.min
        v = to_int(r.get("version")) or -1
        return (d, v)

    ordered = sorted(deduped, key=sort_key, reverse=True)
    results = ordered[:max_results]

    if doi:
        interval_total = records_scanned
    truncated = interval_total is None or records_scanned < interval_total
    result = {
        "success": True,
        "records_scanned": records_scanned,
        "interval_total": interval_total,
        "truncated": truncated,
        "matched_before_dedup": len(matched),
        "matched_after_dedup": len(deduped),
        "versions_collapsed": versions_collapsed,
        "result_count": len(results),
        "results": results,
    }
    if truncated and not doi:
        total = "an unknown number of" if interval_total is None else f"{interval_total}"
        result["note"] = (
            f"Only {records_scanned} of {total} records in {interval} were scanned (scan_limit={scan_limit}), "
            "so a missing preprint may simply not have been reached. Raise scan_limit or narrow the dates."
        )
    return result
