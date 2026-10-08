"""Literature-search tools: DOI resolution, arXiv/PubMed/Scholar queries, web and PDF extraction.

Most of these tools need a third-party package that the core install does not ship -- `arxiv`,
`scholarly`, `googlesearch` (web search), `bs4` (HTML parsing) and `PyPDF2` (PDF text). They are
imported inside the functions that use them rather than at module scope, because a module-scope
import makes *one* absent package remove *all eight* tools: `PyPDF2` is reached by
`extract_pdf_content` alone, yet importing it up here also took out `query_pubmed` (which needs
`pymed`), `query_scholar` and `advanced_web_search_claude`, none of which touch it. `pip install
-e .` declares only pydantic/langchain/dotenv/pyyaml, so that was the state of every core install.

Each loader raises a message naming the missing distribution, since the agent's next move is to
install it and `NameError: name 'PyPDF2' is not defined` does not say what to install.

That deferral has a cost, and :func:`requirement_for` is the payment. `hasattr(literature,
"query_arxiv")` is `True` on an install with no `arxiv` package, so any caller deciding
reachability that way decides it wrong -- and one did, writing recovery steps that named functions
guaranteed to raise `ModuleNotFoundError`. Ask this module instead; it is the only place that knows
which import each function defers.
"""

import contextlib
import os
import re
import threading
import time
from io import BytesIO
from urllib.parse import unquote, urldefrag, urljoin, urlparse

import requests

# Import name -> the distribution you install. They differ for two of the three, which is exactly
# the detail a caller staring at a NameError does not have.
_OPTIONAL_DISTRIBUTIONS = {
    "PyPDF2": "PyPDF2",
    "anthropic": "anthropic",
    "arxiv": "arxiv",
    "bs4": "beautifulsoup4",
    "googlesearch": "googlesearch-python",
    "pymed": "pymed",
    "scholarly": "scholarly",
}

#: Public function -> the import it defers into its body. A function absent from this table needs
#: nothing beyond the core install. Every public function here is in the table: PubMed is reached
#: through `pymed`, not through the bare `requests` the module imports at the top. Kept
#: beside the table above rather than derived from the source, because a grep for `import` inside a
#: function body is exactly the kind of cleverness that stops being true after one refactor.
_FUNCTION_IMPORTS = {
    "advanced_web_search_claude": "anthropic",
    "extract_pdf_content": "PyPDF2",
    "extract_url_content": "bs4",
    "fetch_supplementary_info_from_doi": "bs4",
    "query_arxiv": "arxiv",
    "query_pubmed": "pymed",
    "query_scholar": "scholarly",
    "search_google": "googlesearch",
}


def requirement_for(name: str) -> str | None:
    """The distribution `name` needs and this environment lacks, or ``None`` if it is callable here.

    `name` may be bare (``"query_arxiv"``) or dotted (``"literature.query_arxiv"``,
    ``"spatialomicsgym.tool.literature.query_arxiv"``); only the last segment is read, so a caller
    can pass whatever spelling it already holds. A name this module does not define also answers
    ``None`` -- the question asked is "what is missing", and "that is not one of mine" is a
    different question with a different right answer.

    One function needs more than a package. ``advanced_web_search_claude`` raises on any agent model
    that is not a Claude model, or with no credential for the endpoint serving it; with
    ``anthropic`` installed this answered ``None`` regardless, so the format-probe recovery route
    named a function that always raised on the shipped (Azure GPT) default (hunt 2026-09-30,
    uT6-literature-5). For it, an installed package with an unmet precondition answers with a short
    phrase naming the precondition -- not a distribution, so not something to ``pip install``. A
    caller that writes ``pip install <answer>`` asks :func:`distribution_for` instead.
    """
    function = str(name).rsplit(".", 1)[-1]
    distribution = distribution_for(function)
    if distribution is not None:
        return distribution
    if function == "advanced_web_search_claude":
        _, _, missing = _claude_search_route()
        return missing
    return None


def distribution_for(name: str) -> str | None:
    """The distribution `name` needs and this environment lacks -- something to ``pip install`` -- or ``None``.

    :func:`requirement_for` answers "can it be called here", and for ``advanced_web_search_claude``
    with ``anthropic`` present its non-``None`` answer is a precondition ("a Claude model as the
    agent's LLM ..."), not a package. Rendered as ``pip install {answer}`` that read "pip install a
    Claude model" (hunt 2026-09-30, uT6-literature-5 review). This one only ever names an entry of
    ``_OPTIONAL_DISTRIBUTIONS``; ``None`` with a non-``None`` ``requirement_for`` means no install fixes it.
    """
    import importlib.util

    function = str(name).rsplit(".", 1)[-1]
    module = _FUNCTION_IMPORTS.get(function)
    if module is None:
        return None
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        found = False
    if not found:
        return _OPTIONAL_DISTRIBUTIONS.get(module, module)
    return None


def _require(module_name: str):
    """Import an optional dependency, or fail with the pip name for it."""
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        distribution = _OPTIONAL_DISTRIBUTIONS.get(module_name, module_name)
        raise ImportError(
            f"This tool needs the optional package '{module_name}', which is not installed in this "
            f"environment. Install it with: pip install {distribution}"
        ) from exc


def _beautiful_soup():
    """The `BeautifulSoup` class. See :func:`_require`."""
    return _require("bs4").BeautifulSoup


def _google_search():
    """The `googlesearch.search` callable. See :func:`_require`."""
    return _require("googlesearch").search


def _env_number(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


#: The fetches below reach whatever host a DOI or a user names, so they go through `requests` rather
#: than the allowlisted HTTP layer -- but never without a deadline or a size bound. They had neither:
#: a stalled publisher blocked the step forever and a multi-gigabyte "supplementary" file was
#: buffered whole in memory (hunt 2026-09-30, uT6-literature-26).
_TIMEOUT_ENV = "SOG_LITERATURE_FETCH_TIMEOUT"
_PAGE_LIMIT_ENV = "SOG_LITERATURE_MAX_PAGE_MB"
_DOWNLOAD_LIMIT_ENV = "SOG_LITERATURE_MAX_DOWNLOAD_MB"
#: Seconds one whole fetch may take. ``timeout=`` bounds each socket read, so a server sending a byte
#: every few seconds kept a step alive without limit (hunt 2026-09-30, uT6-literature-26 review).
_DEADLINE_ENV = "SOG_LITERATURE_FETCH_DEADLINE"


def _fetch_timeout() -> float:
    return _env_number(_TIMEOUT_ENV, 30.0)


def _fetch_deadline() -> float:
    return _env_number(_DEADLINE_ENV, 600.0)


def _body_pieces(response: requests.Response, url: str, deadline: float, chunk_size: int):
    """The decoded body of a streamed ``response``, piece by piece, ending at ``deadline`` (monotonic).

    ``iter_content`` blocks until a whole chunk has arrived, however slowly, so the deadline is checked
    between single socket reads instead: urllib3's ``read1`` makes at most one. A response without
    that raw stream (a stand-in) is read by ``iter_content``, with the deadline checked per chunk.
    """
    import urllib3.exceptions

    raw = getattr(response, "raw", None)
    read1 = getattr(raw, "read1", None)
    pieces = None if callable(read1) else response.iter_content(chunk_size)
    while True:
        if time.monotonic() > deadline:
            raise requests.Timeout(f"{url} was still sending after {_fetch_deadline():.0f}s (raise {_DEADLINE_ENV})")
        try:
            chunk = read1(chunk_size, decode_content=True) if pieces is None else next(pieces, b"")
        except urllib3.exceptions.ReadTimeoutError as exc:
            raise requests.Timeout(f"{url} stalled for {_fetch_timeout():.0f}s (raise {_TIMEOUT_ENV})") from exc
        except urllib3.exceptions.HTTPError as exc:  # a reset, a short body, a bad encoding
            raise requests.ConnectionError(f"{url}: {exc}") from exc
        if not chunk:
            return
        yield chunk


def _limit_bytes(env: str, default_mb: float) -> int:
    return int(_env_number(env, default_mb) * 1024 * 1024)


def _bounded_get(url: str, *, headers: dict | None = None) -> requests.Response:
    """``requests.get`` with a timeout and the body read under a size cap, into memory.

    Raises ``requests.Timeout`` naming the knob, or ``ValueError`` when the body is over the cap.
    """
    limit = _limit_bytes(_PAGE_LIMIT_ENV, 64)
    deadline = time.monotonic() + _fetch_deadline()
    try:
        response = requests.get(url, headers=headers, timeout=_fetch_timeout(), stream=True)
    except requests.Timeout as exc:
        raise requests.Timeout(f"{url} did not answer within {_fetch_timeout():.0f}s (raise {_TIMEOUT_ENV})") from exc
    chunks, size = [], 0
    with response:
        for chunk in _body_pieces(response, url, deadline, 65536):
            size += len(chunk)
            if size > limit:
                raise ValueError(
                    f"{url} returned more than {limit // 2**20} MiB, the most this tool reads into memory "
                    f"(raise {_PAGE_LIMIT_ENV})"
                )
            chunks.append(chunk)
    response._content = b"".join(chunks)
    return response


def _download(url: str, destination: str, *, headers: dict | None = None) -> int:
    """Stream ``url`` to ``destination`` under a size cap, atomically. Returns the bytes written."""
    limit = _limit_bytes(_DOWNLOAD_LIMIT_ENV, 1024)
    partial = f"{destination}.partial"
    written = 0
    deadline = time.monotonic() + _fetch_deadline()
    try:
        try:
            response = requests.get(url, headers=headers, timeout=_fetch_timeout(), stream=True)
        except requests.Timeout as exc:
            raise requests.Timeout(
                f"{url} did not answer within {_fetch_timeout():.0f}s (raise {_TIMEOUT_ENV})"
            ) from exc
        with response:
            response.raise_for_status()
            with open(partial, "wb") as handle:
                # _body_pieces names its own knob: the per-read timeout or the whole-fetch deadline.
                for chunk in _body_pieces(response, url, deadline, 1 << 20):
                    written += len(chunk)
                    if written > limit:
                        raise ValueError(
                            f"{url} is larger than {limit // 2**20} MiB, the download cap (raise {_DOWNLOAD_LIMIT_ENV})"
                        )
                    handle.write(chunk)
        os.replace(partial, destination)
    finally:
        if os.path.exists(partial):
            os.unlink(partial)
    return written


def _safe_file_name(link: str, taken: set[str], index: int) -> str:
    """A file name for ``link`` that is plain, non-empty and not already used in this download."""
    base = os.path.basename(unquote(urlparse(link).path).rstrip("/"))
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or f"supplementary_{index}"
    stem, ext = os.path.splitext(base)
    candidate, n = base, 1
    while candidate in taken:
        candidate = f"{stem}_{n}{ext}"
        n += 1
    taken.add(candidate)
    return candidate


def fetch_supplementary_info_from_doi(doi: str, output_dir: str = "supplementary_info"):
    """Fetches supplementary information for a paper given its DOI and returns a research log.

    Args:
        doi: The paper DOI.
        output_dir: Directory to save supplementary files.

    Returns:
        dict: A dictionary containing a research log and the downloaded file paths.

    """
    # Always ``{"log": [...], "files": [...]}``. The no-links path returned a bare list and success a
    # joined string, so ``result["files"]`` failed exactly when it mattered (hunt 2026-09-30,
    # uT6-literature-23).
    research_log = []
    research_log.append(f"Starting process for DOI: {doi}")

    # CrossRef API to resolve DOI to a publisher page
    crossref_url = f"https://doi.org/{doi}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = _bounded_get(crossref_url, headers=headers)
    except (requests.RequestException, ValueError) as exc:
        research_log.append(f"Failed to resolve DOI: {doi}. {exc}")
        return {"log": research_log, "files": []}

    if response.status_code != 200:
        log_message = f"Failed to resolve DOI: {doi}. Status Code: {response.status_code}"
        research_log.append(log_message)
        return {"log": research_log, "files": []}

    publisher_url = response.url
    research_log.append(f"Resolved DOI to publisher page: {publisher_url}")

    # The resolver already followed the redirects to the publisher page, so that response is the page.
    # Parse page content
    soup = _beautiful_soup()(response.content, "html.parser")
    supplementary_links = []
    page_itself = urldefrag(publisher_url)[0]

    # Look for supplementary materials by keywords or links
    for link in soup.find_all("a", href=True):
        href = link.get("href")
        text = link.get_text().lower()
        if "supplementary" in text or "supplemental" in text or "appendix" in text:
            full_url = urljoin(publisher_url, href)
            # Only a separate http(s) resource is a file. "#Sec25" is the article's own
            # "Supplementary information" heading -- it was downloaded as the article and reported as
            # a supplementary file -- and a javascript:/mailto: href crashed the tool.
            if urlparse(full_url).scheme not in ("http", "https") or urldefrag(full_url)[0] == page_itself:
                continue
            if full_url in supplementary_links:
                continue
            supplementary_links.append(full_url)
            research_log.append(f"Found supplementary material link: {full_url}")

    if not supplementary_links:
        log_message = f"No supplementary materials found for DOI {doi}."
        research_log.append(log_message)
        return {"log": research_log, "files": []}

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    research_log.append(f"Created output directory: {output_dir}")

    # Download supplementary materials
    downloaded_files = []
    taken = set(os.listdir(output_dir))
    for index, link in enumerate(supplementary_links, 1):
        # Unique per download: two links sharing a basename overwrote each other, and a link ending
        # in "/" named the directory itself.
        file_name = os.path.join(output_dir, _safe_file_name(link, taken, index))
        try:
            _download(link, file_name, headers=headers)
        except (requests.RequestException, OSError, ValueError) as exc:
            research_log.append(f"Failed to download file from {link}: {exc}")
            continue
        downloaded_files.append(file_name)
        research_log.append(f"Downloaded file: {file_name}")

    if downloaded_files:
        research_log.append(f"Successfully downloaded {len(downloaded_files)} file(s).")
    else:
        research_log.append(f"No files could be downloaded for DOI {doi}.")

    return {"log": research_log, "files": downloaded_files}


def query_arxiv(query: str, max_papers: int = 10) -> str:
    """Query arXiv for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.
    - max_papers (int): The maximum number of papers to retrieve (default: 10).

    Returns
    -------
    - str: The formatted search results or an error message.

    """
    import arxiv

    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance)
        results = "\n\n".join([f"Title: {paper.title}\nSummary: {paper.summary}" for paper in client.results(search)])
        return results if results else "No papers found on arXiv."
    except Exception as e:
        return f"Error querying arXiv: {e}"


def query_scholar(query: str) -> str:
    """Query Google Scholar for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.

    Returns
    -------
    - str: The first search result formatted or an error message.

    """
    from scholarly import ProxyGenerator, scholarly

    try:
        # Set up a ProxyGenerator object to use free proxies
        # This needs to be done only once per session
        # Inside the `try`, and on a clock. scholarly probes up to 200 scraped proxies with no overall
        # bound and raises MaxTriesExceededException when none works; outside the `try` that escaped
        # as a traceback instead of the error string this function promises (hunt 2026-09-30,
        # uT6-literature-9 -- the proxy route itself stays, by decision).
        pg = _set_up_free_proxies(ProxyGenerator())
        scholarly.use_proxy(pg)
        search_query = scholarly.search_pubs(query)
        result = next(search_query, None)
        if result:
            bib = result.get("bib") or {}
            return (
                f"Title: {bib.get('title', 'n/a')}\nYear: {bib.get('pub_year', 'n/a')}\n"
                f"Venue: {bib.get('venue', 'n/a')}\nAbstract: {bib.get('abstract', 'n/a')}"
            )
        else:
            return "No results found on Google Scholar."
    except Exception as e:
        return f"Error querying Google Scholar: {e}"


#: Seconds `query_scholar` waits for scholarly to find a working free proxy.
_SCHOLAR_PROXY_BUDGET_ENV = "SOG_SCHOLAR_PROXY_BUDGET_SECONDS"


#: The one probe allowed to run: ``{"thread", "pg", "outcome"}`` of the last one started.
_proxy_probe: dict = {}
_proxy_probe_lock = threading.Lock()


def _set_up_free_proxies(pg):
    """``pg.FreeProxies()`` under a wall-clock budget; the generator to use, or raises.

    The probing runs on a daemon thread because scholarly offers no way to stop it: past the budget
    the call returns an error and the thread is left to finish its own retries in the background.
    Only one such thread runs at a time. Each call used to start its own, so repeated calls piled up
    threads each still working through up to 200 proxies (hunt 2026-09-30, uT6-literature-9 review);
    a call that finds a probe still running waits on that one, and uses its generator if it succeeds.
    """
    budget = _env_number(_SCHOLAR_PROXY_BUDGET_ENV, 90.0)
    with _proxy_probe_lock:
        running = _proxy_probe.get("thread")
        if running is not None and running.is_alive():
            worker, pg, outcome = running, _proxy_probe["pg"], _proxy_probe["outcome"]
        else:
            outcome = {}

            def probe(pg=pg, outcome=outcome) -> None:
                try:
                    outcome["ok"] = pg.FreeProxies(timeout=1, wait_time=min(budget, 120))
                except BaseException as exc:  # handed to the caller's thread below
                    outcome["error"] = exc

            worker = threading.Thread(target=probe, name="scholar-free-proxies", daemon=True)
            _proxy_probe.update(thread=worker, pg=pg, outcome=outcome)
            worker.start()
    worker.join(budget)
    if worker.is_alive():
        raise TimeoutError(
            f"no working free proxy was found within {budget:.0f}s (raise {_SCHOLAR_PROXY_BUDGET_ENV}); the "
            "search continues in the background and the next call waits on it rather than starting another"
        )
    if "error" in outcome:
        raise outcome["error"]
    if not outcome.get("ok", True):
        raise RuntimeError("scholarly could not set up a free proxy")
    return pg


def query_pubmed(query: str, max_papers: int = 10, max_retries: int = 3) -> str:
    """Query PubMed for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.
    - max_papers (int): The maximum number of papers to retrieve (default: 10).
    - max_retries (int): Maximum number of retry attempts with modified queries (default: 3).

    Returns
    -------
    - str: The formatted search results or an error message.

    """
    from pymed import PubMed

    try:
        # An honest identity: this tool's name, and the operator's address only if they gave one
        # (NCBI_EMAIL). It sent tool="MyTool", email="your-email@example.com" to NCBI on every call
        # (hunt 2026-09-30, uT6-literature-24).
        pubmed = PubMed(tool="spatialomicsgym", email=os.environ.get("NCBI_EMAIL") or None)

        # Initial attempt
        papers = list(pubmed.query(query, max_results=max_papers))
        used_query = query

        # Retry with modified queries if no results
        retries = 0
        while not papers and retries < max_retries:
            retries += 1
            # Simplify query with each retry by removing the last word
            simplified_query = " ".join(query.split()[:-retries]) if len(query.split()) > retries else query
            time.sleep(1)  # Add delay between requests
            papers = list(pubmed.query(simplified_query, max_results=max_papers))
            used_query = simplified_query

        if papers:
            results = "\n\n".join(
                [f"Title: {paper.title}\nAbstract: {paper.abstract}\nJournal: {paper.journal}" for paper in papers]
            )
            if used_query != query:
                # These hits answer a shorter query than the one asked, and nothing said so: results
                # for "Xenium glioblastoma" came back as answers to "Xenium glioblastoma CD8" (hunt
                # 2026-09-30, uT6-literature-24).
                results = (
                    f"Query used: {used_query!r} -- the original query {query!r} returned nothing, so "
                    f"these results do not match all of its terms.\n\n{results}"
                )
            return results
        else:
            return "No papers found on PubMed after multiple query attempts."
    except Exception as e:
        return f"Error querying PubMed: {e}"


def search_google(query: str, num_results: int = 3, language: str = "en") -> list[dict]:
    """Search using Google search.

    Args:
        query (str): The search query (e.g., "protocol text or seach question")
        num_results (int): Number of results to return (default: 10)
        language (str): Language code for search results (default: 'en')
        pause (float): Pause between searches to avoid rate limiting (default: 2.0 seconds)

    Returns:
        List[dict]: List of dictionaries containing search results with title and URL

    """
    # Initialised outside the `try` on purpose: it used to be the first statement inside it, so any
    # failure before the loop -- `print` raising UnicodeEncodeError on a non-UTF-8 stdout is the one
    # that happens -- left `return results_string` raising UnboundLocalError out of a function whose
    # whole shape says it never fails.
    results_string = ""
    try:
        search_query = f"{query}"

        print(f"Searching for {search_query} with {num_results} results and {language} language")

        for res in _google_search()(search_query, num_results=num_results, lang=language, advanced=True):
            print(f"Found result: {res.title}")
            title = res.title
            url = res.url
            description = res.description

            results_string += f"Title: {title}\nURL: {url}\nDescription: {description}\n\n"

    except Exception as e:
        # The failure has to travel in the return value. Printing it and returning "" hands the
        # caller a result indistinguishable from a search that ran and found nothing -- the one
        # answer that sends an agent confidently down the wrong path. That mattered most for the
        # missing-package case, where the return was the only place `pip install
        # googlesearch-python` could have surfaced, and stdout here is captured by the MCP layer.
        message = f"Error performing search: {e}"
        # Suppressed because the handler's own report must not become the next exception: if the
        # thing that failed above was `print` hitting a stdout it cannot encode, printing the
        # message describing it fails the same way and takes the function down from inside the
        # `except`. The return value is the channel that reaches the caller regardless.
        with contextlib.suppress(Exception):
            print(message)
        return results_string or message
    return results_string


def advanced_web_search_claude(
    query: str,
    max_searches: int = 1,
    max_retries: int = 3,
) -> tuple[str, list[dict[str, str]], list]:
    """
    Initiate an advanced web search by launching a specialized agent to collect relevant information and citations through multiple rounds of web searches for a given query.
    Craft the query carefully for the search agent to find the most relevant information.

    Parameters
    ----------
    query : str
        The search phrase you want Claude to look up.
    max_searches : int, optional
        Upper-bound on searches Claude may issue inside this request.
    max_retries : int, optional
        Maximum number of retry attempts with exponential backoff.

    Returns
    -------
    full_text : str
        A formatted string containing the full text response from Claude and the citations.
    """
    import random

    import anthropic

    model, client_kwargs, missing = _claude_search_route()
    if "claude" not in model.lower():
        raise ValueError("Model must be a Claude model.")
    if missing:
        raise ValueError(f"advanced_web_search_claude needs {missing}.")

    client = anthropic.Anthropic(**client_kwargs)
    tool_def = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_searches,
    }

    delay = random.randint(1, 10)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": query}],
                tools=[tool_def],
            )

            paragraphs, citations = [], []
            response.content = response.content
            formatted_response = ""
            for blk in response.content:
                if blk.type == "text":
                    paragraphs.append(blk.text)
                    formatted_response += blk.text

                    if blk.citations:
                        for cite in blk.citations:
                            citations.append({"url": cite.url, "title": cite.title, "cited_text": cite.cited_text})
                            formatted_response += f"(Citation: {cite.title} - {cite.url})"
            return formatted_response

        except Exception as e:
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            print(f"Error performing web search after {max_retries} attempts: {str(e)}")
            return f"Error performing web search after {max_retries} attempts: {str(e)}"


def _claude_search_route() -> tuple[str, dict, str | None]:
    """``(model, anthropic_client_kwargs, None)``, or ``(model, {}, what_is_missing)``.

    The model is the agent's own (``default_config.llm``); the credential follows the provider that
    serves it. ``default_config.api_key`` is the custom gateway's key ("Only for custom models") and
    was sent to api.anthropic.com with no ``base_url`` -- one provider's credential posted to another
    (hunt 2026-09-30, uT6-literature-25). Now a gateway's key goes only to that gateway, the Anthropic
    API gets only ``ANTHROPIC_API_KEY``, and anything else is reported missing rather than tried.
    """
    try:
        from spatialomicsgym.config import default_config
    except ImportError:
        return "", {}, "the spatialomicsgym config (it could not be imported)"
    model = str(getattr(default_config, "llm", "") or "")
    if "claude" not in model.lower():
        return model, {}, f"a Claude model as the agent's LLM (configured: {model or 'none'})"
    base_url = getattr(default_config, "base_url", None)
    try:
        from spatialomicsgym.llm import effective_source

        source = effective_source(model, base_url=base_url, config=default_config)
    except Exception:  # an unresolvable provider is not one this call can use
        source = "Custom" if base_url else "Anthropic"
    if source == "Custom":
        api_key = getattr(default_config, "api_key", None)
        if base_url and api_key:
            return model, {"api_key": api_key, "base_url": base_url}, None
        return model, {}, "SOG_CUSTOM_BASE_URL and SOG_CUSTOM_API_KEY for the custom endpoint serving the Claude model"
    if source == "Anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            return model, {"api_key": api_key}, None
        return model, {}, "ANTHROPIC_API_KEY"
    return model, {}, f"the Anthropic API (the configured Claude model is served through {source})"


def extract_url_content(url: str) -> str:
    """Extract the text content of a webpage using requests and BeautifulSoup.

    Args:
        url: Webpage URL to extract content from

    Returns:
        Text content of the webpage

    """
    try:
        response = _bounded_get(url, headers={"User-Agent": "Mozilla/5.0"})
    except (requests.RequestException, ValueError) as exc:
        return f"Error fetching {url}: {exc}"
    if not response.ok:
        # A 403 "access denied" or a 404 page used to come back as if it were the article's text
        # (hunt 2026-09-30, uT6-literature-27).
        return f"Error fetching {url}: HTTP {response.status_code}. The page's content was not retrieved."

    # Check if the response is in text format
    if "text/plain" in response.headers.get("Content-Type", "") or "application/json" in response.headers.get(
        "Content-Type", ""
    ):
        return response.text.strip()  # Return plain text or JSON response directly

    # If it's HTML, use BeautifulSoup to parse
    soup = _beautiful_soup()(response.text, "html.parser")

    # Try to find main content first, fallback to body
    # ... and to the whole document: an XML/RSS body parsed as HTML has no <body>, and calling None
    # raised "'NoneType' object is not callable" (hunt 2026-09-30, uT6-literature-27).
    content = soup.find("main") or soup.find("article") or soup.body or soup

    # Remove unwanted elements
    for element in content(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
        element.decompose()

    # Extract text with better formatting
    paragraphs = content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    cleaned_text = []

    for p in paragraphs:
        text = p.get_text().strip()
        if text:  # Only add non-empty paragraphs
            cleaned_text.append(text)

    return "\n\n".join(cleaned_text)


def extract_pdf_content(url: str) -> str:
    """Extract the text content of a PDF file given its URL.

    Args:
        url: URL of the PDF file to extract text from

    Returns:
        The extracted text content from the PDF

    """
    try:
        # First, before any download. The import used to sit inside a `try` that only printed its
        # error, so a missing PyPDF2 came back as "an image-based PDF requiring OCR" and sent the
        # agent off to OCR a PDF it could have read (hunt 2026-09-30, uT6-literature-6).
        pdf_reader_class = _require("PyPDF2").PdfReader

        # Check if the URL ends with .pdf
        # (its path, so "...a.pdf?download=1" is a PDF and not a landing page)
        if not urlparse(url).path.lower().endswith(".pdf"):
            # If not, try to find a PDF link on the page
            response = _bounded_get(url)
            if response.status_code == 200:
                # Look for PDF links in the HTML content
                pdf_links = re.findall(r'href=[\'"]([^\'"]+\.pdf(?:[?#][^\'"]*)?)[\'"]', response.text)
                if pdf_links:
                    # Use the first PDF link found, resolved against the page it was found on. It was
                    # glued to the site root, so "files/a.pdf" on /dir/page.html became /files/a.pdf
                    # and "//cdn.x.org/a.pdf" became a path on the page's own host (hunt 2026-09-30,
                    # uT6-literature-28).
                    url = urljoin(response.url or url, pdf_links[0])
                else:
                    return f"No PDF file found at {url}. Please provide a direct link to a PDF file."

        # Download the PDF
        response = _bounded_get(url)

        # Check if we actually got a PDF file (by checking content type or magic bytes)
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type and not response.content.startswith(b"%PDF"):
            return f"The URL did not return a valid PDF file. Content type: {content_type}"

        pdf_file = BytesIO(response.content)

        pdf_reader = pdf_reader_class(pdf_file)
        text = ""
        failed_pages = 0
        for page in pdf_reader.pages:
            try:
                text += (page.extract_text() or "") + "\n\n"
            except Exception:
                failed_pages += 1

        # Clean up the text
        text = re.sub(r"\s+", " ", text).strip()

        if not text and failed_pages:
            return (
                f"Error extracting text from PDF: extraction failed on {failed_pages} of {len(pdf_reader.pages)} pages."
            )
        if not text:
            return "The PDF file did not contain any extractable text. It may be an image-based PDF requiring OCR."

        return text

    except requests.exceptions.RequestException as e:
        return f"Error downloading PDF: {str(e)}"
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"
