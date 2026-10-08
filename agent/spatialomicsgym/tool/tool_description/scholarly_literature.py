"""Schema for the scholarly literature tools in ``tool/scholarly_literature.py``.

This list -- not the docstrings next door -- is what ``read_module2api()`` collects and
``utils/formatting.py:textify_api_dict`` renders into the function signatures the model sees. A
parameter absent from here is a knob the agent cannot turn; a parameter here that the function does
not accept is a ``TypeError`` on the first call. Keep both files in step.

Three services, and the division of labour between them is worth stating because picking the wrong
one wastes a turn. **Europe PMC** is the biomedical index and the only one of the three that serves
open-access **full text** -- use it to read methods. **OpenAlex** is the open scholarly graph -- use
it for who wrote what, where they work, what cites what, and how much. **Crossref** is the DOI
registration agency -- it is authoritative for what a DOI *is*, who published it and who funded it,
and it is the only one of the three with a funder registry.

Adapted from ToolUniverse -- https://github.com/mims-harvard/ToolUniverse -- at commit
``f075c2a75e8b35ae5dbb220d48d4e87e980388b1``, Apache-2.0, Copyright [2025] [ToolUniverse team].
CHANGED BY SPATIALOMICSGYM as Apache-2.0 section 4(b) requires: the upstream JSON tool catalogs
(``europe_pmc_tools.json``, ``openalex_tools.json``, ``crossref_tools.json``) were re-expressed as
this Python ``description`` list; the config-driven ``operation`` discriminator was replaced by one
function per operation; no API key, ``mailto`` or browser User-Agent is advertised because none is
needed; and ``extract_terms_from_fulltext`` was dropped from the search tool in favour of
``europe_pmc_get_fulltext_snippets``. The implementation module's docstring lists the upstream
defects fixed here (F9-F15) with the measurement behind each. See ``VENDORING.md``.
"""

description = [
    {
        "description": (
            "Search Europe PMC -- PubMed, PubMed Central, preprints and patents in one index -- and "
            "get back title, journal, year, authors, DOI, PMID, PMCID and abstract per hit. This is "
            "the first stop for a biomedical literature question. num_found is Europe PMC's own "
            "total across the whole corpus while the returned list is only the page you asked for, "
            "so a small list never means 'few papers exist' -- read num_found and truncated. The "
            'query supports fielded syntax: AUTH:"Regev A", PUB_YEAR:2024, JOURNAL:"Nature '
            'Methods", TITLE:"spatial transcriptomics", and AND/OR/NOT between them. Every hit '
            "carries a pmcid when open-access full text exists, which is what the fulltext tools "
            "take. Print the returned dict."
        ),
        "name": "europe_pmc_search_articles",
        "optional_parameters": [
            {
                "name": "limit",
                "type": "int",
                "description": "Number of articles to return, 1-100",
                "default": 5,
            },
            {
                "name": "require_fulltext",
                "type": "bool",
                "description": (
                    "Return only articles whose full text Europe PMC has indexed. Narrower than "
                    "'has a free full-text link somewhere', and the right filter when the next step "
                    "is reading methods rather than skimming abstracts"
                ),
                "default": False,
            },
            {
                "name": "fulltext_terms",
                "type": "list[str]",
                "description": (
                    "Terms that must appear in the article body rather than merely its abstract -- "
                    "use this for a method, reagent or software name that is only ever mentioned in "
                    "the methods section. Several terms are OR-ed, so a hit needs any one of them. "
                    "Only articles whose full text Europe PMC has indexed can match"
                ),
                "default": None,
            },
            {
                "name": "enrich_missing_abstract",
                "type": "bool",
                "description": (
                    "For hits with no abstract in the index but with a pmcid, fetch the article and "
                    "pull the abstract from it. Costs one extra request per such hit"
                ),
                "default": False,
            },
        ],
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": (
                    "What to search for. Free text, or Europe PMC fielded syntax such as "
                    "'AUTH:\"Regev A\" AND PUB_YEAR:2024'"
                ),
            },
        ],
    },
    {
        "description": (
            "List the articles that cite a given article, newest and most relevant first. Use it to "
            "walk forward in time from a foundational paper to the work that built on it -- which "
            "methods were adopted, which results were challenged. num_found is the true citation "
            "count; the returned list is one page of it. Print the returned dict."
        ),
        "name": "europe_pmc_get_citations",
        "optional_parameters": [
            {
                "name": "source",
                "type": "str",
                "description": (
                    "Which identifier namespace article_id belongs to: 'MED' for a PMID, 'PMC' for "
                    "a PMCID, 'PPR' for a preprint id"
                ),
                "default": "MED",
            },
            {
                "name": "page_size",
                "type": "int",
                "description": "Citations per page, 1-100",
                "default": 25,
            },
            {
                "name": "page",
                "type": "int",
                "description": "Which page to return, 1-based",
                "default": 1,
            },
        ],
        "required_parameters": [
            {
                "name": "article_id",
                "type": "str",
                "description": (
                    "The article's identifier in the namespace named by source, e.g. '32226684' "
                    "with source='MED' or 'PMC7096075' with source='PMC'"
                ),
            },
        ],
    },
    {
        "description": (
            "List the works a given article cites -- its reference list. Use it to walk backward to "
            "the methods and datasets a paper was built on. num_found is the full length of the "
            "reference list; the returned list is one page of it. Print the returned dict."
        ),
        "name": "europe_pmc_get_references",
        "optional_parameters": [
            {
                "name": "source",
                "type": "str",
                "description": (
                    "Which identifier namespace article_id belongs to: 'MED' for a PMID, 'PMC' for "
                    "a PMCID, 'PPR' for a preprint id"
                ),
                "default": "MED",
            },
            {
                "name": "page_size",
                "type": "int",
                "description": "References per page, 1-100",
                "default": 25,
            },
            {
                "name": "page",
                "type": "int",
                "description": "Which page to return, 1-based",
                "default": 1,
            },
        ],
        "required_parameters": [
            {
                "name": "article_id",
                "type": "str",
                "description": (
                    "The article's identifier in the namespace named by source, e.g. '32226684' "
                    "with source='MED' or 'PMC7096075' with source='PMC'"
                ),
            },
        ],
    },
    {
        "description": (
            "Fetch the complete open-access full text of a PubMed Central article as readable text "
            "-- the whole paper, not just the abstract. This is how you read a methods section. It "
            "tries four sources in order (Europe PMC, the NCBI PMC OAI record, NCBI efetch, then "
            "the PMC article page) and reports in trace what each one returned, so a failure says "
            "which source refused rather than just 'not available'. Give it a pmcid, or a pmid and "
            "it resolves the PMCID first. Articles that are not open access have no full text to "
            "serve and will return an error saying so -- that is a property of the article, not a "
            "fault. Long articles are truncated at max_chars with truncated=True and total_chars "
            "telling you the real length. If you only need the passages mentioning a few terms, "
            "europe_pmc_get_fulltext_snippets is far cheaper. Print the returned dict."
        ),
        "name": "europe_pmc_get_fulltext",
        "optional_parameters": [
            {
                "name": "pmcid",
                "type": "str",
                "description": "PubMed Central id, e.g. 'PMC7096075' or bare '7096075'",
                "default": None,
            },
            {
                "name": "pmid",
                "type": "str",
                "description": (
                    "PubMed id, e.g. '32226684'. Resolved to a PMCID first; an article with no PMC "
                    "record has no open-access full text"
                ),
                "default": None,
            },
            {
                "name": "article_id",
                "type": "str",
                "description": "Alternative identifier, paired with source_db",
                "default": None,
            },
            {
                "name": "source_db",
                "type": "str",
                "description": "Namespace for article_id: 'MED' or 'PMC'",
                "default": None,
            },
            {
                "name": "output_format",
                "type": "str",
                "description": (
                    "'text' for readable prose with its paragraphs kept, or 'raw' for the "
                    "underlying JATS XML when you need the markup"
                ),
                "default": "text",
            },
            {
                "name": "max_chars",
                "type": "int",
                "description": (
                    "Characters to return before truncating. total_chars always reports the untruncated length"
                ),
                "default": 200000,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Search inside one article's full text and return only the passages around each term, "
            "with a count of how often each term occurs. Use this instead of europe_pmc_get_fulltext "
            "when the question is 'does this paper use X, and how' -- it answers in a few hundred "
            "characters rather than tens of thousands. counts reports every occurrence found even "
            "when only the first few snippets are returned, and truncated says plainly when "
            "passages were left out because a cap was hit. Searches the abstract and body only, so "
            "a term that appears nowhere but the bibliography does not count as a mention. Print "
            "the returned dict."
        ),
        "name": "europe_pmc_get_fulltext_snippets",
        "optional_parameters": [
            {
                "name": "pmcid",
                "type": "str",
                "description": "PubMed Central id, e.g. 'PMC7096075' or bare '7096075'",
                "default": None,
            },
            {
                "name": "pmid",
                "type": "str",
                "description": "PubMed id, e.g. '32226684'. Resolved to a PMCID first",
                "default": None,
            },
            {
                "name": "article_id",
                "type": "str",
                "description": "Alternative identifier, paired with source_db",
                "default": None,
            },
            {
                "name": "source_db",
                "type": "str",
                "description": "Namespace for article_id: 'MED' or 'PMC'",
                "default": None,
            },
            {
                "name": "window_chars",
                "type": "int",
                "description": "Characters of context to keep either side of each match",
                "default": 220,
            },
            {
                "name": "max_snippets_per_term",
                "type": "int",
                "description": ("Most passages to return per term. counts still reports every occurrence"),
                "default": 3,
            },
            {
                "name": "max_total_chars",
                "type": "int",
                "description": "Total character budget across all snippets",
                "default": 8000,
            },
        ],
        "required_parameters": [
            {
                "name": "terms",
                "type": "list[str]",
                "description": (
                    "The words or phrases to find in the body, e.g. ['Visium', 'DAPI', "
                    "'10x Genomics']. Matching is case-insensitive"
                ),
            },
        ],
    },
    {
        "description": (
            "Fetch one article's full text already split into its named sections -- abstract, "
            "introduction, methods, results, discussion, conclusion -- plus the title and the "
            "article's own section headings. Use this when you want a specific part of a paper: "
            "read data['methods'] rather than scanning the whole text. counts gives each section's "
            "length so you can see what the article actually contains before reading it. Read one "
            "section rather than the whole text. Print the returned dict."
        ),
        "name": "europe_pmc_get_structured_fulltext",
        "optional_parameters": [
            {
                "name": "pmcid",
                "type": "str",
                "description": "PubMed Central id, e.g. 'PMC7096075' or bare '7096075'",
                "default": None,
            },
            {
                "name": "pmid",
                "type": "str",
                "description": "PubMed id, e.g. '32226684'. Resolved to a PMCID first",
                "default": None,
            },
            {
                "name": "max_section_chars",
                "type": "int",
                "description": "Characters to keep per section before truncating that section",
                "default": 50000,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Search OpenAlex -- an open index of over 250 million scholarly works across every "
            "field, not just biomedicine -- with the filters you most often want already exposed as "
            "arguments: year range, open access, and full-text terms. Each hit carries title, year, "
            "venue, authors, institutions, DOI, citation count and abstract. Prefer Europe PMC for "
            "a biomedical question where you may want to read the paper; prefer OpenAlex for "
            "coverage beyond biomedicine, for citation counts, or when you need author and "
            "institution links. Note that many OpenAlex records now carry no abstract at all, so "
            "'abstract not available' is common and not an error. num_found is the total matching "
            "the query. Print the returned dict."
        ),
        "name": "openalex_literature_search",
        "optional_parameters": [
            {
                "name": "max_results",
                "type": "int",
                "description": "Number of works to return, 1-100",
                "default": 10,
            },
            {
                "name": "year_from",
                "type": "int",
                "description": "Earliest publication year to include, e.g. 2020",
                "default": None,
            },
            {
                "name": "year_to",
                "type": "int",
                "description": "Latest publication year to include, e.g. 2025",
                "default": None,
            },
            {
                "name": "open_access",
                "type": "bool",
                "description": ("True for open-access works only, False for closed only, omit for both"),
                "default": None,
            },
            {
                "name": "require_has_fulltext",
                "type": "bool",
                "description": (
                    "Restrict to works whose full text OpenAlex has indexed. This narrows results "
                    "substantially and is not implied by fulltext_terms, because OpenAlex's "
                    "full-text search also covers title and abstract"
                ),
                "default": False,
            },
            {
                "name": "fulltext_terms",
                "type": "list[str]",
                "description": (
                    "Terms to find in the indexed full text, e.g. ['Visium', 'MERFISH']. Each term "
                    "is applied as a separate filter, so all of them must match"
                ),
                "default": None,
            },
        ],
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": "What to search for, e.g. 'spatial transcriptomics tumour microenvironment'",
            },
        ],
    },
    {
        "description": (
            "Query OpenAlex works with its native filter syntax, for the questions the convenience "
            "wrapper openalex_literature_search cannot express. filter takes comma-separated "
            "clauses that are AND-ed: 'publication_year:2025', 'is_oa:true', "
            "'authorships.institutions.ror:https://ror.org/05a0ya142', 'cited_by_count:>100', "
            "'type:review'. sort takes 'cited_by_count:desc', 'publication_date:desc' or "
            "'relevance_score:desc'. Use it to enumerate an institution's output, a journal's "
            "papers in a year, or the most-cited work on a topic. An unrecognised filter name is "
            "rejected by OpenAlex rather than silently ignored. Print the returned dict."
        ),
        "name": "openalex_search_works",
        "optional_parameters": [
            {
                "name": "search",
                "type": "str",
                "description": "Free-text search across title, abstract and full text",
                "default": None,
            },
            {
                "name": "filter",
                "type": "str",
                "description": (
                    "Comma-separated OpenAlex filter clauses, AND-ed, e.g. 'publication_year:2025,is_oa:true'"
                ),
                "default": None,
            },
            {
                "name": "per_page",
                "type": "int",
                "description": "Works per page, 1-100",
                "default": 10,
            },
            {
                "name": "page",
                "type": "int",
                "description": "Which page to return, 1-based",
                "default": 1,
            },
            {
                "name": "sort",
                "type": "str",
                "description": (
                    "Sort order, e.g. 'cited_by_count:desc' or 'publication_date:desc'. Omit for relevance"
                ),
                "default": None,
            },
            {
                "name": "require_has_fulltext",
                "type": "bool",
                "description": "Restrict to works whose full text OpenAlex has indexed",
                "default": False,
            },
            {
                "name": "fulltext_terms",
                "type": "list[str]",
                "description": "Terms to find in the indexed full text; all of them must match",
                "default": None,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Fetch one OpenAlex work by its id and get the full record: title, year, venue, type, "
            "open-access status, every author with their institutions, concepts, citation count, "
            "reference count and abstract. Use it after a search when you need the detail a hit "
            "summary leaves out. Print the returned dict."
        ),
        "name": "openalex_get_work",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "openalex_id",
                "type": "str",
                "description": (
                    "OpenAlex work id, e.g. 'W2005501262', or the full "
                    "'https://openalex.org/W2005501262'. Every search hit carries one"
                ),
            },
        ],
    },
    {
        "description": (
            "Fetch an OpenAlex work by DOI rather than by OpenAlex id. Use it to turn a DOI from a "
            "reference list, a dataset record or a user's message into a full bibliographic record "
            "with citation count, authors and institutions. Print the returned dict."
        ),
        "name": "openalex_get_work_by_doi",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "doi",
                "type": "str",
                "description": ("The DOI, bare as '10.1038/s41586-020-2649-2' or as a 'https://doi.org/...' URL"),
            },
        ],
    },
    {
        "description": (
            "Find researchers by name in OpenAlex and get their id, ORCID, last known institution, "
            "works count and citation count. Use it to disambiguate an author before enumerating "
            "their work -- names collide, OpenAlex ids do not. Print the returned dict."
        ),
        "name": "openalex_search_authors",
        "optional_parameters": [
            {
                "name": "per_page",
                "type": "int",
                "description": "Authors per page, 1-100",
                "default": 5,
            },
            {
                "name": "page",
                "type": "int",
                "description": "Which page to return, 1-based",
                "default": 1,
            },
        ],
        "required_parameters": [
            {
                "name": "search",
                "type": "str",
                "description": "Researcher name, e.g. 'Aviv Regev'",
            },
        ],
    },
    {
        "description": (
            "Fetch one researcher's OpenAlex profile: ORCID, affiliations, works count, citation "
            "count, h-index and the topics they publish in. To then list their papers, pass "
            "filter='authorships.author.id:<id>' to openalex_search_works. Print the returned dict."
        ),
        "name": "openalex_get_author",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "author_id",
                "type": "str",
                "description": (
                    "OpenAlex author id, e.g. 'A5023888391'. openalex_search_authors returns one on every hit"
                ),
            },
        ],
    },
    {
        "description": (
            "Find research institutions by name in OpenAlex and get their id, ROR id, country, type "
            "and output counts. Use it to resolve an affiliation string into an id you can filter "
            "works by. Print the returned dict."
        ),
        "name": "openalex_search_institutions",
        "optional_parameters": [
            {
                "name": "per_page",
                "type": "int",
                "description": "Institutions per page, 1-100",
                "default": 5,
            },
            {
                "name": "page",
                "type": "int",
                "description": "Which page to return, 1-based",
                "default": 1,
            },
        ],
        "required_parameters": [
            {
                "name": "search",
                "type": "str",
                "description": "Institution name, e.g. 'Broad Institute'",
            },
        ],
    },
    {
        "description": (
            "Fetch one institution's OpenAlex record: ROR id, country, type, homepage, works count, "
            "citation count and alternate names. To then list its output, pass "
            "filter='authorships.institutions.lineage:<id>' to openalex_search_works. Print the "
            "returned dict."
        ),
        "name": "openalex_get_institution",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "institution_id",
                "type": "str",
                "description": (
                    "OpenAlex institution id, e.g. 'I4210109156'. openalex_search_institutions returns one on every hit"
                ),
            },
        ],
    },
    {
        "description": (
            "Find journals, conferences and repositories by name in OpenAlex and get their id, "
            "ISSN, publisher, open-access status and output counts. Use it to resolve a venue name "
            "before filtering works by it. Print the returned dict."
        ),
        "name": "openalex_search_sources",
        "optional_parameters": [
            {
                "name": "per_page",
                "type": "int",
                "description": "Sources per page, 1-100",
                "default": 5,
            },
            {
                "name": "page",
                "type": "int",
                "description": "Which page to return, 1-based",
                "default": 1,
            },
        ],
        "required_parameters": [
            {
                "name": "search",
                "type": "str",
                "description": "Venue name, e.g. 'Nature Methods'",
            },
        ],
    },
    {
        "description": (
            "Fetch one venue's OpenAlex record: ISSNs, publisher, whether it is fully open access, "
            "works count and citation count. To then list what it published, pass "
            "filter='primary_location.source.id:<id>' to openalex_search_works. Print the returned "
            "dict."
        ),
        "name": "openalex_get_source",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "source_id",
                "type": "str",
                "description": (
                    "OpenAlex source id, e.g. 'S4210194219'. openalex_search_sources returns one on every hit"
                ),
            },
        ],
    },
    {
        "description": (
            "Search Crossref, the DOI registration agency's own metadata -- every DOI its members "
            "have registered, across all publishers and all disciplines, including the records "
            "OpenAlex and Europe PMC derive from. Returns title, DOI, journal, publisher, type, "
            "dates, authors with ORCIDs and affiliations, reference and citation counts, licence, "
            "and funder acknowledgements. Prefer it when you care about the registered record "
            "itself: publisher, licence, funding, or the exact issue and page numbers. filter "
            "takes comma-separated clauses such as 'from-pub-date:2025-01-01', 'type:journal-"
            "article', 'has-funder:true', 'has-orcid:true'. Print the returned dict."
        ),
        "name": "crossref_search_works",
        "optional_parameters": [
            {
                "name": "limit",
                "type": "int",
                "description": "Number of works to return, 1-1000",
                "default": 10,
            },
            {
                "name": "offset",
                "type": "int",
                "description": (
                    "How many results to skip, for paging. Crossref caps offset + limit at 10000; "
                    "past that, narrow the query with filter rather than paging further"
                ),
                "default": None,
            },
            {
                "name": "filter",
                "type": "str",
                "description": (
                    "Comma-separated Crossref filter clauses, AND-ed, e.g. "
                    "'type:journal-article,from-pub-date:2024-01-01'"
                ),
                "default": None,
            },
        ],
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": "What to search for, e.g. 'spatial transcriptomics'",
            },
        ],
    },
    {
        "description": (
            "Fetch the registered Crossref record for one DOI: title, container journal, publisher, "
            "type, issued and published dates, volume, issue, pages, ISSNs, every author with ORCID "
            "and affiliation, licence terms, funder list with award numbers, reference count and "
            "citation count. Use it to verify a citation, to find who funded a paper, or to check a "
            "licence before reusing content. Print the returned dict."
        ),
        "name": "crossref_get_work",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "doi",
                "type": "str",
                "description": ("The DOI, bare as '10.1038/s41586-021-03634-9' or as a 'https://doi.org/...' URL"),
            },
        ],
    },
    {
        "description": (
            "Fetch a journal's Crossref record by ISSN: title, publisher, all its ISSNs, the total "
            "DOIs registered, and coverage flags saying what fraction of its records carry "
            "abstracts, ORCIDs, licences, full-text links, funders and references. Use it to judge "
            "how complete a journal's metadata is before relying on a field being present. Print "
            "the returned dict."
        ),
        "name": "crossref_get_journal",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "issn",
                "type": "str",
                "description": "Journal ISSN, print or electronic, e.g. '1548-7091'",
            },
        ],
    },
    {
        "description": (
            "Search the Crossref Funder Registry -- the controlled list of research funders used in "
            "funding acknowledgements. Returns each funder's id, name, alternate names, country and "
            "any replaced-by pointers. Use it to turn a funder name into the id you then pass to "
            "crossref_get_funder, or to crossref_search_works as 'funder:<id>'. Search results "
            "carry no work counts and no descendant list at all -- those live only on the "
            "single-funder record, so a missing count here means 'not in this response', not "
            "'zero'. Print the returned dict."
        ),
        "name": "crossref_list_funders",
        "optional_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": (
                    "Funder name to search for, e.g. 'National Institutes of Health'. Omit to list "
                    "the registry from the start"
                ),
                "default": None,
            },
            {
                "name": "limit",
                "type": "int",
                "description": "Number of funders to return, 1-1000",
                "default": 20,
            },
            {
                "name": "offset",
                "type": "int",
                "description": "How many results to skip, for paging",
                "default": None,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Fetch one funder's registry record: name, alternate names, country, and its sub-"
            "organisations. Two counts are reported and they mean different things -- work_count is "
            "what this funder alone is credited with, descendant_work_count includes every body "
            "beneath it, which for an umbrella funder is far larger. descendants lists those "
            "sub-organisation ids, and descendant_names gives the names the registry has for them; "
            "not every descendant is named, so a missing name is normal. Pass any of those ids to "
            "crossref_search_works as 'funder:<id>' to see the actual papers. Print the returned "
            "dict."
        ),
        "name": "crossref_get_funder",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "funder_id",
                "type": "str",
                "description": (
                    "Funder Registry id, e.g. '100000002' for the NIH. crossref_list_funders returns one on every hit"
                ),
            },
        ],
    },
    {
        "description": (
            "List the work types Crossref recognises -- 'journal-article', 'posted-content' (which "
            "is what preprints are), 'book-chapter', 'dataset', 'proceedings-article' and the rest. "
            "Use it before writing a 'type:...' filter, so the filter names a type that exists "
            "rather than silently matching nothing. Takes no arguments. Print the returned dict."
        ),
        "name": "crossref_list_types",
        "optional_parameters": [],
        "required_parameters": [],
    },
    {
        "description": (
            "Search Crossref members -- the publishers and societies that register DOIs. Returns "
            "each member's id, primary name, location, the DOI counts they have registered, and "
            "coverage flags for abstracts, ORCIDs, licences and references. Match is against the "
            "registered legal name, which is often not the imprint you know: a well-known brand "
            "name may return nothing while the shorter root word finds it. If a search comes back "
            "empty, try one distinctive word rather than the full name. Print the returned dict."
        ),
        "name": "crossref_search_members",
        "optional_parameters": [
            {
                "name": "limit",
                "type": "int",
                "description": "Number of members to return, 1-1000",
                "default": 20,
            },
            {
                "name": "offset",
                "type": "int",
                "description": "How many results to skip, for paging",
                "default": None,
            },
        ],
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": "Publisher name to search for, e.g. 'Springer' or 'eLife'",
            },
        ],
    },
    {
        "description": (
            "Fetch one publisher's Crossref member record: primary and alternate names, location, "
            "total DOIs registered, current and backfile counts, and the coverage flags saying what "
            "fraction of their records carry abstracts, ORCIDs, licences, full-text links, funders "
            "and references. Use it to judge whether a publisher's metadata is rich enough to rely "
            "on, or pass the id to crossref_search_works as 'member:<id>' to list their output. "
            "Print the returned dict."
        ),
        "name": "crossref_get_member",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "member_id",
                "type": "str",
                "description": (
                    "Crossref member id, e.g. '297' for Springer. crossref_search_members returns one on every hit"
                ),
            },
        ],
    },
]
