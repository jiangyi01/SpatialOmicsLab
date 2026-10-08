description = [
    {
        "name": "search_arxiv_advanced",
        "description": "Search arXiv with advanced filtering including category restriction, "
        "date range filtering, phrase search, and raw arXiv query syntax. "
        "Returns structured results with paper metadata (authors, abstract, URLs, categories). "
        "Complements the basic query_arxiv by supporting category filtering (e.g., q-bio.GN), "
        "date filtering, pagination, and sort options.",
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": "Search query (plain text or raw arXiv syntax with ti:/au:/abs:/cat: prefixes)",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "max_results",
                "type": "int",
                "description": "Maximum number of results (1-2000)",
                "default": 10,
            },
            {
                "name": "phrase",
                "type": "bool",
                "description": "Treat query as a single phrase",
                "default": False,
            },
            {
                "name": "category",
                "type": "str",
                "description": "Restrict to arXiv category (e.g., 'q-bio.GN', 'cs.LG')",
                "default": None,
            },
            {
                "name": "days",
                "type": "int",
                "description": "Only return papers published within the last N days",
                "default": None,
            },
            {
                "name": "sort",
                "type": "str",
                "description": "Sort field: 'relevance', 'lastUpdatedDate', or 'submittedDate'",
                "default": "relevance",
            },
            {
                "name": "start",
                "type": "int",
                "description": "0-based result offset for pagination",
                "default": 0,
            },
        ],
    },
    {
        "name": "fetch_arxiv_by_ids",
        "description": "Fetch arXiv papers by their IDs. Returns full metadata for each paper.",
        "required_parameters": [
            {
                "name": "ids",
                "type": "str",
                "description": "Comma-separated arXiv paper IDs (e.g., '2301.00001,2301.00002')",
                "default": None,
            }
        ],
        "optional_parameters": [],
    },
    {
        "name": "search_biorxiv",
        "description": "Search bioRxiv preprints with local keyword and author filtering. "
        "Queries the bioRxiv API with date range or category filters, then applies "
        "local text matching on title/abstract/authors. Supports OR queries, quoted phrases, "
        "author name variant expansion, and automatic version deduplication.",
        "required_parameters": [],
        "optional_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": "Keyword query (supports OR between groups, quoted phrases)",
                "default": None,
            },
            {
                "name": "max_results",
                "type": "int",
                "description": "Maximum results to return",
                "default": 10,
            },
            {
                "name": "days",
                "type": "int",
                "description": "Search the most recent N days",
                "default": None,
            },
            {
                "name": "start_date",
                "type": "str",
                "description": "Start date in YYYY-MM-DD format (use with end_date)",
                "default": None,
            },
            {
                "name": "end_date",
                "type": "str",
                "description": "End date in YYYY-MM-DD format (use with start_date)",
                "default": None,
            },
            {
                "name": "category",
                "type": "str",
                "description": "bioRxiv subject category (e.g., 'genomics', 'bioinformatics')",
                "default": None,
            },
            {
                "name": "authors",
                "type": "str",
                "description": "Semicolon-separated author names for filtering",
                "default": None,
            },
            {
                "name": "doi",
                "type": "str",
                "description": "Fetch a specific bioRxiv DOI directly",
                "default": None,
            },
            {
                "name": "phrase",
                "type": "bool",
                "description": "Treat query as a single phrase",
                "default": False,
            },
            {
                "name": "scan_limit",
                "type": "int",
                "description": "Maximum API records to inspect locally",
                "default": 300,
            },
        ],
    },
    {
        "name": "assess_scientific_impact",
        "description": "Assess publication impact using OpenAlex citation data and Altmetric scores. "
        "Retrieves citation counts, citation percentiles, journal information, and "
        "social media / news attention. Set OPENALEX_MAILTO env var for polite-pool access. "
        "Set ALTMETRIC_API_KEY env var for Altmetric data.",
        "required_parameters": [],
        "optional_parameters": [
            {
                "name": "doi",
                "type": "str",
                "description": "DOI of the publication (provide doi or openalex_id)",
                "default": None,
            },
            {
                "name": "openalex_id",
                "type": "str",
                "description": "OpenAlex work ID (e.g., 'W2741809807')",
                "default": None,
            },
            {
                "name": "mailto",
                "type": "str",
                "description": "Email for OpenAlex polite-pool (or set OPENALEX_MAILTO env var)",
                "default": None,
            },
        ],
    },
    {
        "name": "validate_doi",
        "description": "Validate a DOI using the Crossref REST API and retrieve publication metadata "
        "including title, journal, year, and a formatted APA citation.",
        "required_parameters": [
            {
                "name": "doi",
                "type": "str",
                "description": "The DOI to validate (e.g., '10.1038/nature12373')",
                "default": None,
            }
        ],
        "optional_parameters": [],
    },
    {
        "name": "search_crossref_by_title",
        "description": "Search for publications by title using the Crossref REST API. "
        "Returns matching publications with DOI, title, journal, and year.",
        "required_parameters": [
            {
                "name": "title",
                "type": "str",
                "description": "Title or partial title to search for",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "max_results",
                "type": "int",
                "description": "Maximum number of results to return",
                "default": 5,
            }
        ],
    },
    {
        "name": "format_citation_from_doi",
        "description": "Format a citation for a DOI in a specified citation style "
        "(APA, Vancouver, AMA, IEEE, or Chicago).",
        "required_parameters": [
            {
                "name": "doi",
                "type": "str",
                "description": "The DOI to format a citation for",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "style",
                "type": "str",
                "description": "Citation style: 'apa', 'vancouver', 'ama', 'ieee', or 'chicago'",
                "default": "apa",
            }
        ],
    },
    {
        "name": "query_jgi_lakehouse",
        "description": "Execute a SQL query against the JGI Lakehouse (Dremio) to query biological "
        "databases including GOLD, IMG, Mycocosm, and Phytozome. Requires DREMIO_PAT "
        "environment variable and LBNL network access.",
        "required_parameters": [
            {
                "name": "sql",
                "type": "str",
                "description": "SQL query to execute",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "limit",
                "type": "int",
                "description": "Maximum rows to return",
                "default": 100,
            },
            {
                "name": "timeout",
                "type": "int",
                "description": "Maximum seconds to wait for query completion",
                "default": 300,
            },
        ],
    },
    {
        "name": "list_jgi_lakehouse_schemas",
        "description": "List all available schemas in the JGI Lakehouse (Dremio) including "
        "GOLD, IMG, Mycocosm, and Phytozome databases. Requires DREMIO_PAT env var.",
        "required_parameters": [],
        "optional_parameters": [],
    },
]
