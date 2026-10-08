"""MCP tool knowledge base for multi-omics analysis.

Note: These are internal SpatialOmicsLab tools, not MCP servers. They provide literature
search, DOI validation, impact assessment, and data access capabilities.
"""

from __future__ import annotations

from typing import Any

OMICS_MCP_TOOLS: dict[str, dict[str, Any]] = {
    "arxiv_search": {
        "task": "literature_search",
        "mcp_function": "search_arxiv_advanced",
        "full_name": "arXiv Advanced Search",
        "description": "Search arXiv with category, date range, and pagination filtering for scientific preprints.",
        "gpu": False,
        "priority": 1,
    },
    "arxiv_fetch": {
        "task": "literature_search",
        "mcp_function": "fetch_arxiv_by_ids",
        "full_name": "arXiv Fetch by IDs",
        "description": "Fetch full metadata and abstracts for specific arXiv papers by their IDs.",
        "gpu": False,
        "priority": 2,
    },
    "biorxiv_search": {
        "task": "literature_search",
        "mcp_function": "search_biorxiv",
        "full_name": "bioRxiv Search",
        "description": "Search bioRxiv preprints with keyword, author, and date filtering for biology research.",
        "gpu": False,
        "priority": 1,
    },
    "doi_validator": {
        "task": "literature_validation",
        "mcp_function": "validate_doi",
        "full_name": "DOI Validator",
        "description": "Validate DOIs and retrieve bibliographic metadata via CrossRef for reference verification.",
        "gpu": False,
        "priority": 1,
    },
    "crossref_search": {
        "task": "literature_search",
        "mcp_function": "search_crossref_by_title",
        "full_name": "CrossRef Title Search",
        "description": "Search CrossRef database by paper title to retrieve DOIs and bibliographic metadata.",
        "gpu": False,
        "priority": 2,
    },
    "impact_assessment": {
        "task": "literature_analysis",
        "mcp_function": "assess_scientific_impact",
        "full_name": "Scientific Impact Assessment",
        "description": "Assess the scientific impact and novelty of research findings relative to existing literature.",
        "gpu": False,
        "priority": 2,
    },
    "jgi_lakehouse": {
        "task": "data_access",
        "mcp_function": "query_jgi_lakehouse",
        "full_name": "JGI Lakehouse Query",
        "description": "Query the JGI data lakehouse for genomics and metagenomics datasets and project metadata.",
        "gpu": False,
        "priority": 2,
    },
}
