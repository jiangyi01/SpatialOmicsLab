"""Schema for the ontology lookup tools in ``tool/ontology.py``.

This list -- not the docstrings next door -- is what ``read_module2api()`` collects and
``utils/formatting.py:textify_api_dict`` renders into the function signatures the model sees. A
parameter absent from here is a knob the agent cannot turn; a parameter here that the function does
not accept is a ``TypeError`` on the first call. Keep both files in step.

Adapted from ToolUniverse -- https://github.com/mims-harvard/ToolUniverse -- at commit
``f075c2a75e8b35ae5dbb220d48d4e87e980388b1``, Apache-2.0, Copyright [2025] [ToolUniverse team].
CHANGED BY SPATIALOMICSGYM as Apache-2.0 section 4(b) requires: the upstream JSON tool catalogs
(``ols_tools.json``, ``efo_tools.json``) were re-expressed as this Python ``description`` list; the
config-driven ``operation`` discriminator was replaced by one function per operation; the five
EFO-scoped entries that differ from the generic tools only by a pre-filled ``ontology="efo"`` were
folded into those tools as a default argument rather than registered as separate names; and every
example identifier was re-verified live, because upstream's name terms EFO has since retired. See
``VENDORING.md``.
"""

description = [
    {
        "description": (
            "Search the EMBL-EBI Ontology Lookup Service (OLS4) for ontology terms by name, synonym "
            "or identifier, across 280+ ontologies or within one. This is how you turn a label from "
            "a marker table, sample sheet or paper into a stable identifier. Set exact_match=True "
            "when you want only terms whose own label or synonym equals the query -- without it a "
            "search also matches terms that merely mention the word in their definition. Print the "
            "returned dict."
        ),
        "name": "ols_search_terms",
        "optional_parameters": [
            {
                "name": "ontology",
                "type": "str",
                "description": (
                    "Restrict to one ontology by its lower-case OLS id: 'cl' (cell types), 'uberon' "
                    "(anatomy), 'hp' (phenotypes), 'mondo' (disease), 'go', 'efo', 'chebi'. Omit to "
                    "search every ontology"
                ),
                "default": None,
            },
            {
                "name": "rows",
                "type": "int",
                "description": "Number of hits to return, 1-200",
                "default": 10,
            },
            {
                "name": "exact_match",
                "type": "bool",
                "description": (
                    "Require the query to equal a term's label or exact synonym in full rather than "
                    "appear anywhere in its record. If the query looks like an identifier the exact "
                    "match is applied to the identifier fields instead"
                ),
                "default": False,
            },
            {
                "name": "obsolete_only",
                "type": "bool",
                "description": (
                    "Search the terms OLS has retired INSTEAD OF the current ones -- the service "
                    "returns one set or the other, never both. Leave False for ordinary lookups; "
                    "turn it on only to chase an identifier from an older paper that no longer "
                    "resolves"
                ),
                "default": False,
            },
        ],
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": (
                    "What to search for: a name ('T cell', 'hepatocyte'), an identifier "
                    "('CL:0000084') or a full IRI"
                ),
                "default": None,
            }
        ],
    },
    {
        "description": (
            "Fetch the full OLS4 record for one ontology term: label, definition, synonyms, whether "
            "it has children, and whether it is obsolete. An obsolete term reports the identifier "
            "that replaced it, which is the fast way to repair a stale identifier -- EFO retired its "
            "whole disease branch to MONDO, so disease identifiers from older papers are often dead "
            "with a live replacement recorded. Print the returned dict."
        ),
        "name": "ols_get_term_info",
        "optional_parameters": [
            {
                "name": "ontology",
                "type": "str",
                "description": (
                    "Lower-case OLS ontology id to look in. Inferred from the identifier's prefix "
                    "when omitted, which is correct almost always"
                ),
                "default": None,
            }
        ],
        "required_parameters": [
            {
                "name": "term_id",
                "type": "str",
                "description": (
                    "The term, as a CURIE ('HP:0001903', 'MONDO:0005148'), the OBO underscore form "
                    "('HP_0001903'), or a full IRI"
                ),
                "default": None,
            }
        ],
    },
    {
        "description": (
            "List the direct children (immediate subclasses) of an ontology term -- one level down "
            "only. Use ols_get_term_descendants for the whole subtree. An empty result is a real "
            "answer: it means the term is a leaf. Print the returned dict."
        ),
        "name": "ols_get_term_children",
        "optional_parameters": [
            {
                "name": "ontology",
                "type": "str",
                "description": (
                    "Lower-case OLS ontology id. Inferred from the identifier's prefix when omitted"
                ),
                "default": None,
            },
            {
                "name": "size",
                "type": "int",
                "description": "Number of children to return, 1-200",
                "default": 20,
            },
            {
                "name": "include_obsolete",
                "type": "bool",
                "description": "Include children OLS has retired",
                "default": False,
            },
        ],
        "required_parameters": [
            {
                "name": "term_id",
                "type": "str",
                "description": (
                    "The parent term, as a CURIE ('CL:0000084'), the underscore form, or a full IRI"
                ),
                "default": None,
            }
        ],
    },
    {
        "description": (
            "List the ancestors of an ontology term -- every class above it, up to the ontology "
            "root, broadest first. This is how you place a term: whether a cell type sits under "
            "'lymphocyte' or under 'myeloid leukocyte' is an ancestor question, and it is the check "
            "that catches a mis-mapped label before it propagates through an annotation. Print the "
            "returned dict."
        ),
        "name": "ols_get_term_ancestors",
        "optional_parameters": [
            {
                "name": "ontology",
                "type": "str",
                "description": (
                    "Lower-case OLS ontology id. Inferred from the identifier's prefix when omitted"
                ),
                "default": None,
            },
            {
                "name": "size",
                "type": "int",
                "description": "Number of ancestors to return, 1-200",
                "default": 20,
            },
            {
                "name": "include_obsolete",
                "type": "bool",
                "description": "Include ancestors OLS has retired",
                "default": False,
            },
        ],
        "required_parameters": [
            {
                "name": "term_id",
                "type": "str",
                "description": (
                    "The term, as a CURIE ('HP:0001903'), the underscore form, or a full IRI"
                ),
                "default": None,
            }
        ],
    },
    {
        "description": (
            "List the full descendant subtree of an ontology term, not just its direct children. The "
            "difference matters: 'T cell' has a handful of direct children and 171 descendants, and "
            "'is this label anywhere under T cell' is the second number. The reported total is the "
            "size of the whole subtree regardless of how many entries are returned. Print the "
            "returned dict."
        ),
        "name": "ols_get_term_descendants",
        "optional_parameters": [
            {
                "name": "ontology",
                "type": "str",
                "description": (
                    "Lower-case OLS ontology id. Inferred from the identifier's prefix when omitted"
                ),
                "default": None,
            },
            {
                "name": "size",
                "type": "int",
                "description": "Number of descendants to return, 1-200",
                "default": 20,
            },
        ],
        "required_parameters": [
            {
                "name": "term_id",
                "type": "str",
                "description": (
                    "The term, as a CURIE ('CL:0000084'), the underscore form, or a full IRI"
                ),
                "default": None,
            }
        ],
    },
    {
        "description": (
            "Map an ontology term onto its equivalents in other vocabularies -- DOID, ICD-10, OMIM, "
            "UMLS, MeSH, NCIt, SNOMED, MedDRA. This is the identifier-translation tool for "
            "controlled vocabularies, the counterpart to TogoID/BridgeDb for genes, and what you "
            "need when a cohort is coded in one vocabulary and an atlas in another. An empty list is "
            "a real answer -- HP and GO largely do not record cross-references. Print the returned "
            "dict."
        ),
        "name": "ols_get_term_xrefs",
        "optional_parameters": [
            {
                "name": "ontology",
                "type": "str",
                "description": (
                    "Lower-case OLS ontology id. Inferred from the identifier's prefix when omitted"
                ),
                "default": None,
            }
        ],
        "required_parameters": [
            {
                "name": "term_id",
                "type": "str",
                "description": (
                    "The term to translate, as a CURIE ('MONDO:0005148', 'UBERON:0002107'), the "
                    "underscore form, or a full IRI"
                ),
                "default": None,
            }
        ],
    },
    {
        "description": (
            "Find terms in the same ontology whose names are close to a given term's name. Useful "
            "when a label almost matches: you have 'CD8 T cell' from a marker table, the ontology "
            "calls it something else, and you want the neighbourhood to choose from. This is lexical "
            "similarity over labels and synonyms, not semantic similarity -- OLS4 exposes no "
            "embedding endpoint, and the returned payload says so. Print the returned dict."
        ),
        "name": "ols_find_similar_terms",
        "optional_parameters": [
            {
                "name": "size",
                "type": "int",
                "description": "Number of neighbouring terms to return, 1-200",
                "default": 10,
            }
        ],
        "required_parameters": [
            {
                "name": "term_id",
                "type": "str",
                "description": (
                    "The reference term, as a CURIE ('CL:0000084'), the underscore form, or a full IRI"
                ),
                "default": None,
            },
            {
                "name": "ontology",
                "type": "str",
                "description": (
                    "Lower-case OLS ontology id to search within, for example 'cl', 'uberon' or 'efo'"
                ),
                "default": None,
            },
        ],
    },
    {
        "description": (
            "Fetch metadata for one ontology: title, description, version, homepage and how many "
            "classes it contains. Worth a call before trusting a hierarchy result, because 'how many "
            "classes does this ontology actually contain' and 'when was it last loaded' decide "
            "whether an empty answer means absence or staleness. Print the returned dict."
        ),
        "name": "ols_get_ontology_info",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "ontology_id",
                "type": "str",
                "description": (
                    "The lower-case OLS ontology abbreviation: 'efo', 'mondo', 'hp', 'go', 'cl', "
                    "'uberon', 'chebi'"
                ),
                "default": None,
            }
        ],
    },
    {
        "description": (
            "List or search the 280+ ontologies OLS4 serves. Call this when you do not know which "
            "vocabulary owns a concept -- the right one is often not the obvious one, for instance "
            "anatomy is UBERON rather than anything named 'anatomy'. Print the returned dict."
        ),
        "name": "ols_search_ontologies",
        "optional_parameters": [
            {
                "name": "search",
                "type": "str",
                "description": (
                    "Free-text filter over ontology titles and descriptions, for example 'disease' "
                    "or 'cell'. Omit to list everything page by page"
                ),
                "default": None,
            },
            {
                "name": "page",
                "type": "int",
                "description": "Zero-based page number",
                "default": 0,
            },
            {
                "name": "size",
                "type": "int",
                "description": "Ontologies per page, 1-200",
                "default": 20,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Look up the ontology identifier for a disease named in plain English, scoped to EFO -- "
            "the vocabulary GWAS Catalog, Open Targets and ArrayExpress annotate with, which imports "
            "MONDO for its disease branch. Candidates are fetched several deep and any whose own "
            "label or exact synonym equals the query is promoted first, because OLS ranks on the "
            "whole indexed record and puts the wrong term first for short names and abbreviations. "
            "Always returns a list, even for rows=1. Print the returned dict."
        ),
        "name": "efo_id_for_disease_name",
        "optional_parameters": [
            {
                "name": "rows",
                "type": "int",
                "description": (
                    "Number of candidate terms to return, 1-200. The default of 1 returns the single "
                    "best match"
                ),
                "default": 1,
            }
        ],
        "required_parameters": [
            {
                "name": "disease",
                "type": "str",
                "description": (
                    "The disease name or abbreviation, for example 'asthma', 'PCOS', or 'type 2 "
                    "diabetes'"
                ),
                "default": None,
            }
        ],
    },
]
