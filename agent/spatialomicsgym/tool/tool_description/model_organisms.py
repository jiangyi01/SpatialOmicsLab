"""Schema for the model-organism tools in ``tool/model_organisms.py``.

``read_module2api()`` collects this list -- not the docstrings next door -- and
``utils/formatting.py:textify_api_dict`` renders it into the function signatures the model sees. A
parameter absent from here is a knob the agent cannot turn; a parameter here that the function does
not accept is a ``TypeError`` on the first call. Keep both files in step.

One service, eight databases. The Alliance of Genome Resources is the joint portal of MGI (mouse),
RGD (rat), ZFIN (zebrafish), FlyBase, WormBase, SGD (yeast), Xenbase and human HGNC, and it answers
the questions a spatial or bulk expression result raises next: what is already known about this
gene, what happens when it is broken, what is its counterpart in the species I actually work in,
and which genes are curated as associated with this disease.

**Everything here is keyed on a prefixed identifier**, and the single most common way to get a
confidently wrong answer rather than an error is to bring the wrong prefix. ``620474`` is RGD's
*Sox9* and ``MGI:620474`` is a different gene entirely, so no prefix is ever inferred from a bare
number -- an unprefixed id is an error naming the eight prefixes.
``model_organisms_list_members`` shows them with a worked example each, and
``model_organisms_search_genes`` turns a symbol into an id. Start at one of those two.

Four traps are written into the parameter prose below because each returns a confident wrong or
empty answer rather than an error. A nonexistent gene answers the sub-endpoints with HTTP 200 and
``total: 0``, so ids are resolved before use. ``stringency="moderate"`` is an advertised value the
server answers with zero rows for every gene tested, so it is served client-side from the row
flags. ``/gene/{id}/alleles`` interleaves nothing but returns imported dbSNP variants *ahead* of
curated alleles with no filter to separate them, so ``include="alleles"`` reads the tail of the
list. And a disease-gene row's relation may be a curated *negative* -- ``is_not_implicated_in`` --
which the field most callers reach for silently reports as its opposite.

Adapted from ToolUniverse -- https://github.com/mims-harvard/ToolUniverse -- at commit
``f075c2a75e8b35ae5dbb220d48d4e87e980388b1``, Apache-2.0, Copyright [2025] [ToolUniverse team].
CHANGED BY SPATIALOMICSGYM as Apache-2.0 section 4(b) requires: the upstream JSON tool catalog
(``alliance_genome_tools.json``, 8 tools) was re-expressed as this Python ``description`` list, and
``model_organisms_list_members`` is ours, not upstream's -- it exists because the parameter most
likely to produce a silent wrong answer, the id prefix, had no discovery call. The implementation
module's docstring lists the upstream defects fixed here (F41-F48) with the measurement behind each.
See ``VENDORING.md``.
"""

description = [
    {
        "description": (
            "List the eight member databases of the Alliance of Genome Resources -- human, mouse, "
            "rat, zebrafish, fruit fly, roundworm, budding yeast and frog -- with the identifier "
            "prefix each one uses and a worked example gene for it. Start here before any other "
            "model_organisms call, because every one of them is keyed on a prefixed id and a bare "
            "number is rejected rather than guessed: '620474' is rat Sox9 under RGD and means "
            "something else entirely under MGI. By default each member's example gene is "
            "re-resolved live, so the result also says which databases are answering right now. "
            "Print the returned dict."
        ),
        "name": "model_organisms_list_members",
        "optional_parameters": [
            {
                "name": "verify",
                "type": "boolean",
                "description": (
                    "Re-resolve every member's example gene against the live service and report "
                    "what came back. True by default and costs about 2-3 s for the eight calls; "
                    "pass False for the offline table alone"
                ),
                "default": True,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Search all eight Alliance member databases by gene symbol, name or synonym and return "
            "the prefixed identifier every other model_organisms call needs. A symbol usually "
            "matches once per species, so 'pax6' comes back as the human, mouse, rat, zebrafish "
            "and frog genes together, each with its own id. Alliance matches symbols and synonyms "
            "rather than descriptive prose -- 'insulin' returns diseases and GO terms but no gene, "
            "while the symbol 'INS' returns the genes -- and when nothing matched but other entity "
            "types did, the result says which rather than looking like an empty database. "
            "Print the returned dict."
        ),
        "name": "model_organisms_search_genes",
        "optional_parameters": [
            {
                "name": "limit",
                "type": "integer",
                "description": (
                    "Genes to return, 1-50. Above 50 the search endpoint times out under load. "
                    "data.total_matched reports how many matched in total, which is usually far "
                    "larger"
                ),
                "default": 10,
            },
            {
                "name": "offset",
                "type": "integer",
                "description": "Rows to skip, for paging through a large match set",
                "default": 0,
            },
            {
                "name": "species",
                "type": "string",
                "description": (
                    "Keep only genes from one organism, matched against either the binomial or the "
                    "common name -- 'Mus musculus' or 'mouse' both work. The filter is applied "
                    "after the fetch, so raise limit when filtering hard"
                ),
                "default": None,
            },
        ],
        "required_parameters": [
            {
                "name": "query",
                "type": "string",
                "description": ("Gene symbol, name fragment or synonym, e.g. 'pax6', 'TP53', 'unc-54'"),
                "required": True,
            },
        ],
    },
    {
        "description": (
            "Fetch the full curated record for one gene: symbol, full name, species, gene type, "
            "synonyms, genomic location and cross-references into the member databases. The record "
            "also names the data provider that curated it, which is not always the one the prefix "
            "suggests. Takes a prefixed Alliance id, never a bare number and never a plain symbol "
            "-- use model_organisms_search_genes to turn a symbol into an id first. An id that "
            "does not exist comes back as an error carrying the service's own explanation, not an "
            "empty record. Print the returned dict."
        ),
        "name": "model_organisms_get_gene",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "gene_id",
                "type": "string",
                "description": (
                    "Prefixed Alliance gene id, e.g. 'MGI:97490', 'HGNC:11998', "
                    "'WB:WBGene00006789'. One of the eight prefixes is required; a bare number is "
                    "rejected rather than guessed"
                ),
                "required": True,
            },
        ],
    },
    {
        "description": (
            "Return the curated phenotype annotations for one gene, each with the PubMed "
            "identifiers supporting it. These are hand-curated from the literature by the member "
            "database that owns the organism, so an annotation means a curator read the paper and "
            "decided the phenotype was demonstrated -- which is what makes them worth more than a "
            "text search. Mouse Pax6 alone carries 184, so raise limit or page through rather than "
            "reading the default 20 as the whole picture; data.total_annotations gives the count "
            "across all pages. Print the returned dict."
        ),
        "name": "model_organisms_get_gene_phenotypes",
        "optional_parameters": [
            {
                "name": "limit",
                "type": "integer",
                "description": "Annotations to return, 1-100",
                "default": 20,
            },
            {
                "name": "page",
                "type": "integer",
                "description": "1-based page number. Page 0 is rejected by the server",
                "default": 1,
            },
        ],
        "required_parameters": [
            {
                "name": "gene_id",
                "type": "string",
                "description": "Prefixed Alliance gene id, e.g. 'MGI:97490'",
                "required": True,
            },
        ],
    },
    {
        "description": (
            "Find the orthologues of a gene in the other Alliance organisms, and its paralogues "
            "within its own, with the prediction methods that support each call. This is the "
            "function that crosses species: Alliance runs a dozen orthology predictors -- Ensembl "
            "Compara, PANTHER, OrthoFinder, InParanoid and others -- and reports how many agreed, "
            "which is what the stringency filter selects on. Use it to carry a finding in one "
            "organism over to the species actually being studied. Print the returned dict."
        ),
        "name": "model_organisms_get_gene_orthologs",
        "optional_parameters": [
            {
                "name": "stringency",
                "type": "string",
                "description": (
                    "'stringent' for best-score-both-ways calls with broad method agreement, "
                    "'all' to include single-method low-confidence calls, or 'moderate' for the "
                    "middle band. Measured on mouse Pax6: 7 stringent, 3 moderate, 19 all. "
                    "'moderate' is served client-side because the server answers that value with "
                    "zero rows for every gene tested, so asking for it directly is "
                    "indistinguishable from a typo"
                ),
                "default": "stringent",
            },
            {
                "name": "limit",
                "type": "integer",
                "description": ("Orthologues to return, 1-100. Paralogues are capped at the same number"),
                "default": 20,
            },
        ],
        "required_parameters": [
            {
                "name": "gene_id",
                "type": "string",
                "description": "Prefixed Alliance gene id, e.g. 'MGI:97490'",
                "required": True,
            },
        ],
    },
    {
        "description": (
            "Return the molecular (physical) interaction partners of one gene, each with the "
            "method by which the interaction was detected and the database it came from. "
            "Genetic interactions (epistasis, synthetic lethality) are not queried, so an empty "
            "or short list says nothing about them. "
            "Interactions are aggregated from BioGRID and IMEx and carry MI-ontology terms for "
            "detection -- 'pull down', 'two hybrid', 'affinity chromatography' -- which is what "
            "separates a directly demonstrated interaction from a high-throughput screen hit, so "
            "read detection_method before treating a partner as real. Print the returned dict."
        ),
        "name": "model_organisms_get_molecular_interactions",
        "optional_parameters": [
            {
                "name": "limit",
                "type": "integer",
                "description": "Interactions to return, 1-100",
                "default": 20,
            },
            {
                "name": "page",
                "type": "integer",
                "description": "1-based page number",
                "default": 1,
            },
        ],
        "required_parameters": [
            {
                "name": "gene_id",
                "type": "string",
                "description": "Prefixed Alliance gene id, e.g. 'MGI:97490'",
                "required": True,
            },
        ],
    },
    {
        "description": (
            "Return the alleles of a gene and the mutant strains carrying them -- the knockouts, "
            "knock-ins, transgenes and fish lines a member database maintains, each with the "
            "phenotypes seen in it and the diseases it is curated as a model of. The allele "
            "endpoint returns two different things in one stream: hand-curated named alleles such "
            "as Pax6<Gt(OST128284)Lex>, and imported dbSNP variants that are nothing but an 'rs' "
            "accession. The variants come first and usually outnumber the alleles ten to one -- "
            "mouse Pax6 has 607 variants ahead of 68 alleles, human TP53 has 2,920 variants and no "
            "curated alleles at all -- and no server-side filter separates them, so a plain page "
            "of results is not what a question about alleles wants. include='alleles' reads the "
            "tail of the list where the curated ones sit and reports the split it found. "
            "Print the returned dict."
        ),
        "name": "model_organisms_get_alleles_and_models",
        "optional_parameters": [
            {
                "name": "include",
                "type": "string",
                "description": (
                    "'alleles' for curated named alleles only, which costs one extra request to "
                    "locate the tail of the list; 'variants' for dbSNP variants only, paged from "
                    "the front; 'all' for both in the server's own order"
                ),
                "default": "alleles",
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Alleles and models each returned, 1-100",
                "default": 20,
            },
        ],
        "required_parameters": [
            {
                "name": "gene_id",
                "type": "string",
                "description": "Prefixed Alliance gene id, e.g. 'MGI:97490'",
                "required": True,
            },
        ],
    },
    {
        "description": (
            "Look up a Disease Ontology term: its name, its definition, its synonyms, and its "
            "parents and children in the hierarchy. Use it to check that a DOID means what it is "
            "assumed to mean before asking for its genes, and to move around the ontology -- "
            "parents generalise the query, children narrow it, and descendant_count says how much "
            "of the tree sits underneath, which is also how many extra terms "
            "model_organisms_get_disease_genes will fold into its answer. Print the returned dict."
        ),
        "name": "model_organisms_get_disease",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "disease_id",
                "type": "string",
                "description": ("Disease Ontology identifier, e.g. 'DOID:9351' for diabetes mellitus"),
                "required": True,
            },
        ],
    },
    {
        "description": (
            "Return the genes curated as associated with a disease, across every Alliance "
            "organism, with the evidence codes and publications behind each. Two things decide "
            "whether the answer means anything and both are surfaced in the result. The relation "
            "is the finding: a row is is_implicated_in, is_marker_for, or one of the negated forms "
            "is_not_implicated_in and is_not_marker_for -- curated statements that a published "
            "association did not hold, which are 33 of the first 500 human rows for diabetes "
            "mellitus. Read relation and negated; a flat 'disease gene list' turns those rows into "
            "their opposite. And the answer spans the ontology subtree, not just the term asked "
            "for -- only 19 of the first 200 rows for diabetes mellitus are annotated to diabetes "
            "mellitus itself, the rest to type-2, type-1 and ten further descendants -- so read "
            "data.by_disease_term before treating the list as one disease. Print the returned dict."
        ),
        "name": "model_organisms_get_disease_genes",
        "optional_parameters": [
            {
                "name": "limit",
                "type": "integer",
                "description": (
                    "Annotations to return, 1-100. DOID:9351 has 12,740 in total, so this is a "
                    "sample unless the disease is narrow"
                ),
                "default": 20,
            },
            {
                "name": "page",
                "type": "integer",
                "description": "1-based page number",
                "default": 1,
            },
            {
                "name": "species",
                "type": "string",
                "description": (
                    "Restrict to one organism by binomial name, e.g. 'Mus musculus'. Filtered "
                    "server-side. Note that a row may carry via_orthology true, meaning it was not "
                    "curated in that organism at all but inferred from a human annotation through "
                    "an orthology call -- 409 of the first 500 mouse rows for DOID:9351"
                ),
                "default": None,
            },
        ],
        "required_parameters": [
            {
                "name": "disease_id",
                "type": "string",
                "description": "Disease Ontology identifier, e.g. 'DOID:9351'",
                "required": True,
            },
        ],
    },
]
