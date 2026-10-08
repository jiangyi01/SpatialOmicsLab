"""Schema for the gene/variant/chemical identifier tools in ``tool/gene_identifiers.py``.

This list -- not the docstrings next door -- is what ``read_module2api()`` collects and
``utils/formatting.py:textify_api_dict`` renders into the function signatures the model sees. A
parameter absent from here is a knob the agent cannot turn; a parameter here that the function does
not accept is a ``TypeError`` on the first call. Keep both files in step.

Adapted from ToolUniverse -- https://github.com/mims-harvard/ToolUniverse -- at commit
``f075c2a75e8b35ae5dbb220d48d4e87e980388b1``, Apache-2.0, Copyright [2025] [ToolUniverse team].
CHANGED BY SPATIALOMICSGYM as Apache-2.0 section 4(b) requires: the upstream JSON tool catalogs were
re-expressed as this Python ``description`` list, tool names were renamed to our convention, the
config-driven ``operation`` discriminator was replaced by one function per operation, and upstream's
duplicate alias parameters were dropped. See ``VENDORING.md``.
"""

description = [
    {
        "description": (
            "Search MyGene.info for genes by symbol, name, keyword or identifier. Accepts free text, "
            "a gene symbol, an identifier, or fielded Lucene syntax. Use this to turn a name into an "
            "identifier. Print the returned dict."
        ),
        "name": "mygene_query_genes",
        "optional_parameters": [
            {
                "name": "species",
                "type": "str",
                "description": "Species name or NCBI taxonomy ID, e.g. 'human', 'mouse', 'rat', or '9606'",
                "default": "human",
            },
            {
                "name": "fields",
                "type": "str",
                "description": (
                    "Comma-separated annotation fields to return. Dotted paths select subfields, "
                    "e.g. 'ensembl.gene'. Use 'all' for the complete record"
                ),
                "default": "symbol,name,entrezgene,ensembl.gene,summary",
            },
            {
                "name": "size",
                "type": "int",
                "description": "Maximum number of hits to return; values above 100 are capped at 100",
                "default": 10,
            },
        ],
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": (
                    "Search string: free text ('kinase involved in apoptosis'), a symbol ('CDK2'), an "
                    "identifier ('ENSG00000123374'), or a fielded query ('symbol:BRCA* AND taxid:9606')"
                ),
                "default": None,
            }
        ],
    },
    {
        "description": (
            "Retrieve the complete MyGene.info annotation record for one gene by identifier, including "
            "GO terms, pathway memberships, InterPro domains and the RefSeq summary. Use "
            "mygene_query_genes first if you only have a name. Print the returned dict."
        ),
        "name": "mygene_get_gene_annotation",
        "optional_parameters": [
            {
                "name": "fields",
                "type": "str",
                "description": ("Comma-separated annotation fields. 'all' returns the complete record, which is large"),
                "default": "symbol,name,entrezgene,ensembl,summary,go,pathway,interpro",
            }
        ],
        "required_parameters": [
            {
                "name": "gene_id",
                "type": "str",
                "description": (
                    "Entrez Gene ID ('1017'), Ensembl gene ID ('ENSG00000123374'), or another identifier "
                    "MyGene indexes as a primary key"
                ),
                "default": None,
            }
        ],
    },
    {
        "description": (
            "Annotate many genes in one request. The tool for translating a whole var_names index or a "
            "marker-gene list: identifiers are matched against Entrez ID, Ensembl gene ID and symbol at "
            "once, so a mixed list works, and unmatched inputs come back flagged rather than dropped. Print the returned dict."
        ),
        "name": "mygene_batch_query",
        "optional_parameters": [
            {
                "name": "species",
                "type": "str",
                "description": "Species name or NCBI taxonomy ID, e.g. 'human', 'mouse', or '9606'",
                "default": "human",
            },
            {
                "name": "fields",
                "type": "str",
                "description": "Comma-separated annotation fields to return for each gene",
                "default": "symbol,name,entrezgene,ensembl.gene",
            },
        ],
        "required_parameters": [
            {
                "name": "gene_ids",
                # Both forms the prose offers. "list" alone made the MCP exposure refuse the
                # comma-separated string before the function ran (hunt 2026-09-30, uT3-atlases-21).
                "type": "str|list[str]",
                "description": (
                    "Gene identifiers to annotate -- symbols, Entrez IDs, Ensembl IDs, or a mix, e.g. "
                    "['CDK2', 'BRCA1', 'ENSG00000123374']. A comma-separated string is also accepted"
                ),
                "default": None,
            }
        ],
    },
    {
        "description": (
            "Search MyVariant.info for variants by rsID, gene, consequence or clinical significance. "
            "Every response names the reference assembly its coordinates are in, which MyVariant's own "
            "payload does not. Print the returned dict."
        ),
        "name": "myvariant_query_variants",
        "optional_parameters": [
            {
                "name": "fields",
                "type": "str",
                "description": "Comma-separated annotation fields to return",
                "default": "dbsnp.rsid,clinvar.rcv.clinical_significance,cadd.phred,gnomad_genome.af.af",
            },
            {
                "name": "size",
                "type": "int",
                "description": "Maximum number of hits to return; values above 100 are capped at 100",
                "default": 10,
            },
            {
                "name": "assembly",
                "type": "str",
                "description": (
                    "Reference assembly for the returned coordinates: 'hg19' (GRCh37) or 'hg38' (GRCh38). "
                    "Coordinates differ between the two by up to megabases, so set this to match the rest "
                    "of your analysis"
                ),
                "default": "hg19",
            },
        ],
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": (
                    "Search string: an rsID ('rs334'), an HGVS id, or a fielded query such as "
                    "'clinvar.gene.symbol:BRCA1 AND clinvar.rcv.clinical_significance:pathogenic'"
                ),
                "default": None,
            }
        ],
    },
    {
        "description": (
            "Retrieve the complete MyVariant.info annotation record for one variant by rsID or HGVS id. "
            "When MyVariant files several records under one rsID and resolves to an identifier-only stub, "
            "the sibling records that do carry the requested annotation are reported alongside it. Print the returned dict."
        ),
        "name": "myvariant_get_variant_annotation",
        "optional_parameters": [
            {
                "name": "fields",
                "type": "str",
                "description": "Comma-separated annotation fields. 'all' returns the complete record",
                "default": "dbsnp,clinvar,cadd,gnomad_genome,dbnsfp",
            },
            {
                "name": "assembly",
                "type": "str",
                "description": (
                    "Reference assembly the coordinates are expressed in: 'hg19' (GRCh37) or 'hg38' "
                    "(GRCh38). An HGVS id must already be written in this assembly -- MyVariant looks ids "
                    "up verbatim and never lifts coordinates over. An rsID resolves in either"
                ),
                "default": "hg19",
            },
        ],
        "required_parameters": [
            {
                "name": "variant_id",
                "type": "str",
                "description": "rsID ('rs334') or HGVS genomic id ('chr11:g.5248232C>A')",
                "default": None,
            }
        ],
    },
    {
        "description": (
            "Retrieve pathogenicity prediction scores for one variant in a single call -- REVEL, CADD, "
            "AlphaMissense, SIFT, PolyPhen-2, MetaRNN, GERP, PhyloP and PhastCons from dbNSFP, plus the "
            "ClinVar significance. Use this when a general variant query returns no dbnsfp data, or when "
            "you specifically need REVEL/AlphaMissense for an ACMG PP3/BP4 call. Print the returned dict."
        ),
        "name": "myvariant_get_pathogenicity_scores",
        "optional_parameters": [
            {
                "name": "fields",
                "type": "str",
                "description": (
                    "Comma-separated fields, pre-set to the pathogenicity-score list. Override only to narrow it"
                ),
                "default": (
                    "dbnsfp.revel.score,cadd.phred,dbnsfp.alphamissense.score,dbnsfp.alphamissense.pred,"
                    "dbnsfp.sift.score,dbnsfp.sift.pred,dbnsfp.polyphen2.hdiv.score,dbnsfp.polyphen2.hdiv.pred,"
                    "dbnsfp.metarnn.score,dbnsfp.metarnn.pred,cadd.gerp.rs,"
                    "dbnsfp.phylop.100way_vertebrate.rankscore,dbnsfp.phastcons.100way_vertebrate.rankscore,"
                    "dbnsfp.vest4.score,dbnsfp.mutationtaster.pred,clinvar.rcv.clinical_significance,dbsnp.rsid"
                ),
            },
            {
                "name": "assembly",
                "type": "str",
                "description": (
                    "Reference assembly the coordinates are expressed in: 'hg19' (GRCh37) or 'hg38' "
                    "(GRCh38). An HGVS id must already be written in this assembly; an rsID resolves in "
                    "either"
                ),
                "default": "hg19",
            },
        ],
        "required_parameters": [
            {
                "name": "variant_id",
                "type": "str",
                "description": "rsID ('rs45478192') or HGVS genomic id ('chr16:g.23635348A>C')",
                "default": None,
            }
        ],
    },
    {
        "description": (
            "Search MyChem.info for drugs and chemicals by name, InChIKey or database identifier, "
            "returning DrugBank, ChEBI, PubChem and ChEMBL cross-references. Print the returned dict."
        ),
        "name": "mychem_query_chemicals",
        "optional_parameters": [
            {
                "name": "fields",
                "type": "str",
                "description": "Comma-separated annotation fields to return",
                "default": "drugbank.name,drugbank.drug_interactions,chebi,pubchem.cid,chembl.molecule_chembl_id",
            },
            {
                "name": "size",
                "type": "int",
                "description": "Maximum number of hits to return; values above 100 are capped at 100",
                "default": 10,
            },
        ],
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": (
                    "Drug or chemical name ('imatinib'), an InChIKey, or a fielded query such as "
                    "'drugbank.name:aspirin'"
                ),
                "default": None,
            }
        ],
    },
    {
        "description": (
            "Retrieve the complete MyChem.info annotation record for one chemical by InChIKey or database "
            "identifier, covering DrugBank, ChEBI, PubChem, ChEMBL and DrugCentral. Print the returned dict."
        ),
        "name": "mychem_get_chemical_annotation",
        "optional_parameters": [
            {
                "name": "fields",
                "type": "str",
                "description": "Comma-separated annotation fields. 'all' returns the complete record",
                "default": "drugbank,chebi,pubchem,chembl,drugcentral",
            }
        ],
        "required_parameters": [
            {
                "name": "chem_id",
                "type": "str",
                "description": (
                    "InChIKey (recommended, e.g. 'KTUFNOKKBVMGRW-UHFFFAOYSA-N'), DrugBank accession, "
                    "ChEMBL ID, or PubChem CID"
                ),
                "default": None,
            }
        ],
    },
    {
        "description": (
            "Convert identifiers between namespaces in bulk using TogoID, which covers 100+ biological "
            "identifier types. Source and target must be directly related in TogoID's graph, so convert "
            "through an intermediate in two calls if an unrelated pair is reported as having no route. "
            "Call togoid_list_datasets first if you are unsure of the exact dataset keys. Print the returned dict."
        ),
        "name": "togoid_convert",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "ids",
                # Both forms the prose offers; "str" alone made the MCP exposure refuse a list
                # (hunt 2026-09-30, uT3-atlases-21).
                "type": "str|list[str]",
                "description": (
                    "Identifiers to convert: a comma-separated string, or a list, e.g. 'ENSG00000012048' "
                    "or ['P38398', 'P04637']"
                ),
                "default": None,
            },
            {
                "name": "source",
                "type": "str",
                "description": "Source dataset key, e.g. 'ensembl_gene', 'hgnc', 'uniprot', 'ncbigene'",
                "default": None,
            },
            {
                "name": "target",
                "type": "str",
                "description": "Target dataset key, e.g. 'ncbigene', 'uniprot', 'pdb', 'chebi'",
                "default": None,
            },
        ],
    },
    {
        "description": (
            "List the identifier namespaces TogoID can convert between, with their exact dataset keys, "
            "labels and categories. Call this before togoid_convert when you are not certain of a key -- "
            "the keys are exact strings ('ensembl_gene', not 'Ensembl') and a wrong one is a failed "
            "conversion. Print the returned dict."
        ),
        "name": "togoid_list_datasets",
        "optional_parameters": [
            {
                "name": "category",
                "type": "str",
                "description": (
                    "Restrict to one category, case-insensitive, e.g. 'gene', 'protein', 'chemical', "
                    "'structure'. Omit to list every dataset"
                ),
                "default": None,
            }
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Get every cross-reference BridgeDb holds for one identifier. BridgeDb's coverage extends "
            "past genes to metabolites and lipids (HMDB, ChEBI, KEGG Compound, SwissLipids), which is "
            "where TogoID's coverage thins out. Print the returned dict."
        ),
        "name": "bridgedb_xrefs",
        "optional_parameters": [
            {
                "name": "organism",
                "type": "str",
                "description": "Species, e.g. 'Human', 'Mouse', 'Rat'",
                "default": "Human",
            },
            {
                "name": "target_source",
                "type": "str",
                "description": (
                    "Restrict the answer to one target database, by name ('ChEBI') or BridgeDb system "
                    "code ('Ce'). Omit to get every cross-reference"
                ),
                "default": None,
            },
        ],
        "required_parameters": [
            {
                "name": "identifier",
                "type": "str",
                "description": "The identifier to look up, e.g. 'ENSG00000012048' or 'HMDB0000122'",
                "default": None,
            },
            {
                "name": "source",
                "type": "str",
                "description": (
                    "The database the identifier comes from, by name ('Ensembl', 'HMDB', 'UniProt', "
                    "'HGNC', 'ChEBI') or BridgeDb system code ('En', 'Ch', 'S', 'H', 'Ce')"
                ),
                "default": None,
            },
        ],
    },
    {
        "description": (
            "Search BridgeDb for identifiers matching a name or symbol across every database it indexes. "
            "Use this when you have a label and do not know which namespace it belongs to -- each hit "
            "reports the database it came from. Print the returned dict."
        ),
        "name": "bridgedb_search",
        "optional_parameters": [
            {
                "name": "organism",
                "type": "str",
                "description": "Species, e.g. 'Human', 'Mouse', 'Rat'",
                "default": "Human",
            }
        ],
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": "Free-text name or symbol to search for, e.g. 'BRCA1' or 'glucose'",
                "default": None,
            }
        ],
    },
    {
        "description": (
            "Get the stored properties of one identifier from BridgeDb -- symbol, full name, synonyms, "
            "chromosome. Complements bridgedb_xrefs: that answers what else this is called elsewhere, "
            "this answers what the database says about it. Print the returned dict."
        ),
        "name": "bridgedb_attributes",
        "optional_parameters": [
            {
                "name": "organism",
                "type": "str",
                "description": "Species, e.g. 'Human', 'Mouse', 'Rat'",
                "default": "Human",
            }
        ],
        "required_parameters": [
            {
                "name": "identifier",
                "type": "str",
                "description": "The identifier to describe, e.g. 'ENSG00000012048' or 'HMDB0000122'",
                "default": None,
            },
            {
                "name": "source",
                "type": "str",
                "description": (
                    "The database the identifier comes from, by name ('Ensembl', 'HGNC', 'HMDB') or "
                    "BridgeDb system code ('En', 'H', 'Ch')"
                ),
                "default": None,
            },
        ],
    },
    {
        "description": (
            "Fetch the authoritative HGNC record for a human gene symbol. The one lookup that tells a "
            "renamed gene apart from a non-existent one: a symbol that is no longer current is resolved "
            "through its previous-symbol and alias-symbol entries and the substitution is reported, so a "
            "marker list from an older paper still resolves and you are told that it did. Print the returned dict."
        ),
        "name": "hgnc_fetch_gene_by_symbol",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "symbol",
                "type": "str",
                "description": "Gene symbol, e.g. 'BRCA1', 'GFAP', or a retired one such as 'CYP2E'",
                "default": None,
            }
        ],
    },
    {
        "description": "Fetch the authoritative HGNC record for a human gene by its HGNC ID. Print the returned dict.",
        "name": "hgnc_fetch_gene_by_id",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "hgnc_id",
                "type": "str",
                "description": "HGNC ID, with or without the prefix -- 'HGNC:1100' and '1100' both work",
                "default": None,
            }
        ],
    },
    {
        "description": (
            "Search HGNC for human genes by symbol, name, alias or any indexed field. Wildcards work "
            "('BRCA*'), which makes this the tool for enumerating a gene family by naming convention. "
            "Results are stubs; pass a hit's symbol to hgnc_fetch_gene_by_symbol for the full record. Print the returned dict."
        ),
        "name": "hgnc_search_genes",
        "optional_parameters": [
            {
                "name": "search_field",
                "type": "str",
                "description": (
                    "Restrict the search to one field, e.g. 'symbol', 'name', 'alias_symbol', "
                    "'prev_symbol'. Omit to search symbol and name together"
                ),
                "default": None,
            }
        ],
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": "Search term; wildcards allowed, e.g. 'BRCA*', 'collagen', 'CD8A'",
                "default": None,
            }
        ],
    },
    {
        "description": "Find every HGNC gene mapped to a cytogenetic band. Print the returned dict.",
        "name": "hgnc_search_by_location",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "location",
                "type": "str",
                "description": (
                    "Cytogenetic band, e.g. '17p13.1' or 'Xq28'. A broader band ('17p13') matches more genes"
                ),
                "default": None,
            }
        ],
    },
    {
        "description": (
            "List every gene HGNC assigns to one curated gene family/group, such as the solute carriers "
            "or the collagens. The gene_group_id of any gene appears in the record returned by "
            "hgnc_fetch_gene_by_symbol, which is how you find the ID to pass here. Print the returned dict."
        ),
        "name": "hgnc_fetch_gene_family_members",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "gene_group_id",
                "type": "str",
                "description": (
                    "HGNC gene group ID, e.g. '2155' (solute carrier family 2) or '2247' (fibrillar collagens)"
                ),
                "default": None,
            }
        ],
    },
]
