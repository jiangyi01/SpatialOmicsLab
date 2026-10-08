"""Schema for the expression atlas tools in ``tool/expression_atlases.py``.

This list -- not the docstrings next door -- is what ``read_module2api()`` collects and
``utils/formatting.py:textify_api_dict`` renders into the function signatures the model sees. A
parameter absent from here is a knob the agent cannot turn; a parameter here that the function does
not accept is a ``TypeError`` on the first call. Keep both files in step.

Three services, and choosing between them is most of the skill. **HPA** (Human Protein Atlas) is
the widest: bulk tissue, blood, brain, single cell, cell line and immune RNA panels plus protein
staining, subcellular location and cancer prognostics, all keyed by gene. **GTEx** is the deepest on
human bulk tissue and the only one of the three with genetics -- eQTLs, sQTLs, fine-mapping -- so it
is where "is this gene's expression under genetic control" is answered. **EBI Expression Atlas**
indexes other people's experiments rather than running one: use it to find the study, then follow
its accession.

Two traps are written into the parameter prose below because both return a confident empty answer
rather than an error. HPA drops a column code it does not recognise from an HTTP 200 response, which
is byte-identical to a gene with no data -- so this module ships the measured column catalogue and
``hpa_list_columns`` reads it. GTEx needs a GENCODE-versioned gene id whose version differs per
dataset, and the wrong version is an empty 200 -- so every GTEx function here resolves the id
against the dataset it is about to query and reports which id it used.

Adapted from ToolUniverse -- https://github.com/mims-harvard/ToolUniverse -- at commit
``f075c2a75e8b35ae5dbb220d48d4e87e980388b1``, Apache-2.0, Copyright [2025] [ToolUniverse team].
CHANGED BY SPATIALOMICSGYM as Apache-2.0 section 4(b) requires: the upstream JSON tool catalogs
(``hpa_tools.json``, ``gtex_v2_tools.json``, ``expression_atlas_tools.json``) were re-expressed as
this Python ``description`` list; ``result_type`` sub-mode switches became separate functions;
``HPA_get_protein_interactions_by_gene`` was dropped (D-012); ``hpa_list_columns`` is ours, not
upstream's, and answers a discovery question upstream's 21-code list answered wrongly. The
implementation module's docstring lists the upstream defects fixed here (F16-F27) with the
measurement behind each. See ``VENDORING.md``.
"""

description = [
    {
        "description": (
            "List the Human Protein Atlas column codes this module knows are real, optionally for "
            "one data family. Start here before any hpa_search_columns call: HPA answers a request "
            "for a column code it does not recognise with HTTP 200 and the column simply absent, "
            "which looks exactly like a gene with no data, so a typo or a retired code produces a "
            "confident wrong answer instead of an error. Makes no network call -- the catalogue was "
            "measured against the live endpoint and ships with the module. Families are: tissue, "
            "blood, brain, brain_single_nucleus, single_cell, cell_line, tissue_protein, "
            "cell_type_protein, dvp, mass_spec, prognostic. Print the returned dict."
        ),
        "name": "hpa_list_columns",
        "optional_parameters": [
            {
                "name": "family",
                "type": "str",
                "description": (
                    "One family name to list members of. Omit to get the family index plus every standalone column code"
                ),
                "default": None,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Free-text search of the Human Protein Atlas gene index, returning each hit's gene "
            "symbol, Ensembl id, description and synonyms. Use it to turn a name, a synonym or a "
            "partial description into the Ensembl id the other HPA functions take. The search is "
            "broad full text, not exact symbol match: 'TP53' returns dozens of rows and 'INS' "
            "returns thousands, so hits are ranked here with exact and prefix symbol matches first "
            "and the total is reported. Print the returned dict."
        ),
        "name": "hpa_search_genes",
        "optional_parameters": [
            {
                "name": "max_results",
                "type": "int",
                "description": "Hits to return after ranking, 1-200",
                "default": 10,
            },
        ],
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": ("Gene symbol, synonym or free text, e.g. 'TP53', 'tumor protein p53', 'EPCAM'"),
            },
        ],
    },
    {
        "description": (
            "Fetch arbitrary Human Protein Atlas columns for whatever the query matches -- the "
            "general-purpose escape hatch behind the specific HPA functions. Every code is checked "
            "against the measured catalogue before the request is spent, so a retired or misspelt "
            "code is an error naming the replacement rather than a silently missing column. Note "
            "that a cell_RNA_* code is a group selector, not one column: cell_RNA_lung_cancer "
            "expands to 232 columns. Labels in the result are whatever HPA returned, never a "
            "local table. Print the returned dict."
        ),
        "name": "hpa_search_columns",
        "optional_parameters": [
            {
                "name": "columns",
                "type": "list[str]",
                "description": (
                    "Column codes to request, e.g. ['g', 'eg', 'rnats', 't_RNA_liver']. Omit for the "
                    "gene identity, description and tissue specificity columns ('g', 'eg', 'gd', "
                    "'rnats'). hpa_list_columns() enumerates the valid codes"
                ),
                "default": None,
            },
            {
                "name": "max_rows",
                "type": "int",
                "description": "Rows to return, 1-200. HPA itself ignores a limit parameter, so "
                "this caps client-side after the response arrives",
                "default": 10,
            },
        ],
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": "Gene symbol, Ensembl id or free text to search for",
            },
        ],
    },
    {
        "description": (
            "One gene's core Human Protein Atlas identity card: symbol, Ensembl id, synonyms, "
            "description, chromosome and position, protein class, evidence level and biological "
            "process annotations. The cheap orienting call before a targeted expression question. "
            "Print the returned dict."
        ),
        "name": "hpa_get_gene_summary",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "ensembl_id",
                "type": "str",
                "description": (
                    "Ensembl gene id, e.g. 'ENSG00000141510'. This endpoint is keyed by Ensembl "
                    "id and rejects a bare symbol with an error naming hpa_search_genes() as the "
                    "way to get one -- it does not resolve symbols itself"
                ),
            },
        ],
    },
    {
        # Described as "everything the record carries, 119 fields" while the function returns a
        # fixed 16-field annotation subset (hunt 2026-09-30, uT3-atlases-7).
        "description": (
            "The functional-annotation part of one gene's Human Protein Atlas record, as HPA labels "
            "it: protein class, biological process, molecular function, disease involvement, "
            "evidence, antibodies and RRIDs, subcellular and secretome location -- 16 fields, with "
            "any the record lacks named in fields_absent. It carries no expression, specificity or "
            "cancer prognostic fields; use hpa_get_gene_details for those, or "
            "hpa_get_gene_summary for the short version. Field names are passed through exactly as "
            "returned, so they can be quoted in a result. Print the returned dict."
        ),
        "name": "hpa_get_gene_annotation",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "ensembl_id",
                "type": "str",
                "description": (
                    "Ensembl gene id, e.g. 'ENSG00000141510'. This endpoint is keyed by Ensembl "
                    "id and rejects a bare symbol with an error naming hpa_search_genes() as the "
                    "way to get one -- it does not resolve symbols itself"
                ),
            },
        ],
    },
    {
        "description": (
            "A structured multi-section profile for one gene: identity, expression specificity, "
            "antibody records and cancer prognostics, each section switchable. Immunohistochemistry "
            "images are deliberately not included -- they live on a separate endpoint that returns "
            "megabytes per gene and yields image URLs a text agent cannot read, while the citable "
            "part, antibody identity, is kept. An empty section means HPA has no data, not that the "
            "call failed. Print the returned dict."
        ),
        "name": "hpa_get_gene_details",
        "optional_parameters": [
            {
                "name": "include_expression",
                "type": "bool",
                "description": "Include the tissue, cell type and cell line specificity summaries",
                "default": True,
            },
            {
                "name": "include_antibodies",
                "type": "bool",
                "description": "Include antibody identifiers and RRIDs. The section is empty when "
                "HPA has no antibody for the gene, rather than holding a placeholder",
                "default": True,
            },
            {
                "name": "include_prognostics",
                "type": "bool",
                "description": "Include the cancer prognostic summaries",
                "default": True,
            },
        ],
        "required_parameters": [
            {
                "name": "ensembl_id",
                "type": "str",
                "description": (
                    "Ensembl gene id, e.g. 'ENSG00000141510'. This endpoint is keyed by Ensembl "
                    "id and rejects a bare symbol with an error naming hpa_search_genes() as the "
                    "way to get one -- it does not resolve symbols itself"
                ),
            },
        ],
    },
    {
        "description": (
            "Where in the cell a protein is, from Human Protein Atlas immunofluorescence: the main "
            "locations, additional locations, the reliability score behind them, and the "
            "single-cell variation annotations. Use before interpreting a spatial or imaging result "
            "that depends on compartment -- a nuclear marker behaving cytoplasmically is a finding "
            "only if the reference says nuclear. Print the returned dict."
        ),
        "name": "hpa_get_subcellular_location",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "gene_name",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'",
            },
        ],
    },
    {
        "description": (
            "The biological processes the Human Protein Atlas annotates for a gene, with an "
            "optional watchlist that marks which of them appear -- it marks, it does not filter, so "
            "the full list comes back either way. Useful for a quick functional read on an unknown "
            "marker before deciding whether it belongs in a signature. Print the returned dict."
        ),
        "name": "hpa_get_biological_processes",
        "optional_parameters": [
            {
                "name": "highlight_processes",
                "type": "list[str]",
                "description": (
                    "Process names to flag if present, matched case-insensitively. Omit to use the "
                    "default watchlist: Apoptosis, Biological rhythms, Cell cycle, Host-virus "
                    "interaction, Necrosis, Transcription, Transcription regulation"
                ),
                "default": None,
            },
        ],
        "required_parameters": [
            {
                "name": "gene_name",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'",
            },
        ],
    },
    {
        "description": (
            "Human Protein Atlas cancer prognostic summaries for one gene: for each cancer type "
            "where HPA found a significant association, whether high expression is favourable or "
            "unfavourable for survival and the p-value behind it. This is a survival association "
            "from TCGA-scale cohorts, not a causal claim and not a biomarker recommendation. Print "
            "the returned dict."
        ),
        "name": "hpa_get_cancer_prognostics",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "ensembl_id",
                "type": "str",
                "description": (
                    "Ensembl gene id, e.g. 'ENSG00000141510'. This endpoint is keyed by Ensembl "
                    "id and rejects a bare symbol with an error naming hpa_search_genes() as the "
                    "way to get one -- it does not resolve symbols itself"
                ),
            },
        ],
    },
    {
        "description": (
            "Consensus bulk RNA expression (nTPM) for one gene in named human tissues, from the "
            "Human Protein Atlas tissue panel. This is the reference level to compare a spatial or "
            "single-cell measurement against. Tissue names are matched against the 52 tissues HPA "
            "actually has, punctuation- and case-insensitively, and a name with no match is an "
            "error listing near misses rather than a silent zero. The expression_level band in the "
            "result is computed here from the number, not published by HPA; the cut-offs are "
            "stated in the response. Print the returned dict."
        ),
        "name": "hpa_get_tissue_rna_expression",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "ensembl_id",
                "type": "str",
                "description": "Ensembl gene id, e.g. 'ENSG00000141510' (a version suffix is "
                "ignored). A gene symbol is searched directly; if HPA has no exact "
                "match the result names the gene it reports instead",
            },
            {
                "name": "tissue_names",
                "type": "list[str]",
                "description": (
                    "Tissues to report, e.g. ['liver', 'brain', 'skeletal muscle']. "
                    "hpa_list_columns('tissue') lists all 52"
                ),
            },
        ],
    },
    {
        "description": (
            "RNA expression for one gene from a chosen Human Protein Atlas data family -- bulk "
            "tissue, blood cell types, brain regions, brain single nuclei, single cell types, or "
            "cell lines. Use when the question names the measurement context: 'which immune cell "
            "expresses it' is the blood family, 'which brain region' is brain. One request per "
            "family, so provenance is structural rather than inferred from a label. Print the "
            "returned dict."
        ),
        "name": "hpa_get_rna_expression_by_source",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "gene_name",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'",
            },
            {
                "name": "source_type",
                "type": "str",
                "description": (
                    "One of: tissue, blood, brain, brain_single_nucleus, single_cell, cell_line, "
                    "tissue_protein, cell_type_protein, dvp, mass_spec, prognostic"
                ),
            },
            {
                "name": "source_names",
                "type": "list[str]",
                "description": (
                    "Members of that family to report, e.g. ['liver', 'kidney'] for tissue or "
                    "['T-reg', 'NK-cell'] for blood. hpa_list_columns(family) lists the members"
                ),
            },
        ],
    },
    {
        "description": (
            "Compare a gene's expression in a group of cancer cell lines against its normal tissue "
            "levels, both from the Human Protein Atlas, with the fold change per line. The question "
            "this answers is whether a line is a defensible model for a gene in a tissue. The "
            "comparison is against the per-tissue bulk panel, not against HPA's tissue-specificity "
            "summary -- that summary is null for most genes, which is why the upstream version of "
            "this comparison had no denominator most of the time. Print the returned dict."
        ),
        "name": "hpa_compare_cell_line_to_tissue",
        "optional_parameters": [
            {
                "name": "tissue_names",
                "type": "list[str]",
                # The default was described as "the tissues HPA reports highest for the gene"; the
                # code has always used the whole panel (hunt 2026-09-30, uT3-atlases-9).
                "description": (
                    "Normal tissues to compare against. Omit to compare against the median of all "
                    "51 tissues in the panel"
                ),
                "default": None,
            },
        ],
        "required_parameters": [
            {
                "name": "gene_name",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'",
            },
            {
                "name": "cell_line_group",
                "type": "str",
                "description": (
                    "A cell line group, e.g. 'lung_cancer', 'breast_cancer', 'liver_cancer'. These "
                    "are groups, not single lines -- hpa_list_columns('cell_line') lists all 30"
                ),
            },
        ],
    },
    {
        "description": (
            "Rank the individual cell lines in a Human Protein Atlas cell line group by how much "
            "they express a gene. Answers 'which line should I use' rather than 'is it expressed': "
            "one group code returns up to 232 individual lines, and this reports the top ones with "
            "their nTPM plus how many lines were behind the ranking. Print the returned dict."
        ),
        "name": "hpa_get_cell_line_expression",
        "optional_parameters": [
            {
                "name": "top_n",
                "type": "int",
                "description": "Cell lines to return, highest first, 1-200",
                "default": 15,
            },
        ],
        "required_parameters": [
            {
                "name": "gene_name",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'",
            },
            {
                "name": "cell_line_group",
                "type": "str",
                "description": ("A cell line group, e.g. 'lung_cancer'. hpa_list_columns('cell_line') lists all 30"),
            },
        ],
    },
    {
        "description": (
            "Expression of one gene in a single named context -- a tissue, a blood cell type, a "
            "brain region, a single cell type or a cell line group -- resolved across all of HPA's "
            "families without needing to know which family the context belongs to. Use it when the "
            "question names one place and you want one number. A context HPA has no data for says "
            "so; it does not report zero expression, which is a different claim. Print the returned "
            "dict."
        ),
        "name": "hpa_get_contextual_expression",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "gene_name",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'",
            },
            {
                "name": "context_name",
                "type": "str",
                "description": (
                    "One context, e.g. 'liver', 'cerebral cortex', 'T-reg', 'lung_cancer'. Matched "
                    "punctuation- and case-insensitively against every family"
                ),
            },
        ],
    },
    {
        "description": (
            "List the GTEx datasets available through the v2 API, with the genome build and GENCODE "
            "annotation version each one uses. Call this first when the analysis is not about the "
            "default bulk release: GTEx keys genes by versioned Ensembl id and the version differs "
            "per dataset, so the same gene is ENSG00000141510.16 in gtex_v8 and .18 in gtex_v10. "
            "Every GTEx function in this module resolves that for you and reports the id it used -- "
            "this call is how you find out which dataset to ask for. Print the returned dict."
        ),
        "name": "gtex_list_datasets",
        "optional_parameters": [],
        "required_parameters": [],
    },
    {
        "description": (
            "List the GTEx tissue sites in a dataset, with the exact tissueSiteDetailId code, the "
            "human-readable name and the sample count behind each. The codes are the vocabulary "
            "every other GTEx function here takes, and they are not guessable -- 'Brain - Cortex' "
            "is Brain_Cortex but 'Cells - Cultured fibroblasts' is Cells_Cultured_fibroblasts. "
            "Sample counts also tell you which tissues have enough power for an eQTL question. "
            "Print the returned dict."
        ),
        "name": "gtex_list_tissue_sites",
        "optional_parameters": [
            {
                "name": "dataset_id",
                "type": "str",
                "description": "GTEx dataset, e.g. 'gtex_v8' or 'gtex_v10'. Tissue lists differ "
                "between releases. gtex_list_datasets() enumerates them",
                "default": "gtex_v8",
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Median bulk RNA expression (TPM) for one gene across GTEx tissues -- the standard "
            "human reference level, and the usual denominator for 'is this enriched here'. One "
            "number per tissue, so it is the compact call; use gtex_get_gene_expression when the "
            "spread across individual donors matters. Print the returned dict."
        ),
        "name": "gtex_get_median_gene_expression",
        "optional_parameters": [
            {
                "name": "tissue_site_detail_ids",
                "type": "list[str]",
                "description": (
                    "Tissues to report, e.g. ['Liver', 'Brain_Cortex']. Omit for every tissue in "
                    "the dataset. gtex_list_tissue_sites() gives the exact codes"
                ),
                "default": None,
            },
            {
                "name": "dataset_id",
                "type": "str",
                "description": "GTEx dataset. Decides the GENCODE version the gene id is resolved "
                "against, so do not mix it with an id taken from another release",
                "default": "gtex_v8",
            },
        ],
        "required_parameters": [
            {
                "name": "gene",
                "type": "str",
                "description": (
                    "Gene symbol or Ensembl id, e.g. 'TP53' or 'ENSG00000141510'. An unversioned "
                    "id is versioned for you against the dataset; a symbol is resolved first"
                ),
            },
        ],
    },
    {
        "description": (
            "Per-sample bulk RNA expression for one gene in GTEx -- every donor's value, not the "
            "median. Use it when the question is about spread, outliers or a bimodal distribution "
            "rather than a typical level; a tissue can have a modest median and a long tail. Large "
            "responses, so it pages and reports the total. Print the returned dict."
        ),
        "name": "gtex_get_gene_expression",
        "optional_parameters": [
            {
                "name": "tissue_site_detail_ids",
                "type": "list[str]",
                "description": "Tissues to report. Omit for all of them, which is a lot of samples",
                "default": None,
            },
            {
                "name": "dataset_id",
                "type": "str",
                "description": "GTEx dataset, e.g. 'gtex_v8'",
                "default": "gtex_v8",
            },
            {
                "name": "page_size",
                "type": "int",
                "description": "Samples per page, 1-250",
                "default": 100,
            },
        ],
        "required_parameters": [
            {
                "name": "gene",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'",
            },
        ],
    },
    {
        "description": (
            "The genes GTEx ranks highest by median expression in one tissue -- the reverse lookup: "
            "'what defines this tissue' rather than 'where is my gene'. Useful for sanity-checking "
            "a spatial cluster's identity against the bulk reference. Mitochondrial genes dominate "
            "the top of an unfiltered ranking in most tissues, which is why they are filtered by "
            "default. Print the returned dict."
        ),
        "name": "gtex_get_top_expressed_genes",
        "optional_parameters": [
            {
                "name": "dataset_id",
                "type": "str",
                "description": "GTEx dataset, e.g. 'gtex_v8'",
                "default": "gtex_v8",
            },
            {
                "name": "filter_mt_gene",
                "type": "bool",
                "description": "Exclude mitochondrial genes. Set False only if MT content is the "
                "question -- otherwise they crowd out the tissue-specific signal",
                "default": True,
            },
            {
                "name": "page_size",
                "type": "int",
                "description": "Genes to return, 1-250",
                "default": 50,
            },
        ],
        "required_parameters": [
            {
                "name": "tissue_site_detail_id",
                "type": "str",
                "description": "One tissue code, e.g. 'Liver'. gtex_list_tissue_sites() lists them",
            },
        ],
    },
    {
        "description": (
            "Median expression per transcript isoform for one gene in GTEx, rather than one number "
            "for the whole gene. Use when isoform usage is the question -- a gene flat across "
            "tissues at gene level can switch dominant isoform between them, which gene-level TPM "
            "hides entirely. Print the returned dict."
        ),
        "name": "gtex_get_median_transcript_expression",
        "optional_parameters": [
            {
                "name": "tissue_site_detail_id",
                "type": "str",
                "description": "One tissue code, e.g. 'Liver'. Omit for every tissue, which "
                "multiplies rows by the number of isoforms",
                "default": None,
            },
            {
                "name": "dataset_id",
                "type": "str",
                "description": "GTEx dataset. Transcript models follow the dataset's GENCODE "
                "version, so isoform ids are not comparable across releases",
                "default": "gtex_v8",
            },
        ],
        "required_parameters": [
            {
                "name": "gene",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'",
            },
        ],
    },
    {
        "description": (
            "Single-nucleus RNA expression for one gene from the GTEx snRNA-seq pilot, broken down "
            "by tissue and cell type -- the bridge between GTEx's bulk reference and a single-cell "
            "or spatial result. Answers 'which cell type carries the bulk signal'. The pilot covers "
            "a handful of tissues, not all of GTEx: gtex_get_single_nucleus_cell_counts() reports "
            "which, and how many nuclei back each. Print the returned dict."
        ),
        "name": "gtex_get_single_nucleus_expression",
        "optional_parameters": [
            {
                "name": "tissue_site_detail_ids",
                "type": "list[str]",
                "description": "Tissues to report. Omit for every tissue in the pilot",
                "default": None,
            },
            {
                "name": "dataset_id",
                "type": "str",
                "description": "Single-nucleus dataset id. The bulk datasets have no cell types, so "
                "this is not interchangeable with 'gtex_v8'",
                "default": "gtex_snrnaseq_pilot",
            },
        ],
        "required_parameters": [
            {
                "name": "gene",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'",
            },
        ],
    },
    {
        "description": (
            "GTEx sample-level metadata -- tissue, sex, age bracket, RIN, ischaemic time, "
            "autolysis score and sequencing platform. Use it to judge whether a tissue's numbers "
            "are trustworthy before building on them: RIN and ischaemic time vary sharply between "
            "GTEx tissues and drive real expression differences that look biological. Print the "
            "returned dict."
        ),
        "name": "gtex_get_sample_info",
        "optional_parameters": [
            {
                "name": "dataset_id",
                "type": "str",
                "description": "GTEx dataset, e.g. 'gtex_v8'",
                "default": "gtex_v8",
            },
            {
                "name": "tissue_site_detail_ids",
                "type": "list[str]",
                "description": "Tissues to report. Omit for every sample in the dataset, which is "
                "tens of thousands of rows -- name a tissue unless you need the lot",
                "default": None,
            },
            {
                "name": "page_size",
                "type": "int",
                "description": "Samples per page, 1-250",
                "default": 100,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "The genes with at least one significant cis-eQTL in a given GTEx tissue (eGenes), with "
            "the lead variant and q-value for each. Answers 'what is under genetic control here' "
            "before you have a gene in mind. The list is long in well-powered tissues, so it pages "
            "and reports the total. Print the returned dict."
        ),
        "name": "gtex_get_eqtl_genes",
        "optional_parameters": [
            {
                "name": "dataset_id",
                "type": "str",
                "description": "GTEx dataset, e.g. 'gtex_v8'",
                "default": "gtex_v8",
            },
            {
                "name": "page_size",
                "type": "int",
                "description": "Genes per page, 1-250",
                "default": 100,
            },
        ],
        "required_parameters": [
            {
                "name": "tissue_site_detail_id",
                "type": "str",
                "description": "One tissue code, e.g. 'Liver'. eQTL discovery power tracks sample "
                "count -- gtex_list_tissue_sites() reports it",
            },
        ],
    },
    {
        "description": (
            "Significant cis-eQTLs from GTEx, per tissue, for a gene or a variant. This is the "
            "everyday eQTL lookup: 'does genotype at this locus change this gene's expression, and "
            "where'. Give a gene, a variant, or both -- the endpoint rejects a query with neither, "
            "so that is reported here without spending a request. Print the returned dict."
        ),
        "name": "gtex_get_single_tissue_eqtls",
        "optional_parameters": [
            {
                "name": "gene",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'. Required unless variant_id is given",
                "default": None,
            },
            {
                "name": "variant_id",
                "type": "str",
                "description": (
                    "GTEx variant id, e.g. 'chr17_7676154_G_A_b38'. Required unless gene is given. "
                    "The build suffix must match the dataset"
                ),
                "default": None,
            },
            {
                "name": "tissue_site_detail_ids",
                "type": "list[str]",
                "description": "Tissues to restrict to. Omit for every tissue with a significant association",
                "default": None,
            },
            {
                "name": "dataset_id",
                "type": "str",
                "description": "GTEx dataset, e.g. 'gtex_v8'",
                "default": "gtex_v8",
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Compute the eQTL effect for one gene-variant-tissue combination on demand, returning "
            "the normalised effect size, p-value and the genotype group sizes behind it. Use it for "
            "a pair that is not in the significant set -- gtex_get_single_tissue_eqtls only returns "
            "associations that passed the significance threshold, so absence there is not evidence "
            "of no effect. Computed on request, so a null result means no association was found, "
            "not that the record is missing. Print the returned dict."
        ),
        "name": "gtex_calculate_eqtl",
        "optional_parameters": [
            {
                "name": "dataset_id",
                "type": "str",
                "description": "GTEx dataset, e.g. 'gtex_v8'",
                "default": "gtex_v8",
            },
        ],
        "required_parameters": [
            {
                "name": "gene",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'",
            },
            {
                "name": "variant_id",
                "type": "str",
                "description": "GTEx variant id, e.g. 'chr17_7676154_G_A_b38'",
            },
            {
                "name": "tissue_site_detail_id",
                "type": "str",
                "description": "One tissue code, e.g. 'Liver'",
            },
        ],
    },
    {
        "description": (
            "Cross-tissue eQTL meta-analysis (METASOFT) for a gene: the m-value per tissue, which "
            "is the posterior probability the effect is present there, alongside the per-tissue "
            "effect sizes. Use it to tell a tissue-specific eQTL from one shared across the body -- "
            "the single-tissue call cannot distinguish 'absent here' from 'underpowered here', and "
            "this is the call that can. Print the returned dict."
        ),
        "name": "gtex_get_multi_tissue_eqtls",
        "optional_parameters": [
            {
                "name": "variant_id",
                "type": "str",
                "description": "Restrict to one GTEx variant id, e.g. 'chr17_7676154_G_A_b38'. "
                "Omit for every variant meta-analysed for the gene",
                "default": None,
            },
            {
                "name": "dataset_id",
                "type": "str",
                "description": "GTEx dataset, e.g. 'gtex_v8'",
                "default": "gtex_v8",
            },
        ],
        "required_parameters": [
            {
                "name": "gene",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'",
            },
        ],
    },
    {
        "description": (
            "Significant cis-sQTLs from GTEx, per tissue, for a gene or a variant -- variants "
            "associated with splicing rather than overall expression level, reported per intron "
            "excision phenotype. Use when a variant has no eQTL but is still suspected functional: "
            "splice effects are a common answer. Give a gene, a variant, or both; neither is "
            "rejected by the endpoint. Print the returned dict."
        ),
        "name": "gtex_get_single_tissue_sqtls",
        "optional_parameters": [
            {
                "name": "gene",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'. Required unless variant_id is given",
                "default": None,
            },
            {
                "name": "variant_id",
                "type": "str",
                "description": "GTEx variant id, e.g. 'chr17_7676154_G_A_b38'. Required unless gene is given",
                "default": None,
            },
            {
                "name": "tissue_site_detail_ids",
                "type": "list[str]",
                "description": "Tissues to restrict to. Omit for every tissue with a significant association",
                "default": None,
            },
            {
                "name": "dataset_id",
                "type": "str",
                "description": "GTEx dataset, e.g. 'gtex_v8'",
                "default": "gtex_v8",
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Statistical fine-mapping of a gene's cis-eQTL signal: the credible set each variant "
            "belongs to and its posterior inclusion probability. Use it to narrow a locus to "
            "candidate causal variants -- the lead variant of an association is frequently not the "
            "causal one, just the best-tagged, and a credible set is what distinguishes the two. "
            "Print the returned dict."
        ),
        "name": "gtex_get_finemapping",
        "optional_parameters": [
            {
                "name": "tissue_site_detail_ids",
                "type": "list[str]",
                "description": "Tissues to restrict to. Omit for every tissue fine-mapped for the gene",
                "default": None,
            },
            {
                "name": "dataset_id",
                "type": "str",
                "description": "GTEx dataset, e.g. 'gtex_v8'",
                "default": "gtex_v8",
            },
        ],
        "required_parameters": [
            {
                "name": "gene",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'",
            },
        ],
    },
    {
        "description": (
            "How many nuclei the GTEx single-nucleus pilot has per tissue and cell type -- the "
            "dataset's cell census, with no gene involved. Call it before "
            "gtex_get_single_nucleus_expression to see which tissues and cell types exist and "
            "whether a cell type has enough nuclei to support a claim; a cell type with a handful "
            "of nuclei will produce a number, and the number will not mean much. Print the "
            "returned dict."
        ),
        "name": "gtex_get_single_nucleus_cell_counts",
        "optional_parameters": [
            {
                "name": "tissue_site_detail_ids",
                "type": "list[str]",
                "description": "Tissues to report. Omit for the whole census",
                "default": None,
            },
            {
                "name": "dataset_id",
                "type": "str",
                "description": "Single-nucleus dataset id",
                "default": "gtex_snrnaseq_pilot",
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "The genes with at least one significant cis-sQTL (sGenes), optionally restricted to "
            "one tissue, with the q-value for each. The splicing counterpart of "
            "gtex_get_eqtl_genes: 'whose splicing is under genetic control here'. Print the "
            "returned dict."
        ),
        "name": "gtex_get_sqtl_genes",
        "optional_parameters": [
            {
                "name": "tissue_site_detail_id",
                "type": "str",
                "description": "One tissue code, e.g. 'Liver'. Omit for every tissue, which is a long list",
                "default": None,
            },
            {
                "name": "dataset_id",
                "type": "str",
                "description": "GTEx dataset, e.g. 'gtex_v8'",
                "default": "gtex_v8",
            },
            {
                "name": "page_size",
                "type": "int",
                "description": "Genes per page, 1-250",
                "default": 100,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Conditionally independent cis-eQTL signals for a gene, ranked. Rank 1 is the primary "
            "signal; rank 2 and above are independent of it, not duplicates of it -- so a gene with "
            "three ranks has three distinct regulatory variants, which a plain eQTL list would show "
            "as one correlated cloud. Use it before concluding a locus has a single causal variant. "
            "Print the returned dict."
        ),
        "name": "gtex_get_independent_eqtls",
        "optional_parameters": [
            {
                "name": "tissue_site_detail_ids",
                "type": "list[str]",
                "description": "Tissues to restrict to. Omit for every tissue with an independent signal for the gene",
                "default": None,
            },
            {
                "name": "dataset_id",
                "type": "str",
                "description": "GTEx dataset, e.g. 'gtex_v8'",
                "default": "gtex_v8",
            },
        ],
        "required_parameters": [
            {
                "name": "gene",
                "type": "str",
                "description": "Gene symbol or Ensembl id, e.g. 'TP53'",
            },
        ],
    },
    {
        "description": (
            "List EBI Expression Atlas baseline experiments for a species -- studies measuring "
            "expression across tissues, cell types or developmental stages in unperturbed samples, "
            "largest study first. Use it to find an existing dataset to compare your own against, "
            "then follow the accession with expression_atlas_get_experiment. Baseline answers "
            "'where is this normally expressed'; expression_atlas_search_differential answers 'what "
            "changes it'. The Atlas has no server-side filter, so each call downloads the whole "
            "catalogue and filters here -- results are capped and paged rather than looped over. "
            "Print the returned dict."
        ),
        "name": "expression_atlas_list_baseline_experiments",
        "optional_parameters": [
            {
                "name": "species",
                "type": "str",
                "description": (
                    "Scientific name, e.g. 'Homo sapiens', 'Mus musculus'. Matched "
                    "case-insensitively; an unrecognised name is an error listing close matches, "
                    "never a silent empty list"
                ),
                "default": "Homo sapiens",
            },
            {
                "name": "limit",
                "type": "int",
                "description": "Experiments to return, 1-200",
                "default": 50,
            },
            {
                "name": "offset",
                "type": "int",
                "description": "Rows to skip, for paging through a long list",
                "default": 0,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Search EBI Expression Atlas differential experiments -- studies with a contrast, "
            "disease versus healthy, treated versus control, knockout versus wild type. Use it to "
            "find published evidence that something changes a gene, and to locate a comparable "
            "design before running your own. The condition search is a substring match over each "
            "experiment's description and experimental factors, so a broad word like 'cancer' "
            "matches widely and a specific one like 'idiopathic pulmonary fibrosis' matches "
            "narrowly. Print the returned dict."
        ),
        "name": "expression_atlas_search_differential",
        "optional_parameters": [
            {
                "name": "condition",
                "type": "str",
                "description": (
                    "Free text to match, e.g. 'breast cancer', 'hypoxia', 'LPS'. Omit to list every "
                    "differential experiment for the species"
                ),
                "default": None,
            },
            {
                "name": "species",
                "type": "str",
                "description": "Scientific name, e.g. 'Homo sapiens'. An unrecognised name is an "
                "error listing close matches",
                "default": "Homo sapiens",
            },
            {
                "name": "limit",
                "type": "int",
                "description": "Experiments to return, 1-200",
                "default": 50,
            },
            {
                "name": "offset",
                "type": "int",
                "description": "Rows to skip, for paging",
                "default": 0,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Search the whole EBI Expression Atlas catalogue -- baseline and differential together, "
            "any species -- by condition text. The widest of the three search functions: use it "
            "when the species is not fixed (a mouse model of a human disease) or when you do not "
            "yet know whether the relevant evidence is baseline or differential. Each result says "
            "which kind it is, so you can narrow afterwards. Print the returned dict."
        ),
        "name": "expression_atlas_search_experiments",
        "optional_parameters": [
            {
                "name": "condition",
                "type": "str",
                "description": "Free text to match against description and experimental factors. "
                "Omit to browse the catalogue",
                "default": None,
            },
            {
                "name": "species",
                "type": "str",
                "description": "Scientific name to restrict to, e.g. 'Mus musculus'. Omit for every species",
                "default": None,
            },
            {
                "name": "limit",
                "type": "int",
                "description": "Experiments to return, 1-200",
                "default": 50,
            },
            {
                "name": "offset",
                "type": "int",
                "description": "Rows to skip, for paging",
                "default": 0,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Open one EBI Expression Atlas experiment by accession: its identity, its design, and "
            "the first page of its gene table. The design is the part worth reading -- for a "
            "baseline experiment, the assay groups with their factor values, ontology term ids and "
            "replicate counts; for a differential one, each named contrast with both sides spelled "
            "out, so you can see what was actually compared before citing a fold change. The gene "
            "rows come with their unit (TPM for baseline, log2 fold change with p-value for "
            "differential). Print the returned dict."
        ),
        "name": "expression_atlas_get_experiment",
        "optional_parameters": [
            {
                "name": "max_gene_rows",
                "type": "int",
                "description": (
                    "Gene rows to return, 1-200. This endpoint has no gene filter -- the rows are "
                    "the service's own first page, not a selection, so for a named gene use the "
                    "HPA or GTEx functions in this module instead"
                ),
                "default": 10,
            },
        ],
        "required_parameters": [
            {
                "name": "accession",
                "type": "str",
                "description": (
                    "Expression Atlas accession, e.g. 'E-MTAB-2836' or 'E-GEOD-26284'. An "
                    "accession the atlas does not hold is rejected rather than answered with an "
                    "empty record, and is reported here as an unrecognised-accession error -- "
                    "including for accessions that once existed and have since been retired, so "
                    "take the accession from expression_atlas_search_experiments() rather than "
                    "from memory"
                ),
            },
        ],
    },
]
