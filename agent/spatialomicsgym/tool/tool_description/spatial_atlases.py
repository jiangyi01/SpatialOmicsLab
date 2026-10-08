"""Schema for the spatial atlas tools in ``tool/spatial_atlases.py``.

This list -- not the docstrings next door -- is what ``read_module2api()`` collects and
``utils/formatting.py:textify_api_dict`` renders into the function signatures the model sees. A
parameter absent from here is a knob the agent cannot turn; a parameter here that the function does
not accept is a ``TypeError`` on the first call. Keep both files in step.

Two atlases, and they answer different questions. **Allen Brain Atlas** is a reference: one
in-situ-hybridisation or microarray measurement of a gene across a whole brain, mapped onto a named
anatomical ontology, so it answers "where in the brain is this gene expressed" with structure names
and numbers. **HuBMAP** is a catalogue of other people's tissue datasets -- Visium, Xenium, CODEX,
snRNAseq and more across 47 organ codes -- so it answers "whose spatial data can I reuse for this
organ or this assay", and hands back an identifier and a DOI rather than a measurement.

Three traps are written into the parameter prose below because each one returns a confident empty
answer rather than an error. An Allen product is tied to one organism and one measurement type, so a
mouse-cased symbol against the human microarray product is an empty 200 -- ``product_id`` names the
product in its description and the mismatch is reported as an error that says which organism the
gene does belong to. HuBMAP's two-letter organ codes split ten organs by laterality, so ``LL`` is
half a lung and the code upstream advertised for lung, ``LU``, is the ureter -- here an organ *name*
expands to every code that shares it. HuBMAP's assay vocabulary is matched exactly, so a plausible
spelling that the live vocabulary does not carry is an error listing near matches, not zero rows.

Adapted from ToolUniverse -- https://github.com/mims-harvard/ToolUniverse -- at commit
``f075c2a75e8b35ae5dbb220d48d4e87e980388b1``, Apache-2.0, Copyright [2025] [ToolUniverse team].
CHANGED BY SPATIALOMICSGYM as Apache-2.0 section 4(b) requires: the upstream JSON tool catalogs
(``allen_brain_tools.json``, ``hubmap_tools.json``) were re-expressed as this Python ``description``
list; ``allen_brain_list_products`` and ``hubmap_list_dataset_types`` are ours, not upstream's, and
exist because the two parameters most likely to produce a silent empty answer had no discovery call.
The implementation module's docstring lists the upstream defects fixed here (F28-F40) with the
measurement behind each. See ``VENDORING.md``.
"""

description = [
    {
        "description": (
            "List the Allen Brain Atlas data products -- the named collections an expression query "
            "runs against -- with the organism and measurement type of each. Start here before any "
            "allen_brain_get_expression_datasets call, because a product is tied to one organism "
            "and one technique: product 1 is mouse in-situ hybridisation and product 2 is the "
            "*human* microarray, so a mouse-cased gene symbol against product 2 returns zero rows "
            "with no error. Print the returned dict."
        ),
        "name": "allen_brain_list_products",
        "optional_parameters": [
            {
                "name": "species",
                "type": "str",
                "description": (
                    "Keep only products for one organism -- 'human', 'mouse' or 'nhp'. Omit to "
                    "list all 64"
                ),
                "default": None,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Search the Allen Brain Atlas gene catalogue by exact acronym or by a fragment of the "
            "gene name, returning the Allen gene id each expression query needs plus the organism "
            "the record belongs to. Allen uses the source organism's capitalisation -- 'Gad1' in "
            "mouse, 'GAD1' in human -- and an acronym that matches nothing is reported as an error "
            "carrying acronyms that do exist, rather than as an empty result. Print the returned "
            "dict."
        ),
        "name": "allen_brain_search_genes",
        "optional_parameters": [
            {
                "name": "acronym",
                "type": "str",
                "description": (
                    "Exact gene acronym, case-sensitive in the source organism's convention, e.g. "
                    "'Gad1' or 'GAD1'. One of acronym or name_contains is required"
                ),
                "default": None,
            },
            {
                "name": "name_contains",
                "type": "str",
                "description": (
                    "Fragment of the full gene name to match, e.g. 'glutamate decarboxylase'. Use "
                    "when the acronym's capitalisation is unknown"
                ),
                "default": None,
            },
            {
                "name": "species",
                "type": "str",
                "description": (
                    "Restrict to one organism -- 'human', 'mouse' or 'rat'. Omit to search every "
                    "organism Allen carries"
                ),
                "default": None,
            },
            {
                "name": "limit",
                "type": "int",
                "description": "Maximum gene records to return",
                "default": 50,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Search the Allen Brain Atlas anatomical ontologies for structures by acronym or name, "
            "reporting for every hit which atlas it belongs to. Acronyms are not unique across "
            "atlases -- 'CA1' is nine different structure records spanning the adult mouse, "
            "developing mouse, human and developing human ontologies, each with its own id -- so a "
            "row without its atlas is an id that may not exist in the atlas you are about to query. "
            "Print the returned dict."
        ),
        "name": "allen_brain_search_structures",
        "optional_parameters": [
            {
                "name": "acronym",
                "type": "str",
                "description": (
                    "Exact structure acronym, e.g. 'CA1' or 'MOp'. Atlas-specific and "
                    "case-sensitive. One of acronym or name_contains is required"
                ),
                "default": None,
            },
            {
                "name": "name_contains",
                "type": "str",
                "description": (
                    "Fragment of the structure name to match, e.g. 'hippocamp'. Use when the "
                    "acronym is unknown"
                ),
                "default": None,
            },
            {
                "name": "atlas",
                "type": "str",
                "description": (
                    "Keep only structures from atlases whose name contains this text, e.g. 'Mouse "
                    "Brain Atlas' or 'Human'. Use it to collapse a multi-atlas acronym to one id"
                ),
                "default": None,
            },
            {
                "name": "species",
                "type": "str",
                "description": (
                    "Keep only atlases for one organism -- 'human' or 'mouse'. A coarser filter "
                    "than atlas"
                ),
                "default": None,
            },
            {
                "name": "limit",
                "type": "int",
                "description": "Maximum structure records to return",
                "default": 50,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Fetch one Allen Brain Atlas structure by its numeric id, with its atlas, its position "
            "in that atlas's hierarchy and its display colour. Use it to confirm that an id from "
            "allen_brain_search_structures is the structure you meant before reading expression "
            "against it. Print the returned dict."
        ),
        "name": "allen_brain_get_structure",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "structure_id",
                "type": "int",
                "description": (
                    "Allen structure id, e.g. 382 for field CA1 in the adult mouse atlas. Ids are "
                    "per-atlas; allen_brain_search_structures returns the id in each atlas"
                ),
            },
        ],
    },
    {
        "description": (
            "List the Allen Brain Atlas experiments that measured one gene, returning the "
            "SectionDataSet id that allen_brain_get_structure_expression_values reads. The product "
            "decides both the organism and the technique, so this is where a species mismatch "
            "surfaces: asking product 2 (the human microarray) for a mouse-cased symbol matches "
            "nothing, and here that is an error naming the organisms Allen does carry the gene for, "
            "not an empty list. Print the returned dict."
        ),
        "name": "allen_brain_get_expression_datasets",
        "optional_parameters": [
            {
                "name": "product_id",
                "type": "int",
                "description": (
                    "Allen product to search. 1 = Mouse Brain in-situ hybridisation (the default), "
                    "2 = *Human* Brain Microarray, 3 = Developing Mouse Brain. "
                    "allen_brain_list_products() lists all 64 with their organisms"
                ),
                "default": 1,
            },
            {
                "name": "limit",
                "type": "int",
                "description": "Maximum experiments to return",
                "default": 50,
            },
            {
                "name": "include_failed",
                "type": "bool",
                "description": (
                    "Include experiments Allen marked as failed quality control. False by default "
                    "because their expression values are not comparable"
                ),
                "default": False,
            },
        ],
        "required_parameters": [
            {
                "name": "gene_acronym",
                "type": "str",
                "description": (
                    "Gene acronym in the source organism's capitalisation, e.g. 'Gad1' for mouse "
                    "or 'GAD1' for human. allen_brain_search_genes resolves an unknown one"
                ),
            },
        ],
    },
    {
        "description": (
            "Read the per-structure expression numbers out of one Allen experiment, ranked so the "
            "highest-signal structures come first. This is the call that answers 'where in the "
            "brain is this gene expressed': one row per anatomical structure, each carrying "
            "expression energy, density and the voxel counts behind them. An experiment covers "
            "roughly 2,400 structures, so the ranking is done by the server across the whole "
            "result set before it is paged -- the default 50 rows are the top 50, not an arbitrary "
            "50. Print the returned dict."
        ),
        "name": "allen_brain_get_structure_expression_values",
        "optional_parameters": [
            {
                "name": "limit",
                "type": "int",
                "description": "Maximum structures to return, taken from the top of the ranking",
                "default": 50,
            },
            {
                "name": "include_structure",
                "type": "bool",
                "description": (
                    "Attach each structure's name, acronym and hierarchy path. Without it the rows "
                    "carry only numeric structure ids"
                ),
                "default": True,
            },
            {
                "name": "sort_by",
                "type": "str",
                "description": (
                    "Measure to rank by: 'expression_energy' (default; density x intensity, the "
                    "usual answer to 'how much'), 'expression_density', 'sum_expressing_pixels', "
                    "'sum_expressing_pixel_intensity', 'sum_pixels', 'sum_pixel_intensity', "
                    "'voxel_energy_mean' or 'voxel_energy_cv'. Pass 'none' for the API's own order"
                ),
                "default": "expression_energy",
            },
        ],
        "required_parameters": [
            {
                "name": "section_data_set_id",
                "type": "int",
                "description": (
                    "Allen SectionDataSet id identifying one experiment, e.g. 480. "
                    "allen_brain_get_expression_datasets returns these for a gene"
                ),
            },
        ],
    },
    {
        "description": (
            "List HuBMAP's organ vocabulary -- every two-letter code, the organ name and ontology "
            "term behind it, and how many published datasets each one has. Start here before any "
            "hubmap_search_datasets call with an organ, because ten organs are split by laterality "
            "into twenty codes (kidney LK/RK, lung LL/RL, eye, ovary, fallopian tube, knee, tonsil, "
            "ureter, mammary gland, main bronchus) and a single code is therefore half an organ. "
            "The counts also show which codes are empty -- 18 of 47 have no published data at all. "
            "Print the returned dict."
        ),
        "name": "hubmap_list_organs",
        "optional_parameters": [
            {
                "name": "with_dataset_counts",
                "type": "bool",
                "description": (
                    "Attach the live published-dataset count per organ code. One extra request; "
                    "set False for the vocabulary alone"
                ),
                "default": True,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "List the assay names HuBMAP actually uses, most-populated first, with the published "
            "dataset count of each. Start here before filtering hubmap_search_datasets by "
            "dataset_type: the names are matched exactly and the live vocabulary spells things "
            "unexpectedly -- there is no 'snATACseq', the assay is called 'ATACseq', and Visium "
            "appears as 'Visium (no probes)'. Print the returned dict."
        ),
        "name": "hubmap_list_dataset_types",
        "optional_parameters": [
            {
                "name": "limit",
                "type": "int",
                "description": "Maximum assay names to return, taken from the most populated end",
                "default": 60,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Search HuBMAP's human tissue atlas for datasets by organ, assay or free text, "
            "returning for each hit its HuBMAP id, title, organ, assay, donor sex and age, and DOI. "
            "This is the reuse question -- whose spatial or single-cell data exists for this tissue "
            "-- and the id it returns is what hubmap_get_dataset and "
            "hubmap_get_dataset_provenance read. Filters narrow independently, and a combination "
            "matching nothing is reported as an error naming each filter so it is clear which one "
            "to drop. Print the returned dict."
        ),
        "name": "hubmap_search_datasets",
        "optional_parameters": [
            {
                "name": "organ",
                "type": "str",
                "description": (
                    "Organ name or two-letter code. A *name* such as 'lung' or 'kidney' expands to "
                    "every code for that organ including both sides of a paired one, which is "
                    "usually what is meant; a code such as 'LL' is one side only. "
                    "hubmap_list_organs() lists all 47"
                ),
                "default": None,
            },
            {
                "name": "dataset_type",
                "type": "str",
                "description": (
                    "Assay name, matched exactly against HuBMAP's live vocabulary, e.g. 'ATACseq', "
                    "'RNAseq', 'Visium (no probes)', 'CODEX'. hubmap_list_dataset_types() lists "
                    "them with counts"
                ),
                "default": None,
            },
            {
                "name": "query",
                "type": "str",
                "description": (
                    "Free-text search over dataset titles and descriptions, e.g. 'spatial'. "
                    "Combines with the organ and assay filters"
                ),
                "default": None,
            },
            {
                "name": "status",
                "type": "str",
                "description": (
                    "Release status. HuBMAP has exactly two, 'Published' (the default) and "
                    "'Retracted'; pass 'any' for both"
                ),
                "default": "Published",
            },
            {
                "name": "limit",
                "type": "int",
                "description": (
                    "Maximum datasets to return. The reported total is separate and may be an "
                    "'at least' estimate, since the index stops counting at 10,000"
                ),
                "default": 10,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Fetch one HuBMAP dataset's full record: title, assay, organ with its ontology term, "
            "donor demographics, contributing group and named contributors, DOI, release status "
            "and the files it contains. Use it after hubmap_search_datasets to judge whether a "
            "dataset is worth downloading. Print the returned dict."
        ),
        "name": "hubmap_get_dataset",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "hubmap_id",
                "type": "str",
                "description": (
                    "HuBMAP id like 'HBM375.NKBD.938' or a 32-character entity uuid. "
                    "hubmap_search_datasets returns both for every hit"
                ),
            },
        ],
    },
    {
        "description": (
            "Trace one HuBMAP dataset back to the donor it came from: the chain of sample entities "
            "between them -- suspension, section, block, organ -- each with its own id and kind, "
            "plus the donor's demographics. Use it to judge comparability, since two datasets "
            "sharing a donor or a tissue block are not independent samples. The result says "
            "whether the chain is 'exact' (each step's real parent) or 'approximate' (ordered by "
            "entity kind because the parent links were incomplete), so an inferred order is never "
            "presented as a measured one. Print the returned dict."
        ),
        "name": "hubmap_get_dataset_provenance",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "entity_id",
                "type": "str",
                "description": (
                    "HuBMAP id like 'HBM375.NKBD.938' or a 32-character entity uuid, for a "
                    "dataset or any sample in its lineage"
                ),
            },
        ],
    },
]
