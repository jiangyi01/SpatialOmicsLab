"""Expression atlases: Human Protein Atlas, GTEx and EBI Expression Atlas.

"Where is this gene expressed?" is the question a spatial-omics result most often has to be checked
against. A cluster is called *cardiomyocyte* because of a handful of markers; a niche is called
*perivascular* because of a few more. Before that claim is worth writing down, somebody has to ask
whether those genes really are restricted to that tissue, that cell type, that region -- in an
independent, orthogonal dataset. These three services are the standard answers, and each is the right
one for a different question:

* **Human Protein Atlas** (``www.proteinatlas.org``) -- the broadest single record per gene:
  bulk-tissue and single-cell RNA, immunohistochemistry-based protein annotation, subcellular
  location, cell lines, cancer cohorts and prognostics. Use it for "what is known about this gene,
  across every modality, in one object", and for protein-level evidence that RNA atlases cannot give.
* **GTEx** (``gtexportal.org``) -- 54 post-mortem tissue sites across ~1,000 donors, with the
  genetics attached: median and per-sample expression, eQTLs, sQTLs, fine-mapping. Use it for
  quantitative cross-tissue comparison and for "does genotype move this gene's expression here".
* **EBI Expression Atlas** (``www.ebi.ac.uk/gxa``) -- a catalogue of curated *experiments*, baseline
  and differential, across species. Use it to find a dataset to reuse, not to look a gene up.

Two properties of these services shape every function below, and both were measured rather than
assumed.

**HPA silently drops a column code it does not recognise.** ``search_download.php`` answers HTTP 200
with the column simply absent, so an invalid code and a gene with genuinely no data produce
byte-identical responses. Eighteen codes that upstream advertises or uses were measured to be exactly
that (see ``_HPA_RETIRED_COLUMNS``); this module refuses them up front and names the working
replacement, rather than passing them on and returning an empty answer that reads like a fact.

**HPA returns its columns in its own canonical order, not the order you asked for.** A code therefore
cannot be mapped to a label by position. Nothing in this module hard-codes a code-to-label table:
every label, unit and value is read out of the live response. What *is* shipped is the measured list
of valid **codes** (591 of them, ``_HPA_*_MEMBERS``), because those are stable identifiers and are
what lets a wrong source name be caught with a suggestion instead of an empty result.

**GTEx requires a GENCODE-versioned gene id, and the version differs per dataset.** ``TP53`` is
``ENSG00000141510.16`` in ``gtex_v8`` (GENCODE v26) and ``.18`` in ``gtex_v10`` (v39); query with the
wrong suffix and the answer is an empty HTTP 200 that reads as "no data for this gene". Every GTEx
function here resolves a symbol or bare Ensembl id against the target dataset first, and says in its
result which id it actually used.

Every outbound call goes through :mod:`spatialomicsgym.utils.http_client`, which enforces HTTPS, an
explicit host allowlist, one shared connection pool, a timeout and bounded retry. No function here
calls ``requests`` directly, and no function here reads an API key -- all three services answer
anonymously (measured 2026-09-17).

Return shape, uniform across the module: ``{"status": "success", "data": ..., ...}`` on success and
``{"status": "error", "error": "<what went wrong and what to do about it>"}`` on failure. Nothing
raises for an expected failure -- a network problem, an outage or a bad argument comes back as a
value, so one failed lookup inside a longer script does not abort the rest of it. The agent loop
recognises ``"status": "error"`` as a failed action, so a failure is never silently read as data.

These functions return a dict; ``print()`` the result (or the part of it you need) or it will not
appear in the observation.

------------------------------------------------------------------------------------------------
Adapted from ToolUniverse -- https://github.com/mims-harvard/ToolUniverse -- at commit
``f075c2a75e8b35ae5dbb220d48d4e87e980388b1``. Copyright [2025] [ToolUniverse team], licensed under
the Apache License, Version 2.0.

CHANGED BY SPATIALOMICSGYM, as Apache-2.0 section 4(b) requires this file to state. The endpoint
knowledge -- which HPA mechanism answers which question, GTEx's nineteen v2 paths, the per-dataset
GENCODE map, and the Expression Atlas JSON routes -- is upstream's; the code is not. Specifically:

* the ``BaseTool``/``register_tool``/config-driven dispatch machinery was not vendored, and each
  upstream tool is re-expressed here as a plain function;
* raw ``requests`` calls were replaced by our HTTP layer, which adds the host allowlist, the HTTPS
  floor and bounded retry that upstream's per-file ``timeout=30`` did not provide;
* upstream's ``result_type`` sub-mode switches (one tool serving two endpoints via a string
  argument) are split into separate named functions, because a tool the retriever can only rank as
  one entry is a tool the model picks for the wrong half of its behaviour;
* **F16** -- ``HPASearchTool`` advertises 21 column codes to the model. Measured individually against
  the live endpoint, **13 do not exist**: ``e u en r p c pt ptm s rnat rnablm rnabrm rnascm``. The
  working equivalents are ``rnascsm``, ``rnabcsm``, ``rnabrsm`` and ``rnabrs``. Because an unknown
  code returns HTTP 200 with the column absent, a model following the advertised list receives an
  answer it cannot distinguish from "no data". Here the retired codes are rejected by name with the
  replacement given, and every response reports the labels it actually came back with;
* **F17** -- ``HPAGetContextualBiologicalProcessTool``'s cell-line branch maps to four codes
  (``cell_RNA_hela``, ``cell_RNA_mcf7``, ``cell_RNA_a549``, ``cell_RNA_hepg2``) that **all** measure
  as non-existent, so that branch always banded 0, always concluded "not expressed", and always
  reported the gene "likely not functionally relevant" in that cell line. Five further advertised
  cell lines had no mapping at all, and ``blood_cells``/``brain_regions`` validated as legal contexts
  with no lookup path. Re-expressed against the measured ``cell_RNA_*`` catalogue, and a context with
  no data now says so instead of asserting absence;
* **F18** -- three upstream tools use ``rnatsm`` ("RNA tissue specific nTPM") as if it were a tissue
  panel. It is an enrichment *summary*: for ``MAPT`` it is ``{"brain": "147.9", "skeletal muscle":
  "83.6"}`` and for ``TP53`` it is ``null``. Two of those tools compute fold changes against it, so
  for any gene without tissue enrichment -- which is most genes -- the comparison had no denominator.
  Here tissue panels come from the ``t_RNA_*`` codes, which are the actual per-tissue columns;
* **F19** -- ``valid_contexts`` spells ``cerebral_cortex`` while HPA's own keys read ``cerebral
  cortex``, and the substring test ran against the unnormalised string, so the tool's own recommended
  spelling could never match. Matching here is normalised on both sides;
* **F20** -- ``_extract_cell_line_expression`` appends a record only ``if cell_info["expression_data"]``,
  which is unconditionally falsy at that point, so the list was always empty; and
  ``_extract_antibodies`` fabricates a placeholder entry, so ``total_antibodies`` could never be 0
  even for a gene with no antibodies. Both corrected;
* **F21** -- ``HPA_get_gene_tsv_data_by_ensembl_id`` (implemented by a class named
  ``HPAGetGeneXMLTool``, and returning neither TSV nor XML) requests seven columns, two of which are
  the non-existent ``cell_RNA_a549`` and ``cell_RNA_hela``. It returns five keys and promises seven;
* **F22** -- GTEx's ``/metadata/dataset`` lists four datasets and ``gtex_snrnaseq_pilot`` is not
  among them, yet it is the only one that answers the single-nucleus endpoints with data. Upstream's
  catalogue already documents that much. What is added here is the trap it does not mention: the same
  call with ``datasetId=gtex_v10`` returns **HTTP 200 with zero rows**, so "use the newer release"
  silently produces an empty answer rather than an error;
* **F23** -- ``expression_atlas_tool.py`` computes a ``gene_mentioned`` flag by text-searching
  experiment *descriptions*, documents in three separate comments that this is "not real per-gene
  filtering", and then uses it as the **primary sort key** (``:310-312`` and ``:419-421``). The one
  text-coincidence experiment is ranked first, above the warning explaining that the coincidence
  means nothing; and when no experiment matches, the key degenerates and the ordering silently
  becomes assay count, so the same call ranks by two different criteria depending on something the
  caller cannot see. We neither compute the flag nor sort by it: ranking is assay count, stated;
* **F24** -- the baseline-experiment tools take a ``gene`` argument that cannot affect the result
  (upstream says so in its own docstring, and the GXA ``/json/experiments`` route ignores a
  ``geneQuery`` filter -- identical counts with and without, and the per-experiment route ignores
  it too, measured in four syntaxes). It is dropped from the signature here; ``condition``
  free-text-searches the same field, so no capability is lost, and the functions point at GTEx and
  HPA for questions that are actually per-gene;
* **F25** -- ``HPAGetBiologicalProcessTool``'s ``filter_processes`` argument never filters. It
  removes nothing from the returned process list; it only populates a second list of matches against
  a hard-coded seven-term watchlist, so a caller who passes it to narrow the answer gets the same
  answer plus a side list. Renamed ``highlight_processes`` here, with the watchlist an argument whose
  default is stated, and the full list returned either way;
* **F26** -- ``_get_experiment`` reads the *catalogue* record's schema out of the *per-experiment*
  response. It asks ``data["experiment"]`` for ``experimentalFactors``, ``technologyType``,
  ``contrasts``, ``numberOfAssays``, ``lastUpdate`` and ``pubmedIds``, none of which that object
  carries -- it has only ``accession``, ``type``, ``species``, ``description`` and ``urls`` -- so six
  fields came back empty every time and the endpoint was written up as exposing no design data. It
  does: the design is under ``columnHeaders`` (assay groups with factor values, ontology term ids and
  replicate counts for a baseline experiment; named contrasts with both sides for a differential
  one), and ``profiles`` carries a page of the actual gene table with its unit. Both are returned
  here. Upstream's HTTP-404 branch is also unreachable, because the service answers an unknown
  accession with **400**; that status is mapped to an unknown-accession message here;
* **F27** -- ``result_type="summary"`` on the single-nucleus tool requires a gene argument for a call
  that provably ignores it: ``/expression/singleNucleusGeneExpressionSummary`` returns the same 114
  rows for ``TP53``, for a nonsense GENCODE id and for no gene at all, while honouring a tissue
  filter (15 rows for ``Lung``). It is a dataset cell census, not a gene result, so the split-out
  function is named ``gtex_get_single_nucleus_cell_counts`` and takes no gene;
* ``HPA_get_protein_interactions_by_gene`` is **not** vendored (decision D-012). Its column code
  ``ppi`` measures as non-existent; the ``interactions`` column that does exist is an integer count
  (``TP53`` 998, ``EGFR`` 926), not a partner list; and upstream's own catalogue marks the tool
  deprecated in favour of STRING. The count is still reachable through ``hpa_search_columns``.
"""

from __future__ import annotations

import difflib
import logging
import re
from urllib.parse import quote

from spatialomicsgym.utils.http_client import HttpError, request_json

logger = logging.getLogger(__name__)

# RL-1 provenance. Every host this module may reach, with the evidence its origin was established
# from. Nothing outside this tuple is reachable: http_client refuses any other host, and refuses
# plain HTTP, before a request leaves the process.
#
#   www.proteinatlas.org  Human Protein Atlas. Run by the Knut and Alice Wallenberg Foundation
#                         programme hosted at KTH Royal Institute of Technology, Stockholm, Sweden.
#                         Swedish .org, Swedish funder, Swedish host institution.
#   gtexportal.org        GTEx Portal. NIH Common Fund programme; the portal is operated by the
#                         Broad Institute (Cambridge, Massachusetts, USA) under NHGRI funding.
#   www.ebi.ac.uk         European Bioinformatics Institute, Hinxton, UK -- an EMBL outstation.
#                         Already cleared and allowlisted for this tree by module 3.
_ALLOWED_HOSTS = ("www.proteinatlas.org", "gtexportal.org", "www.ebi.ac.uk")

_HPA_SEARCH_URL = "https://www.proteinatlas.org/api/search_download.php"
_HPA_RECORD_URL = "https://www.proteinatlas.org/{ensembl_id}.json"
_GTEX_BASE = "https://gtexportal.org/api/v2"
_GXA_BASE = "https://www.ebi.ac.uk/gxa"

# The Expression Atlas catalogue is one 2.6 MB document with no server-side filter, and the
# 30 s house default is not enough for it on a cold connection.
_GXA_CATALOG_TIMEOUT = 90.0

_DEFAULT_MAX_ROWS = 10
_MAX_ROWS_CEILING = 200
_DEFAULT_PAGE_SIZE = 100
_MAX_PAGE_SIZE = 250

_WS_RE = re.compile(r"\s+")
_ENSEMBL_RE = re.compile(r"^ENSG\d{11}$")
#: A GENCODE-versioned Ensembl gene id, the one input whose '.' is a version separator.
_GTEX_VERSIONED_ID_RE = re.compile(r"^ENSG\d{11}\.\d+(_PAR_Y)?$", re.IGNORECASE)

# Column codes upstream advertises or requests that the live endpoint does not honour. Each was
# measured one at a time against `search=TP53`, paired with a known-good sentinel column, so
# "absent" is a measurement and not an inference. The value is what to use instead.
_HPA_RETIRED_COLUMNS = {
    "e": "use 'eg' for the Ensembl id",
    "u": "use 'up' for the UniProt accession",
    "en": "use 'eg' for the Ensembl id",
    "r": "use 'rnats' (RNA tissue specificity) or a 't_RNA_<tissue>' column",
    "p": "use 'pe' for protein evidence, or 'pc' for protein class",
    "c": "use 'chr' for the chromosome",
    "pt": "use 'prts' for protein tissue specificity",
    "ptm": "no equivalent column exists; HPA exposes no post-translational modification column",
    "s": "use 'scl', 'scml' or 'scal' for subcellular location",
    "rnat": "use 'rnats' for RNA tissue specificity, or 't_RNA_<tissue>' for values",
    "rnablm": "use 'rnabcsm' (blood cell) or 'rnabcs' for the specificity category",
    "rnabrm": "use 'rnabrsm' (brain regional specific nTPM) or 'rnabrs' for the category",
    "rnascm": "use 'rnascsm' (single cell type specific nCPM) or 'rnascs' for the category",
    "cell_RNA_hela": "HeLa is not a separate column; use 'cell_RNA_cervical_cancer'",
    "cell_RNA_mcf7": "MCF7 is not a separate column; use 'cell_RNA_breast_cancer'",
    "cell_RNA_a549": "A549 is not a separate column; use 'cell_RNA_lung_cancer'",
    "cell_RNA_hepg2": "HepG2 is not a separate column; use 'cell_RNA_liver_cancer'",
    "ppi": "no interaction-partner column exists; 'interactions' returns a count, not partners",
}

# The measured member lists for every HPA column family. These are shipped as *codes* only --
# never as labels, because HPA returns columns in its own canonical order and a label cannot be
# mapped to a code by position (see the module docstring). Codes are stable identifiers, which is
# what lets a wrong source name be caught with a suggestion instead of an empty HTTP 200.
#
# `_tau` in a family is that family's TAU specificity score, not a tissue or cell type. It is a
# valid code and is kept in the validation set, but it is not a member of the panel it sits in.
# t_RNA_<member> -- bulk tissue RNA, consensus nTPM (52 members, measured)
_HPA_TISSUE_MEMBERS = (
    "_tau",
    "adipose_tissue",
    "adrenal_gland",
    "amygdala",
    "appendix",
    "basal_ganglia",
    "blood_vessel",
    "bone_marrow",
    "breast",
    "cerebellum",
    "cerebral_cortex",
    "cervix",
    "choroid_plexus",
    "colon",
    "duodenum",
    "endometrium_1",
    "epididymis",
    "esophagus",
    "fallopian_tube",
    "gallbladder",
    "heart_muscle",
    "hippocampal_formation",
    "hypothalamus",
    "kidney",
    "liver",
    "lung",
    "lymph_node",
    "midbrain",
    "ovary",
    "pancreas",
    "parathyroid_gland",
    "pituitary_gland",
    "placenta",
    "prostate",
    "rectum",
    "retina",
    "salivary_gland",
    "seminal_vesicle",
    "skeletal_muscle",
    "skin_1",
    "small_intestine",
    "smooth_muscle",
    "spinal_cord",
    "spleen",
    "stomach_1",
    "testis",
    "thymus",
    "thyroid_gland",
    "tongue",
    "tonsil",
    "urinary_bladder",
    "vagina",
)
# blood_RNA_<member> -- blood immune cell RNA, nTPM (20 members, measured)
_HPA_BLOOD_MEMBERS = (
    "MAIT_T-cell",
    "NK-cell",
    "T-reg",
    "_tau",
    "basophil",
    "classical_monocyte",
    "eosinophil",
    "gdT-cell",
    "intermediate_monocyte",
    "memory_B-cell",
    "memory_CD4_T-cell",
    "memory_CD8_T-cell",
    "myeloid_DC",
    "naive_B-cell",
    "naive_CD4_T-cell",
    "naive_CD8_T-cell",
    "neutrophil",
    "non-classical_monocyte",
    "plasmacytoid_DC",
    "total_PBMC",
)
# brain_RNA_<member> -- brain region RNA (bulk), nTPM (14 members, measured)
_HPA_BRAIN_MEMBERS = (
    "_tau",
    "amygdala",
    "basal_ganglia",
    "cerebellum",
    "cerebral_cortex",
    "choroid_plexus",
    "hippocampal_formation",
    "hypothalamus",
    "medulla_oblongata",
    "midbrain",
    "pons",
    "spinal_cord",
    "thalamus",
    "white_matter",
)
# Brain_sn_RNA_<member> -- brain single-nucleus RNA cell types, nTPM (35 members, measured)
_HPA_BRAIN_SN_MEMBERS = (
    "Bergmann_glia",
    "CGE_interneuron",
    "LAMP5-LHX6_and_Chandelier",
    "MGE_interneuron",
    "_tau",
    "amygdala_excitatory",
    "astrocyte",
    "central_nervous_system_macrophage",
    "cerebellar_inhibitory",
    "choroid_plexus_epithelial_cell",
    "committed_oligodendrocyte_precursor",
    "deep-layer_corticothalamic_and_6b",
    "deep-layer_intratelencephalic",
    "deep-layer_near-projecting",
    "eccentric_medium_spiny_neuron",
    "endothelial_cell",
    "ependymal_cell",
    "fibroblast",
    "hippocampal_CA1-3",
    "hippocampal_CA4",
    "hippocampal_dentate_gyrus",
    "leukocyte",
    "lower_rhombic_lip",
    "mammillary_body",
    "medium_spiny_neuron",
    "midbrain-derived_inhibitory",
    "miscellaneous",
    "oligodendrocyte",
    "oligodendrocyte_precursor_cell",
    "pericyte",
    "splatter",
    "thalamic_excitatory",
    "upper-layer_intratelencephalic",
    "upper_rhombic_lip",
    "vascular_associated_smooth_muscle_cell",
)
# sc_RNA_<member> -- single cell type RNA, nTPM (155 members, measured)
_HPA_SINGLE_CELL_MEMBERS = (
    "Adipocytes",
    "Adrenal_cortex_cells",
    "Adrenal_medulla_cells",
    "Alveolar_cells_type_1",
    "Alveolar_cells_type_2",
    "Astrocytes",
    "B-cells",
    "Basal_keratinocytes",
    "Basal_prostatic_cells",
    "Bergmann_glia",
    "Brain_excitatory_neurons",
    "Brain_inhibitory_neurons",
    "Breast_hormone-responsive_cells",
    "Breast_lactating_cells",
    "Breast_myoepithelial_cells",
    "Breast_secretory_cells",
    "Cardiomyocytes",
    "Cholangiocytes",
    "Choroid_plexus_epithelial_cells",
    "Colonocytes",
    "Cone_photoreceptor_cells",
    "Conjunctival_goblet_cells",
    "Corticotrophs",
    "Cytotrophoblasts",
    "Decidual_stromal_cells",
    "Differentiating_spermatogonia",
    "Distal_convoluted_tubule_cells",
    "Early_primary_spermatocytes",
    "Early_spermatids",
    "Endometrial_ciliated_cells",
    "Endometrial_glandular_cells",
    "Endometrial_luminal_cells",
    "Endometrial_secretory_cells",
    "Endometrial_stromal_cells",
    "Enteric_stem_cells",
    "Enteric_transient_amplifying_cells",
    "Enterocytes",
    "Ependymal_cells",
    "Epicardial_cells",
    "Epididymal_basal_cells",
    "Epididymal_clear_cells",
    "Epididymal_efferent_duct_absorptive_cells",
    "Epididymal_efferent_duct_ciliated_cells",
    "Epididymal_principal_cells",
    "Erythrocyte_progenitors",
    "Erythrocytes",
    "Esophageal_apical_cells",
    "Esophageal_basal_cells",
    "Esophageal_suprabasal_cells",
    "Extravillous_trophoblasts",
    "Fallopian_secretory_cells",
    "Fallopian_tube_ciliated_cells",
    "Fibro-adipogenic_progenitors",
    "Fibroblasts",
    "Foveolar_cells",
    "Gastric_chief_cells",
    "Gastric_progenitor_cells",
    "Goblet_cells",
    "Gonadotrophs",
    "Granulosa_cells",
    "Hematopoietic_stem_cells",
    "Hepatic_stellate_cells",
    "Hepatocytes",
    "Hofbauer_cells",
    "Innate_lymphoid_cells",
    "Kupffer_cells",
    "Lacrimal_acinar_cells",
    "Lactotrophs",
    "Late_primary_spermatocytes",
    "Late_spermatids",
    "Leydig_cells",
    "Loop_of_henle_epithelial_cells",
    "Lymphatic_endothelial_cells",
    "Macrophages",
    "Mast_cells",
    "Medullary_thymic_epithelial_cells",
    "Megakaryocyte-Erythroid_progenitors",
    "Megakaryocyte_progenitors",
    "Megakaryocytes",
    "Melanocytes",
    "Mesothelial_cells",
    "Microglia",
    "Migrating_cytotrophoblasts",
    "Monocyte_progenitors",
    "Mucous_neck_cells",
    "Myonuclei",
    "Myosatellite_cells",
    "Müller_glia",
    "NK-cells",
    "Neuroendocrine_cells",
    "Neutrophil_progenitors",
    "Neutrophils",
    "Ocular_epithelial_cells",
    "Oligodendrocyte_progenitor_cells",
    "Oligodendrocytes",
    "Oocytes",
    "Other_brain_neurons",
    "Ovarian_stromal_cells",
    "Pancreatic_acinar_cells",
    "Pancreatic_duct_cells",
    "Pancreatic_islet_cells",
    "Paneth_cells",
    "Papillary_tip_epithelial_cells",
    "Parietal_cells",
    "Pericytes",
    "Peritubular_myoid_cells",
    "Pituicytes/FSCs",
    "Pituitary_stem_cells",
    "Plasma_cells",
    "Platelets",
    "Podocytes",
    "Prostatic_club_cells",
    "Prostatic_glandular_cells",
    "Prostatic_hillock_cells",
    "Proximal_tubule_cells",
    "Renal_collecting_duct_intercalated_cells",
    "Renal_collecting_duct_principal_cells",
    "Renal_connecting_tubule_cells",
    "Respiratory_basal_cells",
    "Respiratory_ciliated_cells",
    "Respiratory_deuterosomal_cells",
    "Respiratory_ionocytes",
    "Respiratory_secretory_cells",
    "Retinal_amacrine_cells",
    "Retinal_bipolar_cells",
    "Retinal_ganglion_cells",
    "Retinal_horizontal_cells",
    "Retinal_pigment_epithelial_cells",
    "Rod_photoreceptor_cells",
    "Salivary_acinar_cells",
    "Salivary_basal_cells",
    "Salivary_duct_cells",
    "Salivary_ionocytes",
    "Salivary_myoepithelial_cells",
    "Schwann_cells",
    "Sertoli_cells",
    "Smooth_muscle_cells",
    "Somatotrophs",
    "Submucosal_glandular_cells",
    "Suprabasal_keratinocytes",
    "Syncytiotrophoblasts",
    "T-cells",
    "Thymic_myoid_cells",
    "Thymocytes",
    "Thyrotrophs",
    "Transitional_alveolar_cells",
    "Tuft_cells",
    "Undifferentiated_spermatogonia",
    "Urothelial_cells",
    "Vascular_endothelial_cells",
    "Vascular_smooth_muscle_cells",
    "_tau",
    "cDC",
    "monocytes",
    "pDCs",
)
# cell_RNA_<member> -- cell line RNA grouped by cancer type, nTPM (30 members, measured)
_HPA_CELL_LINE_MEMBERS = (
    "adrenocortical_cancer",
    "bile_duct_cancer",
    "bladder_cancer",
    "bone_cancer",
    "brain_cancer",
    "breast_cancer",
    "cervical_cancer",
    "colorectal_cancer",
    "esophageal_cancer",
    "gallbladder_cancer",
    "gastric_cancer",
    "head_and_neck_cancer",
    "kidney_cancer",
    "leukemia",
    "liver_cancer",
    "lung_cancer",
    "lymphoma",
    "myeloma",
    "neuroblastoma",
    "non-cancerous",
    "ovarian_cancer",
    "pancreatic_cancer",
    "prostate_cancer",
    "rhabdoid",
    "sarcoma",
    "skin_cancer",
    "testis_cancer",
    "thyroid_cancer",
    "uncategorized",
    "uterine_cancer",
)
# t_APE_<member> -- tissue protein expression, IHC annotation score (59 members, measured)
_HPA_TISSUE_PROTEIN_MEMBERS = (
    "adipose_tissue",
    "adrenal_gland",
    "appendix",
    "bone_marrow",
    "breast",
    "bronchus",
    "brown_adipose_tissue",
    "cartilage",
    "caudate",
    "cerebellum",
    "cerebral_cortex",
    "cervix",
    "choroid_plexus",
    "colon",
    "dorsal_raphe",
    "duodenum",
    "efferent_ducts",
    "endometrium",
    "epididymis",
    "esophagus",
    "eye",
    "fallopian_tube",
    "gallbladder",
    "hair",
    "heart_muscle",
    "hippocampus",
    "hypothalamus",
    "kidney",
    "lactating_breast",
    "liver",
    "lung",
    "lymph_node",
    "nasopharynx",
    "oral_mucosa",
    "ovary",
    "pancreas",
    "parathyroid_gland",
    "pituitary_gland",
    "placenta",
    "prostate",
    "rectum",
    "retina",
    "salivary_gland",
    "seminal_vesicle",
    "skeletal_muscle",
    "skin",
    "small_intestine",
    "smooth_muscle",
    "soft_tissue",
    "sole_of_foot",
    "spleen",
    "stomach",
    "substantia_nigra",
    "testis",
    "thymus",
    "thyroid_gland",
    "tonsil",
    "urinary_bladder",
    "vagina",
)
# ct_APE_<member> -- cell type protein expression, IHC annotation score (59 members, measured)
_HPA_CELL_TYPE_PROTEIN_MEMBERS = (
    "adipose_tissue",
    "adrenal_gland",
    "appendix",
    "bone_marrow",
    "breast",
    "bronchus",
    "brown_adipose_tissue",
    "cartilage",
    "caudate",
    "cerebellum",
    "cerebral_cortex",
    "cervix",
    "choroid_plexus",
    "colon",
    "dorsal_raphe",
    "duodenum",
    "efferent_ducts",
    "endometrium",
    "epididymis",
    "esophagus",
    "eye",
    "fallopian_tube",
    "gallbladder",
    "hair",
    "heart_muscle",
    "hippocampus",
    "hypothalamus",
    "kidney",
    "lactating_breast",
    "liver",
    "lung",
    "lymph_node",
    "nasopharynx",
    "oral_mucosa",
    "ovary",
    "pancreas",
    "parathyroid_gland",
    "pituitary_gland",
    "placenta",
    "prostate",
    "rectum",
    "retina",
    "salivary_gland",
    "seminal_vesicle",
    "skeletal_muscle",
    "skin",
    "small_intestine",
    "smooth_muscle",
    "soft_tissue",
    "sole_of_foot",
    "spleen",
    "stomach",
    "substantia_nigra",
    "testis",
    "thymus",
    "thyroid_gland",
    "tonsil",
    "urinary_bladder",
    "vagina",
)
# ct_dvp_<member> -- deep visual proteomics cell type, protein abundance (28 members, measured)
_HPA_DVP_MEMBERS = (
    "Alveolar_cells_type_1",
    "Alveolar_cells_type_2",
    "Astrocytes_and_Neuropil",
    "B-cells",
    "CD4_T-cells",
    "CD8_T-cells",
    "Capillaries",
    "Cardiomyocytes",
    "Ciliated_cells",
    "Collecting_ducts",
    "Distal_tubules",
    "Glomeruli",
    "Granulosa_cells",
    "Hepatocytes",
    "Keratinocytes",
    "Large_ducts",
    "Macrophages",
    "Microglia_and_Neuropil",
    "Neurons",
    "Neutrophils",
    "Oocytes",
    "Pancreas_epithelial_cells",
    "Pancreatic_islets",
    "Proximal_tubules",
    "Secretory_cells",
    "Skeletal_myofibers",
    "Smooth_muscle_cells",
    "_tau",
)
# t_ms_<member> -- tissue mass spectrometry, protein concentration (21 members, measured)
_HPA_MASS_SPEC_MEMBERS = (
    "Stomach",
    "_tau",
    "blood_vessel",
    "bone_marrow",
    "cerebral_cortex",
    "colon",
    "duodenum",
    "fallopian_tube",
    "heart_muscle",
    "ileum",
    "jejunum",
    "kidney",
    "liver",
    "lung",
    "lymph_node",
    "ovary",
    "pancreas",
    "salivary_gland",
    "skeletal_muscle",
    "skin",
    "spleen",
)
# prognostic_<member> -- cancer prognostic association, per TCGA/validation cohort (31 members, measured)
_HPA_PROGNOSTIC_MEMBERS = (
    "Bladder_Urothelial_Carcinoma_(TCGA)",
    "Breast_Invasive_Carcinoma_(TCGA)",
    "Breast_Invasive_Carcinoma_(validation)",
    "Cervical_Squamous_Cell_Carcinoma_and_Endocervical_Adenocarcinoma_(TCGA)",
    "Colon_Adenocarcinoma_(TCGA)",
    "Colon_Adenocarcinoma_(validation)",
    "Glioblastoma_Multiforme_(TCGA)",
    "Glioblastoma_Multiforme_(validation)",
    "Head_and_Neck_Squamous_Cell_Carcinoma_(TCGA)",
    "Kidney_Chromophobe_(TCGA)",
    "Kidney_Renal_Clear_Cell_Carcinoma_(TCGA)",
    "Kidney_Renal_Clear_Cell_Carcinoma_(validation)",
    "Kidney_Renal_Papillary_Cell_Carcinoma_(TCGA)",
    "Liver_Hepatocellular_Carcinoma_(TCGA)",
    "Liver_Hepatocellular_Carcinoma_(validation)",
    "Lung_Adenocarcinoma_(TCGA)",
    "Lung_Adenocarcinoma_(validation)",
    "Lung_Squamous_Cell_Carcinoma_(TCGA)",
    "Lung_Squamous_Cell_Carcinoma_(validation)",
    "Ovary_Serous_Cystadenocarcinoma_(TCGA)",
    "Ovary_Serous_Cystadenocarcinoma_(validation)",
    "Pancreatic_Adenocarcinoma_(TCGA)",
    "Pancreatic_Adenocarcinoma_(validation)",
    "Prostate_Adenocarcinoma_(TCGA)",
    "Rectum_Adenocarcinoma_(TCGA)",
    "Rectum_Adenocarcinoma_(validation)",
    "Skin_Cutaneous_Melanoma_(TCGA)",
    "Stomach_Adenocarcinoma_(TCGA)",
    "Testicular_Germ_Cell_Tumor_(TCGA)",
    "Thyroid_Carcinoma_(TCGA)",
    "Uterine_Corpus_Endometrial_Carcinoma_(TCGA)",
)

# source_type -> (column prefix, measured members, what the source is, what the values mean). Used by
# every tool that takes a source name, so a wrong name is caught before a request is spent and the
# caller is told which names exist rather than handed an empty result.
_HPA_SOURCES = {
    "tissue": ("t_RNA_", _HPA_TISSUE_MEMBERS, "bulk tissue RNA", "consensus nTPM"),
    "blood": ("blood_RNA_", _HPA_BLOOD_MEMBERS, "blood immune cell RNA", "nTPM"),
    "brain": ("brain_RNA_", _HPA_BRAIN_MEMBERS, "brain region RNA (bulk)", "nTPM"),
    "brain_single_nucleus": ("Brain_sn_RNA_", _HPA_BRAIN_SN_MEMBERS, "brain single-nucleus RNA cell types", "nTPM"),
    "single_cell": ("sc_RNA_", _HPA_SINGLE_CELL_MEMBERS, "single cell type RNA", "nTPM"),
    "cell_line": (
        "cell_RNA_",
        _HPA_CELL_LINE_MEMBERS,
        "cell line RNA, grouped by the cancer type the lines come from",
        "nTPM",
    ),
    "tissue_protein": ("t_APE_", _HPA_TISSUE_PROTEIN_MEMBERS, "tissue protein expression (IHC)", "annotation score"),
    "cell_type_protein": (
        "ct_APE_",
        _HPA_CELL_TYPE_PROTEIN_MEMBERS,
        "cell type protein expression (IHC)",
        "annotation score",
    ),
    "dvp": ("ct_dvp_", _HPA_DVP_MEMBERS, "deep visual proteomics cell types", "protein abundance"),
    "mass_spec": ("t_ms_", _HPA_MASS_SPEC_MEMBERS, "tissue mass spectrometry", "protein concentration"),
    "prognostic": ("prognostic_", _HPA_PROGNOSTIC_MEMBERS, "cancer prognostic association", "p-value and direction"),
}

# A cell-line group holds many lines: `cell_RNA_lung_cancer` expands to 232 columns, not one. The
# expansion is a measured property of the endpoint, which is why a cell-line request can return
# hundreds of keys from a single code.
_HPA_CELL_LINE_NOTE = (
    "each cell_RNA_ code is a *group* of cell lines, not one line -- 'lung_cancer' returns 232 columns"
)

# Derived once, so the validator and the suggester cannot drift from the catalogue above.
_HPA_FAMILY_BY_PREFIX = {prefix: (members, what, units) for prefix, members, what, units in _HPA_SOURCES.values()}
_HPA_ALL_FAMILY_CODES = tuple(
    prefix + member for prefix, (members, _w, _u) in _HPA_FAMILY_BY_PREFIX.items() for member in members
)

# GTEx annotates each dataset against a different GENCODE release and only matches ids carrying that
# release's suffix. Reported by /metadata/dataset; re-verified live 2026-09-17.
_GTEX_GENCODE_VERSION = {
    "gtex_v7": "v19",
    "gtex_v8": "v26",
    "gtex_v10": "v39",
    "gtex_snrnaseq_pilot": "v26",
    "kids_first_harmonization": "v26",
}
_GTEX_DEFAULT_DATASET = "gtex_v8"
_GTEX_SNRNASEQ_DATASET = "gtex_snrnaseq_pilot"

# The 87 single-value HPA columns, measured. Anything outside these plus the family codes above is
# not a column this endpoint knows, and asking for it returns HTTP 200 with the column absent.
_HPA_SCALAR_COLUMNS = frozenset(
    (
        "ab",
        "abrr",
        "blconcia",
        "blconcms",
        "ccdp",
        "ccdt",
        "chr",
        "chrp",
        "di",
        "ecblood",
        "ecbrain",
        "eccellline",
        "ecsinglecell",
        "ectissue",
        "eg",
        "evih",
        "evin",
        "eviu",
        "g",
        "gd",
        "gs",
        "interactions",
        "pc",
        "pe",
        "prctd",
        "prcts",
        "prctsm",
        "prctss",
        "prtd",
        "prts",
        "prtsm",
        "prtss",
        "relce",
        "relih",
        "relmb",
        "rnabcd",
        "rnabcs",
        "rnabcsm",
        "rnabcss",
        "rnabld",
        "rnabls",
        "rnablsm",
        "rnablss",
        "rnabrd",
        "rnabrs",
        "rnabrsm",
        "rnabrss",
        "rnacad",
        "rnacas",
        "rnacasm",
        "rnacass",
        "rnacld",
        "rnacls",
        "rnaclsm",
        "rnaclss",
        "rnambrd",
        "rnambrs",
        "rnambrsm",
        "rnambrss",
        "rnapbrd",
        "rnapbrs",
        "rnapbrsm",
        "rnapbrss",
        "rnascd",
        "rnascgd",
        "rnascgs",
        "rnascgsm",
        "rnascgss",
        "rnascs",
        "rnascsm",
        "rnascss",
        "rnasnbd",
        "rnasnbs",
        "rnasnbsm",
        "rnasnbss",
        "rnatd",
        "rnats",
        "rnatsm",
        "rnatss",
        "rtcte",
        "scal",
        "scl",
        "scml",
        "secl",
        "up",
        "up_mf",
        "upbp",
    )
)

# The only two response labels this module names. They are the labels for the identity columns `g`
# and `eg`, measured live, and they exist so a measurement column can be found by set difference --
# everything that is not identity is a measurement, read with whatever label HPA returned. If HPA
# ever renames them the split degrades to "more measurement keys", never to a wrong number.
_HPA_IDENTITY_LABELS = ("Gene", "Ensembl")

# Ported from upstream, which got this right: HPA publishes the number, not a band, so any band
# we print has to say it is ours. The cut-offs are upstream's; the single vocabulary is not --
# upstream carried two (one of them feeding a prose "functional relevance" verdict), and F17 means
# that second vocabulary was banding zeros anyway.
_HPA_BANDS = ((50.0, "very high"), (10.0, "high"), (1.0, "medium"), (0.1, "low"))
_HPA_BAND_FLOOR = "very low"
_HPA_BAND_BASIS = (
    "level is computed here from the numeric value, and only for nTPM values -- a protein score, "
    "concentration or abundance, or a p-value, keeps its number with level null. HPA publishes the "
    "number only and does not report this classification. Cut-offs: >50 very high, >10 high, "
    ">1 medium, >0.1 low, <=0.1 very low. Recompute or ignore the banding as your analysis requires."
)

#: The member name of a family's TAU specificity score (see the note above the member lists).
_HPA_TAU = "_tau"

# The watchlist upstream's HPA_get_biological_process_by_gene highlights. Kept as the *default* for
# `highlight_processes` rather than as a hidden filter -- see the F25 note on that function.
_HPA_DEFAULT_PROCESS_HIGHLIGHTS = (
    "Apoptosis",
    "Biological rhythms",
    "Cell cycle",
    "Host-virus interaction",
    "Necrosis",
    "Transcription",
    "Transcription regulation",
)


def _error(message, **extra):
    """An error payload in the shape ``execution.py:_EXEC_ERROR_RE`` recognises as a failed action."""
    payload = {"status": "error", "error": message}
    payload.update(extra)
    return payload


def _ok(data, **extra):
    payload = {"status": "success", "data": data}
    payload.update(extra)
    return payload


def _missing(name, hint, *, plural=False):
    return _error(f"{name} {'are' if plural else 'is'} required. {hint}")


def _seg(value):
    """Percent-encode a caller-supplied value going into a URL *path* segment."""
    return quote(str(value), safe="")


def _page_size(size, default=_DEFAULT_PAGE_SIZE, maximum=_MAX_PAGE_SIZE):
    try:
        return max(1, min(int(size), maximum))
    except (TypeError, ValueError):
        return default


def _as_int(value, default=0):
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value):
    """HPA and GTEx both return numbers as strings in places, and 'N/A' where there is no value."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean(text):
    return _WS_RE.sub(" ", str(text or "")).strip()


def _norm(text):
    """Fold a tissue or cell-type name for matching.

    HPA's own spellings disagree with each other -- the column code is ``t_RNA_cerebral_cortex``
    while the label reads ``cerebral cortex`` -- so every comparison in this module normalises both
    sides. F19 is exactly the bug that happens when only one side is folded.
    """
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")


def _suggest(value, candidates, limit=5):
    """Closest known names for a value that did not match, so a typo is one hop from correct."""
    folded = {_norm(c): c for c in candidates if c != _HPA_TAU}
    close = difflib.get_close_matches(_norm(value), list(folded), n=limit, cutoff=0.6)
    if not close:
        close = [k for k in folded if _norm(value) and _norm(value) in k][:limit]
    return [folded[k] for k in close]


def _band(value, units="nTPM"):
    """``(level, numeric_value)``; ``(None, None)`` when the value is not a number.

    The cut-offs are nTPM cut-offs. A mass-spec concentration, a DVP abundance or a prognostic
    p-value was banded on that scale too, so 80 of anything read "very high" (hunt 2026-09-30,
    uT3-atlases-13); outside nTPM the number is kept and the level is None.
    """
    number = _as_float(value)
    if number is None:
        return None, None
    if "ntpm" not in str(units).lower():
        return None, number
    for cutoff, name in _HPA_BANDS:
        if number > cutoff:
            return name, number
    return _HPA_BAND_FLOOR, number


def _fetch_json(url, **kwargs):
    """``(payload, None)`` on success, ``(None, error_payload)`` on failure."""
    try:
        return request_json(url, allowed_hosts=_ALLOWED_HOSTS, **kwargs), None
    except HttpError as exc:
        return None, _error(exc.detail, retryable=bool(exc.status) and exc.status >= 500)


# --------------------------------------------------------------------------------------------- HPA


def _hpa_check_codes(codes):
    """``(codes, None)`` when every code is one the endpoint honours, else ``(None, error)``.

    This is the whole point of shipping the measured catalogue. ``search_download.php`` answers an
    unknown column with HTTP 200 and the column simply missing, which is byte-identical to a gene
    with no data -- so a code that does not exist produces a confident, empty, wrong answer. Caught
    here, before a request is spent.
    """
    retired, unknown = [], []
    for code in codes:
        if code in _HPA_RETIRED_COLUMNS:
            retired.append(f"'{code}' ({_HPA_RETIRED_COLUMNS[code]})")
            continue
        if code in _HPA_SCALAR_COLUMNS:
            continue
        for prefix, (members, _what, _units) in _HPA_FAMILY_BY_PREFIX.items():
            if code.startswith(prefix) and code[len(prefix) :] in members:
                break
        else:
            unknown.append(code)
    if retired:
        return None, _error(
            "These HPA column codes no longer exist: "
            + "; ".join(retired)
            + ". HPA answers an unknown column with HTTP 200 and the column absent, so asking anyway "
            "returns an empty result that reads like 'no data for this gene'."
        )
    if unknown:
        hints = []
        for code in unknown[:3]:
            close = _suggest(code, sorted(_HPA_SCALAR_COLUMNS) + list(_HPA_ALL_FAMILY_CODES))
            if close:
                hints.append(f"'{code}' -- did you mean {', '.join(close)}?")
        return None, _error(
            "Unknown HPA column code(s): "
            + ", ".join(unknown)
            + ". "
            + (" ".join(hints) if hints else "")
            + " Use hpa_list_columns() to see every code this endpoint honours."
        )
    return list(codes), None


def _hpa_rows(search_term, codes):
    """One ``search_download.php`` call. ``(rows, None)`` or ``(None, error)``."""
    checked, err = _hpa_check_codes(codes)
    if err:
        return None, err
    payload, err = _fetch_json(
        _HPA_SEARCH_URL,
        params={
            "search": search_term,
            "format": "json",
            "columns": ",".join(checked),
            "compress": "no",
        },
    )
    if err:
        return None, err
    if not isinstance(payload, list):
        return None, _error(
            f"HPA returned {type(payload).__name__} where a list of rows was expected. "
            "The search endpoint may be in maintenance; retry, or use hpa_get_gene_summary() which "
            "reads the per-gene record instead."
        )
    return payload, None


def _hpa_term(gene):
    """``gene`` as HPA is searched for it: a versioned Ensembl id loses its suffix.

    ``ENSG00000141510.16`` (how GTEx and many count matrices write it) matched no row, because the
    folded ``_16`` never equals HPA's bare id (hunt 2026-09-30, uT3-atlases-6).
    """
    text = _clean(gene)
    base = text.split(".")[0].upper()
    return base if _ENSEMBL_RE.match(base) else text


def _hpa_is_gene(row, wanted):
    """Is ``row`` the gene ``wanted`` names, by symbol or Ensembl id -- not by a synonym or text hit?"""
    folded = _norm(_hpa_term(wanted))
    return bool(folded) and folded in (_norm((row or {}).get("Gene")), _norm((row or {}).get("Ensembl")))


def _hpa_substitution(requested, row):
    """``{}`` when ``row`` is the gene asked for, else the fields that say it is another one.

    ``_hpa_pick_row`` falls back to HPA's top search hit so that a synonym (HER2) still reaches its
    gene (ERBB2). That fallback was silent: a lncRNA, a clone name or a typo that some other gene's
    text mentions came back as status success carrying that other gene's numbers (hunt 2026-09-30,
    uT3-atlases-6).
    """
    if not row or _hpa_is_gene(row, requested):
        return {}
    requested = _clean(requested)
    return {
        "gene_requested": requested,
        "gene_match_note": (
            f"HPA has no gene whose symbol or Ensembl id is '{requested}'. These values are for "
            f"{row.get('Gene') or 'an unnamed row'} ({row.get('Ensembl') or 'no Ensembl id'}), the top "
            "hit of HPA's full-text search, which also matches synonyms and descriptions. Confirm it "
            f"is the gene you meant with hpa_search_genes('{requested}') before using these numbers."
        ),
    }


def _hpa_pick_row(rows, wanted):
    """The row whose gene symbol matches ``wanted``, else the first row.

    A bare search matches synonyms and descriptions, so ``search=TP53`` returns TP53 among others
    and the first row is not reliably the gene asked for. When nothing matches, the first row is
    returned and :func:`_hpa_substitution` is what tells the caller so.
    """
    folded = _norm(_hpa_term(wanted))
    for row in rows:
        if _norm(row.get("Gene")) == folded:
            return row
    for row in rows:
        if _norm(row.get("Ensembl")) == folded:
            return row
    return rows[0] if rows else None


def _hpa_split(row):
    """``(identity, measurements)`` -- identity is the two named id columns, the rest is data.

    Measurements keep whatever label HPA returned them under. Nothing here maps a code to a label.
    """
    identity = {k: row[k] for k in _HPA_IDENTITY_LABELS if k in row}
    measurements = {k: v for k, v in row.items() if k not in identity}
    return identity, measurements


def _hpa_resolve(gene, *, extra_codes=()):
    """``(ensembl_id, symbol, row, None)`` or ``(None, None, None, error)``.

    An Ensembl id is used as given and costs no request; a symbol is looked up.
    """
    text = _clean(gene)
    if not text:
        return (
            None,
            None,
            None,
            _missing("gene", "Pass a gene symbol ('TP53') or an Ensembl gene id ('ENSG00000141510')."),
        )
    codes = ["g", "eg", *extra_codes]
    if _ENSEMBL_RE.match(text.split(".")[0].upper()):
        base = text.split(".")[0].upper()
        if not extra_codes:
            return base, None, None, None
        rows, err = _hpa_rows(base, codes)
        if err:
            return None, None, None, err
        row = _hpa_pick_row(rows or [], base)
        if not row:
            # An empty search came back as no error and no row, which the caller then reported as
            # a gene with nothing annotated (hunt 2026-09-30, uT3-atlases-12).
            return (
                None,
                None,
                None,
                _error(
                    f"HPA has no gene matching '{base}'. The id is well formed, so it is most likely a "
                    "non-coding gene or a retired Ensembl id; confirm the current id with hpa_search_genes()."
                ),
            )
        return base, row.get("Gene"), row, None
    rows, err = _hpa_rows(text, codes)
    if err:
        return None, None, None, err
    row = _hpa_pick_row(rows or [], text)
    if not row:
        return (
            None,
            None,
            None,
            _error(
                f"HPA has no gene matching '{text}'. Check the symbol, or search with "
                "hpa_search_genes() which returns the Ensembl ids of near matches."
            ),
        )
    ensembl = _clean(row.get("Ensembl")) or None
    if not ensembl:
        return (
            None,
            None,
            None,
            _error(
                f"HPA matched '{text}' but returned no Ensembl id for it, so per-gene record lookups "
                "are not possible for this row."
            ),
        )
    return ensembl, _clean(row.get("Gene")) or text, row, None


def _hpa_record(ensembl_id):
    """The per-gene JSON record: ~119 labelled fields covering every modality HPA carries."""
    text = _clean(ensembl_id).split(".")[0].upper()
    if not _ENSEMBL_RE.match(text):
        return None, _error(
            f"'{ensembl_id}' is not an Ensembl gene id. This endpoint is keyed by Ensembl id "
            "(ENSG00000141510); pass a symbol to hpa_search_genes() to get one."
        )
    try:
        payload = request_json(_HPA_RECORD_URL.format(ensembl_id=_seg(text)), allowed_hosts=_ALLOWED_HOSTS)
    except HttpError as exc:
        # A well-formed id HPA does not carry comes back as a bare 404, whose message names a URL
        # and nothing the caller can act on. HPA covers protein-coding genes, so the usual cause is
        # a non-coding or retired id rather than a typo, and that is a different fix. Decided on the
        # status: the message embeds the URL, so testing it for "404" made a 503 or a timeout for
        # an id like ENSG00000140443 (IGF1R) read as "no record" and lose its retryable flag
        # (hunt 2026-09-30, uT3-atlases-5).
        if exc.status == 404:
            return None, _error(
                f"The Human Protein Atlas has no record for {text}. The id is well formed, so it "
                "is most likely a non-coding gene, a retired Ensembl id, or a gene from another "
                "species -- HPA covers human genes with protein-level evidence. Confirm the "
                "current id with hpa_search_genes()."
            )
        return None, _error(exc.detail, retryable=bool(exc.status) and exc.status >= 500)
    if not isinstance(payload, dict):
        return None, _error(f"HPA returned no record for {text}.")
    return payload, None


def _hpa_source(source_type):
    key = _norm(source_type)
    if key in _HPA_SOURCES:
        return key, _HPA_SOURCES[key], None
    return None, None, _error(f"Unknown source_type '{source_type}'. Available: {', '.join(sorted(_HPA_SOURCES))}.")


def _hpa_members_for(prefix, members, names):
    """Resolve caller-supplied source names to column codes, normalising both sides (F19)."""
    codes, resolved, unmatched = [], [], []
    # `_tau` is the family's TAU specificity score, not a member: resolving "tau" reported that score
    # as an expression value, banded as one (hunt 2026-09-30, uT3-atlases-13).
    lookup = {_norm(m): m for m in members if m != _HPA_TAU}
    for name in names:
        member = lookup.get(_norm(name))
        if member is None:
            unmatched.append(name)
            continue
        codes.append(prefix + member)
        resolved.append(member)
    return codes, resolved, unmatched


def _hpa_tau_hint(names):
    """The sentence an error needs when a caller asked a panel for its TAU score by name."""
    if not any(_norm(n) == _norm(_HPA_TAU) for n in names):
        return ""
    return (
        " '_tau' is a family's TAU tissue-specificity score, not a member of the panel and not an "
        "expression value; request it by code with hpa_search_columns(), e.g. columns=['t_RNA__tau']."
    )


def hpa_list_columns(family=None):
    """List the Human Protein Atlas column codes that actually exist, without spending a request.

    Not a vendored upstream tool -- an adaptation. Upstream advertises a fixed list of 21 codes to
    the model of which 13 do not exist (F16), and there is no HPA endpoint that enumerates valid
    columns, so the measured catalogue shipped with this module is the only way to find out what can
    be asked for. Every code returned here was confirmed one at a time against the live endpoint.

    Parameters
    ----------
    family : str, optional
        Narrow to one column family. One of the ``source_type`` names
        (``tissue``, ``blood``, ``brain``, ``brain_single_nucleus``, ``single_cell``, ``cell_line``,
        ``tissue_protein``, ``cell_type_protein``, ``dvp``, ``mass_spec``, ``prognostic``), or
        ``"scalar"`` for the 87 single-value columns. Default lists every family's size plus the
        scalar codes.

    Returns
    -------
    dict
        ``{"status": "success", "data": {...}}``. ``print()`` it or it will not be observed.

    Examples
    --------
    >>> print(hpa_list_columns())  # doctest: +SKIP
    >>> print(hpa_list_columns(family="blood"))  # doctest: +SKIP
    """
    if family is None:
        return _ok(
            {
                "families": {
                    name: {
                        "prefix": prefix,
                        "members": len(members),
                        "measures": what,
                        "units": units,
                        "example_code": prefix + members[0],
                    }
                    for name, (prefix, members, what, units) in sorted(_HPA_SOURCES.items())
                },
                "scalar_columns": sorted(_HPA_SCALAR_COLUMNS),
                "retired_columns": dict(sorted(_HPA_RETIRED_COLUMNS.items())),
                "total_codes": len(_HPA_ALL_FAMILY_CODES) + len(_HPA_SCALAR_COLUMNS),
            },
            note=(
                "Codes are stable; the labels HPA returns them under are not positional -- read "
                "labels from the response, never from the request order. " + _HPA_CELL_LINE_NOTE
            ),
        )
    key = _norm(family)
    if key == "scalar":
        return _ok({"family": "scalar", "codes": sorted(_HPA_SCALAR_COLUMNS)})
    resolved, source, err = _hpa_source(key)
    if err:
        return err
    prefix, members, what, units = source
    return _ok(
        {
            "family": resolved,
            "prefix": prefix,
            "measures": what,
            "units": units,
            "codes": [prefix + m for m in members],
            "members": list(members),
        },
        note=_HPA_CELL_LINE_NOTE if prefix == "cell_RNA_" else None,
    )


def hpa_search_genes(query, max_results=10):
    """Find genes in the Human Protein Atlas and get the Ensembl ids everything else is keyed by.

    The usual first call: HPA's per-gene record, tissue panels and prognostics are all addressed by
    Ensembl gene id, and this is how a symbol, a synonym or a free-text description becomes one.

    Parameters
    ----------
    query : str
        Gene symbol, synonym, Ensembl id, or free text ('kinase', 'tumor suppressor'). HPA searches
        symbols, synonyms and descriptions, so its own order does not put the symbol first; hits are
        re-ranked here (exact symbol or Ensembl match, then symbol prefix, then the rest in HPA's
        order) and the exact match, when there is one, is also reported separately.
    max_results : int, optional
        How many rows to return, 1-200. Default 10.

    Returns
    -------
    dict
        ``data`` carries ``matches`` (each with the live labels HPA returned), ``exact_match`` when
        one row's symbol equals the query, and ``total_matched``.

    Examples
    --------
    >>> hits = hpa_search_genes("TP53")  # doctest: +SKIP
    >>> print(hits["data"]["exact_match"])  # doctest: +SKIP
    >>> print(hpa_search_genes("aquaporin", max_results=5))  # doctest: +SKIP
    """
    text = _clean(query)
    if not text:
        return _missing("query", "Pass a gene symbol, an Ensembl id, or free text to search for.")
    limit = _page_size(max_results, default=_DEFAULT_MAX_ROWS, maximum=_MAX_ROWS_CEILING)
    # 'gs' and the ranking are what the tool description promises: synonyms in each hit, exact and
    # prefix symbol matches first. HPA's own order put 'INS' somewhere among thousands of text hits,
    # past the max_results cut (hunt 2026-09-30, uT3-atlases-8).
    rows, err = _hpa_rows(text, ["g", "eg", "gs", "gd", "pe", "rnats", "scml"])
    if err:
        return err
    if not rows:
        return _ok(
            {"query": text, "matches": [], "exact_match": None, "total_matched": 0},
            note="HPA matched nothing. Symbols are case-insensitive but must be current HUGO names.",
        )
    folded = _norm(text)
    rows = sorted(
        rows,
        key=lambda row: 0 if _hpa_is_gene(row, text) else 1 if _norm(row.get("Gene")).startswith(folded) else 2,
    )
    exact = None
    for row in rows:
        if _norm(row.get("Gene")) == _norm(text):
            exact = row
            break
    return _ok(
        {
            "query": text,
            "total_matched": len(rows),
            "returned": min(limit, len(rows)),
            "exact_match": exact,
            "matches": rows[:limit],
        },
        note=(
            "Ranked exact symbol or Ensembl match first, then symbols starting with the query, then "
            "HPA's other text hits in HPA's order. Field labels are HPA's own, read from this response. "
            "Use the 'Ensembl' value with hpa_get_gene_summary(), hpa_get_cancer_prognostics() or "
            "hpa_get_tissue_rna_expression()."
        ),
    )


def hpa_search_columns(query, columns=None, max_rows=10):
    """Ask the Human Protein Atlas for any columns it has, by code.

    The escape hatch under the typed functions: anything in the 591-code catalogue can be requested
    here in one call. Codes are validated against the measured catalogue first, because HPA answers
    an unknown column with HTTP 200 and the column missing -- indistinguishable from no data.

    Parameters
    ----------
    query : str
        What to search for: a gene symbol, an Ensembl id, or free text.
    columns : list of str, optional
        Column codes, e.g. ``["g", "eg", "t_RNA_liver", "t_RNA_lung"]``. Call
        ``hpa_list_columns()`` to see every code. Default ``["g", "eg", "gd", "rnats"]``.
    max_rows : int, optional
        How many rows to return, 1-200. Default 10.

    Returns
    -------
    dict
        ``data.rows`` as returned, plus ``labels_returned`` -- the labels HPA actually used, which
        are not in request order and are the only labels you should read values by.

    Examples
    --------
    >>> print(hpa_search_columns("TP53", columns=["g", "eg", "interactions"]))  # doctest: +SKIP
    >>> print(hpa_search_columns("GFAP", columns=["g", "brain_RNA_cerebellum"]))  # doctest: +SKIP
    """
    text = _clean(query)
    if not text:
        return _missing("query", "Pass a gene symbol, an Ensembl id, or free text to search for.")
    codes = list(columns) if columns else ["g", "eg", "gd", "rnats"]
    if not codes:
        return _missing("columns", "Pass at least one column code, or omit it for the default set.", plural=True)
    limit = _page_size(max_rows, default=_DEFAULT_MAX_ROWS, maximum=_MAX_ROWS_CEILING)
    rows, err = _hpa_rows(text, codes)
    if err:
        return err
    labels = sorted({label for row in rows for label in row})
    return _ok(
        {
            "query": text,
            "columns_requested": codes,
            "labels_returned": labels,
            "total_matched": len(rows),
            "rows": rows[:limit],
        },
        note=(
            "A grouped code expands to many columns (cell_RNA_lung_cancer returns 232), so more "
            "labels than codes is expected. Labels are not in request order."
        ),
    )


def hpa_get_gene_summary(ensembl_id):
    """One-screen summary of a gene from the Human Protein Atlas per-gene record.

    Identity, what the gene is, where its protein sits, and the headline specificity calls for
    tissue, blood, brain and single cell -- enough to decide whether a marker claim is plausible
    before spending further calls.

    Parameters
    ----------
    ensembl_id : str
        Ensembl gene id, e.g. ``"ENSG00000141510"``. A version suffix is stripped. Get one from
        :func:`hpa_search_genes`.

    Returns
    -------
    dict
        ``data.summary`` with the fields that were present, and ``data.fields_absent`` naming any
        the record did not carry -- so a schema change is visible rather than silent.

    Examples
    --------
    >>> print(hpa_get_gene_summary("ENSG00000141510"))  # doctest: +SKIP
    """
    record, err = _hpa_record(ensembl_id)
    if err:
        return err
    # "Position" is in the description's "chromosome and position" and was never asked for
    # (hunt 2026-09-30, uT3-atlases-8).
    wanted = (
        "Gene",
        "Gene synonym",
        "Ensembl",
        "Gene description",
        "Uniprot",
        "Chromosome",
        "Position",
        "Protein class",
        "Evidence",
        "Subcellular location",
        "Subcellular main location",
        "RNA tissue specificity",
        "RNA tissue distribution",
        "RNA single cell type specificity",
        "RNA blood cell specificity",
        "RNA brain regional specificity",
        "Secretome location",
        "Interactions",
        "Biological process",
        "Molecular function",
        "Disease involvement",
    )
    present = {k: record[k] for k in wanted if k in record}
    return _ok(
        {
            "ensembl_id": _clean(ensembl_id).split(".")[0].upper(),
            "summary": present,
            "fields_absent": [k for k in wanted if k not in record],
            "record_fields_total": len(record),
        },
        note=(
            "'Interactions' is a count, not a partner list -- HPA exposes no partner column. "
            "Tissue *panels* are not in this record; use hpa_get_tissue_rna_expression()."
        ),
    )


def hpa_get_gene_annotation(ensembl_id):
    """Functional annotation for a gene: protein class, processes, molecular function, disease.

    The "what is this gene for" half of the HPA record, separated from expression so an annotation
    question does not return several hundred numbers alongside it.

    Parameters
    ----------
    ensembl_id : str
        Ensembl gene id, e.g. ``"ENSG00000141510"``.

    Returns
    -------
    dict
        ``data.annotation`` keyed by HPA's own field labels, plus ``fields_absent``.

    Examples
    --------
    >>> print(hpa_get_gene_annotation("ENSG00000141510"))  # doctest: +SKIP
    """
    record, err = _hpa_record(ensembl_id)
    if err:
        return err
    wanted = (
        "Gene",
        "Gene description",
        "Protein class",
        "Biological process",
        "Molecular function",
        "Disease involvement",
        "Evidence",
        "Antibody",
        "Antibody RRID",
        "Uniprot",
        "Subcellular location",
        "Subcellular main location",
        "Subcellular additional location",
        "Secretome location",
        "Secretome function",
        "Molecular function (UniProt)",
    )
    present = {k: record[k] for k in wanted if k in record}
    return _ok(
        {
            "ensembl_id": _clean(ensembl_id).split(".")[0].upper(),
            "annotation": present,
            "fields_absent": [k for k in wanted if k not in record],
        }
    )


def hpa_get_gene_details(ensembl_id, include_expression=True, include_antibodies=True, include_prognostics=True):
    """The fuller Human Protein Atlas picture for one gene, from a single record fetch.

    Everything the per-gene record carries, grouped and toggleable. One HTTP request regardless of
    which sections are asked for.

    Immunohistochemistry *images* are deliberately not included: they live on a separate XML
    endpoint that returns megabytes per gene and yields image URLs a text agent cannot read.
    Antibody identity -- which is the part that is citable -- is in the record and is returned here.

    Parameters
    ----------
    ensembl_id : str
        Ensembl gene id, e.g. ``"ENSG00000141510"``.
    include_expression : bool, optional
        Include the expression-cluster and specificity fields. Default True.
    include_antibodies : bool, optional
        Include antibody identifiers and RRIDs. Default True.
    include_prognostics : bool, optional
        Include the per-cohort cancer prognostic entries. Default True.

    Returns
    -------
    dict
        ``data`` with ``identity`` always present and ``expression``/``antibodies``/``prognostics``
        present when asked for, plus ``sections_empty`` naming any that the record had nothing for.

    Examples
    --------
    >>> print(hpa_get_gene_details("ENSG00000141510", include_prognostics=False))  # doctest: +SKIP
    """
    record, err = _hpa_record(ensembl_id)
    if err:
        return err
    identity = {
        k: record[k]
        for k in (
            "Gene",
            "Gene synonym",
            "Ensembl",
            "Gene description",
            "Uniprot",
            "Chromosome",
            "Position",
            "Protein class",
            "Evidence",
        )
        if k in record
    }
    data = {"ensembl_id": _clean(ensembl_id).split(".")[0].upper(), "identity": identity}
    empty = []
    if include_expression:
        section = {
            k: v
            for k, v in record.items()
            if (
                "specificity" in k.lower()
                or "distribution" in k.lower()
                or "expression cluster" in k.lower()
                or k.startswith("Subcellular")
            )
        }
        data["expression"] = section
        if not section:
            empty.append("expression")
    if include_antibodies:
        section = {k: record[k] for k in ("Antibody", "Antibody RRID") if k in record}
        data["antibodies"] = section
        if not any(section.values()):
            empty.append("antibodies")
    if include_prognostics:
        section = {k: v for k, v in record.items() if k.lower().startswith("cancer prognostics")}
        data["prognostics"] = section
        if not section:
            empty.append("prognostics")
    data["sections_empty"] = empty
    return _ok(
        data,
        note=(
            "Sections are selected by HPA's own field labels, read from this response. An empty "
            "section means the record carried nothing, not that the request failed."
        ),
    )


def hpa_get_subcellular_location(gene_name):
    """Where in the cell a protein has been observed, by immunofluorescence.

    Useful for sanity-checking a marker: a gene annotated as secreted or nuclear behaves differently
    in a spatial assay than a membrane protein does.

    Parameters
    ----------
    gene_name : str
        Gene symbol ('TP53') or Ensembl gene id. A symbol costs one extra lookup.

    Returns
    -------
    dict
        ``data`` with ``main_location``, ``additional_location``, ``location`` (the combined call)
        and ``reliability`` where the record carries it.

    Examples
    --------
    >>> print(hpa_get_subcellular_location("TP53"))  # doctest: +SKIP
    """
    ensembl, symbol, row, err = _hpa_resolve(gene_name)
    if err:
        return err
    record, err = _hpa_record(ensembl)
    if err:
        return err
    fields = {k: v for k, v in record.items() if k.startswith("Subcellular")}
    if not fields:
        return _ok(
            {"gene": symbol or gene_name, "ensembl_id": ensembl, "locations": {}, **_hpa_substitution(gene_name, row)},
            note="HPA carries no subcellular localisation for this gene (no validated antibody).",
        )
    return _ok(
        {
            "gene": symbol or record.get("Gene") or gene_name,
            "ensembl_id": ensembl,
            "locations": fields,
            "reliability": record.get("Reliability (IF)"),
            **_hpa_substitution(gene_name, row),
        }
    )


def hpa_get_biological_processes(gene_name, highlight_processes=None):
    """Curated biological processes a gene participates in, with an optional watchlist.

    The parameter is named for what it does. Upstream calls the equivalent argument
    ``filter_processes``, but it never removes anything from the process list (F25) -- it only
    populates a side list of hits against a hard-coded seven-term watchlist. Here the watchlist is
    an argument, its default is stated, and the full process list is returned either way.

    Parameters
    ----------
    gene_name : str
        Gene symbol or Ensembl gene id.
    highlight_processes : list of str, optional
        Terms to flag within the returned processes, matched case-insensitively as substrings.
        Default: Apoptosis, Biological rhythms, Cell cycle, Host-virus interaction, Necrosis,
        Transcription, Transcription regulation. Pass ``[]`` to highlight nothing.

    Returns
    -------
    dict
        ``data.processes`` (everything HPA lists) and ``data.highlighted`` (the watchlist hits).

    Examples
    --------
    >>> print(hpa_get_biological_processes("TP53"))  # doctest: +SKIP
    >>> print(hpa_get_biological_processes("BMAL1", highlight_processes=[]))  # doctest: +SKIP
    """
    ensembl, symbol, row, err = _hpa_resolve(gene_name, extra_codes=["upbp"])
    if err:
        return err
    _identity, measurements = _hpa_split(row or {})
    raw = next(iter(measurements.values()), None) if measurements else None
    label = next(iter(measurements), None) if measurements else None
    if isinstance(raw, list):
        processes = [_clean(p) for p in raw if _clean(p)]
    else:
        text = _clean(raw)
        processes = [] if text in ("", "N/A") else [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    watchlist = list(_HPA_DEFAULT_PROCESS_HIGHLIGHTS) if highlight_processes is None else list(highlight_processes)
    highlighted = [
        {"term": term, "matched": process}
        for process in processes
        for term in watchlist
        if term.lower() in process.lower()
    ]
    return _ok(
        {
            "gene": symbol or gene_name,
            "ensembl_id": ensembl,
            "source_label": label,
            "processes": processes,
            "total_processes": len(processes),
            "highlighted": highlighted,
            "watchlist": watchlist,
            **_hpa_substitution(gene_name, row),
        },
        note="'processes' is everything HPA lists; 'highlighted' is a view of it, not a filter of it.",
    )


def hpa_get_cancer_prognostics(ensembl_id):
    """Per-cohort cancer prognostic association for a gene, from TCGA and validation cohorts.

    HPA reports, for each of ~31 cancer cohorts, whether expression of this gene is associated with
    survival and in which direction. Reported as HPA reports it, with no significance call added.

    Parameters
    ----------
    ensembl_id : str
        Ensembl gene id, e.g. ``"ENSG00000141510"``.

    Returns
    -------
    dict
        ``data.cohorts`` keyed by HPA's cohort labels, plus ``data.cohorts_with_data``.

    Examples
    --------
    >>> print(hpa_get_cancer_prognostics("ENSG00000141510"))  # doctest: +SKIP
    """
    record, err = _hpa_record(ensembl_id)
    if err:
        return err
    cohorts = {k: v for k, v in record.items() if k.lower().startswith("cancer prognostics")}
    with_data = {k: v for k, v in cohorts.items() if v}
    if not cohorts:
        return _ok(
            {"ensembl_id": _clean(ensembl_id).split(".")[0].upper(), "cohorts": {}, "cohorts_with_data": 0},
            note="HPA carries no prognostic entries for this gene.",
        )
    return _ok(
        {
            "ensembl_id": _clean(ensembl_id).split(".")[0].upper(),
            "gene": record.get("Gene"),
            "cohorts": cohorts,
            "cohorts_total": len(cohorts),
            "cohorts_with_data": len(with_data),
        },
        note="Cohort labels and their contents are HPA's, read from this response.",
    )


def hpa_get_tissue_rna_expression(ensembl_id, tissue_names):
    """Consensus RNA expression (nTPM) for a gene across named bulk tissues.

    The real per-tissue panel, from the ``t_RNA_`` columns. Not to be confused with HPA's
    'RNA tissue specific nTPM' field, which is an enrichment *summary* and is null for most genes
    (F18) -- three upstream tools compute fold changes against it and therefore had no denominator.

    Parameters
    ----------
    ensembl_id : str
        Ensembl gene id or gene symbol.
    tissue_names : list of str
        Tissue names, e.g. ``["liver", "lung", "cerebral cortex"]``. Spelling is normalised, so
        ``"cerebral_cortex"`` and ``"cerebral cortex"`` both work. Call
        ``hpa_list_columns(family="tissue")`` for all 52.

    Returns
    -------
    dict
        ``data.expression`` as ``{live HPA label: {"value": float, "level": str}}``, plus
        ``tissues_unmatched`` with suggestions.

    Examples
    --------
    >>> print(hpa_get_tissue_rna_expression("ENSG00000141510", ["liver", "lung"]))  # doctest: +SKIP
    """
    names = [n for n in (tissue_names or []) if _clean(n)]
    if not names:
        return _missing(
            "tissue_names",
            "Pass one or more tissue names; hpa_list_columns(family='tissue') lists all 52.",
            plural=True,
        )
    prefix, members, what, units = _HPA_SOURCES["tissue"]
    codes, resolved, unmatched = _hpa_members_for(prefix, members, names)
    if not codes:
        return _error(
            f"None of {names} is an HPA tissue. "
            f"Closest: {', '.join(_suggest(names[0], members)) or 'no close match'}. "
            "hpa_list_columns(family='tissue') lists all 52." + _hpa_tau_hint(names)
        )
    rows, err = _hpa_rows(_hpa_term(ensembl_id), ["g", "eg", *codes])
    if err:
        return err
    row = _hpa_pick_row(rows or [], ensembl_id)
    if row is None:
        return _error(f"HPA has no gene matching '{ensembl_id}'.")
    identity, measurements = _hpa_split(row)
    expression = {}
    for label, value in measurements.items():
        level, number = _band(value, units)
        expression[label] = {"value": number, "raw": value, "level": level}
    return _ok(
        {
            "gene": identity.get("Gene"),
            "ensembl_id": identity.get("Ensembl"),
            **_hpa_substitution(ensembl_id, row),
            "measures": what,
            "units": units,
            "tissues_requested": names,
            "tissues_resolved": resolved,
            "tissues_unmatched": [{"name": n, "suggestions": _suggest(n, members)} for n in unmatched],
            "expression": expression,
        },
        note=_HPA_BAND_BASIS,
    )


def hpa_get_rna_expression_by_source(gene_name, source_type, source_names):
    """RNA expression for a gene from any Human Protein Atlas source: tissue, blood, brain, cells.

    One entry point over every expression family HPA publishes, so a question about immune cells and
    a question about brain regions are the same call with a different ``source_type``.

    Parameters
    ----------
    gene_name : str
        Gene symbol or Ensembl gene id.
    source_type : str
        One of ``tissue``, ``blood``, ``brain``, ``brain_single_nucleus``, ``single_cell``,
        ``cell_line``, ``tissue_protein``, ``cell_type_protein``, ``dvp``, ``mass_spec``.
        ``hpa_list_columns()`` describes each and what its values mean.
    source_names : list of str
        Names within that source, e.g. ``["neutrophil", "NK-cell"]`` for ``blood``.

    Returns
    -------
    dict
        ``data.expression`` keyed by the labels HPA returned, with the numeric value and the band.

    Examples
    --------
    >>> print(hpa_get_rna_expression_by_source("CD19", "blood", ["memory_B-cell"]))  # doctest: +SKIP
    >>> print(hpa_get_rna_expression_by_source("GFAP", "brain", ["cerebellum"]))  # doctest: +SKIP
    """
    resolved_type, source, err = _hpa_source(source_type)
    if err:
        return err
    prefix, members, what, units = source
    names = [n for n in (source_names or []) if _clean(n)]
    if not names:
        return _missing(
            "source_names",
            f"Pass one or more names from the '{resolved_type}' source; "
            f"hpa_list_columns(family='{resolved_type}') lists all {len(members)}.",
            plural=True,
        )
    codes, resolved, unmatched = _hpa_members_for(prefix, members, names)
    if not codes:
        return _error(
            f"None of {names} is a '{resolved_type}' source name. "
            f"Closest: {', '.join(_suggest(names[0], members)) or 'no close match'}." + _hpa_tau_hint(names)
        )
    rows, err = _hpa_rows(_hpa_term(gene_name), ["g", "eg", *codes])
    if err:
        return err
    row = _hpa_pick_row(rows or [], gene_name)
    if row is None:
        return _error(f"HPA has no gene matching '{gene_name}'.")
    identity, measurements = _hpa_split(row)
    expression = {}
    for label, value in measurements.items():
        level, number = _band(value, units)
        expression[label] = {"value": number, "raw": value, "level": level}
    return _ok(
        {
            "gene": identity.get("Gene"),
            "ensembl_id": identity.get("Ensembl"),
            **_hpa_substitution(gene_name, row),
            "source_type": resolved_type,
            "measures": what,
            "units": units,
            "names_requested": names,
            "names_resolved": resolved,
            "names_unmatched": [{"name": n, "suggestions": _suggest(n, members)} for n in unmatched],
            "columns_returned": len(expression),
            "expression": expression,
        },
        note=((_HPA_CELL_LINE_NOTE + ". ") if prefix == "cell_RNA_" else "") + _HPA_BAND_BASIS,
    )


def _hpa_panel(gene, codes, units):
    """``(identity, {label: {"value","raw","level"}}, None)`` for one set of column codes.

    Each panel is fetched in its own request. That is deliberate: two families in one request come
    back interleaved in HPA's canonical order, and nothing in the response says which family a label
    belongs to -- so keeping them in separate calls is the only way to know, without guessing.
    ``units`` is the family's, and decides whether a value is banded at all (see :func:`_band`).
    """
    rows, err = _hpa_rows(_hpa_term(gene), ["g", "eg", *codes])
    if err:
        return None, None, err
    row = _hpa_pick_row(rows or [], gene)
    if row is None:
        return None, None, _error(f"HPA has no gene matching '{gene}'.")
    identity, measurements = _hpa_split(row)
    banded = {}
    for label, value in measurements.items():
        level, number = _band(value, units)
        banded[label] = {"value": number, "raw": value, "level": level}
    return identity, banded, None


def _summarise(banded):
    """Count, median and the highest-valued label across a panel, ignoring non-numeric entries."""
    pairs = [(label, item["value"]) for label, item in banded.items() if item["value"] is not None]
    if not pairs:
        return {"measured": 0, "median": None, "max": None}
    values = sorted(v for _label, v in pairs)
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    top_label, top_value = max(pairs, key=lambda pair: pair[1])
    return {
        "measured": len(pairs),
        "median": median,
        "max": {"label": top_label, "value": top_value, "level": banded[top_label]["level"]},
    }


def hpa_compare_cell_line_to_tissue(gene_name, cell_line_group, tissue_names=None):
    """Compare a gene's RNA in cancer cell lines against the normal tissue panel.

    The "is this marker a cancer thing or a normal-tissue thing" check. Cell-line values come from
    the ``cell_RNA_`` family and normal tissue from ``t_RNA_``, in two separate requests so each
    number's provenance is known rather than inferred from a label.

    Note what this is *not* doing. Upstream's equivalents compare against HPA's
    'RNA tissue specific nTPM' field, which is an enrichment summary, null for most genes (F18) --
    so the comparison had no denominator for any gene without tissue enrichment. And upstream's
    contextual path requests four cell-line codes that do not exist (F17), banding a missing number
    as zero and concluding the gene was "not expressed" and "likely not functionally relevant".

    Parameters
    ----------
    gene_name : str
        Gene symbol or Ensembl gene id.
    cell_line_group : str
        A cancer type from the cell-line family, e.g. ``"lung_cancer"``. Each is a *group* of lines,
        not one line -- ``lung_cancer`` covers 232 of them. ``hpa_list_columns(family='cell_line')``
        lists all 30.
    tissue_names : list of str, optional
        Normal tissues to compare against. Default: every tissue in the panel (51 -- the family's
        ``_tau`` code is a specificity score, not a tissue).

    Returns
    -------
    dict
        ``data.cell_lines`` and ``data.tissues`` each summarised (count, median, highest), plus
        ``data.ratio`` -- cell-line median over tissue median, or None with a reason -- and, per
        line in ``data.cell_lines.values``, ``fold_vs_tissue_median`` (that line over the same
        tissue median) whenever the ratio is defined.

    Examples
    --------
    >>> print(hpa_compare_cell_line_to_tissue("EGFR", "lung_cancer"))  # doctest: +SKIP
    >>> print(hpa_compare_cell_line_to_tissue("TP53", "breast_cancer", ["breast"]))  # doctest: +SKIP
    """
    gene = _clean(gene_name)
    if not gene:
        return _missing("gene_name", "Pass a gene symbol or an Ensembl gene id.")
    cl_prefix, cl_members, cl_what, cl_units = _HPA_SOURCES["cell_line"]
    cl_codes, cl_resolved, _unmatched = _hpa_members_for(cl_prefix, cl_members, [cell_line_group])
    if not cl_codes:
        return _error(
            f"'{cell_line_group}' is not an HPA cell-line group. "
            f"Closest: {', '.join(_suggest(cell_line_group, cl_members)) or 'no close match'}. "
            "hpa_list_columns(family='cell_line') lists all 30."
        )
    t_prefix, t_members, _t_what, t_units = _HPA_SOURCES["tissue"]
    wanted = [n for n in (tissue_names or []) if _clean(n)] or [m for m in t_members if m != "_tau"]
    t_codes, t_resolved, t_unmatched = _hpa_members_for(t_prefix, t_members, wanted)
    if not t_codes:
        return _error(
            f"None of {wanted} is an HPA tissue. "
            f"Closest: {', '.join(_suggest(wanted[0], t_members)) or 'no close match'}." + _hpa_tau_hint(wanted)
        )
    identity, cell_lines, err = _hpa_panel(gene, cl_codes, cl_units)
    if err:
        return err
    _identity2, tissues, err = _hpa_panel(gene, t_codes, t_units)
    if err:
        return err
    cl_summary = _summarise(cell_lines)
    t_summary = _summarise(tissues)
    ratio, ratio_note = None, None
    if cl_summary["median"] is None or t_summary["median"] is None:
        ratio_note = "one side had no numeric values, so no ratio is defined"
    elif t_summary["median"] == 0:
        ratio_note = "tissue median is 0 nTPM, so a ratio would divide by zero"
    else:
        ratio = cl_summary["median"] / t_summary["median"]
        # The per-line fold change the tool description promises; only the group ratio was ever
        # computed (hunt 2026-09-30, uT3-atlases-9).
        for item in cell_lines.values():
            if item["value"] is not None:
                item["fold_vs_tissue_median"] = item["value"] / t_summary["median"]
    return _ok(
        {
            "gene": identity.get("Gene") or gene,
            "ensembl_id": identity.get("Ensembl"),
            **_hpa_substitution(gene, identity),
            "units": cl_units,
            "cell_lines": {
                "group": cl_resolved[0],
                "column_code": cl_codes[0],
                "measures": cl_what,
                **cl_summary,
                "values": cell_lines,
            },
            "tissues": {
                "requested": len(wanted),
                "resolved": t_resolved,
                "unmatched": [{"name": n, "suggestions": _suggest(n, t_members)} for n in t_unmatched],
                "units": t_units,
                **t_summary,
                "values": tissues,
            },
            "ratio": ratio,
            "ratio_undefined_because": ratio_note,
        },
        note=("Medians are over the columns HPA returned, computed here, not reported by HPA. " + _HPA_BAND_BASIS),
    )


def hpa_get_cell_line_expression(gene_name, cell_line_group, top_n=15):
    """Rank the individual cancer cell lines expressing a gene, within one cancer type.

    Answers "which lines should I actually use", which the group summary cannot. A ``cell_RNA_``
    code is a group selector: one code returns every line in that cancer type (232 for lung), and
    this returns them ranked with the full distribution summarised.

    Parameters
    ----------
    gene_name : str
        Gene symbol or Ensembl gene id.
    cell_line_group : str
        A cancer type, e.g. ``"breast_cancer"``. ``hpa_list_columns(family='cell_line')`` lists all 30.
    top_n : int, optional
        How many of the highest-expressing lines to return, 1-200. Default 15.

    Returns
    -------
    dict
        ``data.top_lines`` ranked high to low, plus counts, median and how many lines were measured.

    Examples
    --------
    >>> print(hpa_get_cell_line_expression("ERBB2", "breast_cancer", top_n=5))  # doctest: +SKIP
    """
    gene = _clean(gene_name)
    if not gene:
        return _missing("gene_name", "Pass a gene symbol or an Ensembl gene id.")
    prefix, members, what, units = _HPA_SOURCES["cell_line"]
    codes, resolved, _unmatched = _hpa_members_for(prefix, members, [cell_line_group])
    if not codes:
        return _error(
            f"'{cell_line_group}' is not an HPA cell-line group. "
            f"Closest: {', '.join(_suggest(cell_line_group, members)) or 'no close match'}. "
            "hpa_list_columns(family='cell_line') lists all 30."
        )
    limit = _page_size(top_n, default=15, maximum=_MAX_ROWS_CEILING)
    identity, banded, err = _hpa_panel(gene, codes, units)
    if err:
        return err
    ranked = sorted(
        ({"cell_line": label, **item} for label, item in banded.items() if item["value"] is not None),
        key=lambda item: item["value"],
        reverse=True,
    )
    summary = _summarise(banded)
    return _ok(
        {
            "gene": identity.get("Gene") or gene,
            "ensembl_id": identity.get("Ensembl"),
            **_hpa_substitution(gene, identity),
            "cell_line_group": resolved[0],
            "column_code": codes[0],
            "measures": what,
            "units": units,
            "lines_returned": len(banded),
            "lines_with_values": summary["measured"],
            "median": summary["median"],
            "top_lines": ranked[:limit],
            "lines_not_expressing": [
                item["cell_line"] for item in ranked if item["value"] is not None and item["value"] <= 0.1
            ][:limit],
        },
        note="Cell-line labels are HPA's, read from this response. " + _HPA_BAND_BASIS,
    )


def hpa_get_contextual_expression(gene_name, context_name):
    """Look up a gene's expression in a named context without knowing which HPA family holds it.

    'cerebellum' is a brain region and a bulk tissue; 'neutrophil' is a blood cell and a single cell
    type. This resolves the name across all eleven families, fetches every family that has it, and
    says which family each number came from.

    This is the tool upstream's contextual path was meant to be. That one validated a fixed list of
    contexts against a mapping that produced four non-existent column codes for cell lines (F17),
    left five advertised contexts with no lookup path at all, and compared its own recommended
    spelling ``cerebral_cortex`` against HPA's ``cerebral cortex`` without folding either side
    (F19) -- so the spelling it told the caller to use could never match.

    Parameters
    ----------
    gene_name : str
        Gene symbol or Ensembl gene id.
    context_name : str
        A tissue, brain region, blood cell, single cell type, cell-line group or DVP cell type.
        Spelling and separators are normalised on both sides.

    Returns
    -------
    dict
        ``data.found_in`` -- one entry per family that carries this context, each with its own
        values and units -- plus ``data.families_searched`` and suggestions when nothing matched.

    Examples
    --------
    >>> print(hpa_get_contextual_expression("GFAP", "cerebellum"))  # doctest: +SKIP
    >>> print(hpa_get_contextual_expression("CD8A", "T-reg"))  # doctest: +SKIP
    """
    gene = _clean(gene_name)
    if not gene:
        return _missing("gene_name", "Pass a gene symbol or an Ensembl gene id.")
    context = _clean(context_name)
    if not context:
        return _missing("context_name", "Pass a tissue, brain region, blood cell or cell type name.")
    hits = []
    for source_type, (prefix, members, what, units) in sorted(_HPA_SOURCES.items()):
        codes, resolved, _unmatched = _hpa_members_for(prefix, members, [context])
        if codes:
            hits.append((source_type, prefix, codes[0], resolved[0], what, units))
    if not hits:
        every = sorted({m for _p, members, _w, _u in _HPA_SOURCES.values() for m in members})
        return _error(
            f"'{context}' is not a context HPA carries. "
            f"Closest: {', '.join(_suggest(context, every)) or 'no close match'}. "
            "hpa_list_columns() lists every family and hpa_list_columns(family=...) its members."
            + _hpa_tau_hint([context])
        )
    found, identity = [], {}
    for source_type, _prefix, code, member, what, units in hits:
        ident, banded, err = _hpa_panel(gene, [code], units)
        if err:
            return err
        identity = identity or ident
        found.append(
            {
                "source_type": source_type,
                "context": member,
                "column_code": code,
                "measures": what,
                "units": units,
                "columns_returned": len(banded),
                "values": banded,
                "summary": _summarise(banded),
            }
        )
    return _ok(
        {
            "gene": identity.get("Gene") or gene,
            "ensembl_id": identity.get("Ensembl"),
            **_hpa_substitution(gene, identity),
            "context": context,
            "families_searched": len(_HPA_SOURCES),
            "families_matched": [entry["source_type"] for entry in found],
            "found_in": found,
        },
        note=(
            "One context can exist in several families measuring different things -- a bulk tissue "
            "nTPM and an IHC annotation score are not comparable numbers. " + _HPA_BAND_BASIS
        ),
    )


# -------------------------------------------------------------------------------------------- GTEx


def _gtex_get(path, params=None):
    """``(payload, None)`` or ``(None, error)`` for one GTEx v2 call."""
    return _fetch_json(_GTEX_BASE + path, params=params)


def _gtex_rows(payload):
    """``(rows, paging)`` from either GTEx response shape.

    ``/metadata/dataset`` answers with a bare list; everything else with
    ``{"data": [...], "paging_info": {...}}``. A few routes key the list by its own name instead.
    """
    if isinstance(payload, list):
        return payload, None
    if not isinstance(payload, dict):
        return [], None
    paging = payload.get("paging_info")
    if isinstance(payload.get("data"), list):
        return payload["data"], paging
    for key, value in payload.items():
        if key != "paging_info" and isinstance(value, list):
            return value, paging
    return [], paging


def _gtex_tissue(tissue_site_detail_id, dataset):
    """``(tissue, None)`` or ``(None, error)`` for one GTEx tissue id, checked against the release.

    Tissue ids are case-sensitive and underscore-separated, and GTEx answers a wrong one with a bare
    HTTP 422 whose body names no field. Every other GTEx parameter in this module is resolved before
    the call -- the dataset against the live release list, the gene against that dataset's GENCODE
    -- and this is the third, so a misspelt tissue says which spelling was meant instead of a status
    code and a URL.
    """
    text = _clean(tissue_site_detail_id)
    if not text:
        return None, _missing(
            "tissue_site_detail_id",
            "Pass an exact tissue id; gtex_list_tissue_sites() lists all 54.",
        )
    listing = gtex_list_tissue_sites(dataset)
    known = (listing.get("data") or {}).get("tissue_ids") or []
    if not known:
        return text, None
    exact = {t: t for t in known}
    if text in exact:
        return text, None
    folded = {_norm(t): t for t in known}
    if _norm(text) in folded:
        return folded[_norm(text)], None
    close = _suggest(text, known)
    return None, _error(
        f"'{tissue_site_detail_id}' is not a GTEx tissue id under {dataset}. "
        + (f"Closest: {', '.join(close)}. " if close else "")
        + "Ids are case-sensitive and underscore-separated ('Brain_Cortex', not 'brain cortex'); "
        "gtex_list_tissue_sites() lists all 54."
    )


def _gtex_dataset(dataset_id):
    key = _clean(dataset_id) or _GTEX_DEFAULT_DATASET
    if key not in _GTEX_GENCODE_VERSION:
        return None, _error(
            f"Unknown dataset_id '{dataset_id}'. Known: {', '.join(sorted(_GTEX_GENCODE_VERSION))}. "
            "gtex_list_datasets() reports which are currently served."
        )
    return key, None


def _gtex_resolve_gene(gene, dataset_id):
    """``(gencode_id, matched_gene, None)`` or ``(None, None, error)``.

    GTEx only matches a GENCODE id carrying the release suffix of the dataset being queried --
    TP53 is ``ENSG00000141510.16`` under gtex_v8 (GENCODE v26) and ``.18`` under gtex_v10 (v39).
    Query with the wrong suffix and GTEx answers HTTP 200 with zero rows, which reads as "this gene
    is not expressed" rather than "you asked wrongly". So every id is resolved against the target
    dataset first, and the resolved id is reported back in the result.
    """
    text = _clean(gene)
    if not text:
        return None, None, _missing("gene", "Pass a gene symbol ('TP53') or an Ensembl/GENCODE id ('ENSG00000141510').")
    version = _GTEX_GENCODE_VERSION[dataset_id]
    # Only a versioned Ensembl id loses its suffix. Cutting every input at its first '.' sent the
    # GENCODE clone-name symbols 10x var_names are full of (AL627309.1, RP11-34P13.7) as 'AL627309',
    # and GTEx answered "no such gene" for a gene it has (hunt 2026-09-30, uT3-atlases-11).
    base = text.split(".")[0] if _GTEX_VERSIONED_ID_RE.match(text) else text
    payload, err = _gtex_get("/reference/gene", {"geneId": base, "gencodeVersion": version})
    if err:
        return None, None, err
    rows, _paging = _gtex_rows(payload)
    if not rows:
        return (
            None,
            None,
            _error(
                f"GTEx has no gene '{text}' under {dataset_id} (GENCODE {version}). Check the symbol, or "
                "try another dataset -- each release is annotated against a different GENCODE version."
            ),
        )
    exact = next((r for r in rows if _norm(r.get("geneSymbol")) == _norm(base)), rows[0])
    gencode_id = _clean(exact.get("gencodeId"))
    if not gencode_id:
        return None, None, _error(f"GTEx matched '{text}' but returned no GENCODE id for it.")
    return gencode_id, exact, None


def gtex_list_datasets():
    """List the GTEx releases this API serves, with sample counts and GENCODE versions.

    Worth calling before anything else, because the dataset decides which GENCODE version a gene id
    must carry, and because of one trap: ``gtex_snrnaseq_pilot`` is **not** in this list yet is the
    only dataset the single-nucleus endpoints return data for. Asking those endpoints for
    ``gtex_v10`` instead returns HTTP 200 with zero rows rather than an error (F22).

    Returns
    -------
    dict
        ``data.datasets`` as GTEx reports them, plus ``data.gencode_version`` -- the release each
        dataset's gene ids must be annotated against.

    Examples
    --------
    >>> print(gtex_list_datasets())  # doctest: +SKIP
    """
    payload, err = _gtex_get("/metadata/dataset")
    if err:
        return err
    rows, _paging = _gtex_rows(payload)
    served = {_clean(r.get("datasetId")) for r in rows if isinstance(r, dict)}
    return _ok(
        {
            "datasets": rows,
            "datasets_served": sorted(served),
            "gencode_version": dict(sorted(_GTEX_GENCODE_VERSION.items())),
            "single_nucleus_dataset": _GTEX_SNRNASEQ_DATASET,
        },
        note=(
            f"'{_GTEX_SNRNASEQ_DATASET}' is not listed here but is the dataset the single-nucleus "
            "endpoints answer; other dataset ids return an empty 200 on those routes."
        ),
    )


def gtex_list_tissue_sites(dataset_id=_GTEX_DEFAULT_DATASET):
    """List GTEx's 54 tissue sites, with the exact ids every other GTEx call requires.

    Tissue ids are case-sensitive and underscore-separated (``"Brain_Cortex"``, not ``"brain
    cortex"``); a wrong one is answered with HTTP 422. This is where to get them right.

    Parameters
    ----------
    dataset_id : str, optional
        GTEx release. Default ``"gtex_v8"``.

    Returns
    -------
    dict
        ``data.tissues`` with ids, display names and per-tissue gene and eGene counts.

    Examples
    --------
    >>> print(gtex_list_tissue_sites())  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    payload, err = _gtex_get(
        "/dataset/tissueSiteDetail", {"datasetId": dataset, "page": 0, "itemsPerPage": _MAX_PAGE_SIZE}
    )
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    return _ok(
        {
            "dataset_id": dataset,
            "tissue_count": len(rows),
            "tissue_ids": [_clean(r.get("tissueSiteDetailId")) for r in rows if isinstance(r, dict)],
            "tissues": rows,
            "paging": paging,
        },
        note="Use tissueSiteDetailId verbatim; GTEx answers a wrong id with HTTP 422, not a guess.",
    )


def gtex_get_median_gene_expression(gene, tissue_site_detail_ids=None, dataset_id=_GTEX_DEFAULT_DATASET):
    """Median expression (TPM) of a gene across GTEx tissues.

    The standard cross-tissue profile: one number per tissue, so "is this liver-specific" has an
    answer with a denominator. Without tissue ids this returns the clustered view across all 54.

    Parameters
    ----------
    gene : str
        Gene symbol or Ensembl/GENCODE id. Resolved against the dataset's GENCODE version first.
    tissue_site_detail_ids : list of str, optional
        Tissue ids from :func:`gtex_list_tissue_sites`. Default: every tissue.
    dataset_id : str, optional
        GTEx release. Default ``"gtex_v8"``.

    Returns
    -------
    dict
        ``data.medians`` as GTEx returns them, plus ``data.gencode_id`` -- the id actually queried.

    Examples
    --------
    >>> print(gtex_get_median_gene_expression("TP53", ["Liver", "Lung"]))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    gencode_id, matched, err = _gtex_resolve_gene(gene, dataset)
    if err:
        return err
    tissues = [_clean(t) for t in (tissue_site_detail_ids or []) if _clean(t)]
    params = {"gencodeId": gencode_id, "datasetId": dataset, "page": 0, "itemsPerPage": _MAX_PAGE_SIZE}
    if tissues:
        params["tissueSiteDetailId"] = tissues
        path = "/expression/medianGeneExpression"
    else:
        # medianGeneExpression requires a tissue; the clustered route is the all-tissue equivalent.
        path = "/expression/clusteredMedianGeneExpression"
    payload, err = _gtex_get(path, params)
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    return _ok(
        {
            "gene": _clean((matched or {}).get("geneSymbol")) or _clean(gene),
            "gencode_id": gencode_id,
            "dataset_id": dataset,
            "gencode_version": _GTEX_GENCODE_VERSION[dataset],
            "route": path,
            "tissues_requested": tissues or "all",
            "medians": rows,
            "paging": paging,
        },
        note=(
            "Values are TPM medians as GTEx publishes them. gencode_id is the versioned id that was "
            "queried -- a different dataset needs a different suffix."
        ),
    )


def gtex_get_gene_expression(
    gene, tissue_site_detail_ids=None, dataset_id=_GTEX_DEFAULT_DATASET, page_size=_DEFAULT_PAGE_SIZE
):
    """Per-sample expression values for a gene, not just the median.

    Use when the spread matters -- a median of 5 TPM over a bimodal distribution is a different
    claim from a median of 5 over a tight one.

    Parameters
    ----------
    gene : str
        Gene symbol or Ensembl/GENCODE id.
    tissue_site_detail_ids : list of str, optional
        Tissue ids from :func:`gtex_list_tissue_sites`. Default: every tissue.
    dataset_id : str, optional
        GTEx release. Default ``"gtex_v8"``.
    page_size : int, optional
        Rows per page, 1-250. Default 100.

    Returns
    -------
    dict
        ``data.expression`` -- one entry per tissue, each carrying the per-sample value array.

    Examples
    --------
    >>> print(gtex_get_gene_expression("TP53", ["Liver"]))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    gencode_id, matched, err = _gtex_resolve_gene(gene, dataset)
    if err:
        return err
    params = {
        "gencodeId": gencode_id,
        "datasetId": dataset,
        "page": 0,
        "itemsPerPage": _page_size(page_size),
    }
    tissues = [_clean(t) for t in (tissue_site_detail_ids or []) if _clean(t)]
    if tissues:
        params["tissueSiteDetailId"] = tissues
    payload, err = _gtex_get("/expression/geneExpression", params)
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    return _ok(
        {
            "gene": _clean((matched or {}).get("geneSymbol")) or _clean(gene),
            "gencode_id": gencode_id,
            "dataset_id": dataset,
            "tissues_requested": tissues or "all",
            "expression": rows,
            "paging": paging,
        },
        note="Each row carries a 'data' array of per-sample values in the unit GTEx names alongside it.",
    )


def gtex_get_top_expressed_genes(
    tissue_site_detail_id, dataset_id=_GTEX_DEFAULT_DATASET, filter_mt_gene=True, page_size=50
):
    """The most highly expressed genes in one GTEx tissue.

    The reverse lookup: instead of asking where a gene is expressed, ask what a tissue expresses.
    Useful for checking whether a cluster's top markers are simply that tissue's top genes.

    Parameters
    ----------
    tissue_site_detail_id : str
        Exact tissue id, e.g. ``"Liver"``. See :func:`gtex_list_tissue_sites`.
    dataset_id : str, optional
        GTEx release. Default ``"gtex_v8"``.
    filter_mt_gene : bool, optional
        Exclude mitochondrial genes, which otherwise dominate. Default True.
    page_size : int, optional
        How many genes, 1-250. Default 50.

    Returns
    -------
    dict
        ``data.genes`` ranked by median expression.

    Examples
    --------
    >>> print(gtex_get_top_expressed_genes("Liver", page_size=10))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    tissue, err = _gtex_tissue(tissue_site_detail_id, dataset)
    if err:
        return err
    payload, err = _gtex_get(
        "/expression/topExpressedGene",
        {
            "tissueSiteDetailId": tissue,
            "datasetId": dataset,
            "filterMtGene": bool(filter_mt_gene),
            "page": 0,
            "itemsPerPage": _page_size(page_size, default=50),
        },
    )
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    return _ok(
        {
            "tissue_site_detail_id": tissue,
            "dataset_id": dataset,
            "mitochondrial_genes_excluded": bool(filter_mt_gene),
            "genes": rows,
            "paging": paging,
        }
    )


def gtex_get_median_transcript_expression(gene, tissue_site_detail_id=None, dataset_id=_GTEX_DEFAULT_DATASET):
    """Median expression per transcript isoform, rather than per gene.

    Matters when a gene's isoforms differ in function or in which probes detect them -- a
    gene-level number can hide an isoform switch entirely.

    Parameters
    ----------
    gene : str
        Gene symbol or Ensembl/GENCODE id.
    tissue_site_detail_id : str, optional
        One tissue id. Default: every tissue.
    dataset_id : str, optional
        GTEx release. Default ``"gtex_v8"``.

    Returns
    -------
    dict
        ``data.transcripts`` -- one entry per transcript and tissue.

    Examples
    --------
    >>> print(gtex_get_median_transcript_expression("TP53", "Liver"))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    gencode_id, matched, err = _gtex_resolve_gene(gene, dataset)
    if err:
        return err
    params = {"gencodeId": gencode_id, "datasetId": dataset, "page": 0, "itemsPerPage": _MAX_PAGE_SIZE}
    tissue = _clean(tissue_site_detail_id)
    if tissue:
        params["tissueSiteDetailId"] = tissue
    payload, err = _gtex_get("/expression/medianTranscriptExpression", params)
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    return _ok(
        {
            "gene": _clean((matched or {}).get("geneSymbol")) or _clean(gene),
            "gencode_id": gencode_id,
            "dataset_id": dataset,
            "tissue_site_detail_id": tissue or "all",
            "transcripts": rows,
            "paging": paging,
        }
    )


def gtex_get_single_nucleus_expression(gene, tissue_site_detail_ids=None, dataset_id=_GTEX_SNRNASEQ_DATASET):
    """Single-nucleus RNA expression summarised per cell type, from the GTEx snRNA-seq pilot.

    The closest GTEx gets to cell-type resolution, and the natural cross-check for a spatial
    deconvolution result. Note the dataset default: ``gtex_snrnaseq_pilot`` is the only one that
    answers this route with data, and it is not listed by :func:`gtex_list_datasets`. Asking for a
    numbered release instead returns HTTP 200 with zero rows (F22).

    Parameters
    ----------
    gene : str
        Gene symbol or Ensembl/GENCODE id.
    tissue_site_detail_ids : list of str, optional
        Tissue ids. Default: every tissue in the pilot.
    dataset_id : str, optional
        Default ``"gtex_snrnaseq_pilot"``. Changing it is almost always a mistake.

    Returns
    -------
    dict
        ``data.cell_types`` with per-cell-type summary statistics.

    Examples
    --------
    >>> print(gtex_get_single_nucleus_expression("PTPRC"))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    gencode_id, matched, err = _gtex_resolve_gene(gene, dataset)
    if err:
        return err
    params = {"gencodeId": gencode_id, "datasetId": dataset, "page": 0, "itemsPerPage": _MAX_PAGE_SIZE}
    tissues = [_clean(t) for t in (tissue_site_detail_ids or []) if _clean(t)]
    if tissues:
        params["tissueSiteDetailId"] = tissues
    payload, err = _gtex_get("/expression/singleNucleusGeneExpressionSummary", params)
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    note = "Cell-type summaries as GTEx reports them."
    if not rows and dataset != _GTEX_SNRNASEQ_DATASET:
        note = (
            f"Zero rows and dataset_id='{dataset}'. This route only carries data for "
            f"'{_GTEX_SNRNASEQ_DATASET}'; other datasets answer 200 with nothing rather than erroring."
        )
    return _ok(
        {
            "gene": _clean((matched or {}).get("geneSymbol")) or _clean(gene),
            "gencode_id": gencode_id,
            "dataset_id": dataset,
            "tissues_requested": tissues or "all",
            "cell_types": rows,
            "paging": paging,
        },
        note=note,
    )


def gtex_get_sample_info(dataset_id=_GTEX_DEFAULT_DATASET, tissue_site_detail_ids=None, page_size=_DEFAULT_PAGE_SIZE):
    """Per-sample metadata behind GTEx's numbers: donor sex and age bracket, ischemic time, RIN.

    The context that decides whether a cross-tissue difference is biology or collection. GTEx
    tissues differ systematically in post-mortem interval and RNA quality.

    Parameters
    ----------
    dataset_id : str, optional
        GTEx release. Default ``"gtex_v8"``.
    tissue_site_detail_ids : list of str, optional
        Tissue ids to restrict to. Default: every tissue.
    page_size : int, optional
        Rows per page, 1-250. Default 100.

    Returns
    -------
    dict
        ``data.samples`` plus ``data.paging`` -- this route has tens of thousands of rows.

    Examples
    --------
    >>> print(gtex_get_sample_info(tissue_site_detail_ids=["Liver"], page_size=5))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    params = {"datasetId": dataset, "page": 0, "itemsPerPage": _page_size(page_size)}
    tissues = [_clean(t) for t in (tissue_site_detail_ids or []) if _clean(t)]
    if tissues:
        params["tissueSiteDetailId"] = tissues
    payload, err = _gtex_get("/dataset/sample", params)
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    return _ok(
        {
            "dataset_id": dataset,
            "tissues_requested": tissues or "all",
            "samples": rows,
            "paging": paging,
        },
        note="Paged: 'paging' reports the total, and this is page 0 only.",
    )


def gtex_get_eqtl_genes(tissue_site_detail_id, dataset_id=_GTEX_DEFAULT_DATASET, page_size=_DEFAULT_PAGE_SIZE):
    """Genes with at least one significant eQTL in a tissue (GTEx "eGenes").

    Tells you which genes in a tissue have expression that genotype demonstrably moves -- the
    population-genetics complement to an expression difference seen in one sample.

    Parameters
    ----------
    tissue_site_detail_id : str
        Exact tissue id, e.g. ``"Liver"``.
    dataset_id : str, optional
        GTEx release. Default ``"gtex_v8"``.
    page_size : int, optional
        Rows per page, 1-250. Default 100.

    Returns
    -------
    dict
        ``data.egenes`` with q-values and effect sizes, plus paging (a tissue has ~15,000).

    Examples
    --------
    >>> print(gtex_get_eqtl_genes("Liver", page_size=5))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    tissue = _clean(tissue_site_detail_id)
    if not tissue:
        return _missing("tissue_site_detail_id", "Pass an exact tissue id; gtex_list_tissue_sites() lists all 54.")
    payload, err = _gtex_get(
        "/association/egene",
        {
            "tissueSiteDetailId": tissue,
            "datasetId": dataset,
            "page": 0,
            "itemsPerPage": _page_size(page_size),
        },
    )
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    return _ok({"tissue_site_detail_id": tissue, "dataset_id": dataset, "egenes": rows, "paging": paging})


def gtex_get_single_tissue_eqtls(
    gene=None, variant_id=None, tissue_site_detail_ids=None, dataset_id=_GTEX_DEFAULT_DATASET
):
    """Significant single-tissue eQTLs, by gene, by variant, or both.

    At least one of ``gene`` or ``variant_id`` is required -- GTEx answers a bare query with
    HTTP 400, since the unfiltered result set is the whole catalogue.

    Parameters
    ----------
    gene : str, optional
        Gene symbol or Ensembl/GENCODE id.
    variant_id : str, optional
        GTEx variant id, e.g. ``"chr1_1000000_A_G_b38"``.
    tissue_site_detail_ids : list of str, optional
        Restrict to these tissues. Default: every tissue.
    dataset_id : str, optional
        GTEx release. Default ``"gtex_v8"``.

    Returns
    -------
    dict
        ``data.eqtls`` with p-values, NES and the tissue each was called in.

    Examples
    --------
    >>> print(gtex_get_single_tissue_eqtls(gene="TP53", tissue_site_detail_ids=["Liver"]))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    if not _clean(gene) and not _clean(variant_id):
        return _missing(
            "gene or variant_id",
            "GTEx rejects an unfiltered eQTL query with HTTP 400; name at least one of them.",
        )
    params = {"datasetId": dataset, "page": 0, "itemsPerPage": _MAX_PAGE_SIZE}
    gencode_id = None
    if _clean(gene):
        gencode_id, _matched, err = _gtex_resolve_gene(gene, dataset)
        if err:
            return err
        params["gencodeId"] = gencode_id
    if _clean(variant_id):
        params["variantId"] = _clean(variant_id)
    tissues = [_clean(t) for t in (tissue_site_detail_ids or []) if _clean(t)]
    if tissues:
        params["tissueSiteDetailId"] = tissues
    payload, err = _gtex_get("/association/singleTissueEqtl", params)
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    return _ok(
        {
            "gene": _clean(gene) or None,
            "gencode_id": gencode_id,
            "variant_id": _clean(variant_id) or None,
            "dataset_id": dataset,
            "tissues_requested": tissues or "all",
            "eqtls": rows,
            "paging": paging,
        }
    )


def gtex_calculate_eqtl(gene, variant_id, tissue_site_detail_id, dataset_id=_GTEX_DEFAULT_DATASET):
    """Compute the eQTL association for one gene-variant-tissue triple, on demand.

    Unlike :func:`gtex_get_single_tissue_eqtls`, which returns pre-computed significant hits, this
    calculates the association for any triple -- so a non-significant pairing still gets an answer
    instead of an empty result.

    Parameters
    ----------
    gene : str
        Gene symbol or Ensembl/GENCODE id.
    variant_id : str
        GTEx variant id, e.g. ``"chr1_1000000_A_G_b38"``.
    tissue_site_detail_id : str
        Exact tissue id.
    dataset_id : str, optional
        GTEx release. Default ``"gtex_v8"``.

    Returns
    -------
    dict
        ``data.association`` with the p-value, effect size and genotype-level expression.

    Examples
    --------
    >>> print(gtex_calculate_eqtl("TP53", "chr17_7676154_G_A_b38", "Liver"))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    missing = [
        name
        for name, value in (("variant_id", variant_id), ("tissue_site_detail_id", tissue_site_detail_id))
        if not _clean(value)
    ]
    if missing:
        return _missing(
            " and ".join(missing),
            "This route computes one association and needs all three of gene, variant and tissue.",
            plural=len(missing) > 1,
        )
    gencode_id, matched, err = _gtex_resolve_gene(gene, dataset)
    if err:
        return err
    payload, err = _gtex_get(
        "/association/dyneqtl",
        {
            "gencodeId": gencode_id,
            "variantId": _clean(variant_id),
            "tissueSiteDetailId": _clean(tissue_site_detail_id),
            "datasetId": dataset,
        },
    )
    if err:
        return err
    return _ok(
        {
            "gene": _clean((matched or {}).get("geneSymbol")) or _clean(gene),
            "gencode_id": gencode_id,
            "variant_id": _clean(variant_id),
            "tissue_site_detail_id": _clean(tissue_site_detail_id),
            "dataset_id": dataset,
            "association": payload,
        },
        note="Computed on request, so a null result means no association, not a missing record.",
    )


def gtex_get_multi_tissue_eqtls(gene, variant_id=None, dataset_id=_GTEX_DEFAULT_DATASET):
    """Multi-tissue eQTL meta-analysis (Metasoft) for a gene across all GTEx tissues at once.

    Answers whether a regulatory effect is tissue-specific or shared, which a per-tissue eQTL list
    cannot show without reading 54 results side by side.

    Parameters
    ----------
    gene : str
        Gene symbol or Ensembl/GENCODE id. Required -- this route rejects a query without one.
    variant_id : str, optional
        Restrict to one variant.
    dataset_id : str, optional
        GTEx release. Default ``"gtex_v8"``.

    Returns
    -------
    dict
        ``data.metasoft`` with per-tissue m-values and the meta-analysis p-value.

    Examples
    --------
    >>> print(gtex_get_multi_tissue_eqtls("TP53"))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    gencode_id, matched, err = _gtex_resolve_gene(gene, dataset)
    if err:
        return err
    params = {"gencodeId": gencode_id, "datasetId": dataset, "page": 0, "itemsPerPage": _MAX_PAGE_SIZE}
    if _clean(variant_id):
        params["variantId"] = _clean(variant_id)
    payload, err = _gtex_get("/association/metasoft", params)
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    return _ok(
        {
            "gene": _clean((matched or {}).get("geneSymbol")) or _clean(gene),
            "gencode_id": gencode_id,
            "variant_id": _clean(variant_id) or None,
            "dataset_id": dataset,
            "metasoft": rows,
            "paging": paging,
        }
    )


def gtex_get_single_tissue_sqtls(
    gene=None, variant_id=None, tissue_site_detail_ids=None, dataset_id=_GTEX_DEFAULT_DATASET
):
    """Splicing QTLs: variants associated with isoform usage rather than total expression.

    A gene whose total expression is flat can still have genotype-driven splicing. sQTLs are where
    that shows up, and they are a common explanation for a marker behaving inconsistently.

    Parameters
    ----------
    gene : str, optional
        Gene symbol or Ensembl/GENCODE id.
    variant_id : str, optional
        GTEx variant id.
    tissue_site_detail_ids : list of str, optional
        Restrict to these tissues. Default: every tissue.
    dataset_id : str, optional
        GTEx release. Default ``"gtex_v8"``.

    Returns
    -------
    dict
        ``data.sqtls`` with the intron phenotype id, p-value and effect size.

    Examples
    --------
    >>> print(gtex_get_single_tissue_sqtls(gene="TP53", tissue_site_detail_ids=["Liver"]))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    if not _clean(gene) and not _clean(variant_id):
        return _missing(
            "gene or variant_id",
            "GTEx rejects an unfiltered sQTL query; name at least one of them.",
        )
    params = {"datasetId": dataset, "page": 0, "itemsPerPage": _MAX_PAGE_SIZE}
    gencode_id = None
    if _clean(gene):
        gencode_id, _matched, err = _gtex_resolve_gene(gene, dataset)
        if err:
            return err
        params["gencodeId"] = gencode_id
    if _clean(variant_id):
        params["variantId"] = _clean(variant_id)
    tissues = [_clean(t) for t in (tissue_site_detail_ids or []) if _clean(t)]
    if tissues:
        params["tissueSiteDetailId"] = tissues
    payload, err = _gtex_get("/association/singleTissueSqtl", params)
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    return _ok(
        {
            "gene": _clean(gene) or None,
            "gencode_id": gencode_id,
            "variant_id": _clean(variant_id) or None,
            "dataset_id": dataset,
            "tissues_requested": tissues or "all",
            "sqtls": rows,
            "paging": paging,
        }
    )


def gtex_get_finemapping(gene, tissue_site_detail_ids=None, dataset_id=_GTEX_DEFAULT_DATASET):
    """Fine-mapped causal variants for a gene's eQTLs, with posterior inclusion probabilities.

    An eQTL list names variants in linkage with the causal one; fine-mapping is the statistical
    attempt to say which variant it actually is. Use before claiming a specific SNP is responsible.

    Parameters
    ----------
    gene : str
        Gene symbol or Ensembl/GENCODE id.
    tissue_site_detail_ids : list of str, optional
        Restrict to these tissues. Default: every tissue.
    dataset_id : str, optional
        GTEx release. Default ``"gtex_v8"``.

    Returns
    -------
    dict
        ``data.finemapping`` with per-variant posterior inclusion probabilities and credible sets.

    Examples
    --------
    >>> print(gtex_get_finemapping("TP53", ["Liver"]))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    gencode_id, matched, err = _gtex_resolve_gene(gene, dataset)
    if err:
        return err
    params = {"gencodeId": gencode_id, "datasetId": dataset, "page": 0, "itemsPerPage": _MAX_PAGE_SIZE}
    tissues = [_clean(t) for t in (tissue_site_detail_ids or []) if _clean(t)]
    if tissues:
        params["tissueSiteDetailId"] = tissues
    payload, err = _gtex_get("/association/fineMapping", params)
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    return _ok(
        {
            "gene": _clean((matched or {}).get("geneSymbol")) or _clean(gene),
            "gencode_id": gencode_id,
            "dataset_id": dataset,
            "tissues_requested": tissues or "all",
            "finemapping": rows,
            "paging": paging,
        }
    )


def gtex_get_single_nucleus_cell_counts(tissue_site_detail_ids=None, dataset_id=_GTEX_SNRNASEQ_DATASET):
    """How many nuclei GTEx has per tissue and cell type, for sizing a snRNA-seq comparison.

    This is a census of the dataset, not a measurement of any gene: it answers "does GTEx have
    enough nuclei of this cell type in this tissue to compare against?" before a per-gene call is
    worth making. Upstream reached this endpoint through a ``result_type="summary"`` switch that
    still required a gene argument; the endpoint ignores it (measured: a real gene, a nonsense gene
    and no gene at all each return the same 114 rows), so there is no gene parameter here.

    Parameters
    ----------
    tissue_site_detail_ids : list of str, optional
        Restrict to these tissues, which the endpoint does honour. Default: every tissue.
    dataset_id : str, optional
        Default ``"gtex_snrnaseq_pilot"``, the only dataset with single-nucleus data.

    Returns
    -------
    dict
        ``data.cell_counts`` with one row per tissue x cell type, carrying ``numCells``.

    Examples
    --------
    >>> print(gtex_get_single_nucleus_cell_counts(["Lung"]))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    params = {"datasetId": dataset, "page": 0, "itemsPerPage": _MAX_PAGE_SIZE}
    tissues = [_clean(t) for t in (tissue_site_detail_ids or []) if _clean(t)]
    if tissues:
        params["tissueSiteDetailId"] = tissues
    payload, err = _gtex_get("/expression/singleNucleusGeneExpressionSummary", params)
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    return _ok(
        {
            "dataset_id": dataset,
            "tissues_requested": tissues or "all",
            "cell_counts": rows,
            "paging": paging,
        },
        note="Cell counts for the dataset as a whole. No gene is involved -- for per-gene, "
        "per-cell-type values use gtex_get_single_nucleus_expression().",
    )


def gtex_get_sqtl_genes(tissue_site_detail_id=None, dataset_id=_GTEX_DEFAULT_DATASET, page_size=_DEFAULT_PAGE_SIZE):
    """Genes with at least one significant splicing QTL (GTEx "sGenes").

    The splicing counterpart of :func:`gtex_get_eqtl_genes`: genes whose isoform usage, rather than
    total expression, is under genetic control in a tissue.

    Parameters
    ----------
    tissue_site_detail_id : str, optional
        Exact tissue id. Optional here -- unlike the eQTL route, this one answers an unfiltered
        query -- but a tissue is what makes the answer interpretable.
    dataset_id : str, optional
        GTEx release. Default ``"gtex_v8"``.
    page_size : int, optional
        Rows per page, 1-250. Default 100.

    Returns
    -------
    dict
        ``data.sgenes`` with the intron cluster phenotype id, empirical p-value and the per-gene
        significance threshold, plus paging.

    Examples
    --------
    >>> print(gtex_get_sqtl_genes("Liver", page_size=5))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    params = {"datasetId": dataset, "page": 0, "itemsPerPage": _page_size(page_size)}
    if _clean(tissue_site_detail_id):
        params["tissueSiteDetailId"] = _clean(tissue_site_detail_id)
    payload, err = _gtex_get("/association/sgene", params)
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    return _ok(
        {
            "tissue_site_detail_id": _clean(tissue_site_detail_id) or "all",
            "dataset_id": dataset,
            "sgenes": rows,
            "paging": paging,
        }
    )


def gtex_get_independent_eqtls(gene, tissue_site_detail_ids=None, dataset_id=_GTEX_DEFAULT_DATASET):
    """Conditionally-independent eQTL signals for a gene, ranked primary, secondary, tertiary.

    A gene's eQTL list is mostly one signal seen through many linked variants. This route reports
    the signals that survive conditioning on each other, so ``rank`` 2 and above are genuinely
    separate regulatory effects rather than more views of the first one.

    Parameters
    ----------
    gene : str
        Gene symbol or Ensembl/GENCODE id. Required -- the route answers HTTP 422 without one.
    tissue_site_detail_ids : list of str, optional
        Restrict to these tissues. Default: every tissue.
    dataset_id : str, optional
        GTEx release. Default ``"gtex_v8"``.

    Returns
    -------
    dict
        ``data.independent_eqtls`` with ``rank``, variant, tissue, effect size and p-value.

    Examples
    --------
    >>> print(gtex_get_independent_eqtls("TP53"))  # doctest: +SKIP
    """
    dataset, err = _gtex_dataset(dataset_id)
    if err:
        return err
    gencode_id, matched, err = _gtex_resolve_gene(gene, dataset)
    if err:
        return err
    params = {"gencodeId": gencode_id, "datasetId": dataset, "page": 0, "itemsPerPage": _MAX_PAGE_SIZE}
    tissues = [_clean(t) for t in (tissue_site_detail_ids or []) if _clean(t)]
    if tissues:
        params["tissueSiteDetailId"] = tissues
    payload, err = _gtex_get("/association/independentEqtl", params)
    if err:
        return err
    rows, paging = _gtex_rows(payload)
    return _ok(
        {
            "gene": _clean((matched or {}).get("geneSymbol")) or _clean(gene),
            "gencode_id": gencode_id,
            "dataset_id": dataset,
            "tissues_requested": tissues or "all",
            "independent_eqtls": rows,
            "paging": paging,
        },
        note="rank 1 is the primary signal; rank 2 and above are independent of it, not duplicates.",
    )


# ------------------------------------------------------------------------ EBI Expression Atlas


def _gxa_catalog():
    """The whole experiment catalogue, or an error. ``(records, None)`` / ``(None, error)``.

    There is no server-side filter: ``?species=...``, ``?experimentType=...`` and ``?geneQuery=...``
    all return a byte-identical 2.6 MB body (measured 2026-09-17, 4,562 experiments). Every filter
    in this module is therefore applied here, after the download, and each call pays for the whole
    catalogue -- which is why the search functions below cap and page rather than inviting a loop.
    """
    payload, err = _fetch_json(_GXA_BASE + "/json/experiments", timeout=_GXA_CATALOG_TIMEOUT)
    if err:
        return None, err
    records = payload.get("experiments") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return None, _error(
            "Expression Atlas returned a catalogue this tool does not recognise "
            f"(expected a list of experiments under 'experiments', got {type(payload).__name__})."
        )
    return records, None


def _gxa_kind(record):
    """``"baseline"``, ``"differential"`` or ``None``, from the code rather than a display label.

    Every one of the eight ``rawExperimentType`` values the catalogue uses ends in ``_BASELINE``,
    ``_BASELINE_DIA`` or ``_DIFFERENTIAL`` (measured over all 4,562 records), so the classification
    is read off the code and needs no table of display names that HPA-style renaming could break.
    """
    raw = (record.get("rawExperimentType") or "").upper()
    if "DIFFERENTIAL" in raw:
        return "differential"
    if "BASELINE" in raw:
        return "baseline"
    return None


def _gxa_slim(record):
    """One catalogue record, reduced to the fields a research log needs."""
    return {
        "accession": record.get("experimentAccession"),
        "kind": _gxa_kind(record),
        "experiment_type": record.get("rawExperimentType"),
        "species": record.get("species"),
        "description": record.get("experimentDescription"),
        "factors": record.get("experimentalFactors") or [],
        "technology": record.get("technologyType") or [],
        "assays": _as_int(record.get("numberOfAssays")),
        "last_update": record.get("lastUpdate"),
    }


def _gxa_condition_match(record, needle):
    """Substring test over the two fields that actually carry condition text."""
    haystack = [record.get("experimentDescription") or ""]
    haystack.extend(record.get("experimentalFactors") or [])
    needle = _norm(needle)
    return any(needle in _norm(field) for field in haystack)


def _gxa_select(records, kind=None, species=None, condition=None):
    """Filter, then rank by assay count. ``(rows, note, error)``.

    Ranking is assay count descending, largest study first, and that is the only ranking -- upstream
    ranked by whether the experiment's *description text* happened to mention the gene, a signal its
    own comments call a coincidence rather than a filter.
    """
    rows = [r for r in records if not kind or _gxa_kind(r) == kind]
    note = None
    if _clean(species):
        wanted = _norm(species)
        matched = [r for r in rows if _norm(r.get("species")) == wanted]
        if not matched:
            names = sorted({r.get("species") for r in rows if r.get("species")})
            close = _suggest(_clean(species), names)
            hint = f" Closest names in the catalogue: {', '.join(close)}." if close else ""
            return (
                None,
                None,
                _error(
                    f"No Expression Atlas experiment is annotated with species '{_clean(species)}'. "
                    "The catalogue spells species as a binomial, e.g. 'Homo sapiens'." + hint
                ),
            )
        rows = matched
    if _clean(condition):
        before = len(rows)
        rows = [r for r in rows if _gxa_condition_match(r, condition)]
        note = (
            f"'{_clean(condition)}' is matched as text against the experiment description and its "
            f"experimental factors: {len(rows)} of {before} experiments matched. It is not an "
            "ontology lookup, so a synonym the curators did not use will not match."
        )
    rows.sort(key=lambda r: (-_as_int(r.get("numberOfAssays")), r.get("experimentAccession") or ""))
    return rows, note, None


def _gxa_page(rows, limit, offset):
    """``(slice, paging)`` -- explicit because the service pages nothing for us."""
    size = _page_size(limit, default=50, maximum=_MAX_ROWS_CEILING)
    start = max(0, _as_int(offset))
    window = rows[start : start + size]
    return window, {"total_matched": len(rows), "offset": start, "returned": len(window), "limit": size}


def expression_atlas_list_baseline_experiments(species="Homo sapiens", limit=50, offset=0):
    """Baseline expression experiments in EBI Expression Atlas for one species.

    Baseline experiments measure expression across conditions with no comparison built in -- tissue
    panels, developmental stages, cell lines. This is how you find the reference dataset behind a
    "which tissues express X" claim, and how you find a study whose design matches yours.

    There is no ``gene`` argument. The GXA catalogue cannot be filtered by gene: the query parameter
    is accepted and ignored, and so is ``geneQuery`` on the per-experiment route (both measured).
    To ask whether a gene is expressed somewhere, use the HPA or GTEx functions in this module; to
    read one experiment's gene table, pass its accession to
    :func:`expression_atlas_get_experiment`.

    Parameters
    ----------
    species : str, optional
        Binomial name as the catalogue spells it, e.g. ``"Homo sapiens"`` (the default),
        ``"Mus musculus"``. Matched case- and punctuation-insensitively; an unknown name is an
        error with suggestions, never a silent empty list.
    limit : int, optional
        Experiments to return, 1-200. Default 50.
    offset : int, optional
        Rows to skip, for paging through a large match. Default 0.

    Returns
    -------
    dict
        ``data.experiments`` ranked by assay count (largest study first), with ``data.paging``.

    Examples
    --------
    >>> print(expression_atlas_list_baseline_experiments("Homo sapiens", limit=5))  # doctest: +SKIP
    """
    records, err = _gxa_catalog()
    if err:
        return err
    rows, _note, err = _gxa_select(records, kind="baseline", species=species)
    if err:
        return err
    window, paging = _gxa_page(rows, limit, offset)
    return _ok(
        {
            "species": _clean(species) or "any",
            "kind": "baseline",
            "experiments": [_gxa_slim(r) for r in window],
            "paging": paging,
        },
        note="Ranked by assay count, descending. Pass an accession to "
        "expression_atlas_get_experiment() for that experiment's design and gene table.",
    )


def expression_atlas_search_differential(condition=None, species="Homo sapiens", limit=50, offset=0):
    """Differential expression experiments, optionally filtered by condition text.

    Differential experiments carry an explicit comparison -- disease versus healthy, treated versus
    control, mutant versus wild type -- so this is where to look for published evidence that a
    condition changes expression, rather than for a baseline level.

    Parameters
    ----------
    condition : str, optional
        Free text matched against the experiment description and its experimental factors, e.g.
        ``"breast cancer"``, ``"hypoxia"``. Omit to list every differential experiment.
    species : str, optional
        Binomial name. Default ``"Homo sapiens"``. Pass ``None`` for every species.
    limit : int, optional
        Experiments to return, 1-200. Default 50.
    offset : int, optional
        Rows to skip. Default 0.

    Returns
    -------
    dict
        ``data.experiments`` ranked by assay count, ``data.paging``, and a ``note`` reporting how
        many of the candidates the condition text matched.

    Examples
    --------
    >>> print(expression_atlas_search_differential("breast cancer", limit=5))  # doctest: +SKIP
    """
    records, err = _gxa_catalog()
    if err:
        return err
    rows, note, err = _gxa_select(records, kind="differential", species=species, condition=condition)
    if err:
        return err
    window, paging = _gxa_page(rows, limit, offset)
    return _ok(
        {
            "condition": _clean(condition) or None,
            "species": _clean(species) or "any",
            "kind": "differential",
            "experiments": [_gxa_slim(r) for r in window],
            "paging": paging,
        },
        note=note
        or "No condition filter applied; every differential experiment for this species "
        "is in scope, ranked by assay count.",
    )


def expression_atlas_search_experiments(condition=None, species=None, limit=50, offset=0):
    """Search the whole Expression Atlas catalogue, baseline and differential together.

    Use when you do not yet know which kind of experiment answers the question -- the result marks
    each hit ``baseline`` or ``differential`` so the next call can be the specific one.

    Parameters
    ----------
    condition : str, optional
        Free text matched against the experiment description and its experimental factors.
    species : str, optional
        Binomial name. Default ``None``, meaning every species -- the catalogue spans 4,562
        experiments across plants, animals and fungi, so a species filter is usually worth giving.
    limit : int, optional
        Experiments to return, 1-200. Default 50.
    offset : int, optional
        Rows to skip. Default 0.

    Returns
    -------
    dict
        ``data.experiments`` with a ``kind`` on each row, ``data.counts`` by kind over everything
        that matched, and ``data.paging``.

    Examples
    --------
    >>> print(expression_atlas_search_experiments("spatial transcriptomics", limit=5))  # doctest: +SKIP
    """
    records, err = _gxa_catalog()
    if err:
        return err
    rows, note, err = _gxa_select(records, species=species, condition=condition)
    if err:
        return err
    counts = {"baseline": 0, "differential": 0, "unclassified": 0}
    for row in rows:
        counts[_gxa_kind(row) or "unclassified"] += 1
    window, paging = _gxa_page(rows, limit, offset)
    return _ok(
        {
            "condition": _clean(condition) or None,
            "species": _clean(species) or "any",
            "counts": counts,
            "experiments": [_gxa_slim(r) for r in window],
            "paging": paging,
        },
        note=note or "No condition filter applied; the whole catalogue is in scope, ranked by assay count.",
    )


def _gxa_design(payload):
    """The experimental design, read from whichever shape ``columnHeaders`` came back in.

    Baseline and differential experiments answer with different column shapes -- baseline columns
    are assay *groups* (a tissue, a stage) carrying ``factorValue`` and an ontology term id;
    differential columns are *contrasts* carrying ``displayName`` and both sides of the comparison.
    Upstream read neither: it asked the per-experiment ``experiment`` object for
    ``experimentalFactors``/``contrasts``/``numberOfAssays``, which only the *catalogue* record
    carries, so those fields came back empty every time and the endpoint was written off as
    metadata-free.
    """
    groups, contrasts = [], []
    for column in payload.get("columnHeaders") or []:
        if "contrastSummary" in column or "testAssayGroup" in column:
            test = column.get("testAssayGroup") or {}
            reference = column.get("referenceAssayGroup") or {}
            contrasts.append(
                {
                    "id": column.get("id"),
                    "comparison": column.get("displayName"),
                    "test_replicates": _as_int(test.get("replicates")),
                    "reference_replicates": _as_int(reference.get("replicates")),
                }
            )
            continue
        summary = column.get("assayGroupSummary") or {}
        groups.append(
            {
                "id": column.get("assayGroupId"),
                "condition": column.get("factorValue"),
                "ontology_term": column.get("factorValueOntologyTermId"),
                "replicates": _as_int(summary.get("replicates")),
            }
        )
    return groups, contrasts


def expression_atlas_get_experiment(accession, max_gene_rows=10):
    """One Expression Atlas experiment: its design, and the first page of its gene table.

    This returns more than experiment metadata. The per-experiment route also carries a slice of
    the expression matrix -- per-gene values across the assay groups for a baseline experiment, or
    log2 fold change and p-value per contrast for a differential one -- along with the total number
    of genes behind that slice. The slice is the service's own first page, not a gene the caller
    chose: ``geneQuery`` is ignored on this route (measured in four syntaxes), so there is no way to
    ask this endpoint about a specific gene.

    Parameters
    ----------
    accession : str
        ArrayExpress-style accession, e.g. ``"E-MTAB-2836"``. An accession the service does not
        know is rejected rather than answered with an empty record -- measured as 404 for a real
        accession the atlas has retired and 400 for a malformed one -- and both are reported here
        as an unknown accession rather than as a transport failure.
    max_gene_rows : int, optional
        How many gene rows to keep from the returned page, 1-200. Default 10. The page itself is
        sized by the service: a differential experiment honours a size request, a baseline one
        returned the same 29 rows of 9,570 at every size tried.

    Returns
    -------
    dict
        ``data.experiment`` (accession, type, species, description), ``data.assay_groups`` or
        ``data.contrasts`` depending on the design, and ``data.gene_rows`` with the unit and the
        total gene count.

    Examples
    --------
    >>> print(expression_atlas_get_experiment("E-MTAB-2836"))  # doctest: +SKIP
    """
    accession = _clean(accession)
    if not accession:
        return _missing(
            "accession",
            "Find one with expression_atlas_search_experiments(); they look like 'E-MTAB-2836'.",
        )
    try:
        payload = request_json(f"{_GXA_BASE}/json/experiments/{_seg(accession)}", allowed_hosts=_ALLOWED_HOSTS)
    except HttpError as exc:
        if exc.status in (400, 404):
            return _error(
                f"Expression Atlas does not recognise the accession '{accession}'. It rejects an "
                "accession it does not hold rather than returning an empty record, so this is not "
                "an experiment with no data -- it is either a typo or an accession the atlas has "
                "retired. Find a current one with expression_atlas_search_experiments()."
            )
        return _error(exc.detail, retryable=bool(exc.status) and exc.status >= 500)
    experiment = payload.get("experiment") or {}
    groups, contrasts = _gxa_design(payload)
    profiles = payload.get("profiles") or {}
    rows = profiles.get("rows") or []
    keep = _page_size(max_gene_rows, default=10, maximum=_MAX_ROWS_CEILING)
    unit = next((row.get("expressionUnit") for row in rows if row.get("expressionUnit")), None)
    data = {
        "experiment": {
            "accession": experiment.get("accession") or accession,
            "experiment_type": experiment.get("type"),
            "species": experiment.get("species"),
            "description": experiment.get("description"),
        },
        "assay_groups": groups,
        "contrasts": contrasts,
        "gene_rows": {
            "unit": unit,
            "genes_in_experiment": _as_int(profiles.get("searchResultTotal")),
            "returned": min(len(rows), keep),
            "rows": rows[:keep],
        },
    }
    return _ok(
        data,
        note="The gene rows are the service's own first page, not a selection -- this endpoint has "
        "no gene filter. For a named gene, use the HPA or GTEx functions in this module.",
    )
