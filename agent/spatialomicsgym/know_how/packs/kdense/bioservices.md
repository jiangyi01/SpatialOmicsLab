# BioServices

## Metadata

**Short Description**: Unified Python interface to 40+ bioinformatics services. Use when querying multiple databases (UniProt, KEGG, ChEMBL, Reactome) in a single workflow with consistent API. Best for cross-database analysis, ID mapping across services. For quick single-database lookups use gget; for sequence/file manipulation use biopython.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/bioservices/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: GPL-3.0 (bioservices) (recorded in packs/MANIFEST.yaml)
**Commercial Use**: This text is MIT and may be used commercially. The wrapped library, bioservices, is GPL-3.0: that licence governs redistribution of the library itself, not of this description, and nothing from the library is vendored here.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 1 install fence(s) replaced by a provisioning pointer; 12 upstream script/reference path(s) marked as not vendored; 3 fence(s) left with nothing runnable replaced by a pointer.

---

## Overview

BioServices is a Python package providing programmatic access to approximately 40 bioinformatics web services and databases. Retrieve biological data, perform cross-database queries, map identifiers, analyze sequences, and integrate multiple biological resources in Python workflows. The package handles both REST and SOAP/WSDL protocols transparently.

**Version note:** Examples target **bioservices 1.16.0** (PyPI, Mar 2026). Requires **Python 3.9–3.12**. UniProt REST changes in mid-2022 (bioservices ≥1.10) mainly affect tabular `columns` names — see upstream `_legacy_names` if parsing breaks. ChEMBL wrappers changed at 1.6.0 (2018 API); use `get_similarity`, `get_substructure`, `get_molecule` instead of pre-1.6 method names.

## When to Use This Skill

This skill should be used when:
- Retrieving protein sequences, annotations, or structures from UniProt, PDB, Pfam
- Analyzing metabolic pathways and gene functions via KEGG or Reactome
- Searching compound databases (ChEBI, ChEMBL, PubChem) for chemical information
- Converting identifiers between different biological databases (KEGG↔UniProt, compound IDs)
- Running sequence similarity searches (BLAST, MUSCLE alignment)
- Querying gene ontology terms (QuickGO, GO annotations)
- Accessing protein-protein interaction data (PSICQUIC, IntactComplex)
- Mining genomic data (BioMart, ArrayExpress, ENA)
- Integrating data from multiple bioinformatics resources in a single workflow

## Core Capabilities

### 1. Protein Analysis

Retrieve protein information, sequences, and functional annotations:

```python
from bioservices import UniProt

u = UniProt(verbose=False)

# Search for protein by name
results = u.search("ZAP70_HUMAN", frmt="tab", columns="id,genes,organism")

# Retrieve FASTA sequence
sequence = u.retrieve("P43403", "fasta")

# Map identifiers between databases
kegg_ids = u.mapping(fr="UniProtKB_AC-ID", to="KEGG", query="P43403")
```

**Key methods:**
- `search()`: Query UniProt with flexible search terms
- `retrieve()`: Get protein entries in various formats (FASTA, XML, tab)
- `mapping()`: Convert identifiers between databases

Reference: (upstream reference file, not included) for complete UniProt API details.

### 2. Pathway Discovery and Analysis

Access KEGG pathway information for genes and organisms:

```python
from bioservices import KEGG

k = KEGG()
k.organism = "hsa"  # Set to human

# Search for organisms
k.lookfor_organism("droso")  # Find Drosophila species

# Find pathways by name
k.lookfor_pathway("B cell")  # Returns matching pathway IDs

# Get pathways containing specific genes
pathways = k.get_pathway_by_gene("7535", "hsa")  # ZAP70 gene

# Retrieve and parse pathway data
data = k.get("hsa04660")
parsed = k.parse(data)

# Extract pathway interactions
interactions = k.parse_kgml_pathway("hsa04660")
relations = interactions['relations']  # Protein-protein interactions

# Convert to Simple Interaction Format
sif_data = k.pathway2sif("hsa04660")
```

**Key methods:**
- `lookfor_organism()`, `lookfor_pathway()`: Search by name
- `get_pathway_by_gene()`: Find pathways containing genes
- `parse_kgml_pathway()`: Extract structured pathway data
- `pathway2sif()`: Get protein interaction networks

Reference: (upstream reference file, not included) for complete pathway analysis workflows.

### 3. Compound Database Searches

Search and cross-reference compounds across multiple databases:

```python
from bioservices import KEGG, UniChem

k = KEGG()

# Search compounds by name
results = k.find("compound", "Geldanamycin")  # Returns cpd:C11222

# Get compound information with database links
compound_info = k.get("cpd:C11222")  # Includes ChEBI links

# Cross-reference KEGG → ChEMBL using UniChem
u = UniChem()
chembl_id = u.get_compound_id_from_kegg("C11222")  # Returns CHEMBL278315
```

**Version caveat:** the per-source `get_compound_id_from_*` helpers are gone from
bioservices 1.16.0 — check `hasattr(u, "get_compound_id_from_kegg")` first, and
otherwise use the current UniChem API (`u.get_compounds(compound, source_type)`
and read `res["compounds"][0]["sources"]`). ChEMBL lookups follow the same rule:
`get_molecule`, not the pre-1.6 `get_compound_by_chemblId`.

**Common workflow:**
1. Search compound by name in KEGG
2. Extract KEGG compound ID
3. Use UniChem for KEGG → ChEMBL mapping
4. ChEBI IDs are often provided in KEGG entries

Reference: (upstream reference file, not included) for complete cross-database mapping guide.

### 4. Sequence Analysis

Run BLAST searches and sequence alignments. NCBI requires a contact email — prefer the `NCBI_EMAIL` environment variable (same convention as BioPython Entrez and other repo skills):

```python
import os
from bioservices import NCBIblast

s = NCBIblast(verbose=False)
email = os.environ["NCBI_EMAIL"]  # set before running: export NCBI_EMAIL=you@lab.org

# Run BLASTP against UniProtKB
jobid = s.run(
    program="blastp",
    sequence=protein_sequence,
    stype="protein",
    database="uniprotkb",
    email=email,
)

# Check job status and retrieve results
s.getStatus(jobid)
results = s.getResult(jobid, "out")
```

**Note:** BLAST jobs are asynchronous. Check status before retrieving results.

### 5. Identifier Mapping

Convert identifiers between different biological databases:

```python
from bioservices import UniProt, KEGG

# UniProt mapping (many database pairs supported)
u = UniProt()
results = u.mapping(
    fr="UniProtKB_AC-ID",  # Source database
    to="KEGG",              # Target database
    query="P43403"          # Identifier(s) to convert
)

# KEGG gene ID → UniProt
kegg_to_uniprot = u.mapping(fr="KEGG", to="UniProtKB_AC-ID", query="hsa:7535")

# For compounds, use UniChem
from bioservices import UniChem
u = UniChem()
chembl_from_kegg = u.get_compound_id_from_kegg("C11222")
```

**Supported mappings (UniProt):**
- UniProtKB ↔ KEGG
- UniProtKB ↔ Ensembl
- UniProtKB ↔ PDB
- UniProtKB ↔ RefSeq
- And many more (see (upstream reference file, not included))

### 6. Gene Ontology Queries

Access GO terms and annotations:

```python
from bioservices import QuickGO

g = QuickGO(verbose=False)

# Retrieve GO term information
term_info = g.Term("GO:0003824", frmt="obo")

# Search annotations
annotations = g.Annotation(protein="P43403", format="tsv")
```

### 7. Protein-Protein Interactions

Query interaction databases via PSICQUIC. **PSICQUIC is not shipped by every
release — it is absent from 1.16.0** — so import it defensively and fall back to
`IntactComplex`, `OmniPath`, or `STRING` when it is missing:

```python
from bioservices import PSICQUIC

s = PSICQUIC(verbose=False)

# Query specific database (e.g., MINT)
interactions = s.query("mint", "ZAP70 AND species:9606")

# List available interaction databases
databases = s.activeDBs
```

**Available databases:** MINT, IntAct, BioGRID, DIP, and 30+ others.

## Multi-Service Integration Workflows

BioServices excels at combining multiple services for comprehensive analysis. Common integration patterns:

### Complete Protein Analysis Pipeline

Execute a full protein characterization workflow:

```bash
export NCBI_EMAIL=your.email@example.com
# (upstream helper script, not vendored)
# Or pass email as optional second argument if NCBI_EMAIL is unset
# (upstream helper script, not vendored)
```

This script demonstrates:
1. UniProt search for protein entry
2. FASTA sequence retrieval
3. BLAST similarity search
4. KEGG pathway discovery
5. PSICQUIC interaction mapping

### Pathway Network Analysis

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

Extracts and analyzes:
- All pathway IDs for organism
- Protein-protein interactions per pathway
- Interaction type distributions
- Exports to CSV/SIF formats

### Cross-Database Compound Search

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

Retrieves:
- KEGG compound ID
- ChEBI identifier
- ChEMBL identifier
- Basic compound properties

### Batch Identifier Conversion

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

## Best Practices

### Output Format Handling

Different services return data in various formats:
- **XML**: Parse using BeautifulSoup (most SOAP services)
- **Tab-separated (TSV)**: Pandas DataFrames for tabular data
- **Dictionary/JSON**: Direct Python manipulation
- **FASTA**: BioPython integration for sequence analysis

### Rate Limiting and Verbosity

Control API request behavior:

```python
from bioservices import KEGG

k = KEGG(verbose=False)  # Suppress HTTP request details
k.TIMEOUT = 30  # Adjust timeout for slow connections
```

### Error Handling

Wrap service calls in try-except blocks:

```python
try:
    results = u.search("ambiguous_query")
    if results:
        # Process results
        pass
except Exception as e:
    print(f"Search failed: {e}")
```

### Organism Codes

Use standard organism abbreviations:
- `hsa`: Homo sapiens (human)
- `mmu`: Mus musculus (mouse)
- `dme`: Drosophila melanogaster
- `sce`: Saccharomyces cerevisiae (yeast)

List all organisms: `k.list("organism")` or `k.organismIds`

### Integration with Other Tools

BioServices works well with:
- **BioPython**: Sequence analysis on retrieved FASTA data
- **Pandas**: Tabular data manipulation
- **PyMOL**: 3D structure visualization (retrieve PDB IDs)
- **NetworkX**: Network analysis of pathway interactions
- **Galaxy**: Custom tool wrappers for workflow platforms

## Resources

### (upstream helper script, not vendored)

Executable Python scripts demonstrating complete workflows:

- `protein_analysis_workflow.py`: End-to-end protein characterization
- `pathway_analysis.py`: KEGG pathway discovery and network extraction
- `compound_cross_reference.py`: Multi-database compound searching
- `batch_id_converter.py`: Bulk identifier mapping utility

Scripts can be executed directly or adapted for specific use cases.

### (upstream reference file, not included)

Detailed documentation loaded as needed:

- `services_reference.md`: Comprehensive list of all 40+ services with methods
- `workflow_patterns.md`: Detailed multi-step analysis workflows
- `identifier_mapping.md`: Complete guide to cross-database ID conversion

Load references when working with specific services or complex integration tasks.

## Installation

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

Dependencies are installed automatically. Upstream CI tests Python 3.9–3.12 ([PyPI](https://pypi.org/project/bioservices/), [docs](https://bioservices.readthedocs.io/)).

## Credentials

Most services need no API key. Exceptions:

| Service | Requirement |
|---------|-------------|
| NCBI BLAST | Contact email via `NCBI_EMAIL` or `email=` in `NCBIblast.run()` |
| Some EBI services | Optional; check service docs if rate-limited |

Set once per shell session:

```bash
export NCBI_EMAIL=your.email@example.com
```

Use a real institutional or lab address — NCBI may contact you about heavy BLAST usage.

## Additional Information

For detailed API documentation and advanced features, refer to:
- Official documentation: https://bioservices.readthedocs.io/
- Source code: https://github.com/cokelaer/bioservices
- Service-specific references in (upstream reference file, not included)
