# PrimeKG Knowledge Graph Skill

## Metadata

**Short Description**: Query the Precision Medicine Knowledge Graph (PrimeKG) for multiscale biological data including genes, drugs, diseases, phenotypes, and more.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/primekg/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: not stated upstream for the tool; the upstream frontmatter `license` field reads "Unknown", and upstream uses that field for the skill text in some files and for the tool in others, so it is not taken as the tool's licence
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 4 upstream script/reference path(s) marked as not vendored; 5 manifest replacement(s) and 0 excision(s) applied.

---

## Overview

PrimeKG is a precision medicine knowledge graph that integrates over 20 primary databases and high-quality scientific literature into a single resource. It contains over 100,000 nodes and 4 million edges across 29 relationship types, including drug-target, disease-gene, and phenotype-disease associations.

**Key capabilities:**
- Search for nodes (genes, proteins, drugs, diseases, phenotypes)
- Retrieve direct neighbors (associated entities and clinical evidence)
- Analyze local disease context (related genes, drugs, phenotypes)
- Identify drug-disease paths (potential repurposing opportunities)

**Data access:** Programmatic access via `query_primekg.py`. Data is stored wherever the PrimeKG download (`kg.csv`) was placed on this machine; pass that path explicitly.

## When to Use This Skill

This skill should be used when:

- **Knowledge-based drug discovery:** Identifying targets and mechanisms for diseases.
- **Drug repurposing:** Finding existing drugs that might have evidence for new indications.
- **Phenotype analysis:** Understanding how symptoms/phenotypes relate to diseases and genes.
- **Multiscale biology:** Bridging the gap between molecular targets (genes) and clinical outcomes (diseases).
- **Network pharmacology:** Investigating the broader network effects of drug-target interactions.

## Core Workflow

### 1. Search for Entities

Find identifiers for genes, drugs, or diseases.

```python
import pandas as pd

# No query helper is vendored here: load the PrimeKG edge list yourself. kg.csv is ~1 GB and
# ~8 million rows (each edge appears in both directions); load it once, in full.
KG_PATH = "/path/to/kg.csv"  # wherever the user's PrimeKG download is; a data lake may list it
kg = pd.read_csv(KG_PATH, low_memory=False)
# columns: relation, display_relation, x_index, x_id, x_type, x_name, x_source,
#          y_index, y_id, y_type, y_name, y_source

def search_nodes(text, node_type=None):
    nodes = kg[["x_index", "x_id", "x_type", "x_name", "x_source"]].drop_duplicates()
    hit = nodes["x_name"].str.contains(text, case=False, na=False, regex=False)
    if node_type:
        hit &= nodes["x_type"] == node_type
    return nodes[hit]

results = search_nodes("Alzheimer", node_type="disease")
```

### 2. Get Neighbors (Direct Associations)

Retrieve all connected nodes and relationship types.

```python
def get_neighbors(x_index, relation_type=None):
    """Every edge leaving one node (PrimeKG's ``x_index`` is unique per node; ``x_id`` is not)."""
    edges = kg[kg["x_index"] == x_index]
    if relation_type:
        edges = edges[edges["relation"] == relation_type]
    return edges[["relation", "display_relation", "y_type", "y_name", "y_id", "y_source"]]

neighbors = get_neighbors(results["x_index"].iloc[0])
```

### 3. Analyze Disease Context

A high-level function to summarize associations for a disease.

```python
def get_disease_context(name):
    """Genes, drugs and phenotypes linked to a disease, by exact node name."""
    edges = kg[(kg["x_type"] == "disease") & (kg["x_name"] == name)]
    return {
        "associated_genes": sorted(edges.loc[edges["relation"] == "disease_protein", "y_name"].unique()),
        "associated_drugs": sorted(edges.loc[edges["y_type"] == "drug", "y_name"].unique()),
        "phenotypes": sorted(edges.loc[edges["relation"] == "disease_phenotype_positive", "y_name"].unique()),
    }

context = get_disease_context("Alzheimer disease")
```

## Relationship Types in PrimeKG

The graph contains several key relationship types including:
- `protein_protein`: Physical PPIs
- `drug_protein`: Drug target/mechanism associations
- `disease_gene`: Genetic associations
- `drug_disease`: Indications and contraindications
- `disease_phenotype`: Clinical signs and symptoms
- `gwas`: Genome-wide association studies evidence

## Best Practices

1. **Use specific IDs:** When using `get_neighbors`, ensure you have the correct ID from `search_nodes`.
2. **Context first:** Use `get_disease_context` for a broad overview before diving into specific genes or drugs.
3. **Filter relationships:** Use the `relation_type` filter in `get_neighbors` to focus on specific evidence (e.g., only `drug_protein`).
4. **Multiscale integration:** Combine with `OpenTargets` for deeper genetic evidence or `Semantic Scholar` for the latest literature context.

## Resources

### Scripts
- (upstream helper script, not vendored): Core functions for searching and querying the knowledge graph.

### Data Path
- Data: `kg.csv`, downloaded from the [PrimeKG Harvard Dataverse](https://dataverse.harvard.edu/dataverse/primekg).
- No loader or default location is vendored: pass the path of the user's `kg.csv` explicitly (as `KG_PATH` above).
- Total nodes: ~129,000
- Total edges: ~4,000,000
- Database: CSV-based, optimized for pandas querying.
