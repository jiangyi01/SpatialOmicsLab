# gget

## Metadata

**Short Description**: Fast CLI/Python queries to 20+ bioinformatics databases. Use for quick lookups: gene info, BLAST/BLAT, viral sequence downloads, AlphaFold structures, enrichment analysis, OpenTargets, COSMIC, CELLxGENE, and 8cube mouse specificity/expression data. Best for interactive exploration and simple queries. For batch processing or advanced BLAST use biopython; for multi-database Python workflows use bioservices.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/gget/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: not stated upstream for the tool; the upstream frontmatter `license` field reads "BSD-2-Clause license", and upstream uses that field for the skill text in some files and for the tool in others, so it is not taken as the tool's licence
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 1 install fence(s) replaced by a provisioning pointer; 1 inline install span(s) rewritten; 9 upstream script/reference path(s) marked as not vendored; 0 manifest replacement(s) and 1 excision(s) applied; 3 source line(s) replaced or removed by the manifest.

---

## Overview

gget is a command-line bioinformatics tool and Python package providing unified access to 20+ genomic databases and analysis methods. Query gene information, sequence analysis, protein structures, viral sequences, expression data, disease associations, and mouse tissue/cell specificity metrics through a consistent interface. Most gget modules work both as command-line tools and as Python functions.

**Important**: The databases queried by gget are continuously updated, which sometimes changes their structure. Guidance here targets gget 0.30.5 (PyPI current as of 2026-06-07). For reproducible work, pin `gget==0.30.5`; for broken upstream database adapters, update gget after checking release notes.

## Installation

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

## Quick Start

Basic usage pattern for all modules:

```bash
# Command-line
gget <module> [arguments] [options]

# Python
gget.module(arguments, options)
```

Most modules return:
- **Command-line**: JSON (default) or CSV with `-csv` flag
- **Python**: DataFrame or dictionary

Common flags across modules:
- `-o/--out`: Save results to file
- `-q/--quiet`: Suppress progress information
- `-csv`: Return CSV format (command-line only)

Python argument names generally match long CLI options without leading dashes. For example, `--census_version` becomes `census_version=...`. Use `gget <module> --help` for the exact current signature.

## Module Categories

gget exposes 23 modules in six categories. Parameters, CLI and Python examples, and
return shapes for every one are in
(upstream reference file, not included) (upstream reference file, not included); fuller per-parameter
documentation is in (upstream reference file, not included) (upstream reference file, not included).

| Category | Modules |
| --- | --- |
| 1. Reference & gene information | `ref` (Ensembl reference downloads), `search` (gene search), `info` (gene/transcript detail), `seq` (nucleotide and protein sequences) |
| 2. Sequence analysis & alignment | `blast`, `blat`, `muscle` (multiple alignment), `diamond` (local alignment) |
| 3. Structural & protein analysis | `pdb` (structures and metadata), `alphafold` (structure prediction), `elm` (linear motifs) |
| 4. Expression & disease data | `archs4` (correlation, tissue expression), `cellxgene` (single-cell), `enrichr` (enrichment), `bgee` (orthology and expression), `opentargets` (disease and drug), `cbio` (cancer genomics), `cosmic` (mutations) |
| 5. Viral & mouse specificity | `virus` (viral sequences), `8cube` (mouse specificity and expression) |
| 6. Additional tools | `mutate` (mutated sequences), `gpt` (text generation), `setup` (installs module dependencies into the running interpreter; not run from this document) |

Several modules depend on packages that upstream's one-time setup step installs, which is not run from this document (`alphafold`, `elm`,
`cellxgene`), and `cosmic` prompts for COSMIC credentials to download its database.

## Common Workflows

Worked multi-module pipelines — gene characterization, structural comparison, expression
and enrichment analysis, disease and drug association, orthology comparison, and
reference-file preparation for kallisto or alignment — are in
(upstream reference file, not included) (upstream reference file, not included), with longer versions in
(upstream reference file, not included) (upstream reference file, not included).

## Best Practices

### Data Retrieval
- Use `--limit` to control result sizes for large queries
- Save results with `-o/--out` for reproducibility
- Check database versions/releases for consistency across analyses
- Use `--quiet` in production scripts to reduce output

### Sequence Analysis
- For BLAST/BLAT, start with default parameters, then adjust sensitivity
- Use `gget diamond` with `--threads` for faster local alignment
- Save DIAMOND databases with `--diamond_db` for repeated queries
- For multiple sequence alignment, use `-s5/--super5` for large datasets

### Expression and Disease Data
- Gene symbols are case-sensitive in cellxgene (e.g., 'PAX7' vs 'Pax7')
- alphafold, cellxgene, elm and gpt need upstream's setup step first; if one fails for a missing dependency, say it is not available here
- For enrichment analysis, use database shortcuts for convenience
- Cache cBioPortal data with `-dd` to avoid repeated downloads
- For OpenTargets, inspect returned column names before writing filters; gget 0.30.5 follows the newer OpenTargets API schema

### Structure Prediction
- AlphaFold multimer predictions: use `-mr 20` for higher accuracy
- Use `-r` flag for AMBER relaxation of final structures
- Visualize results in Python with `plot=True`
- Check PDB database first before running AlphaFold predictions

### Viral Data
- Use restrictive filters with `gget virus` before requesting broad viral datasets
- Keep `command_summary.txt` with downstream results for reproducibility and recovery after partial downloads
- Use `--baseline` and `--merge-results` to resume interrupted viral metadata/sequence downloads

### Error Handling
- Database structures change; when an adapter breaks, check upstream release notes and pin the newer fixed version explicitly
- Pin the known-good version for reproducible environments: (not installed from this document: if the import fails, the library is not available here -- say so, do not install it)
- Process max ~1000 Ensembl IDs at once with gget info
- For large-scale analyses, implement rate limiting for API queries
- Keep COSMIC and OpenAI credentials in named environment variables or interactive prompts; do not write real credentials into examples, notebooks, or logs

## Output Formats

### Command-line
- Default: JSON
- CSV: Add `-csv` flag
- FASTA: gget seq, gget mutate
- PDB: gget pdb, gget alphafold
- PNG: gget cbio plot
- FASTA/CSV/JSONL folder: gget virus

### Python
- Default: DataFrame or dictionary
- JSON: Add `json=True` parameter
- Save to file: Add `save=True` or specify `out="filename"`
- AnnData: gget cellxgene
- DataFrame/JSON: gget 8cube specificity, psi_block, expression

## Resources

This skill includes reference documentation for detailed module information:

### (upstream reference file, not included)
- `module_reference.md` - Comprehensive parameter reference for all modules
- `database_info.md` - Information about queried databases and their update frequencies
- `workflows.md` - Extended workflow examples and use cases

For additional help:
- Official documentation: https://pachterlab.github.io/gget/
- GitHub issues: https://github.com/pachterlab/gget/issues
- Citation: Luebbert, L. & Pachter, L. (2023). Efficient querying of genomic reference databases with gget. Bioinformatics. https://doi.org/10.1093/bioinformatics/btac836
