# Ontology Term Resolution

## Metadata

**Short Description**: Resolve free-text scientific labels to ontology term IDs and validate existing CURIEs against the EBI Ontology Lookup Service (OLS4). Also look up prefixes in Bioregistry, resolve compact identifiers via Identifiers.org, map lab shorthand with ZOOMA, and build Ontobee term pages. Use whenever an ontology identifier must be produced or checked - annotating tissue, cell type, disease, phenotype, assay, chemical, organism, sex, or developmental stage fields; preparing metadata for GEO, ENA, BioSamples, CELLxGENE, HCA, or ISA-Tab submission; auditing a metadata table of term IDs; checking whether a term is obsolete and what replaced it; or deciding HPO vs HP. Triggers include "ontology term", "ontology ID", "CURIE", "controlled vocabulary", "UBERON", "CL:", "MONDO", "HPO", "EFO", "ChEBI", "NCBITaxon", "GO term", "PATO", "Zooma", "Bioregistry", "Identifiers.org", "Ontobee", "annotate this tissue/cell type/disease", and any request to emit or verify an identifier shaped like PREFIX:0001234.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/ontology-term-resolution/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: not stated upstream for the tool; the upstream frontmatter `license` field reads "MIT", and upstream uses that field for the skill text in some files and for the tool in others, so it is not taken as the tool's licence
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 22 upstream script/reference path(s) marked as not vendored; 4 fence(s) left with nothing runnable replaced by a pointer; 1 manifest replacement(s) and 0 excision(s) applied.

---

## When to use

Any time an ontology identifier is about to be written down or trusted: annotating a metadata
column, filling a submission template, auditing a table someone else produced, or checking whether
an ID in an old file is still current.

## The rule

**Never write an ontology ID from memory, and never accept one without checking it.**

Ontology IDs are memorable in form and arbitrary in detail. A plausible-looking `UBERON:0002108`
is a real term (small intestine) that is not the liver, and nothing downstream will catch the
substitution — the ID is well-formed, the ontology is right, and the metadata is silently wrong.
Reviewers cannot spot it either, which is why these errors persist into published datasets.

Every ID this skill emits comes from a live OLS lookup. Every ID it is handed gets verified.
Bioregistry, Identifiers.org, ZOOMA, and Ontobee answer prefix, landing-page, and shorthand
questions — they do not replace that OLS check.

## Which service

| Question | Script | Authority |
| --- | --- | --- |
| What is the term for "left ventricle"? | (upstream helper script, not vendored) | OLS |
| OLS missed lab shorthand (`PBMC`, `WT`) | (upstream helper script, not vendored), then `validate_terms.py` | ZOOMA proposes; OLS decides |
| Is `EFO:0001067` real, current, correctly labelled? | (upstream helper script, not vendored) | OLS |
| Is `HPO` a real prefix? Does `HP:notanid` match the pattern? | (upstream helper script, not vendored) | Bioregistry |
| Which landing page should this CURIE open? | (upstream helper script, not vendored) | Identifiers.org + Ontobee URLs |

All four scripts take single values or files, emit TSV or JSON, and need no packages beyond the
standard library. Full traps for the non-OLS services are in (upstream reference file, not included).

## Resolve text to terms

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

```
query   rank  curie           label  ontology  match_type   strategy  defining_ontology
liver   1     UBERON:0002107  liver  uberon    exact_label  exact     true
```

```bash
# a column of tissue names; anything not an exact hit is reported, not guessed
python (upstream helper script, not vendored: resolve_terms.py) --input tissues.txt --ontology uberon \
    --exact-only --format tsv -o resolved.tsv

# accept fuzzy fallbacks, then review the partial hits by hand
python (upstream helper script, not vendored: resolve_terms.py) "left ventrical of heart" --ontology uberon --top 3
```

The search escalates `exact` (label and synonym) → `token` → `fulltext` and stops at the first
strategy that returns anything, reporting which one fired. `--exact-only` disables the ladder.
`--branch UBERON:0000465` restricts candidates to descendants of a term.

**Read `match_type` before using a result.** `exact_label` and `exact_synonym` are safe;
`partial` means OLS returned its best guess for a string that does not exist as written, and
needs a human decision. `unresolved` is a legitimate output — see (upstream reference file, not included)
for the normalisations worth retrying first.

## Validate existing IDs

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

```
id              status     actual_label                  ontology  replacement     detail
UBERON:0002107  ok         liver                         uberon
EFO:0001067     obsolete   obsolete_parasitic infection  efo       MONDO:0005135   obsolete; replaced by MONDO:0005135
UBERON:9999999  not_found                                                          no such term in the ontology this prefix names
```

Exit code is 1 if anything failed, 0 otherwise, 2 on usage or network trouble — so it works as a
CI gate on a metadata file:

```bash
# id + label columns; catches IDs that exist but are labelled as something else
python (upstream helper script, not vendored: validate_terms.py) --input metadata.tsv --strict

# a tissue column must hold UBERON anatomical entities and nothing else
python (upstream helper script, not vendored: validate_terms.py) --input tissue_ids.tsv \
    --branch UBERON:0000465 --expect-ontology uberon
```

| Status | Meaning | Verdict |
| --- | --- | --- |
| `ok` | Exists, current, consistent with everything asserted | pass |
| `matched_synonym` | Claimed label is a synonym; primary label differs | warn |
| `imported_only` | Home ontology no longer asserts this ID | warn |
| `not_a_class` | Term is a property or individual | warn |
| `not_found` | No such term | fail |
| `obsolete` | Obsoleted; `replacement` gives the successor when one exists | fail |
| `label_mismatch` | ID and claimed label describe different things | fail |
| `wrong_ontology` | Right kind of ID, wrong ontology for this column | fail |
| `wrong_branch` | Not a descendant of the required root | fail |
| `malformed_curie` | Not of the form `PREFIX:local` | fail |

`--strict` promotes warnings to failures.

## Check a prefix or compact identifier

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

```
query        status          preferred_prefix  canonical_curie  pattern    detail
HP           ok              HP                                 ^\d{7}$
HPO          synonym_prefix  HP                                 ^\d{7}$    'HPO' is a synonym of preferred prefix HP
HP:0001250   ok              HP                HP:0001250       ^\d{7}$
HPO:0001250  synonym_prefix  HP                HP:0001250       ^\d{7}$    'HPO' is a synonym of preferred prefix HP
```

Bioregistry accepts synonym prefixes. Identifiers.org does not — `HPO:0001250` is HTTP 400.
Rewrite to the preferred prefix before handing a CURIE to OLS. Landing-page columns come from
Bioregistry mappings (`providers.miriam`, `mappings.ontobee`), not from templating that
preferred prefix: `ORPHA:558` is a 400, `orphanet:558` is a 200, and OBA has no Identifiers.org
namespace at all. Empty cells mean the service does not host the prefix. This script does
**not** say the term exists; that is still `validate_terms.py`.

## Map lab shorthand (ZOOMA)

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

`--ontology` is required. Unfiltered ZOOMA annotate returns FOODON, XAO, and BTO alongside UBERON
for `liver`, all at HIGH confidence. HIGH/GOOD hits are candidates only — run `validate_terms.py`
on every CURIE before writing it down.

## API behaviour that will mislead you

These are verified against the live service. Upstream ships scripts that handle them; none is
vendored here, so a direct OLS or ZOOMA call must handle each one itself. Full detail in (upstream reference file, not included).

| Trap | Consequence |
| --- | --- |
| `exact=true` is exact **token** matching | `liver` returns 161 hits in UBERON; adding `queryFields=label` returns 1 |
| `/search` never returns `is_obsolete` or `term_replaced_by` | Named in `fieldList` they are dropped silently; only term detail can answer "is this ID still current" |
| `ontology=efo` returns MONDO and CL hits | Ontologies import each other; filter on the CURIE prefix yourself |
| The same term appears once per importing ontology | Deduplicate on `obo_id`, keep `is_defining_ontology: true` |
| The `obo_id` index has holes | `MONDO:0000001` is live but unindexed by `obo_id`; an IRI fallback is required to avoid a false `not_found` |
| IRIs are not all OBO PURLs | EFO and Orphanet use their own namespaces — resolve IRIs, do not template them |
| OxO is retired | Returns HTML with HTTP 200; use term cross-references or SSSOM instead |
| A branch check does not exclude cell types from anatomy | CARO puts `cell` under `anatomical structure`; constrain the prefix too |
| ZOOMA without an ontology filter | `liver` returns 100+ HIGH hits across FOODON, XAO, BTO, UBERON |
| Identifiers.org synonym prefixes | `HPO:0001250` is HTTP 400; Bioregistry accepted the same CURIE |
| Identifiers.org encoded colon | `HP%3A0001250` is HTTP 400; the path must keep `:` |
| Bioregistry `preferred_prefix` is not the Identifiers.org namespace | `ORPHA:558` is 400; `orphanet:558` is 200. `hp:0001250` and `chebi:15377` are 400 because those namespaces embed the prefix in the LUI. Use `providers.miriam` from `/api/reference/{CURIE}`; omit the URL when that mapping is missing (OBA, XAO, ECTO) |
| Ontobee search | HTML page only — no JSON API; do not scrape it |

## Choosing the ontology

MONDO for disease, HP for phenotype, UBERON for tissue, CL for cell type, EFO for assay, ChEBI for
compounds, NCBITaxon for organism, PATO for sex and for `normal`. Prefix-to-OLS-id mappings (`HP`
is served as `hp`, `Orphanet` as `ordo`), branch roots for `--branch`, and the overlapping-ontology
judgement calls are in (upstream reference file, not included).

## Reporting results

Give the ID **and** the label, and say how each was matched. A table of bare IDs cannot be
reviewed. State unresolved terms explicitly rather than filling them with the nearest hit.

## References

- (upstream reference file, not included) — endpoints, parameters, response fields, and every verified OLS trap.
- (upstream reference file, not included) — Bioregistry, Identifiers.org, ZOOMA, and Ontobee: when to use
  each, and the traps that make an unfiltered or synonym-prefix call look successful.
- (upstream reference file, not included) — prefix/ontology-id table, branch roots, which ontology owns
  which concept.
- (upstream reference file, not included) — candidate-selection procedure, normalisations to retry,
  auditing an existing table, obsolete terms, cross-ontology mapping.
