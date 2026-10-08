# AlphaGenome and the AlphaGenome Atlas

## Metadata

**Short Description**: Look up precomputed AlphaGenome Atlas effects for any GRCh38 single-nucleotide variant (AVI score with Phred and 18 SHAP feature attributions, plus raw and quantile scores for RNA-seq, DNase, ATAC, ChIP-TF, ChIP-histone, CAGE, PRO-cap, splicing, polyadenylation and contact-map tracks), score variants or scan windows on demand with the AlphaGenome model for human and mouse (variant scoring, in silico mutagenesis, REF-versus-ALT track prediction), and build Atlas website deep links. Use when the user mentions AlphaGenome, AlphaGenome Atlas, AVI or AlphaGenome Variant Impact, DeepMind variant effect prediction, or wants to prioritise or mechanistically interpret non-coding, regulatory, splicing, enhancer, promoter, or chromatin-accessibility effects of SNVs from a VCF, credible set, or region. Research use only; not a clinical tool.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/alphagenome/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: Apache-2.0 (the alphagenome client, per this document's body); model weights and outputs carry their own terms (recorded in packs/MANIFEST.yaml)
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 1 install fence(s) replaced by a provisioning pointer; 1 inline install span(s) rewritten; 27 upstream script/reference path(s) marked as not vendored; 2 fence(s) left with nothing runnable replaced by a pointer; 1 manifest replacement(s) and 1 excision(s) applied; an Overview heading added above the opening prose.

---

## Overview

AlphaGenome is DeepMind's sequence-to-function model: 1 Mb of DNA in, base-pair
predictions for eleven assay types across thousands of human and mouse tracks
out. The **AlphaGenome Atlas** (released 2026-09-08) is that model run once over
every possible single-nucleotide change in GRCh38, about 9 billion variants,
stored with a single ranking number, the **AlphaGenome Variant Impact (AVI)**
score, its genome-wide percentile, and an 18-way attribution of what drives it.
Both are reached through one (not installed from this document: if the import fails, the library is not available here -- say so, do not install it) and one API key.

> Research and theoretical modelling only. Outputs must not be used to train
> other models, and are not for diagnostic procedures or medical decisions.

## When to use which

| You have | Use | Why |
| --- | --- | --- |
| hg38 SNVs (a VCF, a credible set, a region up to ~1 kb) | **Atlas** via (upstream helper script, not vendored) | precomputed, higher quota, includes AVI and attributions |
| indels, mouse variants, a non-reference background, a custom scorer or window | **model** via (upstream helper script, not vendored) or Python | the Atlas is SNV-only and hg38-only |
| a hypothesis to explain (which motif, which tissue, REF vs ALT tracks) | model `predict_variant` + plots, Atlas track scores, portal link | mechanism, not just rank |
| GRCh37 coordinates, rsIDs, unnormalised indels | `genomic-coordinates` first, then come back | wrong build or swapped REF gives a plausible wrong answer |
| ClinVar assertions, gene-disease validity, ACMG framing | `folklore-variant-evidence` | AlphaGenome is one evidence line, never the verdict |

## Setup

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

Never put the key on a command line or in a file you commit; the scripts only
read it from the environment. An invalid key surfaces as `ValueError: API key
not valid`, not as a permission error.

## The coordinate contract

- A variant is **1-based** `chr:pos:ref>alt` (`chr22:36201698:A>C`). gnomAD
  (`22-36201698-A-C`), GTEx (`chr22_36201698_A_C_b38`), and Open Targets
  spellings are accepted by the scripts and by `genome.Variant.from_str`.
- An interval on the command line is **1-based closed** `chr:start-end`; the
  SDK's `genome.Interval` is **0-based half-open**. The scripts convert.
- Human is **GRCh38 only**. The Atlas key is `chr:pos:alt`; REF is implied by
  the reference, so a variant with REF and ALT swapped, or on GRCh37, returns a
  wrong record silently. Check REF against the FASTA before trusting a lookup.
- rsIDs are not accepted by the API or the portal. Resolve them to coordinates.
- Use the `chr` prefix; `MT` becomes `chrM`.

## Atlas workflow

### 1. Rank with AVI

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

Output, one row per variant:

| Column | Meaning |
| --- | --- |
| `avi_raw` | composite model output (the 18 attributions sum to it) |
| `avi_cdf_quantile` | cumulative quantile against all genome-wide SNVs, as served |
| `avi_tail_quantile`, `avi_phred`, `avi_top_percent` | `tail = 1 - cdf`, `phred = -10 log10(tail)`; Phred 20 = top 1 %, 30 = top 0.1 % |
| `top_feature`, `top_feature_value` | largest absolute SHAP attribution and its value |
| `fi_MERGED_SPLICING` ... `fi_IS_DELETION` | all 18 attributions (keys in (upstream reference file, not included)) |
| `top_track_*` (with `--with-tracks`) | the strongest track behind the top feature: scorer, track, biosample, ontology CURIE, gene, raw score |
| `atlas_url` | deep link to the variant on the portal |
| `error` | per-variant lookup failure (indel, `N` base, wrong REF) instead of a crash |

The Atlas report's advice: **rank, do not threshold**, and pick thresholds by
region or application. Pathogenic regulatory variants sit in lower AVI bins
than protein-truncating or splice-motif variants, so a single genome-wide
cut-off under-calls exactly the variants this resource was built for.

Read the attribution before the number. `MERGED_SPLICING` or `ALPHAMISSENSE`
on top means a splice or coding mechanism; `MAX_ABS_DNASE`, `MAX_ABS_CHIP_TF`,
`MAX_ABS_RNA_SEQ` mean a regulatory mechanism you can resolve by track;
`CACTUS_241_WAY` or `PHASTCONS_470_WAY` on top means conservation is carrying
the score and the molecular mechanism is not resolved.

### 2. Resolve the mechanism by track

```bash
python (upstream helper script, not vendored: atlas_query.py) scorers                                   # what the server serves right now
python (upstream helper script, not vendored: atlas_query.py) tracks --scorer RNA_SEQ --query colon     # find ontology CURIEs
python (upstream helper script, not vendored: atlas_query.py) scores --variant chr22:36201698:A>C \
    --scorers RNA_SEQ DNASE SPLICE_SITE_USAGE --ontology UBERON:0001157 -o colon.tsv
python (upstream helper script, not vendored: atlas_query.py) scores --interval chr11:5225727-5226575 --scorers CHIP_TF --gene HBB -o hbb_tf.tsv
```

One row per variant x track (x gene for `RNA_SEQ`, `POLYADENYLATION`,
`SPLICE_*`), with `raw_score` and, where served, `quantile_score`. Track-level
scorer names: `ATAC`, `DNASE`, `CHIP_TF`, `CHIP_HISTONE`, `CAGE`, `PROCAP`,
`RNA_SEQ`, `POLYADENYLATION`, `SPLICE_SITES`, `SPLICE_SITE_USAGE`,
`SPLICE_JUNCTIONS`, `CONTACT_MAPS`, plus `*_ACTIVE` variants; `scorers` is the
authority on the live list. Filter by the tissue the question is about, not
by the genome-wide maximum: 9,440 tracks means something is always extreme
somewhere.

### 3. Send the reader to the portal

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

No key, no network. The site shows the AVI track, per-modality heatmaps over
every biosample, REF-vs-ALT prediction tracks, and motif instances. Attach a
link to every variant you report.

### In Python

```python
import os
from alphagenome.atlas import atlas
from alphagenome.data import genome

client = atlas.create(os.environ["ALPHAGENOME_API_KEY"], timeout=30)
scores = client.query_variant(
    genome.Variant.from_str("chr22:36201698:A>C"),
    requested_scorers=["AVI_SCORE", "AVI_SCORE_FEATURE_IMPORTANCE", "RNA_SEQ"],
    ontology_terms=["UBERON:0001157"],          # optional; ignored for scorers without ontology metadata
)
avi = scores["AVI_SCORE"]                        # AnnData: X (1,1) raw; layers['quantiles'] (1,1) cdf
fi = scores["AVI_SCORE_FEATURE_IMPORTANCE"]      # AnnData: X (1,18); var['name'] = feature keys
rna = scores["RNA_SEQ"]                          # AnnData: obs = variant x gene, var = tracks, X = log2 FC
client.query_interval(genome.Interval("chr11", 5225726, 5226575), requested_scorers=["AVI_SCORE"])
```

`query_interval` returns all 3 SNVs per base, in 32 bp chunks. Keep windows
to about 1 kb (3,000 variants); `atlas_query.py` refuses more unless
`--max-window` is raised. `query_variants` stops at the first failed lookup;
the script queries one variant at a time so misses become `error` cells.

## Model workflow

### Score variants the Atlas does not hold

```bash
python (upstream helper script, not vendored: score_variants.py) --variant chr22:36201698:A>C -o scores.tsv                   # 12 recommended scorers, 1 Mb
python (upstream helper script, not vendored: score_variants.py) --input indels.vcf --scorers RNA_SEQ SPLICE_SITE_USAGE \
    --ontology UBERON:0001157 --min-abs-quantile 0.99 -o colon.tsv
python (upstream helper script, not vendored: score_variants.py) --organism mouse --variant chr7:45000000:A>G --sequence-length 500KB
python (upstream helper script, not vendored: score_variants.py) --list-scorers
python (upstream helper script, not vendored: score_variants.py) --list-tracks --output-type RNA_SEQ --query liver -o tracks.tsv
```

Output is the official tidy table from `variant_scorers.tidy_scores`: one row
per variant x scorer x track (x gene) with `raw_score` and `quantile_score`,
sorted by |raw|. Default scorers are the 12 recommended difference scorers;
`--include-active` adds the seven `*_ACTIVE` activity scorers. At most 20
scorers per request.

```python
from alphagenome.models import dna_client, variant_scorers
model = dna_client.create(os.environ["ALPHAGENOME_API_KEY"])
variant = genome.Variant.from_str("chr22:36201698:A>C")
interval = variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)
adatas = model.score_variant(interval, variant, variant_scorers=[variant_scorers.RECOMMENDED_VARIANT_SCORERS["RNA_SEQ"]])
df = variant_scorers.tidy_scores(adatas)         # filter df.ontology_curie afterwards; score_variant takes no ontology_terms
```

### Predict tracks and mutagenise

```python
vo = model.predict_variant(interval, variant,
                           requested_outputs=[dna_client.OutputType.RNA_SEQ, dna_client.OutputType.DNASE],
                           ontology_terms=["UBERON:0001157"])
vo.reference.rna_seq.values, vo.alternate.rna_seq.values      # (1048576, n_tracks)

window = genome.Interval("chr20", 3_753_000, 3_753_400).resize(dna_client.SEQUENCE_LENGTH_16KB)
ism = model.score_ism_variants(interval=window, ism_interval=window.resize(256),
                               variant_scorers=[variant_scorers.CenterMaskScorer(
                                   requested_output=dna_client.OutputType.DNASE, width=501,
                                   aggregation_type=variant_scorers.AggregationType.DIFF_MEAN)])
```

Supported windows: 16 kb, 100 kb, 500 kb, 1 Mb (`2**14` to `2**20`); 1 Mb is
the default and is required for distal enhancers and contact maps. Ontology
terms are CURIEs (`UBERON:0002048` lung, `CL:0000084` T cell); discover them
with `--list-tracks` or `model.output_metadata(...).concatenate()`. Plotting,
gene annotation (GENCODE v46 Feather on GCS), splicing and haplotype recipes:
(upstream reference file, not included).

## Reading the numbers

Always report raw score **and** quantile or Phred, with the scorer, track,
biosample CURIE, and gene. `raw_score` is the effect size on the scorer's scale
(RNA_SEQ is log2 fold change: -1 is half); `quantile_score` is the rank against
common variants and saturates near 0.99999. A quantile above 0.99 with |raw| <
0.1 is the standard artefact of a quiet region and means **no effect**. Unsigned
scorers (`SPLICE_*`, `POLYADENYLATION`, `CONTACT_MAPS`, `*_ACTIVE`) have no
direction. Most variants are benign; "AlphaGenome predicts no molecular effect"
is a complete answer, and a variant inside a peak whose REF and ALT tracks are
identical is not "disrupting" anything. Full rules, tissue matching, and the
reporting checklist: (upstream reference file, not included).

What the model cannot see: trans effects, non-polyadenylated RNAs (snRNA genes
such as *RNU4-2*), cell types absent from training, protein-level consequences
(AlphaMissense is folded into AVI for that), RNA structure and miRNA biology,
diploid dosage, developmental time, species other than human and mouse.

## Limits, quota, terms

- Atlas: GRCh38 SNVs only for now; indels were scored for the paper and are
  promised later. Reference `N` bases were never scored.
- Quotas are per key and unpublished; the Atlas is documented as having a
  larger query rate than on-demand prediction. Transient `RESOURCE_EXHAUSTED`
  and `UNAVAILABLE` are retried by the client (5 attempts, back-off to 60 s).
- Access tiers (Atlas report): AVI scores are also a **permissively licensed**
  Tabix download at https://alphagenome.google/downloads; feature attributions
  and splicing scores are non-commercial downloads; all other raw track scores
  are API-only and non-commercial. Commercial API access is "coming soon" via
  Google Cloud Model Garden.
- The `alphagenome` client is Apache-2.0; model weights and outputs carry
  DeepMind's terms. Cite Avsec et al., *Nature* 649:1206 (2026) and the Atlas
  report (Cheng, Taylor, Nicolaisen, Pan, Bycroft, Perino, Ward et al., 2026).

## References

- (upstream reference file, not included) - what the Atlas contains, the 19 scorer
  configurations with track counts, AVI training and the 18 features, quantile
  to Phred, the client API and AnnData layout, error mapping, access tiers,
  portal URL grammar, GTF and download locations.
- (upstream reference file, not included) - `dna_client` cheat sheet: coordinates, sequence
  lengths, output types and track counts, ontology metadata, predict and score
  calls, recommended scorer configurations, ISM, gene annotation, plotting.
- (upstream reference file, not included) - raw versus quantile, AVI thresholds,
  tissue matching, negative results, model blind spots, coordinate hygiene,
  reporting checklist.
- Scripts: (upstream helper script, not vendored) (Atlas: `avi`, `scores`, `scorers`,
  `tracks`), (upstream helper script, not vendored) (model scoring, `--list-scorers`,
  `--list-tracks`), (upstream helper script, not vendored) (portal deep links, offline).
