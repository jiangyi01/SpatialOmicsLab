# Waypoint: Outpost Bio's Open Microbiome Foundation Models

## Metadata

**Short Description**: Use when working with Outpost Bio's open microbiome foundation models - the Waypoint checkpoints (Waypoint-6m, Waypoint-45m, Waypoint-170m), the Atlas pretraining corpus, the Compass eight-task benchmark, or the `waypoint` CLI from the `waypoint-bio` package. Covers embedding microbiome samples, fine-tuning on taxonomic abundance data, benchmarking a checkpoint on Compass, pretraining a GPT-2 model on taxonomic abundance profiles, and converting MetaPhlAn, Kraken2, QIIME 2, or MGnify abundance tables into waypoint format.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/waypoint-bio/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: Apache-2.0 (Outpost Bio's three released artefacts, per this document's body) (recorded in packs/MANIFEST.yaml)
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 1 install fence(s) replaced by a provisioning pointer; 14 upstream script/reference path(s) marked as not vendored; 2 fence(s) left with nothing runnable replaced by a pointer; 5 source line(s) replaced or removed by the manifest.

---

## Overview

Outpost Bio open-sourced three artefacts under Apache 2.0, described in
[Treloar et al., bioRxiv 2026.05.02.722381](https://www.biorxiv.org/content/10.64898/2026.05.02.722381v2):

| Artefact | What it is | Hugging Face |
| --- | --- | --- |
| **Waypoint** | GPT-2-style causal LMs over taxonomic tokens, 6M–170M params | `outpost-bio/Waypoint-6m`, `-45m`, `-170m` |
| **Atlas** | 539,308 microbiome samples scraped from MGnify (485,377 pretrain / 53,931 benchmark) | `outpost-bio/Atlas` |
| **Compass** | Eight downstream tasks over four studies | `outpost-bio/Compass` |

The unifying idea: a microbiome sample is a *sentence*. Each taxon is one token, tokens are ordered
by descending abundance z-score, and the model is trained with next-token prediction. A pretrained
checkpoint then supplies sample-level embeddings or a fine-tuning backbone for prediction tasks.

All of it is driven by one CLI, `waypoint`, with five subcommands: `prepare-dataset`, `embed`,
`finetune`, `benchmark`, `pretrain`.

## When to use

- Embedding 16S/shotgun taxonomic profiles into fixed-size vectors for clustering, visualisation, or
  a downstream classifier.
- Fine-tuning a Waypoint checkpoint to predict a phenotype, treatment, or continuous readout from
  community composition.
- Scoring your own microbiome model against Compass so the number is comparable to the paper.
- Pretraining a taxonomic language model on Atlas or on your own corpus.
- Converting profiler output (MetaPhlAn, Kraken2/Bracken, QIIME 2, MGnify TSVs) into the input format
  these tools expect.

**Do not reach for this** when you have fewer than ~1,000 labelled samples — see
[Scientific caveats](#scientific-caveats). A random forest on relative abundances is the better tool
there, and the paper says so.

## Setup

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

Atlas, Compass, and every Waypoint checkpoint are **gated**. Access is auto-approved, but you must
click through once per repo and then authenticate:

1. Request access on each repo page you need: [Waypoint-6m](https://huggingface.co/outpost-bio/Waypoint-6m),
   [Waypoint-45m](https://huggingface.co/outpost-bio/Waypoint-45m),
   [Waypoint-170m](https://huggingface.co/outpost-bio/Waypoint-170m),
   [Atlas](https://huggingface.co/datasets/outpost-bio/Atlas),
   [Compass](https://huggingface.co/datasets/outpost-bio/Compass).
2. Authenticate with a read-scoped Hugging Face token the user has already configured (the `HF_TOKEN` environment variable); this document does not log in or store a token.

A 401/403 from any subcommand almost always means access was never requested on that specific repo —
a token alone is not enough. Use a read-scoped token. The tokenizer loads via
`trust_remote_code=True`, so pin a `revision` if you need the remote code fixed across runs.

## The waypoint data format

Everything except `prepare-dataset` consumes **waypoint format**: a `.parquet` / `.csv` / `.tsv`
whose rows are samples, with two aligned list-columns plus any label columns you need.

| Column | Type | Notes |
| --- | --- | --- |
| `Taxa` | `list[str]` | Full lineage strings, `;`-separated: `k__Bacteria; p__Firmicutes; ...; g__Lactobacillus` |
| `Relative Abundances` | `list[float]` | Same length as `Taxa`, same order |
| *(any)* | scalar | Targets, covariates, or a `Split` column |

Prefer parquet. CSV/TSV stores the lists as `repr` strings and round-trips through `ast.literal_eval`.

**Give full lineages, not bare names.** The tokenizer extracts the genus segment (`g__`) from each
lineage and falls back to the most specific higher rank when genus is missing. Bare names disable
that fallback entirely.

## Workflow

### 1. Get your data into waypoint format

If you already have a sample × taxa (or taxa × sample) abundance matrix with lineage labels:

```bash
waypoint prepare-dataset \
    --input abundance_matrix.tsv \
    --metadata sample_labels.csv \
    --output dataset.parquet
```

Orientation is auto-detected from the first column header (`taxonomy`, `lineage`, `taxon`, `otu`,
`#otu id` ⇒ taxa-as-rows); override with `--orientation`. Rows are normalised to sum to 1 unless you
pass `--no_normalize`, and zeros are dropped unless you pass `--keep_zeros`.

`prepare-dataset` cannot read profiler output directly — MetaPhlAn uses `|` separators, Kraken2
reports encode the hierarchy as indentation, and QIIME 2/SILVA prefixes the domain `d__` instead of
`k__` (which the tokenizer silently ignores). Use the bundled converter for those:

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

See (upstream reference file, not included) for every input layout, rank handling, and the `d__`/`|` gotchas.

### 2. Check vocabulary coverage before anything else

Waypoint's vocabulary is fixed at pretraining time from Atlas. Taxa absent from it become `<unk>` and
are **silently dropped** by `waypoint embed`; the paper names this as the models' main limitation. A
sample whose taxa are all out-of-vocabulary yields a degenerate `[BOS][EOS]` embedding.

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

It reports per-sample and abundance-weighted coverage and flags samples below a threshold. Treat
median abundance-weighted coverage under ~0.8 as a reason to re-examine your taxonomy labels before
trusting any downstream number.

### 3. Embed samples

```bash
waypoint embed \
    --model outpost-bio/Waypoint-6m \
    --data dataset.parquet \
    --output embeddings.parquet
```

Output is indexed by sample ID with columns `dim_0 … dim_{H-1}` (`H` = 256 for 6m, 512 for 45m,
768 for 170m). Defaults: `--pooling last_token`, `--batch_size 32`, `--max_length 512`, device
auto-detected (`cuda` → `mps` → `cpu`).

Keep `--pooling last_token` unless you have a reason to change it: it matches how the checkpoints
were pretrained and how `benchmark` and `finetune` pool. `mean` is a reasonable alternative for
unsupervised use; `first_token`/`cls_token` return the BOS position and carry little signal in a
causal LM.

### 4. Fine-tune on your labels

```bash
# classification
waypoint finetune \
    --model outpost-bio/Waypoint-45m \
    --data dataset.parquet \
    --output_dir outputs/ft_disease \
    --task_type classification \
    --target "Disease Status" \
    --config configs/finetune_classification.yaml

# regression, with a categorical covariate one-hot appended to the pooled embedding
waypoint finetune \
    --model outpost-bio/Waypoint-45m \
    --data dataset.parquet \
    --output_dir outputs/ft_degradation \
    --task_type regression \
    --target "Degradation Rate" \
    --covariate_column Drug \
    --config configs/finetune_regression.yaml
```

Config paths resolve against the bundled `waypoint_bio/configs/` tree, so `configs/...` works from
any directory without cloning.

Defaults worth overriding for small datasets: `warmup_steps: 1000` (drop to ~50 so warmup finishes
before early stopping), `num_epochs: 1` in the shipped configs (raise it — early stopping on
validation loss is what actually terminates training), and `use_lora: true` when VRAM is tight
(~1% of parameters trained; adapters are merged back before saving, so the checkpoint stays a plain
`AutoModel`).

Splits default to a random 80/10/10. **Set `split_column` to a `Split` column whenever samples are
correlated** — repeated measures, one donor sampled over time, technical replicates — or a random
split leaks and the test score is meaningless.

Outputs land in `--output_dir`: `best_model/` (loadable by `embed`/`benchmark`),
`test_metrics.json`, `training_log.csv` + `.html`, and `finetune_results.json`.

### 5. Benchmark on Compass

```bash
waypoint benchmark --model outpost-bio/Waypoint-6m --output_dir outputs/benchmark
waypoint benchmark --model outputs/pretrain/best_model --tasks 1 6 --output_dir outputs/smoke
```

Fine-tunes a fresh head per task and writes `benchmark_results.json`. Classification tasks score
macro-F1; the one regression task scores R² clamped to [0, 1]; `final_score` is the unweighted mean
across tasks. Full task table, metric keys, and result-file schema: (upstream reference file, not included).

### 6. Pretrain

```bash
waypoint pretrain \
    --model_config configs/models/gpt2-45m.yaml \
    --pretrain_config configs/pretraining.yaml \
    --output_dir outputs/pretrain_45m
```

Downloads Atlas, builds a taxonomic tokenizer from the corpus, computes per-token abundance
mean/std for z-score ordering, then trains with next-token prediction and early stopping. Add
`--data my_corpus.parquet` to pretrain on your own waypoint-format corpus instead, and
`--max_samples N` for a smoke test.

Nine architectures ship, from `gpt2-6m.yaml` (8 layers, 256 hidden) to `gpt2-170m.yaml` (24 layers,
768 hidden); per-head dimension is fixed at 64 throughout. (upstream reference file, not included) has the
full table and every config key.

## Scientific caveats

These are load-bearing. Ignoring them produces numbers that look fine and mean nothing.

- **Below ~1,000 labelled examples, Waypoint underperforms a random forest on raw abundances.** The
  paper's crossover against the RF baseline sits near **10,000** training examples. Fit the baseline
  first; only adopt the transformer if it wins on your data.
- **Out-of-vocabulary taxa are dropped, not flagged.** Every Compass dataset carries some. Run
  (upstream helper script, not vendored) and report the coverage alongside your results.
- **45M, not 170M, was the best benchmark model.** Pretraining loss keeps falling with scale, but
  downstream Compass score does not — start at 6m or 45m and only scale up if it demonstrably helps.
- **Genus-level tokenisation is the default**, so species-level distinctions are collapsed. Changing
  `taxon_rank` requires re-pretraining, not just re-tokenising.
- **Compositional data.** Relative abundances are constrained to sum to 1; differences in one taxon
  induce apparent changes in others. This affects interpretation of any per-taxon attribution.
- **Batch and study effects dominate microbiome data.** Atlas spans MGnify pipelines v1.0–v5.0 and
  four sequencing modalities. Never let a study or run boundary coincide with your label boundary.
- **Not a clinical or diagnostic tool.** The model cards state this explicitly.

## References

- (upstream reference file, not included) — every subcommand flag, every config key, the model-size table.
- (upstream reference file, not included) — the eight tasks, filters, metrics, `benchmark_results.json` schema.
- (upstream reference file, not included) — waypoint format, profiler conversions, taxonomy string rules.
- (upstream reference file, not included) — using the tokenizer, datasets, heads, and checkpoints from Python.

## Scripts

- (upstream helper script, not vendored) — MetaPhlAn / Kraken2 / QIIME 2 / generic lineage tables → waypoint format.
- (upstream helper script, not vendored) — tokenizer coverage report for a waypoint-format file.

## Upstream

Code [github.com/Outpost-Bio/waypoint](https://github.com/Outpost-Bio/waypoint) ·
package `waypoint-bio` ·
paper [bioRxiv 2026.05.02.722381](https://www.biorxiv.org/content/10.64898/2026.05.02.722381v2) ·
community [Waypoint Slack](https://join.slack.com/t/outpostbio-waypoint/shared_invite/zt-3w6ivgtba-WJOCkdxiISxQpwVq9ZZxTA) ·
contact `waypoint@outpost.bio`.

Cite Treloar, N. J., Ur-Rehman, S., Yang, J., & Outpost Bio (2026). *Learning the Language of the
Microbiome with Transformers.* bioRxiv. Per-artefact DOIs are listed at
[outpost.bio/citations](https://www.outpost.bio/citations).
