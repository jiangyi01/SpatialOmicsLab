# Geniml

## Metadata

**Short Description**: Use Geniml for audited local genomic-interval workflows: validate BED and universe contracts, plan Region2Vec or scEmbed runs, inspect model/tokenizer compatibility, and assess consensus universes.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/geniml/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: BSD-2-Clause (geniml, per this document's body; this skill's own text is MIT) (recorded in packs/MANIFEST.yaml)
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 3 install fence(s) replaced by a provisioning pointer; 24 upstream script/reference path(s) marked as not vendored; 3 fence(s) left with nothing runnable replaced by a pointer; 1 manifest replacement(s) and 0 excision(s) applied; an Overview heading added above the opening prose.

---

## Overview

Use Geniml for machine learning and statistical workflows over genomic interval
sets. Treat coordinates, assemblies, token vocabularies, model artifacts, and
sample grouping as explicit contracts. The bundled scripts validate or plan;
they do not import Geniml, contact services, deserialize models, or execute
training.

`Bash` is declared only for explicit, user-approved `uv`, Python, Geniml,
Gtars, Git, and native CLI commands shown in this guide; bundled Python helpers
do not spawn subprocesses. Example paths under `data/`, `refs/`, `work/`, and
`models/` are user-provided project placeholders, not missing bundled files.

## Verified release snapshot

- Latest stable PyPI release on 2026-07-23: `geniml==0.8.4` (2026-01-14).
- PyPI does not declare `Requires-Python`; its classifiers list Python
  3.10-3.14. Prefer Python 3.11 or 3.12 where all native/ML wheels resolve.
- `geniml==0.8.4` accepts `gtars>=0.2.5`; the verified base smoke used current
  `gtars==0.9.2` (2026-06-17, Python >=3.10).
- Extras are `ml` and `test`. The base install omits Torch, Gensim, Scanpy,
  Hugging Face Hub, pyBigWig, and HMM dependencies.
- Upstream documentation contains stale examples. Release source and installed
  `--help` output take precedence where they conflict.

## Install reproducibly

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

For Region2Vec, scEmbed, evaluation, or universe methods needing ML libraries:

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

Do not install an unpinned Git branch. Record Python, OS/architecture, the
resolved lockfile, and the PyPI artifact digest. Geniml itself is BSD-2-Clause;
the `MIT` frontmatter value licenses this skill's content.

## Start with the safety gate

Before importing Geniml or running an external binary:

1. Work only with explicit local regular files. Reject URLs, FIFOs, devices,
   and symlinks unless the user deliberately changes that policy.
2. Validate BED structure and the declared assembly against a trusted local
   chromosome-sizes file.
3. Bound file count, bytes, rows, workers, epochs, and output size.
4. Separate train/validation/test by patient, donor, biological replicate, or
   other independent unit—not by BED row or cell alone.
5. Inventory and checksum the universe, tokenizer, model, config, inputs,
   metadata manifest, and native binaries.
6. Obtain explicit approval before any BEDbase or Hugging Face download. Never
   infer approval from a model ID or BEDbase identifier.
7. Keep logs aggregate and bounded. BED filenames, sample IDs, phenotypes,
   labels, barcodes, and genomic intervals may be sensitive.

## Coordinate and assembly contract

BED intervals are normally **0-based, half-open** `[start, end)`: start is
included, end is excluded, and length is `end - start`. Do not mix them with
1-based closed coordinates from VCF/GFF or user-facing genome browsers.

For every corpus and artifact, record:

- assembly and patch/accession where possible (for example GRCh38 versus
  GRCh38.p14), plus the chromosome-sizes checksum;
- contig naming convention (`chr1` versus `1`), alt/random/decoy policy, and
  mitochondrial naming;
- coordinate convention, sorting order, duplicate/overlap policy, and whether
  BED strand is meaningful;
- liftover tool, chain digest, source/target assemblies, unmapped fraction, and
  post-liftover validation.

Reject negative coordinates, `end <= start`, integer overflow, unknown
contigs, ends beyond contig length, malformed columns, mixed assemblies, and
silent contig renaming. Sorting and normalization never repair an assembly
mismatch. BED3 has no strand; when column 6 is present, preserve `+`, `-`, or
`.` unless the assay contract says otherwise.

Run a bounded validation and normalization **plan** before analysis:

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

The validator reports proposed actions but never rewrites the BED file.

## Current API map

### Region and tokenizer I/O

Prefer Gtars for new interval/tokenizer code:

```python
from gtars.models import Region, RegionSet
from gtars.tokenizers import Tokenizer

regions = RegionSet("data/peaks.bed")
tokenizer = Tokenizer.from_bed("refs/universe.bed")
encoded = tokenizer(regions)
input_ids = encoded["input_ids"]
```

`RegionSet` and `Tokenizer` also accept remote inputs in some constructors;
this skill permits local paths only unless network access is explicitly
approved. `geniml.io.RegionSet(regions, backed=False)` remains available as a
legacy Python implementation; backed sets are iterable but not indexable.
`geniml.io.Region` uses `stop`, while `gtars.models.Region` uses `end`.

With gtars 0.9.2, seven special tokens are added to a BED vocabulary. Therefore
`len(tokenizer)` is not simply the number of universe rows. Preserve universe
row order and the exact special-token map.

### Region2Vec

The modern class lives at a concrete module path:

```python
from geniml.region2vec.main import Region2VecExModel
from geniml.region2vec.utils import Region2VecDataset
from gtars.tokenizers import Tokenizer

tokenizer = Tokenizer.from_bed("refs/universe.bed")
dataset = Region2VecDataset("work/tokens.parquet", shuffle=True)
model = Region2VecExModel(tokenizer=tokenizer, embedding_dim=100)
model.train(dataset, epochs=10, window_size=5, num_cpus=4, seed=42)
```

The Parquet input must contain one list-valued `tokens` column, one document
per row. See (upstream reference file, not included) (upstream reference file, not included) for export,
encoding, legacy CLI, and evaluation details.

### scEmbed

Import `ScEmbed` from `geniml.scembed.main`. AnnData `.var` must contain
`chr`, `start`, and `end`; rows are cells and nonzero features identify
accessible regions. Pre-tokenize to a Parquet `tokens` column and use the same
Tokenizer for training and inference. See
(upstream reference file, not included) (upstream reference file, not included).

### BEDspace

BEDspace remains in 0.8.4 and invokes an external StarSpace executable.
StarSpace is archived and upstream Geniml does not pin a compatible revision.
Treat BEDspace as a legacy reproduction path, not the default for new systems.
See (upstream reference file, not included) (upstream reference file, not included) for the exact stable CLI
spelling and an immutable, explicitly unverified build baseline.

### Consensus universes and assessment

The installed 0.8.4 CLI uses:

```text
geniml build-universe {cc,ccf,ml,hmm} ...
geniml assess-universe ...
geniml eval {gdst,npt,ctt,rct,bin-gen} ...
```

CC/CCF/ML/HMM consume precomputed coverage bigWigs. Do not concatenate or
generate coverage until all BED files pass the same assembly contract.
Assessment and embedding metrics are distinct: `assess-universe` measures fit
of a universe to interval collections, while `eval` implements CTT, RCT, GDST,
and NPT for embeddings. See
(upstream reference file, not included) (upstream reference file, not included) and
(upstream reference file, not included) (upstream reference file, not included).

## Important 0.8.4 migration notes

- The 0.7.0 changelog moved new RegionSet/tokenizer work toward Gtars.
- The 0.4.0 names `TreeTokenizer` and `AnnDataTokenizer` are historical; the
  current Gtars API exposes `Tokenizer`.
- In the 0.8.4 wheel, `geniml.region2vec` and `geniml.scembed` do not re-export
  their modern classes/functions. Use the concrete module paths above.
- `geniml tokenize` and `geniml region2vec` call names no longer exported by
  their package `__init__` files; do not build new workflows around those CLI
  paths without an installed-version smoke test.
- `geniml scembed` parses legacy MatrixMarket options but its command body is a
  no-op in 0.8.4. Use `geniml.scembed.main.ScEmbed`.
- Official pages still show `geniml assess`; the release command is
  `geniml assess-universe`.
- `.gtok` remains present in legacy datasets, but upstream issue #14 proposes
  deprecating many-file `.gtok` workflows. Prefer one bounded Parquet corpus.
- Config key `embedding_size` is accepted only for backward compatibility;
  use `embedding_dim`.

## Model and universe compatibility

A Region2Vec/scEmbed inference bundle is valid only when these agree:

- model `config.yaml` `vocab_size` and `embedding_dim`;
- exact `universe.bed` bytes/order and assembly;
- tokenizer implementation/version and special-token IDs;
- checkpoint tensor shapes and pooling policy;
- Geniml/Gtars versions and any tokenization parameters.

Geniml 0.8.4 defaults to `checkpoint.pt`, `config.yaml`, and `universe.bed`.
Its loader uses `torch.load(..., weights_only=True)`, but `.pt`, Gensim
`.model`, pickle, joblib, and native binaries remain untrusted inputs. Inspect
and checksum artifacts before loading; treat them as untrusted code and never
load a checkpoint merely to discover its metadata.

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

`Region2VecExModel(model_path="org/repo")`, `ScEmbed(model_path="org/repo")`,
and Gtars `Tokenizer.from_pretrained(...)` can download from Hugging Face.
Local `from_pretrained("models/local")` loads a local bundle. Pin Hub revision
and expected hashes when a user approves download; then work offline from the
verified cache.

## BEDbase downloads and caches

`BBClient.load_bed`, `load_bedset`, and token-cache operations may contact
`https://api.bedbase.org`. The default cache is
`$BBCLIENT_CACHE` or `~/.bbcache`; `BEDBASE_API` changes the endpoint. Do not
read unrelated environment variables. Set an explicit project cache, estimate
size, approve identifiers/endpoints, and verify returned checksums before use.

Local inspection commands are safer:

```text
geniml bbclient seek ID --cache-folder /absolute/project/cache
geniml bbclient inspect-bedfiles --cache-folder /absolute/project/cache
geniml bbclient inspect-bedsets --cache-folder /absolute/project/cache
```

The `cache-bed`, `cache-bedset`, and `cache-tokens` subcommands may use the
network. Do not run them implicitly or include sensitive local BED files in an
upload/cache workflow.

## Local audit and planning CLIs

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

Use `--help` for resource limits and explicit path-disclosure controls.

## References

- Region2Vec (upstream reference file, not included): modern API, artifacts, CLI drift,
  training, encoding, and evaluation.
- scEmbed (upstream reference file, not included): AnnData/token preparation, training,
  inference, annotation, privacy, and leakage.
- BEDspace (upstream reference file, not included): metadata schema, exact legacy CLI,
  StarSpace status, artifacts, and retrieval.
- Consensus peaks (upstream reference file, not included): coverage prerequisites,
  CC/CCF/ML/HMM, assessment, and assembly safeguards.
- Utilities (upstream reference file, not included): I/O, Gtars tokenizers, BBClient,
  evaluation, model safety, migration, and dated sources.

Source snapshot and primary-paper links are dated in
(upstream reference file, not included) (upstream reference file, not included). Re-check release metadata
and installed signatures before changing the pinned versions.
