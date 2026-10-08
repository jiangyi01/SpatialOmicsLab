# Gtars

## Metadata

**Short Description**: Use Gtars for local genomic interval models and set algebra, overlaps and counts, consensus and coverage, tokenization, fragment processing, and refget/BEDbase planning across Python, Rust, and the CLI.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/gtars/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: MIT per the published gtars crates; the GitHub repository root displays BSD-2-Clause -- verify the exact artifact (per this document's body) (recorded in packs/MANIFEST.yaml)
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 2 install fence(s) replaced by a provisioning pointer; 16 upstream script/reference path(s) marked as not vendored; 2 fence(s) left with nothing runnable replaced by a pointer; 1 manifest replacement(s) and 0 excision(s) applied; an Overview heading added above the opening prose.

---

## Overview

Gtars provides native Rust implementations, Python bindings, and a feature-gated
`gtars` binary for genomic interval and reference-sequence work. Start with the
bundled local inspectors; call upstream code only after the data contract,
provenance, resource bounds, and side effects are explicit.

## Verified snapshot (2026-07-23)

- Python: [`gtars==0.9.2`](https://pypi.org/project/gtars/), released
  2026-06-17, `Requires-Python >=3.10`.
- Rust meta-crate: [`gtars=0.9.0`](https://crates.io/crates/gtars), released
  2026-06-15. Its default feature set is empty.
- CLI crate/binary: [`gtars-cli=0.9.0`](https://crates.io/crates/gtars-cli);
  the installed binary is named `gtars`.
- Direct refget crate: [`gtars-refget=0.9.1`](https://crates.io/crates/gtars-refget),
  released 2026-06-17. `gtars=0.9.0` itself pins its component release set, which
  includes refget 0.9.0.
- Upstream intentionally versions workspace crates, Python bindings, and CLI
  independently. Do not assume matching numbers mean matching artifacts.
- The published docs changelog stops at 0.5.1. API examples here were checked
  against the 0.9.2 Python stubs/runtime and the `v0.9.0` CLI/Rust source.

The `license: MIT` field covers this skill. Published `gtars` crates declare MIT,
while the GitHub repository currently displays BSD-2-Clause at the root; verify
the exact artifact's license before redistribution.

## Native-code trust gate and exact pins

The Python wheel contains a PyO3 native extension. Cargo installation compiles a
native binary and can run dependency build scripts. Treat either path as code
execution:

1. Confirm the official PyPI/crates.io/GitHub owner and immutable version.
2. Review filenames, platform tags, release provenance, license, and SHA-256.
   GitHub's v0.9.0 binary release includes per-archive `.sha256` sidecars.
3. Never run an untrusted prebuilt binary, wheel, source tree, Cargo build script,
   or archive installer. Use isolation and CPU/RAM/disk/time limits.
4. Keep a lockfile and artifact hashes with the analysis manifest.

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

For a Rust project, pin the wrapper exactly and enable only required features:

```toml
[dependencies]
gtars = { version = "=0.9.0", default-features = false, features = [
  "core", "overlaprs", "uniwig", "tokenizers", "refget"
] }
```

Use `gtars-refget = "=0.9.1"` directly only when the newer direct component API is
required and compatibility has been tested. Do not replace these pins with a Git
branch or an unreviewed release.

## Genomic data contract

Apply this contract before every operation:

1. **Coordinates:** BED intervals are 0-based and half-open: `[start, end)`.
   Require `0 <= start < end <= contig_length`. Gtars coordinates are `u32`, so
   reject values above `4,294,967,295`.
2. **Assembly:** record an assembly accession/version and the SHA-256 of the exact
   chromosome-sizes or refget sequence-collection metadata. Never infer assembly
   from filenames or `chr` prefixes.
3. **Contigs:** compare names exactly. `1` and `chr1`, alternate loci, decoys, and
   mitochondrial aliases are not interchangeable. Rename or liftover only as a
   separately reviewed transformation.
4. **Sorting:** preserve the original file, then sort a copy by chromosome-sizes
   order and numeric start/end when the operation requires it. Python
   `RegionSet(path)` currently sorts lexicographically by contig and start while
   loading; do not rely on original row order afterward.
5. **Strand:** BED6 uses `+`, `-`, or `.`. `Region.rest` retains trailing BED
   fields, but a file-backed Python `RegionSet` currently initializes its separate
   `strands` vector to `*`. Several set operations drop strand. Preserve and
   validate strand externally when it is scientifically meaningful.
6. **Duplicates/adjacency:** choose policies explicitly. `reduce()` and consensus
   merge overlapping **and adjacent** intervals; ordinary half-open overlap does
   not treat `[0,10)` and `[10,20)` as overlapping.

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

## Safe local workflow

1. Inventory local files, checksums, assembly, contig dictionary, coordinate
   system, strand policy, patient/replicate groups, and intended outputs.
2. Validate BED/fragments and estimate work. Pilot a small synthetic file.
3. Choose Python, CLI, or Rust from the documented surface; do not translate API
   names by guesswork.
4. Set hard limits for input bytes/records/files, threads/jobs, memory, temporary
   disk, output size, and wall time.
5. Run in a dedicated output directory. Refuse collisions unless overwrite was
   explicitly approved.
6. Revalidate output sorting, bounds, row counts, checksums, and provenance.

## Current Python core

Imports are from submodules, not the `gtars` top level:

```python
from gtars.models import Region, RegionSet

query = RegionSet.from_regions(
    [
        Region(chr="chr1", start=100, end=200, rest=None),
        Region(chr="chr1", start=300, end=400, rest=None),
    ],
    strands=["+", "-"],
)
universe = RegionSet.from_vectors(
    ["chr1", "chr1"],
    [150, 500],
    [350, 600],
)

counts = query.count_overlaps(universe)       # one count per query region
flags = query.any_overlaps(universe)          # one bool per query region
indices = query.find_overlaps(universe)       # indices into universe
pieces = query.intersect_all(universe)        # all intersection fragments
fraction = query.coverage(universe)           # fraction of query bp covered
```

`RegionSet.sort()` mutates and returns `None`. Set algebra includes `reduce`,
`setdiff`, `pintersect` (pairs by index), `concat`, `union`, `jaccard`,
`coverage`, `overlap_coefficient`, `intersect_all`, `closest`, `cluster`, and
`gaps`. Read (upstream reference file, not included) before relying on ordering or strand.

Consensus is a Python binding in a different module:

```python
from gtars.genomic_distributions import consensus

rows = consensus([query, universe])
# rows: [{"chr": ..., "start": ..., "end": ..., "count": ...}, ...]
```

Signal-track generation is **not** exposed as `gtars.uniwig` in Python 0.9.2;
use the reviewed CLI or Rust API. `RegionSet.coverage()` is a base-pair set metric,
not a WIG/bigWig generator.

## Tokenizers, fragments, and reference stores

Use only local constructors by default:

```python
from gtars.models import RegionSet
from gtars.tokenizers import Tokenizer

tokenizer = Tokenizer.from_bed("reviewed-universe.bed")
regions = RegionSet("local-query.bed")
tokens = tokenizer.tokenize(regions)
encoding = tokenizer(regions)
ids = encoding["input_ids"]
```

`Tokenizer.from_pretrained(name)` contacts Hugging Face and writes its cache when
the argument is not an existing local directory; it exposes no revision or cache
argument. Obtain explicit approval, fetch an immutable revision through a reviewed
mechanism, verify checksums, then pass the local snapshot directory. See
(upstream reference file, not included).

For refget, prefer `RefgetStore.in_memory()` or `RefgetStore.open_local(path)`.
`open_remote(cache_path, remote_url)` contacts a remote service, creates/uses a
local cache, and performs on-demand range reads. See (upstream reference file, not included).

## Network and cache gate

No download or cache write is implicit in this skill. Before any network-capable
upstream call:

- obtain explicit user approval for the exact host, endpoint, data, and cache;
- allowlist HTTPS hosts and reject unreviewed redirects;
- record immutable revision/identifier, retrieval time, expected SHA-256 and
  domain digest, assembly accession, size quota, and provenance;
- disclose sensitive BED coordinates, barcodes, sample labels, and reference
  choices that could leave the approved environment;
- validate downloaded content as untrusted before using it.

Important side effects:

- `RegionSet(path)` has HTTP support; a nonexistent local string may be treated as
  a URL. Check that the local path exists before construction.
- `Tokenizer.from_pretrained` may download `universe.bed.gz` into the Hugging Face
  cache.
- `RefgetStore.on_disk` creates/writes a store. `open_remote` loads remote metadata
  and enables persistence by default.
- `gtars bbcache` creates cache directories even when constructing the client.
  Cache/download commands use `BBCLIENT_CACHE` (default `~/.bbcache`) and
  `BEDBASE_API` (default `https://api.bedbase.org`).

## Sensitive metadata and leakage

Genomic intervals, rare loci, barcodes, sample names, phenotypes, and assembly
choices can be identifying. Keep full paths and raw coordinates out of logs;
default bundled reports redact paths and emit only counts/checksums.

Freeze splits by patient/donor first, then keep all technical and biological
replicates in the same split. Fit consensus sets, universes, tokenizers, scaling,
thresholds, and QC rules on training data only. Do not create a universe from all
samples and then split: that leaks validation/test locus support. Record excluded
samples and replicate aggregation separately.

## Bundled deterministic CLIs

All six helpers reject URLs, traversal, symlinks, and special files; apply byte,
record, file, coordinate, and worker caps; use no network or gtars import; and
write no output files. Plans contain fixed argv templates and never launch them.

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

Run synthetic tests without bytecode:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s tests/gtars -p 'test_*.py' -v
```

## Migration traps removed in 1.1

Do not use stale examples containing `gtars.RegionSet`,
`RegionSet.from_bed`, `TreeTokenizer`, `gtars.igd.build_index`,
`gtars.uniwig.coverage_from_bed`, `gtars.RefgetStore`, global
`set_option`/`set_log_level`, `parallel_apply`, or invented exception classes.
CLI forms such as `uniwig generate`, `igd build`, `scoring score`, and
`fragsplit cluster-split` are also stale for 0.9.0.

Upstream's published docs and stubs have some drift (for example the older
`GlobalRefgetStore` tutorial and incomplete 0.9.2 stubs). Prefer installed
signature smoke tests plus immutable tagged source when they conflict.

## Upstream references (not included)

Upstream ships six reference files with this skill; none is included here:

- (upstream reference file, not included) — exact Python 0.9.2 imports and behavior
- (upstream reference file, not included) — overlap/count/set algebra and consensus semantics
- (upstream reference file, not included) — uniwig, bigWig, coverage, sorting, and resources
- (upstream reference file, not included) — tokenizer/universe and fragment compatibility
- (upstream reference file, not included) — digests, stores, BEDbase, network/cache controls
- (upstream reference file, not included) — CLI 0.9.0 commands, features, and migrations
