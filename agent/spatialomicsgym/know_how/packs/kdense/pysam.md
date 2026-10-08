# pysam

## Metadata

**Short Description**: Python/HTSlib workflows for genomic files. Use when reading, querying, filtering, or writing SAM/BAM/CRAM, VCF/BCF, FASTA/FASTQ, or tabix data with pysam, including pileup, coverage, indexing, and CRAM references.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/pysam/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: not stated upstream for the tool; the upstream frontmatter `license` field reads "MIT", and upstream uses that field for the skill text in some files and for the tool in others, so it is not taken as the tool's licence
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 1 install fence(s) replaced by a provisioning pointer; 27 upstream script/reference path(s) marked as not vendored; 1 fence(s) left with nothing runnable replaced by a pointer.

---

## Overview

Use pysam for low-level, streaming access to HTSlib-supported genomic formats:

- `AlignmentFile` and `AlignedSegment` for SAM/BAM/CRAM
- `VariantFile`, `VariantHeader`, and `VariantRecord` for VCF/BCF
- `FastaFile` for indexed FASTA and `FastxFile` for sequential FASTA/FASTQ
- `TabixFile` for BGZF-compressed, tabix-indexed BED/GFF/GTF/custom tables
- `pysam.samtools` and `pysam.bcftools` for wrapped command dispatchers

Current upstream baseline: **pysam 0.24.0** (27 April 2026), wrapping
HTSlib/samtools/bcftools 1.23.1. Read (upstream reference file, not included) before updating
version-specific guidance.

## Installation

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

Confirm the runtime:

```python
import pysam

print(pysam.__version__)           # 0.24.0
print(pysam.__samtools_version__)  # 1.23.1
```

Prebuilt wheels are available for supported macOS and Linux platforms. A
source build needs a C compiler and HTSlib build dependencies; read the
official installation guide linked from (upstream reference file, not included).

## First Decide

Before writing code:

1. Identify the real format, compression, sort order, and available index.
2. Decide whether coordinates are numeric Python coordinates or a region
   string. Do not mix them.
3. For CRAM, identify the exact reference assembly and FASTA.
4. Prefer indexed region access; use sequential iteration only when intended.
5. Preserve headers when writing and write to a new path by default.
6. State filtering semantics: mapping/base quality, flags, overlap handling,
   duplicate handling, and pileup depth cap.

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

## Bundled Scripts

| Script | Purpose | Typical call |
|---|---|---|
| (upstream helper script, not vendored) | Metadata-only inspection for alignment, variant, FASTA, FASTQ, and tabix files | (upstream helper script, not vendored) |
| (upstream helper script, not vendored) | Streaming aggregate read/QC counts as JSON | (upstream helper script, not vendored) |
| (upstream helper script, not vendored) | Streaming variant, FILTER, and genotype summary as JSON | (upstream helper script, not vendored) |
| (upstream helper script, not vendored) | Filter SAM/BAM/CRAM without changing record order | (upstream helper script, not vendored) |

All scripts refuse to overwrite existing outputs. Run each with `--help` for
coordinate, index, and privacy notes.

## Coordinate Contract

**Numeric coordinates accepted by pysam APIs are 0-based, half-open.** This
includes numeric `AlignmentFile.fetch()`, `VariantFile.fetch()`,
`FastaFile.fetch()`, `TabixFile.fetch()`, and `pileup()` arguments.

**Region strings are samtools-style: 1-based and inclusive.**

```python
# The same 100 bases:
bam.fetch("chr1", 99, 199)          # [99, 199)
bam.fetch(region="chr1:100-199")    # 1-based inclusive
```

VCF text uses 1-based `POS`, while record properties expose both systems:

```python
record.pos    # 1-based
record.start  # 0-based inclusive
record.stop   # 0-based exclusive
```

Read (upstream reference file, not included) for format conversions, overlap
semantics, index choices, and contig-name checks.

## Alignment Files

Use context managers and explicit modes:

```python
import pysam

with pysam.AlignmentFile("sample.bam", "rb", threads=4) as bam:
    for read in bam.fetch("chr1", 1_000, 2_000):
        if (
            not read.is_unmapped
            and not read.is_secondary
            and not read.is_supplementary
            and read.mapping_quality >= 30
        ):
            print(read.query_name, read.reference_start, read.cigarstring)
```

Use `fetch(until_eof=True)` to stream every record in file order, including
unplaced unmapped reads, without requiring an index:

```python
with pysam.AlignmentFile("sample.bam", "rb") as bam:
    for read in bam.fetch(until_eof=True):
        ...
```

Important distinctions:

- `fetch()` returns alignment records overlapping a region.
- `count()` counts records and defaults to `read_callback="nofilter"`.
- `count_coverage()` returns A/C/G/T base counts and defaults to base quality
  15 plus `read_callback="all"`.
- `pileup()` exposes per-column reads and has its own filtering, base-quality,
  overlap, orphan, and `max_depth=8000` defaults.

For exact-region pileups, set `truncate=True` and explicit filters:

```python
with pysam.FastaFile("reference.fa") as fasta, pysam.AlignmentFile(
    "sample.bam", "rb"
) as bam:
    for column in bam.pileup(
        "chr1",
        1_000,
        2_000,
        truncate=True,
        stepper="samtools",
        fastafile=fasta,
        min_mapping_quality=20,
        min_base_quality=20,
        max_depth=100_000,
    ):
        print(column.reference_pos, column.get_num_aligned())
```

Read (upstream reference file, not included) for flags, CIGAR operations, tags,
modified bases, writing records, pileup details, and iterator lifetime.

## Variant Files

Input format is auto-detected. Numeric fetch coordinates remain 0-based:

```python
import pysam

with pysam.VariantFile("cohort.vcf.gz", threads=4) as variants:
    for record in variants.fetch("chr1", 999_999, 2_000_000):
        print(record.contig, record.pos, record.ref, record.alts)
        for sample_name, call in record.samples.items():
            print(sample_name, call.get("GT"))
```

Subset samples **before retrieving records**:

```python
with pysam.VariantFile("cohort.bcf") as variants:
    variants.subset_samples(["sample_A", "sample_B"])
    for record in variants:
        ...
```

When changing a header, copy each record and translate it to the destination
header before assigning newly declared INFO/FORMAT/FILTER fields. Do not
manually clear and rebuild `header.samples`.

Read (upstream reference file, not included) for safe headers, writing, sample
subsetting, missing genotypes, symbolic alleles, filtering, translation, and
indexing.

## FASTA, FASTQ, and Tabix

Indexed FASTA uses numeric 0-based coordinates:

```python
with pysam.FastaFile("reference.fa") as fasta:
    sequence = fasta.fetch("chr1", 999, 1_099)
```

`FastxFile` is sequential. `persist=False` is faster but yielded records become
invalid after iteration advances:

```python
with pysam.FastxFile("reads.fastq.gz", persist=False) as reads:
    for read in reads:
        qualities = read.get_quality_array()
        ...
```

Tabix input must be coordinate-sorted and BGZF-compressed, not ordinary gzip.
Use a non-destructive two-step workflow:

```python
pysam.tabix_compress("regions.bed", "regions.bed.gz")
pysam.tabix_index("regions.bed.gz", preset="bed")

with pysam.TabixFile("regions.bed.gz", parser=pysam.asBed()) as tbx:
    for interval in tbx.fetch("chr1", 1_000, 2_000):
        print(interval.contig, interval.start, interval.end)
```

Read (upstream reference file, not included) for FASTA/FASTQ records and safe tabix
creation.

## CRAM, Remote I/O, and Threads

pysam 0.24 changed inherited HTSlib behavior:

- Newly written CRAM defaults to CRAM 3.1, not 3.0.
- HTSlib no longer contacts the EBI reference server by default.
- Prefer `reference_filename="reference.fa"` for deterministic local reads and
  writes.

```python
with pysam.AlignmentFile(
    "sample.cram",
    "rc",
    reference_filename="reference.fa",
    threads=4,
) as cram:
    for read in cram.fetch("chr1", 1_000, 2_000):
        ...
```

Only configure `REF_PATH`/`REF_CACHE` when reference-by-MD5 lookup is
intentional. Do not assume a CRAM is self-contained. `threads=` accelerates
compression/decompression; it does not parallelize Python analysis.

Read (upstream reference file, not included) before CRAM conversion, remote access,
or concurrent iteration.

## Wrapped samtools and bcftools

Import command modules explicitly. Pass each command-line token as a separate
string:

```python
import pysam.samtools
import pysam.bcftools

pysam.samtools.sort(
    "-@", "4", "-o", "sorted.bam", "input.bam", catch_stdout=False
)
pysam.samtools.index("-@", "4", "sorted.bam", catch_stdout=False)

pysam.bcftools.index("--csi", "variants.vcf.gz", catch_stdout=False)
```

Dispatchers capture stdout by default. For large or binary output, use the
tool's `-o` option with `catch_stdout=False`, or `save_stdout=...`, rather than
returning the complete output in memory.

```python
try:
    pysam.samtools.quickcheck("-v", "sample.bam")
except pysam.SamtoolsError as error:
    messages = pysam.samtools.quickcheck.get_messages()
    raise RuntimeError(messages or str(error)) from error
```

Use the Python API for record-level logic and dispatchers for mature bulk
operations such as sort, index, merge, view, and normalization. Never compose
dispatcher arguments by splitting an untrusted shell command.

## Writing Rules

- Copy or construct a valid header before opening output.
- Write to a new path; do not use `force=True` unless replacement is explicit.
- Preserve sort order if the output will be indexed.
- Set `query_sequence` before `query_qualities`.
- Prefer `pysam.CIGAR_OPS` enum members; top-level constants such as
  `pysam.CMATCH` are compatibility aliases slated for future removal.
- Validate outputs with `pysam.samtools.quickcheck()` for alignments and reopen
  variant/sequence outputs before downstream use.
- Use CSI rather than BAI/TBI when references or coordinates exceed legacy
  index limits.

## Reference Map

| Need | Read |
|---|---|
| Alignment API, flags, CIGAR, pileup, modified bases | (upstream reference file, not included) |
| VCF/BCF headers, records, samples, writing | (upstream reference file, not included) |
| FASTA/FASTQ and tabix-indexed tables | (upstream reference file, not included) |
| Coordinate conversion and index selection | (upstream reference file, not included) |
| CRAM references, remote I/O, threads, performance | (upstream reference file, not included) |
| Correct integrated analysis patterns | (upstream reference file, not included) |
| Compact current API signatures and defaults | (upstream reference file, not included) |
| Upgrade notes for existing environments | (upstream reference file, not included) |
| Official docs, specifications, and release sources | (upstream reference file, not included) |

## Common Failure Modes

- Treating numeric `VariantFile.fetch()` coordinates as 1-based
- Using ordinary gzip where BGZF plus tabix/CSI is required
- Calling region fetch without an index
- Assuming `fetch()` includes unplaced unmapped alignments
- Forgetting `truncate=True` for an exact pileup interval
- Ignoring pileup defaults such as base quality 13 and depth cap 8000
- Sharing one file handle across active iterators or threads
- Decoding CRAM without its exact reference
- Assigning a new VCF field before declaring it in the output header
- Capturing large samtools/bcftools output in memory
- Using a SNP base-counting method for indels or symbolic alleles
