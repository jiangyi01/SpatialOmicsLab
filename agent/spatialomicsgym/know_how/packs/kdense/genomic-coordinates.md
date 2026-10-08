# Genomic Coordinates

## Metadata

**Short Description**: Convert genomic intervals between coordinate conventions, normalise and compare variant representations, and detect assembly or contig-naming mismatches before they corrupt an analysis. Use whenever coordinates cross a format, tool, or assembly boundary - converting between BED, GFF/GTF, VCF, SAM/BAM, WIG, PSL, genePred, Picard interval_list, or region strings; reconciling 0-based half-open with 1-based inclusive; left-aligning or trimming indels; checking whether two variant records describe the same change; mapping genomic to transcript, CDS, or protein positions; auditing a BED/GTF/VCF for convention violations; or diagnosing GRCh37 vs hg19 vs GRCh38 vs T2T, chr-prefix, and liftover problems. Triggers include "off by one", "0-based", "1-based", "half-open", "coordinate system", "left-align", "normalize variant", "bcftools norm", "chr prefix", "wrong genome build", "liftover", "REF mismatch", and "HGVS".
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/genomic-coordinates/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: not stated upstream for the tool; the upstream frontmatter `license` field reads "MIT", and upstream uses that field for the skill text in some files and for the tool in others, so it is not taken as the tool's licence
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 21 upstream script/reference path(s) marked as not vendored; 4 fence(s) left with nothing runnable replaced by a pointer.

---

## When to use

Any time a coordinate crosses a boundary: between two file formats, between two
tools, between two assemblies, or between the genome and a transcript.

## The rule

**A coordinate is three facts, not one: the number, the convention it is written
in, and the assembly it was measured against.** Carry all three or the number is
not interpretable.

Coordinate errors are the quietest class of bug in genomics. An off-by-one BED
file parses, sorts, and intersects without complaint. A GRCh37 VCF joined against
a GRCh38 annotation returns rows. A right-shifted indel simply fails to match its
entry in ClinVar, and the result is a variant reported as novel. Nothing raises
an error; the answer is just wrong, and it is wrong in a direction that looks
plausible.

So: convert with the table, not from memory, and verify against the reference
whenever a reference is available.

## The two conversions

```
1-based inclusive  ->  0-based half-open :  start - 1,  end
0-based half-open  ->  1-based inclusive :  start + 1,  end
```

The end coordinate never moves. If a conversion changed both numbers, it is wrong.

## Which format is which

| 0-based, half-open | 1-based, inclusive |
| --- | --- |
| BED, bedGraph, bigWig, narrowPeak | GFF3, GTF, VCF |
| BAM/CRAM (binary POS) | SAM (text POS) |
| PSL, genePred, refFlat | WIG, Picard interval_list |
| MAF (UCSC multiple alignment) | MAF (TCGA mutation annotation) |
| PyRanges, pybedtools | GRanges/IRanges, samtools & UCSC & Ensembl region strings |

Both "MAF" formats exist, they mean different things, and they disagree. UCSC
serves 0-based files through a 1-based browser box. (upstream reference file, not included)
has the full table with per-format detail.

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

```
contig  input                 output           length  status  detail
chr7    chr7:5530601-5530625  5530600-5530625  25      ok
```

Zero-length BED features (`chromStart == chromEnd`, a legal insertion point) are
reported as `unrepresentable` rather than converted to `end = start - 1`. Exit
code is 1 when any interval is degenerate or invalid.

## Variants are not intervals

A VCF `POS` for an indel is the **anchor base** — the base *before* the event,
itself unchanged. And the same change can be written many ways:
`chr1:7:CAC:C`, `chr1:3:CAC:C` and `chr1:2:GCA:G` are one deletion. Joining,
deduplicating, or looking up variants before normalising loses real matches
silently, and it loses them preferentially in repeats, where indels concentrate.

Normalise — trim to parsimony, then left-align against the reference — before any
comparison:

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

```
input         normalized    type      pos_shift  ref_check  changed
chr1:7:CAC:C  chr1:2:GCA:G  deletion  5          ok         yes
```

Every record's `REF` is checked against the FASTA first. A `MISMATCH` means the
variants and the reference are different assemblies — stop and run
`check_contigs.py` rather than adjusting coordinates. Multi-allelic records must
be split with `--split` **before** normalising, never after.

HGVS shifts indels the opposite way, 3'-most along the transcript. For a
minus-strand gene that is the opposite genomic direction from VCF's
left-alignment. Details and the full procedure: (upstream reference file, not included).

## Check the assembly before trusting a join

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

```
file          kind    contigs  naming        assembly  detail
ref.fa.fai    sizes   25       plain         GRCh37    24/24 primary chromosome lengths match;
                                                       chrM is 16569 bp, i.e. GRCh37/38 (rCRS MT)
```

The script reads `.fai`, `.chrom.sizes`, VCF headers, SAM headers, FASTA, BED,
and GTF/GFF, identifies the assembly from primary-chromosome lengths, and reports
every reason a join between two files would go wrong: naming mismatch, length
conflict, coordinates past a contig end, contigs present in one file only. Exit
code 1 on any incompatibility.

**GRCh37 and hg19 differ only in the mitochondrion** — 16,569 bp (rCRS) versus
16,571 bp. Nuclear coordinates are identical, so a mixed pipeline runs fine and
only the mtDNA results are wrong. `check_contigs.py` reports which one it found.
Builds, naming schemes, ALT contigs, and liftover pitfalls:
(upstream reference file, not included).

## Audit a file against its own format

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

Looks for the evidence that a coordinate mistake leaves behind:

| Finding | What it proves |
| --- | --- |
| `start_below_one` in GFF/GTF | 0-based data in a 1-based file; everything is one base left |
| `many_zero_length` in BED | 1-based single-base features written into a 0-based file |
| `past_contig_end` | wrong assembly, or an off-by-one at the contig edge |
| `mixed_contig_naming` | any join will silently match one subset |
| `first_block_offset` | BED12 `blockStarts` written as absolute coordinates |
| `not_parsimonious` | untrimmed alleles; normalise before joining |
| `bad_alt_allele` | Ensembl/VEP `-` notation in a VCF, which has no anchor base |

Exit code 1 on any fatal finding, so it works as a CI gate on a data directory.

## Transcript, CDS, and protein positions

`c.742` and `chr17:7,674,220` are both "position", and neither converts to the
other by arithmetic. Transcript coordinates count spliced bases in transcription
order — decreasing genomic coordinate on the minus strand — and `c.1` is the `A`
of the initiator `ATG`, not the start of the transcript.

The rules that get mis-remembered: there is no `c.0`; 5' UTR positions are
negative and 3' UTR positions take a `*`; GFF phase is the bases to *remove* to
reach the next codon, not `start % 3`; and a `c.` description is meaningless
without a versioned transcript accession, because the same variant numbers
differently in each transcript. (upstream reference file, not included) has the
conversion procedure and the boundary cases.

Do the conversion with a tool that holds the transcript model — VEP,
`bcftools csq`, Mutalyzer, the `hgvs` package — not by hand.

## Reporting results

State the assembly next to the coordinates, every time.
`chr7:5,530,601-5,530,625` is not a location; `chr7:5,530,601-5,530,625 (GRCh38)`
is. Say which convention a coordinate column is in, in the column header or the
file's documentation. When a conversion produced a result, say which direction it
went.

## References

- (upstream reference file, not included) — every format's convention, with per-format
  detail, BED12 block rules, region-string syntax, and tool behaviour.
- (upstream reference file, not included) — VCF allele conventions, the
  normalisation algorithm, equivalence checking, multi-allelic splitting, and how
  HGVS disagrees with VCF.
- (upstream reference file, not included) — build signatures, GRCh37 vs hg19, ALT
  contigs, naming schemes, and liftover failure modes.
- (upstream reference file, not included) — genomic ↔ transcript ↔ CDS ↔ protein,
  HGVS numbering, phase, and transcript choice.
