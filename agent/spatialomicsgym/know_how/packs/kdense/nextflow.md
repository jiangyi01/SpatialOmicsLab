# Nextflow

## Metadata

**Short Description**: Build, run, and debug Nextflow data pipelines and nf-core workflows end to end. Use whenever the user mentions Nextflow, nf-core, .nf files, nextflow.config, DSL2, processes/channels/operators, samplesheets, or wants to run a community pipeline (e.g. nf-core/rnaseq, nf-core/sarek), write or test a module/subworkflow with nf-test, configure executors/containers (Docker, Singularity/Apptainer, Conda, Wave), scale a workflow to HPC/SLURM or cloud (AWS Batch, Google Batch, Azure, Kubernetes), or debug a failed/-resume run. Make sure to use this skill for any reproducible scientific/bioinformatics workflow work even if the user does not say the word "Nextflow", and for authoring nf-core-compliant pipelines, modules, configs, and linting.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/nextflow/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: not stated upstream for the tool; the upstream frontmatter `license` field reads "Apache-2.0", and upstream uses that field for the skill text in some files and for the tool in others, so it is not taken as the tool's licence
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 3 install fence(s) replaced by a provisioning pointer; 22 upstream script/reference path(s) marked as not vendored; 1 manifest replacement(s) and 0 excision(s) applied.

---

## Overview

Nextflow is a workflow language and runtime for building **reproducible, portable, scalable** data pipelines. It is dominant in bioinformatics but works for any data-heavy computation. nf-core is a community curating production-grade Nextflow pipelines, reusable modules, and the `nf-core` tooling on top of Nextflow.

Key ideas:
- **Dataflow programming**: pipelines are `process` tasks connected by **channels**. Nextflow infers execution order and parallelism from data dependencies — there is no explicit scheduler to write.
- **Write once, run anywhere**: the same pipeline runs locally, on HPC (SLURM, SGE, LSF, PBS), and on cloud (AWS Batch, Google Batch, Azure Batch, Kubernetes) by changing config/profiles, not code.
- **Reproducibility**: per-task containers (Docker/Singularity/Apptainer/Conda/Wave) + `-resume` caching + pinned pipeline revisions.
- **DSL2** is the modern, required syntax: modular `process`/`workflow`/`include` definitions.

This skill covers both **running** existing pipelines and **developing** your own (Nextflow language + nf-core conventions, testing with nf-test, configuration, and deployment).

## When to Use This Skill

Use this skill when the user wants to:
- Run an nf-core or custom Nextflow pipeline, or debug a failing/resuming run.
- Write or modify `.nf` scripts, `nextflow.config`, profiles, or `nextflow_schema.json`.
- Author or test nf-core-style modules/subworkflows (`main.nf`, `meta.yml`, `tests/`, nf-test).
- Configure executors, containers, or resources; scale to HPC or cloud.
- Build a reproducible scientific/bioinformatics workflow (even if "Nextflow" is not named).
- Understand processes, channels, operators, `take`/`emit`, `publishDir`, `ext.args`, meta maps.

## Setup

Nextflow needs **Bash** and **Java 17 or newer** (17–25 supported). Verify with `java -version`.

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

Pin the engine for reproducibility: `export NXF_VER=24.10.0` (use an [edge] release only if needed). For air-gapped/HPC, see (upstream reference file, not included) (offline mode) and (upstream reference file, not included).

## Two Modes of Work

Decide which path the user is on — it changes everything:

| Goal | Start here |
|------|-----------|
| **Run** an existing pipeline (nf-core or a `.nf` you were given) | (upstream reference file, not included) |
| **Develop** a new pipeline / module / subworkflow | (upstream reference file, not included) + (upstream reference file, not included) |
| **Configure / scale** (HPC, cloud, containers, resources) | (upstream reference file, not included) + (upstream reference file, not included) |
| **Test** modules/pipelines | (upstream reference file, not included) |

## Quick Start

### Run an nf-core pipeline

Always smoke-test with the bundled `test` profile first; it uses tiny data and proves your environment works.

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

- `-profile` (single dash) selects bundled config profiles; **combine** them comma-separated, e.g. `test,docker`. Container/infra profiles (`docker`, `singularity`, `conda`) are mutually exclusive — pick one.
- `--input`, `--genome`, `--outdir` (double dash) are **pipeline** parameters. nf-core pipelines take a **samplesheet CSV**, not loose files.
- `-resume` reuses cached results from the last run. `-r <version>` pins a release for reproducibility.

Use `nf-core pipelines launch <name>` for an interactive, schema-validated way to build the command and a `-params-file`. See (upstream reference file, not included).

### Write a minimal pipeline

```nextflow
#!/usr/bin/env nextflow

process SAYHELLO {
    tag "$greeting"
    publishDir "results", mode: 'copy'

    input:
    val greeting

    output:
    path "${greeting}.txt"

    script:
    """
    echo '$greeting world' > ${greeting}.txt
    """
}

workflow {
    channel.of('hello', 'bonjour', 'hola') | SAYHELLO
}
```

```bash
nextflow run main.nf            # add -resume on reruns
```

The full language (processes, channels, operators, DSL2 workflows with `take`/`main`/`emit`, modules) is in (upstream reference file, not included).

## Core Concepts at a Glance

- **Process**: a unit of work that runs a script (Bash by default). Declares `input:`, `output:`, optional `directives` (resources, container, `publishDir`, `tag`, `errorStrategy`), and a `script:`/`shell:`/`exec:` block. Each task runs in its own isolated work directory (`work/xx/yy…`).
- **Channel**: the async queues that connect processes. **Queue channels** are consumable streams; **value channels** hold a single reusable value. Created with factories like `channel.of`, `channel.fromPath`, `channel.fromFilePairs`, `channel.value`.
- **Operator**: transforms/combines channels — `map`, `filter`, `collect`, `groupTuple`, `join`, `combine`, `mix`, `flatten`, `branch`, `multiMap`, `splitCsv`, `view`, `set`.
- **Workflow**: composes processes. DSL2 workflows can declare `take:` (inputs), `main:` (logic), `emit:` (named outputs) and be `include`d as subworkflows. The unnamed `workflow {}` is the entry point.
- **Module**: a `.nf` file exposing processes/workflows via `include { NAME } from './path'` (supports `as` aliasing).
- **Configuration**: `nextflow.config` sets `params`, `process` directives, `executor`, container engines, and named `profiles`. Selectors `withName:`/`withLabel:` target specific processes. See (upstream reference file, not included).
- **meta map** (nf-core): the convention of carrying a metadata map (`[ id:'sample1', single_end:false ]`) alongside files in input/output tuples so samples stay labeled through the pipeline. See (upstream reference file, not included).

## nf-core tools CLI

nf-core tools (v3+) group subcommands under `pipelines`, `modules`, and `subworkflows`. (Bare forms like `nf-core lint` still work but warn — prefer the grouped form.)

| Command | Purpose |
|---------|---------|
| `nf-core pipelines list` | List/search nf-core pipelines (`--json`, keywords) |
| `nf-core pipelines create` | Scaffold a new pipeline from the nf-core template |
| `nf-core pipelines launch <name>` | Interactive, schema-driven run command + params file |
| `nf-core pipelines download <name>` | Download pipeline + containers for offline/HPC use |
| `nf-core pipelines lint` | Lint a pipeline against nf-core standards (run in repo root) |
| `nf-core pipelines schema build` | Build/edit `nextflow_schema.json` via web GUI |
| `nf-core pipelines create-params-file <name>` | Generate a documented YAML params file |
| `nf-core pipelines bump-version` / `sync` | Bump version / sync with template updates |
| `nf-core modules list/info/install/update/remove` | Manage modules from nf-core/modules |
| `nf-core modules create` / `lint` / `test` | Author, lint, and nf-test a module |
| `nf-core modules patch` / `bump-versions` | Patch an installed module / bump tool versions |
| `nf-core subworkflows install/create/lint/test` | Same lifecycle for subworkflows |

Full command reference, flags, and examples: (upstream reference file, not included).

## Essential `nextflow` CLI

| Command | Purpose |
|---------|---------|
| `nextflow run <pipeline> -profile <p> --outdir <dir>` | Run a pipeline (path, `.nf`, or `user/repo`) |
| `-resume` | Reuse cached results from prior run |
| `-r <rev>` | Run a specific git revision/tag/branch |
| `-params-file params.yml` | Supply parameters from YAML/JSON |
| `-c custom.config` | Layer in an extra config file |
| `-with-report -with-trace -with-timeline -with-dag flow.html` | Execution report, trace, timeline, DAG |
| `-stub-run` | Run `stub:` blocks only (dry-run plumbing) |
| `nextflow log` | Inspect past runs |
| `nextflow clean -f -before <run>` | Delete old `work/` data |
| `nextflow pull / drop / list / info <repo>` | Manage cached remote pipelines |

Config, executors, caching internals, and tracing details: (upstream reference file, not included).

## Best Practices (high-value habits)

- **Always `test` first**: run the pipeline's `test` profile before real data — fast and catches environment problems. Its container or Conda profiles pull images or build per-process environments, which this document does not do: run only where the user has already provisioned the engine.
- **Pin everything**: pipeline revision (`-r`), `NXF_VER`, and tool versions (containers). Don't run `latest` for science you'll publish.
- **Use `-resume`** and understand caching: a task re-runs if its inputs, script, or container change. See cache-debugging in (upstream reference file, not included).
- **Parameterize via config/params-file**, not hardcoded paths. Keep `params` and profiles in `nextflow.config`.
- **One container/conda env per process**; never rely on tools installed on the host.
- **For nf-core dev**: reuse existing modules (`nf-core modules install`) before writing new ones; pass tool flags through `ext.args` (not hardcoded in the script); always include a `stub:` block and nf-test tests; run `nf-core pipelines lint` and `prettier` before committing.
- **Right-size resources** with `process_low/medium/high` labels and `errorStrategy 'retry'` with dynamic `task.attempt` scaling instead of one giant request.
- **Write forward-compatible syntax**: the strict-syntax parser becomes the default in Nextflow 26.04. Prefer lowercase `channel.of(...)`, explicit closure params (`{ v -> ... }`), `def` for all variables, and `emit:`-named outputs. Check with `nextflow lint`.

## Reference Files

Read the relevant file when you need depth — each is self-contained:

- (upstream reference file, not included) — DSL2 language: processes, directives, channels, operators, workflows (`take`/`emit`), modules, dynamic resources, error handling.
- (upstream reference file, not included) — `nextflow.config`, scopes, `profiles`, `withName`/`withLabel` selectors, executors (local/SLURM/cloud), caching/`-resume` internals, tracing/reports, the `nextflow` CLI.
- (upstream reference file, not included) — Docker, Singularity/Apptainer, Podman, Conda, Wave containers; choosing and enabling engines; common gotchas.
- (upstream reference file, not included) — finding/running nf-core pipelines, samplesheets, params files, reference genomes (iGenomes), offline runs, institutional configs, Seqera Platform.
- (upstream reference file, not included) — complete `nf-core` CLI reference (pipelines/modules/subworkflows), flags, and workflows.
- (upstream reference file, not included) — authoring nf-core pipelines & modules: template layout, module `main.nf`/`meta.yml`, meta maps, `ext.args`/`modules.config`, subworkflows, resource labels, linting & Harshil alignment style.
- (upstream reference file, not included) — nf-test for modules/subworkflows/pipelines: test structure, assertions, snapshots, tags, running tests, CI.

Official docs: Nextflow https://www.nextflow.io/docs/latest/ · nf-core https://nf-co.re/docs/ · Training https://training.nextflow.io/
