# LatchBio Integration

## Metadata

**Short Description**: Build, register, debug, and operate bioinformatics workflows on Latch using the Python SDK, CLI, Latch Data and Registry, Nextflow, Snakemake, programmatic execution, and Latch MCP. Use when authoring or deploying Latch workflows, configuring resources or interfaces, moving data, integrating Registry, or launching and monitoring runs.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/latchbio-integration/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: not stated upstream for the tool; the upstream frontmatter `license` field reads "MIT", and upstream uses that field for the skill text in some files and for the tool in others, so it is not taken as the tool's licence
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 3 install fence(s) replaced by a provisioning pointer; 10 upstream script/reference path(s) marked as not vendored.

---

## Current Baseline

This skill targets **Latch SDK 2.76.8**, released July 10, 2026. The package
metadata supports Python 3.9–3.12 and declares Python 3.9+.

Treat the installed package and its changelog as authoritative when a guide
disagrees with the SDK. Some Latch guides retain older Python ranges or
compatibility-specific pre-release pins, especially the Snakemake v2 tutorial.
Never combine commands or imports from different tracks without checking their
version requirements.

## When to Use

Use this skill to:

- Create or maintain Python SDK workflows and task graphs
- Package and register Python, Nextflow, or Snakemake pipelines
- Configure task CPU, memory, storage, GPU, caching, retries, and timeouts
- Work with Latch Data through `LPath`, `LatchFile`, `LatchDir`, or the CLI
- Read or update Latch Registry projects, tables, and records
- Design workflow forms, launch plans, samplesheets, messages, and result links
- Stage and debug workflow images with `latch register --staging` and `latch develop`
- Launch and monitor workflows through Python or Latch MCP
- Discover and use ready-to-run Latch workflows

## Route to the Right Reference

Read only the references needed for the task:

| Need | Reference |
|---|---|
| Python workflows, tasks, maps, conditions, caching | (upstream reference file, not included) |
| `LPath`, legacy file types, Latch URLs, data CLI | (upstream reference file, not included) |
| Registry reads, transactions, samplesheets | (upstream reference file, not included) |
| CPU, memory, storage, GPU, dynamic resources | (upstream reference file, not included) |
| Nextflow and Snakemake packaging | (upstream reference file, not included) |
| Metadata, forms, launch plans, messages, automations | (upstream reference file, not included) |
| Registration, development, execution, monitoring | (upstream reference file, not included) |
| Ready-to-use workflows and `latch.verified` | (upstream reference file, not included) |
| Remote MCP setup and tool workflow | (upstream reference file, not included) |

Before relying on a symbol, run (upstream helper script, not vendored) against the
target SDK version. It performs local imports only and does not authenticate or
make network requests.

## Installation and Authentication

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

On Windows, use WSL for the documented Linux workflow tooling.

Authenticate through the supported OAuth flow; do not read, print, copy, or
parse `~/.latch/token` manually:

```bash
latch login
latch workspace
```

Select a workspace non-interactively when its numeric ID is already known:

```bash
latch workspace --id 12345
```

`latch login` credentials are for the SDK and CLI. Latch MCP uses a separate
OAuth authorization and its credentials cannot be reused for general SDK
access.

## Fast Path

Create and remotely register the maintained subprocess template:

```bash
latch init covid-wf --template subprocess
latch register --yes --open covid-wf
```

Remote image building is the default. Use `--no-remote` only when a local
Docker daemon is available and a local build is intentional.

## Minimal Python Workflow

Keep workflow bodies declarative: invoke tasks and return their promises.
Perform computation and side effects inside tasks.

```python
from latch import small_task, workflow


@small_task
def reverse_complement(sequence: str) -> str:
    table = str.maketrans("ACGTacgt", "TGCAtgca")
    return sequence.translate(table)[::-1]


@workflow
def reverse_complement_workflow(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return reverse_complement(sequence=sequence)
```

Use `@workflow(metadata)` when the generated interface needs custom labels,
sections, validation rules, samplesheets, or documentation links. Use `LatchFile` or
`LatchDir` for automatic task input staging and output upload; use `LPath` for
imperative remote path operations.

## Recommended Development Lifecycle

1. **Inspect compatibility**
   - Confirm the installed SDK and Python version.
   - Identify whether the project is Python, Nextflow, the legacy Snakemake
     flag path, or the separately pinned Snakemake v2 tutorial track.

2. **Define a typed interface**
   - Annotate every workflow and task input and output.
   - Keep module import time free of network calls, data mutations, and secret
     retrieval. Isolate documented exceptions such as `workflow_reference`,
     which resolves the active workspace when its decorator is evaluated.
   - Use dataclasses and enums for structured parameters.

3. **Configure metadata and resources**
   - Match metadata parameter keys to the workflow signature.
   - Start with named task decorators, then use `custom_task` only when measured
     requirements justify it.

4. **Validate in the execution image**

   Fresh Nextflow and Snakemake projects must generate their
   version-compatible Python entrypoint before staging. In SDK 2.76.8, the
   staging branch does not generate one from `--nf-script` or `--snakefile`.

   ```bash
   latch register --staging .
   latch develop .
   ```

   Re-run staging registration after changing the Dockerfile or dependencies.
   Edits made inside the development container are not synced back.

5. **Register deliberately**

   ```bash
   latch register --yes --open .
   ```

   Useful controls:

   ```bash
   latch register --workspace-id 12345 .
   latch register --mark-as-release .
   latch register --workflow-module wf.custom_entrypoint .
   ```

   Duplicate registration exits with status `2`; it is not the same as a build
   failure.

6. **Launch only after reviewing cost and parameters**
   - Prefer the Console or Latch MCP for interactive operation.
   - Prefer `latch_cli.services.launch.launch_v2` for Python automation.
   - Do not use the deprecated `latch launch` CLI as a new integration pattern.

7. **Monitor and verify**
   - Check terminal status, task logs, result links, and scientific outputs.
   - Treat successful orchestration as necessary but not sufficient scientific
     validation.

## Operational Safety

- Ask for confirmation before launching paid compute, especially GPU or large
  batch runs.
- Ask for confirmation before `LPath.rmr`, `latch rmr`, Registry deletion, or
  overwriting shared destinations.
- Never log secrets, SDK tokens, signed URLs, or secret values.
- Call `get_secret()` only inside a task, use the returned value only for its
  intended service, and never return it as workflow output.
- Do not pass untrusted strings through shell commands. Prefer argument lists
  with `subprocess.run(..., check=True)`.
- Pin the SDK and workflow dependencies for releases. Upgrade only after
  reviewing the changelog and re-running staging tests.
- Treat generated files as generated: customize the documented extension file
  rather than editing output that the CLI will overwrite.

## Inspect the Installed SDK

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

## Authoritative Sources

- Documentation index: https://wiki.latch.bio/llms.txt
- Workflow and SDK guides: https://wiki.latch.bio/workflows/overview
- SDK API reference: https://wiki.latch.bio/reference/sdk
- PyPI package: https://pypi.org/project/latch/
- SDK 2.76.8 release source: https://github.com/latchbio/latch/tree/0faa9dcd8186444ac008f50adf95d43f0fa30e06
- SDK changelog: https://github.com/latchbio/latch/blob/0faa9dcd8186444ac008f50adf95d43f0fa30e06/CHANGELOG.md
- Latch Console: https://console.latch.bio
