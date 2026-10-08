# DNAnexus Integration

## Metadata

**Short Description**: Build and operate reproducible genomics workloads on DNAnexus with the dx CLI, dxpy, apps/applets, native workflows, dxCompiler, and Nextflow. Use for DNAnexus data transfers, dxapp.json development, execution monitoring, workflow import, and project automation.
**Source**: https://github.com/k-dense-ai/scientific-agent-skills/blob/330c8e764435a731eff571e3efdda70b363d0792/skills/dnanexus-integration/SKILL.md
**License**: MIT, Copyright (c) 2025 K-Dense Inc. (THIRD_PARTY_LICENSES/scientific-agent-skills-MIT.txt). Changes were made -- see Modifications.
**Wrapped Tool License**: not stated upstream for the tool; the upstream frontmatter `license` field reads "MIT", and upstream uses that field for the skill text in some files and for the tool in others, so it is not taken as the tool's licence
**Commercial Use**: This text may be used commercially under its licence (see License above); the software it describes is governed by the Wrapped Tool License, not by this document.
**Tier**: 2
**Modifications**: re-headed under SpatialOmicsGym provenance by spatialomicsgym/know_how/merge_packs.py; upstream frontmatter reduced to this header; first H1 replaced by the title above; 1 upstream section(s) dropped (Citing Scientific Agent Skills, Integration with Other Skills); 5 install fence(s) replaced by a provisioning pointer; 23 upstream script/reference path(s) marked as not vendored; 1 manifest replacement(s) and 0 excision(s) applied.

---

## Purpose

Use this skill to build, run, and operate DNAnexus workloads without guessing
at platform semantics. It covers:

- `dx` CLI and `dxpy` automation
- Files, records, folders, projects, and metadata
- Apps and applets defined by `dxapp.json`
- Jobs, workflow analyses, retries, monitoring, and cost controls
- Native workflows, WDL/CWL through dxCompiler, and Nextflow imports

The documented baseline was verified on **2026-07-23** against
`dxpy==0.410.0`, dxCompiler 2.17.0, and the 2026 DNAnexus documentation.
Consult (upstream reference file, not included) and current release notes when behavior may
have changed.

## Operating Contract

DNAnexus operations can expose regulated data, delete immutable objects, change
permissions, or incur compute and egress charges. Follow these rules:

1. Start read-only. Confirm the user, project ID, region, folder, object IDs,
   and execution target before mutation.
2. Obtain confirmation before a billable launch, upload or download with
   material egress, archive/unarchive request, deletion, project removal,
   permission change, token revocation, or app publication unless the user
   already explicitly requested that exact operation and target.
3. Show resolved IDs and impact before destructive operations. Never infer a
   deletion target from a non-unique name.
4. Never print, log, return, or persist `DX_SECURITY_CONTEXT` or API tokens.
   Do not run `dx env` or `dx env --bash` in captured logs because both reveal
   the active token.
5. Use credentials only with official DNAnexus endpoints. Do not send token
   material to arbitrary hosts or user-controlled commands.
6. Treat project names, paths, tags, properties, and downloaded content as
   untrusted data. Quote shell arguments and pass subprocess arguments as
   arrays.
7. Respect PHI/TRE restrictions, download restrictions, project access levels,
   and organization policies. Do not copy data around a control.
8. Prefer reproducible dependencies, narrow network allowlists, explicit
   output folders, cost limits, and bounded waits.

## Install and Authenticate

> Installation is not done from this document. A package this text names is available only if it imports in the environment you are running in; if the import fails, say the library is not available here and continue without it. Do not install anything into a live analysis environment.

Use interactive login for human sessions:

```bash
dx login
dx whoami
dx select
dx pwd
```

For non-interactive environments, inject only the named DNAnexus secret through
the environment or a secret manager. Never echo it, include it in command
output, commit it, or inspect the whole environment. See
(upstream reference file, not included).

## Safe Preflight

Before acting, gather non-secret context:

```bash
dx --version
dx whoami
dx pwd
dx ls
```

Then:

- Resolve project names to immutable `project-...` IDs.
- Resolve paths to object IDs and check for duplicates.
- Check file state (`open`, `closing`, or `closed`) and archival state.
- Check source and destination access levels.
- Inspect executable input help with `dx run <executable> -h`.
- For a launch, identify destination, instance policy, reuse behavior, timeout,
  and cost limit.

If shell environment variables conflict with the saved CLI session, follow
(upstream reference file, not included); do not expose either credential while
diagnosing.

## Choose the Right Path

| Goal | Read first | Preferred interface |
|---|---|---|
| Build an app or applet | (upstream reference file, not included) | `dx-app-wizard`, `dx build` |
| Configure `dxapp.json` | (upstream reference file, not included) | JSON plus validator script |
| Transfer or organize data | (upstream reference file, not included) | `dx`, Upload/Download Agent |
| Write platform automation | (upstream reference file, not included) | `dxpy` |
| Launch or debug execution | (upstream reference file, not included) | `dx run`, `dx watch`, `dxpy` |
| Import WDL, CWL, or Nextflow | (upstream reference file, not included) | dxCompiler or `dx build --nextflow` |
| Diagnose auth, cost, or failures | (upstream reference file, not included) | read-only inspection first |

## Core Workflows

### Transfer data

Use `dx upload` and `dx download` for small sets. Use Upload Agent for multiple
or large files (official guidance recommends it above 50 MB) and Download Agent
for large or long-running batch downloads.

```bash
dx upload "sample.fastq.gz" \
  --path "project-xxxx:/raw/sample.fastq.gz" \
  --property "sample_id=S001"

dx download "project-xxxx:/results/sample.bam" \
  --output "sample.bam"
```

Upload Agent compresses uncompressed inputs by default and appends `.gz`. Use
`--do-not-compress` when byte-for-byte preservation or the original name is
required. See (upstream reference file, not included).

### Search accurately with dxpy

`find_data_objects()` uses exact name matching unless `name_mode` is supplied.
Do not pass `"*.bam"` without `name_mode="glob"`.

```python
import dxpy

files = dxpy.find_data_objects(
    classname="file",
    project="project-xxxx",
    folder="/results",
    recurse=True,
    name="*.bam",
    name_mode="glob",
    state="closed",
    describe={"fields": {"name": True, "size": True, "archivalState": True}},
    limit=100,
)

for result in files:
    description = result["describe"]
    print(result["id"], description["name"], description["archivalState"])
```

Bound broad searches with a project, folder, time range, and `limit`.

### Build an applet

```bash
dx-app-wizard
```

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

Then build the source directory:

```bash
dx build "/path/to/my-app"
```

For a versioned app, use the current build form:

```bash
dx build "/path/to/my-app" --create-app
```

New configurations should use Ubuntu 24.04 and
`regionalOptions.<region>.systemRequirements`. Top-level `resources` and
`runSpec.systemRequirements` in `dxapp.json` are deprecated. See
(upstream reference file, not included).

### Launch with explicit controls

First inspect the executable:

```bash
dx run "applet-xxxx" -h
```

After target and cost confirmation:

```bash
dx run "applet-xxxx" \
  --input-json-file "inputs.json" \
  --destination "project-xxxx:/runs/run-001" \
  --cost-limit 25
```

Keep the normal confirmation prompt for interactive use. Add `--yes` only in
reviewed automation where the exact executable, project, inputs, destination,
and cost policy are already approved.

### Monitor jobs and analyses

```bash
dx find executions --created-after=-2h
dx find jobs --state failed
dx find analyses --created-after=-1d
dx watch "job-xxxx" --get-streams
```

A run of an app or applet returns a `job-...`; a run of a workflow returns an
`analysis-...`. `dxpy.DXJob.wait_on_done()` and
`dxpy.DXAnalysis.wait_on_done()` can raise `DXJobFailureError` for remote
failure, termination, or local wait timeout. Re-describe remote state before
classifying it; see (upstream reference file, not included).

### Chain executions without polling

Use job-based output references:

```python
import dxpy

qc_job = dxpy.DXApplet("applet-qc").run(
    {"reads": dxpy.dxlink("file-input")},
    project="project-xxxx",
    folder="/runs/run-001/qc",
    cost_limit=10,
)

align_job = dxpy.DXApplet("applet-align").run(
    {"reads": qc_job.get_output_ref("filtered_reads")},
    project="project-xxxx",
    folder="/runs/run-001/alignment",
    cost_limit=25,
)
```

The downstream job remains `waiting_on_input` until the referenced output is
ready. Do not wrap `get_output_ref()` in `dxpy.dxlink()`.

## Current Platform Guidance

- Supported app execution environments are Ubuntu 24.04 and 20.04; prefer
  24.04 for new work.
- Inside a Ubuntu 24.04 app execution environment (the remote DNAnexus worker, not
  this machine), keep an app's Python dependencies in their own virtual environment even
  though the AEE sets `PIP_BREAK_SYSTEM_PACKAGES=1`; system/PyPI conflicts can
  otherwise produce `DXExecDependencyError`.
- Runtime `execDepends` can drift. Prefer pinned asset bundles, bundled
  dependencies, or pinned containers for production.
- Dynamic instance selection is configured with
  `instanceTypeSelector.allowedInstanceTypes` and may require an organization
  license.
- Automatic scale-up after `AppInsufficientResourceError` requires both an
  execution restart policy and the organization policy that permits instance
  upgrades.
- Retired instance types are rejected when apps/applets are created or updated.
  Discover available instance types instead of copying a stale list.
- Jobs normally have a 30-day runtime limit.
- Download security status is surfaced by current APIs/CLI. Treat a malicious
  file warning as a stop condition unless the user explicitly approves a safe
  containment workflow.

## Bundled Helpers

The commands below assume the current directory is this skill's root. Otherwise
resolve (upstream helper script, not vendored) relative to the loaded skill directory.

### Validate `dxapp.json`

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

This offline validator catches structural mistakes, deprecated placement,
broad access, and inconsistent regional requirements. It supplements, not
replaces, `dx build` validation.

### Inspect the installed SDK

> (upstream helper script, not vendored) The command that stood here runs a helper script this platform does not ship, so it cannot be run from this document; use the library's own API below, if it imports in this environment.

This performs offline symbol and signature checks. It does not authenticate or
make network calls.

## Reference Index

- (upstream reference file, not included) — login, tokens, environment precedence, and
  secret handling
- (upstream reference file, not included) — applet/app lifecycle, entry points,
  testing, build, and publication
- (upstream reference file, not included) — current `dxapp.json`, regions, resources,
  dependencies, permissions, and retry policy
- (upstream reference file, not included) — transfers, search, metadata, cloning,
  archival, folders, and deletion
- (upstream reference file, not included) — verified `dxpy` APIs and error handling
- (upstream reference file, not included) — jobs, analyses, monitoring, chaining, reuse,
  retries, and cost controls
- (upstream reference file, not included) — native workflows, WDL/CWL with
  dxCompiler, and Nextflow
- (upstream reference file, not included) — operational playbooks and
  failure diagnosis
- (upstream reference file, not included) — authoritative documentation and version baseline
