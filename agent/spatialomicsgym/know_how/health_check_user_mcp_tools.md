# Health Check User MCP Tools

## Metadata
- **Category**: tool_management
- **Triggers**: check tools, health check, are my tools working, tool status, verify tools, diagnose tools, tool health, are tools ok, test tools, validate tools, tools broken, tool diagnostics, deep check, thorough check, test with real data, full validation
- **Requires**: (none — read-only operation, works regardless of tool_creation_enabled)
- **Version**: 1.0
- **Last Updated**: 2026-04-15

## Overview

Run a comprehensive health check on all user-created MCP tools. This is a **read-only** operation — it never modifies files, configs, or state. It validates tool integrity, tests functionality, displays tool details, and provides actionable recommendations.

The health check includes **spatialomicsgym_name validation** — verifying that the `spatialomicsgym_name` field in `mcp_config_user.yaml` matches the actual `@mcp.tool()` function name in the MCP server file. A mismatch causes tool calls to fail silently at runtime with "Unknown tool".

**Key principle:** Import, run, format, present — all in one go. No human input needed during execution.

## Health-Check-Specific HARD RULES

Health check is **read-only**, **idempotent**, and **observational**.
Its job is to REPORT state, not to FIX state. Observation quality beats
verdict speed.

1. **Zero write operations.** Ever. No install, no uninstall, no env
   repair, no file edit, no config change. If the check finds something
   fixable, RECOMMEND the fix (point user at delete/modify/recreate
   flow); don't auto-execute.

2. **Report structure: PER-TOOL health with DEGRADED/BROKEN split**:
   - `HEALTHY` — all checks pass; tool usable in production
   - `DEGRADED` — callable but non-critical issue (e.g. missing
     backups, missing knowledge, parameter drift, soft warnings)
   - `BROKEN` — critical failure (import fails, missing files,
     config/schema error, unresolvable spatialomicsgym_name — tool
     will crash or resolve to nothing if called)
   - `CHECK_FAILED` — health-check itself errored (tool state
     unknown — treat as BROKEN for conservatism)

   Every finding must be tagged with which level it triggers.

3. **Light vs deep**:
   - `include_real_data=False` (default) = fast checks only (files,
     conda_env, infrastructure, import, syntax, imports_scan, dry_run,
     config, param_consistency, spatialomicsgym_name, knowledge,
     backups). Those are the exact `name` values in each tool's
     `checks` list — look for those, not for a friendlier synonym.
     Should finish in <5s per tool.
   - `include_real_data=True` = runs the worker on a minimal fixture.
     Can take minutes per tool. Only on explicit user request.

4. **Never end a response with "would you like"/"Option A/B/C".**
   Health-check produces a REPORT — the "what to do next" is a
   single-sentence recommendation, never a menu.

5. **Never `import` a user MCP server file.** Inspect via:
   - `ast.parse` for the server/worker source
   - `json.loads` for install_log
   - `yaml.safe_load` for mcp_config_user.yaml
   - `subprocess.run` for `conda env list` / `Rscript -e 'library()'`
   Never via `importlib.import_module("mcp_servers.*")`.

6. **Language is determined by worker file extension** (P12 fix).
   Look for `{tool_id}_worker.{py,R,sh}` on disk as the authoritative
   signal. If `install_log.language="R"` but only `{tool_id}_worker.py`
   exists, report this mislabel as DEGRADED + recommend modify to
   fix install_log.language.

7. **The spatialomicsgym_name check compares the config to the server
   file, not to a naming convention** (P22). It reads the
   `spatialomicsgym_name` in `mcp_config_user.yaml` and the name of the
   `@mcp.tool()` function in `{tool_id}_mcp_server.py`. Equal is PASS —
   including when that name is not the standardized `{tool_id}_run`. A
   consistently non-standard name is not a finding; don't report one.
   Different is `critical: True` + FAIL and the tool's health is
   **BROKEN**: the configured name resolves to no function, so a call
   comes back "Unknown tool". Report BROKEN, don't call the tool, and
   pass through the recommendation the check already produced — it
   names the one-line repair (set `spatialomicsgym_name` to the
   `@mcp.tool()` name it found). Recreating the tool is not the remedy.

## Health-Check Invariants (READ FIRST)

Health check is the system's audit function — it must be *read-only* and *idempotent*.
Running it ten times in a row on a quiescent system must produce identical reports. If
the report ever says "HEALTHY" but `real_data` test would fail, the cost landed on the
user — treat that as a bug in the check itself, not a tolerable approximation.

| Layer | What it verifies | Severity if FAIL |
|-------|------------------|-------------------|
| Shared symlinks | `base_mcp.py` / `worker_utils.py` in `tools_user/` still resolve | CRITICAL → BROKEN. Emitted as check `infrastructure` on **every** active tool, because no tool can run without them — a single dangling symlink turns the whole registry BROKEN, and that is the correct reading |
| Registry hygiene | `install_log.json` parses, no duplicate or field-incomplete entries, no orphaned `*_worker.py` without a log entry, no stale `user_*` in `mcp_config_user.yaml` | Global only — lands in `report["infrastructure"]`, never enters a tool's `checks`, never moves a tool's health. Report it alongside the per-tool table |
| Files | worker + server + env.yaml all present | CRITICAL → BROKEN |
| Syntax | `ast.parse` of worker + server succeeds | CRITICAL → BROKEN |
| Conda env | `user_{tool_id}` appears in `conda env list --json` | CRITICAL → BROKEN |
| Import | `import {module}` succeeds inside the env OR vendor_path is set AND present | CRITICAL → BROKEN |
| Imports scan | every `import X` in worker/server resolves via `importlib.util.find_spec` | CRITICAL → BROKEN |
| Dry-run | worker `--input /nonexistent --output-dir /tmp/x` returns WorkerOutput JSON with status=="error" | CRITICAL → BROKEN |
| Config consistency | `mcp_config_user.yaml[user_{tool_id}]` matches `install_log[tool_id]` (enabled + tools array non-empty + spatialomicsgym_name present in server file) | CRITICAL → BROKEN |
| spatialomicsgym_name | `mcp_config_user.yaml` `spatialomicsgym_name` == the `@mcp.tool()` function name in the server file (HARD RULE 7) | CRITICAL → BROKEN |
| Drift | `install_log.module` still importable — that half is the Import row above. Whether it still matches the `from X import Y` in the worker is **not implemented**; there is no `drift` check, so don't report on it | n/a |
| Parameter consistency | worker CLI params ⟷ MCP server parameters ⟷ `.knowledge/{tid}/function_signatures.json` | NON-CRITICAL → DEGRADED |
| Knowledge completeness | `.knowledge/{tid}/` exists AND contains `install_plan.json` + `api_discovery.json` + `final_audit.json` + at least one backup | NON-CRITICAL → DEGRADED |
| Real-data | worker runs on benchmark h5ad and produces ≥ 1 non-empty output file | NON-CRITICAL → DEGRADED (only run in deep mode) |

**Health status decision rule:**
- `BROKEN` ⇔ at least one CRITICAL check FAILs.
- `DEGRADED` ⇔ all CRITICAL pass AND at least one NON-CRITICAL FAILs.
- `HEALTHY` ⇔ all CRITICAL + NON-CRITICAL pass.
- `TRASHED` ⇔ `install_log[tid].status == 'trashed'` (the tool isn't tested, just listed).
- `CHECK_FAILED` ⇔ the audit itself raised an exception — NOT the same as BROKEN.

**Read-only guarantee:**
The health check must NOT create, mutate, or delete any file, conda env, config entry, or install_log entry. It runs the worker's dry-run (which writes to `/tmp/`), nothing else.

**Drift detection (the silent-failure catcher):**
The most dangerous health-check false-positive is "import X succeeds" but `X` is a different package than the worker actually uses (e.g. BANKSY's install_log said `module=pybanksy` but the worker import-ed `scanpy`). The drift check catches this by parsing the worker's import statements and verifying each top-level module against `install_log.module`. Any mismatch is DEGRADED + a recommendation to update install_log or rewrite the worker — never silently healthy.

**When to run automatically (even without user prompt):**
1. Immediately after `add_new_mcp_tool` Phase 6 — the final-invariant-check is a subset of this.
2. Immediately after `modify_user_mcp_tool` Step 6 — catch regressions before the user calls the tool.
3. Immediately after `restore_tool` from trash — catch partial restores.
4. Once per user session on startup, as a silent background check (log only, don't surface unless something broke).

## Pre-flight Check

```python
from spatialomicsgym.setup.constants import ensure_repo_importable
ensure_repo_importable()   # `tools_user` is not an installed package; this puts the repo on sys.path

from tools_user.knowledge_manager import health_check_tools
print("Health check function loaded.")
```

## Default Health Check (fast, autonomous)

Run this by default when a user asks about tool status. Skips the expensive real_data test.

```python
report = health_check_tools()

# Print infrastructure status
print(f"\n## Infrastructure: {report['registry_status']}")
infra = report["infrastructure"]
print(f"  install_log: {'valid' if infra['install_log_valid'] else 'INVALID'}")
for w in infra["install_log_warnings"]:
    print(f"    WARNING: {w}")
print(f"  base_mcp.py: {infra['base_mcp_symlink']['detail']}")
print(f"  worker_utils.py: {infra['worker_utils_symlink']['detail']}")
if report["orphaned_files"]:
    print(f"  Orphaned files: {report['orphaned_files']}")
if report["stale_mcp_entries"]:
    print(f"  Stale MCP entries: {report['stale_mcp_entries']}")
for u in report["unwired_tools"]:
    # An active tool no `mcp_servers` entry points at: installed, files on disk, NOT callable.
    print(f"  UNWIRED: {u['tool_id']} -- {u['reason']}")

# Print per-tool results
for tool in report["tools"]:
    info = tool.get("info", {})
    print(f"\n## Tool: {tool['tool_id']} ({tool['health']})")
    print(f"  Package: {info.get('package', '?')} | Task: {info.get('task_type', '?')} | Language: {info.get('language', '?')}")
    if info.get("source_url"):
        print(f"  Source: {info['source_url']}")
    print(f"  Created: {info.get('created_at', '?')} | Modifications: {info.get('modification_count', 0)}")
    if info.get("pipeline_steps"):
        print(f"  Pipeline: {' → '.join(str(s) for s in info['pipeline_steps'][:10])}")
    if info.get("worker_parameters"):
        params = info["worker_parameters"]
        param_strs = []
        for pname, pinfo in list(params.items())[:8]:
            clean_name = pname.lstrip("-").replace("-", "_")
            ptype = pinfo.get("type", "?")
            default = pinfo.get("default")
            if default is not None:
                param_strs.append(f"{clean_name} ({ptype}, {default})")
            elif pinfo.get("required"):
                param_strs.append(f"{clean_name} ({ptype}, required)")
            else:
                param_strs.append(f"{clean_name} ({ptype})")
        print(f"  Parameters: {', '.join(param_strs)}")
    if info.get("mcp_signature"):
        sig_params = list(info["mcp_signature"].keys())[:8]
        print(f"  MCP signature: ({', '.join(sig_params)})")
    
    # Checks summary
    checks = tool.get("checks", [])
    passed = sum(1 for c in checks if c["result"] == "PASS")
    total = len(checks)
    skipped = sum(1 for c in checks if c["result"] == "SKIP")
    failed = [c for c in checks if c["result"] in ("FAIL", "WARN")]
    skip_note = f" ({skipped} skipped)" if skipped else ""
    print(f"  Checks: {passed}/{total} passed{skip_note}")
    for c in failed:
        icon = "FAIL" if c["result"] == "FAIL" else "WARN"
        print(f"    [{icon}] {c['name']}: {c['detail'][:80]}")
    
    # Recommendations
    if tool.get("recommendations"):
        print("  Recommendations:")
        for rec in tool["recommendations"]:
            print(f"    - {rec}")

# Summary
s = report["summary"]
print(f"\n## Summary: {s['total']} tool(s) checked — {s['healthy']} healthy, {s['degraded']} degraded, {s['broken']} broken")
if s.get("trashed"):
    print(f"  ({s['trashed']} trashed)")
du = report["disk_usage"]
print(f"  Disk: active conda {du['active_conda_size']}, reclaimable {du['total_reclaimable']}")
print(f"  Duration: {report['duration_seconds']}s")
```

## When to Run Health Checks

Run `health_check_tools()` automatically in these situations — don't wait for the user to ask:
1. **After any modification** — verify the tool is still HEALTHY after changes
2. **After rollback** — confirm the restored version works correctly
3. **After tool creation** — validate the new tool passes all checks
4. **After restore from trash** — ensure files were restored correctly
5. **When user asks about tool status** — the primary use case

```python
# Quick health check after an operation
report = health_check_tools(tool_ids=[tool_id])
tool = report["tools"][0]
if tool["health"] == "BROKEN":
    print(f"WARNING: {tool_id} is BROKEN after operation!")
    failed = [c for c in tool["checks"] if c["result"] == "FAIL"]
    for c in failed:
        print(f"  FAIL: {c['name']}: {c.get('detail', '')[:80]}")
    # Consider rollback if this was a modification
elif tool["health"] == "DEGRADED":
    print(f"NOTE: {tool_id} is DEGRADED — non-critical issues found")
else:
    print(f"{tool_id} is HEALTHY")
```

## Deep Health Check (with real data, user-triggered)

Only run this when the user explicitly asks for a thorough or deep check. It runs the expensive real_data test which executes each tool's worker with actual benchmark data.

**Trigger phrases:** "deep check", "thorough check", "test with real data", "full validation"

```python
report = health_check_tools(include_real_data=True)
# ... same formatting as above ...
```

The worker can only run where benchmark data exists. On a host without it (a pip install prunes
it), `real_data` comes back `result: "SKIP"` with an `INCONCLUSIVE` detail and a `Note:` line in
`recommendations`. That is not a failure and the tool stays HEALTHY — but nothing was executed on
data, so a deep check that skipped every `real_data` test verified exactly what the fast check
verifies. Report that in the summary instead of letting HEALTHY stand for more than it covers.

## Single Tool Check

When the user asks about a specific tool:

```python
report = health_check_tools(tool_ids=["sopa"])
# ... same formatting as above ...
```

## Interpreting Results

### Health Status Levels

| Status | Meaning | Action |
|--------|---------|--------|
| **HEALTHY** | All checks pass | No action needed |
| **DEGRADED** | Critical checks pass, some non-critical warnings | Review recommendations |
| **BROKEN** | One or more critical checks fail | Fix immediately |
| **TRASHED** | Tool is in trash (not tested) | Restore or permanently delete |
| **CHECK_FAILED** | Health check itself errored | Investigate manually |

### Critical vs Non-Critical Checks

These are the literal `name` values in each tool's `checks` list. Don't infer the split — every
entry carries its own `critical` flag, and a tool is BROKEN iff some check has `result == "FAIL"`
and `critical` true.

**Critical (failure = BROKEN):** files, conda_env, infrastructure, import, syntax, imports_scan,
dry_run, config, spatialomicsgym_name
**Non-critical (failure = DEGRADED):** param_consistency, knowledge, backups, real_data

`infrastructure` appears only when the shared symlinks are broken, and then on every active tool.
`real_data` appears only under `include_real_data=True`.

## Acting on Recommendations

Each finding has one remedy. Name it in the report as the recommendation for that finding —
HARD RULE 4: a sentence, not a menu, and the health check does not execute it. If the user then
says to go ahead, that is a new turn and the corresponding flow takes over.

| Finding | Remedy to name |
|---------|----------------|
| Missing knowledge | `bootstrap_knowledge(tool_id)` |
| Stale knowledge | `bootstrap_knowledge(tool_id)` refreshes it |
| Missing backup | `create_backup(tool_id)` |
| Parameter drift | reconcile the params through the `modify_user_mcp_tool` flow |
| spatialomicsgym_name mismatch | set `spatialomicsgym_name` in `mcp_config_user.yaml` to the `@mcp.tool()` name the check reported; the catalog resyncs on the next turn |
| Broken shared symlink | re-link `tools_user/base_mcp.py` and `worker_utils.py` to `tools/`; until then no user tool runs |
| Syntax error | the `modify_user_mcp_tool` flow can attempt a repair |
| Orphaned files | register them, or move them out through the delete flow |
| Stale MCP config | remove the stale `user_*` entries |
| Unwired tool | installed but not callable — move the block under `mcp_servers:` (top-level case) or re-add its entry; the catalog resyncs on the next turn |

**Important:** the health check states these; it never runs them. It is read-only (HARD RULE 1),
and the recommendation belongs in the report, not in a closing question (HARD RULE 4).

## When to Run

- User asks: "check my tools", "are my tools working", "tool status"
- After system maintenance or environment changes
- When a tool unexpectedly fails during use — say a health check would show why, and run it if
  the user asks; don't close the turn on the question
- Before running a critical analysis with user tools
