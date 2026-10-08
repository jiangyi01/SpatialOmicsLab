# Modify User MCP Tools

## Metadata
- **Category**: tool_management
- **Triggers**: modify tool, change tool, update tool, edit worker, change parameter, add step, switch function, modify pipeline, change output, add feature, swap method, adjust parameter, add error handling, add visualization, change resolution, change default, add parameter, remove step, edit pipeline, update pipeline, rollback tool, undo modification, show modification history, show backups
- **Requires**: `tool_creation_enabled = True`
- **Version**: 1.0
- **Last Updated**: 2026-04-15

## Overview

This document guides STCoscientist in modifying existing user-created MCP tools through a **safe modification system** with automatic backup, testing, and rollback. All modifications follow a consistent pattern: load knowledge, backup, modify, test, commit-or-rollback.

**Key principle:** NEVER modify tool files without a backup. ALWAYS test after modification. ALWAYS rollback on failure.

### Run this FIRST, before any other code block

Your working directory is wherever the user launched from — **not** the repo. Use these names for
the registry files below; a bare `Path("tools_user/...")` would read an empty file in the user's
data directory, and `rollback_to_backup` restores the config entry through these same constants.

```python
from spatialomicsgym.setup.constants import ensure_repo_importable
ensure_repo_importable()   # `tools_user` is not an installed package; this puts the repo on sys.path
from tools_user.knowledge_manager import INSTALL_LOG, KNOWLEDGE_DIR, mcp_config_user_path, TOOLS_USER_DIR
```

## Modification-Specific HARD RULES

Modification is **conservative** and **test-gated**. Different
modification types carry different risk and should be handled differently:

1. **Backup is ATOMIC and ALWAYS-FIRST** — `begin_modification(tool_id)`
   must complete before ANY edit (it snapshots the current, pre-edit files).
   If backup fails for any reason, abort — do not proceed with an
   un-backed-up modification.

2. **Modification-type taxonomy** (classify before editing):
   | Type | Risk | Verification |
   |------|------|--------------|
   | `parameter_change` (default value, type) | LOW | ast.parse + 1 dry-run |
   | `parameter_add` (new optional arg) | LOW | ast.parse + dry-run + schema check |
   | `import_swap` (scanpy → squidpy style) | MEDIUM | env-closure fixture loop (P15) |
   | `algorithm_change` (method swap) | HIGH | env-closure + real-data smoke |
   | `cross_language` (Python worker → R worker) | CRITICAL | full re-creation instead |

   A MEDIUM/HIGH modification MUST run env-closure. Never stop at
   ast.parse for anything beyond LOW.

3. **Preserve the standardized spatialomicsgym_name `{tool_id}_run`** (P22).
   The `@mcp.tool()` decorated function name is frozen at creation.
   Modifications cannot rename it. If a rename is requested: trash
   the old tool_id, create a new tool. Mixing "rename + behavior
   change" in a single modify is banned — too many degrees of freedom.

4. **Commit-or-rollback is ATOMIC**. If tests fail after modification:
   a. `rollback_to_backup(tool_id)` — bring files back to pre-modification state
   b. Verify restoration via diff — the restored files must match
      the backup exactly.
   c. Log the failure to `modify_issues.json`.
   d. Print root cause + what was attempted, never "Option A/B/C".

5. **Never end a response with "would you like"/"Option A/B/C"**
   — same rule as creation's P7. When a modification fails, the
   response is: what broke, root cause, rollback result, and
   what-to-do-next as a single recommendation (not a menu).

6. **Never `import` a user MCP server file** — same P13 rule.
   Edit via filesystem + `ast` (read) + `write_atomic` (write).

7. **Self-capture issues** (P-capture rule). On any non-trivial
   issue resolved during modification, append to
   `KNOWLEDGE_DIR / tool_id / "modify_issues.json"` with shape
   `{symptom, root_cause, fix, modification_type, timestamp}`.
   Use the imported `KNOWLEDGE_DIR`, never a bare
   `tools_user/.knowledge/` — that resolves against the user's
   launch directory, and the knowledge the next modification reads
   is the repo-anchored copy.

## Quick Reference

| User Intent | STCoscientist Action | Mod Type |
|------------|-----------|----------|
| "change resolution to 1.0" | Edit argparse default + MCP signature | `parameter_change` |
| "add UMAP visualization" | Add code block to worker | `pipeline_addition` |
| "remove the scaling step" | Remove code block from worker | `pipeline_removal` |
| "switch from leiden to louvain" | Rewrite core function call | `function_swap` |
| "also save a CSV" | Add output code to worker | `output_change` |
| "add check for min 100 cells" | Add validation to worker | `error_handling` |
| "add min_cells parameter" | Edit worker argparse + MCP signature | `signature_change` |
| "add scikit-learn to env" | Install package in conda env | `environment_change` |
| "rollback sopa" / "undo changes" | `rollback_to_backup("sopa")` | (manual rollback) |
| "show modification history" | `list_backups("sopa")` | (query) |
| "show tool knowledge" | `read_knowledge("sopa")` | (query) |

## Modification Lifecycle Invariants (READ FIRST)

A modification is a state machine with **four checkpoints** per attempt. Each one has
a hard precondition on entry and a hard post-condition that must hold before the next
step runs. Violating the ordering is the #1 source of broken rollbacks.

| Step | Pre-condition | Post-condition | On-failure action |
|------|---------------|----------------|-------------------|
| 1. begin_modification | tool is ACTIVE, health is not BROKEN, latest knowledge readable | backup `v{N}` persisted to `.knowledge/{tid}/backups/`; current files untouched | `rollback_to_backup(tid, N)` would be a no-op — nothing to undo |
| 2. edit files | backup `v{N}` exists; no active modification lock | new worker / server files written atomically; files parse via `ast.parse` | revert via `rollback_to_backup(tid, N)` — `v{N}` is the pre-edit snapshot Step 1 took |
| 3. complete_modification (tests) | edits done, `ast.parse` clean | every test that ran PASSed (import, syntax, imports_scan, dry_run, config, and `real_data` when `real_data_validated` is true) | `rollback_to_backup(tid, N)`; modification is NOT committed |
| 4. finalize_modification | Step 3 passed | `install_log[tid].modification_count += 1`, knowledge + backup retained, health still HEALTHY/DEGRADED | caller resumes — no cleanup |

**Version invariant:** `N` is the number `begin_modification()` returns as `backup_version`, and it is
always the version to roll back to. `v{N}` holds the files as they were *before* your edits, so
`rollback_to_backup(tid, N)` undoes exactly this attempt. Never subtract from it: `N-1` is the
*previous* modification's snapshot, so on a first modification it is `v000`, which
`rollback_to_backup` refuses with `RollbackError: Backup version 0 not found`, and on any later one
it silently reverts a change the user already shipped.

**Ordering invariant (the #1 broken-rollback cause):**
`begin_modification()` **must** happen BEFORE any file is edited. If you edit files first and *then* call `begin_modification()`, the backup already contains your edits and `rollback_to_backup()` becomes useless. Order:

```
begin_modification()    # snapshots CURRENT (pre-edit) files
read files, plan       # Step 2 of the 7-step flow
write modified files   # Step 3
complete_modification() # Step 4 — runs tests on the NEW files
```

**Atomicity rule:**
All file writes in Step 3 must go through an atomic writer (tmp + `os.replace`). Never edit with `f.write()` directly — a crash mid-write leaves a half-edited file that `ast.parse` in Step 4 will flag but the old backup from Step 1 is needed to recover. Reuse the helpers from `add_new_mcp_tool.md` Phase 4 (`write_atomic`, `yaml_atomic`, `json_atomic`).

**Idempotence rule:**
Calling `begin_modification(tid)` while a modification is already open raises `ModificationError` — never create two overlapping backups. The knowledge manager enforces this via a lock file `.knowledge/{tid}/.modification.lock`. Only two calls clear it: `finalize_modification()` and `rollback_to_backup()`. `complete_modification()` does **not** — it runs the tests and leaves the modification open, so a second change to the same tool in one session must finalize (or roll back) the first one before it begins.

**Stale lock (a session that died between begin and finalize):**
The lock is a file, so it survives the process. `begin_modification()` runs `_repair_modification_state()` first, but that only completes interrupted rollbacks and removes half-written backups — it does not clear the lock. Every later `begin_modification(tid)` then raises "already in progress" with no modification running. That refusal quotes what the lock records — when the open modification began, how long ago that was, and the request it was serving — so read it before acting: an age of hours or days is a dead session, while one of seconds means a modification may still be running in another process, and the remedy below would undo its work mid-flight. The way out is `rollback_to_backup(tid, version=<the open backup>)`: it undoes whatever the interrupted session had written and clears the lock. Do that before telling the user the tool cannot be modified. If the release itself fails — the lock was written by another user, or its directory is read-only — `finalize_modification()` and `rollback_to_backup()` still report `success: True` (their own work did happen) and add a `warning` naming the lock file. Pass that warning on: deleting that one file is then the only way to modify the tool again. `finalize_modification()` uses the same `warning` key for one other thing, so read what it says rather than assuming the lock: if `install_log.json` could not be updated — it did not parse, or the write failed — the modification is still recorded in `modification_log.json`, but the tool's `modification_count` and `last_modified_at` stay at their old values, so the health check will keep reporting the previous number of modifications until that file is repaired.

**Interrupted rollback (a crash between staging and committing):**
`rollback_to_backup()` stages each file as `<name>.rollback_tmp` and then commits it, so a crash — or a file it cannot replace — leaves the tool part one version and part another. The next `begin_modification(tid)` finishes that rollback for you. If it *cannot*, it refuses with a `ModificationError` naming the staged files, rather than snapshotting the mixture as `v{N}` and handing you a rollback target that restores a state the tool was never in. That refusal is **not** the "already in progress" one above, and `rollback_to_backup()` does not clear it — it would stage the same files and stop at the same place. The staged copies are kept on disk on purpose: clear whatever blocks replacing the originals (permissions, another process holding the file, a full disk) and begin again.

**Post-test cross-consistency (run this after Step 4 passes):**
After `complete_modification()` reports all 6 tests green, verify that the modification didn't silently divide the world:
- Worker AND server files still both parse via `ast.parse`.
- Every import in the edited files resolves inside `user_{tid}`.
- `install_log[tid].module` still imports (the module field did not drift).
- MCP server entry in `mcp_config_user.yaml` still matches the spatialomicsgym_name in the server file.

Run the snippet in "Post-commit cross-consistency audit" (below) before returning control to the caller.

## Pre-flight Check

Execute this before ANY modification operation:

```python
# 1. Verify feature is enabled
from spatialomicsgym.config import default_config
assert default_config.tool_creation_enabled, "tool_creation_enabled is False! Enable it first."

# 2. Import knowledge manager
from tools_user.knowledge_manager import (
    read_knowledge, begin_modification, complete_modification,
    finalize_modification, rollback_to_backup, update_knowledge,
    bootstrap_knowledge, knowledge_exists, list_backups, knowledge_info,
    KnowledgeError, KnowledgeSafetyError, KnowledgeStateError,
    KnowledgeNotFoundError, ModificationError, RollbackError,
)
print("Knowledge manager loaded successfully.")

# 3. Import trash manager for tool lookup
from tools_user.trash_manager import list_user_tools, _find_log_entry

# 4. Parse-sanity + pre-health-check — refuse to modify a BROKEN tool
import json
from pathlib import Path
log = json.loads(INSTALL_LOG.read_text()) \
      if INSTALL_LOG.exists() else []
entry = next((e for e in log if e.get("tool_id") == tool_id), None)
if entry is None:
    raise KnowledgeNotFoundError(f"{tool_id} is not registered")
if entry.get("status") != "active":
    raise KnowledgeStateError(
        f"{tool_id} has status {entry.get('status')!r}; cannot modify non-active tools. "
        "Restore first if trashed, or recreate if permanently deleted."
    )

# Quick health gate — BROKEN tools should be rollback/rebuild candidates, not modify
from tools_user.knowledge_manager import health_check_tools
pre_health = health_check_tools(tool_ids=[tool_id], include_real_data=False)
tool_report = pre_health["tools"][0]
if tool_report["health"] == "BROKEN":
    raise KnowledgeStateError(
        f"{tool_id} is BROKEN before modification. Fix via rollback to a known-good "
        f"backup or rebuild via rollback()+creation — do not modify a broken tool."
    )
print(f"Pre-flight OK: {tool_id} is {tool_report['health']}, ready to modify.")
```

## Post-commit cross-consistency audit (run after Step 6 Commit succeeds)

```python
import ast, json, subprocess, yaml
from pathlib import Path

def post_modification_audit(tool_id: str) -> dict:
    out = {"tool_id": tool_id, "ok": True, "failures": [], "checks": {}}
    def fail(k, d): out["ok"] = False; out["checks"][k] = False; out["failures"].append(f"{k}: {d}")
    def ok(k): out["checks"][k] = True

    worker = TOOLS_USER_DIR / f"{tool_id}_worker.py"
    server = TOOLS_USER_DIR / f"{tool_id}_mcp_server.py"
    for p in (worker, server):
        if not p.is_file():
            fail(p.name, "missing after commit")
            continue
        try:
            ast.parse(p.read_text())
            ok(p.name)
        except SyntaxError as e:
            fail(p.name, f"SyntaxError: {e}")

    # Imports resolve in env
    log = json.loads(INSTALL_LOG.read_text())
    entry = next(e for e in log if e["tool_id"] == tool_id)
    env = entry["env_name"]
    mod = entry.get("module")
    if mod:
        r = subprocess.run(["conda", "run", "-n", env, "python", "-c",
                            f"import {mod}; print('OK')"],
                           capture_output=True, text=True, timeout=30)
        if "OK" in r.stdout:
            ok("module_importable")
        else:
            fail("module_importable", f"import {mod} failed post-modification")

    # spatialomicsgym_name still aligned between config and server file
    cfg = yaml.safe_load(mcp_config_user_path().read_text())
    srv = (cfg.get("mcp_servers") or {}).get(f"user_{tool_id}")
    if not srv:
        fail("mcp_cfg_entry", f"missing user_{tool_id}")
    else:
        spatialomicsgym_names = [t.get("spatialomicsgym_name") for t in (srv.get("tools") or [])]
        for bn in spatialomicsgym_names:
            if bn not in server.read_text():
                fail("spatialomicsgym_name_drift",
                     f"config lists spatialomicsgym_name={bn!r} but server file no longer defines it")
                break
        else:
            ok("spatialomicsgym_name_alignment")

    return out
```

Call `post_modification_audit(tool_id)` after `finalize_modification()`. If `ok` is False, roll back: `rollback_to_backup(tool_id, version=backup_version)` — the commit succeeded but the world is inconsistent, and a pre-edit backup is always preferable to a half-consistent post-edit state.

## Identifying the Target Tool

Extract the tool_id from the user's request. If unclear, list available tools:

```python
# If user mentions a tool name:
tool_id = "<extracted from user request>"
entry = _find_log_entry(tool_id)
if entry is None:
    # Show available tools
    tools = list_user_tools(status="active")
    print("Available tools:")
    for t in tools:
        print(f"  - {t['tool_id']} ({t.get('function_name', '')})")
    # Ask user which tool to modify
else:
    assert entry.get("status") == "active", f"Tool must be active, got '{entry.get('status')}'"
    print(f"Target tool: {tool_id} (function: {entry.get('function_name', '')})")
```

## Dynamic Tool Discovery (Before Any Modification)

When you need to modify a tool, ALWAYS gather this information first. Do NOT skip this step — modifying without understanding current state leads to broken edits and contaminated backups.

```python
# 1. Get tool metadata from install_log
from tools_user.trash_manager import _find_log_entry
entry = _find_log_entry(tool_id)
print(f"Package: {entry.get('package')}, Task: {entry.get('task_type')}")
print(f"Language: {entry.get('language')}, Env: {entry.get('env_name')}")

# 2. Read current knowledge (function signatures, pipeline steps)
knowledge = read_knowledge(tool_id)
if knowledge.get("function_signatures"):
    sigs = knowledge["function_signatures"]
    print("Current parameters:", list(sigs.get("worker_cli", {}).get("parameters", {}).keys()))
    print("Pipeline steps:", [s["call"] for s in sigs.get("pipeline_steps", [])])

# 3. Read CURRENT file contents (not cached — files may have been modified)
worker_code = (TOOLS_USER_DIR / f"{tool_id}_worker.py").read_text()
server_code = (TOOLS_USER_DIR / f"{tool_id}_mcp_server.py").read_text()

# 4. Check current health status
report = health_check_tools(tool_ids=[tool_id])
print(f"Current health: {report['tools'][0]['health']}")

# 5. Check existing backups
backups = list_backups(tool_id)
print(f"Existing backups: {len(backups)} versions")
```

**Why this matters:** Skipping discovery leads to:
- Editing wrong parameters (name mismatch between worker and server)
- Missing existing pipeline steps that depend on the code you're changing
- Creating backups of already-broken state
- Not knowing the correct module name for `complete_modification()`

## The Universal 7-Step Modification Flow

**Every modification follows these 7 steps regardless of type.** Do NOT skip any step.

### Step 1: Begin Modification (Pre-flight + Backup + Load Knowledge)

```python
# Classify the modification type from the quick reference table above
modification_type = "<one of: parameter_change, pipeline_addition, pipeline_removal, function_swap, output_change, error_handling, signature_change, environment_change>"
description = "<human-readable description of what the user asked>"

result = begin_modification(tool_id, user_request=description,
                            modification_type=modification_type)
knowledge = result["knowledge"]
backup_version = result["backup_version"]
print(f"Backup v{backup_version} created. Knowledge loaded.")
```

### Step 2: Review Current State

Read and display relevant knowledge to understand the current tool:

```python
from pathlib import Path

# Read current tool files
worker_path = TOOLS_USER_DIR / f"{tool_id}_worker.py"
server_path = TOOLS_USER_DIR / f"{tool_id}_mcp_server.py"
worker_code = worker_path.read_text()
server_code = server_path.read_text()

# Display current pipeline steps if available
if knowledge.get("function_signatures"):
    sigs = knowledge["function_signatures"]
    print("\nCurrent pipeline steps:")
    for step in sigs.get("pipeline_steps", []):
        print(f"  {step['step']}. {step['call']} -- {step['purpose']}")
    print("\nCurrent worker CLI parameters:")
    for param, info in sigs.get("worker_cli", {}).get("parameters", {}).items():
        print(f"  {param}: type={info.get('type')}, default={info.get('default')}")
    print("\nCurrent MCP function parameters:")
    for param, info in sigs.get("mcp_function", {}).get("parameters", {}).items():
        print(f"  {param}: type={info.get('type')}, default={info.get('default')}")
```

### Step 3: Make Modifications

**CRITICAL ORDERING RULE:** Step 1 (`begin_modification()`) MUST complete BEFORE you edit any files. The backup created in Step 1 captures the current file state. If you edit files first, the backup contains your changes and rollback becomes useless. The correct order is:
1. `begin_modification()` → creates backup of CURRENT state
2. Read files, plan changes (Step 2)
3. Write modified files (this step)
4. `complete_modification()` → tests the changes (Step 4)
Violating this order is the #1 cause of broken rollbacks.

**IMPORT SAFETY — Check BEFORE Writing Code:**

When your modification adds new `import` statements (Python) or `library()` calls (R), verify the packages are installed in the tool's conda env BEFORE writing the code:

```python
import subprocess
env_name = f"user_{tool_id}"

# Python packages: check via importlib.util.find_spec
new_py_packages = ["sklearn", "umap"]  # replace with your actual imports
for pkg in new_py_packages:
    r = subprocess.run(
        ["conda", "run", "-n", env_name, "python", "-c",
         f"import importlib.util; print('OK' if importlib.util.find_spec('{pkg}') else 'MISSING')"],
        capture_output=True, text=True, timeout=15,
    )
    if "MISSING" in r.stdout or r.returncode != 0:
        print(f"Installing {pkg} into {env_name}...")
        subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install", pkg],
            capture_output=True, text=True, timeout=300,
        )

# R packages: check via requireNamespace
new_r_packages = ["ggplot2"]  # replace with your actual library() calls
for pkg in new_r_packages:
    r = subprocess.run(
        ["conda", "run", "-n", env_name, "Rscript", "-e",
         f'if (requireNamespace("{pkg}", quietly=TRUE)) cat("OK") else cat("MISSING")'],
        capture_output=True, text=True, timeout=15,
    )
    if "MISSING" in r.stdout or r.returncode != 0:
        print(f"Installing R package {pkg} into {env_name}...")
        subprocess.run(
            ["conda", "run", "-n", env_name, "Rscript", "-e",
             f'install.packages("{pkg}", repos="https://cloud.r-project.org", quiet=TRUE)'],
            capture_output=True, text=True, timeout=600,
        )
```

**Why:** The `complete_modification()` 6-test suite includes an `imports_scan` test that extracts ALL imports from your code (Python AST + R `library()` calls) and verifies each is available. If you add `import sklearn` or `library(ggplot2)` without installing, the test FAILS and the modification is rejected.

**Common Python package name mismatches:** `sklearn` → `pip install scikit-learn`, `cv2` → `opencv-python`, `PIL` → `Pillow`, `yaml` → `PyYAML`

**Note:** The pre-edit check above is a best-effort early catch. Step 3.5 below is the
automated safety net that scans ALL imports after editing — it will catch anything you miss.

Apply the changes based on modification type (see Type-Specific Instructions below).

**Write the modified code to the files:**
```python
worker_path.write_text(modified_worker_code)
# If MCP server also changed:
server_path.write_text(modified_server_code)
```

### Step 3.5: Post-Edit Dependency Audit (MANDATORY)

**After writing modified files, you MUST scan them for ALL import statements and verify
that every dependency is installed in the conda env. Do NOT proceed to Step 4 (Test)
until every import resolves successfully.**

This prevents the `imports_scan` test in Step 4 from failing on fixable dependency issues.
Without this step, a modification that adds `import sklearn` will fail testing, trigger a
rollback attempt, and waste time — when the fix is just `pip install scikit-learn`.

```python
import ast, sys, subprocess
from pathlib import Path

env_name = f"user_{tool_id}"
worker_path = TOOLS_USER_DIR / f"{tool_id}_worker.py"
server_path = TOOLS_USER_DIR / f"{tool_id}_mcp_server.py"

# 1. Extract all imports from modified files
all_imports = set()
for fpath in [worker_path, server_path]:
    if not fpath.exists():
        continue
    tree = ast.parse(fpath.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                all_imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                all_imports.add(node.module.split(".")[0])

# 2. Filter stdlib and local modules
stdlib = set(sys.stdlib_module_names) | {"worker_utils", "base_mcp"}
third_party = sorted(all_imports - stdlib)
print(f"Third-party imports found: {third_party}")

# 3. Test each import in the conda env
missing = []
for mod in third_party:
    r = subprocess.run(
        ["conda", "run", "-n", env_name, "python", "-c", f"import {mod}"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        missing.append(mod)
        print(f"  MISSING: {mod}")
    else:
        print(f"  OK: {mod}")

# 4. Install missing deps
if missing:
    print(f"Installing {len(missing)} missing dependencies: {missing}")
    import_to_pip = {
        "sklearn": "scikit-learn", "cv2": "opencv-python",
        "PIL": "Pillow", "skimage": "scikit-image",
        "yaml": "pyyaml", "Bio": "biopython",
    }
    pip_names = [import_to_pip.get(m, m) for m in missing]
    r = subprocess.run(
        ["conda", "run", "-n", env_name, "pip", "install"] + pip_names,
        capture_output=True, text=True, timeout=1800,
    )
    if r.returncode != 0:
        print(f"Batch install failed, trying one by one...")
        for pkg in pip_names:
            subprocess.run(
                ["conda", "run", "-n", env_name, "pip", "install", pkg],
                capture_output=True, text=True, timeout=600,
            )

    # 5. Verify all imports now resolve
    still_missing = []
    for mod in missing:
        r = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", f"import {mod}"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            still_missing.append(mod)
    if still_missing:
        print(f"CRITICAL: Still missing after install: {still_missing}")
    else:
        print("All dependencies installed and verified.")

    # 6. Re-export env.yaml after installing new packages
    env_yaml = subprocess.run(
        ["conda", "env", "export", "-n", env_name, "--no-builds"],
        capture_output=True, text=True, timeout=60,
    ).stdout
    (TOOLS_USER_DIR / f"{tool_id}_env.yaml").write_text(env_yaml)
    print(f"Environment re-exported to tools_user/{tool_id}_env.yaml")
else:
    print("All dependencies already available — no action needed.")
```

**This step is NON-NEGOTIABLE. Every import in the modified files MUST resolve in the
conda env before proceeding to Step 4. A modification that fails `imports_scan` wastes
a rollback attempt on a fixable issue.**

### Step 4: Test

Run the 6-test suite on the modified files:

```python
test_result = complete_modification(
    tool_id,
    backup_version=backup_version,
    module=entry.get("module") or entry.get("package", tool_id),
    task_type=entry.get("task_type", "spatial_clustering"),
)
print(f"\nTest results: {test_result['tests_passed']}/{test_result['tests_total']} tests ran and passed")
for r in test_result["results"]:
    status = "SKIPPED" if r.get("skipped") else ("PASS" if r["passed"] else "FAIL")
    print(f"  [{status}] {r['test']}: {r['detail'][:100]}")
if not test_result["real_data_validated"]:
    print("  NOTE: real_data did not run on this host — the worker was never executed on real data.")
```

`tests_passed`/`tests_total` count only the tests that **ran**. The `real_data` test needs a
benchmark fixture, and a host without one (a pip-installed deployment, where `benchmark_data` is
pruned from the wheel) gets back a `skipped` entry with an `INCONCLUSIVE` detail instead. So
`5/5 passed` and `success=True` are a normal, honest result for a suite whose strongest test never
ran — that is what `real_data_validated` is for. Report it: when it is false, tell the user the
change passed the static and dry-run checks but was **not** exercised on data. Do not describe the
modification as "fully tested", and do not print a plain `[FAIL]` for a test that was skipped —
it did not fail, it did not happen.

### Step 5: Fix or Rollback

If tests fail, try to fix the issue (up to 2 attempts). If still failing, rollback:

```python
if not test_result["success"]:
    # Attempt to fix based on test error messages
    # ... read error details, fix the code, rewrite files ...
    
    # Re-test
    test_result = complete_modification(tool_id, backup_version=backup_version,
                                         module=entry.get("module") or entry.get("package", tool_id),
                                         task_type=entry.get("task_type", "spatial_clustering"))
    
    if not test_result["success"]:
        # After 2 failed attempts, ROLLBACK
        rb = rollback_to_backup(tool_id, version=backup_version,
                                reason="Tests failed after modification")
        print(f"\nRolled back to v{rb['version_restored']}. Original tool restored.")
        print("Tell user: Modification could not be applied — tests failed. Original tool is intact.")
        # STOP HERE — do NOT proceed to Step 6
```

### Step 6: Commit (Only if All Tests Passed)

**CRITICAL: `finalize_modification()` MUST ONLY be called after `complete_modification()` returns `success=True`. Calling it on a failed test run will record a broken modification.**

```python
if test_result["success"]:
    files_changed = [f"{tool_id}_worker.py"]
    # Add server if it was modified:
    # files_changed.append(f"{tool_id}_mcp_server.py")
    
    finalize_modification(
        tool_id,
        backup_version=backup_version,
        user_request=description,
        modification_type=modification_type,
        files_changed=files_changed,
        test_results=test_result,
    )
    print(f"\nModification committed successfully. Backup v{backup_version} preserved for rollback.")
```

### Step 7: Update Knowledge

Update stored knowledge files to reflect the changes:

```python
if test_result["success"]:
    # Re-bootstrap to capture updated state
    # Or selectively update function_signatures if parameters changed
    bootstrap_knowledge(tool_id)  # Simplest approach: full re-parse
    print("Knowledge updated.")
```

**After completing all 7 steps, tell the user:**
- What was changed
- Which tests ran and passed — and, when `real_data_validated` is false, that the worker was never
  run on real data on this host. Never say "all 6 tests passed" without reading that flag.
- That a backup is available for rollback
- That changes take effect on agent restart (in-memory state is stale)

---

## Type-Specific Instructions

### Type: `parameter_change`

User says: "change resolution to 1.0", "use 20 neighbors", "set cluster_key to my_clusters"

**What to change:**
1. In `{tool_id}_worker.py`: Find the `parser.add_argument("--resolution", type=float, default=0.6)` line and change the `default=` value
2. In `{tool_id}_mcp_server.py`: Find the function signature `resolution: float = 0.6` and change the default
3. Both files MUST agree on the new default value

**Use knowledge:** Read `function_signatures.json` → `worker_cli.parameters` to find the exact parameter name and its current default. Read `mcp_function.parameters` to find the corresponding MCP parameter name.

### Type: `pipeline_addition`

User says: "add UMAP visualization after clustering", "add normalization step"

**What to change:**
1. Read `pipeline_steps` from knowledge to understand current flow order
2. Write new code block at the correct position in the worker (after the step it follows)
3. If new imports needed, add them to the import section
4. If the addition produces new output files, add them to `WorkerOutput.add_output_files()`

**Use knowledge:** Read `api_discovery.json` → `all_functions` to check if the desired function was seen during creation. Read `pipeline_steps` to determine insertion point.

### Type: `pipeline_removal`

User says: "remove the scaling step", "skip PCA"

**What to change:**
1. Read `pipeline_steps` to identify the exact code block to remove
2. Comment out or remove the code lines
3. Verify no downstream steps depend on the removed step's output

### Type: `function_swap`

User says: "switch from leiden to louvain", "use a different clustering method"

**This is the most complex type.** Steps:
1. Check `api_discovery.json` → `rejected_alternatives` for the alternative function
2. If found there: use the stored signature
3. If NOT found: re-discover via inspect in the conda env:
   ```python
   import subprocess
   r = subprocess.run(["conda", "run", "-n", f"user_{tool_id}", "python", "-c",
                       f"import {module}; import inspect; print(inspect.getsource({module}.{new_function}))"],
                      capture_output=True, text=True, timeout=30)
   ```
4. Replace the core function call in the worker
5. Update imports if the new function is from a different module
6. Update knowledge files after successful modification

### Type: `output_change`

User says: "also save a CSV", "change output format to TSV", "add a UMAP plot"

**What to change:**
1. Add output code after the processing section in the worker
2. Add the new file to `WorkerOutput.add_output_files()` call
3. No MCP server changes needed (output_dir already captures all files)

### Type: `error_handling`

User says: "add check for fewer than 100 cells", "handle missing spatial coordinates"

**What to change:**
1. Add validation in the `data_readiness_checks()` function (or create one if absent)
2. The check should append to the `issues` list
3. Existing error handling in the try/except block catches RuntimeError

### Type: `signature_change`

User says: "add a min_cells parameter", "remove the cluster_key parameter"

**What to change — BOTH files must be updated:**
1. In `{tool_id}_worker.py`: Add/remove `parser.add_argument("--min_cells", type=int, default=100)`
2. In `{tool_id}_mcp_server.py`: Add/remove parameter in the function signature AND in the `args` list passed to `run_worker_cli()`
3. Wire the parameter through: MCP function → args list → worker argparse → core logic

### Type: `environment_change`

User says: "add scikit-learn to the environment", "install plotly"

**What to change:**
1. Install the package: `subprocess.run(["conda", "run", "-n", f"user_{tool_id}", "pip", "install", package], ...)`
2. Re-export env.yaml to the anchored path — a bare `tools_user/...` resolves against the process
   CWD, and `health_check_tools` then keeps reading (and offering to rebuild the env from) the stale
   copy: `subprocess.run(["conda", "env", "export", "-n", f"user_{tool_id}", "--no-builds", "-f", str(TOOLS_USER_DIR / f"{tool_id}_env.yaml")], ...)`
3. If the installation is for a new import in the worker, also modify the worker code

---

## Flow: Manual Rollback

When user says "rollback sopa", "undo changes to sopa", "restore previous version":

**Rollback Best Practices:**
- Always use `list_backups()` to see available versions BEFORE rolling back
- Use the EXACT version number from `list_backups()` output
- **Only roll back to a version whose `restorable` is True.** A version can be listed and still be
  refused: its metadata is missing (a backup interrupted before it was written — the checksums
  rollback verifies against were never recorded) or unparseable, it records no files, one of the
  files it records is not in the directory, or a file no longer matches its recorded SHA-256. The
  `refusal` field on each entry says which. Never pass `version=None` (latest) without checking:
  latest is exactly the version most likely to be the interrupted one.
- Do NOT guess version numbers or use "latest minus one"
- After rollback, verify files match expected state

```python
# Show available backups
backups = list_backups(tool_id)
usable = [b for b in backups if b.get("restorable")]
if not backups:
    print(f"No backups available for '{tool_id}'.")
elif not usable:
    print(f"'{tool_id}' has {len(backups)} backup director(ies) but NONE can be restored:")
    for b in backups:
        print(f"  v{b['version']} ({b['timestamp']}): {b.get('refusal', 'rollback refuses it')}")
    print("Tell the user the tool cannot be rolled back, and why. Do NOT call rollback_to_backup().")
else:
    print(f"Available backups for '{tool_id}':")
    for b in backups:
        # `refusal` is the reason rollback would refuse this version -- report it rather than a
        # bare UNRESTORABLE, because what the user does next depends on which ground it is.
        mark = "" if b.get("restorable") else f"  [UNRESTORABLE: {b.get('refusal', '')}]"
        print(f"  v{b['version']} ({b['timestamp']}): {b.get('user_request', 'N/A')}{mark}")

    # Roll back to the newest RESTORABLE version -- not to `None` (= latest), which lands on an
    # interrupted backup if that is what the last one was.
    version = max(b["version"] for b in usable)
    rb = rollback_to_backup(tool_id, version=version, reason="User requested rollback")
    print(f"\nRolled back to v{rb['version_restored']}.")
    print(f"Files restored: {rb['files_restored']}")
```

### Post-Rollback Verification

After any rollback, ALWAYS verify the restored state:

```python
# Read restored files and verify key properties
worker_code = (TOOLS_USER_DIR / f"{tool_id}_worker.py").read_text()
server_code = (TOOLS_USER_DIR / f"{tool_id}_mcp_server.py").read_text()

# Check that the modification was actually undone
# For example, if you rolled back a parameter addition:
# Verify the added parameter is no longer in the code
print(f"Worker length: {len(worker_code)} chars")
print(f"Server length: {len(server_code)} chars")

# Run health check to confirm tool is still functional
report = health_check_tools(tool_ids=[tool_id])
print(f"Post-rollback health: {report['tools'][0]['health']}")
```

If verification fails, the backup may have been contaminated (e.g., files were edited before `begin_modification()` was called). Use `list_backups()` to find an earlier clean version — one with `restorable: True` and a lower version number — and rollback to that instead.

## Flow: View Modification History

When user says "show modification history", "what changes were made to sopa":

```python
knowledge = read_knowledge(tool_id)
# read_knowledge never raises -- it returns what it could parse and lists what it could not. A
# damaged modification_log.json therefore arrives as an ABSENT key, indistinguishable from a tool
# that was never modified. Report the warnings first, or this flow answers "no modifications" for a
# tool whose history is on disk and unreadable.
for w in knowledge.get("warnings", []):
    print(f"WARNING: {w}")
if knowledge.get("modification_log"):
    mods = knowledge["modification_log"].get("modifications", [])
    print(f"Modification history for '{tool_id}' ({len(mods)} modifications):")
    for m in mods:
        # `total` counts only the tests that RAN. real_data is skipped on a host with no benchmark
        # fixture, so 5/5 on its own does not mean the worker was ever exercised -- always report
        # real_data_validated beside the fraction. Read every field with .get(): a record written
        # before this field existed, or one a user hand-edited, must not KeyError mid-listing.
        tr = m.get("test_results", {})
        validated = tr.get("real_data_validated")
        if validated is True:
            note = "real_data validated"
        elif validated is False:
            note = "real_data NOT run -- worker unexercised"
        else:
            note = "real_data unknown -- record predates this field"
        print(f"  {m['mod_id']} ({m['timestamp']}): {m['type']} — {m['user_request']}")
        print(f"    Files: {m['files_changed']}, Tests: {tr.get('passed', '?')}/{tr.get('total', '?')} ({note})")
elif any("modification_log.json" in w for w in knowledge.get("warnings", [])):
    print(f"The modification history for '{tool_id}' EXISTS but could not be read (see warning).")
    print("Do NOT report this tool as unmodified -- how many times it changed is unknown.")
else:
    print(f"No modifications recorded for '{tool_id}'.")

# Also show backups
backups = list_backups(tool_id)
print(f"\n{len(backups)} backup(s) available.")
```

## Flow: View Tool Knowledge

When user says "show tool knowledge", "what do you know about sopa":

```python
knowledge = read_knowledge(tool_id)
# Every branch below is `if knowledge.get(<key>)`, and a file that failed to parse has no key -- so
# without this the flow just prints less, which reads as "nothing is stored".
for w in knowledge.get("warnings", []):
    print(f"WARNING: could not read {w}")
if not knowledge.get("has_knowledge"):
    print(f"No stored knowledge for '{tool_id}'. Will bootstrap on first modification.")
else:
    if knowledge.get("api_discovery"):
        disc = knowledge["api_discovery"]
        print(f"Package: {disc.get('package')}")
        print(f"Source: {disc.get('source_url')}")
        print(f"Selected function: {disc.get('selected_function', {}).get('module_path')}")
        if disc.get("rejected_alternatives"):
            print(f"Rejected alternatives: {len(disc['rejected_alternatives'])}")
    if knowledge.get("function_signatures"):
        sigs = knowledge["function_signatures"]
        steps = sigs.get("pipeline_steps", [])
        print(f"\nPipeline ({len(steps)} steps):")
        for s in steps:
            print(f"  {s['step']}. {s['call']}")
    if knowledge.get("bootstrapped"):
        print("\n(Knowledge was bootstrapped from existing files, not captured during creation)")
```

---

## Safety Rules

1. **ALWAYS** call `begin_modification()` before ANY file changes. This creates a backup.
2. **ALWAYS** run `complete_modification()` after changes. This runs ALL 6 tests.
3. **ALWAYS** rollback on test failure after at most 2 fix attempts.
4. **NEVER** modify protected infrastructure files: `base_mcp.py`, `worker_utils.py`, `user_skill.py`, `trash_manager.py`, `knowledge_manager.py`, `install_log.json`
5. **NEVER** modify files outside `tools_user/{tool_id}_*` — only the target tool's files.
6. **NEVER** delete or recreate the conda environment during modification. Code changes only.
7. **NEVER** modify a trashed tool. Tell user: "Restore it first with 'restore {tool_id}'."
8. **NEVER** modify tools during active benchmarking.
9. **ALWAYS** update stored knowledge after a successful modification.
10. If unsure which tool the user means, call `list_user_tools()` and show options.
11. On test failure, report what failed with the exact error message.
12. **NEVER** ask the user to write code. Do all modifications autonomously.
13. The 6-test suite is MANDATORY — never skip a test yourself. `real_data` is the one the *host* can
    skip for you: with no benchmark fixture present it comes back `skipped` with an `INCONCLUSIVE`
    detail and is left out of the `tests_passed`/`tests_total` count, so `5/5 passed` and
    `success=True` do not mean the worker ran. Read `real_data_validated` and report it.
14. After modification, tell user: "Changes take effect on agent restart."

### spatialomicsgym_name consistency

If the modification renames the `@mcp.tool()` function in the MCP server file,
you MUST also update the `spatialomicsgym_name` field in `mcp_config_user.yaml` to match.
The `finalize_modification()` function will auto-detect and fix this, but explicit
consistency is preferred. A mismatch causes `session.call_tool()` to fail with "Unknown tool".

## Edge Cases

- **Tool has no knowledge directory**: `begin_modification()` auto-bootstraps from existing files.
- **Tool is trashed**: Blocked with clear error: "Restore first."
- **Parameter value causes bad results**: Test 5 (real data) catches this *when it runs*. On a host
  with no fixture it is skipped, `real_data_validated` is false, and nothing checked the new value —
  say so rather than calling the change verified. Rollback preserves original either way.
- **New import not in conda env**: Install via `conda run -n user_{tool_id} pip install <package>` first.
- **Multiple parameters to change**: Apply all changes in one modification cycle (one backup, one test run).
- **User wants to see what will change before applying**: Read the files, show the diff, get confirmation, then apply.
