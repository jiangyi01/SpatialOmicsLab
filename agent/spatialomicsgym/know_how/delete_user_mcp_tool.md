# Delete / Manage User MCP Tools

## Metadata
- **Category**: tool_management
- **Triggers**: delete tool, remove tool, uninstall tool, trash tool, restore tool, empty trash, show trash, list my tools, trash info
- **Requires**: `tool_creation_enabled = True`
- **Version**: 1.0
- **Last Updated**: 2026-04-15

## Overview

This document guides STCoscientist in managing user-created MCP tools through a **two-stage trash box system** (like a desktop recycle bin). Tools are first moved to trash (reversible), then permanently deleted only from trash (irreversible, requires user confirmation).

**Key principle:** "delete" always means TRASH (soft delete). Permanent deletion requires explicit user intent and confirmation.

### Run this FIRST, before any other code block

Your working directory is wherever the user launched from — **not** the repo. Use these names for
the registry files below; a bare `Path("tools_user/...")` would read an empty file in the user's
data directory and report "no such tool" for a tool that is installed.

```python
from spatialomicsgym.setup.constants import ensure_repo_importable
ensure_repo_importable()   # `tools_user` is not an installed package; this puts the repo on sys.path
from tools_user.knowledge_manager import INSTALL_LOG, KNOWLEDGE_DIR, mcp_config_user_path, TOOLS_USER_DIR
```

## Deletion-Specific HARD RULES

Deletion is **reversibility-preferring** and **safety-first**. Different
flows have different auto-mode semantics — get this right:

1. **Confirmation policy by flow**:
   - Soft-delete (trash), restore, list, trash-info → **execute
     immediately, NO confirmation** (reversible operations).
   - Permanent-delete → **ALWAYS preview + require explicit confirmation
     step**. The know-how's existing `preview_permanent_delete(tid)` →
     user-confirm → `permanent_delete(tid)` flow is correct; do NOT
     short-circuit it in auto-mode. The only time "would you like to
     permanently delete?" is the right response is here — elsewhere it's
     a P7 violation.
   - On mid-flow failure: print what failed + root cause, never
     "Option A/B/C" menu.

2. **Tool lookup — case/slash/`.git`-tolerant**:
   ```python
   target = user_input.rstrip("/").lower()
   if target.endswith(".git"): target = target[:-4]
   entries = json.loads(INSTALL_LOG.read_text())
   matches = [e for e in entries
              if e.get("source_url","").rstrip("/").lower().removesuffix(".git") == target
              or target in (e.get("tool_id") or "").lower()]
   ```
   Prevents "No install_log entry found" on a URL that differs only in
   trailing slash or `.git` suffix.

3. **Post-op verification is MANDATORY** (deletion's core correctness
   guarantee). After every state-changing flow, re-observe:
   - `install_log.json` — status field for target tool_id
   - `mcp_config_user.yaml` — presence of `user_{tool_id}` server key
   - `.trash/{tool_id}/` directory
   - `conda env list` — presence/absence of `user_{tool_id}`
   Use the post-condition table above as the ground truth. Never claim
   PASS from the return value of `trash_tool()` / `restore_tool()` alone
   — the primitive can return success while leaving inconsistent state
   on an interrupted run.

4. **Atomicity: any mid-flow exception → do NOT leave partial state.**
   Wrap trash/restore/permadelete invocations in try/except; on exception,
   re-read state and reconcile toward the LAST known-good checkpoint
   (e.g., on trash failure, ensure files are either fully in tools_user/
   or fully in .trash/ — never split across both).
   `TrashStateError` and `TrashNotFoundError` are the exception to this.
   Both come from the entry guard, before anything has been touched, so
   there is no partial state and nothing to reconcile — reconciling after
   one of them undoes the operation that already succeeded. See the
   Idempotence rule for what to do with them instead.

5. **Never `import` a user MCP server file** — same as creation's P13.
   Inspect via filesystem reads + `list_user_tools()` / `trash_info()`,
   never via Python module-import on `mcp_servers.user_*`.

6. **Self-capture NEW edge cases**. If a flow hits an error not listed
   in Edge Cases below, append to
   `KNOWLEDGE_DIR / tool_id / "delete_issues.json"` before rolling back.
   Use the imported `KNOWLEDGE_DIR`, never a bare `tools_user/.knowledge/`
   path — that one resolves against the user's launch directory, so the
   capture lands somewhere nothing reads. The deletion surface is small;
   every new edge case is signal.

7. **Naming convention is read-only here** — creation side guarantees
   `spatialomicsgym_name = {tool_id}_run`. Deletion never re-derives or renames.
   If an existing install_log entry has a legacy spatialomicsgym_name that
   doesn't match `{tool_id}_run`, trash/restore/permadelete still
   works (they operate on tool_id + files, not spatialomicsgym_name) — don't
   refuse the operation.

## Quick Reference

| User Intent | STCoscientist Action | Confirmation? |
|------------|-----------|---------------|
| "delete sopa" / "remove sopa" | `trash_tool("sopa")` | No (reversible) |
| "restore sopa" / "undelete sopa" | `restore_tool("sopa")` | No (reversible) |
| "show trash" / "what's in trash" | `list_user_tools(status="trashed")` | No |
| "list my tools" / "show tools" | `list_user_tools()` | No |
| "trash info" / "disk usage" | `trash_info()` | No |
| "permanently delete sopa" | `preview_permanent_delete("sopa")` then confirm then `permanent_delete("sopa")` | **YES** |
| "empty trash" | Preview each, confirm, `permanent_delete()` each | **YES** |
| "delete all tools" | `trash_tool()` for each active tool | No (reversible) |

## Deletion Lifecycle Invariants (READ FIRST)

Every trash / restore / permanent-delete operation must satisfy a pre-condition on
entry and leave a verifiable post-condition on exit. The table is the single source
of truth; do not invent extra steps or skip checks.

| Flow | Pre-condition | Post-condition (all must hold) | Cross-consistency check |
|------|---------------|--------------------------------|--------------------------|
| 1. Trash | `status=='active'`, tool files present, conda env present | `status=='trashed'`, `.trash/{tid}/` contains worker/server + `.trash_meta.json`, config entry REMOVED, conda env RETAINED | `install_log.trashed ⇒ not in mcp_config.mcp_servers` |
| 2. Restore | `status=='trashed'`, `.trash/{tid}/` exists, conda env still present | `status=='active'`, files back under `tools_user/`, config entry RESTORED, `.trash/{tid}/` GONE | `install_log.active ⇒ mcp_config has user_{tid}` |
| 3. Permanent delete | `status=='trashed'` (MUST have been trashed first) | no install_log entry, no `.trash/{tid}/`, no `.knowledge/{tid}/`, no `vendor_{tid}/`, no `user_{tid}` conda env, no mcp_config entry | `list_user_tools(...).filter(tool_id=tid) == []` |
| 4. Empty trash | ≥ 1 tool in trash | every trashed tool fully purged per Flow 3 | same as Flow 3 but for all |
| 5. List | — | unchanged | — |
| 6. Trash info | — | unchanged | — |

**Idempotence rule:**
A repeated call does not come back with a flag — it raises, from the entry guard, before anything
is touched. That raise is the "already in the state you asked for" signal. Catch it, report the
state, and stop: do not roll back (HARD RULE 4), do not retry, do not call the other primitive to
"fix" it.

| Repeated call | What it raises | What to tell the user |
|---------------|----------------|------------------------|
| `trash_tool(tid)` on a trashed tool | `TrashStateError` — "already in trash" | already in the trash; nothing to do |
| `restore_tool(tid)` on an active tool | `TrashStateError` — "active, not trashed" | already active; nothing to do |
| `permanent_delete(tid)` on an unknown tool | `TrashNotFoundError` — "not found in install_log" | already gone; nothing to do |

Each message names the call to use instead — pass that through rather than paraphrasing it. In a
loop over several tools ("delete all tools", "empty trash"), catch per tool and keep going: one
tool already in the trash must not abort the tools after it.

**Atomicity rule:**
`trash_manager` writes `install_log.json` and `mcp_config_user.yaml` through a temp file and `os.replace()`, so neither of those can tear. The tool's own files move with `shutil.move`, which is a copy-then-unlink whenever `.trash/` sits on a different filesystem — not a rename, and not atomic. Two things bound that: `.trash_meta.json` is written before the first move, and the move loop carries its own rollback. STCoscientist must never bypass any of it with `os.remove` / `shutil.move` / direct config writes. If `trash_manager` raises, inspect `.trash/{tid}/.trash_meta.json` to see which step committed and report per-step status — do not try to "clean up" by hand. The rollback can itself fail (it logs and continues), so files split across `tools_user/` and `.trash/` is a reachable state, not an impossible one: HARD RULE 3's post-op verification is what catches it.

**Cross-file divergence (must-NOT conditions):**
After any flow, the following pairs must agree — if they don't, the system is in an inconsistent state and must be audited:
- `install_log.json[tid].status == 'active'`  ⇔  `mcp_config_user.yaml.mcp_servers['user_' + tid]` is present and `enabled: true`.
- `install_log.json[tid].status == 'trashed'` ⇔  `mcp_config_user.yaml.mcp_servers['user_' + tid]` is absent  AND  `tools_user/.trash/{tid}/` exists.
- No entry in `install_log.json`  ⇔  no `mcp_config_user.yaml` key AND no `.trash/{tid}/` AND no `.knowledge/{tid}/` AND no `vendor_{tid}/` AND no `user_{tid}` conda env.

Run the cross-consistency snippet in the "Post-operation audit" section below after every state-changing flow.

## Pre-flight Check

Execute this before ANY trash/restore/delete operation:

```python
# 1. Verify feature is enabled
from spatialomicsgym.config import default_config
assert default_config.tool_creation_enabled, "tool_creation_enabled is False! Enable it first."

# 2. Import trash manager
from tools_user.trash_manager import (
    trash_tool, restore_tool, permanent_delete, preview_permanent_delete,
    list_user_tools, trash_info, check_tool_id_available,
    TrashError, TrashSafetyError, TrashStateError, TrashNotFoundError,
)

# 3. Parse-sanity of the two state files (do NOT mutate yet)
import json, yaml
from pathlib import Path
log_path = INSTALL_LOG
cfg_path = mcp_config_user_path()
if log_path.exists():
    try:
        json.loads(log_path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"install_log.json corrupt before operation: {e}. "
                         "Fix or restore from backup — do NOT proceed.")
if cfg_path.exists():
    try:
        yaml.safe_load(cfg_path.read_text())
    except yaml.YAMLError as e:
        raise SystemExit(f"mcp_config_user.yaml corrupt before operation: {e}. "
                         "Fix or restore from backup — do NOT proceed.")
print("Trash manager loaded; state files parse cleanly.")
```

## Post-operation audit (MANDATORY after any flow that mutates state)

Run this right after Flows 1, 2, 3, or 4. It verifies every cross-consistency
invariant from the table above. Any FAIL means the system is in a divergent
state and needs inspection — do not declare success.

```python
import json, yaml, subprocess, os
from pathlib import Path

def post_op_audit(tool_id: str, expected_state: str) -> dict:
    """expected_state ∈ {'active','trashed','absent'}."""
    out = {"tool_id": tool_id, "expected": expected_state,
           "ok": True, "failures": []}

    def fail(msg):
        out["ok"] = False
        out["failures"].append(msg)

    # Observed state
    log = json.loads(INSTALL_LOG.read_text()) \
        if INSTALL_LOG.exists() else []
    entry = next((e for e in log if e.get("tool_id") == tool_id), None)
    observed_status = (entry or {}).get("status") if entry else "absent"

    cfg = yaml.safe_load(mcp_config_user_path().read_text() or "") \
        if mcp_config_user_path().exists() else {}
    server_key = f"user_{tool_id}"
    # Read the block the merger reads, not the one the file ought to hold. A block the agent
    # appended at column 0 instead of nesting it is rescued by
    # mcp_config_merger._recover_top_level_servers and served, so "not under mcp_servers" is not
    # the same as "not wired" -- a trashed tool whose block sits up there is still advertised.
    # The fallback uses the merger's own predicate (a command and a tools list) so bookkeeping
    # keys beside mcp_servers stay invisible.
    block = (cfg.get("mcp_servers") or {}).get(server_key)
    if not isinstance(block, dict):
        stray = cfg.get(server_key)
        if isinstance(stray, dict) and "command" in stray and isinstance(stray.get("tools"), list):
            block = stray
    in_cfg = isinstance(block, dict)
    # mcp_config_merger skips `enabled: false`, so present is not served -- see the invariant above.
    cfg_enabled = in_cfg and bool(block.get("enabled", True))

    trash_dir = TOOLS_USER_DIR / f".trash/{tool_id}"
    know_dir = (KNOWLEDGE_DIR / tool_id)
    worker = TOOLS_USER_DIR / f"{tool_id}_worker.py"
    server = TOOLS_USER_DIR / f"{tool_id}_mcp_server.py"

    r = subprocess.run(["conda", "env", "list", "--json"],
                       capture_output=True, text=True, timeout=30)
    envs = []
    try:
        envs = [os.path.basename(p) for p in json.loads(r.stdout).get("envs", [])]
    except Exception:
        pass
    env_present = f"user_{tool_id}" in envs

    out["observed"] = {
        "install_log_status": observed_status, "mcp_cfg_present": in_cfg,
        "mcp_cfg_enabled": cfg_enabled,
        "trash_dir_present": trash_dir.exists(),
        "knowledge_dir_present": know_dir.exists(),
        "worker_file_present": worker.is_file(),
        "server_file_present": server.is_file(),
        "conda_env_present": env_present,
    }

    # Apply invariants based on expected state
    if expected_state == "active":
        if observed_status != "active":
            fail(f"install_log status is {observed_status!r}, expected 'active'")
        if not in_cfg:
            fail(f"mcp_config_user.yaml missing user_{tool_id}")
        elif not cfg_enabled:
            fail(f"mcp_config_user.yaml has user_{tool_id} but enabled: false, "
                 "so the merger skips it and the tool cannot be called")
        if trash_dir.exists():
            fail(f".trash/{tool_id}/ still exists (should be gone after restore)")
        if not worker.is_file() or not server.is_file():
            fail("worker/server .py files not back in tools_user/")
        if not env_present:
            fail(f"conda env user_{tool_id} is gone (should still exist)")
    elif expected_state == "trashed":
        if observed_status != "trashed":
            fail(f"install_log status is {observed_status!r}, expected 'trashed'")
        if in_cfg:
            fail(f"mcp_config_user.yaml still has user_{tool_id} (should be removed)")
        if not trash_dir.exists():
            fail(f".trash/{tool_id}/ missing (should hold moved files)")
        if worker.is_file() or server.is_file():
            fail("worker/server .py still in tools_user/ (should be in trash)")
        if not env_present:
            fail(f"conda env user_{tool_id} gone (must stay for later restore)")
    elif expected_state == "absent":
        if entry is not None:
            fail(f"install_log entry still present ({observed_status})")
        if in_cfg:
            fail(f"mcp_config_user.yaml still has user_{tool_id}")
        if trash_dir.exists():
            fail(f".trash/{tool_id}/ still present")
        if know_dir.exists():
            fail(f".knowledge/{tool_id}/ still present")
        if env_present:
            fail(f"conda env user_{tool_id} still exists")

    return out

audit = post_op_audit(tool_id, expected_state="trashed")  # or 'active' / 'absent'
if not audit["ok"]:
    print("POST-OP AUDIT FAILED:")
    for f in audit["failures"]:
        print(f"  - {f}")
    print("State-machine divergence — do not claim success.")
```

Use this right after every flow (trash ⇒ expected='trashed', restore ⇒ 'active', permanent-delete ⇒ 'absent'). Persist the audit dict into the knowledge dir (or a post-mortem log if the tool no longer exists) for traceability.

## CRITICAL RULES FOR DELETION

1. **"delete" = `trash_tool()`, NEVER `permanent_delete()`**
   When a user says "delete", "remove", or "uninstall" — ALWAYS use `trash_tool()`.
   `permanent_delete()` is ONLY for tools already in trash, with explicit user confirmation like "permanently delete" or "I confirm permanent deletion".

2. **Verify tool_id exists BEFORE calling any delete function:**
```python
# ALWAYS verify first
tools = list_user_tools()
target = next((t for t in tools if t["tool_id"] == tool_id), None)
if target is None:
    # Try fuzzy match
    matches = [t for t in tools if tool_id in t["tool_id"] or t["tool_id"] in tool_id]
    if matches:
        tool_id = matches[0]["tool_id"]
        print(f"Using closest match: {tool_id}")
    else:
        print(f"Tool '{tool_id}' not found. Available: {[t['tool_id'] for t in tools]}")
        # STOP — do not proceed
```

3. **Two-step deletion flow (MANDATORY):**
   - Step 1: `trash_tool(tool_id)` — moves to trash (reversible)
   - Step 2: `permanent_delete(tool_id)` — only after trash, only with explicit user confirmation
   - NEVER skip Step 1. NEVER call `permanent_delete()` on an active tool.

## Flow 1: Trash a Tool (Soft Delete)

When user says "delete", "remove", or "uninstall" a tool:

```python
try:
    result = trash_tool("sopa")
    print(f"Tool '{result['tool_id']}' moved to trash at {result['trashed_at']}.")
    print(f"  Files moved: {result['files_moved']}")
    print(f"  Conda env preserved for restore.")
    print()
    print("What you can do next:")
    print(f"  - Restore:           'restore sopa'")
    print(f"  - Permanently delete: 'permanently delete sopa'")
    print(f"  - View trash:         'show trash'")
except TrashNotFoundError as e:
    print(f"Tool not found: {e}")
    tools = list_user_tools()
    if tools:
        print("Available tools:")
        for t in tools:
            print(f"  - {t['tool_id']} (status: {t['status']})")
    else:
        print("No user tools installed.")
except TrashStateError as e:
    print(f"Cannot trash: {e}")
except TrashSafetyError as e:
    print(f"Safety check failed: {e}")
```

**Tell the user:** "Tool moved to trash. Files preserved. Restore anytime with 'restore sopa'. Permanently delete with 'permanently delete sopa'."

## Flow 2: Restore a Tool

When user says "restore" or "undelete":

```python
try:
    result = restore_tool("sopa")
    print(f"Tool '{result['tool_id']}' restored and active.")
    print(f"  Files restored: {result['files_restored']}")
    print("  Ready to use immediately.")
except TrashStateError as e:
    print(f"Cannot restore: {e}")
except TrashNotFoundError as e:
    print(f"Tool not found: {e}")
    trashed = list_user_tools(status="trashed")
    if trashed:
        print("Tools in trash:")
        for t in trashed:
            print(f"  - {t['tool_id']}")
    else:
        print("Trash is empty.")
```

**Tell the user:** "Tool restored and active. Ready to use."

## Flow 3: Permanently Delete a Single Tool from Trash

When user explicitly says "permanently delete" or "permanently remove":

**Step 1 - ALWAYS preview first (mandatory):**

```python
try:
    preview = preview_permanent_delete("sopa")
except TrashStateError as e:
    # Tool is active, not trashed
    print(f"{e}")
    print("Use 'delete sopa' to move it to trash first.")
    # STOP HERE - do not proceed
except TrashNotFoundError as e:
    print(f"{e}")
    # STOP HERE - do not proceed
```

**Step 2 - Show preview and ASK for confirmation:**

```python
if preview["success"]:
    print(f"The following will be PERMANENTLY removed for '{preview['tool_id']}':")
    print(f"  Conda environment: {preview['conda_env']} ({preview['conda_env_size']})")
    if preview["trash_files"]:
        print(f"  Tool files:")
        for fname, fsize in preview["trash_files"].items():
            print(f"    - {fname} ({fsize})")
    print(f"  Total disk to reclaim: {preview['total_reclaimable']}")
    print()
    print("This CANNOT be undone. Proceed? (yes/no)")
```

**Step 3 - ONLY after user confirms "yes":**

```python
result = permanent_delete("sopa")
if result["success"]:
    print(f"Tool '{result['tool_id']}' permanently deleted.")
    for step, status in result["steps"].items():
        print(f"  [{status.upper()}] {step}")
    print(f"  Disk reclaimed: {result['disk_reclaimed']}")
else:
    print(f"Partial failure for '{result['tool_id']}':")
    for step, status in result["steps"].items():
        print(f"  [{status.upper()}] {step}")
    for err in result["errors"]:
        print(f"  ERROR: {err}")
    print("You can retry - the tool entry is preserved for recovery.")
```

## Flow 4: Empty Trash (Bulk Permanent Delete)

When user says "empty trash":

```python
trashed = list_user_tools(status="trashed")
if not trashed:
    print("Trash is empty. Nothing to delete.")
else:
    # Show summary
    info = trash_info()
    print(f"Trash contains {info['trashed_count']} tool(s):")
    for t in info["per_tool"]:
        if t["status"] == "trashed":
            sizes = f"conda: {t['conda_size']}, files: {t.get('files_size', 'N/A')}"
            print(f"  - {t['tool_id']} ({sizes}, vendor: {t.get('vendor_size', 'N/A')})")
    print(f"Total reclaimable: {info['total_reclaimable']}")
    print()
    print("Permanently delete ALL trashed tools? This CANNOT be undone. (yes/no)")
    # WAIT for user confirmation before proceeding
```

After user confirms:

```python
for tool in trashed:
    tid = tool["tool_id"]
    result = permanent_delete(tid)
    status = "OK" if result["success"] else "PARTIAL"
    print(f"  [{status}] {tid}: {result['steps']}")
print("Trash emptied.")
```

## Flow 5: List Tools

```python
tools = list_user_tools()
if not tools:
    print("No user tools installed.")
else:
    active = [t for t in tools if t["status"] == "active"]
    trashed = [t for t in tools if t["status"] == "trashed"]
    if active:
        print(f"Active tools ({len(active)}):")
        for t in active:
            env_ok = "env OK" if t["conda_env_exists"] else "env MISSING"
            print(f"  - {t['tool_id']} ({t.get('task_type', 'unknown')}, {env_ok})")
    if trashed:
        print(f"Trashed tools ({len(trashed)}):")
        for t in trashed:
            print(f"  - {t['tool_id']} (trashed at {t.get('trashed_at', 'unknown')})")
```

## Flow 6: Trash Info (Disk Usage)

```python
info = trash_info()
print(f"Active tools:  {info['active_count']} ({info['active_conda_size']} in conda envs)")
print(f"Trashed tools: {info['trashed_count']} ({info['trashed_conda_size']} in conda envs)")
print(f"Trash files:   {info['trash_files_size']}")
print(f"Vendor trees:  {info['trashed_vendor_size']}")
print(f"Reclaimable:   {info['total_reclaimable']}")  # the three trashed sizes above
```

## Integration with Tool Creation Pipeline

When creating new tools (add_new_mcp_tool.md), STCoscientist should check for name conflicts with trashed tools during Phase 1:

```python
# During Phase 1, AFTER extracting tool_id:
try:
    from tools_user.trash_manager import check_tool_id_available
    avail = check_tool_id_available(tool_id)
    if not avail["available"]:
        print(f"ERROR: {avail['reason']}")
        # STOP creation - user must permanently delete the trashed tool first
        # or choose a different tool_id
except ImportError:
    pass  # trash_manager not installed yet, skip check
```

## Edge Cases

### Tool is active, user says "permanently delete"
```
"Tool 'sopa' is active. Use 'delete sopa' to move it to trash first."
```

### Tool not found
```
"Tool 'xyz' not found. Available tools: sopa, banksy"
```

### No tool name specified
Show list of tools with `list_user_tools()` and ask which one.

### Only 1 tool in trash, user says "permanently delete"
Show preview for that tool, ask confirmation.

### Conda env already manually removed
`permanent_delete()` reports "skipped" for conda step, continues with file cleanup.

### Partial failure on permanent delete
Show per-step report. Tool stays in install_log for retry.

### User says "delete all tools"
Trash each active tool (reversible). Do NOT permanently delete.

## Safety Rules

**MANDATORY - Follow ALL rules without exception:**

1. **ALWAYS** import and call `trash_manager` functions. **NEVER** write your own deletion, move, or cleanup code.
2. **NEVER** use `os.remove()`, `shutil.rmtree()`, or `subprocess` for file/directory operations on user tools. Only `trash_manager` functions.
3. **NEVER** modify files in `tools/`, `spatialomicsgym/`, `skills/`, or `MCP_server/mcp_config.yaml`.
4. **Default/built-in tools CANNOT be deleted.** Only user tools (those in `install_log.json`).
5. **Permanent delete requires user confirmation.** Trash does NOT (it's reversible).
6. **Cannot permanently delete an active tool.** MUST trash first (two-stage guarantee).
7. **ALWAYS** call `preview_permanent_delete()` and show results BEFORE `permanent_delete()`.
8. On partial failure, **report what succeeded/failed**. User can retry.
9. When user says "delete", **default to TRASH** (not permanent delete).
10. **NEVER** delete `install_log.json` itself. Only modify entries within it.
11. **NEVER** touch conda environments that don't start with `user_`.
12. If unsure which tool user means, call `list_user_tools()` and show options.
