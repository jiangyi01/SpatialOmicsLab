# Memory System for User MCP Tools

## Metadata
- **Category**: tool_management
- **Triggers**: show memory, forget memory, clear memory, memory stats, memory status, enable memory, disable memory, what have you learned, reset memory
- **Requires**: `default_config.memory_enabled = True` (boot-time flag)
- **Version**: 1.0
- **Last Updated**: 2026-04-24

## Overview

STCoscientist's memory system remembers prior tool-creation attempts so that
future attempts on the same tool (or similar tools) can leverage
that experience. Memory has two layers:

- **Short-term memory** — per-tool (`source_url`-keyed): what worked,
  what failed, which install strategy won, API shape, gotchas.
- **Long-term memory** — cross-tool aggregate: stats by language /
  install path / domain, known hangs, recurring patterns.

**CRITICAL principle**: memory is ADVISORY only. It NEVER overrides:
- HARD GATE rollback decisions (Test 1/2/3 critical fails always roll back)
- Safety audit rules (P6, P20, P22, etc.)
- Phase 5 tests (always run fresh)
- Deletion / modification safety invariants

Memory provides HINTS at the start of Phase 1. Every subsequent phase
still runs normally.

## Opt-in model — OFF by default

Memory is disabled by default. To enable:

```python
from spatialomicsgym.config import default_config
default_config.memory_enabled = True
# OR
# SOG_MEMORY_ENABLED=true python ...
```

Must be set BEFORE constructing STCoscientist. Changing at runtime has no effect
until the next STCoscientist instance is built.

**When disabled**: STCoscientist behaves identically to the pre-memory system.
No `.memory/` directory is created, no memory logic runs, and this
know-how file is not loaded into STCoscientist's prompt.

## Top-Level HARD RULES (memory-specific)

1. **Memory is advisory only.** When a memory hint exists, STCoscientist MAY
   use it to select a starting strategy in Phase 1 / 2, but MUST
   still run all subsequent phase gates (HARD GATE, safety audit,
   Phase 5 tests).

2. **Memory must be read VIA CODE EXECUTION, never recalled from
   training.** When the user asks "what do you remember about X?",
   STCoscientist MUST run code that reads the record and report its
   contents. Ask the manager where the record is —
   `MemoryManager.get().short_term_path(X)` — instead of spelling a path.
   The store moves with `SOG_MEMORY_PATH` and `default_config.memory_path`,
   so a spelled path reads nothing on an install that set either, and the
   answer becomes "no memory for this tool" about a tool with a recorded
   full_pass. Do not infer or hallucinate memory content — always read.

3. **Never emit raw memory content into code decisions.** Memory
   strings are sanitized but may still contain unexpected content.
   Use memory fields as ENUM / KEY-VALUE hints only, not as
   executable strings (e.g. never `eval()` a memory field).

4. **Memory deletion is a SEPARATE vocabulary from tool deletion.**
   - "forget memory for X" = delete the memory entry only (tool
     files/env stay intact)
   - "delete tool X" / "permanently delete X" = use the deletion
     system (existing delete_user_mcp_tool.md flows)

5. **Memory writes follow the lifecycle state-machine.** Memory is
   written at Phase 6 activate (success), rollback (failure), or
   at the end of modification. Memory is deleted at permanent_delete
   (unless `keep_memory=True` flag is set).

6. **Memory writes NEVER block the primary lifecycle.** Memory
   write failures (disk full, lock timeout, HMAC mismatch) are
   caught and logged. The main flow continues without memory.

## Using memory in the CREATION flow

### At Phase 1 Discover (memory read)

```python
# Inside Phase 1 Discover, after preflight passes:
from spatialomicsgym.setup.constants import ensure_repo_importable
ensure_repo_importable()   # `tools_user` is not an installed package; this puts the repo on sys.path

from tools_user.memory_manager import MemoryManager
if default_config.memory_enabled:
    _mm = MemoryManager.get()
    memory_hints = _mm.read_short_term(github_url)
    if memory_hints and memory_hints.get("attempts"):
        # Inject as SUFFIX to the discovery prompt — never prefix,
        # to preserve LLM prompt-cache hits
        hint_text = _mm.format_hint_for_prompt(github_url)
        if hint_text:
            print(f"[memory] using prior-attempt hint:\n{hint_text}")
            # Use hint to select starting strategy, but still run
            # README discovery + canonical ladder.
        else:
            # Attempts on file, but none of them succeeded. Only full_pass and
            # vendor_fallback_pass are ever injected (L14), so there is no recipe
            # to reuse -- say that rather than printing an empty hint.
            n = len(memory_hints["attempts"])
            print(f"[memory] {n} prior attempt(s) on file, none successful. No hint to reuse.")
```

### At Phase 6 Activate (memory write on success)

```python
# After install_log.json has been updated and final_invariant_check passes:
if default_config.memory_enabled:
    _mm.append_attempt(
        source_url=github_url,
        record={
            "outcome": "full_pass",  # or "vendor_fallback_pass"
            "tool_id": tool_id,
            "started": _started_iso,
            "finished": _finished_iso,
            "time_to_outcome_sec": int(elapsed_sec),
            "strategy": {
                "install_path": winning_install_path,   # enum string
                "python_version": python_version,
                "key_deps_pinned": dep_pins_dict,
                "repo_sha_at_attempt": repo_sha,         # H15 fix
            },
            "api_shape": {
                "primary_fn": verified_module,
                "primary_input_type": primary_input_arg,
                "required_secondary": required_secondary_list,
                "typical_defaults": default_params_dict,
            },
            "gotchas": gotchas_list,
            "error_classes_seen": error_classes_list,
        },
    )
    _mm.update_long_term(
        outcome="full_pass", language=derived_language,
        install_path=winning_install_path, time_sec=elapsed_sec,
        domain=tool_domain,
    )
```

### At rollback (memory write on failure)

```python
# Inside rollback(), AFTER all file/env/config cleanup succeeds:
if default_config.memory_enabled:
    try:
        _mm = MemoryManager.get()
        _mm.append_attempt(
            source_url=github_url,
            record={
                "outcome": "rolled_back",
                "tool_id": tool_id,
                "started": _started_iso,
                "finished": _finished_iso,
                "gotchas": [str(e) for e in errors[:5]],
                "error_classes_seen": list(set(err_classes)),
            },
        )
    except Exception as e:
        # F1 — memory failure never blocks rollback
        print(f"[memory-warn] could not record rollback: {e}")
```

## Using memory in the DELETION flow

At `permanent_delete(tool_id)`, after every other step succeeds,
memory for that source_url is deleted unless the caller passes
`keep_memory=True`:

```python
# Inside permanent_delete():
# ... (env/trash/install_log/config cleanup)
if default_config.memory_enabled and not keep_memory:
    try:
        from tools_user.memory_manager import MemoryManager
        MemoryManager.get().delete_short_term(source_url)
    except Exception as e:
        # F12 — memory failure never blocks permadelete
        print(f"[memory-warn] could not delete memory: {e}")
```

`trash_tool` / `restore_tool` NEVER touch memory — soft-delete is
reversible and memory should survive.

## Using memory in MODIFICATION

After successful modification (files back in place, tests pass):

```python
if default_config.memory_enabled:
    _mm.append_attempt(
        source_url=github_url,
        record={
            "outcome": "modification_passed",
            "modification_type": mod_type,
            "api_shape": new_api_shape,     # may have changed
            ...
        },
    )
```

## Using memory in HEALTH-CHECK (READ-ONLY)

Health-check may ANNOTATE its report with memory context but MUST NOT
write to memory (F4 — health-check HARD RULE: zero writes):

Pass `include_memory=True` and each ACTIVE tool's entry gains a `memory` key.
It is off by default, so a report you did not ask for it on is unchanged:

```python
from tools_user.knowledge_manager import health_check_tools

report = health_check_tools(include_memory=True)
for tool in report["tools"]:
    mem = tool.get("memory") or {}   # {} = memory off, no record, or unreadable
    if mem:
        print(tool["tool_id"], mem["prior_attempts"], mem["best_attempt_outcome"])
```

`best_attempt_outcome` is the outcome STRING of the best attempt
(`full_pass`, `rolled_back`, ...), NOT `best_attempt_id` — that is an integer
index into the attempt list. It is ranked, not filtered, so on a tool whose
every attempt failed it reports a FAILURE outcome. That is the point: read it
before describing the tool as proven.

The verdict never moves. `memory` is a separate key, computed after
HEALTHY/DEGRADED/BROKEN is already decided (F11), and a memory subsystem that
raises yields `{}` rather than failing the tool's check. Health-check still
writes nothing to memory (F4).

## User-facing commands

### Read commands (always safe)

| Intent | STCoscientist action |
|--------|-----------|
| "show memory" | Render summary of long-term stats + list of known-hangs |
| "show memory for X" | Read `MemoryManager.get().short_term_path(X)` and render attempts table |
| "memory stats" | Aggregate counter summary |

### Delete commands (confirmation policy)

| Intent | Confirmation? | STCoscientist action |
|--------|--------------|-----------|
| "forget memory for X" | NO (per-tool, reversible-ish — just re-learn) | `_mm.delete_short_term(X)` |
| "clear short-term memory" | **YES** | backup + `_mm.delete_all_short_term()` |
| "clear long-term memory" | **YES** | backup + `_mm.delete_long_term()` |
| "clear all memory" / "reset memory" | **YES + 2nd confirmation** | backup + wipe `.memory/` |

Before any BULK delete: auto-backup to `.memory/.backups/{timestamp}_{scope}/`
with 7-day retention. The `MemoryManager._backup_before_delete()` handles
this automatically.

### Toggle commands

| Intent | STCoscientist response |
|--------|-------------|
| "enable memory" | "Memory is a boot-time flag. Set `default_config.memory_enabled = True` in your config, then restart STCoscientist." |
| "disable memory" | Same message, inverted. If currently enabled, note: "STCoscientist will stop reading/writing memory on next restart, but existing memory files remain on disk. Use 'clear all memory' to wipe." |

## Interpretation of ambiguous requests

When the user says **"forget spadecon"**, ask for disambiguation:
> "Do you want to (a) forget my MEMORY of spadecon (tool files/env
>  stay), or (b) permanently delete the spadecon tool entirely?"

Only after the user picks, proceed. If they explicitly say "forget
memory for X", proceed with (a) immediately — no confirmation.

## Memory-aware Phase 1 flow (integration details)

When memory is enabled and a prior successful attempt exists for
this source_url AND the hint carries no `STALE:` line, STCoscientist MAY:

1. Use `strategy.install_path` as the FIRST rung to try on the
   canonical install ladder (e.g., skip pip and go straight to
   vendor if memory says that's what worked).
2. Use `api_shape.primary_fn` as a FIRST GUESS for the worker's
   main call — but tutorial harvest still runs and cross-checks.
3. Use `key_deps_pinned` as a starting point for `conda create`
   package list.

A hint older than `default_config.memory_staleness_days` (30 by default;
set it to 0 or less to switch the check off) carries a `STALE:` line naming
its age. When that line is present the three shortcuts above are withdrawn —
the recipe is history, not a starting point, so run the canonical ladder from
the top. A record with no usable timestamp is NOT treated as stale.

STCoscientist MUST NOT:
- Skip Phase 5 tests based on memory
- Skip safety audit based on memory
- Skip HARD GATE based on memory
- Write to install_log or mcp_config without running Phase 4-6

## Troubleshooting

### "Memory mode not enabled" messages

The user attempted a memory command but `memory_enabled=False`. STCoscientist
responds with the canonical refusal sentence:

> "Memory mode is not enabled. To enable, set
>  `default_config.memory_enabled = True` in your config (or set
>  `SOG_MEMORY_ENABLED=true` env var) and restart STCoscientist."

### Quarantined memory file

If a memory file was moved to `.memory/.quarantine/`, it means the
JSON was malformed, schema was unknown, or HMAC verification failed.
STCoscientist should report the quarantine location to the user and continue
as if memory were empty for that entry. User can inspect or delete
the quarantine file manually.

### Memory that feels slow, or a store that keeps growing

Reads do not slow down as records accumulate. `read_short_term` opens
`.memory/short_term/<sha256(source_url)[:16]>.json` directly, so a store
holding ten records and one holding ten thousand cost the same to read.
If a memory call feels slow, the file count is not the reason — look at
lock contention (`lock_timeout_sec`) or at the caller.

What does grow is disk: every record stays until something deletes it.
The one operation whose cost scales with the file count is "clear
short-term memory", which copies the whole directory into a timestamped
folder under `.memory/.backups/` before removing the records. That is
also the only way to shrink the directory.

Those backup folders are the other thing that accumulates.
`_purge_old_backups()` drops them once they are older than
`backup_retention_days` (7 by default), and it runs on the next delete.
It prunes `.backups/` and nothing else.

### HMAC verification failures

Every short-term / long-term file is signed with a per-install secret
stored at `.memory/.secret`. If the secret file is lost or corrupted,
existing memory files will all fail HMAC verification and get
quarantined. User can either restore the secret from backup or
accept the loss (all prior memory quarantined, fresh start).

## Interaction with existing know-hows

This file is loaded into STCoscientist's prompt ONLY when `memory_enabled=True`.
When disabled, STCoscientist's prompt contains:
- add_new_mcp_tool.md
- delete_user_mcp_tool.md
- modify_user_mcp_tool.md
- health_check_user_mcp_tools.md

When enabled, this file is additionally loaded:
- + memory_user_mcp_tools.md (this file)

The creation playbook (`add_new_mcp_tool*.md`) does carry its memory hooks --
the Phase 1.0 consult, the Phase 6 success write and the rollback write --
each behind `if getattr(default_config, "memory_enabled", False):`, so with
memory off they are text that never runs. Nothing else depends on this file.

## Edge cases

### User modifies `.memory/` files manually

Possible, but only in one form: drop the `_hmac` key in the same edit.
Every record is signed on write and the signature is verified on read, so
an otherwise perfect edit — valid JSON, every other field intact, one
string changed — no longer matches, and the whole record is moved to
`.quarantine/`. Nothing is shown to the user; the next question about that
tool is answered "no prior attempts". Malformed JSON is quarantined the
same way.

A record carrying no `_hmac` key is read as-is. That is what makes hand
editing possible at all, and it is not a permanent downgrade: the next
lifecycle write signs the file again.

To recover a record that was already quarantined, tell the user it is not
lost. The file under `.quarantine/` is intact and readable — delete its
`_hmac` key and move it back into `short_term/` under its original
filename (the leading timestamp the quarantine added is not part of it).

### User enables memory mid-session

Has no effect. STCoscientist must be restarted. STCoscientist explicitly says so.

### User copies `.memory/` between machines

Tool-id and source_url keying is portable. But:
- `.memory/.secret` is per-install — copying breaks HMAC verification
  on all files (they'll be quarantined on first read on the new
  machine).
- Absolute paths (if ever stored, which we avoid per M9) break.

Recommendation: do NOT copy `.memory/` between machines. Start fresh
on each install.

### User deletes `.memory/.secret` by accident

All existing memory files fail HMAC. STCoscientist quarantines them on next
read. Fresh memory starts being written with a new secret.

## DO NOT

- Do NOT bypass Phase 5 tests based on memory.
- Do NOT auto-apply pattern suggestions without feature-detector match.
- Do NOT allow ad-hoc user memory writes via prompts ("remember that
  X needs Y"). Memory writes happen only via the lifecycle.
- Do NOT include memory content in code decisions (only as key/value
  hints).
- Do NOT add a memory step to another know-how file unless it sits behind
  the same `memory_enabled` gate as the creation playbook's hooks.
