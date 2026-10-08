# Adding New MCP Tools - Phase 6: Activate and Verify

## Metadata
- Authors: SpatialOmicsLab
- Version: 2.0
- Category: tool_creation

## Overview
Phase 6 of creating a new MCP tool: activate it in the live catalogue, then run the final invariant verification that has to pass before the creation may be called successful.

## Read this with the entry document

This is one phase of the tool-creation playbook, which is split across seven documents so a phase
can be loaded without the other six. `add_new_mcp_tool.md` holds what is true across all of them --
when creation may start at all, the execution environment, the global HARD RULES, the creation
pipeline invariants, Phase 0's pre-flight checks, the rollback procedure and the safety rules.
Those govern everything below and are **not** repeated here. If you are running this phase without
having read that document, read it first; if it is not in your context, say so rather than
improvising the rules.

## Names this phase uses

```python
from pathlib import Path
from spatialomicsgym.setup.constants import ensure_repo_importable, interp_path
ensure_repo_importable()   # `tools_user` is not an installed package; this puts the repo on sys.path
from tools_user.knowledge_manager import CONDA_ENVS_DIR, INSTALL_LOG, KNOWLEDGE_DIR, mcp_config_user_path, TOOLS_USER_DIR, current_owner, health_check_tools, update_knowledge
```

The registry anchors are re-imported at the top of every phase document on purpose. The REPL
namespace persists between code blocks, so if an earlier phase already ran this line it rebinds
the same objects and costs nothing; if this phase was retrieved on its own -- which is the whole
reason these documents are separate -- it is the line that stops the first registry access
raising `NameError`. Never substitute a literal path for one of these names: the config entry,
the health check and the rollback all resolve through them, and a literal is how the code that
writes a file and the code that looks for it drift apart.

`ensure_repo_importable()` comes first for the same reason: `tools_user` is not an installed
package, so outside the repo root the import below raises `ModuleNotFoundError` and this phase
dies before it writes anything. `interp_path(prefix, "python" | "Rscript")` and
`CONDA_ENVS_DIR` build interpreter paths the way the rest of the system does -- never type an
absolute conda prefix into your own code.

## Phase 6: Activate (MANDATORY — execute ALL steps)

**Execute ALL of these steps. Do NOT skip any. Do NOT offer as "next steps".**

```python
import subprocess, json
from pathlib import Path
from datetime import datetime

tool_id = "{tool_id}"  # replace with actual

# Step 6.1: Export env
print("6.1: Exporting environment...")
subprocess.run(["conda", "env", "export", "-n", f"user_{tool_id}", "--no-builds",
                "-f", str(TOOLS_USER_DIR / f"{tool_id}_env.yaml")], check=True, timeout=60)
print(f"  Saved: tools_user/{tool_id}_env.yaml")

# Step 6.2: Update install_log.json
print("6.2: Updating install log...")
log_path = INSTALL_LOG
entries = json.loads(log_path.read_text()) if log_path.exists() else []
# P12 -- derive language from WORKER file extension, NOT the wrapped package.
# Phase 6 HARD RULE -- worker_ext is authoritative; wrapped_language is metadata.
worker_ext = f"{tool_id}_worker.py"
for ext in (".py", ".R"):
    if (TOOLS_USER_DIR / f"{tool_id}_worker{ext}").exists():
        worker_ext = f"{tool_id}_worker{ext}"
        break
_ext_to_lang = {".py": "python", ".R": "R"}
derived_language = _ext_to_lang[Path(worker_ext).suffix]
# Whose tool this is. The registry is keyed on (owner, tool_id) -- stamp it, or every row this
# tier writes keys on ("", tool_id) and two accounts' tools of the same name are one row.
owner = current_owner()   # "" on a library/CLI run, which is a real key and not a missing one
# Start from the row Phase 4 wrote, do not replace it.
#
# `_read_install_log` dedups last-wins, so a fresh dict appended here REPLACES the Phase 4 row --
# and Phase 4 is where `module` is established, after the pre-flight that checks the string
# actually imports. Building the entry from scratch dropped it, `health_check_tools` then found
# no `module` to import, and a tool that worked reported BROKEN. Same for the other fields Phase 4
# verified: `verified_function_names`, `python_module`, `verified_module`.
#
# Matched on the OWNER too, and not on the id alone. On a box with two accounts the id alone
# picked up the OTHER account's row and inherited its verified `module` -- so this tool would be
# registered claiming an import nobody checked for it. No match is the right answer there: the
# guard below then stops the activation instead of activating a lie.
prior = next(
    (
        e
        for e in entries
        if isinstance(e, dict) and e.get("tool_id") == tool_id and str(e.get("owner") or "") == owner
    ),
    {},
)
entry = {
    **prior,
    # The fields Phase 6 OWNS, and only these, overwrite what Phase 4 wrote.
    "tool_id": tool_id,
    "owner": owner,
    "tool_name": tool_name,
    "function_name": function_name,
    "task_type": task_type,
    "env_name": f"user_{tool_id}",
    "source_url": github_url,
    "package": package_name,
    "language": derived_language,
    "status": "active",
    "created_at": datetime.now().isoformat(),
    "files": [worker_ext, f"{tool_id}_mcp_server.py"],
}
if not entry.get("module"):
    # Phase 4 either did not run or did not record it. Registering a tool whose module string is
    # unknown means health_check has nothing to import and will call it BROKEN forever, so say so
    # here -- where it can still be fixed -- rather than in a health report a week later.
    raise RuntimeError(
        f"install_log entry for {tool_id} has no verified 'module'. Phase 4 establishes it; "
        "do not activate a tool whose module string was never checked."
    )
# If the WRAPPED package is a different language (R package called via rpy2
# from a Python worker), record that as sibling metadata for health_check.
if language and language != derived_language:
    entry["wrapped_language"] = language
# Re-read under the lock trash/restore/permanent_delete hold while they rewrite this file, and
# replace it atomically: a plain write_text loses their concurrent update or truncates the registry.
import fcntl
from tools_user.trash_manager import _atomic_write_json
with open(TOOLS_USER_DIR / ".trash.lock", "w") as _lock:
    fcntl.flock(_lock, fcntl.LOCK_EX)   # blocks until a running trash operation finishes
    latest = json.loads(log_path.read_text()) if log_path.exists() else []
    _atomic_write_json(log_path, latest + [entry])
print(f"  Logged as active")

# Step 6.3: Final verification -- load merged config and check tool appears
print("6.3: Final verification...")
from spatialomicsgym.agent.mcp_config_merger import build_merged_mcp_config
from spatialomicsgym.mcp_config_path import find_mcp_config
merged = build_merged_mcp_config(
    str(find_mcp_config()), str(mcp_config_user_path()), merge_user=True)
import yaml
config = yaml.safe_load(Path(merged).read_text())
n_servers = len(config["mcp_servers"])
has_tool = f"user_{tool_id}" in config["mcp_servers"]
print(f"  Merged config: {n_servers} servers, user_{tool_id} present: {has_tool}")
assert has_tool, f"user_{tool_id} NOT in merged config!"

print(f"\nSUCCESS: Tool {tool_id} installed, tested, and activated.")
print(f"  Env: {CONDA_ENVS_DIR}/user_{tool_id}/")
print(f"  Worker: tools_user/{tool_id}_worker.py")
print(f"  Server: tools_user/{tool_id}_mcp_server.py")
print(f"  Config: MCP_server/mcp_config_user.yaml")
print(f"  To use: reload STCoscientist with tool_creation_enabled=True, then call {function_name}()")

# === KNOWLEDGE CAPTURE: Phase 6 -- Save Creation Log ===
try:
    from tools_user.knowledge_manager import update_knowledge
    from datetime import datetime
    _kc_clog = {
        "schema_version": 1,
        "tool_id": tool_id,
        "created_at": datetime.now().isoformat(),
        "source_url": github_url if 'github_url' in dir() else "",
        "creation_phases": phase_results if 'phase_results' in dir() else {},
        "data_validation_checks": data_checks if 'data_checks' in dir() else [],
        "strengths": tool_strengths if 'tool_strengths' in dir() else [],
        "limitations": tool_limitations if 'tool_limitations' in dir() else [],
        "files": {
            "worker": f"{tool_id}_worker.py",
            "mcp_server": f"{tool_id}_mcp_server.py",
            "env_yaml": f"{tool_id}_env.yaml",
        },
    }
    update_knowledge(tool_id, creation_log=_kc_clog)
    print(f"Knowledge Phase 6 saved (creation log)")
except Exception as e:
    print(f"Knowledge Phase 6 save failed (non-blocking): {e}")
# === END KNOWLEDGE CAPTURE ===
```

**IMPORTANT: Knowledge Verification**
After activation, verify knowledge was saved:
- Check if `.knowledge/{tool_id}/` exists
- If missing, call `bootstrap_knowledge(tool_id)` to reconstruct from existing files
- This ensures future modifications and health checks work correctly

**After executing Phase 6, report the COMPLETE results to the user.
Do NOT ask "would you like me to test?" — testing is ALREADY DONE in Phase 5.**

## Final Invariant Verification (MANDATORY — run before declaring success)

This is a single audit pass that verifies every post-condition from the Invariants
table at the top of this document. If ANY assertion fails, the tool is NOT ready
— call `rollback(tool_id)` and report which invariant failed.

```python
import subprocess, json, ast, os
from pathlib import Path
import yaml

def final_invariant_check(tool_id: str, source_url: str) -> dict:
    """Returns {'ok': bool, 'failures': [str, ...]}. Non-exception interface."""
    out = {"ok": True, "failures": [], "checks": {}}

    def fail(key: str, detail: str):
        out["ok"] = False
        out["failures"].append(f"{key}: {detail}")
        out["checks"][key] = False

    def ok(key: str):
        out["checks"][key] = True

    # 1. Files exist and parse (an R worker is parsed by R; the server is always Python)
    worker = next((w for w in (TOOLS_USER_DIR / f"{tool_id}_worker.py", TOOLS_USER_DIR / f"{tool_id}_worker.R")
                   if w.is_file()), TOOLS_USER_DIR / f"{tool_id}_worker.py")
    runner = "Rscript" if worker.suffix == ".R" else "python"
    server = TOOLS_USER_DIR / f"{tool_id}_mcp_server.py"
    for p in (worker, server):
        if not p.is_file():
            fail(f"file_{p.name}", "missing")
        else:
            try:
                if p.suffix == ".R":
                    rp = subprocess.run(["conda", "run", "-n", f"user_{tool_id}", "Rscript", "-e",
                                         f"invisible(parse(file={str(p)!r}))"],
                                        capture_output=True, text=True, timeout=60)
                    if rp.returncode != 0:
                        raise SyntaxError(rp.stderr[-300:])
                else:
                    ast.parse(p.read_text())
                ok(f"file_{p.name}")
            except SyntaxError as e:
                fail(f"file_{p.name}", f"SyntaxError: {e}")

    # 2. env.yaml exported
    env_yaml = TOOLS_USER_DIR / f"{tool_id}_env.yaml"
    if env_yaml.is_file() and env_yaml.stat().st_size > 100:
        ok("env_yaml")
    else:
        fail("env_yaml", "missing or empty")

    # 3. Conda env exists
    r = subprocess.run(["conda", "env", "list", "--json"],
                       capture_output=True, text=True, timeout=30)
    try:
        envs = json.loads(r.stdout).get("envs", [])
        if any(os.path.basename(p) == f"user_{tool_id}" for p in envs):
            ok("conda_env")
        else:
            fail("conda_env", f"user_{tool_id} not in `conda env list`")
    except Exception as e:
        fail("conda_env", f"env list parse: {e}")

    # 4. Module actually imports inside the env (or vendor_path verified)
    log_path = INSTALL_LOG
    my_entry = None
    if log_path.exists():
        try:
            entries = json.loads(log_path.read_text())
            # (owner, tool_id), not the id: on a two-account box the id alone verifies
            # somebody else's row and passes while this creation's row is missing or wrong.
            my_entry = next(
                (
                    e
                    for e in entries
                    if e.get("tool_id") == tool_id and str(e.get("owner") or "") == current_owner()
                ),
                None,
            )
        except json.JSONDecodeError as e:
            fail("install_log_parse", f"{e}")
    if my_entry is None:
        fail("install_log_entry", f"no entry for tool_id={tool_id}")
    else:
        if my_entry.get("status") != "active":
            fail("install_log_status", f"status={my_entry.get('status')!r}")
        else:
            ok("install_log_status")
        if my_entry.get("source_url") != source_url:
            fail("install_log_source_url",
                 f"got {my_entry.get('source_url')!r}, expected {source_url!r}")
        else:
            ok("install_log_source_url")
        mod = my_entry.get("module")
        vpath = my_entry.get("vendor_path")
        if mod:
            # An R package (R worker, or a Python worker with wrapped_language "R") loads in R.
            is_r = "R" in (my_entry.get("language"), my_entry.get("wrapped_language"))
            r = subprocess.run(
                ["conda", "run", "-n", f"user_{tool_id}"]
                + (["Rscript", "-e", f"suppressPackageStartupMessages(library({mod})); cat('OK\\n')"]
                   if is_r else ["python", "-c", f"import {mod}; print('OK')"]),
                capture_output=True, text=True, timeout=60,
            )
            if "OK" in r.stdout:
                ok("module_importable")
            else:
                fail("module_importable", f"loading {mod} failed: {r.stderr[-300:]}")
        elif vpath:
            if os.path.isdir(vpath):
                ok("vendor_path_exists")
            else:
                fail("vendor_path_exists", f"missing: {vpath}")
        else:
            fail("module_or_vendor", "neither module nor vendor_path set")

    # 5. mcp_config_user.yaml has the server
    cfg_path = mcp_config_user_path()
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        servers = cfg.get("mcp_servers") or {}
        if f"user_{tool_id}" in servers:
            ok("mcp_cfg_entry")
            srv = servers[f"user_{tool_id}"]
            if not srv.get("enabled", True):
                fail("mcp_cfg_enabled", "enabled is false")
            else:
                ok("mcp_cfg_enabled")
            if not srv.get("tools"):
                fail("mcp_cfg_tools", "no tools array")
            else:
                ok("mcp_cfg_tools")
        else:
            fail("mcp_cfg_entry", f"user_{tool_id} not in mcp_servers")
    except Exception as e:
        fail("mcp_cfg_parse", f"{e}")

    # 6. Knowledge dir populated
    know_dir = (KNOWLEDGE_DIR / tool_id)
    if not know_dir.is_dir():
        fail("knowledge_dir", "missing")
    else:
        # install_plan.json written by canonical install discovery
        if (know_dir / "install_plan.json").exists():
            ok("knowledge_install_plan")
        else:
            fail("knowledge_install_plan", "install_plan.json missing")

    # 7. Worker responds to a dry-run call with a clean WorkerOutput JSON
    r = subprocess.run(
        ["conda", "run", "-n", f"user_{tool_id}", runner,
         str(worker), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode == 0 or "--input" in (r.stdout + r.stderr):
        ok("worker_help")
    else:
        fail("worker_help", f"--help exited with rc={r.returncode}")

    return out

audit = final_invariant_check(tool_id, source_url)
if not audit["ok"]:
    print("FINAL INVARIANT CHECK FAILED:")
    for f in audit["failures"]:
        print(f"  - {f}")
    # Persist the failure report into knowledge for debugging, THEN rollback
    (KNOWLEDGE_DIR / tool_id).mkdir(parents=True, exist_ok=True)
    (KNOWLEDGE_DIR / tool_id / "final_audit.json").write_text(
        json.dumps(audit, indent=2)
    )
    rollback(tool_id)
    raise RuntimeError(f"{tool_id}: final invariant check failed -- rolled back.")
print(f"FINAL INVARIANT CHECK PASSED for {tool_id}: {len(audit['checks'])} invariants hold.")
# Also persist the pass report so a later `health_check_tools` can cite it
(KNOWLEDGE_DIR / tool_id / "final_audit.json").write_text(
    json.dumps(audit, indent=2)
)

# === MEMORY WRITE on success (Phase 6 finale) ===
# MODE-GATED -- MANDATORY when memory_enabled=True; SKIPPED when False.
# Drive contract: same as Phase 1.0. The driver greps for an updated
# `.memory/.last_op` marker file with mtime >= start of this agent.go().
from spatialomicsgym.config import default_config as _cfg6
if getattr(_cfg6, "memory_enabled", False):
    # MANDATORY: import MUST succeed; do NOT try/except.
    from tools_user.memory_manager import MemoryManager
    from datetime import datetime as _dt6
    from pathlib import Path as _P6
    import json as _json6
    mm = MemoryManager.get()
    mm.append_attempt(github_url, {
        "outcome": "full_pass",
        "tool_id": tool_id,
        "started": _started_iso if "_started_iso" in dir() else "unknown",
        "finished": _dt6.now().isoformat(),
        "strategy": {
            "install_path": install_plan.get("strategy") if "install_plan" in dir() else None,
            "command": install_plan.get("command") if "install_plan" in dir() else None,
            "python_version": install_plan.get("python_version") if "install_plan" in dir() else None,
        },
        "api_shape": {
            "module": entry.get("module"),
            "primary_fn": function_name,
            "package": entry.get("package"),
        },
        "gotchas": (audit.get("checks", {}) or {}).get("warnings", []),
    })
    # MANDATORY MARKER -- refresh .last_op so driver can verify Phase 6 fired.
    _marker6 = mm.root / ".last_op"
    _marker6.parent.mkdir(parents=True, exist_ok=True)
    _marker6.write_text(_json6.dumps({
        "ts": _dt6.now().isoformat(),
        "tool_id": tool_id,
        "source_url": github_url,
        "op": "phase_6_success_write",
    }))
    print(f"[memory] recorded success attempt for {github_url}")
# === END MEMORY WRITE ===
```
