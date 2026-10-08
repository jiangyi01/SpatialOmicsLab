# Adding New MCP Tools from GitHub/Paper Links

## Metadata
- Authors: SpatialOmicsLab
- Version: 2.0
- Category: tool_creation

## Overview
This document describes how to create new MCP tools from GitHub or paper links.

**ONLY proceed if ALL of these conditions are true:**
1. The user EXPLICITLY asked to add, install, or create a new tool
2. `tool_creation_enabled=True` in the config (check `default_config.tool_creation_enabled`)
3. The user provided a GitHub URL, paper URL, or package name

**When these conditions are met, proceed IMMEDIATELY without asking for confirmation.
The user's explicit request + tool_creation_enabled=True IS the confirmation.
Do NOT ask "are you sure?" or "please confirm". Just execute all phases.**

**Do NOT trigger if:**
- User is just asking about a tool ("what is SpaGCN?")
- User mentions a GitHub link in conversation without asking to install
- `tool_creation_enabled` is False

## Execution Environment

**You have FULL access to run conda, pip, subprocess, and file I/O commands.
You CAN and MUST execute these commands directly in your Python code blocks.
Do NOT generate scripts for the user to run later — execute everything NOW.**

You have:
- `subprocess.run(["conda", "create", ...])` — works, you have conda
- `subprocess.run(["conda", "run", "-n", env, "pip", "install", ...])` — works
- File write access to `tools_user/` and `MCP_server/mcp_config_user.yaml`
- Network access for `pip install` and GitHub fetches
- Full disk access for conda env creation under this host's `envs/` directory (`CONDA_ENVS_DIR`)

**Do NOT say "this can't be done in this environment" or "run this script later".
Execute ALL phases directly in your code execution blocks.**

### Run this FIRST, before any other code block

Your working directory is wherever the user launched from — **not** the repo — and your conda is
wherever this deployment installed it, which you do not get to assume. Every path below is one of
these names, so run this once and reuse them; a bare `Path("tools_user/...")` would write into the
user's data directory instead, where the health check and the rollback path (which use these same
constants) will never find it, and a hand-written absolute env path is `FileNotFoundError` on a
host whose conda lives under `~/miniconda3`, a Miniforge prefix or a Slurm module.

```python
from pathlib import Path
from spatialomicsgym.setup.constants import ensure_repo_importable, interp_path
ensure_repo_importable()   # `tools_user` is not an installed package; this puts the repo on sys.path
from tools_user.knowledge_manager import CONDA_ENVS_DIR, INSTALL_LOG, KNOWLEDGE_DIR, mcp_config_user_path, TOOLS_USER_DIR, current_owner
```

`CONDA_ENVS_DIR` is this host's real `envs/` directory (`Path(conda_envs_root())`) and
`interp_path(prefix, "python" | "Rscript")` builds an interpreter path inside a named env the way
the rest of the system does. Use them for every env path. **Never type an absolute conda prefix
into your own code** — it is right on this box and wrong on the user's.

## Top-Level HARD RULES (global, enforced across all phases)

These rules govern every response in every phase. If you are about to emit
text or code that violates any of them, rewrite until it doesn't.

1. **Never end a response with "would you like", "shall I", "please confirm",
   "Option A/B/C", or "if you'd like".** The Deliberative-Thinking Protocol
   forces you to commit to ONE next action before printing. If you
   genuinely lack information to choose, execute the SAFEST recoverable
   action (usually: roll back, do not change state) and print exactly what
   was done. This applies even when creation FAILS — on a rollback,
   print the rollback steps taken and the root cause, NOT a menu of
   options for the user. "I rolled back because X" is a complete
   response. "I rolled back. Would you like me to try Y, or Z, or W?"
   is NOT.

2. **Never `import` a user MCP server file.** MCP servers are not Python
   modules — they are subprocesses spoken to via stdio JSON-RPC. If your
   generated code contains any of:
   - `from mcp_servers.user_* import ...`
   - `import mcp_servers.user_*`
   - `importlib.import_module("mcp_servers.*")`
   - `mcp_servers.user_<tool>.<spatialomicsgym_name>(...)` (attribute-access style)
   - referring to a function as `"mcp_servers.user_<tool>.<name>"` in a
     log/plan line
   — rewrite it. Written into the tool's own files (they run in its worker
   env) or run from a `#!BASH` / `#!CLI` cell, the imports fail with
   "No module named 'mcp_servers'"; in a python `<execute>` cell they gain
   nothing over the name already in scope.

   **The correct invocation pattern** — copy verbatim:
   ```python
   # Exactly ONE correct way. The spatialomicsgym_name is registered by
   # STCoscientist's framework after add_mcp() loaded the merged config.
   # Call it as if it were any other STCoscientist tool:
   result = spavae_run(input_h5ad="/path/to/input.h5ad",
                       output_dir="/path/to/outdir")
   # — NO imports. NO "mcp_servers." prefix. NO attribute access.
   # Just the spatialomicsgym_name function, as a top-level callable.
   ```

   If the spatialomicsgym_name isn't in scope, the agent doesn't have the tool
   loaded — that is a symptom of the tool not being registered. Re-read
   the prompt's "IMPORTANT" block: it tells you to look up the `spatialomicsgym_name`
   from `MCP_server/mcp_config_user.yaml`'s `tools[].spatialomicsgym_name` entry.

3. **Before moving on, verify the env.** After Phase 2 and again after
   every codegen step (Phase 3, 3.5, and the new 3.75 env-closure step),
   run a real import/fixture test. Do NOT mark a phase PASS until every
   missing package/binary has been installed. If an install loop exceeds
   its cap (5 iterations), roll back — do not register a known-broken
   tool. See Phase 3.75 below.

4. **Every script you write or execute MUST define its own constants at
   the top.** Applies to: persistent workers / servers / env scripts saved
   to `tools_user/`, AND short-lived scripts fed to python/bash via stdin,
   heredoc, subprocess, or a tmp `.sh`/`.py` file. Recurring failure mode:
   STCoscientist generates a script that references `TOOL_ID`, `MODULE`, `ENV_NAME`,
   `VENDOR_PATH`, etc. without defining them, producing
   `NameError: name 'TOOL_ID' is not defined` or
   `KeyError: 'tool_id_worker'` partway through. This wastes at minimum
   one full regen cycle per occurrence.

   **The rule — no exceptions, every generated script:**
   - The first executable lines (after `#!` shebang + imports) MUST be
     concrete literal assignments for every identifier the script will
     reference later. No `string.Template`, no `$`/`{}` placeholders
     surviving into the executed script — substitute everything BEFORE
     the script runs.
   - Before executing or saving, `ast.parse(script_text)` the result
     and grep for unresolved placeholder patterns (`${`, `{{`, `__.*__`,
     literal `$tool_id`). Refuse to run a script whose placeholders
     still appear.

   **Canonical pattern for any generated script (ad-hoc or saved):**
   ```python
   tool_id   = "banksy"          # <- resolved in STCoscientist's working scope
   module    = "banksy"
   env_name  = f"user_{tool_id}"
   vendor    = str(TOOLS_USER_DIR / f"vendor_{tool_id}") if install_mode == "vendor" else ""

   # Build the script with concrete literals baked in as the first lines:
   script = (
       f"TOOL_ID = {tool_id!r}\n"
       f"MODULE = {module!r}\n"
       f"ENV_NAME = {env_name!r}\n"
       f"VENDOR_PATH = {vendor!r}\n"
       + TEMPLATE_BODY                 # body uses the names above
   )

   import ast
   ast.parse(script)                   # fail fast if malformed
   for bad in ("${", "{{", "__TOOL_ID__", "$tool_id"):
       assert bad not in script, f"Unresolved placeholder: {bad!r}"

   subprocess.run(["python", "-c", script], check=True, ...)
   ```

   This rule supersedes the Phase 3 "curly-brace collision" warning for
   worker/server code (it still applies there), and ALSO extends to the
   heredoc / tmp-file executor scripts that wrap environment operations
   (conda env create, pip install dry-run, worker smoke test, etc.).
   If a script of any kind uses a bare identifier, define it at the top.

4. **README is the source of truth for install commands.** Always prefer
   README-stated install invocations (`pip install X`, `devtools::install_github`,
   `conda install -c channel`, `Rscript -e 'remotes::install_github(...)'`,
   shell scripts in the repo) over heuristic guesses. Read the README FIRST,
   extract every install command it contains, and try them in the order
   the README lists them before falling through to the canonical ladder.
   Balance: don't spend more than one attempt per README command — if
   the first try fails, don't keep hammering the same command; move down
   the canonical install ladder.

   **Sub-rule — language is determined in Phase 1, never revisited after
   Phase 2 failures** (P21). The repo's language is determined ONCE in
   Phase 1 by inspecting, in this order:
     a. Presence of `DESCRIPTION` (R) / `setup.py` / `setup.cfg` /
        `pyproject.toml` (Python) / `Cargo.toml` (Rust) / `package.json`
        (JS) / `Makefile` + `.sh` only (CLI wrapper).
     b. README install commands (`pip install` → Python; `install.packages`
        / `install_github` / `BiocManager::install` → R; etc.).
   Once the language is decided, Phase 2 install failures DO NOT justify
   re-classifying the language. If a Python `pip install` fails, the
   fallback is Python-vendor, NOT "try it as R". Specifically:
   - Python install failed → Python vendor-only → vendor-path worker
   - R install failed → R diagnose_r_load → R vendor (library with lib.loc)
   - Never cross over unless the repo truly has both setup.py AND
     DESCRIPTION (rare hybrid case; even then, stay on the path Phase 1
     determined was primary).

   Past run observed: spadecon (Python, setup.py) failed `pip install`,
   STCoscientist then "chose Alt B: robust R install with hybrid env", which was
   a misidentification. Correct path was vendor-only Python fallback.

5. **The `language` field in `install_log.json` MUST reflect the WORKER
   file extension, not the wrapped package:**
   - `*_worker.py`                → `language="python"`
   - `*_worker.R`                 → `language="R"`
   No `.r` or shell worker: health check and trash know only these two; a
   CLI binary gets the Python CLI-wrapper worker.
   If the wrapped tool is R but the worker is Python (rpy2/Rscript), set
   `language="python"` and `wrapped_language="R"`: the checks then load
   `module` with `Rscript library()` and run the `.py` worker.

6. **Self-capture issues you hit.** Whenever you resolve a non-obvious
   install/runtime issue during a creation, append an entry to
   `tools_user/.knowledge/{tool_id}/creation_issues.json` of the shape
   `{symptom, root_cause, fix, tool_id, phase}`. These entries are
   tool-agnostic lessons future creations will read on startup.

7. **Never print a PASS checklist before the on-disk artifacts exist.**
   (P17 hallucination guard.) Before emitting ANY verification checklist
   that includes a PASS line, re-read `install_log.json` and assert:
   ```python
   entries = json.loads(INSTALL_LOG.read_text())
   assert any(e.get("tool_id") == tool_id for e in entries), \
       f"VERIFICATION ABORT: {tool_id!r} not in install_log.json"
   assert ((TOOLS_USER_DIR / f"{tool_id}_worker.py").exists() or
           (TOOLS_USER_DIR / f"{tool_id}_worker.R").exists()), \
       f"VERIFICATION ABORT: {tool_id} worker file missing"
   ```
   If either assertion fails, you have NOT created the tool. Print the
   rollback/abort message, NOT the PASS checklist.

8. **`module` in install_log.json is the CANONICAL tool identifier, not
   a worker utility import.** (P18 fix.) Never set `module` to `yaml`,
   `json`, `os`, `argparse`, `pathlib`, or any stdlib/config-parsing
   library. The rule:
   - Pure Python tool: `module` = the wrapped package's top-level import
     (`scanpy`, `cellpose`, `gaston`, etc.)
   - Pure R tool: `module` = R package name (`Seurat`, `Giotto`, ...)
   - Hybrid (Python worker calling R via rpy2/Rscript): `module` = R
     package name, `language = "python"` (worker ext), `wrapped_language
     = "R"`
   - CLI wrapper / vendor-only: `module = null`, add `vendor_path` or
     `cli_binary` instead.

   If your `verified_module` discovery returns yaml/json/os — that
   means the worker has imports beyond the tool's own, and you need to
   look up the actual tool module from the repo's README /
   pyproject.toml / DESCRIPTION, not from your worker's imports.

9. **HARD NAMING CONVENTION — ZERO ambiguity, ZERO guessing** (P22 standardization).

   Every identifier derived from a tool MUST follow this exact pattern.
   No creative deviations. No verb-guessing ("cluster" vs "run" vs "fit").
   STCoscientist caller will always be able to compute these mechanically from the
   GitHub URL:

   | Identifier | Formula | Example (banksy) |
   |---|---|---|
   | `tool_id` | lowercased repo-name, `[^a-z0-9] → _`, strip leading `_`/digits | `banksy` |
   | conda env | `user_{tool_id}` | `user_banksy` |
   | worker file | `{tool_id}_worker.{py,R}` (ext = language) | `banksy_worker.py` |
   | MCP server file | `{tool_id}_mcp_server.py` | `banksy_mcp_server.py` |
   | **`function_name` = `spatialomicsgym_name`** | **ALWAYS `{tool_id}_run`** | `banksy_run` |
   | vendor dir (if used) | `vendor_{tool_id}/` | `vendor_banksy/` |
   | env export | `{tool_id}_env.yaml` | `banksy_env.yaml` |
   | knowledge dir | `.knowledge/{tool_id}/` | `.knowledge/banksy/` |

   **`function_name` / `spatialomicsgym_name` is ALWAYS `{tool_id}_run`.** Not
   `{tool_id}_cluster`, not `{tool_id}_domains`, not `{tool_id}_fit`.
   Every semantic verb a reader might want (cluster, deconvolve,
   impute, …) is just the tool's job — the ONE callable function is
   named `{tool_id}_run`. Tools that expose multiple primary verbs
   should surface them via a `mode: str` argument inside the single
   `{tool_id}_run(...)` function, not as separate @mcp.tool functions.

   Consequences for STCoscientist:
   - The `@mcp.tool()` decorated function in `{tool_id}_mcp_server.py`
     MUST be literally `def {tool_id}_run(...)`.
   - `install_log.json.function_name` MUST be `{tool_id}_run`.
   - `mcp_config_user.yaml` `tools[].spatialomicsgym_name` MUST be `{tool_id}_run`.
   - The real_data prompt caller can always compute the correct name
     from the tool_id without consulting the config — no lookup
     surprises, no "Unknown tool" mismatches.

   This rule eliminates the entire class of spatialomicsgym_name/function_name/
   invocation-path mismatch failures observed previously (P13 "Unknown
   tool" symptom). If you emit a decorator like `def banksy_cluster`,
   that is a HARD RULE violation — rename to `def banksy_run`.

   Validate post-codegen (Phase 3.5 addition):
   ```python
   src = (TOOLS_USER_DIR / f"{tool_id}_mcp_server.py").read_text()
   expected = f"def {tool_id}_run("
   assert expected in src, (
       f"NAMING VIOLATION: {tool_id}_mcp_server.py must declare "
       f"{expected!r} — found: "
       f"{[ln.strip() for ln in src.split(chr(10)) if ln.strip().startswith('def ')]}"
   )
   ```

10. **Worker MUST write at least one non-empty output file before
    emitting status="ok"** (P23). A worker that returns `{"status":"ok"}`
    without writing any artifact looks successful to the MCP protocol
    but fails any real-data verification. The WorkerOutput wrapper
    already enforces this via `add_output_file(key, path)` — use it
    for EVERY file the worker produces, and fail fast (WorkerOutput.error)
    if the result object you're about to save is None/empty. The rule:
    - If core analysis succeeded → write the result(s) and call
      `out.add_output_file(...)` for each.
    - If core analysis returned nothing useful (e.g. clustering found
      zero clusters, deconv returned empty matrix) → `out.add_warning(...)`
      + still write at least a diagnostic text file documenting what was
      attempted and why it's empty. Never exit with zero files.

11. **No cross-language syntax in codegen — Python literals are
    Python-only** (P24). When generating code, STCoscientist sometimes mixes R
    literals into Python strings: `6L` (R long int), `TRUE`/`FALSE`
    (R boolean), `NULL` (R nil), `NA`, `c(1,2,3)` (R vector). These
    SyntaxError when the Python parser loads the generated file.
    Cross-check before writing:
    ```python
    import ast
    try:
        ast.parse(worker_source)
    except SyntaxError as e:
        raise SystemExit(f"P24 VIOLATION: generated Python has R-syntax leak: {e}")
    ```
    For R-worker files, the same check applies in reverse — Python
    keywords (`True`/`False`/`None`) leaking into R source. In R,
    validate with: `Rscript -e 'parse(text=readLines("<file>"))'`.

12. **Every `module.attribute` chain in the worker MUST resolve via
    getattr-chain inside the user env** (P26 — API surface). Phase 3.6
    enforces this before Phase 4 registration: an AST walker collects
    every `<imported_root>.attr1.attr2…` chain in the worker and runs a
    `python -c "import M; getattr(M, 'attr1'); getattr(_, 'attr2'); …"`
    inside `user_{tool_id}` for each. Any unresolved chain rolls back
    the tool. Past failures this catches: `scanpy.tl.nmf` (does not
    exist), hallucinated `liana.foo`, renamed `nichecompass.bar`.
    Opt-out: append `# noqa: P26` to a worker line for legitimately
    dynamic getattr (plugin systems). Use sparingly.

## Creation Pipeline Invariants (READ FIRST)

A tool-creation run is a state machine with **seven phases**. Each phase has a hard
precondition it checks on entry and a hard post-condition it verifies on exit.
If *any* precondition is not satisfied, the phase **refuses to run** — it does not
guess. If any post-condition fails, the phase **rolls back its own writes** and
does not hand control to the next phase.

| Phase | Pre-condition (must hold before) | Post-condition (must hold after) | Rollback scope on fail |
|-------|----------------------------------|----------------------------------|------------------------|
| 0 Pre-flight | conda + pip + git present, disk > 20 GB, `user_envs` < cap | all assertions passed | none (read-only) |
| 1 Discover | Phase 0 OK, GitHub URL reachable | `install_plan.json` written with strategy in {pip, cran, bioc, env.yml, git+pip, install_github, install.sh, vendor} | none |
| 2 Create Env | strategy chosen, `user_{tool_id}` env does NOT exist | `user_{tool_id}` env exists AND target `module` imports OR vendor_path populated | `conda remove -n user_{tool_id} --all -y` |
| 3 Generate Code | Phase 2 OK, API discovered, micro-test passed | `{tool_id}_worker.py` + `{tool_id}_mcp_server.py` parse via `ast.parse` AND contain no template placeholder strings | delete the two files |
| 3.5 Dep Audit | Phase 3 OK | every import in worker/server resolves in env | install missing; if impossible, unwind Phase 3 |
| 4 Register | Phase 3.5 OK, `verified_module` set | `install_log.json` has NEW entry with status=='active', source_url, module, tool_id; `mcp_config_user.yaml` has `user_{tool_id}` server | remove new entry + new config key |
| 5 Test | Phase 4 OK | Tests 1–4 all PASS (critical); Test 5 PASS or soft-fail | HARD GATE — see Phase 5 |
| 6 Activate | Phase 5 OK (or Test 5 soft-fail) | `.knowledge/{tool_id}/{install_plan,api_discovery,creation_log}.json` exist; env.yaml exported; final invariant check passes | remove knowledge dir only |

**Atomicity rule:** Anything a phase writes becomes committed only when the phase's
post-condition passes. If it fails, undo in reverse order until the pre-state of
that phase is restored.

**Idempotence rule:** Re-running a phase when its post-condition already holds
is a no-op. Re-running it when post-condition partially holds triggers cleanup
first, then re-execution.

**State taxonomy for partial failures:**
- `clean` — nothing written yet (Phase 1 incomplete)
- `env_only` — Phase 2 done, no files yet (Phase 3 incomplete)
- `files_only` — Phase 3 done, no registration (Phase 4 incomplete)
- `registered_untested` — Phase 4 done, no tests (Phase 5 incomplete)
- `active_degraded` — Tests passed but Test 5 soft-failed
- `active_healthy` — Everything green

The rollback function at the bottom of this doc handles all five transitional
states correctly — always call `rollback(tool_id)` on a fatal error, never try
to clean up manually.

## Phase Documents (this playbook is split)

The phases below live in their own documents so one can be loaded without the other five. This document is the one that always applies: the preconditions above, the HARD RULES, the pipeline invariants, Phase 0, the rollback procedure and the safety rules govern every phase, wherever it is written down.

- **`add_new_mcp_tool_discover.md`** - Phase 1 of creating a new MCP tool from a GitHub or paper link: identify the package behind the link, read its real API rather than guessing it, decide which of its functions become MCP tools, and harvest its tutorials for worked parameter values.
- **`add_new_mcp_tool_environment.md`** - Phase 2 of creating a new MCP tool: build the isolated conda environment its worker will run in, install the package and the dependencies it actually needs, and prove the import works before a single line of worker code is written..
- **`add_new_mcp_tool_generate_code.md`** - Phase 3 of creating a new MCP tool: write the worker script and the MCP server that wraps it, following the argument, path, schema and output-contract rules the generated code has to satisfy to be callable by the agent..
- **`add_new_mcp_tool_dependency_audit.md`** - The three mandatory checks between writing a new MCP tool's code and registering it: audit the dependencies the code actually imports, close the environment against the runtime fixtures it will meet, and block on any attribute the worker uses but never defines..
- **`add_new_mcp_tool_register_and_test.md`** - Phases 4 and 5 of creating a new MCP tool: write the config and registry entries atomically, then run the whole mandatory test ladder -- import, schema, smoke and a real data run -- every rung of which must be executed rather than described..
- **`add_new_mcp_tool_activate.md`** - Phase 6 of creating a new MCP tool: activate it in the live catalogue, then run the final invariant verification that has to pass before the creation may be called successful..

Run them in that order. If the document for the phase you have reached is not in your context, ask for it by name rather than reconstructing the phase from memory -- each one carries mandatory checks whose omission is exactly what Phase 0 and the rollback procedure exist to catch.

## Phase 0: Pre-flight Checks (MANDATORY; abort creation on any FAIL)

Verify these BEFORE you touch any state:

```python
import subprocess, os, shutil, json, socket, urllib.request
from pathlib import Path
# Same value the setup fence bound; re-derived here so this block also runs standalone.
from spatialomicsgym.setup.constants import conda_envs_root
CONDA_ENVS_DIR = Path(conda_envs_root())

preflight = {"ok": True, "checks": []}

def check(name, ok, detail=""):
    preflight["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    if not ok:
        preflight["ok"] = False

# 1. Config flag
from spatialomicsgym.config import default_config
check("tool_creation_enabled", default_config.tool_creation_enabled,
      "set default_config.tool_creation_enabled = True before calling agent.go")

# 2. Binaries on PATH
for bin_name in ("conda", "pip", "git", "Rscript"):
    r = subprocess.run(["which", bin_name], capture_output=True, text=True, timeout=5)
    # Rscript is optional — only required for R tools; we still record presence
    check(f"bin_{bin_name}", bool(r.stdout.strip()) or bin_name == "Rscript",
          r.stdout.strip() or "not on PATH")

# 3. Disk — need 20 GB free for conda env (R envs can be 5–8 GB on their own)
#    Measure the mount, not the envs dir, which may not exist yet on a fresh install.
free_gb = shutil.disk_usage(CONDA_ENVS_DIR.parent).free / (1024**3)
check("disk_ge_20gb", free_gb > 20, f"{free_gb:.1f} GB free")

# 4. User-env cap not exceeded
envs_now = os.listdir(CONDA_ENVS_DIR) if CONDA_ENVS_DIR.is_dir() else []
user_envs = [d for d in envs_now if d.startswith('user_')]
check("user_env_cap", len(user_envs) < default_config.max_user_envs,
      f"{len(user_envs)} / {default_config.max_user_envs} user envs")

# 5. Network reachability — GitHub (primary), PyPI (install), conda-forge (R tools)
for host in ("github.com", "pypi.org", "conda.anaconda.org"):
    try:
        socket.create_connection((host, 443), timeout=5).close()
        check(f"net_{host}", True)
    except OSError as e:
        check(f"net_{host}", False, f"{type(e).__name__}: {e}")

# 6. install_log.json parseable (or empty/missing)
log_path = INSTALL_LOG
if log_path.exists():
    try:
        json.loads(log_path.read_text())
        check("install_log_parse", True)
    except json.JSONDecodeError as e:
        check("install_log_parse", False, f"corrupt: {e}")
else:
    check("install_log_parse", True, "missing (OK — will create)")

# 7. mcp_config_user.yaml parseable (or empty/missing)
cfg_path = mcp_config_user_path()
if cfg_path.exists():
    try:
        import yaml
        yaml.safe_load(cfg_path.read_text())
        check("mcp_user_cfg_parse", True)
    except Exception as e:
        check("mcp_user_cfg_parse", False, f"corrupt: {e}")
else:
    check("mcp_user_cfg_parse", True, "missing (OK — will create)")

# 8. tool_id (HARD RULE 9, from the URL) not taken. Env, files and wiring are named by the id
#    alone, so a row under ANY account is a collision -- and rollback() would delete its files.
from tools_user.trash_manager import check_tool_id_available
try:
    avail = check_tool_id_available(tool_id)
except Exception as e:   # TrashSafetyError: not a valid tool_id
    avail = {"available": False, "reason": str(e)}
check("tool_id_available", avail["available"], avail.get("reason", ""))

if not preflight["ok"]:
    print("\n".join(f"  [{'OK' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}"
                    for c in preflight["checks"]))
    raise SystemExit("Pre-flight FAILED — refuse to proceed.")
print("Pre-flight OK.")
```

Any FAIL ⇒ do NOT proceed. If the user forgot `tool_creation_enabled`,
report it clearly — do not attempt to set it yourself.

## On Any Failure — Rollback

The rollback below is order-sensitive: remove the *visible* state (config + log entry)
FIRST so that an observer querying `list_user_tools()` mid-rollback doesn't see a stale
tool; remove the *heavy* state (conda env, vendor dir) LAST so that an interrupted
rollback still leaves the next run a clean slate to work with.

```python
import subprocess, os, json, shutil, tempfile
from pathlib import Path
import yaml
from tools_user.knowledge_manager import current_owner


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def locate_source_tree(tool_id: str, package_name: str | None) -> Path | None:
    """Best-effort locator for a vendor-source fallback.

    Returns the directory that holds the package source tree inside the tool's
    conda env — or the vendor_<tool_id>/ dir Strategy-8 populated. The worker can
    `sys.path.insert(0, <dir>)` + `importlib.import_module(<pkg>)` to import.
    Tool-agnostic: works for any Python or R package whose sources are on disk.
    """
    import subprocess as _sp, re as _re
    env_name = f"user_{tool_id}"

    # 1. Preferred — the Strategy-8 vendor dir. Same spelling Strategy 8 creates and the
    #    rollback removes; `vendor/{tool_id}` is a directory nothing in this playbook writes.
    vend = TOOLS_USER_DIR / f"vendor_{tool_id}"
    if vend.exists() and any(vend.iterdir()):
        # If it contains a single child dir, descend into it (pip-style layout).
        kids = [p for p in vend.iterdir() if p.is_dir()]
        return kids[0] if len(kids) == 1 else vend

    # 2. Python — ask pip where the package got installed.
    if package_name:
        try:
            r = _sp.run(["conda", "run", "-n", env_name, "pip", "show", package_name],
                        capture_output=True, text=True, timeout=60)
            m = _re.search(r"^Location:\s*(.+)$", r.stdout, _re.M)
            if m:
                return Path(m.group(1).strip())
        except Exception:
            pass

    # 3. R — glob the first .libPaths() entry for a matching dir.
    try:
        r = _sp.run(["conda", "run", "-n", env_name, "Rscript", "-e",
                     'cat(.libPaths()[1])'],
                    capture_output=True, text=True, timeout=30)
        libp = Path(r.stdout.strip())
        if package_name:
            cand = libp / package_name
            if cand.exists():
                return cand
        if libp.exists():
            return libp
    except Exception:
        pass

    return None


def retry_import_from_vendor(vendor_dir: Path, module: str | None = None,
                             *, env_name: str = "") -> bool:
    """Attempt to import a module with `sys.path.insert(0, vendor_dir)` prepended.

    For R, `module` should be the package name and a `library()` is attempted
    with .libPaths() pointed at vendor_dir. Returns True on success.
    """
    import subprocess as _sp
    if module is None:
        return False

    # Heuristic: treat as R if vendor_dir has a DESCRIPTION or NAMESPACE file.
    is_r = (vendor_dir / "DESCRIPTION").exists() or (vendor_dir / "NAMESPACE").exists()
    if is_r:
        cmd = ["conda", "run", "-n", env_name, "Rscript", "-e",
               f'.libPaths(c("{vendor_dir.parent}", .libPaths())); '
               f'res <- try(suppressMessages(library({module})), silent=TRUE); '
               f'if (inherits(res, "try-error")) quit(status=1) else cat("LOAD_OK")']
        r = _sp.run(cmd, capture_output=True, text=True, timeout=120)
        return "LOAD_OK" in r.stdout and r.returncode == 0

    # Python path
    cmd = ["conda", "run", "-n", env_name, "python", "-c",
           f"import sys; sys.path.insert(0, {str(vendor_dir)!r}); "
           f"import {module}; print('IMPORT_OK')"]
    r = _sp.run(cmd, capture_output=True, text=True, timeout=60)
    return "IMPORT_OK" in r.stdout and r.returncode == 0


def locate_r_library(tool_id: str, r_pkg_name: str | None) -> Path | None:
    """R analog of locate_source_tree — find the directory holding the R
    package source, so the worker can `library(PKG, lib.loc=<vendor>)`.
    """
    import subprocess as _sp
    env_name = f"user_{tool_id}"
    # 1. Strategy-8 vendor dir (if we extracted into tools_user/vendor_<tool_id>/)
    vend = TOOLS_USER_DIR / f"vendor_{tool_id}"
    if vend.exists() and any(vend.iterdir()):
        kids = [p for p in vend.iterdir() if p.is_dir()]
        return kids[0] if len(kids) == 1 else vend
    # 2. The env's .libPaths()[1] directly
    try:
        r = _sp.run(["conda", "run", "-n", env_name, "Rscript", "-e",
                     'cat(.libPaths()[1])'],
                    capture_output=True, text=True, timeout=30)
        libp = Path(r.stdout.strip())
        if r_pkg_name:
            cand = libp / r_pkg_name
            if cand.exists():
                return cand
        if libp.exists():
            return libp
    except Exception:
        pass
    return None


def retry_library_from_vendor(vendor_dir: Path, r_pkg_name: str,
                              *, env_name: str = "") -> bool:
    """R analog of retry_import_from_vendor.

    Runs `library(PKG, lib.loc=<vendor_dir.parent>)` so R loads from the
    raw package source location even if the env's main .libPaths can't
    resolve it.
    """
    import subprocess as _sp
    cmd = ["conda", "run", "-n", env_name, "Rscript", "-e",
           f'.libPaths(c("{vendor_dir.parent}", .libPaths())); '
           f'res <- try(suppressMessages(library({r_pkg_name})), silent=TRUE); '
           f'if (inherits(res, "try-error")) quit(status=1) else cat("LOAD_OK")']
    r = _sp.run(cmd, capture_output=True, text=True, timeout=120)
    return "LOAD_OK" in r.stdout and r.returncode == 0


def parse_requirements(root: Path) -> list[str]:
    """Best-effort extraction of a package's declared dependencies.

    Looks at (in order): requirements.txt, setup.cfg [options]/install_requires,
    setup.py install_requires (regex fallback), pyproject.toml dependencies,
    DESCRIPTION Imports/Depends (for R). Returns a flat list of package specs.
    """
    import re as _re
    specs: list[str] = []
    # requirements.txt — simplest
    rf = root / "requirements.txt"
    if rf.exists():
        for ln in rf.read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                specs.append(ln.split(";")[0].strip())
    # setup.cfg
    sc = root / "setup.cfg"
    if sc.exists():
        txt = sc.read_text()
        m = _re.search(r"install_requires\s*=\s*([\s\S]*?)(?:\n\s*\w+\s*=|\Z)", txt)
        if m:
            for ln in m.group(1).splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    specs.append(ln)
    # setup.py
    sp = root / "setup.py"
    if sp.exists():
        txt = sp.read_text()
        m = _re.search(r"install_requires\s*=\s*\[([\s\S]*?)\]", txt)
        if m:
            for qstr in _re.findall(r"['\"]([^'\"]+)['\"]", m.group(1)):
                specs.append(qstr)
    # pyproject.toml
    pp = root / "pyproject.toml"
    if pp.exists():
        txt = pp.read_text()
        m = _re.search(r"dependencies\s*=\s*\[([\s\S]*?)\]", txt)
        if m:
            for qstr in _re.findall(r"['\"]([^'\"]+)['\"]", m.group(1)):
                specs.append(qstr)
    # DESCRIPTION (R)
    desc = root / "DESCRIPTION"
    if desc.exists():
        txt = desc.read_text()
        for fld in ("Imports", "Depends"):
            m = _re.search(rf"^{fld}:\s*([\s\S]*?)(?:\n[A-Z]\w+:|\Z)", txt, _re.M)
            if m:
                for part in m.group(1).split(","):
                    pkg = _re.sub(r"\s*\([^)]*\)", "", part).strip()
                    if pkg and pkg.lower() != "r":
                        specs.append(pkg)
    # De-duplicate, keep order
    seen: set[str] = set()
    out: list[str] = []
    for s in specs:
        key = _re.split(r"[<>=!~\s]", s)[0].lower()
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def rollback(tool_id: str, *, keep_knowledge: bool = True, owner: str | None = None) -> dict:
    """Remove every trace of a partial or failed creation.

    Safe to call on any transitional state (clean, env_only, files_only,
    registered_untested, active_degraded). Returns a dict of per-step status
    so the caller can log it into .knowledge/{tool_id}/rollback.json.

    `owner` defaults to whoever's turn is rolling back, which is whoever's creation this was.
    It scopes the install_log removal in A2: the registry is keyed on (owner, tool_id), so
    dropping every row whose id matches would delete ANOTHER account's working tool as part of
    cleaning up this one -- and there is nothing to restore it from.
    """
    env_name = f"user_{tool_id}"
    owner = current_owner() if owner is None else owner
    steps: dict[str, str] = {}
    # Files, env, vendor/trash dirs and wiring are named by the id alone. If ANOTHER account has a
    # row for this id they are its tool: remove only this creation's row (A2), keep the rest.
    # An unreadable registry counts as foreign -- a kept orphan is recoverable, a deleted tool is not.
    try:
        _rows = json.loads(INSTALL_LOG.read_text()) if INSTALL_LOG.exists() else []
        foreign = any(e.get("tool_id") == tool_id and str(e.get("owner") or "") != owner for e in _rows)
    except Exception:
        foreign = True
    _kept = "kept (another account's tool shares this id)"

    # ---- Step A: Visible state ----
    # STCoscientist. mcp_config_user.yaml — remove server entry atomically
    cfg_path = mcp_config_user_path()
    if foreign:
        steps["mcp_config"] = _kept
    elif cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            servers = cfg.get("mcp_servers") or {}
            removed = servers.pop(f"user_{tool_id}", None) is not None
            if removed:
                cfg["mcp_servers"] = servers
            # A block the agent appended at column 0 instead of nesting it is rescued by
            # mcp_config_merger._recover_top_level_servers, so leaving it here would keep the
            # merger advertising the tool whose files and env this same call is deleting. The
            # merger's own predicate (a command and a tools list) keeps bookkeeping keys safe.
            stray = cfg.get(f"user_{tool_id}")
            if isinstance(stray, dict) and "command" in stray and isinstance(stray.get("tools"), list):
                del cfg[f"user_{tool_id}"]
                removed = True
            if removed:
                _write_atomic(cfg_path, yaml.dump(cfg, default_flow_style=False, sort_keys=False))
                steps["mcp_config"] = "removed"
            else:
                steps["mcp_config"] = "not_present"
        except Exception as e:
            steps["mcp_config"] = f"error: {e}"
    else:
        steps["mcp_config"] = "no_file"

    # A2. install_log.json — remove entry atomically
    log_path = INSTALL_LOG
    if log_path.exists():
        try:
            entries = json.loads(log_path.read_text())
            before = len(entries)
            entries = [
                e for e in entries
                if not (e.get("tool_id") == tool_id and str(e.get("owner") or "") == owner)
            ]
            if len(entries) < before:
                _write_atomic(log_path, json.dumps(entries, indent=2))
                steps["install_log"] = f"removed ({before - len(entries)} entries)"
            else:
                steps["install_log"] = "not_present"
        except Exception as e:
            steps["install_log"] = f"error: {e}"
    else:
        steps["install_log"] = "no_file"

    # ---- Step B: File state ----
    # B1. Generated code files
    removed_files = []
    for fname in () if foreign else (f"{tool_id}_worker.py", f"{tool_id}_worker.R",
                                     f"{tool_id}_mcp_server.py", f"{tool_id}_env.yaml"):
        # TOOLS_USER_DIR, not Path("tools_user"): the latter resolves against the process CWD, and
        # nothing in the package chdirs -- run from a data directory this removes nothing and the
        # step still reports "removed 0". Steps B2 and B4 below already anchor the same way.
        p = TOOLS_USER_DIR / fname
        if p.exists():
            try:
                p.unlink()
                removed_files.append(fname)
            except Exception as e:
                steps["files"] = f"error removing {fname}: {e}"
                break
    else:
        steps["files"] = _kept if foreign else f"removed {len(removed_files)}"

    # B2. Vendor directory (if any)
    vendor_dir = TOOLS_USER_DIR / f"vendor_{tool_id}"
    if foreign:
        steps["vendor_dir"] = _kept
    elif vendor_dir.exists():
        try:
            shutil.rmtree(vendor_dir)
            steps["vendor_dir"] = "removed"
        except Exception as e:
            steps["vendor_dir"] = f"error: {e}"
    else:
        steps["vendor_dir"] = "no_dir"

    # B3. Knowledge dir — keep by default (useful for post-mortem)
    know_dir = (KNOWLEDGE_DIR / tool_id)
    if know_dir.exists():
        if keep_knowledge or foreign:
            steps["knowledge"] = _kept if foreign else "kept (for post-mortem)"
        else:
            try:
                shutil.rmtree(know_dir)
                steps["knowledge"] = "removed"
            except Exception as e:
                steps["knowledge"] = f"error: {e}"
    else:
        steps["knowledge"] = "no_dir"

    # B4. Trash dir (if creation was restarting from a trashed state)
    trash_dir = TOOLS_USER_DIR / f".trash/{tool_id}"
    if foreign:
        steps["trash_dir"] = _kept
    elif trash_dir.exists():
        try:
            shutil.rmtree(trash_dir)
            steps["trash_dir"] = "removed"
        except Exception as e:
            steps["trash_dir"] = f"error: {e}"
    else:
        steps["trash_dir"] = "no_dir"

    # ---- Step C: Heavy state ----
    # C1. Conda env — can take 30-120 s for R envs; do this last so earlier
    #     cleanup surfaces even if this is slow.
    r = subprocess.run(["conda", "env", "list", "--json"],
                       capture_output=True, text=True, timeout=30)
    env_exists = False
    try:
        envs = json.loads(r.stdout).get("envs", [])
        env_exists = any(os.path.basename(p) == env_name for p in envs)
    except Exception:
        pass
    if foreign:
        steps["conda_env"] = _kept
    elif env_exists:
        r2 = subprocess.run(["conda", "remove", "-n", env_name, "--all", "-y"],
                            capture_output=True, text=True, timeout=600)
        steps["conda_env"] = "removed" if r2.returncode == 0 else f"failed: {r2.stderr[:200]}"
    else:
        steps["conda_env"] = "not_present"

    # Persist rollback log
    if know_dir.exists() and keep_knowledge:
        try:
            _write_atomic(know_dir / "rollback.json",
                          json.dumps({"tool_id": tool_id, "steps": steps}, indent=2))
        except Exception:
            pass

    # === MEMORY WRITE on rollback ===
    # MODE-GATED — MANDATORY when memory_enabled=True; SKIPPED when False.
    # Records the failure so future creations of THIS source_url can avoid
    # the same dead-end strategy. source_url is read from .knowledge/{tid}/
    # install_plan.json (Fix-C4 ensures Phase 1 writes that field).
    from spatialomicsgym.config import default_config as _cfg_rb
    if getattr(_cfg_rb, "memory_enabled", False):
        from tools_user.memory_manager import MemoryManager
        from datetime import datetime as _dt_rb
        from pathlib import Path as _P_rb
        import json as _json_rb
        _src_url = None
        _kp = (KNOWLEDGE_DIR / tool_id / "install_plan.json")
        if _kp.exists():
            try:
                _ip = json.loads(_kp.read_text())
                _src_url = _ip.get("source_url") or _ip.get("github_url")
            except Exception:
                pass
        if _src_url:
            mm = MemoryManager.get()
            mm.append_attempt(_src_url, {
                "outcome": "rolled_back",
                "tool_id": tool_id,
                "finished": _dt_rb.now().isoformat(),
                "rollback_steps": steps,
            })
            # MANDATORY MARKER — driver verifies rollback memory fired.
            _marker_rb = mm.root / ".last_op"
            _marker_rb.parent.mkdir(parents=True, exist_ok=True)
            _marker_rb.write_text(_json_rb.dumps({
                "ts": _dt_rb.now().isoformat(),
                "tool_id": tool_id,
                "source_url": _src_url,
                "op": "rollback_write",
            }))
            print(f"[memory] recorded rollback for {_src_url}")
        else:
            # source_url unrecoverable — write a "no_url" marker so driver can
            # distinguish "didn't try" from "tried but no source_url".
            _marker_rb = MemoryManager.get().root / ".last_op"
            _marker_rb.parent.mkdir(parents=True, exist_ok=True)
            _marker_rb.write_text(_json_rb.dumps({
                "ts": _dt_rb.now().isoformat(),
                "tool_id": tool_id,
                "source_url": None,
                "op": "rollback_write_no_url",
            }))
            print(f"[memory] rollback: no source_url in install_plan.json — marker still written")
    # === END MEMORY WRITE ===

    print(f"rollback({tool_id}) steps: {steps}")
    return {"tool_id": tool_id, "steps": steps,
            "ok": all(v.startswith(("removed", "not_present", "no_file", "no_dir", "kept"))
                      for v in steps.values())}
```

## Safety Rules
- NEVER modify files in `tools/` directory
- NEVER modify `MCP_server/mcp_config.yaml`
- NEVER install into any env in `spatialomicsgym.setup.constants.PROTECTED_ENVS` (base, spatialomicsgym_env, spatialomicsgym_e1, ...) -- only into `user_{tool_id}`
- NEVER use system pip — always use the new env's pip via `conda run`
- ALWAYS use Python `yaml.dump()` to write YAML files — NEVER shell echo/append
- ALWAYS use the atomic writers (`write_atomic` / `yaml_atomic` / `json_atomic`) from Phase 4
- ALWAYS use `WorkerOutput` from `worker_utils.py` for worker output
- ALWAYS validate merged config after writing user config
- ALWAYS run `final_invariant_check(tool_id, source_url)` at end of Phase 6
- On ANY failure after 3 retries → call `rollback(tool_id)`
- On ANY phase where post-condition in the Invariants table fails → call `rollback(tool_id)` — do NOT register a half-finished tool
