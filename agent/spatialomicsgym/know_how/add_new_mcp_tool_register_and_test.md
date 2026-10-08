# Adding New MCP Tools - Phases 4 and 5: Register and Test

## Metadata
- Authors: SpatialOmicsLab
- Version: 2.0
- Category: tool_creation

## Overview
Phases 4 and 5 of creating a new MCP tool: write the config and registry entries atomically, then run the whole mandatory test ladder -- import, schema, smoke and a real data run -- every rung of which must be executed rather than described.

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
from tools_user.knowledge_manager import INSTALL_LOG, KNOWLEDGE_DIR, mcp_config_user_path, TOOLS_USER_DIR, health_check_tools
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

## Phase 4: Register

### PRE-FLIGHT: install_log `module` must match the tool's runtime language

For R tools accessed via rpy2 from a Python worker, the install_log entry MUST NOT
set `module: "rpy2"` — rpy2 is the bridge, not the tool. `health_check_tools` looks up
this field to run `library(<module>)` inside R; if you wrote `rpy2` there, R raises
"there is no package called 'rpy2'" and the tool is reported BROKEN even though the
worker runs fine.

Rule:
- Pure Python tool → `module`: the Python import name that resolves (verified_module).
- Pure R tool → `module`: the R package name (`library(<module>)` must succeed).
- **Hybrid Python-worker + R-via-rpy2** → `module`: the R package name, `language: "python"` (the worker's, HARD RULE 5), `wrapped_language: "R"`. Set `wrapped_language = "R"` in your REPL too: `_verify` below, Phase 5 Test 1 and the final invariant check then load `module` with `Rscript library()` and run the `.py` worker.

Incorrect vs correct example for semla (R tool called from Python worker via rpy2):

```json
// WRONG — the checks will try `library(rpy2)` and fail
{"tool_id": "semla", "language": "python", "wrapped_language": "R", "module": "rpy2"}

// CORRECT — the checks run `library(semla)` in R, then the .py worker
{"tool_id": "semla", "language": "python", "wrapped_language": "R", "module": "semla"}
```

### PRE-FLIGHT: Verify the ENTRY-POINT FUNCTION actually exists (not just the module)

A common past failure: STCoscientist reads the README, picks `main_function = "run_Banksy"` for
example, then writes a worker that calls `banksy.run_Banksy(...)`. At creation time,
Phase 5 Tests 1-4 pass because `import banksy` works and the worker parses cleanly.
But Test 5 / real-data runs fail because `run_Banksy` isn't actually an attribute of
the installed version (renamed, nested, moved).

Before Phase 4 registration, verify the entry-point function is callable in the env:

```python
r = subprocess.run(["conda", "run", "-n", env_name, "python", "-c", f"""
import inspect, {module}
f = getattr({module}, {main_function!r}, None)
if f is None:
    # Also check nested modules (e.g. banksy.run_banksy.run_Banksy)
    for sub_name in dir({module}):
        sub = getattr({module}, sub_name, None)
        if hasattr(sub, {main_function!r}):
            f = getattr(sub, {main_function!r})
            break
print('CALLABLE' if callable(f) else 'MISSING')
"""], capture_output=True, text=True, timeout=30)
if "CALLABLE" not in r.stdout:
    # Enumerate viable alternatives
    r2 = subprocess.run(["conda", "run", "-n", env_name, "python", "-c", f"""
import inspect, {module}
cands = []
for n, v in inspect.getmembers({module}):
    if callable(v) and not n.startswith('_') and getattr(v, '__doc__', None):
        cands.append((n, (v.__doc__ or '').strip()[:80]))
print('\\n'.join(f'{{n}}\\t{{d}}' for n, d in cands[:10]))
"""], capture_output=True, text=True, timeout=30)
    print("Entry-point verification FAILED. Viable alternatives:")
    print(r2.stdout)
    # Pick the top candidate whose docstring matches the intended task_type
    # (re-running Phase 3's API discovery with a bounded alternative list)
```

Record the verified function name in install_log as `verified_function_names: [...]`
so downstream health_check and modify flows reference what actually exists.

### PRE-FLIGHT: Verify the `module` string is actually importable

A recurring past bug: the install_log gets `"module": "pybanksy"` but `import pybanksy` fails
inside the env, so every future health_check / modification reports the tool as BROKEN. Before
writing the install_log row, do one last check that the module name you're about to persist
is the *exact* string that imports without error:

```python
import subprocess, sys
env_name = f"user_{tool_id}"

# LANGUAGE-AWARE verification: Python tools use `python -c "import X"`;
# R tools use `Rscript -e 'library(X)'`. Previous version only tried Python
# and falsely rejected R tools (hdWGCNA, SpaNorm, semla) that load fine in R.
# `language` was derived in Phase 1 — it is one of {"python", "R", "hybrid", "cli"}.
# wrapped_language == "R": a Python worker driving an R package (HARD RULE 8) -- library() too.
wrapped_language = wrapped_language if "wrapped_language" in dir() else None

candidates = [module, module.replace("-", "_"), module.lower(), tool_id]
verified_module = None

def _verify(mod_name: str, lang: str) -> bool:
    if lang == "R" or wrapped_language == "R":
        # R tools use Rscript -e 'library(...)'
        r = subprocess.run(
            ["conda", "run", "-n", env_name, "Rscript", "-e",
             f'suppressPackageStartupMessages(library({mod_name})); cat("IMPORT_OK\\n")'],
            capture_output=True, text=True, timeout=60,
        )
    elif lang == "cli":
        # CLI tools: verify the binary runs with --help / --version
        r = subprocess.run(
            ["conda", "run", "-n", env_name, mod_name, "--version"],
            capture_output=True, text=True, timeout=30,
        )
        # CLI tools are "importable" if the binary returns any output
        return r.returncode in (0, 1, 2) and bool((r.stdout + r.stderr).strip())
    else:
        # python, hybrid (hybrid means a Python worker that shells to R; Python
        # side is still an `import` check)
        r = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c",
             f"import {mod_name}; print('IMPORT_OK')"],
            capture_output=True, text=True, timeout=30,
        )
    return "IMPORT_OK" in r.stdout

for m in candidates:
    if _verify(m, language):
        verified_module = m
        break

if verified_module is None:
    print(f"None of {candidates} verified in user_{tool_id} (language={language}). "
          "Do NOT register with a bogus module.")
    print("Either vendor the tool and record `module: null, vendor_path: '...'` instead, "
          "or rollback and re-diagnose Phase 2 install.")
    # Do not write an install_log entry with a module that does not verify.
else:
    print(f"Verified module for {tool_id} ({language}): {verified_module}")
```

**Critical:** once `verified_module` is set, write install_log with the matching
language field so `health_check_tools` later uses the correct verification path:

```python
from tools_user.knowledge_manager import current_owner

install_log_entry = {
    "tool_id": tool_id,
    # The registry is keyed on (owner, tool_id). Write it HERE as well as in Phase 6: Phase 6
    # starts from this row to keep `module`, and it looks the row up by the whole key -- an
    # unowned row is invisible to an owned lookup, so a Phase 4 that skips this throws away the
    # very module string it just verified.
    "owner": current_owner(),   # "" on a library/CLI run, which is a real key, not a missing one
    "tool_name": tool_name,
    "module": verified_module,
    "language": language,   # "python" | "R" | "hybrid" | "cli"
    "wrapped_language": wrapped_language,   # "R" when a Python worker drives an R package
    "source_url": github_url,
    "status": "active",
    # ... other fields
}
```

The install_log entry for this tool must set `module` to `verified_module` (or leave it
`null` and set `vendor_path` when the source is vendored rather than pip-installed).

### CRITICAL: Correct YAML format for mcp_config_user.yaml

**When creating the file for the first time OR appending to it, use Python to write proper YAML.
Do NOT use shell echo/append commands — they produce invalid YAML.**

**Anchor both paths on the repo, not on the cwd.** Nothing in the package chdirs, so the cwd is
wherever the user launched from. `knowledge_manager` already exports the two constants the health
check and the rollback path read — import those and writer and reader cannot drift apart:

```python
import yaml
from tools_user.knowledge_manager import mcp_config_user_path, TOOLS_USER_DIR

config_path = mcp_config_user_path()  # <repo>/MCP_server/mcp_config_user.yaml unless SOG_MCP_USER_CONFIG moves it

# Load existing or create new
if config_path.exists():
    existing = yaml.safe_load(config_path.read_text()) or {}
else:
    existing = {}

if "mcp_servers" not in existing:
    existing["mcp_servers"] = {}

# Add new tool entry
existing["mcp_servers"][f"user_{tool_id}"] = {
    "command": ["python", str(TOOLS_USER_DIR / f"{tool_id}_mcp_server.py")],
    "enabled": True,
    "description": f"[USER] {description}",
    "tools": [{
        "spatialomicsgym_name": function_name,
        "description": f"[USER TOOL] {detailed_description}",
        "parameters": {
            # parameter entries with type, required, description
        }
    }]
}

# Write complete file (not append!) -- atomically, re-read under the lock trash_manager holds
# while trash/restore/permanent_delete rewrite this file, so their edits are not lost.
import fcntl
from tools_user.trash_manager import _atomic_write_yaml
with open(TOOLS_USER_DIR / ".trash.lock", "w") as _lock:
    fcntl.flock(_lock, fcntl.LOCK_EX)   # blocks until a running trash operation finishes
    latest = (yaml.safe_load(config_path.read_text()) or {}) if config_path.exists() else {}
    latest.setdefault("mcp_servers", {})[f"user_{tool_id}"] = existing["mcp_servers"][f"user_{tool_id}"]
    _atomic_write_yaml(config_path, latest)
```

**NEVER do this (creates invalid YAML):**
```bash
# WRONG — do NOT use shell append
echo "mcp_servers: {}" > file.yaml
echo "  user_tool:" >> file.yaml  # This creates BROKEN YAML!
```

### Atomic config writes — use `write_atomic()`

Two commits to `mcp_config_user.yaml` + `install_log.json` happen during Phase 4. If STCoscientist
crashes between them, the two files diverge (tool in config but not in log, or vice
versa) and subsequent `list_user_tools()` / health-check reports become nonsense.

Always use the atomic-write helper:

```python
import os, tempfile, yaml, json, shutil
from pathlib import Path

def write_atomic(path: Path, content: str) -> None:
    """Write text to path via a same-directory tmp file + rename.

    Rationale: POSIX guarantees rename() within a directory is atomic, so a
    reader either sees the old file or the new file — never a half-written
    one. A crash mid-write leaves the old file intact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)   # atomic on POSIX
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

def yaml_atomic(path: Path, obj: dict) -> None:
    # Dump, round-trip through load to catch dump errors BEFORE the rename
    blob = yaml.dump(obj, default_flow_style=False, sort_keys=False)
    yaml.safe_load(blob)  # sanity parse — raises if malformed
    write_atomic(path, blob)

def json_atomic(path: Path, obj) -> None:
    blob = json.dumps(obj, indent=2, default=str)
    json.loads(blob)  # sanity parse
    write_atomic(path, blob)
```

### Install-log schema — enforce it on every write

Past runs saw install_log entries with missing fields (no `source_url`, no `module`,
no `status`). Downstream health_check then reports these as BROKEN because required
fields are absent. Enforce the schema with a validator before you write:

```python
REQUIRED_LOG_FIELDS = [
    "tool_id", "owner", "tool_name", "function_name", "task_type", "env_name",
    "source_url", "package", "module", "language", "status", "created_at", "files",
]

def validate_log_entry(entry: dict) -> list[str]:
    errors = []
    for f in REQUIRED_LOG_FIELDS:
        if f not in entry:
            errors.append(f"missing field: {f}")
    if entry.get("status") not in ("active", "trashed"):
        errors.append(f"bad status: {entry.get('status')!r}")
    import re
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,29}", entry.get("tool_id", "")):
        errors.append(f"tool_id regex fail: {entry.get('tool_id')!r}")
    if entry.get("source_url") and not entry["source_url"].startswith(("http://", "https://")):
        errors.append(f"bad source_url: {entry.get('source_url')!r}")
    return errors

issues = validate_log_entry(new_entry)
if issues:
    raise ValueError(f"install_log entry invalid — do NOT commit: {issues}")
```

### Validate after writing:
```python
import yaml
merged = yaml.safe_load(mcp_config_user_path().read_text())
assert f"user_{tool_id}" in merged.get("mcp_servers", {}), "Tool not in config!"
```

### Two-file consistency check (do this IMMEDIATELY after both writes)

```python
import yaml, json
cfg = yaml.safe_load(mcp_config_user_path().read_text()) or {}
log_entries = json.loads(INSTALL_LOG.read_text())

# Owner-scoped: on a two-account box another account's active row of the same name would
# satisfy this while THIS creation's row is missing, and the divergence would read as clean.
_owner = current_owner()
in_cfg = f"user_{tool_id}" in (cfg.get("mcp_servers") or {})
in_log = any(
    e["tool_id"] == tool_id and str(e.get("owner") or "") == _owner and e.get("status") == "active"
    for e in log_entries
)

assert in_cfg == in_log, (
    f"DIVERGENCE: cfg has entry={in_cfg} but log has entry={in_log}. "
    "Either both commits succeeded or both must be rolled back."
)
if not in_cfg:
    raise RuntimeError("Phase 4 registration failed silently — neither file committed.")
```

## Phase 5: Test (MANDATORY — you MUST execute ALL of these, not just describe them)

**CRITICAL: You MUST run every test below as actual code execution. Do NOT skip any test.
Do NOT say "I would test" or "you can test" — actually EXECUTE each test and report results.
If any test fails, fix the issue and re-test. Only proceed to Phase 6 when ALL tests pass.**

**HARD RULE — Phase 5 must CAPTURE the precise failing test's stderr.**

When Phase 5 aborts with "critical failure", the rollback message MUST
include: (a) WHICH test failed (1-5 mapping: import/syntax/dry-run/config/
real-data), (b) the test's exact command, and (c) the captured stderr
tail (≥ 1 KB). "Phase 5 failed critically" alone is a bug — STCoscientist and the
user both lose the diagnostic.

Persist the detail to `tools_user/.knowledge/{tool_id}/phase5_failure.json`
of shape:
```json
{
  "failing_test": "1-import" | "2-syntax" | "3-dryrun" | "4-config" | "5-realdata",
  "command": ["conda", "run", "-n", "user_X", "python", "-c", "..."],
  "stdout_tail": "...",
  "stderr_tail": "...",
  "returncode": 1,
  "time_iso": "2026-04-24T...",
  "prior_diagnostic_history": [...]
}
```

Every `subprocess.run(..., capture_output=True)` test call in Phase 5
must be followed by: if non-zero exit OR wrong output pattern, populate
this JSON BEFORE appending to `errors`. Without this, STCoscientist hits a
dead-end diagnostic (as observed on spanorm).

Execute this EXACT code block:

```python
import subprocess, json, ast
from pathlib import Path

tool_id = "{tool_id}"  # replace with actual
module = "{module}"     # replace with actual
language = "{language}"  # replace with actual: the WORKER's, "python" | "R" (HARD RULE 5)
wrapped_language = wrapped_language if "wrapped_language" in dir() else None   # "R": HARD RULE 8
env_name = f"user_{tool_id}"
worker = TOOLS_USER_DIR / f"{tool_id}_worker{'.R' if language == 'R' else '.py'}"
runner = "Rscript" if worker.suffix == ".R" else "python"
errors = []

# Test 1: Import test -- an R package (R worker, or a HARD RULE 8 hybrid) loads with library()
print("Test 1: Import test...")
load = (["Rscript", "-e", f"suppressPackageStartupMessages(library({module})); cat('OK\\n')"]
        if "R" in (language, wrapped_language) else ["python", "-c", f"import {module}; print('OK')"])
r = subprocess.run(["conda", "run", "-n", env_name, *load],
                   capture_output=True, text=True, timeout=60)
if "OK" in r.stdout:
    print("  PASS")
else:
    errors.append(f"Import failed: {r.stderr[:200]}")
    print(f"  FAIL: {r.stderr[:200]}")

# Test 2: Syntax check of generated files (an R worker is parsed by R)
print("Test 2: Syntax check...")
try:
    if runner == "Rscript":
        r = subprocess.run(["conda", "run", "-n", env_name, "Rscript", "-e",
                            f"invisible(parse(file={str(worker)!r}))"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise SyntaxError(r.stderr[-300:])
    else:
        ast.parse(worker.read_text())
    ast.parse((TOOLS_USER_DIR / f"{tool_id}_mcp_server.py").read_text())
    print("  PASS")
except (SyntaxError, OSError) as e:
    errors.append(f"Syntax error: {e}")
    print(f"  FAIL: {e}")

# Test 3: Worker dry-run with nonexistent input
print("Test 3: Worker dry-run...")
r = subprocess.run(["conda", "run", "-n", env_name, runner, str(worker),
                    "--input", "/nonexistent/file.h5ad",
                    "--output-dir", f"/tmp/test_{tool_id}"],
                   capture_output=True, text=True, timeout=60)
try:
    out = json.loads(r.stdout.strip().split('\n')[-1])
    if out.get("status") == "error":
        print(f"  PASS (got expected error: {out['error'][:80]})")
    else:
        errors.append(f"Expected error status, got: {out.get('status')}")
        print(f"  FAIL: expected error, got {out.get('status')}")
except (json.JSONDecodeError, IndexError) as e:
    errors.append(f"Worker output not valid JSON: {r.stdout[:200]}")
    print(f"  FAIL: not valid JSON: {r.stdout[:100]}")

# Test 4: Merged config validation
print("Test 4: Merged config validation...")
try:
    import yaml
    config = yaml.safe_load(mcp_config_user_path().read_text())
    if f"user_{tool_id}" in config.get("mcp_servers", {}):
        print("  PASS")
    else:
        errors.append(f"user_{tool_id} not in user config")
        print(f"  FAIL: tool not in config")
except Exception as e:
    errors.append(f"Config validation failed: {e}")
    print(f"  FAIL: {e}")

# Summary -- NOT the verdict: self-review, the vendor fallback and DEGRADED registration all need
# `errors` after this. The HARD GATE is the single raise `_EXEC_ERROR_RE` sees (not a printed "FAILED:").
for e in errors:
    print(f"  - {e}")
print(f"\n{len(errors)} failure(s) so far -- NOT ready: run Test 5, then the HARD GATE decides."
      if errors else "\nALL 4 TESTS PASSED -- run Test 5")
```

```python
# Test 5: REAL DATA test (CRITICAL — catches wrong core logic)
#
# TOOL-AGNOSTIC PATTERN: pick a small sample file whose FORMAT matches the
# tool's declared input parameter (h5ad / rds / bam / vcf / fastq / image /
# csv / pdb / dir). The candidate table below covers common bioinformatics
# formats; if your tool consumes something outside this table, generate a
# 10-row / 10KB synthetic sample on the fly with the appropriate library.
print("Test 5: Real data test...")
import os
TEST5_TIMEOUT_S = 300   # one worker run on the dataset; a timeout is a SOFT failure -- raise this

# Derive expected format from the FIRST input_* parameter declared in the
# MCP server file. You saved this during Phase 3 as `input_param_name`.
#
# STCoscientist must run in clean installs that ship without `benchmarks/` data.
# Each candidate list tries the benchmark fixture FIRST (when present), then the
# spatial-library catalogue, and finally synthesises one -- see the `if not
# test_data` branch below the catalogue lookup. Every path in this table is absent
# from a fresh clone, so on most hosts the synthesis rung is the one that runs.
FORMAT_SAMPLES = {
    "input_h5ad":  ["benchmarks/benchmark_data/mini/mini_visium_clustering.h5ad",
                    "benchmarks/benchmark_data/mini/mini_merfish_clustering.h5ad",
                    "data/spatialomicsgym_data/benchmark/mini_visium.h5ad"],
    "input_rds":   ["benchmarks/benchmark_data/mini/mini_seurat.rds",
                    "data/spatialomicsgym_data/benchmark/mini_seurat.rds"],
    "input_bam":   ["benchmarks/benchmark_data/mini/mini.bam"],
    "input_vcf":   ["benchmarks/benchmark_data/mini/mini.vcf"],
    "input_fastq": ["benchmarks/benchmark_data/mini/mini_R1.fastq.gz"],
    "input_image": ["benchmarks/benchmark_data/mini/mini.tif"],
    "input_csv":   ["benchmarks/benchmark_data/mini/mini.csv"],
    "input_pdb":   ["benchmarks/benchmark_data/mini/mini.pdb"],
    "input_path":  ["benchmarks/benchmark_data/mini/mini_visium_clustering.h5ad",
                    "data/spatialomicsgym_data/benchmark/mini_visium.h5ad"],  # CLI fallback
    "input_dir":   ["benchmarks/benchmark_data/mini/mini_dir/"],
}
# `input_param_name` comes from Phase 3's API discovery
candidates = FORMAT_SAMPLES.get(input_param_name, FORMAT_SAMPLES["input_path"])
test_data = None
for candidate in candidates:
    if os.path.exists(candidate):
        test_data = candidate
        break
# Fallback to the spatial-library catalogue. Resolve it the way the shipped code does
# (transcriptomics_skills._spatial_registry_candidates, spatial_library_worker.REGISTRY_PATH):
# the documented override first, then the in-repo snapshot, then the legacy locations. Naming
# only the legacy path finds nothing on most installs, which silently downgrades this whole
# real-data test to the synthetic branch below.
if not test_data:
    import glob
    import json
    reg_candidates = [
        os.environ.get("SOG_SPATIAL_LIBRARY_REGISTRY", "").strip(),
        str(TOOLS_USER_DIR.parent / "spatialomicsgym" / "data" / "spatial_library" / "registry.json"),
        "/workspace/spatial_library_backup/registry.json",
        "/workspace/data/spatial_library/registry.json",
    ]
    for reg_path in [p for p in reg_candidates if p and os.path.exists(p)]:
        for entry in json.loads(open(reg_path).read()):
            # No catalogued sample records an h5ad_path -- spatial_dir is the locator and the
            # .h5ad sits inside it. Read both with .get(): a registry is hand-writable through
            # SOG_SPATIAL_LIBRARY_REGISTRY, so a subscript here raises KeyError on a valid file.
            hit = entry.get("h5ad_path") or ""
            if not hit:
                sdir = entry.get("spatial_dir") or ""
                found = sorted(glob.glob(os.path.join(sdir, "*.h5ad"))) if sdir else []
                hit = found[0] if found else ""
            # Only accept a dataset that is actually on this disk. A catalogued path that is not
            # present must fall through to synthesis, not be handed to the worker as a real input.
            if hit and os.path.exists(hit):
                test_data = hit
                break
        if test_data:
            break

# Terminal rung: synthesise. Every path above names a file a fresh clone does not have
# (the mini fixtures are untracked and benchmarks/ is pruned from the wheel), and the
# catalogue is empty until the user points at a library -- so on any host but the one
# those fixtures were built on, both ladders come back empty. Without this rung the
# else-branch below prints a skip that the summary counts as a pass, and Test 5 reports
# success for a tool that was never handed a single byte of data.
synthetic_input = False
untested_reason = ""
# Kinds that need a real file of that exact format; an AnnData cannot stand in.
NOT_SYNTHESISABLE = ("input_rds", "input_bam", "input_vcf", "input_fastq",
                     "input_image", "input_csv", "input_pdb", "input_dir")
if not test_data and input_param_name not in NOT_SYNTHESISABLE:
    import anndata as ad
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n_rows, n_cols, n_genes = 6, 4, 60
    n_spots = n_rows * n_cols
    adata = ad.AnnData(
        X=rng.poisson(3.0, size=(n_spots, n_genes)).astype("float32"),
        obs=pd.DataFrame(index=[f"spot_{i}" for i in range(n_spots)]),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)]),
    )
    adata.obsm["spatial"] = np.array(
        [[float(i % n_cols), float(i // n_cols)] for i in range(n_spots)], dtype="float32"
    )
    test_data = f"/tmp/sog_test5_{tool_id}.h5ad"
    adata.write_h5ad(test_data)
    synthetic_input = True
    print(f"  no real dataset on this host; synthesised {test_data} "
          f"({n_spots} spots x {n_genes} genes, with obsm['spatial'])")

if test_data:
    test_out = f"/tmp/test_{tool_id}_real"
    os.makedirs(test_out, exist_ok=True)
    try:
        r = subprocess.run(
            ["conda", "run", "-n", env_name, runner, str(worker),
             "--input", test_data,
             "--output-dir", test_out],
            capture_output=True, text=True, timeout=TEST5_TIMEOUT_S)
        out = json.loads(r.stdout.strip().split('\n')[-1])
        if out.get("status") == "ok":
            # Verify output files actually exist and have content
            output_files = out.get("output_files", {})
            has_results = False
            for key, path in output_files.items():
                if os.path.exists(path) and os.path.getsize(path) > 100:
                    has_results = True
                    # A CLUSTERING h5ad must carry labels; deconvolution / SVG results live in
                    # obsm / var, so for them a non-empty file is the expectation.
                    if path.endswith(".h5ad") and task_type == "spatial_clustering":
                        import anndata as ad
                        result_adata = ad.read_h5ad(path)
                        obs_cols = list(result_adata.obs.columns)
                        n_unique = max((result_adata.obs[c].nunique() for c in obs_cols if result_adata.obs[c].dtype in ['object','category','int64','int32']), default=0)
                        if n_unique > 1:
                            print(f"  PASS (h5ad has {n_unique} unique labels in obs)")
                        else:
                            errors.append(f"Output h5ad has no cluster labels (obs cols: {obs_cols})")
                            print(f"  FAIL: no cluster labels in h5ad obs")
                    elif path.endswith(".csv"):
                        import pandas as pd
                        df = pd.read_csv(path)
                        if len(df) > 0 and df.shape[1] >= 2:
                            print(f"  PASS (csv has {len(df)} rows, {df.shape[1]} cols)")
                        else:
                            errors.append(f"Output CSV empty or has <2 columns")
                            print(f"  FAIL: CSV has {len(df)} rows, {df.shape[1]} cols")
            if not has_results:
                errors.append("No valid output files produced")
                print("  FAIL: no output files found")
            # Sanity check: if clustering tool, cluster count should be reasonable
            # (not 1, not equal to spot count — those indicate wrong output)
            if has_results and task_type == "spatial_clustering":
                for key, path in output_files.items():
                    if path.endswith(".h5ad") and os.path.exists(path):
                        result_adata = ad.read_h5ad(path)
                        for c in result_adata.obs.columns:
                            if result_adata.obs[c].dtype in ['object','category','int64','int32']:
                                n_unique = result_adata.obs[c].nunique()
                                n_spots = len(result_adata)
                                if n_unique > n_spots * 0.5:
                                    print(f"  WARNING: obs['{c}'] has {n_unique} unique values on {n_spots} spots — may be continuous, not cluster labels")
                                elif n_unique == 1:
                                    print(f"  WARNING: obs['{c}'] has only 1 unique value — clustering may have failed")
        else:
            errors.append(f"Worker returned error on real data: {out.get('error','')[:200]}")
            print(f"  FAIL: {out.get('error','')[:100]}")
    except subprocess.TimeoutExpired:
        # Soft (no CRITICAL_PATTERNS match): a slow tool registers DEGRADED, it is not rolled back.
        errors.append(f"Real data test timed out: no result within TEST5_TIMEOUT_S={TEST5_TIMEOUT_S}s "
                      f"on {test_data}; raise TEST5_TIMEOUT_S and re-run Test 5")
        print(f"  FAIL (soft): timed out after TEST5_TIMEOUT_S={TEST5_TIMEOUT_S}s")
    except Exception as e:
        errors.append(f"Real data test failed: {e}, stderr: {r.stderr[:200]}")
        print(f"  FAIL: {e}")
    # Cleanup
    import shutil
    shutil.rmtree(test_out, ignore_errors=True)
else:
    untested_reason = (
        f"{input_param_name!r} needs a real file of a format this host does not have "
        f"and that cannot be synthesised (rds/bam/vcf/fastq/image/csv/pdb/dir)"
    )
    print(f"  NOT RUN: {untested_reason}. The worker has never been executed on data.")

# Summary -- not the verdict either (see the Tests 1-4 summary): the HARD GATE below raises.
if errors:
    for _e in errors:
        print(f"  - {_e}")
    print(f"\n{len(errors)} failure(s) -- fix them (read the error, the upstream source or examples, "
          "fix the worker, re-test), then Phase 5.1 and the HARD GATE decide. NOT ready for Phase 6.")
elif untested_reason:
    print(f"\nOther tests passed, but Test 5 (real data) DID NOT RUN: {untested_reason}.")
    print("Stage a small sample of that format and re-run Test 5 before Phase 6.")
elif synthetic_input:
    print("\nALL TESTS PASSED on a SYNTHETIC fixture -- ready for Phase 6, but the "
          "output has not been checked against real data. Phase 5 must run on the "
          "user's own file before this tool is trusted for analysis.")
else:
    print("\nALL TESTS PASSED — ready for Phase 6")
```

**If any test fails (especially Test 5), you MUST fix it yourself. Do NOT ask the user.**

**Test 5 failure recovery (execute ALL of these, do NOT skip):**
1. Read the error output from the worker
2. Fetch the GitHub README and find usage examples:
   `urllib.request.urlopen(f"https://raw.githubusercontent.com/{org}/{repo}/main/README.md")`
3. Fetch example scripts from the repo's `examples/`, `tests/`, `tutorials/` directories
4. Read the installed package source code:
   `conda run -n user_{tool_id} python -c "import {module}.{submod}; import inspect; print(inspect.getsource({module}.{submod}))"`
5. Fix the worker's core logic based on what you learned
6. Re-run ALL 6 tests
7. Repeat until Test 5 passes (max 3 iterations)
8. If still failing after 3 fix iterations: clone the repo and read its test files:
   `git clone https://github.com/{org}/{repo} /tmp/{tool_id}_source && cat /tmp/{tool_id}_source/tests/*.py`

**You MUST NOT ask the user for help with the API. You have the source code — read it.**
**You MUST NOT say "I couldn't find the API" — every Python package has inspectable source.**
**You MUST NOT skip Test 5 — it's the most important test.**

### Phase 5.1 — Self-review (MODE-GATED — MANDATORY when self_review_enabled=True)

When self-review is enabled, Phase 5 test failures are classified and
remediated BEFORE the HARD GATE fires. This drastically reduces
rollback-and-retry loops for transient failures (missing dep, schema
typo, wrong API attribute).

=== MANDATORY GATE — execute exactly per the contract below ===

If `default_config.self_review_enabled` is True AND Phase 5 `errors` is
non-empty (any of Tests 1-5 failed), you **MUST** invoke `self_review_loop`
BEFORE calling rollback. Skipping this when the mode is on is a contract
violation; the driver will log a `self_review_not_invoked` issue.

`self_review_loop` (`tools_user/self_review.py`) is guaranteed to write
`tools_user/.knowledge/{tool_id}/self_review_history.json` on every
classification attempt. The driver greps this file as evidence the loop
fired.

If `default_config.self_review_enabled` is False, you **MUST NOT** invoke
`self_review_loop` (no-op, no print, no history file). Behavior is
identical to a system without the self-review module installed.

=== END MANDATORY GATE ===

STCoscientist does not need to implement classification/remediation logic —
the `tools_user.self_review` module provides it. Invoke it once, at
the point where Phase 5 `errors` is non-empty:

```python
if errors and getattr(default_config, "self_review_enabled", False):
    # MANDATORY: import MUST succeed; do NOT try/except.
    from tools_user.self_review import self_review_loop, RemediationContext
    ctx = RemediationContext(
        tool_id=tool_id, env_name=env_name, source_url=github_url,
        worker_path=worker,   # the .py or .R worker Tests 1-5 ran
        server_path=TOOLS_USER_DIR / f"{tool_id}_mcp_server.py",
        language=language,
        knowledge_dir=(KNOWLEDGE_DIR / tool_id),
    )
    aggregated_stdout, aggregated_stderr = "", "\n".join(errors)   # what Tests 1-5 reported
    def _rerun():
        """Re-run the same Phase 5 tests after a remediation. Return
        (success, stdout, stderr). Implement by calling Test 1..5 again
        and returning the aggregated result."""
        # ... re-execute the failing tests, aggregate results ...
        return (not new_errors, aggregated_stdout, aggregated_stderr)
    success, final_err, history = self_review_loop(
        ctx, error_msg=errors[0],
        stdout=aggregated_stdout, stderr=aggregated_stderr,
        rerun=_rerun,
    )
    # MANDATORY: confirm the history file was written. If not, raise loud.
    _srh = (KNOWLEDGE_DIR / tool_id / "self_review_history.json")
    assert _srh.exists(), (
        f"self_review_loop returned but did not write {_srh} — "
        f"this should not happen; verify tools_user/self_review.py is current"
    )
    if success:
        print(f"[self-review] recovered after {len(history)} round(s); continuing to Phase 6.")
        errors = []
    else:
        print(f"[self-review] exhausted; proceeding to HARD GATE. History: {history}")
```

When the flag is OFF (default), this block is skipped — the creation
flow behaves EXACTLY as before. No new directories, no new files, no
new module imports.

Self-review does NOT:
- Override the HARD GATE — if it fails to remediate, HARD GATE still fires.
- Skip Phase 5 tests — the tests re-run after each remediation.
- Allow STCoscientist to hallucinate remediations — the classifier is code, not prose.
- Touch memory or install_log without explicit integration (bookkeeping
  happens inside `install_log.remediated_deps` only).

(Strategy notes formerly under benchmarks/creation_deletion/debug_log/patterns/ —
this is benchmark-internal documentation, not a runtime dependency. STCoscientist does
NOT need to read this file to operate; it's a reference for benchmark authors.)

### HARD GATE between Phase 5 and Phase 6 — do NOT register DEGRADED tools

A recurring failure mode in past runs: Phase 5's Test 1 (import) fails, the tool gets
registered anyway in `install_log.json` with `status="active"`, and every future invocation
breaks. That produces a dead-weight tool that *looks* installed but isn't.

Apply this decision rule AFTER running the tests above, BEFORE going to Phase 6:

```python
# Read the `errors` list populated by the tests above, then categorise:
critical_keys = ("Import failed", "Syntax error", "not valid JSON")
critical = [e for e in errors if any(k in e for k in critical_keys)]
soft     = [e for e in errors if e not in critical]

if critical:
    # HARD RULE — VENDOR-ONLY FALLBACK (tool-agnostic, try this BEFORE rollback)
    #
    # If the only critical failure is "Import failed" (Python) or
    # "library(X) could not be verified" (R), the package files WERE written
    # to disk by Phase 2 but the runtime can't resolve them as an installable
    # module. Instead of unconditionally rolling back and leaving the user
    # with zero tools, degrade to vendor-only registration:
    #
    #   1. Locate the source tree (`pip show <pkg> | grep Location`, or
    #      `Rscript -e '.libPaths()' + list.dirs()`, or the `vendor/<pkg>`
    #      dir populated by the Strategy-8 vendor-source ladder step).
    #   2. Rewrite the worker so it uses `sys.path.insert(0, VENDOR_DIR)` +
    #      `importlib.import_module(...)` for Python, OR
    #      `devtools::load_all(VENDOR_DIR)` for R.
    #   3. Re-run Test 1. If the vendor-load import now succeeds, continue
    #      to Phase 6 and register with an extra field
    #      `install_mode="vendor"` in install_log.json + health annotation
    #      DEGRADED so health_check_tools surfaces the caveat.
    #   4. If the vendor-load still fails, fall through to rollback below.
    #
    # Why: the same build artefacts that pip/devtools could not link into
    # the active site-packages tree may still be importable when the worker
    # explicitly prepends the vendor dir. A tool the user can call via
    # vendor-path is infinitely more useful than no tool at all.
    # Applies to any language; just swap the load mechanism.
    import_failed = any("Import failed" in e or "could not be verified" in e
                        for e in critical)
    if import_failed:
        print("Attempting vendor-only fallback before rollback...")
        vendor_dir = locate_source_tree(tool_id, package_name)  # see helpers
        if vendor_dir and retry_import_from_vendor(vendor_dir):
            install_mode = "vendor"
            health_annotation = "DEGRADED"
            critical = []  # suppress — vendor-load succeeded
        else:
            print("Vendor-only fallback did not resolve the import.")

if critical:
    print(f"REGISTRATION BLOCKED: {len(critical)} critical failure(s):")
    for e in critical:
        print(f"  - {e}")
    print("Calling rollback(tool_id) and NOT proceeding to Phase 6.")
    rollback(tool_id)   # function defined at the bottom of this doc
    raise RuntimeError(f"Phase 5 critical failures for {tool_id} — rolled back.")

# Soft failures (real-data subtest warnings, empty cluster labels, etc.) are allowed —
# the tool is registered but marked DEGRADED via the status field in install_log:
install_log_status = "active" if not soft else "active"   # status stays active
health_annotation  = "HEALTHY" if not soft else "DEGRADED"
# Record health_annotation inside `tools_user/.knowledge/{tool_id}/creation_log.json`
# so downstream health_check_tools can report it accurately.
```

**Rule of thumb:** Test 1 (import) and Test 2 (syntax) failing means the worker literally
cannot run — ALWAYS rollback. Test 3 (dry-run JSON) failing means the error-path isn't
WorkerOutput-compliant — ALWAYS rollback. Test 4 (config) failing means you never
actually registered — fix and retry, don't proceed. Test 5 (real-data) failing is the only
category where DEGRADED registration is acceptable, and even there, annotate the health
status so downstream callers know.

### HARD GATE — mechanical enforcement (no narrative override)

The following block is MANDATORY to execute verbatim AFTER Phase 5 tests and
BEFORE Phase 6. Do not paraphrase it, do not split it into narrative
prose ("though I considered registering anyway...") — run it as-is. Past
runs registered tools with Test 1 FAIL because the rule was expressed as
prose and STCoscientist rationalized past it.

```python
# ==== MECHANICAL HARD GATE (do not rewrite, do not split) ====
CRITICAL_PATTERNS = (
    "Import failed", "Syntax error", "not valid JSON",
    "could not be verified", "IMPORT_FAIL", "SYNTAX_FAIL",
    "library() failed", "not loadable",
    # P9-b — missing-module errors are ALWAYS critical,
    # regardless of what surrounding text calls them ("soft warning"):
    "ModuleNotFoundError", "No module named", "ImportError",
    "Please install",
)
critical = [e for e in errors if any(k in str(e) for k in CRITICAL_PATTERNS)]

if critical:
    # Try vendor-only fallback (Python or R) before giving up.
    ok = False
    if language == "python":
        vendor_dir = locate_source_tree(tool_id, package_name)
        if vendor_dir:
            ok = retry_import_from_vendor(vendor_dir, verified_module,
                                          env_name=env_name)
        if ok:
            # P9-a — vendor-install the tool's declared requirements
            # with --no-deps so the env has pandas/numpy/scipy/etc
            # even though the tool itself is source-only:
            reqs = parse_requirements(vendor_dir)
            for req in reqs:
                subprocess.run(["conda", "run", "-n", env_name, "pip", "install",
                                "--no-deps", req],
                               capture_output=True, text=True, timeout=600)
            install_mode = "vendor"
    elif language == "R":
        vendor_r = locate_r_library(tool_id, r_pkg_name)
        if vendor_r:
            ok = retry_library_from_vendor(vendor_r, r_pkg_name, env_name=env_name)
        if ok:
            install_mode = "vendor-r"
    if ok:
        health_annotation = "DEGRADED"
        critical = []   # suppress — vendor-load succeeded
    else:
        print(f"HARD GATE BLOCKED: {len(critical)} critical failure(s).")
        for e in critical:
            print(f"  - {e}")
        rollback(tool_id, keep_knowledge=True)
        raise SystemExit(1)   # hard stop
soft = [e for e in errors if not any(k in str(e) for k in CRITICAL_PATTERNS)]
if soft:   # e.g. a Test 5 timeout or an output check: register, but DEGRADED (Phase 6 records it)
    health_annotation = "DEGRADED"
    print(f"Registering DEGRADED: {soft}")
# ==== END HARD GATE ====
```
