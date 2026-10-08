# Adding New MCP Tools - Phases 3.5 to 3.75: Audit the Code and Close the Environment

## Metadata
- Authors: SpatialOmicsLab
- Version: 2.0
- Category: tool_creation

## Overview
The three mandatory checks between writing a new MCP tool's code and registering it: audit the dependencies the code actually imports, close the environment against the runtime fixtures it will meet, and block on any attribute the worker uses but never defines.

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
from tools_user.knowledge_manager import KNOWLEDGE_DIR, TOOLS_USER_DIR
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

## Phase 3.5: Dependency Audit (MANDATORY)

**After writing the worker and MCP server files, you MUST scan them for ALL import
statements and verify that every dependency is installed in the conda env.
Do NOT proceed to Phase 4 until every import resolves successfully.**

This is the #1 cause of tool creation failures: the worker imports `anndata`, `scanpy`,
`matplotlib`, `sklearn`, etc. but only the core package was installed in Phase 2.
The tool then passes import/syntax tests (which only test the core module) but fails
on real data because the worker's own imports are missing.

### Why this happens

- Phase 2 installs the **tool package** (e.g., `gaston-spatial`), but the **worker script**
  you wrote in Phase 3 typically also imports data I/O libraries (`anndata`, `scanpy`),
  numerical libraries (`numpy`, `scipy`, `sklearn`), and plotting libraries (`matplotlib`,
  `seaborn`) that are NOT dependencies of the tool package itself.
- The health check `imports_scan` test will catch this later, but by then the tool is
  already registered and broken. Fix it here, before registration.

### P29 — Pip name vs import name disambiguation (MANDATORY)

A frequent failure mode: the **import name** differs from the **pip package
name**. Past benchmark failures:

| import (worker says)   | wrong pip name STCoscientist tried | correct pip name |
|---|---|---|
| `import skmisc`        | `pip install skmisc`    | `pip install scikit-misc` |
| `import sklearn`       | `pip install sklearn`   | `pip install scikit-learn` |
| `import liana.tl`      | `pip install liana`     | `pip install liana-py` |
| `import cv2`           | `pip install cv2`       | `pip install opencv-python` |
| `import skimage`       | `pip install skimage`   | `pip install scikit-image` |
| `import yaml`          | `pip install yaml`      | `pip install pyyaml` |
| `import PIL`           | `pip install PIL`       | `pip install pillow` |
| `import bs4`           | `pip install bs4`       | `pip install beautifulsoup4` |

**Rule when an import fails with `ModuleNotFoundError`:**

1. Try `pip install <import_name>` first.
2. If that fails with "No matching distribution", search the well-known
   import→pip name table above; if the name is in it, use the corrected pip name.
3. If still no match, query PyPI:
   ```python
   r = subprocess.run(["conda", "run", "-n", env_name, "pip", "index",
                       "versions", import_name],
                      capture_output=True, text=True, timeout=30)
   if "Available versions" not in r.stdout:
       # Try common transformations
       for cand in [f"scikit-{import_name}", f"py{import_name}",
                    f"{import_name}-py", import_name.replace('_', '-')]:
           r2 = subprocess.run(["conda", "run", "-n", env_name, "pip", "index",
                                "versions", cand],
                               capture_output=True, text=True, timeout=30)
           if "Available versions" in r2.stdout:
               print(f"[dep-audit] {import_name!r} → pip install {cand!r}")
               break
   ```
4. Record the resolved (import_name → pip_name) mapping in
   `.knowledge/{tool_id}/dep_resolution.json` so health_check + future
   re-runs can use it.

### Step 1: Extract all imports from generated files

```python
import ast, subprocess
from pathlib import Path

env_name = f"user_{tool_id}"
worker_path = TOOLS_USER_DIR / f"{tool_id}_worker.py"
server_path = TOOLS_USER_DIR / f"{tool_id}_mcp_server.py"

# Parse both files and collect all top-level imported modules
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

# Remove stdlib and local modules (worker_utils, base_mcp, os, sys, etc.)
import sys as _sys
stdlib = set(_sys.stdlib_module_names) | {"worker_utils", "base_mcp"}
third_party = sorted(all_imports - stdlib)
print(f"Third-party imports found: {third_party}")
```

### Step 2: Test each import in the conda env

```python
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
```

### Step 3: Install ALL missing dependencies

```python
if missing:
    print(f"Installing {len(missing)} missing dependencies: {missing}")
    # Map common import names to pip package names where they differ
    import_to_pip = {
        "sklearn": "scikit-learn",
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "skimage": "scikit-image",
        "yaml": "pyyaml",
        "Bio": "biopython",
    }
    pip_names = [import_to_pip.get(m, m) for m in missing]
    r = subprocess.run(
        ["conda", "run", "-n", env_name, "pip", "install"] + pip_names,
        capture_output=True, text=True, timeout=1800,
    )
    if r.returncode != 0:
        print(f"pip install failed: {r.stderr[:500]}")
        print("Trying packages one by one...")
        for pkg in pip_names:
            subprocess.run(
                ["conda", "run", "-n", env_name, "pip", "install", pkg],
                capture_output=True, text=True, timeout=600,
            )

    # Verify ALL imports now resolve
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
        print("Fix these manually before proceeding!")
    else:
        print("All dependencies installed and verified.")
else:
    print("All dependencies already available — no action needed.")
```

### Step 4: Also check for common implicit dependencies

Some packages are almost always needed for spatial tools but may not appear
as direct imports in the worker (they're used by the tool package at runtime):

```python
# Common implicit deps for spatial transcriptomics tools
# DOMAIN-AWARE runtime-deps: install ONLY the group whose anchor package
# the worker actually imports. Previous versions hardcoded a scanpy-specific
# list that got installed into every env — a waste for non-biology tools and
# a confusing signal for future maintainers.
#
# Each group lists the runtime-only companions that the ANCHOR package calls
# out to internally but that don't appear in any static import scan. If the
# worker doesn't import the anchor, we skip the whole group.

RUNTIME_DEPS_BY_DOMAIN = {
    # Anchor: single-cell / spatial-omics (scanpy, anndata, liana, squidpy)
    "single_cell_omics": {
        "anchors": ["scanpy", "anndata", "liana", "squidpy", "sopa"],
        "deps":    ["leidenalg", "python-igraph", "igraph",
                    "scikit-misc", "pynndescent", "louvain"],
    },
    # Anchor: image / vision (cv2, PIL, skimage, tifffile)
    "imaging": {
        "anchors": ["cv2", "PIL", "skimage", "tifffile", "zarr"],
        "deps":    ["opencv-python", "Pillow", "scikit-image", "tifffile",
                    "imagecodecs"],
    },
    # Anchor: deep learning
    "deep_learning": {
        "anchors": ["torch", "tensorflow", "jax"],
        "deps":    [],   # nothing extra; framework brings its own
    },
    # Anchor: sequence bio (pysam, Bio, pyfastx, cyvcf2)
    "sequence_bio": {
        "anchors": ["pysam", "Bio", "pyfastx", "cyvcf2"],
        "deps":    [],   # usually self-contained
    },
    # Anchor: tabular data only
    "tabular": {
        "anchors": ["pandas", "polars", "pyarrow"],
        "deps":    [],
    },
    # Anchor: structural biology
    "structural_bio": {
        "anchors": ["biotite", "Bio.PDB", "prody"],
        "deps":    [],
    },
}

# Read worker imports once (computed earlier in Phase 3.5). `worker_imports`
# is the set of top-level module names found by the AST scan.
def pick_runtime_deps(worker_imports: set[str]) -> list[str]:
    out: list[str] = []
    for domain, cfg in RUNTIME_DEPS_BY_DOMAIN.items():
        if any(a in worker_imports for a in cfg["anchors"]):
            out.extend(cfg["deps"])
    return out

common_spatial_deps = pick_runtime_deps(worker_imports)
# Idiomatic Python: if the worker is a pure CLI wrapper with no scanpy/cv2/
# etc., this list will be empty and the loop below becomes a no-op.
for dep in common_spatial_deps:
    r = subprocess.run(
        ["conda", "run", "-n", env_name, "python", "-c", f"import {dep}"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        print(f"  Installing common dep: {dep}")
        subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install", dep],
            capture_output=True, text=True, timeout=300,
        )
```

### Step 4b: Runtime dry-run to catch indirect deps (CRITICAL)

**Static import scans miss transitive imports** — e.g. `scanpy.pp.highly_variable_genes`
only imports `skmisc` when `flavor="seurat_v3"` is called at runtime, not at module
load. Past runs had workers that passed Phase 3.5's static scan but blew up at real-data
time with `ModuleNotFoundError: No module named 'skmisc'`. Mitigation: after the static
scan, run the worker end-to-end with a nonexistent input path so the error path
exercises every import branch without needing real data.

```python
r = subprocess.run(
    ["conda", "run", "-n", env_name, "python",
     str(TOOLS_USER_DIR / f"{tool_id}_worker.py"),
     "--input", "/nonexistent/dryrun.h5ad",
     "--output-dir", f"/tmp/dryrun_{tool_id}"],
    capture_output=True, text=True, timeout=120,
)
# Look for ModuleNotFoundError anywhere in stderr — that means an indirect dep
import re
missing_modules = re.findall(r"No module named ['\"]([a-zA-Z0-9_.]+)['\"]", r.stderr)
known_name_fixes = {
    "skmisc": "scikit-misc",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
}
for mod in set(missing_modules):
    pip_name = known_name_fixes.get(mod, mod)
    print(f"  [dryrun] detected indirect dep: import {mod} → installing {pip_name}")
    subprocess.run(["conda", "run", "-n", env_name, "pip", "install", pip_name],
                   capture_output=True, text=True, timeout=600)
```

This dry-run runs the worker's full error path (input not found → WorkerOutput.error),
which is enough to exercise `import` statements inside every branch. If any
`ModuleNotFoundError` surfaces, install the missing package. Repeat once if new
modules pop up after the first install.

### Step 5: Export updated environment

```python
# Re-export env.yaml after all deps are installed
subprocess.run(
    ["conda", "env", "export", "-n", env_name, "--no-builds"],
    capture_output=True, text=True, timeout=60,
)
# Write to tools_user/{tool_id}_env.yaml
env_yaml = subprocess.run(
    ["conda", "env", "export", "-n", env_name, "--no-builds"],
    capture_output=True, text=True, timeout=60,
).stdout
(TOOLS_USER_DIR / f"{tool_id}_env.yaml").write_text(env_yaml)
print(f"Environment exported to tools_user/{tool_id}_env.yaml")
```

**This phase is NON-NEGOTIABLE. Every import in the worker and server files MUST resolve
in the conda env before you proceed. A tool that fails `imports_scan` is BROKEN.**

### Phase 3.5b: Worker safety audit (MANDATORY — block on errors, silent on warnings)

After dep-audit, run a static AST scan over the generated worker +
server before proceeding to Phase 3.75. Errors block with rollback;
warnings log to `.knowledge/{tool_id}/safety_audit.json` and do NOT
print to user (per UX-3).

### Error / warning policy table

| Category | On detection | User output |
|---|---|---|
| `os.system`, `subprocess.Popen(shell=True)`, `eval`, `exec`, `__import__`, `marshal.*` | BLOCK + rollback | 1-line rationale + fix |
| P13: `import mcp_servers.*` / `from mcp_servers.* import` | BLOCK + rollback | 1-line rationale |
| P20: `@mcp.tool` with >2 required params | BLOCK + rollback | 1-line rationale |
| P22: `@mcp.tool` function not named `{tool_id}_run` | AUTO-FIX (rename the def; the gate resets `function_name`, so the YAML matches) | 1-line note |
| Writes to `install_log.json` outside Phase 6 | BLOCK + rollback | 1-line rationale |
| `pickle.load` on arbitrary bytes | WARNING | silent |
| Missing `add_output_file` / `emit_error` call | WARNING | silent (P23 risk note) |
| `open(args.X + Y)` concat that looks path-traversal-prone | WARNING | silent |
| `subprocess.run` calling conda/pip at runtime (worker mutating env) | WARNING | silent |
| R worker `system()`, `system2()`, `eval(parse())`, `unlink(recursive=TRUE)` | BLOCK + rollback | 1-line rationale |

### Implementation

```python
import ast, json, re
from pathlib import Path

DANGEROUS_CALLS = {
    "os.system", "eval", "exec", "__import__",
    "pickle.load", "pickle.loads",
    "marshal.load", "marshal.loads",
    "subprocess.getoutput", "subprocess.getstatusoutput",
}

def _unparse(node) -> str:
    """ast.unparse fallback for Py<3.9."""
    try:
        return ast.unparse(node)
    except Exception:
        if isinstance(node, ast.Attribute):
            return f"{_unparse(node.value)}.{node.attr}"
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Call):
            return _unparse(node.func)
        return "<unparseable>"

def _allow_line(src_lines: list[str], lineno: int) -> bool:
    """Check 3-line window for `# safety-audit: allow` comment."""
    for ln in range(max(0, lineno - 3), lineno):
        if ln < len(src_lines) and "safety-audit: allow" in src_lines[ln]:
            return True
    return False

def audit_worker_safety(worker_path: Path, server_path: Path,
                        *, language: str = "python",
                        install_mode: str | None = None
                        ) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    if language == "R":
        return _audit_r_worker(worker_path, server_path)

    worker_src = worker_path.read_text(errors="replace")
    server_src = server_path.read_text(errors="replace")
    try:
        worker_tree = ast.parse(worker_src)
    except SyntaxError as e:
        return [f"[syntax] worker has SyntaxError at line {e.lineno}: {e.msg}"], []
    try:
        server_tree = ast.parse(server_src)
    except SyntaxError as e:
        return [f"[syntax] server has SyntaxError at line {e.lineno}: {e.msg}"], []

    worker_lines = worker_src.split("\n")

    # Dangerous calls + subprocess(shell=True)
    for node in ast.walk(worker_tree):
        if not isinstance(node, ast.Call):
            continue
        fname = _unparse(node.func)
        if fname in DANGEROUS_CALLS and not _allow_line(worker_lines, node.lineno):
            errors.append(
                f"[security] {fname}() at line {node.lineno}. "
                f"Fix: remove or mark with `# safety-audit: allow` if intentional."
            )
        if fname in ("subprocess.run", "subprocess.Popen",
                     "subprocess.check_output", "subprocess.check_call",
                     "subprocess.call"):
            for kw in node.keywords:
                if (kw.arg == "shell" and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                        and not _allow_line(worker_lines, node.lineno)):
                    errors.append(
                        f"[security] {fname}(shell=True) at line {node.lineno}. "
                        f"Fix: pass a list of args (shell=False) instead of a string."
                    )

    # P13 — import mcp_servers.*
    for node in ast.walk(worker_tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mcp_servers"):
            errors.append(
                f"[P13] `from {node.module} import ...` at line {node.lineno}. "
                f"Fix: MCP tools are not Python modules. Call via spatialomicsgym_name."
            )
        if isinstance(node, ast.Import):
            for nm in node.names:
                if nm.name.startswith("mcp_servers"):
                    errors.append(
                        f"[P13] `import {nm.name}` at line {node.lineno}. "
                        f"Fix: remove this import; invoke via spatialomicsgym_name instead."
                    )

    # Writes to install_log.json from worker (forbidden)
    for node in ast.walk(worker_tree):
        if isinstance(node, ast.Call):
            fname = _unparse(node.func)
            if (fname.endswith(".write_text") or fname == "open") and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str) \
                        and "install_log.json" in a0.value:
                    errors.append(
                        f"[integrity] worker writes install_log.json at line "
                        f"{node.lineno}. Fix: only Phase 6 registration writes this."
                    )

    # P20 — @mcp.tool parameter count
    for node in ast.walk(server_tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        is_mcp_tool = any(
            (isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "tool")
            or (isinstance(d, ast.Attribute) and d.attr == "tool")
            for d in node.decorator_list
        )
        if not is_mcp_tool:
            continue
        n_required = len(node.args.args) - len(node.args.defaults)
        if n_required > 2:
            errors.append(
                f"[P20] @mcp.tool {node.name} has {n_required} required args. "
                f"Fix: mark all but primary input + output_dir as Optional[...] with defaults."
            )
        # P22 -- naming (auto-fixable, don't block; return as note). The id comes from the
        # server's own filename: ids contain '_', so a split on '_' renames st_gears_* to st_run.
        new_name = server_path.name[: -len("_mcp_server.py")] + "_run"
        if node.name != new_name:
            warnings.append(f"[P22-auto-fix] @mcp.tool function {node.name!r} renamed to {new_name!r}.")
            server_src = re.sub(rf"\bdef\s+{re.escape(node.name)}\s*\(",
                                f"def {new_name}(", server_src)
            server_path.write_text(server_src)
            # The gate below resets function_name, which Phase 4 writes as the YAML spatialomicsgym_name

    # WARNINGS (silent by default)
    # P23 — worker never calls add_output_file / add_output_files
    has_out = any(
        isinstance(n, ast.Call) and _unparse(n.func).endswith(
            ("add_output_file", "add_output_files"))
        for n in ast.walk(worker_tree)
    )
    if not has_out:
        warnings.append(
            "[P23] worker never calls add_output_file — silent no-op risk on real data."
        )
    has_emit_error = any(
        isinstance(n, ast.Call) and (
            "emit_error" in _unparse(n.func)
            or _unparse(n.func).endswith("WorkerOutput.error"))
        for n in ast.walk(worker_tree)
    )
    if not has_emit_error:
        warnings.append(
            "[P23] worker never calls WorkerOutput.emit_error — errors may swallow silently."
        )
    # pickle.load warning
    for node in ast.walk(worker_tree):
        if isinstance(node, ast.Call):
            fname = _unparse(node.func)
            if fname in ("pickle.load", "pickle.loads") and _allow_line(worker_lines, node.lineno):
                pass  # explicitly allowed — no warning
            elif fname in ("pickle.load", "pickle.loads"):
                warnings.append(
                    f"[pickle] {fname}() at line {node.lineno} — safe only on "
                    f"local trusted files. Add `# safety-audit: allow` to silence."
                )
    # Path-traversal smell
    for node in ast.walk(worker_tree):
        if isinstance(node, ast.Call) and _unparse(node.func) in ("open", "builtins.open"):
            if node.args and isinstance(node.args[0], ast.BinOp):
                warnings.append(
                    f"[path] open() at line {node.lineno} concatenates paths. "
                    f"Verify no user input can escape output_dir."
                )

    return errors, warnings


def _audit_r_worker(worker_path: Path, server_path: Path) -> tuple[list[str], list[str]]:
    """Regex-based R audit (no ast.parse for R)."""
    errors, warnings = [], []
    src = worker_path.read_text(errors="replace")
    src_lines = src.split("\n")
    DANGEROUS_R = [
        (r"\bsystem\s*\(", "system()",
         "Fix: use subprocess from the Python wrapper instead of R system()."),
        (r"\bsystem2\s*\(", "system2()",
         "Fix: use subprocess from the Python wrapper instead of R system2()."),
        (r"\beval\s*\(\s*parse\b", "eval(parse())",
         "Fix: eval(parse(text=...)) is code injection surface."),
        (r"unlink\s*\([^)]*recursive\s*=\s*TRUE", "unlink(recursive=TRUE)",
         "Fix: recursive unlink is dangerous; delete only specific known paths."),
    ]
    for pat, label, fix in DANGEROUS_R:
        for m in re.finditer(pat, src):
            line = src[:m.start()].count("\n") + 1
            if _allow_line(src_lines, line):
                continue
            errors.append(f"[security-R] {label} at line {line}. {fix}")
    return errors, warnings
```

### Gate invocation (place this right before Phase 3.75)

```python
errors, warnings = audit_worker_safety(
    TOOLS_USER_DIR / f"{tool_id}_worker{'.R' if language == 'R' else '.py'}",
    TOOLS_USER_DIR / f"{tool_id}_mcp_server.py",
    language=language,
    install_mode=locals().get("install_mode"),
)
function_name = f"{tool_id}_run"   # P22 renamed the def to this; Phase 4 writes the YAML from it

# Always persist for debugging
(KNOWLEDGE_DIR / tool_id).mkdir(parents=True, exist_ok=True)
(KNOWLEDGE_DIR / tool_id / "safety_audit.json").write_text(
    json.dumps({"errors": errors, "warnings": warnings}, indent=2)
)

# UX-3: errors block + 1-line rationale; warnings SILENT unless verbose
if errors:
    print(f"[safety-audit] BLOCKING — {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    rollback(tool_id, keep_knowledge=True)
    raise SystemExit("Phase 3.5b worker safety audit failed — rolled back.")
# warnings: silent on happy path. Opt-in via `verbose_audit=True`.
```

### Escape hatches

- Per-line: append `# safety-audit: allow <reason>` to the flagged line
  (or the 2 lines above). The auditor skips that node.
- Tool-wide opt-out: `default_config.strict_safety_audit = False`
  disables the audit entirely. Use only for rapid prototyping.

### HARD RULE — Pre-install the "usual-suspects" transitive deps based on worker imports (P25)

Static scans and fixture dry-runs BOTH miss transitive deps that only
surface when a specific real-data code path runs. The following
high-frequency offenders MUST be pre-installed when the matching
parent import is present in the worker or the tool's README. These are
gathered from repeated failures across many tool creations — treat
this list as a KNOWN-ALWAYS-NEEDED mapping, not a suggestion.

```python
# HARD RULE P25 — preemptive transitive-dep install.
# Pre-install the suspect list BEFORE running Phase 3.5/3.75 dry-runs.
# Applies when the parent package is imported anywhere in the worker.
TRANSITIVE_DEPS = {
    "scanpy":  ["leidenalg", "python-igraph", "louvain", "scikit-misc",
                "harmonypy"],
    # scanpy clustering (sc.tl.leiden / sc.tl.louvain) NEEDS leidenalg +
    # python-igraph. HVG seurat_v3 flavor NEEDS scikit-misc. Integration
    # workflows often need harmonypy.
    "squidpy": ["leidenalg", "python-igraph", "scikit-misc",
                "libpysal", "esda"],
    "stlearn": ["leidenalg", "python-igraph"],
    "scvi":    ["pytorch-lightning", "lightning"],
    "anndata": [],  # no common transitive gaps
    "rpy2":    ["anndata2ri"],  # Python<->R bridge often needs this
    "liana":   ["decoupler", "omnipath", "leidenalg", "python-igraph"],
    "cellrank":["pygam"],
    "mofapy2": ["dtw-python"],
}

import re
from pathlib import Path
worker_src = (TOOLS_USER_DIR / f"{tool_id}_worker{'.R' if language == 'R' else '.py'}").read_text()
server_src = (TOOLS_USER_DIR / f"{tool_id}_mcp_server.py").read_text()
combined = worker_src + "\n" + server_src
to_install = set()
for parent, transitives in TRANSITIVE_DEPS.items():
    if re.search(rf"\bimport\s+{re.escape(parent)}\b", combined) or \
       re.search(rf"\bfrom\s+{re.escape(parent)}\b", combined):
        to_install.update(transitives)

if to_install:
    print(f"[P25] Pre-installing suspect transitive deps: {sorted(to_install)}")
    subprocess.run(
        ["conda", "run", "-n", env_name, "pip", "install", *sorted(to_install)],
        capture_output=True, text=True, timeout=900,
    )
```

The list above is mechanical + auditable. When a new transitive-dep
gap is discovered in the wild, ADD A NEW ENTRY and the next tool
benefits. Do NOT leave Phase 3.75's reactive loop as the only safety
net — by the time it fires, you've already wasted ~20 min on a
failed attempt.

**Why static scan misses them**: `sc.tl.leiden` is a SCANPY function
call, not a top-level Python import. Our worker has `import scanpy as
sc` only. `leidenalg` is loaded at CALL-TIME inside scanpy.tl.leiden's
implementation. AST-level import audit cannot see it.

**Why Phase 3.75 env closure CAN miss them**: the minimal fixture we
run may not exercise the leiden code path at all (e.g., if the tool's
default pipeline uses K-means or Louvain-without-leidenalg). Real-data
may take a different branch.

**Conclusion**: preemptive P25 install is the ONLY reliable
mitigation. The 5-7 seconds it adds to Phase 3.5 saves 20+ minutes of
a failed real-data attempt.

## Phase 3.75: Env Closure (MANDATORY — runtime fixture loop)

**Purpose**: a static import scan (Phase 3.5) catches `import X` where X is missing,
but cannot catch transitive deps that only load when a specific code path runs
(scanpy's `highly_variable_genes(flavor="seurat_v3")` requires `scikit-misc`;
rpy2 workers need `Rscript` on PATH; R workers need `R_HOME`; etc.). These are
runtime-visible gaps. Close them here via an iterative fixture loop before
committing to Phase 4 registration.

```python
import json, re, subprocess
from pathlib import Path

env_name = f"user_{tool_id}"
# The worker the tool will run, and the interpreter that runs it (an R worker is Rscript's).
worker_file = TOOLS_USER_DIR / f"{tool_id}_worker{'.R' if language == 'R' else '.py'}"
runner = "Rscript" if worker_file.suffix == ".R" else "python"
CLOSURE_TIMEOUT_S = 300   # one worker run on the fixture; raise it for a tool slow even on 24 spots

# Resolve a TOOL-AGNOSTIC minimal fixture matching the tool's input kind. An h5ad-shaped kind is
# ALWAYS synthesised below -- never a library slide, which would turn this into a slow real-data
# run (Phase 5 Test 5 is that run). Synthetic input is adequate HERE and only here: this loop asks
# "does the worker's code path reach a dependency that is not installed", not "is the answer
# correct". The shipped modification gate deliberately does NOT synthesise, because it reports
# a real-data verdict. Do not unify the two.
# Kinds a synthesised AnnData cannot stand in for. Each needs a real file of that
# exact format, so these are the only kinds this phase is allowed to skip. Paths are
# relative to the checkout and resolved against it below, never against the CWD.
FIXTURES_BY_KIND = {
    "input_rds":   ["benchmarks/benchmark_data/mini/mini_seurat.rds"],
    "input_bam":   ["benchmarks/benchmark_data/mini/mini.bam"],
    "input_vcf":   ["benchmarks/benchmark_data/mini/mini.vcf"],
    "input_fastq": ["benchmarks/benchmark_data/mini/mini_R1.fastq.gz"],
    "input_image": ["benchmarks/benchmark_data/mini/mini.tif"],
    "input_csv":   ["benchmarks/benchmark_data/mini/mini.csv"],
    "input_pdb":   ["benchmarks/benchmark_data/mini/mini.pdb"],
    "input_dir":   ["benchmarks/benchmark_data/mini/mini_dir/"],
}

def _resolve_fixture(input_kind):
    repo_root = TOOLS_USER_DIR.parent
    for rel in FIXTURES_BY_KIND.get(input_kind, []):
        if (repo_root / rel).exists():
            return str(repo_root / rel)
    return None   # an h5ad-shaped kind (input_h5ad, input_path, st_h5ad, ...) is synthesised below

FIXTURE = _resolve_fixture(primary_input_arg)
if FIXTURE is None and primary_input_arg not in FIXTURES_BY_KIND:
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
    FIXTURE = f"/tmp/sog_env_closure_{tool_id}.h5ad"
    adata.write_h5ad(FIXTURE)
    print(f"[Phase 3.75] Synthesised {FIXTURE} "
          f"({n_spots} spots x {n_genes} genes, with obsm['spatial']). That is enough "
          f"to walk the worker's code path and surface missing runtime deps. It is NOT "
          f"a correctness check -- Phase 5 still runs on the user's real data.")

if FIXTURE is None:
    print(f"[Phase 3.75] NOT RUN: {primary_input_arg!r} needs a real file of a format "
          f"this host does not have and that cannot be synthesised. Runtime-only "
          f"dependency gaps are therefore UNCHECKED for this tool -- Phase 3.6's static "
          f"audit covers imports only. Say so in the Phase 4 registration note, and "
          f"expect the first real run (Phase 5) to surface any missing package.")
    # The loop below then runs zero times; proceed to Phase 4 registration.

outcome, history = None, []
for iteration in range(5 if FIXTURE is not None else 0):
    try:
        r = subprocess.run(
            ["conda", "run", "-n", env_name, runner, str(worker_file),
             "--input", FIXTURE,   # the worker contract (P33), whatever the MCP parameter is called
             "--output-dir", f"/tmp/env_closure_{tool_id}"],
            capture_output=True, text=True, timeout=CLOSURE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        # Slow is not a missing dependency: record it, never roll back a working tool for it.
        history.append(f"iter {iteration}: no result within CLOSURE_TIMEOUT_S={CLOSURE_TIMEOUT_S}s; deps "
                       "past that point are unchecked -- raise CLOSURE_TIMEOUT_S to check them")
        outcome = "timed_out"
        break
    combined = (r.stderr or "") + (r.stdout or "")

    # Class A — Python ModuleNotFoundError / No module named 'X'
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", combined)
    if m:
        pkg = m.group(1).split(".")[0]
        history.append(f"iter {iteration}: missing Python module '{pkg}' — installing")
        subprocess.run(["conda", "run", "-n", env_name, "pip", "install", pkg],
                       capture_output=True, text=True, timeout=900)
        continue

    # Class B — "Please install X" or "No module named X" from nested tools
    m = re.search(r"[Pp]lease install (?:the )?([A-Za-z0-9_.-]+)", combined)
    if m:
        pkg = m.group(1)
        history.append(f"iter {iteration}: upstream asked for '{pkg}' — installing")
        subprocess.run(["conda", "run", "-n", env_name, "pip", "install", pkg],
                       capture_output=True, text=True, timeout=900)
        continue

    # Class C — FileNotFoundError for a CLI binary (Rscript, samtools, etc.)
    m = re.search(r"No such file or directory: ['\"]([A-Za-z0-9_.+-]+)['\"]", combined)
    if m:
        cli = m.group(1)
        history.append(f"iter {iteration}: missing CLI '{cli}' — conda-installing")
        subprocess.run(
            ["conda", "install", "-n", env_name, "-c", "conda-forge", "-c", "bioconda",
             cli, "-y"],
            capture_output=True, text=True, timeout=1200,
        )
        continue

    # Class D -- R missing package / shared object. Diagnose the package R NAMED: only the R path
    # defines r_pkg_name. diagnose_r_load is Phase 2's -- run its def first if Phase 2 never needed it.
    m = re.search(r"there is no package called\W{1,3}([A-Za-z][A-Za-z0-9.]*)", combined)
    if m or "unable to load shared object" in combined:
        ok, r_hist = diagnose_r_load(env_name, m.group(1) if m else module, max_iter=2)
        history.extend([f"iter {iteration}: R-load: {h}" for h in r_hist])
        if ok:
            continue
        outcome = "r_load_failed"
        break

    # Worker completed cleanly OR emitted a valid WorkerOutput error JSON:
    # env is closed, we can exit the loop.
    if r.returncode in (0, 1) and ('"status":' in combined or "status:" in combined):
        history.append(f"iter {iteration}: env closed (worker emitted valid JSON)")
        outcome = "closed"
        break

    # Unrecognised pattern -- stash last 500 chars and bail (NOT closed)
    history.append(f"iter {iteration}: unrecognised failure; tail: {combined[-500:]}")
    outcome = "unrecognised"
    break

if FIXTURE is not None:
    # Persisted on every path for post-hoc debugging; "closed" only when the worker really ran.
    (KNOWLEDGE_DIR / tool_id).mkdir(parents=True, exist_ok=True)
    (KNOWLEDGE_DIR / tool_id / "env_closure.json").write_text(json.dumps(
        {"closed": outcome == "closed", "outcome": outcome or "exhausted", "history": history}, indent=2))
    print(f"[Phase 3.75] {outcome or 'exhausted 5 iterations'}: {history[-1] if history else ''}")
    if outcome not in ("closed", "timed_out"):
        rollback(tool_id, keep_knowledge=True)
        raise RuntimeError(f"{tool_id}: env closure failed ({outcome or 'exhausted 5 iterations'})")
```

Only after this loop exits cleanly do you proceed to Phase 4. If it does not
close within the cap, do not register a broken tool — rollback and surface
the diagnosis.

## Phase 3.6: Worker API surface check (MANDATORY — block on undefined attrs)

**Why this exists.** A frequent failure mode in past benchmarks: STCoscientist generates
a worker referencing a `module.attribute` that does NOT exist (e.g.,
`scanpy.tl.nmf` is not a real attribute; `liana.foo` is not exported;
`nichecompass.bar` was renamed). Phase 5 Test 1 (import) only verifies the
package imports — it does NOT verify each `mod.attr` chain the worker uses.
Real-data execution then crashes with `AttributeError`. Phase 3.6 closes
that gap with a static AST + dynamic getattr-chain check before Phase 4.

**Procedure** — run before registering. Roll back on any unresolved chain.

```python
import ast
import subprocess
from pathlib import Path

# Python attribute chains only: an R worker has none, so for language == "R" this verifies 0 chains.
src = "" if language == "R" else (TOOLS_USER_DIR / f"{tool_id}_worker.py").read_text()
tree = ast.parse(src)

# Collect import bindings.
# - `import a`              → local 'a' bound to module a;
#                             "a.b" in source resolves as getattr(a, 'b').
# - `import a.b.c`          → local 'a' bound to top package only;
#                             "a.b.c.foo" resolves as
#                             getattr(getattr(getattr(a,'b'),'c'),'foo').
# - `import a.b.c as x`     → local 'x' bound to the leaf module a.b.c.
# - `from x import C`       → local 'C' bound to the imported name C
#                             (could be class/function/sub-module).
# - `from x.y import C as Y` → local 'Y' bound to x.y.C.
import_bindings: dict[str, dict] = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            full = alias.name              # e.g. "a.b.c"
            top = full.split('.')[0]       # e.g. "a"
            if alias.asname:
                # `import a.b.c as x` — x is the leaf module.
                import_bindings[alias.asname] = {
                    "kind": "import", "root_module": full,
                }
            else:
                # `import a.b.c` — only top-level `a` is bound; .b.c is
                # attribute access on the bound name.
                import_bindings[top] = {"kind": "import", "root_module": top}
    elif isinstance(node, ast.ImportFrom) and node.module:
        for alias in node.names:
            local = alias.asname or alias.name
            import_bindings[local] = {
                "kind": "from",
                "root_module": node.module,
                "from_name": alias.name,
            }

# Walk every Attribute chain rooted at an imported identifier.
attr_chains: set[tuple[str, str, tuple[str, ...]]] = set()
src_lines = src.splitlines()
for node in ast.walk(tree):
    if not isinstance(node, ast.Attribute):
        continue
    cur = node
    chain: list[str] = []
    while isinstance(cur, ast.Attribute):
        chain.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name) or cur.id not in import_bindings:
        continue
    line_text = src_lines[node.lineno - 1] if 0 < node.lineno <= len(src_lines) else ""
    if "noqa: p26" in line_text.lower() or "noqa:p26" in line_text.lower():
        continue
    binding = import_bindings[cur.id]
    if binding["kind"] == "import":
        attr_chains.add(("import", binding["root_module"], tuple(reversed(chain))))
    else:
        attr_chains.add((
            "from",
            f"{binding['root_module']}::{binding['from_name']}",
            tuple(reversed(chain)),
        ))

# Verify each chain inside the user env.
api_errors: list[str] = []
for kind, key, chain in sorted(attr_chains):
    if kind == "import":
        mod = key
        expr = mod + ("." + ".".join(chain) if chain else "")
        py_check = (
            f"import {mod} as _m\n"
            f"obj = _m\n"
            + "".join(f"obj = getattr(obj, {seg!r}); assert obj is not None\n" for seg in chain)
        )
    else:  # kind == "from"
        root_module, from_name = key.split("::", 1)
        expr = f"{root_module}.{from_name}" + ("." + ".".join(chain) if chain else "")
        py_check = (
            f"from {root_module} import {from_name} as _m\n"
            f"obj = _m\n"
            + "".join(f"obj = getattr(obj, {seg!r}); assert obj is not None\n" for seg in chain)
        )
    r = subprocess.run(
        ["conda", "run", "-n", env_name, "python", "-c", py_check],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        stderr_first = (r.stderr or "").splitlines()[-1] if r.stderr else "<no stderr>"
        api_errors.append(f"  - {expr}: {stderr_first}")

if api_errors:
    raise SystemExit(
        "Phase 3.6 API surface check FAILED — rolling back. The worker references "
        "module attributes that do not exist in the installed package. Fix the worker "
        "(typically: re-do Phase 1 API discovery and pick correct function names) and "
        "re-create.\n" + "\n".join(api_errors)
    )

print(f"Phase 3.6 PASS — {len(attr_chains)} module.attribute chain(s) verified.")

# MANDATORY MARKER — write the verified-chain record so the driver and
# health_check can audit whether this tool went through Phase 3.6.
# Always written when tool_creation_enabled=True (which is required to be
# in this phase at all).
from pathlib import Path as _P36
import json as _json36
from datetime import datetime as _dt36
_audit_path = TOOLS_USER_DIR / f".knowledge/{tool_id}/api_audit.json"
_audit_path.parent.mkdir(parents=True, exist_ok=True)
_audit_path.write_text(_json36.dumps({
    "ts": _dt36.now().isoformat(),
    "tool_id": tool_id,
    "n_chains_verified": len(attr_chains),
    "errors": api_errors,
    "passed": len(api_errors) == 0,
}, indent=2))
```

**Allowed opt-out**: a worker line ending in `# noqa: P26` (or `# noqa:P26`)
exempts ONLY that line's getattr chain from the check. Use sparingly, and
only for legitimately-dynamic attribute access (plugin systems, optional
features). A line like `obj.compute_dynamic(method=name)  # noqa: P26` is
acceptable.

**On failure**: rollback the in-progress tool — same rollback path as Phase
3.5 / 3.5b — and re-attempt creation. Do NOT register a broken tool.

(Per Top-Level HARD RULE #12: every `module.attribute` chain in the worker
MUST resolve via getattr-chain inside the env. Phase 3.6 enforces this
before registration.)
