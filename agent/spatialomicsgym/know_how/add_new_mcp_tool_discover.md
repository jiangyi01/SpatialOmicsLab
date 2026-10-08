# Adding New MCP Tools - Phase 1: Discover

## Metadata
- Authors: SpatialOmicsLab
- Version: 2.0
- Category: tool_creation

## Overview
Phase 1 of creating a new MCP tool from a GitHub or paper link: identify the package behind the link, read its real API rather than guessing it, decide which of its functions become MCP tools, and harvest its tutorials for worked parameter values. Also holds the error taxonomy and retry ladder that every later phase refers back to.

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
from tools_user.knowledge_manager import CONDA_ENVS_DIR, KNOWLEDGE_DIR
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

## Phase 1: Discover

### Phase 1.0: Memory consult (MODE-GATED — MANDATORY when memory_enabled=True)

Before fetching the README, consult the long-term memory for any prior
attempts on this same `source_url`. Memory is keyed by source_url, not
tool name, so it survives tool_id drift.

=== MANDATORY GATE — execute exactly per the contract below ===

If `default_config.memory_enabled` is True, you **MUST** execute the block
below in full. The driver verifies a marker file (`.last_op`, inside the memory
store — `tools_user/.memory/` unless the store has been relocated; always take it
from `mm.root`, never hardcode the path) was written with mtime >= the start of
this `agent.go()`. If you skip the
block, the driver will log a `memory_mode_not_exercised` issue.

If `default_config.memory_enabled` is False, you **MUST NOT** execute the
block (no-op, no print, no marker). The driver does not check the marker
in this case.

**Hard rule**: if memory is enabled but `MemoryManager` cannot be imported
(missing module / file-corruption / HMAC mismatch), STOP and rollback the
tool — do NOT silently swallow the exception. A working memory subsystem
is a precondition for `memory_enabled=True`.

=== END MANDATORY GATE ===

```python
from spatialomicsgym.config import default_config as _cfg

memory_hints = {}
if getattr(_cfg, "memory_enabled", False):
    # MANDATORY: import MUST succeed when mode is on. Don't try/except this.
    from tools_user.memory_manager import MemoryManager
    mm = MemoryManager.get()
    memory_hints = mm.read_short_term(github_url) or {}
    if memory_hints.get("attempts"):
        print(f"[memory] {len(memory_hints['attempts'])} prior attempts for {github_url}")
        # "" unless the best attempt SUCCEEDED -- never announce a failed one as a winning strategy
        hint_text = mm.format_hint_for_prompt(github_url)
        print(hint_text or "[memory] none of them succeeded; no recipe to reuse")
    else:
        print(f"[memory] no prior attempts on {github_url}")
    # MANDATORY MARKER — driver greps this file post-agent.go()
    # mm.root, not a hardcoded path: the marker belongs to the store, wherever it is.
    import json as _json
    from datetime import datetime as _dt
    _marker = mm.root / ".last_op"
    _marker.parent.mkdir(parents=True, exist_ok=True)
    _marker.write_text(_json.dumps({
        "ts": _dt.now().isoformat(),
        "tool_id": tool_id,
        "source_url": github_url,
        "op": "phase_1_0_consult",
    }))
```

The hints are advisory. Consider the winning strategy if it exists, but
do not skip Phase 1 discovery — verify the install path independently.

### Phase 1.1: README + repo inspection

1. Fetch README from the GitHub URL
2. **Shallow-clone the repo and read its install manifest** — do NOT rely only on the README.
   The README often says "pip install X" but the real install path is in `setup.py` / `pyproject.toml`
   / `environment.yml` / `install.sh` / CI workflow files. Missing this step is what caused SpaGE,
   SpaVAE and SpaDecon to hit `pip install git+...` errors in past runs.
3. Extract: tool name, tool_id (lowercase), language (Python/R), **canonical install command**
   (see "Canonical install discovery" below), main module, main function, parameters, task type.
4. Check conflicts: tool_id vs existing tools in both config files.
5. Check env: `user_{tool_id}` must not already exist in `CONDA_ENVS_DIR`.

### README command extraction — READ THE AUTHOR'S ACTUAL INSTRUCTIONS FIRST

The single most reliable install path is what the repository's own README / docs
tell you to run. Before you consult the priority table below, extract every
install-looking command from the README and treat those as ground truth. Past
runs registered tools successfully ONLY when STCoscientist honoured an author's specific
hint (hdWGCNA needed `ref="dev"` — missing that caused 30 minutes of dep-mesh
thrashing; Stereopy needed `pip install stereo` alt name; Seurat-family tools
need conda-preinstalled `r-sf`/`r-terra` to compile).

Run this extraction BEFORE anything else in Phase 1:

```python
import re as _re
readme_cmds = []

# Python: pip / pipx / uv / conda
for m in _re.finditer(r"(?:pip|pipx|uv|pip3)\s+install\s+[^\n\r;`]+", readme_text or ""):
    readme_cmds.append(("py_install", m.group(0).strip()))
for m in _re.finditer(r"conda\s+install[^\n\r;`]*", readme_text or ""):
    readme_cmds.append(("conda_install", m.group(0).strip()))
for m in _re.finditer(r"mamba\s+install[^\n\r;`]*", readme_text or ""):
    readme_cmds.append(("mamba_install", m.group(0).strip()))

# R: install.packages / remotes / devtools / BiocManager, with ref=/branch=/@tag
for m in _re.finditer(
    r'(?:remotes|devtools)::install_github\s*\(\s*[\'"]([^\'"]+)[\'"](?:\s*,\s*(?:ref|branch|tag)\s*=\s*[\'"]([^\'"]+)[\'"])?',
    readme_text or ""
):
    readme_cmds.append(("r_install_github", {"repo": m.group(1), "ref": m.group(2) or None}))
for m in _re.finditer(r'BiocManager::install\s*\(\s*[\'"]([^\'"]+)[\'"]', readme_text or ""):
    readme_cmds.append(("r_biocmanager", {"pkg": m.group(1)}))
for m in _re.finditer(r'install\.packages\s*\(\s*[\'"]([^\'"]+)[\'"]', readme_text or ""):
    readme_cmds.append(("r_install_packages", {"pkg": m.group(1)}))

# CLI: fenced bash blocks with build commands
for m in _re.finditer(r"```(?:bash|sh|shell)\n([\s\S]+?)\n```", readme_text or ""):
    if _re.search(r"\b(make|cargo|go build|cmake|configure)\b", m.group(1)):
        readme_cmds.append(("shell_build", m.group(1).strip()[:500]))

# Docker / container (rare but real)
for m in _re.finditer(r"docker\s+(?:run|pull)[^\n\r;`]*", readme_text or ""):
    readme_cmds.append(("docker", m.group(0).strip()))

# Persist the harvest so Phase 2 / 3 have full context
(KNOWLEDGE_DIR / tool_id).mkdir(parents=True, exist_ok=True)
(KNOWLEDGE_DIR / tool_id / "readme_install_cmds.json").write_text(
    json.dumps(readme_cmds, indent=2)
)
print(f"Extracted {len(readme_cmds)} install-related commands from README:")
for kind, cmd in readme_cmds[:10]:
    print(f"  [{kind}] {cmd if isinstance(cmd, str) else json.dumps(cmd)}")
```

**Decision rule after extraction:**

1. If the README has ANY command that unambiguously names the install path
   (e.g. `remotes::install_github("smorabit/hdWGCNA", ref="dev")`), use that
   command literally. Do NOT substitute a different strategy.
2. If the README lists MULTIPLE options (e.g. pip + conda + source), prefer the
   order: conda-forge/bioconda > PyPI > git+pip > source (less prone to compile
   issues in a fresh env).
3. If the README has NO specific install command, fall through to the priority
   table below ("Canonical install discovery").
4. If the README contradicts what you'd otherwise guess (e.g. repo name is
   "Stereopy" but README says `pip install stereo`), trust the README.

Log the decision in `install_plan.json` with a `source: "readme" | "priority_table" | "vendor_fallback"` field so a post-mortem can see why STCoscientist picked what it picked.

### Canonical install discovery — ALWAYS do this before writing Phase 2

When STCoscientist receives a GitHub URL, the install strategy MUST be derived from what the repo
itself declares, in this order of precedence (stop at the first one that succeeds):

| Priority | Signal | Canonical install command |
|----------|--------|----------------------------|
| 1 | Published on PyPI (package name in README install section, or `grep "^name = " pyproject.toml`, or the README literally says `pip install <pkg>`) | `pip install <pkg>` inside `user_{tool_id}` |
| 2 | R package on CRAN (README: "Available on CRAN" or tests pass `install.packages(...)` from CRAN) | `Rscript -e 'install.packages("<pkg>", repos="https://cloud.r-project.org")'` |
| 3 | R package on Bioconductor (README mentions `BiocManager::install(...)`, or has a `DESCRIPTION` with `biocViews:` field) | `BiocManager::install("<pkg>", ask=FALSE, update=FALSE)` |
| 4 | Bundled conda env: repo has `environment.yml` / `environment.yaml` at repo root | `conda env update -n user_{tool_id} -f environment.yml` |
| 5 | GitHub-only Python package WITH `setup.py` / `pyproject.toml` / `setup.cfg` | `pip install git+https://github.com/{org}/{repo}@{commit_or_tag}` |
| 6 | GitHub-only R package with `DESCRIPTION` file | `remotes::install_github("{org}/{repo}")` |
| 7 | Install script at repo root (`install.sh`, `INSTALL`, `Makefile` with `install` target) | Read the script, extract its *actual* commands, re-run them inside `user_{tool_id}` |
| 8 | None of the above — repo is source code without packaging | **Vendor the source** (see Phase 2 "From GitHub" → vendor fallback) — do NOT guess |

**Execute this check literally, don't skip it:**

```python
import os, re as _re, subprocess, tempfile, json
from pathlib import Path

tmp = tempfile.mkdtemp(prefix=f"{tool_id}_probe_")   # kept: Phase 2 strategies 4/7 run files from it
# Shallow clone — only the tip, no history, no LFS, fast
subprocess.run(
    ["git", "clone", "--depth=1", "--no-tags", "--filter=blob:none",
     f"https://github.com/{org}/{repo}", tmp],
    check=True, timeout=180,
)
root = Path(tmp)

# Walk the priority table above. Save findings to install_plan for Phase 2.
install_plan = {
    "strategy": None,
    "command": None,
    "notes": [],
    # P27: source_url MUST be recorded so rollback() can recover it from
    # .knowledge/{tid}/install_plan.json and write a memory record on failure.
    "source_url": github_url,
    "tool_id": tool_id,
    "source_checkout": tmp,
}
plan_path = KNOWLEDGE_DIR / tool_id / "install_plan.json"

def _declared(fname, pattern):
    f = root / fname
    m = _re.search(pattern, f.read_text(errors="replace"), _re.M) if f.exists() else None
    return m.group(1).strip() if m else None

# The names the package declares. tool_id is the last resort, never a guessed pip name.
pkg_guess = (_declared("pyproject.toml", r"^name\s*=\s*['\"]([^'\"]+)")
             or _declared("setup.cfg", r"^name\s*=\s*(\S+)")
             or _declared("setup.py", r"name\s*=\s*['\"]([^'\"]+)") or tool_id)
r_pkg_name = _declared("DESCRIPTION", r"^Package:\s*(\S+)")   # Phase 2's diagnose_r_load reads it
for name in ("pyproject.toml", "setup.py", "setup.cfg"):
    if (root / name).exists():
        install_plan["notes"].append(f"has Python packaging: {name}")
        install_plan["strategy"] = "pip_git"
        install_plan["command"] = f"pip install git+https://github.com/{org}/{repo}"
        break
if not install_plan["strategy"] and (root / "DESCRIPTION").exists():
    desc = (root / "DESCRIPTION").read_text()
    if "biocViews:" in desc:
        install_plan["strategy"] = "bioconductor"
        install_plan["command"] = f"BiocManager::install(\"{r_pkg_name}\", ask=FALSE)"
    else:
        install_plan["strategy"] = "install_github"
        # CRITICAL: honour README-specified ref= / branch= / @tag hints before
        # defaulting to HEAD. Some R pkgs (hdWGCNA, Seurat-wrappers, etc.)
        # publish install instructions that explicitly target a dev branch
        # or a stable tag — ignoring that causes library() load failures
        # because the main branch may use newer API than what's actually
        # working / tested.
        readme_install_ref = None
        if readme_text:
            # Patterns: install_github("org/repo", ref="dev") OR @dev OR branch="main"
            m_ref = _re.search(r"install_github\s*\(\s*['\"]%s/%s['\"]\s*,\s*ref\s*=\s*['\"]([^'\"]+)['\"]" %
                               (_re.escape(org), _re.escape(repo)), readme_text)
            if not m_ref:
                m_ref = _re.search(r"install_github\s*\(\s*['\"]%s/%s@([^'\"]+)['\"]" %
                                   (_re.escape(org), _re.escape(repo)), readme_text)
            if not m_ref:
                m_ref = _re.search(r"install_github\s*\(\s*['\"]%s/%s['\"]\s*,\s*branch\s*=\s*['\"]([^'\"]+)['\"]" %
                                   (_re.escape(org), _re.escape(repo)), readme_text)
            if m_ref:
                readme_install_ref = m_ref.group(1)

        if readme_install_ref:
            install_plan["command"] = (
                f"remotes::install_github(\"{org}/{repo}\", ref=\"{readme_install_ref}\", "
                "dependencies=TRUE, upgrade=\"never\")"
            )
            install_plan["notes"].append(f"README specifies ref={readme_install_ref!r}")
        else:
            install_plan["command"] = (
                f"remotes::install_github(\"{org}/{repo}\", "
                "dependencies=TRUE, upgrade=\"never\")"
            )
if not install_plan["strategy"]:
    for env_file in ("environment.yml", "environment.yaml", "conda_env.yml"):
        if (root / env_file).exists():
            install_plan["strategy"] = "conda_env_file"
            install_plan["command"] = f"conda env update -n user_{tool_id} -f {root / env_file}"
            break
if not install_plan["strategy"]:
    for sh in ("install.sh", "INSTALL", "Makefile"):
        if (root / sh).exists():
            install_plan["strategy"] = "install_script"
            install_plan["command"] = (
                f"bash {root / sh}"
                if sh.endswith('.sh') else
                f"make -C {root} install"
            )
            install_plan["notes"].append(f"read {sh} before running — adapt to the env")
            break
if not install_plan["strategy"]:
    install_plan["strategy"] = "vendor_source"
    install_plan["command"] = f"extract zip of repo into tools_user/vendor_{tool_id}/"
    install_plan["notes"].append("no packaging found — VENDOR, do not pip install")

# Persist for Phase 2 + knowledge backup BEFORE the optional probe: rollback() reads source_url here.
plan_path.parent.mkdir(parents=True, exist_ok=True)
plan_path.write_text(json.dumps(install_plan, indent=2))

# Also check PyPI independently -- sometimes the GitHub mirror lags behind a released package
try:
    r = subprocess.run(["pip", "index", "versions", pkg_guess], capture_output=True, text=True, timeout=30)
    if "Available versions" in r.stdout:
        install_plan["notes"].append(f"ALSO on PyPI -- consider `pip install {pkg_guess}` "
                                      "(may be more stable than git)")
        plan_path.write_text(json.dumps(install_plan, indent=2))
except (OSError, subprocess.TimeoutExpired) as e:
    print(f"PyPI probe skipped: {e}")
print(json.dumps(install_plan, indent=2))
```

**Rules of engagement for using `install_plan.strategy` in Phase 2:**

- Always TRY the canonical strategy first; a single attempt with the command literally derived
  from the repo is much more likely to succeed than guessing.
- If the canonical install fails (compile error, version clash, network), record the exact
  failure in `install_plan.notes` AND try the next lower-priority strategy in the table —
  don't re-run the same failing command with slightly different flags.
- Only fall back to `vendor_source` when all of the following are true:
  (a) no packaging file exists, (b) `pip install git+...` produced
  *"does not appear to be a Python project"*, and (c) no `environment.yml` / `install.sh`.
- The vendoring fallback is NEVER the first thing you try — it's the last resort. Vendored
  tools have worse upgradeability and need a `sys.path.insert` in their worker.

**Never:** guess at a `pip install <pkg>` name by lowercasing the repo name. Always verify
the package name against PyPI (`pip index versions <pkg>`) or against the `name = …` field
in `pyproject.toml` / the `packages = […]` in `setup.py`. Past runs burned multiple minutes
trying `pip install spadecon` / `pip install SpaDecon` / `pip install git+...` because the
repo had no such pip name at all.

### Tool ID Naming Rules

- **Use the package name directly** as tool_id: `squidpy`, `sopa`, `scanpy`
- Do NOT append suffixes like `_stats`, `_spatial`, `_analysis`
- Do NOT use hyphens — use underscores only
- tool_id must match regex: `^[a-z][a-z0-9_]{1,29}$`
- The tool_id is used to name: conda env (`user_{tool_id}`), worker file (`{tool_id}_worker.py`), server file (`{tool_id}_mcp_server.py`), knowledge dir (`.knowledge/{tool_id}/`)
- **Consistency is critical** — all downstream operations (modification, health check, trash, restore) use this exact tool_id

Examples:
- `https://github.com/scverse/squidpy` → tool_id = `squidpy` (NOT `squidpy_stats`)
- `https://github.com/gustaveroussy/sopa` → tool_id = `sopa` (NOT `sopa_spatial`)

**TIE-BREAKER for repo basenames with language suffixes** — strip `_py` / `_python` /
`_r` / `_R` / `-py` / `-python` / `-r` / `-R` BEFORE applying the
`[^a-z0-9]→_` rule. Examples:

- `Banksy_py`     → tool_id = `banksy`     (NOT `banksy_py`)
- `hdWGCNA-R`     → tool_id = `hdwgcna`    (NOT `hdwgcna_r`)
- `spaVAE`        → tool_id = `spavae`
- `liana-py`      → tool_id = `liana`      (NOT `liana_py`)
- `STAGATE_pyG`   → tool_id = `stagate`    (NOT `stagate_pyg`)

The stripped, lowercased name is canonical_tid. **Echo your chosen tool_id in
your first plan message AND grep `tools_user/install_log.json` for that exact
id** to confirm no clash before any registration.

**Why**: a previous benchmark observed STCoscientist alternating between `banksy` and
`banksy_py` across attempts of the same tool (drift), causing duplicate
state and orphan envs/files. Locking on the stripped canonical form makes
the tool_id deterministic across attempts and avoids the drift.

### Language Detection and Hybrid Env Detection

**Three categories — detect which one BEFORE creating the env:**

**Category A: Pure Python** — no R dependency
- Default if no R signals found
- Env: `conda create -n user_{tool_id} python=3.11`

**Category B: Pure R** — R package with no Python
- Signals: "R package", "Bioconductor", "CRAN", `install.packages()`, `.R` files only
- Env: `conda create -n user_{tool_id} r-base=4.3 r-essentials -c conda-forge`

**Category C: Hybrid Python+R** — Python package that uses R via rpy2
- Signals: `rpy2` in requirements/imports, `mclust`, `Seurat`, `robjects` in source code
- Common pattern: Python spatial tools that use R clustering (mclust, louvain via igraph)
- Env needs BOTH Python AND R + rpy2 bridge
- This is the MOST COMPLEX case — requires careful version matching

**How to detect hybrid (execute this after Phase 1 discovery):**
```python
# Check if the package has R dependencies
hybrid_signals = []

# 1. Check README for R mentions
r_keywords = ["rpy2", "mclust", "Rscript", "R package", "install.packages",
              "BiocManager", "Seurat", "robjects", "r_packages"]
for kw in r_keywords:
    if kw.lower() in readme_text.lower():
        hybrid_signals.append(kw)

# 2. Check requirements.txt/setup.py for rpy2
if "rpy2" in install_requirements:
    hybrid_signals.append("rpy2 in requirements")

# 3. After install, check package source for R imports
r = subprocess.run(["conda", "run", "-n", env_name, "python", "-c",
    f"import {module}; import inspect; src = inspect.getsource({module}); "
    f"print('HAS_RPY2' if 'rpy2' in src else 'NO_RPY2')"],
    capture_output=True, text=True, timeout=30)
if "HAS_RPY2" in r.stdout:
    hybrid_signals.append("rpy2 in source code")

if hybrid_signals:
    print(f"HYBRID detected: {hybrid_signals}")
    language = "hybrid"  # Python + R
else:
    language = "python"  # or "R" based on earlier detection
```

### Hybrid Python+R Environment Creation

**If hybrid detected, install R + rpy2 + R packages INTO the same conda env:**

```python
env_name = f"user_{tool_id}"

# Step 1: Create env with Python (already done in Phase 2)
# Step 2: Add R and rpy2 to the SAME env
subprocess.run(["conda", "install", "-n", env_name, "-c", "conda-forge",
                "r-base=4.3", "rpy2", "-y"], check=True, timeout=1800)

# Step 3: Install R packages that the tool needs
# Common R packages needed by spatial Python tools:
r_packages_to_try = []
if "mclust" in str(hybrid_signals).lower() or "mclust" in readme_text.lower():
    r_packages_to_try.append("mclust")
if "seurat" in str(hybrid_signals).lower():
    r_packages_to_try.append("Seurat")

rscript = interp_path(CONDA_ENVS_DIR / env_name, "Rscript")
for rpkg in r_packages_to_try:
    subprocess.run([rscript, "-e",
                    f'install.packages("{rpkg}", repos="https://cloud.r-project.org", quiet=TRUE)'],
                   timeout=600)

# Step 4: Verify rpy2 bridge works
r = subprocess.run(["conda", "run", "-n", env_name, "python", "-c",
                    "import rpy2.robjects as ro; ro.r('cat(\"R_OK\\n\")'); print('PY_OK')"],
                   capture_output=True, text=True, timeout=30)
assert "R_OK" in r.stdout and "PY_OK" in r.stdout, f"rpy2 bridge failed: {r.stderr}"
print("Hybrid env verified: Python + R + rpy2 bridge working")
```

**Version matching gotchas for hybrid envs:**
| Issue | Cause | Fix |
|-------|-------|-----|
| `rpy2` import error | R version mismatch with rpy2 | Install both via conda (not pip for rpy2) |
| `mclust` segfault | rpy2 numpy2ri bug | Use `robjects.r()` calls, avoid auto-conversion |
| `library(X)` fails in rpy2 | R package installed for wrong R version | Use conda `r-{pkg}` instead of `install.packages()` |
| `R_HOME not set` | conda env R not activated | Use `conda run -n env` (sets R_HOME automatically) |
| `libR.so not found` | LD_LIBRARY_PATH missing R lib | Inside the env: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/R/lib:$LD_LIBRARY_PATH"` |

**When writing the worker for hybrid tools:**
- Always wrap rpy2 imports in try/except with a clear fallback message
- Use `robjects.r()` for R code execution (not numpy2ri auto-conversion)
- Test R functionality in the micro-test
- Document in the MCP tool description which R packages are optional vs required

## Error Taxonomy & Retry Ladder

When any install / import / test command fails, classify the error before retrying.
Blind retry of the same command is a past-observed failure mode that burned minutes on
SpaTopic (libpng compile), SpaGE (no setup.py), SpaDecon (wrong pip name). Use the
table below:

| Error class | Detected by (substring match on stderr) | Response |
|-------------|-----------------------------------------|----------|
| `NET_FAIL` | "connection refused", "timed out", "name or service not known", "503", "EAI_AGAIN" | exponential backoff (15 s → 45 s → 120 s), max 3 retries |
| `NO_PACKAGING` | "does not appear to be a Python project", "setup.py not found", "pyproject.toml not found" | drop to next install strategy in priority table — NEVER retry the same pip command |
| `VERSION_CLASH` | "ResolutionImpossible", "conflicting dependencies", "incompatible" | try strategy 4 (environment.yml) if available; else pin python=3.10 or 3.9 and retry Phase 2 from start |
| `COMPILE_FAIL` | "compilation failed for package", "fatal error:", "Cannot find -l", "gcc: error" | install missing system libs (libpng/libjpeg-turbo/libtiff/hdf5/gsl/libxml2 via conda-forge) in the same env, then retry ONCE |
| `LAZY_LOAD_FAIL` (R) | "lazy loading failed", "there is no package called" | see "R iterative load diagnostic" below — parse the specific missing pkg name, install it, retry up to 5 times |
| `MISSING_MODULE` | "ModuleNotFoundError" on an import that looked fine | Phase 3.5 bug — scan imports harder (use AST, not string grep) |
| `CLI_TOOL` | "command not found", tool has no Python API | wrap as CLI worker — see Phase 3 CLI template; set_meta(cli_name, subcommand, extra_args). If the input is TSV/non-h5ad, mark `real_data_testable: partial`. Affects FICTURE. |
| `WRONG_NAME` | "No matching distribution found" | rerun Phase 1 canonical-install discovery; the pip name ≠ the repo name |
| `PYG_WHEEL` | "torch_sparse", "torch_scatter", "pyg-lib", "GLIBC", "data.pyg.org" | match PyG wheels to the installed torch version + CPU via the PyG wheel index URL (`-f https://data.pyg.org/whl/torch-<VER>+cpu.html`); else fall back to the `pyg` conda channel. Affects NicheCompass. |
| `BUILD_FAIL_ALTNAME` | "Failed to build", "subprocess-exited-with-error" on a package that DOES exist on PyPI | the primary distribution has a broken sdist/wheel — try the package's ALTERNATE PyPI name (e.g. `stereopy` → `stereo`) before dropping to git+pip, and record `verified_module` accordingly. Affects Stereopy. |
| `KNOWN_HANG` | install/compile makes no progress past the env-create timeout (esp. R/Bioconductor source compiles) | do NOT loop. Abort after one bounded attempt, `rollback(tool_id)`, and call `MemoryManager.get().mark_known_hang(url, reason)` so future runs skip fast. Affects Voyager (R/Bioc infinite compile on CPU). |
| `UNKNOWN` | anything else | stop, capture the full stderr into knowledge dir, rollback. Do NOT guess. |

**Per-strategy retry budget:** each install strategy gets at most **one** retry after
mitigation. If it still fails, move to the next strategy in the priority table —
do not loop on the same strategy.

**Global retry budget:** 3 strategy attempts total per Phase 2. If all three fail,
`rollback(tool_id)` and report which three strategies were tried.

```python
import re, subprocess, time

ERR_SIGNATURES = [
    ("NET_FAIL",      re.compile(r"connection refused|timed out|name or service not known|503|EAI_AGAIN", re.I)),
    ("NO_PACKAGING",  re.compile(r"does not appear to be a Python project|setup\.py not found|pyproject\.toml not found", re.I)),
    ("VERSION_CLASH", re.compile(r"ResolutionImpossible|conflicting dependencies|incompatible", re.I)),
    ("COMPILE_FAIL",  re.compile(r"compilation failed for package|fatal error:|Cannot find -l|gcc: error", re.I)),
    ("LAZY_LOAD_FAIL",re.compile(r"lazy loading failed|there is no package called", re.I)),
    ("MISSING_MODULE",re.compile(r"ModuleNotFoundError", re.I)),
    ("WRONG_NAME",    re.compile(r"No matching distribution found", re.I)),
    ("PYG_WHEEL",     re.compile(r"torch[-_]sparse|torch[-_]scatter|pyg[-_]lib|data\.pyg\.org|GLIBC", re.I)),
    ("BUILD_FAIL_ALTNAME", re.compile(r"Failed to build|subprocess-exited-with-error", re.I)),
]
# KNOWN_HANG is detected by elapsed-time (no stderr signature): if an install/compile
# exceeds the env-create timeout with no progress, abort, rollback, and
# MemoryManager.get().mark_known_hang(url, reason). Never loop a hanging source compile.

def classify_err(stderr: str) -> str:
    for label, pat in ERR_SIGNATURES:
        if pat.search(stderr):
            return label
    return "UNKNOWN"

def try_install(cmd: list[str], timeout_s: int = 1800) -> tuple[bool, str, str]:
    """Returns (ok, err_class, stderr). Never raises."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if r.returncode == 0:
            return True, "OK", ""
        return False, classify_err(r.stderr), r.stderr[-2000:]
    except subprocess.TimeoutExpired as e:
        return False, "TIMEOUT", str(e)
```

Use `try_install` everywhere in Phase 2 instead of raw `subprocess.run(..., check=True)`.
The `err_class` returned drives your next action per the table above, not your own
intuition about what "looks similar".

## Phase 1.5: Tutorial Harvest (RECOMMENDED — best-effort, non-blocking)

### Why

README shows install commands; tutorials show the actual *API shape*:
primary entry function, typical input/output types, sensible default
parameters, required secondary inputs. Reading tutorials BEFORE writing
the worker (Phase 3) reliably reduces first-attempt codegen bugs.
Observed in prior runs:
- Spatopic real-data failed because STCoscientist guessed `cell_type_prop_csv` as
  input; the vignette shows the canonical call takes a labeled count
  matrix.
- SpaDecon failed because STCoscientist didn't know `sc_h5ad` was the canonical
  secondary input; README only shows install.

### Hard constraints (UX invariants)

- **Best-effort, non-blocking.** If harvest fails for ANY reason
  (network, malformed notebook, OOM on huge file), catch the exception,
  log to `.knowledge/{tool_id}/tutorials_harvest_error.log`, and
  return empty hints. Creation proceeds without hints. Never abort
  creation because of a tutorial harvest issue.
- **Bounded.** File count, file size, and total extracted bytes are
  all capped. Parsing a 50 MB notebook is a bug, not an aspiration.
- **Soft/hard time caps per UX-4.** Soft: 10 min (warn, continue).
  Hard: 30 min (abort harvest, keep creation going without hints).

### Procedure

```python
import json
from pathlib import Path

MAX_FILES_PER_FOLDER = 8
MAX_TOTAL_FILES      = 16
MAX_FILE_BYTES       = 256 * 1024          # 256 KB per file
MAX_TOTAL_BYTES      = 64 * 1024           # 64 KB extracted snippet corpus

TUTORIAL_FOLDERS_BY_LANG = {
    "python": ["tutorials", "tutorial", "examples", "example",
               "notebooks", "docs", "doc", "tests", "test"],
    "R":      ["vignettes", "tutorials", "examples", "docs", "inst/doc"],
    "any":    [],
}
ALLOW_REMOTE_DOC_HOSTS = ("readthedocs.io", "github.io",
                          "raw.githubusercontent.com")

def harvest_tutorials(repo_dir: Path, tool_id: str,
                      language: str = "python") -> dict:
    """Compact extraction: primary_fn, example_io, param_defaults."""
    import re, subprocess
    candidates = []
    folders = TUTORIAL_FOLDERS_BY_LANG.get(language, []) + ["."]
    for folder in folders:
        d = repo_dir / folder if folder != "." else repo_dir
        if not d.is_dir():
            continue
        per_folder = []
        for pat in ("*.ipynb", "*.py", "*.R", "*.Rmd", "*.qmd", "*.md", "*.rst"):
            per_folder.extend(sorted(d.glob(pat))[:4])              # depth 1
            per_folder.extend(sorted(d.glob(f"*/{pat}"))[:4])       # depth 2
        candidates.extend(per_folder[:MAX_FILES_PER_FOLDER])
    for readme in sorted(repo_dir.glob("README*"))[:2]:
        candidates.append(readme)
    candidates = candidates[:MAX_TOTAL_FILES]

    tool_tokens = _tool_identifier_tokens(repo_dir)  # helpers below

    snippets = []
    total_bytes = 0
    for f in candidates:
        try:
            stat = f.stat()
            if stat.st_size > MAX_FILE_BYTES:
                continue
            if f.suffix == ".ipynb":
                blocks = _extract_ipynb_cells(f)
            elif f.suffix in (".py", ".R"):
                blocks = [f.read_text(errors="replace")[:MAX_FILE_BYTES]]
            elif f.suffix in (".Rmd", ".qmd"):
                blocks = _extract_rmd_code_blocks(f, language)
            elif f.suffix in (".md", ".rst") or f.name.startswith("README"):
                blocks = _extract_fenced_code(f.read_text(errors="replace"), language)
            else:
                continue
            for b in blocks:
                score = _score_snippet(b, tool_tokens)
                if score <= 0:
                    continue
                if total_bytes + len(b) > MAX_TOTAL_BYTES:
                    break
                snippets.append({"file": str(f.relative_to(repo_dir)),
                                 "score": score, "code": b})
                total_bytes += len(b)
            if total_bytes >= MAX_TOTAL_BYTES:
                break
        except Exception:
            pass

    snippets.sort(key=lambda x: -x["score"])
    summary = _summarize_snippets(snippets[:5], tool_tokens)

    hints = {
        "snippets_count": len(snippets),
        "bytes_total": total_bytes,
        "top_snippets": snippets[:5],
        "summary": summary,    # {primary_fn, primary_input_type,
                               #  common_param_defaults, required_secondary_inputs}
    }
    knowledge_dir = (KNOWLEDGE_DIR / tool_id)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    (knowledge_dir / "tutorials.json").write_text(json.dumps(hints, indent=2))
    return hints

# --- helpers (inline for clarity; in practice add to worker_utils) ---

def _extract_ipynb_cells(path: Path) -> list[str]:
    """Parse notebook, return code cell sources. Strips heavy outputs."""
    import json as _json
    nb = _json.loads(path.read_text(errors="replace"))
    return [
        "".join(cell.get("source", [])) if isinstance(cell.get("source"), list)
        else cell.get("source", "")
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]

def _extract_rmd_code_blocks(path: Path, lang: str) -> list[str]:
    import re
    txt = path.read_text(errors="replace")
    lang_tag = "r" if lang == "R" else "python"
    # Fenced blocks like ```{r} ... ```  or ```{python} ... ```
    return re.findall(rf"```\{{{lang_tag}[^}}]*\}}\r?\n([\s\S]*?)\r?\n```", txt, re.I)

def _extract_fenced_code(txt: str, lang: str) -> list[str]:
    import re
    tags = {"python": "(?:python|py)?", "R": "[rR]?"}.get(lang, "")
    return re.findall(rf"```{tags}\r?\n([\s\S]*?)\r?\n```", txt)

def _tool_identifier_tokens(repo_dir: Path) -> set[str]:
    tokens = set()
    for f in ("pyproject.toml", "setup.py", "setup.cfg", "DESCRIPTION"):
        p = repo_dir / f
        if not p.exists():
            continue
        txt = p.read_text(errors="replace")
        for m in _re.finditer(r"(?:name|Package)\s*[:=]\s*['\"]?([A-Za-z0-9_.-]+)", txt):
            tokens.add(m.group(1))
    return tokens

def _score_snippet(code: str, tokens: set[str]) -> int:
    score = 0
    for tok in tokens:
        if tok and tok.lower() in code.lower():
            score += 3
    for kw in ("example", "usage", "quickstart"):
        if kw in code.lower():
            score += 1
    for bad in ("Traceback", "ERROR:"):
        if bad in code:
            score -= 5
    return score

def _summarize_snippets(snippets: list[dict], tokens: set[str]) -> dict:
    """Extract the most-called tool function + its typical kwargs."""
    import ast as _ast, re as _re
    call_counts = {}
    kwarg_examples = {}
    primary_input = None
    secondary_inputs = []
    for s in snippets:
        code = s["code"]
        try:
            tree = _ast.parse(code)
            for node in _ast.walk(tree):
                if not isinstance(node, _ast.Call):
                    continue
                name = ""
                if isinstance(node.func, _ast.Attribute):
                    name = node.func.attr
                elif isinstance(node.func, _ast.Name):
                    name = node.func.id
                else:
                    continue
                # Only count calls whose module is one of this tool's tokens
                mod_src = _ast.unparse(node.func) if hasattr(_ast, "unparse") else ""
                if not any(t.lower() in mod_src.lower() for t in tokens if t):
                    continue
                call_counts[name] = call_counts.get(name, 0) + 1
                for kw in node.keywords:
                    if kw.arg and isinstance(kw.value, (_ast.Constant, _ast.Num, _ast.Str)):
                        kwarg_examples.setdefault(kw.arg, []).append(
                            repr(getattr(kw.value, "value", kw.value)))
        except SyntaxError:
            continue
    primary_fn = max(call_counts, key=call_counts.get) if call_counts else None
    common_defaults = {k: v[0] for k, v in kwarg_examples.items() if len(v) >= 2}
    return {
        "primary_fn": primary_fn,
        "common_param_defaults": common_defaults,
        "required_secondary_inputs": secondary_inputs,  # populated heuristically
    }
```

### Post-harvest verification (guards outdated tutorials)

If `summary.primary_fn` is set, verify it actually exists in the
freshly-created env:

```python
if hints["summary"].get("primary_fn"):
    fn = hints["summary"]["primary_fn"]
    r = subprocess.run(
        ["conda", "run", "-n", env_name, "python", "-c",
         f"import {tool_module}; print(hasattr({tool_module}, {fn!r}))"],
        capture_output=True, text=True, timeout=60,
    )
    if "True" not in r.stdout:
        hints["summary"]["primary_fn_stale"] = True
        hints["summary"]["primary_fn_note"] = (
            f"Tutorial references {fn} but it is not exported by "
            f"{tool_module} in this env. Use runtime introspection in Phase 3."
        )
```

### Integration point

Run after Phase 1 discovery completes + env is created in Phase 2.
The `hints` dict is passed as CONTEXT to Phase 3 codegen — STCoscientist uses
`hints["summary"]["primary_fn"]` as a first-class signal for which
function the worker should call.

Failure modes that fall back gracefully (UX-2):
- Network-down during shallow-clone: `hints = {}`, creation proceeds.
- Notebook bigger than MAX_FILE_BYTES: skip that file, try others.
- No tutorial folders present: `hints = {}`, fallback to README only.
- Rmd/.qmd with non-English comments: parse anyway, unicode-safe.
