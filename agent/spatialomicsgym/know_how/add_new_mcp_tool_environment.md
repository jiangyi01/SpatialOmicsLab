# Adding New MCP Tools - Phase 2: Create the Environment

## Metadata
- Authors: SpatialOmicsLab
- Version: 2.0
- Category: tool_creation

## Overview
Phase 2 of creating a new MCP tool: build the isolated conda environment its worker will run in, install the package and the dependencies it actually needs, and prove the import works before a single line of worker code is written.

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
from tools_user.knowledge_manager import CONDA_ENVS_DIR, KNOWLEDGE_DIR, TOOLS_USER_DIR
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

## Phase 2: Create Environment

**CRITICAL: Always use `conda run -n user_{tool_id}` for ALL commands. NEVER use `source activate`.**

### HARD RULE: no orphan envs

If Phase 2 exhausts its strategy budget without achieving the post-condition (env exists
AND target module imports OR vendor_path populated), you MUST call `rollback(tool_id)`
**before** raising. The rollback removes the half-built conda env so the next run starts
clean. Without rollback, a half-built `user_*` env persists as dead disk and
counts against the env cap for subsequent
attempts.

The pattern is non-negotiable:

```python
strategies_tried = []
installed_ok = False
for strategy in canonical_strategies:   # PyPI, CRAN, Bioc, env.yml, git+pip, …
    ok, err_class, stderr = try_install(strategy.command)
    strategies_tried.append({"strategy": strategy.name,
                             "ok": ok,
                             "err_class": err_class,
                             "stderr_tail": stderr[-400:]})
    if ok and _module_imports(env_name, module):
        installed_ok = True
        break
    # classify via Error Taxonomy and either mitigate (one retry) or move on

if not installed_ok:
    # Persist the diagnostic so a human can see what was tried
    (KNOWLEDGE_DIR / tool_id).mkdir(parents=True, exist_ok=True)
    (KNOWLEDGE_DIR / tool_id / "phase2_failure.json").write_text(
        json.dumps({"tool_id": tool_id, "strategies": strategies_tried}, indent=2)
    )
    # MANDATORY cleanup before raising — prevents the orphan-env class of failures
    rollback(tool_id, keep_knowledge=True)  # keeps the failure diagnostic for post-mortem
    raise RuntimeError(
        f"Phase 2 exhausted all install strategies for {tool_id}; rolled back. "
        f"Tried: {[s['strategy'] for s in strategies_tried]}"
    )
```

Do NOT proceed to Phase 3 without this check. A half-built env + no registration is
strictly worse than a clean rollback: it blocks the name, eats the user_env cap, and
confuses downstream health_check.

### HARD RULE: no orphan registrations

Symmetrically, if Phase 4 reaches its register step but fails (yaml corrupt, schema
validation, two-file consistency check), you MUST call `rollback(tool_id)` — do NOT
leave a half-registered tool (files + env exist, install_log missing or cfg missing).
Past run saw `seagal` land in this state when its `install_log` write succeeded but
the config write didn't, and no cleanup happened.

### HARD RULE: Phase 3/4/5 failures ALSO roll back the conda env

A stricter generalisation of the above two rules. If Phase 2 succeeded (env built) but
any subsequent phase fails — Phase 3 code-generation SyntaxError, Phase 3.5 missing
dep, Phase 4 registration inconsistency, Phase 5 critical test fail — you MUST call
`rollback(tool_id)` which removes the conda env too. A common past failure mode:

- BANKSY Phase 2 PASSED (env `user_pybanksy` built)
- BANKSY Phase 3 FAILED on Unicode em-dash in generated code
- STCoscientist aborted but left `user_pybanksy` conda env sitting as an orphan

That orphan then blocks the next-try `conda create -n user_pybanksy` and consumes the
user-env cap. The rollback function defined in this doc already removes the conda env,
but ONLY if STCoscientist calls it. Any `raise SystemExit(...)` / `raise RuntimeError(...)` on a
failed Phase 3/4/5 must be preceded by `rollback(tool_id, keep_knowledge=True)`.

Defensive try-finally pattern to apply wrapping each phase-change attempt:

```python
try:
    # Phase 3: generate code
    # Phase 3.5: deps audit
    # Phase 4: register
    # Phase 5: tests
    ...
except BaseException:  # includes SystemExit, KeyboardInterrupt
    rollback(tool_id, keep_knowledge=True)  # keep diagnostic files
    raise
```

This ensures zero orphan envs regardless of where in Phases 3-5 the failure lands.

**MANDATORY FORM — Phases 3, 3.5, 4, 5 must live inside the SAME try/except block.**

Don't emit four independent subprocess runs, each followed by its own `raise`
on failure. That leaves the env live when Phase 3 fails but Phase 2 already
built the env. Instead, emit one outer try that spans all of Phase 3→5 and
one single `except BaseException: rollback(tool_id); raise`. Any assertion,
any non-zero subprocess, any `raise RuntimeError` inside that block is caught
by the same except and routed through `rollback(tool_id)`. This is the only
shape that is provably orphan-free. If you're tempted to split the try/except
across phases "for clearer error messages", DON'T — prepend a phase label
inside each assertion instead and keep the outer frame whole.

### Python tool:
```python
import subprocess

env_name = f"user_{tool_id}"

# Create env
subprocess.run(["conda", "create", "-n", env_name, "python=3.11", "-y", "--no-default-packages"],
               check=True, timeout=1800)

# Install package (try pip first, then conda, then git+)
subprocess.run(["conda", "run", "-n", env_name, "pip", "install", package_name],
               check=True, timeout=1800)

# Verify import
result = subprocess.run(["conda", "run", "-n", env_name, "python", "-c", f"import {module}; print('OK')"],
                       capture_output=True, text=True, timeout=30)
assert "OK" in result.stdout, f"Import failed: {result.stderr}"
```

### R iterative load diagnostic (for stubborn `library(X)` failures)

When `library(TOOL_R_PKG)` succeeds on install but fails on load — and the error doesn't
name a specific missing package at the top level — the real cause is usually one of
these patterns hiding 2-3 dependencies deep:

1. **Nested `there is no package called 'Y'`** — a dep of a dep is missing. R's error
   message often only shows the top-level library, not the inner missing pkg.
2. **`unable to load shared object '.../libZ.so'`** — a system library isn't on the
   linker path. Fix is to `conda install` the matching system lib (zlib, hdf5, openssl,
   icu, libiconv, etc.).
3. **`lazy loading failed for package 'W'`** — `W`'s own compilation silently produced
   a broken binary. Usually from a compile warning that got promoted to error, or an
   ABI mismatch with another library.

STCoscientist must use this iterative diagnostic instead of bulk-reinstalling and hoping for the best:

```python
def diagnose_r_load(env_name: str, pkg: str, max_iter: int = 5) -> tuple[bool, list[str]]:
    """Iteratively install any missing R pkg surfaced in library() load failures."""
    history: list[str] = []
    import re as _re, subprocess as _sp
    for attempt in range(max_iter):
        # Run library(pkg) with FULL stderr capture -- no suppression. LOAD_OK only on success, and a
        # non-zero exit otherwise: try() alone makes Rscript exit 0 on a failed load.
        cmd = ["conda", "run", "-n", env_name, "Rscript", "-e",
               f'res <- try({{library({pkg})}}, silent=FALSE); '
               f'if (inherits(res, "try-error")) {{ cat(conditionMessage(attr(res, "condition")), "\\n"); '
               f'quit(status=1) }} else cat("LOAD_OK\\n")']
        r = _sp.run(cmd, capture_output=True, text=True, timeout=120)
        full = r.stderr + r.stdout
        if "LOAD_OK" in r.stdout and r.returncode == 0:
            history.append(f"attempt {attempt}: library({pkg}) OK")
            return True, history

        # Pattern A — "there is no package called 'X'"
        m = _re.search(r"there is no package called[\s\S]{0,10}?['\"\u2018]([^'\"\u2019]+)['\"\u2019]", full)
        if m:
            missing = m.group(1)
            history.append(f"attempt {attempt}: missing R pkg '{missing}', installing")
            # Try CRAN first, then Bioconductor, then GitHub
            rin = f'install.packages("{missing}", repos="https://cloud.r-project.org"); '
            rin += f'if (!requireNamespace("{missing}", quietly=TRUE)) '
            rin += f'  try(BiocManager::install("{missing}", ask=FALSE, update=FALSE), silent=TRUE)'
            _sp.run(["conda", "run", "-n", env_name, "Rscript", "-e", rin],
                    capture_output=True, text=True, timeout=1200)
            continue

        # Pattern B — "unable to load shared object '.../libZ.so(.N)?'"
        m = _re.search(r"unable to load shared object.*?['\"]([^'\"]*lib[^'\"]*\.so[^'\"]*)['\"]", full)
        if m:
            libso = m.group(1).rsplit("/", 1)[-1]
            # Map common R-adjacent .so → conda-forge package name
            so_to_conda = {
                "libxml2": "libxml2", "libssl": "openssl", "libcrypto": "openssl",
                "libhdf5": "hdf5", "libcurl": "curl", "libicui18n": "icu", "libicudata": "icu",
                "libiconv": "libiconv", "libmagick": "imagemagick", "libjpeg": "libjpeg-turbo",
                "libpng": "libpng", "libtiff": "libtiff", "libfreetype": "freetype",
                "libharfbuzz": "harfbuzz", "libpango": "pango", "libcairo": "cairo",
                "libgdal": "gdal", "libgeos": "geos", "libproj": "proj",
            }
            pkg_conda = next((v for k, v in so_to_conda.items() if k in libso), None)
            if pkg_conda:
                history.append(f"attempt {attempt}: missing .so '{libso}', installing {pkg_conda}")
                _sp.run(["conda", "install", "-n", env_name, "-c", "conda-forge",
                         pkg_conda, "-y"], capture_output=True, text=True, timeout=900)
                continue
            history.append(f"attempt {attempt}: missing .so '{libso}' with no known mapping")
            return False, history

        # Pattern C — "lazy loading failed for package 'W'" (W's binary is broken)
        m = _re.search(r"lazy loading failed for package[\s\S]{0,10}?['\"\u2018]([^'\"\u2019]+)['\"\u2019]", full)
        if m:
            broken = m.group(1)
            history.append(f"attempt {attempt}: lazy-load broken for '{broken}', reinstalling with Ncpus=1")
            rin = f'remove.packages("{broken}"); '
            rin += f'install.packages("{broken}", repos="https://cloud.r-project.org", Ncpus=1)'
            _sp.run(["conda", "run", "-n", env_name, "Rscript", "-e", rin],
                    capture_output=True, text=True, timeout=1800)
            continue

        # Pattern D — LAST-RESORT ABI REALIGNMENT (tool-agnostic).
        #
        # When the error doesn't name a specific missing pkg or .so but R
        # still won't load — e.g. generic "package 'X' failed to load",
        # silent segfaults, or "symbol lookup error" — the usual cause is an
        # ABI mismatch in the geospatial/image stack that many bioinformatics
        # R packages transitively depend on (sf -> gdal/geos/proj/udunits2,
        # magick -> imagemagick, Cairo -> cairo/pango/harfbuzz). Force-
        # reinstalling the stack from a single conda-forge solve realigns
        # every shared library to a consistent ABI version.
        #
        # Only attempted once per diagnostic session (gated on `abi_tried`)
        # so we don't loop. If it doesn't fix things, fall through to the
        # unknown-pattern branch which will rollback.
        if attempt >= 2 and not locals().get("abi_tried"):
            history.append(f"attempt {attempt}: unmatched error; force-reinstalling geospatial/image ABI stack")
            _sp.run(["conda", "install", "-n", env_name, "-c", "conda-forge",
                     "-c", "bioconda", "--force-reinstall", "-y",
                     "gdal", "geos", "proj", "udunits2", "libcurl",
                     "imagemagick", "cairo", "pango", "harfbuzz"],
                    capture_output=True, text=True, timeout=1800)
            abi_tried = True
            continue

        # Unknown pattern — don't loop forever
        history.append(f"attempt {attempt}: unrecognised error; tail: {full[-300:]}")
        return False, history

    history.append(f"exhausted {max_iter} attempts")
    return False, history

ok, history = diagnose_r_load(env_name, r_pkg_name, max_iter=5)
(KNOWLEDGE_DIR / tool_id).mkdir(parents=True, exist_ok=True)
(KNOWLEDGE_DIR / tool_id / "r_load_diagnostic.json").write_text(
    json.dumps({"ok": ok, "history": history}, indent=2)
)
if not ok:
    # Per HARD RULE, rollback before raising
    rollback(tool_id, keep_knowledge=True)
    raise RuntimeError(f"R library({r_pkg_name}) failed to load after {len(history)} diagnostic iterations.")
```

Why this works where bulk-reinstall doesn't:
- It reads the *actual* error every iteration, so each install is targeted at one specific gap.
- It distinguishes missing-package vs missing-system-lib vs broken-binary — each needs a different remedy.
- It bounds the retries (5) so STCoscientist can't loop indefinitely on truly unresolvable platform issues.
- It persists a `r_load_diagnostic.json` into knowledge so a human can post-mortem what was tried.

Use this procedure on ANY R install where the post-install `library(X)` check fails.

### R tool:

**Include common system libraries from conda-forge at env-creation time.** Most R CRAN
packages have a thin shim but compile against system libs (`libpng`, `libjpeg-turbo`, `libtiff`,
`libxml2`, `hdf5`, `gsl`) that are NOT pulled in by `r-base` alone. Omitting these causes
opaque `ERROR: compilation failed for package 'png'` / `lazy loading failed` errors deep in
install that then force multi-retry recompile loops. Install them up front:

```python
# R VERSION PINNING: default 4.4 covers modern Bioconductor and Seurat 5
# while remaining compatible with older packages. Some modern packages now
# require R >= 4.4 (e.g. normalization toolkits built against newer
# Bioconductor releases), so a hardcoded 4.3 base silently breaks them.
# If the target DESCRIPTION specifies a higher minimum, override:
import re as _re
r_min = "4.4"
if (root / "DESCRIPTION").exists():
    desc = (root / "DESCRIPTION").read_text()
    m = _re.search(r"Depends:.*R\s*\(\s*>=?\s*(\d+)\.(\d+)", desc, _re.S)
    if m:
        req_major, req_minor = int(m.group(1)), int(m.group(2))
        if (req_major, req_minor) > (4, 4):
            r_min = f"{req_major}.{req_minor}"

# DETECT HEAVY DEP CLASSES from DESCRIPTION's Imports/Depends — pre-install
# the matching conda-forge packages so STCoscientist doesn't re-compile them at runtime.
# Past semla iteration spent ~15 min discovering sf/terra/magick/EBImage deps
# and installing gdal/geos/proj/udunits2/imagemagick. Baking this in once
# saves that time for every R tool in the same class.
needs = {"geospatial": False, "image": False, "seurat": False, "bioc": False,
         "singlecell_bioc": False, "ggplot_graphics": False}
if (root / "DESCRIPTION").exists():
    desc = (root / "DESCRIPTION").read_text()
    def mentions(*names):
        return any(_re.search(rf"\b{_re.escape(n)}\b", desc) for n in names)
    needs["geospatial"] = mentions("sf", "terra", "sp", "rgeos", "rgdal", "stars")
    needs["image"] = mentions("magick", "EBImage", "OpenImageR", "imager")
    needs["seurat"] = mentions("Seurat", "SeuratObject", "SeuratData")
    needs["bioc"] = mentions("BiocManager", "Biobase", "S4Vectors", "IRanges")
    needs["singlecell_bioc"] = mentions("SingleCellExperiment", "SummarizedExperiment",
                                        "DelayedArray", "DESeq2", "edgeR", "limma")
    needs["ggplot_graphics"] = mentions("ggplot2", "patchwork", "ggrepel")

conda_extras: list[str] = []
if needs["geospatial"]:
    # System libs + their R bindings
    conda_extras += ["gdal", "geos", "proj", "udunits2", "sqlite",
                     "r-sf", "r-terra", "r-sp"]
if needs["image"]:
    conda_extras += ["imagemagick", "r-magick"]
if needs["seurat"]:
    # r-seurat pulls its own dep mesh (SeuratObject, Matrix, etc.)
    conda_extras += ["r-seurat"]
if needs["bioc"] or needs["singlecell_bioc"]:
    # BiocManager itself + common single-cell Bioc pkgs via bioconda
    conda_extras += ["bioconductor-biobase", "bioconductor-s4vectors",
                     "bioconductor-summarizedexperiment", "bioconductor-singlecellexperiment"]
if needs["ggplot_graphics"]:
    conda_extras += ["r-ggplot2", "r-patchwork"]

subprocess.run([
    "conda", "create", "-n", env_name,
    f"r-base={r_min}", "r-essentials",
    # Compile-time system libs most CRAN/Bioconductor packages need:
    "libpng", "libjpeg-turbo", "libtiff", "libxml2", "hdf5", "gsl",
    # Graphics font / text-rendering deps (freetype is hit by any
    # Bioconductor graphics-heavy package — Seurat plots, ggplot-based
    # tools, etc. — a missing `ft2build.h` breaks any of these):
    "freetype", "harfbuzz", "fribidi", "pango", "cairo", "fontconfig",
    # Network/compress deps for R pkgs that call out to remote services:
    "openssl", "curl", "libssh2", "zlib", "bzip2", "xz", "zstd",
    # Build toolchain (Linux):
    "gcc_linux-64", "gxx_linux-64", "gfortran_linux-64", "make",
    # Headers for Rcpp-based packages:
    "pkg-config",
    # DETECTED-CLASS extras appended based on the tool's DESCRIPTION —
    # see the `needs` dict above. This block is zero for simple tools and
    # ~5-10 packages for geospatial/image/Seurat/Bioc tools.
    *conda_extras,
    "-c", "conda-forge", "-c", "bioconda", "-y",
], check=True, timeout=3600)

# For Bioconductor-specific packages (SpaNorm, etc.), install BiocManager once:
subprocess.run(["conda", "run", "-n", env_name, "Rscript", "-e",
                'if (!require("BiocManager", quietly=TRUE)) install.packages("BiocManager", '
                'repos="https://cloud.r-project.org")'],
               check=True, timeout=600)

# Install the actual tool. For a CRAN tool:
subprocess.run(["conda", "run", "-n", env_name, "Rscript", "-e",
                f'install.packages("{pkg}", repos="https://cloud.r-project.org", dependencies=TRUE)'],
               check=True, timeout=3000)

# For a GitHub-only R tool (use remotes, not devtools — lighter):
#   Rscript -e 'install.packages("remotes")'
#   Rscript -e 'remotes::install_github("{org}/{repo}", dependencies=TRUE)'

# For a Bioconductor tool:
#   Rscript -e 'BiocManager::install("{pkg}", ask=FALSE, update=FALSE)'

# Verify the library loads — THIS is the pass/fail gate, not the install exit code.
result = subprocess.run(["conda", "run", "-n", env_name, "Rscript", "-e",
                        f'library({pkg}); cat("OK\\n")'],
                       capture_output=True, text=True, timeout=60)
if "OK" not in result.stdout:
    print(f"R library load failed; stderr: {result.stderr[:500]}")
    # Do NOT proceed to Phase 3 with a broken R stack. Fall back to rollback.
```

### From GitHub:

**BEFORE running `pip install git+...`, check that the repo is actually a Python package.**
Many research repos have `.py` code but no `setup.py` / `pyproject.toml`, so `pip install` fails with
*"does not appear to be a Python project"* and wastes 1–3 minutes per retry. Always do the pre-flight check first:

```python
import json, subprocess, zipfile, io, urllib.request, os
from pathlib import Path

# 1. Probe the repo for packaging metadata (HEAD request, no download of files)
org, repo = "ORG", "REPO"  # replace
branch = "main"            # fall back to "master" if this 404s
packaged = False
for f in ("setup.py", "pyproject.toml", "setup.cfg"):
    url = f"https://raw.githubusercontent.com/{org}/{repo}/{branch}/{f}"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as r:
            if r.status == 200:
                packaged = True
                break
    except Exception:
        pass

if packaged:
    # Safe to pip install from GitHub
    subprocess.run(["conda", "run", "-n", env_name, "pip", "install",
                    f"git+https://github.com/{org}/{repo}"],
                   check=True, timeout=1800)
else:
    # 2. Repo has no Python packaging — VENDOR the source instead.
    print(f"{repo} has no setup.py/pyproject.toml — vendoring source into tools_user/vendor_{tool_id}/")

    # P28 — VENDOR-MODE MUST STILL CREATE THE CONDA ENV. The worker will
    # invoke this env via `conda run -n user_{tool_id} python …`. If the env
    # doesn't exist, real-data ALWAYS fails with EnvironmentLocationNotFound.
    # Past failures: ficture, spage — env was never created in the vendor branch.
    if not (CONDA_ENVS_DIR / env_name).exists():
        subprocess.run(["conda", "create", "-n", env_name, "python=3.11", "-y"],
                       check=True, timeout=600)

    zip_url = f"https://github.com/{org}/{repo}/archive/refs/heads/{branch}.zip"
    with urllib.request.urlopen(zip_url, timeout=120) as resp:
        data = resp.read()
    vendor_dir = TOOLS_USER_DIR / f"vendor_{tool_id}"
    vendor_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(vendor_dir)
    # The extracted tree sits under vendor_{tool_id}/{repo}-{branch}/…
    # Record the path for your worker to sys.path.insert(0, …) at runtime.

    # P28 (continued) — install the vendored tool's runtime deps INTO env_name.
    # Parse requirements.txt / setup.py extras / README pip-install lines
    # and `conda run -n env_name pip install <each>`. Without this step the
    # worker imports inside `conda run -n env_name` will fail for every
    # transitive dep (numpy, scipy, anndata, scanpy, torch, …).
    # Worker template MUST `sys.path.insert(0, "<vendor_dir>/{repo}-{branch}")` with the ABSOLUTE
    # vendor_dir printed below -- the worker runs with the agent's CWD, not the repo root.
    print(f"Vendored to {vendor_dir}. Your worker MUST run inside `user_{tool_id}` "
          f"conda env via `conda run -n user_{tool_id} python …`, with vendor_path "
          f"prepended to sys.path. NEVER invoke the worker with the base env's python3.")
```

Set `packaged=False` ⇒ your Phase-3 worker uses `importlib.util.spec_from_file_location` or
`sys.path.insert(0, vendor_dir_str)` to load the tool. Record `vendor_path` in the knowledge JSON
so the worker can find it next invocation.

### After install, check for hybrid R dependency:
```python
# Check if installed package needs R at runtime
r = subprocess.run(["conda", "run", "-n", env_name, "python", "-c",
    f"import warnings; warnings.simplefilter('always'); "
    f"import {module}; "
    f"import sys; print('STDERR:', sys.stderr.getvalue() if hasattr(sys.stderr,'getvalue') else '')"],
    capture_output=True, text=True, timeout=30)

needs_r = False
r_warnings = ["rpy2", "mclust", "R is not", "No rpy2", "install R"]
for kw in r_warnings:
    if kw.lower() in r.stderr.lower():
        needs_r = True
        print(f"Package warns about missing R: '{kw}' found in import warnings")

if needs_r:
    print("Installing R + rpy2 into the same env for full functionality...")
    subprocess.run(["conda", "install", "-n", env_name, "-c", "conda-forge",
                    "r-base=4.3", "rpy2", "-y"], check=True, timeout=1800)
    # Install common R packages (mclust is the most common need)
    rscript = interp_path(CONDA_ENVS_DIR / env_name, "Rscript")
    for rpkg in ["mclust"]:
        subprocess.run([rscript, "-e",
                        f'install.packages("{rpkg}", repos="https://cloud.r-project.org", quiet=TRUE)'],
                       timeout=600)
    # Verify rpy2 bridge
    r2 = subprocess.run(["conda", "run", "-n", env_name, "python", "-c",
                         "import rpy2.robjects as ro; ro.r('cat(\"R_OK\\n\")'); print('PY_OK')"],
                        capture_output=True, text=True, timeout=30)
    if "R_OK" in r2.stdout:
        print("Hybrid env: R + rpy2 installed and verified")
    else:
        print(f"WARNING: rpy2 bridge verification failed: {r2.stderr[:200]}")
        print("Tool will work with Python-only features (e.g., leiden instead of mclust)")
```

On failure: try alternative install methods (pip→conda, different Python version).
After 3 retries exhausted → rollback: `conda remove -n {env_name} --all -y`
