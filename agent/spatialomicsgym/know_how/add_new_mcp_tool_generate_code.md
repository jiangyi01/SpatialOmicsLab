# Adding New MCP Tools - Phase 3: Generate the Code

## Metadata
- Authors: SpatialOmicsLab
- Version: 2.0
- Category: tool_creation

## Overview
Phase 3 of creating a new MCP tool: write the worker script and the MCP server that wraps it, following the argument, path, schema and output-contract rules the generated code has to satisfy to be callable by the agent.

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
from tools_user.knowledge_manager import TOOLS_USER_DIR, save_knowledge, update_knowledge
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

## Phase 3: Generate Code

Write files to `tools_user/` directory ONLY. Never write to `tools/`.

### P30 — HARD RULE: worker MUST NOT import `spatialomicsgym.*`

The worker runs inside `user_{tool_id}` conda env, which **does not have
spatialomicsgym installed**. Past failure: sopa worker had
`from spatialomicsgym.tool.spatial_image_processor import …` → `ModuleNotFoundError:
spatialomicsgym` at runtime → real-data fails despite create succeeding.

**The worker is a standalone script** that consumes `--input <h5ad>` /
`--output-dir` flags and uses ONLY:
- the tool's installed package (in user_{tid} env)
- standard data libs: `anndata`, `scanpy`, `numpy`, `pandas`, `matplotlib`
- vendored source via `sys.path.insert(0, "tools_user/vendor_{tid}/<root>")`

**Forbidden in worker code** (rolls back the tool when seen):
- `import spatialomicsgym`, `from spatialomicsgym…`
- `from MCP_server…`
- Any path that reaches outside the user conda env

If the worker conceptually needs a spatialomicsgym utility (e.g. histology image
export, run_spatial_pipeline), the right answer is to **inline the small
utility** as a function in the worker file, NOT import spatialomicsgym. Phase 3.6
also enforces this — spatialomicsgym imports will fail the API surface check.

### P31 — Worker output MUST be a single JSON dict on stdout

Past failure (spanorm): worker emitted plain text on stdout → driver's
parser failed with `json_parse_fail`. Every worker MUST end with exactly ONE
JSON line on stdout, written by `WorkerOutput` (`worker_utils.py`, beside the
worker in `tools_user/`). `output_files` is a DICT of key -> path: Phase 5
Test 5 and `knowledge_manager._run_test_suite` read it with `.values()`.

```python
import sys
from pathlib import Path
from worker_utils import WorkerOutput

# At end of worker logic, regardless of success/failure:
if error is None:
    out = WorkerOutput(TOOL_ID, task="<task_type>")   # e.g. spatial_clustering / segmentation
    out.add_output_files({str(p.relative_to(output_dir)): str(p) for p in Path(output_dir).glob("**/*")
                          if p.is_file() and p.stat().st_size > 0})
    out.emit()
else:
    WorkerOutput.emit_error(TOOL_ID, str(error), task="<task_type>")
sys.exit(0 if error is None else 1)
```

Wrap the whole worker body in `try/except Exception as error:` so any
runtime exception still produces a parseable JSON line. Never `print` raw
text or `print(traceback.format_exc())` — that breaks the contract.

### P32 — Worker MUST gracefully handle missing input metadata

Past failures (nichecompass: `gp_targets_mask_key ValueError`; spavae: no
output produced because tool needed h5ad obs columns the input doesn't have):
the input h5ad may not contain the obs/var/obsm metadata the tool's primary
function expects. The worker MUST:

1. Discover the tool's required signature (already done in Phase 1.5 / 3).
2. For each REQUIRED arg the tool function takes: probe the input h5ad —
   does the obs / var / obsm key exist? If not, either:
   - **Best**: synthesize a reasonable default (e.g., for `groupby="cell_type"`,
     run scanpy's `sc.tl.leiden` and use the result; for missing
     `gp_targets_mask`, generate a uniform mask of ones).
   - **Acceptable fallback**: write a `degraded.json` next to the output
     stating which metadata was missing AND emit at least ONE non-empty
     output file (e.g., the input h5ad with a `_processed` flag added in
     `obs`) so the real-data PASS check still finds an output.
3. NEVER let the tool crash with `KeyError` / `ValueError` when the cause
   is missing optional input — those are tool-template gaps, not test
   failures, and they make real-data testing useless.

### P33 — HARD RULE: worker `--input` MUST be a file PATH string, never a dict

**Past failure (spage iter3):** STCoscientist's lifecycle invocation called the worker with
`--input` set to the JSON dict produced by `_enrich_prompt_with_spatial_diagnosis`
or by an `auto-diagnosis` step (e.g. `{"mcp_ready": true, "input_h5ad": "/path/...", "modality": "visium"}`).
The worker's argparse received the literal dict-as-string, `os.path.isfile()` returned
False, and the worker crashed with "input file not found" or `JSONDecodeError`.

**The contract — applies to BOTH the worker generator AND every invocation:**

1. **Worker side (generated code) — argparse + validator MUST be:**
   ```python
   import argparse, os, sys, json
   ap = argparse.ArgumentParser()
   ap.add_argument("--input", required=True,
                   help="ABSOLUTE PATH to the input .h5ad/.h5/etc. NEVER a JSON dict.")
   ap.add_argument("--output-dir", required=True)   # args.output_dir; every caller passes --output-dir
   args = ap.parse_args()

   # MANDATORY GUARD — fail fast with a clear message if STCoscientist (or a caller) ever
   # passes a JSON-encoded dict instead of a path. This catches the spage class.
   if args.input.lstrip().startswith("{") or args.input.lstrip().startswith("["):
       sys.stderr.write(
           "P33 violation: --input got a JSON literal, expected a file path. "
           f"Received first 80 chars: {args.input[:80]!r}\n"
       )
       sys.exit(2)
   if not os.path.isfile(args.input):
       sys.stderr.write(f"P33 violation: --input path does not exist: {args.input!r}\n")
       sys.exit(2)

   import anndata as ad
   adata = ad.read_h5ad(args.input)   # load INSIDE the worker, not outside
   ```

2. **Caller side (the STCoscientist lifecycle invocation in Phase 5 / real-data test):**
   - The `--input` value is the literal h5ad PATH — whatever
     `_resolve_real_test_dataset()` returned on THIS host (that is what Phase 5
     calls), or a repaired-copy path produced by `run_spatial_pipeline`'s
     `repaired_h5ad` field. Never type a dataset path from another machine.
   - The output of `_enrich_prompt_with_spatial_diagnosis` (the JSON blob
     auto-injected into your prompt) is **DIAGNOSTIC INFO**, not an input.
     Read its `input_h5ad` field as a PATH STRING; do NOT serialize the
     whole dict and pass it on the CLI.
   - If you obtained a dict like `{"input_h5ad": "/path/x.h5ad", "modality": "visium"}`,
     extract the path with `dict.get("input_h5ad")`, NOT `json.dumps(dict)`.

3. **Don't pass an `AnnData` on the CLI either.** AnnData objects are not
   string-encodable. Always pass a PATH; the worker calls `ad.read_h5ad(path)`
   on its own. If you need a new path (e.g., for a repaired copy), write the
   `AnnData` to a temp file with `adata.write_h5ad(tmp_path)` and pass `tmp_path`.

This rule SHOULD prevent the spage / similar real-data failures where STCoscientist
chains a diagnosis step with a worker invocation and feeds the diagnosis
output as input.

### P34 — HARD RULE: do NOT reimplement `write_atomic` / `atomic_write_json`

**Past failure (spage iter4):** STCoscientist wrote a homegrown `write_atomic()` that called
`tempfile.mkdtemp(prefix=path.name + ".", dir=path.parent)` BEFORE the proper
`tempfile.mkstemp(...)` call. `mkdtemp` creates a DIRECTORY and returns its
path; the directory is never cleaned up. Every call leaked one empty dir
named `tools_user/install_log.json.<8-rand>/`,
`tools_user/spage_worker.py.<8-rand>/`, etc. The driver flagged these as
orphans and FAILed an otherwise-passing attempt (real-data PASSED).

**The rule:**
- DO NOT write your own atomic-write helper. The framework already provides
  `tools_user/trash_manager.py::_atomic_write_json` and
  `tools_user/knowledge_manager.py::_atomic_write_json` — both use the safe
  `path.with_suffix(".json.tmp")` + `os.replace()` pattern.
- If you genuinely need a custom atomic write inside generated code, use
  `mkstemp` ONLY (file, not directory). Never call `mkdtemp` for an
  atomic-replace use case.
- If you DO call `mkdtemp(...)` for any reason, you MUST `shutil.rmtree(d)`
  the resulting directory in a `try/finally` block. Leaking even one
  `mkdtemp` result per tool causes a residue-fail.
- Never use `tempfile.mkdtemp(prefix=path.name + ".", dir=path.parent)` —
  this exact signature was the spage iter4 footgun. If you see this in
  your generated code, delete it and use `mkstemp` instead.

**Detection (you MUST verify):** before emitting `LIFECYCLE_COMPLETE`, run:

```bash
ls tools_user/${TID}_*.* tools_user/install_log.json.* 2>/dev/null
```

Any output = residue from this class of leak. Fix: `rm -rf` those entries
before terminating, OR emit `LIFECYCLE_HARD_ERROR: residue: <list>`.

### P35 — HARD RULE: `permanent_delete` requires `trashed` state, not `active`

**Past failure (novae iter5):** STCoscientist called `permanent_delete(tool_id)` on a
tool whose status was `active` (because the previous lifecycle step was
`restore_tool`). `trash_manager.permanent_delete()` RAISES `TrashStateError`
("... is active. Must trash first") -- `TrashNotFoundError` when there is no
row -- without touching install_log, conda env, or files; under the outer
try that exception rolls back whatever else is in flight. The orphan install_log entry remained,
the driver flagged it as residue, and the attempt FAILED even though
real-data PASSED.

**The rule:** `permanent_delete()` is a SECOND-STAGE delete. It requires the
tool to be in `trashed` state. If the lifecycle does
`soft_delete → restore → permanent_delete`, you MUST call `trash_tool()`
again between `restore` and `permanent_delete`:

```python
from tools_user.trash_manager import trash_tool, restore_tool, permanent_delete

# step 5: soft delete
trash_tool(tool_id)            # active → trashed
# step 6: restore
restore_tool(tool_id)          # trashed → active
# step 6.5 (REQUIRED before step 7):
trash_tool(tool_id)            # active → trashed (again)
# step 7: permanent delete
result = permanent_delete(tool_id)
assert result["success"], f"perm_delete failed: {result.get('errors')}"
```

**Verification:** check `result["success"]` and assert it's True (a partial
delete returns `success: False` with `errors`). If not, log the errors and
emit `LIFECYCLE_HARD_ERROR` — do NOT silently continue.

This prevents the "install_log still has entry (active)" residue class.

### P36 — HARD RULE: emit LANGUAGE_DECISION before any worker code-gen

**Past failure (semla iter6 attempts):** STCoscientist generated a Python worker for
`ludvigla/semla` (a pure R Bioconductor-style package). The Python worker
then failed at real-data because it tried to install/use `leidenalg` /
`python-igraph` instead of using the R `Seurat`/`semla` ecosystem. STCoscientist's
language detection (already required by HARD RULE #5 + P21) was silent —
no audit trail of which language STCoscientist picked or why.

**The rule:** Before generating ANY worker code, STCoscientist MUST print a structured
LANGUAGE_DECISION block to its execution output. This makes the language
choice explicit, auditable by the driver, and visible to STCoscientist itself in
subsequent reasoning loops.

```python
# MANDATORY pre-codegen step. Failing to emit this block is a contract violation.
import subprocess, json
from pathlib import Path

# Probe the cloned repo for language signals
repo_path = TOOLS_USER_DIR / f"vendor_{tool_id}"  # OR wherever you cloned
has_description  = (repo_path / "DESCRIPTION").exists()
has_setuppy      = (repo_path / "setup.py").exists()
has_pyproject    = (repo_path / "pyproject.toml").exists()
n_r_files        = len(list(repo_path.rglob("*.R"))) + len(list(repo_path.rglob("*.r"))) + len(list(repo_path.rglob("*.Rmd")))
n_py_files       = len(list(repo_path.rglob("*.py")))

# Decide language with tie-breaker rules:
# 1. DESCRIPTION present + n_r_files > n_py_files            -> R
# 2. setup.py / pyproject.toml present + n_py_files > 0      -> Python
# 3. n_r_files > 5*n_py_files                                 -> R
# 4. Default                                                  -> Python
if has_description and n_r_files > n_py_files:
    language, ext = "R", ".R"
elif (has_setuppy or has_pyproject) and n_py_files > 0:
    language, ext = "python", ".py"
elif n_r_files > 5 * max(n_py_files, 1):
    language, ext = "R", ".R"
else:
    language, ext = "python", ".py"

print("LANGUAGE_DECISION:")
print(f"  tool_id: {tool_id}")
print(f"  language: {language}")
print(f"  worker_extension: {ext}")
print(f"  evidence:")
print(f"    DESCRIPTION_present: {has_description}")
print(f"    setup.py_present: {has_setuppy}")
print(f"    pyproject.toml_present: {has_pyproject}")
print(f"    n_r_files: {n_r_files}")
print(f"    n_py_files: {n_py_files}")
print(f"  worker_path: tools_user/{tool_id}_worker{ext}")
```

**Consequences of the decision:**
- If `language == "R"`: generate `tools_user/{tid}_worker.R`. Conda env
  must include `r-base` + `r-seurat` (or BiocManager-installed packages).
  Worker invocation: `conda run -n user_{tid} Rscript tools_user/{tid}_worker.R`.
  Do NOT install `leidenalg` / `python-igraph` for an R tool — those are
  Python-side packages used for Leiden clustering in scanpy. R uses
  `Seurat::FindClusters` (Louvain by default; Leiden via algorithm=4
  needing R `igraph` package).
- If `language == "python"`: generate `tools_user/{tid}_worker.py`.
- The `language` field in `install_log.json` MUST match this decision.

**Once decided, do NOT switch languages mid-attempt.** P21 still applies:
language is decided ONCE. If Phase 2 R install fails, do NOT pivot to a
Python wrapper of an R package — that path leads to ImportError /
AttributeError cascades. Roll back instead.

### P37 — HARD RULE: R workers MUST use headless graphics

**Past failure (semla iter7 attempt 2):** STCoscientist correctly detected R, generated
a hybrid Python→R worker that built a Seurat object, then called
`DimPlot(seurat_obj)` / `SpatialDimPlot(seurat_obj)`. Both failed because
they default to an X11/Cairo display device, and the conda env has no
DISPLAY variable, no X11 server, no Cairo library. The Rscript exited
non-zero, marking real-data as FAIL even though the analysis itself
succeeded.

**The rule:** Every generated `*_worker.R` (and every R block called via
`subprocess.run(["Rscript", ...])` from a Python worker) MUST configure
headless graphics at the very top of the script, BEFORE any package loads
that might initialize a graphics device.

```r
# MANDATORY top-of-script for ALL R workers (do not skip)
Sys.setenv(R_DEFAULT_DEVICE = "png")
options(bitmapType = "cairo")        # falls back to Xlib if cairo missing — see below
options(device = function(...) png(filename = tempfile(fileext=".png"), ...))

# Detect cairo availability and switch to ragg if not present (more portable)
if (!capabilities("cairo")) {
  if (requireNamespace("ragg", quietly = TRUE)) {
    options(device = function(...) ragg::agg_png(filename = tempfile(fileext=".png"), ...))
  } else {
    # Fall back to a headless null device — plots are no-ops but won't crash
    options(device = function(...) pdf(file = tempfile(fileext=".pdf"), ...))
  }
}
# Disable any interactive prompts that block headless runs
options(menu.graphics = FALSE)
```

**For plotting calls** — wrap every plot in an explicit device block, do
NOT rely on top-level `print()` of ggplot/Seurat plot objects:

```r
# WRONG (relies on screen device):
DimPlot(seurat_obj)

# CORRECT (explicit headless device):
out_file <- file.path(out_dir, "dimplot.png")
png(filename = out_file, width = 1200, height = 1000, res = 150, type = "cairo")
print(DimPlot(seurat_obj))
dev.off()
```

OR use `ggsave()` which handles devices internally:

```r
p <- DimPlot(seurat_obj)
ggsave(filename = file.path(out_dir, "dimplot.png"), plot = p,
       width = 8, height = 6, dpi = 150, device = "png")
```

**Conda env requirements** — when generating `*_env.yaml` for an R tool
that produces plots:

```yaml
dependencies:
  - r-base
  - r-cairo           # or r-ragg as alternative
  - r-ggplot2         # ggsave
  # (omit r-cairo only if you're CERTAIN no plots are produced)
```

**Verification before marking real-data PASS for an R tool:** the worker's
output_dir MUST contain at least one `.png` / `.pdf` file with size > 0.
If only `.csv` / `.rds` outputs are present, STCoscientist should run a final
`Rscript -e "ggsave(...)"` to produce a plot artifact, OR the tool truly
has no graphical output (acceptable, but document this).

This rule eliminates the "R Seurat pipeline returned non-zero" failure
class that blocks Seurat-based tools (semla, Giotto, hdwgcna with
WGCNA::plot*, etc.) in headless conda environments.

### P38 — HARD RULE: partial-create rollback MUST clean conda env

**Past failure (semla iter8):** STCoscientist ran `conda create -n user_semla` (Phase 2),
then failed to install the `semla` R package (deep geospatial deps missing).
STCoscientist correctly emitted `LIFECYCLE_UNBUILDABLE` but did NOT clean up the
conda env it created. Driver's residue check found `user_semla` still
present and FAILed the attempt.

**The rule:** STCoscientist's rollback path (whenever create fails after env creation
but before successful registration in `install_log.json`) MUST run:

```python
import subprocess
# When emitting LIFECYCLE_UNBUILDABLE or LIFECYCLE_HARD_ERROR after a failed
# Phase 2/3/4, BEFORE the terminator:
subprocess.run(
    ["conda", "env", "remove", "-n", f"user_{tool_id}", "-y"],
    check=False, timeout=300
)
# Also remove any partial files written
import shutil
from pathlib import Path
for p in [
    str(TOOLS_USER_DIR / f"{tool_id}_worker.py"),
    str(TOOLS_USER_DIR / f"{tool_id}_worker.R"),
    str(TOOLS_USER_DIR / f"{tool_id}_mcp_server.py"),
    str(TOOLS_USER_DIR / f"{tool_id}_env.yaml"),
    str(TOOLS_USER_DIR / f"vendor_{tool_id}"),
    str(TOOLS_USER_DIR / f".knowledge/{tool_id}"),
]:
    pp = Path(p)
    if pp.exists():
        shutil.rmtree(pp) if pp.is_dir() else pp.unlink()
```

**This rule applies even when:**
- Tool was never registered in `install_log.json` (so `trash_tool()` /
  `permanent_delete()` are no-ops — they require an entry)
- Tool's `creation_log.json` shows `status="rolled_back"` (registration
  cleanup is automatic, but env cleanup is NOT)
- Phase 2 install failed (env exists with default Python only — STILL
  must be removed)

**Verification:** the residue self-check already greps for `user_<tid>`
in `conda env list`. If it finds the env after rollback, the rollback
was incomplete — STCoscientist must explicitly run `conda env remove` before
emitting the terminator.

This eliminates the "conda env user_<tid> still present" residue class
that fails attempts even when STCoscientist's `LIFECYCLE_UNBUILDABLE` was correct.

### P39 — HARD RULE: enumerate API with `dir()` before assuming scanpy-style namespaces

**Past failure (liana_ iter10):** STCoscientist generated worker calling
`liana.tl.rank_aggregate(...)` and `liana.tl.score_lr(...)`. Both raised
`AttributeError` because LIANA-py exposes its main analysis under
`liana.mt` (methods), not `liana.tl` (transformations as in scanpy).
STCoscientist had checked `hasattr(liana.tl, 'rank_aggregate')` (returned False) but
fell through to the wrong path instead of trying `liana.mt`. Real-data
failed; lifecycle marked UNBUILDABLE despite being a pure API-discovery
miss — the right function exists, just at a different namespace.

**The rule:** Before generating any worker that imports a library you
haven't used before (i.e. NOT scanpy/anndata/numpy/pandas/etc.), you
MUST enumerate its top-level API and PRINT the result. Do NOT default
to scanpy-style namespaces (`tl`/`pp`/`pl`/`tools`/`preprocess`/`plot`)
unless you've verified those attributes exist.

```python
# MANDATORY for unfamiliar libraries (e.g. liana, gaston, novae,
# stagate, sedr, etc.) -- before generating worker. Runs INSIDE user_{tool_id}:
# the library is installed there, not in the agent's env, and its import-time
# code must not run in the agent process. Never pip-install it here to "fix" this.
import subprocess
# For each candidate analysis function name (e.g. "rank_aggregate",
# "cluster", "fit", "predict"), search the top level and every submodule:
candidates = ["rank_aggregate", "score_lr", "liana_pipe"]   # tool-specific
probe = (
    "import importlib, pkgutil\n"
    f"name, candidates = {library_name!r}, {candidates!r}\n"
    "mod = importlib.import_module(name)\n"
    "print('DIR', [a for a in dir(mod) if not a.startswith('_')])\n"
    "subs = [m.name for m in pkgutil.iter_modules(mod.__path__)] if hasattr(mod, '__path__') else []\n"
    "print('SUBMODULES', subs)\n"
    "for sm in [''] + subs:\n"
    "    try:\n"
    "        smod = importlib.import_module(name + ('.' + sm if sm else ''))\n"
    "    except Exception:\n"
    "        continue\n"
    "    for c in candidates:\n"
    "        if hasattr(smod, c):\n"
    "            print('FOUND', smod.__name__ + '.' + c)\n"
)
r = subprocess.run(["conda", "run", "-n", f"user_{tool_id}", "python", "-c", probe],
                   capture_output=True, text=True, timeout=120)
print(r.stdout or r.stderr[-1000:])
```

**Then write the worker referencing only the verified chains.** If the
candidate function isn't found in any submodule, the worker MUST NOT call
it — either find an equivalent or document the gap.

**Common non-scanpy namespace conventions in spatial-omics libs:**
- LIANA-py: `liana.mt.*` (methods), `liana.fun.*` (function pre-built)
- LIANA-r: lives in R as `liana::liana_wrap` etc.
- Squidpy: `sq.gr.*` (graph), `sq.im.*` (image), `sq.pl.*` (plot)
- Spatialdata: `sd.io.*`, `sd.transformations.*`
- Stagate / SEDR: top-level only — no submodule namespace
- Novae: `novae.Novae` class direct, no `tl/pp/pl`
- Banksy_py: `banksy.main.run_banksy`, `banksy.embedding.*`
- Giotto-R: lives in R as `Giotto::createGiottoObject` etc.

When in doubt, run the dir()/iter_modules() probe — it costs <1s and
prevents a 10-minute hallucinated-API real-data failure.

This rule complements Phase 3.6 (which validates chains STCoscientist already
generated) by ensuring STCoscientist generates the RIGHT chains in the first place.

### CRITICAL: Code-generation pitfall — non-ASCII characters leak from know-how

Generated Python / R / shell code MUST be pure ASCII unless you explicitly write `# -*- coding: utf-8 -*-` and are confident the runtime accepts it. Past run (BANKSY) failed with
`Error: invalid character '—' (U+2014)` because STCoscientist copied an em-dash from the know-how's
prose into a generated code string. The Python parser in the STCoscientist execution runner rejects
non-ASCII characters in source code even when they appear only in comments.

Before writing any generated file, sanitise:

```python
import unicodedata as _ud

def _ascii_safe(src: str) -> str:
    # Common offenders in prose copied from docs: em-dash, en-dash, smart quotes,
    # ellipsis, non-breaking space, bullet. Map to ASCII equivalents.
    repl = {
        "\u2014": "-",   # em dash →  -
        "\u2013": "-",   # en dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00a0": " ",   # non-breaking space
        "\u2022": "*",   # bullet
        "\u2192": "->",  # right arrow
    }
    for k, v in repl.items():
        src = src.replace(k, v)
    # Final guard: blank anything still non-ASCII. Never touch '?' -- regexes and strings use it.
    return "".join(c if ord(c) < 128 else " " for c in src)

worker_code = _ascii_safe(worker_code)
server_code = _ascii_safe(server_code)
# Round-trip check before saving:
import ast
ast.parse(worker_code)   # raises SyntaxError early if anything is still wrong
ast.parse(server_code)
```

Apply this sanitiser to EVERY string you're about to write as `.py` / `.R` / `.sh` code,
no matter where it came from.

### CRITICAL: Code-generation pitfall — curly-brace collision in templates

When you write the worker / server file via an f-string or `.format()`, any `{` / `}`
inside the *generated* code (e.g., dict literals, `f"..."` strings) MUST be escaped as
`{{` / `}}`, OR you must build the template with `string.Template` / `str.replace` which
does NOT parse braces. A common concrete failure: you write

```python
worker_code = f"""
TOOL_ID = "{tool_id}"
result = {{"status": "ok"}}          # ← this { will be swallowed by f-string parsing
"""
```

and get `NameError: name 'TOOL_ID_worker' is not defined` or a silent half-written file.
This burns minutes per tool on cryptic heredoc-substitution bugs. Prefer ONE of:

```python
# Option A — string.Template (safe; uses $-substitution, never touches {})
from string import Template
worker_code = Template("""
TOOL_ID = "$tool_id"
MODULE = "$module"
result = {"status": "ok"}   # braces untouched
""").substitute(tool_id=tool_id, module=module)

# Option B — sentinel + .replace (equivalent, no f-string)
worker_code = (TEMPLATE
               .replace("__TOOL_ID__", tool_id)
               .replace("__MODULE__", module))
```

After writing, `ast.parse(Path(...).read_text())` the file to fail fast on any leftover
substitution placeholder. Never ship a worker file that contains `{tool_id}` / `$tool_id`
literal tokens — those are template bugs, not runtime features.

### CRITICAL: Investigate the actual API BEFORE writing code

**You MUST inspect the installed package to find the correct function signatures.
Do NOT guess function names like "run()", "fit()", "main()". Actually discover them.**

Execute this to discover the API:
```python
import subprocess
env_name = f"user_{tool_id}"

# 1. List package contents
r = subprocess.run(["conda", "run", "-n", env_name, "python", "-c",
    f"import {module}; print(dir({module}))"],
    capture_output=True, text=True, timeout=30)
print(f"Module contents: {r.stdout}")

# 2. Try to find main functions
r = subprocess.run(["conda", "run", "-n", env_name, "python", "-c",
    f"import {module}; import inspect; "
    f"funcs = [n for n,v in inspect.getmembers({module}) if callable(v) and not n.startswith('_')]; "
    f"print('Functions:', funcs)"],
    capture_output=True, text=True, timeout=30)
print(r.stdout)

# 3. Get signature of main function
r = subprocess.run(["conda", "run", "-n", env_name, "python", "-c",
    f"import {module}; import inspect; "
    f"print(inspect.signature({module}.{main_function}))"],
    capture_output=True, text=True, timeout=30)
print(f"Signature: {r.stdout}")
```

Use the discovered function signatures to write the worker's core logic.
If the package has submodules (e.g., `banksy.initialize_banksy`), inspect those too.

**If `dir(module)` returns empty or few items, the package uses submodules. Discover them:**
```python
# Find ALL submodules
r = subprocess.run(["conda", "run", "-n", env_name, "python", "-c",
    f"import pkgutil, {module}; "
    f"subs = [m.name for m in pkgutil.walk_packages({module}.__path__, prefix='{module}.')]; "
    f"print('Submodules:', subs)"],
    capture_output=True, text=True, timeout=30)
print(r.stdout)

# For EACH submodule, list functions
for sub in discovered_submodules:
    r = subprocess.run(["conda", "run", "-n", env_name, "python", "-c",
        f"import {sub}; import inspect; "
        f"funcs = [(n, str(inspect.signature(v))) for n,v in inspect.getmembers({sub}) if callable(v) and not n.startswith('_')]; "
        f"print(funcs[:10])"],
        capture_output=True, text=True, timeout=30)
    print(f"{sub}: {r.stdout}")
```

### KNOWLEDGE CAPTURE: Phase 1

**After discovering the API, save the knowledge for future modifications.**
Save these as named variables during discovery (you already computed them):
- `readme_text` — the fetched README content
- `discovered_submodules` — list of package submodules found
- `discovered_functions` — dict of function_path → signature_str
- `main_function` — the chosen function name
- `selection_reason` — why this function was chosen over alternatives
- `function_signature_str` — the inspect.signature() output
- `rejected_alternatives` — list of {module_path, reason} dicts for functions NOT chosen

```python
# === KNOWLEDGE CAPTURE: Phase 1 — Save README + API Discovery ===
try:
    from tools_user.knowledge_manager import save_knowledge
    from datetime import datetime
    _kc_api = {
        "schema_version": 1,
        "tool_id": tool_id,
        "discovered_at": datetime.now().isoformat(),
        "source_url": github_url,
        "package": package_name,
        "module": module,
        "language": language,
        "all_submodules": discovered_submodules if 'discovered_submodules' in dir() else [],
        "all_functions": discovered_functions if 'discovered_functions' in dir() else {},
        "selected_function": {
            "module_path": f"{module}.{main_function}" if 'main_function' in dir() else "",
            "name": main_function if 'main_function' in dir() else "",
            "reason": selection_reason if 'selection_reason' in dir() else "",
            "signature": function_signature_str if 'function_signature_str' in dir() else "",
        },
        "rejected_alternatives": rejected_alternatives if 'rejected_alternatives' in dir() else [],
        "install_method": install_command if 'install_command' in dir() else "",
    }
    save_knowledge(tool_id,
                   readme=readme_text if 'readme_text' in dir() else "",
                   api_discovery=_kc_api)
    print("Knowledge Phase 1 saved (README + API discovery)")
except Exception as e:
    print(f"Knowledge Phase 1 save failed (non-blocking): {e}")
# === END KNOWLEDGE CAPTURE ===
```

### MANDATORY: Micro-test before writing the worker

**After discovering the API, you MUST test the core logic with a tiny synthetic dataset
BEFORE writing the final worker file. This catches wrong function signatures early.**

```python
# Create a micro-test script and run it in the tool's env
micro_test = f'''
import anndata as ad
import numpy as np
import scipy.sparse as sp

# Create tiny test AnnData (10 spots, 50 genes)
np.random.seed(42)
X = sp.random(10, 50, density=0.3, format='csr')
adata = ad.AnnData(X=X)
adata.obsm["spatial"] = np.random.rand(10, 2) * 100
adata.var_names = [f"Gene_{{i}}" for i in range(50)]
adata.obs_names = [f"Spot_{{i}}" for i in range(10)]

# === TRY THE ACTUAL API CALL HERE ===
# import {module}
# result = {module}.{function}(adata, ...)
# Verify: check that adata.obs has a cluster column or result has labels
# print("Cluster labels:", adata.obs.columns.tolist())
# === END ===

print("MICRO_TEST_OK")
'''

# Write and run
with open(f"/tmp/micro_test_{tool_id}.py", "w") as f:
    f.write(micro_test)

r = subprocess.run(["conda", "run", "-n", env_name, "python", f"/tmp/micro_test_{tool_id}.py"],
                   capture_output=True, text=True, timeout=120)

if "MICRO_TEST_OK" in r.stdout:
    print("Micro-test PASSED — API call works correctly")
else:
    print(f"Micro-test FAILED: {r.stderr[:500]}")
    print("FIX the API call before writing the worker!")
    # Read the error, check GitHub source/examples, fix the call, retry
```

**If the micro-test fails, you MUST do ALL of these steps (do NOT ask the user for help):**

1. Read the error traceback carefully
2. **Fetch the README directly and read the usage examples:**
```python
import urllib.request
# Try README
for branch in ["main", "master"]:
    try:
        url = f"https://raw.githubusercontent.com/{org}/{repo}/{branch}/README.md"
        readme = urllib.request.urlopen(url, timeout=10).read().decode()
        # Find code blocks in README
        import re
        code_blocks = re.findall(r'```python\n(.*?)```', readme, re.DOTALL)
        for i, block in enumerate(code_blocks):
            print(f"=== README code block {i} ===")
            print(block[:500])
        break
    except: continue
```

3. **Fetch example scripts from the repo:**
```python
# List files in examples/ or tests/
import json
for dir_name in ["examples", "tests", "tutorials", "notebooks"]:
    try:
        url = f"https://api.github.com/repos/{org}/{repo}/contents/{dir_name}"
        resp = urllib.request.urlopen(url, timeout=10).read()
        files = json.loads(resp)
        for f in files:
            if f["name"].endswith((".py", ".ipynb")):
                print(f"Found: {dir_name}/{f['name']}")
                # Fetch and read the first .py example
                if f["name"].endswith(".py"):
                    content = urllib.request.urlopen(f["download_url"], timeout=10).read().decode()
                    print(content[:2000])
    except: continue
```

4. **If no examples found, read the package source code directly:**
```python
# Read the main module source in the installed package
r = subprocess.run(["conda", "run", "-n", env_name, "python", "-c",
    f"import {module}; import inspect; "
    f"src = inspect.getsource({module}); print(src[:3000])"],
    capture_output=True, text=True, timeout=30)
print(r.stdout)

# Read submodule sources to find the actual workflow
for submod in discovered_submodules:
    r = subprocess.run(["conda", "run", "-n", env_name, "python", "-c",
        f"import {submod}; import inspect; "
        f"src = inspect.getsource({submod}); print(src[:2000])"],
        capture_output=True, text=True, timeout=30)
    print(f"=== {submod} source ===")
    print(r.stdout[:1000])
```

5. Based on what you learn from README examples + source code, fix the function call
6. Re-run the micro-test until it passes
7. ONLY THEN write the final worker

**You MUST NOT ask the user for the API. Discover it yourself from the source code and examples.
This is the most critical part of tool creation — get the API right.**

### Micro-test for hybrid tools (if R/rpy2 detected):
Also verify R functionality works:
```python
micro_test_r = f'''
import rpy2.robjects as ro
# Test R is accessible
ro.r('cat("R bridge OK\\n")')
# Test mclust if installed
try:
    ro.r('library(mclust); cat("mclust OK\\n")')
except:
    print("mclust not available — tool will use Python clustering only")
print("MICRO_TEST_R_OK")
'''
r = subprocess.run(["conda", "run", "-n", env_name, "python", "-c", micro_test_r],
                   capture_output=True, text=True, timeout=30)
if "MICRO_TEST_R_OK" in r.stdout:
    print("R micro-test PASSED")
else:
    print(f"R micro-test issue (non-blocking): {r.stderr[:200]}")
```

### Real-data robustness patterns (READ before writing the worker)

A common failure pattern: tools register cleanly in Phase 4 + 5 but then fail the first
time a real user runs them on a raw Visium h5ad. The pattern of failure is almost always
one of the six cases below — bake the mitigation into the generated worker upfront, not
as an afterthought.

1. **Missing obs labels (LIANA+, STAMP, Giotto-niche, etc.)** — many downstream tools
   require `adata.obs[{groupby}]` to be populated with cluster labels. Raw Visium from
   the registry has no clusters. Worker MUST auto-fill a default before calling the
   tool:
   ```python
   # If the tool needs `adata.obs[groupby]` and it's missing, compute Leiden first
   if groupby and groupby not in adata.obs.columns:
       import scanpy as sc
       if "X_pca" not in adata.obsm:
           sc.pp.normalize_total(adata, target_sum=1e4)
           sc.pp.log1p(adata)
           sc.pp.pca(adata, n_comps=min(30, adata.shape[1] - 1))
       sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
       sc.tl.leiden(adata, resolution=0.5, key_added=groupby)
   ```

2. **Dir passed where file expected (BANKSY run_spatial_pipeline "Is a directory" HDF5
   error)** — the MCP tool's `output_dir` is a directory but many library functions
   want `output_h5ad_path = output_dir + "/result.h5ad"` explicitly. Worker MUST
   construct a concrete file path and pass that, not the raw `args.output_dir`.

3. **Multi-input tools (SpaGE needs sc_h5ad + spatial h5ad)** — declare ALL required
   inputs as CLI parameters (never assume a second dataset can be inferred). If the
   second input isn't provided, emit `WorkerOutput.error(...)` with a clear message —
   don't crash mid-run.

4. **Pydantic over-validation in MCP server** — if the `@mcp.tool()` decorator declares
   required parameters but the tool has sensible defaults, the server rejects calls
   with missing args *before they reach your worker*. Always default every MCP
   parameter with `Optional[...]` and a concrete default value; let the *worker*
   decide what's required, not the schema.

5. **`diagnose_spatial_data` returns JSON string in some code paths** — STCoscientist's worker
   code often calls this then does `.get("mcp_ready")`, which crashes on strings.
   Always `if isinstance(x, str): x = json.loads(x)` before treating it as a dict.

6. **Missing QC metrics / image scalefactors** — raw h5ads often lack QC metrics but
   are still valid for most tools. Don't reject them — worker should compute what it
   needs from what's present and emit a `warnings: [...]` list in the output meta.

### Python Worker Template

The worker MUST use `worker_utils.WorkerOutput` for standardized output.

**HARD RULE — runtime env setup at worker startup (tool-agnostic).**

Every worker, regardless of tool, must configure its runtime env *before*
importing the wrapped package. The most common silent failures here are not
the tool's fault but the worker's setup:

- **Matplotlib / graphics** — any tool that imports `matplotlib`, `seaborn`,
  `scanpy.plotting`, `ggplot2`-via-rpy2 will crash with
  `$DISPLAY` / `_tkinter.TclError` on a headless node. Fix once, for all
  tools: set `MPLBACKEND="Agg"` in the env before any import.
- **R_HOME / R_LIBS** — any worker that touches R (rpy2, anndata2ri, or
  `subprocess.run(["Rscript", ...])`) will silently use the wrong R install
  (often the system R, not the conda env's R) if `R_HOME` isn't set. The
  canonical recipe: `R_HOME = subprocess.check_output(["R", "RHOME"]).strip()`
  then `os.environ["R_HOME"] = R_HOME` and prepend the env's R library path.
  Missing `R_HOME` was the root cause of the last `semla`-class worker bug.
- **CUDA visibility** — any GPU tool crashes on CPU-only boxes unless the
  worker falls back via `CUDA_VISIBLE_DEVICES=""` or an explicit device check.
- **Thread/core limits** — numerics packages (numpy, torch, sklearn,
  lightgbm) can oversubscribe CPUs on shared nodes. Cap via
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` from the
  worker's `--threads` arg (default 4).
- **Output buffering** — set `PYTHONUNBUFFERED=1` so log lines arrive in
  real time when the MCP server is tailing stderr.

Emit one helper near the top of every worker (put it in `worker_utils` if you
prefer, but embedding it makes the worker self-contained):

```python
def _configure_runtime_env(*, needs_r: bool = False, gpu_optional: bool = True,
                          threads: int = 4) -> None:
    import os, sys, subprocess, shutil
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, str(threads))
    if gpu_optional and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        # Don't force CPU if the user set it; default to GPU-if-present.
        pass
    # P14 — prepend the running interpreter's env bin to PATH so subprocess
    # calls to Rscript, samtools, bedtools, etc. resolve to env-bundled
    # binaries rather than whatever the MCP-server parent's PATH happened
    # to be. Without this, workers fail with "Rscript: No such file or
    # directory" even though R is installed in the env.
    env_bin = os.path.join(sys.prefix, "bin")
    cur_path = os.environ.get("PATH", "")
    if env_bin not in cur_path.split(os.pathsep):
        os.environ["PATH"] = env_bin + (os.pathsep + cur_path if cur_path else "")
    if needs_r and shutil.which("R"):
        try:
            rhome = subprocess.check_output(["R", "RHOME"], text=True).strip()
            os.environ.setdefault("R_HOME", rhome)
            # Prepend the env's R library path so rpy2/Rscript find it first.
            rlibs = os.path.join(rhome, "library")
            existing = os.environ.get("R_LIBS", "")
            os.environ["R_LIBS"] = rlibs + (":" + existing if existing else "")
        except Exception:
            pass  # let the import fail loudly rather than mask it
```

Call `_configure_runtime_env(needs_r=<True if the wrapped tool touches R>)`
as the very first line of `main()`. The `needs_r` flag is the ONLY tool-
specific bit; everything else is hard-coded generic defaults.

```python
#!/usr/bin/env python
import argparse, json, os, sys, traceback

# Add tools_user to path so worker_utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import WorkerOutput

def log(msg):
    sys.stderr.write(f"[{TOOL_ID}-worker] {msg}\n")
    sys.stderr.flush()

TOOL_ID = "{tool_id}"

def main():
    _configure_runtime_env(needs_r={NEEDS_R})  # <-- first line, always
    parser = argparse.ArgumentParser(description="{tool_name} worker")
    parser.add_argument("--input", required=True, help="Input h5ad path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    # Add tool-specific parameters here
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        import {module}
        import anndata as ad
        import scanpy as sc

        # Validate input
        if not os.path.isfile(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")

        log(f"Loading {args.input}")
        adata = ad.read_h5ad(args.input)
        log(f"Loaded: {adata.n_obs} spots x {adata.n_vars} genes")

        # === CORE LOGIC (fill from README) ===
        # result = {module}.{function}(adata, ...)
        # === END CORE LOGIC ===

        # Save outputs
        out_h5ad = os.path.join(args.output_dir, f"{TOOL_ID}_result.h5ad")
        out_csv = os.path.join(args.output_dir, f"{TOOL_ID}_clusters.csv")
        adata.write_h5ad(out_h5ad)
        # adata.obs[[cluster_key]].to_csv(out_csv)

        # Emit standardized output
        out = WorkerOutput(TOOL_ID, task="{task_type}")
        out.set_data(n_spots=adata.n_obs, n_genes=adata.n_vars)
        out.add_output_files({{"result_h5ad": out_h5ad, "clusters_csv": out_csv}})
        out.emit()

    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc())
        WorkerOutput.emit_error(TOOL_ID, str(e), task="{task_type}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Naming the MCP function

Before writing the server, define `function_name` — the name for the `@mcp.tool()` decorated function.
This MUST be used consistently in both the server code AND the YAML `spatialomicsgym_name` field.

It is ALWAYS `{tool_id}_run` (HARD RULE 9) -- never a task verb: `sopa_run`, `gaston_run`.

```python
# Define the MCP function name (used in server code AND YAML spatialomicsgym_name)
function_name = f"{tool_id}_run"   # HARD RULE 9; a second mode goes in a `mode` argument
```

**CRITICAL**: The `spatialomicsgym_name` in mcp_config_user.yaml MUST exactly match the `@mcp.tool()` function name.
If they differ, `session.call_tool()` will fail with "Unknown tool".

### MCP Server Template

**TOOL-AGNOSTIC INPUT RULE:** this pipeline is NOT specific to spatial transcriptomics /
h5ad. STCoscientist must choose the correct input parameter name and reader based on what the
**wrapped tool** expects. The table below is the decision matrix:

| Wrapped tool consumes… | MCP parameter name | Worker reads via |
|------------------------|--------------------|------------------|
| AnnData `.h5ad` | `input_h5ad: str` | `anndata.read_h5ad` |
| SingleCellExperiment `.rds` / Seurat | `input_rds: str` | R `readRDS` via rpy2 or R worker |
| Raw sequencing `.bam` / `.sam` / `.cram` | `input_bam: str` | `pysam.AlignmentFile` |
| Variant `.vcf` / `.vcf.gz` / `.bcf` | `input_vcf: str` | `pysam.VariantFile` or cyvcf2 |
| FASTQ (`.fastq` / `.fastq.gz`) | `input_fastq: str` | pyfastx / Bio.SeqIO |
| Image (`.tif`, `.png`, `.zarr`) | `input_image: str` | `tifffile` / `PIL` / `zarr` |
| Tabular (`.csv` / `.tsv` / `.parquet`) | `input_csv: str` | `pandas.read_csv` / `read_parquet` |
| Protein structure (`.pdb` / `.cif`) | `input_pdb: str` | `biotite.structure.io` / `prody` |
| Generic directory of files | `input_dir: str` | `pathlib.Path(input_dir).iterdir()` |
| CLI-only — tool reads any path | `input_path: str` | pass-through to CLI binary |
| No structured input (query/config) | `query: str` + `config: dict` | parse directly |

For tools that accept MULTIPLE inputs (SpaGE needs spatial + single-cell, STAR needs
fastq + genome), declare each as a separate `input_*` parameter:

```python
def spage_run(input_h5ad: str, input_sc_h5ad: str, output_dir: str,
                 n_pv: Optional[int] = 50, ...) -> Dict[str, Any]:
```

**HARD RULE for parameter schema:** Every `@mcp.tool()` parameter EXCEPT the input(s)
and `output_dir` MUST have `Optional[...]` typing and a concrete default value. Pydantic
(used internally by FastMCP) raises "N validation errors" if the user calls the tool
without providing some required argument — this silently turns registered tools into
uncallable dead weight. Let the **worker** decide what's required (error out cleanly
via `WorkerOutput.error`), never the MCP schema.

**HARD RULE input-counting (P20 strengthened):** "the input(s)" in the rule above
means EXACTLY ONE primary input parameter. Tools that sometimes need multiple
inputs (e.g., SpaDecon needs spatial h5ad + single-cell h5ad) MUST still mark
every input beyond the first as `Optional[str] = None` at the MCP schema level.
The worker then checks `if sc_h5ad is None: return WorkerOutput.error(...)`.
The rule is about schema, not semantics: a caller invoking the tool with only
the primary input MUST get a clean worker-level error message, never a
pydantic "1 validation error" from the MCP framework. Required at the framework
level = DOA if the caller doesn't know the secondary input exists.

After writing the MCP server, STCoscientist MUST verify programmatically:
```python
# Schema sanity check — every @mcp.tool parameter except the FIRST
# positional input and output_dir must have a default value.
import ast
src = (TOOLS_USER_DIR / f"{tool_id}_mcp_server.py").read_text()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        has_mcp_tool = any(
            isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "tool"
            for d in node.decorator_list
        )
        if not has_mcp_tool:
            continue
        args = node.args.args
        defaults = node.args.defaults
        # first N-len(defaults) args are required; must be <= 2 (primary input + output_dir)
        n_required = len(args) - len(defaults)
        assert n_required <= 2, (
            f"MCP schema violation: {node.name} has {n_required} required "
            f"args. HARD RULE allows at most 2 (primary input + output_dir). "
            f"Rewrite every extra param as Optional[...] with a default."
        )
```

```python
#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Any, Dict, Optional
from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "{tool_id}"
# The worker always sits beside this server file, so derive its path -- never hardcode one.
# An absolute path typed here is the path of THIS box: on any other checkout the server still
# starts, then every call dies with "can't open file". `{TOOL_ID_UPPER}_WORKER` still overrides.
_HERE = Path(__file__).resolve().parent
PYTHON_ENV, WORKER_PY = get_worker_paths(
    "USER_{TOOL_ID_UPPER}",
    "/opt/conda/envs/user_{tool_id}/bin/python",
    str(_HERE / "{tool_id}_worker.py"),
)
mcp = create_mcp(TOOL_NAME)

@mcp.tool()
def {function_name}(
    input_h5ad: str,
    # Relative, so it lands where the caller ran. `os.makedirs` below has exist_ok=True, which makes
    # an absolute default quiet rather than safe: as root it creates the directory at the filesystem
    # root and the user's results end up outside the data directory they passed, while the call still
    # reports success; as an ordinary user it raises PermissionError on a path they never named.
    output_dir: str = "./{tool_id}_output",
    # EVERY extra parameter must be Optional[...] with a concrete default.
    # Example of correct shape — Pydantic will accept ANY subset of the
    # non-required params without rejecting the call:
    n_neighbors: Optional[int] = 15,
    n_pcs: Optional[int] = 30,
    resolution: Optional[float] = 0.5,
    groupby: Optional[str] = None,     # None → worker will compute Leiden
    random_state: Optional[int] = 0,
) -> Dict[str, Any]:
    \"""{tool_name} — {description}.\"""
    os.makedirs(output_dir, exist_ok=True)
    args = ["--input", input_h5ad, "--output-dir", output_dir]
    if n_neighbors is not None: args += ["--n-neighbors", str(n_neighbors)]
    if n_pcs       is not None: args += ["--n-pcs",       str(n_pcs)]
    if resolution  is not None: args += ["--resolution",  str(resolution)]
    if groupby     is not None: args += ["--groupby",     groupby]
    if random_state is not None: args += ["--random-state", str(random_state)]
    return run_worker_cli(TOOL_NAME, PYTHON_ENV, WORKER_PY, args)

if __name__ == "__main__":
    mcp.run()
```

### CLI-wrapper Worker Template (for tools like FICTURE / STAR / samtools)

Some tools expose no Python API — only a command-line binary. Wrap them like this so
they appear to the rest of the system just like any other MCP tool:

```python
#!/usr/bin/env python
import argparse, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import WorkerOutput, env_bin

TOOL_ID = "{tool_id}"
CLI_NAME = "{cli_binary_name}"   # e.g. "ficture", "STAR", "samtools"
CLI_TIMEOUT_S = 1800              # one CLI run; raise it here for a long job

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    # plus any tool-specific flags the CLI accepts
    p.add_argument("--threads", default="4")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out = WorkerOutput(tool=TOOL_ID)

    # Locate the binary inside the env this worker is running under. env_bin() derives that from
    # sys.executable, so it still finds the CLI on a host whose conda lives somewhere else.
    bin_path = env_bin(CLI_NAME)
    if not os.path.exists(bin_path):
        # WorkerOutput.error() returns a dict; emit_error() prints it as the one JSON line
        WorkerOutput.emit_error(TOOL_ID, f"CLI binary {CLI_NAME} not found at {bin_path}")
        return

    cmd = [bin_path,
           "--input", args.input,
           "--output-dir", args.output_dir,
           "--threads", args.threads]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=CLI_TIMEOUT_S)
        if r.returncode != 0:
            WorkerOutput.emit_error(TOOL_ID, f"{CLI_NAME} exited {r.returncode}. stderr tail: {r.stderr[-500:]}")
            return
        # Collect every non-empty file under output_dir as an output artefact
        outs = {p.name: str(p) for p in Path(args.output_dir).rglob("*")
                if p.is_file() and p.stat().st_size > 0}
        (out.add_output_files(outs)
            .set_summary(stdout_bytes=len(r.stdout), n_outputs=len(outs))
            .set_meta(cli_name=CLI_NAME, cli_args=cmd[1:])
            .emit())
    except subprocess.TimeoutExpired:
        WorkerOutput.emit_error(TOOL_ID, f"{CLI_NAME} still running after CLI_TIMEOUT_S={CLI_TIMEOUT_S}s; "
                                         f"raise CLI_TIMEOUT_S in {TOOL_ID}_worker.py")

if __name__ == "__main__":
    main()
```

### R Worker Template

```r
#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library({main_r_package}))

log_msg <- function(...) message(sprintf("[{tool_id}-worker] %s", paste0(...)))

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  # Parse --key value pairs
  opts <- list()
  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    key <- gsub("-", "_", key)
    if (i < length(args) && !startsWith(args[i+1], "--")) {
      opts[[key]] <- args[i+1]
      i <- i + 2
    } else {
      opts[[key]] <- TRUE
      i <- i + 1
    }
  }

  result <- tryCatch({
    # === CORE R LOGIC ===
    # ...
    # === END ===

    list(status = "ok", tool = "{tool_id}",
         output_files = list(result = "path"),
         data = list(n_spots = 0, n_genes = 0))
  }, error = function(e) {
    log_msg("ERROR: ", conditionMessage(e))
    list(status = "error", tool = "{tool_id}",
         error = conditionMessage(e))
  })

  cat(toJSON(result, auto_unbox = TRUE), "\n")
}

main()
```

### KNOWLEDGE CAPTURE: Phase 3

**After writing the worker and MCP server, save function signatures and micro-test code.**
Save these as named variables during code generation:
- `discovered_parameters` — from inspect.signature() of the core function
- `worker_cli_params` — dict of argparse parameters (name → {type, default, help})
- `mcp_params` — dict of MCP function parameters (name → {type, default, required})
- `pipeline_steps` — list of pipeline step dicts [{step, call, purpose}]
- `micro_test` — the micro-test source code string

```python
# === KNOWLEDGE CAPTURE: Phase 3 — Save Function Signatures + Micro-Test ===
try:
    from tools_user.knowledge_manager import update_knowledge, save_knowledge
    from datetime import datetime
    _kc_sigs = {
        "schema_version": 1,
        "tool_id": tool_id,
        "core_function": {
            "import_path": f"{module}.{main_function}" if 'main_function' in dir() else "",
            "parameters": discovered_parameters if 'discovered_parameters' in dir() else {},
        },
        "worker_cli": {
            "parameters": worker_cli_params if 'worker_cli_params' in dir() else {},
        },
        "mcp_function": {
            "name": function_name if 'function_name' in dir() else "",
            "parameters": mcp_params if 'mcp_params' in dir() else {},
        },
        "pipeline_steps": pipeline_steps if 'pipeline_steps' in dir() else [],
    }
    update_knowledge(tool_id, function_signatures=_kc_sigs)
    if 'micro_test' in dir() and micro_test:
        save_knowledge(tool_id, micro_test_code=micro_test)
    print("Knowledge Phase 3 saved (signatures + micro-test)")
except Exception as e:
    print(f"Knowledge Phase 3 save failed (non-blocking): {e}")
# === END KNOWLEDGE CAPTURE ===
```
