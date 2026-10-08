# Spatial Tool Schema Pre-check

## Step 0 (before any of this): pick the right tool with `recommend_analysis_tools()`

Before deciding which spatial MCP tool to call, **always call** `recommend_analysis_tools(h5ad_path="<file>.h5ad", analysis_goals="<your goal>")` first. It returns a JSON **string** — `json.loads()` it before reading; slicing it gives you characters, not tools. Inside is a ranked list of tools with `priority=1` (P1) being the empirical multi-LLM benchmark winner for that task type on that platform. Use the P1 unless you have a documented, data-driven reason to deviate — your prior knowledge about "popular" tools is not such a reason (verified 2026-05-11: gpt-5.4-mini ignored P1s and picked weaker alternatives in 3/3 task types when this step was skipped).

If you only have a fuzzy name or alias (e.g. user typed "rctd" or "deepST"), call `resolve_tool_name(query)` to map it to the canonical registered MCP `spatialomicsgym_name`.

## How to invoke an MCP tool (the only correct pattern)

After `add_mcp()` registers a server, every tool's `spatialomicsgym_name` is **already in scope** inside `<execute>` Python blocks. Call it directly:

```python
# CORRECT (note that the input-path kwarg is named differently by each tool -- read its schema,
# do not assume a shared name, and do not assume `n_clusters` is how a tool spells its K):
result = somde_run(h5ad_path="/path/to/in.h5ad", output_dir="/path/to/out")
result = deepst_identify_domains(st_h5ad="/path/to/in.h5ad", n_domains=5, output_dir="/path/to/out")
```

The following patterns are **forbidden** (a P13 violation per `add_new_mcp_tool.md`):

```python
# WRONG:
from mcp_servers.somde import somde_run
import mcp_servers.cell2location
importlib.import_module("mcp_servers.prost")
```

`mcp_servers.<name>` is a *virtual* module that exists only in the agent's own Python process. In a python `<execute>` cell the `from mcp_servers...` and `importlib.import_module` forms therefore hand back, at best, the function already in scope, and `import mcp_servers.<name>` fails with `ModuleNotFoundError: No module named 'mcp_servers'`. From a `#!BASH` or `#!CLI` cell (`python -c`, a heredoc, a script) all three fail that way. The invocation contract is "call by spatialomicsgym_name directly".

## When to use this capability (the rest of this doc)

Whenever you are about to invoke a spatial transcriptomics MCP tool that you have NOT used before in the current conversation, OR whose docstring mentions specific keys in `adata.obs`, `adata.obsm`, `adata.uns`, or `adata.layers` as inputs.

This check takes ~10 seconds and prevents long failure loops where the tool runtime errors out because the input h5ad doesn't have the expected schema. **An ACTION-FIRST prompt does NOT override this check.** ACTION-FIRST means "skip *unnecessary* preprocessing"; it does not mean "skip schema validation of structured inputs the tool depends on".

## The failure mode this prevents

Real example (2026-05-06): STCoscientist was prompted to call `run_spacel_splane` ACTION-FIRST on `visium_dlpfc_domain`. SPACEL Splane requires cell-type proportions in `adata.obs[<celltype_col>]` AND a list in `adata.uns['celltypes']` — it is a *post-deconvolution* spatial domain method, not a from-scratch clustering tool. The Visium DLPFC h5ad only has raw expression + layer labels. The worker failed at `_prepare_celltypes_for_splane`. All 4 LLM cells burned ~6 minutes each (~24 min wasted) producing identical schema-error stack traces.

A 30-second `adata.obs.columns` print before the tool call would have surfaced "no celltype column" and let STCoscientist either skip the tool or report incompatibility.

## How to use

Right BEFORE the first MCP tool call for an unfamiliar tool, do ONE structured peek:

```python
import anndata as ad
a = ad.read_h5ad("<input_path>")
print("n_spots, n_genes:", a.shape)
print("obs columns:", list(a.obs.columns))
print("obsm keys:", list(a.obsm.keys()))
print("uns keys:", list(a.uns.keys()))
print("layers:", list(a.layers.keys()))
```

Then check the tool's required-key list (from its docstring or wrapper signature) against what's actually present:

- If a required key (e.g. cell-type column, `obsm['spatial']`, `uns['celltypes']`) is **missing**, do NOT invoke the tool. Instead emit a single-line OUTPUT_INVALID note: `tool=<name> requires <key> in adata.<location>; not present in this dataset` and stop.
- If keys are present but named differently (e.g. `cell_type` vs `celltype`), pass the actual column name as a tool kwarg if the wrapper supports it; otherwise rename in a brief preprocessing step BEFORE the tool call.
- If everything is present, proceed with the tool call exactly as the ACTION-FIRST prompt instructs.

## What this is NOT

This is not a license to:
- Run full diagnostic helpers like `repair_spatial_h5ad` / `validate_spatial_h5ad` — those do far more than the peek and are still blocked by ACTION-FIRST prompts.
- Add preprocessing steps (HVG, normalize, log1p) when the tool wrapper handles them internally.
- Print the entire `adata` repr or all `obs.head()` rows — keep the peek to ~5 lines of structured output.

The peek is a metadata sniff, not a transformation. Its only purpose is to confirm the structured inputs a tool requires actually exist.

## Tools where this peek is mandatory (non-exhaustive)

These tools have non-trivial schema requirements beyond `X` + `obsm['spatial']`:

- **`run_spacel_splane`** — needs `obs[<celltype_col>]` + `uns['celltypes']` (post-deconvolution method).
- **`run_stdgcn`** — actually deconvolution, not clustering; needs an SC reference h5ad with `obs[cell_type_key]`.
- **`run_iris`** — via MCP it takes two SPATIAL CSVs (`spatial_counts_csv` + `spatial_coords_csv`), not an h5ad, and exposes no SC-reference params (runs IRISfree / spatial-only). No h5ad `obs`/`obsm` peek applies — just supply the two spatial CSVs.
- **`run_bass`** and **`precast_spatial_clustering`** — take CSVs and refuse an h5ad. BASS wants `spatial_counts_csv` (raw counts, genes x spots) + `spatial_coords_csv`; PRECAST wants the lists `counts_csvs` + `coords_csvs`, one entry per sample. Make them with `convert_h5ad_to_csv` first (its `counts.csv` is genes x spots by default, beside `coordinates.csv`).
- **`spatialprompt_cluster`** — `input_mode` is required: `"h5ad"` with `st_h5ad`, or `"visium_10x"` with `counts_h5` + `spatial_dir`.
- **Any tool whose docstring mentions `cell_type_key`, `batch_key`, `celltypes`, `proportions`, `signature_csv`, or `obsm['<custom_key>']`** — peek first, confirm the named key/column exists.

## Tools where this peek is NOT needed

Tools whose only inputs are raw expression + spatial coordinates can skip the peek; the wrapper handles its own validation:

- `prost_pnn_domains`, `run_scanpy_spatial_domain`, `cellcharter_cluster_spatial_domains`, `stlearn_spatial_clustering`, `seurat_qc_cluster`, `run_miso`, `run_sedr`, `spaceflow_spatial_domains`, `graphst_spatial_clustering`, `deepst_identify_domains`, `stagate_spatial_domains`.

These have been used hundreds of times and their schema requirements are met by any standardized spatial h5ad in this benchmark.
