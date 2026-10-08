# Spatial Dataset Library

You have access to a built-in collection of standardized spatial transcriptomics datasets that can be used for analysis when the user does not provide their own data.

## When to use this capability

**Trigger criteria (ALL must be true):**
1. User's question involves spatial transcriptomics analysis (clustering, SVG detection, deconvolution, differential expression, visualization, etc.)
2. No .h5ad, .csv, or any file path is mentioned in the query or conversation history
3. **The user has not referred to data of their own at all** — see the possessive rule below
4. The question is analytical — requesting actual computation or results, not just conceptual explanation

**The possessive rule (criterion 3, stated generally).** Any **possessive reference to the user's own
specimen or data** means they have data and you do not have it yet. Treat it as a request to analyze
*their* material, no matter how the sentence is phrased and even though no path has been given.
This covers, non-exhaustively: **"my tissue", "my sample", "my slide", "my section", "my experiment",
"my dataset", "my data", "my file"**, the same phrases with "our", and equivalents like "the tissue
I sequenced" or "the slide we ran". Do not match against this list literally — it is illustrative.
If a reasonable reading is "the user is talking about something they have", the rule applies.

**When the possessive rule applies, ask for the path. Do not run an analysis on a substitute
dataset without asking first** — not as a demonstration, not "to show the workflow". A single
clarifying question costs one turn. Substituting a built-in dataset costs a full analysis run and
answers a question the user did not ask. (Live failure, 2026-07-28: "Can you check whether **my
tissue** has any interesting spatial structure?" triggered a library search, then a 25-minute
GraphST clustering of an unrelated glioblastoma sample — while the agent's own answer conceded it
still needed the user's file path.)

**Do NOT trigger if:**
- The possessive rule above applies, or a file path was mentioned
- Data was provided or referenced in previous conversation turns
- User says they will provide data later or is preparing data
- Question is purely theoretical (e.g., "what is spatial transcriptomics?", "explain Visium technology")
- User is asking about tool usage without wanting to run it now

**Do trigger** when the request is genuinely about spatial data in the abstract and no ownership is
implied: "show me what a Visium clustering looks like", "analyze a human breast cancer sample",
"what brain datasets do you have", "run a deconvolution demo".

## How to use

**IMPORTANT: You MUST call the `search_spatial_datasets` MCP tool to search the library. Do NOT try to search manually via code — the datasets are indexed in a registry that only the MCP tool can access.**

1. Call the `search_spatial_datasets` MCP tool with filters extracted from the user's query
2. **Use only results with `available: true` and `available_for_spatial: true`;** a hit whose `has_spatial_coordinates` is false is an expression atlas with no positions. The registry is a catalog of dataset metadata; the data files themselves may or may not be on this machine. Each result carries `available` (readable here) and, when it is, an `h5ad_path` that is already resolved and directly usable — no globbing or existence-checking needed on your side.
3. If the search returns `n_available: 0`, the datasets are simply not installed here. Do NOT try to open the listed paths and do NOT report the tool as broken — relay the tool's `message` and ask the user for a data path.
4. Pass the `h5ad_path` of an available dataset to spatial analysis MCP tools (e.g., `run_scanpy_spatial_domain`, `hotspot_spatial_modules`, `tangram_map_sc_to_spatial`)
5. Always mention the dataset source and key metadata (organ, disease status, technology) in your response

**Search examples — call the MCP tool like this:**
- "analyze brain cancer" → call `search_spatial_datasets(organ="Brain", disease="cancer")`
- "find healthy lung tissue" → call `search_spatial_datasets(organ="Lung", is_healthy=True)`
- "compare breast cancer samples" → call `search_spatial_datasets(organ="Breast", disease="cancer", max_results=10)`
- "what datasets are available" → call `search_spatial_datasets(keyword="", max_results=20)`

## Current library contents

The library is auto-indexed from standardized datasets and **grows over time**, so do NOT assume a fixed
count or a fixed organ list — call `search_spatial_datasets(keyword="", max_results=100)` (optionally with
`organ=` / `technology=` / `is_healthy=` filters) to get the live inventory before telling a user whether
something is available.

- The collection spans many human organs and both healthy and diseased tissue (a range of cancers:
  glioblastoma, breast, lung, ovarian, melanoma, and others).
- **Technologies:** Visium v1, Visium CytAssist, Visium HD, and Xenium.
- Datasets provide a standardized `.h5ad` (Scanpy-ready), with spatial coordinates and, for Visium,
  tissue images. A dataset without coordinates (a single-cell atlas) is marked
  `has_spatial_coordinates: false` and is not usable by spatial tools.
