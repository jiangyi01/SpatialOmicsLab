# MCP tools: portals, workers and the config

Every analysis method is an MCP server. This page explains the three files behind one, how the agent finds and starts
them, how a call travels, and how the skill catalogue routes a task to the right one. The methods themselves are
listed in the [tool catalogue](../tools/index.md).

## One method, three files

| File | What it is |
|---|---|
| `agent/tools/<key>_mcp_server.py` | The **portal**: the MCP server the agent starts. Built on `base_mcp.py`, it publishes the method's functions, and for each call runs the worker in a subprocess, parses its JSON reply and returns it (errors as a dict, never an exception). It runs in the agent environment. |
| `agent/tools/<key>_worker.py` or `<key>_worker.R` | The **worker**: the code that runs the method, under the method's own interpreter. Python workers reply through `worker_utils.py` and must stay importable by Python 3.7, the oldest interpreter a method env ships; R workers run under the env's `Rscript`. |
| `install/recipes/tool_specs/<key>.yaml` | The **spec**: the portal and worker file names, the conda env the server runs in (`source_env`) or the interpreter it borrows, the env recipe (`tool_specs/env/<key>.env.yaml`), GPU and size hints, and the analysis category of each function. `sog-setup capture` writes it from a machine that has the env. |

`<key>` is the server key in `agent/MCP_server/mcp_config.yaml`. A few keys differ from their file stems
(`cellpose_seg` → `cellpose_*`, `spark_spatial` → `spark_*`, `spatial_miso` → `miso_*`); the catalogue lists them.

The directory is flat on purpose. Every portal does `from base_mcp import ...` from its own directory, every
path-healer looks for a file by basename directly in `tools/`, the installer rewrites any path containing `/tools/`
to this directory plus the file name, and the wheel payload copies `*.py` and `*.R` from here without recursing. A
file in a subfolder is not found, not shipped, and its recorded path is rewritten into one that does not exist.

Shared helpers: `base_mcp.py` (portal support), `worker_utils.py` (the standard reply and the shared input readers:
coordinates, the counts matrix, the in-tissue filter, gene-ID matching), `eval_metrics.py` (scores for clustering,
SVGs and deconvolution), `result_collector.py`, `ucd_token.py`, the two R converters, and `.ruff.toml`, which lints
the directory for Python 3.7.

## The config

`agent/MCP_server/mcp_config.yaml` lists every shipped server. Each block carries the launch command, the two
environment variables the portal reads to find its worker and interpreter, whether the server is enabled, a
description, and every function with its description and parameters:

```yaml
mcp_servers:
  graphst:
    command: [python, /abs/path/to/agent/tools/graphst_mcp_server.py]
    enabled: true
    description: GraphST-based spatial transcriptomics tools for spatial domain identification and scRNA-ST deconvolution.
    env:
      GRAPHST_PYTHON: /opt/conda/envs/GraphST/bin/python
      GRAPHST_WORKER: /abs/path/to/agent/tools/graphst_worker.py
    tools:
      - spatialomicsgym_name: graphst_spatial_clustering
        description: Run GraphST spatial clustering / domain identification on a single spatial transcriptomics AnnData (.h5ad) file ...
        parameters:
          st_h5ad: {type: str, required: true, description: Path to spatial AnnData (.h5ad) ...}
          output_dir: {type: str, required: true, description: Output directory ...}
          n_clusters: {type: int, required: true, description: Number of spatial domains ...}
          cluster_tool: {type: str, required: false, default: mclust, description: ...}
```

The paths are absolute because each worker runs in its own conda environment. The tracked file is frozen to the
build host's paths; on any other machine `sog-setup` rewrites it (after a timestamped backup under
`.sog_setup/backups/`) with this machine's interpreters and script paths, and sets `enabled` to whether the env
actually exists and passes the import probe. When a recorded path is stale anyway (a clone moved, a renamed checkout),
the portal, the agent and the installer heal it by basename in `agent/tools/`.

The function descriptions and parameter descriptions in this file are what the model reads. They are long and
specific by design: they state what the input must contain, what the tool does to it, what it writes and what it
reports back. The [tool pages](../tools/index.md) render the same text.

### Which config is used

Two different questions, two orders.

**Which config to serve** (the terminal chat, `add_mcp()`, `sog-setup conncheck`): an explicit `--mcp PATH` or
`add_mcp(path)` first; otherwise the `SOG_MCP_CONFIG` pointer that `sog-setup` records in `.env`, then the generated
`install/recipes/mcp_config.setup.yaml`, then the canonical `agent/MCP_server/mcp_config.yaml`.

**Which servers this package ships** (the tool catalogue and parameter contracts): the canonical file found from the
package's own location, then the pointer, then the working directory, then the wheel's read-only copy.

**The user overlay.** `agent/MCP_server/mcp_config_user.yaml` (or `SOG_MCP_USER_CONFIG`) lists the tools the agent
created on this machine. With tool creation enabled, `add_mcp()` merges it into the served config: the merge never
edits `mcp_config.yaml`, a shipped server wins any name conflict, and a corrupt overlay is skipped with a warning.

## A call, end to end

1. The agent starts the portal with the configured command (an MCP server over stdio) and lists its tools. The
   handshake is bounded by `SOG_MCP_HANDSHAKE_TIMEOUT` (30 s); a dead handshake is an environment problem, not a
   slow analysis.
2. The model calls a function with keyword arguments. The portal is a FastMCP server whose `@mcp.tool()` functions
   carry typed signatures, so the arguments are checked before anything runs.
3. The portal resolves its worker (`<PREFIX>_WORKER`) and interpreter (`<PREFIX>_PYTHON` or `_RSCRIPT`), healing a
   stale path by basename, and runs the worker as a subprocess, passing the arguments on the command line
   (`run_worker_cli`) or as a JSON payload (`run_worker_json`). The call is bounded by `SOG_TIMEOUT_SECONDS`. A
   worker whose call omitted `output_dir` writes under `$SOG_WORK_DIR` (else a writable `/workspace/work`, else
   `./work`).
4. The worker reads the inputs with the shared readers, checks them (raw counts where the method needs them, the
   in-tissue filter, coordinate presence), runs the method, writes its files and replies with the standard JSON
   document: `status`, `data`, `output_files`, `params` (including `params.ignored`), `summary`, `analysis`.
5. The portal returns the reply; the agent reads it as the observation of that step, marked as tool data so it cannot
   write the next step. Post-analysis then picks the result files up from `output_files`.

## Skill categories and routing

`agent/skills/` files every MCP function under an analysis category: `spatial_clustering`, `deconvolution`,
`svg_detection`, `cell_segmentation`, `spatial_alignment`, `spatial_communication`, `spatial_analysis`,
`data_conversion` and `omics`. Each category is a `BaseSkill` subclass with an `mcp_tools.py` that maps server key →
MCP function, full name, one-line description, GPU flag and priority. `SkillRegistry.create_default()` assembles
them.

The catalogue exists because an MCP tool is invisible to the agent's primary tool index, which is built from
`tool/tool_description/*.py` (the in-process tools). The agent renders the whole skills catalogue into one string
that the retriever scores on every query, so a function that is not filed here can be callable yet never chosen. The
same categories are the menu `sog-setup` offers, and the grouping of the tool pages here. Routing data beyond the
catalogue, an empirical leaderboard of methods per task and the data-validation rules, lives in
`spatialomicsgym/agent/`.

## Tools the agent creates

With `SOG_TOOL_CREATION_ENABLED=true`, the agent can build a new MCP tool from a method's GitHub repository: it writes
a portal, a worker and an env recipe into `agent/tools_user/` (the same three files, flat, under the names
`<id>_mcp_server.py`, `<id>_worker.py`/`.R`, `<id>_env.yaml`), builds a `user_<id>` conda environment, tests it, and
wires it into `mcp_config_user.yaml`. The library behind this lives in the same directory: `knowledge_manager.py`
(creation-time knowledge, backups, the install log), `trash_manager.py` (two-stage trash), `memory_manager.py`
(memory of earlier attempts, signed), `self_review.py` (diagnoses a failed creation test), `declarative.py`
(declarative tools with no env and no subprocess) and `user_skill.py` (exposes created tools to the skill registry).
The broker (`spatialomicsgym/agent/broker.py`) gates what the creation may write and install: only files whose names
match its allowlist, only packages from named indexes and channels, only repositories named in the egress policy.

`sog-setup pack` carries created tools to another machine; `sog-setup`'s `rebase_user_config` repoints their paths
there. The machine state beside them (`.knowledge/`, `.memory/`, `install_log.json`, `.trash/`) never moves on its own.

## Adding a method to the platform

In short: write the portal and the worker in `agent/tools/`; register the server and its functions in
`mcp_config.yaml`; file each function under its category in `agent/skills/<category>/mcp_tools.py`; on a machine that
has the env, `sog-setup capture --only <key>`; regenerate `agent/tools/CATALOG.md` with
`python -m sog_install.tools_catalog --write`; then `sog-setup --only <key>` and `sog-setup conncheck`. The
[contribution guide](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/CONTRIBUTION.md) has the full
checklist, including how to add an in-process tool, a dataset, or software to the agent environment.
