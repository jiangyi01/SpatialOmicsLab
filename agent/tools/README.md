# agent/tools: the built-in MCP portals

**What this is:** one MCP portal per analysis method (`<key>_mcp_server.py`), most with a worker script that runs
in the method's own conda env (`<key>_worker.py` or `<key>_worker.R`), plus the helpers they share.
[CATALOG.md](CATALOG.md) lists every portal by analysis category, with its worker, env and functions.

**How runtime finds it:** `agent/MCP_server/mcp_config.yaml` names each portal and each worker by path. The agent
starts a portal as `python <portal>`, so the portal's own directory is `sys.path[0]`, and the portal runs its worker
under the interpreter the config names.

**What must not change:** the directory stays flat, keeps the name `tools`, and stays a sibling of `tools_user/` and
`MCP_server/`. Any change to a `.py` file here changes the agent source pin (`AGENT_SRC_HASH`, computed by
`agent/spatialomicsgym/source_pin.py`); `.R` files and Markdown do not.

## Why the directory is flat

Four separate layers assume that every portal, worker and helper sits directly in this directory. A file moved into
a subfolder breaks each of them, and the category view a subfolder would give is in [CATALOG.md](CATALOG.md)
instead.

1. **Imports.** Every portal does `from base_mcp import ...` and nearly every Python worker does
   `from worker_utils import ...`. Both resolve only because the script's own directory is `sys.path[0]` (some
   workers also insert it explicitly).
2. **Path healing joins basenames.** When a recorded path is missing, each healer looks for the same file name
   directly in this directory, and none of them recurse:
   - `resolve_worker_script` and `_local_worker_script` in `agent/tools/base_mcp.py`, which search
     `tools/` and `tools_user/` next to their own file (so the two directories stay siblings);
   - `_rebase_script_args` in `agent/spatialomicsgym/agent/mcp_integration.py`;
   - `_rebased_script_args` in `install/sog_install/conncheck.py`;
   - `_worker_in_this_checkout` in `agent/spatialomicsgym/tuning/integration.py`.
3. **The installer flattens.** `rebase_tools_path` in `install/sog_install/wiring.py` rewrites any path that
   contains `/tools/` to this directory plus the file name, whether or not the path exists. The specs in
   `install/recipes/tool_specs/` record `worker_file` and `server_file` as bare file names. A subfolder path would be
   rewritten into a path that does not exist on the next `sog-setup` run.
4. **Packaging is not recursive.** `PLATFORM_PAYLOAD` in `agent/spatialomicsgym/platform_root.py` copies `*.py`,
   `*.R` and `.ruff.toml` from this directory only, and `MANIFEST.in` includes the same three patterns. A file in a
   subfolder is left out of the wheel, of a seeded home and of the sdist, without any error.

Two more couplings depend on the place and the name:
- `ucdeconvolve_worker.py` puts the parent of this directory on `sys.path` and imports `tools.ucd_token`.
- Workers that need an upstream source tree look for `third_party/<Repo>` next to their own file (below).

## File names

| Pattern | What it is |
|---|---|
| `<key>_mcp_server.py` | The portal: the MCP server the agent starts. It publishes the functions and runs the worker in a subprocess. |
| `<key>_worker.py` | A Python worker, run by the method's own interpreter. Must stay importable by Python 3.7 (see `.ruff.toml`). |
| `<key>_worker.R` | An R worker, run by the method's own `Rscript`. |

`<key>` is the server key in `agent/MCP_server/mcp_config.yaml` and in `install/recipes/tool_specs/<key>.yaml`.
The servers whose key differs from the file stem, the portal that nothing wires, and a worker shared by two portals
are listed in the notes of [CATALOG.md](CATALOG.md).

The portal finds its worker through two environment variables set in the config: `<PREFIX>_WORKER` holds the worker
path and `<PREFIX>_PYTHON` the interpreter. For most R workers `<PREFIX>_PYTHON` holds an `Rscript` path; a few
read `<PREFIX>_RSCRIPT` instead. The exact names for each server are recorded in its spec (`override_vars`,
`worker_var`).

## Shared helpers

| File | What it is | Who uses it |
|---|---|---|
| `base_mcp.py` | Portal support: creating the MCP server, resolving and healing the worker path, running the worker, parsing its JSON reply, returning errors as a dict. | Every portal. `agent/tools_user/base_mcp.py` is a symlink to it. |
| `worker_utils.py` | Worker support: the standard JSON reply (status, data, output files, params, summary) and the shared input readers and checks (coordinates, the counts matrix, the in-tissue filter, gene-ID matching). | The Python workers, and the user-created workers through the `agent/tools_user/worker_utils.py` symlink. |
| `eval_metrics.py` | Scores tool outputs: clustering (ARI, NMI), spatially variable genes (overlap, Moran's I), deconvolution (RMSE, correlation, JSD). | `eval_mcp_server.py`, `agent/spatialomicsgym/tuning/`, `agent/benchmarks/evaluation/`. |
| `result_collector.py` | Collects tool results, scores them with `eval_metrics.py` and writes a summary report. | `eval_mcp_server.py`; also runs on its own. |
| `ucd_token.py` | Finds and remembers a UCDeconvolve token so that each run does not need one passed in. | The `ucdeconvolve` portal and worker. |
| `build_spatial_library_registry.py` | Builds the `registry.json` of the spatial dataset library from a directory of `.h5ad` datasets. | `spatial_library_worker.py`; a developer command. |
| `h5ad_to_seurat.R` | Assembles a Seurat object from the flat files (MTX + CSV) that Python exports. | `data_converter_worker.py`. |
| `r_to_h5ad_converter.R` | Exports an R object (Seurat, matrix, data frame; `.rds` or `.rda`) to flat files that Python assembles into `.h5ad`. | `data_converter_worker.py`; the in-process converter in `agent/spatialomicsgym/tool/`. |
| `.ruff.toml` | Lints this directory for Python 3.7, the oldest interpreter a worker runs under, so an automatic fix never writes syntax a worker env rejects. | `ruff`, and `pre-commit`. |

## `third_party/`

Upstream source trees that some workers run from, one directory per method. It is git-ignored
(`agent/tools/third_party/` in `.gitignore`), so a fresh clone does not have it, and a README inside it would not be
shipped either. That is why it is described here.

- A worker that needs one looks for `third_party/<Repo>` next to its own file. Most accept an override in
  `<TOOL>_SRC` (for example `SEDR_SRC`); Celloscope and CellPie read `CELLOSCOPE_REPO` and `CELLPIE_REPO`.
- Some conda envs install the upstream package from this directory as an editable install. On a fresh clone, the
  env recipes in `install/recipes/tool_specs/env/` install it from its upstream repository instead.
- Nothing in `third_party/` is linted, pinned, packaged or shipped.

Any other git-ignored file here is local output, typically from a tool that ran with this directory as its working
directory. Nothing reads it.

## Adding a portal

1. Write `<key>_mcp_server.py` (built on `base_mcp`) and `<key>_worker.py` or `<key>_worker.R` (replying through
   `worker_utils` for Python), directly in this directory.
2. Add the server to `agent/MCP_server/mcp_config.yaml`: the launch command, `<PREFIX>_WORKER`, the interpreter
   variable, and each function with its description and parameters.
3. List each function in the skill for its category under `agent/skills/`. That is where the catalogue's category
   comes from.
4. On a machine that has the method's conda env, run `sog-setup capture --only <key>`. It writes
   `install/recipes/tool_specs/<key>.yaml`, the recipe a fresh clone rebuilds the env from.
5. Regenerate the catalogue:

   ```bash
   python -m sog_install.tools_catalog --write
   ```

Tools that users create while the agent runs go to `agent/tools_user/`, never here.
