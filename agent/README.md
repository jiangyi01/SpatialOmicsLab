# agent/: the agent package, its tools and their configuration

**What this is.** Everything the analysis agent runs from: its Python package, the shipped MCP tools, the tools it creates on this machine, the MCP configuration, the skill catalogue and the benchmark harness.
**How runtime finds it.** By name and place. Each part below is a top-level import name once `agent/` is on `sys.path`, a path that other code joins, or both.
**What must not change.** The names and places of these parts, and the rules under [What must not move](#what-must-not-move).

## The parts

| Part | What it holds | How runtime finds it | Map |
|---|---|---|---|
| `spatialomicsgym/` | The agent package: the ReAct loop and prompt builder, the LLM providers, the in-process tool catalogue, the know-how corpus, post-analysis and reports | Import name `spatialomicsgym` (editable install or wheel). Its locator modules (`layout.py`, `platform_root.py`, `mcp_config_path.py`, `mcp_user_config.py`, `source_pin.py`) find the other parts from their own place in the tree | [README](spatialomicsgym/README.md) |
| `tools/` | One MCP portal (`<key>_mcp_server.py`) per analysis method, usually with a worker (`<key>_worker.py` or `<key>_worker.R`) that runs in the method's conda env, and the helpers they share | By path. `MCP_server/mcp_config.yaml` names each portal and worker by absolute path; when such a path is stale, the healers find the file again by its basename in `tools/`. A wheel carries a flat copy | [README](tools/README.md), [catalogue](tools/CATALOG.md) |
| `tools_user/` | The tool-creation library (tracked) and the tools the agent creates on this machine (not tracked) | Import name `tools_user`, only while `agent/` is on `sys.path` (no installed finder maps it); and by path, as the sibling of `tools/` and `MCP_server/` | [README](tools_user/README.md) |
| `MCP_server/` | Configuration only: the canonical `mcp_config.yaml` and the machine-local `mcp_config_user.yaml` overlay | By path: relative to the package, through the `SOG_MCP_CONFIG` pointer, or the `--mcp` argument. With `tools/` it marks the instance root | [README](MCP_server/README.md) |
| `skills/` | The skill domains that file each MCP function under an analysis category, for tool routing | Import name `skills` (editable install or wheel) | [README](skills/README.md) |
| `benchmarks/` | The benchmark harness: the evaluator that scores tool outputs, the SVG gene reader, the smoke-test driver, and an older strategy/runner framework | Import name `benchmarks` (editable install or wheel). At run time only the tuning objectives import it | [README](benchmarks/README.md) |

## Who puts `agent/` on `sys.path`

`tools_user` and `tools` are importable only while `agent/` is on `sys.path`. The editable install maps
`spatialomicsgym`, `skills` and `benchmarks` (with `sog_install`) by absolute path, so moving
one of those three directories also needs `pip install -e .` again in every env that uses it.

| Site | What it does |
|---|---|
| `pyproject.toml:235` | pytest `pythonpath = ["agent"]` |
| `pyproject.toml:189-190` | packaging: `where = ["agent", "install"]`; the package names come from the directories directly under `agent/` |
| `install/sog_install/__init__.py:33`, `install/sog_install/constants.py:158-175` | `import sog_install` calls `ensure_repo_importable()`, which appends `agent/` (CLI, `sog-setup`, and the tool-creation playbooks) |
| `agent/spatialomicsgym/agent/stcoscientist.py:678-682` | inserts `agent/` at the front of `sys.path`, then imports `skills`; any failure is caught and the skill catalogue is dropped with a printed warning |
| `agent/spatialomicsgym/tool/repl_client.py:118-129`, `agent/spatialomicsgym/postanalysis/as_owner.py:40-53` | child processes get `PYTHONPATH=<repo>/agent:<repo>/install` |

## Tracked, generated, machine state

| Kind | Where | In git |
|---|---|---|
| Tracked code and configuration | the package, the portals and workers in `tools/`, the tool-creation library in `tools_user/`, `skills/`, the `benchmarks/` code, `MCP_server/mcp_config.yaml` | tracked, and shipped by a clone |
| Generated | `tools/CATALOG.md` (generated from the tool specs and the canonical config; edit those, not the file); each created tool's `<id>_*` files in `tools_user/`; `MCP_server/mcp_config.yaml`, which the finalize phase of `sog-setup` rewrites with this machine's paths | `CATALOG.md` and `mcp_config.yaml` tracked; created tools not tracked |
| Vendored upstream sources | `tools/third_party/`, `tools_user/vendor_<id>/` | not tracked: a copy of the checkout carries them, a clone does not |
| Machine state | `MCP_server/mcp_config_user.yaml`; `tools_user/.knowledge/`, `.memory/`, `install_log.json`, `declarative/`, `.trash/`; `benchmarks/results/` | not tracked; never copy it between machines (`tools_user/.memory/.secret` is a signing key) |
| Caches | `__pycache__/`, `.ruff_cache/` | not tracked; safe to delete |

## What must not move

1. **`tools/` stays flat.** Each portal runs as a script and imports its helpers from its own directory
   (`from base_mcp import ...`, `agent/tools/scanpy_spatial_mcp_server.py:6`). Every healer joins
   `tools/<basename>`: `agent/tools/base_mcp.py:136-153` (`_local_worker_script`),
   `agent/spatialomicsgym/agent/mcp_integration.py:555-585`. The installer flattens any path that contains
   `/tools/` (`install/sog_install/wiring.py:114-123`), and the wheel payload patterns do not recurse
   (`agent/spatialomicsgym/platform_root.py:116-117,344-373`). A file in a subfolder is not found, not shipped,
   and its configured path is rewritten to one that does not exist.
2. **`tools/`, `tools_user/` and `MCP_server/` stay siblings directly under `agent/`.** `_local_worker_script`
   looks in `<agent>/tools` and then `<agent>/tools_user` (`agent/tools/base_mcp.py:146-149`); the agent and the
   installer do the same (`agent/spatialomicsgym/agent/mcp_integration.py:575`,
   `install/sog_install/conncheck.py:419`). The tool-creation managers write `<agent>/MCP_server/mcp_config_user.yaml`
   from their own directory (`agent/tools_user/knowledge_manager.py:73`, `agent/tools_user/trash_manager.py:58`),
   and `agent/tools_user/base_mcp.py` and `worker_utils.py` are relative symlinks into `../tools/`. A wheel keeps
   the same three names side by side (`agent/spatialomicsgym/platform_root.py:147-153`).
3. **`MCP_server/mcp_config.yaml` together with `tools/` is the instance-root marker.**
   `agent/spatialomicsgym/platform_root.py:155,215-218` (`_carries_the_trees`) decides whether a directory is a
   checkout or a seeded home. Without the marker, `running_from_checkout()` turns False and `instance_root()`
   silently falls through to the working directory and then to `~/.spatialomicsgym`, and with it the state
   directory, the recipes and every healer that keys on them. The path also appears in model-visible prompt text (`agent/spatialomicsgym/agent/prompt_builder.py:2288,2324`).
4. **Never add a file to `spatialomicsgym/know_how/`, not even a README.** The know-how pin is a filesystem glob
   over `know_how/*.md`, not a git listing (`agent/spatialomicsgym/source_pin.py:90-102`), and the loader puts
   every `*.md` there into every system prompt (`agent/spatialomicsgym/know_how/loader.py:93-104`).
5. **Any rename, move, addition or deletion of a `.py` file under `spatialomicsgym/` or `tools/` changes
   `AGENT_SRC_HASH`.** The pin folds each file's content together with its repo-relative path, and it counts
   untracked files that git does not ignore (`agent/spatialomicsgym/source_pin.py:58-87`). Files that are not
   `.py`, such as this README, do not count. Read the current values with `python -m spatialomicsgym.source_pin --json`.
6. **The import names `tools_user`, `skills` and `benchmarks` are fixed.** The tool-creation playbooks the agent
   executes import `tools_user.*` by name (`agent/spatialomicsgym/know_how/add_new_mcp_tool*.md`), as do the
   broker, the REPL and the portal. Every importer of `skills` catches its failure and carries on without the
   catalogue (`agent/spatialomicsgym/agent/stcoscientist.py:682-695`), so a broken name passes silently. The
   tuning objectives import `benchmarks` (`agent/spatialomicsgym/tuning/objectives.py:69-99`). One worker also
   imports `tools.ucd_token` by package name (`agent/tools/ucdeconvolve_worker.py:533-536`).
7. **`benchmarks/results` is a name the code relies on.** The post-analysis write guard refuses any path in which
   `benchmarks` is directly followed by `results` (`agent/spatialomicsgym/postanalysis/sources.py:318-334`), the
   tuning cache defaults to `agent/benchmarks/results/tuning` (`agent/spatialomicsgym/tuning/persistence.py:51`),
   and a dataset path containing `benchmarks/` routes tuning into benchmark mode
   (`agent/spatialomicsgym/tuning/mode_router.py:36`).

## Module names that moved inside the package

Some modules of `spatialomicsgym/` were grouped into role folders (`providers/`, `turn/`, `contracts/`,
`software_catalog/`, `policy/`, `legacy/`). Their old import names still work and give the same module object
(`spatialomicsgym.llm` is `spatialomicsgym.providers.llm`), through the table in
`agent/spatialomicsgym/_aliases.py`. The package README lists every old name with the file it now names
([Old import names keep working](spatialomicsgym/README.md#old-import-names-keep-working)).

## See also

- [README.md](../README.md): the repository layout and how to install and run the system.
- [CONTRIBUTION.md](CONTRIBUTION.md): adding a tool, data, software or a benchmark.
