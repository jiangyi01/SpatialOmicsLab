# Contributing to SpatialOmicsLab

Thank you for your interest in contributing to SpatialOmicsLab! We welcome contributions from the community.
Contributors with significant contributions will be invited to co-author related publications.

All paths below are relative to the repository root.

## Getting started

```bash
git clone https://github.com/jiangyi01/SpatialOmicsLab.git
cd SpatialOmicsLab
pip install -e ".[webui,dev]"
pre-commit install
```

Pre-commit runs biome, ruff (line length 120), private-key and credential scans, and refuses direct commits to
`main`, so work on a feature branch.

## Types of contributions

SpatialOmicsLab has two kinds of tools:

- **MCP tools** are analysis methods (GraphST, cell2location, ...). Each runs as an MCP server in its own conda
  environment. Add one when the method has its own dependencies.
- **In-process tools** are Python functions that run inside the agent's own interpreter: database queries,
  literature search and light utilities. Add one when the code needs nothing beyond the agent's environment.

### 🧬 Adding an analysis method (MCP tool)

Each method has a server key (for example `graphst`). The key is used in file names, in the MCP config and in the
tool spec.

**Steps:**

1. **Write the server and the worker** in `agent/tools/`:
   - `<key>_mcp_server.py` is the MCP server. Build it on `base_mcp.py`.
   - `<key>_worker.py` or `<key>_worker.R` is the worker, which runs in the method's own environment. Python workers
     reply through `worker_utils.py` and must stay importable by Python 3.7.
2. **Register the server** in `agent/MCP_server/mcp_config.yaml`: its launch command, the `<PREFIX>_WORKER` and
   interpreter variables, and each function with its description and parameters.
3. **List each function** in the skill for its category, under `agent/skills/<category>/mcp_tools.py`. That is where
   the catalog's category comes from.
4. **Capture the environment recipe.** On a machine that has the method's conda environment, run:

   ```bash
   sog-setup capture --only <key>
   ```

   This writes `install/recipes/tool_specs/<key>.yaml` and its environment recipe under
   `install/recipes/tool_specs/env/`, so a fresh clone can rebuild the environment.
5. **Regenerate the catalog:**

   ```bash
   python -m sog_install.tools_catalog --write
   ```

6. **Test it end to end:**

   ```bash
   sog-setup --only <key>
   sog-setup conncheck
   ```

   Then ask the agent a question that should use the tool.
7. **Submit a pull request.** Include the test prompt you used and a small public dataset (or a download link).

[agent/tools/README.md](tools/README.md) explains the server/worker conventions in more detail, and
[agent/tools/CATALOG.md](tools/CATALOG.md) lists every existing server. To add a new skill category, see
[agent/skills/README.md](skills/README.md).

### 🛠️ Adding an in-process tool

In-process tools are Python functions in `agent/spatialomicsgym/tool/<subject>.py`, organized by subject area
(database, genomics, literature, ...).

**Steps:**

1. **Implement and test** your function in the module for its subject.
2. **Add a tool description** to the `description` list in `agent/spatialomicsgym/tool/tool_description/<subject>.py`,
   following the existing entries. The agent registers every entry there by name.

   *Tip: use this helper to draft a description from your function's source:*

   ```python
   from spatialomicsgym.providers.llm import get_llm
   from spatialomicsgym.utils import function_to_api_schema

   llm = get_llm()  # uses SOG_SOURCE / SOG_LLM from your .env
   desc = function_to_api_schema(function_code, llm)
   ```

3. **Write a test prompt** that uses your tool and check that the agent picks it and runs it correctly.
4. **Submit a pull request** that includes the test prompt.

### 📊 Adding new data

**If the data source has a web API:**

1. **Check that it is new**, with no overlap with existing sources.
2. **Add a `query_<source>` function** to `agent/spatialomicsgym/tool/database.py`, following the other functions.
3. **Add its description** to `agent/spatialomicsgym/tool/tool_description/database.py`.

**If the data source has no API:**

1. **Check that it is new**, with no overlap with existing sources.
2. **Stage the file(s)** in the agent's data-lake directory, `<path>/spatialomicsgym_data/data_lake/`, where `<path>` is
   the agent's data directory (`SOG_DATA_PATH`, default `./data`). The agent scans this directory and shows every file
   in it to the LLM; there is no catalog to register in.
3. **If the data needs dedicated handling**, add a loader as an in-process tool (see above).
4. **Submit a pull request** that describes the dataset and how to obtain it. Include a download link if the data can be
   redistributed.

Also record the data source's license in [license_info.md](../license_info.md), especially if it restricts commercial
use.

### 💻 Adding software to the agent environment

This is for software that the agent's own code-execution environment needs. A method with its own dependencies
should be an MCP tool instead.

1. **Test locally** that it does not conflict with the existing environment.
2. **Add an installation script**, following `agent/spatialomicsgym/spatialomicsgym_env/new_software_v008.sh`.
3. **Add an entry** to `library_content_dict` in `agent/spatialomicsgym/software_catalog/env_desc.py`, so the agent
   knows the software is available.
4. **Submit a pull request** with the installation script.

### 🎯 Benchmarks

Benchmark code lives in `agent/benchmarks/`. Much of the older strategy/runner framework there is dormant, and the
dataset registry is no longer in the repository. Read [agent/benchmarks/README.md](benchmarks/README.md) for what is
live, and open an issue before starting a new benchmark. Do not add benchmarks to `agent/spatialomicsgym/legacy/`.

### 🐛 Bug fixes and enhancements

We welcome bug fixes and enhancements! For anything beyond a small fix, **open an issue first** to discuss it with
the SpatialOmicsLab team.

- Clearly describe the problem or the enhancement.
- Include tests when applicable.
- Follow the existing code patterns.
- Update the documentation if needed.

## Submission process

1. **Fork** the repository.
2. **Create a feature branch** from `main`.
3. **Make your changes**, following the guidelines above.
4. **Run the checks:** `ruff check .`, `pre-commit run --all-files`, and `sog-setup conncheck` if you touched tools.
5. **Submit a pull request** to [jiangyi01/SpatialOmicsLab](https://github.com/jiangyi01/SpatialOmicsLab) with a
   clear description.

The SpatialOmicsLab team will review your pull request and may ask for changes.

## Questions?

If you have questions about contributing, please open an issue on
[GitHub](https://github.com/jiangyi01/SpatialOmicsLab/issues).
