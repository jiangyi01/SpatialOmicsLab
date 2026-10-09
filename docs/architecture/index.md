# How it fits together

This section is the map of the repository and of a run: what the parts are, where they live, how a question becomes
an analysis, and which environment each piece runs in. The two pages after it go deeper into the
[agent package](agent-package.md) and the [MCP tools](mcp-tools.md).

## Repository layout

| Path | Contents |
|---|---|
| `agent/spatialomicsgym/` | The `spatialomicsgym` package: the ST-Coscientist agent, the terminal chat, the in-process research tools, the know-how corpus, post-analysis and report rendering. |
| `agent/tools/` | The MCP servers (*portals*) and the per-method *workers* they call. |
| `agent/MCP_server/` | `mcp_config.yaml`, the list of MCP servers the agent can wire, and the machine-local overlay `mcp_config_user.yaml`. |
| `agent/skills/` | The skill categories that group the MCP functions, for tool routing. |
| `agent/tools_user/` | The tool-creation library, and the tools the agent creates at run time on this machine. |
| `agent/benchmarks/` | The benchmark harness and evaluator. |
| `install/sog_install/` | The `sog-setup` installer. |
| `install/recipes/` | One spec per tool (`tool_specs/<key>.yaml`) and the conda recipe it is built from (`tool_specs/env/`). |
| `figures/`, `THIRD_PARTY_LICENSES/`, `license_info.md` | The framework figure; the licenses of vendored code and of the integrated datasets. |
| `docs/` | This documentation. |

Two source roots, `agent/` and `install/`, hold the importable packages: `spatialomicsgym`, `skills` and
`benchmarks` under the first, `sog_install` under the second (`pyproject.toml`, `[tool.setuptools.packages.find]`).
`agent/tools`, `agent/tools_user` and `agent/MCP_server` are not packages: the agent reaches them by path, and a
wheel carries a read-only copy of them.

## The life of a question

The framework figure on the [front page](../index.md) shows four stages. In code they are:

1. **The question arrives** through a front door: the terminal chat (`spatialomicsgym.chat_cli`) or the Python API
   (`STCoscientist.go` / `go_stream`). The front door loads `.env`, builds the agent, and wires the MCP tools when
   asked to (`--mcp`, `add_mcp()`).
2. **The prompt is enriched** before the loop starts. A question guard keeps conceptual questions from triggering
   tools; parameter validation spots format mismatches and missing references; spatial data paths named in the
   question are diagnosed (format, coordinates, counts); and, when the session has memory, the recap of earlier turns
   is attached. With the tool retriever on (the default), one LLM call pre-selects the tools, datasets and know-how
   documents this task may use, so the system prompt carries what is relevant rather than everything.
3. **The ReAct loop runs** (`spatialomicsgym.agent.stcoscientist`, on LangGraph). Each turn the model either writes
   an action (Python for its REPL, an R or bash cell, or an MCP tool call) or a final answer. Actions are executed,
   the observation goes back into the conversation, and the loop continues until the answer. Tool output is marked
   as data so it cannot write the agent's next step. A step is bounded by `SOG_TIMEOUT_SECONDS`.
4. **Tools run where their dependencies are.** An in-process tool is a Python function imported into the agent's
   own interpreter. An MCP tool is a portal process the agent started, which runs the method's worker in the
   method's own conda environment and returns a JSON reply. Both kinds of output are files on disk plus the reply the
   model reads.
5. **Post-analysis** (`spatialomicsgym.postanalysis`) scans each result, draws the figures that fit the task type,
   reviews them and may launch a bounded follow-up; `spatialomicsgym.report` renders the self-contained
   `report.html`. See [Results and reports](../usage/results.md).
6. **Self-repair, when needed.** A tool whose environment fails to import is reported to the model once, with the
   general-purpose environment offered for an in-process redo that the answer must disclose; a turn that ends on a
   give-up gets one more bounded round. Both are off under benchmarking, so a scored run measures the platform as
   shipped.

## Environments

Three kinds of environment, with different owners:

| Environment | Holds | Built by |
|---|---|---|
| The **agent environment** (`spatialomicsgym_env`) | The package, LangChain/LangGraph, scanpy/anndata and the in-process tools' dependencies. ~1.6 GB. No torch, no R. | `conda env create -f agent/spatialomicsgym/spatialomicsgym_env/spatialomicsgym_env.yml`, or `sog-setup`'s `base_env` phase. |
| One **tool environment per method** (`GraphST`, `cell2loc_env`, `spacexr`, ...) | That method and its stack, at the versions its recipe pins. | `sog-setup`, from `install/recipes/tool_specs/env/<key>.env.yaml`. |
| The **general-purpose environment** (`spatialomicsgym_env_general`) | The agent recipe plus an analysis stack (squidpy, decoupler, gseapy, scikit-image, ...), for the in-process redo above. | By hand, from `spatialomicsgym_env_general.yml`; `sog-setup` never touches it. |

This is what keeps methods with incompatible dependencies on one machine: nothing is ever imported into the agent
environment that belongs to a method. The price is that every MCP call crosses a process boundary, and every path in
the MCP config is absolute, which is why `sog-setup` rewrites the config for each machine and why the portals heal a
stale path by file name.

## Where runtime finds the parts

The agent locates its parts from its own position in the tree, through a few locator modules
(`layout.py`, `platform_root.py`, `mcp_config_path.py`, `mcp_user_config.py`): `agent/tools/` and
`agent/MCP_server/mcp_config.yaml` together mark a directory as an *instance root*, a checkout or a seeded `$SOG_HOME`.
Three rules follow from this, and they are the ones a contributor most needs to know:

- `agent/tools/` stays flat: every portal imports its helpers from its own directory, and every path-healer looks for
  a file by basename directly there.
- `tools/`, `tools_user/` and `MCP_server/` stay siblings directly under `agent/`.
- Nothing is ever added to `agent/spatialomicsgym/know_how/`: every `*.md` there enters every system prompt and the
  know-how fingerprint.

The complete list, with the code sites that depend on each rule, is in the READMEs inside the repository:
[`agent/README.md`](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/README.md),
[`agent/spatialomicsgym/README.md`](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/spatialomicsgym/README.md),
[`agent/tools/README.md`](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/tools/README.md),
[`agent/MCP_server/README.md`](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/MCP_server/README.md) and
[`agent/tools_user/README.md`](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/tools_user/README.md).

## Fingerprints

Two hashes identify what a run used: `AGENT_SRC_HASH`, over every `.py` file under `agent/spatialomicsgym/` and
`agent/tools/` (content and path), and `KNOW_HOW_HASH`, over the know-how corpus. Any added, moved or renamed `.py`
file changes the first; any `.md` under `know_how/` changes the second. `python -m spatialomicsgym.source_pin --json`
prints both.

## Security model

The agent executes code the model writes, as the user who started it. This repository ships the agent without the
web portal, so there is no privilege boundary: run it on a machine or container you are prepared to give it. What
*is* enforced is egress: the in-process tools reach the network through one HTTP client that applies a per-module
host allowlist, re-checks redirects, refuses private destinations and caps responses, from the policy in
`agent/spatialomicsgym/policy/egress.yaml`; package installs by the tool-creation broker are limited to named indexes,
channels and repositories. What is not covered (processes that bypass Python's HTTP client, reads, package contents)
is stated plainly in [SECURITY.md](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/SECURITY.md).
