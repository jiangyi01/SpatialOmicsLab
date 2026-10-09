# The agent package

`spatialomicsgym` is the agent core: the research loop and its prompt, the LLM providers, the in-process research
tools, post-analysis, reports, and the settings they all read. Its source root is `agent/`; a checkout uses an
editable install and a wheel puts the package in `site-packages`. The {doc}`API reference <../api/index>` documents
every module; this page is the map.

## Package map

```text
agent/spatialomicsgym/
│   top level: package bootstrap, locations, settings, entry points
├── __init__.py              package entry; binds clean_answer and __version__; installs the alias finder
├── _aliases.py              old import name -> new module, same object
├── version.py               __version__
├── config.py                settings: SpatialOmicsGymConfig and the default_config singleton
├── layout.py                where the checkout's parts are
├── platform_root.py         instance root and platform trees across checkout, wheel and SOG_HOME
├── paths.py                 output roots, results search roots, path display and scrubbing
├── mcp_config_path.py       finds mcp_config.yaml
├── mcp_user_config.py       finds mcp_config_user.yaml and the install log of created tools
├── chat_cli.py              the stcoscientist / sog-chat terminal chat
├── redaction.py             the one credential-shape list; also a log filter
├── source_pin.py            AGENT_SRC_HASH and KNOW_HOW_HASH
│
│   role folders
├── providers/               the LLM vendor layer (get_llm, per-provider quirks, backoff, streaming)
├── turn/                    reading one assistant turn: the action to run vs the final answer
├── contracts/               vocabularies several layers share: stream events, task types, generated reports
├── software_catalog/        the software and data-lake lists the system prompt shows
├── policy/                  security policy: egress rules and the bind-exposure check
├── legacy/                  Biomni-era code with no production caller
│
│   research loop
├── agent/                   ReAct loop, prompt builder, MCP wiring, broker, routing data
├── model/                   ToolRetriever (the name is historical)
├── know_how/                prompt corpus and its loader. Never add a file here
│
│   in-process tools and execution
├── tool/                    in-process tool catalogue + REPL runtime. Dotted names are model-facing
├── utils/                   execution, HTTP egress client, format probe, prompt formatting, child shims
│
│   results
├── postanalysis/            L1 engine and L2 review over tool outputs
├── report/                  L3: HTML presentation of a post-analysis manifest
├── research/                multi-round research loop, citations, research report
├── viz/                     analysis-aware visualisation; render/ and ccc/ (cell-cell communication)
├── spatial3d/               serial-section 3D: diagnose, align, validate
│
│   scored runs
├── benchmarking/            scored-run output inspection and standardisation
├── tuning/                  hyperparameter tuning (modes, Optuna, persistence)
│
│   non-code
├── data/                    package data (leaderboard cache, SVG routing table, spatial-library registry)
└── spatialomicsgym_env/     conda recipes and setup scripts
```

## Subpackages by role

Labels: **code**; **data**; **model-facing** (the name appears in the system prompt, so renaming it changes every
prompt and the code the model writes).

| Subpackage | What it holds |
|---|---|
| `agent/` (code) | The research loop (`stcoscientist.py`), the prompt (`prompt_builder.py`), execution (`execution.py`), MCP wiring (`mcp_integration.py`, `mcp_config_merger.py`, `tool_management.py`), the privilege broker for tool creation (`broker.py`), routing data (`empirical_leaderboard.py`, `data_validation.py`) and turn helpers (`conversation.py`, `premise_check.py`, `rescue.py`, `env_fallback.py`, `tool_call_memo.py`, `usage.py`). Importing any module here loads the full LangChain stack and, unless `SOG_SKIP_DOTENV` is set, the install's `.env`. |
| `model/` (code) | `retriever.py`: `ToolRetriever`, the one LLM call that pre-selects the tools, data and know-how a turn may use. |
| `know_how/` (code + data) | Tier-1 playbooks (`*.md` at the top, every one of them in every prompt), tier-2 `packs/` (merged external skill packs), `resource/` (files the playbooks read). `enrolment.py` decides which documents enter the prompt. |
| `tool/` (code + data, model-facing) | The in-process tool catalogue: one module per subject (`database.py`, `literature.py`, `genomics.py`, ...), each described in `tool/tool_description/`, plus `schema_db/`, `protocols/` and `omics_scripts/`. The REPL runtime lives here too: `support_tools.py` (`run_python_repl`), `repl_client.py`, `repl_host.py`, `general_env.py`, `tool_registry.py`. The `spatialomicsgym.tool.<module>` names are printed into the prompt as import lines. |
| `utils/` (code) | `execution.py` (subprocess runners for R, bash and CLI cells), `http_client.py` (the one outbound HTTP path; enforces `policy/egress.yaml`), `format_probe.py` (data-format detection), `formatting.py`, `tool_conversion.py` and `tool_discovery.py` (tool schemas), `file_io.py`, `obs_aliases.py`, `logging_utils.py`, and the shims put into model-launched child interpreters (`child_site/`, `anndata_compat.py`, `pandas_display.py`). |
| `postanalysis/` (code, model-facing) | The L1 engine (`engine.py`, `run_post_analysis`), L2 review and next step (`review.py`, `next_step.py`), `plots.py`, `tables.py`, `manifest.py` (writes `manifest.json`), and `tasks/` (one runner per task type). The prompt tells the model to call `from spatialomicsgym.postanalysis import run_post_analysis`. |
| `report/` (code) | L3 presentation of a manifest: `manifest.py` (read and validate), `render.py` (HTML), `discover.py` (find runs), `metrics.py`. Run as `python -m spatialomicsgym.report`. |
| `research/` (code) | The bounded multi-round research loop (`loop.py`), `citations.py`, `ledger.py`, `prompts.py`, and `report.py` (the research report). |
| `viz/` (code) | Analysis-aware visualisation: profiles, layers, specs, palettes, `pipelines.py`, `render/` and `ccc/` (one adapter per cell-cell communication method). |
| `spatial3d/` (code) | Serial-section 3D diagnosis, alignment and validation. |
| `benchmarking/` (code) | `output_inspector.py`, `output_standardizer.py`, `tool_output_registry.py`, `workflow_gates.py`, `svg_input_prep.py`. Used by post-analysis, tuning, the benchmark harness and the installer's import probe. |
| `tuning/` (code + data) | Hyperparameter tuning: modes, strategies (Optuna among them), objectives, persistence; defaults and search spaces in `configs/*.yaml`. |
| `providers/` (code) | `llm.py` builds the chat model (`get_llm`) and holds the per-provider quirks; `provider_names.py` is the stdlib-only provider-name rules shared with the installer; `provider_backoff.py` waits out rate limits inside a turn; `responses_stream.py` streams Responses-API replies. |
| `turn/` (code) | `action.py` reads the action out of an assistant turn (`<execute>` blocks, fences, runnable code, and which wins against a final answer); `answer.py` cleans the final answer for display (`clean_answer`). |
| `contracts/` (code) | `stream_events.py` (the streaming event vocabulary), `task_types.py` (the canonical analysis task names), `generated_report.py` (recognises a report this system wrote, so it is not read back as tool output). |
| `software_catalog/` (code) | The software and data-lake lists for the system prompt: `env_desc.py` for academic mode, `env_desc_cm.py` for commercial mode. |
| `policy/` (code + data) | `egress.yaml` is the egress policy and `egress.py` loads it; `netbind.py` answers whether a bind host is reachable from outside the machine. |
| `legacy/` (code) | Biomni-era code with no production caller (task extractors, HLE and LAB-Bench loaders, a sample MCP server). Importable under old and new names; not documented here. |

## Names that are easy to confuse

| Name | What it is | Not to be confused with |
|---|---|---|
| `agent/` (the subpackage) | The research loop: `agent/stcoscientist.py`, the prompt builder, MCP wiring, the broker. | The repository folder `agent/`, which holds this package next to `agent/tools`, `agent/MCP_server`, `agent/skills`, ... |
| `model/` | `ToolRetriever`, the resource pre-selection call. The name is historical. | A machine-learning model. The chat model is built in `providers/llm.py`. |
| `benchmarking/` | `spatialomicsgym.benchmarking`: inspects and standardises a tool's output for a scored run. | `agent/benchmarks`, the separate top-level package `benchmarks`: the harness, workflows, evaluation and recorded results. |
| `data/` (in the package) | Read-only package data: the leaderboard cache, the SVG routing table, the curated spatial-library registry. | The runtime data directory (`SOG_DATA_PATH`), where datasets are read and `outputs/` is written. |
| `tool/` | The in-process tool catalogue: Python functions the model imports and calls in its REPL, plus the REPL runtime. | `agent/tools`, the MCP portals and workers, each a separate process in its own conda env; `agent/tools_user`, tools created at run time. |
| `execution.py` (two files) | `agent/execution.py`: the loop's execution graph and workflow settings. | `utils/execution.py`: subprocess runners for R, bash and CLI cells. |
| `manifest.py` (two files) | `postanalysis/manifest.py` writes `manifest.json`. | `report/manifest.py` reads and validates it for the HTML report. |
| "skill" | `agent/skills`: the routing catalogue over the MCP functions (`SkillRegistry`). | `know_how/packs/`: external skill packs (tier-2 know-how); `tool/transcriptomics_skills.py` and `tool/omics_skills.py`: in-process tool modules; `agent/tools_user/user_skill.py`: the skill for runtime-created tools. |
| "catalog" | `software_catalog/`: the software and data-lake lists the prompt shows. | `agent/tools/CATALOG.md`: the generated list of MCP portals; the `tool/` catalogue (`tool/tool_description/`); the skills catalogue the retriever scores. |
| `spatialomicsgym_env/` | A folder of conda recipes and setup scripts inside the package. | The conda environment named `spatialomicsgym_env`, which one of those recipes creates. |

## Old import names keep working

Several modules moved into the role folders (`providers/`, `turn/`, `contracts/`, `software_catalog/`, `policy/`,
`legacy/`). Every old dotted name still imports and gives the very same module object as the new one, through the
table in `_aliases.py` and a finder that sits first on `sys.meta_path`: `spatialomicsgym.llm` *is*
`spatialomicsgym.providers.llm`, `spatialomicsgym.answer` *is* `spatialomicsgym.turn.answer`, and so on. Module
state exists once, `mock.patch` on either name patches the one real module, and `-m` with an old name runs the moved
file. What an alias does not cover: loading by file path, `__file__` arithmetic, and editors' go-to-definition. The
full table is in
[`agent/spatialomicsgym/README.md`](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/spatialomicsgym/README.md#old-import-names-keep-working).

## Rules for changing the package

- Never add a file to `know_how/`: `KNOW_HOW_HASH` and the prompt both come from a filesystem glob of `know_how/*.md`.
- Any `.py` added, moved or renamed changes `AGENT_SRC_HASH`.
- A role folder's `__init__.py` holds a docstring only, so the light import paths (the installer's stdlib-only probe,
  the bare-`python3` log redactor) stay light.
- The model-facing names stay: `spatialomicsgym.tool.<module>` and `spatialomicsgym.postanalysis.run_post_analysis`
  are in the system prompt.
- A module that moves gets an `ALIASES` entry, keeps its file name, and leaves nothing behind at its old path.
