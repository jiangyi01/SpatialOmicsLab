# `spatialomicsgym`: the agent package

**What this is.** The agent core: the research loop and its prompt, the LLM providers, the in-process research tools,
post-analysis, reports and the settings they all read.
**How runtime finds it.** As the import package `spatialomicsgym`, with `agent/` as its source root
(`pyproject.toml`, `[tool.setuptools.packages.find] where`; pytest's `pythonpath = ["agent"]`). A checkout uses an
editable install; a wheel puts the package in site-packages.
**What must not change.** The files listed under [Top level](#top-level-the-files-that-stay) stay where they are.
Nothing is added to `know_how/`. Every old import name keeps resolving ([alias table](#old-import-names-keep-working)).

Paths in this file are relative to this directory (`agent/spatialomicsgym/`). Paths outside the package are written
from the repository root: they start with `backend/`, `install/`, `frontend/`, `test/`, or with `agent/tools`,
`agent/tools_user`, `agent/MCP_server`, `agent/skills` or `agent/benchmarks`; root files such as `pyproject.toml`
are named bare.

- [Glossary of confusing names](#glossary-of-confusing-names)
- [Package map](#package-map)
- [Top level: the files that stay](#top-level-the-files-that-stay)
- [Role folders](#role-folders)
- [Subpackages by role](#subpackages-by-role)
- [Data and other non-code trees](#data-and-other-non-code-trees)
- [`legacy/`](#legacy)
- [Old import names keep working](#old-import-names-keep-working)
- [Rules for changing this package](#rules-for-changing-this-package)
- [See also](#see-also)

---

## Glossary of confusing names

| Name | What it is | Not to be confused with |
|---|---|---|
| `agent/` (this package's subpackage) | The research loop: `agent/stcoscientist.py`, the prompt builder, MCP wiring, the privilege broker | The repository folder `agent/`, which holds this package next to `agent/tools`, `agent/tools_user`, `agent/MCP_server`, `agent/skills` and `agent/benchmarks` |
| `model/` | `model/retriever.py` (`ToolRetriever`): one LLM call that pre-selects the tools, data and know-how a turn may use. The name is historical | A machine-learning model. The chat model itself is built in `providers/llm.py` |
| `benchmarking/` | `spatialomicsgym.benchmarking`: finds, inspects and standardises a tool's output for a scored run (output inspector, standardiser, workflow gates, tool-output registry). Off by default | `agent/benchmarks`, the separate top-level package `benchmarks`: the benchmark harness, workflows, evaluation and recorded results |
| `data/` (here) | Read-only package data that ships with the code: the leaderboard cache, the SVG routing table, the curated spatial-library registry | The repository-root `data/`, which is runtime state next to `.sog_setup/` and `work/` (see the comment above `_part` in `layout.py`) |
| `tool/` | The in-process tool catalogue: Python functions the model imports and calls inside its REPL, plus the REPL runtime | `agent/tools`, the MCP portals and their workers, each run as a separate process in its own conda env; `agent/tools_user`, tools created at runtime |
| `execution.py` (two files) | `agent/execution.py`: the loop's execution graph and workflow settings; it re-exports the action readers from `turn/action.py` | `utils/execution.py`: subprocess runners for R, bash and CLI cells (timeouts, UTF-8 locale for R, partial output) |
| `manifest.py` (two files) | `postanalysis/manifest.py`: writes the post-analysis `manifest.json` (the producer) | `report/manifest.py`: reads, validates and normalises it for the HTML report and the portal (the consumer). `viz/manifest_io.py` is where the visualisation pipelines write their own manifest |
| "report" (three places) | `report/`: the HTML presentation of a post-analysis manifest | `research/report.py`: the document a research run hands back; `contracts/generated_report.py`: the rule that recognises a report this system wrote, so it is not read back as tool output |
| "task" (three places) | `contracts/task_types.py`: the canonical analysis task names | `postanalysis/tasks/`: one post-analysis runner per task type; `legacy/task/`: Biomni-era HLE and LAB-Bench loaders |
| "skill" (six places) | `agent/skills`: the top-level package `skills`, a routing catalogue over the MCP functions (`SkillRegistry`; see `agent/skills/README.md`) | `know_how/packs/`: merged external "skill packs" (tier-2 know-how); `know_how/resource/omics_skills/`: reference notes the loader never reads; `tool/transcriptomics_skills.py` and `tool/omics_skills.py`: in-process tool modules; `backend/sog_portal/skills_api.py`: the portal's skills API; `agent/tools_user/user_skill.py`: `UserToolSkill`, the skill for runtime-created tools |
| "catalog" (four places) | `software_catalog/`: the software and data-lake lists the system prompt shows | [`agent/tools/CATALOG.md`](../tools/CATALOG.md): the generated list of MCP portals by category; the `tool/` catalogue (`tool/tool_description/`, read by `utils/tool_discovery.py`); the skills catalogue, the `skills_catalog` text that `agent/execution.py` builds from `agent/skills` for the retriever |
| `spatialomicsgym_env/` | A folder of conda recipes and setup scripts inside the package. It is not a Python package | The conda environment named `spatialomicsgym_env`, which one of these recipes creates |

---

## Package map

Every entry of this directory, grouped by role. `(a)`–`(d)` is the reason a top-level file stays on top (see the
next section).

```
agent/spatialomicsgym/
├── README.md                this file
│
│   top level: package bootstrap, locations, settings, entry points
├── __init__.py              (a)  package entry; binds clean_answer, __version__; installs the alias finder
├── _aliases.py              (a)  old import name -> new module, same object
├── version.py               (a)  __version__ (the wheel's version attribute)
├── config.py                (d)  settings: SpatialOmicsGymConfig and the default_config singleton
├── layout.py                (b)  where the checkout's parts are (frontend, backend, agent, tools, ...)
├── platform_root.py         (b)  instance root and platform trees across checkout, wheel and SOG_HOME
├── paths.py                 (b)  output roots, results search roots, path display and scrubbing
├── mcp_config_path.py       (b)  finds mcp_config.yaml
├── mcp_user_config.py       (b)  finds mcp_config_user.yaml and the install log; user tool names
├── chat_cli.py              (c)  the stcoscientist / sog-chat terminal chat and shared front-door helpers
├── redaction.py             (b)(c) the one credential-shape list; also a stdin-to-stdout log filter
├── source_pin.py            (b)(c) AGENT_SRC_HASH and KNOW_HOW_HASH for scored runs
│
│   role folders
├── providers/               the LLM vendor layer
├── turn/                    reading one assistant turn: the action to run vs the final answer
├── contracts/               vocabularies several layers share and none owns
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
├── postanalysis/            L1 engine and L2 review over tool outputs. Entry point is model-facing
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
├── data/                    package data (see "Data and other non-code trees")
└── spatialomicsgym_env/     conda recipes and setup scripts
```

---

## Top level: the files that stay

On 2026-10-07 the package top level went from 27 files to these 12. A module stays on top only if one of these holds:

- **(a)** it is package bootstrap or a packaging attribute;
- **(b)** it computes a location, from its own depth or because other code loads it by file path;
- **(c)** an installed entry point or a live launcher names it;
- **(d)** it is the settings singleton that the executed playbooks import.

Everything else lives in a folder named for its role.

| File | Rule | Why it cannot move |
|---|---|---|
| `__init__.py` | (a) | Package entry. Binds `clean_answer` (and `spatialomicsgym.answer`), `__version__`, mirrors `BIOMNI_*` variables to `SOG_*`, and installs the alias finder. Must stay stdlib-only: the bare-`python3` log redactor imports it |
| `_aliases.py` | (a) | The meta-path finder behind every old import name. Installed by `__init__.py` before anything asks for an old name |
| `version.py` | (a) | `pyproject.toml:175` reads `spatialomicsgym.version.__version__` as the distribution version |
| `config.py` | (d) | `default_config` is built at import (`config.py:437`) and snapshots the environment, so the `.env` must be loaded first. It is imported across the package, the portal and the tool-creation library, and code in the `know_how/` playbooks imports `spatialomicsgym.config` by name |
| `layout.py` | (b) | `_HERE = Path(__file__).resolve().parent` and `PACKAGED_FRONTEND = _HERE / "_frontend"` (`layout.py:37,45`): the wheel's frontend copy sits at the package root |
| `platform_root.py` | (b) | `parents[2]` is the checkout root (`platform_root.py:244`) and `parent / "_platform"` the wheel's payload (`:295`). `setup.py:46-48` loads this file by path at wheel build |
| `paths.py` | (b) | Computes where output lives and how a path is shown. It stays beside the other location modules, and `test/test_output_roots_agree.py` pins it light (no numpy, pandas or langgraph on import) |
| `mcp_config_path.py` | (b) | `Path(__file__).resolve().parents[1]` joined with `MCP_server/mcp_config.yaml` (`mcp_config_path.py:68`) |
| `mcp_user_config.py` | (b) | `_REPO_ROOT = parents[1]` (`mcp_user_config.py:32`). `test/smoke/registry.py:54` loads it by path, and `sog-setup capture` runs that registry (`install/sog_install/capture.py:418`) |
| `chat_cli.py` | (c) | The `stcoscientist` and `sog-chat` console scripts name `spatialomicsgym.chat_cli:main` (`pyproject.toml:110-111`). The portal and the installer also import its front-door helpers |
| `redaction.py` | (b)(c) | `.pre-commit-config.yaml:80` loads it by file path. `restart_portal.sh:79` pipes the portal log through `-m spatialomicsgym.redaction` under a bare `python3` with only `agent/` on `PYTHONPATH` |
| `source_pin.py` | (b)(c) | `REPO = Path(__file__).resolve().parents[2]` (`source_pin.py:44`). Run as `python -m spatialomicsgym.source_pin` to compute the pins for a scored run |

---

## Role folders

Each role folder's `__init__.py` holds a docstring and nothing else (`policy/` adds `from __future__ import
annotations`). That keeps the light import paths light: the installer's stdlib-only probe imports
`providers/provider_names.py`, and `import spatialomicsgym` loads `turn/answer.py` under the bare-`python3` redactor.

| Folder | Files | What they do |
|---|---|---|
| `providers/` | `llm.py`, `provider_names.py`, `provider_backoff.py`, `responses_stream.py` | `llm.py` builds the chat model (`get_llm`) and holds the per-provider quirks. `provider_names.py` is the stdlib-only provider-name rules shared with the installer. `provider_backoff.py` waits out rate limits and transient errors inside a turn. `responses_stream.py` streams Responses-API replies and stops at the first stop sequence. Named `providers/`, not `llm/`, so that `spatialomicsgym.llm` can stay a module name |
| `turn/` | `action.py`, `answer.py` | `action.py` reads the action out of an assistant turn (`<execute>` blocks, fences, runnable code, and which wins against a final answer). `answer.py` cleans the final answer for display (`clean_answer`, `SOLUTION_TAG_RE`) |
| `contracts/` | `stream_events.py`, `task_types.py`, `generated_report.py` | `stream_events.py` is the portal's SSE frame-event vocabulary; `frontend/src/lib/streamEvents.ts` mirrors it. `task_types.py` is the canonical task-type table over `postanalysis/manifest.py`. `generated_report.py` recognises a report this system wrote. `contracts/__init__.py` must stay empty: `task_types` imports `postanalysis`, and benchmarking imports `generated_report` |
| `software_catalog/` | `env_desc.py`, `env_desc_cm.py` | The software and data-lake lists for the system prompt: `env_desc.py` for academic mode, `env_desc_cm.py` for commercial mode. Code mutates and compares these module-level dicts, so both import names must reach one object |
| `policy/` | `egress.yaml`, `egress.py`, `netbind.py` | `egress.yaml` is the egress policy and `egress.py` loads it; `pyproject.toml` ships the yaml as package data. `netbind.py` (`is_exposed_bind`) answers whether a bind host is reachable from outside the machine |
| `legacy/` | see [`legacy/`](#legacy) | Biomni-era code with no production caller |

---

## Subpackages by role

Labels: **code**, **data**, **dev-time** (run by hand, never imported at runtime), **model-facing** (the name appears in
the system prompt, so renaming it changes every prompt and the code the model writes).

| Subpackage | Labels | What it holds |
|---|---|---|
| `agent/` | code | The research loop (`stcoscientist.py`), the prompt (`prompt_builder.py`), execution (`execution.py`), MCP wiring (`mcp_integration.py`, `mcp_config_merger.py`, `tool_management.py`), the privilege broker (`broker.py`), routing data (`empirical_leaderboard.py`, `data_validation.py`) and turn helpers (`conversation.py`, `premise_check.py`, `rescue.py`, `env_fallback.py`, `tool_call_memo.py`, `usage.py`). `agent/__init__.py` imports `STCoscientist`, so importing any `agent/` module loads the full LangChain stack and, unless `SOG_SKIP_DOTENV` is set, the install's `.env`; `install/sog_install/wiring.py` loads `mcp_config_merger.py` by path for that reason |
| `model/` | code | `retriever.py`: `ToolRetriever`, the resource pre-selection call |
| `know_how/` | code + data | Tier-1 playbooks (`*.md` at the top, globbed non-recursively by `loader.py`), tier-2 `packs/` (merged external skill packs and `packs/MANIFEST.yaml`), and `resource/` (files the playbook code reads; never loaded as know-how). `enrolment.py` decides which documents enter the prompt. `merge_packs.py` is dev-time. The `*.md` glob feeds both `KNOW_HOW_HASH` and every prompt |
| `tool/` | code + data, model-facing | Domain modules whose functions the model imports in its REPL, each described in `tool/tool_description/`, with `schema_db/` (pickled schemas for `database.py`), `protocols/` (protocol texts) and `omics_scripts/` (helpers for `omics_skills.py`). The REPL runtime lives here too: `support_tools.py` (`run_python_repl`), `repl_client.py`, `repl_host.py` (the portal starts it as `-m spatialomicsgym.tool.repl_host`), `repl_protocol.py`, `general_env.py`, `tool_registry.py`, `conversion_record.py`. The `spatialomicsgym.tool.<module>` names are printed into the prompt as `Import file:` lines |
| `utils/` | code | `execution.py` (subprocess runners), `http_client.py` (the one outbound HTTP path; it enforces `policy/egress.yaml`), `format_probe.py` (data-format detection), `formatting.py` (prompt and transcript formatting), `tool_conversion.py` and `tool_discovery.py` (tool schemas; the `tool/` catalogue), `file_io.py`, `obs_aliases.py`, `logging_utils.py`, `data_processing.py` (Biomni-era gene-ID helpers), and the shims put into model-launched child Pythons: `child_site/`, `anndata_compat.py`, `pandas_display.py` (loaded by path from `child_site/sitecustomize.py`, so the three move together or not at all) |
| `postanalysis/` | code, model-facing | The L1 engine (`engine.py`, `run_post_analysis`), L2 review and next step (`review.py`, `next_step.py`), `plots.py`, `tables.py`, `manifest.py`, and `tasks/` (one runner per task type). Started as `-m spatialomicsgym.postanalysis` under the output owner's account (`as_owner.py`). The prompt tells the model to call `from spatialomicsgym.postanalysis import run_post_analysis` |
| `report/` | code | L3 presentation of a post-analysis manifest: `manifest.py` (read and validate), `render.py` (HTML), `discover.py` (find runs), `metrics.py`. Run as `-m spatialomicsgym.report` |
| `research/` | code | The bounded multi-round research loop (`loop.py`), `citations.py`, `ledger.py`, `prompts.py`, and `report.py` (the research report) |
| `viz/` | code | Analysis-aware visualisation: profiles, layers, specs, palettes, `pipelines.py`, `render/` and `ccc/` (cell-cell communication, with one adapter per method). Portal workers import it by name from another interpreter, and recipes check `import_check: spatialomicsgym.viz` |
| `spatial3d/` | code | Serial-section 3D diagnosis, alignment and validation, in the module order its `__init__.py` documents. Imported by name by portal workers. `_calibrate.py` is dev-time |
| `benchmarking/` | code | `output_inspector.py`, `output_standardizer.py`, `tool_output_registry.py`, `workflow_gates.py`, `svg_input_prep.py`. Used by post-analysis, tuning, the benchmark harness in `agent/benchmarks` and the installer's import probe |
| `tuning/` | code + data | Hyperparameter tuning: modes, strategies (Optuna among them), objectives, persistence; defaults and search spaces in `configs/*.yaml` |
| `data/` | data | See the next section |
| `spatialomicsgym_env/` | data | Conda recipes, requirement lists and setup scripts (see its own `README.md`). `run_core_tests.sh` finds the repository root three levels up, so the folder cannot move deeper |

Dev-time scripts, not imported at runtime: `know_how/merge_packs.py` (renders `know_how/packs/` from pinned upstream
clones) and `spatial3d/_calibrate.py` (measures the 3D classifier thresholds).

---

## Data and other non-code trees

These directories hold files that code finds by path. Each is found from a fixed anchor, so moving one means
changing its readers.

| Tree | Read by | Ships through |
|---|---|---|
| `know_how/*.md`, `know_how/packs/` | `know_how/loader.py:83` (`Path(__file__).parent`); `source_pin.py:97` for the pin | `MANIFEST.in:28` |
| `know_how/resource/` | code inside the playbooks, via `Path(know_how.__file__).parent / "resource"` | `MANIFEST.in:28` |
| `tool/tool_description/` | `utils/tool_discovery.py:8` (`Path(__file__).parent.parent / "tool" / "tool_description"`). A missing folder gives an empty catalogue, silently | `MANIFEST.in:8` (`*.py`) |
| `tool/schema_db/` | `tool/database.py` (`os.path.dirname(__file__)` joined with `schema_db`) | `MANIFEST.in:13` |
| `tool/protocols/` | `tool/protocols.py:184` (`os.path.dirname(spatialomicsgym.__file__)`: from the package root, not from the module) | `MANIFEST.in:29` |
| `tuning/configs/` | `tuning/defaults.py:19`, `tuning/parameter_registry.py`, `tuning/mode_router.py` | `MANIFEST.in:30` |
| `policy/egress.yaml` | `policy/egress.py:85` | `pyproject.toml` package-data for `spatialomicsgym.policy` |
| `data/` | `agent/empirical_leaderboard.py:84,94,97` (`parents[1] / "data"`); `tool/transcriptomics_skills.py:51`; `backend/sog_portal/services/datastore.py:983`; `agent/tools/spatial_library_worker.py:34` (assumes `agent/tools` and this package are siblings) | `MANIFEST.in:31` |
| `spatialomicsgym_env/` | `install/sog_install/base_env.py:170`; `run_core_tests.sh` (three levels up) | `MANIFEST.in:61` |
| `utils/child_site/` | `utils/execution.py:308` puts it on the child's `PYTHONPATH` | `MANIFEST.in:8` (`*.py`) |

No README may be placed in a directory a loader globs: `know_how/` and everything under it, `tool/tool_description/`,
`tool/protocols/`, `tool/schema_db/`, `tuning/configs/` and `data/`.

---

## `legacy/`

`legacy/` holds code from the Biomni era that no production code calls. It stays importable, under both its old and
its new names, so nothing breaks. Do not extend it: new benchmark work belongs in `agent/benchmarks`. Deleting it
is a separate, later decision.

| Module | Came from | Evidence that no production code calls it |
|---|---|---|
| `generate_function.py` | the package top level | A CLI over `function_generator.py`. Not a console script; only two tests run it with `-m` |
| `extract_biorxiv_tasks.py` | the package top level | Mines bioRxiv PDFs for tasks with `env_collection.PaperTaskExtractor`. Its only caller is `process_all_subjects.py`. Its default metadata CSV, `agent/data/biorxiv_metadata.csv`, exists nowhere in the repository |
| `process_all_subjects.py` | the package top level | No importer. It runs its sibling `extract_biorxiv_tasks.py` as a subprocess, so the two files move together |
| `env_collection.py` | `agent/` | `PaperTaskExtractor`. Used only by `extract_biorxiv_tasks.py` |
| `function_generator.py` | `agent/` | Used only by `generate_function.py` |
| `qa_llm.py` | `agent/` | No importer outside the tests |
| `eval/` (`spatialomicsgym_eval1.py`) | `eval/` | The Biomni Eval1 loader (a Hugging Face parquet). Tests only |
| `task/` (`base_task.py`, `hle.py`, `lab_bench.py`) | `task/` | HLE and LAB-Bench task loaders. `hle.py` has one test; `lab_bench.py` has no reference at all |
| `example_mcp_tools/pubmed_mcp.py` | `tool/example_mcp_tools/` | A Biomni sample MCP server with no reference anywhere. The folder has no `__init__.py`, as before the move |

Evidence: on 2026-10-07, `git grep` over `agent/`, `backend/` and `install/` (outside `legacy/` and `_aliases.py`)
found no importer of these modules, under either name. The remaining mentions are comments: the `pyproject.toml`
dependency note and the two `spatialomicsgym_env` recipe comments listed [below](#comments-that-name-an-old-path).

Two path fixes came with the move: `extract_biorxiv_tasks.py` and `process_all_subjects.py` climb two directories
instead of one, so `sys.path` still gains `agent/` and the CSV default still resolves to the same path.

---

## Old import names keep working

On 2026-10-07, 21 modules and packages of this package moved into the role folders. Every old dotted name still
imports, and gives the very same module object as the new name. `_aliases.py` keeps one table (`ALIASES`) and a
finder that sits first on `sys.meta_path`. For an old name, or any dotted name under it, the finder returns the module
already imported under the new name. So all of these behave exactly as before the move:

- `import old.name`, `from old.name import x`, and `from spatialomicsgym import llm`;
- `importlib.import_module` and `importlib.util.find_spec` (the spec's `origin` is the moved file);
- `mock.patch("old.name.attr")` and `monkeypatch.setattr("old.name.attr", ...)`, which patch the one real module;
- `-m` with an old name, which runs the moved file;
- module-level state, such as the `env_desc` dicts, which exists once.

Apart from the two import lines in `__init__.py`, no importer was rewritten: the portal, the installer, the
benchmark harness, the tests and the moved modules themselves still use the old names. No `know_how/` playbook names
a moved module.

| Old name | New name | File |
|---|---|---|
| `spatialomicsgym.llm` | `spatialomicsgym.providers.llm` | `providers/llm.py` |
| `spatialomicsgym.provider_names` | `spatialomicsgym.providers.provider_names` | `providers/provider_names.py` |
| `spatialomicsgym.provider_backoff` | `spatialomicsgym.providers.provider_backoff` | `providers/provider_backoff.py` |
| `spatialomicsgym.responses_stream` | `spatialomicsgym.providers.responses_stream` | `providers/responses_stream.py` |
| `spatialomicsgym.action` | `spatialomicsgym.turn.action` | `turn/action.py` |
| `spatialomicsgym.answer` | `spatialomicsgym.turn.answer` | `turn/answer.py` |
| `spatialomicsgym.stream_events` | `spatialomicsgym.contracts.stream_events` | `contracts/stream_events.py` |
| `spatialomicsgym.task_types` | `spatialomicsgym.contracts.task_types` | `contracts/task_types.py` |
| `spatialomicsgym.generated_report` | `spatialomicsgym.contracts.generated_report` | `contracts/generated_report.py` |
| `spatialomicsgym.env_desc` | `spatialomicsgym.software_catalog.env_desc` | `software_catalog/env_desc.py` |
| `spatialomicsgym.env_desc_cm` | `spatialomicsgym.software_catalog.env_desc_cm` | `software_catalog/env_desc_cm.py` |
| `spatialomicsgym.netbind` | `spatialomicsgym.policy.netbind` | `policy/netbind.py` |
| `spatialomicsgym.generate_function` | `spatialomicsgym.legacy.generate_function` | `legacy/generate_function.py` |
| `spatialomicsgym.extract_biorxiv_tasks` | `spatialomicsgym.legacy.extract_biorxiv_tasks` | `legacy/extract_biorxiv_tasks.py` |
| `spatialomicsgym.process_all_subjects` | `spatialomicsgym.legacy.process_all_subjects` | `legacy/process_all_subjects.py` |
| `spatialomicsgym.agent.env_collection` | `spatialomicsgym.legacy.env_collection` | `legacy/env_collection.py` |
| `spatialomicsgym.agent.function_generator` | `spatialomicsgym.legacy.function_generator` | `legacy/function_generator.py` |
| `spatialomicsgym.agent.qa_llm` | `spatialomicsgym.legacy.qa_llm` | `legacy/qa_llm.py` |
| `spatialomicsgym.eval` (and everything under it) | `spatialomicsgym.legacy.eval` | `legacy/eval/` |
| `spatialomicsgym.task` (and everything under it) | `spatialomicsgym.legacy.task` | `legacy/task/` |
| `spatialomicsgym.tool.example_mcp_tools` (and everything under it) | `spatialomicsgym.legacy.example_mcp_tools` | `legacy/example_mcp_tools/` |
| `spatialomicsgym.webui` (and everything under it) | `sog_portal` | `backend/sog_portal/` |
| `spatialomicsgym.setup` (and everything under it) | `sog_install` | `install/sog_install/` |

The last two entries come from the earlier re-layout and are removed in a later release. The other entries stay until
someone deliberately rewrites their importers.

A key matches its own name and the names under it, never a name that only starts with the same letters:
`spatialomicsgym.task` does not capture `spatialomicsgym.task_types`. When two keys match, the longer one wins.

**What an alias does not cover.**

- Loading by file path (`spec_from_file_location`, `runpy.run_path`). No such load targets a moved file.
- `__file__` arithmetic inside a moved file. Only the two biorxiv scripts had any, and they were fixed in the move.
- The module's own names. `__name__`, `__module__` and `__file__` report the new location, so tracebacks and reprs
  show, for example, `spatialomicsgym.providers.responses_stream.StreamedReplyTimeout`.
- `sys.modules` keys. `import spatialomicsgym` registers `spatialomicsgym.turn.answer`; the key
  `spatialomicsgym.answer` appears only after the first import under the old name. Removing only an old key from
  `sys.modules` no longer forces a fresh module on the next import.
- Editors. Go-to-definition (pyright, jedi) cannot follow a meta-path alias. Use the table above.

**No file and no directory may remain at an old path.** A leftover `eval/` holding only `__pycache__/` would become a
namespace package that only the finder's priority hides.

**Before anyone rewrites importers to the new names**, three things need care:

- Code outside `agent/` that imports one of the three modules moved out of `agent/` keeps the old name. Importing
  the old name runs `agent/__init__.py`, which loads the `.env` before `default_config` takes its snapshot; the new
  name skips that step.
- Tests that block a module with `sys.modules["spatialomicsgym.llm"] = None` block only that spelling. They must stub
  both names first.
- `test/test_the_stream_vocabulary_has_one_source.py` checks that consumers contain the literal text
  `from spatialomicsgym.stream_events import`.

### Comments that name an old path

Recorded on 2026-10-07. These comments in files that did not move name a moved file by its old path. They were left
unedited on purpose: the dotted names in them still resolve, and editing the frontend file would force a rebuild of
the served bundle. Fix them when the importers are rewritten, and the frontend one with the next real frontend build.

- `chat_cli.py:91` (`spatialomicsgym/llm.py`) and `chat_cli.py:1168` (`spatialomicsgym/answer.py`)
- `install/sog_install/_agent_probe.py:58` (`spatialomicsgym/answer.py`)
- `install/sog_install/credentials.py:5` (`spatialomicsgym/llm.py`)
- `frontend/src/lib/streamEvents.ts:4` (`spatialomicsgym/stream_events.py`)
- `spatialomicsgym_env/spatialomicsgym_env.yml:17` and `spatialomicsgym_env/spatialomicsgym_env_general.yml:25`
  (`agent/env_collection.py`)

---

## Rules for changing this package

- **Never add a file to `know_how/`.** `KNOW_HOW_HASH` and the prompt both come from a filesystem glob of
  `know_how/*.md`, so even an untracked file there changes the pin and enters every prompt.
- **Any `.py` added, moved or renamed here changes `AGENT_SRC_HASH`.** The pin folds each file's repository-relative
  path together with its content (`source_pin.py`), so a pure move changes it too.
- **A role folder's `__init__.py` holds a docstring only.** An import there would pull heavy modules into the
  installer's stdlib-only probe or into the bare-`python3` redactor.
- **The model-facing names stay.** `spatialomicsgym.tool.<module>` names and the `run_post_analysis` entry point of
  `spatialomicsgym.postanalysis` are in the system prompt. Renaming them changes every prompt, and with it the code the
  model writes, even if an alias keeps the old import working.
- **A module that moves gets an `ALIASES` entry** in `_aliases.py`, keeps its file name, and leaves nothing behind at
  its old path.
- **Keep this map current.** A new top-level module or subpackage gets a line in the package map and a row in the
  matching table above.

The rules for `agent/` as a whole (`tools/` stays flat, the instance-root marker, the fixed import names) are in
[`agent/README.md`](../README.md#what-must-not-move).

## See also

- [`agent/README.md`](../README.md): the parts of `agent/` and how runtime finds each one.
- [`agent/tools/CATALOG.md`](../tools/CATALOG.md): every MCP portal by analysis category.
- [`agent/skills/README.md`](../skills/README.md): the skill domains used for tool routing.
- [`spatialomicsgym_env/README.md`](spatialomicsgym_env/README.md): the conda environments and how to build them.
