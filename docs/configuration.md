# Configuration

Settings are read from the environment and from `.env`. The template is
[`.env.example`](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/.env.example) at the repository root;
`sog-setup` writes `.env` for you, validating the key with a live ping, and `cp .env.example .env` is the manual
route. Only three lines are required: `SOG_SOURCE`, `SOG_LLM` and the key block of the one provider you use. Never
commit a real `.env`.

## Precedence

1. An explicit argument wins: a `STCoscientist(...)` constructor argument, or a `stcoscientist` command-line flag.
2. Then a variable exported in the shell.
3. Then the install's own `.env` (the clone root, or `$SOG_HOME` for a wheel install), loaded with `override=False`.
4. Then an extra dotenv file: `stcoscientist --env-file FILE` (default `./.env`), which fills only what is still
   unset and is named on stderr whenever it is read.

`SOG_SKIP_DOTENV=1` skips steps 3 and 4 (an `--env-file` named explicitly is still read). The settings are
snapshotted once, when `spatialomicsgym.config` is imported; set variables before importing the agent from Python.
Legacy `BIOMNI_*` names are mirrored onto their `SOG_*` equivalents, the new name winning.

## Providers

`SOG_SOURCE` selects the provider and `SOG_LLM` names the model (or, on Azure, the deployment). The provider is
normally inferred from the model name (`claude-*`, `gpt-*`, `azure-*`, `gemini-*`, ...); `SOG_SOURCE` and
`--source` settle the ambiguous cases.

| `SOG_SOURCE` | Key block | Notes |
|---|---|---|
| `Anthropic` | `ANTHROPIC_API_KEY` | Keys at <https://console.anthropic.com/settings/keys>. |
| `OpenAI` | `OPENAI_API_KEY` | `gpt-5*` models are routed through the Responses API automatically. |
| `AzureOpenAI` | `OPENAI_API_KEY`, `OPENAI_ENDPOINT` (`https://<resource>.openai.azure.com`); optional `OPENAI_API_VERSION`, `OPENAI_USE_RESPONSES_API` | `SOG_LLM` is `azure-<deployment>`: the `azure-` prefix selects the provider, the rest is your deployment id. The deployment never comes from the endpoint URL. |
| `Gemini` | `GEMINI_API_KEY` | |
| `Groq` | `GROQ_API_KEY` | |
| `Bedrock` | `AWS_REGION`, then either `AWS_BEARER_TOKEN_BEDROCK` or `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | `SOG_LLM` is the Bedrock model id, e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0`. |
| `Custom` | `SOG_CUSTOM_BASE_URL` (e.g. `http://localhost:8000/v1`), `SOG_CUSTOM_API_KEY` (omit if the server needs none) | Any OpenAI-compatible server: vLLM, SGLang, TGI, ... |
| `Ollama` | none | A local daemon on `:11434`; `ollama pull <model>` first. |

`stcoscientist --list-models` prints common model names per provider.

## Core settings

| Variable | Default | Meaning |
|---|---|---|
| `SOG_SOURCE` | | The provider (table above). |
| `SOG_LLM` (alias `SOG_LLM_MODEL`) | `azure-gpt-6-astra` | Model or deployment name. |
| `SOG_DATA_PATH` | `./data` | The agent's data directory: where datasets are read and `outputs/` is written. `sog-setup` and the settings panel write this one. |
| `SOG_PATH` | | The same setting under its older name; it outranks `SOG_DATA_PATH` when the two disagree (a warning says so). |
| `SOG_TIMEOUT_SECONDS` | `600` | Maximum seconds per code-execution step. |
| `SOG_TEMPERATURE` | `0.7` | Sampling temperature. |
| `SOG_USE_TOOL_RETRIEVER` | `true` | Pre-select the tools, datasets and know-how relevant to each task with one LLM call before the loop starts. |
| `SOG_COMMERCIAL_MODE` | `false` | Exclude datasets that are non-commercial or need a commercial license. |
| `SOG_MCP_CONFIG` | set by `sog-setup` | The MCP config a bare `--mcp` / `add_mcp()` uses. |
| `SOG_LLM_REQUEST_TIMEOUT` | `600` | Seconds per LLM request. |
| `SOG_LLM_MAX_RETRIES` | `2` | Retries per LLM request. |
| `SOG_LLM_TRANSIENT_WAIT_SECONDS` | `300` | A call that failed on a rate limit, an overload or a 5xx is retried only if the retry starts within this many seconds of the first call (0–3600). |
| `SOG_MCP_HANDSHAKE_TIMEOUT` | `30` | Seconds to wait for an MCP server to start and answer the handshake. |
| `SOG_PRE_ANSWER_VERIFICATION` | `false` | Experimental: run the post-analysis verdict before a final answer is accepted. |
| `SOG_RETRIEVAL_QUERY_LAST` | `false` | Experimental: place the retrieved resources in a cacheable system prefix, with the query last. |

## Workflow switches

| Variable | Default | Meaning |
|---|---|---|
| `SOG_POST_ANALYSIS_ENABLED` | `true` | After a tool runs: scan the result, draw figures, review, write `manifest.json` and `report.html`. |
| `SOG_POST_ANALYSIS_MAX_FOLLOWUP_ROUNDS` | `1` | Follow-on analyses the agent may launch off its own review of a result; 0 keeps the review but not the acting; hard ceiling 3. |
| `SOG_ENV_FALLBACK_ENABLED` | `true` | When a tool's own environment fails to import on this machine, let the agent redo the step in the general-purpose environment (`spatialomicsgym_env_general`); the answer discloses the substitute. Forced off under benchmarking. |
| `SOG_UNSOLVED_RESCUE_ENABLED` | `true` | A turn that ends on a give-up gets one more bounded ReAct round. Forced off under benchmarking. |
| `SOG_GENERAL_PYTHON` | auto | The interpreter of the general-purpose environment, when it lives somewhere the installer would not look. |
| `SOG_GENERAL_ENV_MAX_CALLS` | `12` | Per-turn cap on general-environment calls (0–200). |
| `SOG_REPL_ISOLATION` | `inprocess` | Where model-written Python runs: `inprocess` keeps the REPL in this process; `process` runs every cell in a separate worker. Ignored under benchmarking. |
| `SOG_RESEARCH_MAX_ROUNDS` | `4` | Rounds of a research run (a bounded investigation the user starts deliberately). |
| `SOG_RESEARCH_MAX_SECONDS` | `3600` | Wall-clock budget of a research run. Refused while benchmarking is on. |
| `SOG_KNOW_HOW_ENROLMENT` | `all` | How much of the know-how corpus sits in the system prompt before retrieval: `all`, `summaries` or `demand`. Forced to `all` under benchmarking. |
| `SOG_KNOW_HOW_PACKS` | `false` | Also load the merged external skill packs (tier-2 know-how). Never loaded under benchmarking. |
| `SOG_KNOW_HOW_PACK_BUDGET` | `3` | The most pack documents one turn's second retrieval pass may add (0–10; 0 disables that pass). |
| `SOG_TUNING_ENABLED` | `false` | Hyperparameter tuning (needs the `tuning` extra, Optuna). |
| `SOG_TUNING_MODE` | | `benchmark_tuning.light`, `benchmark_tuning.full` or `adaptive_tuning`. |
| `SOG_TUNING_STRATEGY` | | Force `grid`, `random`, `staged` or `bayesian`. |
| `SOG_BENCHMARKING_ENABLED` | `false` | Scored runs: output inspection becomes mandatory before any evaluation, and the self-repair paths above are forced off. |
| `SOG_EVALUATION_ENABLED` | `false` | Allow metric computation after inspection. |
| `SOG_BENCHMARK_USER_TOOLS` | `false` | Include user-created MCP tools in benchmark runs. |

## Tool creation

With tool creation on, the agent can build a new MCP tool from a method's GitHub repository, in its own conda
environment, and wire it in. The tools it creates live in `agent/tools_user/` and are listed in
`agent/MCP_server/mcp_config_user.yaml`.

| Variable | Default | Meaning |
|---|---|---|
| `SOG_TOOL_CREATION_ENABLED` | `false` | Let the agent build new tools. |
| `SOG_MAX_USER_ENVS` | `20` | Maximum conda environments for user-created tools. |
| `SOG_SELF_REVIEW_ENABLED` | `false` | Auto-diagnose and repair failed builds and tests. |
| `SOG_SELF_REVIEW_TOTAL_CAP` | `5` | Maximum self-review rounds per session. |
| `SOG_SELF_REVIEW_TOTAL_BUDGET_SEC` | `1800` | Time budget for self-review. |
| `SOG_MEMORY_ENABLED` | `false` | Advisory memory of prior tool-creation attempts. |
| `SOG_MEMORY_PATH` | `agent/tools_user/.memory` | Where that memory lives. |
| `SOG_MEMORY_MAX_ATTEMPTS` | `10` | Attempts remembered per tool. |
| `SOG_MEMORY_SHORTCUT_ENABLED` | `false` | Let memory skip phases, rather than only hint. |

## Per-tool credentials

Only needed for the specific tool that uses them.

| Variable | Used by |
|---|---|
| `PROTOCOLS_IO_ACCESS_TOKEN` | Full protocols.io method text. |
| `SYNAPSE_AUTH_TOKEN` | Controlled Synapse (Sage Bionetworks) downloads. |
| `UCD_TOKEN` | UCDeconvolve cloud deconvolution (the `ucdeconvolve` server). |
| `NCBI_EMAIL` | Higher-rate NCBI Entrez queries (identifies you; not secret). |
| `DREMIO_PAT`, `DREMIO_SCHEME`, `DREMIO_ALLOW_PLAINTEXT_TOKEN` | JGI Lakehouse (`query_jgi_lakehouse`). `http` only for a Dremio without TLS, and then `DREMIO_ALLOW_PLAINTEXT_TOKEN=1` accepts sending the token in cleartext. |
| `SOG_CHATNT_REVISION` | The commit at which `query_chatnt` runs ChatNT's remote code. |

Credentials cross into a worker process only when named in `SOG_AGENT_ENV_ALLOW` (below).

## Literature fetching

| Variable | Default | Meaning |
|---|---|---|
| `SOG_LITERATURE_FETCH_TIMEOUT` | `30` | Seconds per socket read (DOI pages, supplementary files). |
| `SOG_LITERATURE_FETCH_DEADLINE` | `600` | Seconds for one whole fetch, however slowly bytes arrive. |
| `SOG_LITERATURE_MAX_PAGE_MB` | `64` | Largest page read into memory. |
| `SOG_LITERATURE_MAX_DOWNLOAD_MB` | `1024` | Largest supplementary file written to disk. |
| `SOG_SCHOLAR_PROXY_BUDGET_SECONDS` | `90` | How long `query_scholar` waits for a working free proxy. |

## Workers and limits

| Variable | Default | Meaning |
|---|---|---|
| `SOG_WORK_DIR` | a writable `/workspace/work`, else `./work` | Where an MCP tool writes when its call omitted `output_dir`. |
| `SOG_AGENT_ENV_ALLOW` | | Comma-separated variables the worker environment may carry (it is an allowlist; a cell can print its environment). |
| `SOG_AGENT_THREADS` | `min(8, cores)` | Thread budget for every OpenMP/BLAS/numba pool in the worker and the tools it launches. |
| `SOG_AGENT_NPROC` | `max(2048, 256 × threads)` | Process (thread) rlimit for the worker's uid. |
| `SOG_AGENT_MEMORY_BYTES` | none | Address-space cap for the worker, in bytes. |
| `SOG_EGRESS_POLICY` | | A YAML of additions to the shipped egress allowlist; read only when owned by root and not group- or world-writable. |
| `SOG_REPL_MAX_FIGURE_BYTES` | `16777216` | The most figure bytes one cell carries into the transcript (saved files are untouched). |
| `SOG_RSCRIPT` | | The `Rscript` a `#!R` cell runs with when the agent env has none on PATH (the R tool methods run in their own envs regardless). |

## Runtime state

| Path | Contents |
|---|---|
| `.sog_setup/` | Wizard logs, setup progress, saved keys (`llm_keys.json`, mode 0600), backups, transcripts. At the repository root in a clone; under `$SOG_HOME` for a wheel install. `SOG_SETUP_STATE_DIR` overrides the location. |
| `$SOG_HOME` (default `~/.spatialomicsgym`) | The instance root of a wheel install without a clone, seeded by `sog-setup` on first run and laid out like a checkout. |
| `install/recipes/mcp_config.setup.yaml` | The install-aware MCP config `sog-setup` writes: every server, enabled only when its environment exists and is healthy, with this machine's paths. `SOG_MCP_CONFIG` points here. |
| `agent/MCP_server/mcp_config.yaml` | The canonical config, which `sog-setup` rewrites for this machine (after a timestamped backup) unless `--keep-agent-config`. |
| `agent/MCP_server/mcp_config_user.yaml` | The machine-local overlay listing tools the agent created. |
| `agent/tools_user/` | The user-created tools and their state (`.knowledge/`, `.memory/`, `.trash/`). Never copy `.memory/.secret` between machines; it is a signing key. |

## Security

The agent executes code that the LLM writes, as the user who started it; this repository ships no web portal and so
no privilege boundary. Run it on a machine or container you are prepared to give it, and keep credentials and
sensitive data out of its reach. Outbound HTTP from the in-process tools goes through one client that enforces the
egress policy in `agent/spatialomicsgym/policy/egress.yaml` (host allowlist, redirect checks, response caps); what that
policy does and does not cover is set out in
[SECURITY.md](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/SECURITY.md).
