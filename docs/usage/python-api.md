# Python API

The agent is the class {py:class}`~spatialomicsgym.agent.stcoscientist.STCoscientist` (imported as `spatialomicsgym.agent.STCoscientist`). The terminal chat is a thin front door over
it; everything the chat does, a script or a notebook can do.

```python
from spatialomicsgym import clean_answer
from spatialomicsgym.agent import STCoscientist

agent = STCoscientist(path="./data")   # where datasets are read and written
agent.add_mcp()                        # wire the analysis tools

log, answer = agent.go("Identify spatial domains in my Visium slide and summarize each domain's marker genes.")
print(clean_answer(answer))
agent.save_conversation_history("analysis_report.pdf")
```

```{important}
`add_mcp()` is what connects the analysis tools; without it the agent has none and will say so. A bare `add_mcp()`
uses the config `sog-setup` recorded in `SOG_MCP_CONFIG`, then `install/recipes/mcp_config.setup.yaml`, then the
canonical `agent/MCP_server/mcp_config.yaml`; pass a path to use another.
```

## Settings and the `.env`

Importing `spatialomicsgym.agent` loads the install's `.env` (unless `SOG_SKIP_DOTENV` is set) and snapshots the
environment into `spatialomicsgym.config.default_config`. Every constructor argument left as `None` takes its value
from there, so a script usually needs no arguments beyond `path`. Set variables **before** the import if you set them
from Python:

```python
import os
os.environ["SOG_SOURCE"] = "Anthropic"
os.environ["SOG_LLM"] = "claude-opus-4-8"

from spatialomicsgym.agent import STCoscientist
```

The full list of variables is on the [configuration](../configuration.md) page.

## Constructor

```python
STCoscientist(
    path=None,                  # data directory (SOG_PATH / SOG_DATA_PATH, default ./data)
    llm=None,                   # model name (SOG_LLM)
    source=None,                # provider; normally inferred from the model name and SOG_SOURCE
    use_tool_retriever=None,    # LLM pre-selection of tools and datasets per task (default True)
    timeout_seconds=None,       # per code-execution step (SOG_TIMEOUT_SECONDS, default 600)
    base_url=None,              # custom OpenAI-compatible endpoint (SOG_CUSTOM_BASE_URL)
    api_key=None,               # key for that endpoint only (SOG_CUSTOM_API_KEY)
    commercial_mode=None,       # exclude non-commercial datasets (SOG_COMMERCIAL_MODE)
    expected_data_lake_files=None,
    conversation_memory=False,  # remember earlier turns of this session (see below)
)
```

`source` is deliberately not pre-filled from the environment: an explicit `llm="gpt-4o"` on an Anthropic `.env` still
builds an OpenAI client, because the provider is resolved from the model name unless you force it.

## Running a task

{py:meth}`~spatialomicsgym.agent.stcoscientist.STCoscientist.go` runs one task to completion and returns `(log, answer)`:
the step log (a list of the messages and tool calls of the ReAct loop) and the raw final message. The raw message
carries the agent's own scaffolding (`<solution>` tags, a routing line); {py:func}`~spatialomicsgym.turn.answer.clean_answer` (exported as `spatialomicsgym.clean_answer`)
strips it for display.

{py:meth}`~spatialomicsgym.agent.stcoscientist.STCoscientist.go_stream` is the same loop as a generator, yielding one `dict` per
step as it happens, for a progress display or a web front end:

```python
for step in agent.go_stream("Deconvolve ./data/visium.h5ad against ./data/reference.h5ad (cell types in obs['cell_type'])."):
    ...  # each step holds the current message and state
```

Two attributes describe how the last turn ended: `agent.last_turn_degraded` is `None` when it finished, or a note
saying why it stopped early; `agent.last_turn_rescued` is set when a first attempt gave up and a rescue round then ran.

## Conversations

By default each `go()` is independent. With `conversation_memory=True`, every turn is told what the earlier turns of
the session asked and answered, so a follow-up ("redo it with 12 domains") need not restate the input path. The
terminal chat turns this on; benchmarks must leave it off.

Both methods take a `thread_id`. A process serving several people passes one value per person, which separates both
the checkpointed message history and the session recap. {py:meth}`~spatialomicsgym.agent.stcoscientist.STCoscientist.forget_conversation`
clears a thread; {py:meth}`~spatialomicsgym.agent.stcoscientist.STCoscientist.restore_conversation` loads earlier turns into one.

## Saving the conversation

{py:meth}`~spatialomicsgym.agent.stcoscientist.STCoscientist.save_conversation_history` writes the whole conversation, with the
captured figures, as a PDF:

```python
agent.save_conversation_history("analysis_report.pdf")          # .pdf is added if missing
agent.save_conversation_history("report", include_images=False)
```

The analysis results themselves (h5ad/csv files, figures, the HTML report) are written to disk by the tools as they
run, not by this call; see [Results and reports](results.md).

## Wiring tools

| Method | What it does |
|---|---|
| `add_mcp(config_path=None, *, merge_user=None)` | Wire the MCP analysis tools from a config. `merge_user` overlays the tools the agent created at run time (default: follows `SOG_TOOL_CREATION_ENABLED`). A config that fails to load is reported, never raised. |
| `reload_user_tools(config_path=None)` | Re-wire after a user tool was created, modified or deleted. |
| `create_mcp_server(tool_modules=None)` | Expose in-process tool modules as an MCP server, for other clients. |
| `configure(self_critic=None, test_time_scale_round=None)` | Turn the self-critic pass and test-time scaling on or off. |

## Custom tools, data and software

The agent's prompt lists the tools, datasets and software it may use. A session can extend each list:

```python
agent.add_tool(my_function)                 # a Python callable, registered under its own name and made retrievable
agent.add_data({"my_atlas.h5ad": "A reference atlas of mouse cortex, cell types in obs['cell_type']"})
agent.add_software({"my_package": "A package for X, installed in the agent env"})
```

`list_custom_tools()`, `get_custom_tool(name)` and `remove_custom_tool(name)` manage the tools; the same trio exists
for data (`*_custom_data`) and software (`*_custom_software`). Additions are per instance: two agents in one process
do not see each other's lists.

With `SOG_TOOL_CREATION_ENABLED=true`, the agent can also build a new MCP tool from a method's GitHub repository on its
own, in a fresh conda environment, and wire it in; see the
[contribution guide](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/CONTRIBUTION.md) for making such a
tool part of the platform.

## Reference

The full signatures are in the API reference: {py:mod}`spatialomicsgym.agent.stcoscientist`,
{py:mod}`spatialomicsgym.config`, {py:mod}`spatialomicsgym.providers.llm` (`get_llm`, the provider layer) and
{py:mod}`spatialomicsgym.chat_cli`.
