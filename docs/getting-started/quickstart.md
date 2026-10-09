# Quick start

This page assumes the [installation](installation.md) is done: the `spatialomicsgym_env` environment is active and
`sog-setup` has configured your provider (and, for the analysis below, built at least one tool environment).

## A first question

Ask something that needs no data and no tool:

```bash
stcoscientist "Which tool detects spatial domains in Visium data?"
```

The agent answers from its tool catalogue and know-how, then exits. Without arguments, `stcoscientist` opens an
interactive session; type `/help` there to list the commands. See [Terminal chat](../usage/terminal-chat.md).

## A first analysis from the terminal

Analysis tools are only connected when you ask for them, with `--mcp`:

```bash
stcoscientist --mcp
```

Then describe the analysis and name the data:

```text
> Identify spatial domains in ./data/visium.h5ad and summarize each domain's marker genes.
```

ST-Coscientist picks a method (say GraphST or STAGATE), runs it in that method's own conda environment, inspects the
result, draws figures and writes a report. Each turn prints the steps as they run; the final answer comes last.
Every analysis writes its conversation log, h5ad/csv results, figures and an HTML report to disk; see
[Results and reports](../usage/results.md).

```{note}
`stcoscientist --mcp` wires the config `sog-setup` recorded in `SOG_MCP_CONFIG`; it only exposes servers whose
environments were built. `sog-setup chat` starts the chat in the environment `sog-setup` built, with the tools
wired, and is the simplest way to get the same thing.
```

## A first analysis from Python

```python
from spatialomicsgym import clean_answer
from spatialomicsgym.agent import STCoscientist

agent = STCoscientist(path="./data")   # where datasets are read and results written
agent.add_mcp()                        # wire the analysis tools

log, answer = agent.go("Identify spatial domains in my Visium slide and summarize each domain's marker genes.")
print(clean_answer(answer))
agent.save_conversation_history("analysis_report.pdf")
```

`go()` returns the step log and the raw final message; `clean_answer()` strips the agent's internal scaffolding from
it. `add_mcp()` is what connects the analysis tools; without it the agent has none. The
[Python API](../usage/python-api.md) page covers the constructor, streaming, conversations and custom tools.

## Where to look next

- [Analysis tools](../tools/index.md): every method the agent can call, by category, with its parameters.
- [Configuration](../configuration.md): the `.env` reference (providers, data path, timeouts, feature switches).
- [Tutorials](../tutorials/index.md): worked examples.
- [Verify the install](verify.md): if something did not work.
