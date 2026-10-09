# Terminal chat

`stcoscientist` (alias: `sog-chat`) is the terminal front door to ST-Coscientist. It runs an interactive session,
or answers one question and exits.

```console
$ stcoscientist                                                   # interactive session; /help lists commands
$ stcoscientist --mcp                                             # interactive, with the analysis tools wired
$ stcoscientist "Which tool detects spatial domains in Visium data?"   # one-shot
$ stcoscientist -q "Summarize ./data/visium.h5ad" > answer.txt    # one-shot, answer only
$ stcoscientist --json "..."                                      # one-shot, {"answer": "..."} for scripts
$ sog-setup chat                                                  # the chat in the env sog-setup built, tools wired
```

Questions can also be piped on stdin. Run `stcoscientist --help` for the full option list of the installed version.

## Options

| Option | Effect |
|---|---|
| `question` | A question to ask; omit it for an interactive session. |
| `-m NAME`, `--model NAME` | The LLM model. Default: `$SOG_LLM`, else the built-in default. |
| `--source PROVIDER` | Force the provider: `Anthropic`, `OpenAI`, `AzureOpenAI`, `Gemini`, `Groq`, `Bedrock`, `Ollama`, `Custom`. Normally inferred from `SOG_SOURCE` and the model name. |
| `-p DIR`, `--path DIR` | Data directory for the agent. Default: `$SOG_PATH`, else `$SOG_DATA_PATH`, else `./data`. |
| `--base-url URL` | Base URL of a custom or self-hosted OpenAI-compatible endpoint. |
| `--api-key KEY` | API key for a custom endpoint (the named providers read their own environment variable instead). |
| `--temperature T` | Sampling temperature. |
| `--timeout SEC` | Per-step code-execution timeout in seconds (default `SOG_TIMEOUT_SECONDS`, 600). |
| `--commercial` | Commercial mode: exclude non-commercial datasets. |
| `--no-tool-retriever` | Skip the LLM pre-selection of tools and datasets (faster start, less focused prompt). |
| `--mcp [CONFIG]` | Wire the MCP analysis tools. Without a path: `$SOG_MCP_CONFIG`, else `install/recipes/mcp_config.setup.yaml`, else `agent/MCP_server/mcp_config.yaml`. |
| `--env-file FILE` | An extra dotenv file, read after the install's own `.env` (default `./.env`). |
| `-q`, `--quiet` | One-shot: print only the final answer, no streamed steps. |
| `--json` | One-shot: print `{"answer": "..."}` (implies `--quiet`). |
| `--list-models` | List common model names per provider and exit. |
| `--no-banner` | Suppress the session banner. |
| `-V`, `--version` | Print the version and the install location. |

```{important}
`--mcp` is what connects the analysis tools; without it the agent can answer questions but run no method. Inside a
session, `/mcp` wires them later. Only servers whose conda environments `sog-setup` built (and found healthy) are
enabled in the config it records.
```

## Session commands

In an interactive session, a line that starts with `/` is a command; anything else goes to the agent.

| Command | Effect |
|---|---|
| `/help` | Show the command list. |
| `/model [NAME]` | Show the active model, or switch to `NAME` (the provider follows the name: `claude-*`, `gpt-*`, ...). |
| `/source` | Show the active provider. |
| `/config` | Show the full active configuration. |
| `/tools` | List the wired analysis tools. |
| `/mcp [CONFIG]` | Wire the MCP analysis tools now, optionally from a given config. |
| `/history` | Show this session's questions and answers. |
| `/save [FILE]` | Save the conversation as a PDF (default `stcoscientist_conversation.pdf`). |
| `/export [FILE]` | Save a Markdown transcript (default `stcoscientist_transcript.md`). |
| `/retry` | Ask the last question again. |
| `/reset`, `/clear` | Start a fresh conversation (clears context and history). |
| `/version` | Show the version. |
| `/exit`, `/quit` | Leave. |

End a line with a backslash to continue it on the next line. Up/Down recall earlier prompts and Tab completes
`/commands`. The session remembers earlier turns, so a follow-up such as "redo it with 12 domains" does not need to
restate the input path.

## Writing a good request

The agent plans from what you say, so say what matters:

- **Name the data.** A path to an `.h5ad` (or the files a method takes) and, for deconvolution, the single-cell
  reference and the `obs` column that holds its cell types.
- **Name the task, not the method, unless you want a specific one.** "Find spatial domains" lets the agent choose; "run
  STAGATE" pins it. Both work.
- **Give the numbers you know.** The expected number of domains, a resolution, a marker list.
- **Say where output goes** if you care: otherwise results land under the data directory (see
  [Results and reports](results.md)).

## Exit status and automation

One-shot mode is script-friendly: `-q` prints only the answer, `--json` wraps it in one JSON object, and a question
can be piped on stdin. Errors under `--json` come back as JSON too (`{"error": "..."}`), so a caller can always parse
stdout.

| Exit status | Meaning |
|---|---|
| `0` | The turn finished and the answer is complete. |
| `1` | The agent raised an error (printed to stderr, or returned as `{"error": ...}` under `--json`). |
| `2` | A bad numeric option (`--timeout`, `--temperature`). |
| `3` | The provider pre-flight failed: no key, an unknown provider, an unreachable endpoint. |
| `4` | *Degraded*: the turn stopped early (a timeout, a model that gave up) and the answer may be partial. The note saying why goes to stderr, or into a `"degraded"` field under `--json`. |
| `130` | Interrupted with Ctrl-C. |
