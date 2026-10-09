# Verify the install

Three checks, from the whole chain down to a single environment.

## End-to-end wiring

```bash
sog-setup conncheck
```

Verifies that the base environment, the MCP servers and ST-Coscientist are wired together: the agent can start every
enabled portal, each portal finds its worker and its interpreter, and the functions the config publishes are the ones
the agent sees.

## Per-tool health

```bash
sog-setup doctor
```

A read-only health report for an existing base environment and its tool environments: which exist, which import
their method, which are missing or broken. Use it after an upgrade, after moving an install, or whenever a tool
fails with an environment error. Rebuild one tool with `sog-setup --only <key>`.

## The agent itself

```bash
stcoscientist "Which tool detects spatial domains in Visium data?"
stcoscientist --version
```

The first needs only a working provider key. `--version` prints the package version and where it is installed from.

```{important}
The tool tests inside `sog-setup` use small datasets from a local `test/` tree that is not part of this repository.
Without it, `sog-setup` falls back to an import check for each tool and says so. That is enough to prove an
environment is built; the real test of a tool is the first analysis you run with it.
```

## Common problems

**The agent says it has no tools.** Analysis tools are connected only with `--mcp` (terminal) or `add_mcp()`
(Python). A bare `--mcp` uses `$SOG_MCP_CONFIG`, which `sog-setup` records in `.env` when it finishes; if the
variable is unset, pass the config path explicitly (`--mcp install/recipes/mcp_config.setup.yaml`).

**A tool environment failed to build.** `sog-setup` is resumable; re-run it to retry, or `sog-setup --only <key>` to
rebuild one server. `sog-setup reset` removes a run's tool environments and test artifacts (keeping the logs) for a
clean start. Logs and progress live in `.sog_setup/` (clone) or `$SOG_HOME/.sog_setup/` (wheel install).

**The key does not validate.** `sog-setup` pings the provider with the key you give it. Check `SOG_SOURCE` and
`SOG_LLM` in `.env` against the [provider table](../configuration.md#providers); an Azure deployment, for example,
is named through `SOG_LLM`, not through the endpoint URL.

**No conda on PATH.** The chat works without conda, but `sog-setup` needs conda, mamba or micromamba to build tool
environments and exits with status 3 when it finds none.
