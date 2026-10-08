# agent/MCP_server/: the MCP server configuration

**What this is.** Configuration only, despite the name: `mcp_config.yaml` lists every shipped MCP server, the interpreter and script that start it, and the functions it publishes; `mcp_config_user.yaml` adds the tools created on this machine. The server code lives in `agent/tools/` and `agent/tools_user/`.
**How runtime finds it.** By path: relative to the agent package, through the `SOG_MCP_CONFIG` pointer, or through the `--mcp` argument the portal is started with. Together with `agent/tools/` the canonical file marks a directory as the instance root.
**What must not change.** The directory name, its place beside `tools/` and `tools_user/`, and the file name `mcp_config.yaml`.

## Files

| File | In git | Written by | Read by |
|---|---|---|---|
| `mcp_config.yaml` | tracked; a wheel carries a read-only copy in its `_platform/` payload | a contributor adding or changing a portal (`sog-setup capture` then records the matching `install/recipes/tool_specs/<key>.yaml` from it); the finalize phase of `sog-setup` rewrites it for this machine (interpreters and absolute script paths) after saving a timestamped backup under `.sog_setup/backups/` (`install/sog_install/mcp_resolver.py:624-641`), unless `sog-setup` runs with `--keep-agent-config` | every front door, `sog-setup conncheck`, the benchmark runners, the instance-root check |
| `mcp_config_user.yaml` | not tracked (ignored), as are its `.bak_*` copies | tool creation (`agent/tools_user/knowledge_manager.py:79-96`, `agent/tools_user/trash_manager.py:63`), the broker's `register_tool` (`agent/spatialomicsgym/agent/broker.py:1002`), and `sog-setup`'s `rebase_user_config` (`install/sog_install/mcp_resolver.py:710`) | the merge in `agent/spatialomicsgym/agent/mcp_config_merger.py` |

Nothing else belongs here apart from this README. A `__pycache__/` in this directory is stale and safe to delete.

Each server block in `mcp_config.yaml` carries `command` (`python` and the absolute path of its
`agent/tools/<key>_mcp_server.py`), `env` (`<KEY>_PYTHON` and `<KEY>_WORKER`), `enabled`, `description` and its
`tools`. The paths are absolute because each worker runs in its own conda env. When a path is stale (a clone at
another path, a renamed checkout), the portal heals it by basename (`agent/tools/base_mcp.py:136-186`), and so do
the agent (`agent/spatialomicsgym/agent/mcp_integration.py:555-585`) and the installer
(`install/sog_install/wiring.py:114-123`); see [tools/README.md](../tools/README.md).

## Which config is used

Two questions, two orders.

**Which config to serve** (the CLI, the portal, `sog-setup conncheck`): `chat_cli._resolve_mcp_config`
(`agent/spatialomicsgym/chat_cli.py:377-460`) takes an explicit `--mcp` path first; its default search is the
`SOG_MCP_CONFIG` pointer that `sog-setup` records in `.env`, then the generated `install/recipes/mcp_config.setup.yaml`,
then the canonical `agent/MCP_server/mcp_config.yaml`. The portal shares this order. `restart_portal.sh` passes
the canonical file explicitly with `--mcp`.

**Which servers this package ships** (the tool catalogue and parameter contracts):
`find_mcp_config()` (`agent/spatialomicsgym/mcp_config_path.py:65-111`) tries, in order:
1. `agent/MCP_server/mcp_config.yaml`, found from the package's own file (`parents[1]`, line 68);
2. the `SOG_MCP_CONFIG` pointer;
3. from the working directory: the generated `mcp_config.setup.yaml`, then `MCP_server/mcp_config.yaml` under the
   directory's agent part;
4. the wheel's read-only `_platform/` copy.

**The user overlay.** `resolve_user_config_path()` (`agent/spatialomicsgym/mcp_user_config.py:56-106`) answers
`SOG_MCP_USER_CONFIG` first; then a path that is absolute or exists from the working directory; then
`agent/MCP_server/mcp_config_user.yaml`, anchored at `agent/`; and off a checkout, the instance root's copy. The
writers listed above honour the same variable. `STCoscientist.add_mcp`
(`agent/spatialomicsgym/agent/stcoscientist.py:796`) merges the overlay into the served config when tool creation
is enabled. The merge never edits `mcp_config.yaml`, a shipped server wins any name conflict, and a corrupt overlay
is skipped with a warning.

## Why this directory cannot be renamed or moved

- **Instance-root marker.** `agent/spatialomicsgym/platform_root.py:155,215-218` treats
  `<agent part>/MCP_server/mcp_config.yaml` plus `<agent part>/tools/` as the sign that a directory is a checkout
  or a seeded home. Without it, `running_from_checkout()` is False and the instance root silently becomes the
  working directory or `~/.spatialomicsgym`.
- **Sibling paths.** The tool-creation managers write `../MCP_server/mcp_config_user.yaml` from
  `agent/tools_user/` (`agent/tools_user/knowledge_manager.py:73`), and the package finds the canonical file at
  `parents[1]/MCP_server/` (`agent/spatialomicsgym/mcp_config_path.py:68`).
- **Recorded paths.** The portal's launch line (`restart_portal.sh:80`), the installer's
  `ORIGINAL_MCP_CONFIG_REL` and `USER_MCP_CONFIG_REL` (`install/sog_install/constants.py:281,285`), the benchmark
  configs (`mcp_config_path: MCP_server/mcp_config.yaml`), `sog-setup pack` bundles
  (`userlayer/MCP_server/mcp_config_user.yaml`), and the `SOG_MCP_CONFIG` pointer in each machine's `.env` all
  spell this path.
- **Prompts.** The path appears in model-visible prompt text
  (`agent/spatialomicsgym/agent/prompt_builder.py:2288,2324`) and in the know-how playbooks; changing it there
  changes the system prompt and `KNOW_HOW_HASH`.
