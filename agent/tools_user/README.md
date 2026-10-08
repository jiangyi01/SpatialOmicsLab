# agent/tools_user/: the tool-creation library and the tools created on this machine

**What this is.** One flat directory with two kinds of content: the tracked library that creates, modifies, trashes and remembers user tools, and the tools the agent creates on this machine, each with its own portal, worker and env recipe.
**How runtime finds it.** As the import name `tools_user` (a namespace package with no `__init__.py`, importable only while `agent/` is on `sys.path`), and by path as the sibling of `agent/tools/` and `agent/MCP_server/`.
**What must not change.** Its name and place, the flat per-tool file names, the two symlinked helpers, and the rule that machine state here never leaves this machine.

## What is in this directory

| Kind | Entries | In git |
|---|---|---|
| Tool-creation library | `knowledge_manager.py`: creation-time knowledge, backups, the install log and safe modification. `trash_manager.py`: the two-stage trash (trash, restore, permanent delete). `memory_manager.py`: memory of earlier creation attempts. `self_review.py`: classifies a failed creation test and tries a targeted fix. `declarative.py`: declarative tools, each a JSON record per owner, with no conda env and no subprocess. `user_skill.py`: exposes created tools to the skill registry | tracked, shipped |
| Operator scripts | `create_all_21_tools.py`: drives memory-assisted creation of the tools from the tool-creation experiment. `seed_tool_creation_memory.py`: seeds the memory those runs read. `reclaim_envs.py`: lists `user_*` conda envs that no created tool claims, and deletes them only with `--remove` | tracked, shipped |
| Helper twins | `base_mcp.py` and `worker_utils.py`, relative symlinks to `../tools/base_mcp.py` and `../tools/worker_utils.py` | tracked; a wheel ships their real bytes |
| Lint configuration | `.ruff.toml`: `extend = "../../pyproject.toml"`, `target-version = "py37"` | tracked, shipped |
| Created tools | for each tool id: `<id>_mcp_server.py`, `<id>_worker.py` or `<id>_worker.R`, `<id>_env.yaml`, and sometimes `<id>_requirements.txt` | not tracked (ignored) |
| Vendored sources | `vendor_<id>/`: the upstream source a created tool was built from | not tracked (ignored) |
| Machine state | `.knowledge/<id>/` (creation knowledge and backups), `.memory/` (the signed memory store), `install_log.json` (the registry of created tools), `declarative/<owner>/` (declarative tool records), `.trash/` (trashed tools), `repos/`, `usage_log.jsonl`, `*.lock` | not tracked |
| Caches | `__pycache__/`, `.ruff_cache/` | not tracked (ignored); the only entries that are safe to delete |

Apart from this README, the tracked files are exactly the ones listed for this directory in `PLATFORM_PAYLOAD`
(`agent/spatialomicsgym/platform_root.py:121-140`) and in `MANIFEST.in`, so each of them ships. A
wheel carries them flat, as `tools_user/` inside its `_platform/` copy.

Created tools come from the tool-creation playbooks (`agent/spatialomicsgym/know_how/add_new_mcp_tool*.md`), which
write through `knowledge_manager`, and from the portal's broker (`agent/spatialomicsgym/agent/broker.py`). The
same steps wire each tool into `agent/MCP_server/mcp_config_user.yaml`; see
[MCP_server/README.md](../MCP_server/README.md).

## Machine state: rules

- `.memory/.secret` is the HMAC key that signs the memory store (written mode 400 by
  `agent/tools_user/memory_manager.py`). Never copy it, never commit it, never paste it.
- The memory store is signed, so write to it through `MemoryManager`, never by editing its JSON.
- `install_log.json` records each tool's files relative to this directory, and `.knowledge/` backups record
  `tools_user/<file>`. State is valid only beside the directory that wrote it; it does not move to another
  checkout or machine.
- Never stage this directory with `git add -A`. On 2026-10-07 `.trash/` and `declarative/` had no ignore rule;
  check with `git check-ignore -v <path>` before adding anything here.
- Delete only the caches. Deleting state by hand leaves the install log, the user config and the conda envs
  disagreeing; use the trash (`trash_manager`) or `reclaim_envs.py` instead.

## How runtime finds it

No single locator owns this directory; each of these derives it.

| Derivation | Site |
|---|---|
| From the module's own file: the directory, `.knowledge/`, `install_log.json`, and `../MCP_server/mcp_config_user.yaml` | `agent/tools_user/knowledge_manager.py:70-73`, `agent/tools_user/trash_manager.py:55-58`, `agent/tools_user/reclaim_envs.py:44-46` |
| `.memory/` beside the module | `agent/tools_user/memory_manager.py:82` |
| `SOG_DECLARATIVE_TOOLS_DIR`, else `layout.tools_user_dir()/declarative`, else `<instance root>/tools_user/declarative` | `agent/tools_user/declarative.py:120-142`, `agent/spatialomicsgym/layout.py:108-109` |
| The install log, anchored at `agent/` | `agent/spatialomicsgym/mcp_user_config.py:32,131-133` (`install_log_path()`) |
| A worker script by basename: `<agent>/tools`, then `<agent>/tools_user` | `agent/tools/base_mcp.py:136-153`, `agent/spatialomicsgym/agent/mcp_integration.py:575`, `install/sog_install/conncheck.py:419`, `agent/spatialomicsgym/tuning/integration.py:378` |
| The installer's copy of the location; on a new machine it rewrites each created tool's paths to this directory | `install/sog_install/constants.py:115-117` (`tools_user_dir()`), `install/sog_install/mcp_resolver.py:710` (`rebase_user_config`) |
| The import name | see [Who puts `agent/` on `sys.path`](../README.md#who-puts-agent-on-syspath) |

## What must not change

1. **Name and place.** `tools_user/` stays directly under `agent/`, beside `tools/` and `MCP_server/`. The
   derivations above, the symlink targets (`../tools/`) and `.ruff.toml` (`extend = "../../pyproject.toml"`) all
   count directories. The tool-creation playbooks import `tools_user.*` by name, and editing them changes
   `KNOW_HOW_HASH`.
2. **Flat per-tool names.** The broker writes only files whose names match its allowlist,
   `<id>_worker.py`, `<id>_mcp_server.py`, `<id>_env.yaml`, `<id>_worker.R`, `vendor_<id>/...` and
   `.knowledge/<id>/...`, relative to this directory (`agent/spatialomicsgym/agent/broker.py:71-73`). That
   allowlist is a security boundary. `KNOWN_SUFFIXES` (`agent/tools_user/knowledge_manager.py:107`) and the
   bundle's user-layer patterns (`install/sog_install/bundle.py:176`) find a tool's files by the same names.
   No created-tool path may contain `/tools/`: the installer would rewrite it into `agent/tools/`
   (`install/sog_install/wiring.py:114-123`).
3. **The helper twins stay beside the created tools.** Created portals do `from base_mcp import ...` from their
   own directory, and created workers that use `worker_utils` put their own directory on `sys.path` first.
   The twins are the same files as `agent/tools/base_mcp.py` and `agent/tools/worker_utils.py`: edit those, keep
   the links relative, and never replace a link with a copy.

## Operator scripts

They import `tools_user`, so run them from `agent/`:

```
cd agent
SOG_MEMORY_ENABLED=true python -m tools_user.seed_tool_creation_memory
SOG_MEMORY_ENABLED=true python -m tools_user.create_all_21_tools --preview
python -m tools_user.reclaim_envs
```

`create_all_21_tools` previews by default when no provider key is set; without `--preview` and with a key it
runs the agent for each tool. `reclaim_envs` only reports unless it is given `--remove`.
