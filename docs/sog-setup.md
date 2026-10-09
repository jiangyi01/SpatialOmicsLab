# The installer: `sog-setup`

`sog-setup` (also `python -m sog_install`) sets SpatialOmicsLab up from a fresh clone: it connects your LLM, lets you
pick the analysis tools you want, builds one conda environment per tool, tests each one, and writes the MCP
configuration that tells the agent what is ready on this machine. It is resumable, non-destructive to a running
project, and needs only the standard library and PyYAML, so it runs before any environment exists.

```console
$ sog-setup                      # the guided wizard (default)
$ sog-setup doctor               # read-only health report for a base env
$ sog-setup conncheck            # verify base env <-> MCP servers <-> ST-Coscientist are wired together
$ sog-setup capture              # freeze live tool envs into committed specs (contributors)
$ sog-setup chat                 # start the terminal chat in the env sog-setup built, tools wired
$ sog-setup pack                 # write a one-file transplant bundle (code + user tools + env recipes)
$ sog-setup unpack BUNDLE        # restore a bundle onto this machine
$ sog-setup reset                # remove a run's tool envs + test artifacts, keep the logs
```

## The wizard

A run walks seven phases in order and saves its state after every transition, so a crash or a Ctrl-C can be
resumed by running `sog-setup` again.

| Phase | Stage | What happens |
|---|---|---|
| `preflight` | A | Checks the machine: Python, conda/mamba/micromamba, disk, GPU, network (skip the network probe with `--no-net`), the local test data, and that the state directory is writable. |
| `onboarding` | A | Connects the LLM: provider, model, key; validates the key with a live ping and writes `.env`. Re-running detects a working `.env` and reuses it. |
| `category_select` | B | You pick skill categories (deconvolution, spatial clustering, ...) and, within them, servers. The picker shows a one-line description and a GPU hint per tool. |
| `base_env` | B | Creates or reuses the base (agent) environment and installs the package into it. |
| `provision` | B | Builds one conda environment per selected server from its recipe in `install/recipes/tool_specs/env/`. |
| `test` | B | Tests each tool: an import check always; a run on a small dataset when the local `test/` tree is present (it is not part of the repository). |
| `finalize` | B | Writes the install-aware MCP config, records `SOG_MCP_CONFIG` in `.env`, and rewrites `agent/MCP_server/mcp_config.yaml` for this machine after a timestamped backup (skip that rewrite with `--keep-agent-config`). |
| `demo` | B | Optional: a short demonstration and an offer to start the chat. Declining never fails the run. |

Stage A always runs on every launch; it is fast and read-only apart from the `.env` write. Stage B honours the resume
cursor: a phase that is already done is not re-run, and its result is read back from the saved state.

Every side-effecting step is proposed before it runs, and you confirm it. The decisions of Stage B come from an LLM-guided
interactive assistant (seeded with the key verified in onboarding) or, with `--answers`, from a scripted file with no
prompts and no LLM calls at all.

### Options

| Option | Effect |
|---|---|
| `--answers FILE.yaml` | Scripted answers: no prompts, no LLM. The reproducible path for servers and CI (below). |
| `--resume` / `--restart` | Continue a previous run without asking / archive it and start clean. |
| `--dry-run` | Print the plan and change nothing (best with `--answers`). |
| `--only A,B` | Restrict provisioning and testing to these server keys. |
| `--yes` | Auto-confirm every propose→execute gate (other prompts stay interactive). |
| `--no-net` | Skip the network preflight probe. |
| `--keep-agent-config` | Do not rewrite `agent/MCP_server/mcp_config.yaml`; the install-aware `install/recipes/mcp_config.setup.yaml` is still written. |
| `--skip-install` | Skip building tool environments and go straight to the chat (a small agent-core env is still built if none exists). |

Exit status: `0` success; `1` an unexpected error (your progress is saved, re-run to resume); `2` a malformed
`--answers` file or a prompt it did not answer; `3` conda not found; `130` interrupted.

### Scripted runs

`sog-setup --answers scenario.yaml` runs the whole Stage B from a file, with secrets taken from the environment, so
the same install can be replayed on another machine or in CI. All keys are optional unless noted:

```yaml
llm:                                   # consumed by onboarding
  provider: anthropic
  fields: {ANTHROPIC_API_KEY: "${env:ANTHROPIC_API_KEY}"}
  model: claude-opus-4-8
  validate: true
  reuse_existing: false
  on_validation_fail: retry            # retry | switch | continue_unvalidated
categories: [deconvolution]            # at least one category, or a non-empty `servers`
servers: [spacexr, tangram]            # optional explicit subset
base_env: {mode: new, name: sogdemo, install_editable: true}   # required; mode: new | reuse
provision: {on_tool_fail: continue}    # continue | abort
tests: {tier1: true, tier2: false, categories: [deconvolution], eval: false}
cleanup: {delete_artifacts: false}
service_keys: {UCD_TOKEN: "${env:UCD_TOKEN}"}
knobs: {SOG_DATA_PATH: ./data}
confirm: true                          # the scripted auto-confirm gate
```

`${env:VAR}` references are resolved from the environment, so no secret lives in the file.

## Subcommands

### doctor

```console
$ sog-setup doctor [--base NAME] [--no-net] [--json]
```

A read-only health report for an existing base environment: which tool environments exist, which import their
method, which are missing or broken. `--base` names the base environment (default: the last run's, from the saved
state); `--json` adds a machine-readable report.

### conncheck

```console
$ sog-setup conncheck [--base NAME] [--config PATH] [--no-net] [--deep] [--json]
```

Verifies the whole chain: the base environment, the MCP config (`--config`; default `SOG_MCP_CONFIG`, then the setup
config, then the canonical one), each portal's worker and interpreter, and the agent's view of the tools. `--deep`
also launches the real MCP portal for a few servers.

### chat

```console
$ sog-setup chat [--base NAME] [--no-mcp] [--dry-run] [-- stcoscientist options]
```

Starts `stcoscientist` inside the environment `sog-setup` built, with the analysis tools wired. `--no-mcp` starts it
without them; `--dry-run` prints the launch command.

### capture

```console
$ sog-setup capture [--only A,B] [--no-export]
```

For contributors. On a machine that already has a method's conda environment, writes
`install/recipes/tool_specs/<key>.yaml` and the pinned environment recipe under `install/recipes/tool_specs/env/`, so
a fresh clone can rebuild the environment. `--no-export` skips the pinned export. See the
[contribution guide](https://github.com/jiangyi01/SpatialOmicsLab/blob/main/agent/CONTRIBUTION.md).

### reset

```console
$ sog-setup reset [--basic NAME] [--keep-envs] [--with-base] [--prune-unrecorded] [--dry-run] [--yes]
```

Removes a run's tool environments and test artifacts and keeps the logs. `--keep-envs` clears only artifacts and
state; `--with-base` also removes the base environment (confirmed separately); `--prune-unrecorded` also removes
look-alike `<basic>_*` environments that have no build record.

### pack and unpack

```console
$ sog-setup pack [OUT] [--with-knowledge] [--with-repos] [--with-conda-pack ENV ...] [--allow-dirty] [--force] [--dry-run]
$ sog-setup unpack BUNDLE [--dest DIR] [--force] [--no-envs | --envs-only] [--yes]
```

The git repository carries the platform; it does not carry the layer a working deployment grows: the tools the agent
created under `agent/tools_user/`, their wiring in `agent/MCP_server/mcp_config_user.yaml`, the setup state and logs,
the curated `.knowledge` corpus, and the `user_*` conda environments. `pack` folds all of that, plus the exact working
tree it ran against, into one deterministic `tar.gz`; `unpack` replays it onto another checkout, rebuilds the `user_*`
environments from their recipes and rebases the user MCP config for the new machine. Built-in tool environments and
LLM keys stay `sog-setup`'s own resumable job on the destination.

Keys never travel in a bundle: `.env` files, the key vault, `.memory/` signing keys, key material and every `.git`
tree are refused by path, and text files are scanned for credential shapes before anything is written.

## Where the installer writes

| Path | Contents |
|---|---|
| `.sog_setup/` | Durable state (`setup_state.json`), logs, backups, the key vault (`llm_keys.json`, mode 0600), transcripts, bundles. Git-ignored. `SOG_SETUP_STATE_DIR` overrides the location. |
| `.env` | The provider settings and `SOG_MCP_CONFIG`. |
| `install/recipes/mcp_config.setup.yaml` | The install-aware MCP config: every server, enabled only when its environment exists and passes the import probe, with this machine's paths. |
| `install/recipes/env_overrides.env` | The flat list of interpreter overrides the config uses. |
| `agent/MCP_server/mcp_config.yaml` | The canonical config, rewritten for this machine unless `--keep-agent-config`. |
| `test/installation/` | Per-run test artifacts (not a test suite; emptied by every run). |
| conda environments | One per selected tool, named from its spec; the base environment; the `user_*` environments of created tools. |

The installer only ever creates or repairs conda environments in its own namespace and never edits a server file.
