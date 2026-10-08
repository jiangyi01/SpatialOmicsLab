"""
Wiring — generate the wizard's OWN MCP config, non-destructively.

``agent.add_mcp(path)`` makes ``path`` the base config and **replaces** the
agent's ``agent/MCP_server/mcp_config.yaml`` (only ``mcp_config_user.yaml`` merges on
top). So the generated ``install/recipes/mcp_config.setup.yaml`` deliberately contains
**only the servers this run provisioned** — the agent can then route to nothing
but the freshly-built ``<basic>_<server>`` envs. The agent's real config is
never read-modified; we only ever *write* new files under ``install/recipes/``.

Per provisioned server we rewrite two things, both grounded in what the MCP
stdio client actually forwards (only ``HOME/LOGNAME/PATH/SHELL/TERM/USER`` + the
server's ``env:`` block reach the subprocess):

* ``command`` → ``[<base-env python>, <clone>/agent/tools/<server>_mcp_server.py]`` —
  the thin server runs in the base env; the script path is rebased to the live
  checkout.
* ``env`` → the interpreter override (``{PREFIX}_PYTHON`` / literal
  ``SEURAT_RSCRIPT``) repointed to ``/opt/conda/envs/<basic>_<server>/bin/…`` and
  the worker override (``{PREFIX}_WORKER``) rebased to ``<clone>/agent/tools/<worker>``.
  Any *other* env vars already on the original server (service keys, CUDA flags)
  are preserved.

The worker override is not cosmetic: cell2location has **no** ``env:`` block and
hardcodes an absolute worker default that is wrong on a fresh clone, so we must
add ``CELL2LOCATION_WORKER`` explicitly.

.. note::
   **Production wiring lives in** :func:`mcp_resolver.resolve_full_config` — it
   iterates the *union* of the canonical set and the specs (drift-robust) and does a
   LIVE per-server interpreter probe. The standalone :func:`generate_config` /
   :func:`build_server_entry` / :func:`wire` here are the earlier spec-only path,
   superseded and **no longer called by the wizard**; they are retained because their
   tests (``test_wiring.py``) also cover the *shared* rebasing primitives
   (:func:`rebase_tools_path`, :meth:`ToolSpec.rewired_override_vars`,
   :func:`base_python_path`) and ``mcp_resolver`` reuses :func:`load_original_servers`
   + :func:`env_overrides_text` from this module. **Fix production wiring behavior in**
   ``mcp_resolver``, not in the standalone trio — editing the dead path changes nothing
   the agent runs.

Stdlib + pyyaml only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from . import constants

if TYPE_CHECKING:
    from .envtools import Conda
    from .session_log import SessionLog
    from .specs import ToolSpec


@dataclass
class WiringResult:
    config_path: Path
    overrides_path: Path
    wired: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # provisioned but no spec / no override


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
# The runtime user-config merger (``spatialomicsgym/agent/mcp_config_merger.py``), loaded BY FILE PATH:
# ``spatialomicsgym.agent.__init__`` imports the full agent stack (langchain and friends),
# which the stdlib+pyyaml setup layer must not require. The merger module itself is
# stdlib-only at import time. ONE loader for the whole setup layer — conncheck reads the
# user overlay and mcp_resolver rewrites it, and both must agree with the live agent on
# what the file means (its path resolution, its stray-top-level-block recovery), so
# neither may fork a private copy of those rules.
_MERGER_PATH = constants.PACKAGE_ROOT / "agent" / "mcp_config_merger.py"


def merger_module():
    """The real ``mcp_config_merger`` module — never a re-implementation of its rules."""
    import importlib.util
    import sys

    name = "_sog_setup_mcp_config_merger"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, _MERGER_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def base_python_path(basic_env: str, *, conda: Conda | None = None) -> str:
    """Absolute python for the base env (resolve via conda; fall back to the convention)."""
    # Per-OS interpreter layout via constants.interp_path (a1c#2): <env>\python.exe on Windows.
    if conda is not None:
        try:
            prefix = conda.env_prefix(basic_env)
        except Exception:
            # A slow/loaded box can make the `conda env list` probe behind env_prefix time out and
            # raise CondaError (envtools `_exec` translates TimeoutExpired/OSError into CondaError even
            # with check=False). That must never crash config resolution — fall through to the
            # convention path (the same value returned when conda is None) so finalize still writes a
            # usable config. Mirrors resolve_interpreter / envs_root's `except Exception` guard. (N1/R20)
            prefix = None
        if prefix:
            return constants.interp_path(prefix, "python")
    return constants.interp_path(f"{constants.conda_envs_root()}/{basic_env}", "python")


def rebase_tools_path(original: str, repo_root: Path) -> str:
    """Repoint any ``…/tools/<file>`` path at the live checkout's ``agent/tools/`` dir.

    Server and worker files are flat under ``tools/``, so the basename is enough; a path recorded
    before the re-layout (``<clone>/tools/<file>``) lands in ``agent/tools`` too.
    A path that isn't under ``tools/`` is returned unchanged.
    """
    p = str(original)
    if "/tools/" in p or p.startswith("tools/"):
        return str(constants.tools_dir(repo_root) / Path(p).name)
    return p


# --------------------------------------------------------------------------- #
# One server entry
# --------------------------------------------------------------------------- #
def build_server_entry(
    original_meta: dict,
    spec: ToolSpec,
    *,
    basic_env: str,
    base_python: str,
    repo_root: Path,
) -> dict:
    """Return a rewired copy of one server's config meta (original untouched)."""
    meta = dict(original_meta) if original_meta else {}
    target_env = spec.target_env(basic_env)

    # command: [base python, rebased server script]
    # A hand-authored scalar `command: "python foo.py"` must NOT be char-split by list(str); guard to
    # a list (mirrors resolve_server_entry's live (C) guard), else fall through to spec.server_file.
    raw_cmd = meta.get("command")
    cmd = list(raw_cmd) if isinstance(raw_cmd, list) else []
    server_script = ""
    if len(cmd) >= 2:
        server_script = rebase_tools_path(cmd[1], repo_root)
    elif spec.server_file:
        server_script = str(constants.tools_dir(repo_root) / spec.server_file)
    meta["command"] = [base_python, server_script] if server_script else [base_python]

    # env: preserve extras, then overwrite the interpreter + worker overrides.
    env = dict(meta.get("env") or {})
    env.update(spec.rewired_override_vars(target_env))  # {PREFIX}_PYTHON / SEURAT_RSCRIPT
    if spec.worker_var and spec.worker_file:
        env[spec.worker_var] = str(constants.tools_dir(repo_root) / spec.worker_file)
    elif spec.worker_var and env.get(spec.worker_var):
        env[spec.worker_var] = rebase_tools_path(env[spec.worker_var], repo_root)
    meta["env"] = env
    meta["enabled"] = True
    return meta


# --------------------------------------------------------------------------- #
# Whole config
# --------------------------------------------------------------------------- #
def load_original_servers(path: Path | None = None) -> dict:
    """Read the agent's MCP config for each server's ``command``/``env`` seed.

    Best-effort and non-fatal: if the agent's config is missing or malformed we
    return ``{}`` so wiring still builds valid entries from each tool's spec
    (server_file + override vars) rather than crashing the provision phase after
    envs are already built. The file is only ever *read* here, never modified.
    """
    p = path or constants.original_mcp_config()
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        # A non-UTF-8 original config raises UnicodeDecodeError (a ValueError, NOT an OSError).
        # This reader is reused live by mcp_resolver.resolve_full_config, so an escape would
        # PERMANENTLY degrade wiring (canonical never rewritten → same crash every re-run).
        return {}
    if not isinstance(cfg, dict):
        return {}
    servers = cfg.get("mcp_servers") or {}
    # `mcp_servers:` written as a list/scalar would otherwise return a non-dict here and break
    # `mcp_resolver.resolve_full_config`'s `original.get(...)`. Best-effort: fall back to {}. (B)
    if not isinstance(servers, dict):
        return {}
    # Per-entry hygiene (R24-D1): resolve_full_config sorts `set(original) | set(specs)`
    # (mcp_resolver.py:487 — a non-STRING key raises TypeError on the mixed-type sort) and builds
    # each entry via `dict(orig_meta or {})` (:270/:430 — a *truthy* non-mapping value raises
    # TypeError/ValueError; None/{}/"" are already coerced to {} there). A single hand-corrupted
    # entry would otherwise poison the resolve of ALL 88 servers, and because this reader runs live
    # on every finalize, a persistent malformed original config would degrade every re-run to
    # all-tools-disabled with no path to recovery. Tolerant-skip ONLY the crash-causing entries so
    # behavior is unchanged for every input that currently works (mirrors specs.load_all_specs's
    # skip-one-bad-spec-keep-the-catalog stance).
    return {k: v for k, v in servers.items() if isinstance(k, str) and (isinstance(v, dict) or not v)}


def generate_config(
    basic_env: str,
    provisioned: list[str],
    specs: dict[str, ToolSpec],
    *,
    base_python: str,
    repo_root: Path | None = None,
    original_servers: dict | None = None,
) -> tuple[dict, list[str], list[str]]:
    """Build the pruned, rewired ``{mcp_servers: {...}}`` dict.

    Returns ``(config, wired, skipped)``. Only servers that are both provisioned
    *and* have a spec with an interpreter override are wired.
    """
    repo_root = repo_root or constants.repo_root()
    original = original_servers if original_servers is not None else load_original_servers()

    servers: dict[str, dict] = {}
    wired: list[str] = []
    skipped: list[str] = []
    for key in provisioned:
        spec = specs.get(key)
        if spec is None or not spec.override_vars:
            skipped.append(key)
            continue
        entry = build_server_entry(
            original.get(key, {}),
            spec,
            basic_env=basic_env,
            base_python=base_python,
            repo_root=repo_root,
        )
        servers[key] = entry
        wired.append(key)
    return {"mcp_servers": servers}, wired, skipped


def env_overrides_text(config: dict) -> str:
    """Flat ``KEY=VALUE`` dump of every provisioned server's env block.

    For the direct-subprocess / self-review remediation path only (the MCP path
    uses the per-server ``env:`` blocks above).
    """
    lines: list[str] = [
        "# Generated by sog-setup wiring — per-tool interpreter/worker overrides.",
        "# Sourced only by the direct-subprocess/remediation path; MCP uses env: blocks.",
    ]
    seen: set[str] = set()
    for _, meta in sorted((config.get("mcp_servers") or {}).items()):
        for k, v in (meta.get("env") or {}).items():
            if k not in seen:
                seen.add(k)
                lines.append(f"{k}={v}")
    return "\n".join(lines) + "\n"


def wire(
    basic_env: str,
    provisioned: list[str],
    specs: dict[str, ToolSpec],
    *,
    conda: Conda | None = None,
    base_python: str | None = None,
    repo_root: Path | None = None,
    log: SessionLog | None = None,
) -> WiringResult:
    """Generate + write ``install/recipes/mcp_config.setup.yaml`` and ``install/recipes/env_overrides.env``.

    Writing config files is always safe (they're the wizard's own, git-ignored);
    the agent's ``agent/MCP_server/mcp_config.yaml`` is never touched.
    """
    repo_root = repo_root or constants.repo_root()
    if base_python is None:
        base_python = base_python_path(basic_env, conda=conda)

    config, wired, skipped = generate_config(
        basic_env,
        provisioned,
        specs,
        base_python=base_python,
        repo_root=repo_root,
    )

    cfg_path = constants.generated_mcp_config()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    ov_path = constants.generated_env_overrides()
    ov_path.write_text(env_overrides_text(config), encoding="utf-8")

    if log is not None:
        log.event("wiring_done", config=str(cfg_path), wired=wired, skipped=skipped, base_python=base_python)
    return WiringResult(config_path=cfg_path, overrides_path=ov_path, wired=wired, skipped=skipped)
