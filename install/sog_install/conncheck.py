"""
``sog-setup conncheck`` — verify the connection chain the whole system rides on:

    base env  ──►  MCP client stack  ──►  MCP servers (tools)
        └────────────►  ST-Coscientist agent

A fresh clone can have every conda env built and still be *unconnected*: the agent
env can't import the agent, the MCP client libs are missing, or the wired
``mcp_config`` points every server at an interpreter that isn't there. ``doctor``
reports on each **tool worker env** in isolation; ``conncheck`` answers the different,
end-to-end question — *is the whole thing actually wired together?*

It is **read-only** and **key-free by default**: it never builds/mutates an env, never
writes config, and needs no LLM key (constructing the agent or calling an LLM is the
opt-in ``--deep`` escalation). Three layers, each a section with per-check marks:

1. **Machine preflight** — the same read-only probe ``doctor`` runs (net/disk/conda).
2. **Agent core & MCP client** — an out-of-process import probe *inside the base env*:
   can it import ``STCoscientist`` and the ``nest_asyncio``/``fastmcp``/``mcp`` stack
   that ``agent.add_mcp()`` needs to launch and talk to servers?
3. **MCP config wiring** — locate the active ``mcp_config``, count enabled servers, and
   confirm each enabled server's interpreter + server script actually exist on disk
   (an ``enabled: true`` server whose interpreter is missing is a broken connection).

``--deep`` additionally runs the in-repo smoke harness's static portal checks (levels 1-2:
server files parse and import what they should, the configured env and interpreter exist) for a
few enabled servers of the TRACKED config. It launches no portal — it is not a handshake — and it
degrades to a note when the harness is not part of this install.

Exit 0 only when the chain is whole: agent + MCP client import, the base env is present,
at least one enabled server is reachable, and the machine preflight has no hard failure.

Stdlib + pyyaml only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from spatialomicsgym import platform_root
from spatialomicsgym.mcp_user_config import shipped_identity, user_function_names, user_server_skip_reason

from . import constants, mcp_resolver, preflight, wiring
from .envtools import Conda, CondaError
from .prompts import PromptIO

if TYPE_CHECKING:
    from collections.abc import Callable
from .state import SetupState

_OK = "ok"
_WARN = "warn"
_FAIL = "fail"

# Sentinel the in-env probe prints its JSON on; we ``rfind`` it so the agent's import-time
# banner noise ("Loaded environment variables from .env", etc.) can never corrupt the contract.
_PROBE_SENTINEL = "SOG_CONNCHECK_JSON "

# The probe body, run as ``python -c`` INSIDE the base env. Two groups: the agent itself,
# and the MCP client stack ``add_mcp()`` depends on. ASCII-only, no f-strings — it is shipped
# verbatim across the ``conda run`` boundary.
_PROBE_SRC = (
    "import json, importlib\n"
    "groups = {\n"
    "    'agent': ['spatialomicsgym', 'spatialomicsgym.agent'],\n"
    "    'mcp_client': ['nest_asyncio', 'fastmcp', 'mcp', 'spatialomicsgym.agent.mcp_config_merger'],\n"
    "}\n"
    "attrs = {'spatialomicsgym.agent': 'STCoscientist',\n"
    "         'spatialomicsgym.agent.mcp_config_merger': 'build_merged_mcp_config'}\n"
    "out = {}\n"
    "for grp, mods in groups.items():\n"
    "    res = {}\n"
    "    for m in mods:\n"
    "        try:\n"
    "            mod = importlib.import_module(m)\n"
    "            a = attrs.get(m)\n"
    "            if a:\n"
    "                getattr(mod, a)\n"
    "            res[m] = 'ok'\n"
    "        except Exception as e:\n"
    "            res[m] = (type(e).__name__ + ': ' + str(e))[:200]\n"
    "    out[grp] = res\n"
    "print('" + _PROBE_SENTINEL + "' + json.dumps(out))\n"
)


# --------------------------------------------------------------------------- #
# Layer 2 — agent + MCP client import probe (runs in the base env)
# --------------------------------------------------------------------------- #
def _run_import_probe(conda: Conda, basic_env: str, *, timeout: int = 180) -> dict:
    """Import ``STCoscientist`` + the MCP client stack inside ``basic_env``; never raises.

    Returns ``{"agent": {mod: status}, "mcp_client": {mod: status}}`` where each status is
    ``"ok"`` or a short ``"ExcType: message"``. On a probe that never produced the sentinel
    (env broken, conda missing, timeout) every module is marked with the failure reason so the
    report degrades to a loud, honest FAIL rather than a crash.
    """
    try:
        res = conda.run(basic_env, ["python", "-c", _PROBE_SRC], timeout=timeout, check=False)
    except CondaError as exc:
        # `Conda._exec` raises even with check=False on a timeout/OSError. For a diagnostic that must
        # always finish its report, that's a *connection finding* (agent import hung / conda flaked),
        # not a reason to abort — turn it into a degraded FAIL row instead of unwinding to exit 3.
        detail = f"probe could not run: {str(exc)[:160]}"
        return {"agent": {"spatialomicsgym.agent": detail}, "mcp_client": {"nest_asyncio": detail}}
    stdout = res.stdout or ""
    idx = stdout.rfind(_PROBE_SENTINEL)
    if idx != -1:
        try:
            parsed = json.loads(stdout[idx + len(_PROBE_SENTINEL) :].splitlines()[0])
            # Require BOTH groups to be dicts, not merely *present*: a payload like
            # {"agent": null, "mcp_client": null} would pass a key-presence check yet later crash the
            # report loop (`agent_group.items()` on None). A malformed/null-group payload degrades to
            # the "probe failed" FAIL row below instead — this function's documented "honest FAIL,
            # never crash" contract.
            if isinstance(parsed, dict) and all(isinstance(parsed.get(k), dict) for k in ("agent", "mcp_client")):
                return parsed
        except (ValueError, IndexError):
            pass
    reason = (res.stderr or stdout or "probe produced no output").strip().splitlines()
    detail = f"probe failed: {reason[-1][:160]}" if reason else "probe failed"
    return {
        "agent": {"spatialomicsgym.agent": detail},
        "mcp_client": {"nest_asyncio": detail},
    }


def _group_ok(group: dict) -> bool:
    return bool(group) and all(v == "ok" for v in group.values())


# --------------------------------------------------------------------------- #
# Layer 3 — MCP config wiring
# --------------------------------------------------------------------------- #
def _discover_config(explicit: str | None, io: PromptIO | None = None) -> Path | None:
    """Which ``mcp_config`` is actually in force. Precedence: explicit ``--config`` >
    ``SOG_MCP_CONFIG`` (what the resolver records) > the wizard's generated setup config >
    the canonical instance-root config > the wheel's read-only ``_platform`` canonical
    (pip-only installs; absent on checkouts). Returns ``None`` only if literally none of
    them exist. The two ``constants`` helpers anchor on ``constants.repo_root()``, so off a
    checkout they already answer inside the seeded SOG_HOME; the packaged rung covers the
    box where nothing has been seeded yet, so the diagnostic can still report on the config
    a bare ``pip install`` session would wire.

    Given an ``io``, a recorded ``SOG_MCP_CONFIG`` that cannot be used is reported rather than
    quietly replaced — "which config is in force?" is the question this whole check exists to
    answer, so a substitution the user did not ask for is exactly what must not go unsaid. Only
    when the pointer was actually consulted: an explicit ``--config`` outranks it, and complaining
    then would send the user to fix something that changed nothing."""
    candidates: list[Path] = []
    if explicit:
        chosen = Path(explicit).expanduser()
        try:
            present = chosen.is_file()
        except OSError:
            present = False
        if not present:
            # A mistyped --config fell through to SOG_MCP_CONFIG / the setup / the canonical config and
            # the report described that other file -- the one substitution this check exists to rule
            # out, and the CLI's own --mcp refuses it (chat_cli._resolve_mcp_config) (hunt 2026-09-30,
            # u37-setup-checks-13).
            if io is not None:
                io.err(f"--config {chosen} is not a file — nothing to check (no other config substituted)")
            return None
        candidates.append(chosen)
    # ``.strip()`` mirrors chat_cli._resolve_mcp_config, which this function is the diagnostic
    # half of. Without it the two disagreed for a pointer carrying stray whitespace -- an
    # `export SOG_MCP_CONFIG="$CFG "`, a path read out of a file with its newline, a hand-edited
    # ``.env`` -- and the CLI wired the pointed-at config while this report named a fallback the
    # session was not using. Reporting on a different file than the session runs is the one
    # failure this whole check exists to rule out.
    env_pointer = (os.environ.get("SOG_MCP_CONFIG") or "").strip()
    if env_pointer:
        candidates.append(Path(env_pointer).expanduser())
    candidates.append(constants.generated_mcp_config())
    candidates.append(constants.original_mcp_config())
    packaged = platform_root.packaged_canonical_config()
    if packaged is not None:
        candidates.append(packaged)
    resolved: Path | None = None
    for c in candidates:
        try:
            if c.is_file():
                resolved = c
                break
        except OSError:
            continue
    if io is not None and not explicit:
        note = constants.stale_mcp_pointer_note(resolved)
        if note:
            io.warn(note)
    return resolved


def _load_servers(config_path: Path) -> dict:
    """Parse ``{server_key: entry}`` out of a config file; ``{}`` on any read/parse error."""
    import yaml

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        # UnicodeDecodeError (a ValueError subclass, NOT an OSError) fires when the config was
        # saved latin-1/utf-16/with stray bytes — the read-only conncheck report must still
        # degrade to "no servers", never abort mid-diagnostic. Mirrors answers/categories/wiring.
        return {}
    if not isinstance(data, dict):
        return {}
    servers = data.get("mcp_servers") or data.get("mcpServers") or {}
    return servers if isinstance(servers, dict) else {}


def _load_user_servers() -> dict:
    """``{server_key: entry}`` from the user-tool config; ``{}`` when there is none.

    The tool-creation playbook writes ``agent/MCP_server/mcp_config_user.yaml`` and the agent
    runtime merges it in at launch (``mcp_config_merger`` — original wins), so a conncheck
    that reads only the canonical/generated config reports "fully wired" while every
    user-created tool goes un-health-checked. The file is ``mcp_resolver.user_config_path()``:
    the ``SOG_MCP_USER_CONFIG`` override the agent honours, else the repo-anchored copy
    (``constants.user_mcp_config()`` — the SOG_SETUP_REPO_ROOT seam) — NOT the merger's
    CWD rung: this diagnostic reports on a chosen root, and the CWD rung would read a
    *different checkout's* overlay whenever conncheck runs from one (and defeats the seam
    in tests). The merger is still loaded (via
    ``wiring.merger_module``, by path so this stdlib+pyyaml diagnostic never pulls the
    agent's langchain stack) but only for its stray-top-level-block recovery, so this
    report and the live agent agree on what the file MEANS. Every failure mode degrades
    to ``{}`` — the user layer is optional, and a malformed user file must not abort the
    report (mirrors ``_load_servers``).
    """
    import yaml

    path = mcp_resolver.user_config_path()  # SOG_MCP_USER_CONFIG first, as the agent reads it (u37-setup-checks-9)
    try:
        merger = wiring.merger_module()
    except Exception:
        merger = None  # no stray-block recovery, mcp_servers: blocks still checked
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    declared = data.get("mcp_servers")
    if not isinstance(declared, dict):
        declared = {}
    recovered: dict = {}
    if merger is not None:
        try:
            recovered = merger._recover_top_level_servers(data, declared)
        except Exception:
            recovered = {}
    # The merger's order (stray top-level blocks first, then ``mcp_servers:``), because a later user
    # server loses a function-name clash to an earlier one and the report must agree on which.
    merged = {**recovered, **declared}
    return {k: v for k, v in merged.items() if isinstance(v, dict)}


def _duplicate_server_keys(config_path: Path) -> dict[str, int]:
    """``{server key: times declared}`` for keys spelled more than once; ``{}`` if unreadable.

    A YAML mapping is last-wins, and ``yaml.safe_load`` applies that rule in silence: a config that
    spells two server keys twice parses with two fewer servers, the earlier block of each pair —
    its ``command``, its ``env`` wiring and every tool it declares — discarded before any reader
    sees it. Every config reader in this system loads with ``safe_load``, so the loss is invisible
    to all of them, including the count ``_config_report`` prints.

    Compose the node tree rather than construct it: the nodes keep every key the file spells,
    duplicates included. Degrades to ``{}`` on the same read/parse failures ``_load_servers``
    swallows — those already have their own channel in the report, and a read-only diagnostic must
    not abort mid-run over one.
    """
    import yaml

    try:
        root = yaml.compose(config_path.read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return {}
    if not isinstance(root, yaml.MappingNode):
        return {}
    # Mirror _load_servers' precedence: 'mcp_servers' wins, 'mcpServers' is the accepted alias.
    by_spelling = {
        str(k.value): v
        for k, v in root.value
        if getattr(k, "value", None) in ("mcp_servers", "mcpServers") and isinstance(v, yaml.MappingNode)
    }
    servers = by_spelling.get("mcp_servers") or by_spelling.get("mcpServers")
    if servers is None:
        return {}
    counts: dict[str, int] = {}
    for key_node, _value in servers.value:
        name = getattr(key_node, "value", None)
        if isinstance(name, str):
            counts[name] = counts.get(name, 0) + 1
    return {name: n for name, n in counts.items() if n > 1}


def _drifted_servers(config_path: Path, project: Callable[[object], object]) -> list[str]:
    """Server keys where ``project`` reads differently here than in the tracked config.

    The config in force is not authored — it is a *copy*. ``mcp_resolver.resolve_full_config`` loads
    the tracked ``agent/MCP_server/mcp_config.yaml`` and rebases each block onto this device;
    ``_rebase_only_entry`` starts from ``dict(original_meta)`` and rewrites exactly ``command``,
    ``env`` and ``enabled``. The ``tools:`` block — every function name, description and declared
    parameter the model is shown — is copied through untouched. So the generated file is a cache
    with no invalidation: pull a fix that corrects a tool's declared parameters and the running
    system keeps serving the old declarations until someone re-runs the wizard.

    Two restrictions, both from the resolver's own behaviour. ``command``/``env``/``enabled`` are
    never compared, because they are *supposed* to differ — rebasing them onto this box is the
    point. And only keys both files declare are compared, because a key present in one alone is
    ordinary: the tracked file gains servers as tools are added, and a config may legitimately be
    trimmed. A ``tools:`` block that differs for a server both files declare is the one difference
    the resolver cannot produce, so it can only be age.

    Comparing the tracked config against itself is a no-op — it is ``_discover_config``'s last
    candidate, so a fresh clone that has never run the wizard reads it directly. An absent or
    unreadable tracked config yields ``[]``: an installed wheel has no ``MCP_server/`` beside it,
    and with nothing to compare against the honest report is silence.
    """
    canonical = constants.original_mcp_config()
    try:
        if config_path.resolve() == canonical.resolve():
            return []
    except OSError:
        return []
    tracked = _load_servers(canonical)
    live = _load_servers(config_path)
    if not tracked or not live:
        return []
    return [key for key in sorted(set(tracked) & set(live)) if project(tracked[key]) != project(live[key])]


def _stale_tool_blocks(config_path: Path) -> list[str]:
    """Server keys whose ``tools:`` block differs from the tracked config's; ``[]`` if incomparable.

    The broad question: has *anything* the model is shown moved since this file was written? What
    it adds over ``_stale_parameter_blocks`` is drift in prose alone — a reworded description, a
    filled-in placeholder — which is age without consequence. Use ``_stale_parameter_blocks`` for
    the part that can change a call.
    """
    return _drifted_servers(config_path, _tools_of)


def _stale_parameter_blocks(config_path: Path) -> list[str]:
    """Server keys whose declared *call contract* differs from the tracked config's.

    The sharp half of ``_stale_tool_blocks``. A parameter the config declares and the portal no
    longer accepts is a knob the model will set, the portal will reject or drop, and the run will
    record as configured — exactly what ``test/test_config_declares_only_real_parameters`` exists
    to prevent, arriving through a file that test does not read. The reverse is a loss too: a
    parameter added on the portal and absent here is a lever the model is never offered. And a
    parameter that is still there under a different ``required``, ``default`` or ``type`` is the
    same class of harm without the name moving: the model is told to omit an argument the portal
    cannot run without, or shown a recommended value the checkout has since rejected.

    Kept separate because the two severities do not travel together, and the gap is wide enough to
    be worth the second number: measured across the two shipped configs, 56 servers had drifted, 40
    of them in the contract and 16 in parameter prose alone.
    """
    return _drifted_servers(config_path, _declared_parameters)


def _first_few(names: list[str], limit: int = 8) -> str:
    """``a, b, c`` — truncated with a count, so a long list cannot swamp the report."""
    head = ", ".join(names[:limit])
    return head + (f", … and {len(names) - limit} more" if len(names) > limit else "")


def _tools_of(entry: object) -> object:
    """The ``tools:`` block of a server entry, or ``None`` when it declares none."""
    return entry.get("tools") if isinstance(entry, dict) else None


def _declared_parameters(entry: object) -> object:
    """``{tool function: {parameter: (required, default, type)}}``, or ``None`` if unreadable.

    The call contract, and only the call contract. ``description`` is left out because it is the one
    field in a ``parameters:`` mapping that cannot move an argument: reword it and the model still
    sends the same call. The other three all can — ``attach_kwarg_signature`` reads ``required`` to
    decide whether a parameter gets a ``None`` default or none at all, and renders ``default`` and
    ``type`` into what the model reads before choosing a value.

    Names survive as the mapping's keys, so a parameter that appeared or vanished still reports.
    """
    tools = entry.get("tools") if isinstance(entry, dict) else None
    if not isinstance(tools, list):
        return None
    declared: dict[str, dict[str, tuple[object, object, object]]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        params = tool.get("parameters")
        contract: dict[str, tuple[object, object, object]] = {}
        if isinstance(params, dict):
            for name, body in params.items():
                spec = body if isinstance(body, dict) else {}
                contract[str(name)] = (spec.get("required"), spec.get("default"), spec.get("type"))
        declared[str(tool.get("spatialomicsgym_name"))] = contract
    return declared


def _rebased_script_args(args: list) -> list:
    """``mcp_integration._rebase_script_args``, restated: a missing ABSOLUTE ``.py``/``.R`` arg whose
    basename this install's ``tools/`` (or its sibling ``tools_user/``, built-ins first) holds is the
    local copy. The agent does this at launch, so a moved or renamed checkout still serves every
    server; this report read the stale path and called the whole box BROKEN (hunt 2026-09-30,
    u37-setup-checks-14). Restated, not imported: that module pulls in the agent package, and this
    diagnostic must stay stdlib + pyyaml. ``test/test_conncheck_reachability_follows_the_agents_launch_gate.py``
    pins the two to the same answers."""
    tools = platform_root.tools_dir()
    if tools is None:
        return list(args)
    search = (tools, tools.parent / "tools_user")
    out = []
    for a in args:
        if isinstance(a, str) and a.endswith((".py", ".R")) and os.path.isabs(a) and not os.path.exists(a):
            local = next((d / os.path.basename(a) for d in search if (d / os.path.basename(a)).exists()), None)
            out.append(str(local) if local is not None else a)
        else:
            out.append(a)
    return out


def _worker_interpreter_missing(entry: dict, portal_interp: str) -> str | None:
    """``"<KEY>=<path>"`` for the first worker interpreter the server's ``env`` pins that is gone, else None.

    The portal (``command[0]``) runs in the base env; the tool's own worker runs under the
    ``<KEY>_PYTHON`` / ``<KEY>_RSCRIPT`` interpreter the wiring pins in ``env``. That pin was never
    read, so a deleted tool env reported "ok" -- while the wizard tells users conncheck catches exactly
    that (hunt 2026-09-30, uL4-honesty-8). The dispatcher (``base_mcp._resolve_worker_python``) still
    tries the env's legacy alias under the pinned root, then the same name and the alias under the
    conda root the portal itself runs from; a pin counts as missing only when none of those exists."""
    env = entry.get("env")
    if not isinstance(env, dict):
        return None
    aliases: dict[str, str] = {}
    for branded, legacy in (getattr(constants, "LEGACY_ENV_ALIASES", None) or {}).items():
        aliases[str(branded)], aliases[str(legacy)] = str(legacy), str(branded)
    live_root = ""
    if os.path.isabs(portal_interp):
        marker = os.sep + "envs" + os.sep
        idx = portal_interp.find(marker)
        live_root = portal_interp[:idx] if idx != -1 else os.path.dirname(os.path.dirname(portal_interp))
    for key, value in env.items():
        path = str(value)
        if not str(key).endswith(("_PYTHON", "_RSCRIPT")) or not os.path.isabs(path) or os.path.exists(path):
            continue
        parts = path.split(os.sep)
        if "envs" in parts and parts.index("envs") + 1 < len(parts):
            i = parts.index("envs")
            name, tail, pinned_root = parts[i + 1], parts[i + 2 :], os.sep.join(parts[:i])
            alias = aliases.get(name)
            tries = [(pinned_root, alias), (live_root, name), (live_root, alias)]
            if any(r and n and os.path.exists(os.sep.join([r, "envs", n, *tail])) for r, n in tries):
                continue
        return f"{key}={path}"
    return None


def _server_reachable(entry: dict) -> tuple[bool, str]:
    """Is this server's wired command runnable? Checks the interpreter, the server script and the
    tool's worker interpreter.

    ``command`` is ``[interpreter, server_script.py, ...]``. An absolute interpreter that
    doesn't exist ⇒ a broken connection (the env was removed / never built / is another box's
    path). A bare ``python`` ⇒ un-resolved config (fresh clone) — reported as a soft WARN, not a
    hard break, since it may still resolve on PATH.

    The order is the agent's launch gate (``mcp_integration``: rebase the script args, gate the
    interpreter, gate the script), so this report and the agent agree on which servers are served
    (hunt 2026-09-30, u37-setup-checks-14). The script gate used to be skipped for a bare
    interpreter, which the agent never skips.
    """
    cmd = entry.get("command")
    if not isinstance(cmd, list) or not cmd:
        return False, "no command"
    interp = str(cmd[0])
    args = _rebased_script_args([str(a) for a in cmd[1:]])
    soft, portal = "ok", interp
    if os.path.isabs(interp):
        if not Path(interp).exists():
            return False, f"interpreter missing: {interp}"
    else:
        # A bare interpreter (e.g. "python") is an un-pinned config (fresh clone). The
        # docstring promises this is a soft case, not a hard break: if it resolves on
        # PATH it can actually launch, so accept it (with a nudge to pin it via the
        # wizard); only a bare name that is NOT on PATH is genuinely unrunnable.
        resolved = shutil.which(interp)
        if not resolved:
            return False, f"interpreter not resolved (bare '{interp}' not on PATH — run the wizard)"
        soft, portal = f"bare '{interp}' resolves on PATH ({resolved}); run the wizard to pin it", resolved
    # Every absolute .py/.R argument must exist, as at the agent's gate (``_missing_script_arg``).
    script = next((a for a in args if a.endswith((".py", ".R")) and os.path.isabs(a) and not os.path.exists(a)), None)
    if script:
        return False, f"server script missing: {script}"
    worker = _worker_interpreter_missing(entry, portal)
    if worker:
        return False, f"worker interpreter missing: {worker} — the tool env was removed or moved; run `sog-setup`"
    return True, soft


def _config_report(config_path: Path | None, io: PromptIO) -> dict:
    """Count servers/enabled and check each enabled server's wiring. Prints as it goes."""
    if config_path is None:
        io.err("no mcp_config found (looked for --config, SOG_MCP_CONFIG, install/recipes/, agent/MCP_server/)")
        return {
            "path": None,
            "servers": 0,
            "enabled": 0,
            "reachable": 0,
            "unpinned": [],
            "unreachable": [],
            "duplicates": {},
            "stale_tools": [],
            "stale_parameters": [],
            "user_servers": 0,
            "user_enabled": 0,
            "user_reachable": 0,
            "user_unpinned": [],
            "user_unreachable": [],
            "user_shadowed": [],
            "user_skipped": [],
        }
    io.say(f"  config: {config_path}")
    servers = _load_servers(config_path)
    if not servers:
        io.err(f"config has no servers (or failed to parse): {config_path}")
        return {
            "path": str(config_path),
            "servers": 0,
            "enabled": 0,
            "reachable": 0,
            "unpinned": [],
            "unreachable": [],
            "duplicates": {},
            "stale_tools": [],
            "stale_parameters": [],
            "user_servers": 0,
            "user_enabled": 0,
            "user_reachable": 0,
            "user_unpinned": [],
            "user_unreachable": [],
            "user_shadowed": [],
            "user_skipped": [],
        }

    # `.get("enabled", True)` — the merger's default, not a truthy read. mcp_config_merger line 214
    # serves a block whose `enabled` key is absent, and nothing obliges a hand-trimmed or
    # agent-authored config to spell the flag out. Reading it as falsey here would report a served
    # server as "0 enabled", skip its reachability check, and send the user to re-run the wizard on
    # wiring that works.
    enabled = {k: v for k, v in servers.items() if isinstance(v, dict) and v.get("enabled", True)}
    reachable = 0
    unpinned: list[str] = []
    unreachable: list[dict] = []
    for key, entry in sorted(enabled.items()):
        ok, detail = _server_reachable(entry)
        if ok:
            reachable += 1
            # A soft-accepted bare interpreter (resolves on PATH but is NOT an absolute path) is
            # reachable-but-UNPINNED. The thin FastMCP portal only needs the base (agent-core) env —
            # fastmcp + base_mcp — NOT each tool's per-env deps (those load in the worker, dispatched
            # via the server's absolute `env: *_PYTHON`). The risk is activation-dependence: a bare
            # `python` resolves against whatever is first on PATH, so on another box it may land on an
            # interpreter that can't even import fastmcp and the portal yields nothing. `sog-setup`
            # pins command[0] to this device's absolute base-env interpreter (the agent runtime
            # likewise pins bare python to its own sys.executable). Surface it, so a fresh-clone config
            # that points every server at bare `python` is not silently reported as fully wired.
            # (Verdict is unchanged — bare python still counts reachable per the tested soft-accept.)
            cmd = entry.get("command")
            if isinstance(cmd, list) and cmd and not os.path.isabs(str(cmd[0])):
                unpinned.append(key)
        else:
            unreachable.append({"server": key, "detail": detail})
    # A duplicated server key silently deletes a whole block (see _duplicate_server_keys), so
    # `len(servers)` is the *survived* count, not what the file declares. Reporting only the
    # survivors makes a broken config indistinguishable from a smaller one.
    duplicates = _duplicate_server_keys(config_path)
    dropped = sum(n - 1 for n in duplicates.values())
    counted = f"{len(servers)} of {len(servers) + dropped}" if dropped else f"{len(servers)}"
    io.say(f"  {counted} server(s) configured; {len(enabled)} enabled; {reachable} reachable")

    # --- the user-tool layer (mcp_config_user.yaml) ------------------------------------- #
    # Checked with the SAME _server_reachable truth-check as the built-ins: before this,
    # a user-created tool whose env was deleted or whose paths came from another machine
    # sat in the config forever and no diagnostic anywhere would say so. Findings are
    # WARN-only and never move the chain verdict -- the user layer is optional by design.
    user_servers = _load_user_servers()
    user_shadowed = sorted(k for k in user_servers if k in servers)
    # "Original wins" (mcp_config_merger's rule): a user block whose name collides with a
    # canonical server is never served, so it is reported as shadowed, not reachability-checked.
    # The merger also drops a user server whose function NAME a shipped tool, or an earlier user
    # server, already owns. Only the key clash was checked here, so such a server read "enabled
    # reachable" while the agent never served it; the merger's own predicate, accumulated in its
    # order, decides now (hunt 2026-09-30, u37-setup-checks-17).
    shipped_names, shipped_functions = shipped_identity(servers)
    claimed: set[str] = set()
    user_served: dict = {}
    user_enabled: dict = {}
    user_skipped: list[dict] = []
    for key, entry in user_servers.items():
        skip = user_server_skip_reason(key, entry, shipped_names, shipped_functions, claimed)
        if skip is None:
            user_served[key] = user_enabled[key] = entry
            claimed |= user_function_names(entry)
        elif skip[0] == "disabled":
            user_served[key] = entry  # a choice in the file: counted, not served
        elif skip[0] != "server_clash":  # the key clash is reported as shadowed, above
            user_skipped.append({"server": key, "detail": skip[1]})
    user_reachable = 0
    user_unpinned: list[str] = []
    user_unreachable: list[dict] = []
    for key, entry in sorted(user_enabled.items()):
        u_ok, u_detail = _server_reachable(entry)
        if u_ok:
            user_reachable += 1
            u_cmd = entry.get("command")
            if isinstance(u_cmd, list) and u_cmd and not os.path.isabs(str(u_cmd[0])):
                user_unpinned.append(key)
        else:
            user_unreachable.append({"server": key, "detail": u_detail})
    if duplicates:
        io.warn(
            f"    {constants.CHECK_FAIL} {dropped} server block(s) dropped: "
            + ", ".join(f"{key} declared {n}x" for key, n in sorted(duplicates.items()))
            + " — YAML keeps the last block under a repeated key, so the earlier one (its command, "
            "its env wiring and every tool it declares) never reaches any reader. Give each block "
            "its own key, or delete the one that is wrong."
        )
    # The config in force is a copy of the tracked one (see _drifted_servers). Nothing invalidates
    # that copy, so a checkout that has moved on leaves the running system serving old declarations.
    # Two severities, reported as one number and its subset: drift in a description is only age;
    # drift in a parameter's name, `required` flag, `default` or `type` can misdirect a call.
    stale = _stale_tool_blocks(config_path)
    stale_parameters = _stale_parameter_blocks(config_path) if stale else []
    if stale:
        io.warn(
            f"    {len(stale)} server(s) declare tools differently from {constants.original_mcp_config().name}: "
            f"{_first_few(stale)} — this config was generated from that file and copies its tool "
            "declarations verbatim, so a difference means it was written before the current checkout. "
            "Re-run `sog-setup` to regenerate it."
        )
    if stale_parameters:
        io.warn(
            f"    of those, {len(stale_parameters)} declare a parameter differently: "
            f"{_first_few(stale_parameters)} — the model is being offered knobs the portal no longer "
            "accepts, not told about ones it now does, or shown a required flag, default or type "
            "the current checkout has moved."
        )
    for u in unreachable[:8]:
        io.warn(f"    {constants.CHECK_FAIL} {u['server']:24s} {u['detail']}")
    if len(unreachable) > 8:
        io.note(f"    … and {len(unreachable) - 8} more unreachable enabled server(s)")
    if unpinned:
        io.warn(
            f"    {len(unpinned)} enabled server(s) use an unpinned bare interpreter (fresh-clone "
            "default) — the portal launches under whatever `python` is first on PATH; run "
            "`sog-setup` to pin each to this device's base env interpreter"
        )
    if enabled and reachable == 0:
        io.err("every enabled server is unreachable — run `sog-setup` to (re)wire the config")
    if user_servers:
        io.say(
            f"  [USER] {len(user_served)} user tool server(s); {len(user_enabled)} enabled; "
            f"{user_reachable} reachable -- merged at agent launch when SOG_TOOL_CREATION_ENABLED=1"
        )
        if user_shadowed:
            io.warn(
                f"    [USER] {len(user_shadowed)} user server(s) shadowed by a built-in of the same "
                f"name (original wins, never served): {_first_few(user_shadowed)} -- rename the user "
                "server key in mcp_config_user.yaml"
            )
        for u in user_skipped[:8]:
            io.warn(f"    {constants.CHECK_FAIL} [USER] never served: {u['detail']}")
        for u in user_unreachable[:8]:
            io.warn(f"    {constants.CHECK_FAIL} [USER] {u['server']:24s} {u['detail']}")
        if len(user_unreachable) > 8:
            io.note(f"    ... and {len(user_unreachable) - 8} more unreachable user server(s)")
        if user_unpinned:
            io.warn(
                f"    [USER] {len(user_unpinned)} user server(s) use an unpinned bare interpreter -- "
                "the agent runtime pins it at launch, but on another machine the on-disk config may "
                "point nowhere; re-run `sog-setup` to rebase the user config onto this device"
            )
    return {
        "path": str(config_path),
        # `servers` stays the number the system will actually serve — three readers destructure it.
        # `duplicates` is the separate fact that the file asked for more than that.
        "servers": len(servers),
        "enabled": len(enabled),
        "reachable": reachable,
        "unpinned": unpinned,
        "unreachable": unreachable,
        "duplicates": duplicates,
        "stale_tools": stale,
        "stale_parameters": stale_parameters,
        "user_servers": len(user_served),
        "user_enabled": len(user_enabled),
        "user_reachable": user_reachable,
        "user_unpinned": user_unpinned,
        "user_unreachable": user_unreachable,
        "user_shadowed": user_shadowed,
        "user_skipped": user_skipped,
    }


# --------------------------------------------------------------------------- #
# Layer 4 — live MCP portal (opt-in --deep)
# --------------------------------------------------------------------------- #
# Runs the smoke harness from the development tree's ``test/`` directory whatever the caller's cwd.
# ``test/`` is not one of the distribution's packages (nor a package at all -- ``test`` is a stdlib
# name), so its ``smoke`` package is imported top-level with ``test/`` on sys.path; ``python -m
# smoke...`` from any other directory died "No module named 'smoke'" -- which read as "harness
# unavailable" and left the verdict CONNECTED on a box where --deep had checked nothing (hunt
# 2026-09-30, u37-setup-checks-4). The test dir arrives as argv[1] and is popped before the harness
# parses its flags.
_SMOKE_LAUNCH = (
    "import runpy, sys; root = sys.argv.pop(1); sys.path.insert(0, root); "
    "runpy.run_module('smoke.run_smoke_test', run_name='__main__', alter_sys=True)"
)


def _smoke_json(out: str) -> dict:
    """The harness's ``--json`` suite out of its stdout; ``{}`` when there is none.

    The suite is printed with ``indent=2``, so it spans many lines and holds nested objects. Decoding
    from the LAST ``{`` landed on an inner object and failed on every real output, so a FAILING smoke
    parsed as no output at all and was reported "unavailable — skipped" (hunt 2026-09-30,
    u37-setup-checks-3). Decoded from the first ``{`` that opens a whole object instead."""
    decoder = json.JSONDecoder()
    idx = out.find("{")
    while idx != -1:
        try:
            obj, _end = decoder.raw_decode(out[idx:])
        except ValueError:
            obj = None
        if isinstance(obj, dict) and ("summary" in obj or "tools" in obj):
            return obj
        idx = out.find("{", idx + 1)
    return {}


def _deep_portal(conda: Conda, basic_env: str, enabled_keys: list[str], io: PromptIO, *, cap: int = 3) -> dict:
    """Run the smoke harness's static portal checks (levels 1-2) for up to ``cap`` enabled servers.

    Levels 1-2 parse the server files, check their imports and the configured env/interpreter on disk;
    nothing is launched, so this is NOT a handshake and is not reported as one (u37-setup-checks-4).
    The harness reads the repository's ``agent/MCP_server/mcp_config.yaml`` (``test/smoke/registry.CONFIG_PATH``),
    which the report says when the config in force is another file. Degrades to a note, never a
    failure of the whole conncheck, only when the harness is not part of this install (no ``test/``).
    """
    subset = enabled_keys[:cap]
    if not subset:
        io.note("  no enabled servers to portal-test")
        return {"ran": False, "reason": "no enabled servers"}
    tree = constants.dev_test_dir()
    if tree is None:
        io.note(f"  {constants.TEST_TREE_MISSING} — skipped")
        return {"ran": False, "reason": "smoke harness not part of this install"}
    if not (tree / "smoke" / "run_smoke_test.py").is_file():
        io.note(f"  the smoke harness is not part of this install (no smoke/ under {tree}) — skipped")
        return {"ran": False, "reason": "smoke harness not part of this install"}
    io.say(f"  static portal checks (smoke levels 1-2, tracked config) for: {', '.join(subset)}")
    argv = [
        "python",
        "-c",
        _SMOKE_LAUNCH,
        str(tree),
        "--levels",
        "1,2",
        "--tools",
        ",".join(subset),
        "--json",
    ]
    try:
        res = conda.run(basic_env, argv, timeout=600, check=False)
    except CondaError as exc:
        io.note(f"  portal smoke could not launch ({str(exc)[:120]}) — skipped")
        return {"ran": False, "reason": f"conda error: {str(exc)[:160]}"}
    parsed = _smoke_json(res.stdout or "")
    if not parsed:
        # No suite came back. Exit 1 is the harness's own "a check FAILED", so with no parseable suite
        # that is still a failure to report, not a skip; only an explicit import miss of the harness
        # itself means it could not run here.
        err = (res.stderr or "").strip()
        if not res.ok and "No module named 'smoke" in err:
            io.note("  the smoke harness could not be imported in the base env — skipped")
            return {"ran": False, "reason": err.splitlines()[-1][:160] if err else "smoke harness unavailable"}
        detail = (err.splitlines()[-1] if err else f"exit {res.returncode}, no result printed")[:160]
        io.err(f"  static portal checks: FAIL ({detail})")
        return {"ran": True, "passed": False, "servers": subset, "detail": detail}
    summary = parsed.get("summary") if isinstance(parsed.get("summary"), dict) else {}
    try:
        n_fail = int(parsed.get("n_fail") or summary.get("fail") or 0)
    except (TypeError, ValueError):
        n_fail = 1  # a count this report cannot read is not a pass
    passed = res.ok and n_fail == 0
    counts = f"{parsed.get('n_pass', '?')} pass / {parsed.get('n_fail', '?')} fail / {parsed.get('n_warn', '?')} warn"
    (io.ok if passed else io.err)(f"  static portal checks: {'PASS' if passed else 'FAIL'} ({counts})")
    return {"ran": True, "passed": passed, "servers": subset, "counts": counts}


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def conncheck(
    basic_env: str,
    *,
    config_path: str | None = None,
    conda: Conda | None = None,
    io: PromptIO | None = None,
    check_net: bool = True,
    deep: bool = False,
    probe=None,
) -> dict:
    """Build (and print) the connection report for ``basic_env``. Returns the report dict.

    ``probe`` is an injectable ``callable(conda, basic_env) -> {"agent":…, "mcp_client":…}`` so
    tests exercise the layers without spawning conda; it defaults to :func:`_run_import_probe`.
    """
    io = io or PromptIO()
    conda = conda or Conda()
    probe = probe or _run_import_probe

    io.banner("SPATIALOMICSLAB — CONNCHECK", f"connection chain for base env '{basic_env}'")

    # 1) machine preflight (read-only; never prune the user's logs just for inspecting)
    io.section("Machine preflight")
    pf = preflight.run_preflight(check_net=check_net, prune=False)
    for r in pf:
        (io.ok if r.level == _OK else io.warn if r.level == _WARN else io.err)(f"{r.name}: {r.detail}")
    n_pf_fail = sum(1 for r in pf if r.level == _FAIL)

    # 2) base env presence + agent/MCP-client import probe
    try:
        base_exists = conda.env_exists(basic_env)
    except CondaError as exc:
        # First conda call on a cold Conda instance: `_env_map` can raise (TimeoutExpired/OSError) even at
        # check=False. A read-only diagnostic must finish its report — degrade to "base not confirmed" and
        # continue rather than unwinding to cli.main's CondaError→exit 3, which would ALSO misdiagnose a
        # slow/loaded conda as "not on PATH". Mirrors probe()'s CondaError guard above.
        io.warn(f"could not query conda for base env '{basic_env}' ({str(exc)[:120]}) — continuing degraded")
        base_exists = False
    io.section(f"Agent core & MCP client (base env '{basic_env}': {'present' if base_exists else 'ABSENT'})")
    if base_exists:
        probed = probe(conda, basic_env)
    else:
        io.err(f"base env '{basic_env}' does not exist — build it with `sog-setup`")
        probed = {
            "agent": {"spatialomicsgym.agent": "base env absent"},
            "mcp_client": {"nest_asyncio": "base env absent"},
        }
    # `or {}` (not `.get(k, {})`): a probe payload with an explicit null group would return None from
    # .get() and crash the `.items()` loops below. Robust regardless of the probe's source — the
    # injectable `probe=` seam bypasses _run_import_probe's dict-group guard.
    agent_group = probed.get("agent") or {}
    mcp_group = probed.get("mcp_client") or {}
    io.say("  ST-Coscientist agent:")
    for mod, status in agent_group.items():
        (io.ok if status == "ok" else io.err)(
            f"    {constants.CHECK_DONE if status == 'ok' else constants.CHECK_FAIL} {mod}: {status}"
        )
    io.say("  MCP client stack:")
    for mod, status in mcp_group.items():
        (io.ok if status == "ok" else io.err)(
            f"    {constants.CHECK_DONE if status == 'ok' else constants.CHECK_FAIL} {mod}: {status}"
        )
    agent_ok = _group_ok(agent_group)
    mcp_ok = _group_ok(mcp_group)

    # 3) MCP config wiring
    io.section("MCP config wiring")
    cfg = _discover_config(config_path, io=io)
    cfg_report = _config_report(cfg, io)
    enabled_keys = (
        sorted(k for k, v in _load_servers(cfg).items() if isinstance(v, dict) and v.get("enabled", True))
        if cfg
        else []
    )

    # 4) static portal checks (opt-in)
    deep_report = None
    if deep:
        io.section("Portal smoke (--deep, static checks)")
        tracked = constants.original_mcp_config()
        if cfg is not None and os.path.realpath(cfg) != os.path.realpath(tracked):
            io.note(f"  the harness checks the tracked {tracked}, not {cfg}")
        if base_exists and enabled_keys:
            deep_report = _deep_portal(conda, basic_env, enabled_keys, io)
        else:
            io.note("  skipped (no base env or no enabled servers)")
            deep_report = {"ran": False, "reason": "no base env or no enabled servers"}

    # Verdict: the chain is whole only if every load-bearing link holds.
    reachable_ok = cfg_report["reachable"] >= 1
    deep_ok = (deep_report is None) or (not deep_report.get("ran")) or deep_report.get("passed", False)
    ok = base_exists and agent_ok and mcp_ok and reachable_ok and n_pf_fail == 0 and deep_ok

    io.section("Summary")
    io.say(f"  agent import      : {'OK' if agent_ok else 'FAILED'}")
    io.say(f"  MCP client import : {'OK' if mcp_ok else 'FAILED'}")
    io.say(f"  servers reachable : {cfg_report['reachable']}/{cfg_report['enabled']} enabled")
    if cfg_report.get("user_servers"):
        io.say(
            f"  user tool servers : {cfg_report['user_reachable']}/{cfg_report['user_enabled']} "
            "enabled reachable (informational -- not part of the chain verdict)"
        )
    if deep_report is not None:
        # The --deep outcome belongs in the summary the user reads last, not only in its own section.
        if deep_report.get("ran"):
            io.say(f"  portal smoke      : {'PASS' if deep_report.get('passed') else 'FAIL'}")
        else:
            io.say(f"  portal smoke      : NOT RUN ({deep_report.get('reason', 'skipped')})")
    if n_pf_fail:
        io.warn(f"  preflight has {n_pf_fail} hard failure(s) — see above")
    (io.ok if ok else io.err)(f"  connection chain: {'CONNECTED' if ok else 'BROKEN — see the failed link(s) above'}")

    return {
        "basic_env": basic_env,
        "base_env_present": base_exists,
        "preflight_failures": n_pf_fail,
        "agent": agent_group,
        "agent_ok": agent_ok,
        "mcp_client": mcp_group,
        "mcp_ok": mcp_ok,
        "config": cfg_report,
        "deep": deep_report,
        "ok": ok,
    }


# --------------------------------------------------------------------------- #
# CLI: sog-setup conncheck
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="sog-setup conncheck",
        description="Verify the base env <-> MCP <-> ST-Coscientist connection chain (read-only, no API key).",
    )
    ap.add_argument("--base", help="basic env name (defaults to the last run's, from state)")
    ap.add_argument("--config", help="mcp_config path (defaults to SOG_MCP_CONFIG / install/recipes / canonical)")
    ap.add_argument("--no-net", action="store_true", help="skip the network preflight probe")
    ap.add_argument("--deep", action="store_true", help="also launch the real MCP portal for a few servers")
    ap.add_argument("--json", action="store_true", help="print the report as JSON (in addition to the table)")
    args = ap.parse_args(argv)

    basic_env = args.base
    if not basic_env:
        st = SetupState.load()
        if st and st.basic_env_name:
            basic_env = st.basic_env_name
    if not basic_env:
        print("no base env given and no saved state found — pass --base <name>")
        return 2

    report = conncheck(
        basic_env,
        config_path=args.config,
        check_net=not args.no_net,
        deep=args.deep,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    # Route the direct ``python -m sog_install.conncheck`` entry through the same friendly
    # cli wrapper as ``sog-setup conncheck`` (KeyboardInterrupt→130, CondaError→3, else a one-liner+1).
    import sys

    from .cli import main as _cli_main

    raise SystemExit(_cli_main(["conncheck", *sys.argv[1:]]))
