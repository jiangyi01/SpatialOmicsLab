"""MCP config merger: safely merge original + user tool configs.

This module merges the original mcp_config.yaml with user-created tools
from mcp_config_user.yaml into a single temp file for add_mcp().

Safety guarantees:
  - Original mcp_config.yaml is NEVER modified
  - If user config is corrupt, original loads fine (warning only)
  - If merge fails for ANY reason, returns original config path
  - Name conflicts: original always wins, user tool skipped
  - User tools tagged [USER] in description for ToolRetriever distinction
"""

from __future__ import annotations

import atexit
import hashlib
import os
import re
import tempfile
from pathlib import Path

from spatialomicsgym.mcp_config_path import CANONICAL_CONFIG_DEFAULT

# The predicates live in the light module so the web portal -- which must not import the agent
# package -- decides "which created servers are wired" by the same rule this merge applies.
from spatialomicsgym.mcp_user_config import (  # noqa: F401
    DEFAULT_USER_CONFIG,
    USER_CONFIG_ENV,
    _server_tools,
    disabled_user_servers,
    resolve_user_config_path,
    shipped_identity,
    user_function_names,
    user_server_skip_reason,
)

# `_REPO_ROOT` is deliberately NOT re-exported. It moved with the resolver, and a re-exported
# constant can still be imported and patched -- while the patch does nothing, because the
# resolver reads its own module's global. A silent no-op in a fixture is worse than the
# AttributeError a caller gets for reaching at the wrong module.


def _looks_like_a_server_block(value: object) -> bool:
    """True only for a mapping that unambiguously describes an MCP server.

    Deliberately strict: both a ``command`` and a ``tools`` list. Bookkeeping keys a config may
    legitimately carry at the top level (``version``, free-text notes, a half-written block with a
    command but no tools) must not be dragged into the wiring.
    """
    return isinstance(value, dict) and "command" in value and isinstance(value.get("tools"), list)


def _recover_top_level_servers(user: dict, user_servers: dict) -> dict:
    """Return the server blocks the agent wrote as SIBLINGS of ``mcp_servers`` instead of children.

    ``mcp_config_user.yaml`` is written by the agent during tool creation, not by hand, and a model
    that emits ``mcp_servers: {}`` followed by a top-level ``user_<id>:`` block produces a file that
    every reader here parses as "no user tools". The tool then exists on disk, reports created, and
    is never callable -- with no warning anywhere, because an empty ``mcp_servers`` is also what a
    genuinely empty config looks like. Recovering the block keeps the created tool usable; a
    correctly nested entry of the same name always wins, so this can only ever add.
    """
    return {
        key: value
        for key, value in user.items()
        if key != "mcp_servers" and key not in user_servers and _looks_like_a_server_block(value)
    }


def normalize_user_config(user_path: str = DEFAULT_USER_CONFIG) -> int:
    """Move server blocks written beside ``mcp_servers`` into it. Returns how many moved.

    :func:`_recover_top_level_servers` already keeps such a tool *callable*, and prints a warning
    saying to move it by hand. That warning goes to the server log, nobody moves it, and the file
    stays wrong forever -- which is where this stops being a cosmetic nit and becomes a real
    divergence: the merger recovers the block, so the agent can call the tool, while every other
    reader parses the same file as "no user tools".

    Measured on this box: a ``liana_run`` that worked in chat and was, at the same time, absent
    from the tool manager, uncounted by the header chip, a 404 on its own detail route,
    un-deletable by every code path, and invisible to the resync pruner -- the proven "merger says
    ['user_liana'], pruner says set()" split.

    Recovering on read cannot fix that, because every reader would have to learn the trick.
    Writing the file back in the shape the schema expects fixes it once, for all of them.

    No-op on a correct config. Never raises -- a repair that cannot be written leaves the file
    alone and the existing recovery still wires the tool. A correctly nested entry of the same
    name always wins, so this can only move a block nothing else claimed.
    """
    import yaml

    path = Path(user_path)
    try:
        user = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return 0
    if not isinstance(user, dict):
        return 0

    nested = user.get("mcp_servers")
    nested = dict(nested) if isinstance(nested, dict) else {}
    stray = _recover_top_level_servers(user, nested)
    if not stray:
        return 0

    nested.update(stray)
    rest = {k: v for k, v in user.items() if k != "mcp_servers" and k not in stray}
    fixed = {"mcp_servers": nested, **rest}

    # Atomic, like every other config write here: a half-written wiring file is worse than a
    # mis-nested one.
    tmp = path.with_name(path.name + ".partial")
    try:
        tmp.write_text(yaml.safe_dump(fixed, sort_keys=False), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return 0
    # Says "top level" deliberately: `test_a_server_the_agent_wrote_at_the_top_level_is_recovered_
    # and_announced` pins that this situation is announced by name and by what was wrong with it,
    # and that property is worth keeping now that the news is better than a warning.
    print(
        f"Repaired '{user_path}': moved {len(stray)} server block"
        f"{'' if len(stray) == 1 else 's'} from the TOP LEVEL into 'mcp_servers' "
        f"({', '.join(sorted(stray))}). There only the merger could see them."
    )
    return len(stray)


_CLEANUP_REGISTERED: set[str] = set()
_MERGED_NAME = re.compile(r"^spatialomicsgym_mcp_merged_(\d+)_[0-9a-f]{16}\.yaml$")


def _sweep_orphaned_merges() -> None:
    """Remove merged configs whose process is gone. Never raises.

    atexit does not run for a process that was killed, so each SIGKILL or timeout left its file in
    the temp dir for good (u14-mcp-wiring-18). Only this user's files, and only when the PID is
    certainly dead: ``kill(pid, 0)`` answering ProcessLookupError.
    """
    try:
        entries = list(os.scandir(tempfile.gettempdir()))
    except OSError:
        return
    me = os.getuid() if hasattr(os, "getuid") else None
    for entry in entries:
        found = _MERGED_NAME.match(entry.name)
        if not found:
            continue
        try:
            if me is not None and entry.stat(follow_symlinks=False).st_uid != me:
                continue
            os.kill(int(found.group(1)), 0)
        except ProcessLookupError:
            _best_effort_unlink(entry.path)
        except (OSError, ValueError):
            continue


def _best_effort_unlink(path: str) -> None:
    """Remove a temp file if it still exists; never raise (runs at interpreter exit)."""
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


def build_merged_mcp_config(
    original_path: str = CANONICAL_CONFIG_DEFAULT,
    user_path: str = DEFAULT_USER_CONFIG,
    merge_user: bool = False,
) -> str:
    """Merge original + user MCP configs into a single temp file.

    Rules:
    - Original servers always included (priority)
    - User servers only if merge_user=True and user config exists
    - Name conflicts: original wins, user tool skipped with warning
    - User tools tagged [USER] in description
    - Merged file written to a per-process temp path (not in repo, regenerated)
    - Original config NEVER modified

    Args:
        original_path: Path to the original mcp_config.yaml (default: the checkout's own
            ``agent/MCP_server/mcp_config.yaml``, absolute)
        user_path: Path to the user-created mcp_config_user.yaml (the default spelling is resolved
            by ``resolve_user_config_path``, not read against the working directory)
        merge_user: Whether to merge user tools (requires tool_creation_enabled)

    Returns:
        Path to the config file to load (merged or original)

    Raises:
        Never raises — falls back to original_path on any error
    """
    import yaml

    if user_path == DEFAULT_USER_CONFIG:
        # The bare relative spelling names the agent part's file, wherever the process stands.
        user_path = resolve_user_config_path(user_path)

    # Always load original — but honor the documented "Never raises" contract: an
    # unreadable / non-UTF-8 / malformed base config degrades to returning the original
    # path, and a top-level non-mapping YAML is treated as empty rather than crashing at
    # ``.get``.
    try:
        original_content = Path(original_path).read_text(encoding="utf-8")
        loaded = yaml.safe_load(original_content)
    except Exception as e:
        print(f"WARNING: Could not read base MCP config '{original_path}': {e}")
        return str(original_path)
    original = loaded if isinstance(loaded, dict) else {}
    # A hand-edited base config can write `mcp_servers:` as null/list/scalar (e.g. every server
    # commented out on a fresh clone leaves the key present but null). `.get(..., {})` only
    # supplies the default when the key is ABSENT, so `dict(None)`/`dict([...])` would raise here
    # -- OUTSIDE the try above -- breaking this module's documented "Never raises" contract and
    # the graceful original-path fallback its callers rely on. Treat any non-mapping value as
    # empty, exactly as the user-server branch below (and wiring.py/categories.py/conncheck.py) do.
    original_servers = original.get("mcp_servers", {})
    if not isinstance(original_servers, dict):
        print(f"WARNING: base MCP config '{original_path}' has a non-mapping 'mcp_servers'; treating as empty.")
        original_servers = {}
    merged_servers = dict(original_servers)
    # Server names and spatialomicsgym_names the shipped catalogue owns, for conflict checking.
    # Snapshotted before the merge loop, and through the same helper the resync pruner uses.
    original_names, original_function_names = shipped_identity(original_servers)
    # Created servers the portal turned off for this session (see ``DISABLED_USER_SERVERS_KEY``).
    turned_off = disabled_user_servers(original)

    # Only merge user tools if explicitly enabled and file exists
    user_tools_merged = 0
    user_tools_skipped = 0

    if merge_user and Path(user_path).exists():
        try:
            # Repair before reading, so the tool manager, the pruner and the header count all see
            # the same servers this merge is about to wire. The recovery below stays as the net
            # for a file that could not be rewritten.
            normalize_user_config(user_path)
            user_content = Path(user_path).read_text(encoding="utf-8")
            user = yaml.safe_load(user_content) or {}
            user_servers = user.get("mcp_servers", {})

            if not isinstance(user_servers, dict):
                print("WARNING: mcp_config_user.yaml has invalid mcp_servers format. Skipping.")
                user_servers = {}

            if isinstance(user, dict):
                stray = _recover_top_level_servers(user, user_servers)
                if stray:
                    print(
                        f"WARNING: {len(stray)} user server(s) are at the TOP LEVEL of "
                        f"'{user_path}' instead of nested under 'mcp_servers': "
                        f"{', '.join(sorted(stray))}. Wiring them anyway; move them under "
                        f"'mcp_servers:' to silence this."
                    )
                    user_servers = {**stray, **user_servers}

            # Names taken so far, so a second user server cannot silently shadow the first.
            # Seeded empty and grown below; the shipped names are a separate argument because
            # their clash has a different remedy and a different message.
            claimed_by_user: set[str] = set()
            for name, meta in user_servers.items():
                # ONE predicate, shared with tool_management's resync pruner -- see
                # `user_server_skip_reason`. A disabled server is a choice and is passed over in
                # silence; the rest mean the file is wrong and are counted and announced.
                if name in turned_off:
                    continue  # a choice, like ``enabled: false`` in the file: passed over in silence
                skip = user_server_skip_reason(name, meta, original_names, original_function_names, claimed_by_user)
                if skip is not None:
                    code, message = skip
                    if code != "disabled":
                        print(f"WARNING: {message}")
                        user_tools_skipped += 1
                    continue

                # Tag as user tool (skip if already tagged). Guard the type: an LLM-authored
                # mcp_config_user.yaml may carry a null/non-string `description`, and a bare
                # `.startswith` would raise AttributeError -- caught by the broad `except` below,
                # which would silently discard THIS tool and every not-yet-processed user tool.
                # An isinstance guard keeps the tool (untagged) and is a no-op for real strings.
                if isinstance(meta.get("description"), str) and not meta["description"].startswith("[USER]"):
                    meta["description"] = f"[USER] {meta['description']}"
                for tool in _server_tools(meta):
                    if isinstance(tool, dict) and isinstance(tool.get("description"), str):
                        if not tool["description"].startswith("[USER"):
                            tool["description"] = f"[USER TOOL] {tool['description']}"

                merged_servers[name] = meta
                claimed_by_user |= user_function_names(meta)
                user_tools_merged += 1

        except yaml.YAMLError as e:
            print(f"WARNING: mcp_config_user.yaml has YAML syntax error: {e}")
            print("Continuing with original tools only.")
        except Exception as e:
            print(f"WARNING: Failed to load user MCP config: {e}")
            print("Continuing with original tools only.")

    if user_tools_merged > 0 or user_tools_skipped > 0:
        print(f"MCP config merge: {user_tools_merged} user tools loaded, {user_tools_skipped} skipped")

    if not merge_user and isinstance(loaded, dict) and isinstance(loaded.get("mcp_servers", {}), dict):
        # Nothing was merged, so the "merged" file would be a copy of the original: written to /tmp
        # on every wire (the eval default) and left behind by every process killed before atexit
        # ran -- 654 of them, 53 MB, were found (u14-mcp-wiring-18). A malformed base still gets
        # the degraded copy below, whose empty `mcp_servers` is the contract for that case.
        return str(original_path)

    # Write merged config to a per-process temp file. A fixed /tmp name races across
    # concurrent agents (and collides on multi-user boxes) — key it by PID + a digest of
    # the content so parallel runs and distinct configs never clobber each other.
    merged = {"mcp_servers": merged_servers}
    merged_yaml = yaml.dump(merged, default_flow_style=False, allow_unicode=True)
    digest = hashlib.sha256(merged_yaml.encode("utf-8")).hexdigest()[:16]
    merged_path = str(Path(tempfile.gettempdir()) / f"spatialomicsgym_mcp_merged_{os.getpid()}_{digest}.yaml")
    _sweep_orphaned_merges()
    try:
        Path(merged_path).write_text(merged_yaml, encoding="utf-8")
    except Exception as e:
        print(f"WARNING: Failed to write merged config to {merged_path}: {e}")
        print("Falling back to original config.")
        return str(original_path)

    # The merged config is needed for the whole agent lifetime, so clean it up at interpreter
    # exit rather than now. Without this, every agent process (new PID) and every config change
    # (new digest) leaves another merged YAML behind in the temp dir.
    if merged_path not in _CLEANUP_REGISTERED:  # one handler per file, not one per wire
        _CLEANUP_REGISTERED.add(merged_path)
        atexit.register(_best_effort_unlink, merged_path)

    return merged_path
