"""Two-stage trash box system for user-created MCP tools.

Provides a desktop-recycle-bin-like system:
  - trash_tool()          : Move tool to trash (reversible, files physically moved)
  - restore_tool()        : Restore tool from trash back to active
  - permanent_delete()    : Irreversibly remove trashed tool (conda env + files + log)
  - preview_permanent_delete() : Preview what will be removed (no side effects)
  - list_user_tools()     : List tools by status
  - trash_info()          : Disk usage summary
  - check_tool_id_available() : Pre-creation conflict check

Safety model (10 layers):
  1. Know-how gate (tool_creation_enabled)
  2. TOOL_ID_RE regex (path traversal prevention)
  3. install_log lookup (only known tools)
  4. Status gate (permanent delete only from "trashed")
  5. PROTECTED_FILES frozenset
  6. PROTECTED_DIRS frozenset
  7. Resolved path boundary check (symlink attack prevention)
  8. File collision check on restore
  9. MCP config conflict check on restore
  10. User confirmation (handled by know-how doc)

All JSON/YAML writes use atomic temp-file-then-rename pattern.
All operations serialized via fcntl advisory lock.

NOTE: .trash/ directory is covered by root .gitignore pattern '.*'
"""

from __future__ import annotations

import copy
import fcntl
import json
import logging
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import yaml

from sog_install.constants import PROTECTED_ENVS as _SETUP_PROTECTED_ENVS
from sog_install.constants import conda_envs_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOLS_USER_DIR = Path(__file__).resolve().parent
TRASH_DIR = TOOLS_USER_DIR / ".trash"
INSTALL_LOG = TOOLS_USER_DIR / "install_log.json"
MCP_CONFIG_USER = Path(__file__).resolve().parent.parent / "MCP_server" / "mcp_config_user.yaml"
#: What ``MCP_CONFIG_USER`` was at import; see ``mcp_config_user_path``.
_DEFAULT_MCP_CONFIG_USER = MCP_CONFIG_USER


def mcp_config_user_path() -> Path:
    """The ``mcp_config_user.yaml`` this module reads and writes, resolved per call.

    ``knowledge_manager.mcp_config_user_path``'s rule, restated here because the two modules keep
    their helpers apart: ``SOG_MCP_USER_CONFIG`` when set, as every reader of the file already does.
    Trashing a tool removed it from the default file while the merger served the override, so the
    trashed tool stayed wired (hunt 2026-09-30, u14-mcp-wiring-7). An assignment to
    ``MCP_CONFIG_USER`` still wins: the more specific instruction, and the seam the tests use.
    """
    if MCP_CONFIG_USER != _DEFAULT_MCP_CONFIG_USER:
        return Path(MCP_CONFIG_USER)
    from spatialomicsgym.mcp_user_config import user_config_override

    override = user_config_override()
    return override if override is not None else MCP_CONFIG_USER


# Derived from the interpreter this runs under, never the /opt/conda of the machine it was written
# on: a miss here is silent and reads as success -- _remove_conda_env returns "skipped" without ever
# calling `conda remove` while permanent_delete reports the space freed, restore_tool refuses a
# restorable tool, and every size total reads 0. Stays a module-level Path because a couple of dozen
# test modules redirect it with monkeypatch.setattr(<module>, "CONDA_ENVS_DIR", tmp_path / "envs").
CONDA_ENVS_DIR = Path(conda_envs_root())

# tool_id must be: lowercase letter, then up to 62 lowercase letters/digits/underscores
TOOL_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")

# Files that must NEVER be moved, deleted, or modified by trash operations
PROTECTED_FILES = frozenset(
    {
        TOOLS_USER_DIR / "base_mcp.py",
        TOOLS_USER_DIR / "worker_utils.py",
        TOOLS_USER_DIR / "user_skill.py",
        TOOLS_USER_DIR / "trash_manager.py",
        TOOLS_USER_DIR / "knowledge_manager.py",
        INSTALL_LOG,
    }
)

# Directories that must NEVER be touched
PROTECTED_DIRS = frozenset(
    {
        TOOLS_USER_DIR / "repos",
        TOOLS_USER_DIR / "skill_entries",
        TOOLS_USER_DIR / "__pycache__",
        TOOLS_USER_DIR / ".knowledge",
    }
)

# Known file suffixes for user tool files
KNOWN_SUFFIXES = ("_worker.py", "_mcp_server.py", "_env.yaml", "_worker.R")

# Conda environments that must NEVER be removed. Single-sourced from sog_install.constants -- a local
# copy here once drifted (it lacked the pre-rename alias env, ``constants.LEGACY_ENV_ALIASES``'s
# value, which the setup side protects), and two lists of "never delete" that disagree means one
# door is unguarded. Kept as a module attribute so tests can monkeypatch it.
PROTECTED_ENVS = _SETUP_PROTECTED_ENVS

# Trash metadata version for forward compatibility
TRASH_META_VERSION = 1

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TrashError(Exception):
    """Base exception for all trash operations."""


class TrashSafetyError(TrashError):
    """Raised when a safety check fails (path/whitelist violation)."""


class TrashStateError(TrashError):
    """Raised when tool is in wrong state for the requested operation."""


class TrashNotFoundError(TrashError):
    """Raised when the requested tool does not exist."""


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


@contextmanager
def _trash_lock():
    """Advisory file lock to serialize all trash operations.

    Uses LOCK_NB (non-blocking) so concurrent attempts get a clear error
    instead of hanging indefinitely.
    """
    lock_path = TOOLS_USER_DIR / ".trash.lock"
    lock_fd = None
    try:
        lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            lock_fd.close()
            raise TrashError("Another trash operation is in progress. Please wait and retry.") from exc
        yield
    finally:
        if lock_fd is not None and not lock_fd.closed:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()


# ---------------------------------------------------------------------------
# Atomic I/O helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically via temp-file-then-rename."""
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        os.replace(str(tmp), str(path))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write_yaml(path: Path, data: dict) -> None:
    """Write YAML atomically via temp-file-then-rename."""
    tmp = path.with_suffix(".yaml.tmp")
    try:
        tmp.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
        os.replace(str(tmp), str(path))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Install log helpers
# ---------------------------------------------------------------------------


def _log_key(entry: dict) -> tuple[str, str]:
    """What makes a registry row unique: its owner and its tool id.

    The same key ``knowledge_manager._log_key`` uses, and deliberately a second copy rather than
    an import -- these two modules mirror each other's install_log helpers on purpose (see the
    header of that section there). What is NOT optional is that they agree: this module writes the
    file that one reads. Keyed on the id alone, two accounts who both create ``liana`` collapse
    into one row and the second silently replaces the first, so the registry describes one tool
    while two sit on disk.

    Backward compatible by construction. No record written before owners existed carries one, so
    they all key on ``("", tool_id)`` and the dedup is byte-identical to what it was.
    """
    return (str(entry.get("owner") or ""), str(entry.get("tool_id") or ""))


def _read_install_log() -> list[dict]:
    """Read install_log.json with dedup by (owner, tool_id) -- last entry wins.

    Safe against: missing file, malformed JSON, non-list root, missing fields.
    """
    if not INSTALL_LOG.exists():
        return []
    try:
        raw = json.loads(INSTALL_LOG.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, list):
        return []
    seen: dict[tuple[str, str], dict] = {}
    for entry in raw:
        if isinstance(entry, dict) and "tool_id" in entry:
            seen[_log_key(entry)] = entry
    return list(seen.values())


def _write_install_log(entries: list[dict]) -> None:
    """Write install_log.json atomically."""
    _atomic_write_json(INSTALL_LOG, entries)


def _find_log_entry(tool_id: str, owner: str | None = None) -> dict | None:
    """One install_log entry.

    ``owner=None`` is "whoever's it is" -- today's behaviour and every existing caller's, because
    until accounts reach this layer there is exactly one. Passing an owner narrows and never
    widens: a caller that knows whose tool it is must not be handed somebody else's row just
    because the ids match. Same contract as ``knowledge_manager._find_log_entry``.
    """
    for entry in _read_install_log():
        if entry.get("tool_id") != tool_id:
            continue
        if owner is not None and str(entry.get("owner") or "") != str(owner):
            continue
        return entry
    return None


def _matching_rows(entries: list[dict], tool_id: str, owner: str | None) -> list[dict]:
    """The rows a write addressed at ``(tool_id, owner)`` is allowed to touch.

    ``owner=None`` means the caller does not know whose row it is. With one row that is fine and
    is what every existing caller does; with two it is not answerable, and the callers below say
    so rather than picking one.
    """
    return [
        e for e in entries if e.get("tool_id") == tool_id and (owner is None or str(e.get("owner") or "") == str(owner))
    ]


def _ambiguous(rows: list[dict], tool_id: str, verb: str) -> str | None:
    """Why a write must not proceed, or None. See ``_matching_rows``."""
    if len(rows) <= 1:
        return None
    owners = sorted(str(r.get("owner") or "") or "(no owner)" for r in rows)
    return (
        f"{len(rows)} accounts have a tool called '{tool_id}' ({', '.join(owners)}), so {verb} "
        f"without naming an owner would have picked one of them at random. Pass owner=."
    )


def _install_log_unreadable() -> str | None:
    """Why install_log.json must not be rewritten from what we just read, or None if it may be.

    ``_read_install_log`` fails open -- a truncated, empty or non-list file reads as "no tools are
    installed" -- and that is the right answer for a reader. For a writer it is not: saving that
    answer back replaces the registry with ``[]`` and every created tool stops existing, from a
    file whose remaining bytes were the only record of them. Trashing or restoring one tool is
    enough to trigger it, because both end in a status update.
    """
    if not INSTALL_LOG.exists():
        return None
    try:
        raw = json.loads(INSTALL_LOG.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return f"install_log.json at {INSTALL_LOG} could not be read ({e})"
    if not isinstance(raw, list):
        return f"install_log.json at {INSTALL_LOG} holds a {type(raw).__name__}, not a list of entries"
    return None


def _update_log_entry(tool_id: str, updates: dict, owner: str | None = None) -> str | None:
    """Update fields on a single install_log entry (atomic).

    Returns None when the entry was updated, or why it was not. Callers that cannot act on the
    answer may ignore it; what they must not do is report the update as done.

    ``owner`` addresses one account's row. The callers in this module all hold the entry they
    resolved, so they pass its owner and the row they update is the row they read -- without that
    they would update whichever row happened to come first, which with two accounts is a coin
    flip. Left unset (the default, and every caller outside this module) the old behaviour
    stands while there is one row, and refuses rather than guessing when there are two.
    """
    unreadable = _install_log_unreadable()
    if unreadable:
        logger.warning("Refusing to rewrite the tool registry: %s", unreadable)
        return (
            f"{unreadable}, so '{tool_id}' was left as it is rather than rewriting the registry "
            f"from a file that did not parse. Repair or remove that file to record the change."
        )
    entries = _read_install_log()
    rows = _matching_rows(entries, tool_id, owner)
    if not rows:
        # Nothing matched, so the old code wrote the list back unchanged and returned as if the
        # update had landed. Say it did not: the fields are lost either way.
        return f"No install_log entry for '{tool_id}', so {sorted(updates)} was not recorded."
    ambiguous = _ambiguous(rows, tool_id, f"recording {sorted(updates)}")
    if ambiguous:
        logger.warning("Refusing an ambiguous registry update: %s", ambiguous)
        return ambiguous
    rows[0].update(updates)
    _write_install_log(entries)
    return None


def _remove_log_entry(tool_id: str, owner: str | None = None) -> str | None:
    """Remove ONE tool's entry from install_log (atomic).

    Returns None on success, or why the entry is still there. Filtering an empty list produces an
    empty list, so on an unreadable registry this path removed everything rather than one tool.

    The filter is on the whole key, not the id: with two accounts holding the same name, dropping
    every row whose ``tool_id`` matches deletes the other account's tool along with this one --
    from a permanent delete, which has nothing to restore it from.
    """
    unreadable = _install_log_unreadable()
    if unreadable:
        logger.warning("Refusing to rewrite the tool registry: %s", unreadable)
        return (
            f"{unreadable}, so '{tool_id}' was not removed. Deleting every other tool's entry is "
            f"not a better outcome than leaving this one in place."
        )
    entries = _read_install_log()
    rows = _matching_rows(entries, tool_id, owner)
    ambiguous = _ambiguous(rows, tool_id, f"removing '{tool_id}'")
    if ambiguous:
        logger.warning("Refusing an ambiguous registry removal: %s", ambiguous)
        return ambiguous
    doomed = {id(r) for r in rows}
    _write_install_log([e for e in entries if id(e) not in doomed])
    return None


# ---------------------------------------------------------------------------
# MCP config helpers
# ---------------------------------------------------------------------------


def _read_mcp_config_user() -> dict:
    """Read mcp_config_user.yaml safely. Returns {} if missing/corrupt."""
    path = mcp_config_user_path()
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _server_block_paths(config: dict, server_key: str) -> list[list[str]]:
    """EVERY place ``server_key`` is wired, in the order the merger resolves them.

    Usually one. A rollback that restored a snapshot taken off a stray used to publish the canonical
    nested block and leave the column-0 copy in place, and the two live on happily -- the merger
    prefers nested (``{**stray, **user_servers}``) -- until a removal takes the first location and
    the survivor keeps advertising a tool whose server file is now in ``.trash/``. Anything that
    unwires a tool has to name all of them.

    The top-level test is the merger's own strictness (a ``command`` and a ``tools`` list), so a
    bookkeeping key that happens to share the name is never removed as if it were wiring.
    """
    paths = []
    servers = config.get("mcp_servers")
    if isinstance(servers, dict) and isinstance(servers.get(server_key), dict):
        paths.append(["mcp_servers", server_key])
    block = config.get(server_key)
    if isinstance(block, dict) and "command" in block and isinstance(block.get("tools"), list):
        paths.append([server_key])
    return paths


def _server_block_path(config: dict, server_key: str) -> list[str] | None:
    """Where ``server_key`` actually lives in the config, or None if it is not there.

    mcp_config_user.yaml is agent-authored, and a model that emits ``mcp_servers: {}`` and then
    appends ``user_<id>:`` at column 0 produces a file the agent still runs from:
    ``mcp_config_merger._recover_top_level_servers`` rescues the misplaced sibling so the created
    tool stays callable. Indexing ``mcp_servers`` straight makes this module blind to a block the
    agent is actively serving -- trashing the tool then leaves it in place, and the merger keeps
    advertising a tool whose server file has just moved to .trash/.

    A correctly nested block always wins -- this is the one the merger serves, so it is the one a
    single-answer lookup (``_get_mcp_server_entry``, the raw-backup reader) must report. Removal
    goes through ``_server_block_paths`` instead: leaving a second copy behind is what put a trashed
    tool back on the model's tool list.
    """
    paths = _server_block_paths(config, server_key)
    return paths[0] if paths else None


def _get_mcp_server_entry(server_key: str) -> dict | None:
    """Get a single MCP server entry by key, wherever the agent wrote it."""
    config = _read_mcp_config_user()
    key_path = _server_block_path(config, server_key)
    if key_path is None:
        return None
    return config["mcp_servers"][server_key] if len(key_path) == 2 else config[server_key]


def _remove_from_mcp_config(server_key: str) -> None:
    """Remove a server entry from mcp_config_user.yaml.

    Safe against missing file, missing key, empty config.
    Always preserves valid YAML with mcp_servers key.
    Uses ruamel.yaml for round-trip formatting preservation.

    Removes the key from EVERY location it is wired in, not just the one a lookup resolves to: a
    tool can be nested and stray at once, and unwiring half of it leaves the merger serving a tool
    whose files have just moved to .trash/.
    """
    config_path = mcp_config_user_path()
    if not config_path.exists():
        return
    for key_path in _server_block_paths(_read_mcp_config_user(), server_key):
        # Use ruamel.yaml for round-trip-safe write (preserves quote styles). One atomic write per
        # location -- each re-reads the file, so a crash between them leaves fewer copies, never a
        # damaged one.
        _roundtrip_remove_key(config_path, key_path)


def _restore_to_mcp_config(server_key: str, config_block: dict) -> None:
    """Restore a server entry to mcp_config_user.yaml.

    Raises TrashStateError if the key already exists (conflict).
    Creates the file with proper structure if it doesn't exist.
    Uses ruamel.yaml for round-trip formatting preservation.
    """
    config = _read_mcp_config_user()
    servers = config.get("mcp_servers", {})
    if not isinstance(servers, dict):
        servers = {}
    if server_key in servers:
        raise TrashStateError(
            f"MCP config already has entry '{server_key}'. Cannot restore - another tool may have taken this name."
        )
    # Use ruamel.yaml for round-trip-safe write (preserves quote styles)
    _roundtrip_add_key(mcp_config_user_path(), ["mcp_servers"], server_key, config_block)


def _roundtrip_remove_key(yaml_path: Path, key_path: list[str]) -> None:
    """Remove a nested key from YAML while preserving formatting.

    Uses ruamel.yaml for round-trip read/write to keep original quote styles,
    comments, and formatting intact. Falls back to PyYAML if ruamel unavailable.
    """
    try:
        from ruamel.yaml import YAML

        rt = YAML()
        rt.preserve_quotes = True
        data = rt.load(yaml_path.read_text())
        if data is None:
            return
        # Navigate to parent, remove final key
        node = data
        for k in key_path[:-1]:
            if not isinstance(node, dict) or k not in node:
                return
            node = node[k]
        if isinstance(node, dict):
            node.pop(key_path[-1], None)
        tmp = yaml_path.with_suffix(".yaml.tmp")
        try:
            with open(tmp, "w") as f:
                rt.dump(data, f)
            os.replace(str(tmp), str(yaml_path))
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    except ImportError:
        # Fallback: plain PyYAML (may alter formatting)
        config = _read_mcp_config_user()
        node = config
        for k in key_path[:-1]:
            node = node.get(k, {})
        if isinstance(node, dict):
            node.pop(key_path[-1], None)
        _atomic_write_yaml(yaml_path, config)


def _roundtrip_add_key(yaml_path: Path, parent_path: list[str], key: str, value: dict) -> None:
    """Add a nested key to YAML while preserving formatting.

    Uses ruamel.yaml for round-trip read/write. Falls back to PyYAML if unavailable.
    """
    try:
        from ruamel.yaml import YAML

        rt = YAML()
        rt.preserve_quotes = True
        if yaml_path.exists():
            data = rt.load(yaml_path.read_text())
        else:
            data = {}
        if data is None:
            data = {}
        # Navigate to parent
        node = data
        for k in parent_path:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[key] = value
        tmp = yaml_path.with_suffix(".yaml.tmp")
        try:
            with open(tmp, "w") as f:
                rt.dump(data, f)
            os.replace(str(tmp), str(yaml_path))
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    except ImportError:
        # Fallback: plain PyYAML (may alter formatting)
        config = _read_mcp_config_user()
        node = config
        for k in parent_path:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[key] = value
        _atomic_write_yaml(yaml_path, config)


def _restore_raw_mcp_yaml(raw_yaml: str, server_key: str) -> bool:
    """Restore ONLY this tool's server entry to mcp_config_user.yaml.

    CRITICAL: this function must NOT overwrite the entire config file with
    `raw_yaml`. Past bug: the raw backup captured the FULL config snapshot
    at trash-time (including every OTHER server active at that moment), so
    overwriting would resurrect stale entries for tools that have since
    been trashed themselves — causing "MCP config already has entry
    user_X" errors on every subsequent restore.

    The correct behaviour is: extract the ONE `server_key` entry from the
    raw backup and merge it into the CURRENT config via the structured
    add-key path. All other current entries are preserved untouched.

    Returns True when the entry was written back, False when the raw backup could not supply it
    (unparseable, or carrying no block for ``server_key``). The caller needs that answer: the
    metadata carries a second, parsed copy of the block, and a bare `if raw_yaml` cannot tell
    "restored" from "found nothing to restore" -- so the fallback the corrupt-backup branch below
    has always pointed at was unreachable.
    """
    # Parse the raw backup and extract only our server_key's block
    try:
        parsed = yaml.safe_load(raw_yaml) or {}
    except Exception:
        return False  # Corrupt backup — caller falls back to _restore_to_mcp_config

    if not isinstance(parsed, dict):
        return False
    # Read the backup the way every other lookup in this module reads a config: a block the agent
    # wrote at column 0 is wiring the merger serves and trash_tool captured, so it is wiring a
    # restore has to put back. It goes back nested -- canonical, and where _restore_to_mcp_config
    # and the merger both look first.
    backup_path = _server_block_path(parsed, server_key)
    if backup_path is None:
        return False  # Backup doesn't contain our entry — nothing to restore

    config_block = parsed["mcp_servers"][server_key] if len(backup_path) == 2 else parsed[server_key]

    # Guard against re-adding an entry that already exists in current config
    current = _read_mcp_config_user()
    current_servers = current.get("mcp_servers") or {}
    if isinstance(current_servers, dict) and server_key in current_servers:
        raise TrashStateError(
            f"MCP config already has entry '{server_key}'. Cannot restore - another tool may have taken this name."
        )

    # Structured add via the round-trip-safe helper — preserves every other
    # entry in the current file and writes atomically.
    _roundtrip_add_key(mcp_config_user_path(), ["mcp_servers"], server_key, config_block)
    return True


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def _validate_tool_id(tool_id: str) -> None:
    """Validate tool_id format. Raises TrashSafetyError on failure."""
    if not tool_id or not TOOL_ID_RE.match(tool_id):
        raise TrashSafetyError(
            f"Invalid tool_id '{tool_id}'. Must match: lowercase letter + "
            f"up to 62 lowercase letters/digits/underscores."
        )


def _validate_path_safe(path: Path, tool_id: str) -> None:
    """Validate a path is safe for trash operations.

    Checks:
    1. Not a symlink
    2. Not in PROTECTED_FILES
    3. Not inside PROTECTED_DIRS
    4. Resolved absolute path within TOOLS_USER_DIR or TRASH_DIR boundaries
    """
    resolved = path.resolve()

    # Reject symlinks
    if path.is_symlink():
        raise TrashSafetyError(f"Path is a symlink, refusing to operate: {path}")

    # Check against protected files
    if resolved in PROTECTED_FILES or path.resolve() in {p.resolve() for p in PROTECTED_FILES}:
        raise TrashSafetyError(f"Path is a protected file: {path}")

    # Check against protected directories
    for pdir in PROTECTED_DIRS:
        if resolved == pdir.resolve() or str(resolved).startswith(str(pdir.resolve()) + "/"):
            raise TrashSafetyError(f"Path is inside a protected directory: {path}")

    # Boundary check: must be within tools_user/ or .trash/
    tools_user_resolved = TOOLS_USER_DIR.resolve()
    trash_resolved = TRASH_DIR.resolve()
    if not (
        str(resolved).startswith(str(tools_user_resolved) + "/")
        or str(resolved).startswith(str(trash_resolved) + "/")
        or resolved == tools_user_resolved
        or resolved == trash_resolved
    ):
        raise TrashSafetyError(f"Path resolves outside allowed boundaries: {path} -> {resolved}")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _discover_tool_files(tool_id: str, search_dir: Path | None = None) -> list[Path]:
    """Find ALL files belonging to a tool.

    Combines:
    1. Files listed in install_log 'files' field
    2. Known suffix patterns ({tool_id}_worker.py, _mcp_server.py, _env.yaml, _worker.R)

    Filters out symlinks and protected files. Uses exact suffix matching
    to prevent prefix overlap (e.g., sopa vs sopa_v2).
    """
    if search_dir is None:
        search_dir = TOOLS_USER_DIR

    found: set[Path] = set()

    # From install_log files field
    entry = _find_log_entry(tool_id)
    if entry:
        for f in entry.get("files", []):
            p = search_dir / f
            if p.exists() and p.is_file() and not p.is_symlink():
                resolved = p.resolve()
                if resolved not in {pf.resolve() for pf in PROTECTED_FILES}:
                    found.add(p)

    # From known suffixes (exact match prevents prefix overlap)
    for suffix in KNOWN_SUFFIXES:
        p = search_dir / f"{tool_id}{suffix}"
        if p.exists() and p.is_file() and not p.is_symlink():
            resolved = p.resolve()
            if resolved not in {pf.resolve() for pf in PROTECTED_FILES}:
                found.add(p)

    return sorted(found)


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------


def _dir_size_bytes(path: Path) -> int:
    """Fast directory size using du -sb (OS-level, much faster than Python walk)."""
    if not path.exists():
        return 0
    try:
        result = subprocess.run(
            ["du", "-sb", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return int(result.stdout.split()[0]) if result.returncode == 0 else 0
    except Exception:
        return 0


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    if size_bytes == 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


# ---------------------------------------------------------------------------
# Conda env helpers
# ---------------------------------------------------------------------------


def _remove_conda_env(tool_id: str, base_timeout: int = 600) -> str:
    """Remove conda environment for a user tool.

    SAFETY: Always constructs env_name from tool_id (never reads from log).
    Validates against protected env names. Dynamic timeout based on env size.

    Returns: "ok", "skipped" (already gone), or raises TrashError.
    """
    env_name = f"user_{tool_id}"

    # Safety: must start with user_ and not be a protected env
    if not env_name.startswith("user_"):
        raise TrashSafetyError(f"Env name '{env_name}' does not start with user_")
    if env_name in PROTECTED_ENVS:
        raise TrashSafetyError(f"Cannot remove protected env '{env_name}'")

    env_path = CONDA_ENVS_DIR / env_name
    if not env_path.exists():
        return "skipped"

    # Dynamic timeout: 600s base + 120s per GB
    size_gb = _dir_size_bytes(env_path) / (1024**3)
    timeout = max(base_timeout, int(600 + size_gb * 120))

    result = subprocess.run(
        ["conda", "remove", "-n", env_name, "--all", "-y"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    # Verify actually removed
    if env_path.exists():
        raise TrashError(
            f"conda remove completed (rc={result.returncode}) but env dir still "
            f"exists at {env_path}. stderr: {result.stderr[:500]}"
        )
    return "ok"


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------


def _integrity_check(tool_id: str, expected_status: str) -> tuple[dict, list[str]]:
    """Validate system state before an operation.

    Returns (install_log_entry, warnings_list).
    Raises TrashNotFoundError if tool doesn't exist.
    Raises TrashStateError if status doesn't match expected.
    """
    _validate_tool_id(tool_id)

    entry = _find_log_entry(tool_id)
    if entry is None:
        raise TrashNotFoundError(
            f"Tool '{tool_id}' not found in install_log.json. "
            f"Available tools: {[e.get('tool_id') for e in _read_install_log()]}"
        )

    status = entry.get("status", "unknown")
    if status != expected_status:
        if expected_status == "active" and status == "trashed":
            raise TrashStateError(
                f"Tool '{tool_id}' is already in trash. "
                f"Use restore_tool('{tool_id}') to restore it, or "
                f"permanent_delete('{tool_id}') to remove permanently."
            )
        elif expected_status == "trashed" and status == "active":
            raise TrashStateError(
                f"Tool '{tool_id}' is active, not trashed. Use trash_tool('{tool_id}') to move it to trash first."
            )
        else:
            raise TrashStateError(f"Tool '{tool_id}' has status '{status}', expected '{expected_status}'.")

    warnings = []

    if expected_status == "active":
        # Check files exist in tools_user/
        files = _discover_tool_files(tool_id)
        if not files:
            warnings.append(f"No tool files found for '{tool_id}' in {TOOLS_USER_DIR}")
        # Check MCP config entry
        server_key = f"user_{tool_id}"
        if _get_mcp_server_entry(server_key) is None:
            warnings.append(f"MCP config entry '{server_key}' not found")

    elif expected_status == "trashed":
        # Check files exist in .trash/
        trash_tool_dir = TRASH_DIR / tool_id
        if not trash_tool_dir.exists():
            warnings.append(f"Trash directory not found: {trash_tool_dir}")
        meta_path = trash_tool_dir / ".trash_meta.json"
        if not meta_path.exists():
            warnings.append(f"Trash metadata not found: {meta_path}")

    # Conda env check (warning, not fatal)
    env_path = CONDA_ENVS_DIR / f"user_{tool_id}"
    if not env_path.exists():
        warnings.append(f"Conda env not found: {env_path}")

    return entry, warnings


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def list_user_tools(status: str | None = None) -> list[dict]:
    """List user tools, optionally filtered by status.

    Each entry is enriched with:
      - files_exist: bool (tool files present on disk)
      - conda_env_exists: bool
      - in_trash: bool (files in .trash/ directory)

    Args:
        status: Filter by "active", "trashed", or None for all.

    Returns:
        List of enriched install_log entries.
    """
    entries = _read_install_log()
    if status:
        entries = [e for e in entries if e.get("status") == status]

    result = []
    for entry in entries:
        enriched = dict(entry)
        tid = entry.get("tool_id", "")
        st = entry.get("status", "")

        # Check if files exist in expected location
        if st == "active":
            enriched["files_exist"] = len(_discover_tool_files(tid)) > 0
        elif st == "trashed":
            trash_tool_dir = TRASH_DIR / tid
            enriched["files_exist"] = trash_tool_dir.exists() and any(trash_tool_dir.iterdir())
        else:
            enriched["files_exist"] = False

        enriched["conda_env_exists"] = (CONDA_ENVS_DIR / f"user_{tid}").exists()
        enriched["in_trash"] = st == "trashed"
        result.append(enriched)

    return result


def trash_info() -> dict:
    """Get disk usage summary for active and trashed tools.

    Reclaimable means what ``permanent_delete`` destroys, which is three things per trashed tool
    and not two: the conda env, ``.trash/{tool_id}/``, and ``tools_user/vendor_{tool_id}/`` for
    vendor-mode installs (its steps 1-3). Vendor checkouts run to hundreds of MB, so leaving them
    out understated the figure this reports -- and ``preview_permanent_delete`` has always counted
    all three, which made the two views of the same question disagree.

    Returns dict with:
      - active_count, trashed_count
      - active_conda_size, trashed_conda_size (human-readable)
      - trash_files_size, trashed_vendor_size (human-readable)
      - total_reclaimable (human-readable) -- the sum of the three trashed sizes above
      - per_tool: list of {tool_id, status, conda_size}, plus files_size and vendor_size on the
        trashed entries (an active tool has neither: nothing is going to delete them)
    """
    entries = _read_install_log()
    active_count = 0
    trashed_count = 0
    active_conda_bytes = 0
    trashed_conda_bytes = 0
    trash_files_bytes = 0
    trashed_vendor_bytes = 0
    per_tool = []

    for entry in entries:
        tid = entry.get("tool_id", "")
        st = entry.get("status", "")
        conda_bytes = _dir_size_bytes(CONDA_ENVS_DIR / f"user_{tid}")

        if st == "active":
            active_count += 1
            active_conda_bytes += conda_bytes
            per_tool.append(
                {
                    "tool_id": tid,
                    "status": "active",
                    "conda_size": _human_size(conda_bytes),
                }
            )
        elif st == "trashed":
            trashed_count += 1
            trashed_conda_bytes += conda_bytes
            files_bytes = _dir_size_bytes(TRASH_DIR / tid)
            trash_files_bytes += files_bytes
            vendor_bytes = _dir_size_bytes(TOOLS_USER_DIR / f"vendor_{tid}")
            trashed_vendor_bytes += vendor_bytes
            per_tool.append(
                {
                    "tool_id": tid,
                    "status": "trashed",
                    "conda_size": _human_size(conda_bytes),
                    "files_size": _human_size(files_bytes),
                    "vendor_size": _human_size(vendor_bytes),
                }
            )

    total_reclaimable = trashed_conda_bytes + trash_files_bytes + trashed_vendor_bytes

    return {
        "active_count": active_count,
        "trashed_count": trashed_count,
        "active_conda_size": _human_size(active_conda_bytes),
        "trashed_conda_size": _human_size(trashed_conda_bytes),
        "trash_files_size": _human_size(trash_files_bytes),
        "trashed_vendor_size": _human_size(trashed_vendor_bytes),
        "total_reclaimable": _human_size(total_reclaimable),
        "per_tool": per_tool,
    }


def check_tool_id_available(tool_id: str) -> dict:
    """Check if a tool_id is available for new tool creation.

    Returns {"available": True} or {"available": False, "reason": "..."}.
    Should be called by creation pipeline before Phase 2.

    Asks the id-only question ON PURPOSE, and must keep doing so even though the registry is
    keyed on ``(owner, tool_id)``. The env-backed tier names real things after the id alone --
    ``tools_user/{tool_id}_worker.py``, the ``user_{tool_id}`` conda env, the
    ``mcp_servers.user_{tool_id}`` wiring -- so a second account taking a name the first is using
    is a genuine collision on disk, whoever owns the row. Narrowing this to the caller's own
    owner would say "available" and let the second creation overwrite the first's files.
    """
    _validate_tool_id(tool_id)
    entry = _find_log_entry(tool_id)
    if entry is None:
        return {"available": True}
    st = entry.get("status", "")
    if st == "active":
        return {
            "available": False,
            "reason": f"Tool '{tool_id}' already exists and is active.",
        }
    if st == "trashed":
        return {
            "available": False,
            "reason": (f"Tool '{tool_id}' is in trash. Permanently delete it first, or choose a different tool_id."),
        }
    return {"available": True}


def trash_tool(tool_id: str) -> dict:
    """Move a user tool to trash (reversible soft delete).

    Physical operation:
    1. Discover all tool files
    2. Save MCP config backup + full metadata to .trash_meta.json
    3. Move files to .trash/{tool_id}/
    4. Update install_log.json status to "trashed"
    5. Remove entry from mcp_config_user.yaml

    Conda env is LEFT IN PLACE (needed for restore).

    Args:
        tool_id: The tool to trash (must be "active").

    Returns:
        {"success": True, "tool_id": str, "trashed_at": str, "files_moved": int}

    Raises:
        TrashSafetyError: Invalid tool_id or path violation
        TrashNotFoundError: Tool not in install_log
        TrashStateError: Tool already trashed, or .trash/{tool_id}/ still holds files
    """
    with _trash_lock():
        # 1. Validate
        entry, warnings = _integrity_check(tool_id, "active")

        # 2. Discover all tool files
        files = _discover_tool_files(tool_id)
        for f in files:
            _validate_path_safe(f, tool_id)

        # 3. Read MCP config block for backup + raw YAML for lossless restore
        server_key = f"user_{tool_id}"
        mcp_backup = _get_mcp_server_entry(server_key)
        config_path = mcp_config_user_path()
        raw_mcp_yaml = config_path.read_text() if config_path.exists() else ""

        # 4. Build trash metadata
        now = datetime.now().isoformat()
        trash_tool_dir = TRASH_DIR / tool_id
        original_files = {}
        for f in files:
            original_files[f.name] = str(f.resolve())

        meta = {
            "tool_id": tool_id,
            "trash_version": TRASH_META_VERSION,
            "trashed_at": now,
            "original_files": original_files,
            "mcp_config_backup": {server_key: mcp_backup} if mcp_backup else {},
            "raw_mcp_yaml": raw_mcp_yaml,
            "install_log_entry": copy.deepcopy(entry),
            "conda_env": f"user_{tool_id}",
            "conda_env_path": str(CONDA_ENVS_DIR / f"user_{tool_id}"),
        }

        # 4b. Refuse if .trash/{tool_id}/ still holds files from an earlier trash. Everything
        #     below assumes the directory is ours: the move overwrites a name already there, the
        #     fresh .trash_meta.json orphans anything it does not list, and restore_tool's
        #     cleanup rmtree's the orphan. _repair_trash_state leaves files here deliberately
        #     (its kept_in_trash) for the user to compare -- and restore_tool refuses the
        #     mirror-image collision rather than pick a winner. Checked before the first write.
        if trash_tool_dir.exists():
            already_there = sorted(
                f.name for f in trash_tool_dir.iterdir() if f.is_file() and f.name != ".trash_meta.json"
            )
            if already_there:
                raise TrashStateError(
                    f"Cannot trash '{tool_id}': {trash_tool_dir} already holds {already_there} from an "
                    f"earlier trash. Trashing now would overwrite them. Move or delete those files first."
                )

        # 5. Create .trash/{tool_id}/ directory
        trash_tool_dir.mkdir(parents=True, exist_ok=True)

        # 6. Write .trash_meta.json FIRST (crash recovery data)
        meta_path = trash_tool_dir / ".trash_meta.json"
        _atomic_write_json(meta_path, meta)

        # 7. Move each file to .trash/{tool_id}/ (with rollback on failure)
        moved_files = []
        try:
            for f in files:
                dest = trash_tool_dir / f.name
                shutil.move(str(f), str(dest))
                moved_files.append((dest, f))
        except Exception:
            # Rollback already-moved files
            for moved_dest, orig_path in reversed(moved_files):
                try:
                    shutil.move(str(moved_dest), str(orig_path))
                except Exception as rb_err:
                    logger.error("Rollback failed moving %s back to %s: %s", moved_dest, orig_path, rb_err)
            raise

        # 8. Remove from mcp_config_user.yaml FIRST (safer crash-recovery order:
        #    if we crash after MCP removal but before install_log update, the tool
        #    is still "active" in install_log but has no MCP entry — health check
        #    will flag it as DEGRADED rather than blocking future restore)
        _remove_from_mcp_config(server_key)

        # 9. Update install_log: status -> trashed. Addressed at the owner of the row step 1
        #    resolved, so the row that changes is the row this call validated.
        _update_log_entry(
            tool_id,
            {
                "status": "trashed",
                "trashed_at": now,
            },
            owner=_log_key(entry)[0],
        )

        return {
            "success": True,
            "tool_id": tool_id,
            "trashed_at": now,
            "files_moved": len(moved_files),
            "warnings": warnings,
        }


def restore_tool(tool_id: str) -> dict:
    """Restore a tool from trash back to active.

    Physical operation:
    1. Read .trash_meta.json
    2. Verify conda env still exists
    3. Check no file collision at original locations
    4. Check no MCP config key conflict
    5. Move files back to original locations
    6. Restore MCP config entry
    7. Update install_log status to "active"
    8. Clean up .trash/{tool_id}/ directory

    Args:
        tool_id: The tool to restore (must be "trashed").

    Returns:
        {"success": True, "tool_id": str, "restored_at": str, "files_restored": int}

    Raises:
        TrashSafetyError: Path violation
        TrashNotFoundError: Tool not in install_log
        TrashStateError: Tool not trashed, or restore blocked by conflict
    """
    with _trash_lock():
        # 1. Validate
        entry, warnings = _integrity_check(tool_id, "trashed")

        # 2. Read .trash_meta.json
        trash_tool_dir = TRASH_DIR / tool_id
        meta_path = trash_tool_dir / ".trash_meta.json"
        meta = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                meta = None

        if meta is None:
            raise TrashStateError(
                f"Trash metadata for '{tool_id}' is missing or corrupted. "
                f"Use _repair_trash_state('{tool_id}') to attempt recovery."
            )

        # 3. Verify conda env exists
        env_path = CONDA_ENVS_DIR / f"user_{tool_id}"
        if not env_path.exists():
            raise TrashStateError(
                f"Cannot restore '{tool_id}': conda env not found at {env_path}. "
                f"The environment may have been manually removed. "
                f"Consider permanently deleting this tool and recreating it."
            )

        # 4. Check file collision at original locations
        original_files = meta.get("original_files", {})
        for _filename, orig_path_str in original_files.items():
            target = Path(orig_path_str)
            if target.exists():
                raise TrashStateError(
                    f"Cannot restore '{tool_id}': file already exists at {target}. Remove it manually before restoring."
                )

        # 5. Check MCP config conflict
        server_key = f"user_{tool_id}"
        existing_entry = _get_mcp_server_entry(server_key)
        if existing_entry is not None:
            raise TrashStateError(
                f"Cannot restore '{tool_id}': MCP config already has entry "
                f"'{server_key}'. Remove it manually before restoring."
            )

        # 6. Move files back to original locations (with rollback on failure)
        moved_files = []
        try:
            for filename, orig_path_str in original_files.items():
                source = trash_tool_dir / filename
                if source.exists():
                    target = Path(orig_path_str)
                    _validate_path_safe(target, tool_id)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(target))
                    moved_files.append((target, source))
        except Exception:
            # Rollback already-moved files back to trash
            for moved_target, orig_source in reversed(moved_files):
                try:
                    shutil.move(str(moved_target), str(orig_source))
                except Exception as rb_err:
                    logger.error("Rollback failed moving %s back to %s: %s", moved_target, orig_source, rb_err)
            raise
        restored = len(moved_files)

        # 7. Restore MCP config (lossless: use raw YAML backup if available)
        raw_mcp_yaml = meta.get("raw_mcp_yaml", "")
        # Lossless restore first: write back the exact original block. Branch on whether it put
        # the entry back, not on whether a raw snapshot exists -- the snapshot is of the whole
        # file and is captured whenever the config existed, i.e. always, so keying the fallback
        # off its mere presence made the parsed copy below dead code.
        restored_wiring = _restore_raw_mcp_yaml(raw_mcp_yaml, server_key) if raw_mcp_yaml else False
        if not restored_wiring:
            # Fallback: reconstruct from parsed backup (may alter formatting)
            mcp_backup = meta.get("mcp_config_backup", {})
            if mcp_backup and server_key in mcp_backup:
                config_block = mcp_backup[server_key]
                if config_block:
                    _restore_to_mcp_config(server_key, config_block)

        # 8. Update install_log: status -> active, remove trash fields (single atomic write).
        #    Keyed on (owner, tool_id) -- the key the file is deduped by -- so this reactivates the
        #    row step 1 validated and not another account's tool of the same name.
        now = datetime.now().isoformat()
        key = _log_key(entry)
        entries = _read_install_log()
        for e in entries:
            if _log_key(e) == key:
                e["status"] = "active"
                e.pop("trashed_at", None)
                e.pop("restored_at", None)
                break
        _write_install_log(entries)

        # 9. Clean up trash directory
        if trash_tool_dir.exists():
            # Remove remaining files (like .trash_meta.json)
            shutil.rmtree(str(trash_tool_dir), ignore_errors=True)

        # 10. Re-validate spatialomicsgym_name (server file may have changed since backup)
        try:
            from tools_user.knowledge_manager import _fix_spatialomicsgym_name, _validate_spatialomicsgym_name

            bn_check = _validate_spatialomicsgym_name(tool_id)
            if not bn_check["ok"] and bn_check["actual_name"]:
                logger.warning(
                    "restore_tool: spatialomicsgym_name mismatch after restore for %s. Auto-fixing YAML to '%s'.",
                    tool_id,
                    bn_check["actual_name"],
                )
                _fix_spatialomicsgym_name(tool_id, bn_check["actual_name"])
        except Exception as e:
            logger.warning("restore_tool: spatialomicsgym_name validation skipped for %s: %s", tool_id, e)

        return {
            "success": True,
            "tool_id": tool_id,
            "restored_at": now,
            "files_restored": restored,
            "warnings": warnings,
        }


def preview_permanent_delete(tool_id: str) -> dict:
    """Preview what will be permanently removed. No side effects.

    Args:
        tool_id: The tool to preview (must be "trashed").

    Returns:
        Dict with success, tool_id, trash_files, conda_env, conda_env_size,
        total_reclaimable, artifacts.

    Raises:
        TrashSafetyError: Invalid tool_id
        TrashNotFoundError: Tool not found
        TrashStateError: Tool not trashed
    """
    _validate_tool_id(tool_id)
    entry = _find_log_entry(tool_id)
    if entry is None:
        trashed = [e.get("tool_id") for e in _read_install_log() if e.get("status") == "trashed"]
        raise TrashNotFoundError(f"Tool '{tool_id}' not found. Trashed tools: {trashed or 'none'}")

    st = entry.get("status", "")
    if st == "active":
        raise TrashStateError(f"Tool '{tool_id}' is active. Use trash_tool('{tool_id}') to move to trash first.")
    if st != "trashed":
        raise TrashStateError(f"Tool '{tool_id}' has unexpected status: '{st}'")

    # Calculate sizes
    trash_tool_dir = TRASH_DIR / tool_id
    trash_files = {}
    if trash_tool_dir.exists():
        for f in sorted(trash_tool_dir.iterdir()):
            if f.is_file() and f.name != ".trash_meta.json":
                trash_files[f.name] = _human_size(f.stat().st_size)

    env_name = f"user_{tool_id}"
    env_path = CONDA_ENVS_DIR / env_name
    vendor_dir = TOOLS_USER_DIR / f"vendor_{tool_id}"
    conda_size_bytes = _dir_size_bytes(env_path)
    trash_size_bytes = _dir_size_bytes(trash_tool_dir)
    vendor_size_bytes = _dir_size_bytes(vendor_dir) if vendor_dir.exists() else 0
    total = conda_size_bytes + trash_size_bytes + vendor_size_bytes

    artifacts = []
    if trash_tool_dir.exists():
        artifacts.append(str(trash_tool_dir))
    if env_path.exists():
        artifacts.append(str(env_path))
    if vendor_dir.exists():
        artifacts.append(str(vendor_dir))

    return {
        "success": True,
        "tool_id": tool_id,
        "trash_files": trash_files,
        "conda_env": env_name,
        "conda_env_exists": env_path.exists(),
        "conda_env_size": _human_size(conda_size_bytes),
        "trash_dir_size": _human_size(trash_size_bytes),
        "vendor_dir_exists": vendor_dir.exists(),
        "vendor_dir_size": _human_size(vendor_size_bytes),
        "total_reclaimable": _human_size(total),
        "artifacts": artifacts,
    }


def _forget_source_memory(source_url: str | None, *, keep_memory: bool) -> str:
    """Drop the per-source memory record for a tool being permanently deleted.

    Returns "kept" / "skipped" / "ok" / "failed" and never raises.

    "skipped" is every case where there was nothing to do and none of them is a fault: no
    ``source_url`` on the install_log entry (memory is keyed by URL, so it cannot be looked up and
    must not be guessed at), memory disabled, the subsystem not importable, or no record on disk.
    "failed" is reserved for a record that was found and is still there afterwards -- the
    distinction ``_knowledge_sweep`` insists on, for the same reason: this dict is the only account
    of an irreversible operation.

    Only short-term memory. Long-term is aggregate -- ``known_hangs`` spans every URL -- so one
    tool's deletion has no business touching it, and the playbook asks only for ``delete_short_term``.
    """
    if keep_memory:
        return "kept"
    if not source_url:
        return "skipped"
    try:
        # Sibling-module private: the enabled/disabled decision has one implementation, and
        # re-spelling its env-var handling here is precisely how the two drift apart.
        from tools_user.memory_manager import MemoryManager, _memory_disabled
    except Exception:
        # Memory is optional (it wants `filelock`, and `tools_user` is a namespace package that
        # resolves only once the repo root is on sys.path). Nothing was attempted.
        return "skipped"
    try:
        if _memory_disabled():
            return "skipped"
        mm = MemoryManager.get()
        path = mm.short_term_path(source_url)
        if not path.exists():
            return "skipped"
        mm.delete_short_term(source_url)
        # Confirm rather than trust the return value: `delete_short_term` reports False for
        # "absent", "disabled" and "the unlink failed" alike.
        return "ok" if not path.exists() else "failed"
    except Exception as e:
        logger.warning("[trash] could not clear memory for %s: %s", source_url, e)
        return "failed"


def permanent_delete(tool_id: str, *, conda_timeout: int = 600, keep_memory: bool = False) -> dict:
    """Permanently delete a trashed tool. IRREVERSIBLE.

    Can ONLY be called on tools with status "trashed" (two-stage guarantee).

    Destruction order:
    1. Remove conda env (dynamic timeout based on size)
    2. Remove .trash/{tool_id}/ directory
    3. Remove tools_user/vendor_{tool_id}/ (vendor-mode installs)
    4. Remove install_log entry (only if primary artifacts gone)
    5. Defensive sweep of mcp_config_user.yaml
    6. Clean up tools_user/.knowledge/{tool_id}/
    7. Drop the source URL's short-term memory record, unless keep_memory

    Per-step error handling: continues on partial failure, reports all results.

    Step 7 is the one step that stays out of `errors`, so it can never move `success` or
    `disk_reclaimed`: memory is metadata about the source URL rather than an artifact of the tool,
    and a busy lock on it must not report a deletion that reclaimed everything as partial. Its
    outcome is carried in `steps["memory"]` instead.

    Args:
        tool_id: The tool to permanently delete (must be "trashed").
        conda_timeout: Base timeout in seconds for conda removal.
        keep_memory: Leave the recorded attempts for this tool's source URL in place, so a later
            re-creation of the same URL can still reuse the recipe.

    Returns:
        {"success": bool, "tool_id": str, "steps": dict, "disk_reclaimed": str, "errors": list}
    """
    with _trash_lock():
        # Validate: MUST be trashed
        _validate_tool_id(tool_id)
        entry = _find_log_entry(tool_id)
        if entry is None:
            raise TrashNotFoundError(f"Tool '{tool_id}' not found in install_log.")
        st = entry.get("status", "")
        if st == "active":
            raise TrashStateError(f"Tool '{tool_id}' is active. Must trash first using trash_tool('{tool_id}').")
        if st != "trashed":
            raise TrashStateError(f"Tool '{tool_id}' has unexpected status: '{st}'")

        # Pre-calculate size for reporting
        env_path = CONDA_ENVS_DIR / f"user_{tool_id}"
        trash_tool_dir = TRASH_DIR / tool_id
        vendor_dir = TOOLS_USER_DIR / f"vendor_{tool_id}"
        pre_size = (
            _dir_size_bytes(env_path)
            + _dir_size_bytes(trash_tool_dir)
            + (_dir_size_bytes(vendor_dir) if vendor_dir.exists() else 0)
        )

        # Read out before step C deletes the entry: the install_log is the only place the source
        # URL lives, and memory is keyed by it.
        source_url = (entry.get("source_url") or "").strip()

        steps = {}
        errors = []

        # Step A: Remove conda env
        try:
            result = _remove_conda_env(tool_id, base_timeout=conda_timeout)
            steps["conda_env"] = result  # "ok" or "skipped"
        except Exception as e:
            steps["conda_env"] = "failed"
            errors.append(f"conda: {e}")
            # Mark in log so restore knows env is gone
            try:
                _update_log_entry(tool_id, {"conda_removed": True}, owner=_log_key(entry)[0])
            except Exception:
                pass

        # Step B: Remove trash directory
        try:
            if trash_tool_dir.exists():
                shutil.rmtree(str(trash_tool_dir))
                steps["trash_dir"] = "ok"
            else:
                steps["trash_dir"] = "skipped"
        except Exception as e:
            steps["trash_dir"] = "failed"
            errors.append(f"trash_dir: {e}")

        # Step F: Remove tools_user/vendor_{tool_id}/ (vendor-mode installs)
        try:
            if vendor_dir.exists():
                _validate_path_safe(vendor_dir, tool_id)
                shutil.rmtree(str(vendor_dir))
                steps["vendor_dir"] = "ok"
            else:
                steps["vendor_dir"] = "skipped"
        except Exception as e:
            steps["vendor_dir"] = "failed"
            errors.append(f"vendor_dir: {e}")

        # Step C: Remove install_log entry (only if ALL primary artifacts are gone)
        if (
            steps.get("conda_env") != "failed"
            and steps["trash_dir"] in ("ok", "skipped")
            and steps["vendor_dir"] in ("ok", "skipped")
        ):
            try:
                # Owner-addressed: this is irreversible, and an id-only filter would take every
                # account's row of that name with it.
                remove_error = _remove_log_entry(tool_id, owner=_log_key(entry)[0])
                if remove_error:
                    steps["install_log"] = "failed"
                    errors.append(f"install_log: {remove_error}")
                else:
                    steps["install_log"] = "ok"
            except Exception as e:
                steps["install_log"] = "failed"
                errors.append(f"install_log: {e}")
        else:
            steps["install_log"] = "skipped"

        # Step D: Defensive sweep of MCP config
        try:
            server_key = f"user_{tool_id}"
            if _get_mcp_server_entry(server_key) is not None:
                _remove_from_mcp_config(server_key)
                steps["mcp_config"] = "cleaned"
            else:
                steps["mcp_config"] = "clean"
        except Exception as e:
            steps["mcp_config"] = "failed"
            errors.append(f"mcp_config: {e}")

        # Step E: Clean up knowledge directory (backups, API discovery, etc.)
        # Defensive: also catch fuzzy-matched variants (drifted tids like
        # `{tid}_v\d+`, case variants, dirs whose creation_log.json points at
        # this tool_id). Run as a candidate sweep; each candidate validated
        # by the inline guards in _knowledge_sweep before rmtree -- NOT by
        # _validate_path_safe, which would reject every one of them because
        # `.knowledge/` is in PROTECTED_DIRS. See that function's docstring.
        knowledge_root = TOOLS_USER_DIR / ".knowledge"

        def _knowledge_candidates(canonical: str) -> list[Path]:
            cands: list[Path] = []
            if not knowledge_root.exists():
                return cands
            canon_lc = canonical.lower()
            try:
                entries = list(knowledge_root.iterdir())
            except Exception:
                return cands
            for sub in entries:
                if not sub.is_dir():
                    continue
                name = sub.name
                if name == canonical:
                    cands.append(sub)
                    continue
                # case-insensitive exact, or {canonical}_v\d+ suffix
                if name.lower() == canon_lc:
                    cands.append(sub)
                    continue
                if re.match(rf"^{re.escape(canonical)}_v\d+$", name):
                    cands.append(sub)
                    continue
                # dir whose creation_log.json / install_plan.json claims this tid
                for meta_name in ("creation_log.json", "install_plan.json"):
                    meta_path = sub / meta_name
                    if not meta_path.exists():
                        continue
                    try:
                        meta = json.loads(meta_path.read_text())
                        if isinstance(meta, dict) and meta.get("tool_id") == canonical:
                            cands.append(sub)
                            break
                    except Exception:
                        continue
            return cands

        def _knowledge_sweep(label: str) -> str:
            """Return 'ok' / 'skipped' / 'partial' / 'failed'.

            Local safety check: candidate parent MUST be exactly `.knowledge/`
            (so we can never sweep anything outside the knowledge area), AND
            the candidate name must be a non-empty plain segment (no `..`,
            no separators). This lets us delete `.knowledge/{tid}/` even
            though `.knowledge/` itself is in PROTECTED_DIRS — the protection
            is for the parent dir, not its tool subdirs.

            A candidate the guards turn down is *refused*, not failed: declining
            to follow a symlink is the safety check working as intended, and it
            must not brand an otherwise-complete delete `success: False`. A
            candidate whose `rmtree` raised IS a failure — the directory is
            still on disk — so it appends to `errors` the way the other five
            steps do, and the return value says how far the sweep got:
            'failed' when nothing went, 'partial' when some of it did.
            `permanent_delete` is irreversible and this dict is the only
            account of what it did, so "found it and could not delete it" must
            not be spelled the same way as "there was nothing to delete".
            """
            try:
                cands = _knowledge_candidates(tool_id)
                if not cands:
                    return "skipped"
                removed = 0
                failed = 0
                for c in cands:
                    try:
                        # Tight scope: parent is exactly knowledge_root, name
                        # is a single safe segment.
                        if c.parent.resolve() != knowledge_root.resolve():
                            continue
                        if not c.name or c.name in (".", "..") or "/" in c.name:
                            continue
                        if c.is_symlink():
                            continue
                        shutil.rmtree(str(c))
                        removed += 1
                    except Exception as e:
                        failed += 1
                        errors.append(f"knowledge_cleanup ({label}): {c.name}: {e}")
                if failed:
                    return "partial" if removed else "failed"
                return "ok" if removed else "skipped"
            except Exception as e:
                errors.append(f"knowledge_cleanup ({label}): {e}")
                return "failed"

        # First pass — at the normal position.
        steps["knowledge_cleanup"] = _knowledge_sweep("first_pass")

        # Second pass — at the very end, AFTER step D's mcp_config write
        # may have had time to race with a late knowledge writer. Tightens
        # the race window; idempotent on already-clean state.
        late = _knowledge_sweep("late_pass")
        if late == "ok" and steps["knowledge_cleanup"] in ("skipped", "ok"):
            steps["knowledge_cleanup_late"] = "swept_late"
        elif late not in ("skipped",):
            steps["knowledge_cleanup_late"] = late

        # Step G: Drop the source URL's memory record. Runs last and stays out of `errors` --
        # see the docstring and _forget_source_memory for why a busy lock here must not turn a
        # complete deletion into a partial one.
        steps["memory"] = _forget_source_memory(source_url, keep_memory=keep_memory)

        success = len(errors) == 0
        return {
            "success": success,
            "tool_id": tool_id,
            "steps": steps,
            "disk_reclaimed": _human_size(pre_size) if success else "partial",
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# Error recovery
# ---------------------------------------------------------------------------


def _repair_trash_state(tool_id: str) -> dict:
    """Attempt to repair inconsistent state for a tool.

    Examines actual disk state vs install_log metadata.

    Scenarios:
    1. Files in tools_user/, status=trashed -> mark active, restore MCP config
    2. Files in .trash/, status=active -> mark trashed, remove MCP config
    3. Split state + valid meta -> complete the interrupted operation
    4. Split state + corrupt meta -> move all to tools_user/ (safer), mark active
    5. No files anywhere -> clean up orphaned log entry

    In 3/4 a trashed file whose name is already taken in tools_user/ is left in .trash/ rather
    than overwriting or discarding either copy -- it is the only copy of the version the user
    trashed, and restore_tool refuses the same collision. Those names come back in
    ``kept_in_trash`` (split state only) and the trash directory survives to hold them.

    Returns:
        {"repaired": bool, "action": str, "details": str} -- plus "kept_in_trash": list[str]
        for the split-state scenarios.
    """
    _validate_tool_id(tool_id)
    entry = _find_log_entry(tool_id)

    files_in_user = _discover_tool_files(tool_id, search_dir=TOOLS_USER_DIR)
    trash_tool_dir = TRASH_DIR / tool_id
    files_in_trash = []
    if trash_tool_dir.exists():
        files_in_trash = [f for f in trash_tool_dir.iterdir() if f.is_file() and f.name != ".trash_meta.json"]

    has_user_files = len(files_in_user) > 0
    has_trash_files = len(files_in_trash) > 0

    if entry is None:
        if has_trash_files:
            # Orphaned trash files with no log entry
            return {
                "repaired": False,
                "action": "manual_needed",
                "details": (
                    f"Found files in {trash_tool_dir} but no install_log entry. "
                    f"Manually inspect and remove if not needed."
                ),
            }
        return {
            "repaired": False,
            "action": "not_found",
            "details": f"No entry and no files found for '{tool_id}'.",
        }

    status = entry.get("status", "")
    # Every write below addresses the row this repair actually read, by the key the file is
    # deduped on. With two accounts holding the same name, matching on the id alone would repair
    # one account's state into the other's row.
    key = _log_key(entry)
    owner = key[0]

    # Scenario 1: Files in tools_user/, status=trashed
    if has_user_files and not has_trash_files and status == "trashed":
        _update_log_entry(tool_id, {"status": "active"}, owner=owner)
        entries = _read_install_log()
        for e in entries:
            if _log_key(e) == key:
                e.pop("trashed_at", None)
                break
        _write_install_log(entries)
        return {
            "repaired": True,
            "action": "marked_active",
            "details": f"Files found in tools_user/, marked '{tool_id}' as active.",
        }

    # Scenario 2: Files in .trash/, status=active
    if has_trash_files and not has_user_files and status == "active":
        now = datetime.now().isoformat()
        _update_log_entry(tool_id, {"status": "trashed", "trashed_at": now}, owner=owner)
        server_key = f"user_{tool_id}"
        _remove_from_mcp_config(server_key)
        return {
            "repaired": True,
            "action": "marked_trashed",
            "details": f"Files found in .trash/, marked '{tool_id}' as trashed.",
        }

    # Scenario 3/4: Split state
    if has_user_files and has_trash_files:
        # Try to read meta for direction
        meta_path = trash_tool_dir / ".trash_meta.json"
        meta = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = None

        # Move to tools_user/ (safer direction). A name already taken there is not overwritten --
        # and the trashed copy is not deleted either. It is the only copy of the version the user
        # trashed, and restore_tool refuses this exact collision ("file already exists at ...")
        # rather than pick a winner. Keep it where it is, and keep the directory holding it: the
        # rest of the split state is still resolved and the log entry still goes active.
        kept_in_trash = []
        moved = []
        for f in files_in_trash:
            dest = TOOLS_USER_DIR / f.name
            if dest.exists():
                kept_in_trash.append(f.name)
            else:
                shutil.move(str(f), str(dest))
                moved.append(f.name)
        # Clean up trash dir (only once nothing of the user's is left in it)
        if not kept_in_trash and trash_tool_dir.exists():
            shutil.rmtree(str(trash_tool_dir), ignore_errors=True)
        _update_log_entry(tool_id, {"status": "active"}, owner=owner)
        entries = _read_install_log()
        for e in entries:
            if _log_key(e) == key:
                e.pop("trashed_at", None)
                break
        _write_install_log(entries)

        # Try to restore MCP config from meta
        if meta:
            mcp_backup = meta.get("mcp_config_backup", {})
            server_key = f"user_{tool_id}"
            if server_key in mcp_backup and mcp_backup[server_key]:
                try:
                    _restore_to_mcp_config(server_key, mcp_backup[server_key])
                except TrashStateError:
                    pass  # Key already exists, that's fine

        if kept_in_trash:
            details = (
                f"Split state detected for '{tool_id}'. Moved {len(moved)} file(s) to tools_user/, marked active. "
                f"Left in {trash_tool_dir}: {kept_in_trash} -- a file of that name already exists in tools_user/. "
                f"Compare the two and delete the copy you do not want."
            )
        else:
            details = f"Split state detected for '{tool_id}'. Moved all files to tools_user/, marked active."

        return {
            "repaired": True,
            "action": "consolidated_to_active",
            "details": details,
            "kept_in_trash": kept_in_trash,
        }

    # Scenario 5: No files anywhere
    if not has_user_files and not has_trash_files:
        return {
            "repaired": False,
            "action": "orphaned_entry",
            "details": (
                f"No files found for '{tool_id}' in either location. "
                f"Entry exists in install_log with status='{status}'. "
                f"This is an orphaned entry. To clean up, permanently delete it."
            ),
        }

    return {
        "repaired": False,
        "action": "unknown_state",
        "details": f"Unexpected state for '{tool_id}': status={status}, user_files={has_user_files}, trash_files={has_trash_files}",
    }
